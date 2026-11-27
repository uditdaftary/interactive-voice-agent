"""Active Real-Time Narration & Interruption Engine for Interactive Voice Agent.

Drives synchronized audio synthesis, teleprompter word cursor tracking,
and real-time LLM query interruption with live internet web search grounding.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from voice_agent.document.bm25 import BM25Index
from voice_agent.cursor.tracker import CursorPosition, CursorTracker
from voice_agent.document.parser import DocumentContent
from voice_agent.llm.client import (
    CHARACTER_PERSONA_PROMPTS,
    OFFLINE_LINE,
    LLMClient,
    search_web,
)
from voice_agent.llm.intents import classify_local, greeting_reply
from voice_agent.llm.prompt import HISTORY_MESSAGES, build_messages
from voice_agent.server.connection_manager import ConnectionManager, Statuses
from voice_agent.server.schemas import (
    CursorUpdateMessage,
    ErrorMessage,
    PromptEnvelopeMessage,
    ReplyErrorMessage,
)
from voice_agent.settings import Settings, load_settings
from voice_agent.tts.backends import Synthesizer, build_tts

logger = logging.getLogger(__name__)

# Called as search(query, max_results); returns the grounding text, "" when nothing was found.
SearchFn = Callable[[str, int], Awaitable[str]]


# Spoken when a reply fails or the user's speech was not understood. Neutral on purpose: no persona, no detail.
REPLY_ERROR_LINE = "Something went wrong answering that."
UNRECOGNIZED_LINE = "I didn't catch that."
# Read before the closest passage of the book when no model answered.
FROM_THE_TEXT = "From the text: "

# The book is the primary source, and the web is not searched, when its best passage scores at least this much.
AUTHORITATIVE_SCORE = 2.5
DEFAULT_PERSONA_PROMPT = (
    "You are an interactive conversational voice audiobook narrator. "
    "The user interrupted the book narration to ask you a question. "
    "Answer concisely, helpfully, and naturally in 1 to 3 spoken sentences. "
    "Use the provided book context and web search results when answering."
)


# After an interruption, a pause or a change of owner, a sentence that was partly spoken starts again this many words
# before the stop point. The HUD is told the same number when narration comes back from a question.
RESUME_PRE_ROLL_WORDS = 4


class EmptyReply(Exception):
    """The language model finished without producing any text."""


@dataclass(frozen=True)
class _Synthesis:
    """One sentence, made: what to send, and what the server expects of its playback."""

    sentence_idx: int
    first_word: int  # the sentence's word that the audio starts at: 0, or a few before the stop point on a restart
    word_count: int  # words in the audio
    duration_s: float  # the estimate: steps the word cursor, and is the length a missing report is waited for beyond
    gap_s: float  # the pause the browser adds before this clip when it follows another
    audio: bytes | None  # None: every synthesizer failed, so there is nothing to play or to wait for


@dataclass(frozen=True)
class _Clip:
    """A sentence handed to the owner (or, without a segment, paced by the estimate alone)."""

    synthesis: _Synthesis
    segment_id: int | None
    generation: int  # the audio generation it was sent under
    played: asyncio.Event = field(default_factory=asyncio.Event, compare=False)  # set once the loop has played it

    @property
    def sentence_idx(self) -> int:
        return self.synthesis.sentence_idx

    @property
    def first_word(self) -> int:
        return self.synthesis.first_word

    @property
    def word_count(self) -> int:
        return self.synthesis.word_count

    @property
    def duration_s(self) -> float:
        return self.synthesis.duration_s

    @property
    def gap_s(self) -> float:
        return self.synthesis.gap_s


# Default resident narrative text (Cyberpunk 2077 / Edgerunners excerpt)
DEFAULT_STORY_SENTENCES = [
    "The neon glow of Night City reflected against the wet pavement of Santo Domingo.",
    "David adjusted the sandevistan cyberware along his spine, feeling the cold rush of neural coolant.",
    "Above the smog-choked megaliths, Arasaka orbital shuttles drifted like artificial constellations.",
    "Lucy turned her eyes toward the distant moon, her monowire humming with faint static charge.",
    "\"If you run now, David,\" she whispered, \"they will track your signature across every sub-net in Pacifica.\"",
    "He stepped forward into the alleyway, cyberware spooling up to combat frequency.",
    "\"Let them trace it,\" David replied, his voice calm beneath the low roar of the city.",
    "In the shadows between mega-buildings, danger was the only currency that never depreciated.",
]


class NarrationEngine:
    """Coordinates document narration, speech synthesis, and Q&A barge-in interruption."""

    def __init__(
        self,
        connection_manager: ConnectionManager,
        llm_client: Optional[LLMClient] = None,
        initial_sentences: Optional[List[str]] = None,
        *,
        settings: Optional[Settings] = None,
        tts: Optional[Synthesizer] = None,
        search: Optional[SearchFn] = None,
    ):
        self.manager = connection_manager
        self.settings = settings if settings is not None else load_settings()
        # The chain follows the settings this engine was given, not whatever the environment holds when it is first used.
        self.tts: Synthesizer = tts if tts is not None else build_tts(self.settings.tts_primary)
        self.search: SearchFn = search if search is not None else search_web
        if llm_client is not None:
            self.llm = llm_client
        else:
            self.llm = LLMClient(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model=self.settings.llm_model,
                web_search_enabled=self.settings.web_search_enabled,
            )
        self._llm_status = "ok"  # how the last request that ran to its end finished; ok before any request
        self._stt_status = "ok"
        self.sentences: List[str] = list(initial_sentences or DEFAULT_STORY_SENTENCES)
        self._bm25_index: BM25Index = BM25Index.from_sentences(self.sentences)
        self.document_title: str = "Night City: Santo Domingo Run"
        self.current_sentence_idx: int = 0
        self.current_word_idx: int = 0
        self._conversation_history: List[dict] = []
        self._cue_pre_roll = False  # narration is restarting after an interruption: the HUD gets a cue first

        # The engine owns turn-taking timing, so it hands the configured waits to the session it drives.
        session = self.manager.session
        session.listen_deadline_s = self.settings.barge_in_listen_s
        session.followup_window_s = self.settings.followup_window_s
        session.add_listener(self._on_state_change)
        self.manager.add_owner_listener(self._on_owner_change)
        self.manager.set_status_provider(self.statuses)

    # --- statuses reported to the HUD ------------------------------------------------------------------------------

    def statuses(self) -> Statuses:
        """The model, speech and recognizer statuses as the HUD is told them. `none` is the no-model mode, whatever the
        last request said."""
        provider = self.llm.provider
        return Statuses(
            llm_status="none" if provider == "none" else self._llm_status,
            llm_provider=provider,
            tts_status=self.tts.status,
            stt_status=self._stt_status,
            web_speech_preview=self.settings.web_speech_preview,
        )

    def set_stt_status(self, status: str) -> None:
        self._stt_status = status
        self.manager.refresh_statuses()

    def _llm_endpoint(self) -> tuple[Any, ...]:
        """What decides which model answers: a change in any of these makes the old endpoint's behaviour irrelevant."""
        config = self.llm.get_config()
        return (config.get("provider"), config.get("base_url"), config.get("model"), getattr(self.llm, "api_key", None))

    def configure_llm(self, **fields: Any) -> None:
        """Change the model configuration. When the endpoint, provider, model or key changed, how the old one behaved
        says nothing about the new one, so the status starts over; a change to anything else (web search) keeps it."""
        before = self._llm_endpoint()
        self.llm.update_config(**fields)
        if self._llm_endpoint() != before:
            self._llm_status = "ok"

    def _on_owner_change(self) -> None:
        """The audio goes to a different tab now. What the old one was playing is stopped either way."""
        state = self.manager.state
        if state == "NARRATING":
            # The new owner has not heard the sentence that is playing: it starts again for them.
            self.manager.stop_audio()
            self.manager.tasks.cancel("narration")
            # The restart is not made here. The owner may have been dropped by a send the narration loop itself was
            # making, and a task is never cancelled by its own call: from here that loop would look alive, and nothing
            # would be started. A callback runs outside every task, where the old loop is cancelled like any other.
            asyncio.get_running_loop().call_soon(self._restart_narration)
        elif state == "REPLYING":
            # The clip being waited for went to the old owner: the wait ends and the turn goes on.
            self.manager.stop_audio()

    def _on_state_change(self, old: str, new: str, reason: str) -> None:
        """Keep the background tasks in step with the session state."""
        if new == "PAUSED" and not self.manager.session.resuming:
            # Pausing stops everything that speaks or listens, wherever the pause came from. (Returning to a
            # paused session after a reply or a deadline is not a pause: nothing is left to stop, and the
            # reply may be the very task finishing.)
            for name in ("narration", "reply", "stt"):
                self.manager.tasks.cancel(name)
            self.manager.stop_audio()
        elif new == "LISTENING":
            # A barge-in, from any source, silences narration; the deadline or the reply starts it again.
            self.manager.tasks.cancel("narration")
            if reason == "barge_in":
                # The user spoke: any reply stops, whatever state the session was in when it started, so
                # nothing narrates or answers over them. The reply's own follow-up window (reason "followup")
                # and a replacement reply (which cancels the old one by name) are not barge-ins.
                self.manager.tasks.cancel("reply")
        elif new == "NARRATING":
            # Coming back from a question or a listening window is announced to the HUD, with the words narration
            # starts back by; a plain resume is not. A repeat starts at the sentence's first word, so it rewinds nothing.
            self._cue_pre_roll = old in ("LISTENING", "REPLYING") and reason != "repeat"
            self._ensure_narration()

    def _restart_narration(self) -> None:
        """Replace the narration loop. Called outside any task (see `_on_owner_change`): by then the session may have
        left NARRATING, in which case whoever moved it already stopped the loop and nothing starts."""
        self.manager.tasks.cancel("narration")
        if self.manager.state == "NARRATING":
            self._ensure_narration()

    def _ensure_narration(self) -> None:
        if self.manager.tasks.get("narration") is None:
            self.manager.tasks.spawn(self._narration_loop(), "narration")

    def _go_to(self, sentence_idx: int) -> None:
        """Put the cursor at the start of a sentence. Narration, if that is what the session is doing, goes on from
        there; in any other state the cursor only moves, and the clients are told.

        The old narration task may be in the middle of a sentence. It is cancelled, so it can never reach its own
        step to the next sentence and skip one.
        """
        self.manager.tasks.cancel("narration")
        self.current_sentence_idx = sentence_idx
        self.current_word_idx = 0
        self.manager.update_cursor(sentence_idx=sentence_idx, word_offset=0)
        state = self.manager.state
        if state == "NARRATING":
            self.manager.stop_audio()  # the clips queued for the sentence being left
            self._ensure_narration()  # the new loop announces the sentence itself
        elif state == "PAUSED":
            self.manager.broadcast_json_sync(CursorUpdateMessage(
                sentence_idx=sentence_idx,
                word_offset=0,
                sentence_text=self.sentences[sentence_idx] if self.sentences else "",
                total_sentences=len(self.sentences),
            ))
        else:
            # LISTENING or REPLYING: the HUD takes a cursor_update to mean narration is running, so it is told
            # through the state snapshot (which carries the cursor) and keeps showing what the session is doing.
            self.manager.broadcast_state_sync()

    def next_sentence(self) -> None:
        """Move to the start of the next sentence, wrapping at the end."""
        if self.sentences:
            self._go_to((self.current_sentence_idx + 1) % len(self.sentences))

    def previous_sentence(self) -> None:
        """Move to the start of the previous sentence, wrapping at the start."""
        if self.sentences:
            self._go_to((self.current_sentence_idx - 1) % len(self.sentences))

    def load_document(self, doc: DocumentContent) -> None:
        """Load a parsed document (PDF, EPUB, TXT) into the active narrator.

        A question or a recording in progress is about the old document, so that turn is abandoned and the session
        goes back to what it was doing, now at the new document's first sentence.
        """
        if not doc.sentences:
            return
        session = self.manager.session
        self.manager.tasks.cancel("narration")
        self.manager.stop_audio()  # whatever is playing, a clip or an answer, is about the old document
        self.sentences = list(doc.sentences)
        self._bm25_index = BM25Index.from_sentences(self.sentences)
        self.document_title = doc.title
        self._conversation_history.clear()
        self.current_sentence_idx = 0
        self.current_word_idx = 0
        self.manager.update_cursor(sentence_idx=0, word_offset=0)

        state_before = session.state
        if state_before in ("LISTENING", "REPLYING"):
            self.manager.tasks.cancel("reply")
            self.manager.tasks.cancel("stt")
            # After the new text is in place: if narration was what the session came from, its loop restarts now.
            session.resume("document_loaded")
        elif state_before == "NARRATING":
            self._ensure_narration()
        if session.state == state_before:
            # No transition announced the new document and cursor.
            self.manager.broadcast_state_sync()
        logger.info(f"Loaded document '{doc.title}' with {len(self.sentences)} sentences, {doc.total_words} words.")

    def start_narration(self) -> None:
        """Start or resume background narration loop. Refused (and logged) from a state that may not narrate.

        While a question is open (LISTENING or REPLYING) the session returns to narration when it ends, whatever it was
        doing before the question, so a resume from a paused session still takes effect. From LISTENING it narrates at
        once: that is how a user stops waiting. A reply that is in progress is not cut off: it ends, and narration
        follows it. (Every tab sends a resume when its socket opens, so a second tab must not be able to cut an answer
        off.)
        """
        session = self.manager.session
        state = session.state
        if state in ("LISTENING", "REPLYING"):
            session.set_resume_to("NARRATING")
        if state == "REPLYING":
            logger.debug("Resume during a reply: the reply goes on and narration follows it")
            return
        if not session.transition("NARRATING", "resume"):
            return
        self._ensure_narration()  # the state listener starts it on a change; this covers a loop that has since ended
        logger.info("Narration started/resumed.")

    def pause_narration(self) -> None:
        """Pause: stops narration, any reply being spoken and any pending transcription."""
        self.manager.session.transition("PAUSED", "pause")
        logger.info("Narration paused.")

    # --- narration ------------------------------------------------------------------------------------------------

    def _restart_point(self) -> tuple[int, int, str]:
        """Where narration starts: the sentence, its first word, and the text spoken from there.

        A sentence that was partly spoken (an interruption, a pause, a new owner) starts again a few words before the
        stop point, so the listener picks the thread up; anything else starts at the sentence's first word.
        """
        index = self.current_sentence_idx % len(self.sentences)
        sentence = self.sentences[index]
        if index != self.current_sentence_idx or self.current_word_idx == 0:
            return index, 0, sentence
        tracker = CursorTracker()
        tracker.sentences = self.sentences
        tracker.pos = CursorPosition(sentence_idx=index, word_offset=self.current_word_idx)
        text = tracker.get_resumption_text(pre_roll_words=RESUME_PRE_ROLL_WORDS)
        if not text:
            return index, 0, sentence
        return index, len(sentence.split()) - len(text.split()), text

    async def _show_word(self, sentence_idx: int, word_idx: int) -> None:
        """Move the cursor to a word and tell the clients."""
        self.current_sentence_idx = sentence_idx
        self.current_word_idx = word_idx
        self.manager.update_cursor(sentence_idx=sentence_idx, word_offset=word_idx)
        await self.manager.broadcast_json(CursorUpdateMessage(
            sentence_idx=sentence_idx,
            word_offset=word_idx,
            sentence_text=self.sentences[sentence_idx],
            total_sentences=len(self.sentences),
        ))

    async def _synthesize(self, sentence_idx: int, first_word: int, text: str) -> _Synthesis:
        """Make the audio for one sentence, with the voice and cadence in force now: a change applies from the next
        sentence that has not been synthesized yet."""
        voice_id = self.manager.active_voice or "david-martinez"
        cadence = self.manager.cadence or 1.0
        speech = await self.tts.synthesize(text, voice_id, cadence)
        self.manager.refresh_statuses()
        return _Synthesis(
            sentence_idx=sentence_idx,
            first_word=first_word,
            word_count=max(1, len(text.split())),
            duration_s=self._estimated_duration_s(text, cadence),
            gap_s=self.settings.inter_sentence_pause_s / cadence,
            audio=speech.data if speech is not None and speech.data else None,
        )

    async def _feed(
        self,
        start: tuple[int, int, str],
        room: asyncio.Semaphore,
        clips: "asyncio.Queue[_Clip | Exception]",
        generation: int,
    ) -> None:
        """Make each sentence's audio, and hand it to the owner as soon as there is room for it. The sentence after the
        one being made is not started before this one has been sent, so at most one is made ahead of what is out. A
        failure goes to the loop through the queue, which reports it.

        `generation` is the audio generation the loop started in. A clip is sent under it, so one still being made when
        audio is stopped is dropped, not played after the stop by a loop that is about to be cancelled."""
        index, first_word, text = start
        first = True
        silent: _Clip | None = None  # the sentence just before this one, if there was nothing to play for it
        try:
            while True:
                synthesis = await self._synthesize(index, first_word, text)
                if silent is not None:
                    # The silent sentence is paced by its estimate when the loop reaches it, and this one is not sent
                    # before that, or it would be heard early. It was made meanwhile, so no time is lost.
                    await silent.played.wait()
                    silent = None
                await room.acquire()
                segment_id = None
                if synthesis.audio:
                    segment_id = await self.manager.send_audio(
                        synthesis.audio, gap_s=0.0 if first else synthesis.gap_s, generation=generation
                    )
                clip = _Clip(synthesis, segment_id, generation)
                if segment_id is None:
                    room.release()  # nothing is out, so this sentence holds no slot
                    silent = clip
                clips.put_nowait(clip)
                first = False
                index = (index + 1) % len(self.sentences)
                first_word, text = 0, self.sentences[index]
        except Exception as exc:  # noqa: BLE001  # handed to the loop, which logs it with its traceback and pauses
            clips.put_nowait(exc)

    async def _play(self, clip: _Clip) -> str:
        """Step the word cursor through the clip that is playing, and wait for the browser to say it has finished.

        Returns "played", "stalled" (the report never came) or "stopped" (audio was stopped under it). The cursor goes
        word by word at the estimated rate, stops at the last word, and moves on only when the clip is done, so it is
        never ahead of what the listener hears. A clip with nothing to report is paced by the estimate alone.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + clip.duration_s + self.settings.playback_ack_timeout_s
        done = self.manager.playback.done_future(clip.segment_id) if clip.segment_id is not None else None
        step_s = clip.duration_s / clip.word_count
        for offset in range(clip.word_count):
            await self._show_word(clip.sentence_idx, clip.first_word + offset)
            if done is None:
                await asyncio.sleep(step_s)
            else:
                await asyncio.wait({done}, timeout=step_s)
                if done.done():
                    break
        if clip.segment_id is None:
            await asyncio.sleep(clip.gap_s)  # no browser to add the pause between sentences
            return "played"
        if await self.manager.playback.wait_done(clip.segment_id, max(0.0, deadline - loop.time())):
            return "played"
        return "stopped" if self.manager.generation != clip.generation else "stalled"

    async def _narration_loop(self) -> None:
        """Narrate, one sentence ahead of the listener and no more.

        Two clips are out at any time: the one playing and the one queued behind it. A feeder task makes the next
        sentence's audio while those play and sends it the moment the browser reports the playing one finished; this
        loop moves the cursor to the queued clip at that report. So a barge-in flushes at most one queued sentence,
        and the book never piles up in the browser.
        """
        generation = self.manager.generation  # before the first await: this loop belongs to the audio that follows the last stop
        feeder: asyncio.Task[None] | None = None
        try:
            if self._cue_pre_roll:
                self._cue_pre_roll = False
                # Queued behind the state_sync that started this loop, ahead of the first cursor_update.
                await self.manager.broadcast_json(PromptEnvelopeMessage(
                    state="NARRATING",
                    context=self.sentences[self.current_sentence_idx % len(self.sentences)] if self.sentences else "",
                    pre_roll_words=RESUME_PRE_ROLL_WORDS,
                ))
            start = self._restart_point()
            await self._show_word(start[0], start[1])  # the sentence is on screen while its audio is being made
            clips: asyncio.Queue[_Clip | Exception] = asyncio.Queue()
            room = asyncio.Semaphore(2)  # the clip playing and the one queued behind it
            feeder = asyncio.create_task(self._feed(start, room, clips, generation), name="narration-feed")
            while self.manager.state == "NARRATING":
                clip = await clips.get()
                if isinstance(clip, Exception):
                    raise clip
                outcome = await self._play(clip)
                if outcome == "stalled":
                    logger.warning("Narration paused: the browser never reported a clip played.")
                    await self.manager.broadcast_json(ErrorMessage(detail="playback stalled"))
                    self.manager.session.transition("PAUSED", "playback_stalled")
                    return
                if outcome == "stopped":
                    return  # whoever stopped the audio cancels or restarts this loop
                clip.played.set()
                if clip.segment_id is not None:
                    room.release()  # a clip with no segment never took a slot

        except Exception as exc:  # noqa: BLE001  # logged with its traceback, then the session pauses and the HUD is told
            logger.exception("Error in narration loop")
            await self.manager.broadcast_json(ErrorMessage(detail=f"narration stopped: {type(exc).__name__}"))
            if self.manager.state == "NARRATING":
                self.manager.session.transition("PAUSED", "narration_error")
        finally:
            if feeder is not None:
                feeder.cancel()
                # Waited for, so that no synthesis is still running when the loop is gone.
                await asyncio.gather(feeder, return_exceptions=True)

    def search_document_passages_with_scores(
        self,
        query: str,
        max_passages: int = 3,
        current_sentence_idx: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Search all sentences/windows in the loaded document using Okapi BM25 and Gaussian proximity boost."""
        if not self.sentences or not query.strip():
            return []

        scored = self._bm25_index.search(
            query=query,
            top_k=max_passages,
            current_sentence_idx=current_sentence_idx,
        )
        return [(text, score) for _, score, text in scored]

    def search_document_passages(
        self,
        query: str,
        max_passages: int = 3,
        current_sentence_idx: Optional[int] = None,
    ) -> List[str]:
        """Search all sentences/windows in loaded document for relevant passages matching query."""
        scored = self.search_document_passages_with_scores(
            query=query,
            max_passages=max_passages,
            current_sentence_idx=current_sentence_idx,
        )
        return [text for text, _ in scored]

    # --- replies --------------------------------------------------------------------------------------------------

    def _run_command(self, query: str) -> bool:
        """Carry out `query` if it is a pause, resume or repeat command, and say whether it was. These never reach the
        model or enter the history, and each is one change of the session at most."""
        intent = classify_local(query)
        if intent not in ("pause", "resume", "repeat"):
            return False  # a greeting is a reply (see `_answer`); everything else is a question
        state_before = self.manager.state
        if intent == "pause":
            self.pause_narration()
        elif intent == "resume":
            self.start_narration()
        else:
            self._repeat_sentence()
        if self.manager.state == state_before:
            self._bring_page_back(intent, state_before)
        return True

    def _bring_page_back(self, intent: str, state: str) -> None:
        """Typed text arrives from a page that has already stopped its own audio and shows LISTENING. When the command
        leaves the state as it was, no transition tells it where the session is, so it is told here, and whatever it
        stopped goes on for it: the narration that was playing starts again a few words back, and a reply whose clip
        was stopped stops waiting for a report that will never come."""
        manager = self.manager
        restart_narration = intent == "resume" and state == "NARRATING"
        if intent == "resume" and state in ("NARRATING", "REPLYING"):
            manager.stop_audio()
        if restart_narration:
            manager.tasks.cancel("narration")
        manager.broadcast_state_sync()  # queued before the restarted loop's first message
        if restart_narration:
            self._cue_pre_roll = True
            self._ensure_narration()

    def _repeat_sentence(self) -> None:
        """Narrate the current sentence again from its first word, from whatever the session was doing."""
        session = self.manager.session
        state = session.state
        self.current_word_idx = 0
        self.manager.update_cursor(sentence_idx=self.current_sentence_idx, word_offset=0)
        if state == "NARRATING":
            self._go_to(self.current_sentence_idx)
            return
        session.set_resume_to("NARRATING")
        if state == "PAUSED":
            session.transition("NARRATING", "repeat")
        else:
            # The question or answer in progress is dropped: the user asked to hear the book again.
            self.manager.tasks.cancel("reply")
            self.manager.tasks.cancel("stt")
            self.manager.stop_audio()
            session.resume("repeat")

    def ask(self, query: str) -> None:
        """Answer `query` now: stop what is playing, take the session over and speak the answer as the `reply` task.
        A pause, resume or repeat command is carried out instead and starts no reply.

        Synchronous on purpose. The state has changed when this returns, so nothing the caller does next (another
        question, a pause, an audio frame) can see a half-started reply, and the messages the change causes are queued
        ahead of everything the reply sends. A reply that is already running is replaced, and a transcription still
        pending is dropped: this question is the newest turn.
        """
        if self._run_command(query):
            return
        self._begin_reply("question", barge_in=True)
        self.manager.tasks.spawn(self._reply(query), "reply", replace=True)

    async def handle_user_interruption(self, query: str) -> None:
        """`ask`, but the caller waits for the whole exchange: the answer spoken and the follow-up window open."""
        if self._run_command(query):
            return
        self._begin_reply("question", barge_in=True)
        await self._reply(query)

    def report_unrecognized(self) -> None:
        """Say that the speech was not understood, then go back to what the session was doing."""
        self._begin_reply("unrecognized", barge_in=False)
        self.manager.tasks.spawn(self._unrecognized_reply(), "reply", replace=True)

    def _begin_reply(self, reason: str, *, barge_in: bool) -> None:
        tasks = self.manager.tasks
        tasks.cancel("stt")  # the caller may itself be the stt task, which `cancel` leaves alone
        tasks.cancel("narration")
        if barge_in:
            self.manager.announce_barge_in()
        self.manager.session.transition("REPLYING", reason)

    # The body coroutines are created inside the task, not handed to it: a task cancelled before its first step
    # never runs, and a coroutine it was given would be reported as never awaited.
    async def _reply(self, query: str) -> None:
        await self._guarded(self._answer(query))

    async def _unrecognized_reply(self) -> None:
        await self._guarded(self._say_unrecognized())

    async def _guarded(self, body: Awaitable[None]) -> None:
        """Run a reply body so that no way of ending it strands the session in REPLYING.

        Success, and cancellation, leave the state to the body and to whoever cancelled it: a barge-in, a pause, a
        replacement reply, a document load and a shutdown each already moved the session. Cancellation is not
        an error and is never reported as one. Any other failure is logged, reported and spoken, and the session
        is returned to what it was doing.
        """
        try:
            await body
        except Exception as exc:  # noqa: BLE001  # logged with its traceback, reported to the HUD, and the session is returned
            logger.exception("Reply failed")
            await self._report_failure(exc)

    async def _report_failure(self, exc: Exception) -> None:
        # Each step is on its own: a client that cannot be told, or a synthesizer that cannot speak, must not keep
        # the session from being returned. Cancellation here propagates, and then the canceller owns the state.
        try:
            await self.manager.broadcast_json(ReplyErrorMessage(detail=type(exc).__name__))
        except Exception:  # noqa: BLE001  # logged; the neutral line and the resume still follow
            logger.exception("Could not tell the clients that the reply failed")
        try:
            await self._speak(REPLY_ERROR_LINE)
        except Exception:  # noqa: BLE001  # logged; the resume still follows
            logger.exception("Could not speak the reply-error line")
        self.manager.session.resume("reply_error")

    def _estimated_duration_s(self, text: str, cadence: float) -> float:
        """How long `text` takes to speak, as far as the server can tell. The one estimate: it paces a sentence that has
        no clip to wait for, steps the word cursor, and is the length a missing report is waited for beyond."""
        words = max(1, len(text.split()))
        return max(self.settings.min_sentence_s, words / (self.settings.words_per_s * cadence))

    async def _speak(self, text: str) -> None:
        """Synthesize `text`, send it to the owner and wait until the browser says it has played. With nobody to say so
        (no owner, or no audio could be made) the wait is the estimate; if the report never comes the wait ends after
        `playback_ack_timeout_s` past the estimate, with a WARNING, and the reply goes on."""
        voice_id = self.manager.active_voice or "david-martinez"
        cadence = self.manager.cadence or 1.0
        speech = await self.tts.synthesize(text, voice_id, cadence)
        self.manager.refresh_statuses()
        estimate_s = self._estimated_duration_s(text, cadence)
        segment_id = await self.manager.send_audio(speech.data) if speech is not None and speech.data else None
        if segment_id is None:
            await asyncio.sleep(estimate_s)
        else:
            await self.manager.playback.wait_done(segment_id, estimate_s + self.settings.playback_ack_timeout_s)

    async def _say_unrecognized(self) -> None:
        current_context = self.sentences[self.current_sentence_idx] if self.sentences else ""
        await self.manager.broadcast_json(PromptEnvelopeMessage(
            state="REPLYING",
            context=current_context,
            query="",
            reply=UNRECOGNIZED_LINE,
        ))
        await self._speak(UNRECOGNIZED_LINE)
        # Back to what the session was doing; its state listener restarts narration if that was narrating.
        self.manager.session.resume("reply_done")

    async def _answer(self, query: str) -> None:
        """Answer the question and open the follow-up window. Raises on any failure.

        A greeting is answered locally. Anything else is grounded in the book (and the web, when the book is not
        authoritative) and put to the model. When the model gives no text, the closest passage of the book is read out
        instead, or the offline line when there is none.
        """
        current_context = self.sentences[self.current_sentence_idx] if self.sentences else ""
        voice_id = self.manager.active_voice or "david-martinez"
        if classify_local(query) == "greeting":
            await self._announce(query, current_context, greeting_reply(voice_id))
            await self._speak_then_listen(query, current_context, greeting_reply(voice_id))
            return

        scored = self.search_document_passages_with_scores(
            query, max_passages=3, current_sentence_idx=self.current_sentence_idx
        )
        passages = [text for text, _ in scored]
        authoritative = bool(scored) and scored[0][1] >= AUTHORITATIVE_SCORE

        has_model = self.llm.provider != "none"
        web_results = ""
        if has_model and self.llm.web_search_enabled and not authoritative:
            await self._announce(query, f"Searching web and document for: {query[:50]}...", "")
            web_results = await self.search(query, 3)

        reply = ""
        if has_model:
            messages = build_messages(
                system_prompt=CHARACTER_PERSONA_PROMPTS.get(voice_id, DEFAULT_PERSONA_PROMPT),
                history=self._conversation_history,
                question=query,
                passages=passages,
                current_sentence=current_context,
                web_results=web_results,
                authoritative=authoritative,
            )
            await self._announce(query, current_context, "Formulating answer...")
            reply = await self._stream_reply(messages, query, current_context, voice_id)

        status = self.llm.status if has_model else "none"
        if not reply and status != "ok":
            # No model, or one that gave nothing usable: read the book's closest passage rather than a canned line.
            passage = passages[0].strip() if passages else ""
            spoken = FROM_THE_TEXT + passage if passage else OFFLINE_LINE
            await self._announce(query, current_context, spoken)
        elif not reply:
            raise EmptyReply("the language model returned no text")
        else:
            spoken = reply
            # Only a real answer is remembered: an offline line or a half-finished reply would be sent back to the model
            # as something it had said, and a small model may repeat it.
            if status == "ok":
                self._remember(query, reply)
        await self._speak_then_listen(query, current_context, spoken)

    async def _announce(self, query: str, context: str, reply: str) -> None:
        await self.manager.broadcast_json(PromptEnvelopeMessage(state="REPLYING", context=context, query=query, reply=reply))

    async def _stream_reply(self, messages: list[dict[str, str]], query: str, context: str, voice_id: str) -> str:
        """The model's text, shown to the HUD as it arrives. The status is read once the stream has ended, and the HUD is
        told only if it changed: never once per token."""
        reply = ""
        # The stream is closed here, whatever ends the loop: a cancelled or failed reply must not leave a
        # connection to the model open until the generator is garbage collected.
        async with contextlib.aclosing(self.llm.stream_generate(messages, character_id=voice_id)) as stream:
            async for token in stream:
                reply += token
                await self._announce(query, context, reply.strip())
        # Not reached by a cancelled reply, so a barge-in is not reported as a model failure.
        self._llm_status = self.llm.status
        self.manager.refresh_statuses()
        return reply.strip()

    def _remember(self, question: str, answer: str) -> None:
        self._conversation_history.append({"role": "user", "content": question})
        self._conversation_history.append({"role": "assistant", "content": answer})
        del self._conversation_history[:-HISTORY_MESSAGES]

    async def _speak_then_listen(self, query: str, context: str, spoken: str) -> None:
        await self._speak(spoken)

        # Conversational turn-taking window: leave comms open for a follow-up. The reply ends here; the session
        # deadline closes the window, and a question asked inside it is a reply of its own.
        self.manager.clear_audio_turn()
        self.manager.session.enter_listening("followup", followup=True)
        # Queued, not awaited: it must reach the clients after the state_sync that the line above announced.
        self.manager.broadcast_json_sync(PromptEnvelopeMessage(state="LISTENING", context=context, query=query, reply=spoken))
