"""Active Real-Time Narration & Interruption Engine for Interactive Voice Agent.

Drives synchronized audio synthesis, teleprompter word cursor tracking,
and real-time LLM query interruption with live internet web search grounding.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import List, Optional, Tuple

from voice_agent.document.bm25 import BM25Index
from voice_agent.document.parser import DocumentContent
from voice_agent.llm.client import (
    CHARACTER_PERSONA_PROMPTS,
    LLMClient,
    search_web,
)
from voice_agent.orchestrator.context import (
    estimate_tokens,
    sanitize_chatml,
    truncate_to_tokens,
)
from voice_agent.server.connection_manager import ConnectionManager
from voice_agent.server.schemas import (
    CursorUpdateMessage,
    ErrorMessage,
    PromptEnvelopeMessage,
    ReplyErrorMessage,
)
from voice_agent.settings import Settings, load_settings
from voice_agent.tts.synthesizer import synthesize_sentence

logger = logging.getLogger(__name__)

# Called as tts(text, voice_id, cadence); returns (audio bytes or None, format name).
TtsFn = Callable[[str, str, float], Awaitable[tuple[bytes | None, str]]]
# Called as search(query, max_results); returns the grounding text, "" when nothing was found.
SearchFn = Callable[[str, int], Awaitable[str]]


async def _default_tts(text: str, voice_id: str, cadence: float) -> tuple[bytes | None, str]:
    # Resolves synthesize_sentence at call time so the module attribute stays patchable.
    return await synthesize_sentence(text, voice_id=voice_id, cadence=cadence)


# Spoken when a reply fails or the user's speech was not understood. Neutral on purpose: no persona, no detail.
REPLY_ERROR_LINE = "Something went wrong answering that."
UNRECOGNIZED_LINE = "I didn't catch that."


class EmptyReply(Exception):
    """The language model finished without producing any text."""


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
        tts: Optional[TtsFn] = None,
        search: Optional[SearchFn] = None,
    ):
        self.manager = connection_manager
        self.settings = settings if settings is not None else load_settings()
        self.tts: TtsFn = tts if tts is not None else _default_tts
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

    def _on_owner_change(self) -> None:
        """The audio goes to a different tab now. What the old one was playing is stopped either way."""
        state = self.manager.state
        if state == "NARRATING":
            # The new owner has not heard the sentence that is playing: it starts again for them.
            self.manager.stop_audio()
            self.manager.tasks.cancel("narration")
            self._ensure_narration()
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
            # Coming back from a question or a listening window is announced to the HUD; a plain resume is not.
            self._cue_pre_roll = old in ("LISTENING", "REPLYING")
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

        Ignored while a reply is in progress: the reply returns the session to what it was doing when it ends, and
        resuming now would run narration under the answer. Every tab sends a resume when its socket opens, so a second
        tab must not be able to cut an answer off. (From LISTENING it still works: that is how a user stops waiting.)
        """
        if self.manager.state == "REPLYING":
            logger.debug("Resume ignored: a reply is in progress and will return the session to what it was doing")
            return
        if not self.manager.session.transition("NARRATING", "resume"):
            return
        self._ensure_narration()  # the state listener starts it on a change; this covers a loop that has since ended
        logger.info("Narration started/resumed.")

    def pause_narration(self) -> None:
        """Pause: stops narration, any reply being spoken and any pending transcription."""
        self.manager.session.transition("PAUSED", "pause")
        logger.info("Narration paused.")

    async def _narration_loop(self) -> None:
        """Main narration loop: synthesizes each sentence, streams audio, and steps cursor."""
        try:
            if self._cue_pre_roll:
                self._cue_pre_roll = False
                # Queued behind the state_sync that started this loop, ahead of the first cursor_update.
                await self.manager.broadcast_json(PromptEnvelopeMessage(
                    state="NARRATING",
                    context=self.sentences[self.current_sentence_idx % len(self.sentences)] if self.sentences else "",
                    pre_roll_words=4,
                ))
            while self.manager.state == "NARRATING":
                if not self.sentences:
                    await asyncio.sleep(self.settings.idle_poll_s)
                    continue

                if self.current_sentence_idx >= len(self.sentences):
                    self.current_sentence_idx = 0  # Loop back to beginning

                sentence_text = self.sentences[self.current_sentence_idx]
                voice_id = self.manager.active_voice or "david-martinez"
                cadence = self.manager.cadence or 1.0

                # 1. Update cursor to sentence start
                self.current_word_idx = 0
                self.manager.update_cursor(
                    sentence_idx=self.current_sentence_idx,
                    word_offset=0,
                )
                cursor_msg = CursorUpdateMessage(
                    sentence_idx=self.current_sentence_idx,
                    word_offset=0,
                    sentence_text=sentence_text,
                    total_sentences=len(self.sentences),
                )
                await self.manager.broadcast_json(cursor_msg)

                # 2. Synthesize audio for current sentence
                audio_bytes, fmt = await self.tts(sentence_text, voice_id, cadence)

                if self.manager.state != "NARRATING":
                    break

                # 3. Send the clip to the owner connection
                if audio_bytes:
                    segment_id = await self.manager.send_audio(audio_bytes)
                    if segment_id is not None:
                        self.manager.playback.forget(segment_id)  # paced by the estimate below

                # 4. Stream word cursor highlighting across sentence duration
                words = sentence_text.split()
                num_words = max(1, len(words))
                duration_sec = self._estimated_duration_s(sentence_text, cadence)
                step_interval = duration_sec / num_words

                for w_idx in range(num_words):
                    if self.manager.state != "NARRATING":
                        break
                    self.current_word_idx = w_idx
                    self.manager.update_cursor(
                        sentence_idx=self.current_sentence_idx,
                        word_offset=w_idx,
                    )
                    await self.manager.broadcast_json(CursorUpdateMessage(
                        sentence_idx=self.current_sentence_idx,
                        word_offset=w_idx,
                        sentence_text=sentence_text,
                        total_sentences=len(self.sentences),
                    ))
                    await asyncio.sleep(step_interval)

                if self.manager.state != "NARRATING":
                    break

                # Pause briefly between sentences
                await asyncio.sleep(self.settings.inter_sentence_pause_s / cadence)

                # Step to next sentence
                self.current_sentence_idx = (self.current_sentence_idx + 1) % len(self.sentences)

        except Exception as exc:  # noqa: BLE001  # logged with its traceback, then the session pauses and the HUD is told
            logger.exception("Error in narration loop")
            await self.manager.broadcast_json(ErrorMessage(detail=f"narration stopped: {type(exc).__name__}"))
            if self.manager.state == "NARRATING":
                self.manager.session.transition("PAUSED", "narration_error")

    def search_document_passages_with_scores(
        self,
        query: str,
        max_passages: int = 3,
        current_sentence_idx: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Search all sentences/windows in the loaded document using Okapi BM25 and Gaussian proximity boost."""
        if not self.sentences or not query.strip():
            return []

        # Cyberpunk voice recognition alias normalization
        query_norm = query.lower()
        if "arakas" in query_norm:
            query_norm += " arasaka"
        if "sande" in query_norm and "sandevistan" not in query_norm:
            query_norm += " sandevistan"

        scored = self._bm25_index.search(
            query=query_norm,
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

    def ask(self, query: str) -> None:
        """Answer `query` now: stop what is playing, take the session over and speak the answer as the `reply` task.

        Synchronous on purpose. The state has changed when this returns, so nothing the caller does next (another
        question, a pause, an audio frame) can see a half-started reply, and the messages the change causes are queued
        ahead of everything the reply sends. A reply that is already running is replaced, and a transcription still
        pending is dropped: this question is the newest turn.
        """
        self._begin_reply("question", barge_in=True)
        self.manager.tasks.spawn(self._reply(query), "reply", replace=True)

    async def handle_user_interruption(self, query: str) -> None:
        """`ask`, but the caller waits for the whole exchange: the answer spoken and the follow-up window open."""
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
        audio_bytes, _fmt = await self.tts(text, voice_id, cadence)
        estimate_s = self._estimated_duration_s(text, cadence)
        segment_id = await self.manager.send_audio(audio_bytes) if audio_bytes else None
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
        """Ground the question, stream the answer, speak it and open the follow-up window. Raises on any failure."""
        current_context = self.sentences[self.current_sentence_idx] if self.sentences else ""
        voice_id = self.manager.active_voice or "david-martinez"

        # 1. Retrieve relevant passages with BM25 & Gaussian proximity boost
        scored_passages = self.search_document_passages_with_scores(
            query,
            max_passages=3,
            current_sentence_idx=self.current_sentence_idx,
        )
        doc_passages = [text for text, _ in scored_passages]
        best_doc_score = scored_passages[0][1] if scored_passages else 0.0
        is_authoritative = best_doc_score >= 2.5

        # 2. Live Internet Web Search Grounding (skipped if document passage has high BM25 confidence >= 2.5)
        web_grounding = ""
        if self.llm.web_search_enabled and not is_authoritative:
            await self.manager.broadcast_json(PromptEnvelopeMessage(
                state="REPLYING",
                context=f"Searching web and document for: {query[:50]}...",
                query=query,
                reply="",
            ))
            web_grounding = await self.search(query, 3)

        # 3. Context Budgeting & ChatML Sanitization (enforces 2048-token limit for RTX 3050 4GB VRAM)
        clean_query = sanitize_chatml(query)
        clean_query = truncate_to_tokens(clean_query, max_tokens=256, from_end=False)
        clean_current_context = sanitize_chatml(current_context)
        clean_passages = [sanitize_chatml(p) for p in doc_passages]
        clean_web_grounding = sanitize_chatml(web_grounding)

        character_prompt = CHARACTER_PERSONA_PROMPTS.get(
            voice_id,
            (
                "You are an interactive conversational voice audiobook narrator. "
                "The user interrupted the book narration to ask you a question. "
                "Answer concisely, helpfully, and naturally in 1 to 3 spoken sentences. "
                "Use the provided book context and web search results when answering."
            ),
        )

        # Calculate token allowances
        system_tokens = estimate_tokens(character_prompt) + 30
        output_headroom = 256  # Reserved for 3-sentence generation
        max_context_budget = 2048
        available_user_tokens = max(256, max_context_budget - system_tokens - output_headroom)
        query_tokens = estimate_tokens(clean_query)
        remaining_context_budget = max(100, available_user_tokens - query_tokens)

        context_parts = []
        if clean_passages:
            if is_authoritative:
                header = "Authoritative Document Passages (Primary Source of Truth):\n"
            else:
                header = "Passages from Document:\n"
            context_parts.append(header + "\n".join(f"- {p}" for p in clean_passages))
        if clean_current_context:
            context_parts.append(f"Currently Reading Sentence: \"{clean_current_context}\"")
        if clean_web_grounding:
            context_parts.append(f"Live Internet Web Search Results:\n{clean_web_grounding}")

        raw_context_block = "\n\n".join(context_parts)
        context_block = truncate_to_tokens(raw_context_block, max_tokens=remaining_context_budget, from_end=False)

        messages = [
            {
                "role": "system",
                "content": character_prompt,
            },
        ]
        # Inject up to 3 prior turns of conversational context for multi-turn dialogue
        for prev_turn in self._conversation_history[-6:]:
            messages.append(prev_turn)

        messages.append({
            "role": "user",
            "content": f"{context_block}\n\nUser Question: {clean_query}",
        })

        # 4. Stream LLM tokens to the prompt envelope
        await self.manager.broadcast_json(PromptEnvelopeMessage(
            state="REPLYING",
            context=current_context,
            query=query,
            reply="Formulating answer...",
        ))
        accumulated_reply = ""
        # The stream is closed here, whatever ends the loop: a cancelled or failed reply must not leave a
        # connection to the model open until the generator is garbage collected.
        async with contextlib.aclosing(self.llm.stream_generate(messages, character_id=voice_id)) as stream:
            async for token in stream:
                accumulated_reply += token
                # Periodic telemetry update
                await self.manager.broadcast_json(PromptEnvelopeMessage(
                    state="REPLYING",
                    context=current_context,
                    query=query,
                    reply=accumulated_reply.strip(),
                ))

        final_reply = accumulated_reply.strip()
        if not final_reply:
            raise EmptyReply("the language model returned no text")

        # The history keeps any non-empty text the client returns, and a reply stays in it even if speaking it fails.
        # The fallback line the client yields when the model is unreachable is non-empty, so it is kept too; telling
        # it apart from a real answer needs the client to say which it was.
        self._conversation_history.append({"role": "user", "content": clean_query})
        self._conversation_history.append({"role": "assistant", "content": final_reply})
        if len(self._conversation_history) > 12:
            self._conversation_history = self._conversation_history[-12:]

        # 5. Synthesize the answer to speech and stream it to the clients
        await self._speak(final_reply)

        # 6. Conversational turn-taking window: leave comms open for a follow-up. The reply ends here; the session
        # deadline closes the window, and a question asked inside it is a reply of its own.
        self.manager.clear_audio_turn()
        self.manager.session.enter_listening("followup", followup=True)
        # Queued, not awaited: it must reach the clients after the state_sync that the line above announced.
        self.manager.broadcast_json_sync(PromptEnvelopeMessage(
            state="LISTENING",
            context=current_context,
            query=query,
            reply=final_reply,
        ))
