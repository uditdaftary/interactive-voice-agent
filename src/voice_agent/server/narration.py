"""Active Real-Time Narration & Interruption Engine for Interactive Voice Agent.

Drives synchronized audio synthesis, teleprompter word cursor tracking,
and real-time LLM query interruption with live internet web search grounding.
"""

import asyncio
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
    PromptEnvelopeMessage,
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
        self._lock = asyncio.Lock()
        self._is_replying = False
        self._conversation_history: List[dict] = []
        self._interruption_count: int = 0
        self._listening_ended = asyncio.Event()  # set whenever the session leaves LISTENING

        # The engine owns turn-taking timing, so it hands the configured waits to the session it drives.
        session = self.manager.session
        session.listen_deadline_s = self.settings.barge_in_listen_s
        session.followup_window_s = self.settings.followup_window_s
        session.add_listener(self._on_state_change)

    def _on_state_change(self, old: str, new: str, reason: str) -> None:
        """Keep the background tasks in step with the session state."""
        if old == "LISTENING" and new != "LISTENING":
            self._listening_ended.set()
        if new == "PAUSED" and not self.manager.session.resuming:
            # Pausing stops everything that speaks or listens, wherever the pause came from. (Returning to a
            # paused session after a reply or a deadline is not a pause: nothing is left to stop, and the
            # reply may be the very task finishing.)
            for name in ("narration", "reply", "stt"):
                self.manager.tasks.cancel(name)
        elif new == "LISTENING":
            # A barge-in, from any source, silences narration; the deadline or the reply starts it again.
            self.manager.tasks.cancel("narration")
        elif new == "NARRATING":
            self._ensure_narration()

    def _ensure_narration(self) -> None:
        if self.manager.tasks.get("narration") is None:
            self.manager.tasks.spawn(self._narration_loop(), "narration")

    def load_document(self, doc: DocumentContent) -> None:
        """Load a parsed document (PDF, EPUB, TXT) into the active narrator."""
        if not doc.sentences:
            return
        self.sentences = list(doc.sentences)
        self._bm25_index = BM25Index.from_sentences(self.sentences)
        self.document_title = doc.title
        self.current_sentence_idx = 0
        self.current_word_idx = 0

        # Broadcast state sync with updated document info
        self.manager.update_cursor(sentence_idx=0, word_offset=0)
        self.manager.broadcast_json_sync(self.manager.get_state_sync_message())
        logger.info(f"Loaded document '{doc.title}' with {len(self.sentences)} sentences, {doc.total_words} words.")

    def start_narration(self) -> None:
        """Start or resume background narration loop. Refused (and logged) from a state that may not narrate."""
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

                # 3. Broadcast audio frame to all connected clients
                if audio_bytes:
                    await self.manager.broadcast_bytes(audio_bytes)

                # 4. Stream word cursor highlighting across sentence duration
                words = sentence_text.split()
                num_words = max(1, len(words))
                # Estimated duration based on cadence (words_per_s at 1.0x)
                duration_sec = max(self.settings.min_sentence_s, num_words / (self.settings.words_per_s * cadence))
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

        except Exception:  # noqa: BLE001  # narration ends; logged with its traceback
            logger.exception("Error in narration loop")

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

    async def handle_user_interruption(self, query: str) -> None:
        """Handle real-time barge-in question: grounds query with web search & book passages, queries LLM, speaks answer."""
        async with self._lock:
            # 1. Instantly terminate any background narration loop to avoid race conditions
            await self.manager.tasks.cancel_and_wait("narration")

            # 2. Instantaneous barge-in drain (< 50ms). The session broadcasts each state change itself.
            self.manager.trigger_barge_in()
            self.manager.session.transition("REPLYING", "question")

            current_context = self.sentences[self.current_sentence_idx] if self.sentences else ""
            voice_id = self.manager.active_voice or "david-martinez"
            cadence = self.manager.cadence or 1.0

            # 3. Retrieve relevant passages with BM25 & Gaussian proximity boost
            scored_passages = self.search_document_passages_with_scores(
                query,
                max_passages=3,
                current_sentence_idx=self.current_sentence_idx,
            )
            doc_passages = [text for text, _ in scored_passages]
            best_doc_score = scored_passages[0][1] if scored_passages else 0.0
            is_authoritative = best_doc_score >= 2.5

            # 4. Live Internet Web Search Grounding (skipped if document passage has high BM25 confidence >= 2.5)
            web_grounding = ""
            if self.llm.web_search_enabled and not is_authoritative:
                await self.manager.broadcast_json(PromptEnvelopeMessage(
                    state="LISTENING",
                    context=f"Searching web and document for: {query[:50]}...",
                    query=query,
                    reply="",
                ))
                web_grounding = await self.search(query, 3)

            # 5. Context Budgeting & ChatML Sanitization (enforces 2048-token limit for RTX 3050 4GB VRAM)
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

            self._interruption_count += 1
            current_turn = self._interruption_count

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

            # 6. Stream LLM tokens to Prompt Envelope
            accumulated_reply = ""

            await self.manager.broadcast_json(PromptEnvelopeMessage(
                state="REPLYING",
                context=current_context,
                query=query,
                reply="Formulating answer...",
            ))

            async for token in self.llm.stream_generate(messages, character_id=voice_id):
                accumulated_reply += token
                # Periodic telemetry update
                await self.manager.broadcast_json(PromptEnvelopeMessage(
                    state="REPLYING",
                    context=current_context,
                    query=query,
                    reply=accumulated_reply.strip(),
                ))

            if not accumulated_reply.strip():
                if "johnny" in voice_id:
                    final_reply = "Got it, samurai. Keep your eyes on the road and your chrome hot. Ready when you are."
                elif "lucy" in voice_id:
                    final_reply = "Understood. The signal is clear. Ready to continue our narrative, David."
                elif "v-female" in voice_id:
                    final_reply = "Copy that. Objectives locked and tactical feed active. Resuming mission."
                elif "v-male" in voice_id or voice_id == "v":
                    final_reply = "Understood choom. Keep your iron ready and stay frosty. Ready when you are."
                else:
                    final_reply = "Understood choomba! Chrome is spooled up and ready. Let's roll whenever you say!"
            else:
                final_reply = accumulated_reply.strip()

            # Record turns in multi-turn conversation history
            self._conversation_history.append({"role": "user", "content": clean_query})
            self._conversation_history.append({"role": "assistant", "content": final_reply})
            if len(self._conversation_history) > 12:
                self._conversation_history = self._conversation_history[-12:]

            # 7. Synthesize LLM answer to speech audio and stream to client
            audio_bytes, fmt = await self.tts(final_reply, voice_id, cadence)
            if audio_bytes:
                await self.manager.broadcast_bytes(audio_bytes)

            # Estimate duration to let speech finish playing in client
            reply_words = len(final_reply.split())
            playback_delay = max(
                self.settings.min_reply_playback_s,
                reply_words / (self.settings.reply_words_per_s * cadence),
            )
            await asyncio.sleep(playback_delay)

            # 8. Conversational Turn-Taking Window: leave comms open for follow-up before auto-resuming
            self._listening_ended.clear()
            self.manager.session.enter_listening("followup", followup=True)
            await self.manager.broadcast_json(PromptEnvelopeMessage(
                state="LISTENING",
                context=current_context,
                query=query,
                reply=final_reply,
            ))

            # The session deadline ends this window (and a user who is speaking extends it), so wait for the
            # session to leave LISTENING rather than counting polls. A newer question or a pause means someone
            # else owns the session now.
            await self._listening_ended.wait()
            if self._interruption_count != current_turn or self.manager.state != "NARRATING":
                return

            # 9. The deadline resumed narration (its state listener restarted the loop): show the pre-roll cue
            await self.manager.broadcast_json(PromptEnvelopeMessage(
                state="NARRATING",
                context=current_context,
                query=query,
                reply=final_reply,
                pre_roll_words=4,
            ))

    async def handle_unrecognized_interruption(self) -> None:
        """Handle cases where barge-in was triggered but speech could not be recognized."""
        async with self._lock:
            await self.manager.tasks.cancel_and_wait("narration")

            voice_id = self.manager.active_voice or "david-martinez"
            cadence = self.manager.cadence or 1.0
            current_context = self.sentences[self.current_sentence_idx] if self.sentences else ""

            if "johnny" in voice_id:
                recovery_text = "Static on the comms, samurai. Didn't catch that. Resuming the run."
            elif "lucy" in voice_id:
                recovery_text = "Signal dropped in the subnet. Let's return to the narrative, David."
            elif "v-female" in voice_id or "v-male" in voice_id or voice_id == "v":
                recovery_text = "Comms were fuzzy, choom—didn't copy that. Getting back to the op."
            else:
                recovery_text = "Couldn't hear ya over the street noise, choombatta! Let's get right back to the story."

            self.manager.session.transition("REPLYING", "unrecognized")
            await self.manager.broadcast_json(PromptEnvelopeMessage(
                state="REPLYING",
                context=current_context,
                query="",
                reply=recovery_text,
            ))

            audio_bytes, fmt = await self.tts(recovery_text, voice_id, cadence)
            if audio_bytes:
                await self.manager.broadcast_bytes(audio_bytes)

            await asyncio.sleep(self.settings.unrecognized_pause_s)

            # Pre-Roll Resumption Cue & return to NARRATING
            await self.manager.broadcast_json(PromptEnvelopeMessage(
                state="NARRATING",
                context=current_context,
                query="",
                reply=recovery_text,
                pre_roll_words=4,
            ))
            await asyncio.sleep(self.settings.preroll_pause_s)

            # Back to what the session was doing; the state listener restarts narration if that was narrating.
            self.manager.session.resume("reply_done")
