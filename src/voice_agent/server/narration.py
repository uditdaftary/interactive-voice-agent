"""Active Real-Time Narration & Interruption Engine for Interactive Voice Agent.

Drives synchronized audio synthesis, teleprompter word cursor tracking,
and real-time LLM query interruption with live internet web search grounding.
"""

import asyncio
import logging
import time
from typing import List, Optional, Tuple
import numpy as np

from voice_agent.document.bm25 import BM25Index
from voice_agent.document.parser import DocumentContent
from voice_agent.llm.client import (
    CHARACTER_PERSONA_PROMPTS,
    LLMClient,
    PunctuationStreamParser,
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
    StateSyncMessage,
)
from voice_agent.tts.synthesizer import synthesize_sentence

logger = logging.getLogger(__name__)

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
    ):
        self.manager = connection_manager
        self.llm = llm_client or LLMClient()
        self.sentences: List[str] = list(initial_sentences or DEFAULT_STORY_SENTENCES)
        self._bm25_index: BM25Index = BM25Index.from_sentences(self.sentences)
        self.document_title: str = "Night City: Santo Domingo Run"
        self.current_sentence_idx: int = 0
        self.current_word_idx: int = 0
        self._narration_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._is_replying = False

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
        """Start or resume background narration loop."""
        self.manager.set_state("NARRATING")
        if self._narration_task is None or self._narration_task.done():
            self._narration_task = asyncio.create_task(self._narration_loop())
        logger.info("Narration started/resumed.")

    def pause_narration(self) -> None:
        """Pause narration loop."""
        if self.manager.state == "NARRATING":
            self.manager.set_state("PAUSED")
        if self._narration_task and not self._narration_task.done():
            self._narration_task.cancel()
        logger.info("Narration paused.")

    async def _narration_loop(self) -> None:
        """Main narration loop: synthesizes each sentence, streams audio, and steps cursor."""
        try:
            while self.manager.state == "NARRATING":
                if not self.sentences:
                    await asyncio.sleep(0.5)
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
                audio_bytes, fmt = await synthesize_sentence(sentence_text, voice_id=voice_id, cadence=cadence)

                if self.manager.state != "NARRATING":
                    break

                # 3. Broadcast audio frame to all connected clients
                if audio_bytes:
                    await self.manager.broadcast_bytes(audio_bytes)

                # 4. Stream word cursor highlighting across sentence duration
                words = sentence_text.split()
                num_words = max(1, len(words))
                # Estimated duration based on cadence (approx 2.5 words per sec at 1.0x)
                duration_sec = max(1.2, num_words / (2.5 * cadence))
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
                await asyncio.sleep(0.35 / cadence)

                # Step to next sentence
                self.current_sentence_idx = (self.current_sentence_idx + 1) % len(self.sentences)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Error in narration loop: {exc}")

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
            if self._narration_task and not self._narration_task.done():
                self._narration_task.cancel()
                try:
                    await self._narration_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._narration_task = None

            # 2. Instantaneous barge-in drain (< 50ms)
            barge_in_event = self.manager.trigger_barge_in()
            await self.manager.broadcast_json(barge_in_event)

            self.manager.set_state("REPLYING")
            sync_msg = self.manager.get_state_sync_message()
            await self.manager.broadcast_json(sync_msg)

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
                web_grounding = await search_web(query, max_results=3)

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

            messages = [
                {
                    "role": "system",
                    "content": character_prompt,
                },
                {
                    "role": "user",
                    "content": f"{context_block}\n\nUser Question: {clean_query}",
                },
            ]

            # 6. Stream LLM tokens to Prompt Envelope
            parser = PunctuationStreamParser()
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

            final_reply = accumulated_reply.strip() or "I have processed your question. Let us return to the narrative."

            # 7. Synthesize LLM answer to speech audio and stream to client
            audio_bytes, fmt = await synthesize_sentence(final_reply, voice_id=voice_id, cadence=cadence)
            if audio_bytes:
                await self.manager.broadcast_bytes(audio_bytes)

            # Estimate duration to let speech finish playing in client
            reply_words = len(final_reply.split())
            playback_delay = max(2.0, reply_words / (2.2 * cadence))
            await asyncio.sleep(playback_delay)

            # 8. Pre-Roll Resumption Cue & return to NARRATING
            await self.manager.broadcast_json(PromptEnvelopeMessage(
                state="NARRATING",
                context=current_context,
                query=query,
                reply=final_reply,
                pre_roll_words=4,
            ))
            await asyncio.sleep(1.0)

            # 9. Cleanly resume background narration
            self.manager.set_state("NARRATING")
            await self.manager.broadcast_json(self.manager.get_state_sync_message())
            self.start_narration()
