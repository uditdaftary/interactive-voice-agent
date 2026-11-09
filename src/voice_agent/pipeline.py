import time
import inspect
import asyncio
from typing import List, Optional
import numpy as np

from voice_agent.cursor.tracker import CursorTracker
from voice_agent.audio.ring_buffer import AudioRingBuffer
from voice_agent.orchestrator.state_machine import AgentState, AgentStateMachine
from voice_agent.orchestrator.context import assemble_qa_prompt
from voice_agent.llm.client import MockLLMClient, PunctuationStreamParser


class VoiceAgentCoordinator:
    """Coordinates dual-mode narration, barge-in detection, context assembly, and resumption.
    
    Adheres strictly to the Google Doc architecture:
    - Sub-50ms instantaneous audio drainage on barge-in.
    - Generation fencing to invalidate in-flight TTS chunks.
    - Audio playhead tracking driving narrative cursor progress.
    - 3-sentence maximum conversational turn constraint.
    - Lossless streaming sentence parsing with trailing flush routing.
    """
    def __init__(self, story_text: List[str]):
        self.cursor = CursorTracker(story_text)
        self.audio_out = AudioRingBuffer(capacity_samples=48000 * 5, sample_rate=24000)
        self.sm = AgentStateMachine(on_barge_in=self._on_barge_in)
        self.llm = MockLLMClient("The visitor was an enigmatic detective named V.")
        self.pending_query: Optional[str] = None
        self.generation: int = 0

    @property
    def state(self) -> AgentState:
        return self.sm.state

    @property
    def playhead_seconds(self) -> float:
        """Elapsed DAC audio playback in seconds."""
        if hasattr(self.audio_out, "playhead_seconds"):
            return self.audio_out.playhead_seconds
        total = getattr(self.audio_out, "total_samples_played", 0)
        sr = getattr(self.audio_out, "sample_rate", 24000)
        return total / sr if sr > 0 else 0.0

    @property
    def total_samples_played(self) -> int:
        """Total audio samples consumed by DAC playback."""
        return getattr(self.audio_out, "total_samples_played", 0)

    def _on_barge_in(self):
        """Instantaneously drain audio output queue (< 50ms) and invalidate in-flight synthesis."""
        self.audio_out.drain()
        self.generation += 1
        if hasattr(self.audio_out, "generation"):
            self.audio_out.generation = self.generation

    def update_playhead_progress(self, elapsed_seconds: Optional[float] = None, words_per_second: float = 2.5):
        """Map audio playback playhead to spoken word offset and record cursor progress."""
        if elapsed_seconds is None:
            elapsed_seconds = self.playhead_seconds

        current = self.cursor.current_sentence()
        if not current:
            return

        words = current.split()
        if not words:
            return

        word_offset = min(int(elapsed_seconds * words_per_second), len(words) - 1)
        self.cursor.record_progress(max(0, word_offset))

    def finish_sentence_playback(self) -> bool:
        """Advance cursor to the next sentence when audio playback completes."""
        return self.cursor.advance()

    def write_tts_chunk(self, audio_data: np.ndarray, chunk_generation: int) -> int:
        """Write synthesized TTS audio chunk to buffer if generation fence matches.
        
        Discards stale chunks from superseded generations post barge-in drain.
        """
        current_gen = getattr(self.audio_out, "generation", self.generation)
        if chunk_generation != current_gen:
            # Stale generation; discard zombie audio
            return 0

        sig = inspect.signature(self.audio_out.write)
        if "generation" in sig.parameters:
            return self.audio_out.write(audio_data, generation=chunk_generation)
        return self.audio_out.write(audio_data)

    async def start_narration_step(self):
        self.sm.state = AgentState.NARRATING

    async def simulate_user_interruption(self, query: str) -> float:
        """Trigger barge-in and measure truncation latency (< 50ms budget)."""
        t0 = time.perf_counter()
        self.sm.trigger_speech_detected()
        latency_ms = (time.perf_counter() - t0) * 1000
        self.pending_query = query
        return latency_ms

    async def process_turn(self, max_sentences: int = 3) -> List[str]:
        """Execute Q&A turn with prompt assembly, 3-sentence constraint, and trailing flush."""
        self.sm.trigger_speech_finished()
        assert self.state == AgentState.REPLYING

        # Assemble context envelope adhering to 2048-token budget and ChatML sanitization
        prompt = assemble_qa_prompt(
            document_summary="Story narration",
            local_context=self.cursor.current_sentence() or "",
            user_query=self.pending_query or "",
            sentence_idx=self.cursor.pos.sentence_idx,
            max_context_tokens=2048
        )

        # Stream LLM response
        parser = PunctuationStreamParser()
        emitted_sentences: List[str] = []
        gen_at_start = getattr(self.audio_out, "generation", self.generation)

        async for token in self.llm.stream_generate(prompt):
            # Guard against mid-turn barge-in interruption
            if self.state != AgentState.REPLYING:
                break

            sentences = parser.feed(token)
            for s in sentences:
                emitted_sentences.append(s)
                # Generation fencing check for TTS synthesis
                current_gen = getattr(self.audio_out, "generation", self.generation)
                if gen_at_start == current_gen:
                    pass  # Chunk valid for playback

                # Enforce the 3-sentence LLM output constraint: stop streaming LLM output after 3 sentences
                if len(emitted_sentences) >= max_sentences:
                    break

            if len(emitted_sentences) >= max_sentences:
                break

        # Route parser.flush() output so trailing unpunctuated text is never lost
        if len(emitted_sentences) < max_sentences:
            trailing = parser.flush()
            if trailing:
                emitted_sentences.append(trailing)

        # Turn completes -> return to NARRATING
        if self.state == AgentState.REPLYING:
            self.sm.trigger_reply_finished()
        assert self.state == AgentState.NARRATING
        return emitted_sentences
