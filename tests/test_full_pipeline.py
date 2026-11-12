import pytest
import numpy as np
from voice_agent.orchestrator.state_machine import AgentState
from voice_agent.pipeline import VoiceAgentCoordinator
from voice_agent.llm.client import MockLLMClient

@pytest.mark.asyncio
async def test_end_to_end_barge_in_and_resumption():
    story = [
        "Sentence one of the story.",
        "Sentence two about a mysterious visitor in the parlor.",
        "Sentence three concluding the chapter."
    ]
    coordinator = VoiceAgentCoordinator(story_text=story)
    
    # Start narration
    await coordinator.start_narration_step()
    assert coordinator.state == AgentState.NARRATING
    assert coordinator.cursor.pos.sentence_idx == 0
    
    # Simulate user barge-in speech
    latency_ms = await coordinator.simulate_user_interruption("Who visited the parlor?")
    assert latency_ms < 50.0  # Barge-in cutoff must be < 50 ms
    assert coordinator.state == AgentState.LISTENING
    
    # Process the turn
    await coordinator.process_turn()
    assert coordinator.state == AgentState.NARRATING
    
    # Narration resumed with continuity
    resumed = coordinator.cursor.get_resumption_text()
    assert "Sentence" in resumed

@pytest.mark.asyncio
async def test_3_sentence_llm_constraint():
    """Verify LLM output streaming is strictly cut off after 3 sentences."""
    story = ["A quiet morning in the valley."]
    coordinator = VoiceAgentCoordinator(story_text=story)
    
    # LLM mock producing 6 sentences
    long_reply = (
        "First sentence explaining the lore. "
        "Second sentence detailing the history. "
        "Third sentence answering the user. "
        "Fourth sentence that should be pruned. "
        "Fifth sentence that must never appear. "
        "Sixth sentence strictly forbidden."
    )
    coordinator.llm = MockLLMClient(long_reply)
    
    await coordinator.simulate_user_interruption("Tell me about the history.")
    emitted = await coordinator.process_turn(max_sentences=3)
    
    assert len(emitted) == 3
    assert emitted[0] == "First sentence explaining the lore."
    assert emitted[1] == "Second sentence detailing the history."
    assert emitted[2] == "Third sentence answering the user."

@pytest.mark.asyncio
async def test_trailing_unpunctuated_text_flush():
    """Verify trailing unpunctuated text from LLM stream is routed without loss via parser.flush()."""
    story = ["The old tower stood silently."]
    coordinator = VoiceAgentCoordinator(story_text=story)
    
    # LLM produces one punctuated sentence followed by trailing unpunctuated text
    coordinator.llm = MockLLMClient("The key is made of bronze. It unlocks the subterranean chamber")
    
    await coordinator.simulate_user_interruption("What does the key unlock?")
    emitted = await coordinator.process_turn(max_sentences=3)
    
    assert len(emitted) == 2
    assert emitted[0] == "The key is made of bronze."
    assert emitted[1] == "It unlocks the subterranean chamber"

@pytest.mark.asyncio
async def test_generation_fencing_anti_zombie():
    """Verify synthesized audio chunks with obsolete generation tags are rejected after barge-in."""
    story = ["Chapter One."]
    coordinator = VoiceAgentCoordinator(story_text=story)
    
    initial_gen = coordinator.generation
    dummy_audio = np.ones(2400, dtype=np.int16)
    
    # Valid write under initial generation
    written = coordinator.write_tts_chunk(dummy_audio, chunk_generation=initial_gen)
    assert written > 0
    
    # Barge-in occurs -> drains buffer and increments generation
    coordinator._on_barge_in()
    assert coordinator.generation == initial_gen + 1
    
    # Stale in-flight TTS synthesis finishing under initial_gen must be discarded
    stale_written = coordinator.write_tts_chunk(dummy_audio, chunk_generation=initial_gen)
    assert stale_written == 0

def test_playhead_tracking_and_cursor_advance():
    """Verify audio playhead updates spoken word offset and finish_sentence advances sentence index."""
    story = [
        "The quick brown fox jumps over the lazy dog.",
        "A second sentence follows."
    ]
    coordinator = VoiceAgentCoordinator(story_text=story)
    assert coordinator.cursor.pos.sentence_idx == 0
    assert coordinator.cursor.pos.word_offset == 0
    
    # 1.5 seconds elapsed @ 2.5 words/sec -> ~3 words offset
    coordinator.update_playhead_progress(elapsed_seconds=1.5, words_per_second=2.5)
    assert coordinator.cursor.pos.word_offset >= 3
    
    # Finish sentence playback
    advanced = coordinator.finish_sentence_playback()
    assert advanced is True
    assert coordinator.cursor.pos.sentence_idx == 1
    assert coordinator.cursor.pos.word_offset == 0
