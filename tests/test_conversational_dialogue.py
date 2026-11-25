from voice_agent.llm.client import CHARACTER_PERSONA_PROMPTS


def test_character_persona_prompts_elevated():
    """Verify CHARACTER_PERSONA_PROMPTS contain natural conversational instructions and anti-robotic rules."""
    for char_id in ["johnny-silverhand", "lucy-kushinada", "david-martinez", "v-male", "v-female"]:
        prompt = CHARACTER_PERSONA_PROMPTS[char_id]
        assert "spoken conversational sentences" in prompt
        assert "NEVER use markdown asterisks" in prompt
        assert len(prompt) > 100
