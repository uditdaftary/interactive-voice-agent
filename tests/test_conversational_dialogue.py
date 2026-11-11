import pytest
from voice_agent.llm.client import LLMClient, CHARACTER_PERSONA_PROMPTS


def test_character_persona_prompts_elevated():
    """Verify CHARACTER_PERSONA_PROMPTS contain natural conversational instructions and anti-robotic rules."""
    for char_id in ["johnny-silverhand", "lucy-kushinada", "david-martinez", "v-male", "v-female"]:
        prompt = CHARACTER_PERSONA_PROMPTS[char_id]
        assert "spoken conversational sentences" in prompt
        assert "NEVER use markdown asterisks" in prompt
        assert len(prompt) > 100


def test_conversational_greetings():
    """Verify natural conversational responses to greetings and social check-ins across character personas."""
    client = LLMClient()
    
    # David Martinez
    resp_david = client._synthesize_fallback_answer("User Question: hey david, what's up?", character_id="david-martinez")
    assert "choom" in resp_david.lower()
    assert "santo domingo" in resp_david.lower() or "sandevistan" in resp_david.lower()
    assert "understood choomba" not in resp_david.lower()

    # Johnny Silverhand
    resp_johnny = client._synthesize_fallback_answer("User Question: hey johnny, are you there?", character_id="johnny-silverhand")
    assert "samurai" in resp_johnny.lower()
    assert "burn" in resp_johnny.lower() or "spit it out" in resp_johnny.lower()
    assert "understood" not in resp_johnny.lower()

    # Lucy Kushinada
    resp_lucy = client._synthesize_fallback_answer("User Question: hello lucy", character_id="lucy-kushinada")
    assert "feed" in resp_lucy.lower() or "interface" in resp_lucy.lower()
    assert "david" in resp_lucy.lower() or "reading" in resp_lucy.lower()

    # V (Male & Female)
    resp_vm = client._synthesize_fallback_answer("User Question: yo v, how's it going?", character_id="v-male")
    assert "choom" in resp_vm.lower()
    assert "comms" in resp_vm.lower() or "listening" in resp_vm.lower()

    resp_vf = client._synthesize_fallback_answer("User Question: hey v", character_id="v-female")
    assert "loud and clear" in resp_vf.lower() or "intel" in resp_vf.lower()


def test_conversational_identity_inquiry():
    """Verify characters introduce themselves authentically rather than generic canned text."""
    client = LLMClient()

    resp_david = client._synthesize_fallback_answer("User Question: who are you?", character_id="david-martinez")
    assert "david martinez" in resp_david.lower()
    assert "sandevistan" in resp_david.lower()

    resp_johnny = client._synthesize_fallback_answer("User Question: what's your name and story?", character_id="johnny-silverhand")
    assert "johnny silverhand" in resp_johnny.lower()
    assert "arasaka" in resp_johnny.lower() or "twenty-twenty-three" in resp_johnny.lower()

    resp_lucy = client._synthesize_fallback_answer("User Question: who are you?", character_id="lucy-kushinada")
    assert "lucy" in resp_lucy.lower()
    assert "netrunner" in resp_lucy.lower()

    resp_v = client._synthesize_fallback_answer("User Question: tell me about yourself", character_id="v-male")
    assert "v" in resp_v.lower()
    assert "mercenary" in resp_v.lower() or "solo" in resp_v.lower()


def test_conversational_repeat_command_with_context():
    """Verify repeat requests quote the current reading sentence in character voice."""
    client = LLMClient()
    user_msg = (
        'Currently Reading Sentence: "Above the smog-choked megaliths, Arasaka orbital shuttles drifted."\n\n'
        'User Question: can you repeat that sentence?'
    )

    resp_david = client._synthesize_fallback_answer(user_msg, character_id="david-martinez")
    assert "Above the smog-choked megaliths" in resp_david
    assert "rewinding" in resp_david.lower() or "choom" in resp_david.lower()

    resp_johnny = client._synthesize_fallback_answer(user_msg, character_id="johnny-silverhand")
    assert "Above the smog-choked megaliths" in resp_johnny
    assert "playback" in resp_johnny.lower() or "samurai" in resp_johnny.lower()


def test_conversational_pause_command():
    """Verify pause/wait commands receive responsive in-character confirmation."""
    client = LLMClient()

    resp_david = client._synthesize_fallback_answer("User Question: hold on a second", character_id="david-martinez")
    assert "holding" in resp_david.lower() or "take your time" in resp_david.lower()

    resp_johnny = client._synthesize_fallback_answer("User Question: pause for a minute", character_id="johnny-silverhand")
    assert "brakes" in resp_johnny.lower() or "smoke" in resp_johnny.lower() or "breath" in resp_johnny.lower()

    resp_lucy = client._synthesize_fallback_answer("User Question: wait a minute", character_id="lucy-kushinada")
    assert "paused" in resp_lucy.lower() or "subnets" in resp_lucy.lower() or "quiet" in resp_lucy.lower()


def test_conversational_lore_opinions():
    """Verify character questions yield authentic lore dialogue without generic robotic refusal."""
    client = LLMClient()

    # Asking David about Lucy
    resp_d_lucy = client._synthesize_fallback_answer("User Question: what do you think of Lucy?", character_id="david-martinez")
    assert "moon" in resp_d_lucy.lower() or "netrunner" in resp_d_lucy.lower()
    assert "nova" in resp_d_lucy.lower() or "skin" in resp_d_lucy.lower()

    # Asking Johnny about Sandevistan
    resp_j_sande = client._synthesize_fallback_answer("User Question: tell me about the sandevistan", character_id="johnny-silverhand")
    assert "chrome" in resp_j_sande.lower() or "cyberpsychosis" in resp_j_sande.lower()

    # Asking David about Sandevistan
    resp_d_sande = client._synthesize_fallback_answer("User Question: how does your sandevistan feel?", character_id="david-martinez")
    assert "spine" in resp_d_sande.lower() or "coolant" in resp_d_sande.lower() or "lightning" in resp_d_sande.lower()


def test_conversational_gratitude_and_appreciation():
    """Verify gratitude and compliments receive natural, appreciative character dialogue."""
    client = LLMClient()

    resp_j = client._synthesize_fallback_answer("User Question: thanks for the help johnny", character_id="johnny-silverhand")
    assert "samurai" in resp_j.lower()
    assert "watch" in resp_j.lower() or "backs" in resp_j.lower() or "ready" in resp_j.lower()

    resp_d = client._synthesize_fallback_answer("User Question: thank you david, good job", character_id="david-martinez")
    assert "choombatta" in resp_d.lower() or "crew" in resp_d.lower()

    resp_l = client._synthesize_fallback_answer("User Question: thanks lucy", character_id="lucy-kushinada")
    assert "david" in resp_l.lower()
    assert "focus" in resp_l.lower() or "continue" in resp_l.lower()


def test_conversational_acknowledgment_and_confirmation():
    """Verify casual agreement (ok, got it, makes sense) yields turn-taking readiness."""
    client = LLMClient()

    resp_j = client._synthesize_fallback_answer("User Question: got it, makes sense", character_id="johnny-silverhand")
    assert "samurai" in resp_j.lower()
    assert "preem" in resp_j.lower() or "optics" in resp_j.lower() or "ready" in resp_j.lower()

    resp_d = client._synthesize_fallback_answer("User Question: okay cool", character_id="david-martinez")
    assert "nova" in resp_d.lower() or "sandevistan" in resp_d.lower() or "standby" in resp_d.lower()


def test_conversational_continuation_commands():
    """Verify resume/continue voice requests trigger immediate transition back to reading."""
    client = LLMClient()

    resp_j = client._synthesize_fallback_answer("User Question: continue reading", character_id="johnny-silverhand")
    assert "picking" in resp_j.lower() or "samurai" in resp_j.lower()

    resp_d = client._synthesize_fallback_answer("User Question: keep going, read on", character_id="david-martinez")
    assert "choomba" in resp_d.lower() or "feed" in resp_d.lower() or "action" in resp_d.lower()


def test_conversational_extended_lore_smasher_and_relic():
    """Verify Adam Smasher and Relic inquiries yield rich lore dialogue."""
    client = LLMClient()

    resp_smasher = client._synthesize_fallback_answer("User Question: who is Adam Smasher?", character_id="johnny-silverhand")
    assert "smasher" in resp_smasher.lower() or "arasaka" in resp_smasher.lower()
    assert "borg" in resp_smasher.lower() or "monster" in resp_smasher.lower() or "twenty-three" in resp_smasher.lower()

    resp_relic = client._synthesize_fallback_answer("User Question: what is the Relic?", character_id="johnny-silverhand")
    assert "relic" in resp_relic.lower()
    assert "biochip" in resp_relic.lower() or "engram" in resp_relic.lower() or "soulkiller" in resp_relic.lower()

    resp_blackwall = client._synthesize_fallback_answer("User Question: what is the Blackwall?", character_id="lucy-kushinada")
    assert "blackwall" in resp_blackwall.lower()
    assert "ai" in resp_blackwall.lower() or "firewall" in resp_blackwall.lower() or "net" in resp_blackwall.lower()
