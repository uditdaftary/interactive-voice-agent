"""Behaviour of the HUD's pure functions, run in Node on the source in `ui/index.html`. The functions under test touch no
page element, so they can be lifted out of the script and called with plain data; the tests skip without Node."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

HUD_SOURCE = (Path(__file__).resolve().parent.parent / "ui" / "index.html").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")


def hud_function_source(name: str) -> str:
    """The text of `function name(...) { ... }` in the page. It counts braces, so the function must hold none in a
    string or a comment."""
    start = HUD_SOURCE.index(f"function {name}(")
    depth = 0
    for i in range(HUD_SOURCE.index("{", start), len(HUD_SOURCE)):
        if HUD_SOURCE[i] == "{":
            depth += 1
        elif HUD_SOURCE[i] == "}":
            depth -= 1
            if depth == 0:
                return HUD_SOURCE[start : i + 1]
    raise AssertionError(f"function {name} is not closed")


def call_hud_function(name: str, calls: list[list[Any]], helpers: tuple[str, ...] = ()) -> list[Any]:
    """Run the page's function once per argument list, in one Node process, and return the results. `helpers` are
    other page functions it calls."""
    script = (
        "\n".join(hud_function_source(helper) for helper in helpers)
        + "\n"
        + hud_function_source(name)
        + "\nconst calls = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        + f"\nprocess.stdout.write(JSON.stringify(calls.map((args) => {name}(...args))));"
    )
    done = subprocess.run(
        ["node", "-e", script], input=json.dumps(calls), capture_output=True, text=True, timeout=30, check=False
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# --- what a config save sends ---------------------------------------------------------------------------------------

LOADED = {"provider": "openai", "base_url": "https://api.example.test/v1", "model": "served-model"}

# Each case changes some fields of a form in which nothing was changed, picked or typed.
CONFIG_CASES: dict[str, dict[str, Any]] = {
    "nothing changed": {},
    "unlisted provider, not picked": {"provider": "local-builtin", "webSearch": False},
    "provider picked": {"provider": "groq", "providerPicked": True},
    "endpoint and model changed": {"baseUrl": "https://api.other.test/v1", "model": "other"},
    "only the model changed": {"model": "other"},
    "only the endpoint changed": {"baseUrl": "https://api.other.test/v1"},
    "no model picked": {"provider": "none", "providerPicked": True, "baseUrl": "", "model": "", "apiKey": "ignored"},
    "web search toggled, key field blank": {"webSearch": False},
    "key typed": {"apiKey": "sk-typed"},
    "removal asked for": {"clearKey": True},
    "removal asked for and a key typed": {"clearKey": True, "apiKey": "sk-typed"},
    "provider switched, key field blank": {
        "provider": "ollama",
        "providerPicked": True,
        "baseUrl": "http://localhost:11434/v1",
    },
    "provider switched, key typed": {"provider": "groq", "providerPicked": True, "apiKey": "sk-new"},
    "same provider picked again, key field blank": {"provider": "openai", "providerPicked": True},
}


@pytest.fixture(scope="module")
def sent() -> dict[str, dict[str, Any]]:
    """The body each case would send, from one Node run."""
    unchanged = {
        "provider": "openai",
        "providerPicked": False,
        "webSearch": True,
        "baseUrl": LOADED["base_url"],
        "model": LOADED["model"],
        "apiKey": "",
        "clearKey": False,
    }
    calls = [[{**unchanged, **fields}, LOADED] for fields in CONFIG_CASES.values()]
    results = call_hud_function("configPayload", calls, helpers=("destinationChanged",))
    return dict(zip(CONFIG_CASES, results, strict=True))


def test_a_save_that_changes_nothing_sends_no_provider_endpoint_or_model(sent: dict[str, dict[str, Any]]) -> None:
    for key in ("provider", "base_url", "model"):
        assert key not in sent["nothing changed"]


def test_a_provider_the_user_did_not_pick_is_never_sent(sent: dict[str, dict[str, Any]]) -> None:
    # The server runs a provider the form has no option for, so the form shows it as itself.
    assert "provider" not in sent["unlisted provider, not picked"]
    assert sent["unlisted provider, not picked"]["web_search_enabled"] is False


def test_a_picked_provider_is_sent(sent: dict[str, dict[str, Any]]) -> None:
    assert sent["provider picked"]["provider"] == "groq"


def test_an_endpoint_or_model_is_sent_only_when_it_changed(sent: dict[str, dict[str, Any]]) -> None:
    assert sent["endpoint and model changed"]["base_url"] == "https://api.other.test/v1"
    assert sent["endpoint and model changed"]["model"] == "other"
    assert "base_url" not in sent["only the model changed"]
    assert "model" not in sent["only the endpoint changed"]


def test_no_model_sends_neither_endpoint_nor_model_nor_key(sent: dict[str, dict[str, Any]]) -> None:
    assert sent["no model picked"] == {"web_search_enabled": True, "provider": "none"}


def test_a_blank_key_field_leaves_the_stored_key_alone_when_the_destination_is_the_same(
    sent: dict[str, dict[str, Any]],
) -> None:
    # Toggling web search with the key field blank must not replace the stored key with an empty one, and picking the
    # provider the server already uses is not a move.
    for case in (
        "nothing changed",
        "web search toggled, key field blank",
        "only the model changed",
        "same provider picked again, key field blank",
    ):
        assert "api_key" not in sent[case]


def test_a_stored_key_is_dropped_when_the_provider_or_endpoint_changes_and_no_key_was_typed(
    sent: dict[str, dict[str, Any]],
) -> None:
    # The old key must not go to the new host.
    for case in ("provider switched, key field blank", "provider picked", "only the endpoint changed"):
        assert sent[case]["api_key"] == ""
    assert sent["endpoint and model changed"]["api_key"] == ""


def test_a_key_typed_with_a_new_destination_is_the_one_sent(sent: dict[str, dict[str, Any]]) -> None:
    assert sent["provider switched, key typed"]["api_key"] == "sk-new"


def test_a_typed_key_is_sent(sent: dict[str, dict[str, Any]]) -> None:
    assert sent["key typed"]["api_key"] == "sk-typed"


def test_removing_the_stored_key_is_explicit(sent: dict[str, dict[str, Any]]) -> None:
    assert sent["removal asked for"]["api_key"] == ""
    # A key typed next to the request replaces the stored one, which also removes it.
    assert sent["removal asked for and a key typed"]["api_key"] == "sk-typed"


# --- the question box ----------------------------------------------------------------------------------------------


def test_the_box_is_cleared_only_while_it_holds_what_the_page_wrote() -> None:
    cleared, kept_follow_up, kept_edit, nothing_written = call_hud_function(
        "boxAfterTurn",
        [
            ["who is lucy", "who is lucy"],  # the preview's text, untouched
            ["what about the", "who is lucy"],  # a follow-up typed while the reply played
            ["who is lucy and", "who is lucy"],  # the preview's text, then edited
            ["typed", ""],  # the page wrote nothing: the box is the user's
        ],
    )
    assert cleared == ""
    assert kept_follow_up == "what about the"
    assert kept_edit == "who is lucy and"
    assert nothing_written == "typed"


def test_only_the_preview_transcript_is_sent_as_a_fallback() -> None:
    with_preview, preview_off, heard_nothing = call_hud_function(
        "voiceEndMessage",
        [
            [True, "who is lucy"],
            [False, "who is lucy"],
            [True, ""],  # the preview is on but heard nothing, whatever is typed in the box
        ],
    )
    assert with_preview == {"type": "voice_end", "fallback_query": "who is lucy"}
    assert preview_off == {"type": "voice_end"}
    assert heard_nothing == {"type": "voice_end"}
