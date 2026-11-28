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
    if HUD_SOURCE[start - 6 : start] == "async ":
        start -= 6
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


# --- fetching the document -----------------------------------------------------------------------------------------

DOCUMENT_HARNESS = """
let STORY_PARAGRAPHS = [];
let currentSentenceIdx = 0;
let currentWordIdx = 0;
let cursorMoves = 0;
let documentSyncTotal = null;
let documentFetchSerial = 0;
const getApiBaseUrl = () => '';
const titleEl = { innerText: '' };
const document = { getElementById: () => titleEl };
const renderNarrativeStream = () => {};
let warnings = [];
console.warn = (...args) => warnings.push(args.map(String).join(' '));
const queued = [];
globalThis.fetch = () => queued.shift()();
const ok = (data) => () => Promise.resolve({ ok: true, status: 200, json: async () => data });
const status = (code) => () => Promise.resolve({ ok: false, status: code });
const down = () => () => Promise.reject(new Error('network down'));
const later = () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  return { reply: () => gate, release };
};
const doc = (title, sentences, idx) => ({ title, sentences, current_sentence_idx: idx });
const settle = () => new Promise((resolve) => setTimeout(resolve, 5));
"""


def run_document_scenario(body: str) -> Any:
    """Run `body` (async JavaScript that returns a JSON-able value) against the page's document fetching."""
    script = (
        hud_function_source("requestDocumentSync")
        + "\n"
        + hud_function_source("syncDocumentFromServer")
        + "\n"
        + DOCUMENT_HARNESS
        + "\n(async () => {"
        + body
        + "})().then((result) => process.stdout.write(JSON.stringify(result)));"
    )
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_failed_fetch_gives_the_count_back_and_logs_so_the_next_update_asks_again() -> None:
    result = run_document_scenario(
        """
        queued.push(status(500));
        requestDocumentSync(3);
        await settle();
        const afterServerError = { latch: documentSyncTotal, warnings: warnings.slice(), shown: STORY_PARAGRAPHS.length };
        queued.push(down());
        requestDocumentSync(3);
        await settle();
        const afterNetworkError = { latch: documentSyncTotal, warnings: warnings.slice(), shown: STORY_PARAGRAPHS.length };
        queued.push(ok(doc('Book', ['A.', 'B.', 'C.'], 1)));
        requestDocumentSync(3);
        await settle();
        return { afterServerError, afterNetworkError, shown: STORY_PARAGRAPHS, idx: currentSentenceIdx, latch: documentSyncTotal };
        """
    )
    assert result["afterServerError"]["latch"] is None
    assert any("HTTP 500" in line for line in result["afterServerError"]["warnings"])
    assert result["afterNetworkError"]["latch"] is None
    assert any("network down" in line for line in result["afterNetworkError"]["warnings"])
    assert result["afterNetworkError"]["shown"] == 0
    assert result["shown"] == ["A.", "B.", "C."]
    assert result["idx"] == 1
    assert result["latch"] == 3


def test_only_the_newest_fetch_is_applied() -> None:
    result = run_document_scenario(
        """
        // The connect-time fetch is slow; a cursor update starts a second one that answers first.
        const slow = later();
        queued.push(slow.reply);
        const first = syncDocumentFromServer();
        queued.push(ok(doc('New', ['N1.', 'N2.'], 1)));
        const second = syncDocumentFromServer();
        await second;
        slow.release({ ok: true, status: 200, json: async () => doc('Old', ['O1.', 'O2.', 'O3.'], 2) });
        await first;
        return { shown: STORY_PARAGRAPHS, idx: currentSentenceIdx, title: titleEl.innerText };
        """
    )
    assert result == {"shown": ["N1.", "N2."], "idx": 1, "title": "[BOOK: New]"}


def test_a_superseded_fetch_that_fails_does_not_give_back_the_newer_ones_count() -> None:
    result = run_document_scenario(
        """
        const slow = later();
        queued.push(slow.reply);
        const first = syncDocumentFromServer();
        queued.push(ok(doc('New', ['N1.'], 0)));
        requestDocumentSync(1);
        await settle();
        slow.release({ ok: false, status: 500 });
        await first;
        return { latch: documentSyncTotal, warnings };
        """
    )
    assert result == {"latch": 1, "warnings": []}


# --- the state the question panel shows ------------------------------------------------------------------------------


def test_the_question_panel_labels_only_a_state_it_knows_and_shows_no_label_for_any_other() -> None:
    table_start = HUD_SOURCE.index("const ENVELOPE_STATE_PILLS")
    table = HUD_SOURCE[table_start : HUD_SOURCE.index("]);", table_start) + 3]
    script = (
        "const setPill = (pill, className, text) => { pill.className = className; pill.text = text; };"
        + "const pill = { className: 'state-pill listening', text: '', style: {} };"
        + "const document = { getElementById: () => pill };"
        + table
        + hud_function_source("showEnvelopeState")
        + "const shown = [];"
        + "for (const state of ['LISTENING', 'REPLYING', 'NARRATING', undefined]) {"
        + "  showEnvelopeState(state); shown.push([pill.className, pill.text, pill.style.display]);"
        + "}"
        + "process.stdout.write(JSON.stringify(shown));"
    )
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [
        ["state-pill listening", "Listening", ""],
        ["state-pill replying", "Replying", ""],
        ["state-pill replying", "Replying", "none"],  # a state with no entry shows no label
        ["state-pill replying", "Replying", "none"],  # nor does none
    ]


# --- what the key field says when a save would drop the stored key ---------------------------------------------------

KEY_FORM = {
    "provider": "openai",
    "providerPicked": False,
    "baseUrl": LOADED["base_url"],
    "apiKey": "",
    "clearKey": False,
}


SENT_LINE = "A key is stored. It is sent to the endpoint above."
REPLACE_LINE = "The key you typed will replace the stored key."


def key_field_texts(has_key: bool, changes: dict[str, Any]) -> dict[str, str]:
    """What the note, the line about the stored key, and the field's placeholder show for a form with `changes`."""
    script = (
        hud_function_source("destinationChanged")
        + hud_function_source("updateKeyNote")
        + f"const loadedConfig = {{...{json.dumps(LOADED)}, hasKey: {json.dumps(has_key)}}};"
        + f"const readConfigForm = () => ({json.dumps({**KEY_FORM, **changes})});"
        + "const fields = { 'cfg-key-note': { style: {} }, 'cfg-key-sent': { style: {} }, 'cfg-api-key': {} };"
        + "const document = { getElementById: (id) => fields[id] };"
        + "updateKeyNote();"
        + "process.stdout.write(JSON.stringify({ note: fields['cfg-key-note'].style.display,"
        + " sent: fields['cfg-key-sent'].style.display, sentText: fields['cfg-key-sent'].textContent,"
        + " placeholder: fields['cfg-api-key'].placeholder }));"
    )
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize(
    "changes", [{"provider": "groq", "providerPicked": True}, {"baseUrl": "https://api.other.test/v1"}]
)
def test_the_form_says_the_stored_key_is_removed_when_the_destination_changes(changes: dict[str, Any]) -> None:
    shown = key_field_texts(True, changes)
    assert shown == {
        "note": "block",
        "sent": "none",
        "sentText": SENT_LINE,
        "placeholder": "Enter a key for the new provider or endpoint",
    }


def test_the_form_says_the_key_is_kept_and_sent_when_nothing_changes_it() -> None:
    shown = key_field_texts(True, {})
    assert shown == {
        "note": "none",
        "sent": "",
        "sentText": SENT_LINE,
        "placeholder": "A key is stored. Leave this blank to keep it.",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"apiKey": "sk-new"},
        {"provider": "groq", "providerPicked": True, "apiKey": "sk-new"},
        {"clearKey": True, "apiKey": "sk-new"},
    ],
)
def test_the_form_says_a_typed_key_replaces_the_stored_one(changes: dict[str, Any]) -> None:
    # With the remove box ticked as well, the typed key still wins: the save replaces the key, it does not only remove it.
    shown = key_field_texts(True, changes)
    assert shown["note"] == "none"
    assert shown["sent"] == ""
    assert shown["sentText"] == REPLACE_LINE


def test_the_form_says_nothing_about_a_key_when_none_is_stored() -> None:
    shown = key_field_texts(False, {"baseUrl": "https://api.other.test/v1"})
    assert shown["note"] == "none"
    assert shown["placeholder"] == "Enter remote API key or leave blank for local Ollama"


def test_a_document_applied_by_an_upload_resets_the_count_a_cursor_update_asks_for() -> None:
    result = run_document_scenario(
        """
        // A cursor update reports 5 sentences while 3 are shown, and the fetch for it is applied.
        queued.push(ok(doc('Old', ['A.', 'B.', 'C.'], 0)));
        requestDocumentSync(5);
        await settle();
        // An upload in this tab then replaces it with 2 sentences.
        queued.push(ok(doc('Mine', ['X.', 'Y.'], 0)));
        await syncDocumentFromServer();
        const afterUpload = { latch: documentSyncTotal, shown: STORY_PARAGRAPHS.length };
        // Another tab loads a document of 5 sentences: it must be fetched, although 5 was asked for before.
        queued.push(ok(doc('Theirs', ['1.', '2.', '3.', '4.', '5.'], 0)));
        requestDocumentSync(5);
        await settle();
        return { afterUpload, shown: STORY_PARAGRAPHS.length, unused: queued.length };
        """
    )
    assert result == {"afterUpload": {"latch": 2, "shown": 2}, "shown": 5, "unused": 0}


@pytest.mark.parametrize("changes", [{"clearKey": True}, {"clearKey": True, "baseUrl": "https://api.other.test/v1"}])
def test_the_form_follows_the_remove_checkbox(changes: dict[str, Any]) -> None:
    # The checkbox explains itself, so no note, but the field and the line above it stop saying the key is kept or sent.
    shown = key_field_texts(True, changes)
    assert shown == {
        "note": "none",
        "sent": "none",
        "sentText": SENT_LINE,
        "placeholder": "The stored key will be removed",
    }


def test_a_document_reply_keeps_a_cursor_the_server_reported_while_it_was_on_its_way() -> None:
    result = run_document_scenario(
        """
        const slow = later();
        queued.push(slow.reply);
        const fetched = syncDocumentFromServer();
        // A cursor_update arrives while the fetch is out, newer than the position the reply will hold.
        cursorMoves += 1; currentSentenceIdx = 2; currentWordIdx = 4;
        slow.release({ ok: true, status: 200, json: async () => doc('Book', ['A.', 'B.', 'C.', 'D.'], 0) });
        await fetched;
        const kept = { idx: currentSentenceIdx, word: currentWordIdx };
        // A position past the end of the new document is brought back into it, to the first word of the last sentence.
        const late = later();
        queued.push(late.reply);
        const clamped = syncDocumentFromServer();
        cursorMoves += 1; currentSentenceIdx = 9; currentWordIdx = 1;
        late.release({ ok: true, status: 200, json: async () => doc('Book', ['A.', 'B.'], 0) });
        await clamped;
        const pastTheEnd = { idx: currentSentenceIdx, word: currentWordIdx };
        // With no cursor update in between, the reply's position is used, from its first word.
        queued.push(ok(doc('Book', ['A.', 'B.', 'C.'], 1)));
        currentWordIdx = 7;
        await syncDocumentFromServer();
        return { kept, pastTheEnd, fromReply: { idx: currentSentenceIdx, word: currentWordIdx } };
        """
    )
    assert result == {
        "kept": {"idx": 2, "word": 4},
        "pastTheEnd": {"idx": 1, "word": 0},
        "fromReply": {"idx": 1, "word": 0},
    }


# --- the text of the question panel ----------------------------------------------------------------------------------


def test_an_envelope_with_no_query_clears_the_question_before_it() -> None:
    script = (
        hud_function_source("showEnvelopeText")
        + "const fields = { 'env-local-ctx': {}, 'env-query': {}, 'env-reply': {} };"
        + "const document = { getElementById: (id) => fields[id] };"
        + "const read = () => ({ context: fields['env-local-ctx'].innerText, query: fields['env-query'].innerText,"
        + " reply: fields['env-reply'].innerText });"
        + "const shown = [];"
        + "showEnvelopeText({ context: 'A line.', query: 'Who is Lucy?', reply: 'A friend.' }); shown.push(read());"
        + "showEnvelopeText({ context: 'A line.', query: '', reply: 'I did not catch that.' }); shown.push(read());"
        + "showEnvelopeText({ query: 'And Maine?' }); shown.push(read());"
        + "process.stdout.write(JSON.stringify(shown));"
    )
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [
        {"context": '"A line."', "query": '"Who is Lucy?"', "reply": '"A friend."'},
        {"context": '"A line."', "query": "", "reply": '"I did not catch that."'},
        {"context": '"A line."', "query": '"And Maine?"', "reply": '"I did not catch that."'},
    ]


# --- the words narration goes back over ------------------------------------------------------------------------------


def marked_pre_roll_words(count: int | None, stop_word: int, sentence_words: int = 12) -> dict[str, Any]:
    """Which words of sentence 2 the page marks for a cue of `count` words, narration having stopped at `stop_word`."""
    argument = "" if count is None else str(count)
    script = (
        hud_function_source("highlightPreRollWords")
        + f"let currentSentenceIdx = 2; let currentWordIdx = {stop_word};"
        + "const word = (s, w) => {"
        + "  const el = { dataset: { sIdx: String(s), wIdx: String(w) }, marked: false, classList: { add: () => { el.marked = true; } } };"
        + "  return el;"
        + "};"
        + f"const wordsArray = [word(1, 5), ...Array.from({{ length: {sentence_words} }}, (_, w) => word(2, w))];"
        + f"highlightPreRollWords({argument});"
        + "process.stdout.write(JSON.stringify({ cursor: currentWordIdx,"
        + " marked: wordsArray.filter((el) => el.marked).map((el) => el.dataset.sIdx + ':' + el.dataset.wIdx) }));"
    )
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_cue_marks_exactly_the_words_it_goes_back_over() -> None:
    # Five words back from word 7 are words 2 to 6; the word narration stopped at is not one of them.
    shown = marked_pre_roll_words(5, 7)
    assert shown == {"cursor": 2, "marked": ["2:2", "2:3", "2:4", "2:5", "2:6"]}


def test_a_cue_longer_than_the_sentence_so_far_marks_only_the_words_before_the_stop() -> None:
    assert marked_pre_roll_words(5, 3) == {"cursor": 0, "marked": ["2:0", "2:1", "2:2"]}
    assert marked_pre_roll_words(5, 0) == {"cursor": 0, "marked": []}


def test_a_cue_with_no_count_goes_back_four_words() -> None:
    assert marked_pre_roll_words(None, 9)["marked"] == ["2:5", "2:6", "2:7", "2:8"]
