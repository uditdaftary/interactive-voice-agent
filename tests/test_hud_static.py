"""Static checks on the single-page HUD (`ui/index.html`): it simulates nothing, talks to one server, and never
writes server data into the page as markup. These read the source, so they run without a browser."""

import re
from pathlib import Path

import pytest

HUD_SOURCE = (Path(__file__).resolve().parent.parent / "ui" / "index.html").read_text(encoding="utf-8")

# What the page must not contain.
REMOVED = [
    # The offline simulation: with the server away the page shows that, and does not play the story by itself.
    "startSimulationLoop",
    "stopSimulationLoop",
    "isLiveMode",
    "advanceWordCursor",
    "narrativeTimer",
    "OFFLINE SIMULATION",
    # One server: the page is served by the app, so there is no second port and no `file:` address.
    "8080",
    "localhost:8000",
    # Numbers the page made up: a timing added in the browser, and levels drawn from a formula.
    "0.04",
    "generateSimulatedFFTBins",
    "-14.2 dB",
    # Claims the page made about the model and the machine, true or not: the server says which provider it uses.
    "Qwen 3.5",
    "ZERO-CLOUD",
    # The preview text went to the server whatever the server allowed.
    "fallback_query: existingVal",
    # The built-in provider and its placeholders: with no model the app reads the closest passage instead.
    "builtin",
    "in-process",
    # Em and en dashes in copy.
    chr(0x2014),  # em dash
    chr(0x2013),  # en dash
    # The audio banner was capitals, mono type and a product slogan.
    "NEURAL AUDIOBOOK",
    # The built-in story and its title: the reader shows the server's document or says there is none.
    "Arrival in the Megabuilding",
    "Night City Run",
    "stood at the edge of the catwalk",
    "scav vehicle",
    "let currentSentenceIdx = 3",
    # A fixed 3 s retry: the wait now grows from 1 s to 30 s.
    "reconnectInterval",
    # Assigned to when the server's document arrives: a const makes that throw, and the catch hid it.
    "const STORY_PARAGRAPHS",
]


@pytest.mark.parametrize("text", REMOVED)
def test_hud_has_no_simulation_second_server_or_invented_number(text: str) -> None:
    assert text not in HUD_SOURCE


def test_hud_audio_banner_says_what_to_do_in_plain_words() -> None:
    assert "Click anywhere to turn on sound." in HUD_SOURCE


def test_hud_says_so_when_it_has_no_document() -> None:
    assert "No document loaded" in HUD_SOURCE
    assert "let STORY_PARAGRAPHS = [];" in HUD_SOURCE


def test_hud_labels_the_online_speech_preview() -> None:
    assert "Preview (uses an online service)" in HUD_SOURCE


# --- no server data in markup -------------------------------------------------------------------------------------

_MARKUP_SINK = re.compile(r"(?:innerHTML|outerHTML)\s*\+?=(?!=)")
# After a closing quote, a line that goes on with one of these continues the expression.
_CONTINUATION = "+-*/%?:.([,|&=<>!"


def _statement_ends_after_literal(source: str, i: int) -> bool:
    """True when the statement is over at `i`: a `;`, a `}`, the end of the text, or a new line that does not go on
    with an operator (a statement ended by automatic semicolon insertion)."""
    while i < len(source) and source[i] in " \t\r":
        i += 1
    if i >= len(source) or source[i] in ";}":
        return True
    if source[i] != "\n":
        return False
    while i < len(source) and source[i] in " \t\r\n":
        i += 1
    return i >= len(source) or source[i] not in _CONTINUATION


def unsafe_markup_assignments(source: str) -> list[str]:
    """The `innerHTML` / `outerHTML` assignments in `source` whose right side is not exactly one string literal.

    Only a single quoted string, or a template literal with no `${`, passes. A conditional, a `||`, a `+`, a
    variable or a call is unsafe even when it holds a literal, because data can sit in the other branch.
    """
    unsafe = []
    for sink in _MARKUP_SINK.finditer(source):
        i = sink.end()
        while i < len(source) and source[i] in " \t\r\n":
            i += 1
        fixed = i < len(source) and source[i] in "'\"`"
        if fixed:
            quote = source[i]
            i += 1
            while i < len(source) and source[i] != quote:
                if source[i] == "\\":
                    i += 1
                elif quote == "`" and source.startswith("${", i):
                    fixed = False
                    break
                i += 1
            fixed = fixed and _statement_ends_after_literal(source, i + 1)
        if not fixed:
            line = source.count("\n", 0, sink.start()) + 1
            unsafe.append(f"line {line}: {source[sink.start():sink.start() + 60]!r}")
    return unsafe


@pytest.mark.parametrize(
    "statement",
    [
        "el.innerHTML = `<b>${name}</b>`;",
        "el.innerHTML += `<i>${name}</i>`;",
        "el.innerHTML = '<b>' + name + '</b>';",
        "el.innerHTML = markup;",
        "el.outerHTML = build(name);",
        "el.innerHTML = `\n  <td>static</td>\n  <td>${name}</td>\n`;",
        "el.innerHTML = ok ? `<b>${name}</b>` : '';",
        "el.innerHTML = ok ? '<b>' : name;",
        "el.innerHTML = name || '';",
        "el.innerHTML = ok ? '<a></a>' : '<b></b>';",
        "el.innerHTML = '<b>'\n  + name;",
        "el.innerHTML = '' + name;",
    ],
)
def test_checker_flags_data_in_markup(statement: str) -> None:
    assert unsafe_markup_assignments(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "el.innerHTML = '';",
        "el.innerHTML = '<path d=\"M2 5h3\"/>';",
        "el.innerHTML = `<span>static; ok</span>`;",
        "el.innerHTML = ''\nnext();",
        "if (a === b) { el.textContent = `${name}`; }",
    ],
)
def test_checker_passes_fixed_markup(statement: str) -> None:
    assert not unsafe_markup_assignments(statement)


def test_hud_never_writes_server_data_as_markup() -> None:
    assert unsafe_markup_assignments(HUD_SOURCE) == []


@pytest.mark.parametrize("call", ["insertAdjacentHTML", "document.write", "eval(", "new Function("])
def test_hud_has_no_other_markup_or_code_sink(call: str) -> None:
    assert call not in HUD_SOURCE
