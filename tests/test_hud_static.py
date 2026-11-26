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
    # The built-in provider and its placeholders: with no model the agent reads the closest passage instead.
    "builtin",
    "in-process",
    # Em and en dashes in copy.
    chr(0x2014),  # em dash
    chr(0x2013),  # en dash
    # A fixed 3 s retry: the wait now grows from 1 s to 30 s.
    "reconnectInterval",
    # Assigned to when the server's document arrives: a const makes that throw, and the catch hid it.
    "const STORY_PARAGRAPHS",
]


@pytest.mark.parametrize("text", REMOVED)
def test_hud_has_no_simulation_second_server_or_invented_number(text: str) -> None:
    assert text not in HUD_SOURCE


def test_hud_labels_the_online_speech_preview() -> None:
    assert "Preview (uses an online service)" in HUD_SOURCE


# --- no server data in markup -------------------------------------------------------------------------------------

_MARKUP_SINK = re.compile(r"(?:innerHTML|outerHTML)\s*\+?=(?!=)")
_LITERAL = "\x00"


def unsafe_markup_assignments(source: str) -> list[str]:
    """The `innerHTML` / `outerHTML` assignments in `source` whose right side is not made of fixed strings.

    A right side is unsafe when a template literal interpolates (`${`), when strings are joined with `+`, or when
    it holds no string literal at all (a variable or a call). A ternary between two fixed strings is fine.
    """
    unsafe = []
    for sink in _MARKUP_SINK.finditer(source):
        code: list[str] = []
        interpolated = False
        i = sink.end()
        while i < len(source) and source[i] != ";":
            if source[i] in "'\"`":
                quote = source[i]
                i += 1
                while i < len(source) and source[i] != quote:
                    if source[i] == "\\":
                        i += 1
                    elif quote == "`" and source.startswith("${", i):
                        interpolated = True
                    i += 1
                code.append(_LITERAL)
            else:
                code.append(source[i])
            i += 1
        right_side = "".join(code)
        if interpolated or "+" in right_side or _LITERAL not in right_side:
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
    ],
)
def test_checker_flags_data_in_markup(statement: str) -> None:
    assert unsafe_markup_assignments(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "el.innerHTML = '';",
        "el.innerHTML = '<path d=\"M2 5h3\"/>';",
        "el.innerHTML = on ? `<span>on; ok</span>` : `<span>off</span>`;",
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
