"""Static checks on the single-page HUD (`ui/index.html`): it simulates nothing, talks to one server, and never
writes server data into the page as markup. These read the source, so they run without a browser."""

from pathlib import Path

import pytest

HUD_SOURCE = (Path(__file__).resolve().parent.parent / "ui" / "index.html").read_text(encoding="utf-8")

# What the page must not contain.
REMOVED = [
    # Assigned to when the server's document arrives: a const makes that throw, and the catch hid it.
    "const STORY_PARAGRAPHS",
]


@pytest.mark.parametrize("text", REMOVED)
def test_hud_has_no_simulation_second_server_or_invented_number(text: str) -> None:
    assert text not in HUD_SOURCE
