"""Static checks on the single-page HUD (`ui/index.html`): it simulates nothing, talks to one server, and never
writes server data into the page as markup. These read the source, so they run without a browser."""

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
    # A fixed 3 s retry: the wait now grows from 1 s to 30 s.
    "reconnectInterval",
    # Assigned to when the server's document arrives: a const makes that throw, and the catch hid it.
    "const STORY_PARAGRAPHS",
]


@pytest.mark.parametrize("text", REMOVED)
def test_hud_has_no_simulation_second_server_or_invented_number(text: str) -> None:
    assert text not in HUD_SOURCE
