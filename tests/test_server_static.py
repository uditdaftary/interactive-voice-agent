"""Tier 2: Static Asset Serving & HUD Dashboard HTTP Integration Tests.

Validates that FastAPI serves the single-page HUD dashboard static assets
(`ui/index.html`, `ui/app.css`, `ui/fonts/`) at `http://localhost:8000/` with status 200,
exact MIME types, zero font 404 errors, and returns 404 for missing assets.
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from voice_agent.server.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture providing a TestClient configured with default project static assets."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_get_root_serves_index_html(client: TestClient) -> None:
    """Test GET / serves ui/index.html with status 200 and required HUD DOM elements."""
    response = client.get("/")
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    assert "text/html" in response.headers.get("content-type", "")

    content = response.text
    # Verify core HUD Dashboard markup
    assert "Voice Agent // Interactive HUD Dashboard" in content
    assert 'id="voice-graph-canvas"' in content
    assert 'id="bargein-wave-canvas"' in content
    assert 'id="narrative-container"' in content
    assert 'id="ws-status-pill"' in content
    assert 'id="mic-toggle-group"' in content
    assert 'id="fft-bars-container"' in content

    # Verify resident character roster table
    assert "David Martinez" in content
    assert "Lucy Kushinada" in content
    assert "Johnny Silverhand" in content


def test_get_app_css_and_design_tokens(client: TestClient) -> None:
    """Test GET /app.css returns status 200 with The Catalog Sheet design system tokens."""
    response = client.get("/app.css")
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
    assert "text/css" in response.headers.get("content-type", "")

    css = response.text
    # Verify palette tokens
    assert "--ground: #0b0b0b;" in css
    assert "--ink: #efede6;" in css
    assert "--high: #e8b622;" in css
    assert "--critical: #e03a2f;" in css
    assert "--sleeve: #121211;" in css

    # Verify 44px button and tab geometry
    assert "min-height: 44px;" in css
    assert "width: 44px;" in css


def test_get_fonts_zero_404(client: TestClient) -> None:
    """Test GET /fonts/* returns status 200 and non-empty font payload with zero 404s."""
    archivo_resp = client.get("/fonts/archivo-latin-wdth-normal.woff2")
    assert archivo_resp.status_code == 200, (
        f"Archivo font returned {archivo_resp.status_code}"
    )
    assert len(archivo_resp.content) == 90104, (
        f"Unexpected Archivo font size: {len(archivo_resp.content)} bytes"
    )

    martian_resp = client.get("/fonts/martian-mono-latin-wdth-normal.woff2")
    assert martian_resp.status_code == 200, (
        f"Martian Mono font returned {martian_resp.status_code}"
    )
    assert len(martian_resp.content) == 38492, (
        f"Unexpected Martian Mono font size: {len(martian_resp.content)} bytes"
    )


def test_non_existent_static_asset_returns_404(client: TestClient) -> None:
    """Test requesting non-existent static assets returns 404 Not Found."""
    resp_missing_page = client.get("/non_existent_page.html")
    assert resp_missing_page.status_code == 404

    resp_missing_css = client.get("/missing_style.css")
    assert resp_missing_css.status_code == 404

    resp_missing_font = client.get("/fonts/missing_font.woff2")
    assert resp_missing_font.status_code == 404


def test_custom_ui_directory_override(tmp_path: Path) -> None:
    """Test create_app accepts and mounts a custom ui_dir correctly."""
    custom_index = tmp_path / "index.html"
    custom_index.write_text("<!DOCTYPE html><html><body>Custom HUD</body></html>", encoding="utf-8")

    custom_css = tmp_path / "app.css"
    custom_css.write_text("body { background: #000; }", encoding="utf-8")

    app = create_app(ui_dir=tmp_path)
    with TestClient(app) as custom_client:
        r_root = custom_client.get("/")
        assert r_root.status_code == 200
        assert "Custom HUD" in r_root.text

        r_css = custom_client.get("/app.css")
        assert r_css.status_code == 200
        assert "background: #000" in r_css.text
