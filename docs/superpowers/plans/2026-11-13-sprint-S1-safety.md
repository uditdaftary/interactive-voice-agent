# Sprint S1: server boundary, input validation and ingestion hardening. Implementation plan

> **For agentic workers:** follow the orchestration protocol in `docs/superpowers/plans/2026-11-13-remediation-roadmap.md`. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The server can't be driven by other web pages or LAN hosts. Every request body is validated before anything is applied. Crafted documents can't exhaust memory or block the event loop, and parse failures are visible.

**Architecture:**
- A new `server/security.py` holds the Host allowlist and one Origin-equals-Host check, used by an HTTP middleware and by the WebSocket handshake.
- Request bodies become Pydantic models in `schemas.py`.
- The parser gains bounded EPUB/PDF reading and OPF spine order, and returns `skipped_parts`.
- Parsing and transcription run in `asyncio.to_thread`.

**Tech stack:** FastAPI/Starlette (`TrustedHostMiddleware` is already installed), Pydantic v2, pypdf, zipfile.

**Spec:** sections 5.5 (`/api/transcribe` only), 5.8 and 8 (interim input path, P1 bullet). Matrix: H3, H4, H5, M5, M8, M9, M10, M14, N5, N7, L2, L3.

## Global constraints
- Branch `sprint/s1-safety`, worktree `C:/Projects/iva-wt/S1`, base `origin/main` (S0.2 merged). Use the S0.2 seams and fakes; no test may touch the network.
- **No request-size caps** (owner decision, 2026-11-13). Keep the decompression guards and the extension allowlist.
- Allowed hosts default to `["127.0.0.1", "localhost"]`. `create_app(allowed_hosts=...)` lets tests pass `["testserver", "127.0.0.1", "localhost"]`. Update the existing tests' app fixtures accordingly.
- `run()` default host is `127.0.0.1`. Any other host logs a WARNING.

## Review focus
1. The UI served by StaticFiles from `http://127.0.0.1:8000` sends `Origin: http://127.0.0.1:8000` on POST and on the WS handshake. It must pass, as must `localhost:8000` and a mixed-case host.
2. A `multipart/form-data` POST from a foreign origin, which needs no browser preflight, must get 403 on both upload routes.
3. A config POST with one invalid field must apply nothing. No partial update.
4. An EPUB whose spine references a missing or oversized member must still load the rest, and report what was skipped.
5. A slow parse (a large PDF) must not stall the WebSocket. Probe loop latency during a threaded parse.

---

### Task 1: Host allowlist and Origin check (H4, N7)

**Files:**
- Create: `src/voice_agent/server/security.py`
- Modify: `src/voice_agent/server/app.py`, `src/voice_agent/server/websocket.py`
- Test: `tests/test_security.py`

**Interfaces:**
- `def origin_allowed(origin: str | None, host: str | None) -> bool`:
  - `None` origin (no header) returns True;
  - `"null"` returns False;
  - otherwise compare `urlsplit(origin)` host:port with the `Host` header, case-insensitively, normalising default ports (80 for http, 443 for https).
- `class OriginCheckMiddleware`: pure ASGI. For HTTP methods other than GET, HEAD and OPTIONS, it returns 403 with JSON `{"detail": "cross-origin request rejected"}` when `origin_allowed` is False.
- `websocket_endpoint` checks `origin_allowed(ws.headers.get("origin"), ws.headers.get("host"))` before `accept`. If it fails, `await websocket.close(code=1008)` and return.
- `create_app(..., allowed_hosts: list[str] | None = None)` adds `TrustedHostMiddleware(allowed_hosts=allowed_hosts or ["127.0.0.1", "localhost"])` and `OriginCheckMiddleware`.
- `run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False)`.

- [ ] **Step 1: Write the failing tests.** Use TestClient with `allowed_hosts=["testserver", "127.0.0.1", "localhost"]`.
  - `test_origin_allowed_cases`: table-driven. `(None, "x")` gives True. `("null", "127.0.0.1:8000")` gives False. `("http://127.0.0.1:8000", "127.0.0.1:8000")` gives True. `("http://LOCALHOST:8000", "localhost:8000")` gives True. `("http://evil.com", "127.0.0.1:8000")` gives False. `("http://127.0.0.1", "127.0.0.1:80")` gives True.
  - `test_cross_origin_json_post_rejected`: POST `/api/config` with `Origin: http://evil.com` returns 403, and the config is unchanged.
  - `test_cross_origin_multipart_upload_rejected`: POST `/api/upload-document` with files and a foreign Origin returns 403.
  - `test_same_origin_post_allowed`: POST with `Origin: http://testserver` and `Host: testserver` returns 200.
  - `test_ws_foreign_origin_closed`: `client.websocket_connect("/ws", headers={"origin": "http://evil.com"})` raises `WebSocketDisconnect` with code 1008.
  - `test_bad_host_rejected`: `Host: evil.com` returns 400.
  - `test_run_defaults_to_loopback`: `inspect.signature(run).parameters["host"].default == "127.0.0.1"`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Run them again and the full suite. Expected: PASS.
- [ ] **Step 4: Commit.** `feat(server): enforce host allowlist and same-origin checks on HTTP and WS (H4, N7)`

### Task 2: validated request bodies (M8, M10)

**Files:**
- Modify: `src/voice_agent/server/schemas.py`, `src/voice_agent/server/app.py`
- Test: `tests/test_rest_validation.py`

**Interfaces:**
- `VOICE_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"`. An identifier pattern, not an enumerated set; the voice registry arrives in S4.2.
- `MicSource = Literal["browser", "host"]`, `PlaybackAction = Literal["resume", "start", "pause", "stop", "next", "prev"]`.
- `class ConfigUpdateRequest(BaseModel)`, `model_config = ConfigDict(extra="forbid")`, every field optional:
  - `base_url: AnyHttpUrl | None`;
  - `api_key: str | None` (max length 512);
  - `provider: str | None` (pattern `^[a-z0-9_-]{1,32}$`);
  - `model: str | None` (max 128);
  - `web_search_enabled: bool | None`;
  - `active_voice: str | None` (`VOICE_ID_PATTERN`);
  - `cadence: float | None` (`ge=0.1`, `le=3.0`);
  - `mic_source: MicSource | None`.
  - `endpoint` and `model_name` are accepted as aliases for `base_url` and `model`.
- `class AskRequest(BaseModel)`: `query: str`, stripped, min 1, max 2,000.
- `class PlaybackControlRequest(BaseModel)`: `action: PlaybackAction = "resume"`.
- `class TextDocumentRequest(BaseModel)`: `text: str` (min 1), `title: str = "Custom Document"` (max 200).
- `SetCadenceMessage`, `SelectVoiceMessage`, `SetMicSourceMessage` and `PlaybackControlMessage` in the WS protocol reuse the same constraints.
- Routes declare these as parameters, so FastAPI returns 422 before any handler code runs. That makes partial updates impossible. JSON routes require `Content-Type: application/json` (FastAPI already returns 422 otherwise; assert it in the test).
- `/api/upload-document`: if the parsed document has zero sentences, return 422 `{"detail": "document contains no readable text"}` and do not call `load_document`. The `narration.py:73` early return stays as a guard.

- [ ] **Step 1: Write the failing tests.**
  - `test_config_rejects_out_of_range_cadence_and_applies_nothing`: `{"cadence": 1e-9, "model": "x"}` returns 422, and `llm.get_config()["model"]` is unchanged.
  - `test_config_rejects_unknown_field`
  - `test_config_text_plain_rejected`: same JSON body with `Content-Type: text/plain` returns 422.
  - `test_playback_rejects_unknown_action`
  - `test_ask_rejects_empty`
  - `test_mic_source_enum`
  - `test_ws_cadence_out_of_range_ignored_with_error_reply` (WS sends `{"type": "error", ...}` back; see Task 4)
  - `test_empty_document_returns_422_and_keeps_old`: upload a `.txt` of whitespace; the response is 422 and `GET /api/document` still shows the default title.
- [ ] **Step 2:** Run them (FAIL), implement, run them again plus the suite (PASS).
- [ ] **Step 3: Commit.** `feat(server): validate REST bodies with Pydantic models; reject empty documents (M8, M10)`

### Task 3: bounded, ordered, honest document parsing (H5, M14, N5)

**Files:**
- Modify: `src/voice_agent/document/parser.py`, `src/voice_agent/server/app.py`
- Test: `tests/test_parser_hardening.py`

**Interfaces:**
- Constants in `parser.py`:
  - `MAX_EPUB_MEMBERS = 2000`
  - `MAX_EPUB_MEMBER_BYTES = 10 * 1024 * 1024`
  - `MAX_EPUB_TOTAL_BYTES = 100 * 1024 * 1024`
  - `MAX_PDF_PAGES = 2000`
- `class DocumentParseError(ValueError)` for files that cannot be read at all: a corrupt zip or PDF, or too many members.
- `DocumentContent` gains `skipped_parts: list[str] = field(default_factory=list)`. Each entry is a human-readable reason, e.g. `"chapter OEBPS/ch3.xhtml: exceeds 10 MB uncompressed"` or `"page 12: extract failed: <exc>"`.
- EPUB:
  - Read `META-INF/container.xml` to find the OPF.
  - Parse the OPF spine (`<itemref idref>` resolved through `<manifest><item id href>`) with regex, consistent with the current no-XML-parser approach.
  - Read chapters in spine order. Fall back to the current sorted list only when no spine is found, and add that to `skipped_parts` as a note.
  - Check `ZipInfo.file_size` before `z.read`.
- PDF: stop after `MAX_PDF_PAGES` and record the rest as skipped.
- Every skipped part is logged at WARNING with the filename.
- The `"Failed to read EPUB: ..."` text injection (`parser.py:161`) is replaced by raising `DocumentParseError`.
- `app.py`:
  - `doc = await asyncio.to_thread(DocumentParser.parse_bytes, content, filename)`;
  - `DocumentParseError` gives 422 with its message;
  - unknown extensions give 415 (allowlist `.pdf .epub .txt .md`);
  - the response JSON includes `skipped_parts`.

- [ ] **Step 1: Write the failing tests.** Build fixtures in memory with `zipfile` and `pypdf.PdfWriter`.
  - `test_epub_spine_order`: the spine lists ch2 before ch1, so the sentences come out in ch2-then-ch1 order.
  - `test_epub_oversized_member_skipped`: patch `MAX_EPUB_MEMBER_BYTES` to 100; the big chapter is skipped and listed, and the others load.
  - `test_epub_zip_bomb_bounded`: a 50 MB-of-zeros member, patched limit; parsing finishes, tracemalloc peak stays under 20 MB, and the member is skipped.
  - `test_corrupt_epub_raises`: `DocumentParseError`, and no error text appears in the sentences.
  - `test_corrupt_pdf_upload_returns_422`
  - `test_unknown_extension_415`
  - `test_upload_parses_off_loop`: monkeypatch `DocumentParser.parse_bytes` to sleep 0.3 s. While the upload runs, a concurrent `asyncio.sleep(0.01)` probe on the loop finishes in under 100 ms. Use `httpx.AsyncClient(transport=ASGITransport(app))`.
- [ ] **Step 2:** FAIL, implement, PASS. Commit per logical change:
  - `fix(parser): read EPUB chapters in OPF spine order (N5)`
  - `fix(parser): bound EPUB/PDF reading and report skipped parts (H5, M14)`
  - `fix(server): parse uploads off the event loop with clear 415/422 errors (H5)`

### Task 4: transcribe endpoint, broadcast robustness, WS errors (M9, M5, L2, L3)

**Files:**
- Modify: `src/voice_agent/server/app.py:31-55,271-294`, `src/voice_agent/server/connection_manager.py`, `src/voice_agent/server/websocket.py`
- Test: `tests/test_transcribe_api.py` (extend), `tests/test_broadcast.py`

**Interfaces:**
- `def extract_audio(content: bytes) -> tuple[np.ndarray, int]`, returning int16 mono samples and the sample rate:
  - WAV: `sampwidth` must be 2, else raise `UnsupportedAudio`; average the channels; return `getframerate()`.
  - Raw PCM: odd length raises `InvalidAudio`; assume 16 kHz.
  - `class UnsupportedAudio(ValueError)` maps to 415; `class InvalidAudio(ValueError)` maps to 400. Both are defined in `app.py`.
- `/api/transcribe`:
  - resample to 16 kHz with `scipy.signal.resample_poly(samples, 16000, rate)` when the rate differs;
  - `transcript = await asyncio.to_thread(stt.transcribe, pcm16_bytes)` with `stt = await ensure_stt_engine(app)` (from S0.2);
  - the extension allowlist is `.wav .pcm .raw`, else 415.
- Broadcast: each `send_text` / `send_bytes` is wrapped in `asyncio.wait_for(..., timeout=2.0)`. A timeout or exception drops the socket and logs at INFO with `type(exc).__name__`.
- `broadcast_json_sync` / `broadcast_bytes_sync` log a WARNING when there is no running loop (L2).
- WS catch-all: per-message handling is wrapped. On an exception, `logger.exception(...)`, send `{"type": "error", "detail": "<short reason>"}` to that socket, and keep the session (L3). Only `WebSocketDisconnect` ends the loop.
- Validation failures in `parse_client_message` also get an `error` reply instead of being silently dropped. Their DEBUG log must not include the payload text (see M13/L13 in S3.1; do it here, since it's the same line, `websocket.py:114`).

- [ ] **Step 1: Write the failing tests.**
  - `test_transcribe_44k_wav_resampled`: the fake STT records the length it received; a 1 s 44.1 kHz WAV arrives as 16,000 samples.
  - `test_transcribe_8bit_wav_415`
  - `test_transcribe_odd_raw_400`
  - `test_stalled_socket_dropped_after_timeout`: the fake socket's `send_text` awaits forever; patch the timeout to 0.05; `broadcast_text` returns, and the socket is removed.
  - `test_sync_broadcast_without_loop_warns`: `caplog` at WARNING.
  - `test_ws_bad_message_keeps_session`: send `"{not json"`, then `{"type": "set_cadence", "cadence": 99}`, then a valid `set_voice`. Two `error` replies, then a `state_sync`. The socket stays open.
- [ ] **Step 2:** FAIL, implement, PASS. Commits:
  - `fix(server): resample and validate uploaded audio for transcription (M9)`
  - `fix(server): time-limited broadcast sends and logged drops (M5, L2)`
  - `fix(ws): keep sessions alive on bad messages with error replies (L3)`

### Task 5: interim accumulator bounds (H3, sec. 8 P1)

**Files:**
- Modify: `src/voice_agent/stt/accumulator.py`, `src/voice_agent/server/websocket.py:249-263`
- Test: `tests/test_speech_accumulator.py` (extend)

**Interfaces:**
- `MAX_UTTERANCE_S = 30.0` and `PRE_ROLL_CHUNKS = 25`, both already 25 in effect.
- `_session_buffer` is filled only while the accumulator is "armed": add `arm()` and `disarm()`. `websocket.py` calls `acc.arm()` when the state is LISTENING and `acc.disarm()` otherwise. `disarm()` clears the session and speech buffers and keeps the pre-roll.
- The session buffer is capped at 30 s of audio (drop the oldest chunks).
- `_speech_buffer` is cut at 30 s: `feed` returns the utterance when the cap is reached, even mid-speech.
- Remove the unused `_silence_duration_ms`.

- [ ] **Step 1: Write the failing tests.**
  - `test_buffers_bounded_over_an_hour`: feed 3,600 s of loud 20 ms chunks while disarmed. The session buffer stays empty and the speech buffer stays at or under 30 s.
  - `test_speech_cut_at_30s`: armed; 31 s of continuous speech returns one utterance of 30 s.
  - `test_quiet_ptt_still_returns_whole_turn`: armed; 5 s below threshold, then `flush()` returns 5 s.
- [ ] **Step 2:** FAIL, implement, PASS. Commit: `fix(stt): bound accumulator buffers and arm session capture only while listening (H3)`

### Task 6: wrap-up
- [ ] **Step 1:** Changelog line: `- S1: same-origin and host checks, validated request bodies, bounded spine-ordered parsing, robust broadcast and WS error replies (H3, H4, H5, M5, M8, M9, M10, M14, N5, N7, L2, L3).`
- [ ] **Step 2:** Rebase, run the gates (ruff, mypy, mypy `--platform linux`, pytest), push, open the PR, and confirm CI.
