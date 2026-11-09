# Task 5 Brief: HUD Client Bi-Directional Speech Synchronization (`ui/index.html`)

## Files to Create/Modify
- Modify: `ui/index.html`
- Modify: `ui/app.css`

## Instructions
1. In `ui/index.html` inside `HUDWebSocketClient`:
   - In `handleJsonMessage(msg)`:
     - Add `case 'user_speech_transcript': this.handleUserSpeechTranscript(msg); break;`
   - Implement `handleUserSpeechTranscript(msg)`:
     - Update `#interruption-query-input` with `msg.transcript`.
     - Update `#speech-ticker` with `msg.transcript`.
     - If `msg.is_final` and `msg.transcript.trim().length > 0`:
       - Show Prompt Envelope with state `LISTENING` and question `msg.transcript`.
2. In `startDirectVoiceBargeIn()`:
   - Ensure `wsClient.startBrowserMicCapture()` is called so microphone stream is active.
   - When barge-in is triggered, set button text to `🛑 STOP & TRANSCRIBE` while recording.
   - When clicked again to stop, send `{ type: 'voice_end' }` over WebSocket to immediately trigger transcription on the server.
3. Keep Web Speech API as an optional client preview, but make server-side STT via WebSocket the primary deterministic engine.
4. Run static server tests: `.venv\Scripts\python -m pytest tests/test_server_static.py -v`.
5. Run full test suite: `.venv\Scripts\python -m pytest tests/`.
6. Commit:
   ```bash
   git add ui/index.html ui/app.css
   git commit -m "feat(ui): synchronize HUD with server-side STT transcripts and voice_end event"
   ```
7. Report back with git commit hash.
