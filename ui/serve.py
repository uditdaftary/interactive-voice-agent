"""Local unified server launcher for Interactive Voice Agent HUD and WebSocket streaming."""
import sys
from pathlib import Path

import uvicorn

# Add project src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))


def run_server(port: int = 8000):
    url = f"http://localhost:{port}/"
    print("\n=======================================================")
    print(" Voice Agent Unified Server & HUD running at:")
    print(f" {url}")
    print(f" WebSocket streaming at: ws://localhost:{port}/ws")
    print(" Press Ctrl+C to stop.")
    print("=======================================================\n")
    uvicorn.run("voice_agent.server.app:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    target_port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        target_port = int(sys.argv[1])
    run_server(target_port)
