from __future__ import annotations

import json
import os
import socket
from urllib.request import urlopen

import uvicorn

try:
    from .api import app
except ImportError:  # pragma: no cover
    from src.api import app


def api_is_running(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/openapi.json", timeout=1) as response:
            return json.loads(response.read()).get("info", {}).get("title") == app.title
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", "8001"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        port_in_use = listener.connect_ex(("127.0.0.1", port)) == 0

    if port_in_use:
        if api_is_running(port):
            print(f"API is already running: http://localhost:{port}/docs", flush=True)
            print("Not starting a second copy.", flush=True)
            raise SystemExit(0)
        raise SystemExit(
            f"Port {port} is already in use by another process. "
            f"Choose another port, for example: $env:APP_PORT='8002'"
        )

    print(f"\nOpen API docs in your browser: http://localhost:{port}/docs", flush=True)
    print(f"Or open the app address: http://localhost:{port}/ (redirects to /docs)", flush=True)
    print("Use localhost in your browser; 0.0.0.0 is only the server bind address.\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
