"""Start both actual server processes, probe local HTTP endpoints, and clean up.

No LLM or embedding request is made. Child processes use hidden windows on Windows.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    """Ask the OS for an unused localhost port for a short-lived test server."""
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def wait_ready(url: str, process: subprocess.Popen, seconds: int = 45) -> httpx.Response:
    """Poll a local readiness URL with short requests and a bounded startup deadline."""
    deadline = time.monotonic() + seconds
    with httpx.Client(trust_env=False, timeout=2) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("A server exited before becoming ready.")
            try:
                response = client.get(url)
                if response.status_code == 200:
                    return response
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
    raise RuntimeError(f"Server readiness timed out: {url}")


def main() -> None:
    """Verify real FastAPI and Streamlit startup, then terminate only owned processes."""
    api_port, ui_port = free_port(), free_port()
    while ui_port == api_port:
        ui_port = free_port()
    api_url = f"http://127.0.0.1:{api_port}"
    ui_url = f"http://127.0.0.1:{ui_port}"
    environment = os.environ.copy()
    environment["BACKEND_URL"] = api_url
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    processes = []
    try:
        commands = [
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
                "--no-access-log",
            ],
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "frontend/app.py",
                "--server.headless=true",
                "--server.address=127.0.0.1",
                f"--server.port={ui_port}",
                "--browser.gatherUsageStats=false",
            ],
        ]
        for command in commands:
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    creationflags=flags,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        health = wait_ready(api_url + "/health", processes[0]).json()
        streamlit_health = wait_ready(ui_url + "/_stcore/health", processes[1]).text
        with httpx.Client(trust_env=False, timeout=5) as client:
            statuses = {
                path: client.get(api_url + path).status_code
                for path in ("/docs", "/redoc", "/openapi.json", "/api/config")
            }
            ui_status = client.get(ui_url).status_code
        assert health["application"] == "IntelliChat"
        assert all(status == 200 for status in statuses.values())
        assert streamlit_health == "ok" and ui_status == 200
        print(
            json.dumps(
                {
                    "fastapi_health": health,
                    "api_http_statuses": statuses,
                    "streamlit_health": streamlit_health,
                    "streamlit_http_status": ui_status,
                    "note": (
                        "Real process HTTP smoke check; UI interactions are covered by AppTest."
                    ),
                },
                indent=2,
            )
        )
    finally:
        for process in reversed(processes):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
