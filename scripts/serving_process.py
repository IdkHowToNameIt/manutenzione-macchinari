"""Avvio e arresto di un'istanza locale del servizio per probe e drill.

Il processo Uvicorn gira in una sottodirectory controllata: al termine viene
sempre arrestato e i suoi log restano nella cartella logs/.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import httpx


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        return int(probe_socket.getsockname()[1])


def _wait_for_health(
    client: httpx.Client, base_url: str, process: subprocess.Popen, timeout_seconds: float = 40.0
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Il servizio si e' chiuso con codice {process.returncode}")
        try:
            response = client.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError as error:
            last_error = error
        time.sleep(0.25)
    raise RuntimeError(f"Il servizio non e' diventato pronto: {last_error}")


@contextmanager
def serving_instance(
    project_root: str | Path,
    log_path: str | Path,
    forced_version: str | None = None,
) -> Iterator[tuple[httpx.Client, str, dict]]:
    """Avvia il servizio su una porta libera e restituisce client, URL e health."""
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment.update(
        {
            "MANUTENZIONE_HOST": "127.0.0.1",
            "MANUTENZIONE_PORT": str(port),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if forced_version:
        environment["MANUTENZIONE_FORCED_VERSION"] = forced_version
    else:
        environment.pop("MANUTENZIONE_FORCED_VERSION", None)

    log_file = Path(log_path).open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "manutenzione.serving.app"],
        cwd=str(project_root),
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        with httpx.Client(timeout=5.0) as client:
            health = _wait_for_health(client, base_url, process)
            yield client, base_url, health
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log_file.close()
