"""Prova del servizio dentro un container Docker.

Avvia l'immagine costruita da docker/Dockerfile, esegue la sequenza v1 → v2 →
v1 con la richiesta incompleta, raccoglie health, metriche e log del container e
poi lo rimuove. Se Docker non e' disponibile, registra esplicitamente
"non eseguito" invece di simulare la prova.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx

from manutenzione.config import load_project_config
from manutenzione.data_loader import load_monitoring_requests

from serving_process import free_local_port

IMAGE_NAME = "manutenzione-predittiva-api:1.0.0"
CONTAINER_NAME = "manutenzione-prova"
ADMIN_HEADERS = {"X-Admin-Token": os.environ.get("MANUTENZIONE_ADMIN_TOKEN", "local-admin-token")}


def _run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def _docker_available() -> tuple[bool, str]:
    try:
        result = _run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return False, f"Docker non disponibile: {error}"
    if result.returncode != 0:
        return False, result.stderr.strip() or "Docker non disponibile"
    return True, result.stdout.strip()


def _wait_for_health(base_url: str, container_name: str, timeout_seconds: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError as error:
            last_error = error
        time.sleep(0.5)
    logs = _run(["docker", "logs", container_name]).stdout
    raise RuntimeError(f"Il container non e' pronto: {last_error}\n{logs[-2000:]}")


def main() -> None:
    config = load_project_config()
    evidence_dir = Path(config["paths"]["evidence_dir"]) / "serving"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict = {"prova": "container Docker", "immagine": IMAGE_NAME}

    available, docker_version = _docker_available()
    evidence["docker_version"] = docker_version
    if not available:
        evidence["esito"] = "non eseguito"
        _write_evidence(evidence, evidence_dir)
        print(f"Prova container non eseguita: {docker_version}")
        return

    requests_by_id = {
        request.request_id: request
        for request in load_monitoring_requests(config["paths"]["requests_csv"])
    }
    host_port = free_local_port()
    base_url = f"http://127.0.0.1:{host_port}"

    _run(["docker", "rm", "-f", CONTAINER_NAME])
    started = _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"{host_port}:8000",
            "-e",
            "MANUTENZIONE_ADMIN_TOKEN=local-admin-token",
            IMAGE_NAME,
        ]
    )
    if started.returncode != 0:
        evidence["esito"] = "fallito"
        evidence["errore"] = started.stderr.strip()
        _write_evidence(evidence, evidence_dir)
        raise SystemExit(f"Avvio container fallito: {started.stderr.strip()}")

    container_id = started.stdout.strip()
    evidence["container_id"] = container_id
    steps: list[dict] = []

    try:
        health = _wait_for_health(base_url, CONTAINER_NAME)
        evidence["health"] = health
        evidence["schema"] = httpx.get(f"{base_url}/schema", timeout=5.0).json()

        def call_predict(request_id: str, label: str) -> None:
            request = requests_by_id[request_id]
            measured = perf_counter()
            response = httpx.post(f"{base_url}/predict", json=request.payload, timeout=10.0)
            steps.append(
                {
                    "passo": label,
                    "request_id_csv": request_id,
                    "status_code": response.status_code,
                    "duration_ms": round((perf_counter() - measured) * 1000.0, 3),
                    "model_version_header": response.headers.get("X-Model-Version"),
                    "body": response.json(),
                }
            )

        def call_admin(path: str, body: dict, label: str) -> None:
            measured = perf_counter()
            response = httpx.post(f"{base_url}{path}", json=body, headers=ADMIN_HEADERS, timeout=10.0)
            steps.append(
                {
                    "passo": label,
                    "endpoint": path,
                    "status_code": response.status_code,
                    "duration_ms": round((perf_counter() - measured) * 1000.0, 3),
                    "body": response.json(),
                }
            )

        call_predict("req_01", "predizione con v1")
        call_admin("/admin/activate", {"version": "v2", "actor": "Andrea", "reason": "prova container"}, "attivazione v2")
        call_predict("req_02", "predizione con v2")
        call_admin("/admin/rollback", {"actor": "Andrea", "reason": "fine prova container"}, "rollback a v1")
        call_predict("req_03", "predizione dopo il ripristino")
        call_predict("req_04", "richiesta incompleta")
        evidence["metrics"] = httpx.get(f"{base_url}/metrics", timeout=5.0).json()
        evidence["health_finale"] = httpx.get(f"{base_url}/health", timeout=5.0).json()
        evidence["passi"] = steps
        evidence["esito"] = "eseguito"
        evidence["controlli"] = {
            "health_con_versione_e_run": bool(health.get("model_version") and health.get("run_id")),
            "stati_coerenti_con_il_csv": all(
                step["status_code"] == requests_by_id[step["request_id_csv"]].expected_status
                for step in steps
                if step.get("request_id_csv")
            ),
            "attivazione_v2_nel_container": steps[1]["body"].get("served_version") == "v2",
            "rollback_v1_nel_container": steps[3]["body"].get("served_version") == "v1",
            "error_rate_0_25": abs(evidence["metrics"]["error_rate"] - 0.25) < 1e-9,
        }
    except Exception as error:  # registriamo comunque l'esito della prova
        evidence["esito"] = "fallito"
        evidence["errore"] = str(error)
        raise
    finally:
        container_logs = _run(["docker", "logs", CONTAINER_NAME], timeout=60)
        log_path = Path(config["paths"]["logs_dir"]) / "container_prova.log"
        log_path.write_text(
            (container_logs.stdout or "") + (container_logs.stderr or ""), encoding="utf-8"
        )
        evidence["container_log"] = Path(log_path).relative_to(
            Path(config["project_root"])
        ).as_posix()
        _run(["docker", "rm", "-f", CONTAINER_NAME], timeout=60)
        evidence["container_rimosso"] = True
        evidence["concluso_il"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_evidence(evidence, evidence_dir)
        print(f"Esito prova container: {evidence.get('esito')}")
        for step in evidence.get("passi", []):
            served = step["body"].get("served_version") or step["body"].get("model_version") or "-"
            print(f"  {step['passo']}: {step['status_code']} (versione {served})")
        if evidence.get("controlli"):
            print(f"  controlli: {json.dumps(evidence['controlli'], ensure_ascii=False)}")


def _write_evidence(evidence: dict, evidence_dir: Path) -> None:
    with (evidence_dir / "container_prova.json").open("w", encoding="utf-8") as evidence_file:
        json.dump(evidence, evidence_file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
