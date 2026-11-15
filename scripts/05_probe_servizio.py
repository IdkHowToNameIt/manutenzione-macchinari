"""S4 - Probe locale del servizio su un'istanza HTTP reale.

Per ogni versione avvia un processo Uvicorn separato con la versione forzata,
esegue 3 richieste di warm-up e 30 misure con lo stesso payload valido
assegnato, e scrive p50/p95, error rate e fingerprint dell'artefatto misurato.
La versione attiva nel registro non viene modificata.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from manutenzione.config import load_project_config
from manutenzione.contracts import load_feature_schema
from manutenzione.data_loader import load_monitoring_requests
from manutenzione.registry import ModelRegistry

from serving_process import serving_instance

WARMUP_REQUESTS = 3
MEASURED_REQUESTS = 30


def run_probe(version: str, payload: dict, config: dict, models_dir: Path) -> dict:
    log_path = Path(config["paths"]["logs_dir"]) / f"probe_server_{version}.log"
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with serving_instance(config["project_root"], log_path, forced_version=version) as (
        client,
        base_url,
        _initial_health,
    ):
        for _ in range(WARMUP_REQUESTS):
            warmup_response = client.post(f"{base_url}/predict", json=payload)
            warmup_response.raise_for_status()

        latencies_ms: list[float] = []
        statuses: list[int] = []
        for _ in range(MEASURED_REQUESTS):
            started = perf_counter()
            response = client.post(f"{base_url}/predict", json=payload)
            latencies_ms.append((perf_counter() - started) * 1000.0)
            statuses.append(response.status_code)

        server_metrics = client.get(f"{base_url}/metrics").json()
        final_health = client.get(f"{base_url}/health").json()

    sorted_latencies = sorted(latencies_ms)
    p50_ms = sorted_latencies[int(round(0.50 * (len(sorted_latencies) - 1)))]
    p95_ms = sorted_latencies[int(round(0.95 * (len(sorted_latencies) - 1)))]
    errors = sum(1 for status in statuses if status >= 400)

    metadata = json.loads(
        (models_dir / version / "model_metadata.json").read_text(encoding="utf-8")
    )
    attributable = (
        final_health["artifact_fingerprint"] == metadata["artifact_fingerprint"]
        and final_health["run_id"] == metadata["run_id"]
        and final_health["model_version"] == version
    )

    return {
        "model_version": version,
        "probe": {
            "base_url": base_url,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "warmup_requests": WARMUP_REQUESTS,
            "measured_requests": MEASURED_REQUESTS,
            "payload_note": "payload valido assegnato dal CSV delle richieste",
            "latencies_ms": [round(value, 3) for value in latencies_ms],
            "p50_ms": round(p50_ms, 3),
            "p95_ms": round(p95_ms, 3),
            "status_codes": {str(code): statuses.count(code) for code in sorted(set(statuses))},
            "errors": errors,
            "error_rate": round(errors / MEASURED_REQUESTS, 6),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "health": final_health,
        "server_metrics": server_metrics,
        "artifact_fingerprint_atteso": metadata["artifact_fingerprint"],
        "run_id_atteso": metadata["run_id"],
        "probe_attribuita_all_artefatto": attributable,
        "server_log": Path(log_path).relative_to(Path(config["project_root"])).as_posix(),
    }


def main() -> None:
    config = load_project_config()
    feature_schema = load_feature_schema(config["paths"]["feature_schema"])
    models_dir = Path(config["paths"]["models_dir"])
    registry = ModelRegistry(models_dir, feature_schema)
    monitoring_requests = load_monitoring_requests(config["paths"]["requests_csv"])

    evidence_dir = Path(config["paths"]["evidence_dir"]) / "serving"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    for version in ("v1", "v2"):
        # Prima del probe verifichiamo che l'artefatto corrisponda ai metadati.
        registry.load_version(version)
        payload = next(
            request.payload
            for request in monitoring_requests
            if request.model_version == version and request.expected_status == 200
        )
        result = run_probe(version, payload, config, models_dir)
        evidence_path = evidence_dir / f"probe_{version}.json"
        with evidence_path.open("w", encoding="utf-8") as evidence_file:
            json.dump(result, evidence_file, indent=2, ensure_ascii=False)

        print(
            f"{version}: p50={result['probe']['p50_ms']:.2f} ms, "
            f"p95={result['probe']['p95_ms']:.2f} ms, "
            f"error_rate={result['probe']['error_rate']}, "
            f"attribuita={result['probe_attribuita_all_artefatto']}"
        )
        print(f"  evidenza: {evidence_path}")


if __name__ == "__main__":
    main()
