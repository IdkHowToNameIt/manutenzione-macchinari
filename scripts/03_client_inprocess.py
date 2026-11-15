"""S3 - Prova del contratto HTTP con client in-process.

Esegue le quattro richieste del CSV assegnato attraverso l'applicazione, senza
rete e senza container: verifica stati attesi, identita' della versione, request
ID e telemetria. Le prove via HTTP e via container sono separate.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from manutenzione.config import load_project_config
from manutenzione.contracts import load_feature_schema
from manutenzione.data_loader import load_monitoring_requests
from manutenzione.serving.app import create_app


def main() -> None:
    config = load_project_config()
    feature_schema = load_feature_schema(config["paths"]["feature_schema"])
    monitoring_requests = load_monitoring_requests(config["paths"]["requests_csv"])
    application = create_app(config=config, feature_schema=feature_schema)

    responses: list[dict] = []
    with TestClient(application) as client:
        health = client.get("/health")
        metrics_before = client.get("/metrics").json()

        for monitoring_request in monitoring_requests:
            response = client.post("/predict", json=monitoring_request.payload)
            body = response.json()
            responses.append(
                {
                    "request_id_csv": monitoring_request.request_id,
                    "model_version_attesa": monitoring_request.model_version,
                    "expected_status": monitoring_request.expected_status,
                    "status_code": response.status_code,
                    "request_id_risposta": response.headers.get("X-Request-ID"),
                    "model_version_risposta": response.headers.get("X-Model-Version"),
                    "body": body,
                }
            )

        metrics_after = client.get("/metrics").json()

    health_body = health.json()
    expected_status_matches = all(
        item["status_code"] == item["expected_status"] for item in responses
    )
    version_identity_ok = all(
        item["status_code"] != 200
        or (
            item["body"].get("model_version") == health_body["model_version"]
            and item["body"].get("run_id") == health_body["run_id"]
            and item["model_version_risposta"] == health_body["model_version"]
        )
        for item in responses
    )

    # Ogni risposta deve avere una riga di log con lo stesso request ID.
    log_path = Path(config["paths"]["logs_dir"]) / "serving.jsonl"
    logged_request_ids: set[str] = set()
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                logged_request_ids.add(json.loads(line).get("request_id"))
    correlation_ok = all(
        item["request_id_risposta"] in logged_request_ids for item in responses
    )

    evidence = {
        "prova": "client in-process (FastAPI TestClient, nessuna rete)",
        "health": health_body,
        "risposte": responses,
        "metrics_prima": metrics_before,
        "metrics_dopo": metrics_after,
        "controlli": {
            "stati_coerenti_con_il_csv": expected_status_matches,
            "identita_versione_e_run": version_identity_ok,
            "request_id_presente_nei_log": correlation_ok,
            "health_ha_versione_e_run_id": bool(
                health_body.get("model_version") and health_body.get("run_id")
            ),
        },
    }

    evidence_dir = Path(config["paths"]["evidence_dir"]) / "requests"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / "client_inprocess.json"
    with evidence_path.open("w", encoding="utf-8") as evidence_file:
        json.dump(evidence, evidence_file, indent=2, ensure_ascii=False)

    print(f"Health: versione {health_body['model_version']}, run {health_body['run_id']}")
    for item in responses:
        extra = ""
        if item["status_code"] == 200:
            extra = (
                f", prediction={item['body']['prediction']}, "
                f"prob={item['body']['probability_target_1']:.4f}"
            )
        print(
            f"  {item['request_id_csv']} ({item['model_version_attesa']}): "
            f"atteso {item['expected_status']}, ricevuto {item['status_code']}{extra}"
        )
    print(f"Metriche dopo: {json.dumps(metrics_after, ensure_ascii=False)}")
    print(f"Controlli: {json.dumps(evidence['controlli'], ensure_ascii=False)}")
    print(f"Evidenza scritta in: {evidence_path}")


if __name__ == "__main__":
    main()
