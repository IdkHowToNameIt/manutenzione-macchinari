"""S5 - Attivazione di v2 e ripristino di v1 con evidenze osservabili.

Sequenza: predizione con v1, attivazione di v2, predizione con v2, rollback a
v1, predizione con v1, richiesta incompleta (422). Il ripristino finale lascia
il servizio sulla baseline, come richiesto dalla consegna, e dimostra che
l'artefatto realmente servito e' cambiato.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from manutenzione.config import load_project_config
from manutenzione.contracts import load_feature_schema
from manutenzione.data_loader import load_monitoring_requests
from manutenzione.registry import ModelRegistry

from serving_process import serving_instance

ADMIN_TOKEN = os.environ.get("MANUTENZIONE_ADMIN_TOKEN", "local-admin-token")
ADMIN_HEADERS = {"X-Admin-Token": ADMIN_TOKEN}


def main() -> None:
    config = load_project_config()
    feature_schema = load_feature_schema(config["paths"]["feature_schema"])
    registry = ModelRegistry(config["paths"]["models_dir"], feature_schema)
    requests_by_id = {
        request.request_id: request
        for request in load_monitoring_requests(config["paths"]["requests_csv"])
    }
    evidence_dir = Path(config["paths"]["evidence_dir"]) / "transitions"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    gate_report_path = Path(config["paths"]["evidence_dir"]) / "gate" / "gate_report.json"
    gate_report = json.loads(gate_report_path.read_text(encoding="utf-8"))
    gate_decision = gate_report["decisione"]
    execution_mode = (
        "attivazione_documentata_dopo_gate"
        if gate_decision == "accettata"
        else "drill_locale_dichiarato_dopo_blocco_gate"
    )

    # Il drill parte sempre dalla baseline attiva.
    pointer_before = registry.read_pointer()
    if pointer_before.get("active_version") != "v1":
        registry.activate(
            "v1",
            actor="Andrea",
            action="drill_preparation",
            reason="preparazione del drill: la baseline deve essere la versione di partenza",
        )
        pointer_before = registry.read_pointer()

    metadata_v1 = registry.read_metadata("v1")
    metadata_v2 = registry.read_metadata("v2")
    log_path = Path(config["paths"]["logs_dir"]) / "drill_server.log"

    steps: list[dict] = []
    with serving_instance(config["project_root"], log_path) as (client, base_url, health):
        health_initial = health

        def call_predict(request_id: str, label: str) -> None:
            request = requests_by_id[request_id]
            started = perf_counter()
            response = client.post(f"{base_url}/predict", json=request.payload)
            duration_ms = (perf_counter() - started) * 1000.0
            steps.append(
                {
                    "passo": label,
                    "request_id_csv": request_id,
                    "stato_atteso": request.expected_status,
                    "status_code": response.status_code,
                    "request_id": response.headers.get("X-Request-ID"),
                    "model_version_header": response.headers.get("X-Model-Version"),
                    "duration_ms": round(duration_ms, 3),
                    "body": response.json(),
                }
            )

        def call_admin(path: str, body: dict, label: str) -> None:
            started = perf_counter()
            response = client.post(f"{base_url}{path}", json=body, headers=ADMIN_HEADERS)
            duration_ms = (perf_counter() - started) * 1000.0
            steps.append(
                {
                    "passo": label,
                    "endpoint": path,
                    "status_code": response.status_code,
                    "request_id": response.headers.get("X-Request-ID"),
                    "duration_ms": round(duration_ms, 3),
                    "body": response.json(),
                }
            )

        call_predict("req_01", "predizione con v1")
        call_admin(
            "/admin/activate",
            {
                "version": "v2",
                "actor": "Andrea",
                "reason": f"gate {gate_decision}: {execution_mode}",
            },
            "attivazione v2",
        )
        call_predict("req_02", "predizione con v2")
        call_admin(
            "/admin/rollback",
            {
                "actor": "Andrea",
                "reason": "ripristino finale richiesto dalla consegna (stato finale su baseline)",
            },
            "rollback a v1",
        )
        call_predict("req_03", "predizione dopo il ripristino")
        call_predict("req_04", "richiesta incompleta")

        health_final = client.get(f"{base_url}/health").json()
        versions_final = client.get(f"{base_url}/versions").json()
        metrics_final = client.get(f"{base_url}/metrics").json()

    pointer_final = registry.read_pointer()
    prediction_steps = [step for step in steps if step.get("request_id_csv")]
    error_count = sum(1 for step in prediction_steps if step["status_code"] >= 400)
    drill_error_rate = error_count / len(prediction_steps)

    checks = {
        "stati_coerenti_con_il_csv": all(
            step["status_code"] == step["stato_atteso"] for step in prediction_steps
        ),
        "v1_iniziale_servita": health_initial["model_version"] == "v1"
        and health_initial["artifact_fingerprint"] == metadata_v1["artifact_fingerprint"],
        "attivazione_ha_caricato_v2": steps[1]["status_code"] == 200
        and steps[1]["body"]["served_version"] == "v2"
        and steps[1]["body"]["run_id"] == metadata_v2["run_id"],
        "predizione_v2_con_run_v2": steps[2]["body"].get("model_version") == "v2"
        and steps[2]["body"].get("run_id") == metadata_v2["run_id"],
        "rollback_ha_caricato_v1": steps[3]["status_code"] == 200
        and steps[3]["body"]["served_version"] == "v1"
        and steps[3]["body"]["run_id"] == metadata_v1["run_id"],
        "predizione_dopo_ripristino_con_v1": steps[4]["body"].get("model_version") == "v1"
        and steps[4]["body"].get("run_id") == metadata_v1["run_id"],
        "richiesta_incompleta_respinta": prediction_steps[-1]["status_code"] == 422,
        "health_finale_su_baseline": health_final["model_version"] == "v1"
        and health_final["run_id"] == metadata_v1["run_id"]
        and health_final["artifact_fingerprint"] == metadata_v1["artifact_fingerprint"],
        "stato_finale_v1_con_precedente_v2": pointer_final["active_version"] == "v1"
        and pointer_final["previous_version"] == "v2",
        "error_rate_del_drill_0_25": abs(drill_error_rate - 0.25) < 1e-9,
    }

    # Correlazione risposta-log: ogni request ID delle predizioni deve essere nel log.
    serving_log = Path(config["paths"]["logs_dir"]) / "serving.jsonl"
    all_events = [
        json.loads(line)
        for line in serving_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    drill_request_ids = {step["request_id"] for step in prediction_steps}
    drill_events = [event for event in all_events if event.get("request_id") in drill_request_ids]
    checks["request_id_correlati_ai_log"] = len(drill_events) == len(drill_request_ids)

    drill_report = {
        "eseguito_il": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decisione_gate": gate_decision,
        "modalita": execution_mode,
        "nota_ripristino": (
            "Il gate ha accettato v2; l'attivazione documentata e' stata seguita da un "
            "ripristino esplicito a v1 per lasciare lo stato finale sulla baseline richiesta."
            if gate_decision == "accettata"
            else (
                "La candidata non ha superato il gate: il cambio versione e' stato eseguito "
                "solo come drill locale dichiarato e lo stato finale e' la baseline."
            )
        ),
        "health_iniziale": health_initial,
        "passi": steps,
        "health_finale": health_final,
        "versions_finali": versions_final,
        "metrics_finali_drill": metrics_final,
        "pointer_finale": pointer_final,
        "riepilogo": {
            "predizioni": len(prediction_steps),
            "errori_http": error_count,
            "error_rate_drill": drill_error_rate,
            "controlli": checks,
            "tutti_i_controlli_superati": all(checks.values()),
        },
    }

    with (evidence_dir / "drill_responses.json").open("w", encoding="utf-8") as report_file:
        json.dump(drill_report, report_file, indent=2, ensure_ascii=False)

    transitions_source = Path(config["paths"]["models_dir"]) / "transitions.jsonl"
    (evidence_dir / "transitions.jsonl").write_text(
        transitions_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (evidence_dir / "serving_drill.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in drill_events) + "\n",
        encoding="utf-8",
    )

    print(f"Modalita': {execution_mode}")
    for step in steps:
        print(
            f"  {step['passo']}: stato {step['status_code']}, "
            f"versione servita {step['body'].get('served_version') or step['body'].get('model_version')}"
        )
    print(f"Error rate del drill: {drill_error_rate} (1 richiesta su 4 respinta)")
    print(f"Controlli: {json.dumps(checks, ensure_ascii=False)}")
    print(f"Evidenze: {evidence_dir}")

    if not all(checks.values()):
        raise SystemExit("Drill incompleto: almeno un controllo non e' superato")


if __name__ == "__main__":
    main()
