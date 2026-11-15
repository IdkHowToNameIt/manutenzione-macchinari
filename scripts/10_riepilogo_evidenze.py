"""Riepilogo delle evidenze raccolte dalle fasi S1-S5.

Legge i file prodotti dagli altri script e genera una tabella unica per la
relazione, senza ricalcolare nulla: ogni numero resta collegato alla sua
evidenza originale.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from manutenzione.config import load_project_config


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def main() -> None:
    config = load_project_config()
    evidence_dir = Path(config["paths"]["evidence_dir"])

    profile = _read_json(evidence_dir / "data" / "profilo_dati.json") or {}
    training = {
        version: _read_json(evidence_dir / "training" / f"{version}.json") or {}
        for version in ("v1", "v2")
    }
    probes = {
        version: _read_json(evidence_dir / "serving" / f"probe_{version}.json") or {}
        for version in ("v1", "v2")
    }
    gate = _read_json(evidence_dir / "gate" / "gate_report.json") or {}
    drill = _read_json(evidence_dir / "transitions" / "drill_responses.json") or {}
    shap_summary = _read_json(evidence_dir / "explanations" / "shap_summary.json") or {}
    drift = _read_json(evidence_dir / "monitoring" / "drift_report.json") or {}
    pointer = _read_json(Path(config["paths"]["models_dir"]) / "active_version.json") or {}
    shap_ranking = _read_csv(evidence_dir / "explanations" / "shap_importance_v2.csv")

    lines = [
        "# Riepilogo delle evidenze",
        "",
        f"Generato il {datetime.now(timezone.utc).isoformat(timespec='seconds')} da "
        "`scripts/10_riepilogo_evidenze.py`.",
        "",
        "## Dati (S1)",
        "",
        f"- Dataset: `{profile.get('dataset_path')}`",
        f"- Fingerprint SHA-256: `{profile.get('dataset_fingerprint')}`",
        f"- Righe: {profile.get('row_count')}, feature: {profile.get('feature_count')}",
        f"- Schema feature: versione {profile.get('feature_schema_version')}, "
        f"fingerprint `{profile.get('feature_schema_fingerprint')}`",
    ]
    for split_name, counts in (profile.get("class_counts") or {}).items():
        lines.append(
            f"- Split {split_name}: target 0 = {counts['0']}, target 1 = {counts['1']}"
        )

    lines.extend(
        [
            "",
            "## Versioni del modello (S2, S4)",
            "",
            "| Versione | Modello | Run MLflow | Soglia | Accuracy | Recall (classe 1) | "
            "Precision (classe 1) | ROC AUC | FP | FN | Fingerprint artefatto |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for version, evidence in training.items():
        metrics = evidence.get("evidence", {})
        matrix = metrics.get("confusion_matrix", {})
        decision_threshold = evidence.get("decision_threshold")
        if decision_threshold is None:
            decision_threshold = metrics.get("decision_threshold")
        lines.append(
            f"| {version} | {evidence.get('model_family')} | `{evidence.get('mlflow_run_id')}` | "
            f"{decision_threshold} | {metrics.get('accuracy')} | "
            f"{metrics.get('recall_target_1')} | {metrics.get('precision_target_1')} | "
            f"{metrics.get('roc_auc')} | {matrix.get('false_positive')} | "
            f"{matrix.get('false_negative')} | `{evidence.get('artifact_fingerprint')}` |"
        )

    lines.extend(
        [
            "",
            "## Probe del servizio (S4)",
            "",
            "| Versione | p50 (ms) | p95 (ms) | Error rate | Probe attribuita | Ambiente |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for version, probe in probes.items():
        measurement = probe.get("probe", {})
        environment = probe.get("environment", {})
        lines.append(
            f"| {version} | {measurement.get('p50_ms')} | {measurement.get('p95_ms')} | "
            f"{measurement.get('error_rate')} | {probe.get('probe_attribuita_all_artefatto')} | "
            f"Python {environment.get('python')}, {environment.get('cpu_count')} CPU |"
        )

    lines.extend(
        [
            "",
            "## Gate di rilascio (S4)",
            "",
            f"Decisione: **{gate.get('decisione', 'non disponibile')}** "
            f"({gate.get('decisa_il', 'n/d')})",
            "",
            "| ID | Requisito | Misura | Soglia | Esito |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for requirement in gate.get("requisiti", []):
        measured = requirement.get("misura")
        if isinstance(measured, dict):
            measured = ", ".join(f"{key}={value}" for key, value in measured.items())
        lines.append(
            f"| {requirement.get('id')} | {requirement.get('requisito')} | {measured} | "
            f"{requirement.get('soglia')} | {requirement.get('esito')} |"
        )

    lines.extend(
        [
            "",
            "## Attivazione e ripristino (S5)",
            "",
            f"Modalita': {drill.get('modalita', 'n/d')}",
            "",
            "| Passo | Stato HTTP | Versione servita | Run ID |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for step in drill.get("passi", []):
        body = step.get("body", {})
        served = body.get("served_version") or body.get("model_version") or "-"
        run_id = body.get("run_id") or "-"
        lines.append(
            f"| {step.get('passo')} | {step.get('status_code')} | {served} | `{run_id}` |"
        )
    summary = drill.get("riepilogo", {})
    lines.extend(
        [
            "",
            f"Error rate del drill: {summary.get('error_rate_drill')} "
            f"({summary.get('errori_http')} errori su {summary.get('predizioni')} richieste).",
            f"Tutti i controlli superati: {summary.get('tutti_i_controlli_superati')}.",
        ]
    )

    if shap_ranking:
        lines.extend(
            [
                "",
                "## Spiegazioni SHAP della candidata v2",
                "",
                "| Feature | Importanza media (abs SHAP) |",
                "| --- | ---: |",
            ]
        )
        for row in shap_ranking[:5]:
            lines.append(f"| {row['feature']} | {row['mean_abs_shap']} |")

    if drift.get("richieste"):
        lines.extend(
            [
                "",
                "## Controllo di drift illustrativo",
                "",
                "| Richiesta | Feature oltre soglia | z-score massimo |",
                "| --- | --- | ---: |",
            ]
        )
        for request in drift["richieste"]:
            flagged = ", ".join(item["feature"] for item in request["feature_oltre_soglia"]) or "nessuna"
            lines.append(
                f"| {request['request_id']} | {flagged} | {request['max_abs_z_score']} |"
            )

    lines.extend(
        [
            "",
            "## Stato finale",
            "",
            f"- Versione attiva: {pointer.get('active_version')}",
            f"- Versione precedente: {pointer.get('previous_version')}",
            f"- Transizioni registrate: {len(pointer.get('history', []))}",
            "",
        ]
    )

    output_path = evidence_dir / "riepilogo_metriche.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"...\nRiepilogo scritto in: {output_path}")


if __name__ == "__main__":
    main()
