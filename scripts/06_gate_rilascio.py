"""S4 - Gate di rilascio della candidata v2.

Ogni requisito ha misura, soglia, esito e motivazione. Il gate confronta v2 con
la v1 gia' salvata e verifica che la probe sia attribuibile all'artefatto
misurato. La sola decisione del gate non cambia la versione attiva.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from manutenzione.config import load_project_config
from manutenzione.contracts import load_feature_schema
from manutenzione.registry import ModelRegistry

PASSED = "superato"
FAILED = "non superato"
NOT_EXECUTED = "non eseguito"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as source_file:
        return json.load(source_file)


def _requirement(
    identifier: str,
    description: str,
    measured: Any,
    threshold: str,
    passed: bool,
    motivation: str,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "requisito": description,
        "misura": measured,
        "soglia": threshold,
        "esito": PASSED if passed else FAILED,
        "motivazione": motivation,
    }


def _missing_requirement(identifier: str, description: str, motivation: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "requisito": description,
        "misura": None,
        "soglia": "misura necessaria",
        "esito": NOT_EXECUTED,
        "motivazione": motivation,
    }


def main() -> None:
    config = load_project_config()
    feature_schema = load_feature_schema(config["paths"]["feature_schema"])
    registry = ModelRegistry(config["paths"]["models_dir"], feature_schema)
    evidence_dir = Path(config["paths"]["evidence_dir"])
    gate_config = config["gate"]

    # Il caricamento verifica fingerprint, schema e metadati di entrambe le versioni.
    verified = True
    verification_error = ""
    try:
        _, metadata_v1 = registry.load_version("v1")
        _, metadata_v2 = registry.load_version("v2")
    except Exception as error:
        verified = False
        verification_error = str(error)
        metadata_v1, metadata_v2 = {}, {}

    training_v1 = _load_json(evidence_dir / "training" / "v1.json") or {}
    training_v2 = _load_json(evidence_dir / "training" / "v2.json") or {}
    probe_v1 = _load_json(evidence_dir / "serving" / "probe_v1.json")
    probe_v2 = _load_json(evidence_dir / "serving" / "probe_v2.json")

    metrics_v1 = metadata_v1.get("validation_metrics", {})
    metrics_v2 = metadata_v2.get("validation_metrics", {})
    requirements: list[dict[str, Any]] = []

    if not verified:
        requirements.append(
            _missing_requirement(
                "R0",
                "Artefatti v1 e v2 verificati prima del confronto",
                f"verifica fallita: {verification_error}",
            )
        )

    # R1 - accuratezza
    if metrics_v1 and metrics_v2:
        accuracy_v1 = float(metrics_v1["accuracy"])
        accuracy_v2 = float(metrics_v2["accuracy"])
        requirements.append(
            _requirement(
                "R1",
                "Accuratezza della candidata non inferiore alla baseline",
                f"v2 {accuracy_v2:.4f} vs v1 {accuracy_v1:.4f}",
                "v2 >= v1",
                accuracy_v2 >= accuracy_v1,
                "Il protocollo comune richiede accuratezza candidata almeno pari alla baseline.",
            )
        )
    else:
        requirements.append(
            _missing_requirement("R1", "Accuratezza della candidata", "metriche di training mancanti")
        )

    # R2 - recall assoluto sulla classe 1
    if metrics_v2:
        recall_v2 = float(metrics_v2["recall_target_1"])
        min_recall = float(gate_config["min_recall_target_1"])
        requirements.append(
            _requirement(
                "R2",
                "Recall sulla classe 1 (guasto entro 7 giorni)",
                f"v2 {recall_v2:.4f}",
                f">= {min_recall:.2f}",
                recall_v2 >= min_recall,
                "Soglia a priori sulla classe 1: un falso negativo e' un intervento mancato.",
            )
        )
    else:
        requirements.append(
            _missing_requirement("R2", "Recall sulla classe 1", "metriche di training mancanti")
        )

    # R3 - nessuna regressione del recall rispetto alla baseline
    if metrics_v1 and metrics_v2:
        recall_v1 = float(metrics_v1["recall_target_1"])
        recall_v2 = float(metrics_v2["recall_target_1"])
        requirements.append(
            _requirement(
                "R3",
                "Recall sulla classe 1 non inferiore alla baseline",
                f"v2 {recall_v2:.4f} vs v1 {recall_v1:.4f}",
                "v2 >= v1",
                recall_v2 >= recall_v1,
                "Un guadagno di accuratezza non deve ridurre la copertura dei guasti.",
            )
        )
    else:
        requirements.append(
            _missing_requirement("R3", "Recall non inferiore alla baseline", "metriche mancanti")
        )

    # R4 - latenza p95 della probe
    max_p95_ms = float(gate_config["max_p95_ms"])
    if probe_v2 and probe_v2.get("probe"):
        p95_v2 = float(probe_v2["probe"]["p95_ms"])
        requirements.append(
            _requirement(
                "R4",
                "Latenza p95 della candidata su 30 misure locali",
                f"{p95_v2:.2f} ms",
                f"<= {max_p95_ms:.0f} ms",
                p95_v2 <= max_p95_ms,
                "Limite del protocollo comune per la risposta di predict.",
            )
        )
    else:
        requirements.append(
            _missing_requirement(
                "R4",
                "Latenza p95 della candidata",
                "probe v2 mancante: senza misura la candidata non e' attivabile",
            )
        )

    # R5 - error rate della probe
    max_error_rate = float(gate_config["max_error_rate"])
    if probe_v2 and probe_v2.get("probe"):
        error_rate_v2 = float(probe_v2["probe"]["error_rate"])
        requirements.append(
            _requirement(
                "R5",
                "Error rate della candidata su 30 misure locali",
                f"{error_rate_v2:.4f}",
                f"<= {max_error_rate}",
                error_rate_v2 <= max_error_rate,
                "Errori HTTP >= 400 sulle richieste valide della probe.",
            )
        )
    else:
        requirements.append(
            _missing_requirement("R5", "Error rate della candidata", "probe v2 mancante")
        )

    # R6 - attribuzione della probe
    if probe_v2 and metadata_v2:
        attributable = bool(probe_v2.get("probe_attribuita_all_artefatto"))
        probe_fingerprint = probe_v2.get("health", {}).get("artifact_fingerprint")
        requirements.append(
            _requirement(
                "R6",
                "Probe v2 attribuita all'artefatto registrato",
                f"fingerprint misurato {probe_fingerprint}",
                "uguale al fingerprint dei metadati v2",
                attributable,
                "Una probe di un altro artefatto non autorizza l'attivazione.",
            )
        )
    else:
        requirements.append(
            _missing_requirement("R6", "Attribuzione della probe v2", "probe o metadati mancanti")
        )

    # R7 - stesso protocollo di confronto
    if metadata_v1 and metadata_v2:
        same_dataset = (
            metadata_v1["training"]["dataset_fingerprint"]
            == metadata_v2["training"]["dataset_fingerprint"]
        )
        same_validation_ids = (
            metadata_v1["training"]["validation_record_ids_fingerprint"]
            == metadata_v2["training"]["validation_record_ids_fingerprint"]
        )
        same_schema = (
            metadata_v1["feature_schema"]["fingerprint"]
            == metadata_v2["feature_schema"]["fingerprint"]
        )
        same_protocol = same_dataset and same_validation_ids and same_schema
        requirements.append(
            _requirement(
                "R7",
                "Stesso dataset, stessi ID di validation e stesso schema",
                {
                    "stesso_dataset": same_dataset,
                    "stessi_id_validation": same_validation_ids,
                    "stesso_schema": same_schema,
                },
                "tutte le uguaglianze vere",
                same_protocol,
                "Il confronto tra versioni e' valido solo a parita' di dati e protocollo.",
            )
        )
    else:
        requirements.append(
            _missing_requirement("R7", "Stesso protocollo di confronto", "metadati mancanti")
        )

    blocking = [item for item in requirements if item["esito"] != PASSED]
    decision = "accettata" if not blocking else "bloccata"
    decided_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    comparison = {
        "accuratezza": {
            "v1": metrics_v1.get("accuracy"),
            "v2": metrics_v2.get("accuracy"),
        },
        "recall_target_1": {
            "v1": metrics_v1.get("recall_target_1"),
            "v2": metrics_v2.get("recall_target_1"),
        },
        "precision_target_1": {
            "v1": metrics_v1.get("precision_target_1"),
            "v2": metrics_v2.get("precision_target_1"),
        },
        "roc_auc": {"v1": metrics_v1.get("roc_auc"), "v2": metrics_v2.get("roc_auc")},
        "matrice_confusione": {
            "v1": metrics_v1.get("confusion_matrix"),
            "v2": metrics_v2.get("confusion_matrix"),
        },
        "soglia_decisione": {
            "v1": metadata_v1.get("decision_threshold"),
            "v2": metadata_v2.get("decision_threshold"),
        },
        "p95_ms": {
            "v1": probe_v1["probe"]["p95_ms"] if probe_v1 else None,
            "v2": probe_v2["probe"]["p95_ms"] if probe_v2 else None,
        },
        "error_rate": {
            "v1": probe_v1["probe"]["error_rate"] if probe_v1 else None,
            "v2": probe_v2["probe"]["error_rate"] if probe_v2 else None,
        },
        "run_id": {"v1": metadata_v1.get("run_id"), "v2": metadata_v2.get("run_id")},
        "artifact_fingerprint": {
            "v1": metadata_v1.get("artifact_fingerprint"),
            "v2": metadata_v2.get("artifact_fingerprint"),
        },
    }

    report = {
        "decisione": decision,
        "decisa_il": decided_at,
        "candidata": "v2",
        "baseline": "v1",
        "requisiti": requirements,
        "requisiti_bloccanti": [item["id"] for item in blocking],
        "confronto": comparison,
        "nota": (
            "La decisione del gate non cambia la versione attiva: l'attivazione e' "
            "un'operazione separata e registrata nel registro delle transizioni."
        ),
    }

    gate_dir = evidence_dir / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    with (gate_dir / "gate_report.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)

    lines = [
        "# Gate di rilascio v2",
        "",
        f"Decisione: **{decision}** ({decided_at})",
        "",
        "| ID | Requisito | Misura | Soglia | Esito | Motivazione |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in requirements:
        measured = item["misura"]
        if isinstance(measured, dict):
            measured = ", ".join(f"{key}={value}" for key, value in measured.items())
        lines.append(
            f"| {item['id']} | {item['requisito']} | {measured} | {item['soglia']} | "
            f"{item['esito']} | {item['motivazione']} |"
        )
    lines.extend(
        [
            "",
            "## Confronto sintetico",
            "",
            "| Metrica | v1 | v2 |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, values in comparison.items():
        if isinstance(values.get("v1"), dict) or isinstance(values.get("v2"), dict):
            continue
        lines.append(f"| {label} | {values.get('v1')} | {values.get('v2')} |")
    lines.extend(
        [
            "",
            "## Nota",
            "",
            report["nota"],
            "",
        ]
    )
    with (gate_dir / "gate_report.md").open("w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))

    print(f"Decisione: {decision}")
    for item in requirements:
        print(f"  {item['id']}: {item['esito']} -> {item['misura']}")
    print(f"Report: {gate_dir / 'gate_report.md'}")


if __name__ == "__main__":
    main()
