"""Controllo di drift illustrativo sulle richieste assegnate.

Confronta i payload validi del CSV richieste con la distribuzione del train
usando z-score robusti (mediana e MAD). Con quattro richieste il controllo e'
solo illustrativo: in produzione servirebbe una finestra aggregata di traffico.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from manutenzione.config import load_project_config
from manutenzione.contracts import load_feature_schema
from manutenzione.data_loader import load_dataset, load_monitoring_requests

DRIFT_THRESHOLD = 3.0


def main() -> None:
    config = load_project_config()
    feature_schema = load_feature_schema(config["paths"]["feature_schema"])
    bundle = load_dataset(
        config["paths"]["dataset_csv"],
        feature_schema,
        id_column=config["training"]["id_column"],
        target_column=config["training"]["target_column"],
        split_column=config["training"]["split_column"],
    )
    train_features = bundle.features[bundle.split_mask("train")]
    median = np.median(train_features, axis=0)
    # MAD scalato a una deviazione standard equivalente per la normale.
    mad = np.median(np.abs(train_features - median), axis=0) * 1.4826
    safe_scale = np.where(mad > 1e-9, mad, 1.0)

    monitoring_requests = load_monitoring_requests(config["paths"]["requests_csv"])
    report: dict = {
        "riferimento": {
            "dataset_fingerprint": bundle.dataset_fingerprint,
            "train_rows": int(train_features.shape[0]),
            "statistica": "z-score robusto = (valore - mediana_train) / (1.4826 * MAD_train)",
            "soglia": DRIFT_THRESHOLD,
        },
        "richieste": [],
        "limiti": (
            "Solo tre payload validi: il controllo segnala valori lontani dalla "
            "distribuzione di training, non un drift statistico consolidato."
        ),
    }

    lines = [
        "# Controllo di drift sulle richieste assegnate",
        "",
        f"Riferimento: train, {train_features.shape[0]} righe, "
        f"fingerprint `{bundle.dataset_fingerprint}`.",
        "",
        f"Soglia: |z-score robusto| > {DRIFT_THRESHOLD}. {report['limiti']}",
        "",
        "| Richiesta | Versione | Feature oltre soglia |",
        "| --- | --- | --- |",
    ]

    for request in monitoring_requests:
        if request.expected_status != 200:
            continue
        features = np.asarray(request.payload["features"], dtype=float)
        z_scores = (features - median) / safe_scale
        flagged = [
            {"feature": feature_schema.names[index], "z_score": round(float(z_scores[index]), 3)}
            for index in np.where(np.abs(z_scores) > DRIFT_THRESHOLD)[0]
        ]
        request_summary = {
            "request_id": request.request_id,
            "model_version": request.model_version,
            "feature_oltre_soglia": flagged,
            "max_abs_z_score": round(float(np.max(np.abs(z_scores))), 3),
        }
        report["richieste"].append(request_summary)
        flagged_text = (
            ", ".join(f"{item['feature']} ({item['z_score']:+.2f})" for item in flagged)
            or "nessuna"
        )
        lines.append(f"| {request.request_id} | {request.model_version} | {flagged_text} |")

    lines.extend(
        [
            "",
            "## Come usarlo in produzione",
            "",
            "Il servizio non salva i payload nei log: un controllo di drift reale deve "
            "leggere una finestra aggregata di statistiche (per esempio medie orarie per "
            "feature) prodotta a monte, oppure un campione autorizzato e anonimizzato. "
            "Questo script verifica soltanto che le richieste assegnate siano plausibili "
            "rispetto al train.",
            "",
        ]
    )

    evidence_dir = Path(config["paths"]["evidence_dir"]) / "monitoring"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    with (evidence_dir / "drift_report.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)
    (evidence_dir / "drift_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(report["richieste"], ensure_ascii=False, indent=2))
    print(f"Report: {evidence_dir / 'drift_report.md'}")


if __name__ == "__main__":
    main()
