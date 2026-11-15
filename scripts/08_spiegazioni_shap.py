"""Spiegazioni SHAP della candidata v2.

Calcola i valori SHAP sul campione di validation e li riassume in una classifica
di importanza media assoluta. Le spiegazioni descrivono il comportamento del
modello, non un rapporto causale tra feature e guasto.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import shap

from manutenzione.config import load_project_config
from manutenzione.contracts import load_feature_schema
from manutenzione.data_loader import load_dataset, load_monitoring_requests
from manutenzione.registry import ModelRegistry


def _class_one_shap_values(explainer: shap.TreeExplainer, features: np.ndarray) -> np.ndarray:
    values = explainer.shap_values(features)
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


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
    registry = ModelRegistry(config["paths"]["models_dir"], feature_schema)
    pipeline, metadata = registry.load_version("v2")

    if list(pipeline.named_steps) != ["classifier"]:
        raise RuntimeError(
            "Lo script SHAP si aspetta la pipeline v2 con il solo classificatore ad albero"
        )
    classifier = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)

    validation_features = bundle.features[bundle.split_mask("validation")]
    shap_values = _class_one_shap_values(explainer, validation_features)
    mean_absolute = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    ranking = sorted(
        zip(feature_schema.names, mean_absolute, mean_signed),
        key=lambda item: item[1],
        reverse=True,
    )

    evidence_dir = Path(config["paths"]["evidence_dir"]) / "explanations"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    csv_path = evidence_dir / "shap_importance_v2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["posizione", "feature", "mean_abs_shap", "mean_shap"])
        position_by_name = {spec.name: spec.position for spec in feature_schema.features}
        for feature_name, absolute_value, signed_value in ranking:
            writer.writerow(
                [
                    position_by_name[feature_name],
                    feature_name,
                    round(float(absolute_value), 6),
                    round(float(signed_value), 6),
                ]
            )

    # Spiegazione della richiesta v2 assegnata, utile per la presentazione.
    monitoring_requests = load_monitoring_requests(config["paths"]["requests_csv"])
    v2_request = next(
        request
        for request in monitoring_requests
        if request.model_version == "v2" and request.expected_status == 200
    )
    single_features = np.asarray(v2_request.payload["features"], dtype=float).reshape(1, -1)
    single_values = _class_one_shap_values(explainer, single_features)[0]
    contributions = sorted(
        zip(feature_schema.names, single_values),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    probability = float(classifier.predict_proba(single_features)[0, 1])

    summary_lines = [
        "# Spiegazioni SHAP della candidata v2",
        "",
        f"Run di riferimento: `{metadata['run_id']}` ",
        f"Fingerprint artefatto: `{metadata['artifact_fingerprint']}`",
        "",
        "## Importanza media assoluta (validation, 200 righe)",
        "",
        "| Posizione | Feature | Importanza media (abs SHAP) | Direzione media |",
        "| ---: | --- | ---: | --- |",
    ]
    for feature_name, absolute_value, signed_value in ranking[:10]:
        direction = "aumenta il rischio" if signed_value > 0 else "riduce il rischio"
        summary_lines.append(
            f"| {next(spec.position for spec in feature_schema.features if spec.name == feature_name)} "
            f"| {feature_name} | {absolute_value:.4f} | {direction} |"
        )
    summary_lines.extend(
        [
            "",
            f"## Richiesta {v2_request.request_id} (probabilita' classe 1 = {probability:.4f})",
            "",
            "| Feature | Contributo SHAP | Segno |",
            "| --- | ---: | --- |",
        ]
    )
    for feature_name, contribution in contributions[:6]:
        direction = "verso la classe 1" if contribution > 0 else "verso la classe 0"
        summary_lines.append(f"| {feature_name} | {contribution:+.4f} | {direction} |")
    summary_lines.extend(
        [
            "",
            "## Limiti",
            "",
            "I valori SHAP descrivono come il modello combina le feature: non dimostrano "
            "un rapporto causale con il guasto. La spiegazione vale per questa versione "
            "dell'artefatto e per la distribuzione dei dati di training.",
            "",
        ]
    )
    summary_path = evidence_dir / "shap_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    # Riepilogo leggibile anche in JSON per il report.
    json_path = evidence_dir / "shap_summary.json"
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            {
                "model_version": "v2",
                "run_id": metadata["run_id"],
                "artifact_fingerprint": metadata["artifact_fingerprint"],
                "validation_rows": int(len(validation_features)),
                "top_features": [
                    {
                        "feature": feature_name,
                        "mean_abs_shap": float(absolute_value),
                        "mean_shap": float(signed_value),
                    }
                    for feature_name, absolute_value, signed_value in ranking[:10]
                ],
                "request_explained": {
                    "request_id": v2_request.request_id,
                    "probability_target_1": probability,
                    "contributions": [
                        {"feature": feature_name, "shap_value": float(value)}
                        for feature_name, value in contributions
                    ],
                },
            },
            json_file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Top feature: {', '.join(name for name, _, _ in ranking[:5])}")
    print(f"CSV: {csv_path}")
    print(f"Riepilogo: {summary_path}")


if __name__ == "__main__":
    main()
