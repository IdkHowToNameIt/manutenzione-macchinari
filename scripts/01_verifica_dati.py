"""S1 - Verifica dei dati e profilo condiviso.

Legge il dataset con il lettore del progetto, applica i controlli, produce il
profilo usato nella relazione e verifica che le richieste operative rispettino
lo schema delle 30 feature.
"""
from __future__ import annotations

import json
from pathlib import Path

from manutenzione.config import load_project_config
from manutenzione.contracts import load_feature_schema
from manutenzione.data_loader import (
    describe_dataset,
    load_dataset,
    load_monitoring_requests,
)


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
    monitoring_requests = load_monitoring_requests(config["paths"]["requests_csv"])

    profile = describe_dataset(bundle, feature_schema)
    profile["checks"] = [
        "file presente e non vuoto",
        "intestazione presente e senza colonne duplicate",
        "record_id univoco e non mancante",
        "valori numerici finiti su tutte le 30 feature",
        "target limitato a 0 e 1",
        "split limitato a train e validation",
        "target e record_id esclusi dalle feature",
        "ordine delle feature allineato a config/feature_schema.json",
    ]
    profile["monitoring_requests"] = [
        {
            "request_id": request.request_id,
            "model_version": request.model_version,
            "expected_status": request.expected_status,
            "feature_count": len(request.payload.get("features", [])),
            "schema_problems": feature_schema.validate_vector(
                request.payload.get("features", [])
            ),
        }
        for request in monitoring_requests
    ]

    evidence_dir = Path(config["paths"]["evidence_dir"]) / "data"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    profile_path = evidence_dir / "profilo_dati.json"
    with profile_path.open("w", encoding="utf-8") as profile_file:
        json.dump(profile, profile_file, indent=2, ensure_ascii=False)

    print(f"Dataset: {bundle.dataset_path}")
    print(f"Fingerprint SHA-256: {bundle.dataset_fingerprint}")
    print(f"Righe: {bundle.row_count}, feature: {len(bundle.feature_names)}")
    for split_name, counts in profile["class_counts"].items():
        total = counts["0"] + counts["1"]
        print(f"  {split_name}: target 0 = {counts['0']}, target 1 = {counts['1']}, totale = {total}")
    print("Richieste operative:")
    for request in profile["monitoring_requests"]:
        problems = "; ".join(request["schema_problems"]) or "schema valido"
        print(
            f"  {request['request_id']} ({request['model_version']}): "
            f"stato atteso {request['expected_status']}, {request['feature_count']} feature -> {problems}"
        )
    print(f"Profilo scritto in: {profile_path}")


if __name__ == "__main__":
    main()
