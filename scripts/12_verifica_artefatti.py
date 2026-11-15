"""Verifica di coerenza degli artefatti presenti nel registro.

Utile in CI e dopo un clone: carica ogni versione, controlla fingerprint,
schema e metadati, e verifica che la versione attiva esista e sia caricabile.
"""
from __future__ import annotations

from manutenzione.config import load_project_config
from manutenzione.contracts import load_feature_schema
from manutenzione.registry import ModelRegistry


def main() -> None:
    config = load_project_config()
    feature_schema = load_feature_schema(config["paths"]["feature_schema"])
    registry = ModelRegistry(config["paths"]["models_dir"], feature_schema)

    versions = registry.list_versions()
    if not versions:
        raise SystemExit("Nessuna versione registrata in models/")

    for entry in versions:
        pipeline, metadata = registry.load_version(entry["model_version"])
        print(
            f"{entry['model_version']}: run={metadata['run_id']}, "
            f"fingerprint={metadata['artifact_fingerprint'][:12]}..., "
            f"feature={metadata['feature_schema']['count']}, "
            f"accuracy={metadata['validation_metrics'].get('accuracy')}"
        )
        if not hasattr(pipeline, "predict_proba"):
            raise SystemExit(f"{entry['model_version']}: pipeline senza predict_proba")

    pointer = registry.read_pointer()
    active_version = pointer.get("active_version")
    if not active_version:
        raise SystemExit("Nessuna versione attiva in models/active_version.json")
    registry.load_version(active_version)
    print(f"Versione attiva verificata: {active_version}")
    print(f"Versione precedente: {pointer.get('previous_version')}")
    print(f"Transizioni registrate: {len(pointer.get('history', []))}")


if __name__ == "__main__":
    main()
