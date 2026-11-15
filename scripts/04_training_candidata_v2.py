"""S4 - Training e registrazione della candidata v2.

Ipotesi: la baseline lineare non cattura le interazioni tra temperatura,
vibrazione e usura; un RandomForest con pesi di classe bilanciati puo' alzare
il recall sulla classe 1. La soglia operativa viene scelta con validazione
incrociata sul solo train, la validation resta la stessa di v1.
"""
from __future__ import annotations

from manutenzione.config import load_project_config
from manutenzione.contracts import load_feature_schema
from manutenzione.data_loader import load_dataset
from manutenzione.registry import ModelRegistry
from manutenzione.training import train_and_register


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

    metadata = train_and_register(
        version="v2",
        model_key="random_forest",
        config=config,
        feature_schema=feature_schema,
        bundle=bundle,
        registry=registry,
        registered_by="Matteo",
        notes=(
            "Candidata non lineare. La soglia e' calibrata con 5 fold sul train "
            "per rispettare il recall minimo richiesto dal gate."
        ),
    )

    print(f"Artefatto: models/v2/{metadata['artifact_file']}")
    print(f"Fingerprint artefatto: {metadata['artifact_fingerprint']}")
    print(f"Soglia di decisione: {metadata['decision_threshold']}")
    print(f"Selezione soglia: {metadata['threshold_selection']}")
    print(f"Run MLflow: {metadata['run_id']}")


if __name__ == "__main__":
    main()
