"""S2 - Training della baseline v1 e prima attivazione locale.

Addestra la pipeline StandardScaler + LogisticRegression sul solo train,
valuta sulla validation, salva modello e metadati in models/v1 e collega la
run MLflow a dati, configurazione e metriche.
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
        version="v1",
        model_key="logistic_regression",
        config=config,
        feature_schema=feature_schema,
        bundle=bundle,
        registry=registry,
        registered_by="Matteo",
        notes="Baseline lineare di riferimento per il confronto con la candidata.",
    )

    # La baseline diventa la versione attiva solo se il registro non ne ha ancora una.
    pointer = registry.read_pointer()
    if pointer.get("active_version") is None:
        registry.activate(
            "v1",
            actor="Matteo",
            action="initial_activation",
            reason="prima versione registrata: baseline di riferimento",
        )
        print("v1 impostata come versione attiva iniziale")
    else:
        print(f"Versione attiva invariata: {pointer['active_version']}")

    print(f"Artefatto: models/v1/{metadata['artifact_file']}")
    print(f"Fingerprint artefatto: {metadata['artifact_fingerprint']}")
    print(f"Run MLflow: {metadata['run_id']}")


if __name__ == "__main__":
    main()
