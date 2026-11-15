"""Training ripetibile, valutazione su validation e registrazione delle versioni.

La stessa funzione addestra baseline e candidata sullo stesso train, valuta
sulla stessa validation e collega ogni versione a una run MLflow. Le
trasformazioni che apprendono parametri (per esempio StandardScaler) stanno
dentro la pipeline: si adattano solo sul train e vengono salvate insieme al
modello, quindi il servizio applica la stessa preparazione usata in training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import FeatureSchema
from .data_loader import DatasetBundle
from .registry import ModelRegistry, utc_timestamp

# Specifiche dei modelli usati. Iperparametri fissati prima della valutazione:
# la validation serve a confrontare le versioni, non a cercare la configurazione migliore.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "model_family": "logistic_regression",
        "hyperparameters": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        "hypothesis": (
            "Baseline lineare: confine di decisione lineare sulle feature standardizzate, "
            "semplice da spiegare e da confrontare."
        ),
    },
    "random_forest": {
        "model_family": "random_forest",
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,
        },
        "hypothesis": (
            "Candidata non lineare: gli effetti termici e di vibrazione si combinano in modo "
            "non lineare, quindi un insieme di alberi con pesi di classe bilanciati puo' "
            "separare meglio i macchinari a rischio."
        ),
        # La soglia operativa viene scelta con validazione incrociata sul solo train,
        # per rispettare il requisito di recall senza usare la validation.
        "calibrate_threshold": True,
    },
}


def build_pipeline(model_key: str, random_state: int) -> Pipeline:
    """Costruisce la pipeline di inferenza per il modello richiesto."""
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Modello sconosciuto: {model_key}")

    hyperparameters = MODEL_SPECS[model_key]["hyperparameters"]
    if model_key == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**hyperparameters, random_state=random_state)),
            ]
        )
    if model_key == "random_forest":
        return Pipeline(
            steps=[
                ("classifier", RandomForestClassifier(**hyperparameters, random_state=random_state)),
            ]
        )
    raise ValueError(f"Modello sconosciuto: {model_key}")


def evaluate_pipeline(
    pipeline: Pipeline,
    features: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Metriche di validation con matrice di confusione in forma esplicita."""
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    return {
        "n_samples": int(len(targets)),
        "positive_rate_observed": float(np.mean(targets)),
        "positive_rate_predicted": float(np.mean(predictions)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "recall_target_1": float(recall_score(targets, predictions, pos_label=1, zero_division=0)),
        "precision_target_1": float(precision_score(targets, predictions, pos_label=1, zero_division=0)),
        "f1_target_1": float(f1_score(targets, predictions, pos_label=1, zero_division=0)),
        "recall_target_0": float(recall_score(targets, predictions, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "decision_threshold": float(threshold),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": matrix.tolist(),
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
    }


def _validation_ids_fingerprint(record_ids: list[str]) -> str:
    joined_ids = "\n".join(record_ids).encode("utf-8")
    return hashlib.sha256(joined_ids).hexdigest()


def select_decision_threshold(
    pipeline: Pipeline,
    features: np.ndarray,
    targets: np.ndarray,
    min_recall_target_1: float,
    random_state: int,
) -> dict[str, Any]:
    """Sceglie la soglia di decisione con validazione incrociata sul solo train.

    Il requisito operativo e' la sensibilita' sulla classe 1 (guasto): tra le
    soglie che raggiungono il recall minimo si prende quella con l'accuratezza
    out-of-fold piu' alta. La validation non viene mai usata in questo passaggio.
    """
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    out_of_fold_probabilities = cross_val_predict(
        clone(pipeline),
        features,
        targets,
        cv=cross_validation,
        method="predict_proba",
    )[:, 1]

    candidates = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        predictions = (out_of_fold_probabilities >= threshold).astype(int)
        accuracy = float(accuracy_score(targets, predictions))
        recall = float(recall_score(targets, predictions, pos_label=1, zero_division=0))
        candidates.append({"threshold": float(threshold), "accuracy": accuracy, "recall": recall})

    feasible = [item for item in candidates if item["recall"] >= min_recall_target_1]
    if feasible:
        selected = max(feasible, key=lambda item: item["accuracy"])
        selection_method = "max_accuracy_con_recall_minimo_out_of_fold"
    else:
        selected = max(candidates, key=lambda item: item["accuracy"])
        selection_method = "max_accuracy_senza_soglia_ammissibile"

    return {
        "selected_threshold": selected["threshold"],
        "selection_method": selection_method,
        "min_recall_target_1": float(min_recall_target_1),
        "out_of_fold_accuracy": selected["accuracy"],
        "out_of_fold_recall": selected["recall"],
        "cross_validation_folds": cross_validation.get_n_splits(),
        "threshold_grid": "0.05-0.95 passo 0.01",
    }


def train_and_register(
    version: str,
    model_key: str,
    config: dict[str, Any],
    feature_schema: FeatureSchema,
    bundle: DatasetBundle,
    registry: ModelRegistry,
    registered_by: str,
    notes: str,
) -> dict[str, Any]:
    """Addestra una versione, la valuta, la registra su MLflow e nel registro locale."""
    project_config = config["project"]
    training_config = config["training"]
    train_split = training_config["train_split"]
    validation_split = training_config["validation_split"]
    threshold = float(config["service"]["probability_threshold"])
    random_state = int(project_config["random_state"])

    train_mask = bundle.split_mask(train_split)
    validation_mask = bundle.split_mask(validation_split)
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("Split train o validation assente nel dataset")

    train_features = bundle.features[train_mask]
    train_targets = bundle.targets[train_mask]
    validation_features = bundle.features[validation_mask]
    validation_targets = bundle.targets[validation_mask]
    validation_record_ids = [
        record_id
        for record_id, is_validation in zip(bundle.record_ids, validation_mask)
        if is_validation
    ]

    pipeline = build_pipeline(model_key, random_state)
    model_spec = MODEL_SPECS[model_key]
    if model_spec.get("calibrate_threshold"):
        threshold_selection = select_decision_threshold(
            pipeline,
            train_features,
            train_targets,
            min_recall_target_1=float(config["gate"]["min_recall_target_1"]),
            random_state=random_state,
        )
        threshold = threshold_selection["selected_threshold"]
    else:
        threshold_selection = {
            "selected_threshold": threshold,
            "selection_method": "soglia_fissa_da_configurazione",
            "min_recall_target_1": float(config["gate"]["min_recall_target_1"]),
        }

    pipeline.fit(train_features, train_targets)
    metrics = evaluate_pipeline(pipeline, validation_features, validation_targets, threshold)

    mlflow.set_tracking_uri(config["paths"]["mlflow_tracking_uri"])
    experiment_name = project_config["experiment_name"]
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_location = Path(config["paths"]["mlflow_artifacts"]).resolve().as_uri()
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{version}-{model_key}") as active_run:
        run_id = active_run.info.run_id

        metadata: dict[str, Any] = {
            "run_id": run_id,
            "registered_by": registered_by,
            "model_family": MODEL_SPECS[model_key]["model_family"],
            "model_key": model_key,
            "hyperparameters": MODEL_SPECS[model_key]["hyperparameters"],
            "hypothesis": MODEL_SPECS[model_key]["hypothesis"],
            "notes": notes,
            "decision_threshold": threshold,
            "threshold_selection": threshold_selection,
            "training": {
                "dataset_path": bundle.dataset_path,
                "dataset_fingerprint": bundle.dataset_fingerprint,
                "row_count": bundle.row_count,
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "train_class_counts": {
                    "0": int(np.sum(train_targets == 0)),
                    "1": int(np.sum(train_targets == 1)),
                },
                "validation_class_counts": {
                    "0": int(np.sum(validation_targets == 0)),
                    "1": int(np.sum(validation_targets == 1)),
                },
                "random_state": random_state,
                "train_split": train_split,
                "validation_split": validation_split,
                "validation_record_ids_fingerprint": _validation_ids_fingerprint(
                    validation_record_ids
                ),
            },
            "validation_metrics": metrics,
            "target_definition": {
                "positive_label": training_config["target_positive_label"],
                "negative_label": training_config["target_negative_label"],
            },
            "registered_at": utc_timestamp(),
        }
        metadata = registry.save_version(version, pipeline, metadata)

        mlflow.log_params(
            {
                "model_key": model_key,
                "model_family": metadata["model_family"],
                "random_state": random_state,
                "feature_count": feature_schema.count,
                "decision_threshold": threshold,
                "train_rows": metadata["training"]["train_rows"],
                "validation_rows": metadata["training"]["validation_rows"],
                **{
                    f"hyperparameter_{name}": value
                    for name, value in MODEL_SPECS[model_key]["hyperparameters"].items()
                },
            }
        )
        mlflow.log_metrics(
            {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        mlflow.set_tags(
            {
                "model_version": version,
                "dataset_fingerprint": bundle.dataset_fingerprint,
                "feature_schema_fingerprint": feature_schema.fingerprint,
                "validation_record_ids_fingerprint": metadata["training"][
                    "validation_record_ids_fingerprint"
                ],
                "target_definition": training_config["target_positive_label"],
            }
        )
        mlflow.log_artifact(
            str(registry.version_directory(version)), artifact_path=f"model_{version}"
        )

    evidence_directory = Path(config["paths"]["evidence_dir"]) / "training"
    evidence_directory.mkdir(parents=True, exist_ok=True)
    # Evidenza portabile: niente percorsi assoluti che espongano l'ambiente locale.
    tracking_uri = config["paths"]["mlflow_tracking_uri"]
    if tracking_uri.startswith("sqlite:///"):
        tracking_uri = f"sqlite:///{Path(tracking_uri.removeprefix('sqlite:///')).name}"
    evidence = {
        "model_version": version,
        "mlflow_run_id": run_id,
        "mlflow_experiment": project_config["experiment_name"],
        "mlflow_tracking_uri": tracking_uri,
        "registered_at": metadata["registered_at"],
        "artifact_fingerprint": metadata["artifact_fingerprint"],
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "feature_schema_fingerprint": feature_schema.fingerprint,
        "model_family": metadata["model_family"],
        "hyperparameters": metadata["hyperparameters"],
        "decision_threshold": metadata["decision_threshold"],
        "evidence": metrics,
    }
    with (evidence_directory / f"{version}.json").open("w", encoding="utf-8") as evidence_file:
        json.dump(evidence, evidence_file, indent=2, ensure_ascii=False)

    print(
        f"{version}: accuracy={metrics['accuracy']:.4f}, "
        f"recall(target=1)={metrics['recall_target_1']:.4f}, "
        f"run_id={run_id}"
    )
    return metadata
