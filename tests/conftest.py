"""Fixture condivise: configurazione temporanea e registro con due versioni."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from manutenzione.config import load_project_config
from manutenzione.contracts import load_feature_schema
from manutenzione.registry import ModelRegistry


@pytest.fixture()
def feature_schema():
    config = load_project_config()
    return load_feature_schema(config["paths"]["feature_schema"])


@pytest.fixture()
def project_config(tmp_path: Path) -> dict:
    """Configurazione reale con cartelle di lavoro isolate nella directory temporanea."""
    config = copy.deepcopy(load_project_config())
    config["paths"]["models_dir"] = tmp_path / "models"
    config["paths"]["logs_dir"] = tmp_path / "logs"
    config["paths"]["evidence_dir"] = tmp_path / "evidence"
    return config


def _fit_small_pipeline(random_state: int, positive_weight: float) -> Pipeline:
    generator = np.random.default_rng(random_state)
    features = generator.normal(size=(120, 30))
    targets = (features[:, 0] + 0.5 * features[:, 1] + generator.normal(scale=0.5, size=120) > 0).astype(int)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, C=positive_weight)),
        ]
    )
    pipeline.fit(features, targets)
    return pipeline


def build_test_registry(models_dir: Path, feature_schema, versions: tuple[str, ...] = ("v1", "v2")) -> ModelRegistry:
    """Crea un registro con versioni fittizie ma con la stessa struttura dei metadati reali."""
    registry = ModelRegistry(models_dir, feature_schema)
    for index, version in enumerate(versions):
        pipeline = _fit_small_pipeline(random_state=10 + index, positive_weight=1.0 + index)
        registry.save_version(
            version,
            pipeline,
            {
                "run_id": f"run-{version}",
                "registered_by": "test",
                "model_family": "logistic_regression",
                "training": {
                    "dataset_fingerprint": "dataset-test",
                    "row_count": 120,
                },
                "validation_metrics": {
                    "accuracy": 0.60 + 0.05 * index,
                    "recall_target_1": 0.40 + 0.05 * index,
                    "decision_threshold": 0.5,
                },
                "decision_threshold": 0.5,
            },
        )
    registry.activate("v1", actor="test", action="initial_activation", reason="fixture")
    return registry


@pytest.fixture()
def test_registry(project_config, feature_schema):
    return build_test_registry(project_config["paths"]["models_dir"], feature_schema)
