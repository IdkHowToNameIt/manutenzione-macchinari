"""Controlli del registro: identita' dell'artefatto, attivazione e rollback."""
from __future__ import annotations

import dataclasses
import json

import pytest

from manutenzione.registry import (
    ArtifactValidationError,
    ModelRegistry,
    RegistryError,
)


def test_version_roundtrip_keeps_metadata(test_registry, feature_schema):
    pipeline, metadata = test_registry.load_version("v2")
    assert metadata["model_version"] == "v2"
    assert metadata["run_id"] == "run-v2"
    assert metadata["feature_schema"]["names"] == feature_schema.names
    assert len(metadata["artifact_fingerprint"]) == 64
    assert hasattr(pipeline, "predict_proba")


def test_tampered_artifact_is_rejected(test_registry):
    artifact_path = test_registry.version_directory("v1") / "model.pkl"
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
    with pytest.raises(ArtifactValidationError, match="Fingerprint dell'artefatto"):
        test_registry.load_version("v1")


def test_changed_feature_schema_is_rejected(test_registry, feature_schema):
    different_schema = dataclasses.replace(feature_schema, fingerprint="schema-diverso")
    strict_registry = ModelRegistry(test_registry.models_dir, different_schema)
    with pytest.raises(ArtifactValidationError, match="Schema delle feature cambiato"):
        strict_registry.load_version("v1")


def test_activation_and_rollback_are_recorded(test_registry):
    pointer = test_registry.activate("v2", actor="Livia", reason="gate superato")
    assert pointer["active_version"] == "v2"
    assert pointer["previous_version"] == "v1"

    restored = test_registry.rollback(actor="Andrea", reason="drill di ripristino")
    assert restored["active_version"] == "v1"
    assert restored["previous_version"] == "v2"
    assert [entry["action"] for entry in restored["history"]] == [
        "initial_activation",
        "activate",
        "rollback",
    ]

    transitions = [
        json.loads(line)
        for line in (test_registry.models_dir / "transitions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert transitions[-1]["to_version"] == "v1"
    assert transitions[-1]["run_id"] == "run-v1"
    assert len(transitions[-1]["artifact_fingerprint"]) == 64


def test_missing_version_metadata_is_rejected(test_registry):
    with pytest.raises(RegistryError, match="Metadati mancanti"):
        test_registry.load_version("v9")


def test_existing_version_is_not_overwritten(test_registry):
    pipeline, metadata = test_registry.load_version("v1")
    with pytest.raises(RegistryError, match="esiste gia'"):
        test_registry.save_version("v1", pipeline, metadata)


def test_rollback_without_previous_version_fails(project_config, feature_schema):
    empty_registry = ModelRegistry(project_config["paths"]["models_dir"], feature_schema)
    with pytest.raises(RegistryError, match="Nessuna versione precedente"):
        empty_registry.rollback(actor="test")
