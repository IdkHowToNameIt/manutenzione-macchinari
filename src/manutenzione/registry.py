"""Registro locale delle versioni del modello.

Ogni versione vive in models/<versione>/ con due file:
- model.pkl: pipeline completa di inferenza (preparazione dati + modello);
- model_metadata.json: run MLflow, fingerprint di dati e artefatto, schema,
  metriche di validation e versioni software.

Il registro conserva anche lo stato attivo (models/active_version.json) e il
registro delle transizioni (models/transitions.jsonl). Prima di usare un
artefatto il caricatore ricalcola il fingerprint dei byte e confronta schema e
ordine delle feature: una copia che non corrisponde ai metadati viene rifiutata
con un motivo esplicito.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from .contracts import FeatureSchema, file_fingerprint

REQUIRED_METADATA_KEYS = (
    "model_version",
    "artifact_file",
    "artifact_fingerprint",
    "run_id",
    "feature_schema",
    "training",
    "validation_metrics",
)


class RegistryError(Exception):
    """Errore del registro (versione assente, transizione non ammessa)."""


class ArtifactValidationError(Exception):
    """L'artefatto non corrisponde ai metadati o allo schema: uso rifiutato."""


def utc_timestamp() -> str:
    """Timestamp ISO 8601 in UTC, usato in metadati e log."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def software_versions() -> dict[str, str]:
    """Versioni realmente usate, da conservare nei metadati."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "mlflow": _safe_package_version("mlflow"),
        "fastapi": _safe_package_version("fastapi"),
    }


def _safe_package_version(package_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return "non disponibile"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Scrive il JSON su file temporaneo e sostituisce il target in un solo passaggio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as temporary_file:
        json.dump(payload, temporary_file, indent=2, ensure_ascii=False)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
        temporary_name = temporary_file.name
    os.replace(temporary_name, path)


class ModelRegistry:
    """Gestisce versioni, verifica degli artefatti e stato attivo."""

    def __init__(self, models_dir: str | Path, feature_schema: FeatureSchema):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feature_schema = feature_schema
        self.pointer_path = self.models_dir / "active_version.json"
        self.transitions_path = self.models_dir / "transitions.jsonl"

    def version_directory(self, version: str) -> Path:
        if not version or "/" in version or "\\" in version:
            raise RegistryError(f"Identificativo di versione non valido: {version!r}")
        return self.models_dir / version

    def save_version(
        self,
        version: str,
        pipeline: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Salva pipeline e metadati in una cartella nuova, senza sovrascrivere versioni esistenti."""
        version_dir = self.version_directory(version)
        metadata_path = version_dir / "model_metadata.json"
        if metadata_path.exists():
            raise RegistryError(
                f"La versione {version} esiste gia' in {version_dir}: usarne un'altra"
            )

        version_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = version_dir / "model.pkl"
        joblib.dump(pipeline, artifact_path)

        enriched_metadata = dict(metadata)
        enriched_metadata.update(
            {
                "model_version": version,
                "artifact_file": artifact_path.name,
                "artifact_format": "joblib-scikit-learn-pipeline",
                "artifact_fingerprint": file_fingerprint(artifact_path),
                "created_at": utc_timestamp(),
                "feature_schema": {
                    "version": self.feature_schema.version,
                    "fingerprint": self.feature_schema.fingerprint,
                    "count": self.feature_schema.count,
                    "names": self.feature_schema.names,
                },
                "software_versions": software_versions(),
            }
        )
        _atomic_write_json(metadata_path, enriched_metadata)
        return enriched_metadata

    def read_metadata(self, version: str) -> dict[str, Any]:
        metadata_path = self.version_directory(version) / "model_metadata.json"
        if not metadata_path.is_file():
            raise RegistryError(f"Metadati mancanti per la versione {version}: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    def load_version(self, version: str) -> tuple[Any, dict[str, Any]]:
        """Carica una versione dopo averne verificato identita', schema e fingerprint."""
        metadata = self.read_metadata(version)
        missing_keys = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
        if missing_keys:
            raise ArtifactValidationError(
                f"Metadati incompleti per {version}: mancano {', '.join(missing_keys)}"
            )
        if metadata["model_version"] != version:
            raise ArtifactValidationError(
                f"Metadati incoerenti: richiesta {version}, metadati {metadata['model_version']}"
            )

        version_dir = self.version_directory(version)
        artifact_path = version_dir / str(metadata["artifact_file"])
        if not artifact_path.is_file():
            raise ArtifactValidationError(f"Artefatto mancante per {version}: {artifact_path}")

        actual_fingerprint = file_fingerprint(artifact_path)
        if actual_fingerprint != metadata["artifact_fingerprint"]:
            raise ArtifactValidationError(
                "Fingerprint dell'artefatto diverso dai metadati: "
                f"atteso {metadata['artifact_fingerprint']}, trovato {actual_fingerprint}"
            )

        schema_metadata = metadata["feature_schema"]
        if schema_metadata.get("fingerprint") != self.feature_schema.fingerprint:
            raise ArtifactValidationError(
                "Schema delle feature cambiato rispetto alla versione "
                f"{version}: fingerprint metadati {schema_metadata.get('fingerprint')}, "
                f"schema corrente {self.feature_schema.fingerprint}"
            )
        if schema_metadata.get("names") != self.feature_schema.names:
            raise ArtifactValidationError(
                f"Ordine o nomi delle feature diversi dallo schema corrente per {version}"
            )

        try:
            pipeline = joblib.load(artifact_path)
        except Exception as error:  # artefatto corrotto o incompatibile
            raise ArtifactValidationError(
                f"Impossibile deserializzare l'artefatto di {version}: {error}"
            ) from error

        if not hasattr(pipeline, "predict_proba"):
            raise ArtifactValidationError(
                f"L'artefatto di {version} non espone predict_proba: non e' una pipeline di classificazione"
            )
        return pipeline, metadata

    def list_versions(self) -> list[dict[str, Any]]:
        versions: list[dict[str, Any]] = []
        for entry in sorted(self.models_dir.iterdir()):
            if not entry.is_dir():
                continue
            metadata_path = entry / "model_metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = self.read_metadata(entry.name)
            metrics = metadata.get("validation_metrics", {})
            versions.append(
                {
                    "model_version": entry.name,
                    "run_id": metadata.get("run_id"),
                    "model_family": metadata.get("model_family"),
                    "created_at": metadata.get("created_at"),
                    "artifact_fingerprint": metadata.get("artifact_fingerprint"),
                    "accuracy": metrics.get("accuracy"),
                    "recall_target_1": metrics.get("recall_target_1"),
                }
            )
        return versions

    def read_pointer(self) -> dict[str, Any]:
        if not self.pointer_path.is_file():
            return {"active_version": None, "previous_version": None, "history": []}
        with self.pointer_path.open("r", encoding="utf-8") as pointer_file:
            return json.load(pointer_file)

    def _write_pointer(self, pointer: dict[str, Any]) -> None:
        _atomic_write_json(self.pointer_path, pointer)

    def activate(
        self,
        version: str,
        actor: str,
        action: str = "activate",
        reason: str = "",
    ) -> dict[str, Any]:
        """Verifica la versione e la rende attiva, registrando la transizione."""
        # La verifica avviene prima della scrittura: un artefatto non coerente non
        # deve mai diventare la versione attiva.
        _, metadata = self.load_version(version)

        pointer = self.read_pointer()
        previous_version = pointer.get("active_version")
        if previous_version == version:
            raise RegistryError(f"La versione {version} e' gia' attiva")

        transition = {
            "timestamp": utc_timestamp(),
            "action": action,
            "actor": actor,
            "from_version": previous_version,
            "to_version": version,
            "run_id": metadata.get("run_id"),
            "artifact_fingerprint": metadata.get("artifact_fingerprint"),
            "feature_schema_fingerprint": metadata.get("feature_schema", {}).get("fingerprint"),
            "dataset_fingerprint": metadata.get("training", {}).get("dataset_fingerprint"),
            "reason": reason,
        }

        pointer = {
            "active_version": version,
            "previous_version": previous_version,
            "updated_at": transition["timestamp"],
            "run_id": transition["run_id"],
            "artifact_fingerprint": transition["artifact_fingerprint"],
            "history": [*pointer.get("history", []), transition],
        }
        self._write_pointer(pointer)
        with self.transitions_path.open("a", encoding="utf-8") as transitions_file:
            transitions_file.write(json.dumps(transition, ensure_ascii=False) + "\n")
        return pointer

    def rollback(self, actor: str, reason: str = "") -> dict[str, Any]:
        """Ripristina la versione precedente, se esiste."""
        pointer = self.read_pointer()
        previous_version = pointer.get("previous_version")
        if not previous_version:
            raise RegistryError("Nessuna versione precedente disponibile per il rollback")
        return self.activate(
            previous_version,
            actor=actor,
            action="rollback",
            reason=reason or "ripristino della versione precedente",
        )
