"""Gestione del modello caricato dal servizio.

Il servizio carica la versione indicata da models/active_version.json. In una
istanza di prova la versione puo' essere forzata da variabile d'ambiente, cosi'
la probe misura un artefatto preciso senza cambiare lo stato attivo.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..registry import ModelRegistry, RegistryError


class ServingError(Exception):
    """Operazione non ammessa sul servizio (per esempio in modalita' forzata)."""


class ModelService:
    """Mantiene in memoria pipeline, soglia e metadati della versione servita."""

    def __init__(self, registry: ModelRegistry, forced_version: str | None = None):
        self.registry = registry
        self.forced_version = forced_version
        self._lock = threading.Lock()
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if forced_version:
            selection_source = "forced_version"
            version = forced_version
        else:
            selection_source = "active_version"
            version = registry.read_pointer().get("active_version")
            if not version:
                raise RegistryError(
                    "Nessuna versione attiva: eseguire prima il training della baseline"
                )
        self.selection_source = selection_source
        self.model_version, self.pipeline, self.metadata = self._load(version)
        self.decision_threshold = self._read_threshold(self.metadata)

    @staticmethod
    def _read_threshold(metadata: dict[str, Any]) -> float:
        if "decision_threshold" in metadata:
            return float(metadata["decision_threshold"])
        return float(metadata.get("validation_metrics", {}).get("decision_threshold", 0.5))

    def _load(self, version: str) -> tuple[str, Any, dict[str, Any]]:
        pipeline, metadata = self.registry.load_version(version)
        return version, pipeline, metadata

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "model_version": self.model_version,
                "run_id": self.metadata.get("run_id", ""),
                "artifact_fingerprint": self.metadata.get("artifact_fingerprint", ""),
                "feature_count": int(self.metadata.get("feature_schema", {}).get("count", 0)),
                "decision_threshold": self.decision_threshold,
                "selection_source": self.selection_source,
            }

    def predict(self, features: list[float]) -> dict[str, Any]:
        """Esegue una predizione con l'artefatto attualmente in memoria."""
        with self._lock:
            pipeline = self.pipeline
            model_version = self.model_version
            run_id = self.metadata.get("run_id", "")
            threshold = self.decision_threshold

        feature_matrix = np.asarray(features, dtype=np.float64).reshape(1, -1)
        probability = float(pipeline.predict_proba(feature_matrix)[0, 1])
        return {
            "prediction": int(probability >= threshold),
            "probability_target_1": probability,
            "decision_threshold": threshold,
            "model_version": model_version,
            "run_id": run_id,
        }

    def _ensure_not_forced(self) -> None:
        if self.forced_version:
            raise ServingError(
                "Servizio avviato con versione forzata: le transizioni sono disabilitate"
            )

    def activate(self, version: str, actor: str, reason: str) -> dict[str, Any]:
        """Rende attiva la versione dopo la verifica dell'artefatto e ricarica il servizio."""
        with self._lock:
            self._ensure_not_forced()
            pointer = self.registry.activate(version, actor=actor, action="activate", reason=reason)
            # Il ricaricamento avviene solo dopo che il registro ha accettato la transizione.
            self.model_version, self.pipeline, self.metadata = self._load(version)
            self.decision_threshold = self._read_threshold(self.metadata)
            return self._transition_summary("activate", pointer)

    def rollback(self, actor: str, reason: str) -> dict[str, Any]:
        """Ripristina la versione precedente e ricarica il servizio."""
        with self._lock:
            self._ensure_not_forced()
            pointer = self.registry.rollback(actor=actor, reason=reason)
            restored_version = pointer["active_version"]
            self.model_version, self.pipeline, self.metadata = self._load(restored_version)
            self.decision_threshold = self._read_threshold(self.metadata)
            return self._transition_summary("rollback", pointer)

    def _transition_summary(self, action: str, pointer: dict[str, Any]) -> dict[str, Any]:
        last_transition = pointer["history"][-1]
        return {
            "action": action,
            "previous_version": last_transition["from_version"],
            "active_version": pointer["active_version"],
            "run_id": last_transition["run_id"] or "",
            "artifact_fingerprint": last_transition["artifact_fingerprint"] or "",
            "decision_threshold": self.decision_threshold,
            "served_version": self.model_version,
            "transition_timestamp": last_transition["timestamp"],
        }
