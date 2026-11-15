"""Contratto condiviso delle feature.

Lo schema delle feature e' descritto in config/feature_schema.json: nome,
posizione, unita', intervallo della simulazione e tipo intero per i conteggi.
Lo stesso file viene usato dal lettore dati, dal training e dal servizio, cosi'
l'ordine delle feature non puo' divergere tra le fasi.

Gli intervalli sono informativi (servono al monitoraggio): il servizio rifiuta
solo input strutturalmente non validi (numero di feature, valori non finiti,
conteggi non interi).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class SchemaValidationError(Exception):
    """Lo schema delle feature non rispetta il contratto atteso."""


def file_fingerprint(path: str | Path) -> str:
    """Impronta SHA-256 dei byte del file, usata per identificare dati e artefatti."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as binary_file:
        for chunk in iter(lambda: binary_file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureSpec:
    """Descrizione di una singola feature."""

    name: str
    position: int
    unit: str
    minimum: float
    maximum: float
    integer: bool
    description: str


@dataclass(frozen=True)
class FeatureSchema:
    """Contratto completo delle feature di input."""

    version: str
    features: tuple[FeatureSpec, ...]
    fingerprint: str

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.features]

    @property
    def count(self) -> int:
        return len(self.features)

    def validate_vector(self, values: Sequence[float]) -> list[str]:
        """Restituisce l'elenco dei problemi dell'input, vuoto se e' valido."""
        problems: list[str] = []
        if len(values) != self.count:
            problems.append(
                f"attese {self.count} feature, ricevute {len(values)}"
            )
            return problems

        for spec, value in zip(self.features, values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"{spec.name}: valore non numerico ({value!r})")
                continue
            if not math.isfinite(float(value)):
                problems.append(f"{spec.name}: valore non finito ({value!r})")
                continue
            if spec.integer and float(value) != int(value):
                problems.append(f"{spec.name}: atteso un conteggio intero, ricevuto {value!r}")
        return problems


def load_feature_schema(path: str | Path) -> FeatureSchema:
    """Carica e verifica il file JSON delle feature."""
    schema_path = Path(path)
    if not schema_path.is_file():
        raise SchemaValidationError(f"Schema feature non trovato: {schema_path}")

    raw_bytes = schema_path.read_bytes()
    payload = json.loads(raw_bytes)
    entries = payload.get("features")
    if not isinstance(entries, list) or not entries:
        raise SchemaValidationError("Lo schema non contiene la lista 'features'")

    features: list[FeatureSpec] = []
    seen_names: set[str] = set()
    for expected_position, entry in enumerate(entries, start=1):
        try:
            spec = FeatureSpec(
                name=str(entry["name"]),
                position=int(entry["position"]),
                unit=str(entry["unit"]),
                minimum=float(entry["min"]),
                maximum=float(entry["max"]),
                integer=bool(entry["integer"]),
                description=str(entry["description"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SchemaValidationError(
                f"Feature alla posizione {expected_position} incompleta: {error}"
            ) from error

        if spec.position != expected_position:
            raise SchemaValidationError(
                f"Posizione incoerente per {spec.name}: attesa {expected_position}, trovata {spec.position}"
            )
        if spec.name in seen_names:
            raise SchemaValidationError(f"Feature duplicata nello schema: {spec.name}")
        if spec.minimum > spec.maximum:
            raise SchemaValidationError(f"Intervallo invertito per {spec.name}")
        seen_names.add(spec.name)
        features.append(spec)

    return FeatureSchema(
        version=str(payload.get("schema_version", "sconosciuta")),
        features=tuple(features),
        fingerprint=hashlib.sha256(raw_bytes).hexdigest(),
    )
