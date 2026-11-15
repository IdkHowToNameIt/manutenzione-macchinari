"""Lettura e validazione dei CSV del progetto.

Il lettore e' volutamente esplicito: legge il CSV con il modulo standard,
verifica i controlli richiesti dalla consegna e restituisce a training e
valutazione solo dati gia' puliti, insieme alle informazioni di identita' del
file (fingerprint, righe, conteggio classi per split).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import FeatureSchema, file_fingerprint

EXPECTED_SPLITS = ("train", "validation")
EXPECTED_TARGETS = (0, 1)
MAX_REPORTED_PROBLEMS = 12


class DataValidationError(Exception):
    """Il file dati non rispetta il contratto; il messaggio elenca i controlli falliti."""


@dataclass
class DatasetBundle:
    """Dati letti e validati, pronti per training e valutazione."""

    feature_names: list[str]
    features: np.ndarray
    targets: np.ndarray
    splits: np.ndarray
    record_ids: list[str]
    dataset_path: str
    dataset_fingerprint: str
    class_counts: dict[str, dict[str, int]]

    @property
    def row_count(self) -> int:
        return int(self.features.shape[0])

    def split_mask(self, split_name: str) -> np.ndarray:
        return self.splits == split_name


@dataclass
class MonitoringRequest:
    """Richiesta operativa del CSV di monitoraggio (solo input, nessuna label)."""

    request_id: str
    model_version: str
    payload: dict[str, Any]
    expected_status: int


def _raise_if_problems(problems: list[str], file_label: str) -> None:
    if not problems:
        return
    shown = problems[:MAX_REPORTED_PROBLEMS]
    extra = len(problems) - len(shown)
    detail = "\n".join(f"- {problem}" for problem in shown)
    if extra > 0:
        detail += f"\n- ... altri {extra} problemi"
    raise DataValidationError(f"Controlli falliti su {file_label}:\n{detail}")


def load_dataset(
    dataset_path: str | Path,
    feature_schema: FeatureSchema,
    id_column: str = "record_id",
    target_column: str = "target",
    split_column: str = "split",
) -> DatasetBundle:
    """Legge il dataset e applica i controlli bloccanti del checkpoint S1."""
    path = Path(dataset_path)
    if not path.is_file():
        raise DataValidationError(f"File dataset non trovato: {path}")
    if path.stat().st_size == 0:
        raise DataValidationError(f"File dataset vuoto: {path}")

    problems: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        header = reader.fieldnames or []
        if not header:
            raise DataValidationError(f"Intestazione assente nel dataset: {path}")

        duplicate_columns = sorted({name for name in header if header.count(name) > 1})
        if duplicate_columns:
            raise DataValidationError(
                f"Colonne duplicate nel dataset: {', '.join(duplicate_columns)}"
            )

        required_columns = [id_column, target_column, split_column, *feature_schema.names]
        missing_columns = [name for name in required_columns if name not in header]
        if missing_columns:
            raise DataValidationError(
                f"Colonne mancanti nel dataset: {', '.join(missing_columns)}"
            )
        if target_column in feature_schema.names or id_column in feature_schema.names:
            raise DataValidationError(
                "Target o record_id compaiono tra le feature: rischio di leakage"
            )

        rows: list[dict[str, str]] = list(reader)

    if not rows:
        raise DataValidationError(f"Il dataset non contiene righe di dati: {path}")

    feature_matrix = np.empty((len(rows), feature_schema.count), dtype=np.float64)
    targets = np.empty(len(rows), dtype=np.int8)
    splits = np.empty(len(rows), dtype=object)
    record_ids: list[str] = []
    seen_ids: set[str] = set()
    class_counts: dict[str, dict[str, int]] = {}

    for row_index, row in enumerate(rows, start=2):  # 2 = prima riga dati nel CSV
        record_id = (row.get(id_column) or "").strip()
        if not record_id:
            problems.append(f"riga {row_index}: record_id mancante")
        elif record_id in seen_ids:
            problems.append(f"riga {row_index}: record_id ripetuto ({record_id})")
        else:
            seen_ids.add(record_id)
            record_ids.append(record_id)

        split_value = (row.get(split_column) or "").strip()
        if split_value not in EXPECTED_SPLITS:
            problems.append(
                f"riga {row_index}: split inatteso ({split_value!r}), attesi {EXPECTED_SPLITS}"
            )
        splits[row_index - 2] = split_value

        try:
            target_value = int(str(row.get(target_column, "")).strip())
        except ValueError:
            problems.append(f"riga {row_index}: target non intero ({row.get(target_column)!r})")
            target_value = -1
        if target_value not in EXPECTED_TARGETS:
            problems.append(f"riga {row_index}: target inatteso ({target_value})")
        targets[row_index - 2] = target_value

        for position, feature_name in enumerate(feature_schema.names):
            raw_value = row.get(feature_name)
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                problems.append(
                    f"riga {row_index}, {feature_name}: valore non numerico ({raw_value!r})"
                )
                feature_matrix[row_index - 2, position] = math.nan
                continue
            if not math.isfinite(numeric_value):
                problems.append(
                    f"riga {row_index}, {feature_name}: valore non finito ({numeric_value!r})"
                )
            feature_matrix[row_index - 2, position] = numeric_value

        split_counts = class_counts.setdefault(split_value, {"0": 0, "1": 0})
        if target_value in EXPECTED_TARGETS:
            split_counts[str(target_value)] += 1

    _raise_if_problems(problems, str(path))

    return DatasetBundle(
        feature_names=feature_schema.names,
        features=feature_matrix,
        targets=targets,
        splits=splits,
        record_ids=record_ids,
        dataset_path=str(path.resolve()),
        dataset_fingerprint=file_fingerprint(path),
        class_counts=class_counts,
    )


def load_monitoring_requests(requests_path: str | Path) -> list[MonitoringRequest]:
    """Legge il CSV delle richieste operative, senza trattarlo come dataset di training."""
    path = Path(requests_path)
    if not path.is_file():
        raise DataValidationError(f"File richieste non trovato: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as requests_file:
        reader = csv.DictReader(requests_file)
        header = reader.fieldnames or []
        required_columns = ["request_id", "model_version", "payload_json", "expected_status"]
        missing_columns = [name for name in required_columns if name not in header]
        if missing_columns:
            raise DataValidationError(
                f"Colonne mancanti nel file richieste: {', '.join(missing_columns)}"
            )
        rows = list(reader)

    if not rows:
        raise DataValidationError(f"Il file richieste non contiene righe: {path}")

    requests: list[MonitoringRequest] = []
    for row_index, row in enumerate(rows, start=2):
        try:
            payload = json.loads(row["payload_json"])
            expected_status = int(row["expected_status"])
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise DataValidationError(
                f"Riga {row_index} del file richieste non valida: {error}"
            ) from error
        requests.append(
            MonitoringRequest(
                request_id=(row["request_id"] or "").strip(),
                model_version=(row["model_version"] or "").strip(),
                payload=payload,
                expected_status=expected_status,
            )
        )
    return requests


def describe_dataset(bundle: DatasetBundle, feature_schema: FeatureSchema) -> dict[str, Any]:
    """Profilo del dataset usato nelle evidenze e nella relazione."""
    train_mask = bundle.split_mask("train")
    train_features = bundle.features[train_mask]

    feature_statistics = []
    for position, spec in enumerate(feature_schema.features):
        column = train_features[:, position]
        feature_statistics.append(
            {
                "position": spec.position,
                "name": spec.name,
                "unit": spec.unit,
                "integer": spec.integer,
                "simulation_range": [spec.minimum, spec.maximum],
                "train_min": float(np.min(column)),
                "train_max": float(np.max(column)),
                "train_mean": float(np.mean(column)),
                "train_std": float(np.std(column)),
            }
        )

    return {
        "dataset_path": bundle.dataset_path,
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "row_count": bundle.row_count,
        "feature_count": len(bundle.feature_names),
        "feature_schema_version": feature_schema.version,
        "feature_schema_fingerprint": feature_schema.fingerprint,
        "class_counts": bundle.class_counts,
        "feature_statistics": feature_statistics,
    }
