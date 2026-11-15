"""Controlli del lettore dati e del contratto delle feature."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from manutenzione.config import load_project_config
from manutenzione.contracts import load_feature_schema
from manutenzione.data_loader import (
    DataValidationError,
    load_dataset,
    load_monitoring_requests,
)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _valid_row(feature_schema) -> list[str]:
    row = ["record-1", "train"]
    row.extend("1" if spec.integer else "1.0" for spec in feature_schema.features)
    row.append("0")
    return row


def test_load_real_dataset_counts(feature_schema):
    config = load_project_config()
    bundle = load_dataset(
        config["paths"]["dataset_csv"],
        feature_schema,
        id_column=config["training"]["id_column"],
        target_column=config["training"]["target_column"],
        split_column=config["training"]["split_column"],
    )
    assert bundle.row_count == 1000
    assert bundle.features.shape == (1000, 30)
    assert bundle.class_counts["train"] == {"0": 501, "1": 299}
    assert bundle.class_counts["validation"] == {"0": 125, "1": 75}
    assert len(bundle.dataset_fingerprint) == 64


def test_fingerprint_changes_with_content(tmp_path, feature_schema):
    header = ["record_id", "split", *feature_schema.names, "target"]
    first = _write_csv(tmp_path / "a.csv", header, [_valid_row(feature_schema)])
    second = _write_csv(tmp_path / "b.csv", header, [_valid_row(feature_schema)])
    first_bundle = load_dataset(first, feature_schema)
    second_bundle = load_dataset(second, feature_schema)
    assert first_bundle.dataset_fingerprint == second_bundle.dataset_fingerprint

    second.write_text(second.read_text(encoding="utf-8").replace("1.0", "2.0", 1), encoding="utf-8")
    changed_bundle = load_dataset(second, feature_schema)
    assert changed_bundle.dataset_fingerprint != first_bundle.dataset_fingerprint


def test_empty_file_rejected(tmp_path, feature_schema):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError, match="vuoto"):
        load_dataset(empty_path, feature_schema)


def test_duplicate_record_id_rejected(tmp_path, feature_schema):
    header = ["record_id", "split", *feature_schema.names, "target"]
    path = _write_csv(
        tmp_path / "duplicated.csv",
        header,
        [_valid_row(feature_schema), _valid_row(feature_schema)],
    )
    with pytest.raises(DataValidationError, match="record_id ripetuto"):
        load_dataset(path, feature_schema)


def test_duplicate_columns_rejected(tmp_path, feature_schema):
    header = ["record_id", "split", *feature_schema.names, "target", feature_schema.names[0]]
    row = _valid_row(feature_schema) + ["1.0"]
    path = _write_csv(tmp_path / "columns.csv", header, [row])
    with pytest.raises(DataValidationError, match="Colonne duplicate"):
        load_dataset(path, feature_schema)


def test_non_finite_value_rejected(tmp_path, feature_schema):
    header = ["record_id", "split", *feature_schema.names, "target"]
    row = _valid_row(feature_schema)
    row[2] = "nan"
    path = _write_csv(tmp_path / "nan.csv", header, [row])
    with pytest.raises(DataValidationError, match="non finito"):
        load_dataset(path, feature_schema)


def test_unexpected_target_and_split_rejected(tmp_path, feature_schema):
    header = ["record_id", "split", *feature_schema.names, "target"]
    row = _valid_row(feature_schema)
    row[1] = "test"
    row[-1] = "2"
    path = _write_csv(tmp_path / "splits.csv", header, [row])
    with pytest.raises(DataValidationError) as error:
        load_dataset(path, feature_schema)
    message = str(error.value)
    assert "split inatteso" in message
    assert "target inatteso" in message


def test_monitoring_requests_are_not_training_data(feature_schema):
    config = load_project_config()
    requests = load_monitoring_requests(config["paths"]["requests_csv"])
    assert len(requests) == 4
    statuses = [request.expected_status for request in requests]
    assert statuses == [200, 200, 200, 422]
    assert all("target" not in request.payload for request in requests)
    assert feature_schema.validate_vector(requests[3].payload["features"]) != []


def test_schema_order_is_the_api_order(feature_schema):
    assert feature_schema.count == 30
    assert feature_schema.names[0] == "temperatura_cuscinetto_media_c"
    assert feature_schema.names[-1] == "livello_lubrificante_pct"
    assert [spec.position for spec in feature_schema.features] == list(range(1, 31))
