"""Controlli del servizio: contratto, telemetria, attivazione e rollback."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from manutenzione.config import load_project_config
from manutenzione.data_loader import load_monitoring_requests
from manutenzione.serving.app import create_app

ADMIN_HEADERS = {"X-Admin-Token": "local-admin-token"}


def _valid_payload() -> dict:
    config = load_project_config()
    requests = load_monitoring_requests(config["paths"]["requests_csv"])
    valid_request = next(request for request in requests if request.expected_status == 200)
    return valid_request.payload


@pytest.fixture()
def client_factory(project_config, feature_schema, test_registry):
    def build(forced_version: str | None = None) -> TestClient:
        application = create_app(
            config=project_config,
            feature_schema=feature_schema,
            registry=test_registry,
            forced_version=forced_version,
        )
        return TestClient(application)

    return build


def test_health_exposes_version_and_run(client_factory):
    with client_factory() as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "v1"
    assert body["run_id"] == "run-v1"
    assert body["feature_count"] == 30
    assert body["selection_source"] == "active_version"


def test_predict_returns_identity_and_request_id(client_factory):
    with client_factory() as client:
        response = client.post("/predict", json=_valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "v1"
    assert body["run_id"] == "run-v1"
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_target_1"] <= 1.0
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert response.headers["X-Model-Version"] == "v1"


def test_incomplete_payload_is_rejected_before_model(client_factory, project_config):
    payload = {"features": _valid_payload()["features"][:-1]}
    with client_factory() as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422

    log_path = Path(project_config["paths"]["logs_dir"]) / "serving.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    rejected = [event for event in events if event["status_code"] == 422]
    assert rejected, "la richiesta respinta deve comparire nel log"
    assert rejected[-1]["request_id"] == response.headers["X-Request-ID"]
    assert "features" not in rejected[-1], "i log non devono contenere il payload grezzo"


def test_non_finite_and_extra_fields_are_rejected(client_factory):
    payload = {"features": _valid_payload()["features"]}
    payload["features"][0] = float("nan")
    # Il client non invia NaN come JSON standard: passiamo il testo grezzo per
    # verificare che il servizio rifiuti esplicitamente il valore non finito.
    raw_body = json.dumps(payload, allow_nan=True)
    with client_factory() as client:
        nan_response = client.post(
            "/predict", content=raw_body, headers={"Content-Type": "application/json"}
        )
        extra_response = client.post(
            "/predict", json={**_valid_payload(), "target": 1}
        )
    assert nan_response.status_code == 422
    assert extra_response.status_code == 422


def test_metrics_count_only_predictions(client_factory):
    with client_factory() as client:
        client.get("/health")
        client.post("/predict", json=_valid_payload())
        client.post("/predict", json={"features": [0.0] * 29})
        metrics = client.get("/metrics").json()
        client.get("/metrics")
    assert metrics["requests_total"] == 2
    assert metrics["errors_total"] == 1
    assert metrics["error_rate"] == 0.5
    assert metrics["p95_ms"] is not None


def test_admin_requires_token(client_factory):
    with client_factory() as client:
        missing = client.post("/admin/activate", json={"version": "v2"})
        wrong = client.post(
            "/admin/activate", json={"version": "v2"}, headers={"X-Admin-Token": "errato"}
        )
    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_activation_changes_served_artifact_then_rollback(client_factory, project_config):
    with client_factory() as client:
        activation = client.post(
            "/admin/activate",
            json={"version": "v2", "actor": "test", "reason": "prova"},
            headers=ADMIN_HEADERS,
        )
        assert activation.status_code == 200
        activated = activation.json()
        assert activated["active_version"] == "v2"
        assert activated["served_version"] == "v2"

        prediction = client.post("/predict", json=_valid_payload()).json()
        assert prediction["model_version"] == "v2"
        assert prediction["run_id"] == "run-v2"

        rollback = client.post("/admin/rollback", headers=ADMIN_HEADERS)
        assert rollback.status_code == 200
        assert rollback.json()["active_version"] == "v1"

        after_rollback = client.post("/predict", json=_valid_payload()).json()
        assert after_rollback["model_version"] == "v1"
        assert after_rollback["run_id"] == "run-v1"

    pointer = json.loads(
        (Path(project_config["paths"]["models_dir"]) / "active_version.json").read_text(
            encoding="utf-8"
        )
    )
    assert pointer["active_version"] == "v1"
    assert pointer["previous_version"] == "v2"


def test_activation_of_unknown_version_is_rejected(client_factory):
    with client_factory() as client:
        response = client.post(
            "/admin/activate", json={"version": "v9"}, headers=ADMIN_HEADERS
        )
    assert response.status_code == 409
    assert "rifiutata" in response.json()["detail"]


def test_forced_version_does_not_change_active_state(client_factory, project_config):
    with client_factory(forced_version="v2") as client:
        health = client.get("/health").json()
        activation = client.post(
            "/admin/activate", json={"version": "v1"}, headers=ADMIN_HEADERS
        )
    assert health["model_version"] == "v2"
    assert health["selection_source"] == "forced_version"
    assert activation.status_code == 409

    pointer = json.loads(
        (Path(project_config["paths"]["models_dir"]) / "active_version.json").read_text(
            encoding="utf-8"
        )
    )
    assert pointer["active_version"] == "v1"
