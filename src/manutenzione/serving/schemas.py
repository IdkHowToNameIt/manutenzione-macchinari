"""Schemi Pydantic del contratto HTTP.

La richiesta di predict contiene una lista `features` con esattamente le 30
feature nell'ordine dello schema condiviso. Un elenco incompleto, un valore non
finito o un conteggio non intero producono 422 prima di chiamare il modello.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts import FeatureSchema


def build_predict_request_model(feature_schema: FeatureSchema) -> type[BaseModel]:
    """Costruisce il modello di richiesta a partire dal contratto delle feature."""
    expected_count = feature_schema.count

    class PredictRequest(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={
                "examples": [{"features": [0.0] * expected_count}],
            },
        )

        features: list[float] = Field(
            min_length=expected_count,
            max_length=expected_count,
            description=(
                f"Le {expected_count} feature numeriche nell'ordine di "
                "config/feature_schema.json"
            ),
        )

        @field_validator("features")
        @classmethod
        def validate_feature_values(cls, values: list[float]) -> list[float]:
            problems = feature_schema.validate_vector(values)
            if problems:
                raise ValueError("; ".join(problems))
            return values

    return PredictRequest


class PredictResponse(BaseModel):
    prediction: int = Field(description="1 = guasto con intervento atteso entro 7 giorni, 0 = no")
    probability_target_1: float = Field(description="Probabilita' stimata della classe 1")
    decision_threshold: float = Field(description="Soglia usata dalla versione attiva")
    model_version: str
    run_id: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    run_id: str
    artifact_fingerprint: str
    feature_count: int
    decision_threshold: float
    selection_source: str
    started_at: str


class MetricsResponse(BaseModel):
    window_minutes: float
    generated_at: str
    requests_total: int
    errors_total: int
    error_rate: float
    p95_ms: float | None
    p50_ms: float | None
    by_version: dict[str, dict[str, Any]]
    note: str


class ActivationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    actor: str = Field(default="amministratore", min_length=1)
    reason: str = Field(default="attivazione manuale richiesta dall'amministratore")


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = Field(default="amministratore", min_length=1)
    reason: str = Field(default="ripristino della versione precedente")


class ActivationResponse(BaseModel):
    action: str
    previous_version: str | None
    active_version: str
    run_id: str
    artifact_fingerprint: str
    decision_threshold: float
    served_version: str
    transition_timestamp: str


class VersionEntry(BaseModel):
    model_version: str
    run_id: str | None
    model_family: str | None
    created_at: str | None
    artifact_fingerprint: str | None
    accuracy: float | None
    recall_target_1: float | None


class VersionsResponse(BaseModel):
    active_version: str | None
    previous_version: str | None
    versions: list[VersionEntry]


class SchemaResponse(BaseModel):
    schema_version: str
    fingerprint: str
    feature_count: int
    feature_names: list[str]
