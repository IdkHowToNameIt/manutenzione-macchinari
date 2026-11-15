"""API FastAPI del servizio di manutenzione predittiva.

Avvio locale:
    python -m manutenzione.serving.app

Variabili d'ambiente:
    MANUTENZIONE_HOST, MANUTENZIONE_PORT   indirizzo di ascolto
    MANUTENZIONE_FORCED_VERSION            carica una versione specifica per una prova
    MANUTENZIONE_ADMIN_TOKEN               token per /admin/* (default solo locale)
"""
import hmac
import os
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..config import load_project_config
from ..contracts import FeatureSchema, load_feature_schema
from ..registry import ArtifactValidationError, ModelRegistry, RegistryError
from .model_service import ModelService, ServingError
from .schemas import (
    ActivationRequest,
    ActivationResponse,
    HealthResponse,
    MetricsResponse,
    PredictResponse,
    RollbackRequest,
    SchemaResponse,
    VersionEntry,
    VersionsResponse,
    build_predict_request_model,
)
from .telemetry import TelemetryCollector

ADMIN_TOKEN_ENV = "MANUTENZIONE_ADMIN_TOKEN"
DEFAULT_LOCAL_ADMIN_TOKEN = "local-admin-token"


def create_app(
    config: dict | None = None,
    feature_schema: FeatureSchema | None = None,
    registry: ModelRegistry | None = None,
    forced_version: str | None = None,
) -> FastAPI:
    """Costruisce l'applicazione con modello, telemetria e contratti."""
    config = config or load_project_config()
    feature_schema = feature_schema or load_feature_schema(config["paths"]["feature_schema"])
    registry = registry or ModelRegistry(config["paths"]["models_dir"], feature_schema)

    forced_version = forced_version or os.environ.get("MANUTENZIONE_FORCED_VERSION") or None
    model_service = ModelService(registry, forced_version=forced_version)
    telemetry = TelemetryCollector(
        Path(config["paths"]["logs_dir"]) / "serving.jsonl",
        window_minutes=float(config["service"]["metric_window_minutes"]),
    )

    app = FastAPI(
        title="Manutenzione predittiva - API",
        version="1.0.0",
        description=(
            "Stima la probabilita' che un macchinario richieda un intervento per guasto "
            "entro 7 giorni, a fine turno."
        ),
    )
    app.state.settings = config
    app.state.feature_schema = feature_schema
    app.state.registry = registry
    app.state.model_service = model_service
    app.state.telemetry = telemetry

    predict_request_model = build_predict_request_model(feature_schema)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, error: RequestValidationError):
        """422 con soli messaggi: gli input non validi (anche NaN) non vengono rimandati indietro."""
        cleaned_errors = [
            {
                "location": list(item.get("loc", [])),
                "message": item.get("msg", "valore non valido"),
                "type": item.get("type", "value_error"),
            }
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "detail": cleaned_errors,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    def require_admin_token(
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> None:
        expected_token = os.environ.get(ADMIN_TOKEN_ENV, DEFAULT_LOCAL_ADMIN_TOKEN)
        if not x_admin_token or not hmac.compare_digest(x_admin_token, expected_token):
            raise HTTPException(
                status_code=401, detail="token amministrativo mancante o non valido"
            )

    @app.middleware("http")
    async def telemetry_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started_at = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # Il dettaglio dell'eccezione resta nei log del processo; al client
            # non esponiamo messaggi interni.
            response = JSONResponse(
                {"detail": "errore interno del servizio", "request_id": request_id},
                status_code=500,
            )

        duration_ms = (time.perf_counter() - started_at) * 1000.0
        state = model_service.state()
        status_code = response.status_code
        if status_code < 400:
            outcome = "success"
        elif status_code < 500:
            outcome = "client_error"
        else:
            outcome = "server_error"

        telemetry.log_request(
            {
                "timestamp": None,  # impostato dal collector in UTC
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 3),
                "model_version": state["model_version"],
                "run_id": state["run_id"],
                "outcome": outcome,
            },
            count_as_prediction=(request.method == "POST" and request.url.path == "/predict"),
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Model-Version"] = state["model_version"]
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Stato del servizio con versione, run ID e fingerprint dell'artefatto."""
        state = model_service.state()
        return HealthResponse(status="ok", started_at=model_service.started_at, **state)

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: predict_request_model, request: Request) -> PredictResponse:
        """Predizione per un macchinario a fine turno."""
        result = model_service.predict(payload.features)
        return PredictResponse(request_id=request.state.request_id, **result)

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        """Metriche della finestra corrente sulle sole richieste di predizione."""
        return MetricsResponse(**telemetry.snapshot())

    @app.get("/versions", response_model=VersionsResponse)
    def versions() -> VersionsResponse:
        """Versioni registrate e stato attivo."""
        pointer = registry.read_pointer()
        return VersionsResponse(
            active_version=pointer.get("active_version"),
            previous_version=pointer.get("previous_version"),
            versions=[VersionEntry(**entry) for entry in registry.list_versions()],
        )

    @app.get("/schema", response_model=SchemaResponse)
    def schema() -> SchemaResponse:
        """Schema atteso dalla richiesta di predizione."""
        return SchemaResponse(
            schema_version=feature_schema.version,
            fingerprint=feature_schema.fingerprint,
            feature_count=feature_schema.count,
            feature_names=feature_schema.names,
        )

    @app.post(
        "/admin/activate",
        response_model=ActivationResponse,
        dependencies=[Depends(require_admin_token)],
    )
    def activate(payload: ActivationRequest) -> ActivationResponse:
        """Attiva una versione verificata e ricarica l'artefatto servito."""
        try:
            summary = model_service.activate(payload.version, payload.actor, payload.reason)
        except (ArtifactValidationError, RegistryError, ServingError) as error:
            raise HTTPException(
                status_code=409, detail=f"attivazione rifiutata: {error}"
            ) from error
        return ActivationResponse(**summary)

    @app.post(
        "/admin/rollback",
        response_model=ActivationResponse,
        dependencies=[Depends(require_admin_token)],
    )
    def rollback(payload: RollbackRequest | None = None) -> ActivationResponse:
        """Ripristina la versione precedente e ricarica l'artefatto servito."""
        request_body = payload or RollbackRequest()
        try:
            summary = model_service.rollback(request_body.actor, request_body.reason)
        except (ArtifactValidationError, RegistryError, ServingError) as error:
            raise HTTPException(
                status_code=409, detail=f"rollback rifiutato: {error}"
            ) from error
        return ActivationResponse(**summary)

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "service": "manutenzione-predittiva",
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
        }

    return app


def main() -> None:
    config = load_project_config()
    feature_schema = load_feature_schema(config["paths"]["feature_schema"])
    host = os.environ.get("MANUTENZIONE_HOST", config["service"]["host"])
    port = int(os.environ.get("MANUTENZIONE_PORT", config["service"]["port"]))
    application = create_app(config=config, feature_schema=feature_schema)
    uvicorn.run(application, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
