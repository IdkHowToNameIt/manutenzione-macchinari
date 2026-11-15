"""Telemetria del servizio: log delle richieste e metriche a finestra.

Le richieste di predizione vengono registrate in un file JSONL con timestamp,
request ID, versione, esito HTTP e durata, senza il payload grezzo. Le metriche
sono calcolate su una finestra esplicita e riguardano solo POST /predict:
health, metrics e gli endpoint amministrativi non entrano nel conteggio.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


class TelemetryCollector:
    """Raccoglie eventi di richiesta e calcola le metriche della finestra."""

    def __init__(self, log_path: str | Path, window_minutes: float = 15.0):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.window = timedelta(minutes=float(window_minutes))
        self._samples: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()

    def log_request(self, event: dict[str, Any], count_as_prediction: bool) -> None:
        """Scrive l'evento nel log JSONL e, se e' una predizione, lo aggiunge alle metriche."""
        timestamp = event.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        event["timestamp"] = timestamp

        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(event, ensure_ascii=False) + "\n")

            if count_as_prediction:
                self._samples.append(
                    {
                        "timestamp": datetime.fromisoformat(timestamp),
                        "duration_ms": float(event["duration_ms"]),
                        "status_code": int(event["status_code"]),
                        "model_version": str(event.get("model_version") or "sconosciuta"),
                    }
                )
                self._prune_locked()

    def _prune_locked(self) -> None:
        cutoff = datetime.now(timezone.utc) - self.window
        while self._samples and self._samples[0]["timestamp"] < cutoff:
            self._samples.popleft()

    def snapshot(self) -> dict[str, Any]:
        """Metriche della finestra corrente, calcolate al momento della richiesta."""
        with self._lock:
            self._prune_locked()
            samples = list(self._samples)

        if samples:
            durations = np.array([sample["duration_ms"] for sample in samples], dtype=float)
            statuses = np.array([sample["status_code"] for sample in samples], dtype=int)
            p50_ms = float(np.percentile(durations, 50))
            p95_ms = float(np.percentile(durations, 95))
        else:
            p50_ms = None
            p95_ms = None
            statuses = np.array([], dtype=int)

        requests_total = len(samples)
        errors_total = int(np.sum(statuses >= 400))
        error_rate = float(errors_total / requests_total) if requests_total else 0.0

        by_version: dict[str, dict[str, Any]] = {}
        for sample in samples:
            bucket = by_version.setdefault(
                sample["model_version"],
                {"count": 0, "errors": 0, "durations": []},
            )
            bucket["count"] += 1
            bucket["durations"].append(sample["duration_ms"])
            if sample["status_code"] >= 400:
                bucket["errors"] += 1
        for bucket in by_version.values():
            durations = bucket.pop("durations")
            bucket["p95_ms"] = round(float(np.percentile(durations, 95)), 3)

        return {
            "window_minutes": self.window.total_seconds() / 60,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "requests_total": requests_total,
            "errors_total": errors_total,
            "error_rate": round(error_rate, 6),
            "p95_ms": round(p95_ms, 3) if p95_ms is not None else None,
            "p50_ms": round(p50_ms, 3) if p50_ms is not None else None,
            "by_version": by_version,
            "note": (
                "Finestra scorrevole su POST /predict; errori = stato HTTP >= 400; "
                "p95 con interpolazione lineare (numpy.percentile)."
            ),
        }
