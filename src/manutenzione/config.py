"""Caricamento della configurazione di progetto.

La configurazione sta in config/project.yaml e contiene i percorsi dei CSV, i
parametri del servizio e le soglie del gate. I percorsi sono relativi alla
radice del progetto (la cartella che contiene config/) e vengono restituiti
gia' risolti come Path assoluti, cosi' gli script funzionano da qualsiasi
directory di lavoro.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "project.yaml"


class ConfigurationError(Exception):
    """Configurazione mancante o non valida."""


@lru_cache(maxsize=4)
def load_project_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Legge il file YAML e restituisce la configurazione con i percorsi risolti."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigurationError(f"File di configurazione non trovato: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ConfigurationError(f"La configurazione non e' un oggetto YAML valido: {path}")
    for required_section in ("project", "paths", "training", "service", "gate"):
        if required_section not in config:
            raise ConfigurationError(f"Sezione mancante nella configurazione: {required_section}")

    project_root = path.resolve().parent.parent
    resolved_paths: dict[str, Any] = {}
    for key, value in config["paths"].items():
        raw_value = str(value)
        if key == "mlflow_tracking_uri" and "://" not in raw_value:
            # Un percorso semplice diventa un database SQLite assoluto: MLflow 3
            # non accetta piu' il backend a sola cartella per le nuove run.
            database_path = (project_root / raw_value).resolve()
            resolved_paths[key] = f"sqlite:///{database_path.as_posix()}"
        else:
            # Percorsi di file e cartelle: assoluti per non dipendere dalla
            # directory da cui si avvia lo script.
            resolved_paths[key] = (project_root / raw_value).resolve()

    config["paths"] = resolved_paths
    config["project_root"] = str(project_root)
    return config


def feature_schema_path(config: dict[str, Any]) -> Path:
    """Percorso del contratto delle feature."""
    return Path(config["paths"]["feature_schema"])
