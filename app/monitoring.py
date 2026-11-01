"""Journalisation structurée (JSONL) des appels de prédiction pour le monitoring.

Un fichier par jour dans `settings.log_dir` : une ligne JSON par appel avec
inputs, outputs, latence et statut. Aucun identifiant client n'est stocké
(conformité RGPD) : seules les features du modèle sont conservées.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()

def log_prediction(
    log_dir: Path,
    *,
    inputs: dict | None,
    output: dict | None,
    latency_ms: float,
    status_code: int,
    error: str | None = None,
) -> None:
    """Ajoute un enregistrement JSON au fichier de log du jour."""
    now = datetime.now(timezone.utc)
    record = {
        "timestamp": now.isoformat(),
        "endpoint": "/predict",
        "status_code": status_code,
        "latency_ms": round(latency_ms, 3),
        "inputs": inputs,
        "output": output,
        "error": error,
    }
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"predictions_{now:%Y-%m-%d}.jsonl"
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
