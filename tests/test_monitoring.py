"""Tests du logging de production et des alertes opérationnelles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from app.monitoring import log_prediction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.monitor import load_logs, operational_alerts  # noqa: E402

from tests.test_api import VALID_PAYLOAD  # noqa: E402


def _read_records(log_dir: Path) -> list[dict]:
    files = list(log_dir.glob("predictions_*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]


def test_predict_ecrit_un_log_json_complet(client, log_dir):
    assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200
    (record,) = _read_records(log_dir)
    assert record["status_code"] == 200
    assert record["latency_ms"] >= 0
    assert record["inputs"]["AMT_CREDIT"] == 500_000.0
    assert 0.0 <= record["output"]["probability_default"] <= 1.0
    assert record["error"] is None


def test_requete_invalide_ne_cree_pas_de_log(client, log_dir):
    client.post("/predict", json=dict(VALID_PAYLOAD, AMT_INCOME_TOTAL=0))
    assert not list(log_dir.glob("*.jsonl"))


def test_log_prediction_erreur_et_append(tmp_path):
    log_prediction(tmp_path, inputs={"a": 1}, output=None, latency_ms=1.0, status_code=500, error="boom")
    log_prediction(tmp_path, inputs={"a": 2}, output={"x": 1}, latency_ms=2.0, status_code=200)
    records = _read_records(tmp_path)
    assert [r["status_code"] for r in records] == [500, 200]
    assert records[0]["error"] == "boom"


def test_load_logs_dossier_vide(tmp_path):
    assert load_logs(tmp_path).empty


def test_alertes_operationnelles_sans_anomalie():
    logs = pd.DataFrame({"status_code": [200] * 20, "latency_ms": [10.0] * 20})
    assert operational_alerts(logs) == []


def test_alertes_taux_erreur_et_latence():
    logs = pd.DataFrame({"status_code": [200] * 8 + [500] * 2, "latency_ms": [900.0] * 10})
    alerts = operational_alerts(logs)
    assert len(alerts) == 2
