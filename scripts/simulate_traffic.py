"""Génère du trafic de test pour le monitoring (logs réels via l'API, sans serveur).

Usage : python scripts/simulate_traffic.py [--n 300]

Crée deux dossiers de logs :
  - logs_sim/normal : clients tirés de application_train.csv (aucun drift attendu)
  - logs_sim/drift  : mêmes clients mais décalés (revenus/crédits x2-x3, plus jeunes,
                      scores externes dégradés) => drift attendu

Analyse : python scripts/monitor.py --log-dir logs_sim/normal --out reports/normal
          python scripts/monitor.py --log-dir logs_sim/drift  --out reports/drift
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402

REQUIRED = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "CNT_CHILDREN"]
EXTRA = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_ANNUITY"]


def make_drift(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AMT_INCOME_TOTAL"] *= 3
    df["AMT_CREDIT"] *= 2
    df["AMT_ANNUITY"] *= 2
    df["DAYS_BIRTH"] = (df["DAYS_BIRTH"] * 0.6).clip(-25550, -6570).round()
    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        df[col] = (df[col] * 0.5).clip(0, 1)
    return df


def send(df: pd.DataFrame, log_dir: Path) -> None:
    if log_dir.exists():
        shutil.rmtree(log_dir)
    settings.log_dir = log_dir
    with TestClient(app) as client:
        if client.app.state.model_bundle is None:
            sys.exit("Modèle absent : lancez d'abord scripts/train.py")
        codes = []
        for row in df.to_dict("records"):
            payload = {k: row[k] for k in REQUIRED}
            payload["DAYS_BIRTH"] = int(payload["DAYS_BIRTH"])
            payload["CNT_CHILDREN"] = int(payload["CNT_CHILDREN"])
            payload["extra_features"] = {k: row[k] for k in EXTRA}
            codes.append(client.post("/predict", json=payload).status_code)
    print(f"{log_dir} : {len(codes)} appels, {codes.count(200)} réussis")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--train-csv", type=Path, default=Path("data/application_train.csv"))
    parser.add_argument("--out", type=Path, default=Path("logs_sim"))
    args = parser.parse_args()

    cols = REQUIRED + EXTRA
    df = pd.read_csv(args.train_csv, usecols=cols).dropna()
    # Écarte les valeurs hors bornes de l'API pour obtenir des requêtes valides.
    df = df[(df["AMT_INCOME_TOTAL"] > 0) & df["DAYS_BIRTH"].between(-25550, -6570)]
    # Tirage différent de la référence de monitor.py (random_state=0).
    sample = df.sample(n=args.n, random_state=1).reset_index(drop=True)

    send(sample, args.out / "normal")
    send(make_drift(sample), args.out / "drift")


if __name__ == "__main__":
    main()
