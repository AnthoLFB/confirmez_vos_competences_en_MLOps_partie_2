"""Préparation des données partagée entre l'entraînement et l'inférence.

Ce module contient une version volontairement allégée du feature engineering du
notebook d'exploration. On garde l'application principale enrichie de quelques
agrégats issus de `bureau.csv` et `previous_application.csv`, ce qui suffit
largement pour obtenir un modèle utile sans alourdir l'API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Construit quelques agrégats simples par client à partir de bureau.csv."""
    numeric_cols = ["DAYS_CREDIT", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "CREDIT_DAY_OVERDUE"]
    numeric_cols = [c for c in numeric_cols if c in bureau.columns]

    agg = bureau.groupby("SK_ID_CURR")[numeric_cols].agg(["mean", "max", "sum"])
    agg.columns = [f"BURO_{col}_{stat}".upper() for col, stat in agg.columns]
    agg["BURO_COUNT"] = bureau.groupby("SK_ID_CURR").size()
    return agg.reset_index()


def _aggregate_previous(previous: pd.DataFrame) -> pd.DataFrame:
    """Construit quelques agrégats simples par client à partir de previous_application.csv."""
    numeric_cols = ["AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY", "CNT_PAYMENT"]
    numeric_cols = [c for c in numeric_cols if c in previous.columns]

    agg = previous.groupby("SK_ID_CURR")[numeric_cols].agg(["mean", "max", "sum"])
    agg.columns = [f"PREV_{col}_{stat}".upper() for col, stat in agg.columns]
    agg["PREV_COUNT"] = previous.groupby("SK_ID_CURR").size()
    return agg.reset_index()


def build_feature_frame(data_dir: Path, source: str = "train") -> pd.DataFrame:
    """Charge les CSV du dataset Home Credit et renvoie un DataFrame prêt pour le modèle.

    Le paramètre `source` vaut `"train"` ou `"test"` pour choisir le fichier
    `application_*`. La colonne `TARGET` n'est présente que pour `train`.
    """
    data_dir = Path(data_dir)
    if source not in {"train", "test"}:
        raise ValueError("source doit valoir 'train' ou 'test'.")

    app = pd.read_csv(data_dir / f"application_{source}.csv")

    # Agrégats optionnels : on ne bloque pas si les fichiers sont absents (utile pour les tests).
    bureau_path = data_dir / "bureau.csv"
    if bureau_path.exists():
        bureau = pd.read_csv(bureau_path, usecols=lambda c: c in {
            "SK_ID_CURR", "DAYS_CREDIT", "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT", "CREDIT_DAY_OVERDUE",
        })
        app = app.merge(_aggregate_bureau(bureau), on="SK_ID_CURR", how="left")

    previous_path = data_dir / "previous_application.csv"
    if previous_path.exists():
        previous = pd.read_csv(previous_path, usecols=lambda c: c in {
            "SK_ID_CURR", "AMT_APPLICATION", "AMT_CREDIT",
            "AMT_ANNUITY", "CNT_PAYMENT",
        })
        app = app.merge(_aggregate_previous(previous), on="SK_ID_CURR", how="left")

    return app


def select_features(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des colonnes numériques utilisables comme features."""
    excluded = {"TARGET", "SK_ID_CURR"}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in excluded]


def prepare_input(
    raw: dict | pd.DataFrame,
    feature_names: list[str],
    medians: dict[str, float],
) -> pd.DataFrame:
    """Transforme un dict ou DataFrame d'entrée en matrice de features alignée.

    Les colonnes manquantes sont remplies avec la médiane calculée lors de
    l'entraînement, ce qui garantit que l'ordre et le nombre de features
    correspondent toujours à ceux attendus par le modèle.
    """
    if isinstance(raw, dict):
        frame = pd.DataFrame([raw])
    else:
        frame = raw.copy()

    # On ajoute les colonnes manquantes afin de disposer de toutes les features du modèle.
    for col in feature_names:
        if col not in frame.columns:
            frame[col] = np.nan

    # On ne conserve que les colonnes attendues et dans le bon ordre.
    frame = frame[feature_names]

    # Les valeurs non convertibles en numérique deviennent NaN avant imputation.
    frame = frame.apply(pd.to_numeric, errors="coerce")

    # Imputation par la médiane apprise à l'entraînement.
    frame = frame.fillna(value=medians)

    # Filet de sécurité pour les colonnes dont la médiane serait absente.
    return frame.fillna(0.0)
