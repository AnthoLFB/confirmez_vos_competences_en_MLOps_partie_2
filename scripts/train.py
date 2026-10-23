"""Script d'entraînement du modèle de scoring crédit.

Ce script charge les CSV du dossier `data/`, applique la préparation définie
dans `app/preprocessing.py`, entraîne un LightGBM avec les hyperparamètres
issus de l'optimisation du notebook 04, puis recherche le seuil minimisant le
coût métier `10 * FN + 1 * FP`. Le modèle et ses métadonnées sont sauvegardés
dans un fichier joblib que l'API chargera au démarrage.

Usage :
    uv run python scripts/train.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from app.core.config import settings
from app.model_service import ModelBundle, save_model
from app.preprocessing import build_feature_frame, select_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")

# Hyperparamètres validés par Optuna dans le notebook 04 ; on les fige pour la reproductibilité.
# On utilise l'API native LightGBM (lgb.train) pour éviter la dépendance à scikit-learn.
LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "is_unbalance": True,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": 8,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "verbosity": -1,
}

# Nombre d'itérations de boosting (équivalent à n_estimators de l'API sklearn).
NUM_BOOST_ROUND = 300


def _roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calcule l'AUC-ROC sans dépendre de scikit-learn (méthode des trapèzes)."""
    desc = np.argsort(y_scores)[::-1]
    y_sorted = y_true[desc]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0], tps / n_pos])
    fpr = np.concatenate([[0], fps / n_neg])
    return float(np.trapezoid(tpr, fpr))


def _train_test_split(
    X: np.ndarray | "pd.DataFrame",
    y: np.ndarray | "pd.Series",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: np.ndarray | "pd.Series" | None = None,
) -> tuple:
    """Split stratifié simple reproduisant le comportement de sklearn."""
    rng = np.random.RandomState(random_state)
    n = len(y)
    y_arr = np.asarray(y)

    if stratify is not None:
        # On sépare par classe puis on échantillonne dans chaque classe.
        train_idx, test_idx = [], []
        for cls in np.unique(y_arr):
            cls_idx = np.where(y_arr == cls)[0]
            rng.shuffle(cls_idx)
            n_test = max(1, int(round(len(cls_idx) * test_size)))
            test_idx.extend(cls_idx[:n_test])
            train_idx.extend(cls_idx[n_test:])
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
    else:
        indices = np.arange(n)
        rng.shuffle(indices)
        n_test = max(1, int(round(n * test_size)))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    import pandas as pd
    if isinstance(X, pd.DataFrame):
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coût métier défini dans les notebooks : un faux négatif coûte 10 fois plus qu'un faux positif."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return float(10 * fn + 1 * fp)


def _find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, float]:
    """Recherche exhaustive du seuil qui minimise le coût métier sur un jeu de validation."""
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = [_business_cost(y_true, (y_probs >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])


def train() -> ModelBundle:
    """Entraîne le modèle et renvoie le bundle prêt à être sérialisé."""
    data_dir = Path(settings.data_dir)
    logger.info("Chargement et préparation des données depuis %s ...", data_dir)
    t0 = time.perf_counter()
    df = build_feature_frame(data_dir, source="train")
    logger.info("Jeu préparé en %.1fs : %s", time.perf_counter() - t0, df.shape)

    if "TARGET" not in df.columns:
        raise RuntimeError("Colonne TARGET absente du dataset d'entraînement.")

    feature_names = select_features(df)
    logger.info("Nombre de features retenues : %d", len(feature_names))

    X = df[feature_names]
    y = df["TARGET"].astype(int)

    # Les médianes servent à imputer les valeurs manquantes à l'entraînement comme à l'inférence.
    medians = X.median(numeric_only=True).to_dict()
    X_imputed = X.fillna(value=medians).fillna(0.0)

    X_train, X_valid, y_train, y_valid = _train_test_split(
        X_imputed, y, test_size=0.2, stratify=y, random_state=42,
    )

    logger.info("Entraînement LightGBM (API native) ...")
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    booster = lgb.train(
        LGBM_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[valid_data],
    )

    y_probs = booster.predict(X_valid)
    auc = _roc_auc_score(np.asarray(y_valid), y_probs)
    threshold, cost = _find_best_threshold(np.asarray(y_valid), y_probs)
    logger.info("AUC validation : %.4f", auc)
    logger.info("Seuil optimal : %.3f (coût métier associé : %.0f)", threshold, cost)

    return ModelBundle(
        model=booster,
        feature_names=feature_names,
        medians={k: float(v) for k, v in medians.items()},
        threshold=threshold,
    )


def main() -> None:
    bundle = train()
    save_model(bundle, settings.model_path)
    logger.info("Modèle sauvegardé dans %s", settings.model_path)


if __name__ == "__main__":
    main()
