"""Service de prédiction : chargement unique du modèle et inférence.

Le modèle est chargé une seule fois au démarrage de l'API via la fonction
`load_model`, puis réutilisé pour toutes les requêtes. Cela évite de payer le
coût de désérialisation à chaque appel et garantit un temps de réponse stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.preprocessing import prepare_input


@dataclass
class ModelBundle:
    """Regroupe le modèle entraîné et ses métadonnées de prédiction."""

    model: Any
    feature_names: list[str]
    medians: dict[str, float]
    threshold: float


def load_model(path: Path) -> ModelBundle:
    """Charge le bundle (modèle + métadonnées) depuis un fichier joblib."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable à l'emplacement {path}. "
            "Lancez d'abord `python scripts/train.py` pour générer le fichier."
        )

    payload = joblib.load(path)
    return ModelBundle(
        model=payload["model"],
        feature_names=list(payload["feature_names"]),
        medians=dict(payload["medians"]),
        threshold=float(payload["threshold"]),
    )


def save_model(bundle: ModelBundle, path: Path) -> None:
    """Sérialise un bundle (modèle + métadonnées) dans un fichier joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle.model,
            "feature_names": bundle.feature_names,
            "medians": bundle.medians,
            "threshold": bundle.threshold,
        },
        path,
    )


def predict(bundle: ModelBundle, features: dict) -> dict:
    """Calcule la probabilité de défaut et la décision associée pour un client."""
    frame = prepare_input(features, bundle.feature_names, bundle.medians)
    # L'API native LightGBM renvoie directement les probabilités de la classe positive.
    probability = float(bundle.model.predict(frame)[0])
    decision = int(probability >= bundle.threshold)
    return {
        "probability_default": probability,
        "threshold": bundle.threshold,
        "decision": decision,
        "label": "refuse" if decision == 1 else "accept",
    }


def predict_batch(bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """Variante batch utile pour les tests et scripts d'évaluation."""
    prepared = prepare_input(frame, bundle.feature_names, bundle.medians)
    probabilities = bundle.model.predict(prepared)
    decisions = (probabilities >= bundle.threshold).astype(int)
    return pd.DataFrame(
        {
            "probability_default": probabilities,
            "decision": decisions,
        }
    )
