"""Fixtures partagées entre les tests.

On construit un `ModelBundle` factice basé sur un petit LightGBM entraîné sur
quelques lignes synthétiques. Cela permet de tester toute la chaîne API sans
dépendre du dataset réel ni du script d'entraînement complet.
"""

from __future__ import annotations

import os

# On force une clé API connue avant tout import de l'application.
os.environ.setdefault("API_KEY", "test-key")

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_service import ModelBundle


FEATURE_NAMES = [
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "DAYS_BIRTH",
    "CNT_CHILDREN",
    "EXT_SOURCE_1",
]


@pytest.fixture(scope="session")
def fake_bundle() -> ModelBundle:
    """Entraîne un petit LightGBM (API native) sur un jeu synthétique pour les tests."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    # On génère une cible corrélée à la première feature pour que le modèle apprenne quelque chose.
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)

    train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES, free_raw_data=False)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "seed": 42},
        train_data,
        num_boost_round=20,
    )

    medians = {name: 0.0 for name in FEATURE_NAMES}
    return ModelBundle(
        model=booster,
        feature_names=FEATURE_NAMES,
        medians=medians,
        threshold=0.5,
    )


@pytest.fixture
def client(fake_bundle: ModelBundle) -> TestClient:
    """Client HTTP de test avec le bundle factice injecté dans app.state."""
    # On injecte le modèle directement, sans passer par le lifespan ni le disque.
    with TestClient(app) as test_client:
        test_client.app.state.model_bundle = fake_bundle
        yield test_client


@pytest.fixture
def client_without_model() -> TestClient:
    """Client dont l'état ne contient aucun modèle (simule un démarrage sans artefact)."""
    with TestClient(app) as test_client:
        test_client.app.state.model_bundle = None
        yield test_client
