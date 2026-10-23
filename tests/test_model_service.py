"""Tests unitaires du service de prédiction et de la persistance du bundle."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.model_service import ModelBundle, load_model, predict, save_model


def test_predict_renvoie_une_probabilite_bornee(fake_bundle: ModelBundle):
    result = predict(fake_bundle, {"AMT_CREDIT": 1.5, "EXT_SOURCE_1": -0.5})
    assert 0.0 <= result["probability_default"] <= 1.0
    assert result["decision"] in (0, 1)
    assert result["label"] in ("accept", "refuse")
    assert result["threshold"] == fake_bundle.threshold


def test_predict_est_deterministe(fake_bundle: ModelBundle):
    r1 = predict(fake_bundle, {"AMT_CREDIT": 2.0})
    r2 = predict(fake_bundle, {"AMT_CREDIT": 2.0})
    assert r1 == r2


def test_save_puis_load_restitue_le_meme_bundle(tmp_path: Path, fake_bundle: ModelBundle):
    path = tmp_path / "model.joblib"
    save_model(fake_bundle, path)

    reloaded = load_model(path)
    assert reloaded.feature_names == fake_bundle.feature_names
    assert reloaded.threshold == fake_bundle.threshold
    assert reloaded.medians == fake_bundle.medians

    # Les deux modèles doivent produire la même probabilité pour la même entrée.
    payload = {"AMT_CREDIT": 0.3}
    assert predict(reloaded, payload) == predict(fake_bundle, payload)


def test_load_model_sans_fichier_leve_une_erreur_explicite(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "inexistant.joblib")
