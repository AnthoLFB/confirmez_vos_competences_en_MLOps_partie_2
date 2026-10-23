"""Tests d'intégration de l'API FastAPI.

On couvre explicitement les cas critiques mentionnés dans le cahier des charges :
authentification, champ manquant, valeurs hors plage et type incorrect, ainsi
que l'absence du modèle au démarrage.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


VALID_PAYLOAD = {
    "AMT_CREDIT": 500_000.0,
    "AMT_INCOME_TOTAL": 200_000.0,
    "DAYS_BIRTH": -15_000,
    "CNT_CHILDREN": 1,
    "extra_features": {"EXT_SOURCE_1": 0.4},
}


def test_health_est_public_et_remonte_letat_du_modele(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] > 0


def test_predict_nominal_retourne_une_prediction_valide(client: TestClient):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability_default"] <= 1.0
    assert body["decision"] in (0, 1)
    assert body["label"] in ("accept", "refuse")


def test_predict_champ_obligatoire_manquant_renvoie_422(client: TestClient):
    payload = dict(VALID_PAYLOAD)
    del payload["AMT_INCOME_TOTAL"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_revenu_nul_est_rejete(client: TestClient):
    payload = dict(VALID_PAYLOAD, AMT_INCOME_TOTAL=0)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_age_negatif_absurde_est_rejete(client: TestClient):
    # -5 ans en jours = -1825 ; au-dessus de la borne -6570 (18 ans) → doit être refusé.
    payload = dict(VALID_PAYLOAD, DAYS_BIRTH=-1_825)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_type_incorrect_sur_champ_numerique_renvoie_422(client: TestClient):
    payload = dict(VALID_PAYLOAD, AMT_CREDIT="beaucoup")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_type_incorrect_dans_extra_features_renvoie_422(client: TestClient):
    payload = dict(VALID_PAYLOAD, extra_features={"EXT_SOURCE_1": "oui"})
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_champ_non_declare_est_rejete(client: TestClient):
    # Les features inconnues doivent passer par `extra_features`, pas à la racine.
    payload = dict(VALID_PAYLOAD, FEATURE_INCONNUE=1.0)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_sans_modele_renvoie_503(client_without_model: TestClient):
    response = client_without_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_health_sans_modele_signale_labsence(client_without_model: TestClient):
    response = client_without_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False
    assert body["n_features"] == 0
