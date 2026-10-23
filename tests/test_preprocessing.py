"""Tests unitaires de la préparation des données."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.preprocessing import prepare_input, select_features


def test_select_features_exclut_target_et_id():
    df = pd.DataFrame({
        "TARGET": [0, 1],
        "SK_ID_CURR": [1, 2],
        "AMT_CREDIT": [1000.0, 2000.0],
        "CATEG": ["a", "b"],
    })
    assert select_features(df) == ["AMT_CREDIT"]


def test_prepare_input_impute_les_colonnes_manquantes_avec_la_mediane():
    features = ["AMT_CREDIT", "AMT_INCOME_TOTAL"]
    medians = {"AMT_CREDIT": 1000.0, "AMT_INCOME_TOTAL": 5000.0}
    result = prepare_input({"AMT_CREDIT": 2000.0}, features, medians)

    assert list(result.columns) == features
    assert result.iloc[0]["AMT_CREDIT"] == 2000.0
    assert result.iloc[0]["AMT_INCOME_TOTAL"] == 5000.0


def test_prepare_input_respecte_lordre_des_features():
    features = ["B", "A", "C"]
    medians = {"A": 1.0, "B": 2.0, "C": 3.0}
    result = prepare_input({"A": 10.0, "C": 30.0, "B": 20.0}, features, medians)
    assert list(result.columns) == features
    assert result.iloc[0].tolist() == [20.0, 10.0, 30.0]


def test_prepare_input_tolere_une_valeur_non_numerique_en_la_remplacant():
    # Les champs textuels sont coercés en NaN puis imputés par la médiane.
    features = ["AMT_CREDIT"]
    medians = {"AMT_CREDIT": 1500.0}
    result = prepare_input({"AMT_CREDIT": "pas_un_nombre"}, features, medians)
    assert result.iloc[0]["AMT_CREDIT"] == 1500.0


def test_prepare_input_rempli_a_zero_si_aucune_mediane():
    features = ["UNKNOWN"]
    medians: dict[str, float] = {}
    result = prepare_input({}, features, medians)
    assert result.iloc[0]["UNKNOWN"] == 0.0


def test_prepare_input_accepte_un_dataframe():
    features = ["A"]
    medians = {"A": 0.0}
    frame = pd.DataFrame({"A": [1.0, np.nan]})
    result = prepare_input(frame, features, medians)
    assert result["A"].tolist() == [1.0, 0.0]
