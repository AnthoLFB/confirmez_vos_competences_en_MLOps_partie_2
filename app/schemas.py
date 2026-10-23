"""Schémas Pydantic pour la validation des entrées et sorties de l'API.

On expose un petit nombre de champs explicitement validés (les plus critiques
d'un point de vue métier) et on autorise des features supplémentaires via le
champ `extra_features`. Cela évite d'écrire manuellement un schéma pour les
~120 colonnes du dataset tout en garantissant les contrôles essentiels.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClientFeatures(BaseModel):
    """Entrée d'une requête de prédiction.

    Les champs listés ici sont considérés comme obligatoires et bornés. Les
    autres features optionnelles du modèle peuvent être passées via
    `extra_features` ; les champs absents seront imputés par la médiane
    apprise à l'entraînement.
    """

    # Les types sont stricts : FastAPI/Pydantic rejettera un texte envoyé à la place d'un nombre.
    model_config = ConfigDict(extra="forbid")

    # Montant du crédit demandé, strictement positif.
    AMT_CREDIT: float = Field(..., gt=0, description="Montant total du crédit demandé.")

    # Revenu annuel du demandeur, strictement positif.
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Revenu annuel du demandeur.")

    # DAYS_BIRTH est négatif dans Home Credit (nombre de jours avant la demande), borné à [-25550, -6570] (18 à 70 ans).
    DAYS_BIRTH: int = Field(..., ge=-25550, le=-6570, description="Âge en jours (valeur négative).")

    # Nombre d'enfants, entier positif.
    CNT_CHILDREN: int = Field(..., ge=0, le=20)

    # Dictionnaire ouvert pour les features additionnelles facultatives.
    extra_features: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra_features")
    @classmethod
    def _check_extra_features(cls, value: dict[str, Any]) -> dict[str, Any]:
        """On refuse les valeurs manifestement incorrectes dans extra_features."""
        for key, val in value.items():
            if val is None:
                continue
            if not isinstance(val, (int, float, bool)):
                raise ValueError(
                    f"La feature additionnelle '{key}' doit être numérique, reçu {type(val).__name__}."
                )
        return value

    def to_feature_dict(self) -> dict[str, Any]:
        """Fusionne les champs validés avec les features additionnelles pour le modèle."""
        base = self.model_dump(exclude={"extra_features"})
        base.update(self.extra_features)
        return base


class PredictionResponse(BaseModel):
    """Réponse renvoyée par l'endpoint /predict."""

    probability_default: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    decision: int = Field(..., ge=0, le=1, description="1 = refus, 0 = acceptation.")
    label: str


class HealthResponse(BaseModel):
    """Réponse de l'endpoint /health."""

    status: str
    model_loaded: bool
    n_features: int
