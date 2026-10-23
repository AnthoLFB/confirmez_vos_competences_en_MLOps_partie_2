"""Configuration applicative chargée depuis les variables d'environnement."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # On lit automatiquement les variables définies dans le fichier .env à la racine du projet.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Clé requise dans le header X-API-Key pour accéder aux endpoints protégés.
    api_key: str = "change-me"

    # Chemin vers le fichier joblib contenant le modèle entraîné et ses métadonnées.
    model_path: Path = Path("models/model.joblib")

    # Dossier contenant les CSV du dataset Home Credit utilisés pour l'entraînement.
    data_dir: Path = Path("data")


# Instance unique partagée par toute l'application pour éviter de recharger le .env à chaque appel.
settings = Settings()
