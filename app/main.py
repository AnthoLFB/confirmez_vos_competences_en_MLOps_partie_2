"""Application FastAPI exposant le modèle de scoring crédit.

Le modèle est chargé une unique fois au démarrage via le gestionnaire de cycle
de vie `lifespan` et stocké dans `app.state`. Cette approche évite tout
rechargement coûteux à chaque requête et facilite les tests qui peuvent
injecter un bundle factice avant l'appel.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status

from app.core.config import settings
from app.model_service import ModelBundle, load_model, predict, save_model
from app.schemas import ClientFeatures, HealthResponse, PredictionResponse

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage et le libère à l'arrêt."""
    try:
        bundle = load_model(settings.model_path)
        app.state.model_bundle = bundle
        logger.info(
            "Modèle chargé depuis %s (%d features, seuil=%.3f)",
            settings.model_path, len(bundle.feature_names), bundle.threshold,
        )
    except FileNotFoundError as exc:
        # L'API démarre quand même pour permettre /health de signaler l'absence du modèle.
        app.state.model_bundle = None
        logger.warning("Démarrage sans modèle : %s", exc)
    yield
    app.state.model_bundle = None


app = FastAPI(
    title="API Scoring Crédit",
    description="Expose le modèle LightGBM entraîné sur le dataset Home Credit.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_bundle(request: Request) -> ModelBundle:
    """Récupère le bundle chargé en mémoire ou renvoie une erreur 503."""
    bundle: ModelBundle | None = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé. Lancez `python scripts/train.py` puis redémarrez l'API.",
        )
    return bundle


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health(request: Request) -> HealthResponse:
    """Endpoint public de supervision pour vérifier l'état du service."""
    bundle: ModelBundle | None = getattr(request.app.state, "model_bundle", None)
    return HealthResponse(
        status="ok",
        model_loaded=bundle is not None,
        n_features=len(bundle.feature_names) if bundle else 0,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
)
def predict_endpoint(
    payload: ClientFeatures,
    bundle: ModelBundle = Depends(get_bundle),
) -> PredictionResponse:
    """Prédit la probabilité de défaut pour un client donné."""
    try:
        result = predict(bundle, payload.to_feature_dict())
    except Exception as exc:  # pragma: no cover - garde-fou défensif
        logger.exception("Erreur pendant la prédiction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne pendant la prédiction : {exc}",
        ) from exc
    return PredictionResponse(**result)


@app.post("/train", tags=["training"])
def train_endpoint(background_tasks: BackgroundTasks, request: Request) -> dict:
    """Démarre un entraînement du modèle en tâche de fond et recharge le bundle.

    L'appel répond immédiatement avec un statut de démarrage. Les logs serveur
    détaillent la progression. À la fin, le modèle est sauvegardé sur disque et
    rechargé en mémoire pour les prédictions suivantes.
    """

    app_ref = request.app

    def _run_training_task() -> None:
        try:
            # Import paresseux pour éviter d'imposer scikit-learn au démarrage de l'API.
            from scripts.train import train as train_job  # type: ignore

            logger.info("[TRAIN] Démarrage de l'entraînement via endpoint /train ...")
            bundle = train_job()
            save_model(bundle, settings.model_path)
            # Remet à jour le bundle en mémoire pour les requêtes suivantes
            app_ref.state.model_bundle = bundle
            logger.info(
                "[TRAIN] Entraînement terminé. Modèle (%d features, seuil=%.3f) sauvegardé dans %s",
                len(bundle.feature_names), bundle.threshold, settings.model_path,
            )
        except ImportError as exc:  # pragma: no cover - cas d'environnement Windows/WDAC
            logger.exception(
                "[TRAIN] Impossible d'importer le module d'entraînement (scikit-learn probablement bloqué) : %s",
                exc,
            )
        except Exception as exc:  # pragma: no cover - garde-fou défensif
            logger.exception("[TRAIN] Échec de l'entraînement: %s", exc)

    background_tasks.add_task(_run_training_task)
    return {"status": "started"}
