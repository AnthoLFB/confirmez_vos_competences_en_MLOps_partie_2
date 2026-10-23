# API de scoring crédit — Projet 8 MLOps

API FastAPI exposant un modèle **LightGBM** entraîné sur le dataset
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).
Elle reçoit les informations d'un client et renvoie la probabilité de défaut
ainsi qu'une décision (accepter / refuser) basée sur un seuil optimisé pour le
coût métier `10 * FN + 1 * FP`.

## Choix du modèle

Les quatre familles testées dans le notebook `03_training_cross_validation.ipynb`
(régression logistique, random forest, XGBoost, LightGBM, MLP) ont été comparées
sur l'AUC et le coût métier. **LightGBM** offre le meilleur compromis :
performance comparable à XGBoost, entraînement plus rapide, excellente gestion
des valeurs manquantes et faible empreinte mémoire pour le service. Les
hyperparamètres utilisés par le script d'entraînement sont ceux issus de
l'optimisation Optuna du notebook `04_optimization_threshold.ipynb`.

## Structure du projet

```
app/
    core/config.py        Configuration via variables d'environnement (.env)
    preprocessing.py      Chargement CSV, agrégats, imputation médiane
    model_service.py      Chargement/sauvegarde du bundle, prédiction
    schemas.py            Schémas Pydantic (validation entrée/sortie)
    main.py               Application FastAPI (endpoints + sécurité)
scripts/
    train.py              Entraînement LightGBM + recherche du seuil optimal
tests/                    Tests unitaires et d'intégration pytest
models/                   Modèle sérialisé (généré par scripts/train.py)
data/                     CSV du dataset Home Credit
```

## Installation

Le projet utilise **UV** pour la gestion des dépendances.

```powershell
uv sync
Copy-Item .env.example .env
# puis éditer .env pour définir une vraie API_KEY
```

Variables d'environnement disponibles (voir `.env.example`) :

| Variable     | Rôle                                                      | Défaut                 |
|--------------|-----------------------------------------------------------|------------------------|
| `API_KEY`    | (Inutilisée pour l'instant) Ancienne clé d'API             | `change-me`            |
| `MODEL_PATH` | Chemin du fichier joblib contenant le modèle              | `models/model.joblib`  |
| `DATA_DIR`   | Dossier contenant les CSV du dataset                      | `data`                 |

## Entraînement du modèle

Place les CSV du dataset Home Credit dans `data/` puis lance :

```powershell
uv run python scripts/train.py
```

Le script produit `models/model.joblib` contenant le modèle, la liste des
features, les médianes d'imputation et le seuil de décision optimal.

## Lancement de l'API

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

ou plus simplement :

```powershell
uv run python main.py
```

Documentation interactive disponible sur `http://localhost:8000/docs`.

## Correspondance noms métier → noms techniques (option 1)

Dans l'API, on conserve les noms techniques des features attendus par le modèle. Pour faciliter l'intégration, voici un rappel des champs principaux et de leur signification métier.

| Nom métier (FR)  | Clé JSON attendue | Description courte |
|------------------|-------------------|--------------------|
| montant_crédit   | `AMT_CREDIT`      | Montant total du crédit demandé (strictement positif). |
| revenu_annuel    | `AMT_INCOME_TOTAL`| Revenu annuel du demandeur (strictement positif). |
| jours_naissance  | `DAYS_BIRTH`      | Âge en jours, valeur négative (par ex. −14000 ≈ 38 ans). Borné à [−25550, −6570]. |
| nb_enfants       | `CNT_CHILDREN`    | Nombre d'enfants (entier ≥ 0). |

Les autres variables du modèle peuvent être passées via le dictionnaire `extra_features` du corps JSON. Les features absentes sont automatiquement imputées par la **médiane** apprise à l'entraînement.

Exemple minimal de requête complète :

```json
{
  "AMT_CREDIT": 500000,
  "AMT_INCOME_TOTAL": 200000,
  "DAYS_BIRTH": -15000,
  "CNT_CHILDREN": 1,
  "extra_features": {
    "EXT_SOURCE_1": 0.42,
    "EXT_SOURCE_2": 0.55
  }
}
```

## Endpoints

### `GET /health` (public)

Vérifie l'état du service et confirme que le modèle est bien chargé.

```json
{ "status": "ok", "model_loaded": true, "n_features": 142 }
```

### `POST /predict`

Corps de requête (exemple minimal) :

```json
{
  "AMT_CREDIT": 500000,
  "AMT_INCOME_TOTAL": 200000,
  "DAYS_BIRTH": -15000,
  "CNT_CHILDREN": 1,
  "extra_features": {
    "EXT_SOURCE_1": 0.42,
    "EXT_SOURCE_2": 0.55
  }
}
```

Les quatre premiers champs sont obligatoires et validés (bornes métier).
`extra_features` est un dictionnaire ouvert pour transmettre les autres
features du modèle ; les colonnes absentes sont imputées par la médiane
apprise à l'entraînement.

Exemple d'appel :

```powershell
curl -X POST http://localhost:8000/predict `
     -H "Content-Type: application/json" `
     -d '{"AMT_CREDIT":500000,"AMT_INCOME_TOTAL":200000,"DAYS_BIRTH":-15000,"CNT_CHILDREN":1}'
```

Réponse :

```json
{
  "probability_default": 0.187,
  "threshold": 0.42,
  "decision": 0,
  "label": "accept"
}
```

#### Exemples de requêtes

Profil « classique » (devrait être accepté) :

```json
{
  "AMT_CREDIT": 300000,
  "AMT_INCOME_TOTAL": 150000,
  "DAYS_BIRTH": -14000,
  "CNT_CHILDREN": 0,
  "extra_features": {
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.55
  }
}
```

Profil « plus risqué » (plus de chances d'être refusé) :

```json
{
  "AMT_CREDIT": 900000,
  "AMT_INCOME_TOTAL": 45000,
  "DAYS_BIRTH": -8000,
  "CNT_CHILDREN": 3,
  "extra_features": {
    "EXT_SOURCE_1": 0.1,
    "EXT_SOURCE_2": 0.15
  }
}
```

### `POST /train`

Lance un entraînement du modèle en tâche de fond. L'appel répond immédiatement
avec `{"status": "started"}`. Les logs serveur détaillent la progression.
À la fin, le modèle est sauvegardé sur disque et rechargé en mémoire.

```powershell
curl -X POST http://localhost:8000/train
```

## Sécurité et bonnes pratiques

- **Authentification** : désactivée pour l'instant. Aucun header n'est requis.
  On pourra réactiver une clé API plus tard si besoin.
- **Validation d'entrée** : Pydantic rejette les champs manquants, les types
  incorrects et les valeurs hors bornes (revenu ≤ 0, âge < 18 ans ou > 70 ans,
  nombre d'enfants négatif, etc.).
- **Chargement du modèle** : le modèle est désérialisé **une seule fois** au
  démarrage via le `lifespan` FastAPI et stocké dans `app.state.model_bundle`.
  Toutes les requêtes réutilisent la même instance, ce qui garantit un temps
  de réponse stable et un usage mémoire maîtrisé.
- **Robustesse** : si le fichier modèle est absent au démarrage, l'API répond
  `503` sur `/predict` et le signale sur `/health` au lieu de planter.

## Tests

```powershell
uv run pytest
```

Les tests couvrent :

- Préparation des données : sélection de features, imputation, ordre des
  colonnes, tolérance aux types invalides.
- Service modèle : prédiction déterministe, sérialisation/désérialisation,
  erreur explicite en cas d'artefact manquant.
- API : champ obligatoire manquant, valeurs hors plage (revenu nul, âge
  aberrant), type incorrect sur un champ principal et sur `extra_features`,
  champ non déclaré, absence de modèle (`503`), état remonté par `/health`.
