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
uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
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

## Monitoring en production

**Stockage** : chaque appel à `/predict` est écrit en JSON Lines dans `logs/predictions_AAAA-MM-JJ.jsonl` (`app/monitoring.py`) : timestamp, inputs (features), output (probabilité, décision), latence en ms, code HTTP, erreur éventuelle. Un fichier par jour : rotation et purge simples (coût de stockage maîtrisé). Aucun identifiant client n'est stocké (RGPD). Le dossier est configurable via `LOG_DIR`. En production, ces fichiers peuvent être collectés vers PostgreSQL/Elasticsearch.

**Analyse** : `python scripts/monitor.py` compare les inputs loggés à un échantillon de `data/application_train.csv` (référence) avec Evidently (`reports/drift_report.html`) et vérifie le taux d'erreur (>5 %), la latence p95 (>500 ms) et la part de colonnes en drift (>30 %). Code retour 1 en cas d'alerte (planifiable via cron/CI).

**Points de vigilance** : minimum de 30 requêtes pour le drift ; surveiller la distribution des scores et du taux de refus ; le drift des inputs ne prouve pas une baisse de performance (les labels réels arrivent en différé) ; ré-entraîner si un drift persistant est confirmé.

## Dashboard de Monitoring

Un tableau de bord interactif **Streamlit** est disponible pour visualiser les performances en temps réel.

```powershell
uv run streamlit run scripts/dashboard.py
```

Il affiche :
- Les métriques clés : volume d'appels, taux d'erreur, latence moyenne et p95.
- La distribution des probabilités de défaut (avec rappel du seuil métier).
- La répartition des codes de réponse HTTP.
- L'intégration directe du rapport de **Data Drift** (Evidently).

## Optimisation et Profiling

L'API a été profilée pour garantir des performances optimales en production.

```powershell
# Génère un rapport top-20 des fonctions les plus lourdes
$env:PYTHONPATH="."
uv run python scripts/profile_api.py
```

Résultats de l'optimisation :
- **Chargement Singleton** : Le modèle LightGBM est chargé une seule fois au démarrage (`lifespan`), évitant des pics de latence et de mémoire.
- **LightGBM Native** : L'utilisation de l'API native (booster) plutôt que le wrapper sklearn réduit l'overhead d'inférence.
- **Validation Pydantic** : Un schéma hybride (champs fixes + dictionnaire ouvert) concilie sécurité métier et flexibilité technique.

## Stockage des données de production

Les données sont actuellement stockées en local au format **JSON Lines** (`logs/predictions_*.jsonl`). Ce format est choisi pour sa robustesse (une ligne = un record) et sa facilité d'ingestion par des outils de type ELK (Elasticsearch, Logstash, Kibana) ou des bases de données relationnelles.

**Schéma des données logguées :**
- `timestamp` : Date/Heure UTC (ISO 8601).
- `inputs` : Features envoyées au modèle (anonymisées, pas d'ID client).
- `output` : Résultat du modèle (probabilité, décision, seuil).
- `latency_ms` : Temps de traitement de la requête.
- `status_code` : Code HTTP de retour (200, 422, 500, etc.).

### Évolutions recommandées
Pour un passage à l'échelle (plusieurs instances d'API), il est recommandé de centraliser ces logs vers :
1. **PostgreSQL** : Pour des requêtes analytiques précises sur les performances.
2. **Elasticsearch** : Pour une visualisation temps réel via Kibana.
3. **S3 / Blob Storage** : Pour l'archivage long terme et le ré-entraînement.

### Les métriques expliquées

| Métrique | Calcul | Seuil d'alerte | Ce que ça signale |
|---|---|---|---|
| Taux d'erreur | part des appels avec `status_code >= 400` | > 5 % | bug, modèle absent (503), erreur interne (500) |
| Latence moyenne | moyenne de `latency_ms` | - | indicateur général, sensible aux valeurs extrêmes |
| Latence p95 | 95 % des appels sont plus rapides que cette valeur | > 500 ms | lenteurs qui touchent une minorité d'utilisateurs, invisibles dans la moyenne |
| Part de colonnes en drift | colonnes dont la distribution diffère de la référence / colonnes analysées | > 30 % | les clients reçus ne ressemblent plus à ceux de l'entraînement |

**Data drift** : Evidently compare, colonne par colonne, la distribution des inputs de production (`current`) à celle de l'entraînement (`reference`, échantillon de `application_train.csv`) via un test statistique (par défaut Kolmogorov-Smirnov pour le numérique, avec une p-value). Une colonne est « en drift » si le test conclut que les deux distributions diffèrent. Le rapport `reports/drift_report.html` détaille chaque colonne. Le drift est calculé uniquement sur les appels réussis (200) et s'il y a au moins 30 requêtes.

**Pourquoi le p95 plutôt que le max ?** Le max dépend d'un seul appel isolé ; le p95 reflète l'expérience réelle de la quasi-totalité des appels.

**Limite** : un drift des inputs ne prouve pas que le modèle se dégrade (les vrais labels arrivent plus tard). C'est un signal d'alerte à investiguer, pas une preuve.

### Tests du monitoring

`tests/test_monitoring.py` vérifie : le contenu d'une ligne de log (inputs, output, latence, statut), l'absence de log pour une requête rejetée (422), l'enregistrement des erreurs, et les alertes de taux d'erreur et de latence. Les tests écrivent dans un dossier temporaire (fixture `log_dir`), jamais dans `logs/`. Lancer : `uv run pytest`.
