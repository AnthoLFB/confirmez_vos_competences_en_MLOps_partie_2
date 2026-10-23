"""Point d'entrée pratique pour lancer l'API en local sans retenir la commande uvicorn."""

import uvicorn


def main() -> None:
    # Rechargement automatique utile en développement ; à désactiver en production.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
