import cProfile
import pstats
import io
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd

# Ajout du parent au path pour importer app
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app

def profile_predict():
    client = TestClient(app)
    
    # Payload d'exemple
    payload = {
        "AMT_CREDIT": 500000,
        "AMT_INCOME_TOTAL": 200000,
        "DAYS_BIRTH": -15000,
        "CNT_CHILDREN": 1,
        "extra_features": {
            "EXT_SOURCE_1": 0.42,
            "EXT_SOURCE_2": 0.55,
            "EXT_SOURCE_3": 0.48,
            "AMT_ANNUITY": 25000
        }
    }

    print("--- Démarrage du Profiling ---")
    pr = cProfile.Profile()
    pr.enable()

    # Simuler 100 requêtes
    for _ in range(100):
        client.post("/predict", json=payload)

    pr.disable()
    print("--- Profiling Terminé ---")

    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20)  # Top 20 des fonctions les plus lourdes
    
    report_path = Path("reports/profiling_report.txt")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(s.getvalue())
    
    print(f"Rapport de profiling sauvegardé dans {report_path}")

if __name__ == "__main__":
    profile_predict()
