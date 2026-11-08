"""Analyse automatique des logs de production : drift, taux d'erreur, latence.

Usage : python scripts/monitor.py [--log-dir logs] [--out reports]

Référence du drift = échantillon des données d'entraînement (application_train.csv).
Sortie : reports/drift_report.html (Evidently) + résumé console.
Code retour 1 si une alerte est levée (utilisable en cron / CI).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ERROR_RATE_MAX = 0.05      # >5 % d'erreurs => alerte
LATENCY_P95_MAX_MS = 500   # p95 > 500 ms => alerte
DRIFT_SHARE_MAX = 0.3      # >30 % de colonnes en drift => alerte
MIN_ROWS = 30              # en dessous, l'analyse de drift n'est pas fiable


def load_logs(log_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(log_dir).glob("predictions_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return pd.DataFrame(rows)


def operational_alerts(logs: pd.DataFrame) -> list[str]:
    alerts = []
    error_rate = float((logs["status_code"] >= 400).mean())
    p95 = float(logs["latency_ms"].quantile(0.95))
    print(f"Appels: {len(logs)} | taux d'erreur: {error_rate:.1%} | "
          f"latence moy: {logs['latency_ms'].mean():.1f} ms | p95: {p95:.1f} ms")
    if error_rate > ERROR_RATE_MAX:
        alerts.append(f"Taux d'erreur élevé : {error_rate:.1%}")
    if p95 > LATENCY_P95_MAX_MS:
        alerts.append(f"Latence p95 anormale : {p95:.0f} ms")
    return alerts


def drift_alerts(logs: pd.DataFrame, train_csv: Path, out_dir: Path) -> list[str]:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    ok = logs[(logs["status_code"] == 200) & logs["inputs"].notna()]
    current = pd.DataFrame(list(ok["inputs"]))
    if len(current) < MIN_ROWS:
        print(f"Drift ignoré : {len(current)} requêtes valides (< {MIN_ROWS}).")
        return []

    cols = [c for c in current.columns if current[c].notna().any()]
    reference = pd.read_csv(train_csv, usecols=lambda c: c in cols).sample(
        n=min(10_000, 100_000), random_state=0
    ) if train_csv.exists() else None
    if reference is None:
        return [f"Référence introuvable : {train_csv}"]
    cols = [c for c in cols if c in reference.columns]
    current = current[cols].apply(pd.to_numeric, errors="coerce")
    reference = reference[cols].apply(pd.to_numeric, errors="coerce")

    snapshot = Report([DataDriftPreset()]).run(current, reference)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_dir / "drift_report.html"))

    result = snapshot.json()
    if isinstance(result, str):
        result = json.loads(result)

    # Recherche du share dans les résultats (Evidently 0.4+)
    share = None
    # On cherche dans le dictionnaire de résultats
    for metric in result.get("metrics", []):
        res = metric.get("result", {})
        
        # evidently 0.4+ DatasetDriftMetric
        if metric.get("metric") == "DatasetDriftMetric":
            share = res.get("drift_share")
        
        # evidently DataDriftPreset / DataDriftTable
        elif "Drift" in str(metric.get("metric", "")):
            # On cherche récursivement ou dans les clés connues
            share = (res.get("dataset_drift", {}).get("share_of_drifted_columns") 
                     or res.get("share_of_drifted_columns")
                     or res.get("share"))
        
        if share is not None:
            break
    
    # DEBUG : Si share est toujours None, on affiche les métriques disponibles
    if share is None and result.get("metrics"):
        available = [m.get("metric") for m in result["metrics"]]
        print(f"DEBUG: Metriques trouvées : {available}")
        # Tentative désespérée : prendre le premier drift_share trouvé n'importe où
        import re
        dump = json.dumps(result)
        match = re.search(r'"(drift_share|share_of_drifted_columns|share)":\s*([0-9.]+)', dump)
        if match:
            share = float(match.group(2))
    print(f"Part de colonnes en drift : {share if share is None else f'{share:.0%}'} "
          f"(rapport : {out_dir / 'drift_report.html'})")
    if share is not None and share > DRIFT_SHARE_MAX:
        return [f"Data drift : {share:.0%} des colonnes dérivent"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/application_train.csv"))
    args = parser.parse_args()

    logs = load_logs(args.log_dir)
    if logs.empty:
        print("Aucun log trouvé.")
        return 0

    alerts = operational_alerts(logs) + drift_alerts(logs, args.train_csv, args.out)
    for alert in alerts:
        print(f"[ALERTE] {alert}")
    if not alerts:
        print("Aucune anomalie détectée.")
    return 1 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
