import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Dashboard Scoring Crédit - Monitoring", layout="wide")

st.title("Dashboard de Monitoring - Prêt à Dépenser")
st.markdown("""
Ce tableau de bord permet de suivre en temps réel les performances de l'API de scoring, 
la distribution des prédictions et de détecter d'éventuelles dérives de données (data drift).
""")

# --- Chargement des données ---
LOG_DIR = Path("logs")

def load_logs(log_dir: Path):
    rows = []
    for path in sorted(log_dir.glob("predictions_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)

if not LOG_DIR.exists() or not list(LOG_DIR.glob("predictions_*.jsonl")):
    st.warning(f"Aucun log trouvé dans {LOG_DIR}. Utilisez `scripts/simulate_traffic.py` pour générer des données.")
    st.stop()

df = load_logs(LOG_DIR)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- Sidebar Filters ---
st.sidebar.header("Filtres")
date_range = st.sidebar.date_input("Période", [df['timestamp'].min().date(), df['timestamp'].max().date()])
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    df = df.loc[mask]

# --- Métriques Clés ---
col1, col2, col3, col4 = st.columns(4)

total_calls = len(df)
error_rate = (df['status_code'] >= 400).mean()
avg_latency = df['latency_ms'].mean()
p95_latency = df['latency_ms'].quantile(0.95)

col1.metric("Total Appels", total_calls)
col2.metric("Taux d'Erreur", f"{error_rate:.2%}", delta=None if error_rate < 0.05 else "ALERTE", delta_color="inverse")
col3.metric("Latence Moyenne", f"{avg_latency:.1f} ms")
col4.metric("Latence P95", f"{p95_latency:.1f} ms", delta=None if p95_latency < 500 else "LENT", delta_color="inverse")

# --- Visualisations ---
st.divider()
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Volume d'appels et Latence")
    # Aggrégation par heure pour la série temporelle
    df_resampled = df.set_index('timestamp').resample('h').agg({
        'status_code': 'count',
        'latency_ms': 'mean'
    }).reset_index()
    
    fig_vol = px.line(df_resampled, x='timestamp', y='status_code', title="Nombre d'appels par heure")
    st.plotly_chart(fig_vol, width="stretch")

with row1_col2:
    st.subheader("Distribution des Scores")
    # Extraire la probabilité des sorties réussies
    scores = df[df['status_code'] == 200]['output'].apply(lambda x: x.get('probability_default') if x else None).dropna()
    if not scores.empty:
        fig_dist = px.histogram(scores, nbins=30, title="Distribution des probabilités de défaut",
                               labels={'value': 'Probabilité'}, color_discrete_sequence=['#636EFA'])
        # Ajouter une ligne pour le seuil (ex: 0.42)
        threshold = df[df['status_code'] == 200]['output'].apply(lambda x: x.get('threshold') if x else 0.42).iloc[-1]
        fig_dist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Seuil ({threshold})")
        st.plotly_chart(fig_dist, width="stretch")
    else:
        st.info("Aucune donnée de prédiction réussie pour afficher la distribution.")

st.divider()
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("État des Réponses (Status Codes)")
    status_counts = df['status_code'].value_counts().reset_index()
    status_counts.columns = ['Status Code', 'Count']
    fig_status = px.pie(status_counts, values='Count', names='Status Code', title="Répartition des codes HTTP")
    st.plotly_chart(fig_status, width="stretch")

with row2_col2:
    st.subheader("Latence par Endpoint")
    fig_lat = px.box(df, y='latency_ms', points="all", title="Distribution de la latence (ms)")
    st.plotly_chart(fig_lat, width="stretch")

# --- Section Data Drift ---
st.divider()
st.header("Analyse du Data Drift (Evidently)")

REPORT_PATH = Path("reports/drift_report.html")
if REPORT_PATH.exists():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    st.info("Le dernier rapport de drift généré par Evidently est disponible ci-dessous.")
    # On utilise st.components.v1.html pour afficher le HTML brut d'Evidently
    import streamlit.components.v1 as components
    components.html(html_content, height=800, scrolling=True)
else:
    st.warning("Aucun rapport de drift trouvé. Lancez `python scripts/monitor.py` pour générer une analyse.")

st.sidebar.markdown("---")
st.sidebar.info(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")
if st.sidebar.button("Rafraîchir"):
    st.rerun()
