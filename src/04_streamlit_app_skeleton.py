"""
04_streamlit_app_skeleton.py

Dashboard base para Herdez Smart-Supply.

Este archivo es un esqueleto ejecutable para mostrar:
- KPIs ejecutivos.
- Alertas por SKU/CEDI.
- Recomendaciones costo-beneficio.
- Espacio futuro para chat con agente CrewAI/LangChain/Gemini.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Herdez Smart-Supply", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


@st.cache_data
def load_outputs():
    features_path = OUTPUTS / "herdez_features_dataset.csv"
    recommendations_path = OUTPUTS / "decision_recommendations.csv"
    scored_path = OUTPUTS / "scored_stockout_alerts.csv"

    features = pd.read_csv(features_path, parse_dates=["Fecha"]) if features_path.exists() else pd.DataFrame()
    recs = pd.read_csv(recommendations_path, parse_dates=["Fecha"]) if recommendations_path.exists() else pd.DataFrame()
    scored = pd.read_csv(scored_path, parse_dates=["Fecha"]) if scored_path.exists() else pd.DataFrame()
    return features, scored, recs


features_df, scored_df, recs_df = load_outputs()

st.title("Herdez Smart-Supply")
st.caption("Prototipo local-first para predicción de quiebres de stock y recomendación costo-beneficio.")

if features_df.empty:
    st.warning(
        "Aún no hay outputs generados. Ejecuta primero:\n"
        "1) python src/01_eda_target_features.py\n"
        "2) python src/02_train_models.py\n"
        "3) python src/03_decision_engine.py"
    )
    st.stop()

# Sidebar filtros
st.sidebar.header("Filtros")
skus = sorted(features_df["SKU_ID"].unique())
cedis = sorted(features_df["CEDI"].unique())
selected_skus = st.sidebar.multiselect("SKU", skus, default=skus)
selected_cedis = st.sidebar.multiselect("CEDI", cedis, default=cedis)

filtered = features_df[
    features_df["SKU_ID"].isin(selected_skus) & features_df["CEDI"].isin(selected_cedis)
].copy()

# KPIs
risk_rate = filtered["Target_Riesgo_Quiebre"].mean() if not filtered.empty else 0
risk_count = int(filtered["Target_Riesgo_Quiebre"].sum()) if not filtered.empty else 0
avg_stock = filtered["Stock_Actual"].mean() if not filtered.empty else 0
avg_coverage = filtered["Dias_Cobertura"].mean() if not filtered.empty else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros en riesgo", f"{risk_count:,}")
c2.metric("Tasa de riesgo", f"{risk_rate:.1%}")
c3.metric("Stock promedio", f"{avg_stock:,.0f}")
c4.metric("Días cobertura prom.", f"{avg_coverage:,.1f}")

st.divider()

# Visualizaciones ejecutivas
left, right = st.columns(2)

with left:
    st.subheader("Riesgo por SKU")
    risk_by_sku = (
        filtered.groupby("SKU_ID")["Target_Riesgo_Quiebre"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(risk_by_sku)

with right:
    st.subheader("Riesgo por CEDI")
    risk_by_cedi = (
        filtered.groupby("CEDI")["Target_Riesgo_Quiebre"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(risk_by_cedi)

st.subheader("Serie temporal de riesgo")
daily_risk = filtered.groupby("Fecha")["Target_Riesgo_Quiebre"].mean()
st.line_chart(daily_risk)

st.divider()

st.subheader("Top combinaciones SKU/CEDI críticas")
pair_risk = (
    filtered.groupby(["SKU_ID", "CEDI"])
    .agg(
        riesgos=("Target_Riesgo_Quiebre", "sum"),
        registros=("Target_Riesgo_Quiebre", "count"),
        tasa_riesgo=("Target_Riesgo_Quiebre", "mean"),
        stock_promedio=("Stock_Actual", "mean"),
        cobertura_promedio=("Dias_Cobertura", "mean"),
    )
    .reset_index()
    .sort_values("tasa_riesgo", ascending=False)
)
st.dataframe(pair_risk, use_container_width=True)

st.divider()

st.subheader("Recomendaciones costo-beneficio")
if recs_df.empty:
    st.info("Aún no hay recomendaciones. Ejecuta `03_decision_engine.py`.")
else:
    st.dataframe(recs_df, use_container_width=True)

    total_loss = recs_df["Perdida_Esperada"].sum()
    total_transfer = recs_df["Costo_Transferencia"].sum()
    total_net = recs_df["Beneficio_Neto"].sum()
    a, b, c = st.columns(3)
    a.metric("Pérdida esperada en alertas", f"${total_loss:,.0f}")
    b.metric("Costo transferencia estimado", f"${total_transfer:,.0f}")
    c.metric("Beneficio neto estimado", f"${total_net:,.0f}")

st.divider()

st.subheader("Chat con agente — siguiente fase")
st.info(
    "Aquí conectaremos CrewAI + LangChain + Gemini. El agente no inventará números: "
    "consultará el modelo, el motor de costos y las reglas de negocio."
)
question = st.text_input("Pregunta ejecutiva de ejemplo", "¿Por qué recomiendas mover inventario?")
if st.button("Simular respuesta"):
    st.write(
        "El sistema recomienda evaluar transferencias cuando el costo de mover inventario "
        "es menor que la pérdida esperada por quiebre, siempre validando que el CEDI origen "
        "no quede por debajo de su cobertura mínima."
    )
