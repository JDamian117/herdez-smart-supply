"""
05_streamlit_dashboard.py
Herdez Smart-Supply — Dashboard ejecutivo guiado

Objetivo:
- Mostrar de forma clara el resultado del pipeline ML + Decision Engine + Agentes A2A-lite.
- Ser entendible para un Director de Supply Chain y defendible ante un Gerente de IA.
- Funcionar en local y Streamlit Cloud leyendo artefactos ya generados en outputs/ y models/.

Ejecutar:
    python -m streamlit run app.py

o directamente:
    python -m streamlit run src/05_streamlit_dashboard.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Plotly es ideal para Streamlit. Si no estuviera instalado, mostramos tablas.
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Herdez Smart-Supply",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. UTILIDADES DE RUTAS Y CARGA DE DATOS
# ============================================================

def find_project_root() -> Path:
    """Encuentra la raíz del proyecto aunque Streamlit se ejecute desde app.py o src/.

    La raíz esperada contiene outputs/, src/ o app.py.
    """
    current = Path(__file__).resolve()
    candidates = [current.parent, current.parent.parent, Path.cwd(), Path.cwd().parent]
    for c in candidates:
        if (c / "outputs").exists() or (c / "app.py").exists() or (c / "src").exists():
            return c.resolve()
    return Path.cwd().resolve()


ROOT_DIR = find_project_root()
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"


def read_csv_safe(filename: str) -> pd.DataFrame:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"No pude leer {filename}: {exc}")
        return pd.DataFrame()


def read_json_safe(filename: str) -> Any:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.warning(f"No pude leer {filename}: {exc}")
        return None


def read_text_safe(filename: str) -> str:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        st.warning(f"No pude leer {filename}: {exc}")
        return ""


def first_existing_text(*filenames: str) -> str:
    for filename in filenames:
        text = read_text_safe(filename)
        if text.strip():
            return text
    return ""


def first_existing_json(*filenames: str) -> Any:
    for filename in filenames:
        obj = read_json_safe(filename)
        if obj is not None:
            return obj
    return None


def normalize_registry(obj: Any) -> pd.DataFrame:
    """Convierte el registry A2A a DataFrame, sea lista o diccionario."""
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, dict):
        data = obj.get("agents", obj.get("agent_cards", obj))
        if isinstance(data, dict):
            data = list(data.values())
    elif isinstance(obj, list):
        data = obj
    else:
        data = []
    try:
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def normalize_artifacts(obj: Any) -> List[Dict[str, Any]]:
    """Normaliza artefactos de agentes en lista de diccionarios."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        if "artifacts" in obj and isinstance(obj["artifacts"], list):
            return [x for x in obj["artifacts"] if isinstance(x, dict)]
        # Si viene como {agent: artifact}
        result = []
        for k, v in obj.items():
            if isinstance(v, dict):
                item = {"agent": k, **v}
            else:
                item = {"agent": k, "content": v}
            result.append(item)
        return result
    return []


def money(value: Any) -> str:
    try:
        return f"${float(value):,.0f} MXN"
    except Exception:
        return "$0 MXN"


def pct(value: Any) -> str:
    try:
        v = float(value)
        if v <= 1:
            v *= 100
        return f"{v:.1f}%"
    except Exception:
        return "0.0%"


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ============================================================
# 3. CARGA DE ARTEFACTOS
# ============================================================

recommendations = read_csv_safe("decision_recommendations.csv")
alerts = read_csv_safe("scored_stockout_alerts.csv")
model_metrics = read_csv_safe("model_comparison_metrics.csv")
risk_by_sku = read_csv_safe("risk_by_sku.csv")
risk_by_cedi = read_csv_safe("risk_by_cedi.csv")
risk_by_sku_cedi = read_csv_safe("risk_by_sku_cedi.csv")
time_series = read_csv_safe("time_series_daily_summary.csv")
features = read_csv_safe("herdez_features_dataset.csv")

agent_brief = first_existing_text(
    "agent_executive_brief_a2a_simple.md",
    "agent_executive_brief_a2a.md",
    "agent_executive_brief.md",
)
agent_registry_obj = first_existing_json(
    "agent_a2a_simple_registry.json",
    "agent_a2a_registry.json",
)
agent_artifacts_obj = first_existing_json(
    "agent_a2a_simple_artifacts.json",
    "agent_a2a_artifacts.json",
)
agent_trace_obj = first_existing_json(
    "agent_a2a_simple_trace.json",
    "agent_a2a_task_trace.json",
)

agent_registry_df = normalize_registry(agent_registry_obj)
agent_artifacts = normalize_artifacts(agent_artifacts_obj)


# ============================================================
# 4. ESTILOS VISUALES
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background: #f7f8fb;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f3b2e 0%, #176b4c 48%, #f2b705 140%);
        color: white;
        box-shadow: 0 16px 40px rgba(15, 59, 46, 0.22);
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
    }
    .hero p {
        font-size: 1.05rem;
        margin: 0.15rem 0;
        opacity: 0.96;
    }
    .section-card {
        padding: 1.2rem 1.4rem;
        border-radius: 20px;
        background: white;
        border: 1px solid #e8ebf0;
        box-shadow: 0 8px 26px rgba(25, 35, 52, 0.06);
        margin-bottom: 1rem;
    }
    .soft-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: #ffffff;
        border: 1px solid #edf0f5;
        box-shadow: 0 5px 18px rgba(25, 35, 52, 0.045);
        min-height: 120px;
    }
    .kpi-label {
        font-size: 0.84rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03rem;
    }
    .kpi-value {
        font-size: 1.7rem;
        color: #0f172a;
        font-weight: 800;
        margin-top: 0.15rem;
    }
    .kpi-help {
        color: #64748b;
        font-size: 0.88rem;
        margin-top: 0.3rem;
    }
    .explain-box {
        background: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        margin: 0.7rem 0 1rem 0;
        color: #14532d;
    }
    .warning-box {
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        margin: 0.7rem 0 1rem 0;
        color: #78350f;
    }
    .info-box {
        background: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        margin: 0.7rem 0 1rem 0;
        color: #1e3a8a;
    }
    .small-muted {
        color: #64748b;
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid #edf0f5;
        box-shadow: 0 6px 20px rgba(25, 35, 52, 0.045);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. VALIDACIÓN DE ARCHIVOS PRINCIPALES
# ============================================================

missing_required = []
if recommendations.empty:
    missing_required.append("outputs/decision_recommendations.csv")
if alerts.empty:
    missing_required.append("outputs/scored_stockout_alerts.csv")
if model_metrics.empty:
    missing_required.append("outputs/model_comparison_metrics.csv")

if missing_required:
    st.markdown(
        """
        <div class="hero">
            <h1>📦 Herdez Smart-Supply</h1>
            <p>Sistema local-first para riesgo de quiebre de stock y recomendaciones costo-beneficio.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error("Faltan artefactos generados por el pipeline.")
    st.write("Archivos faltantes:")
    for item in missing_required:
        st.code(item)
    st.info(
        "Ejecuta primero 01 → 02 → 03 → 04 desde la raíz del proyecto. "
        "Después vuelve a correr el dashboard."
    )
    st.code(
        "python src/01_eda_target_features.py --input data/Data_Prueba_Tecnica_Herdez_IA.xlsx\n"
        "python src/02_train_models.py --processed outputs/herdez_features_dataset.csv\n"
        "python src/03_decision_engine.py --processed outputs/herdez_features_dataset.csv --model models/best_stockout_model.joblib\n"
        "python src/04_agent_system_a2a_simple.py --mode fallback --max-alerts 8\n"
        "python -m streamlit run app.py",
        language="bash",
    )
    st.stop()


# ============================================================
# 6. PREPARACIÓN DE COLUMNAS FLEXIBLES
# ============================================================

sku_col = find_col(recommendations, ["SKU_ID", "sku", "SKU"])
cedi_col = find_col(recommendations, ["CEDI", "cedi_destino", "destination_cedi", "cedi"])
action_col = find_col(recommendations, ["recommended_action", "accion_recomendada", "action"])
risk_col = find_col(recommendations, ["risk_probability", "probabilidad_riesgo", "risk", "stockout_risk_probability"])
benefit_col = find_col(recommendations, ["net_benefit", "beneficio_neto", "beneficio_neto_mxn"])
expected_loss_col = find_col(recommendations, ["expected_loss", "perdida_esperada", "expected_loss_mxn"])
transfer_cost_col = find_col(recommendations, ["transfer_cost", "costo_transferencia", "transfer_cost_mxn"])
units_col = find_col(recommendations, ["units_to_transfer", "unidades_transferir", "recommended_units"])
origin_col = find_col(recommendations, ["cedi_origen", "origin_cedi", "source_cedi"])

# Crear columnas auxiliares si no existen para evitar errores visuales.
rec = recommendations.copy()
if risk_col and risk_col in rec.columns:
    rec["_risk_pct"] = rec[risk_col].astype(float).apply(lambda x: x * 100 if x <= 1 else x)
else:
    rec["_risk_pct"] = 0.0
if benefit_col and benefit_col in rec.columns:
    rec["_benefit"] = pd.to_numeric(rec[benefit_col], errors="coerce").fillna(0)
else:
    rec["_benefit"] = 0.0
if expected_loss_col and expected_loss_col in rec.columns:
    rec["_expected_loss"] = pd.to_numeric(rec[expected_loss_col], errors="coerce").fillna(0)
else:
    rec["_expected_loss"] = 0.0
if transfer_cost_col and transfer_cost_col in rec.columns:
    rec["_transfer_cost"] = pd.to_numeric(rec[transfer_cost_col], errors="coerce").fillna(0)
else:
    rec["_transfer_cost"] = 0.0

# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Panel de control")
st.sidebar.caption("Filtra la vista para explicar un SKU/CEDI específico en la demo.")

selected_sku = "Todos"
selected_cedi = "Todos"
filtered = rec.copy()

if sku_col and sku_col in rec.columns:
    sku_options = ["Todos"] + sorted(rec[sku_col].dropna().astype(str).unique().tolist())
    selected_sku = st.sidebar.selectbox("SKU", sku_options)
    if selected_sku != "Todos":
        filtered = filtered[filtered[sku_col].astype(str) == selected_sku]

if cedi_col and cedi_col in rec.columns:
    cedi_options = ["Todos"] + sorted(rec[cedi_col].dropna().astype(str).unique().tolist())
    selected_cedi = st.sidebar.selectbox("CEDI destino", cedi_options)
    if selected_cedi != "Todos":
        filtered = filtered[filtered[cedi_col].astype(str) == selected_cedi]

min_risk = st.sidebar.slider("Riesgo mínimo (%)", 0, 100, 0, 5)
filtered = filtered[filtered["_risk_pct"] >= min_risk]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Cómo leer esta demo")
st.sidebar.markdown(
    """
1. Revisa los **KPIs ejecutivos**.  
2. Identifica **SKU/CEDI crítico**.  
3. Mira la **recomendación costo-beneficio**.  
4. Usa el **chat** para explicar la decisión.  
5. Cierra con la **arquitectura cloud-ready**.  
"""
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Raíz detectada: `{ROOT_DIR}`")


# ============================================================
# 8. HERO PRINCIPAL
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📦 Herdez Smart-Supply</h1>
        <p><b>ML + Decision Engine + Agentes A2A-lite</b> para anticipar quiebres de stock y recomendar acciones logísticas.</p>
        <p>Diseñado como prototipo <b>local-first</b>, portable y escalable a GCP.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
    <b>Idea central:</b> el modelo XGBoost predice riesgo de quiebre; el motor determinista calcula costo-beneficio; 
    los agentes A2A-lite explican y auditan la recomendación. El LLM no inventa números: interpreta artefactos estructurados.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 9. KPIs EJECUTIVOS
# ============================================================

st.subheader("1) Resumen ejecutivo")

num_alerts = len(filtered)
high_risk_count = int((filtered["_risk_pct"] >= 70).sum())
total_expected_loss = float(filtered["_expected_loss"].sum())
total_transfer_cost = float(filtered["_transfer_cost"].sum())
total_net_benefit = float(filtered["_benefit"].sum())
avg_risk = float(filtered["_risk_pct"].mean()) if len(filtered) else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Alertas evaluadas", f"{num_alerts:,}", help="Número de recomendaciones filtradas que está mostrando el dashboard.")
k2.metric("Riesgo promedio", f"{avg_risk:.1f}%", help="Promedio de probabilidad de quiebre de stock en las alertas visibles.")
k3.metric("Alertas alto riesgo", f"{high_risk_count:,}", help="Alertas con riesgo igual o superior a 70%.")
k4.metric("Pérdida esperada", money(total_expected_loss), help="Estimación financiera si no se actúa ante los quiebres.")
k5.metric("Beneficio neto", money(total_net_benefit), help="Pérdida evitada menos costo logístico de transferencia.")

st.markdown(
    """
    <div class="explain-box">
    <b>Cómo explicarlo al Director de Supply Chain:</b> estas métricas convierten el modelo en una decisión de negocio.
    No solo decimos “hay riesgo”; mostramos cuánto se puede perder, cuánto cuesta actuar y cuál es el beneficio neto estimado.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 10. RECOMENDACIONES PRIORIZADAS
# ============================================================

st.subheader("2) Recomendaciones priorizadas")

if filtered.empty:
    st.warning("No hay recomendaciones con los filtros actuales.")
else:
    display_cols = []
    for col in [sku_col, cedi_col, origin_col, action_col, units_col, risk_col, expected_loss_col, transfer_cost_col, benefit_col]:
        if col and col in filtered.columns and col not in display_cols:
            display_cols.append(col)

    table = filtered.sort_values("_benefit", ascending=False).copy()
    table_to_show = table[display_cols].head(25) if display_cols else table.head(25)
    st.dataframe(table_to_show, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="info-box">
        <b>Cómo leer esta tabla:</b> cada fila representa una decisión candidata. El sistema prioriza casos donde el riesgo es alto,
        la pérdida esperada es significativa y el beneficio neto de mover inventario supera el costo logístico.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sku_col and cedi_col and len(table):
        top = table.iloc[0]
        top_sku = top.get(sku_col, "SKU no disponible")
        top_cedi = top.get(cedi_col, "CEDI no disponible")
        top_origin = top.get(origin_col, "origen sugerido") if origin_col else "origen sugerido"
        top_action = top.get(action_col, "acción sugerida") if action_col else "acción sugerida"
        top_units = top.get(units_col, 0) if units_col else 0
        st.markdown(
            f"""
            <div class="warning-box">
            <b>Recomendación más relevante:</b> para <b>{top_sku}</b> en <b>{top_cedi}</b>,
            la acción sugerida es <b>{top_action}</b>. Si aplica, transferir aproximadamente <b>{top_units}</b> unidades desde <b>{top_origin}</b>.
            El beneficio neto estimado es <b>{money(top['_benefit'])}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 11. VISUALIZACIONES PRINCIPALES
# ============================================================

st.subheader("3) Visualizaciones para entender el problema")

if px is None:
    st.warning("Plotly no está instalado. Se omiten gráficas interactivas.")
else:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Riesgo por SKU")
        st.caption("Muestra qué productos concentran más riesgo de quiebre. Sirve para priorizar categorías críticas.")
        if not risk_by_sku.empty:
            # Detectar columnas flexibles.
            r_sku = find_col(risk_by_sku, ["SKU_ID", "sku", "SKU"])
            r_rate = find_col(risk_by_sku, ["risk_rate", "tasa_riesgo", "Riesgo_Quiebre", "riesgo_promedio"])
            r_count = find_col(risk_by_sku, ["risk_count", "riesgos", "sum", "Riesgos"])
            y_col = r_rate or r_count or risk_by_sku.columns[-1]
            x_col = r_sku or risk_by_sku.columns[0]
            chart_df = risk_by_sku.copy()
            fig = px.bar(chart_df, x=x_col, y=y_col, text_auto=True, title="SKUs con mayor riesgo")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        elif sku_col:
            tmp = filtered.groupby(sku_col, as_index=False)["_risk_pct"].mean().sort_values("_risk_pct", ascending=False)
            fig = px.bar(tmp, x=sku_col, y="_risk_pct", text_auto=".1f", title="Riesgo promedio por SKU")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos suficientes para riesgo por SKU.")

    with c2:
        st.markdown("#### Riesgo por CEDI")
        st.caption("Muestra en qué centro de distribución se concentra el problema operativo.")
        if not risk_by_cedi.empty:
            r_cedi = find_col(risk_by_cedi, ["CEDI", "cedi"])
            r_rate = find_col(risk_by_cedi, ["risk_rate", "tasa_riesgo", "Riesgo_Quiebre", "riesgo_promedio"])
            r_count = find_col(risk_by_cedi, ["risk_count", "riesgos", "sum", "Riesgos"])
            y_col = r_rate or r_count or risk_by_cedi.columns[-1]
            x_col = r_cedi or risk_by_cedi.columns[0]
            fig = px.bar(risk_by_cedi, x=x_col, y=y_col, text_auto=True, title="CEDIs con mayor riesgo")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        elif cedi_col:
            tmp = filtered.groupby(cedi_col, as_index=False)["_risk_pct"].mean().sort_values("_risk_pct", ascending=False)
            fig = px.bar(tmp, x=cedi_col, y="_risk_pct", text_auto=".1f", title="Riesgo promedio por CEDI")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos suficientes para riesgo por CEDI.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Beneficio neto por recomendación")
        st.caption("Ayuda a priorizar acciones con mayor impacto financiero.")
        if not filtered.empty and sku_col and cedi_col:
            tmp = filtered.copy().sort_values("_benefit", ascending=False).head(12)
            tmp["SKU_CEDI"] = tmp[sku_col].astype(str) + " | " + tmp[cedi_col].astype(str)
            fig = px.bar(tmp, x="_benefit", y="SKU_CEDI", orientation="h", title="Top recomendaciones por beneficio neto")
            fig.update_layout(height=500, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay recomendaciones para graficar beneficio neto.")

    with c4:
        st.markdown("#### Pérdida esperada vs costo logístico")
        st.caption("Si la pérdida esperada es mayor que el costo, transferir inventario puede tener sentido económico.")
        if not filtered.empty and sku_col and cedi_col:
            tmp = filtered.copy().sort_values("_expected_loss", ascending=False).head(12)
            tmp["SKU_CEDI"] = tmp[sku_col].astype(str) + " | " + tmp[cedi_col].astype(str)
            melted = tmp.melt(
                id_vars=["SKU_CEDI"],
                value_vars=["_expected_loss", "_transfer_cost"],
                var_name="Métrica",
                value_name="MXN",
            )
            melted["Métrica"] = melted["Métrica"].replace({"_expected_loss": "Pérdida esperada", "_transfer_cost": "Costo transferencia"})
            fig = px.bar(melted, x="SKU_CEDI", y="MXN", color="Métrica", barmode="group", title="Comparación económica")
            fig.update_layout(height=500, xaxis_tickangle=-35, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos para comparar pérdida y costo.")


# ============================================================
# 12. MODELO PREDICTIVO Y EDA
# ============================================================

st.subheader("4) Modelo predictivo y calidad del análisis")

m1, m2 = st.columns([1.1, 1])

with m1:
    st.markdown("#### Comparación de modelos")
    st.caption("El objetivo no es solo accuracy: en supply chain importa detectar quiebres reales sin generar demasiadas falsas alarmas.")
    if not model_metrics.empty:
        st.dataframe(model_metrics, use_container_width=True, hide_index=True)
        if px is not None:
            metric_cols = [c for c in model_metrics.columns if c.lower() in ["accuracy", "precision_risk", "recall_risk", "f1_risk", "roc_auc", "f1"]]
            model_col = find_col(model_metrics, ["model", "Modelo", "modelo"])
            if model_col and metric_cols:
                plot_df = model_metrics[[model_col] + metric_cols].melt(id_vars=model_col, var_name="Métrica", value_name="Valor")
                fig = px.bar(plot_df, x=model_col, y="Valor", color="Métrica", barmode="group", title="Desempeño comparativo")
                fig.update_layout(height=420, xaxis_tickangle=-25, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No encontré métricas de modelos.")

with m2:
    st.markdown("#### Por qué XGBoost tiene sentido aquí")
    st.markdown(
        """
        <div class="section-card">
        <b>XGBoost</b> es adecuado para datos tabulares porque puede capturar relaciones no lineales entre ventas, stock,
        lead time, promociones, clima y costos. Además, permite regularización y suele funcionar bien con datasets estructurados.
        <br><br>
        <b>Decisión importante:</b> el modelo no decide mover inventario. Solo estima riesgo. La decisión final pasa por el motor costo-beneficio.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not time_series.empty and px is not None:
        date_col = find_col(time_series, ["Fecha", "fecha", "date"])
        ventas_col = find_col(time_series, ["Ventas_Unidades", "ventas_totales", "total_sales", "ventas"])
        riesgo_col_ts = find_col(time_series, ["riesgo_promedio", "risk_mean", "Riesgo_Quiebre", "risk_rate"])
        if date_col and (ventas_col or riesgo_col_ts):
            y = ventas_col or riesgo_col_ts
            fig = px.line(time_series, x=date_col, y=y, markers=True, title="Tendencia temporal del histórico")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 13. AGENTES A2A-LITE
# ============================================================

st.subheader("5) Agentes A2A-lite: explicación y trazabilidad")

st.markdown(
    """
    <div class="info-box">
    <b>Qué significa A2A-lite aquí:</b> cada agente tiene una responsabilidad clara, recibe un artefacto estructurado y produce otro.
    En producción, esos agentes podrían convertirse en servicios independientes interoperando por A2A real.
    </div>
    """,
    unsafe_allow_html=True,
)

ac1, ac2 = st.columns([1, 1])
with ac1:
    st.markdown("#### Tarjetas de agentes")
    if not agent_registry_df.empty:
        st.dataframe(agent_registry_df, use_container_width=True, hide_index=True)
    else:
        st.info("No encontré registry de agentes.")

with ac2:
    st.markdown("#### Brief ejecutivo del agente")
    if agent_brief.strip():
        st.markdown(agent_brief)
    else:
        st.info("No encontré brief del agente.")

with st.expander("Ver artefactos estructurados de agentes"):
    if agent_artifacts:
        for idx, artifact in enumerate(agent_artifacts, start=1):
            agent_name = artifact.get("agent") or artifact.get("agent_name") or artifact.get("name") or f"Agente {idx}"
            st.markdown(f"**{idx}. {agent_name}**")
            st.json(artifact)
    else:
        st.info("No encontré artefactos A2A-lite.")

with st.expander("Ver traza técnica"):
    if agent_trace_obj is not None:
        st.json(agent_trace_obj)
    else:
        st.info("No encontré traza técnica del agente.")


# ============================================================
# 14. SIMULADOR DE DECISIÓN
# ============================================================

st.subheader("6) Simulador de costo-beneficio")
st.caption("Usa esta sección para explicar la lógica de negocio sin entrar en código.")

sim1, sim2, sim3, sim4 = st.columns(4)
with sim1:
    sim_risk = st.slider("Probabilidad de quiebre", 0.0, 1.0, 0.80, 0.05)
with sim2:
    sim_daily_loss = st.number_input("Costo de quiebre diario (MXN)", min_value=0.0, value=4500.0, step=500.0)
with sim3:
    sim_lead_time = st.number_input("Lead time (días)", min_value=1, value=5, step=1)
with sim4:
    sim_transfer_cost = st.number_input("Costo total transferencia (MXN)", min_value=0.0, value=6000.0, step=500.0)

sim_expected_loss = sim_risk * sim_daily_loss * sim_lead_time
sim_net_benefit = sim_expected_loss - sim_transfer_cost

s1, s2, s3 = st.columns(3)
s1.metric("Pérdida esperada", money(sim_expected_loss))
s2.metric("Costo transferencia", money(sim_transfer_cost))
s3.metric("Beneficio neto", money(sim_net_benefit))

if sim_net_benefit > 0:
    st.success("Recomendación simulada: transferir inventario puede ser económicamente conveniente.")
else:
    st.warning("Recomendación simulada: esperar o monitorear puede ser mejor porque el costo supera la pérdida esperada.")


# ============================================================
# 15. ARQUITECTURA
# ============================================================

st.subheader("7) Arquitectura de la solución")

arch1, arch2 = st.columns(2)
with arch1:
    st.markdown(
        """
        <div class="section-card">
        <h4>MVP local-first</h4>
        <ol>
            <li>Excel histórico</li>
            <li>Pandas / Feature Engineering</li>
            <li>XGBoost para riesgo de quiebre</li>
            <li>Decision Engine costo-beneficio</li>
            <li>Agentes A2A-lite</li>
            <li>Streamlit Dashboard</li>
        </ol>
        <p class="small-muted">Ventaja: bajo costo, portable, explicable y rápido para demo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with arch2:
    st.markdown(
        """
        <div class="section-card">
        <h4>Ruta cloud-ready en GCP</h4>
        <ol>
            <li>Cloud Storage / Pub/Sub / Dataflow</li>
            <li>BigQuery como data warehouse</li>
            <li>Vertex AI Pipelines y Training</li>
            <li>Vertex AI Model Registry / Monitoring</li>
            <li>Cloud Run para servicios de tools</li>
            <li>Agent Platform / A2A para agentes interoperables</li>
        </ol>
        <p class="small-muted">Ventaja: escalabilidad, monitoreo, MLOps y gobierno empresarial.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 16. CHAT EJECUTIVO CON CONTEXTO
# ============================================================

st.subheader("8) Chat ejecutivo con el agente")

st.markdown(
    """
    <div class="explain-box">
    Este chat responde usando el contexto real del dashboard: recomendaciones, KPIs, métricas del modelo y brief de agentes.
    Si configuras <b>GEMINI_API_KEY</b>, puede usar Gemini para respuestas más naturales. Si no, usa un modo local explicativo.
    </div>
    """,
    unsafe_allow_html=True,
)

# Inicializar historial
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hola, soy el copiloto de Supply Chain. Puedes preguntarme cosas como: "
                "¿cuál es la alerta más importante?, ¿por qué conviene transferir inventario?, "
                "¿cómo funciona el modelo?, ¿cómo explico esto al director?"
            ),
        }
    ]


def build_context_summary() -> str:
    top_lines = []
    if not filtered.empty:
        top_df = filtered.sort_values("_benefit", ascending=False).head(5)
        for _, row in top_df.iterrows():
            sku = row.get(sku_col, "SKU") if sku_col else "SKU"
            cedi = row.get(cedi_col, "CEDI") if cedi_col else "CEDI"
            action = row.get(action_col, "acción") if action_col else "acción"
            origin = row.get(origin_col, "N/A") if origin_col else "N/A"
            units = row.get(units_col, "N/A") if units_col else "N/A"
            top_lines.append(
                f"- {sku} en {cedi}: acción={action}, origen={origin}, unidades={units}, "
                f"riesgo={row['_risk_pct']:.1f}%, beneficio={money(row['_benefit'])}."
            )
    metrics_summary = ""
    if not model_metrics.empty:
        metrics_summary = model_metrics.head(10).to_string(index=False)
    return f"""
Contexto del dashboard Herdez Smart-Supply:
- Alertas visibles: {num_alerts}
- Riesgo promedio: {avg_risk:.1f}%
- Alertas de alto riesgo: {high_risk_count}
- Pérdida esperada total: {money(total_expected_loss)}
- Costo total de transferencia: {money(total_transfer_cost)}
- Beneficio neto total: {money(total_net_benefit)}

Top recomendaciones:
{chr(10).join(top_lines) if top_lines else 'No hay recomendaciones filtradas.'}

Métricas de modelos:
{metrics_summary}

Brief del agente:
{agent_brief[:2500] if agent_brief else 'No disponible'}
"""


def local_answer(question: str) -> str:
    q = question.lower()
    context_intro = (
        f"Con los filtros actuales veo {num_alerts} alertas, riesgo promedio de {avg_risk:.1f}% "
        f"y un beneficio neto agregado de {money(total_net_benefit)}. "
    )

    if any(word in q for word in ["importante", "prioridad", "principal", "top", "urgente"]):
        if filtered.empty:
            return "No hay alertas con los filtros actuales. Baja el umbral de riesgo o selecciona 'Todos'."
        top = filtered.sort_values("_benefit", ascending=False).iloc[0]
        sku = top.get(sku_col, "SKU") if sku_col else "SKU"
        cedi = top.get(cedi_col, "CEDI") if cedi_col else "CEDI"
        action = top.get(action_col, "acción recomendada") if action_col else "acción recomendada"
        origin = top.get(origin_col, "origen sugerido") if origin_col else "origen sugerido"
        units = top.get(units_col, "N/A") if units_col else "N/A"
        return (
            f"La prioridad principal es **{sku} en {cedi}**. "
            f"El sistema recomienda **{action}**, con origen **{origin}** y aproximadamente **{units} unidades**. "
            f"El riesgo estimado es **{top['_risk_pct']:.1f}%** y el beneficio neto esperado es **{money(top['_benefit'])}**. "
            "Esto significa que la pérdida esperada por no actuar supera el costo logístico de transferencia."
        )

    if any(word in q for word in ["modelo", "xgboost", "machine", "ml", "predice"]):
        return (
            "El modelo predictivo estima la probabilidad de quiebre de stock por combinación SKU/CEDI. "
            "Usamos variables como ventas, stock actual, lead time, promoción, clima y costos. "
            "XGBoost es útil porque trabaja bien con datos tabulares y captura relaciones no lineales. "
            "Importante: el modelo no decide mover inventario; solo estima riesgo. La decisión se toma después con el motor costo-beneficio."
        )

    if any(word in q for word in ["transfer", "mover", "inventario", "beneficio", "costo"]):
        return (
            context_intro
            + "La lógica es: **beneficio neto = pérdida esperada evitada - costo de transferencia**. "
            "Si el beneficio neto es positivo y el CEDI origen no queda en riesgo, se recomienda transferir. "
            "Si el costo logístico supera la pérdida esperada, se recomienda esperar, monitorear o escalar reabasto."
        )

    if any(word in q for word in ["director", "negocio", "explicar", "presentar"]):
        return (
            "Para explicarlo a negocio diría: 'El sistema prioriza productos y CEDIs donde existe riesgo de quiebre, "
            "estima la pérdida económica si no actuamos y compara esa pérdida contra el costo de mover inventario. "
            "La recomendación final busca mejorar nivel de servicio y evitar ventas perdidas sin gastar logística innecesaria.'"
        )

    if any(word in q for word in ["arquitectura", "gcp", "cloud", "a2a"]):
        return (
            "La arquitectura MVP es local-first: Excel, pandas, XGBoost, decision engine, agentes A2A-lite y Streamlit. "
            "La ruta cloud-ready en GCP sería BigQuery para datos, Vertex AI para entrenamiento/registro/monitoreo, "
            "Cloud Run para tools deterministas y Agent Platform/A2A para agentes interoperables."
        )

    return (
        context_intro
        + "Puedes preguntarme por la alerta más importante, por qué se recomienda transferir inventario, "
        "cómo funciona XGBoost, cómo explicarlo al director o cómo escalarlo a GCP."
    )


def gemini_answer(question: str) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=api_key,
        )
        system_prompt = """
Eres un copiloto ejecutivo de Supply Chain para el proyecto Herdez Smart-Supply.
Tu objetivo es explicar recomendaciones de inventario de forma clara, útil y orientada a negocio.

Reglas:
- Responde en español.
- Usa SOLO el contexto proporcionado.
- No inventes métricas ni montos.
- Si falta información, dilo y sugiere cómo obtenerla.
- Explica para dos públicos cuando sea útil: Director de Supply Chain y Gerente de IA.
- Sé claro, estructurado y accionable.
"""
        user_prompt = f"""
{build_context_summary()}

Pregunta del usuario:
{question}
"""
        resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        return getattr(resp, "content", str(resp))
    except Exception as exc:
        return f"No pude usar Gemini en este momento. Respuesta local:\n\n{local_answer(question)}\n\nDetalle técnico: {exc}"


use_gemini = st.toggle("Usar Gemini si hay API key disponible", value=False)

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Pregunta al agente sobre riesgo, recomendaciones, modelo o arquitectura...")
if user_question:
    st.session_state.chat_messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if use_gemini:
        answer = gemini_answer(user_question) or local_answer(user_question)
    else:
        answer = local_answer(user_question)

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)


# ============================================================
# 17. CIERRE
# ============================================================

st.markdown("---")
st.markdown(
    """
    <p class="small-muted">
    Herdez Smart-Supply · Prototipo técnico local-first · ML predictivo + motor determinista + agentes A2A-lite · Streamlit Dashboard
    </p>
    """,
    unsafe_allow_html=True,
)
