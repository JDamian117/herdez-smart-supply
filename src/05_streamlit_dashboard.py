# ============================================================
# 05_streamlit_dashboard.py
# Herdez Smart-Supply | Dashboard ejecutivo premium
# ------------------------------------------------------------
# Objetivo del archivo:
#   Convertir los artefactos generados por el pipeline 01-04 en
#   una experiencia ejecutiva, guiada y entendible para usuarios
#   sin conocimientos técnicos ni de Supply Chain.
#
# Flujo del sistema:
#   01 EDA/Features  -> outputs/herdez_features_dataset.csv
#   02 Modelos ML    -> outputs/model_comparison_metrics.csv
#   03 Decision Eng. -> outputs/decision_recommendations.csv
#   04 Agentes       -> outputs/agent_executive_brief_a2a_simple.md
#   05 Dashboard     -> este archivo, solo visualiza y explica
#
# Cómo correr:
#   python -m streamlit run app.py
#   o
#   python -m streamlit run src/05_streamlit_dashboard.py
# ============================================================

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ocultar el icono de GitHub y el menú de Streamlit
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


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
# 2. UTILIDADES DE RUTA Y CARGA DE DATOS
# ============================================================


def find_project_root() -> Path:
    """Encuentra la raíz del proyecto aunque Streamlit se ejecute desde app.py o desde src/.

    Buscamos una carpeta que contenga app.py, outputs/ o requirements.txt.
    Esto evita errores típicos de rutas relativas en Streamlit Cloud y Windows.
    """
    candidates = [Path.cwd()]
    if "__file__" in globals():
        current = Path(__file__).resolve()
        candidates.extend([current.parent, current.parent.parent])

    for base in candidates:
        if (base / "outputs").exists() or (base / "app.py").exists() or (base / "requirements.txt").exists():
            return base

    return Path.cwd()


ROOT_DIR = find_project_root()
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"


@st.cache_data(show_spinner=False)
def load_csv(relative_path: str) -> pd.DataFrame:
    """Carga CSV desde la raíz del proyecto de forma segura."""
    path = ROOT_DIR / relative_path
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"No pude leer {relative_path}: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(relative_path: str) -> Any:
    """Carga JSON desde la raíz del proyecto de forma segura."""
    path = ROOT_DIR / relative_path
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.warning(f"No pude leer {relative_path}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_text(relative_path: str) -> str:
    """Carga texto/markdown desde la raíz del proyecto."""
    path = ROOT_DIR / relative_path
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


# Carga principal de artefactos del pipeline.
decisions = load_csv("outputs/decision_recommendations.csv")
alerts = load_csv("outputs/scored_stockout_alerts.csv")
model_metrics = load_csv("outputs/model_comparison_metrics.csv")
risk_by_sku = load_csv("outputs/risk_by_sku.csv")
risk_by_cedi = load_csv("outputs/risk_by_cedi.csv")
risk_by_sku_cedi = load_csv("outputs/risk_by_sku_cedi.csv")
time_series = load_csv("outputs/time_series_daily_summary.csv")
features = load_csv("outputs/herdez_features_dataset.csv")
agent_registry = load_json("outputs/agent_a2a_simple_registry.json")
agent_artifacts = load_json("outputs/agent_a2a_simple_artifacts.json") or {}
agent_trace = load_json("outputs/agent_a2a_simple_trace.json") or {}
agent_brief = load_text("outputs/agent_executive_brief_a2a_simple.md")


# ============================================================
# 3. CSS PREMIUM: CONTRASTE, TARJETAS Y SISTEMA VISUAL
# ============================================================


def inject_css() -> None:
    """Inyecta CSS personalizado.

    Principios aplicados:
    - Contraste suficiente en modo claro/oscuro.
    - Cards con border-radius 12px y sombra suave.
    - Jerarquía visual para reducir carga cognitiva.
    - Paleta sobria: azul apagado, carbón, verde operativo y rojo alerta.
    """
    st.markdown(
        """
        <style>
        :root {
            --bg-soft: rgba(248, 250, 252, 0.92);
            --card-bg: rgba(255, 255, 255, 0.94);
            --card-border: rgba(15, 23, 42, 0.10);
            --text-main: #0f172a;
            --text-muted: #64748b;
            --muted-blue: #3b6f9e;
            --muted-blue-2: #6b8faf;
            --charcoal: #1f2937;
            --danger: #b91c1c;
            --danger-soft: #fee2e2;
            --success: #15803d;
            --success-soft: #dcfce7;
            --warning: #b45309;
            --warning-soft: #fef3c7;
            --info-soft: #e0f2fe;
            --shadow-soft: 0 12px 28px rgba(15, 23, 42, 0.08);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg-soft: rgba(15, 23, 42, 0.72);
                --card-bg: rgba(30, 41, 59, 0.92);
                --card-border: rgba(226, 232, 240, 0.12);
                --text-main: #f8fafc;
                --text-muted: #cbd5e1;
                --danger-soft: rgba(127, 29, 29, 0.42);
                --success-soft: rgba(20, 83, 45, 0.42);
                --warning-soft: rgba(120, 53, 15, 0.42);
                --info-soft: rgba(12, 74, 110, 0.42);
                --shadow-soft: 0 12px 28px rgba(0, 0, 0, 0.28);
            }
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 4rem;
            max-width: 1400px;
        }

        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 48%, #3b6f9e 100%);
            color: white;
            border-radius: 20px;
            padding: 28px 30px;
            margin-bottom: 24px;
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.24);
        }

        .hero h1 {
            font-size: 2.35rem;
            margin: 0 0 8px 0;
            letter-spacing: -0.03em;
        }

        .hero p {
            color: rgba(255,255,255,0.86);
            font-size: 1.03rem;
            max-width: 980px;
            margin: 0;
            line-height: 1.55;
        }

        .section-title {
            margin-top: 18px;
            margin-bottom: 8px;
        }

        .section-title h2 {
            font-size: 1.48rem;
            margin: 0;
            color: var(--text-main);
            letter-spacing: -0.02em;
        }

        .section-title p {
            margin: 6px 0 0 0;
            color: var(--text-muted);
            font-size: 0.98rem;
            line-height: 1.45;
        }

        .premium-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 18px 18px;
            box-shadow: var(--shadow-soft);
            margin-bottom: 16px;
        }

        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 18px 16px;
            box-shadow: var(--shadow-soft);
            min-height: 132px;
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.84rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 8px;
        }

        .metric-value {
            color: var(--text-main);
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 6px;
        }

        .metric-help {
            color: var(--text-muted);
            font-size: 0.86rem;
            line-height: 1.35;
        }

        .accent-danger { border-left: 5px solid var(--danger); }
        .accent-success { border-left: 5px solid var(--success); }
        .accent-info { border-left: 5px solid var(--muted-blue); }
        .accent-warning { border-left: 5px solid var(--warning); }

        .pill {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 6px;
            margin-bottom: 6px;
        }

        .pill-danger { background: var(--danger-soft); color: var(--danger); }
        .pill-success { background: var(--success-soft); color: var(--success); }
        .pill-info { background: var(--info-soft); color: var(--muted-blue); }
        .pill-warning { background: var(--warning-soft); color: var(--warning); }

        .explain-box {
            background: var(--bg-soft);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 14px 16px;
            color: var(--text-main);
            margin-top: 10px;
            margin-bottom: 12px;
        }

        .explain-box strong {
            color: var(--text-main);
        }

        .analogy {
            border-left: 4px solid var(--muted-blue);
            background: var(--info-soft);
            padding: 12px 14px;
            border-radius: 10px;
            color: var(--text-main);
            margin: 10px 0;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        div[data-testid="stMetric"] {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            padding: 1rem;
            border-radius: 12px;
            box-shadow: var(--shadow-soft);
        }

        .small-note {
            font-size: 0.86rem;
            color: var(--text-muted);
            line-height: 1.45;
        }

        .footer-note {
            color: var(--text-muted);
            font-size: 0.86rem;
            text-align: center;
            margin-top: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ============================================================
# 4. FUNCIONES DE FORMATO Y TARJETAS
# ============================================================


def money(value: Any) -> str:
    """Formato moneda MXN robusto."""
    try:
        if pd.isna(value):
            return "$0"
        return f"${float(value):,.0f} MXN"
    except Exception:
        return "$0 MXN"



def pct(value: Any, decimals: int = 1) -> str:
    """Formato porcentaje robusto.

    Si el valor viene como 0.28, se muestra 28.0%.
    Si viene como 28, se muestra 28.0% también.
    """
    try:
        v = float(value)
        if abs(v) <= 1:
            v *= 100
        return f"{v:.{decimals}f}%"
    except Exception:
        return "0.0%"



def number(value: Any) -> str:
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "0"



def section(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )



def metric_card(label: str, value: str, help_text: str, accent: str = "info") -> None:
    st.markdown(
        f"""
        <div class="metric-card accent-{accent}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def explanation(title: str, body: str, kind: str = "info") -> None:
    pill_class = {
        "danger": "pill-danger",
        "success": "pill-success",
        "warning": "pill-warning",
        "info": "pill-info",
    }.get(kind, "pill-info")
    st.markdown(
        f"""
        <div class="explain-box">
            <span class="pill {pill_class}">{title}</span><br>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )



def analogy(text: str) -> None:
    st.markdown(f"<div class='analogy'>💡 <strong>Analogía:</strong> {text}</div>", unsafe_allow_html=True)



def clean_action(action: Any) -> str:
    mapping = {
        "TRANSFER_INVENTORY": "Transferir inventario",
        "EXPEDITE_REPLENISHMENT_OR_REVIEW": "Reabasto urgente / revisión",
        "MONITOR": "Monitorear",
        "WAIT": "Esperar reabasto",
        "NO_ACTION": "Sin acción",
    }
    return mapping.get(str(action), str(action))


# ============================================================
# 5. VALIDACIÓN DE ARTEFACTOS
# ============================================================


required = {
    "outputs/decision_recommendations.csv": not decisions.empty,
    "outputs/scored_stockout_alerts.csv": not alerts.empty,
    "outputs/model_comparison_metrics.csv": not model_metrics.empty,
}

if not all(required.values()):
    st.markdown(
        """
        <div class="hero">
            <h1>📦 Herdez Smart-Supply</h1>
            <p>El dashboard está listo, pero faltan artefactos generados por el pipeline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error("Faltan archivos en outputs/. Ejecuta primero los scripts 01, 02, 03 y 04.")
    st.code(
        """python src/01_eda_target_features.py --input data/Data_Prueba_Tecnica_Herdez_IA.xlsx
python src/02_train_models.py --processed outputs/herdez_features_dataset.csv
python src/03_decision_engine.py --processed outputs/herdez_features_dataset.csv --model models/best_stockout_model.joblib
python src/04_agent_system_a2a_simple.py --mode fallback --max-alerts 8
python -m streamlit run app.py""",
        language="bash",
    )
    st.stop()


# Normalización ligera de fechas y acciones.
for df in [decisions, alerts, features, time_series]:
    if not df.empty and "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

if not decisions.empty and "Accion_Recomendada" in decisions.columns:
    decisions["Accion_Legible"] = decisions["Accion_Recomendada"].apply(clean_action)


# ============================================================
# 6. SIDEBAR: CENTRO DE CONTROL GUIADO
# ============================================================

with st.sidebar:
    st.markdown("## 🎛️ Centro de Control Guiado")
    st.caption("Usa este panel como manual interactivo. Está diseñado para explicar el caso sin asumir conocimiento previo de Supply Chain.")

    with st.expander("🧭 ¿Cómo leo este dashboard?", expanded=True):
        st.markdown(
            """
            **Lee de arriba hacia abajo:**
            1. **Estado:** ve si hay riesgo operativo.
            2. **Exploración:** identifica SKU, CEDI y costo.
            3. **Contexto:** entiende por qué el agente recomienda mover o esperar.

            La idea no es ver muchas gráficas; es responder una pregunta:
            **¿qué acción evita más pérdida con menor costo?**
            """
        )

    with st.expander("👥 Historias de usuario"):
        st.markdown(
            """
            **Director de Supply Chain:**
            > Quiero saber qué productos están en riesgo y qué acción reduce ventas perdidas.

            **Gerente de IA:**
            > Quiero ver separación entre modelo ML, motor determinista y agente explicativo.

            **Analista operativo:**
            > Quiero filtrar por CEDI o SKU para decidir prioridades del día.
            """
        )

    with st.expander("💬 Preguntas que puedes probar en el chat"):
        suggested_questions = [
            "¿Cuál es la alerta más importante y por qué?",
            "¿Cómo sé si me voy a quedar sin producto la próxima semana?",
            "¿Qué significa que el riesgo esté en color rojo?",
            "¿Por qué no siempre conviene mover inventario?",
            "¿Cómo puedo reducir costo logístico usando este dashboard?",
            "Explícame la arquitectura para alguien de negocio.",
            "Explícame la arquitectura para el gerente técnico.",
        ]
        for q in suggested_questions:
            if st.button(q, key=f"suggest_{q}"):
                st.session_state["pending_question"] = q

    st.divider()
    st.markdown("### 🔎 Filtros de exploración")

    cedis = sorted(decisions["CEDI_Destino"].dropna().unique().tolist()) if "CEDI_Destino" in decisions else []
    skus = sorted(decisions["SKU_ID"].dropna().unique().tolist()) if "SKU_ID" in decisions else []
    actions = sorted(decisions["Accion_Legible"].dropna().unique().tolist()) if "Accion_Legible" in decisions else []

    selected_cedis = st.multiselect("CEDI destino", cedis, default=cedis)
    selected_skus = st.multiselect("SKU", skus, default=skus)
    selected_actions = st.multiselect("Acción recomendada", actions, default=actions)

    min_risk = st.slider("Riesgo mínimo", 0, 100, 0, 5)
    top_n = st.slider("Número de alertas visibles", 5, 50, 15, 5)

    st.divider()
    st.caption("Versión local-first: el dashboard lee artefactos preprocesados. No reentrena modelos en cada carga.")


# ============================================================
# 7. FILTRADO DE DATOS
# ============================================================

filtered = decisions.copy()
if selected_cedis and "CEDI_Destino" in filtered.columns:
    filtered = filtered[filtered["CEDI_Destino"].isin(selected_cedis)]
if selected_skus and "SKU_ID" in filtered.columns:
    filtered = filtered[filtered["SKU_ID"].isin(selected_skus)]
if selected_actions and "Accion_Legible" in filtered.columns:
    filtered = filtered[filtered["Accion_Legible"].isin(selected_actions)]
if "Riesgo_Probabilidad" in filtered.columns:
    filtered = filtered[filtered["Riesgo_Probabilidad"] >= (min_risk / 100)]

sort_col = "Beneficio_Neto" if "Beneficio_Neto" in filtered.columns else "Riesgo_Probabilidad"
if sort_col in filtered.columns:
    filtered = filtered.sort_values(sort_col, ascending=False)

visible_decisions = filtered.head(top_n).copy()


# ============================================================
# 8. HERO / INTRO NARRATIVA
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📦 Herdez Smart-Supply</h1>
        <p>
        Sistema local-first de IA para anticipar quiebres de stock y recomendar acciones correctivas.
        Combina Machine Learning, cálculo costo-beneficio y agentes A2A-lite para explicar decisiones de inventario
        en lenguaje de negocio.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

explanation(
    "Objetivo del producto",
    "Este tablero responde: <strong>¿qué productos podrían quedarse sin inventario, cuánto costaría no actuar y cuándo conviene transferir inventario desde otro CEDI?</strong>",
    "info",
)


# ============================================================
# 9. CAPA 1: ESTADO MACRO
# ============================================================

section(
    "1) Capa de Estado: ¿qué tan crítica está la operación?",
    "Esta capa resume la salud del sistema. Usa colores psicológicos: rojo para alerta, verde para beneficio, azul para contexto analítico y amarillo para revisión.",
)

n_alerts = len(filtered)
high_risk_count = int((filtered.get("Nivel_Riesgo", pd.Series(dtype=str)).astype(str).str.lower() == "alto").sum()) if not filtered.empty else 0
transfer_count = int((filtered.get("Accion_Recomendada", pd.Series(dtype=str)) == "TRANSFER_INVENTORY").sum()) if not filtered.empty else 0
review_count = int((filtered.get("Accion_Recomendada", pd.Series(dtype=str)) == "EXPEDITE_REPLENISHMENT_OR_REVIEW").sum()) if not filtered.empty else 0
expected_loss = filtered.get("Perdida_Esperada_Sin_Actuar", pd.Series(dtype=float)).sum() if not filtered.empty else 0
net_benefit = filtered.get("Beneficio_Neto", pd.Series(dtype=float)).sum() if not filtered.empty else 0
avg_risk = filtered.get("Riesgo_Probabilidad", pd.Series(dtype=float)).mean() if not filtered.empty else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Alertas filtradas", number(n_alerts), "Casos SKU/CEDI que requieren atención en la vista actual.", "info")
with c2:
    metric_card("Riesgo promedio", pct(avg_risk), "Probabilidad promedio de quiebre según el modelo ML.", "danger" if avg_risk >= 0.7 else "warning")
with c3:
    metric_card("Pérdida esperada", money(expected_loss), "Estimación de ventas/servicio en riesgo si no se actúa.", "danger")
with c4:
    metric_card("Beneficio neto", money(net_benefit), "Valor potencial de actuar después de restar costo logístico.", "success")

analogy("El inventario funciona como tener comida extra en el refrigerador. Si sabes que llegan visitas antes de que puedas ir al súper, necesitas suficiente reserva; si no, tienes que pedir apoyo o comprar de emergencia.")


# ============================================================
# 10. CAPA 2: EXPLORACIÓN VISUAL
# ============================================================

section(
    "2) Capa de Exploración: ¿dónde está el problema?",
    "Aquí exploramos el riesgo por producto, CEDI y acción recomendada. Las gráficas no son decoración: cada una responde una pregunta operativa concreta.",
)

plot_template = "plotly_white"
muted_sequence = ["#3b6f9e", "#6b8faf", "#1f2937", "#7c8da5", "#94a3b8", "#0f766e", "#b45309", "#b91c1c"]

left, right = st.columns([1.15, 0.85])

with left:
    with st.container(border=True):
        st.markdown("#### 🏭 Riesgo por CEDI")
        explanation(
            "Cómo leerlo",
            "Un CEDI con mayor pérdida o más alertas puede convertirse en cuello de botella. Prioriza donde el impacto económico y operativo sea más alto.",
            "info",
        )
        if not filtered.empty and "CEDI_Destino" in filtered.columns:
            cedi_chart = (
                filtered.groupby("CEDI_Destino", as_index=False)
                .agg(
                    alertas=("SKU_ID", "count"),
                    perdida=("Perdida_Esperada_Sin_Actuar", "sum"),
                    beneficio=("Beneficio_Neto", "sum"),
                    riesgo=("Riesgo_Probabilidad", "mean"),
                )
                .sort_values("perdida", ascending=False)
            )
            fig = px.bar(
                cedi_chart,
                x="CEDI_Destino",
                y="perdida",
                color="riesgo",
                text="alertas",
                color_continuous_scale=[[0, "#6b8faf"], [0.55, "#b45309"], [1, "#b91c1c"]],
                labels={"CEDI_Destino": "CEDI destino", "perdida": "Pérdida esperada", "riesgo": "Riesgo promedio"},
            )
            fig.update_traces(texttemplate="%{text} alertas", textposition="outside")
            fig.update_layout(template=plot_template, height=410, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos para graficar por CEDI con los filtros actuales.")

with right:
    with st.container(border=True):
        st.markdown("#### 🎯 Acciones recomendadas")
        explanation(
            "Cómo leerlo",
            "No todas las alertas se resuelven moviendo inventario. Si no hay CEDI origen seguro, el sistema recomienda revisar reabasto o escalar.",
            "warning",
        )
        if not filtered.empty and "Accion_Legible" in filtered.columns:
            action_chart = filtered["Accion_Legible"].value_counts().reset_index()
            action_chart.columns = ["Acción", "Casos"]
            fig = px.pie(
                action_chart,
                names="Acción",
                values="Casos",
                hole=0.58,
                color_discrete_sequence=muted_sequence,
            )
            fig.update_layout(template=plot_template, height=410, margin=dict(l=10, r=10, t=20, b=10), showlegend=True)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay acciones con los filtros actuales.")


left2, right2 = st.columns([1, 1])

with left2:
    with st.container(border=True):
        st.markdown("#### 📦 Productos más críticos")
        explanation(
            "Cómo leerlo",
            "Si un SKU aparece arriba, no significa que sea malo; significa que necesita atención porque combina demanda, inventario y costo de quiebre.",
            "info",
        )
        if not filtered.empty and "SKU_ID" in filtered.columns:
            sku_chart = (
                filtered.groupby("SKU_ID", as_index=False)
                .agg(
                    perdida=("Perdida_Esperada_Sin_Actuar", "sum"),
                    beneficio=("Beneficio_Neto", "sum"),
                    riesgo=("Riesgo_Probabilidad", "mean"),
                )
                .sort_values("perdida", ascending=True)
            )
            fig = px.bar(
                sku_chart,
                y="SKU_ID",
                x="perdida",
                orientation="h",
                color="riesgo",
                color_continuous_scale=[[0, "#6b8faf"], [0.55, "#b45309"], [1, "#b91c1c"]],
                labels={"SKU_ID": "SKU", "perdida": "Pérdida esperada", "riesgo": "Riesgo promedio"},
            )
            fig.update_layout(template=plot_template, height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos por SKU con los filtros actuales.")

with right2:
    with st.container(border=True):
        st.markdown("#### 📈 Evolución temporal del riesgo")
        explanation(
            "Cómo leerlo",
            "Muestra si el riesgo aumenta o disminuye por día. Es útil para explicar si el problema es puntual o una tendencia.",
            "info",
        )
        if not time_series.empty and {"Fecha", "riesgo_promedio"}.issubset(time_series.columns):
            fig = px.line(
                time_series,
                x="Fecha",
                y="riesgo_promedio",
                markers=True,
                labels={"Fecha": "Fecha", "riesgo_promedio": "Riesgo promedio"},
            )
            fig.update_traces(line=dict(color="#3b6f9e", width=3), marker=dict(size=6))
            fig.update_layout(template=plot_template, height=420, margin=dict(l=10, r=10, t=20, b=10), yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No encontré la serie temporal del riesgo.")


# ============================================================
# 11. CAPA 3: CONTEXTO Y RECOMENDACIÓN OPERATIVA
# ============================================================

section(
    "3) Capa de Contexto: ¿qué decisión tomamos y por qué?",
    "Esta capa traduce datos a acción. La recomendación no sale directamente del LLM: primero se calcula con reglas y costo-beneficio.",
)

if visible_decisions.empty:
    st.warning("No hay alertas con los filtros actuales. Reduce filtros o baja el riesgo mínimo.")
else:
    # Selección de alerta para explicación detallada.
    option_labels = []
    for idx, row in visible_decisions.iterrows():
        label = f"{row.get('SKU_ID', 'SKU')} | {row.get('CEDI_Destino', 'CEDI')} | {clean_action(row.get('Accion_Recomendada', ''))} | Beneficio {money(row.get('Beneficio_Neto', 0))}"
        option_labels.append((idx, label))

    selected_idx = st.selectbox(
        "Selecciona una alerta para explicarla como caso de negocio",
        options=[idx for idx, _ in option_labels],
        format_func=lambda idx: dict(option_labels).get(idx, str(idx)),
    )
    selected = visible_decisions.loc[selected_idx]

    a, b, c = st.columns([1, 1, 1])
    with a:
        metric_card("SKU / CEDI", f"{selected.get('SKU_ID', 'N/D')}", f"Destino: {selected.get('CEDI_Destino', 'N/D')}", "info")
    with b:
        metric_card("Riesgo estimado", pct(selected.get("Riesgo_Probabilidad", 0)), "Probabilidad de quiebre calculada por el modelo.", "danger")
    with c:
        metric_card("Acción", clean_action(selected.get("Accion_Recomendada", "N/D")), "Recomendación generada por el motor costo-beneficio.", "success" if selected.get("Accion_Recomendada") == "TRANSFER_INVENTORY" else "warning")

    l, r = st.columns([1, 1])
    with l:
        with st.container(border=True):
            st.markdown("#### 🧮 Proyección de cobertura")
            explanation(
                "Qué representa",
                "Compara el inventario actual contra la demanda estimada durante el lead time. Si la demanda supera el stock, aparece el riesgo de quiebre.",
                "danger",
            )
            projection_df = pd.DataFrame(
                {
                    "Concepto": ["Stock actual", "Demanda estimada durante lead time", "Unidades necesarias"],
                    "Unidades": [
                        float(selected.get("Stock_Actual", 0) or 0),
                        float(selected.get("Demanda_Estimada_LT_Segura", 0) or 0),
                        float(selected.get("Unidades_Necesarias", 0) or 0),
                    ],
                }
            )
            fig = px.bar(
                projection_df,
                x="Concepto",
                y="Unidades",
                color="Concepto",
                color_discrete_sequence=["#3b6f9e", "#b45309", "#b91c1c"],
                text="Unidades",
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_layout(template=plot_template, height=390, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
            analogy("El lead time es como el tiempo que tarda en llegar tu pedido del súper. Si tu comida no alcanza hasta que llegue el pedido, necesitas conseguir comida de otra fuente.")

    with r:
        with st.container(border=True):
            st.markdown("#### 💰 Comparación económica")
            explanation(
                "Qué representa",
                "Compara cuánto se perdería si no actuamos contra cuánto cuesta transferir inventario. Si el beneficio neto es positivo, actuar tiene sentido económico.",
                "success",
            )
            money_df = pd.DataFrame(
                {
                    "Concepto": ["Pérdida sin actuar", "Costo transferencia", "Beneficio neto"],
                    "MXN": [
                        float(selected.get("Perdida_Esperada_Sin_Actuar", 0) or 0),
                        float(selected.get("Costo_Transferencia", 0) or 0),
                        float(selected.get("Beneficio_Neto", 0) or 0),
                    ],
                }
            )
            fig = px.bar(
                money_df,
                x="Concepto",
                y="MXN",
                color="Concepto",
                color_discrete_sequence=["#b91c1c", "#b45309", "#15803d"],
                text="MXN",
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig.update_layout(template=plot_template, height=390, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### 🧾 Explicación ejecutiva de la alerta seleccionada")
        explanation_text = selected.get("Explicacion_Ejecutiva", "No hay explicación disponible para esta alerta.")
        st.markdown(f"> {explanation_text}")
        st.caption(f"Razón técnica: {selected.get('Razon', 'N/D')}")

    with st.expander("Ver tabla completa de recomendaciones priorizadas", expanded=False):
        display_cols = [
            "Fecha",
            "SKU_ID",
            "CEDI_Destino",
            "Riesgo_Probabilidad",
            "Nivel_Riesgo",
            "Accion_Legible",
            "CEDI_Origen_Recomendado",
            "Unidades_A_Transferir",
            "Perdida_Esperada_Sin_Actuar",
            "Costo_Transferencia",
            "Beneficio_Neto",
            "Razon",
        ]
        available_cols = [c for c in display_cols if c in visible_decisions.columns]
        table = visible_decisions[available_cols].copy()
        if "Riesgo_Probabilidad" in table.columns:
            table["Riesgo_Probabilidad"] = table["Riesgo_Probabilidad"].apply(lambda x: pct(x))
        for col in ["Perdida_Esperada_Sin_Actuar", "Costo_Transferencia", "Beneficio_Neto"]:
            if col in table.columns:
                table[col] = table[col].apply(money)
        st.dataframe(table, use_container_width=True, hide_index=True)


# ============================================================
# 12. MODELO ML Y AGENTES A2A-LITE
# ============================================================

section(
    "4) Modelo ML + Agentes: ¿cómo se construye confianza?",
    "La confianza no viene de decir 'la IA lo dijo'. Viene de separar responsabilidades: el modelo predice, el motor calcula y los agentes explican.",
)

m1, m2 = st.columns([1, 1])

with m1:
    with st.container(border=True):
        st.markdown("#### 🤖 Comparación de modelos")
        explanation(
            "Por qué importa",
            "No elegimos un modelo por moda. Comparamos alternativas y priorizamos detectar quiebres reales sin disparar demasiadas falsas alarmas.",
            "info",
        )
        if not model_metrics.empty:
            metric_to_show = st.selectbox(
                "Métrica para comparar",
                options=[c for c in ["f1_risk", "recall_risk", "precision_risk", "roc_auc", "accuracy"] if c in model_metrics.columns],
                index=0,
            )
            mm = model_metrics.sort_values(metric_to_show, ascending=True)
            fig = px.bar(
                mm,
                y="model",
                x=metric_to_show,
                orientation="h",
                color=metric_to_show,
                color_continuous_scale=[[0, "#6b8faf"], [1, "#15803d"]],
                labels={"model": "Modelo", metric_to_show: "Score"},
            )
            fig.update_layout(template=plot_template, height=390, margin=dict(l=10, r=10, t=20, b=10), xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No encontré métricas de modelos.")

with m2:
    with st.container(border=True):
        st.markdown("#### 🧠 Agentes A2A-lite")
        explanation(
            "Qué significa A2A-lite",
            "Cada agente recibe un artefacto estructurado y produce otro. Es como una cadena de especialistas: riesgo → costo → política → comunicación ejecutiva.",
            "info",
        )
        if isinstance(agent_registry, dict):
            registry_data = agent_registry.get("agents", [])
        elif isinstance(agent_registry, list):
            registry_data = agent_registry
        else:
            registry_data = []
        if registry_data:
            registry_df = pd.DataFrame(registry_data)
            st.dataframe(registry_df, use_container_width=True, hide_index=True)
        else:
            st.info("No encontré registro de agentes.")

if agent_brief:
    with st.expander("📄 Ver brief ejecutivo generado por el agente", expanded=False):
        st.markdown(agent_brief)

with st.expander("🔍 Ver artefactos estructurados del agente", expanded=False):
    if agent_artifacts:
        st.json(agent_artifacts)
    else:
        st.info("No encontré artefactos del agente.")


# ============================================================
# 13. SIMULADOR INTERACTIVO
# ============================================================

section(
    "5) Simulador: ¿qué pasa si cambio la decisión?",
    "Este módulo ayuda a explicar la lógica costo-beneficio sin fórmulas complicadas. Permite jugar con unidades, costo y pérdida para entender cuándo conviene actuar.",
)

sim_col1, sim_col2 = st.columns([0.9, 1.1])

with sim_col1:
    with st.container(border=True):
        st.markdown("#### 🎚️ Parámetros")
        sim_risk = st.slider("Probabilidad de quiebre", 0, 100, 80, 5) / 100
        sim_daily_loss = st.number_input("Costo diario de quiebre (MXN)", min_value=0, value=8000, step=500)
        sim_lead_time = st.slider("Lead time (días)", 1, 10, 5, 1)
        sim_units = st.number_input("Unidades a transferir", min_value=0, value=500, step=50)
        sim_cost_unit = st.number_input("Costo de transferencia por unidad", min_value=0.0, value=12.0, step=1.0)

        sim_expected_loss = sim_risk * sim_daily_loss * sim_lead_time
        sim_transfer_cost = sim_units * sim_cost_unit
        sim_net = sim_expected_loss - sim_transfer_cost

        metric_card("Resultado simulado", money(sim_net), "Beneficio neto = pérdida esperada evitada - costo de transferencia.", "success" if sim_net > 0 else "danger")

with sim_col2:
    with st.container(border=True):
        st.markdown("#### 📊 Lectura del simulador")
        sim_df = pd.DataFrame(
            {
                "Concepto": ["Pérdida esperada", "Costo transferencia", "Beneficio neto"],
                "MXN": [sim_expected_loss, sim_transfer_cost, sim_net],
            }
        )
        fig = px.bar(
            sim_df,
            x="Concepto",
            y="MXN",
            color="Concepto",
            color_discrete_sequence=["#b91c1c", "#b45309", "#15803d" if sim_net >= 0 else "#b91c1c"],
            text="MXN",
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(template=plot_template, height=390, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if sim_net > 0:
            explanation("Interpretación", "La acción tiene sentido económico porque la pérdida evitada es mayor que el costo logístico.", "success")
        else:
            explanation("Interpretación", "No conviene mover inventario bajo estos supuestos. Sería mejor esperar reabasto o revisar otra fuente.", "danger")


# ============================================================
# 14. ARQUITECTURA
# ============================================================

section(
    "6) Arquitectura: local-first hoy, cloud-ready mañana",
    "El prototipo evita costos de nube durante la prueba, pero está diseñado para escalar a GCP con BigQuery, Vertex AI y servicios de agentes.",
)

arch1, arch2 = st.columns([1, 1])
with arch1:
    with st.container(border=True):
        st.markdown("#### 🧪 MVP local-first")
        st.markdown(
            """
            ```text
            Excel histórico
                ↓
            pandas / DuckDB
                ↓
            XGBoost
                ↓
            Decision Engine
                ↓
            Agentes A2A-lite
                ↓
            Streamlit Dashboard
            ```
            """
        )
        explanation("Ventaja", "Corre barato, rápido y con baja dependencia de infraestructura externa.", "success")

with arch2:
    with st.container(border=True):
        st.markdown("#### ☁️ Ruta cloud-ready en GCP")
        st.markdown(
            """
            ```text
            ERP / WMS / POS
                ↓
            Cloud Storage / Pub/Sub / Dataflow
                ↓
            BigQuery
                ↓
            Vertex AI Training + Registry
                ↓
            Vertex AI Endpoint / Batch Prediction
                ↓
            Agentes A2A / Agent Platform
                ↓
            Looker / Streamlit / Cloud Run
            ```
            """
        )
        explanation("Ventaja", "Permite MLOps, monitoreo, versionado de modelos y escalamiento empresarial.", "info")


# ============================================================
# 15. CHAT EXPLICATIVO
# ============================================================

section(
    "7) Chat explicativo: pregúntale al agente de negocio",
    "El chat está diseñado para dar respuestas detalladas usando el contexto real del dashboard. Puede funcionar localmente o con Gemini si configuras la API key.",
)


def build_context_summary() -> str:
    """Construye un resumen compacto del estado actual para el chat."""
    top = filtered.sort_values("Beneficio_Neto", ascending=False).head(3) if "Beneficio_Neto" in filtered.columns else filtered.head(3)
    top_lines = []
    for _, row in top.iterrows():
        top_lines.append(
            f"- {row.get('SKU_ID','N/D')} en {row.get('CEDI_Destino','N/D')}: "
            f"riesgo {pct(row.get('Riesgo_Probabilidad',0))}, "
            f"acción {clean_action(row.get('Accion_Recomendada','N/D'))}, "
            f"beneficio {money(row.get('Beneficio_Neto',0))}."
        )
    return "\n".join(top_lines)



def local_chat_answer(question: str) -> str:
    """Responde con reglas locales y datos cargados.

    Esta función no pretende reemplazar un LLM. Sirve para demo estable sin API.
    Usa patrones de pregunta frecuentes y responde con contexto real.
    """
    q = question.lower().strip()
    top_context = build_context_summary()

    top_alert = None
    if not filtered.empty:
        order_col = "Beneficio_Neto" if "Beneficio_Neto" in filtered.columns else "Riesgo_Probabilidad"
        top_alert = filtered.sort_values(order_col, ascending=False).iloc[0]

    if any(word in q for word in ["más importante", "prioridad", "principal", "primero"]):
        if top_alert is None:
            return "No tengo alertas con los filtros actuales. Reduce los filtros para ver prioridades."
        return (
            f"La alerta prioritaria es **{top_alert.get('SKU_ID','N/D')} en {top_alert.get('CEDI_Destino','N/D')}**.\n\n"
            f"La priorizo porque combina: riesgo de quiebre de **{pct(top_alert.get('Riesgo_Probabilidad',0))}**, "
            f"pérdida esperada de **{money(top_alert.get('Perdida_Esperada_Sin_Actuar',0))}** y beneficio neto estimado de "
            f"**{money(top_alert.get('Beneficio_Neto',0))}**.\n\n"
            f"Acción recomendada: **{clean_action(top_alert.get('Accion_Recomendada','N/D'))}**.\n\n"
            "En lenguaje de negocio: esta es la alerta donde actuar puede proteger más venta o nivel de servicio con el mejor retorno operativo."
        )

    if any(word in q for word in ["sin producto", "quedar sin", "próxima semana", "stockout", "quiebre"]):
        return (
            "Para saber si puedes quedarte sin producto, mira la sección **Proyección de cobertura**.\n\n"
            "La lógica es sencilla: comparamos el **stock actual** contra la **demanda esperada durante el lead time**. "
            "Si la demanda esperada es mayor que el stock, el producto puede agotarse antes de que llegue el reabasto.\n\n"
            "En el dashboard, el color rojo indica que el riesgo es alto. Es parecido a revisar si la comida del refrigerador alcanza hasta la próxima ida al súper."
        )

    if any(word in q for word in ["rojo", "color", "riesgo alto", "índice"]):
        return (
            "El color rojo significa **alerta operativa**. No quiere decir que el sistema falló; significa que esa combinación SKU/CEDI requiere atención.\n\n"
            "En este proyecto, el rojo suele aparecer cuando el modelo estima alta probabilidad de quiebre o cuando la pérdida esperada es elevada. "
            "El objetivo del color es reducir carga cognitiva: el usuario no necesita leer toda la tabla; primero mira dónde está el riesgo."
        )

    if any(word in q for word in ["no siempre", "por qué no", "mover", "transferir"]):
        return (
            "No siempre conviene mover inventario porque transferir también cuesta.\n\n"
            "El motor compara dos cosas:\n"
            "1. **Pérdida esperada si no actúas**.\n"
            "2. **Costo de transferir inventario**.\n\n"
            "Si el costo logístico es mayor que la pérdida evitada, mover inventario destruye valor. Además, el sistema revisa que el CEDI origen no quede desprotegido."
        )

    if any(word in q for word in ["costo", "almacenamiento", "reducir", "ahorro"]):
        return (
            "Para reducir costo, usa los filtros de CEDI, SKU y acción recomendada. Busca casos donde el **beneficio neto** sea positivo y alto.\n\n"
            "La idea no es mover todo, sino mover donde el retorno sea claro. En términos ejecutivos: prioriza acciones con alta pérdida evitada, bajo costo de transferencia y sin afectar al CEDI origen.\n\n"
            f"Top alertas actuales:\n{top_context}"
        )

    if any(word in q for word in ["arquitectura", "gcp", "cloud", "técnico", "tecnico"]):
        return (
            "La arquitectura está separada por responsabilidades:\n\n"
            "- **XGBoost** predice riesgo de quiebre.\n"
            "- **Decision Engine** calcula costo-beneficio con reglas deterministas.\n"
            "- **Agentes A2A-lite** interpretan, critican y explican.\n"
            "- **Streamlit** traduce todo a una interfaz ejecutiva.\n\n"
            "Para GCP, la ruta natural es BigQuery para datos, Vertex AI para entrenamiento/despliegue, Model Registry para versionado, Model Monitoring para drift y A2A/Agent Platform para agentes interoperables."
        )

    if any(word in q for word in ["modelo", "xgboost", "ml", "machine learning"]):
        best_model = "N/D"
        if not model_metrics.empty and "f1_risk" in model_metrics.columns:
            best_model = model_metrics.sort_values("f1_risk", ascending=False).iloc[0].get("model", "N/D")
        return (
            f"El modelo se usa para estimar el riesgo de quiebre por SKU/CEDI. En las métricas actuales, el mejor desempeño por F1 de riesgo aparece como **{best_model}**.\n\n"
            "Aunque XGBoost es el modelo elegido para defensa técnica, lo importante es que el pipeline compara modelos y no decide por intuición. "
            "La métrica clave no es solo accuracy; en inventario importa mucho el **recall de riesgo**, porque dejar pasar quiebres puede causar venta perdida."
        )

    return (
        "Puedo ayudarte a interpretar el dashboard desde tres ángulos:\n\n"
        "1. **Negocio:** qué producto está en riesgo, cuánto se perdería y qué acción conviene.\n"
        "2. **Técnico:** cómo se conectan XGBoost, decision engine y agentes.\n"
        "3. **Operativo:** qué CEDI/SKU priorizar hoy.\n\n"
        f"Resumen de alertas actuales:\n{top_context}"
    )



def get_gemini_answer(question: str) -> Optional[str]:
    """Respuesta opcional con Gemini vía LangChain.

    Si no hay API key o falta la librería, regresa None y se usa fallback local.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None
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
        context = build_context_summary()
        system_prompt = f"""
Eres ExecSupplyAI, un agente ejecutivo de Supply Chain para Grupo Herdez.

Tu misión:
Explicar de forma clara y útil el dashboard Herdez Smart-Supply a usuarios sin conocimiento técnico.

Límites:
- No inventes datos.
- Usa solo el contexto proporcionado.
- Si no hay información suficiente, dilo.
- Explica con analogías simples.
- Responde en español profesional y claro.

Contexto de datos filtrados:
Alertas visibles: {len(filtered)}
Riesgo promedio: {pct(avg_risk)}
Pérdida esperada agregada: {money(expected_loss)}
Beneficio neto agregado: {money(net_benefit)}
Top alertas:
{context}
"""
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
        return response.content
    except Exception:
        return None


with st.container(border=True):
    use_gemini = st.toggle("Usar Gemini para respuestas extendidas si hay API key", value=False)
    st.caption("Si Gemini no está disponible, el chat usa un modo local basado en reglas y datos del dashboard.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hola, soy ExecSupplyAI. Puedo explicarte las alertas, los costos, el modelo y la arquitectura. Prueba una pregunta del Centro de Control Guiado o escribe la tuya.",
            }
        ]

    pending = st.session_state.pop("pending_question", None)
    if pending:
        st.session_state.messages.append({"role": "user", "content": pending})
        answer = get_gemini_answer(pending) if use_gemini else None
        if not answer:
            answer = local_chat_answer(pending)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Pregunta algo como: ¿cuál es la alerta más importante y por qué?")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Analizando contexto del dashboard..."):
                answer = get_gemini_answer(user_question) if use_gemini else None
                if not answer:
                    answer = local_chat_answer(user_question)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


# ============================================================
# 16. FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer-note">
        Herdez Smart-Supply · Prototipo local-first · Artefactos leídos desde: <code>{OUTPUTS_DIR}</code><br>
        Pipeline: EDA → XGBoost → Decision Engine → Agentes A2A-lite → Dashboard Ejecutivo
    </div>
    """,
    unsafe_allow_html=True,
)
