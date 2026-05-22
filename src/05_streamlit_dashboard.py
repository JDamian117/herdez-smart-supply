"""
05_streamlit_dashboard.py
Herdez Smart-Supply | Dashboard operacional con predicción en vivo y chat

Objetivo del archivo
--------------------
Convertir el prototipo en un producto demostrable:
1) Mostrar el estado ejecutivo del sistema.
2) Permitir seleccionar un caso real del dataset.
3) Ejecutar predicción en vivo con el modelo XGBoost guardado.
4) Calcular una recomendación costo-beneficio con reglas deterministas.
5) Ofrecer un chat explicativo para usuarios de negocio.

Principio de arquitectura
-------------------------
- El modelo ML predice riesgo de quiebre.
- El motor determinista calcula la recomendación y el impacto económico.
- El chat/agente explica la decisión, pero no inventa números críticos.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =============================================================================
# 1. CONFIGURACIÓN GENERAL
# =============================================================================

st.set_page_config(
    page_title="Herdez Smart-Supply",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 2. RUTAS ROBUSTAS
# =============================================================================

def find_project_root() -> Path:
    """Busca la raíz del proyecto aunque la app corra desde app.py o desde src/."""
    current = Path(__file__).resolve()
    candidates = [
        current.parent,
        current.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidates:
        if (candidate / "outputs").exists() and (candidate / "src").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = find_project_root()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# 3. CSS / SISTEMA VISUAL PREMIUM
# =============================================================================

def inject_css() -> None:
    """Inyecta estilos para cards, contraste, sombras y accesibilidad visual."""
    st.markdown(
        """
        <style>
        :root {
            --card-bg: rgba(255, 255, 255, 0.78);
            --card-border: rgba(20, 36, 58, 0.10);
            --text-muted: #64748b;
            --accent-blue: #2563eb;
            --accent-green: #16a34a;
            --accent-red: #dc2626;
            --accent-amber: #d97706;
            --charcoal: #111827;
            --soft-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --card-bg: rgba(17, 24, 39, 0.72);
                --card-border: rgba(255, 255, 255, 0.12);
                --text-muted: #cbd5e1;
                --charcoal: #f8fafc;
                --soft-shadow: 0 14px 35px rgba(0, 0, 0, 0.28);
            }
        }

        .main .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }

        .hero {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(22, 163, 74, 0.10));
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            box-shadow: var(--soft-shadow);
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.2rem;
            color: var(--charcoal);
        }

        .hero-subtitle {
            font-size: 1rem;
            color: var(--text-muted);
            line-height: 1.55;
            max-width: 1050px;
        }

        .section-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 1.05rem 1.15rem;
            box-shadow: var(--soft-shadow);
            margin-bottom: 1rem;
        }

        .mini-title {
            font-size: 1.05rem;
            font-weight: 750;
            margin-bottom: 0.25rem;
            color: var(--charcoal);
        }

        .helper-text {
            font-size: 0.92rem;
            color: var(--text-muted);
            line-height: 1.5;
        }

        .pill {
            display: inline-block;
            padding: 0.22rem 0.62rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 0.35rem;
        }
        .pill-blue { background: rgba(37,99,235,0.14); color: #2563eb; }
        .pill-green { background: rgba(22,163,74,0.14); color: #16a34a; }
        .pill-red { background: rgba(220,38,38,0.14); color: #dc2626; }
        .pill-amber { background: rgba(217,119,6,0.14); color: #d97706; }

        .decision-box {
            border-left: 5px solid #2563eb;
            padding: 0.95rem 1rem;
            border-radius: 12px;
            background: rgba(37, 99, 235, 0.08);
            margin-top: 0.6rem;
        }

        .risk-high { border-left-color: #dc2626; background: rgba(220, 38, 38, 0.08); }
        .risk-medium { border-left-color: #d97706; background: rgba(217, 119, 6, 0.10); }
        .risk-low { border-left-color: #16a34a; background: rgba(22, 163, 74, 0.08); }

        div[data-testid="stMetric"] {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: var(--soft-shadow);
        }

        .stChatMessage {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =============================================================================
# 4. FUNCIONES AUXILIARES DE DATOS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_json(filename: str) -> Any:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return {} if filename.endswith(".json") else None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_markdown(filename: str) -> str:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


@st.cache_resource(show_spinner=False)
def load_model_artifact() -> Optional[Dict[str, Any]]:
    path = MODELS_DIR / "best_stockout_model.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        st.warning(f"No pude cargar el modelo guardado: {exc}")
        return None


def money(value: Any) -> str:
    try:
        return f"${float(value):,.0f} MXN"
    except Exception:
        return "$0 MXN"


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "0.0%"


def safe_col(df: pd.DataFrame, col: str, default: Any = 0) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


# =============================================================================
# 5. LÓGICA DE MODELO Y DECISIÓN EN VIVO
# =============================================================================

class DecisionConfig:
    """Parámetros de negocio editables sin reentrenar el modelo."""
    safety_factor = 1.10
    min_source_coverage_ratio = 1.20
    high_risk_threshold = 0.70
    medium_risk_threshold = 0.40
    min_net_benefit = 0.0


def classify_risk(prob: float) -> str:
    if prob >= DecisionConfig.high_risk_threshold:
        return "Alto"
    if prob >= DecisionConfig.medium_risk_threshold:
        return "Medio"
    return "Bajo"


def normalize_live_row(row: pd.Series) -> pd.Series:
    """Recalcula features dependientes cuando el usuario modifica variables."""
    row = row.copy()
    ventas = float(row.get("Ventas_Unidades", 0) or 0)
    ventas_media_7d = float(row.get("Ventas_Media_7d", ventas) or ventas or 1)
    stock = float(row.get("Stock_Actual", 0) or 0)
    lead = float(row.get("Lead_Time_Dias", 1) or 1)

    # Si el usuario modifica ventas/lead/stock, estas variables deben actualizarse.
    row["Ventas_Media_3d"] = float(row.get("Ventas_Media_3d", ventas_media_7d) or ventas_media_7d)
    row["Ventas_Media_7d"] = ventas_media_7d
    row["Ventas_Max_7d"] = max(float(row.get("Ventas_Max_7d", ventas_media_7d) or ventas_media_7d), ventas_media_7d)
    row["Demanda_Estimada_LT"] = ventas_media_7d * lead
    row["Dias_Cobertura"] = stock / max(ventas_media_7d, 1)
    row["Gap_Estimado_LT"] = stock - row["Demanda_Estimada_LT"]
    row["Ratio_Cobertura_LT"] = stock / max(row["Demanda_Estimada_LT"], 1)
    row["Promocion_Activa_Num"] = int(row.get("Promocion_Activa_Num", row.get("Promocion_Activa", 0)) or 0)
    return row


def predict_live_risk(row: pd.Series, model_artifact: Optional[Dict[str, Any]]) -> Tuple[float, str]:
    """Predice riesgo con XGBoost. Si no hay modelo, usa baseline de cobertura."""
    row = normalize_live_row(row)

    if model_artifact is not None:
        try:
            feature_columns = model_artifact["feature_columns"]
            pipeline = model_artifact["pipeline"]
            one = pd.DataFrame([row.to_dict()])
            for col in feature_columns:
                if col not in one.columns:
                    one[col] = np.nan
            prob = float(pipeline.predict_proba(one[feature_columns])[:, 1][0])
            return prob, "XGBoost guardado"
        except Exception as exc:
            st.warning(f"No se pudo usar XGBoost para este caso. Uso baseline. Detalle: {exc}")

    # Baseline si no hay modelo: cuanto menor la cobertura, mayor el riesgo.
    ratio = float(row.get("Ratio_Cobertura_LT", 0) or 0)
    prob = float(np.clip(1.0 - ratio, 0.0, 1.0))
    return prob, "Baseline cobertura"


def source_surplus(source_row: pd.Series) -> int:
    ventas = float(source_row.get("Ventas_Media_7d", source_row.get("Ventas_Unidades", 0)) or 0)
    lead = float(source_row.get("Lead_Time_Dias", 1) or 1)
    stock = float(source_row.get("Stock_Actual", 0) or 0)
    required = ventas * lead * DecisionConfig.safety_factor * DecisionConfig.min_source_coverage_ratio
    return int(max(0, np.floor(stock - required)))


def recommend_live_case(scored_df: pd.DataFrame, live_row: pd.Series, risk_prob: float) -> Dict[str, Any]:
    """Calcula recomendación determinista para el caso seleccionado/simulado."""
    row = normalize_live_row(live_row)
    row["Riesgo_Probabilidad"] = risk_prob
    risk_level = classify_risk(risk_prob)

    demand_safe = float(row["Ventas_Media_7d"] * row["Lead_Time_Dias"] * DecisionConfig.safety_factor)
    units_needed = int(np.ceil(max(0, demand_safe - float(row["Stock_Actual"]))))
    expected_loss = float(risk_prob * float(row["Costo_Quiebre_Stock_Diario"]) * float(row["Lead_Time_Dias"]))

    candidates = scored_df[
        (pd.to_datetime(scored_df["Fecha"]).dt.date == pd.to_datetime(row["Fecha"]).date())
        & (scored_df["SKU_ID"] == row["SKU_ID"])
        & (scored_df["CEDI"] != row["CEDI"])
    ].copy()

    scenarios: List[Dict[str, Any]] = []
    for _, source in candidates.iterrows():
        surplus = source_surplus(source)
        units = int(min(units_needed, surplus))
        if units <= 0:
            continue
        coverage_ratio = min(units / max(units_needed, 1), 1.0)
        avoided_loss = expected_loss * coverage_ratio
        transfer_cost = float(units * float(row["Costo_Transferencia_Unidad"]))
        net = avoided_loss - transfer_cost
        scenarios.append(
            {
                "CEDI_Origen": source["CEDI"],
                "Stock_Origen": float(source["Stock_Actual"]),
                "Excedente_Origen": surplus,
                "Unidades_A_Transferir": units,
                "Cobertura_Faltante": coverage_ratio,
                "Costo_Transferencia": transfer_cost,
                "Perdida_Evitada": avoided_loss,
                "Beneficio_Neto": net,
            }
        )

    best = max(scenarios, key=lambda s: (s["Beneficio_Neto"], s["Perdida_Evitada"]), default=None)

    if units_needed <= 0:
        action = "MONITOR"
        reason = "El inventario actual cubre la demanda estimada durante el lead time."
    elif risk_prob < DecisionConfig.medium_risk_threshold:
        action = "MONITOR"
        reason = "El riesgo estimado aún no justifica una transferencia inmediata."
    elif best and best["Beneficio_Neto"] > DecisionConfig.min_net_benefit:
        action = "TRANSFER_INVENTORY"
        reason = "Existe un CEDI origen viable y la pérdida evitada supera el costo logístico."
    elif best:
        action = "WAIT_REPLENISHMENT"
        reason = "Hay inventario transferible, pero el beneficio neto no justifica el costo logístico."
    else:
        action = "EXPEDITE_REPLENISHMENT_OR_REVIEW"
        reason = "No se encontró un CEDI origen con excedente suficiente sin ponerlo en riesgo."

    if best:
        transfer_cost = best["Costo_Transferencia"]
        avoided_loss = best["Perdida_Evitada"]
        net_benefit = best["Beneficio_Neto"]
        origin = best["CEDI_Origen"]
        units = best["Unidades_A_Transferir"]
    else:
        transfer_cost = 0.0
        avoided_loss = 0.0
        net_benefit = 0.0
        origin = "No disponible"
        units = 0

    explanation = (
        f"El SKU {row['SKU_ID']} en {row['CEDI']} tiene riesgo {risk_level.lower()} "
        f"de quiebre ({pct(risk_prob)}). La demanda segura estimada durante el lead time es "
        f"de {demand_safe:,.0f} unidades y se requieren {units_needed:,} unidades adicionales. "
        f"Recomendación: {action}. {reason}"
    )

    return {
        "Fecha": str(pd.to_datetime(row["Fecha"]).date()),
        "SKU_ID": row["SKU_ID"],
        "CEDI_Destino": row["CEDI"],
        "Riesgo_Probabilidad": risk_prob,
        "Nivel_Riesgo": risk_level,
        "Stock_Actual": float(row["Stock_Actual"]),
        "Ventas_Media_7d": float(row["Ventas_Media_7d"]),
        "Lead_Time_Dias": int(row["Lead_Time_Dias"]),
        "Demanda_Estimada_LT_Segura": demand_safe,
        "Unidades_Necesarias": units_needed,
        "Perdida_Esperada_Sin_Actuar": expected_loss,
        "Accion_Recomendada": action,
        "CEDI_Origen_Recomendado": origin,
        "Unidades_A_Transferir": units,
        "Costo_Transferencia": transfer_cost,
        "Perdida_Evitada": avoided_loss,
        "Beneficio_Neto": net_benefit,
        "Razon": reason,
        "Explicacion_Ejecutiva": explanation,
        "Escenarios": scenarios,
    }


@st.cache_data(show_spinner=False)
def score_dataset_cached(features_df: pd.DataFrame) -> pd.DataFrame:
    """Genera riesgos para todo el dataset si no existe scored_stockout_alerts.csv."""
    artifact = load_model_artifact()
    if artifact is None or features_df.empty:
        return features_df.copy()
    scored = features_df.copy()
    feature_columns = artifact["feature_columns"]
    pipeline = artifact["pipeline"]
    for col in feature_columns:
        if col not in scored.columns:
            scored[col] = np.nan
    scored["Riesgo_Probabilidad"] = pipeline.predict_proba(scored[feature_columns])[:, 1]
    scored["Alerta_Riesgo"] = (scored["Riesgo_Probabilidad"] >= float(artifact.get("threshold", 0.5))).astype(int)
    return scored


# =============================================================================
# 6. CHAT / AGENTE EXPLICADOR
# =============================================================================

def get_secret(name: str) -> Optional[str]:
    """Lee secretos desde Streamlit Cloud o variables de entorno."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


def build_context_summary(
    recommendations: pd.DataFrame,
    model_metrics: pd.DataFrame,
    live_decision: Optional[Dict[str, Any]],
    agent_brief: str,
) -> str:
    """Resume el contexto real disponible para el chat."""
    lines = []
    if not recommendations.empty:
        total_loss = recommendations["Perdida_Esperada_Sin_Actuar"].sum()
        total_net = recommendations["Beneficio_Neto"].sum()
        top = recommendations.sort_values("Beneficio_Neto", ascending=False).head(3)
        lines.append(f"Alertas priorizadas: {len(recommendations)}")
        lines.append(f"Pérdida esperada agregada: {money(total_loss)}")
        lines.append(f"Beneficio neto potencial agregado: {money(total_net)}")
        lines.append("Top recomendaciones:")
        for _, r in top.iterrows():
            lines.append(
                f"- {r['SKU_ID']} en {r['CEDI_Destino']}: {r['Accion_Recomendada']}, "
                f"beneficio {money(r['Beneficio_Neto'])}"
            )
    if not model_metrics.empty:
        best = model_metrics.sort_values("f1_risk", ascending=False).head(1).iloc[0]
        lines.append(
            f"Modelo con mejor F1 riesgo en comparación: {best.get('model', 'N/D')} "
            f"con F1={best.get('f1_risk', 0):.3f}, recall={best.get('recall_risk', 0):.3f}."
        )
    if live_decision:
        lines.append("Caso vivo seleccionado:")
        lines.append(json.dumps({k: v for k, v in live_decision.items() if k != "Escenarios"}, ensure_ascii=False, default=str))
    if agent_brief:
        lines.append("Brief A2A-lite disponible: sí.")
    return "\n".join(lines)


def local_agent_answer(question: str, context: str, live_decision: Optional[Dict[str, Any]]) -> str:
    """Fallback local para responder sin API key. No inventa datos; usa reglas simples."""
    q = question.lower()

    if any(word in q for word in ["quiebre", "stock", "riesgo", "quedar sin producto"]):
        if live_decision:
            return (
                f"Para el caso seleccionado, el riesgo estimado es **{pct(live_decision['Riesgo_Probabilidad'])}** "
                f"y el nivel es **{live_decision['Nivel_Riesgo']}**. En lenguaje de negocio: el sistema está evaluando "
                f"si el inventario actual alcanza hasta que llegue el reabasto. La demanda segura estimada es "
                f"**{live_decision['Demanda_Estimada_LT_Segura']:,.0f} unidades** y se requieren "
                f"**{live_decision['Unidades_Necesarias']:,} unidades** adicionales."
            )
        return "El riesgo de quiebre indica la probabilidad de que un SKU no tenga inventario suficiente durante el lead time. Es como revisar si la comida en el refrigerador alcanza hasta la próxima compra."

    if any(word in q for word in ["mover", "transfer", "transferir", "cedi", "origen"]):
        if live_decision:
            return (
                f"La recomendación para el caso seleccionado es **{live_decision['Accion_Recomendada']}**. "
                f"Origen recomendado: **{live_decision['CEDI_Origen_Recomendado']}**. "
                f"Unidades a transferir: **{live_decision['Unidades_A_Transferir']:,}**. "
                f"La razón es: {live_decision['Razon']}"
            )
        return "La transferencia se recomienda solo si hay un CEDI origen con excedente suficiente y si la pérdida evitada supera el costo logístico."

    if any(word in q for word in ["costo", "beneficio", "dinero", "roi", "perdida", "pérdida"]):
        if live_decision:
            return (
                f"Para el caso seleccionado: pérdida esperada sin actuar = **{money(live_decision['Perdida_Esperada_Sin_Actuar'])}**, "
                f"costo de transferencia = **{money(live_decision['Costo_Transferencia'])}**, "
                f"pérdida evitada = **{money(live_decision['Perdida_Evitada'])}**, "
                f"beneficio neto = **{money(live_decision['Beneficio_Neto'])}**. "
                f"La lógica es simple: conviene actuar cuando evitar la pérdida cuesta menos que quedarse sin producto."
            )
        return "El motor compara pérdida esperada contra costo de transferencia. Si el beneficio neto es positivo, la acción tiene sentido financiero."

    if any(word in q for word in ["modelo", "xgboost", "ml", "machine", "por qué"]):
        return (
            "El modelo XGBoost se usa porque funciona muy bien con datos tabulares y puede capturar relaciones no lineales, "
            "por ejemplo: bajo stock + alto lead time + promoción activa = mayor riesgo. Pero el modelo no decide solo: "
            "su probabilidad alimenta un motor determinista de costo-beneficio."
        )

    if any(word in q for word in ["agente", "a2a", "chat", "llm", "gemini"]):
        return (
            "El agente funciona como una capa de explicación. El patrón A2A-lite separa responsabilidades: un agente interpreta riesgo, "
            "otro costos, otro políticas y otro comunica al usuario. Para evitar alucinaciones, los números vienen del modelo y del decision engine, no del LLM."
        )

    if any(word in q for word in ["gcp", "cloud", "vertex", "bigquery", "arquitectura"]):
        return (
            "La ruta cloud-ready sería: fuentes ERP/WMS/POS → BigQuery → Vertex AI para entrenamiento/registro/monitoreo → "
            "servicios Cloud Run para tools deterministas → agentes interoperables vía A2A → dashboard ejecutivo."
        )

    return (
        "Puedo ayudarte a interpretar el riesgo, la recomendación, los costos, el modelo, los agentes o la arquitectura. "
        "Ejemplo: '¿por qué conviene mover inventario en este caso?' o 'explícame el beneficio neto'."
    )


def gemini_agent_answer(question: str, context: str) -> Optional[str]:
    """Respuesta opcional con Gemini vía LangChain. Si falla, regresa None."""
    api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2,
        )
        prompt = f"""
# Tu identidad
Eres un copiloto ejecutivo de Supply Chain y Applied AI. Explicas decisiones de inventario de forma clara para usuarios no técnicos.

# Tu misión
Responder preguntas sobre el dashboard Herdez Smart-Supply usando únicamente el contexto disponible.

# Límites
- No inventes números.
- Si falta información, dilo claramente.
- Explica con lenguaje de negocio y, si ayuda, agrega una analogía sencilla.
- Mantén la respuesta en español.

# Contexto disponible
{context}

# Pregunta del usuario
{question}
"""
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))
    except Exception:
        return None


# =============================================================================
# 7. CARGA PRINCIPAL
# =============================================================================

features = load_csv("herdez_features_dataset.csv")
scored_alerts = load_csv("scored_stockout_alerts.csv")
recommendations = load_csv("decision_recommendations.csv")
model_metrics = load_csv("model_comparison_metrics.csv")
risk_by_sku = load_csv("risk_by_sku.csv")
risk_by_cedi = load_csv("risk_by_cedi.csv")
agent_brief = load_markdown("agent_executive_brief_a2a_simple.md")
agent_registry = load_json("agent_a2a_simple_registry.json")
model_artifact = load_model_artifact()

if scored_alerts.empty and not features.empty:
    scored_alerts = score_dataset_cached(features)


# =============================================================================
# 8. SIDEBAR / CENTRO DE CONTROL GUIADO
# =============================================================================

with st.sidebar:
    st.markdown("## 🎛️ Centro de Control")
    st.caption("Navega el producto como si fueras Director de Supply Chain o Gerente de IA.")

    page = st.radio(
        "Modo de trabajo",
        [
            "1) Estado ejecutivo",
            "2) Predicción en vivo",
            "3) Simulador operativo",
            "4) Chat con agente",
            "5) Arquitectura y trazabilidad",
        ],
        index=1,
    )

    st.divider()
    with st.expander("📘 ¿Cómo usar esta demo?", expanded=True):
        st.markdown(
            """
1. Revisa el **Estado ejecutivo** para ver el panorama.
2. Entra a **Predicción en vivo** y selecciona un SKU/CEDI.
3. Modifica stock, ventas o lead time para simular cambios.
4. Consulta el **chat** para explicar la recomendación.

Analogía: el sistema revisa si tienes suficiente producto en el “refrigerador” hasta que llegue la próxima compra. Si no alcanza, calcula si conviene pedir prestado producto a otro CEDI.
            """
        )

    with st.expander("💬 Preguntas que puedes probar"):
        st.markdown(
            """
- ¿Cómo sé si me voy a quedar sin producto?
- ¿Por qué conviene transferir inventario?
- ¿Qué significa beneficio neto?
- ¿Por qué XGBoost y no solo un LLM?
- ¿Cómo escalaría esto a GCP?
            """
        )


# =============================================================================
# 9. HEADER
# =============================================================================

st.markdown(
    """
    <div class="hero">
      <div class="hero-title">📦 Herdez Smart-Supply</div>
      <div class="hero-subtitle">
        Producto analítico local-first para anticipar quiebres de stock, calcular si conviene mover inventario
        y explicar la decisión con un agente conversacional. No es solo un reporte: permite probar casos en vivo.
      </div>
      <br/>
      <span class="pill pill-blue">ML Predictivo</span>
      <span class="pill pill-green">Costo-beneficio</span>
      <span class="pill pill-amber">A2A-lite</span>
      <span class="pill pill-red">Prevención de quiebres</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if features.empty or recommendations.empty:
    st.error("No encontré los archivos mínimos en outputs/. Sube outputs/ y models/ al repo o ejecuta 01→04 primero.")
    st.code(
        "python src/01_eda_target_features.py --input data/Data_Prueba_Tecnica_Herdez_IA.xlsx\n"
        "python src/02_train_models.py --processed outputs/herdez_features_dataset.csv\n"
        "python src/03_decision_engine.py --processed outputs/herdez_features_dataset.csv --model models/best_stockout_model.joblib\n"
        "python src/04_agent_system_a2a_simple.py --mode fallback --max-alerts 8",
        language="bash",
    )
    st.stop()


# =============================================================================
# 10. KPIs GLOBALES
# =============================================================================

total_alerts = len(recommendations)
high_risk = int((recommendations["Nivel_Riesgo"] == "Alto").sum()) if "Nivel_Riesgo" in recommendations else 0
expected_loss = float(recommendations["Perdida_Esperada_Sin_Actuar"].sum())
net_benefit = float(recommendations["Beneficio_Neto"].sum())
transfer_units = int(recommendations["Unidades_A_Transferir"].sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Alertas priorizadas", f"{total_alerts}")
k2.metric("Riesgo alto", f"{high_risk}")
k3.metric("Pérdida esperada", money(expected_loss))
k4.metric("Beneficio neto", money(net_benefit))
k5.metric("Unidades sugeridas", f"{transfer_units:,}")


# =============================================================================
# 11. PÁGINAS
# =============================================================================

if page.startswith("1"):
    st.markdown("### 1) Estado ejecutivo")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="mini-title">¿Qué estoy viendo?</div>
<div class="helper-text">
Esta sección resume dónde está el riesgo operativo. La prioridad no es solo detectar quiebres,
sino decidir si actuar genera valor económico. Rojo significa urgencia, verde significa beneficio neto positivo,
y azul representa tendencias o comparación.
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            recommendations.groupby("CEDI_Destino", as_index=False)["Riesgo_Probabilidad"].mean().sort_values("Riesgo_Probabilidad", ascending=False),
            x="CEDI_Destino",
            y="Riesgo_Probabilidad",
            title="Riesgo promedio por CEDI destino",
            color="Riesgo_Probabilidad",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis_tickformat=".0%", height=390, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Lectura: identifica qué centro de distribución concentra mayor probabilidad de quedarse sin producto.")

    with c2:
        fig = px.bar(
            recommendations.groupby("SKU_ID", as_index=False)["Beneficio_Neto"].sum().sort_values("Beneficio_Neto", ascending=False),
            x="SKU_ID",
            y="Beneficio_Neto",
            title="Beneficio neto potencial por SKU",
            color="Beneficio_Neto",
            color_continuous_scale="Greens",
        )
        fig.update_layout(height=390, coloraxis_showscale=False, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Lectura: prioriza productos donde actuar puede evitar más pérdida económica.")

    st.markdown("### Top recomendaciones")
    cols = [
        "Fecha", "SKU_ID", "CEDI_Destino", "Riesgo_Probabilidad", "Accion_Recomendada",
        "CEDI_Origen_Recomendado", "Unidades_A_Transferir", "Perdida_Esperada_Sin_Actuar", "Beneficio_Neto",
    ]
    st.dataframe(recommendations.sort_values("Beneficio_Neto", ascending=False)[cols], use_container_width=True, hide_index=True)


elif page.startswith("2") or page.startswith("3"):
    st.markdown("### 2) Predicción en vivo y simulación operativa")
    st.info("Aquí ya no solo vemos archivos precomputados: seleccionamos un caso real, modificamos variables y ejecutamos la predicción + recomendación en vivo.")

    # Filtros para elegir caso base
    f1, f2, f3 = st.columns(3)
    with f1:
        sku = st.selectbox("SKU", sorted(features["SKU_ID"].dropna().unique()))
    with f2:
        cedi = st.selectbox("CEDI", sorted(features["CEDI"].dropna().unique()))
    case_df = features[(features["SKU_ID"] == sku) & (features["CEDI"] == cedi)].sort_values("Fecha")
    with f3:
        fecha = st.selectbox("Fecha", case_df["Fecha"].dt.date.astype(str).tolist())

    base_row = case_df[case_df["Fecha"].dt.date.astype(str) == fecha].iloc[0]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("<div class='mini-title'>Variables editables del caso</div>", unsafe_allow_html=True)
    st.markdown("<div class='helper-text'>Modifica estas variables para responder preguntas tipo: ¿qué pasa si baja el stock?, ¿si sube la demanda?, ¿si el proveedor tarda más?</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        stock = st.number_input("Stock actual", min_value=0.0, value=float(base_row["Stock_Actual"]), step=10.0)
        ventas = st.number_input("Ventas del día", min_value=0.0, value=float(base_row["Ventas_Unidades"]), step=5.0)
    with e2:
        ventas7 = st.number_input("Ventas media 7 días", min_value=1.0, value=float(base_row["Ventas_Media_7d"]), step=5.0)
        lead = st.number_input("Lead time días", min_value=1, max_value=30, value=int(base_row["Lead_Time_Dias"]), step=1)
    with e3:
        promo = st.selectbox("Promoción activa", [0, 1], index=int(base_row.get("Promocion_Activa_Num", 0)))
        clima_options = sorted(features["Clima"].dropna().unique().tolist())
        clima = st.selectbox("Clima", clima_options, index=clima_options.index(base_row["Clima"]) if base_row["Clima"] in clima_options else 0)
    with e4:
        costo_quiebre = st.number_input("Costo quiebre diario", min_value=0.0, value=float(base_row["Costo_Quiebre_Stock_Diario"]), step=500.0)
        costo_transfer = st.number_input("Costo transferencia unidad", min_value=0.0, value=float(base_row["Costo_Transferencia_Unidad"]), step=1.0)

    live_row = base_row.copy()
    live_row["Stock_Actual"] = stock
    live_row["Ventas_Unidades"] = ventas
    live_row["Ventas_Media_7d"] = ventas7
    live_row["Lead_Time_Dias"] = lead
    live_row["Promocion_Activa"] = promo
    live_row["Promocion_Activa_Num"] = promo
    live_row["Clima"] = clima
    live_row["Costo_Quiebre_Stock_Diario"] = costo_quiebre
    live_row["Costo_Transferencia_Unidad"] = costo_transfer

    risk_prob, model_used = predict_live_risk(live_row, model_artifact)
    live_decision = recommend_live_case(scored_alerts if not scored_alerts.empty else features, live_row, risk_prob)
    st.session_state["live_decision"] = live_decision

    risk_class = "risk-high" if live_decision["Nivel_Riesgo"] == "Alto" else "risk-medium" if live_decision["Nivel_Riesgo"] == "Medio" else "risk-low"

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Riesgo estimado", pct(risk_prob), help=f"Calculado con {model_used}")
    r2.metric("Nivel", live_decision["Nivel_Riesgo"])
    r3.metric("Unidades faltantes", f"{live_decision['Unidades_Necesarias']:,}")
    r4.metric("Beneficio neto", money(live_decision["Beneficio_Neto"]))

    st.markdown(f"<div class='decision-box {risk_class}'><b>Recomendación:</b> {live_decision['Accion_Recomendada']}<br>{live_decision['Explicacion_Ejecutiva']}</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        mini = pd.DataFrame(
            {
                "Concepto": ["Stock actual", "Demanda segura LT", "Unidades faltantes"],
                "Unidades": [live_decision["Stock_Actual"], live_decision["Demanda_Estimada_LT_Segura"], live_decision["Unidades_Necesarias"]],
            }
        )
        fig = px.bar(mini, x="Concepto", y="Unidades", color="Concepto", title="Cobertura de inventario del caso", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Si la demanda segura supera al stock actual, aparece un faltante operativo.")
    with c2:
        money_df = pd.DataFrame(
            {
                "Concepto": ["Pérdida esperada", "Costo transferencia", "Pérdida evitada", "Beneficio neto"],
                "Monto": [
                    live_decision["Perdida_Esperada_Sin_Actuar"],
                    live_decision["Costo_Transferencia"],
                    live_decision["Perdida_Evitada"],
                    live_decision["Beneficio_Neto"],
                ],
            }
        )
        fig = px.bar(money_df, x="Concepto", y="Monto", color="Concepto", title="Economía de la decisión", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("La acción tiene sentido si la pérdida evitada supera el costo de transferencia.")

    with st.expander("Ver escenarios de CEDI origen evaluados"):
        scenarios = pd.DataFrame(live_decision.get("Escenarios", []))
        if scenarios.empty:
            st.warning("No hubo CEDI origen con excedente suficiente para este caso.")
        else:
            st.dataframe(scenarios, use_container_width=True, hide_index=True)


elif page.startswith("4"):
    st.markdown("### 4) Chat con agente")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="mini-title">Chat ejecutivo</div>
<div class="helper-text">
Pregunta en lenguaje natural. El chat usa el contexto real del dashboard: recomendaciones, métricas,
caso vivo seleccionado y brief A2A-lite. Si activas Gemini y hay API key, responde con LLM; si no, usa fallback local.
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    use_gemini = st.toggle("Usar Gemini si hay API key configurada", value=False)

    example_cols = st.columns(3)
    examples = [
        "¿Por qué conviene transferir inventario en el caso seleccionado?",
        "Explícame el beneficio neto como si fuera Director de Supply Chain.",
        "¿Por qué el LLM no debe tomar la decisión financiera solo?",
    ]
    for col, ex in zip(example_cols, examples):
        if col.button(ex):
            st.session_state["pending_chat"] = ex

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": "Hola. Puedo explicarte riesgos, recomendaciones, costos, modelo, agentes y arquitectura. Selecciona un caso en 'Predicción en vivo' para dar respuestas más específicas."}
        ]

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Pregunta sobre el riesgo, costos, modelo, agentes o arquitectura…")
    if "pending_chat" in st.session_state:
        prompt = st.session_state.pop("pending_chat")

    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        live_decision = st.session_state.get("live_decision")
        context = build_context_summary(recommendations, model_metrics, live_decision, agent_brief)
        answer = gemini_agent_answer(prompt, context) if use_gemini else None
        if not answer:
            answer = local_agent_answer(prompt, context, live_decision)

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


else:
    st.markdown("### 5) Arquitectura y trazabilidad")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            """
#### MVP local-first
```text
Excel histórico
  ↓
pandas + feature engineering
  ↓
XGBoost guardado en joblib
  ↓
Decision Engine determinista
  ↓
Agentes A2A-lite / Chat
  ↓
Streamlit Dashboard
```
            """
        )
    with a2:
        st.markdown(
            """
#### Ruta cloud-ready GCP
```text
ERP/WMS/POS
  ↓
BigQuery
  ↓
Vertex AI Pipelines + Registry
  ↓
Endpoint / Batch Prediction
  ↓
Tools en Cloud Run
  ↓
Agentes interoperables vía A2A
  ↓
Dashboard ejecutivo
```
            """
        )

    st.markdown("#### Registro A2A-lite")
    if agent_registry:
        if isinstance(agent_registry, dict):
            registry_data = agent_registry.get("agents", [])
        elif isinstance(agent_registry, list):
            registry_data = agent_registry
        else:
            registry_data = []
        st.dataframe(pd.DataFrame(registry_data), use_container_width=True, hide_index=True)
    else:
        st.info("No encontré agent_a2a_simple_registry.json")

    st.markdown("#### Comparación de modelos")
    if not model_metrics.empty:
        st.dataframe(model_metrics, use_container_width=True, hide_index=True)
        fig = px.bar(model_metrics, x="model", y=["precision_risk", "recall_risk", "f1_risk"], barmode="group", title="Métricas para clase de riesgo")
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 12. FOOTER
# =============================================================================

st.caption(
    "Herdez Smart-Supply | El modelo predice, el decision engine decide con reglas y el agente explica. "
    "Diseñado para demostrar Applied AI con enfoque negocio + arquitectura."
)
