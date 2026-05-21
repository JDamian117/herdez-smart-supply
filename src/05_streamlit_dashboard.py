"""
05_streamlit_dashboard.py
Dashboard simple para Herdez Smart-Supply.

Objetivo:
- Mostrar alertas de quiebre de stock.
- Mostrar recomendaciones costo-beneficio.
- Mostrar el brief del agente A2A-lite.
- Incluir un chat sencillo para explicar las decisiones.

Cómo correr:
    python -m streamlit run app.py
    o
    python -m streamlit run src/05_streamlit_dashboard.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Herdez Smart-Supply",
    page_icon="📦",
    layout="wide",
)


# ============================================================
# 2. RUTAS ROBUSTAS DEL PROYECTO
# ============================================================

def get_project_root() -> Path:
    """
    Detecta la raíz del proyecto.

    Si ejecutas:
        streamlit run src/05_streamlit_dashboard.py
    entonces __file__ está dentro de src/ y la raíz es la carpeta padre.

    Si ejecutas:
        streamlit run app.py
    y app.py llama este archivo, también funciona si el working directory
    es la raíz del proyecto.
    """
    current_file = Path(__file__).resolve()

    if current_file.parent.name == "src":
        return current_file.parent.parent

    return current_file.parent


PROJECT_ROOT = get_project_root()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# 3. FUNCIONES AUXILIARES DE CARGA
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def money(value: Any) -> str:
    try:
        return f"${float(value):,.2f} MXN"
    except Exception:
        return "$0.00 MXN"


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "0.0%"


def safe_sum(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def safe_mean(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).mean())


# ============================================================
# 4. CARGA DE ARCHIVOS DEL PIPELINE
# ============================================================

recommendations = load_csv(OUTPUTS_DIR / "decision_recommendations.csv")
scored_alerts = load_csv(OUTPUTS_DIR / "scored_stockout_alerts.csv")
model_metrics = load_csv(OUTPUTS_DIR / "model_comparison_metrics.csv")
risk_by_sku = load_csv(OUTPUTS_DIR / "risk_by_sku.csv")
risk_by_cedi = load_csv(OUTPUTS_DIR / "risk_by_cedi.csv")
agent_registry = load_json(OUTPUTS_DIR / "agent_a2a_simple_registry.json")
agent_artifacts = load_json(OUTPUTS_DIR / "agent_a2a_simple_artifacts.json")
agent_trace = load_json(OUTPUTS_DIR / "agent_a2a_simple_trace.json")
agent_brief = load_text(OUTPUTS_DIR / "agent_executive_brief_a2a_simple.md")
eda_findings = load_text(OUTPUTS_DIR / "eda_business_findings.md")


# ============================================================
# 5. VALIDACIÓN INICIAL
# ============================================================

st.title("📦 Herdez Smart-Supply")
st.caption(
    "Sistema local-first para predecir riesgo de quiebre de stock y recomendar acciones "
    "basadas en costo-beneficio mediante ML + agentes A2A-lite."
)

if recommendations.empty:
    st.error("No encontré `outputs/decision_recommendations.csv`.")
    st.write("Ejecuta primero el pipeline desde la raíz del proyecto:")
    st.code(
        "python src/01_eda_target_features.py\n"
        "python src/02_train_models.py\n"
        "python src/03_decision_engine.py\n"
        "python src/04_agent_system_a2a_simple.py --mode fallback\n"
        "python -m streamlit run app.py",
        language="bash",
    )
    st.stop()


# ============================================================
# 6. FILTROS SENCILLOS
# ============================================================

st.sidebar.header("Filtros")

sku_options = sorted(recommendations["SKU_ID"].dropna().unique()) if "SKU_ID" in recommendations.columns else []
cedi_options = sorted(recommendations["CEDI_Destino"].dropna().unique()) if "CEDI_Destino" in recommendations.columns else []
action_options = sorted(recommendations["Accion_Recomendada"].dropna().unique()) if "Accion_Recomendada" in recommendations.columns else []

selected_skus = st.sidebar.multiselect("SKU", sku_options, default=sku_options)
selected_cedis = st.sidebar.multiselect("CEDI destino", cedi_options, default=cedi_options)
selected_actions = st.sidebar.multiselect("Acción", action_options, default=action_options)

filtered = recommendations.copy()

if selected_skus and "SKU_ID" in filtered.columns:
    filtered = filtered[filtered["SKU_ID"].isin(selected_skus)]

if selected_cedis and "CEDI_Destino" in filtered.columns:
    filtered = filtered[filtered["CEDI_Destino"].isin(selected_cedis)]

if selected_actions and "Accion_Recomendada" in filtered.columns:
    filtered = filtered[filtered["Accion_Recomendada"].isin(selected_actions)]


# ============================================================
# 7. KPIS EJECUTIVOS
# ============================================================

st.subheader("1. Resumen ejecutivo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Alertas evaluadas", f"{len(filtered):,}")

with col2:
    high_risk_count = 0
    if "Nivel_Riesgo" in filtered.columns:
        high_risk_count = int((filtered["Nivel_Riesgo"].astype(str).str.lower() == "alto").sum())
    st.metric("Alertas de riesgo alto", f"{high_risk_count:,}")

with col3:
    st.metric("Pérdida esperada", money(safe_sum(filtered, "Perdida_Esperada_Sin_Actuar")))

with col4:
    st.metric("Beneficio neto potencial", money(safe_sum(filtered, "Beneficio_Neto")))

col5, col6, col7 = st.columns(3)

with col5:
    st.metric("Costo de transferencia", money(safe_sum(filtered, "Costo_Transferencia")))

with col6:
    st.metric("Riesgo promedio", pct(safe_mean(filtered, "Riesgo_Probabilidad")))

with col7:
    transfer_count = 0
    if "Accion_Recomendada" in filtered.columns:
        transfer_count = int((filtered["Accion_Recomendada"] == "TRANSFER_INVENTORY").sum())
    st.metric("Transferencias sugeridas", f"{transfer_count:,}")


# ============================================================
# 8. TABLA PRINCIPAL DE RECOMENDACIONES
# ============================================================

st.subheader("2. Recomendaciones priorizadas")

main_columns = [
    "Fecha",
    "SKU_ID",
    "CEDI_Destino",
    "Riesgo_Probabilidad",
    "Nivel_Riesgo",
    "Accion_Recomendada",
    "CEDI_Origen_Recomendado",
    "Unidades_A_Transferir",
    "Perdida_Esperada_Sin_Actuar",
    "Costo_Transferencia",
    "Beneficio_Neto",
    "Razon",
]

available_columns = [col for col in main_columns if col in filtered.columns]

show_df = filtered[available_columns].copy()

if "Riesgo_Probabilidad" in show_df.columns:
    show_df["Riesgo_Probabilidad"] = show_df["Riesgo_Probabilidad"].apply(pct)

for money_col in ["Perdida_Esperada_Sin_Actuar", "Costo_Transferencia", "Beneficio_Neto"]:
    if money_col in show_df.columns:
        show_df[money_col] = show_df[money_col].apply(money)

st.dataframe(show_df, use_container_width=True, hide_index=True)


# ============================================================
# 9. GRÁFICAS CLARAS
# ============================================================

st.subheader("3. Visualización de riesgos y beneficio")

left, right = st.columns(2)

with left:
    if "Accion_Recomendada" in filtered.columns:
        action_count = filtered["Accion_Recomendada"].value_counts().reset_index()
        action_count.columns = ["Accion_Recomendada", "conteo"]
        fig = px.bar(
            action_count,
            x="Accion_Recomendada",
            y="conteo",
            title="Acciones recomendadas",
            text="conteo",
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    if {"CEDI_Destino", "Beneficio_Neto"}.issubset(filtered.columns):
        benefit_by_cedi = (
            filtered.groupby("CEDI_Destino", as_index=False)["Beneficio_Neto"]
            .sum()
            .sort_values("Beneficio_Neto", ascending=False)
        )
        fig = px.bar(
            benefit_by_cedi,
            x="CEDI_Destino",
            y="Beneficio_Neto",
            title="Beneficio neto potencial por CEDI",
            text_auto=".2s",
        )
        st.plotly_chart(fig, use_container_width=True)

left2, right2 = st.columns(2)

with left2:
    if not risk_by_sku.empty and {"SKU_ID", "tasa_riesgo"}.issubset(risk_by_sku.columns):
        sku_plot = risk_by_sku.sort_values("tasa_riesgo", ascending=False)
        fig = px.bar(
            sku_plot,
            x="SKU_ID",
            y="tasa_riesgo",
            title="Tasa histórica de riesgo por SKU",
            text_auto=".1%",
        )
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

with right2:
    if not risk_by_cedi.empty and {"CEDI", "tasa_riesgo"}.issubset(risk_by_cedi.columns):
        cedi_plot = risk_by_cedi.sort_values("tasa_riesgo", ascending=False)
        fig = px.bar(
            cedi_plot,
            x="CEDI",
            y="tasa_riesgo",
            title="Tasa histórica de riesgo por CEDI",
            text_auto=".1%",
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 10. BRIEF DEL AGENTE A2A-LITE
# ============================================================

st.subheader("4. Brief del agente A2A-lite")

if agent_brief:
    st.markdown(agent_brief)
else:
    st.info("No encontré `agent_executive_brief_a2a_simple.md`. Ejecuta el script 04.")

with st.expander("Ver agentes A2A-lite y trazabilidad"):
    st.markdown("### Agentes registrados")

    if agent_registry:
        if isinstance(agent_registry, dict):
            registry_data = agent_registry.get("agents", [])
        elif isinstance(agent_registry, list):
            registry_data = agent_registry
        else:
            registry_data = []

        st.dataframe(pd.DataFrame(registry_data), use_container_width=True, hide_index=True)
    else:
        st.info("No encontré registro de agentes.")

    st.markdown("### Artefactos")
    if agent_artifacts:
        st.json(agent_artifacts)
    else:
        st.info("No encontré artefactos del agente.")

    st.markdown("### Traza")
    if agent_trace:
        st.json(agent_trace)
    else:
        st.info("No encontré traza del agente.")


# ============================================================
# 11. MODELO ML Y EDA
# ============================================================

st.subheader("5. Modelo predictivo y EDA")

model_left, model_right = st.columns(2)

with model_left:
    st.markdown("#### Comparación de modelos")
    if not model_metrics.empty:
        cols = [c for c in ["model", "accuracy", "precision_risk", "recall_risk", "f1_risk", "roc_auc"] if c in model_metrics.columns]
        st.dataframe(model_metrics[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No encontré `model_comparison_metrics.csv`.")

with model_right:
    st.markdown("#### Hallazgos del EDA")
    if eda_findings:
        st.markdown(eda_findings)
    else:
        st.write(
            "El EDA revisa calidad de datos, nulos, duplicados, distribuciones, "
            "riesgo por SKU/CEDI y comportamiento temporal."
        )


# ============================================================
# 12. ARQUITECTURA SIMPLE PARA EXPLICAR
# ============================================================

st.subheader("6. Arquitectura de la solución")

st.markdown(
    """
**Flujo local-first del prototipo:**

```text
Excel histórico
   ↓
EDA + Feature Engineering
   ↓
XGBoost predice riesgo de quiebre
   ↓
Decision Engine calcula costo-beneficio
   ↓
Agentes A2A-lite interpretan, critican y explican
   ↓
Dashboard Streamlit para negocio y técnico
```

**Idea clave:** el LLM no inventa los números. El modelo predice, el motor determinista calcula,
y el agente explica la recomendación.
"""
)


# ============================================================
# 13. CHAT EJECUTIVO SIMPLE
# ============================================================

st.subheader("7. Chat con el agente")
st.caption(
    "Este chat responde usando los archivos generados por el pipeline. "
    "Funciona en modo local y no depende de que Gemini esté activo."
)


def build_context_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No hay recomendaciones disponibles."

    total_alerts = len(df)
    total_loss = money(safe_sum(df, "Perdida_Esperada_Sin_Actuar"))
    total_benefit = money(safe_sum(df, "Beneficio_Neto"))
    avg_risk = pct(safe_mean(df, "Riesgo_Probabilidad"))

    top_row = df.sort_values("Beneficio_Neto", ascending=False).iloc[0] if "Beneficio_Neto" in df.columns else df.iloc[0]

    return (
        f"Alertas evaluadas: {total_alerts}. "
        f"Riesgo promedio: {avg_risk}. "
        f"Pérdida esperada total: {total_loss}. "
        f"Beneficio neto potencial: {total_benefit}. "
        f"Caso más relevante: {top_row.get('SKU_ID', 'N/A')} en {top_row.get('CEDI_Destino', 'N/A')}, "
        f"acción {top_row.get('Accion_Recomendada', 'N/A')}, "
        f"beneficio {money(top_row.get('Beneficio_Neto', 0))}."
    )


def answer_locally(question: str, df: pd.DataFrame) -> str:
    """
    Chat simple basado en reglas.
    Esto es intencional: para la demo es estable, auditable y no depende de internet.
    """
    q = question.lower().strip()

    if df.empty:
        return "No tengo recomendaciones cargadas. Ejecuta primero los scripts 01, 02, 03 y 04."

    if any(word in q for word in ["resumen", "general", "estado", "qué pasa", "que pasa"]):
        return build_context_summary(df)

    if any(word in q for word in ["sku", "producto", "productos", "critico", "crítico"]):
        if {"SKU_ID", "Beneficio_Neto"}.issubset(df.columns):
            top = (
                df.groupby("SKU_ID", as_index=False)
                .agg(alertas=("SKU_ID", "count"), beneficio=("Beneficio_Neto", "sum"))
                .sort_values("beneficio", ascending=False)
                .head(5)
            )
            lines = ["Los SKUs más relevantes por beneficio neto potencial son:"]
            for _, row in top.iterrows():
                lines.append(f"- {row['SKU_ID']}: {int(row['alertas'])} alertas, {money(row['beneficio'])} de beneficio potencial.")
            return "\n".join(lines)

    if any(word in q for word in ["cedi", "centro", "almacen", "almacén"]):
        if {"CEDI_Destino", "Beneficio_Neto"}.issubset(df.columns):
            top = (
                df.groupby("CEDI_Destino", as_index=False)
                .agg(alertas=("CEDI_Destino", "count"), beneficio=("Beneficio_Neto", "sum"))
                .sort_values("beneficio", ascending=False)
            )
            lines = ["Prioridad por CEDI destino:"]
            for _, row in top.iterrows():
                lines.append(f"- {row['CEDI_Destino']}: {int(row['alertas'])} alertas, {money(row['beneficio'])} de beneficio potencial.")
            return "\n".join(lines)

    if any(word in q for word in ["mover", "transferir", "transferencia", "inventario"]):
        transfer_df = df[df["Accion_Recomendada"] == "TRANSFER_INVENTORY"] if "Accion_Recomendada" in df.columns else df
        if transfer_df.empty:
            return "No encontré recomendaciones de transferencia en el filtro actual. La recomendación sería revisar reabasto o monitorear."
        row = transfer_df.sort_values("Beneficio_Neto", ascending=False).iloc[0]
        return (
            f"La transferencia más atractiva es para {row.get('SKU_ID', 'N/A')} en {row.get('CEDI_Destino', 'N/A')}. "
            f"Se recomienda mover {int(row.get('Unidades_A_Transferir', 0))} unidades desde "
            f"{row.get('CEDI_Origen_Recomendado', 'N/A')}. "
            f"La pérdida esperada sin actuar es {money(row.get('Perdida_Esperada_Sin_Actuar', 0))}, "
            f"el costo de transferencia es {money(row.get('Costo_Transferencia', 0))} y "
            f"el beneficio neto estimado es {money(row.get('Beneficio_Neto', 0))}."
        )

    if any(word in q for word in ["modelo", "xgboost", "ml", "machine learning", "métrica", "metricas", "métricas"]):
        if not model_metrics.empty:
            best = model_metrics.sort_values("f1_risk", ascending=False).iloc[0] if "f1_risk" in model_metrics.columns else model_metrics.iloc[0]
            return (
                f"El modelo se compara contra baselines y modelos clásicos. "
                f"El mejor F1 de riesgo en la tabla es {best.get('model', 'N/A')} con "
                f"F1={best.get('f1_risk', 0):.3f}, recall={best.get('recall_risk', 0):.3f} y "
                f"precision={best.get('precision_risk', 0):.3f}. "
                f"La decisión de negocio no depende solo del modelo: después pasa por el motor costo-beneficio."
            )
        return "El modelo predictivo estima el riesgo de quiebre; luego el decision engine convierte ese riesgo en una acción de negocio."

    if any(word in q for word in ["agente", "a2a", "llm", "gemini", "razonamiento"]):
        return (
            "La capa de agentes usa un patrón A2A-lite: RiskLens analiza riesgo, CostGuard revisa costo-beneficio, "
            "PolicyCritic valida restricciones y ExecSupplyAI genera el brief ejecutivo. "
            "El punto clave es que el LLM no inventa números: trabaja sobre artefactos estructurados generados por ML y reglas deterministas."
        )

    if any(word in q for word in ["arquitectura", "gcp", "cloud", "bigquery", "vertex"]):
        return (
            "La arquitectura local-first usa Excel, pandas, XGBoost, decision engine, agentes A2A-lite y Streamlit. "
            "En GCP escalaría con Cloud Storage/PubSub/Dataflow para ingesta, BigQuery para datos, "
            "Vertex AI para entrenamiento/despliegue/monitoring, y agentes desplegados como servicios interoperables vía A2A."
        )

    return (
        "Puedo responder sobre: resumen ejecutivo, SKUs críticos, CEDIs críticos, transferencias recomendadas, "
        "modelo XGBoost, agentes A2A-lite o arquitectura GCP. "
        "Ejemplo: '¿qué SKU es más crítico?' o '¿por qué conviene mover inventario?'."
    )


def get_gemini_key() -> Optional[str]:
    # Streamlit Cloud permite st.secrets; localmente usamos variables de entorno.
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def answer_with_gemini(question: str, df: pd.DataFrame) -> Optional[str]:
    """
    Opción LLM: intenta responder con Gemini usando LangChain.
    Si falla, regresa None y se usa el fallback local.
    """
    api_key = get_gemini_key()
    if not api_key:
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception:
        return None

    context = build_context_summary(df)
    sample_cols = [
        "SKU_ID",
        "CEDI_Destino",
        "Riesgo_Probabilidad",
        "Accion_Recomendada",
        "CEDI_Origen_Recomendado",
        "Unidades_A_Transferir",
        "Beneficio_Neto",
        "Razon",
    ]
    sample_cols = [c for c in sample_cols if c in df.columns]
    sample = df[sample_cols].head(8).to_dict(orient="records")

    prompt = f"""
# Tu identidad
Eres ExecSupplyAI, un agente de IA para Supply Chain con experiencia en inventarios, CEDIs y costo-beneficio.

# Tu misión
Responder preguntas ejecutivas sobre el prototipo Herdez Smart-Supply usando solo los datos proporcionados.

# Límites
- No inventes cifras.
- Si el dato no está en el contexto, dilo claramente.
- Sé breve y orientado a negocio.
- No cambies la recomendación calculada por el motor determinista.

# Contexto resumido
{context}

# Muestra de recomendaciones
{sample}

# Pregunta del usuario
{question}
"""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2,
        )
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))
    except Exception:
        return None


use_gemini = st.checkbox("Usar Gemini si hay API key disponible", value=False)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "Hola, soy ExecSupplyAI. Pregúntame por SKUs críticos, CEDIs críticos, transferencias, modelo ML, agentes A2A-lite o arquitectura GCP.",
        }
    ]

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ejemplo: ¿por qué conviene mover inventario?")

if user_question:
    st.session_state.chat_messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if use_gemini:
        answer = answer_with_gemini(user_question, filtered)
        if answer is None:
            answer = answer_locally(user_question, filtered)
    else:
        answer = answer_locally(user_question, filtered)

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)


# ============================================================
# 14. PIE DE PÁGINA
# ============================================================

st.divider()
st.caption(
    "Herdez Smart-Supply | MVP local-first: EDA + XGBoost + Decision Engine + A2A-lite + Streamlit."
)
