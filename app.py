"""
05_streamlit_dashboard.py
Dashboard ejecutivo para Herdez Smart-Supply.

Objetivo:
- Mostrar alertas de riesgo de quiebre de stock.
- Mostrar recomendaciones del motor costo-beneficio.
- Mostrar el brief del sistema de agentes A2A-lite.
- Permitir simular escenarios de transferencia.

Principio de diseño:
El dashboard NO inventa recomendaciones. Solo presenta datos generados por:
1) Modelo XGBoost.
2) Decision Engine determinista.
3) Sistema de agentes A2A-lite para explicación ejecutiva.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# 1. Configuración general
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Herdez Smart-Supply",
    page_icon="📦",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# -----------------------------------------------------------------------------
# 2. Funciones auxiliares de carga
# -----------------------------------------------------------------------------

@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    """Carga un CSV desde outputs/ y estandariza fechas si existen."""
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    return df


@st.cache_data
def load_json(filename: str) -> dict[str, Any]:
    """Carga un JSON desde outputs/. Si no existe, regresa diccionario vacío."""
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_markdown(filename: str) -> str:
    """Carga un archivo Markdown desde outputs/."""
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def format_money(value: float | int | None) -> str:
    """Formatea valores monetarios para lectura ejecutiva."""
    if value is None or pd.isna(value):
        return "$0"
    return f"${value:,.0f} MXN"


def format_pct(value: float | int | None) -> str:
    """Formatea probabilidades/tasas."""
    if value is None or pd.isna(value):
        return "0.0%"
    return f"{value * 100:.1f}%"


# -----------------------------------------------------------------------------
# 3. Carga de datos
# -----------------------------------------------------------------------------

recommendations = load_csv("decision_recommendations.csv")
alerts = load_csv("scored_stockout_alerts.csv")
model_metrics = load_csv("model_comparison_metrics.csv")
risk_by_sku = load_csv("risk_by_sku.csv")
risk_by_cedi = load_csv("risk_by_cedi.csv")
agent_brief = load_markdown("agent_executive_brief_a2a_simple.md")
agent_trace = load_json("agent_a2a_simple_trace.json")
agent_artifacts = load_json("agent_a2a_simple_artifacts.json")
agent_registry = load_json("agent_a2a_simple_registry.json")


# -----------------------------------------------------------------------------
# 4. Encabezado
# -----------------------------------------------------------------------------

st.title("📦 Herdez Smart-Supply")
st.caption(
    "Sistema local-first para predecir riesgo de quiebre de stock y recomendar acciones "
    "basadas en costo-beneficio mediante ML + agentes A2A-lite."
)

if recommendations.empty:
    st.error(
        "No encontré `outputs/decision_recommendations.csv`. "
        "Ejecuta primero los scripts 01, 02, 03 y 04."
    )
    st.code(
        "python src/01_eda_target_features.py\n"
        "python src/02_train_models.py\n"
        "python src/03_decision_engine.py\n"
        "python src/04_agent_system_a2a_simple.py --mode fallback\n"
        "streamlit run src/05_streamlit_dashboard.py",
        language="bash",
    )
    st.stop()


# -----------------------------------------------------------------------------
# 5. Filtros ejecutivos
# -----------------------------------------------------------------------------

st.sidebar.header("Filtros")

sku_options = ["Todos"] + sorted(recommendations["SKU_ID"].dropna().unique().tolist())
cedi_options = ["Todos"] + sorted(recommendations["CEDI_Destino"].dropna().unique().tolist())
action_options = ["Todas"] + sorted(recommendations["Accion_Recomendada"].dropna().unique().tolist())
risk_options = ["Todos"] + sorted(recommendations["Nivel_Riesgo"].dropna().unique().tolist())

selected_sku = st.sidebar.selectbox("SKU", sku_options)
selected_cedi = st.sidebar.selectbox("CEDI destino", cedi_options)
selected_action = st.sidebar.selectbox("Acción recomendada", action_options)
selected_risk = st.sidebar.selectbox("Nivel de riesgo", risk_options)

filtered = recommendations.copy()
if selected_sku != "Todos":
    filtered = filtered[filtered["SKU_ID"] == selected_sku]
if selected_cedi != "Todos":
    filtered = filtered[filtered["CEDI_Destino"] == selected_cedi]
if selected_action != "Todas":
    filtered = filtered[filtered["Accion_Recomendada"] == selected_action]
if selected_risk != "Todos":
    filtered = filtered[filtered["Nivel_Riesgo"] == selected_risk]


# -----------------------------------------------------------------------------
# 6. KPIs principales
# -----------------------------------------------------------------------------

total_alerts = len(filtered)
high_risk = int((filtered["Nivel_Riesgo"] == "Alto").sum()) if not filtered.empty else 0
expected_loss = float(filtered["Perdida_Esperada_Sin_Actuar"].sum()) if not filtered.empty else 0.0
net_benefit = float(filtered["Beneficio_Neto"].sum()) if not filtered.empty else 0.0
transfer_units = int(filtered["Unidades_A_Transferir"].sum()) if not filtered.empty else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Alertas priorizadas", f"{total_alerts}")
kpi2.metric("Riesgo alto", f"{high_risk}")
kpi3.metric("Pérdida esperada", format_money(expected_loss))
kpi4.metric("Beneficio neto potencial", format_money(net_benefit))
kpi5.metric("Unidades a transferir", f"{transfer_units:,}")

st.divider()


# -----------------------------------------------------------------------------
# 7. Tabs principales
# -----------------------------------------------------------------------------

tab_exec, tab_alerts, tab_agents, tab_model, tab_simulator, tab_arch = st.tabs(
    [
        "Resumen ejecutivo",
        "Alertas y recomendaciones",
        "Agentes A2A-lite",
        "Modelo ML",
        "Simulador",
        "Arquitectura",
    ]
)


# -----------------------------------------------------------------------------
# Tab 1: Resumen ejecutivo
# -----------------------------------------------------------------------------

with tab_exec:
    st.subheader("Resumen ejecutivo")
    st.write(
        "Este tablero está diseñado para un Director de Supply Chain: muestra dónde existe "
        "riesgo de quiebre, cuánto costaría no actuar y cuál es la recomendación operativa."
    )

    if agent_brief:
        st.markdown("### Brief generado por el agente")
        st.markdown(agent_brief)
    else:
        st.info("No encontré el brief del agente. Ejecuta `04_agent_system_a2a_simple.py`.")

    st.markdown("### Top recomendaciones por beneficio neto")
    top_recs = filtered.sort_values("Beneficio_Neto", ascending=False).head(10)
    st.dataframe(
        top_recs[
            [
                "Fecha",
                "SKU_ID",
                "CEDI_Destino",
                "Riesgo_Probabilidad",
                "Accion_Recomendada",
                "CEDI_Origen_Recomendado",
                "Unidades_A_Transferir",
                "Perdida_Esperada_Sin_Actuar",
                "Beneficio_Neto",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# Tab 2: Alertas y recomendaciones
# -----------------------------------------------------------------------------

with tab_alerts:
    st.subheader("Alertas y recomendaciones")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Riesgo promedio por CEDI")
        if not filtered.empty:
            cedi_chart = (
                filtered.groupby("CEDI_Destino", as_index=False)["Riesgo_Probabilidad"]
                .mean()
                .sort_values("Riesgo_Probabilidad", ascending=False)
                .set_index("CEDI_Destino")
            )
            st.bar_chart(cedi_chart)
        else:
            st.info("No hay datos con los filtros actuales.")

    with right:
        st.markdown("#### Beneficio neto por SKU")
        if not filtered.empty:
            sku_chart = (
                filtered.groupby("SKU_ID", as_index=False)["Beneficio_Neto"]
                .sum()
                .sort_values("Beneficio_Neto", ascending=False)
                .set_index("SKU_ID")
            )
            st.bar_chart(sku_chart)
        else:
            st.info("No hay datos con los filtros actuales.")

    st.markdown("#### Tabla detallada")
    st.dataframe(
        filtered.sort_values(["Beneficio_Neto", "Riesgo_Probabilidad"], ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# Tab 3: Agentes A2A-lite
# -----------------------------------------------------------------------------

with tab_agents:
    st.subheader("Sistema de agentes A2A-lite")
    st.write(
        "El prototipo usa un patrón A2A-lite local: cada agente recibe un artefacto estructurado, "
        "lo analiza y genera otro artefacto para el siguiente agente. En producción, cada agente "
        "podría exponerse como servicio A2A real."
    )

    st.markdown("### Registro de agentes")
    if agent_registry:
        registry_df = pd.DataFrame(agent_registry.get("agents", agent_registry))
        st.dataframe(registry_df, use_container_width=True, hide_index=True)
    else:
        st.info("No encontré `agent_a2a_simple_registry.json`.")

    st.markdown("### Traza de ejecución")
    if agent_trace:
        st.json(agent_trace)
    else:
        st.info("No encontré `agent_a2a_simple_trace.json`.")

    st.markdown("### Artefactos generados")
    if agent_artifacts:
        st.json(agent_artifacts)
    else:
        st.info("No encontré `agent_a2a_simple_artifacts.json`.")


# -----------------------------------------------------------------------------
# Tab 4: Modelo ML
# -----------------------------------------------------------------------------

with tab_model:
    st.subheader("Modelo predictivo")
    st.write(
        "El modelo estima riesgo de quiebre dentro de la ventana operativa del lead time. "
        "La salida del modelo alimenta el motor costo-beneficio, pero no toma la decisión final."
    )

    if not model_metrics.empty:
        st.markdown("### Comparación de modelos")
        st.dataframe(model_metrics, use_container_width=True, hide_index=True)

        metric_chart = model_metrics.set_index("model")[["precision_risk", "recall_risk", "f1_risk", "roc_auc"]]
        st.bar_chart(metric_chart)
    else:
        st.info("No encontré métricas de modelos.")

    left, right = st.columns(2)
    with left:
        st.markdown("### Tasa histórica de riesgo por SKU")
        if not risk_by_sku.empty:
            st.bar_chart(risk_by_sku.set_index("SKU_ID")[["tasa_riesgo"]])
        else:
            st.info("No encontré risk_by_sku.csv")

    with right:
        st.markdown("### Tasa histórica de riesgo por CEDI")
        if not risk_by_cedi.empty:
            st.bar_chart(risk_by_cedi.set_index("CEDI")[["tasa_riesgo"]])
        else:
            st.info("No encontré risk_by_cedi.csv")


# -----------------------------------------------------------------------------
# Tab 5: Simulador
# -----------------------------------------------------------------------------

with tab_simulator:
    st.subheader("Simulador de transferencia")
    st.write(
        "Este simulador permite modificar unidades a transferir para ver cómo cambia el beneficio neto. "
        "Es útil para explicar el criterio costo-beneficio al Director de Supply Chain."
    )

    if filtered.empty:
        st.info("No hay recomendaciones disponibles con los filtros actuales.")
    else:
        options = filtered.index.tolist()
        selected_idx = st.selectbox(
            "Selecciona una alerta",
            options,
            format_func=lambda idx: (
                f"{filtered.loc[idx, 'SKU_ID']} | {filtered.loc[idx, 'CEDI_Destino']} | "
                f"{format_pct(filtered.loc[idx, 'Riesgo_Probabilidad'])} riesgo"
            ),
        )
        row = filtered.loc[selected_idx]

        st.markdown("### Caso seleccionado")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SKU", str(row["SKU_ID"]))
        c2.metric("CEDI destino", str(row["CEDI_Destino"]))
        c3.metric("Riesgo", format_pct(row["Riesgo_Probabilidad"]))
        c4.metric("Unidades necesarias", f"{int(row['Unidades_Necesarias']):,}")

        max_units = max(int(row["Unidades_Necesarias"]), int(row["Unidades_A_Transferir"]), 1)
        default_units = int(row["Unidades_A_Transferir"])
        units = st.slider(
            "Unidades a transferir",
            min_value=0,
            max_value=max_units,
            value=min(default_units, max_units),
            step=max(1, max_units // 100),
        )

        unit_transfer_cost = 0.0
        if row["Unidades_A_Transferir"] > 0:
            unit_transfer_cost = row["Costo_Transferencia"] / row["Unidades_A_Transferir"]
        else:
            # aproximación conservadora si el motor no recomendó transferencia
            unit_transfer_cost = row["Costo_Transferencia"] if row["Costo_Transferencia"] > 0 else 12.0

        coverage_ratio = min(units / max(row["Unidades_Necesarias"], 1), 1.0)
        simulated_transfer_cost = units * unit_transfer_cost
        simulated_avoided_loss = row["Perdida_Esperada_Sin_Actuar"] * coverage_ratio
        simulated_net_benefit = simulated_avoided_loss - simulated_transfer_cost

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Cobertura del faltante", format_pct(coverage_ratio))
        s2.metric("Costo transferencia", format_money(simulated_transfer_cost))
        s3.metric("Pérdida evitada", format_money(simulated_avoided_loss))
        s4.metric("Beneficio neto", format_money(simulated_net_benefit))

        if simulated_net_benefit > 0 and units > 0:
            st.success("La simulación sugiere que la transferencia es financieramente conveniente.")
        elif units == 0:
            st.warning("No transferir unidades mantiene la pérdida esperada sin mitigación.")
        else:
            st.error("La simulación sugiere que el costo supera la pérdida evitada.")


# -----------------------------------------------------------------------------
# Tab 6: Arquitectura
# -----------------------------------------------------------------------------

with tab_arch:
    st.subheader("Arquitectura del prototipo y ruta cloud-ready")

    st.markdown(
        """
### MVP local-first

```text
Excel histórico
   ↓
pandas / feature engineering
   ↓
XGBoost
   ↓
Decision Engine determinista
   ↓
Agentes A2A-lite + Gemini/LangChain opcional
   ↓
Streamlit Dashboard
```

### Producción cloud-ready en GCP

```text
ERP / WMS / POS / Promociones / Clima
   ↓
Cloud Storage / Pub/Sub / Dataflow
   ↓
BigQuery
   ↓
Vertex AI Pipelines + Training + Model Registry
   ↓
Vertex AI Endpoint / Batch Prediction
   ↓
Agentes interoperables vía A2A
   ↓
Gemini / Function Calling / Cloud Run services
   ↓
Dashboard ejecutivo / Looker / Streamlit en Cloud Run
```

### Tesis técnica

El prototipo es local-first y portable. La arquitectura separa tres responsabilidades:

1. **Predicción:** XGBoost estima riesgo de quiebre.
2. **Decisión:** el motor determinista calcula costo-beneficio y valida reglas.
3. **Explicación:** los agentes A2A-lite interpretan y comunican la recomendación.

Así se reducen alucinaciones, se mantiene trazabilidad y se conserva una ruta clara hacia GCP.
        """
    )
