"""
05_streamlit_dashboard.py
Herdez Smart-Supply | Dashboard operacional con línea de tiempo, predicción en vivo y chat

Qué demuestra esta versión
--------------------------
1) Usa TODO el histórico disponible para explorar ventas, stock y riesgo en el tiempo.
2) Permite filtrar por intervalo de fechas, SKU y CEDI.
3) Mantiene las alertas priorizadas como capa de decisión, no como único dato analizado.
4) Ejecuta predicción en vivo con el modelo guardado y calcula recomendación costo-beneficio.
5) Incluye chat explicativo con fallback local y Gemini opcional.

Principio de arquitectura
-------------------------
- El modelo ML predice riesgo de quiebre.
- El motor determinista calcula la recomendación y el impacto económico.
- El chat/agente explica la decisión, pero no inventa números críticos.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    candidates = [current.parent, current.parent.parent, Path.cwd(), Path.cwd().parent]
    for candidate in candidates:
        if (candidate / "outputs").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = find_project_root()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"


# =============================================================================
# 3. CSS / SISTEMA VISUAL
# =============================================================================

def inject_css() -> None:
    """Estilos para cards, contraste, sombras y legibilidad en modo claro/oscuro."""
    st.markdown(
        """
        <style>
        :root {
            --card-bg: rgba(255, 255, 255, 0.82);
            --card-border: rgba(15, 23, 42, 0.11);
            --text-muted: #64748b;
            --charcoal: #111827;
            --blue: #2563eb;
            --green: #16a34a;
            --red: #dc2626;
            --amber: #d97706;
            --soft-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --card-bg: rgba(17, 24, 39, 0.78);
                --card-border: rgba(255, 255, 255, 0.14);
                --text-muted: #cbd5e1;
                --charcoal: #f8fafc;
                --soft-shadow: 0 16px 36px rgba(0, 0, 0, 0.26);
            }
        }

        .main .block-container {
            padding-top: 1.35rem;
            padding-bottom: 3rem;
            max-width: 1420px;
        }

        .hero {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(22, 163, 74, 0.10));
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 1.35rem 1.55rem;
            box-shadow: var(--soft-shadow);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.15rem;
            font-weight: 850;
            letter-spacing: -0.045em;
            color: var(--charcoal);
            margin-bottom: 0.25rem;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: var(--text-muted);
            line-height: 1.55;
            max-width: 1100px;
        }
        .pill {
            display: inline-block;
            padding: 0.24rem 0.68rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 750;
            margin-right: 0.35rem;
            margin-top: 0.45rem;
        }
        .pill-blue { background: rgba(37,99,235,0.14); color: #2563eb; }
        .pill-green { background: rgba(22,163,74,0.14); color: #16a34a; }
        .pill-red { background: rgba(220,38,38,0.14); color: #dc2626; }
        .pill-amber { background: rgba(217,119,6,0.14); color: #d97706; }

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
            font-weight: 800;
            margin-bottom: 0.22rem;
            color: var(--charcoal);
        }
        .helper-text {
            font-size: 0.93rem;
            color: var(--text-muted);
            line-height: 1.55;
        }
        .decision-box {
            border-left: 5px solid var(--blue);
            padding: 0.95rem 1rem;
            border-radius: 12px;
            background: rgba(37, 99, 235, 0.08);
            margin-top: 0.6rem;
            margin-bottom: 1rem;
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
        .agent-step {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 0.95rem 1.05rem;
            box-shadow: var(--soft-shadow);
            margin-bottom: 0.75rem;
        }
        .agent-step-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 850;
            color: var(--charcoal);
            margin-bottom: 0.25rem;
        }
        .agent-badge {
            display: inline-block;
            padding: 0.16rem 0.52rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
            background: rgba(37,99,235,0.12);
            color: #2563eb;
        }
        .agent-arrow {
            text-align: center;
            color: var(--text-muted);
            font-weight: 900;
            margin: -0.15rem 0 0.45rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =============================================================================
# 4. FUNCIONES AUXILIARES DE CARGA Y FORMATO
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
        return {}
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
        st.warning(f"No pude cargar el modelo guardado. Usaré baseline de cobertura. Detalle: {exc}")
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


def percentage_points(value: Any) -> float:
    try:
        return float(value) * 100
    except Exception:
        return 0.0


# =============================================================================
# 5. LÓGICA DE PREDICCIÓN Y DECISIÓN EN VIVO
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
    """Predice riesgo con XGBoost. Si no hay modelo compatible, usa baseline de cobertura."""
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
        except Exception:
            pass

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
    risk_level = classify_risk(risk_prob)

    demand_safe = float(row["Ventas_Media_7d"] * row["Lead_Time_Dias"] * DecisionConfig.safety_factor)
    units_needed = int(np.ceil(max(0, demand_safe - float(row["Stock_Actual"]))))
    expected_loss = float(risk_prob * float(row["Costo_Quiebre_Stock_Diario"]) * float(row["Lead_Time_Dias"]))

    # Buscamos origen el mismo día, mismo SKU, CEDI diferente.
    base = scored_df.copy()
    if "Fecha" in base.columns:
        base["Fecha"] = pd.to_datetime(base["Fecha"], errors="coerce")
    candidates = base[
        (base["Fecha"].dt.date == pd.to_datetime(row["Fecha"]).date())
        & (base["SKU_ID"] == row["SKU_ID"])
        & (base["CEDI"] != row["CEDI"])
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
    try:
        scored = features_df.copy()
        feature_columns = artifact["feature_columns"]
        pipeline = artifact["pipeline"]
        for col in feature_columns:
            if col not in scored.columns:
                scored[col] = np.nan
        scored["Riesgo_Probabilidad"] = pipeline.predict_proba(scored[feature_columns])[:, 1]
        threshold = float(artifact.get("threshold", 0.5))
        scored["Alerta_Riesgo"] = (scored["Riesgo_Probabilidad"] >= threshold).astype(int)
        return scored
    except Exception:
        return features_df.copy()



# =============================================================================
# 6. TRAZABILIDAD A2A-LITE / CONSENSO ENTRE AGENTES
# =============================================================================

def normalize_agent_artifacts(raw_artifacts: Any) -> Dict[str, Dict[str, Any]]:
    """Normaliza artefactos A2A-lite.

    El archivo puede venir como:
    - dict con claves risk_analysis/cost_analysis/policy_review/executive_brief
    - dict con clave artifacts
    - lista de artefactos

    Devolvemos siempre un diccionario uniforme para renderizar consenso.
    """
    if not raw_artifacts:
        return {}

    if isinstance(raw_artifacts, dict) and "artifacts" in raw_artifacts:
        raw_artifacts = raw_artifacts.get("artifacts", {})

    if isinstance(raw_artifacts, dict):
        return {str(k): v for k, v in raw_artifacts.items() if isinstance(v, dict)}

    if isinstance(raw_artifacts, list):
        normalized: Dict[str, Dict[str, Any]] = {}
        for i, item in enumerate(raw_artifacts):
            if isinstance(item, dict):
                key = str(item.get("artifact_name") or item.get("output_artifact") or item.get("agent_name") or f"artifact_{i}")
                normalized[key] = item
        return normalized

    return {}


def build_agent_consensus_steps(raw_artifacts: Any, live_decision: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Construye una narrativa simple de cómo los agentes llegan a la recomendación.

    Importante: no mostramos razonamiento interno oculto. Mostramos trazabilidad del sistema:
    entradas, artefactos y conclusiones resumidas.
    """
    artifacts = normalize_agent_artifacts(raw_artifacts)

    order = [
        ("risk_analysis", "RiskLens", "Analista de riesgo", "Evalúa si el SKU/CEDI puede quedarse sin producto."),
        ("cost_analysis", "CostGuard", "Analista costo-beneficio", "Compara pérdida esperada contra costo logístico."),
        ("policy_review", "PolicyCritic", "Validador de política", "Revisa si la acción no traslada el problema a otro CEDI."),
        ("executive_brief", "ExecSupplyAI", "Comunicador ejecutivo", "Convierte la conclusión en lenguaje de negocio."),
    ]

    fallback_summaries = {
        "risk_analysis": "El riesgo proviene del modelo ML y del histórico filtrado. Si el riesgo es alto, el caso pasa a evaluación económica.",
        "cost_analysis": "La acción solo se recomienda si la pérdida evitada supera el costo de transferencia.",
        "policy_review": "La recomendación debe respetar reglas operativas, especialmente no dejar al CEDI origen vulnerable.",
        "executive_brief": "El resultado final se resume para negocio: riesgo, costo, beneficio y siguiente acción.",
    }

    steps: List[Dict[str, str]] = []
    for key, agent_name, role, purpose in order:
        artifact = artifacts.get(key, {})
        summary = str(artifact.get("summary") or fallback_summaries[key])
        next_action = str(artifact.get("next_action") or "Continuar al siguiente agente.")
        caveats = artifact.get("risks_or_caveats") or []
        if isinstance(caveats, list):
            caveat_text = "; ".join(str(c) for c in caveats[:2])
        else:
            caveat_text = str(caveats)

        steps.append(
            {
                "agent": str(artifact.get("agent_name") or agent_name),
                "role": role,
                "purpose": purpose,
                "summary": summary,
                "caveat": caveat_text,
                "next_action": next_action,
            }
        )

    if live_decision:
        final_action = str(live_decision.get("Accion_Recomendada", "N/D"))
        final_benefit = money(live_decision.get("Beneficio_Neto", 0))
        steps.append(
            {
                "agent": "Decision Engine",
                "role": "Motor determinista",
                "purpose": "Cierra la recomendación con números verificables.",
                "summary": f"Recomendación del caso vivo: {final_action}. Beneficio neto estimado: {final_benefit}.",
                "caveat": "El LLM no modifica este cálculo; solo lo explica.",
                "next_action": "Mostrar recomendación al usuario y permitir preguntas en el chat.",
            }
        )

    return steps


def consensus_text(raw_artifacts: Any, live_decision: Optional[Dict[str, Any]] = None) -> str:
    """Genera texto resumido para el chat sobre el acuerdo entre agentes."""
    steps = build_agent_consensus_steps(raw_artifacts, live_decision)
    lines = ["Así se coordinan los agentes A2A-lite:"]
    for i, step in enumerate(steps, start=1):
        lines.append(
            f"{i}. {step['agent']} ({step['role']}): {step['summary']} "
            f"Siguiente acción: {step['next_action']}"
        )
    lines.append(
        "Conclusión: no hay un agente que decida solo. RiskLens detecta riesgo, CostGuard valida dinero, "
        "PolicyCritic revisa restricciones y ExecSupplyAI comunica la decisión. Los números críticos vienen del modelo y del motor determinista."
    )
    return "\n".join(lines)


def render_agent_consensus(raw_artifacts: Any, live_decision: Optional[Dict[str, Any]] = None) -> None:
    """Muestra una vista visual de consenso/hand-off entre agentes."""
    steps = build_agent_consensus_steps(raw_artifacts, live_decision)

    st.markdown(
        """
        <div class="section-card">
          <div class="mini-title">🧠 Cómo se ponen de acuerdo los agentes</div>
          <div class="helper-text">
          Esta vista no muestra pensamiento oculto del modelo. Muestra la trazabilidad del sistema:
          qué artefacto produce cada agente, qué valida y cómo se pasa la decisión al siguiente paso.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, step in enumerate(steps):
        st.markdown(
            f"""
            <div class="agent-step">
              <div class="agent-step-header">
                <span class="agent-badge">Paso {idx + 1}</span>
                {step['agent']} · {step['role']}
              </div>
              <div class="helper-text"><b>Qué revisa:</b> {step['purpose']}</div>
              <div class="helper-text"><b>Conclusión:</b> {step['summary']}</div>
              <div class="helper-text"><b>Cuidado:</b> {step['caveat'] if step['caveat'] else 'Sin advertencias adicionales.'}</div>
              <div class="helper-text"><b>Siguiente paso:</b> {step['next_action']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if idx < len(steps) - 1:
            st.markdown('<div class="agent-arrow">↓</div>', unsafe_allow_html=True)



def build_visible_agent_activity(question: str, live_decision: Optional[Dict[str, Any]], raw_artifacts: Any) -> List[Dict[str, str]]:
    """Crea una bitácora visible de coordinación entre agentes para el chat.

    Nota importante de diseño:
    - Esto NO intenta mostrar el razonamiento interno privado del LLM.
    - Muestra una traza pública/auditable: qué revisa cada agente, qué dato usa y qué artefacto entrega.
    - Es ideal para demo porque enseña coordinación multiagente sin exponer chain-of-thought.
    """
    risk = pct(live_decision.get("Riesgo_Probabilidad")) if live_decision else "riesgo no seleccionado"
    level = str(live_decision.get("Nivel_Riesgo")) if live_decision else "sin caso seleccionado"
    action = str(live_decision.get("Accion_Recomendada")) if live_decision else "respuesta general"
    benefit = money(live_decision.get("Beneficio_Neto", 0)) if live_decision else "no aplica"
    sku = str(live_decision.get("SKU_ID")) if live_decision else "SKU no seleccionado"
    cedi = str(live_decision.get("CEDI_Destino")) if live_decision else "CEDI no seleccionado"

    return [
        {
            "agent": "Router",
            "status": "Recibe la pregunta",
            "visible_work": f"Clasifica la intención del usuario: '{question[:90]}'.",
            "artifact": "Tipo de consulta: histórico, riesgo, costos, agentes, arquitectura o caso vivo.",
        },
        {
            "agent": "RiskLens",
            "status": "Consulta riesgo",
            "visible_work": f"Revisa el caso {sku} en {cedi} y recupera la probabilidad de quiebre.",
            "artifact": f"Artefacto de riesgo: {risk}, nivel {level}.",
        },
        {
            "agent": "CostGuard",
            "status": "Valida economía",
            "visible_work": "Compara pérdida esperada, costo de transferencia y beneficio neto.",
            "artifact": f"Artefacto financiero: recomendación {action}, beneficio neto {benefit}.",
        },
        {
            "agent": "PolicyCritic",
            "status": "Revisa restricciones",
            "visible_work": "Verifica que la respuesta no prometa acciones fuera de los datos y que no invente números.",
            "artifact": "Artefacto de control: usar solo datos del modelo, motor de decisión y archivos generados.",
        },
        {
            "agent": "ExecSupplyAI",
            "status": "Redacta respuesta",
            "visible_work": "Convierte la salida técnica en una explicación ejecutiva para negocio.",
            "artifact": "Respuesta final: clara, accionable y en lenguaje no técnico.",
        },
    ]


def render_live_agent_activity(question: str, live_decision: Optional[Dict[str, Any]], raw_artifacts: Any) -> None:
    """Muestra una simulación auditable de coordinación mientras se construye la respuesta del chat."""
    st.markdown("#### 🔄 Coordinación visible de agentes")
    st.caption(
        "Esta bitácora muestra pasos públicos y auditables del sistema. "
        "No expone pensamiento interno oculto; muestra qué datos revisa cada agente y qué artefacto entrega."
    )

    progress = st.progress(0)
    feed = st.container()
    steps = build_visible_agent_activity(question, live_decision, raw_artifacts)

    completed: List[Dict[str, str]] = []
    for i, step in enumerate(steps, start=1):
        completed.append(step)
        progress.progress(i / len(steps))
        with feed:
            st.markdown(
                f"""
                <div class="agent-step">
                  <div class="agent-step-header">
                    <span class="agent-badge">{step['agent']}</span>
                    {step['status']}
                  </div>
                  <div class="helper-text"><b>Trabajo visible:</b> {step['visible_work']}</div>
                  <div class="helper-text"><b>Artefacto entregado:</b> {step['artifact']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        time.sleep(0.25)

    st.success("Consenso listo: el chat usará estos artefactos para responder.")

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
    features_filtered: pd.DataFrame,
    scored_filtered: pd.DataFrame,
    model_metrics: pd.DataFrame,
    live_decision: Optional[Dict[str, Any]],
    agent_brief: str,
    agent_artifacts: Any = None,
) -> str:
    lines = []
    lines.append(f"Registros históricos en el intervalo: {len(features_filtered)}")
    lines.append(f"Predicciones en el intervalo: {len(scored_filtered)}")
    lines.append(f"Alertas priorizadas en el intervalo: {len(recommendations)}")
    if not recommendations.empty:
        total_loss = recommendations["Perdida_Esperada_Sin_Actuar"].sum()
        total_net = recommendations["Beneficio_Neto"].sum()
        lines.append(f"Pérdida esperada agregada: {money(total_loss)}")
        lines.append(f"Beneficio neto potencial agregado: {money(total_net)}")
        top = recommendations.sort_values("Beneficio_Neto", ascending=False).head(3)
        lines.append("Top recomendaciones:")
        for _, r in top.iterrows():
            lines.append(
                f"- {r['SKU_ID']} en {r['CEDI_Destino']}: {r['Accion_Recomendada']}, "
                f"beneficio {money(r['Beneficio_Neto'])}"
            )
    if not model_metrics.empty and {"model", "f1_risk"}.issubset(model_metrics.columns):
        best = model_metrics.sort_values("f1_risk", ascending=False).head(1).iloc[0]
        lines.append(
            f"Mejor modelo por F1 riesgo: {best.get('model', 'N/D')} con F1={best.get('f1_risk', 0):.3f}."
        )
    if live_decision:
        compact = {k: v for k, v in live_decision.items() if k != "Escenarios"}
        lines.append("Caso vivo seleccionado:")
        lines.append(json.dumps(compact, ensure_ascii=False, default=str))
    if agent_brief:
        lines.append("Brief A2A-lite disponible: sí.")
    if agent_artifacts:
        lines.append("Consenso A2A-lite disponible:")
        lines.append(consensus_text(agent_artifacts, live_decision))
    return "\n".join(lines)


def local_agent_answer(question: str, context: str, live_decision: Optional[Dict[str, Any]], agent_artifacts: Any = None) -> str:
    """Fallback local para responder sin API key. No inventa datos."""
    q = question.lower()

    if any(w in q for w in ["1200", "50", "datos", "registros", "histórico", "historico"]):
        return (
            "El sistema usa varias capas de datos. El histórico completo sirve para entender comportamiento general; "
            "el dataset modelable sirve para entrenar/evaluar el modelo; y las 50 alertas priorizadas son solo los casos "
            "más accionables para no saturar al usuario. Es como revisar todo el historial médico, pero mostrar primero las señales urgentes."
        )

    if any(w in q for w in ["fecha", "periodo", "intervalo", "tiempo", "temporal"]):
        return (
            "El filtro de fechas permite analizar un periodo específico. Primero vemos ventas y stock en el tiempo, luego el riesgo promedio, "
            "y finalmente las alertas priorizadas de ese mismo intervalo. Así no se pierde el contexto histórico antes de tomar una decisión."
        )

    if any(w in q for w in ["quiebre", "stock", "riesgo", "quedar sin producto"]):
        if live_decision:
            return (
                f"Para el caso seleccionado, el riesgo estimado es **{pct(live_decision['Riesgo_Probabilidad'])}** "
                f"y el nivel es **{live_decision['Nivel_Riesgo']}**. El sistema revisa si el inventario actual alcanza hasta el reabasto. "
                f"La demanda segura estimada es **{live_decision['Demanda_Estimada_LT_Segura']:,.0f} unidades** y faltan "
                f"**{live_decision['Unidades_Necesarias']:,} unidades**."
            )
        return "El riesgo de quiebre estima si un SKU podría quedarse sin inventario antes de que llegue el reabasto."

    if any(w in q for w in ["mover", "transfer", "transferir", "cedi", "origen"]):
        if live_decision:
            return (
                f"La recomendación del caso seleccionado es **{live_decision['Accion_Recomendada']}**. "
                f"Origen recomendado: **{live_decision['CEDI_Origen_Recomendado']}**. "
                f"Unidades a transferir: **{live_decision['Unidades_A_Transferir']:,}**. "
                f"Razón: {live_decision['Razon']}"
            )
        return "La transferencia se recomienda si hay CEDI origen con excedente y si la pérdida evitada supera el costo logístico."

    if any(w in q for w in ["costo", "beneficio", "dinero", "roi", "perdida", "pérdida"]):
        if live_decision:
            return (
                f"Pérdida esperada sin actuar: **{money(live_decision['Perdida_Esperada_Sin_Actuar'])}**. "
                f"Costo de transferencia: **{money(live_decision['Costo_Transferencia'])}**. "
                f"Pérdida evitada: **{money(live_decision['Perdida_Evitada'])}**. "
                f"Beneficio neto: **{money(live_decision['Beneficio_Neto'])}**. "
                "La regla de negocio es: conviene actuar cuando evitar la pérdida cuesta menos que no hacer nada."
            )
        return "El motor compara pérdida esperada contra costo de transferencia para estimar beneficio neto."

    if any(w in q for w in ["modelo", "xgboost", "ml", "machine"]):
        return (
            "XGBoost se usa porque funciona bien con datos tabulares y detecta combinaciones no lineales, por ejemplo: "
            "bajo stock + alto lead time + promoción activa. Aun así, el modelo no toma la decisión final: solo entrega una probabilidad."
        )

    if any(w in q for w in ["acuerdo", "consenso", "ponen de acuerdo", "pusieron de acuerdo", "coordina", "coordinan", "traza", "trazabilidad", "artefacto", "artefactos"]):
        return consensus_text(agent_artifacts, live_decision)

    if any(w in q for w in ["agente", "a2a", "chat", "llm", "gemini"]):
        base = (
            "El agente explica la decisión, pero no inventa números. El patrón A2A-lite separa responsabilidades: análisis de riesgo, "
            "costos, políticas y comunicación ejecutiva. Esto reduce alucinaciones y mejora trazabilidad."
        )
        return base + "\n\n" + consensus_text(agent_artifacts, live_decision)

    if any(w in q for w in ["cloud", "gcp", "vertex", "bigquery", "arquitectura"]):
        return (
            "La ruta cloud-ready sería: ERP/WMS/POS → BigQuery → Vertex AI para entrenamiento y monitoreo → tools en Cloud Run → "
            "agentes interoperables vía A2A → dashboard ejecutivo."
        )

    return (
        "Puedo ayudarte a interpretar el histórico, el filtro de fechas, el riesgo, costos, modelo, agentes o arquitectura. "
        "Prueba: '¿por qué solo hay 50 alertas?' o 'explícame el beneficio neto del caso seleccionado'."
    )


def gemini_agent_answer(question: str, context: str) -> Optional[str]:
    api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2)
        prompt = f"""
# Tu identidad
Eres un copiloto ejecutivo de Supply Chain y Applied AI. Explicas decisiones de inventario para usuarios no técnicos.

# Tu misión
Responder preguntas sobre el dashboard Herdez Smart-Supply usando únicamente el contexto disponible.

# Límites
- No inventes números.
- Si falta información, dilo claramente.
- Usa lenguaje de negocio.
- Si ayuda, usa analogías sencillas.
- Responde en español.

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
agent_artifacts = load_json("agent_a2a_simple_artifacts.json")
model_artifact = load_model_artifact()

if scored_alerts.empty and not features.empty:
    scored_alerts = score_dataset_cached(features)

if features.empty or recommendations.empty:
    st.error("No encontré los archivos mínimos del dashboard. Ejecuta el pipeline 01→04 y vuelve a cargar la app.")
    st.code(
        "python src/01_eda_target_features.py --input data/Data_Prueba_Tecnica_Herdez_IA.xlsx\n"
        "python src/02_train_models.py --processed outputs/herdez_features_dataset.csv\n"
        "python src/03_decision_engine.py --processed outputs/herdez_features_dataset.csv --model models/best_stockout_model.joblib\n"
        "python src/04_agent_system_a2a_simple.py --mode fallback --max-alerts 8",
        language="bash",
    )
    st.stop()


# =============================================================================
# 8. SIDEBAR / CENTRO DE CONTROL
# =============================================================================

# Fechas globales disponibles.
all_date_series = []
for df in [features, scored_alerts, recommendations]:
    if not df.empty and "Fecha" in df.columns:
        all_date_series.append(df["Fecha"].dropna())

if all_date_series:
    all_dates = pd.concat(all_date_series)
    min_date = all_dates.min().date()
    max_date = all_dates.max().date()
else:
    min_date = max_date = pd.Timestamp.today().date()

with st.sidebar:
    st.markdown("## 🎛️ Centro de Control")
    st.caption("Filtra el periodo, producto y CEDI antes de explorar el sistema.")

    page = st.radio(
        "Modo de trabajo",
        [
            "1) Estado ejecutivo",
            "2) Línea de tiempo",
            "3) Predicción y simulación",
            "4) Alertas priorizadas",
            "5) Chat con agente",
            "6) Modelo y arquitectura",
        ],
        index=1,
    )

    st.divider()
    selected_range = st.date_input(
        "Intervalo de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filtra histórico, predicciones y alertas priorizadas dentro del mismo periodo.",
    )
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date, end_date = min_date, max_date

    sku_options = ["Todos"] + sorted(features["SKU_ID"].dropna().unique().tolist())
    cedi_options = ["Todos"] + sorted(features["CEDI"].dropna().unique().tolist())
    selected_sku = st.selectbox("SKU", sku_options)
    selected_cedi = st.selectbox("CEDI", cedi_options)

    with st.expander("📘 Guía rápida", expanded=True):
        st.markdown(
            """
- **Línea de tiempo:** entiende todos los datos del periodo.
- **Predicción:** prueba un caso real y modifica variables.
- **Alertas:** revisa las decisiones más urgentes.
- **Chat:** pide una explicación ejecutiva.

Analogía: primero revisamos toda la despensa; después vemos qué producto puede faltar; finalmente decidimos si conviene mover producto desde otra ubicación.
            """
        )

    with st.expander("💬 Preguntas para probar"):
        st.markdown(
            """
- ¿Por qué solo hay 50 alertas priorizadas?
- ¿Qué pasó en este intervalo de fechas?
- ¿Cómo sé si me voy a quedar sin producto?
- ¿Por qué conviene transferir inventario?
- ¿Qué significa beneficio neto?
            """
        )


def filter_date(df: pd.DataFrame, date_col: str = "Fecha") -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df.copy()
    out = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)].copy()
    return out


def apply_sku_cedi_filters(df: pd.DataFrame, cedi_col: str = "CEDI") -> pd.DataFrame:
    out = df.copy()
    if selected_sku != "Todos" and "SKU_ID" in out.columns:
        out = out[out["SKU_ID"] == selected_sku]
    if selected_cedi != "Todos" and cedi_col in out.columns:
        out = out[out[cedi_col] == selected_cedi]
    return out


features_period = apply_sku_cedi_filters(filter_date(features), "CEDI")
scored_period = apply_sku_cedi_filters(filter_date(scored_alerts), "CEDI")
recommendations_period = apply_sku_cedi_filters(filter_date(recommendations), "CEDI_Destino")


# =============================================================================
# 9. HEADER
# =============================================================================

st.markdown(
    """
    <div class="hero">
      <div class="hero-title">📦 Herdez Smart-Supply</div>
      <div class="hero-subtitle">
        Centro de decisión para anticipar quiebres de stock. Primero muestra el contexto histórico completo,
        después calcula riesgo con ML, prioriza alertas por costo-beneficio y finalmente explica la decisión con un chat/agente.
      </div>
      <span class="pill pill-blue">Histórico completo</span>
      <span class="pill pill-green">Predicción en vivo</span>
      <span class="pill pill-amber">Costo-beneficio</span>
      <span class="pill pill-red">Chat explicativo</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 10. KPIs DE CAPAS DE DATOS
# =============================================================================

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros históricos", f"{len(features_period):,}", help="Datos usados para entender ventas, inventario y comportamiento operativo.")
k2.metric("Predicciones", f"{len(scored_period):,}", help="Registros con probabilidad de riesgo calculada por el modelo.")
k3.metric("Alertas priorizadas", f"{len(recommendations_period):,}", help="Casos críticos donde sí se evalúa costo-beneficio.")
loss_total = recommendations_period["Perdida_Esperada_Sin_Actuar"].sum() if not recommendations_period.empty else 0
benefit_total = recommendations_period["Beneficio_Neto"].sum() if not recommendations_period.empty else 0
k4.metric("Pérdida esperada", money(loss_total))
k5.metric("Beneficio neto", money(benefit_total))


# =============================================================================
# 11. PÁGINAS DEL DASHBOARD
# =============================================================================

if page.startswith("1"):
    st.markdown("### 1) Estado ejecutivo")
    st.markdown(
        """
        <div class="section-card">
          <div class="mini-title">Lectura ejecutiva</div>
          <div class="helper-text">
          Esta vista resume el periodo seleccionado. Los datos no se reducen a las alertas: primero se cuenta el histórico,
          luego las predicciones y finalmente las recomendaciones accionables. Esto reduce carga cognitiva: el usuario ve todo el contexto,
          pero actúa solo sobre lo prioritario.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if not recommendations_period.empty:
            tmp = (
                recommendations_period.groupby("CEDI_Destino", as_index=False)["Riesgo_Probabilidad"]
                .mean()
                .sort_values("Riesgo_Probabilidad", ascending=False)
            )
            fig = px.bar(tmp, x="CEDI_Destino", y="Riesgo_Probabilidad", title="Riesgo promedio en alertas por CEDI", color="Riesgo_Probabilidad", color_continuous_scale="Blues")
            fig.update_layout(yaxis_tickformat=".0%", height=390, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Lectura: identifica qué CEDI concentra más riesgo en las alertas priorizadas del periodo.")
        else:
            st.info("No hay alertas priorizadas para el periodo/filtros seleccionados.")

    with c2:
        if not recommendations_period.empty:
            tmp = (
                recommendations_period.groupby("SKU_ID", as_index=False)["Beneficio_Neto"]
                .sum()
                .sort_values("Beneficio_Neto", ascending=False)
            )
            fig = px.bar(tmp, x="SKU_ID", y="Beneficio_Neto", title="Beneficio neto potencial por SKU", color="Beneficio_Neto", color_continuous_scale="Greens")
            fig.update_layout(height=390, coloraxis_showscale=False, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Lectura: prioriza productos donde actuar podría evitar más pérdida económica.")
        else:
            st.info("No hay beneficio neto calculado para el periodo/filtros seleccionados.")

    st.markdown("#### Top recomendaciones del periodo")
    if not recommendations_period.empty:
        cols = [
            "Fecha", "SKU_ID", "CEDI_Destino", "Riesgo_Probabilidad", "Accion_Recomendada",
            "CEDI_Origen_Recomendado", "Unidades_A_Transferir", "Perdida_Esperada_Sin_Actuar", "Beneficio_Neto",
        ]
        st.dataframe(recommendations_period.sort_values("Beneficio_Neto", ascending=False)[cols], use_container_width=True, hide_index=True)
    else:
        st.warning("No hay recomendaciones con los filtros actuales.")


elif page.startswith("2"):
    st.markdown("### 2) Línea de tiempo del negocio")
    st.markdown(
        """
        <div class="section-card">
          <div class="mini-title">¿Por qué esta sección importa?</div>
          <div class="helper-text">
          Aquí usamos el histórico del periodo seleccionado, no solo las alertas. Sirve para ver si las ventas subieron,
          si el stock bajó y si el riesgo aumentó antes de decidir transferencias. Es la vista de contexto antes de la acción.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if features_period.empty:
        st.warning("No hay histórico con los filtros seleccionados.")
    else:
        daily = (
            features_period.groupby("Fecha", as_index=False)
            .agg(Ventas_Totales=("Ventas_Unidades", "sum"), Stock_Promedio=("Stock_Actual", "mean"))
            .sort_values("Fecha")
        )
        if not scored_period.empty and "Riesgo_Probabilidad" in scored_period.columns:
            daily_risk = scored_period.groupby("Fecha", as_index=False).agg(Riesgo_Promedio=("Riesgo_Probabilidad", "mean"))
            daily = daily.merge(daily_risk, on="Fecha", how="left")
        else:
            daily["Riesgo_Promedio"] = np.nan
        if not recommendations_period.empty:
            daily_alerts = recommendations_period.groupby("Fecha", as_index=False).agg(Alertas_Priorizadas=("SKU_ID", "count"), Beneficio_Neto=("Beneficio_Neto", "sum"))
            daily = daily.merge(daily_alerts, on="Fecha", how="left")
        else:
            daily["Alertas_Priorizadas"] = 0
            daily["Beneficio_Neto"] = 0
        daily[["Alertas_Priorizadas", "Beneficio_Neto"]] = daily[["Alertas_Priorizadas", "Beneficio_Neto"]].fillna(0)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(daily, x="Fecha", y="Ventas_Totales", markers=True, title="Ventas totales por día")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Lectura: si las ventas suben, el inventario se consume más rápido.")
        with c2:
            fig = px.line(daily, x="Fecha", y="Stock_Promedio", markers=True, title="Stock promedio por día")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Lectura: si el stock cae mientras las ventas suben, aumenta el riesgo operativo.")

        c3, c4 = st.columns(2)
        with c3:
            if daily["Riesgo_Promedio"].notna().any():
                fig = px.line(daily, x="Fecha", y="Riesgo_Promedio", markers=True, title="Riesgo promedio predicho por día")
                fig.update_layout(height=360, yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Lectura: muestra en qué días el modelo percibe mayor probabilidad de quiebre.")
            else:
                st.info("No hay riesgo predicho disponible en este periodo.")
        with c4:
            fig = px.bar(daily, x="Fecha", y="Alertas_Priorizadas", title="Alertas priorizadas por día")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Lectura: no todos los registros requieren acción; aquí se ven solo los casos priorizados.")

        with st.expander("Ver histórico filtrado"):
            st.dataframe(features_period.sort_values("Fecha"), use_container_width=True, hide_index=True)


elif page.startswith("3"):
    st.markdown("### 3) Predicción en vivo y simulación")
    st.info("Selecciona un caso real del dataset, modifica variables y observa cómo cambian riesgo, faltante y recomendación.")

    base_for_case = features_period.copy()
    if base_for_case.empty:
        base_for_case = features.copy()
        st.warning("No hay datos con el filtro actual; se muestran casos del histórico completo para permitir la simulación.")

    f1, f2, f3 = st.columns(3)
    with f1:
        sku_options_live = sorted(base_for_case["SKU_ID"].dropna().unique())
        sku_live = st.selectbox("SKU del caso", sku_options_live)
    case_sku = base_for_case[base_for_case["SKU_ID"] == sku_live]
    with f2:
        cedi_options_live = sorted(case_sku["CEDI"].dropna().unique())
        cedi_live = st.selectbox("CEDI del caso", cedi_options_live)
    case_df = case_sku[case_sku["CEDI"] == cedi_live].sort_values("Fecha")
    with f3:
        fecha_live = st.selectbox("Fecha del caso", case_df["Fecha"].dt.date.astype(str).tolist())

    base_row = case_df[case_df["Fecha"].dt.date.astype(str) == fecha_live].iloc[0]

    st.markdown(
        """
        <div class="section-card">
          <div class="mini-title">Variables editables</div>
          <div class="helper-text">
          Modifica stock, ventas, lead time o costos para responder preguntas tipo: ¿qué pasa si baja el inventario?,
          ¿si sube la demanda?, ¿si el proveedor tarda más? El modelo calcula riesgo y el motor decide si conviene actuar.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown(
        f"<div class='decision-box {risk_class}'><b>Recomendación:</b> {live_decision['Accion_Recomendada']}<br>{live_decision['Explicacion_Ejecutiva']}</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        inv_df = pd.DataFrame(
            {
                "Concepto": ["Stock actual", "Demanda segura LT", "Unidades faltantes"],
                "Unidades": [live_decision["Stock_Actual"], live_decision["Demanda_Estimada_LT_Segura"], live_decision["Unidades_Necesarias"]],
            }
        )
        fig = px.bar(inv_df, x="Concepto", y="Unidades", color="Concepto", title="Cobertura de inventario", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Si la demanda segura supera el stock, existe faltante operativo.")
    with c2:
        econ_df = pd.DataFrame(
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
        fig = px.bar(econ_df, x="Concepto", y="Monto", color="Concepto", title="Economía de la decisión", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Conviene actuar cuando la pérdida evitada supera el costo de transferencia.")

    with st.expander("Ver CEDIs origen evaluados"):
        scenarios = pd.DataFrame(live_decision.get("Escenarios", []))
        if scenarios.empty:
            st.warning("No hubo CEDI origen con excedente suficiente para este caso.")
        else:
            st.dataframe(scenarios, use_container_width=True, hide_index=True)


elif page.startswith("4"):
    st.markdown("### 4) Alertas priorizadas")
    st.markdown(
        """
        <div class="section-card">
          <div class="mini-title">¿Por qué aquí no aparecen todos los registros?</div>
          <div class="helper-text">
          Esta tabla muestra los casos que requieren atención. El histórico completo se analiza en la línea de tiempo;
          aquí solo se muestran las alertas donde el sistema evaluó costo-beneficio y recomendación operativa.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if recommendations_period.empty:
        st.warning("No hay alertas priorizadas con los filtros actuales.")
    else:
        st.dataframe(recommendations_period.sort_values(["Beneficio_Neto", "Riesgo_Probabilidad"], ascending=False), use_container_width=True, hide_index=True)


elif page.startswith("5"):
    st.markdown("### 5) Chat con agente")
    st.markdown(
        """
        <div class="section-card">
          <div class="mini-title">Chat ejecutivo</div>
          <div class="helper-text">
          Pregunta en lenguaje natural. El chat usa el intervalo de fechas, las recomendaciones filtradas,
          el caso vivo seleccionado y el contexto del modelo. Gemini es opcional; si no está activo, responde con lógica local.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    use_gemini = st.toggle("Usar Gemini si hay API key configurada", value=False)
    show_live_coordination = st.toggle("Mostrar coordinación de agentes mientras responde", value=True)
    examples = [
        "¿Por qué solo hay 50 alertas priorizadas?",
        "¿Qué significa beneficio neto?",
        "¿Cómo se ponen de acuerdo los agentes?",
        "Explícame el caso seleccionado como si fuera Director de Supply Chain.",
    ]
    b1, b2, b3, b4 = st.columns(4)
    for col, ex in zip([b1, b2, b3, b4], examples):
        if col.button(ex):
            st.session_state["pending_chat"] = ex

    render_agent_consensus(agent_artifacts, st.session_state.get("live_decision"))

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": "Hola. Puedo explicar el histórico, el filtro de fechas, las alertas priorizadas, el riesgo, costos, modelo, agentes y arquitectura."}
        ]

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Pregunta sobre histórico, riesgo, costos, modelo o recomendación…")
    if "pending_chat" in st.session_state:
        prompt = st.session_state.pop("pending_chat")

    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        live_decision = st.session_state.get("live_decision")
        context = build_context_summary(recommendations_period, features_period, scored_period, model_metrics, live_decision, agent_brief, agent_artifacts)

        with st.chat_message("assistant"):
            if show_live_coordination:
                render_live_agent_activity(prompt, live_decision, agent_artifacts)
                st.divider()

            answer = gemini_agent_answer(prompt, context) if use_gemini else None
            if not answer:
                answer = local_agent_answer(prompt, context, live_decision, agent_artifacts)

            st.markdown(answer)

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})


else:
    st.markdown("### 6) Modelo y arquitectura")

    st.markdown("#### Comparación de modelos")
    if not model_metrics.empty:
        st.dataframe(model_metrics, use_container_width=True, hide_index=True)
        if {"model", "precision_risk", "recall_risk", "f1_risk", "roc_auc"}.issubset(model_metrics.columns):
            chart = model_metrics.set_index("model")[["precision_risk", "recall_risk", "f1_risk", "roc_auc"]].reset_index()
            fig = px.bar(chart, x="model", y=["precision_risk", "recall_risk", "f1_risk", "roc_auc"], barmode="group", title="Métricas comparativas")
            fig.update_layout(height=390, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No encontré métricas de modelos.")

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
XGBoost
  ↓
Decision Engine determinista
  ↓
Agente / Chat explicativo
  ↓
Streamlit Dashboard
```
            """
        )
    with a2:
        st.markdown(
            """
#### Ruta cloud-ready
```text
ERP / WMS / POS
  ↓
Data warehouse analítico
  ↓
Pipeline de entrenamiento y monitoreo
  ↓
Servicio de predicción
  ↓
Tools deterministas de negocio
  ↓
Agentes interoperables
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
        st.info("No encontré registro de agentes.")

    st.markdown("#### Consenso entre agentes")
    render_agent_consensus(agent_artifacts, st.session_state.get("live_decision"))

    with st.expander("Ver artefactos A2A-lite en JSON"):
        if agent_artifacts:
            st.json(agent_artifacts)
        else:
            st.info("No encontré artefactos del agente.")
