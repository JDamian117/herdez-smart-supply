"""
04_agent_system_a2a.py

Herdez Smart-Supply - Fase 4 replanteada
Sistema multiagente estilo A2A-lite + prompts profesionales + salida estructurada.

Objetivo
--------
Replantear el agente para que la demo muestre arquitectura moderna de agentes sin
perder el enfoque local-first:

1) XGBoost y 03_decision_engine.py siguen siendo la fuente de verdad numérica.
2) Los agentes se comunican con mensajes y artefactos estilo A2A.
3) El LLM no recalcula costos ni inventa decisiones; solo analiza, critica y explica.
4) Las instrucciones siguen patrones profesionales: identidad, misión, metodología,
   límites y ejemplos.
5) La salida se valida con Pydantic para obtener JSON estable y auditable.
6) Si no hay API key o librerías, se genera un fallback determinista para demo segura.

Nota arquitectónica
-------------------
Este archivo NO implementa un servidor oficial A2A. Implementa un "A2A-lite local":
AgentCard, Task, Message, Artifact y un broker local en memoria/JSON. Esto demuestra
el patrón de arquitectura y deja una ruta clara para exponer cada agente como servicio
A2A real en producción.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field, ValidationError


# =============================================================================
# 1. Configuración general
# =============================================================================

class AgentMode(str, Enum):
    """Modos de ejecución.

    fallback: no usa LLM; produce artefactos deterministas.
    langchain: intenta usar LangChain + Gemini.
    auto: intenta LangChain + Gemini y cae a fallback si falla.
    """

    AUTO = "auto"
    LANGCHAIN = "langchain"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ModelProfile:
    """Perfil de configuración de modelo.

    Usamos temperaturas bajas porque este caso es de decisión empresarial:
    los agentes deben ser consistentes, no creativos con números.
    """

    provider: str = "gemini"
    model_name: str = "gemini-2.5-flash"
    temperature_factual: float = 0.2
    temperature_planning: float = 0.3
    temperature_executive: float = 0.4
    max_output_tokens: int = 2048


@dataclass(frozen=True)
class AgentSystemConfig:
    """Configuración del sistema de agentes."""

    output_language: str = "es"
    prompt_language: str = "en"  # instrucciones en inglés, respuesta en español
    max_alerts_context: int = 8
    recommendations_file: str = "outputs/decision_recommendations.csv"
    scenarios_file: str = "outputs/decision_transfer_scenarios.csv"
    output_dir: str = "outputs"
    mode: AgentMode = AgentMode.AUTO
    profile: ModelProfile = field(default_factory=ModelProfile)


# =============================================================================
# 2. Esquemas A2A-lite: AgentCard, Task, Message, Artifact
# =============================================================================

class AgentCard(BaseModel):
    """Tarjeta de capacidades del agente.

    Equivale al concepto de descubrimiento: qué sabe hacer este agente,
    qué entradas espera y qué artefactos produce.
    """

    name: str
    role: str
    description: str
    capabilities: List[str]
    input_artifacts: List[str] = Field(default_factory=list)
    output_artifacts: List[str] = Field(default_factory=list)
    deterministic: bool = False
    llm_enabled: bool = True


class A2AMessage(BaseModel):
    """Mensaje local entre agentes."""

    message_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    sender: str
    receiver: str
    content_type: Literal["application/json", "text/markdown"] = "application/json"
    content: Dict[str, Any]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class A2AArtifact(BaseModel):
    """Artefacto producido por un agente.

    Un artefacto es la evidencia auditable que se pasa al siguiente agente.
    """

    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    producer: str
    artifact_type: str
    content: Dict[str, Any]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class A2ATask(BaseModel):
    """Unidad de trabajo para el flujo multiagente."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    objective: str
    status: Literal["created", "running", "completed", "failed"] = "created"
    messages: List[A2AMessage] = Field(default_factory=list)
    artifacts: List[A2AArtifact] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_artifact(self, artifact: A2AArtifact) -> None:
        self.artifacts.append(artifact)

    def add_message(self, message: A2AMessage) -> None:
        self.messages.append(message)


# =============================================================================
# 3. Esquemas estructurados para outputs del LLM
# =============================================================================

class RiskAnalysisOutput(BaseModel):
    risk_summary: str = Field(description="Resumen de riesgo de quiebre en español")
    critical_patterns: List[str] = Field(description="Patrones críticos observados")
    priority_alerts: List[str] = Field(description="Alertas que requieren mayor atención")
    data_caveats: List[str] = Field(default_factory=list, description="Limitaciones de datos o supuestos")


class CostAnalysisOutput(BaseModel):
    financial_summary: str = Field(description="Resumen financiero de la recomendación")
    best_actions: List[str] = Field(description="Acciones con mejor beneficio neto")
    cost_risks: List[str] = Field(default_factory=list, description="Riesgos de costo o supuestos financieros")


class PolicyCriticOutput(BaseModel):
    approved_actions: List[str] = Field(default_factory=list, description="Acciones que cumplen política")
    blocked_or_risky_actions: List[str] = Field(default_factory=list, description="Acciones que requieren revisión")
    critic_notes: List[str] = Field(description="Observaciones críticas del flujo")
    human_review_required: bool = Field(description="Si se requiere revisión humana")


class ExecutiveBriefOutput(BaseModel):
    executive_summary: str = Field(description="Resumen ejecutivo para director de Supply Chain")
    recommended_plan: List[str] = Field(description="Plan recomendado en pasos")
    expected_business_impact: str = Field(description="Impacto en negocio")
    technical_trace_summary: str = Field(description="Resumen técnico breve")
    questions_to_prepare: List[str] = Field(description="Preguntas probables de entrevista y cómo responderlas")


# =============================================================================
# 4. Utilidades de datos
# =============================================================================

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


def clean_numeric(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def row_get(row: pd.Series, candidates: List[str], default: Any = "") -> Any:
    for col in candidates:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                return val
    return default


def load_decision_outputs(recommendations_file: str, scenarios_file: str) -> Dict[str, pd.DataFrame]:
    rec_path = Path(recommendations_file)
    if not rec_path.exists():
        raise FileNotFoundError(
            f"No encontré {rec_path}. Ejecuta primero: python src/03_decision_engine.py"
        )

    recommendations = pd.read_csv(rec_path)
    scenarios = pd.DataFrame()
    sc_path = Path(scenarios_file)
    if sc_path.exists():
        scenarios = pd.read_csv(sc_path)
    return {"recommendations": recommendations, "scenarios": scenarios}


def normalize_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score_candidates = [
        "net_benefit", "beneficio_neto", "Beneficio_Neto",
        "expected_net_benefit", "beneficio_estimado"
    ]
    score_col = next((c for c in score_candidates if c in df.columns), None)
    if score_col:
        df["_sort_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    elif "risk_probability" in df.columns:
        df["_sort_score"] = pd.to_numeric(df["risk_probability"], errors="coerce").fillna(0)
    else:
        df["_sort_score"] = 0
    return df.sort_values("_sort_score", ascending=False).reset_index(drop=True)


def build_context(recommendations: pd.DataFrame, scenarios: pd.DataFrame, max_alerts: int) -> Dict[str, Any]:
    recommendations = normalize_recommendations(recommendations)
    top = recommendations.head(max_alerts)

    alerts: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        risk = clean_numeric(row_get(row, ["risk_probability", "probabilidad_riesgo", "Riesgo_Probabilidad"], 0))
        expected_loss = clean_numeric(row_get(row, ["expected_loss", "perdida_esperada", "Perdida_Esperada"], 0))
        transfer_cost = clean_numeric(row_get(row, ["transfer_cost", "costo_transferencia", "Costo_Transferencia"], 0))
        net_benefit = clean_numeric(row_get(row, ["net_benefit", "beneficio_neto", "Beneficio_Neto"], 0))
        units = clean_numeric(row_get(row, ["units_to_transfer", "unidades_transferir", "Unidades_Transferir"], 0))

        alerts.append({
            "fecha": str(row_get(row, ["fecha", "Fecha"], "")),
            "sku": str(row_get(row, ["sku_id", "SKU_ID", "sku"], "")),
            "cedi_destino": str(row_get(row, ["cedi_destino", "CEDI_Destino", "CEDI"], "")),
            "cedi_origen": str(row_get(row, ["cedi_origen", "CEDI_Origen", "source_cedi"], "")),
            "risk_probability": risk,
            "risk_level": str(row_get(row, ["risk_level", "nivel_riesgo", "Nivel_Riesgo"], "")),
            "recommended_action": str(row_get(row, ["recommended_action", "accion_recomendada", "Accion_Recomendada"], "")),
            "units_to_transfer": units,
            "expected_loss": expected_loss,
            "transfer_cost": transfer_cost,
            "net_benefit": net_benefit,
            "policy_status": str(row_get(row, ["policy_status", "estado_politica", "Policy_Status"], "")),
            "business_explanation": str(row_get(row, ["business_explanation", "explicacion_negocio", "Business_Explanation"], "")),
        })

    summary = {
        "total_recommendations": int(len(recommendations)),
        "alerts_sent_to_agents": int(len(alerts)),
        "total_expected_loss_top": float(sum(a["expected_loss"] for a in alerts)),
        "total_transfer_cost_top": float(sum(a["transfer_cost"] for a in alerts)),
        "total_net_benefit_top": float(sum(a["net_benefit"] for a in alerts)),
        "transfer_actions_top": int(sum("transfer" in a["recommended_action"].lower() or "mover" in a["recommended_action"].lower() for a in alerts)),
        "urgent_review_actions_top": int(sum("urgent" in a["recommended_action"].lower() or "reabasto" in a["recommended_action"].lower() or "review" in a["recommended_action"].lower() for a in alerts)),
    }

    return {
        "case_name": "Herdez Smart-Supply",
        "business_goal": "Evitar quiebres de stock en CEDIs minimizando costos logísticos y ventas perdidas.",
        "architecture_principle": "ML predice riesgo; motor determinista calcula costo-beneficio; agentes razonan, critican y explican.",
        "summary": summary,
        "top_alerts": alerts,
        "scenario_rows_available": int(len(scenarios)),
    }


def context_to_markdown(context: Dict[str, Any]) -> str:
    s = context["summary"]
    lines = [
        "# Contexto operacional",
        f"Caso: {context['case_name']}",
        f"Objetivo de negocio: {context['business_goal']}",
        f"Principio: {context['architecture_principle']}",
        "",
        "## Resumen cuantitativo",
        f"- Alertas enviadas a agentes: {s['alerts_sent_to_agents']}",
        f"- Pérdida esperada agregada: {money(s['total_expected_loss_top'])}",
        f"- Costo logístico agregado: {money(s['total_transfer_cost_top'])}",
        f"- Beneficio neto agregado: {money(s['total_net_benefit_top'])}",
        "",
        "## Alertas prioritarias",
    ]
    for idx, a in enumerate(context["top_alerts"], start=1):
        lines.extend([
            f"### Alerta {idx}",
            f"- SKU: {a['sku']}",
            f"- CEDI destino: {a['cedi_destino']}",
            f"- CEDI origen recomendado: {a['cedi_origen'] or 'N/D'}",
            f"- Riesgo: {pct(a['risk_probability'])} ({a['risk_level']})",
            f"- Acción: {a['recommended_action']}",
            f"- Unidades: {a['units_to_transfer']:.0f}",
            f"- Pérdida esperada: {money(a['expected_loss'])}",
            f"- Costo transferencia: {money(a['transfer_cost'])}",
            f"- Beneficio neto: {money(a['net_benefit'])}",
            f"- Estado política: {a['policy_status']}",
        ])
    return "\n".join(lines)


# =============================================================================
# 5. Prompt engineering profesional
# =============================================================================

def build_instruction(agent_name: str, role: str, mission: str, methodology: List[str], limits: List[str], examples: List[str]) -> str:
    """Construye instrucciones siguiendo los 5 patrones: identidad, misión, metodología, límites y ejemplos."""
    methodology_md = "\n".join(f"{i+1}. **{step.split(':', 1)[0]}**: {step.split(':', 1)[1].strip() if ':' in step else step}" for i, step in enumerate(methodology))
    limits_md = "\n".join(f"- {x}" for x in limits)
    examples_md = "\n\n".join(examples)
    return f"""
# Identity
You are {agent_name}, acting as {role} with senior experience in AI-enabled supply chain decision systems.

# Mission
{mission}

# How you work
{methodology_md}

# Behavioral limits
{limits_md}

# Output rules
- Answer in Spanish.
- Use only the facts, alerts, costs, probabilities and policy results provided in the input context.
- Keep financial numbers exactly as provided or clearly mark derived qualitative interpretations.
- Do not expose hidden reasoning. Provide concise business reasoning and traceable evidence.
- If data is insufficient, say so and request human review.

# Examples
{examples_md}
""".strip()


def build_prompt_pack() -> Dict[str, str]:
    common_limits = [
        "Never invent SKU, CEDI, probability, cost, units or ROI values.",
        "Never override the deterministic decision engine with intuition.",
        "Never recommend a transfer if the policy artifact marks it as blocked.",
        "Always distinguish between measured data, model prediction and LLM interpretation.",
        "Always preserve auditability: mention which artifact supports the conclusion.",
    ]

    return {
        "risk_analyst": build_instruction(
            "RiskLens",
            "Inventory Risk Analyst",
            "Analyze stockout risk alerts and identify the most urgent SKU/CEDI patterns while preserving traceability.",
            [
                "Read: inspect the provided risk alerts only.",
                "Prioritize: identify the largest risk and business impact combinations.",
                "Caveat: flag data limitations and assumptions.",
                "Summarize: produce a concise risk interpretation for downstream agents.",
            ],
            common_limits,
            [
                "Input: high risk but low benefit. Output: 'Riesgo alto, pero la acción debe validarse financieramente antes de mover inventario.'",
                "Input: missing origin CEDI. Output: 'No hay CEDI origen viable en los datos; se recomienda revisión humana o reabasto urgente.'",
            ],
        ),
        "cost_analyst": build_instruction(
            "CostGuard",
            "Logistics Cost Analyst",
            "Explain the financial trade-off between expected stockout loss and transfer cost.",
            [
                "Compare: evaluate expected loss, transfer cost and net benefit.",
                "Rank: identify the actions with highest positive net benefit.",
                "Warn: highlight cost assumptions or cases where moving inventory does not pay off.",
            ],
            common_limits,
            [
                "Input: positive net benefit. Output: 'La transferencia es financieramente defendible porque la pérdida evitada supera el costo logístico.'",
                "Input: zero units to transfer. Output: 'No existe acción logística viable en este escenario; conviene escalar a reabasto.'",
            ],
        ),
        "policy_critic": build_instruction(
            "PolicyCritic",
            "Policy Validator and Critic Agent",
            "Audit the recommendation for policy, operational risk and hidden assumptions before executive communication.",
            [
                "Validate: check if recommendations respect policy status and source CEDI constraints.",
                "Critique: identify risks such as transferring the problem to another CEDI.",
                "Escalate: require human review when data is insufficient or blocked.",
            ],
            common_limits,
            [
                "Input: policy approved. Output: 'La acción puede comunicarse como recomendación operativa sujeta a ejecución.'",
                "Input: policy blocked. Output: 'No debe presentarse como recomendación automática; requiere revisión humana.'",
            ],
        ),
        "executive_communicator": build_instruction(
            "ExecSupplyAI",
            "Executive Supply Chain Communicator",
            "Convert audited agent artifacts into an executive brief for the Supply Chain Director and a technical trace for the AI Manager.",
            [
                "Synthesize: combine risk, financial and policy artifacts.",
                "Translate: explain in business language without losing traceability.",
                "Prepare: include likely interview questions and strong answers.",
            ],
            common_limits,
            [
                "Input: approved transfer with positive benefit. Output: 'Recomiendo transferir porque protege ventas y el beneficio neto supera el costo logístico.'",
                "Input: missing data. Output: 'El sistema recomienda revisión humana porque los datos no permiten una decisión automática auditable.'",
            ],
        ),
    }


# =============================================================================
# 6. Agentes locales A2A-lite
# =============================================================================

class LocalA2AAgent:
    """Agente local que produce artefactos estructurados.

    Si use_llm=True y hay dependencias/API key, intenta usar Gemini vía LangChain.
    Si falla, usa una salida determinista segura.
    """

    def __init__(self, card: AgentCard, instruction: str, output_schema: Any, temperature: float = 0.2):
        self.card = card
        self.instruction = instruction
        self.output_schema = output_schema
        self.temperature = temperature

    def run(self, task: A2ATask, input_payload: Dict[str, Any], use_llm: bool) -> A2AArtifact:
        if use_llm and self.card.llm_enabled:
            try:
                content = self._run_langchain_gemini(input_payload)
            except Exception as exc:
                content = self._fallback(input_payload, reason=f"LLM fallback: {type(exc).__name__}: {exc}")
        else:
            content = self._fallback(input_payload, reason="fallback deterministic mode")

        artifact = A2AArtifact(
            task_id=task.task_id,
            producer=self.card.name,
            artifact_type=self.card.output_artifacts[0] if self.card.output_artifacts else "generic_artifact",
            content=content,
        )
        task.add_artifact(artifact)
        return artifact

    def _run_langchain_gemini(self, input_payload: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("No GEMINI_API_KEY/GOOGLE_API_KEY found")

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as exc:
            raise RuntimeError("langchain_google_genai is not installed") from exc

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=self.temperature,
            max_output_tokens=2048,
            google_api_key=api_key,
        )

        schema_hint = self.output_schema.model_json_schema()
        user_prompt = f"""
You must return valid JSON matching this schema:
{json.dumps(schema_hint, ensure_ascii=False, indent=2)}

Input payload:
{json.dumps(input_payload, ensure_ascii=False, indent=2)}
""".strip()

        response = llm.invoke([
            ("system", self.instruction),
            ("user", user_prompt),
        ])

        raw = response.content if hasattr(response, "content") else str(response)
        parsed = parse_json_from_text(raw)
        validated = self.output_schema(**parsed)
        return validated.model_dump()

    def _fallback(self, input_payload: Dict[str, Any], reason: str) -> Dict[str, Any]:
        context = input_payload.get("context", {})
        artifacts = input_payload.get("artifacts", {})
        alerts = context.get("top_alerts", [])
        summary = context.get("summary", {})
        name = self.card.name

        if self.output_schema is RiskAnalysisOutput:
            critical = [
                f"{a['sku']} en {a['cedi_destino']} con riesgo {pct(a['risk_probability'])} y beneficio neto {money(a['net_benefit'])}"
                for a in alerts[:3]
            ]
            return RiskAnalysisOutput(
                risk_summary=f"Se analizaron {summary.get('alerts_sent_to_agents', 0)} alertas priorizadas. El riesgo se concentra en combinaciones SKU/CEDI con alto beneficio neto potencial.",
                critical_patterns=critical,
                priority_alerts=[f"Priorizar {a['sku']} en {a['cedi_destino']}" for a in alerts[:3]],
                data_caveats=[reason, "La interpretación usa únicamente recomendaciones calculadas por el decision engine."],
            ).model_dump()

        if self.output_schema is CostAnalysisOutput:
            best = [
                f"{a['recommended_action']} para {a['sku']} en {a['cedi_destino']}: beneficio neto {money(a['net_benefit'])}"
                for a in alerts[:3]
            ]
            return CostAnalysisOutput(
                financial_summary=f"El beneficio neto agregado de las alertas enviadas es {money(summary.get('total_net_benefit_top', 0))}; la pérdida esperada agregada es {money(summary.get('total_expected_loss_top', 0))}.",
                best_actions=best,
                cost_risks=["Validar capacidad real de transporte y disponibilidad operativa antes de ejecutar."],
            ).model_dump()

        if self.output_schema is PolicyCriticOutput:
            blocked = [
                f"{a['sku']} en {a['cedi_destino']} requiere revisión: acción={a['recommended_action']}, política={a['policy_status']}"
                for a in alerts if "urgent" in a["recommended_action"].lower() or "reabasto" in a["recommended_action"].lower() or "review" in a["recommended_action"].lower()
            ]
            approved = [
                f"{a['sku']} en {a['cedi_destino']} con origen {a['cedi_origen']}"
                for a in alerts if a["net_benefit"] > 0 and a["units_to_transfer"] > 0
            ]
            return PolicyCriticOutput(
                approved_actions=approved[:5],
                blocked_or_risky_actions=blocked[:5],
                critic_notes=[
                    "La decisión final debe conservar la regla: no transferir si el CEDI origen queda en riesgo.",
                    "Las acciones de reabasto urgente se comunican como escalamiento, no como transferencia automática.",
                    reason,
                ],
                human_review_required=bool(blocked),
            ).model_dump()

        if self.output_schema is ExecutiveBriefOutput:
            risk_art = artifacts.get("risk_analysis", {})
            cost_art = artifacts.get("cost_analysis", {})
            policy_art = artifacts.get("policy_critic", {})
            top_plan = [
                f"Ejecutar {a['recommended_action']} para {a['sku']} en {a['cedi_destino']} ({money(a['net_benefit'])} de beneficio neto)."
                for a in alerts[:3]
            ]
            return ExecutiveBriefOutput(
                executive_summary=(
                    f"El sistema Smart-Supply priorizó {summary.get('alerts_sent_to_agents', 0)} alertas. "
                    f"El beneficio neto potencial agregado es {money(summary.get('total_net_benefit_top', 0))}. "
                    "La recomendación se basa en riesgo ML, costo-beneficio y validación de política."
                ),
                recommended_plan=top_plan,
                expected_business_impact=(
                    "Reducir ventas perdidas por quiebre y evitar movimientos logísticos no rentables, "
                    "manteniendo trazabilidad de cada recomendación."
                ),
                technical_trace_summary=(
                    "XGBoost genera riesgo; 03_decision_engine calcula pérdida esperada, costo de transferencia y beneficio neto; "
                    "los agentes A2A-lite producen artefactos de riesgo, costo, política y comunicación ejecutiva."
                ),
                questions_to_prepare=[
                    "¿Por qué no dejar que el LLM decida? Porque los números críticos son deterministas y auditables.",
                    "¿Por qué A2A? Porque permite separar agentes por dominio y preparar interoperabilidad enterprise.",
                    "¿Cómo escala a GCP? BigQuery para datos, Vertex AI para modelos, Cloud Run/Agent Platform para agentes y A2A para interoperabilidad.",
                ],
            ).model_dump()

        return {"message": f"{name} completed in fallback", "reason": reason}


# =============================================================================
# 7. Orquestador A2A-lite
# =============================================================================

def create_agent_registry(prompt_pack: Dict[str, str]) -> Dict[str, LocalA2AAgent]:
    cards = {
        "risk_analyst": AgentCard(
            name="RiskLens",
            role="Inventory Risk Analyst",
            description="Analiza riesgo de quiebre usando alertas calculadas por ML y decision engine.",
            capabilities=["risk_interpretation", "sku_cedi_prioritization", "data_caveats"],
            input_artifacts=["decision_context"],
            output_artifacts=["risk_analysis"],
            deterministic=False,
            llm_enabled=True,
        ),
        "cost_analyst": AgentCard(
            name="CostGuard",
            role="Logistics Cost Analyst",
            description="Evalúa pérdida esperada, costo de transferencia y beneficio neto.",
            capabilities=["cost_benefit_explanation", "financial_ranking"],
            input_artifacts=["decision_context", "risk_analysis"],
            output_artifacts=["cost_analysis"],
            deterministic=False,
            llm_enabled=True,
        ),
        "policy_critic": AgentCard(
            name="PolicyCritic",
            role="Policy Validator and Critic",
            description="Critica recomendaciones contra reglas operativas y necesidad de revisión humana.",
            capabilities=["policy_validation", "risk_critique", "human_review_flag"],
            input_artifacts=["decision_context", "risk_analysis", "cost_analysis"],
            output_artifacts=["policy_critic"],
            deterministic=False,
            llm_enabled=True,
        ),
        "executive_communicator": AgentCard(
            name="ExecSupplyAI",
            role="Executive Communicator",
            description="Construye brief ejecutivo y trace técnico para entrevista.",
            capabilities=["executive_summary", "technical_trace", "interview_preparation"],
            input_artifacts=["decision_context", "risk_analysis", "cost_analysis", "policy_critic"],
            output_artifacts=["executive_brief"],
            deterministic=False,
            llm_enabled=True,
        ),
    }
    return {
        "risk_analyst": LocalA2AAgent(cards["risk_analyst"], prompt_pack["risk_analyst"], RiskAnalysisOutput, temperature=0.2),
        "cost_analyst": LocalA2AAgent(cards["cost_analyst"], prompt_pack["cost_analyst"], CostAnalysisOutput, temperature=0.2),
        "policy_critic": LocalA2AAgent(cards["policy_critic"], prompt_pack["policy_critic"], PolicyCriticOutput, temperature=0.2),
        "executive_communicator": LocalA2AAgent(cards["executive_communicator"], prompt_pack["executive_communicator"], ExecutiveBriefOutput, temperature=0.4),
    }


def run_a2a_lite_workflow(context: Dict[str, Any], config: AgentSystemConfig) -> Dict[str, Any]:
    """Ejecuta el flujo multiagente en orden controlado.

    Orden:
    1) RiskLens produce risk_analysis.
    2) CostGuard produce cost_analysis.
    3) PolicyCritic produce policy_critic.
    4) ExecSupplyAI produce executive_brief.
    """

    prompt_pack = build_prompt_pack()
    agents = create_agent_registry(prompt_pack)
    use_llm = config.mode in {AgentMode.AUTO, AgentMode.LANGCHAIN}

    task = A2ATask(
        title="Herdez Smart-Supply stockout response",
        objective="Generate an auditable, business-oriented recommendation based on deterministic decision outputs.",
        status="running",
    )

    artifacts_payload: Dict[str, Any] = {}

    risk_artifact = agents["risk_analyst"].run(task, {"context": context}, use_llm=use_llm)
    artifacts_payload["risk_analysis"] = risk_artifact.content

    cost_artifact = agents["cost_analyst"].run(task, {"context": context, "artifacts": artifacts_payload}, use_llm=use_llm)
    artifacts_payload["cost_analysis"] = cost_artifact.content

    policy_artifact = agents["policy_critic"].run(task, {"context": context, "artifacts": artifacts_payload}, use_llm=use_llm)
    artifacts_payload["policy_critic"] = policy_artifact.content

    executive_artifact = agents["executive_communicator"].run(task, {"context": context, "artifacts": artifacts_payload}, use_llm=use_llm)
    artifacts_payload["executive_brief"] = executive_artifact.content

    task.status = "completed"

    registry = {key: agent.card.model_dump() for key, agent in agents.items()}
    return {
        "task": task.model_dump(),
        "agent_registry": registry,
        "prompt_pack": prompt_pack,
        "context": context,
        "artifacts": artifacts_payload,
        "mode_used": config.mode.value,
        "llm_attempted": use_llm,
    }


# =============================================================================
# 8. Reportes
# =============================================================================

def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Extrae JSON de respuestas que pueden venir con ```json ... ```.

    Evita que una envoltura markdown rompa la validación estructurada.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def render_markdown_report(result: Dict[str, Any]) -> str:
    context = result["context"]
    artifacts = result["artifacts"]
    exec_brief = artifacts.get("executive_brief", {})
    risk = artifacts.get("risk_analysis", {})
    cost = artifacts.get("cost_analysis", {})
    policy = artifacts.get("policy_critic", {})

    lines = [
        "# Herdez Smart-Supply — Brief Ejecutivo A2A-lite",
        "",
        "## Resumen ejecutivo",
        exec_brief.get("executive_summary", "No disponible."),
        "",
        "## Plan recomendado",
    ]
    for item in exec_brief.get("recommended_plan", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Impacto esperado",
        exec_brief.get("expected_business_impact", "No disponible."),
        "",
        "## Lectura de riesgo",
        risk.get("risk_summary", "No disponible."),
        "",
        "### Patrones críticos",
    ])
    for item in risk.get("critical_patterns", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Lectura financiera",
        cost.get("financial_summary", "No disponible."),
        "",
        "### Mejores acciones financieras",
    ])
    for item in cost.get("best_actions", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Validación y crítica",
        f"Revisión humana requerida: {'Sí' if policy.get('human_review_required') else 'No'}",
        "",
        "### Notas críticas",
    ])
    for item in policy.get("critic_notes", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Traza técnica",
        exec_brief.get("technical_trace_summary", "No disponible."),
        "",
        "## Preguntas probables de entrevista",
    ])
    for item in exec_brief.get("questions_to_prepare", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Arquitectura A2A-lite aplicada",
        "- AgentCards describen capacidades y artefactos esperados por agente.",
        "- A2ATask agrupa la unidad de trabajo completa.",
        "- Cada agente produce un Artifact estructurado y auditable.",
        "- En producción, cada agente puede exponerse como servicio A2A real.",
    ])

    return "\n".join(lines)


def save_outputs(result: Dict[str, Any], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "agent_a2a_task_trace.json").write_text(json.dumps(result["task"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "agent_a2a_registry.json").write_text(json.dumps(result["agent_registry"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "agent_a2a_prompt_pack.json").write_text(json.dumps(result["prompt_pack"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "agent_a2a_context.json").write_text(json.dumps(result["context"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "agent_a2a_artifacts.json").write_text(json.dumps(result["artifacts"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "agent_executive_brief_a2a.md").write_text(render_markdown_report(result), encoding="utf-8")


# =============================================================================
# 9. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Herdez Smart-Supply 04 - A2A-lite agent system")
    parser.add_argument("--mode", choices=[m.value for m in AgentMode], default="auto")
    parser.add_argument("--recommendations", default="outputs/decision_recommendations.csv")
    parser.add_argument("--scenarios", default="outputs/decision_transfer_scenarios.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-alerts", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AgentSystemConfig(
        mode=AgentMode(args.mode),
        recommendations_file=args.recommendations,
        scenarios_file=args.scenarios,
        output_dir=args.output_dir,
        max_alerts_context=args.max_alerts,
    )

    data = load_decision_outputs(config.recommendations_file, config.scenarios_file)
    context = build_context(data["recommendations"], data["scenarios"], config.max_alerts_context)
    result = run_a2a_lite_workflow(context, config)
    save_outputs(result, config.output_dir)

    print("\n✅ 04_agent_system_a2a.py ejecutado correctamente")
    print(f"Modo solicitado: {config.mode.value}")
    print(f"Brief ejecutivo: {Path(config.output_dir) / 'agent_executive_brief_a2a.md'}")
    print(f"Traza A2A-lite: {Path(config.output_dir) / 'agent_a2a_task_trace.json'}")
    print(f"Registry de agentes: {Path(config.output_dir) / 'agent_a2a_registry.json'}")


if __name__ == "__main__":
    main()
