"""
04_agent_system_a2a_simple.py

Herdez Smart-Supply - Sistema de agentes A2A-lite simplificado.

Idea principal
--------------
Este archivo toma las recomendaciones calculadas por 03_decision_engine.py y las
convierte en una explicación ejecutiva usando un flujo de agentes simple.

La arquitectura respeta tres reglas:
1) XGBoost predice riesgo, pero NO toma la decisión final.
2) El Decision Engine calcula costos, beneficio neto y política de inventario.
3) Los agentes solo interpretan, critican y comunican; no inventan números.

Por qué es A2A-lite
-------------------
No levantamos servidores A2A reales para no complicar la demo. En su lugar,
simulamos el patrón con:
- AgentCard: describe qué hace cada agente.
- Artifact: resultado estructurado que un agente entrega al siguiente.
- Task: traza completa del flujo.

En producción, cada agente podría exponerse como un servicio A2A real.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field, ValidationError


# =============================================================================
# 1. Configuración simple
# =============================================================================

@dataclass
class Config:
    """Parámetros mínimos para ejecutar el sistema."""

    recommendations_path: str = "outputs/decision_recommendations.csv"
    output_dir: str = "outputs"
    max_alerts: int = 8
    mode: str = "fallback"  # fallback | llm | auto
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.2  # baja porque estamos explicando decisiones de negocio


# =============================================================================
# 2. Contratos A2A-lite
# =============================================================================

@dataclass
class AgentCard:
    """Tarjeta del agente: quién es y qué artefacto produce."""

    name: str
    role: str
    goal: str
    input_artifacts: List[str]
    output_artifact: str


class AgentArtifact(BaseModel):
    """Salida estructurada común para todos los agentes.

    Usamos un solo esquema para que el código sea fácil de explicar.
    """

    agent_name: str = Field(description="Nombre del agente que produce el artefacto")
    summary: str = Field(description="Resumen claro en español")
    key_points: List[str] = Field(description="Puntos principales sustentados en datos")
    risks_or_caveats: List[str] = Field(default_factory=list, description="Riesgos, límites o supuestos")
    next_action: str = Field(description="Siguiente acción recomendada")


@dataclass
class TaskTrace:
    """Traza completa de la tarea multiagente."""

    task_id: str
    title: str
    artifacts: Dict[str, Dict[str, Any]]


# =============================================================================
# 3. Utilidades de formato y carga de datos
# =============================================================================

def money(value: Any) -> str:
    """Formatea dinero de forma segura."""
    try:
        return f"${float(value):,.2f} MXN"
    except Exception:
        return "$0.00 MXN"


def pct(value: Any) -> str:
    """Formatea probabilidades."""
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "0.0%"


def get_first(row: pd.Series, columns: List[str], default: Any = "") -> Any:
    """Busca el primer nombre de columna disponible.

    Esto hace al código más robusto si cambian nombres entre versiones.
    """
    for col in columns:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return default


def load_top_alerts(path: str, max_alerts: int) -> pd.DataFrame:
    """Carga recomendaciones del Decision Engine y selecciona las más importantes."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"No encontré {file_path}. Ejecuta primero: python src/03_decision_engine.py"
        )

    df = pd.read_csv(file_path)

    # Priorizamos por beneficio neto si existe; si no, por probabilidad de riesgo.
    if "net_benefit" in df.columns:
        sort_col = "net_benefit"
    elif "risk_probability" in df.columns:
        sort_col = "risk_probability"
    else:
        sort_col = df.columns[0]

    return df.sort_values(sort_col, ascending=False).head(max_alerts).reset_index(drop=True)


def build_context(alerts_df: pd.DataFrame) -> Dict[str, Any]:
    """Convierte la tabla en un contexto compacto para los agentes."""
    alerts: List[Dict[str, Any]] = []

    for _, row in alerts_df.iterrows():
        alert = {
            "sku": str(get_first(row, ["sku", "SKU_ID", "sku_id"])),
            "cedi_destino": str(get_first(row, ["cedi_destino", "CEDI", "CEDI_Destino"])),
            "cedi_origen": str(get_first(row, ["cedi_origen", "source_cedi", "CEDI_Origen"], "N/D")),
            "risk_probability": float(get_first(row, ["risk_probability", "probabilidad_riesgo"], 0)),
            "recommended_action": str(get_first(row, ["recommended_action", "accion_recomendada"], "N/D")),
            "units_to_transfer": float(get_first(row, ["units_to_transfer", "unidades_transferir"], 0)),
            "expected_loss": float(get_first(row, ["expected_loss", "perdida_esperada"], 0)),
            "transfer_cost": float(get_first(row, ["transfer_cost", "costo_transferencia"], 0)),
            "net_benefit": float(get_first(row, ["net_benefit", "beneficio_neto"], 0)),
            "policy_status": str(get_first(row, ["policy_status", "estado_politica"], "N/D")),
        }
        alerts.append(alert)

    return {
        "case": "Herdez Smart-Supply",
        "principle": (
            "XGBoost predice riesgo; el motor determinista calcula costo-beneficio; "
            "los agentes interpretan, critican y comunican."
        ),
        "summary": {
            "alerts_count": len(alerts),
            "total_expected_loss": sum(a["expected_loss"] for a in alerts),
            "total_transfer_cost": sum(a["transfer_cost"] for a in alerts),
            "total_net_benefit": sum(a["net_benefit"] for a in alerts),
        },
        "alerts": alerts,
    }


def context_as_markdown(context: Dict[str, Any]) -> str:
    """Convierte el contexto a Markdown para que el LLM lo lea mejor."""
    s = context["summary"]
    lines = [
        "# Contexto del caso",
        f"Caso: {context['case']}",
        f"Principio arquitectónico: {context['principle']}",
        "",
        "## Resumen cuantitativo",
        f"- Alertas analizadas: {s['alerts_count']}",
        f"- Pérdida esperada agregada: {money(s['total_expected_loss'])}",
        f"- Costo de transferencia agregado: {money(s['total_transfer_cost'])}",
        f"- Beneficio neto agregado: {money(s['total_net_benefit'])}",
        "",
        "## Alertas prioritarias",
    ]

    for i, a in enumerate(context["alerts"], start=1):
        lines.extend([
            f"### Alerta {i}",
            f"- SKU: {a['sku']}",
            f"- CEDI destino: {a['cedi_destino']}",
            f"- CEDI origen: {a['cedi_origen']}",
            f"- Riesgo: {pct(a['risk_probability'])}",
            f"- Acción recomendada por motor: {a['recommended_action']}",
            f"- Unidades a transferir: {a['units_to_transfer']:.0f}",
            f"- Pérdida esperada: {money(a['expected_loss'])}",
            f"- Costo de transferencia: {money(a['transfer_cost'])}",
            f"- Beneficio neto: {money(a['net_benefit'])}",
            f"- Política: {a['policy_status']}",
            "",
        ])

    return "\n".join(lines)


# =============================================================================
# 4. Prompts profesionales, pero simples
# =============================================================================

def build_instruction(card: AgentCard) -> str:
    """Crea la instrucción con los 5 patrones: identidad, misión, metodología, límites y ejemplo."""
    return f"""
# Identity
You are {card.name}, acting as {card.role} for an AI-powered supply chain system.

# Mission
{card.goal}

# Methodology
1. Read only the context and artifacts provided.
2. Identify the most important facts for your role.
3. Explain your conclusion in Spanish.
4. Produce valid JSON that matches the required schema.

# Limits
- Never invent SKU, CEDI, probabilities, costs, units or ROI.
- Never override the deterministic decision engine.
- Always separate model prediction, deterministic calculation and LLM interpretation.
- If the available data is not enough, say that human review is required.
- Keep the answer clear enough for a business user.

# Example
Input: an alert says risk is 95%, net benefit is positive and policy is approved.
Output: explain that the action is financially and operationally defensible, without changing the numbers.
""".strip()


def agent_cards() -> List[AgentCard]:
    """Define los agentes del flujo.

    Son pocos a propósito: suficientes para mostrar arquitectura, sin volver el código pesado.
    """
    return [
        AgentCard(
            name="RiskLens",
            role="Inventory Risk Analyst",
            goal="Interpret stockout risk patterns by SKU and CEDI using the decision engine outputs.",
            input_artifacts=["decision_context"],
            output_artifact="risk_analysis",
        ),
        AgentCard(
            name="CostGuard",
            role="Logistics Cost Analyst",
            goal="Explain whether the expected avoided loss justifies the logistics transfer cost.",
            input_artifacts=["decision_context", "risk_analysis"],
            output_artifact="cost_analysis",
        ),
        AgentCard(
            name="PolicyCritic",
            role="Policy Validator",
            goal="Check operational caveats and decide if the recommendation needs human review.",
            input_artifacts=["decision_context", "risk_analysis", "cost_analysis"],
            output_artifact="policy_review",
        ),
        AgentCard(
            name="ExecSupplyAI",
            role="Executive Communicator",
            goal="Create a short executive brief and technical trace for the interview demo.",
            input_artifacts=["decision_context", "risk_analysis", "cost_analysis", "policy_review"],
            output_artifact="executive_brief",
        ),
    ]


# =============================================================================
# 5. Ejecución del agente: LLM opcional + fallback determinista
# =============================================================================

def parse_json(text: str) -> Dict[str, Any]:
    """Extrae JSON aunque el modelo lo envuelva en bloques ```json."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def call_gemini(card: AgentCard, context: Dict[str, Any], artifacts: Dict[str, Any], config: Config) -> AgentArtifact:
    """Llama Gemini vía LangChain y valida la respuesta con Pydantic."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("No hay GEMINI_API_KEY ni GOOGLE_API_KEY")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as exc:
        raise RuntimeError("Falta instalar langchain-google-genai") from exc

    llm = ChatGoogleGenerativeAI(
        model=config.model_name,
        temperature=config.temperature,
        google_api_key=api_key,
    )

    schema = AgentArtifact.model_json_schema()
    prompt = f"""
Return ONLY valid JSON matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Context:
{context_as_markdown(context)}

Previous artifacts:
{json.dumps(artifacts, ensure_ascii=False, indent=2)}
""".strip()

    response = llm.invoke([
        ("system", build_instruction(card)),
        ("user", prompt),
    ])

    raw_text = response.content if hasattr(response, "content") else str(response)
    parsed = parse_json(raw_text)
    return AgentArtifact(**parsed)


def fallback_agent(card: AgentCard, context: Dict[str, Any], artifacts: Dict[str, Any], reason: str) -> AgentArtifact:
    """Respuesta determinista para demo segura.

    Esto garantiza que el proyecto funcione aunque falle la API del LLM.
    """
    alerts = context["alerts"]
    summary = context["summary"]
    top = alerts[:3]

    if card.name == "RiskLens":
        return AgentArtifact(
            agent_name=card.name,
            summary=f"Se revisaron {summary['alerts_count']} alertas priorizadas de riesgo de quiebre.",
            key_points=[
                f"{a['sku']} en {a['cedi_destino']} tiene riesgo {pct(a['risk_probability'])} y beneficio neto {money(a['net_benefit'])}."
                for a in top
            ],
            risks_or_caveats=["El riesgo proviene del modelo ML y de los datos históricos disponibles.", reason],
            next_action="Priorizar las alertas con mayor beneficio neto y mayor riesgo operativo.",
        )

    if card.name == "CostGuard":
        return AgentArtifact(
            agent_name=card.name,
            summary=f"El beneficio neto agregado estimado es {money(summary['total_net_benefit'])}.",
            key_points=[
                f"{a['recommended_action']} para {a['sku']} en {a['cedi_destino']}: costo {money(a['transfer_cost'])}, pérdida esperada {money(a['expected_loss'])}."
                for a in top
            ],
            risks_or_caveats=["Validar capacidad real de transporte y disponibilidad operativa antes de ejecutar."],
            next_action="Ejecutar solo transferencias con beneficio neto positivo y política aprobada.",
        )

    if card.name == "PolicyCritic":
        blocked = [a for a in alerts if "urgent" in a["recommended_action"].lower() or "review" in a["recommended_action"].lower() or "reabasto" in a["recommended_action"].lower()]
        return AgentArtifact(
            agent_name=card.name,
            summary="Las recomendaciones deben ejecutarse solo si no trasladan el riesgo al CEDI origen.",
            key_points=[
                "La política principal es no dejar al CEDI origen por debajo de su cobertura mínima.",
                f"Casos que requieren revisión humana: {len(blocked)}.",
            ],
            risks_or_caveats=[
                "Una transferencia mal validada puede mover el problema de un CEDI a otro.",
                "Las recomendaciones de reabasto urgente no son transferencias automáticas.",
            ],
            next_action="Aprobar transferencias viables y escalar a humano los casos sin origen suficiente.",
        )

    # Executive communicator
    return AgentArtifact(
        agent_name=card.name,
        summary=(
            f"Smart-Supply priorizó {summary['alerts_count']} alertas con beneficio neto potencial de "
            f"{money(summary['total_net_benefit'])}."
        ),
        key_points=[
            "El modelo XGBoost detecta riesgo de quiebre.",
            "El motor determinista calcula pérdida esperada, costo de transferencia y beneficio neto.",
            "Los agentes A2A-lite interpretan, validan y explican la recomendación.",
        ],
        risks_or_caveats=["El LLM no modifica números ni reemplaza las reglas de negocio."],
        next_action="Mostrar el brief en Streamlit y permitir preguntas del Director de Supply Chain.",
    )


def run_agent(card: AgentCard, context: Dict[str, Any], artifacts: Dict[str, Any], config: Config) -> AgentArtifact:
    """Ejecuta un agente con LLM si está disponible; si falla, usa fallback."""
    if config.mode == "fallback":
        return fallback_agent(card, context, artifacts, reason="Modo fallback sin LLM.")

    try:
        return call_gemini(card, context, artifacts, config)
    except Exception as exc:
        if config.mode == "llm":
            raise
        return fallback_agent(card, context, artifacts, reason=f"Fallback por error LLM: {type(exc).__name__}")


# =============================================================================
# 6. Orquestador A2A-lite simple
# =============================================================================

def run_a2a_workflow(context: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """Corre los agentes en orden y guarda cada salida como artefacto."""
    artifacts: Dict[str, Any] = {}

    for card in agent_cards():
        artifact = run_agent(card, context, artifacts, config)
        artifacts[card.output_artifact] = artifact.model_dump()

    trace = TaskTrace(
        task_id=str(uuid4()),
        title="Herdez Smart-Supply A2A-lite decision explanation",
        artifacts=artifacts,
    )

    registry = [asdict(card) for card in agent_cards()]

    return {
        "registry": registry,
        "context": context,
        "trace": asdict(trace),
        "artifacts": artifacts,
    }


# =============================================================================
# 7. Reporte final
# =============================================================================

def render_report(result: Dict[str, Any]) -> str:
    """Convierte los artefactos en un reporte Markdown legible."""
    artifacts = result["artifacts"]
    exec_art = artifacts["executive_brief"]

    lines = [
        "# Herdez Smart-Supply — Brief Ejecutivo A2A-lite Simplificado",
        "",
        "## Resumen ejecutivo",
        exec_art["summary"],
        "",
        "## Plan recomendado",
    ]

    for point in exec_art["key_points"]:
        lines.append(f"- {point}")

    lines.extend([
        "",
        "## Lectura por agente",
    ])

    for artifact_name, art in artifacts.items():
        lines.extend([
            f"### {art['agent_name']} — {artifact_name}",
            art["summary"],
            "",
            "Puntos clave:",
        ])
        for point in art["key_points"]:
            lines.append(f"- {point}")
        if art.get("risks_or_caveats"):
            lines.append("Riesgos o límites:")
            for risk in art["risks_or_caveats"]:
                lines.append(f"- {risk}")
        lines.extend(["Siguiente acción:", f"- {art['next_action']}", ""])

    lines.extend([
        "## Cómo explicarlo en entrevista",
        "El prototipo usa una arquitectura local-first y A2A-lite: cada agente produce un artefacto estructurado que se entrega al siguiente. En producción, estos agentes podrían exponerse como servicios A2A reales.",
        "",
        "La decisión crítica no depende del LLM: XGBoost predice riesgo, el motor determinista calcula costo-beneficio y los agentes explican la recomendación de forma auditable.",
    ])

    return "\n".join(lines)


def save_outputs(result: Dict[str, Any], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "agent_a2a_simple_registry.json").write_text(json.dumps(result["registry"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "agent_a2a_simple_context.json").write_text(json.dumps(result["context"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "agent_a2a_simple_trace.json").write_text(json.dumps(result["trace"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "agent_a2a_simple_artifacts.json").write_text(json.dumps(result["artifacts"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "agent_executive_brief_a2a_simple.md").write_text(render_report(result), encoding="utf-8")


# =============================================================================
# 8. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Herdez Smart-Supply 04 simple A2A-lite agent system")
    parser.add_argument("--recommendations", default="outputs/decision_recommendations.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-alerts", type=int, default=8)
    parser.add_argument("--mode", choices=["fallback", "llm", "auto"], default="fallback")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        recommendations_path=args.recommendations,
        output_dir=args.output_dir,
        max_alerts=args.max_alerts,
        mode=args.mode,
    )

    alerts_df = load_top_alerts(config.recommendations_path, config.max_alerts)
    context = build_context(alerts_df)
    result = run_a2a_workflow(context, config)
    save_outputs(result, config.output_dir)

    print("\n✅ 04_agent_system_a2a_simple.py ejecutado correctamente")
    print(f"Modo: {config.mode}")
    print(f"Brief: {Path(config.output_dir) / 'agent_executive_brief_a2a_simple.md'}")
    print(f"Artefactos: {Path(config.output_dir) / 'agent_a2a_simple_artifacts.json'}")


if __name__ == "__main__":
    main()
