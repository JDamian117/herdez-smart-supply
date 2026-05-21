"""
04_agent_system.py

Herdez Smart-Supply - Fase 4
Sistema multiagente para explicar y auditar recomendaciones de Supply Chain.

Objetivo del archivo
--------------------
Conectar el motor determinista de decisión (Fase 3) con una capa GenAI.

Idea central de arquitectura:
    - XGBoost predice riesgo de quiebre.
    - El Decision Engine calcula costo-beneficio y aplica reglas.
    - CrewAI coordina agentes especializados.
    - Gemini genera razonamiento, crítica y explicación ejecutiva.
    - LangChain se usa como capa opcional de abstracción/fallback de LLM.

Diseño importante
-----------------
Este archivo está hecho para demo segura:
    1) Si hay GEMINI_API_KEY o GOOGLE_API_KEY, puede ejecutar CrewAI con Gemini.
    2) Si no hay API key o faltan librerías, genera un reporte determinista de fallback.

Esto permite presentar el flujo aunque falle internet o no haya credenciales.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# -----------------------------------------------------------------------------
# 1. Configuración general del agente
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentSystemConfig:
    """
    Configuración del sistema multiagente.

    En una solución empresarial, estos parámetros podrían venir de variables de
    entorno, Streamlit Secrets o Secret Manager en GCP.
    """

    model_name: str = "gemini/gemini-2.5-flash"
    language: str = "es"
    max_rows_context: int = 8
    output_dir: str = "outputs"
    recommendations_file: str = "outputs/decision_recommendations.csv"
    scenarios_file: str = "outputs/decision_transfer_scenarios.csv"
    force_fallback: bool = False


# -----------------------------------------------------------------------------
# 2. Utilidades de carga y preparación de contexto
# -----------------------------------------------------------------------------

def money(value: Any) -> str:
    """Formatea cantidades monetarias de manera legible."""
    try:
        return f"${float(value):,.2f} MXN"
    except Exception:
        return "$0.00 MXN"


def pct(value: Any) -> str:
    """Formatea probabilidades o proporciones."""
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "0.0%"


def load_decision_outputs(recommendations_file: str, scenarios_file: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Carga las recomendaciones generadas por 03_decision_engine.py.

    Por qué hacemos esto:
    - El agente no debe recalcular la decisión financiera.
    - El agente debe razonar sobre evidencia ya calculada y auditable.
    - Así evitamos alucinaciones numéricas.
    """
    rec_path = Path(recommendations_file)
    if not rec_path.exists():
        raise FileNotFoundError(
            f"No encontré {rec_path}. Primero ejecuta 03_decision_engine.py para generar decision_recommendations.csv"
        )

    recommendations = pd.read_csv(rec_path)

    scenarios = pd.DataFrame()
    if scenarios_file:
        sc_path = Path(scenarios_file)
        if sc_path.exists():
            scenarios = pd.read_csv(sc_path)

    return {"recommendations": recommendations, "scenarios": scenarios}


def normalize_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres esperados y ordena las alertas por impacto.

    Por qué lo hacemos:
    - CrewAI/Gemini necesitan un contexto corto, limpio y priorizado.
    - En una demo conviene pasar al agente las alertas más relevantes, no todo el CSV.
    """
    df = df.copy()

    # Detecta columnas numéricas clave aunque cambien ligeramente en versiones futuras.
    possible_score_cols = [
        "net_benefit", "beneficio_neto", "Net_Benefit", "Beneficio_Neto",
        "expected_net_benefit", "beneficio_estimado"
    ]
    score_col = next((c for c in possible_score_cols if c in df.columns), None)

    if score_col:
        df["_sort_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    elif "risk_probability" in df.columns:
        df["_sort_score"] = pd.to_numeric(df["risk_probability"], errors="coerce").fillna(0)
    else:
        df["_sort_score"] = 0

    return df.sort_values("_sort_score", ascending=False).reset_index(drop=True)


def row_get(row: pd.Series, candidates: List[str], default: Any = "") -> Any:
    """Obtiene el primer valor disponible de una lista de nombres de columna posibles."""
    for col in candidates:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                return val
    return default


def build_agent_context(recommendations: pd.DataFrame, scenarios: pd.DataFrame, max_rows: int = 8) -> Dict[str, Any]:
    """
    Construye un contexto compacto para agentes.

    Este objeto es lo que después recibe CrewAI/Gemini.
    La regla es: menos contexto, más relevante, más controlado.
    """
    recommendations = normalize_recommendations(recommendations)
    top = recommendations.head(max_rows).copy()

    alerts: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        risk_value = row_get(row, ["risk_probability", "probabilidad_riesgo", "Riesgo_Quiebre_Prob", "Riesgo_Probabilidad"], 0)
        net_benefit = row_get(row, ["net_benefit", "beneficio_neto", "Beneficio_Neto"], 0)
        expected_loss = row_get(row, ["expected_loss", "perdida_esperada", "Perdida_Esperada", "Perdida_Esperada_Sin_Actuar"], 0)
        transfer_cost = row_get(row, ["transfer_cost", "costo_transferencia", "Costo_Transferencia"], 0)
        units = row_get(row, ["units_to_transfer", "unidades_transferir", "Unidades_Transferir", "Unidades_A_Transferir"], 0)

        alerts.append({
            "fecha": str(row_get(row, ["fecha", "Fecha"], "")),
            "sku": str(row_get(row, ["sku_id", "SKU_ID", "sku"], "")),
            "cedi_destino": str(row_get(row, ["cedi_destino", "CEDI_Destino", "CEDI"], "")),
            "cedi_origen": str(row_get(row, ["cedi_origen", "CEDI_Origen", "CEDI_Origen_Recomendado", "source_cedi"], "")),
            "risk_probability": float(pd.to_numeric(risk_value, errors="coerce") if pd.notna(risk_value) else 0),
            "risk_level": str(row_get(row, ["risk_level", "nivel_riesgo", "Nivel_Riesgo"], "")),
            "recommended_action": str(row_get(row, ["recommended_action", "accion_recomendada", "Accion_Recomendada"], "")),
            "units_to_transfer": float(pd.to_numeric(units, errors="coerce") if pd.notna(units) else 0),
            "expected_loss": float(pd.to_numeric(expected_loss, errors="coerce") if pd.notna(expected_loss) else 0),
            "transfer_cost": float(pd.to_numeric(transfer_cost, errors="coerce") if pd.notna(transfer_cost) else 0),
            "net_benefit": float(pd.to_numeric(net_benefit, errors="coerce") if pd.notna(net_benefit) else 0),
            "policy_status": str(row_get(row, ["policy_status", "estado_politica", "Policy_Status"], "")),
            "explanation": str(row_get(row, ["business_explanation", "explicacion_negocio", "Business_Explanation", "Explicacion_Ejecutiva"], "")),
        })

    summary = {
        "total_recommendations": int(len(recommendations)),
        "shown_alerts": int(len(alerts)),
        "transfer_count": int(recommendations.astype(str).apply(lambda s: s.str.contains("transfer", case=False, na=False)).any(axis=1).sum()),
        "urgent_review_count": int(recommendations.astype(str).apply(lambda s: s.str.contains("urgente|reabasto|review", case=False, na=False, regex=True)).any(axis=1).sum()),
        "total_net_benefit_top": float(sum(a["net_benefit"] for a in alerts)),
        "total_expected_loss_top": float(sum(a["expected_loss"] for a in alerts)),
        "total_transfer_cost_top": float(sum(a["transfer_cost"] for a in alerts)),
    }

    return {
        "business_context": "Herdez Smart-Supply: priorizar acciones para evitar quiebres de stock en CEDIs minimizando costos logísticos.",
        "architecture_principle": "ML predicts risk; deterministic tools calculate cost-benefit; LLM agents reason, critique and explain.",
        "summary": summary,
        "top_alerts": alerts,
    }


def context_to_markdown(context: Dict[str, Any]) -> str:
    """Convierte el contexto estructurado en markdown legible para prompts y reportes."""
    s = context["summary"]
    lines = [
        "# Contexto operativo para agentes",
        "",
        f"Caso: {context['business_context']}",
        f"Principio de arquitectura: {context['architecture_principle']}",
        "",
        "## Resumen",
        f"- Recomendaciones totales: {s['total_recommendations']}",
        f"- Alertas enviadas al agente: {s['shown_alerts']}",
        f"- Beneficio neto agregado top alertas: {money(s['total_net_benefit_top'])}",
        f"- Pérdida esperada agregada top alertas: {money(s['total_expected_loss_top'])}",
        f"- Costo logístico agregado top alertas: {money(s['total_transfer_cost_top'])}",
        "",
        "## Alertas priorizadas",
    ]

    for i, a in enumerate(context["top_alerts"], start=1):
        lines.extend([
            f"### Alerta {i}",
            f"- SKU: {a['sku']}",
            f"- CEDI destino: {a['cedi_destino']}",
            f"- CEDI origen sugerido: {a['cedi_origen'] or 'No asignado'}",
            f"- Riesgo predicho: {pct(a['risk_probability'])} ({a['risk_level']})",
            f"- Acción recomendada por motor determinista: {a['recommended_action']}",
            f"- Unidades a transferir: {a['units_to_transfer']:,.0f}",
            f"- Pérdida esperada: {money(a['expected_loss'])}",
            f"- Costo de transferencia: {money(a['transfer_cost'])}",
            f"- Beneficio neto: {money(a['net_benefit'])}",
            f"- Estado política: {a['policy_status']}",
            "",
        ])

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 3. Prompts base: en inglés para instrucciones, salida en español
# -----------------------------------------------------------------------------

SYSTEM_DESIGN_PRINCIPLES = """
You are part of a supply-chain AI architecture for Grupo Herdez.
Important rules:
1. Do not invent numerical values.
2. Use only the provided deterministic recommendation context.
3. ML predicts stockout risk; deterministic tools calculate financial impact.
4. LLM agents are used for reasoning, critique and explanation, not for replacing audited calculations.
5. Final answer must be in Spanish.
6. Communicate clearly for two audiences: AI Manager and Supply Chain Director.
""".strip()


def build_tasks_text(context_markdown: str) -> Dict[str, str]:
    """Crea instrucciones de tareas para los agentes."""
    return {
        "risk_analysis": f"""
{SYSTEM_DESIGN_PRINCIPLES}

Analyze the stockout alerts from a technical ML perspective.
Identify the most critical SKU/CEDI combinations and explain why the risk matters.
Use this context:

{context_markdown}

Return Spanish output with:
- Critical alerts
- Technical interpretation
- Why the ML prediction should trigger operational analysis
""".strip(),
        "cost_analysis": f"""
{SYSTEM_DESIGN_PRINCIPLES}

Analyze the cost-benefit evidence.
Explain expected loss, transfer cost and net benefit.
Do not recalculate with invented data; interpret the given values.
Use this context:

{context_markdown}

Return Spanish output with:
- Financial interpretation
- Which actions have strongest ROI
- Why deterministic calculation matters
""".strip(),
        "strategy": f"""
{SYSTEM_DESIGN_PRINCIPLES}

Build an operational recommendation for the Supply Chain Director.
Compare: transfer inventory, wait for replenishment, urgent review, or monitor.
Use this context:

{context_markdown}

Return Spanish output with:
- Recommended action plan
- Operational risk
- Business priority
""".strip(),
        "critic": f"""
{SYSTEM_DESIGN_PRINCIPLES}

Act as a critic/auditor. Find weaknesses, risks or missing assumptions in the recommendation.
Examples: source CEDI risk, data quality, model confidence, policy constraints, business continuity.
Use this context:

{context_markdown}

Return Spanish output with:
- Risks to validate
- Guardrails
- Human-in-the-loop checkpoints
""".strip(),
        "executive": f"""
{SYSTEM_DESIGN_PRINCIPLES}

Create the final executive brief in Spanish.
Audience: Supply Chain Director and AI Manager.
Tone: clear, concise, business-oriented but technically credible.
Use this context:

{context_markdown}

Return a final answer with:
1. Executive summary
2. Top recommended actions
3. Estimated business impact
4. Technical architecture justification
5. Controls and auditability
""".strip(),
    }


# -----------------------------------------------------------------------------
# 4. Fallback determinista: garantiza demo aunque no haya LLM
# -----------------------------------------------------------------------------

def deterministic_agent_brief(context: Dict[str, Any]) -> str:
    """
    Genera un reporte tipo agente sin llamar a Gemini.

    Por qué existe:
    - En entrevista/demo puede fallar la API, la conexión o las credenciales.
    - El sistema debe degradarse de forma segura.
    - Esto demuestra pensamiento de arquitectura resiliente.
    """
    s = context["summary"]
    alerts = context["top_alerts"]

    transfer_alerts = [a for a in alerts if "transfer" in a["recommended_action"].lower() or "transfer" in a["explanation"].lower()]
    urgent_alerts = [a for a in alerts if a not in transfer_alerts]

    lines = [
        "# Herdez Smart-Supply — Brief ejecutivo generado por fallback determinista",
        "",
        "## 1. Resumen ejecutivo",
        (
            f"Se analizaron {s['shown_alerts']} alertas priorizadas de riesgo de quiebre. "
            f"El beneficio neto potencial agregado en estas alertas es de {money(s['total_net_benefit_top'])}, "
            f"comparando pérdida esperada contra costo logístico de transferencia."
        ),
        "",
        "## 2. Acciones recomendadas",
    ]

    for a in alerts[:5]:
        lines.append(
            f"- **{a['sku']} en {a['cedi_destino']}**: {a['recommended_action']}. "
            f"Riesgo {pct(a['risk_probability'])}, transferir {a['units_to_transfer']:,.0f} unidades "
            f"desde {a['cedi_origen'] or 'CEDI por validar'}, beneficio neto {money(a['net_benefit'])}."
        )

    lines.extend([
        "",
        "## 3. Interpretación de negocio",
        "Las recomendaciones priorizan evitar ventas perdidas cuando la pérdida esperada por quiebre supera el costo logístico. Esto protege el nivel de servicio y evita transferencias innecesarias.",
        "",
        "## 4. Controles y auditabilidad",
        "- El LLM no inventa la decisión financiera.",
        "- El riesgo viene del modelo predictivo.",
        "- El costo-beneficio viene del motor determinista.",
        "- Las reglas de política evitan trasladar el problema a otro CEDI.",
        "- La recomendación puede revisarse con human-in-the-loop antes de ejecutar la transferencia.",
        "",
        "## 5. Riesgos a validar antes de producción",
        "- Confirmar inventario real-time del CEDI origen.",
        "- Validar ventanas reales de transporte y disponibilidad de unidades.",
        "- Ajustar umbrales de riesgo según nivel de servicio objetivo.",
        "- Monitorear drift del modelo cuando cambien demanda, promociones o condiciones logísticas.",
    ])

    if urgent_alerts:
        lines.extend([
            "",
            "## 6. Alertas que requieren revisión humana",
        ])
        for a in urgent_alerts[:5]:
            lines.append(
                f"- **{a['sku']} en {a['cedi_destino']}**: {a['recommended_action']}. "
                "No se recomienda transferencia automática; validar reabasto o inventario alternativo."
            )

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 5. Ejecución opcional con CrewAI
# -----------------------------------------------------------------------------

def get_api_key() -> Optional[str]:
    """Obtiene API key desde variables de entorno comunes."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def run_crewai_system(context: Dict[str, Any], config: AgentSystemConfig) -> str:
    """
    Ejecuta un flujo multiagente con CrewAI.

    Nota:
    - CrewAI puede usar LiteLLM para modelos como Gemini.
    - Si falla por librerías/credenciales, el main regresará a fallback determinista.
    """
    try:
        from crewai import Agent, Crew, Process, Task
        try:
            from crewai import LLM
        except Exception:
            LLM = None
    except Exception as exc:
        raise RuntimeError(
            "No se pudo importar CrewAI. Instala con: pip install crewai"
        ) from exc

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No encontré GEMINI_API_KEY ni GOOGLE_API_KEY en variables de entorno.")

    context_md = context_to_markdown(context)
    tasks_text = build_tasks_text(context_md)

    # CrewAI usa proveedores compatibles con LiteLLM. Para Gemini suele usarse el prefijo gemini/.
    if LLM is not None:
        llm = LLM(model=config.model_name, api_key=api_key, temperature=0.2)
    else:
        # Compatibilidad con versiones antiguas: algunos ejemplos aceptaban string directo.
        llm = config.model_name

    risk_agent = Agent(
        role="Inventory Risk Analyst",
        goal="Interpretar el riesgo de quiebre por SKU/CEDI usando evidencia del modelo ML.",
        backstory=(
            "Especialista en analítica de inventarios. Entiende ventas, cobertura, lead time "
            "y riesgo operativo en centros de distribución."
        ),
        llm=llm,
        verbose=True,
    )

    cost_agent = Agent(
        role="Logistics Cost Analyst",
        goal="Interpretar la relación entre pérdida esperada, costo logístico y beneficio neto.",
        backstory=(
            "Especialista en costos logísticos y evaluación financiera de transferencias entre CEDIs. "
            "No inventa números: solo interpreta cálculos auditables."
        ),
        llm=llm,
        verbose=True,
    )

    strategist_agent = Agent(
        role="Supply Chain Strategist",
        goal="Proponer un plan operativo accionable para reducir quiebres de stock.",
        backstory=(
            "Arquitecto de decisiones de supply chain. Balancea nivel de servicio, costo logístico, "
            "riesgo operativo y velocidad de ejecución."
        ),
        llm=llm,
        verbose=True,
    )

    critic_agent = Agent(
        role="Policy Validator and Critic",
        goal="Auditar la recomendación para detectar riesgos, supuestos débiles y puntos de control humano.",
        backstory=(
            "Auditor de IA responsable. Evalúa guardrails, trazabilidad, riesgo de alucinación, "
            "calidad de datos y restricciones de negocio."
        ),
        llm=llm,
        verbose=True,
    )

    executive_agent = Agent(
        role="Executive Communicator",
        goal="Traducir la recomendación técnica a una explicación clara para dirección y gerencia técnica.",
        backstory=(
            "Consultor senior capaz de explicar IA, costos y operación a perfiles técnicos y de negocio."
        ),
        llm=llm,
        verbose=True,
    )

    risk_task = Task(
        description=tasks_text["risk_analysis"],
        expected_output="Análisis técnico en español de las alertas críticas y su interpretación ML.",
        agent=risk_agent,
    )
    cost_task = Task(
        description=tasks_text["cost_analysis"],
        expected_output="Análisis financiero en español de pérdida esperada, transferencia y beneficio neto.",
        agent=cost_agent,
    )
    strategy_task = Task(
        description=tasks_text["strategy"],
        expected_output="Plan operativo recomendado en español.",
        agent=strategist_agent,
        context=[risk_task, cost_task],
    )
    critic_task = Task(
        description=tasks_text["critic"],
        expected_output="Lista de riesgos, guardrails y validaciones humanas en español.",
        agent=critic_agent,
        context=[risk_task, cost_task, strategy_task],
    )
    executive_task = Task(
        description=tasks_text["executive"],
        expected_output="Brief ejecutivo final en español, claro y defendible.",
        agent=executive_agent,
        context=[risk_task, cost_task, strategy_task, critic_task],
    )

    crew = Crew(
        agents=[risk_agent, cost_agent, strategist_agent, critic_agent, executive_agent],
        tasks=[risk_task, cost_task, strategy_task, critic_task, executive_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


# -----------------------------------------------------------------------------
# 6. LangChain fallback opcional para una sola llamada directa
# -----------------------------------------------------------------------------

def run_langchain_single_call(context: Dict[str, Any], model_name: str = "gemini-2.5-flash") -> str:
    """
    Usa LangChain como abstracción de modelo para una llamada directa a Gemini.

    Esto NO reemplaza CrewAI. Sirve como fallback intermedio:
    - más simple que CrewAI,
    - mantiene desacoplamiento del proveedor,
    - genera un brief ejecutivo cuando no quieres correr varios agentes.
    """
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No encontré GEMINI_API_KEY ni GOOGLE_API_KEY.")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as exc:
        raise RuntimeError(
            "No se pudo importar langchain-google-genai. Instala con: pip install langchain-google-genai"
        ) from exc

    context_md = context_to_markdown(context)
    prompt = f"""
{SYSTEM_DESIGN_PRINCIPLES}

Act as a senior AI Architect and Supply Chain analytics advisor.
Create a Spanish executive brief using the deterministic recommendation context below.
Do not invent numbers.

{context_md}
""".strip()

    llm = ChatGoogleGenerativeAI(
        model=model_name.replace("gemini/", ""),
        google_api_key=api_key,
        temperature=0.2,
    )
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


# -----------------------------------------------------------------------------
# 7. Guardado de salidas
# -----------------------------------------------------------------------------

def save_outputs(context: Dict[str, Any], report: str, output_dir: str, mode: str) -> Dict[str, str]:
    """Guarda reporte, contexto y trazabilidad mínima."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_path = out / "agent_executive_brief.md"
    context_path = out / "agent_prompt_context.json"
    trace_path = out / "agent_technical_trace.json"

    report_path.write_text(report, encoding="utf-8")
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    trace = {
        "mode": mode,
        "architecture": "CrewAI/LangChain/Gemini over deterministic ML + decision engine",
        "principle": context["architecture_principle"],
        "input_recommendations": context["summary"]["total_recommendations"],
        "alerts_sent_to_agent": context["summary"]["shown_alerts"],
        "notes": [
            "No se envía todo el dataset al LLM; solo contexto priorizado.",
            "El LLM no calcula riesgo ni beneficio; interpreta salidas deterministas.",
            "Este diseño permite fallback si falla el proveedor de LLM.",
        ],
    }
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "report": str(report_path),
        "context": str(context_path),
        "trace": str(trace_path),
    }


# -----------------------------------------------------------------------------
# 8. CLI principal
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Herdez Smart-Supply - Sistema multiagente")
    parser.add_argument("--recommendations", default="outputs/decision_recommendations.csv", help="CSV de recomendaciones del decision engine")
    parser.add_argument("--scenarios", default="outputs/decision_transfer_scenarios.csv", help="CSV de escenarios de transferencia")
    parser.add_argument("--output-dir", default="outputs", help="Directorio de salida")
    parser.add_argument("--max-rows-context", type=int, default=8, help="Número máximo de alertas enviadas al agente")
    parser.add_argument("--model", default="gemini/gemini-2.5-flash", help="Modelo para CrewAI/LiteLLM")
    parser.add_argument(
        "--mode",
        choices=["auto", "crewai", "langchain", "fallback"],
        default="auto",
        help="Modo de ejecución. auto intenta CrewAI, luego LangChain, luego fallback determinista.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = AgentSystemConfig(
        model_name=args.model,
        max_rows_context=args.max_rows_context,
        output_dir=args.output_dir,
        recommendations_file=args.recommendations,
        scenarios_file=args.scenarios,
        force_fallback=args.mode == "fallback",
    )

    data = load_decision_outputs(config.recommendations_file, config.scenarios_file)
    context = build_agent_context(data["recommendations"], data["scenarios"], max_rows=config.max_rows_context)

    mode_used = "fallback"
    report = ""

    if args.mode in {"auto", "crewai"} and not config.force_fallback:
        try:
            print("Ejecutando CrewAI multiagente...")
            report = run_crewai_system(context, config)
            mode_used = "crewai"
        except Exception as exc:
            if args.mode == "crewai":
                raise
            print(f"No se pudo ejecutar CrewAI: {exc}")
            print("Intentaré LangChain como fallback intermedio...")

    if not report and args.mode in {"auto", "langchain"} and not config.force_fallback:
        try:
            report = run_langchain_single_call(context, model_name=config.model_name)
            mode_used = "langchain"
        except Exception as exc:
            if args.mode == "langchain":
                raise
            print(f"No se pudo ejecutar LangChain/Gemini: {exc}")
            print("Usaré fallback determinista para mantener la demo funcionando.")

    if not report:
        report = deterministic_agent_brief(context)
        mode_used = "fallback"

    paths = save_outputs(context, report, config.output_dir, mode_used)

    print("\nSistema agente ejecutado correctamente.")
    print(f"Modo usado: {mode_used}")
    print("Archivos generados:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
