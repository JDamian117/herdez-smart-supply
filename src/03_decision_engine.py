"""
03_decision_engine.py

Herdez Smart-Supply - Fase 3
Motor determinista de decisión costo-beneficio.

Objetivo del archivo
--------------------
Convertir la salida del modelo predictivo en una recomendación operativa:
    1) ¿Hay riesgo de quiebre?
    2) ¿Cuántas unidades faltan para cubrir el lead time?
    3) ¿Existe otro CEDI que pueda mandar inventario sin quedar en riesgo?
    4) ¿El costo logístico de transferir inventario se justifica contra la pérdida esperada?
    5) ¿Qué acción recomendamos y cómo la explicamos?

Diseño importante
-----------------
Este archivo NO usa LLMs. Es intencional.

La razón es de arquitectura:
    - El modelo ML predice probabilidad de riesgo.
    - Este motor calcula costos y aplica reglas de negocio.
    - El agente GenAI, en una fase posterior, solo orquesta y explica.

Así evitamos que el LLM invente números o tome decisiones financieras no auditables.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Configuración de negocio
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionConfig:
    """
    Parámetros de negocio que pueden ajustarse sin reentrenar el modelo.

    En una empresa real, estos umbrales podrían venir de una tabla de políticas,
    de un sistema ERP/WMS o de una configuración editable en el dashboard.
    """

    # Colchón adicional para cubrir incertidumbre de demanda.
    safety_factor: float = 1.10

    # El CEDI origen debe conservar al menos este múltiplo de demanda durante lead time.
    min_source_coverage_ratio: float = 1.20

    # Umbrales para explicar el riesgo al usuario de negocio.
    high_risk_threshold: float = 0.70
    medium_risk_threshold: float = 0.40

    # Beneficio mínimo para recomendar una acción logística.
    # Puede subirse si la empresa quiere evitar transferencias de bajo impacto.
    min_net_benefit: float = 0.0

    # Número máximo de alertas a convertir en recomendaciones.
    max_alerts: int = 50


@dataclass
class TransferScenario:
    """
    Escenario evaluado para mover inventario desde un CEDI origen a un CEDI destino.
    """

    fecha: str
    sku_id: str
    cedi_destino: str
    cedi_origen: str
    unidades_necesarias: int
    unidades_a_transferir: int
    riesgo_destino: float
    riesgo_origen: float
    stock_destino: float
    stock_origen: float
    excedente_origen: int
    perdida_esperada_antes: float
    perdida_esperada_despues: float
    perdida_evitada: float
    costo_transferencia: float
    beneficio_neto: float
    politica_origen_ok: bool


# -----------------------------------------------------------------------------
# 2. Carga de datos y predicción
# -----------------------------------------------------------------------------

def load_model(model_path: Path) -> Dict:
    """Carga el artefacto entrenado en la fase 02."""
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    return joblib.load(model_path)


def load_processed_dataset(processed_path: Path) -> pd.DataFrame:
    """Carga el dataset procesado generado por 01_eda_target_features.py."""
    if not processed_path.exists():
        raise FileNotFoundError(f"No existe el dataset procesado: {processed_path}")
    df = pd.read_csv(processed_path, parse_dates=["Fecha"])
    return df


def predict_risk(df: pd.DataFrame, model_artifact: Dict) -> pd.DataFrame:
    """
    Agrega probabilidad de quiebre y alerta binaria usando el modelo entrenado.

    El modelo devuelve probabilidad. La alerta usa el threshold guardado en entrenamiento.
    Separar probabilidad de alerta es importante porque el dashboard puede mostrar ambas:
        - probabilidad: qué tan riesgoso es
        - alerta: si cruza el umbral operativo
    """
    feature_columns = model_artifact["feature_columns"]
    pipeline = model_artifact["pipeline"]
    threshold = float(model_artifact.get("threshold", 0.5))

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas por el modelo: {missing}")

    scored = df.copy()
    scored["Riesgo_Probabilidad"] = pipeline.predict_proba(scored[feature_columns])[:, 1]
    scored["Alerta_Riesgo"] = (scored["Riesgo_Probabilidad"] >= threshold).astype(int)
    return scored


# -----------------------------------------------------------------------------
# 3. Funciones de negocio deterministas
# -----------------------------------------------------------------------------

def classify_risk_level(probability: float, config: DecisionConfig) -> str:
    """Convierte probabilidad numérica en etiqueta entendible para negocio."""
    if probability >= config.high_risk_threshold:
        return "Alto"
    if probability >= config.medium_risk_threshold:
        return "Medio"
    return "Bajo"


def estimate_demand_during_lead_time(row: pd.Series, config: DecisionConfig) -> float:
    """
    Estima la demanda esperada durante el lead time.

    Usamos Ventas_Media_7d porque suaviza ruido diario.
    Se multiplica por Lead_Time_Dias porque ese es el tiempo de reacción operativa.
    Se aplica safety_factor para agregar un colchón conservador.
    """
    return float(row["Ventas_Media_7d"] * row["Lead_Time_Dias"] * config.safety_factor)


def estimate_units_needed(row: pd.Series, config: DecisionConfig) -> int:
    """
    Calcula cuántas unidades faltan para cubrir la demanda del lead time.

    Si Stock_Actual ya cubre la demanda estimada, entonces no necesitamos mover unidades.
    """
    demand_lt = estimate_demand_during_lead_time(row, config)
    needed = max(0.0, demand_lt - float(row["Stock_Actual"]))
    return int(np.ceil(needed))


def expected_stockout_loss(row: pd.Series) -> float:
    """
    Pérdida esperada por quiebre.

    No asumimos que el quiebre ocurrirá con certeza.
    Ponderamos el costo por la probabilidad del modelo.

    Fórmula:
        perdida_esperada = probabilidad_quiebre * costo_quiebre_diario * lead_time
    """
    return float(
        row["Riesgo_Probabilidad"]
        * row["Costo_Quiebre_Stock_Diario"]
        * row["Lead_Time_Dias"]
    )


def calculate_transfer_cost(row: pd.Series, units_to_move: int) -> float:
    """Costo logístico total de mover cierto número de unidades."""
    return float(units_to_move * row["Costo_Transferencia_Unidad"])


def calculate_source_surplus(source_row: pd.Series, config: DecisionConfig) -> int:
    """
    Calcula excedente disponible del CEDI origen.

    Política:
        No se puede vaciar un CEDI para salvar otro.
        El origen debe conservar inventario suficiente para cubrir su propia demanda esperada.
    """
    source_demand_lt = estimate_demand_during_lead_time(source_row, config)
    min_stock_required = source_demand_lt * config.min_source_coverage_ratio
    surplus = float(source_row["Stock_Actual"]) - min_stock_required
    return int(max(0, np.floor(surplus)))


def estimate_loss_after_transfer(
    alert_row: pd.Series,
    units_transferred: int,
    units_needed: int,
    expected_loss_before: float,
) -> Tuple[float, float]:
    """
    Estima pérdida residual después de transferir inventario.

    Razón:
        Si solo cubrimos parte del faltante, no debemos asumir que evitamos toda la pérdida.
        Estimamos la pérdida evitada proporcionalmente al faltante cubierto.

    Retorna:
        (perdida_despues, perdida_evitada)
    """
    if units_needed <= 0:
        return 0.0, expected_loss_before

    coverage_ratio = min(1.0, units_transferred / units_needed)
    loss_avoided = expected_loss_before * coverage_ratio
    loss_after = expected_loss_before - loss_avoided
    return float(loss_after), float(loss_avoided)


# -----------------------------------------------------------------------------
# 4. Generación de escenarios
# -----------------------------------------------------------------------------

def get_source_candidates(scored_df: pd.DataFrame, alert_row: pd.Series) -> pd.DataFrame:
    """
    Busca otros CEDIs con el mismo SKU en la misma fecha.

    Esto simula la pregunta operativa:
        "Si CEDI Occidente está en riesgo para Salsa Verde hoy,
         ¿qué otros CEDIs tienen Salsa Verde hoy?"
    """
    return scored_df[
        (scored_df["Fecha"] == alert_row["Fecha"])
        & (scored_df["SKU_ID"] == alert_row["SKU_ID"])
        & (scored_df["CEDI"] != alert_row["CEDI"])
    ].copy()


def build_transfer_scenarios(
    scored_df: pd.DataFrame,
    alert_row: pd.Series,
    config: DecisionConfig,
) -> List[TransferScenario]:
    """
    Construye escenarios de transferencia desde CEDIs alternos.

    Cada escenario calcula:
        - cuánto puede mandar el origen
        - cuánto cuesta moverlo
        - cuánta pérdida se evita
        - cuál es el beneficio neto
    """
    units_needed = estimate_units_needed(alert_row, config)
    expected_loss_before = expected_stockout_loss(alert_row)
    candidates = get_source_candidates(scored_df, alert_row)

    scenarios: List[TransferScenario] = []
    for _, source_row in candidates.iterrows():
        source_excess = calculate_source_surplus(source_row, config)
        units_to_transfer = min(units_needed, source_excess)
        policy_ok = source_excess > 0 and units_to_transfer > 0

        if not policy_ok:
            # Guardamos únicamente escenarios accionables para mantener limpio el reporte.
            continue

        loss_after, loss_avoided = estimate_loss_after_transfer(
            alert_row=alert_row,
            units_transferred=units_to_transfer,
            units_needed=units_needed,
            expected_loss_before=expected_loss_before,
        )
        cost = calculate_transfer_cost(alert_row, units_to_transfer)
        net_benefit = loss_avoided - cost

        scenarios.append(
            TransferScenario(
                fecha=str(pd.to_datetime(alert_row["Fecha"]).date()),
                sku_id=str(alert_row["SKU_ID"]),
                cedi_destino=str(alert_row["CEDI"]),
                cedi_origen=str(source_row["CEDI"]),
                unidades_necesarias=int(units_needed),
                unidades_a_transferir=int(units_to_transfer),
                riesgo_destino=float(alert_row["Riesgo_Probabilidad"]),
                riesgo_origen=float(source_row.get("Riesgo_Probabilidad", np.nan)),
                stock_destino=float(alert_row["Stock_Actual"]),
                stock_origen=float(source_row["Stock_Actual"]),
                excedente_origen=int(source_excess),
                perdida_esperada_antes=round(expected_loss_before, 2),
                perdida_esperada_despues=round(loss_after, 2),
                perdida_evitada=round(loss_avoided, 2),
                costo_transferencia=round(cost, 2),
                beneficio_neto=round(net_benefit, 2),
                politica_origen_ok=bool(policy_ok),
            )
        )

    return scenarios


def choose_best_scenario(scenarios: Iterable[TransferScenario]) -> Optional[TransferScenario]:
    """
    Selecciona el mejor escenario.

    Prioridad:
        1. Mayor beneficio neto
        2. Mayor pérdida evitada
        3. Mayor cantidad transferida
    """
    scenarios = list(scenarios)
    if not scenarios:
        return None
    return sorted(
        scenarios,
        key=lambda s: (s.beneficio_neto, s.perdida_evitada, s.unidades_a_transferir),
        reverse=True,
    )[0]


# -----------------------------------------------------------------------------
# 5. Recomendación final
# -----------------------------------------------------------------------------

def generate_business_explanation(
    alert_row: pd.Series,
    config: DecisionConfig,
    action: str,
    reason: str,
    best_scenario: Optional[TransferScenario],
    units_needed: int,
    expected_loss: float,
) -> str:
    """Genera explicación ejecutiva sin usar LLM."""
    risk_level = classify_risk_level(float(alert_row["Riesgo_Probabilidad"]), config)

    if best_scenario is not None and action == "TRANSFER_INVENTORY":
        return (
            f"El SKU {alert_row['SKU_ID']} en {alert_row['CEDI']} presenta riesgo {risk_level.lower()} "
            f"de quiebre ({alert_row['Riesgo_Probabilidad']:.1%}). "
            f"Se recomienda transferir {best_scenario.unidades_a_transferir} unidades desde "
            f"{best_scenario.cedi_origen}. La pérdida esperada antes de actuar es de "
            f"${best_scenario.perdida_esperada_antes:,.2f}; la transferencia cuesta "
            f"${best_scenario.costo_transferencia:,.2f} y genera un beneficio neto estimado de "
            f"${best_scenario.beneficio_neto:,.2f}."
        )

    return (
        f"El SKU {alert_row['SKU_ID']} en {alert_row['CEDI']} presenta riesgo {risk_level.lower()} "
        f"de quiebre ({alert_row['Riesgo_Probabilidad']:.1%}). "
        f"Se estiman {units_needed} unidades faltantes durante el lead time y una pérdida esperada "
        f"de ${expected_loss:,.2f}. Recomendación: {reason}"
    )


def recommend_action_for_alert(
    scored_df: pd.DataFrame,
    alert_row: pd.Series,
    config: DecisionConfig,
) -> Tuple[Dict, List[TransferScenario]]:
    """
    Genera recomendación determinista para una alerta de riesgo.

    Decisiones posibles:
        - TRANSFER_INVENTORY: mover inventario desde otro CEDI.
        - WAIT_REPLENISHMENT: esperar reabasto porque transferir no conviene.
        - EXPEDITE_REPLENISHMENT_OR_REVIEW: no hay origen viable; revisar reabasto urgente.
        - MONITOR: riesgo no suficientemente alto o no faltan unidades.
    """
    risk_probability = float(alert_row["Riesgo_Probabilidad"])
    risk_level = classify_risk_level(risk_probability, config)
    units_needed = estimate_units_needed(alert_row, config)
    expected_loss = expected_stockout_loss(alert_row)

    scenarios = build_transfer_scenarios(scored_df, alert_row, config)
    best_scenario = choose_best_scenario(scenarios)

    if units_needed <= 0:
        action = "MONITOR"
        reason = "El inventario actual cubre la demanda estimada durante el lead time."
    elif risk_probability < config.medium_risk_threshold:
        action = "MONITOR"
        reason = "El riesgo estimado aún no justifica una acción logística inmediata."
    elif best_scenario is not None and best_scenario.beneficio_neto > config.min_net_benefit:
        action = "TRANSFER_INVENTORY"
        reason = "Existe un CEDI origen viable y la pérdida evitada supera el costo logístico."
    elif best_scenario is not None:
        action = "WAIT_REPLENISHMENT"
        reason = "Existe inventario transferible, pero el beneficio neto no justifica el costo logístico."
    else:
        action = "EXPEDITE_REPLENISHMENT_OR_REVIEW"
        reason = "No se encontró un CEDI origen con excedente suficiente sin ponerlo en riesgo."

    explanation = generate_business_explanation(
        alert_row=alert_row,
        config=config,
        action=action,
        reason=reason,
        best_scenario=best_scenario,
        units_needed=units_needed,
        expected_loss=expected_loss,
    )

    recommendation = {
        "Fecha": str(pd.to_datetime(alert_row["Fecha"]).date()),
        "SKU_ID": alert_row["SKU_ID"],
        "CEDI_Destino": alert_row["CEDI"],
        "Riesgo_Probabilidad": round(risk_probability, 6),
        "Nivel_Riesgo": risk_level,
        "Stock_Actual": float(alert_row["Stock_Actual"]),
        "Ventas_Media_7d": round(float(alert_row["Ventas_Media_7d"]), 2),
        "Lead_Time_Dias": int(alert_row["Lead_Time_Dias"]),
        "Demanda_Estimada_LT_Segura": round(estimate_demand_during_lead_time(alert_row, config), 2),
        "Unidades_Necesarias": int(units_needed),
        "Perdida_Esperada_Sin_Actuar": round(expected_loss, 2),
        "Accion_Recomendada": action,
        "CEDI_Origen_Recomendado": None if best_scenario is None else best_scenario.cedi_origen,
        "Unidades_A_Transferir": 0 if best_scenario is None else best_scenario.unidades_a_transferir,
        "Costo_Transferencia": 0.0 if best_scenario is None else best_scenario.costo_transferencia,
        "Perdida_Evitada": 0.0 if best_scenario is None else best_scenario.perdida_evitada,
        "Beneficio_Neto": 0.0 if best_scenario is None else best_scenario.beneficio_neto,
        "Razon": reason,
        "Explicacion_Ejecutiva": explanation,
    }
    return recommendation, scenarios


def generate_recommendations(
    scored_df: pd.DataFrame,
    config: DecisionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convierte alertas en recomendaciones y guarda también todos los escenarios evaluados.
    """
    alerts = (
        scored_df[scored_df["Alerta_Riesgo"] == 1]
        .sort_values("Riesgo_Probabilidad", ascending=False)
        .head(config.max_alerts)
    )

    recommendations: List[Dict] = []
    all_scenarios: List[Dict] = []

    for _, row in alerts.iterrows():
        recommendation, scenarios = recommend_action_for_alert(scored_df, row, config)
        recommendations.append(recommendation)
        for scenario in scenarios:
            all_scenarios.append(asdict(scenario))

    return pd.DataFrame(recommendations), pd.DataFrame(all_scenarios)


# -----------------------------------------------------------------------------
# 6. Reporte para entrevista
# -----------------------------------------------------------------------------

def build_decision_engine_report(
    recommendations: pd.DataFrame,
    config: DecisionConfig,
) -> str:
    """Crea un mini reporte en Markdown para explicar el módulo en entrevista."""
    lines = []
    lines.append("# Herdez Smart-Supply - Decision Engine")
    lines.append("")
    lines.append("## ¿Qué hace este módulo?")
    lines.append(
        "Convierte alertas de riesgo de quiebre generadas por el modelo ML en recomendaciones "
        "operativas basadas en costo-beneficio."
    )
    lines.append("")
    lines.append("## Principio de arquitectura")
    lines.append(
        "El LLM no toma decisiones financieras directamente. La predicción, el cálculo de pérdida, "
        "el costo logístico y la validación de política se calculan de forma determinista y auditable."
    )
    lines.append("")
    lines.append("## Parámetros de negocio")
    lines.append(f"- Safety factor de demanda: {config.safety_factor}")
    lines.append(f"- Cobertura mínima del CEDI origen: {config.min_source_coverage_ratio}")
    lines.append(f"- Umbral riesgo alto: {config.high_risk_threshold}")
    lines.append(f"- Umbral riesgo medio: {config.medium_risk_threshold}")
    lines.append("")

    if recommendations.empty:
        lines.append("## Resultado")
        lines.append("No se generaron recomendaciones con el umbral actual.")
    else:
        action_counts = recommendations["Accion_Recomendada"].value_counts().to_dict()
        total_net_benefit = recommendations["Beneficio_Neto"].sum()
        total_loss_avoided = recommendations["Perdida_Evitada"].sum()
        total_transfer_cost = recommendations["Costo_Transferencia"].sum()

        lines.append("## Resumen de recomendaciones")
        for action, count in action_counts.items():
            lines.append(f"- {action}: {count}")
        lines.append("")
        lines.append(f"- Pérdida evitada estimada: ${total_loss_avoided:,.2f}")
        lines.append(f"- Costo logístico estimado: ${total_transfer_cost:,.2f}")
        lines.append(f"- Beneficio neto estimado: ${total_net_benefit:,.2f}")
        lines.append("")
        lines.append("## Top 5 recomendaciones")
        top_cols = [
            "Fecha", "SKU_ID", "CEDI_Destino", "Nivel_Riesgo", "Accion_Recomendada",
            "CEDI_Origen_Recomendado", "Unidades_A_Transferir", "Beneficio_Neto"
        ]
        top_table = recommendations[top_cols].head(5).to_markdown(index=False)
        lines.append(top_table)

    lines.append("")
    lines.append("## Mensaje para entrevista")
    lines.append(
        "> El modelo predictivo identifica el riesgo; el motor determinista calcula si conviene actuar; "
        "el agente GenAI explica la decisión y permite conversar con el usuario."
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 7. CLI principal
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Motor de decisión costo-beneficio para Herdez Smart-Supply"
    )
    parser.add_argument("--processed", type=str, default="outputs/herdez_features_dataset.csv")
    parser.add_argument("--model", type=str, default="models/best_stockout_model.joblib")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--max-alerts", type=int, default=50)
    parser.add_argument("--safety-factor", type=float, default=1.10)
    parser.add_argument("--source-coverage", type=float, default=1.20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DecisionConfig(
        safety_factor=args.safety_factor,
        min_source_coverage_ratio=args.source_coverage,
        max_alerts=args.max_alerts,
    )

    df = load_processed_dataset(Path(args.processed))
    artifact = load_model(Path(args.model))
    scored_df = predict_risk(df, artifact)

    recommendations, scenarios = generate_recommendations(scored_df, config)

    scored_path = output_dir / "scored_stockout_alerts.csv"
    recs_path = output_dir / "decision_recommendations.csv"
    scenarios_path = output_dir / "decision_transfer_scenarios.csv"
    report_path = output_dir / "decision_engine_report.md"
    config_path = output_dir / "decision_engine_config.json"

    scored_df.to_csv(scored_path, index=False)
    recommendations.to_csv(recs_path, index=False)
    scenarios.to_csv(scenarios_path, index=False)
    report_path.write_text(build_decision_engine_report(recommendations, config), encoding="utf-8")
    config_path.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Herdez Smart-Supply | Decision Engine ===")
    print(f"Alertas evaluadas: {len(recommendations)}")
    if not recommendations.empty:
        print("\nDistribución de acciones:")
        print(recommendations["Accion_Recomendada"].value_counts().to_string())
        print("\nTop recomendaciones:")
        cols = [
            "Fecha", "SKU_ID", "CEDI_Destino", "Riesgo_Probabilidad", "Nivel_Riesgo",
            "Accion_Recomendada", "CEDI_Origen_Recomendado", "Unidades_A_Transferir",
            "Beneficio_Neto"
        ]
        print(recommendations[cols].head(10).to_string(index=False))

    print("\nArchivos generados:")
    print("-", scored_path)
    print("-", recs_path)
    print("-", scenarios_path)
    print("-", report_path)
    print("-", config_path)


if __name__ == "__main__":
    main()
