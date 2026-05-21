"""
01_eda_target_features.py

Herdez Smart-Supply - Fase 1
EDA formal + creación del target de riesgo de quiebre + feature engineering.

Idea central:
    No queremos predecir simplemente "stock bajo hoy".
    Queremos predecir si el inventario actual alcanzará para cubrir la demanda futura
    durante el lead time de reabastecimiento.

Target:
    Riesgo_Quiebre = 1 si Stock_Actual <= demanda real acumulada de los próximos Lead_Time_Dias.

¿Por qué así?
    En supply chain, el riesgo operativo no depende solo del inventario actual, sino de si
    el inventario alcanza hasta que llegue el siguiente reabastecimiento.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = [
    "Fecha",
    "SKU_ID",
    "CEDI",
    "Ventas_Unidades",
    "Stock_Actual",
    "Lead_Time_Dias",
    "Promocion_Activa",
    "Precio_Combustible_MXN",
    "Clima",
    "Costo_Quiebre_Stock_Diario",
    "Costo_Transferencia_Unidad",
]

NUMERIC_COLUMNS = [
    "Ventas_Unidades",
    "Stock_Actual",
    "Lead_Time_Dias",
    "Precio_Combustible_MXN",
    "Costo_Quiebre_Stock_Diario",
    "Costo_Transferencia_Unidad",
]

CATEGORICAL_COLUMNS = ["SKU_ID", "CEDI", "Promocion_Activa", "Clima"]


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Carga la hoja principal del Excel y normaliza la columna Fecha."""
    df = pd.read_excel(input_path, sheet_name="Historico_Inventarios")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Valida reglas mínimas de calidad.

    ¿Por qué se hace?
    La regla de oro del MLOps es "Garbage in, garbage out".
    Antes de modelar, se valida que existan columnas, que no haya duplicados de granularidad
    y que el dataset esté en el nivel correcto: Fecha + SKU + CEDI.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    duplicates = df.duplicated(subset=["Fecha", "SKU_ID", "CEDI"]).sum()
    if duplicates > 0:
        raise ValueError(
            f"Hay {duplicates} duplicados por Fecha + SKU_ID + CEDI. "
            "Esto rompería la granularidad del análisis."
        )


def basic_profile(df: pd.DataFrame) -> Dict[str, object]:
    """Devuelve un resumen compacto del dataset."""
    return {
        "filas": len(df),
        "columnas": df.shape[1],
        "fecha_min": df["Fecha"].min().date().isoformat(),
        "fecha_max": df["Fecha"].max().date().isoformat(),
        "dias_historicos": int(df["Fecha"].nunique()),
        "num_skus": int(df["SKU_ID"].nunique()),
        "num_cedis": int(df["CEDI"].nunique()),
        "combinaciones_sku_cedi": int(df[["SKU_ID", "CEDI"]].drop_duplicates().shape[0]),
        "nulos_totales": int(df.isna().sum().sum()),
        "duplicados_fecha_sku_cedi": int(df.duplicated(subset=["Fecha", "SKU_ID", "CEDI"]).sum()),
    }


def univariate_analysis(df: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Análisis univariado.

    ¿Qué buscamos?
    - Rangos raros: ventas negativas, stock negativo, costos imposibles.
    - Variables con poca variabilidad: quizá no aportan al modelo.
    - Outliers: pueden representar errores o casos operativos críticos.
    """
    numeric_summary = df[NUMERIC_COLUMNS].describe().T
    categorical_summary = []
    for col in CATEGORICAL_COLUMNS:
        counts = df[col].value_counts(dropna=False).reset_index()
        counts.columns = [col, "conteo"]
        counts["porcentaje"] = counts["conteo"] / len(df)
        counts.insert(0, "variable", col)
        categorical_summary.append(counts.rename(columns={col: "valor"}))

    categorical_summary_df = pd.concat(categorical_summary, ignore_index=True)

    numeric_summary.to_csv(output_dir / "eda_numeric_summary.csv")
    categorical_summary_df.to_csv(output_dir / "eda_categorical_summary.csv", index=False)

    # Histogramas y boxplots simples para documentar distribución y outliers.
    for col in NUMERIC_COLUMNS:
        plt.figure(figsize=(8, 4))
        df[col].hist(bins=30)
        plt.title(f"Distribución de {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{col}.png", dpi=140)
        plt.close()

        plt.figure(figsize=(8, 3))
        plt.boxplot(df[col].dropna(), vert=False)
        plt.title(f"Boxplot de {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(output_dir / f"boxplot_{col}.png", dpi=140)
        plt.close()

    return numeric_summary, categorical_summary_df


def create_future_demand_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea target supervisado usando demanda futura dentro del lead time.

    ¿Por qué NO basta con Stock_Actual <= Ventas_Unidades * Lead_Time_Dias?
    Esa regla es útil como baseline, pero usa la venta de un día como aproximación.
    Para entrenar supervisado con histórico, podemos usar la demanda real observada en los próximos días.

    Cuidado de leakage:
    - La demanda futura SOLO se usa para construir la etiqueta histórica.
    - No se usa como feature del modelo.
    """
    df = df.sort_values(["SKU_ID", "CEDI", "Fecha"]).copy()
    future_demands: List[float] = []

    for _, group in df.groupby(["SKU_ID", "CEDI"], sort=False):
        ventas = group["Ventas_Unidades"].to_numpy()
        lead_times = group["Lead_Time_Dias"].astype(int).to_numpy()
        n = len(group)

        for i in range(n):
            lt = lead_times[i]
            start = i + 1
            end = i + 1 + lt
            if end <= n:
                future_demands.append(float(ventas[start:end].sum()))
            else:
                # No hay suficientes días futuros para etiquetar correctamente.
                future_demands.append(np.nan)

    df["Demanda_Futura_LT"] = future_demands
    df["Target_Riesgo_Quiebre"] = np.where(
        df["Demanda_Futura_LT"].notna(),
        (df["Stock_Actual"] <= df["Demanda_Futura_LT"]).astype(int),
        np.nan,
    )
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables explicativas disponibles al momento de decidir.

    Variables clave:
    - Ventas_Media_3d / 7d: estiman demanda reciente sin mirar al futuro.
    - Dias_Cobertura: cuántos días cubre el stock si la demanda sigue como el promedio reciente.
    - Gap_Estimado_LT: inventario menos demanda esperada durante lead time.
    - Ratio_Cobertura_LT: relación stock/demanda estimada; valores < 1 indican riesgo.
    """
    df = df.sort_values(["SKU_ID", "CEDI", "Fecha"]).copy()

    group_cols = ["SKU_ID", "CEDI"]

    df["Ventas_Lag_1d"] = df.groupby(group_cols)["Ventas_Unidades"].shift(1)
    df["Stock_Lag_1d"] = df.groupby(group_cols)["Stock_Actual"].shift(1)

    df["Ventas_Media_3d"] = (
        df.groupby(group_cols)["Ventas_Unidades"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    )
    df["Ventas_Media_7d"] = (
        df.groupby(group_cols)["Ventas_Unidades"]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
    )
    df["Ventas_Max_7d"] = (
        df.groupby(group_cols)["Ventas_Unidades"]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).max())
    )

    # Para el primer día de cada serie no hay historial; usamos la venta del día como fallback.
    for col in ["Ventas_Lag_1d", "Ventas_Media_3d", "Ventas_Media_7d", "Ventas_Max_7d"]:
        df[col] = df[col].fillna(df["Ventas_Unidades"])
    df["Stock_Lag_1d"] = df["Stock_Lag_1d"].fillna(df["Stock_Actual"])

    df["Demanda_Estimada_LT"] = df["Ventas_Media_7d"] * df["Lead_Time_Dias"]
    df["Dias_Cobertura"] = df["Stock_Actual"] / df["Ventas_Media_7d"].clip(lower=1)
    df["Gap_Estimado_LT"] = df["Stock_Actual"] - df["Demanda_Estimada_LT"]
    df["Ratio_Cobertura_LT"] = df["Stock_Actual"] / df["Demanda_Estimada_LT"].clip(lower=1)

    df["Dia_Semana"] = df["Fecha"].dt.dayofweek
    df["Semana_Anio"] = df["Fecha"].dt.isocalendar().week.astype(int)
    df["Dia_Mes"] = df["Fecha"].dt.day

    # Normalizamos promoción a entero 0/1 si viene booleana o texto.
    if df["Promocion_Activa"].dtype == bool:
        df["Promocion_Activa_Num"] = df["Promocion_Activa"].astype(int)
    else:
        df["Promocion_Activa_Num"] = df["Promocion_Activa"].astype(str).str.lower().isin(
            ["1", "true", "sí", "si", "yes", "activo", "activa"]
        ).astype(int)

    return df


def bivariate_and_multivariate_analysis(df_model: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Análisis bivariado/multivariado.

    ¿Qué buscamos?
    - Qué variables se mueven junto con el riesgo.
    - Qué SKUs y CEDIs concentran más riesgo.
    - Si hay correlaciones fuertes que puedan indicar redundancia o relación operativa.
    """
    numeric_cols = [
        "Ventas_Unidades",
        "Stock_Actual",
        "Lead_Time_Dias",
        "Precio_Combustible_MXN",
        "Costo_Quiebre_Stock_Diario",
        "Costo_Transferencia_Unidad",
        "Ventas_Media_3d",
        "Ventas_Media_7d",
        "Demanda_Estimada_LT",
        "Dias_Cobertura",
        "Gap_Estimado_LT",
        "Ratio_Cobertura_LT",
        "Target_Riesgo_Quiebre",
    ]
    corr = df_model[numeric_cols].corr(numeric_only=True)
    corr.to_csv(output_dir / "eda_correlation_matrix.csv")

    # Heatmap básico con matplotlib para evitar dependencia extra de seaborn.
    plt.figure(figsize=(11, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlación")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Matriz de correlación de variables numéricas")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_correlaciones.png", dpi=160)
    plt.close()

    risk_by_sku = (
        df_model.groupby("SKU_ID")
        .agg(riesgos=("Target_Riesgo_Quiebre", "sum"), registros=("Target_Riesgo_Quiebre", "count"))
        .assign(tasa_riesgo=lambda x: x["riesgos"] / x["registros"])
        .sort_values("tasa_riesgo", ascending=False)
    )
    risk_by_cedi = (
        df_model.groupby("CEDI")
        .agg(riesgos=("Target_Riesgo_Quiebre", "sum"), registros=("Target_Riesgo_Quiebre", "count"))
        .assign(tasa_riesgo=lambda x: x["riesgos"] / x["registros"])
        .sort_values("tasa_riesgo", ascending=False)
    )
    risk_by_pair = (
        df_model.groupby(["SKU_ID", "CEDI"])
        .agg(riesgos=("Target_Riesgo_Quiebre", "sum"), registros=("Target_Riesgo_Quiebre", "count"))
        .assign(tasa_riesgo=lambda x: x["riesgos"] / x["registros"])
        .sort_values("tasa_riesgo", ascending=False)
    )

    risk_by_sku.to_csv(output_dir / "risk_by_sku.csv")
    risk_by_cedi.to_csv(output_dir / "risk_by_cedi.csv")
    risk_by_pair.to_csv(output_dir / "risk_by_sku_cedi.csv")

    return corr


def time_series_analysis(df_model: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Análisis temporal.

    ¿Por qué aplica aquí?
    Inventario y ventas son procesos temporales. Si ignoramos el tiempo, podríamos hacer un split aleatorio
    que mezcle pasado y futuro, inflando artificialmente la métrica del modelo.
    """
    daily = (
        df_model.groupby("Fecha")
        .agg(
            ventas_totales=("Ventas_Unidades", "sum"),
            stock_total=("Stock_Actual", "sum"),
            riesgo_promedio=("Target_Riesgo_Quiebre", "mean"),
        )
        .reset_index()
    )
    daily.to_csv(output_dir / "time_series_daily_summary.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(daily["Fecha"], daily["ventas_totales"], marker="o", linewidth=1)
    plt.title("Ventas totales por día")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas unidades")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "ts_ventas_totales.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(daily["Fecha"], daily["riesgo_promedio"], marker="o", linewidth=1)
    plt.title("Tasa promedio de riesgo de quiebre por día")
    plt.xlabel("Fecha")
    plt.ylabel("Riesgo promedio")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "ts_riesgo_promedio.png", dpi=150)
    plt.close()

    return daily


def temporal_split(df_model: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Split temporal.

    ¿Por qué no split aleatorio?
    Porque el modelo se usará para predecir días futuros. Evaluar en las últimas fechas simula mejor producción.
    """
    fechas = np.sort(df_model["Fecha"].unique())
    cutoff_idx = int(len(fechas) * train_ratio)
    cutoff_date = pd.Timestamp(fechas[cutoff_idx])
    train_df = df_model[df_model["Fecha"] < cutoff_date].copy()
    test_df = df_model[df_model["Fecha"] >= cutoff_date].copy()
    return train_df, test_df, cutoff_date


def save_business_findings(df_model: pd.DataFrame, output_dir: Path, profile: Dict[str, object]) -> None:
    """Guarda un resumen legible para usar en la presentación."""
    total = len(df_model)
    risks = int(df_model["Target_Riesgo_Quiebre"].sum())
    risk_rate = risks / total if total else 0

    top_sku = (
        df_model.groupby("SKU_ID")["Target_Riesgo_Quiebre"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
    )
    top_cedi = (
        df_model.groupby("CEDI")["Target_Riesgo_Quiebre"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
    )

    lines = [
        "# Hallazgos EDA - Herdez Smart-Supply",
        "",
        "## Resumen del dataset",
    ]
    for k, v in profile.items():
        lines.append(f"- {k}: {v}")

    lines.extend(
        [
            "",
            "## Target de riesgo",
            f"- Registros modelables: {total}",
            f"- Registros en riesgo: {risks}",
            f"- Tasa de riesgo: {risk_rate:.2%}",
            "",
            "## Principales insights",
            f"- SKU con mayor tasa de riesgo: {top_sku.index[0]} ({top_sku.iloc[0]:.2%})",
            f"- CEDI con mayor tasa de riesgo: {top_cedi.index[0]} ({top_cedi.iloc[0]:.2%})",
            "- Las variables de cobertura durante lead time son candidatas fuertes para predecir quiebre.",
            "- La evaluación debe priorizar recall en la clase de riesgo para evitar ventas perdidas.",
        ]
    )
    (output_dir / "eda_business_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA + target + features para Herdez Smart-Supply")
    parser.add_argument("--input", type=str, default="data/Data_Prueba_Tecnica_Herdez_IA.xlsx")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    validate_dataset(df)

    profile = basic_profile(df)
    pd.Series(profile).to_csv(output_dir / "eda_dataset_profile.csv")

    print("\n=== Perfil inicial ===")
    for key, value in profile.items():
        print(f"{key}: {value}")

    print("\n=== Análisis univariado ===")
    univariate_analysis(df, output_dir)

    print("\n=== Creando target futuro dentro del lead time ===")
    df_target = create_future_demand_target(df)
    df_features = add_features(df_target)

    # Quitamos filas donde no se puede construir target por falta de futuro histórico.
    df_model = df_features.dropna(subset=["Target_Riesgo_Quiebre"]).copy()
    df_model["Target_Riesgo_Quiebre"] = df_model["Target_Riesgo_Quiebre"].astype(int)

    print("Registros originales:", len(df))
    print("Registros modelables:", len(df_model))
    print("Tasa de riesgo:", f"{df_model['Target_Riesgo_Quiebre'].mean():.2%}")

    print("\n=== Análisis bivariado/multivariado ===")
    bivariate_and_multivariate_analysis(df_model, output_dir)

    print("\n=== Análisis temporal ===")
    time_series_analysis(df_model, output_dir)

    train_df, test_df, cutoff_date = temporal_split(df_model)
    print("\n=== Split temporal ===")
    print("Fecha de corte:", cutoff_date.date())
    print("Train:", train_df.shape, "riesgo:", f"{train_df['Target_Riesgo_Quiebre'].mean():.2%}")
    print("Test:", test_df.shape, "riesgo:", f"{test_df['Target_Riesgo_Quiebre'].mean():.2%}")

    df_model.to_csv(output_dir / "herdez_features_dataset.csv", index=False)
    train_df.to_csv(output_dir / "train_dataset.csv", index=False)
    test_df.to_csv(output_dir / "test_dataset.csv", index=False)
    save_business_findings(df_model, output_dir, profile)

    print("\nArchivos generados en:", output_dir.resolve())
    print("- herdez_features_dataset.csv")
    print("- train_dataset.csv")
    print("- test_dataset.csv")
    print("- eda_business_findings.md")


if __name__ == "__main__":
    main()
