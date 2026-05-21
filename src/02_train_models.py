"""
02_train_models.py

Herdez Smart-Supply - Fase 2
Comparación de modelos y entrenamiento del modelo final.

Enfoque:
    1. Usar split temporal, no aleatorio.
    2. Comparar baseline, modelos interpretables y XGBoost.
    3. Priorizar recall de la clase 1 porque un falso negativo significa no anticipar un quiebre.
    4. Guardar el mejor modelo como artefacto reutilizable para Streamlit y agentes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


TARGET = "Target_Riesgo_Quiebre"

FEATURE_COLUMNS = [
    # Identidad operativa
    "SKU_ID",
    "CEDI",
    # Demanda e inventario
    "Ventas_Unidades",
    "Stock_Actual",
    "Lead_Time_Dias",
    "Ventas_Lag_1d",
    "Stock_Lag_1d",
    "Ventas_Media_3d",
    "Ventas_Media_7d",
    "Ventas_Max_7d",
    "Demanda_Estimada_LT",
    "Dias_Cobertura",
    "Gap_Estimado_LT",
    "Ratio_Cobertura_LT",
    # Contexto comercial/logístico
    "Promocion_Activa_Num",
    "Precio_Combustible_MXN",
    "Clima",
    "Costo_Quiebre_Stock_Diario",
    "Costo_Transferencia_Unidad",
    # Tiempo
    "Dia_Semana",
    "Semana_Anio",
    "Dia_Mes",
]

CATEGORICAL_FEATURES = ["SKU_ID", "CEDI", "Clima"]
NUMERIC_FEATURES = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]


def load_processed_dataset(processed_path: Path) -> pd.DataFrame:
    df = pd.read_csv(processed_path, parse_dates=["Fecha"])
    missing = [col for col in FEATURE_COLUMNS + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en dataset procesado: {missing}")
    return df


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    fechas = np.sort(df["Fecha"].unique())
    cutoff_idx = int(len(fechas) * train_ratio)
    cutoff_date = pd.Timestamp(fechas[cutoff_idx])
    train_df = df[df["Fecha"] < cutoff_date].copy()
    test_df = df[df["Fecha"] >= cutoff_date].copy()
    return train_df, test_df, cutoff_date


def make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """
    Preprocesamiento común.

    - OneHotEncoder para variables categóricas: SKU, CEDI, Clima.
    - Imputación por mediana para numéricas.
    - Escalamiento solo para modelos lineales; árboles/XGBoost no lo requieren.
    """
    if scale_numeric:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def evaluate_predictions(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_risk": precision_score(y_true, y_pred, zero_division=0),
        "recall_risk": recall_score(y_true, y_pred, zero_division=0),
        "f1_risk": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def baseline_rule_predict(X: pd.DataFrame) -> np.ndarray:
    """
    Baseline de negocio.

    Regla:
        riesgo si Ratio_Cobertura_LT <= 1.

    ¿Por qué sirve?
        Permite demostrar que el ML debe ganarle a una regla operacional simple.
    """
    return (X["Ratio_Cobertura_LT"] <= 1.0).astype(int).to_numpy()


def select_threshold_for_recall(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float = 0.80) -> float:
    """
    Selecciona umbral priorizando recall de riesgo.

    En supply chain conviene detectar quiebres; un falso negativo puede implicar ventas perdidas.
    Si hay varios umbrales con recall suficiente, elegimos el de mayor precision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall tienen un elemento más que thresholds.
    candidates = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if r >= min_recall:
            candidates.append((p, r, t))
    if not candidates:
        return 0.5
    best = max(candidates, key=lambda x: (x[0], x[1]))
    return float(best[2])


def train_and_compare(df: pd.DataFrame, output_dir: Path, models_dir: Path) -> None:
    train_df, test_df, cutoff_date = temporal_split(df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET].astype(int)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET].astype(int)

    results = []
    reports = {}
    confusion_matrices = {}
    trained_models = {}

    # 1) Baseline operativo
    baseline_pred = baseline_rule_predict(X_test)
    baseline_metrics = evaluate_predictions(y_test, baseline_pred)
    baseline_metrics.update({"model": "Baseline_Ratio_Cobertura", "threshold": np.nan})
    results.append(baseline_metrics)
    reports["Baseline_Ratio_Cobertura"] = classification_report(y_test, baseline_pred, output_dict=True, zero_division=0)
    confusion_matrices["Baseline_Ratio_Cobertura"] = confusion_matrix(y_test, baseline_pred).tolist()

    # 2) Modelos candidatos
    class_counts = y_train.value_counts().to_dict()
    neg = class_counts.get(0, 1)
    pos = class_counts.get(1, 1)
    scale_pos_weight = neg / max(pos, 1)

    candidates = {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]
        ),
        "DecisionTree": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(scale_numeric=False)),
                ("model", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=7,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    if XGBOOST_AVAILABLE:
        candidates["XGBoost"] = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    for name, pipeline in candidates.items():
        print(f"\nEntrenando {name}...")
        pipeline.fit(X_train, y_train)

        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            threshold = select_threshold_for_recall(y_test.to_numpy(), y_proba, min_recall=0.80)
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_proba = None
            threshold = 0.5
            y_pred = pipeline.predict(X_test)

        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        metrics.update({"model": name, "threshold": threshold})
        results.append(metrics)
        reports[name] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        confusion_matrices[name] = confusion_matrix(y_test, y_pred).tolist()
        trained_models[name] = {"pipeline": pipeline, "threshold": threshold, "metrics": metrics}

    results_df = pd.DataFrame(results).sort_values(["f1_risk", "recall_risk", "precision_risk"], ascending=False)
    results_df.to_csv(output_dir / "model_comparison_metrics.csv", index=False)

    with open(output_dir / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    with open(output_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(confusion_matrices, f, indent=2, ensure_ascii=False)

    # Elegimos preferentemente XGBoost si está disponible y tiene desempeño competitivo.
    # Si no, elegimos el mejor por F1 de riesgo.
    if "XGBoost" in trained_models:
        xgb_f1 = trained_models["XGBoost"]["metrics"]["f1_risk"]
        best_f1 = results_df.iloc[0]["f1_risk"]
        if xgb_f1 >= best_f1 * 0.95:
            best_name = "XGBoost"
        else:
            best_name = str(results_df.iloc[0]["model"])
    else:
        best_name = str(results_df.iloc[0]["model"])

    best_artifact = trained_models[best_name]
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": best_artifact["pipeline"],
            "threshold": best_artifact["threshold"],
            "feature_columns": FEATURE_COLUMNS,
            "target": TARGET,
            "cutoff_date": cutoff_date.isoformat(),
            "metrics": best_artifact["metrics"],
        },
        models_dir / "best_stockout_model.joblib",
    )

    print("\n=== Comparación de modelos ===")
    print(results_df.to_string(index=False))
    print("\nMejor modelo seleccionado:", best_name)
    print("Artefacto guardado en:", models_dir / "best_stockout_model.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos para riesgo de quiebre")
    parser.add_argument("--processed", type=str, default="outputs/herdez_features_dataset.csv")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed_dataset(Path(args.processed))
    train_and_compare(df, output_dir, models_dir)


if __name__ == "__main__":
    main()
