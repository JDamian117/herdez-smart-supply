# Herdez Smart-Supply - Prototipo Local-First

Este proyecto implementa la primera parte del caso técnico: análisis de datos, creación del target de riesgo de quiebre, feature engineering, comparación de modelos y un motor determinista de decisión costo-beneficio.

## Estructura

```text
herdez_smart_supply_project/
├── data/
│   └── Data_Prueba_Tecnica_Herdez_IA.xlsx
├── notebooks/
│   └── Herdez_SmartSupply_EDA_Modelado_Colab.ipynb
├── src/
│   ├── 01_eda_target_features.py
│   ├── 02_train_models.py
│   ├── 03_decision_engine.py
│   └── 04_streamlit_app_skeleton.py
├── outputs/
├── models/
└── requirements.txt
```

## Ejecución local

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python src/01_eda_target_features.py --input data/Data_Prueba_Tecnica_Herdez_IA.xlsx
python src/02_train_models.py --processed outputs/herdez_features_dataset.csv
python src/03_decision_engine.py --processed outputs/herdez_features_dataset.csv --model models/best_stockout_model.joblib
streamlit run src/04_streamlit_app_skeleton.py
```

## Ejecución en Colab

1. Abre `notebooks/Herdez_SmartSupply_EDA_Modelado_Colab.ipynb`.
2. Sube el archivo Excel cuando el notebook lo pida.
3. Ejecuta las celdas en orden.

## Tesis técnica

El prototipo respeta el enfoque Local-First: usa Excel, Python, XGBoost, Streamlit y un motor determinista de decisión. La arquitectura puede escalar a GCP con BigQuery, Vertex AI, Agent Platform y A2A.


## Fase 4 - Sistema multiagente

Este paso consume las recomendaciones del motor determinista y genera un brief ejecutivo con CrewAI/LangChain/Gemini o fallback determinista.

```bash
python src/04_agent_system.py --mode auto
```

Para usar Gemini configura una variable de entorno:

```bash
set GEMINI_API_KEY=tu_api_key    # Windows CMD
export GEMINI_API_KEY=tu_api_key # Linux/Mac
```

Modos disponibles:

- `--mode crewai`: fuerza CrewAI multiagente.
- `--mode langchain`: usa una llamada directa con LangChain + Gemini.
- `--mode fallback`: no llama ningún LLM y genera un brief determinista.
- `--mode auto`: intenta CrewAI, luego LangChain, luego fallback.

Archivos generados:

- `outputs/agent_executive_brief.md`
- `outputs/agent_prompt_context.json`
- `outputs/agent_technical_trace.json`

## 04 simplificado recomendado para explicar en entrevista

La versión más fácil de explicar del sistema de agentes es:

```bash
python src/04_agent_system_a2a_simple.py --mode fallback --max-alerts 8
```

Con Gemini, usando fallback automático si falla la API:

```bash
python src/04_agent_system_a2a_simple.py --mode auto --max-alerts 8
```

Esta versión conserva los conceptos principales: A2A-lite, prompts estructurados, Pydantic, temperatura baja y fallback seguro.

## 05 Dashboard Streamlit

Este paso cierra el prototipo funcional. El dashboard muestra:

- KPIs ejecutivos de riesgo y beneficio neto.
- Alertas por SKU/CEDI.
- Recomendaciones del motor costo-beneficio.
- Brief del sistema de agentes A2A-lite.
- Trazabilidad de agentes y artefactos.
- Simulador de transferencia.
- Arquitectura local-first y cloud-ready.

### Ejecutar localmente

```bash
streamlit run src/05_streamlit_dashboard.py
```

### Ejecutar en Streamlit Cloud

El proyecto incluye `app.py` en la raíz, que apunta al dashboard. En Streamlit Cloud selecciona:

```text
app.py
```

Si despliegas desde GitHub, asegúrate de subir también:

```text
requirements.txt
outputs/
models/
src/
data/
```

### Orden completo de ejecución

```bash
python src/01_eda_target_features.py --input data/Data_Prueba_Tecnica_Herdez_IA.xlsx
python src/02_train_models.py --processed outputs/herdez_features_dataset.csv
python src/03_decision_engine.py --processed outputs/herdez_features_dataset.csv --model models/best_stockout_model.joblib
python src/04_agent_system_a2a_simple.py --mode fallback --max-alerts 8
streamlit run src/05_streamlit_dashboard.py
```
