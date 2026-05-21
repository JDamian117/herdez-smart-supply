# Hallazgos EDA - Herdez Smart-Supply

## Resumen del dataset
- filas: 1200
- columnas: 11
- fecha_min: 2024-03-01
- fecha_max: 2024-04-29
- dias_historicos: 60
- num_skus: 5
- num_cedis: 4
- combinaciones_sku_cedi: 20
- nulos_totales: 0
- duplicados_fecha_sku_cedi: 0

## Target de riesgo
- Registros modelables: 1110
- Registros en riesgo: 318
- Tasa de riesgo: 28.65%

## Principales insights
- SKU con mayor tasa de riesgo: HZ-Salsa-Verde-200g (46.85%)
- CEDI con mayor tasa de riesgo: CEDI_Occidente (34.91%)
- Las variables de cobertura durante lead time son candidatas fuertes para predecir quiebre.
- La evaluación debe priorizar recall en la clase de riesgo para evitar ventas perdidas.