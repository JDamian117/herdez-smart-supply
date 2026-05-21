# Herdez Smart-Supply - Decision Engine

## ¿Qué hace este módulo?
Convierte alertas de riesgo de quiebre generadas por el modelo ML en recomendaciones operativas basadas en costo-beneficio.

## Principio de arquitectura
El LLM no toma decisiones financieras directamente. La predicción, el cálculo de pérdida, el costo logístico y la validación de política se calculan de forma determinista y auditable.

## Parámetros de negocio
- Safety factor de demanda: 1.1
- Cobertura mínima del CEDI origen: 1.2
- Umbral riesgo alto: 0.7
- Umbral riesgo medio: 0.4

## Resumen de recomendaciones
- TRANSFER_INVENTORY: 41
- EXPEDITE_REPLENISHMENT_OR_REVIEW: 9

- Pérdida evitada estimada: $2,089,015.82
- Costo logístico estimado: $187,612.76
- Beneficio neto estimado: $1,901,403.06

## Top 5 recomendaciones
| Fecha      | SKU_ID              | CEDI_Destino   | Nivel_Riesgo   | Accion_Recomendada               | CEDI_Origen_Recomendado   |   Unidades_A_Transferir |   Beneficio_Neto |
|:-----------|:--------------------|:---------------|:---------------|:---------------------------------|:--------------------------|------------------------:|-----------------:|
| 2024-03-11 | HZ-Salsa-Roja-200g  | CEDI_Occidente | Alto           | TRANSFER_INVENTORY               | CEDI_Sur                  |                     611 |          42437.1 |
| 2024-03-10 | HZ-Salsa-Verde-200g | CEDI_Occidente | Alto           | EXPEDITE_REPLENISHMENT_OR_REVIEW |                           |                       0 |              0   |
| 2024-04-24 | HZ-Salsa-Roja-200g  | CEDI_Bajio     | Alto           | TRANSFER_INVENTORY               | CEDI_Sur                  |                     615 |          40861.7 |
| 2024-03-18 | HZ-Salsa-Verde-200g | CEDI_Occidente | Alto           | TRANSFER_INVENTORY               | CEDI_Norte                |                     896 |          64674.5 |
| 2024-03-17 | HZ-Salsa-Roja-200g  | CEDI_Occidente | Alto           | TRANSFER_INVENTORY               | CEDI_Bajio                |                     249 |          23468.9 |

## Mensaje para entrevista
> El modelo predictivo identifica el riesgo; el motor determinista calcula si conviene actuar; el agente GenAI explica la decisión y permite conversar con el usuario.