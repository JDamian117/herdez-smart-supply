# Herdez Smart-Supply — Brief Ejecutivo A2A-lite

## Resumen ejecutivo
El sistema Smart-Supply priorizó 8 alertas. El beneficio neto potencial agregado es $561,295.24 MXN. La recomendación se basa en riesgo ML, costo-beneficio y validación de política.

## Plan recomendado
- Ejecutar TRANSFER_INVENTORY para HZ-Salsa-Roja-200g en CEDI_Occidente ($73,112.04 MXN de beneficio neto).
- Ejecutar TRANSFER_INVENTORY para HZ-Salsa-Verde-200g en CEDI_Occidente ($73,053.24 MXN de beneficio neto).
- Ejecutar TRANSFER_INVENTORY para HZ-Salsa-Roja-200g en CEDI_Occidente ($71,908.20 MXN de beneficio neto).

## Impacto esperado
Reducir ventas perdidas por quiebre y evitar movimientos logísticos no rentables, manteniendo trazabilidad de cada recomendación.

## Lectura de riesgo
Se analizaron 8 alertas priorizadas. El riesgo se concentra en combinaciones SKU/CEDI con alto beneficio neto potencial.

### Patrones críticos
- HZ-Salsa-Roja-200g en CEDI_Occidente con riesgo 100.0% y beneficio neto $73,112.04 MXN
- HZ-Salsa-Verde-200g en CEDI_Occidente con riesgo 100.0% y beneficio neto $73,053.24 MXN
- HZ-Salsa-Roja-200g en CEDI_Occidente con riesgo 100.0% y beneficio neto $71,908.20 MXN

## Lectura financiera
El beneficio neto agregado de las alertas enviadas es $561,295.24 MXN; la pérdida esperada agregada es $0.00 MXN.

### Mejores acciones financieras
- TRANSFER_INVENTORY para HZ-Salsa-Roja-200g en CEDI_Occidente: beneficio neto $73,112.04 MXN
- TRANSFER_INVENTORY para HZ-Salsa-Verde-200g en CEDI_Occidente: beneficio neto $73,053.24 MXN
- TRANSFER_INVENTORY para HZ-Salsa-Roja-200g en CEDI_Occidente: beneficio neto $71,908.20 MXN

## Validación y crítica
Revisión humana requerida: No

### Notas críticas
- La decisión final debe conservar la regla: no transferir si el CEDI origen queda en riesgo.
- Las acciones de reabasto urgente se comunican como escalamiento, no como transferencia automática.
- fallback deterministic mode

## Traza técnica
XGBoost genera riesgo; 03_decision_engine calcula pérdida esperada, costo de transferencia y beneficio neto; los agentes A2A-lite producen artefactos de riesgo, costo, política y comunicación ejecutiva.

## Preguntas probables de entrevista
- ¿Por qué no dejar que el LLM decida? Porque los números críticos son deterministas y auditables.
- ¿Por qué A2A? Porque permite separar agentes por dominio y preparar interoperabilidad enterprise.
- ¿Cómo escala a GCP? BigQuery para datos, Vertex AI para modelos, Cloud Run/Agent Platform para agentes y A2A para interoperabilidad.

## Arquitectura A2A-lite aplicada
- AgentCards describen capacidades y artefactos esperados por agente.
- A2ATask agrupa la unidad de trabajo completa.
- Cada agente produce un Artifact estructurado y auditable.
- En producción, cada agente puede exponerse como servicio A2A real.