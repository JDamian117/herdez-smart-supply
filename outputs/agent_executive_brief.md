# Herdez Smart-Supply — Brief ejecutivo generado por fallback determinista

## 1. Resumen ejecutivo
Se analizaron 8 alertas priorizadas de riesgo de quiebre. El beneficio neto potencial agregado en estas alertas es de $561,295.24 MXN, comparando pérdida esperada contra costo logístico de transferencia.

## 2. Acciones recomendadas
- **HZ-Salsa-Roja-200g en CEDI_Occidente**: TRANSFER_INVENTORY. Riesgo 100.0%, transferir 160 unidades desde CEDI_Bajio, beneficio neto $73,112.04 MXN.
- **HZ-Salsa-Verde-200g en CEDI_Occidente**: TRANSFER_INVENTORY. Riesgo 100.0%, transferir 187 unidades desde CEDI_Bajio, beneficio neto $73,053.24 MXN.
- **HZ-Salsa-Roja-200g en CEDI_Occidente**: TRANSFER_INVENTORY. Riesgo 100.0%, transferir 308 unidades desde CEDI_Norte, beneficio neto $71,908.20 MXN.
- **HZ-Salsa-Roja-200g en CEDI_Occidente**: TRANSFER_INVENTORY. Riesgo 100.0%, transferir 429 unidades desde CEDI_Sur, beneficio neto $70,356.53 MXN.
- **HZ-Salsa-Verde-200g en CEDI_Bajio**: TRANSFER_INVENTORY. Riesgo 100.0%, transferir 604 unidades desde CEDI_Occidente, beneficio neto $69,143.69 MXN.

## 3. Interpretación de negocio
Las recomendaciones priorizan evitar ventas perdidas cuando la pérdida esperada por quiebre supera el costo logístico. Esto protege el nivel de servicio y evita transferencias innecesarias.

## 4. Controles y auditabilidad
- El LLM no inventa la decisión financiera.
- El riesgo viene del modelo predictivo.
- El costo-beneficio viene del motor determinista.
- Las reglas de política evitan trasladar el problema a otro CEDI.
- La recomendación puede revisarse con human-in-the-loop antes de ejecutar la transferencia.

## 5. Riesgos a validar antes de producción
- Confirmar inventario real-time del CEDI origen.
- Validar ventanas reales de transporte y disponibilidad de unidades.
- Ajustar umbrales de riesgo según nivel de servicio objetivo.
- Monitorear drift del modelo cuando cambien demanda, promociones o condiciones logísticas.