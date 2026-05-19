# Model Card - SafePlay AI

## 1. Resumen del modelo

SafePlay AI es un prototipo de machine learning explicable diseñado para detectar patrones tempranos de juego de riesgo en usuarios de plataformas de gaming.

El sistema analiza variables de comportamiento agregadas a nivel usuario, como frecuencia de sesiones, depósitos, pérdidas, actividad nocturna, duración de sesiones y cambios recientes frente al histórico.

El modelo principal genera una probabilidad de pertenencia a tres niveles de riesgo:

```text
low
medium
high
```

El output se utiliza para priorizar revisión preventiva y recomendar intervenciones responsables proporcionales.

---

## 2. Uso previsto

SafePlay AI está diseñado como sistema de apoyo para equipos de responsible gaming, compliance, riesgo o analítica operativa.

Usos previstos:

* priorizar usuarios para revisión preventiva;
* identificar cambios bruscos de comportamiento;
* explicar los principales factores detrás de cada alerta;
* recomendar intervenciones proporcionales;
* apoyar auditoría interna y trazabilidad;
* demostrar un pipeline de IA responsable con datos sintéticos.

El sistema debe utilizarse como apoyo a revisión humana, no como mecanismo de decisión automática.

---

## 3. Uso no previsto

SafePlay AI no debe utilizarse para:

* diagnosticar ludopatía o cualquier condición clínica;
* imponer sanciones automáticas;
* bloquear usuarios automáticamente sin revisión humana;
* tomar decisiones regulatorias sin validación experta;
* segmentar usuarios vulnerables para campañas comerciales;
* maximizar gasto, frecuencia de juego o depósitos;
* sustituir procesos internos de juego responsable;
* evaluar personas reales sin validación legal, ética y técnica previa.

---

## 4. Tipo de modelo

El MVP compara tres enfoques:

### Logistic Regression

Modelo baseline interpretable utilizado como punto de comparación.

### XGBoost Classifier

Modelo principal supervisado. Se selecciona porque captura relaciones no lineales entre variables temporales, monetarias y conductuales.

### Isolation Forest

Capa no supervisada de anomalías. Se utiliza para identificar usuarios con patrones atípicos que podrían requerir revisión aunque no encajen exactamente en la etiqueta sintética.

---

## 5. Datos utilizados

Los datos utilizados son completamente sintéticos.

No se emplean:

* nombres reales;
* DNI;
* emails;
* direcciones;
* teléfonos;
* identificadores personales reales;
* datos financieros reales;
* datos clínicos;
* datos de usuarios reales.

El dataset simula:

* usuarios ficticios;
* sesiones de juego;
* importes apostados;
* premios;
* pérdidas netas;
* depósitos;
* actividad nocturna;
* retornos rápidos después de pérdidas;
* uso o cancelación de límites;
* preferencias de canal.

---

## 6. Ventana temporal

El prototipo simula aproximadamente 180 días de actividad.

Las variables se calculan usando ventanas como:

* últimos 7 días;
* últimos 14 días;
* últimos 30 días;
* histórico previo a los últimos 7 días;
* histórico de 90 días anterior a la ventana reciente.

El objetivo es detectar cambios recientes frente al comportamiento histórico del usuario.

---

## 7. Variables usadas

El modelo utiliza variables agregadas a nivel usuario.

### Variables de frecuencia

* `sessions_7d`
* `sessions_14d`
* `sessions_30d`
* `active_days_30d`
* `frequency_increase_ratio`
* `days_since_last_session`

### Variables monetarias

* `total_wagered_7d`
* `total_wagered_30d`
* `net_loss_7d`
* `net_loss_30d`
* `avg_bet_amount_30d`
* `stake_increase_ratio`
* `deposit_amount_7d`
* `deposit_count_7d`
* `deposit_increase_ratio`
* `loss_increase_ratio`

### Variables de comportamiento

* `avg_session_duration_30d`
* `max_session_duration_30d`
* `night_sessions_ratio_30d`
* `loss_chasing_events_30d`
* `product_switch_count_30d`
* `channel_switch_count_30d`

### Variables de protección

* `limit_configured`
* `cancelled_limit_recently`
* `self_exclusion_attempt`
* `cooling_off_used`

### Variables contextuales no personales

* `age_band`
* `province`
* `channel_preference`
* `marketing_opt_in`

Estas variables son sintéticas y se usan únicamente para construir un MVP demostrativo.

---

## 8. Variables excluidas

Para evitar riesgos de privacidad, leakage o uso indebido, se excluyen:

* nombre;
* email;
* DNI;
* dirección;
* teléfono;
* datos personales reales;
* datos clínicos;
* `user_id` como variable predictiva;
* `synthetic_profile`;
* `risk_score_rule`;
* `flag_high_loss_growth`;
* `flag_high_frequency_growth`;
* `flag_high_deposit_growth`;
* `flag_loss_chasing_events`;
* `flag_high_night_activity`;
* `flag_long_sessions`.

Las columnas `risk_score_rule` y `flag_*` codifican directamente la regla sintética de etiquetado, por lo que no deben entrenar el modelo.

---

## 9. Etiquetado

Las etiquetas se generan mediante una regla sintética transparente basada en señales de riesgo conductual.

La regla considera:

* crecimiento reciente de pérdidas;
* crecimiento de frecuencia;
* crecimiento de depósitos;
* retornos rápidos después de pérdidas;
* actividad nocturna;
* sesiones largas;
* cancelación reciente de límites.

Resultado:

```text
low: comportamiento sin señales relevantes o con señales débiles aisladas
medium: señales tempranas que justifican monitorización
high: acumulación de señales que requiere revisión humana
```

Importante: estas etiquetas son sintéticas y sirven para construir un prototipo. En producción, deberían validarse con expertos, datos históricos reales y criterios regulatorios internos.

---

## 10. Rendimiento del modelo

El modelo principal seleccionado es XGBoost.

Resultados obtenidos en el conjunto de test:

```text
Precision high: 0.9608
Recall high:    0.9515
F1 high:        0.9561
PR-AUC high:    0.9927
Macro F1:       0.9779
Weighted F1:    0.9948
```

Comparación con baseline:

```text
Logistic Regression
Precision high: 0.7087
Recall high:    0.8738
F1 high:        0.7826
PR-AUC high:    0.9070
```

Interpretación:

XGBoost mejora sustancialmente al baseline en precision, recall, F1 y PR-AUC para la clase de alto riesgo.

Dado que las etiquetas son sintéticas, el rendimiento alto es esperable. Estos resultados no deben interpretarse como evidencia de rendimiento clínico, regulatorio o productivo sobre usuarios reales.

---

## 11. Métricas priorizadas

La métrica principal es recall en la clase de alto riesgo.

Justificación:

En un contexto de juego responsable, es especialmente importante reducir falsos negativos, es decir, usuarios con señales relevantes que el sistema no detecta.

También se monitoriza precision en alto riesgo para evitar saturar al equipo de revisión con demasiados falsos positivos.

Métricas reportadas:

* recall de clase alto riesgo;
* precision de clase alto riesgo;
* F1-score de clase alto riesgo;
* PR-AUC de clase alto riesgo;
* macro F1;
* weighted F1;
* matriz de confusión.

---

## 12. Explicabilidad

El sistema utiliza SHAP para explicar el modelo principal.

La explicabilidad se ofrece en dos niveles:

### Explicabilidad global

Identifica las variables que más influyen en la clasificación de riesgo alto a nivel agregado.

Principales drivers observados:

* incremento de depósitos frente al histórico;
* pérdida neta reciente elevada;
* retornos rápidos después de pérdidas;
* aumento del importe apostado en los últimos 7 días;
* incremento reciente de frecuencia frente al histórico;
* incremento de pérdidas frente al histórico;
* actividad nocturna recurrente;
* sesiones más largas.

### Explicabilidad individual

Para cada usuario, se generan los principales factores que empujan la predicción hacia riesgo alto.

Ejemplo:

```text
Usuario: U005307
Principales drivers:
1. retornos rápidos después de pérdidas
2. actividad nocturna recurrente
3. pérdida neta reciente elevada
4. incremento de depósitos frente al histórico
5. incremento reciente de frecuencia frente al histórico
```

---

## 13. Salida del sistema

Para cada usuario, el sistema puede devolver:

```yaml
user_id: U005307
risk_score_high: 0.9999
risk_level: high
main_drivers:
  - retornos rápidos después de pérdidas
  - actividad nocturna recurrente
  - pérdida neta reciente elevada
recommended_action:
  - excluir de campañas comerciales
  - revisión humana
  - mensaje de pausa preventiva
  - sugerir herramientas de autoexclusión o enfriamiento
```

---

## 14. Reglas de intervención

### Riesgo bajo

Acción recomendada:

* sin intervención individual;
* comunicación general sobre juego responsable;
* sugerencia opcional de límites.

### Riesgo medio

Acción recomendada:

* mensaje preventivo;
* sugerir límites voluntarios;
* reducir presión promocional;
* monitorizar evolución;
* registrar alerta para auditoría.

### Riesgo alto

Acción recomendada:

* excluir de campañas comerciales;
* activar revisión humana;
* mostrar mensaje de pausa;
* sugerir herramientas de autoexclusión o enfriamiento;
* registrar alerta para auditoría.

---

## 15. Supervisión humana

SafePlay AI requiere supervisión humana en los casos de alto riesgo.

El sistema no debe ejecutar automáticamente acciones sensibles como:

* bloqueo permanente;
* suspensión de cuenta;
* diagnóstico;
* restricción irreversible;
* reporte externo;
* decisiones regulatorias.

La función del modelo es priorizar revisión y aportar evidencia explicable.

---

## 16. Riesgos conocidos

### Falsos positivos

Usuarios pueden ser marcados como riesgo medio o alto aunque su comportamiento tenga una explicación benigna.

Mitigación:

* revisión humana;
* intervención proporcional;
* uso de mensajes preventivos no punitivos;
* auditoría de alertas.

### Falsos negativos

Usuarios con señales de riesgo pueden no ser detectados.

Mitigación:

* priorizar recall alto riesgo;
* capa adicional de anomalías;
* monitorización temporal;
* revisión periódica de reglas y features.

### Sesgos por canal o perfil

El modelo podría comportarse de forma diferente en usuarios online vs retail, o entre grupos de edad sintéticos.

Mitigación:

* análisis de métricas por segmento;
* exclusión de variables innecesariamente sensibles;
* revisión de fairness antes de producción.

### Uso indebido comercial

Existe el riesgo de usar señales de vulnerabilidad para campañas comerciales.

Mitigación:

* supresión comercial para usuarios medium y high;
* documentación explícita de usos no previstos;
* auditoría de decisiones.

---

## 17. Limitaciones

1. El dataset es sintético.
2. Las etiquetas son generadas por reglas, no por evaluación experta real.
3. Los resultados no prueban validez clínica.
4. Los resultados no prueban cumplimiento regulatorio.
5. El modelo no detecta adicción.
6. El comportamiento real de usuarios podría ser más complejo.
7. No se han realizado pruebas de drift.
8. No se ha realizado validación con expertos.
9. No se ha implementado feedback loop de revisores humanos.
10. No se ha realizado evaluación formal de fairness.

---

## 18. Consideraciones de privacidad

El MVP está diseñado para evitar datos personales reales.

En un entorno productivo deberían aplicarse controles adicionales:

* minimización de datos;
* seudonimización;
* control de acceso;
* logging de consultas;
* retención limitada;
* evaluación de impacto;
* revisión legal y regulatoria;
* separación entre datos de juego responsable y marketing.

---

## 19. Monitorización recomendada

Antes de cualquier uso real, se recomienda monitorizar:

* distribución de scores;
* tasa de alertas por semana;
* precision estimada tras revisión humana;
* recall estimado con casos confirmados;
* drift de variables;
* drift de predicciones;
* métricas por canal;
* métricas por grupo de edad;
* volumen de falsos positivos;
* saturación del equipo de revisión;
* tasa de usuarios excluidos de campañas.

---

## 20. Recomendaciones antes de producción

Antes de considerar un entorno real, serían necesarias al menos estas acciones:

1. Validación con expertos de juego responsable.
2. Revisión legal y regulatoria.
3. Definición formal de criterios de alerta.
4. Validación con datos históricos reales.
5. Calibración de probabilidades.
6. Evaluación de fairness y sesgos.
7. Diseño de feedback loop con revisores humanos.
8. Pruebas de robustez temporal.
9. Monitorización de drift.
10. Auditoría de uso comercial.
11. Documentación operativa para equipos internos.
12. Mecanismo de apelación o revisión de decisiones sensibles.

---

## 21. Principio central

SafePlay AI debe utilizarse para proteger al usuario, no para maximizar su gasto.

El score de riesgo debe entenderse como una señal preventiva explicable, no como diagnóstico ni como decisión automática.
