# Responsible AI Assessment - SafePlay AI

## 1. Resumen ejecutivo

SafePlay AI es un prototipo de machine learning explicable para detectar patrones tempranos de juego de riesgo y recomendar intervenciones responsables.

El sistema utiliza datos sintéticos de comportamiento de usuarios, sesiones, depósitos, pérdidas y actividad temporal para generar alertas preventivas.

El objetivo principal no es automatizar decisiones sensibles, sino apoyar a equipos humanos en la priorización de revisiones preventivas.

Principio central:

> SafePlay AI debe utilizarse para proteger al usuario, no para maximizar su gasto.

---

## 2. Finalidad responsable del sistema

SafePlay AI se diseña con una finalidad preventiva:

* identificar cambios de comportamiento potencialmente preocupantes;
* explicar por qué se genera una alerta;
* recomendar intervenciones proporcionales;
* evitar presión comercial sobre usuarios con señales de riesgo;
* mantener revisión humana en casos sensibles;
* documentar limitaciones y riesgos de uso.

El sistema no diagnostica ludopatía ni sustituye procesos clínicos, regulatorios o internos de juego responsable.

---

## 3. Decisiones que el sistema puede apoyar

SafePlay AI puede apoyar decisiones como:

* priorizar usuarios para revisión preventiva;
* mostrar mensajes generales de juego responsable;
* sugerir límites voluntarios de depósito o tiempo;
* reducir presión promocional;
* excluir temporalmente de campañas comerciales;
* registrar alertas para auditoría;
* escalar casos de alto riesgo a revisión humana.

---

## 4. Decisiones que el sistema no debe tomar automáticamente

SafePlay AI no debe tomar automáticamente decisiones como:

* diagnosticar adicción o ludopatía;
* bloquear usuarios permanentemente;
* suspender cuentas sin revisión;
* imponer sanciones;
* limitar derechos del usuario sin supervisión;
* reportar usuarios a terceros;
* realizar segmentación comercial agresiva;
* aumentar promociones a usuarios con señales de riesgo.

---

## 5. Riesgo 1: falsos positivos

### Descripción

Un falso positivo ocurre cuando el sistema clasifica como medio o alto riesgo a un usuario cuyo comportamiento tiene una explicación benigna.

Ejemplos:

* un usuario juega más durante vacaciones;
* un usuario realiza varios depósitos pequeños por preferencia personal;
* un usuario participa en eventos puntuales;
* un usuario tiene actividad nocturna por horario laboral.

### Impacto potencial

* preocupación innecesaria del usuario;
* intervención no deseada;
* carga operativa para el equipo de revisión;
* posible percepción de vigilancia excesiva.

### Mitigaciones

* intervención proporcional según nivel de riesgo;
* revisión humana para riesgo alto;
* mensajes preventivos no punitivos;
* registro de razones de alerta;
* análisis periódico de precision;
* posibilidad de ajustar umbrales.

---

## 6. Riesgo 2: falsos negativos

### Descripción

Un falso negativo ocurre cuando el sistema no detecta a un usuario con señales relevantes de riesgo.

### Impacto potencial

* retraso en la intervención preventiva;
* pérdida de oportunidad para sugerir límites o pausas;
* posible agravamiento de comportamiento problemático.

### Mitigaciones

* priorizar recall en la clase de alto riesgo;
* usar una capa adicional de detección de anomalías;
* revisar evolución temporal, no solo valores absolutos;
* monitorizar drift de comportamiento;
* revisar periódicamente reglas y features;
* incorporar feedback de revisores humanos en futuras versiones.

---

## 7. Riesgo 3: sesgos por canal o perfil

### Descripción

El modelo podría generar más alertas para ciertos segmentos aunque el riesgo real no sea mayor.

Posibles dimensiones a revisar:

* canal online vs retail;
* edad sintética por banda;
* provincia;
* preferencia de producto;
* antigüedad del usuario.

### Impacto potencial

* tratamiento desigual entre grupos;
* sobrealerta en ciertos canales;
* infraalerta en otros segmentos;
* pérdida de confianza en el sistema.

### Mitigaciones

* evaluar métricas por segmento;
* revisar distribución de alertas por canal y edad;
* evitar variables sensibles no necesarias;
* documentar variables usadas;
* realizar análisis de fairness antes de producción;
* validar criterios con expertos de negocio y compliance.

---

## 8. Riesgo 4: uso indebido para marketing

### Descripción

Un sistema que identifica patrones de vulnerabilidad podría utilizarse indebidamente para aumentar campañas comerciales sobre usuarios de riesgo.

Este es uno de los riesgos éticos más importantes del proyecto.

### Impacto potencial

* explotación de usuarios vulnerables;
* incremento de daño al usuario;
* incumplimiento de principios de juego responsable;
* riesgo reputacional y regulatorio.

### Mitigaciones

* usuarios medium y high se marcan para reducción o supresión comercial;
* la documentación declara explícitamente que el sistema no debe usarse para maximizar gasto;
* las acciones recomendadas son protectoras;
* las alertas deben quedar registradas para auditoría;
* separación entre casos de riesgo y campañas comerciales;
* revisión interna de cualquier uso del score fuera del equipo responsable.

---

## 9. Riesgo 5: interpretar el score como diagnóstico

### Descripción

Existe el riesgo de que stakeholders interpreten el score como una medida clínica o definitiva de adicción.

### Impacto potencial

* decisiones injustas o desproporcionadas;
* daño reputacional;
* uso incorrecto por equipos no técnicos;
* falsa sensación de certeza.

### Mitigaciones

* lenguaje explícito: “señales conductuales compatibles con riesgo”, no diagnóstico;
* Model Card con usos no previstos;
* dashboard con nota de gobernanza visible;
* revisión humana obligatoria para alto riesgo;
* formación de usuarios internos del sistema;
* evitar etiquetas clínicas en interfaz y documentación.

---

## 10. Riesgo 6: automatización excesiva

### Descripción

El sistema podría integrarse erróneamente como una herramienta de decisión automática.

### Impacto potencial

* bloqueo o restricción injustificada;
* ausencia de contexto humano;
* escalado de errores del modelo;
* incumplimiento de buenas prácticas de IA responsable.

### Mitigaciones

* las recomendaciones se formulan como apoyo, no decisión;
* las alertas de alto riesgo requieren revisión humana;
* la interfaz muestra drivers explicables;
* las acciones sensibles no se automatizan;
* se registran motivos de alerta y acciones recomendadas.

---

## 11. Riesgo 7: privacidad y minimización de datos

### Descripción

Aunque el MVP usa datos sintéticos, un sistema real podría manejar datos sensibles de comportamiento.

### Impacto potencial

* exposición de información sensible;
* uso no autorizado;
* riesgo de reidentificación;
* incumplimiento normativo.

### Mitigaciones recomendadas en producción

* minimización de datos;
* seudonimización de usuarios;
* control de acceso por rol;
* cifrado en reposo y tránsito;
* logging de accesos;
* retención limitada;
* separación entre datos de juego responsable y marketing;
* revisión legal y evaluación de impacto.

---

## 12. Riesgo 8: dependencia de etiquetas sintéticas

### Descripción

Las etiquetas del MVP se generan mediante reglas sintéticas. Esto permite construir y demostrar el pipeline, pero no garantiza validez en datos reales.

### Impacto potencial

* sobreestimación del rendimiento;
* patrones demasiado limpios;
* menor generalización en producción;
* falsa confianza en el modelo.

### Mitigaciones

* declarar claramente que el dataset es sintético;
* no presentar métricas como evidencia productiva;
* validar con datos históricos reales antes de producción;
* calibrar umbrales con expertos;
* incorporar feedback de revisión humana;
* comparar contra reglas internas existentes.

---

## 13. Controles incorporados en el MVP

El MVP ya incorpora varios controles de IA responsable:

```text
[x] Datos sintéticos sin información personal real
[x] Exclusión de variables de leakage
[x] Comparativa contra baseline interpretable
[x] Métricas centradas en alto riesgo
[x] Explicabilidad global con SHAP
[x] Explicabilidad individual por usuario
[x] Reglas de intervención proporcional
[x] Revisión humana para riesgo alto
[x] Supresión comercial para medium/high
[x] Dashboard con nota de gobernanza
[x] Model Card
[x] Documentación de limitaciones
```

---

## 14. Controles pendientes antes de producción

Antes de cualquier uso real, deberían añadirse:

```text
[ ] Validación con expertos de juego responsable
[ ] Revisión legal y regulatoria
[ ] Evaluación de impacto de protección de datos
[ ] Validación con datos históricos reales
[ ] Calibración de probabilidades
[ ] Evaluación formal de fairness
[ ] Monitorización de drift
[ ] Feedback loop de revisores humanos
[ ] Procedimiento de apelación o revisión
[ ] Políticas de retención de datos
[ ] Auditoría de usos comerciales
[ ] Pruebas de robustez temporal
[ ] Tests automatizados del pipeline
```

---

## 15. Principios de diseño responsable

### 1. Protección del usuario

El sistema debe priorizar la prevención de daño y la promoción de herramientas de juego responsable.

### 2. Proporcionalidad

La intervención debe ser proporcional al nivel de riesgo estimado.

### 3. Supervisión humana

Las decisiones sensibles deben revisarse por personas formadas.

### 4. Explicabilidad

Cada alerta debe incluir drivers comprensibles para negocio y revisión.

### 5. Trazabilidad

Las alertas y acciones recomendadas deben quedar registradas para auditoría.

### 6. No explotación comercial

Las señales de riesgo no deben utilizarse para aumentar gasto, promociones o presión comercial.

### 7. Minimización de datos

El sistema debe utilizar solo las variables necesarias para el propósito preventivo.

### 8. Honestidad sobre limitaciones

El modelo no debe presentarse como diagnóstico ni como solución definitiva.

---

## 16. Matriz de riesgo y mitigación

| Riesgo                              | Severidad | Probabilidad MVP | Mitigación principal                        |
| ----------------------------------- | --------: | ---------------: | ------------------------------------------- |
| Falsos positivos                    |     Media |            Media | Intervención proporcional y revisión humana |
| Falsos negativos                    |      Alta |            Media | Priorizar recall y usar anomalías           |
| Sesgo por canal o perfil            |     Media |            Media | Métricas segmentadas y fairness review      |
| Uso indebido para marketing         |      Alta |       Baja-Media | Supresión comercial medium/high             |
| Score interpretado como diagnóstico |      Alta |            Media | Gobernanza, disclaimers y formación         |
| Automatización excesiva             |      Alta |             Baja | Revisión humana obligatoria                 |
| Privacidad                          |      Alta |      Baja en MVP | Datos sintéticos y minimización             |
| Dependencia de etiquetas sintéticas |     Media |             Alta | Documentación y validación futura           |

---

## 17. Política de intervención recomendada

| Nivel | Acción                              | Revisión humana | Supresión comercial | Auditoría |
| ----- | ----------------------------------- | --------------: | ------------------: | --------: |
| Bajo  | Sin intervención individual         |              No |                  No |        No |
| Medio | Mensaje preventivo y monitorización |  No obligatoria |                  Sí |        Sí |
| Alto  | Revisión humana y pausa preventiva  |              Sí |                  Sí |        Sí |

---

## 18. Lenguaje recomendado

Usar:

* “señales tempranas de riesgo”;
* “patrones conductuales compatibles con riesgo”;
* “priorización para revisión preventiva”;
* “score de apoyo a revisión”;
* “intervención proporcional”.

Evitar:

* “diagnóstico de ludopatía”;
* “detección de adictos”;
* “usuario problemático”;
* “bloqueo automático”;
* “predicción clínica”.

---

## 19. Recomendación final

SafePlay AI es adecuado como prototipo demostrativo de IA responsable aplicada al sector gaming.

Sus puntos fuertes son:

* enfoque preventivo;
* explicabilidad individual;
* acciones proporcionales;
* revisión humana;
* documentación de limitaciones;
* separación explícita entre protección del usuario y explotación comercial.

No debería utilizarse en producción sin validación experta, revisión legal, datos reales representativos, evaluación de fairness y mecanismos de auditoría robustos.

---

## 20. Declaración de principio

SafePlay AI no busca maximizar el valor económico de usuarios vulnerables.

Busca ayudar a detectar señales tempranas, explicar alertas y facilitar intervenciones responsables orientadas a la protección del usuario.
