business_case.md# Business Case - SafePlay AI

## 1. Resumen ejecutivo

SafePlay AI es un prototipo de sistema explicable para detectar patrones tempranos de juego de riesgo y recomendar intervenciones responsables.

El proyecto está diseñado como una herramienta interna para empresas del sector gaming, apuestas y ocio, donde la protección del usuario, la trazabilidad, la regulación y la reputación corporativa son factores críticos.

El valor del sistema no está solo en predecir riesgo, sino en convertir señales conductuales en alertas explicables y acciones proporcionales.

---

## 2. Problema de negocio

En plataformas de juego, algunos usuarios pueden mostrar cambios de comportamiento compatibles con señales tempranas de riesgo:

* aumento brusco de frecuencia de juego;
* incremento reciente de depósitos;
* sesiones cada vez más largas;
* actividad nocturna recurrente;
* pérdidas concentradas en pocos días;
* retornos rápidos después de pérdidas;
* cancelación reciente de límites;
* cambios fuertes frente al comportamiento histórico.

Sin un sistema analítico, estos patrones pueden ser difíciles de detectar de forma temprana, especialmente cuando el volumen de usuarios y sesiones es elevado.

---

## 3. Oportunidad

SafePlay AI permite pasar de una revisión reactiva a una revisión preventiva basada en datos.

La oportunidad consiste en:

* detectar señales antes de que escalen;
* priorizar usuarios para revisión humana;
* explicar el motivo de cada alerta;
* recomendar intervenciones proporcionales;
* reducir presión comercial sobre usuarios con señales de riesgo;
* mejorar trazabilidad ante auditorías internas o regulatorias;
* reforzar la cultura de juego responsable.

---

## 4. Usuarios internos del sistema

SafePlay AI podría ser utilizado por:

### Equipo de Juego Responsable

Para revisar alertas, analizar perfiles individuales y decidir intervenciones proporcionales.

### Compliance

Para auditar criterios, revisar trazabilidad y asegurar que el sistema no se utiliza para fines indebidos.

### Data Science / Analytics

Para monitorizar performance, drift, sesgos, calibración y calidad de datos.

### CRM / Marketing

No para explotar señales de riesgo, sino para excluir usuarios medium/high de campañas comerciales o reducir presión promocional.

### Dirección

Para visualizar indicadores agregados, volumen de alertas y evolución de riesgo.

---

## 5. Propuesta de valor

SafePlay AI aporta valor en cinco dimensiones:

### 1. Prevención

Permite detectar señales tempranas y activar intervenciones antes de que el patrón empeore.

### 2. Priorización operativa

Convierte miles de usuarios en una cola ordenada de alertas revisables.

### 3. Explicabilidad

Cada alerta incluye drivers comprensibles, como depósitos recientes, pérdidas, actividad nocturna o retorno tras pérdidas.

### 4. Gobernanza

El sistema documenta uso previsto, limitaciones, variables usadas, métricas, matriz de confusión y política de revisión humana.

### 5. Protección reputacional

Refuerza la postura de la empresa ante juego responsable, auditoría y cumplimiento.

---

## 6. Qué hace SafePlay AI

El sistema realiza los siguientes pasos:

1. Genera o recibe datos de usuarios y sesiones.
2. Construye variables temporales agregadas.
3. Calcula señales de cambio frente al histórico.
4. Clasifica usuarios en bajo, medio o alto riesgo.
5. Detecta anomalías con un modelo no supervisado.
6. Explica predicciones con SHAP.
7. Recomienda acciones responsables.
8. Presenta resultados en un dashboard operativo.

---

## 7. Qué no hace SafePlay AI

SafePlay AI no:

* diagnostica ludopatía;
* sustituye a especialistas;
* impone sanciones automáticas;
* bloquea usuarios sin revisión humana;
* decide exclusiones permanentes;
* maximiza gasto de usuarios vulnerables;
* sustituye políticas regulatorias internas;
* garantiza validez sobre datos reales sin validación.

---

## 8. Casos de uso principales

### Caso de uso 1: Monitor de alertas

El equipo de juego responsable accede a una tabla priorizada con usuarios medium/high.

Campos principales:

* usuario;
* nivel de riesgo;
* score;
* canal;
* driver principal;
* acción recomendada;
* necesidad de revisión humana;
* supresión comercial.

### Caso de uso 2: Revisión individual

Un analista selecciona un usuario y visualiza:

* evolución de apuestas;
* depósitos recientes;
* sesiones por semana;
* pérdidas acumuladas;
* drivers SHAP;
* recomendación de intervención.

### Caso de uso 3: Gobernanza del modelo

Compliance o Data Science revisa:

* variables usadas;
* variables excluidas;
* métricas;
* matriz de confusión;
* limitaciones;
* política de revisión humana;
* notas de uso responsable.

### Caso de uso 4: Supresión comercial

Usuarios medium y high se marcan para reducir o bloquear campañas promocionales.

Esto evita que señales de vulnerabilidad se utilicen para incrementar presión comercial.

---

## 9. Métricas de éxito del producto

Para evaluar SafePlay AI como producto interno, se podrían usar métricas como:

### Métricas de modelo

* recall de alto riesgo;
* precision de alto riesgo;
* F1 de alto riesgo;
* PR-AUC;
* estabilidad temporal del score;
* drift de variables.

### Métricas operativas

* número de alertas semanales;
* porcentaje de alertas revisadas;
* tiempo medio de revisión;
* tasa de alertas escaladas;
* carga por analista;
* porcentaje de alertas con explicación completa.

### Métricas de responsible gaming

* usuarios excluidos de campañas;
* usuarios que configuran límites tras intervención;
* usuarios que usan herramientas de pausa;
* reducción de recurrencia de señales tras intervención;
* trazabilidad de acciones tomadas.

### Métricas de gobernanza

* porcentaje de casos high con revisión humana;
* cobertura de auditoría;
* métricas por canal;
* métricas por segmento;
* revisión periódica de umbrales.

---

## 10. Impacto esperado

SafePlay AI puede generar impacto en varias áreas:

### Impacto operativo

Reduce el trabajo manual de exploración y ayuda a priorizar los casos más relevantes.

### Impacto en cumplimiento

Facilita documentación, trazabilidad y evidencia de revisión preventiva.

### Impacto reputacional

Demuestra compromiso con juego responsable y protección del usuario.

### Impacto analítico

Crea una base para monitorización continua, explicabilidad y mejora de modelos.

### Impacto comercial responsable

Permite reducir campañas sobre usuarios con señales de riesgo, evitando incentivos inapropiados.

---

## 11. KPIs del MVP

En la simulación actual, SafePlay AI analiza 10.000 usuarios sintéticos.

Resultados del MVP:

```text
Usuarios analizados: 10.000
Riesgo bajo:         83,8%
Riesgo medio:        12,1%
Riesgo alto:          4,1%
Alertas operativas:  1.620
Revisiones humanas:    409
```

Distribución de intervención:

```text
Bajo:   sin intervención individual
Medio:  mensaje preventivo y monitorización
Alto:   revisión humana y pausa preventiva
```

---

## 12. Priorización de alertas

El sistema genera una cola priorizada de alertas para revisión.

La prioridad se basa en:

* probabilidad de alto riesgo;
* nivel de riesgo predicho;
* detección de anomalías;
* cancelación reciente de límites;
* drivers explicables.

Esto permite ordenar el trabajo del equipo responsable sin automatizar decisiones sensibles.

---

## 13. Explicabilidad como ventaja de negocio

La explicabilidad es clave porque permite responder:

* ¿por qué se ha generado esta alerta?
* ¿qué comportamiento cambió?
* ¿qué variable pesa más?
* ¿qué acción es proporcional?
* ¿cómo justificamos esta revisión internamente?

Ejemplo:

```text
Usuario U005307
Riesgo: Alto
Drivers:
- retornos rápidos después de pérdidas
- actividad nocturna recurrente
- pérdida neta reciente elevada
- incremento de depósitos frente al histórico
- incremento reciente de frecuencia frente al histórico
```

Esta explicación es mucho más útil operativamente que un score aislado.

---

## 14. Encaje con una empresa tipo Grupo Orenes

Para una empresa con actividad en gaming, ocio, apuestas, retail y online, SafePlay AI puede conectar con varios retos reales:

* gestión de usuarios en canales online y presenciales;
* trazabilidad de alertas;
* juego responsable;
* equilibrio entre negocio y protección del cliente;
* reporting interno;
* cumplimiento regulatorio;
* reputación corporativa;
* analítica avanzada aplicada a operaciones.

La narrativa del proyecto encaja especialmente bien porque no se limita a crear un modelo predictivo. Integra producto, gobernanza, explicabilidad y acción operativa.

---

## 15. Diferenciación frente a un modelo genérico

SafePlay AI se diferencia de un proyecto genérico de clasificación porque incluye:

* datos sintéticos diseñados para un caso sectorial concreto;
* features temporales específicas;
* detección de cambios frente al histórico;
* foco en clase minoritaria de alto riesgo;
* explicabilidad global e individual;
* intervención responsable;
* supresión comercial;
* dashboard operativo;
* documentación ética;
* Model Card;
* Responsible AI Assessment.

Esto demuestra pensamiento de producto y no solo habilidad técnica.

---

## 16. Riesgos de adopción

### Riesgo técnico

El modelo podría no generalizar a datos reales.

Mitigación:

* validación con histórico real;
* calibración;
* monitorización de drift;
* feedback de revisores.

### Riesgo operativo

Demasiadas alertas pueden saturar al equipo.

Mitigación:

* ajuste de umbrales;
* priorización por score;
* colas diferenciadas medium/high;
* revisión periódica de capacidad operativa.

### Riesgo legal o regulatorio

El sistema puede ser malinterpretado como decisión automática.

Mitigación:

* revisión humana;
* documentación de uso no previsto;
* auditoría;
* revisión legal.

### Riesgo reputacional

Un mal uso del score podría dañar la confianza.

Mitigación:

* prohibición de uso comercial agresivo;
* supresión comercial para usuarios de riesgo;
* transparencia interna;
* controles de acceso.

---

## 17. Roadmap de madurez

### Nivel 1: MVP sintético

Estado actual.

* datos sintéticos;
* modelo supervisado;
* SHAP;
* dashboard;
* documentación.

### Nivel 2: Piloto interno con datos históricos

* validación con datos reales seudonimizados;
* revisión con expertos;
* calibración de umbrales;
* análisis de fairness;
* comparación con reglas existentes.

### Nivel 3: Sistema asistido en operación

* integración con sistemas internos;
* cola de revisión;
* feedback de analistas;
* registro de acciones;
* monitorización de drift;
* reporting semanal.

### Nivel 4: Gobernanza avanzada

* auditorías periódicas;
* validación externa;
* controles de acceso;
* documentación regulatoria;
* simulación de escenarios;
* mejora continua.

---

## 18. Requisitos para producción

Antes de producción serían necesarios:

1. Datos reales seudonimizados.
2. Validación con expertos en juego responsable.
3. Revisión legal y regulatoria.
4. Evaluación de impacto de protección de datos.
5. Calibración de probabilidades.
6. Definición de umbrales por capacidad operativa.
7. Fairness review por canal y segmento.
8. Monitorización de drift.
9. Feedback loop de revisores humanos.
10. Control de acceso por rol.
11. Auditoría de uso de scores.
12. Separación de marketing y responsible gaming.

---

## 19. Coste de implementación estimado

Para un MVP técnico como el actual:

```text
Datos sintéticos y features:     1-2 semanas
Modelado y evaluación:           1 semana
Explicabilidad:                  1 semana
Dashboard:                       1 semana
Documentación y gobernanza:      1 semana
```

Total aproximado:

```text
5-6 semanas
```

Para un piloto real con datos internos, el esfuerzo dependería de:

* disponibilidad y calidad de datos;
* integración de fuentes online/retail;
* criterios internos de responsible gaming;
* validación legal;
* capacidad operativa de revisión;
* requisitos de seguridad.

---

## 20. Valor para portfolio profesional

SafePlay AI demuestra competencias en:

* generación de datos sintéticos;
* feature engineering temporal;
* modelado supervisado;
* detección de anomalías;
* evaluación de clase minoritaria;
* explicabilidad con SHAP;
* Streamlit;
* diseño de producto de datos;
* IA responsable;
* documentación técnica;
* comunicación de negocio.

También demuestra criterio sectorial:

* juego responsable;
* trazabilidad;
* cumplimiento;
* protección del usuario;
* no explotación comercial;
* revisión humana.

---

## 21. Mensaje para presentación

Una forma breve de presentar el business case:

> SafePlay AI es una herramienta interna de responsible gaming que transforma datos de comportamiento en alertas explicables y acciones proporcionales. El sistema permite detectar cambios tempranos, priorizar revisión humana, reducir presión comercial sobre usuarios vulnerables y documentar el proceso con trazabilidad y gobernanza.

---

## 22. Conclusión

SafePlay AI es valioso porque conecta tres dimensiones que suelen aparecer separadas:

1. capacidad técnica de data science;
2. utilidad operativa para negocio;
3. principios de IA responsable.

El resultado no es solo un modelo predictivo, sino un prototipo de producto de datos que podría servir como base para una herramienta interna de prevención, revisión y gobernanza en el sector gaming.
