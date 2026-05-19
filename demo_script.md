# Demo Script - SafePlay AI

## Objetivo del guion

Este guion está pensado para presentar SafePlay AI en una entrevista o reunión técnica en 3-5 minutos.

La idea no es explicar cada línea de código, sino demostrar que el proyecto combina:

* entendimiento de negocio;
* data science aplicado;
* feature engineering temporal;
* explicabilidad;
* intervención responsable;
* gobernanza de modelo.

---

## Versión corta: pitch de 60 segundos

SafePlay AI es un prototipo de IA responsable aplicado al sector gaming. El objetivo es detectar patrones tempranos de juego de riesgo usando datos de comportamiento, como frecuencia de sesiones, depósitos, pérdidas, actividad nocturna y cambios frente al histórico del usuario.

El sistema genera un score de riesgo, explica cada alerta con SHAP y recomienda intervenciones proporcionales, como mensajes preventivos, sugerencia de límites o revisión humana.

Lo importante es que no es una caja negra ni un sistema de decisión automática. Está diseñado con una capa de gobernanza, documentación de limitaciones y enfoque de protección del usuario.

---

%## Versión entrevista: guion de 3-5 minutos

### 1. Introducción

“SafePlay AI es un proyecto de data science aplicado a juego responsable. Lo planteé como si fuera una herramienta interna para una empresa del sector gaming, donde no basta con predecir, sino que también hay que explicar, priorizar y actuar de forma responsable.”

“Utilizo datos sintéticos de usuarios y sesiones para detectar patrones tempranos de riesgo, generar alertas explicables y recomendar intervenciones proporcionales.”

---

### 2. Problema que resuelve

“El problema es que algunos usuarios pueden mostrar cambios de comportamiento preocupantes: empiezan a jugar con más frecuencia, aumentan depósitos, concentran pérdidas, vuelven rápido después de perder o juegan más de noche.”

“SafePlay AI no diagnostica ludopatía ni toma decisiones automáticas. Lo que hace es priorizar usuarios para revisión preventiva.”

---

### 3. Pipeline del proyecto

“El pipeline tiene cinco grandes bloques.”

```text
1. Generación de datos sintéticos
2. Feature engineering temporal
3. Modelado predictivo
4. Explicabilidad con SHAP
5. Dashboard y recomendaciones responsables
```

“Primero genero usuarios y sesiones sintéticas. Después agrego la información por usuario usando ventanas temporales de 7, 14 y 30 días. Luego entreno modelos para clasificar riesgo bajo, medio y alto. Finalmente explico cada alerta y la convierto en una recomendación operativa.”

---

### 4. Datos sintéticos

“Como no uso datos reales, construí un generador sintético con tres perfiles: bajo, medio y alto riesgo.”

“Los usuarios de bajo riesgo juegan ocasionalmente, con importes pequeños y sesiones cortas. Los de riesgo medio muestran más frecuencia y algún aumento reciente. Los de alto riesgo tienen sesiones frecuentes, depósitos crecientes, mayor actividad nocturna, pérdidas recientes y retornos rápidos después de pérdidas.”

“Generé 10.000 usuarios y más de 200.000 sesiones sintéticas.”

---

### 5. Feature engineering

“Para mí, la parte clave del proyecto está en las features temporales.”

“En vez de mirar solo valores absolutos, calculo cambios recientes frente al histórico del usuario.”

Ejemplos:

```text
frequency_increase_ratio = sesiones últimos 7 días / media semanal de los 90 días anteriores
stake_increase_ratio = ticket medio últimos 7 días / ticket medio histórico
deposit_increase_ratio = depósitos recientes / media histórica de depósitos
```

“Esto permite detectar cambios de comportamiento, no solo usuarios que juegan mucho de forma estable.”

---

### 6. Modelos

“Entrené una Logistic Regression como baseline y un XGBoost como modelo principal.”

“El baseline es útil porque es simple e interpretable. XGBoost captura mejor relaciones no lineales entre frecuencia, depósitos, pérdidas y comportamiento nocturno.”

“También añadí Isolation Forest como capa de anomalías para identificar usuarios atípicos aunque no encajen perfectamente en la etiqueta sintética.”

---

### 7. Métricas

“En este caso no vendo accuracy como métrica principal.”

“Me centro en la clase de alto riesgo, especialmente en recall, porque no quiero dejar pasar usuarios con señales relevantes. Pero también miro precision para no saturar al equipo de revisión.”

Resultados del MVP:

```text
XGBoost
Precision high: 0.9608
Recall high:    0.9515
F1 high:        0.9561
PR-AUC high:    0.9927
```

“Como las etiquetas son sintéticas, estos resultados no deben interpretarse como rendimiento productivo. Lo importante es validar el pipeline completo.”

---

### 8. Explicabilidad

“Después uso SHAP para explicar el modelo.”

“Esto es importante porque en un caso de responsible gaming no basta con decir que un usuario tiene score alto. Hay que poder explicar por qué.”

Drivers principales observados:

```text
- incremento de depósitos frente al histórico
- pérdida neta reciente elevada
- retornos rápidos después de pérdidas
- aumento del importe apostado en los últimos 7 días
- incremento reciente de frecuencia
- actividad nocturna recurrente
- sesiones largas
```

“En la vista individual, cada usuario tiene sus propios drivers.”

Ejemplo:

```text
Usuario U005307
Riesgo: Alto
Drivers:
1. retornos rápidos después de pérdidas
2. actividad nocturna recurrente
3. pérdida neta reciente elevada
4. incremento de depósitos frente al histórico
5. incremento reciente de frecuencia
```

---

### 9. Reglas de intervención

“Una parte importante es que el proyecto no termina en la predicción. Cada nivel de riesgo tiene una intervención responsable.”

```text
Riesgo bajo:
- sin intervención individual
- comunicación general de juego responsable

Riesgo medio:
- mensaje preventivo
- sugerencia de límites
- reducción de presión promocional
- monitorización

Riesgo alto:
- exclusión de campañas comerciales
- revisión humana
- mensaje de pausa
- sugerencia de autoexclusión o cooling-off
- registro para auditoría
```

“El principio es claro: el sistema debe proteger al usuario, no maximizar su gasto.”

---

### 10. Dashboard

“Construí un dashboard en Streamlit con cuatro páginas.”

```text
1. Executive Overview
2. Alert Monitor
3. User Profile
4. Model Governance
```

#### Executive Overview

“Aquí muestro KPIs generales: usuarios analizados, distribución de riesgo, alertas generadas, acciones recomendadas y principales drivers globales.”

Frase para decir durante la demo:

“En esta simulación analizamos 10.000 usuarios. La mayoría queda en riesgo bajo, un 12% en riesgo medio y un 4% en riesgo alto. Se generan 1.620 alertas operativas.”

#### Alert Monitor

“Esta página simula la cola de trabajo de un equipo de responsible gaming. Se pueden filtrar alertas por riesgo, canal y score.”

Frase para decir:

“No todos los usuarios requieren revisión humana. Los de alto riesgo sí; los de riesgo medio reciben intervenciones ligeras y monitorización.”

#### User Profile

“Esta es la pantalla más importante para la demo. Selecciono un usuario de alto riesgo y enseño su evolución temporal, sus depósitos, sesiones, pérdida acumulada y explicación SHAP.”

Frase para decir:

“Este usuario no aparece como alto riesgo por una única variable. El sistema detecta una combinación de retorno tras pérdidas, actividad nocturna, pérdidas recientes e incremento de depósitos.”

#### Model Governance

“Esta página documenta variables usadas, variables excluidas, métricas, matriz de confusión, limitaciones y política de revisión humana.”

Frase para decir:

“Quería que el proyecto no fuera solo un modelo, sino una herramienta con gobernanza: uso previsto, usos no previstos, limitaciones y controles.”

---

## Orden recomendado para enseñar la demo

### Paso 1: Executive Overview

Mostrar:

* usuarios analizados;
* distribución de riesgo;
* alertas operativas;
* drivers globales.

Mensaje clave:

“Esto da una vista ejecutiva del estado de riesgo en la base de usuarios.”

### Paso 2: Alert Monitor

Mostrar:

* tabla de alertas;
* filtros;
* acción recomendada;
* revisión humana;
* supresión comercial.

Mensaje clave:

“Esto convierte el modelo en una cola operativa accionable.”

### Paso 3: User Profile

Mostrar:

* usuario alto riesgo;
* score;
* drivers;
* evolución de apuestas;
* depósitos;
* sesiones;
* pérdida acumulada;
* recomendación.

Mensaje clave:

“Aquí se ve la explicación individual y el contexto temporal de la alerta.”

### Paso 4: Model Governance

Mostrar:

* métricas;
* matriz de confusión;
* variables usadas;
* variables excluidas;
* limitaciones;
* política de revisión humana.

Mensaje clave:

“El sistema está diseñado para apoyar decisiones, no para automatizarlas.”

---

## Preguntas probables y respuestas recomendadas

### ¿Por qué datos sintéticos?

“Porque no tengo acceso a datos reales de usuarios y porque es un caso sensible. Preferí construir un generador sintético realista para demostrar el pipeline completo sin usar información personal.”

### ¿El modelo detecta ludopatía?

“No. Detecta señales conductuales compatibles con riesgo y prioriza revisión preventiva. No es un diagnóstico clínico.”

### ¿Por qué XGBoost?

“Porque captura relaciones no lineales entre frecuencia, depósitos, pérdidas y comportamiento temporal. Aun así, comparo contra Logistic Regression como baseline.”

### ¿Por qué no usar accuracy?

“Porque el problema está desbalanceado y la clase relevante es alto riesgo. Me interesa especialmente recall y precision en esa clase.”

### ¿Qué harías antes de producción?

“Validar con expertos de juego responsable, usar datos históricos reales seudonimizados, revisar fairness, calibrar probabilidades, definir umbrales operativos y crear un feedback loop con revisores humanos.”

### ¿Cómo evitarías uso indebido para marketing?

“Separando explícitamente responsible gaming de CRM comercial. En el prototipo, los usuarios medium y high se marcan para supresión o reducción promocional, no para campañas agresivas.”

### ¿Qué parte consideras más importante?

“El feature engineering temporal y la capa de gobernanza. El modelo es importante, pero el valor real está en detectar cambios frente al histórico, explicar las alertas y convertirlas en acciones responsables.”

---

## Frase para LinkedIn o CV

Desarrollé SafePlay AI, un sistema de machine learning explicable para detectar patrones tempranos de juego de riesgo mediante datos sintéticos de sesiones, depósitos y comportamiento temporal. El proyecto incluye generación de datos, feature engineering por ventanas temporales, modelo predictivo con XGBoost, detección de anomalías, explicabilidad con SHAP, recomendaciones de intervención responsable y dashboard operativo en Streamlit.

---

## Frase orientada a empresa del sector

“He querido construir un proyecto que no solo demuestre capacidad técnica, sino entendimiento del contexto real del sector: regulación, juego responsable, trazabilidad, explicabilidad y equilibrio entre negocio y protección del cliente.”

---

## Cierre recomendado

“Para mí, SafePlay AI demuestra cómo un proyecto de data science puede ir más allá del modelo. Incluye datos, features, predicción, explicabilidad, acción responsable, dashboard y documentación de gobernanza. Por eso lo planteo como un p
