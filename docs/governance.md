# Documentación de Gobernanza para el Proyecto TheroydCancer

Este documento describe las prácticas de gobernanza aplicadas en el proyecto para asegurar la calidad, reproducibilidad y cumplimiento de los modelos de machine learning.

## 1. Gobernanza de Modelos (Model Governance)

La gobernanza de modelos se asegura de que cada versión del modelo esté documentada, rastreable y cumpla con criterios de calidad y seguridad antes de ser puesta en producción.

### 1.1 Registro de Modelos y Versionado

- **Registro**: Todos los modelos entrenados se registran en el MLflow Model Registry. Esto permite la gestión de versiones, facilitando la transición de los modelos entre diferentes etapas de desarrollo (por ejemplo, `staging` y `production`).
- **Versionado**: Cada versión del modelo recibe una etiqueta con el número de versión y una etiqueta de estado (`stage`) que indica si está en pruebas o en producción.

### 1.2 Métricas de Desempeño

- **Métricas Clave**: Antes de pasar un modelo a producción, se evalúan métricas de precisión, recall y tasa de falsos positivos/falsos negativos.
- **Umbrales de Calidad**: Los modelos deben alcanzar una precisión mínima del 85% y mantener una tasa de falsos positivos menor al 5% antes de ser desplegados.

### 1.3 Ciclo de Vida del Modelo (Etiquetas y Control de Versiones)

- **Etiquetas**: Se utilizan etiquetas en MLflow para describir el estado del modelo. Las etiquetas incluyen:
  - `stage`: Describe el estado del modelo (por ejemplo, `staging`, `production`).
  - `version`: Número de versión del modelo, siguiendo una convención `major.minor.patch`.
  - `risk_level`: Nivel de riesgo asociado al modelo (por ejemplo, `low`, `medium`, `high`).
- **Transición de Estados**: Los modelos pasan de `staging` a `production` una vez cumplen con los requisitos de calidad y riesgo.

## 2. Estándares de Código

Los estándares de código aseguran que el código sea claro, documentado y siga las mejores prácticas. Esto facilita la colaboración y el mantenimiento a largo plazo.

### 2.1 Herramientas de Calidad de Código

- **flake8**: Se utiliza `flake8` para verificar que el código siga las normas de estilo de Python (PEP 8). Ejecuta `flake8 src/` para revisar el código.
- **black**: Se utiliza `black` para asegurar un formato uniforme en todo el proyecto. Ejecuta `black src/` para formatear el código automáticamente.

### 2.2 Documentación del Código

Cada función y módulo del proyecto incluye **docstrings** que describen su propósito, parámetros de entrada y valores de retorno. Esto permite una rápida comprensión del código por parte de otros desarrolladores.

## 3. Verificación Ética y de Riesgo

Para asegurar que el modelo cumple con estándares éticos y minimiza riesgos, se realizan evaluaciones de sesgo en los datos y de riesgo en el desempeño del modelo.

### 3.1 Evaluación de Sesgo en los Datos

Antes de entrenar el modelo, se revisa el dataset para identificar y corregir posibles sesgos. Esto incluye:
- **Distribución de Clases**: Asegurarse de que las clases en la variable objetivo están equilibradas.
- **Representación de Grupos**: Verificar que los grupos poblacionales están representados de manera justa para evitar sesgos en el modelo.

### 3.2 Evaluación de Riesgo del Modelo

- **Umbrales de Riesgo**: Establecemos un umbral máximo del 5% para la tasa de falsos positivos y falsos negativos.
- **Métricas de Riesgo**: Las métricas de riesgo se evalúan después de cada entrenamiento para asegurar que el modelo no pone en riesgo a los usuarios.

