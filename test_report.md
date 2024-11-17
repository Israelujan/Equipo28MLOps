# Informe de Pruebas del Proyecto

Este informe documenta los resultados de las pruebas unitarias e integrales realizadas en el proyecto. Las pruebas se diseñaron para garantizar la integridad del dataset y la correcta funcionalidad de los componentes del pipeline de Machine Learning.

---

## Sección 1: Pruebas Unitarias de Integridad de Datos

Las pruebas unitarias en `test_data_integrity.py` verifican que el dataset cumpla con los estándares básicos de integridad de datos antes del análisis y entrenamiento.

### Pruebas de Integridad de Datos

1. **Prueba: Valores Nulos (test_no_missing_values)**
   - **Descripción**: Verifica que no haya valores nulos en el dataset.
   - **Resultado**: ✅ Pasó sin problemas.
   
2. **Prueba: Tipos de Columnas (test_column_types)**
   - **Descripción**: Asegura que las columnas tengan los tipos de datos correctos.
   - **Resultado**: ⚠️ Falló inicialmente debido a que las columnas `TumorSize` y `Outcome` no estaban presentes. Se revisaron los nombres y se ajustó la prueba.

3. **Prueba: Valores Únicos en Categóricas (test_unique_values_in_categorical)**
   - **Descripción**: Comprueba que las columnas categóricas tengan un número razonable de valores únicos.
   - **Resultado**: ✅ Pasó sin problemas.

4. **Prueba: Rango de Valores (test_value_ranges)**
   - **Descripción**: Verifica que ciertos valores estén dentro de rangos lógicos, como edad entre 0 y 120.
   - **Resultado**: ✅ Pasó sin problemas.

5. **Prueba: Duplicados (test_no_duplicates)**
   - **Descripción**: Asegura que no haya filas duplicadas en el dataset.
   - **Resultado**: ⚠️ Falló al detectar 19 filas duplicadas. Se decidió eliminarlas en la fase de preprocesamiento.

### Conclusión de Pruebas Unitarias

La mayoría de las pruebas unitarias pasaron sin problemas. Se realizaron algunos ajustes en la prueba de tipos de columnas y se eliminó duplicados en el dataset como parte del preprocesamiento.

---

## Sección 2: Pruebas Integrales de Entrenamiento de Modelo

Las pruebas integrales verifican el pipeline completo de entrenamiento y registro del modelo en MLflow.

### Pruebas Integrales

1. **Prueba: Entrenamiento Completo del Modelo**
   - **Descripción**: Verifica que el modelo se entrene correctamente con el dataset limpio.
   - **Métricas Registradas**:
     - Precisión: 0.96
     - Recall: 0.94
     - F1-Score: 0.95
   - **Resultado**: ✅ Pasó sin problemas. Los resultados fueron registrados correctamente en MLflow.

2. **Prueba: Registro de Modelo en MLflow**
   - **Descripción**: Asegura que el modelo se registre en MLflow y se le asigne una nueva versión.
   - **Resultado**: ✅ Pasó sin problemas. Varias versiones del modelo se registraron debido a pruebas adicionales.

### Conclusión de Pruebas Integrales

Las pruebas integrales confirmaron que el pipeline de entrenamiento y registro funciona según lo esperado, y que el modelo se registra en MLflow correctamente.

---

## Resumen General

Ambas secciones de pruebas validaron la integridad del dataset y la funcionalidad de entrenamiento del modelo. Se realizaron los siguientes ajustes:

- Ajuste en los nombres de columnas en `test_column_types`.
- Eliminación de filas duplicadas en el preprocesamiento de datos.

Este informe proporciona una visión general de las pruebas realizadas y de las acciones tomadas para asegurar la calidad y consistencia de los datos y el modelo.
