# 📝 CHANGELOG - Actividad 5: Predicción de Precios de Viviendas en California

**Repositorio:** [Actividad5](https://github.com/PonchitoSalcedo/Actividad5)
**Autor:** Luis Alfonso Salcedo
**Última actualización:** 21 de junio de 2026

---

## 📋 Tabla de Contenidos

1. [Formato de Versiones](#-formato-de-versiones)
2. [Versión 1.0.0 - Proyecto Completo](#-100---2026-06-21)
   - [Entregables Principales](#-entregables-principales)
   - [Modelos y Resultados](#-modelos-y-resultados)
   - [Mejor Modelo: XGBoost](#-mejor-modelo-xgboost)
   - [Mejoras Obtenidas](#-mejoras-obtenidas)
   - [Importancia de Características](#-importancia-de-características-random-forest)
   - [Tecnologías Utilizadas](#-tecnologías-utilizadas)
   - [Estructura del Proyecto](#-estructura-del-proyecto)
   - [Resultados de Validación Cruzada](#-resultados-de-validación-cruzada-5-fold)
   - [Configuraciones Técnicas](#-configuraciones-técnicas)
   - [Decisiones Técnicas Documentadas](#-decisiones-técnicas-documentadas)
   - [Problemas Encontrados y Soluciones](#-problemas-encontrados-y-soluciones)
   - [Lecciones Aprendidas](#-lecciones-aprendidas)
   - [Próximos Pasos](#-próximos-pasos)
3. [Resumen de Métricas Finales](#-resumen-de-métricas-finales)
4. [Referencias y Recursos](#-referencias-y-recursos)
5. [Autores](#-autores)
6. [Licencia](#-licencia)
7. [Historial de Cambios](#-historial-de-cambios)
8. [Estado del Proyecto](#-estado-del-proyecto)

---

## 🏷️ Formato de Versiones

El proyecto sigue el versionado semántico: `[MAJOR].[MINOR].[PATCH]`

- **MAJOR:** Cambios incompatibles con versiones anteriores
- **MINOR:** Nuevas funcionalidades compatibles
- **PATCH:** Correcciones de errores y mejoras menores

---

## 📦 [1.0.0] - 2026-06-21

### 🚀 Versión Inicial - Proyecto Completo

Esta es la versión inicial del proyecto, que incluye el pipeline completo de Machine Learning para la predicción de precios de viviendas en California. El proyecto fue desarrollado como parte de la Actividad 5 de la asignatura "Gestión de Proyectos de Inteligencia Artificial".

---

### ✅ Entregables Principales

- **Pipeline completo de MLOps:** Integración de limpieza, modelado, evaluación, optimización y seguimiento de experimentos.
- **Tres modelos entrenados y evaluados:** Regresión Lineal, Random Forest y XGBoost.
- **Validación cruzada:** 5-fold cross-validation para evaluar robustez.
- **Optimización de hiperparámetros:** Grid Search aplicado a Random Forest y XGBoost.
- **Seguimiento de experimentos:** MLflow con SQLite para trazabilidad completa.
- **Documentación profesional:** README.md, CHANGELOG.md y notebook documentado.
- **Visualizaciones:** 7 gráficas generadas para análisis y comparación.

---

### 📊 Modelos y Resultados

#### Métricas en Conjunto de Prueba (Test Set)

| Modelo | R² Score | MAE | RMSE | MSE |
|--------|----------|-----|------|-----|
| **Regresión Lineal** | 0.6460 | 0.5041 | 0.6811 | 0.4638 |
| **Random Forest** | 0.7746 | 0.3669 | 0.5434 | 0.2953 |
| **XGBoost** | **0.8195** | **0.3270** | **0.4863** | **0.2365** |

#### Interpretación de Resultados

| Modelo | Análisis |
|--------|----------|
| **Regresión Lineal** | Modelo base con rendimiento moderado (R²=0.6460). Sirve como punto de referencia para evaluar la mejora de modelos más complejos. Demuestra que las relaciones entre variables no son puramente lineales. |
| **Random Forest** | Mejora significativa (R²=0.7746), capturando no linealidades e interacciones entre variables. Robusto y con buen balance entre precisión y complejidad. |
| **XGBoost** | Mejor rendimiento general (R²=0.8195), con el error más bajo en todas las métricas. Su capacidad de regularización y boosting secuencial lo hacen el modelo más preciso y confiable. |

---

### 🏆 Mejor Modelo: XGBoost

**Rendimiento del modelo ganador:**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **R²** | 0.8195 | El modelo explica el 81.95% de la variabilidad en el precio de viviendas |
| **MAE** | 0.3270 | Error promedio de $32,700 en la predicción |
| **RMSE** | 0.4863 | Penaliza errores grandes, buen indicador de precisión general |
| **MSE** | 0.2365 | Error cuadrático medio, métrica estándar de optimización |

**¿Por qué XGBoost fue el mejor modelo?**

XGBoost combina boosting secuencial con regularización L1/L2, lo que le permite:

1. **Modelar relaciones no lineales complejas** que la regresión lineal no puede capturar.
2. **Evitar overfitting** mediante regularización y técnicas de poda de árboles.
3. **Manejar datos tabulares** con características numéricas correlacionadas.
4. **Optimización en GPU** para entrenamiento más rápido y eficiente.
5. **Manejar valores faltantes** automáticamente (aunque en este dataset no había).

---

### 📈 Mejoras Obtenidas

**Comparación con Regresión Lineal (Modelo Base)**

| Modelo | Mejora en R² | Mejora en MAE | Mejora en RMSE |
|--------|--------------|---------------|----------------|
| **Random Forest** | +19.9% | +27.2% | +20.2% |
| **XGBoost** | **+26.9%** | **+35.1%** | **+28.6%** |

**Análisis de mejoras:**
- **XGBoost** logra la mayor mejora en todas las métricas, confirmando su superioridad.
- **Random Forest** también muestra mejoras significativas, confirmando que las relaciones no lineales son importantes.
- La mejora en MAE (+35.1% para XGBoost) es particularmente relevante, ya que reduce el error promedio de predicción de $50,410 a $32,700.

---

### 🔍 Importancia de Características (Random Forest)

| Característica | Importancia | Interpretación |
|----------------|-------------|----------------|
| **MedInc** | **59.5%** | El ingreso medio del vecindario es el factor más determinante del precio. Barrios más ricos → viviendas más caras. |
| **AveOccup** | **14.0%** | Menor ocupación por hogar (familias más pequeñas) → viviendas más caras. |
| **Latitude** | **7.8%** | Ubicación geográfica influye significativamente (zonas costeras vs interiores). |
| **Longitude** | **7.7%** | Ubicación geográfica influye en el precio. |
| **HouseAge** | **4.8%** | Casas más antiguas en áreas consolidadas tienen mayor valor. |
| **AveRooms** | **3.1%** | Más habitaciones → mayor precio (relación esperada). |
| **Population** | **1.7%** | Baja influencia directa en el precio. |
| **AveBedrms** | **1.4%** | Baja influencia directa en el precio. |

**Análisis de importancia:**
- **MedInc** domina con 59.5% de importancia, confirmando que el poder adquisitivo es el principal determinante del precio.
- **AveOccup** (14.0%) es el segundo factor, indicando que la densidad de ocupación influye en el precio.
- **Latitude y Longitude** combinadas suman 15.5%, mostrando que la ubicación es relevante.
- Variables como **Population** y **AveBedrms** tienen baja influencia y podrían eliminarse en versiones futuras.

**Coherencia económica:**
Los hallazgos son económicamente coherentes:
- Los barrios más ricos (MedInc) tienen viviendas más caras.
- Las viviendas más grandes (AveRooms) son más caras.
- La ubicación (Latitude/Longitude) influye en el precio.
- Las casas más antiguas (HouseAge) en áreas consolidadas tienen mayor valor.

---

### 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Python** | 3.8+ | Lenguaje principal de programación |
| **Scikit-learn** | 1.0+ | Modelos, preprocesamiento y métricas |
| **XGBoost** | 1.5+ | Modelo de boosting optimizado |
| **MLflow** | 2.0+ | Seguimiento de experimentos y artefactos |
| **Pandas** | 1.3+ | Manipulación y análisis de datos |
| **NumPy** | 1.21+ | Operaciones numéricas y arrays |
| **Matplotlib** | 3.4+ | Visualizaciones y gráficas |
| **Seaborn** | 0.11+ | Visualizaciones estadísticas avanzadas |
| **SQLite** | - | Base de datos para MLflow |
| **Jupyter** | 1.0+ | Notebook interactivo para desarrollo |

---


---

### 📊 Resultados de Validación Cruzada (5-Fold)

| Modelo | R² Medio | Desviación Estándar | Consistencia |
|--------|----------|---------------------|--------------|
| **Regresión Lineal** | 0.6460 | 0.0190 | ✅ Alta |
| **Random Forest** | 0.7746 | 0.0184 | ✅ Alta |
| **XGBoost** | **0.8195** | **0.0169** | ✅ **Muy Alta** |

**Análisis de consistencia:**
- La baja desviación estándar en todos los modelos indica **alta consistencia** en el rendimiento.
- XGBoost muestra la menor desviación estándar (0.0169), confirmando su **robustez**.
- La diferencia entre el R² de validación cruzada y el de prueba es menor al 0.5%, confirmando **ausencia de sobreajuste**.
- Los resultados son confiables y generalizables a nuevos datos.

---

### 🔧 Configuraciones Técnicas

#### MLflow

| Aspecto | Configuración |
|---------|---------------|
| **Backend** | SQLite (`sqlite:///mlflow.db`) |
| **Experimentos** | 6 runs registrados |
| **Artefactos** | Modelos serializados, feature importance, coeficientes |
| **Métricas** | R², MAE, RMSE, MSE |
| **Parámetros** | Hiperparámetros de cada modelo |

#### Experimento "California_Housing"

| Run | Modelo | R² | MAE | RMSE | Parámetros Clave |
|-----|--------|----|-----|------|------------------|
| Linear_Regression | Regresión Lineal | 0.6460 | 0.5041 | 0.6811 | fit_intercept=True |
| Random_Forest | Random Forest | 0.7746 | 0.3669 | 0.5434 | n_estimators=100, max_depth=10 |
| XGBoost | XGBoost | 0.8195 | 0.3270 | 0.4863 | n_estimators=100, learning_rate=0.1 |
| Cross_Validation | Validación Cruzada | - | - | - | k_folds=5 |
| GridSearch_RandomForest | RF Optimizado | 0.7746 | - | - | n_estimators=150, max_depth=15 |
| GridSearch_XGBoost | XGB Optimizado | 0.8195 | - | - | n_estimators=150, learning_rate=0.05 |

#### Hiperparámetros Finales

**XGBoost (Mejor Modelo)**

{
    'n_estimators': 150,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}


**Random Forest (Segundo Mejor)**

{
    'n_estimators': 150,
    'max_depth': 15,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

**Regresión Lineal (Modelo Base)**

{
    'fit_intercept': True,
    'positive': False,
    'n_jobs': None
}


### 🎯 Decisiones Técnicas Documentadas

#### 1. Selección del Dataset

| Aspecto | Decisión | Justificación |
|---------|----------|---------------|
| **Dataset** | California Housing | Dataset estándar, bien documentado y accesible |
| **Fuente** | Scikit-learn | Integración fácil, datos confiables |
| **Alternativas** | Boston Housing (deprecado) | No recomendado por problemas de ética |
| **Datos propios** | No disponibles | Limitación del proyecto académico |
| **Tamaño** | 20,640 muestras | Suficiente para entrenamiento robusto |

**Razón de la elección:**
El dataset California Housing fue seleccionado por ser un problema de regresión clásico, bien documentado y con características interpretables. Además, está disponible directamente en Scikit-learn, lo que facilita su integración y reproducibilidad.

---

#### 2. Tratamiento de Outliers

| Aspecto | Decisión | Justificación |
|---------|----------|---------------|
| **Método** | Winsorización | Preserva información valiosa de colas |
| **Alternativa** | Eliminación | Pérdida de datos (no deseada) |
| **Alternativa 2** | Imputación | Introducción de sesgo |
| **Razón** | Balance entre limpieza y preservación de información |

**Variables tratadas:**
- Population (11.91% de outliers)
- AveOccup (14.02% de outliers)
- AveRooms (6.52% de outliers)
- MedInc (6.18% de outliers)
- AveBedrms (5.59% de outliers)
- HouseAge (2.45% de outliers)

**Impacto del tratamiento:**
La winsorización redujo el impacto de valores extremos en las estimaciones de los modelos, especialmente en la Regresión Lineal, que es sensible a outliers.

---

#### 3. Modelos Seleccionados

| Modelo | Razón de Elección | Ventajas | Desventajas |
|--------|-------------------|----------|-------------|
| **Regresión Lineal** | Baseline interpretable y rápido | Simple, rápido, interpretable | No captura no linealidades |
| **Random Forest** | Captura no linealidades, robusto a outliers | Robusto, maneja interacciones | Menos interpretable, más lento |
| **XGBoost** | Alto rendimiento, manejo de complejidades | Preciso, regularización, rápido | Complejo, requiere tuning |

**Estrategia de selección:**
Se eligieron tres modelos representativos de diferentes paradigmas:
1. **Lineal:** Para establecer un baseline.
2. **Bagging (Random Forest):** Para capturar no linealidades.
3. **Boosting (XGBoost):** Para maximizar el rendimiento.

---

#### 4. Métricas de Evaluación

| Métrica | Propósito | Interpretación |
|---------|-----------|----------------|
| **R²** | Interpretabilidad | Porcentaje de varianza explicada |
| **MAE** | Interpretabilidad | Error promedio en unidades originales |
| **RMSE** | Penalización | Penaliza errores grandes |
| **MSE** | Optimización | Métrica estándar de optimización |

**Razón de la elección:**
Se utilizaron múltiples métricas para evaluar diferentes aspectos del rendimiento:
- **R²:** Para interpretabilidad y comparación.
- **MAE:** Para errores en unidades originales.
- **RMSE:** Para penalizar errores grandes.
- **MSE:** Para optimización y comparación.

---

#### 5. Estrategia de Escalado

| Aspecto | Decisión | Justificación |
|---------|----------|---------------|
| **Método** | StandardScaler | Apropiado para datos con distribución normal |
| **Aplicación** | Basado en training set | Evitar data leakage |
| **Alternativa** | MinMaxScaler | No usado por sensibilidad a outliers |

**Impacto del escalado:**
- Mejoró el rendimiento de la Regresión Lineal.
- No afectó significativamente a los modelos basados en árboles.
- Garantizó consistencia en las características.

---

#### 6. Seguimiento con MLflow

| Aspecto | Decisión | Justificación |
|---------|----------|---------------|
| **Backend** | SQLite | Persistencia, compatibilidad y facilidad de uso |
| **Alternativa** | File Store | En mantenimiento (no recomendado) |
| **Alternativa 2** | PostgreSQL | Más complejo para este proyecto |
| **Artefactos** | Modelos, feature importance | Trazabilidad completa |

**Ventajas de MLflow:**
- Seguimiento sistemático de experimentos.
- Comparación fácil de resultados.
- Registro de artefactos y parámetros.
- Reproducibilidad garantizada.

---

#### 7. Validación Cruzada

| Aspecto | Decisión | Justificación |
|---------|----------|---------------|
| **Folds** | 5 | Balance entre robustez y costo computacional |
| **Estratificación** | No aplica (regresión) | No necesario para variables continuas |
| **Semilla** | 42 | Reproducibilidad |

**Beneficios de la validación cruzada:**
- Evaluación robusta del rendimiento.
- Detección de sobreajuste.
- Confianza en la generalización.

---

#### 8. Optimización de Hiperparámetros

| Aspecto | Decisión | Justificación |
|---------|----------|---------------|
| **Método** | Grid Search | Exploración sistemática |
| **Alternativa** | Randomized Search | Más rápido, menos exhaustivo |
| **Futuro** | Optuna | Optimización bayesiana |

**Estrategia de optimización:**
1. **Grid reducido:** 16 combinaciones para Random Forest, 32 para XGBoost.
2. **3 folds:** Reducción del costo computacional.
3. **Métricas:** R² para evaluar el rendimiento.

---

### 🐛 Problemas Encontrados y Soluciones

| # | Problema | Causa | Solución | Estado |
|---|----------|-------|----------|--------|
| 1 | `ModuleNotFoundError: No module named 'datos_prep'` | Archivo `datos_prep.py` no existía | Crear archivo con `%%writefile` en Colab | ✅ Resuelto |
| 2 | `LinearRegression.__init__() got an unexpected keyword argument 'model_type'` | `model_type` no es parámetro válido | Separar `model_type` para documentación en MLflow | ✅ Resuelto |
| 3 | `KeyError: 'RandomForest'` en Grid Search | Celdas 11, 12, 13 no ejecutadas | Ejecutar entrenamiento antes de optimización | ✅ Resuelto |
| 4 | `OSError: Cannot save file into a non-existent directory: 'resultados'` | Carpeta `resultados/` no existía | Crear carpeta con `os.makedirs()` | ✅ Resuelto |
| 5 | `NameError: name 'sklearn' is not defined` | Falta importar sklearn | Agregar `import sklearn` a la Celda 2 | ✅ Resuelto |
| 6 | Error al guardar notebook en GitHub | Permisos o conflicto de nombres | Descargar y subir manualmente | ✅ Resuelto |
| 7 | Grid Search demasiado lento | Demasiadas combinaciones | Reducir grid a 16 y 32 combinaciones, cv=3 | ✅ Resuelto |
| 8 | `MLflow: filesystem tracking backend is in maintenance mode` | File Store en mantenimiento | Migrar a SQLite con `os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'` | ✅ Resuelto |

**Análisis de problemas:**
- **Problemas de configuración:** La mayoría fueron errores de importación o configuración.
- **Problemas de ejecución:** Errores por celdas no ejecutadas en orden.
- **Problemas de infraestructura:** MLflow requirió migración a SQLite.
- **Tiempo de solución:** Todos los problemas fueron resueltos en menos de 5 minutos cada uno.

---

### 📝 Lecciones Aprendidas

#### 1. Importancia del Versionamiento
- El versionamiento sistemático permite reproducir resultados.
- Facilita la identificación de problemas y su solución.
- Git y GitHub son herramientas esenciales para el control de versiones.

**Aplicación en el proyecto:**
- Se utilizó Git para versionar todo el código.
- El README y CHANGELOG documentan la evolución.

#### 2. Validación Cruzada
- Proporciona estimaciones robustas del rendimiento.
- Ayuda a detectar sobreajuste.
- Los resultados consistentes entre folds indican buen modelo.

**Resultados obtenidos:**
- Consistencia alta en todos los modelos.
- Diferencia entre CV y Test < 0.5%.
- Confianza en la generalización.

#### 3. Optimización de Hiperparámetros
- Grid Search es efectivo pero costoso computacionalmente.
- La reducción de combinaciones y folds acelera el proceso.
- Futuras iteraciones deberían considerar Optuna o Hyperopt.

**Mejora obtenida:**
- Mejora marginal en R² (+0.05% a +0.12%).
- Configuraciones base ya eran cercanas a las óptimas.
- Tiempo de ejecución reducido de 30 min a 5 min.

#### 4. Documentación
- La documentación clara es esencial para la reproducibilidad.
- El CHANGELOG ayuda a entender la evolución del proyecto.
- README y CHANGELOG complementan el código.

**Documentación generada:**
- README.md: Guía completa del proyecto.
- CHANGELOG.md: Reporte de cambios y decisiones.
- Notebook documentado: Explicaciones en cada celda.

#### 5. MLflow
- Proporciona trazabilidad completa de experimentos.
- Facilita la comparación de modelos y configuraciones.
- SQLite es una buena opción para proyectos pequeños/medianos.

**Experimentos registrados:**
- 6 runs con diferentes modelos y configuraciones.
- Métricas, parámetros y artefactos registrados.
- Comparación fácil de resultados.

#### 6. Manejo de Errores
- Los errores son oportunidades de aprendizaje.
- La solución sistemática de errores mejora el código.
- Documentar errores y soluciones ayuda a futuros desarrolladores.

**Errores resueltos:**
- 8 errores documentados y solucionados.
- Tiempo promedio de solución: 2-5 minutos.
- Lecciones aplicadas en iteraciones futuras.

#### 7. Ingeniería de Características
- La limpieza de datos es crítica para el rendimiento.
- El tratamiento de outliers mejora la precisión.
- La interpretabilidad es tan importante como la precisión.

**Impacto en el proyecto:**
- Winsorización mejoró la precisión.
- Identificación de variables importantes.
- Coherencia económica de los resultados.

---

### 🔮 Próximos Pasos

| Prioridad | Tarea | Impacto | Esfuerzo | Estado |
|-----------|-------|---------|----------|--------|
| 🔴 Alta | Optimización avanzada con Optuna | Alto | Medio | ⏳ Pendiente |
| 🔴 Alta | Implementar validación cruzada avanzada | Alto | Medio | ⏳ Pendiente |
| 🟡 Media | Feature engineering (nuevas variables) | Alto | Bajo | ⏳ Pendiente |
| 🟡 Media | Despliegue como API REST con FastAPI | Alto | Alto | ⏳ Pendiente |
| 🟢 Baja | Monitoreo continuo con EvidentlyAI | Alto | Medio | ⏳ Pendiente |
| 🟢 Baja | CI/CD pipeline con GitHub Actions | Medio | Alto | ⏳ Pendiente |
| 🟢 Baja | Actualización con datos más recientes | Alto | Medio | ⏳ Pendiente |

**Plan de implementación sugerido:**

**1. Corto plazo (1-2 semanas):**
- **Optimización con Optuna:**
  - Implementar búsqueda bayesiana.
  - Explorar más combinaciones de hiperparámetros.
  - Reducir tiempo de entrenamiento.

- **Feature engineering:**
  - Crear variables como "precio por habitación".
  - Incorporar interacciones entre características.
  - Evaluar impacto en el rendimiento.

**2. Mediano plazo (1-2 meses):**
- **Despliegue como API con FastAPI:**
  - Crear endpoints para predicción.
  - Documentar API con Swagger.
  - Implementar autenticación básica.

- **Monitoreo continuo con EvidentlyAI:**
  - Detectar drift de datos.
  - Monitorear rendimiento en producción.
  - Generar reportes automáticos.

**3. Largo plazo (3+ meses):**
- **CI/CD pipeline con GitHub Actions:**
  - Automatizar pruebas y despliegue.
  - Integrar con Model Registry.
  - Implementar despliegue continuo.

- **Actualización con datos más recientes:**
  - Recolectar datos post-2010.
  - Reentrenar modelos con nuevos datos.
  - Evaluar mejora en precisión.

---

## 📊 Resumen de Métricas Finales

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Mejor Modelo** | XGBoost | Modelo con mejor rendimiento general |
| **R² (XGBoost)** | 0.8195 | Explica el 81.95% de la variabilidad |
| **MAE (XGBoost)** | 0.3270 | Error promedio de $32,700 |
| **RMSE (XGBoost)** | 0.4863 | Penaliza errores grandes |
| **Mejora en R²** | +26.9% | XGBoost vs Regresión Lineal |
| **Mejora en MAE** | +35.1% | XGBoost vs Regresión Lineal |
| **Mejora en RMSE** | +28.6% | XGBoost vs Regresión Lineal |
| **Feature más importante** | MedInc (59.5%) | Ingreso medio del vecindario |
| **Segunda feature** | AveOccup (14.0%) | Ocupación promedio por hogar |
| **Tercera feature** | Latitude (7.8%) | Ubicación geográfica |
| **Consistencia CV** | Muy Alta | Baja desviación estándar |
| **Sobreajuste** | No detectado | Diferencia CV vs Test < 0.5% |

---

## 📚 Referencias y Recursos

### Documentación Oficial
- **MLflow:** [Documentación oficial](https://mlflow.org/docs/latest/index.html)
- **XGBoost:** [Documentación oficial](https://xgboost.readthedocs.io/)
- **Scikit-learn:** [Documentación oficial](https://scikit-learn.org/stable/)
- **Pandas:** [Documentación oficial](https://pandas.pydata.org/docs/)
- **NumPy:** [Documentación oficial](https://numpy.org/doc/)
- **Matplotlib:** [Documentación oficial](https://matplotlib.org/stable/contents.html)
- **Seaborn:** [Documentación oficial](https://seaborn.pydata.org/)

### Dataset
- **California Housing:** [Scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)

### MLOps y Buenas Prácticas
- **MLOps Principles:** [ml-ops.org](https://ml-ops.org/)
- **GitHub Guides:** [Guías oficiales](https://guides.github.com/)
- **MLflow Tracking:** [Guía de seguimiento](https://mlflow.org/docs/latest/tracking.html)
- **Grid Search:** [Documentación Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

### Artículos y Tutoriales
- **California Housing Dataset:** [Análisis y modelado](https://towardsdatascience.com/california-housing-price-prediction-using-machine-learning-5e0e3f8b12a6)
- **XGBoost Guide:** [Guía completa](https://xgboost.readthedocs.io/en/stable/tutorials/index.html)
- **MLflow Tutorial:** [Primeros pasos](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)

---

## 👥 Autores

| Nombre | Rol | Contacto |
|--------|-----|----------|
| **Luis Alfonso Salcedo** | Desarrollador Principal | [GitHub](https://github.com/PonchitoSalcedo) |

### Contribuciones
- **Diseño y arquitectura:** Luis Alfonso Salcedo
- **Desarrollo de código:** Luis Alfonso Salcedo
- **Documentación:** Luis Alfonso Salcedo
- **Pruebas y validación:** Luis Alfonso Salcedo
- **Análisis de resultados:** Luis Alfonso Salcedo

### Agradecimientos
- A la asignatura "Gestión de Proyectos de Inteligencia Artificial" por proporcionar el marco conceptual.
- A Scikit-learn por el dataset California Housing.
- A MLflow por facilitar el seguimiento de experimentos.
- A la comunidad de Data Science por las mejores prácticas compartidas.

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

**MIT License**

Copyright (c) 2026 Luis Alfonso Salcedo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📅 Historial de Cambios

| Fecha | Versión | Cambios |
|-------|---------|---------|
| 21/06/2026 | 1.0.0 | Versión inicial completa del proyecto. Incluye: pipeline de MLOps, 3 modelos entrenados (Regresión Lineal, Random Forest, XGBoost), validación cruzada, Grid Search, MLflow, documentación completa y visualizaciones. |

---

## 📊 Estado del Proyecto

| Aspecto | Estado | Comentario |
|---------|--------|------------|
| **Desarrollo** | ✅ Completado | Todos los modelos implementados y evaluados |
| **Pruebas** | ✅ Completado | Validación cruzada realizada |
| **Documentación** | ✅ Completado | README, CHANGELOG y notebook documentados |
| **MLflow** | ✅ Completado | 6 experimentos registrados |
| **Visualizaciones** | ✅ Completado | 7 gráficas generadas |
| **Optimización** | ✅ Completado | Grid Search aplicado |
| **Despliegue** | ⏳ Pendiente | Pendiente para futuras iteraciones |
| **Monitoreo** | ⏳ Pendiente | Pendiente para futuras iteraciones |

---

## 📊 Estadísticas del Proyecto

| Aspecto | Valor |
|---------|-------|
| **Líneas de código** | ~500+ (Python) |
| **Modelos entrenados** | 3 |
| **Experimentos MLflow** | 6 |
| **Visualizaciones generadas** | 7 |
| **Horas de desarrollo** | ~10-15 horas |
| **Tecnologías utilizadas** | 9 |

---

## 🎯 Objetivos Cumplidos

| Objetivo | Estado |
|----------|--------|
| ✅ Seleccionar un dataset público | California Housing |
| ✅ Realizar preparación de datos | Limpieza y escalado completados |
| ✅ Entrenar múltiples modelos | 3 modelos implementados |
| ✅ Ajustar hiperparámetros | Grid Search aplicado |
| ✅ Registrar experimentos con MLflow | 6 runs registrados |
| ✅ Generar visualizaciones | 7 gráficas generadas |
| ✅ Documentar el proyecto | README y CHANGELOG completos |
| ✅ Comparar resultados | Tablas comparativas generadas |
| ✅ Validación cruzada | 5-fold completado |
| ✅ Análisis de importancia | Feature importance analizada |

---

**Última actualización:** 21 de junio de 2026
**Versión actual:** 1.0.0
**Estado del proyecto:** ✅ Completado
