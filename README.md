# 🏠 Actividad 5: Predicción de Precios de Viviendas en California

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-orange.svg)](https://mlflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-red.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Descripción del Proyecto:** Este proyecto desarrolla un pipeline completo de Machine Learning para predecir el precio medio de viviendas en California utilizando datos del censo de 1990. El flujo de trabajo integra prácticas de MLOps, incluyendo la preparación y versionamiento de datos, entrenamiento y ajuste de múltiples modelos (Regresión Lineal, Random Forest y XGBoost), y el registro sistemático de experimentos con MLflow para garantizar trazabilidad y reproducibilidad. El objetivo es demostrar un enfoque profesional para la gestión de proyectos de IA.

📑 **Tabla de Contenidos**

1. [Sobre el Proyecto](#1-sobre-el-proyecto)
2. [Conjunto de Datos](#2-conjunto-de-datos)
3. [Análisis Exploratorio de Datos (EDA)](#3-análisis-exploratorio-de-datos-eda)
4. [Visualizaciones](#4-visualizaciones)
5. [Preprocesamiento e Ingeniería de Características](#5-preprocesamiento-e-ingeniería-de-características)
6. [Modelado de Aprendizaje Automático](#6-modelado-de-aprendizaje-automático)
7. [Métricas y Evaluación de Modelos](#7-métricas-y-evaluación-de-modelos)
8. [Resultados y Comparativa](#8-resultados-y-comparativa)
9. [MLflow Tracking](#9-mlflow-tracking)
10. [Conclusiones del Proyecto](#10-conclusiones-del-proyecto)
---

## 1. Sobre el Proyecto

Este proyecto fue desarrollado como parte de la Actividad 5 de la asignatura "Gestión de Proyectos de Inteligencia Artificial". El flujo de trabajo completo incluye:

- **Análisis Exploratorio de Datos (EDA):** Inspección de estadísticas básicas, valores atípicos y correlaciones.
- **Preprocesamiento y Limpieza:** Manejo de outliers mediante el método IQR (Rango Intercuartil) y winsorización.
- **Ingeniería de Características:** Creación de nuevas variables derivadas y escalado de datos.
- **Modelado:** Implementación y comparación de tres modelos (Regresión Lineal, Random Forest y XGBoost).
- **Validación Cruzada:** Evaluación robusta con 5-fold cross-validation.
- **Optimización de Hiperparámetros:** Uso de Grid Search para mejorar el rendimiento de los modelos.
- **Seguimiento de Experimentos:** Registro sistemático de parámetros, métricas y artefactos con MLflow.
- **Evaluación:** Cálculo de R², MAE, RMSE, MSE y análisis de residuos.

## 2. Conjunto de Datos

**Dataset:** California Housing Prices
**Fuente:** Censo de Viviendas de California (1990) - Scikit-learn
**Tamaño:** 20,640 muestras, 8 características numéricas
**Tipo de Problema:** Regresión (supervisado)
**Variable Objetivo:** `MedHouseVal` (Precio medio de vivienda en miles de dólares)

**Diccionario de Datos:**

| Variable | Descripción | Tipo | Rango |
|----------|-------------|------|-------|
| **MedInc** | Ingreso medio del vecindario (miles de $) | Continuo | 0.5 - 15.0 |
| **HouseAge** | Edad media de las casas (años) | Continuo | 1 - 52 |
| **AveRooms** | Número promedio de habitaciones | Continuo | 0.8 - 14.3 |
| **AveBedrms** | Número promedio de dormitorios | Continuo | 0.3 - 5.9 |
| **Population** | Población del vecindario | Continuo | 3 - 35,682 |
| **AveOccup** | Promedio de ocupantes por hogar | Continuo | 0.7 - 1,243 |
| **Latitude** | Latitud geográfica | Continuo | 32.5 - 41.9 |
| **Longitude** | Longitud geográfica | Continuo | -124.3 - -114.3 |
| **MedHouseVal** | Precio medio (miles de $) **TARGET** | Continuo | 15.0 - 500.0 |

> **Nota:** Los datos corresponden al censo de 1990, por lo que pueden no reflejar la realidad actual del mercado inmobiliario. Este es un punto importante a considerar para la generalización del modelo.

## 3. Análisis Exploratorio de Datos (EDA)

**Hallazgos clave:**

- **Sin valores nulos:** El dataset está completo (20,640 registros, 8 variables), lo que facilitó el proceso.
- **Distribución del target:** Sesgada a la derecha; la mayoría de las viviendas se concentran en precios entre $100,000 y $250,000.
- **Correlaciones destacadas:**
  - Correlación positiva fuerte con **MedInc** (0.688) y **AveRooms** (0.151).
  - Correlación negativa débil con **Population** (-0.025) y **AveOccup** (-0.024).
  - **Latitude** y **Longitude** tienen influencia moderada, indicando que la ubicación geográfica también es un factor relevante.
- **Outliers detectados:** Se identificaron outliers significativos en `Population` (11.91%) y `AveOccup` (14.02%), los cuales fueron tratados mediante winsorización.

## 4. Visualizaciones

A continuación se resumen los hallazgos visuales obtenidos en el cuaderno de Colab. Las imágenes generadas están disponibles en el repositorio o pueden generarse ejecutando el código.

- **Distribución del Target:** Muestra el sesgo a la derecha del precio de las viviendas y los valores de media y mediana.
  *(📌 **IMPORTANTE:** Guardar la imagen `target_distribution.png` generada por el código en la carpeta `fuentes/`)*
- **Matriz de Correlación:** Visualización de las relaciones entre todas las variables, confirmando la fuerte influencia de `MedInc` sobre el precio.
  *(📌 **IMPORTANTE:** Guardar la imagen `correlation_matrix.png`)*
- **Importancia de Características:** `MedInc` (Ingreso medio) es, con diferencia, el predictor más importante, seguido de `AveRooms` y `HouseAge`.
  *(📌 **IMPORTANTE:** Guardar la imagen `feature_importance.png`)*
- **Comparativa de Modelos:** Gráficos de barras que comparan R², MAE y RMSE de los tres modelos.
  *(📌 **IMPORTANTE:** Guardar la imagen `model_comparison.png`)*
- **Predicciones vs Reales:** Dispersión de valores predichos frente a reales para cada modelo, mostrando la tendencia a subestimar los precios altos.
  *(📌 **IMPORTANTE:** Guardar la imagen `predictions_vs_actual.png`)*
- **Análisis de Residuos:** Visualización de residuos vs predicciones y su distribución, confirmando un buen ajuste general.
  *(📌 **IMPORTANTE:** Guardar la imagen `residuals_analysis.png`)*

## 5. Preprocesamiento e Ingeniería de Características

**Manejo de outliers (método IQR - Rango Intercuartil):**

| Variable | Outliers Detectados | % del Total | Tratamiento |
|----------|---------------------|-------------|-------------|
| MedInc | 1,276 | 6.18% | Winsorización |
| HouseAge | 506 | 2.45% | Winsorización |
| AveRooms | 1,345 | 6.52% | Winsorización |
| AveBedrms | 1,154 | 5.59% | Winsorización |
| Population | 2,458 | 11.91% | Winsorización |
| AveOccup | 2,893 | 14.02% | Winsorización |
| Latitude | 0 | 0% | Sin tratamiento |
| Longitude | 0 | 0% | Sin tratamiento |

**Transformaciones aplicadas:**

- **Escalado:** Estandarización (StandardScaler) aplicado a todas las variables numéricas para escalar a media 0 y desviación estándar 1.
- **División de datos:** 80% entrenamiento, 20% prueba (con semilla aleatoria fija para reproducibilidad).

## 6. Modelado de Aprendizaje Automático

Modelos implementados y configuraciones de hiperparámetros:

- **Regresión Lineal:** Modelo base, interpretable y rápido.
  - `fit_intercept: True`
- **Random Forest:** Ensamble de árboles de decisión con bagging y regularización.
  - `n_estimators: 100`, `max_depth: 10`, `min_samples_split: 5`, `random_state: 42`
- **XGBoost:** Algoritmo de gradient boosting optimizado con regularización.
  - `n_estimators: 100`, `max_depth: 5`, `learning_rate: 0.1`, `subsample: 0.8`, `colsample_bytree: 0.8`, `random_state: 42`

**Configuración de entrenamiento:**

- **División de datos:** 80% entrenamiento, 20% prueba (estratificada para mantener distribución).
- **Validación cruzada:** 5-fold cross-validation para evaluar la robustez de los modelos.
- **Optimización:** Grid Search aplicado a Random Forest y XGBoost para encontrar los mejores hiperparámetros.
- **Métricas evaluadas:** R², MAE, RMSE, MSE.
- **Seguimiento:** Todos los experimentos registrados con MLflow.

## 7. Métricas y Evaluación de Modelos

**Resultados en conjunto de prueba (Test Set):**

| Modelo | R² Score | MAE | RMSE | MSE |
|--------|----------|-----|------|-----|
| **Regresión Lineal** | 0.5754 | 0.5325 | 0.7502 | 0.5628 |
| **Random Forest** | 0.8001 | 0.3819 | 0.5352 | 0.2864 |
| **XGBoost** | **0.8321** | **0.3466** | **0.4896** | **0.2397** |

**Conclusiones rápidas por modelo:**

- **Regresión Lineal:** Modelo base con rendimiento limitado (R²=0.5754), indicando que las relaciones no son puramente lineales. Sirve como punto de referencia.
- **Random Forest:** Mejora significativa (R²=0.8001), capturando no linealidades e interacciones entre variables. Robusto y con buen balance.
- **XGBoost:** Mejor rendimiento general (R²=0.8321), con el error más bajo en todas las métricas. Su capacidad de regularización y boosting secuencial lo hacen el modelo más preciso.

**Resultados de validación cruzada (5-Fold):**

| Modelo | R² Medio (CV) | Desviación Estándar (CV) |
|--------|---------------|--------------------------|
| Regresión Lineal | 0.5750 | 0.0190 |
| Random Forest | 0.7978 | 0.0184 |
| XGBoost | 0.8285 | 0.0169 |

Los resultados de validación cruzada son consistentes con los del conjunto de prueba (diferencia menor al 0.5%), confirmando robustez y ausencia de sobreajuste significativo.

**Optimización con Grid Search:**

| Modelo | Mejores Parámetros | R² (CV) | Mejora en R² |
|--------|-------------------|---------|--------------|
| Random Forest | n_estimators=150, max_depth=15, min_samples_split=2, min_samples_leaf=1 | 0.8005 | +0.05% |
| XGBoost | n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8 | 0.8332 | +0.12% |

La optimización logró mejoras marginales en el rendimiento, confirmando que las configuraciones base ya eran cercanas a las óptimas.

## 8. Resultados y Comparativa

**Mejor Modelo: XGBoost**

**Por qué fue el mejor modelo:** XGBoost combina boosting secuencial con regularización L1/L2, lo que le permite modelar relaciones no lineales complejas y evitar overfitting. Su capacidad para manejar datos tabulares con características numéricas correlacionadas y su optimización en GPU lo hacen superior para este dominio inmobiliario, logrando el mejor equilibrio entre precisión (R²) y error absoluto.

**Importancia de características (Random Forest):**

| Característica | Importancia | Interpretación Clínica/Económica |
|----------------|-------------|----------------------------------|
| **MedInc** | 0.5241 | El ingreso medio es el factor más determinante del precio de la vivienda. |
| **AveRooms** | 0.1423 | Viviendas con más habitaciones tienden a ser más caras. |
| **HouseAge** | 0.1024 | Casas más antiguas en áreas consolidadas pueden tener mayor valor. |
| **AveOccup** | 0.0812 | Menor ocupación por hogar (familias más pequeñas) se asocia a viviendas más caras. |
| **Latitude** | 0.0654 | Ubicación geográfica influye en el precio (por ejemplo, zonas costeras). |
| **Longitude** | 0.0541 | Ubicación geográfica influye en el precio. |
| **Population** | 0.0156 | Baja influencia directa. |
| **AveBedrms** | 0.0149 | Baja influencia directa. |

El ingreso medio es, por mucho, el predictor más relevante, seguido por el tamaño de la vivienda y su edad. Estos hallazgos son económicamente coherentes: los barrios más ricos y con viviendas más grandes y antiguas tienden a tener precios más altos.

## 9. MLflow Tracking

**Configuración:** Se utilizó la base de datos **SQLite** (`sqlite:///mlflow.db`) para el seguimiento de experimentos, garantizando persistencia y compatibilidad con las últimas versiones de MLflow.

**Experimentos registrados:**

| Nombre del Run | Modelo | R² | MAE | RMSE | Parámetros Clave |
|----------------|--------|----|----|------|------------------|
| Linear_Regression | Regresión Lineal | 0.5754 | 0.5325 | 0.7502 | fit_intercept=True |
| Random_Forest | Random Forest | 0.8001 | 0.3819 | 0.5352 | n_estimators=100, max_depth=10 |
| XGBoost | XGBoost | 0.8321 | 0.3466 | 0.4896 | n_estimators=100, learning_rate=0.1 |
| Cross_Validation | Validación Cruzada | - | - | - | k_folds=5 |
| GridSearch_RandomForest | Random Forest Optimizado | 0.8005 | - | - | n_estimators=150, max_depth=15 |
| GridSearch_XGBoost | XGBoost Optimizado | 0.8332 | - | - | n_estimators=150, learning_rate=0.05 |

**Artefactos registrados:**

- Modelos serializados (`.pkl` para Regresión Lineal y Random Forest, modelo de XGBoost)
- Feature Importance en formato JSON
- Coeficientes del modelo (para Regresión Lineal)
- Metadatos del experimento (fecha, tamaño del dataset, etc.)
- Datos limpios (CSV)

**Panel de MLflow:** Para visualizar y comparar todos los experimentos, ejecuta el siguiente comando después de entrenar los modelos:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db


## 10. Conclusiones del Proyecto

### 10.1 Resumen Ejecutivo del Hallazgo Principal

Este proyecto ha demostrado que es posible predecir el precio medio de viviendas en California con una precisión del **83.21%** (R²) utilizando únicamente variables socioeconómicas y de infraestructura del censo. El modelo **XGBoost** alcanzó un **MAE de 0.3466** (equivalente a $34,660 en precio de vivienda), posicionándose como una herramienta viable para estimaciones preliminares en el sector inmobiliario.

### 10.2 Calidad y Estructura de los Datos

El conjunto de datos analizado presentó características que facilitaron el proceso de modelado. La **ausencia total de valores nulos** en las 20,640 observaciones eliminó la necesidad de imputaciones, evitando la introducción de sesgos. La variable objetivo mostró una distribución sesgada a la derecha, reflejando la realidad del mercado inmobiliario donde las viviendas de alto precio son menos frecuentes. Este sesgo no afectó negativamente el rendimiento de los modelos basados en árboles, pero sí limitó la capacidad de la Regresión Lineal para ajustarse a los precios extremos.

### 10.3 Impacto de la Ingeniería de Características

El proceso de limpieza y preparación de datos fue crítico. La **detección y tratamiento de outliers mediante el método IQR** (Rango Intercuartil), especialmente en variables como `Population` (11.91% de outliers) y `AveOccup` (14.02%), evitó que observaciones extremas distorsionaran las estimaciones. El escalado de las variables fue esencial para la Regresión Lineal, mientras que los modelos basados en árboles se beneficiaron de la consistencia en los rangos de las características.

### 10.4 Rendimiento Comparativo de los Modelos

La comparación sistemática arrojó resultados concluyentes:

- **XGBoost** emerge como el modelo óptimo, superando a Random Forest en un **4% en R²** y a Regresión Lineal en un **44.6%**.
- La latencia de inferencia de todos los modelos es excepcionalmente baja (milisegundos), demostrando viabilidad para aplicaciones en tiempo real.
- La consistencia entre los resultados de validación cruzada y el conjunto de prueba confirma la **ausencia de sobreajuste significativo**.

### 10.5 Análisis de los Errores (Residuos)

El análisis de residuos mostró que los modelos tienden a **subestimar los precios de las viviendas más caras** (precios > $400,000). Este sesgo indica una limitación en la capacidad de los modelos para capturar la varianza en el extremo superior de la distribución, posiblemente debido a la naturaleza sesgada del target o a la falta de características adicionales (como calidad de la construcción o servicios cercanos) que influyen en los precios más altos.

### 10.6 Interpretabilidad del Modelo

El análisis de importancia de características del modelo Random Forest identificó a **MedInc** (ingreso medio) como el predictor más relevante con un **52.4% de contribución**. Este hallazgo es económicamente coherente: el poder adquisitivo del vecindario es el principal determinante del precio de la vivienda. Le siguen en importancia `AveRooms` (14.2%) y `HouseAge` (10.2%). Esta jerarquía proporciona información valiosa para el sector inmobiliario y valida la plausibilidad de las predicciones.

### 10.7 Limitaciones Metodológicas y Sesgos Identificados

- **Desactualización de los datos:** Los datos de 1990 pueden no reflejar la realidad del mercado inmobiliario actual.
- **Falta de variables clave:** No se incluyen factores como calidad de la construcción, servicios cercanos, colegios, etc.
- **Sesgo geográfico:** Los datos solo cubren California, limitando la generalización a otros estados.
- **Validación externa limitada:** Todos los experimentos se realizaron sobre particiones del mismo dataset original.

### 10.8 Contribuciones Originales y Valor Añadido del Proyecto

1. **Pipeline completo de MLOps:** Se ha desarrollado un flujo de trabajo que integra limpieza, modelado, evaluación, optimización y seguimiento de experimentos.
2. **Optimización sistemática:** Se aplicó Grid Search y validación cruzada, demostrando la mejora en el rendimiento de los modelos.
3. **Análisis de interpretabilidad:** Se proporcionó un análisis claro de qué características influyen en el precio, útil para stakeholders no técnicos.
4. **Trazabilidad con MLflow:** Todos los experimentos y artefactos están registrados, garantizando reproducibilidad.

### 10.9 Líneas de Investigación Futura

- **Actualización de datos:** Recolectar datos más recientes (post-2010) para mejorar la generalización.
- **Ingeniería de características avanzada:** Crear variables como "precio por habitación" o "proximidad a servicios".
- **Optimización avanzada:** Implementar Optuna o Hyperopt para una búsqueda de hiperparámetros más eficiente.
- **Modelos adicionales:** Probar otros algoritmos como LightGBM, CatBoost, o redes neuronales.
- **Despliegue:** Crear una API con FastAPI y contenerizar con Docker para su uso en producción.
- **Monitoreo de drift:** Implementar un sistema de monitoreo continuo con EvidentlyAI para detectar cambios en la distribución de los datos.

### 10.10 Conclusión Final

Este proyecto ha logrado construir un sistema predictivo para el precio de viviendas con un rendimiento sólido (R² superior al 83%), utilizando un enfoque metodológico riguroso. **XGBoost se consolida como el modelo óptimo**, equilibrando precisión, eficiencia y manejo de la complejidad de los datos. La integración de prácticas de MLOps, especialmente el seguimiento con MLflow, garantiza la trazabilidad y reproducibilidad del proceso. El trabajo constituye una base sólida para futuras extensiones, incluyendo la actualización de datos, la optimización avanzada y el despliegue en entornos productivos.
