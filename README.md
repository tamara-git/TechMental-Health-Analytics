# TechMental-Health-Analytics

Proyecto de Análisis Exploratorio de Datos (EDA) y Machine Learning enfocado en predecir si los problemas de salud mental interfieren en el rendimiento laboral de los empleados del sector tecnológico.

---

## 📋 Descripción del Proyecto
Este proyecto analiza el conjunto de datos de la encuesta **Mental Health in Tech Survey (2014)**. El objetivo principal es responder a la siguiente pregunta mediante algoritmos de aprendizaje automático: 
> *"Si tiene una condición de salud mental, ¿siente que interfiere con su trabajo?"* (`work_interfere`)

A través de un proceso riguroso de limpieza, transformación de variables y entrenamiento de modelos, se busca entender qué factores demográficos, laborales y corporativos tienen mayor peso en la interferencia de la salud mental en el ámbito IT.

---

## 🛠️ Tecnologías y Librerías Utilizadas
* **Python** 🐍
* **Pandas & NumPy:** Manipulación y procesamiento de datos.
* **Matplotlib & Seaborn:** Visualización de datos y gráficos estadísticos.
* **Scikit-Learn:** Preprocesamiento, estandarización y entrenamiento de modelos de Machine Learning.

---

## 🔍 Fases del Proyecto

1. **Análisis Exploratorio de Datos (EDA):**
   * Inspección inicial de más de 20 variables de la encuesta.
   * Visualización de distribuciones mediante gráficos de conteo y distribución.
   * Detección de valores nulos y registros atípicos (por ejemplo, limpieza y corrección de rangos de edad erróneos y unificación manual de más de 40 variantes en la variable de género).

2. **Limpieza y Transformación:**
   * Eliminación de variables sin valor predictivo (`Timestamp`, `state`, `comments`, etc.).
   * Codificación de variables categóricas utilizando **One-Hot Encoding** para transformarlas en formato numérico apto para los modelos.
   * Tratamiento específico de los valores nulos en la variable objetivo (`work_interfere`).

3. **Modelado y Evaluación:**
   Se entrenaron, evaluaron y compararon múltiples algoritmos de clasificación:
   * **Árboles de Decisión** (con pruebas en distintas profundidades para controlar el *overfitting*).
   * **Random Forest** (incluyendo ajuste de hiperparámetros como `n_estimators` y `max_depth`).
   * **Regresión Logística**.
   * **K-Nearest Neighbors (KNN)** (probando modelos con y sin estandarización de datos mediante `StandardScaler`).

---

## 📊 Modelos y Resultados
* Se evaluó el rendimiento de los modelos utilizando métricas de exactitud (**Accuracy**), matrices de confusión y reportes de clasificación detallados por clase (**Precision, Recall, F1-Score**).
* Se experimentó con agrupaciones alternativas de la variable objetivo para observar el comportamiento de las clases minoritarias frente a las mayoritarias.

---

## 🚀 Cómo Reproducir el Proyecto
Si deseas clonar y ejecutar este proyecto en tu entorno local:

1. Clona el repositorio:
   ```bash
   git clone [https://github.com/tu-usuario/nombre-de-tu-repo.git](https://github.com/tu-usuario/nombre-de-tu-repo.git)
