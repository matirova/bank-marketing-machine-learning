#  Predicción de Conversión en Campañas de Marketing Bancario

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

##  Descripción del Proyecto
Este proyecto implementa un pipeline completo de **Machine Learning de extremo a extremo** para optimizar una campaña de marketing directo de una institución financiera. El objetivo principal es predecir si un cliente suscribirá o no un **depósito a plazo fijo**, permitiendo al banco enfocar sus recursos comerciales en los clientes con mayor propensión de compra, reduciendo costos operativos y minimizando la fatiga del cliente.

El dataset consta de **45,211 registros** con un fuerte desbalance de clases (la tasa de conversión real es de aproximadamente el **11.7%**).

---

##  Flujo de Trabajo (Data Science Pipeline)

1. **Análisis Exploratorio de Datos (EDA):**
   * Verificación de integridad (0% valores nulos).
   * Análisis de estacionalidad mensual y su impacto en las conversiones.
   * Identificación de variables clave como el éxito en campañas pasadas (`poutcome_success`) y la duración de la llamada (`duration`).

2. **Preprocesamiento y Preparación de Datos:**
   * **One-Hot Encoding:** Transformación de variables categóricas aplicando `drop_first=True` para prevenir multicolinealidad.
   * **Train/Test Split Estratificado:** División 80/20 manteniendo la proporción de clases (`stratify=y`).
   * **Escalado (`StandardScaler`):** Normalización de variables numéricas aplicada **exclusivamente al set de entrenamiento** para evitar fugas de datos (*data leakage*).

3. **Modelado y Manejo de Desbalance:**
   * **Modelo 1 (Baseline):** Regresión Logística con pesos balanceados.
   * **Modelo 2:** Random Forest Classifier (con y sin SMOTE).
   * **Manejo Avanzado de Clases:** Aplicación de **SMOTE** (*Synthetic Minority Over-sampling Technique*) en el set de entrenamiento.
   * **Modelo 3 (Final Avanzado):** **XGBoost Classifier** utilizando `scale_pos_weight` dinámico para optimizar el gradiente potenciado.

---

##  Resultados y Comparativa de Modelos

| Modelo | Accuracy General | Precisión (Clase 1) | Recall / Sensibilidad (Clase 1) | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: |
| **Regresión Logística** | 84.6% | 42% | 81% | - |
| **Random Forest** | 90.4% | 69% | 33% | - |
| **Random Forest + SMOTE** | 89.9% | 56% | 59% | - |
| **XGBoost (Final)** | **87.3%** | **48%** | **82%** | **0.927** |

* **Logro Clave:** El modelo **XGBoost** logró un excelente equilibrio, alcanzando un **Recall del 82%** en la clase minoritaria (capaz de capturar a más de 8 de cada 10 clientes interesados reales) y un **AUC de 0.927**.

---

##  Principales Hallazgos (Feature Importance)
Mediante el análisis de importancia de variables del modelo XGBoost, se identificaron los factores que más influyen en la decisión del cliente:
1. **`poutcome_success` (16.1%):** El historial positivo en campañas previas es el predictor número uno de recompra.
2. **`contact_unknown` (14.0%):** El canal de contacto o el estado de registro desconocido condiciona fuertemente la probabilidad de éxito.
3. **Estacionalidad (`month_mar`, `month_jul`, etc.):** Meses como marzo y julio presentan tasas de conversión muy superiores al promedio anual.
4. **`duration` (5.2%):** La profundidad de la llamada comercial mantiene una relación directa con el interés del cliente.

---

## Recomendaciones de Negocio
* **Campañas Personalizadas por Historial:** Priorizar leads que posean antecedentes de éxito (`poutcome_success`) para maximizar el Retorno de Inversión (ROI).
* **Distribución Presupuestaria Estacional:** Concentrar los esfuerzos de telemarketing en los meses con mayor propensión histórica detectada por el modelo (marzo, julio, octubre, diciembre), evitando llamadas masivas inefectivas en meses de baja conversión.

---

##  Tecnologías y Librerías Utilizadas
* **Lenguaje:** Python
* **Manipulación y Análisis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
* **Visualización:** Matplotlib, Seaborn
* **Entorno:** Google Colab
