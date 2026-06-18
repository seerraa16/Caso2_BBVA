# Caso2_BBVA — Alpha-guard

Proyecto de análisis y predicción de series financieras de **BBVA** y **Banco Santander**, desarrollado como Caso 2. Combina un cuaderno de investigación (análisis exploratorio, econométrico y entrenamiento de modelos de Deep Learning) con un dashboard interactivo en Streamlit que genera predicciones a corto plazo y señales de inversión.

## ¿Qué hace este repositorio?

1. **Recopilación de datos**: descarga histórica (desde el 1 de enero de 2000) de las cotizaciones de BBVA y Santander vía `yfinance`, junto con macrovariables relevantes (Euribor, IPC, PIB, desempleo, VIX, IBEX, Brent, prima de riesgo, etc.).
2. **Análisis de series financieras** (`IntentoDefinitivo.ipynb`):
   - Descomposición STL de las series de BBVA y Santander.
   - Filtro Hodrick-Prescott (HP) sobre variables macro.
   - Detección de picos de estrés (rolling mean + z-score) en VIX, IBEX, Brent y prima de riesgo.
   - Identificación de fechas de shocks sistémicos (coincidencia de picos entre indicadores).
   - Correlaciones dinámicas entre BBVA, Santander y variables macro.
   - Estimación de volatilidad condicional (rolling y GARCH).
   - Test de cointegración entre BBVA y Santander.
   - Análisis de correlación cruzada con lags (¿un indicador anticipa a otro?).
   - Clusterización de periodos de crisis mediante PCA + KMeans.
   - Tabla resumen de fechas críticas para informe.
3. **Entrenamiento de modelos de predicción**: redes neuronales recurrentes (SimpleRNN, GRU y LSTM) entrenadas de forma univariada/autorregresiva para predecir el precio de cierre de BBVA y Santander a corto plazo. Los modelos entrenados se guardan en [modelos_prediccion_noviembre/](modelos_prediccion_noviembre/) en formato `.h5`.
4. **Dashboard interactivo** ([app.py](app.py)): aplicación Streamlit ("Alpha-guard") que:
   - Carga los modelos GRU (BBVA) y LSTM (Santander) ya entrenados.
   - Permite configurar el rango histórico y el número de días a predecir.
   - Muestra gráficas interactivas (Plotly) del histórico y la predicción.
   - Genera señales de inversión (🟢 Oportunidad, 🟡 Vigilar, 🔴 Riesgo) según la variación prevista.
   - Exporta un informe en PDF con las predicciones, señales y recomendaciones.

## Estructura del repositorio

| Archivo / carpeta | Descripción |
| --- | --- |
| [IntentoDefinitivo.ipynb](IntentoDefinitivo.ipynb) | Notebook principal con la recopilación de datos, análisis de series y entrenamiento de modelos. |
| [app.py](app.py) | Dashboard Streamlit con predicciones y señales de inversión. |
| [bbva.csv](bbva.csv) / [bbva_completo.csv](bbva_completo.csv) | Datos históricos de BBVA (precios y con macrovariables combinadas). |
| [santander.csv](santander.csv) / [santander_completo.csv](santander_completo.csv) | Datos históricos de Santander (precios y con macrovariables combinadas). |
| [modelos_prediccion_noviembre/](modelos_prediccion_noviembre/) | Modelos entrenados (SimpleRNN, GRU, LSTM) para BBVA y Santander en formato `.h5`. |
| [requirements.txt](requirements.txt) | Dependencias del proyecto. |

## Cómo ejecutar el dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

La aplicación carga los CSV y modelos desde el mismo directorio que `app.py`, permite seleccionar el rango de fechas histórico y el horizonte de predicción (1-5 días), y muestra resultados, señales e informe PDF descargable.
