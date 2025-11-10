import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import yfinance as yf

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="Predicción BBVA & Santander", layout="wide")

st.title("📈 Predicción de Acciones BBVA y Santander")
st.write("Interfaz interactiva para visualizar predicciones y tendencias de mercado.")

# ---------- SECCIÓN 1: INPUT ----------
ticker = st.selectbox("Selecciona el activo:", ["BBVA.MC", "SAN.MC"])
n_days = st.slider("Días de predicción futura", 1, 10, 4)

model_type = st.selectbox("Modelo recurrente:", ["SimpleRNN", "LSTM", "GRU"])

start_date = st.date_input("Fecha de inicio de los datos", dt.date(2020, 1, 1))
end_date = dt.date.today()

if st.button("🔄 Obtener y Predecir"):
    with st.spinner("Descargando datos..."):
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data[["Close"]].reset_index()

    # ---------- ESCALADO ----------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data["Close"].values.reshape(-1,1))

    timesteps = 60
    X = np.array([scaled[-timesteps:]]).reshape(1, timesteps, 1)

    # ---------- CARGAR MODELO ----------
    try:
        model_path = f"modelos_prediccion_noviembre/{ticker.split('.')[0]}_{model_type}_forecast.h5"
        model = load_model(model_path)
        pred_scaled = model.predict(X).flatten()
        pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        st.stop()

    # ---------- FECHAS FUTURAS ----------
    last_date = data["Date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    # ---------- GRÁFICOS ----------
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(data["Date"], data["Close"], label="Histórico", color="blue")
    ax.plot(future_dates, pred[:n_days], 'ro-', label="Predicción")
    ax.plot([data["Date"].iloc[-1], future_dates[0]], [data["Close"].iloc[-1], pred[0]], 'r--', alpha=0.6)
    ax.set_title(f"{ticker} - {model_type} (Predicción {n_days} días)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (€)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ---------- DATOS PREDICCIÓN ----------
    st.subheader("📊 Predicciones futuras")
    df_pred = pd.DataFrame({"Fecha": future_dates, "Predicción": pred[:n_days]})
    st.dataframe(df_pred)
