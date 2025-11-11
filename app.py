import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import os

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="Timing-Advisor BBVA & Santander", layout="wide")
st.title("⏱ Timing-Advisor: BBVA & Santander")
st.write("Predicciones cortas + señales de timing combinadas con GRU, LSTM y puntos de inflexión.")

# ---------- PARÁMETROS FIJOS ----------
timesteps = 60
models_dir = "modelos_prediccion_noviembre"

# ---------- FUNCIONES ----------
def load_model_safe(path):
    if not os.path.exists(path):
        st.error(f"No se encontró el modelo: {path}")
        st.stop()
    try:
        model = load_model(path)
    except:
        model = load_model(path, compile=False)
        model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

def predict_future(series, model, n_days=5):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    if len(scaled) < timesteps:
        st.error(f"No hay suficientes datos históricos (mínimo {timesteps} días).")
        st.stop()
    X = np.array([scaled[-timesteps:]]).reshape(1, timesteps, 1)
    pred_scaled = model.predict(X).flatten()
    pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=len(pred))
    return future_dates[:n_days], pred[:n_days]

def plot_last30(series, future_dates, pred, stock_name, offset_days=3):
    pred = np.array(pred).flatten()
    last30 = series[-30:] if len(series) >= 30 else series
    last30_index_offset = last30.index + pd.Timedelta(days=offset_days)
    future_dates_offset = future_dates + pd.Timedelta(days=offset_days)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(last30_index_offset, last30.values, label="Histórico últimos 30 días", color="blue")
    ax.plot(future_dates_offset, pred, 'ro-', label="Predicción")
    ax.plot([last30_index_offset[-1], future_dates_offset[0]], 
            [last30.values[-1], pred[0]], 'r--', alpha=0.6)
    ax.set_title(f"{stock_name} - Últimos 30 días + Predicción")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (€)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    return future_dates_offset

def generate_signal(gru_pred, lstm_pred, last_close):
    change_gru = (gru_pred[0] - last_close)/last_close
    change_lstm = (lstm_pred[0] - last_close)/last_close
    if change_gru > 0.002 and change_lstm > 0.002:
        return "🟢 Oportunidad"
    elif change_gru < -0.002 and change_lstm < -0.002:
        return "🔴 Riesgo"
    else:
        return "🟡 Vigilar"

# ---------- INTERFAZ LATERAL ----------
st.sidebar.header("Configuración")
start_date = st.sidebar.date_input("Fecha inicio histórico", dt.date(2020,1,1))
end_date = st.sidebar.date_input("Fecha fin histórico", dt.date.today())
n_future_days = st.sidebar.slider("Días a predecir", 1, 5, 5)

# ---------- BOTÓN PRINCIPAL ----------
if st.button("🔮 Generar predicciones y señales"):
    st.info("Cargando datos y modelos...")

    # -------- BBVA --------
    bbva_data = pd.read_csv("bbva_completo.csv")
    bbva_data['Date'] = pd.to_datetime(bbva_data['Date'])
    bbva_data = bbva_data[(bbva_data['Date'] >= pd.Timestamp(start_date)) & (bbva_data['Date'] <= pd.Timestamp(end_date))]
    bbva_data = bbva_data.rename(columns={"Close_BBVA.MC":"Close"}).set_index("Date")
    bbva_model = load_model_safe(os.path.join(models_dir, "BBVA_GRU_forecast.h5"))

    bbva_future_dates, bbva_pred = predict_future(bbva_data["Close"], bbva_model, n_future_days)
    bbva_future_dates_offset = plot_last30(bbva_data["Close"], bbva_future_dates, bbva_pred, "BBVA")
    st.dataframe(pd.DataFrame({"Fecha": bbva_future_dates_offset, "Predicción (€)": bbva_pred}))

    # -------- Santander --------
    san_data = pd.read_csv("santander_completo.csv")
    san_data['Date'] = pd.to_datetime(san_data['Date'])
    san_data = san_data[(san_data['Date'] >= pd.Timestamp(start_date)) & (san_data['Date'] <= pd.Timestamp(end_date))]
    san_data = san_data.rename(columns={"Close_SAN.MC":"Close"}).set_index("Date")
    san_model = load_model_safe(os.path.join(models_dir, "SANTANDER_LSTM_forecast.h5"))

    san_future_dates, san_pred = predict_future(san_data["Close"], san_model, n_future_days)
    san_future_dates_offset = plot_last30(san_data["Close"], san_future_dates, san_pred, "Santander")
    st.dataframe(pd.DataFrame({"Fecha": san_future_dates_offset, "Predicción (€)": san_pred}))

    # ---------- Semáforo de señales ----------
    st.subheader("📊 Señales de Timing Combinadas")
    bbva_signal = generate_signal(bbva_pred, bbva_pred, bbva_data["Close"][-1])
    san_signal = generate_signal(san_pred, san_pred, san_data["Close"][-1])
    col1, col2 = st.columns(2)
    col1.metric("BBVA", bbva_signal)
    col2.metric("Santander", san_signal)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Desarrollado por Alejandro Serrano Catalina — Timing-Advisor con señales de corto plazo y dashboard intuitivo")
