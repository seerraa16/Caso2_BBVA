import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import io
import datetime as dt
import os
import gdown  # Para descargar archivos de Google Drive

# =========================
# 🎨 CONFIGURACIÓN DE PÁGINA
# =========================
st.set_page_config(page_title="Timing-Advisor BBVA & Santander", layout="wide")
st.markdown("""
<style>
body {background-color: #0e1117; color: white;}
.stMetric {background-color: #1e2130; border-radius: 12px; padding: 15px;}
.stButton button {border-radius: 10px; background-color: #0096c7; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("⏱ Timing-Advisor: BBVA & Santander")
st.markdown("### Predicciones a corto plazo y señales de inversión con modelos GRU y LSTM.")

# =========================
# 📊 PARÁMETROS GLOBALES
# =========================
timesteps = 60
MAX_DATE = dt.date(2025, 10, 31)

# =========================
# 🧠 DESCARGA DE ARCHIVOS DESDE GOOGLE DRIVE
# =========================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(DATA_DIR, exist_ok=True)

# IDs de archivos en Google Drive
files_to_download = {
    "bbva_completo.csv": "1rZ0WkSHTNXd4F3bl4kHRt2XJgXzExAMP",
    "santander_completo.csv": "1SfvS0xjhR8H1J1u_X4w8jzVQ2F1K9bQW",
    "BBVA_GRU_forecast.h5": "1vHkH0n4X2KnOp2f9WxH_1VnU3Y5GQeJQ",
    "SANTANDER_LSTM_forecast.h5": "1kGx6Z9uPq9Iu1d8jL6m8Yx5zO7B1R4Vc"
}

for filename, file_id in files_to_download.items():
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)

# Rutas locales tras descarga
bbva_csv_path = os.path.join(DATA_DIR, "bbva_completo.csv")
san_csv_path = os.path.join(DATA_DIR, "santander_completo.csv")
bbva_model_path = os.path.join(DATA_DIR, "BBVA_GRU_forecast.h5")
san_model_path = os.path.join(DATA_DIR, "SANTANDER_LSTM_forecast.h5")

# =========================
# 🧠 FUNCIONES AUXILIARES
# =========================
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
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X = np.array([scaled[-timesteps:]]).reshape(1, timesteps, 1)
    pred_scaled = model.predict(X).flatten()
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=4), periods=len(pred))
    return future_dates[:n_days], pred[:n_days]

def plot_interactive(series, future_dates, pred, stock_name, days_hist):
    last = series[-days_hist:] if len(series) > days_hist else series
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=last.index, y=last.values,
                             mode='lines', name='Histórico',
                             line=dict(color='#00B4D8', width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=pred,
                             mode='lines+markers', name='Predicción',
                             line=dict(color='orange', dash='dot'),
                             marker=dict(size=8)))
    fig.update_layout(
        title=f"{stock_name} — Últimos {days_hist} días + Predicción (desde 03/11/2025)",
        xaxis_title="Fecha", yaxis_title="Precio (€)",
        hovermode="x unified", template="plotly_dark",
        height=400, margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def generate_signal(gru_pred, lstm_pred, last_close):
    change_gru = (gru_pred[0] - last_close) / last_close
    change_lstm = (lstm_pred[0] - last_close) / last_close
    if change_gru > 0.002 and change_lstm > 0.002:
        return "🟢 Oportunidad", "#006400"
    elif change_gru < -0.002 and change_lstm < -0.002:
        return "🔴 Riesgo", "#8B0000"
    else:
        return "🟡 Vigilar", "#8B8B00"

def generar_informe_pdf(bbva_signal, san_signal, bbva_pred, san_pred):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle("titulo", parent=styles["Heading1"], alignment=1,
                                 textColor=colors.HexColor("#0055a4"))
    normal_style = ParagraphStyle("normal", parent=styles["Normal"], fontSize=11, leading=16)

    elements.append(Paragraph("📊 Informe de Inversión — Timing-Advisor", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Fecha de generación: {dt.datetime.now().strftime('%d/%m/%Y %H:%M')}", normal_style))
    elements.append(Spacer(1, 18))

    def analisis_texto(signal, entidad):
        if "🟢" in signal:
            return f"{entidad}: Oportunidad de entrada. Riesgo bajo y tendencia positiva."
        elif "🔴" in signal:
            return f"{entidad}: Zona de riesgo. Prudencia y esperar confirmación."
        else:
            return f"{entidad}: Fase neutra. Mantener vigilancia."

    bbva_pred_str = ", ".join([f"{x:.2f}" for x in bbva_pred])
    san_pred_str = ", ".join([f"{x:.2f}" for x in san_pred])

    data = [
        ["Entidad", "Señal", "Interpretación", "Predicciones (€)"],
        ["BBVA", bbva_signal, analisis_texto(bbva_signal, "BBVA"), bbva_pred_str],
        ["Santander", san_signal, analisis_texto(san_signal, "Santander"), san_pred_str],
    ]

    table = Table(data, colWidths=[60, 60, 300, 120], rowHeights=[30, 60, 60])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0055a4")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
    ]))

    for i, signal in enumerate([bbva_signal, san_signal], start=1):
        if "🟢" in signal:
            bg = colors.HexColor("#00cc66")
        elif "🔴" in signal:
            bg = colors.HexColor("#ff4d4d")
        else:
            bg = colors.HexColor("#ffcc00")
        table.setStyle(TableStyle([('BACKGROUND', (1, i), (1, i), bg)]))

    elements.append(table)
    elements.append(Spacer(1, 18))

    rec_style = ParagraphStyle("rec", parent=styles["Heading2"],
                               textColor=colors.HexColor("#0070d1"), spaceBefore=12)
    elements.append(Paragraph("📈 Recomendación general", rec_style))

    if "🟢" in bbva_signal and "🟢" in san_signal:
        texto = "Escenario positivo para ambos valores. Momento favorable para inversores buscando exposición controlada."
    elif "🔴" in bbva_signal and "🔴" in san_signal:
        texto = "Fase de riesgo elevado para ambos valores. Se recomienda cautela y esperar estabilización."
    else:
        texto = "Señales mixtas. Vigilancia del mercado y selección de entradas controladas según la evolución de cada banco."
    elements.append(Paragraph(texto, normal_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =========================
# ⚙️ INTERFAZ DE USUARIO
# =========================
st.sidebar.header("⚙️ Configuración")
start_date = st.sidebar.date_input("Fecha inicio histórico", dt.date(2020, 1, 1))
end_date = st.sidebar.date_input("Fecha fin histórico", MAX_DATE, max_value=MAX_DATE)
n_future_days = st.sidebar.slider("Días a predecir", 1, 5, 5)
days_hist = st.sidebar.slider("Ver histórico de los últimos días", 30, 1000, 120, step=10)

# =========================
# 🚀 BOTÓN PRINCIPAL
# =========================
if st.button("🔮 Generar predicciones y señales"):
    st.info("Cargando datos y modelos...")

    # --- BBVA ---
    bbva_data = pd.read_csv(bbva_csv_path)
    bbva_data['Date'] = pd.to_datetime(bbva_data['Date'])
    bbva_data = bbva_data[(bbva_data['Date'] >= pd.Timestamp(start_date)) &
                          (bbva_data['Date'] <= pd.Timestamp(end_date))]
    bbva_data = bbva_data.rename(columns={"Close_BBVA.MC": "Close"}).set_index("Date")
    bbva_model = load_model_safe(bbva_model_path)
    bbva_future_dates, bbva_pred = predict_future(bbva_data["Close"], bbva_model, n_future_days)

    st.subheader("🏦 BBVA")
    plot_interactive(bbva_data["Close"], bbva_future_dates, bbva_pred, "BBVA", days_hist)

    # --- Santander ---
    san_data = pd.read_csv(san_csv_path)
    san_data['Date'] = pd.to_datetime(san_data['Date'])
    san_data = san_data[(san_data['Date'] >= pd.Timestamp(start_date)) &
                        (san_data['Date'] <= pd.Timestamp(end_date))]
    san_data = san_data.rename(columns={"Close_SAN.MC": "Close"}).set_index("Date")
    san_model = load_model_safe(san_model_path)
    san_future_dates, san_pred = predict_future(san_data["Close"], san_model, n_future_days)

    st.subheader("🏛 Santander")
    plot_interactive(san_data["Close"], san_future_dates, san_pred, "Santander", days_hist)

    # --- Señales ---
    st.markdown("## 🚦 Señales de Inversión")
    bbva_signal, bbva_color = generate_signal(bbva_pred, bbva_pred, bbva_data["Close"][-1])
    san_signal, san_color = generate_signal(san_pred, san_pred, san_data["Close"][-1])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div style='background-color:{bbva_color};padding:1em;border-radius:10px;text-align:center;'>"
                    f"<h3>BBVA</h3><h1>{bbva_signal}</h1></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='background-color:{san_color};padding:1em;border-radius:10px;text-align:center;'>"
                    f"<h3>Santander</h3><h1>{san_signal}</h1></div>", unsafe_allow_html=True)

    # --- PDF ---
    st.markdown("### 🧾 Generar informe detallado")
    buffer = generar_informe_pdf(bbva_signal, san_signal, bbva_pred, san_pred)
    st.download_button("📥 Descargar Informe PDF", buffer, file_name="Informe_TimingAdvisor.pdf")

# =========================
# ⚙️ FOOTER
# =========================
st.markdown("---")
st.caption("💡 Timing-Advisor v4.4 — Dashboard financiero con RNNs, GRU, LSTM, señales y PDF con predicciones numéricas.")
