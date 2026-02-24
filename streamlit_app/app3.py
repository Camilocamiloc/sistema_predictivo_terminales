# =========================
# app.py - Sistema Predictivo Terminales (Versión Profesional)
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go

# ------------------------
# 1️⃣ Configuración general
# ------------------------
st.set_page_config(
    page_title="Sistema Predictivo - Terminales Medellín",
    layout="wide"
)

st.title("🚦 Sistema Predictivo de Flujo Terrestre - Medellín")
st.markdown(
    "Proyección horaria de pasajeros con identificación automática de horas pico "
    "para soporte a decisiones operativas."
)

# ------------------------
# 2️⃣ Sidebar
# ------------------------
st.sidebar.header("🔎 Configuración")

terminal = st.sidebar.selectbox(
    "Seleccione terminal",
    ["TERMINAL NORTE", "TERMINAL SUR"]
)

fecha_seleccion = st.sidebar.date_input(
    "Seleccione fecha",
    value=datetime(2025,1,1),
    min_value=datetime(2025,1,1),
    max_value=datetime(2026,12,31)
)

# ------------------------
# 3️⃣ Cargar modelos
# ------------------------
model_norte = joblib.load("train_models/modelo_norte.pkl")
model_sur   = joblib.load("train_models/modelo_sur.pkl")

model = model_norte if terminal == "TERMINAL NORTE" else model_sur

# ------------------------
# 4️⃣ Crear rango horario
# ------------------------
def crear_rango_horas(fecha):
    horas = pd.date_range(
        start=pd.Timestamp(fecha.year, fecha.month, fecha.day, 6),
        end=pd.Timestamp(fecha.year, fecha.month, fecha.day, 21),
        freq='H'
    )
    df_pred = pd.DataFrame({"ds": horas})
    df_pred["is_weekend"] = df_pred["ds"].dt.dayofweek.isin([5,6]).astype(int)
    return df_pred

df_pred = crear_rango_horas(fecha_seleccion)

# ------------------------
# 5️⃣ Ajuste logístico si aplica
# ------------------------
if hasattr(model, "growth") and model.growth == "logistic":
    df_pred["floor"] = 0
    df_pred["cap"] = model.history["y"].max() * 1.2

# ------------------------
# 6️⃣ Predicción
# ------------------------
forecast = model.predict(df_pred)

if hasattr(model, "y_log_transformed") and model.y_log_transformed:
    forecast["yhat"] = np.expm1(forecast["yhat"])

forecast["yhat"] = forecast["yhat"].round().astype(int)

df_display = forecast[["ds","yhat"]].rename(
    columns={"ds":"Hora","yhat":"Pasajeros"}
)

# ------------------------
# 7️⃣ KPIs Ejecutivos
# ------------------------
total_pasajeros = df_display["Pasajeros"].sum()
hora_pico_row = df_display.loc[df_display["Pasajeros"].idxmax()]
hora_pico = hora_pico_row["Hora"].strftime("%H:%M")
max_pasajeros = hora_pico_row["Pasajeros"]
promedio_hora = int(df_display["Pasajeros"].mean())

st.markdown("### 📊 Indicadores Clave del Día")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Proyectado", f"{total_pasajeros:,}")
col2.metric("Hora Pico", hora_pico)
col3.metric("Pasajeros Hora Pico", f"{max_pasajeros:,}")
col4.metric("Promedio por Hora", f"{promedio_hora:,}")

# ------------------------
# 8️⃣ Gráfico Profesional
# ------------------------
st.markdown("### 📈 Proyección Horaria")

fig = go.Figure()

# Área sombreada
fig.add_trace(go.Scatter(
    x=df_display["Hora"],
    y=df_display["Pasajeros"],
    fill='tozeroy',
    mode='lines',
    line=dict(color='#1f77b4', width=3),
    name='Pasajeros'
))

# Marcador especial hora pico
fig.add_trace(go.Scatter(
    x=[hora_pico_row["Hora"]],
    y=[max_pasajeros],
    mode='markers',
    marker=dict(color='red', size=12),
    name='Hora Pico'
))

fig.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Hora del Día",
    yaxis_title="Número de Pasajeros",
    hovermode="x unified",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------
# 9️⃣ Tablas debajo del gráfico
# ------------------------
st.markdown("### 📋 Detalle Horario")

st.dataframe(df_display, use_container_width=True)

st.markdown("### ⏰ Top 3 Horas de Mayor Afluencia")

top_horas = df_display.sort_values(
    "Pasajeros", ascending=False
).head(3)

st.dataframe(top_horas, use_container_width=True)
