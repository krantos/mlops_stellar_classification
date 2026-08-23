import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8800")

st.title("Stellar Classification")

st.subheader("Estado de la API")
if st.button("Probar conexión"):
    try:
        resp = requests.get(f"{API_URL}/", timeout=5)
        st.json(resp.json())
    except requests.RequestException as e:
        st.error(f"No se pudo conectar a la API: {e}")

st.subheader("Modelos registrados en MLflow")
if st.button("Listar modelos"):
    try:
        resp = requests.get(f"{API_URL}/modelos", timeout=5)
        st.json(resp.json())
    except requests.RequestException as e:
        st.error(f"No se pudo conectar a la API: {e}")

st.subheader("Modelo en producción")
if st.button("Consultar modelo en producción"):
    try:
        resp = requests.get(f"{API_URL}/modelos/produccion", timeout=5)
        data = resp.json()
        if resp.ok:
            st.json(data)
        else:
            st.error(data.get("detail", "No se pudo obtener el modelo en producción"))
    except requests.RequestException as e:
        st.error(f"No se pudo conectar a la API: {e}")

st.subheader("Nueva observación")
with st.form("observacion_form"):
    alpha = st.number_input("Alpha (ascensión recta)", format="%.6f")
    delta = st.number_input("Delta (declinación)", format="%.6f")
    u = st.number_input("u (magnitud ultravioleta)", format="%.6f")
    g = st.number_input("g (magnitud verde)", format="%.6f")
    r = st.number_input("r (magnitud roja)", format="%.6f")
    i = st.number_input("i (magnitud infrarroja cercana)", format="%.6f")
    z = st.number_input("z (magnitud infrarroja)", format="%.6f")
    redshift = st.number_input("Redshift", format="%.6f")
    enviado = st.form_submit_button("Enviar")

if enviado:
    payload = {
        "alpha": alpha,
        "delta": delta,
        "u": u,
        "g": g,
        "r": r,
        "i": i,
        "z": z,
        "redshift": redshift,
    }
    try:
        resp = requests.post(f"{API_URL}/observaciones", json=payload, timeout=5)
        data = resp.json()
        if resp.ok:
            st.success(data.get("message", "Predicción generada"))
            st.metric("Clase predicha", data.get("clase", "-"))
        elif "errores" in data:
            st.error(data.get("message", "Datos inválidos"))
            for error in data["errores"]:
                st.warning(f"{error['campo']}: {error['mensaje']}")
        else:
            st.error(data.get("detail", "No se pudo generar la predicción"))
    except requests.RequestException as e:
        st.error(f"No se pudo conectar a la API: {e}")
