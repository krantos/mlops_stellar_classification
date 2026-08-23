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
