import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8800")
PUBLIC_HOST = os.getenv("PUBLIC_HOST", "localhost")

ENLACES_SERVICIOS = {
    "Airflow": f"http://{PUBLIC_HOST}:8080",
    "MLflow": f"http://{PUBLIC_HOST}:5001",
    "MinIO": f"http://{PUBLIC_HOST}:9001",
    "API (docs)": f"http://{PUBLIC_HOST}:8800/docs",
}
ETIQUETAS_SALUD = {
    "fastapi": "FastAPI",
    "mlflow": "MLflow",
    "airflow": "Airflow",
    "minio": "MinIO",
}

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

EJEMPLOS = {
    "GALAXY": {
        "alpha": 135.69, "delta": 32.49,
        "u": 23.87, "g": 22.27, "r": 20.39, "i": 19.16, "z": 18.79,
        "redshift": 0.644,
    },
    "QSO": {
        "alpha": 27.65, "delta": 10.05,
        "u": 21.19, "g": 20.87, "r": 20.62, "i": 20.35, "z": 20.27,
        "redshift": 1.997,
    },
    "STAR": {
        "alpha": 328.16, "delta": -0.65,
        "u": 18.66, "g": 17.35, "r": 16.83, "i": 16.66, "z": 16.61,
        "redshift": 0.0001,
    },
}

st.subheader("Nueva observación")

st.caption("Cargar un ejemplo:")
columnas = st.columns(len(EJEMPLOS))
for columna, (nombre, valores) in zip(columnas, EJEMPLOS.items()):
    if columna.button(nombre):
        for campo, valor in valores.items():
            st.session_state[campo] = valor

with st.form("observacion_form"):
    alpha = st.number_input("Alpha (ascensión recta)", format="%.6f", key="alpha")
    delta = st.number_input("Delta (declinación)", format="%.6f", key="delta")
    u = st.number_input("u (magnitud ultravioleta)", format="%.6f", key="u")
    g = st.number_input("g (magnitud verde)", format="%.6f", key="g")
    r = st.number_input("r (magnitud roja)", format="%.6f", key="r")
    i = st.number_input("i (magnitud infrarroja cercana)", format="%.6f", key="i")
    z = st.number_input("z (magnitud infrarroja)", format="%.6f", key="z")
    redshift = st.number_input("Redshift", format="%.6f", key="redshift")
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

            probabilidades = data.get("probabilidades", {})
            if probabilidades:
                st.caption("Probabilidad por clase")
                for clase, prob in sorted(probabilidades.items(), key=lambda item: -item[1]):
                    st.write(f"{clase}: {prob:.1%}")
                    st.progress(prob)
        elif "errores" in data:
            st.error(data.get("message", "Datos inválidos"))
            for error in data["errores"]:
                st.warning(f"{error['campo']}: {error['mensaje']}")
        else:
            st.error(data.get("detail", "No se pudo generar la predicción"))
    except requests.RequestException as e:
        st.error(f"No se pudo conectar a la API: {e}")

st.subheader("Estado de los servicios")

st.caption("Accesos directos:")
columnas_enlaces = st.columns(len(ENLACES_SERVICIOS))
for columna, (nombre, url) in zip(columnas_enlaces, ENLACES_SERVICIOS.items()):
    columna.link_button(nombre, url)


@st.fragment(run_every=5)
def mostrar_salud():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        estado = resp.json() if resp.ok else {}
    except requests.RequestException:
        estado = {}

    columnas = st.columns(len(ETIQUETAS_SALUD))
    for columna, (clave, etiqueta) in zip(columnas, ETIQUETAS_SALUD.items()):
        saludable = estado.get(clave)
        with columna:
            if saludable:
                st.success(etiqueta)
            elif saludable is False:
                st.error(etiqueta)
            else:
                st.warning(f"{etiqueta} (sin datos)")


mostrar_salud()
