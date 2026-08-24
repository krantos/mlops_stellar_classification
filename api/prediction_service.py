import asyncio

import mlflow.xgboost
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import get_model_info
from mlflow.tracking import MlflowClient

from mlflow_service import MODEL_NAME, obtener_modelo_produccion
from schemas import StellarObservation


class PrediccionError(Exception):
    """Error al predecir con el modelo de producción"""


# Fallback para modelos entrenados antes de que el ETL empezara a guardar
# "class_mapping" en la metadata del modelo (ver etl_process.py / model_training.py).
CLASE_POR_CODIGO_FALLBACK = {"0": "GALAXY", "1": "QSO", "2": "STAR"}


def _construir_features(observacion: StellarObservation) -> pd.DataFrame:
    """Reconstruye las mismas features derivadas que arma el proceso de ETL"""
    datos = observacion.model_dump()
    datos["u_g"] = datos["u"] - datos["g"]
    datos["g_r"] = datos["g"] - datos["r"]
    datos["r_i"] = datos["r"] - datos["i"]
    datos["i_z"] = datos["i"] - datos["z"]
    return pd.DataFrame([datos])


def _cargar_y_predecir(version: str, observacion: StellarObservation) -> dict:
    model_uri = f"models:/{MODEL_NAME}/{version}"
    modelo = mlflow.xgboost.load_model(model_uri)
    probabilidades = modelo.predict_proba(_construir_features(observacion))[0]

    metadata = get_model_info(model_uri).metadata or {}
    mapeo = metadata.get("class_mapping", CLASE_POR_CODIGO_FALLBACK)

    probabilidades_por_clase = {
        mapeo.get(str(int(codigo)), str(int(codigo))): float(prob)
        for codigo, prob in zip(modelo.classes_, probabilidades)
    }
    clase_predicha = max(probabilidades_por_clase, key=probabilidades_por_clase.get)

    return {"clase": clase_predicha, "probabilidades": probabilidades_por_clase}


async def predecir_clase(client: MlflowClient, observacion: StellarObservation) -> dict:
    """Predice la clase del astro y sus probabilidades con el modelo en producción, sin bloquear el event loop"""
    modelo_info = await obtener_modelo_produccion(client)
    try:
        return await asyncio.to_thread(_cargar_y_predecir, modelo_info["version"], observacion)
    except MlflowException as e:
        raise PrediccionError(str(e)) from e
