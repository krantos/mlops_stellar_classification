import asyncio

import mlflow.pyfunc
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlflow_service import MODEL_NAME, obtener_modelo_produccion
from schemas import StellarObservation


class PrediccionError(Exception):
    """Error al predecir con el modelo de producción"""


def _construir_features(observacion: StellarObservation) -> pd.DataFrame:
    """Reconstruye las mismas features derivadas que arma el proceso de ETL"""
    datos = observacion.model_dump()
    datos["u_g"] = datos["u"] - datos["g"]
    datos["g_r"] = datos["g"] - datos["r"]
    datos["r_i"] = datos["r"] - datos["i"]
    datos["i_z"] = datos["i"] - datos["z"]
    return pd.DataFrame([datos])


def _cargar_y_predecir(version: str, observacion: StellarObservation) -> str:
    modelo = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    prediccion = modelo.predict(_construir_features(observacion))
    return str(prediccion[0])


async def predecir_clase(client: MlflowClient, observacion: StellarObservation) -> str:
    """Predice la clase del astro con el modelo en producción, sin bloquear el event loop"""
    modelo_info = await obtener_modelo_produccion(client)
    try:
        return await asyncio.to_thread(_cargar_y_predecir, modelo_info["version"], observacion)
    except MlflowException as e:
        raise PrediccionError(str(e)) from e
