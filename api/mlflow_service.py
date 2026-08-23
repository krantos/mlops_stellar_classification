import asyncio

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME = "stellar_classification_xgb"


class ModeloNoEncontradoError(Exception):
    """No hay un modelo en producción para el nombre configurado"""


class ConexionMlflowError(Exception):
    """Error al comunicarse con el servidor de MLflow"""


def _buscar_ultima_version_produccion(client: MlflowClient):
    versiones = client.get_latest_versions(name=MODEL_NAME, stages=["Production"])
    if not versiones:
        raise ModeloNoEncontradoError(f"No hay modelos en producción para '{MODEL_NAME}'")
    return versiones[0]


async def obtener_modelo_produccion(client: MlflowClient) -> dict:
    """Obtiene el último modelo en producción desde MLflow sin bloquear el event loop"""
    try:
        version = await asyncio.to_thread(_buscar_ultima_version_produccion, client)
    except ModeloNoEncontradoError:
        raise
    except MlflowException as e:
        raise ConexionMlflowError(str(e)) from e

    return {
        "name": version.name,
        "version": version.version,
        "stage": version.current_stage,
        "status": version.status,
        "run_id": version.run_id,
        "source": version.source,
    }
