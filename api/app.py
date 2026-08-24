from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient

from health_service import verificar_salud
from mlflow_service import ConexionMlflowError, ModeloNoEncontradoError, obtener_modelo_produccion
from prediction_service import PrediccionError, predecir_clase
from schemas import StellarObservation
from services import procesar_observacion

app = FastAPI()
client = MlflowClient("http://mlflow:5000")


@app.exception_handler(RequestValidationError)
def manejar_error_validacion(request: Request, exc: RequestValidationError):
    """Devuelve los errores de validación en un formato simple para el frontend"""
    errores = [
        {"campo": ".".join(str(p) for p in err["loc"] if p != "body"), "mensaje": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"message": "Datos inválidos", "errores": errores})


@app.get("/")
def read_root():
    return {"message": "Stellar Classification API"}

@app.get("/health")
async def health():
    """Devuelve el estado de los servicios del stack, para mostrar en el frontend"""
    return await verificar_salud()

@app.get("/modelos")
def listar_modelos():
    """Lista todos los modelos registrados en MLflow"""
    modelos = client.search_model_versions("")
    return [
        {
            "version": v.version,
            "name": v.name,
            "stage": v.current_stage,
            "status": v.status
        }
        for v in modelos
    ]

@app.get("/modelos/produccion")
async def modelo_en_produccion():
    """Devuelve el último modelo de Stellar Classification en producción"""
    try:
        return await obtener_modelo_produccion(client)
    except ModeloNoEncontradoError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ConexionMlflowError as e:
        raise HTTPException(status_code=502, detail=f"No se pudo conectar con MLflow: {e}")

@app.post("/observaciones")
async def recibir_observacion(observacion: StellarObservation):
    """Recibe una observación nueva y predice su clase con el modelo en producción"""
    procesar_observacion(observacion)
    try:
        resultado = await predecir_clase(client, observacion)
    except ModeloNoEncontradoError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ConexionMlflowError as e:
        raise HTTPException(status_code=502, detail=f"No se pudo conectar con MLflow: {e}")
    except PrediccionError as e:
        raise HTTPException(status_code=502, detail=f"No se pudo generar la predicción: {e}")

    return {
        "message": "Predicción generada",
        "clase": resultado["clase"],
        "probabilidades": resultado["probabilidades"],
    }