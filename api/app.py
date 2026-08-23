from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient

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
    return {"message": "Welcome to the Model Service"}

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

@app.get("/hello")
def hello():
    return {"message": "Holo"}

@app.get("/test")
def test():
    return {"message": "test"}

@app.post("/observaciones")
def recibir_observacion(observacion: StellarObservation):
    """Recibe una observación nueva del frontend"""
    return procesar_observacion(observacion)