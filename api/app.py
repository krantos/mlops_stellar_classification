from fastapi import FastAPI
from mlflow.tracking import MlflowClient

app = FastAPI()
client = MlflowClient("http://mlflow:5000")

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