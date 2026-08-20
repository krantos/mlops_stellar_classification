import time
from contextlib import asynccontextmanager
from pathlib import Path

import model_service
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from schemas import ModelInput, ModelOutput

FRONTEND = Path(__file__).parent / "frontend" / "index.html"


def _load_state(app: FastAPI) -> None:
    (
        app.state.model,
        app.state.version,
        app.state.class_mapping,
        app.state.features,
        app.state.load_error,
    ) = model_service.load_model()
    app.state.last_check = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_state(app)
    if app.state.load_error:
        # arranca igual: /health reporta not_ready y /predict devuelve 503
        print(f"AVISO: modelo no disponible al iniciar: {app.state.load_error}")
    yield


app = FastAPI(
    title="Stellar Classifier API",
    description="Clasifica objetos del SDSS-17 en GALAXY, QSO o STAR usando XGBoost.",
    version="1.0",
    lifespan=lifespan,
)


def check_model():
    """Si cambió la versión del alias champion, recarga el modelo."""
    if time.monotonic() - app.state.last_check < model_service.CHECK_INTERVAL:
        return
    app.state.last_check = time.monotonic()
    try:
        if model_service.champion_version() != app.state.version:
            _load_state(app)
    except Exception as exc:
        print(f"AVISO: no se pudo chequear la version champion: {exc}")


@app.get("/", include_in_schema=False)
def frontend() -> HTMLResponse:
    return HTMLResponse(FRONTEND.read_text())


@app.get("/health")
def health():
    loaded = app.state.model is not None
    return {
        "status": "ok" if loaded else "not_ready",
        "model_loaded": loaded,
        "model_version": app.state.version,
    }


@app.get("/info")
def info():
    return {
        "model_name": model_service.MODEL_NAME,
        "alias": model_service.ALIAS,
        "model_version": app.state.version,
        "features": app.state.features,
        "class_mapping": app.state.class_mapping,
        "load_error": app.state.load_error,
    }


@app.post("/predict", response_model=ModelOutput)
def predict(features: ModelInput, background_tasks: BackgroundTasks) -> ModelOutput:
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo aún no entrenado, ejecutá el DAG train_model en Airflow",
        )

    # índices de color derivados de las magnitudes
    values = features.model_dump()
    values["u_g"] = values["u"] - values["g"]
    values["g_r"] = values["g"] - values["r"]
    values["r_i"] = values["r"] - values["i"]
    values["i_z"] = values["i"] - values["z"]
    df = pd.DataFrame([values])[app.state.features]
    proba = app.state.model.predict_proba(df)[0]

    mapping = app.state.class_mapping
    probabilities = {
        mapping[str(int(c))]: float(p) for c, p in zip(app.state.model.classes_, proba)
    }
    prediction = max(probabilities, key=probabilities.get)

    background_tasks.add_task(check_model)
    return ModelOutput(
        prediction=prediction,
        probabilities=probabilities,
        model_version=str(app.state.version),
    )


@app.post("/reload")
def reload():
    """Recarga el modelo"""
    _load_state(app)
    return {
        "model_version": app.state.version,
        "model_loaded": app.state.model is not None,
    }
