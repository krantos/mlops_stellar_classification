import json

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "stellar_model_prod"
ALIAS = "champion"
FEATURE_ORDER = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "u_g",
    "g_r",
    "r_i",
    "i_z",
    "alpha",
    "delta",
    "redshift",
]
CHECK_INTERVAL = 60  # segundos entre chequeos de version nueva


def _resolve_class_mapping(model_data) -> dict:
    tag = model_data.tags.get("class_mapping")
    if tag:
        return json.loads(tag)
    else:
        raise ValueError(f"El modelo {model_data.version} no tiene tag class_mapping")


def champion_version() -> str:
    return MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version


def load_model():
    """
    Trae el modelo champion desde MLflow con su mapeo de clases y features.
    Devuelve (model, version, class_mapping, features, error).
    """
    try:
        client = MlflowClient()
        model_data = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        model = mlflow.sklearn.load_model(model_data.source)
    except Exception as exc:
        return None, None, None, FEATURE_ORDER, str(exc)

    # El orden de columnas sale del modelo entrenado, no de una copia a mano
    features = list(getattr(model, "feature_names_in_", FEATURE_ORDER))
    return model, model_data.version, _resolve_class_mapping(model_data), features, None
