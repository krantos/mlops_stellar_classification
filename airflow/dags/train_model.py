import datetime

from airflow.decorators import dag, task

markdown_text = """
  Entrena un XGBoost con el ultimo split que dejo el ETL, loguea todo a MLflow
  y registra el modelo como champion o challenger segun el F1 en test.
"""

default_args = {
    "owner": "Estrellados",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=15),
}

EXPERIMENT_NAME = "Stellar Classification"
MODEL_NAME = "stellar_model_prod"
# Mejores hiperparametros que encontro Optuna en el notebook de AM1
DEFAULT_XGB_PARAMS = {
    "n_estimators": 268,
    "max_depth": 6,
    "learning_rate": 0.0534,
    "subsample": 0.730,
    "colsample_bytree": 0.904,
    "reg_alpha": 0.101,
    "reg_lambda": 0.0033,
}


@dag(
    dag_id="train_model",
    description="Train XGBoost model for stellar classification",
    doc_md=markdown_text,
    tags=["train", "stellar classification"],
    default_args=default_args,
    catchup=False,
    schedule=None,
)
def train_model():

    @task.virtualenv(
        task_id="load_latest_metadata",
        requirements=["boto3~=1.34"],
        system_site_packages=True,
    )
    def load_latest_metadata():
        """
        Lee el latest.json que deja el ETL para saber que version de datos usar.
        """
        import json

        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.client("s3")
        try:
            obj = s3.get_object(Bucket="data", Key="metadata/latest.json")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                raise ValueError(
                    "No existe s3://data/metadata/latest.json, "
                    "corré primero el DAG etl_process"
                )
            raise

        metadata = json.loads(obj["Body"].read())
        for key in ("paths", "feature_sets", "run_version"):
            if key not in metadata:
                raise ValueError(f"Falta la clave '{key}' en latest.json")

        run_version = metadata["run_version"]
        print(f"Usando datos de la versión {run_version}")

        # Solo metadata liviana por XCom, los datos quedan en S3
        return {
            "run_version": run_version,
            "paths": metadata["paths"],
            "features": metadata["feature_sets"]["with_z"],
            "target": metadata.get("target", "class"),
        }

    @task.virtualenv(
        task_id="train_and_log",
        requirements=[
            "mlflow==2.22.2",
            "xgboost==2.1.4",
            "scikit-learn~=1.5",
            "awswrangler==3.6.0",
            "matplotlib",
            "boto3~=1.34",
        ],
        system_site_packages=True,
    )
    def train_and_log(metadata: dict, experiment_name: str, default_params: dict):
        """
        Entrena el XGBoost con los splits de S3 y loguea params, metricas
        y el modelo a MLflow.
        """
        import json
        import os

        import awswrangler as wr
        import boto3
        import matplotlib

        matplotlib.use("Agg")
        import mlflow
        from mlflow.models import infer_signature
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            accuracy_score,
            f1_score,
        )
        from sklearn.utils.class_weight import compute_sample_weight
        from xgboost import XGBClassifier

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment(experiment_name)

        paths = metadata["paths"]
        features = metadata["features"]
        target = metadata.get("target", "class")
        run_version = metadata["run_version"]

        X_train = wr.s3.read_csv(path=paths["X_train"])[features]
        y_train = wr.s3.read_csv(path=paths["y_train"])[target]
        X_test = wr.s3.read_csv(path=paths["X_test"])[features]
        y_test = wr.s3.read_csv(path=paths["y_test"])[target]

        mapping_uri = paths["class_mapping"]
        bucket, key = mapping_uri.replace("s3://", "").split("/", 1)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        class_mapping = json.loads(obj["Body"].read())

        # Los defaults son los que salieron de Optuna en Aprendizaje de Máquinas
        try:
            from airflow.models import Variable

            params = json.loads(Variable.get("stellar_xgb_params"))
            print("Hiperparametros leidos de la Variable stellar_xgb_params")
        except Exception as e:
            params = default_params
            print(
                f"Variable stellar_xgb_params no disponible ({e}), uso los defaults del notebook"
            )

        sample_weight = compute_sample_weight("balanced", y_train)

        model = XGBClassifier(
            objective="multi:softprob",
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
            **params,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = model.predict(X_test)

        f1_macro = f1_score(y_test, y_pred, average="macro")
        accuracy = accuracy_score(y_test, y_pred)
        codes = sorted(int(c) for c in class_mapping)
        f1_per_class = f1_score(y_test, y_pred, labels=codes, average=None)
        metrics = {
            "f1_macro_test": float(f1_macro),
            "accuracy_test": float(accuracy),
        }
        for code, f1_cls in zip(codes, f1_per_class):
            metrics[f"f1_{class_mapping[str(code)]}"] = float(f1_cls)

        print(f"F1 macro en test: {f1_macro:.4f}")
        print(f"Accuracy en test: {accuracy:.4f}")

        run_name = f"train_{run_version.replace('/', '-')}"
        with mlflow.start_run(
            run_name=run_name,
            tags={"dataset_version": run_version, "model": "xgboost"},
        ) as run:
            mlflow.log_params(
                {
                    **params,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "features": json.dumps(features),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_dict(metadata, "dataset_metadata.json")

            disp = ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_pred,
                display_labels=[class_mapping[str(c)] for c in codes],
                normalize="true",
            )
            mlflow.log_figure(disp.figure_, "confusion_matrix.png")

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=infer_signature(X_train, y_pred),
                serialization_format="cloudpickle",
                input_example=X_test.head(3),
            )
            run_id = run.info.run_id

        print(f"Run {run_id} logueado en MLflow")
        return {
            "run_id": run_id,
            "model_uri": f"runs:/{run_id}/model",
            "f1_macro_test": float(f1_macro),
            "dataset_version": run_version,
            "class_mapping": class_mapping,
        }

    @task.virtualenv(
        task_id="register_and_promote",
        requirements=["mlflow==2.22.2"],
        system_site_packages=True,
    )
    def register_and_promote(result: dict, model_name: str):
        """
        Registra la nueva version y la promueve a champion si le gana
        al champion actual, si no queda como challenger.
        """
        import json
        import os

        import mlflow
        from mlflow.exceptions import MlflowException

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = mlflow.MlflowClient()

        try:
            client.create_registered_model(
                model_name,
                description="Clasificador de objetos del SDSS17 (GALAXY/QSO/STAR)",
            )
        except MlflowException:
            pass  # ya existe

        new_f1 = result["f1_macro_test"]
        version = client.create_model_version(
            name=model_name,
            source=result["model_uri"],
            run_id=result["run_id"],
            tags={
                "f1_macro_test": str(new_f1),
                "dataset_version": result["dataset_version"],
                "class_mapping": json.dumps(result["class_mapping"]),
            },
        )

        try:
            champ = client.get_model_version_by_alias(model_name, "champion")
        except MlflowException as e:
            # Solo si no existe el alias, tenemos el primer modelo, cualquier
            # otro error tiene que tirar el task en vez de promover a ciegas.
            alias_missing = (
                getattr(e, "error_code", "")
                in (
                    "RESOURCE_DOES_NOT_EXIST",
                    "INVALID_PARAMETER_VALUE",
                )
                and "alias" in str(e).lower()
            )
            if not alias_missing:
                raise
            client.set_registered_model_alias(model_name, "champion", version.version)
            print("Primer modelo registrado, promovido a champion")
            return

        champ_f1 = float(champ.tags.get("f1_macro_test", -1))
        if new_f1 > champ_f1:
            client.set_registered_model_alias(model_name, "champion", version.version)
            client.set_registered_model_alias(model_name, "challenger", champ.version)
            print(
                f"Nuevo champion: versión {version.version} "
                f"(F1 {new_f1:.4f} > {champ_f1:.4f})"
            )
        else:
            client.set_registered_model_alias(model_name, "challenger", version.version)
            print(
                f"El champion sigue siendo la versión {champ.version}, "
                "la nueva queda como challenger"
            )

    register_and_promote(
        train_and_log(load_latest_metadata(), EXPERIMENT_NAME, DEFAULT_XGB_PARAMS),
        MODEL_NAME,
    )


dag = train_model()
