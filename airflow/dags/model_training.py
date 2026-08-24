import datetime

from airflow.decorators import dag, task

markdown_text = """
### Reentrenamiento del modelo de Stellar Classification

Este DAG reentrena el modelo con datos nuevos, compara el modelo anterior con el nuevo, y pone en
producción el que tenga mejor desempeño. Usa el F1-score para evaluar el modelo con los datos de test.

No tiene schedule propio: `etl_process` lo dispara automáticamente en cuanto los datos están listos.
También se puede disparar manualmente para un reentrenamiento puntual.

"""

default_args = {
    'owner': "Estrellados",
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=60)
}

@dag(
    dag_id="train_the_model",
    description="Entrena el modelo",
    doc_md=markdown_text,
    tags=["train", "Stellar Classification"],
    default_args=default_args,
    start_date=datetime.datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
)
def processing_dag():
  @task.virtualenv(
    task_id="buscar_hiperparametros",
    requirements=["scikit-learn==1.3.2",
                  "mlflow==2.10.2",
                  "pandas",
                  "numpy",
                  "xgboost",
                  "optuna",
                  "awswrangler==3.6.0"],
    system_site_packages=True
  )
  def buscar_hiperparametros():
    import mlflow
    import optuna
    import pandas as pd
    import numpy as np

    import awswrangler as wr
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier
    from optuna.samplers import TPESampler

    SEED = 42
    N_TRIALS = 30
    CV_SPLITS = 3
    MLFLOW_TRACKING_URI = "http://mlflow:5000"

    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Stellar Classification - Hyperparameter Tuning")

    # Cargar datos desde S3
    print("Cargando datos de búsqueda desde S3...")
    X_search = wr.s3.read_csv("s3://data/final/search/stellar_x_search.csv")
    y_search = wr.s3.read_csv("s3://data/final/search/stellar_y_search.csv")
    y_search = y_search.iloc[:, 0]

    print("Iniciando búsqueda de hiperparámetros con MLflow...")

    def get_xgb_params_fijos():
      return dict(
          objective="multi:softprob",
          tree_method="hist",
          n_jobs=-1,
          random_state=SEED,
          eval_metric="mlogloss",
          verbosity=0
      )
    
    def sugerir_hiperparametros(trial):
      return dict(
          n_estimators=trial.suggest_int("n_estimators", 100, 400),
          max_depth=trial.suggest_int("max_depth", 3, 9),
          learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
          subsample=trial.suggest_float("subsample", 0.6, 1.0),
          colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
          reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
          reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
      )
    
    def crear_modelo(params):
      params_fijos = get_xgb_params_fijos()
      return XGBClassifier(**params_fijos, **params)
    
    # Crear objetivo de Optuna
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
      
    def objetivo(trial):
      params = sugerir_hiperparametros(trial)
      scores = []
      
      for tr_idx, va_idx in cv.split(X_search, y_search):
          X_tr, X_va = X_search.iloc[tr_idx], X_search.iloc[va_idx]
          y_tr, y_va = y_search.iloc[tr_idx], y_search.iloc[va_idx]
          
          model = crear_modelo(params)
          pesos = compute_sample_weight("balanced", y_tr)
          model.fit(X_tr, y_tr, sample_weight=pesos)
          
          y_pred = model.predict(X_va)
          score = f1_score(y_va, y_pred, average="macro", zero_division=0)
          scores.append(score)
      
      return float(np.mean(scores))
  
    # Ejecutar Optuna dentro de un run de MLflow
    with mlflow.start_run(run_name="Hyperparameter_Search") as run:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        estudio = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=SEED)
        )
        
        estudio.optimize(objetivo, n_trials=N_TRIALS, show_progress_bar=True)
        
        print(f"  Mejor F1-macro (CV): {estudio.best_value:.4f}")
        print(f"  Mejores hiperparámetros: {estudio.best_params}")

        # Loguear parámetros y métricas en MLflow
        mlflow.log_params({
            "n_trials": N_TRIALS,
            "cv_splits": CV_SPLITS,
            "seed": SEED,
            **{f"best_{k}": v for k, v in estudio.best_params.items()}
        })
        mlflow.log_metric("best_f1_macro_cv", estudio.best_value)
        
        # Loguear configuración como artefacto JSON
        hp_dict = {
            "best_params": estudio.best_params,
            "best_value": float(estudio.best_value),
            "n_trials": N_TRIALS,
            "cv_splits": CV_SPLITS
        }
        
        hp_json = pd.DataFrame([hp_dict])
        mlflow.log_table(hp_json, artifact_file="best_hyperparameters.json")
        
        # Guardar en S3 también
        wr.s3.to_json(df=hp_json, path="s3://data/temp/best_hyperparameters.json")
        print("Hiperparámetros guardados en S3 y MLflow")
        
        # Loguear tags para mejor trazabilidad
        mlflow.set_tag("model_type", "XGBClassifier")
        mlflow.set_tag("task", "stellar_classification")
        mlflow.set_tag("stage", "hyperparameter_tuning")
    
    return f"Búsqueda completada. Mejor F1: {estudio.best_value:.4f}"
  
  @task.virtualenv(
        task_id="entrenar_modelo_final",
        requirements=[
            "pandas",
            "numpy",
            "scikit-learn",
            "xgboost",
            "boto3",
            "mlflow==2.10.2",
            "awswrangler==3.6.0"
        ],
        system_site_packages=True
    )
  def entrenar_modelo_final():
    """
      Creamos un nuevo experimento en mlflow. Entrena modelo final y lo registramos en mlflow.
    """
    import json

    import boto3
    import pandas as pd
    import numpy as np
    import awswrangler as wr
    import mlflow
    import mlflow.xgboost
    from datetime import datetime
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    SEED = 42
    MLFLOW_TRACKING_URI = "http://mlflow:5000"
    MODEL_NAME = "stellar_classification_xgb"

    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Stellar Classification - Final Training")

    # Cargar datos
    print("Cargando datos de entrenamiento y prueba...")
    X_train = wr.s3.read_csv("s3://data/final/train/stellar_X_train.csv")
    X_test = wr.s3.read_csv("s3://data/final/test/stellar_X_test.csv")
    y_train = wr.s3.read_csv("s3://data/final/train/stellar_y_train.csv")
    y_test = wr.s3.read_csv("s3://data/final/test/stellar_y_test.csv")
    y_train = y_train.iloc[:, 0]  # Convertir DataFrame a Series
    y_test = y_test.iloc[:, 0]

    # Cargar hiperparámetros desde S3
    print("Cargando hiperparámetros desde S3...")
    hp_json = wr.s3.read_json("s3://data/temp/best_hyperparameters.json")
    hp_dict = hp_json.to_dict(orient="records")[0]
    mejores_params = hp_dict["best_params"]
    f1_cv = hp_dict["best_value"]

    # Cargar el mapeo de clases generado en el ETL, para adjuntarlo al modelo
    print("Cargando mapeo de clases desde S3...")
    mapping_obj = boto3.client("s3").get_object(Bucket="data", Key="metadata/class_mapping.json")
    class_mapping = json.loads(mapping_obj["Body"].read())

    print("Entrenando modelo final con dataset completo...")
    
    # Crear modelo con parámetros optimizados
    params_fijos = dict(
        objective="multi:softprob",
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        eval_metric="mlogloss",
        verbosity=0
    )

    modelo = XGBClassifier(**params_fijos, **mejores_params)
        
    # Entrenar con pesos balanceados
    pesos = compute_sample_weight("balanced", y_train)
    modelo.fit(X_train, y_train, sample_weight=pesos)
    
    # Predicciones
    y_pred = modelo.predict(X_test)

    # Métricas
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  F1-macro (Test): {f1_macro:.4f}")

    metricas = {
        "timestamp": datetime.now().isoformat(),
        "f1_macro_test": float(f1_macro),
        "f1_macro_cv": float(f1_cv),
        "mejores_hiperparametros": mejores_params,
        "datos": {
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "n_features": X_train.shape[1]
        }
    }

    # Registrar en MLflow
    with mlflow.start_run(run_name="Final_Training") as run:
        
        # Loguear parámetros fijos y optimizados
        mlflow.log_params(params_fijos)
        mlflow.log_params({f"opt_{k}": v for k, v in mejores_params.items()})
        
        # Loguear todas las métricas
        mlflow.log_metric("f1_macro_test", f1_macro)
        mlflow.log_metric("f1_macro_cv", f1_cv)
        
        # Loguear información del dataset
        mlflow.log_params({
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "n_features": X_train.shape[1],
            "seed": SEED
        })
        
        # Loguear tags
        mlflow.set_tag("model_type", "XGBClassifier")
        mlflow.set_tag("task", "stellar_classification")
        mlflow.set_tag("stage", "production_candidate")
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        
        # Inferir firma del modelo
        signature = infer_signature(X_test, y_pred)
        
        # Registrar el modelo
        mlflow.xgboost.log_model(
            modelo,
            artifact_path="stellar_model",
            signature=signature,
            metadata={
                "f1_macro_test": f1_macro,
                "f1_macro_cv": f1_cv,
                "training_date": datetime.now().isoformat(),
                "class_mapping": class_mapping
            }
        )
        
        run_id = run.info.run_id
        mlflow.end_run()

    # Registrar el modelo en el Model Registry
    print("Registrando modelo en MLflow Model Registry...")
    client = MlflowClient()
    
    try:
        client.get_registered_model(MODEL_NAME)
    except mlflow.exceptions.RestException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            client.create_registered_model(MODEL_NAME)
            print(f"Modelo '{MODEL_NAME}' creado en el registry")

    # Crear versión del modelo
    model_version = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_id}/stellar_model",
        run_id=run_id
    )
    
    print("Modelo registrado exitosamente")
    print(f"   Model Name: {MODEL_NAME}")
    print(f"   Version: {model_version.version}")
    print(f"   F1-macro (Test): {f1_macro:.4f}")
    print(f"   F1-macro (CV): {f1_cv:.4f}")
    
    # Guardar métricas finales en S3
    metricas_df = pd.DataFrame([metricas])
    wr.s3.to_json(df=metricas_df, path="s3://data/temp/final_metrics.json")
    
    return {
        "run_id": run_id,
        "model_version": model_version.version,
        "f1_macro_test": float(f1_macro),
        "f1_macro_cv": float(f1_cv)
    }

  @task.virtualenv(
        task_id="comparar_y_promover",
        requirements=[
            "mlflow==2.10.2",
            "boto3"
        ],
        system_site_packages=True
    )
  def comparar_y_promover(resultado_entrenamiento):
    """
      Compara el nuevo modelo con el que está en producción.
      Si el nuevo es mejor, lo promueve a Production y archiva el anterior.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_TRACKING_URI = "http://mlflow:5000"
    MODEL_NAME = "stellar_classification_xgb"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    challenger_f1 = resultado_entrenamiento["f1_macro_test"]
    challenger_version = resultado_entrenamiento["model_version"]

    print("Comparando modelos...")
    print(f"   Challenger F1: {challenger_f1:.4f} (Version {challenger_version})")

    # Buscar el modelo en producción
    try:
        latest_versions = client.get_latest_versions(
            name=MODEL_NAME, 
            stages=["Production"]
        )
        champion = latest_versions[0] if latest_versions else None
    except mlflow.exceptions.RestException:
        champion = None

    if champion:
        # Obtener el F1 del champion
        champion_metrics = client.get_run(champion.run_id).data.metrics
        champion_f1 = champion_metrics.get("f1_macro_test")

        if champion_f1 is None:
            print("Champion no tiene métrica F1. Promoviendo challenger...")
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=challenger_version,
                stage="Production"
            )
            return "Champion promovido sin comparación"

        print(f"   Champion F1: {champion_f1:.4f} (Version {champion.version})")

        # Comparar y promover
        if challenger_f1 > champion_f1:
            print("Challenger es mejor, Promoviendo a Production...")
            
            # Archivar el champion
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=champion.version,
                stage="Archived"
            )
            
            # Promover challenger
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=challenger_version,
                stage="Production"
            )
            
            print(f"Modelo versión {challenger_version} está ahora en Production")
            return f"Modelo promovido a Production (mejora de {challenger_f1 - champion_f1:.4f} en F1)"
        else:
            print("El champion sigue siendo mejor. No se promueve.")
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=challenger_version,
                stage="Staging"
            )
            return "Challenger en Staging. Champion seguirá en Production"
    else:
        # Si no hay champion, promover directamente
        print("No hay champion actual. Promoviendo challenger a Production...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=challenger_version,
            stage="Production"
        )
        return f"Primer modelo promovido a Production (Version {challenger_version})"

  # Definir dependencias
  resultado = entrenar_modelo_final()
  buscar_hiperparametros() >> resultado >> comparar_y_promover(resultado)

dag = processing_dag()