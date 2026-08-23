import datetime

from airflow.decorators import dag, task

markdown_text = """
### Re-Train the Model for Stellar classification

This DAG re-trains the model based on new data, tests the previous model, and put in production the new one 
if it performs  better than the old one. It uses the F1 score to evaluate the model with the test data.

"""

default_args = {
    'owner': "Estrellados",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

@dag(
    dag_id="train_the_model",
    description="Train the model",
    doc_md=markdown_text,
    tags=["train", "Stellar Classification"],
    default_args=default_args,
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
                  "catboost",
                  "awswrangler==3.6.0"],
    system_site_packages=True
  )
  def buscar_hiperparametros():
    import datetime
    import mlflow
    import optuna
    import pandas as pd
    import numpy as np

    import awswrangler as wr
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    from mlflow.models import infer_signature
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier
    from optuna.samplers import TPESampler
    from catboost import CatBoostClassifier
    
    SEED = 42
    N_TRIALS = 30
    CV_SPLITS = 3

    # Cargar datos desde S3
    print("Cargando datos de búsqueda desde S3...")
    X_search = wr.s3.read_csv("s3://data/final/search/stellar_x_search.csv")
    y_search = wr.s3.read_csv("s3://data/final/search/stellar_y_search.csv")
    y_search = y_search.iloc[:, 0]

    print("Iniciando búsqueda de hiperparámetros...")

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
  
    # Ejecutar Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    estudio = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED)
    )
    
    estudio.optimize(objetivo, n_trials=N_TRIALS, show_progress_bar=True)
    
    print(f"  Mejor F1-macro (CV): {estudio.best_value:.4f}")
    print(f"  Mejores hiperparámetros: {estudio.best_params}")

    # Guardar hiperparámetros en S3
    hp_dict = {
        "best_params": estudio.best_params,
        "best_value": float(estudio.best_value)
    }
    wr.s3.to_json(df=pd.DataFrame(hp_dict), path="s3://data/temp/best_hyperparameters.json")
    print(f"✓ Hiperparámetros guardados en S3/{s3_hp_path}")
    
    return f"Búsqueda completada. Mejor F1: {estudio.best_value:.4f}"
  
  @task.virtualenv(
        task_id="entrenar_modelo_final",
        requirements=[
            "pandas",
            "numpy",
            "scikit-learn",
            "xgboost",
            "boto3",
            "awswrangler==3.6.0"
        ],
        system_site_packages=True
    )
  def entrenar_modelo_final():
    """Entrena modelo final y guarda en S3."""
    import pandas as pd
    import numpy as np
    import awswrangler as wr
    import json
    import pickle
    from datetime import datetime
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    s3_bucket = "stellar"
    s3_model_path = "models/xgb_model.pkl"
    s3_metrics_path = "metrics/training_metrics.json"

    X_train = wr.se.read_csv("s3://data/final/train/stellar_X_train.csv")
    X_test = wr.se.read_csv("s3://data/final/test/stellar_X_test.csv")
    y_train = wr.se.read_csv("s3://data/final/train/stellar_y_train.csv")
    y_test = wr.se.read_csv("s3://data/final/test/stellar_y_test.csv")
    y_train = y_train.iloc[:, 0]  # Convertir DataFrame a Series
    y_test = y_test.iloc[:, 0]

    # Cargar hiperparámetros desde S3
    print("Cargando hiperparámetros desde S3...")
    hp_bytes = wr.s3.read_json("s3://data/temp/best_hyperparameters.json")
    hp_dict = hp_bytes.to_dict
    mejores_params = hp_dict["best_params"]
    f1_cv = hp_dict["best_value"]

    print("Entrenando modelo final con dataset completo...")
    
    # Crear modelo
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
    y_proba = modelo.predict_proba(X_test)
    
    # Métricas
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  F1-macro (Test): {f1_macro:.4f}")

    # Guardar modelo en S3
    print("Guardando modelo en S3...")
    modelo_bytes = pickle.dumps(modelo)
    wr.s3.put_object(
        bucket=s3_bucket,
        key=s3_model_path,
        body=modelo_bytes
    )
    print(f"✓ Modelo guardado en s3://{s3_bucket}/{s3_model_path}")
  
    # Guardar métricas en S3
    print("Guardando métricas en S3...")
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
    
    metricas_json = json.dumps(metricas)
    wr.s3.put_object(
        bucket=s3_bucket,
        key=s3_metrics_path,
        body=metricas_json
    )
    print(f"✓ Métricas guardadas en s3://{s3_bucket}/{s3_metrics_path}")


  buscar_hiperparametros() >> entrenar_modelo_final()

my_dag = processing_dag()