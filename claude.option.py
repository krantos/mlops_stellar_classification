
Airflow dag · PY
import datetime
from airflow.decorators import dag, task
import pandas as pd
from datetime import timedelta
 
# ============================================================================
# CONSTANTES DE PATHS S3
# ============================================================================
 
S3_BUCKET = "your-bucket-name"
S3_INPUT_DATASET = "data/cleaned_dataset.csv"
 
# Paths intermedios (se guardan/cargan entre tareas)
S3_PREPARED_DATA_TRAIN = "temp/prepared_X_train_full.parquet"
S3_PREPARED_DATA_TEST = "temp/prepared_X_test_full.parquet"
S3_PREPARED_DATA_SEARCH = "temp/prepared_X_search.parquet"
S3_PREPARED_LABELS_TRAIN = "temp/prepared_y_train.parquet"
S3_PREPARED_LABELS_TEST = "temp/prepared_y_test.parquet"
S3_PREPARED_LABELS_SEARCH = "temp/prepared_y_search.parquet"
 
# Paths de salida final
S3_MODEL_PATH = "models/xgb_model.pkl"
S3_METRICS_PATH = "metrics/training_metrics.json"
 
# ============================================================================
# CONFIGURACIÓN DE LA DAG
# ============================================================================
 
markdown_text = """
  Pipeline de Entrenamiento XGBoost con Búsqueda de Hiperparámetros
  
  - Carga dataset limpio desde S3
  - Prepara splits train/test y submuestra para búsqueda
  - Búsqueda de hiperparámetros con Optuna (3-fold CV)
  - Entrenamiento final con dataset completo
  - Guarda modelo y métricas en S3
"""
 
default_args = {
    'owner': 'data-science',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'dagrun_timeout': timedelta(minutes=60)
}
 
 
@dag(
    dag_id='xgb_model_training',
    description='Entrenamiento de modelo XGBoost con búsqueda de hiperparámetros',
    doc_md=markdown_text,
    tags=["ML", "training", "XGBoost"],
    default_args=default_args,
    catchup=False,
    schedule='@weekly'
)
def xgb_training_pipeline():
    """Pipeline de entrenamiento de modelo XGBoost."""
 
    @task.virtualenv(
        task_id="cargar_datos",
        requirements=[
            "pandas",
            "boto3",
            "awswrangler==3.6.0"
        ],
        system_site_packages=True
    )
    def cargar_datos_s3():
        """Carga dataset limpio desde S3."""
        import pandas as pd
        import awswrangler as wr
        
        s3_bucket = "your-bucket-name"
        s3_dataset_path = "data/cleaned_dataset.csv"
        s3_path = f"s3://{s3_bucket}/{s3_dataset_path}"
        
        print(f"Cargando dataset desde {s3_path}")
        df = wr.s3.read_csv(path=s3_path)
        print(f"Dataset cargado: {df.shape}")
        
        return "Dataset cargado exitosamente"
 
    @task.virtualenv(
        task_id="preparar_datos",
        requirements=[
            "pandas",
            "scikit-learn",
            "numpy",
            "awswrangler==3.6.0"
        ],
        system_site_packages=True
    )
    def preparar_datos():
        """Prepara train/test/search splits y guarda en S3."""
        import pandas as pd
        import numpy as np
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        
        s3_bucket = "your-bucket-name"
        s3_input_dataset = "data/cleaned_dataset.csv"
        s3_train_full = "temp/prepared_X_train_full.parquet"
        s3_test_full = "temp/prepared_X_test_full.parquet"
        s3_search = "temp/prepared_X_search.parquet"
        s3_y_train = "temp/prepared_y_train.parquet"
        s3_y_test = "temp/prepared_y_test.parquet"
        s3_y_search = "temp/prepared_y_search.parquet"
        
        SEED = 42
        SEARCH_SAMPLE = 20000
        TEST_SIZE = 0.2
        
        # Cargar dataset desde S3
        s3_path = f"s3://{s3_bucket}/{s3_input_dataset}"
        print(f"Cargando dataset desde {s3_path}")
        df = wr.s3.read_csv(path=s3_path)
        print("Preparando datos...")
        
        # Separar features y target
        X = df.drop(columns=["target"])
        y = df["target"]
        
        # Train/Test split
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )
        
        # Submuestra estratificada para acelerar búsqueda
        X_search, _, y_search, _ = train_test_split(
            X_train_full, y_train, train_size=SEARCH_SAMPLE, 
            stratify=y_train, random_state=SEED
        )
        
        print(f"  Train full: {X_train_full.shape}")
        print(f"  Test: {X_test_full.shape}")
        print(f"  Search (CV): {X_search.shape}")
        
        # Guardar en S3 como parquet
        print("Guardando datos preparados en S3...")
        wr.s3.to_parquet(X_train_full, f"s3://{s3_bucket}/{s3_train_full}")
        wr.s3.to_parquet(X_test_full, f"s3://{s3_bucket}/{s3_test_full}")
        wr.s3.to_parquet(X_search, f"s3://{s3_bucket}/{s3_search}")
        wr.s3.to_parquet(y_train, f"s3://{s3_bucket}/{s3_y_train}")
        wr.s3.to_parquet(y_test, f"s3://{s3_bucket}/{s3_y_test}")
        wr.s3.to_parquet(y_search, f"s3://{s3_bucket}/{s3_y_search}")
        
        print("✓ Datos preparados y guardados en S3")
        return "Datos preparados exitosamente"
 
    @task.virtualenv(
        task_id="buscar_hiperparametros",
        requirements=[
            "pandas",
            "numpy",
            "scikit-learn",
            "xgboost",
            "optuna",
            "awswrangler==3.6.0"
        ],
        system_site_packages=True
    )
    def buscar_hiperparametros():
        """Busca mejores hiperparámetros con Optuna."""
        import pandas as pd
        import numpy as np
        import awswrangler as wr
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score
        from sklearn.utils.class_weight import compute_sample_weight
        from xgboost import XGBClassifier
        import optuna
        from optuna.samplers import TPESampler
        
        s3_bucket = "your-bucket-name"
        s3_search = "temp/prepared_X_search.parquet"
        s3_y_search = "temp/prepared_y_search.parquet"
        
        SEED = 42
        N_TRIALS = 30
        CV_SPLITS = 3
        
        # Cargar datos desde S3
        print("Cargando datos de búsqueda desde S3...")
        X_search = wr.s3.read_parquet(f"s3://{s3_bucket}/{s3_search}")
        y_search = wr.s3.read_parquet(f"s3://{s3_bucket}/{s3_y_search}")
        y_search = y_search.iloc[:, 0]  # Convertir DataFrame a Series
        
        print("Iniciando búsqueda de hiperparámetros...")
        
        # Función para crear modelo
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
        import json
        s3_hp_path = "temp/best_hyperparameters.json"
        hp_dict = {
            "best_params": estudio.best_params,
            "best_value": float(estudio.best_value)
        }
        wr.s3.put_object(
            bucket=s3_bucket,
            key=s3_hp_path,
            body=json.dumps(hp_dict)
        )
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
        
        SEED = 42
        s3_bucket = "your-bucket-name"
        s3_train_full = "temp/prepared_X_train_full.parquet"
        s3_test_full = "temp/prepared_X_test_full.parquet"
        s3_y_train = "temp/prepared_y_train.parquet"
        s3_y_test = "temp/prepared_y_test.parquet"
        s3_hp_path = "temp/best_hyperparameters.json"
        s3_model_path = "models/xgb_model.pkl"
        s3_metrics_path = "metrics/training_metrics.json"
        
        # Cargar datos desde S3
        print("Cargando datos de entrenamiento desde S3...")
        X_train_full = wr.s3.read_parquet(f"s3://{s3_bucket}/{s3_train_full}")
        X_test_full = wr.s3.read_parquet(f"s3://{s3_bucket}/{s3_test_full}")
        y_train = wr.s3.read_parquet(f"s3://{s3_bucket}/{s3_y_train}")
        y_test = wr.s3.read_parquet(f"s3://{s3_bucket}/{s3_y_test}")
        y_train = y_train.iloc[:, 0]  # Convertir DataFrame a Series
        y_test = y_test.iloc[:, 0]
        
        # Cargar hiperparámetros desde S3
        print("Cargando hiperparámetros desde S3...")
        hp_bytes = wr.s3.get_object(bucket=s3_bucket, key=s3_hp_path)
        hp_dict = json.loads(hp_bytes)
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
        modelo.fit(X_train_full, y_train, sample_weight=pesos)
        
        # Predicciones
        y_pred = modelo.predict(X_test_full)
        y_proba = modelo.predict_proba(X_test_full)
        
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
                "train_size": X_train_full.shape[0],
                "test_size": X_test_full.shape[0],
                "n_features": X_train_full.shape[1]
            }
        }
        
        metricas_json = json.dumps(metricas)
        wr.s3.put_object(
            bucket=s3_bucket,
            key=s3_metrics_path,
            body=metricas_json
        )
        print(f"✓ Métricas guardadas en s3://{s3_bucket}/{s3_metrics_path}")
        
        return metricas
 
    @task
    def resumen(metricas):
        """Imprime resumen final."""
        print("\n" + "=" * 70)
        print("RESUMEN DEL ENTRENAMIENTO")
        print("=" * 70)
        print(f"F1-macro (CV):   {metricas['f1_macro_cv']:.4f}")
        print(f"F1-macro (Test): {metricas['f1_macro_test']:.4f}")
        print(f"Muestras train:  {metricas['datos']['train_size']}")
        print(f"Muestras test:   {metricas['datos']['test_size']}")
        print(f"Features:        {metricas['datos']['n_features']}")
        print("=" * 70)
        return "Pipeline completado exitosamente"
 
    # ========================================================================
    # FLUJO DE TAREAS
    # ========================================================================
 
    cargar = cargar_datos_s3()
    preparar = preparar_datos()
    buscar = buscar_hiperparametros()
    entrenar = entrenar_modelo_final()
    resumen_final = resumen(entrenar)
    
    # Definir dependencias
    cargar >> preparar >> buscar >> entrenar >> resumen_final
 
 
dag = xgb_training_pipeline()
 
