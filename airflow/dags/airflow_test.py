import datetime
from airflow.decorators import dag, task
import mlflow

# PARAMETROS - CAMBIAR AQUI
MLFLOW_TRACKING_URI = "http://mlflow:5000"

markdown_text = """
MLflow Connection Test
"""

default_args = {
    'owner': 'data_eng',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id='mlflow_test_dag',
    description='Test conexion con MLflow',
    doc_md=markdown_text,
    tags=["mlflow", "test"],
    default_args=default_args,
    catchup=False,
    schedule=None
)
def mlflow_test_dag():
    
    @task
    def test_connection():
        """Prueba conexion a MLflow"""
        try:
            client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            client.search_experiments()
            print(f"✓ Conexion exitosa a {MLFLOW_TRACKING_URI}")
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            raise
    
    test_connection()

mlflow_test_dag()