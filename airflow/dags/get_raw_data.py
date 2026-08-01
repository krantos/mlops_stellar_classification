import datetime

from airflow.decorators import dag, task

markdown_text = """
  ETL Process for Stellar Classification
"""

default_args = {
  'owner': 'Estrellados',
  'depends_on_past': False,
  'schedule_interval': None,
  'retries': 1,
  'retry_delay': datetime.timedelta(minutes=5),
  'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
  dag_id='get_raw_data',
  description='ETL Process for stellar classification',
  doc_md=markdown_text,
  tags=["ETL", "stellar classification"],
  default_args=default_args,
  catchup=False,
  schedule='@monthly'
)
def get_raw_data():

    @task.virtualenv(
      task_id="obtain_original_data",
      requirements=["kagglehub",
                    "awswrangler==3.6.0",
                    "pandas"
                    ],
      system_site_packages=True
    )
    def get_data():
      """
        Download the raw data from Kaggle
      """
      import kagglehub
      import awswrangler as wr
      import pandas as pd
      import glob
      import os

      path = kagglehub.dataset_download("fedesoriano/stellar-classification-dataset-sdss17")
      s3_path = "s3://data/raw/stellar_classification.csv"
      csv_file = glob.glob(os.path.join(path, "*.csv"))
      if not csv_file:
        raise FileNotFoundError(f"No se encontro CSV")
      
      path = csv_file[0]
      df = pd.read_csv(path)
      wr.s3.to_csv(df=df, path=s3_path, index=False)
      print("Database subido a: ", path)

    get_data()

dag = get_raw_data()

