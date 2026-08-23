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
  dag_id='etl_process',
  description='ETL Process for stellar classification',
  doc_md=markdown_text,
  tags=["ETL", "stellar classification"],
  default_args=default_args,
  catchup=False,
  schedule='@monthly'
)
def etl_process():

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
      s3_path = f"s3://data/raw/stellar_classification.csv"
      csv_file = glob.glob(os.path.join(path, "*.csv"))        
      path = csv_file[0]
      df = pd.read_csv(path)
      wr.s3.to_csv(df=df, path=s3_path, index=False)
      print("Database subido a: ", s3_path)


    @task.virtualenv(
      task_id="clean_data",
      requirements=[
         "pandas",
         "awswrangler==3.6.0"
      ]
    )    
    def clean_data():
      """
       Clean dataset by removing duplicates, nulls and errors, and remove id's without meainig.
      """
      import pandas as pd
      import awswrangler as wr

      s3_path = "s3://data/raw/stellar_classification.csv"
      df = wr.s3.read_csv(path=s3_path)
      
      mask_missings = (df[["u", "g", "z"]] == -9999).any(axis=1)
      df = df[~mask_missings].reset_index(drop=True)
      df = df.dropna().copy()
      drop_cols = ["obj_ID", "spec_obj_ID", "run_ID", "rerun_ID", "cam_col",
            "field_ID", "fiber_ID", "plate", "MJD"]
      df = df.drop(columns=drop_cols)
      
      s3_clean = "s3://data/clean/stellar_classification.csv"
      wr.s3.to_csv(df=df, path=s3_clean, index=False)
      print(f"Dataset limpio guardado en S3 {s3_clean}")


    @task.virtualenv(
       task_id="feature_engineering_and_encoding",
       requirements=[
          "pandas",
          "awswrangler==3.6.0"
       ]
    )
    def feature_engineering_and_encoding():
      """
        Feature engineering step
      """
      import pandas as pd
      import awswrangler as wr

      s3_path = "s3://data/clean/stellar_classification.csv"
      df = wr.s3.read_csv(path=s3_path)
      df["u_g"] = df["u"] - df["g"]
      df["g_r"] = df["g"] - df["r"]
      df["r_i"] = df["r"] - df["i"]
      df["i_z"] = df["i"] - df["z"]
      s3_feature = "s3://data/features/stellar_classification.csv"
      wr.s3.to_csv(df=df, path=s3_feature, index=False)
      print(f"Dataset limpio guardado en S3 {s3_feature}")

    get_data() >> clean_data() >> feature_engineering_and_encoding()

dag = etl_process()

