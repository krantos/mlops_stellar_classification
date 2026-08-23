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
       task_id="feature_engineering",
       requirements=[
          "pandas",
          "awswrangler==3.6.0"
       ]
    )
    def feature_engineering():
      """
        Feature engineering step
      """
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

  
    @task.virtualenv(
      task_id="split_dataset",
      requirements=[
          "awswrangler==3.6.0",
          "scikit-learn==1.3.2",
      ],
      system_site_packages=True
    )
    def split_dataset():
      """
      Generate a dataset split into a training part and a test part
      """
      import pandas as pd
      import awswrangler as wr
      from sklearn.model_selection import train_test_split
      from sklearn.preprocessing import LabelEncoder

      SEED = 42
      SEARCH_SAMPLE = 20000
      TEST_SIZE = 0.2

      def save_to_csv(df, path):
          wr.s3.to_csv(df=df, path=path, index=False)

      s3_path = "s3://data/clean/stellar_classification.csv"
      dataset = wr.s3.read_csv(path=s3_path)

      le = LabelEncoder()
      X = dataset.drop(columns=["class"])
      y = pd.Series(le.fit_transform(dataset["class"]), name="class")

      X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
      )

      X_search, _, y_search, _ = train_test_split(
        X_train, y_train, train_size=SEARCH_SAMPLE, 
        stratify=y_train, random_state=SEED
      )

      save_to_csv(X_train, "s3://data/final/train/stellar_X_train.csv")
      save_to_csv(X_test, "s3://data/final/test/stellar_X_test.csv")
      save_to_csv(X_search, "s3://data/final/search/stellar_x_search.csv")
      save_to_csv(y_train.to_frame(), "s3://data/final/train/stellar_y_train.csv")
      save_to_csv(y_test.to_frame(), "s3://data/final/test/stellar_y_test.csv")
      save_to_csv(y_search.to_frame(), "s3://data/final/search/stellar_y_search.csv")

    clean_data() >> feature_engineering() >> split_dataset()

dag = etl_process()

