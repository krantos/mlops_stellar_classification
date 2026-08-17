import datetime

from airflow.decorators import dag, task

markdown_text = """
  ETL Process for Stellar Classification
"""

default_args = {
    "owner": "Estrellados",
    "depends_on_past": False,
    "schedule_interval": None,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=15),
}

# Cada corrida escribe sus artefactos bajo esta version, asi una corrida nueva
# nunca pisa los datos de una anterior. Los retries reusan el mismo valor porque
# sale de la fecha del run y no del reloj. Granularidad a segundos para que dos
# triggers manuales en la misma hora no colisionen. En Airflow 3 los triggers
# manuales pueden venir sin logical_date, por eso el fallback a run_after.
RUN_VERSION = (
    "{{ (dag_run.logical_date or dag_run.run_after).strftime('%Y/%m/%d/%H%M%S') }}"
)


@dag(
    dag_id="etl_process",
    description="ETL Process for stellar classification",
    doc_md=markdown_text,
    tags=["ETL", "stellar classification"],
    default_args=default_args,
    catchup=False,
    schedule=None,
)
def etl_process():

    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["kagglehub", "awswrangler==3.6.0", "pandas"],
        system_site_packages=True,
    )
    def get_data(run_version: str):
        """
        Download the raw data from Kaggle
        """
        import glob
        import os

        import awswrangler as wr
        import kagglehub
        import pandas as pd

        path = kagglehub.dataset_download(
            "fedesoriano/stellar-classification-dataset-sdss17"
        )
        s3_path = f"s3://data/{run_version}/raw/stellar_classification.csv"
        csv_file = glob.glob(os.path.join(path, "*.csv"))
        path = csv_file[0]
        df = pd.read_csv(path)
        wr.s3.to_csv(df=df, path=s3_path, index=False)
        print(f"Dataset crudo subido a S3 {s3_path}")

    @task.virtualenv(
        task_id="clean_data", requirements=["pandas", "awswrangler==3.6.0"]
    )
    def clean_data(run_version: str):
        """
        Clean dataset by removing duplicates, nulls and errors, and drop IDs without meaning.
        """
        import awswrangler as wr

        s3_path = f"s3://data/{run_version}/raw/stellar_classification.csv"
        df = wr.s3.read_csv(path=s3_path)

        mask_missings = (df[["u", "g", "z"]] == -9999).any(axis=1)
        df = df[~mask_missings].reset_index(drop=True)
        df = df.dropna().copy()
        drop_cols = [
            "obj_ID",
            "spec_obj_ID",
            "run_ID",
            "rerun_ID",
            "cam_col",
            "field_ID",
            "fiber_ID",
            "plate",
            "MJD",
        ]
        df = df.drop(columns=drop_cols)

        s3_clean = f"s3://data/{run_version}/clean/stellar_classification.csv"
        wr.s3.to_csv(df=df, path=s3_clean, index=False)
        print(f"Dataset limpio guardado en S3 {s3_clean}")

    @task.virtualenv(
        task_id="feature_engineering_and_encoding",
        requirements=["pandas", "awswrangler==3.6.0", "boto3", "scikit-learn"],
    )
    def feature_engineering_and_encoding(run_version: str):
        """
        Build the color indices and encode the target with LabelEncoder.
        The class mapping is stored in S3 so inference can map codes back to labels.
        """
        import json

        import awswrangler as wr
        import boto3
        from sklearn.preprocessing import LabelEncoder

        s3_path = f"s3://data/{run_version}/clean/stellar_classification.csv"
        df = wr.s3.read_csv(path=s3_path)
        df["u_g"] = df["u"] - df["g"]
        df["g_r"] = df["g"] - df["r"]
        df["r_i"] = df["r"] - df["i"]
        df["i_z"] = df["i"] - df["z"]

        le = LabelEncoder()
        df["class"] = le.fit_transform(df["class"])
        class_mapping = {int(code): str(name) for code, name in enumerate(le.classes_)}
        print(f"Mapeo de clases: {class_mapping}")

        s3_feature = f"s3://data/{run_version}/features/stellar_classification.csv"
        wr.s3.to_csv(df=df, path=s3_feature, index=False)
        print(f"Dataset con features guardado en S3 {s3_feature}")

        mapping_key = f"{run_version}/metadata/class_mapping.json"
        boto3.client("s3").put_object(
            Bucket="data",
            Key=mapping_key,
            Body=json.dumps(class_mapping, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"Mapeo de clases guardado en S3 s3://data/{mapping_key}")

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["pandas", "awswrangler==3.6.0", "boto3", "scikit-learn"],
    )
    def split_dataset(run_version: str):
        """
        Single stratified 80/20 split, reused by every model so the comparison stays fair.
        Also writes a pointer to this version on latest.json to track it.
        """
        import json

        import awswrangler as wr
        import boto3
        from sklearn.model_selection import train_test_split

        SEED = 42
        TEST_SIZE = 0.20

        s3_feature = f"s3://data/{run_version}/features/stellar_classification.csv"
        df = wr.s3.read_csv(path=s3_feature)

        mags = ["u", "g", "r", "i", "z"]
        colors = ["u_g", "g_r", "r_i", "i_z"]
        coords = ["alpha", "delta"]
        feature_sets = {
            "with_z": mags + colors + coords + ["redshift"],
            "without_z": mags + colors + coords,
        }

        X = df[feature_sets["with_z"]]
        y = df["class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )
        print(f"Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

        base = f"s3://data/{run_version}/final"
        splits = {
            f"{base}/train/X.csv": X_train,
            f"{base}/train/y.csv": y_train.to_frame(name="class"),
            f"{base}/test/X.csv": X_test,
            f"{base}/test/y.csv": y_test.to_frame(name="class"),
        }
        for path, frame in splits.items():
            wr.s3.to_csv(df=frame, path=path, index=False)
            print(f"Guardado en S3 {path}")

        metadata = {
            "run_version": run_version,
            "seed": SEED,
            "test_size": TEST_SIZE,
            "feature_sets": feature_sets,
            "target": "class",
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "class_proportions_train": {
                str(code): round(float(prop), 4)
                for code, prop in y_train.value_counts(normalize=True)
                .sort_index()
                .items()
            },
            "paths": {
                "features": s3_feature,
                "class_mapping": f"s3://data/{run_version}/metadata/class_mapping.json",
                "X_train": f"{base}/train/X.csv",
                "y_train": f"{base}/train/y.csv",
                "X_test": f"{base}/test/X.csv",
                "y_test": f"{base}/test/y.csv",
            },
        }

        s3 = boto3.client("s3")
        body = json.dumps(metadata, indent=2).encode("utf-8")
        for key in (f"{run_version}/metadata/split.json", "metadata/latest.json"):
            s3.put_object(
                Bucket="data", Key=key, Body=body, ContentType="application/json"
            )
            print(f"Metadata del split guardada en S3 s3://data/{key}")

    (
        get_data(RUN_VERSION)
        >> clean_data(RUN_VERSION)
        >> feature_engineering_and_encoding(RUN_VERSION)
        >> split_dataset(RUN_VERSION)
    )


dag = etl_process()
