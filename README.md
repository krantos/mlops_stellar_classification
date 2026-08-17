# Clasificación estelar, MLOps CEIA - FIUBA

TP final de Operaciones de Aprendizaje Automático (CEIA, FIUBA), nivel contenedores.

Participantes:

- Marcos Riveros (a2537)
- Franco Morero (a2533)
- Tadeo Riveros (a2536)

La idea del TP es agarrar el modelo de clasificación estelar que hicimos en Aprendizaje de Máquina I y armarle todo el ciclo MLOps alrededor: ETL versionado, entrenamiento con tracking y registro de modelos, y una API que sirve el mejor modelo disponible.

El dataset es el [SDSS17 de Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17), 100k observaciones del Sloan Digital Sky Survey. El objetivo es clasificar cada observación como GALAXY, QSO o STAR. El modelo es un XGBoost con hiperparámetros que ya habíamos encontrado con Optuna en el notebook del [TP Final - Aprendizaje de Máquinas](https://github.com/francomor/tp-final-aprmaq).


![Diagrama de servicios](final_assign.png)

## Cómo funciona el flujo

1. **DAG `etl_process`** (Airflow, corre mensual o lo disparás a mano). Baja el dataset de Kaggle, lo limpia (saca la fila con valores -9999 y las columnas de IDs que no aportan nada), genera los índices de color (`u_g`, `g_r`, `r_i`, `i_z`), codifica el target con `LabelEncoder` y hace un split 80/20 estratificado con seed 42. Todo queda versionado en `s3://data/{YYYY/MM/DD/HH}/...` en MinIO, y un puntero en `s3://data/metadata/latest.json` apunta siempre a la última versión.
2. **DAG `train_model`** (manual, hay que correrlo después del ETL). Lee `latest.json`, entrena un `XGBClassifier` con sample weights balanceados y loguea todo a MLflow: parámetros, métricas (F1 macro, accuracy, F1 por clase), matriz de confusión y la metadata del dataset usado. Después registra el modelo en `stellar_model_prod` con lógica champion/challenger: si el F1 macro en test supera al del champion actual, lo destrona. Si no, queda registrado como challenger. El primer modelo que se entrena pasa a champion directo.
3. **MLflow** hace el tracking, con backend en PostgreSQL (`mlflow_db`) y los artefactos en `s3://mlflow` (MinIO).
4. **API con FastAPI** (puerto 8800). Al arrancar carga el champion del registry y expone `POST /predict`. Recibe las mediciones crudas (`u`, `g`, `r`, `i`, `z`, `alpha`, `delta`, `redshift`) y los índices de color se calculan en el server. También tiene `GET /health`, `GET /info` y `POST /reload`. Si aparece un champion nuevo, la API se recarga sola en background. Si todavía no entrenaste nada, devuelve 503.
5. **Frontend** en http://localhost:8800: un formulario con las 8 features, botones con ejemplos reales de cada clase y barras de probabilidad para ver qué tan seguro está el modelo.

## Servicios y puertos

| Servicio | URL / puerto | Credenciales |
|---|---|---|
| Apache Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5001 | - |
| MinIO (API S3) | http://localhost:9000 | minio / minio123 |
| MinIO (consola) | http://localhost:9001 | minio / minio123 |
| API FastAPI | http://localhost:8800 (docs en `/docs`) | - |
| PostgreSQL | localhost:5432 | - |
| Valkey | interno (lo usa Airflow, no está expuesto) | - |

Cuando se levantan los contenedores se crean solos los buckets `s3://data` y `s3://mlflow`, y las bases `mlflow_db` (MLflow) y `airflow` (Airflow).

## Cómo levantar todo

Necesitás Docker y Docker Compose instalados.

Si estás en Linux, antes de levantar nada reemplazá `AIRFLOW_UID` en el archivo `.env` por el UID de tu usuario (lo sacás con `id -u`). Si no, Airflow deja sus carpetas internas como root y después no podés tocar los DAGs ni los plugins.

```bash
docker compose --profile all up -d --build
```

Verificá con `docker ps -a` que todos los servicios estén healthy antes de usar nada. Si estás en un servidor externo, reemplazá `localhost` por su IP en todas las URLs (y revisá firewalls).

Para bajar los servicios cuando no los usás:

```bash
docker compose --profile all down
```

Y si querés borrar todo, imágenes y volúmenes incluidos:

```bash
docker compose --profile all down -v --rmi all
```

Ojo que con esto perdés todo lo que haya en los buckets y en las bases de datos.

## Cómo correr el pipeline

El orden importa: primero `etl_process`, después `train_model`. Si corrés el entrenamiento sin haber corrido el ETL, el DAG falla y el mensaje de error te dice qué pasó.

Podés disparar los DAGs desde la UI de Airflow (http://localhost:8080).

## Usar la API

Con el modelo cargado, le pegás a `/predict` con las mediciones crudas:

```bash
curl -X POST http://localhost:8800/predict \
  -H "Content-Type: application/json" \
  -d '{
    "u": 23.87,
    "g": 22.27,
    "r": 20.39,
    "i": 19.16,
    "z": 18.79,
    "alpha": 135.69,
    "delta": 32.49,
    "redshift": 0.644
  }'
```

Respuesta esperada:

```json
{"prediction": "GALAXY", "probabilities": {...}, "model_version": "1"}
```

Además tenés:

- `GET /health` para chequear si la API está viva y con modelo cargado
- `GET /info` para ver qué versión del modelo está sirviendo
- `POST /reload` para forzar la recarga del ultimo modelo
- `/docs` con la documentación interactiva de FastAPI

O directamente usá el frontend en http://localhost:8800, que tiene ejemplos precargados de cada clase.

## Decisiones que tomamos

Usamos los hiperparámetros fijos del notebook de AM1 en vez de re-correr Optuna dentro del DAG. La búsqueda ya se hizo una vez y no tenía sentido repetirla en cada entrenamiento, así el DAG queda rápido. Igual se pueden pisar con la Variable de Airflow `stellar_xgb_params` si se quiere probar otra cosa.

Los datos quedan versionados por timestamp en S3, así que nada se pisa entre corridas y siempre podés ir a ver exactamente con qué datos se entrenó cada modelo.

Los índices de color se calculan en la API y no en el cliente. La idea es que quien consume la API mande solo las mediciones crudas del telescopio y no tenga que saber cómo se derivan las features.

## Detalles de Airflow

En el `docker-compose.yaml`, dentro de `x-airflow-common`, están las variables de entorno de Airflow por si necesitás ajustar algo. Se pueden agregar [otras](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html).

Airflow usa un ejecutor [Celery](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/executor/celery.html), o sea que las tareas corren en otro contenedor (el worker).

Si necesitás debuggear con la CLI de Airflow:

```bash
docker compose --profile all --profile debug up
```

y con el contenedor andando, por ejemplo para ver la configuración:

```bash
docker compose run airflow-cli config list
```

Las variables para los DAGs van en `secrets/variables.yaml` y las conexiones en `secrets/connections.yaml`. También se pueden crear desde la UI, pero esas no persisten si borrás los volúmenes. Las de `secrets/` no aparecen en la UI, pero existen igual.

## Conectarse a los buckets desde afuera

Como no usamos Amazon S3 sino MinIO, si querés usar `boto3`, `awswrangler` o `awscli` desde tu máquina tenés que setear estas variables de entorno:

```bash
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123
AWS_ENDPOINT_URL_S3=http://localhost:9000
```

MLflow tiene la suya propia:

```bash
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```


## Valkey

Valkey lo usa Airflow internamente como broker de Celery. El puerto no está expuesto para uso externo, pero se puede habilitar tocando el `docker-compose.yaml` si hace falta.
