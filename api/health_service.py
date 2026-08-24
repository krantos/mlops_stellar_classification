import httpx

SERVICIOS = {
    "mlflow": "http://mlflow:5000/",
    "airflow": "http://airflow-apiserver:8080/api/v2/version",
    "minio": "http://s3:9000/minio/health/live",
}


async def verificar_salud() -> dict:
    """Chequea la disponibilidad de los servicios externos del stack"""
    estado = {"fastapi": True}
    async with httpx.AsyncClient(timeout=3) as cliente:
        for nombre, url in SERVICIOS.items():
            try:
                resp = await cliente.get(url)
                estado[nombre] = resp.status_code < 500
            except httpx.HTTPError:
                estado[nombre] = False
    return estado
