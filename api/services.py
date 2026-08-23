from schemas import StellarObservation


def procesar_observacion(observacion: StellarObservation) -> dict:
    """Lógica de negocio para una observación nueva: por ahora, solo se imprime."""
    print(observacion)
    return {"message": "Datos recibidos", "data": observacion}
