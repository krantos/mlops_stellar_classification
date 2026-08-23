from schemas import StellarObservation


def procesar_observacion(observacion: StellarObservation) -> None:
    """Registra la observación recibida (por ahora, solo se imprime)."""
    print(observacion)
