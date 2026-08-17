from pydantic import BaseModel, Field


class ModelInput(BaseModel):
    u: float = Field(ge=10, le=32, description="Magnitud en el filtro u")
    g: float = Field(ge=10, le=32, description="Magnitud en el filtro g")
    r: float = Field(ge=10, le=32, description="Magnitud en el filtro r")
    i: float = Field(ge=10, le=32, description="Magnitud en el filtro i")
    z: float = Field(ge=10, le=32, description="Magnitud en el filtro z")
    alpha: float = Field(ge=0, le=360, description="Ascensión recta en grados")
    delta: float = Field(ge=-90, le=90, description="Declinación en grados")
    redshift: float = Field(ge=-0.05, le=7.1, description="Corrimiento al rojo")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "u": 23.87,
                    "g": 22.27,
                    "r": 20.39,
                    "i": 19.16,
                    "z": 18.79,
                    "alpha": 135.69,
                    "delta": 32.49,
                    "redshift": 0.644,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    model_version: str

    # pydantic reserva el prefijo model_, esto apaga el warning
    model_config = {"protected_namespaces": ()}
