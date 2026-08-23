from pydantic import BaseModel, Field


class StellarObservation(BaseModel):
    alpha: float = Field(ge=0, lt=360, description="Ascensión recta (grados)")
    delta: float = Field(ge=-90, le=90, description="Declinación (grados)")
    u: float = Field(ge=0, le=40, description="Magnitud en banda u")
    g: float = Field(ge=0, le=40, description="Magnitud en banda g")
    r: float = Field(ge=0, le=40, description="Magnitud en banda r")
    i: float = Field(ge=0, le=40, description="Magnitud en banda i")
    z: float = Field(ge=0, le=40, description="Magnitud en banda z")
    redshift: float = Field(ge=-1, le=10, description="Corrimiento al rojo")
