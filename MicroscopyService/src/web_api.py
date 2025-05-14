from typing import List
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    image: str


class Annotation(BaseModel):
    name: str
    confidence: float
    coordinates: List[dict]


class PredictionResponse(BaseModel):
    mitotic: List[Annotation]
    maybe_mitotic: List[Annotation]


class ConversionResponse(BaseModel):
    image: str
