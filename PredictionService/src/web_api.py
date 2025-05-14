from typing import List
from enum import Enum
from pydantic import BaseModel


class AnnotationType(str, Enum):  # Using str makes the values JSON-serializable
    AI = "AI"


class BiologicalType(str, Enum):  # Using str makes the values JSON-serializable
    # Mitotic AI Models
    MITOTIC = "MITOTIC"
    # Radiology AI Models
    DISEASE = "DISEASE"


class Shape(str, Enum):  # Using str makes the values JSON-serializable
    # Mitotic AI Models
    SQUARE = "SQUARE"
    RECTANGLE = "RECTANGLE"


class Modality(str, Enum):
    PATHOLOGY = "PATHOLOGY"
    RADIOLOGY = "RADIOLOGY"


class PredictionRequest(BaseModel):
    image_path: str
    patient_slide_id: str
    modality: Modality


class Annotation(BaseModel):
    name: str
    biological_type: BiologicalType
    shape: Shape
    confidence: float
    description: str
    coordinates: List[dict]


class PredictionResponse(BaseModel):
    patient_slide_id: str
    annotation_type: AnnotationType
    annotations: List[Annotation]
