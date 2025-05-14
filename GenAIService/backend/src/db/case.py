from uuid import UUID, uuid4
from typing import Dict, List
from enum import Enum

from beanie import Document
from pydantic import Field, BaseModel


class Modality(str, Enum):
    RADIOLOGY = "Radiology"
    PATHOLOGY = "Pathology"


class Shape(str, Enum):
    SQUARE = "Square"
    CIRCLE = "Circle"
    SEGMENTATION = "Segmentation"


class Annotation(BaseModel):
    shape: Shape = Field(..., description="Shape of the annotation")
    Coordinates: List[list] = Field(..., description="Coordinates of the annotation")


class SourceImage(BaseModel):
    source_image_url: str = Field(..., description="URL of the associated image")
    annotations: Dict[UUID, Annotation] = Field(
        ..., description="annotations associated with image"
    )


class Case(Document):
    source_case_id: str = Field(..., description="URL of the associated image")
    source_images: Dict[UUID, SourceImage] = Field(
        ..., description="URL of the associated image"
    )
    modality: Modality = Field(..., description="modality of the case")

    class Settings:
        name = "case"
