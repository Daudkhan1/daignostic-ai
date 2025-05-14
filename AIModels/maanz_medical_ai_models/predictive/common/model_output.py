from enum import Enum
from typing import List


class BiologicalType(str, Enum):
    # Mitotic AI Models
    MITOTIC = "MITOTIC"
    # Radiology AI Models
    DISEASE = "DISEASE"


class Shape(str, Enum):
    # Mitotic AI Models
    SQUARE = "SQUARE"
    RECTANGLE = "RECTANGLE"


class Coordinate:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y

    def x(self):
        return self.__x

    def y(self):
        return self.__y

    def multiply(self, x_factor, y_factor):
        self.__x = self.__x * x_factor
        self.__y = self.__y * y_factor

    def get_tuple(self):
        return (self.__x, self.__y)

    def get_dict(self):
        return {"x": self.__x, "y": self.__y}

    def get_list(self):
        return [self.__x, self.__y]

    def copy(self):
        return Coordinate(self.__x, self.__y)


class BoundingBox:
    def __init__(
        self,
        top_left: Coordinate,
        top_right: Coordinate,
        bottom_right: Coordinate,
        bottom_left: Coordinate,
    ):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right
        self.bottom_left = bottom_left

    def multiply(self, x_factor, y_factor):
        self.top_left.multiply(x_factor, y_factor)
        self.top_right.multiply(x_factor, y_factor)
        self.bottom_right.multiply(x_factor, y_factor)
        self.bottom_left.multiply(x_factor, y_factor)

    def copy(self):
        return BoundingBox(
            self.top_left.copy(),
            self.top_right.copy(),
            self.bottom_right.copy(),
            self.bottom_left.copy(),
        )

    def get_list_of_dicts(self):
        return [
            self.top_left.get_dict(),
            self.top_right.get_dict(),
            self.bottom_right.get_dict(),
            self.bottom_left.get_dict(),
        ]


class Annotation:
    def __init__(
        self,
        biological_type: BiologicalType,
        shape: Shape,
        confidence: float,
        normalized_coordinates: BoundingBox,
        original_coordinates: BoundingBox,
        description: str = "This is AI Description",
    ):
        self.biological_type = biological_type.value
        self.shape = shape.value
        self.confidence = confidence
        self.normalized_coordinates = normalized_coordinates
        self.original_coordinates = original_coordinates
        self.description = description

    def to_dict(self):
        return {
            "biological_type": self.biological_type,
            "shape": self.shape,
            "confidence": self.confidence,
            "normalized_coordinates": self.normalized_coordinates.get_list_of_dicts(),
            "original_coordinates": self.original_coordinates.get_list_of_dicts(),
            "description": self.description,
        }
