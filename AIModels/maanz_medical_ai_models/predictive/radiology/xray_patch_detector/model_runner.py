import os

import numpy as np
from ultralytics import YOLO
from decouple import config

from maanz_medical_ai_models.predictive.common.model_input import load_source_image
from maanz_medical_ai_models.predictive.common.model_output import (
    Annotation,
    BiologicalType,
    Shape,
    Coordinate,
    BoundingBox,
)


# Load YOLOv11 model
current_dir = os.path.dirname(os.path.abspath(__file__))
weights_folder = os.path.join(current_dir, "data", "weights", "radiology_yolo.pt")
DATA_MODEL_PATH = config(
    "XRAY_PATCH_DETECTOR_RADIOLOGY_MODEL_PATH", default=weights_folder, cast=str
)
model = YOLO(DATA_MODEL_PATH)


def run_model(source_image: str):
    CONFIDENCE_THRESHOLD = 0.25

    # download the image
    image = load_source_image(source_image)

    # Run inference using predict()
    results = model.predict(source=image, conf=CONFIDENCE_THRESHOLD)
    result = results[0]

    # Get class names from the model
    class_names = model.names if hasattr(model, "names") else None

    image_width, image_height = image.size

    confidence_scores = []

    annotations = []
    # Draw only the boxes we want to keep (skipping class 0 or 'No finding')
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = class_names[class_id] if class_names else str(class_id)

        confidence_score = float(box.conf)

        # Skip class ID 0 or class name 'No finding'
        if class_id == 0 or class_name == "No finding":
            continue

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_length = x2 - x1

        top_left = Coordinate(int(x1), int(y1))
        top_right = Coordinate(int(x1 + box_length), int(y1))
        bottom_right = Coordinate(int(x2), int(y2))
        bottom_left = Coordinate(int(x2 - box_length), int(y2))
        original_coordinates = BoundingBox(
            top_left, top_right, bottom_right, bottom_left
        )

        normalized_coordinates = original_coordinates.copy()
        normalized_coordinates.multiply(100 / image_width, 100 / image_height)

        annotation = Annotation(
            biological_type=BiologicalType.DISEASE,
            confidence=confidence_score,
            shape=Shape.RECTANGLE,
            normalized_coordinates=normalized_coordinates,
            original_coordinates=original_coordinates,
        )

        annotations.append(annotation)
        confidence_scores.append(float(box.conf))

    sorted_indices = np.argsort(confidence_scores)

    sorted_annotations = []
    for idx in sorted_indices:
        sorted_annotations.append(annotations[idx])

    return sorted_annotations
