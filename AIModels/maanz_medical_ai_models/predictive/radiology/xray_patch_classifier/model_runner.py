import os
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torchvision import models
from decouple import config
from PIL import Image

from maanz_medical_ai_models.predictive.common.model_input import load_source_image
from maanz_medical_ai_models.predictive.common.model_output import (
    Annotation,
    BoundingBox,
)

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
weights_folder = os.path.join(current_dir, "data", "weights", "weights.pth")
DATA_MODEL_PATH = config(
    "XRAY_PATCH_CLASSIFIER_RADIOLOGY_MODEL_PATH", default=weights_folder, cast=str
)

# Define your disease classes
CLASS_NAMES = [
    "Aortic Enlargement",
    "Cardiomegaly",
    "Others",
    "PE-LO-NM",
    "Pleural Thickening",
    "Pulmonary Fibrosis",
]

PATCH_SIZE = 64


def load_model():
    model = models.efficientnet_b0(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_inputs = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_inputs, PATCH_SIZE),
        nn.SiLU(),
        nn.Dropout(0.8),
        nn.Linear(PATCH_SIZE, len(CLASS_NAMES)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    efficientnet_model = model
    efficientnet_model.load_state_dict(torch.load(DATA_MODEL_PATH, map_location=device))
    efficientnet_model.eval()

    return efficientnet_model


# Load Model on file import so it can be reused
MODEL = load_model()


def extract_patches(image, boxes: List[BoundingBox]):
    patches = []
    for box in boxes:
        x1, y1 = box.top_left.get_tuple()
        x2, y2 = box.bottom_right.get_tuple()

        # Extract the patch
        patch = image[y1:y2, x1:x2]

        # Resize to 64x64 using cubic interpolation
        resized_patch = cv2.resize(
            patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC
        )

        patches.append((resized_patch, (x1, y1, x2, y2)))

    return patches


def classify_patches(patches):
    results = []

    # Define preprocessing transformations
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for (patch, box_coords) in patches:
        # Preprocess the patch
        input_tensor = transform(patch).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            outputs = MODEL(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get predicted class
            _, predicted_idx = torch.max(probabilities, 1)
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            confidence = probabilities[0][predicted_idx.item()].item()

        results.append((box_coords, predicted_class, confidence))

    return results


def make_prediction(source_image: str, annotations: List[Annotation]):
    loaded_image = np.array(load_source_image(source_image))
    image = cv2.cvtColor(loaded_image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)

    output_image = image.copy()

    patches = extract_patches(
        image, [annotation.original_coordinates for annotation in annotations]
    )

    classification_results = classify_patches(patches)

    color = (0, 255, 0)  # Green color (BGR)
    thickness = 2  #

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (255, 0, 0)  # White color
    text_thickness = 2

    for result in classification_results:
        cv2.rectangle(
            output_image,
            (result[0][0], result[0][1]),
            (result[0][2], result[0][3]),
            color,
            thickness,
        )

        cv2.putText(
            output_image,
            result[1],
            (max(result[0][0] - 5, 0), result[0][1]),
            font,
            font_scale,
            text_color,
            text_thickness,
        ),

    # cv2.imwrite("/home/ahmed/Pictures/patch_test.jpg", output_image)

    return Image.fromarray(output_image)
