import os
import sys
import warnings
import glob

import numpy as np
from tqdm import tqdm
from decouple import config
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

from maanz_medical_ai_models.predictive.pathology.model import MitoticModel
from maanz_medical_ai_models.predictive.pathology.dataset import ImageDataset
from maanz_medical_ai_models.predictive.common.model_constants import (
    BATCH_SIZE,
)
from maanz_medical_ai_models.predictive.common.model_output import (
    Annotation,
    BiologicalType,
    Shape,
    Coordinate,
    BoundingBox,
)

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
weights_folder = os.path.join(current_dir, "data", "weights", "model_weights_od.pth")
DATA_MODEL_PATH = config(
    "PREDICTIVE_PATHOLOGY_MODEL_PATH", default=weights_folder, cast=str
)

# Initialize the model once when the server starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mitotic_model = MitoticModel()
mitotic_model.load_state_dict(torch.load(DATA_MODEL_PATH, map_location=device))
mitotic_model.eval()
mitotic_model = mitotic_model.to(device)


def make_prediction(
    image_path: str, mitotic_tile_size, confidence_threshold, white_patch_mean
):
    # Load dataset and image
    WHITE_PATCH_MEAN = white_patch_mean
    CONFIDENCE_THRESHOLD = confidence_threshold
    MITOTIC_TILE_SIZE = mitotic_tile_size
    BASE_TILE_SIZE = 100

    print("Configuration: ")
    print("   Tile size: ", MITOTIC_TILE_SIZE)
    print("   WHITE_PATCH_MEAN: ", WHITE_PATCH_MEAN)
    print("   CONFIDENCE_THRESHOLD: ", CONFIDENCE_THRESHOLD)

    dataset = ImageDataset(image_path, mitotic_tile_size, BASE_TILE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    width, height = dataset.get_width_and_height()

    mitotic_annotation_list = []
    mitotic_confidence_scores = []

    maybe_mitotic_annotation_list = []
    maybe_mitotic_confidence_scores = []

    for images, row_indexes, col_indexes in tqdm(dataloader):
        squeezed_images = torch.squeeze(images.to(device))
        prediction = mitotic_model(squeezed_images)
        probs = F.softmax(prediction[0], dim=1).detach().cpu().numpy()
        labels = np.argmax(probs, axis=1)
        indexes = np.argwhere(labels > 0)

        for index in indexes:
            row_counter, col_counter = (
                row_indexes[index].detach().cpu().numpy()[0],
                col_indexes[index].detach().cpu().numpy()[0],
            )

            mitotic_prob = probs[index]
            mitotic_confidence_score = mitotic_prob[0][-1]
            maybe_mitotic_confidence_score = mitotic_prob[0][-2]
            if (
                mitotic_confidence_score < CONFIDENCE_THRESHOLD
                and maybe_mitotic_confidence_score < CONFIDENCE_THRESHOLD
            ):
                continue

            # This means we are mostly dealing with a white patch
            if np.mean(images[index].numpy() * 255) >= WHITE_PATCH_MEAN:
                continue

            top_left = Coordinate(
                int(MITOTIC_TILE_SIZE * col_counter),
                int(MITOTIC_TILE_SIZE * row_counter),
            )

            top_right = Coordinate(
                int(MITOTIC_TILE_SIZE * col_counter) + MITOTIC_TILE_SIZE,
                int(MITOTIC_TILE_SIZE * row_counter),
            )

            bottom_right = Coordinate(
                int(MITOTIC_TILE_SIZE * col_counter) + MITOTIC_TILE_SIZE,
                int((MITOTIC_TILE_SIZE * row_counter) + MITOTIC_TILE_SIZE),
            )

            bottom_left = Coordinate(
                int(MITOTIC_TILE_SIZE * col_counter),
                int((MITOTIC_TILE_SIZE * row_counter)) + MITOTIC_TILE_SIZE,
            )

            original_coordinates = BoundingBox(
                top_left, top_right, bottom_right, bottom_left
            )

            # Copy original coordinates and multiply with related factor to normalize it
            normalized_coordinates = original_coordinates.copy()
            normalized_coordinates.multiply(100 / width, 100 / height)

            confidence_score = 0
            annotation_list = None
            if mitotic_confidence_score > CONFIDENCE_THRESHOLD:
                confidence_score = mitotic_confidence_score
                annotation_list = mitotic_annotation_list
                confidence_scores = mitotic_confidence_scores
            else:
                confidence_score = maybe_mitotic_confidence_score
                annotation_list = maybe_mitotic_annotation_list
                confidence_scores = maybe_mitotic_confidence_scores

            annotation = Annotation(
                biological_type=BiologicalType.MITOTIC,
                shape=Shape.SQUARE,
                confidence=confidence_score,
                normalized_coordinates=normalized_coordinates,
                original_coordinates=original_coordinates,
            )

            annotation_list.append(annotation)
            confidence_scores.append(confidence_score)

    def sort_indices(input_list, input_scores):
        sorted_indices = np.argsort(input_scores)

        sorted_annotations = []
        for idx in sorted_indices:
            sorted_annotations.append(input_list[idx])

        return sorted_annotations

    print("Removing file: ", dataset.file_name)
    # print("Before deleting")
    # print(glob.glob("/tmp/*"))
    # print("-----------------------")
    os.remove(dataset.file_name)
    # print("After deleting")
    # print(glob.glob("/tmp/*"))
    # print()

    return {
        "mitotic": sort_indices(mitotic_annotation_list, mitotic_confidence_scores),
        "maybe_mitotic": sort_indices(
            maybe_mitotic_annotation_list, maybe_mitotic_confidence_scores
        ),
    }
