import os
import glob
from datetime import datetime
import traceback

from fastapi import FastAPI, HTTPException

from maanz_medical_ai_models.predictive.pathology.model_runner import make_prediction
from maanz_medical_ai_models.predictive.radiology.xray_patch_detector.model_runner import (
    run_model,
)
from maanz_medical_ai_models.predictive.common.model_input import download_image, load_source_image
from web_api import (
    PredictionResponse,
    PredictionRequest,
    Modality,
    BiologicalType,
    Shape,
    AnnotationType,
    Annotation,
)


def convert_from_ai_annotations_to_response(ai_annotations, request):
    response_annotations = []
    for annotation in ai_annotations:
        dictionary = annotation.to_dict()
        response_annotations.append(
            Annotation(
                name="AI Detection",
                biological_type=BiologicalType(dictionary["biological_type"]),
                shape=Shape(dictionary["shape"]),
                confidence=dictionary["confidence"],
                description=dictionary["description"],
                coordinates=dictionary["normalized_coordinates"],
            )
        )

    return PredictionResponse(
        patient_slide_id=request.patient_slide_id,
        annotation_type=AnnotationType.AI,
        annotations=response_annotations,
    )


# Initialize API Model
app = FastAPI()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        print("-----------Received Request----------")

        # Get current system time
        # Print in a formatted string (e.g., HH:MM:SS)
        current_time = datetime.now()
        print("Start Time:", current_time.strftime("%H:%M:%S"))

        print("Downloading image: ")
        file_name = download_image(request.image_path)

        print("Modality: ", request.modality)
        print("Patient Slide ID: ", request.patient_slide_id)
        print("Clipped URL: ", request.image_path[: min(len(request.image_path), 150)])

        # Load the image
        image = load_source_image(file_name)

        # Get width, height, and bands
        width = image.width
        height = image.height

        print("Width: ", width, "height, ", height)

        annotations = None
        if request.modality == Modality.RADIOLOGY:
            annotations = run_model(request.image_path)
        elif request.modality == Modality.PATHOLOGY:
            # Default Parameters for whole slide image
            MITOTIC_TILE_SIZE = 100
            # The max can be 255 this means we are not thresholding
            WHITE_PATCH_MEAN = 3000
            CONFIDENCE_THRESHOLD = 0.8

            # Whole slide image
            if (width == 3000 and height == 4000) or (width == 4000 and height == 3000):
                MITOTIC_TILE_SIZE = 400
                WHITE_PATCH_MEAN = 160
                CONFIDENCE_THRESHOLD = 0.5

            annotations = make_prediction(
                file_name, MITOTIC_TILE_SIZE, CONFIDENCE_THRESHOLD, WHITE_PATCH_MEAN
            )
            annotations = annotations["mitotic"]
            print("Total Annotations: ", len(annotations))
        else:
            raise Exception("Received Unknown Modality: ", request.modality)

        current_time = datetime.now()
        print("End Time:", current_time.strftime("%H:%M:%S"))
        print("-------------------------------------")

        os.remove(file_name)

        return convert_from_ai_annotations_to_response(annotations, request)

    except Exception as e:
        # Get all .dat files in /tmp
        dat_files = glob.glob("/tmp/*.dat")

        # Delete each file
        for file_path in dat_files:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
