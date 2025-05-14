from io import BytesIO
import base64
import traceback

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from PIL import Image

from maanz_medical_ai_models.predictive.pathology.model_runner import make_prediction
from web_api import (
    PredictionResponse,
    PredictionRequest,
    Annotation,
    ConversionResponse,
)


def b64_to_pil(b64_string):
    """Convert a base64 string to a PIL Image."""
    image_bytes = base64.b64decode(b64_string)
    image_buffer = BytesIO(image_bytes)
    return Image.open(image_buffer)


def pil_to_b64_tif(pil_image: Image):
    """Convert a PIL Image to base64 string .tiff."""
    tiff_byte_stream = BytesIO()
    pil_image.save(tiff_byte_stream, format="TIFF")
    return base64.b64encode(tiff_byte_stream.getvalue()).decode("utf-8")


def convert_from_ai_annotations_to_response(ai_annotations):
    response_annotations = []
    for annotation in ai_annotations:
        dictionary = annotation.to_dict()
        response_annotations.append(
            Annotation(
                name="AI Detection",
                confidence=dictionary["confidence"],
                coordinates=dictionary["original_coordinates"],
            )
        )

    return response_annotations


# Initialize API Model
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # You can restrict this to ["http://localhost:5173"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        print("-----------Received Request----------")
        input_image = b64_to_pil(request.image)

        print("Size: ", input_image.size)

        MITOTIC_TILE_SIZE = 400
        CONFIDENCE_THRESHOLD = 0.5
        WHITE_PATCH_MEAN = 160
        annotations = make_prediction(
            input_image, MITOTIC_TILE_SIZE, CONFIDENCE_THRESHOLD, WHITE_PATCH_MEAN
        )
        mitotic_annotations = annotations["mitotic"]
        maybe_mitotic_annotations = annotations["maybe_mitotic"]

        print("Mitotic Annotations: ", len(mitotic_annotations))
        print("Maybe Mitotic Annotations: ", len(maybe_mitotic_annotations))

        print("-------------------------------------")

        mitotic_annotations = convert_from_ai_annotations_to_response(
            mitotic_annotations
        )
        maybe_mitotic_annotations = convert_from_ai_annotations_to_response(
            maybe_mitotic_annotations
        )

        return PredictionResponse(
            mitotic=mitotic_annotations, maybe_mitotic=maybe_mitotic_annotations
        )
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/convert", response_model=ConversionResponse)
async def convert(request: PredictionRequest):
    try:
        print("-----------Received Request----------")
        input_image = b64_to_pil(request.image)
        converted_image = pil_to_b64_tif(input_image)
        print("Size: ", input_image.size)
        return ConversionResponse(image=converted_image)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)[:100])
