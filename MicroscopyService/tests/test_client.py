from io import BytesIO
import base64
import sys
import json

sys.path.append("../src")

import requests
from PIL import Image

from web_api import PredictionRequest

# Define the API endpoint
API_URL = "http://localhost:8000/predict"  # Update with your actual FastAPI endpoint


def pil_to_b64(image, encode_format="PNG"):
    """Convert a PIL Image to binary format."""
    buffer = BytesIO()
    image.save(buffer, format=encode_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def b64_to_pil(b64_data):
    """Convert a PIL Image to binary format."""
    img_data = base64.b64decode(b64_data)
    img = Image.open(BytesIO(img_data))
    return img


def print_prediction_response(response_obj: dict):
    print("Annotations: ")

    def print_coordinate(message, coordinates):
        print(
            "       "
            + message
            + ": "
            + str(coordinates["x"])
            + ", "
            + str(coordinates["y"])
        )

    for idx, annotation in enumerate(response_obj):
        print(" " + str(idx) + ".")
        print("   name: ", annotation["name"])
        print("   confidence: ", annotation["confidence"])
        print("   coordinates: ")

        coordinates = annotation["coordinates"]
        print_coordinate("Top Left", coordinates[0])
        print_coordinate("Top Right", coordinates[1])
        print_coordinate("Bottom Right", coordinates[2])
        print_coordinate("Bottom Left", coordinates[3])


# Create request payload
image = Image.open("mitotic_test.png")
image = Image.open("microscope_uvc_capture_histopathology_1.jpg")
payload = PredictionRequest(image=pil_to_b64(image)).model_dump()

print("Sending Request for Pathology: ")
# Send POST request
#response = requests.post(API_URL, json=payload)

API_URL = "http://localhost:8000/convert"  # Update with your actual FastAPI endpoint
payload = PredictionRequest(image=pil_to_b64(image)).model_dump()
response = requests.post(API_URL, json=payload)
tiff_image = b64_to_pil(response.json()["image"])
tiff_image.save("tiff_output.tiff", format="TIFF")

import sys
sys.exit(0)

# Print response
print("Status Code:", response.status_code)
print("Mitotic:")
print_prediction_response(response.json()["mitotic"])
print()
print("Maybe Mitotic")
print_prediction_response(response.json()["maybe_mitotic"])
print()
print("Mitotic Length:", len(response.json()["mitotic"]))
print("Maybe Mitotic Length:", len(response.json()["maybe_mitotic"]))


