from typing import List
from io import BytesIO
import base64
import json

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

from config import GeminiConfig
from web_api import ChatResponse
from maanz_medical_ai_models.predictive.radiology.xray_patch_detector.model_runner import (
    run_model as detect_patches,
)
from maanz_medical_ai_models.predictive.radiology.xray_segmentor.model_runner import (
    make_prediction as segment_image,
)
from maanz_medical_ai_models.predictive.radiology.xray_patch_classifier.model_runner import (
    make_prediction as classify_patches,
)


class ImageType(BaseModel):
    is_xray_image: bool


class ChatSession:
    def __init__(self):
        self.__HISTORY_LENGTH = 10

        self.__client = genai.Client(api_key=GeminiConfig.API_KEY)

        self.__message_history = []
        self.__last_images = []

    def message_history(self):
        return self.__message_history

    def add_message(self, message: ChatResponse):
        self.__message_history.append(message)

    def b64_to_binary(self, b64_string):
        """Convert a base64 string to a PIL Image."""
        image_bytes = base64.b64decode(b64_string)
        return BytesIO(image_bytes).getvalue()

    def b64_to_pil(self, b64_string):
        """Convert a base64 string to a PIL Image."""
        image_bytes = base64.b64decode(b64_string)
        image_buffer = BytesIO(image_bytes)
        return Image.open(image_buffer)

    def pil_to_binary(self, image, encode_format="JPEG"):
        """Convert a PIL Image to binary format."""
        buffer = BytesIO()
        rgb_image = image.convert("RGB")
        rgb_image.save(buffer, format=encode_format)
        return buffer.getvalue()

    def pil_to_b64(self, image, encode_format="JPEG"):
        """Convert a PIL Image to binary format."""
        buffer_value = self.pil_to_binary(image, encode_format)
        return base64.b64encode(buffer_value).decode("utf-8")

    def convert_binary_format_image_for_api(self, input_image):
        MIME_TYPE = "image/jpeg"
        return types.Part.from_bytes(data=input_image, mime_type=MIME_TYPE)

    def medical_image_type(self, images: List[str]):
        CONVERSION_TEXT = "Are the given image/images a xray or not?"
        model_input_contents = [types.Part.from_text(text=CONVERSION_TEXT)]

        for image in images:
            model_input_contents.append(
                self.convert_binary_format_image_for_api(self.b64_to_binary(image))
            )

        return self.__client.models.generate_content(
            model=GeminiConfig.MODEL_NAME,
            contents=model_input_contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ImageType,
            ),
        )

    def construct_history(self):
        total_messages = len(self.__message_history)
        selected_history = self.__message_history[
            max(0, total_messages - self.__HISTORY_LENGTH) :
        ]

        history = "------------- Chat History ----------------\n"
        for response in selected_history:
            history += response.message

        return history

    def process_radiology_image(self, image: str):
        input_image = self.b64_to_pil(image)
        annotations = detect_patches(input_image)
        patches_image = classify_patches(input_image, annotations)
        patches_image_binary = self.pil_to_binary(patches_image)

        segmented_image = segment_image(input_image)
        segmented_image_binary = self.pil_to_binary(segmented_image)

        # Add original scan as well
        self.__last_images.append(
            self.convert_binary_format_image_for_api(self.pil_to_binary(input_image))
        )

        self.__last_images.append(
            self.convert_binary_format_image_for_api(segmented_image_binary)
        )
        self.__last_images.append(
            self.convert_binary_format_image_for_api(patches_image_binary)
        )

        return [self.pil_to_b64(patches_image), self.pil_to_b64(segmented_image)]

    def generate_output(self, question: str, images: List[str]):
        medical_image_type = json.loads(self.medical_image_type(images).text)

        final_question = question
        # TODO(ahmed.nadeem): this is not a nice way please discuss this with AI team
        # This means we have received an image so we need to generate the question
        if not question and len(images) > 0:
            print("Augmenting question!!!")
            if medical_image_type["is_xray_image"]:
                # We only received images means it's the first message add prepared question
                final_question = "Here is a raw scan, along with two augmentation images from our other AI models depicting their detections of the abnormalities. Use this information to give a detailed analysis on the raw scan uploaded."
            else:
                final_question = "Here is a raw pathology scan please give a detailed analysis of this scan on potential problems that this might or might not have"

        history_str = self.construct_history()
        model_input_contents = [
            types.Part.from_text(text=history_str + "\n" + final_question)
        ]

        returned_images = []

        if images:
            self.__last_images = []

        if medical_image_type["is_xray_image"]:
            for image in images:
                processed_images = self.process_radiology_image(image)
                if images:
                    returned_images.extend(processed_images)
        else:
            for image in images:
                self.__last_images.append(
                    self.convert_binary_format_image_for_api(self.b64_to_binary(image))
                )

        # Always add the last sent images so the model can keep the context
        model_input_contents.extend(self.__last_images)

        MODEL_PROMPT = "You are an expert capable of analysing scans in a comprehensive way, detecting anomalies, locating and describing them in relation to the organ, please keep in mind the following points when conversing.\n 1) Give in-depth and accurate analysis for the purpose of aiding medical students in learning \n 2)Don't explicitly add disclaimers that your an AI based assistant as the users are well aware of the limitations\n 3) Don't explicitly add  anything related to learning as well \n 4) Keep the conversation flowing naturally\n 5) Answer according to the question asked\n"
        return (
            self.__client.models.generate_content_stream(
                model=GeminiConfig.MODEL_NAME,
                contents=model_input_contents,
                config=types.GenerateContentConfig(
                    system_instruction=MODEL_PROMPT
                    # max_output_tokens=3,
                    # temperature=0.3,
                ),
            ),
            returned_images,
        )
