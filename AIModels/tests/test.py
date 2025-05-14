import os

from PIL import ImageDraw, Image

from maanz_medical_ai_models.predictive.pathology.model_runner import make_prediction
from maanz_medical_ai_models.predictive.radiology.xray_patch_detector.model_runner import (
    run_model,
)
from maanz_medical_ai_models.predictive.radiology.xray_segmentor.model_runner import (
    make_prediction as xseg_runner,
)
from maanz_medical_ai_models.predictive.radiology.xray_patch_classifier.model_runner import (
    make_prediction as xpatch_runner,
)


def draw_rectange(draw, annotations, color):
    for annotation in annotations:
        # Define box coordinates (left, top, right, bottom)
        top_left = annotation.original_coordinates.top_left
        bottom_right = annotation.original_coordinates.bottom_right
        box_coords = (top_left.x(), top_left.y(), bottom_right.x(), bottom_right.y())

        # Draw a rectangle (outline only)
        draw.rectangle(box_coords, outline=color, width=10)


def load_image(path, output_path):
    annotation = make_prediction(path, 350)
    input_image = Image.open(path)
    draw = ImageDraw.Draw(input_image)
    print(annotation)
    draw_rectange(draw, annotation["mitotic"], "red")
    draw_rectange(draw, annotation["maybe_mitotic"], "blue")

    input_image.save(output_path)


def annotate_folder(input_folder, output_folder):
    files = [
        os.path.join(input_folder, file)
        for file in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, file))
    ]

    for file in files:
        file_name = os.path.basename(file)
        write_path = os.path.join(output_folder, file_name)
        print("Input file: ", file)
        print("Writing to: ", write_path)
        load_image(file, write_path)


#annotations = run_model("radiology.tiff")
#print("Annotations from radiology: ", annotations)
#print(
#    "Annotations from radiology patch: ",
#    xpatch_runner("radiology.tiff", annotations),
#)
#print("Annotations from radiology segmentation: ", xseg_runner("xray_segmentor.jpg"))


# annotate_folder(
#    "/home/ahmed/Pictures/wsi/microscope", "/home/ahmed/Pictures/experiments/350x350"
# )

make_prediction("/home/ahmed/Pictures/wsi/microscope/slide_1.tiff", 100, 0.8, 3000)
