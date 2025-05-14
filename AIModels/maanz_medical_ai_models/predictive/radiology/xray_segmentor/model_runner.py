import os

import numpy as np
import cv2
import scipy.sparse as sp
import torch
from decouple import config
from PIL import Image

from maanz_medical_ai_models.predictive.common.model_input import load_source_image
from maanz_medical_ai_models.predictive.radiology.xray_segmentor.models.hybrid_gnet_2igsc import (
    Hybrid,
)
from maanz_medical_ai_models.predictive.radiology.xray_segmentor.utils.utils import (
    scipy_to_torch_sparse,
    generate_matrices_lungs_heart,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
weights_folder = os.path.join(current_dir, "data", "weights", "weights.pt")
DATA_MODEL_PATH = config(
    "XRAY_SEGMENTOR_RADIOLOGY_MODEL_PATH", default=weights_folder, cast=str
)

# Initialize the model once when the server starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hybrid = None

PATCH_SIZE = 1024


def get_dense_mask(landmarks, h, w):

    RL = landmarks[0:44]
    LL = landmarks[44:94]
    H = landmarks[94:]

    img = np.zeros([h, w], dtype="uint8")

    RL = RL.reshape(-1, 1, 2).astype("int")
    LL = LL.reshape(-1, 1, 2).astype("int")
    H = H.reshape(-1, 1, 2).astype("int")

    img = cv2.drawContours(img, [RL], -1, 1, -1)
    img = cv2.drawContours(img, [LL], -1, 1, -1)
    img = cv2.drawContours(img, [H], -1, 2, -1)

    return img


def draw_on_top(input_image, landmarks, original_shape):
    h, w = original_shape
    output = get_dense_mask(landmarks, h, w)

    image = np.zeros([h, w, 3])
    image[:, :, 0] = (
        input_image
        + 0.3 * (output == 1).astype("float")
        - 0.1 * (output == 2).astype("float")
    )
    image[:, :, 1] = (
        input_image
        + 0.3 * (output == 2).astype("float")
        - 0.1 * (output == 1).astype("float")
    )
    image[:, :, 2] = (
        input_image
        - 0.1 * (output == 1).astype("float")
        - 0.2 * (output == 2).astype("float")
    )

    image = np.clip(image, 0, 1)

    RL, LL, H = landmarks[0:44], landmarks[44:94], landmarks[94:]

    # Draw the landmarks as dots

    for l in RL:
        image = cv2.circle(image, (int(l[0]), int(l[1])), 5, (1, 0, 1), -1)
    for l in LL:
        image = cv2.circle(image, (int(l[0]), int(l[1])), 5, (1, 0, 1), -1)
    for l in H:
        image = cv2.circle(image, (int(l[0]), int(l[1])), 5, (1, 1, 0), -1)

    return image


def load_model(device):
    A, AD, D, U = generate_matrices_lungs_heart()
    N1 = A.shape[0]
    N2 = AD.shape[0]

    A = sp.csc_matrix(A).tocoo()
    AD = sp.csc_matrix(AD).tocoo()
    D = sp.csc_matrix(D).tocoo()
    U = sp.csc_matrix(U).tocoo()

    D_ = [D.copy()]
    U_ = [U.copy()]

    config = {}

    config["n_nodes"] = [N1, N1, N1, N2, N2, N2]
    A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]

    A_t, D_t, U_t = (
        [scipy_to_torch_sparse(x).to(device) for x in X] for X in (A_, D_, U_)
    )

    config["latents"] = 64
    config["inputsize"] = PATCH_SIZE

    f = 32
    config["filters"] = [2, f, f, f, f // 2, f // 2, f // 2]
    config["skip_features"] = f

    hybrid = Hybrid(config.copy(), D_t, U_t, A_t).to(device)
    hybrid.load_state_dict(
        torch.load(DATA_MODEL_PATH, map_location=torch.device(device))
    )
    hybrid.eval()

    return hybrid


def pad_to_square(img):
    h, w = img.shape[:2]

    if h > w:
        padw = h - w
        auxw = padw % 2
        img = np.pad(img, ((0, 0), (padw // 2, padw // 2 + auxw)), "constant")

        padh = 0
        auxh = 0

    else:
        padh = w - h
        auxh = padh % 2
        img = np.pad(img, ((padh // 2, padh // 2 + auxh), (0, 0)), "constant")

        padw = 0
        auxw = 0

    return img, (padh, padw, auxh, auxw)


def preprocess(input_img):
    img, padding = pad_to_square(input_img)

    h, w = img.shape[:2]
    if h != PATCH_SIZE or w != PATCH_SIZE:
        img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)

    return img, (h, w, padding)


def remove_preprocess(output, info):
    h, w, padding = info

    if h != PATCH_SIZE or w != PATCH_SIZE:
        output = output * h
    else:
        output = output * PATCH_SIZE

    padh, padw, auxh, auxw = padding

    output[:, 0] = output[:, 0] - padw // 2
    output[:, 1] = output[:, 1] - padh // 2

    output = output.astype("int")

    return output


def make_prediction(source_image):
    global hybrid, device

    if hybrid is None:
        hybrid = load_model(device)

    # The image was being loaded in grayscale so we convert it after loading
    loaded_image = np.array(load_source_image(source_image))
    input_image = cv2.cvtColor(loaded_image, cv2.COLOR_RGB2GRAY) / 255.0

    original_shape = input_image.shape[:2]
    image, (height, width, padding) = preprocess(input_image)

    data = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device).float()
    with torch.no_grad():
        output = hybrid(data)[0].cpu().numpy().reshape(-1, 2)

    output = remove_preprocess(output, (height, width, padding))

    outseg = draw_on_top(input_image, output, original_shape)

    output_image = (outseg.copy() * 255).astype("uint8")
    #cv2.imwrite("/home/ahmed/Pictures/segmentation_test.jpg", output_image)

    return Image.fromarray(output_image)
