import os
from urllib.parse import urlparse
import requests
from PIL import Image

import numpy as np

def is_local_file_path(path: str) -> bool:
    """Checks if the given string is a valid local file path on Ubuntu."""
    return os.path.exists(path)


def is_url(path: str) -> bool:
    """Checks if the given string is a URL."""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https", "ftp", "file") and bool(parsed.netloc)


def load_image(input_: str):
    if is_local_file_path(input_):
    elif is_url(input_):
        response = requests.get(image_url)
        response.raise_for_status()
        tiff_image = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(tiff_image)[:, :, ::-1].copy()
