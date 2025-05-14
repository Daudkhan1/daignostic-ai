import os
import sys
import uuid

from urllib.parse import urlparse
from PIL import Image
import requests
from io import BytesIO


def is_pil_image(obj):
    return isinstance(obj, Image.Image)


def is_local_file_path(path: str) -> bool:
    """Checks if the given string is a valid local file path on Ubuntu."""
    return os.path.exists(path)


def is_url(path: str) -> bool:
    """Checks if the given string is a URL."""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https", "ftp", "file") and bool(parsed.netloc)


def load_source_image(input_image):
    # Download file
    if is_pil_image(input_image):
        if input_image.mode == "RGB":
            return input_image
        else:
            print(f"Image is in {input_image.mode} mode. Converting to RGB.")
            return input_image.convert("RGB")
    elif is_local_file_path(input_image):
        image = Image.open(input_image)  # Create a file-like object
    elif is_url(input_image):
        response = requests.get(input_image)
        response.raise_for_status()  # Ensure the request was successful
        image = Image.open(BytesIO(response.content))  # Create a file-like object
    else:
        raise Exception(
            "Is not a valid url or a file path check if it is a path it exists: \n"
            + input_image
        )

    return image


def download_image(image_url):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        file_name = str(uuid.uuid4())
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB

        mb_total = total_size / (1024**2)
        print("Size of file is: ", mb_total, " MBs")

        downloaded = 0
        with open(file_name, "wb") as file:
            for chunk in response.iter_content(block_size):
                file.write(chunk)

                # Progress bar
                downloaded += len(chunk)
                done = int(50 * downloaded / total_size) if total_size else 0
                percent = (downloaded / total_size) * 100 if total_size else 0
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {percent:6.2f}%")
                sys.stdout.flush()

        print()
        print(f"Image downloaded successfully as {file_name}")
        return file_name
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
