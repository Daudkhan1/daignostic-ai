import os
import uuid
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from maanz_medical_ai_models.predictive.common.model_input import load_source_image

Image.MAX_IMAGE_PIXELS = None


class ImageDataset(Dataset):
    def __init__(self, source_image, mitotic_tile_size, base_tile_size):
        img = load_source_image(source_image)

        # Create tiles
        self.width = img.width
        self.height = img.height

        BASE_TILE_SIZE = base_tile_size
        MITOTIC_TILE_SIZE = mitotic_tile_size

        if img.width % MITOTIC_TILE_SIZE != 0:
            self.idx_i = img.width // MITOTIC_TILE_SIZE + 1
        else:
            self.idx_i = img.width // MITOTIC_TILE_SIZE

        if img.height % MITOTIC_TILE_SIZE != 0:
            self.idx_j = img.height // MITOTIC_TILE_SIZE + 1
        else:
            self.idx_j = img.height // MITOTIC_TILE_SIZE

        # Estimate the total number of tiles
        num_tiles = self.idx_i * self.idx_j

        # Save in tmp because it is removed on each restart
        self.file_name = "/tmp/" + str(uuid.uuid4()) + ".dat"

        start_time = time.time()
        # Create a memory-mapped array on disk (this avoids memory overload)
        self.images = np.memmap(
            self.file_name,
            dtype="uint8",
            mode="w+",
            shape=(num_tiles, BASE_TILE_SIZE, BASE_TILE_SIZE, 3),
        )

        tile_idx = 0
        self.row_indexes = []
        self.col_indexes = []
        # Loop through the image in 100x100 tiles
        for top in range(0, img.height, MITOTIC_TILE_SIZE):
            for left in range(0, img.width, MITOTIC_TILE_SIZE):
                # Define the box for the current tile (left, top, right, bottom)
                box = (
                    left,
                    top,
                    min(left + MITOTIC_TILE_SIZE, img.width),
                    min(top + MITOTIC_TILE_SIZE, img.height),
                )

                # Crop the tile from the image
                tile = img.crop(box).convert("RGB")
                tile = tile.resize((BASE_TILE_SIZE, BASE_TILE_SIZE))
                tile_array = np.array(tile)

                # Append the tile array to the list
                self.images[
                    tile_idx, : tile_array.shape[0], : tile_array.shape[1], :
                ] = tile_array
                tile_idx += 1
                self.row_indexes.append(top // MITOTIC_TILE_SIZE)
                self.col_indexes.append(left // MITOTIC_TILE_SIZE)

                del tile_array

        self.images.flush()

        end_time = time.time()
        print(
            f"  Processing time for image to np.memmap: {end_time - start_time:.2f} seconds"
        )

        img.close()

    def __len__(self):
        return self.images.shape[0]

    def get_cols_and_rows(self):
        return self.idx_i, self.idx_j

    def get_width_and_height(self):
        return self.width, self.height

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.from_numpy(image)
        image = (
            (image / 255.0).permute(2, 0, 1).unsqueeze(dim=0)
        )  # scale image in a range

        return image, self.row_indexes[idx], self.col_indexes[idx]
