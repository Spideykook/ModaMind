"""
Public interface for converting clothing images into embedding vectors.

This module has no Django dependencies and can be imported and tested in
isolation (see scripts/test_pipeline.py).
"""

import io
import logging
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model_loader import ModelLoader

logger = logging.getLogger(__name__)

ImageInput = Union[str, bytes, Image.Image]


class EmbeddingService:
    """
    Converts an input image into an L2-normalized 2048-dimensional
    embedding vector using a ResNet50 backbone with its final
    classification layer removed.

    Because the output is unit-norm, the dot product of two embeddings
    equals their cosine similarity — which is exactly what
    faiss.IndexFlatIP computes. This makes the output of
    `extract_embedding()` directly compatible with FaissManager.
    """

    EMBEDDING_DIM = 2048

    def __init__(self) -> None:
        loader = ModelLoader()
        self.model = loader.get_model()
        self.device = loader.get_device()
        self.transform = loader.get_transforms()

    def preprocess_image(self, image_input: ImageInput) -> torch.Tensor:
        """
        Normalize a variety of image input types into a single
        (1, 3, 224, 224) tensor ready for the model's forward pass.

        Args:
            image_input: One of
                - a filesystem path (str)
                - raw image bytes (e.g. from an uploaded file's .read())
                - an already-open PIL.Image.Image

        Returns:
            A 4D torch.Tensor of shape (1, 3, 224, 224).
        """
        if isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)!r}")

        # Convert to RGB to normalize away PNG alpha channels, grayscale
        # scans, or palette-based images before feeding ResNet50.
        image = image.convert("RGB")

        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # add batch dimension -> (1, 3, 224, 224)

    @torch.no_grad()
    def extract_embedding(self, image_input: ImageInput) -> np.ndarray:
        """
        Run the full pipeline: preprocess -> forward pass -> flatten ->
        L2-normalize.

        Args:
            image_input: see `preprocess_image`.

        Returns:
            A 1D float32 numpy array of shape (2048,) with unit L2 norm.
        """
        tensor = self.preprocess_image(image_input).to(self.device)

        features = self.model(tensor)                     # (1, 2048, 1, 1)
        features = torch.flatten(features, start_dim=1)   # (1, 2048)

        normalized = F.normalize(features, p=2, dim=1)     # unit L2 norm, dim=1

        return normalized.cpu().numpy().astype("float32").squeeze(0)
