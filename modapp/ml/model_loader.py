"""
Thread-safe singleton loader for the ResNet50 feature-extraction backbone.

This module guarantees the (relatively large) pretrained ResNet50 weights
are loaded into memory exactly once per process, regardless of how many
times EmbeddingService() is instantiated or how many concurrent requests
the Django dev server / WSGI workers handle.
"""

import logging
import threading

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from .transforms import image_transforms

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Thread-safe singleton that owns the ResNet50 feature-extraction model.

    The standard `__new__` double-checked locking pattern below ensures
    that even if multiple threads call ModelLoader() simultaneously on
    first use, the expensive `_initialize()` step (downloading/loading
    weights onto a device) runs only once.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("ModaMind: loading ResNet50 backbone on device '%s'", self.device)

        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Strip the final classification layer ('fc').
        # nn.Module.children() yields ResNet50's top-level layers in
        # definition order; the last one is the 1000-class 'fc' head.
        # Everything before it (conv stem -> residual blocks -> avgpool)
        # is preserved, producing a (batch, 2048, 1, 1) feature map.
        self.model = nn.Sequential(*list(base_model.children())[:-1])

        self.model.to(self.device)
        self.model.eval()  # disable dropout / freeze batchnorm running stats

        logger.info("ModaMind: ResNet50 feature extractor ready (2048-d output).")

    def get_model(self) -> nn.Module:
        return self.model

    def get_device(self) -> torch.device:
        return self.device

    def get_transforms(self):
        return image_transforms
