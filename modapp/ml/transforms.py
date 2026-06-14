"""
Image preprocessing pipeline for the ResNet50 feature extractor.

These transforms mirror the preprocessing ResNet50 was trained with on
ImageNet: a fixed 224x224 spatial resolution and per-channel normalization
using the ImageNet mean/std statistics.
"""

from torchvision import transforms

# Standard ImageNet normalization constants.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Resize -> Tensor -> Normalize.
# Resize uses a single (224, 224) target, which is sufficient for a
# similarity-search use case where exact aspect ratio is less critical
# than consistent input dimensions across the whole catalog.
image_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)
