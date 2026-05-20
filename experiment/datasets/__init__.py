"""Dataset adapters."""

from datasets.base import ImageDataset, ImageItem, StimulusBundle
from datasets.lvis_coco import LVISCOCODataset

DATASET_REGISTRY: dict[str, type[ImageDataset]] = {
    "lvis_coco": LVISCOCODataset,
}

__all__ = ["ImageDataset", "ImageItem", "StimulusBundle", "DATASET_REGISTRY"]
