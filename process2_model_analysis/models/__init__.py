"""Encoder zoo: EEGNet, ATM, fusion."""

from .eegnet import EEGNet
from .atm import ATMEncoder
from .fusion import (
    EEGOnlyClassifier,
    ImgOnlyClassifier,
    EEGImgFusionClassifier,
    EEGImgPatchFusion,
)

__all__ = [
    "EEGNet", "ATMEncoder",
    "EEGOnlyClassifier", "ImgOnlyClassifier",
    "EEGImgFusionClassifier", "EEGImgPatchFusion",
]
