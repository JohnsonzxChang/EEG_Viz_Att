from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageItem:
    image_id: str
    file_path: Path
    relative_path: str = ""
    targets: list[str] = field(default_factory=list)
    target_areas: dict[str, float] = field(default_factory=dict)
    bboxes: dict[str, list[float]] = field(default_factory=dict)
    split: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StimulusBundle:
    targets: list[str]
    K: int
    images_by_target: dict[str, list[ImageItem]]
    all_images: list[ImageItem]

    def __len__(self) -> int:
        return len(self.all_images)


class ImageDataset(ABC):
    """Two-phase dataset interface: select_stimuli (Phase1) -> bundle_from_selection (Phase2)."""

    @abstractmethod
    def load(self, config: dict[str, Any]) -> None: ...

    @abstractmethod
    def get_categories(self) -> list[str]: ...

    @abstractmethod
    def select_stimuli(self, config: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def bundle_from_selection(
        self, selection: dict[str, Any], dataset_root: str | None = None
    ) -> StimulusBundle: ...
