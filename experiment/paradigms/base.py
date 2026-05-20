from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.display import DisplayEngine
    from core.logger import EventLogger
    from datasets.base import StimulusBundle
    from markers.base import MarkerManager


class Paradigm(ABC):
    def __init__(
        self,
        display: "DisplayEngine",
        logger: "EventLogger",
        marker_mgr: "MarkerManager",
    ) -> None:
        self.display = display
        self.logger = logger
        self.marker_mgr = marker_mgr

    @abstractmethod
    def run(self, bundle: "StimulusBundle", config: dict[str, Any]) -> None:
        """Execute the paradigm."""
