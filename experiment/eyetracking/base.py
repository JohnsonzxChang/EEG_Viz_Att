from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class EyeTracker(ABC):
    @abstractmethod
    def start(self, output_path: Path) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def mark(self, tag: str) -> None:
        """Insert a marker line in the gaze stream, time-stamped with QPC.
        Useful for offline alignment with EEG markers / paradigm events.
        """

    def get_latest_gaze(self) -> tuple[float, float] | None:
        """Return the most recent gaze sample in normalized [-1, 1] screen
        coordinates (Tobii TGI native), or ``None`` if no sample is yet
        available. Implementations should make this thread-safe and
        non-blocking — it is called every display frame from the paradigm
        loop to drive the live ISI cursor and the on-stimulus visual-
        deviation check.

        Default implementation returns ``None`` so that paradigms that
        request live gaze degrade gracefully under ``NullEyeTracker``.
        """
        return None

    def wait_for_data(self, timeout_s: float = 2.5) -> bool:
        """Block up to ``timeout_s`` waiting for the first valid gaze
        sample. Used as a startup gate — if the tracker hardware is
        disconnected or the SDK fails to deliver any data, the experiment
        should *fail fast* rather than silently record empty files.

        Default returns True so that ``NullEyeTracker`` (when eye-tracking
        is disabled in config) does not block startup.
        """
        return True


class NullEyeTracker(EyeTracker):
    """No-op eye tracker. Used when eye-tracking is disabled."""

    def start(self, output_path: Path) -> None: ...
    def stop(self) -> None: ...
    def mark(self, tag: str) -> None: ...
    def get_latest_gaze(self) -> tuple[float, float] | None:
        return None
    def wait_for_data(self, timeout_s: float = 2.5) -> bool:
        return True
