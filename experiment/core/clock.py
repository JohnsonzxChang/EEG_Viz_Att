from __future__ import annotations

import time

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore[assignment]


class PerfClock:
    """High-precision clock using time.perf_counter (Windows QPC backed).

    Both Python's time.perf_counter and the C bridge's QueryPerformanceCounter
    sample the same underlying QPC counter on Windows, so timestamps from
    Python and the Tobii C bridge are directly comparable.
    """

    def __init__(self) -> None:
        self.origin = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter()

    def seconds_since_origin(self, tick: float | None = None) -> float:
        if tick is None:
            tick = self.now()
        return tick - self.origin

    def reset(self) -> None:
        self.origin = time.perf_counter()


class CV2Clock:
    """Clock backed by cv2.getTickCount / cv2.getTickFrequency.

    On Windows cv2.getTickCount() wraps QueryPerformanceCounter, so its tick
    is identical (down to the same QPC count) to time.perf_counter (and to
    the Tobii bridge's tb_qpc_seconds). Used as a fallback ONSET timestamp
    source when EEG TTL is disabled — the per-stimulus onset event then
    carries a cv2-derived `cv2_tick_qpc` field that offline analysis can
    rely on without the EEG amplifier's hardware trigger.
    """

    def __init__(self) -> None:
        if cv2 is None:
            raise ImportError("cv2 (opencv-python) is required for CV2Clock")
        self._freq = float(cv2.getTickFrequency())
        self.origin_ticks = int(cv2.getTickCount())
        self.origin = self.origin_ticks / self._freq

    def now(self) -> float:
        return float(cv2.getTickCount()) / self._freq

    def now_ticks(self) -> int:
        return int(cv2.getTickCount())

    def seconds_since_origin(self, tick: float | None = None) -> float:
        if tick is None:
            tick = self.now()
        return tick - self.origin

    @property
    def frequency(self) -> float:
        return self._freq
