from __future__ import annotations

import numpy as np


class PhotodiodeMarker:
    """White/black square stamped in a screen corner for photodiode detection."""

    CORNERS = {
        "top_left": lambda w, h, s: (0, 0),
        "top_right": lambda w, h, s: (w - s, 0),
        "bottom_left": lambda w, h, s: (0, h - s),
        "bottom_right": lambda w, h, s: (w - s, h - s),
    }

    def __init__(self, size: int = 60, corner: str = "bottom_right") -> None:
        self.size = size
        self.corner = corner
        if corner not in self.CORNERS:
            raise ValueError(f"Invalid corner: {corner!r}. Must be one of {list(self.CORNERS)}")

    def stamp(self, frame: np.ndarray, active: bool) -> np.ndarray:
        # Some upstream sources (PIL → np.asarray) return read-only buffer
        # views that cannot be mutated in-place. Defensively copy when not
        # writable so the stamp succeeds regardless of where `frame`
        # originated. Cost is one allocation per HINT/REST screen — these
        # frames are not in the high-rate ON/ISI path, so the overhead is
        # negligible.
        if not frame.flags.writeable:
            frame = np.array(frame, copy=True)
        h, w = frame.shape[:2]
        s = min(self.size, w, h)
        x0, y0 = self.CORNERS[self.corner](w, h, s)
        color = 255 if active else 0
        frame[y0 : y0 + s, x0 : x0 + s] = color
        return frame
