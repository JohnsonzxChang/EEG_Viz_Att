from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── PIL TTF font cache (lazy import — Pillow only needed for CJK text) ──


@lru_cache(maxsize=8)
def _load_truetype_font(font_path: str, size: int):
    """Load a TTF/TTC at a given size. Cached because PIL font parsing is
    not free and we re-use the same HINT font across the whole session."""
    try:
        from PIL import ImageFont  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Chinese HINT rendering requires Pillow — `pip install Pillow`"
        ) from e
    p = Path(font_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"HINT font not found: {font_path}. "
            "Set paradigm.hint_font_path in the YAML to a valid TTF/TTC."
        )
    # ttc files contain multiple faces — index 0 is the regular face.
    if p.suffix.lower() == ".ttc":
        return ImageFont.truetype(str(p), size=size, index=0)
    return ImageFont.truetype(str(p), size=size)


def make_blank(width: int, height: int, color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def make_fixation(
    width: int,
    height: int,
    arm_length: int = 32,
    thickness: int = 4,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    fg_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    frame = make_blank(width, height, bg_color)
    cx, cy = width // 2, height // 2
    cv2.line(frame, (cx - arm_length, cy), (cx + arm_length, cy), fg_color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm_length), (cx, cy + arm_length), fg_color, thickness, cv2.LINE_AA)
    return frame


def make_text_canvas(
    text: str,
    width: int,
    height: int,
    *,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    fg_color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 4.0,
    thickness: int = 6,
    subtitle: str | None = None,
    subtitle_scale: float = 1.6,
) -> np.ndarray:
    """Centered text canvas. Used for HINT cue at the start of each trial."""
    frame = make_blank(width, height, bg_color)
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (width - tw) // 2
    y = (height + th) // 2
    cv2.putText(frame, text, (x, y), font, font_scale, fg_color, thickness, cv2.LINE_AA)

    if subtitle:
        (sw, sh), _ = cv2.getTextSize(subtitle, font, subtitle_scale, max(2, thickness // 2))
        sx = (width - sw) // 2
        sy = y + th + sh + 24
        cv2.putText(frame, subtitle, (sx, sy), font, subtitle_scale,
                    (180, 180, 180), max(2, thickness // 2), cv2.LINE_AA)

    return frame


def make_text_canvas_pil(
    text: str,
    width: int,
    height: int,
    *,
    font_path: str,
    font_size: int = 240,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    fg_color: tuple[int, int, int] = (255, 255, 255),
    subtitle: str | None = None,
    subtitle_size: int = 56,
    subtitle_color: tuple[int, int, int] = (180, 180, 180),
    subtitle_gap_px: int = 32,
) -> np.ndarray:
    """Centered text canvas using PIL+TTF — required for CJK characters
    because cv2.putText only ships with Hershey vector fonts.

    Output is RGB uint8, ready to feed into the OpenGL display path.
    Falls back to :func:`make_text_canvas` (cv2) if Pillow is unavailable
    or the font cannot be loaded; this preserves the legacy English path.
    """
    try:
        from PIL import Image, ImageDraw  # type: ignore

        font = _load_truetype_font(font_path, font_size)
        sub_font = (
            _load_truetype_font(font_path, subtitle_size)
            if subtitle else None
        )
    except Exception as e:
        log.warning(
            "Falling back to cv2 text rendering (CJK glyphs may not display): %s",
            e,
        )
        return make_text_canvas(
            text, width, height,
            bg_color=bg_color, fg_color=fg_color,
            font_scale=4.0, thickness=6,
            subtitle=subtitle,
        )

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # PIL font.getbbox returns (l, t, r, b) of the rendered glyph string.
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) // 2 - bbox[0]
    y = (height - th) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=fg_color)

    if subtitle and sub_font is not None:
        sb = draw.textbbox((0, 0), subtitle, font=sub_font)
        sw = sb[2] - sb[0]
        sh = sb[3] - sb[1]
        sx = (width - sw) // 2 - sb[0]
        sy = y + th + subtitle_gap_px - sb[1]
        draw.text((sx, sy), subtitle, font=sub_font, fill=subtitle_color)

    # `np.asarray(PIL.Image)` returns a read-only buffer view; downstream
    # consumers (photodiode.stamp, draw_annotation_outline) mutate the
    # array in-place. Always return a writable copy.
    return np.array(img, dtype=np.uint8)


def load_image(path: str) -> np.ndarray | None:
    """Read image from disk and convert BGR -> RGB for OpenGL."""
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit_image_to_canvas(
    image: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    canvas, _, _, _ = fit_image_to_canvas_with_transform(
        image, canvas_width, canvas_height, bg_color)
    return canvas


def fit_image_to_canvas_with_transform(
    image: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, float, int, int]:
    canvas = make_blank(canvas_width, canvas_height, bg_color)
    img_h, img_w = image.shape[:2]
    scale = min(canvas_width / img_w, canvas_height / img_h)
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (canvas_width - new_w) // 2
    y0 = (canvas_height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas, scale, x0, y0


def draw_annotation_outline(
    canvas: np.ndarray,
    *,
    bbox: list[float] | None = None,
    segmentation: object | None = None,
    scale: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: int = 5,
) -> np.ndarray:
    """Draw an annotation outline after the image is fitted to the canvas.

    Polygon LVIS/COCO segmentations are drawn as contours. Compressed RLE
    segmentations require pycocotools, so the bbox is used as a fallback.
    """
    drawn = False
    if isinstance(segmentation, list):
        for poly in segmentation:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = pts[:, 0] * scale + offset_x
            pts[:, 1] = pts[:, 1] * scale + offset_y
            pts_i = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_i], isClosed=True, color=color,
                          thickness=thickness, lineType=cv2.LINE_AA)
            drawn = True

    if not drawn and bbox:
        x, y, w, h = [float(v) for v in bbox]
        p1 = (int(round(x * scale + offset_x)),
              int(round(y * scale + offset_y)))
        p2 = (int(round((x + w) * scale + offset_x)),
              int(round((y + h) * scale + offset_y)))
        cv2.rectangle(canvas, p1, p2, color=color, thickness=thickness,
                      lineType=cv2.LINE_AA)

    return canvas
