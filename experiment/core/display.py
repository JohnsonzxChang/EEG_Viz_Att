from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import glfw
import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_LINE_LOOP,
    GL_LINE_SMOOTH,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGB,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLE_FAN,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glClear,
    glClearColor,
    glColor3f,
    glDeleteTextures,
    glDisable,
    glEnable,
    glEnd,
    glFinish,
    glGenTextures,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glVertex2f,
    glViewport,
)

from core.clock import CV2Clock, PerfClock

if TYPE_CHECKING:
    from core.logger import EventLogger
    from markers.base import MarkerManager
    from markers.photodiode import PhotodiodeMarker


@dataclass
class TextureFrame:
    pixels: np.ndarray
    width: int
    height: int


@dataclass
class OverlayState:
    """Per-frame overlay drawn via OpenGL primitives on top of the
    textured stimulus quad. Updated by paradigm code; read by the
    display loop. Coordinates are in OpenGL NDC matching the glOrtho
    set in :class:`DisplayEngine` ((-1, -1) bottom-left, (1, 1) top-right).

    Gaze coords from Tobii are normalized to the device's TrackRectangle,
    which on multi-monitor rigs may not match this stimulus window's
    NDC frame. Cursor coordinates outside [-1, 1] are therefore valid
    inputs (e.g., subject looked at the other monitor) — the cursor is
    rendered at the actual NDC location and naturally clipped by the
    viewport. Set ``cursor_clip_to_edge=True`` to instead pin an
    out-of-bounds cursor to the screen edge so it stays visible as a
    direction indicator.

    Setting any field to its disabling value (None for cursor, 0 for
    radii, False for warning_active) skips that primitive cheaply.
    """
    cursor_xy_norm: tuple[float, float] | None = None
    cursor_radius_px: int = 16
    cursor_color_rgb: tuple[int, int, int] = (0, 255, 0)
    # When True, clamp out-of-bounds cursor to a small margin from the
    # screen edge so it remains visible as a direction indicator.
    # When False (default), the cursor renders at the true NDC location
    # and is clipped by the OpenGL viewport — i.e. disappears when gaze
    # leaves the stimulus monitor. False is honest: it tells the operator
    # the subject is looking off-screen.
    cursor_clip_to_edge: bool = False
    # Color used when cursor would render off-screen — only takes effect
    # when cursor_clip_to_edge is True.
    cursor_offscreen_color_rgb: tuple[int, int, int] = (255, 200, 80)

    fixation_marker_radius_px: int = 0
    fixation_marker_color_rgb: tuple[int, int, int] = (255, 255, 255)

    warning_active: bool = False
    warning_ring_radius_px: int = 64
    warning_ring_color_rgb: tuple[int, int, int] = (255, 50, 50)
    warning_ring_thickness_px: int = 4


class DisplayEngine:
    """GLFW + OpenGL 2.1 stimulus display engine with precise timing.

    Adapted from EEG_Paradiam_Stim/core/display.py — kept API compatible so
    paradigms in this project share the same calling convention as that
    reference codebase.

    TTL toggle:
        set_ttl_enabled(False) suppresses calls to MarkerManager.on_flip
        from this engine. Photodiode and onset logging still happen so
        offline analysis retains a usable timestamp via the cv2_tick_qpc
        field on every stim_onset event.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fullscreen: bool = False,
        vsync: int = 1,
        warmup_frames: int = 10,
        window_name: str = "EEG Stimulus",
        monitor_index: int = 0,
    ) -> None:
        self.clock = PerfClock()
        try:
            self.cv2_clock: CV2Clock | None = CV2Clock()
        except ImportError:
            self.cv2_clock = None
        self._logger: "EventLogger | None" = None
        self._marker_mgr: "MarkerManager | None" = None
        self._photodiode: "PhotodiodeMarker | None" = None
        self._ttl_enabled: bool = True
        self._overlay = OverlayState()

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        monitor = None
        if fullscreen:
            monitors = glfw.get_monitors() or []
            if monitor_index < len(monitors):
                monitor = monitors[monitor_index]
            else:
                monitor = glfw.get_primary_monitor()

            native = glfw.get_video_mode(monitor)
            if width <= 0:
                width = native.size.width
            if height <= 0:
                height = native.size.height

            best_hz = native.refresh_rate
            for m in (glfw.get_video_modes(monitor) or []):
                if m.size.width == width and m.size.height == height:
                    best_hz = max(best_hz, m.refresh_rate)
            glfw.window_hint(glfw.REFRESH_RATE, best_hz)

        self.width = width
        self.height = height
        self._monitor = monitor
        self._window = glfw.create_window(width, height, window_name, monitor, None)
        if self._window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(vsync)
        glfw.set_key_callback(self._window, self._key_callback)

        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)

        self._texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        self._current_tex_size: tuple[int, int] | None = None

        self.refresh_hz = self._detect_refresh_rate()
        self.frame_interval_s = 1.0 / self.refresh_hz if self.refresh_hz > 0 else 1.0 / 60.0

        self._close_requested = False
        self._last_key: int | None = None

        blank = self.to_texture_frame(np.zeros((height, width, 3), dtype=np.uint8))
        for _ in range(max(1, warmup_frames)):
            self._present_internal(blank)

    # ── Setup ──────────────────────────────────────────────

    def set_logger(self, logger: "EventLogger") -> None:
        self._logger = logger

    def set_marker_manager(self, mgr: "MarkerManager") -> None:
        self._marker_mgr = mgr

    def set_photodiode(self, pd: "PhotodiodeMarker") -> None:
        self._photodiode = pd

    def set_ttl_enabled(self, enabled: bool) -> None:
        self._ttl_enabled = bool(enabled)

    # ── Overlay control ────────────────────────────────────
    # The overlay is a tiny set of OpenGL primitives drawn on top of the
    # textured stimulus quad each frame. It costs <0.05 ms/frame, does
    # NOT trigger texture re-uploads, and therefore does not perturb the
    # stimulus timing budget. Paradigms set it before calling present_for
    # and clear it afterwards.

    @property
    def overlay(self) -> OverlayState:
        return self._overlay

    def set_overlay(self, overlay: OverlayState) -> None:
        self._overlay = overlay

    def clear_overlay(self) -> None:
        self._overlay = OverlayState()

    def set_cursor(
        self,
        xy_norm: tuple[float, float] | None,
        *,
        radius_px: int | None = None,
        color_rgb: tuple[int, int, int] | None = None,
    ) -> None:
        self._overlay.cursor_xy_norm = xy_norm
        if radius_px is not None:
            self._overlay.cursor_radius_px = int(radius_px)
        if color_rgb is not None:
            self._overlay.cursor_color_rgb = tuple(color_rgb)  # type: ignore[assignment]

    def set_warning(self, active: bool) -> None:
        self._overlay.warning_active = bool(active)

    # ── Texture helpers ────────────────────────────────────

    def to_texture_frame(self, image: np.ndarray) -> TextureFrame:
        contiguous = np.ascontiguousarray(image)
        h, w = contiguous.shape[:2]
        return TextureFrame(pixels=contiguous, width=w, height=h)

    # ── Window state ───────────────────────────────────────

    def should_close(self) -> bool:
        return self._close_requested or glfw.window_should_close(self._window)

    def _key_callback(self, window: Any, key: int, scancode: int, action: int, mods: int) -> None:
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self._close_requested = True
        if action == glfw.PRESS:
            self._last_key = key

    def wait_for_key(self, frame: TextureFrame,
                     allowed_keys: set[int] | None = None) -> bool:
        self._last_key = None
        self._present_internal(frame)
        while not self.should_close():
            if self._last_key is not None:
                if allowed_keys is None or self._last_key in allowed_keys:
                    return True
                self._last_key = None
            glfw.wait_events_timeout(0.05)
        return not self.should_close()

    def wait_for_enter(self, frame: TextureFrame) -> bool:
        return self.wait_for_key(
            frame,
            allowed_keys={glfw.KEY_ENTER, glfw.KEY_KP_ENTER},
        )

    # ── Refresh rate detection ─────────────────────────────

    def _detect_refresh_rate(self) -> float:
        monitor = self._monitor or glfw.get_primary_monitor()
        if monitor is None:
            return 0.0
        mode = glfw.get_video_mode(monitor)
        return float(mode.refresh_rate) if mode else 0.0

    # ── Core rendering ─────────────────────────────────────

    def _upload_and_draw(self, frame: TextureFrame) -> None:
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        if self._current_tex_size != (frame.width, frame.height):
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB,
                frame.width, frame.height, 0,
                GL_RGB, GL_UNSIGNED_BYTE, frame.pixels,
            )
            self._current_tex_size = (frame.width, frame.height)
        else:
            glTexSubImage2D(
                GL_TEXTURE_2D, 0, 0, 0,
                frame.width, frame.height,
                GL_RGB, GL_UNSIGNED_BYTE, frame.pixels,
            )

        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, -1.0)
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, 1.0)
        glEnd()

    # ── OpenGL primitive overlays ───────────────────────────────
    # All radii are given in PIXELS but the projection matrix is
    # glOrtho(-1, 1, -1, 1) so we convert to NDC: rx = px / (W/2),
    # ry = px / (H/2). Aspect-correct circles use both (so a "circle"
    # is a proper visual circle, not a screen-stretched ellipse).

    def _gl_filled_circle(
        self,
        cx: float, cy: float,
        radius_px: int,
        color_rgb: tuple[int, int, int],
        segments: int = 48,
    ) -> None:
        rx = radius_px / max(1.0, self.width / 2.0)
        ry = radius_px / max(1.0, self.height / 2.0)
        r, g, b = color_rgb
        glColor3f(r / 255.0, g / 255.0, b / 255.0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for i in range(segments + 1):
            theta = 2.0 * math.pi * i / segments
            glVertex2f(cx + rx * math.cos(theta), cy + ry * math.sin(theta))
        glEnd()
        glColor3f(1.0, 1.0, 1.0)  # restore

    def _gl_ring(
        self,
        cx: float, cy: float,
        radius_px: int,
        color_rgb: tuple[int, int, int],
        thickness_px: int = 4,
        segments: int = 64,
    ) -> None:
        rx = radius_px / max(1.0, self.width / 2.0)
        ry = radius_px / max(1.0, self.height / 2.0)
        r, g, b = color_rgb
        glColor3f(r / 255.0, g / 255.0, b / 255.0)
        glLineWidth(float(max(1, thickness_px)))
        try:
            glEnable(GL_LINE_SMOOTH)
        except Exception:
            pass
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            glVertex2f(cx + rx * math.cos(theta), cy + ry * math.sin(theta))
        glEnd()
        glLineWidth(1.0)
        glColor3f(1.0, 1.0, 1.0)

    def _draw_overlay(self) -> None:
        ov = self._overlay
        cursor_set = ov.cursor_xy_norm is not None
        if (not cursor_set
                and not ov.warning_active
                and ov.fixation_marker_radius_px <= 0):
            return

        glDisable(GL_TEXTURE_2D)
        try:
            if ov.fixation_marker_radius_px > 0:
                self._gl_filled_circle(
                    0.0, 0.0,
                    ov.fixation_marker_radius_px,
                    ov.fixation_marker_color_rgb,
                )
            if ov.warning_active and ov.warning_ring_radius_px > 0:
                self._gl_ring(
                    0.0, 0.0,
                    ov.warning_ring_radius_px,
                    ov.warning_ring_color_rgb,
                    ov.warning_ring_thickness_px,
                )
            if cursor_set:
                cx_raw, cy_raw = ov.cursor_xy_norm  # type: ignore[misc]
                cx = float(cx_raw)
                cy = float(cy_raw)
                # On dual-monitor rigs Tobii's gaze normalisation reflects
                # its TrackRectangle, not this stimulus window — so values
                # outside [-1, 1] are routine when the subject looks at
                # the other screen. Two presentation choices:
                #   - clip_to_edge=False (default): pass through; OpenGL
                #     clips by viewport, so off-screen gaze produces no
                #     visible cursor (honest "look back at me" cue).
                #   - clip_to_edge=True : pin to screen edge in a contrasting
                #     color so the cursor always stays visible as a
                #     direction marker.
                offscreen = (abs(cx) > 1.0) or (abs(cy) > 1.0)
                if offscreen and ov.cursor_clip_to_edge:
                    cx = max(-0.97, min(0.97, cx))
                    cy = max(-0.97, min(0.97, cy))
                    color = ov.cursor_offscreen_color_rgb
                else:
                    color = ov.cursor_color_rgb
                self._gl_filled_circle(
                    cx, cy,
                    ov.cursor_radius_px,
                    color,
                )
        finally:
            glEnable(GL_TEXTURE_2D)

    def _present_internal(self, frame: TextureFrame, marker_code: int | None = None) -> bool:
        if self.should_close():
            return False

        self._upload_and_draw(frame)
        self._draw_overlay()
        glFinish()
        glfw.swap_buffers(self._window)

        if marker_code is not None and self._marker_mgr is not None and self._ttl_enabled:
            self._marker_mgr.on_flip(marker_code)

        glfw.poll_events()
        return not self.should_close()

    # ── Public presentation API ────────────────────────────

    def present(self, frame: TextureFrame, marker_code: int | None = None) -> bool:
        return self._present_internal(frame, marker_code)

    def present_for(
        self,
        frame: TextureFrame,
        duration_ms: int,
        marker_code: int | None = None,
        on_frame: Callable[[float], None] | None = None,
    ) -> bool:
        """Present `frame` for `duration_ms`. If `on_frame` is given it is
        invoked with elapsed seconds *before* every flip after the first;
        paradigms use this to update :class:`OverlayState` (cursor position,
        warning ring) without re-uploading textures."""
        duration_s = duration_ms / 1000.0
        if not self._present_internal(frame, marker_code):
            return False
        onset = self.clock.now()
        while self.clock.now() - onset < duration_s:
            if on_frame is not None:
                try:
                    on_frame(self.clock.now() - onset)
                except Exception:
                    # A buggy callback must never crash a running session.
                    pass
            if not self._present_internal(frame):
                return False
        return True

    def present_until(
        self,
        frame: TextureFrame,
        stop_predicate: Callable[[float], bool],
        on_frame: Callable[[float], None] | None = None,
        max_duration_s: float = 600.0,
        marker_code: int | None = None,
    ) -> bool:
        """Present `frame` until ``stop_predicate(elapsed_s)`` returns True
        or ``max_duration_s`` is hit (10-min safety cap by default).

        Used by the gaze-loss pause loop: paradigm presents the Chinese
        "请保持注视" screen and resumes only after the tracker has
        delivered continuous valid samples for the configured window.

        Returns True on natural stop, False if the user closed the window."""
        if not self._present_internal(frame, marker_code):
            return False
        onset = self.clock.now()
        while True:
            elapsed = self.clock.now() - onset
            try:
                if stop_predicate(elapsed):
                    return True
            except Exception:
                # A buggy predicate must not strand the experiment forever.
                return True
            if elapsed > max_duration_s:
                return True
            if on_frame is not None:
                try:
                    on_frame(elapsed)
                except Exception:
                    pass
            if not self._present_internal(frame):
                return False

    def present_for_logged(
        self,
        frame: TextureFrame,
        duration_ms: int,
        marker_code: int | None = None,
        event_name: str = "stimulus",
        on_frame: Callable[[float], None] | None = None,
        should_abort: Callable[[float], bool] | None = None,
        **payload: Any,
    ) -> bool:
        """Same contract as :meth:`present_for_logged` of yore, plus two
        per-frame callbacks:

        * ``on_frame(elapsed_s)`` — invoked once per refresh after the
          onset. Intended for *passive* checks (read latest gaze, flip a
          flag). Do not block — it runs inside the present loop.

        * ``should_abort(elapsed_s)`` — invoked once per refresh; if it
          returns True, the loop exits early (offset event still logged
          with ``aborted=True``). Used by paradigms to interrupt a
          stimulus when the eye-tracker reports the subject is no longer
          fixating (so the EEG epoch is saved as-is and the experiment
          can pause for gaze recovery before the next trial starts).
        """
        duration_s = duration_ms / 1000.0
        if self.should_close():
            return False
        self._upload_and_draw(frame)
        self._draw_overlay()
        glFinish()
        glfw.swap_buffers(self._window)
        onset_tick = self.clock.now()
        # cv2.getTickCount() reading at the SAME flip — used as the
        # authoritative ONSET timestamp when TTL is disabled (or as a
        # second timestamp source for QPC cross-check when TTL is on).
        onset_cv2 = self.cv2_clock.now() if self.cv2_clock is not None else None

        if marker_code is not None and self._marker_mgr is not None and self._ttl_enabled:
            self._marker_mgr.on_flip(marker_code)

        if self._logger is not None:
            self._logger.log(
                f"{event_name}_onset",
                onset_tick=onset_tick,
                cv2_tick_qpc=onset_cv2,
                marker_code=marker_code,
                ttl_enabled=self._ttl_enabled,
                **payload,
            )

        glfw.poll_events()
        if self.should_close():
            return False

        aborted = False
        while self.clock.now() - onset_tick < duration_s:
            elapsed = self.clock.now() - onset_tick
            if on_frame is not None:
                try:
                    on_frame(elapsed)
                except Exception:
                    pass
            if should_abort is not None:
                try:
                    if should_abort(elapsed):
                        aborted = True
                        break
                except Exception:
                    pass
            if not self._present_internal(frame):
                return False

        if self._logger is not None:
            offset_tick = self.clock.now()
            offset_cv2 = self.cv2_clock.now() if self.cv2_clock is not None else None
            self._logger.log(
                f"{event_name}_offset",
                offset_tick=offset_tick,
                cv2_tick_qpc=offset_cv2,
                duration_actual_s=offset_tick - onset_tick,
                aborted=aborted,
                **payload,
            )
        return True

    # ── Photodiode integration ─────────────────────────────

    def stamp_photodiode(self, image: np.ndarray, active: bool) -> np.ndarray:
        if self._photodiode is not None:
            return self._photodiode.stamp(image, active)
        return image

    # ── Cleanup ────────────────────────────────────────────

    def close(self) -> None:
        try:
            glDeleteTextures(int(self._texture_id))
        except Exception:
            pass
        try:
            glfw.destroy_window(self._window)
        except Exception:
            pass
        glfw.terminate()
