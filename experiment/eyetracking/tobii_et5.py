"""Tobii Eye Tracker 5 — official Game Integration SDK via ctypes vtable calls.

Uses tobii_gameintegration_x64.dll (the official TGI SDK v9.0.4, same DLL
used by the eye_track_ssvep project). The SDK has a C++ class interface;
we call virtual methods through vtable pointer manipulation from ctypes.

No C bridge compilation needed. No Game Hub needed.
Just the DLL + Tobii Experience (the standard ET5 companion app).

Storage backends are pluggable via :mod:`eyetracking.recorder` — pick
``format="csv"`` for the legacy plain-text format or ``format="hdf5"``
for chunked + gzip-compressed HDF5 (recommended for long sessions).

Time alignment: every gaze sample is stamped with QueryPerformanceCounter,
the same counter Python's time.perf_counter exposes. So tick_qpc_s in the
output file is directly comparable to the EventLogger's `tick` field.

Live gaze API: :meth:`get_latest_gaze` returns the most recent sample in
normalized [-1, 1] screen coordinates (Tobii native). It is non-blocking
and thread-safe — paradigms call it every display frame to drive the
real-time ISI cursor and onset-ON deviation monitoring.
"""

from __future__ import annotations

import ctypes as C
import logging
import os
import threading
import time
from pathlib import Path

from eyetracking.base import EyeTracker
from eyetracking.recorder import GazeRecorderBase, make_recorder

log = logging.getLogger(__name__)

# ── QPC helper (same clock as time.perf_counter on Windows) ──

_qpc_freq = C.c_longlong()
C.windll.kernel32.QueryPerformanceFrequency(C.byref(_qpc_freq))


def _qpc_seconds() -> float:
    counter = C.c_longlong()
    C.windll.kernel32.QueryPerformanceCounter(C.byref(counter))
    return counter.value / _qpc_freq.value


# ── C++ struct layouts (must match tobii_gameintegration.h) ──


class _GazePoint(C.Structure):
    """TobiiGameIntegration::GazePoint — 16 bytes."""
    _fields_ = [
        ("TimeStampMicroSeconds", C.c_int64),
        ("X", C.c_float),
        ("Y", C.c_float),
    ]


class _Rotation(C.Structure):
    """TobiiGameIntegration::Rotation — 12 bytes."""
    _fields_ = [
        ("YawDegrees", C.c_float),
        ("PitchDegrees", C.c_float),
        ("RollDegrees", C.c_float),
    ]


class _Position(C.Structure):
    """TobiiGameIntegration::Position — 12 bytes."""
    _fields_ = [
        ("X", C.c_float),
        ("Y", C.c_float),
        ("Z", C.c_float),
    ]


class _HeadPose(C.Structure):
    """TobiiGameIntegration::HeadPose : Transformation + timestamp.
    Transformation = { Rotation, Position }. Base members come first.
    """
    _fields_ = [
        ("Rotation", _Rotation),
        ("Position", _Position),
        ("TimeStampMicroSeconds", C.c_int64),
    ]


class _Rectangle(C.Structure):
    """TobiiGameIntegration::Rectangle — 16 bytes."""
    _fields_ = [
        ("Left", C.c_int32),
        ("Top", C.c_int32),
        ("Right", C.c_int32),
        ("Bottom", C.c_int32),
    ]


# ── C++ vtable helpers ──
# MSVC x64: unified calling convention. 'this' in RCX.
# No implicit virtual destructors in these structs (none declared).
# vtable[i] = i-th virtual method in declaration order.

PTR = C.c_void_p
_PTR_SZ = C.sizeof(PTR)  # 8 on x64


def _read_ptr(addr: int) -> int:
    """Read a pointer-sized value from memory."""
    return C.c_void_p.from_address(addr).value or 0


def _vfn(obj_ptr: int, index: int) -> int:
    """Get virtual function pointer at vtable[index] for a C++ object."""
    vtable = _read_ptr(obj_ptr)
    return _read_ptr(vtable + index * _PTR_SZ)


# ITobiiGameIntegrationApi vtable indices (from tobii_gameintegration.h)
_API_GET_TRACKER_CONTROLLER = 0
_API_GET_STREAMS_PROVIDER = 1
_API_IS_INITIALIZED = 5
_API_UPDATE = 6
_API_SHUTDOWN = 7

# IStreamsProvider vtable indices
_SP_GET_LATEST_HEAD_POSE = 1
_SP_GET_LATEST_GAZE_POINT = 3
_SP_IS_PRESENT = 6

# ITrackerController vtable indices
_TC_TRACK_RECTANGLE = 5
_TC_IS_CONNECTED = 8


# ── Monitor detection ──


def _enum_monitors() -> list[tuple[int, int, int, int]]:
    """Return list of (left, top, right, bottom) for each display monitor."""
    rects: list[tuple[int, int, int, int]] = []

    @C.WINFUNCTYPE(C.c_int, PTR, PTR, C.POINTER(_Rectangle), C.c_longlong)
    def _cb(_hmon, _hdc, lprect, _lparam):
        r = lprect.contents
        rects.append((r.Left, r.Top, r.Right, r.Bottom))
        return 1

    C.windll.user32.EnumDisplayMonitors(None, None, _cb, 0)
    return rects


def _get_screen_rect(monitor: int | None = None) -> tuple[int, int, int, int]:
    """Get screen rectangle in virtual-screen pixel coordinates.

    monitor=None  → full virtual desktop (all monitors combined)
    monitor=0     → primary monitor
    monitor=N     → Nth monitor (by enumeration order)

    NOTE: Do NOT call SetProcessDPIAware() here — changing the process DPI
    context after the Tobii SDK has initialised breaks its service link.
    """
    user32 = C.windll.user32
    rects = _enum_monitors()
    if rects:
        log.info("Monitors detected: %s", rects)

    if monitor is not None and rects and 0 <= monitor < len(rects):
        return rects[monitor]
    if monitor is not None:
        log.warning("Monitor %d not found (%d available), using virtual desktop",
                    monitor, len(rects))

    # Full virtual desktop
    x = user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
    y = user32.GetSystemMetrics(77)   # SM_YVIRTUALSCREEN
    w = user32.GetSystemMetrics(78)   # SM_CXVIRTUALSCREEN
    h = user32.GetSystemMetrics(79)   # SM_CYVIRTUALSCREEN
    if w > 0 and h > 0:
        return (x, y, x + w, y + h)

    # Fallback: primary monitor
    w = user32.GetSystemMetrics(0)    # SM_CXSCREEN
    h = user32.GetSystemMetrics(1)    # SM_CYSCREEN
    return (0, 0, w or 1920, h or 1080)


# ── DLL search ──

_DLL_NAME = "tobii_gameintegration_x64.dll"

_SEARCH_DIRS = [
    lambda: Path(__file__).parent / "tobii_bridge",
    lambda: (Path(__file__).parent.parent
             / "eye_track_ssvep" / "stims" / "bin" / "x64"),
]


def _find_dll(explicit: str | Path | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        if p.is_dir():
            candidate = p / _DLL_NAME
            if candidate.is_file():
                return candidate
        return p

    for factory in _SEARCH_DIRS:
        try:
            d = factory()
            if not d or not d.is_dir():
                continue
            candidate = d / _DLL_NAME
            if candidate.is_file():
                return candidate
        except Exception:
            continue

    return Path(__file__).parent / "tobii_bridge" / _DLL_NAME


# TGI version must match DLL header
_TGI_MAJOR, _TGI_MINOR, _TGI_REVISION = 9, 0, 4


class TobiiET5(EyeTracker):
    """Tobii ET5 via tobii_gameintegration_x64.dll (official C++ SDK).

    Calls C++ virtual methods through vtable manipulation.
    Same approach as GazeSample.cpp from the SDK, but in pure Python.
    """

    def __init__(
        self,
        dll_path: str | Path | None = None,
        monitor: int | None = None,
        *,
        recorder_format: str = "csv",
        hdf5_chunk_size: int = 2048,
        hdf5_compression: str | None = "gzip",
        hdf5_compression_level: int = 4,
        hdf5_flush_every_s: float = 2.0,
        stale_threshold_s: float = 0.08,
    ) -> None:
        self._dll: C.CDLL | None = None
        self._dll_path = _find_dll(dll_path)
        self._monitor = monitor
        self._screen_rect: tuple[int, int, int, int] = (0, 0, 1920, 1080)
        self._started = False
        self._init_rc: int | None = None
        self._last_start_rc: int | None = None
        self._load_error: str | None = None

        # Recorder (CSV or HDF5)
        self._recorder: GazeRecorderBase = make_recorder(
            recorder_format,
            chunk_size=hdf5_chunk_size,
            compression=hdf5_compression,
            compression_level=hdf5_compression_level,
            flush_every_s=hdf5_flush_every_s,
        )
        self._recorder_format = (recorder_format or "csv").lower()

        # C++ object pointers (integers)
        self._api_ptr: int = 0
        self._streams_ptr: int = 0
        self._tc_ptr: int = 0

        # Cached virtual function callables
        self._update = None
        self._get_gaze = None
        self._get_head = None
        self._is_connected = None
        self._is_initialized = None
        self._is_present = None
        self._api_shutdown = None
        self._track_rect = None

        # Polling state
        self._running = False
        self._worker: threading.Thread | None = None

        # Latest gaze cache. Read every frame by the paradigm.
        # Tuple layout:
        #   (x, y, sample_qpc_s, advance_qpc_s, tobii_ts_us)
        # where:
        #   sample_qpc_s   = local QPC at the moment we polled this sample
        #   advance_qpc_s  = local QPC of the most recent FRESH sample
        #                    (Tobii TS strictly advanced); used by
        #                    get_latest_gaze() to detect the SDK going
        #                    "stuck" (Tobii TGI keeps returning the last
        #                    valid GazePoint with the same TimeStampMicroSeconds
        #                    when eyes are not detected).
        #   tobii_ts_us    = the GazePoint.TimeStampMicroSeconds we last saw
        # None until the first valid sample arrives.
        self._latest_gaze: (
            tuple[float, float, float, float, int] | None
        ) = None
        self._gaze_lock = threading.Lock()
        # A gaze sample is considered LIVE only if Tobii's per-sample
        # TimeStampMicroSeconds advanced within the last
        # `_stale_threshold_s` seconds. ET5's native gaze rate is ~33-60
        # Hz so the longest legitimate gap between fresh samples is
        # ~30 ms; 80 ms tolerates ~2-3 missed polls before declaring
        # "no eyes". This is the fix for the "SDK keeps echoing the last
        # value when eyes disappear" failure mode.
        self._stale_threshold_s = float(stale_threshold_s)

        if not self._dll_path.exists():
            log.warning(
                "Tobii DLL not found at %s — eye-tracking will be a no-op.",
                self._dll_path,
            )
            return

        try:
            os.add_dll_directory(str(self._dll_path.parent))
        except (AttributeError, OSError):
            pass

        try:
            self._dll = C.CDLL(str(self._dll_path))
        except OSError as e:
            self._load_error = str(e)
            log.warning("Failed to load Tobii DLL: %s", e)
            self._dll = None
            return

        # ── Call GetApi (extern "C" export) ──
        self._dll.GetApi.restype = PTR
        self._dll.GetApi.argtypes = [
            C.c_char_p, C.c_int, C.c_int, C.c_int,
            C.c_void_p, C.c_uint, C.c_bool,
        ]

        self._api_ptr = self._dll.GetApi(
            b"EEG_Viz_Att RSVP",
            _TGI_MAJOR, _TGI_MINOR, _TGI_REVISION,
            None, 0, False,
        ) or 0

        if not self._api_ptr:
            self._init_rc = 1
            self._load_error = "GetApi() returned NULL"
            log.warning("%s", self._load_error)
            self._dll = None
            return

        # ── Build vtable bindings ──
        try:
            self._setup_vtable()
        except Exception as e:
            self._init_rc = 3
            self._load_error = f"vtable setup failed: {e}"
            log.warning("%s", self._load_error)
            self._dll = None
            return

        # Set tracking rectangle BEFORE the Update loop (matches SDK sample)
        if self._track_rect and self._tc_ptr:
            l, t, r, b = _get_screen_rect(self._monitor)
            self._screen_rect = (l, t, r, b)
            rect = _Rectangle(l, t, r, b)
            self._track_rect(self._tc_ptr, C.byref(rect))
            log.info("TrackRectangle set to (%d, %d, %d, %d)", l, t, r, b)

        # Pump updates to establish connection (3 s timeout)
        connected = False
        for _ in range(60):
            self._update(self._api_ptr)
            if self._is_connected and self._is_connected(self._tc_ptr):
                connected = True
                break
            time.sleep(0.05)

        if not connected:
            log.warning("Tobii ET5 not connected after 3 s — will keep trying "
                        "in poll loop")

        self._init_rc = 0
        log.info(
            "Tobii TGI initialized (connected=%s, rect=%s, recorder=%s)",
            connected, self._screen_rect, self._recorder_format,
        )

    def _setup_vtable(self) -> None:
        """Resolve C++ virtual functions via vtable pointers."""
        a = self._api_ptr

        # void Update(this)
        self._update = C.CFUNCTYPE(None, PTR)(
            _vfn(a, _API_UPDATE))

        # bool IsInitialized(this)
        self._is_initialized = C.CFUNCTYPE(C.c_bool, PTR)(
            _vfn(a, _API_IS_INITIALIZED))

        # void Shutdown(this)
        self._api_shutdown = C.CFUNCTYPE(None, PTR)(
            _vfn(a, _API_SHUTDOWN))

        # IStreamsProvider* GetStreamsProvider(this)
        get_sp = C.CFUNCTYPE(PTR, PTR)(_vfn(a, _API_GET_STREAMS_PROVIDER))
        self._streams_ptr = get_sp(a) or 0
        if not self._streams_ptr:
            raise RuntimeError("GetStreamsProvider() returned NULL")

        sp = self._streams_ptr

        # bool GetLatestGazePoint(this, GazePoint&)
        self._get_gaze = C.CFUNCTYPE(
            C.c_bool, PTR, C.POINTER(_GazePoint))(
            _vfn(sp, _SP_GET_LATEST_GAZE_POINT))

        # bool GetLatestHeadPose(this, HeadPose&)
        self._get_head = C.CFUNCTYPE(
            C.c_bool, PTR, C.POINTER(_HeadPose))(
            _vfn(sp, _SP_GET_LATEST_HEAD_POSE))

        # bool IsPresent(this)
        self._is_present = C.CFUNCTYPE(C.c_bool, PTR)(
            _vfn(sp, _SP_IS_PRESENT))

        # ITrackerController* GetTrackerController(this)
        get_tc = C.CFUNCTYPE(PTR, PTR)(
            _vfn(a, _API_GET_TRACKER_CONTROLLER))
        self._tc_ptr = get_tc(a) or 0

        if self._tc_ptr:
            tc = self._tc_ptr

            # bool IsConnected(this)
            self._is_connected = C.CFUNCTYPE(C.c_bool, PTR)(
                _vfn(tc, _TC_IS_CONNECTED))

            # bool TrackRectangle(this, const Rectangle&)
            self._track_rect = C.CFUNCTYPE(
                C.c_bool, PTR, C.POINTER(_Rectangle))(
                _vfn(tc, _TC_TRACK_RECTANGLE))

    # ── Properties ────────────────────────────────────────────────────

    @property
    def dll_path(self) -> Path:
        return self._dll_path

    @property
    def init_rc(self) -> int | None:
        return self._init_rc

    @property
    def last_start_rc(self) -> int | None:
        return self._last_start_rc

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def is_available(self) -> bool:
        return self._dll is not None and self._api_ptr != 0

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def output_path(self) -> Path | None:
        return self._recorder.output_path

    def diagnostics(self) -> dict[str, object]:
        diag: dict[str, object] = {
            "dll_path": str(self._dll_path),
            "dll_exists": self._dll_path.exists(),
            "dll_loaded": self._dll is not None,
            "init_rc": self._init_rc,
            "last_start_rc": self._last_start_rc,
            "load_error": self._load_error,
            "is_available": self.is_available,
            "is_started": self.is_started,
            "monitor": self._monitor,
            "screen_rect": list(self._screen_rect),
            "recorder_format": self._recorder_format,
            "output_path": str(self.output_path) if self.output_path else None,
        }
        if self.is_available:
            try:
                diag["is_initialized"] = bool(
                    self._is_initialized(self._api_ptr))
                if self._is_connected and self._tc_ptr:
                    diag["is_connected"] = bool(
                        self._is_connected(self._tc_ptr))
                if self._is_present and self._streams_ptr:
                    diag["is_present"] = bool(
                        self._is_present(self._streams_ptr))
            except Exception:
                pass
        return diag

    # ── Polling thread ────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        gp = _GazePoint()
        hp = _HeadPose()
        # Track the last Tobii TS we wrote so that "GetLatestGazePoint
        # / GetLatestHeadPose returned True with the same
        # TimeStampMicroSeconds" can be detected as a stale (echoed)
        # sample rather than a fresh one.
        # Both streams need this: offline blink detection distinguishes
        # blink from head-out by checking whether HEAD TS advanced
        # during a GAZE TS gap; if we wrote head every poll regardless,
        # head would always look "advancing" even when Tobii had
        # actually frozen it (subject left tracking volume).
        last_gaze_tobii_ts: int = -1
        last_head_tobii_ts: int = -1

        while self._running:
            try:
                self._update(self._api_ptr)
            except Exception:
                break

            t = _qpc_seconds()

            # Gaze — distinguish FRESH (TS advanced) from STALE (TS repeated)
            try:
                if self._get_gaze(self._streams_ptr, C.byref(gp)):
                    new_ts = int(gp.TimeStampMicroSeconds)
                    if new_ts != last_gaze_tobii_ts:
                        # Fresh sample — write to disk, update cache,
                        # mark advance time so consumers can compute
                        # staleness.
                        gx, gy = float(gp.X), float(gp.Y)
                        self._recorder.write_gaze(t, gx, gy, new_ts)
                        with self._gaze_lock:
                            self._latest_gaze = (gx, gy, t, t, new_ts)
                        last_gaze_tobii_ts = new_ts
                    # else: SDK echoed the previous sample — do NOT
                    # advance _latest_gaze. The unchanged advance_qpc_s
                    # will trip the staleness check in get_latest_gaze
                    # once the gap exceeds _stale_threshold_s.
            except Exception:
                pass

            # Head pose — same TS-dedup logic. /head sample density in
            # the file is a function of how many *fresh* head poses
            # Tobii actually delivered, not how often our loop polled.
            try:
                if self._get_head(self._streams_ptr, C.byref(hp)):
                    new_head_ts = int(hp.TimeStampMicroSeconds)
                    if new_head_ts != last_head_tobii_ts:
                        t2 = _qpc_seconds()
                        self._recorder.write_head(
                            t2,
                            hp.Rotation.PitchDegrees,
                            hp.Rotation.YawDegrees,
                            hp.Rotation.RollDegrees,
                            hp.Position.X, hp.Position.Y, hp.Position.Z,
                            new_head_ts,
                        )
                        last_head_tobii_ts = new_head_ts
            except Exception:
                pass

            time.sleep(1.0 / 120)

    # ── EyeTracker interface ──────────────────────────────────────────

    def start(self, output_path: Path) -> None:
        self._last_start_rc = None
        if not self.is_available:
            return
        try:
            actual_path = self._recorder.open(Path(output_path))
        except Exception as e:
            self._last_start_rc = 3
            log.warning(
                "Cannot open recorder (%s) at %s: %s",
                self._recorder_format, output_path, e,
            )
            return

        self._recorder.write_mark(_qpc_seconds(), "tb_start")
        self._recorder.flush()

        self._running = True
        self._worker = threading.Thread(target=self._poll_loop, daemon=True)
        self._worker.start()

        self._last_start_rc = 0
        self._started = True
        log.info(
            "Tobii ET5 recording (%s) -> %s",
            self._recorder_format, actual_path,
        )

    def mark(self, tag: str) -> None:
        if not self.is_available or not self._started:
            return
        try:
            self._recorder.write_mark(_qpc_seconds(), tag)
        except Exception:
            log.warning("mark(%r) failed", tag, exc_info=True)

    def stop(self) -> None:
        if not self.is_available or not self._started:
            return
        try:
            self._running = False
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=2.0)
            try:
                self._recorder.write_mark(_qpc_seconds(), "tb_stop")
            except Exception:
                pass
            try:
                self._recorder.close()
            except Exception:
                log.warning("recorder.close() failed", exc_info=True)
        finally:
            self._started = False

    def shutdown(self) -> None:
        """Release Game Integration resources."""
        if self._started:
            self.stop()
        if self._api_shutdown and self._api_ptr:
            try:
                self._api_shutdown(self._api_ptr)
            except Exception:
                pass
            self._api_ptr = 0

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def now_qpc_seconds(self) -> float:
        return _qpc_seconds()

    # ── Live gaze (consumed by paradigm per-frame) ────────────────────

    def get_latest_gaze(self) -> tuple[float, float] | None:
        """Return the most recent gaze sample (x, y) in Tobii native
        normalized coordinates, or None if no sample is yet available
        OR if the last fresh sample is older than `stale_threshold_s`.

        Staleness handling is critical because the Tobii TGI SDK keeps
        returning the LAST valid GazePoint (with its original
        TimeStampMicroSeconds) when the subject's eyes leave the
        tracking volume — `_get_gaze()` returns True but the values are
        frozen. The poll loop only updates `_latest_gaze` when Tobii's
        timestamp advances, so an outdated `advance_qpc_s` in this
        function reliably means "no new data → eyes lost".

        Non-blocking, thread-safe.
        """
        with self._gaze_lock:
            if self._latest_gaze is None:
                return None
            x, y, _sample_qpc, advance_qpc, _ts = self._latest_gaze
        if _qpc_seconds() - advance_qpc > self._stale_threshold_s:
            return None
        return (x, y)

    def wait_for_data(self, timeout_s: float = 2.5) -> bool:
        """Block until the first valid gaze sample arrives, or timeout.
        Returns True if a sample was observed within ``timeout_s``,
        False otherwise. ``False`` means the device is either not
        connected, lost service link, or has no eyes in view at startup
        — the caller (phase2_run.py) treats this as a fatal startup
        error and aborts before any stimulus is shown."""
        if not self.is_available:
            log.error(
                "Tobii ET5 unavailable (DLL load_error=%r init_rc=%s) — "
                "cannot verify data flow.",
                self._load_error, self._init_rc,
            )
            return False
        if not self._started:
            log.error(
                "wait_for_data() called before start() — "
                "no recording thread is running.",
            )
            return False
        deadline = time.perf_counter() + max(0.1, float(timeout_s))
        while time.perf_counter() < deadline:
            if self.get_latest_gaze() is not None:
                return True
            time.sleep(0.05)
        return False

    def get_latest_gaze_with_age(
        self, now_qpc_s: float | None = None,
    ) -> tuple[float, float, float] | None:
        """Like :meth:`get_latest_gaze` but also returns sample age in
        seconds (now_qpc - last_advance_qpc). Useful for diagnostics.
        Returns None on the same conditions as get_latest_gaze."""
        with self._gaze_lock:
            if self._latest_gaze is None:
                return None
            x, y, _sample_qpc, advance_qpc, _ts = self._latest_gaze
        now = now_qpc_s if now_qpc_s is not None else _qpc_seconds()
        age = now - advance_qpc
        if age > self._stale_threshold_s:
            return None
        return (x, y, age)
