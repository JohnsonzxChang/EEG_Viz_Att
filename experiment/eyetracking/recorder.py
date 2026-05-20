"""Gaze stream recorders.

Two on-disk backends with the same write API are exposed:

* ``CSVRecorder`` — legacy plain-text format. Wide adoption, easy to grep,
  but bloated (~70 bytes / sample) and slow at >100 Hz polling.

* ``HDF5Recorder`` — chunked + compressed compound datasets via h5py.
  Layout::

      /gaze    compound{tick_qpc_s:f8, x:f4, y:f4}
      /head    compound{tick_qpc_s:f8, pitch:f4, yaw:f4, roll:f4,
                        x:f4, y:f4, z:f4}
      /marks   compound{tick_qpc_s:f8, tag:S128}

  Each dataset is created with ``maxshape=(None,)`` and uses chunked storage
  so any session length is supported with bounded RAM. gzip + ``shuffle=True``
  yields ~5–10× smaller files than CSV. Per-stream ring buffers are flushed
  to disk on chunk fill and on a wall-clock timer to survive crashes.

Both backends are thread-safe; the Tobii poll loop calls ``write_gaze`` /
``write_head`` from a background thread while the main thread may call
``write_mark`` and ``flush`` concurrently.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ───────────────────────────── Base ─────────────────────────────


class GazeRecorderBase(ABC):
    """Common interface for gaze stream backends."""

    @abstractmethod
    def open(self, output_path: Path) -> Path:
        """Open the file at ``output_path`` for writing.

        Returns the actual path written (the backend may adjust the
        suffix to match its native format, e.g., ``.csv`` -> ``.h5``).
        """

    @abstractmethod
    def write_gaze(
        self,
        t_qpc_s: float,
        x: float,
        y: float,
        tobii_ts_us: int = 0,
    ) -> None: ...

    @abstractmethod
    def write_head(
        self,
        t_qpc_s: float,
        pitch: float,
        yaw: float,
        roll: float,
        x: float,
        y: float,
        z: float,
        tobii_ts_us: int = 0,
    ) -> None: ...

    @abstractmethod
    def write_mark(self, t_qpc_s: float, tag: str) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    def output_path(self) -> Path | None:
        return getattr(self, "_path", None)


# ───────────────────────────── CSV ─────────────────────────────


_CSV_HEADER = (
    "tick_qpc_s,kind,"
    "gaze_x,gaze_y,"
    "gaze_origin_x,gaze_origin_y,gaze_origin_z,"
    "head_pitch,head_yaw,head_roll,head_x,head_y,head_z,"
    "validity,tag,tobii_ts_us\n"
)


class CSVRecorder(GazeRecorderBase):
    """Plain-text CSV writer. Backward-compatible with prior pipelines."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._fh = None
        self._lock = threading.Lock()

    def open(self, output_path: Path) -> Path:
        path = Path(output_path)
        if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            path = path.with_suffix(".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")
        self._fh.write(_CSV_HEADER)
        self._path = path
        return path

    def write_gaze(
        self,
        t_qpc_s: float,
        x: float,
        y: float,
        tobii_ts_us: int = 0,
    ) -> None:
        line = (
            f"{t_qpc_s:.9f},gaze,{x:.6f},{y:.6f},,,,,,,,,,1,,{int(tobii_ts_us)}\n"
        )
        with self._lock:
            if self._fh is not None:
                self._fh.write(line)

    def write_head(
        self,
        t_qpc_s: float,
        pitch: float,
        yaw: float,
        roll: float,
        x: float,
        y: float,
        z: float,
        tobii_ts_us: int = 0,
    ) -> None:
        line = (
            f"{t_qpc_s:.9f},gaze,,,,,,"
            f"{pitch:.6f},{yaw:.6f},{roll:.6f},"
            f"{x:.6f},{y:.6f},{z:.6f},2,,{int(tobii_ts_us)}\n"
        )
        with self._lock:
            if self._fh is not None:
                self._fh.write(line)

    def write_mark(self, t_qpc_s: float, tag: str) -> None:
        safe = tag.replace(",", ";").replace("\n", ";").replace("\r", ";")
        line = f"{t_qpc_s:.9f},mark,,,,,,,,,,,,{safe},0\n"
        with self._lock:
            if self._fh is not None:
                self._fh.write(line)
                self._fh.flush()

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None


# ───────────────────────────── HDF5 ─────────────────────────────


# Compound dtypes — keep float32 for samples to halve disk usage; tick_qpc_s
# is float64 (~ns precision over hours).
#
# `tobii_ts_us` (int64) is the per-sample TimeStampMicroSeconds reported by
# Tobii itself — included so offline analyses can verify sample freshness
# (the poll loop already deduplicates by this value, but having it on disk
# lets downstream tooling cross-check sample uniqueness and detect any
# residual SDK echoing).
_GAZE_DTYPE = np.dtype([
    ("tick_qpc_s", "<f8"),
    ("x", "<f4"),
    ("y", "<f4"),
    ("tobii_ts_us", "<i8"),
])
_HEAD_DTYPE = np.dtype([
    ("tick_qpc_s", "<f8"),
    ("pitch", "<f4"),
    ("yaw", "<f4"),
    ("roll", "<f4"),
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("tobii_ts_us", "<i8"),
])
_MARK_DTYPE = np.dtype([
    ("tick_qpc_s", "<f8"),
    ("tag", "S128"),
])


class _BufferedDataset:
    """In-memory ring buffer that flushes into an extendable HDF5 dataset.

    `chunk_size` is both the on-disk HDF5 chunk shape and the in-memory
    flush threshold. Buffering amortises the per-call cost of resizing
    and writing to the dataset.
    """

    def __init__(self, ds, chunk_size: int) -> None:
        self.ds = ds
        self.chunk_size = max(1, int(chunk_size))
        self.buf = np.empty(self.chunk_size, dtype=ds.dtype)
        self.n = 0
        self.total_written = 0

    def append_record(self, rec: tuple) -> None:
        if self.n >= self.chunk_size:
            self.flush()
        self.buf[self.n] = rec
        self.n += 1

    def flush(self) -> None:
        if self.n == 0:
            return
        new_size = self.total_written + self.n
        self.ds.resize((new_size,))
        self.ds[self.total_written:new_size] = self.buf[: self.n]
        self.total_written = new_size
        self.n = 0


class HDF5Recorder(GazeRecorderBase):
    """Chunked, compressed HDF5 backend (h5py).

    Parameters
    ----------
    chunk_size:
        Rows per chunk for /gaze and /head. /marks uses ``max(64, chunk_size//8)``
        because marks are sparse.
    compression:
        ``"gzip"``, ``"lzf"``, or ``None``.
    compression_level:
        gzip level 1..9. Ignored for lzf/None.
    flush_every_s:
        Periodic wall-clock flush threshold; bounded data loss on crash.
    """

    def __init__(
        self,
        chunk_size: int = 2048,
        compression: str | None = "gzip",
        compression_level: int = 4,
        flush_every_s: float = 2.0,
    ) -> None:
        self._path: Path | None = None
        self._h5 = None
        self._lock = threading.Lock()
        self._chunk_size = int(chunk_size)
        self._compression = compression if compression else None
        self._compression_opts: int | None = None
        if self._compression == "gzip":
            self._compression_opts = int(compression_level)
        self._flush_every_s = float(flush_every_s)
        self._last_flush = 0.0

        self._gaze_buf: _BufferedDataset | None = None
        self._head_buf: _BufferedDataset | None = None
        self._mark_buf: _BufferedDataset | None = None

    # ── lifecycle ─────────────────────────────────────────────────

    def open(self, output_path: Path) -> Path:
        try:
            import h5py
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "HDF5Recorder requires h5py — `pip install h5py`"
            ) from e

        path = Path(output_path)
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            path = path.with_suffix(".h5")
        path.parent.mkdir(parents=True, exist_ok=True)

        self._h5 = h5py.File(path, "w", libver="latest")
        self._h5.attrs["format_version"] = "1.0"
        self._h5.attrs["clock_source"] = (
            "QueryPerformanceCounter (Windows QPC)"
        )
        self._h5.attrs["created_unix_s"] = time.time()
        self._h5.attrs["chunk_size"] = self._chunk_size
        if self._compression:
            self._h5.attrs["compression"] = self._compression

        gaze_ds = self._make_ds("gaze", _GAZE_DTYPE, self._chunk_size)
        head_ds = self._make_ds("head", _HEAD_DTYPE, self._chunk_size)
        mark_chunk = max(64, self._chunk_size // 8)
        mark_ds = self._make_ds("marks", _MARK_DTYPE, mark_chunk)

        self._gaze_buf = _BufferedDataset(gaze_ds, self._chunk_size)
        self._head_buf = _BufferedDataset(head_ds, self._chunk_size)
        self._mark_buf = _BufferedDataset(mark_ds, mark_chunk)

        self._path = path
        self._last_flush = time.perf_counter()
        # SWMR optional — skip for simplicity; reopen+SWMR is doable later.
        return path

    def _make_ds(self, name: str, dtype: np.dtype, chunk: int):
        kwargs = {
            "shape": (0,),
            "maxshape": (None,),
            "dtype": dtype,
            "chunks": (chunk,),
        }
        if self._compression:
            kwargs["compression"] = self._compression
            if self._compression_opts is not None:
                kwargs["compression_opts"] = self._compression_opts
            kwargs["shuffle"] = True
        return self._h5.create_dataset(name, **kwargs)

    # ── writes (called from poll thread) ──────────────────────────

    def write_gaze(
        self,
        t_qpc_s: float,
        x: float,
        y: float,
        tobii_ts_us: int = 0,
    ) -> None:
        with self._lock:
            if self._gaze_buf is None:
                return
            self._gaze_buf.append_record((t_qpc_s, x, y, int(tobii_ts_us)))
            self._maybe_flush_locked()

    def write_head(
        self,
        t_qpc_s: float,
        pitch: float,
        yaw: float,
        roll: float,
        x: float,
        y: float,
        z: float,
        tobii_ts_us: int = 0,
    ) -> None:
        with self._lock:
            if self._head_buf is None:
                return
            self._head_buf.append_record(
                (t_qpc_s, pitch, yaw, roll, x, y, z, int(tobii_ts_us)))
            self._maybe_flush_locked()

    def write_mark(self, t_qpc_s: float, tag: str) -> None:
        # Tags are sparse and useful for offline alignment — flush
        # immediately so a crash doesn't lose the most recent mark.
        with self._lock:
            if self._mark_buf is None:
                return
            tag_b = tag.encode("utf-8", errors="replace")[:128]
            self._mark_buf.append_record((t_qpc_s, tag_b))
            self._mark_buf.flush()
            self._gaze_buf.flush() if self._gaze_buf else None
            self._head_buf.flush() if self._head_buf else None
            try:
                self._h5.flush()
            except Exception:
                pass
            self._last_flush = time.perf_counter()

    def _maybe_flush_locked(self) -> None:
        now = time.perf_counter()
        if now - self._last_flush < self._flush_every_s:
            return
        if self._gaze_buf:
            self._gaze_buf.flush()
        if self._head_buf:
            self._head_buf.flush()
        try:
            self._h5.flush()
        except Exception:
            pass
        self._last_flush = now

    def flush(self) -> None:
        with self._lock:
            if self._gaze_buf:
                self._gaze_buf.flush()
            if self._head_buf:
                self._head_buf.flush()
            if self._mark_buf:
                self._mark_buf.flush()
            try:
                if self._h5 is not None:
                    self._h5.flush()
            except Exception:
                pass
            self._last_flush = time.perf_counter()

    def close(self) -> None:
        with self._lock:
            try:
                if self._gaze_buf:
                    self._gaze_buf.flush()
                if self._head_buf:
                    self._head_buf.flush()
                if self._mark_buf:
                    self._mark_buf.flush()
                if self._h5 is not None:
                    self._h5.attrs["closed_unix_s"] = time.time()
                    self._h5.close()
            except Exception:
                log.warning("HDF5 close failed", exc_info=True)
            finally:
                self._h5 = None
                self._gaze_buf = None
                self._head_buf = None
                self._mark_buf = None


# ─────────────────────────── Factory ───────────────────────────


def make_recorder(
    fmt: str,
    *,
    chunk_size: int = 2048,
    compression: str | None = "gzip",
    compression_level: int = 4,
    flush_every_s: float = 2.0,
) -> GazeRecorderBase:
    """Create a recorder by name.

    ``fmt`` is case-insensitive. Unknown names fall back to CSV with a
    warning — better to record *something* than to crash on a typo.
    """
    f = (fmt or "csv").strip().lower()
    if f in {"hdf5", "h5"}:
        return HDF5Recorder(
            chunk_size=chunk_size,
            compression=compression,
            compression_level=compression_level,
            flush_every_s=flush_every_s,
        )
    if f != "csv":
        log.warning("Unknown gaze recorder format %r — falling back to CSV", fmt)
    return CSVRecorder()
