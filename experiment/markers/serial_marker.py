from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    import serial as _serial
except ImportError:
    _serial = None  # type: ignore[assignment]


class SerialMarker:
    """1-byte UART marker sender (pyserial). Gracefully no-ops if pyserial is unavailable.

    code is masked to 1 byte (0..255). For BrainProducts/g.tec/NeuroScan with a
    USB-COM TTL bridge, this is the standard interface. Latency is dominated
    by the FTDI buffer (~0.3-1 ms with timeout=0.001).
    """

    def __init__(self, port: str = "COM3", baudrate: int = 115200) -> None:
        self.port = port
        self.baudrate = baudrate
        self._ser = None

        if _serial is None:
            log.warning("pyserial not installed. SerialMarker is a no-op. "
                        "Install with: pip install pyserial")
            return

        try:
            self._ser = _serial.Serial(port=port, baudrate=baudrate,
                                       timeout=0.001, write_timeout=0.001)
            log.info("Serial marker opened on %s @ %d baud", port, baudrate)
        except Exception:
            log.warning("Failed to open serial port %s. SerialMarker is a no-op.",
                        port, exc_info=True)
            self._ser = None

    def send(self, code: int) -> None:
        if self._ser is not None:
            self._ser.write(bytes([code & 0xFF]))
            self._ser.flush()

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
