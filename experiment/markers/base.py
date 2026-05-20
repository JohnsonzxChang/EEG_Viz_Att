from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


@runtime_checkable
class MarkerSender(Protocol):
    def send(self, code: int) -> None: ...
    def close(self) -> None: ...


class MarkerManager:
    """Fan-out marker dispatcher. Sends the same code to all registered senders.

    on_flip is called immediately after glfwSwapBuffers, so the wall-clock skew
    between marker arrival at the EEG amplifier and the actual photon emission
    is bounded by (display latency + UART latency) ≈ 1-2 ms.
    """

    def __init__(self) -> None:
        self._senders: list[MarkerSender] = []

    def add_sender(self, sender: MarkerSender) -> None:
        self._senders.append(sender)

    def on_flip(self, code: int) -> None:
        for sender in self._senders:
            try:
                sender.send(code)
            except Exception:
                log.warning("Marker sender %s failed for code %d",
                            type(sender).__name__, code, exc_info=True)

    def close(self) -> None:
        for sender in self._senders:
            try:
                sender.close()
            except Exception:
                log.warning("Failed to close marker sender %s",
                            type(sender).__name__, exc_info=True)
        self._senders.clear()
