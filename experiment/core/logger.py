from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.clock import PerfClock


@dataclass
class TimingEvent:
    name: str
    tick: float
    seconds: float
    payload: dict[str, Any] = field(default_factory=dict)


class EventLogger:
    """Append-only timing event log.

    Every event records both the raw QPC tick (`tick`, suitable for
    cross-referencing with the eye-tracker's QPC timestamps) and a relative
    seconds value (`seconds`, since clock origin) for human reading.
    """

    def __init__(self, clock: PerfClock) -> None:
        self.clock = clock
        self.events: list[TimingEvent] = []

    def log(self, name: str, **payload: Any) -> TimingEvent:
        tick = self.clock.now()
        event = TimingEvent(
            name=name,
            tick=tick,
            seconds=self.clock.seconds_since_origin(tick),
            payload=payload,
        )
        self.events.append(event)
        return event

    def save(self, path: Path, session_info: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {}
        if session_info:
            data.update(session_info)

        data["clock_source"] = "time.perf_counter (Windows QPC)"
        data["clock_origin_tick"] = self.clock.origin
        data["total_events"] = len(self.events)
        data["events"] = [
            {
                "name": e.name,
                "tick": e.tick,
                "seconds": e.seconds,
                "payload": e.payload,
            }
            for e in self.events
        ]

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if path.is_dir():
            path = path / f"session_{timestamp}.json"

        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
