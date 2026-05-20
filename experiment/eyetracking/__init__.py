"""Eye-tracking integration. Tobii Eye Tracker 5 via Game Integration SDK.

Uses tobii_gameintegration_x64.dll (official TGI SDK v9.0.4) directly from
Python ctypes via C++ vtable calls. Works with ET5 + Tobii Experience.
No C bridge compilation, no Game Hub, no Pro SDK needed.

Time alignment: every gaze sample is stamped with QueryPerformanceCounter,
the same counter Python's time.perf_counter exposes. So tick_qpc_s in the
CSV is directly comparable to the EventLogger's `tick` field.
"""

from eyetracking.base import EyeTracker, NullEyeTracker

__all__ = ["EyeTracker", "NullEyeTracker"]
