# Tobii ET5 Eye Tracking — Stream Engine

Tobii Eye Tracker 5 (consumer) supports three APIs:

| API | Runtime needed | Works with ET5? | Notes |
|-----|----------------|-----------------|-------|
| **Stream Engine** | Tobii Experience | Yes | Low-level C, what we use |
| Game Integration | Game Hub | Yes (needs Game Hub) | Gaming-focused C++ |
| Pro SDK | Tobii services | **No** (ignores consumer ET5) | Research only |

We use **Stream Engine** (`tobii_stream_engine.dll`) — the lowest-level
API. Tobii Experience itself is built on it. It works with consumer ET5
out of the box, no extra licenses or runtime services needed.

## Setup

### Step 1 — Install Tobii Experience

This is the standard companion app for ET5. If your eye tracker works in
Tobii Experience (shows gaze dot), you're good.

### Step 2 — Get tobii_stream_engine.dll

Place `tobii_stream_engine.dll` (x64) in **this folder** (`eyetracking/tobii_bridge/`).

**From NuGet (easiest):**
1. Download `Tobii.StreamEngine.Native` from https://www.nuget.org/packages/Tobii.StreamEngine.Native
2. Rename `.nupkg` to `.zip` and extract.
3. Copy `build/native/lib/x64/tobii_stream_engine.dll` here.

Or one-liner in PowerShell:
```powershell
Invoke-WebRequest "https://www.nuget.org/api/v2/package/Tobii.StreamEngine.Native" -OutFile se.zip
Expand-Archive se.zip -DestinationPath se_extract
Copy-Item se_extract\build\native\lib\x64\tobii_stream_engine.dll .
Remove-Item se.zip, se_extract -Recurse
```

### Step 3 — Verify

```bat
C:\Users\thlab\.conda\envs\VIZ\python.exe debug_eyetracking.py ^
  --config configs\default.yaml ^
  --duration 10
```

## How it works

The Python wrapper (`eyetracking/tobii_et5.py`) calls Stream Engine's
flat C API via ctypes:

1. `tobii_api_create()` — create API context
2. `tobii_enumerate_local_device_urls()` — find connected ET5
3. `tobii_device_create()` — open device connection
4. `tobii_gaze_point_subscribe()` — register gaze callback
5. `tobii_head_pose_subscribe()` — register head pose callback
6. Polling thread: `tobii_wait_for_callbacks()` → `tobii_device_process_callbacks()`
7. Callbacks write CSV rows with QPC timestamps

## Legacy files

`tobii_bridge.cpp` and `build.bat` are the old C++ bridge approach (required
Game Integration SDK registration + MSVC compilation). Kept for reference.
`Tobii.GameIntegration.dll` can be removed — it requires Game Hub runtime.

## Time alignment

Every sample is stamped via `QueryPerformanceCounter`, the same counter
Python's `time.perf_counter()` uses on Windows.

## Output schema (CSV)

| column | meaning |
|---|---|
| tick_qpc_s | QPC timestamp in seconds (== Python `time.perf_counter()`) |
| kind | `gaze` or `mark` |
| gaze_x, gaze_y | normalized screen coords [0, 1] (Stream Engine convention) |
| gaze_origin_* | reserved |
| head_pitch/yaw/roll | head orientation degrees |
| head_x/y/z | head translation mm relative to display |
| validity | 1 = gaze, 2 = head pose |
| tag | non-empty for `mark` rows |
