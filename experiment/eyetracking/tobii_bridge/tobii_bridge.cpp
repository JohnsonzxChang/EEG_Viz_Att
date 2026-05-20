// tobii_bridge.cpp — Thin C wrapper around the Tobii Game Integration C++ API.
//
// Compiles to tobii_bridge.dll for ctypes consumption from Python.
//
// Build (Windows, with MSVC's `cl.exe` on PATH; tested with VS 2022 x64):
//
//   cl /LD /O2 /EHsc /std:c++17 ^
//      /I"%TOBII_GAMEHUB_SDK%\Include" ^
//      tobii_bridge.cpp ^
//      "%TOBII_GAMEHUB_SDK%\Lib\x64\tobii_gameintegration_x64.lib" ^
//      /Fe:tobii_bridge.dll
//
// Then copy tobii_gameintegration_x64.dll alongside tobii_bridge.dll.
//
// Public symbols (extern "C"):
//   int    tb_init();
//   int    tb_start(const char* output_csv);
//   int    tb_mark(const char* tag);
//   int    tb_stop();
//   void   tb_shutdown();
//   double tb_qpc_seconds();
//
// Time alignment: every gaze sample is stamped with QueryPerformanceCounter,
// so it is directly comparable with Python's time.perf_counter (which uses
// the same QPC source on Windows).

#include <windows.h>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <string>

#include "tobii_gameintegration.h"   // from Tobii Game Hub SDK

using namespace TobiiGameIntegration;

// ──── globals ──────────────────────────────────────────────────────────
static ITobiiGameIntegrationApi*      g_api      = nullptr;
static IStreamsProvider*              g_streams  = nullptr;
static FILE*                          g_csv      = nullptr;
static std::atomic<bool>              g_running{false};
static std::thread                    g_worker;
static std::mutex                     g_io_mtx;

static LARGE_INTEGER                  g_qpc_freq{};

// ──── helpers ──────────────────────────────────────────────────────────

static double qpc_seconds() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return double(c.QuadPart) / double(g_qpc_freq.QuadPart);
}

static void csv_write_header() {
    std::lock_guard<std::mutex> lk(g_io_mtx);
    fprintf(g_csv,
            "tick_qpc_s,kind,"
            "gaze_x,gaze_y,"
            "gaze_origin_x,gaze_origin_y,gaze_origin_z,"
            "head_pitch,head_yaw,head_roll,head_x,head_y,head_z,"
            "validity,tag\n");
    fflush(g_csv);
}

static void csv_write_gaze(double t,
                           float gx, float gy,
                           float gox, float goy, float goz,
                           float hp, float hy, float hr,
                           float hx, float hy_, float hz,
                           int validity) {
    std::lock_guard<std::mutex> lk(g_io_mtx);
    fprintf(g_csv,
            "%.9f,gaze,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,\n",
            t, gx, gy, gox, goy, goz, hp, hy, hr, hx, hy_, hz, validity);
}

static void csv_write_mark(double t, const char* tag) {
    std::lock_guard<std::mutex> lk(g_io_mtx);
    // tag may contain commas; use trivial escaping by replacing them with ';'
    std::string s = tag ? tag : "";
    for (auto& c : s) if (c == ',' || c == '\n' || c == '\r') c = ';';
    fprintf(g_csv,
            "%.9f,mark,,,,,,,,,,,,,%s\n",
            t, s.c_str());
    fflush(g_csv);
}

// ──── polling worker ───────────────────────────────────────────────────
static void poll_loop() {
    // Tobii Game Integration delivers gaze at ~60 Hz on ET5.
    // We tick every 5 ms to capture all samples without missing.
    while (g_running.load(std::memory_order_acquire)) {
        if (g_api) {
            g_api->Update();
        }
        if (g_streams) {
            // Pull all queued gaze points
            GazePoint gp;
            while (g_streams->GetLatestGazePoint(gp)) {
                double t = qpc_seconds();
                // GazePoint X/Y in normalized [-1, 1] screen coords
                csv_write_gaze(t,
                               gp.X, gp.Y,
                               0.f, 0.f, 0.f,
                               0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                               1);
            }
            // Pull head pose
            HeadPose hp;
            if (g_streams->GetLatestHeadPose(hp)) {
                double t = qpc_seconds();
                csv_write_gaze(t,
                               0.f, 0.f,
                               0.f, 0.f, 0.f,
                               hp.Rotation.Pitch, hp.Rotation.Yaw, hp.Rotation.Roll,
                               hp.Position.X,    hp.Position.Y,    hp.Position.Z,
                               2);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// ──── exported C API ───────────────────────────────────────────────────

extern "C" __declspec(dllexport) int tb_init() {
    QueryPerformanceFrequency(&g_qpc_freq);
    if (g_api) return 0;  // already initialised
    g_api = GetApi("EEG_Viz_Att RSVP");
    if (!g_api) return 1;
    g_streams = g_api->GetStreamsProvider();
    if (!g_streams) return 2;
    return 0;
}

extern "C" __declspec(dllexport) int tb_start(const char* output_csv) {
    if (!g_api) return 1;
    if (g_running.load()) return 0;
    g_csv = nullptr;
    if (fopen_s(&g_csv, output_csv, "w") != 0 || !g_csv) return 3;
    csv_write_header();
    csv_write_mark(qpc_seconds(), "tb_start");
    g_running.store(true, std::memory_order_release);
    g_worker = std::thread(poll_loop);
    return 0;
}

extern "C" __declspec(dllexport) int tb_mark(const char* tag) {
    if (!g_csv) return 1;
    csv_write_mark(qpc_seconds(), tag);
    return 0;
}

extern "C" __declspec(dllexport) int tb_stop() {
    if (!g_running.load()) return 0;
    g_running.store(false, std::memory_order_release);
    if (g_worker.joinable()) g_worker.join();
    if (g_csv) {
        csv_write_mark(qpc_seconds(), "tb_stop");
        std::lock_guard<std::mutex> lk(g_io_mtx);
        fflush(g_csv);
        fclose(g_csv);
        g_csv = nullptr;
    }
    return 0;
}

extern "C" __declspec(dllexport) void tb_shutdown() {
    if (g_running.load()) tb_stop();
    g_streams = nullptr;
    g_api = nullptr;
    // Note: ITobiiGameIntegrationApi singleton is owned by the Game Hub SDK;
    // we don't delete it here. The runtime will release on process exit.
}

extern "C" __declspec(dllexport) double tb_qpc_seconds() {
    return qpc_seconds();
}
