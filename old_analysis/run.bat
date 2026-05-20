@echo off
REM ==========================================================
REM  RSVP-COCO Multi-Label EEG Classification - Runner
REM  Usage:  run.bat [task] [extra_args]
REM
REM  Tasks:
REM    classify  - BCEWithLogitsLoss multi-label (main_pure_claM.py)
REM    circle    - ASL + Circle Loss (main_circle.py)
REM    retrieval - EEG-image contrastive retrieval (main_retrieval.py)
REM    enhanced  - Circle + Proxy-Anchor + Memory Queue (main_enhanced.py)
REM    fused     - CLIP-fused ERP classification (main_clip_fused.py)
REM    rsvp_v3   - Mini-ERP + CLIP prototypes (main_rsvp_clip_v3.py)
REM
REM  DATA_ROOT should point to the directory containing data/
REM    e.g.  set DATA_ROOT=C:\Users\thlab\Desktop\ES_coco
REM ==========================================================

REM -- Set code page to UTF-8
chcp 65001 >nul 2>nul

REM -- Remember script directory
set "SCRIPT_DIR=%~dp0"

REM -- Activate conda
call C:\ProgramData\miniconda3\Scripts\activate.bat >nul 2>nul
call conda activate VIZ >nul 2>nul

REM -- Set DATA_ROOT to ES_coco root (two levels up from rsvp_coco)
if not defined DATA_ROOT (
    for %%I in ("%SCRIPT_DIR%\..\..\") do set "DATA_ROOT=%%~fI"
)

REM -- cd into script directory so Python relative imports work
cd /d "%SCRIPT_DIR%"

set TASK=%1
if "%TASK%"=="" set TASK=classify
shift

echo ===== RSVP-COCO [%TASK%] %date% %time% =====
echo DATA_ROOT=%DATA_ROOT%

if "%TASK%"=="classify"  python -u "%SCRIPT_DIR%main_pure_claM.py" %1 %2 %3 %4 %5 %6 %7 %8 %9
if "%TASK%"=="circle"    python -u "%SCRIPT_DIR%main_circle.py" %1 %2 %3 %4 %5 %6 %7 %8 %9
if "%TASK%"=="retrieval" python -u "%SCRIPT_DIR%main_retrieval.py" %1 %2 %3 %4 %5 %6 %7 %8 %9
if "%TASK%"=="enhanced"  python -u "%SCRIPT_DIR%main_enhanced.py" %1 %2 %3 %4 %5 %6 %7 %8 %9
if "%TASK%"=="fused"     python -u "%SCRIPT_DIR%main_clip_fused.py" %1 %2 %3 %4 %5 %6 %7 %8 %9
if "%TASK%"=="rsvp_v3"   python -u "%SCRIPT_DIR%main_rsvp_clip_v3.py" %1 %2 %3 %4 %5 %6 %7 %8 %9

echo ===== Done %date% %time% =====
pause
