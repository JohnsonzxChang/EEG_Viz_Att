@echo off
REM Phase 1: stimulus selection (writes JSON to stimuli_select/)
"C:\Users\thlab\.conda\envs\VIZ\python.exe" "%~dp0phase1_select.py" --config "%~dp0configs\default.yaml" %*

pause
