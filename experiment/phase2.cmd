@echo off
REM Phase 2: stimulus presentation. Auto-picks the latest stimuli_*.json
REM unless --stimuli is given explicitly.
"C:\Users\thlab\.conda\envs\VIZ\python.exe" "%~dp0phase2_run.py" --config "%~dp0configs\default.yaml" %*

pause
