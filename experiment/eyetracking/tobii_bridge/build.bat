@echo off
REM Build tobii_bridge.dll using MSVC.
REM
REM Prerequisites:
REM   1. Install Tobii Game Hub from https://gaming.tobii.com/getstarted/
REM   2. Download Tobii Game Integration SDK (C++) and unzip somewhere.
REM      Set TOBII_GAMEHUB_SDK to its root folder (containing Include/ and Lib/).
REM   3. Open "x64 Native Tools Command Prompt for VS 2022" (so cl.exe is on PATH).
REM   4. cd into this folder and run: build.bat
REM
REM On success you'll get tobii_bridge.dll in this folder.

if "%TOBII_GAMEHUB_SDK%"=="" (
    echo TOBII_GAMEHUB_SDK is not set. Set it to the SDK root.
    echo e.g. set TOBII_GAMEHUB_SDK=C:\Tobii\GameIntegrationSDK
    exit /b 1
)

cl /LD /O2 /EHsc /std:c++17 ^
   /I"%TOBII_GAMEHUB_SDK%\Include" ^
   tobii_bridge.cpp ^
   "%TOBII_GAMEHUB_SDK%\Lib\x64\tobii_gameintegration_x64.lib" ^
   /Fe:tobii_bridge.dll

if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo Build OK.
echo Copy tobii_gameintegration_x64.dll alongside tobii_bridge.dll
echo (it is in %%TOBII_GAMEHUB_SDK%%\Lib\x64\).
