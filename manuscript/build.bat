@echo off
REM ==================================================================
REM build.bat -- compile main.tex with MiKTeX (pdflatex + bibtex)
REM Usage:   open a CMD in C:\Users\thlab\Documents\Claude\Projects\EEG_Viz_Att\manuscript and run "build.bat"
REM Requires:  C:\Program Files\MiKTeX\miktex\bin\x64 on PATH
REM            (otherwise this script prepends it for the current call).
REM ==================================================================

setlocal
set "MIKTEX_BIN=C:\Program Files\MiKTeX\miktex\bin\x64"
set "PATH=%MIKTEX_BIN%;%PATH%"

echo === [1/4] pdflatex pass 1 ===
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
if errorlevel 1 goto :err

echo === [2/4] bibtex ===
pushd build
bibtex main
popd
if errorlevel 1 goto :err

echo === [3/4] pdflatex pass 2 ===
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
if errorlevel 1 goto :err

echo === [4/4] pdflatex pass 3 ===
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
if errorlevel 1 goto :err

echo.
echo ==================================================================
echo  SUCCESS.  PDF is at:  build\main.pdf
echo ==================================================================
goto :eof

:err
echo.
echo ==================================================================
echo  BUILD FAILED -- check build\main.log for the first error.
echo ==================================================================
exit /b 1
