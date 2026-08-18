@echo off
REM ---------------------------------------------------------------------------
REM  Film profile generator -- entry points.
REM
REM  This file used to contain one line, "python film_sim.py Lady.png -p all",
REM  which renders an image and regenerates nothing. The regeneration sequence
REM  lived only as a loose command list in doc\README.md, and that list still
REM  ends with the DEPRECATED gen_film_names.py -- which, run after
REM  cpp_codegen.py, silently replaces the film_names.txt the effect panel
REM  loads. build.py is the ordered, gated sequence; it never runs the
REM  deprecated script.
REM
REM    run.cmd            regenerate + audit everything, then render the sample
REM    run.cmd build      regenerate + audit only
REM    run.cmd check      READ-ONLY audit: reports drift, writes nothing
REM    run.cmd render     render the sample image only (the old behaviour)
REM
REM  build.py exits non-zero on any failure, so this file stops on one.
REM ---------------------------------------------------------------------------
setlocal

if /I "%~1"=="check"  goto check
if /I "%~1"=="build"  goto build
if /I "%~1"=="render" goto render

:all
python build.py || exit /b 1
goto render

:build
python build.py || exit /b 1
goto :eof

:check
python build.py --check || exit /b 1
goto :eof

:render
python film_sim.py Lady.png -p all || exit /b 1
goto :eof
