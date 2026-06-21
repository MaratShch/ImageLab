@echo off
setlocal enableextensions

REM ============================================================================
REM  Qt 6.11 build for Windows 11 x64 - tuned for a simple Widgets desktop app
REM
REM  Goals encoded in this script:
REM    1. Smallest install possible (qtbase + qttools only, no examples/tests)
REM    2. Fewest possible DLLs LINKED by the app (Core, Gui, Widgets).
REM       Note: a few extra DLLs (Network, Sql, Xml) get BUILT because Qt's
REM       own tools (windeployqt, Designer, Assistant) need them, but they
REM       won't be SHIPPED with your app - windeployqt only copies what your
REM       .exe actually links to.
REM    3. AVX2 baseline - NO AVX-512 codegen even though host CPU has it
REM    4. Highest practical optimization (release + LTCG)
REM    5. Parallel build using configurable cores
REM
REM  Build machine assumptions:
REM    - 128 GB RAM (so 24 parallel cl.exe instances are safe)
REM    - MSVC 2022 Community, CMake 4.x, Ninja (from VS), Python 3.12
REM    - NO Perl required
REM
REM  Run from a plain cmd.exe (or the VS x64 Native Tools prompt - both work;
REM  the script calls vcvars64.bat itself, the second call is a no-op).
REM ============================================================================

REM ---------------------------------------------------------------------------
REM  1) EDIT THESE PATHS for your machine
REM ---------------------------------------------------------------------------
set "QT_SRC=C:\WORK\QT\6.11\Src"
set "QT_BUILD=C:\WORK\QT\6.11\build"
set "QT_INSTALL=C:\Qt\6.11.1"

set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Parallelism knobs:
set "PARALLEL_JOBS=10"
set "LTCG_THREADS=4"

REM ---------------------------------------------------------------------------
REM  2) MSVC x64 environment
REM ---------------------------------------------------------------------------
if not exist "%VS_VCVARS%" (
    echo [ERROR] vcvars64.bat not found at: %VS_VCVARS%
    exit /b 1
)
call "%VS_VCVARS%"

REM ---------------------------------------------------------------------------
REM  3) Verify tools
REM ---------------------------------------------------------------------------
for %%T in (cmake.exe ninja.exe python.exe cl.exe) do (
    where %%T >nul 2>&1 || (echo [ERROR] %%T not in PATH & exit /b 1)
)
echo --- Tools in use ---
where cmake
where ninja
where python
where cl
echo --- Parallelism ---
echo Compile jobs (cl.exe in parallel): %PARALLEL_JOBS%
echo LTCG link threads (per link.exe):  %LTCG_THREADS%
echo -------------------

REM ---------------------------------------------------------------------------
REM  4) CLEAN the build dir before reconfiguring
REM     This is important: a previous failed configure leaves a stale
REM     CMakeCache.txt that will poison the next attempt. We wipe and recreate.
REM ---------------------------------------------------------------------------
if exist "%QT_BUILD%" (
    echo Cleaning previous build dir: %QT_BUILD%
    rmdir /S /Q "%QT_BUILD%"
)
mkdir "%QT_BUILD%"
cd /d "%QT_BUILD%" || (echo [ERROR] cannot enter %QT_BUILD% & exit /b 1)

REM ---------------------------------------------------------------------------
REM  5) Configure
REM
REM  Features we DISABLE (safe - nothing inside Qt needs them):
REM    -no-feature-printsupport   no Qt6PrintSupport.dll
REM    -no-feature-testlib        no Qt6Test.dll
REM    -no-feature-concurrent     no Qt6Concurrent.dll
REM    -no-opengl                 no Qt6OpenGL.dll / Qt6OpenGLWidgets.dll
REM    -no-feature-avx512f        no AVX-512 codegen (cascades to all AVX-512 sub-features)
REM
REM  Features we KEEP (required by Qt's own tools):
REM    network -> needed by windeployqt (in qtbase) and Qt Designer
REM    sql     -> needed by Qt Assistant's Help module
REM    xml     -> needed by qdbusviewer and qtbase tools
REM    These DLLs WILL be built and live in the Qt install dir, but your
REM    app won't ship them unless it actually links to them.
REM
REM  /arch:AVX2  - hard-caps the MSVC compiler at AVX2 instruction set.
REM  /cgthreads  - parallelises link.exe's LTCG code generation step.
REM ---------------------------------------------------------------------------
call "%QT_SRC%\configure.bat" ^
    -prefix "%QT_INSTALL%" ^
    -release ^
    -ltcg ^
    -submodules qtbase,qttools ^
    -nomake examples ^
    -nomake tests ^
    -nomake benchmarks ^
    -no-feature-printsupport ^
    -no-feature-testlib ^
    -no-feature-concurrent ^
    -no-opengl ^
    -no-feature-avx512f ^
    -- ^
    -DCMAKE_C_FLAGS=/arch:AVX2 ^
    -DCMAKE_CXX_FLAGS=/arch:AVX2 ^
    -DCMAKE_EXE_LINKER_FLAGS=/cgthreads:%LTCG_THREADS% ^
    -DCMAKE_SHARED_LINKER_FLAGS=/cgthreads:%LTCG_THREADS% ^
    -DCMAKE_MODULE_LINKER_FLAGS=/cgthreads:%LTCG_THREADS%
if errorlevel 1 (
    echo [ERROR] configure failed.
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM  6) Build  (Ninja, parallel compile jobs)
REM ---------------------------------------------------------------------------
cmake --build . --parallel %PARALLEL_JOBS%
if errorlevel 1 (
    echo [ERROR] build failed.
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM  7) Install
REM ---------------------------------------------------------------------------
cmake --install .
if errorlevel 1 (
    echo [ERROR] install failed.
    exit /b 1
)

echo.
echo ============================================================================
echo   Qt 6.11 (minimal, AVX2, LTCG) installed at: %QT_INSTALL%
echo.
echo   DLLs you will actually SHIP with your Widgets app (via windeployqt):
echo       Qt6Core.dll  Qt6Gui.dll  Qt6Widgets.dll
echo       + platforms\qwindows.dll  + styles\qmodernwindowsstyle.dll
echo.
echo   DLLs that exist in the install but won't be shipped unless your app
echo   links to them:
echo       Qt6Network.dll  Qt6Sql.dll  Qt6Xml.dll
echo.
echo   Build your app with:
echo       cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH="%QT_INSTALL%"
echo       cmake --build build
echo ============================================================================
endlocal