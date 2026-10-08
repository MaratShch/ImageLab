"""engine_env.py -- where the C++ engines find the owner's FFT library.

Since 2026-10-06 the scalar and AVX2 engines apply film_sim's frequency-domain
transfers exactly, on the owner's FFT library (namespace FourierTransform; the
optimised copy is delivered beside the six archives, the original stays in
C:\\WORK\\PYTHON.TST\\FFT untouched). The engine includes its headers
(fft_real2d.hpp, fft_real2d_avx2.hpp, fft_plan.hpp, fft_lane_*.hpp) and needs
nothing else from it.

Every tool here that compiles engine sources imports this module first. It puts
the library's include directory on CPLUS_INCLUDE_PATH (g++ and clang++ read it
for every compile they run, including the subprocesses the audits start), so no
compile command anywhere has to be edited and none can forget the path.

Lookup order, first hit wins:
    1. $FILMSIM_FFT_ROOT                      (the library root, holding include/)
    2. <FILMSIM_ROOT>/../FFT  and  <this dir>/../FFT
    3. /root/work/fftopt/FFT                 (the build machine)
A missing library is reported once on stderr; the compile that needs it then
fails with the compiler's own "fft_real2d.hpp: No such file" -- never silently.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKER = "fft_real2d.hpp"


def fft_include_dir() -> Path | None:
    cands = []
    if os.environ.get("FILMSIM_FFT_ROOT"):
        cands.append(Path(os.environ["FILMSIM_FFT_ROOT"]))
    if os.environ.get("FILMSIM_ROOT"):
        cands.append(Path(os.environ["FILMSIM_ROOT"]).parent / "FFT")
    cands.append(HERE.parent / "FFT")
    cands.append(Path("/root/work/fftopt/FFT"))
    for c in cands:
        inc = c / "include"
        if (inc / MARKER).is_file():
            return inc.resolve()
    return None


def ensure_fft_include() -> Path | None:
    inc = fft_include_dir()
    if inc is None:
        if not os.environ.get("_FILMSIM_FFT_WARNED"):
            os.environ["_FILMSIM_FFT_WARNED"] = "1"
            print("[engine_env] owner FFT library not found (set FILMSIM_FFT_ROOT "
                  "to the folder holding include/%s)" % MARKER, file=sys.stderr)
        return None
    cur = os.environ.get("CPLUS_INCLUDE_PATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    if str(inc) not in parts:
        os.environ["CPLUS_INCLUDE_PATH"] = os.pathsep.join([str(inc)] + parts)
    return inc


FFT_INCLUDE = ensure_fft_include()
