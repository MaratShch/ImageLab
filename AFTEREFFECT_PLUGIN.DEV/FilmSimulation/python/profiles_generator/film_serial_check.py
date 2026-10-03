#!/usr/bin/env python3
"""film_serial_check.py -- build and run the engine tree's test_film_serial.cpp.

The GPU transfer format of one film::FilmProfile (the generated
film_profile_serial.hpp) carries kMaxSerializedFilmProfileSize, which
serial_codegen.py computes from the PYTHON database. This audit is the C++
side of that equation: it compiles the database objects and the harness from
`--root`, serializes every profile in C++, and fails unless the largest equals
the constant, every round trip is bit-exact and every malformed buffer is
rejected (anything but "RESULT PASS").

What the harness checks is documented in test_film_serial.cpp: round trip of
all 222 profiles with bit-exact comparison of every serialized field,
capacity never overrun, ~12 000 malformed buffers rejected, 40 000 random
corruptions read in bounds, the maximum pinned exactly.

    python3 film_serial_check.py --root <project root> [--assert]

Like stage_parity / chain_parity it FAILS, rather than skips, when the header
is present and the harness is not.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADER = "film_profile_serial.hpp"
HARNESS = "test_film_serial.cpp"


def _root(argv_root: str | None) -> Path:
    if argv_root:
        return Path(argv_root).resolve()
    env = os.environ.get("FILMSIM_ROOT")
    return Path(env).resolve() if env else HERE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--assert", dest="strict", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the build directory")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    import build as b                       # engine_root
    import stage_parity as sp               # _cxx, _compile_all

    root = _root(args.root)
    eng = b.engine_root(root)
    header = eng / HEADER
    harness = eng / HARNESS
    if not header.is_file():
        print("[!] %s not in %s" % (HEADER, eng))
        return 2
    if not harness.is_file():
        print("[!] %s is in %s but %s is not -- this audit needs it and will "
              "not pretend it passed without it" % (HEADER, eng, HARNESS))
        return 2

    cxx = sp._cxx()
    base = ["-std=c++14", "-O1"]
    tmp = Path(tempfile.mkdtemp(prefix="fs_serial_"))
    try:
        obj = tmp / "obj"
        obj.mkdir()
        db = ([eng / "film_profiles.cpp", eng / "LoadFilmDataBase.cpp"]
              + sorted(eng.glob("film_profiles_data_*.cpp")))
        db_obj = [obj / ("db_" + p.stem + ".o") for p in db]
        sp._compile_all([([cxx] + base + ["-I", str(eng), "-c", str(s), "-o", str(o)],
                          "database " + s.name) for s, o in zip(db, db_obj)], False)

        # the harness (and through it the header) must compile with ZERO output
        hobj = obj / "test_film_serial.o"
        r = subprocess.run([cxx] + base + ["-Wall", "-Wextra", "-I", str(eng),
                            "-c", str(harness), "-o", str(hobj)],
                           capture_output=True, text=True)
        noise = (r.stdout + r.stderr).strip()
        if r.returncode != 0 or noise:
            print("[!] %s: exit %d, %d bytes of compiler output:\n  %s"
                  % (HARNESS, r.returncode, len(noise.encode()),
                     "\n  ".join(noise.splitlines()[:12])))
            return 1
        # the reader must also compile without the database headers
        probe = tmp / "reader_only.cpp"
        probe.write_text('#define FILM_SERIAL_READER_ONLY\n#include "%s"\n'
                         'int main(){return film::serial::ValidateFilmProfileBuffer(nullptr,0)?1:0;}\n' % HEADER)
        r = subprocess.run([cxx] + base + ["-Wall", "-Wextra", "-I", str(eng),
                            "-c", str(probe), "-o", str(obj / "reader_only.o")],
                           capture_output=True, text=True)
        noise = (r.stdout + r.stderr).strip()
        if r.returncode != 0 or noise:
            print("[!] %s reader-only build: exit %d:\n  %s"
                  % (HEADER, r.returncode, "\n  ".join(noise.splitlines()[:12])))
            return 1

        exe = tmp / "test_film_serial"
        r = subprocess.run([cxx, "-o", str(exe), str(hobj)] + [str(o) for o in db_obj],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("[!] link failed:\n  %s" % (r.stderr or r.stdout).strip()[:1200])
            return 1
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        lines = out.splitlines()
        if r.returncode != 0 or "RESULT PASS" not in lines:
            for ln in lines:
                if ln.startswith("  FAIL") or ln.startswith("RESULT"):
                    print("[FAIL] " + ln.strip())
            print("[!] test_film_serial exit %d" % r.returncode)
            return 1
        size = next((ln for ln in lines if ln.startswith("size: min")), "")
        checks = next((ln for ln in lines if ln.startswith("profiles ")), "")
        mal = next((ln for ln in lines if ln.startswith("malformed:")), "")
        m = re.search(r"max (\d+) \(([^)]+)\).*kMaxSerializedFilmProfileSize (\d+)", size)
        print("[OK] film serial: %s; %s; %s; largest %s = %s B == kMaxSerializedFilmProfileSize %s"
              % (checks, mal, size.split(",")[0], m.group(2), m.group(1), m.group(3))
              if m else "[OK] film serial: %s" % checks)
        return 0
    finally:
        if args.keep:
            print("build dir kept: %s" % tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
