"""Scalar vs AVX2 numeric parity for ALL 27 pipeline stages, per pixel.

WHY THIS EXISTS
---------------
Until today the numeric twin comparison covered THREE stages. `interimage_parity`
runs 8b and 9 in both flavours, `bromide_parity` runs 9c in both. The other
twenty-four had no automated numeric comparison between the scalar (double)
project and the vector (float) one at all -- only `cpp_parity`'s textual token
grep, which cannot see a number.

⚠ THAT GAP LET A REAL DEFECT SHIP, AND THE DEFECT IS THE SPECIFICATION FOR THIS
FILE. AVX2 stage 14 evaluated 10^-d with a raw Schraudolph bit-hack: wrong by
2.98 % relative, 5.57 eight-bit code values, the white point landing at 0.9782
instead of 1.0 -- a visible cast over the whole frame. Nothing caught it. It was
found by the owner looking at a rendered picture, which is the slowest and least
reliable instrument in the project.

⚠ AND A MEAN WOULD NOT HAVE CAUGHT IT EITHER, WHICH IS WHY THIS IS NOT
`test_stage_parity.cpp`. That harness exists, it is good, and its own header says
"run it from both builds and diff" -- by hand. It prints each stage's mean, min
and max. A mean is an integral: a divergence that is +e over half the frame and
-e over the other half integrates to zero and leaves no trace in it, and a
min/max pair inspects two pixels out of 3N^2. So this audit computes the only
quantity that cannot hide anything -- the MAX ABSOLUTE DIFFERENCE over every
pixel and every channel -- and prints the coordinate where it occurs, because a
number without a location is not something anyone can go and look at.

HOW THE TWO BUILDS ARE PRODUCED
-------------------------------
⚠ `-I <root>/AVX2` DOES NOT WORK AND THE REASON IS A C++ RULE, not a build-script
bug: a quoted `#include "AlgoTypes.hpp"` resolves FIRST against the directory of
the file doing the including. Every shared header lives in the scalar root, so
each of them drags in the scalar `AlgoTypes.hpp`, `AlgoType` comes out `double`,
and the vector TU tries to `_mm256_loadu_ps` a `const double*`. The fix is
`interimage_parity.stage_avx2_tree`, reused here verbatim: copy both sets into
ONE directory, AVX2 LAST, so the includer's own directory is the right answer for
every file.

⚠ THE DATABASE OBJECTS ARE COMPILED ONCE AND LINKED INTO BOTH BUILDS. The
twenty-odd `film_profiles_data_NN.cpp` slots are the bulk of the compile time and
`film_profiles.hpp` does not include `AlgoTypes.hpp`, so they are byte-identical
between the flavours. Sharing them is what keeps this audit affordable on every
build instead of something people learn to `--skip`.

THE TOLERANCE, AND WHERE IT COMES FROM
--------------------------------------
The two builds run the same laws in different types. float32's unit roundoff is
u = 2^-24 = 5.96e-08, so the expected disagreement is float32 rounding
accumulated down a 27-stage chain that contains two separable blurs, an anchor
fixed-point solve and a curve inversion. Nobody can bound the true condition
number of that chain from first principles, so the budget is stated as a number
of float32 ULPs of the PLANE'S OWN WORKING MAGNITUDE and the measured floor is
recorded against it:

    tol(stage) = BUDGET_ULPS * u * scale(stage)
    scale      = max(1, max|value| on the scalar plane)

⚠ SCALE-RELATIVE AND NOT A FLAT ABSOLUTE, because the three domains this pipeline
passes through are three different magnitudes. The exposure-domain planes
(stages 2..7) carry LINEAR relative exposure and reach 141 on this test field --
a flat 1e-5 there would be 0.07 ULP and fail on the first rounding. The
density-domain planes are 1.7 to 3.9 D. The display-domain planes are
transmittances in 0..1.

⚠ THE DISPLAY DOMAIN GETS ONE EXTRA FACTOR OF ln(10) AND IT IS NOT A FUDGE.
Stage 14 evaluates T = 10^-d, whose derivative is -ln(10)*T, so a density error e
arriving from stage 13 leaves as up to 2.303*e while the plane's magnitude FALLS
from Dmax to 1. Without that factor the budget would tighten by 2.3x exactly
where the arithmetic loosens by 2.3x. Measured: FUJI_VELVIA_50 stage 17 sits at
2.6e-05 against a 1.0 scale, which no scale-only rule can accommodate.

MEASURED FLOOR (5 stocks x 25 retained planes x 3 frame sizes, 2026-09-11):
the worst disagreement on any stage where the two twins run the SAME algorithm is
4.1e-05 absolute (exposure domain, scale 140 -> 2.9e-07 relative), 2.0e-05 in the
density domain and 2.6e-05 in the display domain. BUDGET_ULPS = 512 puts the gate
3x to 100x above that floor and 100x to 280x BELOW the two divergences it found.
That separation is the whole argument for the number: the defect class this audit
exists to catch -- a transcendental replaced by an approximation -- is three
orders of magnitude larger than float32 rounding, so the gate does not need to be
tight to be decisive.

WHAT IT FOUND
-------------
⚠ STAGE 13's AVX2 PRINT CURVE IS A DIFFERENT ALGORITHM FROM ITS SCALAR TWIN, AND
IT IS OFF BY UP TO 1.4e-02 D. `AVX2/Algo_13_Sim.cpp::printPlane` keeps a FAST
APPROXIMATE SOFTPLUS where the scalar twin calls the exact `AlgoSoftplus`, and
its own comment documents that choice with timings -- it is a deliberate mode
difference, not an accident. What the comment states as the cost is "about
2.4e-03 to 3.2e-03 in plane MEAN". Measured here per pixel: 1.03e-02 to 1.37e-02
D, four to five times the figure the comment gives, because a mean is an integral
and the error is signed. Through 10^-d that lands as 7.4e-03 to 7.7e-03 of
transmittance -- 1.9 to 2.0 eight-bit code values -- on every stock that prints.

⚠ IT IS PINNED, NOT FORGIVEN. `KNOWN_DIVERGENCE` carries a per-(stage, stock)
CEILING with the measured number in it, and the ceiling is set BELOW the size of
the stage-14 defect that motivated this file (2.2e-02 of transmittance at the
white point). So a second approximation of that class, dropped into the print
chain tomorrow, still fails the gate. A pinned entry that starts passing strictly
also reports -- the same discipline `build.py`'s VERIFY_BASELINE uses -- so the
table shrinks deliberately rather than by accident.

⚠ SKIP MEANS ONE THING ONLY: THE AVX2 TREE IS NOT ON DISK. If both trees are
present and a stage disagrees, that is a FAIL. The 2026-09-08 reversal-sign
defect survived precisely because a `[SKIP]` line sat in a green log and read as
a pass, and this file will not repeat it: it also FAILS if the two builds report
the same `sizeof(AlgoType)`, because two scalar runs agreeing perfectly proves
nothing about the vector twin.

Run:
    python stage_parity.py
    python stage_parity.py --assert
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from interimage_parity import stage_avx2_tree

HERE = Path(__file__).resolve().parent

#: The harness. A FILE in the plugin tree rather than a string in this module,
#: unlike the probes in `interimage_parity` and `bromide_parity`, and for a
#: reason those two do not have: this one enumerates all 25 retained buffer
#: triples of `MemHandler`. That list changes when the PIPELINE changes, so it
#: belongs next to `test_stage_parity.cpp` where whoever adds a stage will see
#: it -- not inside a generator script they have no reason to open.
#: The `test_` prefix keeps it out of the production archives, by the same rule
#: `stage_delivery._is_test_source` applies to every other harness.
HARNESS = "test_stage_dump.cpp"

#: Engine TUs are everything in the root except the harnesses, the profiling
#: driver and the generated database -- the database is compiled separately and
#: shared between the two builds. Discovered rather than listed: a new
#: `Algo_18_Sim.cpp` must join both builds without anyone remembering to edit a
#: tuple here.
_NOT_ENGINE = ("e2e.cpp", "profall.cpp", "film_profiles.cpp",
               "LoadFilmDataBase.cpp")

#: ⚠ FIVE STOCKS, CHOSEN FOR THE PATHS THEY TAKE AND NOT FOR POPULARITY. The
#: frame is small and the audit runs on every build, so every entry has to earn
#: its render:
#:   KODAK_PORTRA_400         colour negative, strong DIR couplers, AND it runs
#:                            the print chain -- so stages 13..17 are exercised
#:                            in their printed branch
#:   FUJI_VELVIA_50           reversal: the other interimage mechanism, the
#:                            negative density excursions the physical floors
#:                            act on, and NO print -- so on this stock stages
#:                            13..17 stay under the STRICT gate and a new
#:                            divergence there cannot hide behind stage 13's
#:                            pinned one
#:   EASTMAN_DOUBLE_X_5222    monochrome: stage 7 collapses to one record and
#:                            every colour stage must be inert on it
#:   DUFAYCOLOR_1937          the ONLY stock in the database with has_reseau, so
#:                            it is the only render in which stage 14b
#:                            reconstructs instead of returning on its first
#:                            branch. Reversal too, so it also keeps 13..17
#:                            strict
#:   KODAK_VISION3_500T_5219  cine negative, the strongest stored couplers and
#:                            halation -- the everyday production path
#: ⚠ NAMED, NOT INDEXED. The harness resolves each against the loaded database,
#: so an enum renumbering fails loudly instead of quietly auditing a different
#: film.
STOCKS = ("KODAK_PORTRA_400", "FUJI_VELVIA_50", "EASTMAN_DOUBLE_X_5222",
          "DUFAYCOLOR_1937", "KODAK_VISION3_500T_5219")

#: ⚠ 44 AND NOT 48: deliberately NOT a multiple of eight, so the AVX2 tail mask
#: runs on every row. A frame that divides evenly by the lane count never
#: executes the masked epilogue, which is where a vector twin's off-by-one lives.
SIZE = 44

#: bit0 grain, bit1 film damage. BOTH ON, and that is a change of posture from
#: `test_stage_parity.cpp`, which switches them off for determinism.
#: ⚠ THE COUNTER RNG MAKES THEM DETERMINISTIC AND THEREFORE COMPARABLE. Every
#: value is a pure function of (seed, frame, stage salt, pixel ordinal) and
#: `AVX2/Algo_11_Sim.cpp` packs the SAME ordinal per lane as the scalar loop, so
#: the two builds draw the same numbers and differ only by float32 rounding on
#: the transform. Measured: stage 11 agrees to 4.7e-06 with grain at full
#: strength. Turning grain and damage off would leave stages 11, 9b and 16
#: unexercised, which is three of the twenty-four this file was written to cover.
FLAGS = 3

#: float32 unit roundoff. Everything below is expressed in these.
_U32 = 2.0 ** -24

#: See the module docstring. 512 ULPs of the plane's own magnitude.
BUDGET_ULPS = 512.0

#: Stages whose planes are transmittance rather than density, and which
#: therefore carry the ln(10) amplification of stage 14's 10^-d.
_DISPLAY_STAGES = frozenset(("14", "14b", "14c", "15", "16", "17"))

_LN10 = 2.302585092994046

#: ⚠ PINNED DIVERGENCES: (stage, stock) -> (ceiling, why). NOT a tolerance and
#: not a place to park an inconvenient number. An entry may exist only where the
#: two twins deliberately run DIFFERENT ALGORITHMS and the engine says so in its
#: own source.
#:
#: ⚠⚠ THE TABLE IS EMPTY, AND IT WAS NOT EMPTY WHEN THIS FILE WAS WRITTEN.
#: On the day this audit was built it pinned stage 13 on the three stocks that
#: print, plus stages 14 to 17 which inherited from it -- 21 entries. The
#: mechanism was `AVX2/Algo_13_Sim.cpp::printPlane` keeping a fast approximate
#: softplus where the scalar twin called the exact `AlgoSoftplus`, which that
#: file's own comment described as a documented mode difference and priced at
#: "2.4e-03 to 3.2e-03 in plane MEAN".
#:
#: This audit measured it PER PIXEL for the first time and found 1.03e-02 to
#: 1.37e-02 D -- four to five times the quoted figure, because a mean is an
#: integral and the error is signed. About two eight-bit code values reached
#: the screen on every stock that prints. So the pin was recording a real
#: defect rather than a defensible trade, and the defect was fixed the same
#: day: the vector stage now uses an accurate exp and log defined in its own
#: translation unit, and stage 13 came back at 8.33e-06 with 14..17 at
#: 4.2e-06 to 6.3e-06 -- all inside the STRICT float32 budget.
#:
#: ⚠ SO EVERY ONE OF THE 27 STAGES NOW AGREES AT FLOAT32 PRECISION WITH NO
#: EXCEPTIONS, and that is the state this table exists to defend. Adding an
#: entry here is admitting the twins compute different things; it should be
#: hard to justify and it should never again be justified by a statistic that
#: averages the error away. If a new entry seems necessary, measure per pixel
#: first.
KNOWN_DIVERGENCE: dict[tuple[str, str], tuple[float, str]] = {}


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def _cxx() -> str:
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        raise SystemExit("[!] no g++/clang++ on PATH (set CXX to override)")
    return cxx


def _engine_tus(tu_dir: Path) -> list[Path]:
    return sorted(p for p in tu_dir.glob("*.cpp")
                  if not p.name.startswith("test_")
                  and not p.name.startswith("film_profiles_data_")
                  and p.name not in _NOT_ENGINE)


def _compile_all(jobs: list[tuple[list[str], str]], echo: bool) -> None:
    """Run every compile, in parallel, and raise on the first that fails.

    ⚠ THE ONE WARNING THAT IS NOT A FAILURE is `Common.hpp`'s "type qualifiers
    ignored on cast result type", which every TU in the project emits and which
    predates this file. `build.py`'s compile stage gates on zero bytes of output;
    this one cannot, so it gates on the exit code and leaves the warning alone
    rather than pretending to have cleaned it up.
    """
    cxx = _cxx()

    def one(job):
        cmd, label = job
        if echo:
            print("    $ %s" % " ".join([Path(cxx).name] + cmd[1:]))
        r = subprocess.run(cmd, capture_output=True, text=True)
        return label, r

    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() or 1)) as ex:
        for label, r in ex.map(one, jobs):
            if r.returncode != 0:
                tail = (r.stderr or r.stdout).strip().splitlines()
                raise SystemExit("[!] %s did not compile:\n  %s"
                                 % (label, "\n  ".join(tail[:16])))


def build_pair(root: Path, tmp: Path, echo: bool):
    """-> (scalar_exe, avx2_exe_or_None). Compiles the database ONCE.

    Returns None for the vector executable when `<root>/AVX2` carries no twins,
    which is the ONLY condition under which this audit is allowed to skip.
    """
    cxx = _cxx()
    obj = tmp / "obj"
    obj.mkdir(parents=True, exist_ok=True)

    base = ["-std=c++14", "-Wall", "-Wextra", "-O1",
            "-DALGO_RETAIN_ALL_STAGES=1"]

    # --- the database, once, shared -----------------------------------------
    db_src = ([root / "film_profiles.cpp", root / "LoadFilmDataBase.cpp"]
              + sorted(root.glob("film_profiles_data_*.cpp")))
    db_obj = [obj / ("db_" + p.stem + ".o") for p in db_src]
    jobs = [([cxx] + base + ["-I", str(root), "-c", str(s), "-o", str(o)],
             "database %s" % s.name)
            for s, o in zip(db_src, db_obj)]
    if echo:
        print("  [1] database TUs, compiled once and linked into BOTH builds")
    _compile_all(jobs, echo and len(jobs) <= 3)
    if echo:
        print("    $ %s -std=c++14 -Wall -Wextra -O1 "
              "-DALGO_RETAIN_ALL_STAGES=1 -I <root> -c "
              "film_profiles.cpp LoadFilmDataBase.cpp "
              "film_profiles_data_*.cpp        (%d TUs)"
              % (Path(cxx).name, len(jobs)))

    out = []
    for avx2 in (False, True):
        tu_dir = root
        extra: list[str] = []
        inc = ["-I", str(root)]
        if avx2:
            staged = stage_avx2_tree(root, tmp)
            if staged is None:
                out.append(None)
                continue
            tu_dir = staged
            extra = ["-mavx2", "-mfma"]
            # ⚠ THE STAGED TREE FIRST, THE ROOT SECOND. The tree holds every
            # shared header plus the vector twins; the root is still needed for
            # film_profiles.hpp and friends, which stage_avx2_tree deliberately
            # does not copy (the database objects above already define them, and
            # a second copy on the include path is a redefinition, not a
            # divergence).
            inc = ["-I", str(staged), "-I", str(root)]

        tag = "avx2" if avx2 else "scalar"
        src = _engine_tus(tu_dir) + [tu_dir / HARNESS]
        if not (tu_dir / HARNESS).is_file():
            raise SystemExit("[!] %s is not in %s -- this audit needs it and "
                             "will not pretend it passed without it"
                             % (HARNESS, tu_dir))
        eobj = [obj / ("%s_%s.o" % (tag, p.stem)) for p in src]
        jobs = [([cxx] + base + extra + inc + ["-c", str(s), "-o", str(o)],
                 "%s %s" % (tag, s.name))
                for s, o in zip(src, eobj)]
        if echo:
            print("  [%d] %s engine + harness (%d TUs)"
                  % (2 + int(avx2), tag, len(jobs)))
            print("    $ %s"
                  % " ".join([Path(cxx).name] + base + extra + inc
                             + ["-c", "<each of the %d>" % len(jobs)]))
        _compile_all(jobs, False)

        exe = tmp / ("stage_dump_" + tag)
        cmd = [cxx] + extra + ["-o", str(exe)] + [str(o) for o in eobj + db_obj]
        if echo:
            print("    $ %s"
                  % " ".join([Path(cxx).name] + extra
                             + ["-o", "stage_dump_" + tag,
                                "<%d objects>" % (len(eobj) + len(db_obj))]))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("[!] %s did not link:\n  %s"
                             % (tag, (r.stderr or r.stdout).strip()[:1200]))
        out.append(exe)
    return out[0], out[1]


# ---------------------------------------------------------------------------
# running and reading
# ---------------------------------------------------------------------------
def run_dump(exe: Path, tmp: Path, tag: str):
    """Render every stock and read the binary back. -> (sizeof, N, {(stock, stage): (3,N,N)})"""
    path = tmp / ("dump_%s.bin" % tag)
    cmd = [str(exe), str(path), str(SIZE), str(FLAGS), *STOCKS]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp))
    if r.returncode != 0:
        raise SystemExit("[!] %s harness exited %d: %s"
                         % (tag, r.returncode,
                            (r.stdout + r.stderr).strip()[-600:]))

    blob = path.read_bytes()
    if blob[:8] != b"SPDUMP01":
        raise SystemExit("[!] %s wrote a file this reader does not know" % tag)
    size, n, nstock, _flags = struct.unpack("<4i", blob[8:24])
    off = 24
    planes: dict[tuple[str, str], np.ndarray] = {}
    order: list[tuple[str, str]] = []
    npx = 3 * n * n
    for _ in range(nstock):
        name = blob[off:off + 64].split(b"\0")[0].decode()
        off += 64
        _idx, nplane = struct.unpack("<2i", blob[off:off + 8])
        off += 8
        for _p in range(nplane):
            stage = blob[off:off + 8].split(b"\0")[0].decode()
            off += 8
            a = np.frombuffer(blob, dtype="<f8", count=npx,
                              offset=off).reshape(3, n, n)
            off += npx * 8
            planes[(name, stage)] = a
            order.append((name, stage))
    if off != len(blob):
        raise SystemExit("[!] %s dump is %d bytes short or long"
                         % (tag, len(blob) - off))
    return size, n, planes, order


def tolerance(stage: str, scale: float) -> float:
    """The strict float32-rounding budget for one plane. See the module note."""
    eff = max(1.0, scale)
    if stage in _DISPLAY_STAGES:
        eff *= _LN10
    return BUDGET_ULPS * _U32 * eff


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/root/work/proot",
                    help="project root holding the plugin's Algo_*.cpp and AVX2/")
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    help="exit non-zero on any disagreement")
    ap.add_argument("--echo", action="store_true",
                    help="print the compile commands")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()

    if not (root / "AlgorithmMain.cpp").is_file():
        print("  [SKIP] plugin sources not present under %s" % root)
        return 0
    if not (root / HARNESS).is_file():
        print("  [FAIL] %s is missing from %s. This audit does not skip on a "
              "missing harness: a green log with a SKIP line in it is how the "
              "last defect got through" % (HARNESS, root))
        return 1 if ns.do_assert else 0

    # ⚠ CHECKED BEFORE ANYTHING IS COMPILED, and that ordering is the point.
    # This is the ONE condition that may skip, so it must be decided on the
    # facts on disk rather than discovered forty seconds into a build -- and
    # deciding it here means a skip cannot ever be the consequence of something
    # having gone wrong in the scalar half.
    twins = root / "AVX2"
    if not twins.is_dir() or not any(p.is_file() for p in twins.iterdir()):
        print("  [SKIP] no AVX2 twins under %s -- nothing to compare against"
              % twins)
        return 0

    with tempfile.TemporaryDirectory(prefix="stage_parity_") as td:
        tmp = Path(td)
        scalar_exe, avx2_exe = build_pair(root, tmp, ns.echo)
        if avx2_exe is None:
            print("  [FAIL] %s exists but the staged tree came back empty"
                  % twins)
            return 1

        s_size, n, s_planes, order = run_dump(scalar_exe, tmp, "scalar")
        v_size, v_n, v_planes, _ = run_dump(avx2_exe, tmp, "avx2")

    # ⚠ TWO RUNS OF THE SAME BUILD AGREE PERFECTLY AND PROVE NOTHING. If the
    # AVX2 tree failed to shadow the scalar headers, both sides come out as
    # double, every line reads OK and the audit is a decoration.
    if s_size == v_size:
        print("  [FAIL] both builds report sizeof(AlgoType) == %d: the AVX2 "
              "tree did not shadow the scalar one, so this run proved nothing "
              "about the vector twin" % s_size)
        return 1
    if n != v_n:
        print("  [FAIL] the two builds rendered different frame sizes, %d and "
              "%d" % (n, v_n))
        return 1

    print("[i] scalar sizeof(AlgoType) = %d, AVX2 = %d; %d x %d frame, grain "
          "%s, damage %s" % (s_size, v_size, n, n,
                             "on" if FLAGS & 1 else "off",
                             "on" if FLAGS & 2 else "off"))
    print("[i] tolerance = %.0f float32 ULPs of the plane's own magnitude "
          "(%.3g x max(1, |plane|), x ln10 on the transmittance stages)"
          % (BUDGET_ULPS, BUDGET_ULPS * _U32))

    fails: list[str] = []
    stale: list[str] = []
    worst_clean = 0.0
    stock_now = None

    for key in order:
        stock, stage = key
        if stock != stock_now:
            stock_now = stock
            print("\n-- %s" % stock)
        a = s_planes[key]
        b = v_planes.get(key)
        if b is None:
            fails.append("%s stage %s: the AVX2 build produced no plane"
                         % (stock, stage))
            print("  [FAIL] %-4s no AVX2 plane" % stage)
            continue

        d = np.abs(a - b)
        worst = float(d.max())
        ch, y, x = (int(v) for v in np.unravel_index(int(np.argmax(d)), d.shape))
        scale = float(np.abs(a).max())
        tol = tolerance(stage, scale)

        pin = KNOWN_DIVERGENCE.get((stage, stock))
        if worst <= tol:
            worst_clean = max(worst_clean, worst)
            print("  [OK]   %-4s max %.3e  (limit %.3e, scale %8.3f)  "
                  "at %s(%d,%d)" % (stage, worst, tol, scale, "RGB"[ch], x, y))
            if pin is not None:
                stale.append("%s stage %s now agrees to %.3e, inside the strict "
                             "%.3e -- drop its KNOWN_DIVERGENCE entry"
                             % (stock, stage, worst, tol))
        elif pin is not None and worst <= pin[0]:
            print("  [KNOWN] %-3s max %.3e  (strict %.3e, pinned ceiling %.3e)  "
                  "at %s(%d,%d)  -- %s"
                  % (stage, worst, tol, pin[0], "RGB"[ch], x, y, pin[1]))
        else:
            over = "pinned ceiling %.3e" % pin[0] if pin else "limit %.3e" % tol
            fails.append("%s stage %s: max |scalar-avx2| = %.3e at %s(%d,%d), "
                         "over its %s" % (stock, stage, worst, "RGB"[ch], x, y,
                                          over))
            print("  [FAIL] %-4s max %.3e  (%s, scale %8.3f)  at %s(%d,%d)"
                  % (stage, worst, over, scale, "RGB"[ch], x, y))

    print()
    npin = sum(1 for k in KNOWN_DIVERGENCE if k[1] in STOCKS)
    print("[i] worst disagreement on a stage where the two twins run the SAME "
          "algorithm: %.3e over %d stocks x %d planes"
          % (worst_clean, len(STOCKS), len(order) // len(STOCKS)))
    if npin:
        # ⚠ PRINTED EVEN WHEN EVERYTHING PASSES. A pinned divergence that stops
        # being mentioned is a pinned divergence nobody remembers is there.
        print("[i] %d pinned (stage, stock) divergences, all from ONE mechanism: "
              "%s" % (npin, _PRINT_CURVE))

    for s in stale:
        print("[WARN] %s" % s)

    if fails:
        print("\n[FAIL] %d stage(s) outside tolerance:" % len(fails))
        for f in fails:
            print("  " + f)
        return 1 if ns.do_assert else 0

    print("\n[OK] all %d retained planes agree between the scalar and AVX2 "
          "builds across %d stocks -- %d within the float32 budget, %d at "
          "pinned known divergences"
          % (len(order), len(STOCKS), len(order) - npin, npin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
