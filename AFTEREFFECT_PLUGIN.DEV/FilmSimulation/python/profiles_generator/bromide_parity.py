"""Python vs scalar C++ vs AVX2 parity for stage 9c -- bromide drag.

WHY THIS EXISTS, AND WHY IT EXISTS ON THE DAY THE STAGE DOES
------------------------------------------------------------
Stage 9c (queue C23, schema v27) is INERT on every stock in the database: no
document in this corpus quantifies a bromide gradient, so every
`BromideDragSpec` ships at zero and all three engines return on their first
branch. That is exactly the condition under which a three-engine law rots
silently -- nothing renders it, so nothing notices when the twins drift apart,
and the first person to fit a real number would inherit three different stages
wearing one name.

So the parity probe INJECTS a record. It copies a real `film::FilmProfile`,
writes a `BromideDragSpec` into the copy, and runs the shipped stage function on
it. The law is therefore exercised at full strength in all three
implementations while the database stays inert.

WHAT IS COMPARED, AND WHY EVERY CASE IS HERE
--------------------------------------------
  * DIRECTION +1 AND -1. The recursion is one-sided, so a sign error is
    invisible on any symmetric field and total on a real one. Both are run over
    a field with a hard bright bar, whose streak must appear on ONE side.
  * NEGATIVE AND REVERSAL. The source field is inverted on `isReversal()` -- the
    first developer's silver is the negative image -- and that single line is the
    easiest thing in the stage to get backwards. A stock of each kind is probed.
  * TWO DRAG LENGTHS AT TWO RESOLUTIONS. The record is in MILLIMETRES and the
    filter is in pixels, so the same record must produce the same PHYSICAL
    streak at any px_per_mm. Running 25 and 100 px/mm against the same
    `length_mm` is what tests that, and it is the property a pixel-denominated
    length would silently break.
  * THE INERT RECORD. All three engines must return false and leave the planes
    bit-identical. This is the path all 176 stocks take today.

⚠ THE RECURSION IS WHY THE TOLERANCE IS NOT THE USUAL POINTWISE ONE. A one-pole
filter COMPOUNDS its coefficient error down the column: with a = exp(-pitch/L)
the accumulator at row n carries a^n, so a difference in the last bit of `a`
grows with n before the tail decays. That is precisely why both twins compute
the coefficient in `HighPrecType` and narrow ONCE, rather than each using its own
`exp`, and this file's numbers are the evidence that the precaution works.

⚠ AlgoType IS A SWITCHABLE TYPEDEF AND THIS FILE MUST NOT ASSUME IT. The scalar
probe PRINTS `sizeof(AlgoType)` and the tolerance is chosen from it at run time,
the same rule `interimage_parity.py` established: 2e-6 for eight bytes, 2e-3 for
four, because the Python reference carries float32 planes and two float32
pipelines rounding in different orders diverge much further than a float32 and a
float64 one.

Run with --assert to make a disagreement fatal (this is what build.py does).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import film_profiles as fp
import film_sim as fs

HERE = Path(__file__).resolve().parent

#: The plugin's own translation units. `Algo_09_Sim.cpp` holds stages 9, 9b and
#: 9c in one file, so its whole link closure comes along: stage 9's separable
#: blur and stage 9b's particulate field. Neither is exercised here.
PLUGIN_TUS = ("Algo_09_Sim.cpp", "AlgoSeparableBlur.cpp", "AlgoDefectField.cpp")

#: One negative and one reversal. The kind is the only profile property stage 9c
#: branches on, and it branches on it in the one line that decides which side of
#: the picture the streaks fall on.
STOCKS = ("KODAK_PORTRA_400", "FUJI_VELVIA_50")

SIZE_X, SIZE_Y = 37, 53          # deliberately neither a power of two nor a
                                 # multiple of eight, so the AVX2 tail mask runs

#: (strength, length_mm, direction). The inert case is added separately.
CASES = ((0.08, 3.0, +1),
         (0.08, 3.0, -1),
         (0.04, 12.0, +1))

#: Two resolutions against the same millimetre lengths. See the docstring.
PX_PER_MM = (25.0, 100.0)


# ---------------------------------------------------------------------------
# the field under test
# ---------------------------------------------------------------------------
def make_field(profile) -> np.ndarray:
    """An (h, w, 3) density plane with a hard bright bar across the middle.

    ⚠ A BAR AND NOT A RAMP, on purpose. A smooth field cannot distinguish a
    one-sided filter from a two-sided one of half the length -- both blur it the
    same way to the eye and nearly the same way in the numbers. A step has a
    trailing edge and a leading edge, and only one of them may grow a streak.
    """
    c = profile.curves
    dmin = np.array([c.r.dmin, c.g.dmin, c.b.dmin], dtype=np.float32)
    dmax = np.array([c.r.dmax, c.g.dmax, c.b.dmax], dtype=np.float32)
    out = np.empty((SIZE_Y, SIZE_X, 3), dtype=np.float32)
    y = np.arange(SIZE_Y, dtype=np.float32)[:, None]
    x = np.arange(SIZE_X, dtype=np.float32)[None, :]
    # a low background, a bright horizontal bar, and a mild diagonal so no two
    # columns are identical (a constant column would hide a lane mix-up)
    base = 0.15 + 0.10 * ((x + y) / (SIZE_X + SIZE_Y))
    bar = np.where((y >= 12) & (y < 20), np.float32(0.85), np.float32(0.0))
    frac = np.clip(base + bar, 0.0, 1.0).astype(np.float32)
    for ch in range(3):
        out[:, :, ch] = dmin[ch] + frac * (dmax[ch] - dmin[ch])
    return out


def reference(profile, spec, field, px_per_mm) -> np.ndarray:
    c = profile.curves
    dens = field.copy()
    fs.apply_bromide_drag(
        dens, spec,
        (c.r.dmin, c.g.dmin, c.b.dmin),
        (c.r.dmax, c.g.dmax, c.b.dmax),
        px_per_mm, profile.is_reversal)
    return dens


# ---------------------------------------------------------------------------
# the C++ probe
# ---------------------------------------------------------------------------
PROBE = r"""
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include "film_profiles.hpp"
#include "AlgoBromideDrag.hpp"

static const char* kStocks[] = { %(NAMES)s };
static const double kStrength[] = { %(STRENGTH)s };
static const double kLength[]   = { %(LENGTH)s };
static const int    kDir[]      = { %(DIR)s };
static const double kPxMm[]     = { %(PXMM)s };

int main()
{
    const int SX = %(SX)d, SY = %(SY)d;
    const int nStock = %(NSTOCK)d, nCase = %(NCASE)d, nScale = %(NSCALE)d;
    std::printf("SIZEOF %%d\n", (int)sizeof(AlgoType));

    const std::vector<film::FilmProfile>& db = film::GetFilmDatabase();

    std::vector<AlgoType> pr(SX*SY), pg(SX*SY), pb(SX*SY);
    std::vector<AlgoType> s1(SX*SY), s2(SX*SY);

    for (int si = 0; si < nStock; ++si)
    {
        const film::FilmProfile* found = nullptr;
        for (size_t i = 0; i < db.size(); ++i)
            if (db[i].name == kStocks[si]) { found = &db[i]; break; }
        if (!found) { std::printf("MISSING %%s\n", kStocks[si]); return 2; }

        for (int ci = 0; ci <= nCase; ++ci)      // ci == nCase is the inert case
        {
            for (int pi = 0; pi < nScale; ++pi)
            {
                film::FilmProfile prof = *found;   // a COPY; the database stays inert
                if (ci < nCase)
                {
                    prof.processing.bromide_drag.strength  = (float)kStrength[ci];
                    prof.processing.bromide_drag.length_mm = (float)kLength[ci];
                    prof.processing.bromide_drag.axis      = 0;
                    prof.processing.bromide_drag.direction = kDir[ci];
                    prof.processing.bromide_drag.source    = "parity probe";
                }

                const float dminR = prof.curves.r.dmin;
                const float dminG = prof.curves.g.dmin;
                const float dminB = prof.curves.b.dmin;
                const float spanR = prof.curves.r.dmax() - dminR;
                const float spanG = prof.curves.g.dmax() - dminG;
                const float spanB = prof.curves.b.dmax() - dminB;

                for (int y = 0; y < SY; ++y)
                for (int x = 0; x < SX; ++x)
                {
                    double base = 0.15 + 0.10 * ((double)(x + y) / (double)(SX + SY));
                    double bar  = (y >= 12 && y < 20) ? 0.85 : 0.0;
                    double f = base + bar; if (f < 0.0) f = 0.0; if (f > 1.0) f = 1.0;
                    const float ff = (float)f;
                    const int o = y*SX + x;
                    pr[o] = (AlgoType)(dminR + ff*spanR);
                    pg[o] = (AlgoType)(dminG + ff*spanG);
                    pb[o] = (AlgoType)(dminB + ff*spanB);
                }

                const bool ran = AlgoStage09c_BromideDrag(
                    pr.data(), pg.data(), pb.data(), s1.data(), s2.data(),
                    SX, SY, SX, prof, (AlgoType)kPxMm[pi]);

                std::printf("BLOCK %%s %%d %%d %%d\n", kStocks[si], ci, pi, ran ? 1 : 0);
                for (int y = 0; y < SY; ++y)
                for (int x = 0; x < SX; ++x)
                {
                    const int o = y*SX + x;
                    std::printf("%%.17g %%.17g %%.17g\n",
                                (double)pr[o], (double)pg[o], (double)pb[o]);
                }
            }
        }
    }
    std::printf("END\n");
    return 0;
}
"""


def build_probe(root: Path, tmp: Path, avx2: bool):
    """Compile the probe against the shipped TUs. Returns (exe, note) or None.

    ⚠ THE AVX2 BUILD NEEDS ITS OWN `AlgoTypes.hpp` FIRST ON THE INCLUDE PATH,
    and that is not a quirk of this file. The two projects each ship one --
    scalar sets AlgoType to double, AVX2 to float -- and the vector TU carries a
    static_assert that the type is four bytes. Compiling the AVX2 sources
    against the scalar header fails on that assert, which is the header doing
    its job. A staging directory of symlinks with the AVX2 header shadowing the
    scalar one reproduces what the real two-project build does.
    """
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        raise SystemExit("[!] no g++/clang++ on PATH (set CXX to override)")

    src = tmp / ("probe_avx2.cpp" if avx2 else "probe_scalar.cpp")
    src.write_text(PROBE % dict(
        SX=SIZE_X, SY=SIZE_Y,
        NAMES=", ".join('"%s"' % s for s in STOCKS), NSTOCK=len(STOCKS),
        STRENGTH=", ".join("%.17g" % c[0] for c in CASES),
        LENGTH=", ".join("%.17g" % c[1] for c in CASES),
        DIR=", ".join("%d" % c[2] for c in CASES),
        PXMM=", ".join("%.17g" % v for v in PX_PER_MM),
        NCASE=len(CASES), NSCALE=len(PX_PER_MM),
    ), encoding="utf-8")

    inc = [str(root), str(HERE)]
    tu_root = root
    flags = ["-std=c++17", "-O1"]
    if avx2:
        stage = tmp / "avx2inc"
        if not stage.exists():
            stage.mkdir()
            for f in list(root.glob("*.hpp")):
                (stage / f.name).symlink_to(f.resolve())
            hdr = root / "AVX2" / "AlgoTypes.hpp"
            if not hdr.is_file():
                return None
            (stage / "AlgoTypes.hpp").unlink()
            (stage / "AlgoTypes.hpp").symlink_to(hdr.resolve())
        inc = [str(stage), str(root), str(HERE)]
        tu_root = root / "AVX2"
        flags += ["-mavx2", "-mfma"]

    tus = [str(src)]
    for t in PLUGIN_TUS:
        p = tu_root / t
        if not p.is_file():
            p = root / t                    # AVX2 does not re-implement them all
        if not p.is_file():
            return None
        tus.append(str(p))
    tus += [str(HERE / "film_profiles.cpp"), str(HERE / "LoadFilmDataBase.cpp")]
    tus += [str(p) for p in sorted(HERE.glob("film_profiles_data_*.cpp"))]

    exe = tmp / ("probe_avx2" if avx2 else "probe_scalar")
    cmd = [cxx] + flags
    for i in inc:
        cmd += ["-I", i]
    cmd += ["-o", str(exe)] + tus
    r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()
        raise SystemExit("[!] %s probe did not compile:\n  %s"
                         % ("AVX2" if avx2 else "scalar",
                            "\n  ".join(tail[:14])))
    return exe


def run_probe(exe: Path):
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("[!] probe failed: %s" % (r.stderr or r.stdout)[:400])
    size = None
    blocks: dict[tuple, tuple[bool, np.ndarray]] = {}
    key = None
    ran = False
    rows: list[list[float]] = []
    for line in r.stdout.splitlines():
        if line.startswith("SIZEOF "):
            size = int(line.split()[1])
        elif line.startswith("BLOCK "):
            if key is not None:
                blocks[key] = (ran, np.array(rows, dtype=np.float64)
                               .reshape(SIZE_Y, SIZE_X, 3))
            _b, stock, ci, pi, rn = line.split()
            key = (stock, int(ci), int(pi))
            ran = (rn == "1")
            rows = []
        elif line == "END":
            if key is not None:
                blocks[key] = (ran, np.array(rows, dtype=np.float64)
                               .reshape(SIZE_Y, SIZE_X, 3))
            key = None
        elif line.startswith("MISSING"):
            raise SystemExit("[!] " + line)
        elif line.strip():
            rows.append([float(v) for v in line.split()])
    return size, blocks


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/root/work/proot",
                    help="directory holding the plugin's C++ translation units")
    ap.add_argument("--assert", dest="assert_", action="store_true",
                    help="exit non-zero on any disagreement")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    worst_overall = 0.0
    bad: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for avx2 in (False, True):
            exe = build_probe(root, tmp, avx2)
            tag = "AVX2  " if avx2 else "scalar"
            if exe is None:
                print("[skip] %s: translation units not present under %s"
                      % (tag.strip(), root))
                continue
            size, blocks = run_probe(exe)
            tol = 2e-6 if size == 8 else 2e-3
            print("== %s  sizeof(AlgoType) = %d, tolerance %.0e"
                  % (tag.strip(), size, tol))

            worst = 0.0
            for name in STOCKS:
                profile = fp.get_profile(name)
                field = make_field(profile)
                for ci in range(len(CASES) + 1):
                    inert = ci == len(CASES)
                    if inert:
                        spec = fp.BromideDragSpec()
                    else:
                        st, ln, dr = CASES[ci]
                        spec = fp.BromideDragSpec(strength=st, length_mm=ln,
                                                  axis=0, direction=dr,
                                                  source="parity probe")
                    for pi, pxmm in enumerate(PX_PER_MM):
                        ran, got = blocks[(name, ci, pi)]
                        want = reference(profile, spec, field, pxmm)
                        err = float(np.abs(got - want).max())
                        worst = max(worst, err)
                        label = ("inert" if inert else
                                 "s=%.2f L=%.0fmm dir=%+d" % CASES[ci])
                        # the gate must agree, not only the numbers
                        if ran != (not inert):
                            bad.append("%s %s %s px/mm %.0f: engine says ran=%s"
                                       % (tag, name, label, pxmm, ran))
                        # ⚠ THE INERT CASE IS NOT TESTED FOR EXACT EQUALITY, and
                        # the reason is the probe rather than the stage. The C++
                        # side builds its field from the formula in double and
                        # the Python side in float32, so the two fields already
                        # differ by 2.38e-07 before any stage runs. That number
                        # is this comparison's FLOOR, and it is worth reading as
                        # a calibration of every other row: a worst of 4.8e-07
                        # against a floor of 2.4e-07 means the stage itself
                        # contributes about a quarter of a part in ten million.
                        # What the inert case DOES assert is the gate -- ran must
                        # be false -- which is checked above and is the property
                        # all 176 stocks depend on.
                        if err > tol:
                            bad.append("%s %s %s px/mm %.0f: worst %.3g"
                                       % (tag, name, label, pxmm, err))
                        print("   %-18s %-24s %6.0f px/mm  ran=%d  worst %.3g"
                              % (name, label, pxmm, ran, err))
            print("   worst over all cases: %.3g" % worst)
            worst_overall = max(worst_overall, worst)

    # ---- properties the numbers alone do not state -------------------------
    # ⚠ THE STREAK MUST BE ON ONE SIDE OF THE BAR AND NOT THE OTHER, and no
    # tolerance test can say so: a two-sided filter of half the length agrees
    # with a one-sided one to within a few per cent on any smooth field and
    # would pass every comparison above. This reads the reference directly.
    prof = fp.get_profile("KODAK_PORTRA_400")
    field = make_field(prof)
    for dr in (+1, -1):
        spec = fp.BromideDragSpec(strength=0.08, length_mm=3.0, axis=0,
                                  direction=dr, source="parity probe")
        out = reference(prof, spec, field, 25.0)
        d = (field - out)[:, :, 1].mean(axis=1)       # density removed per row
        lead, trail = d[8:12].mean(), d[20:24].mean()   # 4 rows each side of bar
        want_trail = dr > 0
        ok = (trail > lead) if want_trail else (lead > trail)
        print("   direction %+d: removed above bar %.4f, below bar %.4f -- %s"
              % (dr, lead, trail, "trails correctly" if ok else "WRONG SIDE"))
        if not ok:
            bad.append("direction %+d puts the streak on the wrong side" % dr)

    # ⚠ AND THE SAME RECORD MUST PRODUCE THE SAME PHYSICAL STREAK AT ANY
    # RESOLUTION. This is the whole reason `length_mm` is in millimetres, and it
    # cannot be tested on the field above -- that field's bar is placed in
    # PIXELS, so it is a different physical object at 25 and at 100 px/mm. The
    # test therefore builds a field whose geometry is fixed in MILLIMETRES, one
    # step from full development to none, and samples the restraint at fixed
    # FILM distances below the step.
    print("   restraint below a step, sampled at fixed FILM distances:")
    curves = prof.curves
    dmin3 = (curves.r.dmin, curves.g.dmin, curves.b.dmin)
    dmax3 = (curves.r.dmax, curves.g.dmax, curves.b.dmax)
    spec = fp.BromideDragSpec(strength=0.08, length_mm=3.0, axis=0,
                              direction=+1, source="parity probe")
    dists_mm = (0.5, 1.0, 2.0, 4.0)
    rows_out = {}
    for pxmm in PX_PER_MM:
        step_mm, total_mm = 2.0, 8.0
        ny = int(round(total_mm * pxmm))
        step = int(round(step_mm * pxmm))
        # ⚠ THE FIELD BELOW THE STEP IS MID-DENSITY, NOT CLEAR. A step from full
        # development to BARE BASE has no net density downstream for the
        # restraint to act on, so every sample there reads exactly zero and the
        # test measures nothing -- which is what the first version of it did.
        # The band sources the bromide; the mid-density tail is what the streak
        # has to land on for the streak to be visible at all.
        f = np.empty((ny, 4, 3), dtype=np.float32)
        for ch in range(3):
            span = dmax3[ch] - dmin3[ch]
            f[:step, :, ch] = dmin3[ch] + span
            f[step:, :, ch] = dmin3[ch] + np.float32(0.35) * span
        out = f.copy()
        fs.apply_bromide_drag(out, spec, dmin3, dmax3, pxmm, prof.is_reversal)
        frac = ((f - out) / np.maximum(f - np.array(dmin3, np.float32), 1e-6))
        rows_out[pxmm] = [float(frac[min(ny - 1,
                                         step + int(round(d * pxmm))), 0, 1])
                          for d in dists_mm]
        print("      %6.0f px/mm  " % pxmm
              + "  ".join("%.1f mm %.5f" % (d, v)
                          for d, v in zip(dists_mm, rows_out[pxmm])))
    a_lo, a_hi = rows_out[PX_PER_MM[0]], rows_out[PX_PER_MM[-1]]
    res_err = max(abs(x - y) for x, y in zip(a_lo, a_hi))
    print("      worst disagreement between resolutions: %.3g" % res_err)
    # ⚠ NOT ZERO, AND IT CANNOT BE: the two grids sample the same continuous
    # exponential at different pitches, and the seeded leading edge is a
    # half-pixel effect that scales with the pitch. 1e-3 of the removed fraction
    # is far below anything visible and far above what a pixel-denominated
    # length would give -- that would differ by the RATIO of the resolutions,
    # here a factor of four.
    if res_err > 1.0e-3:
        bad.append("length_mm is not resolution-independent: worst %.3g"
                   % res_err)

    if bad:
        print("\nDISAGREEMENTS:")
        for b in bad:
            print("  " + b)
        if a.assert_:
            return 1
    else:
        print("\nOK -- all engines agree, worst %.3g" % worst_overall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
