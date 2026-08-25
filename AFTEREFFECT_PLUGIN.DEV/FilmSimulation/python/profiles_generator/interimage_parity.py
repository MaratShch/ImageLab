"""Python vs C++ parity for the two DIR-coupler stages -- 8b and 9.

WHY THIS EXISTS
---------------
`cpp_parity.py` proves the Python and C++ **grain** and **MTF** laws agree. It
does not touch the two stages that carry the DIR-coupler chemistry, and those
are the largest COLOUR effect in the chain. Measured 2026-08-20, by disabling
them and re-rendering: up to **143/255** on Velvia's saturated patches, 23/255
on Portra's reds, 16/255 on Portra's off-anchor neutrals.

Both stages exist twice:

    vertical (inter-image)  film_sim.apply_interimage   <->  AlgoStage08b_Interimage
    lateral  (adjacency)    film_sim.apply_dir_couplers <->  AlgoStage09_DirCoupler

and until this file nothing compared them. That is exactly the configuration
that produced the C1b bug: one law, two languages, and a cross-check that was a
manual one-off from a finished session -- i.e. it guarded nothing.

⚠ THE TWO STAGES ARE NOT EQUALLY TESTABLE, and pretending otherwise would be
the whole point missed.

  * **Stage 8b is POINTWISE.** No spatial filter anywhere in it, so agreement is
    limited only by arithmetic precision and the two curve evaluators. This is
    checked at full strength on real profiles and is the load-bearing half of
    this file.
  * **Stage 9 CONTAINS TWO BLURS, and they are different implementations.**
    Python multiplies by the ANALYTIC Gaussian transfer in the frequency domain;
    C++ runs a separable spatial Gaussian with the kernel truncated at 4 sigma
    (`ALGO_BLUR_SIGMA_CUTOFF`). Both wrap at the edges, so the comparison is
    meaningful -- but only where the blur is RESOLVED; see the SCALES note. **On
    a FLAT field every blur is the identity**, so the pointwise algebra of
    stage 9 is exactly testable there, and that is the case pinned hardest.

⚠ AlgoType IS A SWITCHABLE TYPEDEF AND THIS FILE MUST NOT ASSUME IT.
`AlgoTypes.hpp` currently sets `using AlgoType = double`, deliberately, so the
owner can flip the whole renderer between 64-bit and 32-bit arithmetic in one
place. The probe therefore PRINTS `sizeof(AlgoType)` and this file picks its
tolerance from that at run time -- 2e-6 for an 8-byte type, 2e-3 for a 4-byte
one, because the Python reference carries its density planes in float32 and two
float32 pipelines that round in different orders diverge far more than a float32
and a float64 one do. Hard-coding a double tolerance would turn a future switch
to float into a spurious failure; hard-coding a float tolerance would blind the
check today.

⚠ ONE REAL DEFECT FOUND AND FIXED BY WRITING THIS FILE. The C++ stage 9 ends
with `MAX_VALUE(rO[x], ALGO_ZERO)` -- its own comment calls it "a physical
floor, not a display clamp". The Python side clamped one line LATER, inside
`simulate()`. So the two PIPELINES agreed and the two FUNCTIONS did not, and
nothing could see it until these functions were compared directly. It surfaced
as a **0.26 D** disagreement on Velvia -- a reversal stock whose ramp drives
density negative, already floored on one side and not yet on the other. The
floor now lives inside `apply_dir_couplers`, where its twin has it. Rendering is
unchanged: max(max(x,0),0) is max(x,0).

✅ THE ONE-SIDED GATE IS CLOSED (2026-08-25d, queue item C17). This file used to
record it as an open divergence: C++ gates BOTH coupler components on
`radiusPx >= ALGO_COUPLER_MIN_SIGMA_PX` (0.25 px, `AlgoDirCoupler.hpp:70`) and
the Python reference had no gate, so below that scale one renderer ran the stage
and the other did not. `apply_dir_couplers` now carries the SAME gate at the SAME
0.25 px. The threshold was adopted from the shipped C++ constant rather than
chosen, which keeps this a pure parity fix: no fidelity judgement is folded in.
The crossover line below is still printed, because the SCALE at which the stage
switches off is worth having as a number -- it is now a shared property of both
renderers instead of a divergence.

⚠ WHAT REMAINS OPEN IS QUEUE ITEM C16, AND IT IS A DIFFERENT QUESTION. The two
blurs are still different FORMS -- an analytic Gaussian transfer here, a
truncated separable spatial kernel there -- agreeing to 6e-5 only above about
1.2 px and diverging to 1.5e-1 at 0.4 px. Stored `edge_um` of 9-13 um is
0.36-0.60 px at 40 px/mm: inside that divergent band and ABOVE the gate. So the
shared threshold's VALUE (0.25 px, versus the ~1.0 px where the two forms
converge) is the open decision, and it changes every render.

Run:
    python interimage_parity.py
    python interimage_parity.py --assert
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

#: The plugin's own translation units. Not generated -- hand-written by the
#: owner, and that is the point: this probes the code that actually ships.
#: ⚠ MORE THAN THE TWO STAGE TUs, and the extras are not padding. `Algo_08_Sim`
#: needs `AlgoSoftplus` (defined in `Algo_05_Sim.cpp`) because the characteristic
#: curve is built from it; `Algo_09_Sim` also contains stage 9b
#: (`AlgoStage09b_NegativeDefects`) in the same TU, which drags in
#: `AlgoDefectField.cpp`. Neither is exercised by this probe -- they are link
#: dependencies of the file layout, not of the law under test.
PLUGIN_TUS = ("Algo_08_Sim.cpp", "Algo_09_Sim.cpp", "AlgoSeparableBlur.cpp",
              "Algo_05_Sim.cpp", "AlgoDefectField.cpp")

#: Stocks probed. Chosen for mechanism coverage, not popularity:
#:   PORTRA_400   strong DIR negative, density_weighting 0
#:   VELVIA_50    reversal -- the OTHER interimage mechanism, weighting 0.65,
#:                and the stock where the stage moves the most
#:   5219         cine negative, the strongest stored coefficients
#:   5250_1959    the "trace" tier, i.e. small coefficients where a sign error
#:                would be least visible and therefore most dangerous
#:   DOUBLE_X     monochrome -- both stages must be inert on it
STOCKS = ("KODAK_PORTRA_400", "FUJI_VELVIA_50", "KODAK_VISION3_500T_5219",
          "EASTMAN_5250_1959", "EASTMAN_DOUBLE_X_5222")

SIZE_X, SIZE_Y = 48, 40          # deliberately not a power of two
ANCHORS = (0.30, 0.25, 0.20)     # fixed; the anchor solve is a different stage
COUPLER_SCALE = 1.0

#: ⚠ TWO SCALES, AND ONLY ONE IS ASSERTED -- because the two BLUR
#: implementations stop being the same operator at SUB-PIXEL sigma. Measured
#: directly, FFT analytic transfer against the truncated separable kernel on one
#: ramp plane:
#:
#:      sigma_px   0.30   0.40   0.60   1.00   1.20   1.76   5.28   9.60
#:      worst err  1.4e-1 1.5e-1 4.3e-2 7.7e-4 6.4e-5 1.3e-6 2.2e-5 1.8e-5
#:
#: Above about 1.2 px they agree to 6e-5 and the choice of implementation does
#: not matter. Below 1 px they disagree by more than a tenth of a density, and
#: no tolerance can paper over that because it is not precision -- a Gaussian
#: narrower than the sample grid is not represented by either form.
#:   THE COUPLER EDGE TERM LIVES IN THAT ZONE IN NORMAL USE. Stored edge_um is
#: 9-13 um, so at 40 px/mm (a 35 mm frame rendered about 960 px wide) the edge
#: sigma is 0.36-0.52 px. So the adjacency half of the DIR chemistry is NOT the
#: same effect in the two renderers at ordinary render sizes -- up to 2.6e-2 D
#: apart, measured. That is a finding, not a tuning problem, and it is reported
#: rather than silenced.
#:   Hence: assert at a scale where every active sigma is resolved, and PROBE
#: the sub-pixel scale as information. Stage 8b, being pointwise, is asserted at
#: both.
SCALES = ((120.0, True), (40.0, False))   # (px per mm, assert this scale)

PROBE = r"""
// Generated by interimage_parity.py -- DO NOT EDIT.
// Probes the plugin's OWN stage 8b and stage 9 against the Python reference.
#include "AlgoInterimage.hpp"
#include "AlgoDirCoupler.hpp"
#include "AlgoCharacteristicCurve.hpp"
#include "LoadFilmDataBase.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

int main(void)
{
    if (!film::LoadFilmDataBase()) { std::printf("LOADFAIL\n"); return 2; }
    const auto& db = film::GetFilmDatabase();

    const int32_t sx = %(SX)d, sy = %(SY)d, pitch = %(SX)d;
    const size_t n = static_cast<size_t>(pitch) * static_cast<size_t>(sy);

    // sizeof is printed so the caller picks its tolerance from the ACTIVE type
    // instead of assuming one. AlgoType is a switchable typedef by design.
    std::printf("ALGOTYPE %%zu\n", sizeof(AlgoType));

    const char* want[] = { %(NAMES)s };
    const int   nWant = %(NWANT)d;
    const int   flatMode[] = { %(FLATS)s };
    const int   nMode = %(NMODE)d;
    const double pxmm[] = { %(PXMMS)s };
    const int    nScale = %(NSCALE)d;

    std::vector<AlgoType> sR(n), sG(n), sB(n), lR(n), lG(n), lB(n);
    std::vector<AlgoType> dR(n), dG(n), dB(n);
    std::vector<AlgoType> t1(n), t2(n), t3(n), t4(n);

    HighPrecType anchor[3] = { static_cast<HighPrecType>(%(A0).17g),
                               static_cast<HighPrecType>(%(A1).17g),
                               static_cast<HighPrecType>(%(A2).17g) };

    AlgoControls params;
    std::memset(&params, 0, sizeof(params));
    params.couplerScale = %(CPSCALE).17g;

    for (int w = 0; w < nWant; w++)
    {
        int idx = -1;
        for (size_t i = 0; i < db.size(); i++)
            if (std::string(want[w]) == std::string(db[i].name)) { idx = static_cast<int>(i); break; }
        if (idx < 0) { std::printf("MISSING %%s\n", want[w]); continue; }
        const film::FilmProfile& prof = db[idx];

        for (int sc = 0; sc < nScale; sc++)
        for (int mode = 0; mode < nMode; mode++)
        {
            const bool flat = (0 != flatMode[mode]);

            // Deterministic inputs, identical to the Python side by construction.
            for (int32_t y = 0; y < sy; y++)
            for (int32_t x = 0; x < sx; x++)
            {
                const size_t o = static_cast<size_t>(y) * pitch + x;
                const double u = flat ? 0.0
                                      : (static_cast<double>(x) / (sx - 1) * 2.0 - 1.0);
                const double v = flat ? 0.0
                                      : (static_cast<double>(y) / (sy - 1) * 2.0 - 1.0);
                lR[o] = static_cast<AlgoType>( 1.60 * u + 0.35 * v - 0.10);
                lG[o] = static_cast<AlgoType>( 1.45 * u - 0.30 * v + 0.05);
                lB[o] = static_cast<AlgoType>( 1.30 * u + 0.15 * v + 0.20);
            }

            // Stage 8's own output: density from the curve, reversal trim included.
            for (size_t o = 0; o < n; o++)
            {
                const AlgoType* le[3] = { lR.data(), lG.data(), lB.data() };
                AlgoType* sd[3] = { sR.data(), sG.data(), sB.data() };
                const film::ToneCurve* cv[3] = { &prof.curves.r, &prof.curves.g,
                                                 &prof.curves.b };
                for (int c = 0; c < 3; c++)
                {
                    const HighPrecType arg = prof.isReversal()
                        ? static_cast<HighPrecType>(-(le[c][o]
                              + static_cast<AlgoType>(anchor[c])))
                        : static_cast<HighPrecType>(le[c][o]);
                    sd[c][o] = static_cast<AlgoType>(
                        AlgoDensityScalar(arg, *cv[c]));
                }
            }

            AlgoStage08b_Interimage(sR.data(), sG.data(), sB.data(),
                                    dR.data(), dG.data(), dB.data(),
                                    lR.data(), lG.data(), lB.data(),
                                    t1.data(), t2.data(), t3.data(),
                                    sx, sy, pitch, prof, anchor);
            std::printf("S08B %%s %%d %%d\n", want[w], flat ? 1 : 0, sc);
            for (size_t o = 0; o < n; o++)
                std::printf("%%.17g %%.17g %%.17g\n",
                            static_cast<double>(dR[o]),
                            static_cast<double>(dG[o]),
                            static_cast<double>(dB[o]));

            // Stage 9 consumes stage 8b's output, as the pipeline does.
            AlgoStage09_DirCoupler(dR.data(), dG.data(), dB.data(),
                                   sR.data(), sG.data(), sB.data(),
                                   t1.data(), t2.data(), t3.data(), t4.data(),
                                   sx, sy, pitch, prof, params,
                                   static_cast<AlgoType>(pxmm[sc]));
            std::printf("S09 %%s %%d %%d\n", want[w], flat ? 1 : 0, sc);
            for (size_t o = 0; o < n; o++)
                std::printf("%%.17g %%.17g %%.17g\n",
                            static_cast<double>(sR[o]),
                            static_cast<double>(sG[o]),
                            static_cast<double>(sB[o]));
        }
    }
    std::printf("END\n");
    return 0;
}
"""


def build_probe(root: Path, tmp: Path) -> Path:
    """Compile the probe against the plugin's real TUs plus the generated data."""
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        raise SystemExit("[!] no g++/clang++ on PATH (set CXX to override)")
    src = tmp / "probe.cpp"
    src.write_text(PROBE % dict(
        SX=SIZE_X, SY=SIZE_Y,
        NAMES=", ".join('"%s"' % s for s in STOCKS), NWANT=len(STOCKS),
        FLATS="1, 0", NMODE=2,
        A0=ANCHORS[0], A1=ANCHORS[1], A2=ANCHORS[2],
        CPSCALE=COUPLER_SCALE,
        PXMMS=", ".join("%.17g" % s for s, _ in SCALES), NSCALE=len(SCALES),
    ), encoding="utf-8")

    gen = sorted(p.name for p in HERE.glob("film_profiles_data_*.cpp"))
    tus = ["probe.cpp"]
    tus += [str(root / t) for t in PLUGIN_TUS]
    tus += [str(HERE / "film_profiles.cpp"), str(HERE / "LoadFilmDataBase.cpp")]
    tus += [str(HERE / g) for g in gen]
    exe = tmp / "probe"
    cmd = [cxx, "-std=c++14", "-O1", "-I", str(root), "-I", str(HERE),
           "-o", str(exe)] + tus
    r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()
        raise SystemExit("[!] probe did not compile:\n  "
                         + "\n  ".join(tail[:12]))
    return exe


def parse(out: str):
    """-> (sizeof_algotype, {(stage, stock, flat): array})"""
    size = None
    blocks: dict[tuple[str, str, int], np.ndarray] = {}
    lines = out.splitlines()
    i = 0
    n = SIZE_X * SIZE_Y
    while i < len(lines):
        parts = lines[i].split()
        if not parts:
            i += 1
            continue
        if parts[0] == "ALGOTYPE":
            size = int(parts[1])
            i += 1
        elif parts[0] in ("S08B", "S09"):
            key = (parts[0], parts[1], int(parts[2]), int(parts[3]))
            vals = np.empty((n, 3), dtype=np.float64)
            for k in range(n):
                vals[k] = [float(v) for v in lines[i + 1 + k].split()]
            blocks[key] = vals.reshape(SIZE_Y, SIZE_X, 3)
            i += 1 + n
        elif parts[0] in ("END", "MISSING", "LOADFAIL"):
            if parts[0] != "END":
                print(f"  [!] probe said: {lines[i]}")
            i += 1
        else:
            i += 1
    return size, blocks


def python_side(name: str, flat: bool, px_per_mm: float):
    """The Python reference, on inputs generated by the same formula as C++."""
    p = fs.get_profile(name) if hasattr(fs, "get_profile") else \
        [q for q in fp.FILM_PROFILES if q.name == name][0]
    cv = p.curves.as_tuple()
    rev = p.is_reversal
    log_e = np.empty((SIZE_Y, SIZE_X, 3), np.float32)
    for y in range(SIZE_Y):
        for x in range(SIZE_X):
            u = 0.0 if flat else (x / (SIZE_X - 1) * 2.0 - 1.0)
            v = 0.0 if flat else (y / (SIZE_Y - 1) * 2.0 - 1.0)
            log_e[y, x, 0] = 1.60 * u + 0.35 * v - 0.10
            log_e[y, x, 1] = 1.45 * u - 0.30 * v + 0.05
            log_e[y, x, 2] = 1.30 * u + 0.15 * v + 0.20
    dens = np.empty_like(log_e)
    for c in range(3):
        arg = -(log_e[:, :, c] + np.float32(ANCHORS[c])) if rev else log_e[:, :, c]
        dens[:, :, c] = fs.density(arg, cv[c])
    d8b = dens.copy()
    if p.interimage.active and not p.is_monochrome:
        fs.apply_interimage(d8b, log_e, cv, p.interimage, ANCHORS, rev)
    grid = fs.FreqGrid(SIZE_Y, SIZE_X, px_per_mm)
    d9 = d8b.copy()
    fs.apply_dir_couplers(d9, p.couplers, grid, COUPLER_SCALE, p.is_monochrome)
    return d8b, d9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(HERE.parent.parent),
                    help="project root holding the plugin's Algo_*.cpp")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()

    missing = [t for t in PLUGIN_TUS if not (root / t).is_file()]
    if missing:
        print(f"  [SKIP] plugin sources not present under {root}: "
              f"{', '.join(missing)}")
        return 0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        exe = build_probe(root, tmp)
        r = subprocess.run([str(exe)], capture_output=True, text=True, cwd=tmp)
        if r.returncode != 0:
            print(f"  [FAIL] probe exited {r.returncode}: {r.stderr[:400]}")
            return 1
        size, blocks = parse(r.stdout)

    if size is None:
        print("  [FAIL] probe did not report sizeof(AlgoType)")
        return 1
    # ⚠ TOLERANCE FROM THE ACTIVE TYPE, never assumed. See the module note.
    tol = 2e-6 if size >= 8 else 2e-3
    print(f"[i] AlgoType is {size} bytes -> tolerance {tol:g} "
          f"(the Python reference carries density in float32 either way)")

    bad = 0
    worst_all = 0.0
    for si, (pxmm, do_assert_scale) in enumerate(SCALES):
        sig = [q.couplers.radius_um * 0.001 * pxmm
               for q in fp.FILM_PROFILES if q.couplers.active]
        edge = [q.couplers.edge_um * 0.001 * pxmm
                for q in fp.FILM_PROFILES
                if q.couplers.active and q.couplers.edge_um > 0]
        print(f"[i] scale {pxmm:g} px/mm: long-term sigma "
              f"{min(sig):.2f}-{max(sig):.2f} px, EDGE-term sigma "
              f"{min(edge):.2f}-{max(edge):.2f} px -- "
              + ("ASSERTED" if do_assert_scale
                 else "REPORTED ONLY: the edge sigma is SUB-PIXEL, where the "
                      "two blur implementations differ by ~1e-1 whatever the "
                      "tolerance"))
        for stock in STOCKS:
            for flat in (True, False):
                py8b, py9 = python_side(stock, flat, pxmm)
                for stage, py in (("S08B", py8b), ("S09", py9)):
                    key = (stage, stock, 1 if flat else 0, si)
                    if key not in blocks:
                        print(f"  [FAIL] {stage} {stock} flat={int(flat)}: "
                              f"no probe output")
                        bad += 1
                        continue
                    cpp = blocks[key]
                    d = float(np.max(np.abs(cpp - py.astype(np.float64))))
                    # a flat field makes every blur the identity, so stage 9's
                    # pointwise algebra is exactly testable there
                    lim = tol if (stage == "S08B" or flat) else max(tol, 1e-3)
                    hard = do_assert_scale or stage == "S08B" or flat
                    if hard:
                        worst_all = max(worst_all, d)
                    ok = d <= lim
                    if not ok and hard:
                        bad += 1
                    tag = "OK  " if ok else ("FAIL" if hard else "note")
                    print(f"  [{tag}] {stage:4s} {stock:24s} "
                          f"{'flat' if flat else 'ramp'}  worst {d:.3e} "
                          f"(limit {lim:g})")

    # ---- the shared sub-pixel gate, reported as a scale ---------------------
    # BOTH sides now disable each coupler component below 0.25 px (C17, closed
    # 2026-08-25d: the gate was C++-only until then). The crossover px/mm is
    # still printed, because the scale at which a stored radius stops being
    # rendered is worth stating -- it is a shared property now, not a divergence.
    act = [q for q in fp.FILM_PROFILES if q.couplers.active]
    if act:
        worst = max(act, key=lambda q: q.couplers.radius_um)
        thin = min(act, key=lambda q: min(q.couplers.edge_um or 1e9,
                                          q.couplers.radius_um))
        def crossover(um):
            return 0.25 / (um * 0.001) if um > 0 else float("inf")
        print(f"[i] SHARED gate ALGO_COUPLER_MIN_SIGMA_PX = 0.25 px: the long "
              f"term switches off below {crossover(worst.couplers.radius_um):.1f} "
              f"px/mm ({worst.name}, radius {worst.couplers.radius_um:.0f} um) and "
              f"the edge term below "
              f"{crossover(thin.couplers.edge_um):.1f} px/mm "
              f"({thin.name}, edge {thin.couplers.edge_um:.0f} um). BOTH renderers "
              f"now gate at this threshold (C17); the remaining C16 question is "
              f"the threshold's value, not its one-sidedness")

    print()
    if bad:
        print(f"[FAIL] {bad} disagreement(s), worst {worst_all:.3e}")
        return 1 if ns.do_assert else 0
    print(f"[OK] stages 8b and 9 agree between Python and the plugin's own C++ "
          f"-- worst {worst_all:.3e} over {len(STOCKS)} stocks x 2 fields x "
          f"{SIZE_X*SIZE_Y*3} values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
