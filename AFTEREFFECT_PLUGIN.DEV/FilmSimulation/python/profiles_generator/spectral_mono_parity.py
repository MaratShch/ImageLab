"""PYTHON <-> C++ PARITY for the monochrome spectral collapse weights.

WHY THIS EXISTS
---------------
`film_sim.spectral_monochrome_weights()` and `AlgoSpectralMonoWeights()` in
`AlgoSpectralSensitivity.cpp` implement one derivation: integrate a stock's
traced pan sensitivity against three primary lobes and normalise to sum 1.
The result is stage 7's channel mix for every monochrome stock, so a
disagreement is a visibly different B&W image from the two engines.

⚠ THIS AUDIT EXISTS BECAUSE THAT DISAGREEMENT WAS ALREADY SHIPPING, FOUND
2026-08-29. The C++ side calls the derivation unconditionally. The Python side
gated it behind `RenderSettings.spectral_mono`, which defaulted to **False**.
So for the 24 stocks carrying a traced pan curve the plugin derived and the
reference renderer did not, and the two rendered different monochrome images
for months. Worst case measured: `KODAK_PLUS_X_125`, blue weight 0.110 stored
against 0.502 derived -- a 4.6x difference on one channel of a black-and-white
film. Nothing caught it: `cpp_parity.py` audits the grain and MTF laws only,
and every visual check compares one engine against itself.

The flag now defaults to True and this file is the guard that keeps the two
sides together.

⚠ KNOWN OPEN FAILURE -- THE GAMUT-REACH GUARD IS PYTHON-ONLY. Python refuses
the derivation for a stock sensitised outside the basis's reach and falls back
to the authored triple; the C++ function has no such test and derives for
every stock that carries `log_s_pan`. `KONICA_INFRARED_750` (peak 750 nm,
0.437 of its energy past 700 nm) therefore derives in C++ to a BLUE-dominant
(0.161, 0.193, 0.646) against the authored, correct, red-dominant
(0.55, 0.15, 0.30). That is an ALGORITHM defect, not a data one, and fixing it
means editing `AlgoSpectralSensitivity.cpp`. This audit reports it as
`GUARD-GAP` and, with `--assert`, fails on it -- deliberately, so the gap
cannot be forgotten. Pass `--allow-guard-gap` to accept the known cases while
the fix is pending.

Run:
    python spectral_mono_parity.py --algodir ../tst
    python spectral_mono_parity.py --algodir ../tst --assert
Skips (exit 0) when the algorithm tree is not present, in the same way the
raster audits skip on an absent sheet.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import film_profiles as fp
import film_sim as fs

HERE = Path(__file__).resolve().parent

#: Both sides integrate in double and the C++ result is cast to AlgoType.
#: The scalar build's AlgoType is double, so the only spread is summation
#: order; 1e-9 is far below that and far above float64 noise on these
#: magnitudes. Measured worst case is ~1e-16.
TOL = 1e-9

#: Stocks where the Python guard refuses and C++ has no guard to refuse with.
#: Listing them is not an excuse -- it is what makes the gap countable.
#:
#: ⚠ EMPTIED 2026-08-30: queue C40 is closed. AlgoSpectralMonoWeights() now
#: carries the same two tests as the Python side -- peak sensitisation and
#: out-of-reach energy share, both measured on the profile's own stored samples
#: rather than the 360-730 nm render grid -- and refuses by returning false, so
#: Algo_07_Sim.cpp falls back to the authored triple exactly as film_sim does.
#: Measured after the change: 68 of 68 monochrome stocks agree, no gaps.
#: KONICA_INFRARED_750 no longer renders at a blue-dominant
#: (0.1611, 0.1931, 0.6458); both engines use its authored (0.55, 0.15, 0.30).
KNOWN_GUARD_GAP: tuple[str, ...] = ()

_PROG = r"""
#include "AlgoSpectralSensitivity.hpp"
#include "film_profiles.hpp"
#include <cstdio>

int main()
{
    const auto& db = film::GetFilmDatabase();
    for (const auto& p : db)
    {
        if (!p.is_monochrome)
            continue;
        AlgoType w[3] = { 0, 0, 0 };
        const bool ok = AlgoSpectralMonoWeights(p, w);
        std::printf("%s\t%d\t%.17g\t%.17g\t%.17g\n",
                    p.name.c_str(), ok ? 1 : 0,
                    static_cast<double>(w[0]),
                    static_cast<double>(w[1]),
                    static_cast<double>(w[2]));
    }
    return 0;
}
"""


def _cxx(algodir: Path, workdir: Path) -> Path | None:
    src = workdir / "mono_parity.cpp"
    src.write_text(_PROG, encoding="utf-8")
    exe = workdir / "mono_parity"
    cmd = ["g++", "-std=c++17", "-O1", "-o", str(exe), str(src),
           str(algodir / "AlgoSpectralSensitivity.cpp"),
           str(HERE / "film_profiles.cpp"),
           str(HERE / "LoadFilmDataBase.cpp")]
    cmd += [str(p) for p in sorted(HERE.glob("film_profiles_data_*.cpp"))]
    cmd += ["-I", str(algodir), "-I", str(HERE)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[FAIL] C++ build failed")
        print(r.stderr.strip()[:4000])
        return None
    return exe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algodir", default="../tst",
                    help="directory holding AlgoSpectralSensitivity.cpp")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ap.add_argument("--allow-guard-gap", action="store_true",
                    help="accept the known Python-only-guard stocks")
    args = ap.parse_args()

    algodir = Path(args.algodir)
    if not algodir.is_absolute():
        algodir = (HERE / algodir).resolve()
    needed = algodir / "AlgoSpectralSensitivity.cpp"
    if not needed.is_file():
        print("[SKIP] %s not present -- algorithm tree not staged" % needed)
        return 0

    with tempfile.TemporaryDirectory() as td:
        exe = _cxx(algodir, Path(td))
        if exe is None:
            return 1
        out = subprocess.run([str(exe)], capture_output=True, text=True)
        if out.returncode != 0:
            print("[FAIL] C++ probe exited %d" % out.returncode)
            return 1
        cpp = {}
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            name, ok, r, g, b = line.split("\t")
            cpp[name] = (ok == "1", (float(r), float(g), float(b)))

    profiles = {p.name: p for p in fp.FILM_PROFILES if p.is_monochrome}
    agree = 0
    gaps: list[str] = []
    bad: list[str] = []
    missing = sorted(set(profiles) - set(cpp))

    for name, p in profiles.items():
        if name not in cpp:
            continue
        c_ok, c_w = cpp[name]
        p_w = fs.spectral_monochrome_weights(p)
        p_ok = p_w is not None

        if p_ok and c_ok:
            worst = max(abs(a - b) for a, b in zip(p_w, c_w))
            if worst <= TOL:
                agree += 1
            else:
                bad.append("%-28s worst |dw| = %.3g  py=%s cpp=%s"
                           % (name, worst,
                              tuple(round(v, 4) for v in p_w),
                              tuple(round(v, 4) for v in c_w)))
        elif p_ok != c_ok:
            who = "C++ derives, Python refuses" if c_ok else \
                  "Python derives, C++ refuses"
            gaps.append("%-28s %s  cpp=%s authored=%s"
                        % (name, who, tuple(round(v, 4) for v in c_w),
                           tuple(round(float(v), 3)
                                 for v in p.spectral_weights)))
        else:
            agree += 1                       # both decline: same behaviour

    print("monochrome spectral collapse weights, Python vs C++")
    print("  monochrome stocks probed : %d" % len(cpp))
    print("  agreeing                 : %d" % agree)
    print("  numeric disagreements    : %d" % len(bad))
    print("  guard gaps               : %d" % len(gaps))
    for line in bad:
        print("    [DIFF] " + line)
    for line in gaps:
        print("    [GUARD-GAP] " + line)
    if missing:
        print("  ⚠ in Python but not in the C++ database: %s"
              % ", ".join(missing))

    unexpected = [ln for ln in gaps
                  if ln.split()[0] not in KNOWN_GUARD_GAP]
    if args.assert_:
        fail = bool(bad) or bool(missing) or bool(unexpected)
        if gaps and not args.allow_guard_gap:
            fail = True
            print("  ⚠ guard gaps present and --allow-guard-gap not given")
        if fail:
            print("[FAIL] monochrome weight parity")
            return 1
        # ⚠ THE ONE-LINE SUMMARY NAMES THE OPEN GAP OUT LOUD, EVERY RUN.
        # build.py shows an audit's last "[OK]" line and nothing else, so a
        # gap reported only in the body would be invisible in the build log --
        # which is how a known defect becomes a forgotten one. If the accepted
        # gap ever closes, this line stops mentioning it and the KNOWN_GUARD_GAP
        # list should be emptied.
        if gaps:
            print("[OK] %d/%d agree exactly; %d ACCEPTED GUARD GAP(S) STILL "
                  "OPEN: %s -- the gamut-reach guard is Python-only, the C++ "
                  "AlgoSpectralMonoWeights() has none (queue C40)"
                  % (agree, len(cpp), len(gaps),
                     ", ".join(ln.split()[0] for ln in gaps)))
        else:
            print("[OK] %d/%d monochrome stocks agree exactly, no guard gaps"
                  % (agree, len(cpp)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
