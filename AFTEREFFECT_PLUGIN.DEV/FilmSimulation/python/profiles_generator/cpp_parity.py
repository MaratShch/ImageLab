"""Python <-> C++ PARITY for the grain and MTF laws -- one definition, twice each.

WHY THIS EXISTS
---------------
`film_profiles.grain_sigma()` (Python, the reference renderer) and
`FilmGrainSigma()` (emitted into film_profiles.hpp, used by the plugin) implement
the same law. Two implementations of one law drift, and the way they drift is the
worst kind: both run, both look plausible, and the plugin renders different grain
from the reference for reasons nobody can see in a frame.

⚠ THIS SCRIPT EXISTS BECAUSE THAT RISK BECAME CONCRETE ON 2026-08-18. Queue item
C1b moved the law's normalisation point from ABSOLUTE density 1.0 to NET density
1.0 (= dmin + 1.0) -- the convention Kodak prints, "Read at a net diffuse visual
density of 1.0, using a 48-micrometre aperture" (5248 p1, 5222 p1). That change
had to be made in both languages, in different code (numpy interp vs a hand-rolled
insertion sort over four anchors), and the previous cross-check had been done ONCE
BY HAND in a session that is now over -- i.e. it protected nothing going forward.
The C++ header even carried a calling-convention instruction that C1b INVERTED:
callers used to be told to multiply by their own sqrt(D - dmin + fog) at D = 1.0,
which now double-counts. A hand check cannot catch that reappearing.

WHAT IT DOES
------------
Compiles a small program against the GENERATED header, walks the REAL database via
GetFilmDatabase(), and evaluates FilmGrainSigma() for every stock, on every
channel, over a density sweep -- then compares against Python evaluated on the same
stocks. Real data, both sides, no hand-written fixtures that could themselves drift
from the database.

The comparison is deliberately whole-database rather than a sample: the two laws
differ only for profiles with a measured shape (11 of 155) and only in the
interpolation, so a sample of "interesting" stocks is exactly the sample that
hides a bug in the other 144.

TOLERANCE: 2e-5 relative. The C++ side is float32 throughout and the Python side
computes in float64 before casting, so bit equality is not available; 2e-5 is two
orders of magnitude below any visible difference and two orders above float32
rounding on these magnitudes. Measured worst case is ~1e-7.

Run:
    python cpp_parity.py --assert     # non-zero exit if the two disagree
Needs g++ (C++14) and the generated film_profiles.hpp/.cpp next to this file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import film_profiles as fp

HERE = Path(__file__).resolve().parent

#: Densities probed per stock per channel. Includes the two reference points the
#: convention turns on (net 1.0 and absolute 1.0 -- equal only for an unmasked
#: stock), the toe, deep shadow, and well past dmax so the flat hold is exercised.
PROBES = ("dmin", "net1", "abs1", "toe+0.1", "0.0", "0.5", "1.5", "2.5", "dmax", "dmax+2")

#: GrainSpec field order in the GENERATED header, verified against it at run time.
#: Aggregate initialisation is positional, so a field inserted upstream without
#: updating this list would silently shift every value -- hence the check.
MTF_FIELDS = ("f50_r", "f50_g", "f50_b", "adjacency", "adjacency_um",
              "resolving_power_lp_mm_lowc", "resolving_power_lp_mm_highc",
              "mtf_rolloff_q", "mtf_measured", "mtf_tail_a", "mtf_tail_f_exp")

FIELDS = ("rms_granularity", "clump_um_r", "clump_um_g", "clump_um_b",
          "clump_gain", "fog_grain", "anisotropy", "rms_r", "rms_g", "rms_b",
          "sigma_shape_toe", "sigma_shape_mid", "sigma_shape_dmax",
          "sigma_shape_peak", "sigma_shape_peak_at", "sigma_shape_toe_at",
          "sigma_shape_dmax_at", "sigma_shape_measured", "size_sigma_log",
          "cluster_um", "dye_cloud_um")

CPP_HEAD = r"""
#include "film_profiles.hpp"
#include <cstdio>

using namespace film;

struct Probe { GrainSpec g; float dmin, dmax, D; const char* name; int ch; int k; };
struct MProbe { MTFSpec m; int ch; float f; const char* name; int k; };

static const Probe PROBES[] = {
"""

CPP_MID = r"""
};

static const MProbe MPROBES[] = {
"""

CPP_TAIL = r"""
};

int main()
{
    const int n = (int)(sizeof(PROBES)/sizeof(PROBES[0]));
    for (int i = 0; i < n; ++i) {
        const Probe& p = PROBES[i];
        printf("G\t%s\t%d\t%d\t%.9g\n", p.name, p.ch, p.k,
               (double)FilmGrainSigma(p.g, p.dmin, p.dmax, p.D));
    }
    const int mn = (int)(sizeof(MPROBES)/sizeof(MPROBES[0]));
    for (int i = 0; i < mn; ++i) {
        const MProbe& p = MPROBES[i];
        printf("M\t%s\t%d\t%d\t%.9g\n", p.name, p.ch, p.k,
               (double)FilmMtfResponse(p.m, p.ch, p.f));
    }
    return 0;
}
"""

TOL = 2e-5


def build_and_run(tmp: Path, probes) -> dict:
    """Compile a probe program against the generated header and collect its output.

    ⚠ THE INPUTS ARE PASSED IN AS LITERALS, NOT READ FROM THE GENERATED DATABASE,
    and that is a correction. The first version walked `GetFilmDatabase()` on the
    C++ side and the Python profiles on the Python side, then compared. It found a
    real disagreement of 1.5e-02 the first time a CURVE was re-traced -- and the
    disagreement was not in the law at all: `build.py` runs this audit BEFORE
    codegen, so the C++ copy still held the previous dmax while Python held the
    new one. Two implementations of one law were being fed different data and
    blamed for it. Feeding both sides identical GrainSpec fields, dmin, dmax and
    density tests THE FUNCTION, which is what this audit is for; data freshness is
    already covered by build.py's sync stage and by verify.py.
    """
    def lit(v):
        # ⚠ "12f" IS NOT A C++ FLOAT LITERAL. %.9g drops the decimal point on whole
        # numbers and g++ then reports 'unable to find numeric literal operator
        # operator""f' -- a confusing way to be told that a clump diameter happened
        # to be 12.0. Force a decimal point.
        if v is True:
            return "true"
        if v is False:
            return "false"
        t = f"{float(v):.9g}"
        if "." not in t and "e" not in t and "E" not in t:
            t += ".0"
        return t + "f"

    lines = []
    for (name, ch, k, spec, dmin, dmax, D) in probes[0]:
        vals = ", ".join(lit(v) for v in spec)
        lines.append(f'    {{ {{{vals}}}, {lit(dmin)}, {lit(dmax)}, {lit(D)}, '
                     f'"{name}", {ch}, {k} }},')
    mlines = []
    for (name, ch, k, spec, f) in probes[1]:
        vals = ", ".join(lit(v) for v in spec)
        mlines.append(f'    {{ {{{vals}}}, {ch}, {lit(f)}, "{name}", {k} }},')
    src = tmp / "parity.cpp"
    src.write_text(CPP_HEAD + "\n".join(lines) + CPP_MID
                   + "\n".join(mlines) + CPP_TAIL)
    exe = tmp / "parity"
    # Only the header is needed now: no database translation units, because the
    # probe carries its own inputs. That also makes this audit ~20x faster.
    cmd = ["g++", "-std=c++14", "-O1", "-I", str(HERE), "-o", str(exe), str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] compile failed")
        print(r.stderr[-4000:])
        raise SystemExit(2)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] probe crashed")
        print(r.stderr[-2000:])
        raise SystemExit(2)
    out = {}
    for line in r.stdout.splitlines():
        fam, nm, c, k, v = line.split("\t")
        out[(fam, nm, int(c), int(k))] = float(v)
    return out


def check_field_order() -> None:
    """The literal initialiser is positional -- verify FIELDS still matches."""
    import re as _re
    hdr = (HERE / "film_profiles.hpp").read_text()
    i = hdr.index("struct GrainSpec {")
    blk = hdr[i:hdr.index("};", i)]
    got = tuple(m[1] for m in _re.findall(r"^\s*(float|bool)\s+(\w+);", blk, _re.M))
    if got != FIELDS:
        raise SystemExit("[!] GrainSpec field order changed in the generated "
                         f"header:\n    header: {got}\n    expected: {FIELDS}")


def probe_table():
    """Two probe families: the grain law and the MTF law.

    ⚠ BOTH LAWS ARE PROBED because both are now duplicated in two languages. C1
    wired grain_sigma() and C2 wired mtf_response(); each has a hand-written C++
    twin in the generated header, and each twin can drift silently. The MTF probe
    deliberately includes frequencies FAR past f50 (up to 6x), because that is
    exactly where the measured power law and the legacy Gaussian diverge -- a
    parity check that only sampled the mid band would pass on a twin that had the
    wrong law.
    """
    grain, mtf = [], []
    for p in fp.FILM_PROFILES:
        gspec = tuple(getattr(p.grain, f) for f in FIELDS)
        for c, cur in enumerate((p.curves.r, p.curves.g, p.curves.b)):
            dmin, dmax = float(cur.dmin), float(cur.dmax)
            for k, D in enumerate([dmin, dmin + 1.0, 1.0, dmin + 0.1, 0.0,
                                   0.5, 1.5, 2.5, dmax, dmax + 2.0]):
                grain.append((p.name, c, k, gspec, dmin, dmax, D))
        mspec = tuple(getattr(p.mtf, f) for f in MTF_FIELDS)
        for c in range(3):
            f50 = p.mtf.f50s()[c]
            for k, mult in enumerate((0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 6.0)):
                mtf.append((p.name, c, k, mspec, f50 * mult))
    return grain, mtf


def python_side(probes) -> dict:
    """The same probes, evaluated through the Python reference laws."""
    out = {}
    for (name, ch, k, _spec, dmin, dmax, D) in probes[0]:
        prof = fp.get_profile(name)
        out[("G", name, ch, k)] = float(
            fp.grain_sigma(prof.grain, dmin, dmax, D))
    for (name, ch, k, _spec, f) in probes[1]:
        prof = fp.get_profile(name)
        out[("M", name, ch, k)] = float(fp.mtf_response(prof.mtf, ch, float(f)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()

    if not (HERE / "film_profiles.hpp").is_file():
        print("  [SKIP] cpp_parity: film_profiles.hpp not generated yet")
        return 0

    check_field_order()
    probes = probe_table()
    with tempfile.TemporaryDirectory() as td:
        cpp = build_and_run(Path(td), probes)
    py = python_side(probes)

    missing = sorted(set(py) - set(cpp))
    extra = sorted(set(cpp) - set(py))
    if missing or extra:
        # ⚠ SAY WHICH FILE IS STALE, because this failure has one common cause and
        # a misleading default message. `build.py` runs the audit stage BEFORE
        # codegen, so on the run that ADDS a stock to film_profiles.py the
        # generated C++ still holds the old database and every probe for the new
        # stock is "missing". That is the correct signal -- the artefacts are stale
        # -- but "probe sets differ" reads like a bug in this script. Naming the
        # absent stocks and the remedy makes the next run obvious.
        py_names = {k[1] for k in py}
        cpp_names = {k[1] for k in cpp}
        only_py = sorted(py_names - cpp_names)
        only_cpp = sorted(cpp_names - py_names)
        print(f"[FAIL] probe sets differ: {len(missing)} missing, {len(extra)} extra")
        if only_py:
            print(f"       {len(only_py)} stock(s) in film_profiles.py but NOT in the "
                  f"generated C++: {', '.join(only_py[:6])}")
            print( "       -> the generated database is STALE. Re-run codegen "
                   "(build.py regenerates it in the stage AFTER this one, so a "
                   "second build.py run clears this).")
        if only_cpp:
            print(f"       {len(only_cpp)} stock(s) in the generated C++ but no "
                  f"longer in film_profiles.py: {', '.join(only_cpp[:6])}")
        for k in (missing[:3] + extra[:3]):
            print("   ", k)
        return 1 if ns.do_assert else 0

    worst, worst_at = 0.0, None
    n_meas = 0
    for k, want in py.items():
        got = cpp[k]
        scale = max(abs(want), 1e-6)
        err = abs(got - want) / scale
        if err > worst:
            worst, worst_at = err, k
        n_meas += 1

    # ⚠ A PARITY CHECK THAT ONLY COMPARES IS NOT ENOUGH: if both sides silently
    # returned the legacy law everywhere, they would agree perfectly and the
    # measured shapes would be dead. So assert the probe actually EXERCISED the
    # measured branch, by requiring the 11 flagged stocks to differ from the
    # legacy law at their own peak density.
    # ⚠ THE FIRST VERSION OF THIS TEST PROBED EACH STOCK AT ITS STORED INTERIOR
    # PEAK and expected 11 hits. It got 10, and the missing stock was not a bug:
    # KODAK_EKTACHROME_100D_5285 is a REVERSAL film whose sigma(D) rises
    # monotonically to dmax, so its maximum IS the dmax anchor and
    # sigma_shape_peak is legitimately 0. Probing "the peak" therefore skipped
    # the one stock whose shape is least like the legacy law. Probe the traced
    # dmax anchor instead -- every measured stock has one, by construction.
    exercised = 0
    for p in fp.FILM_PROFILES:
        g, c = p.grain, p.curves.g
        if not g.sigma_shape_measured:
            continue
        D = g.sigma_shape_dmax_at or c.dmax
        legacy = (float(np.sqrt(max(D - c.dmin, 0.0) + g.fog_grain))
                  / float(np.sqrt(1.0 + g.fog_grain)))
        if abs(fp.grain_sigma(g, c.dmin, c.dmax, D) - legacy) > 0.05:
            exercised += 1

    print(f"[i] {n_meas} probes over {len(fp.FILM_PROFILES)} stocks: "
          f"{len(probes[0])} grain (3 channels x 10 densities) + "
          f"{len(probes[1])} MTF (3 channels x 8 frequencies to 6x f50)")
    print(f"[i] measured-shape branch exercised on {exercised} stocks")
    print(f"[i] worst relative disagreement {worst:.2e} at {worst_at}")

    bad = 0
    if worst > TOL:
        print(f"[FAIL] Python and C++ grain laws disagree by {worst:.2e} "
              f"(tolerance {TOL:.0e}) at {worst_at}")
        bad += 1
    if exercised < 11:
        print(f"[FAIL] only {exercised} of 11 measured stocks differ from the "
              f"legacy law -- the probe is not testing what it claims to")
        bad += 1
    if bad:
        return 1 if ns.do_assert else 0
    print("[OK] the Python and C++ grain and MTF laws agree on the whole database")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
