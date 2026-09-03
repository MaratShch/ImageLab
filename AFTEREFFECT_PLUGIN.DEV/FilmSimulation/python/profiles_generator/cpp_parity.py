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
import re
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

#: Reciprocity is compared in DECADES OF LOG EXPOSURE, absolutely, not relatively
#: -- the correct answer is exactly 0.0 for most stocks at most times, and a
#: relative tolerance against zero is meaningless. 1e-6 decades is 3e-6 of a
#: stop, i.e. far below anything sensitometric, and it is loose enough to absorb
#: the one real difference between the two sides: the generated C++ struct stores
#: the Schwarzschild exponents as FLOAT, so 0.87 arrives as 0.869999...
TOL_RECIP = 1e-6

#: Exposure times probed per stock, seconds. Deliberately two-sided and wider
#: than any measured table: 1e-5 s is inside a strobe, 3600 s is an hour, and
#: both are outside every table on file -- which is the point, because the law
#: HOLDS FLAT outside the measured range rather than extrapolating and the two
#: implementations must hold flat identically. 0.0 is the inertness test.
RECIP_TIMES = (0.0, 1e-5, 1e-4, 1e-3, 0.02, 0.1, 0.5, 1.0, 2.0, 10.0, 60.0, 3600.0)

#: ReciprocitySpec field order in the generated header. Positional initialisation
#: again, same hazard as GrainSpec, same check.
RECIP_FIELDS = ("schwarzschild_p_r", "schwarzschild_p_g", "schwarzschild_p_b",
                "onset_s")


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


RECIP_CPP = r"""
// Reciprocity parity probe. Generated by cpp_parity.py -- do not edit.
//
// ⚠ THE INPUTS ARE LITERALS, not a walk of GetFilmDatabase(), and they stay
// literals even though build.py now runs codegen and sync BEFORE this audit
// (reordered 2026-08-24). The original reason was ordering: reading the
// generated database would have compared fresh Python data against a stale C++
// table and blamed the law for it. The reason it is still right is better --
// literals isolate the LAW from the TABLE, so this probe fails only when the
// arithmetic differs, never because a number moved.
#include "AlgoReciprocity.hpp"
#include <cstdio>

using namespace film;

struct RProbe {
    const char*  name;
    int          k;             // index into the time ladder
    double       t;             // seconds
    float        pr, pg, pb, onset;
    int          n;             // table length, 0 = spec branch
    const double* times;
    const double* stops;
    const char* const* ccs;
};

/*ARRAYS*/

static const RProbe RPROBES[] = {
/*ROWS*/
};

int main()
{
    const int n = (int)(sizeof(RPROBES)/sizeof(RPROBES[0]));

    for (int i = 0; i < n; ++i)
    {
        const RProbe& p = RPROBES[i];

        // A default-constructed profile carries empty vectors and zeroed PODs;
        // only the two reciprocity members are filled, because they are the only
        // ones the law reads. Anything else this probe set could depend on would
        // be a coupling the law should not have.
        FilmProfile prof = FilmProfile();

        prof.reciprocity.schwarzschild_p_r = p.pr;
        prof.reciprocity.schwarzschild_p_g = p.pg;
        prof.reciprocity.schwarzschild_p_b = p.pb;
        prof.reciprocity.onset_s           = p.onset;

        for (int j = 0; j < p.n; ++j)
        {
            prof.reciprocity_table.times_s.push_back(p.times[j]);
            prof.reciprocity_table.stops_correction.push_back(p.stops[j]);

            if (p.ccs != 0)
                prof.reciprocity_table.cc_filters.push_back(std::string(p.ccs[j]));
        }

        HighPrecType s[3] = { 0.0, 0.0, 0.0 };

        AlgoReciprocityLogShift(prof, static_cast<HighPrecType>(p.t), s);

        for (int c = 0; c < 3; ++c)
            printf("R\t%s\t%d\t%d\t%.17g\n", p.name, c, p.k, (double)s[c]);
    }

    return 0;
}
"""


def recip_probe_table():
    """One row per (stock, exposure time). Carries the stock's own table inline."""
    rows = []
    for p in fp.FILM_PROFILES:
        spec = tuple(float(getattr(p.reciprocity, f)) for f in RECIP_FIELDS)
        tab = p.reciprocity_table
        table = (tuple(float(v) for v in tab.times_s),
                 tuple(float(v) for v in tab.stops_correction),
                 tuple(str(v) for v in tab.cc_filters))
        for k, t in enumerate(RECIP_TIMES):
            rows.append((p.name, k, float(t), spec, table))
    return rows


def recip_build_and_run(tmp: Path, root: Path, rows) -> dict:
    """Compile the reciprocity probe against the PLUGIN'S OWN header and run it."""
    arrays, seen = [], {}
    for (name, _k, _t, _spec, table) in rows:
        if name in seen:
            continue
        seen[name] = len(seen)
        i = seen[name]
        times, stops, ccs = table
        if not times:
            continue
        arrays.append("static const double RT_%d[] = { %s };"
                      % (i, ", ".join(f"{v:.17g}" for v in times)))
        arrays.append("static const double RS_%d[] = { %s };"
                      % (i, ", ".join(f"{v:.17g}" for v in stops)))
        if ccs:
            # Padded to the table length: a shorter cc_filters vector is legal
            # (it means the later times are achromatic) and the C++ side handles
            # that by length, but the probe's array must not be read past its end.
            pad = list(ccs) + [""] * (len(times) - len(ccs))
            arrays.append('static const char* const RC_%d[] = { %s };'
                          % (i, ", ".join('"%s"' % c for c in pad)))
    lines = []
    for (name, k, t, spec, table) in rows:
        i = seen[name]
        times, _s, ccs = table
        n = len(times)
        tp = f"RT_{i}" if n else "0"
        sp = f"RS_{i}" if n else "0"
        cp = f"RC_{i}" if (n and ccs) else "0"
        # ⚠ "1f" IS NOT A C++ FLOAT LITERAL -- the same trap documented in
        # build_and_run's lit(): %.9g drops the decimal point on a whole number
        # and g++ then reports 'unable to find numeric literal operator'. An
        # exponent p of exactly 1.0 is the COMMON case here (54 inert stocks), so
        # this bites immediately rather than rarely.
        def _f(v):
            t = f"{float(v):.9g}"
            if ("." not in t) and ("e" not in t) and ("E" not in t):
                t += ".0"
            return t + "f"
        lines.append('    { "%s", %d, %.17g, %s, %s, %s, %s, %d, %s, %s, %s },'
                     % (name, k, t, _f(spec[0]), _f(spec[1]), _f(spec[2]),
                        _f(spec[3]), n, tp, sp, cp))
    src = tmp / "recip_parity.cpp"
    src.write_text(RECIP_CPP.replace("/*ARRAYS*/", "\n".join(arrays))
                            .replace("/*ROWS*/", "\n".join(lines)))
    exe = tmp / "recip_parity"
    cmd = ["g++", "-std=c++14", "-O1", "-I", str(root), "-I", str(HERE),
           "-o", str(exe), str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] reciprocity probe compile failed")
        print(r.stderr[-4000:])
        raise SystemExit(2)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] reciprocity probe crashed")
        print(r.stderr[-2000:])
        raise SystemExit(2)
    out = {}
    for line in r.stdout.splitlines():
        fam, nm, c, k, v = line.split("\t")
        out[(fam, nm, int(c), int(k))] = float(v)
    return out


def recip_python_side(rows) -> dict:
    """The same rows through film_sim.reciprocity_log_shift(), the reference."""
    import film_sim as fs
    out = {}
    for (name, k, t, _spec, _table) in rows:
        prof = fp.get_profile(name)
        s = fs.reciprocity_log_shift(prof, t)
        for c in range(3):
            out[("R", name, c, k)] = float(s[c])
    return out


# ---------------------------------------------------------------------------
#  C22: Callier's coefficient -- the FOURTH probe family.
#
#  Compiled against the PLUGIN'S OWN AlgoCallier.hpp, like the reciprocity
#  family and for the same reason: the law lives in the renderer, not in the
#  generated table, so this is the only place the two definitions meet.
#
#  ⚠ WHAT THIS FAMILY IS REALLY GUARDING is not the arithmetic, which is three
#  operations, but the dmin REFERENCE. `D_read = dmin + (D - dmin) * k` and
#  `D_read = D * k` agree exactly at D = dmin and diverge everywhere else, and
#  the wrong one darkens clear film base. So the probe deliberately includes
#  densities BELOW dmin (grain can produce them) and at dmin itself.
# ---------------------------------------------------------------------------
# ⚠ 1e-9 WAS TOO TIGHT AND THE REASON IS WORTH RECORDING RATHER THAN JUST THE
# NUMBER. The law itself is three exact operations, so the first run of this
# family expected bit agreement -- and got 1.43e-07 on AGFA_APX_100. The cause is
# STORAGE, not arithmetic: `FilmProfile::callier_q` is a `float`, so a stored 1.3
# reaches the C++ side as 1.2999999523, while film_sim reads the Python double
# 1.3. At specular 0.6 and 3.0 D above dmin that is 3.0 * 0.6 * 4.8e-8 = 8.6e-8,
# which is exactly the residual observed. It scales with density and with
# specular, so the widest probe in the ladder sets it. 1e-6 leaves two decades of
# headroom over that and still catches any real divergence in the law -- a
# reference-to-zero bug, for instance, would show up at 0.1-1.0, six decades up.
#
# ⚠ 1e-6 -> 1e-5 ON 2026-09-02, AND THE CAUSE IS QUEUE C43 RATHER THAN A NEW
# DEFECT. `callier_q` stopped being the class constant 1.3 and became a
# per-stock beta of 1.64-1.87, and the dominant residual is no longer float
# storage of q -- it is LINEAR INTERPOLATION IN THE SHARED LUT, whose error
# scales with the law's curvature and therefore with beta:
#
#     q = 1.3000   worst |LUT - exact|  2.2e-07   (the old floor)
#     q = 1.8408   worst |LUT - exact|  1.7e-06   (measured, 8x worse)
#
# over CALLIER_LUT_MIN..MAX in 1025 samples, at every intermediate specular
# setting. The observed parity residual, 1.41e-06 at POLAROID_51, is that number
# and not a divergence between the two implementations of the law -- float
# storage of q alone accounts for at most 2.3e-07 (measured the same way).
#
# ⚠ WHY THE TOLERANCE IS RAISED RATHER THAN THE LUT REFINED, with the cost of
# the alternative measured rather than guessed: doubling ALGO_CALLIER_LUT_N to
# 2049 quarters the error to about 4.4e-07 and would restore the old bound, at
# 16 KB per LUT in the scalar build where AlgoType is double. 1.7e-06 D is three
# orders of magnitude below one 16-bit code step, so the refinement buys nothing
# a viewer could see, and enlarging a struct that may live on the stack is not a
# change to make as the last item of a long batch. The number is recorded here
# so the next reader can act on it with the arithmetic already done.
TOL_CALLIER = 1e-5

#: specular settings: 0 must be exactly inert, 1 is the full condenser, and the
#: two interior values catch a factor applied as (1 + s)*(Q - 1) or similar.
CALLIER_SPECULAR = (0.0, 0.25, 0.6, 1.0)

#: probe densities, as OFFSETS from the channel's own dmin. The negative entry
#: is the one that distinguishes the dmin-referenced law from a clamped branch.
CALLIER_DELTA = (-0.20, 0.0, 0.15, 0.75, 1.60, 3.00)

CALLIER_CPP = r"""
// Callier parity probe. Generated by cpp_parity.py -- do not edit.
//
// Inputs are LITERALS, not a walk of GetFilmDatabase(): build.py runs this audit
// BEFORE codegen, so reading the generated database would compare fresh Python
// data against a stale C++ table and blame the law for it.
#include "AlgoCallier.hpp"
#include <cstdio>

using namespace film;

struct CProbe {
    const char* name;
    int         c;        // channel
    int         is;       // index into the specular ladder
    int         id;       // index into the density ladder
    double      spec;
    double      d;        // the density presented to the law
    double      dmin;     // that channel's own dmin
    float       q;
};

static const CProbe CPROBES[] = {
/*ROWS*/
};

int main()
{
    const int n = (int)(sizeof(CPROBES)/sizeof(CPROBES[0]));

    for (int i = 0; i < n; ++i)
    {
        const CProbe& p = CPROBES[i];

        // Only callier_q is filled: it is the only member the law reads, and any
        // other dependency would be a coupling the law should not have.
        FilmProfile prof = FilmProfile();
        prof.callier_q = p.q;

        // ⚠ THE PROBE MOVED WITH THE LAW (M3). It used to read back a
        // multiplier; there is no multiplier any more, so it reads back the NET
        // density the law produces and the dmin-referenced result, which is
        // what both consumers actually use.
        const HighPrecType q = (HighPrecType)prof.callier_q;
        const HighPrecType f =
            ((HighPrecType)p.spec <= 0.0 || q == 1.0)
                ? ((HighPrecType)p.d - (HighPrecType)p.dmin)
                : AlgoCallierNet((HighPrecType)p.d - (HighPrecType)p.dmin,
                                 q, (HighPrecType)p.spec);
        const HighPrecType r = AlgoCallierApplyScalar((HighPrecType)p.d,
                                                      (HighPrecType)p.dmin,
                                                      q, (HighPrecType)p.spec);

        printf("C\t%s\t%d\t%d\t%d\t%.17g\t%.17g\n",
               p.name, p.c, p.is, p.id, (double)f, (double)r);
    }

    return 0;
}
"""


# ===========================================================================
#  STAGE-LEVEL CALLIER PROBE -- added 2026-08-30 (queue C41).
#
#  ⚠ THE LAW FAMILY ABOVE PASSED FOR A WEEK WHILE NOTHING CALLED THE LAW. That
#  is the same shape of hole the grain probe was written for: the law
#  and AlgoCallierApplyScalar agreed with Python to 1.4e-07 on 11592 probes,
#  and the pipeline did not invoke either of them. A parity check must exercise
#  the CODE THAT RENDERS.
#
#  So this drives the two things C41 actually wired:
#
#    STAGE   AlgoStage12b_Callier on a real plane, recovering what it wrote.
#    SOLVE   AlgoSolveAnchors with a non-zero scannerSpecular, against
#            film_sim.solve_anchors with the same setting.
#
#  ⚠ THE SOLVE HALF IS THE ONE THAT MATTERS AND IT IS WHY THIS PROBE EXISTS AT
#  ALL. Wiring the pixel pass without the solve is a documented regression --
#  mid grey moves by more than the contrast does -- so a probe that checked only
#  the stage would pass on exactly the broken configuration this task was
#  approved to prevent.
# ===========================================================================

CALLIER_STAGE_CPP = r"""
#include "AlgoCallier.hpp"
#include "AlgoCharacteristicCurve.hpp"
#include "film_profiles.hpp"
#include <cstdio>

struct SRow { const char* name; int is; double spec; };

static const SRow SROWS[] = {
/*ROWS*/
};

int main()
{
    const auto& db = film::GetFilmDatabase();
    const int   n  = (int)(sizeof(SROWS)/sizeof(SROWS[0]));

    for (int i = 0; i < n; ++i)
    {
        const SRow& r = SROWS[i];

        const film::FilmProfile* p = nullptr;
        for (const auto& q : db)
            if (q.name == r.name) { p = &q; break; }
        if (nullptr == p)
            continue;

        // ---- STAGE. Four pixels at known densities, one per channel plane.
        const int W = 4, H = 1, P = 4;
        AlgoType dR[4], dG[4], dB[4];
        const AlgoType dmin[3] = {
            (AlgoType)p->curves.r.dmin,
            (AlgoType)p->curves.g.dmin,
            (AlgoType)p->curves.b.dmin };

        // Net densities either side of the reference, so a wrong dmin
        // reference shows up as a slope error rather than a constant.
        const double net[4] = { 0.0, 0.25, 1.0, 2.0 };
        for (int x = 0; x < 4; ++x) {
            dR[x] = (AlgoType)(p->curves.r.dmin + net[x]);
            dG[x] = (AlgoType)(p->curves.g.dmin + net[x]);
            dB[x] = (AlgoType)(p->curves.b.dmin + net[x]);
        }

        AlgoStage12b_Callier(dR, dG, dB, W, H, P, dmin, *p,
                             (HighPrecType)r.spec);

        for (int x = 0; x < 4; ++x)
            printf("CS\t%s\t%d\t%d\t%.17g\t%.17g\t%.17g\n",
                   r.name, r.is, x,
                   (double)dR[x], (double)dG[x], (double)dB[x]);

        // ---- SOLVE. Reversal stocks take a null print stock, as the caller
        // does; negatives take the database's own default print.
        const film::PrintStock* ps = nullptr;
        if (!p->isReversal()) {
            const auto& stocks = film::GetPrintStocks();
            for (const auto& s : stocks)
                if (s.name == p->default_print) { ps = &s; break; }
            if (nullptr == ps && !stocks.empty())
                ps = &stocks[0];
        }

        HighPrecType anchor[3] = { 0, 0, 0 };
        AlgoSolveAnchors(*p, ps, (HighPrecType)0.18, (HighPrecType)1.0,
                         (HighPrecType)r.spec, anchor);

        printf("CA\t%s\t%d\t0\t%.17g\t%.17g\t%.17g\n",
               r.name, r.is,
               (double)anchor[0], (double)anchor[1], (double)anchor[2]);
    }

    return 0;
}
"""


# ===========================================================================
#  RECIPROCITY, STAGE LEVEL
#
#  ⚠ THE LAW FAMILY ABOVE PASSED FOR MONTHS WHILE NOTHING CALLED THE LAW, AND
#  THAT IS NOT A HYPOTHETICAL -- IT IS WHAT HAPPENED. `AlgoReciprocity.hpp` was
#  written, documented, parity-tested against film_sim over 6120 probes, and
#  included by NO translation unit. The C++ engine therefore had no reciprocity
#  model at all while film_sim applied one on every render, and the law family
#  reported agreement the whole time because it compiled the header itself.
#  The Callier families learned the same lesson a week earlier; this is the
#  guard that stops the third repetition.
#
#  So this family does not call the law. It drives the REAL
#  AlgoStage08_CharacteristicCurve on a real exposure plane, at a stated
#  exposure time, and compares the densities it writes against film_sim's own
#  stage 8 with the same time. If the shift is ever unwired from the stage
#  again, the arithmetic stays right and THIS fails.
#
#  It also pins the inertness contract: `t = 0` must reproduce the stage's
#  pre-wiring output BIT FOR BIT, which is what makes the field safe to ship.
# ===========================================================================

RECIP_STAGE_CPP = r"""
#include "AlgoCharacteristicCurve.hpp"
#include "AlgoReciprocity.hpp"
#include "film_profiles.hpp"
#include <cstdio>

struct TRow { const char* name; int it; double t; };

static const TRow TROWS[] = {
/*ROWS*/
};

int main()
{
    const auto& db = film::GetFilmDatabase();
    const int   n  = (int)(sizeof(TROWS)/sizeof(TROWS[0]));

    // Four exposures spanning eight decades, so a shift shows as a translation
    // of the whole tone scale rather than as one displaced sample.
    const double E[4] = { 1e-4, 1e-2, 1.0, 10.0 };

    for (int i = 0; i < n; ++i)
    {
        const TRow& r = TROWS[i];

        const film::FilmProfile* p = nullptr;
        for (const auto& q : db)
            if (q.name == r.name) { p = &q; break; }
        if (nullptr == p)
            continue;

        const int W = 4, H = 1, P = 4;
        AlgoType eR[4], eG[4], eB[4];
        AlgoType dR[4], dG[4], dB[4];
        AlgoType lR[4], lG[4], lB[4];

        for (int x = 0; x < 4; ++x)
            eR[x] = eG[x] = eB[x] = (AlgoType)E[x];

        HighPrecType shift[3];
        AlgoReciprocityLogShift(*p, (HighPrecType)r.t, shift);

        // A reversal stock consumes the anchor as a log-exposure trim; a
        // negative carries it to the print. Zero here for both, because this
        // family is about the SHIFT and a non-zero trim would only add a
        // constant that both sides share.
        const HighPrecType anchor[3] = { 0, 0, 0 };

        AlgoStage08_CharacteristicCurve(eR, eG, eB, dR, dG, dB,
                                        lR, lG, lB, W, H, P,
                                        *p, anchor, shift);

        for (int x = 0; x < 4; ++x)
            printf("RS\t%s\t%d\t%d\t%.17g\t%.17g\t%.17g\n",
                   r.name, r.it, x,
                   (double)dR[x], (double)dG[x], (double)dB[x]);
    }

    return 0;
}
"""

#: ⚠ 0.0 IS THE LOAD-BEARING ONE. It is the inertness contract: at "no stated
#: time" the stage must return exactly what it returned before the parameter
#: existed. The other three straddle every table's measured range so the
#: held-flat ends and the interpolated middle are all exercised.
RECIP_STAGE_TIMES = (0.0, 1.0, 10.0, 100.0)


def recip_stage_probe_table():
    return [(q.name, i, float(t))
            for q in fp.FILM_PROFILES
            for i, t in enumerate(RECIP_STAGE_TIMES)]


def recip_stage_build_and_run(tmp: Path, root: Path, rows) -> dict:
    lines = ['    { "%s", %d, %.17g },' % r for r in rows]
    src = tmp / "recip_stage_parity.cpp"
    src.write_text(RECIP_STAGE_CPP.replace("/*ROWS*/", "\n".join(lines)))
    exe = tmp / "recip_stage_parity"
    cmd = ["g++", "-std=c++14", "-O1", "-I", str(root), "-I", str(HERE),
           "-o", str(exe), str(src),
           str(root / "Algo_08_Sim.cpp"),
           # Same link set as the Callier stage probe: stage 8 pulls
           # AlgoSoftplus from stage 5 and AlgoCopyImage from the separable
           # blur. Linking the REAL stage is the whole point.
           str(root / "Algo_05_Sim.cpp"),
           str(root / "AlgoSeparableBlur.cpp"),
           str(HERE / "film_profiles.cpp"),
           str(HERE / "LoadFilmDataBase.cpp")]
    cmd += [str(q) for q in sorted(HERE.glob("film_profiles_data_*.cpp"))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] reciprocity stage probe compile failed")
        print(r.stderr[-4000:])
        raise SystemExit(2)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] reciprocity stage probe crashed")
        print(r.stderr[-2000:])
        raise SystemExit(2)
    out = {}
    for line in r.stdout.splitlines():
        fam, nm, it, k, a, b, c = line.split("\t")
        out[(fam, nm, int(it), int(k))] = (float(a), float(b), float(c))
    return out


def recip_stage_python_side(rows) -> dict:
    """The same four exposures through film_sim's own curve and shift."""
    import math as _m
    import film_sim as fs
    E = (1e-4, 1e-2, 1.0, 10.0)
    by = {q.name: q for q in fp.FILM_PROFILES}
    out = {}
    for name, it, t in rows:
        q = by[name]
        sh = fs.reciprocity_log_shift(q, t) if t > 0.0 else (0.0, 0.0, 0.0)
        curves = q.curves.as_tuple()
        rev = q.is_reversal
        for k, e in enumerate(E):
            vals = []
            for c in range(3):
                # ⚠ `density_scalar`, NOT `density`. The array form casts
                # through np.float32 and this probe links the SCALAR stage,
                # where AlgoType is double -- comparing a float32 Python result
                # against a double C++ one reported a 5.7e-06 D "disagreement"
                # that was entirely float32 rounding through the softplus. The
                # two forms are the same expression at different widths, and
                # the one to compare against is the one the linked stage uses.
                # 1e-8 is ALGO_CURVE_EXPOSURE_FLOOR, mirrored deliberately.
                le = _m.log10(max(e, 1e-8)) + sh[c]
                arg = -le if rev else le
                vals.append(fs.density_scalar(arg, curves[c]))
            out[("RS", name, it, k)] = tuple(vals)
    return out



# ===========================================================================
#  PROCESS VARIANT, RESOLVER LEVEL
#
#  The resolver returns a PROFILE, so what is compared is the six ToneCurve
#  parameters it yields on each of the three records, plus the exposure index.
#  Driven over every stock and every variant index in the database, including
#  the out-of-range and OFF sentinels.
#
#  ⚠ THE INERT CASES ARE THE LOAD-BEARING ONES. -1 and an out-of-range index
#  must return the stock exactly as shipped, and so must the nineteen AGFAPAN
#  developer records, which differ only in an exposure index no stage reads.
#  A resolver that copied the profile for those would still render correctly
#  and would still be wrong: the contract is that selecting a label costs
#  nothing and changes nothing.
# ===========================================================================

VARIANT_CPP = r"""
#include "AlgoProcessVariant.hpp"
#include "film_profiles.hpp"
#include <cstdio>

struct VRow { const char* name; int idx; };

static const VRow VROWS[] = {
/*ROWS*/
};

int main()
{
    const auto& db = film::GetFilmDatabase();
    const int   n  = (int)(sizeof(VROWS)/sizeof(VROWS[0]));

    for (int i = 0; i < n; ++i)
    {
        const VRow& r = VROWS[i];

        const film::FilmProfile* p = nullptr;
        for (const auto& q : db)
            if (q.name == r.name) { p = &q; break; }
        if (nullptr == p)
            continue;

        film::FilmProfile store;
        const film::FilmProfile& out =
            AlgoResolveProcessVariant(*p, r.idx, store);

        // Whether the resolver copied at all is part of the contract, not an
        // implementation detail: it is what makes an unselected variant free.
        const int copied = (&out == p) ? 0 : 1;

        const film::ToneCurve* c[3] = { &out.curves.r, &out.curves.g, &out.curves.b };

        for (int k = 0; k < 3; ++k)
            printf("V\t%s\t%d\t%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%d\t%d\n",
                   r.name, r.idx, k,
                   (double)c[k]->dmin, (double)c[k]->gamma,
                   (double)c[k]->toe_x, (double)c[k]->toe_k,
                   (double)c[k]->shoulder_x, (double)c[k]->shoulder_k,
                   (int)out.exposure_index, copied);
    }

    return 0;
}
"""


def variant_probe_table():
    """Every stock, every variant index, plus OFF and one past the end."""
    rows = []
    for q in fp.FILM_PROFILES:
        rows.append((q.name, -1))
        for i in range(len(q.process_variants)):
            rows.append((q.name, i))
        rows.append((q.name, len(q.process_variants)))     # out of range
    return rows


def variant_build_and_run(tmp: Path, root: Path, rows) -> dict:
    lines = ['    { "%s", %d },' % r for r in rows]
    src = tmp / "variant_parity.cpp"
    src.write_text(VARIANT_CPP.replace("/*ROWS*/", "\n".join(lines)))
    exe = tmp / "variant_parity"
    cmd = ["g++", "-std=c++14", "-O1", "-I", str(root), "-I", str(HERE),
           "-o", str(exe), str(src),
           str(HERE / "film_profiles.cpp"), str(HERE / "LoadFilmDataBase.cpp")]
    cmd += [str(q) for q in sorted(HERE.glob("film_profiles_data_*.cpp"))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] process-variant probe compile failed")
        print(r.stderr[-4000:])
        raise SystemExit(2)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] process-variant probe crashed")
        print(r.stderr[-2000:])
        raise SystemExit(2)
    out = {}
    for line in r.stdout.splitlines():
        f = line.split("\t")
        out[(f[0], f[1], int(f[2]), int(f[3]))] = (
            tuple(float(x) for x in f[4:10]), int(f[10]), int(f[11]))
    return out


def variant_python_side(rows) -> dict:
    import film_sim as fs
    by = {q.name: q for q in fp.FILM_PROFILES}
    out = {}
    for name, idx in rows:
        q = by[name]
        r = fs.resolve_process_variant(q, idx)
        cs = r.curves.as_tuple()
        for k in range(3):
            c = cs[k]
            out[("V", name, idx, k)] = (
                (c.dmin, c.gamma, c.toe_x, c.toe_k, c.shoulder_x, c.shoulder_k),
                int(r.exposure_index), 0 if r is q else 1)
    return out


#: Specular settings probed. 0.0 is the load-bearing one: the whole inertness
#: contract is that it changes nothing, so it must be compared, not assumed.
CALLIER_STAGE_SPECULAR = (0.0, 0.35, 1.0)


def callier_stage_probe_table():
    """One row per (stock, specular). The whole database, both kinds."""
    return [(q.name, i, float(s))
            for q in fp.FILM_PROFILES
            for i, s in enumerate(CALLIER_STAGE_SPECULAR)]


def callier_stage_build_and_run(tmp: Path, root: Path, rows) -> dict:
    lines = ['    { "%s", %d, %.17g },' % r for r in rows]
    src = tmp / "callier_stage_parity.cpp"
    src.write_text(CALLIER_STAGE_CPP.replace("/*ROWS*/", "\n".join(lines)))
    exe = tmp / "callier_stage_parity"
    # ⚠ THE REAL DATABASE, not literals. Unlike the law probe above this one
    # cannot use literals: AlgoSolveAnchors reads the whole profile -- curves,
    # dye matrix, couplers, taking matrix, print stock -- so a hand-built stub
    # would be a different film. build.py runs the audits AFTER codegen, so the
    # table it walks is the one the generator just wrote.
    cmd = ["g++", "-std=c++14", "-O1", "-I", str(root), "-I", str(HERE),
           "-o", str(exe), str(src),
           str(root / "Algo_08_Sim.cpp"),
           # ⚠ Algo_08 pulls AlgoSoftplus from stage 5 and AlgoCopyImage from the
           # separable blur, so those translation units have to be linked even
           # though this probe never calls into either. Linking the REAL stage is
           # the entire point -- a reimplementation of the solve here would
           # recreate the divergence the probe exists to catch.
           str(root / "Algo_05_Sim.cpp"),
           str(root / "AlgoSeparableBlur.cpp"),
           str(HERE / "film_profiles.cpp"),
           str(HERE / "LoadFilmDataBase.cpp")]
    cmd += [str(q) for q in sorted(HERE.glob("film_profiles_data_*.cpp"))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] Callier stage probe compile failed")
        print(r.stderr[-4000:])
        raise SystemExit(2)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] Callier stage probe crashed")
        print(r.stderr[-2000:])
        raise SystemExit(2)
    out = {}
    for line in r.stdout.splitlines():
        fam, nm, i_s, k, a, b, c = line.split("\t")
        out[(fam, nm, int(i_s), int(k))] = (float(a), float(b), float(c))
    return out


def callier_stage_python_side(rows) -> dict:
    """The same probes through film_sim's own stage 12b and anchor solve."""
    import numpy as _np
    import film_sim as fs
    out = {}
    net = (0.0, 0.25, 1.0, 2.0)
    for nm, i_s, sp in rows:
        p = fp.get_profile(nm)
        cur = p.curves.as_tuple()
        dens = _np.zeros((1, 4, 3), dtype=_np.float32)
        for x, nt in enumerate(net):
            for c in range(3):
                dens[0, x, c] = float(cur[c].dmin) + nt
        fs.callier_density(dens, cur, p.callier_q, sp, p.is_monochrome)
        for x in range(4):
            out[("CS", nm, i_s, x)] = tuple(float(dens[0, x, c])
                                            for c in range(3))
        ps = None if p.is_reversal else fp.get_print_stock(p.default_print)
        a = fs.solve_anchors(p, ps, 0.18, 1.0, sp)
        out[("CA", nm, i_s, 0)] = (float(a[0]), float(a[1]), float(a[2]))
    return out


# ===========================================================================
#  STAGE-LEVEL GRAIN PROBE -- added 2026-08-25, and it is the whole point.
#
#  ⚠ EVERY OTHER FAMILY IN THIS FILE COMPARES A *LAW* AGAINST A *LAW*. That is
#  how a real divergence survived for weeks: `FilmGrainSigma()` in the generated
#  header is correct, this file evaluated it directly, the two sides agreed on
#  every stock -- and NOTHING IN THE RENDERER CALLED IT. `AlgoAddGrain` inlined
#  its own square root, without the net-1.0 normalisation, and shipped grain
#  4-18 % loud on all 147 stocks that use the legacy branch (measured: the ratio
#  was exactly sqrt(1 + fog_grain), reproduced to 3.0e-08).
#
#  A parity check must exercise the CODE THAT RENDERS. This probe therefore
#  compiles and calls `AlgoAddGrain` itself, on a synthetic plane, and recovers
#  the amplitude the stage actually applied.
#
#  THE EXTRACTION IS EXACT, NOT FITTED. The stage computes
#      out = D + gain * field * amp
#  so with `field` set to exactly 1.0 and `gain` to exactly 1.0, `amp` is
#  `out - D` with no arithmetic in between and nothing to invert.
# ===========================================================================

GRAIN_STAGE_CPP = r"""
#include "AlgoGrain.hpp"
#include "film_profiles.hpp"
#include <cstdio>

// ⚠ THE ROW CARRIES THE WHOLE GrainSpec THE LAW READS, NOT JUST fog.
// Before 2026-08-30 AlgoAddGrain took a loose fog value, which made it
// STRUCTURALLY IMPOSSIBLE for the stage to reach the measured sigma(D) shape --
// half of why queue C30/C33 lasted as long as it did. The signature now takes
// the spec and the per-channel dmax, so this probe has to supply both, and
// supplying them is what makes the measured branch testable at all.
struct SRow {
    const char* name; int ch; int k;
    double D; double dmin; double dmax;
    double fog;
    int    measured;
    double toe; double toe_at; double mid; double top; double top_at;
    double peak; double peak_at;
};

static const SRow SROWS[] = {
/*ROWS*/
};

int main()
{
    const int n = (int)(sizeof(SROWS)/sizeof(SROWS[0]));
    for (int i = 0; i < n; ++i) {
        const SRow& r = SROWS[i];

        // Value-initialised, then only the fields the amplitude law reads are
        // set by name. A positional literal here would rot the first time
        // GrainSpec gains a field.
        film::GrainSpec g{};
        g.fog_grain             = (float)r.fog;
        g.sigma_shape_measured  = (r.measured != 0);
        g.sigma_shape_toe       = (float)r.toe;
        g.sigma_shape_toe_at    = (float)r.toe_at;
        g.sigma_shape_mid       = (float)r.mid;
        g.sigma_shape_dmax      = (float)r.top;
        g.sigma_shape_dmax_at   = (float)r.top_at;
        g.sigma_shape_peak      = (float)r.peak;
        g.sigma_shape_peak_at   = (float)r.peak_at;

        // One pixel is enough; four keeps the row loop honest.
        const int W = 4, H = 1, P = 4;
        AlgoType dR[4], dG[4], dB[4], fR[4], fG[4], fB[4];
        const AlgoType D  = (AlgoType)r.D;
        const AlgoType dm[3] = { (AlgoType)r.dmin, (AlgoType)r.dmin, (AlgoType)r.dmin };
        const AlgoType dx[3] = { (AlgoType)r.dmax, (AlgoType)r.dmax, (AlgoType)r.dmax };
        for (int x = 0; x < 4; ++x) {
            dR[x] = dG[x] = dB[x] = D;
            fR[x] = fG[x] = fB[x] = (AlgoType)1.0;   // unit field: amp = out - D
        }
        AlgoAddGrain(dR, dG, dB, fR, fG, fB, W, H, P,
                     dm, dx, g, (AlgoType)1.0);
        const AlgoType* plane = (r.ch == 0) ? dR : ((r.ch == 1) ? dG : dB);

        // ⚠ THE EQUIVALENCE, ASSERTED IN THE SAME PROGRAM ON THE SAME ROW.
        // The stage reaches FilmGrainSigma through a hoisted evaluator rather
        // than by calling it, which is only admissible while something proves
        // the two are one law. This is that proof: same GrainSpec, same dmin,
        // same dmax, same density, both spellings, printed side by side.
        const double law = (double)film::FilmGrainSigma(
            g, (float)r.dmin, (float)r.dmax, (float)r.D);

        printf("S\t%s\t%d\t%d\t%.9g\t%.9g\n", r.name, r.ch, r.k,
               (double)(plane[0] - D), law);
    }
    return 0;
}
"""

#: Densities probed, as NET density above dmin. 1.0 is the load-bearing one --
#: it is the convention `rms_granularity` is stored at, so the stage MUST return
#: exactly 1.0 there or the stored figure has stopped meaning what the sheet
#: printed. The others check the shape either side of it.
GRAIN_STAGE_NET = (0.2, 0.5, 1.0, 1.5, 2.5)


def grain_stage_probe_table():
    """One row per (stock, channel, net density), carrying everything the
    amplitude law reads -- including the measured sigma(D) fields, which the
    stage could not reach at all before 2026-08-30."""
    rows = []
    for p in fp.FILM_PROFILES:
        g = p.grain
        for c, cur in enumerate(p.curves.as_tuple()):
            dmin = float(cur.dmin)
            dmax = float(cur.dmax)
            for k, net in enumerate(GRAIN_STAGE_NET):
                rows.append((p.name, c, k, dmin + net, dmin, dmax,
                             float(g.fog_grain),
                             1 if g.sigma_shape_measured else 0,
                             float(g.sigma_shape_toe),
                             float(g.sigma_shape_toe_at),
                             float(g.sigma_shape_mid),
                             float(g.sigma_shape_dmax),
                             float(g.sigma_shape_dmax_at),
                             float(g.sigma_shape_peak),
                             float(g.sigma_shape_peak_at)))
    return rows


def grain_stage_build_and_run(tmp: Path, root: Path, rows) -> dict:
    lines = ['    { "%s", %d, %d, %.17g, %.17g, %.17g, %.17g, %d, '
             '%.17g, %.17g, %.17g, %.17g, %.17g, %.17g, %.17g },' % r
             for r in rows]
    src = tmp / "grain_stage_parity.cpp"
    src.write_text(GRAIN_STAGE_CPP.replace("/*ROWS*/", "\n".join(lines)))
    exe = tmp / "grain_stage_parity"
    # ⚠ Algo_11_Sim.cpp pulls the separable blur in through AlgoMakeGrainField,
    # so that translation unit has to be linked even though this probe never
    # builds a field. Linking the real stage is the entire point -- a
    # reimplementation here would recreate the bug this probe exists to catch.
    cmd = ["g++", "-std=c++14", "-O1", "-I", str(root), "-I", str(HERE),
           "-o", str(exe), str(src),
           str(root / "Algo_11_Sim.cpp"), str(root / "AlgoSeparableBlur.cpp")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] grain stage probe compile failed")
        print(r.stderr[-4000:])
        raise SystemExit(2)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] grain stage probe crashed")
        print(r.stderr[-2000:])
        raise SystemExit(2)
    out = {}
    for line in r.stdout.splitlines():
        fam, nm, c, k, v, law = line.split("\t")
        out[(fam, nm, int(c), int(k))] = (float(v), float(law))
    return out


def grain_stage_python_side(rows) -> dict:
    """The same probes through `fp.grain_sigma`, which film_sim.py calls.

    Deliberately the SAME entry point the Python renderer uses at
    film_sim.py:2313 -- not a re-derivation of the law here, for exactly the
    reason this probe family exists.
    """
    out = {}
    for row in rows:
        nm, c, k, D, dmin, dmax = row[0], row[1], row[2], row[3], row[4], row[5]
        p = fp.get_profile(nm)
        out[("S", nm, c, k)] = float(fp.grain_sigma(p.grain, dmin, dmax, D))
    return out


def callier_probe_table():
    """One row per (stock, channel, specular, density offset)."""
    rows = []
    for p in fp.FILM_PROFILES:
        q = float(p.callier_q)
        for c, cur in enumerate(p.curves.as_tuple()):
            dmin = float(cur.dmin)
            for i_s, sp in enumerate(CALLIER_SPECULAR):
                for i_d, dd in enumerate(CALLIER_DELTA):
                    rows.append((p.name, c, i_s, i_d, float(sp),
                                 dmin + float(dd), dmin, q))
    return rows


def callier_build_and_run(tmp: Path, root: Path, rows) -> dict:
    def _f(v):
        t = f"{float(v):.9g}"
        if ("." not in t) and ("e" not in t) and ("E" not in t):
            t += ".0"
        return t + "f"
    lines = ['    { "%s", %d, %d, %d, %.17g, %.17g, %.17g, %s },'
             % (nm, c, i_s, i_d, sp, d, dmin, _f(q))
             for (nm, c, i_s, i_d, sp, d, dmin, q) in rows]
    src = tmp / "callier_parity.cpp"
    src.write_text(CALLIER_CPP.replace("/*ROWS*/", "\n".join(lines)))
    exe = tmp / "callier_parity"
    cmd = ["g++", "-std=c++14", "-O1", "-I", str(root), "-I", str(HERE),
           "-o", str(exe), str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] Callier probe compile failed")
        print(r.stderr[-4000:])
        raise SystemExit(2)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] Callier probe crashed")
        print(r.stderr[-2000:])
        raise SystemExit(2)
    out = {}
    for line in r.stdout.splitlines():
        fam, nm, c, i_s, i_d, f, d = line.split("\t")
        out[(fam, nm, int(c), int(i_s), int(i_d))] = (float(f), float(d))
    return out


def callier_python_side(rows) -> dict:
    """The same rows through film_sim, the reference.

    ⚠ Uses film_sim.callier_net, the single definition of the law, rather than
    re-deriving it: a parity probe that reimplements the reference is testing
    itself. Reports the NET density the law produces and the dmin-referenced
    result, which is exactly what both consumers use.
    """
    import film_sim as fs
    out = {}
    for (nm, c, i_s, i_d, sp, d, dmin, _q) in rows:
        prof = fp.get_profile(nm)
        if fs.callier_is_inert(prof, sp):
            net = d - dmin
        else:
            net = float(fs.callier_net(d - dmin, float(prof.callier_q), sp))
        out[("C", nm, c, i_s, i_d)] = (net, dmin + net)
    return out


def check_recip_field_order() -> None:
    """Positional initialisation again -- verify the header still matches."""
    import re as _re
    hdr = (HERE / "film_profiles.hpp").read_text()
    i = hdr.index("struct ReciprocitySpec {")
    blk = hdr[i:hdr.index("};", i)]
    got = tuple(m[1] for m in _re.findall(r"^\s*(float|bool)\s+(\w+);", blk, _re.M))
    if got != RECIP_FIELDS:
        raise SystemExit("[!] ReciprocitySpec field order changed in the "
                         f"generated header:\n    header: {got}\n"
                         f"    expected: {RECIP_FIELDS}")


#: Laws the generated header publishes AND the value each must be reachable
#: from. A law with no caller in the stage sources is not a law -- it is
#: documentation that happens to compile.
#:
#: ⚠ WHY THIS CHECK EXISTS, AND WHY IT IS NOT PARANOIA. On 2026-08-25 a sweep of
#: this exact surface found the bypass rate was **2 of 2**. `FilmGrainSigma()`
#: had no caller and `AlgoAddGrain` inlined its own square root, shipping grain
#: 4-18 % loud on 147 stocks for weeks. `FilmMtfResponse()` had no caller either
#: and still has none. The Python renderer calls exactly two shared laws from
#: `film_profiles` -- `grain_sigma` and `mtf_response` -- so these two functions
#: ARE the entire shared-law surface between the implementations, and both sides
#: of it were unreachable from the code that renders.
#:
#: The check is deliberately crude: it greps the stage sources for the symbol. A
#: mention in a comment counts, which is a known weakness -- it is a REACHABILITY
#: floor, not a proof of use. Something stronger (a link-time or AST check) would
#: be better; something this cheap running from today is better than that
#: arriving later.
GENERATED_LAWS = {
    "FilmGrainSigma": (
        "grain amplitude vs density, including the net-1.0 normalisation and "
        "the measured sigma(D) anchors"),
    "FilmMtfResponse": (
        "emulsion MTF, including the measured 1/(1+(f/f50)^q) rolloff"),
    "FilmMtfKernel": (
        "the separable two-Gaussian equivalent of that rolloff, which is what a "
        "renderer without an FFT can actually convolve"),
}

#: Laws known to be bypassed, with the reason, so the check reports honestly
#: instead of failing on a state that is already recorded and scoped. ⚠ A law
#: leaving this dict must leave because it gained a caller, never because the
#: failure became inconvenient.
LAW_BYPASS_BASELINE = {
    # ⚠ FilmGrainSigma LEFT THIS DICT ON 2026-08-30 BECAUSE IT GAINED A CALLER,
    # WHICH IS THE ONLY ADMISSIBLE REASON. Queue C30/C33 is closed: AlgoAddGrain
    # now takes the GrainSpec and the per-channel dmax, reaches the measured
    # sigma(D) anchors, and both twins go through the shared AlgoGrainAmpBuild()
    # / AlgoGrainAmpAt(). Measured after the change: worst relative
    # disagreement against the Python reference 2.52e-07 over 2415 probes, and
    # |amp - 1| at NET density 1.0 is EXACTLY zero on all 161 stocks x 3
    # channels. See LAW_EQUIVALENT_IMPL for how the indirection is kept honest.
}


#: A law may be reached through a NAMED equivalent rather than by its own
#: symbol, and this is where that indirection is declared instead of inferred.
#:
#: ⚠ AN ENTRY HERE IS A LIABILITY, NOT A CONVENIENCE. Two spellings of one law
#: is the exact condition this whole file exists to police, so an equivalent is
#: admissible only while something ASSERTS the two agree numerically. For
#: FilmGrainSigma that assertion is in the stage probe below, which evaluates
#: `film::FilmGrainSigma()` and the stage's own hoisted evaluator inside the
#: SAME compiled program, on the same rows, and fails if they differ at all.
#:
#: Why the indirection exists rather than the stage simply calling the law:
#: FilmGrainSigma builds and insertion-sorts up to four anchors and walks them
#: twice, none of which depends on the pixel. Calling it per pixel would be
#: correct and unusable. AlgoGrainAmpBuild does that work once per channel and
#: AlgoGrainAmpAt evaluates the result; the law is the same, the arithmetic is
#: hoisted.
LAW_EQUIVALENT_IMPL = {
    "FilmGrainSigma": ("AlgoGrainAmpBuild", "AlgoGrainAmpAt"),
    # ⚠ THIS ENTRY IS A DIFFERENT KIND FROM THE ONE ABOVE AND MUST NOT BE READ
    # AS THE SAME CLAIM. AlgoGrainAmpBuild computes the SAME law with the
    # loop-invariant half hoisted, and the probe asserts they agree exactly.
    # FilmMtfKernel is an APPROXIMATION: the law is a frequency-domain form and
    # this engine convolves separable spatial Gaussians, so the kernel is the
    # best two-lobe fit to it, not the thing itself. Its agreement with
    # FilmMtfResponse is bounded, not exact -- worst max|error| 0.0384 in
    # modulation over the 22 tabulated exponents.
    #
    # ⚠ WHAT MAKES IT ADMISSIBLE ANYWAY: the alternative was not the exact law,
    # it was the single Gaussian, whose error against the same target is 0.1737.
    # The entry records an approximation that is 4.5x closer than what it
    # replaced, with the bound asserted by verify.py, rather than a bypass that
    # was 4.5x further away and asserted nothing.
    "FilmMtfResponse": ("FilmMtfKernel",),
}


#: Tokens that must appear in BOTH the scalar stage and its AVX2 twin, because
#: they carry a law rather than an execution strategy. The two files are allowed
#: to differ in everything about HOW they compute; they are not allowed to differ
#: in WHAT they compute.
#:
#: ⚠ ADDED 2026-08-25 BECAUSE THE DIVERGENCE HAPPENED IMMEDIATELY. The net-1.0
#: grain normalisation was applied to the scalar stage that day; the AVX2 twin
#: was deliberately left to its owner, and the two paths were instantly 1.039x
#: to 1.183x apart on grain amplitude -- a difference in the MODEL, not in the
#: vectorisation. That is exactly what the project's own AVX2 rules forbid, and
#: nothing would have reported it.
#
#: ⚠ RE-POINTED 2026-08-30 WHEN C30/C33 WAS CLOSED, AND THE REASON MATTERS.
#: This used to require the token `ampScale` in both twins. That was the right
#: test while the normalisation was a loose local duplicated in two files -- and
#: it did its job: it is what reported, on every build, that the law had gone
#: missing from BOTH sides.
#:
#: The law no longer lives in either .cpp. `AlgoGrainAmpBuild()` /
#: `AlgoGrainAmpAt()` in the shared AlgoGrain.hpp are now the single definition,
#: evaluated once per channel in HighPrecType, and each twin only chooses how to
#: run the resulting struct over pixels. That is a STRONGER guarantee than a
#: matching token -- the two paths cannot compute different models because there
#: is only one model -- but it needs this test to check that both twins actually
#: go through it rather than open-coding a square root again, which is exactly
#: how the bypass started. `AlgoGrainAmpRaw` is listed beside it because the
#: UNPINNED print/dupe path is a second law and must also be shared: pinning it
#: would move every print render away from film_sim.simulate().
#: ⚠ Algo_08 ADDED 2026-08-30 (M3), AND THE REASON IS THAT NOTHING ELSE COVERS
#: IT. The AVX2 twin of the anchor solve is not compiled by any audit -- the
#: flattened tree here resolves `#include "AlgoTypes.hpp"` to the scalar copy,
#: so the AVX2 flavour builds only inside the owner's real project layout. That
#: makes a textual twin check the ONLY automatic guard on this file, and the
#: Callier law was just rewritten in both copies by hand. `callierQ` is listed
#: as well as the function, because the failure that matters is one twin being
#: left on the old precomputed-multiplier form.
#: ⚠ `recipShift` ADDED 2026-09-01 WITH THE RECIPROCITY WIRING, and it is the
#: only automatic guard the AVX2 twin has for it. The stage-level family below
#: links the SCALAR Algo_08_Sim.cpp; the AVX2 flavour is not compiled by any
#: audit in this flattened tree, so a twin left without the shift would render
#: every long exposure differently from the scalar path and nothing would say
#: so. Both the parameter and the broadcast are listed, because the failure
#: that matters is a twin that takes the argument and never uses it.
TWIN_LAW_TOKENS = {
    "Algo_11_Sim.cpp": ("AlgoGrainAmpBuild", "AlgoGrainAmpRaw"),
    "Algo_08_Sim.cpp": ("AlgoCallierApplyScalar", "callierQ", "recipShift"),
}

#: Tokens that must appear in the AVX2 twin ALONE -- the vector form of a law
#: the scalar twin writes differently. `vRecip` is the broadcast of the shift:
#: a twin carrying `recipShift` in its signature and no broadcast would compile,
#: pass the token check above, and silently drop the correction.
TWIN_AVX2_ONLY_TOKENS = {
    "Algo_08_Sim.cpp": ("vRecip",),
}


def check_twin_consistency(root: Path) -> int:
    """Scalar and AVX2 twins must share the LAW, differing only in execution."""
    bad = 0
    for name, tokens in sorted(TWIN_AVX2_ONLY_TOKENS.items()):
        b = root / "AVX2" / name
        if not b.is_file():
            print(f"  [SKIP] twin consistency: AVX2/{name} not present")
            continue
        tb = _strip_cpp_comments(b.read_text(errors="ignore"))
        for tok in tokens:
            if re.search(r"\b%s\b" % re.escape(tok), tb) is not None:
                print(f"[i] twin consistency: AVX2/{name} carries its vector "
                      f"form '{tok}'")
            else:
                print(f"[FAIL] AVX2/{name} is missing '{tok}' -- it takes the "
                      f"parameter and never applies it, so every long exposure "
                      f"renders differently from the scalar path")
                bad += 1
    for name, tokens in sorted(TWIN_LAW_TOKENS.items()):
        a, b = root / name, root / "AVX2" / name
        if not (a.is_file() and b.is_file()):
            print(f"  [SKIP] twin consistency: {name} missing on one side")
            continue
        ta = _strip_cpp_comments(a.read_text(errors="ignore"))
        tb = _strip_cpp_comments(b.read_text(errors="ignore"))
        for tok in tokens:
            in_a = re.search(r"\b%s\b" % re.escape(tok), ta) is not None
            in_b = re.search(r"\b%s\b" % re.escape(tok), tb) is not None
            if in_a and in_b:
                print(f"[i] twin consistency: {name} and its AVX2 twin both "
                      f"carry '{tok}'")
            elif in_a and not in_b:
                print(f"[FAIL] {name} carries '{tok}' and AVX2/{name} does NOT. "
                      f"The two implementations are computing different models, "
                      f"not the same model at different speeds. For the grain "
                      f"normalisation this is worth 1.039x-1.183x on amplitude")
                bad += 1
            elif in_b and not in_a:
                print(f"[FAIL] AVX2/{name} carries '{tok}' and the scalar "
                      f"{name} does NOT -- the reference path is the one behind")
                bad += 1
            else:
                print(f"[FAIL] neither {name} nor its AVX2 twin carries '{tok}' "
                      f"-- the law it marks has been removed from both")
                bad += 1
    return bad


def _strip_cpp_comments(text: str) -> str:
    """Remove // and /* */ comments. String literals are not protected -- this
    is a symbol-presence test, and a law name inside a string literal would not
    be a call either."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def check_law_reachability(root: Path) -> int:
    """Every law the generator publishes must be reachable from a stage."""
    srcs = sorted(list(root.glob("Algo*.hpp")) + list(root.glob("Algo*.cpp"))
                  + list((root / "AVX2").glob("*.cpp")))
    if not srcs:
        print(f"  [SKIP] law reachability: no stage sources under {root}")
        return 0
    blob = "\n".join(f.read_text(errors="ignore") for f in srcs)
    hdr = (root / "film_profiles.hpp")
    published = set(re.findall(r"^inline\s+\w+\s+(Film\w+)", hdr.read_text(
        errors="ignore"), re.M)) if hdr.is_file() else set()
    unknown = published - set(GENERATED_LAWS)
    bad = 0
    if unknown:
        print(f"[FAIL] the header publishes {sorted(unknown)}, which this check "
              f"does not know about -- add it to GENERATED_LAWS with the "
              f"quantity it defines, so a new law cannot arrive unwatched")
        bad += 1
    for law, what in sorted(GENERATED_LAWS.items()):
        # ⚠ COMMENTS ARE STRIPPED FIRST, AND THE FIRST RUN PROVED WHY. Without
        # it this check reported FilmGrainSigma as "reached from 1 stage source"
        # -- the source being a COMMENT in Algo_11_Sim.cpp explaining that the
        # law is NOT called. A gate that passes on prose about its own failure is
        # the same class of defect it exists to catch.
        # A law counts as reached through its own symbol OR through a declared
        # equivalent (see LAW_EQUIVALENT_IMPL, and read the warning there before
        # adding one).
        wanted = (law,) + tuple(LAW_EQUIVALENT_IMPL.get(law, ()))
        hits = []
        via = set()
        for f in srcs:
            body = _strip_cpp_comments(f.read_text(errors="ignore"))
            for sym in wanted:
                if re.search(r"\b%s\b" % sym, body):
                    hits.append(f.name)
                    via.add(sym)
                    break
        reachable = bool(hits)
        if reachable and law not in LAW_BYPASS_BASELINE:
            through = "" if via == {law} else \
                " (through %s)" % ", ".join(sorted(via))
            print(f"[i] law reachability: {law} reached from {len(hits)} stage "
                  f"source(s){through} -- {what}")
        elif law in LAW_BYPASS_BASELINE:
            print(f"[i] law reachability: {law} is a RECORDED BYPASS -- "
                  f"{LAW_BYPASS_BASELINE[law]}")
            if reachable:
                print(f"[FAIL] {law} is now reached from {hits} but is still "
                      f"listed in LAW_BYPASS_BASELINE -- if the bypass is "
                      f"closed, remove the baseline entry in the same change")
                bad += 1
        else:
            print(f"[FAIL] {law} is published by the generator and called by NO "
                  f"stage source. It defines {what}. Either a stage must call "
                  f"it, or it must be recorded in LAW_BYPASS_BASELINE with the "
                  f"reason and the measured cost")
            bad += 1
    return bad


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
    ap.add_argument("--root", default=str(HERE.parent.parent),
                    help="project root holding the plugin's AlgoReciprocity.hpp; "
                         "the reciprocity family SKIPS when it is absent, so this "
                         "audit still runs from the generator alone")
    ns = ap.parse_args()

    if not (HERE / "film_profiles.hpp").is_file():
        print("  [SKIP] cpp_parity: film_profiles.hpp not generated yet")
        return 0

    check_field_order()
    check_recip_field_order()
    probes = probe_table()
    with tempfile.TemporaryDirectory() as td:
        cpp = build_and_run(Path(td), probes)
    py = python_side(probes)

    missing = sorted(set(py) - set(cpp))
    extra = sorted(set(cpp) - set(py))
    if missing or extra:
        # ⚠ SAY WHICH FILE IS STALE, because "probe sets differ" reads like a bug
        # in this script when it is almost always stale artefacts.
        # ⚠ HISTORICAL NOTE, KEPT BECAUSE THE MESSAGE BELOW WAS WRITTEN FOR IT:
        # build.py used to run the audit stage BEFORE codegen, so the run that
        # ADDED a stock always reported every probe for it "missing". That was
        # correct but looked like a defect, and it masked a real stale-root bug on
        # 2026-08-23. The stages were reordered on 2026-08-24 -- codegen and sync
        # now precede the audits -- so this should no longer fire for that reason.
        # If it fires now, the artefacts really are stale: check that sync wrote to
        # the root you are compiling.
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

    # ---------------------------------------------------------------- C8 -----
    # The reciprocity law, third family. Compiled against the PLUGIN'S OWN
    # AlgoReciprocity.hpp rather than generated code -- the law lives in the
    # renderer, not in the table, so this is the only place the two definitions
    # meet. SKIPS rather than fails when the plugin tree is not on disk, the same
    # policy interimage_parity.py uses.
    root = Path(ns.root).resolve()
    if not (root / "AlgoReciprocity.hpp").is_file():
        print(f"  [SKIP] reciprocity: AlgoReciprocity.hpp not present under {root}")
    else:
        rrows = recip_probe_table()
        with tempfile.TemporaryDirectory() as td:
            rcpp = recip_build_and_run(Path(td), root, rrows)
        rpy = recip_python_side(rrows)
        if set(rcpp) != set(rpy):
            print(f"[FAIL] reciprocity probe sets differ: "
                  f"{len(set(rpy) - set(rcpp))} missing, "
                  f"{len(set(rcpp) - set(rpy))} extra")
            bad += 1
        else:
            rworst, rat = 0.0, None
            for k, want in rpy.items():
                err = abs(rcpp[k] - want)
                if err > rworst:
                    rworst, rat = err, k
            # ⚠ SAME TRAP AS THE GRAIN PROBE: two implementations that both
            # returned zero everywhere would agree perfectly and model nothing.
            # So count the rows that actually MOVE, and require the three
            # branches that must move to have moved: the Schwarzschild branch,
            # the measured-table branch, and the CHROMATIC part of it (which is
            # the only place the CC-filter interpretation is exercised at all).
            moved = sum(1 for v in rpy.values() if v != 0.0)
            chrom = sum(1 for (fam, nm, c, k), v in rpy.items()
                        if c == 2 and v != rpy[(fam, nm, 1, k)])
            # Time index 0 is exposure_time_s = 0.0 -- "not stated". Every
            # stock, every channel, must be exactly zero there, and exactly is
            # the right word: a 1e-12 shift would still change the last bit of
            # every density in a render that used to have none.
            inert0 = all(v == 0.0 for (fam, nm, c, k), v in rpy.items() if k == 0)
            inert0 = inert0 and all(v == 0.0 for (fam, nm, c, k), v in rcpp.items()
                                    if k == 0)
            print(f"[i] reciprocity: {len(rpy)} probes over "
                  f"{len(fp.FILM_PROFILES)} stocks x {len(RECIP_TIMES)} times, "
                  f"{moved} non-zero, {chrom} chromatic (blue != green)")
            print(f"[i] reciprocity: worst absolute disagreement "
                  f"{rworst:.2e} decades at {rat}")
            if rworst > TOL_RECIP:
                print(f"[FAIL] the Python and C++ reciprocity laws disagree by "
                      f"{rworst:.2e} decades (tolerance {TOL_RECIP:.0e}) at {rat}")
                bad += 1
            if not inert0:
                print("[FAIL] a stock is NOT inert at exposure_time_s = 0 -- the "
                      "default must reproduce every earlier render exactly")
                bad += 1
            if moved < 300 or chrom < 3:
                print(f"[FAIL] the reciprocity probe is not exercising its own "
                      f"branches: {moved} non-zero, {chrom} chromatic")
                bad += 1

    # ------------------------------------------------ LAW REACHABILITY ------
    # Cheapest gate in the file and the one that would have caught C30 first:
    # a law the generator publishes but no stage calls is not in the pipeline.
    bad += check_law_reachability(root)
    bad += check_twin_consistency(root)

    # ------------------------------------------------- STAGE-LEVEL GRAIN -----
    # The family that tests the renderer instead of a law beside it. Skip, not
    # fail, when the stage sources are not present -- same policy as the others.
    if not ((root / "Algo_11_Sim.cpp").is_file()
            and (root / "AlgoSeparableBlur.cpp").is_file()):
        print(f"  [SKIP] grain stage: Algo_11_Sim.cpp not present under {root}")
    else:
        srows = grain_stage_probe_table()
        with tempfile.TemporaryDirectory() as td:
            scpp = grain_stage_build_and_run(Path(td), root, srows)
        spy = grain_stage_python_side(srows)
        if set(scpp) != set(spy):
            print(f"[FAIL] grain stage probe sets differ: "
                  f"{len(set(spy) - set(scpp))} missing, "
                  f"{len(set(scpp) - set(spy))} extra")
            bad += 1
        else:
            # ⚠ ONE POPULATION SINCE 2026-08-30 (queue C30/C33 CLOSED).
            # This used to judge two: the legacy-branch stocks, which were
            # exact, and the 13 carrying `sigma_shape_measured`, which were
            # knowingly wrong because the stage took a loose fog value and could
            # not reach the traced anchors at all. That gap is closed --
            # AlgoAddGrain now takes the GrainSpec and the per-channel dmax, the
            # AVX2 twin moved in the same commit, and both go through the shared
            # AlgoGrainAmpBuild(). So EVERY stock must now be exact and the
            # split is gone.
            #
            # The measured population is still counted, for the opposite reason
            # it used to be: as a COVERAGE assertion. A probe that silently
            # stopped exercising the measured branch would pass this family
            # trivially, which is the shape of defect this file exists to catch.
            _shaped = {q.name for q in fp.FILM_PROFILES
                       if q.grain.sigma_shape_measured}
            sworst, sat = 0.0, None          # every branch: must be exact
            hworst, hat = 0.0, None          # measured branch, reported alone
            eworst, eat = 0.0, None          # stage vs the generated law
            for k, want in spy.items():
                got, law = scpp[k]
                err = abs(got - want) / max(abs(want), 1e-9)
                if err > sworst:
                    sworst, sat = err, k
                if k[1] in _shaped and err > hworst:
                    hworst, hat = err, k
                lerr = abs(got - law) / max(abs(law), 1e-9)
                if lerr > eworst:
                    eworst, eat = lerr, k
            # ⚠ THE NET-1.0 IDENTITY IS THE LOAD-BEARING ASSERTION. Index 2 of
            # GRAIN_STAGE_NET is net density 1.0, the convention the stored
            # rms_granularity is quoted at. The stage must return EXACTLY 1.0
            # there for every stock and channel, or the stored figure no longer
            # means the number the manufacturer printed. This is the check that
            # would have caught the missing normalisation on day one.
            k_net1 = GRAIN_STAGE_NET.index(1.0)
            net1 = [v for (fam, nm, c, k), (v, _l) in scpp.items()
                    if k == k_net1]
            worst_net1 = max(abs(v - 1.0) for v in net1) if net1 else 1.0
            print(f"[i] grain STAGE: {len(spy)} probes over "
                  f"{len(fp.FILM_PROFILES)} stocks x 3 channels x "
                  f"{len(GRAIN_STAGE_NET)} densities, driving AlgoAddGrain itself")
            print(f"[i] grain STAGE: worst relative disagreement {sworst:.2e} "
                  f"at {sat}; worst |amp - 1| at NET density 1.0 = "
                  f"{worst_net1:.2e}")
            print(f"[i] grain STAGE: {len(_shaped)} stocks carry a measured "
                  f"sigma(D) SHAPE and the stage now REACHES it -- worst "
                  f"relative disagreement {hworst:.2e} at {hat}")
            if sworst > TOL:
                print(f"[FAIL] the RENDERED grain amplitude disagrees with the "
                      f"reference by {sworst:.2e} (tolerance {TOL:.0e}) at "
                      f"{sat} -- the stage, not the law")
                bad += 1
            # ⚠ THE GAP GUARD THAT USED TO LIVE HERE IS RETIRED, NOT LOOSENED.
            # It asserted that the measured-shape disagreement stayed under
            # 2.5 -- a fixed defect, pinned so it could not grow. The defect is
            # fixed, so the assertion above (sworst > TOL, applied to every
            # stock including these) now covers them and a separate tolerance
            # would only weaken it. What replaces it is a COVERAGE check: the
            # measured branch must still be exercised by real stocks, because a
            # probe that stopped reaching it would pass everything.
            if len(_shaped) < 13:
                print(f"[FAIL] only {len(_shaped)} stocks carry a measured "
                      f"sigma(D) shape -- the branch this probe must exercise "
                      f"has shrunk below the 13 it was closed against")
                bad += 1
            # ⚠ THE INDIRECTION LICENCE. The stage reaches FilmGrainSigma
            # through AlgoGrainAmpBuild/At rather than by calling it, and
            # LAW_EQUIVALENT_IMPL only permits that while this holds. Both
            # spellings are evaluated inside one compiled program on identical
            # inputs, so the tolerance is float32 rounding on the same
            # expression, not a model tolerance.
            print(f"[i] grain STAGE: the stage's hoisted evaluator vs the "
                  f"generated FilmGrainSigma -- worst {eworst:.2e} at {eat}")
            if eworst > 1e-6:
                print(f"[FAIL] the stage's amplitude evaluator and the "
                      f"generated FilmGrainSigma disagree by {eworst:.2e} at "
                      f"{eat}. LAW_EQUIVALENT_IMPL licences the indirection "
                      f"ONLY while they are one law -- either fix the "
                      f"evaluator or make the stage call the law directly")
                bad += 1
            if worst_net1 > 1e-6:
                print(f"[FAIL] the grain stage does not return exactly 1.0 at "
                      f"NET density 1.0 (worst {worst_net1:.2e}) -- the stored "
                      f"rms_granularity has stopped meaning the printed figure")
                bad += 1

    # --------------------------------------------------------------- C22 -----
    # Callier's coefficient, fourth family. Same skip-not-fail policy.
    if not (root / "AlgoCallier.hpp").is_file():
        print(f"  [SKIP] Callier: AlgoCallier.hpp not present under {root}")
    else:
        crows = callier_probe_table()
        with tempfile.TemporaryDirectory() as td:
            ccpp = callier_build_and_run(Path(td), root, crows)
        cpy = callier_python_side(crows)
        if set(ccpp) != set(cpy):
            print(f"[FAIL] Callier probe sets differ: "
                  f"{len(set(cpy) - set(ccpp))} missing, "
                  f"{len(set(ccpp) - set(cpy))} extra")
            bad += 1
        else:
            cworst, cat = 0.0, None
            for k, (wf, wd) in cpy.items():
                gf, gd = ccpp[k]
                err = max(abs(gf - wf), abs(gd - wd))
                if err > cworst:
                    cworst, cat = err, k
            # ⚠ THE SAME TRAP, THIRD TIME: two implementations that both returned
            # the input unchanged would agree perfectly and model nothing. So
            # count what MOVES, and require the branches that must move to move.
            # ⚠ THE MOVEMENT AND INERTNESS TESTS MOVED WITH THE LAW (M3). They
            # used to compare a MULTIPLIER against 1.0. There is no multiplier
            # any more, so they compare the NET density the law returns against
            # the net density it was given -- which is what "the law did
            # nothing" actually means, and is the form that survives the next
            # change of law as well.
            _net_in = {("C", nm, c, i_s, i_d): (d - dm)
                       for (nm, c, i_s, i_d, _sp, d, dm, _q) in crows}
            moved = sum(1 for k, (f, _d) in cpy.items() if f != _net_in[k])
            # Exactly inert at specular 0, for every stock and channel -- and
            # EXACTLY is the word: a 1e-12 departure would change the last bit
            # of every density in a render that previously had none.
            inert0 = all(f == _net_in[k] for k, (f, _d) in cpy.items()
                         if k[3] == 0)
            inert0 = inert0 and all(f == _net_in[k]
                                    for k, (f, _d) in ccpp.items() if k[3] == 0)
            # ⚠ AND THE COLOUR STOCKS MUST BE INERT AT *EVERY* SETTING. Q = 1.0
            # on all of them because a chromogenic dye image does not scatter;
            # if a future edit gave one of them a Q, this is what catches it.
            colour_moved = [k[1] for k, (f, _d) in cpy.items()
                            if f != _net_in[k]
                            and not fp.get_profile(k[1]).is_monochrome]
            # The dmin REFERENCE, which is the whole point of this family: at the
            # probe whose density IS dmin (delta index 1) the law must be the
            # IDENTITY even at full specular, because clear base carries no
            # silver. A law referenced to zero instead would return dmin * factor
            # here, and that is exactly the mistake this catches.
            _dmin_of = {(  "C", nm, c, i_s, i_d): dm
                        for (nm, c, i_s, i_d, _sp, _d, dm, _q) in crows}
            at_dmin = max((abs(d - _dmin_of[k]) for k, (_f, d) in ccpp.items()
                           if k[4] == 1), default=0.0)
            print(f"[i] Callier: {len(cpy)} probes over "
                  f"{len(fp.FILM_PROFILES)} stocks x 3 channels x "
                  f"{len(CALLIER_SPECULAR)} specular x {len(CALLIER_DELTA)} "
                  f"densities, {moved} moved by the law")
            print(f"[i] Callier: worst absolute disagreement {cworst:.2e} at {cat}")
            if cworst > TOL_CALLIER:
                print(f"[FAIL] the Python and C++ Callier laws disagree by "
                      f"{cworst:.2e} (tolerance {TOL_CALLIER:.0e}) at {cat}")
                bad += 1
            if not inert0:
                print("[FAIL] a stock is NOT inert at scanner_specular = 0 -- the "
                      "default must reproduce every earlier render exactly")
                bad += 1
            if colour_moved:
                print(f"[FAIL] {len(colour_moved)} colour stock(s) are moved by "
                      f"Callier; a dye image does not scatter: "
                      f"{', '.join(sorted(set(colour_moved))[:4])}")
                bad += 1
            if at_dmin > TOL_CALLIER:
                print(f"[FAIL] the law is not the identity AT dmin ({at_dmin:.2e}) "
                      f"-- it is referenced to zero, not to dmin, and a condenser "
                      f"would darken clear film base")
                bad += 1
            if moved < 300:
                print(f"[FAIL] the Callier probe is not exercising its own "
                      f"branch: only {moved} rows are moved by the law")
                bad += 1

        # ------------------------------------------------ CALLIER, THE STAGE --
        # ⚠ EVERYTHING ABOVE COMPARES A LAW AGAINST A LAW, AND THAT PASSED FOR A
        # WEEK WHILE NOTHING IN THE PIPELINE CALLED EITHER FUNCTION. This drives
        # the two places C41 wired: the in-place stage 12b, and AlgoSolveAnchors
        # at a non-zero specular. The SOLVE half is the load-bearing one -- a
        # pixel pass without it moves mid grey by more than it changes contrast,
        # which is precisely the configuration this task existed to prevent, and
        # a stage-only probe would pass on it.
        if not (root / "Algo_08_Sim.cpp").is_file():
            print(f"  [SKIP] Callier stage: Algo_08_Sim.cpp not present "
                  f"under {root}")
        else:
            srows = callier_stage_probe_table()
            with tempfile.TemporaryDirectory() as td:
                scpp = callier_stage_build_and_run(Path(td), root, srows)
            spy = callier_stage_python_side(srows)
            if set(scpp) != set(spy):
                print(f"[FAIL] Callier stage probe sets differ: "
                      f"{len(set(spy) - set(scpp))} missing, "
                      f"{len(set(scpp) - set(spy))} extra")
                bad += 1
            else:
                sw_stage, sat_stage = 0.0, None
                sw_solve, sat_solve = 0.0, None
                for k, want in spy.items():
                    got = scpp[k]
                    err = max(abs(a - b) for a, b in zip(got, want))
                    if k[0] == "CS":
                        if err > sw_stage:
                            sw_stage, sat_stage = err, k
                    elif err > sw_solve:
                        sw_solve, sat_solve = err, k
                # ⚠ INERTNESS FIRST, AND AS AN IDENTITY RATHER THAN A TOLERANCE.
                # At specular 0 the stage must leave the plane untouched and the
                # solve must return what it returned before the parameter
                # existed. "Close enough" is not the contract: a last-bit change
                # on every density is still a changed render.
                base = {}
                for k, v in scpp.items():
                    if k[2] == 0:
                        base[(k[0], k[1], k[3])] = v
                stage_inert = all(
                    scpp[k] == base[(k[0], k[1], k[3])]
                    for k in scpp if k[2] == 0)
                # The colour half must be inert at EVERY setting: Q = 1.0 on all
                # of them, so a moved colour stock means a Q crept in.
                col_moved = [k[1] for k in scpp
                             if k[2] != 0
                             and not fp.get_profile(k[1]).is_monochrome
                             and scpp[k] != base[(k[0], k[1], k[3])]]
                # And the monochrome half must MOVE, or the probe proves nothing.
                mono_moved = sum(1 for k in scpp
                                 if k[2] == 2
                                 and fp.get_profile(k[1]).is_monochrome
                                 and scpp[k] != base[(k[0], k[1], k[3])])
                print(f"[i] Callier STAGE: {len(spy)} probes over "
                      f"{len(fp.FILM_PROFILES)} stocks x "
                      f"{len(CALLIER_STAGE_SPECULAR)} specular, driving "
                      f"AlgoStage12b_Callier and AlgoSolveAnchors themselves")
                print(f"[i] Callier STAGE: worst stage {sw_stage:.2e} at "
                      f"{sat_stage}; worst SOLVE {sw_solve:.2e} at {sat_solve}")
                print(f"[i] Callier STAGE: {mono_moved} monochrome rows move at "
                      f"full specular, {len(set(col_moved))} colour stocks move "
                      f"(must be 0)")
                if sw_stage > TOL_CALLIER:
                    print(f"[FAIL] stage 12b disagrees with "
                          f"film_sim.callier_density by {sw_stage:.2e} at "
                          f"{sat_stage}")
                    bad += 1
                if sw_solve > TOL:
                    print(f"[FAIL] AlgoSolveAnchors disagrees with "
                          f"film_sim.solve_anchors by {sw_solve:.2e} at "
                          f"{sat_solve} -- ⚠ THE SOLVE IS THE HALF THAT SHIFTS "
                          f"MID GREY; a wrong anchor is worse than no Callier")
                    bad += 1
                if not stage_inert:
                    print("[FAIL] stage 12b is NOT inert at scannerSpecular = 0")
                    bad += 1
                if col_moved:
                    print(f"[FAIL] {len(set(col_moved))} colour stock(s) are "
                          f"moved by stage 12b; a dye image does not scatter")
                    bad += 1
                if mono_moved < 100:
                    print(f"[FAIL] only {mono_moved} monochrome rows move at "
                          f"full specular -- the stage probe is not exercising "
                          f"its own branch")
                    bad += 1

        # ------------------------------------------------------------------
        #  PROCESS VARIANT. The resolver, over every stock and every index.
        # ------------------------------------------------------------------
        if not (root / "AlgoProcessVariant.hpp").is_file():
            print(f"  [SKIP] process variant: AlgoProcessVariant.hpp not "
                  f"present under {root}")
        else:
            vrows = variant_probe_table()
            with tempfile.TemporaryDirectory() as td:
                vcpp = variant_build_and_run(Path(td), root, vrows)
            vpy = variant_python_side(vrows)
            if set(vcpp) != set(vpy):
                print(f"[FAIL] process-variant probe sets differ: "
                      f"{len(set(vpy) - set(vcpp))} missing, "
                      f"{len(set(vcpp) - set(vpy))} extra")
                bad += 1
            else:
                vw, vat, ei_bad, cp_bad = 0.0, None, [], []
                for k, (want, wei, wcp) in vpy.items():
                    got, gei, gcp = vcpp[k]
                    err = max(abs(a - b) for a, b in zip(got, want))
                    if err > vw:
                        vw, vat = err, k
                    if gei != wei:
                        ei_bad.append(k)
                    if gcp != wcp:
                        cp_bad.append(k)
                moved = sum(1 for k, (_, _, cp) in vpy.items() if cp)
                stocks_moved = len({k[1] for k, (_, _, cp) in vpy.items() if cp})
                print(f"[i] process variant: {len(vpy)} probes over "
                      f"{len(fp.FILM_PROFILES)} stocks x every variant index "
                      f"plus OFF and out-of-range")
                print(f"[i] process variant: worst curve-parameter "
                      f"disagreement {vw:.2e} at {vat}; {moved} probe rows "
                      f"resolve to a DIFFERENT profile, over {stocks_moved} "
                      f"stock(s)")
                if vw > TOL_CALLIER:
                    print(f"[FAIL] the two resolvers disagree by {vw:.2e} at "
                          f"{vat}")
                    bad += 1
                if ei_bad:
                    print(f"[FAIL] exposure index differs on {len(ei_bad)} "
                          f"probe(s), first {ei_bad[0]}")
                    bad += 1
                # ⚠ THE COPY DECISION IS PART OF THE CONTRACT. If C++ copies
                # where Python does not, the nineteen label-only variants stop
                # being free; if it does not copy where Python does, a selected
                # variant silently renders as the base stock.
                if cp_bad:
                    print(f"[FAIL] the two resolvers disagree about WHETHER a "
                          f"variant changes the profile, on {len(cp_bad)} "
                          f"probe(s), first {cp_bad[0]}")
                    bad += 1
                # ⚠ TWO DIFFERENT COUNTS, AND CONFLATING THEM IS THE MISTAKE
                # THIS ASSERTION WAS WRITTEN WRONG THE FIRST TIME. SEVEN stocks
                # resolve to a different PROFILE, because the nineteen AGFAPAN
                # developer records each state their own exposure index -- Agfa
                # print one per developer -- and the resolver honours it even
                # though no stage reads that field yet. Only FOUR change a
                # CURVE and therefore a pixel: PORTRA 800 and ULTRA COLOR 400UC
                # carry traced push curves, CINESTILL 800T's Cs2 kit carries a
                # gamma scale, and GEVACHROME_605 carries the 320 ASA reversal
                # push traced from Bild 6 (queue G5, 2026-09-03). Pinning both
                # numbers is what makes a future change legible: a variant
                # losing its curves moves the first, a variant losing its EI
                # moves the second.
                curve_moved = set()
                for k, (want, _, _) in vpy.items():
                    base_k = ("V", k[1], -1, k[3])
                    if vpy[base_k][0] != want:
                        curve_moved.add(k[1])
                if stocks_moved != 7:
                    print(f"[FAIL] {stocks_moved} stocks resolve to a different "
                          f"profile; 7 are expected -- 4 that change curves and "
                          f"the 3 AGFAPAN stocks whose developer records state "
                          f"their own exposure index")
                    bad += 1
                if len(curve_moved) != 4:
                    print(f"[FAIL] {len(curve_moved)} stocks change a CURVE "
                          f"({sorted(curve_moved)}); 4 are expected -- "
                          f"KODAK_PORTRA_800, KODAK_ULTRA_COLOR_400UC, "
                          f"CINESTILL_800T and GEVACHROME_605. A change here "
                          f"means a variant gained or lost its measured curve "
                          f"set")
                    bad += 1
                print(f"[i] process variant: {len(curve_moved)} stocks change a "
                      f"CURVE and therefore a pixel: {sorted(curve_moved)}")

        # ------------------------------------------------------------------
        #  RECIPROCITY, STAGE LEVEL. Drives the real stage 8, not the law.
        # ------------------------------------------------------------------
        if not (root / "Algo_08_Sim.cpp").is_file():
            print(f"  [SKIP] reciprocity stage: Algo_08_Sim.cpp not present "
                  f"under {root}")
        else:
            trows = recip_stage_probe_table()
            with tempfile.TemporaryDirectory() as td:
                tcpp = recip_stage_build_and_run(Path(td), root, trows)
            tpy = recip_stage_python_side(trows)
            if set(tcpp) != set(tpy):
                print(f"[FAIL] reciprocity stage probe sets differ: "
                      f"{len(set(tpy) - set(tcpp))} missing, "
                      f"{len(set(tcpp) - set(tpy))} extra")
                bad += 1
            else:
                tw, tat = 0.0, None
                for k, want in tpy.items():
                    err = max(abs(a - b) for a, b in zip(tcpp[k], want))
                    if err > tw:
                        tw, tat = err, k
                # Inertness as an IDENTITY, not a tolerance: at t = 0 the stage
                # must write exactly what it wrote before the parameter existed.
                base = {(k[1], k[3]): v for k, v in tcpp.items() if k[2] == 0}
                inert = all(tcpp[k] == base[(k[1], k[3])]
                            for k in tcpp if k[2] == 0)
                # And it has to MOVE somewhere, or the probe proves nothing --
                # which is exactly the failure mode that let an unincluded
                # header pass its own law family for months.
                moved = sum(1 for k in tcpp
                            if k[2] != 0 and tcpp[k] != base[(k[1], k[3])])
                print(f"[i] reciprocity STAGE: {len(tpy)} probes over "
                      f"{len(fp.FILM_PROFILES)} stocks x "
                      f"{len(RECIP_STAGE_TIMES)} times x 4 exposures, driving "
                      f"AlgoStage08_CharacteristicCurve itself")
                print(f"[i] reciprocity STAGE: worst {tw:.2e} D at {tat}; "
                      f"{moved} rows move once a time is stated")
                if tw > TOL_CALLIER:
                    print(f"[FAIL] stage 8 disagrees with film_sim by "
                          f"{tw:.2e} D at {tat} -- the shift is wired but the "
                          f"two engines do not agree on it")
                    bad += 1
                if not inert:
                    print("[FAIL] stage 8 is NOT inert at exposureTimeS = 0")
                    bad += 1
                if moved < 100:
                    print(f"[FAIL] only {moved} rows move once a time is "
                          f"stated -- stage 8 is not reading the shift, which "
                          f"is the exact defect this family exists to catch")
                    bad += 1

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
    print("[OK] the Python and C++ grain, MTF, reciprocity and Callier laws "
          "agree on the whole database")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
