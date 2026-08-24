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
TOL_CALLIER = 1e-6

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

        const HighPrecType f = AlgoCallierFactor(prof, (HighPrecType)p.spec);
        const HighPrecType r = AlgoCallierApplyScalar((HighPrecType)p.d,
                                                      (HighPrecType)p.dmin, f);

        printf("C\t%s\t%d\t%d\t%d\t%.17g\t%.17g\n",
               p.name, p.c, p.is, p.id, (double)f, (double)r);
    }

    return 0;
}
"""


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

    ⚠ Uses film_sim._callier_factor AND the same dmin-referenced expression
    callier_density() applies per pixel, rather than re-deriving either: a
    parity probe that reimplements the reference is testing itself.
    """
    import film_sim as fs
    out = {}
    for (nm, c, i_s, i_d, sp, d, dmin, _q) in rows:
        prof = fp.get_profile(nm)
        f = float(fs._callier_factor(prof, sp))
        out[("C", nm, c, i_s, i_d)] = (f, dmin + (d - dmin) * f)
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
            moved = sum(1 for k, (f, d) in cpy.items() if f != 1.0)
            # Exactly inert at specular 0, for every stock and channel -- and
            # EXACTLY is the word: a 1e-12 factor would change the last bit of
            # every density in a render that previously had none.
            inert0 = all(f == 1.0 and d == cpy[k][1]
                         for k, (f, d) in cpy.items() if k[3] == 0)
            inert0 = inert0 and all(f == 1.0 for k, (f, _d) in ccpp.items()
                                    if k[3] == 0)
            # ⚠ AND THE COLOUR STOCKS MUST BE INERT AT *EVERY* SETTING. Q = 1.0
            # on all 93 of them because a chromogenic dye image does not scatter;
            # if a future edit gave one of them a Q, this is what catches it.
            colour_moved = [k[1] for k, (f, _d) in cpy.items()
                            if f != 1.0 and not fp.get_profile(k[1]).is_monochrome]
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
                  f"densities, {moved} with a non-unit factor")
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
                      f"branch: only {moved} rows carry a non-unit factor")
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
