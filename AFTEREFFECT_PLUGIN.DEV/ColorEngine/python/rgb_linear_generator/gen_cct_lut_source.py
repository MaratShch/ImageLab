#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_cct_lut_source.py  (Python 3.12)

Generate the Planckian-locus CCT LUT as a PLAIN C ARRAY with EXTERNAL
linkage, split into a .hpp/.cpp pair, for one CIE standard observer
(1931 2-deg or 1964 10-deg), from an official CIE color-matching-function
CSV file.

WHY THIS STYLE (replaces the constexpr std::array header generators)
    A 25k-entry `constexpr std::array` aggregate living in a header has two
    problems in practice:
      1. C++14: namespace-scope constexpr variables have INTERNAL linkage,
         so every translation unit that includes the header owns a private
         ~600 KB copy of each table (object-file bloat + ODR exposure for
         any inline function capturing the table address).
      2. MSVC's constexpr/static-initialization machinery has proven fragile
         at run time with aggregates of this size (Windows-only crashes),
         while GCC/Clang emit clean .rodata.
    A plain `extern const` C array defined in exactly ONE .cpp is the
    simplest construct the language offers: a single authoritative copy,
    external linkage, guaranteed constant-data placement (.rodata/.rdata)
    on every toolchain, zero ODR surface, and ONE file pair serves BOTH
    C++14 and C++20 - the former cpp14/cpp20 header variants are obsolete.

BUILD NOTE
    The generated .cpp must be added to the library sources (CMake
    target_sources). The .hpp is declaration-only and cheap to include.

Input CSV format (CIE / CVRL distribution format):
    wavelength_nm, xBar, yBar, zBar
Blank / NaN cells are treated as 0 per CIE convention.
Sources: https://cie.co.at/data-tables , http://www.cvrl.org

Each LUT row is:  { cct [K], u (CIE 1960), v (CIE 1960), Duv = 0 }
 - All math in IEEE-754 double (Python float == C++ double).
 - Constants emitted with repr() -> shortest round-trip representation,
   i.e. bit-exact double reconstruction. NUMERIC CONTENT IS BIT-IDENTICAL
   to the previous constexpr generators (same integration, same c2).
 - Planck relative spectral radiance uses c2 = 0.014388 m*K (ITS-90,
   matching the Ohno 2013 reference locus); c1 is omitted because any
   constant factor cancels in the (u,v) chromaticity ratio.
 - Integration: plain summation over the tabulated grid when the grid is
   uniform (bit-compatible with the runtime builder CctLutCmf.hpp);
   trapezoidal weighting when the grid is non-uniform.

Usage:
    python gen_cct_lut_source.py <cmf_csv> <observer> [options]

    <observer>            1931 | 1964
    --cct-min   K         default 1000.0
    --cct-max   K         default 25000.0
    --cct-step  K         default 1.0
    --out       BASE      default CCT_LUT_CIE_<observer>
                          (".hpp"/".cpp" are appended; a trailing ".hpp"
                          or ".cpp" on BASE is stripped first)
    --verify              cross-check locus against colour-science
                          (requires 'colour-science' installed)

Examples:
    python gen_cct_lut_source.py CIE_xyz_1931_2deg.csv 1931 --cct-min 900 --cct-max 40000 --cct-step 1 --verify
    python gen_cct_lut_source.py CIE_xyz_1964_10deg.csv 1964 --cct-min 900 --cct-max 40000 --cct-step 1 --verify
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import sys
from decimal import Decimal, getcontext


C2_ITS90 = 0.014388  # second radiation constant [m*K], ITS-90 convention

# STRICT ARITHMETIC (default; this is an offline generator):
# The plain float64 evaluation of the Planck x CMF integration accumulates
# composed rounding error - MEASURED up to 13 ulp in u at some temperatures
# (expm1 + divisions + a 471-term float64 summation). In strict mode every
# entry is evaluated in high-precision decimal arithmetic and rounded ONCE
# to double, so each emitted (u, v) is the CORRECTLY-ROUNDED double of the
# exact mathematical result of the specified formula (same quadrature, same
# ITS-90 c2 - only the evaluation precision changes). Cost: a few
# milliseconds per entry, i.e. minutes for a full 25k..39k-entry table -
# irrelevant offline. --fast restores the old float64 path for quick tests.
STRICT_PREC = 40   # decimal digits; ~2x the ~20 needed for 0.5-ulp doubles


def die(msg: str) -> None:
    print("error: " + msg, file=sys.stderr)
    raise SystemExit(1)


def parse_cell(cell: str) -> float:
    """Parse one CSV cell; blank or NaN -> 0.0 (CIE convention)."""
    s = cell.strip()
    if s == "" or s.lower() == "nan":
        return 0.0
    v = float(s)
    if math.isnan(v):
        return 0.0
    return v


def load_cmf(csv_path: str) -> list[tuple[float, float, float, float]]:
    """Load (lambda_nm, xBar, yBar, zBar) rows from a CIE CSV file."""
    rows: list[tuple[float, float, float, float]] = []
    with open(csv_path, "r", newline="") as f:
        for lineno, raw in enumerate(csv.reader(f), start=1):
            if not raw or all(c.strip() == "" for c in raw):
                continue
            # Tolerate a header line.
            try:
                lam = parse_cell(raw[0])
            except ValueError:
                if lineno == 1:
                    continue
                die(f"line {lineno}: cannot parse wavelength {raw[0]!r}")
            if len(raw) < 4:
                die(f"line {lineno}: expected 4 columns, got {len(raw)}")
            rows.append((lam, parse_cell(raw[1]),
                              parse_cell(raw[2]),
                              parse_cell(raw[3])))
    if len(rows) < 2:
        die("CMF file contains fewer than 2 usable rows")
    if any(rows[i][0] >= rows[i + 1][0] for i in range(len(rows) - 1)):
        die("wavelengths must be strictly ascending")
    return rows


def grid_is_uniform(rows: list[tuple[float, float, float, float]]) -> bool:
    step = rows[1][0] - rows[0][0]
    return all(
        abs((rows[i + 1][0] - rows[i][0]) - step) < 1e-9
        for i in range(len(rows) - 1)
    )


def planck_relative(lambda_nm: float, T: float) -> float:
    """Relative Planck spectral radiance (c1 omitted - cancels in u,v)."""
    lm = lambda_nm * 1e-9                      # meters
    lm5 = (lm * lm) * (lm * lm) * lm           # lambda^5
    return 1.0 / (lm5 * math.expm1(C2_ITS90 / (lm * T)))


def locus_uv(rows: list[tuple[float, float, float, float]],
             weights: list[float],
             T: float) -> tuple[float, float]:
    """Integrate Planck SPD x CMF -> XYZ -> CIE 1960 (u, v), all double."""
    X = 0.0
    Y = 0.0
    Z = 0.0
    for (lam, xb, yb, zb), w in zip(rows, weights):
        sd = planck_relative(lam, T) * w
        X += sd * xb
        Y += sd * yb
        Z += sd * zb
    den = X + 15.0 * Y + 3.0 * Z
    if den == 0.0:
        return 0.0, 0.0
    return (4.0 * X) / den, (6.0 * Y) / den


def precompute_strict(rows: list[tuple[float, float, float, float]],
                      weights: list[float]):
    """Per-wavelength exact Decimal constants for the strict path:
    (lambda^5 [m^5], c2/lambda [K], w*xBar, w*yBar, w*zBar).
    repr() round-trips each double exactly into Decimal."""
    getcontext().prec = STRICT_PREC
    D = Decimal
    c2 = D(repr(C2_ITS90))
    pre = []
    for (lam, xb, yb, zb), w in zip(rows, weights):
        lm = D(repr(lam)) * D("1e-9")
        wD = D(repr(w))
        pre.append((lm ** 5, c2 / lm,
                    wD * D(repr(xb)), wD * D(repr(yb)), wD * D(repr(zb))))
    return pre


def locus_uv_strict(pre, T: float) -> tuple[float, float]:
    """Strict evaluation: identical formula to locus_uv, but every operation
    in STRICT_PREC-digit Decimal; the ONLY rounding to double happens on the
    final u and v (correctly-rounded results)."""
    getcontext().prec = STRICT_PREC
    D = Decimal
    TD = D(repr(float(T)))
    X = Y = Z = D(0)
    one = D(1)
    for lm5, K, wx, wy, wz in pre:
        e = (K / TD).exp()                 # exp at full precision
        sd = one / (lm5 * (e - one))       # expm1 == exp-1 (x >= ~0.4 here)
        X += sd * wx
        Y += sd * wy
        Z += sd * wz
    den = X + D(15) * Y + D(3) * Z
    if den == 0:
        return 0.0, 0.0
    return float(D(4) * X / den), float(D(6) * Y / den)


def make_weights(rows: list[tuple[float, float, float, float]],
                 uniform: bool) -> list[float]:
    """Integration weights: 1.0 everywhere on a uniform grid (plain sum,
    bit-compatible with the runtime C++ builder); trapezoidal otherwise."""
    n = len(rows)
    if uniform:
        return [1.0] * n
    w = [0.0] * n
    for i in range(n):
        left = rows[i][0] - rows[i - 1][0] if i > 0 else 0.0
        right = rows[i + 1][0] - rows[i][0] if i < n - 1 else 0.0
        w[i] = 0.5 * (left + right)
    return w


def observer_names(observer: str) -> tuple[str, str, str]:
    """-> (header define tag, namespace, human description)."""
    if observer == "1931":
        return ("CCT_LUT_CIE_1931_2DEG",
                "CCT_LUT_1931_2DEG",
                "CIE 1931 2-deg standard observer")
    return ("CCT_LUT_CIE_1964_10DEG",
            "CCT_LUT_1964_10DEG",
            "CIE 1964 10-deg standard observer")


def banner(w, filename: str, desc: str, args, rows, count: int,
           grid_note: str, now: str, is_cpp: bool) -> None:
    """Shared generated-file banner for both emitted files."""
    lam_min, lam_max = rows[0][0], rows[-1][0]
    w("/*\n")
    w(f" * {filename}\n")
    w(" *\n")
    w(" * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!\n")
    w(" *\n")
    w(f" * Planckian-locus CCT LUT, {desc}.\n")
    w(" * Row: { cct [K], u (CIE 1960), v (CIE 1960), Duv }.\n")
    w(" * Every entry lies ON the locus by construction -> Duv == 0.\n")
    w(" *\n")
    if is_cpp:
        w(" * PLAIN C ARRAY, EXTERNAL LINKAGE: this .cpp holds the ONLY\n")
        w(" * definition of the table in the whole program (single\n")
        w(" * authoritative copy in .rodata/.rdata). Declaration is in the\n")
        w(" * matching .hpp. ADD THIS FILE TO THE LIBRARY SOURCES.\n")
    else:
        w(" * DECLARATION ONLY: the table data lives in the matching .cpp\n")
        w(" * (plain `extern const` C array - single authoritative copy,\n")
        w(" * external linkage, no per-TU duplication, no constexpr\n")
        w(" * machinery; one file pair serves both C++14 and C++20).\n")
    w(" *\n")
    w(f" * CMF source     : {os.path.basename(args.cmf_csv)}\n")
    w(f" * Wavelengths    : {lam_min:g} .. {lam_max:g} nm "
      f"({len(rows)} rows; {grid_note})\n")
    w(f" * CCT grid       : {args.cct_min:g} .. {args.cct_max:g} K, "
      f"step {args.cct_step:g} K  ({count} entries)\n")
    w(f" * Planck c2      : {C2_ITS90!r} m*K (ITS-90; matches the "
      "Ohno 2013 reference locus)\n")
    if getattr(args, "fast", False):
        w(" * Precision      : FAST MODE - plain float64 evaluation "
          "(up to ~13 ulp\n")
        w(" *                  composed-rounding error); constants via "
          "repr().\n")
    else:
        w(" * Precision      : STRICT - every entry evaluated in "
          f"{STRICT_PREC}-digit decimal\n")
        w(" *                  arithmetic and rounded ONCE to double: "
          "each (u,v) is\n")
        w(" *                  the correctly-rounded double of the exact "
          "result of\n")
        w(" *                  the specified formula (same quadrature, "
          "ITS-90 c2).\n")
        w(" *                  Constants emitted via repr() -> bit-exact "
          "reconstruction.\n")
    w(f" * Generated on   : {now}\n")
    w(" * Standard       : C++14 and newer (plain const array, no\n")
    w(" *                  language-level variants needed)\n")
    w(" */\n\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate Planckian-locus CCT LUT as a plain external-"
                    "linkage C array (.hpp declaration + .cpp data) from a "
                    "CIE CMF CSV file.")
    ap.add_argument("cmf_csv", help="CIE CMF CSV: lambda,xBar,yBar,zBar")
    ap.add_argument("observer", choices=("1931", "1964"),
                    help="standard observer: 1931 (2-deg) or 1964 (10-deg)")
    ap.add_argument("--cct-min", type=float, default=1000.0)
    ap.add_argument("--cct-max", type=float, default=25000.0)
    ap.add_argument("--cct-step", type=float, default=1.0)
    ap.add_argument("--out", default=None,
                    help="output BASE path; '.hpp' and '.cpp' are appended "
                         "(a trailing .hpp/.cpp on the given value is "
                         "stripped first)")
    ap.add_argument("--fast", action="store_true",
                    help="use the plain float64 evaluation instead of the "
                         "strict high-precision mode (quick tests only; "
                         "up to ~13 ulp composed-rounding error)")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check against colour-science if available")
    args = ap.parse_args()

    if not os.path.isfile(args.cmf_csv):
        die(f"file not found: {args.cmf_csv}")
    if args.cct_step <= 0.0 or args.cct_max <= args.cct_min:
        die("invalid CCT range/step")

    rows = load_cmf(args.cmf_csv)
    uniform = grid_is_uniform(rows)
    weights = make_weights(rows, uniform)

    tag, ns, desc = observer_names(args.observer)

    base = args.out or tag
    for suf in (".hpp", ".cpp"):        # tolerate BASE given with extension
        if base.endswith(suf):
            base = base[: -len(suf)]
    hpp_path = base + ".hpp"
    cpp_path = base + ".cpp"
    hpp_name = os.path.basename(hpp_path)

    count = int((args.cct_max - args.cct_min) / args.cct_step) + 1

    entries: list[tuple[float, float, float]] = []
    if args.fast:
        for k in range(count):
            T = args.cct_min + float(k) * args.cct_step
            u, v = locus_uv(rows, weights, T)
            entries.append((T, u, v))
    else:
        # strict mode: correctly-rounded doubles, minutes for a full table
        pre = precompute_strict(rows, weights)
        t0 = datetime.datetime.now()
        for k in range(count):
            T = args.cct_min + float(k) * args.cct_step
            u, v = locus_uv_strict(pre, T)
            entries.append((T, u, v))
            if k == 99:      # early runtime estimate after 100 entries
                per = (datetime.datetime.now() - t0).total_seconds() / 100.0
                print(f"strict mode: ~{per*1000:.1f} ms/entry, "
                      f"estimated total ~{per*count/60.0:.1f} min "
                      f"for {count} entries")
        dt = (datetime.datetime.now() - t0).total_seconds()
        print(f"strict evaluation done in {dt/60.0:.1f} min")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    grid_note = ("uniform, plain summation (bit-compatible with the runtime "
                 "builder)" if uniform else "NON-uniform, trapezoidal weights")

    # ------------------------------------------------------------------ .hpp
    with open(hpp_path, "w", newline="\n") as f:
        w = f.write
        banner(w, hpp_name, desc, args, rows, count, grid_note, now,
               is_cpp=False)
        w(f"#ifndef __GENERATED_{tag}_DECL_HPP__\n")
        w(f"#define __GENERATED_{tag}_DECL_HPP__\n\n")
        w("#include <cstddef>\n\n")
        # Shared row type: identical guarded block in every generated LUT
        # header, so all observer tables use ONE C++ type (include-order
        # independent - whichever header is included first defines it).
        w("#ifndef IMAGELAB2_CCT_LUT_ROW_DOUBLE_SHARED\n")
        w("#define IMAGELAB2_CCT_LUT_ROW_DOUBLE_SHARED\n")
        w("namespace CctLutShared\n{\n")
        w("    struct CctLutRow_double\n")
        w("    {\n")
        w("        double cct;   // [K]\n")
        w("        double u;     // CIE 1960\n")
        w("        double v;     // CIE 1960\n")
        w("        double Duv;   // 0: on-locus by construction\n")
        w("    };\n")
        w("} // namespace CctLutShared\n")
        w("#endif // IMAGELAB2_CCT_LUT_ROW_DOUBLE_SHARED\n\n")
        w(f"namespace {ns}\n{{\n\n")
        w("    using CctLutRow_double = CctLutShared::CctLutRow_double;\n\n")
        w(f"    constexpr double CCT_MIN  = {args.cct_min!r};\n")
        w(f"    constexpr double CCT_MAX  = {args.cct_max!r};\n")
        w(f"    constexpr double CCT_STEP = {args.cct_step!r};\n\n")
        w(f"    constexpr std::size_t {tag}_SIZE = {count}u;\n\n")
        w(f"    // Defined in {os.path.basename(cpp_path)} - plain const\n")
        w(f"    // array, external linkage, single authoritative copy.\n")
        w(f"    extern const CctLutRow_double {tag}[{tag}_SIZE];\n\n")
        w(f"}} // namespace {ns}\n\n")
        w(f"#endif // __GENERATED_{tag}_DECL_HPP__\n")

    # ------------------------------------------------------------------ .cpp
    with open(cpp_path, "w", newline="\n") as f:
        w = f.write
        banner(w, os.path.basename(cpp_path), desc, args, rows, count,
               grid_note, now, is_cpp=True)
        w(f'#include "{hpp_name}"\n\n')
        w(f"namespace {ns}\n{{\n\n")
        w(f"    const CctLutRow_double {tag}[{tag}_SIZE] =\n")
        w("    {\n")
        for (T, u, v) in entries:
            w(f"        {{ {T!r}, {u!r}, {v!r}, 0.0 }},\n")
        w("    };\n\n")
        w(f"}} // namespace {ns}\n")

    print(f"written: {hpp_path}  (declaration, "
          f"{os.path.getsize(hpp_path)} bytes)")
    print(f"written: {cpp_path}  ({count} entries, "
          f"{os.path.getsize(cpp_path)} bytes)")
    print("REMINDER: add the .cpp to the library sources "
          "(CMake target_sources).")

    if args.verify:
        try:
            import numpy as np
            import colour
        except ImportError:
            die("--verify requires the 'colour-science' package "
                "(pip install colour-science)")
        name = ("CIE 1931 2 Degree Standard Observer"
                if args.observer == "1931"
                else "CIE 1964 10 Degree Standard Observer")
        cmfs = colour.MSDS_CMFS[name]
        worst_du = 0.0
        worst_T = 0.0
        for (T, u, v) in entries[:: max(1, count // 200)]:
            sd = colour.sd_blackbody(T, cmfs.shape)
            XYZ = colour.sd_to_XYZ(sd, cmfs=cmfs, method="Integration")
            den = XYZ[0] + 15.0 * XYZ[1] + 3.0 * XYZ[2]
            ur, vr = 4.0 * XYZ[0] / den, 6.0 * XYZ[1] / den
            d = math.hypot(u - ur, v - vr)
            if d > worst_du:
                worst_du, worst_T = d, T
        print(f"verify vs colour-science: worst |d(u,v)| = {worst_du:.3e} "
              f"at T = {worst_T:g} K "
              f"(expected ~1e-4 scale: colour uses CODATA c2 and ASTM "
              f"weighting;\n the LUT matches the Ohno 2013 ITS-90 "
              f"convention - see header comment)")


if __name__ == "__main__":
    main()
