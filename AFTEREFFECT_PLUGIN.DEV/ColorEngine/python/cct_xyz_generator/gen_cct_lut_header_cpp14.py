#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_cct_lut_header.py  (Python 3.12)

Generate a C++14 header containing the Planckian-locus CCT LUT as a
constexpr std::array, for one CIE standard observer (1931 2-deg or
1964 10-deg), from an official CIE color-matching-function CSV file.

Input CSV format (CIE / CVRL distribution format):
    wavelength_nm, xBar, yBar, zBar
Blank / NaN cells are treated as 0 per CIE convention.
Sources: https://cie.co.at/data-tables , http://www.cvrl.org

Each LUT row is:  { cct [K], u (CIE 1960), v (CIE 1960), Duv = 0 }
 - All math in IEEE-754 double (Python float == C++ double).
 - Constants emitted with repr() -> shortest round-trip representation,
   i.e. bit-exact double reconstruction (maximal C++14-representable
   accuracy).
 - Planck relative spectral radiance uses c2 = 0.014388 m*K (ITS-90,
   matching the Ohno 2013 reference locus); c1 is omitted because any
   constant factor cancels in the (u,v) chromaticity ratio.
 - Integration: plain summation over the tabulated grid when the grid is
   uniform (bit-compatible with the runtime builder CctLutCmf.hpp);
   trapezoidal weighting when the grid is non-uniform.

Usage:
    python gen_cct_lut_header.py <cmf_csv> <observer> [options]

    <observer>            1931 | 1964
    --cct-min   K         default 1000.0
    --cct-max   K         default 25000.0
    --cct-step  K         default 1.0
    --out       FILE      default CCT_LUT_CIE_<observer>.hpp
    --verify              cross-check locus against colour-science
                          (requires 'colour-science' installed)

Examples:
    python gen_cct_lut_header.py CIE_xyz_1931_2deg.csv 1931 --verify
    python gen_cct_lut_header.py CIE_xyz_1964_10deg.csv 1964
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import sys


C2_ITS90 = 0.014388  # second radiation constant [m*K], ITS-90 convention


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate constexpr Planckian-locus CCT LUT header "
                    "(C++14) from a CIE CMF CSV file.")
    ap.add_argument("cmf_csv", help="CIE CMF CSV: lambda,xBar,yBar,zBar")
    ap.add_argument("observer", choices=("1931", "1964"),
                    help="standard observer: 1931 (2-deg) or 1964 (10-deg)")
    ap.add_argument("--cct-min", type=float, default=1000.0)
    ap.add_argument("--cct-max", type=float, default=25000.0)
    ap.add_argument("--cct-step", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="output header path")
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
    out_path = args.out or (tag + ".hpp")

    count = int((args.cct_max - args.cct_min) / args.cct_step) + 1

    entries: list[tuple[float, float, float]] = []
    for k in range(count):
        T = args.cct_min + float(k) * args.cct_step
        u, v = locus_uv(rows, weights, T)
        entries.append((T, u, v))

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lam_min, lam_max = rows[0][0], rows[-1][0]
    grid_note = ("uniform, plain summation (bit-compatible with the runtime "
                 "builder)" if uniform else "NON-uniform, trapezoidal weights")

    with open(out_path, "w", newline="\n") as f:
        w = f.write
        w("/*\n")
        w(f" * {os.path.basename(out_path)}\n")
        w(" *\n")
        w(" * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!\n")
        w(" *\n")
        w(f" * Planckian-locus CCT LUT, {desc}.\n")
        w(" * Row: { cct [K], u (CIE 1960), v (CIE 1960), Duv }.\n")
        w(" * Every entry lies ON the locus by construction -> Duv == 0.\n")
        w(" *\n")
        w(f" * CMF source     : {os.path.basename(args.cmf_csv)}\n")
        w(f" * Wavelengths    : {lam_min:g} .. {lam_max:g} nm "
          f"({len(rows)} rows; {grid_note})\n")
        w(f" * CCT grid       : {args.cct_min:g} .. {args.cct_max:g} K, "
          f"step {args.cct_step:g} K  ({count} entries)\n")
        w(f" * Planck c2      : {C2_ITS90!r} m*K (ITS-90; matches the "
          "Ohno 2013 reference locus)\n")
        w(" * Precision      : all math in IEEE-754 double; constants "
          "emitted via\n")
        w(" *                  repr() -> shortest round-trip form, i.e. "
          "bit-exact\n")
        w(" *                  double reconstruction (maximal C++14 "
          "accuracy).\n")
        w(f" * Generated on   : {now}\n")
        w(" * Standard       : C++14 (no newer features used)\n")
        w(" */\n\n")
        w(f"#ifndef __GENERATED_{tag}_HPP__\n")
        w(f"#define __GENERATED_{tag}_HPP__\n\n")
        w("#include <array>\n")
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
        w(f"    constexpr std::array<CctLutRow_double, {tag}_SIZE> "
          f"{tag} =\n")
        w("    { {\n")
        for (T, u, v) in entries:
            w(f"        {{ {T!r}, {u!r}, {v!r}, 0.0 }},\n")
        w("    } };\n\n")
        w(f"}} // namespace {ns}\n\n")
        w(f"#endif // __GENERATED_{tag}_HPP__\n")

    print(f"written: {out_path}  ({count} entries, "
          f"{os.path.getsize(out_path)} bytes)")

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
