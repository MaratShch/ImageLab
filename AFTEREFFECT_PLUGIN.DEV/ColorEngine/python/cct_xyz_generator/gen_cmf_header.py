#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_cmf_header.py

Generate a C++14 header containing a `constexpr std::array` of CIE colour-matching
function (CMF) rows from a CSV file of the form:

    wavelength_nm , x_bar , y_bar , z_bar

Blank cells and "NaN" (any case) are emitted as 0.0 per CIE convention
(CIE datasets list out-of-band values as NaN; they are set to zero for
computational purposes).

Usage:
    python gen_cmf_header.py <path-to-csv> [float|double]

    - <path-to-csv> : full path to the CMF CSV file (required)
    - [float|double]: emitted scalar type. Default: double

Output:
    Writes the generated header to stdout. Redirect to a file, e.g.:
        python gen_cmf_header.py CIE_xyz_1964_10deg.csv double > CmfTable1964.hpp

Target: Python 3.12 (works on 3.8+). Emitted C++ is C++14-compatible.
"""

import sys
import os
import csv
import math
import datetime


def die(msg: str) -> None:
    sys.stderr.write("error: " + msg + "\n")
    sys.exit(1)


def parse_cell(cell: str) -> float:
    """Parse a numeric cell. Empty / NaN / non-finite -> 0.0 (CIE convention)."""
    s = cell.strip()
    if s == "":
        return 0.0
    if s.lower() == "nan":
        return 0.0
    try:
        v = float(s)
    except ValueError:
        die("could not parse numeric value: {!r}".format(cell))
    if not math.isfinite(v):
        return 0.0
    return v


def sanitize_identifier(name: str) -> str:
    """Turn a file stem into a valid C++ identifier fragment (upper snake)."""
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    ident = "".join(out).upper()
    # C++ identifiers may not start with a digit
    if ident and ident[0].isdigit():
        ident = "_" + ident
    return ident or "CMF"


def fmt_value(v: float, cpp_type: str) -> str:
    """Format a float literal for the chosen C++ type.

    double : full round-trip repr, no suffix
    float  : round-trip repr with 'f' suffix
    """
    if cpp_type == "float":
        # repr() gives shortest round-trip decimal in Python 3
        return repr(v) + "f"
    else:
        return repr(v)


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 1 or len(argv) > 2:
        die("usage: python gen_cmf_header.py <path-to-csv> [float|double]")

    csv_path = argv[0]
    cpp_type = "double"
    if len(argv) == 2:
        cpp_type = argv[1].strip().lower()
        if cpp_type not in ("float", "double"):
            die("second argument must be 'float' or 'double' (got {!r})".format(argv[1]))

    if not os.path.isfile(csv_path):
        die("file not found: {}".format(csv_path))

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for lineno, raw in enumerate(reader, start=1):
            # skip fully blank lines
            if not raw or all(c.strip() == "" for c in raw):
                continue
            if len(raw) < 4:
                die("line {}: expected 4 columns, got {}".format(lineno, len(raw)))
            lam = parse_cell(raw[0])
            xb = parse_cell(raw[1])
            yb = parse_cell(raw[2])
            zb = parse_cell(raw[3])
            rows.append((lam, xb, yb, zb))

    if not rows:
        die("no data rows parsed from {}".format(csv_path))

    n = len(rows)
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    ident = sanitize_identifier(stem)
    guard = "__GENERATED_CMF_{}_{}__".format(ident, cpp_type.upper())
    array_name = "CMF_{}".format(ident)
    lambda_min = rows[0][0]
    lambda_max = rows[-1][0]

    # Derive an integer step if the grid is uniform (informational only).
    step_str = "non-uniform"
    if n >= 2:
        step0 = rows[1][0] - rows[0][0]
        uniform = all(abs((rows[i + 1][0] - rows[i][0]) - step0) < 1e-6
                      for i in range(n - 1))
        if uniform:
            # present as int if it is effectively integral
            if abs(step0 - round(step0)) < 1e-6:
                step_str = "{} nm".format(int(round(step0)))
            else:
                step_str = "{} nm".format(step0)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out = sys.stdout.write

    # ---- header banner -----------------------------------------------------
    out("/*\n")
    out(" * {}\n".format(array_name + ".hpp"))
    out(" *\n")
    out(" * !!! GENERATED FILE - DO NOT EDIT MANUALLY !!!\n")
    out(" *\n")
    out(" * This header was produced automatically by gen_cmf_header.py from the\n")
    out(" * CIE colour-matching-function data file:\n")
    out(" *     {}\n".format(os.path.basename(csv_path)))
    out(" *\n")
    out(" * Any manual changes will be lost the next time the generator is run.\n")
    out(" * To change the contents, edit the source CSV and re-run the generator:\n")
    out(" *     python gen_cmf_header.py <path-to-csv> [float|double]\n")
    out(" *\n")
    out(" * Scalar type    : {}\n".format(cpp_type))
    out(" * Rows           : {}\n".format(n))
    out(" * Wavelength grid: {} nm .. {} nm  (step: {})\n".format(
        _fmt_lambda(lambda_min), _fmt_lambda(lambda_max), step_str))
    out(" * Blank / NaN cells were set to 0 per CIE convention.\n")
    out(" * Generated on   : {}\n".format(now))
    out(" *\n")
    out(" * Standard       : C++14 (no newer features used)\n")
    out(" */\n\n")

    # ---- includes / guard --------------------------------------------------
    out("#ifndef {}\n".format(guard))
    out("#define {}\n\n".format(guard))
    out("#include <array>\n")
    out("#include <cstddef>\n\n")

    # ---- row struct --------------------------------------------------------
    out("namespace GeneratedCMF\n{\n\n")
    out("    // One tabulated CMF sample: wavelength (nm) and the three\n")
    out("    // colour-matching-function values x_bar, y_bar, z_bar.\n")
    out("    struct CmfRow_{}\n".format(cpp_type))
    out("    {\n")
    out("        {} lambda;\n".format(cpp_type))
    out("        {} x;\n".format(cpp_type))
    out("        {} y;\n".format(cpp_type))
    out("        {} z;\n".format(cpp_type))
    out("    };\n\n")

    out("    constexpr std::size_t {}_SIZE = {}u;\n\n".format(array_name, n))

    # ---- the array ---------------------------------------------------------
    # std::array<T,N> is an aggregate wrapping a C array T[N]; strict C++14
    # (with -pedantic-errors) requires explicit bracing for BOTH the std::array
    # and its internal C array, hence the outer "{ {" ... "} }".
    out("    constexpr std::array<CmfRow_{}, {}_SIZE> {} =\n".format(
        cpp_type, array_name, array_name))
    out("    { {\n")
    for i, (lam, xb, yb, zb) in enumerate(rows):
        comma = "," if i < n - 1 else ""
        out("        {{ {}, {}, {}, {} }}{}\n".format(
            fmt_value(lam, cpp_type),
            fmt_value(xb, cpp_type),
            fmt_value(yb, cpp_type),
            fmt_value(zb, cpp_type),
            comma,
        ))
    out("    } };\n\n")

    out("} // namespace GeneratedCMF\n\n")
    out("#endif // {}\n".format(guard))


def _fmt_lambda(v: float) -> str:
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return str(v)


if __name__ == "__main__":
    main()
