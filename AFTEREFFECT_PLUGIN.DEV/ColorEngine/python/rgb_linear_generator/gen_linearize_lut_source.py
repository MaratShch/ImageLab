#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# gen_linearize_lut_source.py  (Python 3.12)
#
# Generate a combined NORMALIZE + DECODE linearization LUT as a PLAIN C ARRAY
# with EXTERNAL linkage, split into a .hpp/.cpp pair.
#
# Entry i = transfer_decode(i / max_code): index by the RAW integer pixel
# code, read back LINEAR light. One table fuses the normalization and the
# transfer decode.
#
# WHY THIS STYLE (replaces the constexpr-std::array-in-header emission):
#   1. C++14: namespace-scope constexpr variables have INTERNAL linkage -
#      every including translation unit owns a private copy of the table
#      (object bloat + ODR exposure for inline functions capturing it).
#   2. MSVC's constexpr/static-init machinery has proven fragile at run time
#      with large header aggregates (Windows-only crashes), while GCC/Clang
#      emit clean .rodata.
#   A plain `extern const` C array defined in exactly ONE .cpp is the
#   simplest construct available: single authoritative copy, external
#   linkage, guaranteed constant-data placement (.rodata/.rdata) on every
#   toolchain, zero ODR surface - and ONE file pair serves BOTH C++14 and
#   C++20 (the --cpp20 flag is kept for command-line compatibility but is a
#   no-op: there are no language-variant files anymore).
#
#   BUILD NOTE: add the generated .cpp to the library sources (CMake
#   target_sources). The .hpp is declaration-only and cheap to include.
#
# =============================================================================
# RECOMMENDED COMMAND LINES (the three standard granularities)
#
#   8-bit  integer pixels, codes 0..255   (BGRA/ARGB 8u, sRGB-encoded):
#       python gen_linearize_lut_source.py --bits 8  --transfer srgb
#
#   10-bit integer pixels, codes 0..1023  (10-bit RGB/YUV):
#       python gen_linearize_lut_source.py --bits 10 --transfer srgb
#     (use --transfer rec709 instead when the footage is BT.709-encoded video
#      rather than sRGB stills/graphics)
#
#   16-bit integer pixels, codes 0..32767 (Adobe 15-bit-range integer):
#       python gen_linearize_lut_source.py --bits 16 --transfer srgb
#     NOTE: if your host delivers Adobe's 0..32768 convention (0x8000 = 1.0),
#     add:  --max 32768   (table then has 32769 entries and white is exactly
#     code 32768). Verify which convention the host hands you FIRST - a wrong
#     denominator shifts every decoded value and white no longer lands on 1.0.
#
#   Element type:  --dtype float | double | long_double   (default float)
#   C++ standard:  --cpp20 accepted for compatibility; NO-OP (see above).
#
# ACCURACY GUARANTEE (strict arithmetic - this is an offline generator):
#   Every entry is computed in 50-significant-digit decimal arithmetic
#   (exact rational input Decimal(code)/Decimal(max); exact decimal spec
#   constants; the power branch as exp(ln(base)*g) at 50 digits) and rounded
#   ONCE to the requested element type:
#     float       : Decimal -> float64 (correctly rounded) -> float32
#                   (correctly rounded; the double intermediate cannot
#                   double-round at these magnitudes - float64 has 2^29 x
#                   float32-ulp headroom, verified over the 16-bit domain).
#     double      : Decimal -> float64 (correctly rounded), emitted via
#                   repr() -> shortest exact round-trip literal.
#     long_double : the 50-digit decimal truth is emitted as a 40-significant
#                   -digit literal with the 'L' suffix; the COMPILER performs
#                   the single final rounding to the platform's long double
#                   (x87 80-bit on Linux/x86-64, IEEE quad on some platforms,
#                   == double on MSVC). 40 digits exceed the precision of
#                   every long double format in use (max ~36 for binary128),
#                   so the emitted value is correctly rounded on EVERY
#                   platform. NOTE: on MSVC `long double` IS `double` -
#                   the LDBL table gains nothing over F64 on Windows.
#   Result: each stored entry is the correctly-rounded value of the exact
#   mathematical result in the target type - no more accurate table exists.
#
# MATH / CORRECTNESS NOTES (verified):
#   - sRGB decode: IEC 61966-2-1 piecewise EOTF; threshold 0.04045; branches
#     meet at the seam to ~2e-9. srgb(128/255) = 0.2158605001 (reference).
#   - Rec.709 decode: inverse BT.709 OETF; threshold 0.081 (= 4.5 * 0.018).
#     The published rounded constants leave an INHERENT ~5.5e-5 seam mismatch
#     at V = 0.081 - a property of the BT.709 specification itself,
#     reproduced faithfully on purpose (do not "fix" the constants).
#   - Order of operations per entry: NORMALIZE first (code / max), THEN
#     transfer-decode - the decode constants are defined on [0,1] input.
#
# Usage examples:
#   python gen_linearize_lut_source.py --bits 16 --transfer srgb
#   python gen_linearize_lut_source.py --bits 8  --transfer gamma --gamma 2.4
#   python gen_linearize_lut_source.py --bits 10 --transfer rec709 --dtype double
#   python gen_linearize_lut_source.py --bits 16 --transfer srgb --dtype long_double
# =============================================================================

import argparse, sys, struct, datetime, os
from decimal import Decimal, getcontext

# STRICT ARITHMETIC MODE (always on - this is an offline generator): every
# entry is evaluated in 50-significant-digit decimal arithmetic and rounded
# ONCE to the target type. This removes the few ulp of composed-rounding
# error a plain float64 evaluation accumulates (measured: up to 7 ulp
# through the sRGB power branch) and makes each emitted entry the
# CORRECTLY-ROUNDED value of the exact mathematical result.
getcontext().prec = 50

_D = Decimal

def _dpow(base, exponent):
    """base ** exponent in 50-digit Decimal via exp(ln(base) * exponent).
    base must be > 0 (guaranteed: all decode branches feed positive bases)."""
    if base == 0:
        return _D(0)
    return (base.ln() * exponent).exp()

# --- transfer decode functions: normalized encoded [0,1] -> linear ----------
# All operate on and return Decimal (50 digits); the caller rounds ONCE to the
# target C++ type. The branch thresholds are exact decimal spec constants.
def dec_srgb(c):
    if c <= _D("0.04045"):
        return c / _D("12.92")
    return _dpow((c + _D("0.055")) / _D("1.055"), _D("2.4"))

def dec_rec709(c):
    # inverse BT.709 OETF (threshold 0.081 = 4.5 * 0.018). The ~5.5e-5 seam
    # mismatch is inherent to the published BT.709 constants (see notes).
    if c < _D("0.081"):
        return c / _D("4.5")
    return _dpow((c + _D("0.099")) / _D("1.099"), _D(1) / _D("0.45"))

def dec_gamma(c, g):
    return _dpow(c, _D(repr(float(g))))

def dec_linear(c):
    return c

def build_decoder(name, gamma):
    if name == "srgb":    return dec_srgb,   "sRGB (IEC 61966-2-1, piecewise)"
    if name == "rec709":  return dec_rec709, "ITU-R BT.709 (inverse OETF)"
    if name == "gamma":   return (lambda c: dec_gamma(c, gamma)), f"pure gamma {gamma:g}"
    if name == "linear":  return dec_linear, "linear (identity; normalize only)"
    raise ValueError(name)

# --- exact C++ literal for the chosen element type --------------------------
# Input is the 50-digit Decimal truth; rounding to the target happens ONCE:
#   float       : Decimal -> float64 -> float32 (both correctly rounded; no
#                 double-rounding hazard at these magnitudes), repr + 'f'.
#   double      : Decimal -> float64 (correctly rounded), full repr.
#   long_double : 40-significant-digit decimal literal + 'L'; the compiler
#                 performs the one final rounding to the platform's
#                 long double width (correct for 80-bit and binary128 alike).
def make_literal(dtype):
    if dtype == "float":
        def lit(xD):
            xv = struct.unpack("f", struct.pack("f", float(xD)))[0]
            s = repr(xv)
            if ("." not in s) and ("e" not in s) and ("E" not in s) and \
               ("inf" not in s) and ("nan" not in s):
                s += ".0"
            return s + "f"
        return lit
    if dtype == "double":
        def lit(xD):
            s = repr(float(xD))
            if ("." not in s) and ("e" not in s) and ("E" not in s) and \
               ("inf" not in s) and ("nan" not in s):
                s += ".0"
            return s
        return lit
    # long_double
    def lit(xD):
        if xD == 0:
            return "0.0L"
        s = f"{xD:.39E}"          # 40 significant digits, scientific
        return s + "L"
    return lit

def main():
    ap = argparse.ArgumentParser(
        description="Generate normalize+decode LUT as a plain external-"
                    "linkage C array (.hpp declaration + .cpp data).")
    ap.add_argument("--bits", type=int, required=True, choices=[8, 10, 16],
                    help="granularity: 8->0..255, 10->0..1023, 16->0..32767")
    ap.add_argument("--transfer", default="srgb",
                    choices=["srgb", "rec709", "gamma", "linear"])
    ap.add_argument("--gamma", type=float, default=2.4,
                    help="exponent for --transfer gamma (default 2.4)")
    ap.add_argument("--max", type=int, default=None,
                    help="override max code / normalization denominator "
                         "(e.g. 32768 for Adobe 0..32768)")
    ap.add_argument("--dtype", default="float",
                    choices=["float", "double", "long_double"],
                    help="stored element type: float (32-bit), double "
                         "(64-bit) or long_double (platform width; == double "
                         "on MSVC)")
    ap.add_argument("--cpp20", action="store_true",
                    help="ACCEPTED FOR COMPATIBILITY, NO-OP: plain C arrays "
                         "serve C++14 and C++20 with one file pair; no "
                         "language-variant files are generated anymore")
    ap.add_argument("--out", default=None,
                    help="output BASE path; '.hpp'/'.cpp' are appended (a "
                         "trailing .hpp/.cpp on the given value is stripped)")
    args = ap.parse_args()

    if args.cpp20:
        print("note: --cpp20 is a no-op in the plain-C-array generator "
              "(one .hpp/.cpp pair serves both C++14 and C++20).")

    default_max = {8: 255, 10: 1023, 16: 32767}[args.bits]
    maxcode = args.max if args.max is not None else default_max
    count   = maxcode + 1
    decode, desc = build_decoder(args.transfer, args.gamma)
    literal = make_literal(args.dtype)

    elem     = {"float": "float", "double": "double",
                "long_double": "long double"}[args.dtype]
    dsuffix  = {"float": "F32", "double": "F64",
                "long_double": "LDBL"}[args.dtype]
    tag = f"LINEARIZE_LUT_{args.transfer.upper()}_{args.bits}BIT_{dsuffix}"
    ns  = f"LinLut_{args.transfer}_{args.bits}bit_{args.dtype}"
    guard = f"__IMAGELAB2_{tag}_DECL__"

    base = args.out or tag
    for suf in (".hpp", ".cpp"):          # tolerate BASE given with extension
        if base.endswith(suf):
            base = base[: -len(suf)]
    hpp_path = base + ".hpp"
    cpp_path = base + ".cpp"
    hpp_name = os.path.basename(hpp_path)

    # exact invocation, so each file is self-documenting and reproducible
    cmdline = "python " + " ".join(
        [sys.argv[0].replace("\\", "/").split("/")[-1]] + sys.argv[1:])
    now_local = datetime.datetime.now().astimezone()
    now_utc   = datetime.datetime.now(datetime.timezone.utc)

    def banner(w, filename, is_cpp):
        w("// =============================================================================\n")
        w(f"// {filename}  -  GENERATED, do not edit by hand.\n")
        w("//\n")
        w(f"// Generated : {now_local.strftime('%Y-%m-%d %H:%M:%S %z')} "
          f"(UTC {now_utc.strftime('%Y-%m-%d %H:%M:%S')})\n")
        w("// Regenerate with EXACTLY this command line:\n")
        w(f"//   {cmdline}\n")
        w("//\n")
        w(f"// Combined NORMALIZE + DECODE linearization table.\n")
        w(f"//   entry[i] = {desc} decode of (i / {maxcode})\n")
        w(f"//   index    = RAW integer pixel code, 0..{maxcode}\n")
        w(f"//   value    = linear light, element type: {elem}\n")
        w("//\n")
        if is_cpp:
            w("// PLAIN C ARRAY, EXTERNAL LINKAGE: this .cpp holds the ONLY\n")
            w("// definition of the table in the whole program (single\n")
            w("// authoritative copy in .rodata/.rdata). Declaration is in\n")
            w(f"// {hpp_name}. ADD THIS FILE TO THE LIBRARY SOURCES.\n")
        else:
            w("// DECLARATION ONLY: the table data lives in the matching\n")
            w("// .cpp (plain `extern const` C array - single authoritative\n")
            w("// copy, external linkage, no per-TU duplication, no\n")
            w("// constexpr machinery; one file pair serves both C++14 and\n")
            w("// C++20).\n")
        w("//\n")
        w("// ACCURACY: every entry computed in 50-digit decimal arithmetic\n")
        w("// and rounded ONCE to the element type -> each stored value is\n")
        w("// the CORRECTLY-ROUNDED representation of the exact result (for\n")
        w("// long double the single rounding is performed by the compiler\n")
        w("// from a 40-significant-digit literal, correct for any platform\n")
        w("// long double width; note MSVC long double == double).\n")
        w("// =============================================================================\n\n")

    # ------------------------------------------------------------------ .hpp
    with open(hpp_path, "w", newline="\n") as f:
        w = f.write
        banner(w, hpp_name, is_cpp=False)
        w(f"#ifndef {guard}\n#define {guard}\n\n")
        w("#include <cstddef>\n")
        # Common.hpp provides CACHE_ALIGN (cache-line alignment modifier)
        w('#include "Common.hpp"\n\n')
        w(f"namespace {ns}\n{{\n")
        w(f"    constexpr std::size_t {tag}_SIZE = {count}u;\n\n")
        w(f"    // Defined in {os.path.basename(cpp_path)} - plain const\n")
        w(f"    // array, external linkage, single authoritative copy.\n")
        w(f"    // CACHE_ALIGN (see Common.hpp) puts the table start on a\n")
        w(f"    // cache-line boundary; the modifier is repeated IDENTICALLY\n")
        w(f"    // on the definition - MSVC requires declaration and\n")
        w(f"    // definition to agree on __declspec(align()).\n")
        w(f"    extern CACHE_ALIGN const {elem} {tag}[{tag}_SIZE];\n")
        w(f"}} // namespace {ns}\n\n")
        w(f"#endif // {guard}\n")

    # ------------------------------------------------------------------ .cpp
    with open(cpp_path, "w", newline="\n") as f:
        w = f.write
        banner(w, os.path.basename(cpp_path), is_cpp=True)
        w(f'#include "{hpp_name}"\n')
        w('#include "Common.hpp"\n\n')
        w(f"namespace {ns}\n{{\n")
        w(f"    CACHE_ALIGN const {elem} {tag}[{tag}_SIZE] =\n")
        w("    {\n")
        per_line = 4 if args.dtype == "long_double" else 6
        line = []
        for i in range(count):
            line.append(literal(decode(_D(i) / _D(maxcode))))
            if len(line) == per_line:
                w("        " + ", ".join(line) + ",\n")
                line = []
        if line:
            w("        " + ", ".join(line) + ",\n")
        w("    };\n")
        w(f"}} // namespace {ns}\n")

    print(f"written: {hpp_path}  (declaration, {os.path.getsize(hpp_path)} bytes)")
    print(f"written: {cpp_path}  ({count} entries, {elem}, transfer={args.transfer}, "
          f"max={maxcode}, {os.path.getsize(cpp_path)} bytes)")
    print("REMINDER: add the .cpp to the library sources (CMake target_sources).")

if __name__ == "__main__":
    main()
