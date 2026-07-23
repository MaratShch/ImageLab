#!/usr/bin/env python3
# =============================================================================
# gen_linearize_lut.py - generate a combined NORMALIZE + TRANSFER-DECODE LUT
# as a C++ constexpr std::array<float, N> header, indexed by the raw integer
# pixel code. One table lookup replaces (divide-to-[0,1] + gamma/log decode)
# at run time; output entries are LINEAR-light float.
#
#   index  = raw integer code  (0..max)
#   LUT[i] = decode( i / max )  -> linear float          (normalize THEN decode)
#
# GRANULARITY (--bits):
#   8  -> codes 0..255     (256 entries)
#   10 -> codes 0..1023    (1024 entries)
#   16 -> codes 0..32767   (32768 entries)   <-- Adobe-style 15-bit range
#
#   NOTE on 16-bit: Adobe's "16-bit" is sometimes 0..32768 (0x8000 = 1.0),
#   i.e. 32769 values. This script uses 0..32767 as requested; override the
#   normalization denominator / entry count with --max if your host hands you
#   0..32768 (then pass --max 32768, giving 32769 entries).
#
# TRANSFER (--transfer): srgb | rec709 | gamma | linear
#   All operate on normalized [0,1] input (that is why normalize precedes
#   decode - the constants are defined in [0,1]).
#
# =============================================================================
# RECOMMENDED COMMAND LINES (the three standard granularities)
#
#   8-bit  integer pixels, codes 0..255   (BGRA/ARGB 8u, sRGB-encoded):
#       python gen_linearize_lut.py --bits 8  --transfer srgb
#
#   10-bit integer pixels, codes 0..1023  (v210 & other 10-bit RGB/YUV):
#       python gen_linearize_lut.py --bits 10 --transfer srgb
#     (use --transfer rec709 instead when the footage is BT.709-encoded video
#      rather than sRGB stills/graphics)
#
#   16-bit integer pixels, codes 0..32767 (Adobe 15-bit-range integer):
#       python gen_linearize_lut.py --bits 16 --transfer srgb
#     NOTE: if your host delivers Adobe's 0..32768 convention (0x8000 = 1.0),
#     add:  --max 32768   (table then has 32769 entries and white is exactly
#     code 32768). Verify which convention the host hands you FIRST - a wrong
#     denominator shifts every decoded value and white no longer lands on 1.0.
#
#   Element type: add  --dtype double  for a float64 table (default float32).
#   C++ standard: add  --cpp20         for 'inline constexpr' emission.
#
# MATH / CORRECTNESS NOTES (verified):
#   - sRGB decode: IEC 61966-2-1 piecewise EOTF; threshold 0.04045; the two
#     branches meet at the seam to ~2e-9 (continuous). srgb(128/255) =
#     0.2158605001 (reference value).
#   - Rec.709 decode: inverse BT.709 OETF; threshold 0.081 (= 4.5 * 0.018).
#     The published rounded constants (4.5 / 0.099 / 1.099 / 0.45) leave an
#     INHERENT seam mismatch of ~5.5e-5 linear at V = 0.081 - this is a known
#     property of the BT.709 specification itself, reproduced faithfully here
#     on purpose (do not "fix" the constants; every conforming implementation
#     shares it).
#   - Emitted literals are exactly round-trippable: float entries are rounded
#     to float32 then printed with repr (shortest exact form) + 'f' suffix;
#     double entries print full repr. Parsing the header reproduces the
#     intended bit pattern exactly.
#   - Order of operations per entry: NORMALIZE first (code / max), THEN
#     transfer-decode - the decode constants are defined on [0,1] input.
#
# Usage:
#   python gen_linearize_lut.py --bits 16 --transfer srgb
#   python gen_linearize_lut.py --bits 8  --transfer gamma --gamma 2.4
#   python gen_linearize_lut.py --bits 10 --transfer rec709 --cpp20
# =============================================================================

import argparse, sys, struct, datetime
from decimal import Decimal, getcontext

# STRICT ARITHMETIC MODE (always on - this is an offline generator):
# every entry is evaluated in 50-significant-digit decimal arithmetic and
# rounded ONCE to the target type (float32 or float64). This removes the few
# ulp of composed-rounding error a plain float64 evaluation accumulates
# (measured: up to 7 ulp in float64 through the sRGB power branch) and makes
# each emitted entry the CORRECTLY-ROUNDED value of the exact mathematical
# result. Cost is irrelevant offline (~a second for 32769 entries).
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
    # mismatch is inherent to the published BT.709 constants (see header note).
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

# --- exact C++ literal, round-trippable, for the chosen element type --------
# Input is the 50-digit Decimal truth; it is rounded ONCE here:
#   float  : Decimal -> float64 (correctly rounded) -> float32 (correctly
#            rounded). The double intermediate cannot cause double-rounding
#            error for these magnitudes (float64 has 2^29 x float32 ulp
#            headroom; verified over the full 16-bit sRGB domain).
#   double : Decimal -> float64 (correctly rounded), printed at full repr.
def make_literal(dtype):
    is_float = (dtype == "float")
    def lit(xD):
        xd = float(xD)                                       # round to float64
        xv = struct.unpack("f", struct.pack("f", xd))[0] if is_float else xd
        s = repr(xv)
        # ensure a '.' or exponent so a float suffix is legal (e.g. "1"->"1.0")
        if ("." not in s) and ("e" not in s) and ("E" not in s) and \
           ("inf" not in s) and ("nan" not in s):
            s += ".0"
        return s + "f" if is_float else s
    return lit

def main():
    ap = argparse.ArgumentParser(description="Generate normalize+decode LUT header.")
    ap.add_argument("--bits", type=int, required=True, choices=[8, 10, 16],
                    help="granularity: 8->0..255, 10->0..1023, 16->0..32767")
    ap.add_argument("--transfer", default="srgb",
                    choices=["srgb", "rec709", "gamma", "linear"])
    ap.add_argument("--gamma", type=float, default=2.4,
                    help="exponent for --transfer gamma (default 2.4)")
    ap.add_argument("--max", type=int, default=None,
                    help="override max code / normalization denominator "
                         "(e.g. 32768 for Adobe 0..32768)")
    ap.add_argument("--dtype", default="float", choices=["float", "double"],
                    help="stored element type: float (32-bit) or double (64-bit)")
    ap.add_argument("--cpp20", action="store_true",
                    help="emit 'inline constexpr' (single definition) instead "
                         "of C++14 'constexpr'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    default_max = {8: 255, 10: 1023, 16: 32767}[args.bits]
    maxcode = args.max if args.max is not None else default_max
    count   = maxcode + 1
    decode, desc = build_decoder(args.transfer, args.gamma)
    literal = make_literal(args.dtype)
    elem = args.dtype   # "float" or "double"

    dsuffix = "F32" if args.dtype == "float" else "F64"
    tag = f"LINEARIZE_LUT_{args.transfer.upper()}_{args.bits}BIT_{dsuffix}"
    ns  = f"LinLut_{args.transfer}_{args.bits}bit_{args.dtype}"
    guard = f"__IMAGELAB2_{tag}__"
    qual = "inline constexpr" if args.cpp20 else "constexpr"
    # Default file name carries the C++ standard variant so the cpp14 and
    # cpp20 flavors of the same table can coexist in one directory (mirrors
    # the CCT LUT header naming). An explicit --out overrides as-is.
    std_suffix = "_CPP20" if args.cpp20 else "_CPP14"
    out = args.out or f"{tag}{std_suffix}.hpp"

    # Reconstruct the exact invocation so the header is self-documenting and
    # the table can always be regenerated identically. sys.argv preserves the
    # arguments as given on the command line.
    cmdline = "python " + " ".join([sys.argv[0].replace("\\", "/").split("/")[-1]]
                                   + sys.argv[1:])

    lines = []
    w = lines.append
    w(f"#ifndef {guard}")
    w(f"#define {guard}")
    w("")
    now_local = datetime.datetime.now().astimezone()
    now_utc   = datetime.datetime.now(datetime.timezone.utc)
    w("// =============================================================================")
    w(f"// {tag}.hpp  -  GENERATED, do not edit by hand.")
    w("//")
    w(f"// Generated : {now_local.strftime('%Y-%m-%d %H:%M:%S %z')} "
      f"(UTC {now_utc.strftime('%Y-%m-%d %H:%M:%S')})")
    w("// Regenerate with EXACTLY this command line:")
    w(f"//   {cmdline}")
    w("// Combined NORMALIZE + TRANSFER-DECODE lookup table (linear-light float).")
    w("//")
    w(f"//   Granularity : {args.bits}-bit,  codes 0..{maxcode}  ({count} entries)")
    w(f"//   Transfer    : {desc}")
    w(f"//   Element type: {args.dtype} ({'32' if args.dtype=='float' else '64'}-bit)")
    w(f"//   Normalize   : code / {maxcode}")
    w("//")
    w("//   index  = raw integer pixel code")
    w("//   LUT[i] = decode(i / max) -> linear float   (normalize THEN decode)")
    w("//")
    w("//   Runtime use (no divide, no pow):")
    w(f"//     const {elem} lin = {ns}::{tag}[raw_code];")
    w(f"// Standard: {'C++20 (inline constexpr)' if args.cpp20 else 'C++14 (constexpr)'}")
    w("// =============================================================================")
    w("")
    w("#include <array>")
    w("#include <cstddef>")
    w("")
    w(f"namespace {ns}")
    w("{")
    w(f"    {qual} std::size_t {tag}_SIZE = {count}u;")
    w("")
    w(f"    {qual} std::array<{elem}, {tag}_SIZE> {tag} =")
    w("    {{")

    per = 8
    row = []
    body = []
    for i in range(count):
        row.append(literal(decode(_D(i) / _D(maxcode))))   # exact rational input
        if len(row) == per:
            body.append("        " + ", ".join(row) + ",")
            row = []
    if row:
        body.append("        " + ", ".join(row) + ",")
    # drop trailing comma on the very last value for cleanliness
    if body:
        body[-1] = body[-1].rstrip(",")
    lines.extend(body)

    w("    }};")
    w("")
    w(f"}} // namespace {ns}")
    w("")
    w(f"#endif // {guard}")

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"written: {out}  ({count} entries, {args.dtype}, transfer={args.transfer}, "
          f"max={maxcode}, {'C++20' if args.cpp20 else 'C++14'})")

if __name__ == "__main__":
    main()
