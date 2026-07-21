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
# Usage:
#   python gen_linearize_lut.py --bits 16 --transfer srgb
#   python gen_linearize_lut.py --bits 8  --transfer gamma --gamma 2.4
#   python gen_linearize_lut.py --bits 10 --transfer rec709 --cpp20
# =============================================================================

import argparse, sys, struct

# --- transfer decode functions: normalized encoded [0,1] -> linear ----------
def dec_srgb(c):
    return c/12.92 if c <= 0.04045 else ((c + 0.055)/1.055) ** 2.4

def dec_rec709(c):
    # inverse BT.709 OETF (threshold 0.081 = 4.5 * 0.018)
    return c/4.5 if c < 0.081 else ((c + 0.099)/1.099) ** (1.0/0.45)

def dec_gamma(c, g):
    return c ** g

def dec_linear(c):
    return c

def build_decoder(name, gamma):
    if name == "srgb":    return dec_srgb,   "sRGB (IEC 61966-2-1, piecewise)"
    if name == "rec709":  return dec_rec709, "ITU-R BT.709 (inverse OETF)"
    if name == "gamma":   return (lambda c: dec_gamma(c, gamma)), f"pure gamma {gamma:g}"
    if name == "linear":  return dec_linear, "linear (identity; normalize only)"
    raise ValueError(name)

# --- exact C++ literal, round-trippable, for the chosen element type --------
#   float  : round to float32 and append the 'f' suffix
#   double : keep full double precision, no suffix
def make_literal(dtype):
    is_float = (dtype == "float")
    def lit(x):
        xv = struct.unpack("f", struct.pack("f", x))[0] if is_float else float(x)
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
    out = args.out or f"{tag}.hpp"

    lines = []
    w = lines.append
    w(f"#ifndef {guard}")
    w(f"#define {guard}")
    w("")
    w("// =============================================================================")
    w(f"// {tag}.hpp  -  GENERATED, do not edit by hand.")
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
        row.append(literal(decode(i / float(maxcode))))
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
