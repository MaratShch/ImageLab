"""ISO status A and status M spectral responses, from ANSI/ISO 5-3-1995.

SOURCE
------
ANSI/ISO 5-3-1995 = ANSI/NAPM IT2.18-1996, *Photography -- Density Measurements
-- Part 3: Spectral Conditions*, Association for Information and Image
Management. **Table 3** (status A, PDF page 16) and **Table 4** (status M, PDF
page 17). In the corpus as `PDF/PROFILES/aimm.it2.18.1996.pdf`; the document's
own metadata records it as public domain and incorporated into U.S. law at
36 CFR 1238.14(d)(2).

⚠ THE VALUES BELOW WERE READ OFF THE PAGE IMAGES, NOT OFF THE TEXT LAYER, AND
THAT WAS NOT PEDANTRY. The scan's OCR is a Xerox paper-capture pass and it
misaligns table columns: in Table 3 the red entries 4,638 and 5,000 float free
of their wavelength rows entirely, so a reader taking the text layer at face
value lands the red peak on 630 nm instead of 620 nm and shifts the entire red
response by 10 nm. Every column here was checked against a 200 dpi render of the
page, and the tails were cross-checked against the printed extrapolation slopes
(status A red runs 2,383 / 1,970 / 1,551 / 1,141 / 0,741 / 0,341, differences of
-0,41 / -0,42 / -0,41 / -0,40 / -0,40 per 10 nm, against a printed
"slope = -0,040/nm" -- they agree).

⚠ AND THE OCR ALSO INVENTS A MINUS SIGN. The text layer renders the caption as
"Status A -loQ1o spectral products". The page reads **"Table 3 -- Status A --
log10 spectral products"**: no minus. It matters, because a negated table would
inverse every response. The self-consistent reading is the printed one -- each
column's PEAK carries the largest value, 5,000, and "(Normalized to 5,000 peak)"
says so.

WHAT THE NUMBERS MEAN
---------------------
Each tabulated value is ``log10(spectral product) + 5.000``, so the relative
response is ``10 ** (value - 5.000)``, unity at the peak and falling into the
tails. Status A peaks at 440 / 530 / 620 nm; status M at 450 / 540 / 640 nm.

Where a column stops, the standard prints an extrapolation SLOPE in log units
per nm rather than more numbers, and those are carried here as `slope_lo` and
`slope_hi`. They are steep -- status A blue falls 0,140 per nm above 500 nm, so
14 log units per 100 nm -- which is the standard's way of saying "this channel
is dead here". `response()` applies them and then floors at zero.

WHICH STATUS APPLIES TO WHICH STOCK
-----------------------------------
The database already records it per profile in `density_metric`: 74 colour
stocks say `status_m` (the condition for colour NEGATIVES) and 23 say `status_a`
(reversal, prints and papers). ⚠ So the choice is not made here and must not be:
this module publishes both responses and the caller reads the profile's own
declaration. Print stocks carry the same field and are read in status A.
"""

from __future__ import annotations

import numpy as np

#: Table 3 -- status A, log10 spectral products, normalised to 5,000 peak.
#: Verified against the page image at 200 dpi. Peaks: blue 440, green 530,
#: red 620 nm.
STATUS_A = {
    "b": dict(
        slope_lo=0.380, slope_hi=-0.140, start_nm=420.0, step_nm=10.0,
        values=(3.602, 4.819, 5.000, 4.912, 4.620, 4.040, 2.989, 1.566,
                0.165),
    ),                                      # 420 .. 500
    "g": dict(
        slope_lo=0.220, slope_hi=-0.170, start_nm=500.0, step_nm=10.0,
        values=(1.650, 3.822, 4.782, 5.000, 4.906, 4.644, 4.221, 3.609,
                2.766, 1.579),
    ),                                      # 500 .. 590
    "r": dict(
        slope_lo=0.270, slope_hi=-0.040, start_nm=600.0, step_nm=10.0,
        values=(2.568, 4.638, 5.000, 4.871, 4.604, 4.286, 3.900, 3.551,
                3.165, 2.776, 2.383, 1.970, 1.551, 1.141, 0.741, 0.341),
    ),                                      # 600 .. 750
}

#: Table 4 -- status M, log10 spectral products, normalised to 5,000 peak.
#: Peaks: blue 450, green 540, red 640 nm.
STATUS_M = {
    "b": dict(
        slope_lo=0.250, slope_hi=-0.220, start_nm=410.0, step_nm=10.0,
        values=(2.103, 4.111, 4.632, 4.871, 5.000, 4.955, 4.743, 4.343,
                3.743, 2.990, 1.852),
    ),                                      # 410 .. 510
    "g": dict(
        slope_lo=0.106, slope_hi=-0.120, start_nm=470.0, step_nm=10.0,
        values=(1.152, 2.207, 3.156, 3.804, 4.272, 4.626, 4.872, 5.000,
                4.995, 4.818, 4.458, 3.915, 3.172, 2.239, 1.070),
    ),                                      # 470 .. 610
    "r": dict(
        slope_lo=0.260, slope_hi=-0.040, start_nm=620.0, step_nm=10.0,
        values=(2.109, 4.479, 5.000, 4.899, 4.578, 4.252, 3.875, 3.491,
                3.099, 2.687, 2.269, 1.859, 1.449, 1.054, 0.654, 0.254),
    ),                                      # 620 .. 770
}

STATUS = {"status_a": STATUS_A, "status_m": STATUS_M}

#: Peak wavelengths the tables must reproduce. Asserted rather than trusted --
#: a 10 nm column shift is the exact failure the OCR produces, and it would
#: leave every response still looking like a plausible densitometer.
PEAK_NM = {
    "status_a": {"b": 440.0, "g": 530.0, "r": 620.0},
    "status_m": {"b": 450.0, "g": 540.0, "r": 640.0},
}

#: Below this the response is treated as absent. The printed slopes fall many
#: log units per 100 nm, so this is reached almost immediately outside the
#: tabulated span and the exact value does not matter.
FLOOR = 1e-9


def log_product(status: str, band: str, lam):
    """log10 spectral product + 5.000 at wavelength(s) `lam`, extrapolated.

    Linear inside the tabulated span; outside it the standard's own printed
    slope, which is why this is not simply `np.interp` with edge clamping --
    clamping would leave a dead channel reading its edge value forever, and
    status A blue would then contribute at 700 nm.
    """
    t = STATUS[status][band]
    v = np.asarray(t["values"], dtype=np.float64)
    n = len(v)
    x = t["start_nm"] + t["step_nm"] * np.arange(n)
    lam = np.asarray(lam, dtype=np.float64)
    out = np.interp(lam, x, v)
    lo = lam < x[0]
    hi = lam > x[-1]
    if np.any(lo):
        out = np.where(lo, v[0] - t["slope_lo"] * (x[0] - lam), out)
    if np.any(hi):
        out = np.where(hi, v[-1] + t["slope_hi"] * (lam - x[-1]), out)
    return out


def response(status: str, band: str, lam):
    """Relative spectral response, unity at the peak, floored at zero."""
    r = np.power(10.0, log_product(status, band, lam) - 5.0)
    return np.where(r < FLOOR, 0.0, r)


def responses(status: str, lam):
    """(red, green, blue) responses on `lam`, in the renderer's channel order.

    ⚠ RED FIRST. The standard tabulates blue, green, red; this file's matrices,
    curves and channel loops are all red, green, blue. Returning the standard's
    order here would put a blue response in the red row of a dye matrix, which
    renders as a plausible and completely wrong colour cast.
    """
    return tuple(response(status, b, lam) for b in ("r", "g", "b"))


def self_check() -> list[str]:
    """Everything that would catch a mistranscribed or shifted column."""
    bad = []
    for status, tables in STATUS.items():
        for band, t in tables.items():
            v = np.asarray(t["values"])
            n = len(v)
            x = t["start_nm"] + t["step_nm"] * np.arange(n)
            if abs(v.max() - 5.000) > 1e-9:
                bad.append("%s %s: peak is %.3f, not 5.000"
                           % (status, band, v.max()))
            pk = float(x[int(np.argmax(v))])
            want = PEAK_NM[status][band]
            if pk != want:
                bad.append("%s %s: peak at %.0f nm, the standard prints %.0f"
                           % (status, band, pk, want))
            if v.min() < 0.0:
                bad.append("%s %s: negative log product %.3f -- the table is "
                           "log10 products, not MINUS log10 products, and the "
                           "OCR's minus sign is not in the printed caption"
                           % (status, band, v.min()))
            # A response must rise to its peak and fall away from it: any
            # interior reversal is a transposed pair of digits or a shifted row.
            k = int(np.argmax(v))
            if any(v[i] >= v[i + 1] for i in range(k)):
                bad.append("%s %s: not monotone rising up to the peak"
                           % (status, band))
            if any(v[i] <= v[i + 1] for i in range(k, n - 1)):
                bad.append("%s %s: not monotone falling after the peak"
                           % (status, band))
        # ⚠ The three bands must peak in the right ORDER and be separated.
        pks = [PEAK_NM[status][b] for b in ("b", "g", "r")]
        if not (pks[0] < pks[1] < pks[2]):
            bad.append("%s: peaks out of order %s" % (status, pks))
    # Status M sits at longer wavelengths than status A in every band -- that is
    # the whole difference between the two conditions.
    for b in ("b", "g", "r"):
        if not PEAK_NM["status_m"][b] > PEAK_NM["status_a"][b]:
            bad.append("status M %s does not peak longer than status A" % b)
    return bad


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="accepted and unused")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    args = ap.parse_args(argv)

    bad = self_check()
    lam = np.arange(380.0, 780.0, 5.0)
    print("ISO 5-3 status responses, ANSI/NAPM IT2.18-1996 tables 3 and 4")
    for status in ("status_a", "status_m"):
        r, g, b = responses(status, lam)
        print("  %s  peaks %s  half-power widths %s"
              % (status,
                 "/".join("%.0f" % PEAK_NM[status][k] for k in ("b", "g", "r")),
                 "/".join("%.0f nm" % (
                     (lam[c >= 0.5].max() - lam[c >= 0.5].min())
                     if np.any(c >= 0.5) else 0.0) for c in (b, g, r))))
    for m in bad:
        print("[!] " + m)
    if args.assert_:
        if bad:
            print("[FAIL] the ISO 5-3 status tables do not self-check")
            return 1
        print("[OK] status A and status M tables reproduce: peaks at "
              "440/530/620 and 450/540/640 nm, unit peaks, monotone flanks, "
              "and status M longer than status A in every band")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
