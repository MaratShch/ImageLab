"""Development TIME vs GAMMA off a Kodak sensitometric panel (H-1-5222 p2).

WHAT IS ADOPTED HERE AND WHAT THIS SCRIPT ACTUALLY DOES
-------------------------------------------------------
The five (time, gamma) pairs stored in `ProcessingFamily` for
EASTMAN_DOUBLE_X_5222 are **printed as text on the sheet**, inside the plot
frame, one label per curve:

    4 min. y=0.50   5 min. y=0.56   6 1/2 min. y=0.66   9 min. y=0.84
    12 min. y=1.05

so they need no tracing and this script adopts nothing. What it does is the
thing the project asks of every adopted number: **re-derive it from the
document**. Each label belongs to one of five drawn curves; the script traces
all five and measures the straight-line slope of each, then checks that the
measured slopes reproduce the printed ones.

⚠ THAT IS A REAL TEST AND IT IS NOT CIRCULAR. The labels are text and the
curves are paths; nothing in the PDF connects them. The association is made
here by CONTRAST ORDER -- a longer development gives a steeper curve, so the
five curves ranked by slope must match the five labels ranked by gamma -- and
the check is then whether the numbers agree, which they need not have.

WHAT IT FOUND, AND THE ONE DISAGREEMENT THAT IS KEPT
-----------------------------------------------------
Four of the five reproduce to 2 % or better:

    4 min    printed 0.50   measured 0.500
    5 min    printed 0.56   measured 0.558
    6 1/2    printed 0.66   measured 0.652
    12 min   printed 1.05   measured 1.060

⚠ **The 9-minute curve does not: printed 0.84, measured 0.813 (3 % low).** It
is also the most window-sensitive of the five -- sweeping the density interval
the slope is fitted over moves it from 0.813 to 0.744, where every other curve
moves by less than 0.012. That is the signature of a curve that is not straight
over the interval being fitted, and Kodak does not print the interval its own
gammas were measured over. So the disagreement is a difference of METHOD that
cannot be resolved from the document, and method rule 4 applies: it is recorded,
not averaged away, and the PRINTED value is what the database stores, because
the printed value is the manufacturer's own statement about their own film.

⚠ THE FOG PLATEAU IS MEASURED AND IT IS NOT CONSTANT. Base+fog rises with
development exactly as the sheet's own Time-Fog inset shows: 0.231 / 0.233 /
0.233 / 0.275 / 0.296 at the five times. The profile's `dmin` is therefore a
statement about ONE development time, and the one it must match is the
recommended control gamma at 6 1/2 minutes: 0.233.

Run:  python kodak_time_gamma.py --root ../.. [--assert]
Needs numpy + PyMuPDF.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dye_density as dd      # noqa: E402

PDF = "KODAK/EASTMAN DOUBLE-X Negative Film 5222.pdf"
PAGE = 2
PROFILE = "EASTMAN_DOUBLE_X_5222"
SOURCE = ("Eastman Kodak Company, «EASTMAN DOUBLE-X Negative Film 5222/7222 "
          "-- Technical Data», KODAK Publication No. H-1-5222, Revised 7-15 "
          "(header JULY 2015), (c) 2015, p2 plot F010_0029AC")

#: The plot frame, in page points. Detected on every run and compared with this.
FRAME = (350.9, 415.2, 535.4, 599.8)

#: Axis ticks: value -> page coordinate.
#: ⚠ THE EXPOSURE AXIS RUNS NEGATIVE AND ITS MINUS SIGNS ARE OVERBARS that do
#: not reach the text layer -- the same Kodak habit that nearly mirrored the
#: 7239 sensitivity axis (see spectral_vector._sign_y_ticks). Here the values
#: are pinned rather than parsed, so the trap is documented, not re-entered.
X_TICKS = {-3.0: 350.7, -2.0: 396.9, -1.0: 442.2, 0.0: 489.5, 1.0: 534.5}
Y_TICKS = {4.0: 415.8, 3.0: 461.9, 2.0: 507.4, 1.0: 553.7, 0.0: 600.6}

#: What the sheet PRINTS, read from the labels inside the frame. This is the
#: data; everything else in this file is the check on it.
PRINTED = ((4.0, 0.50), (5.0, 0.56), (6.5, 0.66), (9.0, 0.84), (12.0, 1.05))

#: Net density interval the straight-line slope is fitted over. Kodak does not
#: print theirs. 0.3-1.2 above base+fog is the interval `kodak_sensitometry.py`
#: already uses for the same purpose, chosen there because a printed curve is a
#: set of chords and only its straight section is chord-free.
NET_LO, NET_HI = 0.3, 1.2

#: Decades of the left plateau used for base+fog. 0.12 rather than 0.35: at
#: 0.35 the 6 1/2-minute curve has already left the plateau and the median comes
#: out 0.270 instead of 0.233, i.e. 0.037 D of the toe folded into the fog.
PLATEAU_DEC = 0.12

#: Measured 2026-08-26. (gamma, base+fog) per printed time, and the tolerances.
EXPECTED = {
    4.0: (0.500, 0.2307), 5.0: (0.558, 0.2326), 6.5: (0.652, 0.2328),
    9.0: (0.798, 0.2747), 12.0: (1.060, 0.2964),
}
TOL_GAMMA, TOL_FOG = 0.010, 0.010

#: ⚠ THE 9-MINUTE CURVE IS ALLOWED TO DISAGREE WITH THE SHEET, and only it.
#: Recorded rather than hidden: printed 0.84, measured 0.798-0.813 depending on
#: the fitting interval. Listing it here means a NEW disagreement on any other
#: curve fails the audit instead of being absorbed into a loose tolerance.
PRINTED_TOL = 0.02
PRINTED_EXEMPT = {9.0: "printed 0.84, measured 0.798 -- 5 % low, and the most "
                       "window-sensitive of the five; Kodak does not print the "
                       "density interval their gamma is measured over"}


def _fit(ticks):
    px = np.array([ticks[k] for k in sorted(ticks)], float)
    v = np.array(sorted(ticks), float)
    m, c = np.polyfit(px, v, 1)
    return float(m), float(c), float(np.abs(m * px + c - v).max())


def extract(root: Path):
    import pymupdf
    pdf = root / "PDF" / "PROFILES" / PDF
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    pg = pymupdf.open(pdf)[PAGE - 1]
    frs = [f for f in dd.frames(pg)
           if all(abs(a - b) < 2.0 for a, b in
                  zip((f.x0, f.y0, f.x1, f.y1), FRAME))]
    if not frs:
        raise SystemExit(f"[!] the sensitometric frame is no longer at {FRAME}")
    fr = frs[0]
    for name, ticks, coord in (("exposure", X_TICKS, "x"),
                               ("density", Y_TICKS, "y")):
        for val, px in ticks.items():
            lo, hi = (fr.x0 - 2, fr.x1 + 2) if coord == "x" else (fr.y0 - 2,
                                                                  fr.y1 + 2)
            if not lo <= px <= hi:
                raise SystemExit(f"[!] {name} tick {val} at {px} is outside "
                                 f"the detected frame")
    mx, cx, xres = _fit(X_TICKS)
    my, cy, yres = _fit(Y_TICKS)

    curves = []
    for p in pg.get_drawings():
        r = p["rect"]
        if not (r.x0 >= fr.x0 - 4 and r.x1 <= fr.x1 + 4
                and r.y0 >= fr.y0 - 4 and r.y1 <= fr.y1 + 4):
            continue
        col = p.get("color")
        if not col or max(col) > 0.30 or (max(col) - min(col)) > 0.05:
            continue
        if sum(1 for it in p["items"] if it[0] in ("l", "c")) < 8:
            continue
        if r.width < 0.4 * fr.width or r.height < 0.15 * fr.height:
            continue
        pts = dd.flatten(p["items"], n=40)
        xs = np.array([q[0] for q in pts], float)
        ys = np.array([q[1] for q in pts], float)
        o = np.argsort(xs)
        le = mx * xs[o] + cx
        den = my * ys[o] + cy
        keep = np.concatenate(([True], np.diff(le) > 1e-9))
        curves.append((le[keep], den[keep]))
    return curves, dict(x_res=xres, y_res=yres, frame=fr)


def measure(curves):
    """[(gamma, base+fog, logE span, n)] per curve, sorted by gamma."""
    out = []
    for le, den in curves:
        fog = float(np.median(den[le < le.min() + PLATEAU_DEC]))
        m = ((den - fog) > NET_LO) & ((den - fog) < NET_HI)
        if m.sum() < 6:
            continue
        g = float(np.polyfit(le[m], den[m], 1)[0])
        out.append((g, fog, float(le[-1] - le[0]), int(len(le))))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()
    try:
        curves, diag = extract(Path(ns.root).resolve())
    except FileNotFoundError as exc:
        print(f"  [SKIP] source not present: {exc}")
        return 0
    print(f"[i] {SOURCE}")
    print(f"    axis residuals {diag['x_res']:.4f} decade / {diag['y_res']:.4f} D; "
          f"{len(curves)} curves traced inside the frame")
    got = measure(curves)
    bad = 0
    if len(got) != len(PRINTED):
        print(f"    [FAIL] traced {len(got)} usable curves, the sheet prints "
              f"{len(PRINTED)} development times")
        return 1 if ns.do_assert else 0
    # ⚠ ASSOCIATION BY CONTRAST ORDER, stated rather than assumed: a longer
    # development gives a steeper curve, so the curves ranked by slope are the
    # times ranked by gamma. Nothing in the PDF links a label to a path.
    for (g, fog, span, n), (minutes, printed) in zip(got, PRINTED):
        wg, wf = EXPECTED[minutes]
        drift = (abs(g - wg) > TOL_GAMMA) or (abs(fog - wf) > TOL_FOG)
        vs = abs(g - printed)
        ok_printed = vs <= PRINTED_TOL or minutes in PRINTED_EXEMPT
        print(f"    {minutes:>4} min: gamma {g:.4f} (printed {printed:.2f}, "
              f"{'agrees' if vs <= PRINTED_TOL else f'OFF BY {vs:+.3f}'}), "
              f"base+fog {fog:.4f}, {n} samples over {span:.2f} decades")
        if drift:
            print(f"      [FAIL] moved from the recorded {wg:.4f} / {wf:.4f}")
            bad += 1
        if not ok_printed:
            print(f"      [FAIL] disagrees with the printed gamma and is not "
                  f"the recorded exemption")
            bad += 1
        elif minutes in PRINTED_EXEMPT and vs > PRINTED_TOL:
            print(f"      [i] known disagreement: {PRINTED_EXEMPT[minutes]}")
    print()
    if bad:
        print(f"[FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("[OK] the printed time-gamma series is reproduced from the drawn "
          "curves (4 of 5 within 2 %, the 9-minute disagreement recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
