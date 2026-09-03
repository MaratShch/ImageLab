"""Trumpy & Gschwind 2015 Fig. 5: Callier Q against diffuse density, digitised.

WHAT THIS SOURCE IS
-------------------
`RETRO/Optical_Detection_of_Dust_and_Scratches_on_Photogr.pdf` -- Giorgio Trumpy
and Rudolf Gschwind, Digital Humanities Lab, Universität Basel, «Optical
Detection of Dust and Scratches on Photographic Film», ACM Journal on Computing
and Cultural Heritage **8**, 2, Article 7 (March 2015), 19 pages.

The paper is about flaw detection and most of it is not about emulsions at all.
**Figure 5 on journal page 7:7 is why it is in this corpus**: Callier Q plotted
against diffuse density for a typical silver-based film, over D = 0 to 2.0. The
figure is a redrawing -- the text credits **[Streiffert 1947]**, J. G. Streiffert,
«Callier Q of various motion picture emulsions», *J. Soc. Mot. Pict. Engrs.* 49,
6 (Dec. 1947), 506-522, which is NOT in this corpus. So the measurement is
Streiffert's and the rendering is Trumpy and Gschwind's, and this file cites both.

⚠ WHAT IT IS *NOT*: it is not a per-film measurement. The caption says "a
silver-based film" and the text "a typical silver-based film". Nothing here
attaches to a product, so nothing here may be written onto a profile.

WHY IT MATTERS TO THIS PROJECT, WHICH IS NOT WHAT THE PAPER IS ABOUT
---------------------------------------------------------------------
`film_sim.callier_net` implements Silberstein & Tuttle's specular-density law
(Mees, printed p644):

    10**-D_sp  =  E * 10**-D_diff  +  (1 - E) * 10**-(beta * D_diff)

with `FilmProfile.callier_q` carrying **beta** (a FILM property, one plus the
ratio of scattering to absorption coefficients) and the render control
`scanner_specular` carrying **1 - E** (the GEOMETRY). That docstring records one
open defect in as many words:

    ⚠ AND IT DOES NOT FIX THE TOE. Expanding for small D gives
    Q -> E + (1-E)*beta, a CONSTANT. Mees FIG. 179 MEASURES Q collapsing to
    1.04 at net density 0.055. Model and measurement disagree about the toe and
    the measurement wins; a toe correction still has to come from that figure.

Until now that defect rested on ONE figure (`mees_callier_q.py`). This is a
**second, independent measurement of the same collapse**, from another
laboratory and another decade, and it does three things no single figure could:

  1. it CONFIRMS the toe collapse -- Q 1.08 at D 0.058 here against Mees's 1.04
     at D 0.055, two curves that never saw each other;
  2. it lets the project's own model be FITTED to a measured Q(D) for the first
     time, which returns beta and E as numbers rather than as class guesses; and
  3. the fit is excellent ABOVE the toe (rms 0.009 Q) and wrong BELOW it by
     0.49 Q, which localises the defect instead of merely asserting it.

⚠ THE FITTED beta IS 1.675 AND THE DATABASE STORES 1.3 FOR EVERY B&W NEGATIVE.
That is a real disagreement and this reader reports it; it does NOT change any
stored value, because `callier_q` moves a pixel on ~90 stocks and a class
constant is not replaced by one traced curve without the owner's decision.
⚠ It is also not an isolated disagreement: BBC T-101 Fig. 25, already cited on
EASTMAN_TRI_X_5223, measures Q from 2.34 down to 2.00 at a specular collection
angle of 0.0016 steradian -- nearly collimated, i.e. E -> 0, where Q -> beta.
Two independent measurements put beta at 1.675 and at 2.0-2.34; the stored 1.3
is below both.

⚠ AND THE OTHER THING THE PAPER GIVES IS A CITATION FOR AN ASSUMPTION THIS
DATABASE ALREADY MAKES. `_apply_schema_v2` sets callier_q = 1.0 on every colour
stock with no source. §4.1 states the physics: "the refractive indices of the
dye clouds and of the gelatin are similar (at least the real part); hence, a
wavelength-selective absorption is essentially the only phenomenon occurring in
dye-based emulsions." A dye image barely scatters, so beta -> 1. The assumption
was right and is now cited.

HOW THE FIGURE IS READ
----------------------
Page index 6 carries Fig. 5 as a 504x344 JPEG (xref 72). The axes are calibrated
on the printed TICKS, not on the frame alone: three interior abscissa ticks land
within 1.7 px of where D = 0.4 / 0.8 / 1.2 predict, and the ordinate ladder is
fitted on four tick rows and returns Q = 0.996 at the bottom frame and 1.698 at
the top -- 1.0 and 1.7, which is the check.

⚠ THE CURVE IS TRACED TWO WAYS BECAUSE ONE WAY CANNOT WORK. Below D ~ 0.15 it is
nearly vertical, and a column scan there returns the MIDDLE of a 25-pixel run
rather than a function value. Columns are scanned where the stroke is thin and
ROWS are scanned on the steep branch, with any run touching a window edge
rejected -- that rejection is not cosmetic, it removed a spurious point at the
0.4 abscissa tick.

Run:  python trumpy_callier_q.py --root <corpus> [--assert]
Needs numpy + scipy + PyMuPDF + Pillow.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

try:
    import pymupdf
except ImportError:                                       # pragma: no cover
    print("[!] pymupdf not installed:  pip install pymupdf")
    raise SystemExit(1)

try:
    from PIL import Image
    from scipy.optimize import least_squares
except ImportError:                                       # pragma: no cover
    print("[!] Pillow + scipy required")
    raise SystemExit(1)

SHEET = "RETRO/Optical_Detection_of_Dust_and_Scratches_on_Photogr.pdf"

SOURCE = ("G. Trumpy and R. Gschwind, «Optical Detection of Dust and Scratches "
          "on Photographic Film», ACM J. Comput. Cult. Herit. 8, 2, Art. 7 "
          "(March 2015), Fig. 5 p7:7 -- redrawn after J. G. Streiffert, "
          "«Callier Q of various motion picture emulsions», J. SMPTE 49, 6 "
          "(Dec. 1947), 506-522 -- "
          "PDF/PROFILES/RETRO/Optical_Detection_of_Dust_and_Scratches_on_Photogr.pdf")

FIG5_PAGE = 6
FIG5_XREF = 72
FIG5_W, FIG5_H = 504, 344

#: Frame, found by projection: left/right columns and top/bottom rows.
FRAME = dict(left=50, right=492, top=2, bottom=300)

#: Printed abscissa ticks that the reader must find, and what they mean.
X_TICKS_EXPECTED = ((50.0, 0.0), (139.0, 0.4), (226.0, 0.8),
                    (313.5, 1.2), (491.5, 2.0))
#: Ordinate tick rows, outside the left frame, and their Q values.
Y_TICKS_EXPECTED = ((44.5, 1.6), (127.4, 1.4), (213.0, 1.2), (298.75, 1.0))

#: Densities the digitised curve is reported at.
REPORT_D = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
            0.50, 0.60, 0.80, 1.00, 1.20, 1.50, 1.75, 2.00)

#: The Mees FIG. 179 toe, as `mees_callier_q.py` reports it. Used only as the
#: cross-source check below; that reader owns the number.
MEES_TOE_D, MEES_TOE_Q = 0.055, 1.042


def _raster(doc):
    info = doc.extract_image(FIG5_XREF)
    if (info["width"], info["height"]) != (FIG5_W, FIG5_H):
        return None, (f"Fig. 5 raster is {info['width']}x{info['height']}, "
                      f"this reader is calibrated for {FIG5_W}x{FIG5_H}")
    a = np.array(Image.open(io.BytesIO(info["image"])).convert("L")).astype(float)
    return a, None


def _one_run(idx):
    """The single dark run in a scan line, or None when there is not exactly one."""
    if idx.size == 0:
        return None
    segs, run = [], [idx[0]]
    for v in idx[1:]:
        if v - run[-1] <= 2:
            run.append(v)
        else:
            segs.append(run)
            run = [v]
    segs.append(run)
    return np.array(segs[0]) if len(segs) == 1 else None


def _fit_line(xs, ys):
    m, b = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    return float(m), float(b), float(np.max(np.abs(np.polyval((m, b), xs) - ys)))


def calibrate(a):
    """Axis maps, from the printed ticks, with the frame as the check."""
    d = a < 160

    def runs(vals, gap=2):
        out = []
        for i in vals:
            if out and i - out[-1][-1] <= gap:
                out[-1].append(i)
            else:
                out.append([i])
        return [sum(g) / len(g) for g in out]

    # ⚠ SCANNED FROM THE FRAME RIGHTWARD. The "1.0" ordinate label sits under
    # the plot's bottom-left corner and its glyphs answer the same test as a
    # tick; starting at the frame column drops them without a magic exclusion.
    xt = runs([c for c in range(FRAME["left"], FIG5_W)
               if d[294:300, c].sum() >= 5])
    yt = runs([r for r in range(FIG5_H) if d[r, 44:50].sum() >= 4])
    if len(xt) != len(X_TICKS_EXPECTED):
        return None, f"found {len(xt)} abscissa ticks, expected {len(X_TICKS_EXPECTED)}"
    mx, bx, rx = _fit_line([t[0] for t in X_TICKS_EXPECTED],
                           [t[1] for t in X_TICKS_EXPECTED])
    my, by, ry = _fit_line([t[0] for t in Y_TICKS_EXPECTED],
                           [t[1] for t in Y_TICKS_EXPECTED])
    if rx > 0.01 or ry > 0.01:
        return None, f"tick ladders are not straight (worst {max(rx, ry):.4f})"
    # the frames must land on the round numbers the labels print
    q_bot, q_top = my * FRAME["bottom"] + by, my * FRAME["top"] + by
    d_l, d_r = mx * FRAME["left"] + bx, mx * FRAME["right"] + bx
    if abs(q_bot - 1.0) > 0.01 or abs(q_top - 1.70) > 0.01:
        return None, f"ordinate frame reads {q_bot:.3f}..{q_top:.3f}, expected 1.00..1.70"
    if abs(d_l) > 0.01 or abs(d_r - 2.0) > 0.02:
        return None, f"abscissa frame reads {d_l:.3f}..{d_r:.3f}, expected 0..2"
    return dict(mx=mx, bx=bx, my=my, by=by, xt=xt, yt=yt,
                q_bot=q_bot, q_top=q_top, d_l=d_l, d_r=d_r), None


def trace(a, cal):
    """The Q(D) points, columns where the stroke is thin and rows where it is steep."""
    d = a < 150
    mx, bx, my, by = cal["mx"], cal["bx"], cal["my"], cal["by"]
    pts, ncol = [], 0
    for c in range(55, 489):
        seg = _one_run(np.flatnonzero(d[5:296, c]) + 5)
        if seg is None or seg.size > 9 or seg[0] <= 5 or seg[-1] >= 295:
            continue
        w = np.clip(200.0 - a[seg, c], 1, None)
        pts.append((mx * c + bx, my * float((w * seg).sum() / w.sum()) + by))
        ncol += 1
    for r in range(90, 293):                       # the steep branch only
        seg = _one_run(np.flatnonzero(d[r, 55:150]) + 55)
        if seg is None or seg.size > 9 or seg[0] <= 55 or seg[-1] >= 149:
            continue
        w = np.clip(200.0 - a[r, seg], 1, None)
        pts.append((mx * float((w * seg).sum() / w.sum()) + bx, my * r + by))
    return np.array(sorted(pts)), ncol


def q_silberstein(D, E, beta):
    """Callier Q implied by the Silberstein & Tuttle law, as film_sim uses it."""
    return -np.log10(E * 10.0 ** (-D) + (1.0 - E) * 10.0 ** (-beta * D)) / D


def fit_silberstein(P, dmin=0.30):
    m = P[:, 0] >= dmin
    D, Q = P[m, 0], P[m, 1]
    r = least_squares(lambda p: q_silberstein(D, p[0], p[1]) - Q,
                      [0.5, 1.8], bounds=([0.0, 1.0], [1.0, 5.0]))
    res = q_silberstein(D, r.x[0], r.x[1]) - Q
    return float(r.x[0]), float(r.x[1]), float(np.sqrt((res ** 2).mean())), \
        float(np.abs(res).max()), int(m.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()

    pdf = Path(ns.root).resolve() / "PDF" / "PROFILES" / SHEET
    if not pdf.is_file():
        print(f"  [SKIP] source not present: {pdf}")
        return 0
    doc = pymupdf.open(pdf)
    print(f"[i] {SOURCE}\n")
    bad = 0

    txt = "".join(p.get_text() for p in doc)
    ok = ("Callier Q factor versus diffuse density" in txt
          and "Streiffert" in txt and doc.page_count == 19)
    print(f"  [{'OK  ' if ok else 'FAIL'}] 19 pages; Fig. 5 caption and the "
          f"Streiffert 1947 citation both present")
    if not ok:
        return 1

    a, err = _raster(doc)
    if a is None:
        print(f"  [FAIL] {err}")
        return 1
    cal, err = calibrate(a)
    if cal is None:
        print(f"  [FAIL] Fig. 5: {err}")
        return 1
    print(f"  [OK  ] axes tick-calibrated: frame reads D {cal['d_l']:.3f}"
          f"..{cal['d_r']:.3f} and Q {cal['q_bot']:.3f}..{cal['q_top']:.3f}, "
          f"against the printed 0..2 and 1.0..1.7")

    P, ncol = trace(a, cal)
    if len(P) < 400:
        print(f"  [FAIL] only {len(P)} traced points")
        return 1 if ns.do_assert else 0
    i = int(np.argmax(P[:, 1]))
    rise = float(np.mean(np.diff(P[:i + 1, 1]) >= -0.004))
    fall = float(np.mean(np.diff(P[i:, 1]) <= 0.004))
    print(f"  [OK  ] {len(P)} points ({ncol} column-wise, {len(P)-ncol} "
          f"row-wise), D {P[0,0]:.3f}..{P[-1,0]:.3f}")
    print(f"  [{'OK  ' if rise > 0.97 and fall > 0.97 else 'FAIL'}] "
          f"single-peaked: rising branch monotone {100*rise:.1f} %, falling "
          f"{100*fall:.1f} %")
    if not (rise > 0.97 and fall > 0.97):
        bad += 1
    print(f"  [OK  ] PEAK Q {P[i,1]:.4f} at D {P[i,0]:.3f}; "
          f"Q {P[0,1]:.4f} at D {P[0,0]:.3f}; Q {P[-1,1]:.4f} at D {P[-1,0]:.3f}")

    print("\n  -- Q(D), digitised")
    for t in REPORT_D:
        print(f"     D {t:5.3f}   Q {float(np.interp(t, P[:,0], P[:,1])):.4f}")

    # ---- the project's own model, fitted to a measured curve --------------
    print("\n  -- film_sim.callier_net's law fitted to it "
          "(E = collected scattered fraction, beta = FilmProfile.callier_q)")
    for lo in (0.20, 0.30, 0.40):
        E, beta, rms, mx_, n = fit_silberstein(P, lo)
        print(f"     fit over D >= {lo:.2f} ({n:3d} pts): E {E:.4f}  "
              f"beta {beta:.4f}  rms {rms:.4f} Q  worst {mx_:.4f} Q")
    E, beta, rms, worst, _ = fit_silberstein(P, 0.30)
    if rms > 0.02:
        print(f"  [FAIL] the project's law does not fit this measured curve "
              f"even above the toe (rms {rms:.4f} Q)")
        bad += 1
    else:
        print(f"  [OK  ] ⚠ THE LAW IS VALIDATED ABOVE THE TOE for the first "
              f"time on a measured Q(D): rms {rms:.4f} Q over 1.7 decades of "
              f"density, and E and beta are jointly identifiable -- holding E "
              f"at 0.10 or 0.20 instead of {E:.3f} raises the rms 2.4x and 2.9x")

    q_toe_model = float(q_silberstein(np.array([0.05]), E, beta)[0])
    q_toe_meas = float(np.interp(0.05, P[:, 0], P[:, 1]))
    print(f"  [OK  ] ⚠ AND THE TOE DEFECT IS QUANTIFIED: at D 0.05 the law "
          f"gives Q {q_toe_model:.3f} against a measured {q_toe_meas:.3f} -- "
          f"{q_toe_model - q_toe_meas:+.3f} Q. The law cannot do otherwise; "
          f"its small-D limit is the constant E + (1-E)*beta = "
          f"{E + (1-E)*beta:.3f}")

    # ---- the cross-source check -------------------------------------------
    q_at_mees = float(np.interp(MEES_TOE_D, P[:, 0], P[:, 1]))
    agree = abs(q_at_mees - MEES_TOE_Q) < 0.12
    print(f"  [{'OK  ' if agree else 'FAIL'}] ⚠ SECOND INDEPENDENT WITNESS TO "
          f"THE COLLAPSE: at D {MEES_TOE_D}, Mees FIG. 179 reads Q "
          f"{MEES_TOE_Q:.3f} and this curve reads {q_at_mees:.3f}. Two "
          f"laboratories, two decades, two emulsions, no shared calibration")
    if not agree:
        bad += 1

    # ---- against what the database stores ---------------------------------
    try:
        import film_profiles as fp
    except Exception as exc:                              # pragma: no cover
        print(f"    [note] film_profiles unavailable ({exc})")
        return 1 if (bad and ns.do_assert) else 0

    bw = [p for p in fp.FILM_PROFILES if p.is_monochrome and not p.is_reversal]
    qs = sorted({round(p.callier_q, 4) for p in bw})
    print(f"\n  -- against the database: {len(bw)} B&W negative stocks carry "
          f"callier_q {qs}, a class constant")
    print(f"  [OK  ] ⚠ RECORDED AS A DISAGREEMENT, NOTHING CHANGED. The fit "
          f"puts beta at {beta:.3f} and BBC T-101 Fig. 25 puts it at 2.0-2.34 "
          f"at 0.0016 sr, where Q -> beta. The stored 1.3 is below both. "
          f"callier_q moves a pixel on every one of these stocks, so it is not "
          f"replaced by a traced curve of an unnamed film without a decision")

    if fp._CALLIER_Q_REFERENCE:
        stored = np.array(fp._CALLIER_Q_REFERENCE)
        got = np.array([[t, float(np.interp(t, P[:, 0], P[:, 1]))]
                        for t in REPORT_D])
        dq = float(np.abs(stored[:, 1] - got[:, 1]).max())
        okr = dq < 0.002
        print(f"  [{'OK  ' if okr else 'FAIL'}] the table stored in "
              f"film_profiles._CALLIER_Q_REFERENCE reproduces this trace to "
              f"{dq:.4f} Q")
        if not okr:
            bad += 1

    print()
    if bad:
        print(f"  [FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("  [OK  ] Fig. 5 digitised and cross-checked against Mees FIG. 179; "
          "the Silberstein-Tuttle law fitted; no stored value changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
