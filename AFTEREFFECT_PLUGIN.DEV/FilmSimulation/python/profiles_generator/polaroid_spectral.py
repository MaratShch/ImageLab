"""Spectral sensitivity from the POLAROID film data sheets (queue item E2).

Four sheets, one panel each, all pure VECTOR (zero page images): Polapan Pro 100
(664), the ISO 3000 pack film (667), Type 52 and Type 55 P/N. Two of the four
stocks already carry a pan set and are read here as cross-checks; **Type 52 and
Type 55 are new data**.

⚠ THE ORDINATE IS *NOT* INVERTED, AND THE QUEUE SAID IT WAS
------------------------------------------------------------
Queue item E2 recorded, as its headline trap: *"the sheets plot the equivalent
energy needed ... so sensitivity is its reciprocal and log sensitivity is its
NEGATION -- the peak of the curve is its LOWEST drawn point"*, and warned that
getting the sign wrong yields a mirrored set that passes every band and ordering
check. **The warning is right and the diagnosis is backwards.** The plotted
quantity is sensitivity, read directly; negating it is what would produce the
mirrored set.

The row was reading the sheets' PROSE, which does say "Spectral Sensitivity:
Shows the equivalent energy needed at each wavelength in order to activate the
emulsion so that it produces a neutral density of .75". That sentence describes
the MEASUREMENT. What Polaroid PLOTS is its reciprocal. Three independent
pieces of evidence, in increasing order of decisiveness:

  1. **The axis on 667 is captioned in units**: "Spectral Sensitivity
     (cm^2/erg)". Sensitivity is area per unit energy; energy required would be
     erg/cm^2. The sheet states the direction itself.
  2. **Speed ordering across the four sheets settles it quantitatively.** Read
     as drawn, the peak plotted value rises monotonically with exposure index
     across a 60x span of speed: Type 55 (EI 50) 9.8, 664 (EI 100) 15.6, Type 52
     (EI 400) 98.0, 667 (EI 3000) 233.1. Read as energy-required, the ISO 3000
     film would need FIFTEEN TIMES MORE light than the ISO 100 film to reach the
     same density. That is not a matter of interpretation, and it is asserted at
     the end of this module rather than left as an argument.
  3. **The red edge.** Every one of these curves falls steeply at its long-wave
     end, which is a sensitising dye running out. Under the inverted reading the
     films would reach PEAK sensitivity exactly at the wavelength where the
     plotted data stops, which no emulsion does.

A fourth, weaker check agrees: 664's own printed filter factors run W25 red 5.6,
W47 blue 6.3, W58 green 10 in daylight -- red the cheapest. Read as drawn the
curve has a red hump above its green trough, which predicts that ordering; read
inverted it predicts the opposite.

⚠ AND THE ALREADY-ADOPTED SETS WERE READ THE RIGHT WAY ROUND. `POLAROID_664`
and `POLAROID_667` were adopted before this module existed and are NOT negated:
664 stores its maximum at 380 nm, which is where the drawn curve is highest.
Following the queue row would have mirrored two correct sets in the course of
"fixing" them. Their `criterion` string, `log_energy_for_neutral_density_0.75`,
names the measurement rather than the stored quantity, which is what misled the
row; it is left alone because it is what the sheet's prose says, and the note
above is the correction.

THE "0" TICK IS NOT A DECADE AND MUST BE DROPPED
-------------------------------------------------
Three of the four y axes print, below the "1" tick and one full decade further
down, a tick labelled **"0"**. On a logarithmic axis that position is 0.1, and
zero has no place on it at all. Fitting it as a value would put log10(0) into
the axis solve; fitting it as 0.0 would bend the whole scale. Only 1 / 10 / 100
/ 1000 are used, and `dye_density._fit_axis` drops the one label per axis that
Polaroid nudged (664's "1000" sits 2.4 pt off the decade line it belongs to).

WHAT THIS MODULE DOES NOT NEED, AND THE QUEUE SAID IT DID
----------------------------------------------------------
⚠ E2 recorded that "667 and 55 place the panel on the RIGHT of p3 with a
different layout and the label sweep returns one or two labels each -- they need
their own windows". They do not. All four panels are found by the same rule,
because the rule keys on the AXIS LABELS rather than on a page region: a column
of {1, 10, 100, 1000} and, below it, a row of wavelength labels. Both sheets
return complete label sets that way -- 667 gives four y labels at a dead-uniform
40.3 pt per decade and five x labels. What defeated the earlier sweep was a
fixed window, not the layout.

Run:  python polaroid_spectral.py [--root .] [--assert] [--dump]
Needs numpy + PyMuPDF. --assert exits non-zero if an extraction moves.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import dye_density as dd

#: The database grid these sets are stored on.
GRID = np.arange(380, 701, 10, dtype=float)

#: "at or below the measurement floor of the source plot", as elsewhere.
FLOOR = -4.0

#: The decade labels an axis is fitted on. ⚠ "0" is deliberately absent: see
#: the module header. It is printed on three of the four sheets and it is not a
#: decade.
Y_DECADES = (1.0, 10.0, 100.0, 1000.0)

#: Wavelength labels, in nm.
X_TICKS = tuple(float(v) for v in range(350, 801, 50))

#: Two labels are in the same column / row when their centres are within this.
AXIS_CLUSTER_PT = 6.0

#: A traced path counts as the curve when it is this blue. ⚠ THE FOUR SHEETS DO
#: NOT SHARE ONE BLUE: 664 and 52 stroke at (0.0012, 0.0, 0.9999) and 667 and 55
#: at exactly (0, 0, 1). Matching an exact palette entry -- which is what the
#: Kodak reader does, correctly, because Kodak's ink convention is a palette --
#: would have read two sheets and silently skipped two.
def _is_blue(col) -> bool:
    return bool(col) and col[2] > 0.85 and col[0] < 0.15 and col[1] < 0.15


#: (pdf under PDF/PROFILES/POLAROID, page, profile).
SHEETS = {
    "664": ("664fds.pdf", 3, "POLAROID_664"),
    "667": ("667fds.pdf", 3, "POLAROID_667"),
    "52": ("52fds.pdf", 3, "POLAROID_52"),
    "55": ("55fds.pdf", 3, "POLAROID_55_PN_NEG"),
}

#: Recorded 2026-08-31. (peak nm, peak PLOTTED value, measured sample count).
#: ⚠ THE PEAK PLOTTED VALUE IS KEPT BECAUSE THE STORED ARRAYS THROW IT AWAY and
#: it is the evidence for the whole reading: peak-normalisation erases the fact
#: that 667 sits 14x above 664, which is the speed ordering that proves the axis
#: is sensitivity and not energy -- 667 peaks at 233.1 against 664's 15.6 while
#: being 30x the film speed. See the module header.
EXPECTED = {
    "664": (380.0, 15.57, 28),
    "667": (430.0, 233.12, 29),
    "52": (380.0, 97.96, 27),
    "55": (380.0, 9.76, 28),
}

#: rms agreement, in decades, against the set the profile ALREADY holds.
#: Asserted, so a later change to either side shows up as drift.
#: ⚠ BOTH AGREE, AND THAT IS THE FOURTH PROOF OF THE READING DIRECTION. These
#: two sets were hand-read years before this module and are stored UNNEGATED;
#: this reader, working from the vector paths, reproduces them to 0.034 and
#: 0.027 decades. Had the queue's inverted reading been right, one of the two
#: readings would have to be a mirror of the other, and a mirror does not agree
#: with its original to a thirtieth of a decade.
EXPECTED_VS_STORED = {
    "664": 0.0343,
    "667": 0.0273,
}

#: How far a pinned agreement may move before the audit calls it drift.
RMS_TOL = 0.015

#: How far a pinned peak value may move, as a fraction.
PEAK_TOL = 0.02


def find_axes(pg):
    """(x_labels, y_labels) for the spectral panel, or (None, None).

    ⚠ KEYED ON THE LABELS, NOT ON A PAGE REGION. See the module header: a fixed
    window is what made two of these four sheets look unreadable.
    """
    xs, ys = [], []
    for w in pg.get_text("words"):
        x0, y0, x1, y1, t = w[0], w[1], w[2], w[3], w[4]
        if not re.fullmatch(r"\d+", t):
            continue
        v = float(t)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if v in X_TICKS:
            xs.append((v, cx, cy))
        if v in Y_DECADES:
            ys.append((v, cx, cy))
    # the y decades form a COLUMN: same cx, four different cy
    cols: dict[float, list] = {}
    for v, cx, cy in ys:
        cols.setdefault(round(cx / AXIS_CLUSTER_PT) * AXIS_CLUSTER_PT, []).append((v, cy))
    ycol = None
    for _k, items in cols.items():
        if len({v for v, _ in items}) >= 3:
            if ycol is None or len(items) > len(ycol):
                ycol = items
    if ycol is None:
        return None, None
    y_lo = min(cy for _v, cy in ycol)
    y_hi = max(cy for _v, cy in ycol)
    # the wavelength labels form a ROW BELOW that column
    rows: dict[float, list] = {}
    for v, cx, cy in xs:
        if cy < y_hi - 4:
            continue                       # a row above the panel: another plot
        rows.setdefault(round(cy / AXIS_CLUSTER_PT) * AXIS_CLUSTER_PT, []).append((v, cx))
    xrow = None
    for _k, items in sorted(rows.items()):
        if len({v for v, _ in items}) >= 3:
            xrow = items
            break
    if xrow is None or not (y_lo < y_hi):
        return None, None
    return xrow, ycol


def extract_sheet(root: Path, tag: str):
    import pymupdf
    fn, pgno, prof = SHEETS[tag]
    pdf = root / "PDF" / "PROFILES" / "POLAROID" / fn
    if not pdf.is_file():
        return None, f"source not present: {fn}"
    pg = pymupdf.open(pdf)[pgno - 1]
    xrow, ycol = find_axes(pg)
    if xrow is None:
        return None, "no spectral panel axes found"
    fx = dd._fit_axis({v: cx for v, cx in xrow})
    fy = dd._fit_axis({np.log10(v): cy for v, cy in ycol})
    if fx is None or fy is None:
        return None, "axis fit failed"
    if fx[2] > dd.TICK_RESID_PT:
        return None, f"wavelength ticks not collinear ({fx[2]:.2f} pt)"
    if fy[2] > dd.TICK_RESID_PT:
        return None, f"sensitivity decades not collinear ({fy[2]:.2f} pt)"
    lo_x, hi_x = min(v for v, _ in xrow), max(v for v, _ in xrow)
    lo_y = np.log10(min(v for v, _ in ycol))
    hi_y = np.log10(max(v for v, _ in ycol))
    cal = (fx[0] * lo_x + fx[1], lo_x, fx[0] * hi_x + fx[1], hi_x,
           fy[0] * lo_y + fy[1], lo_y, fy[0] * hi_y + fy[1], hi_y)

    # the curve: the longest blue path inside the axes
    x_lo, x_hi = min(cx for _v, cx in xrow) - 8, max(cx for _v, cx in xrow) + 8
    y_lo_pt = min(cy for _v, cy in ycol) - 10
    y_hi_pt = max(cy for _v, cy in ycol) + 14
    best = None
    for p in pg.get_drawings():
        if not _is_blue(p.get("color")):
            continue
        r = p["rect"]
        if not (r.x0 >= x_lo and r.x1 <= x_hi and r.y0 >= y_lo_pt and r.y1 <= y_hi_pt):
            continue
        n = sum(1 for it in p["items"] if it[0] in ("l", "c"))
        if n < 20:
            continue
        if best is None or n > best[0]:
            best = (n, p)
    if best is None:
        return None, "no blue curve inside the panel axes"
    pts = dd.flatten(best[1]["items"])
    # dd.resample maps y through the calibration, which here is LOG10 of the
    # plotted value -- i.e. it returns log10(sensitivity) directly.
    log_s = dd.resample(pts, cal, GRID)
    lam = [(x - cal[0]) / (cal[2] - cal[0]) * (cal[3] - cal[1]) + cal[1]
           for x, _y in pts]
    ext = (min(lam), max(lam))
    inside = (GRID >= ext[0] - 1e-9) & (GRID <= ext[1] + 1e-9)
    if not inside.any():
        return None, "the traced extent misses the stored grid"
    peak_log = float(log_s[inside].max())
    norm = np.where(inside, log_s - peak_log, FLOOR)
    norm = np.clip(norm, FLOOR, 0.0)
    meas = norm > FLOOR + 1e-9
    return dict(tag=tag, profile=prof, file=fn, page=pgno,
                log_s_pan=norm, peak=10.0 ** peak_log,
                lam=float(GRID[int(np.argmax(np.where(meas, norm, -np.inf)))]),
                n_meas=int(meas.sum()), extent=ext,
                x_resid=fx[2], y_resid=fy[2],
                n_x=len(xrow), n_y=len(ycol)), None


def _rms(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    a, b = a[:n], b[:n]
    m = (a > FLOOR + 0.01) & (b > FLOOR + 0.01)
    if m.sum() < 3:
        return float("nan")
    return float(np.sqrt(((a[m] - b[m]) ** 2).mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()
    bad = skipped = 0
    got_all = {}
    print(f"[i] corpus root {root}")
    for tag in SHEETS:
        got, err = extract_sheet(root, tag)
        if got is None:
            if "not present" in (err or ""):
                print(f"  [SKIP] {tag}: {err}")
                skipped += 1
            else:
                print(f"  [FAIL] {tag}: {err}")
                bad += 1
            continue
        got_all[tag] = got
        want_lam, want_peak, want_n = EXPECTED[tag]
        ok = (got["lam"] == want_lam
              and abs(got["peak"] - want_peak) <= PEAK_TOL * want_peak
              and got["n_meas"] == want_n)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {tag:4s} {got['profile']:20s} "
              f"peak {got['peak']:8.2f} @ {got['lam']:.0f} nm, "
              f"{got['n_meas']} samples, extent "
              f"{got['extent'][0]:.0f}-{got['extent'][1]:.0f} nm, "
              f"ticks {got['n_x']}x/{got['n_y']}y "
              f"resid {got['x_resid']:.2f}/{got['y_resid']:.2f} pt")
        if not ok:
            print(f"         expected peak {want_peak} @ {want_lam:.0f} nm, "
                  f"{want_n} samples")
            bad += 1
        if tag in EXPECTED_VS_STORED:
            from film_profiles import get_profile
            st = get_profile(got["profile"]).spectral.log_s_pan
            r = _rms(st, got["log_s_pan"])
            drift = not (abs(r - EXPECTED_VS_STORED[tag]) < RMS_TOL)
            print(f"         vs the ADOPTED set: rms {r:.4f} decades "
                  f"({'DRIFTED' if drift else 'as recorded'})")
            if drift:
                bad += 1
        if ns.dump:
            print("            log_s_pan=("
                  + ", ".join("%.2f" % v for v in got["log_s_pan"]) + "),")

    # ---- the cross-sheet check that decides the whole reading ---------------
    # ⚠ THIS IS THE EVIDENCE THAT THE AXIS IS SENSITIVITY, NOT ENERGY, and it is
    # asserted rather than argued. Peak plotted value must rise with exposure
    # index across all four sheets. Read the other way round -- the reading
    # queue item E2 prescribed -- the ISO 3000 film would need fifteen times
    # more light than the ISO 100 one.
    if len(got_all) == len(SHEETS):
        from film_profiles import get_profile
        order = sorted(got_all.values(),
                       key=lambda g: get_profile(g["profile"]).exposure_index)
        peaks = [g["peak"] for g in order]
        eis = [get_profile(g["profile"]).exposure_index for g in order]
        ok = all(b > a for a, b in zip(peaks, peaks[1:]))
        print(f"  [{'OK  ' if ok else 'FAIL'}] peak plotted sensitivity rises "
              f"with exposure index: "
              + ", ".join(f"EI {e} -> {p:.1f}" for e, p in zip(eis, peaks)))
        if not ok:
            bad += 1

    print(f"\n[i] {len(SHEETS) + 1 - bad - skipped} reproduced, {bad} failed, "
          f"{skipped} skipped")
    if ns.do_assert and bad:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
