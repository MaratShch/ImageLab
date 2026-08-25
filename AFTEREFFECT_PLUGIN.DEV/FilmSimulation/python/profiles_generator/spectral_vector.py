"""Spectral SENSITIVITY from Kodak brochure VECTOR paths (queue item C10).

Reads the "SPECTRAL SENSITIVITY CURVES" panel off a Kodak H-1 motion-picture
brochure and returns the three layer curves on this database's spectral grid
(380-680 nm, 10 nm, peak-normalised to 0.0, floor -4.0).

WHY THIS SCRIPT EXISTS AT ALL
-----------------------------
Every spectral set in the database until now came from either a 2026-08-02
RASTER batch or `agfa_vista.py`'s dash-legend reader. Nothing could read a
VECTOR sensitivity panel, so the panel on the one sheet whose characteristic
curves, granularity and MTF are all already traced -- H-1-5201 -- sat unread.

THE ASSIGNMENT PROBLEM, AND WHY INK SOLVES IT
---------------------------------------------
Three curves in one frame, nothing printed inside it to say which layer is
which. The sheet's legend names them ("Sensitivity of the yellow / magenta /
cyan dye forming layer") but the legend is text beside the plot, not a label on
a curve. What connects the two is Kodak's ink convention, and it is physical
rather than decorative: EACH TRACE IS DRAWN IN THE COLOUR OF LIGHT IT RESPONDS
TO. The blue-sensitive (yellow-forming) layer is drawn in BLUE ink, the
green-sensitive (magenta-forming) layer in GREEN, and the red-sensitive
(cyan-forming) layer in RED -- which is not one of the four process inks, so
Kodak makes it by overprinting YELLOW UNDER MAGENTA. Two coincident paths, one
curve. `dye_density.extract_inked` already collapses that pair, and this script
reuses it rather than re-deriving the palette.

THE ASSIGNMENT IS THEN CHECKED THREE WAYS, none of which is the ink:
  1. Legend swatches. The 1 pt lines left of the legend text carry the same
     inks: green sits on "magenta dye forming layer", amber on "cyan dye
     forming layer". Kodak's own words, machine-read.
  2. Absorption bands. The three traces must peak in ascending wavelength order
     and each inside its own band. On 5201 they peak at 470 / 540 / 645 nm.
  3. The sibling sheets. 5218 and 5217 are the same product family and their
     spectral sets were adopted from a different method (the raster batch); the
     shapes have to agree to within the reading error of a printed plot.

⚠ WHAT THIS SCRIPT DOES NOT KNOW: THE DENSITY CRITERION
-------------------------------------------------------
The panel's footnote reads, in full: "Sensitivity = reciprocal of exposure
(erg/cm2) required to produce specified density". IT DOES NOT SAY WHAT DENSITY.
The three sets already in the database (5218, 5217, 5219) carry
`criterion="log_reciprocal_erg_cm2_D0.2_above_dmin"`, and checking the sources:
5218 and 5217 print the same unspecified wording, and 5219's footnote is not in
its text layer at all. So the "D0.2 above dmin" half of that string is not
printed on any of the three. Owner decision 2026-08-25: store 5201 as the sheet
prints it, leave the other three alone, and record the discrepancy (method rule
4 -- a conflict is recorded, not averaged, and not quietly propagated).

KNOWN LIMITS, stated rather than discovered later
-------------------------------------------------
The PANEL FINDER, not the ink reader, is what limits coverage. A corpus sweep on
2026-08-25 found a rotated LOG SENSITIVITY caption on only **5 pages** against 24
pages carrying a readable dye panel, so "the ink rule generalises to every
brochure" is true of the identification and NOT yet of the anchor. Measured
failures, with their causes:

  * `Ektachrome_100d.pdf` p4 (5285) -- "no frame right of the axis label". The
    caption sits INSIDE a decorative outer box whose x0 (42.0) is LEFT of the
    label (51.4), so `dye_density.pick`'s "frame must be right of the label"
    rule rejects it, and the real plot frame is not drawn as a separate path.
    Relaxing that rule would let the outer box win, and the tick windows are
    measured from the frame edges, so the calibration would then be wrong
    rather than absent. Needs its own anchor, not a looser tolerance.
  * `KODAK VISION Color Print Film 2383.pdf` p6 -- only 2 sensitivity ticks
    found against the frame; a print stock's panel is laid out differently.
  * Most other sheets draw the caption as OUTLINED VECTOR ART, so there is no
    rotated text to find at all -- the same class of problem that hid
    8532's printed date until 2026-08-23.

Neither 5285 nor 2383 blocks anything: 5285's spectral set is already adopted
from the raster batch and 2383 is a print stock.

Run:  python spectral_vector.py --root ../.. [--assert] [--sheet 5201]
Needs numpy + PyMuPDF. --assert exits non-zero if an extraction moves, or if a
sheet stops agreeing with an already-adopted set.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

import dye_density as dd

#: The database's spectral grid: film_profiles.SpectralSensitivity stores
#: lambda_start_nm 380 with lambda_step_nm 10, 31 samples, on every colour stock.
GRID = np.arange(380, 681, 10, dtype=float)

#: The floor sentinel, from the SpectralSensitivity docstring: "at or below the
#: measurement floor of the source plot". Also used to pad outside the traced
#: extent -- an extrapolated sensitisation tail would be an invention.
FLOOR = -4.0

AXIS_WORDS = {"LOG", "SENSITIVITY"}

#: Peak must land in this band, per layer. Wide enough to be a real test rather
#: than a restatement of the answer: a swap of any two layers fails it.
BANDS = {"b": (380.0, 500.0), "g": (500.0, 590.0), "r": (590.0, 700.0)}

#: sheet tag -> (pdf filename under PDF/PROFILES/KODAK, page, profile)
SHEETS = {
    "5201": ("Kodak VISION2 50D 5201.pdf", 3, "KODAK_VISION2_50D_5201"),
    # ⚠ 5217 IS THE VALIDATION SHEET, ADDED 2026-08-25d, AND ITS SET IS NOT
    # RE-ADOPTED. KODAK_VISION2_200T_5217 already carries a spectral set from the
    # 2026-08-02 RASTER batch -- a different image, a different method, a
    # different author. Re-deriving it from the same sheet's VECTOR paths is
    # therefore a genuine cross-validation of both, and it is the only one
    # available: no other sheet in the corpus has both an adopted spectral set
    # and a vector panel this reader can reach.
    #   AGREEMENT (see EXPECTED_VS_STORED): blue rms 0.049, green 0.091, red
    #   0.109 decades over the mutually-measured samples; peaks within ONE 10 nm
    #   grid step on all three layers (blue identical at 470).
    # That is inside the reading error of a printed plot, so neither method is
    # corrected and the stored arrays are left exactly as they were -- a wash is
    # not a reason to churn adopted data (the same rule dye_density.py applied
    # when its calibration changed).
    "5217": ("5217-Vision2-200T.pdf", 3, "KODAK_VISION2_200T_5217"),
}

#: Recorded 2026-08-25. --assert fails if an extraction stops reproducing.
#: (peak nm per layer r/g/b, absolute peak LOG SENSITIVITY per layer r/g/b).
#: The absolute peaks are kept because the stored arrays throw them away: the
#: schema normalises each layer to 0.0, so the fact that 5201's blue layer is
#: 0.22 decades more sensitive than its green and red ones survives only here
#: and in the profile's source string.
EXPECTED = {
    # Peaks land on the stored 10 nm grid, so 650 here is the sheet's ~645 nm
    # maximum rounded to the nearest sample -- the grid is the database's, not
    # the plot's. Absolute peaks 1.76 / 1.78 / 1.99 decades: the BLUE layer is
    # 0.22 decades more sensitive than the other two, which the stored arrays
    # cannot show because the schema normalises each layer separately.
    "5201": ((650.0, 540.0, 470.0), (1.76, 1.78, 1.99)),
    "5217": ((640.0, 540.0, 470.0), (2.17, 2.43, 2.81)),
}

#: For sheets whose profile ALREADY carries an independently-adopted set: the
#: rms agreement, per layer (r, g, b), in decades, over the samples both call
#: measured. Asserted, so a later change to either side shows up as a drift in
#: the agreement rather than silently replacing one method's numbers.
#: ⚠ THE STORED ARRAYS ARE NOT TOUCHED. This is a check, not an adoption.
EXPECTED_VS_STORED = {
    "5217": (0.109, 0.091, 0.049),
}


def rot_labels(pg):
    """Rotated y-axis captions containing LOG + SENSITIVITY, one per plot.

    Same column-then-vertical-run grouping as `dye_density.rot_labels`, and for
    the same reason: Kodak stacks several plots in one column, each with its own
    rotated caption, so grouping by x-centre alone merges them and the frame
    search then picks the wrong plot. That bug cost the 7239 sheet a fortnight;
    it is not being reintroduced here.
    """
    rot = []
    for x0, y0, x1, y1, t, *_ in pg.get_text("words"):
        if (y1 - y0) > 1.6 * (x1 - x0) and t.upper().strip(",.:*") in AXIS_WORDS:
            rot.append((x0, y0, x1, y1, t.upper().strip(",.:*")))
    cols: dict[float, list] = {}
    for w in rot:
        cols.setdefault(round((w[0] + w[2]) / 2 / 6) * 6, []).append(w)
    out = []
    for _, items in cols.items():
        runs: list[list] = []
        for w in sorted(items, key=lambda w: w[1]):
            if runs and w[1] - max(v[3] for v in runs[-1]) <= dd.LABEL_STACK_GAP:
                runs[-1].append(w)
            else:
                runs.append([w])
        for run in runs:
            if {"LOG", "SENSITIVITY"} <= {w[4] for w in run}:
                out.append((max(i[2] for i in run), min(i[1] for i in run),
                            max(i[3] for i in run), "LOG SENSITIVITY"))
    return out


def axis_cal(pg, fr):
    """(cal tuple, x residual, y residual, n_x, n_y) for one sensitivity frame.

    ⚠ THE TOP TICK LABEL IS NUDGED AND MUST BE ALLOWED TO LOSE. On 5201 the
    y labels 0.0/1.0/2.0/3.0 sit 22.95 pt apart and the 4.0 label sits 3.6 pt
    off that line, because it would otherwise collide with the frame edge. A
    two-point calibration anchored on 0.0 and 4.0 would spread the whole axis by
    16 %. `dd._fit_axis` drops the outlier instead, which is exactly the failure
    mode it was written for on the 5218 dye panel.
    """
    xs: dict[float, float] = {}
    ys: dict[float, float] = {}
    for a, b, c, d, t, *_ in pg.get_text("words"):
        if not re.fullmatch(r"-?\d+(\.\d+)?", t):
            continue
        v = float(t)
        cx, cy = (a + c) / 2, (b + d) / 2
        if (fr.x0 - 8 <= cx <= fr.x1 + 8 and fr.y1 - 2 <= cy <= fr.y1 + 14
                and 200 <= v <= 900):
            xs.setdefault(v, cx)
        if (fr.x0 - 32 <= cx < fr.x0 - 1 and fr.y0 - 10 <= cy <= fr.y1 + 10
                and 0 <= v <= 6):
            ys.setdefault(v, cy)
    if len(xs) < 3:
        return None, f"only {len(xs)} wavelength ticks against the frame"
    if len(ys) < 3:
        return None, f"only {len(ys)} sensitivity ticks against the frame"
    fx = dd._fit_axis(xs)
    fy = dd._fit_axis(ys)
    if fx is None or fy is None:
        return None, "axis fit failed"
    if fx[2] > dd.TICK_RESID_PT:
        return None, f"wavelength ticks not collinear ({fx[2]:.2f} pt)"
    if fy[2] > dd.TICK_RESID_PT:
        return None, f"sensitivity ticks not collinear ({fy[2]:.2f} pt)"
    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = min(ys), max(ys)
    cal = (fx[0] * lo_x + fx[1], lo_x, fx[0] * hi_x + fx[1], hi_x,
           fy[0] * lo_y + fy[1], lo_y, fy[0] * hi_y + fy[1], hi_y)
    return (cal, fx[2], fy[2], len(xs), len(ys)), None


def _trace_extent(pg, cal, fr, ink_name):
    """(lambda_min, lambda_max) of the first path in `ink_name`, page order."""
    for p in pg.get_drawings():
        r = p["rect"]
        if not (r.x0 >= fr.x0 - 6 and r.x1 <= fr.x1 + 6
                and r.y0 >= fr.y0 - 6 and r.y1 <= fr.y1 + 6):
            continue
        if dd._ink(p) != ink_name:
            continue
        if sum(1 for it in p["items"] if it[0] in ("l", "c")) < dd.INK_MIN_SEG:
            continue
        if r.width > 0.98 * fr.width and r.height > 0.98 * fr.height:
            continue
        lam = [(x - cal[0]) / (cal[2] - cal[0]) * (cal[3] - cal[1]) + cal[1]
               for x, _ in dd.flatten(p["items"])]
        return min(lam), max(lam)
    return None


def normalise(raw, extent):
    """Peak-normalise to 0.0, pad outside the traced extent with the floor.

    The raw values are LOG SENSITIVITY as printed. Outside the traced extent the
    plot says nothing at all -- the trace simply stops, usually where it dives
    off the bottom of the frame -- so those samples get FLOOR rather than a
    continuation of the last value, which would invent sensitisation.
    """
    lo, hi = extent
    inside = (GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)
    if not inside.any():
        return None, 0.0
    peak = float(raw[inside].max())
    out = np.where(inside, raw - peak, FLOOR)
    return np.clip(out, FLOOR, 0.0), peak


def extract_sheet(root: Path, tag: str):
    import pymupdf
    fn, pgno, prof = SHEETS[tag]
    pdf = root / "PDF" / "PROFILES" / "KODAK" / fn
    if not pdf.is_file():
        return None, f"source not present: {fn}"
    pg = pymupdf.open(pdf)[pgno - 1]
    for ax in rot_labels(pg):
        fr = dd.pick(pg, ax)
        if fr is None:
            continue
        cal_r, err = axis_cal(pg, fr)
        if cal_r is None:
            continue
        cal = cal_r[0]
        inked = dd.extract_inked(pg, cal, fr, GRID)
        # red = the yellow-under-magenta overprint; assert the pair coincides
        reds = inked.get("yellow", []) + inked.get("magenta", [])
        if len(reds) == 2 and float(np.abs(reds[0] - reds[1]).max()) > 1e-9:
            return None, "the two red-ink paths are not an overprint pair"
        raw = {}
        if len(inked.get("blue", [])) == 1:
            raw["b"] = (inked["blue"][0], _trace_extent(pg, cal, fr, "blue"))
        if len(inked.get("green", [])) == 1:
            raw["g"] = (inked["green"][0], _trace_extent(pg, cal, fr, "green"))
        if reds:
            raw["r"] = (reds[0], _trace_extent(pg, cal, fr, "magenta")
                        or _trace_extent(pg, cal, fr, "yellow"))
        if len(raw) != 3:
            return None, (f"expected 3 inked layers in the frame, got "
                          f"{sorted(raw)} from {sorted(inked)}")
        out, peaks, lams = {}, {}, {}
        for k, (v, ext) in raw.items():
            if ext is None:
                return None, f"no extent for the {k} layer"
            norm, peak = normalise(v, ext)
            if norm is None:
                return None, f"the {k} layer's extent misses the stored grid"
            out[k], peaks[k] = norm, peak
            # the peak is the sample at 0.0 by construction, but it must be
            # sought among the MEASURED samples only -- the floor-padded ones
            # carry no information and an all--inf comparison would silently
            # return index 0, i.e. the left edge of the grid.
            lams[k] = float(GRID[np.argmax(
                np.where(norm > FLOOR + 1e-9, norm, -np.inf))])
        for k, (lo, hi) in BANDS.items():
            if not lo <= lams[k] <= hi:
                return None, (f"the {k} layer peaks at {lams[k]:.0f} nm, "
                              f"outside {lo:.0f}-{hi:.0f}")
        if not lams["b"] < lams["g"] < lams["r"]:
            return None, "layer peaks are not in ascending wavelength order"
        return dict(tag=tag, profile=prof, file=fn, page=pgno,
                    log_s_r=out["r"], log_s_g=out["g"], log_s_b=out["b"],
                    peak_r=peaks["r"], peak_g=peaks["g"], peak_b=peaks["b"],
                    lam_r=lams["r"], lam_g=lams["g"], lam_b=lams["b"],
                    x_resid=cal_r[1], y_resid=cal_r[2],
                    n_x=cal_r[3], n_y=cal_r[4]), None
    return None, "no LOG SENSITIVITY panel yielded three inked layers"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--sheet", action="append", choices=sorted(SHEETS))
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--dump", action="store_true",
                    help="print the arrays in film_profiles.py form")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()
    tags = ns.sheet or sorted(SHEETS)
    bad = skipped = 0
    print(f"[i] corpus root {root}")
    for tag in tags:
        got, err = extract_sheet(root, tag)
        if got is None:
            if "not present" in (err or ""):
                print(f"  [SKIP] {tag}: {err}")
                skipped += 1
            else:
                print(f"  [FAIL] {tag}: {err}")
                bad += 1
            continue
        want_lams, want_peaks = EXPECTED[tag]
        lams = (got["lam_r"], got["lam_g"], got["lam_b"])
        peaks = (got["peak_r"], got["peak_g"], got["peak_b"])
        ok = (lams == want_lams
              and all(abs(a - b) < 0.03 for a, b in zip(peaks, want_peaks)))
        print(f"  [{'OK  ' if ok else 'FAIL'}] {tag} {got['profile']:24s} "
              f"peaks r{lams[0]:.0f} g{lams[1]:.0f} b{lams[2]:.0f} nm  "
              f"log S {peaks[0]:.2f}/{peaks[1]:.2f}/{peaks[2]:.2f}  "
              f"ticks {got['n_x']}x/{got['n_y']}y "
              f"resid {got['x_resid']:.2f}/{got['y_resid']:.2f} pt")
        if not ok:
            print(f"         expected peaks at {want_lams} nm, "
                  f"log S {want_peaks}")
            bad += 1
        if tag in EXPECTED_VS_STORED:
            # Cross-validate against the already-adopted set. Compared only on
            # samples BOTH sides call measured: a floor sentinel on either side
            # carries no information, and including it would manufacture a
            # 4-decade "disagreement" out of two different trace extents.
            from film_profiles import get_profile
            st = get_profile(got["profile"]).spectral
            got_rms, drift = [], False
            for key, stored in (("log_s_r", st.log_s_r), ("log_s_g", st.log_s_g),
                                ("log_s_b", st.log_s_b)):
                a = np.asarray(stored, dtype=float)
                b = got[key]
                m = (a > FLOOR + 0.01) & (b > FLOOR + 0.01)
                got_rms.append(float(np.sqrt(((a[m] - b[m]) ** 2).mean()))
                               if m.any() else float("nan"))
            want_rms = EXPECTED_VS_STORED[tag]
            drift = any(not (abs(a - b) < 0.01)
                        for a, b in zip(got_rms, want_rms))
            print(f"         vs the ADOPTED set: rms r {got_rms[0]:.3f} "
                  f"g {got_rms[1]:.3f} b {got_rms[2]:.3f} decades "
                  f"({'DRIFTED' if drift else 'as recorded'}) -- "
                  f"cross-check only, nothing re-adopted")
            if drift:
                print(f"         expected rms {want_rms}")
                bad += 1
        if ns.dump:
            for k in ("log_s_r", "log_s_g", "log_s_b"):
                print(f"            {k}=({', '.join('%.2f' % v for v in got[k])}),")
    print(f"\n[i] {len(tags) - bad - skipped} reproduced, {bad} failed, "
          f"{skipped} skipped")
    if ns.do_assert and bad:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
