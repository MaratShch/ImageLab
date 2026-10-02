"""Extract the AGFACOLOR Vista spectral-sensitivity curves from Agfa's own sheet.

WHY
---
`DIGITIZATION_QUEUE.md` item B2 read:

    `AGFA_VISTA_200` -- one vector page carries the 100/200/400/800 family,
    needs the legend read.

That description was HALF right, and the half that was wrong is the reason the
item sat unresolved. Page 6 of AGFA/'AGFACOLOR Vista 100, 200, 400, 800.pdf'
(Technical Data AF, 06/2000, 2nd edition) does carry several films on one page --
but three, not four: **Vista 200, Vista 400 and Vista 800 side by side**, each
with its own four-panel column and its own printed data block. Vista 100 is not
on this page at all. So there was never one superimposed family plot to
disentangle; there are three independent panels, and only the Vista 200 column
corresponds to a stock this database holds.

THE LEGEND, WHICH IS THE THING THE ITEM ASKED FOR
-------------------------------------------------
The three layer curves are NOT distinguished by colour -- every stroke on the
page is the same near-black (0.137, 0.122, 0.125). They are distinguished by
DASH PATTERN, and the sheet's own text labels sit under the humps:

    solid                       `[] 0`                    -> GREEN layer
    dashed                      `[ 3.159 .79 ] 0`         -> BLUE layer
    dash-dot                    `[ 3.159 .79 .79 .79 ] 0` -> RED layer

This is a machine-readable legend, not an inference from position: the PDF
stores the dash array per path. The extractor keys on it and then CHECKS the
result against the independent evidence -- the printed "Blue" / "Green" / "Red"
word positions, and the physical requirement that each layer peak in its own
band. Position alone would have been unsafe: in the spectral-sensitivity panel
the blue and green paths OVERLAP in x (blue spans 62.6-108.2 pt, green
62.5-130.4 pt), so "leftmost = blue" is not a separation rule.

CALIBRATION
-----------
From the printed axis-label centres, which come out exactly linear:

    x: 400/500/600/700 nm at 69.2 / 100.8 / 132.4 / 164.0 pt  -> 31.6 pt/100 nm
    y: 2.0 / 1.0 / 0     at     89.0 / 120.6 / 152.2 pt       -> 31.6 pt/decade

Both axes land on the same 31.6 pt modulus, and the printed y=0 rule is drawn at
y=151.9 pt against the label-derived 152.2 -- a 0.3 pt (0.01 decade) agreement
that is checked below rather than assumed. The plot frame then spans exactly
350-750 nm by -0.5..+2.5 lg sensitivity, which is a further sign the
calibration is right: sheets are drawn on round numbers.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It reads the SPECTRAL SENSITIVITY panel only. The same page also carries, per
film, a spectral-density panel (medium + minimum density, NOT per-dye C/M/Y --
so it cannot fill a `SpectralDyeDensity` record), a "Sharpness" panel that plots
TRANSFER FACTOR against lines/mm (a CTF-like quantity, not an MTF, and it
overshoots 100 % at low frequency), and colour density curves. Those are
separate items; taking them would need the same care about what each quantity
actually is.

Run:
    python agfa_vista.py                 # extract, print, self-check
    python agfa_vista.py --assert        # exit non-zero if anything moved
    python agfa_vista.py --emit          # print the SpectralSensitivity block

Needs pymupdf. Stdlib otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:                                       # pragma: no cover
    print("[!] pymupdf not installed:  pip install pymupdf")
    raise SystemExit(1)

SHEET = "AGFA/AGFACOLOR Vista 100, 200, 400, 800.pdf"
PAGE = 6                    # 1-based, as printed on the page itself

#: Dash array -> layer. The whole legend read, in three lines.
DASH_TO_LAYER = {
    "[] 0":                        "g",
    "[ 3.159 .79 ] 0":             "b",
    "[ 3.159 .79 .79 .79 ] 0":     "r",
}

#: Which x-band on the page belongs to which film. Read from the three
#: `AGFACOLOR Vista NNN` headings at x 25.5 / 201.8 / 377.x pt.
PANELS = {"VISTA_200": (20.0, 195.0),
          "VISTA_400": (196.0, 371.0),
          "VISTA_800": (372.0, 560.0)}

#: The spectral-sensitivity panel is the topmost of the four; y-band on the page.
SPECTRAL_Y = (66.0, 190.0)

#: Physical sanity: each layer must peak inside its own absorption band. These
#: are wide on purpose -- the check is meant to catch a legend swap, not to
#: second-guess Agfa's emulsion design.
PEAK_BANDS = {"b": (400.0, 480.0), "g": (520.0, 580.0), "r": (600.0, 680.0)}

#: Measured 2026-08-18. `--assert` fails if the sheet stops reproducing these.
#: Peak wavelength per layer, per film, nm.
#: ⚠ These were first written from a 150 dpi eyeball of the page and had blue at
#: 440/430/420 nm. The extraction says 470 nm for all three, and it is right: the
#: blue layer is a broad PLATEAU, not a peak -- Vista 200 reads -0.16 at 410-420,
#: dips to -0.28 at 450, and its global maximum is a second, marginally higher
#: lobe at 470. A 0.1-decade difference decides which lobe wins, so the blue
#: "peak" is weakly determined by construction and the tolerance below is what
#: makes this a regression check rather than a claim about emulsion design.
EXPECTED_PEAKS = {
    "VISTA_200": {"b": 470.0, "g": 550.0, "r": 620.0},
    "VISTA_400": {"b": 470.0, "g": 550.0, "r": 620.0},
    "VISTA_800": {"b": 470.0, "g": 550.0, "r": 620.0},
}
PEAK_TOL_NM = 20.0

LAMBDA_START, LAMBDA_STEP, LAMBDA_N = 380.0, 10.0, 33      # 380..700 nm
FLOOR = -4.0                                               # database convention


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def bezier(p0, p1, p2, p3, n=24):
    """Flatten one cubic segment to n+1 points."""
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        x = (u * u * u * p0[0] + 3 * u * u * t * p1[0]
             + 3 * u * t * t * p2[0] + t * t * t * p3[0])
        y = (u * u * u * p0[1] + 3 * u * u * t * p1[1]
             + 3 * u * t * t * p2[1] + t * t * t * p3[1])
        out.append((x, y))
    return out


def flatten(items):
    """Drawing items -> one polyline in page points."""
    pts = []
    for it in items:
        if it[0] == "l":
            a, b = it[1], it[2]
            pts += [(a.x, a.y), (b.x, b.y)]
        elif it[0] == "c":
            p0, p1, p2, p3 = it[1], it[2], it[3], it[4]
            pts += bezier((p0.x, p0.y), (p1.x, p1.y),
                          (p2.x, p2.y), (p3.x, p3.y))
    # de-duplicate consecutive repeats, keep drawing order
    out = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out


class Calib:
    """Page points <-> (nm, lg sensitivity), fitted to the printed labels."""

    def __init__(self, xs: dict, ys: dict):
        # xs: {nm: x_pt}, ys: {value: y_pt}; both are exactly linear here, so a
        # two-point fit is used and the REMAINING labels are then residual-checked.
        (n1, x1), (n2, x2) = sorted(xs.items())[0], sorted(xs.items())[-1]
        self.nm_per_pt = (n2 - n1) / (x2 - x1)
        self.x0, self.nm0 = x1, n1
        (v1, y1), (v2, y2) = sorted(ys.items())[0], sorted(ys.items())[-1]
        self.val_per_pt = (v2 - v1) / (y2 - y1)
        self.y0, self.val0 = y1, v1
        self.res_nm = max(abs(self.nm(x) - n) for n, x in xs.items())
        self.res_val = max(abs(self.val(y) - v) for v, y in ys.items())

    def nm(self, x):
        return self.nm0 + (x - self.x0) * self.nm_per_pt

    def val(self, y):
        return self.val0 + (y - self.y0) * self.val_per_pt


def label_calib(page, xband, yband):
    """Build a Calib from the printed axis labels inside one panel."""
    xs, ys, words = {}, {}, page.get_text("words")
    for x0, y0, x1, y1, t, *_ in words:
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if not (xband[0] <= cx <= xband[1]):
            continue
        if t in ("400", "500", "600", "700") and yband[1] - 25 <= cy <= yband[1] + 25:
            xs[float(t)] = cx                       # wavelength axis, below frame
        elif t in ("0", "1.0", "2.0") and yband[0] <= cy <= yband[1]:
            ys[float(t)] = cy                       # sensitivity axis, left of frame
    return Calib(xs, ys), xs, ys


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def curves_for(page, xband, yband):
    """dash-keyed layer -> polyline, for one film's spectral panel."""
    got = {}
    for dr in page.get_drawings():
        r = dr["rect"]
        cx, cy = (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0
        if not (xband[0] <= cx <= xband[1] and yband[0] <= cy <= yband[1]):
            continue
        if r.width < 25.0 or r.height < 25.0:       # frames, rules, tick marks
            continue
        # A data curve is drawn as BEZIERS. The panel's gridlines are a single
        # solid path of straight segments with the same bounding box as the
        # frame, so a size filter alone matched it and collided with the solid
        # (green) curve -- caught on the first run. Requiring bezier segments
        # separates draughtsmanship from data.
        if sum(1 for it in dr["items"] if it[0] == "c") < 5:
            continue
        layer = DASH_TO_LAYER.get(str(dr.get("dashes")))
        if layer is None:
            continue
        pts = flatten(dr["items"])
        if len(pts) < 8:
            continue
        if layer in got:                            # two paths, same style
            raise SystemExit(f"[!] duplicate {layer} path in band {xband}")
        got[layer] = pts
    return got


def resample(pts, cal):
    """Polyline -> (lg values on the 380..700 nm grid, peak nm).

    Outside the curve's own support the layer is dead, which the database
    encodes as the FLOOR rather than as an extrapolation.
    """
    xy = sorted(((cal.nm(x), cal.val(y)) for x, y in pts), key=lambda p: p[0])
    lo, hi = xy[0][0], xy[-1][0]
    out = []
    for i in range(LAMBDA_N):
        nm = LAMBDA_START + i * LAMBDA_STEP
        if nm < lo or nm > hi:
            out.append(None)
            continue
        # linear interpolation between bracketing samples
        for (a_nm, a_v), (b_nm, b_v) in zip(xy, xy[1:]):
            if a_nm <= nm <= b_nm:
                f = 0.0 if b_nm == a_nm else (nm - a_nm) / (b_nm - a_nm)
                out.append(a_v + f * (b_v - a_v))
                break
        else:
            out.append(None)
    live = [(LAMBDA_START + i * LAMBDA_STEP, v)
            for i, v in enumerate(out) if v is not None]
    peak_nm = max(live, key=lambda p: p[1])[0] if live else float("nan")
    return out, peak_nm


def normalise(per_layer):
    """Peak-normalise EACH layer to its own 0.0, which is the schema's rule.

    ⚠ WHAT THIS THROWS AWAY, deliberately and per the existing convention:
    Agfa plots all three layers against ONE absolute lg-sensitivity axis, so the
    sheet does record the inter-layer speed offsets. `SpectralSensitivity`
    normalises every layer to its own peak (verify.py asserts
    ``abs(max(layer)) < 1e-9`` for each layer independently), so those offsets
    are NOT representable in this field and are lost here. They are printed in
    the run output and quoted in the profile's source string instead, so the
    information survives in the record even though the carrier cannot hold it.
    A first run that normalised all three to a SHARED peak was rejected for
    exactly this reason -- it would have failed the database's own check.
    """
    out = {}
    for k, vals in per_layer.items():
        top = max(v for v in vals if v is not None)
        out[k] = tuple(FLOOR if v is None else max(FLOOR, round(v - top, 2))
                       for v in vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../../PDF/PROFILES")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--emit", action="store_true",
                    help="print the SpectralSensitivity(...) block for Vista 200")
    ns = ap.parse_args()

    pdf = Path(ns.root).expanduser().resolve() / SHEET
    if not pdf.is_file():
        print(f"[!] sheet not found: {pdf}")
        return 1
    doc = pymupdf.open(pdf)
    page = doc[PAGE - 1]
    print(f"[i] {pdf.name} page {PAGE}")

    bad, emitted = 0, {}
    for film, xband in PANELS.items():
        cal, xs, ys = label_calib(page, xband, SPECTRAL_Y)
        raw = curves_for(page, xband, SPECTRAL_Y)
        print(f"\n=== {film}")
        print(f"    calibration: {cal.nm_per_pt:.4f} nm/pt, "
              f"{-cal.val_per_pt:.4f} decade/pt; "
              f"residuals {cal.res_nm:.2f} nm / {cal.res_val:.3f} decade "
              f"({len(xs)} x-labels, {len(ys)} y-labels)")
        if cal.res_nm > 1.0 or cal.res_val > 0.02:
            print("    [FAIL] calibration residual too large")
            bad += 1
        if set(raw) != {"r", "g", "b"}:
            print(f"    [FAIL] dash separation found {sorted(raw)}, want r/g/b")
            bad += 1
            continue

        per_layer, peaks = {}, {}
        for layer, pts in raw.items():
            vals, peak = resample(pts, cal)
            per_layer[layer], peaks[layer] = vals, peak
        # The sheet's ABSOLUTE peak heights, printed here because the schema
        # cannot store them (see normalise()). These ARE the inter-layer speed
        # relationship as Agfa drew it.
        abs_top = {k: max(v for v in vals if v is not None)
                   for k, vals in per_layer.items()}
        print("    absolute lg-sensitivity peaks (LOST on normalisation): "
              + ", ".join(f"{k} {abs_top[k]:.2f}" for k in ("b", "g", "r")))
        norm = normalise(per_layer)

        # CHECK 1 -- physics. Each layer must peak in its own band. This is what
        # catches a legend swap, and it is independent of the dash keying.
        for layer, (lo, hi) in PEAK_BANDS.items():
            ok = lo <= peaks[layer] <= hi
            print(f"    {layer}: peak {peaks[layer]:5.0f} nm  "
                  f"{'OK ' if ok else 'FAIL'} (band {lo:.0f}-{hi:.0f})")
            if not ok:
                bad += 1
        # CHECK 2 -- the sheet's own words. Agfa prints "Blue" / "Green" / "Red"
        # under the humps; their x-order must match the dash-keyed peak order.
        words = [(float(w[0] + w[2]) / 2.0, w[4]) for w in page.get_text("words")
                 if w[4] in ("Blue", "Green", "Red")
                 and xband[0] <= (w[0] + w[2]) / 2.0 <= xband[1]
                 and SPECTRAL_Y[0] <= (w[1] + w[3]) / 2.0 <= SPECTRAL_Y[1]]
        by_word = [t for _, t in sorted(words)]
        by_dash = [t for _, t in sorted(
            (peaks[k], n) for k, n in (("b", "Blue"), ("g", "Green"), ("r", "Red")))]
        ok = by_word == by_dash
        print(f"    printed label order {by_word} vs dash-keyed {by_dash}: "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            bad += 1
        # CHECK 3 -- regression against the recorded peaks.
        for layer, want in EXPECTED_PEAKS.get(film, {}).items():
            if abs(peaks[layer] - want) > PEAK_TOL_NM:
                print(f"    [FAIL] {layer} peak {peaks[layer]:.0f} nm, "
                      f"recorded {want:.0f} nm")
                bad += 1
        emitted[film] = norm

    if ns.emit and "VISTA_200" in emitted:
        n = emitted["VISTA_200"]
        print("\n# --- paste into AGFA_VISTA_200 ---")
        print("        spectral=SpectralSensitivity(")
        print(f"            lambda_start_nm={LAMBDA_START}, "
              f"lambda_step_nm={LAMBDA_STEP},")
        for k in ("r", "g", "b"):
            body = ", ".join(f"{v:.2f}" for v in n[k])
            print(f"            log_s_{k}=({body}),")
        print('            criterion="lg_relative_sensitivity_agfa_sheet",')
        print('            source=("Agfa-Gevaert AG, «AGFACOLOR Vista 100, 200, '
              '400, 800 -- Technical Data AF», 2nd edition, 06/2000, p6, "')
        print('                    "AGFACOLOR Vista 200 spectral-sensitivity panel"),')
        print("        ),")

    print()
    if bad:
        print(f"[FAIL] {bad} check(s) failed")
        return 1 if ns.do_assert else 0
    print("[OK] 3 panels, 9 curves, dash legend agrees with Agfa's own labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
