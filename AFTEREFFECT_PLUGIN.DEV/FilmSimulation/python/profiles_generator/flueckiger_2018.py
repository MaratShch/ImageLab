"""Flueckiger et al. 2018, «Investigation of Film Material–Scanner Interaction».

WHAT THIS SOURCE IS
-------------------
`RETRO/flueckigeretal_investigationfilmmaterialscannerinteraction_2018_v_1-1b.pdf`
-- Barbara Flueckiger, David Pfluger, Giorgio Trumpy, Simone Croci, Tunç Aydın
and Aljoscha Smolic, University of Zurich Department of Film Studies, version
1.1, 18 February 2018, 88 pages. The final report of the DIASTOR project.

⚠ IT IS NOT A RETRO DOCUMENT AND IT IS NOT A DATASHEET. It is a scanner study,
and most of it -- eight scanner walkthroughs, an eleven-expert subjective rating
study, spider charts -- has no bearing on a film-simulation database. Four
things in it do, and they are the reason it is registered:

  §2.8.2 Fig. 16   the analytical densities of the THREE-STRIP TECHNICOLOR dyes,
                   measured on a real 1949 print and extracted by the Ohta PCA
                   method. TECHNICOLOR_THREE_STRIP is one of the corpus's nine
                   stocks with NO source of any kind.
  §2.8.3 Fig. 21   the transmittance of the three DUFAYCOLOR réseau filter
                   elements, 400-700 nm at 20 nm, measured through 16 bandpass
                   interference filters on a Leitz DIALUX microscope.
  §2.8.3 Fig. 22   the integral transmittance of the same sample, measured on a
                   bench spectrophotometer over 300-900 nm -- which lets Fig. 21
                   be checked against an instrument of a different kind.
  §4.1   Fig. 61   the MEASURED MTF of eight film scanners, plus Table 3's
         Table 3   sampling resolutions. Scanner data, not film data: it is
                   documented in `doc/SCANNER_CHARACTERISTICS.md` and is
                   deliberately kept OUT of the film database.

THE CHECK THAT MAKES FIGURES 21 AND 22 TRUSTWORTHY
----------------------------------------------------
The report states equation (7):

    integral_T% = 0.28 blue_T% + 0.32 green_T% + 0.40 red_T%

with 28 / 32 / 40 % the measured area fractions of the réseau's three elements.
Figure 21 gives the three single-dye transmittances and Figure 22 gives the
integral. ⚠ THE TWO FIGURES ARE PLOTTED ON DIFFERENT AXES, IN DIFFERENT UNITS
OF WAVELENGTH RANGE, AND THIS READER TRACES THEM SEPARATELY -- so recomputing
equation (7) from the Figure 21 trace and comparing it with the Figure 22
markers tests both calibrations, both marker extractions and the printed area
fractions at once, with no free parameter. It closes to **rms 0.28 transmittance
points, worst 0.65**, over the fourteen wavelengths where all three markers are
unoccluded.

⚠ AND IT RECOVERS THE TWO THAT ARE OCCLUDED. Red at 560 nm and green at 640 nm
are hidden behind other markers in Figure 21. Inverting equation (7) against
Figure 22 returns them. The method is validated on green at 640, where a
partially visible blob reads 9.7 % and the inversion returns 9.60 %.

⚠ FIGURE 21'S CAPTION IS WRONG AND THE FIGURE IS RIGHT. The caption says
"Resulting **absorbance** curves"; the figure's own ordinate is labelled
"TRANSMITTANCE %" and its three series are named transRED / transGREEN /
transBlue. This reader stores transmittance and says so. A reader who trusted
the caption would have stored 1 - T as if it were A.

WHAT FIGURE 16 CAN AND CANNOT GIVE
------------------------------------
⚠ ITS ORDINATE HAS NO SCALE, NO TICKS AND NO LABEL. The abscissa is gridded at
350-800 nm in 50 nm steps and calibrates cleanly; the ordinate is a bare axis.
So the SHAPE is measurable and the LEVEL is not, and the stored curves are
normalised to unit peak with the axis taken as zero -- an assumption, recorded
as one. What validates the trace is the report's own printed peak list:

    yellow ≈ 460 nm;  magenta ≈ 540 nm (main) and 575 (secondary);
    cyan ≈ 660 nm (main) and 720 (secondary)

and the trace returns 460 / 540 / 660 / 720 exactly. The magenta 575 secondary
is a shoulder rather than a local maximum at this raster resolution and is not
claimed.

FIGURE 15 IS NOT NEW EVIDENCE AND THIS FILE SAYS SO
-----------------------------------------------------
⚠ Figure 15 is the SAME Callier Q artwork as Trumpy & Gschwind 2015 Fig. 5,
already digitised by `trumpy_callier_q.py` -- Trumpy is a co-author of both, and
both trace back to Streiffert 1947. Re-digitising it here would be
double-counting one measurement, so it is used only as a REPRODUCIBILITY CHECK
on this project's raster pipeline: two different PDFs, two different scans of one
drawing, traced by the same method, agree to rms 0.0053 Q.

Run:  python flueckiger_2018.py --root <corpus> [--assert]
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
    from scipy import ndimage as ndi
except ImportError:                                       # pragma: no cover
    print("[!] Pillow + scipy required")
    raise SystemExit(1)

SHEET = ("RETRO/flueckigeretal_investigationfilmmaterialscanner"
         "interaction_2018_v_1-1b.pdf")

SOURCE = ("B. Flueckiger, D. Pfluger, G. Trumpy, S. Croci, T. Aydın and "
          "A. Smolic, «Investigation of Film Material–Scanner Interaction», "
          "University of Zurich, DIASTOR, v1.1, 18 Feb 2018 -- "
          "PDF/PROFILES/" + SHEET)

#: Figure 16 is stored on page index 21 as five horizontal JPEG strips.
FIG16_STRIPS = (96, 97, 98, 99, 100)
FIG21_XREF, FIG22_XREF = 118, 119
FIG15_XREF = 89
FIG61_STRIPS = (293, 294)

#: The réseau area fractions the report prints for equation (7).
AREA_BLUE, AREA_GREEN, AREA_RED = 0.28, 0.32, 0.40

#: The peak wavelengths §2.8.2 prints in the running text. The trace must
#: reproduce them; they are never used to calibrate anything.
TECHNICOLOR_PEAKS = {"yellow": (460,), "magenta": (540, 575), "cyan": (660, 720)}

#: Table 3, printed p59: spatial resolution in pixels/mm. The Nyquist column is
#: exactly half of it and is not stored twice.
TABLE3_PXL_PER_MM = {
    "Altra mk3": 105, "ARRISCAN": 170, "D-Archiver Cine10-A": 67,
    "The Director": 152, "Kinetta": 172, "Northlight 1": 170,
    "Scanity (Digimage)": 186, "Scanity (Sound & Vision)": 85,
}

#: Figure 61's abscissa: the spatial frequencies present in the ARRI AQUA target.
MTF_FREQ_LPMM = (0, 10, 12, 14, 16, 20, 28, 40)


# ---------------------------------------------------------------------------
def _img(doc, xref):
    info = doc.extract_image(xref)
    return np.array(Image.open(io.BytesIO(info["image"])).convert("RGB")).astype(int)


def _stitch(doc, xrefs):
    ims = [_img(doc, x) for x in xrefs]
    w = max(i.shape[1] for i in ims)
    return np.vstack([i for i in ims if i.shape[1] == w])


def _runs(vals, gap=2):
    out = []
    for i in vals:
        if out and i - out[-1][-1] <= gap:
            out[-1].append(i)
        else:
            out.append([i])
    return [sum(g) / len(g) for g in out]


def _grid(a, axis, frac, lo=0, thresh=225):
    """Positions of the light-grey chart gridlines along one axis."""
    R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    grey = (abs(R - G) < 25) & (abs(G - B) < 25) & (R < thresh)
    n = a.shape[1] if axis == "col" else a.shape[0]
    m = a.shape[0] - lo if axis == "col" else a.shape[1]
    if axis == "col":
        return _runs([c for c in range(n) if grey[lo:, c].sum() > frac * m])
    return _runs([r for r in range(lo, n) if grey[r, :].sum() > frac * m])


# ---------------------------------------------------------------------------
#  Figure 21 -- Dufaycolor réseau, three elements, 16 wavelengths
# ---------------------------------------------------------------------------
def fig21(doc):
    a = _img(doc, FIG21_XREF)
    if a.shape[:2] != (381, 586):
        return None, f"Fig. 21 raster is {a.shape[1]}x{a.shape[0]}, expected 586x381"
    R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    rows = _grid(a, "row", 0.35)
    if len(rows) != 11:
        return None, f"Fig. 21: {len(rows)} ordinate gridlines, expected 11"
    my, by = np.polyfit(rows, list(range(100, -1, -10)), 1)
    cols = _grid(a, "col", 0.5)
    if len(cols) < 4:
        return None, f"Fig. 21: {len(cols)} abscissa gridlines, expected >= 4"
    # the visible verticals are 400 / 500 / 550 / 600 / 700 -- the 450 and 650
    # lines are covered by the curves. Their spacing identifies them.
    known = {53: 400, 194: 500, 266: 550, 336: 600, 479: 700}
    pairs = [(c, known[min(known, key=lambda k: abs(k - c))]) for c in cols
             if min(abs(k - c) for k in known) < 6]
    if len(pairs) < 4:
        return None, "Fig. 21: could not identify the abscissa gridlines"
    mx, bx = np.polyfit([p[0] for p in pairs], [p[1] for p in pairs], 1)

    masks = {"red":   (R > 120) & (R - G > 60) & (R - B > 60),
             "green": (G > 90) & (G - R > 40) & (G - B > 30),
             "blue":  (B > 110) & (B - R > 50) & (B - G > 40)}
    wl = list(range(400, 701, 20))
    out = {}
    for k, m in masks.items():
        lab, n = ndi.label(m)
        blobs = []
        for i in range(1, n + 1):
            sz = int((lab == i).sum())
            if sz < 40:
                continue
            cy, cx = ndi.center_of_mass(lab == i)
            if cx > 500:              # the legend
                continue
            blobs.append((mx * cx + bx, my * cy + by, sz))
        out[k] = {w: max((b for b in blobs if abs(b[0] - w) <= 7),
                         key=lambda b: b[2], default=(0, np.nan, 0))[1]
                  for w in wl}
    return {"wl": wl, "T": out, "cal": (mx, bx, my, by)}, None


# ---------------------------------------------------------------------------
#  Figure 22 -- integral transmittance, calculated markers + measured curve
# ---------------------------------------------------------------------------
def fig22(doc):
    a = _img(doc, FIG22_XREF)
    if a.shape[:2] != (332, 584):
        return None, f"Fig. 22 raster is {a.shape[1]}x{a.shape[0]}, expected 584x332"
    R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    rows = _grid(a, "row", 0.5)
    if len(rows) != 10:
        return None, f"Fig. 22: {len(rows)} ordinate gridlines, expected 10"
    my, by = np.polyfit(rows, list(range(90, -1, -10)), 1)
    X = lambda c: 300.0 + (c - 45.0) / 87.0 * 100.0
    Y = lambda r: my * r + by

    blue = (B > 140) & (B - R > 35) & (B - G > 15)
    lab, n = ndi.label(blue)
    mk = []
    for i in range(1, n + 1):
        sz = int((lab == i).sum())
        if sz < 25:
            continue
        cy, cx = ndi.center_of_mass(lab == i)
        if cy < 60:                    # the legend
            continue
        mk.append((X(cx), Y(cy)))
    mk.sort()

    red = (R > 110) & (R - G > 50) & (R - B > 50)
    pts = []
    for c in range(46, 568):
        idx = np.flatnonzero(red[:, c])
        idx = idx[(idx > 60) & (idx < 290)]
        if idx.size == 0 or idx.size > 14:
            continue
        pts.append((X(c), Y(idx.mean())))
    return {"calc": np.array(mk), "meas": np.array(pts)}, None


# ---------------------------------------------------------------------------
#  Figure 16 -- Technicolor eigenspectra
# ---------------------------------------------------------------------------
def fig16(doc):
    a = _stitch(doc, FIG16_STRIPS)
    if a.shape[1] != 887:
        return None, f"Fig. 16 stitched width {a.shape[1]}, expected 887"
    sub = a[:, 430:]
    R, G, B = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    # ⚠ Figure 16's gridlines are printed lighter than Figure 21's, so the
    # default grey threshold misses all ten of them and finds one.
    cols = _grid(sub, "col", 0.5, thresh=240)
    if len(cols) != 10:
        return None, f"Fig. 16: {len(cols)} abscissa gridlines, expected 10 (350-800 nm)"
    mx, bx = np.polyfit(cols, list(range(350, 801, 50)), 1)
    BOT = 278
    masks = {"yellow":  (R > 140) & (G > 110) & (B < 130) & (R - B > 50) & (abs(R - G) < 80),
             "magenta": (R > 140) & (B > 110) & (G < 120) & (R - G > 50) & (B - G > 30),
             "cyan":    (G > 100) & (B > 130) & (R < 140) & (B - R > 40) & (G - R > 20)}
    out = {}
    for k, m in masks.items():
        pts = []
        for c in range(int(cols[0]), int(cols[-1]) + 1):
            idx = np.flatnonzero(m[:BOT, c])
            idx = idx[idx > 10]
            if idx.size == 0 or idx.size > 16:
                continue
            pts.append((mx * c + bx, BOT - idx.mean()))
        out[k] = np.array(pts)
    return out, None


# ---------------------------------------------------------------------------
#  Figure 61 -- scanner MTF
# ---------------------------------------------------------------------------
FIG61_HUES = {
    "Altra mk3": (178, 205, 0.30), "D-Archiver Cine10-A": (350, 12, 0.45),
    "The Director": (262, 288, 0.28), "Kinetta": (296, 340, 0.20),
    "Northlight 1": (215, 248, 0.40), "Scanity (Digimage)": (18, 52, 0.30),
    "Scanity (Sound & Vision)": (95, 155, 0.30),
}


def fig61(doc):
    a = _stitch(doc, FIG61_STRIPS)
    im = Image.fromarray(a.astype("uint8"), "RGB").convert("HSV")
    hsv = np.array(im).astype(float)
    H, S, V = hsv[:, :, 0] * 360 / 255.0, hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
    cols = _grid(a, "col", 0.55, lo=90)
    rows = _grid(a, "row", 0.55, lo=90)
    if len(cols) != 9 or len(rows) != 6:
        return None, (f"Fig. 61: grid is {len(cols)}x{len(rows)}, expected 9x6")
    X0, PX = cols[0], (cols[-1] - cols[0]) / 40.0
    Y1, Y0 = rows[0], rows[-1]
    fval = lambda r: (Y0 - r) / (Y0 - Y1)

    def read(mask, f, widths=(3, 5, 8, 12)):
        c = int(round(X0 + PX * f))
        for w in widths:
            wnd = mask[:, max(0, c - w):c + w + 1]
            rr = np.flatnonzero(wnd.any(1))
            rr = rr[(rr > 90) & (rr < Y0 + 6)]
            if rr.size:
                cnt = wnd.sum(1)
                best = rr[np.argmax(cnt[rr])]
                return round(float(fval(rr[abs(rr - best) <= 4].mean())), 3)
        return None

    out = {}
    for name, (lo, hi, smin) in FIG61_HUES.items():
        m = ((H >= lo) & (H <= hi)) if lo < hi else ((H >= lo) | (H <= hi))
        m = m & (S >= smin) & (V > 0.25)
        m[:92, :] = False
        out[name] = {f: (1.0 if f == 0 else read(m, f)) for f in MTF_FREQ_LPMM}
    # ARRISCAN is drawn in neutral grey; the gridlines are lighter, so a value
    # threshold separates them. ⚠ Without it the ξ = 20 sample returns exactly
    # 1.000 -- the top gridline, read through the vertical gridline at 20.
    m = (S < 0.30) & (V < 0.50)
    m[:92, :] = False
    out["ARRISCAN"] = {f: (1.0 if f == 0 else read(m, f)) for f in MTF_FREQ_LPMM}
    return out, None


# ---------------------------------------------------------------------------
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
    ok = (doc.page_count == 88
          and "INVESTIGATION OF FILM MATERIAL" in txt.upper()
          and "TECHNICOLOR - eigenspectra" not in txt      # it is in the raster
          and "Ohta" in txt)
    print(f"  [{'OK  ' if ok else 'FAIL'}] 88 pages, DIASTOR scanner report, "
          f"cites the Ohta PCA method")
    if not ok:
        return 1

    # ---- Figure 21 --------------------------------------------------------
    f21, err = fig21(doc)
    if f21 is None:
        print(f"  [FAIL] {err}")
        return 1 if ns.do_assert else 0
    mx, bx, my, by = f21["cal"]
    nfound = sum(1 for k in f21["T"] for w in f21["wl"]
                 if not np.isnan(f21["T"][k][w]))
    print(f"  [OK  ] Fig. 21 gridline-calibrated (ordinate {my*336+by:.2f}"
          f"..{my*33+by:.2f} % against a printed 0..100); "
          f"{nfound}/48 markers found")
    if nfound < 46:
        print(f"  [FAIL] only {nfound} of 48 Dufaycolor markers")
        bad += 1

    # ---- Figure 22 --------------------------------------------------------
    f22, err = fig22(doc)
    if f22 is None:
        print(f"  [FAIL] {err}")
        return 1 if ns.do_assert else 0
    print(f"  [OK  ] Fig. 22: {len(f22['calc'])} calculated markers, "
          f"{len(f22['meas'])} measured-curve points "
          f"({f22['meas'][0,0]:.0f}-{f22['meas'][-1,0]:.0f} nm)")
    if len(f22["calc"]) != 16:
        print(f"  [FAIL] expected 16 calculated markers")
        bad += 1

    # ---- the equation (7) cross-check ------------------------------------
    T = f21["T"]
    integ = lambda w: float(np.interp(w, f22["calc"][:, 0], f22["calc"][:, 1]))
    errs, used = [], []
    for w in f21["wl"]:
        r, g, b = T["red"][w], T["green"][w], T["blue"][w]
        if any(np.isnan(v) for v in (r, g, b)):
            continue
        errs.append(AREA_RED * r + AREA_GREEN * g + AREA_BLUE * b - integ(w))
        used.append(w)
    e = np.array(errs)
    rms = float(np.sqrt((e ** 2).mean()))
    print(f"\n  -- ⚠ THE CHECK THIS READER RESTS ON: equation (7), "
          f"{AREA_BLUE} B + {AREA_GREEN} G + {AREA_RED} R, recomputed from the "
          f"Fig. 21 trace against the Fig. 22 markers")
    print(f"  [{'OK  ' if rms < 0.8 else 'FAIL'}] {len(e)} wavelengths: "
          f"rms {rms:.3f} transmittance points, worst {e[np.argmax(abs(e))]:+.3f}"
          f" -- two separately calibrated figures, no free parameter")
    if rms >= 0.8:
        bad += 1

    # recover the occluded pair by inverting the same equation
    rec = {}
    for w in f21["wl"]:
        r, g, b = T["red"][w], T["green"][w], T["blue"][w]
        if np.isnan(r) and not (np.isnan(g) or np.isnan(b)):
            rec[("red", w)] = (integ(w) - AREA_BLUE * b - AREA_GREEN * g) / AREA_RED
        if np.isnan(g) and not (np.isnan(r) or np.isnan(b)):
            rec[("green", w)] = (integ(w) - AREA_BLUE * b - AREA_RED * r) / AREA_GREEN
    for (k, w), v in sorted(rec.items()):
        print(f"        occluded {k} marker at {w} nm recovered by inverting "
              f"eq (7): {v:.2f} %")

    # ---- Figure 16 --------------------------------------------------------
    f16, err = fig16(doc)
    if f16 is None:
        print(f"  [FAIL] {err}")
        return 1 if ns.do_assert else 0
    print(f"\n  -- Fig. 16, TECHNICOLOR eigenspectra "
          f"(⚠ the ordinate carries no scale, no ticks and no label)")
    grid = np.arange(360, 791, 10.0)
    norm = {}
    for k in ("yellow", "magenta", "cyan"):
        P = f16[k]
        v = np.interp(grid, P[:, 0], P[:, 1])
        norm[k] = v / v.max()
        peak = grid[int(np.argmax(norm[k]))]
        want = TECHNICOLOR_PEAKS[k][0]
        good = abs(peak - want) <= 5
        print(f"  [{'OK  ' if good else 'FAIL'}] {k:8s} {len(P)} traced points, "
              f"main peak {peak:.0f} nm against the printed {want} nm")
        if not good:
            bad += 1
    # the cyan secondary is a genuine local maximum and must be found
    c = norm["cyan"]
    sec = [grid[i] for i in range(3, len(c) - 3)
           if c[i] == max(c[i - 3:i + 4]) and grid[i] > 690]
    good = any(abs(s - 720) <= 10 for s in sec)
    print(f"  [{'OK  ' if good else 'FAIL'}] cyan SECONDARY peak at "
          f"{sec[0] if sec else float('nan'):.0f} nm against the printed 720 nm"
          f" -- the magenta 575 secondary is a shoulder at this raster "
          f"resolution and is deliberately not claimed")
    if not good:
        bad += 1

    # ---- Figure 61 + Table 3 ---------------------------------------------
    f61, err = fig61(doc)
    if f61 is None:
        print(f"  [FAIL] {err}")
        return 1 if ns.do_assert else 0
    print(f"\n  -- Fig. 61, SCANNER MTF (fraction of reproduced contrast). "
          f"⚠ SCANNER DATA, NOT FILM DATA -- documented in "
          f"doc/SCANNER_CHARACTERISTICS.md, stored in no profile")
    print("     scanner                    " +
          "  ".join(f"{f:>5}" for f in MTF_FREQ_LPMM) + "   px/mm  Nyquist")
    for name in sorted(f61):
        row = f61[name]
        got = sum(1 for f in MTF_FREQ_LPMM if row.get(f) is not None)
        px = TABLE3_PXL_PER_MM.get(name)
        print(f"     {name:26s}" +
              "  ".join(("  -  " if row.get(f) is None else f"{row[f]:.3f}")
                        for f in MTF_FREQ_LPMM) +
              (f"   {px:4d}   {px/2:5.1f}" if px else "      -       -"))
        if got < 6:
            print(f"  [FAIL] only {got} of 8 points read for {name}")
            bad += 1
    missing = set(TABLE3_PXL_PER_MM) - set(f61)
    if missing:
        print(f"  [FAIL] Table 3 names scanners absent from Fig. 61: {missing}")
        bad += 1
    else:
        print(f"  [OK  ] all 8 scanners in Table 3 are traced in Fig. 61")

    # ⚠ TABLE 3 IS CHECKED, NOT TRANSCRIBED ON TRUST. Figure 62 is Figure 61
    # with the abscissa divided by these same pixel counts (the report's
    # equation 9), so each series' last point lands at max(ξ)/pxl_per_mm.
    print(f"  -- Table 3 against Fig. 62's endpoints (eq. 9, "
          f"lp/pixel = lp/mm ÷ pixel/mm)")
    for name, px in sorted(TABLE3_PXL_PER_MM.items()):
        fmax = max(f for f in MTF_FREQ_LPMM if f61[name].get(f) is not None)
        print(f"     {name:26s} last ξ {fmax:2d} lp/mm ÷ {px:3d} px/mm = "
              f"{fmax/px:.3f} lp/pixel")

    # ---- Figure 15, reproducibility only ---------------------------------
    try:
        import film_profiles as fp
        ref = np.array(fp._CALLIER_Q_REFERENCE)
    except Exception:                                     # pragma: no cover
        ref = None
    if ref is not None:
        a15 = np.array(Image.open(io.BytesIO(
            doc.extract_image(FIG15_XREF)["image"])).convert("L")).astype(float)
        MX = 2.0 / (450.5 - 48.0)
        X = lambda c: MX * c - 48.0 * MX
        Y = lambda r: 1.0 + (299.5 - r) * 0.2 / 82.333
        d = a15 < 150

        def _one(idx):
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

        pts = []
        for c in range(50, 450):          # columns where the stroke is thin
            s = _one(np.flatnonzero(d[18:295, c]) + 18)
            if s is None or s.size > 8 or s[0] <= 18 or s[-1] >= 294:
                continue
            w = np.clip(200.0 - a15[s, c], 1, None)
            pts.append((X(c), Y(float((w * s).sum() / w.sum()))))
        # ⚠ ROWS ON THE TOE, for the same reason as trumpy_callier_q.py: below
        # D ~ 0.15 the curve is near-vertical and a column scan returns the
        # middle of the run rather than a function value.
        for r in range(95, 294):
            s = _one(np.flatnonzero(d[r, 50:130]) + 50)
            if s is None or s.size > 8 or s[0] <= 50 or s[-1] >= 129:
                continue
            w = np.clip(200.0 - a15[r, s], 1, None)
            pts.append((X(float((w * s).sum() / w.sum())), Y(r)))
        P = np.array(sorted(pts))
        diffs = [float(np.interp(dd, P[:, 0], P[:, 1])) - q
                 for dd, q in ref if P[0, 0] <= dd <= P[-1, 0]]
        r = float(np.sqrt(np.mean(np.square(diffs))))
        print(f"\n  [{'OK  ' if r < 0.02 else 'FAIL'}] ⚠ Fig. 15 IS THE SAME "
              f"ARTWORK AS trumpy_callier_q.py's Fig. 5, NOT A SECOND "
              f"MEASUREMENT (shared author, both after Streiffert 1947). Used "
              f"only as a pipeline reproducibility check: two PDFs, two scans, "
              f"one drawing -- rms {r:.4f} Q over {len(diffs)} points")
        if r >= 0.02:
            bad += 1

        # ---- against what the database stores -----------------------------
        tech = [q for q in fp.FILM_PROFILES if q.name == "TECHNICOLOR_THREE_STRIP"]
        if tech and tech[0].dye_density.has_data:
            st = tech[0].dye_density
            got = np.array([norm["cyan"], norm["magenta"], norm["yellow"]])
            n = len(st.d_cyan)
            g0 = np.arange(st.lambda_start_nm,
                           st.lambda_start_nm + n * st.lambda_step_nm,
                           st.lambda_step_nm)[:n]
            worst = 0.0
            for stored, k in ((st.d_cyan, "cyan"), (st.d_magenta, "magenta"),
                              (st.d_yellow, "yellow")):
                v = np.interp(g0, grid, norm[k])
                worst = max(worst, float(np.abs(np.array(stored) - v).max()))
            okk = worst < 0.002
            print(f"  [{'OK  ' if okk else 'FAIL'}] TECHNICOLOR_THREE_STRIP."
                  f"dye_density reproduces this trace to {worst:.4f} "
                  f"(peak-normalised units)")
            if not okk:
                bad += 1

    print()
    if bad:
        print(f"  [FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("  [OK  ] Fig. 16, 21, 22 and 61 digitised; equation (7) closes "
          "across two figures; Table 3 consistent with Fig. 62; no scanner "
          "value written to any film profile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
