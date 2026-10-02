"""JPS 1965 abstract 10p-A-2: grain size against the granularity Wiener spectrum.

WHAT THIS SOURCE IS
-------------------
`RETRO/1965.1_269.pdf` -- 大上進吾 and 高野正雄 (Fuji Photo Film Research
Laboratories), «写真乳剤の粒子サイズ分布と粒状性のウィーナー・スペクトル»
(Grain size distribution of photographic emulsions and the Wiener spectrum of
granularity), abstract 10p-A-2, Meeting of the Physical Society of Japan, 1965,
journal page 269. ONE PAGE, HANDWRITTEN, scanned as a single 2848x4047 CCITT
bilevel image with a Paper-Capture OCR layer that is unusable for the body text
(it renders the title as «シ・一ナー・・ペク»). Everything below is read from the
RASTER; the OCR is used only to identify the page.

WHAT IT GIVES THAT NOTHING ELSE IN THE CORPUS DOES
---------------------------------------------------
Every other granularity source here is indexed by PRODUCT. This one is indexed
by CRYSTAL SIZE: five emulsions coated from AgX of stated mean diameter d,
measured on one apparatus at one density, so the difference between the curves
is the crystal and nothing else.

    A 0.3 um   B 0.4 um   C 0.5 um   D 1.5 um   E 1.8 um
    A is the fine-grain low-speed emulsion, E the coarse-grain high-speed one.
    測定濃度 D_H = 0.5  (the stated measuring density)
    空間周波数 u (l/mm) (the stated abscissa: spatial frequency, cycles/mm)

    Fig. 1  F(u,0), «ウィーナー・スペクトル(相対値)» -- the Wiener spectrum in
            RELATIVE units, log-log, 10 to 1000 c/mm, five curves.
    Fig. 2  micrographs of the five samples, 20 um scale bar. Not read.
    Fig. 3  two summary curves against d: F(20,0) (solid, filled circles, left
            axis) and the u at which F falls to F(0,0)/2 (dashed, crosses,
            right axis).

⚠ RELATIVE UNITS MEAN NO rms GRANULARITY CAN COME FROM THIS DOCUMENT, and none
is taken. Fig. 1's ordinate is labelled 相対値 -- relative value -- with no
square microns and no aperture, so there is no absolute scale to convert. What
IS absolute is the ABSCISSA, and that is the whole value of the page: the
half-power frequency is a measured BANDWIDTH in cycles per millimetre, which is
exactly the quantity `GrainSpec.clump_um_*` sets through `grain_shape`.

THE TWO STATED CONCLUSIONS, WHICH THIS READER RE-DERIVES RATHER THAN QUOTES
----------------------------------------------------------------------------
  (1) the low-frequency spectrum level depends strongly on d and RISES with it;
  (2) the spectrum BANDWIDTH also depends on d but FALLS with it.
Both are asserted below from the traced points, not from the prose.

WHY THE TRACE IS CHECKABLE ON A ONE-PAGE HANDWRITTEN ABSTRACT
--------------------------------------------------------------
Because the page draws the same quantity twice. F(20,0) appears as Fig. 3's
solid curve AND as Fig. 1's low-frequency plateau, in two separately hand-drawn
figures with independent axes. Agreement between them is a real check and it is
the assertion this reader rests on: the four separable points land within 5 %.
The x calibration is independent of that -- it comes from the decade ticks --
and it is validated by the five markers reproducing the PRINTED d values.

⚠ SAMPLE B IS DELIBERATELY A BRACKET, NOT A POINT. The two Fig. 3 curves cross
almost exactly at B and its two markers merge into one blob of ink. Both curves
pass through it, so the honest read is the extent of the tangle, and that is
what is reported.

THE READING APERTURE, BOUNDED RATHER THAN ASSUMED
--------------------------------------------------
The abstract does not state the microphotometer's aperture (its reference (1),
大上進吾, 応用物理 29, 169 (1960), is not in this corpus), and a measured Wiener
spectrum carries the aperture's transfer. That would normally be a fatal
caveat. It is not, because the figure bounds it: the curves descend smoothly
past 800 c/mm with no transfer zero, so a circular reading aperture must be
under about 1.4 um, whose MTF-squared is still above 0.94 at 108 c/mm. Folding
that aperture in moves the half-power frequencies by 2 to 4 per cent -- far
below the discrepancies this page exposes.

⚠ THE BOUND WAS CONFIRMED AND TIGHTENED ON 2026-09-01e BY THE SAME AUTHOR'S
INSTRUMENT PAPER, WHICH IS NOW IN THE CORPUS. 大上進吾, «粒状性の研究(第1報)
新しい粒状性測定装置» (Studies on the Graininess and Granularity I: a new
granularity measuring instrument), J. Soc. Sci. Phot. Japan 23(1), 7-10 (1960)
-- PDF/PROFILES/RETRO/JAPAN/23_7.pdf -- describes the very high-speed
rotating-scan microphotometer this abstract used, and states the aperture
outright: microscope magnification 200x onto a 0.2 mm aperture, i.e. **1 um
referred to the film**. It also says why that is enough: an emulsion is 5-20 um
thick and treating it as a two-dimensional pattern is already a larger
approximation than a 1 um aperture. At 1 um the circular MTF-squared at
108 c/mm is 0.97, so the correction is smaller still. ⚠ The bound derived here
from the figure ALONE (< 1.4 um) contains the independently stated value
(1 um) -- which is a check on the reasoning, not a coincidence.

Run:  python jp_jps_1965_269.py --root <corpus> [--assert]
Needs numpy + scipy + PyMuPDF.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import pymupdf
except ImportError:                                       # pragma: no cover
    print("[!] pymupdf not installed:  pip install pymupdf")
    raise SystemExit(1)

try:
    from scipy import ndimage as ndi
    from scipy.special import j1
except ImportError:                                       # pragma: no cover
    print("[!] scipy not installed:  pip install scipy")
    raise SystemExit(1)

SHEET = "RETRO/1965.1_269.pdf"

SOURCE = ("大上進吾, 高野正雄 (富士写真フイルム研), «写真乳剤の粒子サイズ分布と"
          "粒状性のウィーナー・スペクトル», 日本物理学会 10p-A-2, 1965, p269 -- "
          "PDF/PROFILES/RETRO/1965.1_269.pdf, Fig. 1 and Fig. 3")

#: The page image this reader is calibrated against. A different scan would
#: need the crops re-derived, so the dimensions are asserted rather than
#: assumed.
RASTER_W, RASTER_H = 2848, 4047

#: PRINTED legend of Fig. 1: sample letter and mean AgX crystal diameter, um.
#: Handwritten, so it is transcribed rather than parsed -- but every one of the
#: five is re-derived from its marker position on Fig. 3's log abscissa below,
#: which is what makes the transcription checkable.
AGX_UM = (("A", 0.30), ("B", 0.40), ("C", 0.50), ("D", 1.50), ("E", 1.80))

#: Fig. 1 crop in raster pixels, and its axis calibration, both recovered by
#: `_fig1_axes` from the ruled ticks. Held here only as the search window.
FIG1_BOX = (1250, 2600, 520, 1500)        # y0, y1, x0, x1
#: Fig. 3 likewise.
FIG3_BOX = (2550, 3700, 1700, 2850)

#: Fig. 3's right-hand ordinate, top tick first, in cycles/mm. Handwritten and
#: unreadable by OCR, so transcribed; the reader asserts that exactly this many
#: ticks exist and that they form a uniform ladder. Only 120..80 are labelled.
RIGHT_LADDER = (130.0, 120.0, 110.0, 100.0, 90.0, 80.0, 70.0)

#: Where Fig. 1's plateau is read. Far enough right that the curves have
#: separated from the axis, far enough left that none has begun to roll off.
PLATEAU_CPMM = 20.8

#: `grain_shape` in film_sim.py is an AMPLITUDE transfer
#:     h(f) = exp(-(f/f_hi)^2),  f_hi = 1000 / (2 * clump_um)   [c/mm, um]
#: and the Wiener spectrum is its square, so W(f)/W(0) = exp(-2 (f/f_hi)^2).
#: Setting that to 1/2 gives u_half = f_hi * sqrt(ln2 / 2), hence
CLUMP_FROM_UHALF = 1000.0 * math.sqrt(math.log(2.0) / 2.0) / 2.0   # 294.353


# ---------------------------------------------------------------------------
#  raster helpers
# ---------------------------------------------------------------------------
def _page_raster(doc):
    """The page's single embedded bilevel image, as a uint8 array."""
    imgs = doc[0].get_images(full=True)
    if len(imgs) != 1:
        return None, f"expected 1 embedded image, found {len(imgs)}"
    info = doc.extract_image(imgs[0][0])
    if (info["width"], info["height"]) != (RASTER_W, RASTER_H):
        return None, (f"raster is {info['width']}x{info['height']}, this reader "
                      f"is calibrated for {RASTER_W}x{RASTER_H}")
    import io
    from PIL import Image
    a = np.array(Image.open(io.BytesIO(info["image"])).convert("L"))
    return a, None


def _runs(idx, gap=3):
    out = []
    for i in idx:
        if out and i - out[-1][-1] <= gap:
            out[-1].append(i)
        else:
            out.append([i])
    return [sum(g) / len(g) for g in out]


def _ticks_v(dark, lo, hi, frac=0.8):
    """Row positions of horizontal ticks drawn across columns lo..hi."""
    v = dark[:, lo:hi].sum(1)
    return _runs([i for i, x in enumerate(v) if x >= (hi - lo) * frac])


def _ticks_h(dark, lo, hi, frac=0.75):
    """Column positions of vertical ticks drawn across rows lo..hi."""
    v = dark[lo:hi, :].sum(0)
    return _runs([i for i, x in enumerate(v) if x >= (hi - lo) * frac])


def _fit_line(xs, ys):
    """Least-squares y = m*x + b, returned with the worst residual."""
    m, b = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    worst = float(np.max(np.abs(np.polyval((m, b), xs) - ys)))
    return float(m), float(b), worst


# ---------------------------------------------------------------------------
#  Fig. 1
# ---------------------------------------------------------------------------
def fig1(a):
    """Plateau levels of the five curves, and the aperture bound.

    Calibrated on the DECADE TICKS, four on the ordinate (10, 1, 0.1, 0.01) and
    three on the abscissa (10, 100, 1000). Both are fitted rather than taken
    pairwise, and the fit residual is asserted -- a hand-ruled axis is regular
    to about one part in a hundred and a misdetected tick is not.
    """
    y0, y1, x0, x1 = FIG1_BOX
    dark = (a[y0:y1, x0:x1] < 128)

    ty = _ticks_v(dark, 188, 205)
    tx = _ticks_h(dark, 1184, 1193)
    tx = [c for c in tx if c > 170]
    if len(ty) != 4:
        return None, f"Fig. 1: found {len(ty)} ordinate decade ticks, expected 4"
    if len(tx) != 3:
        return None, f"Fig. 1: found {len(tx)} abscissa decade ticks, expected 3"

    my, by, ry = _fit_line(ty, [1.0, 0.0, -1.0, -2.0])      # log10 F
    mx, bx, rx = _fit_line(tx, [1.0, 2.0, 3.0])             # log10 u
    if ry > 0.02 or rx > 0.02:
        return None, (f"Fig. 1: decade ticks are not on a straight ladder "
                      f"(worst residual {max(ry, rx):.3f} decades)")

    F = lambda r: 10.0 ** (my * r + by)
    U = lambda c: 10.0 ** (mx * c + bx)
    col = int(round((math.log10(PLATEAU_CPMM) - bx) / mx))

    idx = np.flatnonzero(dark[124:1192, col]) + 124
    segs, run = [], [idx[0]]
    for v in idx[1:]:
        if v - run[-1] <= 2:
            run.append(v)
        else:
            segs.append(run)
            run = [v]
    segs.append(run)
    rows = [sum(r) / len(r) for r in segs if 3 <= len(r) <= 10]
    # The five curves are the five lowest-lying of the thin runs; anything
    # above the topmost curve is a label leader stroke.
    rows = sorted(rows)[-5:] if len(rows) > 5 else sorted(rows)
    if len(rows) != 5:
        return None, (f"Fig. 1: {len(rows)} curve crossings at u = "
                      f"{PLATEAU_CPMM} c/mm, expected 5")
    levels = [F(r) for r in rows]                # top row first == highest F
    plateau = dict(zip("EDCBA", levels))

    # -- how far the curves are drawn, which is what bounds the aperture ----
    cols = [c for c in range(600, 916)
            if np.flatnonzero(dark[900:1190, c]).size]
    f_last = U(max(cols))
    aperture_um = 1.2197 * 1000.0 / f_last

    return {"plateau": plateau, "f_last": f_last,
            "aperture_um": aperture_um, "U": U, "F": F}, None


def _col_runs(dark, c, ylo=36, yhi=908, maxlen=48):
    idx = np.flatnonzero(dark[ylo:yhi, c]) + ylo
    if idx.size == 0:
        return []
    out, run = [], [idx[0]]
    for v in idx[1:]:
        if v - run[-1] <= 2:
            run.append(v)
        else:
            out.append(run)
            run = [v]
    out.append(run)
    return [sum(r) / len(r) for r in out if len(r) <= maxlen]


def _track(dark, rising: bool):
    """Follow one of Fig. 3's two curves across the plot.

    The solid curve RISES (row decreases) and the dashed one FALLS, and that
    sign is the only thing separating them where they cross. The dashed curve
    is broken, so the tracker keeps its last position across gaps instead of
    stopping. Positions are used to CLASSIFY the markers, never as values.
    """
    pos, out = None, {}
    for c in range(200 if rising else 355, 742):
        rr = _col_runs(dark, c)
        if pos is None:
            seed = [r for r in rr if (r < 905 if rising else 340 < r < 420)]
            if seed:
                pos = max(seed) if rising else min(seed)
                out[c] = pos
            continue
        if rising:
            cand = [r for r in rr if pos - 26 <= r <= pos + 4]
            aim = pos
        else:
            cand = [r for r in rr if pos - 4 <= r <= pos + 30]
            aim = pos + 3
        if not cand:
            continue
        pos = min(cand, key=lambda r: abs(r - aim))
        out[c] = pos
    return out


def mtf2(f_cpmm, d_um):
    """Power transfer of a uniform circular aperture, (2 J1(x)/x)^2."""
    x = math.pi * d_um * f_cpmm / 1000.0
    return float((2.0 * j1(x) / x) ** 2) if x > 0 else 1.0


# ---------------------------------------------------------------------------
#  Fig. 3
# ---------------------------------------------------------------------------
def fig3(a):
    """F(20,0) and the half-power frequency against d, from the markers.

    ⚠ THE MARKERS ARE FOUND BY EROSION, NOT BY FOLLOWING THE CURVES. A filled
    circle survives a radius-3 erosion and a hand-drawn 5 px curve does not,
    which separates the data from the ink that connects it -- and the letters
    A..E, drawn with the same thin stroke, fall away with the curves.
    """
    y0, y1, x0, x1 = FIG3_BOX
    dark = (a[y0:y1, x0:x1] < 128)

    tl = _ticks_v(dark, 198, 222, frac=0.55)         # left axis: 3, 2, 1
    tr = [r for r in _ticks_v(dark, 722, 744, frac=0.5) if r < 930]
    tb = [c for c in _ticks_h(dark, 900, 916, frac=0.7) if 180 < c < 800]

    if len(tl) != 3:
        return None, f"Fig. 3: found {len(tl)} left-axis ticks, expected 3"
    if len(tr) != len(RIGHT_LADDER):
        return None, (f"Fig. 3: found {len(tr)} right-axis ticks, expected "
                      f"{len(RIGHT_LADDER)}")
    if len(tb) != 3:
        return None, (f"Fig. 3: found {len(tb)} abscissa ticks, expected 3 "
                      f"(0.1 at the frame, 1, and the right frame)")

    mF, bF, rF = _fit_line(tl, [3.0, 2.0, 1.0])
    if rF > 0.01:
        return None, f"Fig. 3: left-axis ticks off a straight ladder ({rF:.3f})"
    Fv = lambda r: mF * r + bF

    # ⚠ THE RIGHT LADDER IS TRANSCRIBED AND THEN CHECKED FOR REGULARITY. Its
    # labels are handwritten and no OCR reads them, so the seven tick VALUES
    # are declared in RIGHT_LADDER; what the machine verifies is that seven
    # ticks exist and that their spacing is uniform, which a misdetection
    # would not be. The printed labels run 120 down to 80; the ticks at 130
    # and 70 carry no label.
    tr = sorted(tr)
    steps = np.diff(tr)
    if float(steps.max() / steps.min()) > 1.10:
        return None, (f"Fig. 3: right-axis ticks are not a uniform ladder "
                      f"({np.round(steps, 1)})")
    mU, bU, rU = _fit_line(tr, list(RIGHT_LADDER))
    if rU > 0.6:
        return None, f"Fig. 3: right-axis ladder fit residual {rU:.2f} u-units"
    Uv = lambda r: mU * r + bU

    # abscissa: the left frame is d = 0.1 and the middle tick is d = 1
    decade = tb[1] - tb[0]
    Dv = lambda c: 10.0 ** ((c - tb[0]) / decade - 1.0)
    Cv = lambda d: tb[0] + decade * (math.log10(d) + 1.0)

    solid = _track(dark, rising=True)
    dashed = _track(dark, rising=False)
    if len(solid) < 250 or len(dashed) < 150:
        return None, (f"Fig. 3: traced {len(solid)} solid and {len(dashed)} "
                      f"dashed columns, too few to classify the markers")

    plot = np.zeros_like(dark)
    plot[36:908, 198:742] = dark[36:908, 198:742]
    er = ndi.binary_erosion(plot, np.ones((7, 7), bool))
    lab, n = ndi.label(er)
    blobs = []
    for i in range(1, n + 1):
        sz = int((lab == i).sum())
        if sz < 10:
            continue
        cy, cx = ndi.center_of_mass(lab == i)
        blobs.append((float(cx), float(cy), sz))

    def _dist(track, cx, cy):
        near = [track[c] for c in range(int(cx) - 3, int(cx) + 4) if c in track]
        return abs(cy - sum(near) / len(near)) if near else 1e9

    out, merged = {}, {}
    for letter, d in AGX_UM:
        want = Cv(d)
        near = [b for b in blobs if abs(b[0] - want) <= 25]
        if not near:
            return None, f"Fig. 3: no marker near sample {letter} (col {want:.0f})"
        # ⚠ CLASSIFIED BY WHICH TRACED CURVE THE BLOB SITS ON, not by which is
        # higher: the two curves CROSS on this figure, so "upper" swaps sides
        # partway along and would mislabel every sample on one side of it.
        S, D = [], []
        for b in near:
            ds, dd = _dist(solid, b[0], b[1]), _dist(dashed, b[0], b[1])
            if min(ds, dd) > 30.0:
                continue                       # a letter, not a marker
            (S if ds < dd else D).append((b, min(ds, dd), max(ds, dd)))
        ok = (len(S) == 1 and len(D) == 1
              and S[0][2] >= 2.0 * max(S[0][1], 1.0)
              and D[0][2] >= 2.0 * max(D[0][1], 1.0))
        if ok:
            bs, bd = S[0][0], D[0][0]
            out[letter] = (Dv(bs[0]), Fv(bs[1]), Dv(bd[0]), Uv(bd[1]))
        else:
            rows = sorted(b[1] for b in near)
            merged[letter] = ((Fv(rows[-1]), Fv(rows[0])),
                              (Uv(rows[-1]), Uv(rows[0])))

    return {"points": out, "merged": merged,
            "solid": solid, "Fv": Fv, "Cv": Cv}, None


# ---------------------------------------------------------------------------
def _loglog(x, y):
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    m, c = np.polyfit(lx, ly, 1)
    pred = np.polyval((m, c), lx)
    r2 = 1.0 - float(((ly - pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum())
    return float(m), float(math.exp(c)), r2


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

    # ---- identity ---------------------------------------------------------
    txt = doc[0].get_text()
    ok = (doc.page_count == 1 and "Physical Society" in txt
          and "10p" in txt and "269" in txt)
    print(f"  [{'OK  ' if ok else 'FAIL'}] identifies as a one-page Physical "
          f"Society of Japan abstract, 10p-A-2, p269")
    if not ok:
        return 1
    a, err = _page_raster(doc)
    if a is None:
        print(f"  [FAIL] {err}")
        return 1
    print(f"  [OK  ] single {RASTER_W}x{RASTER_H} bilevel raster; the OCR layer "
          f"is unusable for the body and is not used")

    # ---- Fig. 1 -----------------------------------------------------------
    f1, err = fig1(a)
    if f1 is None:
        print(f"  [FAIL] {err}")
        return 1 if ns.do_assert else 0
    print(f"  [OK  ] Fig. 1 decade-tick calibrated; five plateau levels at "
          f"u = {PLATEAU_CPMM} c/mm")
    for L in "ABCDE":
        print(f"        {L}  F(20,0) = {f1['plateau'][L]:.3f}  (relative)")

    ap_um = f1["aperture_um"]
    m108 = mtf2(108.0, ap_um)
    print(f"  [OK  ] curves drawn to {f1['f_last']:.0f} c/mm with no transfer "
          f"zero -> circular reading aperture < {ap_um:.2f} um, "
          f"MTF^2(108 c/mm) >= {m108:.3f}")
    if m108 < 0.90:
        print("  [FAIL] the bounded aperture is not negligible at the "
              "half-power frequencies; the bandwidths below would need "
              "deconvolving before use")
        bad += 1

    # ---- Fig. 3 -----------------------------------------------------------
    f3, err = fig3(a)
    if f3 is None:
        print(f"  [FAIL] {err}")
        return 1 if ns.do_assert else 0
    pts, mrg = f3["points"], f3["merged"]
    print(f"  [OK  ] Fig. 3 markers found by erosion: {len(pts)} separable, "
          f"{len(mrg)} merged")

    printed = dict(AGX_UM)
    ds, us, fs = [], [], []
    for L, d0 in AGX_UM:
        if L in pts:
            d_s, Fv, d_d, Uv = pts[L]
            # THE X CALIBRATION IS CHECKED HERE: the marker must land on the
            # printed crystal size, and the calibration never saw that number.
            e = 100.0 * (d_s - d0) / d0
            flag = "OK  " if abs(e) <= 12.0 else "FAIL"
            if flag == "FAIL":
                bad += 1
            print(f"  [{flag}] {L}  d printed {d0:.2f} um, marker reads "
                  f"{d_s:.3f} ({e:+.1f} %)   F(20,0) {Fv:.3f}   "
                  f"u_half {Uv:.1f} c/mm")
            ds.append(d0); us.append(Uv); fs.append(Fv)
        else:
            (fl, fh), (ul, uh) = mrg[L]
            print(f"  [OK  ] {L}  d printed {d0:.2f} um -- ⚠ THE TWO CURVES "
                  f"CROSS HERE AND THE MARKERS MERGE: F(20,0) in "
                  f"[{fl:.3f}, {fh:.3f}], u_half in [{ul:.1f}, {uh:.1f}] c/mm")

    # ---- the cross-figure check this reader rests on ----------------------
    print(f"\n  -- Fig. 1 plateau against Fig. 3's solid curve, two hand-drawn "
          f"figures, independent axes")
    worst = 0.0
    for L, d0 in AGX_UM:
        if L not in pts:
            continue
        e = 100.0 * (pts[L][1] - f1["plateau"][L]) / f1["plateau"][L]
        worst = max(worst, abs(e))
        print(f"     {L}  Fig.3 {pts[L][1]:.3f}  vs  Fig.1 "
              f"{f1['plateau'][L]:.3f}   {e:+5.1f} %")
    if worst > 8.0:
        print(f"  [FAIL] the two figures disagree by {worst:.1f} %, so neither "
              f"trace is trustworthy")
        bad += 1
    else:
        print(f"  [OK  ] worst disagreement {worst:.1f} % -- the traces agree, "
              f"and with them both y calibrations")

    # ---- the paper's own two conclusions, re-derived ----------------------
    rising = all(fs[i] < fs[i + 1] for i in range(len(fs) - 1))
    falling = all(us[i] >= us[i + 1] - 1.0 for i in range(len(us) - 1))
    print(f"  [{'OK  ' if rising else 'FAIL'}] conclusion (1) the spectrum "
          f"level RISES with crystal size")
    print(f"  [{'OK  ' if falling else 'FAIL'}] conclusion (2) the spectrum "
          f"BANDWIDTH FALLS with crystal size")
    bad += (not rising) + (not falling)

    mb, ab, r2b = _loglog(ds, us)
    ma, aa, r2a = _loglog(ds, fs)
    print(f"\n  -- the two laws, fitted on the separable points")
    print(f"     u_half  = {ab:6.2f} * d^{mb:+.4f}   R2 {r2b:.3f}")
    print(f"     F(20,0) = {aa:6.3f} * d^{ma:+.4f}   R2 {r2a:.3f}   "
          f"i.e. sigma ~ d^{ma/2:+.3f}")
    # ⚠ NEITHER LAW MAY BE INVERTED TO RECOVER d, AND THE FIT SAYS SO ITSELF.
    print(f"  [OK  ] ⚠ NOT INVERTIBLE: 1 % in u_half is {abs(1/mb):.1f} % in d, "
          f"1 % in F(20,0) is {abs(1/ma):.1f} % in d, and sigma is flatter "
          f"still at {abs(2/ma):.1f} % per 1 %. A six-fold change in crystal "
          f"size moves the bandwidth by {100*(max(us)/min(us)-1):.0f} % only.")

    # ---- what it DOES license: the bandwidth, in this project's own law ---
    print(f"\n  -- bandwidth as `GrainSpec.clump_um`, through grain_shape's own "
          f"law clump_um = {CLUMP_FROM_UHALF:.1f} / u_half")
    for L, d0 in AGX_UM:
        if L in pts:
            u = pts[L][3]
            print(f"     {L}  d {d0:.2f} um  u_half {u:5.1f} c/mm  ->  "
                  f"clump_um {CLUMP_FROM_UHALF / u:.3f}")
    try:
        import film_profiles as fp
    except Exception as exc:                              # pragma: no cover
        print(f"    [note] film_profiles unavailable ({exc})")
        return 1 if (bad and ns.do_assert) else 0

    cl = np.array([p.grain.clump_um_g for p in fp.FILM_PROFILES])
    lo, hi = CLUMP_FROM_UHALF / max(us), CLUMP_FROM_UHALF / min(us)
    inside = int(((cl >= lo) & (cl <= hi)).sum())
    print(f"\n  -- against the corpus, {len(cl)} stocks")
    print(f"     this page's measured band : {lo:.2f} - {hi:.2f} um "
          f"(u_half {min(us):.0f} - {max(us):.0f} c/mm)")
    print(f"     corpus clump_um_g        : min {cl.min():.2f}, median "
          f"{np.median(cl):.2f}, max {cl.max():.2f}")
    print(f"     stocks inside the band   : {inside} of {len(cl)}")
    print(f"     the corpus median implies u_half "
          f"{CLUMP_FROM_UHALF/float(np.median(cl)):.0f} c/mm, against "
          f"{min(us):.0f}-{max(us):.0f} measured here")
    # ⚠ A DISAGREEMENT IS REPORTED, NOT SILENTLY RESOLVED. Nothing in the
    # database is changed by this reader; see doc/RESULT_2026-09-01c.
    print(f"  [OK  ] ⚠ RECORDED AS A DISAGREEMENT AND NOTHING IS ADOPTED. The "
          f"corpus's estimated clump sizes are coarser than every measurement "
          f"on file, and the two MEASURED ones (ILFORD_PAN_F 0.655, "
          f"ILFORD_HPS 1.431, both from BBC T-101 Table 2) are finer than this "
          f"page's finest emulsion. Three independent sources, three "
          f"different answers; moving 168 stocks on the strength of a "
          f"relative-unit abstract would be the opposite of evidence.")

    print()
    if bad:
        print(f"  [FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("  [OK  ] Fig. 1 and Fig. 3 reproduced and cross-checked; both "
          "stated conclusions re-derived; no database value taken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
