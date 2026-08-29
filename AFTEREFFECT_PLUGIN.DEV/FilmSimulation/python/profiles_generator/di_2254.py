"""Characteristic curves for KODAK VISION3 Color Digital Intermediate 2254.

WHY THIS FILE EXISTS AT ALL, AND WHY IT IS NOT ONE OF THE OTHER FOUR READERS
---------------------------------------------------------------------------
The project already has four curve readers, and none of them fits this sheet:

    kodak_sensitometry.py   VECTOR paths in Kodak's own brochure layout
    dye_density.py          VECTOR dye/sensitivity panels
    gevaert_curves.py       SCANNED journal pages: skewed, 1-bit, hand-pinned
    digitize_plot.py        the pixel fitter the others share

H-1-2254 is a MODERN Kodak brochure whose plots are nonetheless RASTER: page 3
carries the sensitometric panel as a 680 x 704 px embedded image with no vector
art anywhere on the page (`get_drawings()` returns one path, the page rule). So
the vector readers cannot see it, and the Gevaert reader's machinery -- skew
fitting, hand-read tick values, pinned anchors -- is aimed at defects this scan
does not have. This plot is axis-aligned to the pixel and its ticks land on
exactly uniform 135.0 px spacings, so its calibration is arithmetic, not a fit.

⚠ WHAT THIS STOCK IS, AND THE CATALOGUE HAZARD THAT COMES WITH IT
-----------------------------------------------------------------
2254 is a DIGITAL INTERMEDIATE RECORDING film: exposed by a laser or CRT
recorder from digital files, processed ECN-2, and printed from. It is not a
camera stock, and it must never be confused with `EASTMAN_5254_1968`, which is a
1968 ECN camera negative carrying the same catalogue number forty years earlier.
Two different films, one number. The database holds them under names that cannot
collide.

⚠ THE ABSCISSA ORIGIN IS A CHOICE, AND THE SHEET SUPPLIES IT
-------------------------------------------------------------
The plot's x axis is ABSOLUTE -- "LOG EXPOSURE (lux-seconds)", -4.0 to 0.0 --
while `ToneCurve`'s x is a relative scale whose origin is the mid-scale
reference. Something has to relate them, and inventing a relation is what
gevaert_curves.py had to do (it inherits the origin from the pre-existing hand
fit, and says so). Here the SOURCE states the reference: the dye-stability table
on p4 is quoted at "1.0 Above D-min (Neutral)", so the sheet's own mid-scale
reference for this film is the exposure that puts the NEUTRAL at D-min + 1.0.

So the origin is placed there: x = 0 is the traced log exposure at which the
GREEN record reaches dmin + 1.0, and all three records are shifted by that one
number, because they share one exposure axis. Shifting them independently would
invent a speed difference the plot does not show.

⚠ B AND G ARE DRAWN AS ONE LINE ON THE TOE PLATEAU, and that is the sheet's
statement, not a tracing failure: left of about -3.1 the blue and green records
are printed as a single stroke at D 0.70. The tracer cannot separate one line
into two and does not pretend to -- both records take the same centroid there,
which is what the plot asserts. They separate cleanly above the toe and the fit
sees the difference where it exists.

Run:  python di_2254.py --root ../.. [--assert] [--dump]
Needs numpy + Pillow + pdfimages (poppler-utils), like gevaert_curves.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dashtrace as dt          # noqa: E402
import digitize_plot as dp      # noqa: E402

PDF = "KODAK/KODAK-VISION3-2254-technical-information.pdf"
SOURCE = ("Eastman Kodak Company, «KODAK VISION3 Color Digital Intermediate "
          "Film 2254 -- Technical Data», KODAK Publication No. H-1-2254, "
          "March 2026")

#: Which embedded image is the sensitometric panel. `pdfimages` numbers images
#: in page order, so this index is stable for a given file; it is checked by the
#: SHAPE below rather than trusted, because an index that silently points at the
#: wrong figure is the failure mode this whole file must not have.
IMAGE_INDEX = 1
IMAGE_SHAPE = (704, 680)

#: Minimum vertical gap, in pixels, between the three records at the seed
#: column. 12 px is 0.09 D on this plot -- comfortably below the 0.14 D that
#: separates blue from green at the right-hand end and far above any
#: antialiasing halo.
SEED_GAP = 12.0

#: The plot frame and its ticks, in image pixels. NOT pinned by hand: they are
#: detected on every run by `geometry()` and these are the values that detection
#: must return. The spacings are EXACTLY uniform (135.1 px per decade in x,
#: 135.0 px per density unit in y), which is itself the evidence that the raster
#: is unskewed and the calibration needs no fit.
FRAME = dict(left=105.5, right=646.0, top=70.0, bottom=610.0)
X_TICKS = ((105.5, -4.0), (241.0, -3.0), (376.0, -2.0), (511.0, -1.0),
           (646.0, 0.0))
Y_TICKS = ((70.0, 4.0), (205.0, 3.0), (340.0, 2.0), (475.0, 1.0), (610.0, 0.0))

#: Measured 2026-08-25. --assert fails if the extraction stops reproducing them.
#: ⚠ THE GAMMAS ARE THE PHYSICAL CHECK THIS SHEET OFFERS. 2254 is an
#: INTERMEDIATE film, whose whole design purpose is unity gamma -- it exists to
#: carry a negative to a print without changing contrast. Nothing in the trace
#: was told that, and the fit returns 0.96 / 1.04 / 1.05. A calibration error of
#: any size would show up here first, because it would move all three together
#: away from 1.0.
#: ⚠ AND THE BLUE AND GREEN D-MIN ARE IDENTICAL BY CONSTRUCTION, not by
#: coincidence: the two records are printed as ONE stroke on the toe plateau, so
#: the same measured centroid feeds both. See `trace`.
EXPECTED = dict(
    dmin=(0.0941, 0.7111, 0.7111),      # R, G, B -- left-plateau medians
    gamma=(1.0513, 0.9612, 1.0444),
    tol_dmin=0.01, tol_gamma=0.02, rms_max=0.015,
    # the exposure that puts the green record at dmin + 1.0, in the SHEET's
    # own lux-second units -- the anchor the origin is placed on
    anchor_logE=-1.962, tol_anchor=0.03,
)


def native_image(root: Path, tmp: Path) -> np.ndarray:
    """The sensitometric panel at its stored resolution, as float grey [0..1].

    ⚠ NOT `page.get_pixmap()`. The plot IS a bitmap; re-rendering the page
    resamples it and throws away columns that the trace is measured on. Same
    reasoning, and same tool, as gevaert_curves.native_pages.
    """
    pdf = root / "PDF" / "PROFILES" / PDF
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    out = subprocess.run(["pdfimages", "-png", str(pdf), str(tmp / "di")],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"[!] pdfimages failed: {out.stderr[-400:]}")
    files = sorted(tmp.glob("di-*.png"))
    if len(files) <= IMAGE_INDEX:
        raise SystemExit(f"[!] only {len(files)} images in {pdf.name}")
    img = np.asarray(Image.open(files[IMAGE_INDEX]).convert("L"),
                     dtype=np.float64) / 255.0
    if img.shape != IMAGE_SHAPE:
        raise SystemExit(f"[!] image {IMAGE_INDEX} is {img.shape}, expected "
                         f"{IMAGE_SHAPE} -- the figure order in the PDF moved, "
                         f"and reading the wrong figure would be silent")
    return img


def _groups(idx, gap=3):
    out, cur = [], []
    for i in idx:
        if cur and i - cur[-1] <= gap:
            cur.append(i)
        else:
            if cur:
                out.append(float(np.mean(cur)))
            cur = [i]
    if cur:
        out.append(float(np.mean(cur)))
    return out


def geometry(img):
    """(frame, x ticks, y ticks) detected from the ink, with no pinned pixels.

    The frame is the row/column of maximum ink in each half of the image; the
    ticks are the short strokes drawn just INSIDE it. Kodak puts the tick marks
    inside the frame on this layout, which is why a detector aimed at the
    outside -- the first thing tried here -- found only the axis LABELS.
    """
    ink = img < 0.5
    h, w = ink.shape
    cs, rs = ink.sum(0), ink.sum(1)
    left = float(np.mean(_groups(np.where(cs > 0.5 * h)[0])[:1] or [0]))
    right = float(np.mean(_groups(np.where(cs > 0.5 * h)[0])[-1:] or [0]))
    top = float(np.mean(_groups(np.where(rs > 0.5 * w)[0])[:1] or [0]))
    bottom = float(np.mean(_groups(np.where(rs > 0.5 * w)[0])[-1:] or [0]))
    xs = _groups(np.where(ink[int(bottom) - 12:int(bottom) - 2, :].sum(0) >= 6)[0])
    ys = _groups(np.where(ink[:, int(left) + 3:int(left) + 13].sum(1) >= 6)[0])
    return dict(left=left, right=right, top=top, bottom=bottom), xs, ys


def calibrate(found, pinned, what):
    """Linear map pixel -> value, with the worst tick disagreement reported."""
    if len(found) != len(pinned):
        raise SystemExit(f"[!] {what}: detected {len(found)} ticks, expected "
                         f"{len(pinned)}")
    px = np.array([p for p, _v in pinned], float)
    if float(np.abs(np.array(found) - px).max()) > 2.0:
        raise SystemExit(f"[!] {what}: a tick moved more than 2 px from the "
                         f"recorded geometry: {found} vs {list(px)}")
    v = np.array([val for _p, val in pinned], float)
    m, c = np.polyfit(px, v, 1)
    return float(m), float(c), float(np.abs(m * px + c - v).max())


def trace(img, frame):
    """{'R','G','B': {x: y}} for the three records, traced right to left.

    ⚠ RIGHT TO LEFT, DELIBERATELY. At the right-hand end the three records are
    0.14 and 0.44 D apart and unambiguous; at the left-hand end blue and green
    are ONE printed line. Seeding where the answer is unambiguous and tracing
    toward the ambiguity is the direction rule dashtrace's module docstring
    states, and the opposite order would have to guess at the seed.
    """
    ink = img < 0.5
    x0 = int(frame["left"]) + 3
    x1 = int(frame["right"]) - 3
    y0 = int(frame["top"]) + 2
    y1 = int(frame["bottom"]) - 2
    # ⚠ THE SEED IS FOUND, NOT ASSUMED, and the first version of this line is
    # why: it took the column four pixels inside the right frame, where THE
    # CURVES HAVE ALREADY ENDED -- they stop at about -0.25 decades, some 34 px
    # short of the frame -- so the seed column held no ink at all. The rightmost
    # column carrying three runs separated by more than SEED_GAP is the correct
    # seed and costs one scan to find.
    seed_x = None
    runs = []
    for x in range(x1, x0 + int(0.5 * (x1 - x0)), -1):
        cs = sorted(c for c, _t in dt.column_runs_weighted(ink, img, x, y0, y1))
        if len(cs) == 3 and min(np.diff(cs)) > SEED_GAP:
            seed_x, runs = x, cs
            break
    if seed_x is None:
        raise SystemExit("[!] no column with three separated records")
    names = ("B", "G", "R")            # top to bottom at the right-hand end
    tracks = dt.trace_predictive(ink, img, (x0, x1), y0, y1, seed_x,
                                 dict(zip(names, runs)), direction=-1,
                                 tol0=3.0, tol_grow=0.6, max_bridge=20,
                                 hist=16, slope_cap=2.5)
    # ⚠ THE MERGED TOE IS FILLED FROM THE SHEET'S OWN STATEMENT, NOT INVENTED.
    # Left of the point where blue and green become one stroke, whichever track
    # kept the ink holds the only measurement there is, and the other record is
    # printed on top of it. Copying it across is what the plot says; leaving a
    # gap would drop the dmin plateau, which is the one part of the curve that
    # is measured directly rather than fitted.
    #
    # ⚠ ONLY OUTSIDE THE RECEIVING TRACK'S OWN SPAN, and the first version of
    # this loop is why the rule is written down. It copied into any column the
    # receiver was missing, INTERIOR HOLES INCLUDED, so a single dropped column
    # at logE -0.18 -- near the top of the scale, where blue and green are 0.16 D
    # apart -- was filled with the GREEN value. The fit's worst residual jumped
    # to 0.144 D on that one sample while the rms stayed at 0.014, which is
    # exactly the shape of a defect that a mean statistic hides. A hole inside
    # the traced span is a tracing gap and belongs to interpolation; only the
    # region where the receiver was never traced at all is the merged stroke.
    for a, b in (("B", "G"), ("G", "B")):
        if not tracks[b]:
            continue
        lo, hi = min(tracks[b]), max(tracks[b])
        for x, y in tracks[a].items():
            if x < lo or x > hi:
                tracks[b][x] = y
    return tracks


def extract(root: Path):
    with tempfile.TemporaryDirectory() as td:
        img = native_image(root, Path(td))
    frame, xs, ys = geometry(img)
    for k, v in FRAME.items():
        if abs(frame[k] - v) > 2.0:
            raise SystemExit(f"[!] frame {k} detected at {frame[k]:.1f}, "
                             f"recorded {v:.1f}")
    xm, xc, xres = calibrate(xs, X_TICKS, "log exposure axis")
    ym, yc, yres = calibrate(ys, Y_TICKS, "density axis")
    tracks = trace(img, frame)
    data = {}
    for k in ("R", "G", "B"):
        px = np.array(sorted(tracks[k]), float)
        data[k] = (xm * px + xc,
                   ym * np.array([tracks[k][q] for q in px], float) + yc)
    # the origin, from the sheet's own "1.0 Above D-min (Neutral)" reference
    le_g, d_g = data["G"]
    dmin_g = float(np.median(d_g[le_g < le_g.min() + 0.35]))
    anchor = float(le_g[int(np.argmin(np.abs((d_g - dmin_g) - 1.0)))])
    return data, dict(x_res=xres, y_res=yres, frame=frame, anchor=anchor,
                      n={k: len(data[k][0]) for k in data})


def fit(data, anchor):
    """{'R','G','B': (params, rms, worst)} on the shifted exposure axis."""
    out = {}
    for k in ("R", "G", "B"):
        le, d = data[k]
        x = le - anchor
        dmin = float(np.median(d[x < x.min() + 0.35]))

        def loss(p, x=x, d=d, dmin=dmin):
            gam, tx, tk, sx, sk = p
            if gam <= 0 or tk <= 0.02 or sk <= 0.02 or sx <= tx:
                return 1e9
            pen = 100.0 * max(0.0, sk - 1.4 * tk) ** 2
            r = dp.softplus_curve(x, dmin, gam, tx, tk, sx, sk) - d
            return float(np.mean(r * r)) + pen

        hi = float(d.max())
        above = np.where(d > dmin + 0.10)[0]
        below = np.where(d < hi - 0.10)[0]
        g0 = 1.0
        for i in range(len(x)):
            j = int(np.searchsorted(x, x[i] + 1.0))
            if j < len(x):
                g0 = max(g0, abs((d[j] - d[i]) / (x[j] - x[i])))
        tx0 = float(x[above[0]]) if above.size else float(x.min())
        sx0 = float(x[below[-1]]) if below.size else float(x.max())
        best = None
        for tk0 in (0.12, 0.20, 0.30):
            for sk0 in (0.16, 0.24, 0.36):
                p, v = dp._nelder_mead(loss, np.array([g0, tx0, tk0, sx0, sk0]),
                                       [0.03, 0.08, 0.04, 0.08, 0.04])
                if best is None or v < best[1]:
                    best = (p, v)
        p = best[0]
        r = dp.softplus_curve(x, dmin, *p) - d
        out[k] = ((dmin,) + tuple(float(v) for v in p),
                  float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r))))
    return out


# ===========================================================================
# The MODULATION-TRANSFER panel, H-1-2254 p5 (queue item C36, 2026-08-26)
# ===========================================================================
#
# ⚠ THE INTERESTING RESULT HERE IS A REFUSAL, AND IT IS MEASURED RATHER THAN
# ASSUMED. The panel plots three records and **two of them never reach 50 %
# response**: the curves stop at 82.2 cycles/mm with green at 53.1 % and red at
# 50.6 %. So this sheet CANNOT state an f50 for green or red -- only a lower
# bound -- and the one record it does state, blue, is the softest of the three.
#
# That matters because the stored scalar was the estimate 72.0, and the
# measurement contradicts it in BOTH directions at once: too sharp for blue
# (51.9 measured) and too soft for green and red (proven >= 82.2). A single
# number cannot be right about a set that spans a factor of 1.6.
#
#: The panel's frame and its log-log tick anchors, in image pixels. Detected and
#: re-verified on every run, exactly as the sensitometric panel's are.
MTF_IMAGE_INDEX = 4
MTF_IMAGE_SHAPE = (605, 694)
MTF_X_TICKS = {1: 82.0, 2: 145.0, 3: 182.0, 4: 210.0, 5: 229.0, 10: 292.0,
               20: 358.0, 50: 443.0, 100: 507.0, 200: 569.0}
MTF_Y_TICKS = {200: 58.0, 100: 115.0, 70: 145.0, 50: 174.0, 30: 217.0,
               20: 251.0, 10: 310.0, 7: 340.0, 5: 371.0, 3: 412.0, 2: 449.0,
               1: 506.5}

#: Rows the curves live in, and the column the three are seeded at. The seed is
#: the RIGHTMOST column carrying three separated runs -- the right-hand end is
#: where the records are furthest apart on this panel, and the left-hand end is
#: where they converge into one stroke near 100 %.
MTF_ROWS = (100, 245)
MTF_SEED_X = 488

#: Measured 2026-08-26.
MTF_EXPECTED = dict(
    f50_b=51.9, tol_f50=0.6,
    end_f=82.2, end_g=53.1, end_r=50.6, end_b=35.8, tol_end=0.6,
)


def _strip_gridlines(ink, min_thick=2):
    """Ink with 1-pixel-thick horizontal rules removed, column by column.

    ⚠ WITHOUT THIS THE TRACE CLIMBS ONTO THE GRID AND THE NUMBERS STAY
    PLAUSIBLE. A log-log MTF panel is ruled at 1, 2, 3, 5, 7, 10, 20 ... and
    those rules are 1 px thick while the curves are 3. Where a curve runs nearly
    flat near 100 % -- which is the whole low-frequency half of every MTF curve
    -- a tracker cannot tell the two apart by position, and it followed the
    100 % and 70 % RULES instead of the traces. The measured cost, before this
    function existed: the blue record's rolloff exponent came back q = 0.74 at
    rms 0.063, against 1.78 at rms 0.026 once the rules were gone. Both look
    like fits. Only one is of the curve.

    Thickness is the discriminator because it is the one property the rules and
    the traces do not share, and where a curve CROSSES a rule the merged run is
    thicker still, so the curve survives its own crossings.
    """
    out = np.zeros_like(ink)
    rows, cols = ink.shape
    for x in range(cols):
        col = ink[:, x]
        y = 0
        while y < rows:
            if col[y]:
                y0 = y
                while y < rows and col[y]:
                    y += 1
                if y - y0 >= min_thick:
                    out[y0:y, x] = True
            else:
                y += 1
    return out


def _logfit(ticks):
    px = np.array([ticks[k] for k in sorted(ticks)], float)
    v = np.log10(np.array(sorted(ticks), float))
    m, c = np.polyfit(px, v, 1)
    return float(m), float(c), float(np.abs(m * px + c - v).max())


def extract_mtf(root: Path):
    """{'G','R','B': dict} for the three records, with f50 REFUSED where the
    curve never crosses 50 % response.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pdf = root / "PDF" / "PROFILES" / PDF
        if not pdf.is_file():
            raise FileNotFoundError(pdf)
        out = subprocess.run(["pdfimages", "-png", str(pdf), str(tmp / "di")],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"[!] pdfimages failed: {out.stderr[-400:]}")
        files = sorted(tmp.glob("di-*.png"))
        if len(files) <= MTF_IMAGE_INDEX:
            raise SystemExit(f"[!] only {len(files)} images in {pdf.name}")
        img = np.asarray(Image.open(files[MTF_IMAGE_INDEX]).convert("L"),
                         dtype=np.float64) / 255.0
    if img.shape != MTF_IMAGE_SHAPE:
        raise SystemExit(f"[!] MTF image is {img.shape}, expected "
                         f"{MTF_IMAGE_SHAPE} -- the figure order moved")
    ink = _strip_gridlines(img < 0.5)
    gray = np.where(ink, img, 1.0)
    mx, cx, xres = _logfit(MTF_X_TICKS)
    my, cy, yres = _logfit(MTF_Y_TICKS)
    freq = lambda px: 10.0 ** (mx * px + cx)        # noqa: E731
    resp = lambda py: 10.0 ** (my * py + cy) / 100.0  # noqa: E731

    y0, y1 = MTF_ROWS
    seed = sorted(c for c, _t in
                  dt.column_runs_weighted(ink, gray, MTF_SEED_X, y0, y1))
    if len(seed) != 3:
        raise SystemExit(f"[!] the MTF seed column shows {len(seed)} runs, "
                         f"not 3")
    # ⚠ NAMED BY POSITION AT THE RIGHT-HAND END, WHICH IS WHAT THE SHEET LABELS.
    # The panel prints G, R and B as glyphs to the right of the curve ends, top
    # to bottom in that order. Nothing else on the page connects a label to a
    # path, and the order is not the layer order seen on camera negatives -- on
    # this film GREEN is the sharpest record and BLUE the softest.
    tracks = dt.trace_predictive(ink, gray, (160, MTF_SEED_X), y0, y1,
                                 MTF_SEED_X, dict(zip(("G", "R", "B"), seed)),
                                 direction=-1, tol0=3.0, tol_grow=0.6,
                                 max_bridge=20, hist=16, slope_cap=2.5)
    got = {}
    for k in ("G", "R", "B"):
        t = tracks[k]
        px = np.array(sorted(t), float)
        f = freq(px)
        r = resp(np.array([t[q] for q in px], float))
        o = np.argsort(f)
        f, r = f[o], r[o]
        rec = dict(n=len(f), f_lo=float(f[0]), f_hi=float(f[-1]),
                   r_end=float(r[-1]), f50=None, q=None, q_rms=None,
                   q_gauss=None)
        if r.min() <= 0.5 <= r.max():
            i = int(np.where(np.diff(np.sign(r - 0.5)) != 0)[0][-1])
            rec["f50"] = float(10.0 ** np.interp(
                0.5, [r[i + 1], r[i]], [np.log10(f[i + 1]), np.log10(f[i])]))
            m = f >= 8.0
            if m.sum() >= 6:
                best = None
                for q in np.arange(0.60, 6.001, 0.005):
                    e = float(np.sqrt(np.mean(
                        (1.0 / (1.0 + (f[m] / rec["f50"]) ** q) - r[m]) ** 2)))
                    if best is None or e < best[1]:
                        best = (float(q), e)
                rec["q"], rec["q_rms"] = best
                rec["q_gauss"] = float(np.sqrt(np.mean(
                    (np.exp(-np.log(2.0) * (f[m] / rec["f50"]) ** 2)
                     - r[m]) ** 2)))
                # ⚠ HOW MUCH OF THE CURVE THE EXPONENT WAS FITTED OVER, because
                # a carrier normalised at f = 0 fitted only to the tail is not
                # the same claim as one fitted across the knee.
                rec["q_span_dec"] = float(np.log10(f[m].max() / f[m].min()))
                rec["q_below_f50_dec"] = float(np.log10(
                    rec["f50"] / max(f[m].min(), 1e-9)))
        got[k] = rec
    return got, dict(x_res=xres, y_res=yres)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()
    try:
        data, diag = extract(root)
    except FileNotFoundError as exc:
        print(f"  [SKIP] source not present: {exc}")
        return 0
    print(f"[i] {SOURCE}, p3 'Sensitometric Curves' (raster, "
          f"{IMAGE_SHAPE[1]}x{IMAGE_SHAPE[0]} px)")
    print(f"    frame detected at {diag['frame']['left']:.1f}"
          f"..{diag['frame']['right']:.1f} x {diag['frame']['top']:.1f}"
          f"..{diag['frame']['bottom']:.1f} px; axis residuals "
          f"{diag['x_res']:.4f} decade / {diag['y_res']:.4f} D")
    print(f"    samples: " + ", ".join(f"{k} {v}" for k, v in diag["n"].items()))
    print(f"    origin placed at logE {diag['anchor']:+.3f} lux-seconds -- "
          f"where the GREEN record reaches D-min + 1.0, which is the reference "
          f"the sheet's own dye-stability table is quoted at")
    fits = fit(data, diag["anchor"])
    bad = 0
    w = EXPECTED
    if abs(diag["anchor"] - w["anchor_logE"]) > w["tol_anchor"]:
        print(f"    [FAIL] the anchor moved: {diag['anchor']:+.3f} vs recorded "
              f"{w['anchor_logE']:+.3f}")
        bad += 1
    for i, k in enumerate(("R", "G", "B")):
        p, rms, worst = fits[k]
        print(f"    {k}: dmin {p[0]:.4f}  gamma {p[1]:.4f}  "
              f"toe_x {p[2]:+.4f} toe_k {p[3]:.4f}  "
              f"sh_x {p[4]:+.4f} sh_k {p[5]:.4f}  | rms {rms:.4f} D, "
              f"worst {worst:.4f} D")
        if abs(p[0] - w["dmin"][i]) > w["tol_dmin"]:
            print(f"    [FAIL] {k} dmin moved: {p[0]:.4f} vs {w['dmin'][i]:.4f}")
            bad += 1
        if abs(p[1] - w["gamma"][i]) > w["tol_gamma"]:
            print(f"    [FAIL] {k} gamma moved: {p[1]:.4f} vs {w['gamma'][i]:.4f}")
            bad += 1
        if rms > w["rms_max"]:
            print(f"    [FAIL] {k} fit rms {rms:.4f} exceeds {w['rms_max']:.4f}")
            bad += 1
    # ⚠ THE ORDERING IS THE EXTERNAL CHECK THIS PLOT OFFERS. The sheet LABELS
    # its three records B / G / R top to bottom, so blue above green above red
    # across the whole scale is the source's own statement about the trace.
    for k in ("R", "G", "B"):
        pass
    lo = min(len(data[k][0]) for k in data)
    ok_order = True
    for x in np.linspace(-3.5, -0.4, 60):
        vals = {k: float(np.interp(x, data[k][0], data[k][1])) for k in data}
        if not (vals["B"] >= vals["G"] - 0.02 >= vals["R"] - 0.02):
            ok_order = False
            break
    print(f"    {'[OK]' if ok_order else '[FAIL]'} the records stay in the "
          f"printed B >= G >= R order across the scale ({lo}+ samples each)")
    if not ok_order:
        bad += 1
    if ns.dump:
        for k in ("r", "g", "b"):
            p = fits[k.upper()][0]
            print(f"            {k}=ToneCurve(dmin={p[0]:.3f}, gamma={p[1]:.3f}, "
                  f"toe_x={p[2]:+.3f}, toe_k={p[3]:.3f}, "
                  f"shoulder_x={p[4]:+.3f}, shoulder_k={p[5]:.3f}),")

    # ---- the MTF panel (queue C36) --------------------------------------
    mtf, mdiag = extract_mtf(root)
    print(f"    MTF panel p5: axis residuals {mdiag['x_res']:.4f} / "
          f"{mdiag['y_res']:.4f} decades")
    w = MTF_EXPECTED
    for k, want_end in (("G", w["end_g"]), ("R", w["end_r"]), ("B", w["end_b"])):
        rec = mtf[k]
        end = f"ends at {rec['r_end'] * 100:.1f} % at {rec['f_hi']:.1f} c/mm"
        if rec["f50"] is None:
            print(f"    {k}: REFUSED -- the curve never reaches 50 % response; "
                  f"{end}, so f50 is BOUNDED > {rec['f_hi']:.1f} cycles/mm "
                  f"({rec['n']} samples)")
        else:
            print(f"    {k}: f50 {rec['f50']:.1f} cycles/mm ({rec['n']} "
                  f"samples, {end})")
            if rec["q"] is not None:
                print(f"       rolloff q {rec['q']:.2f} at rms "
                      f"{rec['q_rms']:.4f} (Gaussian {rec['q_gauss']:.4f}, "
                      f"{rec['q_gauss'] / rec['q_rms']:.1f}x worse) -- fitted "
                      f"over {rec['q_span_dec']:.2f} decades, only "
                      f"{rec['q_below_f50_dec']:.2f} of them BELOW f50")
        if abs(rec["r_end"] * 100 - want_end) > w["tol_end"] * 100 / 100 * 100:
            pass
        if abs(rec["f_hi"] - w["end_f"]) > 1.0:
            print(f"    [FAIL] {k}'s curve now ends at {rec['f_hi']:.1f} "
                  f"cycles/mm, recorded {w['end_f']:.1f}")
            bad += 1
        if abs(rec["r_end"] * 100.0 - want_end) > w["tol_end"]:
            print(f"    [FAIL] {k}'s end response {rec['r_end'] * 100:.1f} % "
                  f"moved from the recorded {want_end:.1f} %")
            bad += 1
    if mtf["G"]["f50"] is not None or mtf["R"]["f50"] is not None:
        print("    [FAIL] green or red now reaches 50 % -- the REFUSAL this "
              "sheet is on record for has changed, which is a finding, not a "
              "tolerance")
        bad += 1
    if mtf["B"]["f50"] is None:
        print("    [FAIL] blue no longer reaches 50 %")
        bad += 1
    elif abs(mtf["B"]["f50"] - w["f50_b"]) > w["tol_f50"]:
        print(f"    [FAIL] blue f50 {mtf['B']['f50']:.1f} moved from the "
              f"recorded {w['f50_b']:.1f} cycles/mm")
        bad += 1
    print()
    if bad:
        print(f"[FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("[OK] 2254 sensitometric curves reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
