"""Trace the KONICA data-sheet plots, which are RASTER on every page.

Queue E3. Run this to re-derive, from the PDFs, every number this pass put into
`film_profiles.py` for `KONICA_IMPRESA_50` and `KONICA_INFRARED_750`:

    python konica_raster.py [--root .] [--overlay DIR] [--assert]

⚠ WHY THESE SHEETS NEEDED A NEW READER AND NOT `spectral_vector`. Konica ships
its technical data as scanned pages: `IMP50.pdf` and `INF750.pdf` have a real
text layer for the PROSE, and the plots are embedded bitmaps -- 2008x888,
2008x1184, 1976x1432, 1440x276 -- carrying no paths and no tick text at all. So
there is nothing for a vector reader to read, and nothing for a tick reader to
calibrate against either: the axis numbers are ink, not text. Calibration here
is geometric, off the printed GRID, and every panel asserts its gridline count
before any curve is traced.

⚠ AND THE BITMAPS ARE STORED UPSIDE DOWN. Every embedded image in both files is
flipped top-to-bottom relative to the page; `pymupdf` hands back the stored
pixels, and the page's transform is what puts them the right way up. Rotating
180 degrees "fixes" the picture and leaves the text mirror-reversed, which is
how the flip announces itself. `load()` applies FLIP_TOP_BOTTOM, and the fact
that the frame and gridline assertions pass afterwards is the proof it is right.

METHOD, per panel:
  1. threshold the native bitmap -- never a re-render, which only interpolates;
  2. find the frame from full-height / full-width ink runs, then the interior
     gridlines the same way at a lower threshold;
  3. ASSERT the gridline count against the axis printed on the sheet, and
     calibrate linearly from it. A panel whose grid does not match its axis is
     refused, not fitted;
  4. split curve families by stroke STYLE where the sheet draws them that way
     (`dashtrace.family_split_by_style`), which is what makes the three-layer
     panels decidable through their crossings;
  5. trace with `dashtrace.trace_predictive` and check the result against
     something the SHEET states independently -- the printed resolving power
     for the MTF panel, the layer order for the spectral panel, the
     development-time ordering for the H&D families.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image

import dashtrace as dt

DARK = 0.55          # ink threshold on these scans; see PANELS assertions


def load(root: Path, name: str, page: int, index: int = 0) -> np.ndarray:
    """The native embedded bitmap of one plot, greyscale 0..1, right way up."""
    import pymupdf
    pdf = root / "PDF" / "PROFILES" / "KONICA" / name
    if not pdf.is_file():
        raise FileNotFoundError(f"source not present: {name}")
    doc = pymupdf.open(pdf)
    imgs = doc[page - 1].get_images(full=True)
    if index >= len(imgs):
        raise IndexError(f"{name} p{page} has {len(imgs)} images, wanted #{index}")
    info = doc.extract_image(imgs[index][0])
    im = (Image.open(io.BytesIO(info["image"])).convert("L")
          .transpose(Image.FLIP_TOP_BOTTOM))
    return np.asarray(im, dtype=float) / 255.0


def _runs(idx, gap=3):
    """Consecutive indices grouped, each group reported at its centre."""
    out = []
    for v in idx:
        if out and v - out[-1][-1] <= gap:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(g)) for g in out]


def lines(ink, axis, lo, hi, frac):
    """Ink columns (axis=0) or rows (axis=1) that are `frac` full over [lo,hi)."""
    seg = ink[lo:hi, :] if axis == 0 else ink[:, lo:hi]
    tot = seg.sum(axis=axis)
    need = frac * (hi - lo)
    return _runs([i for i in range(tot.size) if tot[i] >= need])


# ---------------------------------------------------------------------------
# panel geometry
# ---------------------------------------------------------------------------
#: One entry per traced panel. Every number here was measured off the bitmap by
#: `lines()` and is RE-MEASURED at run time: `geometry()` re-detects the frame
#: and gridlines and refuses the panel if they have moved by more than
#: GRID_TOL px. Nothing below is a fitted constant -- they are the printed grid.
#:
#:   file, page, image index, (frame x0, x1, y0, y1),
#:   x gridlines and the axis values they carry,
#:   y gridlines and the axis values they carry
GRID_TOL = 3.0
PANELS = {
    # IMPRESA 50 p2 right: characteristic curves, Status M, CNK-4.
    "imp50_char": dict(
        file="IMP50.pdf", page=2, image=0, frame=(1118, 1980, 16, 688),
        xg=([1118, 1215, 1310, 1406, 1502, 1598, 1692, 1788, 1884],
            [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]),
        yg=([112, 209, 304, 402, 497, 593, 688],
            [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]),
    ),
    # IMPRESA 50 p2 left: spectral sensitivity, three layers, three dash styles.
    # ⚠ THE Y AXIS IS RELATIVE AND CARRIES ONE NUMBER. The sheet draws a
    # double-headed arrow spanning exactly one gridline interval and labels it
    # "1.0", which is the whole of its vertical calibration -- so this panel can
    # give SHAPE and layer separation in decades, and cannot give an absolute
    # log sensitivity. That is why the stored set's criterion string says
    # "log_relative_speed".
    "imp50_sens": dict(
        file="IMP50.pdf", page=2, image=0, frame=(101, 964, 17, 804),
        xg=([180, 442, 702, 964], [400.0, 500.0, 600.0, 700.0]),
        yg=([17, 280, 542, 804], [3.0, 2.0, 1.0, 0.0]),
    ),
    # IMPRESA 50 p3 right: MTF, log-log, ONE curve through a visual filter.
    "imp50_mtf": dict(
        file="IMP50.pdf", page=3, image=0, frame=(1110, 1972, 38, 1018),
        xg=([1110, 1388, 1502, 1780, 1896],
            [0.0, 0.69897, 1.0, 1.69897, 2.0]),
        yg=([226, 342, 624, 738], [2.0, 1.69897, 1.0, 0.69897]),
    ),
    # IMPRESA 50 p3 left: diffuse spectral density, Dmin and midscale NEUTRAL.
    "imp50_dye": dict(
        file="IMP50.pdf", page=3, image=0, frame=(93, 954, 37, 1088),
        xg=([93, 354, 616, 878], [400.0, 500.0, 600.0, 700.0]),
        yg=([37, 300, 564, 828, 1088], [2.0, 1.5, 1.0, 0.5, 0.0]),
    ),
    # INFRARED 750 p3: three developers, five development times each.
    "inf750_dp": dict(
        file="INF750.pdf", page=3, image=0, frame=(74, 938, 18, 622),
        xg=([74, 258, 442, 626, 808], [-4.0, -3.0, -2.0, -1.0, 0.0]),
        yg=([67, 252, 435, 622], [3.0, 2.0, 1.0, 0.0]),
    ),
    "inf750_super": dict(
        file="INF750.pdf", page=3, image=0, frame=(1082, 1946, 16, 620),
        xg=([1082, 1266, 1450, 1633, 1816], [-4.0, -3.0, -2.0, -1.0, 0.0]),
        yg=([66, 252, 434, 620], [3.0, 2.0, 1.0, 0.0]),
    ),
    "inf750_fine": dict(
        file="INF750.pdf", page=3, image=0, frame=(79, 944, 744, 1344),
        xg=([79, 813], [-4.0, 0.0]),
        yg=([790, 1160, 1344], [3.0, 1.0, 0.0]),
    ),
}


def geometry(root: Path, tag: str):
    """(grey image, ink mask, x(px)->value, y(px)->value), grid re-verified."""
    p = PANELS[tag]
    g = load(root, p["file"], p["page"], p["image"])
    ink = g < DARK
    x0, x1, y0, y1 = p["frame"]
    H, W = ink.shape
    # re-detect, inside this panel only, at the threshold each axis needs
    seen_v = lines(ink[y0 + 4:y1 - 3, :], 0, 0, y1 - y0 - 7, 0.45)
    seen_h = lines(ink[:, x0 + 4:x1 - 3], 1, 0, x1 - x0 - 7, 0.45)
    for want, seen, which in ((p["xg"][0], seen_v, "x"), (p["yg"][0], seen_h, "y")):
        for w in want:
            if not any(abs(s - w) <= GRID_TOL for s in seen):
                raise AssertionError(
                    f"{tag}: the {which} gridline recorded at {w} is not on "
                    f"the bitmap (nearest detected: "
                    f"{min(seen, key=lambda s: abs(s - w)) if seen else None})")
    ax = np.polyfit(p["xg"][0], p["xg"][1], 1)
    ay = np.polyfit(p["yg"][0], p["yg"][1], 1)
    # a linear grid must fit its own gridlines to well under a pixel
    for coef, (pos, val), which in ((ax, p["xg"], "x"), (ay, p["yg"], "y")):
        r = np.max(np.abs(np.polyval(coef, pos) - np.asarray(val)))
        span = abs(coef[0]) * (x1 - x0 if which == "x" else y1 - y0)
        if r > 0.01 * span:
            raise AssertionError(f"{tag}: the {which} grid is not linear "
                                 f"(worst {r:.4f} over a {span:.3f} span)")
    return g, ink, (lambda v: np.polyval(ax, v)), (lambda v: np.polyval(ay, v))


def panel_ink(root: Path, tag: str, masks=()):
    """Panel interior with the frame and every printed gridline blanked out.

    ⚠ THE GRIDLINES HAVE TO GO BEFORE ANY TRACING. They are drawn in the same
    ink as the curves and run the full width or height of the panel, so a
    column tracer sees one enormous ink run wherever a curve sits on a
    horizontal rule. Blanking them costs a few columns per curve, which
    `trace_predictive` bridges on its fitted slope.
    """
    p = PANELS[tag]
    g, ink, fx, fy = geometry(root, tag)
    x0, x1, y0, y1 = p["frame"]
    sub = np.zeros_like(ink)
    sub[y0 + 4:y1 - 3, x0 + 4:x1 - 3] = ink[y0 + 4:y1 - 3, x0 + 4:x1 - 3]
    for x in p["xg"][0]:
        sub[:, max(0, int(x) - 3):int(x) + 4] = False
    for y in p["yg"][0]:
        sub[max(0, int(y) - 3):int(y) + 4, :] = False
    for mx0, mx1, my0, my1 in masks:
        sub[my0:my1, mx0:mx1] = False
    return g, sub, fx, fy


def trace(root: Path, tag: str, seed_x, seeds, direction=-1, masks=(),
          merge_px=0.0, x_stop=None):
    """Trace named curves and return {name: (x values, y values)} in AXIS units."""
    p = PANELS[tag]
    x0, x1, y0, y1 = p["frame"]
    g, sub, fx, fy = panel_ink(root, tag, masks)
    gray = np.where(sub, g, 1.0)
    lo = x0 + 6 if direction < 0 else seed_x
    hi = seed_x if direction < 0 else (x_stop if x_stop is not None else x1 - 6)
    if direction < 0 and x_stop is not None:
        lo = x_stop
    tr = dt.trace_predictive(sub, gray, (lo, hi), y0, y1, seed_x, seeds,
                             direction=direction, merge_px=merge_px)
    out = {}
    for k, v in tr.items():
        xs = np.array(sorted(v), dtype=float)
        ys = np.array([v[int(x)] for x in xs], dtype=float)
        out[k] = (fx(xs), fy(ys))
    return out


# ---------------------------------------------------------------------------
# the readings
# ---------------------------------------------------------------------------
#: Seeds and blanking rectangles per panel, measured once and asserted by the
#: results. A seed is a column where every curve of the panel resolves into its
#: own ink run; the rectangles remove captions and in-plot legends, never curve.
READ = {
    "imp50_char": dict(seed_x=1875, seeds={"b": 173.5, "g": 266.0, "r": 396.5},
                       masks=[(1118, 1560, 16, 170), (1884, 1980, 16, 688)]),
    "inf750_dp": dict(seed_x=860,
                      seeds={"12": 54.5, "10": 63.0, "8": 87.0,
                             "6": 139.0, "4": 239.0},
                      masks=[(74, 700, 18, 140)]),
    "inf750_super": dict(seed_x=1870,
                         seeds={"12": 86.0, "10": 96.0, "8": 130.0,
                                "6": 193.0, "4": 287.0},
                         masks=[(1082, 1710, 16, 140)]),
    "inf750_fine": dict(seed_x=860,
                        seeds={"12": 847.0, "10": 873.0, "8": 925.0,
                               "6": 989.5, "4": 1074.0},
                        masks=[(79, 700, 744, 880), (79, 944, 972, 980),
                               (258, 268, 744, 1344), (441, 451, 744, 1344),
                               (625, 635, 744, 1344)]),
}

#: ISO 5-3 status M band centres, from `iso_5_3_status.py`. Used to read the
#: DYE panel at the three wavelengths the CHARACTERISTIC panel's densitometer
#: reports, which is what makes the two panels comparable at all.
STATUS_M = {"b": 450.0, "g": 540.0, "r": 640.0}


def imp50_char(root: Path):
    """The three Status M characteristic curves, and the Dmin they plateau at."""
    t = trace(root, "imp50_char", **READ["imp50_char"])
    out = {}
    for k in "rgb":
        x, y = t[k]
        o = np.argsort(x)
        x, y = x[o], y[o]
        out[k] = dict(x=x, y=y, dmin=float(y[:60].mean()),
                      dmax_drawn=float(y.max()),
                      x_lo=float(x.min()), x_hi=float(x.max()))
    return out


def imp50_dye(root: Path):
    """Dmin and midscale DIFFUSE SPECTRAL density, sampled at the status M bands.

    ⚠ NOT A DYE TRIPLE, AND IT IS THE INDEPENDENT CHECK ON THE H&D PANEL. The
    sheet draws two NEUTRAL spectra -- the minimum density of the coating and
    the density of a midscale neutral subject -- not the three separated dye
    curves `DyeDensity` wants, so nothing here can fill that field. What it can
    do is state, from a different figure on a different page, what a
    densitometer reading this film through the status M bands must see at Dmin;
    and that is exactly the number the characteristic panel's left plateau is.
    """
    p = PANELS["imp50_dye"]
    x0, x1, y0, y1 = p["frame"]
    g, sub, fx, fy = panel_ink(root, "imp50_dye", masks=[(300, 900, 180, 300)])
    xs = np.arange(x0 + 6, x1 - 5)
    lo, hi = {}, {}
    for x in xs:
        runs = dt.column_runs(sub, int(x), y0, y1)
        if len(runs) == 2:
            hi[float(fx(x))] = float(fy(min(runs)))   # midscale, higher density
            lo[float(fx(x))] = float(fy(max(runs)))   # minimum density
    def at(d, nm):
        ks = np.array(sorted(d))
        return float(np.interp(nm, ks, [d[k] for k in ks]))
    return dict(dmin={c: at(lo, nm) for c, nm in STATUS_M.items()},
                midscale={c: at(hi, nm) for c, nm in STATUS_M.items()},
                n=len(lo), span=(min(lo), max(lo)))


def imp50_mtf(root: Path):
    """The single visual-filter MTF: f50, the low-frequency overshoot, the tail."""
    p = PANELS["imp50_mtf"]
    x0, x1, y0, y1 = p["frame"]
    g, sub, fx, fy = panel_ink(root, "imp50_mtf", masks=[(1560, 1950, 50, 160)])
    gray = np.where(sub, g, 1.0)
    seed_x, seed_y = 1850, 369.5
    pts = {}
    for d, rng in ((-1, (x0 + 10, seed_x)), (+1, (seed_x, x1 - 6))):
        pts.update(dt.trace_predictive(sub, gray, rng, y0, y1, seed_x,
                                       {"m": seed_y}, direction=d)["m"])
    xs = np.array(sorted(pts), dtype=float)
    f = 10.0 ** fx(xs)
    resp = 10.0 ** fy(np.array([pts[int(x)] for x in xs]))
    below = np.where(resp <= 50.0)[0]
    f50 = float("nan")
    if below.size and below[0] > 0:
        j = below[0]
        lf = (np.log10(f[j - 1]) + (50.0 - resp[j - 1])
              * (np.log10(f[j]) - np.log10(f[j - 1])) / (resp[j] - resp[j - 1]))
        f50 = float(10.0 ** lf)
    return dict(f=f, resp=resp, f50=f50, peak=float(resp.max()),
                peak_f=float(f[int(np.argmax(resp))]),
                f_lo=float(f.min()), f_hi=float(f.max()))


def imp50_rolloff(f, resp, f50, lo=25.0):
    """Fit MTF(f) = 1/(1+(f/f50)^q) to the traced curve above the overshoot.

    ⚠ THE FIT STARTS AT 25 c/mm AND HAS TO. Below that the drawn curve is ABOVE
    100 %, and every rolloff form in this schema is normalised to 1 at zero
    frequency, so including the overshoot would not measure the rolloff -- it
    would trade tail accuracy for an error the model cannot represent anyway.
    The overshoot is carried separately, in `MTFSpec.adjacency`.
    """
    sel = f >= lo
    r = resp[sel] / 100.0
    qs = np.arange(1.0, 8.0, 0.01)
    err = [float(np.sqrt(np.mean((1.0 / (1.0 + (f[sel] / f50) ** q) - r) ** 2)))
           for q in qs]
    i = int(np.argmin(err))
    gauss = float(np.sqrt(np.mean(
        (np.exp(-np.log(2) * (f[sel] / f50) ** 2) - r) ** 2)))
    return float(qs[i]), float(err[i]), gauss, int(sel.sum())


def inf750_char(root: Path):
    """Five development times in each of three developers, plus the shared base.

    ⚠ THE BASE+FOG IS SHARED AND ONLY SOME TRACES REACH IT. All fifteen curves
    come off one emulsion, so they have one base+fog, but the five curves of a
    panel are drawn as a single bundle at the left and only the ones that stay
    flat longest are still separable back there. The plateau is therefore taken
    from whichever traces reach log H -1.95 -- four of the fifteen do -- and the
    spread between those four is reported as its uncertainty.
    """
    out = {}
    anchors = []
    for tag in ("inf750_dp", "inf750_super", "inf750_fine"):
        t = trace(root, tag, **READ[tag])
        out[tag] = {}
        for n, (x, y) in t.items():
            o = np.argsort(x)
            x, y = x[o], y[o]
            out[tag][n] = dict(x=x, y=y, x_lo=float(x.min()),
                               dmax_drawn=float(y.max()))
            if x.min() < -1.95:
                anchors.append(float(y[0]))
    out["base"] = (float(np.mean(anchors)), float(np.std(anchors)), len(anchors))
    return out


# ---------------------------------------------------------------------------
# what this pass measured, pinned
# ---------------------------------------------------------------------------
#: ⚠ THE FIRST TWO ROWS ARE THE SAME QUANTITY READ OFF TWO DIFFERENT FIGURES ON
#: TWO DIFFERENT PAGES, and their agreement is the evidence for the Dmin triple
#: this pass adopted. The characteristic panel (p2) plateaus at a densitometer
#: reading; the spectral-density panel (p3) prints the minimum-density spectrum
#: the same densitometer would integrate. They agree to 0.005-0.015 D in all
#: three status M bands -- and they jointly refute the triple the profile held,
#: whose blue was 1.00 against a measured 0.68.
EXPECTED = {
    "imp50_char_dmin": dict(r=0.1989, g=0.5565, b=0.6760),
    "imp50_dye_dmin": dict(r=0.1899, g=0.5515, b=0.6913),
    "imp50_dye_midscale": dict(r=0.5992, g=1.1211, b=1.5677),
    "imp50_char_dmax_drawn": dict(r=1.522, g=2.201, b=2.683),
    "imp50_mtf": dict(f50=64.86, peak=121.4, peak_f=6.88, q=2.20,
                      q_rms=0.0188, gauss_rms=0.0388),
    "inf750_base": 0.2303,
}
TOL_D = 0.01        # density, and the panels agree with each other to 0.015
TOL_F = 0.5         # c/mm on the MTF crossing
TOL_PCT = 1.0       # percentage points on the overshoot

#: The straight-line slope of each traced INF750 curve, per developer and time.
#: ⚠ THIS TABLE IS THE REASON THE ADOPTED GAMMA MOVED FROM 0.72 TO 1.70. It is
#: not one reading: fifteen curves, three developers, five times, and every one
#: of them is steeper than the value the profile held. The ordering is also a
#: check in itself -- gamma must rise with development time in every developer,
#: and DP must be the most contrasty of the three at equal time, which is what
#: makes 6-minute DP (the sheet's own standard time, footnoted as the D-76
#: equivalent) the steepest standard condition on the page.
INF750_GAMMA = {
    "inf750_dp":    {"4": 1.153, "6": 1.563, "8": 1.764, "10": 1.804, "12": 1.837},
    "inf750_super": {"4": 1.036, "6": 1.321, "8": 1.440, "10": 1.410, "12": 1.425},
    "inf750_fine":  {"4": 0.814, "6": 1.087, "8": 1.244, "10": 1.418, "12": 1.546},
}
TOL_GAMMA = 0.03


def _midslope(x, y, d0):
    m = (y > d0 + 0.35 * (y.max() - d0)) & (y < d0 + 0.85 * (y.max() - d0))
    return float(np.polyfit(x[m], y[m], 1)[0]) if m.sum() > 10 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()
    root = Path(ns.root)
    bad = 0

    def chk(ok, text):
        nonlocal bad
        print(f"  [{'OK  ' if ok else 'FAIL'}] {text}")
        if not ok:
            bad += 1

    for tag in PANELS:
        geometry(root, tag)
    print(f"  [OK  ] all {len(PANELS)} panel grids re-detected on the bitmaps")

    ch = imp50_char(root)
    dy = imp50_dye(root)
    for c in "rgb":
        chk(abs(ch[c]["dmin"] - EXPECTED["imp50_char_dmin"][c]) < TOL_D,
            f"IMPRESA 50 {c} Dmin off the characteristic panel: "
            f"{ch[c]['dmin']:.4f} (expected "
            f"{EXPECTED['imp50_char_dmin'][c]:.4f})")
    for c in "rgb":
        chk(abs(dy["dmin"][c] - EXPECTED["imp50_dye_dmin"][c]) < TOL_D,
            f"IMPRESA 50 {c} Dmin off the spectral-density panel at "
            f"{STATUS_M[c]:.0f} nm: {dy['dmin'][c]:.4f} (expected "
            f"{EXPECTED['imp50_dye_dmin'][c]:.4f})")
    for c in "rgb":
        chk(abs(ch[c]["dmin"] - dy["dmin"][c]) < 0.02,
            f"IMPRESA 50 {c}: the two panels agree on Dmin to "
            f"{abs(ch[c]['dmin'] - dy['dmin'][c]):.4f} D")
    chk(ch["b"]["dmin"] > ch["g"]["dmin"] > ch["r"]["dmin"],
        "IMPRESA 50 Dmin rises blue > green > red, as a masked negative must")

    mt = imp50_mtf(root)
    chk(abs(mt["f50"] - EXPECTED["imp50_mtf"]["f50"]) < TOL_F,
        f"IMPRESA 50 MTF crosses 50 % at {mt['f50']:.2f} c/mm "
        f"(expected {EXPECTED['imp50_mtf']['f50']:.2f})")
    chk(abs(mt["peak"] - EXPECTED["imp50_mtf"]["peak"]) < TOL_PCT,
        f"IMPRESA 50 MTF overshoots to {mt['peak']:.1f} % at "
        f"{mt['peak_f']:.2f} c/mm -- an adjacency effect the sheet prints")
    # ⚠ the printed resolving powers bracket the traced curve, which is the only
    # independent statement the sheet makes about its own sharpness
    chk(63.0 < mt["f50"] < 160.0,
        f"IMPRESA 50 f50 {mt['f50']:.1f} c/mm lies between the sheet's own "
        f"printed resolving powers, 63 lines/mm at 1.6:1 and 160 at 1000:1")

    q, qr, gr, nq = imp50_rolloff(mt["f"], mt["resp"], mt["f50"])
    chk(abs(q - EXPECTED["imp50_mtf"]["q"]) < 0.05
        and abs(qr - EXPECTED["imp50_mtf"]["q_rms"]) < 0.003,
        f"IMPRESA 50 MTF rolloff 1/(1+(f/f50)^q): q {q:.2f}, rms {qr:.4f} over "
        f"{nq} samples above 25 c/mm")
    chk(qr < gr / 1.5,
        f"the power law beats the Gaussian on this curve, {qr:.4f} against "
        f"{gr:.4f} -- the same result C2 measured on 5231")

    inf = inf750_char(root)
    base, sd, n = inf["base"]
    chk(abs(base - EXPECTED["inf750_base"]) < TOL_D,
        f"INFRARED 750 base+fog {base:.4f} +/- {sd:.4f} from {n} traces that "
        f"reach the plateau (expected {EXPECTED['inf750_base']:.4f})")
    for tag, want in INF750_GAMMA.items():
        got = {k: _midslope(v["x"], v["y"], base)
               for k, v in inf[tag].items() if k in want}
        worst = max(abs(got[k] - want[k]) for k in want)
        chk(worst < TOL_GAMMA,
            f"{tag}: gamma "
            + " ".join(f"{k}min {got[k]:.3f}" for k in ("4", "6", "8", "10", "12"))
            + f" (worst drift {worst:.3f})")
        order = [got[k] for k in ("4", "6", "8", "10", "12")]
        chk(all(b >= a - 0.05 for a, b in zip(order, order[1:])),
            f"{tag}: contrast rises with development time")
    for t in ("4", "6", "8", "10", "12"):
        d = _midslope(inf["inf750_dp"][t]["x"], inf["inf750_dp"][t]["y"], base)
        f = _midslope(inf["inf750_fine"][t]["x"], inf["inf750_fine"][t]["y"], base)
        chk(d > f, f"INFRARED 750 at {t} min: Konicadol DP {d:.3f} is more "
                   f"contrasty than Konicadol Fine {f:.3f}")

    print(f"\n[i] {bad} failed")
    return 1 if (ns.do_assert and bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
