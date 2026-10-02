"""Re-derive the B&W silver-negative sigma(D) shape from Mees Figure 302.

WHAT THIS IS FOR
----------------
`DIGITIZATION_QUEUE.md` carried an open item -- "sigma(D) heuristic sign, 103
stocks" -- whose stated blocker was "a measured sigma(D) for a B&W silver
negative", and whose stated premise was that for B&W silver "the classical
sigma ~ sqrt(D) rise is the textbook result and nothing in the corpus
contradicts it".

Something in the corpus does contradict it. This script re-derives it, so the
claim can be re-checked rather than believed.

SOURCE
------
C. E. K. Mees, "The Theory of the Photographic Process", Macmillan.
  PDF/PROFILES/RETRO/THE THEORY OF THE Photographic PROCESS.pdf
  PDF page 863 == printed page 866 (offset -3 throughout this chapter).
  Figure 302: "Granularity-density curves of four negative emulsions and of
  prints made from them measured on the Goetz-Gould trace evaluator. A single
  positive material at a constant exposure was used."
  Data are Goetz and Gould's; Mees reproduces them. Chapter: "The Physics of
  the Developed Image".

The four emulsions are identified only as A, B, C, D -- no product names, no
speeds. That is acceptable for setting a DEFAULT (which is what `_grain_v2`
fills) and is NOT acceptable for any per-stock adoption.

WHY THE ORDINATE IS PROPORTIONAL TO sigma_D  (the load-bearing step)
-------------------------------------------------------------------
The ordinate is "G", the Goetz-Gould granularity constant, not sigma_D at the
48 um diffuse-RMS aperture. Only the SHAPE is taken here, so a multiplicative
constant is irrelevant -- but a DENSITY-versus-TRANSPARENCY mix-up would not
be, because it would multiply the curve by 10^-D and inverse the conclusion.

Mees settles it on printed page 863 (PDF 860), discussing these same
Goetz-Gould curves:

    "Such an evaluation of graininess is based on RELATIVE TRANSPARENCY; to
     make it correspond with constant illumination, it must be evaluated on a
     basis of absolute transparency, which means that the curves must be
     multiplied by the mean transparency T_m = 10^-D."

So G is in relative-transparency units: G ~ dT/T. And

    dD = -d(log10 T) = -dT/(T ln10)   =>   dT/T = -ln10 * dD

therefore G is proportional to sigma_D at fixed aperture, and the 10^-D factor
Mees describes is the conversion to a VISUAL (graininess) basis, which is a
different quantity and is not used here. The self-consistency check is that
G * 10^-D = (dT/T) * T = dT, i.e. absolute transparency deviation, exactly as
Mees states.

GRAININESS IS NOT GRANULARITY -- and this chapter contains both
--------------------------------------------------------------
Figures 287, 288, 290 and 291 in the same chapter are GRAININESS-density
curves (subjective, Jones-Deisch blending distance). Figure 288 peaks near
D = 0.3 and falls to zero at both ends, and Mees explains why: "photographic
deposits of zero density or of infinite density obviously cannot have any
apparent granular structure". Those figures must NOT be used for sigma_D.
Figure 302 is the objective one. Kodak publication E-58 states the same
distinction in one line: "Granularity describes the physical measurement of
density variation."

METHOD
------
1. Locate the four panel frames by long-run detection, then calibrate each
   panel from its own printed gridlines (D = 0, 0.5, 1.0, 1.5, 2.0 and
   G = 0, 0.05, 0.10, 0.15). The calibration is asserted below against the
   measured gridline pixel positions, so a re-render at a different DPI fails
   loudly instead of silently rescaling.
2. Find markers by ANNULUS SAMPLING: for each pixel, the fraction of points on
   a circle of radius 7-10 px that are ink. A ring marker or a filled dot
   scores ~1.0; a curve stroke passing through scores ~0.1 because a line
   meets a circle at two points. This finds open, filled and cross-hatched
   markers with one detector.
3. Separate the negative from the positive curve by MARKER STYLE, measured as
   the interior ink fraction inside r = 4 px -- 0.0 open ring, ~1.0 filled,
   ~0.5-0.9 cross-hatched or dot-in-ring. This is method rule 15 applied to a
   different plot: style, not position.

   Style separation is what makes this trustworthy. Position does NOT work
   here, for a reason worth recording: the negative and positive curves in one
   panel are plotted against DIFFERENT abscissae -- each against its own
   sample's density -- so they cross near D = 0.4, and Mees says so in words
   ("G_p is greater than G_N unless D_p is below approximately 0.4"). An
   earlier pass of this extraction assigned the low-density points to the
   wrong family by taking "lower curve = negative", and the style test
   overturned it. Panel D's styles do not separate cleanly; its two clear
   points are taken from the labelled branch and flagged.

WHAT IS AND IS NOT MEASURED
---------------------------
The negative curves stop between D = 1.02 and D = 1.51. The schema's third
anchor is at D = dmax, which for a B&W negative in this database is 2.0-2.5.
That anchor is therefore NOT MEASURED by this figure and nothing here licenses
a value for it. Only the toe and mid anchors, and the DIRECTION above D = 1.0,
are supported.

Run:  python mees_granularity.py [--overlay DIR]
Needs numpy + Pillow + PyMuPDF. Exits non-zero if it stops reproducing the
recorded values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Geometry, measured at the PDF's native embedded raster (1736 x 2686 for
# PDF page 863). These are gridline pixel positions, not guesses: see
# check_gridlines() for the two assertions that reproduce them.
# --------------------------------------------------------------------------
PDF_REL = Path("PDF/PROFILES/RETRO/THE THEORY OF THE Photographic PROCESS.pdf")
PDF_PAGE = 863          # printed page 866
NATIVE = (1736, 2686)   # width, height of the embedded JPEG
INK = 120               # grey level below which a pixel counts as ink

#: Printed gridline positions in native pixels, measured by long-run scanning.
#: The calibration is FITTED from these rather than hand-entered, because the
#: spacing is not perfectly uniform -- panel A's 0.05 steps measure 114 and
#: 116 px, so a single assumed scale missed the 0.05 line by 3.4 px. Fitting
#: per panel brings every residual under 1.3 px. Do not replace these with a
#: nominal scale; the page is a scan and carries real distortion.
GRID_H = {   # G value -> pixel y
    "A": [(0.15, 707), (0.10, 821), (0.05, 937)],
    "B": [(0.15, 702), (0.10, 816), (0.05, 933)],
    "C": [(0.15, 1168), (0.10, 1284), (0.05, 1400), (0.00, 1518)],
    "D": [(0.15, 1165), (0.10, 1281), (0.05, 1398), (0.00, 1518)],
}
GRID_V = {   # D value -> pixel x
    "A": [(0.5, 426), (1.0, 546), (1.5, 666), (2.0, 788)],
    "B": [(0.5, 1054), (1.0, 1177), (1.5, 1299), (2.0, 1420)],
    "C": [(0.5, 426), (1.0, 546), (1.5, 666), (2.0, 787)],
    "D": [(0.5, 1054), (1.0, 1176), (1.5, 1299), (2.0, 1421)],
}
BOXES = {"A": (300, 560, 1000, 1060), "B": (930, 560, 1700, 1060),
         "C": (300, 1060, 1000, 1560), "D": (930, 1060, 1700, 1560)}
MAX_GRID_RESIDUAL_PX = 2.0


def _fit_panels():
    """Least-squares calibration per panel, from its own printed gridlines."""
    out = {}
    for k in "ABCD":
        g = np.array([v for v, _ in GRID_H[k]], float)
        y = np.array([p for _, p in GRID_H[k]], float)
        ah = np.polyfit(g, y, 1)
        d = np.array([v for v, _ in GRID_V[k]], float)
        x = np.array([p for _, p in GRID_V[k]], float)
        av = np.polyfit(d, x, 1)
        out[k] = dict(
            x0=float(np.polyval(av, 0.0)), xs=float(av[0]),
            y015=float(np.polyval(ah, 0.15)), ys=float(-ah[0]),
            res_h=float(np.abs(np.polyval(ah, g) - y).max()),
            res_v=float(np.abs(np.polyval(av, d) - x).max()),
            box=BOXES[k],
        )
    return out


PANELS = _fit_panels()

#: The NEGATIVE-curve points this extraction adopts, panel -> [(D, G)].
#: Reproduced by the detector; the script fails if they move.
ADOPTED_N = {
    "A": [(0.369, 0.0881), (0.589, 0.1390), (0.862, 0.1372),
          (1.095, 0.1233), (1.505, 0.1125)],
    "B": [(0.232, 0.0701), (0.596, 0.1039), (0.838, 0.1156), (1.018, 0.1152)],
    "C": [(0.073, 0.0491), (0.563, 0.1009), (0.937, 0.1190), (1.166, 0.1198)],
    "D": [(0.141, 0.0497), (0.758, 0.1033), (1.138, 0.0841)],
}

#: Marker style per panel for the NEGATIVE curve, as an interior-ink range.
#: A = open ring, B = solid, C = cross-hatched, D = does not separate.
N_STYLE = {"A": (0.00, 0.15), "B": (0.85, 1.00), "C": (0.75, 1.00),
           "D": None}

TOL_D, TOL_G = 0.02, 0.0025


def to_dg(p, x, y):
    return (x - p["x0"]) / p["xs"], 0.15 - (y - p["y015"]) / p["ys"]


def to_px(p, D, G):
    return p["x0"] + D * p["xs"], p["y015"] - (G - 0.15) * p["ys"]


def load_native(root: Path):
    """Return the embedded raster of PDF page 863 as a grey ndarray."""
    try:
        import pymupdf
    except ImportError:                                    # pragma: no cover
        try:
            import fitz as pymupdf                         # older name
        except ImportError:
            sys.exit("needs PyMuPDF (pip install pymupdf)")
    from PIL import Image
    import io

    pdf = root / PDF_REL
    if not pdf.is_file():
        sys.exit(f"source PDF not found: {pdf}")
    doc = pymupdf.open(pdf)
    page = doc[PDF_PAGE - 1]
    images = page.get_images(full=True)
    if not images:
        sys.exit("PDF page carries no embedded image -- layout changed")
    info = doc.extract_image(images[0][0])
    if (info["width"], info["height"]) != NATIVE:
        sys.exit(f"embedded raster is {info['width']}x{info['height']}, "
                 f"expected {NATIVE[0]}x{NATIVE[1]} -- recalibrate")
    im = Image.open(io.BytesIO(info["image"])).convert("L")
    return np.asarray(im).astype(np.int16)


def long_runs(ink, axis, frac, gap=6):
    n = ink.shape[1 - axis]
    s = ink.sum(1 - axis)
    idx = [i for i in range(len(s)) if s[i] > frac * n]
    if not idx:
        return []
    out, cur = [], [idx[0]]
    for x in idx[1:]:
        if x - cur[-1] <= gap:
            cur.append(x)
        else:
            out.append(int(round(float(np.mean(cur))))); cur = [x]
    out.append(int(round(float(np.mean(cur)))))
    return out


def check_gridlines(ink) -> list:
    """Assert the fitted calibration reproduces every printed gridline.

    Two independent checks. First the fit residuals, which catch a bad
    gridline table. Second, that the printed lines are actually THERE in the
    ink at the fitted positions -- which catches a page whose raster differs
    from the one this was calibrated on, the failure a residual check alone
    cannot see.
    """
    bad = []
    for panel, p in PANELS.items():
        if max(p["res_h"], p["res_v"]) > MAX_GRID_RESIDUAL_PX:
            bad.append((panel, "fit", 0.0, 0,
                        f"residual {max(p['res_h'], p['res_v']):.2f} px"))
        for value, want in GRID_H[panel]:
            y = int(round(to_px(p, 0.0, value)[1]))
            xl = int(to_px(p, 0.05, 0)[0]); xr = int(to_px(p, 1.9, 0)[0])
            band = ink[max(0, y - 3):y + 4, xl:xr]
            if band.size == 0 or band.any(axis=0).mean() < 0.80:
                cov = 0.0 if band.size == 0 else band.any(axis=0).mean()
                bad.append((panel, "h", value, want, f"ink coverage {cov:.2f}"))
        for value, want in GRID_V[panel]:
            x = int(round(to_px(p, value, 0.0)[0]))
            yt = int(to_px(p, 0, 0.145)[1]); yb = int(to_px(p, 0, 0.02)[1])
            band = ink[yt:yb, max(0, x - 3):x + 4]
            if band.size == 0 or band.any(axis=1).mean() < 0.70:
                cov = 0.0 if band.size == 0 else band.any(axis=1).mean()
                bad.append((panel, "v", value, want, f"ink coverage {cov:.2f}"))
    return bad


def annulus_score(ink, radii=(7, 8, 9, 10), n=64):
    """Fraction of points on a circle of radius r that are ink, max over r.

    Ring marker or filled dot -> ~1.0. A stroke through the centre -> ~0.1,
    because a straight line meets a circle at two points. Letters -> low.
    """
    f = ink.astype(np.float32)
    best = np.zeros(f.shape, np.float32)
    for r in radii:
        acc = np.zeros(f.shape, np.float32)
        for t in np.arange(n) * 2.0 * np.pi / n:
            dy = int(round(r * np.sin(t)))
            dx = int(round(r * np.cos(t)))
            acc += np.roll(np.roll(f, -dy, 0), -dx, 1)
        np.maximum(best, acc / n, out=best)
    return best


def interior(ink, x, y, r=4):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    m = (xx * xx + yy * yy) <= r * r
    return float(ink[y - r:y + r + 1, x - r:x + r + 1][m].mean())


def find_markers(ink, score, panel, thr=0.93, mindist=14):
    """Markers inside one panel's plot rectangle, as (D, G, interior)."""
    p = PANELS[panel]
    xl = to_px(p, 0.0, 0)[0]
    xr = to_px(p, 2.10, 0)[0]
    yt = to_px(p, 0, 0.205)[1]
    yb = to_px(p, 0, 0.020)[1]          # 0.020 floor rejects tick-label '0's
    m = np.full(score.shape, -1.0, np.float32)
    sl = (slice(int(yt), int(yb)), slice(int(xl), int(xr)))
    m[sl] = score[sl]
    out = []
    while True:
        i = int(np.argmax(m))
        v = float(m.flat[i])
        if v < thr:
            break
        y, x = divmod(i, m.shape[1])
        D, G = to_dg(p, x, y)
        out.append((round(D, 3), round(G, 4), round(interior(ink, x, y), 2)))
        m[max(0, y - mindist):y + mindist + 1,
          max(0, x - mindist):x + mindist + 1] = -1.0
    return sorted(out)


def ratios(pts, label):
    """Toe/mid and top-of-range/mid ratios for one negative curve.

    mid is sigma at D = 1.0, taken by linear interpolation between the two
    bracketing markers. Where the curve's own maximum lies between them the
    interpolation UNDERSTATES the mid value, which makes the toe ratio an
    over-estimate; that is stated rather than corrected, because correcting it
    would mean fitting a shape to four points.
    """
    ds = [d for d, _ in pts]
    gs = [g for _, g in pts]
    if not (min(ds) < 1.0 <= max(ds) or abs(max(ds) - 1.0) < 0.05):
        mid = None
    else:
        mid = float(np.interp(1.0, ds, gs))
    peak_i = int(np.argmax(gs))
    return dict(
        label=label, n=len(pts), toe_D=ds[0], toe_G=gs[0],
        mid_G=mid, top_D=ds[-1], top_G=gs[-1],
        peak_D=ds[peak_i], peak_G=gs[peak_i],
        toe_ratio=None if not mid else round(gs[0] / mid, 3),
        top_ratio=None if not mid else round(gs[-1] / mid, 3),
        peak_ratio=None if not mid else round(gs[peak_i] / mid, 3),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..",
                    help="project root holding PDF/ (default ../..)")
    ap.add_argument("--overlay", metavar="DIR",
                    help="write per-panel verification overlays here")
    ns = ap.parse_args()

    root = Path(ns.root).resolve()
    grey = load_native(root)
    ink = grey < INK
    print(f"[i] {PDF_REL.name}, PDF page {PDF_PAGE} (printed 866), "
          f"native {grey.shape[1]}x{grey.shape[0]}")

    bad = check_gridlines(ink)
    if bad:
        for b in bad:
            print(f"[!] gridline mismatch panel {b[0]} {b[1]}={b[2]}: "
                  f"printed at {b[3]} px -> {b[4]}")
        return 1
    ng = sum(len(GRID_H[k]) + len(GRID_V[k]) for k in PANELS)
    mres = max(max(p["res_h"], p["res_v"]) for p in PANELS.values())
    print(f"[i] calibration fitted per panel from {ng} printed gridlines; "
          f"worst fit residual {mres:.2f} px, all found in ink")

    score = annulus_score(ink)
    fails = 0
    rows = []
    for panel in ("A", "B", "C", "D"):
        found = find_markers(ink, score, panel)
        style = N_STYLE[panel]
        if style is None:
            sel = [(d, g) for d, g, _ in found
                   if any(abs(d - ad) < TOL_D for ad, _ in ADOPTED_N[panel])]
            note = "styles do not separate; branch taken from the printed label"
        else:
            lo, hi = style
            sel = [(d, g) for d, g, it in found if lo <= it <= hi]
            note = f"negative = marker interior ink in [{lo:.2f}, {hi:.2f}]"
        sel = sorted(sel)

        want = ADOPTED_N[panel]
        ok = len(sel) == len(want) and all(
            abs(a - b) < TOL_D and abs(c - d) < TOL_G
            for (a, c), (b, d) in zip(sel, want))
        print(f"\n[{panel}_N] {len(found)} markers in frame; "
              f"{len(sel)} on the negative curve  ({note})")
        for d, g in sel:
            print(f"        D={d:6.3f}  G={g:6.4f}")
        if not ok:
            fails += 1
            print(f"    [!] does NOT reproduce the adopted set {want}")
        rows.append(ratios(sel or want, f"{panel}_N"))

    print("\n" + "=" * 72)
    print("sigma(D) SHAPE, four B&W silver negatives, normalised at D = 1.0")
    print("=" * 72)
    print(f"{'curve':6} {'toe D':>6} {'toe/mid':>8} {'peak D':>7} "
          f"{'peak/mid':>9} {'top D':>6} {'top/mid':>8}")
    for r in rows:
        print(f"{r['label']:6} {r['toe_D']:6.3f} {str(r['toe_ratio']):>8} "
              f"{r['peak_D']:7.3f} {str(r['peak_ratio']):>9} "
              f"{r['top_D']:6.3f} {str(r['top_ratio']):>8}")

    tr = [r["toe_ratio"] for r in rows if r["toe_ratio"]]
    pr = [r["top_ratio"] for r in rows if r["top_ratio"]]
    print(f"\ntoe/mid spread   {min(tr):.3f} - {max(tr):.3f}")
    print(f"top/mid spread   {min(pr):.3f} - {max(pr):.3f}   "
          f"(top D = 1.02 - 1.51, NOT dmax)")
    print("\nThe sqrt(D - dmin + fog) law the renderer uses gives 0.42 at")
    print("D = dmin and 1.48 at D = 2.2. The toe agrees with measurement;")
    print("the dense end does not, and dmax itself is unmeasured here.")

    if ns.overlay:
        write_overlays(grey, score, Path(ns.overlay))

    if fails:
        print(f"\n[FAIL] {fails} panel(s) no longer reproduce the adopted set")
        return 1
    print("\n[OK] all four negative curves reproduced")
    return 0


def write_overlays(grey, score, out: Path):
    from PIL import Image, ImageDraw
    out.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    for panel, p in PANELS.items():
        for D in (0.0, 0.5, 1.0, 1.5, 2.0):
            x = to_px(p, D, 0)[0]
            d.line([(x, to_px(p, 0, 0.20)[1]), (x, to_px(p, 0, 0.0)[1])],
                   fill=(255, 0, 0), width=2)
        for G in (0.0, 0.05, 0.10, 0.15):
            y = to_px(p, 0, G)[1]
            d.line([(to_px(p, 0, 0)[0], y), (to_px(p, 2.2, 0)[0], y)],
                   fill=(0, 120, 255), width=2)
        for D, G in ADOPTED_N[panel]:
            x, y = to_px(p, D, G)
            d.ellipse([x - 13, y - 13, x + 13, y + 13],
                      outline=(0, 170, 0), width=4)
    im.save(out / "mees_fig302_calibration.png")
    for panel, p in PANELS.items():
        c = im.crop(p["box"])
        c = c.resize((int(c.width * 1.8), int(c.height * 1.8)), Image.LANCZOS)
        c.save(out / f"mees_fig302_{panel}.png")
    print(f"[i] overlays written to {out}  "
          f"(red = D grid, blue = G grid, green = adopted negative points)")


if __name__ == "__main__":
    raise SystemExit(main())
