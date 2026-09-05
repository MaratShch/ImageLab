"""Super Anscochrome, PS&E 1(1) p12 Fig. 4 -- four development curves.

Gifford, H.C. / Gerhardt, W.J., «Characteristics of Super Anscochrome Film»,
Photographic Science and Engineering 1(1) p12, July 1957. Ansco Development
Department, Binghamton NY; received 10 June 1957.

WHAT THIS EXTRACTS, and why the figure is worth a module of its own.

Fig. 4 draws FOUR characteristic curves of one reversal stock at FOUR first-
development times, with the exposure index each is rated at printed in a table
beside it:

    A   14 min first / 14 min colour developer, 68 degF   EI  80
    B   16 / 14   ("Normal")                              EI 100
    C   19 / 18                                           EI 150
    D   22 / 18                                           EI 200

That is the only four-point development ladder on a reversal stock anywhere in
this corpus; the runner-up is GEVACHROME_605's two points. B becomes the
profile's own curves, A/C/D become `ProcessVariant` records, and all four gammas
and base-fog values become a `ProcessingFamily`.

WHY THE THREE CHANNELS COME OUT IDENTICAL, and why that is a measurement.
Fig. 4 plots the THREE-LAYER AVERAGE. It does so because Fig. 1 -- same film,
same process, per layer -- draws cyan, magenta and yellow inside a single line
width over almost the whole range; the paper's phrase is "good curve
conformity". Fig. 1 is therefore USELESS for per-layer curves and Fig. 4 is the
right panel, which is the opposite of the usual preference and was established
by rendering both.

⚠ THE ABSCISSA UNIT IS STATED NOWHERE IN THE PAPER. The axis is labelled "LOG
EXPOSURE" and runs 7.5-9.3 with no lux-seconds, no metre-candle-seconds and no
speed point, so the stored curves are RELATIVE log exposure and the origin is a
convention. It is placed where curve B reaches D = 1.20; see the profile comment
for the argument and for the invariance that makes the choice harmless.

⚠ AND FIG. 1 AND FIG. 2 HAVE DIFFERENT ORIGINS FOR THE SAME FILM. Fig. 1 runs
8.1-9.9 and Fig. 2 runs 7.5-9.3. Anyone combining panels from this paper must
align them on the three-layer average first. This module reads Fig. 4 only.

THAT THE AXIS IS log10 IS CHECKED, NOT ASSUMED. `ei_ladder_check` compares the
measured horizontal offsets between the four curves against log10 of the printed
EI ratios. A base other than 10 would scale all three offsets by one constant,
and none does: the mean measured/predicted ratio is 1.006 over 36 comparisons.

Run as a script to reprint every number the profile cites.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import dashtrace as dt
import dye_density as dd
import kodak_still_curves as ks

# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------
PDF = ("PDF/PROFILES/RETRO/PSE/"
       "sim_journal-of-imaging-science_1957-07_1_1.pdf")
#: PDF page INDEX (0-based) of printed page 12.
PAGE = 16
#: Clip rectangle in PDF points around Fig. 4, chosen to include the tick
#: labels on both axes and to exclude the neighbouring column.
CLIP = (45.0, 375.0, 270.0, 640.0)
DPI = 400
SCALE = DPI / 72.0

#: Tick labels, value -> PDF-point coordinate of the label's centre, harvested
#: from the page's OCR TEXT LAYER rather than from the pixel grid. This page has
#: one, and using it is why the calibration residuals are a fifth of what a
#: pixel hunt gives on a 1957 halftone.
Y_TICKS = {4.0: 405.5, 3.5: 431.6, 3.0: 458.9, 2.5: 486.0,
           2.0: 513.3, 1.5: 539.0, 1.0: 566.2, 0.5: 592.6}
X_TICKS = {7.5: 87.3, 7.8: 113.8, 8.1: 138.9, 8.4: 166.6, 8.7: 193.5,
           9.0: 220.8}

#: ⚠ 195, NOT 170. At 170 the dashed segments of A, C and D break up badly
#: enough that the tracer loses them, and the running text above the frame is
#: still picked up. Measured at 195: 16540 ink pixels inside the panel and every
#: dash of all four curves recovered. This is a faded halftone, not vector art.
INK_MAX = 195

#: Panel interior in RENDERED PIXELS. The figure has no drawn frame -- only a
#: left ordinate at x = 182 and a bottom abscissa at y = 1365 -- so the interior
#: is bounded by hand, just inside those two rules and clear of the running text
#: above.
PANEL = dict(y0=150, y1=1355, x0=190, x1=1160)

#: Seed column and the four curve positions there, in rendered pixels. x = 370
#: (lg 7.78) is the leftmost column at which all four curves resolve into four
#: separate ink runs; at 350 only C is present.
SEED_X = 370
SEEDS = {"A": 206.0, "B": 273.0, "C": 373.0, "D": 492.5}

#: ⚠ merge_px IS 0 AND MUST STAY 0. `trace_predictive`'s crossing-coast exists
#: for panels where two curves genuinely cross; these four never do (more first
#: development moves a reversal curve down AND left, so the order A>B>C>D holds
#: at every abscissa). What they DO is converge to 11-16 px apart at the right
#: end, and a non-zero merge_px there would blank all four tracks over the last
#: 100 columns for a crossing that never happens.
TRACK = dict(tol0=6.0, tol_grow=0.8, max_bridge=40, hist=20,
             slope_cap=8.0, merge_px=0.0)

#: The table printed beside Fig. 4: curve -> (first dev min, colour dev min, EI).
LADDER = {"A": (14.0, 14.0, 80),
          "B": (16.0, 14.0, 100),
          "C": (19.0, 18.0, 150),
          "D": (22.0, 18.0, 200)}
NORMAL = "B"

#: Density at which curve B is placed at x = 0. See the module docstring.
MID_GREY_D = 1.20

#: What the adopted profile carries, so this module can fail loudly if a rerun
#: stops reproducing it. (dmin, gamma, toe_x, toe_k, shoulder_x, shoulder_k).
EXPECTED = {
    "A": (0.1530, 5.3801, -0.2430, 0.1322, 0.4502, 0.1278),
    "B": (0.1444, 5.2706, -0.1809, 0.1163, 0.4893, 0.1295),
    "C": (0.1161, 5.5856, -0.0118, 0.1066, 0.5708, 0.1181),
    "D": (0.0799, 5.9167, 0.1767, 0.1187, 0.6572, 0.0669),
}
#: Straight-line gamma over D 0.5-2.0, and fitted Dmin, as adopted.
EXPECTED_GAMMA_LINE = {"A": 3.5469, "B": 3.9368, "C": 4.3809, "D": 4.5702}


# ---------------------------------------------------------------------------
# axis
# ---------------------------------------------------------------------------
def fit_axis(ticks: dict[float, float]) -> tuple[float, float, float]:
    """value -> PDF point, least squares over ALL ticks. (slope, intercept,
    worst residual in points).

    ⚠ NO OUTLIER DROP HERE, unlike `dye_density._fit_axis`. That function's drop
    is tuned to residuals measured in POINTS and this module works in 400 dpi
    PIXELS, 5.6x larger; feeding it pixels silently discarded three of the six
    abscissa ticks for residuals that are fine. Fitting in points and converting
    afterwards keeps every tick and keeps the tolerance meaningful. All fourteen
    ticks on this panel are used.
    """
    v = np.array(sorted(ticks), dtype=float)
    px = np.array([ticks[k] for k in sorted(ticks)], dtype=float)
    m, c = np.linalg.lstsq(np.vstack([v, np.ones(len(v))]).T, px, rcond=None)[0]
    return float(m), float(c), float(np.abs(m * v + c - px).max())


class Axes:
    """Both axes of Fig. 4, in rendered pixels."""

    def __init__(self) -> None:
        self.my, self.cy, self.ry = fit_axis(Y_TICKS)
        self.mx, self.cx, self.rx = fit_axis(X_TICKS)

    @property
    def worst_density(self) -> float:
        return self.ry / abs(self.my)

    @property
    def worst_decade(self) -> float:
        return self.rx / abs(self.mx)

    def lg_of_px(self, px: float) -> float:
        return ((px / SCALE + CLIP[0]) - self.cx) / self.mx

    def d_of_px(self, py: float) -> float:
        return ((py / SCALE + CLIP[1]) - self.cy) / self.my


# ---------------------------------------------------------------------------
# raster and trace
# ---------------------------------------------------------------------------
def render(root: Path) -> np.ndarray:
    import pymupdf
    doc = pymupdf.open(str(root / PDF))
    pm = doc[PAGE].get_pixmap(dpi=DPI, clip=pymupdf.Rect(*CLIP),
                              colorspace=pymupdf.csGRAY)
    g = np.frombuffer(pm.samples, dtype=np.uint8)
    return g.reshape(pm.height, pm.width).astype(float)


def panel_ink(gray: np.ndarray) -> np.ndarray:
    ink = gray < INK_MAX
    ink[:PANEL["y0"], :] = False
    ink[PANEL["y1"]:, :] = False
    ink[:, :PANEL["x0"]] = False
    ink[:, PANEL["x1"]:] = False
    return ink


def trace(gray: np.ndarray, ink: np.ndarray) -> dict[str, dict[int, float]]:
    """Four tracks, seeded once and run BOTH ways.

    ⚠ BIDIRECTIONAL IS ADMISSIBLE HERE AND WAS NOT ON THE VISION3 SHEETS, and
    the difference is not stylistic. `dashtrace`'s direction rule exists because
    a density curve and a granularity curve meet TANGENTIALLY there, decidable
    in one direction only. These four curves never meet at all, so the only
    hazard a second direction adds is the ordinary one, and `check_order`
    asserts against it afterwards.
    """
    out: dict[str, dict[int, float]] = {}
    fwd = dt.trace_predictive(ink, gray, (PANEL["x0"], PANEL["x1"]),
                              PANEL["y0"], PANEL["y1"], SEED_X, SEEDS,
                              direction=+1, **TRACK)
    rev = dt.trace_predictive(ink, gray, (PANEL["x0"], PANEL["x1"]),
                              PANEL["y0"], PANEL["y1"], SEED_X, SEEDS,
                              direction=-1, **TRACK)
    for k in SEEDS:
        merged = dict(rev[k])
        merged.update(fwd[k])
        out[k] = merged
    return out


def check_order(tracks: dict[str, dict[int, float]]) -> tuple[int, int]:
    """(violations, columns tested). A > B > C > D top to bottom, everywhere."""
    keys = ["A", "B", "C", "D"]
    cols = set(tracks[keys[0]])
    for k in keys[1:]:
        cols &= set(tracks[k])
    bad = 0
    for x in cols:
        ys = [tracks[k][x] for k in keys]
        if any(b <= a for a, b in zip(ys, ys[1:])):
            bad += 1
    return bad, len(cols)


def check_solid_curve(ink: np.ndarray, track: dict[int, float]) -> tuple[int, int]:
    """Independent identity check on B, the one SOLID curve.

    ⚠ THIS IS THE STRONGEST CHECK IN THE MODULE, and it is available only
    because the paper drew one curve unbroken. A solid curve is ONE connected
    component of the ink mask; a track that wandered onto a neighbouring dashed
    curve would leave it. Returns (points off the component, points tested).
    """
    lab, info = dt._components(ink)
    biggest = max(info, key=lambda k: info[k][2])
    mask = lab == biggest
    off = 0
    for x, y in track.items():
        iy = int(round(y))
        if not mask[max(0, iy - 2):iy + 3, x].any():
            off += 1
    return off, len(track)


# ---------------------------------------------------------------------------
# the EI cross-check
# ---------------------------------------------------------------------------
def x_at_density(pts: list[tuple[float, float]], d: float) -> float | None:
    """Abscissa where a DESCENDING traced curve passes density `d`."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    for i in range(len(ys) - 1):
        if (ys[i] - d) * (ys[i + 1] - d) <= 0 and ys[i] != ys[i + 1]:
            t = (d - ys[i]) / (ys[i + 1] - ys[i])
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None


def ei_ladder_check(curves: dict[str, list[tuple[float, float]]],
                    densities=(2.8, 2.5, 2.2, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8,
                               0.6, 0.4, 0.3)):
    """Measured curve separations against the printed exposure indices.

    Yields (density, {curve: (measured, predicted)}) with offsets in decades,
    positive to the RIGHT of the normal curve. Predicted is log10(EI_normal /
    EI_curve): a slower rating sits further right.
    """
    ei0 = LADDER[NORMAL][2]
    for d in densities:
        base = x_at_density(curves[NORMAL], d)
        if base is None:
            continue
        row = {}
        for k, pts in curves.items():
            if k == NORMAL:
                continue
            got = x_at_density(pts, d)
            if got is None:
                continue
            row[k] = (got - base, math.log10(ei0 / LADDER[k][2]))
        yield d, row


def straight_line_gamma(pts, lo=0.5, hi=2.0) -> tuple[float, int]:
    """Least-squares slope over a density window, sign flipped to positive.

    The quantity a datasheet calls gamma. Reported alongside `ToneCurve.gamma`
    because the two are NOT the same number on a short-scale film: the softplus
    `gamma` is the asymptotic straight-line slope the fit extrapolates to, and
    this is the slope actually drawn between D 0.5 and 2.0.
    """
    a = np.array(pts)
    m = (a[:, 1] >= lo) & (a[:, 1] <= hi)
    slope = np.polyfit(a[m, 0], a[m, 1], 1)[0]
    return float(-slope), int(m.sum())


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------
def extract(root: Path | str = ".") -> dict:
    root = Path(root)
    ax = Axes()
    gray = render(root)
    ink = panel_ink(gray)
    tracks = trace(gray, ink)

    curves = {k: sorted((ax.lg_of_px(x), ax.d_of_px(y))
                        for x, y in t.items())
              for k, t in tracks.items()}

    bad, cols = check_order(tracks)
    off, tested = check_solid_curve(ink, tracks[NORMAL])

    # fit each curve on a provisional origin, then shift the whole family so
    # that the NORMAL curve reads MID_GREY_D at x = 0. One shift for all four:
    # they share one exposure axis and separating them would destroy the only
    # thing the panel measures about the four developments together.
    prov = 8.45
    fits = {}
    for k, pts in curves.items():
        p, rms, worst = ks.fit_tone_curve(
            sorted((-(x - prov), y) for x, y in pts), iters=8000)
        fits[k] = dict(params=p, rms=rms, worst=worst)

    pb = fits[NORMAL]["params"]
    lo, hi = -2.0, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (ks.model_density(lo, *pb) - MID_GREY_D) * \
           (ks.model_density(mid, *pb) - MID_GREY_D) <= 0:
            hi = mid
        else:
            lo = mid
    shift = 0.5 * (lo + hi)

    for k in fits:
        p = list(fits[k]["params"])
        p[2] -= shift
        p[4] -= shift
        fits[k]["adopted"] = tuple(round(v, 4) for v in p)
        fits[k]["dmax"] = p[0] + p[1] * (p[4] - p[2])
        fits[k]["gamma_line"], fits[k]["n_line"] = straight_line_gamma(curves[k])

    return dict(axes=ax, curves=curves, tracks=tracks, fits=fits,
                order_violations=bad, order_columns=cols,
                solid_off=off, solid_tested=tested,
                anchor_lg=prov - shift)


def main(root: Path | str = ".") -> int:
    r = extract(root)
    ax: Axes = r["axes"]
    print("Super Anscochrome -- Gifford & Gerhardt, PS&E 1(1) p12, Fig. 4")
    print("axis: %d y ticks worst %.3f pt = %.4f D | %d x ticks worst %.3f pt "
          "= %.4f decade"
          % (len(Y_TICKS), ax.ry, ax.worst_density,
             len(X_TICKS), ax.rx, ax.worst_decade))
    print("ordering A>B>C>D: %d violations in %d shared columns"
          % (r["order_violations"], r["order_columns"]))
    print("curve B against the solid component: %d of %d points off it"
          % (r["solid_off"], r["solid_tested"]))
    print("x = 0 anchored where B reads D %.2f -- traced lg %.4f"
          % (MID_GREY_D, r["anchor_lg"]))
    print()
    bad = 0
    for k in "ABCD":
        f = r["fits"][k]
        fd, cd, ei = LADDER[k]
        print("%s  %2.0f/%2.0f min  EI %3d  n %3d  rms %.4f worst %.4f"
              % (k, fd, cd, ei, len(r["curves"][k]), f["rms"], f["worst"]))
        print("    ToneCurve(%.4f, %.4f, %.4f, %.4f, %.4f, %.4f)"
              % f["adopted"])
        print("    dmax %.4f  gamma(D 0.5-2.0) %.4f over %d pts  sh_k/toe_k %.3f"
              % (f["dmax"], f["gamma_line"], f["n_line"],
                 f["adopted"][5] / f["adopted"][3]))
        if max(abs(a - b) for a, b in zip(f["adopted"], EXPECTED[k])) > 5e-4:
            print("    ⚠ DRIFTED from the adopted profile: %s" % (EXPECTED[k],))
            bad += 1
        if abs(f["gamma_line"] - EXPECTED_GAMMA_LINE[k]) > 5e-4:
            print("    ⚠ straight-line gamma drifted from %.4f"
                  % EXPECTED_GAMMA_LINE[k])
            bad += 1
    print()
    print("EI ladder: measured offset from B vs log10(EI ratio), in decades")
    for d, row in ei_ladder_check(r["curves"]):
        cells = "  ".join("%s %+.4f (%+.4f)" % (k, m, p)
                          for k, (m, p) in sorted(row.items()))
        print("  D %.1f   %s" % (d, cells))
    ratios = [m / p for _d, row in ei_ladder_check(r["curves"])
              for m, p in row.values()]
    print("  mean measured/predicted %.3f over %d comparisons"
          % (sum(ratios) / len(ratios), len(ratios)))

    if r["order_violations"] or r["solid_off"] or bad:
        print("\nFAIL")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent),
                    help="directory holding PDF/ (default: beside this file)")
    # ⚠ ACCEPTED AND IGNORED, on purpose. Every check this module can make is
    # already fatal -- ordering, the solid-component identity of curve B, and
    # the six adopted parameters of all four curves -- so there is no lenient
    # mode to switch out of. The flag exists so build.py can invoke this module
    # with the same argument vector as every other extractor in its audit list.
    ap.add_argument("--assert", dest="assert_", action="store_true",
                    help="accepted for build.py; this module always asserts")
    a = ap.parse_args()
    raise SystemExit(main(a.root))
