"""sigma(D) from a VECTOR Kodak granularity plot -- all eight vector sheets.

WHY THIS EXISTS ALONGSIDE vision3_granularity.py
------------------------------------------------
`vision3_granularity.py` reads the same plot type off the four VISION3 sheets,
where it is a RASTER image, so it has to threshold ink, split curve families by
stroke style in pixel space and trace predictively. All of that machinery exists
because the source is pixels.

Eight sheets in the corpus draw the identical plot in VECTOR paths. Nothing needs
tracing: the curves are analytic, the axis ticks are text, and the stroke colour
and dash pattern are stored per path. So this reads the numbers instead of
recovering them, and the only real work is the geometry.

SCOPE, as of 2026-08-18 (queue item C1c -- the completing sigma(D) harvest):

    5285   H-1-5285 p4                EKTACHROME 100D, a REVERSAL stock
    5245   5245.pdf p4                EXR 50D
    5246   5246.pdf p4                VISION 250D
    5248   5248.pdf p3                EXR 100T
    5274   5274.pdf p3                VISION 200T
    5279   5279.pdf p2                VISION 500T
    5218   H-1-5218 p4                VISION2 500T
    5219v  VISION3 500T brochure p2   CROSS-CHECK ONLY -- 5219's shape was
                                      already traced from the RASTER plot on its
                                      technical sheet by the other extractor, so
                                      this is two independent extractions, two
                                      documents and two media for one stock.

Found by sweeping every staged Kodak PDF for a rotated GRANULARITY caption on a
page carrying zero embedded images. Seven of the eight are adopted into
film_profiles.py with sigma_shape_measured=True; the eighth is the cross-check.

Why it matters more than "seven more stocks": before this, every sigma(D) triple
in the database came from a colour NEGATIVE, and the shape was carried by three
monotone anchors. 5285 is a colour REVERSAL film -- density falls with exposure,
dmin sits at the highlight end -- and its sigma(D) rises about twentyfold with
density, the opposite of what negative experience predicts. The six negatives
added alongside it all turn OVER: sigma peaks between D 0.65 and 0.74 at
1.38-1.62x its D = 1.0 value and falls to 0.50-0.90x by dmax.

THE PLOT, and the four things that make it readable
--------------------------------------------------
One frame carries six curves (five on the brochure) against a shared
log-exposure axis:

  * three SOLID characteristic curves, density on the LEFT axis;
  * two or three granularity curves, sigma_D on the RIGHT axis, LOG scaled.

sigma(D) is then a composition, exactly as the sheet's own instructions describe
it: "find the density on the left vertical scale and follow horizontally to the
characteristic curve and then go vertically ... to the granularity curve. At that
point, follow horizontally to the Granularity Sigma D scale on the right."

  1. ⚠ NOTHING STRUCTURAL GENERALISES ACROSS THE EIGHT SHEETS -- not the dash
     array, not "one path per family", not "one path per layer", not even "one
     curve per path". Kodak packages this plot eight different ways; the measured
     table is in curves() and stitch(). What generalises is the PHYSICS: on the
     shared density axis a characteristic curve must traverse the film's whole
     density scale while a granularity curve wanders within a few tenths, so the
     families split on vertical SPAN with a measured gap of 0.83-1.59 D against a
     0.60 D threshold.
  2. ⚠ CURVES ARRIVE IN PIECES, in three different ways: as dozens of dash
     fragments (5245 and 5248), as several abutting path objects (5279), and as
     flat 2-point straight dashes where the curve stops bending. All three are
     stitched, with the join constrained by the chain's own local slope so a
     crossing cannot swap two curves silently.
  3. LAYER IDENTITY IS PRINTED, never inferred. Seven sheets print R / G / B
     letters beside both families, and the assignment is solved as an exhaustive
     bijection under the known partition -- six letters, six curves, one triple
     per family. The brochure prints no letters and states identity in INK
     instead, so that sheet is read by stroke colour.
  4. THE OVERLAY IS THE GATE. Every traced point is drawn back onto the rendered
     panel (--overlay); no sheet's numbers were adopted before its overlay was
     looked at. Internally consistent numbers from a hybrid curve are the failure
     mode this catches and the numbers cannot.

AXES: all three are least-squares fits over every harvested tick with a residual
test, for the reason recorded in dye_density.py -- a two-point span is exact when
the labels are right and silently wrong when one is not. The LOG-EXPOSURE axis is
OPTIONAL: sigma(D) is a composition of two curves sharing one abscissa, so the
abscissa cancels and the composition is done in raw pixel x. Verified by
reproducing 5285's calibrated result to the last digit in pixel space.

⚠ THE X LABELS MAY HAVE NO MINUS SIGNS in the text layer. On 5285 they read
3.0 / 2.0 / 1.0 / 0.0 left to right, i.e. DESCENDING magnitudes, which can only
be -3 -2 -1 0. The script decides from the data instead of assuming a convention.

⚠ THE TOE ANCHOR IS NOT ALWAYS UNIQUE. Below the toe the characteristic curve is
flat: density holds at dmin while sigma keeps changing, so sigma(D) is
multivalued there. This is the measured explanation of the ONLY real
disagreement between the 5219 raster and vector traces (toe/mid 0.67 vs 0.40,
while dmax/mid agreed to 0.02 and the peak location to 0.03 D). Where it happens
the sigma span over the plateau is printed.

Run:
    python granularity_vector.py --root ../..            # extract + self-check
    python granularity_vector.py --root ../.. --assert   # non-zero if it moves
    python granularity_vector.py --root ../.. --overlay /tmp/o.png

Needs numpy + PyMuPDF.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

#: tag -> (pdf under PDF/PROFILES/KODAK, page, profile, rotated right-axis word)
SHEETS = {
    "5285": ("Ektachrome_100d.pdf", 4, "KODAK_EKTACHROME_100D_5285"),
    # ---- 2026-08-18, the completing harvest. Every remaining sheet in the
    # ---- corpus whose granularity plot is VECTOR, found by sweeping all 27
    # ---- staged Kodak PDFs for a rotated GRANULARITY caption on a page with
    # ---- zero embedded images.
    "5245": ("5245.pdf", 4, "EASTMAN_EXR_50D_5245"),
    "5246": ("5246.pdf", 4, "KODAK_VISION_250D_5246"),
    "5248": ("5248.pdf", 3, "EASTMAN_EXR_100T_5248"),
    "5274": ("5274.pdf", 3, "KODAK_VISION_200T_5274"),
    "5279": ("5279.pdf", 2, "KODAK_VISION_500T_5279"),
    "5218": ("5218-Vision2-500T-H-1-5218t.pdf", 4, "KODAK_VISION2_500T_5218"),
    # ⚠ A CROSS-CHECK, NOT A NEW STOCK. 5219's shape was traced from the RASTER
    # plot on its TI sheet by vision3_granularity.py; the brochure prints the same
    # plot as VECTOR art. Two independent extractions of one stock by two
    # different methods is the strongest validation available here, so it is run
    # every time rather than kept as a one-off note.
    # ⚠ AND IT DOES NOT PRINT THE SAME PLOT. The brochure omits the RED
    # granularity curve (5 curves, not 6) and prints no R / G / B letters at
    # all -- it states layer identity in INK instead. Both facts are declared
    # here rather than discovered as a failure.
    "5219v": ("KODAK-VISION3-500T-5219-7219-brochure.pdf", 2,
              "KODAK_VISION3_500T_5219",
              dict(want=(3, 2), identity="colour")),
    # ---- 2026-08-20. A NINTH sheet, and the first one that carries a stock the
    # ---- database did not have at all. H-1-5201 (10-2005) is a BROCHURE, so it
    # ---- is read the same way as 5219v: no R / G / B letters anywhere, layer
    # ---- identity stated in INK. It does print all six curves, unlike 5219v.
    # ⚠ THE RED RECORD IS DRAWN TWICE, once in yellow and once in magenta on top
    # of it, so the composite reads red on paper. colour_assign() sees two
    # coincident curves per red record; they are the SAME geometry (measured:
    # identical rects to 0.1 pt), so the duplicate is dropped rather than being
    # allowed to occupy a slot in the bijection.
    "5201": ("Kodak VISION2 50D 5201.pdf", 3, "KODAK_VISION2_50D_5201",
             dict(identity="colour")),
}

#: Measured 2026-08-18. --assert fails if the sheet stops reproducing these.
#: sigma at (dmin, D=1.0, dmax), normalised to the D=1.0 anchor, green record.
#: ⚠ MEASURED, not predicted. An earlier version of this file carried a GUESSED
#: triple (1.60 / 1.00 / 0.37) written before the extraction ran, and it was
#: wrong in both direction and magnitude -- a reminder that for a reversal stock
#: the shape is not intuitable from negative experience. The real shape rises
#: steeply with density: sigma at dmin is a SIXTH of its value at D = 1.0, and at
#: dmax it is about DOUBLE. Green record, the visually weighted one.
#: ⚠ EVERY SHEET IS PINNED, not just the first one. Seven of the eight are the
#: numbers actually adopted into film_profiles.py, so a change in the extractor
#: that moves them is a change to the DATABASE and must be seen as a failure
#: here first. The eighth (5219v) is a CROSS-CHECK of a shape traced by a
#: different extractor from a raster plot -- pinned so that agreement, and the
#: one disagreement it exposes, cannot silently drift either.
#: `peak` is the interior maximum as a multiple of the D = 1.0 value and
#: `peak_at` the density it occurs at.
EXPECTED = {
    "5285": dict(toe=0.15, mid=1.00, dmax=3.10, peak=3.13, peak_at=3.34, tol=0.06),
    "5245": dict(toe=1.19, mid=1.00, dmax=0.72, peak=1.47, peak_at=0.73, tol=0.06),
    "5246": dict(toe=0.94, mid=1.00, dmax=0.90, peak=1.62, peak_at=0.66, tol=0.06),
    "5248": dict(toe=1.19, mid=1.00, dmax=0.84, peak=1.58, peak_at=0.74, tol=0.06),
    "5274": dict(toe=0.80, mid=1.00, dmax=0.61, peak=1.38, peak_at=0.68, tol=0.06),
    "5279": dict(toe=0.96, mid=1.00, dmax=0.50, peak=1.42, peak_at=0.65, tol=0.06),
    "5218": dict(toe=1.17, mid=1.00, dmax=0.70, peak=1.56, peak_at=0.74, tol=0.06),
    # ⚠ THE BROCHURE AND THE TECHNICAL SHEET DO NOT AGREE ON ABSOLUTE SIGMA, and
    # the conflict is recorded rather than averaged (method rule 4). Same stock,
    # two documents, two extractors, two plot media:
    #     raster, H-1-5219 p3     7.11 / 10.60 / 5.84, peak 1.32x at D 0.79
    #     vector, brochure p2     3.20 /  8.03 / 4.60, peak 1.24x at D 0.76
    # SHAPE agrees: dmax/mid 0.55 vs 0.57, peak location 0.79 vs 0.76, peak
    # height 1.32x vs 1.24x. ABSOLUTE sigma differs by a near-uniform 1.3x, so
    # one of the two sigma-axis calibrations is off by about a third of a decade;
    # the brochure's own ladder is internally consistent to 0.5 pt over two
    # decades, which is why it is pinned here and NOT adopted over the raster
    # trace on its own. The toe disagreement (0.67 vs 0.40) is NOT a document
    # conflict at all -- see the plateau note in main().
    "5219v": dict(toe=0.40, mid=1.00, dmax=0.57, peak=1.24, peak_at=0.76, tol=0.06),
    # 2026-08-20, H-1-5201 p3. The flattest sigma(D) in the corpus: the interior
    # peak is only 1.20x the D = 1.0 value, against 1.38-1.62x on the six other
    # negatives. That is consistent with what the sheet claims in words -- "the
    # measured granularity is exceptionally low" -- and it is the reason this
    # stock's grain will look different in kind, not just in amount.
    "5201": dict(toe=0.54, mid=1.00, dmax=0.89, peak=1.20, peak_at=0.80, tol=0.06),
}

TICK_RESID_PT = 1.5
DASH_GRAN = "[ 3.2401 1.6201 ] 0"
DASH_CHAR = "[] 0"


# --------------------------------------------------------------------------
def subpaths(items):
    """Split one drawing's items into disjoint polylines.

    ⚠ THIS IS THE FUNCTION THE EXTRACTION TURNS ON. Kodak emits all three
    granularity curves as ONE path object; treating it as one curve yields a
    trace that teleports between layers and still looks plausible when plotted.
    """
    out, cur, last = [], [], None
    for it in items:
        if it[0] == "l":
            pts = [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        elif it[0] == "c":
            p = [it[1], it[2], it[3], it[4]]
            pts = []
            for k in range(25):
                t = k / 24.0
                u = 1.0 - t
                pts.append((
                    u**3*p[0].x + 3*u*u*t*p[1].x + 3*u*t*t*p[2].x + t**3*p[3].x,
                    u**3*p[0].y + 3*u*u*t*p[1].y + 3*u*t*t*p[2].y + t**3*p[3].y))
        else:
            continue
        if last is not None and (abs(pts[0][0]-last[0]) > 0.15
                                or abs(pts[0][1]-last[1]) > 0.15):
            if len(cur) >= 2:
                out.append(cur)
            cur = []
        cur += pts
        last = pts[-1]
    if len(cur) >= 2:
        out.append(cur)
    return out


def fit(pairs, label, min_keep=4):
    """{value: pixel} -> (slope, intercept, worst kept residual, n kept, dropped).

    ⚠ ONE TICK ON THIS SHEET IS GENUINELY MISPLACED and the outlier rejection is
    not defensive programming, it is required. The sigma axis prints nine labels
    from .001 to .100; a fit over all nine leaves residuals under 1.1 pt for eight
    of them and **-1.94 pt for ".010"**. The give-away is the decade spacing:
    taken at face value the labels put .001->.010 at 42.5 px and .010->.100 at
    39.2 px, which cannot both be a decade on a log axis. Dropping ".010" makes
    them 40.6 and 41.1 px. So the label is nudged, the axis is fine, and a fit
    that trusted every label would have carried a 5 % sigma error into an adopted
    number. Rejection stops while at least `min_keep` ticks remain, so a sparse
    axis cannot be whittled down to a fabricated line.
    """
    v = np.array(sorted(pairs), dtype=float)
    px = np.array([pairs[k] for k in sorted(pairs)], dtype=float)
    keep = np.ones(len(v), bool)
    dropped = []
    while True:
        A = np.vstack([v[keep], np.ones(keep.sum())]).T
        m, c = np.linalg.lstsq(A, px[keep], rcond=None)[0]
        res = np.abs(m*v + c - px)
        masked = np.where(keep, res, -1.0)
        worst = int(np.argmax(masked))
        if res[worst] <= TICK_RESID_PT or keep.sum() <= min_keep:
            break
        keep[worst] = False
        dropped.append((float(v[worst]), float(res[worst])))
    worst_kept = float(res[keep].max())
    if worst_kept > TICK_RESID_PT:
        raise SystemExit(f"[!] {label} ticks not collinear: {worst_kept:.2f} pt "
                         f"over {int(keep.sum())} kept ticks")
    return m, c, worst_kept, int(keep.sum()), dropped


def density_axis(fr, dens):
    """The left density axis, with a FRAME-SPAN fallback for jittered labels.

    ⚠ ON ONE SHEET THE PRINTED LABELS ARE TYPOGRAPHICALLY JITTERED and a fit
    through them is not collinear at any usable tolerance. H-1-5201 (the 10-2005
    brochure) sets its four density labels at y 436.91 / 401.13 / 373.54 / 345.59
    while the panel's own horizontal gridlines sit at 436.26 / 406.13 / 375.99 /
    345.85 -- a uniform 30.14 pt ladder. The "1.0" label is 5.0 pt off its own
    gridline, i.e. 0.17 D, and the label fit fails at 3.24 pt.

    The recovery is NOT a loosened tolerance. The density axis of every one of
    these panels spans the frame exactly: measured on the eight sheets whose
    labels ARE collinear, the frame-span slope agrees with the label-fit slope to
    0.02-0.7 %, and on 5201 to 0.06 % (-30.138 vs -30.155 pt/D) with an intercept
    landing on the gridline ladder to 0.00 pt. So the fallback is the frame edges
    carrying the labels' own extreme VALUES, admitted only when its slope agrees
    with the (bad) label fit to 2 % -- which is the check that the labels are
    jittered about the right line rather than mis-read wholesale.

    Returns fit()'s tuple, so callers cannot tell which branch produced it.
    """
    try:
        return fit(dens, "density")
    except SystemExit:
        pass
    v = sorted(dens)
    if len(v) < 3:
        raise SystemExit("[!] density ticks not collinear and too few to recover")
    lo, hi = v[0], v[-1]
    m_frame = (fr.y0 - fr.y1) / (hi - lo)
    A = np.vstack([np.array(v, float), np.ones(len(v))]).T
    m_lab = np.linalg.lstsq(A, np.array([dens[k] for k in v], float),
                            rcond=None)[0][0]
    if abs(m_frame / m_lab - 1.0) > 0.02:
        raise SystemExit(f"[!] density ticks not collinear and the frame-span "
                         f"fallback disagrees: {m_frame:.3f} vs {m_lab:.3f} pt/D")
    c_frame = fr.y1 - lo * m_frame
    worst = max(abs(m_frame * k + c_frame - dens[k]) for k in v)
    print(f"    density      FRAME-SPAN fallback: labels jitter up to "
          f"{worst:.2f} pt about a {m_frame:.3f} pt/D axis pinned to the frame "
          f"edges ({lo:.1f} at the bottom, {hi:.1f} at the top)")
    return m_frame, c_frame, 0.0, len(v), []


#: ⚠ TWO PRINTED FORMS, and missing the second one hid the best validation this
#: file has. The technical sheets label the sigma axis `.001 .002 ... .500`; the
#: 5219 BROCHURE labels the same axis `0.001 0.002 ... 0.10`. A pattern anchored
#: on a leading dot found the ladder on six sheets and reported "no sigma ladder"
#: on the seventh -- which happens to be the only stock in the corpus whose shape
#: was already traced INDEPENDENTLY, from the raster plot on its TI sheet. So the
#: one sheet the regex rejected was the one that could check all the others.
#: Two or three decimals excludes the density labels (0.0, 0.2, 1.0 -- one
#: decimal) while keeping 0.10 and .100.
SIGMA_TICK = re.compile(r"0?\.\d{2,3}$")


def sigma_ladder(pg):
    """The sigma_D tick ladder: (x_centre, {value: y}) for the best column.

    ⚠ THIS, NOT THE CAPTION, IS THE ANCHOR -- and the first version of this file
    got it wrong. Anchoring on the rotated "GRANULARITY" caption and taking the
    frame to its left worked on the 5285 sheet and failed on all six others: the
    caption sits right of the plot on some sheets and left on others, some pages
    carry three captions, and the 5219 brochure has no frame left of it at all.
    What every one of these plots DOES have is a right-hand logarithmic sigma
    axis printed as `.001 .002 .003 .005 .010 ... .500` in one column. Nothing
    else on a Kodak sheet looks like that, so it identifies the plot outright.
    """
    cols = {}
    for a, b, c, d, t, *_ in pg.get_text("words"):
        if not SIGMA_TICK.fullmatch(t):
            continue
        v = float(t)
        if not (0.0 < v < 1.0):
            continue
        cols.setdefault(round((a + c) / 12.0), {})[v] = (b + d) / 2.0
    best = max(cols.items(), key=lambda kv: len(kv[1]), default=(None, {}))
    if len(best[1]) < 4:
        return None, {}
    xs = [((a + c) / 2.0) for a, b, c, d, t, *_ in pg.get_text("words")
          if SIGMA_TICK.fullmatch(t) and round((a + c) / 12.0) == best[0]]
    return sum(xs) / len(xs), best[1]


def frame_and_ticks(pg):
    """Locate the granularity plot from its sigma ladder, and read three axes."""
    sig_x, sig = sigma_ladder(pg)
    if sig_x is None:
        raise SystemExit("[!] no sigma_D tick ladder (.001 .002 ...) on this page")
    # the plot frame is the widest box whose right edge sits just LEFT of the
    # ladder and whose vertical span contains it
    # ⚠ AND IT MUST BE SQUARE. Every one of these plots is drawn square -- measured
    # 172x173, 155x155, 184x184, 117x117, 89x90 pt across the nine sheets -- so
    # squareness is a property of the figure, not a tuning knob. It has to be
    # tested because the H-1-5201 brochure draws its panels at 89 pt, below the
    # old 110x100 floor, and lowering that floor alone let a full-page 308x808
    # background box win the "widest qualifying" contest and produce a frame with
    # zero density ticks in it.
    ys = list(sig.values())
    best = None
    for p in pg.get_drawings():
        r = p["rect"]
        if r.width < 80 or r.height < 80 or r.width > 560:
            continue
        if abs(r.width - r.height) > 0.15 * max(r.width, r.height):
            continue
        if not (r.x1 <= sig_x + 4 and sig_x - r.x1 < 70):
            continue
        if not (r.y0 - 12 <= min(ys) and max(ys) <= r.y1 + 12):
            continue
        if best is None or r.width * r.height > best.width * best.height:
            best = r
    if best is None:
        raise SystemExit(f"[!] no frame just left of the sigma ladder at x={sig_x:.0f}")
    fr = best

    xs, dens = {}, {}
    for a, b, c, d, t, *_ in pg.get_text("words"):
        cx, cy = (a + c) / 2.0, (b + d) / 2.0
        if not re.fullmatch(r'-?\d+(\.\d+)?', t):
            continue
        v = float(t)
        if fr.x0 - 2 <= cx <= fr.x1 + 12 and fr.y1 + 1 <= cy <= fr.y1 + 22:
            xs[v] = cx                              # log exposure, below
        elif fr.x0 - 30 <= cx <= fr.x0 + 10 and fr.y0 - 8 <= cy <= fr.y1 + 8 \
                and 0.0 <= v <= 6.0:
            dens[v] = cy                            # density, left axis
    # ⚠ THE X LABELS MAY OR MAY NOT CARRY THEIR MINUS SIGNS. On the 5285 sheet
    # the text layer drops them, so the labels read 3.0 / 2.0 / 1.0 / 0.0 left to
    # right -- descending magnitudes that can only be -3 -2 -1 0. Other sheets in
    # the same family print the signs. Decide from the data instead of assuming
    # either: if any label is already negative, trust the printed signs; else if
    # the magnitudes descend left to right, negate them; else take them as they
    # are. Guessing one convention is what broke five of the seven sheets.
    if xs and not any(k < 0 for k in xs):
        order = [xs[k] for k in sorted(xs)]
        if len(order) > 1 and all(order[i] > order[i + 1] for i in range(len(order) - 1)):
            xs = {-k: v for k, v in xs.items()}
    # ⚠ THE LOG-EXPOSURE AXIS IS NOT NEEDED, and discovering that removed a whole
    # class of failure. sigma(D) is a COMPOSITION of two curves that share one
    # abscissa, so the abscissa cancels: interpolating in raw PIXEL x gives the
    # identical answer and needs no x calibration at all. That matters because
    # the 5245 sheet prints "RELATIVE LOG EXPOSURE" as a title with NO numeric
    # tick labels whatsoever, and two other sheets put the density axis's own
    # "0.0" inside the band where x labels were being harvested. So the x fit is
    # now OPTIONAL and purely informational -- it is reported when the labels
    # exist, and its absence no longer stops the extraction.
    # Verified: re-running 5285 in pixel space reproduces its calibrated result
    # to the last digit, which is what a cancelling parameter must do.
    if len(dens) < 3 or len(sig) < 4:
        raise SystemExit(f"[!] ticks: density={len(dens)} sigma={len(sig)} "
                         f"(frame {fr.width:.0f}x{fr.height:.0f})")
    fx = fit(xs, "log-exposure") if len(xs) >= 3 else None
    fd = density_axis(fr, dens)
    fs = fit({np.log10(k): v for k, v in sig.items()}, "log sigma")
    return fr, fx, fd, fs, xs, dens, sig


def curves(pg, fr):
    """The six curves inside the granularity frame, split into two families.

    ⚠ KODAK PACKAGES THIS PLOT SIX DIFFERENT WAYS ACROSS SEVEN SHEETS, and every
    structural rule tried first worked on one sheet and failed on the rest.
    Measured, per sheet, as (paths inside the frame -> subpaths each):

        5285   2 paths -> 3 + 3, and the two families differ in DASH pattern
        5246   3 paths -> 2 + 2 + 2, one path per LAYER, all solid
        5218   4 paths -> 2 + 1 + 2 + 1, all solid
        5274   6 paths -> 1 each, all solid
        5279   5 paths -> 1 each (two curves share a path), all solid
        5248   1 path  -> 3, all solid
        5245   2 paths -> 2 + 1, all solid

    So neither the dash array, nor "one path per family", nor "one path per
    layer" generalises. What DOES generalise is the physics: on the shared
    DENSITY axis the three characteristic curves climb to 2.4-3.5 D while the
    three granularity curves stay below ~1.8 D, a gap of at least 0.6 D on every
    sheet. The families are therefore split on maximum density, and the gap is
    ASSERTED rather than assumed -- if the two groups are not clearly separated
    the sheet is refused instead of guessed at.

    Layer identity comes from the sheet's own printed R / G / B letters, one
    triple per family, assigned to the nearest curve. That is checked too: each
    family must end up with exactly one R, one G and one B.
    """
    polys = []
    for p in pg.get_drawings():
        r = p["rect"]
        if not (r.x0 >= fr.x0-4 and r.x1 <= fr.x1+4
                and r.y0 >= fr.y0-4 and r.y1 <= fr.y1+4):
            continue
        # ⚠ THE PER-PATH ITEM FLOOR USED TO BE 8, AND IT COST THE 5279 SHEET ITS
        # SIXTH CURVE. That sheet draws one granularity curve as THREE separate
        # path objects of 12 / 6 / 5 items, abutting in x (79.8..107.6,
        # 107.5..131.2, 131.2..235.0); the two short ones were discarded before
        # stitch() ever saw them, so the sheet presented 5 curves and was
        # refused. The floor is now 2 items, and the geometric filters below do
        # the rejecting instead: a subpath must carry >= 4 POINTS, which drops
        # every axis tick (one "l" item -> 2 points) and the frame rectangle
        # ("re" -> no line items at all), and a stitched result must still span
        # 20 % of the frame width. Verified: all seven other sheets reproduce
        # their previous numbers to the last digit under the lower floor.
        if sum(1 for it in p["items"] if it[0] in ("l", "c")) < 1:
            continue
        subs = subpaths(p["items"])
        if not subs:
            continue
        # ⚠ AXIS TICKS ARE REJECTED PER PATH, NOT PER SEGMENT, and that is what
        # makes it safe to accept 2-point fragments at all. Measured on 5245:
        # a tick rack is its OWN path whose every subpath is a single 2-point
        # segment (path#3 = 9 vertical bottom ticks, path#8/#9 = the left-axis
        # ticks, drawn 4.6 pt long and perfectly horizontal -- geometrically
        # indistinguishable from a flat dash). A curve path always carries at
        # least one subpath of >= 4 points. So the classification is: a path
        # containing no subpath longer than 3 points draws ticks, and every
        # subpath of a path that does is curve ink.
        if max(len(q) for q in subs) < 4:
            continue
        # ⚠ THE STROKE COLOUR IS CARRIED ALONG, because on one sheet it IS the
        # layer identity. The technical sheets print R / G / B letters and draw
        # everything in black; the 5219 BROCHURE prints no letters at all and
        # draws each layer in its own ink. Colour is as printed as a letter is --
        # both are the sheet stating which record it means -- so it is kept here
        # rather than re-derived later, and stitch() also refuses to chain across
        # colours, which no black sheet notices and a colour sheet needs.
        col = p.get("color")
        col = tuple(round(float(c), 3) for c in col) if col else None
        for q in subs:
            if len(q) < 2:
                continue
            polys.append((q, col))
    # ⚠ STITCH EVERYTHING, ALWAYS -- do not shortcut when six wide pieces exist.
    # The earlier version returned immediately if six subpaths were each wide
    # enough on their own, and 5279 is the counter-example: it draws one
    # granularity curve as three abutting pieces, the RIGHTMOST of which spans
    # 67 % of the frame by itself. So "six wide pieces" was satisfied by five
    # whole curves plus one curve's right-hand third, the sheet passed the count
    # test, and the left two thirds of that curve were silently discarded --
    # a fragment that survives the count test is worse than one that fails it.
    # Stitching whole curves is safe because a chain can only grow rightward
    # across a <= DASH_GAP_PT gap: a curve that already reaches the frame's right
    # edge has nothing to absorb, which is why the four sheets that never needed
    # stitching reproduce their numbers to the last digit under this path.
    joined = [(q, c) for q, c in stitch(polys)
              if len(q) >= 10
              and (max(x for x, _ in q) - min(x for x, _ in q)) >= 0.20*fr.width]
    return drop_overprints(joined)


OVERPRINT_TOL_PT = 0.4


def drop_overprints(joined):
    """Drop a curve that is another curve redrawn in a second ink.

    ⚠ ON H-1-5201 THE RED RECORD IS DRAWN TWICE, once in yellow (0.97, 0.76,
    0.14) and once in magenta (0.93, 0.00, 0.55) laid exactly on top of it, so
    the composite reads red on paper. Both families do it, so the sheet presents
    EIGHT qualifying curves where the physics says six, and the count gate
    refuses it. That refusal is correct and the fix is not to relax the count:
    the two strokes are the SAME GEOMETRY, measured identical to 0.02 pt over
    every shared sample, so one of them carries no information.

    The pair is collapsed rather than both being kept because colour_assign()
    solves a bijection: two coincident curves would claim two different records
    for one physical layer and silently corrupt the identity of a THIRD.

    Which one survives matters, and it is decided by ink, not by draw order:
    yellow + magenta composites to red, so the survivor is recoloured to pure
    red -- the colour the sheet actually prints. Nothing else in the file has to
    know that Kodak spelled red this way.
    """
    # ⚠ THE TWO STROKES ARE NOT THE SAME POINT LIST. Measured on 5201's
    # characteristic red: the yellow pass stitches to 100 points and the magenta
    # pass to 106, from 7 and 10 source fragments -- identical rects, different
    # sampling. So coincidence is tested by RESAMPLING both on a shared x grid,
    # not by zipping two point lists, which is what an earlier version did and
    # why it collapsed the granularity pair and missed the characteristic pair.
    def sample(q, gx):
        pts = sorted(q)
        return np.interp(gx, [p[0] for p in pts], [p[1] for p in pts])

    out = []
    for q, col in joined:
        x0, x1 = min(p[0] for p in q), max(p[0] for p in q)
        dup = None
        for i, (q2, _) in enumerate(out):
            a0, a1 = min(p[0] for p in q2), max(p[0] for p in q2)
            lo, hi = max(x0, a0), min(x1, a1)
            if hi - lo < 0.90 * min(x1 - x0, a1 - a0):
                continue
            gx = np.linspace(lo, hi, 50)
            if float(np.max(np.abs(sample(q, gx) - sample(q2, gx)))) \
                    <= OVERPRINT_TOL_PT:
                dup = i
                break
        if dup is None:
            out.append((q, col))
            continue
        # yellow over magenta (or the reverse) is a red composite; say so once,
        # and keep the DENSER of the two point lists
        pair = {col, out[dup][1]}
        keep = q if len(q) > len(out[dup][0]) else out[dup][0]
        new = out[dup][1]
        if len(pair) == 2 and all(c is not None and c[0] > 0.85 for c in pair):
            new = (1.0, 0.0, 0.0)
        out[dup] = (keep, new)
    return out


DASH_GAP_PT = 7.0        # measured: fragments are 3-4 pt long, 2-4 pt apart
DASH_Y_TOL_PT = 2.6      # a fragment must continue within this much in y


def stitch(frags):
    """Chain dash fragments into whole curves. Method rule 6, applied here.

    ⚠ TWO OF THE SEVEN SHEETS DRAW THEIR GRANULARITY CURVES AS DASHED STROKES
    EMITTED AS DOZENS OF 3-4 pt SEGMENTS (5245 has 44 such fragments, 5248 has
    59), and one more sheet each fragments a single curve. Without stitching,
    those sheets present three curves instead of six and the extraction refuses
    them -- which is the correct refusal, but it leaves a third of the corpus
    unread for a purely typographic reason.
    Chaining is greedy and deliberately timid: a fragment may only continue a
    curve if it starts within DASH_GAP_PT in x AND within DASH_Y_TOL_PT in y of
    where the curve currently ends. Both bounds come from the measured dash
    geometry, not from taste -- method rule 6 says bridge only up to the measured
    dash period, never a guessed tolerance. Where curves cross, chaining can in
    principle swap them; that is why the OVERLAY is the gate and every stitched
    sheet is looked at before its numbers are used.
    """
    # ⚠ A 2-POINT FRAGMENT IS A DASH, NOT NOISE, and this filter used to say
    # `>= 4`. Where a granularity curve goes FLAT its dashes need no bezier, so
    # the writer emits them as single straight segments -- on 5245 the whole
    # right-hand half of the blue curve is 2-point dashes at a constant
    # y = 202.2 pt. Dropping them cut that curve off at 52 % of the frame width
    # while leaving a chain long enough to pass every count test. Ticks cannot
    # get in here: they were rejected at PATH level in curves().
    frags = [(sorted(f), c) for f, c in frags if len(f) >= 2]
    frags.sort(key=lambda fc: fc[0][0][0])
    used = [False]*len(frags)
    out = []
    for i, (f, fcol) in enumerate(frags):
        if used[i]:
            continue
        used[i] = True
        cur = list(f)
        while True:
            ex, ey = cur[-1]
            # ⚠ EXTRAPOLATE, DO NOT JUST TAKE THE NEAREST y. The granularity
            # curves of a colour negative CROSS each other -- on 5245 the R and G
            # curves meet near the middle of the plot -- and at the crossing the
            # nearest-in-y rule can hand a fragment of one curve to the other and
            # continue happily. Two curves that cross have DIFFERENT SLOPES
            # there, so a local slope taken from the chain's own tail predicts
            # where its next dash must start and separates them. The tolerance is
            # applied to the residual against that prediction, not to raw dy.
            sl = 0.0
            tail = cur[-6:]
            if len(tail) >= 2 and abs(tail[-1][0] - tail[0][0]) > 1e-6:
                sl = (tail[-1][1] - tail[0][1]) / (tail[-1][0] - tail[0][0])
            best, bd = None, None
            for j, (g, gcol) in enumerate(frags):
                if used[j]:
                    continue
                # A colour sheet states layer identity in ink; two curves of
                # different colours are never one curve, whatever the geometry
                # says. On the black sheets every stroke has the same colour, so
                # this test is inert there -- verified by the unchanged numbers.
                if gcol != fcol:
                    continue
                gx, gy = g[0]
                # ⚠ THE LOWER BOUND IS NEGATIVE ON PURPOSE. 5279 draws one
                # granularity curve as three ABUTTING path objects
                # (…107.6 | 107.5… | 131.2…): the pieces share an endpoint, so the
                # measured x step at the junction is -0.1 pt, and a `>= 0.0` test
                # refused to join them -- which is why that sheet reported five
                # curves and was refused for weeks. Half a point of backward
                # tolerance admits a shared endpoint and nothing else.
                if not (-0.6 <= gx - ex <= DASH_GAP_PT):
                    continue
                res = abs(gy - (ey + sl*(gx - ex)))
                if res > DASH_Y_TOL_PT:
                    continue
                if bd is None or res < bd:
                    bd, best = res, j
            if best is None:
                break
            used[best] = True
            cur += list(frags[best][0])
        out.append((cur, fcol))
    return out


def split_families(polys, to_d, want=(3, 3)):
    """Curves -> (characteristic, granularity, gap in D). None if unclear.

    ⚠ THE DISCRIMINATOR IS VERTICAL SPAN, NOT MAXIMUM DENSITY, and the difference
    is the whole reliability of this function. Splitting on max density looked
    obvious and MEASURED BADLY: the gap between the two families came out
    0.32 / 0.27 / 0.18 D on the 5246 / 5274 / 5218 sheets, because a granularity
    curve that ends high and a red characteristic curve that tops out at 1.55 D
    are barely distinguishable by their maxima. Span separates them on every
    sheet in the corpus, because the two quantities are plotted on the SAME
    density axis while covering utterly different ranges: a characteristic curve
    must traverse the film's whole density scale, a granularity curve wanders
    within a few tenths.

        sheet | characteristic spans | granularity spans | gap
        5285  | 3.37 3.35 3.06       | 1.47 1.25 1.19    | 1.59
        5246  | 1.64 1.62 1.42       | 0.52 0.28 0.23    | 0.90
        5274  | 1.63 1.62 1.42       | 0.46 0.33 0.27    | 0.96
        5218  | 1.72 1.62 1.51       | 0.43 0.34 0.29    | 1.08

    Every gap is at least 0.90 D against a 0.60 D threshold, and the maximum-density
    rule would have failed three of the four.

    ⚠ THE GRANULARITY FAMILY IS NOT ALWAYS THREE CURVES. The 5219 brochure draws
    three characteristic curves but only TWO granularity curves -- the red
    record's granularity curve is simply not printed (verified by rendering the
    panel: three rising curves in blue / green / red, two flat ones in blue and
    green, no red flat curve anywhere). So the expected count per family is
    declared by the sheet, not assumed to be 3 + 3, and a sheet that produces the
    wrong TOTAL is still refused.
    """
    n_ch, n_gr = want
    if len(polys) != n_ch + n_gr:
        return None, None, 0.0
    def span(qc):
        ds = [to_d(y) for _, y in qc[0]]
        return max(ds) - min(ds)
    ranked = sorted(((span(q), i) for i, q in enumerate(polys)), reverse=True)
    gap = ranked[n_ch-1][0] - ranked[n_ch][0]
    ch = [polys[i] for _, i in ranked[:n_ch]]
    gr = [polys[i] for _, i in ranked[n_ch:]]
    return ch, gr, gap


def label_assign(pg, fr, ch, gr):
    """Assign the sheet's printed R/G/B letters to curves, per family.

    ⚠ NEAREST-LABEL-WINS IS NOT ENOUGH, and it cost four of the eight sheets.
    The first version of this function took each printed letter and gave it to
    the closest curve of either family. That resolves 5285 / 5246 / 5274 / 5218
    and silently loses letters on 5245 / 5248 / 5279, because on those sheets a
    granularity letter sits closer to a passing CHARACTERISTIC curve than to its
    own -- the two families share the plot area, and on 5279 one label triple is
    printed at the LEFT edge instead of the right. Nothing about the sheet is
    ambiguous; the greedy rule just threw away a constraint the sheet states.

    The constraint: a sheet prints SIX letters, two of each of R / G / B, one
    triple per family, against SIX curves already split into two families of
    three. So the assignment is a bijection under a known partition. This
    function enumerates it exactly -- which copy of each letter belongs to the
    characteristic family (2**3 = 8 ways) times the two within-family
    permutations (3! * 3! = 36) = 288 candidates -- and takes the one of least
    total label-to-curve distance. Cheap, exhaustive, no tolerance to tune.

    Verified: on the four sheets the greedy rule already resolved, this returns
    the IDENTICAL assignment. That is the point of replacing it this way -- a
    stricter method that changed a previously-accepted answer would mean one of
    the two is wrong, and would have to be settled before either was used.

    Returns ({"ch": {...}, "gr": {...}}, cost, runner_up_cost). The two costs are
    reported because their RATIO is the confidence: a sheet whose best fit is
    2 pt better than the next is telling you the labels are nearly equidistant.
    """
    labs = {}
    for w in pg.get_text("words"):
        if w[4] not in ("R", "G", "B"):
            continue
        lx, ly = (w[0]+w[2])/2.0, (w[1]+w[3])/2.0
        if not (fr.x0 <= lx <= fr.x1 and fr.y0 <= ly <= fr.y1):
            continue
        labs.setdefault(w[4], []).append((lx, ly))
    if sorted(labs) != ["B", "G", "R"] or any(len(v) != 2 for v in labs.values()):
        return None, 0.0, 0.0

    def dist(lab, curve):
        lx, ly = lab
        return min((px-lx)**2 + (py-ly)**2 for px, py in curve[0]) ** 0.5

    import itertools
    best = second = None
    for pick in itertools.product((0, 1), repeat=3):
        chl = {t: labs[t][p] for t, p in zip("RGB", pick)}
        grl = {t: labs[t][1-p] for t, p in zip("RGB", pick)}
        for pc in itertools.permutations(range(3)):
            for pgr in itertools.permutations(range(3)):
                c = sum(dist(chl[t], ch[pc[i]]) for i, t in enumerate("RGB"))
                c += sum(dist(grl[t], gr[pgr[i]]) for i, t in enumerate("RGB"))
                cand = (c, pc, pgr)
                if best is None or c < best[0]:
                    second, best = best, cand
                elif second is None or c < second[0]:
                    second = cand
    c, pc, pgr = best
    out = {"ch": {t: pc[i] for i, t in enumerate("RGB")},
           "gr": {t: pgr[i] for i, t in enumerate("RGB")}}
    return out, c, (second[0] if second else 0.0)


def colour_assign(ch, gr):
    """Layer identity from the STROKE COLOUR, for a sheet that prints no letters.

    ⚠ COLOUR IS PRINTED IDENTITY, NOT INFERENCE, and that distinction is the only
    reason this function is allowed to exist next to method rule 3. The KODAK
    VISION3 500T brochure draws its plot with one ink per record -- the blue,
    green and red curves ARE the sheet saying blue, green and red record -- while
    printing no R / G / B letters anywhere inside the frame. Refusing the sheet
    for want of letters would throw away the only INDEPENDENT check this file
    has: 5219's sigma(D) shape was traced once already, from the RASTER plot on
    the technical sheet, by a completely different extractor.

    Each curve's stroke colour is matched to whichever of pure red, green or blue
    it is nearest in RGB, and the mapping must come out one-to-one within each
    family, so an ambiguous palette is refused instead of guessed. A family may
    be short a layer (the brochure prints no red granularity curve at all); the
    layers it does print are still assigned.
    """
    IDEAL = {"R": (1.0, 0.0, 0.0), "G": (0.0, 1.0, 0.0), "B": (0.0, 0.0, 1.0)}
    out = {}
    for fam, curves_ in (("ch", ch), ("gr", gr)):
        got = {}
        for i, (_, col) in enumerate(curves_):
            if not col or len(col) < 3:
                return None
            best = min(IDEAL, key=lambda t: sum(
                (col[k]-IDEAL[t][k])**2 for k in range(3)))
            if best in got:
                return None            # two curves claim one record -- refuse
            got[best] = i
        out[fam] = got
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--overlay", metavar="PNG")
    ns = ap.parse_args()
    import pymupdf

    bad = 0
    for tag, spec in SHEETS.items():
        # ⚠ THE PER-SHEET SPEC IS OPTIONAL AND EXPLICIT. Seven of the eight sheets
        # are (file, page, profile) and take the defaults -- 3 + 3 curves, layer
        # identity from the printed letters. The eighth declares what it actually
        # prints. A default that silently absorbed both cases would hide exactly
        # the thing worth stating: the brochure omits a curve.
        fn, pgno, prof = spec[0], spec[1], spec[2]
        opt = spec[3] if len(spec) > 3 else {}
        n_want = opt.get("want", (3, 3))
        identity = opt.get("identity", "letters")
        pdf = Path(ns.root).resolve() / "PDF" / "PROFILES" / "KODAK" / fn
        if not pdf.is_file():
            print(f"  [SKIP] {tag}: source not present: {fn}")
            continue
        pg = pymupdf.open(pdf)[pgno-1]
        fr, fx, fd, fs, xs, dens, sig = frame_and_ticks(pg)
        print(f"[i] {fn} p{pgno}  frame {fr.width:.0f}x{fr.height:.0f} pt")
        for nm, f, n in ((("log-exposure", fx, len(xs)),) if fx else ()) + (
                         ("density", fd, len(dens)),
                         ("log sigma_D", fs, len(sig))):
            drop = ("" if not f[4] else "  DROPPED " + ", ".join(
                f"{10**v:.3f} ({r:.2f} pt off)" if nm == "log sigma_D"
                else f"{v:g} ({r:.2f} pt off)" for v, r in f[4]))
            print(f"    {nm:12s} {f[3]}/{n} ticks kept, "
                  f"worst residual {f[2]:.2f} pt{drop}")
        to_d = lambda py: (py - fd[1]) / fd[0]
        polys = curves(pg, fr)
        ch, gr, gap = split_families(polys, to_d, n_want)
        if ch is None:
            print(f"    [FAIL] {len(polys)} qualifying curves inside the frame, "
                  f"expected {n_want[0]}+{n_want[1]}")
            bad += 1
            continue
        # ⚠ THE GAP IS THE GATE. The families are split on maximum density, so
        # the split is only trustworthy if the two groups are clearly separated.
        # Measured 0.6-1.9 D on the seven sheets in this corpus; below 0.4 D the
        # sheet is refused rather than guessed at.
        # ⚠ COVERAGE IS PRINTED NEXT TO THE COUNT, method rule 19. A count test
        # alone accepted a curve's right-hand third as a whole curve on 5279 --
        # five whole curves plus one fragment satisfied "six curves" and the
        # missing two thirds were discarded in silence. The percentage of the
        # frame width each curve spans makes that visible in one line: curves
        # legitimately start at different x (Kodak draws each layer only where it
        # separates from its neighbours), so this is not asserted, but a 30 %
        # entry next to five 100 % ones is a stitching failure every time.
        cov = " ".join(
            "%.0f%%" % ((max(x for x, _ in q) - min(x for x, _ in q))
                        / fr.width * 100.0) for q, _c in ch + gr)
        print(f"    {len(polys)} curves ({n_want[0]} characteristic + {n_want[1]} "
              f"granularity), split on vertical SPAN with a {gap:.2f} D gap"
              f"\n    x coverage per curve (characteristic first): {cov}")
        if gap < 0.60:
            print("    [FAIL] families not clearly separated -- refusing to guess")
            bad += 1
            continue
        if identity == "colour":
            lab = colour_assign(ch, gr)
            if lab is None:
                print("    [FAIL] stroke colours do not map one-to-one onto "
                      "R / G / B -- layer identity cannot be read")
                bad += 1
                continue
            a_ch, a_gr = lab["ch"], lab["gr"]
            print(f"    colours: characteristic {a_ch}  granularity {a_gr}")
        else:
            lab, lcost, lnext = label_assign(pg, fr, ch, gr)
            if lab is None:
                print("    [FAIL] the sheet does not print two each of R / G / B "
                      "inside the frame -- layer identity cannot be read")
                bad += 1
                continue
            a_ch, a_gr = lab["ch"], lab["gr"]
            print(f"    labels: characteristic {a_ch}  granularity {a_gr}"
                  f"   (fit {lcost:.1f} pt, runner-up {lnext:.1f} pt)")

        to_x = lambda px: px          # pixel x IS the shared parameter; see above
        to_s = lambda py: 10.0 ** ((py - fs[1]) / fs[0])

        # ⚠ THE ADOPTED LEVEL IS NOT THE `mid` COLUMN BELOW. The per-layer column
        # printed for each record is sigma at ABSOLUTE density 1.0, which is what
        # the shape anchors are normalised to. `rms_granularity` means something
        # else: Kodak's own footnote on 5248 p1 and 5222 p1 reads "at a NET
        # diffuse visual density of 1.0", and C1d adopted the six 2026-08-18
        # triples at ONE exposure point -- the abscissa where the GREEN record
        # reaches its own dmin + 1.0. Recomputing that here rather than in a
        # throwaway snippet is what lets a NEW stock be adopted by the same route
        # the six were, instead of by a number nobody can re-derive.
        net = {}
        for layer in ("R", "G", "B"):
            if layer not in a_ch or layer not in a_gr:
                print(f"    {layer}: the sheet does not print this record in "
                      f"both families -- nothing to compose")
                continue
            cpoly = np.array(sorted(ch[a_ch[layer]][0]))
            gpoly = np.array(sorted(gr[a_gr[layer]][0]))
            cx, cd = to_x(cpoly[:, 0]), to_d(cpoly[:, 1])
            gx, gs_ = to_x(gpoly[:, 0]), to_s(gpoly[:, 1])
            # D is monotonic in x for a reversal film (falling); sort by D so the
            # composition sigma(D) can be interpolated in either direction
            o = np.argsort(cd)
            d_sorted, x_of_d = cd[o], cx[o]
            keep = np.concatenate(([True], np.diff(d_sorted) > 1e-9))
            d_sorted, x_of_d = d_sorted[keep], x_of_d[keep]
            og = np.argsort(gx)
            gx_s, gs_s = gx[og], gs_[og]
            k2 = np.concatenate(([True], np.diff(gx_s) > 1e-9))
            gx_s, gs_s = gx_s[k2], gs_s[k2]

            def sigma_at(D):
                if not (d_sorted[0] <= D <= d_sorted[-1]):
                    return float("nan")
                x = float(np.interp(D, d_sorted, x_of_d))
                if not (gx_s[0] <= x <= gx_s[-1]):
                    return float("nan")
                return float(np.interp(x, gx_s, gs_s))

            # ⚠ ANCHORS ARE CLAMPED TO THE OVERLAP OF THE TWO CURVES, and this
            # is not a convenience. Kodak draws each layer only over the range
            # where it is distinguishable from its neighbours, so the six curves
            # have SIX DIFFERENT left-hand starts: the R pair begins at log E
            # -3.0, the G pair at -2.5, the B pair at -2.1. Worse, the two curves
            # of a PAIR do not start at exactly the same x either -- G's
            # characteristic reaches its dmax at -2.54 while G's granularity curve
            # begins at -2.50. Asking for sigma at the characteristic curve's own
            # dmax therefore fell 0.04 decade off the end of the sigma curve and
            # returned NaN. The anchors below are the extreme densities at which
            # BOTH curves exist, and the density actually used is printed, so a
            # clamp can never be mistaken for the sheet's own endpoint.
            # Work in x, not in D. Round-tripping D -> x -> sigma fails at the
            # ends: the characteristic curve is FLAT near dmax, so many x values
            # share one density and np.interp returns the smallest of them --
            # which is exactly the one that falls off the sigma curve's start.
            # Evaluating at the overlap's own x avoids the ambiguity entirely.
            ox = np.sort(cx); od = cd[np.argsort(cx)]
            x_lo = max(float(ox.min()), float(gx_s[0]))
            x_hi = min(float(ox.max()), float(gx_s[-1]))
            sig_x = lambda x: float(np.interp(x, gx_s, gs_s))
            d_x = lambda x: float(np.interp(x, ox, od))
            ends = sorted([(d_x(x_lo), sig_x(x_lo)), (d_x(x_hi), sig_x(x_hi))])
            (dmin, s_lo), (dmax, s_hi) = ends[0], ends[1]
            # ⚠ RECORDED BEFORE THE ABSOLUTE-1.0 ANCHOR IS TESTED, deliberately.
            # A record can be unreadable at ABSOLUTE D 1.0 and perfectly readable
            # at the NET 1.0 exposure point, because the two are different
            # abscissae: 5248's blue pair only overlaps above D 1.0, yet its
            # adopted rms_b of 11.29 comes off this sheet. Registering the record
            # only after the absolute anchor succeeded silently dropped exactly
            # the layers the net read exists to recover.
            net[layer] = (ox, od, gx_s, gs_s, dmin)
            s_mid = sigma_at(1.0)
            if not np.isfinite(s_mid) or s_mid <= 0:
                print(f"    {layer}: D=1.0 is outside the overlap of the two "
                      f"curves (density {dmin:.2f}-{dmax:.2f}) -- no anchor")
                continue
            # ⚠ THE INTERIOR PEAK IS PART OF THE MEASUREMENT, not a detail. Every
            # sigma(D) curve in this corpus rises to a maximum somewhere between
            # dmin and D = 1.0 and falls away above it, so three anchors describe
            # a shape that is monotonic and the sheet's is not (schema v8 carries
            # sigma_shape_peak / _peak_at precisely for this). It is found by
            # scanning the composed curve on a dense x grid inside the overlap --
            # in x, not in D, for the same reason the anchors are.
            xs_grid = np.linspace(x_lo, x_hi, 400)
            sg = np.array([sig_x(x) for x in xs_grid])
            kpk = int(np.argmax(sg))
            s_pk, d_pk = float(sg[kpk]), d_x(float(xs_grid[kpk]))
            # ⚠ AND THE TOE ANCHOR SITS ON A PLATEAU. Below the toe the
            # characteristic curve is FLAT: density holds at dmin over a stretch
            # of x while sigma keeps changing, so sigma(D) is multivalued exactly
            # at the toe anchor's density and the value depends on which x of the
            # plateau you land on. This is the measured explanation of the one
            # real disagreement between the 5219 raster trace and the 5219
            # brochure vector trace (toe/mid 0.67 vs 0.40 while dmax/mid agreed
            # to 0.02): not two sheets contradicting each other, one ill-posed
            # anchor. The span of sigma over the plateau is printed so the
            # ambiguity is visible instead of implied.
            plate = [x for x in xs_grid if abs(d_x(x) - dmin) < 0.02]
            pl = ""
            if len(plate) > 4:
                lo = min(sig_x(x) for x in plate)*1000
                hi = max(sig_x(x) for x in plate)*1000
                if hi > 1.15*lo:
                    pl = (f"   [plateau] within 0.02 D of the toe anchor sigma "
                          f"spans {lo:.2f}-{hi:.2f} -- toe anchor is not unique")
            print(f"    {layer}: overlap D {dmin:.3f}..{dmax:.3f}  "
                  f"rms(=1000*sigma) {s_lo*1000:6.2f} / {s_mid*1000:6.2f} / "
                  f"{s_hi*1000:6.2f}  ->  toe/mid {s_lo/s_mid:.2f}  "
                  f"dmax/mid {s_hi/s_mid:.2f}   peak {s_pk*1000:6.2f} "
                  f"({s_pk/s_mid:.2f}x) at D {d_pk:.2f}{pl}")
            if layer == "G":
                pin = EXPECTED.get(tag)
                if pin:
                    got_toe, got_dmax = s_lo/s_mid, s_hi/s_mid
                    got_pk, got_pk_at = s_pk/s_mid, d_pk
                    if (abs(got_toe - pin["toe"]) > pin["tol"]
                            or abs(got_dmax - pin["dmax"]) > pin["tol"]):
                        print(f"    [FAIL] green triple moved: "
                              f"{got_toe:.2f}/1.00/{got_dmax:.2f} vs recorded "
                              f"{pin['toe']:.2f}/1.00/{pin['dmax']:.2f}")
                        bad += 1
                    # The peak carries the fourth anchor that schema v8 stores, so
                    # it is pinned on the same footing as the triple. Its DENSITY
                    # gets a wider window (0.05 D) than its height because a broad
                    # maximum moves in D under a one-pixel change in the fit while
                    # its height barely moves -- measured, not assumed: the 5219
                    # raster and vector traces put it at D 0.79 and 0.76.
                    if (abs(got_pk - pin["peak"]) > pin["tol"]
                            or abs(got_pk_at - pin["peak_at"]) > 0.05):
                        print(f"    [FAIL] interior peak moved: "
                              f"{got_pk:.2f}x at D {got_pk_at:.2f} vs recorded "
                              f"{pin['peak']:.2f}x at D {pin['peak_at']:.2f}")
                        bad += 1

        if "G" in net:
            gox, god, _, _, gdmin = net["G"]
            d_ref = gdmin + 1.0
            # a REVERSAL stock's density FALLS with exposure, so there is no "the
            # exposure where green reaches net 1.0" in the sense the negatives'
            # convention means. Refuse rather than interpolate a descending table,
            # which numpy will do silently and wrongly.
            # ⚠ THE TEST IS THE TREND, NOT STRICT MONOTONICITY. A bezier-sampled
            # curve wobbles by fractions of a point, so `all(diff >= 0)` was false
            # on both brochures -- i.e. on the two sheets that most needed the net
            # read -- and dropped them silently. Direction comes from the endpoints
            # and the interpolation is done on a density-sorted, de-duplicated
            # table, the same way the absolute anchor already does it.
            rising = bool(god[-1] > god[0])
            if rising and god.max() >= d_ref:
                _o = np.argsort(god)
                _ds, _xs = god[_o], gox[_o]
                _k = np.concatenate(([True], np.diff(_ds) > 1e-9))
                x_ref = float(np.interp(d_ref, _ds[_k], _xs[_k]))
                cells, own = [], []
                for layer in ("R", "G", "B"):
                    if layer not in net:
                        cells.append(f"{layer} --")
                        continue
                    ox_, od_, gxs, gss, dmn = net[layer]
                    v = (float(np.interp(x_ref, gxs, gss)) * 1000.0
                         if gxs[0] <= x_ref <= gxs[-1] else float("nan"))
                    cells.append(f"{layer} {v:6.2f}")
                    # each record read at ITS OWN net 1.0, for the 5 % agreement
                    # claim in RESULT_2026-08-18g -- printed so it stays checkable
                    if od_.max() >= dmn + 1.0:
                        xo = float(np.interp(dmn + 1.0, od_, ox_))
                        own.append(f"{layer} {float(np.interp(xo, gxs, gss))*1000:.2f}"
                                   if gxs[0] <= xo <= gxs[-1] else f"{layer} --")
                print(f"    NET 1.0 triple (green's own dmin {gdmin:.2f} + 1.0, "
                      f"one exposure point):  {'  /  '.join(cells)}")
                print(f"    each record at its own net 1.0:  {'  '.join(own)}")

        if ns.overlay:
            # ⚠ THE OVERLAY IS THE FIRST GATE, NOT THE LAST (method rule from the
            # VISION3 adoption, which took four attempts precisely because
            # internally-consistent numbers were produced from hybrid curves).
            # Every traced point is drawn back onto the rendered panel: the
            # characteristic family in one colour, the granularity family in
            # another, and a marker at each anchor actually used. If a trace
            # jumped families or a label was mis-assigned, it is visible here and
            # invisible in the numbers.
            import pymupdf as _f
            SC = 3.0                       # render scale, 216 dpi
            clip = _f.Rect(fr.x0-42, fr.y0-16, fr.x1+46, fr.y1+30)
            pix = pg.get_pixmap(matrix=_f.Matrix(SC, SC), clip=clip)
            import struct as _st, zlib as _zl
            w, h = pix.width, pix.height
            buf = bytearray(pix.samples)
            npx = pix.n
            def put(px, py, rgb):
                xi, yi = int(round((px-clip.x0)*SC)), int(round((py-clip.y0)*SC))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        X, Y = xi+dx, yi+dy
                        if 0 <= X < w and 0 <= Y < h:
                            o = (Y*w + X)*npx
                            buf[o:o+3] = bytes(rgb)
            for q, _c in ch:
                for px, py in q:
                    put(px, py, (220, 30, 30))          # characteristic: red
            for q, _c in gr:
                for px, py in q:
                    put(px, py, (20, 90, 230))          # granularity: blue
            rows = b"".join(b"\x00" + bytes(buf[y*w*npx:(y+1)*w*npx][:w*3])
                            for y in range(h))
            def chunk(tag, data):
                c = tag + data
                return (_st.pack(">I", len(data)) + c
                        + _st.pack(">I", _zl.crc32(c) & 0xffffffff))
            png = (b"\x89PNG\r\n\x1a\n"
                   + chunk(b"IHDR", _st.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                   + chunk(b"IDAT", _zl.compress(rows, 6))
                   + chunk(b"IEND", b""))
            out = ns.overlay if len(SHEETS) == 1 else \
                ns.overlay.replace(".png", f"_{tag}.png")
            Path(out).write_bytes(png)
            print(f"    overlay written to {out} "
                  f"(red = characteristic, blue = granularity)")
        if False:
            import pymupdf as _f
            pix = pg.get_pixmap(dpi=300, clip=_f.Rect(fr.x0-45, fr.y0-15,
                                                      fr.x1+40, fr.y1+25))
            pix.save(ns.overlay)
            print(f"    overlay written to {ns.overlay}")

    print()
    if bad:
        print(f"[FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("[OK] sigma(D) reproduced from the sheet's vector paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
