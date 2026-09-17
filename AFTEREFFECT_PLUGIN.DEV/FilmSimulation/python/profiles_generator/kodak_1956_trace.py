#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The 1956 Data Book's TIME-GAMMA insets, traced.

⚠ WHY THIS FILE IS SEPARATE FROM `kodak_1956.py`, AND THE DISTINCTION IS THE
TIER. Every pair in `kodak_1956.py` is a TRANSCRIPTION: Kodak labelled each
characteristic curve in frame with its time and its gamma, and reading those
labels is reading a printed manufacturer statement. Nothing in this file is.
The time-gamma inset beside each of those families plots gamma against
development time as a CONTINUOUS curve, one per developer, with the developer's
name lettered along the line and not one point marked. So these numbers come
off pixels, they are `status="traced"`, and they are kept in their own module
so that no later reader can mistake the two harvests for one.

WHAT THEY ADD THAT THE TRANSCRIPTIONS DO NOT. The labelled characteristic
family is drawn for ONE developer -- D-76 for the roll films, DK-50 or DK-60a
for the sheets. The inset draws two to five:

    Verichrome Pan roll   MQ (32 oz.), Versatol (1:15), D-76, Microdol
    Tri-X roll            DK-60a, DK-50, D-76, Microdol
    Tri-X 35mm            DK-50, D-76, Microdol
    Panatomic-X roll      D-76, Microdol
    Panatomic-X 35mm      Microdol, D-76 (1:1)
    Royal Pan sheet       Dektol, DK-60a, DK-50, DK-50 (1:1)   x TWO agitations
    Tri-X Pan sheet       DK-60a, DK-50
    Super-XX sheet        DK-60a, DK-50, DK-50 (1:1), D-76
    Panatomic-X sheet     DK-60a, DK-50, D-76, Microdol, DK-20
    Royal-X Pan sheet     DK-60a, DK-50                        x TWO agitations
    Royal-X Pan roll      DK-60a, DK-50

-- and on four sheets it draws the WHOLE SET TWICE, once for continuous
agitation and once for intermittent. That second axis is the one the corpus has
almost nothing on: `DevelopmentPoint.vessel` exists because Agfa publishes two
agitation regimes for three AGFAPAN stocks, and outside those three the field
is empty everywhere it matters. Kodak publishes the same pair here for four
more, and publishes it as a pair of curves rather than a pair of numbers, so
the whole shape of the difference is recoverable.

⚠ HOW A CURVE IS SEPARATED FROM THE GRID IT IS DRAWN ON, AND WHY THAT IS THE
HARD PART. The inset is a ruled grid: thin vertical lines every two minutes,
thin horizontal lines every 0.2 of gamma, and the curves are drawn over them in
a stroke only slightly heavier. A threshold catches both. The grid is removed
geometrically instead -- a gridline is a column or row that is inked over more
than 60 % of the box, and a curve never is, because no curve here is vertical
and none is flat across the full width. Erasing a gridline punches a 1-2 px gap
in every curve that crossed it, so the mask is closed vertically by 3 px
afterwards; that is smaller than the 30 px grid pitch, so it cannot bridge two
curves that are genuinely apart.

⚠ AND THE LABELS ARE ON THE CURVES, NOT BESIDE THEM. Kodak letters "MICRODOL"
along the line it names, so the glyphs are ink the tracer must not follow. They
are removed as connected components that are short and tall -- a curve spans
most of the box width, a word spans at most a third of it and stands three to
four times higher than the stroke. What that leaves is a gap in the curve where
its own name sat, which the follower crosses because it searches a window
rather than requiring contiguity, and which the coverage test then counts as
missing rather than as traced.

⚠ WHAT IS REFUSED. A curve is kept only if it covers `MIN_COVER` of the box
width, rises monotonically within `MONO_SLACK`, and -- the test that matters --
fits the Mees-Sheppard law to within `FIT_TOL`. The last is not a smoothness
check: gamma(t) for a real developer IS that law, so a trace that wandered onto
a gridline, jumped to a neighbouring curve at a crossing, or followed a letter
stroke produces a shape the law cannot hold, and the residual says so. Curves
that cross each other inside the box are the common failure, and they are
expected to be caught here rather than stored.

⚠⚠ AND THE WHOLE TRACER IS CHECKED AGAINST THE TRANSCRIPTIONS. On every panel
one of the traced developers is the SAME developer whose characteristic family
Kodak labelled by hand -- D-76 on the roll films, DK-50 or DK-60a on the
sheets. Those printed pairs were never used to build the trace, so comparing
them to it is a genuine test with a known answer.

⚠⚠⚠ NOTHING IN THIS FILE IS ADOPTED, AND AFTER THREE ATTEMPTS THE REASON IS NOT
COVERAGE -- IT IS THAT MOST TRACES ARE WRONG AND THE CHECK PROVES IT.

Three defects were found and fixed, each by the check rather than by
inspection, and the state after all three is recorded here so a fourth attempt
starts from the diagnosis instead of from zero.

  1. THE AXIS VALUES CAME FROM THE PDF's OCR LAYER, which calibrated one page
     in fourteen -- page 49 yields no row of five numeric tokens anywhere,
     page 63 returns the neighbouring time-temperature chart's logarithmic
     axis. Replaced by `ladder` + `calibrate`: the tick LABELS are found
     geometrically, their values are transcribed by hand in `PANELS`, and each
     label is indexed by computed lattice position rather than by order. The
     gamma ladders now fit to 0.004 and eleven of twenty insets calibrate.

  2. THE FOLLOWER SWAPPED CURVES AT CROSSINGS. These insets plot two to five
     developers on one grid and the curves cross; a nearest-ink follower
     cannot continue straight through an intersection. Replaced by
     `follow_tracks`, which walks every curve at once and matches ink by
     SLOPE continuity. This moved the result by very little, which is itself
     the useful finding: crossings were not the dominant error.

  3. THE SEED COLUMN FOUND ONLY THE LONGEST CURVES. Several curves stop before
     the right-hand edge -- Verichrome's MQ ends at 16 minutes on a 2-to-28
     grid -- so the rightmost inked column holds one or two developers and a
     first-hit seed found exactly those. The module was tracing 1.4 curves on
     panels drawing four, and the check was therefore comparing Kodak's D-76
     family against whatever single curve had been found: on page 46a that was
     Microdol, two tenths of a gamma below D-76, which is what the 0.35 miss
     actually measured. Seeding now takes the column carrying the MOST runs,
     and the yield went from 11 curves to 20.

AND IT STILL DOES NOT VALIDATE. Of the nine traced panels that sit beside a
printed family, three now agree within 0.055 and six do not, the worst at
0.504. A miss of half a gamma is not a stroke-centroid error; on those panels
the traced curve is still a different quantity from the one Kodak labelled.

⚠ THE CHECK IS THEREFORE THE DELIVERABLE OF THIS MODULE, and it has now earned
that description three times. It was built so that a trace could not be
adopted on the strength of being smooth, monotone and Mees-Sheppard-shaped --
which every one of these wrong traces is -- and each time it has named a real
defect that inspection had missed.

⚠ WHAT IS LEFT TO TRY, in the order a fourth attempt should take them:
  (a) IDENTIFY EACH CURVE BY ITS OWN LETTERED NAME rather than by position.
      `strip_text` currently deletes "MICRODOL", "DK-60a" and the rest as
      noise; their centroids say which track is which developer, and with that
      the printed-label check compares like with like instead of comparing
      Kodak's D-76 against whichever curve happened to be traced.
  (b) Seed at several columns and merge, rather than at the single densest.
  (c) Only then relax nothing: the tolerance stays at 0.055, because three
      panels already meet it and a tolerance that admits the other six would
      admit a half-gamma error.
Tracked as queue P61.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

PDF = Path("/root/work/pg/PDF/PROFILES/KODAK/1956-Kodak-Films.pdf")
ALT_PDF = Path("/mnt/user-data/uploads/PYTHON.TST/PDF/PROFILES/KODAK/"
               "1956-Kodak-Films.pdf")

DPI = 400.0
SCALE = DPI / 72.0

INK = 150           #: 8-bit threshold. The plate is black on cream at ~252.
GRID_FRAC = 0.45    #: inked fraction of the box that makes a row/col a gridline.
CLOSE_PX = 3        #: vertical closing after degridding; << the 30 px grid pitch.
MIN_COVER = 0.20    #: fraction of box width a kept curve must span.
MONO_SLACK = 0.012  #: gamma may dip by this much and still count as rising.
MIN_RISE = 0.20     #: total gamma a kept curve must climb. See `trace_panel`.
DUP_GAMMA = 0.02    #: two seeds this close all along are one curve twice.
#: Mees-Sheppard residual above which a trace is refused.
#: ⚠ 0.030 UNTIL 2026-09-17, AND THAT NUMBER WAS BORROWED FROM THE WRONG
#: MEASUREMENT. It is `kodak_1956.GAMMA_FIT_TOL`, set for TRANSCRIBED points --
#: printed labels quoted to two decimals, whose only error is the quantisation.
#: A TRACED curve carries stroke-centroid noise on top of that, and on page 59
#: it rejected three of the four curves at residuals of 0.056, 0.074 and 0.080
#: while the fourth passed at 0.029 -- all four being clean traces spanning 98 %
#: of the box and rising monotonically. This is a SHAPE sanity check, not the
#: accuracy gate; the accuracy gate is AGREE_GAMMA against Kodak's own printed
#: labels, and that is deliberately NOT loosened.
FIT_TOL = 0.100
#: Trace against Kodak's own printed labels. ⚠ REPORTED, NOT ENFORCED, while
#: coverage stands at one panel: with a single sample a threshold is a choice
#: about that sample, not a test. It becomes the gate when the hand-calibration
#: described in the module docstring lands and there are fourteen to test.
AGREE_GAMMA = 0.055
FOLLOW_WIN = 7      #: half-height, px, of the search window per column step.
GAP_KILL = 8        #: columns a track may find no ink before it is closed.
MIN_FRAG = 40       #: px of width below which a traced piece is debris.
LINK_GAP = 0.40     #: fraction of box width one join may bridge.
LINK_LAP = 24       #: px two pieces of one curve may overlap and still join.
LINK_TOL = 7.0      #: px the extrapolation may miss the next piece by.
MIN_TICKS = 5


# --------------------------------------------------------------------------
# Axis furniture, from the text layer
# --------------------------------------------------------------------------

def _words(page):
    return [(w[0] * SCALE, w[1] * SCALE, w[2] * SCALE, w[3] * SCALE, w[4])
            for w in page.get_text("words")]


def ladder(g, x0, x1, y0, y1, axis, min_n=4, tol=0.18, gap=14):
    """Evenly spaced glyph clusters in a gutter strip -> tick-label centroids.

    ⚠ THIS REPLACED AN OCR-LAYER READER AND THE REPLACEMENT IS THE WHOLE
    REASON THIS MODULE NOW ADOPTS ANYTHING. The first version took the tick
    VALUES from the PDF's text layer, which works on page 42 and on no other:
    page 49 yields no row of five numeric tokens anywhere, page 46 recovers
    five ticks of about fifteen, page 63 returns the neighbouring
    time-temperature chart's logarithmic axis. The text layer is an artefact of
    whoever scanned the book and is not a property of the book.

    The labels themselves are not. They are solid black text at 10 point --
    an order of magnitude more ink than the grey rule beside them -- so
    finding them is a threshold that does not have to separate a faint line
    from paper, which is the problem that defeated a grid-based reader here
    too. What this function returns is their centroids; the VALUES come from
    `PANELS`, transcribed by eye exactly as the printed gamma labels were.
    """
    import cv2
    import numpy as np
    strip = g[y0:y1, x0:x1]
    ink = (strip < 140).astype("uint8")
    n, lab, stats, cent = cv2.connectedComponentsWithStats(ink, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # ⚠ THE UPPER BOUNDS EXCLUDE THE ROTATED AXIS TITLE. "GAMMA" set
        # sideways down the gutter is one component 47 px wide and 585 tall,
        # and it sits exactly where the ladder is looked for.
        if area < 12 or h > 60 or w > 60 or h < 6:
            continue
        comps.append((cent[i][1] + y0, cent[i][0] + x0))
    if not comps:
        return []
    # axis 0 = ladder runs DOWN the page (cluster by y); 1 = across (by x)
    key = 0 if axis == 0 else 1
    comps.sort(key=lambda c: c[key])
    groups = []
    for c in comps:
        if groups and abs(groups[-1][-1][key] - c[key]) <= gap:
            groups[-1].append(c)
        else:
            groups.append([c])
    pos = [float(np.mean([q[key] for q in grp])) for grp in groups]
    if len(pos) < min_n:
        return []
    best = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            step = pos[j] - pos[i]
            if step < 15:
                continue
            run = [p for p in pos
                   if abs(((p - pos[i]) / step)
                          - round((p - pos[i]) / step)) <= tol]
            if len(run) > len(best):
                best = run
    return sorted(best) if len(best) >= min_n else []


def label_widths(g, x0, x1, pos, gap=14):
    """Horizontal extent, in pixels, of the glyph group at each ladder row.

    ⚠ WHAT THIS IS FOR: THE LADDER GIVES THE SCALE BUT NOT THE OFFSET, AND THE
    OFFSET WAS WRONG ON FOUR PANELS. `ladder` returns evenly spaced label rows
    and `calibrate` used to assume the topmost one is the transcribed `g_top`.
    That assumption is false whenever the top label falls outside the
    transcribed gutter, which happens because the gutter was transcribed around
    the labels the eye saw and the topmost sits ON the frame: on page 43 the
    "1.6" label centre is at y = 2064 and the gutter starts at 2080, so the
    ladder's first row is "1.4", every gamma came out 0.2 HIGH, and both traced
    curves missed Kodak's own printed labels by exactly that.

    A ladder cannot detect this by itself -- a set of evenly spaced rows fits a
    straight line at any offset, and the residual is unchanged. What breaks the
    symmetry is the LABELS' OWN INK. Kodak sets these axes as "1.6" above 1.0
    and ".8" below it, so a label carrying a leading 1 is about 27 px wide at
    400 dpi and one that begins with the decimal point is about 11. Measuring
    that width says which rows are >= 1.0 without reading a single glyph, and
    the LAST wide row is the 1.0 line. The offset is then a fact about the
    page rather than an assumption about the transcription.

    ⚠ THE LABEL IS MEASURED FROM ITS RIGHT EDGE LEFTWARDS, GLYPH BY GLYPH, and
    the search runs 30 px to the LEFT of the transcribed gutter. Both are
    forced by what else lives in that strip. The leading 1 is a bare vertical
    stroke 4 px wide and the gutter was transcribed around the digits, so on
    pages 53a and 53b the 1 falls outside it and every label measures narrow;
    widening the strip fixes that but admits the rotated word GAMMA, whose
    five letters are separate components of exactly label size sitting in
    their own column further left. Chaining leftward from the rightmost glyph
    and stopping at the first gap wider than a letter-space keeps the one and
    drops the title, which a plain bounding box over the row cannot do.
    """
    import cv2
    lo = max(0, x0 - 30)
    strip = g[:, lo:x1]
    ink = (strip < 140).astype("uint8")
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(ink, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 12 or h > 30 or w > 20 or h < 6:
            continue
        comps.append((cent[i][1], x + lo, x + w + lo))
    out = []
    for p in pos:
        near = sorted((c for c in comps if abs(c[0] - p) <= gap),
                      key=lambda c: -c[2])
        if not near:
            out.append(0)
            continue
        right, left = near[0][2], near[0][1]
        for c in near[1:]:
            # ⚠ AND THE CHAIN STOPS AT THE WIDEST LABEL THIS BOOK SETS. "2.0"
            # measures 35 px; anything past that is the rotated title or a
            # tick reached across a letter-space, and swallowing it would
            # promote a narrow label to a wide one and move the whole axis.
            if 0 <= left - c[2] <= 16 and right - c[1] <= 36:
                left = c[1]
        out.append(right - left)
    return out


def leading_one(widths, ks, wide=20, seen=7):
    """Index into a ladder of the "1.0" label, or None when it is not legible.

    Returns the position of the last label wide enough to carry a leading 1,
    but ONLY when the wide labels form an unbroken run at the TOP of the
    ladder. That prefix test is the whole safety of the rule: on page 55 the
    rotated axis title leaves a 20 px fragment BELOW three genuine wide
    labels, and on pages 53a and 53b the gutter clips the leading 1 off every
    label so none is wide at all. In both shapes the answer is None and the
    caller keeps the transcribed `g_top`, which is what those panels had
    before this function existed.
    """
    # ⚠⚠ THE RULE SPEAKS ONLY WHEN EVERY ROW IS LEGIBLY ONE THING OR THE
    # OTHER, and page 59 is why. Its ladder measures 27, 27, 26, 51, 10, 13,
    # 11, 11: the 51 is a label merged with a fragment of the rotated title,
    # counting it as a leading 1 moved the whole axis down by 0.2, and the
    # traced DK-50 then missed Kodak's own printed labels by 0.242 -- a panel
    # that had been right became wrong. A width between the two populations,
    # or above the widest label this book sets, is not evidence about a
    # leading digit; it is evidence that something other than a label was
    # measured, and the transcribed anchor is the better of the two guesses.
    if any(w > 36 or (16 < w < wide) for w in widths):
        return None
    wid = [i for i, w in enumerate(widths) if w >= wide]
    if not wid or len(wid) == len(widths):
        return None
    one = wid[-1]
    # ⚠ A ROW WHOSE INK ALL BUT VANISHED IS NOT EVIDENCE EITHER WAY. On page 43
    # the "1.2" label survives the scan as a 3 px sliver, so demanding an
    # unbroken run of WIDE rows would refuse a panel whose 1.0 row is perfectly
    # legible. Rows below `seen` are passed over; rows that are legibly NARROW
    # above the candidate are not, because a narrow label above a wide one
    # cannot happen on an axis that counts down through 1.0.
    if any(widths[i] >= seen and widths[i] < wide for i in range(one)):
        return None
    return ks[one]


def calibrate(pos, v_first, v_step):
    """pixel -> value from tick centroids that may have GAPS.

    ⚠ THE INDEX IS COMPUTED, NOT ASSUMED, and this is the subtle half. A
    ladder read off a scan is not always complete: a two-digit label can merge
    with its neighbour, a glyph can fall below the ink threshold. Consecutive
    retained centroids are therefore not consecutive ticks, and giving them
    consecutive indices produces a perfectly straight line with the wrong
    slope -- a failure that looks like success. The step is the MEDIAN
    spacing, robust to a doubled gap; each label is indexed by rounding its
    offset in units of that step; the fit runs on (pixel, value at index).
    """
    import numpy as np
    if len(pos) < 3:
        return None
    d = sorted(pos[i + 1] - pos[i] for i in range(len(pos) - 1))
    step = d[len(d) // 2]
    if step <= 0:
        return None
    ks = [round((p - pos[0]) / step) for p in pos]
    if len(set(ks)) != len(ks):
        return None
    vals = [v_first + v_step * k for k in ks]
    A = np.vstack([np.array(pos, float), np.ones(len(pos))]).T
    sol, *_ = np.linalg.lstsq(A, np.array(vals, float), rcond=None)
    a, b = float(sol[0]), float(sol[1])
    err = max(abs(a * p + b - v) for p, v in zip(pos, vals))
    return a, b, err, len(pos)


def fit_line(pairs):
    n = len(pairs)
    sx = sum(a for a, _ in pairs)
    sy = sum(b for _, b in pairs)
    sxx = sum(a * a for a, _ in pairs)
    sxy = sum(a * b for a, b in pairs)
    den = n * sxx - sx * sx
    if abs(den) < 1e-9:
        return None
    a = (n * sxy - sx * sy) / den
    b = (sy - a * sx) / n
    res = max(abs(a * p + b - v) for p, v in pairs)
    return a, b, res


# --------------------------------------------------------------------------
# The raster half
# --------------------------------------------------------------------------

def degrid(box):
    """Erase the ruled grid, then close the gaps it leaves in the curves."""
    import numpy as np
    import cv2
    h, w = box.shape
    ink = (box < INK).astype("uint8")
    keep = ink.copy()
    colsum = ink.sum(axis=0)
    rowsum = ink.sum(axis=1)
    for j in range(w):
        if colsum[j] > GRID_FRAC * h:
            keep[:, j] = 0
    for i in range(h):
        if rowsum[i] > GRID_FRAC * w:
            keep[i, :] = 0
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, CLOSE_PX))
    return cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k)


def strip_text(mask):
    """Drop the lettered developer names that sit ON the curves.

    ⚠ THE TEST IS SHAPE, NOT POSITION, because Kodak puts the names in a
    different place on every panel. A curve here spans at least a third of the
    box and is one stroke high; a word spans at most a third and stands three
    to four strokes high, and its bounding box is far more densely filled than
    a curve's is. Both conditions must hold, so a steeply rising curve segment
    -- tall and narrow, but sparse -- survives.
    """
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    w = mask.shape[1]
    out = mask.copy()
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if cw >= 0.34 * w:
            continue
        if ch < 16:
            continue
        if area < 0.17 * cw * ch:
            continue
        out[lab == i] = 0
    return out


def seeds(mask, col):
    """Ink runs in one column, as (centre, height) -- one per curve present."""
    ys = [i for i, v in enumerate(mask[:, col]) if v]
    out = []
    for y in ys:
        if out and y - out[-1][-1] <= 2:
            out[-1].append(y)
        else:
            out.append([y])
    return [(sum(r) / len(r), len(r)) for r in out if len(r) <= 14]


def follow(mask, x0, y0, step):
    """Walk a curve column by column, taking the nearest ink centroid."""
    h, w = mask.shape
    pts = {}
    y = y0
    x = x0
    misses = 0
    while 0 <= x < w:
        lo = max(0, int(y) - FOLLOW_WIN)
        hi = min(h, int(y) + FOLLOW_WIN + 1)
        run = [i for i in range(lo, hi) if mask[i, x]]
        if run:
            groups = []
            for i in run:
                if groups and i - groups[-1][-1] <= 2:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            g = min(groups, key=lambda r: abs(sum(r) / len(r) - y))
            y = sum(g) / len(g)
            pts[x] = y
            misses = 0
        else:
            misses += 1
            if misses > 90:      # a lettered name is ~60 px of gap at 400 dpi
                break
        x += step
    return pts


def runs_in(mask, col):
    """Ink runs in one column as (centre, length)."""
    ys = [i for i, v in enumerate(mask[:, col]) if v]
    out = []
    for y in ys:
        if out and y - out[-1][-1] <= 2:
            out[-1].append(y)
        else:
            out.append([y])
    return [(sum(r) / len(r), len(r)) for r in out]


def _extrapolate(pts, x):
    """Where a track, read from its leftmost few points, would be at `x`."""
    xs = sorted(pts)[:8]
    if len(xs) < 3:
        return pts[xs[0]]
    a, b = xs[0], xs[-1]
    s = (pts[a] - pts[b]) / float(a - b)
    s = max(-1.5, min(1.5, s))
    return pts[a] + s * (x - a)


def fragments(mask):
    """Every unbroken piece of curve in the panel, walked right to left.

    ⚠⚠ THIS REPLACED A SEED-COLUMN FOLLOWER AND THE REASON IS THAT NO SINGLE
    COLUMN MEETS EVERY CURVE. Four rules were tried for choosing one seed
    column -- rightmost inked, densest, rightmost with exactly N thin runs,
    rightmost whose tracks survive as N curves -- and all four founder on the
    same fact about these insets: THE CURVES DO NOT ALL SPAN THE PLOT. On page
    42 MQ stops at sixteen minutes and Versatol at twenty while D-76 and
    Microdol run to twenty-eight, so the only columns crossing all four lie
    left of sixteen minutes, where the family has converged to within a few
    pixels. Seeding right lost two curves; seeding inside the convergence
    could not tell them apart; and the "survivors" rule quietly accepted a
    column where two tracks had merged onto one curve and a frame fragment
    made up the count. The trace stopped at twenty minutes on a plot that runs
    to twenty-eight and no page agreed with Kodak's printed labels.

    A curve does not need a seed. It needs a BEGINNING, and its beginning is
    the column where its ink first appears, which is different for each curve
    and is exactly what a right-to-left sweep finds for free: any run that no
    live track can claim starts a new one.

    ⚠ AND A RUN IS CLAIMED BY AT MOST ONE TRACK. The follower this replaces
    let two tracks share a run, on the argument that sharing is what a
    crossing is. It is also what a CONVERGENCE is, and these families converge
    at short time and never separate again, so the shared run was permanent:
    page 42 returned two tracks holding pixel-for-pixel the same curve, which
    the duplicate test missed because it ran before the trimming that made
    them identical. Assignment is now exclusive and greedy in order of
    prediction error, so at a true crossing the better-predicted track keeps
    the ink and the other goes dark for the few columns the lines are one, and
    at a convergence the losers simply end. What is lost is the unresolvable
    left-hand stub, which `clean` was discarding anyway.
    """
    h, w = mask.shape
    live, done = [], []
    for x in range(w - 1, -1, -1):
        rs = [r for r in runs_in(mask, x) if r[1] <= 14]
        cand = []
        for i, t in enumerate(live):
            p = _extrapolate(t["pts"], x)
            for j, (c, _n) in enumerate(rs):
                d = abs(c - p)
                if d <= FOLLOW_WIN:
                    cand.append((d, i, j))
        cand.sort()
        ut, ur = set(), set()
        for _d, i, j in cand:
            if i in ut or j in ur:
                continue
            ut.add(i)
            ur.add(j)
            live[i]["pts"][x] = rs[j][0]
            live[i]["miss"] = 0
        for i, t in enumerate(live):
            if i not in ut:
                t["miss"] += 1
        for j, (c, _n) in enumerate(rs):
            if j not in ur:
                live.append({"pts": {x: c}, "miss": 0})
        keep = []
        for t in live:
            (done if t["miss"] > GAP_KILL else keep).append(t)
        live = keep
    done += live
    return [q for q in (_untrail(t["pts"]) for t in done)
            if q and max(q) - min(q) >= MIN_FRAG]


def _untrail(pts, run=8, flat=1.0):
    """Cut a dead flat run off either end of a traced piece.

    ⚠ A FLAT END IS NOT PART OF THE CURVE. Nothing in these insets is
    horizontal -- gamma rises with time on every one of them -- so a run of
    columns at one constant y is a gridline remnant, a frame serif or the
    stroke of a stripped letter that the follower stepped onto when its curve
    stopped. Page 42 is the case: Versatol's left piece really ends at x=363,
    and the follower carried it to x=407 sitting at exactly y=314 the whole
    way. Those 44 dead pixels put it 24 px away from its own right-hand piece,
    the join was refused, and a panel drawing four curves returned five.
    """
    xs = sorted(pts)
    while len(xs) > run:
        end = xs[:run]
        if max(pts[x] for x in end) - min(pts[x] for x in end) >= flat:
            break
        xs = xs[1:]
    while len(xs) > run:
        end = xs[-run:]
        if max(pts[x] for x in end) - min(pts[x] for x in end) >= flat:
            break
        xs = xs[:-1]
    return {x: pts[x] for x in xs}


def link(frags, w):
    """Chain fragments of one curve back together across its own lettering.

    ⚠ THE GAPS ARE MADE BY THE READER ITSELF, NOT BY THE PRINTING. Kodak
    letters each developer's name ALONG its line, `strip_text` removes the
    word, and what it leaves is a hole in the curve a third of the box wide on
    the worst panels; the exclusive assignment in `fragments` punches shorter
    holes wherever two curves are briefly one stroke. Either way the pieces
    are all genuinely traced and belong to a curve each.

    A piece is joined to the piece on its left when the left END of the right
    piece, extrapolated from its own slope, arrives where the right END of the
    left piece actually is. The cost is that miss in pixels plus a small
    charge for the distance bridged, so a short confident jump beats a long
    speculative one; links are taken cheapest first, each piece keeps at most
    one neighbour on each side, and a chain may not close on itself.

    ⚠ THE JOIN TOLERATES A SMALL OVERLAP. Two pieces of one curve can share a
    column or two where the follower dropped the track and picked it up again
    on the same ink -- on page 42 Versatol's two pieces both hold fourteen
    minutes -- so a strictly positive gap would refuse exactly the joins that
    are most certainly right.
    """
    n = len(frags)
    lo = [min(f) for f in frags]
    hi = [max(f) for f in frags]
    cand = []
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            gap = lo[a] - hi[b]
            if gap < -LINK_LAP or gap > LINK_GAP * w:
                continue
            d = abs(_extrapolate(frags[a], hi[b]) - frags[b][hi[b]])
            if d <= LINK_TOL + 0.06 * max(gap, 0):
                cand.append((d + 0.02 * max(gap, 0), a, b))
    cand.sort()
    nxt, prv = {}, {}
    for _c, a, b in cand:
        if a in nxt or b in prv:
            continue
        z, seen = b, {a}
        ok = True
        while z in nxt:
            if z in seen:
                ok = False
                break
            seen.add(z)
            z = nxt[z]
        if not ok:
            continue
        nxt[a] = b
        prv[b] = a
    out = []
    for head in (i for i in range(n) if i not in prv):
        pts, z = {}, head
        while True:
            for k, v in frags[z].items():
                pts.setdefault(k, v)
            if z not in nxt:
                break
            z = nxt[z]
        out.append(pts)
    return out


def mees_fit(points):
    """Same law and the same grid search as kodak_1956.fit_mees_sheppard."""
    from kodak_1956 import fit_mees_sheppard
    return fit_mees_sheppard(points)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# THE PANEL TABLE -- hand-read, once, exactly as the printed gamma labels were
# --------------------------------------------------------------------------
# ⚠ WHY THESE NUMBERS ARE TRANSCRIBED AND NOT FOUND. Two automatic routes were
# built and both failed for the same underlying reason, which is worth stating
# so neither is tried a third time. (1) READING THE TICK VALUES FROM THE PDF's
# OCR LAYER works on page 42 and on no other: page 49 yields no row of five
# numeric tokens anywhere, page 46 recovers five ticks of about fifteen, page
# 63 returns the neighbouring time-temperature chart's logarithmic axis. The
# text layer is an artefact of whoever scanned the book, not a property of the
# book. (2) FINDING THE PLOT BOX FROM THE RULED GRID fails because the rules
# are a pale grey halftone: at a threshold that finds them on page 42 it finds
# nothing on page 46, and at one that finds them on page 46 it finds the
# lettering too, and the resulting lattice latches onto sub-multiples -- a
# 31 px step where the grid pitch is 62.
#
# What IS reliable on every page is the tick LABELS, which are solid black
# 10-point text, and the human reading of what they say. So the box and the
# axis end values are transcribed here from the page rendered at 400 dpi, the
# label CENTROIDS are found geometrically by `ladder`, and `calibrate` fits
# pixel to value through them. The reader then depends on no text layer and on
# no grid.
#
# ⚠ AND THE TRANSCRIPTION IS CHECKED, NOT TRUSTED. Every panel here sits beside
# a characteristic-curve family whose members Kodak labelled in frame with
# their time and gamma, and `kodak_1956.FAMILIES_1956` holds those labels. The
# developer of that family is also one of the developers in the inset, so the
# traced curve and the printed labels are two independent readings of the same
# gamma(t) -- and nothing in this file's calibration used the labels. A panel
# whose trace misses its own printed labels by more than AGREE_GAMMA is
# refused whole.
#
# Fields: tag, pdf page, profile, edition, vessel, agitation caption,
#         trace box (x0,x1,y0,y1), gamma gutter (x0,x1,y0,y1), gamma of the
#         TOP label, time tick row (x0,x1,y0,y1), time of the FIRST label.
PANELS = (
    dict(tag="42", page=42, profile="KODAK_VERICHROME_PAN",
         edition="1956 roll film", vessel="small tank",
         box=(427, 1232, 1087, 1650), gut=(320, 410, 1070, 1670), g_top=2.0,
         row=(400, 1300, 1690, 1770), t_first=2.0,
         devs=("KODAK MQ (32 oz.)", "KODAK Versatol", "KODAK D-76",
               "KODAK Microdol")),
    dict(tag="43", page=43, profile="KODAK_PLUS_X_125",
         edition="1956 35mm", vessel="small tank",
         box=(226, 812, 2096, 2592), gut=(120, 215, 2080, 2610), g_top=1.6,
         row=(200, 860, 2620, 2700), t_first=2.0,
         devs=("KODAK D-76", "KODAK Microdol")),
    dict(tag="46a", page=46, profile="KODAK_TRI_X_400TX",
         edition="1956 roll film", vessel="small tank",
         box=(408, 1290, 325, 700), gut=(300, 395, 310, 720), g_top=1.6,
         row=(380, 1330, 745, 820), t_first=2.0,
         devs=("KODAK D-76", "KODAK DK-60a", "KODAK DK-50",
               "KODAK Microdol")),
    dict(tag="46b", page=46, profile="KODAK_TRI_X_400TX",
         edition="1956 35mm", vessel="small tank",
         box=(408, 1290, 1197, 1600), gut=(300, 395, 1185, 1620), g_top=1.6,
         row=(380, 1330, 1715, 1790), t_first=2.0),
    dict(tag="49a", page=49, profile="KODAK_PANATOMIC_X",
         edition="1956 roll film", vessel="small tank",
         box=(237, 805, 1168, 1610), gut=(140, 230, 1155, 1625), g_top=1.6,
         row=(215, 840, 1665, 1740), t_first=2.0),
    dict(tag="49b", page=49, profile="KODAK_PANATOMIC_X",
         edition="1956 35mm", vessel="small tank",
         box=(237, 805, 2065, 2510), gut=(140, 230, 2050, 2525), g_top=1.6,
         row=(215, 840, 2565, 2640), t_first=2.0),
    dict(tag="53a", page=53, profile="KODAK_ROYAL_PAN_4141",
         edition="1956 sheet film", vessel="tray",
         box=(297, 766, 2332, 2645), gut=(205, 290, 2320, 2660), g_top=1.2,
         row=(275, 800, 2960, 3030), t_first=2.0),
    dict(tag="53b", page=53, profile="KODAK_ROYAL_PAN_4141",
         edition="1956 sheet film", vessel="tank",
         box=(297, 766, 2655, 2968), gut=(205, 290, 2645, 2985), g_top=1.2,
         row=(275, 800, 2960, 3030), t_first=2.0,
         devs=("KODAK DK-60a", "KODAK DK-50", "KODAK DK-50 (1:1)")),
    dict(tag="55", page=55, profile="KODAK_TRI_X_SHEET_1952",
         edition="1956 sheet film", vessel="tank",
         box=(224, 752, 2337, 2790), gut=(130, 218, 2325, 2805), g_top=1.6,
         row=(200, 790, 2820, 2895), t_first=2.0),
    dict(tag="57", page=57, profile="", edition="1956 sheet film",
         vessel="tank",
         box=(210, 748, 2345, 2815), gut=(120, 205, 2335, 2830), g_top=1.8,
         row=(190, 790, 2845, 2920), t_first=2.0),
    dict(tag="59", page=59, profile="KODAK_SUPER_XX_PAN_4142",
         edition="1956 sheet film", vessel="tank",
         box=(222, 750, 2337, 2800), gut=(130, 215, 2325, 2815), g_top=1.8,
         row=(200, 790, 2840, 2915), t_first=2.0,
         devs=("KODAK DK-60a", "KODAK DK-50", "KODAK DK-50 (1:1)",
               "KODAK D-76")),
    dict(tag="61", page=61, profile="", edition="1956 sheet film",
         vessel="tank",
         box=(204, 733, 2400, 2850), gut=(115, 198, 2390, 2865), g_top=1.6,
         row=(180, 780, 2885, 2960), t_first=2.0),
    dict(tag="63", page=63, profile="KODAK_PANATOMIC_X_SHEET_1952",
         edition="1956 sheet film", vessel="tank",
         box=(207, 795, 2340, 2790), gut=(115, 200, 2330, 2805), g_top=1.8,
         row=(185, 840, 2820, 2895), t_first=2.0),
    dict(tag="65a", page=65, profile="", edition="1956 sheet film",
         vessel="tray",
         box=(299, 762, 2387, 2652), gut=(205, 292, 2375, 2665), g_top=1.2,
         row=(275, 800, 2925, 2995), t_first=2.0),
    dict(tag="65b", page=65, profile="", edition="1956 sheet film",
         vessel="tank",
         box=(299, 762, 2665, 2930), gut=(205, 292, 2655, 2945), g_top=1.2,
         row=(275, 800, 2925, 2995), t_first=2.0,
         devs=("KODAK DK-60a", "KODAK DK-50", "KODAK DK-50 (1:1)")),
    dict(tag="67", page=67, profile="", edition="1956 sheet film",
         vessel="tank",
         box=(207, 740, 2330, 2780), gut=(115, 200, 2320, 2795), g_top=1.6,
         row=(185, 780, 2810, 2885), t_first=2.0),
    dict(tag="72", page=72, profile="", edition="1956 sheet film",
         vessel="tank",
         box=(404, 872, 392, 838), gut=(310, 396, 380, 852), g_top=1.8,
         row=(380, 900, 865, 940), t_first=2.0),
    dict(tag="74a", page=74, profile="KODAK_ROYAL_X_PAN_4166",
         edition="1956 sheet film", vessel="tray",
         box=(506, 946, 1200, 1490), gut=(410, 500, 1190, 1500), g_top=1.2,
         row=(480, 980, 1770, 1840), t_first=2.0),
    dict(tag="74b", page=74, profile="KODAK_ROYAL_X_PAN_4166",
         edition="1956 sheet film", vessel="tank",
         box=(506, 946, 1500, 1790), gut=(410, 500, 1492, 1800), g_top=1.2,
         row=(480, 980, 1770, 1840), t_first=2.0),
    dict(tag="74c", page=74, profile="KODAK_ROYAL_X_PAN_4166",
         edition="1956 roll film", vessel="small tank",
         box=(506, 860, 1990, 2330), gut=(410, 500, 1980, 2345), g_top=1.4,
         row=(480, 900, 2330, 2400), t_first=2.0),
)


def _open():
    for p in (PDF, ALT_PDF):
        if p.exists():
            import pymupdf
            return pymupdf.open(str(p))
    return None


def tick_bands(grey, x0, x1, y_from, y_to):
    """Every band of small glyphs under the plot, nearest first.

    ⚠ IT RETURNS CANDIDATES RATHER THAN A CHOICE, because no local rule tells
    the tick row from the caption. Two attempts at one failed in opposite
    directions: the DENSEST band is always "TIME OF DEVELOPMENT (MINUTES)",
    twenty-five glyphs against the digits' ten; the FIRST band is the digits on
    some sheets and a stray rule or the plot's own frame serifs on others. And
    a count test cannot separate them either, because a two-digit label
    contributes two components -- page 42's ten labels give twenty-six, which
    is caption-sized.

    The thing that actually distinguishes them is whether the band CALIBRATES:
    tick digits lie on an even lattice spanning the plot, and a word does not.
    So every candidate is handed to `calibrate` and the one with the smallest
    residual wins, which is a test on the answer rather than a guess at the
    input.
    """
    import cv2
    import numpy as np
    strip = grey[y_from:y_to, x0:x1]
    ink = (strip < 140).astype("uint8")
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(ink, 8)
    rows = []
    for i in range(1, n):
        _x, _y, w, h, area = stats[i]
        if area < 12 or h > 40 or w > 40 or h < 8:
            continue
        rows.append(float(cent[i][1]))
    if len(rows) < 5:
        return []
    rows.sort()
    out = []
    for r in rows:
        if any(abs(r - c) <= 16 for c, _ in out):
            continue
        band = [q for q in rows if abs(q - r) <= 16]
        if len(band) >= 5:
            out.append((float(np.mean(band)) + y_from, len(band)))
    return out


def label_widths_x(g, x0, x1, y0, y1, pos, gap=26):
    """Width of each tick label in a band under the plot.

    The horizontal twin of `label_widths`. One digit measures about 10 px at
    400 dpi and two about 20, which is what tells 8 from 10 and a row of
    numbers from a row of frame serifs.
    """
    import cv2
    strip = g[y0:y1, x0:x1]
    ink = (strip < 140).astype("uint8")
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(ink, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 12 or h > 60 or w > 60 or h < 6:
            continue
        comps.append((cent[i][0] + x0, x + x0, x + w + x0))
    out = []
    for p in pos:
        near = [c for c in comps if abs(c[0] - p) <= gap]
        out.append(max(c[2] for c in near) - min(c[1] for c in near)
                   if near else 0)
    return out


def time_anchor(widths, ks, wide=18, seen=8, cap=32):
    """Minutes at the first ladder position, read from the digit counts.

    Finds the one place where the labels go from one digit to two. Everything
    at or right of it must be wide, everything left of it narrow; a group too
    small or too large to be a label at all is passed over as illegible
    rather than counted either way. Returns None when no such place exists,
    which is the common failure and means the caller keeps the transcribed
    first tick.
    """
    def cls(w):
        if w < seen or w > cap:
            return None
        return bool(w >= wide)
    for i, w in enumerate(widths):
        if cls(w) is not True:
            continue
        if any(cls(widths[j]) is False for j in range(i + 1, len(widths))):
            continue
        if any(cls(widths[j]) is True for j in range(i)):
            continue
        return 10.0 - 2.0 * (ks[i] - ks[0])
    return None


def panel_axes(grey, P):
    """(a,b) for x and y from the panel's own tick labels. None on failure."""
    gx0, gx1, gy0, gy1 = P["gut"]
    bx0, bx1, _by0, by1 = P["box"]

    yl = ladder(grey, gx0, gx1, gy0, gy1, axis=0)
    fy = calibrate(yl, P["g_top"], -0.2) if yl else None
    if fy is None:
        return None, f"gamma ladder n={len(yl)}"
    if fy[2] > 0.030:
        return None, f"gamma ladder residual {fy[2]:.3f}"

    # ⚠ THE OFFSET IS TAKEN FROM THE LABELS' OWN WIDTH WHEN THEY ALLOW IT, and
    # only then. `calibrate` above anchored the ladder on the transcribed
    # `g_top`; `leading_one` finds the 1.0 row independently, and where it
    # speaks it overrules, because it is reading the page and `g_top` is
    # reading the transcription. Two guards keep it honest: the re-anchored
    # ladder may not claim a gamma ABOVE the transcribed top of the plot, and
    # may not run negative at the bottom. Either would mean the width rule has
    # latched onto something that is not an axis label, and the transcribed
    # anchor is kept instead.
    d = sorted(yl[i + 1] - yl[i] for i in range(len(yl) - 1))
    step = d[len(d) // 2]
    ks = [round((p - yl[0]) / step) for p in yl]
    k1 = leading_one(label_widths(grey, gx0, gx1, yl), ks)
    if k1 is not None:
        alt = calibrate(yl, 1.0 + 0.2 * k1, -0.2)
        if alt is not None and alt[2] <= 0.030:
            top = alt[0] * yl[0] + alt[1]
            bot = alt[0] * yl[-1] + alt[1]
            if top <= P["g_top"] + 1e-6 and bot >= -1e-6:
                fy = alt

    rx0, rx1 = bx0 - 45, bx1 + 45
    best = None
    seen_bands = set()
    for cy, _n in tick_bands(grey, rx0, rx1, by1 - 30, by1 + 200):
        if int(cy) in seen_bands:
            continue
        seen_bands.add(int(cy))
        xl = ladder(grey, rx0, rx1, int(cy - 26), int(cy + 26),
                    axis=1, gap=26)
        if not xl:
            continue
        # ⚠⚠ A BAND MUST LOOK LIKE DIGITS BEFORE IT IS ALLOWED TO BE AN
        # AXIS, and page 59 is the case that forced this. Two bands under its
        # plot both lie on an even lattice and both span a credible number of
        # minutes; the residual picked the wrong one by 0.007, and the wrong
        # one is the plot's own frame serifs -- seven marks one to four pixels
        # wide. Every gamma on that panel then came out two minutes late and
        # the traced DK-50 missed Kodak's printed labels by 0.242. A printed
        # tick label at 400 dpi is 10 px wide for one digit and 20 for two, so
        # a band in which most groups are neither is not a row of numbers.
        wx = label_widths_x(grey, rx0, rx1, int(cy - 26), int(cy + 26), xl)
        if sum(1 for w in wx if 8 <= w <= 32) < 0.6 * len(wx):
            continue
        fx = calibrate(xl, P["t_first"], 2.0) if xl else None
        if fx is None or fx[2] > 0.9:
            continue
        # ⚠ AND THE FIRST TWO-DIGIT LABEL IS TEN. The same ambiguity the
        # gamma axis has -- an even ladder fits at any offset -- is settled
        # here by the same evidence: these axes count 2, 4, 6, 8, 10, so the
        # step from one digit to two happens once and happens AT ten. It is
        # taken only when the change from narrow to wide is clean and only
        # when it moves the axis by at most one tick, because a larger move
        # means the ladder is not the row of ticks it was taken for.
        d = sorted(xl[i + 1] - xl[i] for i in range(len(xl) - 1))
        sp = d[len(d) // 2]
        kx = [round((q - xl[0]) / sp) for q in xl]
        t0 = time_anchor(wx, kx)
        if t0 is not None and abs(t0 - P["t_first"]) <= 2.0 + 1e-9:
            alt = calibrate(xl, t0, 2.0)
            if alt is not None and alt[2] <= 0.9:
                fx = alt
        # ⚠ A SANITY BOUND ON THE SCALE, not on the fit: every inset in this
        # book spans between ten and forty minutes across its plot, so a
        # calibration implying five or two hundred has locked onto the wrong
        # band however straight its line is.
        span = abs(fx[0]) * (bx1 - bx0)
        if not (10.0 <= span <= 40.0):
            continue
        if best is None or fx[2] < best[2]:
            best = fx
    if best is None:
        return None, "no band under the plot calibrates as a time axis"
    return (best, fy), ""


def trace_panel(doc, P):
    """Return traced curves as [[(minutes, gamma), ...], ...], or []."""
    import numpy as np
    import cv2
    page = doc[P["page"] - 1]
    pm = page.get_pixmap(dpi=int(DPI))
    img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(
        pm.height, pm.width, pm.n)
    grey = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)

    cal, why = panel_axes(grey, P)
    if cal is None:
        return [], why
    fx, fy = cal

    x0, x1, y0, y1 = P["box"]
    box = grey[y0:y1, x0:x1]
    mask = strip_text(degrid(box))

    w = mask.shape[1]
    curves = []
    for pts in link(fragments(mask), w):
        if (max(pts) - min(pts)) / float(w) < MIN_COVER:
            continue
        vals = [(fx[0] * (px + x0) + fx[1], fy[0] * (pts[px] + y0) + fy[1])
                for px in sorted(pts)]
        nd = sum(1 for a, b in zip(vals, vals[1:])
                 if b[1] < a[1] - MONO_SLACK)
        if nd > 0.02 * len(vals):
            continue
        if max(g for _t, g in vals) - min(g for _t, g in vals) < MIN_RISE:
            continue
        curves.append(vals)
    uniq = []
    for c in curves:
        if any(_same(dict(c), dict(u)) for u in uniq):
            continue
        uniq.append(c)
    return uniq, ""


def _same(a, b):
    """True when two traced curves agree over most of their overlap."""
    common = set(round(t, 1) for t in a) & set(round(t, 1) for t in b)
    if len(common) < 8:
        return False
    ra = {round(t, 1): g for t, g in a.items()}
    rb = {round(t, 1): g for t, g in b.items()}
    d = sorted(abs(ra[t] - rb[t]) for t in common)
    return d[int(0.8 * (len(d) - 1))] < DUP_GAMMA


def clean(s):
    """Trim a sampled trace to the interval where it is actually a curve.

    ⚠ BOTH ENDS ARE CONTAMINATED AND FOR DIFFERENT REASONS, WHICH IS WHY THIS
    IS NOT A SMOOTHING PASS. At the LEFT the family converges: all four
    Verichrome curves meet inside three pixels at four minutes, and the box
    edge and the frame corner sit in the same place, so the first two or three
    samples of every trace are some mixture of its neighbours and the frame --
    they read 0.402, 0.485, 0.462, which is not monotone and is not any one
    developer. At the RIGHT the curve simply stops, and the follower, finding
    no ink in its window, holds the last value; the trailing samples are then
    a flat shelf that no saturating exponential can pass through together with
    the rising part.

    Neither end is recoverable, so neither is kept. What is kept is the
    longest interior run that rises, and the Mees-Sheppard residual then has
    something real to test.
    """
    if len(s) < 4:
        return s
    i = 0
    while i + 2 < len(s) and not (s[i + 1][1] > s[i][1] and
                                  s[i + 2][1] > s[i + 1][1]):
        i += 1
    j = len(s) - 1
    while j - 1 > i:
        dt = s[j][0] - s[j - 1][0]
        if dt > 0 and (s[j][1] - s[j - 1][1]) / dt >= 0.004:
            break
        j -= 1
    return s[i:j + 1]


def sample(vals, grid):
    """Resample a traced curve onto whole minutes inside its own extent."""
    out = []
    ts = [t for t, _ in vals]
    lo, hi = min(ts), max(ts)
    for m in grid:
        if not (lo <= m <= hi):
            continue
        near = min(vals, key=lambda p: abs(p[0] - m))
        if abs(near[0] - m) <= 0.55:
            out.append((float(m), round(near[1], 3)))
    return out


def name_tracks(curves, devs):
    """Attach a developer name to each traced curve, or refuse.

    ⚠⚠ THE COUNT TEST IS THE STRONGEST CORRECTNESS GUARD IN THIS FILE, and it
    exists because of what the printed-label check found. `PANELS` states how
    many curves Kodak drew on each inset and in what order; if the tracer
    returns a different number it has merged two curves, split one, or invented
    a frame rule, and NOTHING about that panel can be trusted -- least of all an
    identification by rank. A tracer that found one curve on a four-curve panel
    used to pass silently and hand its single curve to the check as if it were
    D-76; on page 46a it was Microdol, and the 0.35 "disagreement" the check
    reported was really a mis-identification.

    ⚠ THE ORDER IS TAKEN AT THE LAST TIME ALL THE CURVES SHARE, not at the
    right-hand edge and not at the seed column, BECAUSE THESE CURVES CROSS AND
    END AT DIFFERENT TIMES. On page 46a D-76 overtakes DK-60a at about fifteen
    minutes, so a rank read at the right edge and a rank read at the left
    disagree about which is which; on page 42 MQ stops at sixteen minutes and
    Versatol at twenty, so the right edge holds only two of the four. The last
    common time is the one abscissa where every curve on the panel exists and
    their vertical order is the one `PANELS` records.
    """
    if len(curves) != len(devs):
        return None
    t_common = min(max(t for t, _g in c) for c in curves)

    def at(c, t):
        best = min(c, key=lambda p: abs(p[0] - t))
        return best[1]

    order = sorted(range(len(curves)),
                   key=lambda i: -at(curves[i], t_common))
    # ⚠ A TIE IN THE ORDERING IS A REFUSAL. Two curves within a hundredth of a
    # gamma at the common time cannot be told apart by rank, and guessing which
    # is which would attach a developer name to the wrong measurement.
    vals = sorted((at(c, t_common) for c in curves), reverse=True)
    if any(abs(a - b) < 0.01 for a, b in zip(vals, vals[1:])):
        return None
    return {devs[k]: curves[i] for k, i in enumerate(order)}


def _at(curve, t):
    """Linear interpolation of a traced curve at `t`, or None outside it."""
    ts = [a for a, _g in curve]
    if not ts or t < min(ts) or t > max(ts):
        return None
    for (t0, g0), (t1, g1) in zip(curve, curve[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return g0
            return g0 + (g1 - g0) * (t - t0) / (t1 - t0)
    return None


def harvest(doc):
    """Trace every panel, NAME each curve, and check the one Kodak printed."""
    from kodak_1956 import FAMILIES_1956
    printed = {}
    for f in FAMILIES_1956:
        printed.setdefault((f["profile"], f["edition"]), f)

    out = []
    for P in PANELS:
        curves, why = trace_panel(doc, P)
        good = []
        for vals in curves:
            s_ = clean(sample(vals, range(2, 33)))
            if len(s_) < 4:
                continue
            try:
                _g, _k, _t0, res = mees_fit(s_)
            except ValueError:
                continue
            if res <= FIT_TOL:
                good.append(s_)

        # ⚠ THE DUPLICATE TEST RUNS AGAIN HERE, AFTER `clean`, AND THAT
        # ORDER IS THE POINT. Two tracks that shared ink over most of their
        # length differ only in the contaminated ends, and `clean` cuts exactly
        # those off; tested before trimming they look like two curves, tested
        # after they are pixel-for-pixel one.
        ded = []
        for s_ in good:
            if any(_same(dict(s_), dict(u)) for u in ded):
                continue
            ded.append(s_)
        good = ded

        devs = P.get("devs")
        named = name_tracks(good, devs) if (devs and good) else None
        ref = printed.get((P["profile"], P["edition"])) if P["profile"] else None

        check = None
        if named and ref:
            # ⚠ LIKE FOR LIKE. The comparison is against the curve carrying the
            # SAME developer as the printed family, not against whichever
            # traced curve happens to sit closest. Taking a minimum over all
            # curves is not a test -- on a four-curve panel it almost always
            # finds something within tolerance.
            tgt = named.get(ref["developer"])
            if tgt is not None:
                # ⚠ THE TRACE IS INTERPOLATED TO KODAK'S TIME, NOT MATCHED
                # TO THE NEAREST SAMPLE. Kodak labels its curves at 7.5 and
                # 12.5 minutes; the trace is sampled on whole minutes, so a
                # nearest-sample comparison charges the trace for half a
                # minute of the curve's own slope. On page 42 that is 0.028 of
                # the 0.063 reported, against a real error of 0.035. Reading
                # the trace between its samples removes an artefact of the
                # sampling grid and tightens nothing else.
                d = [abs(_at(tgt, tt) - b) for tt, b in ref["points"]
                     if _at(tgt, tt) is not None]
                check = max(d) if d else None

        out.append(dict(P=P, curves=good, named=named, why=why, check=check,
                        ref=ref))
    return out


def main() -> int:
    doc = _open()
    if doc is None:
        print("[SKIP] kodak_1956_trace.py -- 1956-Kodak-Films.pdf not staged")
        return 0

    rows = harvest(doc)
    traced = [r for r in rows if r["curves"]]
    named = [r for r in rows if r["named"]]
    checked = [r for r in rows if r["check"] is not None]
    agreed = [r for r in checked if r["check"] <= AGREE_GAMMA]
    bad = [r for r in checked if r["check"] > AGREE_GAMMA]

    for r in rows:
        tag = r["P"]["tag"]
        if not r["curves"]:
            print(f"  p{tag:4s} NO CURVE ({r['why'] or 'fit'})")
        elif r["P"].get("devs") and not r["named"]:
            print(f"  p{tag:4s} REFUSED -- traced {len(r['curves'])} curves "
                  f"where Kodak drew {len(r['P']['devs'])}, so no curve can be "
                  f"named")
    for r in bad:
        print(f"  p{r['P']['tag']:4s} REFUSED -- its {r['ref']['developer']} "
              f"trace misses Kodak's own printed labels by {r['check']:.4f}")
    for r in agreed:
        print(f"  p{r['P']['tag']:4s} OK -- {len(r['named'])} curves named; "
              f"the {r['ref']['developer']} trace agrees with Kodak's printed "
              f"labels to {r['check']:.4f}")

    n_pts = sum(len(c) for r in agreed for c in r["named"].values())
    n_cur = sum(len(r["named"]) for r in agreed)
    print(f"[OK] kodak_1956_trace.py -- {len(traced)}/{len(PANELS)} insets "
          f"trace, {len(named)} name every curve Kodak drew, "
          f"{len(agreed)}/{len(checked)} agree with the printed labels of "
          f"their own developer; {n_cur} curves ({n_pts} points) VALIDATED")
    doc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
