#!/usr/bin/env python3
"""Curve reader for KODAK STILL-FILM technical-data sheets (E-series).

WHAT THIS READS, AND WHY IT IS A SIXTH READER RATHER THAN A FLAG ON AN
EXISTING ONE
------------------------------------------------------------------------
The corpus already had five plot readers, and none of them can touch these
pages:

  * ``spectral_vector.py``  -- built on Kodak's CINE ink convention (each trace
    drawn in the colour of the light it concerns). ⚠ THAT CONVENTION DOES NOT
    HOLD HERE. Every panel in all eleven still-film sheets probed on
    2026-08-26 came back monochrome: E-190 (2003) page 9 draws its entire
    figure set in one ink, ``(0.0, 0.0, 0.0)``; E-4051 and E-7022 use
    ``(0.137, 0.122, 0.125)`` for rules and ``(0.004, 0.008, 0.008)`` /
    ``(0.011, 0.016, 0.017)`` for curves -- three near-blacks that encode
    LAYER ORDER IN THE PDF, not channel identity. Colour carries no signal.
  * ``granularity_vector.py`` / ``vision3_granularity.py`` -- granularity vs
    density panels, which these sheets do not print at all (they print Print
    Grain Index instead; see the PGI note below).
  * ``mtf_vector.py`` -- the cine MTF layout, log-log with a channel legend
    inside the frame. These sheets put the legend outside it.
  * ``di_2254.py`` -- raster reader. These pages are vector; using a raster
    reader on them would throw away the exact path coordinates.

So the panels are separated GEOMETRICALLY here, by three facts that the
2026-08-26 probe established and that this module depends on:

  1. **A panel's three traces are dense and non-crossing.** E-190 p9's
     characteristic panel holds 102, 114 and 127 vertices per trace, each
     spanning the full frame width (x 120.3 to 267.9 pt). No merge-coast
     logic is needed -- contrast ``dashtrace.trace_predictive``'s
     ``merge_px``, which exists because the cine granularity panels DO cross.
  2. **The three traces are packed into fewer PDF paths than there are
     traces.** On that same panel, path #4 carries TWO of them (a pen-up
     discontinuity in the middle) and path #5 carries the third. Splitting on
     discontinuity -- the same idiom ``spectral_vector.subpaths`` uses -- is
     mandatory. Counting paths would find two curves on a three-curve panel.
  3. **Channel identity is printed, not implied.** Single letters ``R``,
     ``G``, ``B`` sit beside their own traces (E-190 p9: B at y 185.2, G at
     203.6, R at 225.2, all at x 202.6). This module assigns channels by
     PROXIMITY TO THOSE LETTERS, evaluated at the letter's own x. It does NOT
     assume the density order B > G > R, even though that order happens to
     hold for a masked colour negative -- an assumption that holds "because
     physics" is exactly the kind that fails silently on the one sheet where
     the artist moved a label.

THE MACRON MINUS, AGAIN, ON THE X AXIS THIS TIME
------------------------------------------------
``spectral_vector._sign_y_ticks`` was written because Kodak draws negative
Y-axis tick labels with an overbar that never reaches the text layer. The same
defect is on the X axis of every characteristic panel in E-190 (2003) and
E-2468, and it is worse there because the exposure axis is MOSTLY negative:
PyMuPDF returns the tick row as

    4.0  3.0  2.0  1.0  0.0  1.0     (at x 88.4, 125.3, 162.2, 198.4, 236.4, 272.2)

which is the sequence -4, -3, -2, -1, 0, +1 with five signs eaten. Taken at
face value it is not even monotonic, so a naive linear fit lands on garbage
and a ``dict``-based reader silently keeps whichever duplicate it saw last.
``_sign_ticks`` below signs by POSITION about the zero tick and then requires
the result to be collinear; a sheet that prints its signs properly (E-7019 and
E-7023 do -- they emit a real "-4.0") passes through the same path unchanged
because signing an already-signed monotonic run is a no-op.

⚠ WHAT THIS MODULE DELIBERATELY DOES NOT DO: CONVERT PRINT GRAIN INDEX
----------------------------------------------------------------------
Every one of these eleven sheets replaced rms granularity with Print Grain
Index, and says so in terms that forbid the conversion a simulator would
like::

    "It replaces rms granularity and has a different scale which cannot be
     compared to rms granularity."          -- E-190 (2003) p8, and 10 others

The obvious move is to invert PGI back to an rms figure and fill the eight
``GrainSpec.rms_*`` triples that currently hold analogy estimates. KODAK
E-58, the method's own publication, is on disk, and it CLOSES that door
explicitly: it lists the three transformation steps (negative granularity ->
print granularity -> visual granularity -> Grain Ruler interval) and then
states, page 2, "We will not describe the mathematical details involved in
each step." Step 1 alone is declared to depend on eleven quantities, four of
which (print material MTF, printing-system MTF, print material granularity,
print material contrast) are properties of the PAPER and are nowhere in this
database. There is no published inverse, and a fitted one from nine E-58 data
points would be an invention wearing a citation.

So PGI is stored as PGI, in its own field, flagged as not render-consuming,
and the rms triples stay marked as estimates. That is a smaller result than
"we finally measured the grain", and it is the true one.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from dataclasses import dataclass

try:
    import pymupdf  # type: ignore
except ImportError:  # pragma: no cover - environment guard
    import fitz as pymupdf  # type: ignore


PDF_DIR = os.path.join("PDF", "PROFILES", "KODAK")

#: Minimum vertices for a subpath to be considered a data trace rather than a
#: tick, a rule, a legend swatch or a frame edge. The real traces on these
#: sheets carry 70-220 vertices; the largest non-trace object found in the
#: 2026-08-26 sweep of all eleven documents was a 21-vertex frame-with-ticks
#: compound. 40 sits in the empty middle of that gap by a wide margin.
MIN_TRACE_VERTS = 40

#: A chain is a data trace only if it spans at least this fraction of the
#: panel's own tick-label extent. Measured across all 77 panels: real traces
#: cover 0.42 to 1.05 of it (a spectral-sensitivity layer covers only its own
#: band, hence the low end; a characteristic curve overruns the last tick,
#: hence above 1.0), while the largest non-trace chain covers 0.11.
TRACE_MIN_SPAN = 0.30

#: Floor for a raw fragment to enter the chainer. Two vertices is a straight
#: piece and still legitimate on the fragmented sheets (E-7022 page 4 has
#: four-vertex pieces), but a one-segment stub is more often a tick, so 3.
FRAG_MIN_VERTS = 3

#: Pen-up detection for subpath splitting, in points. Consecutive segments of
#: one trace on these sheets join to within 0.05 pt; the smallest real pen-up
#: measured was 3.7 pt. 1.0 is comfortably inside that gap.
SUBPATH_GAP_PT = 1.0

#: Tick labels are grouped into an axis row/column when their centres agree to
#: within this many points. Measured spread within one axis: 0.02 pt.
TICK_ALIGN_PT = 2.5

#: A signed tick run is accepted only if a straight line through it leaves no
#: tick further off than this, in points. The macron-minus fix is only correct
#: if the result is collinear; refusing to guess is the whole point.
TICK_FIT_TOL_PT = 1.5


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
#: Cubic-bezier flattening steps per segment. ⚠ THIS IS NOT COSMETIC.
#: The 2016 Kodak Alaris re-issues draw their traces as BEZIERS, not
#: polylines: E-4051 page 4 holds its three characteristic curves as three
#: paths of 20 cubic segments each, and its dye-density pair as 31 and 30.
#: The first version of this module skipped every non-line item, so those
#: panels reported "0 traces" from a perfectly good three-curve figure -- the
#: same class of silent-zero bug that MONO_MAX_CHANNEL caused on 5222.
#: A bezier is the exact shape the artist drew, so flattening loses nothing
#: but sampling density; 24 steps turns a 20-segment trace into 480 points,
#: which is denser than the 100-odd vertices the polyline sheets provide.
BEZIER_STEPS = 24


def _flatten_bezier(p0, p1, p2, p3, steps=BEZIER_STEPS):
    out = []
    for i in range(1, steps + 1):
        t = i / steps
        u = 1.0 - t
        out.append((
            u * u * u * p0.x + 3 * u * u * t * p1.x
            + 3 * u * t * t * p2.x + t * t * t * p3.x,
            u * u * u * p0.y + 3 * u * u * t * p1.y
            + 3 * u * t * t * p2.y + t * t * t * p3.y,
        ))
    return out


def subpaths(items, gap: float = SUBPATH_GAP_PT):
    """Split one PDF path's items into pen-down runs, flattening beziers.

    Mandatory on these sheets: E-190 p9 packs two of the three characteristic
    traces into a single path object (see fact 2 in the module docstring), and
    the 2016 re-issues draw traces as beziers (see BEZIER_STEPS).
    """
    out, cur = [], []
    for it in items:
        if it[0] == "l":
            a, b = it[1], it[2]
            pts = [(a.x, a.y), (b.x, b.y)]
        elif it[0] == "c":
            a, b, c, dd = it[1], it[2], it[3], it[4]
            pts = [(a.x, a.y)] + _flatten_bezier(a, b, c, dd)
        else:
            # A rectangle is a frame or a legend swatch, never a data trace.
            # Break rather than absorb, so a box cannot fuse two traces.
            if cur:
                out.append(cur)
                cur = []
            continue
        if cur and (abs(cur[-1][0] - pts[0][0]) > gap
                    or abs(cur[-1][1] - pts[0][1]) > gap):
            out.append(cur)
            cur = []
        if not cur:
            cur.append(pts[0])
        cur.extend(pts[1:])
    if cur:
        out.append(cur)
    return out


#: Fragment chaining tolerances, in points.
#:
#: ⚠ WHY A CHAINER IS NEEDED AT ALL, AND WHY IT IS NOT THE DASH PROBLEM.
#: Four of the eleven sheets do not draw a curve as a curve. E-7019 page 4's
#: spectral-sensitivity panel is 60-odd separate PDF path objects each spanning
#: TWO POINTS of x; E-7022 page 4's is 41 objects of four points each; E-7024
#: page 3's is six objects of ~110 vertices. All report ``dashes = '[] 0'``,
#: i.e. solid -- so this is export fragmentation, not a dash pattern, and the
#: PDF dash attribute cannot be used to group them (it is identical on every
#: fragment of every curve). The same defect truncates two CHARACTERISTIC
#: curves: E-7019's R trace arrives as 32 + 17 vertices split at x 218, and
#: E-7024's as 38 + 9 split at x 401, so a threshold-only reader silently
#: drops the red channel from both -- which is exactly what the first run of
#: this module did (it reported ``named=['B', 'G']`` on both sheets).
#:
#: Chaining is a decision procedure and can be wrong, so it refuses rather
#: than guesses: a chain extends only to a fragment whose start lands within
#: CHAIN_Y_TOL of the slope-extrapolated prediction, and ``extract_panel``
#: reports the resulting chain count so a caller can insist on the expected
#: number instead of trusting whatever fell out. This is the same discipline
#: as dashtrace's merge coast -- there, ambiguity means "neither track claims
#: the ink"; here it means "the chain stops".
CHAIN_GAP_MAX_PT = 14.0
CHAIN_OVERLAP_PT = 2.5
CHAIN_Y_TOL = 5.0


def _frag_info(pts):
    p = sorted(pts)
    n = len(p)
    k = max(2, min(6, n // 2))
    x0, y0 = p[0]
    x1, y1 = p[-1]
    dx = p[-1][0] - p[-k][0]
    slope = ((p[-1][1] - p[-k][1]) / dx) if abs(dx) > 1e-6 else 0.0
    return {"pts": p, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "slope": slope}


def chain_fragments(frags, gap_max=CHAIN_GAP_MAX_PT,
                    overlap=CHAIN_OVERLAP_PT, y_tol=CHAIN_Y_TOL):
    """Reassemble x-ordered fragments into curves.

    Greedy left-to-right: seed on the leftmost unused fragment, then repeatedly
    extend by the unused fragment whose left end best matches the chain tail's
    slope-extrapolated prediction. A candidate must start no more than
    ``gap_max`` beyond the tail (and no more than ``overlap`` before it), and
    must land within ``y_tol`` of the prediction. Returns a list of point
    lists, longest first.
    """
    info = [_frag_info(f) for f in frags if len(f) >= 2]
    used = [False] * len(info)
    chains = []
    order = sorted(range(len(info)), key=lambda i: info[i]["x0"])
    for seed in order:
        if used[seed]:
            continue
        used[seed] = True
        cur = dict(info[seed])
        pts = list(cur["pts"])
        while True:
            best, bestd = None, None
            for j in order:
                if used[j]:
                    continue
                c = info[j]
                if c["x0"] < cur["x1"] - overlap:
                    continue
                if c["x0"] > cur["x1"] + gap_max:
                    continue
                pred = cur["y1"] + cur["slope"] * (c["x0"] - cur["x1"])
                d = abs(c["y0"] - pred)
                if d <= y_tol and (bestd is None or d < bestd):
                    best, bestd = j, d
            if best is None:
                break
            used[best] = True
            pts.extend(info[best]["pts"])
            merged = _frag_info(pts)
            cur = merged
        chains.append(sorted(pts))
    chains.sort(key=len, reverse=True)
    return chains


def _cluster(vals, tol):
    """Group scalars into runs that agree to within ``tol``. Returns lists."""
    groups = []
    for v in sorted(vals):
        if groups and abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def _fit_line(pos, val):
    """Least-squares val = a*pos + b. Returns (a, b, worst_abs_residual_pos)."""
    n = len(pos)
    mp = sum(pos) / n
    mv = sum(val) / n
    sxx = sum((p - mp) ** 2 for p in pos)
    if sxx <= 0.0:
        return None
    a = sum((p - mp) * (v - mv) for p, v in zip(pos, val)) / sxx
    b = mv - a * mp
    if a == 0.0:
        return None
    # residual expressed in POINTS, so the tolerance is a geometric one
    worst = max(abs(p - (v - b) / a) for p, v in zip(pos, val))
    return a, b, worst


def _sign_ticks(pos, mag, log_axis=False, want_slope=0,
                with_residual=False):
    """Recover signs Kodak's overbar minus dropped, then demand collinearity.

    ``pos`` are tick centres along the axis in points, ``mag`` the unsigned
    magnitudes PyMuPDF returned. Returns ``(a, b)`` for ``value = a*pos + b``.

    Two candidate readings are tried, in this order:

      1. As printed. Correct for a sheet whose signs survived (E-7019,
         E-7023) and for a wholly non-negative axis.
      2. Signed by position about the zero tick: magnitudes on the low-position
         side of the ``0.0`` label are negated. This is the macron-minus case.

    Whichever fits a straight line first, wins; if neither is collinear to
    ``TICK_FIT_TOL_PT`` the function returns None and the caller must SKIP the
    panel loudly (method rule 20) rather than trace it against a bad axis.
    A mirrored axis is a perfectly collinear wrong answer, which is why the
    zero tick -- not the fit quality alone -- decides the sign.

    ⚠ ``want_slope`` IS NOT OPTIONAL POLISH; IT IS THE ONLY THING THAT CAUGHT
    A LIVE MIRROR. With reading 1 accepting any monotonic run, E-190 (2003)
    page 13's "Characteristic Curves, EI 800 (Push 1)" panel calibrated to
    ``x 4.000..0.003`` -- a perfectly collinear, perfectly wrong LEFT-HANDED
    exposure axis, and its traces came out as logE -0.555..3.438 instead of
    -3.44..0.57. It is collinear because a mirror IS linear; no fit-quality
    test can see it. What sees it is physics: on every one of these panels
    LOG EXPOSURE, WAVELENGTH and SPATIAL FREQUENCY increase left to right, and
    DENSITY, LOG SENSITIVITY and RESPONSE increase upward, which in PDF page
    coordinates (y grows downward) means the y slope must be NEGATIVE. Pass
    ``want_slope=+1`` for an x axis and ``-1`` for a y axis and the mirror is
    rejected instead of adopted; the correct signed reading is then tried.
    """
    if len(pos) < 3:
        return None
    order = sorted(range(len(pos)), key=lambda i: pos[i])
    p = [pos[i] for i in order]
    m = [mag[i] for i in order]

    if log_axis:
        # A log-ruled axis (the MTF panels: 1 2 3 4 5 7 10 20 30 50 70 100 200
        # 600) is linear in log10 of the label, and its labels are all
        # positive, so the macron-minus branch is inapplicable by construction.
        if any(v <= 0.0 for v in m):
            return None
        fit = _fit_line(p, [math.log10(v) for v in m])
        if fit and fit[2] <= TICK_FIT_TOL_PT and _slope_ok(fit[0], want_slope):
            return (fit[0], fit[1], fit[2]) if with_residual else (fit[0], fit[1])
        return None

    cand = [list(m)]
    zeros = [i for i, v in enumerate(m) if abs(v) < 1e-9]
    if zeros:
        z = zeros[0]
        cand.append([-v if i < z else v for i, v in enumerate(m)])
    for vals in cand:
        # a monotonic run is a necessary condition for a real axis
        if not (all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
                or all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))):
            continue
        fit = _fit_line(p, vals)
        if fit and fit[2] <= TICK_FIT_TOL_PT and _slope_ok(fit[0], want_slope):
            return (fit[0], fit[1], fit[2]) if with_residual else (fit[0], fit[1])
    return None


def _slope_ok(a, want):
    if want > 0:
        return a > 0.0
    if want < 0:
        return a < 0.0
    return True


#: A label centre is re-anchored onto a drawn tick mark within this distance.
#: ⚠ WHY THIS EXISTS. Label centres are NOT tick positions. On E-7019 page 4
#: the drawn x ticks sit at 93.6, 130.6, 167.5, 204.4, 241.3, 278.2 pt -- an
#: exact 36.9 pt pitch -- while the six label centres read 92.7, 130.1, 170.3,
#: 204.7, 241.5, 277.9, so "-2.0" is misplaced by 2.8 pt. Fitting the labels
#: leaves a 3.6 pt worst residual, the panel fails the 1.5 pt collinearity
#: test, and a real characteristic curve set is SKIPPED for a typesetting
#: wobble. (It is not a systematic minus-glyph shift: the other five labels
#: agree with their ticks to within 0.9 pt.) Snapping to the geometry first
#: and fitting second cures it without loosening the tolerance -- which would
#: have been the wrong fix, because 3.6 pt on a 185 pt axis is 0.1 decade of
#: exposure and this pass exists to preserve resolution, not to spend it.
TICK_SNAP_PT = 6.0

#: A tick mark is a short segment perpendicular to its axis. Longest real tick
#: measured across the eleven documents: 6.0 pt (E-190 2003, y axis). Gridlines
#: and the traces themselves are far longer, so 10.0 separates them cleanly.
TICK_MAX_LEN_PT = 10.0


def _tick_candidates(page, box, axis):
    """Positions of drawn tick marks and frame edges along ``axis``.

    Frame edges count: on E-7019 page 4 the extreme ticks are not drawn at all
    -- the inner frame rectangle (93.7, 85.0)-(278.2, 269.5) IS the -4.0 and
    +1.0 (and 0.0 and 4.0) position.
    """
    out = []
    for dr in page.get_drawings():
        r = dr["rect"]
        if r.x1 < box[0] - 12 or r.x0 > box[2] + 12:
            continue
        if r.y1 < box[1] - 12 or r.y0 > box[3] + 12:
            continue
        for it in dr["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if axis == "x":
                    if abs(a.x - b.x) < 0.35 and abs(a.y - b.y) <= TICK_MAX_LEN_PT:
                        out.append(a.x)
                else:
                    if abs(a.y - b.y) < 0.35 and abs(a.x - b.x) <= TICK_MAX_LEN_PT:
                        out.append(a.y)
            elif it[0] == "re":
                rr = it[1]
                out += [rr.x0, rr.x1] if axis == "x" else [rr.y0, rr.y1]
    return sorted(out)


#: A chain is a ROW OF SEPARATE STROKES rather than a data trace when it is
#: both sparse and mostly vertical. Both conditions are required and the
#: margins are wide in each direction; see `_is_stroke_row`.
STROKE_ROW_MAX_VERTS = 20
STROKE_ROW_FLAT_FRAC = 0.25


def _is_stroke_row(chain):
    """True when ``chain`` is a row of unrelated strokes welded by the chainer.

    ⚠ WHAT THIS REJECTS, AND WHY THE SPAN TEST CANNOT. E-4035 (May 2007) p6's
    spectral panel carries a three-line caption box whose text underlines are
    drawn as four separate VERTICAL strokes at x 76.6, 139.8, 154.2 and 183.1.
    `chain_fragments` welds them into one 9-vertex chain that crosses 53 % of
    the frame, so it passes the span test, and the panel then reported FOUR
    traces where three were wanted -- which blocked the chained reading and
    left the cyan-forming layer's curve unread. The same artefact appears on
    E-7024 p3 and inside several MTF panels.

    The discriminator is that a data trace advances in x at nearly every
    vertex, while a row of vertical strokes does not advance at most of them.
    Measured over all 602 candidate chains in the 14-document corpus, the
    zero-x-advance fraction splits them cleanly:

        real traces          0.000 - 0.200   (the 0.200 case has 276 vertices)
        stroke rows          0.333 - 0.800   (every one has 13 vertices or fewer)

    ⚠ BOTH CONDITIONS ARE REQUIRED because either alone would misfire. A bare
    vertex count would throw away E-4050's genuine 38-vertex curves -- the
    mistake `MIN_TRACE_VERTS` already made once, recorded in `keep`'s own
    comment. A bare zero-advance fraction would throw away E-7019's
    spectral-sensitivity traces, which reach 0.199 and 0.200 legitimately
    because they are drawn with 182 and 276 vertices including repeats. Taken
    together the gap is an order of magnitude on the vertex count (13 against
    133) and a third on the fraction, so nothing here sits near a boundary.
    """
    n = len(chain)
    if n > STROKE_ROW_MAX_VERTS:
        return False
    flat = sum(1 for i in range(n - 1)
               if abs(chain[i + 1][0] - chain[i][0]) < 0.01)
    return flat / max(n - 1, 1) >= STROKE_ROW_FLAT_FRAC


#: A gridline must cross at least this fraction of the panel's other dimension
#: to count as one. The spectral-sensitivity panels' gridlines cross 100 % of
#: the frame; the longest non-gridline vertical segment inside E-4035 p6's
#: panel crosses 17 %, so the two do not overlap and the threshold is not
#: fitted to a boundary case.
GRID_MIN_SPAN_FRAC = 0.60


def _grid_candidates(page, box, axis):
    """Positions of RULED GRIDLINES and frame edges along ``axis``.

    ⚠ A SEPARATE SOURCE FROM `_tick_candidates`, ON PURPOSE, AND NOT MERGED
    INTO IT. That function rejects any perpendicular line longer than
    `TICK_MAX_LEN_PT` precisely so a data trace cannot pose as a tick, which
    also throws away every gridline. On the spectral-sensitivity panels that
    leaves nothing usable: E-4035 (May 2007) p6's wavelength labels are typeset
    with a spacing that wanders from 22.3 to 27.5 pt where the ruling is an
    exact 25.07, so the labels miss a straight line by 2.8 pt and the panel is
    skipped -- while the seven drawn gridlines (x 99.51 to 249.92) plus the
    frame rectangle's own edges (74.54 and 275.08) give the axis to the
    hundredth of a point. The gridline IS the position, in the same sense the
    module already grants a tick.
    """
    lo, hi = (box[1], box[3]) if axis == "x" else (box[0], box[2])
    need = GRID_MIN_SPAN_FRAC * max(hi - lo, 1e-6)
    out = []
    for dr in page.get_drawings():
        r = dr["rect"]
        if r.x1 < box[0] - 12 or r.x0 > box[2] + 12:
            continue
        if r.y1 < box[1] - 12 or r.y0 > box[3] + 12:
            continue
        for it in dr["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if axis == "x":
                    if abs(a.x - b.x) < 0.35 and abs(a.y - b.y) >= need:
                        out.append(a.x)
                else:
                    if abs(a.y - b.y) < 0.35 and abs(a.x - b.x) >= need:
                        out.append(a.y)
            elif it[0] == "re":
                rr = it[1]
                if axis == "x" and (rr.y1 - rr.y0) >= need:
                    out += [rr.x0, rr.x1]
                elif axis == "y" and (rr.x1 - rr.x0) >= need:
                    out += [rr.y0, rr.y1]
    return sorted(out)


def _snap_all(pos, cands, tol=TICK_SNAP_PT):
    if not cands:
        return list(pos)
    out = []
    for p in pos:
        best = min(cands, key=lambda c: abs(c - p))
        out.append(best if abs(best - p) <= tol else p)
    return out


#: How many tick candidates a single label may be tried against in the
#: collinear fallback. A wavelength axis inside a 6 pt window offers at most a
#: handful; the cap exists so a pathologically ruled panel cannot turn the
#: search into a combinatorial explosion.
SNAP_FANOUT = 4

#: Product of per-label candidate counts above which the collinear fallback
#: gives up rather than enumerate. 5 labels x 4 candidates = 1024, well inside.
SNAP_COMBOS_MAX = 20000


def _snap_collinear(pos, mag, cands, grid, log_axis, want_slope,
                    tol=TICK_SNAP_PT):
    """Assign each label to a drawn tick BY COLLINEARITY, not by proximity.

    ⚠ WHY NEAREST-NEIGHBOUR SNAPPING IS NOT ENOUGH, AND WHERE IT WAS CAUGHT.
    E-4035 (May 2007) page 7 draws both 400UC characteristic panels with the
    DENSITY labels typeset 0 / -2.9 / -2.9 / -1.9 / 0 pt off a straight line.
    Fitting the labels leaves a 2.9 pt worst residual and the panel is skipped.
    `_snap_all` does not rescue it either: the "1.0" label sits at 218.9 and
    there are three tick candidates in its window -- 218.6, 218.8 and 221.0 --
    so nearest-neighbour takes 218.8, keeps the 2.25 pt error and the panel is
    STILL skipped, while 221.0 is the one a ruled axis wants (it lands 0.05 pt
    off the line through the other four). Two complete three-channel figures,
    one of them the only EI 800 push this emulsion has, were being lost to a
    tick chosen by the wrong criterion.

    The criterion here is that a ruled axis IS linear by construction, so among
    the candidates inside the snap window the correct one is the one that makes
    the run collinear. Enumerated exhaustively and scored by the same
    `_sign_ticks` test everything else uses, so a run this accepts is collinear
    to the same 1.5 pt as one the other two paths accept -- this widens WHICH
    positions are considered, never the tolerance they must meet.

    ⚠ TRIED LAST, AFTER PLAIN SNAPPING AND AFTER THE RAW LABELS. Every panel
    that already fitted keeps its previous reading bit for bit; this only sees
    inputs all earlier paths rejected.

    ⚠ TWO CANDIDATE SETS, RULED FIRST. `grid` (drawn gridlines and frame edges,
    from `_grid_candidates`) is tried before `cands` (short tick marks) because
    on the panels that reach this function the tick list is not a tick list:
    E-4035 p6's spectral panel offers 100+ candidates within a few points of
    each other -- the curve's own near-vertical segments, which are short
    enough to pass for ticks -- and nine labels against four of those apiece is
    262144 combinations, over `SNAP_COMBOS_MAX`, so the search would decline.
    The nine ruled positions are unambiguous and give one candidate per label.
    """
    # ⚠ BOTH SOURCES ARE SEARCHED AND THE BETTER RESIDUAL WINS, rather than
    # ruled-first. Trying the gridlines first looked right and was not: on
    # E-4035 p7's characteristic panels there are no internal gridlines, so the
    # ruled set is just the two frame edges, and an assignment that snaps only
    # the extreme labels is collinear enough to be returned while the
    # tick-based assignment is the exact one. That cost the panel its exact
    # axis (y 0.000..4.000 became -0.028..3.981) and moved its dmins by 0.026 --
    # away from E-190 (2003) p13's independent reading of the same emulsion,
    # which is the check that says which of the two is right.
    best = None
    for src in (grid, cands):
        got = _snap_collinear_one(pos, mag, src, log_axis, want_slope, tol)
        if got and (best is None or got[0][2] < best[0][2]):
            best = got
    return best


def _snap_collinear_one(pos, mag, cands, log_axis, want_slope, tol):
    """One pass of the collinear search over a single candidate set."""
    if not cands:
        return None
    opts = []
    for p in pos:
        near = sorted((c for c in cands if abs(c - p) <= tol),
                      key=lambda c: abs(c - p))[:SNAP_FANOUT]
        # The label's own centre stays a candidate: a panel where only SOME
        # labels are misplaced must be able to keep the good ones.
        opts.append([p] + [c for c in near if abs(c - p) > 1e-9])
    total = 1
    for o in opts:
        total *= len(o)
    if total > SNAP_COMBOS_MAX:
        return None
    best = None
    for combo in itertools.product(*opts):
        cand = list(combo)
        # A tick cannot serve two labels.
        if len(set(round(v, 4) for v in cand)) != len(cand):
            continue
        fit = _sign_ticks(cand, mag, log_axis=log_axis,
                          want_slope=want_slope, with_residual=True)
        if fit is None:
            continue
        if best is None or fit[2] < best[0][2]:
            best = (fit, cand)
    if best is None:
        return None
    return best[0], best[1]


def _best_axis(pos, mag, cands, grid, log_axis, want_slope):
    """Fit the axis from label centres AND from tick-snapped centres; keep the
    better of the two.

    ⚠ SNAPPING IS NOT UNCONDITIONALLY BETTER, WHICH IS WHY BOTH ARE TRIED.
    Snapping every label fixed the characteristic and dye-density panels (they
    became exact: x -4.000..1.000, y 0.000..4.000 where the labels alone gave
    -4.001..0.991) but BROKE the spectral-sensitivity and MTF panels, which had
    been fitting cleanly. Those two have many closely-spaced tick candidates --
    a wavelength axis is ruled every 20 pt, and the MTF panel is ruled at
    1/2/3/5/7/10/20/... in both directions -- so a 6 pt snap window can pull a
    label onto its NEIGHBOUR's tick and manufacture a non-collinear run out of
    a good one. Choosing per panel by residual keeps the gain without the loss;
    the ORDER is snapped-first, and it is not a tie-break by residual: a drawn
    tick IS the position, so when the snapped run is collinear it is the more
    accurate reading even if its residual is marginally larger. Falling back to
    the labels happens only when snapping has visibly gone wrong -- i.e. when it
    destroys collinearity, which is exactly the mis-snap case.

    ⚠ AND THE COLLINEAR SEARCH IS TRIED LAST, NOT SECOND. Inserting it between
    the two existing paths moved five already-adopted readings -- e4050 (2016)
    p4's three gammas by 0.004, PORTRA-2003 p9's MTF green f50 by 0.6 cy/mm,
    E-7023 p4's dye peaks by 0.012 -- because those panels reach the RAW path
    today and the search would have intercepted them. None of those five is a
    panel anybody complained about, and there is no evidence the intercepted
    reading is the better one, so the order here guarantees the search sees
    only inputs that both older paths REJECTED: a panel that fits today keeps
    its number bit for bit, and the only panels this can change are ones that
    were being skipped outright.
    """
    for p in (_snap_all(pos, cands), list(pos)):
        fit = _sign_ticks(p, mag, log_axis=log_axis, want_slope=want_slope,
                          with_residual=True)
        if fit:
            return fit[0], fit[1], p
    # ⚠ LAST RESORT, ADDED 2026-09-03 FOR E-4035 p7. See `_snap_collinear`.
    got = _snap_collinear(pos, mag, cands, grid, log_axis, want_slope)
    if got:
        fit, cand = got
        return fit[0], fit[1], cand
    return None


def _num(txt):
    t = txt.strip().replace("−", "-").replace("–", "-")
    try:
        return float(t)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# panel model
# ---------------------------------------------------------------------------
@dataclass
class Axis:
    a: float
    b: float

    def value(self, pos: float) -> float:
        return self.a * pos + self.b


@dataclass
class Panel:
    """One located plot: its axis calibrations and its labelled traces."""

    x_axis: Axis
    y_axis: Axis
    x_span: tuple[float, float]
    y_span: tuple[float, float]
    traces: dict          # label -> [(x_value, y_value), ...]
    unlabelled: list      # traces with no legend letter nearby


def _words(page):
    return [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]


def _numeric_axes(page, box, log_x=False, log_y=False):
    """Find the x tick row and y tick column of the plot inside ``box``.

    ``box`` is (x0, y0, x1, y1) in page coordinates, generously sized: the
    routine looks for the largest aligned run of numeric labels inside it, so
    a loose box costs nothing but a slightly wider search.

    ``log_x`` / ``log_y`` switch the corresponding axis to log10 ruling. On a
    log axis the returned Axis maps position to the LOG10 of the quantity, not
    the quantity: an MTF response of 70% comes back as 1.845. Callers that
    want the percentage must exponentiate, and the ones below do it explicitly
    so the transform is never applied twice.
    """
    x0, y0, x1, y1 = box
    nums = []
    for wx0, wy0, wx1, wy1, t in _words(page):
        v = _num(t)
        if v is None:
            continue
        if not (x0 <= wx0 and wx1 <= x1 and y0 <= wy0 and wy1 <= y1):
            continue
        nums.append((wx0, wy0, wx1, wy1, v))

    tx = _tick_candidates(page, box, "x")
    ty = _tick_candidates(page, box, "y")
    # Ruled gridlines, used ONLY by the collinear last resort inside
    # `_best_axis`; see `_grid_candidates` for why they are kept apart.
    gx = _grid_candidates(page, box, "x")
    gy = _grid_candidates(page, box, "y")

    # --- x axis: numbers sharing a y centre, spread in x -------------------
    best_x = None
    for grp in _cluster([(n[1] + n[3]) / 2.0 for n in nums], TICK_ALIGN_PT):
        row = [n for n in nums if any(abs((n[1] + n[3]) / 2.0 - g) < 1e-9
                                     for g in grp)]
        if len(row) < 3:
            continue
        pos = [(n[0] + n[2]) / 2.0 for n in row]
        fit = _best_axis(pos, [n[4] for n in row], tx, gx, log_x, +1)
        if fit and (best_x is None or len(row) > best_x[1]):
            best_x = (Axis(fit[0], fit[1]), len(row), (min(fit[2]), max(fit[2])))

    # --- y axis: numbers sharing a right edge, spread in y -----------------
    best_y = None
    for grp in _cluster([n[2] for n in nums], TICK_ALIGN_PT):
        col = [n for n in nums if any(abs(n[2] - g) < 1e-9 for g in grp)]
        if len(col) < 3:
            continue
        pos = [(n[1] + n[3]) / 2.0 for n in col]
        fit = _best_axis(pos, [n[4] for n in col], ty, gy, log_y, -1)
        if fit and (best_y is None or len(col) > best_y[1]):
            best_y = (Axis(fit[0], fit[1]), len(col), (min(fit[2]), max(fit[2])))

    return best_x, best_y


def _legend_letters(page, box, letters=("R", "G", "B")):
    x0, y0, x1, y1 = box
    out = []
    for wx0, wy0, wx1, wy1, t in _words(page):
        if t.strip() not in letters:
            continue
        if not (x0 <= wx0 and wx1 <= x1 and y0 <= wy0 and wy1 <= y1):
            continue
        out.append((t.strip(), (wx0 + wx1) / 2.0, (wy0 + wy1) / 2.0))
    return out


def _at_x(trace, xq, clamp=False):
    """Linear interpolation of a trace at page-x ``xq``, or None.

    ``clamp=True`` evaluates at the nearest end instead of refusing when
    ``xq`` is outside the trace. ⚠ THIS IS REQUIRED FOR THE MTF PANELS AND
    ONLY FOR THEM. On the characteristic panels the R/G/B letters sit ON the
    curves (E-190 p9: letters at x 202.6, traces spanning 120.3-267.9), so a
    strict reading works. On the MTF panels the letters sit BEYOND the right
    end of every trace (p9: letters at x 513.4-513.7, traces ending at 504.7),
    so a strict reading returns None for all three and the whole panel comes
    back unlabelled. Clamping reads each trace at its own right end, which is
    exactly what the artist was pointing at.
    """
    pts = sorted(trace)
    if not pts:
        return None
    if xq < pts[0][0] or xq > pts[-1][0]:
        if not clamp:
            return None
        return pts[0][1] if xq < pts[0][0] else pts[-1][1]
    for i in range(1, len(pts)):
        if pts[i][0] >= xq:
            xa, ya = pts[i - 1]
            xb, yb = pts[i]
            if xb == xa:
                return ya
            t = (xq - xa) / (xb - xa)
            return ya + t * (yb - ya)
    return None


def extract_panel(page, box, letters=("R", "G", "B"), min_verts=MIN_TRACE_VERTS,
                  log_x=False, log_y=False, expect=0):
    """Locate and read one panel inside ``box``.

    Returns a ``Panel`` or None. Returning None is a real outcome, not a
    failure to try: a panel whose axis will not fit a straight line is skipped
    loudly by the caller (method rule 20).
    """
    ax, ay = _numeric_axes(page, box, log_x=log_x, log_y=log_y)
    if not ax or not ay:
        return None
    x_axis, _, xspan = ax
    y_axis, _, yspan = ay

    # Candidate fragments: every pen-down run inside the box. NOTE the low
    # floor -- FRAG_MIN_VERTS, not min_verts. On the four fragmented sheets a
    # real curve arrives as dozens of 4-vertex pieces, so filtering at the
    # trace threshold before chaining throws the curve away. The threshold is
    # applied AFTER chaining instead, where it means what it says.
    frags = []
    for dr in page.get_drawings():
        r = dr["rect"]
        if r.x1 < box[0] or r.x0 > box[2] or r.y1 < box[1] or r.y0 > box[3]:
            continue
        for sp in subpaths(dr["items"]):
            inside = [q for q in sp
                      if box[0] <= q[0] <= box[2] and box[1] <= q[1] <= box[3]]
            if len(inside) < FRAG_MIN_VERTS:
                continue
            frags.append(inside)
    # ⚠ KEEP CHAINS BY HOW MUCH OF THE AXIS THEY SPAN, NOT BY VERTEX COUNT.
    # A vertex threshold is a proxy for "is this a data trace", and it is a bad
    # one once bezier sheets and fragmented sheets are both in scope: the same
    # curve arrives as 481 points on E-4051, 127 on E-190, and 38 on E-4050.
    # At MIN_TRACE_VERTS = 40 the E-4050 characteristic panel lost TWO of its
    # three curves (38 and 39 vertices) and reported ``named=['R']`` -- a
    # two-thirds silent data loss from an arbitrary constant. Span is the
    # property that actually distinguishes a curve from a tick: a trace crosses
    # most of its frame, a tick crosses none of it.
    span_x = abs(xspan[1] - xspan[0]) or 1.0

    def keep(chains):
        out = []
        for c in chains:
            if len(c) < FRAG_MIN_VERTS * 2:
                continue
            if (max(p[0] for p in c)
                    - min(p[0] for p in c)) < TRACE_MIN_SPAN * span_x:
                continue
            if _is_stroke_row(c):
                continue
            out.append(c)
        return out

    # ⚠ CHAIN ONLY IF YOU MUST. Chaining is a decision procedure, and on a
    # panel whose curves CROSS it decides wrong: the spectral-sensitivity
    # panels have the yellow-forming layer falling through the magenta-forming
    # layer's rise near 495 nm and the magenta through the cyan near 575 nm
    # (E-190 2003 p9, verified against the 300 dpi page image), so a left-to-
    # right chainer happily welds the blue layer's descent onto the green
    # layer's ascent and returns TWO traces where the sheet drew three. Those
    # same panels need no chaining at all -- their three curves are already
    # three clean subpaths of 94-105 vertices. So: take the unchained reading
    # when it already yields the number of traces the panel kind demands, and
    # chain only when it comes up short, which is the fragmented-export case
    # (E-7019, E-7022, E-7024) and the split-trace case (E-7019/E-7024 red
    # characteristic curve). Refusing to chain when chaining is unnecessary is
    # the same instinct as dashtrace's merge coast: do not decide what you were
    # not asked to decide.
    def cover(chains):
        return sum(max(p[0] for p in c) - min(p[0] for p in c) for c in chains)

    raw = keep(frags)
    chained = keep(chain_fragments(frags))
    if expect and len(raw) != expect:
        if abs(len(chained) - expect) < abs(len(raw) - expect):
            raw = chained
    # ⚠ A CORRECT TRACE COUNT IS NOT A COMPLETE READING, AND THAT WAS COSTING
    # DATA SILENTLY. E-4035 (May 2007) page 6 draws 100UC's red characteristic
    # curve as TWO subpaths meeting exactly at (210.76, 228.67), spanning
    # x 114.8-210.8 and 210.8-262.4. The first half covers 52 % of the frame
    # and survives `keep`; the second covers 28 % and does not. The unchained
    # reading therefore returned THREE traces on a three-curve panel -- exactly
    # the count the panel demands -- with the red curve stopping at log H -0.54
    # instead of +0.86, losing the upper half of its straight line and its
    # shoulder entirely. `measure_char` declined it outright and the tone fit
    # that did run took its gamma off the toe. The count test could not see
    # any of this: it was already satisfied.
    #
    # So when chaining returns the SAME number of traces and covers MORE of the
    # x axis, the chained reading is the better one and is taken. Equal-count is
    # the whole safeguard, and it is what keeps the crossing panels safe: on the
    # spectral-sensitivity panels chaining welds the blue layer's descent onto
    # the green layer's ascent and collapses three traces to two, so the counts
    # differ and this branch cannot fire.
    elif (expect and len(chained) == len(raw) == expect
            and cover(chained) > cover(raw) + 1e-6):
        raw = chained

    # Channel assignment by proximity to the printed legend letter, evaluated
    # at that letter's own x -- never by density order (see docstring fact 3).
    #
    # ⚠ GLOBALLY GREEDY, NOT LETTER-BY-LETTER. Two legend letters can sit
    # closer to each other than either sits to its curve: on E-190 (2003) p11
    # the MTF panel's G and B letters are 6.8 pt apart (y 383.7 and 376.9).
    # Resolving letters in list order lets whichever came first claim the
    # nearer curve and pushes the other onto a wrong one. Sorting ALL
    # (letter, trace) pairs by distance and consuming them in that order gives
    # the confident pairings first -- the same idiom dashtrace uses for its
    # candidate/prediction pairs.
    legend = _legend_letters(page, box, letters)
    named, taken, used = {}, set(), set()
    pairs = []
    for li, (lab, lx, ly) in enumerate(legend):
        for i, tr in enumerate(raw):
            yv = _at_x(tr, lx, clamp=True)
            if yv is None:
                continue
            pairs.append((abs(yv - ly), li, i, lab))
    for _d, li, i, lab in sorted(pairs):
        if li in used or i in taken:
            continue
        used.add(li)
        taken.add(i)
        named[lab] = raw[i]

    def conv(tr):
        return [(x_axis.value(px), y_axis.value(py)) for px, py in tr]

    return Panel(
        x_axis=x_axis,
        y_axis=y_axis,
        x_span=xspan,
        y_span=yspan,
        traces={k: conv(v) for k, v in named.items()},
        unlabelled=[conv(raw[i]) for i in range(len(raw)) if i not in taken],
    )


# ---------------------------------------------------------------------------
# layer assignment for the panels that print no letters
# ---------------------------------------------------------------------------
def assign_layers(traces):
    """Name three spectral-sensitivity traces r/g/b by where they sit in nm.

    The sheets label these three by a three-line caption -- "Yellow-/Forming/
    Layer", "Magenta-/Forming/Layer", "Cyan-/Forming/Layer" -- laid out left to
    right above the panel, not by letters on the curves, so the letter matcher
    finds nothing. The assignment is nonetheless forced rather than assumed:
    the yellow-forming layer IS the blue-sensitive one, the magenta-forming
    layer IS the green-sensitive one and the cyan-forming layer IS the
    red-sensitive one, by the definition of subtractive colour, and each
    layer's sensitivity band is disjoint from the others by construction (that
    is what makes the film colour-separating at all). So ordering the three
    traces by their band CENTROID -- not by peak, which on a 40 nm plateau is
    arbitrary, the trap the 5246/5205 blue-peak guard was written for -- gives
    b, g, r in that order.

    Returns a dict or None if there are not exactly three traces.
    """
    if len(traces) != 3:
        return None
    def centroid(tr):
        # weight by sensitivity above the trace's own floor, so a long low tail
        # cannot drag the centroid the way a plain mid-range would
        lo = min(p[1] for p in tr)
        w = [(p[0], p[1] - lo) for p in tr]
        s = sum(q[1] for q in w)
        if s <= 0:
            return sum(p[0] for p in tr) / len(tr)
        return sum(q[0] * q[1] for q in w) / s
    order = sorted(traces, key=centroid)
    return {"b": order[0], "g": order[1], "r": order[2]}


#: The wavelength range every spectral-dye-density panel in this corpus is
#: drawn over, and how far a traced extent may fall from it. 2.5 nm is about
#: 1.5 pt on these panels -- wider than any correct reading's error (the worst
#: is 0.2 nm) and far narrower than the 4.6 nm mis-calibration it must catch.
DYE_NM_LO, DYE_NM_HI, DYE_NM_TOL = 400.0, 700.0, 2.5


def assign_dye_pair(traces):
    """Name two spectral-dye-density traces neutral/dmin.

    The caption is "Typical densities for a midscale neutral subject and
    D-min", with the two curves keyed by a legend the text layer scrambles.
    They are separated by the only thing that cannot be scrambled: a midscale
    neutral is DENSER than the base at every wavelength, because it is the base
    PLUS image dye. Mean density therefore orders them, and the ordering is
    checked pointwise -- if the two ever cross, this returns None rather than
    label them, because a crossing would mean one of them is not what the
    caption says.

    ⚠ AND THE WAVELENGTH EXTENT IS CHECKED, BECAUSE A COLLINEAR AXIS FIT CAN
    STILL BE THE WRONG ONE. Every spectral-dye-density panel in this corpus is
    drawn 400 to 700 nm with both curves crossing the whole frame, so a traced
    extent that falls outside `DYE_NM_TOL` of those two numbers is not a
    narrower panel -- it is a mis-calibrated axis, and the reader must decline
    it rather than hand back a shifted array.

    E-4035 (May 2007) p7 is exactly that case and is why this check exists. Its
    "400" tick label centre sits 3.7 pt right of the drawn tick, while four
    vertices of the near-vertical D-min curve pass the tick test 1.0-2.0 pt from
    the label. `_snap_all` takes one of those, the resulting run is collinear
    enough to be accepted, and the panel reads 395.4-697.6 nm -- a 4.6 nm error
    that nothing else in the pipeline would have questioned. The 100UC panel on
    p6 of the same document reads 399.8-700.2 and passes.
    """
    if len(traces) != 2:
        return None
    hi, lo = sorted(traces, key=lambda t: sum(p[1] for p in t) / len(t),
                    reverse=True)
    for x, y in lo:
        yh = _at_x(hi, x)
        if yh is not None and yh < y - 0.02:
            return None
    for t in (hi, lo):
        if (abs(min(p[0] for p in t) - DYE_NM_LO) > DYE_NM_TOL
                or abs(max(p[0] for p in t) - DYE_NM_HI) > DYE_NM_TOL):
            return None
    return {"neutral": hi, "dmin": lo}


def resample(trace, lo, hi, step):
    """Uniform resample of a trace onto ``lo:hi:step``.

    Returns a list of (x, y_or_None); None outside the trace's own extent --
    the sentinel that keeps "the plot stops here" distinct from "the value is
    small here", which is the same distinction the -4.0 spectral floor makes.
    """
    out = []
    n = int(round((hi - lo) / step))
    for i in range(n + 1):
        x = lo + i * step
        out.append((x, _at_x(trace, x)))
    return out


def f_at_response(trace, pct):
    """Spatial frequency (cycles/mm) where a log-log MTF trace crosses ``pct``.

    ``trace`` carries (log10 f, log10 response). Returns None when the trace
    never crosses -- which is a real answer for a sheet whose curve stops above
    the level asked for, and must not be reported as the last frequency drawn.
    """
    tgt = math.log10(pct)
    pts = sorted(trace)
    for i in range(1, len(pts)):
        y0, y1 = pts[i - 1][1], pts[i][1]
        if (y0 - tgt) * (y1 - tgt) <= 0.0 and y0 != y1:
            x0, x1 = pts[i - 1][0], pts[i][0]
            t = (tgt - y0) / (y1 - y0)
            return 10.0 ** (x0 + t * (x1 - x0))
    return None


# ---------------------------------------------------------------------------
# curve model fitting
# ---------------------------------------------------------------------------
def _sp(x: float, k: float) -> float:
    z = x / k
    if z > 60.0:
        return x
    if z < -60.0:
        return 0.0
    return k * math.log1p(math.exp(z))


def model_density(x, dmin, gamma, toe_x, toe_k, sh_x, sh_k):
    return dmin + gamma * (_sp(x - toe_x, toe_k) - _sp(x - sh_x, sh_k))


#: dmin is read as the mean density over the samples within this much of the
#: trace's own minimum. Every characteristic panel in the eleven documents
#: begins ON the base+fog plateau -- verified numerically: E-190 p9's local
#: slope at the left edge is 0.004 (R), 0.005 (G), 0.015 (B) -- so the minimum
#: IS the plateau and not a point on the toe. Cross-check that it is not an
#: accident of one sheet: the 400-speed sheets' plateaux sit a uniform +0.050
#: to +0.051 D above the 160-speed sheets' in all three channels (0.2527 vs
#: 0.2018, 0.6563 vs 0.6059, 0.856 vs 0.8085), which is a base difference,
#: not three independent reading errors.
DMIN_BAND = 0.02

#: gamma is the OLS slope over the samples whose density lies this far above
#: dmin. ⚠ NOT the maximum local slope, and not a median of the steep region.
#: The steepest local slope on these curves sits just past the toe, where the
#: curve is still bending: taking it gave 160NC red gamma 0.5456 where the
#: drawn straight line reads 0.527-0.529 over its whole length, a 3 percent
#: high bias, and on the 481-point bezier sheets the same rule wandered into
#: noise and reported a "straight section" 0.38 decades long at the wrong end
#: of the curve (E-4051 red: straight -0.52..-0.14, rms 0.039, worst 0.147).
#: A fixed density band is where the straight line is BY DEFINITION -- it is
#: also the band Kodak's own contrast-index construction uses -- and it does
#: not move when the sampling density changes.
STRAIGHT_D_LO, STRAIGHT_D_HI = 0.50, 1.60

#: Half-width of the local-slope window, in decades of log exposure.
SLOPE_WIN = 0.15


def local_slopes(trace, win=SLOPE_WIN):
    tr = sorted(trace)
    out = []
    for x, _y in tr:
        lo = _at_x(tr, x - win)
        hi = _at_x(tr, x + win)
        if lo is None or hi is None:
            continue
        out.append((x, (hi - lo) / (2.0 * win)))
    return out


def measure_char(trace):
    """Measure what a Kodak still-film characteristic panel ACTUALLY shows.

    ⚠ THESE PANELS HAVE NO SHOULDER, AND THAT CHANGES WHAT MAY BE ADOPTED.
    Local-slope profiling of E-190 (2003) p9 gives, per channel, a rise from
    0.004 through the toe to a plateau that then holds DEAD FLAT to the right
    edge of the plot: R 0.527, 0.528, 0.528, 0.529, 0.528, 0.527 over the last
    six samples; G 0.550 six times running; B 0.608, 0.607, 0.608, 0.608,
    0.609, 0.609. The curve is straight where it stops. Every one of the
    eleven sheets is drawn the same way, which is correct behaviour for a
    colour negative -- its shoulder lies far above the plotted exposure range.

    The consequence is that a free six-parameter fit of ToneCurve to these
    traces INVENTS three of its six numbers. Run unconstrained on 160NC's red
    channel it returned shoulder_x = 1.276 against a plot that ends at logE
    0.874, and therefore dmax = 2.251 against a traced maximum of 1.884 -- an
    extrapolated 0.37 D of density that no measurement supports. It also
    corrupts the two numbers that ARE measurable, because the phantom shoulder
    pulls the slope: it reported gamma 0.601 where the drawn straight line is
    0.528, a 14 percent error in the single most important curve parameter.

    So this function measures only the three things drawn -- ``dmin``,
    ``gamma``, and the toe (``toe_x``, ``toe_k``, fitted with the first two
    HELD and the shoulder pushed six decades outside the traced range so it
    cannot contribute). ``shoulder_x``, ``shoulder_k`` and ``dmax`` are NOT
    returned, because they are not in the source. A caller wanting a complete
    ToneCurve must carry them over from whatever it had, and say so.
    """
    tr = sorted(trace)
    if len(tr) < 8:
        return None
    dlo = min(y for _x, y in tr)
    band = [y for _x, y in tr if y <= dlo + DMIN_BAND]
    dmin = sum(band) / len(band)

    mid = [(x, y) for x, y in tr
           if dmin + STRAIGHT_D_LO <= y <= dmin + STRAIGHT_D_HI]
    if len(mid) < 5:
        return None
    n = len(mid)
    mx = sum(p[0] for p in mid) / n
    my = sum(p[1] for p in mid) / n
    sxx = sum((p[0] - mx) ** 2 for p in mid)
    if sxx <= 0:
        return None
    gamma = sum((p[0] - mx) * (p[1] - my) for p in mid) / sxx

    far = tr[-1][0] + 6.0
    best, berr = None, None
    span = tr[-1][0] - tr[0][0]
    for i in range(80):
        toe_x = tr[0][0] + 0.0125 * i * span
        for j in range(70):
            toe_k = 0.04 + 0.01 * j
            e = 0.0
            for x, y in tr:
                m = dmin + gamma * (_sp(x - toe_x, toe_k) - _sp(x - far, toe_k))
                e += (m - y) ** 2
            e /= len(tr)
            if berr is None or e < berr:
                best, berr = (toe_x, toe_k), e
    toe_x, toe_k = best
    res = [dmin + gamma * (_sp(x - toe_x, toe_k) - _sp(x - far, toe_k)) - y
           for x, y in tr]
    return {"dmin": dmin, "gamma": gamma, "toe_x": toe_x, "toe_k": toe_k,
            "rms": math.sqrt(sum(r * r for r in res) / len(res)),
            "worst": max(abs(r) for r in res),
            "x0": tr[0][0], "x1": tr[-1][0],
            "straight": (min(p[0] for p in mid), max(p[0] for p in mid)),
            "d0": tr[0][1], "d1": tr[-1][1]}


def fit_tone_curve(pts, seed=None, iters=4000):
    """Fit film_profiles.ToneCurve's five-and-a-half parameters to a trace.

    Coordinate descent with shrinking steps. Deliberately not scipy: the
    generator has no numeric dependency and adding one to fit six numbers
    would be a poor trade. Returns (params, rms, worst).

    ⚠ The fit is constrained to ``sh_k <= 1.4 * toe_k``, which is the bound
    ToneCurve's own docstring asks new stocks to respect (validate() only
    rejects above 2x, but measured reversals of order 1e-6 appear between
    1.4x and 2x and verify.py checks for them). An unconstrained fit happily
    walks past it.
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lo, hi = min(xs), max(xs)
    dmin0 = min(ys)
    dmax0 = max(ys)
    if seed is None:
        seed = [dmin0, (dmax0 - dmin0) / max(hi - lo, 1e-6) * 1.6,
                lo + 0.15 * (hi - lo), 0.30, hi - 0.05 * (hi - lo), 0.42]
    p = list(seed)

    def err(q):
        if q[1] <= 0 or q[3] <= 0 or q[5] <= 0 or q[4] <= q[2]:
            return 1e9
        if q[5] > 1.4 * q[3]:
            return 1e9
        s = 0.0
        for x, y in zip(xs, ys):
            d = model_density(x, *q) - y
            s += d * d
        return s / len(xs)

    step = [0.05, 0.05, 0.20, 0.05, 0.20, 0.05]
    e = err(p)
    for _ in range(iters):
        improved = False
        for i in range(6):
            for s in (step[i], -step[i]):
                q = list(p)
                q[i] += s
                e2 = err(q)
                if e2 < e:
                    p, e = q, e2
                    improved = True
                    break
        if not improved:
            step = [s * 0.5 for s in step]
            if max(step) < 1e-5:
                break
    res = [model_density(x, *p) - y for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(r * r for r in res) / len(res))
    return p, rms, max(abs(r) for r in res)


# ---------------------------------------------------------------------------
# panel discovery
# ---------------------------------------------------------------------------
#: Caption text -> (panel kind, log_x, log_y, legend letters).
#:
#: The kinds and their axis rulings, all confirmed against the page images on
#: 2026-08-26:
#:   char  -- LOG EXPOSURE (lux-seconds) vs DENSITY, both linear
#:   sens  -- WAVELENGTH (nm) vs LOG SENSITIVITY, both linear. The traces are
#:            labelled by a three-line caption ("Yellow-/Forming/Layer"), NOT
#:            by single letters, so the letter matcher finds nothing and every
#:            trace lands in ``unlabelled`` -- see assign_layers().
#:   dye   -- WAVELENGTH (nm) vs DIFFUSE SPECTRAL DENSITY, both linear. Two
#:            traces, "Midscale Neutral" and "Minimum Density": exactly the
#:            neutral+Dmin pair schema v14 added, and NOT three separated dyes.
#:   mtf   -- SPATIAL FREQUENCY (cycles/mm) vs RESPONSE (%), both log10.
#: The fifth element is HOW MANY traces the panel must have. It is not a
#: convenience: it is what lets the reader tell "I found the figure" from "I
#: found some of the figure", and it is what gates the chainer (see
#: extract_panel). Every one of these counts is fixed by what the panel means
#: -- three sensitive layers, three channels, one neutral and one D-min -- not
#: by what any particular sheet happened to draw.
CAPTIONS = {
    "characteristic curves": ("char", False, False, ("R", "G", "B"), 3),
    "spectral-sensitivity curves": ("sens", False, False, (), 3),
    "spectral sensitivity curves": ("sens", False, False, (), 3),
    "spectral-dye-density curves": ("dye", False, False, (), 2),
    "modulation transfer function": ("mtf", True, True, ("R", "G", "B"), 3),
}

#: Caption-relative box, in points: (dx0, dy0, dx1, dy1) from the caption's
#: top-left. Kodak's still-film grid puts the caption above and slightly right
#: of its panel's y-axis labels; these offsets were measured on the E-190 p9
#: 2x2 grid and then checked against the 2x3 (E-190 p13), 1x3 (E-190 p14) and
#: centred-single (E-7019 p4) variants. dy1 is overridden by the next caption
#: in the same column when there is one.
BOX_DX0, BOX_DY0, BOX_DX1, BOX_DY1 = -80.0, 4.0, 185.0, 250.0

#: Two captions belong to the same column when their x agree within this.
COL_TOL_PT = 100.0


#: (pdf basename, page) -> extra panels the caption scan cannot find, as
#: (kind, label, box, log_x, log_y, letters, expect).
#:
#: ⚠ ONE ENTRY, FOR ONE REAL DEFECT IN THE SOURCE. E-7022 (March 2022, Kodak
#: Alaris) prints its characteristic panel with NO panel caption -- page 4
#: carries the section heading "CURVES" and then captions only
#: "Spectral-Sensitivity Curves" and "Spectral-Dye-Density Curves", leaving the
#: characteristic figure (rotated axis title "DENSITY" at 54,146; traces inside
#: 82,130-260,226; "Log H Ref: -1.14" printed beside it) anonymous. Every other
#: sheet in the corpus captions it. The box below is read off that geometry, not
#: guessed, and it is an OVERRIDE rather than a loosened caption rule on
#: purpose: relaxing the scan to "any panel with a DENSITY axis" would start
#: matching axis titles across the whole corpus and trade one named exception
#: for an unbounded number of silent ones.
#:
#: ⚠ SECOND ENTRY GROUP, 2026-09-03, FOR A DIFFERENT DEFECT: E-4035's captions
#: are CENTRED over their panels where every other sheet in the corpus
#: left-aligns them near the y-axis labels. The caption-relative box therefore
#: lands about 30 pt right of where it should, and the consequence is not a
#: skip but something worse -- a panel that reads with a slightly wrong axis.
#: E-4035 p7's EI 400 panel read dmin 0.3095 / 0.7409 / 1.0289 from the
#: caption box against 0.3364 / 0.7670 / 1.0543 from the boxes below, and the
#: second set is the right one: E-190 (2003) p13 reads the SAME emulsion
#: independently at 0.3338 / 0.7630 / 1.0508, agreeing with it to 0.004 D and
#: disagreeing with the caption-box reading by 0.024. The boxes here are
#: anchored on the figure code each panel prints at its own bottom-left
#: (F009_0585AC and its siblings) plus the caption's baseline, so they are read
#: off the page rather than offset from a caption.
#:
#: ⚠ AND THEY SUPPRESS the caption-derived panels for those pages -- see
#: PANEL_SUPPRESS -- because otherwise every panel would be found twice.
PANEL_OVERRIDES = {
    ("E7022-1.pdf", 4): [
        ("char", "Characteristic Curves [uncaptioned in source]",
         (40.0, 50.0, 300.0, 296.0), False, False, ("R", "G", "B"), 3),
    ],
    ("e4035-100UC_400UC.pdf", 6): [
        ("char", "100UC Characteristic Curves",
         (46.0, 92.0, 300.0, 304.0), False, False, ("R", "G", "B"), 3),
        ("dye", "100UC Spectral-Dye-Density Curves",
         (320.0, 92.0, 580.0, 303.0), False, False, (), 2),
        ("sens", "100UC Spectral-Sensitivity Curves",
         (41.0, 336.0, 300.0, 545.0), False, False, (), 3),
        ("mtf", "100UC Modulation Transfer Function",
         (320.0, 336.0, 580.0, 539.0), True, True, ("R", "G", "B"), 3),
    ],
    ("e4035-100UC_400UC.pdf", 7): [
        ("char", "400UC Characteristic Curves, EI 400",
         (46.0, 76.0, 300.0, 284.0), False, False, ("R", "G", "B"), 3),
        ("char", "400UC Characteristic Curves, EI 800 (Push 1)",
         (320.0, 75.0, 580.0, 282.0), False, False, ("R", "G", "B"), 3),
        ("sens", "400UC Spectral-Sensitivity Curves",
         (41.0, 320.0, 300.0, 520.0), False, False, (), 3),
        ("dye", "400UC Spectral-Dye-Density Curves",
         (320.0, 310.0, 580.0, 517.0), False, False, (), 2),
    ],
    ("e4035-100UC_400UC.pdf", 8): [
        ("mtf", "400UC Modulation Transfer Function",
         (44.0, 93.0, 300.0, 281.0), True, True, ("R", "G", "B"), 3),
    ],
}

#: (pdf basename, page) whose caption-derived panels are DISCARDED in favour of
#: the PANEL_OVERRIDES boxes above. Only E-4035, and only because its captions
#: are centred rather than left-aligned; see the note on PANEL_OVERRIDES.
PANEL_SUPPRESS = {
    ("e4035-100UC_400UC.pdf", 6),
    ("e4035-100UC_400UC.pdf", 7),
    ("e4035-100UC_400UC.pdf", 8),
}


def find_panels(page, pdf_name=None):
    """Locate every captioned plot on ``page``.

    Returns a list of ``(kind, caption_text, box, log_x, log_y, letters)``.
    Each tuple is (kind, caption, box, log_x, log_y, letters, expected_traces).
    The box's bottom edge is clipped at the next caption in the same column,
    so a two-panels-in-a-column page cannot leak one panel's tick labels into
    the other's axis fit.
    """
    caps = []
    for blk in page.get_text("dict").get("blocks", []):
        for ln in blk.get("lines", []):
            if tuple(ln.get("dir", (1, 0))) != (1, 0):
                continue
            txt = "".join(s.get("text", "") for s in ln.get("spans", [])).strip()
            low = txt.lower()
            # ⚠ SUBSTRING, NOT PREFIX. E-7022 (February 2007) is a TWO-FILM
            # sheet and captions its panels "KODAK GOLD 100 Film Characteristic
            # Curves" and "KODAK GOLD 200 Film Characteristic Curves" -- the
            # panel kind is at the END of the line, not the start. A prefix
            # match found neither, and the page reported only its spectral and
            # dye panels while two complete three-channel characteristic
            # figures sat there unread. The longest key is preferred so that a
            # line containing two kind names cannot be claimed by the shorter.
            hit = max((k for k in CAPTIONS if k in low), key=len, default=None)
            if hit is None:
                continue
            caps.append((ln["bbox"][0], ln["bbox"][1], txt, hit))
    out = []
    for cx, cy, txt, key in caps:
        kind, lx, ly, letters, expect = CAPTIONS[key]
        below = [c[1] for c in caps
                 if abs(c[0] - cx) <= COL_TOL_PT and c[1] > cy + 20.0]
        y1 = (min(below) - 8.0) if below else min(cy + BOX_DY1, page.rect.y1 - 20.0)
        box = (cx + BOX_DX0, cy + BOX_DY0, cx + BOX_DX1, y1)
        out.append((kind, txt, box, lx, ly, letters, expect))
    if pdf_name is not None:
        if (pdf_name, page.number + 1) in PANEL_SUPPRESS:
            out = []
        out += PANEL_OVERRIDES.get((pdf_name, page.number + 1), [])
    return out


#: The eleven documents this pass covers, with the products each one is FOR.
#: Deliberately explicit rather than globbed: the task's own rule is that a
#: figure belongs to the film its own page names, and two of these documents
#: (E-190 in both vintages) carry six and five different films respectively on
#: separate pages, while three others are re-issues of the same publication
#: number with different content. A glob would blur all of that.
DOCS = [
    ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", "E-190", "2003-05",
     "PORTRA 160NC/160VC/400NC/400VC/400UC/800"),
    ("e190-Portra-2006.pdf", "E-190", "2006-10",
     "PORTRA 160NC/160VC/400NC/400VC/800"),
    ("e2468-Portra_100T.pdf", "E-2468", "2006-10", "PORTRA 100T"),
    ("e4051_portra_160.pdf", "E-4051", "2016-02", "PORTRA 160"),
    ("e4050_portra_400.pdf", "E-4050", "2016-02", "PORTRA 400"),
    ("portra400-techpub-e4050.pdf", "E-4050", "2010-09", "PORTRA 400"),
    ("e4040_portra_800.pdf", "E-4040", "2016-02", "PORTRA 800"),
    ("E7019_en-Ultra_Max_400.pdf", "E-7019", "2007-02", "ULTRA MAX 400"),
    ("E7023_max_400.pdf", "E-7023", "2016-02", "ULTRA MAX 400"),
    ("E7024-Ultra_Max_800.pdf", "E-7024", "2007-12", "ULTRA MAX 800"),
    ("E7022-1.pdf", "E-7022", "2022-03", "GOLD 200"),
    ("E7022-Gold_100_200.pdf", "E-7022", "2007-02", "GOLD 100 + GOLD 200"),
    ("e29-Pro_100T_PRT.pdf", "E-29", "1999-04", "Pro 100T / PRT"),
    # ⚠ ADDED 2026-09-03. Supplied by the owner after the 2026-08-26 pass, and
    # it is the sheet BOTH ULTRA COLOR profiles already cited while it was not
    # in the corpus -- `_PROCESS_VARIANTS["KODAK_ULTRA_COLOR_400UC"]` says so in
    # as many words. Nine panels, all nine now reading.
    ("e4035-100UC_400UC.pdf", "E-4035", "2007-05",
     "ULTRA COLOR 100UC + 400UC"),
]


def probe(pdf, pub, date, product, pages=None):
    doc = pymupdf.open(os.path.join(PDF_DIR, pdf))
    print("=" * 76)
    print("%s  [%s %s]  %s" % (pdf, pub, date, product))
    for pno in range(1, doc.page_count + 1):
        if pages and pno not in pages:
            continue
        page = doc[pno - 1]
        for kind, txt, box, lx, ly, letters, expect in find_panels(page, pdf):
            pan = extract_panel(page, box, letters=letters, log_x=lx, log_y=ly,
                                expect=expect)
            if pan is None:
                print("  [SKIP] p%-2d %-38s no collinear axis pair" % (pno, txt))
                continue
            xa = (pan.x_axis.value(pan.x_span[0]), pan.x_axis.value(pan.x_span[1]))
            ya = (pan.y_axis.value(pan.y_span[1]), pan.y_axis.value(pan.y_span[0]))
            print("  p%-2d %-38s x %8.3f..%-8.3f y %7.3f..%-7.3f named=%s free=%d"
                  % (pno, txt, xa[0], xa[1], ya[0], ya[1],
                     sorted(pan.traces) or "-", len(pan.unlabelled)))
            for lab in sorted(pan.traces):
                tr = pan.traces[lab]
                print("        %-3s n=%-4d x %8.3f..%-8.3f y %7.3f..%-7.3f"
                      % (lab, len(tr), min(p[0] for p in tr),
                         max(p[0] for p in tr), min(p[1] for p in tr),
                         max(p[1] for p in tr)))
            for i, tr in enumerate(pan.unlabelled):
                print("        f%-2d n=%-4d x %8.3f..%-8.3f y %7.3f..%-7.3f"
                      % (i, len(tr), min(p[0] for p in tr),
                         max(p[0] for p in tr), min(p[1] for p in tr),
                         max(p[1] for p in tr)))
    doc.close()



# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
#: What a re-run must reproduce, recorded 2026-08-26. Keys are (pdf, page,
#: panel kind). Each value is what the ADOPTED numbers were derived from, so a
#: drift here means the database and the source have parted company.
#:
#: The tolerances are deliberately tight -- 0.002 in density, 0.003 in gamma,
#: 0.5 cycles/mm in f50 -- because nothing in this pipeline is stochastic. A
#: loose tolerance here would hide exactly the class of defect this module kept
#: producing while it was being written: a mirrored axis, a mis-snapped tick, a
#: chain welded across a crossing. Every one of those moved a number by far more
#: than these bounds, and every one of them looked plausible.
EXPECTED = {
    # The independent cross-check, and the reason it is listed twice: PORTRA
    # 160NC's panel appears in E-190 (2003) p9 and E-190 (2006) p8, in two files
    # with different md5s, and the reader must return the same numbers from
    # both. It does, to four decimals. Nothing else in this module validates the
    # tick fitting, the subpath splitting and the letter matching all at once.
    ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 9, "char"): {
        "R": (0.2044, 0.5279), "G": (0.6089, 0.5501), "B": (0.8116, 0.6078)},
    ("e190-Portra-2006.pdf", 8, "char"): {
        "R": (0.2044, 0.5279), "G": (0.6089, 0.5501), "B": (0.8116, 0.6078)},
    # ⚠ THE COPY-PASTE DEFECT, PINNED. E-2468's characteristic panel is
    # PORTRA 160VC's figure F009_0154AC, and these are 160VC's numbers, not
    # PORTRA 100T's. The entry exists so that the defect stays VISIBLE: if a
    # later edition of E-2468 ever carries 100T's own curves these values stop
    # reproducing, and that is precisely when the profile should be revisited.
    ("e2468-Portra_100T.pdf", 5, "char"): {
        "R": (0.2045, 0.5809), "G": (0.6087, 0.6050), "B": (0.8121, 0.6691)},
    ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 10, "char"): {
        "R": (0.2045, 0.5809), "G": (0.6087, 0.6050), "B": (0.8121, 0.6691)},
    # The four adopted curve sets.
    ("e4051_portra_160.pdf", 4, "char"): {
        "R": (0.2163, 0.4939), "G": (0.6303, 0.5423), "B": (0.8573, 0.6224)},
    ("e4050_portra_400.pdf", 4, "char"): {
        "R": (0.2482, 0.5562), "G": (0.6653, 0.5486), "B": (0.8805, 0.6360)},
    ("E7023_max_400.pdf", 4, "char"): {
        "R": (0.2908, 0.5071), "G": (0.6968, 0.5299), "B": (0.9849, 0.6023)},
    # ⚠ RED REPINNED 2026-09-03: dmin 0.3313 -> 0.3128, gamma unchanged at
    # 0.5185. This panel draws its red curve as a 9-vertex flat base+fog stub
    # (x 378.5-400.7) plus the curve proper (x 400.7-526.1); the stub spans 15 %
    # of the frame, the span filter dropped it, and the surviving piece made the
    # trace count correct -- so nothing complained while `measure_char` read the
    # plateau off a trace that did not contain the plateau. With the stub
    # chained back on, dmin is the drawn plateau. Gamma never depended on it.
    # ⚠ KODAK_ULTRA_MAX_800's stored red dmin is the OLD value and is NOT
    # changed here: that is a database edit and the owner decides it.
    ("E7024-Ultra_Max_800.pdf", 3, "char"): {
        "R": (0.3128, 0.5185), "G": (0.7093, 0.5276), "B": (1.0267, 0.6103)},
    ("E7022-1.pdf", 4, "char"): {
        "R": (0.2601, 0.4922), "G": (0.6652, 0.5086), "B": (0.9674, 0.5970)},
    ("e190-Portra-2006.pdf", 12, "char"): {
        "R": (0.2200, 0.5372), "G": (0.6551, 0.5185), "B": (1.0072, 0.6448)},
    # ---- added 2026-08-26f, the two-document follow-up -------------------
    # ⚠ E-7022's TWO-FILM EDITION IS THE REGRESSION TEST FOR THE CAPTION
    # MATCHER. Both panels below were INVISIBLE until the matcher moved from
    # startswith() to substring: this sheet captions them "KODAK GOLD 100 Film
    # Characteristic Curves" and "KODAK GOLD 200 Film Characteristic Curves",
    # with the panel kind at the END of the line. Two complete three-channel
    # figures were being skipped in silence. Both are pinned, and the GOLD 200
    # one is a CROSS-DOCUMENT check rather than an adoption: its numbers must
    # keep agreeing with the 2022 edition's uncaptioned panel, which this
    # module finds by geometry override instead. Two different location
    # mechanisms, fifteen years apart, on the same emulsion.
    ("E7022-Gold_100_200.pdf", 4, "char"): {
        "R": (0.2618, 0.4992), "G": (0.6666, 0.5148), "B": (0.9769, 0.6052)},
    ("e29-Pro_100T_PRT.pdf", 4, "char"): {
        "R": (0.2145, 0.5584), "G": (0.6337, 0.6218), "B": (0.8850, 0.6570)},
}

#: The GOLD 200 panel on the 2007 two-film sheet, checked separately because
#: EXPECTED is keyed by (pdf, page, kind) and that sheet holds two panels of
#: the same kind on one page. Values are dmin/gamma per channel.
#:
#: ⚠ EXTENDED 2026-08-31 (queue K3) TO THE PUSH PANELS, which are the other
#: case of several same-kind panels on one page -- E-190 (2006) p12 draws three
#: and E-190 (2003) p13 and p14 draw two and three. They are keyed by their
#: printed caption, which is what distinguishes them, and pinned because they
#: are now ADOPTED: `film_profiles._PROCESS_VARIANTS` stores the E-190 (2006)
#: pair for PORTRA 800 and the E-190 (2003) p13 pair for 400UC.
#:
#: ⚠ AND THE UNADOPTED READINGS ARE PINNED TOO, DELIBERATELY. The E-190 (2003)
#: PORTRA 800 panels and the E-4040 (2016) ones are NOT stored -- see the
#: variant records for why -- but they are the evidence that three editions of
#: one film's push disagree, and an unpinned disagreement is a claim nobody can
#: re-check. Red gamma at EI 1600 reads 0.6883 (2003), 0.6100 (2006) and 0.6341
#: (2016) under labels that all say the same film and the same push.
EXPECTED_SECOND_CHAR = {
    ("E7022-Gold_100_200.pdf", 4, "GOLD 200"): {
        "R": (0.2593, 0.5003), "G": (0.6640, 0.5157), "B": (0.9687, 0.5974)},
    # ---- ADOPTED: PORTRA 800's pushes, E-190 (2006) p12 --------------------
    # ⚠ RED REPINNED 2026-09-03 ON BOTH PANELS, same cause as E-7024 p3 above:
    # a dropped flat base+fog stub on the red trace. EI 1600 dmin 0.2779 ->
    # 0.2569 (gamma 0.6100 -> 0.6094); EI 3200 dmin 0.3173 -> 0.3011 (gamma
    # 0.6918 -> 0.6907). ⚠ THESE TWO ARE ADOPTED -- `_PROCESS_VARIANTS` stores
    # them for PORTRA 800 -- and the stored records still carry the OLD red
    # dmins. Left as they are pending the owner's decision; the gamma shift is
    # inside G_TOL's spirit but the 0.021 / 0.016 D dmin shift is not.
    ("e190-Portra-2006.pdf", 12, "Characteristic Curves, EI 1600 (Push 1)"): {
        "R": (0.2569, 0.6094), "G": (0.6631, 0.6019), "B": (1.0030, 0.7050)},
    ("e190-Portra-2006.pdf", 12, "Characteristic Curves, EI 3200 (Push 2)"): {
        "R": (0.3011, 0.6907), "G": (0.6938, 0.7162), "B": (1.0468, 0.7872)},
    # ⚠ THE ANCHOR THAT MAKES THOSE TWO A PUSH. This page's EI 800 panel must
    # keep reproducing the profile's own stored curves; if it stops, the two
    # records above are no longer a delta from anything.
    ("e190-Portra-2006.pdf", 12, "Characteristic Curves, EI 800"): {
        "R": (0.2200, 0.5372), "G": (0.6551, 0.5185), "B": (1.0072, 0.6448)},
    # ---- ADOPTED: 400UC, E-190 (2003) p13, both panels ---------------------
    ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 13,
     "Characteristic Curves, EI 400"): {
        "R": (0.3338, 0.5505), "G": (0.7630, 0.5761), "B": (1.0508, 0.6655)},
    ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 13,
     "Characteristic Curves, EI 800 (Push 1)"): {
        "R": (0.3989, 0.6149), "G": (0.7923, 0.6582), "B": (1.0752, 0.7570)},
    # ---- NOT ADOPTED, pinned as the evidence of the disagreement -----------
    ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 14,
     "Characteristic Curves, EI 800"): {
        "R": (0.3168, 0.5638), "G": (0.7462, 0.5989), "B": (1.0323, 0.6841)},
    ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 14,
     "Characteristic Curves, EI 1600 (Push 1)"): {
        "R": (0.3599, 0.6883), "G": (0.7874, 0.7115), "B": (1.0920, 0.7720)},
    ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 14,
     "Characteristic Curves, EI 3200 (Push 2)"): {
        "R": (0.3932, 0.7862), "G": (0.8214, 0.8018), "B": (1.1504, 0.8254)},
    ("e4040_portra_800.pdf", 4, "Characteristic Curves, EI 1600 (Push 1)"): {
        "R": (0.2404, 0.6341), "G": (0.6614, 0.6364), "B": (1.0139, 0.7282)},
    ("e4040_portra_800.pdf", 4, "Characteristic Curves, EI 3200 (Push 2)"): {
        "R": (0.3050, 0.6909), "G": (0.7049, 0.7177), "B": (1.0625, 0.7855)},
    # ---- E-4035 (May 2007), ULTRA COLOR 100UC and 400UC, added 2026-09-03 ---
    # ⚠ THE SHEET BOTH PROFILES ALREADY CITED. Neither stored curve set matches
    # it: 100UC and 400UC both store dmin near 0.20 / 0.19 / 0.19 -- three
    # near-equal values, i.e. NO ORANGE MASK -- against the 0.30 / 0.71 / 1.00
    # and 0.34 / 0.77 / 1.05 below, and the stored pair differs from each other
    # by a flat +0.01 dmin / -0.004 gamma on all three channels, which is one
    # hand edit rather than two readings. Both profiles' provenance says
    # `fitted_from = 'analogy'`, so that is consistent -- the curves were never
    # from this document.
    # ⚠ THE 400UC EI 400 PANEL IS THE CROSS-DOCUMENT CHECK, and it is what says
    # the reading is the film and not the fit: E-190 (2003) p13 reads the same
    # emulsion independently at 0.3338 / 0.7630 / 1.0508 and 0.5505 / 0.5761 /
    # 0.6655, agreeing with this sheet to 0.004 in dmin and 0.001 in gamma
    # across two publications four years apart.
    # ⚠ AND THE TWO FILMS DO DIFFER, in the direction the database has backwards:
    # 400UC is HIGHER in base density (+0.037 / +0.061 / +0.059) and CONTRASTIER
    # (+0.030 / +0.010 / +0.014), where the stored pair makes 400UC the softer
    # of the two. The gamma difference being channel-dependent is what rules out
    # an axis or tracing error, which would move all three the same way.
    ("e4035-100UC_400UC.pdf", 6, "100UC Characteristic Curves"): {
        "R": (0.2992, 0.5212), "G": (0.7064, 0.5654), "B": (0.9958, 0.6510)},
    ("e4035-100UC_400UC.pdf", 7, "400UC Characteristic Curves, EI 400"): {
        "R": (0.3364, 0.5511), "G": (0.7670, 0.5754), "B": (1.0543, 0.6652)},
    ("e4035-100UC_400UC.pdf", 7,
     "400UC Characteristic Curves, EI 800 (Push 1)"): {
        "R": (0.4083, 0.6367), "G": (0.8032, 0.6729), "B": (1.0851, 0.7831)},
}

#: ⚠ A PANEL THAT MUST STAY UNREADABLE, AND WHY THAT IS AN ASSERTION RATHER
#: THAN A GAP. E-4040 (2016) p4's "Characteristic Curves, EI 800" cannot be
#: calibrated: its printed LOG EXPOSURE axis reads -4.0, -2.0, -3.0, -1.0, 0.0,
#: 1.0 across six evenly spaced ticks -- the second and third labels transposed
#: in Kodak's own artwork, confirmed against the page rendered at 6x, so this
#: is the plate and not the text layer. `_sign_ticks` refuses it because no
#: signing of those labels is collinear, which is the correct outcome: the
#: alternative is a fitted axis wrong by a decade in the middle of the plot.
#:
#: Asserted rather than merely noted, because the day this panel starts reading
#: is the day either a corrected edition arrived or the tick fitter began
#: tolerating a non-collinear axis, and those need opposite responses. It also
#: decides a stored number: it is why PORTRA 800's push sets are taken from
#: E-190 (2006), whose EI 800 panel DOES read and reproduces the profile's own
#: curves, rather than from the newest sheet.
EXPECTED_UNREADABLE = (
    ("e4040_portra_800.pdf", 4, "Characteristic Curves, EI 800",
     "the printed exposure axis transposes its -2.0 and -3.0 labels"),
)

#: f50 in cycles/mm per channel for the three adopted MTF sets. A None entry
#: means the traced curve NEVER REACHES 50 % within the plotted frequency range
#: -- a censored reading, not a missing one, and it must stay distinguishable
#: from a number (E-190 2003 p9's blue channel is still at 55 % at 80 c/mm).
EXPECTED_MTF = {
    ("e4051_portra_160.pdf", 4): (34.9, 65.8, 56.6),
    ("e4050_portra_400.pdf", 4): (38.2, 58.6, 69.3),
    ("e4040_portra_800.pdf", 5): (33.7, 54.8, 72.1),
    ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 9): (49.1, 73.3, None),
    # ⚠ E-4035's BLUE IS CENSORED, NOT MISSING, on both pages: the blue trace is
    # still above 50 % where the plotted frequency range ends, exactly as on
    # E-190 (2003) p9. 100UC is the sharper film of the two on every channel
    # that yields a number, which is the expected direction for the slower
    # emulsion and is the only thing here the stored MTFSpecs already get right.
    ("e4035-100UC_400UC.pdf", 6): (40.4, 69.0, None),
    ("e4035-100UC_400UC.pdf", 8): (37.5, 62.3, None),
}

#: The two dye panels RECOVERED on 2026-09-03 by `_is_stroke_row`, pinned as
#: readings rather than adoptions. Both had been recorded as refusals because a
#: welded row of caption-underline strokes made the panel report three traces;
#: neither refusal was about the curves. Peaks are (neutral, D-min).
#:
#: ⚠ E-4050 p4 IS VERIFIED AGAINST THE PAGE: printed neutral 1.89 at about
#: 448 nm reads 1.890 at 448.3, printed D-min 0.82 at about 443 reads 0.820 at
#: 443.7, both curves spanning the full 400-700 nm at 721 vertices each.
RECOVERED_DYE = {
    ("e4050_portra_400.pdf", 4): (1.890, 0.820),
    # ⚠ E-4035 p6 (100UC) reads; p7 (400UC) does NOT -- see DECLINED_DYE. One
    # sheet, one panel good and one mis-calibrated, which is why the extent
    # check in `assign_dye_pair` is per panel and not per document.
    ("e4035-100UC_400UC.pdf", 6): (1.940, 1.436),
}

#: Dye panels that STILL decline, each for a reason that is about the trace
#: rather than about the reader, asserted so that a later change which starts
#: accepting one has to say so.
#:
#: ⚠ E-7019 p4 is the near miss. `_is_stroke_row` recovered its two curves and
#: they order correctly -- neutral peak 2.434, D-min 1.837, both at 400.1 nm --
#: but the D-min trace stops at 650.9 nm where the neutral reaches 699.8, so
#: 49 nm of the red end is missing and `assign_dye_pair`'s extent check
#: declines the pair. That is the right answer: half a curve resampled onto a
#: 5 nm grid would silently invent nothing and silently omit a fifth of the
#: spectrum.
#: ⚠ E-4035 p7 (400UC) is the mis-calibrated one, described in
#: `assign_dye_pair`: the axis fit lands 4.6 nm wide because a D-min vertex
#: passes for the 400 nm tick. 100UC's panel on p6 of the same sheet is fine,
#: so this is one panel's typesetting, not the document's.
DECLINED_DYE = (
    ("E7019_en-Ultra_Max_400.pdf", 4),
    ("e4035-100UC_400UC.pdf", 7),
)

#: Peak diffuse spectral density of the neutral and D-min curves for the four
#: adopted dye pairs. Cross-checked against 300 dpi renders of the panels:
#: E-4051's printed peaks read 1.80 and 0.825 at 450 nm against 1.798 and 0.825
#: traced, and E-190 (2003) p9's read 1.99 and 1.63 at 400 nm against 1.990 and
#: 1.630.
EXPECTED_DYE = {
    ("e4051_portra_160.pdf", 4): (1.798, 0.825),
    ("e4040_portra_800.pdf", 5): (1.742, 0.996),
    ("E7023_max_400.pdf", 4): (1.800, 0.989),
    ("E7022-1.pdf", 4): (1.842, 0.988),
    ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 9): (1.990, 1.630),
    # ⚠ THE SHARED-PANEL PROOF. E-7022 (February 2007) prints ONE dye panel for
    # TWO films; E-7022 (March 2022) prints one for GOLD 200 alone. They are the
    # same artwork -- identical peaks here, and 0.0005 D worst / 0.00009 D rms
    # over 59 resampled points of BOTH curves. That is what fixes the shared
    # panel's identity as GOLD 200's and keeps GOLD 100 honestly empty.
    ("E7022-Gold_100_200.pdf", 4): (1.842, 0.988),
    # ⚠ REPINNED 2026-09-03 FROM (1.616, 0.871) TO (2.121, 1.611), AND THE OLD
    # PAIR WAS NOT THIS PANEL'S PEAK AT ALL. Both curves on E-29 p4 rise almost
    # vertically at the left edge -- the panel's own 400 nm samples are 2.12 and
    # 1.61, verified on a 150 dpi render of the page -- and that rise is drawn
    # as five short subpaths (spans 18, 13, 12, 4 and 2 pt) which the span
    # filter discarded. What survived began at 373 and 385 pt, so the pinned
    # "peaks" were the values at roughly 420 nm and the real ones were 0.51 and
    # 0.74 D higher. The reading changed because `extract_panel` now prefers a
    # chained reading that covers more of the x axis at equal trace count; see
    # the comment there. ⚠ THE STORED ARRAY FOR THIS FILM IS THEREFORE SHORT AT
    # ITS BLUE END and has NOT been touched here -- that is a database change
    # and needs the owner's decision, not a reader commit.
    ("e29-Pro_100T_PRT.pdf", 4): (2.121, 1.611),
}

D_TOL, G_TOL, F_TOL, DYE_TOL = 0.002, 0.003, 0.5, 0.002


def run_assert():
    """Re-derive every adopted number and fail loudly on any drift."""
    bad, checked = [], 0
    for (pdf, pno, kind), want in sorted(EXPECTED.items()):
        doc = pymupdf.open(os.path.join(PDF_DIR, pdf))
        page = doc[pno - 1]
        got = None
        for k, _txt, box, lx, ly, letters, exp in find_panels(page, pdf):
            if k != kind:
                continue
            pan = extract_panel(page, box, letters=letters, log_x=lx, log_y=ly,
                                expect=exp)
            if pan is None or not pan.traces:
                continue
            got = pan
            break
        if got is None:
            bad.append("%s p%d %s: panel not located" % (pdf, pno, kind))
            doc.close()
            continue
        for ch, (wd, wg) in sorted(want.items()):
            tr = got.traces.get(ch)
            m = measure_char(tr) if tr else None
            if m is None:
                bad.append("%s p%d %s: unmeasurable" % (pdf, pno, ch))
                continue
            checked += 1
            if abs(m["dmin"] - wd) > D_TOL or abs(m["gamma"] - wg) > G_TOL:
                bad.append("%s p%d %s: dmin %.4f vs %.4f, gamma %.4f vs %.4f"
                           % (pdf, pno, ch, m["dmin"], wd, m["gamma"], wg))
        doc.close()

    for (pdf, pno), want in sorted(EXPECTED_MTF.items()):
        doc = pymupdf.open(os.path.join(PDF_DIR, pdf))
        page = doc[pno - 1]
        for k, _t, box, lx, ly, letters, exp in find_panels(page, pdf):
            if k != "mtf":
                continue
            pan = extract_panel(page, box, letters=letters, log_x=lx, log_y=ly,
                               expect=exp)
            if pan is None:
                bad.append("%s p%d mtf: panel not located" % (pdf, pno))
                break
            for ch, wf in zip("RGB", want):
                f = f_at_response(pan.traces[ch], 50.0) if ch in pan.traces                     else None
                checked += 1
                if (wf is None) != (f is None):
                    bad.append("%s p%d mtf %s: censored/measured flipped "
                               "(%s vs %s)" % (pdf, pno, ch, f, wf))
                elif wf is not None and abs(f - wf) > F_TOL:
                    bad.append("%s p%d mtf %s: f50 %.1f vs %.1f"
                               % (pdf, pno, ch, f, wf))
            break
        doc.close()

    for (pdf, pno), (wn, wd) in sorted(EXPECTED_DYE.items()):
        doc = pymupdf.open(os.path.join(PDF_DIR, pdf))
        page = doc[pno - 1]
        for k, _t, box, lx, ly, letters, exp in find_panels(page, pdf):
            if k != "dye":
                continue
            pan = extract_panel(page, box, letters=letters, log_x=lx, log_y=ly,
                                expect=exp)
            pr = assign_dye_pair(pan.unlabelled) if pan else None
            if not pr:
                bad.append("%s p%d dye: pair refused" % (pdf, pno))
                break
            gn = max(q[1] for q in pr["neutral"])
            gd = max(q[1] for q in pr["dmin"])
            checked += 2
            if abs(gn - wn) > DYE_TOL or abs(gd - wd) > DYE_TOL:
                bad.append("%s p%d dye: peaks %.3f/%.3f vs %.3f/%.3f"
                           % (pdf, pno, gn, gd, wn, wd))
            break
        doc.close()

    # The second characteristic panel on the two-film sheet.
    for (pdf, pno, want), exp_ch in sorted(EXPECTED_SECOND_CHAR.items()):
        doc = pymupdf.open(os.path.join(PDF_DIR, pdf))
        page = doc[pno - 1]
        hit = None
        for k, txt, box, lx, ly, letters, exp in find_panels(page, pdf):
            if k == "char" and want in txt:
                hit = extract_panel(page, box, letters=letters, log_x=lx,
                                    log_y=ly, expect=exp)
                break
        if hit is None:
            bad.append("%s p%d %s: panel not located" % (pdf, pno, want))
            doc.close()
            continue
        for ch, (wd, wg) in sorted(exp_ch.items()):
            m = measure_char(hit.traces[ch]) if ch in hit.traces else None
            checked += 1
            if m is None:
                bad.append("%s %s %s: unmeasurable" % (pdf, want, ch))
            elif abs(m["dmin"] - wd) > D_TOL or abs(m["gamma"] - wg) > G_TOL:
                bad.append("%s %s %s: dmin %.4f vs %.4f, gamma %.4f vs %.4f"
                           % (pdf, want, ch, m["dmin"], wd, m["gamma"], wg))
        doc.close()

    # ⚠ AND ONE PANEL IS ASSERTED TO STAY UNREADABLE. See EXPECTED_UNREADABLE:
    # a Kodak plate with two transposed axis labels, which the tick fitter must
    # go on refusing. A silent success here would be a fitted axis wrong by a
    # decade, and it would move an adopted push set to the wrong document.
    for pdf, pno, want, why in EXPECTED_UNREADABLE:
        doc = pymupdf.open(os.path.join(PDF_DIR, pdf))
        page = doc[pno - 1]
        found = False
        for k, txt, box, lx, ly, letters, exp in find_panels(page, pdf):
            if k != "char" or want not in txt:
                continue
            found = True
            checked += 1
            if extract_panel(page, box, letters=letters, log_x=lx, log_y=ly,
                             expect=exp) is not None:
                bad.append("%s p%d %s: now READS, but %s -- either the "
                           "edition changed or the tick fitter stopped "
                           "refusing a non-collinear axis"
                           % (pdf, pno, want, why))
            break
        if not found:
            bad.append("%s p%d %s: the panel this check is about is no longer "
                       "even located" % (pdf, pno, want))
        doc.close()

    # ⚠ THE TWO "REFUSALS" WERE NEVER ABOUT THE DATA, AND ON 2026-09-03 THEY
    # STOPPED BEING REFUSALS. Both panels were rejected by `assign_dye_pair`
    # for returning three traces instead of two -- the third being a welded row
    # of caption-underline strokes, the artefact `_is_stroke_row` now removes.
    # With it gone both read cleanly and were checked against 150 dpi renders of
    # their pages: E-4050 p4's neutral peaks 1.890 at 448.3 nm against a printed
    # 1.89 at about 448, its D-min 0.820 at 443.7 against 0.82, both spanning
    # the full 400-700 nm. So these are two spectral-dye-density pairs the
    # corpus can now use, not two panels that had to be declined.
    # ⚠ NEITHER IS ADOPTED HERE. They are pinned below as READINGS; putting
    # them into PORTRA 400's and ULTRA MAX 400's profiles is a database change
    # and needs the owner's decision.
    for pdf, pno in sorted(RECOVERED_DYE):
        doc = pymupdf.open(os.path.join(PDF_DIR, pdf))
        page = doc[pno - 1]
        for k, _t, box, lx, ly, letters, exp in find_panels(page, pdf):
            if k != "dye":
                continue
            pan = extract_panel(page, box, letters=letters, log_x=lx, log_y=ly,
                               expect=exp)
            pr = assign_dye_pair(pan.unlabelled) if pan else None
            checked += 1
            if pr is None:
                bad.append("%s p%d dye: no longer reads; it was recovered on "
                           "2026-09-03 and must stay recovered" % (pdf, pno))
                break
            want = RECOVERED_DYE[(pdf, pno)]
            got = (max(p[1] for p in pr["neutral"]),
                   max(p[1] for p in pr["dmin"]))
            if (abs(got[0] - want[0]) > DYE_TOL
                    or abs(got[1] - want[1]) > DYE_TOL):
                bad.append("%s p%d dye: peaks %.3f/%.3f vs %.3f/%.3f"
                           % (pdf, pno, got[0], got[1], want[0], want[1]))
            break
        doc.close()

    for pdf, pno in DECLINED_DYE:
        path = os.path.join(PDF_DIR, pdf)
        if not os.path.exists(path):
            continue
        doc = pymupdf.open(path)
        page = doc[pno - 1]
        for k, _t, box, lx, ly, letters, exp in find_panels(page, pdf):
            if k != "dye":
                continue
            pan = extract_panel(page, box, letters=letters, log_x=lx, log_y=ly,
                               expect=exp)
            checked += 1
            if pan is not None and assign_dye_pair(pan.unlabelled) is not None:
                bad.append("%s p%d dye: now ACCEPTED, and it is recorded as "
                           "declined for a reason about the trace" % (pdf, pno))
            break
        doc.close()

    if bad:
        print("[FAIL] " + "; ".join(bad))
        return 1
    # ⚠ COUNTED FROM THE TABLES, NOT TYPED. Every one of these numbers was
    # wrong at least once while this module was being extended, and a summary
    # that overstates its own coverage is worse than no summary.
    n_panels = len(EXPECTED) + len(EXPECTED_SECOND_CHAR)
    n_pairs = 3 * n_panels
    n_dye = len(EXPECTED_DYE) + len(RECOVERED_DYE)
    n_docs = len({pdf for pdf, *_ in list(EXPECTED) + list(EXPECTED_MTF)
                  + list(EXPECTED_DYE) + list(RECOVERED_DYE)
                  + list(EXPECTED_SECOND_CHAR)})
    print("[OK] %d values re-derived from %d KODAK still-film sheets: %d "
          "characteristic dmin/gamma pairs across %d panels, %d MTF f50 "
          "readings with the censored ones kept censored, %d dye-pair peaks, %d "
          "dye panels still declining and 1 panel still unreadable. Six of the "
          "checks are cross-document rather than per-number: PORTRA 160NC read "
          "independently from both E-190 vintages; GOLD 200 read from the 2007 "
          "sheet by CAPTION and from the 2022 sheet by GEOMETRY OVERRIDE, "
          "fifteen years and two location mechanisms apart; the shared 2007 "
          "dye panel pinned to the peaks the 2022 GOLD-200-only sheet prints, "
          "which is what keeps GOLD 100 empty; E-2468's characteristic panel "
          "pinned as PORTRA 160VC's figure F009_0154AC so that Kodak's "
          "copy-paste defect stays visible; E-190 (2006) p12's EI 800 panel "
          "pinned to PORTRA 800's own stored curves, which is what makes the "
          "two push sets beside it a PUSH and not an edition difference; and "
          "E-4040 (2016) p4's EI 800 panel asserted to stay UNREADABLE, "
          "because Kodak printed its exposure axis -4.0 / -2.0 / -3.0 / -1.0 "
          "/ 0.0 / 1.0 and a fitter that accepted that would be wrong by a "
          "decade mid-plot"
          % (checked, n_docs, n_pairs, n_panels, 3 * len(EXPECTED_MTF),
             n_dye, len(DECLINED_DYE)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", action="store_true",
                    help="locate panels and print their extents; adopt nothing")
    ap.add_argument("--doc", action="append",
                    help="restrict to documents whose filename contains this")
    ap.add_argument("--page", action="append", type=int,
                    help="restrict to these page numbers")
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    help="re-derive every adopted number and fail on drift")
    ap.add_argument("--root", default=".",
                    help="project root holding PDF/PROFILES")
    ns = ap.parse_args(argv)
    if ns.root != ".":
        os.chdir(ns.root)
    if ns.do_assert:
        return run_assert()
    docs = [d for d in DOCS
            if not ns.doc or any(s.lower() in d[0].lower() for s in ns.doc)]
    if ns.probe:
        for pdf, pub, date, product in docs:
            probe(pdf, pub, date, product, pages=ns.page)
        return 0
    print("nothing to do: pass --probe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
