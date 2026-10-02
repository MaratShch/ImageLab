"""Callier Q against diffuse density and gamma, from Mees FIG. 179.

WHAT THIS READS
---------------
Mees, *The Theory of the Photographic Process* (Macmillan), Chapter XVII "The
Measurement of Density", **printed page 643, FIG. 179** -- "Curves showing the
relation between Q, gamma, and the diffuse density". Five curves, one per
development gamma: **0.21, 0.37, 0.69, 1.20, 1.65**.

⚠ READ THIS BEFORE USING A SINGLE NUMBER FROM HERE.

**1. NO MEASUREMENT GEOMETRY IS STATED, ANYWHERE IN THE CHAPTER.** Callier's Q
is the ratio of specular to diffuse density, and the specular half is
meaningless without the collection angle -- that is the whole thesis behind
splitting `FilmProfile.callier_q` from `AlgoControls::scannerSpecular` (queue
C22). The chapter describes the two methods qualitatively (FIG. 178: an
integrating sphere at a distance reads specular, in contact reads diffuse) and
never gives an angle, an aperture or an f-number. Searched: the whole book's
text layer for angle / aperture / steradian / cone / numerical near the Callier
discussion -- two hits, neither a specification.

   CONSEQUENCE: **these curves cannot calibrate `scanner_specular`.** They say
   how Q MOVES with density and gamma; they cannot say what Q IS for a stated
   reader.

**2. IT IS ONE PRINT STOCK, NOT FIVE FILMS.** The text is explicit: "the values
of Q found for the densities of sensitometric strips of MOTION-PICTURE POSITIVE
FILM". One emulsion developed to five gammas. Not a camera negative, and not a
survey across emulsions. Nothing here licenses a per-stock `callier_q`.

**3. THE PROVENANCE IS A PRIVATE COMMUNICATION.** The figure's own footnote
reads "* O. Sandvik, private communication." Unpublished, undated, and the film
is not named. Method rule 14 territory: it is Kodak-internal data relayed by a
Kodak-authored textbook, which is better than hearsay and weaker than a sheet.

**4. TABLE LXVI ON THE FACING PAGE IS NOT USED HERE AND MUST NOT BE.** That
table gives Q 1.0-1.9 for six classes of PLATE (Lippmann, lantern,
medium-speed, high-speed, mercury-intensified). Glass plates of 1909 vintage,
no geometry either, and -- decisively -- the same chapter overturns its premise
two paragraphs later: Callier "found that Q is constant for all values of
density", then Renwick & Bloch found it is not, Tuttle verified that, and THIS
figure is the demonstration. Table LXVI is the historical value the book itself
supersedes.

SO WHAT IS IT GOOD FOR
----------------------
The SHAPE. Four properties fall out of the trace and all four are things the
renderer currently gets wrong or cannot express:

  * Q COLLAPSES TO UNITY AT THE TOE, and does it inside a tenth of a density.
    The traced envelope reaches Q 1.04 at D 0.055 and is only back to 1.40 by
    D 0.10. `AlgoCallierFactor` holds Q constant, so on all 68 monochrome
    stocks it applies a condenser's full scatter gain to densities that have
    almost none of it.
  * Every curve rises to a maximum and then decays. Measured: the maximum sits
    at D 0.32 for gamma 1.20 and D 0.51 for gamma 1.65; the two low-gamma
    curves reach a plateau instead of a peak and simply stop.
  * The maximum scales with development gamma -- 1.153, 1.261, 1.475, 1.670,
    1.723 for gamma 0.21, 0.37, 0.69, 1.20, 1.65.
  * The decay above the maximum is shallow: 8 % from peak to D 2.0 at gamma
    1.65, 10 % at gamma 1.20.

That corroborates BBC T-101 Fig. 25 (Q falling ~15 % from D 0.1 to 1.0 on
Tri-X 5223) from an independent source, and adds the gamma axis T-101 only
hinted at. Two sources agreeing on a shape is this project's usual threshold for
adopting one.

⚠ NOTHING HERE IS WIRED INTO A PROFILE. This module traces, pins and reports.
`callier_q` is untouched.

METHOD
------
600 ppi grayscale page scan, `pdfimages` straight out of the PDF -- no
re-rendering, because the embedded image IS the source resolution. The plot
frame is found from the two long horizontal and two long vertical runs; the
axes are calibrated on the tick marks INSIDE the frame, least-squares, with
residuals reported (0.0016 D and 0.0009 Q rms, on 14 and 7 ticks).

⚠ TWO MEASUREMENTS, NOT ONE, BECAUSE THE PLATE IS TWO DIFFERENT PICTURES. Above
D = 0.25 the five curves are separated and are traced BY COLUMNS, one track
each. Below it the engraver drew them ON TOP OF ONE ANOTHER as a single
near-vertical stroke, so there is no per-gamma information to recover and none
is reported; that stroke is traced BY ROWS instead, which is the only way to
follow something near-vertical, and reported as one shared envelope.

⚠ EVERY CURVE IS DRAWN THROUGH ITS OWN SCATTER AND THE SCATTER IS DENSE. That,
not the geometry, is what this tracer is mostly about: markers sit on the lines,
merge with them, and leave thin slivers at their rims that a nearest-run tracker
prefers to the line itself. See MAX_RUN_PX, `follow` and `smooth` -- each of the
three carries the specific failure it exists to prevent.

⚠ THE CURVES NEVER CROSS, AND THAT IS WHAT MAKES THIS TRACTABLE. They are
stacked in gamma order over the whole plotted range, so a tracker that keeps
them ordered cannot silently swap two of them -- the failure that cost three
attempts on the BBC T-101 plate. It is asserted below rather than assumed.

Run:
    python mees_callier_q.py --probe      # geometry and diagnostics
    python mees_callier_q.py              # trace and print the table
    python mees_callier_q.py --assert     # non-zero exit on drift
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

#: The page image, extracted with:
#:   pdfimages -f 642 -l 642 -png "THE THEORY OF THE Photographic PROCESS.pdf"
#: PDF page 642 carries printed page 643. The PDF's own page numbering runs one
#: behind the printed folio throughout the volume.
PAGE = None  # set from --root at run time; see page_path()

#: Ink threshold. The scan is bilevel-ish already; anything under this is ink.
INK = 128

#: Printed ordinate labels, top to bottom, and abscissa labels left to right.
#: The abscissa prints "2 4 6 8" for 0.2-0.8 and then "1.0 1.2 ... 2.8".
Q_LABELS = (1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0)
D_LABELS = tuple(round(0.2 * i, 1) for i in range(1, 15))      # 0.2 .. 2.8

#: The five curves, in the order they stack on the plate: lowest Q first.
GAMMAS = (0.21, 0.37, 0.69, 1.20, 1.65)

#: Ink in one column between the axes, above which the column is a RULE and not
#: a tick stub. Five near-horizontal curves cut a column for a few px each; a
#: rule cuts every row it passes. Measured on this plate the two populations sit
#: an order of magnitude apart, so the threshold is not delicate.
RULE_INK_PX = 150

#: A drawn line is this thick at 600 ppi. Anything fatter is a scatter marker,
#: a label glyph or two features touching.
#:
#: ⚠ FAT IS NOT THE SAME AS WRONG ON THIS PLATE, AND TREATING IT AS WRONG IS
#: WHAT BROKE THE FIRST RUN. Every curve is drawn THROUGH its own scatter, and
#: the scatter is dense -- at D = 0.46 the gamma = 0.37 curve is nine separate
#: ink runs in one column because six markers sit on the line. Rejecting fat
#: runs outright made the tracker coast blind across its own data points and
#: die. A fat run that STRADDLES the prediction is the curve wearing a marker:
#: the tracker keeps the prediction and carries on, rather than trusting the
#: blob's centre or throwing the column away.
MAX_RUN_PX = 13

#: Runs closer than this are one feature seen through a gap in the halftone.
MERGE_GAP_PX = 6

#: How far the tracker will look from its prediction, and how fast the
#: prediction may turn. Both in pixels per column step.
#:
#: ⚠ THE WINDOW IS SET BY THE MARKERS, NOT BY THE CURVE SEPARATION, AND IT HAS
#: TO BE WIDE. When a marker finally clears the line the line reappears up to
#: 12 px from where the blob's centre left it -- a 10 px window calls that a
#: miss and gamma = 1.65 died at D = 1.34 with a perfectly clean 3 px line
#: running on ahead of it for another 1.1 D. 22 px is generous against a marker
#: and still far inside the smallest gap between two curves above TOE_MERGE_D,
#: which is about 67 px.
TRACK_WINDOW_PX = 22.0
TRACK_SLOPE_LIMIT = 2.2

#: How much of the slope survives one coasted column. See `follow`.
COAST_SLOPE_DECAY = 0.8

#: Smoothing spans, in columns. See `smooth`.
SMOOTH_MEDIAN_PX = 21
SMOOTH_MEAN_PX = 9

#: Consecutive columns with no acceptable run before a curve is declared ended.
#:
#: ⚠ THIS IS A TERMINATION CRITERION, NOT A ROBUSTNESS KNOB, AND IT IS TIGHT ON
#: PURPOSE. Each curve stops where its emulsion ran out of density -- gamma 0.21
#: at D ~ 0.55, gamma 1.65 at ~2.42 -- and every one of them has its "gamma = x"
#: CAPTION SET INSIDE THE FRAME, 24 to 50 px past the end, at the height of the
#: line it labels. A generous miss limit does not make the tracker robust; it
#: makes it coast over the gap and latch onto the caption, then report the
#: caption's height as Q. 12 px is under the smallest gap on the plate.
TRACK_MISS_LIMIT = 12

#: Densities the traced curves are reported on. Chosen to straddle the peak and
#: to include the two densities the other sources speak about: 0.3 (this
#: figure's own stated maximum) and 1.0 (T-101's upper comparison point).
REPORT_D = (0.30, 0.40, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.40)

#: ⚠ THE FIVE CURVES ARE DRAWN COINCIDENT BELOW ABOUT THIS DENSITY. Not close
#: -- coincident, one stroke on the plate, rising near-vertically from Q ~ 1.04.
#: Nothing can separate them there because the engraver did not. Column tracing
#: is refused below this and the toe is measured separately, as one curve, by
#: rows. Anything that claimed a per-gamma Q at D = 0.05 would be inventing it.
#:
#: ⚠ 0.25, NOT 0.15. The gamma = 1.65 and gamma = 1.20 curves are still within
#: about 5 px of one another at D = 0.18 -- the audit's own no-crossing check
#: fires there, correctly. They are 67 px apart by 0.25.
TOE_MERGE_D = 0.25

#: ⚠ THE ABSCISSA IS NET DENSITY ABOVE BASE-PLUS-FOG, AND THE CHAPTER NEVER SAYS
#: SO. This matters more than anything else on the plate, because the law it
#: feeds is referenced to `dmin` -- get it wrong and every density is offset by a
#: base. The text says only "the densities of sensitometric strips of
#: motion-picture positive film ... plotted against the diffuse densities".
#:
#: The PLATE settles it. Ink runs down to D = 0.044 carrying Q = 1.042. Any
#: motion-picture positive carries base-plus-fog of roughly 0.05 to 0.07 TOTAL,
#: so on a total-density axis that point would be at or below clear base -- where
#: there is no silver, and Q is 1.000 by definition, not 1.042. A curve cannot
#: measure 4 % of scatter from a deposit that is not there. So the axis is net.
#: ⚠ Recorded as an INFERENCE FROM THE FIGURE, not as a statement of the source.
NET_DENSITY_AXIS = True

#: Per-gamma left limit for COLUMN tracing, in density.
#:
#: ⚠ ONE GLOBAL LIMIT WAS WRONG AND HID TWO DIFFERENT FAILURES BEHIND ONE
#: NUMBER. TOE_MERGE_D = 0.25 is where the gamma 1.20 and 1.65 pair merges, and
#: for that pair it is right. The other three separate much earlier and are
#: traceable to about 0.10 -- but they fail for the OTHER reason instead: below
#: their own plateau each one dives almost vertically into the bundle, and a
#: column tracker cannot follow a vertical line. Traced past these limits they do
#: not stop, they WANDER: gamma 0.21 reports 1.16 at D 0.042, ABOVE its own
#: plateau of 1.13, having stepped onto a marker belonging to the bundle.
TRACE_LEFT_D = {0.21: 0.10, 0.37: 0.10, 0.69: 0.13, 1.20: 0.25, 1.65: 0.25}

#: Net-density grid the composed family is emitted on: 0 to 3.0 by 0.025.
#: Uniform, because the C++ side indexes it with a multiply and a floor. Fine
#: enough at the toe, which is the only place the curve moves fast -- Q travels
#: from 1.0 to about 1.43 between D 0 and D 0.10 on the middle curve.
SHAPE_D_MAX = 3.0
SHAPE_D_STEP = 0.025
SHAPE_N = int(round(SHAPE_D_MAX / SHAPE_D_STEP)) + 1        # 121

#: ⚠ THE DENSITY AT WHICH `callier_q` IS DEFINED, AND IT HAD TO BE NAMED.
#: Method rule 21: a level is meaningless until its reference density is named,
#: and the stored 1.3 / 1.25 never named one -- which was survivable only while
#: Q was held constant. The moment Q varies with density the number has to say
#: WHERE. Net density 1.0 is chosen to match `grain_sigma`, which this project
#: already normalises at net 1.0; a second convention would be a second thing to
#: remember. So `callier_q` now means **Q at net density 1.0**, and the shape is
#: normalised to exactly 1.0 there.
SHAPE_REF_D = 1.0


# ---------------------------------------------------------------------------
#  geometry
# ---------------------------------------------------------------------------
def page_path(root=None):
    return os.path.join(root or HERE, "PDF", "PROFILES", "RETRO",
                        "mees_fig179_p643.png")


def load_page(page=None):
    page = page or PAGE
    if not os.path.isfile(page):
        return None
    return np.asarray(Image.open(page).convert("L")) < INK


def find_frame(d):
    """The plot box: two long horizontals, two long verticals.

    ⚠ THE BOX IS NOT A RECTANGLE AND THE CODE MUST NOT ASSUME ONE. Measured on
    this plate: the right rule walks from x=1532 at the top to x=1521 at the
    bottom (10.5 px), the top rule from y=577 at the left to y=572 at the right,
    the bottom rule the other way by 3 px. Part scan skew, part a rule drawn by
    hand in 1942. So each edge is taken at its DARKEST row/column rather than at
    the first one over a threshold -- a threshold picks whichever end of the
    smear happens to clear it, which is how L came out as 347 when the line's
    core is 350.
    """
    H, W = d.shape
    rows = d.sum(1)
    cols = d.sum(0)
    hr = [i for i in range(H) if rows[i] > W * 0.45]
    vc = [j for j in range(W) if cols[j] > H * 0.18]
    if len(hr) < 2 or len(vc) < 2:
        return None

    def _peak(idx, weight, gap=12):
        """Group the over-threshold indices and return each group's darkest."""
        groups, cur = [], [idx[0]]
        for x in idx[1:]:
            if x - cur[-1] <= gap:
                cur.append(x)
            else:
                groups.append(cur)
                cur = [x]
        groups.append(cur)
        return [max(g, key=lambda i: weight[i]) for g in groups]

    hp = _peak(hr, rows)
    vp = _peak(vc, cols)
    if len(hp) < 2 or len(vp) < 2:
        return None
    return int(vp[0]), int(vp[-1]), int(hp[0]), int(hp[-1])


def _runs(vals, gap=4):
    """Collapse a sorted index list into run centres."""
    out = []
    s = p = None
    for x in vals:
        if s is None:
            s = p = x
        elif x - p <= gap:
            p = x
        else:
            out.append((s + p) / 2.0)
            s = p = x
    if s is not None:
        out.append((s + p) / 2.0)
    return out


def _ladder(vals):
    """The largest subset of vals that sits on one uniform ladder.

    ⚠ THIS IS THE ONLY THING THAT KEEPS A CURVE OUT OF THE TICK LIST. The stub
    scan is a local ink test, so anything that grazes the scan band scores: on
    the ordinate the gamma = 1.20 curve produces a tenth 'tick' at y = 1592,
    39 px from the real one at 1552 on an axis whose spacing is 150. A count
    check would not catch it -- it made the count RIGHT by replacing a tick the
    same curve had obscured. Ladder membership catches it; a nearest-slot
    contest then keeps whichever candidate is closer to the predicted position.
    """
    v = sorted(float(x) for x in vals)
    if len(v) < 3:
        return v
    g = float(np.median(np.diff(v)))
    if g <= 0:
        return v
    best = []
    for anchor in v:
        slots = {}
        for x in v:
            k = (x - anchor) / g
            s = int(round(k))
            if abs(k - s) > 0.3:
                continue
            want = anchor + s * g
            if s not in slots or abs(x - want) < abs(slots[s] - want):
                slots[s] = x
        sel = [slots[s] for s in sorted(slots)]
        if len(sel) > len(best):
            best = sel
    return best


def find_ticks(d, frame):
    """Tick centres just INSIDE the frame, which is where this plate puts them.

    ⚠ NOT outside it. The first attempt scanned the outer margin and picked up
    the LABEL DIGITS -- 29 'ticks' on an axis with 14, at spacings that fit
    nothing. Numerals are the densest ink near an axis and will always win a
    naive scan; the marks themselves are short stubs crossing the frame line.

    ⚠ AND THE RIGHT-HAND RULE SCORES AS A TICK. It leans left far enough by the
    bottom of the plate to sit inside the scan band, and it lands exactly on the
    tick ladder -- at D = 3.0, one step past the last printed label, 2.8. So the
    ladder cannot reject it and the count check reads 15 for 14. It is rejected
    on what actually distinguishes it: a tick is a stub near the axis, a rule
    runs the height of the plate.
    """
    L, R, T, B = frame
    ys = [y for y in range(T, B + 2) if d[y, L + 4:L + 24].sum() >= 10]
    xs = [x for x in range(L, R + 2) if d[B - 24:B - 4, x].sum() >= 10]
    yt = [v for v in _runs(ys) if T + 6 < v < B + 2]
    xt = [v for v in _runs(xs) if L + 6 < v < R - 2]

    #: A curve crossing a column contributes at most its own thickness a few
    #: times over; a vertical rule contributes the whole span. Comfortably apart.
    mid_lo, mid_hi = T + 60, B - 40
    xt = [v for v in xt
          if max(int(d[mid_lo:mid_hi, c].sum())
                 for c in range(int(round(v)) - 2, int(round(v)) + 3))
          < RULE_INK_PX]

    return _ladder(yt), _ladder(xt)


def fit_axis(pix, vals):
    """Least squares value = a*pixel + b. Returns (a, b, rms, worst)."""
    p = np.asarray(pix, dtype=np.float64)
    v = np.asarray(vals, dtype=np.float64)
    a, b = np.polyfit(p, v, 1)
    res = a * p + b - v
    return float(a), float(b), float(np.sqrt((res ** 2).mean())), \
        float(np.abs(res).max())


def calibrate(d, frame):
    """Both axes from the tick marks, with the ordinate's top tick recovered.

    ⚠ THE 1.7 TICK IS ROUTINELY MISSED AND IS RECOVERED RATHER THAN INVENTED.
    The gamma = 1.65 curve runs across it, so the stub merges into the curve and
    the stub scan returns seven ordinate ticks for eight labels. The fit is made
    on the seven that ARE clean and the eighth is then CHECKED at its predicted
    position; it is never added as a datum. If the check fails the calibration
    is refused, because a missing tick that is missing for a different reason
    means the frame or the threshold is wrong.
    """
    L, R, T, B = frame
    yt, xt = find_ticks(d, frame)

    # Abscissa: 14 labelled ticks, evenly spaced, no ambiguity.
    xt = [v for v in xt if v > L + 40]
    if len(xt) != len(D_LABELS):
        return None, "abscissa: %d ticks for %d labels" % (len(xt),
                                                           len(D_LABELS))
    ax, bx, rx, wx = fit_axis(xt, D_LABELS)

    # Ordinate: work from the bottom up; the bottom frame IS Q = 1.0.
    yt = sorted(v for v in yt if v > T + 40)
    if len(yt) < 7:
        return None, "ordinate: only %d ticks" % len(yt)
    # Sorted ascending in pixel, the clean ticks run top-to-bottom from the
    # SECOND printed label down -- the first, 1.7, is the one the curve hides.
    lower = list(Q_LABELS[1:])
    yt = yt[-len(lower):]
    ay, by, ry, wy = fit_axis(yt, lower)

    y17 = (Q_LABELS[0] - by) / ay
    ok17 = d[int(round(y17)) - 3:int(round(y17)) + 4, L + 4:L + 24].sum() >= 8
    if not ok17:
        return None, "the 1.7 tick is absent at its predicted y=%.0f" % y17

    return dict(ax=ax, bx=bx, rms_x=rx, worst_x=wx,
                ay=ay, by=by, rms_y=ry, worst_y=wy,
                n_x=len(xt), n_y=len(yt), y17=y17), None


# ---------------------------------------------------------------------------
#  curve following
# ---------------------------------------------------------------------------
def column_spans(d, x, frame, merge=MERGE_GAP_PX):
    """Ink spans in one column, inside the frame, as (lo, hi) inclusive.

    ⚠ THE INSET IS NOT COSMETIC. The bottom rule of this plate is not level --
    it sits at y = 1701 under the left of the figure and y = 1705 under the
    right -- so a fixed two-pixel inset leaves part of the rule inside the scan
    at one end and not the other, and every column at that end grows a spurious
    run one pixel from Q = 1.0. Inset past the whole tilt.
    """
    L, R, T, B = frame
    lo0, hi0 = T + 10, B - 12
    col = d[lo0:hi0, x]
    out = []
    s = None
    for i, v in enumerate(col):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append([s + lo0, i - 1 + lo0])
            s = None
    if s is not None:
        out.append([s + lo0, hi0 - 1])

    merged = []
    for span in out:
        if merged and span[0] - merged[-1][1] <= merge:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    return [(a, b) for a, b in merged]


def column_runs(d, x, frame):
    """Spans as (centre, length), the form the seed and diagnostics want."""
    return [((a + b) / 2.0, b - a + 1) for a, b in column_spans(d, x, frame)]


def follow(d, frame, x0, y0, step, x_stop=None):
    """Walk one curve from a seed, returning {x: y}.

    Slope-predicted, turn-limited, and -- the part that matters on this plate --
    tolerant of its own scatter. Three outcomes per column:

      * a THIN span near the prediction: the curve, take its centre.
      * a FAT span STRADDLING the prediction: the curve wearing one or more
        markers. Keep the prediction, count it as a hit, do not move the slope.
        The blob's centre would be the marker's centre, which is not the line.
      * nothing near the prediction: a miss. Coast, and give up after
        TRACK_MISS_LIMIT of them.

    ⚠ THE SECOND CASE IS THE WHOLE DIFFERENCE BETWEEN A TRACE AND A STUB. With
    fat spans simply refused, four of the five curves died inside 0.2 D of the
    seed -- gamma 0.21 produced five pixels of trace and a peak Q the audit then
    happily compared against a tolerance.
    """
    L, R, T, B = frame
    out = {int(x0): float(y0)}
    y = float(y0)
    slope = 0.0
    miss = 0
    x = x0
    while True:
        x += step
        if x <= L + 3 or x >= R - 3:
            break
        if x_stop is not None and ((step > 0 and x > x_stop) or
                                   (step < 0 and x < x_stop)):
            break
        pred = y + slope * step
        thin = None
        straddle = None
        for a, b in column_spans(d, x, frame):
            if a - 2 <= pred <= b + 2:
                straddle = (a, b)
            if b - a + 1 > MAX_RUN_PX:
                continue
            c = (a + b) / 2.0
            dy = abs(c - pred)
            if dy > TRACK_WINDOW_PX:
                continue
            if thin is None or dy < thin[1]:
                thin = (c, dy)

        # ⚠ INK UNDER THE PREDICTION OUTRANKS A THIN RUN THAT IS MERELY NEARBY.
        # A marker sitting on the line is read as one fat span plus, wherever
        # the halftone leaves a gap, a thin sliver at the marker's EDGE. That
        # sliver is a few pixels off the line and perfectly thin, so a
        # nearest-thin rule prefers it, and the track walks the marker's rim
        # away from the curve. Only a thin run practically on the prediction is
        # allowed to overrule ink the prediction is already standing in.
        if straddle is not None and thin is not None and thin[1] > 5.0:
            thin = None

        if thin is not None:
            miss = 0
            new = thin[0]
            s = (new - y) / float(step)
            s = max(-TRACK_SLOPE_LIMIT, min(TRACK_SLOPE_LIMIT, s))
            slope = 0.6 * slope + 0.4 * s
            y = new
        elif straddle is not None:
            # ⚠ COASTING MUST BE HELD INSIDE THE INK, AND THE SLOPE MUST DECAY
            # WHILE IT COASTS. Free-running the last known slope across a marker
            # walks the prediction out through the far side of the blob, and
            # then the tracker is off the line with nothing to snap back to:
            # gamma = 1.65 climbed 0.008 Q over nine coasted columns and died at
            # D = 0.53 with 1.9 D of curve still drawn ahead of it. Clamped into
            # the span and halving the slope each coasted column, the same nine
            # columns come out flat and the curve runs to its printed end.
            a, b = straddle
            miss = 0
            y = (a + b) / 2.0 if b - a <= 2 else min(max(pred, a + 1.0), b - 1.0)
            slope *= COAST_SLOPE_DECAY
        else:
            miss += 1
            if miss > TRACK_MISS_LIMIT:
                break
            y = pred
            continue
        out[int(x)] = y
    return out


def seed_columns(d, frame, cal):
    """Find a column where all five curves are present and separated.

    ⚠ SEEDING IS WHERE THIS PLATE LIES TO YOU. "A column with exactly five
    thin runs" picked x = 590 -- a column where the gamma = 0.21 CURVE HAS
    ALREADY ENDED and the fifth run was the first stroke of its caption. The
    seed then tracked the caption. Two things fix it: spans are merged before
    counting, so one curve's scatter is one run and not nine; and the search is
    confined to the density band where all five are known to be drawn and
    already separated, which the plate settles -- above TOE_MERGE_D, below the
    gamma = 0.21 curve's end near D = 0.55.
    """
    L, R, T, B = frame
    lo = int(round((0.38 - cal["bx"]) / cal["ax"]))
    hi = int(round((0.50 - cal["bx"]) / cal["ax"]))
    best = None
    for x in range(lo, hi + 1):
        rr = column_runs(d, x, frame)
        if len(rr) != len(GAMMAS):
            continue
        # ⚠ AND ALL FIVE MUST BE THIN. A column can hold exactly five separated
        # runs and still be useless: at x = 521 the gamma = 1.20 run is 28 px of
        # marker cluster whose centroid sits 7 px BELOW the line it contains.
        # Seeded there, the track starts off the curve and never recovers.
        if any(n > MAX_RUN_PX for _c, n in rr):
            continue
        cs = sorted(c for c, _n in rr)
        gaps = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
        if min(gaps) < 40:
            continue
        if best is None or min(gaps) > best[0]:
            best = (min(gaps), x, cs)
    return best


def trace(d, frame, cal, left_d=None):
    """Follow all five, right to their printed ends and left to a stated limit.

    `left_d` is one density for every curve (the audit table's view, where the
    five are compared on a common footing) or a per-gamma mapping (the curve
    FAMILY's view, where each is taken as far left as it is individually
    readable). Both are stated by the caller; neither is a default the tracer
    picked for itself.
    """
    seed = seed_columns(d, frame, cal)
    if seed is None:
        return None, "no column carries five separated curve runs"
    _score, x0, ys = seed
    ys.sort(reverse=True)                     # lowest Q (largest y) first
    curves = []
    for g, y0 in zip(GAMMAS, ys):
        lim = left_d[g] if isinstance(left_d, dict) else (
            TOE_MERGE_D if left_d is None else left_d)
        x_stop = int(round((lim - cal["bx"]) / cal["ax"]))
        pts = {}
        pts.update(follow(d, frame, x0, y0, +1))
        pts.update(follow(d, frame, x0, y0, -1, x_stop=x_stop))
        curves.append(pts)
    return (x0, curves), None


def trace_toe(d, frame, cal):
    """The shared near-vertical rise, traced BY ROWS because it is one stroke.

    ⚠ THIS IS NOT A FIFTH-CURVE FALLBACK, IT IS A DIFFERENT MEASUREMENT. Below
    D ~ 0.15 the engraver drew the five curves on top of one another: a single
    line climbing from Q ~ 1.04 at the left frame. There is no per-gamma
    information there to recover, so none is reported. What IS recoverable is
    the property the renderer gets wrong -- that Q collapses to unity as density
    goes to zero -- and a row scan recovers it exactly, because a near-vertical
    stroke is single-valued in y and multi-valued in x.
    """
    L, R, T, B = frame
    x_hi = int(round((TOE_MERGE_D - cal["bx"]) / cal["ax"]))

    #: ⚠ THE SCAN MUST CLEAR BOTH SETS OF TICK STUBS, AND THE TOE IS THE ONE
    #: PLACE ON THE PLATE WHERE THAT IS NOT AUTOMATIC. This measurement lives in
    #: the bottom-left corner, which is precisely where the ordinate stubs stick
    #: in from the left and the abscissa stubs stick up from the bottom. Scanned
    #: naively it reads them as the curve and reports, with a straight face,
    #: "Q = 1.10 reached at D = 0.059" -- the ordinate's own 1.1 tick -- and a
    #: minimum of Q = 1.008 at D = 0.203, which is the abscissa's 0.2 tick.
    #: Both are ink, both are in range, and neither is data.
    #:
    #: ⚠ AND THEY CANNOT BOTH BE CLEARED THE SAME WAY. The abscissa stubs sit in
    #: a band of rows below everything the bundle occupies, so a row cut removes
    #: them and costs nothing. The ordinate stubs do NOT sit in a band of
    #: columns the bundle avoids -- at Q = 1.1 the stub ends around x = L+15 and
    #: the bundle is at x = L+21, six pixels away. An x cut wide enough to clear
    #: the stub eats the toe, which is exactly what "lowest Q 1.083 at D 0.080"
    #: was: the collapse cropped off and the crop reported as the measurement.
    #: The stubs are attached to the rule and the bundle is not, so they are
    #: told apart by where the run STARTS, not by how far right it reaches.
    x_lo = L + 4
    y_lo = B - 30
    stub_x = L + 12

    out = []
    for y in range(y_lo, T + 10, -1):
        idx = [i for i, v in enumerate(d[y, x_lo:x_hi]) if v]
        if not idx:
            continue
        runs = [[idx[0], idx[0]]]
        for i in idx[1:]:
            if i - runs[-1][1] > MERGE_GAP_PX:
                runs.append([i, i])
            else:
                runs[-1][1] = i
        # The leftmost run that is not welded to the ordinate rule. Anything
        # further right at this height is a marker on a curve that has already
        # separated from the bundle.
        run = next((r for r in runs if x_lo + r[0] >= stub_x), None)
        if run is None or run[1] - run[0] + 1 > 24:
            continue
        x = x_lo + (run[0] + run[1]) / 2.0
        out.append((cal["ax"] * x + cal["bx"], cal["ay"] * y + cal["by"]))
    return out


def smooth(xy):
    """Running median then a short mean, over columns.

    ⚠ WITHOUT THIS THE PEAK IS BIASED UPWARDS AND THE BIAS IS INVISIBLE. Every
    reported peak is a MAXIMUM over the trace, and the trace's error is not
    symmetric noise around the line -- it is the tracker riding whichever
    scatter marker happens to sit on the line at that column, up to 7 px, one
    marker radius. Taking a maximum over that reads the highest marker, not the
    highest point of the curve: gamma = 1.20 came out at 1.671 against a line
    the plate draws at 1.663. A median over 21 columns is wider than a marker
    and narrower than any real feature of these curves -- the peaks are broad,
    tens of columns across -- so it removes the bias without moving the peak.
    """
    if len(xy) < 5:
        return list(xy)
    xs = [p[0] for p in xy]
    ys = np.asarray([p[1] for p in xy], dtype=np.float64)
    n = len(ys)
    half = min(SMOOTH_MEDIAN_PX // 2, max(1, n // 4))
    med = np.array([np.median(ys[max(0, i - half):i + half + 1])
                    for i in range(n)])
    k = min(SMOOTH_MEAN_PX // 2, max(1, n // 6))
    out = np.array([med[max(0, i - k):i + k + 1].mean() for i in range(n)])
    return list(zip(xs, out.tolist()))


def to_data(curves, cal):
    """Pixel curves -> smoothed [(density, Q)] lists, sorted by density."""
    out = []
    for pts in curves:
        xy = sorted(pts.items())
        out.append(smooth([(cal["ax"] * x + cal["bx"],
                            cal["ay"] * y + cal["by"]) for x, y in xy]))
    return out


def sample(curve, d_query):
    """Q at a density, or None outside the traced range -- never extrapolated."""
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    if d_query < xs[0] or d_query > xs[-1]:
        return None
    return float(np.interp(d_query, xs, ys))


# ---------------------------------------------------------------------------
#  what a re-run must reproduce
# ---------------------------------------------------------------------------
#: Peak Q and the density it occurs at, per gamma, plus Q at density 2.0 where
#: the curves have flattened. Tolerances are loose in DENSITY at the peak -- the
#: maximum is broad and flat, so its position is the least determined thing on
#: the plate, and pinning it tightly would make the audit fail on a one-pixel
#: difference that means nothing.
EXPECTED_PEAK_Q = {0.21: 1.153, 0.37: 1.261, 0.69: 1.475,
                   1.20: 1.670, 1.65: 1.723}

#: Where each curve is drawn TO. The engraver stopped each one where that
#: development ran out of density, and the stopping point is a fact about the
#: plate that the tracker must reproduce: a trace that runs past it has walked
#: into the caption, and one that stops short has lost the line.
EXPECTED_END_D = {0.21: 0.51, 0.37: 0.77, 0.69: 1.49, 1.20: 2.21, 1.65: 2.42}
END_D_TOL = 0.06

Q_TOL = 0.02
CAL_RMS_Q = 0.004
CAL_RMS_D = 0.006

#: The toe envelope must show the collapse to unity, and show it FAST.
TOE_Q_MAX = 1.08          #: the envelope must reach at least this low
TOE_D_AT_Q15 = 0.15       #: ... and be back up to Q = 1.5 by this density


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None,
                    help="project root holding PDF/PROFILES/RETRO; defaults "
                         "to this file's own directory")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    args = ap.parse_args(argv)

    page = page_path(args.root)
    d = load_page(page)
    if d is None:
        print("[SKIP] %s not present" % page)
        return 0

    frame = find_frame(d)
    if frame is None:
        print("[!] plot frame not found")
        return 1
    L, R, T, B = frame

    cal, err = calibrate(d, frame)
    if cal is None:
        print("[!] calibration failed: %s" % err)
        return 1

    if args.probe:
        print("frame  L=%d R=%d T=%d B=%d" % frame)
        print("abscissa  %d ticks, %.6f D/px, rms %.4f, worst %.4f"
              % (cal["n_x"], cal["ax"], cal["rms_x"], cal["worst_x"]))
        print("ordinate  %d ticks, %.6f Q/px, rms %.4f, worst %.4f  "
              "(1.7 recovered at y=%.0f)"
              % (cal["n_y"], cal["ay"], cal["rms_y"], cal["worst_y"],
                 cal["y17"]))

    got, err = trace(d, frame, cal)
    if got is None:
        print("[!] trace failed: %s" % err)
        return 1
    x0, curves = got
    data = to_data(curves, cal)

    if len(data) != len(GAMMAS):
        print("[!] %d curves traced, expected %d" % (len(data), len(GAMMAS)))
        return 1

    # ⚠ ORDER, ASSERTED NOT ASSUMED. The whole tracking scheme rests on the five
    # curves never crossing. If two ever did, a tracker that keeps them ordered
    # would swap them silently and every number after that would be plausible
    # and wrong -- the failure that cost three attempts on the T-101 plate.
    crossed = []
    for i in range(len(data) - 1):
        lo, hi = data[i], data[i + 1]
        lo_x = [p[0] for p in lo]
        for dx, q in hi:
            if dx < lo_x[0] or dx > lo_x[-1]:
                continue
            if q <= np.interp(dx, lo_x, [p[1] for p in lo]):
                crossed.append((GAMMAS[i], GAMMAS[i + 1], round(dx, 2)))
                break

    bad = 0
    print("Mees FIG. 179 -- Callier Q against diffuse density and gamma")
    print("  source: printed p643, 600 ppi page scan, %d x %d px"
          % (d.shape[1], d.shape[0]))
    print("  seed column x=%d, abscissa rms %.4f D, ordinate rms %.4f Q"
          % (x0, cal["rms_x"], cal["rms_y"]))
    print("")
    head = "  gamma  span D        peak Q @ D    " + \
        "".join("%7.2f" % v for v in REPORT_D)
    print(head)
    for g, cur in zip(GAMMAS, data):
        qs = [p[1] for p in cur]
        ds = [p[0] for p in cur]
        pk = int(np.argmax(qs))
        row = "".join(
            ("%7s" % "-") if sample(cur, v) is None else ("%7.3f" % sample(cur, v))
            for v in REPORT_D)
        print("  %-6.2f %4.2f-%4.2f    %5.3f @ %4.2f  %s"
              % (g, ds[0], ds[-1], qs[pk], ds[pk], row))
        want = EXPECTED_PEAK_Q[g]
        if abs(qs[pk] - want) > Q_TOL:
            print("[!] gamma %.2f peak Q %.3f, expected %.3f +/- %.2f"
                  % (g, qs[pk], want, Q_TOL))
            bad += 1
        if not (0.25 <= ds[pk] <= 0.75):
            print("[!] gamma %.2f peak at D %.2f -- the text says the maximum "
                  "is near 0.3" % (g, ds[pk]))
            bad += 1
        if abs(ds[-1] - EXPECTED_END_D[g]) > END_D_TOL:
            print("[!] gamma %.2f traced to D %.2f, the plate draws it to "
                  "%.2f -- short means the line was lost, long means the "
                  "caption was traced" % (g, ds[-1], EXPECTED_END_D[g]))
            bad += 1

    # ⚠ THE TOE IS A SEPARATE MEASUREMENT AND THE ONLY REASON THIS FIGURE IS
    # WORTH KEEPING. It is the one thing `AlgoCallierFactor` cannot express: the
    # renderer holds Q constant, so it applies a condenser's full scatter gain
    # to a density the plate says has almost none.
    toe = trace_toe(d, frame, cal)
    if len(toe) < 200:
        print("[!] toe envelope: only %d rows" % len(toe))
        bad += 1
    else:
        q_lo = min(q for _dd, q in toe)
        at15 = [dd for dd, q in toe if q >= 1.50]
        d15 = min(at15) if at15 else None
        print("")
        print("  toe (all five curves drawn as one stroke below D = %.2f):"
              % TOE_MERGE_D)
        print("    %d rows, lowest Q %.3f at D %.3f"
              % (len(toe), q_lo,
                 min(dd for dd, q in toe if q <= q_lo + 0.002)))
        for q in (1.05, 1.10, 1.20, 1.40, 1.60):
            hit = [dd for dd, qq in toe if abs(qq - q) < 0.005]
            print("      Q %.2f reached at D %s"
                  % (q, "-" if not hit else "%.3f" % float(np.mean(hit))))
        if q_lo > TOE_Q_MAX:
            print("[!] the toe envelope bottoms at Q %.3f, not below %.2f -- "
                  "the collapse to unity is the claim being kept"
                  % (q_lo, TOE_Q_MAX))
            bad += 1
        if d15 is None or d15 > TOE_D_AT_Q15:
            print("[!] the toe reaches Q = 1.5 at D %s, expected below %.2f"
                  % ("never" if d15 is None else "%.3f" % d15, TOE_D_AT_Q15))
            bad += 1

    if crossed:
        print("[!] curves cross, which the tracking scheme assumes they never "
              "do: %s" % crossed)
        bad += 1
    if cal["rms_y"] > CAL_RMS_Q or cal["rms_x"] > CAL_RMS_D:
        print("[!] axis calibration has drifted: rms %.4f Q / %.4f D"
              % (cal["rms_y"], cal["rms_x"]))
        bad += 1

    if args.assert_:
        if bad:
            print("[FAIL] Mees FIG. 179 does not reproduce")
            return 1
        print("[OK] Mees FIG. 179 reproduces: five curves, peaks 1.153-1.723, "
              "each ending where the plate draws it, no crossings, and a toe "
              "collapsing to Q = 1.05 below D = 0.06")
    return 0


if __name__ == "__main__":
    sys.exit(main())
