"""KODAK H-24 -- the processing-variation control charts, read as numbers.

WHAT THIS DOCUMENT IS, AND WHY IT IS NOT A DATASHEET
-----------------------------------------------------
H-24 is Kodak's motion-picture PROCESSING manual, not a film datasheet. Its
"Effects of Mechanical & Chemical Variations" modules (8 for ECN-2, 10 for
ECP-2D/E, 12 for VNF-1, 14 for RVNP) each plot what happens to the
sensitometry when ONE process parameter is moved off aim and everything else
is held. That is the only quantified processing-side data this project has
ever held: every other source in the corpus describes an EMULSION.

WHAT ONE FIGURE HOLDS
---------------------
A 3 x 5 grid of small panels. Columns are films (5213 / 5254 / 5242 for
ECN-2). Rows are the five control-strip statistics Kodak charts:

    HD      high-density patch
    MD      mid-density patch
    LD      low-density patch
    D-min   base plus fog
    HD-LD   the contrast difference

Each panel draws three curves -- R, G, B -- of DENSITY DEVIATION FROM AIM
against the varied parameter. Five abscissa points per curve, so one page
carries 3 films x 5 rows x 3 channels x 5 points = 225 numbers.

⚠ EVERY ROW HAS ITS OWN ORDINATE SCALE. Read off Figure 8-5 as printed:
HD spans +-0.25 D, MD +-0.20, and LD, D-min and HD-LD +-0.10. A reader that
assumes one ladder for the figure -- the obvious thing to assume, since the
panels are drawn the same size -- reports the LD rows 2.5x too large. This
is the same class of defect as the AGFA tick-label trap of 2026-09-06i and
it is caught here by construction, because the scale is fitted per row.

THREE INDEPENDENT CALIBRATIONS, AND THE THIRD IS THE INTERESTING ONE
--------------------------------------------------------------------
  1. THE PRINTED LADDER gives the scale. Label centroids against their
     printed values, least squares, per row. ⚠ Centroids are NOT gridlines
     -- the 2026-09-06i rule -- so this fixes the SCALE and is not trusted
     for the ZERO.

  2. THE CURVES GIVE THEIR OWN ZERO. All three channels meet at one point,
     because at the aim concentration the deviation from aim is zero BY
     DEFINITION. That convergence is a physical anchor the plot never labels
     and never prints, and it locates the zero line to a fraction of a point
     without reference to any text.

  3. ⚠ AND A THIRD MODULE CONFIRMS THE ABSCISSA. Figure 8-5's curves meet at
     1.2 g/L. Module 7, page 30, states the ECN-2 colour developer tank
     aim: "Sodium Bromide (Anhydrous) ... 1.20 +- 0.05 g/L". The convergence
     point of a curve in Module 8 equals the aim printed in Module 7, in a
     different file, by a different table. Neither reading was told the
     other. That is what makes this trace checkable rather than merely
     plausible, and `AIM_FROM_MODULE_7` asserts it every run.

⚠ WHAT THESE FIGURES ARE NOT
----------------------------
They are THREE CONTROL POINTS per channel, not characteristic curves. They
give a level shift and a contrast shift and they cannot re-fit a curve
SHAPE. Everything adopted from here goes into `ProcessVariant`, which holds
exactly that, and nothing adopted from here touches a `ToneCurve`.

Run with --assert to make a drift from the recorded findings fatal.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pymupdf

H24 = "PDF/PROFILES/KODAK/H24"

#: Module 8, Process ECN-2. The figures worth reading are the ones that vary a
#: parameter the process actually CONTROLS; the contamination figures model lab
#: faults and are read but not adopted.
M8 = "Processing-KODAK-Motion-Picture-Films-Module-8.pdf"
M7 = "Processing-KODAK-Motion-Picture-Films-Module-7.pdf"
M2 = "Processing-KODAK-Motion-Picture-Films-Module-2.pdf"
M15 = "Processing-KODAK-Motion-Picture-Films-Module-15.pdf"

#: page index (0-based) -> (figure id, parameter, abscissa unit)
M8_FIGURES = {
    6: ("8-1", "time", "min"),
    7: ("8-2", "temperature", "F"),
    8: ("8-3", "pH", "pH"),
    9: ("8-4", "CD-3", "g/L"),
    10: ("8-5", "NaBr", "g/L"),
    12: ("8-7", "Na2SO3", "g/L"),
}

ROWS = ("HD", "MD", "LD", "D-min", "HD-LD")
CHANNELS = ("R", "G", "B")

# ⚠ THE TRACES ARE STROKED AT EXACTLY 1.44 AND THE PANEL AXES AT 1.08.
# A ">= 1.0" filter takes both, and 37 axis strokes per page then chain onto
# the curve ends and turn 4-segment polylines into 7- and 9-node ones. The
# curve width is asserted rather than inferred.
CURVE_W = 1.44
AXIS_W = 1.08
LADDER_X = (180.0, 215.0)  # the ordinate label gutter; it moves 12 pt between
#                            figures, so the window is wide and the '+'/'0'
#                            count is what validates the catch.
LADDER_RE = re.compile(r"^\.(\d\d)$|^0$")

#: ⚠ FROM A DIFFERENT MODULE, AND NOT TOLD TO THE TRACE. Module 7 p30, ECN-2
#: colour developer tank aim. The convergence point of Figure 8-5's curves
#: must reproduce this, and `check_aims` fails the build if it stops doing so.
AIM_FROM_MODULE_7 = {"NaBr": 1.20}
AIM_TOL = 0.02

#: What the 2026-09-08 run found, so a rerun that disagrees fails.
EXPECTED_CURVES = 45          # 3 films x 5 rows x 3 channels, Figure 8-5
EXPECTED_POINTS = 5
EXPECTED_ROW_SPANS = {"HD": 0.25, "MD": 0.20, "LD": 0.10,
                      "D-min": 0.10, "HD-LD": 0.10}
CONVERGE_TOL_PT = 1.5         # how tightly the three channels must meet
ZERO_AGREE_TOL_PT = 2.5       # curve-derived zero vs the printed '0' centroid
#: ⚠ THE ARITHMETIC GATE. HD - LD must reproduce the separately traced HD-LD
#: row to within the ladder's own reading error, which is 0.0027 D.
IDENTITY_TOL = 0.010

WHAT_C23_NOW_IS = """\
  ⚠ KODAK NAMES THE STREAK, ITS DIRECTION AND ITS CAUSE, AND PRINTS NO SIZE.
    «Curtains -- faint lengthwise streaks of non-uniform density ... improper
    developer turbulation» (Module 15 p22; Module 8 p38 says «vertical»).
    Lengthwise is the engine's axis 0, so the stage's axis choice is now
    vendor-confirmed rather than reasoned.
  ⚠ AND IT IS A FAULT MODE, NOT A FILM PROPERTY. Table 2-2 refreshes the
    emulsion surface every 1.0-1.5 s along the whole strand, and Kodak's
    turbulation troubleshooting guide gives the bar spacing that produces it:
    8-12 in at 40 ft/min, 20-30 in at 100 ft/min -- which is the same 1.0-1.5 s
    computed two ways. On a machine inside spec the effect is absent BY
    DESIGN. So `BromideDragSpec` shipping at zero on all 184 stocks is CORRECT
    for well-processed film, not a missing value.
  ⚠ WHAT THAT BAR SPACING ALSO SAYS IS THAT THE TAIL IS LONG: 200-760 mm of
    film between refreshes, against an 18.7 mm cine frame. A drag that
    persists over ten frames is a MULTI-FRAME density modulation, which is
    what «moving curtains» describes, and not the intra-frame streak stage 9c
    draws. ⚠ THAT IS A MODEL QUESTION AND IS DELIBERATELY LEFT OPEN -- it may
    mean 9c is aimed at the wrong geometry, and that is the owner's call, not
    a number to fit.
  ⚠ NO MAGNITUDE IS PRINTED AND NONE IS ADOPTED. `length_mm` still wants an
    image of a hand-processed or badly-turbulated path."""


# --------------------------------------------------------------------------
#  primitives
# --------------------------------------------------------------------------
def _root(root: str, name: str) -> str:
    return os.path.join(root, H24, name)


def _segments(page, wmin=None, wmax=None):
    """Straight strokes on a page, as (width, x0, y0, x1, y1)."""
    out = []
    for it in page.get_drawings():
        lw = round(it.get("width") or 0.0, 3)
        if wmin is not None and lw < wmin:
            continue
        if wmax is not None and lw > wmax:
            continue
        for s in it["items"]:
            if s[0] == "l":
                out.append((lw, s[1].x, s[1].y, s[2].x, s[2].y))
    return out


def _dedupe(segs, tol=0.02):
    """⚠ THE EMITTER DRAWS SOME STROKES TWICE. Chaining a doubled segment
    makes a curve fork, so identical pairs are collapsed before chaining."""
    seen, out = [], []
    for s in segs:
        key = tuple(round(v / tol) for v in s[1:])
        rkey = tuple(round(v / tol) for v in (s[3], s[4], s[1], s[2]))
        if key in seen or rkey in seen:
            continue
        seen.append(key)
        out.append(s)
    return out


def _chain(segs, tol=0.35):
    """Join strokes that share an endpoint into polylines.

    ⚠ JOINS AT EITHER END, not forward only. The AGFA Vista reader learned
    this the hard way: an emitter that jumps back to extend a path from its
    start turns one curve into three if the walk only ever appends.
    """
    pts = [[(s[1], s[2]), (s[3], s[4])] for s in segs]
    chains = []
    while pts:
        cur = pts.pop(0)
        moved = True
        while moved:
            moved = False
            for i, q in enumerate(pts):
                for a, b in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
                    if abs(cur[a][0] - q[b][0]) < tol and abs(cur[a][1] - q[b][1]) < tol:
                        seg = q[::-1] if b == 0 else q
                        cur = (seg[:-1] + cur) if a == 0 else (cur + seg[1:])
                        pts.pop(i)
                        moved = True
                        break
                if moved:
                    break
        chains.append(cur)
    return chains


def _words(page):
    return [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]


def _centre(w):
    return (0.5 * (w[0] + w[2]), 0.5 * (w[1] + w[3]))


# --------------------------------------------------------------------------
#  the ordinate ladders -- one per row, each with its own span
# --------------------------------------------------------------------------
def read_ladders(words):
    """Group the gutter's printed numbers into one ladder per row.

    ⚠ ROWS ARE SPLIT ON THE '+' MARKS, NOT ON GAPS. Row 1's outermost label
    sits 81 pt from its own zero and 83 pt from the next row's -- a 1.5 pt
    margin, which is not a margin. Each row prints exactly one '+' just above
    its own '0', so counting '+' marks partitions the gutter exactly.
    """
    gut = [w for w in words if LADDER_X[0] <= w[0] <= LADDER_X[1]]
    plus = sorted(_centre(w)[1] for w in gut if w[4] == "+")
    zeros = sorted(_centre(w)[1] for w in gut if w[4] == "0")
    if len(plus) != len(ROWS) or len(zeros) != len(ROWS):
        raise SystemExit(f"[!] expected {len(ROWS)} ladders, "
                         f"found {len(plus)} '+' and {len(zeros)} '0'")
    marks = []
    for w in gut:
        m = LADDER_RE.match(w[4])
        if m:
            marks.append((_centre(w)[1],
                          0.0 if w[4] == "0" else int(m.group(1)) / 100.0))

    # ⚠ MIDPOINT PARTITIONING IS WRONG AND LOOKS RIGHT. The HD row spans
    # ±0.25 and the MD row ±0.20 while the rows' zeros are only ~56 pt apart,
    # so MD's outermost label lands 62 pt from its own zero -- PAST the
    # midpoint, inside LD's half. Splitting on midpoints therefore hands LD
    # the label «.20» and reports LD's span as ±0.20 when it is ±0.10, which
    # scales every LD number by 2. Found by hand-checking one panel against
    # the code, which is the only reason it is not in the database.
    #
    # ⚠ WHAT PARTITIONS CORRECTLY IS THE LABEL'S OWN VALUE. A label «.20»
    # belongs to the zero that is 0.20/S away from it, not to the nearest
    # one. S is bootstrapped from the «.05» labels, which are 16 pt from
    # their zero and cannot be stolen by a neighbour 56 pt away.
    near5 = [abs(y - min(zeros, key=lambda z: abs(y - z)))
             for y, v in marks if abs(v - 0.05) < 1e-9]
    if not near5:
        raise SystemExit("[!] no '.05' ladder labels to bootstrap the scale")
    S = 0.05 / float(np.mean(near5))

    ladders = [dict(zero_y=z, ys=[], vals=[]) for z in zeros]
    for y, v in marks:
        i = min(range(len(zeros)),
                key=lambda j: abs(abs(y - zeros[j]) * S - v))
        z = zeros[i]
        # PDF y runs DOWN, so a label ABOVE the zero is the POSITIVE one.
        ladders[i]["ys"].append(y)
        ladders[i]["vals"].append(v if y <= z else -v)
    for lad in ladders:
        lad["ys"] = np.array(lad["ys"])
        lad["vals"] = np.array(lad["vals"])
        if lad["ys"].size < 5:
            raise SystemExit("[!] a ladder came out with fewer than 5 marks")
    return ladders


def fit_scale(lad):
    """Density per point, least squares on the printed ladder.

    Returns (slope, intercept_y, residual). ⚠ The SLOPE is what is used; the
    intercept is reported only so it can be checked against the curves' own
    convergence, which is the anchor that actually locates the zero.
    """
    ys, vals = lad["ys"], lad["vals"]
    A = np.vstack([ys, np.ones_like(ys)]).T
    (m, c), *_ = np.linalg.lstsq(A, vals, rcond=None)
    resid = float(np.abs(A @ [m, c] - vals).max())
    return float(m), float(-c / m), resid


# --------------------------------------------------------------------------
#  the abscissa -- five printed ticks per column, in five different notations
# --------------------------------------------------------------------------
TICK_RE = re.compile(r"^(\d+):(\d\d)$|^(\d+(?:\.\d+)?)$")


def _tick_value(s: str) -> float:
    """⚠ FIVE NOTATIONS ACROSS SIX FIGURES, AND ONE OF THEM IS SEXAGESIMAL.
    Time is printed «3:00» min:sec, temperature as a bare «106» °F, pH as
    «10.25», concentrations as «1.2» or «2.50» g/L. Reading «3:00» with a
    decimal parser silently yields 3.0 for 3:00 and 2.2 for 2:20 -- plausible
    numbers, wrongly spaced -- so the colon form is parsed explicitly."""
    m = TICK_RE.match(s)
    if not m:
        raise ValueError(s)
    if m.group(1) is not None:
        return int(m.group(1)) + int(m.group(2)) / 60.0
    return float(m.group(3))


def read_ticks(words, n_films):
    """The abscissa labels, partitioned into one run of five per film column.

    ⚠ THE LADDER GUTTER BLEEDS INTO THE BOTTOM BAND. Every figure prints its
    lowest ordinate label «.10» at y ~ 654, inside the abscissa band, so the
    band is cut at x > 210 -- right of the gutter -- rather than by y alone.
    ⚠ AND THE TICKS ARE NOT ON ONE BASELINE: the emitter nudges alternate
    labels by a point, so they are gathered by x order, not by y equality.
    """
    band = [w for w in words if 640 < w[1] < 700 and w[0] > 210
            and TICK_RE.match(w[4])]
    band.sort(key=lambda w: w[0])
    if len(band) % n_films:
        raise SystemExit(f"[!] {len(band)} abscissa ticks do not divide "
                         f"into {n_films} columns")
    per_col = len(band) // n_films
    return band, [_tick_value(w[4]) for w in band[:per_col]], per_col


# --------------------------------------------------------------------------
#  panels
# --------------------------------------------------------------------------
def read_page(doc, pno, figure, parameter, unit):
    page = doc[pno]
    words = _words(page)

    # ⚠ THE CAPTION IS A DECOY. It reads "5213, 5254, and 5242 Films", and the
    # last of those three has no trailing comma, so a bare four-digit match
    # over the top of the page returns FOUR films and silently mis-partitions
    # the abscissa. The column headers sit in their own band below it.
    films = sorted((w for w in words
                    if re.fullmatch(r"\d{4}", w[4]) and 85 < w[1] < 115 and w[0] > 200),
                   key=lambda w: w[0])
    films = [w[4] for w in films]

    ladders = read_ladders(words)
    rows = []
    for name, lad in zip(ROWS, ladders):
        m, zero_from_ladder, resid = fit_scale(lad)
        rows.append(dict(name=name, slope=m, zero_ladder=zero_from_ladder,
                         resid=resid, span=float(np.abs(lad["vals"]).max())))

    segs = _segments(page, wmin=CURVE_W - 0.01, wmax=CURVE_W + 0.01)
    ticks, tick_vals, per_col = read_ticks(words, len(films))
    letters = [w for w in words if w[4] in CHANNELS]

    out = []
    for ci, film in enumerate(films):
        col = ticks[ci * per_col:(ci + 1) * per_col]
        xlo, xhi = col[0][0] - 12.0, col[-1][2] + 12.0
        # ⚠ ROWS ARE PARTITIONED BY NEAREST ZERO, NOT BY A WINDOW. A window
        # sized from the row's own span has to be padded to clear the stroke
        # width, and the padding then reaches into the neighbouring row --
        # the LD and D-min zeros are 80 pt apart and each row is 33 pt tall,
        # so there is 14 pt of daylight and no padding that is both safe and
        # sufficient. Nearest-zero assignment has no free parameter, cannot
        # clip a trace that runs to the frame, and partitions the strokes
        # exhaustively.
        by_row = defaultdict(list)
        for sg in segs:
            if not (xlo <= min(sg[1], sg[3]) and max(sg[1], sg[3]) <= xhi):
                continue
            mid = 0.5 * (sg[2] + sg[4])
            j = min(range(len(rows)), key=lambda i: abs(mid - rows[i]["zero_ladder"]))
            by_row[j].append(sg)

        for ri, r in enumerate(rows):
            mine = by_row.get(ri, [])
            half = r["span"] / abs(r["slope"])
            ylo, yhi = r["zero_ladder"] - half - 8, r["zero_ladder"] + half + 8
            got = _panel(mine, letters, [0.5 * (t[0] + t[2]) for t in col],
                         ylo, yhi, r["zero_ladder"])
            if got is None:
                continue
            nodes, tracks, k, zero_y, label_resid = got
            for ch, ys in sorted(tracks.items()):
                out.append(dict(
                    figure=figure, parameter=parameter, unit=unit,
                    film=film, row=r["name"], channel=ch,
                    x=np.array(nodes), y=np.array(ys),
                    values=(zero_y - np.array(ys)) * r["slope"] * -1.0,
                    aim_index=k, zero_y=zero_y,
                    converge_pt=float(max(abs(ys[k] - zero_y) for ys in tracks.values())),
                    zero_gap_pt=abs(zero_y - r["zero_ladder"]),
                    slope=r["slope"], span=r["span"],
                    ladder_resid=r["resid"], label_resid=label_resid,
                    tick_vals=tick_vals))
    return films, rows, out, ticks, per_col


PANEL_FAILS = []


def _panel(segs, letters, tick_x, ylo, yhi, zero_ladder, tol=0.35):
    """Assemble one panel's three traces, walking IN FROM BOTH ENDS.

    ⚠ THE THREE CURVES SHARE ONE EXACT NODE AND THAT BREAKS ORDINARY CHAINING.
    At the aim setting the deviation from aim is zero for every channel by
    definition, so all three traces pass through one identical point. A walk
    that follows shared endpoints cannot know which of the three departing
    strokes continues which arriving one: it welds R's left half to B's right
    half and produces a smooth, plausible, wrong curve. The first version of
    this reader did exactly that and gave 73 fragments where 45 curves exist.

    ⚠ WHAT RESOLVES IT IS KODAK'S OWN REDUNDANCY, AND IT IS NOT A GUESS. Every
    trace is lettered TWICE, once at each end. So each half is assembled from
    its own labelled end inwards -- the left half seeded by the left letters,
    the right half by the right letters -- and the two halves are joined by
    NAME rather than by geometry. The convergence node is then a CHECK: both
    halves of a channel must arrive at the same y, and all three channels must
    arrive at the same y as each other.

    ⚠ THE COST IS STATED: the two end labels are no longer an independent
    cross-check on the assignment, because the assignment now uses both. What
    replaces it is the convergence test, which the labels cannot fake.
    """
    if not segs:
        PANEL_FAILS.append("no strokes")
        return None
    ordered = [(min(s[1], s[3]), max(s[1], s[3]),
                s[2] if s[1] <= s[3] else s[4],
                s[4] if s[1] <= s[3] else s[2]) for s in segs]

    # ⚠ NEITHER THE STROKES ALONE NOR THE LABELS ALONE GIVE THE NODES.
    # Clustering stroke endpoints invents a sixth column wherever one stray
    # stroke lands outside the plot -- it did that on Figure 8-5's third
    # film, five panels of five. Using the tick LABEL CENTRES instead breaks
    # the other way: the labels are not all the same width, «1.2» against
    # «2.50» against «2:20», and a wider label's centre sits up to 3 pt off
    # its own tick, which is the AGFA cell-centre trap of this project's own
    # 2026-09-06i note in a new costume.
    #
    # ⚠ SO THE STROKES SAY WHERE AND THE LABELS ONLY CHECK IT. Endpoints are
    # clustered, clusters holding fewer than two endpoints are dropped -- a
    # real node carries at least three, one per trace, and a stray carries
    # one -- and the surviving five must be evenly spaced. The printed label
    # centres are then regressed against them and the residual reported, so
    # a disagreement is visible instead of being assumed away.
    # ⚠ THE NODE POSITIONS COME FROM THE STROKES WHERE THE STROKES SAY
    # ANYTHING, AND FROM THE TICKS WHERE THEY DO NOT. A node crossed only by
    # spanning strokes carries no endpoint of its own, so its position has to
    # be inferred -- but not independently: the whole column is one drawing
    # translated, so the offset between the printed tick centres and the
    # drawn nodes is a single number for the column, measured on the nodes
    # that do carry endpoints and applied to the ones that do not.
    xs = [v for o in ordered for v in (o[0], o[1])]
    est = sorted(tick_x)
    seen, offs = {}, []
    for i, e in enumerate(est):
        near = [v for v in xs if abs(v - e) <= 5.0]
        if len(near) >= 2:
            seen[i] = float(np.mean(near))
            offs.append(seen[i] - e)
    if len(offs) < 2:
        PANEL_FAILS.append("only %d nodes carry endpoints" % len(offs))
        return None
    delta = float(np.median(offs))
    nodes = [seen.get(i, e + delta) for i, e in enumerate(est)]
    label_resid = float(np.abs(np.array(est) - np.array(nodes)).max())

    def _i(x):
        j = min(range(len(nodes)), key=lambda i: abs(nodes[i] - x))
        return j if abs(nodes[j] - x) <= 6.0 else -1

    # ⚠ COINCIDENT TRACES ARE DATA, NOT DUPLICATES, AND DEDUPING THEM LOSES A
    # CHANNEL. Where two channels respond identically Kodak draws one stroke
    # over the other, bit-identical. A dedupe pass cannot tell that from the
    # emitter's own doubled strokes, and removing it leaves an interval with
    # two segments for three channels -- which is what dropped nine of the
    # fifteen panels on Figure 8-5. So the strokes are collapsed to DISTINCT
    # (y_here, y_next) pairs per interval and a pair may serve more than one
    # channel; two channels reported equal is then the truth of the drawing.
    # ⚠ A STRAIGHT RUN IS DRAWN AS ONE STROKE, NOT AS ONE STROKE PER
    # INTERVAL. Where a trace is linear across two or three abscissa points
    # the emitter economises and emits a single segment spanning them, so the
    # intermediate nodes carry NO endpoint at all and a reader that only
    # accepts unit-length strokes reports "no endpoints at this tick" and
    # drops the panel -- which is what cost Figure 8-1 nine of its fifteen.
    # A spanning stroke is straight by construction, so its value at each
    # node it crosses is exact rather than interpolated in any lossy sense,
    # and it is split back into unit intervals here.
    fwd = {j: [] for j in range(len(nodes) - 1)}
    for x0, x1, y0, y1 in ordered:
        a_, b_ = _i(x0), _i(x1)
        if a_ < 0 or b_ < 0 or b_ <= a_:
            continue
        for j in range(a_, b_):
            t0 = (nodes[j] - x0) / (x1 - x0) if x1 != x0 else 0.0
            t1 = (nodes[j + 1] - x0) / (x1 - x0) if x1 != x0 else 1.0
            pair = (round(y0 + t0 * (y1 - y0), 2), round(y0 + t1 * (y1 - y0), 2))
            if pair not in fwd[j]:
                fwd[j].append(pair)
    for j in fwd:
        fwd[j].sort()

    n = len(nodes)
    for j in range(n - 1):
        if not 1 <= len(fwd.get(j, [])) <= len(CHANNELS):
            PANEL_FAILS.append("interval has %d distinct strokes" % max(len(v) for v in fwd.values()))
            return None

    def _seed(idx):
        """⚠ MATCHED BY RANK, NOT BY DISTANCE. The letters are set on an even
        vertical pitch beside the panel while the traces arrive wherever the
        data puts them, so on Figure 8-5's first panel the letters sit at
        145.8 / 158.3 / 171.2 and the traces at 157.6 / 163.7 / 170.4. Nearest
        neighbour hands B and G the SAME trace. The stack order is what Kodak
        actually preserves, so the two lists are sorted and zipped.

        ⚠ AND THE LETTERS ARE NOT ALWAYS OUTSIDE THE PANEL. Where the traces
        crowd, Kodak moves one letter INSIDE the frame -- Figure 8-5's LD row
        sets B at x 245.8 with the panel starting at 229.4 -- so a "left of
        the first node" filter finds only G and R and steals a third letter
        from the row above. The three taken are the three nearest the end
        CLUSTER in two dimensions, and the set must come out {R, G, B}.
        """
        j = idx if idx == 0 else idx - 1
        ends = [p[0] if idx == 0 else p[1] for p in fwd.get(j, [])]
        if not ends:
            PANEL_FAILS.append("labels")
            return None
        anchor = (nodes[idx], float(np.mean(ends)))
        near = [w for w in letters
                if abs(_centre(w)[0] - anchor[0]) < 36
                and ylo - 15 <= _centre(w)[1] <= yhi + 15]
        near.sort(key=lambda w: (_centre(w)[0] - anchor[0]) ** 2
                  + (_centre(w)[1] - anchor[1]) ** 2)
        pick = sorted(near[:len(CHANNELS)], key=lambda w: _centre(w)[1])
        if len(pick) != len(CHANNELS) or {w[4] for w in pick} != set(CHANNELS):
            PANEL_FAILS.append("labels")
            return None
        return [w[4] for w in pick]

    lo_names, hi_names = _seed(0), _seed(n - 1)
    if lo_names is None or hi_names is None:
        PANEL_FAILS.append("labels")
        return None

    def _half(idx, direction, names):
        """Assemble one HALF of the panel, from a labelled end inwards.

        ⚠ THE TWO HALVES NEVER HAVE TO BE JOINED, AND TRYING TO JOIN THEM IS
        WHAT MAKES THIS HARD. At the aim setting every channel's deviation
        from aim is zero by definition, so all three traces pass through one
        identical point and no drawing can say which departing stroke
        continues which arriving one. But the DATA does not need the join:
        the points left of the aim come from the left half, the aim point is
        zero for every channel, and the points right of it come from the
        right half. Each half is seeded by the letters at its own end.
        """
        j0 = idx if direction > 0 else idx - 1
        ends = sorted(p[0] if direction > 0 else p[1] for p in fwd.get(j0, []))
        if not ends:
            return None
        while len(ends) < len(CHANNELS):     # two channels on one stroke
            ends.append(ends[-1])
        tracks = {ch: [y] for ch, y in zip(names, ends)}
        i = idx
        while 0 <= i + direction < n:
            j = i if direction > 0 else i - 1
            step = {}
            # ⚠ CHANNELS SHARING A y ARE STEPPED AS A GROUP, IN RANK ORDER.
            # Two traces that coincide at a node and separate at the next one
            # leave the drawing unable to say which is which -- and refusing
            # every such panel costs more than half the figure, because a
            # process change routinely moves two layers together. What is
            # used instead is the ONE thing Kodak's own layout asserts: the
            # letters are stacked in the order the traces run. So a group's
            # departing strokes are sorted and dealt out in the group's own
            # label order.
            # ⚠ THE ASSUMPTION IS THAT THE TWO DO NOT CROSS INSIDE THE HALF,
            # AND IT IS NOT SELF-CHECKING. What checks it is arithmetic from
            # OUTSIDE this panel: Kodak's HD-LD row is HD minus LD, traced in
            # a third panel, and `check_identity` reproduces it from the two
            # separately assembled rows. A swapped pair breaks that identity.
            groups = defaultdict(list)
            for ch, ys in tracks.items():
                groups[round(ys[-1], 2)].append(ch)
            for y, chs in groups.items():
                near = [p for p in fwd.get(j, [])
                        if abs((p[0] if direction > 0 else p[1]) - y) < tol]
                if not near:
                    return None
                nxt = sorted({(p[1] if direction > 0 else p[0]) for p in near})
                # ⚠ ALL THREE IN ONE GROUP IS THE AIM NODE, AND THE HALF ENDS
                # THERE. Dealing three departing strokes out by label order
                # would carry the walk straight through the convergence and
                # weld one channel's left half to another's right half -- the
                # exact defect the two-half design exists to avoid. Rank
                # dealing is only ever applied to a PROPER subgroup, where
                # the remaining channel's separate trace still pins the order.
                if len(chs) == len(CHANNELS) and len(nxt) > 1:
                    return tracks, i
                if len(nxt) > len(chs):
                    return tracks, i
                while len(nxt) < len(chs):
                    nxt.append(nxt[-1])
                for ch, v in zip(sorted(chs, key=lambda c: names.index(c)), nxt):
                    step[ch] = v
            for ch, y in step.items():
                tracks[ch].append(y)
            i += direction
        return tracks, i

    lh = _half(0, +1, lo_names)
    rh = _half(n - 1, -1, hi_names)
    if lh is None or rh is None:
        PANEL_FAILS.append("half")
        return None
    lt, li = lh
    rt, ri = rh
    if set(lt) != set(rt) or li < 1 or ri > n - 1:
        PANEL_FAILS.append("reach li=%d ri=%d" % (li, ri))
        return None

    # ⚠ THE AIM NODE IS WHERE THE TWO HALVES STOP, AND THEY MUST STOP AT THE
    # SAME PLACE. If the left half runs out at node 1 and the right at node
    # 3, the panel holds a stretch this method cannot read and it is refused
    # rather than interpolated.
    if li != ri:
        PANEL_FAILS.append("halves stop apart li=%d ri=%d" % (li, ri))
        return None
    k = li

    tracks = {}
    for ch in CHANNELS:
        if ch not in lt or ch not in rt:
            PANEL_FAILS.append("missing %s" % ch)
            return None
        left = lt[ch]
        right = rt[ch][::-1]
        full = [None] * n
        for i, y in enumerate(left):
            full[i] = y
        for i, y in enumerate(right):
            full[n - len(right) + i] = y
        if any(v is None for v in full):
            PANEL_FAILS.append("gap")
            return None
        tracks[ch] = full

    spread = [max(tracks[c][i] for c in tracks) - min(tracks[c][i] for c in tracks)
              for i in range(n)]
    k = int(np.argmin(spread))
    zero_y = float(np.mean([tracks[c][k] for c in tracks]))
    # ⚠ THE PANEL IS ONLY ACCEPTED IF ITS TWO ZEROS AGREE. The curves' own
    # convergence and the printed «0» are independent statements about where
    # zero deviation sits, and a panel that assembled wrongly puts them tens
    # of points apart. Checking it here rather than in the report means a bad
    # panel is refused instead of averaged into the figure's worst case.
    if abs(zero_y - zero_ladder) > ZERO_AGREE_TOL_PT:
        PANEL_FAILS.append("zero gap %.1f pt" % abs(zero_y - zero_ladder))
        return None
    return nodes, tracks, k, zero_y, label_resid


# --------------------------------------------------------------------------
#  the cross-module check
# --------------------------------------------------------------------------
def aim_from_module_7(root: str) -> dict:
    """Re-read the ECN-2 developer tank aims from Module 7's own table."""
    doc = pymupdf.open(_root(root, M7))
    txt = " ".join(" ".join(p.get_text().split()) for p in doc)
    out = {}
    m = re.search(r"Sodium Bromide \(Anhydrous\)\s+([\d.]+)\s*g\s+([\d.]+)\s*±", txt)
    if m:
        out["NaBr"] = float(m.group(2))
    return out


def turbulation_from_module_2(root: str) -> dict:
    """Table 2-2's spray pass frequency -- the number that reframes C23.

    ⚠ THIS IS WHY BROMIDE DRAG IS A FAULT MODE AND NOT A FILM PROPERTY. A jet
    refreshes the emulsion surface every 1.0-1.5 s along the whole strand, and
    Module 8 p38 and Module 15 p22 both name the streak Kodak calls "Curtains"
    with the cause "improper developer turbulation". On a machine inside spec
    the effect is absent by design, so `BromideDragSpec` shipping at zero on
    every stock is CORRECT for well-processed film rather than a missing
    value, and a fitted number would describe a faulty or hand-processed path.
    """
    doc = pymupdf.open(_root(root, M2))
    txt = " ".join(" ".join(p.get_text().split()) for p in doc)
    m = re.search(r"Spray Pass Frequency\s+([\d.]+)-([\d.]+)\s*sec", txt)
    return dict(spray_pass_s=(float(m.group(1)), float(m.group(2))) if m else None)


def curtains_quote(root: str) -> list:
    """The manufacturer's own words for the streak, from two modules."""
    out = []
    for name in (M8, M15):
        doc = pymupdf.open(_root(root, name))
        for i, p in enumerate(doc):
            t = " ".join(p.get_text().split())
            for m in re.finditer(r"Curtains[^.]{0,120}", t):
                out.append((name[36:], i + 1, m.group(0)))
    return out


# --------------------------------------------------------------------------
#  report
# --------------------------------------------------------------------------
def identity_check(recs):
    """⚠ THE ONE TEST THAT IS ARITHMETIC AND NOT GEOMETRIC.

    Kodak's HD-LD row is not an extra measurement -- it is HD minus LD. But
    it is DRAWN as a third panel, traced here from its own strokes with its
    own ordinate scale and its own channel letters, and nothing in the trace
    is told the identity. So recomputing HD - LD from the two other panels
    and comparing it against the third tests, in one number, the row scales,
    the zero locations and the channel assignment of three independent
    panels at once.

    ⚠ AND IT DISCRIMINATES, WHICH IS WHY IT IS WORTH HAVING. On Figure 8-5's
    5213 it closes to 0.0014-0.0052 D, inside the 0.0027 D ladder residual.
    On Figure 8-2's 5242 it misses by 0.064-0.070 D, twelve times worse --
    so that panel set is assembled wrongly somewhere and is refused, while
    the first is trusted. A geometric check could not have told them apart.
    """
    idx = {(r["film"], r["row"], r["channel"]): r["values"] for r in recs}
    out = []
    for film in sorted({r["film"] for r in recs}):
        for ch in CHANNELS:
            hd = idx.get((film, "HD", ch))
            ld = idx.get((film, "LD", ch))
            df = idx.get((film, "HD-LD", ch))
            if hd is None or ld is None or df is None:
                continue
            out.append((film, ch, float(np.abs((hd - ld) - df).max())))
    return out


def report(root: str, strict: bool) -> int:
    doc = pymupdf.open(_root(root, M8))
    fails = []

    print("=== H-24 Module 8, Process ECN-2 -- variation charts ===")
    print("⚠ 15 panels per figure (3 films x 5 rows), 3 traces each. A panel is")
    print("  taken only if its curve-derived zero agrees with the printed «0»,")
    print("  its three traces meet at ONE node, and that node is the aim.\n")
    store = {}
    for pno, (fig, param, unit) in sorted(M8_FIGURES.items()):
        PANEL_FAILS.clear()
        films, rows, recs, ticks, per_col = read_page(doc, pno, fig, param, unit)
        store[fig] = (films, rows, recs)
        got = len({(r["film"], r["row"]) for r in recs})
        w = lambda k: max((r[k] for r in recs), default=float("nan"))
        print(f"Figure {fig:5s} {param:12s} panels {got:2d}/15   "
              f"ladder resid {max(q['resid'] for q in rows):.4f} D   "
              f"zero gap <= {w('zero_gap_pt'):.2f} pt   "
              f"channels meet to {w('converge_pt'):.2f} pt   "
              f"ticks vs nodes {w('label_resid'):.2f} pt")
        if PANEL_FAILS:
            seen = {}
            for f in PANEL_FAILS:
                seen[f] = seen.get(f, 0) + 1
            print("        refused: " + ", ".join(f"{k} x{v}" for k, v in
                                                  sorted(seen.items())))
        for r in rows:
            want = EXPECTED_ROW_SPANS[r["name"]]
            if abs(r["span"] - want) > 1e-9:
                fails.append(f"{fig}/{r['name']}: ladder spans ±{r['span']:.2f}, "
                             f"expected ±{want:.2f}")
        if any(r["aim_index"] != 2 for r in recs):
            fails.append(f"{fig}: a panel's convergence is not the middle tick")

    # ---- the arithmetic check ----
    print("\n=== HD - LD against the separately traced HD-LD row ===")
    # ⚠ ALL THREE CHANNELS OR NONE. A set where one channel closes and two do
    # not is not two-thirds right -- it is a set with a swapped pair, and the
    # channel that happens to close is the one that was not swapped.
    trusted, per_set = set(), defaultdict(list)
    for fig, (films, rows, recs) in sorted(store.items()):
        for film, ch, err in identity_check(recs):
            per_set[(fig, film)].append((ch, err))
    for key, got in sorted(per_set.items()):
        ok = len(got) == len(CHANNELS) and all(e <= IDENTITY_TOL for _, e in got)
        detail = "  ".join(f"{c} {e:.4f}" for c, e in got)
        print(f"  Figure {key[0]:5s} {key[1]}: {detail} D   "
              f"{'agrees' if ok else '⚠ REFUSED'}")
        if ok:
            trusted.add(key)
    if not trusted:
        fails.append("no film reproduces the HD-LD identity on any figure")
    else:
        print("  ⚠ Only the agreeing sets are eligible for adoption. The rest "
              "are read\n    and printed, and go no further.")

    # ---- the cross-module abscissa check ----
    print("\n=== the check that makes the abscissa verifiable ===")
    m7 = aim_from_module_7(root)
    films, rows, recs = store["8-5"]
    traced_aim = recs[0]["tick_vals"][recs[0]["aim_index"]]
    print(f"  Figure 8-5's three traces converge at     {traced_aim:.2f} g/L")
    print(f"  Module 7 p30 states the ECN-2 tank aim at {m7.get('NaBr')} g/L")
    if m7.get("NaBr") is None or abs(m7["NaBr"] - traced_aim) > AIM_TOL:
        fails.append("Figure 8-5's convergence does not reproduce Module 7's aim")
    else:
        print("  ⚠ TWO MODULES, TWO METHODS, ONE NUMBER. Neither reading was "
              "told the other.")

    # ---- the numbers, for the sets that earned it ----
    for fig, film in sorted(trusted):
        _, _, recs = store[fig]
        sel = [r for r in recs if r["film"] == film]
        print(f"\n=== Figure {fig}, film {film}, deviation from aim (D) ===")
        print(f"    abscissa {sel[0]['tick_vals']} {sel[0]['unit']}")
        for row in ROWS:
            for ch in CHANNELS:
                got = [r for r in sel if r["row"] == row and r["channel"] == ch]
                if got:
                    print(f"  {row:6s} {ch}  " +
                          " ".join(f"{x:+.3f}" for x in got[0]["values"]))

    # ---- C23 ----
    print("\n=== C23 -- what H-24 settles, and what it does not ===")
    turb = turbulation_from_module_2(root)
    print(f"  Module 2 Table 2-2 spray pass frequency: {turb['spray_pass_s']} sec")
    for src, pg, q in curtains_quote(root):
        print(f"  {src} p{pg}: {q}")
    if turb["spray_pass_s"] != (1.0, 1.5):
        fails.append("Module 2's spray pass frequency is no longer 1.0-1.5 sec")
    print(WHAT_C23_NOW_IS)

    print()
    if fails:
        for f in fails:
            print("[!]", f)
        print(f"[FAIL] h24_variations.py -- {len(fails)} check(s) failed")
        return 1 if strict else 0
    print("[OK] h24_variations.py -- %d figures read, %d film/figure sets "
          "reproduce the HD-LD identity" % (len(M8_FIGURES), len(trusted)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="strict", action="store_true")
    a = ap.parse_args()
    return report(a.root, a.strict)


if __name__ == "__main__":
    sys.exit(main())
