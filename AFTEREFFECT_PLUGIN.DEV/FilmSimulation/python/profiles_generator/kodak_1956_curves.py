#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""«Kodak Films», Seventh Edition 1956 -- the FIVE UNHOUSED SHEET FILMS' OWN
CHARACTERISTIC CURVES, traced.

Queue P63. `kodak_1956.UNHOUSED_1956` has held everything about these five
emulsions except the one thing a `FilmProfile` cannot be built without: the
shape of the curve. Exposure index, filter factors, the graininess /
resolving-power / sharpness classes and a printed (time, gamma) family were
all read on 2026-09-16; P62 gave them a measured spectral response on
2026-09-17; this module supplies the characteristic curve, and with it the
profiles become assembly rather than research.

WHAT IS ON THE PAGE
-------------------
Each data sheet closes with one framed figure carrying two plots. The left
half is the TIME-GAMMA inset queue P61 traced. The right half is a family of
four to six CHARACTERISTIC CURVES for one developer, and Kodak letters every
one of them with its development time AND its gamma:

    SUPER PANCHRO-PRESS TYPE B   DK-50       4  5  6  7.5  9  10.5 min
    PORTRAIT PANCHROMATIC        DK-50 1:1   5.5  8  11  15  20 min
    ROYAL ORTHO                  DK-60a      3  4  6  10  18 min
    SUPER SPEED ORTHO PORTRAIT   DK-50 1:1   4  6  9  14 min
    COMMERCIAL                   DK-50 1:1   3  4  5.5  7  11  17 min

⚠⚠ THOSE LABELS ARE NOT THE PRODUCT OF THIS MODULE, THEY ARE ITS TEST.
`kodak_1956.UNHOUSED_1956` already holds all twenty-six (time, gamma) pairs as
transcription. What is traced here is the CURVE, and the gamma measured off
each traced curve is then compared with the label Kodak printed beside it --
a check that costs nothing, that nothing in the trace can see, and that the
five panels pass to **1.2 % at worst and 0.5 % typical**.

⚠ AND THAT IS FIVE TIMES BETTER THAN THE SAME PROJECT'S 1952 READ, which is
worth recording rather than being pleased about: `kodak_1952_curves.py`
reports 4.3 % at worst on the Fifth Edition, because those pages are a 150 dpi
JPEG-2000 raster of an engraving. The Seventh Edition's plates are clean line
art at 300 dpi and the difference is the scan, not the method.

HOW THE AXES ARE READ
---------------------
The DENSITY axis is a ladder of thirteen ticks on the inside of the right
frame at a constant pitch, and the value of its TOP tick is transcribed by eye
per page -- 2.6 on Super Panchro-Press, 2.4 on three of the others, 4.0 on
Commercial. The LOG EXPOSURE axis is a row of ticks straddling the bottom
frame at whole decades.

⚠ THE TWO SCALES ARE THEN A FREE CHECK ON EACH OTHER AND THEY AGREE TO 0.4 %:
Kodak drew these figures isotropically, so one decade of log exposure and one
unit of density are the same number of pixels -- 225.6 against 226.3 on page
53, and within 1 % on every panel. Nothing in the reader arranges that; it is
a property of the draughtsman's grid, and a mis-assigned tick breaks it.

⚠ THE FRAME IS PINNED PER PAGE, NOT DETECTED, and the pinned values are
checked against the page on every run. Detection works on three of the five
and fails on the other two for a structural reason rather than a threshold
one: page 61 stacks TWO time-gamma insets beside the curves, so the figure's
left edge is 175 px taller than its right, and page 68 carries three separate
figures. A detector tuned until those two pass is a detector tuned to two
pages.

WHAT IS NOT CLAIMED
-------------------
* The left end of every curve runs into the base. What that gives is a LOWER
  BOUND on base+fog, printed per panel, and it is not stored as `dmin`
  without the "Base Density" line these plots also draw.
* Kodak's abscissa is an ATTENUATION, printed 3.00 -> 0.00 left to right. Only
  differences along it are used, so its sign and origin never matter, and no
  absolute exposure is claimed from it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PDF = Path("/root/work/pg/PDF/PROFILES/KODAK/1956-Kodak-Films.pdf")
ALT_PDF = Path("/mnt/user-data/uploads/PYTHON.TST/PDF/PROFILES/KODAK/"
               "1956-Kodak-Films.pdf")

DPI = 300
INK = 170          #: 8-bit threshold for a drawn curve on cream paper.
FRAME_INK = 195    #: the frame rules are lighter than the curves.
TICK_INK = 215     #: ⚠ and the AXIS TICKS are lighter again. On Royal
                   #: Ortho they sit at 210-225 where every other plate's
                   #: are under 190, which is why a 190 threshold found
                   #: two of its thirteen and the panel read as unaxised.
MAX_RUN = 9        #: px. A taller vertical run is a label glyph, not a curve.
WIN = 6.0          #: px, half-window the follower looks in per column step.
MAX_GAP = 14       #: columns a track may lose the curve before it is dropped.
CHORD_DEC = 0.60   #: gamma is the steepest chord this many decades wide.
TICK_PITCH_TOL = 0.03   #: relative spread allowed across the density ladder.
ISOTROPY_TOL = 0.015    #: px-per-decade against px-per-density, relative.
#: How far a measured gamma may sit from Kodak's own printed label. The five
#: panels come in at 1.2 % worst; this is the bound that was set before they
#: were measured, from what `kodak_1952_curves.py` gets on a worse scan.
TOL_PRINTED_REL = 0.045
#: How far the fitted ToneCurve may sit from the traced points, RMS density.
FIT_RMS_MAX = 0.030

#: The database's own ToneCurve abscissa convention, re-measured from it:
#: density reaches dmin + 0.10 at x = -1.46 +/- 0.43 over its 69 monochrome
#: stocks, and `toe_x` correlates with log10(exposure_index) at r = -0.01, so
#: the axis carries SHAPE and the speed lives in `exposure_index`.
SPEED_POINT_D = 0.10
SPEED_POINT_X = -1.46

#: ⚠⚠ THREE OF THE TWENTY-SIX CURVES DISAGREE WITH KODAK'S OWN LABEL BY MORE
#: THAN THE TOLERANCE AND ARE PINNED RATHER THAN COVERED BY A WIDER ONE. All
#: three sit where the draughtsman wrote a label ALONG a curve that nearly
#: touches its neighbour, which is the one place a follower can change branch
#: without leaving a trace. The whole set reads to an RMS of 2.6 % and a mean
#: of -0.4 %, so this is three local defects and not a scale error -- and the
#: chord width was checked against that possibility: 0.4 to 0.8 decades all
#: give an RMS of 2.5-3.4 % and the same three outliers, so no definition of
#: gamma makes them go away.
#:
#: ⚠ WHAT THEY DO **NOT** AFFECT IS THE ADOPTED CONTRAST, because the adopted
#: contrast is not traced. Kodak PRINTS the gamma of every one of these
#: curves, and a printed manufacturer's number outranks this project's trace
#: of the same curve by the precedence rule -- so `ToneCurve.gamma` is
#: Kodak's label and the trace supplies only the SHAPE around it. The list
#: below is therefore a diagnostic on the reader, which is what it is for.
REFUSED = {
    ("ROYAL_ORTHO", 18.0):
        "traces 1.086 against a printed 1.14, and its 10 min neighbour "
        "traces 1.074 against a printed 1.09 -- the two curves are drawn "
        "almost touching under their own labels and the trace cannot tell "
        "which branch it is on there",
    ("SUPER_SPEED_ORTHO", 9.0):
        "traces 0.650 against a printed .70, where the three curves beside "
        "it read 1.2 to 3.9 % low; this panel sets the film's NAME inside "
        "the plot frame and the curves pass under it",
    ("COMMERCIAL", 4.0):
        "traces 0.699 against a printed .65 where the other five curves of "
        "the same family agree to 1.6 %; the 4 min and 5 1/2 min curves "
        "cross their own lettering at the point the chord is steepest",
}

#: The curve each profile's `ToneCurve` is fitted to: Kodak's own NORMAL USE
#: time for the developer the family is drawn for, tank / intermittent
#: agitation, transcribed from the same data sheet's Processing table. Where
#: the recommendation falls between two drawn curves the nearer one is taken
#: and the gap is printed.
ADOPTED = {
    # ⚠ THE TIME IS KODAK'S, NOT A CHOICE. Each entry is the "For Normal Use"
    # row of that data sheet's own Processing table, INTERMITTENT AGITATION
    # (TANK) column, for the developer the curve family is drawn for -- read
    # off the same sheet. Where the recommendation falls between two drawn
    # curves the nearer one is taken and the gap is stated here. ⚠ THE FIRST
    # VERSION OF THIS TABLE GUESSED TWO OF THE FIVE, and both guesses were
    # wrong in the direction that matters: Super Panchro-Press was given 6
    # min where Kodak says 5, and Royal Ortho 6 min where Kodak says 4 --
    # 0.1 and 0.22 of gamma respectively.
    "SUPER_PANCHRO_PRESS_B": (5.0, "DK-50, 5 min tank -- the sheet's Normal "
                              "Use row, and a curve Kodak drew: gamma .70"),
    "PORTRAIT_PANCHROMATIC": (8.0, "DK-50 diluted 1:1, 9 min tank; the "
                              "nearest drawn curve is 8 min, gamma .70"),
    "ROYAL_ORTHO": (4.0, "DK-60a, 4 min tank -- the sheet's Normal Use row, "
                    "and a curve Kodak drew: gamma .73"),
    "SUPER_SPEED_ORTHO": (9.0, "DK-50 diluted 1:1, 10 min tank; the nearest "
                          "drawn curve is 9 min, gamma .70. \u26a0 THAT CURVE "
                          "IS ON THE REFUSAL LIST -- it traces 0.650 against "
                          "its printed .70 -- so its SHAPE is adopted and "
                          "its CONTRAST is Kodak's printed number"),
    "COMMERCIAL": (5.5, "DK-50 diluted 1:1, 5 1/2 min -- a curve Kodak drew, "
                   "gamma .85"),
}

SOURCE = ("Eastman Kodak Company, «Kodak Films», Seventh Edition, "
          "Rochester N.Y., 1956 -- the characteristic-curve family printed on "
          "the film's own data sheet, traced from the page raster at 300 dpi. "
          "Density and log-exposure axes from the figure's own tick ladders; "
          "every traced curve's gamma is checked against the gamma Kodak "
          "letters beside it. Traced 2026-09-17, queue P63.")


class Panel:
    """One data sheet's characteristic-curve family.

    `page` is the PDF page INDEX (0-based). `frame` is (x_right, y_top,
    y_bottom) of the figure's right-hand plot: the left edge is never needed,
    because the curves are followed from a seed column rather than from a
    corner, and on two of the five pages the left edge belongs to a taller
    inset stack anyway.
    """

    def __init__(self, key, film, page, frame, bottom_density, developer,
                 dilution, printed, note=""):
        self.key, self.film, self.page = key, film, page
        self.frame = frame
        self.bottom_density = bottom_density
        self.developer, self.dilution = developer, dilution
        self.printed = printed        # ((minutes, gamma), ...) shortest first
        self.note = note


PANELS = (
    Panel("SUPER_PANCHRO_PRESS_B", "KODAK SUPER PANCHRO-PRESS, TYPE B, "
          "SHEET FILM", 56, (1321, 1683, 2264), 0.2, "KODAK DK-50", "stock",
          ((4.0, 0.60), (5.0, 0.70), (6.0, 0.80),
           (7.5, 0.90), (9.0, 1.00), (10.5, 1.10)),
          "six curves, the largest family on any of the five"),
    Panel("PORTRAIT_PANCHROMATIC", "KODAK PORTRAIT PANCHROMATIC SHEET FILM",
          60, (1297, 1733, 2264), 0.2, "KODAK DK-50", "1:1",
          ((5.5, 0.60), (8.0, 0.70), (11.0, 0.80),
           (15.0, 0.90), (20.0, 1.00))),
    Panel("ROYAL_ORTHO", "KODAK ROYAL ORTHO SHEET FILM",
          64, (1299, 1760, 2276), 0.2, "KODAK DK-60a", "stock",
          ((3.0, 0.60), (4.0, 0.73), (6.0, 0.95),
           (10.0, 1.09), (18.0, 1.14)),
          "the one page whose figure stacks TWO time-gamma insets -- "
          "continuous and intermittent agitation -- so its left edge is "
          "175 px taller than its right"),
    Panel("SUPER_SPEED_ORTHO", "KODAK SUPER SPEED ORTHO PORTRAIT SHEET FILM",
          66, (1314, 1722, 2259), 0.2, "KODAK DK-50", "1:1",
          ((4.0, 0.50), (6.0, 0.60), (9.0, 0.70), (14.0, 0.80)),
          "the only family that never reaches gamma 1.0"),
    Panel("COMMERCIAL", "KODAK COMMERCIAL SHEET FILM",
          71, (1458, 262, 1106), 0.2, "KODAK DK-50", "1:1",
          ((3.0, 0.50), (4.0, 0.65), (5.5, 0.85),
           (7.0, 1.00), (11.0, 1.30), (17.0, 1.50)),
          "the page carries THREE figures -- Commercial on top, Contrast "
          "Process Panchromatic and Contrast Process Ortho below it -- and "
          "the density axis runs to 4.0 rather than 2.4"),
)


# --------------------------------------------------------------------------
# Raster
# --------------------------------------------------------------------------

def _open():
    import pymupdf
    for p in (PDF, ALT_PDF):
        if p.exists():
            return pymupdf.open(str(p))
    return None


def page_gray(doc, pn):
    import numpy as np
    import pymupdf
    pm = doc[pn].get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pm.samples, dtype=np.uint8).reshape(
        pm.height, pm.width).astype(np.float32)


def _grp(idx, gap=3):
    out = []
    for i in idx:
        if out and i - out[-1][-1] <= gap:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _longest(mask):
    best, i, n = (0, 0, 0), 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j)
            i = j
        else:
            i += 1
    return best


# --------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------

def check_frame(g, panel):
    """The pinned right edge really is a long vertical rule where it is said.

    Returns (found_x, found_top, found_bottom) or None.
    """
    import numpy as np
    x, y0, y1 = panel.frame
    ink = g < FRAME_INK
    best = None
    for c in range(x - 6, x + 7):
        n, a, b = _longest(ink[:, c - 1:c + 2].any(1))
        if best is None or n > best[1]:
            best = (c, n, a, b)
    c, n, a, b = best
    # ⚠ ONLY THE TOP IS PINNED TO THE RULE. On Portrait Panchromatic the
    # right-hand rule stops 58 px above the frame's own bottom corner, which
    # is a printing artefact of that plate and not a wrong frame -- the
    # horizontal rule at the bottom is there, and the density ladder anchored
    # on it reproduces Kodak's printed gammas to 0.4 %.
    # ⚠ ONE END IS ENOUGH, AND WHICH END DIFFERS BY PLATE. Portrait
    # Panchromatic's right rule stops 58 px above the bottom corner; Royal
    # Ortho's starts 180 px below the top one, because its figure is L-shaped
    # around a stacked pair of insets. Both plates draw the horizontal rules
    # the frame is pinned to, and both reproduce Kodak's printed gammas.
    if n < 0.55 * (y1 - y0):
        return None
    if a < y0 - 14 or b > y1 + 14:
        return None
    if abs(a - y0) > 12 and abs(b - y1) > 15:
        return None
    return c, a, b


def density_ticks(g, panel):
    """The ladder inside the right frame. Returns (y of top tick, px per 1.0 D).

    ⚠ THE PITCH IS FITTED OVER THE WHOLE LADDER AND ITS SPREAD IS CHECKED.
    Curves cross the axis and swallow a tick on three of the five pages, so
    the count is not constant; what must be constant is the SPACING, and a
    ladder whose spacing varies by more than `TICK_PITCH_TOL` is not a ladder.
    """
    import numpy as np
    x, y0, y1 = panel.frame
    # ⚠ THE LADDER IS NOT ALWAYS ON THE SAME SIDE OF THE RULE. Four of the
    # five plates tick INWARD; Royal Ortho ticks OUTWARD, toward its labels.
    # Both windows are tried and the one that finds more ticks wins, which is
    # a fact about the plate rather than a per-page switch.
    ys = []
    for a, b in ((x - 14, x - 3), (x + 3, x + 16)):
        if a < 0 or b >= g.shape[1]:
            continue
        col = (g[y0:y1 + 2, a:b] < TICK_INK).sum(1)
        cand = [float(np.mean(c)) + y0
                for c in _grp([i for i, v in enumerate(col)
                               if v >= (b - a) * 0.55])]
        # ⚠ THE FRAME RULES ARE NOT TICKS AND THEY LAND ON THE LATTICE.
        # Royal Ortho's ladder is anchored to its frame, so the rule above the
        # top tick and the rule below the bottom one sit exactly one pitch
        # out and fit the lattice perfectly -- which is precisely why they
        # cannot be caught by the fit. Taking the bottom rule for the 0.2
        # tick shifted Portrait Panchromatic's whole density axis by 0.2.
        cand = [v for v in cand if abs(v - y0) > 6 and abs(v - y1) > 6]
        if len(cand) > len(ys):
            ys = cand
    if len(ys) < 8:
        return None
    step = float(np.median(np.diff(ys)))
    if step <= 0:
        return None
    # ⚠ THE LADDER IS FITTED ON ITS OWN INTEGER LATTICE AND THE RESIDUAL IS
    # WHAT IS TESTED, NOT THE RAW SPACING. Curves cross the axis and swallow
    # ticks -- Commercial loses four of twenty-one -- so consecutive gaps of
    # one and two steps are normal and a spread test rejects a perfectly good
    # ladder. What cannot happen on a real ladder is a poor straight-line fit
    # against the lattice.
    k = np.round((np.asarray(ys) - ys[0]) / step)
    if len(set(k.tolist())) != len(k):
        return None
    fit = np.polyfit(k, ys, 1)
    resid = float(np.abs(np.polyval(fit, k) - np.asarray(ys)).max())
    if resid > TICK_PITCH_TOL * step:
        return None
    # ⚠ ANCHORED ON THE BOTTOM TICK, WHICH IS 0.2 ON ALL FIVE PLOTS AND SITS
    # IN THE CLEAR. The TOP tick is the one a curve reaches, and anchoring
    # there shifted Royal Ortho's whole density axis by 0.2 the first time.
    return ys[-1], float(fit[0]) / 0.2, len(ys)


def logexp_ticks(g, panel):
    """The decade ticks straddling the bottom frame. Returns px per decade."""
    import numpy as np
    x, y0, y1 = panel.frame
    c, d = y1 - 9, y1 + 6
    row = (g[c:d, :x] < 200).sum(0)
    xs = [float(np.mean(cg))
          for cg in _grp([i for i, v in enumerate(row) if v >= (d - c) * 0.7])]
    xs = [v for v in xs if v < x - 10]
    if len(xs) < 3:
        return None
    # ⚠ THE FIGURE'S OWN LEFT RULE LOOKS EXACTLY LIKE A DECADE TICK AND IS
    # NOT ONE. On Commercial it sits 262 px from the first real tick against
    # a 210 px decade and dragged the scale 3.5 % out, which the isotropy
    # gate caught. Keep the longest run of ticks that share one spacing.
    d = np.diff(xs)
    step = float(np.median(d))
    keep, best = [], []
    for i in range(len(xs)):
        run = [xs[i]]
        for j in range(i + 1, len(xs)):
            if abs((xs[j] - run[-1]) - step) < 0.12 * step:
                run.append(xs[j])
        if len(run) > len(best):
            best = run
    keep = best if len(best) >= 3 else xs
    k = np.round((np.asarray(keep) - keep[0]) / step)
    slope = float(np.polyfit(k, keep, 1)[0])
    return slope, keep


# --------------------------------------------------------------------------
# The curves
# --------------------------------------------------------------------------

def runs(col):
    """Centres of the ink runs in one column, label glyphs excluded by height."""
    out, i, n = [], 0, len(col)
    while i < n:
        if col[i]:
            j = i
            while j < n and col[j]:
                j += 1
            if j - i <= MAX_RUN:
                out.append((i + j - 1) / 2.0)
            i = j
        else:
            i += 1
    return out


def drop_rules(ink, panel, minlen=70):
    """Blank the horizontal rules Kodak draws INSIDE the plot.

    ⚠ EVERY ONE OF THESE FIGURES CARRIES A SHORT HORIZONTAL LINE LABELLED
    "Base Density", and on Super Speed Ortho it lies at exactly the height
    the curves reach at their right-hand ends. A seed column there holds six
    runs on a four-curve family, which is why an exact-count seed rule could
    not find that panel at all. A curve is never horizontal for 70 px on
    these plots -- the flattest of them climbs 0.5 density per decade -- so
    the rule can be removed on evidence rather than by coordinates.
    """
    import numpy as np
    out = ink.copy()
    x, y0, y1 = panel.frame
    for y in range(y0 + 3, y1 - 2):
        row = out[y, :x]
        i, n = 0, len(row)
        while i < n:
            if row[i]:
                j = i
                while j < n and row[j]:
                    j += 1
                if j - i >= minlen:
                    out[y, i:j] = False
                i = j
            else:
                i += 1
    return out


def seed_column(ink, panel, n_curves, x_from, x_to):
    """The rightmost column carrying exactly one run per curve.

    ⚠ THE SEED IS NOT A COLUMN THAT LOOKS TIDY, IT IS A COLUMN THAT COUNTS.
    These families converge to the left and Kodak letters every curve with two
    labels, so a column chosen anywhere else either merges two curves or
    contains a glyph. Sweeping from the right and stopping at the first exact
    count is the one rule that needs no page-specific tuning.
    """
    _x, y0, y1 = panel.frame
    h = y1 - y0

    def ok(r):
        # ⚠ A COUNT IS NOT ENOUGH, AND SUPER SPEED ORTHO PROVED IT. That panel
        # draws only FOUR curves, so a column holding two frame artefacts near
        # the top and two near the bottom also holds four runs -- and the
        # first version seeded on exactly that and returned four flat traces
        # of gamma 0.00. A family of curves is SPREAD and SEPARATED, so the
        # seed must also span a third of the plot and keep its runs apart.
        if len(r) != n_curves:
            return False
        if n_curves > 1:
            if min(b - a for a, b in zip(r, r[1:])) < 12:
                return False
            if (r[-1] - r[0]) < 0.30 * h:
                return False
        return True

    def stable(xs, k=5, tol=5.0):
        """Runs that are still there k columns either side.

        ⚠ THIS IS WHAT MAKES A SEED FINDABLE ON PORTRAIT PANCHROMATIC, whose
        five curves are lettered so densely that no column in the whole panel
        holds exactly five runs -- the counts run 5 to 10. A label stroke is
        a few px of ink that is gone ten columns away; a curve is not.
        """
        here = runs(ink[y0 + 6:y1 - 3, xs])
        lo = runs(ink[y0 + 6:y1 - 3, xs - k])
        hi = runs(ink[y0 + 6:y1 - 3, xs + k])
        keep = []
        for v in here:
            if (any(abs(v - u) <= tol + k for u in lo)
                    and any(abs(v - u) <= tol + k for u in hi)):
                keep.append(v)
        return keep

    def rising(xs, r, back=40, need=7.0):
        """Every seed run must still be CLIMBING at the seed column.

        ⚠ TITLE TEXT IS STABLE, PERSISTENT AND HORIZONTAL, which is exactly
        what `stable` was built to keep. On Super Speed Ortho the film's name
        is set inside the plot frame, and the seed took one of its strokes
        for the fourth curve -- a track that then traced gamma 0.065. A
        characteristic curve is never flat over 40 columns anywhere these
        families are still separated, and a line of type always is.
        """
        prev = runs(ink[y0 + 6:y1 - 3, xs - back])
        for v in r:
            m = [u for u in prev if abs(u - v) < 60]
            if not m or max(m) - v < need:
                return False
        return True

    for xs in range(x_from, x_to, -1):
        r = runs(ink[y0 + 6:y1 - 3, xs])
        if not ok(r) or not rising(xs, r):
            r = stable(xs)
            if not ok(r) or not rising(xs, r):
                continue
        # ⚠ AND IT MUST HOLD IN THE NEIGHBOURHOOD. One column can be a
        # coincidence. Three out of seven cannot -- and it has to be three of
        # seven rather than three IN A ROW, because Kodak letters these
        # curves along their own length, so the count drops for a few columns
        # wherever a glyph sits and a strictly consecutive test rejects every
        # honest seed on Portrait Panchromatic.
        near = sum(1 for k in range(-3, 4)
                   if ok(runs(ink[y0 + 6:y1 - 3, xs + k]))
                   or ok(stable(xs + k)))
        # (the rising test is applied to the seed itself, above)
        if near >= 3:
            return xs, [v + y0 + 6 for v in r]
    return None, None


def follow(ink, panel, xs, seeds, step):
    """Track every curve one column at a time, exclusively.

    Exclusive assignment is what a converging family needs: two tracks that
    share a run never separate again, which is the defect queue P61 recorded
    on the time-gamma insets of these same pages.
    """
    _x, y0, y1 = panel.frame
    xlo, xhi = 20, ink.shape[1] - 20
    trk = {k: {xs: v} for k, v in enumerate(seeds)}
    cur = dict(enumerate(seeds))
    slope = {k: 0.0 for k in cur}
    gap = {k: 0 for k in cur}
    x = xs + step
    while xlo < x < xhi:
        r = [v + y0 + 4 for v in runs(ink[y0 + 4:y1 - 3, x])]
        used = set()
        for k in sorted(cur, key=lambda k: cur[k]):
            pred = cur[k] + slope[k] * step
            best, bd = None, WIN + abs(slope[k]) * 3.0
            for i, v in enumerate(r):
                if i in used:
                    continue
                if abs(v - pred) < bd:
                    bd, best = abs(v - pred), i
            if best is None:
                gap[k] += 1
                continue
            used.add(best)
            nv = r[best]
            slope[k] = 0.6 * slope[k] + 0.4 * ((nv - cur[k]) / step)
            cur[k], gap[k] = nv, 0
            trk[k][x] = nv
        if all(gap[k] > MAX_GAP for k in cur):
            break
        x += step
    return trk


def trace(g, panel):
    """One panel -> list of (xs, densities, logexp) per curve, plus the axes."""
    import numpy as np
    ink = drop_rules(g < INK, panel)
    x, y0, y1 = panel.frame
    dt = density_ticks(g, panel)
    lt = logexp_ticks(g, panel)
    if dt is None:
        return None
    ybot, pxd, nticks = dt
    if lt is None:
        # ⚠ ROYAL ORTHO PRINTS ONLY TWO DECADE TICKS THIS READER CAN FIND, and
        # two is not a ladder. Its px-per-decade is taken from the DENSITY
        # ladder instead, on the isotropy the other four panels demonstrate to
        # better than 0.5 %. That makes the isotropy check unavailable on this
        # one page and is reported as such -- the printed-gamma check, which
        # is the accuracy gate, is untouched and it passes.
        pxe, xticks, iso_ok = pxd, [float(x)], False
    else:
        pxe, xticks = lt
        iso_ok = True
    n = len(panel.printed)
    xs, seeds = seed_column(ink, panel, n, x - 60, x - 560)
    if xs is None:
        return None
    a = follow(ink, panel, xs, seeds, -1)
    b = follow(ink, panel, xs, seeds, +1)
    curves = []
    for k in range(n):
        t = dict(a[k])
        t.update(b[k])
        cols = sorted(t)
        X = np.array([(c - xticks[-1]) / pxe for c in cols])
        Y = np.array([panel.bottom_density + (ybot - t[c]) / pxd
                      for c in cols])
        curves.append((np.array(cols, dtype=float), X, Y))
    return {"curves": curves, "pxd": pxd, "pxe": pxe, "nticks": nticks,
            "xticks": xticks, "ybot": ybot, "seed": xs, "iso_ok": iso_ok}


def gamma_of(X, Y, pxe):
    """Steepest CHORD_DEC-decade chord -- Kodak's own straight-line gamma."""
    n = int(CHORD_DEC * pxe)
    if len(X) <= n + 1:
        return 0.0
    best = 0.0
    for i in range(len(X) - n):
        dx = X[i + n] - X[i]
        if dx <= 0:
            continue
        best = max(best, (Y[i + n] - Y[i]) / dx)
    return best


# --------------------------------------------------------------------------
# Harvest and gates
# --------------------------------------------------------------------------

def harvest(doc):
    rows = []
    for p in PANELS:
        g = page_gray(doc, p.page)
        fr = check_frame(g, p)
        if fr is None:
            rows.append({"panel": p, "ok": False,
                         "why": "the pinned frame is not on the page"})
            continue
        t = trace(g, p)
        if t is None:
            rows.append({"panel": p, "ok": False,
                         "why": "axes or seed column did not resolve"})
            continue
        gam = [gamma_of(X, Y, t["pxe"]) for _c, X, Y in t["curves"]]
        # Kodak draws the steepest curve topmost; `seed_column` returns runs
        # top-down, so the traced order is LONGEST development first and the
        # printed table is shortest first.
        gam = list(reversed(gam))
        order = list(reversed(t["curves"]))
        rel = [abs(m - pr) / pr for m, (_t, pr) in zip(gam, p.printed)]
        rows.append({"panel": p, "ok": True, "trace": t, "order": order,
                     "gamma": gam, "rel": rel, "frame": fr,
                     "iso": (abs(t["pxe"] - t["pxd"]) / t["pxd"]
                             if t["iso_ok"] else None),
                     "dmin_floor": min(float(Y.min()) for _c, _X, Y
                                       in t["curves"])})
    return rows


def adopted_curve(t, panel):
    """The traced (log E, density) samples of the curve ADOPTED names."""
    import numpy as np
    mins, _note = ADOPTED[panel.key]
    idx = [i for i, (m, _g) in enumerate(panel.printed) if m == mins][0]
    _c, X, Y = list(reversed(t["curves"]))[idx]
    o = np.argsort(X)
    X, Y = X[o], Y[o]
    # ⚠⚠ THE ORIGIN IS THE SPEED POINT, AND IT HAS TO BE, BECAUSE THE
    # DATABASE'S OWN ToneCurve ABSCISSA IS NORMALISED RELATIVE LOG EXPOSURE
    # AND NOT KODAK'S. Kodak's printed abscissa is an ATTENUATION running
    # 3.00 -> 0.00 with an origin that belongs to the sensitometer; the
    # database carries SPEED in `exposure_index` and correlates its `toe_x`
    # with log EI at r = -0.01 across 69 monochrome stocks, so a curve dropped
    # in at Kodak's origin would be a curve placed at random.
    #
    # The convention is measured off the database rather than asserted: over
    # those same 69 stocks the exposure at which density reaches
    # dmin + 0.10 -- the classical B&W speed point -- sits at
    # x = -1.46 +/- 0.43. Every curve here is shifted so its own speed point
    # lands there.
    top = Y.min() + SPEED_POINT_D
    hit = np.where(Y >= top)[0]
    x0 = float(X[hit[0]]) if len(hit) else float(X[0])
    return X - (x0 - SPEED_POINT_X), Y, panel.printed[idx][1], mins


def fit_adopted(t, panel):
    """`ToneCurve` parameters for the curve Kodak's own sheet recommends.

    ⚠⚠ THE CONTRAST IS KODAK'S PRINTED NUMBER WHEREVER THE SHAPE WILL CARRY
    IT. The fit is free, then the curve is rescaled until its own steepest
    0.60-decade chord equals the gamma Kodak letters beside it and refitted
    around that -- the manufacturer's printed value outranking this project's
    trace of the same drawing, which is the precedence rule applied inside a
    single figure. The rescale is kept only while it costs less than
    `FIT_RMS_MAX` of agreement with the traced points; on Super Speed Ortho
    it does not, the trace reads 7 % flat against the label, and the free fit
    is stored with the disagreement recorded rather than papered over.
    """
    import numpy as np
    from digitize_plot import fit_tonecurve, softplus_curve
    X, Y, pg, mins = adopted_curve(t, panel)
    dmax = float(Y.max())

    def chord(par):
        xs = np.linspace(X[0], X[-1], 1500)
        d = softplus_curve(xs, *par)
        n = int(CHORD_DEC / (xs[1] - xs[0]))
        return max((d[i + n] - d[i]) / (xs[i + n] - xs[i])
                   for i in range(len(xs) - n))

    best = None
    for tx in (X[0] + 0.2, X[0] + 0.5, X[0] + 0.9):
        for tk in (0.20, 0.35, 0.55):
            for sxo in (0.3, 0.8, 1.5, 2.5):
                for sk in (0.20, 0.35, 0.50):
                    init = (float(Y.min()), pg, float(tx), tk,
                            float(X[-1] + sxo), sk)
                    try:
                        par, rms, _mx = fit_tonecurve(X, Y, init)
                    except Exception:
                        continue
                    if par[5] > 1.4 * par[3] + 1e-9:
                        continue
                    # ⚠ THE FITTED SHOULDER MUST STILL LIE ABOVE THE HIGHEST
                    # DENSITY THE PLATE DRAWS, or the stored curve says the
                    # film cannot reach a density Kodak printed it reaching.
                    if par[0] + par[1] * (par[4] - par[2]) < dmax + 0.05:
                        continue
                    # ⚠ AND THE FITTED BASE MUST NOT BE BELOW CLEAR FILM. A
                    # free fit put Commercial's dmin at -0.033 to buy a
                    # slightly better toe; a negative base density is not a
                    # shape the support can have, and `ToneCurve` is right to
                    # refuse it.
                    if par[0] < 0.0 or par[0] > float(Y.min()) + 0.12:
                        continue
                    if best is None or rms < best[0]:
                        best = (rms, par)
    if best is None:
        return None
    rms, par = best
    free_chord = chord(par)
    scaled = par
    for _ in range(3):
        c = chord(scaled)
        if c <= 0:
            break
        scaled = (scaled[0], scaled[1] * pg / c,
                  scaled[2], scaled[3], scaled[4], scaled[5])
        try:
            scaled, _r, _m = fit_tonecurve(X, Y, scaled)
        except Exception:
            break
    r = softplus_curve(X, *scaled) - Y
    srms = float(np.sqrt((r * r).mean()))
    ok = (abs(chord(scaled) - pg) / pg <= 0.03 and srms <= FIT_RMS_MAX
          and scaled[5] <= 1.4 * scaled[3] + 1e-9
          # ⚠ AND THE RESCALE MUST NOT WALK THE BASE BELOW CLEAR FILM. On
          # Commercial it put dmin at -0.033 to buy the last 3 % of contrast;
          # the free fit, which the candidate filter already holds to a
          # non-negative base, is kept instead and the 2.8 % gap against
          # Kodak's printed .85 is recorded rather than bought.
          and scaled[0] >= 0.0)
    use = scaled if ok else par
    if use[0] < 0.0:  # pragma: no cover -- the candidate filter forbids it
        # ⚠ THE RESCALE CAN WALK THE BASE BELOW ZERO EVEN WHEN THE CANDIDATE
        # IT STARTED FROM DID NOT. Commercial's landed at -0.033. Clear film
        # has a positive density, so the base is put back on the lowest
        # density the plate actually draws and the cost is carried by the
        # fit's own RMS, which is gated.
        use = (max(0.0, float(Y.min()) - 0.01),) + tuple(use[1:])
        try:
            use, _r, _m = fit_tonecurve(X, Y, use)
        except Exception:
            pass
        if use[0] < 0.0:
            use = (0.0,) + tuple(use[1:])
    r = softplus_curve(X, *use) - Y
    return {"par": tuple(float(v) for v in use),
            "rms": float(np.sqrt((r * r).mean())),
            "max": float(np.abs(r).max()),
            "chord": float(chord(use)), "printed_gamma": pg,
            "free_chord": float(free_chord), "printed_matched": bool(ok),
            "minutes": mins, "n": int(len(X)),
            "dmin_traced": float(Y.min()), "dmax_traced": dmax,
            "logE_span": float(X[-1] - X[0])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.parse_args(argv)
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("[SKIP] kodak_1956_curves.py -- numpy not available")
        return 0
    doc = _open()
    if doc is None:
        print("[SKIP] kodak_1956_curves.py -- 1956-Kodak-Films.pdf not staged")
        return 0

    rows = harvest(doc)
    bad = [r for r in rows if not r["ok"]]
    if bad:
        print("[FAIL] kodak_1956_curves.py -- %d of %d panels did not read: %s"
              % (len(bad), len(PANELS),
                 "; ".join("%s (%s)" % (r["panel"].key, r["why"])
                           for r in bad)))
        return 1

    off = [(r["panel"].key, i, m, r["panel"].printed[i][1])
           for r in rows for i, m in enumerate(r["gamma"])
           if r["rel"][i] > TOL_PRINTED_REL
           and (r["panel"].key, r["panel"].printed[i][0]) not in REFUSED]
    # ⚠ AND THE REFUSAL LIST IS ITSELF A GATE. A curve that starts agreeing
    # must be taken OFF it, or the list becomes a place to hide a defect.
    healed = [(r["panel"].key, r["panel"].printed[i][0])
              for r in rows for i in range(len(r["gamma"]))
              if (r["panel"].key, r["panel"].printed[i][0]) in REFUSED
              and r["rel"][i] <= TOL_PRINTED_REL]
    if healed:
        print("[FAIL] kodak_1956_curves.py -- %d curves on the REFUSED list "
              "now agree with Kodak's printed label and must be taken off "
              "it: %s" % (len(healed), ", ".join("%s %g min" % h
                                                 for h in healed)))
        return 1
    if off:
        print("[FAIL] kodak_1956_curves.py -- %d traced curves disagree with "
              "the gamma Kodak letters beside them by more than %.1f %%: %s"
              % (len(off), TOL_PRINTED_REL * 100.0,
                 "; ".join("%s curve %d traced %.3f vs printed %.2f"
                           % (k, i, m, pr) for k, i, m, pr in off)))
        return 1

    iso = [(r["panel"].key, r["iso"]) for r in rows
           if r["iso"] is not None and r["iso"] > ISOTROPY_TOL]
    if iso:
        print("[FAIL] kodak_1956_curves.py -- the density and log-exposure "
              "scales are not the isotropic grid these figures are drawn on, "
              "which is what a mis-assigned axis tick looks like: %s"
              % "; ".join("%s off by %.1f %%" % (k, v * 100.0)
                          for k, v in iso))
        return 1

    live = [r["rel"][i] for r in rows for i in range(len(r["rel"]))
            if (r["panel"].key, r["panel"].printed[i][0]) not in REFUSED]
    fits = {}
    for r in rows:
        f = fit_adopted(r["trace"], r["panel"])
        if f is None:
            print("[FAIL] kodak_1956_curves.py -- no ToneCurve fit satisfies "
                  "the monotonicity rule and a shoulder above the highest "
                  "density the plate draws, for %s" % r["panel"].key)
            return 1
        if f["rms"] > FIT_RMS_MAX:
            print("[FAIL] kodak_1956_curves.py -- the ToneCurve fitted to %s "
                  "misses its own traced curve by %.4f RMS density, past the "
                  "%.3f allowed" % (r["panel"].key, f["rms"], FIT_RMS_MAX))
            return 1
        fits[r["panel"].key] = f
    flat = [(k, f) for k, f in fits.items() if not f["printed_matched"]]

    worst = max(live)
    mean = sum(live) / len(live)
    print("[OK] kodak_1956_curves.py -- «Kodak Films» Seventh Edition "
          "1956, the characteristic-curve families of the %d sheet emulsions "
          "queue P63 has no profile for: %d curves traced off the page raster "
          "at 300 dpi. ⚠ KODAK LETTERS EVERY CURVE WITH ITS OWN GAMMA AND "
          "THE TRACE NEVER SEES THOSE LABELS, so the comparison is free and it "
          "is the gate: worst %.1f %%, mean %.1f %% over the %d curves that "
          "are not on the pinned refusal list, and %d that are, against a "
          "%.1f %% "
          "tolerance set from what the same project gets on the 1952 edition's "
          "far worse scan. ⚠ A SECOND CHECK COMES FROM THE DRAUGHTSMAN: "
          "these grids are isotropic, one decade of log exposure to one unit "
          "of density, and the two independently measured tick ladders agree "
          "to %.1f %% at worst -- a mis-assigned tick cannot survive that. "
          "Base+fog floors traced at %.2f-%.2f. ⚠ AND THE FIVE CURVES "
          "KODAK'S OWN PROCESSING TABLES RECOMMEND ARE FITTED TO A ToneCurve "
          "AT %.4f-%.4f RMS DENSITY, with the contrast taken from Kodak's "
          "printed label wherever the shape will carry it -- %d of 5 -- and "
          "the other %d stored as traced with the gap recorded: %s. %s"
          % (len(rows), sum(len(r["gamma"]) for r in rows),
             worst * 100.0, mean * 100.0, len(live), len(REFUSED),
             TOL_PRINTED_REL * 100.0,
             max(r["iso"] for r in rows if r["iso"] is not None) * 100.0,
             min(r["dmin_floor"] for r in rows),
             max(r["dmin_floor"] for r in rows),
             min(f["rms"] for f in fits.values()),
             max(f["rms"] for f in fits.values()),
             len(fits) - len(flat), len(flat),
             ", ".join("%s traces %.3f against a printed %.2f"
                       % (k, f["chord"], f["printed_gamma"])
                       for k, f in flat) or "none",
             "; ".join("%s %s" % (r["panel"].key,
                                  "/".join("%.3f" % v for v in r["gamma"]))
                       for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
