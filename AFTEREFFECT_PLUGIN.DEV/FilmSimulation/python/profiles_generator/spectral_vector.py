"""Spectral SENSITIVITY from Kodak brochure VECTOR paths (queue item C10).

Reads the "SPECTRAL SENSITIVITY CURVES" panel off a Kodak H-1 motion-picture
brochure and returns the three layer curves on this database's spectral grid
(380-680 nm, 10 nm, peak-normalised to 0.0, floor -4.0).

WHY THIS SCRIPT EXISTS AT ALL
-----------------------------
Every spectral set in the database until now came from either a 2026-08-02
RASTER batch or `agfa_vista.py`'s dash-legend reader. Nothing could read a
VECTOR sensitivity panel, so the panel on the one sheet whose characteristic
curves, granularity and MTF are all already traced -- H-1-5201 -- sat unread.

THE ASSIGNMENT PROBLEM, AND WHY INK SOLVES IT
---------------------------------------------
Three curves in one frame, nothing printed inside it to say which layer is
which. The sheet's legend names them ("Sensitivity of the yellow / magenta /
cyan dye forming layer") but the legend is text beside the plot, not a label on
a curve. What connects the two is Kodak's ink convention, and it is physical
rather than decorative: EACH TRACE IS DRAWN IN THE COLOUR OF LIGHT IT RESPONDS
TO. The blue-sensitive (yellow-forming) layer is drawn in BLUE ink, the
green-sensitive (magenta-forming) layer in GREEN, and the red-sensitive
(cyan-forming) layer in RED -- which is not one of the four process inks, so
Kodak makes it by overprinting YELLOW UNDER MAGENTA. Two coincident paths, one
curve. `dye_density.extract_inked` already collapses that pair, and this script
reuses it rather than re-deriving the palette.

THE ASSIGNMENT IS THEN CHECKED THREE WAYS, none of which is the ink:
  1. Legend swatches. The 1 pt lines left of the legend text carry the same
     inks: green sits on "magenta dye forming layer", amber on "cyan dye
     forming layer". Kodak's own words, machine-read.
  2. Absorption bands. The three traces must peak in ascending wavelength order
     and each inside its own band. On 5201 they peak at 470 / 540 / 645 nm.
  3. The sibling sheets. 5218 and 5217 are the same product family and their
     spectral sets were adopted from a different method (the raster batch); the
     shapes have to agree to within the reading error of a printed plot.

⚠ WHAT THIS SCRIPT DOES NOT KNOW: THE DENSITY CRITERION
-------------------------------------------------------
The panel's footnote reads, in full: "Sensitivity = reciprocal of exposure
(erg/cm2) required to produce specified density". IT DOES NOT SAY WHAT DENSITY.
The three sets already in the database (5218, 5217, 5219) carry
`criterion="log_reciprocal_erg_cm2_D0.2_above_dmin"`, and checking the sources:
5218 and 5217 print the same unspecified wording, and 5219's footnote is not in
its text layer at all. So the "D0.2 above dmin" half of that string is not
printed on any of the three. Owner decision 2026-08-25: store 5201 as the sheet
prints it, leave the other three alone, and record the discrepancy (method rule
4 -- a conflict is recorded, not averaged, and not quietly propagated).

THREE FIXES ON 2026-08-25 THAT THE 7239 SHEET FORCED, AND WHAT EACH ONE TAUGHT
------------------------------------------------------------------------------
7239 had been on the "unreachable" list for a fortnight. It turned out to be
blocked by three independent defects, none of them in the source:

  1. **The caption finder could not see short words.** `rot_labels` decides a
     word is rotated from its aspect ratio, `(y1-y0) > 1.6*(x1-x0)`. True of
     "SENSITIVITY" rotated, FALSE of "LOG" -- three characters is not obviously
     taller than wide. The pair never met. `rot_lines` replaces the heuristic
     with PyMuPDF's per-line WRITING DIRECTION, which is not a guess at all.
  2. **The frame picker took the nearest frame and stopped.** On this page two
     rects qualify and the tick labels sit BETWEEN them, so the nearer frame is
     uncalibratable. Candidates are now tried in order until one calibrates.
  3. **The y axis has a minus sign that is not in the text layer.** See
     `_sign_y_ticks`. This one was the dangerous defect: the old `setdefault`
     silently dropped the duplicate labels and happened to keep the right
     branch, so the sheet would have read CORRECTLY -- by luck -- and an
     identical sheet with the opposite emission order would have read mirrored
     with every check still passing.

⚠ AND THE SHEET IS NOT INKED. Its three traces are all BLACK, so the convention
this module was built on says nothing about them. `extract_mono` reads it, and
the assignment then rests on the band test, the ordering test and Kodak's own
in-frame captions -- one fewer independent check than an inked panel gets, which
is stated in the adopted profile rather than left to be inferred.

WHAT THE 2026-08-26 CROSS-CHECK SWEEP FOUND, INCLUDING THAT IT FOUND NO DEFECT
-----------------------------------------------------------------------------
Eleven of the fifteen panels the writing-direction fix made reachable were
re-derived from their vector paths and compared against the sets adopted from
the 2026-08-02 RASTER batch. ⚠ **NONE OF THEM ARE NEW DATA** -- every one of the
eleven stocks already carried a spectral set, which is worth saying because this
task was scoped on the assumption that they were.

⚠ AND THE APPARENT DISAGREEMENTS WERE ARTEFACTS OF THE COMPARISON, NOT DEFECTS
IN THE ADOPTED DATA. Two kinds turned up and both were chased to the ground:

  * LEVEL differences (5245 blue "rms 0.340 decades") come from comparing a
    TRUNCATED trace against a complete one after per-layer peak normalisation.
    Normalising two curves that stop at different wavelengths to their own
    maxima makes them disagree by construction.
  * PEAK differences (5246 blue traced 430 nm against a stored 470) are ARGMAX
    NOISE ON A PLATEAU. Measured plateau width, samples within 0.05 decades of
    the maximum: 5274 0 nm, 5245 10 nm, 5205 40 nm, 5246 40 nm. On 5246 BOTH
    readings agree the plateau runs 430-470; they differ only in which sample
    argmax lands on.

The traces themselves were verified the way this project verifies traces --
by rendering the points back onto the page. On both 5245 and 5246 they lie on
the printed curves. So the readers are sound, the adopted data is sound, and
nothing was re-adopted: a wash is not a reason to churn adopted data, which is
the same rule that left 5217 alone on 2026-08-25.

⚠ THE SWEEP DID PRODUCE TWO REAL RESULTS, neither of them a new curve:
  1. The DENSITY CRITERION question moved from a decision to a measurement. The
     panels print their criteria, and reading all of them showed that the "0.2
     above D-min" carried by 16 profiles IS printed -- on 5205 (both editions)
     and 5218 -- which contradicts what this project asserted from 2026-08-25 to
     2026-08-26. See the corrected note in verify.py.
  2. A GUARD WAS FOUND TO BE FRAGILE. The blue-peak 6/4 split pins an argmax on
     stocks whose maximum is a 40 nm plateau, so a re-trace by any reader could
     move 5246 and 5205 between its two groups with no data change. A second
     guard now asserts the stable property instead: each stored blue maximum
     must lie inside its own measured plateau.

⚠ AND CAPTION-BASED ASSIGNMENT WAS TRIED AND REJECTED, on evidence. These
panels do print "Yellow-/Magenta-/Cyan-Forming Layer" inside the frame, so
assigning layers by caption position looked strictly better than assigning them
by absorption band. It is not: on 5245 the "Magenta-" caption sits 43 nm from
one curve's peak and 47 nm from another's, which is not a decision. The BAND
test is the sound rule here and the captions are corroboration, not the key --
the reverse of the situation on the 7239 mono panel.

KNOWN LIMITS, stated rather than discovered later
-------------------------------------------------
The PANEL FINDER, not the ink reader, is what limits coverage -- but it limits it
much less than it did. A corpus sweep over all 2159 PDF pages, re-run 2026-08-25
after the writing-direction fix:

    aspect-ratio finder (`rot_labels`)      6 pages
    writing-direction finder (`rot_lines`)  21 pages

so the fix made **15 more sensitivity panels findable**, a 3.5x gain, including
5231 p3, 5245 p4, 5246 p5, 5248 p3, 5274 p4, 5279 p3, 5293 p4, V200T p4, the two
5205 sheets, 5218 p4, the 5219 brochure p3, 8532 p1, eterna_vivid500 p1 and 7239
p3. Only 7239 has been read; the rest are findable, not yet extracted, and each
still has to pass the tick calibration and the three-layer test.

Measured failures, with their causes:

  * `Ektachrome_100d.pdf` p4 (5285) -- "no frame right of the axis label". The
    caption sits INSIDE a decorative outer box whose x0 (42.0) is LEFT of the
    label (51.4), so `dye_density.pick`'s "frame must be right of the label"
    rule rejects it, and the real plot frame is not drawn as a separate path.
    Relaxing that rule would let the outer box win, and the tick windows are
    measured from the frame edges, so the calibration would then be wrong
    rather than absent. Needs its own anchor, not a looser tolerance.
  * `KODAK VISION Color Print Film 2383.pdf` p6 -- only 2 sensitivity ticks
    found against the frame; a print stock's panel is laid out differently.
  * Most other sheets draw the caption as OUTLINED VECTOR ART, so there is no
    rotated text to find at all -- the same class of problem that hid
    8532's printed date until 2026-08-23.

Neither 5285 nor 2383 blocks anything: 5285's spectral set is already adopted
from the raster batch and 2383 is a print stock.

Run:  python spectral_vector.py --root ../.. [--assert] [--sheet 5201]
Needs numpy + PyMuPDF. --assert exits non-zero if an extraction moves, or if a
sheet stops agreeing with an already-adopted set.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

import dye_density as dd

#: The database's spectral grid: film_profiles.SpectralSensitivity stores
#: lambda_start_nm 380 with lambda_step_nm 10, 31 samples, on every colour stock.
GRID = np.arange(380, 681, 10, dtype=float)

#: The floor sentinel, from the SpectralSensitivity docstring: "at or below the
#: measurement floor of the source plot". Also used to pad outside the traced
#: extent -- an extrapolated sensitisation tail would be an invention.
FLOOR = -4.0

AXIS_WORDS = {"LOG", "SENSITIVITY"}

#: Peak must land in this band, per layer. Wide enough to be a real test rather
#: than a restatement of the answer: a swap of any two layers fails it.
BANDS = {"b": (380.0, 500.0), "g": (500.0, 590.0), "r": (590.0, 700.0)}

#: sheet tag -> (pdf filename under PDF/PROFILES/KODAK, page, profile)
SHEETS = {
    "5201": ("Kodak VISION2 50D 5201.pdf", 3, "KODAK_VISION2_50D_5201"),
    # ⚠ 5217 IS THE VALIDATION SHEET, ADDED 2026-08-25d, AND ITS SET IS NOT
    # RE-ADOPTED. KODAK_VISION2_200T_5217 already carries a spectral set from the
    # 2026-08-02 RASTER batch -- a different image, a different method, a
    # different author. Re-deriving it from the same sheet's VECTOR paths is
    # therefore a genuine cross-validation of both, and it is the only one
    # available: no other sheet in the corpus has both an adopted spectral set
    # and a vector panel this reader can reach.
    #   AGREEMENT (see EXPECTED_VS_STORED): blue rms 0.049, green 0.091, red
    #   0.109 decades over the mutually-measured samples; peaks within ONE 10 nm
    #   grid step on all three layers (blue identical at 470).
    # That is inside the reading error of a printed plot, so neither method is
    # corrected and the stored arrays are left exactly as they were -- a wash is
    # not a reason to churn adopted data (the same rule dye_density.py applied
    # when its calibration changed).
    "5217": ("5217-Vision2-200T.pdf", 3, "KODAK_VISION2_200T_5217"),
    # ⚠ 7239 IS A MONOCHROME PANEL AND IS READ BY A DIFFERENT RULE. Its three
    # traces are all BLACK, so the ink convention this module was built on says
    # nothing about them; see extract_mono. It is also the sheet that forced two
    # other fixes -- the writing-direction caption finder (its caption is "LOG
    # SENSITIVITY", and the old aspect test could not see "LOG") and the
    # macron-minus tick reader (its y axis runs 2.0 to -2.0 with the negative
    # signs drawn as overbars that are not in the text layer).
    "7239": ("Kodak Eastman EKTACHROME Film (Daylight) 7239.pdf", 3,
             "EASTMAN_EKTACHROME_7239"),
}

#: Recorded 2026-08-25. --assert fails if an extraction stops reproducing.
#: (peak nm per layer r/g/b, absolute peak LOG SENSITIVITY per layer r/g/b).
#: The absolute peaks are kept because the stored arrays throw them away: the
#: schema normalises each layer to 0.0, so the fact that 5201's blue layer is
#: 0.22 decades more sensitive than its green and red ones survives only here
#: and in the profile's source string.
EXPECTED = {
    # Peaks land on the stored 10 nm grid, so 650 here is the sheet's ~645 nm
    # maximum rounded to the nearest sample -- the grid is the database's, not
    # the plot's. Absolute peaks 1.76 / 1.78 / 1.99 decades: the BLUE layer is
    # 0.22 decades more sensitive than the other two, which the stored arrays
    # cannot show because the schema normalises each layer separately.
    "5201": ((650.0, 540.0, 470.0), (1.76, 1.78, 1.99)),
    "5217": ((640.0, 540.0, 470.0), (2.17, 2.43, 2.81)),
    # 7239, read MONO (see extract_mono). Absolute peaks 1.25 / 1.10 / 1.70:
    # the blue-sensitive layer is 0.45 decades faster than the green one, which
    # is what a daylight-balanced reversal stock should show and what the stored
    # per-layer normalisation throws away.
    "7239": ((660.0, 560.0, 410.0), (1.25, 1.10, 1.70)),
}

#: For sheets whose profile ALREADY carries an independently-adopted set: the
#: rms agreement, per layer (r, g, b), in decades, over the samples both call
#: measured. Asserted, so a later change to either side shows up as a drift in
#: the agreement rather than silently replacing one method's numbers.
#: ⚠ THE STORED ARRAYS ARE NOT TOUCHED. This is a check, not an adoption.
EXPECTED_VS_STORED = {
    "5217": (0.109, 0.091, 0.049),
}


def rot_lines(pg, words):
    """Rotated text LINES whose text contains all of ``words``.

    ⚠ THIS REPLACES A HEURISTIC THAT FAILED ON SHORT WORDS, AND THE FAILURE COST
    A WHOLE SHEET. `rot_labels` below decides a word is rotated by its aspect
    ratio, `(y1 - y0) > 1.6 * (x1 - x0)`. That is true of "SENSITIVITY" rotated
    -- 11 characters tall, one wide -- and FALSE of "LOG", which is only three
    characters tall and therefore not obviously taller than it is wide. On
    EASTMAN_EKTACHROME_7239 p3 the caption reads "LOG SENSITIVITY": the long word
    was detected, the short one was not, the pair never met, and the panel was
    recorded as unreachable. Widening the aspect threshold would have admitted
    ordinary horizontal text everywhere else.

    PyMuPDF already knows the answer exactly. `get_text("dict")` gives each LINE
    a writing direction, and a rotated line reports `dir == (0, -1)` instead of
    `(1, 0)`. That is not a heuristic at all -- it is the direction the text was
    actually laid out in -- and it returns the caption already assembled, so the
    LABEL_STACK_GAP grouping is unnecessary here as well: PyMuPDF has done the
    grouping the way the document itself specifies.

    Returns the same shape `pick()` expects: (right edge, top, bottom, text).
    """
    want = {w.upper() for w in words}
    out = []
    for blk in pg.get_text("dict").get("blocks", []):
        for ln in blk.get("lines", []):
            if tuple(ln.get("dir", (1, 0))) == (1, 0):
                continue                       # horizontal: not an axis caption
            txt = " ".join(sp.get("text", "") for sp in ln.get("spans", []))
            up = txt.upper()
            if not all(w in up for w in want):
                continue
            x0, y0, x1, y1 = ln["bbox"]
            out.append((x1, y0, y1, " ".join(sorted(want))))
    return out


def rot_labels(pg):
    """Rotated y-axis captions containing LOG + SENSITIVITY, one per plot.

    Same column-then-vertical-run grouping as `dye_density.rot_labels`, and for
    the same reason: Kodak stacks several plots in one column, each with its own
    rotated caption, so grouping by x-centre alone merges them and the frame
    search then picks the wrong plot. That bug cost the 7239 sheet a fortnight;
    it is not being reintroduced here.
    """
    rot = []
    for x0, y0, x1, y1, t, *_ in pg.get_text("words"):
        if (y1 - y0) > 1.6 * (x1 - x0) and t.upper().strip(",.:*") in AXIS_WORDS:
            rot.append((x0, y0, x1, y1, t.upper().strip(",.:*")))
    cols: dict[float, list] = {}
    for w in rot:
        cols.setdefault(round((w[0] + w[2]) / 2 / 6) * 6, []).append(w)
    out = []
    for _, items in cols.items():
        runs: list[list] = []
        for w in sorted(items, key=lambda w: w[1]):
            if runs and w[1] - max(v[3] for v in runs[-1]) <= dd.LABEL_STACK_GAP:
                runs[-1].append(w)
            else:
                runs.append([w])
        for run in runs:
            if {"LOG", "SENSITIVITY"} <= {w[4] for w in run}:
                out.append((max(i[2] for i in run), min(i[1] for i in run),
                            max(i[3] for i in run), "LOG SENSITIVITY"))
    return out


def _sign_y_ticks(raw):
    """[(value, cy), ...] -> ({value: cy}, error) with the MACRON MINUS restored.

    ⚠ THE MINUS SIGN IS NOT IN THE TEXT LAYER. The 7239 sensitivity panel's y
    axis reads 2.0 / 1.0 / 0.0 / -1.0 / -2.0, and Kodak sets the two negative
    labels with an OVERBAR rather than a hyphen. PyMuPDF returns them as plain
    "1.0" and "2.0". Searched for the bar and it is not findable from here: it is
    neither a separate glyph in the span text nor a drawing -- the page holds no
    small path anywhere near the label column.

    ⚠ AND THE OLD CODE HID THAT BEHIND AN ACCIDENT. It keyed ticks by value with
    `setdefault`, so the second occurrence of each repeated value was dropped in
    silence. That LOOKS correct on this sheet, because the first occurrence in
    page order happens to be the positive branch -- but it is luck, not a rule.
    Had the page emitted the lower branch first, the fitted axis would have come
    out MIRRORED, still perfectly collinear, still inside TICK_RESID_PT, and
    every stored sensitivity would have carried the wrong sign with nothing
    anywhere to catch it. A silent coin flip is not a reader.

    The invariant that IS safe, and that these plots never violate: a sensitivity
    axis increases UPWARD. So a repeated label below the zero tick is the
    negative branch and one above it is positive. Once signed, ALL the ticks must
    fall on ONE line -- and that is a genuine test rather than a restatement,
    because a wrong sign assignment cannot be collinear with the rest.

    Unrepeated axes (5201, 5217) take the same path and are unchanged by it.
    """
    if not raw:
        return {}, "no sensitivity ticks against the frame"
    counts: dict[float, int] = {}
    for v, _cy in raw:
        counts[v] = counts.get(v, 0) + 1
    if max(counts.values()) == 1:
        return {v: cy for v, cy in raw}, None
    zeros = [cy for v, cy in raw if v == 0.0]
    if len(zeros) != 1:
        # ⚠ Fail rather than guess. Without exactly one zero tick there is no
        # anchor to sign the branches against, and the wrong choice is invisible.
        return {}, (f"repeated y tick values {sorted(k for k, n in counts.items() if n > 1)} "
                    f"with {len(zeros)} zero ticks -- cannot resolve the macron minus")
    cy0 = zeros[0]
    out: dict[float, float] = {}
    for v, cy in raw:
        signed = -v if cy > cy0 else v          # page y grows DOWNWARD
        if signed in out and abs(out[signed] - cy) > 1e-6:
            return {}, f"two different pixel rows both read as {signed:+.1f}"
        out[signed] = cy
    return out, None


def axis_cal(pg, fr):
    """(cal tuple, x residual, y residual, n_x, n_y) for one sensitivity frame.

    ⚠ THE TOP TICK LABEL IS NUDGED AND MUST BE ALLOWED TO LOSE. On 5201 the
    y labels 0.0/1.0/2.0/3.0 sit 22.95 pt apart and the 4.0 label sits 3.6 pt
    off that line, because it would otherwise collide with the frame edge. A
    two-point calibration anchored on 0.0 and 4.0 would spread the whole axis by
    16 %. `dd._fit_axis` drops the outlier instead, which is exactly the failure
    mode it was written for on the 5218 dye panel.
    """
    xs: dict[float, float] = {}
    y_raw: list[tuple[float, float]] = []
    for a, b, c, d, t, *_ in pg.get_text("words"):
        if not re.fullmatch(r"-?\d+(\.\d+)?", t):
            continue
        v = float(t)
        cx, cy = (a + c) / 2, (b + d) / 2
        if (fr.x0 - 8 <= cx <= fr.x1 + 8 and fr.y1 - 2 <= cy <= fr.y1 + 14
                and 200 <= v <= 900):
            xs.setdefault(v, cx)
        if (fr.x0 - 32 <= cx < fr.x0 - 1 and fr.y0 - 10 <= cy <= fr.y1 + 10
                and 0 <= v <= 6):
            y_raw.append((v, cy))
    ys, y_err = _sign_y_ticks(y_raw)
    if len(xs) < 3:
        return None, f"only {len(xs)} wavelength ticks against the frame"
    if y_err:
        return None, y_err
    if len(ys) < 3:
        return None, f"only {len(ys)} sensitivity ticks against the frame"
    fx = dd._fit_axis(xs)
    fy = dd._fit_axis(ys)
    if fx is None or fy is None:
        return None, "axis fit failed"
    if fx[2] > dd.TICK_RESID_PT:
        return None, f"wavelength ticks not collinear ({fx[2]:.2f} pt)"
    if fy[2] > dd.TICK_RESID_PT:
        return None, f"sensitivity ticks not collinear ({fy[2]:.2f} pt)"
    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = min(ys), max(ys)
    cal = (fx[0] * lo_x + fx[1], lo_x, fx[0] * hi_x + fx[1], hi_x,
           fy[0] * lo_y + fy[1], lo_y, fy[0] * hi_y + fy[1], hi_y)
    return (cal, fx[2], fy[2], len(xs), len(ys)), None


def _trace_extent(pg, cal, fr, ink_name):
    """(lambda_min, lambda_max) of the first path in `ink_name`, page order."""
    for p in pg.get_drawings():
        r = p["rect"]
        if not (r.x0 >= fr.x0 - 6 and r.x1 <= fr.x1 + 6
                and r.y0 >= fr.y0 - 6 and r.y1 <= fr.y1 + 6):
            continue
        if dd._ink(p) != ink_name:
            continue
        if sum(1 for it in p["items"] if it[0] in ("l", "c")) < dd.INK_MIN_SEG:
            continue
        if r.width > 0.98 * fr.width and r.height > 0.98 * fr.height:
            continue
        lam = [(x - cal[0]) / (cal[2] - cal[0]) * (cal[3] - cal[1]) + cal[1]
               for x, _ in dd.flatten(p["items"])]
        return min(lam), max(lam)
    return None


#: A mono subpath must cover at least this fraction of the frame in x AND vary
#: at least this fraction in y to be a TRACE rather than a rule. Set from what
#: the two kinds actually look like, not tuned: on 7239 the three traces span
#: 0.60-0.93 of the frame width and 0.28-0.65 of its height, while every grid
#: line has ZERO extent in one axis by construction. Nothing sits in between.
MONO_MIN_X_FRAC = 0.15
MONO_MIN_Y_FRAC = 0.05

#: What counts as "black ink" for the mono reader: DARK and NEUTRAL, not equal
#: to zero.
#:
#: ⚠ THE FIRST VERSION TESTED `every channel <= 0.10` AND MISSED A WHOLE SHEET.
#: 7239 draws its traces at exactly (0, 0, 0), so the threshold looked settled.
#: H-1-5222 draws its two curves at (0.137, 0.122, 0.125) -- a rich black with a
#: slight warm cast, which is an ordinary thing for a printer to specify and
#: which that test rejected outright, returning "0 curves" from a panel that has
#: two. The property actually wanted is that the stroke carries no HUE: a
#: coloured trace on these sheets means something (see the ink convention at the
#: top of this file), and a neutral one does not.
MONO_MAX_CHANNEL = 0.30
MONO_MAX_CHROMA = 0.05


def subpaths(items):
    """Vertices of each SUBPATH in a drawing, split at pen-up discontinuities.

    ⚠ WITHOUT THIS, THREE CURVES READ AS ONE. On the 7239 sensitivity panel the
    whole plot is TWO drawing objects, and neither is one curve: object 26 holds
    the yellow-forming and magenta-forming traces (290 segments) and object 29
    holds the cyan-forming trace alone (1067). Any reader working at drawing
    granularity therefore sees a curve that jumps from 552 nm back to 404 nm and
    resamples the two into a single meaningless function of wavelength.

    PDF marks the break exactly: a new subpath begins where a segment's start
    point is not the previous segment's end point. That is not a heuristic --
    it is the pen lifting -- so the split is as reliable as the path data.
    """
    out, cur, last = [], [], None
    for it in items:
        if it[0] == "l":
            a, b = it[1], it[2]
        elif it[0] == "c":
            a, b = it[1], it[4]
        else:                                    # "re" and friends: not a trace
            if cur:
                out.append(cur)
            cur, last = [], None
            continue
        if last is None or abs(a.x - last.x) > 1e-6 or abs(a.y - last.y) > 1e-6:
            if cur:
                out.append(cur)
            cur = [(a.x, a.y)]
        cur.append((b.x, b.y))
        last = b
    if cur:
        out.append(cur)
    return out


def extract_mono(pg, cal, fr, grid):
    """[(curve on ``grid``, (lambda_lo, lambda_hi)), ...] for a BLACK panel.

    ⚠ THE INK RULE DOES NOT APPLY HERE, AND SAYING SO IS THE POINT. Everything
    this module did until now rested on Kodak drawing each trace in the colour of
    light it concerns. Some sheets -- 7239 among them -- print the sensitivity
    panel entirely in BLACK and name the layers with captions inside the frame
    instead. On such a sheet the assignment cannot come from the palette, so it
    comes from the two checks that were previously corroboration: each trace must
    peak inside its own absorption band, and the three peaks must ascend. Those
    are applied by the caller, unchanged, and `layer_captions` adds Kodak's own
    in-frame words as an independent third.

    ⚠ Consequence worth stating plainly: a mono panel is read with ONE FEWER
    independent check than an inked one. It is adopted only when the band test,
    the ordering test and the caption order all agree.
    """
    out = []
    for p in pg.get_drawings():
        r = p["rect"]
        if not (r.x0 >= fr.x0 - 6 and r.x1 <= fr.x1 + 6
                and r.y0 >= fr.y0 - 6 and r.y1 <= fr.y1 + 6):
            continue
        col = p.get("color")
        if (not col or max(col) > MONO_MAX_CHANNEL
                or (max(col) - min(col)) > MONO_MAX_CHROMA):
            continue
        for pts in subpaths(p["items"]):
            xs = [q[0] for q in pts]
            ys = [q[1] for q in pts]
            if (max(xs) - min(xs)) < MONO_MIN_X_FRAC * fr.width:
                continue                          # a vertical rule, or a stub
            if (max(ys) - min(ys)) < MONO_MIN_Y_FRAC * fr.height:
                continue                          # a horizontal rule
            if ((max(xs) - min(xs)) > 0.98 * fr.width
                    and (max(ys) - min(ys)) > 0.98 * fr.height):
                continue                          # the frame itself
            y = dd.resample(pts, cal, grid)
            if not np.isfinite(y).all():
                continue
            lam = [(x - cal[0]) / (cal[2] - cal[0]) * (cal[3] - cal[1]) + cal[1]
                   for x in xs]
            out.append((y, (min(lam), max(lam))))
    return out


#: The in-frame captions Kodak prints on a mono sensitivity panel, in the layer
#: order this module uses. Matched on the first word only: the caption is set
#: over three lines ("Yellow-" / "Forming" / "Layer") and PyMuPDF returns each
#: line separately, so the distinguishing word is the one that matters.
CAPTION_WORDS = (("b", "YELLOW"), ("g", "MAGENTA"), ("r", "CYAN"))


def layer_captions(pg, fr, cal):
    """{layer: caption centre in nm} for captions printed INSIDE the frame.

    Kodak's own words, machine-read -- the same standing as the legend swatches
    the inked reader checks against. Returns only what it finds; the caller
    treats a missing caption as "no third check available", not as a failure,
    because most sheets do not print them.
    """
    out = {}
    for blk in pg.get_text("dict").get("blocks", []):
        for ln in blk.get("lines", []):
            txt = " ".join(sp.get("text", "") for sp in ln.get("spans", []))
            up = txt.upper()
            x0, y0, x1, y1 = ln["bbox"]
            if not (fr.x0 <= (x0 + x1) / 2 <= fr.x1
                    and fr.y0 <= (y0 + y1) / 2 <= fr.y1):
                continue
            for key, word in CAPTION_WORDS:
                if word in up and key not in out:
                    cx = (x0 + x1) / 2
                    out[key] = ((cx - cal[0]) / (cal[2] - cal[0])
                                * (cal[3] - cal[1]) + cal[1])
    return out


def normalise(raw, extent):
    """Peak-normalise to 0.0, pad outside the traced extent with the floor.

    The raw values are LOG SENSITIVITY as printed. Outside the traced extent the
    plot says nothing at all -- the trace simply stops, usually where it dives
    off the bottom of the frame -- so those samples get FLOOR rather than a
    continuation of the last value, which would invent sensitisation.
    """
    lo, hi = extent
    inside = (GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)
    if not inside.any():
        return None, 0.0
    peak = float(raw[inside].max())
    out = np.where(inside, raw - peak, FLOOR)
    return np.clip(out, FLOOR, 0.0), peak


def extract_sheet(root: Path, tag: str):
    import pymupdf
    fn, pgno, prof = SHEETS[tag]
    pdf = root / "PDF" / "PROFILES" / "KODAK" / fn
    if not pdf.is_file():
        return None, f"source not present: {fn}"
    pg = pymupdf.open(pdf)[pgno - 1]
    # Writing-direction lines first -- exact, and it finds captions the
    # aspect-ratio detector misses. The word-based finder is kept as a fallback
    # for pages whose text layer has no usable line structure.
    axes = rot_lines(pg, ("LOG", "SENSITIVITY")) or rot_labels(pg)
    for ax in axes:
        # ⚠ TRY EVERY CANDIDATE FRAME, NOT JUST THE NEAREST. `dd.pick` returns the
        # frame with the smallest x0 that lies right of the caption, which is the
        # right rule for the dye panels it was written for and the wrong one here:
        # on 7239 p3 two rects qualify (x0 351.8 and 362.7) and the TICK LABELS
        # SIT BETWEEN THEM at cx 352.6. The nearer frame therefore has its own
        # labels inside it, where the "left of the frame" tick window cannot see
        # them, and the panel reads as uncalibratable. Ordering the candidates by
        # distance and taking the first that actually CALIBRATES costs nothing on
        # the sheets that already worked -- the nearest frame is still tried
        # first -- and turns a hard failure into a fallback.
        lx, ly0, ly1 = ax[0], ax[1], ax[2]
        cands = sorted(
            (fr for fr in dd.frames(pg)
             if fr.x0 >= lx - 2 and not (fr.y1 < ly0 - 30 or fr.y0 > ly1 + 30)),
            key=lambda r: r.x0)
        cal_r = None
        for fr in cands:
            cal_r, err = axis_cal(pg, fr)
            if cal_r is not None:
                break
        if cal_r is None:
            continue
        cal = cal_r[0]
        inked = dd.extract_inked(pg, cal, fr, GRID)
        # red = the yellow-under-magenta overprint; assert the pair coincides
        reds = inked.get("yellow", []) + inked.get("magenta", [])
        if len(reds) == 2 and float(np.abs(reds[0] - reds[1]).max()) > 1e-9:
            return None, "the two red-ink paths are not an overprint pair"
        raw = {}
        if len(inked.get("blue", [])) == 1:
            raw["b"] = (inked["blue"][0], _trace_extent(pg, cal, fr, "blue"))
        if len(inked.get("green", [])) == 1:
            raw["g"] = (inked["green"][0], _trace_extent(pg, cal, fr, "green"))
        if reds:
            raw["r"] = (reds[0], _trace_extent(pg, cal, fr, "magenta")
                        or _trace_extent(pg, cal, fr, "yellow"))
        method = "ink"
        captions = ""
        if len(raw) != 3:
            # ⚠ MONO FALLBACK. Not every Kodak panel is inked; see extract_mono.
            # Tried only after the palette has failed, so no inked sheet changes
            # method, and the three-layer requirement is identical either way.
            mono = extract_mono(pg, cal, fr, GRID)
            if len(mono) != 3:
                return None, (f"expected 3 layers in the frame: ink gave "
                              f"{sorted(raw)} from {sorted(inked)}, "
                              f"black paths gave {len(mono)}")

            def _peak_nm(item):
                v, (lo, hi) = item
                ins = (GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)
                return float(GRID[np.argmax(np.where(ins, v, -np.inf))])

            mono.sort(key=_peak_nm)
            raw = {"b": mono[0], "g": mono[1], "r": mono[2]}
            method = "mono"
            caps = layer_captions(pg, fr, cal)
            if len(caps) == 3:
                order = [k for k, _v in sorted(caps.items(),
                                               key=lambda kv: kv[1])]
                if order != ["b", "g", "r"]:
                    return None, ("the in-frame layer captions run "
                                  + "/".join(order)
                                  + " across the frame, not b/g/r")
                captions = " ".join(f"{k}@{v:.0f}" for k, v in
                                    sorted(caps.items(), key=lambda kv: kv[1]))
        out, peaks, lams = {}, {}, {}
        for k, (v, ext) in raw.items():
            if ext is None:
                return None, f"no extent for the {k} layer"
            norm, peak = normalise(v, ext)
            if norm is None:
                return None, f"the {k} layer's extent misses the stored grid"
            out[k], peaks[k] = norm, peak
            # the peak is the sample at 0.0 by construction, but it must be
            # sought among the MEASURED samples only -- the floor-padded ones
            # carry no information and an all--inf comparison would silently
            # return index 0, i.e. the left edge of the grid.
            lams[k] = float(GRID[np.argmax(
                np.where(norm > FLOOR + 1e-9, norm, -np.inf))])
        for k, (lo, hi) in BANDS.items():
            if not lo <= lams[k] <= hi:
                return None, (f"the {k} layer peaks at {lams[k]:.0f} nm, "
                              f"outside {lo:.0f}-{hi:.0f}")
        if not lams["b"] < lams["g"] < lams["r"]:
            return None, "layer peaks are not in ascending wavelength order"
        return dict(tag=tag, profile=prof, file=fn, page=pgno,
                    log_s_r=out["r"], log_s_g=out["g"], log_s_b=out["b"],
                    peak_r=peaks["r"], peak_g=peaks["g"], peak_b=peaks["b"],
                    lam_r=lams["r"], lam_g=lams["g"], lam_b=lams["b"],
                    x_resid=cal_r[1], y_resid=cal_r[2],
                    n_x=cal_r[3], n_y=cal_r[4],
                    method=method, captions=captions), None
    return None, "no LOG SENSITIVITY panel yielded three inked layers"


#: MONOCHROME sheets: one sensitive layer, so one curve and `log_s_pan`.
#: (pdf under PDF/PROFILES/KODAK, page, profile, the printed caption to adopt)
#:
#: ⚠ THE CAPTION IS PART OF THE KEY BECAUSE THESE PANELS DRAW MORE THAN ONE
#: CURVE. H-1-5222 prints TWO, "D = 0.3 Above Gross Fog" and "D = 1.0 Above
#: Gross Fog" -- the same emulsion read to two density criteria, which on a
#: colour sheet would be two layers. Picking "the curve" without naming the
#: criterion would silently store whichever the page happened to emit first,
#: and the two differ by about 0.55 decades at their peaks.
MONO_SHEETS = {
    "5222": ("EASTMAN DOUBLE-X Negative Film 5222.pdf", 3,
             "EASTMAN_DOUBLE_X_5222", "D = 1.0 Above Gross Fog"),
}

#: Recorded 2026-08-26. (peak nm, absolute peak log sensitivity, measured
#: sample count, rms in decades against the set the profile ALREADY stored).
#:
#: ⚠ THE LAST NUMBER IS A CROSS-VALIDATION OF TWO EDITIONS OF ONE PUBLICATION,
#: and it is why this set was re-adopted rather than left alone. The corpus held
#: H-1-5222 revised 3-26, whose panels are RASTER (p3 carries three images and
#: three drawing paths); the owner supplied H-1-5222 revised 7-15, whose panels
#: are entirely VECTOR. Same publication, eleven years apart, same curve. The
#: old set was read off the raster by hand with no method recorded; this one is
#: traced off the paths with its residuals reported, and the two agree to
#: **rms 0.037 decades, worst 0.122**, peaking on the same 430 nm sample.
#: A wash is not normally a reason to churn adopted data -- that rule is why
#: 5217's set was left alone on 2026-08-25. The difference here is that the
#: agreement is not between two comparable methods: it is a hand reading being
#: confirmed by a machine one, so adopting the trace upgrades the provenance
#: without moving the numbers, and the 0.037 is the evidence for that claim.
#: ⚠ THE LAST NUMBER CHANGED MEANING THE MOMENT THE SET WAS ADOPTED, and it is
#: worth stating because the first version of this line failed the build one run
#: after it was written. Before adoption it was the agreement against the OLD
#: hand-read set: 0.037. After adoption the profile holds THIS trace, so the same
#: comparison now measures only the 2-decimal rounding the profile literal
#: applies -- 0.003. A pinned cross-check against a set you have just replaced
#: is a check against yourself, and the honest thing is to say so: the 0.037
#: agreement is recorded in the prose above and in the profile's source string,
#: where it stays true, and this number now guards the rounding instead.
MONO_EXPECTED = {
    "5222": (430.0, 0.904, 24, 0.003),
}


def extract_mono_sheet(root: Path, tag: str):
    """One panchromatic curve, chosen by the criterion caption printed beside it.

    Same reader as the colour path -- same caption finder, same frame fallback,
    same macron-minus tick signing, same `extract_mono` -- with the three-layer
    assignment replaced by a one-curve SELECTION. The curve is identified by the
    in-frame caption that names its density criterion, matched by geometry: the
    caption sits directly above its own curve, so evaluating every traced curve
    at the caption's x-centre and taking the nearest one BELOW it is the
    association the page itself draws.
    """
    import pymupdf
    fn, pgno, prof, want_caption = MONO_SHEETS[tag]
    pdf = root / "PDF" / "PROFILES" / "KODAK" / fn
    if not pdf.is_file():
        return None, f"source not present: {fn}"
    pg = pymupdf.open(pdf)[pgno - 1]
    axes = rot_lines(pg, ("LOG", "SENSITIVITY")) or rot_labels(pg)
    for ax in axes:
        lx, ly0, ly1 = ax[0], ax[1], ax[2]
        cands = sorted(
            (fr for fr in dd.frames(pg)
             if fr.x0 >= lx - 2 and not (fr.y1 < ly0 - 30 or fr.y0 > ly1 + 30)),
            key=lambda r: r.x0)
        cal_r = None
        for fr in cands:
            cal_r, _err = axis_cal(pg, fr)
            if cal_r is not None:
                break
        if cal_r is None:
            continue
        cal = cal_r[0]
        curves = extract_mono(pg, cal, fr, GRID)
        if not curves:
            continue
        # the criterion captions printed inside the frame, with their positions
        caps = []
        for blk in pg.get_text("dict").get("blocks", []):
            for ln in blk.get("lines", []):
                txt = " ".join(sp.get("text", "") for sp in ln.get("spans", []))
                x0, y0, x1, y1 = ln["bbox"]
                if not (fr.x0 <= (x0 + x1) / 2 <= fr.x1
                        and fr.y0 <= (y0 + y1) / 2 <= fr.y1):
                    continue
                if "ABOVE" in txt.upper() and "=" in txt:
                    caps.append((txt.strip(), (x0 + x1) / 2, y1))
        if not caps:
            return None, "no density-criterion caption printed inside the frame"
        hit = [c for c in caps if c[0] == want_caption]
        if not hit:
            return None, (f"the sheet prints {[c[0] for c in caps]}, not "
                          f"{want_caption!r} -- the criterion moved")
        _txt, cap_x, cap_y = hit[0]
        cap_nm = ((cap_x - cal[0]) / (cal[2] - cal[0]) * (cal[3] - cal[1])
                  + cal[1])
        # ⚠ MATCH BY GEOMETRY, NOT BY PAGE ORDER. Each caption sits directly
        # above its own curve; page order is an emission detail and would
        # silently pick the other criterion on a sheet that emitted them the
        # other way round.
        best, best_gap = None, None
        for v, ext in curves:
            lo, hi = ext
            if not (lo - 1e-9 <= cap_nm <= hi + 1e-9):
                continue
            here = float(np.interp(cap_nm, GRID, v))
            # page y grows downward; "below the caption" is a LOWER log
            # sensitivity than the caption's own row
            cap_val = ((cap_y - cal[4]) / (cal[6] - cal[4]) * (cal[7] - cal[5])
                       + cal[5])
            gap = cap_val - here
            if gap < 0:
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = (v, ext), gap
        if best is None:
            return None, f"no traced curve sits below the {want_caption!r} caption"
        raw, ext = best
        norm, peak = normalise(raw, ext)
        if norm is None:
            return None, "the traced extent misses the stored grid"
        meas = norm > FLOOR + 1e-9
        lam = float(GRID[np.argmax(np.where(meas, norm, -np.inf))])
        return dict(tag=tag, profile=prof, file=fn, page=pgno,
                    log_s_pan=norm, peak=peak, lam=lam, n_meas=int(meas.sum()),
                    caption=want_caption, captions=[c[0] for c in caps],
                    x_resid=cal_r[1], y_resid=cal_r[2],
                    n_x=cal_r[3], n_y=cal_r[4]), None
    return None, "no LOG SENSITIVITY panel yielded a monochrome curve"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--sheet", action="append",
                    choices=sorted(set(SHEETS) | set(MONO_SHEETS)))
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--dump", action="store_true",
                    help="print the arrays in film_profiles.py form")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()
    # ⚠ --sheet may name a MONO sheet, which the colour loop must not try to
    # open: SHEETS and MONO_SHEETS are different registries read by different
    # functions, and the argparse `choices` list is the union of the two.
    tags = [t for t in (ns.sheet or sorted(SHEETS)) if t in SHEETS]
    bad = skipped = 0
    print(f"[i] corpus root {root}")
    for tag in tags:
        got, err = extract_sheet(root, tag)
        if got is None:
            if "not present" in (err or ""):
                print(f"  [SKIP] {tag}: {err}")
                skipped += 1
            else:
                print(f"  [FAIL] {tag}: {err}")
                bad += 1
            continue
        want_lams, want_peaks = EXPECTED[tag]
        lams = (got["lam_r"], got["lam_g"], got["lam_b"])
        peaks = (got["peak_r"], got["peak_g"], got["peak_b"])
        ok = (lams == want_lams
              and all(abs(a - b) < 0.03 for a, b in zip(peaks, want_peaks)))
        print(f"  [{'OK  ' if ok else 'FAIL'}] {tag} {got['profile']:24s} "
              f"peaks r{lams[0]:.0f} g{lams[1]:.0f} b{lams[2]:.0f} nm  "
              f"log S {peaks[0]:.2f}/{peaks[1]:.2f}/{peaks[2]:.2f}  "
              f"ticks {got['n_x']}x/{got['n_y']}y "
              f"resid {got['x_resid']:.2f}/{got['y_resid']:.2f} pt")
        if not ok:
            print(f"         expected peaks at {want_lams} nm, "
                  f"log S {want_peaks}")
            bad += 1
        if tag in EXPECTED_VS_STORED:
            # Cross-validate against the already-adopted set. Compared only on
            # samples BOTH sides call measured: a floor sentinel on either side
            # carries no information, and including it would manufacture a
            # 4-decade "disagreement" out of two different trace extents.
            from film_profiles import get_profile
            st = get_profile(got["profile"]).spectral
            got_rms, drift = [], False
            for key, stored in (("log_s_r", st.log_s_r), ("log_s_g", st.log_s_g),
                                ("log_s_b", st.log_s_b)):
                a = np.asarray(stored, dtype=float)
                b = got[key]
                m = (a > FLOOR + 0.01) & (b > FLOOR + 0.01)
                got_rms.append(float(np.sqrt(((a[m] - b[m]) ** 2).mean()))
                               if m.any() else float("nan"))
            want_rms = EXPECTED_VS_STORED[tag]
            drift = any(not (abs(a - b) < 0.01)
                        for a, b in zip(got_rms, want_rms))
            print(f"         vs the ADOPTED set: rms r {got_rms[0]:.3f} "
                  f"g {got_rms[1]:.3f} b {got_rms[2]:.3f} decades "
                  f"({'DRIFTED' if drift else 'as recorded'}) -- "
                  f"cross-check only, nothing re-adopted")
            if drift:
                print(f"         expected rms {want_rms}")
                bad += 1
        if ns.dump:
            for k in ("log_s_r", "log_s_g", "log_s_b"):
                print(f"            {k}=({', '.join('%.2f' % v for v in got[k])}),")
    # ---- the MONOCHROME sheets: one curve, chosen by its criterion caption ----
    for tag in (ns.sheet or sorted(MONO_SHEETS)):
        if tag not in MONO_SHEETS:
            continue
        got, err = extract_mono_sheet(root, tag)
        if got is None:
            if "not present" in (err or ""):
                print(f"  [SKIP] {tag}: {err}")
                skipped += 1
            else:
                print(f"  [FAIL] {tag}: {err}")
                bad += 1
            continue
        want_lam, want_peak, want_n, want_rms = MONO_EXPECTED[tag]
        from film_profiles import get_profile
        st = np.asarray(get_profile(got["profile"]).spectral.log_s_pan,
                        dtype=float)
        rms = float("nan")
        if st.size:
            b = got["log_s_pan"][:st.size]
            m = (st > FLOOR + 0.01) & (b > FLOOR + 0.01)
            if m.any():
                rms = float(np.sqrt(((st[m] - b[m]) ** 2).mean()))
        ok = (got["lam"] == want_lam
              and abs(got["peak"] - want_peak) < 0.03
              and got["n_meas"] == want_n
              and (rms != rms or abs(rms - want_rms) < 0.01))
        print(f"  [{'OK  ' if ok else 'FAIL'}] {tag} {got['profile']:24s} "
              f"pan peak {got['peak']:.2f} @ {got['lam']:.0f} nm, "
              f"{got['n_meas']} measured samples  "
              f"ticks {got['n_x']}x/{got['n_y']}y "
              f"resid {got['x_resid']:.2f}/{got['y_resid']:.2f} pt")
        print(f"         criterion adopted from the panel's own caption "
              f"{got['caption']!r} (the sheet also prints "
              f"{[c for c in got['captions'] if c != got['caption']]})")
        print(f"         vs the set the profile now holds: rms {rms:.3f} "
              f"decades -- this is the storage rounding, NOT the cross-method "
              f"agreement; that was 0.037 against the hand reading this "
              f"replaced, and it is recorded in the profile's source string")
        if not ok:
            print(f"         expected peak {want_peak:.2f} @ {want_lam:.0f} nm, "
                  f"{want_n} samples, rms {want_rms:.3f}")
            bad += 1
        if ns.dump:
            print("            log_s_pan=("
                  + ", ".join("%.2f" % v for v in got["log_s_pan"]) + "),")
    print(f"\n[i] {len(tags) + len(MONO_SHEETS) - bad - skipped} reproduced, "
          f"{bad} failed, {skipped} skipped")
    if ns.do_assert and bad:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
