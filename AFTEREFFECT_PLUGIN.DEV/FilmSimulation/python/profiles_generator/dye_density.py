"""Spectral dye density from Kodak/Fuji datasheet VECTOR paths.

Extracts the cyan / magenta / yellow (and, where the sheet prints it, the
visual-neutral) traces from a "Spectral Dye Density" / "DIFFUSE SPECTRAL
DENSITY" plot and resamples them onto the schema's 400-700 nm / 10 nm grid.

VALIDATION -- this is the part that makes the numbers trustworthy
----------------------------------------------------------------
Two sets were already adopted in the database from ad-hoc extractions that were
never reproducible. This script re-derives both and is asserted against them:

  * KODAK_EKTACHROME_100D_5285 (Ektachrome_100d.pdf p4, publication H-1-5285)
    reproduces the adopted arrays to RMS 0.0029 / 0.0014 / 0.0048 / 0.0039 D
    for cyan / magenta / yellow / neutral. The next-best curve assignment is
    two orders of magnitude worse (RMS 0.52-0.69), so the assignment is not a
    coin toss.
  * KODAK_2383_RELEASE (KODAK VISION Color Print Film 2383.pdf p6) closes its
    neutral = C+M+Y check to 0.135 D, against the 0.128 D that sheet's own
    provenance records as its base-absorber offset.

TWO NORMALISATION FAMILIES, and getting this wrong rejects everything
--------------------------------------------------------------------
Kodak does not plot this the same way on every sheet, and the schema already
anticipates it (SpectralDyeDensity.normalisation lists "peak_1.0"):

  A. AS-PRINTED + NEUTRAL (5285, 2383): four traces, and the visual-neutral
     trace equals the sum of the three dyes. That identity IDENTIFIES the
     quartet -- everything else on these pages (spectral sensitivity,
     granularity, MTF, characteristic curves) fails it. It is NOT exact on
     print stock: 2383 needs a 0.14 D tolerance because of the base absorber.
  B. PEAK-NORMALISED (the VISION2/VISION3 sheets): three traces, each scaled to
     unit peak, and NO neutral. The sum identity cannot hold. A first pass that
     required it rejected every sheet except 5285.

     What family B captures is the SHAPE of each dye's absorption -- including
     the off-band unwanted absorption a 3x3 dye_matrix cannot express -- and
     NOT the absolute density level. Adopted values are tagged "peak_1.0" so
     that limitation travels with the data.
  C. PEAK-NORMALISED + NEUTRAL + DMIN (5201), added 2026-08-25, queue item C9.
     FIVE traces: the three unit-peak dyes plus an as-printed Midscale Neutral
     and an as-printed Minimum Density on the same axes. Only the three dyes are
     returned and stored -- mixing an as-printed trace into a peak_1.0 record
     would make the record mean two different things at once, which is the same
     reason 5217 and 5218 do not store theirs.

     ⚠ THE QUEUE ENTRY'S DIAGNOSIS WAS WRONG, and the real cause is worth having
     written down. C9 recorded this sheet as a FAMILY CLASSIFIER limitation --
     "built for 3 dye traces or 3 dyes + neutral, not 3 + neutral + dmin". It is
     not: family B takes combinations of THREE out of however many curves are
     offered, so two extra traces cost it nothing. What actually happened is that
     the CYAN trace never reached the classifier. Kodak draws it as TWO
     overprinted paths (yellow under magenta, making red on the page) of 7
     segments each, and `extract`'s `n < 8` segment filter dropped both. With no
     curve left in the 615-700 nm band, no triple could pass `_bands_ok` and the
     sheet reported "no curve set matched either normalisation family" -- a true
     statement about a curve list that was missing the curve.

     The fix is not a lower segment threshold, which would let gridline stubs in
     on every other sheet. It is to identify the traces by their INK, which this
     sheet makes unambiguous.

PLOT LOCATION: these pages carry four or five plots and their tick labels share
the page, so a purely numeric detector merges them (it gave 5285 y[0-4.0]
instead of y[0-1.5]). The plot is found from its ROTATED axis label, then the
frame right of that label, then only the tick labels that fall against THAT
frame. This is method rule 2 applied to this plot type: confirm against the
title text near the path's rect.

KNOWN LIMITS, stated rather than discovered later
-------------------------------------------------
  * Sheets with HORIZONTAL axis labels are not handled. 5247.pdf p4 is the
    known case: it is vector (34 paths, zero images) but carries no rotated
    text, no plot frame over 80x50 pt and no wavelength ticks in the text
    layer, so nothing anchors the search. It needs a visual pass.
  * "No raster image on the page" is what plot_inventory.py calls vector, and
    that is NOT the same as "a vector plot is present" -- 5247 p4 proves it.
  * Sheets not yet handled: 5246 ONLY. Three of the original five failures --
    7239, 5217, 5218 -- were recovered on 2026-08-18 and NONE of them needed
    anything from the source; all three were defects in this script: a y-axis
    caption anchor that merged two stacked plots into one band, a two-point tick
    calibration with nothing checking it, and a stroke-width filter referenced to
    the thickest path in the frame rather than to the curves. See
    LABEL_STACK_GAP, _fit_axis and the note in extract().

    ⚠ 5248 WAS NEVER A FAILED EXTRACTION AND HAS BEEN RECLASSIFIED (2026-08-25d).
    The recorded symptom -- "only 2 curves survive inside the frame, so the other
    traces are being lost before selection" -- assumed traces that do not exist.
    5248 p3 prints, in words: "Typical densities for a midscale neutral subject
    and D-min.", and draws exactly TWO labelled traces, Midscale Neutral and
    Minimum Density. There are no separate dye curves on that sheet at all, so
    `SpectralDyeDensity.validate()` (which requires cyan AND magenta AND yellow)
    can never be satisfied from it. This is the SAME schema-shape mismatch
    already recorded for FUJI_SUPER_F125_8532, and 5248 is now its second
    instance -- which is the evidence the pending schema decision needs (whether
    to carry an as-printed neutral+Dmin pair), not an extractor bug.

    5246 p5 IS REFUSED, AND AS OF 2026-08-26 THE ALTERNATIVES ARE EXCLUDED BY
    MEASUREMENT RATHER THAN LEFT OPEN. Three explanations were on the table and
    all three are now dead:
      1. "A tolerance problem" -- no; see the peak detail kept below.
      2. "A label-matching problem" -- no, and this was tested with the tool
         built for exactly it. The MONO reader and geometric caption matcher
         written on 2026-08-25/26 for the 7239 and 5222 panels -- which associate
         a caption with the curve directly beneath it -- were pointed at this
         panel and do not resolve it either. THE LABELS ARE PLACED IN WHITESPACE,
         NOT ON CURVES: "Cyan" sits at 558 nm, where the cyan dye is at 0.37 and
         four other traces lie within 0.2 D of it; "Magenta" sits at 681 nm, on
         the magenta tail rather than near its 542 nm peak. No positional rule
         can work on a legend laid out this way, and that is a property of the
         SHEET, not of the reader.
      3. "Two products on one plate" -- the most attractive explanation, since
         this sheet's header names 5246 AND 7246, and it is now EXCLUDED. Two
         products would pair the traces into near-parallel couples. They do not:
         over every shared span the closest pair has standard deviation 0.103 D
         and a range of 0.330 D across 451-670 nm, and the next closest has mean
         difference -0.016 D with sd 0.120 -- a CROSSING, not an offset. There is
         no pairing structure to find.
    ⚠ WHAT THE PANEL ACTUALLY CONTAINS, counted at one sample per 20 nm: SEVEN
    solid traces coexist between 480 and 580 nm, plus one dashed, against FIVE
    legend entries (Yellow, Magenta, Cyan, Midscale Neutral, Minimum Density).
    Two solid traces are unaccounted for by the legend, and the family-C identity
    `Neutral - Dmin = k(C+M+Y)` fails in every assignment tried: the best
    combination returns coefficients spread 136 % where 5201 gave 5.4 %.
    ⚠ The dashed trace IS identifiable and is the only one that is: Minimum
    Density, by shape (0.97 at 450 nm falling to 0.15 at 700 -- a masked
    negative's mask) and by being the one label whose nearest trace is
    unambiguous, 0.077 D against 0.512 for the runner-up.
    ⚠ SO THIS SHEET IS NOT BLOCKED ON THIS EXTRACTOR. It needs a statement of
    what its two extra traces are, which no amount of tracing can supply. Queue
    B1 listed it as "path proven, no dependency"; that was wrong and is corrected.

    The earlier record, kept because the peak detail is still true:
      * The panel draws SEVEN traces and labels FIVE. Nearest-label assignment
        resolves Yellow -> a trace peaking 1.008 at 446 nm and Magenta -> 1.006
        at 542 nm, both unit-peak as the panel's own note requires ("Cyan,
        Magenta, and Yellow Dye Curves are peak-normalized"), so those two are
        not in doubt.
      * THE CYAN IS. The trace nearest the "Cyan" label peaks at **0.943** at
        660 nm -- 0.057 short of the unit peak the sheet claims -- and there is a
        SECOND unlabelled trace in the same band peaking 0.660 at 683 nm, plus a
        second unlabelled high trace (1.303 at 435) beside the labelled Midscale
        Neutral (1.270 at 451). Two traces on this plate are unaccounted for.
      * So the blocker is not a tolerance. Widening the peak tolerance to admit
        0.943 would also admit false sets on other sheets, and picking the
        labelled trace anyway would be adopting a curve that fails the sheet's
        own stated normalisation. What would close it: any statement of what the
        two extra traces are -- H-1-5246t names 5246 AND 7246 in its header, so
        a second gauge or a second process condition are both live candidates.

Run:  python dye_density.py [--assert] [--sheet 5219]
Needs numpy + PyMuPDF. --assert exits non-zero if the adopted sets move.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np

GRID = np.arange(400, 701, 10, dtype=float)
AXIS_WORDS = {"DIFFUSE", "SPECTRAL", "DENSITY", "DYE"}
Y_BAND = (405, 480)
M_BAND = (510, 590)
C_BAND = (615, 700)

#: sheet tag -> (pdf filename relative to PDF/PROFILES/KODAK, page, profile)
SHEETS = {
    "5285": ("Ektachrome_100d.pdf", 4, "KODAK_EKTACHROME_100D_5285"),
    "2383": ("KODAK VISION Color Print Film 2383.pdf", 6, "KODAK_2383_RELEASE"),
    "5205": ("5205t.pdf", 4, "KODAK_VISION2_250D_5205"),
    "5219": ("KODAK-VISION3-500T-5219-7219-brochure.pdf", 3,
             "KODAK_VISION3_500T_5219"),
    "5245": ("5245.pdf", 4, "EASTMAN_EXR_50D_5245"),
    "5274": ("5274.pdf", 4, "KODAK_VISION_200T_5274"),
    "5279": ("5279.pdf", 3, "KODAK_VISION_500T_5279"),
    "5293": ("5293.pdf", 4, "EASTMAN_EXR_200T_5293"),
    # ADDED 2026-08-18 (queue item E0b). This sheet was on the FAILED list from
    # 2026-08-18 until the LABEL_STACK_GAP fix in rot_labels() -- the source was
    # always vector and always fine; the anchor picked the wrong plot. Its dye
    # panel prints its own normalisation statement, "Normalized dyes to form a
    # visual neutral density of 1.0 for a viewing illuminant of 5400 K", and
    # LABELS each curve Yellow / Magenta / Cyan, so the peak-based assignment can
    # be checked against Kodak's own words: 440 / 550 / 670 nm respectively.
    "7239": ("Kodak Eastman EKTACHROME Film (Daylight) 7239.pdf", 3,
             "EASTMAN_EKTACHROME_7239"),
    # ALSO RECOVERED 2026-08-18 by the same two fixes (LABEL_STACK_GAP in
    # rot_labels, the least-squares tick fit in _fit_axis, and the stroke-width
    # filter change in extract). Both were on the FAILED list and neither needed
    # anything from the source. 5218's failure was the most instructive: with the
    # old two-point calibration its curves came out at 2.14-2.24 D inside a frame
    # whose axis stops at 1.8 D, which is impossible and was the clue that a tick
    # label had been misassigned.
    "5217": ("5217-Vision2-200T.pdf", 3, "KODAK_VISION2_200T_5217"),
    "5218": ("5218-Vision2-500T-H-1-5218t.pdf", 4, "KODAK_VISION2_500T_5218"),
    # ADDED 2026-08-25 (queue item C9), and it took a THIRD family plus an
    # ink-based reader -- see family C in the module docstring for why the
    # originally recorded cause (a family-classifier limitation) was wrong. The
    # panel is H-1-5201 p3, five traces, and it prints its own normalisation:
    # "NOTE: Cyan, Magenta, and Yellow Dye Curves are peak-normalized."
    "5201": ("Kodak VISION2 50D 5201.pdf", 3, "KODAK_VISION2_50D_5201"),
}

#: Recorded 2026-08-18. --assert fails if an extraction stops reproducing.
#: ⚠ ALL RESIDUALS RE-RECORDED 2026-08-18 when the calibration changed from a
#: two-point span to a least-squares fit (see _fit_axis). The extracted CURVES
#: barely moved -- peak wavelengths are identical on every sheet, and the two
#: calibrations agree to <= 0.015 D on the peak-normalised family, which is a
#: useful error bar to have measured. The change was ADOPTED rather than reverted
#: because of an independent check: on the two sheets whose arrays were adopted
#: by a SEPARATE earlier extraction (2383 and 5285), the new fit reproduces the
#: stored arrays to RMS 0.0005 and 0.0003 D against the old fit's 0.0185 and
#: 0.0029 -- 10x to 40x closer to numbers it did not derive.
#: The stored peak_1.0 arrays were NOT re-adopted: they came from the old
#: calibration, the independent |max - 1.0| test is a wash between the two
#: methods (new better on 3 sheets, old better on 3, all within 0.003), and
#: re-adopting on a wash would be churn dressed as progress.
EXPECTED = {
    # 5201's residual is the family-C identity rms (neutral - dmin against the
    # three dyes), not a sum-identity max or a unit-peak spread, so it is not
    # comparable with the other rows -- the mode column says which test ran.
    "5201": ("peak_1.0", 0.0188),
    "5285": ("as_printed_plus_neutral", 0.0132),
    "2383": ("as_printed_plus_neutral", 0.1287),
    "5205": ("peak_1.0", 0.0092),
    "5217": ("peak_1.0", 0.0097),
    "5218": ("peak_1.0", 0.0195),
    "5219": ("peak_1.0", 0.0090),
    "5245": ("peak_1.0", 0.0034),
    "5274": ("peak_1.0", 0.0097),
    "5279": ("peak_1.0", 0.0162),
    "5293": ("peak_1.0", 0.0011),
    "7239": ("as_printed_plus_neutral", 0.0398),
}


def flatten(items, n=24):
    """Path vertices with cubic beziers evaluated, in order."""
    pts=[]
    for it in items:
        if it[0]=="l":
            pts.append((it[1].x,it[1].y)); pts.append((it[2].x,it[2].y))
        elif it[0]=="c":
            p0,p1,p2,p3=it[1],it[2],it[3],it[4]
            for k in range(n+1):
                t=k/n; u=1-t
                x=u**3*p0.x+3*u*u*t*p1.x+3*u*t*t*p2.x+t**3*p3.x
                y=u**3*p0.y+3*u*u*t*p1.y+3*u*t*t*p2.y+t**3*p3.y
                pts.append((x,y))
        elif it[0]=="re":
            r=it[1]; pts += [(r.x0,r.y0),(r.x1,r.y1)]
    return pts

def resample(pts, cal, grid):
    """pts in page coords -> density on the wavelength grid."""
    x0,l0,x1,l1,y0,d0,y1,d1 = cal
    lam=[(p[0]-x0)/(x1-x0)*(l1-l0)+l0 for p in pts]
    den=[(p[1]-y0)/(y1-y0)*(d1-d0)+d0 for p in pts]
    o=np.argsort(lam); lam=np.array(lam)[o]; den=np.array(den)[o]
    keep=np.concatenate(([True], np.diff(lam)>1e-9))
    return np.interp(grid, lam[keep], den[keep])

#: Vertical gap, in points, above which two rotated axis words in the SAME
#: column belong to DIFFERENT plots. Kodak stacks two or three plots in one
#: column and gives each a rotated y-axis caption, so a pure x-centre grouping
#: merges them. Set from the measured geometry: within one caption the words sit
#: 0-3 pt apart ("DIFFUSE"/"SPECTRAL"/"DENSITY" are contiguous), while the gap
#: to the next plot's caption is 180 pt on the 7239 sheet. 24 pt is far outside
#: the first and far inside the second.
LABEL_STACK_GAP = 24.0


def rot_labels(pg):
    """Rotated y-axis captions containing SPECTRAL + DENSITY, one per plot.

    ⚠ WHY THE Y-GAP SPLIT EXISTS -- this was the bug that made the 7239 sheet
    "fail" and it was never a problem with the source. Grouping the rotated words
    by x-centre ALONE merges every caption in a column into one band: on
    Kodak's H-1-5239 p3 that produced a single pseudo-label spanning y 127-440,
    made of the SPECTRAL SENSITIVITY plot's "DENSITY" plus the DIFFUSE SPECTRAL
    DENSITY plot's three words. `pick()` then chose the frame nearest the top of
    that band -- the SENSITIVITY plot -- and the dye-density curves were never
    looked at. The sheet was recorded as a failed extraction for a fortnight on
    the strength of it.
    """
    rot=[]
    for x0,y0,x1,y1,t,*_ in pg.get_text("words"):
        if (y1-y0)>1.6*(x1-x0) and t.upper().strip(",.:") in AXIS_WORDS:
            rot.append((x0,y0,x1,y1,t.upper()))
    g={}
    for x0,y0,x1,y1,t in rot: g.setdefault(round((x0+x1)/2/6)*6,[]).append((x0,y0,x1,y1,t))
    out=[]
    for cx,it in g.items():
        # split the column into runs of vertically adjacent words = one caption
        runs=[]
        for w in sorted(it, key=lambda w: w[1]):
            if runs and w[1] - max(v[3] for v in runs[-1]) <= LABEL_STACK_GAP:
                runs[-1].append(w)
            else:
                runs.append([w])
        for run in runs:
            words={t for *_,t in run}
            if {"SPECTRAL","DENSITY"} <= words:
                out.append((max(i[2] for i in run), min(i[1] for i in run),
                            max(i[3] for i in run), " ".join(sorted(words))))
    return out

def frames(pg):
    """Plot frames: wide, tall paths. The frame is the window every tick label
    must fall against -- using it instead of a guessed pixel window is what
    makes this work on multi-plot pages."""
    fr=[]
    for p in pg.get_drawings():
        r=p["rect"]
        if r.width>90 and r.height>55 and r.width<560 and r.height<420:
            fr.append(r)
    return fr

def pick(pg, ax):
    lx, ly0, ly1, _ = ax
    best=None
    for r in frames(pg):
        if r.x0 < lx-2: continue                       # frame must be right of label
        if r.y1 < ly0-30 or r.y0 > ly1+30: continue    # and vertically aligned
        d=r.x0-lx
        if best is None or d<best[0]: best=(d,r)
    return None if best is None else best[1]

def ticks(pg, fr):
    xs={}; ys={}
    for a,b,c,d,t,*_ in pg.get_text("words"):
        if not re.fullmatch(r'-?\d+(\.\d+)?', t): continue
        v=float(t); cx=(a+c)/2; cy=(b+d)/2
        if fr.x0-6<=cx<=fr.x1+6 and fr.y1-2<=cy<=fr.y1+22 and 300<=v<=800:
            xs.setdefault(v,cx)
        if fr.x0-40<=cx<fr.x0-1 and fr.y0-8<=cy<=fr.y1+8 and 0<=v<=5:
            ys.setdefault(v,cy)
    return xs, ys

#: An axis calibration is only trusted if EVERY harvested tick sits this close
#: to the fitted line. Both axes on these sheets are exactly linear, so a real
#: residual is a fraction of a point; anything larger means a label was assigned
#: to the wrong axis or the wrong plot.
TICK_RESID_PT = 1.5


def _fit_axis(pairs):
    """value -> pixel dict  ->  (slope, intercept, worst residual, n_used).

    ⚠ WHY THIS IS A FIT AND NOT A TWO-POINT SPAN. The original code calibrated
    from the LOWEST and HIGHEST tick only. That is exact when every label is
    right and silently wrong when one is not, because two points always define a
    line and nothing checks it. Measured consequence on the 5218 sheet: the
    two-point span produced curve maxima of 2.14-2.24 D inside a frame whose own
    axis topped out at 1.8 D -- physically impossible, since the curves are
    clipped to the frame, and therefore proof that a tick had been misread. A
    least-squares fit over all ticks plus a residual test catches that instead of
    propagating it into an adopted density.
    """
    import numpy as _np
    v=_np.array(sorted(pairs), dtype=float)
    px=_np.array([pairs[k] for k in sorted(pairs)], dtype=float)
    if len(v) < 3:
        return None
    # iteratively drop the worst outlier while it exceeds the tolerance and at
    # least three ticks would remain -- one stray label must not veto a good axis
    keep=_np.ones(len(v), bool)
    while keep.sum() >= 3:
        A=_np.vstack([v[keep], _np.ones(keep.sum())]).T
        m,c = _np.linalg.lstsq(A, px[keep], rcond=None)[0]
        res=_np.abs(m*v + c - px)
        worst=int(_np.argmax(_np.where(keep, res, -1.0)))
        if res[worst] <= TICK_RESID_PT or keep.sum() == 3:
            return m, c, float(res[keep].max()), int(keep.sum())
        keep[worst]=False
    return None


def extract(pg, ax, grid):
    fr=pick(pg,ax)
    if fr is None: return None,"no frame right of the axis label"
    xs,ys=ticks(pg,fr)
    if len(xs)<3: return None,f"only {len(xs)} x ticks against the frame"
    if len(ys)<3: return None,f"only {len(ys)} y ticks against the frame"
    fx=_fit_axis(xs); fy=_fit_axis(ys)
    if fx is None or fy is None:
        return None,"axis fit failed"
    if fx[2] > TICK_RESID_PT:
        return None,f"x ticks not collinear ({fx[2]:.2f} pt worst residual)"
    if fy[2] > TICK_RESID_PT:
        return None,f"y ticks not collinear ({fy[2]:.2f} pt worst residual)"
    # cal keeps the two-point form the resampler expects, but the two points are
    # now taken FROM THE FIT rather than from two possibly-bad labels.
    vx=sorted(xs); vy=sorted(ys)
    cal=(fx[0]*vx[0]+fx[1], vx[0], fx[0]*vx[-1]+fx[1], vx[-1],
         fy[0]*vy[0]+fy[1], vy[0], fy[0]*vy[-1]+fy[1], vy[-1])
    inside=[p for p in pg.get_drawings()
            if p["rect"].x0>=fr.x0-6 and p["rect"].x1<=fr.x1+6
            and p["rect"].y0>=fr.y0-6 and p["rect"].y1<=fr.y1+6]
    if not inside: return None,"no paths inside the frame"
    # ⚠ STROKE-WIDTH FILTER RELAXED 2026-08-18. It was
    # `width < 0.6*max(width inside)`, which is a filter against the THICKEST
    # thing in the frame -- and on the 5248 sheet the thickest thing is a rule,
    # not a curve, so four of the six dye traces were discarded and only two
    # survived. The MEDIAN width of the long paths is the right reference: the
    # curves are the population, the rule is the outlier. Non-curves that get
    # through are rejected downstream on physics (a gridline is flat, so it fails
    # both the sum identity and the "must fall away from its peak" test), which
    # is a safer place to reject them than on stroke weight.
    import statistics as _st
    longw=[(p.get("width") or 0.0) for p in inside
           if sum(1 for it in p["items"] if it[0] in ("l","c")) >= 8]
    ref=_st.median(longw) if longw else 0.0
    cs=[]
    for p in inside:
        n=sum(1 for it in p["items"] if it[0] in ("l","c"))
        if n<8 or (p.get("width") or 0) < 0.4*ref: continue
        y=resample(flatten(p["items"]),cal,grid)
        if np.isfinite(y).all(): cs.append(y)
    return (cal,fr,vx,vy,cs), None

# Kodak does not plot this the same way on every sheet, and the schema already
# anticipates it -- SpectralDyeDensity.normalisation lists "peak_1.0" beside the
# as-printed forms. Two families exist in this corpus:
#
#   A. AS-PRINTED + NEUTRAL (5285, 2383): four traces, and the visual-neutral
#      trace equals the sum of the three dyes. The sum identity validated the
#      5285 extraction and is used here to identify the quartet.
#      NOTE it is NOT exact on print stock: 2383's own provenance records the
#      neutral closing to 0.128 D because of the base absorber, so the
#      tolerance has to be loose enough to admit that.
#   B. PEAK-NORMALISED (the VISION2/VISION3 sheets): three traces, each scaled
#      to unit peak, no neutral at all. The sum identity CANNOT hold and must
#      not be required -- requiring it is what made a first pass reject every
#      sheet but 5285.
#
# Peak wavelengths are the physical anchor in both cases: yellow absorbs blue
# (~430-460), magenta green (~530-570), cyan red (~640-700).

def _bands_ok(lams):
    return (Y_BAND[0]<=lams[0]<=Y_BAND[1] and M_BAND[0]<=lams[1]<=M_BAND[1]
            and C_BAND[0]<=lams[2]<=C_BAND[1])

def _peaks(curves, idxs, grid):
    return sorted((float(grid[curves[i].argmax()]), i) for i in idxs)

def pick_dye_set(curves, grid, tol_sum=0.14, tol_peak=0.04):
    """Return (cyan, magenta, yellow, neutral_or_None, mode, residual)."""
    n=len(curves)
    best=None
    # --- family A: quartet with neutral = C+M+Y ---------------------------
    for quad in itertools.combinations(range(n),4):
        for ni in quad:
            rest=[i for i in quad if i!=ni]
            res=float(np.abs(curves[ni]-sum(curves[i] for i in rest)).max())
            if res>tol_sum: continue
            pk=_peaks(curves,rest,grid)
            if not _bands_ok([p[0] for p in pk]): continue
            if not (0.25 <= max(curves[i].max() for i in quad) <= 4.0): continue
            cand=(res,"as_printed_plus_neutral",curves[pk[2][1]],
                  curves[pk[1][1]],curves[pk[0][1]],curves[ni])
            if best is None or res<best[0]: best=cand
    if best: 
        r,mode,c,m,y,neu=best; return c,m,y,neu,mode,r
    # --- family B: three traces each normalised to unit peak --------------
    for tri in itertools.combinations(range(n),3):
        mx=[float(curves[i].max()) for i in tri]
        if max(abs(v-1.0) for v in mx) > tol_peak: continue
        pk=_peaks(curves,tri,grid)
        if not _bands_ok([p[0] for p in pk]): continue
        # each trace must actually fall away from its peak: a flat line at 1.0
        # (a gridline or an axis) would otherwise qualify
        ok=True
        for _,i in pk:
            v=curves[i]
            if v.min() > 0.5: ok=False; break
        if not ok: continue
        spread=max(abs(v-1.0) for v in mx)
        cand=(spread,"peak_1.0",curves[pk[2][1]],curves[pk[1][1]],curves[pk[0][1]],None)
        if best is None or spread<best[0]: best=cand
    if not best: return None
    r,mode,c,m,y,neu=best
    return c,m,y,neu,mode,r

# ---------------------------------------------------------------------------
# FAMILY C: identify the traces by INK, not by segment count (queue item C9).
#
# Kodak's H-1 brochures draw every plot on this page in a fixed four-ink
# palette, and the rule behind it is physical rather than decorative: EACH TRACE
# IS DRAWN IN THE COLOUR OF THE LIGHT IT CONCERNS. On the dye panel the yellow
# dye -- which absorbs blue -- is drawn in BLUE ink; the magenta dye, which
# absorbs green, in GREEN ink; the cyan dye, which absorbs red, in RED, and red
# is not one of the four inks, so Kodak makes it by overprinting YELLOW UNDER
# MAGENTA. That last detail is the whole reason this sheet failed for weeks.
#
# The mapping was read off the panel's own legend swatches -- 1 pt horizontal
# lines at the left of each legend line -- and then confirmed twice over:
#   * the green swatch sits on "Magenta Dye" and the amber one on "Cyan Dye",
#     which is the complementary rule stated outright by the sheet;
#   * the resulting traces peak at 450 / 540 / 680 nm, i.e. each in its own
#     absorption band, so the ink assignment and the physics agree.
# The neutral and dmin traces are the two DARK ones, told apart by their dash
# pattern: Midscale Neutral solid, Minimum Density dashed (drawn as a solid line
# with white rectangles punched over it, which is why `dashes` carries the
# pattern on the path itself).
INK_BLUE = (0.0, 0.455, 0.737)
INK_GREEN = (0.0, 0.459, 0.204)
INK_YELLOW = (0.969, 0.765, 0.141)
INK_MAGENTA = (0.925, 0.0, 0.549)
INK_DARK = (0.137, 0.122, 0.125)
#: Per-channel tolerance for matching a stroke colour to the palette above.
#: The inks are process primaries and come out of the PDF bit-identical across
#: paths and across panels; 0.02 is slack for a re-exported sheet, not a fudge.
INK_TOL = 0.02


def _ink(path) -> str | None:
    """Palette name of a stroke colour, or None if it is not in the palette."""
    col = path.get("color")
    if not col:
        return None
    for name, ref in (("blue", INK_BLUE), ("green", INK_GREEN),
                      ("yellow", INK_YELLOW), ("magenta", INK_MAGENTA),
                      ("dark", INK_DARK)):
        if all(abs(a - b) <= INK_TOL for a, b in zip(col, ref)):
            return name
    return None


def _is_dashed(path) -> bool:
    d = (path.get("dashes") or "").strip()
    return bool(d) and not d.startswith("[]")


#: Minimum segments for an INKED trace. Much lower than `extract`'s 8, and safe
#: only because the ink already identified the path: the cyan trace really is 7
#: segments. Applying 4 in `extract` would admit gridline stubs on every sheet.
INK_MIN_SEG = 4


def extract_inked(pg, cal, fr, grid):
    """{palette name: [curve, ...]} for the paths inside one plot frame.

    Overprinted duplicates are collapsed: Kodak's red is a yellow path and a
    magenta path with IDENTICAL geometry, so the pair is one curve, not two. The
    collapse is asserted rather than assumed -- if the two ever stop coinciding
    they are different traces and must not be merged.
    """
    out: dict[str, list] = {}
    for p in pg.get_drawings():
        r = p["rect"]
        if not (r.x0 >= fr.x0 - 6 and r.x1 <= fr.x1 + 6
                and r.y0 >= fr.y0 - 6 and r.y1 <= fr.y1 + 6):
            continue
        name = _ink(p)
        if name is None:
            continue
        if sum(1 for it in p["items"] if it[0] in ("l", "c")) < INK_MIN_SEG:
            continue
        # the plot frame itself is a dark path spanning the whole rect
        if (r.width > 0.98 * fr.width and r.height > 0.98 * fr.height):
            continue
        y = resample(flatten(p["items"]), cal, grid)
        if not np.isfinite(y).all():
            continue
        key = name + ("_dashed" if _is_dashed(p) else "")
        out.setdefault(key, []).append(y)
    return out


#: The three unit-peak dyes must each reach 1.0 this closely, and the neutral
#: identity below must close this well. Both are measured on the 5201 sheet
#: (0.005 and 0.019) with room to spare, not fitted to it.
FAMILY_C_PEAK_TOL = 0.04
FAMILY_C_IDENTITY_RMS = 0.05


def pick_dye_set_inked(inked, grid):
    """Family C: peak_1.0 dyes + as-printed neutral + as-printed dmin.

    Returns (cyan, magenta, yellow, None, "peak_1.0", rms) or None. The neutral
    is NOT returned: it is as-printed while the dyes are not, and one record
    cannot carry both conventions.

    THE VALIDATOR IS THE POINT OF THIS FAMILY. Family A identifies its quartet
    by neutral = C+M+Y, which cannot hold when the dyes are peak-normalised and
    the neutral is not. The generalisation that DOES hold, and which nothing here
    is fitted to produce, is

        Neutral - Dmin  =  k_c*C + k_m*M + k_y*Y

    with the three coefficients EQUAL, because that is what makes the result a
    visual NEUTRAL. On 5201 the unconstrained least-squares solution comes out
    0.628 / 0.604 / 0.595 -- a 5 % spread on numbers that were free to be
    anything -- at rms 0.019 D. Dropping the Dmin term makes the fit 4.5x worse
    (rms 0.085) and scatters the coefficients over 0.86-1.61, which is what
    identifies which dark trace is which.
    """
    dyes = {}
    for key, want in (("blue", Y_BAND), ("green", M_BAND)):
        for c in inked.get(key, []):
            if abs(float(c.max()) - 1.0) > FAMILY_C_PEAK_TOL:
                continue
            lam = float(grid[c.argmax()])
            if want[0] <= lam <= want[1]:
                dyes[key] = c
    # red = the overprinted pair; either member will do once they are known equal
    reds = []
    for key in ("yellow", "magenta"):
        reds += inked.get(key, [])
    for c in reds:
        if abs(float(c.max()) - 1.0) > FAMILY_C_PEAK_TOL:
            continue
        if C_BAND[0] <= float(grid[c.argmax()]) <= C_BAND[1]:
            dyes["red"] = c
    if len(dyes) != 3:
        return None
    if len(reds) == 2 and float(np.abs(reds[0] - reds[1]).max()) > 1e-9:
        return None     # not an overprint after all -- two different traces
    neutrals = inked.get("dark", [])
    dmins = inked.get("dark_dashed", [])
    if len(neutrals) != 1 or len(dmins) != 1:
        return None
    y, m, c = dyes["blue"], dyes["green"], dyes["red"]
    A = np.vstack([c, m, y]).T
    b = neutrals[0] - dmins[0]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    rms = float(np.sqrt(((A @ coef - b) ** 2).mean()))
    if rms > FAMILY_C_IDENTITY_RMS:
        return None
    if coef.min() <= 0.0 or (coef.max() - coef.min()) / coef.mean() > 0.15:
        return None     # not a neutral: the three dyes do not contribute equally
    return c, m, y, None, "peak_1.0", rms


def extract_sheet(root: Path, tag: str):
    import pymupdf
    fn, pgno, prof = SHEETS[tag]
    pdf = root / "PDF" / "PROFILES" / "KODAK" / fn
    if not pdf.is_file():
        return None, f"source not present: {fn}"
    pg = pymupdf.open(pdf)[pgno - 1]
    for ax in rot_labels(pg):
        r, err = extract(pg, ax, GRID)
        if not r:
            continue
        sel = pick_dye_set(r[4], GRID)
        if sel is None:
            # FAMILY C, tried only after A and B have failed on this frame, so
            # the 11 sheets that already work never reach it and cannot be
            # disturbed by it. --assert proves that claim on every run.
            sel = pick_dye_set_inked(extract_inked(pg, r[0], r[1], GRID), GRID)
        if sel:
            c, m, y, neu, mode, res = sel
            return dict(tag=tag, profile=prof, file=fn, page=pgno, mode=mode,
                        residual=res, cyan=c, magenta=m, yellow=y,
                        neutral=neu), None
    return None, "no curve set matched any of the three normalisation families"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--sheet", action="append", choices=sorted(SHEETS))
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()
    tags = ns.sheet or sorted(SHEETS)
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
        want_mode, want_res = EXPECTED[tag]
        ok = got["mode"] == want_mode and abs(got["residual"] - want_res) < 0.004
        print(f"  [{'OK  ' if ok else 'FAIL'}] {tag} {got['profile']:28s} "
              f"{got['mode']:24s} res={got['residual']:.4f} "
              f"y{int(GRID[got['yellow'].argmax()])} "
              f"m{int(GRID[got['magenta'].argmax()])} "
              f"c{int(GRID[got['cyan'].argmax()])}")
        if not ok:
            print(f"         expected {want_mode} res={want_res:.4f}")
            bad += 1
    print(f"\n[i] {len(tags) - bad - skipped} reproduced, {bad} failed, "
          f"{skipped} skipped")
    if ns.do_assert and bad:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
