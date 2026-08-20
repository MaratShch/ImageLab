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
  * Sheets not yet handled: 5246 and 5248 ONLY. Three of the original five
    failures -- 7239, 5217, 5218 -- were recovered on 2026-08-18 and NONE of
    them needed anything from the source; all three were defects in this script:
    a y-axis caption anchor that merged two stacked plots into one band, a
    two-point tick calibration with nothing checking it, and a stroke-width
    filter referenced to the thickest path in the frame rather than to the
    curves. See LABEL_STACK_GAP, _fit_axis and the note in extract().
    The two that remain, with their measured near-misses rather than a shrug:
      5246 p5 -- 9 curves inside the frame; the best peak_1.0 triple has maxima
        1.008 / 0.997 / 0.926, and the 0.926 is 0.074 off unit peak, well past
        the 0.04 tolerance. Widening the tolerance to admit it would also admit
        false sets, so it stays out until the extra traces are identified.
      5248 p3 -- only 2 curves survive inside the frame even after the width
        filter was relaxed, so the other traces are being lost before selection.

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
    # ⚠ 5201 IS DELIBERATELY NOT HERE, and this is the record of why. H-1-5201 p3
    # draws FIVE traces in its dye panel -- Midscale Neutral, Cyan, Magenta,
    # Yellow and Minimum Density -- and prints "Cyan, Magenta, and Yellow Dye
    # Curves are peak-normalized", i.e. a peak_1.0 dye set PLUS two as-printed
    # traces on one pair of axes. Registering it returns "no curve set matched
    # either normalisation family": the family classifier is built for 3 dye
    # traces or 3 dyes + neutral, not 3 dyes + neutral + dmin, and widening it
    # blind would put the 11 sheets above at risk. Queue item C9 does it properly.
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
        if sel:
            c, m, y, neu, mode, res = sel
            return dict(tag=tag, profile=prof, file=fn, page=pgno, mode=mode,
                        residual=res, cyan=c, magenta=m, yellow=y,
                        neutral=neu), None
    return None, "no curve set matched either normalisation family"


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
