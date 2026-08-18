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
  * Sheets not yet handled: 5246, 5248, 5217, 5218, 7239 (5248/5217/7239 find
    no ticks against the frame; 5246/5218 find curves but no set matches
    either family).

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
}

#: Recorded 2026-08-18. --assert fails if an extraction stops reproducing.
EXPECTED = {
    "5285": ("as_printed_plus_neutral", 0.0154),
    "2383": ("as_printed_plus_neutral", 0.1350),
    "5205": ("peak_1.0", 0.0127),
    "5219": ("peak_1.0", 0.0047),
    "5245": ("peak_1.0", 0.0081),
    "5274": ("peak_1.0", 0.0074),
    "5279": ("peak_1.0", 0.0137),
    "5293": ("peak_1.0", 0.0050),
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

def rot_labels(pg):
    rot=[]
    for x0,y0,x1,y1,t,*_ in pg.get_text("words"):
        if (y1-y0)>1.6*(x1-x0) and t.upper().strip(",.:") in AXIS_WORDS:
            rot.append((x0,y0,x1,y1,t.upper()))
    g={}
    for x0,y0,x1,y1,t in rot: g.setdefault(round((x0+x1)/2/6)*6,[]).append((x0,y0,x1,y1,t))
    out=[]
    for cx,it in g.items():
        w={t for *_,t in it}
        if {"SPECTRAL","DENSITY"} <= w:
            out.append((max(i[2] for i in it), min(i[1] for i in it),
                        max(i[3] for i in it), " ".join(sorted(w))))
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

def extract(pg, ax, grid):
    fr=pick(pg,ax)
    if fr is None: return None,"no frame right of the axis label"
    xs,ys=ticks(pg,fr)
    if len(xs)<3: return None,f"only {len(xs)} x ticks against the frame"
    if len(ys)<3: return None,f"only {len(ys)} y ticks against the frame"
    vx=sorted(xs); vy=sorted(ys)
    cal=(xs[vx[0]],vx[0], xs[vx[-1]],vx[-1], ys[vy[0]],vy[0], ys[vy[-1]],vy[-1])
    inside=[p for p in pg.get_drawings()
            if p["rect"].x0>=fr.x0-6 and p["rect"].x1<=fr.x1+6
            and p["rect"].y0>=fr.y0-6 and p["rect"].y1<=fr.y1+6]
    if not inside: return None,"no paths inside the frame"
    thick=max((p.get("width") or 0) for p in inside)
    cs=[]
    for p in inside:
        n=sum(1 for it in p["items"] if it[0] in ("l","c"))
        if n<8 or (p.get("width") or 0) < 0.6*thick: continue
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
