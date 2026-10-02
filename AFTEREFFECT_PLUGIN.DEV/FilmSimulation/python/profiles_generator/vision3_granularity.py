"""Reproduce the adopted VISION3 sigma(D) shapes from the four Kodak TI sheets.

Run this to re-derive, from the PDFs, every number that went into the four
`sigma_shape_toe/mid/dmax` triples in film_profiles.py. It is the audit trail
for that adoption: if a future pass changes `dashtrace.py` and this script stops
reproducing the table at the bottom, the change broke the extraction.

    python vision3_granularity.py --pdfdir ../../PDF/PROFILES/KODAK [--overlay out]

Why this file exists rather than a notebook: the numbers are [T1] and the queue's
method rule 13 requires every adoption to be re-runnable. Requires PyMuPDF only
to pull the page-3 raster out of the PDF; everything after that is numpy, and the
tracing itself is `dashtrace.family_split_by_style` + `dashtrace.trace_predictive`
+ `dashtrace.check_cross_family`, i.e. the shipped tool, not a private copy.

METHOD, in the order it must be done (see dashtrace.py's STATUS block for why):
  1. the plot is RASTER on all four sheets -- get_drawings() gives 2 paths and
     none with >= 30 items -- so the embedded image is used at its NATIVE
     resolution. Re-rendering the page at 600 dpi only interpolates it;
  2. split the two curve families by STROKE STYLE before tracing anything;
  3. trace the granularity family inside its own mask, seeded at the right where
     the sheet's own B/G/R labels fix identity;
  4. trace the density family in the complementary mask, seeded at the LEFT-EDGE
     dmin plateau, RIGHTWARD ONLY -- the crossings are decidable in that
     direction only;
  5. assert cross-family separation, and WRITE THE OVERLAY. The overlay is the
     first gate, not the last: three previous passes produced internally
     consistent numbers from hybrid curves.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import dashtrace as dt

# --- per-sheet geometry, measured; see doc/RESULT_2026-08-17f_*.md -----------
# tag: (pdf glob, frame left_inner, right_inner, top, bottom, style, n_gran)
SHEETS = {
    '5203': ('*VISION3-50D-5203*.pdf',  64, 505, 51, 493, 'bold', 3),
    '5207': ('*VISION3-250D-5207*.pdf', 63, 505, 62, 505, 'dash', 3),
    '5213': ('*VISION3-200T-5213*.pdf', 70, 505, 41, 476, 'bold', 1),
    '5219': ('*5219*Technical*.pdf',    61, 507, 60, 506, 'dash', 3),
}
# y pixel of sigma = 0.001 (the frame bottom) and px per decade, both read off
# the tick comb outside the right frame line. The px/decade figure MUST be
# measured: ratios are invariant to the axis offset but not to its scale.
LOGCAL = {'5203': (492.5, 139.00), '5207': (505.5, 139.75),
          '5213': (475.0, 139.00), '5219': (506.5, 140.25)}
# in-plot legend/caption rectangles to blank (x0, x1, y0, y1); 5219 boxes a
# six-line legend that no curve passes through.
MASKS = {'5219': [(80, 275, 90, 240)]}
# The [T1] H&D dmin ladders already in film_profiles.py, r/g/b, for the identity
# check. NOT used to seed anything -- used to verify what the trace found.
HD_DMIN = {'5203': (0.1341, 0.5688, 0.8434), '5207': (0.1539, 0.5708, 0.8392),
           '5213': (0.1681, 0.5813, 0.8510), '5219': (0.1867, 0.5811, 0.8374)}
DARK = 0.50          # ink threshold; stable over [0.40, 0.50], breaks at 0.60


def load_plot(pdfdir: str, tag: str) -> np.ndarray:
    """Greyscale [0..1] of the page-3 granularity raster, native resolution."""
    import pymupdf
    from PIL import Image
    hits = glob.glob(os.path.join(pdfdir, SHEETS[tag][0]))
    if not hits:
        raise FileNotFoundError(f'{tag}: no PDF matching {SHEETS[tag][0]}')
    doc = pymupdf.open(hits[0])
    page = doc[2]
    vector = [g for g in page.get_drawings() if len(g['items']) >= 30]
    if vector:
        raise RuntimeError(f'{tag}: page 3 now has {len(vector)} large vector '
                           f'paths -- re-check, vector beats any trace')
    best = None
    for im in page.get_images(full=True):
        pix = pymupdf.Pixmap(doc, im[0])
        rect = page.get_image_rects(im[0])[0]
        if rect.y0 < 200 and rect.x0 > 300:      # upper-right panel = granularity
            best = pix
    if best is None:
        raise RuntimeError(f'{tag}: granularity panel not found on page 3')
    import io
    img = Image.open(io.BytesIO(best.tobytes('png'))).convert('L')
    return np.asarray(img, dtype=np.float64) / 255.0


def density(tag, y):
    _g, _l, _r, t, b, _s, _n = SHEETS[tag]
    return 3.0 * (b - y) / (b - t)


def sigma(tag, y):
    y0, per = LOGCAL[tag]
    return 0.001 * 10.0 ** ((y0 - y) / per)


def log_e(tag, x):
    _g, left, right, _t, _b, _s, _n = SHEETS[tag]
    return 5.0 * (x - left) / (right - left)


def trace_sheet(tag, gray):
    """Returns (density_tracks, granularity_tracks, gran_ink, dens_ink)."""
    _g, left, right, top, bot, style, ngran = SHEETS[tag]
    ink = gray < DARK
    box = np.zeros_like(ink)
    box[top + 3:bot - 3, left + 3:right - 1] = ink[top + 3:bot - 3, left + 3:right - 1]
    for (x0, x1, y0, y1) in MASKS.get(tag, []):
        box[y0:y1 + 1, x0:x1 + 1] = False

    gran_ink, dens_ink = dt.family_split_by_style(box, style)

    # -- granularity: seeded at the right, where the printed B/G/R labels sit --
    if ngran == 3:
        gx = gseed = None
        for x in range(right - 6, left + 20, -1):
            cs = sorted(c for c, _t in dt.column_runs_weighted(gran_ink, gray, x, top, bot))
            if len(cs) == 3 and min(np.diff(cs)) >= 5:
                gx, gseed = x, cs
                break
        if gx is None:
            raise RuntimeError(f'{tag}: no column with three separated granularity runs')
        names = ['Gb', 'Gg', 'Gr']
        fwd = dt.trace_predictive(gran_ink, gray, (left + 3, right - 2), top, bot,
                                  gx, dict(zip(names, gseed)), direction=+1)
        rev = dt.trace_predictive(gran_ink, gray, (left + 3, right - 2), top, bot,
                                  gx, dict(zip(names, gseed)), direction=-1)
        gran = {k: {**fwd[k], **rev[k]} for k in names}
    else:
        # 5213 draws its three granularity curves as one overlapping bold band.
        # Per-layer separation is not available; take the band centre and carry
        # its half-width as the uncertainty.
        names = ['Gpool']
        gran = {'Gpool': {}}
        for x in range(left + 4, right - 1):
            ys = np.nonzero(gran_ink[:, x])[0]
            if ys.size:
                gran['Gpool'][x] = 0.5 * (ys.min() + ys.max())
        dens_ink = dens_ink.copy()
        for x, _y in gran['Gpool'].items():
            ys = np.nonzero(gran_ink[:, x])[0]
            dens_ink[max(0, ys.min() - 3):ys.max() + 4, x] = False

    # -- density: LEFT-EDGE seed, RIGHTWARD only (the direction rule) ---------
    dx = dseed = None
    for x in range(left + 9, left + int(0.30 * (right - left))):
        cs = sorted(c for c, _t in dt.column_runs_weighted(dens_ink, gray, x, top, bot))
        if len(cs) == 3 and min(np.diff(cs)) > 15:
            dx, dseed = x, cs
            break
    if dx is None:
        raise RuntimeError(f'{tag}: no left-edge density seed (three separated runs)')
    dnames = ['Db', 'Dg', 'Dr']
    dens = dt.trace_predictive(dens_ink, gray, (left + 3, right - 2), top, bot,
                               dx, dict(zip(dnames, dseed)), direction=+1,
                               tol0=2.4, tol_grow=0.35, max_bridge=30, hist=14,
                               slope_cap=2.2)
    plateau = dt.trace_predictive(dens_ink, gray, (left + 3, right - 2), top, bot,
                                  dx, dict(zip(dnames, dseed)), direction=-1,
                                  tol0=2.0, tol_grow=0.2, max_bridge=6, hist=8,
                                  slope_cap=0.3)
    dens = {k: {**plateau[k], **dens[k]} for k in dnames}

    # 5203's green density curve is overdrawn by the bold granularity curve for
    # 0.6 of a decade, wider than any bridge, so the left-seeded trace stops at
    # the plateau. Fill the part ABOVE the gap from a right-seeded pass -- which
    # is allowed to be wrong below the crossing and demonstrably is, so it is
    # accepted ONLY at densities the left-seeded branch already exceeded. Nothing
    # is interpolated across the gap itself; it stays empty and is declared.
    dxr = dsr = None
    for x in range(right - 3, right - 60, -1):
        cs = sorted(c for c, _t in dt.column_runs_weighted(dens_ink, gray, x, top, bot)
                    if density(tag, c) > 1.5)
        if len(cs) == 3 and min(np.diff(cs)) > 20:
            dxr, dsr = x, cs
            break
    if dxr is not None:
        rev_d = dt.trace_predictive(dens_ink, gray, (left + 3, right - 2), top, bot,
                                    dxr, dict(zip(dnames, dsr)), direction=-1,
                                    tol0=2.4, tol_grow=0.35, max_bridge=30,
                                    hist=14, slope_cap=2.2)
        for k in dnames:
            if not dens[k]:
                continue
            floor = max(density(tag, v) for v in dens[k].values()) + 0.05
            for x, y in rev_d[k].items():
                if x not in dens[k] and density(tag, y) > floor:
                    dens[k][x] = y

    # granularity ends that the style mask lost (where a granularity curve is
    # overdrawn by a density curve) are recovered on the density-free remainder
    keep = gran_ink | (box & ~_corridor(box, dens, pad=3))
    for k in names:
        tr = gran[k]
        if not tr:
            continue
        solo = keep & ~_corridor(keep, {kk: vv for kk, vv in gran.items() if kk != k}, pad=2)
        for end, direction in ((min(tr), -1), (max(tr), +1)):
            sub = dt.trace_predictive(solo, gray, (left + 3, right - 2), top, bot,
                                      end, {k: tr[end]}, direction=direction,
                                      tol0=2.5, tol_grow=0.5, max_bridge=20,
                                      hist=16, slope_cap=1.2)
            for x, y in sub[k].items():
                tr.setdefault(x, y)
    return dens, gran, gran_ink, dens_ink


def _corridor(ink, tracks, pad):
    m = np.zeros_like(ink)
    for tr in tracks.values():
        for x, y in tr.items():
            y0 = int(round(y))
            m[max(0, y0 - pad):y0 + pad + 1, x] = True
    return m


def anchors(tag, dens, gran, layer='Dg'):
    """sigma at dmin, D = 1.0 and the highest jointly covered D, plus the peak."""
    gk = {'Db': 'Gb', 'Dg': 'Gg', 'Dr': 'Gr'}[layer]
    if gk not in gran:
        gk = 'Gpool'
    tr = dens[layer]
    xs = sorted(tr)
    dv = [density(tag, tr[x]) for x in xs]
    keep = [0]
    for i in range(1, len(xs)):          # keep the monotone-consistent subset
        if dv[i] >= dv[keep[-1]] - 0.02:
            keep.append(i)
    xs = [xs[i] for i in keep]
    lo = np.array([log_e(tag, x) for x in xs])
    dd = np.array([density(tag, tr[x]) for x in xs])
    gxs = sorted(gran[gk])
    glo = np.array([log_e(tag, x) for x in gxs])
    gs = np.array([sigma(tag, gran[gk][x]) for x in gxs])
    d_lo = float(dd.min())
    d_hi = float(np.interp(min(lo.max(), glo.max()), lo, dd))

    def at(D):
        return float(np.interp(float(np.interp(D, dd, lo)), glo, gs))

    band = (glo >= lo.min()) & (glo <= lo.max())
    ip = int(np.argmax(gs[band]))
    return dict(dmin=d_lo, dmax=d_hi, s_toe=at(d_lo), s_mid=at(1.0),
                s_dmax=at(d_hi), s_peak=float(gs[band][ip]),
                d_peak=float(np.interp(glo[band][ip], lo, dd)))


def overlay(gray, tracks, path):
    from PIL import Image
    rgb = np.stack([(gray * 255).astype(np.uint8)] * 3, axis=-1)
    cols = {'Db': (0, 80, 255), 'Dg': (0, 170, 0), 'Dr': (220, 0, 0),
            'Gb': (0, 200, 255), 'Gg': (160, 220, 0), 'Gr': (255, 120, 0),
            'Gpool': (255, 0, 200)}
    for k, tr in tracks.items():
        c = cols.get(k, (255, 0, 255))
        for x, y in tr.items():
            y0 = int(round(y))
            for dy in (-1, 0, 1):
                if 0 <= y0 + dy < rgb.shape[0]:
                    rgb[y0 + dy, x] = c
    Image.fromarray(rgb).resize((rgb.shape[1] * 2, rgb.shape[0] * 2),
                                Image.NEAREST).save(path)


# adopted 2026-08-17; this script must keep reproducing them
ADOPTED = {'5203': (0.39, 1.00, 0.63), '5207': (0.59, 1.00, 0.57),
           '5213': (0.41, 1.00, 0.58), '5219': (0.67, 1.00, 0.55)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pdfdir', default=os.path.join('..', '..', 'PDF', 'PROFILES', 'KODAK'))
    ap.add_argument('--overlay', default=None, help='directory for overlay PNGs')
    ns = ap.parse_args()
    if ns.overlay:
        os.makedirs(ns.overlay, exist_ok=True)

    print('sheet | dmin  dmax  | s_toe  s_mid  s_dmax | toe/mid dmax/mid | '
          'peak s @ D      | cross-family (viol/cols, worst px)')
    ok = True
    for tag in ('5203', '5207', '5213', '5219'):
        gray = load_plot(ns.pdfdir, tag)
        dens, gran, _gi, _di = trace_sheet(tag, gray)
        a = anchors(tag, dens, gran)
        toe, dmx = a['s_toe'] / a['s_mid'], a['s_dmax'] / a['s_mid']
        viol, shared, worst, at = dt.check_cross_family(dens, gran, min_margin=3.0)
        order_d = dt.check_ordering(dens, ['Db', 'Dg', 'Dr'])
        print(f"{tag}  | {a['dmin']:.3f} {a['dmax']:.3f} | "
              f"{a['s_toe']*1000:6.2f} {a['s_mid']*1000:6.2f} {a['s_dmax']*1000:6.2f} | "
              f"  {toe:.3f}   {dmx:.3f}  | "
              f"{a['s_peak']*1000:5.2f} @ {a['d_peak']:.2f} ({a['s_peak']/a['s_mid']:.2f}x) | "
              f"{viol}/{shared}, {worst:.1f} px; density order {order_d[0]}/{order_d[1]}")
        want = ADOPTED[tag]
        if abs(round(toe, 2) - want[0]) > 0.011 or abs(round(dmx, 2) - want[2]) > 0.011:
            print(f'   !! {tag} no longer reproduces the adopted triple {want}')
            ok = False
        if ns.overlay:
            overlay(gray, {**dens, **gran}, os.path.join(ns.overlay, f'ov_{tag}.png'))
    print()
    print('Reproduces the adopted triples.' if ok else 'MISMATCH -- see above.')
    print()
    print('HOW TO READ THE CROSS-FAMILY COLUMN, honestly. Zero violations does '
          'NOT mean the families never cross -- they demonstrably do, and the '
          'worst margins of 3.5-6.9 px are how close the TRACED points get. It '
          'means no column carries a traced point from both families within '
          '3 px, so no traced column is ambiguous. At the crossings themselves '
          'one side is usually uncovered (a dash gap, or the style mask), which '
          'is why they cannot be counted. The check is a regression guard on a '
          'known-good state, not a proof of correctness.')
    print('THE OVERLAY IS THE GATE. Run with --overlay and look at it before '
          'trusting any number here; three earlier passes produced internally '
          'consistent numbers from cross-family hybrid curves.')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
