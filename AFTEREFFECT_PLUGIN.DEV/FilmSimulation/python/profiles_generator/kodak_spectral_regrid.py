#!/usr/bin/env python3
"""Queue P95 (2026-09-30): the 2026-08-16 Kodak still-film spectral sweep,
RE-READ AGAINST THE DRAWN GRIDLINES.

The 08-16 sweep calibrated each panel's axes to the printed TICK LABELS. On
E-2468 p5 (PORTRA 100T, re-read 2026-09-29b) the labels sit 0.3-1.5 nm and
~1 % of a decade off the drawn gridlines, which moved the stored set by up to
0.11 log on the steep flanks. This module re-reads the other panels of that
sweep the same way.

⚠ WHY A RASTER GRID FINDER. `kodak_2026_09_29b.spectral_grid` finds the grid
from vector STROKES. Most of these sheets draw the grid as FILLED hairline
paths, which that finder does not see (it finds 0 verticals on E-7023). The
grid is therefore located on a 600 dpi render of the panel -- every gridline is
a run of >50 % dark pixels across the frame -- and converted back to page
units (0.12 pt per pixel); the CURVES stay vector, read exactly from the PDF.
Each gridline takes the value of the printed label nearest to it, so the
labels only NAME the lines and never position them.

REFUSED: GOLD 100 / 200 (E-7022 p4). The layer labels «Yellow-Forming Layer»
etc. are set INSIDE the plot and cut every curve into pieces whose ends do
not meet, and the short-wave tails of the magenta and cyan records then peak
where the blue record does, so the peak rule cannot assign them. The 08-16
arrays stand; queue P95 records it.

Usage:  python kodak_spectral_regrid.py --root <project root> [--dump]
        (audit: the stored arrays must equal a fresh read, to the last place)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GRID = np.arange(380.0, 710.0, 10.0)          # the database grid, 33 points
#: stock -> (file, page); the 08-16 source edition's page
PANELS = {
    "KODAK_PORTRA_100T": ("e2468-Portra_100T.pdf", 5),
    "KODAK_ULTRAMAX_400": ("E7023-Ultra_Max_400.pdf", 4),
    "KODAK_ULTRAMAX_800": ("E7024-Ultra_Max_800.pdf", 3),
    "KODAK_EKTAR_100": ("e4046_ektar_100.pdf", 4),
    "KODAK_PORTRA_160": ("e4051_portra_160.pdf", 4),
    "KODAK_PORTRA_800": ("e4040_portra_800.pdf", 4),
    "KODAK_TRI_X_400TX": ("f4017_TriX.pdf", 7),
    "KODAK_TMAX_100": ("f4016_TMax_100.pdf", 8),
    "KODAK_TMAX_400": ("f4043_TMax_400.pdf", 7),
    # the 2019 edition: its panel is the one the 08-16 sweep read (the 2018
    # printing's curve differs by 0.2 log on average and is not this drawing)
    "KODAK_TMAX_P3200": ("f4001-P3200TMZ-2019.pdf", 7),
    "KODAK_PLUS_X_125": ("f4018-125PX-2007.pdf", 9),
    "KODAK_T400CN": ("f2350-T400CN.pdf", 6),
    "KODAK_BW400CN": ("f4036-BW400CN.pdf", 5),
}
NUM = re.compile(r"^-?\d+(\.\d+)?$")
#: a log-sensitivity axis label is always printed with one decimal («2.0»)
YNUM = re.compile(r"^-?\d\.\d$")


def _clusters(idx):
    out, cur = [], []
    for i in idx:
        if cur and i > cur[-1] + 1:
            out.append(cur); cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return [0.5 * (c[0] + c[-1]) for c in out]


def _centroids(dark_frac, ink, s, origin, lo, hi):
    """Darkness-weighted centre of every line run, in page units; runs within
    1 pt of the frame edges are the FRAME, not grid, and are dropped."""
    out = []
    idx = np.where(dark_frac > 0.5)[0]
    runs, cur = [], []
    for i in idx:
        if cur and i > cur[-1] + 1:
            runs.append(cur); cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    for r in runs:
        j = np.arange(max(r[0] - 2, 0), min(r[-1] + 3, len(ink)))
        w = ink[j]
        c = origin + (float(np.sum((j + 0.5) * w) / np.sum(w))) * s
        if lo + 1.0 < c < hi - 1.0:
            out.append(c)
    return out


def raster_grid(page, fr, dpi=1200):
    import pymupdf
    pix = page.get_pixmap(clip=fr, dpi=dpi, colorspace=pymupdf.csGRAY)
    g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).astype(float)
    a = g < 200
    ink = 255.0 - g
    s = 72.0 / dpi
    gx = _centroids(a.mean(0), ink.mean(0), s, fr.x0, fr.x0, fr.x1)
    gy = _centroids(a.mean(1), ink.mean(1), s, fr.y0, fr.y0, fr.y1)
    return gx, gy


def _labels(page, fr):
    xs, ys = [], []
    for x0, y0, x1, y1, t, *_ in page.get_text("words"):
        if not NUM.match(t):
            continue
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if fr.x0 - 6 <= cx <= fr.x1 + 6 and fr.y1 - 2 <= y0 <= fr.y1 + 16 and 200 <= float(t) <= 800:
            xs.append((cx, float(t)))
        if fr.x0 - 34 <= x1 <= fr.x0 + 2 and fr.y0 - 8 <= cy <= fr.y1 + 8 and YNUM.match(t):
            ys.append((cy, float(t)))
    # ⚠ SOME SHEETS DROP THE MINUS SIGN below zero (F-4016 and F-4001 print
    # «2.0 1.0 0.0 1.0 2.0» down the axis). The axis falls monotonically from
    # top to bottom, so every label below the «0.0» is negative.
    ys.sort()
    z = next((i for i, (c, v) in enumerate(ys) if v == 0.0), None)
    if z is not None:
        ys = ys[:z + 1] + [(c, -abs(v)) for c, v in ys[z + 1:]]
    return xs, ys


def _assign(lines, labels, spacing):
    """Each label names its nearest line; lines between labels are interpolated
    at the label pitch. Returns (line coords, values) for named lines only."""
    L, V = [], []
    for c, v in labels:
        j = int(np.argmin([abs(g - c) for g in lines]))
        if abs(lines[j] - c) < 0.45 * spacing:
            L.append(lines[j]); V.append(v)
    return np.array(L), np.array(V)


def curves(page, fr):
    """Every vector curve stroke inside the frame, as (x, y) page arrays."""
    import kodak_2026_09_29b as K
    out = []
    for d in page.get_drawings():
        if d.get("type") == "f" or not fr.intersects(d["rect"]):
            continue
        cur, last = [], None
        for it in d["items"]:
            if it[0] == "l":
                seg = np.array([[it[1].x, it[1].y], [it[2].x, it[2].y]])
            elif it[0] == "c":
                seg = K._bez(it[1:5])
            else:
                continue
            if last is not None and abs(seg[0] - last).sum() > 0.05:
                out.append(np.array(cur)); cur = []
            cur += list(seg) if not cur else list(seg[1:])
            last = seg[-1]
        if cur:
            out.append(np.array(cur))
    # ⚠ SOME SHEETS SPLIT ONE CURVE OVER SEVERAL PATHS (E-7024 draws each of
    # its three curves as two to three pieces). Pieces whose ends meet within
    # 0.3 pt are chained back into one stroke before anything is measured.
    out = [r for r in out if len(r) >= 2 and np.ptp(r[:, 0]) >= 1.0 and np.ptp(r[:, 1]) >= 1.0]
    chained = True
    while chained:
        chained = False
        for i in range(len(out)):
            for j in range(len(out)):
                if i == j:
                    continue
                if np.abs(out[i][-1] - out[j][0]).sum() < 0.3:
                    out[i] = np.vstack([out[i], out[j][1:]]); del out[j]; chained = True; break
                if np.abs(out[i][-1] - out[j][-1]).sum() < 0.3:
                    out[i] = np.vstack([out[i], out[j][::-1][1:]]); del out[j]; chained = True; break
            if chained:
                break
    keep = []
    for r in out:
        if len(r) < 20:
            continue
        if np.ptp(r[:, 0]) < 1.0 or np.ptp(r[:, 1]) < 1.0:      # axis-aligned: grid / frame
            continue
        inside = (r[:, 0] > fr.x0 - 0.5) & (r[:, 0] < fr.x1 + 0.5) & (r[:, 1] > fr.y0 - 0.5) & (r[:, 1] < fr.y1 + 0.5)
        if inside.mean() > 0.9:
            keep.append(r[inside])
    return keep


def read_panel(pdf, pno):
    import pymupdf
    import kodak_2026_09_29b as K
    page = pymupdf.open(str(pdf))[pno - 1]
    box = K._panel_rect(page, "Spectral-Sensitivity") or K._panel_rect(page, "Spectral")
    if box is None:
        raise ValueError("%s p%d: no spectral panel" % (pdf.name, pno))
    # the plot frame: the smallest drawn rectangle over 100 x 100 pt that
    # overlaps the panel's title box by most of its own area (F-4043's frame
    # overhangs its title box to the right, which `_frame_in` requires it not to)
    B0 = pymupdf.Rect(box)
    cands = [it[1] for d in page.get_drawings() for it in d["items"]
             if it[0] == "re" and it[1].width > 100 and it[1].height > 100
             and (it[1] & B0).get_area() > 0.6 * it[1].get_area()]
    # ... and whose left edge carries the log-sensitivity labels
    ylab = [w for w in page.get_text("words") if YNUM.match(w[4])]
    cands = [r for r in cands
             if sum(1 for w in ylab if -1.0 <= r.x0 - w[2] <= 12.0 and r.y0 - 8 <= w[1] <= r.y1) >= 3]
    fr = min(cands, key=lambda r: r.get_area()) if cands else B0
    if abs(fr.width - B0.width) < 1e-6:
        # no drawn frame rectangle (E-7022): the plot area is bounded from its
        # own axis labels, widened 4 pt so the real frame lines fall INSIDE
        # and are read as the outermost named gridlines
        B = pymupdf.Rect(box)
        xs = [0.5 * (w[0] + w[2]) for w in page.get_text("words")
              if NUM.match(w[4]) and 200 <= float(w[4]) <= 800 and B.contains(pymupdf.Point(w[0], w[1]))]
        ys = [0.5 * (w[1] + w[3]) for w in page.get_text("words")
              if YNUM.match(w[4]) and B.contains(pymupdf.Point(w[0], w[1]))]
        lx = min(w[2] for w in page.get_text("words")
                 if YNUM.match(w[4]) and B.contains(pymupdf.Point(w[0], w[1])))
        fr = pymupdf.Rect(lx + 2.0, min(ys) - 4.0, max(xs) + 24.0, max(ys) + 4.0)
    gx, gy = raster_grid(page, fr)
    xl, yl = _labels(page, fr)
    dx = np.median(np.diff(gx)); dy = np.median(np.diff(gy))
    LX, VX = _assign(gx, xl, dx)
    LY, VY = _assign(gy, yl, dy)
    if len(LX) < 3 or len(LY) < 2:
        raise ValueError("%s p%d: %d x / %d y gridlines named" % (pdf.name, pno, len(LX), len(LY)))
    cx = np.polyfit(LX, VX, 1); cy = np.polyfit(LY, VY, 1)
    res_x = float(np.abs(np.polyval(cx, LX) - VX).max())
    res_y = float(np.abs(np.polyval(cy, LY) - VY).max())
    # where the printed LABELS sit against the lines they name, for the record
    lab_x = [abs(np.polyval(cx, c) - v) for c, v in xl]
    lab_y = [abs(np.polyval(cy, c) - v) for c, v in yl]
    rec = {}
    for r in curves(page, fr):
        x = np.polyval(cx, r[:, 0]); y = np.polyval(cy, r[:, 1])
        if x.max() - x.min() < 60:
            continue
        o = np.argsort(x); x, y = x[o], y[o]
        pk = x[np.argmax(y)]
        lab = "b" if pk < 500 else ("g" if pk < 590 else "r")
        v = np.where((GRID >= x[0] - 0.5) & (GRID <= x[-1] + 0.5), np.interp(GRID, x, y), np.nan)
        if np.all(np.isnan(v)):
            continue
        gm = float(np.nanmax(v))
        arr = np.where(np.isnan(v), -4.0, np.round(v - gm, 2)) + 0.0
        if lab in rec:          # two strokes on one layer: keep the longer
            if np.sum(rec[lab][0] > -4) >= np.sum(arr > -4):
                continue
        rec[lab] = (arr, float(x[0]), float(x[-1]))
    meta = dict(res_x=res_x, res_y=res_y, label_x=max(lab_x or [0]), label_y=max(lab_y or [0]),
                nx=len(gx), ny=len(gy))
    return rec, meta


def stored(p):
    sp = p.spectral
    if p.is_monochrome:
        return {"pan": np.asarray(sp.log_s_pan)}
    return {k: np.asarray(getattr(sp, "log_s_" + k)) for k in "rgb"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args(argv)
    import film_profiles as fp
    kd = Path(a.root) / "PDF" / "PROFILES" / "KODAK"
    bad = ran = 0
    for st, (f, pno) in PANELS.items():
        pdf = kd / f
        if not pdf.is_file():
            print("SKIP  %s: %s not present" % (st, f)); continue
        rec, meta = read_panel(pdf, pno)
        p = fp.get_profile(st)
        want = stored(p)
        if p.is_monochrome:
            if len(rec) != 1:
                print("FAIL  %s: %d curves on a monochrome panel" % (st, len(rec))); bad += 1; ran += 1; continue
            rec = {"pan": next(iter(rec.values()))}
        worst = max(float(np.abs(rec[k][0] - want[k]).max()) for k in want) if set(rec) == set(want) else 9.9
        # PORTRA 100T was re-read on 2026-09-29b by the VECTOR-grid reader and is
        # the cross-check of this raster one: one unit in the last stored place
        # between the two readers, as between two printings in kodak_2026_09_29b
        ok = worst <= (0.0100001 if st == "KODAK_PORTRA_100T" else 1e-9)
        ran += 1; bad += not ok
        print("%s  %s %s p%d: stored arrays equal the gridline read (worst %.2g); grid fit %.2g nm / %.2g log; "
              "printed labels sit up to %.2g nm / %.3g log off their lines"
              % ("PASS" if ok else "FAIL", st, f, pno, worst, meta["res_x"], meta["res_y"], meta["label_x"], meta["label_y"]))
        if a.dump:
            for k in sorted(rec):
                print("   %s = (%s)" % (k, ", ".join("%.2f" % v for v in rec[k][0])))
    print("[%s] kodak_spectral_regrid.py -- %d checks, %d failed" % ("OK" if not bad else "FAIL", ran, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
