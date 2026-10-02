"""Audit of the 2026-09-29g Soviet handbook harvest (Гурлев 1986 figures).

Re-traces, from the raster scan
`SOVIET/Справочник по фотографии (светотехника и материалы).pdf`, every
figure the batch took a number from, and fails on drift:

  * Рис. 197 ЦНЛ-32, Рис. 199 ЛН-8, Рис. 198 ЦО-32Д, Рис. 178 ОЧ-45 (12 min):
    the stored ToneCurves must sit on the re-traced points;
  * Рис. 176 (СТ-2 kinetics): the stored gamma points of the four «Фото»
    stocks must sit on the re-traced gamma curves;
  * Рис. 197 ДС-4 / ЦНД-32 / ЦНЛ-65 and Рис. 200 ЦО-Т-90ЛМ, adopted
    2026-09-30 on the owner's decision: the same re-trace check;
  * the owner's decisions stay visible: ДС-4 and ЦНЛ-65 are drawn outside the
    documents they replaced, ЦО-Т-90ЛМ's re-trace lands inside its band.

Tracing: gridlines located from pixel projections, erased, and the curves
ranked in every column (and, for steep reversal curves, every row) where the
expected number of separate strokes is found, so layers never swap.

Usage:  python gurlev_1986_2026_09_29g.py --root <project root> [--assert]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import film_profiles as fp  # noqa: E402

RESULTS: list[bool] = []


def chk(ok, label, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s   %s" % ("PASS" if ok else "FAIL", label, detail))


def _page(doc, p):
    import pymupdf
    pg = doc[p - 1]
    pix = pymupdf.Pixmap(doc, pg.get_images()[0][0])
    if pix.n != 1:
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
    return (np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width) < 140)


def _runs(col):
    idx = np.where(col)[0]
    out, s = [], None
    for i in idx:
        if s is None or i > pv + 1:
            if s is not None:
                out.append(((s + pv) / 2.0, pv - s + 1))
            s = i
        pv = i
    if s is not None:
        out.append(((s + pv) / 2.0, pv - s + 1))
    return out


def _rank(a, gx, gy, box, n, rows, maxw=10):
    a = a.copy()
    for g in gx:
        g = int(round(g)); a[:, g - 2:g + 3] = False
    for g in gy:
        g = int(round(g)); a[g - 2:g + 3, :] = False
    x0, y0, x1, y1 = box
    T = [[] for _ in range(n)]

    def put(cw, order, mk):
        cs = [c for c, w in cw if w <= maxw]
        if len(cs) == n and len(cw) == n:
            for k, c in zip(order, cs):
                T[k].append(mk(c))
        elif len(cw) == n - 1:
            ws = [w for c, w in cw]
            k = int(np.argmax(ws))
            if ws[k] >= 1.6 * min(ws) and ws[k] > 0.6 * maxw:
                c, w = cw[k]
                lst = [cc for cc, _ in cw[:k]] + [c - w / 4.0, c + w / 4.0] + [cc for cc, _ in cw[k + 1:]]
                for kk, cc in zip(order, lst):
                    T[kk].append(mk(cc))
    for x in range(x0, x1 + 1):
        cw = [(c + y0, w) for c, w in _runs(a[y0:y1, x]) if 2 <= w <= 2 * maxw]
        put(cw, list(range(n)), lambda c, x=x: (x, c))
    if rows:
        for y in range(y0, y1 + 1):
            cw = [(c + x0, w) for c, w in _runs(a[y, x0:x1]) if 2 <= w <= 2 * maxw]
            put(cw, rows, lambda c, y=y: (c, y))
    return [np.array(sorted(set(t)), float) for t in T]


def _md(x, q):
    sp = lambda z, k: k * np.logaddexp(0, z / k)
    return q[0] + q[1] * (sp(x - q[2], q[3]) - sp(x - q[4], q[5]))


def _clean(X, Y):
    o = np.argsort(X); X, Y = X[o], Y[o]
    keep = np.array([abs(Y[i] - np.median(Y[np.abs(X - X[i]) <= 0.12])) <= 0.08 for i in range(len(X))])
    return X[keep], Y[keep]


def _rms(X, Y, q, lim=None):
    X, Y = _clean(X, Y)
    if lim:
        m = (X >= lim[0]) & (X <= lim[1]); X, Y = X[m], Y[m]
    r = np.abs(_md(X, q) - Y)
    ok = r < 0.1
    # label strokes and gridline stubs survive the ranking in a few columns;
    # they are counted, and the fit must hold on at least 90 % of the points
    if ok.mean() < 0.9:
        return 9.9, len(X)
    return float(np.sqrt(np.mean(r[ok] ** 2))), int(ok.sum())


def _q(c):
    return (c.dmin, c.gamma, c.toe_x, c.toe_k, c.shoulder_x, c.shoulder_k)


PANELS = {
    # stock: (page, gx, gy, box, xref px, px per lg H, yref px, px per D, rows, extra)
    "SVEMA_CNL_32": (354, [108.5, 198, 286, 374.5, 463.5, 550, 638], [1469.5, 1558.5, 1645.5, 1736, 1825.5],
                     (165, 1471, 586, 1823), 374.5 + 0.5 * 88.0, 88.0, 1825.5, 90.0, None),
    "SVEMA_LN_8": (356, [810, 870.5, 929.5, 987, 1046, 1105, 1164, 1222, 1281.5, 1340.5],
                   [873, 931.5, 990.5, 1051, 1110, 1169, 1229], (812, 875, 1330, 1227),
                   1164, 117.33, 1229, 119.25, None),
    "SVEMA_CO_32D": (355, [882, 941, 999.5, 1057, 1118, 1175.5, 1234.5, 1292, 1351.5],
                     [1318, 1378, 1437, 1496, 1555, 1614.5, 1672], (886, 1320, 1350, 1670),
                     1118, 117.0, 1672, 117.5, [2, 1, 0]),
    # 2026-09-30, adopted on the owner's decision (Рис. 197, same page as ЦНЛ-32)
    "SVEMA_DS_4": (354, [168.5, 256, 345, 433.5, 520, 609.5], [1146.5, 1232.5, 1324],
                   (171, 1072, 607, 1321), 433.5, 88.2, 1324, 88.75, None),
    "SVEMA_CND_32": (354, [811.5, 901.5, 990, 1076.5, 1165.5, 1251.5], [1144, 1232, 1321.5],
                     (814, 1068, 1249, 1319), 1076.5, 88.0, 1321.5, 88.75, None),
    "SVEMA_CNL_65": (354, [767.5, 855.5, 942, 1029.5, 1117, 1205, 1292.5], [1469, 1557.5, 1646, 1738],
                     (770, 1471, 1290, 1736), 1029.5, 87.5, 1738, 89.67, None),
    # Рис. 200, p358; reversal on -lg H
    "SVEMA_CO_T_90LM": (358, [216, 302.5, 392, 479, 567], [210, 297.5, 384.5, 472.5],
                        (220, 212, 565, 470), 392, 87.75, 472.5, 87.5, [2, 1, 0]),
}
#: per-panel extras: x range the fit is held to (the layer labels sit to the
#: right on ЦНЛ-65), a rectangle blanked before ranking (the figure title on
#: ЦО-Т-90ЛМ, where curves never reach), and the reversal re-origin.
EXTRA = {
    "SVEMA_CNL_65": {"lim": (-2.5, 1.3)},
    "SVEMA_CO_32D": {"shift": 0.5432400960673237, "label": True},
    "SVEMA_CO_T_90LM": {"shift": 1.0215, "blank": (212, 300, 370, 565)},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--assert", dest="strict", action="store_true")
    a = ap.parse_args(argv)
    import pymupdf
    doc = pymupdf.open(Path(a.root) / "PDF" / "PROFILES" / "SOVIET" /
                       "Справочник по фотографии (светотехника и материалы).pdf")
    chk(len(doc) == 368, "the scan is the 368-page Гурлев 1986 edition", "%d pages" % len(doc))
    pages = {}
    # -- colour curves -------------------------------------------------------
    for st, (pg, gx, gy, box, xr, kx, yr, ky, rows) in PANELS.items():
        img = pages.setdefault(pg, _page(doc, pg))
        ex = EXTRA.get(st, {})
        if "blank" in ex:
            img = img.copy()
            y0_, y1_, x0_, x1_ = ex["blank"]
            img[y0_:y1_, x0_:x1_] = False
        T = _rank(img, gx, gy, box, 3, rows)
        prof = fp.get_profile(st)
        rev = prof.is_reversal
        shift = 0.0
        if rev:
            # stored x = -lg H - shift; recover the shift from the stored green
            # curve (D 1.20 at x = 0 by construction)
            shift = 0.0
        worst = 0.0
        for k, ch in enumerate(("b", "g", "r")):
            p = T[k]
            X = (p[:, 0] - xr) / kx
            Y = (yr - p[:, 1]) / ky
            if rev:
                X = -X
                if ex.get("label"):
                    m = ~((X < -0.5) & (Y > 1.0))  # the «ЦО-32Д» label
                    X, Y = X[m], Y[m]
                X = X - ex["shift"]
            e, n = _rms(X, Y, _q(getattr(prof.curves, ch)), lim=ex.get("lim"))
            worst = max(worst, e)
        chk(worst < (0.04 if rev else 0.02), "%s: the stored curves sit on the re-traced Гурлев figure" % st,
            "worst layer rms %.4f D" % worst)
    # -- ОЧ-45, Рис. 178, 12 min (third of four from the top) ----------------
    img = pages.setdefault(298, _page(doc, 298))
    T = _rank(img, [164, 226, 284, 342, 401.5, 460, 519, 576, 634.5, 693, 751],
              [445.5, 504.5, 562.5, 621.5, 681, 740, 797.5], (240, 447, 749, 796), 4, [3, 2, 1, 0])
    p = T[2]
    X = -((p[:, 0] - 634.5) / 116.83) - 1.2936213446118812
    Y = (797.5 - p[:, 1]) / 117.5
    e, n = _rms(X, Y, _q(fp.get_profile("TASMA_OCH_45").curves.g), lim=(0.15 - 1.2936, 2.4 - 1.2936))
    chk(e < 0.03, "TASMA_OCH_45: the stored curve sits on Рис. 178's 12 min curve", "rms %.4f D over %d points" % (e, n))
    # -- Рис. 176 kinetics --------------------------------------------------
    img = pages.setdefault(294, _page(doc, 294))
    K = {"SVEMA_FOTO_32": ((200, 1100, 517, 1260), 195.5, 12.9,
                           [(977.5, 10), (1032, 5), (1096, 2), (1150, 1), (1202, 0.5), (1284, 0.2), (1336, 0.1)]),
         "SVEMA_FOTO_65": ((818, 1150, 1136, 1300), 816.5, 12.85,
                           [(1029, 10), (1081, 5), (1134, 2), (1185.5, 1), (1234.5, 0.5), (1292.5, 0.2), (1334, 0.1)]),
         "SVEMA_FOTO_130": ((203, 1950, 587, 2130), 201.5, 13.075,
                            [(1814, 10), (1865, 5), (1931, 2), (1984, 1), (2036, 0.5), (2100.5, 0.2), (2153.5, 0.1)]),
         "SVEMA_FOTO_250": ((825, 1950, 1207, 2130), 823.5, 12.875,
                            [(1815, 10), (1864.5, 5), (1929, 2), (1984, 1), (2037, 0.5), (2100.5, 0.2), (2153, 0.1)])}
    worst = 0.0
    for st, (box, t0, ppm, lab) in K.items():
        ys = [a_ for a_, _ in lab]; ls = [np.log10(b_) for _, b_ in lab]
        pts = [q for q in fp.get_profile(st).processing_family.points if q.developer == "СТ-2" and q.gamma > 0]
        for q in pts:
            x = int(round(t0 + q.minutes * ppm))
            cand = []
            for dx in range(-3, 4):
                for c, w in _runs(img[box[1]:box[3], x + dx]):
                    if 3 <= w <= 9:
                        g = 10 ** np.interp(c + box[1], ys, ls)
                        cand.append(g)
            if cand:
                worst = max(worst, min(abs(g - q.gamma) for g in cand))
            else:
                worst = max(worst, 9.9)
    chk(worst < 0.04, "Рис. 176: every stored СТ-2 gamma point lies on a stroke of its panel's gamma curve",
        "worst %.3f" % worst)
    # -- the 2026-09-30 owner decisions, pinned so they stay visible --------
    # ДС-4 is drawn above the ТУ-84 aims and ЦНЛ-65 above its ГОСТ band on all
    # three layers: both ADOPTED anyway on the owner's decision, so the
    # disagreement must stay true of the stored curve or the note is stale.
    # ЦО-Т-90ЛМ's re-trace lands inside 1.4-1.6 where the 29g record did not.
    F = fp.GURLEV_1986_CURVE_FITS
    def slope(q):
        x = np.linspace(-4, 4, 8001); return float(np.gradient(_md(x, q), x).max())
    ds4 = [slope(_q(getattr(fp.get_profile("SVEMA_DS_4").curves, c))) for c in "rgb"]
    c65 = fp.get_profile("SVEMA_CNL_65")
    t65 = next(t for t in c65.tolerance if t.is_default)
    above = [getattr(c65.curves, c).gamma > t65.gamma_hi_rgb[i] for i, c in enumerate("rgb")]
    cot = [getattr(fp.get_profile("SVEMA_CO_T_90LM").curves, c).gamma for c in "rgb"]
    chk(min(ds4) > 0.75 and all(above) and all(1.4 <= g <= 1.6 for g in cot)
        and ("SVEMA_DS_5M", "g", "record") in F,
        "owner decisions: ДС-4 drawn above its ТУ-84 aims, ЦНЛ-65 above ГОСТ 25120-82's band (kept), ЦО-Т-90ЛМ re-trace inside 1.4-1.6; ДС-5М still record-only",
        "ДС-4 slope %.2f/%.2f/%.2f, ЦНЛ-65 above band %s, ЦО-Т-90ЛМ gamma %.2f/%.2f/%.2f" % (*ds4, above, *cot))
    n_bad = RESULTS.count(False)
    print("[%s] gurlev_1986_2026_09_29g.py -- %d checks, %d failed" % ("OK" if not n_bad else "FAIL", len(RESULTS), n_bad))
    return 1 if (a.strict and n_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
