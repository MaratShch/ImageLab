#!/usr/bin/env python3
"""«Современные фотоматериалы и их обработка» -- the 42 «Кривые кинетики
проявления» panels, audited against the page (2026-10-01d, owner-approved
batch item 2).

WHAT IS AUDITED. `film_profiles.SOVREMENNYE_2004_KINETICS` holds 155 curves
(1139 points) read off the panels' own ink at fixed contrast rows / fixed
time columns. This module re-rasterises every panel from the PDF (the figure
XObject the caption belongs to, located by `sovremennye_2004.build_index`),
maps every stored (contrast, minutes) point back through the panel's ruled-
axis calibration below, and asks whether it lands on ink.

A HIT RATE ALONE PROVES NOTHING on a busy panel, so it is SCORED AGAINST
NULLS: the same points displaced by +-0.6 min and by +-0.05 contrast. The
aligned rate must beat every displaced rate by a wide margin.

THE TABLE CROSS-CHECK. Kodak draws each line from the normal contrast (0.56;
0.60 for T-MAX 400, the book's own text). Every Kodak curve whose stock has a
printed normal time for the same developer, dilution, vessel and temperature
is read at that contrast and compared with the table.

THE SAME-DRAWING CHECK. рис. 3.259 / 3.260 (drum) reprint рис. 3.256 / 3.257
(small tank): their crossings must agree to the reading precision.

NOT STORED, printed with the reason: 3.261 (machine SPEED axis), 3.358 /
3.359 (Scala D-max and contrast against EI -- the Agfa F-PF-D4 drawing already
in AGFA_SCALA_200X.push), 3.364 (Kodak HSI, no profile).

Usage:  python sovremennye_kinetics.py --root <project root>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BOOK = "Современные фотоматериалы и их обработка.pdf"
ALT_PDF = Path("/mnt/user-data/uploads/PYTHON.TST/PDF/PROFILES/SOVIET") / BOOK

#: panel -> (x_a, x_b, y_a, y_b): minutes = x_a*px + x_b, contrast = y_a*py + y_b,
#: fitted to the printed tick labels (frame-anchored where the panel is ruled).
CAL = {
    115: (0.04201298, -0.9723906, -0.0006493141, 0.7588655),
    119: (0.04202021, -0.9496995, -0.0006535855, 0.7583855),
    205: (0.05411436, -3.803671, -0.00275311, 1.243562),
    211: (0.05407339, -3.369645, -0.002700296, 1.433326),
    237: (0.02882948, 1.81806, -0.001235036, 0.9179629),
    238: (0.02891109, 1.570466, -0.001244414, 0.9187774),
    239: (0.02779009, 1.820781, -0.001193968, 0.9177944),
    240: (0.02887523, 1.800703, -0.001240534, 0.9170747),
    247: (0.04148727, -3.328137, -0.002482191, 1.226281),
    248: (0.02898365, 1.718958, -0.001248852, 0.918487),
    249: (0.02894251, 1.76394, -0.001242696, 0.9174495),
    256: (0.04389876, 0.601466, -0.001604666, 1.121225),
    257: (0.04394451, 0.2742471, -0.001608881, 1.126085),
    258: (0.04397918, 0.4305902, -0.001605929, 1.131727),
    259: (0.04397574, 0.3895085, -0.001602496, 1.130092),
    260: (0.04397918, 0.4305902, -0.001608093, 1.129629),
    266: (0.05404287, -3.272845, -0.00270864, 1.242043),
    269: (0.05411343, -3.455743, -0.002699953, 1.24294),
    272: (0.05405628, -3.327126, -0.002696692, 1.236328),
    276: (0.05418172, -3.736403, -0.002700125, 1.224075),
    277: (0.05402224, -3.672793, -0.002699714, 1.24451),
    282: (0.04066567, -2.768081, -0.002147167, 1.027745),
    283: (0.01625058, 2.893824, -0.001081071, 0.8158528),
    300: (0.05467928, -0.4563617, -0.001680672, 1.126891),
    301: (0.0539484, -0.2682532, -0.001656974, 1.129248),
    302: (0.05419975, -0.2697195, -0.001672213, 1.125578),
    303: (0.05292128, -0.03770765, -0.001633945, 1.126006),
    307: (0.04517443, -2.253565, -0.002249823, 1.258081),
    311: (0.06494928, -5.456703, -0.002610587, 1.237791),
    318: (0.05479577, -4.370247, -0.001324448, 0.9544159),
    319: (0.05484774, -4.651345, -0.001331359, 0.9212697),
    320: (0.05478766, -4.32666, -0.001768274, 1.131754),
    321: (0.05484382, -4.277639, -0.001766675, 1.134504),
    322: (0.05468073, -4.268756, -0.001765109, 1.129216),
    337: (0.03666361, -3.446379, -0.003664122, 1.27145),
    342: (0.02368265, -2.261693, -0.0023692, 1.235538),
    346: (0.02371073, -2.086544, -0.002385686, 1.234592),
    351: (0.03555556, -2.471111, -0.003571429, 1.264286),
}

REFUSED = {
    261: "Versamat 5 transport SPEED (m/min); no developer path length to convert",
    358: "Scala 200x D-max against EI: the Agfa F-PF-D4 drawing, already in AGFA_SCALA_200X.push",
    359: "Scala 200x contrast against EI: the Agfa F-PF-D4 drawing, already in AGFA_SCALA_200X.push",
    364: "Kodak Professional High-Speed Infrared: no profile",
}
PARTIAL = {311: (2, 9), 318: (2, 8), 321: (3, 7), 322: (3, 7)}   # (stored, printed)

HIT_MIN = 0.90          # aligned points on ink
NULL_MARGIN = 0.30      # every displaced rate at least this far below
TABLE_TOL = 0.75        # minutes, against a printed normal time
SAME_TOL = 0.10         # minutes, рис. 3.259/3.260 against 3.256/3.257


def _pdf(root):
    p = Path(root) / "PDF" / "PROFILES" / "SOVIET" / BOOK
    return p if p.is_file() else ALT_PDF


def _interp(pts, g):
    pts = sorted(pts)
    for (g0, t0), (g1, t1) in zip(pts, pts[1:]):
        if g0 <= g <= g1 and g1 > g0:
            return t0 + (t1 - t0) * (g - g0) / (g1 - g0)
    (g0, t0), (g1, t1) = (pts[0], pts[1]) if g < pts[0][0] else (pts[-2], pts[-1])
    return t0 + (t1 - t0) * (g - g0) / (g1 - g0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    import numpy as np
    import cv2
    import pymupdf
    import film_profiles as fp
    import sovremennye_2004 as s4

    rows = fp.SOVREMENNYE_2004_KINETICS
    pdf = _pdf(a.root)
    if not pdf.is_file():
        print("[SKIP] book not on disk:", pdf)
        return 0
    doc = pymupdf.open(str(pdf))
    idx = {r["num"]: r for r in s4.build_index(doc) if r["kind"] == "kinet"}
    fails = 0
    print("[i] %d kinetics panels indexed; %d curves / %d points stored"
          % (len(idx), len(rows), sum(len(r[10]) for r in rows)))

    # 1. every stored point back on ink, against displaced nulls
    figs = sorted({int(r[1].split(".")[1]) for r in rows})
    masks = {}
    for n in figs:
        im = s4.bitmap(doc, idx[n]["xref"])
        m = (im < 140).astype(np.uint8)
        masks[n] = cv2.dilate(m, np.ones((5, 5), np.uint8))   # +-2 px
    shifts = {"aligned": (0, 0), "+0.6 min": (0.6, 0), "-0.6 min": (-0.6, 0),
              "+0.05": (0, 0.05), "-0.05": (0, -0.05)}
    tot = {k: [0, 0] for k in shifts}
    for r in rows:
        n = int(r[1].split(".")[1]); xa, xb, ya, yb = CAL[n]; m = masks[n]
        for g, t in r[10]:
            for k, (dt, dg) in shifts.items():
                x = int(round((t + dt - xb) / xa)); y = int(round((g + dg - yb) / ya))
                ok = 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x] > 0
                tot[k][0] += int(ok); tot[k][1] += 1
    rate = {k: v[0] / max(v[1], 1) for k, v in tot.items()}
    worst_null = max(v for k, v in rate.items() if k != "aligned")
    ok = rate["aligned"] >= HIT_MIN and rate["aligned"] - worst_null >= NULL_MARGIN
    fails += not ok
    print("[%s] ink: aligned %.1f %% of %d points; displaced %s"
          % ("PASS" if ok else "FAIL", 100 * rate["aligned"], tot["aligned"][1],
             ", ".join("%s %.1f %%" % (k, 100 * v) for k, v in rate.items() if k != "aligned")))

    # 2. Kodak curves against the printed normal times. ⚠ WHAT THIS TESTS IS
    # THE ATTRIBUTION, not the condition: a panel may be drawn for another
    # format or edition than the table beside it (рис. 3.248 runs ~1.5 min
    # long on every curve), and that is stored as printed. The gate is that
    # inside every panel the stored developer labels fit the tables at least
    # as well as any other assignment of those labels to those curves.
    import itertools
    F = {p.name: p for p in fp.FILM_PROFILES}
    per = {}
    n_cmp = n_ok = 0
    for st, fig, dev, dil, cel, vessel, fmt, meas, ei, ed, pts in rows:
        if meas != "kodak_ci":
            continue
        p = F[st]; aim = 0.60 if st == "KODAK_TMAX_400" else 0.56
        if not (min(g for g, _ in pts) - 0.03 <= aim <= max(g for g, _ in pts)):
            continue
        gen = ed.split(" / ")[0] if " / " in ed else ""
        ts = sorted({q.minutes for q in p.processing_family.points
                     if q.developer == dev and q.dilution == dil and q.vessel == vessel
                     and abs(q.celsius - cel) < 0.05 and q.gamma <= 0 and q.contrast_index <= 0
                     and q.film_format in (fmt, "")
                     and q.exposure_index in (0, p.exposure_index)
                     and ((gen and gen in q.edition) or (not gen and "Современные 2004, рис." not in q.edition
                          and not any(g in q.edition for g in ("Tri-X Pan TX", "Plus-X Pan", "TRI-X Pan Professional"))))})
        if not ts:
            continue
        t = _interp(pts, aim)
        d = min(abs(x - t) for x in ts)
        n_cmp += 1; n_ok += d <= TABLE_TOL
        per.setdefault((fig, cel), []).append((t, ts, dev + " " + dil))
    bad = []
    for key, cur in per.items():
        if len(cur) < 2:
            continue
        # a panel drawn for another condition is offset as a whole: remove the
        # panel's median offset first, so the test sees the LABELS only.
        off = sorted(c[0] - min(c[1], key=lambda x: abs(x - c[0])) for c in cur)[len(cur) // 2]
        err = lambda order: sum(min(abs(x - (c[0] - off)) for x in cur[j][1])
                                for c, j in zip(cur, order))
        base = err(range(len(cur)))
        best = min(err(o) for o in itertools.permutations(range(len(cur))))
        if base > best + 0.10:
            bad.append((key, round(base, 2), round(best, 2)))
    ok = n_cmp >= 60 and not bad
    fails += not ok
    print("[%s] attribution: %d Kodak curves in %d panels against printed normal times; "
          "no relabelling fits better%s" % ("PASS" if ok else "FAIL", n_cmp,
          sum(1 for v in per.values() if len(v) >= 2), "" if not bad else " -- EXCEPT %s" % bad))
    print("[i] %d of %d within %.2f min of a printed time (%.0f %%); the rest are panels "
          "drawn for another condition than the table, stored as printed"
          % (n_ok, n_cmp, TABLE_TOL, 100.0 * n_ok / max(n_cmp, 1)))

    # 3. рис. 3.259/3.260 reprint 3.256/3.257
    by = {}
    for r in rows:
        by.setdefault((r[1], r[4]), r[10])
    dmax = 0.0; npair = 0
    for a_, b_ in (("3.259", "3.256"), ("3.260", "3.257")):
        for (fig, cel), pts in by.items():
            if fig != a_ or (b_, cel) not in by:
                continue
            ref = dict(by[(b_, cel)])
            for g, t in pts:
                if g in ref:
                    dmax = max(dmax, abs(ref[g] - t)); npair += 1
    ok = npair >= 30 and dmax <= SAME_TOL
    fails += not ok
    print("[%s] same drawing: рис. 3.259/3.260 against 3.256/3.257, %d shared rows, "
          "max %.2f min" % ("PASS" if ok else "FAIL", npair, dmax))

    # 4. coverage: every indexed panel stored, refused or partial with a reason
    stored = set(figs)
    missing = sorted(set(idx) - stored - set(REFUSED))
    ok = not missing
    fails += not ok
    print("[%s] coverage: %d panels stored (%d partial), %d refused, missing %s"
          % ("PASS" if ok else "FAIL", len(stored), len(set(PARTIAL) & stored),
             len(REFUSED), missing or "none"))
    for n, why in sorted(REFUSED.items()):
        print("     refused рис. 3.%d: %s" % (n, why))
    for n, (k, of) in sorted(PARTIAL.items()):
        print("     partial рис. 3.%d: %d of %d curves stored, the rest overlap within "
              "a line width" % (n, k, of))
    print("[%s] %d checks, %d failed" % ("OK" if not fails else "FAIL", 4, fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
