#!/usr/bin/env python3
"""The AGFAPAN processing axis of «Technical Data PF»: the gamma-time panels,
the temperature tables, and which VESSEL each of them is (2026-09-06i).

    PDF/PROFILES/AGFA/agfa_films.pdf -- Agfa-Gevaert, «Technical Data PF»,
    1st edition 09/1998, printed p10, the fourth panel of each APX column.

⚠⚠ THE LAST UNREAD CURVE SET IN THIS DOCUMENT. `agfa_1998_curves.py` has
digitised these twelve curves since 2026-09-01 and printed them on every run;
nothing ever adopted them, because the database's `ProcessingFamily` was filled
from a different Agfa publication -- «Technical Data P-16-C», 08/1999
(`agfa_film_chem.pdf`) -- whose contrast tables give the same quantity as text.
Two sources for one relation, never compared.

⚠⚠ AND COMPARING THEM ANSWERS A QUESTION THE STORED RECORD ADMITS IT CANNOT.
`AGFA_APX_*.processing_family.source` says, in its own words:

    "THE METHOD IS IN THE TRAILING COMMENT, NOT IN A FIELD: DevelopmentPoint
     has developer, dilution, minutes, celsius and gamma but no agitation or
     vessel, so the drum and small-tank rows are distinguishable only by their
     times."

P-16-C prints each contrast table twice -- «Rotary process (drum)» and «Small
tank, tray» -- and the two disagree by 20-40 % of time. The panel plots ONE of
them and does not say which. It says so by its numbers:

    mean |panel t(gamma 0.65) - drum time|        1.152 min
    mean |panel t(gamma 0.65) - small-tank time|  0.112 min

over all FIFTEEN film x developer combinations, with every single one closer to
the small tank and the worst tank miss (0.32 min) still smaller than the BEST
drum agreement (0.39 min). ⚠ THE MIDDLE POINT IS THE ONE THAT DISCRIMINATES and
it is the only one P-16-C prints for the small tank at all -- that table has a
single gamma 0.65 row where the drum table has three. Which is exactly why the
panel is worth adopting: it supplies the small tank's **gamma 0.55 and gamma
0.75 times, which Agfa never print anywhere in this corpus**.

⚠ ONE CURVE CARRIES TWO DEVELOPER NAMES AND THAT IS AGFA'S CLAIM, NOT REUSED
ARTWORK. RODINAL SPECIAL and STUDIONAL LIQUID share a single drawn curve on all
three panels. Unlike the APX 100 / APX 400 sharpness drawing -- where one
drawing across two FILMS means at most one of them is a measurement --
P-16-C independently prints identical times for these two developers on all
three films (APX 25 4 min, APX 100 4 min, APX 400 4.5 min at gamma 0.65). Agfa
are asserting the two developers are equivalent here, so both names carry the
reading and neither is a duplicate of the other.

⚠ ATOMAL FF IS NOT PLOTTED. Five developer names sit beside four curves; the
fifth name pairs with STUDIONAL as above, and ATOMAL FF appears in P-16-C's
tables but on no curve. Its stored points keep vessel "" -- unknown -- rather
than being assigned to the family the others fall in.

⚠⚠ AND p11 PRINTS A THIRD AXIS NOTHING HAD READ: DEVELOPING TIME AGAINST
TEMPERATURE, 18 / 20 / 22 / 24 C, in three blocks captioned «Processing in
trays», «Processing in drums» and «Processing in tanks». Every stored
DevelopmentPoint in this database sits at 20 C; this table is the only
temperature axis any AGFAPAN stock has.

⚠ ITS QUANTITY IS NOT PRINTED AND IS NOT ASSUMED -- IT IS PROVED ON 30 CELLS.
The table's columns carry a temperature and nothing else: no gamma, no contrast
index. But its 20 C column reproduces P-16-C's gamma 0.65 figures EXACTLY --
all fifteen tray cells and all fifteen drum cells, across five developers and
three films, to the printed half-minute. So the quantity the table tabulates is
the gamma 0.65 time, and the other three columns are that same contrast at
another temperature. Without that check the 18 C column would be a time to an
unstated contrast, which is not a datum.

⚠⚠ AGFA'S OWN VESSEL NAMES CONTRADICT EACH OTHER ON THIS PAGE AND THE
CONTRADICTION IS RECORDED, NOT RESOLVED. p11 prints THREE blocks -- trays,
drums, tanks -- where P-16-C prints two, «Rotary process (drum)» and «Small
tank, tray». The 20 C agreement above pins P-16-C's combined caption to p11's
TRAYS column. p11's «tanks» block matches neither: REFINAL reads 7 min there
against 6 in trays and 5 in drums, and P-16-C has no 7 anywhere. ⚠ AND THE
EXPOSURE-INDEX TABLE ON THE SAME PAGE CALLS THE 6-MINUTE FIGURE "small tank"
("*) Processing in small tank at 20 C", REFINAL 6 min) -- so Agfa use "small
tank" for the tray column in one footnote and for a different column in the
table above it. P-16-C section 3.1 lists four methods, "tray, small tank, drum,
large tank", which is probably what the third block is; probably is not a
citation, so it is stored under the caption it is printed with, `tank`, and
this note travels with it.

Run:  python agfa_1998_gamma_time.py [--root .] [--assert]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PDF = "PDF/PROFILES/AGFA/agfa_films.pdf"

#: How far outside a traced curve's own gamma span an endpoint may be read.
ENDPOINT_TOL = 0.005

SOURCE = (
    "Agfa-Gevaert AG, «Technical Data PF», 1st edition 09/1998 -- "
    "PDF/PROFILES/AGFA/agfa_films.pdf p10, the gamma / developing-time panel "
    "of each AGFAPAN column; traced 2026-09-06i by agfa_1998_gamma_time.py")

#: P-16-C's printed contrast tables, read as TEXT off `agfa_film_chem.pdf`
#: pp5-6. Per film and developer: ((drum t@0.55, t@0.65, t@0.75), tank t@0.65).
#: `None` is a printed dash, which is a refusal and not a zero.
#: ⚠ THE SMALL-TANK TABLE HAS ONE ROW. That is the whole reason this panel is
#: worth reading: everything the small tank does away from gamma 0.65 is
#: printed only as this curve.
CHEM = {
    "AGFAPAN APX 25": {
        "RODINAL 1+25":     ((None, 4, 8), 6),
        "REFINAL":          ((3, 5, 8), 6),
        "RODINAL 1+50":     ((4, 9, 15), 10),
        "RODINAL SPECIAL":  ((None, 3, 5), 4),
        "STUDIONAL LIQUID": ((None, 3, 5), 4),
    },
    "AGFAPAN APX 100": {
        "RODINAL 1+25":     ((4, 7, 10), 8),
        "REFINAL":          ((3, 5, 8), 6),
        "RODINAL 1+50":     ((8, 14, 19), 17),
        "RODINAL SPECIAL":  ((None, 3.5, 5), 4),
        "STUDIONAL LIQUID": ((None, 3.5, 5), 4),
    },
    "AGFAPAN APX 400": {
        "RODINAL 1+25":     ((4, 5, 6), 7),
        "REFINAL":          ((3, 5, 8), 6),
        "RODINAL 1+50":     ((7, 9, 11), 11),
        "RODINAL SPECIAL":  ((3, 4, 5), 4.5),
        "STUDIONAL LIQUID": ((3, 4, 5), 4.5),
    },
}

#: p11's «Processing» tables: {profile: {block: {developer: [18, 20, 22, 24 C]}}}
#: in minutes to gamma 0.65, `None` for a printed dash. Read as TEXT, and
#: re-derived from the page on every run by `read_temperatures`.
TEMPS = (18.0, 20.0, 22.0, 24.0)

#: The caption each printed block is stored under. ⚠ «trays» AND P-16-C's
#: «Small tank, tray» ARE ONE VESSEL -- proved on 30 cells at 20 C -- so they
#: share one label; «tanks» gets its own because it matches neither.
VESSEL = {"trays": "small tank, tray", "drums": "drum", "tanks": "tank"}

PROFILE = {"AGFAPAN APX 25": "AGFA_APX_25",
           "AGFAPAN APX 100": "AGFA_APX_100",
           "AGFAPAN APX 400": "AGFA_APX_400"}

#: What the panels read at the small tank's three contrasts, pinned.
#: ⚠ gamma 0.65 IS THE CROSS-CHECK AND 0.55 / 0.75 ARE THE HARVEST. The middle
#: column can be compared against a printed number; the outer two cannot,
#: because Agfa never print them for this vessel.
EXPECTED: dict[str, dict[str, tuple]] = {
    "AGFAPAN APX 25": {
        "REFINAL":           (2.87, 5.93, 9.87),
        "RODINAL 1+25":      (2.90, 6.15, 10.91),
        "RODINAL 1+50":      (5.93, 9.95, 16.86),
        "RODINAL SPECIAL":   (1.93, 3.92, 7.01),
        "STUDIONAL LIQUID":  (1.93, 3.92, 7.01),
    },
    "AGFAPAN APX 100": {
        "REFINAL":           (2.87, 5.90, 9.84),
        "RODINAL 1+25":      (4.24, 7.92, 12.27),
        "RODINAL 1+50":      (12.27, 16.74, None),
        "RODINAL SPECIAL":   (2.89, 3.92, 5.91),
        "STUDIONAL LIQUID":  (2.89, 3.92, 5.91),
    },
    "AGFAPAN APX 400": {
        "REFINAL":           (3.31, 5.88, 9.46),
        "RODINAL 1+25":      (3.95, 6.97, 11.84),
        "RODINAL 1+50":      (5.90, 11.32, None),
        "RODINAL SPECIAL":   (2.76, 4.38, 6.98),
        "STUDIONAL LIQUID":  (2.76, 4.38, 6.98),
    },
}


def read_all(root: Path):
    """{printed film: {developer: (t55, t65, t75) or None}} plus the diagnosis."""
    # ⚠ THE RAW TRACE, NOT `collect`'s `samples`. That field is resampled on a
    # half-minute grid starting at `ceil(t_min * 2) / 2`, which trims both ends
    # of every curve -- and the ends are exactly what this module is after,
    # because Agfa draw each curve to stop at gamma 0.55 and 0.75. Read through
    # `samples`, all thirty harvested times come back None and the module
    # reports nothing while appearing to work.
    import agfa_1998_curves as C
    doc = pymupdf.open(str(root / PDF))
    out, diag = {}, []
    for profile, printed, pageno, (xlo, xhi), kind in C.COLUMNS:
        if kind != "mono":
            continue
        pg = doc[pageno - 1]
        ws = C._words(pg)
        panel, err = C._panel(pg, ws, "gamma", C.BANDS["curves"], xlo, xhi)
        if panel is None:
            print("  [FAIL] %s: no gamma-time panel (%s)" % (printed, err))
            continue
        segs = C._split_curves(pg, panel)
        owners = C._assign(C._named_labels(pg, panel, C.DEVELOPERS), segs)
        fam = {}
        for i, (xs, ys) in enumerate(segs):
            arr = np.column_stack([panel.X(xs), panel.Y(ys)])
            for o in owners[i]:
                fam[o.split(" (")[0]] = arr
        got = {}
        for dev, arr in sorted(fam.items()):
            t, g = arr[:, 0], arr[:, 1]
            o = np.argsort(g)
            t, g = t[o], g[o]
            row = []
            for target in (0.55, 0.65, 0.75):
                # ⚠⚠ NEVER EXTRAPOLATE A DEVELOPING TIME -- BUT DO NOT REFUSE
                # ONE FOR A THOUSANDTH EITHER. Agfa draw each curve to stop at
                # exactly gamma 0.55 and 0.75, and the trace comes back
                # 0.550..0.749: a strict containment test refuses the two
                # endpoints that are the entire harvest, on 0.001 of gamma
                # that is stroke geometry and not data. `ENDPOINT_TOL` is
                # 0.005, five times that and forty times under the gap that
                # matters -- APX 100's RODINAL 1+50 curve genuinely stops at
                # 0.670, and reading 0.75 off it would invent five minutes.
                lo, hi = float(g.min()), float(g.max())
                if lo - ENDPOINT_TOL <= target <= hi + ENDPOINT_TOL:
                    row.append(round(float(np.interp(
                        min(max(target, lo), hi), g, t)), 2))
                else:
                    row.append(None)
            got[dev] = tuple(row)
            chem = CHEM.get(printed, {}).get(dev)
            if chem and row[1] is not None:
                (_d55, d65, _d75), t65 = chem
                diag.append((printed, dev, row[1], d65, t65))
        out[printed] = got
    return out, diag


def read_temperatures(root: Path):
    """p11's three «Processing» blocks: {profile: {vessel: {developer: [4]}}}.

    ⚠ THE BLOCK CAPTION IS READ FROM THE PAGE, not assumed from row order. The
    three blocks are unlabelled in the text layer except by the sentence
    «Processing in trays» / «drums» / «tanks» that precedes each, and the
    AGFAPAN columns repeat that sentence three times per page, once per film.
    """
    doc = pymupdf.open(str(root / PDF))
    pg = doc[10]
    rows = {}
    for x0, y0, _x1, _y1, w, *_ in pg.get_text("words"):
        if 55 < y0 < 250:
            rows.setdefault(round(y0, 0), []).append((x0, w))
    cols = {"AGFA_APX_25": 35.0, "AGFA_APX_100": 215.0, "AGFA_APX_400": 391.0}
    out, block = {}, None
    for y in sorted(rows):
        line = " ".join(w for _x, w in sorted(rows[y]))
        m = re.search(r"Processing in (\w+)", line)
        if m:
            block = m.group(1)
            continue
        if block is None or "Developer" in line or "°C" in line:
            continue
        for prof, a in cols.items():
            seg = sorted([(x, w) for x, w in rows[y] if a <= x < a + 176.0])
            if not seg:
                continue
            # ⚠ THE DEVELOPER NAME IS EVERYTHING LEFT OF THE FIRST VALUE
            # COLUMN, and "1 + 25" has to survive it: a filter that drops
            # numeric tokens collapses RODINAL 1+25 and RODINAL 1+50 onto one
            # key and silently keeps whichever row is read second.
            name = " ".join(w for x, w in seg if x < a + 70.0).replace("1 + ", "1+")
            vals = []
            for x, w in seg:
                if x < a + 70.0:
                    continue
                if w == "\u00bd":
                    if vals and vals[-1] is not None:
                        vals[-1] += 0.5
                    continue
                if w in ("\u2013", "-"):
                    vals.append(None)
                    continue
                try:
                    vals.append(float(w))
                except ValueError:
                    pass
            if len(vals) == 4:
                out.setdefault(prof, {}).setdefault(VESSEL[block], {})[name] = vals
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()
    if not (root / PDF).is_file():
        print("  [SKIP] source not present: %s" % (root / PDF))
        return 0

    print("AGFA «Technical Data PF» 09/1998 p10 -- the three gamma-time panels")
    out, diag = read_all(root)
    bad = 0

    # ---- WHICH VESSEL -----------------------------------------------------
    print("\n  WHICH PROCESS DOES THE PANEL PLOT? P-16-C prints the contrast "
          "table twice, for the drum and for the small tank; the panel says "
          "which by its numbers.")
    dd = [abs(p - d) for _f, _v, p, d, _t in diag]
    dt = [abs(p - t) for _f, _v, p, _d, t in diag]
    closer = sum(1 for a, b in zip(dd, dt) if b < a)
    for f, v, p, d, t in diag:
        print("      %-16s %-17s panel %6.2f   drum %5s  tank %5s   "
              "|d| %5.2f / %5.2f  -> %s"
              % (f.replace("AGFAPAN ", ""), v, p, d, t,
                 abs(p - d), abs(p - t), "TANK" if abs(p - t) < abs(p - d)
                 else "drum"))
    print("      mean |panel - drum| %.3f min   mean |panel - small tank| "
          "%.3f min   over %d combinations" % (np.mean(dd), np.mean(dt), len(dd)))
    print("      worst tank miss %.2f min, best drum agreement %.2f min -- "
          "the two families do not overlap" % (max(dt), min(dd)))
    if not (closer == len(diag) and np.mean(dt) < 0.4 * np.mean(dd)
            and max(dt) < min(dd)):
        print("      [FAIL] the vessel identification no longer holds: %d of "
              "%d closer to the tank" % (closer, len(diag)))
        bad += 1
    else:
        print("      ✅ ALL %d ARE CLOSER TO THE SMALL TANK. The panel plots "
              "«Small tank, tray», so its gamma 0.55 and 0.75 readings are "
              "that vessel's times -- which P-16-C never prints." % closer)

    # ---- WHAT IS HARVESTED ------------------------------------------------
    print("\n  THE SMALL-TANK TIMES, gamma 0.55 / 0.65 / 0.75:")
    for printed in sorted(out):
        print("    %s" % printed)
        for dev, (a, b, c) in sorted(out[printed].items()):
            chem = CHEM.get(printed, {}).get(dev)
            mark = ""
            if chem and b is not None:
                mark = "   (P-16-C prints %s min at 0.65)" % chem[1]
            print("      %-17s %s / %s / %s min%s"
                  % (dev,
                     "  --" if a is None else "%5.2f" % a,
                     "  --" if b is None else "%5.2f" % b,
                     "  --" if c is None else "%5.2f" % c, mark))

    for printed, want in EXPECTED.items():
        for dev, wv in want.items():
            got = (out.get(printed) or {}).get(dev)
            if got is None:
                print("  [MISMATCH] %s / %s missing" % (printed, dev))
                bad += 1
                continue
            for i, (g, w) in enumerate(zip(got, wv)):
                if (g is None) != (w is None):
                    print("  [MISMATCH] %s / %s gamma %.2f: %s vs pinned %s"
                          % (printed, dev, (0.55, 0.65, 0.75)[i], g, w))
                    bad += 1
                    continue
                if w is None:
                    continue
                if abs(g - w) > 0.15:
                    print("  [MISMATCH] %s / %s gamma %.2f: %s vs pinned %.2f"
                          % (printed, dev, (0.55, 0.65, 0.75)[i], g, w))
                    bad += 1

    # ---- p11's TEMPERATURE TABLES -----------------------------------------
    print("\n  p11's «Processing» TABLES -- developing time against "
          "TEMPERATURE, the only such axis any AGFAPAN stock has:")
    T = read_temperatures(root)
    hits = miss20 = 0
    for prof, blocks in sorted(T.items()):
        printed = [k for k, v in PROFILE.items() if v == prof][0]
        for vessel, devs in sorted(blocks.items()):
            for dev, vals in sorted(devs.items()):
                print("      %-14s %-17s %-16s %s"
                      % (prof.replace("AGFA_", ""), vessel, dev,
                         "  ".join("  -- " if v is None else "%5.1f" % v
                                   for v in vals)))
                chem = CHEM.get(printed, {}).get(dev)
                if chem is None or vals[1] is None:
                    continue
                (_d55, d65, _d75), t65 = chem
                want = d65 if vessel == "drum" else (
                    t65 if vessel == "small tank, tray" else None)
                if want is None:
                    continue
                hits += 1
                if abs(vals[1] - want) > 1e-9:
                    miss20 += 1
                    print("        [FAIL] 20 C reads %.1f, P-16-C prints %.1f "
                          "for this vessel" % (vals[1], want))
    print("      ⚠ THE 20 C COLUMN IS THE PROOF OF WHAT THE TABLE MEASURES: "
          "%d of %d cells reproduce P-16-C's printed gamma 0.65 time EXACTLY, "
          "across two vessels, five developers and three films. That is what "
          "licenses reading 18 / 22 / 24 C as the same contrast."
          % (hits - miss20, hits))
    if miss20 or hits < 30:
        print("      [FAIL] the 20 C cross-check no longer holds (%d cells, "
              "%d misses)" % (hits, miss20))
        bad += 1
    print("      ⚠⚠ AND «tanks» MATCHES NEITHER P-16-C COLUMN. REFINAL reads "
          "7 min there against 6 in trays and 5 in drums, on all three films. "
          "P-16-C section 3.1 lists a fourth method, «large tank»; that is a "
          "guess, so the block is stored under its own printed caption.")

    # ---- AGAINST THE DATABASE ---------------------------------------------
    # ⚠ THE POINT OF THIS BLOCK is that the module must be able to disagree
    # with `film_profiles.py`. It reads the source; the database is a separate
    # artefact that can drift from it.
    try:
        import film_profiles as fp
        miss = []
        for printed, devs in sorted(out.items()):
            pts = fp.get_profile(PROFILE[printed]).processing_family.points
            for dev, (a, _b, c) in sorted(devs.items()):
                for t, gam in ((a, 0.55), (c, 0.75)):
                    if t is None:
                        continue
                    hit = [q for q in pts
                           if q.developer == dev and q.vessel == "small tank, tray"
                           and abs(q.gamma - gam) < 1e-9
                           and abs(q.minutes - t) < 0.02]
                    if not hit:
                        miss.append("%s %s gamma %.2f at %.2f min"
                                    % (PROFILE[printed], dev, gam, t))
        print("\n  AGAINST THE DATABASE: %s"
              % ("every traced small-tank point is stored" if not miss
                 else "%d NOT STORED -- %s" % (len(miss), "; ".join(miss[:4]))))
        if miss:
            bad += 1
    except Exception as exc:                                  # pragma: no cover
        print("\n  [WARN] could not compare against film_profiles: %s" % exc)

    if ns.assert_ and bad:
        print("\n[FAIL] the AGFAPAN gamma-time panels do not reproduce")
        return 1
    print("\n[OK] three AGFAPAN gamma-time panels read, the vessel identified "
          "from fifteen independent comparisons against «Technical Data "
          "P-16-C», and the small tank's gamma 0.55 and 0.75 times -- which "
          "Agfa print in no table -- harvested from the curve.")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
