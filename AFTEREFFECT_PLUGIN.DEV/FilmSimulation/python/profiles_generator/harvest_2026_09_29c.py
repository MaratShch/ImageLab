"""Audit of the 2026-09-29c harvest.

Re-derives, from the documents themselves, the numbers the 2026-09-29c batch
wrote into `film_profiles.py`, and fails on drift. Four documents:

  FUJI/datasheet_neopan400presto120_01.pdf   NEOPAN 400 PRESTO (120), Ref. 163AR0121A (2007)
  28bwfilmscompared.pdf                      Popular Photography 2003, «28 B&W Films Compared!»
  1427.pdf                                   Classic Camera / Black & White N.92 (2014)
  SOVIET/Современные фотоматериалы и их обработка.pdf   (the label repair; optional)

Usage:  python harvest_2026_09_29c.py --root <project root> [--assert]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import film_profiles as fp  # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def chk(ok: bool, label: str, detail: str = "") -> None:
    RESULTS.append((bool(ok), label, detail))
    print("%s  %s   %s" % ("PASS" if ok else "FAIL", label, detail))


def _bez(P, n=40):
    s = np.linspace(0.0, 1.0, n)[:, None]
    A = np.array([[p.x, p.y] for p in P])
    return (1 - s) ** 3 * A[0] + 3 * (1 - s) ** 2 * s * A[1] + 3 * (1 - s) * s * s * A[2] + s ** 3 * A[3]


def _path_points(d):
    out = []
    for it in d["items"]:
        if it[0] == "c":
            out.append(_bez(it[1:5]))
        elif it[0] == "l":
            out.append(np.array([[it[1].x, it[1].y], [it[2].x, it[2].y]]))
    return np.vstack(out)


def _frac(tok: str):
    """Fuji Japan's «41/4» = 4 1/4, «111/4» = 11 1/4; «－» = not given."""
    tok = tok.strip()
    if tok in ("－", "-", "─"):
        return None
    m = re.fullmatch(r"(\d+?)([123])/([24])", tok)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    return float(tok)


# ---------------------------------------------------------------------------
def presto(pdf: Path) -> None:
    import pymupdf
    doc = pymupdf.open(pdf)
    p1, p2, p3, p4 = (doc[i].get_text() for i in range(4))
    chk("ネオパン 400 PRESTO" in p1 and "163AR0121A" in doc[3].get_text()
        and "0.104mm" in p1.replace(" ", ""),
        "PRESTO: the sheet is Ref. 163AR0121A, 120 only, TAC 0.104 mm",
        "the 120 base thickness AF3-706E prints for NEOPAN 400")
    # -- the development tables, row by row in document order --------------
    toks = [t.strip() for t in p2.split("\n") if t.strip()]
    rows, i = [], 0
    while i < len(toks) - 5:
        if toks[i] in ("200", "250", "320", "400", "800", "1600"):
            try:
                vals = [_frac(t) for t in toks[i + 1:i + 6]]
                if all(v is None or 0 < v < 30 for v in vals):
                    rows.append((int(toks[i]), tuple(vals)))
                    i += 6
                    continue
            except ValueError:
                pass
        i += 1
    fuji = [(d, dl, ei, ms) for d, dl, ei, ms in fp._PRESTO_TABLE if d != "Microdol-X"]
    shared = list(fp.NEOPAN_400_PRESTO_SHARED)
    mdx400 = [r for r in fp._PRESTO_TABLE if r[0] == "Microdol-X"]
    want = [(ei, ms) for _, _, ei, ms in fuji] + [(ei, ms) for _, _, ei, ms in shared[:18]] \
        + [(ei, ms) for _, _, ei, ms in shared[18:22]] + [(shared[22][2], shared[22][3]), (mdx400[0][2], mdx400[0][3])]
    got = [(ei, tuple(ms)) for ei, ms in rows[:len(want)]]
    chk(got == [(ei, tuple(ms)) for ei, ms in want],
        "PRESTO: every stored development row equals the sheet's p2 tables cell for cell",
        "%d rows parsed, %d expected" % (len(rows), len(want)))
    np4 = fp.get_profile("FUJI_NEOPAN_400")
    held = {(q.developer, q.dilution, q.exposure_index, q.celsius): q.minutes
            for q in np4.processing_family.points
            if q.film_format == "120" and not q.contrast_index and q.vessel == "small tank"}
    n = same = 0
    for d, dl, ei, ms in shared:
        for t, m in zip(fp._PT, ms):
            if m is None:
                continue
            n += 1
            same += held.get((d, dl, ei, t)) == m
    chk(n == 110 and same == 110,
        "PRESTO: its 110 third-party cells equal AF3-207U p38 [120] already held",
        "%d of %d identical" % (same, n))
    mdx = [held.get(("Microdol-X", "stock", 400, t)) for t in fp._PT]
    chk(mdx == [12.0, 10.0, 8.5, 7.0, 6.0],
        "PRESTO: the Microdol-X EI 400 row the guide lacks is now held", str(mdx))
    rec = fp.get_profile("FUJI_NEOPAN_400").reciprocity_table
    flat = p1.replace(" ", "").replace("\n", "")
    chk("1/2秒より短い場合は補正の必要はありません" in flat
        and "1/2絞り開く1絞り開く2絞り開く" in flat
        and rec.times_s == (0.5, 1.0, 10.0, 100.0)
        and rec.stops_correction == (0.0, 0.5, 1.0, 2.0),
        "PRESTO: the reciprocity ladder is stored as printed",
        "none < 1/2 s; 1 s +1/2, 10 s +1, 100 s +2")
    labels = re.findall(r"G＝(\d\.\d\d)", p3)
    chk(sorted(labels) == sorted(["0.53", "0.41", "0.64", "0.82", "0.65", "0.53", "0.83", "0.64", "0.54"])
        and sorted(g for _, _, g in fp._PRESTO_GBAR) == [0.41, 0.53, 0.53, 0.64, 0.65, 0.82],
        "PRESTO: the six adopted G-bar labels are among the nine printed", " ".join(labels))
    # -- same emulsion: the D-76 7:30 curve against the stored 135 curve ----
    page = doc[2]
    X0, Y0, X1, Y1 = 92.852, 637.317, 274.270, 746.168
    frame = pymupdf.Rect(X0, Y0, X1, Y1)
    runs = [d for d in page.get_drawings()
            if abs((d.get("width") or 0) - 0.75) < 0.02 and frame.intersects(d["rect"])]
    P = _path_points(runs[0])
    x = -3.5 + 5.0 * (P[:, 0] - X0) / (X1 - X0)
    y = 3.0 * (Y1 - P[:, 1]) / (Y1 - Y0)
    # three strokes share one path; the lowest-contrast one ends lowest
    breaks = np.where(np.abs(np.diff(x)) > 1.0)[0]
    strokes = np.split(np.column_stack([x, y]), breaks + 1)
    low = min(strokes, key=lambda s: s[:, 1].max())
    grid = np.linspace(-2.8, 0.5, 34)
    o = np.argsort(low[:, 0])
    pr = np.interp(grid, low[o, 0], low[o, 1])
    q = fp.get_profile("FUJI_NEOPAN_400").curves.g
    sp = lambda z, k: k * np.log1p(np.exp(np.clip(z / k, -40, 40)))
    st = q.dmin + q.gamma * (sp(grid - q.toe_x, q.toe_k) - sp(grid - q.shoulder_x, q.shoulder_k))
    d = pr - st
    chk(-0.16 < d.mean() < -0.09 and d.std() < 0.04,
        "PRESTO: its 7:30 D-76 curve is the stored 135 curve less the grey base",
        "offset %.3f D, spread %.3f" % (d.mean(), d.std()))
    # -- spectral: shape against the stored log_s_pan ----------------------
    sp4 = doc[3]
    spec = [d for d in sp4.get_drawings() if abs((d.get("width") or 0) - 0.75) < 0.02
            and abs(d["rect"].x0 - 94.7) < 0.3]
    S = _path_points(spec[0])
    wl = 400.0 + (S[:, 0] - 99.04) / 0.49315
    ls = -S[:, 1] / 39.66
    o = np.argsort(wl)
    ss = fp.get_profile("FUJI_NEOPAN_400").spectral
    L = np.array(ss.log_s_pan)
    W = ss.lambda_start_nm + ss.lambda_step_nm * np.arange(len(L))
    m = (W >= 400) & (W <= 630)
    dd = np.interp(W[m], wl[o], ls[o]) - L[m]
    dd -= dd.mean()
    chk(np.sqrt(np.mean(dd ** 2)) < 0.04,
        "PRESTO: its daylight spectral curve is the stored NEOPAN 400 curve",
        "rms %.3f log over 400-630 nm after the free offset" % np.sqrt(np.mean(dd ** 2)))
    # -- the two new laws against the vector time-G curves -----------------
    X0, Y1 = 337.64, 236.92
    sx, sy = (519.07 - 337.64) / 20.0, (236.92 - 128.04) / 1.2
    fam = fp.get_profile("FUJI_NEOPAN_400").processing_family
    worst = 0.0
    for d in sp4.get_drawings():
        r = d["rect"]
        if abs((d.get("width") or 0) - 0.75) > 0.02 or r.x0 < 370 or r.y1 > 210:
            continue
        T = _path_points(d)
        t = (T[:, 0] - X0) / sx
        g = (Y1 - T[:, 1]) / sy
        dev = "SPD [Super Prodol]" if g.max() > 0.9 else ("D-76" if g.max() > 0.8 else "Microfine")
        if dev == "D-76":
            continue
        law = fam.law_for(dev, vessel="small tank")
        pred = np.array([law.gamma_at(v) for v in t])
        worst = max(worst, float(np.sqrt(np.mean((pred - g) ** 2))))
    chk(worst < 0.01, "PRESTO: the SPD and Microfine laws reproduce Fuji's drawn time-G curves",
        "worst rms %.4f G" % worst)


# ---------------------------------------------------------------------------
def popphoto(pdf: Path) -> None:
    import pymupdf
    t = "\n".join(p.get_text() for p in pymupdf.open(pdf))
    flat = re.sub(r"\s+", " ", t)
    chk("all data presented in the fol" in flat and "came from the ﬁlm manufacturers" in flat,
        "POPPHOTO: the chart states its data are the manufacturers'")
    probes = {
        "FUJI_NEOPAN_400": r"400 NEOPAN 125 10 ",
        "FUJI_NEOPAN_1600": r"1600 NEOPAN 100 16 ",
        "KODAK_TECHNICAL_PAN": r"25 TP 320 5 ",
        "KODAK_TRI_X_320TXP": r"320TXP 100 16",
        "KODAK_T400CN": r"T400 CN NA 9",
    }
    bad = [n for n, pat in probes.items() if pat not in flat]
    chk(not bad, "POPPHOTO: the five adopted stocks' chart cells read as stored", ", ".join(bad) or "5 of 5")
    rms_bad = ["%s %s!=%s" % (n, fp.get_profile(n).grain.rms_granularity, v)
               for n, v in fp.POPPHOTO_2003_RMS.items()
               if fp.get_profile(n).grain.rms_granularity != v]
    rp_bad = ["%s" % n for n, v in fp.POPPHOTO_2003_RP.items()
              if fp.get_profile(n).mtf.resolving_power_lp_mm_highc != v]
    chk(not rms_bad and not rp_bad, "POPPHOTO: stored rms and 1000:1 resolving power equal the chart",
        "; ".join(rms_bad + rp_bad) or "5 rms, 4 resolving powers")
    agree = {"AGFA_APX_100": 150, "AGFA_APX_400": 110, "FUJI_NEOPAN_ACROS_100": 200,
             "KODAK_PLUS_X_125": 125, "KODAK_TMAX_100": 200, "KODAK_TMAX_400": 125,
             "KODAK_TMAX_P3200": 125, "KODAK_TRI_X_400TX": 100, "AGFA_SCALA_200X": 120}
    off = [n for n, v in agree.items() if fp._RESOLVING_POWER[n][1] != v]
    chk(not off, "POPPHOTO: its resolution column equals the stored 1000:1 figure on all nine shared stocks",
        ", ".join(off) or "9 of 9 -- the column is the high-contrast one")
    chk(abs(fp.get_profile("KODAK_PLUS_X_125").grain.rms_granularity - 9.51) < 1e-9,
        "POPPHOTO: PLUS-X keeps its tier-1 9.51 against the chart's 10 (conflict logged, not overwritten)")


# ---------------------------------------------------------------------------
_CC14_BITMAPS = {  # xref: md5 of the embedded table images the rows were read from
    95: "Ilford powder developers on non-Ilford films (p8, lower)",
    93: "Ilford powder developers on Ilford films (p8, upper)",
    89: "Kodak F-4017 Tri-X push table (p7, lower)",
}


def classic_camera(pdf: Path) -> None:
    import pymupdf
    doc = pymupdf.open(pdf)
    t = re.sub(r"\s+", " ", "\n".join(doc[i].get_text() for i in range(2, 17)))
    chk("RMS pari a 9" in t and "140 lp/mm" in t,
        "CLASSIC CAMERA: prints NEOPAN 400 RMS 9 / 140 lp/mm (the logged conflict with PopPhoto's 10 / 125)")
    chk("Ilford ID-11 non diluito" in t and "sviluppo in ID-11 non diluito, 7 minuti" in t.replace("  ", " "),
        "CLASSIC CAMERA: the test itself used ID-11 stock (7 min for Tri-X)")
    sizes = {x: (doc.extract_image(x)["width"], doc.extract_image(x)["height"]) for x in _CC14_BITMAPS}
    chk(sizes == {95: (699, 667), 93: (701, 738), 89: (924, 392)},
        "CLASSIC CAMERA: the three table bitmaps are the ones read", str(sizes))
    add = fp.DEV_HARVEST_0929C_ADDED
    n_ilf = len(fp._ILFORD_POWDER_ROWS)
    n_push = sum(1 for _, _, _, ms in fp._TRI_X_PUSH_ROWS for m in ms if m is not None)
    chk(n_ilf == 113 and n_push == 58,
        "CLASSIC CAMERA: 113 Ilford-table cells and 58 Tri-X push cells transcribed",
        "%d / %d" % (n_ilf, n_push))
    chk(sum(add.values()) == 225 and add.get("ILFORD_DELTA_3200") == 17,
        "CLASSIC CAMERA + PRESTO: 225 points appended (exact twins skipped); DELTA 3200 gains its first 17",
        str(dict(sorted(add.items()))))
    tx = [q for q in fp.get_profile("KODAK_TRI_X_400TX").processing_family.points
          if q.exposure_index in (1600, 3200) and q.vessel == "small tank"
          # 2026-09-30 (P98): the book's Табл. 3.212 adds 24 small-tank push
          # cells for the OLDER «TRI-X Pan / TX», tagged by `edition`
          and not q.edition]
    chk(len(tx) == 58, "CLASSIC CAMERA: TRI-X 400 now carries the EI 1600 / 3200 push times", "%d points" % len(tx))


# ---------------------------------------------------------------------------
def sovremennye(pdf: Path | None) -> None:
    bad = [(k, r[0]) for k, rows in fp._SOVREMENNYE_2004_POINTS.items() for r in rows
           if any("Ѐ" <= c <= "ӿ" for c in r[0]) or "—" in r[0]]
    chk(not bad, "СОВРЕМЕННЫЕ: no stored developer label carries a Cyrillic letter or a refusal dash",
        "%d left" % len(bad))
    tx = [r for r in fp._SOVREMENNYE_2004_POINTS["KODAK_TRI_X_400TX"] if r[0] == "MICRODOL-X" and r[1] == "1:3"]
    chk(sorted((r[4], r[3], r[2]) for r in tx) == sorted(
        [("small tank", 18, 18.75), ("small tank", 20, 17), ("small tank", 21, 16), ("small tank", 22, 15),
         ("small tank", 24, 13.5), ("large tank", 20, 19.5), ("large tank", 21, 18.25),
         ("large tank", 22, 17.25), ("large tank", 24, 15.5)]),
        "СОВРЕМЕННЫЕ: Табл. 3.241's «Не» row is MICRODOL-X (1:3), and equals F-4017's row as reproduced in 1427")
    if pdf is None:
        print("SKIP  СОВРЕМЕННЫЕ page re-read: book not present")
        return
    import pymupdf
    import sovremennye_dev_tables as R
    doc = pymupdf.open(pdf)
    held = {}
    for k, rows in fp._SOVREMENNYE_2004_POINTS.items():
        for r in rows:
            held.setdefault((r[2], float(r[3])), set()).add(k)
    miss = 0
    tot = 0
    for pg in (349, 350, 360, 380, 387, 392):
        for tab in R.read_page(doc[pg - 1]):
            got = R.table_points(tab)
            for name, vessel, m, c in got or ():
                if any("Ѐ" <= ch <= "ӿ" for ch in name) or "—" in name:
                    tot += 1
                    miss += (m, c) not in held
    chk(tot > 0 and miss == 0,
        "СОВРЕМЕННЫЕ: every corrupted-label cell the reader still emits is held under a clean label",
        "%d cells, %d unplaced (Табл. 3.250's EI 1250 push rows are dropped by design)" % (tot, miss))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--assert", dest="strict", action="store_true")
    a = ap.parse_args(argv)
    base = Path(a.root) / "PDF" / "PROFILES"
    presto(base / "FUJI" / "datasheet_neopan400presto120_01.pdf")
    popphoto(base / "28bwfilmscompared.pdf")
    classic_camera(base / "1427.pdf")
    sv = base / "SOVIET" / "Современные фотоматериалы и их обработка.pdf"
    sovremennye(sv if sv.is_file() else None)
    n_bad = sum(1 for ok, _, _ in RESULTS if not ok)
    print("[%s] harvest_2026_09_29c.py -- %d checks, %d failed"
          % ("OK" if n_bad == 0 else "FAIL", len(RESULTS), n_bad))
    return 1 if (a.strict and n_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
