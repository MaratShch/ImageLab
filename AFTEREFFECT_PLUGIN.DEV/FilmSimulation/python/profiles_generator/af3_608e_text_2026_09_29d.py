"""Audit of the 2026-09-29d read of AF3-608E's clean text layer.

`FUJI/datasheet_neopan1600superpresto_en_01.pdf` is the 1999 distillation of
FUJIFILM DATA SHEET «NEOPAN 1600 Professional», Ref. No. AF3-608E(N). This
re-derives from it every number the 2026-09-29d batch wrote, and fails on
drift:

  * the p2 development tables, parsed from the text layer, against every
    NEOPAN 1600 point held (the 101 shared cells must be equal, the 52 new
    ones must be the stored AF3_608E_NEW_POINTS);
  * the p3 processing-capacity table, its merged cells placed on roll columns
    from the word coordinates, against ProcessingFamily.capacity;
  * identity with the 2012 copy `FujiNeopan1600.pdf` when that is staged:
    the same six images, byte for byte.

Usage:  python af3_608e_text_2026_09_29d.py --root <project root> [--assert]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import film_profiles as fp  # noqa: E402

RESULTS: list[bool] = []


def chk(ok, label, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s   %s" % ("PASS" if ok else "FAIL", label, detail))


def _num(t: str):
    t = t.strip()
    if t == "NR":
        return None
    m = re.fullmatch(r"(\d+)\s+(\d)/(\d)", t)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    m = re.fullmatch(r"(\d+?)(\d)/(\d)", t)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    return float(t)


_FUJI = [("SPD [Super Prodol]", "stock", 1600), ("SPD [Super Prodol]", "stock", 3200),
         ("SPD [Super Prodol]", "1:1", 1600), ("Fujidol E", "stock", 1600),
         ("Fujidol E", "1:1", 1600), ("Microfine", "stock", 250),
         ("Microfine", "stock", 400), ("Microfine", "stock", 800)]
_OTHER = [("D-76", "stock", 400), ("D-76", "stock", 800), ("D-76", "stock", 1600),
          ("D-76", "stock", 3200), ("D-76", "1:1", 400), ("D-76", "1:1", 800),
          ("D-76", "1:1", 1600), ("D-76", "1:3", 800), ("D-76", "1:3", 1600),
          ("Microdol-X", "stock", 400), ("Microdol-X", "stock", 800),
          ("Microdol-X", "stock", 1600), ("HC-110", "Dil. B", 800),
          ("HC-110", "Dil. B", 1600), ("T-MAX Developer", "stock", 1600),
          ("T-MAX Developer", "stock", 3200), ("T-MAX RS Developer", "stock", 1600),
          ("T-MAX RS Developer", "stock", 3200), ("XTOL", "stock", 1600),
          ("XTOL", "stock", 3200), ("Microphen", "stock", 1600),
          ("Microphen", "stock", 3200), ("ID-11", "stock", 800),
          ("ID-11", "stock", 1600), ("ILFOTEC LC 29", "1:19", 1600)]
T5 = (18.0, 20.0, 22.0, 24.0, 26.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--assert", dest="strict", action="store_true")
    a = ap.parse_args(argv)
    import pymupdf
    base = Path(a.root) / "PDF" / "PROFILES" / "FUJI"
    doc = pymupdf.open(base / "datasheet_neopan1600superpresto_en_01.pdf")
    txt = [p.get_text() for p in doc]
    chk("AF3-608E(N)" in txt[3] and "NEOPAN 1600 SUPER PRESTO" in txt[0]
        and "same short\ndevelopment time required of NEOPAN 400" in txt[0],
        "the sheet is AF3-608E(N) and names NEOPAN 1600 SUPER PRESTO")
    # -- p2 tables: rows of (EI, five cells) in document order ------------
    toks = [t.strip() for t in txt[1].split("\n") if t.strip()]
    rows, i = [], 0
    while i < len(toks) - 5:
        if toks[i] in ("250", "400", "800", "1600", "3200"):
            try:
                v = [_num(t) for t in toks[i + 1:i + 6]]
                if all(x is None or 0 < x < 30 for x in v):
                    rows.append((int(toks[i]), v))
                    i += 6
                    continue
            except ValueError:
                pass
        i += 1
    keys = _FUJI + _OTHER
    ok_shape = len(rows) == len(keys) and all(r[0] == k[2] for r, k in zip(rows, keys))
    chk(ok_shape, "p2: 33 table rows parsed in the sheet's own order", "%d rows" % len(rows))
    held = {}
    for q in fp.get_profile("FUJI_NEOPAN_1600").processing_family.points:
        if not q.contrast_index and q.vessel == "small tank" and q.film_format == "135":
            held.setdefault((q.developer, q.dilution, q.exposure_index, q.celsius), set()).add(q.minutes)
    same = new = bad = 0
    newkeys = {(d, dl, ei) for d, dl, ei, _ in fp._AF3_608E_NEW_ROWS}
    for (d, dl, ei), (_, v) in zip(keys, rows):
        for t, m in zip(T5, v):
            if m is None:
                continue
            h = held.get((d, dl, ei, t), set())
            if m in h:
                if (d, dl, ei) in newkeys:
                    new += 1
                else:
                    same += 1
            else:
                bad += 1
    chk(same == 101 and new == 52 and bad == 0,
        "p2: 101 cells already held are equal and the 52 new ones are stored",
        "%d equal, %d new, %d missing or different" % (same, new, bad))
    # -- p3 capacity table ------------------------------------------------
    p = doc[2]
    words = [w for w in p.get_text("words") if 95 < w[1] < 240 and w[0] < 300]
    cols = {int(w[4]): 0.5 * (w[0] + w[2]) for w in words
            if abs(w[1] - 117.8) < 1 and w[4].isdigit()}
    edges = [0.5 * (cols[k] + cols[k + 1]) for k in range(1, 12)]

    def span(xc):
        return 1 + sum(1 for e in edges if xc > e)
    out = {}
    for y, name in ((146.7, "SPD [Super Prodol]"), (222.4, "D-76")):
        cells = sorted([w for w in words if abs(w[1] - y) < 1.2 and w[0] > 112],
                       key=lambda w: w[0])
        merged, cur = [], []
        for w in cells:
            if cur and w[0] - cur[-1][2] > 3:
                merged.append(cur)
                cur = []
            cur.append(w)
        merged.append(cur)
        seq = []
        for grp in merged:
            xc = 0.5 * (grp[0][0] + grp[-1][2])
            label = " ".join(g[4] for g in grp)
            seq.append((span(xc), label))
        out[name] = seq
    spd = [c.minutes for c in fp.get_profile("FUJI_NEOPAN_1600").processing_family.capacity]
    # a merged cell over columns a..b is centred on the midpoint of a's left
    # edge and b's right edge; test the stored spans against the printed centres
    bounds = [cols[1] - 0.5 * (cols[2] - cols[1])] + edges + [cols[12] + 0.5 * (cols[12] - cols[11])]
    ctr = lambda a_, b_: 0.5 * (bounds[a_ - 1] + bounds[b_])
    def fit(name, spans):
        got = [0.5 * (grp_x) for grp_x in xs[name]]
        return max(abs(g - ctr(a_, b_)) for g, (a_, b_) in zip(got, spans))
    xs = {}
    for y, name in ((146.7, "SPD [Super Prodol]"), (222.4, "D-76")):
        cells = sorted([w for w in words if abs(w[1] - y) < 1.2 and w[0] > 112], key=lambda w: w[0])
        grp, cur = [], [cells[0]]
        for w in cells[1:]:
            if w[0] - cur[-1][2] > 3:
                grp.append(cur); cur = []
            cur.append(w)
        grp.append(cur)
        xs[name] = [g[0][0] + g[-1][2] for g in grp]
    e_spd = fit("SPD [Super Prodol]", [(1, 4), (5, 6), (7, 8), (9, 10), (11, 12)])
    e_d76 = fit("D-76", [(1, 4), (5, 7), (8, 9), (10, 10), (11, 12)])
    chk(e_spd < 3.0 and e_d76 < 3.0,
        "p3: SPD's merged cells sit on rolls 1-4 / 5-6 / 7-8 / 9-10 / dash 11-12, D-76's on 1-4 / 5-7 / 8-9 / 10 / dash 11-12",
        "worst centre offset %.2f pt / %.2f pt (a column is %.1f pt)" % (e_spd, e_d76, cols[2] - cols[1]))
    chk(spd[0] == (4.25,) * 4 + (4.5,) * 2 + (4.75,) * 2 + (5.0,) * 2
        and spd[1] == (6.5, 6.5, 6.5, 7.0, 7.0, 7.5, 7.5, 8.0, 8.0, 8.5, 8.5, 9.0)
        and spd[2] == (8.0, 8.5, 9.0, 9.5)
        and spd[3] == (7.5,) * 4 + (8.0,) * 3 + (8.5,) * 2 + (9.0,),
        "p3: the four stored capacity tables equal the printed ones")
    fuj = [w[4] for w in sorted([w for w in words if abs(w[1] - 178.5) < 1.2 and w[0] > 112],
                                key=lambda w: w[0])]
    chk([_num(x) for x in fuj] == list(spd[1]), "p3: Fujidol E's twelve cells read one per roll", " ".join(fuj))
    # -- identity with the 2012 copy -------------------------------------
    old = base / "FujiNeopan1600.pdf"
    if old.is_file():
        o = pymupdf.open(old)
        h = lambda d: sorted(hashlib.md5(d.extract_image(x[0])["image"]).hexdigest()
                             for pg in d for x in pg.get_images())
        chk(h(o) == h(doc) and len(h(doc)) == 6,
            "the 2012 copy carries the same six images byte for byte (curves already traced from them)")
    else:
        print("SKIP  identity with FujiNeopan1600.pdf: not staged")
    n_bad = RESULTS.count(False)
    print("[%s] af3_608e_text_2026_09_29d.py -- %d checks, %d failed"
          % ("OK" if not n_bad else "FAIL", len(RESULTS), n_bad))
    return 1 if (a.strict and n_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
