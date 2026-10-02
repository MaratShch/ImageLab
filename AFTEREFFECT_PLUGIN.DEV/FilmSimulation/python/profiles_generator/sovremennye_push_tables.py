#!/usr/bin/env python3
"""«Современные фотоматериалы и их обработка» -- the PUSH development tables,
read with their EXPOSURE INDEX (queue P98, 2026-09-30, owner decision).

WHY A SEPARATE READER. `sovremennye_dev_tables.py` reads a table's column
blocks as VESSELS («Малый бак» / «Большой бак»). The push tables use the same
two-block layout for a different axis -- «EI 1600 (Push-2)» / «EI 3200
(Push-3)» -- or state one EI in the caption («при экспозиционном индексе
EI 500», «(EI 1250)»), and two of them carry the VESSEL in the row direction
(«Малый бак, перемешивание ...», «Поддон, ...»). The general reader either
refused those tables (two blocks and no vessel caption) or stored their cells
at EI 0, where `_apply_sovremennye_2004`'s de-duplication then let a push time
displace, or be displaced by, the box-speed time at the same key.

Each table is described below by what its page prints, and the reader takes
every number from WORD COORDINATES: body cells are assigned to the nearest
temperature anchor of the header line, and a series is kept only when its time
falls (or holds) as the temperature rises. «Не рекомендуется» and «—» are
empty cells, never zero.

REFUSED, and why:
  * Табл. 3.199 XTOL rows: the two column blocks are two FILMS (PX / PXP) and
    the rows are two FORMATS (135 / 120); which cell is which film in which
    format is not determinable from the layout. Its T-MAX and T-MAX RS rows
    print the same times in both blocks and are kept once.
  * Табл. 3.211: percentage increases, not times.

Usage:  python sovremennye_push_tables.py --root <project root> [--dump]
        (as an audit: every point below must be in the database, at its EI)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TEMP = re.compile(r"^(\d{2})°[СC]\*?$")
CELL = re.compile(r"^(\d{1,2})(1/2|1/4|3/4)?$")
FR = {"1/2": 0.5, "1/4": 0.25, "3/4": 0.75}

#: table -> (page, stock, block axis, blocks, format, caption vessel)
#: block axis: "vessel" (blocks are vessels, EI from caption), "ei" (blocks
#: are exposure indices, vessel from caption), "one" (a single block).
TABLES = {
    "3.198": (351, "KODAK_PLUS_X_125", "vessel", (("small tank", 500), ("large tank", 500)), None),
    "3.199": (351, "KODAK_PLUS_X_125", "same", (("drum", 500), ("drum", 500)), None),
    "3.200": (351, "KODAK_PLUS_X_125", "one", (("drum", 500),), "sheet"),
    "3.212": (361, "KODAK_TRI_X_400TX", "ei", (("small tank", 1600), ("small tank", 3200)), None),
    "3.213": (361, "KODAK_TRI_X_400TX", "ei", (("large tank", 1600), ("large tank", 3200)), None),
    "3.214": (361, "KODAK_TRI_X_400TX", "ei", (("drum", 1600), ("drum", 3200)), None),
    "3.247": (394, "KODAK_TRI_X_400TX", "ei", (("small tank", 1600), ("small tank", 3200)), None),
    "3.248": (395, "KODAK_TRI_X_400TX", "ei", (("large tank", 1600), ("large tank", 3200)), None),
    "3.249": (395, "KODAK_TRI_X_400TX", "ei", (("drum", 1600), ("drum", 3200)), None),
    # blocks are FORMATS, the EI is a row sub-header «EI 500 (Push-2 процесс)»
    "3.227": (381, "KODAK_PLUS_X_125", "fmt", (("small tank", "135"), ("small tank", "120")), None),
    "3.228": (381, "KODAK_PLUS_X_125", "fmt", (("large tank", "135"), ("large tank", "120")), None),
    "3.229": (382, "KODAK_PLUS_X_125", "fmt", (("drum", "135"), ("drum", "120")), None),
    "3.250": (395, "KODAK_TRI_X_320TXP", "rows", (("", 1250),), "120"),
    "3.251": (395, "KODAK_TRI_X_320TXP", "rows", (("", 1250),), "sheet"),
}
#: row-direction vessel captions (Табл. 3.250 / 3.251)
ROW_VESSEL = (("Малый", "small tank"), ("Большой", "large tank"),
              ("Роторно-барабанный", "drum"), ("Поддон", "tray"))
#: developer labels as the database spells them
LABEL = {"T-MAX": ("T-MAX", ""), "T-MAX RS": ("T-MAX RS", ""), "D-76": ("D-76", ""),
         "D-76 (1:1)": ("D-76", "1:1"), "XTOL": ("XTOL", ""), "XTOL (1:1)": ("XTOL", "1:1"),
         "HC-110 (Dil B)": ("HC-110", "Dil B")}
STOP = ("Примечание", "Окончательная", "Табл.")


def _rows(page, tol=3.5):
    rows = {}
    for x0, y0, x1, y1, t, *_ in page.get_text("words"):
        t = t.strip()
        if not t:
            continue
        k = next((k for k in rows if abs(k - y0) <= tol), y0)
        rows.setdefault(k, []).append((0.5 * (x0 + x1), t))
    return [(y, sorted(v)) for y, v in sorted(rows.items())]


def _merge_fractions(words):
    """«8» «3/4» set as two words (Табл. 3.247 XTOL) -> «83/4»."""
    out = []
    for x, t in words:
        if out and t in FR and CELL.match(out[-1][1]) and x - out[-1][0] < 16:
            out[-1] = (out[-1][0] + 6, out[-1][1] + t)
        else:
            out.append((x, t))
    return out


def _cell(t):
    m = CELL.match(t)
    return None if not m else float(m.group(1)) + FR.get(m.group(2) or "", 0.0)


def read_table(doc, tid):
    pg, stock, axis, blocks, fmt = TABLES[tid]
    rows = _rows(doc[pg - 1])
    i0 = next(i for i, (y, w) in enumerate(rows)
              if any(t == "Табл." for _, t in w) and any(t.startswith(tid + ".") for _, t in w))
    ih = next(i for i in range(i0 + 1, len(rows))
              if sum(1 for _, t in rows[i][1] if TEMP.match(t)) >= 5)
    anchors = [(x, int(TEMP.match(t).group(1))) for x, t in rows[ih][1] if TEMP.match(t)]
    nb = len(blocks)
    if len(anchors) != 5 * nb:
        raise ValueError("%s: %d temperature anchors for %d blocks" % (tid, len(anchors), nb))
    # every block of anchors must be strictly increasing
    for b in range(nb):
        ts = [a[1] for a in anchors[5 * b:5 * b + 5]]
        if ts != sorted(set(ts)):
            raise ValueError("%s: block %d header %s" % (tid, b, ts))
    if axis == "fmt":
        fl = " ".join(t for i in (ih - 1, ih - 2, ih - 3) for _, t in rows[i][1])
        if re.findall(r"(135|120)\s+формат", fl) != [f for _, f in blocks]:
            raise ValueError("%s: format header %r" % (tid, fl))
    elif axis == "ei":
        ei_line = " ".join(t for i in (ih - 1, ih - 2, ih - 3) for _, t in rows[i][1])
        eis = [int(v) for v in re.findall(r"EI\s+(\d{3,4})", ei_line)]
        if eis != [e for _, e in blocks]:
            raise ValueError("%s: EI header %s, expected %s" % (tid, eis, [e for _, e in blocks]))
    else:
        cap = " ".join(t for i in range(i0, ih) for _, t in rows[i][1])
        eis = re.findall(r"EI\s*(\d{3,4})", cap)
        if [int(e) for e in eis] != [blocks[0][1]]:
            raise ValueError("%s: caption EI %s" % (tid, eis))
    pts, prev_name, row_vessel, row_ei = [], "", "", 0
    for y, w in rows[ih + 1:]:
        words = _merge_fractions(w)
        text = " ".join(t for _, t in words)
        if any(text.startswith(s) for s in STOP):
            break
        if axis == "fmt":
            me = re.match(r"^EI\s+(\d{3,4})\s+\(Push", text)
            if me:
                row_ei = int(me.group(1))
                continue
        if axis == "rows":
            rv = next((v for ru, v in ROW_VESSEL if text.startswith(ru)), None)
            if rv:
                row_vessel = rv
                continue
        name_w = [t for x, t in words if x < anchors[0][0] - 20 and _cell(t) is None
                  and t not in ("Не", "рекомендуется", "рекомен-", "дуется", "—", "-")]
        name = " ".join(name_w).strip().rstrip("*")
        cells = [(x, t) for x, t in words if x >= anchors[0][0] - 20]
        if not any(_cell(t) is not None for _, t in cells):
            if name in LABEL:
                prev_name = name        # «XTOL» above its format sub-rows
            continue
        row_fmt = fmt
        mf = re.match(r"^PXP?\s*[–-]\s*(135|120)$", name)
        if mf:                          # «PX – 135 формат» under «XTOL»
            name, row_fmt = prev_name, mf.group(1)
        if name not in LABEL:
            raise ValueError("%s: unknown row label %r" % (tid, name))
        got = {}
        for x, t in cells:
            v = _cell(t)
            if v is None:
                continue            # «—», «Не», «рекомендуется»
            k = min(range(len(anchors)), key=lambda j: abs(anchors[j][0] - x))
            if abs(anchors[k][0] - x) > 20 or k in got:
                raise ValueError("%s: cell %s at %.0f unplaced" % (tid, t, x))
            got[k] = v
        dev, dil = LABEL[name]
        per_block = {}
        for k, v in got.items():
            per_block.setdefault(k // 5, []).append((anchors[k][1], v))
        if axis == "same":
            if len(per_block) != 2 or sorted(per_block[0]) != sorted(per_block[1]):
                continue            # refused: see the module docstring
            per_block = {0: per_block[0]}
        for b, series in per_block.items():
            series.sort()
            if any(m1 > m0 + 1e-9 for (_, m0), (_, m1) in zip(series, series[1:])):
                raise ValueError("%s: %s block %d not falling: %s" % (tid, name, b, series))
            vessel, ei = blocks[b]
            if axis == "rows":
                vessel = row_vessel
            if axis == "fmt":
                vessel, row_fmt, ei = blocks[b][0], blocks[b][1], row_ei
                if not ei:
                    raise ValueError("%s: a row before any EI sub-header" % tid)
            for c, m in series:
                pts.append((stock, dev, dil, m, c, vessel, row_fmt or "", ei, tid))
        prev_name = name
    return pts


def read_all(pdf):
    import pymupdf
    doc = pymupdf.open(pdf)
    out = []
    for tid in TABLES:
        out += read_table(doc, tid)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args(argv)
    pdf = Path(a.root) / "PDF" / "PROFILES" / "SOVIET" / "Современные фотоматериалы и их обработка.pdf"
    if not pdf.is_file():       # the staged copy, as sovremennye_2004.py reads it
        pdf = Path("/mnt/user-data/uploads/PYTHON.TST/PDF/PROFILES/SOVIET/"
                   "Современные фотоматериалы и их обработка.pdf")
    pts = read_all(pdf)
    if a.dump:
        for p in pts:
            print(p)
    import film_profiles as fp
    held = {tuple(r[:8]) for r in fp.SOVREMENNYE_2004_PUSH_POINTS}
    read = {p[:8] for p in pts}
    miss = sorted(read - held)
    extra = sorted(held - read)
    stored_ok = []
    for st, dev, dil, m, c, vessel, fmt, ei in read:
        prof = fp.get_profile(st)
        stored_ok.append(any(q.developer == dev and q.dilution == dil and abs(q.minutes - m) < 1e-9
                             and q.celsius == c and q.vessel == vessel and q.film_format == fmt
                             and q.exposure_index == ei for q in prof.processing_family.points))
    ok = not miss and not extra and all(stored_ok)
    print("%s  %d push cells re-read from %d tables; literal holds %d; %d missing, %d extra; "
          "%d of %d on their profiles at their EI"
          % ("PASS" if ok else "FAIL", len(read), len(TABLES), len(held), len(miss), len(extra),
             sum(stored_ok), len(stored_ok)))
    for r in (miss + extra)[:8]:
        print("   ", r)
    print("[%s] sovremennye_push_tables.py -- 1 check, %d failed" % ("OK" if ok else "FAIL", 0 if ok else 1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
