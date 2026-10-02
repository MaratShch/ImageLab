#!/usr/bin/env python3
"""«Современные фотоматериалы и их обработка» -- the T-MAX push tables and the
Kodak chapter-5 processing tables, read WITH THEIR EXPOSURE INDEX (queue P98b,
2026-10-01, owner decision).

WHY A THIRD READER. `sovremennye_dev_tables.py` (P54) reads column blocks as
VESSELS and `sovremennye_push_tables.py` (P98) reads a fixed list of Plus-X /
Tri-X layouts from word coordinates. The tables here use five more layouts:

  A  two films side by side, an EI span per column block, a temperature per
     column (Табл. 3.164-3.168, 3.180-3.182) -- and the same with no EI row,
     at box speed (3.183, 3.184);
  B  the same two films STACKED, each with its own header (3.185, 3.186);
  C  one film, developer in column 0 spanning several EI ROWS in column 1
     (the P3200 tables 3.169-3.171, 3.187-3.189, 3.187 continuing overleaf);
  D  many films, film in column 0 spanning EI rows in column 1, one developer
     per section (5.149-5.154 T-MAX RS, 5.158-5.159 T-MAX; 5.167 D-76);
  E  many films, two VESSEL blocks per row, no EI (5.162, 5.164, 5.165 D-76;
     5.163 P3200 with EI in column 0); and 5.166, whose EI is a row inside the
     body.

So this reader does not use word coordinates at all. PyMuPDF's `find_tables`
recovers the ruled grid, including merged cells (a merged cell is its text in
the first cell and '' after it), and every table here is fully ruled. What
the reader then has to decide is only which header string governs which
column, and that is done per column by forward-filling each header row to the
right -- a span covers the columns after it until the next label.

THE PHYSICAL CHECK. Within one series -- one film, developer, dilution,
vessel, format, EI and edition -- the time must not rise as the temperature
rises. A series that breaks it is HELD, not stored, and listed.

EI CONVENTION. The EI is stored as printed, except that a printed EI (or the
first figure of a printed range «100/200») equal to the stock's own box speed
is stored as 0 -- the convention of every normal-development point in the
database, so a box-speed row here is the same condition as a box-speed row
elsewhere. A row carrying an older generation's EDITION keeps its printed box
speed instead: a tagged point must say which speed its generation was rated. T-MAX P3200 (box 1000) has no such row: all its EIs are explicit.

EDITION. The book names two generations of several films -- «T-MAX 100
Professional» against «Professional T-MAX 100», «TRI-X Pan» against
«Professional TRI-X 400 / 400TX» -- and prints different times for them. The
current name carries edition ""; an older one carries its own label, as P98
already does for Tri-X Pan TX and Plus-X Pan.

REFUSED: 5.155 / 5.156 (machine processing, no temperature axis); 5.157,
5.160, 5.161 (compensation and capacity, not times); «High Speed Infrared /
HSI» rows (no profile).

Usage:  python sovremennye_tmax_push.py --root <project root> [--dump] [--literal]
        (as an audit: every point the literal holds is re-read from the page)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BOOK = "Современные фотоматериалы и их обработка.pdf"
ALT_PDF = Path("/mnt/user-data/uploads/PYTHON.TST/PDF/PROFILES/SOVIET") / BOOK

#: table -> (pages, developer, dilution, film, ei source, vessel, format)
#:   developer None = from column 0; film "hdr" = from the film header row,
#:   "col0" = from column 0, else a stock name; ei "hdr" / "col0" / "col1" /
#:   "rows" / "box"; vessel a vocabulary value or "hdr" (Малый / Большой бак)
TABLES = {
    "3.164": ((308,), None, None, "hdr", "hdr", "small tank", ""),
    "3.165": ((308,), None, None, "hdr", "hdr", "large tank", ""),
    "3.166": ((308,), None, None, "hdr", "hdr", "large tank", "sheet"),
    "3.167": ((309,), None, None, "hdr", "hdr", "drum", ""),
    "3.168": ((309,), None, None, "hdr", "hdr", "drum", "sheet"),
    "3.169": ((310,), None, None, "KODAK_TMAX_P3200|T-MAX P3200 Professional", "col1", "small tank", ""),
    "3.170": ((310,), None, None, "KODAK_TMAX_P3200|T-MAX P3200 Professional", "col1", "large tank", ""),
    "3.171": ((311,), None, None, "KODAK_TMAX_P3200|T-MAX P3200 Professional", "col1", "drum", ""),
    "3.180": ((326,), None, None, "hdr", "hdr", "small tank", ""),
    "3.181": ((327,), None, None, "hdr", "hdr", "large tank", ""),
    "3.182": ((327,), None, None, "hdr", "hdr", "large tank", "sheet"),
    "3.183": ((327,), None, None, "hdr", "box", "drum", ""),
    "3.184": ((328,), None, None, "hdr", "box", "drum", "sheet"),
    "3.185": ((328,), None, None, "hdr", "hdr", "drum", ""),
    "3.186": ((329,), None, None, "hdr", "hdr", "drum", "sheet"),
    "3.187": ((329, 330), None, None, "KODAK_TMAX_P3200|", "col1", "small tank", ""),
    "3.188": ((330,), None, None, "KODAK_TMAX_P3200|", "col1", "large tank", ""),
    "3.189": ((331,), None, None, "KODAK_TMAX_P3200|", "col1", "drum", ""),
    "5.149": ((679,), "T-MAX RS", "", "col0", "col1", "small tank", ""),
    "5.150": ((680,), "T-MAX RS", "", "col0", "col1", "large tank", ""),
    "5.151": ((680,), "T-MAX RS", "", "col0", "col1", "large tank", "sheet"),
    "5.152": ((681,), "T-MAX RS", "", "col0", "col1", "tray", "sheet"),
    "5.153": ((681,), "T-MAX RS", "", "col0", "col1", "drum", ""),
    "5.154": ((682,), "T-MAX RS", "", "col0", "col1", "drum", "sheet"),
    "5.158": ((683, 684), "T-MAX", "", "col0", "col1", "", ""),
    "5.159": ((684, 685), "T-MAX", "", "col0", "col1", "drum", ""),
    "5.162": ((686, 687), "D-76", "", "col0", "box", "hdr", ""),
    "5.163": ((687,), "D-76", "", "KODAK_TMAX_P3200|T-MAX P3200 Professional", "col0", "hdr", ""),
    "5.164": ((687,), "D-76", "1:1", "col0", "box", "hdr", ""),
    "5.165": ((688,), "D-76", "", "col0", "box", "hdr", "sheet"),
    "5.166": ((688,), "D-76", "", "col0", "rows", "small tank", ""),
    "5.167": ((688,), "D-76", "", "col0", "col1", "drum", ""),
}

#: QUEUE P54c (2026-10-01): the NORMAL-development tables whose cells the P54
#: word-coordinate reader HELD (705 values: two films or two vessels in one
#: table split between them wrongly). Same grid reader, same physical check.
#: Pages 308-309, 326-329, 351, 381-382 and 395 were already re-read above
#: (P98 / P98b); these are the rest.
HELD_TABLES = {
    "3.159": ((306,), None, None, "hdr", "box", "small tank", ""),
    "3.160": ((306,), None, None, "hdr", "box", "large tank", ""),
    "3.161": ((307,), None, None, "hdr", "box", "tray", "sheet"),
    "3.162": ((307,), None, None, "hdr", "box", "drum", ""),
    "3.163": ((307,), None, None, "hdr", "box", "drum", "sheet"),
    "3.176": ((325,), None, None, "hdr", "box", "small tank", ""),
    "3.177": ((325,), None, None, "hdr", "box", "large tank", ""),
    "3.194": ((350,), None, None, "KODAK_PLUS_X_125|Plus-X Pan PX/PXP/PXE/PXT", "box", "hdr", ""),
    "3.195": ((350,), None, None, "KODAK_PLUS_X_125|Plus-X Pan PX/PXP/PXE/PXT", "box", "drum", ""),
    "3.196": ((350,), None, None, "KODAK_PLUS_X_125|Plus-X Pan PX/PXP/PXE/PXT", "box", "drum", ""),
    "3.197": ((350,), None, None, "KODAK_PLUS_X_125|Plus-X Pan PX/PXP/PXE/PXT", "box", "drum", ""),
    "3.207": ((360,), None, None, "KODAK_TRI_X_320TXP|TRI-X Pan Professional TXP", "box", "hdr", ""),
    "3.208": ((360,), None, None, "KODAK_TRI_X_320TXP|TRI-X Pan Professional TXP", "box", "hdr", "sheet"),
    "3.209": ((360,), None, None, "KODAK_TRI_X_400TX|Tri-X Pan TX", "box", "drum", ""),
    "3.210": ((360,), None, None, "KODAK_TRI_X_320TXP|TRI-X Pan Professional TXP", "box", "drum", ""),
    "3.225": ((380,), None, None, "KODAK_PLUS_X_125|", "box", "hdr", ""),
    "3.226": ((380,), None, None, "KODAK_PLUS_X_125|", "box", "drum", ""),
    "3.241": ((392,), None, None, "KODAK_TRI_X_400TX|", "box", "hdr", ""),
    "3.242": ((392,), None, None, "KODAK_TRI_X_320TXP|", "box", "hdr", ""),
    "3.263": ((411, 412), None, None, "FUJI_NEOPAN_ACROS_100|", "col1", "small tank", ""),
    "3.264": ((412,), None, None, "FUJI_NEOPAN_ACROS_100|", "col1", "large tank", ""),
}
SETS = {"tmax": TABLES, "held": HELD_TABLES}

TEMP_TOK = re.compile(r"(\d\d)\s*°")
FRAC = re.compile(r"^(\d+?)\s*([13])\s*/\s*([24])\b")
INT = re.compile(r"^(\d{1,2})\b")


def value(cell: str):
    """A printed time in minutes, or None for «—», «Не рекомендуется», «NR»,
    «См. табл.» and blanks. Footnote stars are dropped; «61/ 2» is 6 1/2."""
    t = (cell or "").replace("\n", " ").replace("*", " ").strip()
    if not t or t[0] not in "0123456789":
        return None
    m = FRAC.match(t)
    if m:
        whole = int(m.group(1)) if m.group(1) else 0
        num, den = int(m.group(2)), int(m.group(3))
        if num >= den:
            return None
        return whole + num / den
    m = INT.match(t)
    return float(m.group(1)) if m else None


def norm(s: str) -> str:
    s = (s or "").replace("\n", " ")
    s = re.sub(r"([A-Z])-\s+([A-Z])", r"\1-\2", s)  # «PLUS- X» -> «PLUS-X»
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)          # «Profes- sional»
    s = s.replace("Р3200", "P3200")                   # Cyrillic Р
    return re.sub(r"\s+", " ", s).strip()


def film_of(label: str):
    """(stock, edition) for a film label as printed, or None (no profile)."""
    l = norm(label).lower()
    if "high speed" in l:
        return None
    if "p3200" in l or "3200 professional" in l or "t-max 3200" in l:
        return ("KODAK_TMAX_P3200", "" if l.startswith("professional")
                or "kodak professional" in l else "T-MAX P3200 Professional")
    if "t-max 100" in l:
        return ("KODAK_TMAX_100", "" if "professional t-max 100" in l
                else "T-MAX 100 Professional")
    if "t-max 400" in l:
        return ("KODAK_TMAX_400", "" if "professional t-max 400" in l
                else "T-MAX 400 Professional")
    if "tri-x 400" in l or "400tx" in l:
        return ("KODAK_TRI_X_400TX", "")
    if "tri-x 320" in l or "320txp" in l:
        return ("KODAK_TRI_X_320TXP", "")
    if "tri-x pan professional" in l:
        return ("KODAK_TRI_X_320TXP", "TRI-X Pan Professional TXP")
    if "tri-x pan" in l:
        return ("KODAK_TRI_X_400TX", "Tri-X Pan TX")
    if "plus-x 125" in l:
        return ("KODAK_PLUS_X_125", "")
    if "plus-x pan" in l:
        return ("KODAK_PLUS_X_125", "Plus-X Pan PX/PXP/PXE/PXT")
    if "verichrome" in l:
        return ("KODAK_VERICHROME_PAN", "")
    if "ektapan" in l:
        return ("KODAK_EKTAPAN_100", "")
    raise ValueError("unknown film label %r" % label)


def dev_of(label: str):
    """(developer, dilution, format) from a row label as printed."""
    t = norm(label)
    t = re.sub(r"\*\s*\d(\s*,\s*\*\s*\d)*", "", t).strip()
    fmt = ""
    m = re.search(r"(?:Для\s+)?(135|120)\s+фор\s*мат\w*", t)
    if m:
        fmt = m.group(1)
        t = (t[:m.start()] + t[m.end():]).strip()
    m = re.match(r"^(.*?)\s*\(?\s*(1:\d+|Dil [A-H])\s*\)?$", t)
    dev, dil = (m.group(1).strip(), m.group(2)) if m else (t, "")
    dev = re.sub(r"\s*\(только для PXT\)", "", dev)   # 3.197: the row is PXT's
    dev = {"X tol": "XTOL", "Super Prodol (SPD)": "Super Prodol"}.get(dev, dev)
    if dev not in ("T-MAX", "T-MAX RS", "XTOL", "D-76", "HC-110", "MICRODOL-X",
                   "Microdol-X", "DK-50", "Super Prodol", "ID-11", "Perceptol",
                   "Minidol", "Finedol", "Super Finedol", "Microfine",
                   "Fujidol E", "Super Fujidol-L", "Neoprodol"):
        raise ValueError("unknown developer label %r" % label)
    if dev == "T-MAX" and dil == "1:4":
        dil = ""                     # the printed standard dilution, footnote *1
    return dev, dil, fmt


def ei_of(text: str):
    m = re.search(r"(\d[\d,]*)", (text or "").replace(" ", ""))
    return int(m.group(1).replace(",", "")) if m else None


def _tables(doc, tid):
    """The find_tables grids that make up one table, in page order."""
    pages = ALL[tid][0]
    out = []
    for k, pno in enumerate(pages):
        pg = doc[pno - 1]
        words = pg.get_text("words")
        tabs = pg.find_tables().tables
        for t in tabs:
            y0 = t.bbox[1]
            caps = [w for w in words if w[4] == "Табл." and w[3] <= y0 + 2
                    and y0 - w[1] < 70]
            cap = None
            if caps:
                c = max(caps, key=lambda w: w[1])
                nxt = sorted((w for w in words if abs(w[1] - c[1]) < 3
                              and w[0] > c[0]), key=lambda w: w[0])
                cap = nxt[0][4].rstrip(".") if nxt else None
            if (k == 0 and cap == tid) or (k > 0 and (cap is None or cap == tid)
                                           and t is tabs[0]):
                out.append(t.extract())
                break
    if len(out) != len(pages):
        raise ValueError("%s: found %d of %d grid parts" % (tid, len(out), len(pages)))
    return out


def read_table(doc, tid, box):
    _pg, fdev, fdil, film_src, ei_src, vessel_src, fmt0 = ALL[tid]
    pts = []
    for grid in _tables(doc, tid):
        ncol = len(grid[0])
        temps = {}                # col -> celsius
        hdr = {"film": {}, "ei": {}, "vessel": {}}
        label0 = ""               # forward-filled column 0
        seg_film = None           # a full-width film row (layout B)
        rows_ei = {}              # 5.166: EI per column from a body row
        for row in grid:
            cells = [norm(c) for c in row]
            # ---- a temperature row: rebuild the temperatures from the joined text,
            # because a few header cells split «24°C 27°C» as «24°C 2» / «7°C».
            if sum(1 for c in cells if "°" in c) >= 2 and not any(
                    value(c) is not None and "°" not in c for c in cells[2:]):
                joined = "".join(c.replace(" ", "") for c in cells)
                ts = [int(x) for x in TEMP_TOK.findall(joined)]
                first = min(j for j, c in enumerate(cells) if "°" in c)
                cols = [j for j, c in enumerate(cells) if c and j >= first]
                if len(cols) != len(ts):
                    raise ValueError("%s: %d temperature cells, %d temperatures %s"
                                     % (tid, len(cols), len(ts), cells))
                temps = dict(zip(cols, ts))
                continue
            # ---- header rows: forward-fill right
            joined = " ".join(cells)
            has_val = any(value(c) is not None for j, c in enumerate(cells)
                          if j in temps)
            if not has_val:
                f = [c for c in cells if c and re.search(r"T-MAX|TRI-X|Plus-X|PLUS-X", c)]
                if f and film_src == "hdr":
                    cur = None
                    for j in range(1, ncol):
                        if cells[j] and re.search(r"T-MAX|TRI-X", cells[j]):
                            cur = cells[j]
                        hdr["film"][j] = cur
                    if sum(1 for c in cells if c) == 1:
                        seg_film = f[0]
                        temps, hdr["ei"] = {}, {}
                if "EI" in joined and ei_src in ("hdr", "rows"):
                    cur = None
                    tgt = rows_ei if ei_src == "rows" else hdr["ei"]
                    for j in range(1, ncol):
                        if "EI" in cells[j]:
                            cur = ei_of(cells[j].split("EI", 1)[1])
                        tgt[j] = cur
                if ("бак" in joined or "Поддон" in joined) and vessel_src == "hdr":
                    cur = None
                    for j in range(1, ncol):
                        if "Малый" in cells[j]:
                            cur = "small tank"
                        elif "Большой" in cells[j]:
                            cur = "large tank"
                        elif "Поддон" in cells[j]:
                            cur = "tray"
                        hdr["vessel"][j] = cur
                continue
            # ---- a body row
            if cells[0] and cells[0] not in ("—",):
                label0 = cells[0]
            if not temps:
                raise ValueError("%s: a body row before any temperature row" % tid)
            if fdev is None:
                dev, dil, fmt = dev_of(label0)
            else:
                dev, dil, fmt = fdev, fdil, ""
            fmt = fmt or fmt0
            row_ei = None
            if ei_src == "col1":
                row_ei = ei_of(cells[1])
            elif ei_src == "col0":
                row_ei = ei_of(cells[0])
            for j, c in temps.items():
                v = value(cells[j])
                if v is None:
                    continue
                if film_src == "hdr":
                    fl = seg_film or hdr["film"].get(j)
                    who = film_of("Kodak " + fl if fl and not fl.startswith("Kodak") else fl)
                elif film_src == "col0":
                    who = film_of(label0)
                else:
                    st, ed = film_src.split("|")
                    who = (st, ed)
                if who is None:
                    continue
                stock, edition = who
                if ei_src == "hdr":
                    ei = hdr["ei"].get(j)
                elif ei_src == "rows":
                    ei = rows_ei.get(j)
                elif ei_src == "box":
                    ei = box[stock]
                else:
                    ei = row_ei
                if ei is None:
                    raise ValueError("%s: no EI for column %d row %r" % (tid, j, cells))
                # box speed is EI 0 -- except on a generation-tagged row, which
                # must state its own speed (verify: a tagged point states its EI)
                ei = 0 if (ei == box[stock] and not edition) else ei
                vessel = hdr["vessel"].get(j) if vessel_src == "hdr" else vessel_src
                pts.append((stock, dev, dil, v, c, vessel, fmt, ei, edition, tid))
    return pts


def physical(pts):
    """(kept, held): series whose time rises with temperature are held."""
    ser = {}
    for p in pts:
        ser.setdefault(p[:3] + p[5:10], []).append((p[4], p[3]))
    bad = set()
    for k, s in ser.items():
        s.sort()
        if any(m1 > m0 + 1e-9 for (_, m0), (_, m1) in zip(s, s[1:])):
            bad.add(k)
    kept = [p for p in pts if (p[:3] + p[5:10]) not in bad]
    held = [p for p in pts if (p[:3] + p[5:10]) in bad]
    return kept, held


ALL = {**TABLES, **HELD_TABLES}


def read_all(pdf, box, which="tmax"):
    import pymupdf
    doc = pymupdf.open(pdf)
    out = []
    for tid in SETS[which]:
        out += read_table(doc, tid, box)
    return physical(out)


def _pdf(root):
    p = Path(root) / "PDF" / "PROFILES" / "SOVIET" / BOOK
    return p if p.is_file() else ALT_PDF


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--literal", action="store_true")
    ap.add_argument("--set", choices=sorted(SETS), default=None,
                    help="with --dump / --literal: which table set")
    a = ap.parse_args(argv)
    pdf = _pdf(a.root)
    if not pdf.is_file():
        print("[SKIP] source not present: %s" % BOOK)
        return 0
    import film_profiles as fp
    stocks = {"KODAK_TMAX_100", "KODAK_TMAX_400", "KODAK_TMAX_P3200",
              "KODAK_TRI_X_400TX", "KODAK_TRI_X_320TXP", "KODAK_PLUS_X_125",
              "KODAK_VERICHROME_PAN", "KODAK_EKTAPAN_100",
              "FUJI_NEOPAN_ACROS_100"}
    box = {s: fp.get_profile(s).exposure_index for s in stocks}
    if a.dump or a.literal:
        kept, held = read_all(pdf, box, a.set or "tmax")
        if a.dump:
            for p in kept:
                print(p)
            print("HELD:")
            for p in held:
                print(p)
        if a.literal:
            last = None
            for p in kept:
                if p[9] != last:
                    print("    # Табл. %s" % p[9])
                    last = p[9]
                print("    %r," % (p,))
        return 0
    literal = {"tmax": fp.SOVREMENNYE_2004_TMAX_POINTS,
               "held": fp.SOVREMENNYE_2004_GRID_POINTS}
    bad = 0
    for which in ("tmax", "held"):
        kept, held = read_all(pdf, box, which)
        have = set(literal[which])
        read = set(kept)
        miss, extra = sorted(read - have), sorted(have - read)
        on_prof = 0
        for st, dev, dil, m, c, vessel, fmt, ei, ed, tid in read:
            prof = fp.get_profile(st)
            on_prof += any(q.developer == dev and q.dilution == dil
                           and abs(q.minutes - m) < 1e-9 and q.celsius == c
                           and q.vessel == vessel and q.film_format == fmt
                           and (q.exposure_index == ei
                                # a disagreeing re-printing is kept under its
                                # table's edition, and a tagged point states
                                # its box speed instead of 0 (film_profiles)
                                or (ei == 0 and q.exposure_index == prof.exposure_index
                                    and ("Табл. " + tid) in q.edition))
                           for q in prof.processing_family.points)
        ok = not miss and not extra and on_prof == len(read) and not held
        bad += not ok
        print("%s  [%s] %d cells re-read from %d tables (%d held by the physical "
              "check); literal holds %d; %d missing, %d extra; %d of %d on their "
              "profiles at their EI"
              % ("PASS" if ok else "FAIL", which, len(read), len(SETS[which]),
                 len(held), len(have), len(miss), len(extra), on_prof, len(read)))
        for r in (miss + extra)[:8]:
            print("   ", r)
    print("[%s] sovremennye_tmax_push.py -- 2 checks, %d failed"
          % ("OK" if not bad else "FAIL", bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
