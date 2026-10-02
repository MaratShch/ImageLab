"""Status M RED aim densities from the KODAK still-film sheets (queue K2).

WHAT AN AIM DENSITY IS, AND WHY IT IS WORTH A SCHEMA FIELD
----------------------------------------------------------
Every one of the thirteen KODAK still-film publications in the corpus prints,
under a heading like "JUDGING NEGATIVE EXPOSURES", a small table of the RED
Status M densities a CORRECTLY EXPOSED and correctly processed negative should
measure on four named subject areas -- a KODAK Gray Card, the lightest step of a
KODAK Paper Gray Scale, and a normally lit forehead of a light and of a dark
complexion. It is the manufacturer's own operational definition of "correct
exposure" for that emulsion, stated in the same units the rest of this database
speaks, and nothing else in the file says what a right answer looks like: the
characteristic curves say what density a given exposure produces, and these say
which of those densities the film was designed to land on.

⚠ THEY ARE RANGES, AND THEY ARE STORED AS RANGES. Every value is printed as
"0.79 to 0.89". Collapsing that to a midpoint would invent a precision the
manufacturer declined to claim, and the WIDTH is itself information -- the
professional PORTRA sheets quote +/-0.05 around their aim, the consumer GOLD and
ULTRA MAX sheets +/-0.10 to +/-0.15, and a pushed stock wider still. A field
holding one number could not record that difference.

⚠ AND THEY ARE PER EXPOSURE INDEX WHERE THE SHEET SAYS SO. PORTRA 800 publishes
three columns (EI 800 / 1600 / 3200) and ULTRA MAX 800 two (EI 800 / 1600),
which is why the carrier is a LIST of records each carrying its own EI rather
than four numbers on the profile. That shape is also what queue K3 needs.

HOW THE TABLE IS READ: GEOMETRY, NOT PAGE ORDER
-----------------------------------------------
⚠ PAGE ORDER IS NOT THE TABLE'S ORDER AND THE TWO LAYOUTS DISAGREE ABOUT IT.
On the PORTRA sheets the forehead cell holds TWO lines ("-light complexion" /
"-dark complexion") inside one row, so the emitted pairs run light/dark for
column 1, then light/dark for column 2. On the ULTRA MAX sheets the same two
readings are two SEPARATE rows, so the pairs run across all columns of the light
row first. A reader that took the pairs in page order would transpose one layout
or the other, and both mistakes produce four plausible densities.

So: every "<num> to <num>" triple in the table's y-band is collected with the
position of its own text, then clustered into COLUMNS by x-centre and into ROWS
by y. The rows are named by the registry, in the order the sheets print them,
and the count of rows and of columns actually found is asserted against it.

⚠ AND THE LABELS ARE CORROBORATION, NOT THE KEY -- the same relation the layer
captions have to the band test in `spectral_vector.py`, and for the same reason:
they are not reliable enough to be the key and are too good to waste. Two of the
sixteen tables put a caption on two lines ("...with light" / "complexion"), so a
line-wise search finds no forehead label at all on the ULTRA MAX 400 sheets; and
every sheet repeats "KODAK Gray Card" in a FOOTNOTE inside the same band, which
a nearest-label rule happily mistakes for a fifth row. Each label that IS found
is required to sit directly above the row the registry names, and a label with
no row under it -- the footnote -- is ignored.

THE THREE PORTRA 800 TABLES DO NOT AGREE, AND THAT IS A FINDING
----------------------------------------------------------------
Three documents publish PORTRA 800's pushed aim densities: E-190 (2003) p7,
E-190 (2006) p6 and E-4040 (2016) p3. The 2006 and 2016 tables are IDENTICAL to
the last digit. The 2003 table is not, and it is not a revision either -- it is
broken, on two independent counts:

  1. ⚠ ITS GRAY CARD AIM FALLS AS THE FILM IS PUSHED: 0.80-1.00 at EI 800,
     0.75-0.95 at 1600, 0.70-0.90 at 3200. Every other pushed table in the
     corpus RISES, including the 400UC table printed immediately above it on
     the same page (0.80-1.00 -> 1.00-1.20) and both later editions of this
     very film. Pushing means developing further, and further development
     raises density at a given exposure; a falling aim is the wrong sign.
  2. ⚠ ITS EI 800 FOREHEAD PAIR IS COPIED FROM ANOTHER TABLE. 1.08-1.18 and
     0.93-1.03 are, to the digit, the 160NC/400NC column of the table above it.
     A 400-speed professional film and an 800-speed one do not share a forehead
     aim, and the two later editions give 0.95-1.25 / 0.75-1.10 here.

This is the same class of defect as the E-2468 characteristic panel already
pinned in `kodak_still_curves.py`: a Kodak publication reusing a block it should
have replaced. ⚠ THE 2003 TABLE IS READ, PINNED AND NOT ADOPTED -- recorded
rather than averaged, and recorded rather than silently dropped, so that the
next reader meets the evidence instead of the conclusion.

Run:  python kodak_aim_density.py [--root .] [--assert] [--dump]
Needs PyMuPDF. --assert exits non-zero if any table stops reproducing.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

#: The four subject areas, in the order the sheets print them. The key is what
#: the field is called; the tuple is the words that identify the label LINE.
#: ⚠ Matched on a distinguishing fragment, not the whole caption, because the
#: wording is not stable across publications: "KODAK Gray Card (gray side)
#: receiving same illumination" (PORTRA) against "The KODAK Gray Card (gray
#: side) receiving the same illumination" (ULTRA MAX), and the forehead rows are
#: "-light complexion" on one family and "person with light complexion" on the
#: other.
AREAS = (
    ("gray_card", ("gray card",)),
    ("gray_scale", ("gray scale",)),
    ("forehead_light", ("light complexion",)),
    ("forehead_dark", ("dark complexion",)),
)

#: The anchors that open an aim-density table, lower-cased. Three spellings
#: across the corpus, and the double space in the E-7019/E-7023 one is real.
ANCHORS = ("area measured", "area  measured on the negative",
           "area on the negative:")

#: A label owns the row that begins no more than this far below it. Measured
#: across all sixteen tables, not tuned: a row sits 4.2 to 18.5 pt below the
#: label line that owns it, and the gap to the NEXT label is never under 27 pt.
LABEL_REACH_PT = 24.0

#: ...and a row may start fractionally ABOVE the top of its own label line: a
#: range is vertically centred in its cell while a label line is measured from
#: its top, and on E-7024 the forehead-light row sits 0.3 pt over its caption.
#: The tightest gap between two rows is 9.5 pt, so 4 pt of slack cannot reach
#: the row above.
LABEL_ABOVE_SLACK_PT = 4.0

#: Two ranges are on the same ROW when their text rows are within this. Rows are
#: 9.5 pt apart at the tightest -- the two lines of a merged forehead cell -- and
#: a row's own ranges share a y to under 0.1 pt.
ROW_TOL_PT = 4.0

#: Two pairs belong to the same column when their x-centres are within this.
#: Columns on these sheets sit 52 pt apart at the narrowest (E-4040), and a
#: column's own pairs vary by under 0.1 pt because the cells are right-aligned.
COLUMN_TOL_PT = 20.0

#: The row order every one of these tables prints. Named per sheet rather than
#: assumed, so a re-issue that drops or reorders a row fails loudly instead of
#: shifting three values up by one.
#: ⚠ AND ALL SIXTEEN PRINT ALL FOUR, WHICH IS NOT WHAT THE FIRST PASS BELIEVED.
#: E-2468's table looked two rows long because the text dump it was read from
#: was truncated at a fixed character count; the page draws the forehead pair
#: like every other sheet. The geometry found the two extra rows and refused to
#: match the registry, which is exactly what naming the rows is for.
ROWS4 = ("gray_card", "gray_scale", "forehead_light", "forehead_dark")

#: (pdf under PDF/PROFILES/KODAK, page, index of the table ON that page,
#:  one column label per printed column left to right, the row order).
#: ⚠ THE COLUMN LABELS ARE PART OF THE KEY, NOT DECORATION. They are what the
#: adoption maps onto profiles and exposure indexes, and asserting the COUNT of
#: columns found against the count named here is what would catch a re-issue
#: that adds or drops one.
SHEETS = {
    # E-190 (2003) p7 carries THREE tables: the NC/VC pair, 400UC at two EIs,
    # and 800 at three. See the header for why the third is not adopted.
    "e190_2003_ncvc": ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 7, 0,
                       ("160NC+400NC", "160VC+400VC"), ROWS4),
    "e190_2003_400uc": ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 7, 1,
                        ("EI 400", "EI 800"), ROWS4),
    "e190_2003_800": ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 7, 2,
                      ("EI 800", "EI 1600", "EI 3200"), ROWS4),
    "e190_2006_ncvc": ("e190-Portra-2006.pdf", 6, 0,
                       ("160NC+400NC", "160VC+400VC"), ROWS4),
    "e190_2006_800": ("e190-Portra-2006.pdf", 6, 1,
                      ("EI 800", "EI 1600", "EI 3200"), ROWS4),
    "e2468_100t": ("e2468-Portra_100T.pdf", 2, 0, ("EI 100",), ROWS4),
    "e4051_160": ("e4051_portra_160.pdf", 3, 0, ("EI 160",), ROWS4),
    "e4050_400_2016": ("e4050_portra_400.pdf", 3, 0, ("EI 400",), ROWS4),
    "e4050_400_2010": ("portra400-techpub-e4050.pdf", 3, 0, ("EI 400",), ROWS4),
    "e4040_800": ("e4040_portra_800.pdf", 3, 0,
                  ("EI 800", "EI 1600", "EI 3200"), ROWS4),
    "e7019_max400": ("E7019_en-Ultra_Max_400.pdf", 3, 0, ("EI 400",), ROWS4),
    "e7023_max400": ("E7023_max_400.pdf", 3, 0, ("EI 400",), ROWS4),
    "e7024_max800": ("E7024-Ultra_Max_800.pdf", 2, 0, ("EI 800", "EI 1600"), ROWS4),
    "e7022_gold_2022": ("E7022-1.pdf", 2, 0, ("EI 200",), ROWS4),
    "e7022_gold_2007": ("E7022-Gold_100_200.pdf", 2, 0,
                        ("GOLD 100", "GOLD 200"), ROWS4),
    "e29_100t": ("e29-Pro_100T_PRT.pdf", 2, 0, ("EI 100",), ROWS4),
}

#: Every table, read and pinned 2026-08-31. {tag: {area: ((lo, hi), ...)}},
#: one (lo, hi) per column in the order SHEETS names them.
EXPECTED = {
    "e190_2003_ncvc": {
        "gray_card": ((0.77, 0.87), (0.81, 0.93)),
        "gray_scale": ((1.13, 1.23), (1.22, 1.34)),
        "forehead_light": ((1.08, 1.18), (1.16, 1.28)),
        "forehead_dark": ((0.93, 1.03), (0.98, 1.10)),
    },
    "e190_2003_400uc": {
        "gray_card": ((0.80, 1.00), (1.00, 1.20)),
        "gray_scale": ((1.25, 1.45), (1.40, 1.60)),
        "forehead_light": ((1.00, 1.30), (1.20, 1.50)),
        "forehead_dark": ((0.80, 1.15), (0.95, 1.30)),
    },
    # ⚠ THE BROKEN ONE. Gray card FALLS with the push and the EI 800 forehead
    # pair is the NC column of the table above it. Pinned, not adopted.
    "e190_2003_800": {
        "gray_card": ((0.80, 1.00), (0.75, 0.95), (0.70, 0.90)),
        "gray_scale": ((1.15, 1.35), (1.15, 1.35), (1.15, 1.35)),
        "forehead_light": ((1.08, 1.18), (0.85, 1.20), (0.80, 1.15)),
        "forehead_dark": ((0.93, 1.03), (0.60, 0.95), (0.55, 0.90)),
    },
    "e190_2006_ncvc": {
        "gray_card": ((0.77, 0.87), (0.81, 0.93)),
        "gray_scale": ((1.13, 1.23), (1.22, 1.34)),
        "forehead_light": ((1.08, 1.18), (1.16, 1.28)),
        "forehead_dark": ((0.93, 1.03), (0.98, 1.10)),
    },
    "e190_2006_800": {
        "gray_card": ((0.75, 0.95), (0.85, 1.05), (0.95, 1.15)),
        "gray_scale": ((1.00, 1.20), (1.20, 1.40), (1.40, 1.60)),
        "forehead_light": ((0.95, 1.25), (1.10, 1.40), (1.25, 1.55)),
        "forehead_dark": ((0.75, 1.10), (0.90, 1.25), (1.00, 1.35)),
    },
    "e2468_100t": {
        "gray_card": ((0.74, 0.94),),
        "gray_scale": ((1.15, 1.35),),
        "forehead_light": ((1.04, 1.34),),
        "forehead_dark": ((0.82, 1.22),),
    },
    "e4051_160": {
        "gray_card": ((0.79, 0.89),),
        "gray_scale": ((1.15, 1.25),),
        "forehead_light": ((1.10, 1.20),),
        "forehead_dark": ((0.95, 1.05),),
    },
    "e4050_400_2016": {
        "gray_card": ((0.77, 0.87),),
        "gray_scale": ((1.13, 1.23),),
        "forehead_light": ((1.08, 1.18),),
        "forehead_dark": ((0.93, 1.03),),
    },
    "e4050_400_2010": {
        "gray_card": ((0.77, 0.87),),
        "gray_scale": ((1.13, 1.23),),
        "forehead_light": ((1.08, 1.18),),
        "forehead_dark": ((0.93, 1.03),),
    },
    "e4040_800": {
        "gray_card": ((0.75, 0.95), (0.85, 1.05), (0.95, 1.15)),
        "gray_scale": ((1.00, 1.20), (1.20, 1.40), (1.40, 1.60)),
        "forehead_light": ((0.95, 1.25), (1.10, 1.40), (1.25, 1.55)),
        "forehead_dark": ((0.75, 1.10), (0.90, 1.25), (1.00, 1.35)),
    },
    "e7019_max400": {
        "gray_card": ((0.80, 1.00),),
        "gray_scale": ((1.20, 1.40),),
        "forehead_light": ((1.10, 1.40),),
        "forehead_dark": ((0.85, 1.25),),
    },
    "e7023_max400": {
        "gray_card": ((0.80, 1.00),),
        "gray_scale": ((1.20, 1.40),),
        "forehead_light": ((1.10, 1.40),),
        "forehead_dark": ((0.85, 1.25),),
    },
    "e7024_max800": {
        "gray_card": ((0.75, 0.95), (0.85, 1.05)),
        "gray_scale": ((1.00, 1.20), (1.20, 1.40)),
        "forehead_light": ((0.95, 1.25), (1.10, 1.40)),
        "forehead_dark": ((0.75, 1.10), (0.90, 1.25)),
    },
    "e7022_gold_2022": {
        "gray_card": ((0.85, 1.05),),
        "gray_scale": ((1.25, 1.45),),
        "forehead_light": ((1.15, 1.45),),
        "forehead_dark": ((0.90, 1.30),),
    },
    "e7022_gold_2007": {
        "gray_card": ((0.90, 1.10), (0.85, 1.05)),
        "gray_scale": ((1.30, 1.50), (1.25, 1.45)),
        "forehead_light": ((1.20, 1.50), (1.15, 1.45)),
        "forehead_dark": ((0.95, 1.35), (0.90, 1.30)),
    },
    "e29_100t": {
        "gray_card": ((0.85, 1.05),),
        "gray_scale": ((1.20, 1.40),),
        "forehead_light": ((1.10, 1.40),),
        "forehead_dark": ((0.95, 1.30),),
    },
}

#: Which of the tables above the database actually adopts, and onto what.
#: {profile: (tag, column index, exposure index)}. ⚠ NOT EVERY TABLE IS HERE
#: AND THAT IS THE POINT: `e190_2003_800` is read and pinned and deliberately
#: absent (see the header), the 2010 and 2016 PORTRA 400 sheets are the same
#: numbers twice, and the 2007 GOLD sheet's GOLD 200 column is superseded by
#: the 2022 sheet -- which prints the identical values, so nothing turns on it.
ADOPTED = {
    "KODAK_PORTRA_160NC": (("e190_2006_ncvc", 0, 160),),
    "KODAK_PORTRA_400NC": (("e190_2006_ncvc", 0, 400),),
    "KODAK_PORTRA_160VC": (("e190_2006_ncvc", 1, 160),),
    "KODAK_PORTRA_400VC": (("e190_2006_ncvc", 1, 400),),
    "KODAK_ULTRA_COLOR_400UC": (("e190_2003_400uc", 0, 400),
                                ("e190_2003_400uc", 1, 800)),
    "KODAK_PORTRA_100T": (("e2468_100t", 0, 100),),
    "KODAK_PORTRA_160": (("e4051_160", 0, 160),),
    "KODAK_PORTRA_400": (("e4050_400_2016", 0, 400),),
    "KODAK_PORTRA_800": (("e4040_800", 0, 800),
                         ("e4040_800", 1, 1600),
                         ("e4040_800", 2, 3200)),
    "KODAK_ULTRAMAX_400": (("e7023_max400", 0, 400),),
    "KODAK_ULTRAMAX_800": (("e7024_max800", 0, 800),
                           ("e7024_max800", 1, 1600)),
    "KODAK_GOLD_100": (("e7022_gold_2007", 0, 100),),
    "KODAK_GOLD_200": (("e7022_gold_2022", 0, 200),),
}


def _table_bands(pg):
    """[(y_top, y_bottom), ...] one per aim-density table on the page."""
    tops = []
    for blk in pg.get_text("dict").get("blocks", []):
        for ln in blk.get("lines", []):
            txt = " ".join(sp.get("text", "") for sp in ln.get("spans", []))
            low = " ".join(txt.lower().split())
            for a in ANCHORS:
                if low.startswith(" ".join(a.split())):
                    tops.append((ln["bbox"][1], ln["bbox"][0]))
                    break
    # ⚠ SORT BY POSITION, NOT BY EMISSION. E-190 (2003) stacks three of these
    # in one column and the text layer does not promise to emit them downward.
    tops.sort()
    out = []
    for i, (y, _x) in enumerate(tops):
        nxt = tops[i + 1][0] if i + 1 < len(tops) else 1e9
        out.append((y, nxt))
    return out


def read_table(root: Path, tag: str):
    """{area: ((lo, hi) per column, ...)} for one printed table, or (None, err)."""
    import pymupdf
    fn, pgno, index, cols, rows = SHEETS[tag]
    pdf = root / "PDF" / "PROFILES" / "KODAK" / fn
    if not pdf.is_file():
        return None, f"source not present: {fn}"
    pg = pymupdf.open(pdf)[pgno - 1]
    bands = _table_bands(pg)
    if index >= len(bands):
        return None, (f"the page carries {len(bands)} aim-density tables, "
                      f"not the {index + 1} this registry expects")
    y0, y1 = bands[index]

    # the label lines inside the band, with their tops
    labels = []
    for blk in pg.get_text("dict").get("blocks", []):
        for ln in blk.get("lines", []):
            ly = ln["bbox"][1]
            if not (y0 <= ly < y1):
                continue
            low = " ".join(" ".join(sp.get("text", "")
                                    for sp in ln.get("spans", [])).lower().split())
            for key, words in AREAS:
                if any(w in low for w in words):
                    labels.append((ly, key))
                    break
    labels.sort()

    # every "<num> to <num>" inside the band, with the position of its own row
    words = pg.get_text("words")
    pairs = []
    for i, t in enumerate(words):
        if t[4] != "to" or i == 0 or i + 1 >= len(words):
            continue
        a, b = words[i - 1], words[i + 1]
        if not (re.fullmatch(r"\d\.\d\d", a[4]) and re.fullmatch(r"\d\.\d\d", b[4])):
            continue
        cy = (a[1] + a[3]) / 2
        if not (y0 <= cy < y1):
            continue
        pairs.append((float(a[4]), float(b[4]), (a[0] + b[2]) / 2, cy))
    if not pairs:
        return None, "no density ranges inside the table band"

    # columns by x-centre
    centres = []
    for _lo, _hi, cx, _cy in sorted(pairs, key=lambda p: p[2]):
        if not centres or cx - centres[-1] > COLUMN_TOL_PT:
            centres.append(cx)
    if len(centres) != len(cols):
        return None, (f"found {len(centres)} columns at x {['%.0f' % c for c in centres]}, "
                      f"but the registry names {len(cols)}: {list(cols)}")

    # rows by y, named by the registry in printed order
    row_y: list[float] = []
    for _lo, _hi, _cx, cy in sorted(pairs, key=lambda p: p[3]):
        if not row_y or cy - row_y[-1] > ROW_TOL_PT:
            row_y.append(cy)
    if len(row_y) != len(rows):
        return None, (f"found {len(row_y)} rows at y {['%.0f' % y for y in row_y]}, "
                      f"but the registry names {len(rows)}: {list(rows)}")

    # ⚠ THE LABELS ARE CHECKED HERE AND DO NOT DECIDE ANYTHING. A label owns the
    # first row starting under it within LABEL_REACH_PT; one with no row under
    # it is a FOOTNOTE ("For best results, use a KODAK Gray Card"), which every
    # sheet prints inside this band, and is ignored rather than made a fifth row.
    # ⚠ AND THE MATCH IS MONOTONE, WHICH IS NOT A DETAIL. Taking "the nearest
    # row" independently per label lets two labels claim one row: on E-29 the
    # merged forehead cell puts its two captions 9.5 pt apart and its two rows
    # 9.5 pt apart, offset by 5.8, so the DARK caption is 3.7 pt from the LIGHT
    # row and 5.8 from its own. Consuming rows top-down cannot make that
    # mistake, and it is also what makes the trailing footnote fall off the end.
    used = -1
    for ly, key in labels:
        under = [i for i in range(used + 1, len(row_y))
                 if -LABEL_ABOVE_SLACK_PT <= row_y[i] - ly <= LABEL_REACH_PT]
        if not under:
            continue
        used = under[0]
        if rows[used] != key:
            return None, (f"the label {key!r} at y {ly:.0f} sits above the row "
                          f"the registry calls {rows[used]!r}")

    out: dict[str, list] = {}
    for lo, hi, cx, cy in pairs:
        ri = min(range(len(row_y)), key=lambda i: abs(row_y[i] - cy))
        col = min(range(len(centres)), key=lambda i: abs(centres[i] - cx))
        row = out.setdefault(rows[ri], [None] * len(cols))
        if row[col] is not None:
            return None, (f"two ranges claim {rows[ri]} column {col}: "
                          f"{row[col]} and {(lo, hi)}")
        row[col] = (lo, hi)
    for key, row in out.items():
        if any(v is None for v in row):
            return None, f"{key} is missing a column: {row}"
    return {k: tuple(v) for k, v in out.items()}, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()
    bad = skipped = 0
    print(f"[i] corpus root {root}")
    for tag in SHEETS:
        got, err = read_table(root, tag)
        if got is None:
            if "not present" in (err or ""):
                print(f"  [SKIP] {tag}: {err}")
                skipped += 1
            else:
                print(f"  [FAIL] {tag}: {err}")
                bad += 1
            continue
        want = EXPECTED[tag]
        ok = got == want
        print(f"  [{'OK  ' if ok else 'FAIL'}] {tag:18s} "
              f"{len(got)} areas x {len(SHEETS[tag][3])} columns "
              f"{list(SHEETS[tag][3])}")
        if not ok:
            for k in sorted(set(got) | set(want)):
                if got.get(k) != want.get(k):
                    print(f"         {k}: read {got.get(k)}, expected {want.get(k)}")
            bad += 1
        if ns.dump:
            for k, v in got.items():
                print(f"            {k}: {v}")

    # ---- the cross-document checks, which are the point of reading all 16 ----
    # 1. The two PORTRA 400 sheets are six years apart and must still agree.
    a, b = EXPECTED["e4050_400_2010"], EXPECTED["e4050_400_2016"]
    ok = a == b
    print(f"  [{'OK  ' if ok else 'FAIL'}] PORTRA 400 reads the same in E-4050 "
          f"(2010) and E-4050 (2016)")
    bad += 0 if ok else 1
    # 2. So do the two ULTRA MAX 400 sheets, nine years apart.
    a, b = EXPECTED["e7019_max400"], EXPECTED["e7023_max400"]
    ok = a == b
    print(f"  [{'OK  ' if ok else 'FAIL'}] ULTRA MAX 400 reads the same in "
          f"E-7019 (2007) and E-7023 (2016)")
    bad += 0 if ok else 1
    # 3. And GOLD 200, fifteen years apart across two different publications
    #    with different layouts -- the 2007 sheet's SECOND column.
    ok = all(EXPECTED["e7022_gold_2007"][k][1] == EXPECTED["e7022_gold_2022"][k][0]
             for k in EXPECTED["e7022_gold_2022"])
    print(f"  [{'OK  ' if ok else 'FAIL'}] GOLD 200 reads the same in E-7022 "
          f"(2007, 2-column) and E-7022 (2022, 1-column)")
    bad += 0 if ok else 1
    # 4. PORTRA 800: 2006 and 2016 identical, 2003 NOT -- and broken in the two
    #    specific ways the header sets out. ⚠ This assertion is inverted on
    #    purpose: it fails if the 2003 table ever starts agreeing, because that
    #    would mean the reader changed rather than the document.
    same = EXPECTED["e190_2006_800"] == EXPECTED["e4040_800"]
    gc03 = EXPECTED["e190_2003_800"]["gray_card"]
    falls = gc03[0][0] > gc03[1][0] > gc03[2][0]
    copied = (EXPECTED["e190_2003_800"]["forehead_light"][0]
              == EXPECTED["e190_2003_ncvc"]["forehead_light"][0]
              and EXPECTED["e190_2003_800"]["forehead_dark"][0]
              == EXPECTED["e190_2003_ncvc"]["forehead_dark"][0])
    ok = same and falls and copied
    print(f"  [{'OK  ' if ok else 'FAIL'}] PORTRA 800: E-190 (2006) and E-4040 "
          f"(2016) identical; E-190 (2003) still shows BOTH defects "
          f"(gray card falls with the push: {falls}; EI 800 forehead copied "
          f"from the NC column: {copied})")
    bad += 0 if ok else 1
    # 5. A pushed aim must RISE, on every table that publishes more than one EI
    #    -- with the 2003 sheet excluded by name, because it is the counter-
    #    example and excluding it by rule would hide it.
    rising = []
    for tag, exp in EXPECTED.items():
        if tag == "e190_2003_800" or len(SHEETS[tag][3]) < 2:
            continue
        if not SHEETS[tag][3][0].startswith("EI"):
            continue          # a two-product table, not a two-EI one
        for area, row in exp.items():
            for i in range(len(row) - 1):
                if not (row[i + 1][0] > row[i][0] and row[i + 1][1] > row[i][1]):
                    rising.append(f"{tag}.{area} {row[i]} -> {row[i + 1]}")
    print(f"  [{'OK  ' if not rising else 'FAIL'}] every pushed aim density "
          f"rises with the push (2003's 800 table excluded by name)"
          + ("" if not rising else "  " + "; ".join(rising)))
    bad += 0 if not rising else 1

    print(f"\n[i] {len(SHEETS) + 5 - bad - skipped} reproduced, {bad} failed, "
          f"{skipped} skipped")
    if ns.do_assert and bad:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
