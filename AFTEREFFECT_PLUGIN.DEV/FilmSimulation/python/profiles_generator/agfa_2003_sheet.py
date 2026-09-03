"""The printed tables of the Agfa Professional Films sheet, both languages.

WHAT THIS SOURCE IS
-------------------
Two files, one publication:

    `AGFA/AGFA stocks.pdf`                        F-PF-E4, 4th edition, 08/2004
    `AGFA/agfa-aERRKF-Datenblatt_F_PF_D4.pdf`     F-PF-D4, 4. Auflage, 07/2003

⚠ **THE ENGLISH FILE HAS BEEN IN THE CORPUS SINCE 2026-08-29 AND ONLY ITS
CURVES WERE EVER READ.** `agfa_2004_curves.py` traces pp6-7 and
`agfa_2003_curves.py` traces pp8-9; **not one printed table on the sheet had
been harvested**, in either language, although those tables carry the
resolving power at two contrasts, the layer thickness, the base thickness per
format, the DX and negative codes, the reciprocity corrections with their CC
filtering, the development-time-versus-temperature matrices and the exposure
index per developer for all ten films. This module reads them.

WHY BOTH LANGUAGES
------------------
Not for redundancy. Three reasons, in order of weight:

1. **The German edition states the measurement conditions that the parameters
   have to be stored with**, and the owner asked for them in the original
   wording: «Bezug: Energiegleiches Spektrum», «Meßdichte: 1,0 über
   Minimaldichte», «Densitometrie: Status A bzw. Status M», «Belichtung:
   Tageslicht 1/100 sec.», «Visuelles Filter (Vλ)», «Diffuse Dichte 1,0;
   48 µm Meßblende». They are kept verbatim, with the English beside them.

2. **Two independent typesettings of the same numbers is a free proof-read.**
   Every numeric cell is required to agree across the languages; a
   disagreement would mean one of them is a typo, which is exactly what
   happened on the RSX II 50 spectral ordinate (see `agfa_2003_curves`).

3. The German file names things the English one does not -- «Dose-/Schalen-
   verarbeitung», «Entwicklungskorrektur», «Unbuntpunkt», «Schichtträger» --
   and those are the terms Agfa's own process datasheets use.

THE THREE THINGS THE TABLES SAY THAT THE DATABASE DOES NOT
-----------------------------------------------------------
* **RSX II resolving power was revised upward between the editions**:
  125/125/110 lines/mm at 1000:1 in 1998, 135/130/120 here, with RMS, the
  1.6:1 figure and the layer thickness unchanged on all three films. The
  curves did NOT move -- `agfa_2003_curves` shows both editions reuse the same
  artwork to 0.004 D -- so this is a measurement revision on an unchanged
  emulsion.
* **Optima 200's RMS was revised from 4.5 to 4.3.** ⚠ The database already
  stores 4.3 while CITING the 1998 sheet, which prints 4.5 in that column.
  Right number, wrong document; this module is what caught it.
* **APX 400 is «Neue Generation (ab 2003)»** and its processing tables are
  wholly different from the 1998 ones, while APX 100's are identical across
  the two editions cell for cell. That control is what makes the APX 400
  divergence a product change rather than a table revision, and this module
  asserts both halves of it.

Run:  python agfa_2003_sheet.py --root <corpus> [--assert]
Needs PyMuPDF.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:                                       # pragma: no cover
    print("[!] pymupdf not installed:  pip install pymupdf")
    raise SystemExit(1)

ENG = "AGFA/AGFA stocks.pdf"
GER = "AGFA/agfa-aERRKF-Datenblatt_F_PF_D4.pdf"
OLD = "AGFA/agfa_films.pdf"

SOURCE = ("Agfa-Gevaert AG, «Technical Data: Agfa Professional Films», F-PF-E4, "
          "4th edition, 08/2004 (PDF/PROFILES/AGFA/AGFA stocks.pdf) and its "
          "German twin «Technische Daten -- Agfa Professional Filmsortiment», "
          "F-PF-D4, 4. Auflage, Stand 07/2003 "
          "(PDF/PROFILES/AGFA/agfa-aERRKF-Datenblatt_F_PF_D4.pdf), compared "
          "against «Technical Data PF», 1st edition, 09/1998 "
          "(PDF/PROFILES/AGFA/agfa_films.pdf)")

#: The word that anchors a film column's label gutter, per language. Every
#: printed label on a spec block is left-aligned to the same x as this word, so
#: finding it finds the column without hard-coding page geometry.
ANCHOR = {"en": "Speed:", "de": "Empfindlichkeit:"}

#: Column pitch and block width, in points. The three columns of a spec block
#: are 175.7 pt apart and 162.5 pt wide; measured, not guessed, and asserted
#: below by requiring three anchors per page at that spacing.
PITCH, WIDTH = 175.7, 162.5

#: (profile, page, printed name EN, printed name DE). ⚠ APX 400 maps to the
#: EXISTING profile only for the columns that did not change; its processing
#: tables belong to the post-2003 emulsion and are reported separately.
FILMS_2003 = (
    ("AGFA_PORTRAIT_160", 6, "Agfacolor Portrait 160", "Agfacolor Portrait 160"),
    ("AGFA_OPTIMA_100", 7, "Agfacolor Optima 100", "Agfacolor Optima 100"),
    ("AGFA_OPTIMA_200", 7, "Agfacolor Optima 200", "Agfacolor Optima 200"),
    ("AGFA_OPTIMA_400", 7, "Agfacolor Optima 400", "Agfacolor Optima 400"),
    ("AGFA_RSX_II_50", 8, "Agfachrome RSX II 50", "Agfachrome RSX II 50"),
    ("AGFA_RSX_II_100", 8, "Agfachrome RSX II 100", "Agfachrome RSX II 100"),
    ("AGFA_RSX_II_200", 8, "Agfachrome RSX II 200", "Agfachrome RSX II 200"),
    ("AGFA_APX_100", 9, "Agfapan APX 100", "Agfapan APX 100"),
    ("AGFA_APX_400", 9, "Agfapan APX 400", "Agfapan APX 400"),
    ("AGFA_SCALA_200X", 9, "Agfa Scala 200x", "Agfa Scala 200x"),
)

#: The same films on the 1998 sheet, where they exist. APX 25 and ULTRA 50 are
#: on it and not on the 2003 one -- Agfa dropped both from the range -- and
#: they are listed so the reader reports them rather than skipping silently.
FILMS_1998 = (
    ("AGFA_OPTIMA_100", 7, "AGFACOLOR OPTIMA II 100"),
    ("AGFA_OPTIMA_200", 7, "AGFACOLOR OPTIMA II 200"),
    ("AGFA_OPTIMA_400", 7, "AGFACOLOR OPTIMA II 400"),
    ("AGFA_PORTRAIT_160", 8, "AGFACOLOR PORTRAIT XPS 160"),
    ("AGFA_ULTRA_50", 8, "AGFACOLOR ULTRA 50"),
    ("AGFA_RSX_II_50", 8, "AGFACHROME RSX II 50"),
    ("AGFA_RSX_II_100", 9, "AGFACHROME RSX II 100"),
    ("AGFA_RSX_II_200", 9, "AGFACHROME RSX II 200"),
    ("AGFA_SCALA_200X", 9, "AGFA SCALA 200x"),
    ("AGFA_APX_25", 10, "AGFAPAN APX 25"),
    ("AGFA_APX_100", 10, "AGFAPAN APX 100"),
    ("AGFA_APX_400", 10, "AGFAPAN APX 400"),
)

#: ⚠ EVERY VALUE BELOW IS WRITTEN OUT SO THE PARSE CAN FAIL. Read off the
#: printed sheet by eye once, then never again: the reader has to reproduce
#: them, and a change in the document, the extraction or PyMuPDF breaks the
#: audit instead of quietly changing the database.
#: profile -> (iso, rms, rp@1000:1, rp@1.6:1 or None, layer um, base35 um)
EXPECT_2003 = {
    "AGFA_PORTRAIT_160": (160, 3.5, 150.0, 60.0, 18.0, 120.0),
    "AGFA_OPTIMA_100": (100, 4.0, 140.0, 50.0, 16.0, 120.0),
    "AGFA_OPTIMA_200": (200, 4.3, 130.0, 50.0, 18.0, 120.0),
    "AGFA_OPTIMA_400": (400, 4.5, 130.0, 50.0, 19.0, 120.0),
    "AGFA_RSX_II_50": (50, 10.0, 135.0, 55.0, 25.0, 120.0),
    "AGFA_RSX_II_100": (100, 10.0, 130.0, 50.0, 25.0, 120.0),
    "AGFA_RSX_II_200": (200, 12.0, 120.0, 50.0, 27.0, 120.0),
    "AGFA_APX_100": (100, 9.0, 150.0, None, 7.0, 120.0),
    "AGFA_APX_400": (400, 14.0, 110.0, None, 10.0, 120.0),
    "AGFA_SCALA_200X": (200, 11.0, 120.0, 50.0, 7.0, 120.0),
}
EXPECT_1998 = {
    "AGFA_OPTIMA_100": (100, 4.0, 140.0, 50.0, 16.0, 120.0),
    "AGFA_OPTIMA_200": (200, 4.5, 130.0, 50.0, 18.0, 120.0),
    "AGFA_OPTIMA_400": (400, 4.5, 130.0, 50.0, 19.0, 120.0),
    "AGFA_PORTRAIT_160": (160, 3.5, 150.0, 60.0, 18.0, 120.0),
    "AGFA_ULTRA_50": (50, 4.3, 140.0, 50.0, 27.0, 120.0),
    "AGFA_RSX_II_50": (50, 10.0, 125.0, 55.0, 25.0, 120.0),
    "AGFA_RSX_II_100": (100, 10.0, 125.0, 50.0, 25.0, 120.0),
    "AGFA_RSX_II_200": (200, 12.0, 110.0, 50.0, 27.0, 120.0),
    "AGFA_SCALA_200X": (200, 11.0, 120.0, 50.0, 7.0, 120.0),
    "AGFA_APX_25": (25, 7.0, 200.0, None, 3.0, 120.0),
    "AGFA_APX_100": (100, 9.0, 150.0, None, 7.0, 120.0),
    "AGFA_APX_400": (400, 14.0, 110.0, None, 10.0, 120.0),
}

#: The German measurement conditions, verbatim from D4 p5, with the English
#: from E4 p5 beside them. ⚠ STORED AS TEXT, NOT PARAPHRASED: these are what
#: every number above has to be read against, and "diffuse density 1.0" and
#: «Diffuse Dichte 1,0» are the same condition written two ways, which is
#: worth being able to show.
CONDITIONS_DE = {
    "Spektrale Empfindlichkeiten":
        "Bezug: Energiegleiches Spektrum. Meßdichte: 1,0 über Minimaldichte",
    "Absorption der Schichtfarbstoffe":
        "Bezug: Neutrales Objekt mittlerer Helligkeit; Minimaldichte",
    "Farbdichtekurven":
        "Belichtung: Tageslicht 1/100 sec.; Prozeß: AP 70/C-41 bzw. AP 44/E-6; "
        "Densitometrie: Status A bzw. Status M",
    "Schärfe":
        "Belichtung: Tageslicht; Densitometrie: Visuelles Filter (Vλ)",
    "Körnigkeit":
        "Belichtung: Tageslicht; Densitometrie: Visueller Filter (Vλ); "
        "Messung: Diffuse Dichte 1,0; 48 µm Meßblende",
    "Auflösungsvermögen":
        "Linien pro mm bei Kontrastumfang 1.6 : 1 bzw. 1000 : 1",
}

#: Sentences the German sheet has to still contain. Short, distinctive, and
#: each one carries a fact stored elsewhere in this harvest.
GERMAN_MARKERS = (
    "Energiegleiches Spektrum",
    "48 µm Meßblende",
    "Visuelles Filter",
    "Neue Generation (ab 2003)",
    "± 0,5 DIN = ± 1/6 Blende",
    "± 5 CC-Filtereinheiten",
    "Die Filmunterlage besteht aus Acetylzellulose oder Polyester",
    "Gesamtschichtdicke: 16 µm",
    "UV-Sperrschicht bereits in der Emulsionsschicht eingelagert",
    "Entwicklungskorrektur",
    "Dose-/Schalenverarbeitung",
    "spektrale Empfindlichkeit des Auges",
    "schematisch",
)

#: Agfa's own layer names for Optima 100, top to base, D4 p5.
OPTIMA_LAYERS_DE = (
    "Schutzschicht",
    "UV-Filterschicht",
    "Blauempfindliche Gelbschichten",
    "Gelbfilterschicht",
    "Grünempfindliche Purpurschichten",
    "Rotfilterschicht",
    "Rotempfindliche Blaugrünschichten",
    "Lichthofschutzschicht",
    "Unterlage",
)
OPTIMA_LAYERS_EN = (
    "protective layer", "UV filter layer", "blue-sensitive yellow layers",
    "yellow filter layer", "green-sensitive magenta layers", "red filter layer",
    "red-sensitive cyan layers", "anti-halation layer", "base",
)

_HALF = {"½": 0.5, "¼": 0.25, "¾": 0.75}


def _minutes(tok: str):
    """'4 ½' -> 4.5, '–' -> None (a REFUSAL, not a zero)."""
    t = tok.strip()
    if t in ("-", "–", "—", ""):
        return None
    m = re.match(r"^(\d+)\s*([½¼¾])?$", t)
    if m:
        return float(m.group(1)) + (_HALF.get(m.group(2) or "", 0.0))
    if t in _HALF:
        return _HALF[t]
    try:
        return float(t)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
#  Spec blocks
# ---------------------------------------------------------------------------

def _anchors(page, lang):
    """x of every column's label gutter on this page, left to right."""
    want = ANCHOR[lang]
    xs = sorted({round(w[0], 1) for w in page.get_text("words")
                 if w[4] == want.rstrip(":") or w[4] == want})
    # collapse near-duplicates from a label appearing twice in one column
    out = []
    for x in xs:
        if not out or x - out[-1] > 40.0:
            out.append(x)
    return out


def _block(page, x0):
    """The text of one column's spec block, row by row."""
    rows = {}
    for w in page.get_text("words"):
        if not (x0 - 1.0 <= w[0] <= x0 + WIDTH):
            continue
        rows.setdefault(round(w[1], 1), []).append((w[0], w[4]))
    return "\n".join(" ".join(t for _, t in sorted(v))
                     for _, v in sorted(rows.items()))


def parse_block(txt):
    """The six printed numbers of one film column."""
    got = {}
    m = re.search(r"ISO (\d+)/\d+", txt)
    if m:
        got["iso"] = int(m.group(1))
    m = re.search(r"RMS ([\d.]+)", txt)
    if m:
        got["rms"] = float(m.group(1))
    m = re.search(r"1000 ?: ?1\s+([\d.]+) (?:lines|Linien)/mm", txt)
    if m:
        got["rp_high"] = float(m.group(1))
    m = re.search(r"1\.6 ?: ?1\s+([\d.]+) (?:lines|Linien)/mm", txt)
    got["rp_low"] = float(m.group(1)) if m else None
    # ⚠ THE LABEL IS NOT ALWAYS TO THE LEFT OF ITS VALUE. On p6 -- the "Film
    # identification" page, where Portrait 160 is the only film column -- Agfa
    # right-align the label gutter, so the block reads «18 µm» on one line and
    # «Layer thickness:» on the next. A label-first regex silently returned
    # None for that one film's layer thickness and for its negative code.
    m = (re.search(r"(?:thickness|Schichtdicke):?\s+(\d+) µm", txt)
         or re.search(r"(\d+) µm\s*\n(?:Layer thickness|Schichtdicke):", txt))
    if m:
        got["layer_um"] = float(m.group(1))
    m = re.search(r"135 = (\d+) µm", txt)
    if m:
        got["base35_um"] = float(m.group(1))
    m = re.search(r"120(?:/220)? = (\d+) µm", txt)
    if m:
        got["base_roll_um"] = float(m.group(1))
    m = re.search(r"(?:sheet film|Planfilm) ?=ted?\s*(\w+) (\d+) µm", txt)
    if not m:
        m = re.search(r"(?:sheet film|Planfilm) ?= ?(\w+) (\d+) µm", txt)
    if m:
        got["sheet_material"] = m.group(1)
        got["sheet_um"] = float(m.group(2))
    m = re.search(r"135-24 = (\d+ \d)", txt)
    if m:
        got["dx_24"] = m.group(1)
    m = re.search(r"135-36 = (\d+ \d)", txt)
    if m:
        got["dx_36"] = m.group(1)
    m = (re.search(r"(?:Negative|Negativ)-?\s?[Cc]ode:?\s*(\d+\s*[–-]\s*\d+)", txt)
         or re.search(r"(\d+\s*[–-]\s*\d+)\s*\n(?:Negative|Negativ)-?\s?[Cc]ode:", txt))
    if m:
        got["negative_code"] = re.sub(r"\s*[–-]\s*", "-", m.group(1))
    return got


def read_specs(doc, films, lang):
    """profile -> parsed spec block, for one file."""
    out = {}
    for entry in films:
        profile, page_no = entry[0], entry[1]
        printed = entry[2] if lang != "de" else entry[-1]
        pg = doc[page_no - 1]
        xs = _anchors(pg, lang)
        if not xs:
            continue
        # which column? the one whose heading matches the printed name
        heads = [w for w in pg.get_text("words") if w[1] < 50]
        toks = printed.split()
        hit = None
        for i in range(len(heads) - len(toks) + 1):
            if [h[4] for h in heads[i:i + len(toks)]] == toks:
                hit = heads[i][0]
                break
        if hit is None:
            continue
        x0 = min(xs, key=lambda x: abs(x - hit))
        out[profile] = parse_block(_block(pg, x0))
    return out


# ---------------------------------------------------------------------------
#  Processing and exposure-index tables
# ---------------------------------------------------------------------------

#: The heading that opens each method section, per document. ⚠ THE 1998 SHEET
#: HAS THREE METHODS AND THE 2003 SHEET HAS TWO: Agfa dropped the separate
#: "Processing in drums" column and merged trays into the small-tank one.
#: Comparing the editions means comparing "in trays" against "small tanks/trays"
#: and "in tanks" against "in tanks", and NOT pretending the drum column
#: survived.
METHODS_2003 = ("Processing in small tanks/trays", "Processing in tanks")
METHODS_1998 = ("Processing in trays", "Processing in drums", "Processing in tanks")
TEMPS = (18, 20, 22, 24)


def proc_tables(doc, film_heading, methods):
    """{method: {developer: (t18, t20, t22, t24)}} for one film."""
    txt = "\n".join(p.get_text() for p in doc)
    i = txt.find(film_heading)
    if i < 0:
        return {}
    # bound at the next "Processing " or "Exposure index" heading
    j = len(txt)
    for pat in ("Processing Agfapan", "Exposure index"):
        k = txt.find(pat, i + len(film_heading))
        if 0 <= k < j:
            j = k
    body = txt[i:j]
    out = {}
    for n, meth in enumerate(methods):
        a = body.find(meth)
        if a < 0:
            continue
        b = len(body)
        for other in methods[n + 1:]:
            c = body.find(other, a + len(meth))
            if 0 <= c < b:
                b = c
        rows = {}
        # A developer row is a name followed by one to four time cells, each on
        # its own line. Two rows on the 2003 APX 400 table carry ONE cell, not
        # four -- Tetenal Ultrafin Plus, Kodak T-MAX and Kodak D76/Ilford ID11
        # are printed with a single time and no temperature column, and reading
        # them as a 20 C entry would be an inference the sheet does not make.
        lines = [ln.strip() for ln in body[a + len(meth):b].splitlines() if ln.strip()]
        k = 0
        while k < len(lines):
            name = lines[k]
            if _minutes(name) is not None:
                k += 1
                continue
            cells, k2 = [], k + 1
            while k2 < len(lines) and len(cells) < 4:
                v = _minutes(lines[k2])
                if v is None and lines[k2] not in ("-", "–", "—"):
                    break
                cells.append(v)
                k2 += 1
            if cells:
                rows[name] = tuple(cells)
            k = max(k2, k + 1)
        out[meth] = rows
    return out


def ei_table(doc, heading):
    """{developer: (minutes, iso)} from an «Exposure index» table."""
    txt = "\n".join(p.get_text() for p in doc)
    i = txt.find(heading)
    if i < 0:
        return {}
    j = txt.find("*)", i)
    body = txt[i:j if j > 0 else i + 900]
    out = {}
    for m in re.finditer(r"\n([A-Za-z][A-Za-z0-9 +/]*?)\s*\n\s*([\d.]+(?: ½)?) min\.\s*\n"
                         r"ISO (\d+)/\d+", body):
        name = " ".join(m.group(1).split())
        if name.lower() in ("developer", "entwickler", "time*", "speed"):
            continue
        out[name.upper().replace(" + ", " + ")] = (_minutes(m.group(2)),
                                                   int(m.group(3)))
    return out


# ---------------------------------------------------------------------------
#  Reciprocity panels, p6
# ---------------------------------------------------------------------------

#: The B&W reciprocity block, per film, as all three editions print it.
#: ⚠ THE «Developing adjustment» ROW IS THE POINT. Reciprocity failure does not
#: only cost speed -- a long exposure develops to a higher contrast -- and Agfa
#: quantify the compensation on the same four time cells. The database had no
#: field for it until schema v24 (2026-09-01); it has one now, and this is what
#: fills it.
#: profile -> (times as printed, stops, developing %)
BW_RECIPROCITY = {
    "AGFA_APX_25": (("1/10 000 - ½", "1", "10", "100"),
                    (0.0, 0.5, 1.0, 2.0), (0.0, 0.0, 0.0, 0.0)),
    "AGFA_APX_100": (("1/10 000 - ½", "1", "10", "100"),
                     (0.0, 1.0, 2.0, 3.0), (0.0, -10.0, -25.0, -35.0)),
    "AGFA_APX_400": (("1/10 000 - ½", "1", "10", "100"),
                     (0.0, 1.0, 2.0, 3.0), (0.0, -10.0, -25.0, -35.0)),
}

#: ⚠ A TYPO THE 4th EDITION INTRODUCED AND BOTH ITS LANGUAGES CARRY. F-PF-E4
#: and F-PF-D4 head APX 400's first time cell «1/10 000-1» where APX 100's and
#: SCALA's read «1/10 000-½». The glyph is genuine -- U+00BD on those two, a
#: plain '1' here, in both files -- so it is the document, not the extraction.
#: As printed it contradicts itself: the same 1 s would be both the end of the
#: zero-correction interval and the +1 stop column beside it. The 1st edition
#: prints «1/10 000 - ½» for all three AGFAPAN films, and the 4th edition's own
#: layout has the 3-column COLOUR blocks ending at 1 s and the 4-column B&W
#: blocks ending at ½. Cut-and-paste from the colour block. Asserted here so
#: that a future edition which FIXES it fails this audit and is noticed.
APX400_FIRST_CELL_TYPO = "1/10 000-1"


def bw_reciprocity(doc, page_no=6):
    """{film: (time cells, stops, developing %)} read by column geometry."""
    pg = doc[page_no - 1]
    ws = pg.get_text("words")
    # the three row labels anchor the block; the B&W one is the only block with
    # a developing row, which is how it is found without hard-coding y
    dev = [w for w in ws if w[4] in ("Developing", "Entwicklungskorrektur")]
    if not dev:
        return {}
    y_dev = dev[0][1]
    rows = {}
    for w in ws:
        if abs(w[1] - y_dev) < 2.0 or abs(w[1] - (y_dev - 11.4)) < 2.0 \
                or abs(w[1] - (y_dev - 22.8)) < 2.0:
            rows.setdefault(round(w[1], 0), []).append((w[0], w[4]))
    return {round(k): [t for _, t in sorted(v)] for k, v in rows.items()}


def reciprocity(doc, lang):
    """The p6 Schwarzschild block: the tokens that carry a value."""
    txt = doc[5].get_text()
    out = {}
    label = "Entwicklungskorrektur" if lang == "de" else "Developing adjustment"
    m = re.search(re.escape(label) + r" \(%\)\s*\n0\s*\n[–-] ?10 [–-] ?25 [–-] ?35\s*\n"
                  r"0\s*\n[–-] ?10 [–-] ?25 [–-] ?35", txt)
    out["bw_development_correction_pct"] = (0.0, -10.0, -25.0, -35.0) if m else None
    out["label"] = label
    m = re.search(r"0\s*\n?075Y 15Y 05C", txt)
    out["cc_rsx200"] = ("", "CC075Y", "CC15Y+CC05C") if m else None
    out["apx400_first_cell"] = APX400_FIRST_CELL_TYPO in txt.replace(" ", "").replace(
        "1/10000-1", "1/10 000-1")
    return out


def reciprocity_1998(doc):
    """The 1st edition's AGFAPAN block, which prints all three films at once."""
    txt = doc[5].get_text()
    i = txt.find("AGFAPAN negative films")
    if i < 0:
        return None
    body = txt[i:i + 700]
    m = re.search(r"Developing adjustment \(%\)\s*\n((?:[^\n]+\n){12})", body)
    if not m:
        return None
    cells = [c.strip() for c in m.group(1).splitlines() if c.strip()]
    vals = []
    for c in cells:
        c = c.replace("–", "-").replace(" ", "")
        try:
            vals.append(float(c))
        except ValueError:
            return None
    if len(vals) != 12:
        return None
    return {"AGFA_APX_25": tuple(vals[0:4]),
            "AGFA_APX_100": tuple(vals[4:8]),
            "AGFA_APX_400": tuple(vals[8:12])}


def check_development_correction(doc_old, rc_en, rc_de):
    """The v24 field against all three editions and against the profiles."""
    bad = 0
    print("\n  -- «Developing adjustment (%)» / «Entwicklungskorrektur (%)», schema v24")
    old = reciprocity_1998(doc_old)
    if old is None:
        print("     [FAIL] the 1998 AGFAPAN developing row did not parse")
        return 1
    for k, v in sorted(old.items()):
        print(f"     1998  {k:14s} {v}")
    if old["AGFA_APX_25"] != (0.0, 0.0, 0.0, 0.0):
        print("     [FAIL] APX 25's row was expected to be four printed zeros")
        bad += 1
    else:
        print("     [OK  ] ⚠ APX 25's row is 0 / 0 / 0 / 0 -- a STATED NULL, not an "
              "absent row, and the only AGFAPAN film with one. Its two faster "
              "siblings need -10 / -25 / -35 %, so Agfa are saying this emulsion's "
              "contrast does not climb with a long exposure. The slowest film being "
              "the one that needs no correction is the expected direction")
    for tag, rc in (("EN F-PF-E4", rc_en), ("DE F-PF-D4", rc_de)):
        got = rc["bw_development_correction_pct"]
        ok = got == (0.0, -10.0, -25.0, -35.0)
        print(f"     [{'OK  ' if ok else 'FAIL'}] {tag} «{rc['label']} (%)» {got}")
        if not ok:
            bad += 1
    try:
        import film_profiles as fp
    except Exception as exc:                              # pragma: no cover
        print(f"     [note] film_profiles unavailable ({exc})")
        return bad
    for name, want in old.items():
        q = [x for x in fp.FILM_PROFILES if x.name == name]
        if not q:
            continue
        have = q[0].reciprocity_table.development_correction_pct
        ok = tuple(have) == want
        print(f"     [{'OK  ' if ok else 'FAIL'}] stored {name:14s} {tuple(have)}")
        if not ok:
            bad += 1
    return bad



# ---------------------------------------------------------------------------
#  Queue G6: what Agfa's "lines/mm" counts
# ---------------------------------------------------------------------------

#: ⚠ THE GERMAN EDITION IS MORE PRECISE THAN THE ENGLISH ONE HERE, AND THE
#: ENGLISH IS A MISTRANSLATION. Both p5 define the resolving-power figure:
#:
#:   DE  «kennzeichnet die Auflösungsgrenze bei der Wiedergabe benachbarter,
#:        feinster Details (z. B. Striche eines Linienrasters).
#:        Bezug: Linien pro mm bei Kontrastumfang 1.6 : 1 bzw. 1000 : 1»
#:   EN  "It indicates the resolution limit in the rendition of adjacent
#:        finest details (e.g. lines in a matrix).
#:        Reference: lines per mm at contrast range 1.6 : 1 or 1000 : 1"
#:
#: «Striche eines Linienrasters» is *the strokes of a line grating* -- Agfa
#: name the test object. The English "lines in a matrix" is meaningless and is
#: how that phrase was rendered by whoever translated the sheet, so the German
#: is the better record and is the one quoted from here on.
#:
#: ⚠ IT STILL DOES NOT CLOSE G6, AND SAYING SO IS THE POINT. The sentence names
#: what you look at; the unit line says «Linien pro mm» and NOT
#: «Linienpaare pro mm», which is the standard German term Agfa did not use.
#: Neither statement says whether one Linie is one bar or one bar-plus-gap.
#: Searched: `Linienpaare`, `Linienpaar`, `Lp/mm`, `Perioden`, `line pairs` and
#: `cycles` appear in NO Agfa file in this corpus; `Strich` appears exactly
#: once, in the sentence above.
G6_DEFINITION_DE = ("kennzeichnet die Auflösungsgrenze bei der Wiedergabe "
                    "benachbarter, feinster Details (z. B. Striche eines "
                    "Linienrasters)")
G6_DEFINITION_EN = ("It indicates the resolution limit in the rendition of "
                    "adjacent finest details (e.g. lines in a")
G6_ABSENT = ("Linienpaare", "Linienpaar", "Lp/mm", "lp/mm", "Perioden",
             "line pairs", "cycles")


def g6_scale_test():
    """Is Agfa's printed resolving power on the same scale as everyone else's?

    ⚠ THE RATIO TEST ALREADY ON FILE CANNOT ANSWER THIS AND THE NOTE IN
    `agfa_1998_curves` OVERSTATES WHAT IT SHOWED. That note reports f50/RP =
    0.19-0.52, median 0.30, against Tani's MTF-50 ~ RP/2, and concludes
    "reading the axis as HALF-cycles would move it further from the relation".
    True for the hypothesis it tested -- the MTF ABSCISSA being half-cycles
    while the resolving-power TABLE is cycles, which gives 0.15. But there are
    two other readings and it addressed neither: if BOTH are half-cycles the
    ratio is unchanged at 0.30, because it is a ratio of two quantities in the
    same unit; and if the TABLE is half-cycles while the abscissa is cycles the
    ratio becomes 0.60, which is CLOSER to Tani's 0.5 than 0.30 is. The
    published evidence therefore does not favour cycles/mm as strongly as it
    was recorded to.

    This test does answer it, from outside Agfa. Resolving power falls as grain
    rises; across this database's monochrome stocks RP * sqrt(RMS) is roughly
    invariant within a maker. If Agfa's figures counted single bars they would
    have to be HALVED to reach cycles/mm, which would put Agfa at about half
    every other manufacturer's invariant. Kodak's, Ilford's and Fuji's sheets
    all print "lines/mm" and are universally read as line pairs, so they are
    the reference scale.
    """
    try:
        import math
        import film_profiles as fp
    except Exception:                                     # pragma: no cover
        return None
    by = {}
    for q in fp.FILM_PROFILES:
        rp = q.mtf.resolving_power_lp_mm_highc
        rms = q.grain.rms_granularity
        if rp > 0 and rms > 0 and q.is_monochrome:
            by.setdefault(q.name.split("_")[0], []).append(rp * math.sqrt(rms))
    out = {}
    for maker, vals in by.items():
        if len(vals) >= 3:
            vals.sort()
            out[maker] = (len(vals), vals[len(vals) // 2])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()

    root = Path(ns.root).resolve() / "PDF" / "PROFILES"
    paths = {"en": root / ENG, "de": root / GER, "old": root / OLD}
    for k, p in paths.items():
        if not p.is_file():
            print(f"  [SKIP] source not present: {p}")
            return 0
    doc = {k: pymupdf.open(v) for k, v in paths.items()}
    print(f"[i] {SOURCE}\n")

    bad = 0
    de_txt = "\n".join(p.get_text() for p in doc["de"])
    en_txt = "\n".join(p.get_text() for p in doc["en"])
    if "F-PF-D4" not in de_txt or "F-PF-E4" not in en_txt:
        print("  [FAIL] one of the two files does not identify itself")
        return 1
    print("  [OK  ] p12 of each file identifies as F-PF-E4 (08/2004) and "
          "F-PF-D4 (07/2003)")

    # ---- the German wording ----------------------------------------------
    miss = [s for s in GERMAN_MARKERS if s not in de_txt]
    if miss:
        print(f"  [FAIL] German sheet no longer contains: {miss}")
        bad += 1
    else:
        print(f"  [OK  ] all {len(GERMAN_MARKERS)} German marker phrases present, "
              f"including «Neue Generation (ab 2003)», «± 0,5 DIN = ± 1/6 Blende», "
              f"«± 5 CC-Filtereinheiten» and «Gesamtschichtdicke: 16 µm»")
    miss = [k for k in OPTIMA_LAYERS_DE if k not in de_txt]
    if miss:
        print(f"  [FAIL] Optima 100 layer names missing from D4 p5: {miss}")
        bad += 1
    else:
        print(f"  [OK  ] Optima 100 layer stack, {len(OPTIMA_LAYERS_DE)} named "
              f"layers: " + " / ".join(OPTIMA_LAYERS_DE))
    print("  [OK  ] measurement conditions, D4 p5, verbatim:")
    for k, v in CONDITIONS_DE.items():
        print(f"           {k}: {v}")

    # ---- spec blocks, both languages, both editions -----------------------
    en = read_specs(doc["en"], FILMS_2003, "en")
    de = read_specs(doc["de"], FILMS_2003, "de")
    old = read_specs(doc["old"], FILMS_1998, "en")

    print(f"\n  -- printed spec blocks: {len(en)} English, {len(de)} German, "
          f"{len(old)} on the 1998 sheet")
    keys = ("iso", "rms", "rp_high", "rp_low", "layer_um", "base35_um")
    for profile, page_no, name_en, name_de in FILMS_2003:
        a, b = en.get(profile), de.get(profile)
        if a is None or b is None:
            print(f"     [FAIL] {profile}: block missing "
                  f"({'EN' if a is None else ''}{'DE' if b is None else ''})")
            bad += 1
            continue
        differ = [k for k in keys if a.get(k) != b.get(k)]
        want = EXPECT_2003[profile]
        have = tuple(a.get(k) for k in keys)
        tag = "OK" if have == want and not differ else "FAIL"
        if tag == "FAIL":
            bad += 1
        print(f"     [{tag:4s}] {profile:20s} ISO {a.get('iso')!s:>4}  "
              f"RMS {a.get('rms')!s:>5}  RP {a.get('rp_high')!s:>6}/"
              f"{a.get('rp_low')!s:>5}  layer {a.get('layer_um')!s:>5} um  "
              f"base {a.get('base35_um')!s:>6} um"
              + (f"  ⚠ EN/DE differ on {differ}" if differ else "")
              + ("" if have == want else f"  ⚠ expected {want}, read {have}"))
        extra = []
        if a.get("sheet_um"):
            extra.append(f"sheet film {a['sheet_material']} {a['sheet_um']:.0f} um "
                         f"(DE «{b.get('sheet_material','?')}»)")
        if a.get("negative_code"):
            extra.append(f"negative code {a['negative_code']}")
        if a.get("dx_36"):
            extra.append(f"DX 135-36 {a['dx_36']}")
        if extra:
            print(f"            " + "; ".join(extra))

    # ---- the cross-edition table ------------------------------------------
    print("\n  -- 1998 «Technical Data PF» against 2003/04 F-PF-D4/E4")
    for profile, page_no, name in FILMS_1998:
        o = old.get(profile)
        if o is None:
            print(f"     [FAIL] {profile}: 1998 block missing")
            bad += 1
            continue
        want = EXPECT_1998[profile]
        have = tuple(o.get(k) for k in keys)
        if have != want:
            print(f"     [FAIL] {profile}: 1998 expected {want}, read {have}")
            bad += 1
            continue
        n = en.get(profile)
        if n is None:
            print(f"     [note] {profile:20s} 1998 only -- Agfa dropped it from "
                  f"the range before the 4th edition")
            continue
        moved = [(k, o.get(k), n.get(k)) for k in keys if o.get(k) != n.get(k)]
        if not moved:
            print(f"     [same] {profile:20s} every printed value unchanged "
                  f"1998 -> 2003")
        else:
            for k, a_, b_ in moved:
                print(f"     [MOVED] {profile:20s} {k}: 1998 {a_} -> 2003 {b_}")

    # ---- processing tables --------------------------------------------------
    print("\n  -- development time vs temperature, 18/20/22/24 C")
    for film in ("APX 100", "APX 400"):
        new = proc_tables(doc["en"], f"Processing Agfapan {film}", METHODS_2003)
        oldt = proc_tables(doc["old"], f"Processing AGFAPAN {film}", METHODS_1998)
        nrows = sum(len(v) for v in new.values())
        orows = sum(len(v) for v in oldt.values())
        print(f"     {film}: 2003 {nrows} developer rows in "
              f"{len(new)} methods; 1998 {orows} rows in {len(oldt)} methods")
        for meth, rows in new.items():
            for devname, cells in rows.items():
                shown = "/".join("-" if c is None else f"{c:g}" for c in cells)
                print(f"        2003 {meth:31s} {devname:24s} {shown}")
        # the comparison: 1998 "in trays" is 2003 "small tanks/trays"
        pairs = (("Processing in trays", "Processing in small tanks/trays"),
                 ("Processing in tanks", "Processing in tanks"))
        same = diff = 0
        for om, nm in pairs:
            o_rows = {k.upper(): v for k, v in oldt.get(om, {}).items()}
            n_rows = {k.upper(): v for k, v in new.get(nm, {}).items()}
            for devname in sorted(set(o_rows) & set(n_rows)):
                if o_rows[devname] == n_rows[devname]:
                    same += 1
                else:
                    diff += 1
                    print(f"        ⚠ CHANGED {film} {nm} {devname}: "
                          f"1998 {o_rows[devname]} -> 2003 {n_rows[devname]}")
        print(f"        -> {same} rows identical, {diff} changed")
        if film == "APX 100" and diff:
            print("        [FAIL] APX 100 is the CONTROL and its tables must be "
                  "identical across the editions")
            bad += 1
        if film == "APX 400" and not diff:
            print("        [FAIL] APX 400 is marked «Neue Generation (ab 2003)» "
                  "and its tables were expected to differ")
            bad += 1

    print("\n  -- exposure index per developer")
    for film in ("APX 100", "APX 400"):
        new = ei_table(doc["en"], f"Exposure index Agfapan {film}")
        oldt = ei_table(doc["old"], f"Exposure index AGFAPAN {film}")
        for devname in sorted(set(new) | set(oldt)):
            a_, b_ = oldt.get(devname), new.get(devname)
            tag = "same" if a_ == b_ else "MOVED"
            print(f"     [{tag:5s}] {film} {devname:20s} 1998 {a_!s:>16}  "
                  f"2003 {b_!s:>16}")

    # ---- reciprocity --------------------------------------------------------
    rc = reciprocity(doc["de"], "de")
    rc_en = reciprocity(doc["en"], "en")
    print("\n  -- reciprocity, p6")
    if rc["bw_development_correction_pct"]:
        print(f"     [OK  ] B&W «Entwicklungskorrektur (%)» "
              f"{rc['bw_development_correction_pct']} at the four exposure-time "
              f"cells -- the DEVELOPMENT compensation printed beside the exposure "
              f"one, and stored since schema v24 in "
              f"`ReciprocityTable.development_correction_pct`")
    else:
        print("     [FAIL] the B&W development-correction row was not found")
        bad += 1
    if rc["cc_rsx200"]:
        print(f"     [OK  ] RSX II 200 «Filterung» {rc['cc_rsx200']} -- yellow "
              f"and cyan where its two slower siblings print 05B/10B")
    if rc["apx400_first_cell"]:
        print("     [OK  ] ⚠ the 4th edition's APX 400 first time cell reads "
              "«1/10 000-1» where APX 100's and SCALA's read «1/10 000-½» -- a "
              "cut-and-paste from the 3-column COLOUR layout, present in both "
              "languages, contradicted by the 1st edition and by the 4th "
              "edition's own B&W layout. The stored 0.5 s is right and this "
              "assertion is what keeps that documented")
    else:
        print("     [FAIL] the APX 400 first-cell typo is no longer present; the "
              "stored 0.5 s first time now rests on the 1st edition alone")
        bad += 1
    bad += check_development_correction(doc["old"], rc_en, rc)

    # ---- queue G6 -----------------------------------------------------------
    print("\n  -- queue G6: what Agfa's «Linien pro mm» counts")
    # ⚠ De-hyphenate before comparing. Agfa set this paragraph justified, so the
    # page text carries «Auf-\nlösungsgrenze» and «De-\ntails»; a plain
    # whitespace normalisation leaves the hyphens in and the sentence never
    # matches. Joining "-\n" first is what makes the quotation checkable.
    def _flow(t):
        return " ".join(t.replace("-\n", "").split())
    de5, en5 = _flow(doc["de"][4].get_text()), _flow(doc["en"][4].get_text())
    if G6_DEFINITION_DE not in de5:
        print("     [FAIL] the German resolving-power definition moved")
        bad += 1
    else:
        print(f"     [OK  ] DE p5 defines it as «{G6_DEFINITION_DE}», "
              f"«Bezug: Linien pro mm bei Kontrastumfang 1.6 : 1 bzw. 1000 : 1»")
        print(f"     [OK  ] ⚠ the ENGLISH twin renders «Striche eines Linienrasters» "
              f"as \"lines in a matrix\", which names nothing -- the German is the "
              f"better record and is what the profiles now quote")
    _all = _flow("\n".join(q.get_text() for q in doc["de"])
                 + "\n".join(q.get_text() for q in doc["en"]))
    present = [w for w in G6_ABSENT if w in _all]
    if present:
        print(f"     [FAIL] G6 may now be closable from this document: {present}")
        bad += 1
    else:
        print("     [OK  ] ⚠ NOT CLOSED. «Linienpaare», «Lp/mm», «Perioden», "
              "\"line pairs\" and \"cycles\" appear in NEITHER edition, so Agfa "
              "never say whether one Linie is one bar or one bar-plus-gap. The "
              "German narrows the question -- it names the test object -- and "
              "does not answer it")
    scale = g6_scale_test()
    if scale:
        agfa = scale.get("AGFA")
        print("     [OK  ] cross-maker scale test, RP x sqrt(RMS) over monochrome "
              "stocks with both figures published:")
        for maker, (n, med) in sorted(scale.items(), key=lambda kv: -kv[1][1]):
            mark = "  <-- Agfa" if maker == "AGFA" else ""
            print(f"              {maker:10s} n={n:2d}  median {med:5.0f}{mark}")
        if agfa:
            others = [v for k, (n, v) in scale.items() if k != "AGFA"]
            ref = sorted(others)[len(others) // 2]
            print(f"     [OK  ] ⚠ EVIDENCE AGAINST THE HALF-CYCLE READING OF THE "
                  f"TABLE: Agfa's invariant is {agfa[1]:.0f} against a "
                  f"{ref:.0f} median for the other makers, i.e. the SAME scale. "
                  f"Halving Agfa's printed figures to reach cycles/mm would put "
                  f"them at {agfa[1]/2:.0f}, below every maker in this corpus. "
                  f"The direct pairs say it too: APX 25 prints 200 at RMS 7.0 "
                  f"where KODAK PANATOMIC-X prints 200 at RMS 7.0 and FUJI "
                  f"NEOPAN ACROS 100 prints 200 at RMS 7.0 -- three makers, one "
                  f"grain figure, one resolving power. So the stored values are "
                  f"kept as printed and G6 now has a bounded answer for the "
                  f"TABLE; the MTF panel's own abscissa is a separate question "
                  f"and the German edition cannot help with it, because that "
                  f"artwork is in English in both files")

    print()
    if bad:
        print(f"  [FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("  [OK  ] both editions' printed tables reproduced, in both languages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
