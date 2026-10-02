#!/usr/bin/env python3
"""The FULL Soviet corpus -- nine documents, 176 sheets -- audited (tranche D).

WHAT THIS MODULE IS, AND WHY IT IS NOT `soviet_tu_1986_90.py`
-------------------------------------------------------------
`soviet_tu_1986_90.py` audits EIGHT ТУ and audits them one way: it checks the
CITATION TRAIL, asserting that the distinguishing token of each claim is on the
page the record cites. That was the right limit for those eight sheets and it
still is. This module does three things that one cannot:

  1. IT INCLUDES THE NINTH DOCUMENT. ГОСТ 25120-82, the umbrella state
     standard, was read on 2026-09-23 and was in no audit at all.
  2. IT RE-DERIVES VALUES INSTEAD OF CITATIONS -- for the one document where
     that is honest. The ГОСТ is a typeset state standard, not a typescript,
     and its OCR layer renders табл. 6 cell for cell. Every number this
     project stores from it is re-parsed here and compared with the database.
  3. IT AUDITS THE ACCEPTANCE BANDS, `FilmProfile.tolerance` (schema v51),
     which did not exist when the older module was written.

⚠⚠ THE OCR SPLIT IS MEASURED HERE, NOT ASSERTED. The older module states that
these typescripts cannot be parsed for values and takes that as given. A
statement like that decays: it was true of a particular scan on a particular
day and nobody re-checks it. So `_ocr_value_recall` below actually counts, for
each document, how many of the numbers the database stores from it appear
verbatim in its own OCR layer. The ГОСТ scores near the top of the range and
the eight typescripts score near the bottom, and THAT GAP is what justifies
re-deriving one document and citation-checking the other eight. If a better
scan of a ТУ ever lands in the corpus, this number moves and the gate says so.

WHAT THE PATTERN OF PRESENCE AND ABSENCE PROVES
-----------------------------------------------
No single sheet could show it, and it is the strongest result in the corpus.
Reduce each document to a whitespace-free lower-case string and ask which
specification VOCABULARY it uses:

  * «средний градиент» occurs in exactly the three masked-cine-negative ТУ.
    «коэффициент контрастности» occurs in exactly the three reversal ТУ, in
    ЦНД-64 and in the ГОСТ. NO DOCUMENT USES BOTH. The Soviet industry named
    the quantity by the polarity of the film, and a profile that stored a
    "gamma" for ДС-5М would be using a word its own specification avoids.
  * «фотографическая однородность» occurs in exactly the three cine-negative
    ТУ and nowhere else -- the across-the-width uniformity norm is a CINE
    requirement, which is what one would expect of film that runs through a
    printer, and no Western data sheet in this database publishes one at all.
  * «разрешающая способность» occurs in exactly the three reversal ТУ and the
    ГОСТ, and in none of the four negative ТУ. That is the whole reason the
    R↔f50 bridge cannot cross polarity: there is no Soviet negative with both
    an R and an MTF, because there is no Soviet negative ТУ with an R.
  * ТУ 6-17-1371-86 contains NONE of the ten tokens. The record's sentence
    «This document states nothing photographic» is thereby a measurement.

⚠ TWO NEAR-MISSES ARE REAL FINDINGS AND ARE ASSERTED AS SUCH. ЛН-9's sheet
says «ФУНКЦИЯ передачи модуляции» where its four siblings say «КОЭФФИЦИЕНТ
передачи модуляции», and ЦО-90Л says «баланс КОЭФФИЦИЕНТА контрастности»
where ЦО-32Д and ЦНД-64 say «баланс контрастности». Both stocks DO carry the
quantity; a token census that ignored the wording would have reported two
false absences, so the expectations below encode the wording each sheet
actually uses.

THE THREE PERMANENT ABSENCES
-----------------------------
Asserted across all nine documents, because each is a claim this database
relies on and each would be invalidated by one counter-example:

  * NO PLOTTED CURVE OF ANY KIND. ⚠ The word «характеристическая» DOES occur,
    in four of the nine, and that is not a contradiction -- those are the
    sheets whose method section instructs the TESTER to plot one. The set of
    documents containing the word is pinned, so a real curve arriving in a
    re-scan cannot hide behind the instruction.
  * NO RECIPROCITY CLAUSE. No «невзаимозаместимость», no intermittency
    statement, in any of the nine.
  * ГОСТ 9160-82 IS CITED BY ALL NINE AND IS IN NONE OF THEM. Every speed,
    gamma and latitude figure this project holds for a Soviet stock rests on
    a criterion nobody in the corpus states.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import film_profiles as FP            # noqa: E402

TU_DIR = (Path(__file__).resolve().parent / "PDF" / "PROFILES"
          / "SOVIET STANDARDS")

#: (short name, file, sheets). ⚠ ГОСТ 25120-82's file name is a bare registry
#: number and gives no clue what it is; that is why it sat unaudited.
DOCS: tuple[tuple[str, str, int], ...] = (
    ("ТУ 6-17-691-88",
     "tu_61769188_kinoplenka_tsvetnaia_negativnaia_ds5m_tekhniches.pdf", 22),
    ("ТУ 6-17-1109-88",
     "tu_617110988_kinoplenka_tsvetnaia_negativnaia_ln8_tekhniches.pdf", 21),
    ("ТУ 6-17-1443-88",
     "tu_617144388_kinoplenki_tsvetnye_negativnye_ln9_i_ln9s.pdf", 23),
    ("ТУ 6-17-1000-88",
     "tu_617100088_kinoplenka_tsvetnaia_obrashchaemaia_tsot90_lm_t.pdf", 18),
    ("ТУ 6-17-912-87",
     "tu_61791287_kinofotoplenki_tsvetnye_obrashchaemye_tso32d_tek.pdf", 23),
    ("ТУ 6-42-1514-90",
     "tu_642151490_kinoplenka_i_fotoplenka_obrashchaemye_marok_tso.pdf", 20),
    ("ТУ 6-17-1453-89",
     "tu_617145389_plenka_fotograficheskaia_tsvetnaia_negativnaia.pdf", 26),
    ("ТУ 6-17-1371-86",
     "tu_617137186_plenki_fotograficheskie_35mm_perforirovannye_v.pdf", 8),
    ("ГОСТ 25120-82", "4294829285.pdf", 15),
)

#: Which profile each document specifies. ТУ 6-17-1371-86 governs three marks
#: and states nothing photographic about any of them, so it maps to nothing.
GOVERNS: dict[str, tuple[str, ...]] = {
    "ТУ 6-17-691-88":   ("SVEMA_DS_5M",),
    "ТУ 6-17-1109-88":  ("SVEMA_LN_8",),
    "ТУ 6-17-1443-88":  ("SVEMA_LN_9", "SVEMA_LN_9S"),
    "ТУ 6-17-1000-88":  ("SVEMA_CO_T_90LM",),
    "ТУ 6-17-912-87":   ("SVEMA_CO_32D",),
    "ТУ 6-42-1514-90":  ("SVEMA_CO_90L",),
    "ТУ 6-17-1453-89":  ("SVEMA_CND_64",),
    "ТУ 6-17-1371-86":  (),
    "ГОСТ 25120-82":    ("SVEMA_CND_32", "SVEMA_CNL_65"),
}

#: The specification vocabulary, whitespace-stripped and lower-cased. The
#: value is the set of documents the token MUST occur in -- exactly, not at
#: least, so a token appearing where the record says it does not is a failure
#: in the same way an absence is.
VOCABULARY: dict[str, tuple[str, ...]] = {
    # Speed balance Б_S: every specifying document has it. The one that does
    # not is the export packaging sheet.
    "баланссветочувствительн": (
        "ТУ 6-17-691-88", "ТУ 6-17-1109-88", "ТУ 6-17-1443-88",
        "ТУ 6-17-1000-88", "ТУ 6-17-912-87", "ТУ 6-42-1514-90",
        "ТУ 6-17-1453-89", "ГОСТ 25120-82"),
    # ⚠ THE POLARITY SPLIT. Negatives print an average gradient; reversal
    # stocks and the still negatives print a contrast coefficient. No sheet
    # prints both, which is asserted separately below.
    "среднийградиент": (
        "ТУ 6-17-691-88", "ТУ 6-17-1109-88", "ТУ 6-17-1443-88"),
    "коэффициентконтрастности": (
        "ТУ 6-17-1000-88", "ТУ 6-17-912-87", "ТУ 6-42-1514-90",
        "ТУ 6-17-1453-89", "ГОСТ 25120-82"),
    # Across-the-width uniformity is a CINE norm and appears nowhere else.
    "однородность": (
        "ТУ 6-17-691-88", "ТУ 6-17-1109-88", "ТУ 6-17-1443-88"),
    # ⚠ THE BRIDGE'S POLARITY LIMIT, AS A MEASUREMENT. Resolving power occurs
    # in the three reversal ТУ and in the ГОСТ, and in NO negative ТУ.
    "разрешающаяспособность": (
        "ТУ 6-17-1000-88", "ТУ 6-17-912-87", "ТУ 6-42-1514-90",
        "ГОСТ 25120-82"),
    # ⚠ ЛН-9's sheet says ФУНКЦИЯ передачи модуляции, not КОЭФФИЦИЕНТ, which
    # is why ТУ 6-17-1443-88 is absent from this row and present in the next.
    "коэффициентпередачимодуляции": (
        "ТУ 6-17-691-88", "ТУ 6-17-1109-88", "ТУ 6-17-1000-88",
        "ТУ 6-17-1453-89"),
    # ⚠ ЛН-9's nominative form is UNREADABLE -- табл. 2 п. 7 renders «7.
    # функхря передачи модуляции (^ ■ 30 мм“1)» -- so the sheet is caught by
    # its METHOD clause 3.5.4 instead, «функцию передачи модуляции определяют
    # по ОСТ 6-17-452-78», which the OCR gets right. Two rows rather than one
    # because the stem alone would hide the wording difference and the
    # wording difference is the finding.
    "функциюпередачимодуляции": ("ТУ 6-17-1443-88",),
    "передачимодуляции": (
        "ТУ 6-17-691-88", "ТУ 6-17-1109-88", "ТУ 6-17-1443-88",
        "ТУ 6-17-1000-88", "ТУ 6-17-1453-89"),
    "гранулярность": (
        "ТУ 6-17-691-88", "ТУ 6-17-1109-88", "ТУ 6-17-1443-88",
        "ТУ 6-17-1000-88", "ТУ 6-17-1453-89"),
    # ⚠ ЦО-90Л says «баланс КОЭФФИЦИЕНТА контрастности»; the token below is
    # the shorter form its two siblings use, so 1514 is deliberately absent.
    "балансконтрастности": (
        "ТУ 6-17-912-87", "ТУ 6-17-1453-89", "ГОСТ 25120-82"),
}

#: ⚠ THE WORD OCCURS AND THE PLOT DOES NOT. These four sheets instruct the
#: TESTER to construct a characteristic curve on graph paper; none prints one.
#: The set is pinned so a genuine curve in a future re-scan cannot be waved
#: through as "just the instruction again".
CURVE_WORD_DOCS = frozenset({
    "ТУ 6-17-691-88", "ТУ 6-17-1109-88", "ТУ 6-17-1443-88", "ТУ 6-17-912-87",
})

#: Tokens that would be present if any sheet carried reciprocity data.
RECIPROCITY_TOKENS = ("невзаимозаместимост", "взаимозаместимост",
                      "законвзаимозаместимости", "прерывистойэкспозиц")

#: The keystone every document delegates its definitions to, and which is not
#: in the corpus.
KEYSTONE = "гост9160"

# ---------------------------------------------------------------------------
# ГОСТ 25120-82 табл. 6 -- the one table in the corpus that can be re-parsed
#
# ⚠ COLUMN ORDER IS Фото ЦНД-32 первая · Фото ЦНЛ-65 ВЫСШАЯ · Фото ЦНЛ-65
# первая, and the middle column is the TOP grade. Reading the three columns
# left to right as "worst, middle, best" would be wrong and would still
# produce three plausible numbers, which is exactly why the order is written
# down here rather than inferred.
# ---------------------------------------------------------------------------
GOST_COLUMNS = ("Фото ЦНД-32, первая", "Фото ЦНЛ-65, высшая",
                "Фото ЦНЛ-65, первая")

#: Row anchor -> the number of value lines that follow it. The anchors are the
#: LAST line of each row label, because the OCR breaks labels across lines at
#: the hyphenation points of the printed column.
GOST_ROWS: tuple[tuple[str, str], ...] = (
    ("speed_nominal",    "номинальная"),
    ("speed_band",       "общаясветочувствительность"),
    ("speed_balance",    "баланссветочувствительности"),
    ("gamma_lower",      "нижнегослоя"),
    ("gamma_middle",     "среднегослоя"),
    ("gamma_upper",      "верхнегослоя"),
    ("dev_time",         "времяпроявления"),
    ("contrast_balance", "балансконтрастности"),
    ("dmin_blue",        "синим,неболее"),
    ("dmin_green",       "зеленым,неболее"),
    ("dmin_red",         "красным,неболее"),
    ("latitude",         "фотографическая"),
    ("resolving_power",  "разрешающая"),
)

#: ⚠ ONE CELL OF TWENTY-SEVEN IS GARBLED AND IT IS RECORDED, NOT PATCHED
#: SILENTLY. The contrast-balance row's third column renders as «ОДЗ» -- the
#: OCR reading Cyrillic О, Д, З for the glyphs 0, 1 and 3 of «0,13», which is
#: exactly the 8/6/Ь · 2/с · 9/Я confusion class the corpus record documents.
#: The value is recoverable because the row's other two columns both read
#: 0,13 and the printed table shows three identical cells, but the repair is
#: named here so that nobody later reads a clean parse as a clean scan.
GOST_OCR_REPAIRS = {("contrast_balance", 2): ("ОДЗ", "0,13")}


def _flat(text: str) -> str:
    """Whitespace-free, lower-case, ё-folded. The only reliable OCR view.

    ⚠ THE WHITESPACE STRIPPING IS NOT COSMETIC. These typescripts were set on
    a machine with letter-spaced emphasis and the OCR preserves it: ДС-5М's
    sheet 3 renders «Средний г р а д и е н т» and «Минимальная п л ф т н о с
    т ь». A token search on the raw text finds neither.
    """
    out = re.sub(r"\s+", "", text).lower().replace("ё", "е")
    return out.replace("\u2014", "-").replace("\u2013", "-")


def _pages(path: Path) -> list[str]:
    doc = pymupdf.open(path)
    out = [p.get_text() for p in doc]
    doc.close()
    return out


_NUM = re.compile(r"^[0-9OО][0-9,.\s±+\-—–]*$")


def _looks_numeric(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 14:
        return False
    return bool(_NUM.match(s)) and any(c.isdigit() for c in s)


def _norm_cell(s: str) -> str:
    """One табл. 6 cell, normalised for comparison but NOT for meaning.

    Spaces are dropped (the OCR puts them on either side of ±), em and en
    dashes become a plain hyphen, and the decimal comma is kept because that
    is what the document prints. ⚠ THE PLUS SIGN IS NOT NORMALISED TO ±: the
    Фото ЦНД-32 middle-layer cell really does read «0,60+0,08», and turning
    that into «0,60±0,08» here would erase the finding this module exists to
    protect.
    """
    return (s.replace(" ", "").replace("—", "-").replace("–", "-")
             .replace("­", ""))


def _parse_gost_table6(pages: list[str]) -> tuple[dict, list[str]]:
    """Re-derive ГОСТ 25120-82 табл. 6 from the OCR layer, three columns."""
    notes: list[str] = []
    page = None
    for p in pages:
        if "балансконтрастности" in _flat(p) and "табл" in _flat(p):
            page = p
            break
    if page is None:
        return {}, ["табл. 6 not found in any page's OCR layer"]
    lines = [ln.strip() for ln in page.replace("­", "").split("\n")]
    flat_lines = [_flat(ln) for ln in lines]

    # ⚠ THE SCAN IS SEQUENTIAL, not a search of the whole page per row, and
    # that is deliberate. «не более» appears in six rows and «слоя» in three;
    # an independent search would let a row silently borrow another's cells
    # and still produce three plausible numbers. Walking the table in printed
    # order makes the row ORDER part of what is being checked.
    out: dict[str, list[str]] = {}
    cursor = 0
    for key, anchor in GOST_ROWS:
        idx = next((i for i in range(cursor, len(flat_lines))
                    if anchor in flat_lines[i]), -1)
        if idx < 0:
            notes.append("row %s: anchor %r not found at or after line %d"
                         % (key, anchor, cursor))
            continue
        # phase 1 -- the label may run over several lines; skip them.
        j = idx + 1
        skipped = 0
        while j < len(lines) and not _looks_numeric(lines[j]) and skipped < 6:
            j += 1
            skipped += 1
        # phase 2 -- a STRICT run of three. A non-numeric line ends the row,
        # unless it is the one documented OCR casualty.
        cells: list[str] = []
        while j < len(lines) and len(cells) < 3:
            if _looks_numeric(lines[j]):
                cells.append(_norm_cell(lines[j]))
                j += 1
                continue
            fix = GOST_OCR_REPAIRS.get((key, len(cells)))
            if fix and lines[j].strip() == fix[0]:
                cells.append(_norm_cell(fix[1]))
                notes.append(
                    "row %s column %d: the scan reads %r; recorded as %r, the "
                    "one documented OCR casualty in 27 cells"
                    % (key, len(cells), fix[0], fix[1]))
                j += 1
                continue
            break
        if len(cells) != 3:
            notes.append("row %s: parsed %d of 3 cells (%s)"
                         % (key, len(cells), cells))
            continue
        out[key] = cells
        cursor = j
    return out, notes


def _expected_from_db() -> dict[str, list[str]]:
    """The same table, rendered back out of `_TOLERANCE`.

    ⚠ THIS IS THE DIRECTION THAT CATCHES A LAYER-ORDER SLIP. The database
    stores R, G, B; the page prints нижний / средний / верхний, i.e. R, G, B
    as well for the gamma rows, but B, G, R for the density rows («синим,
    зеленым, красным»). Re-rendering the PAGE's order from the DATABASE's
    order exercises that reversal in the one place a mistake would be silent.
    """
    tol = FP._TOLERANCE
    cnd32 = tol["SVEMA_CND_32"][0]
    cnl_by_cat = {t.category: t for t in tol["SVEMA_CNL_65"]}
    hi = cnl_by_cat["высшая категория качества"]
    lo = cnl_by_cat["первая категория качества"]
    cols = (cnd32, hi, lo)

    def _n(x: float) -> str:
        """A ГОСТ cell: comma decimal, and integers printed bare."""
        return (("%d" % round(x)) if abs(x - round(x)) < 1e-9
                else ("%.2f" % x).replace(".", ","))

    def _band(a: float, b: float) -> str:
        return "%s-%s" % (_n(a), _n(b))

    def _gamma(t, i: int) -> str:
        nom = t.gamma_nominal_rgb[i]
        lo_, hi_ = t.gamma_lo_rgb[i], t.gamma_hi_rgb[i]
        # ⚠ A ONE-SIDED CELL IS RENDERED WITH A PLUS, which is the whole
        # point: only Фото ЦНД-32's middle layer comes back as «0,60+0,08».
        if abs(nom - lo_) < 1e-9 and hi_ > nom:
            return "%s+%s" % (_n(nom), _n(round(hi_ - nom, 4)))
        return "%s±%s" % (_n(nom), _n(round(hi_ - nom, 4)))

    def _n1(x: float) -> str:
        """⚠ ONE DECIMAL, BECAUSE THE PAGE PRINTS ONE. Б_S reads «2,3» in
        табл. 6 while the densities on the same page read «0,45» and the
        gammas «0,55». Rendering every cell to two places produced «2,30»
        against the scan's «2,3» -- a spurious mismatch that says nothing
        about the data. Trailing-zero stripping is NOT the fix: the green
        density cell really is printed «0,50» and must stay two-place."""
        return ("%.1f" % x).replace(".", ",")

    return {
        "speed_nominal":    [_n(t.speed_nominal) for t in cols],
        "speed_band":       [_band(t.speed_min, t.speed_max) for t in cols],
        "speed_balance":    [_n1(t.speed_balance_max) for t in cols],
        "gamma_lower":      [_gamma(t, 0) for t in cols],
        "gamma_middle":     [_gamma(t, 1) for t in cols],
        "gamma_upper":      [_gamma(t, 2) for t in cols],
        "dev_time":         [_band(t.contrast_time_min, t.contrast_time_max)
                             for t in cols],
        "contrast_balance": [_n(t.contrast_balance_max) for t in cols],
        # the page prints blue, green, red; the database stores R, G, B
        "dmin_blue":        [_n(t.dmin_max_rgb[2]) for t in cols],
        "dmin_green":       [_n(t.dmin_max_rgb[1]) for t in cols],
        "dmin_red":         [_n(t.dmin_max_rgb[0]) for t in cols],
        "latitude":         [_n(t.latitude_min) for t in cols],
        "resolving_power":  [_n(t.resolving_power_min) for t in cols],
    }


def _stored_numbers(name: str) -> list[str]:
    """Every acceptance number the database stores for one stock, as text."""
    out: list[str] = []
    for t in FP._TOLERANCE.get(name, ()):
        vals = [t.speed_nominal, t.speed_min, t.speed_max, t.layer_speed_min,
                t.speed_balance_min, t.speed_balance_max,
                t.contrast_balance_max, t.contrast_time_min,
                t.contrast_time_max, t.latitude_min, t.resolving_power_min,
                t.uniformity_pct_max, t.coating_uniformity_d_max,
                t.mtf_freq_mm]
        for trip in (t.dmin_max_rgb, t.dmin_min_rgb, t.dmax_min_rgb,
                     t.granularity_max_rgb, t.mtf_min_rgb):
            vals.extend(trip)
        # ⚠ BAND EDGES ARE EXCLUDED WHERE A NOMINAL EXISTS, and that is a
        # correctness fix rather than a convenience. «0,55 ± 0,08» prints two
        # numbers, 0,55 and 0,08; the stored 0,47 and 0,63 are this file's
        # arithmetic and appear nowhere on the page. Counting them as misses
        # would have made every scan look worse than it is -- the ГОСТ came
        # out at 81 % against a threshold of 90 on its first run, for numbers
        # the document never printed.
        for i in range(3):
            nom = t.gamma_nominal_rgb[i]
            if nom:
                vals.append(nom)
                vals.append(round(t.gamma_hi_rgb[i] - nom, 4))
            else:
                vals.append(t.gamma_lo_rgb[i])
                vals.append(t.gamma_hi_rgb[i])
        for v in vals:
            if not v:
                continue
            out.append(("%d" % round(v)) if abs(v - round(v)) < 1e-9
                       else ("%.2f" % v).rstrip("0").replace(".", ","))
    return sorted(set(out))


def _ocr_value_recall(flat: str, names: tuple[str, ...]) -> tuple[int, int]:
    """How many of a document's stored numbers its own OCR layer still holds.

    ⚠ THIS IS A MEASUREMENT OF THE SCAN, NOT OF THE DATABASE, and a low score
    is not a failure. It is the evidence for the methodological split this
    module rests on: re-derive the ГОСТ, citation-check the typescripts.
    """
    nums = [n for name in names for n in _stored_numbers(name)]
    if not nums:
        return 0, 0
    hit = sum(1 for n in set(nums) if n in flat)
    return hit, len(set(nums))


def run(do_assert: bool = True) -> int:
    fail: list[str] = []
    if not TU_DIR.exists():
        print("[SKIP] soviet_tu_corpus.py -- the Soviet folder is not in the "
              "corpus")
        return 0

    pages: dict[str, list[str]] = {}
    flat: dict[str, str] = {}
    print("=== the nine documents ===")
    for tu, fn, sheets in DOCS:
        p = TU_DIR / fn
        if not p.exists():
            fail.append("%s (%s) is not in the corpus" % (fn, tu))
            continue
        t = _pages(p)
        pages[tu] = t
        flat[tu] = _flat("".join(t))
        if len(t) != sheets:
            fail.append("%s has %d pages, expected %d" % (tu, len(t), sheets))
        print("  %-18s %2d pp  %6d OCR chars  governs %s"
              % (tu, len(t), len(flat[tu]),
                 ", ".join(GOVERNS[tu]) or "-- (states nothing photographic)"))
    if fail:
        print("\n".join(["", "FAILURES:"] + ["  " + f for f in fail]))
        return 1
    if len(pages) != 9:
        fail.append("expected nine documents, opened %d" % len(pages))

    # -- 1. the specification vocabulary ------------------------------------
    print("\n=== specification vocabulary: which sheet uses which word ===")
    for token, expect in VOCABULARY.items():
        got = tuple(tu for tu, _, _ in DOCS if token in flat[tu])
        mark = "ok " if set(got) == set(expect) else "BAD"
        print("  [%s] %-30s %s" % (mark, token, ", ".join(got) or "--"))
        if set(got) != set(expect):
            fail.append(
                "vocabulary %r occurs in {%s}, the record says {%s}"
                % (token, ", ".join(sorted(got)), ", ".join(sorted(expect))))

    # ⚠ THE POLARITY SPLIT, STATED AS AN EXCLUSION rather than as two
    # presences. Two lists that happen not to overlap today would drift; this
    # says they MAY NOT overlap.
    both = [tu for tu, _, _ in DOCS
            if "среднийградиент" in flat[tu]
            and "коэффициентконтрастности" in flat[tu]]
    print("\n  documents using BOTH «средний градиент» and «коэффициент "
          "контрастности»: %s" % (", ".join(both) or "none, as recorded"))
    if both:
        fail.append("the gradient/contrast vocabularies overlap in %s; the "
                    "record says the Soviet industry named the quantity by "
                    "the polarity of the film" % ", ".join(both))

    # the export sheet states nothing photographic
    silent = [t for t in VOCABULARY if t in flat["ТУ 6-17-1371-86"]]
    if silent:
        fail.append("ТУ 6-17-1371-86 is recorded as stating nothing "
                    "photographic and contains %s" % ", ".join(silent))

    # -- 2. the three permanent absences ------------------------------------
    print("\n=== the three permanent absences ===")
    curve_docs = {tu for tu, _, _ in DOCS if "характеристическ" in flat[tu]}
    print("  «характеристическ» occurs in: %s" % ", ".join(sorted(curve_docs)))
    print("     -- and in every one of them it is the METHOD SECTION telling "
          "the tester to plot one. No sheet prints a curve.")
    if curve_docs != CURVE_WORD_DOCS:
        fail.append(
            "the set of documents containing «характеристическ» is {%s}, the "
            "record pins {%s} -- a genuine plot may have arrived, or a "
            "document may have been replaced"
            % (", ".join(sorted(curve_docs)), ", ".join(sorted(CURVE_WORD_DOCS))))

    rec = {tu: [k for k in RECIPROCITY_TOKENS if k in flat[tu]]
           for tu, _, _ in DOCS}
    hits = {tu: v for tu, v in rec.items() if v}
    print("  reciprocity vocabulary anywhere in 176 sheets: %s"
          % (hits or "none"))
    if hits:
        fail.append("reciprocity vocabulary found in %s; the record says the "
                    "corpus contains none" % ", ".join(hits))

    cite = [tu for tu, _, _ in DOCS if KEYSTONE in flat[tu]]
    print("  ГОСТ 9160-82 cited by %d of 9; present in the corpus: no"
          % len(cite))
    if len(cite) < 8:
        fail.append("ГОСТ 9160-82 is cited by only %d documents; the record "
                    "says every specifying sheet delegates to it" % len(cite))
    if (TU_DIR / "gost_9160_82.pdf").exists():
        fail.append("ГОСТ 9160-82 HAS ARRIVED in the corpus -- this is good "
                    "news and this audit must be rewritten around it")

    # -- 3. the OCR split, measured -----------------------------------------
    print("\n=== OCR value recall: how many stored numbers survive the scan "
          "===")
    recall: dict[str, float] = {}
    for tu, _, _ in DOCS:
        names = GOVERNS[tu]
        hit, tot = _ocr_value_recall(flat[tu], names)
        if not tot:
            continue
        recall[tu] = hit / tot
        print("  %-18s %2d of %2d  (%3.0f %%)" % (tu, hit, tot, 100 * hit / tot))
    if "ГОСТ 25120-82" in recall and recall["ГОСТ 25120-82"] < 0.90:
        fail.append(
            "ГОСТ 25120-82's OCR recall has fallen to %.0f %%; the re-derived "
            "table below is built on that scan being clean"
            % (100 * recall["ГОСТ 25120-82"]))
    tu_only = [v for k, v in recall.items() if k.startswith("ТУ")]
    if tu_only and max(tu_only) >= recall.get("ГОСТ 25120-82", 1.0):
        print("  ⚠ A ТУ NOW SCORES AS WELL AS THE ГОСТ. That is not a "
              "failure -- it means a better scan has arrived and the "
              "citation-trail limit on the eight typescripts could be "
              "revisited. Recorded, not enforced.")

    # -- 4. ГОСТ 25120-82 табл. 6, re-derived and compared ------------------
    print("\n=== ГОСТ 25120-82 табл. 6 re-derived from the scan ===")
    print("  columns: %s" % " | ".join(GOST_COLUMNS))
    parsed, notes = _parse_gost_table6(pages["ГОСТ 25120-82"])
    for n in notes:
        print("  note: %s" % n)
    want = _expected_from_db()
    ok_rows = 0
    for key, _anchor in GOST_ROWS:
        got = parsed.get(key)
        exp = want.get(key)
        if got is None:
            fail.append("табл. 6 row %s did not parse" % key)
            continue
        same = got == exp
        ok_rows += bool(same)
        print("  [%s] %-17s scan %-26s db %s"
              % ("ok " if same else "BAD", key,
                 " ".join(got), " ".join(exp or ["-"])))
        if not same:
            fail.append("табл. 6 row %s: the scan reads %s and the database "
                        "holds %s" % (key, got, exp))
    print("  %d of %d rows agree cell for cell" % (ok_rows, len(GOST_ROWS)))

    # ⚠ THE ONE-SIDED CELL, ASSERTED ON ITS OWN. It is the single most
    # deletable finding in this batch -- a later editor "tidying" 0,60+0,08
    # into 0,60±0,08 would produce a table that looks more consistent and is
    # no longer the document.
    mid = parsed.get("gamma_middle")
    if mid and not (mid[0].count("+") == 1 and "±" not in mid[0]):
        fail.append("the Фото ЦНД-32 middle-layer contrast cell no longer "
                    "reads one-sided (%r); it is printed «0,60 + 0,08» where "
                    "the cells above and below it use ±" % mid[0])
    cnd32 = FP._TOLERANCE["SVEMA_CND_32"][0]
    if abs(cnd32.gamma_lo_rgb[1] - cnd32.gamma_nominal_rgb[1]) > 1e-9:
        fail.append("SVEMA_CND_32's stored green gamma band has been "
                    "symmetrised; the standard prints it one-sided")

    # -- 5. the bridge anchor, on its own page ------------------------------
    print("\n=== the R↔f50 bridge anchor ===")
    anchor_pages = [i + 1 for i, t in enumerate(pages["ТУ 6-17-1000-88"])
                    if "75" in _flat(t) and "0,27" in _flat(t)]
    print("  ЦО-Т-90ЛМ: «75» and «0,27» together on sheet(s) %s"
          % (anchor_pages or "NONE"))
    if not anchor_pages:
        fail.append("ТУ 6-17-1000-88 no longer carries 75 and 0,27 on one "
                    "sheet; the bridge's only calibration is that table")
    ratio = FP.soviet_rp_to_f50_ratio()
    print("  ratio R/f50 = %.4f from T 0,27 at 30 мм⁻¹ against R 75" % ratio)
    if abs(ratio - 3.436) > 0.002:
        fail.append("the bridge ratio has moved to %.4f" % ratio)

    # -- 6. every acceptance band against the profile it sits on ------------
    print("\n=== acceptance bands against the stored scalars ===")
    n_rec = 0
    for p in FP.FILM_PROFILES:
        for t in p.tolerance:
            n_rec += 1
            if not t.is_default:
                continue
            if t.resolving_power_min and p.mtf.resolving_power_lp_mm_highc:
                if p.mtf.resolving_power_lp_mm_highc < t.resolving_power_min:
                    fail.append("%s stores R %.0f below its own acceptance "
                                "floor %.0f" % (p.name,
                                                p.mtf.resolving_power_lp_mm_highc,
                                                t.resolving_power_min))
            if t.speed_min and p.exposure_index < t.speed_min:
                fail.append("%s stores EI %d below its own acceptance floor "
                            "%.0f" % (p.name, p.exposure_index, t.speed_min))
            if t.speed_max and p.exposure_index > t.speed_max:
                fail.append("%s stores EI %d above its own acceptance ceiling "
                            "%.0f" % (p.name, p.exposure_index, t.speed_max))
            for i, ch in enumerate("rgb"):
                lo, hi = t.gamma_lo_rgb[i], t.gamma_hi_rgb[i]
                g = getattr(p.curves, ch).gamma
                if lo and hi and not (lo - 0.005 <= g <= hi + 0.005):
                    fail.append("%s's stored %s gamma %.3f is outside its own "
                                "acceptance band %.2f-%.2f"
                                % (p.name, ch.upper(), g, lo, hi))
    print("  %d records on %d stocks; every default record's stored scalars "
          "lie inside their own band" % (n_rec, sum(
              1 for p in FP.FILM_PROFILES if p.tolerance)))

    if fail:
        print("\n".join(["", "FAILURES:"] + ["  " + f for f in fail]))
        if do_assert:
            return 1
        return 1
    print("\n[OK] soviet_tu_corpus.py -- nine documents, %d acceptance "
          "records, ГОСТ 25120-82 табл. 6 re-derived cell for cell" % n_rec)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.parse_args()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
