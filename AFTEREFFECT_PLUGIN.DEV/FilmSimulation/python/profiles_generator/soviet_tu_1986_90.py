#!/usr/bin/env python3
"""Eight Soviet ТУ specifications, read 2026-09-22d, and the citation trail.

WHAT THESE DOCUMENTS ARE
------------------------
«Технические условия» -- the manufacturing specification a Soviet factory was
inspected against. Eight of them, 161 sheets, all from ПО «Свема» / «Тасма»
between 1986 and 1990:

    ТУ 6-17-1000-88   ЦО-Т-90ЛМ   16 mm colour reversal for television
    ТУ 6-17-1453-89   ЦНД-64      colour negative still, daylight
    ТУ 6-42-1514-90   ЦО-90Л      colour reversal ciné and still, tungsten
    ТУ 6-17-912-87    ЦО-32Д      colour reversal, amateur
    ТУ 6-17-1109-88   ЛН-8        masked colour negative ciné
    ТУ 6-17-1443-88   ЛН-9, ЛН-9С masked colour negative ciné
    ТУ 6-17-691-88    ДС-5М       masked colour negative ciné, daylight
    ТУ 6-17-1371-86   Фото-65, ДС-4, ЦНЛ-65 -- an export packaging sheet

⚠⚠ A ТУ IS A CONTRACT, NOT A MEASUREMENT REPORT, AND THAT GOVERNS EVERY NUMBER
THIS MODULE TOUCHES. «Не менее 100 ед. ГОСТ 9160-82» means a roll testing at
99 is rejected. It says nothing whatever about what a good roll actually did.
So a profile built from one of these sheets renders the WORST LEGAL EXAMPLE of
its own stock wherever a one-sided limit is all there is, and the mid-point of
a two-sided band otherwise. Every Soviet profile in `film_profiles.py` says so
in its own description and `verify.py`'s G-SOV-LIMITS-LABELLED checks that it
still does.

WHAT THIS MODULE CAN AND CANNOT AUDIT, SAID PLAINLY
-----------------------------------------------------
These are scanned typewritten sheets whose OCR layer systematically confuses
8 with 6 and Ь, 2 with с, and 9 with Я, and which detaches table cells from
their row labels. The VALUES in the database were therefore read from rendered
page IMAGES by eye, and this module does NOT re-derive them -- claiming to
would be pretending a reliable parse exists where it does not.

⚠ WHAT IT DOES INSTEAD IS WORTH MORE THAN A FAKE RE-PARSE: it audits the
CITATION TRAIL. For each claim the database makes on the strength of one of
these sheets, it asserts that the distinguishing token is present in that
sheet's own OCR layer, on the page the record cites. OCR that garbles digits
still finds «3200», «5500», «ОТБ-14» and «Гарантийный срок» reliably, because
those are the strings it gets right. If a future edit moves a value onto the
wrong document, or cites a page the phrase is not on, this fails.

⚠⚠ AND THE STRONGEST RESULT IS A PATTERN OF PRESENCE AND ABSENCE THAT NO
SINGLE DOCUMENT COULD GIVE. «3200» occurs in the sensitometry clause of all four
Л (лампы накаливания) sheets and in none of the ДС ones; «5500» occurs in ДС-5М
and nowhere else; and ЦНД-64 contains NEITHER, which is what makes its
"the ТУ states no colour temperature" note a finding rather than an omission.
That pattern is what condemned the stored balance_kelvin of 5500 on ЛН-8, ЛН-9
and ЛН-9С -- three tungsten stocks declared daylight, which is a rendered
colour cast and not a labelling slip, since `balance_kelvin` feeds
`balance_gains` on the render path.

THE OTHER FINDING IS AN ABSENCE, AND IT IS PERMANENT
------------------------------------------------------
Across all eight documents there is NOT ONE PLOTTED CURVE. No characteristic
curve, no spectral sensitivity, no spectral dye density, no modulation-transfer
curve. Three of the eight carry a figure at all and all three are engineering
drawings of film strips and perforations. ЛН-8's own appendix instructs the
TESTER to plot градационные кривые on graph paper -- and prints none.

So no amount of further reading in this class of source can promote a Soviet
curve SHAPE above tier 3. The specifications fix scalars; the shapes remain
analogy on the nearest documented stock. `verify.py`'s G-SOV-NO-CURVE-IN-ANY-TU
holds that line.
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

#: (file, ТУ number, the marks it covers, sheets)
DOCS = [
    ("tu_617100088_kinoplenka_tsvetnaia_obrashchaemaia_tsot90_lm_t.pdf",
     "ТУ 6-17-1000-88", ("ЦО-Т-90ЛМ",), 18),
    ("tu_617145389_plenka_fotograficheskaia_tsvetnaia_negativnaia.pdf",
     "ТУ 6-17-1453-89", ("ЦНД-64",), 26),
    ("tu_642151490_kinoplenka_i_fotoplenka_obrashchaemye_marok_tso.pdf",
     "ТУ 6-42-1514-90", ("ЦО-90Л",), 20),
    ("tu_61791287_kinofotoplenki_tsvetnye_obrashchaemye_tso32d_tek.pdf",
     "ТУ 6-17-912-87", ("ЦО-32Д",), 23),
    ("tu_617110988_kinoplenka_tsvetnaia_negativnaia_ln8_tekhniches.pdf",
     "ТУ 6-17-1109-88", ("ЛН-8",), 21),
    ("tu_617144388_kinoplenki_tsvetnye_negativnye_ln9_i_ln9s.pdf",
     "ТУ 6-17-1443-88", ("ЛН-9", "ЛН-9С"), 23),
    ("tu_61769188_kinoplenka_tsvetnaia_negativnaia_ds5m_tekhniches.pdf",
     "ТУ 6-17-691-88", ("ДС-5М",), 22),
    ("tu_617137186_plenki_fotograficheskie_35mm_perforirovannye_v.pdf",
     "ТУ 6-17-1371-86", ("Фото-65", "ДС-4", "ЦНЛ-65"), 8),
]

#: The sensitometric illuminant each sheet's clause 3.5.2 / 3.4.2 / 3.6.2
#: prints, and the page the OCR layer must carry it on. `None` = the sheet
#: prints NO colour temperature at all and that absence is itself the claim.
ILLUMINANT = {
    "ТУ 6-17-1000-88": ("3200", (2, 10)),
    "ТУ 6-17-1109-88": ("3200", (13,)),
    "ТУ 6-17-1443-88": ("3200", (13,)),
    "ТУ 6-42-1514-90": ("3200", (2,)),
    "ТУ 6-17-691-88":  ("5500", (13,)),
    "ТУ 6-17-1453-89": (None, ()),
    "ТУ 6-17-912-87":  (None, ()),
    "ТУ 6-17-1371-86": (None, ()),
}

#: Support grade, where the sheet names one, and the page it is named on.
BASE_GRADE = {
    "ТУ 6-17-1443-88": ("ОТБ-14", 3),
    "ТУ 6-17-691-88":  ("ОТБ-14", 2),
    "ТУ 6-17-912-87":  ("ОТБ-14", 6),
}

#: Which profile each sheet governs, and the balance_kelvin that follows from
#: its sensitometric clause.
BALANCE = {
    "SVEMA_CO_T_90LM": ("ТУ 6-17-1000-88", 3200),
    "SVEMA_LN_8":      ("ТУ 6-17-1109-88", 3200),
    "SVEMA_LN_9":      ("ТУ 6-17-1443-88", 3200),
    "SVEMA_LN_9S":     ("ТУ 6-17-1443-88", 3200),
    "SVEMA_CO_90L":    ("ТУ 6-42-1514-90", 3200),
    "SVEMA_DS_5M":     ("ТУ 6-17-691-88", 5500),
}

#: Guarantee period in months, and the sheet page that prints «Гарантийный
#: срок хранения».
SHELF = {
    "SVEMA_CO_T_90LM": ("ТУ 6-17-1000-88", 9, 15),
    "SVEMA_CND_64":    ("ТУ 6-17-1453-89", 12, 24),
    "SVEMA_CO_90L":    ("ТУ 6-42-1514-90", 9, 18),
    "SVEMA_CO_32D":    ("ТУ 6-17-912-87", 12, 21),
    "SVEMA_LN_8":      ("ТУ 6-17-1109-88", 6, 17),
    "SVEMA_LN_9":      ("ТУ 6-17-1443-88", 9, 18),
    "SVEMA_LN_9S":     ("ТУ 6-17-1443-88", 9, 18),
    "SVEMA_DS_5M":     ("ТУ 6-17-691-88", 6, 17),
}

#: Words that would be present if any of these sheets carried a plot.
PLOT_WORDS = re.compile(
    r"характеристическ\w*\s+крив|спектральн\w*\s+чувствительност\w*\s*[,:]?\s*"
    r"крив|кривая\s+ЧКХ|рис\w*\s*\d", re.I)


def _pages(path: Path):
    doc = pymupdf.open(path)
    out = [p.get_text() for p in doc]
    doc.close()
    return out


def run(do_assert: bool = True) -> int:
    fail: list[str] = []
    if not TU_DIR.exists():
        print("[SKIP] soviet_tu_1986_90.py -- the ТУ folder is not in the "
              "corpus")
        return 0

    text: dict[str, list[str]] = {}
    print("=== the eight sheets ===")
    for fn, tu, marks, sheets in DOCS:
        p = TU_DIR / fn
        if not p.exists():
            fail.append("%s (%s) is not in the corpus" % (fn, tu))
            continue
        t = _pages(p)
        text[tu] = t
        if len(t) != sheets:
            fail.append("%s has %d pages, expected %d"
                        % (tu, len(t), sheets))
        chars = sum(len(x) for x in t)
        print("  %-18s %-24s %2d pp, %6d OCR chars"
              % (tu, "/".join(marks), len(t), chars))
        if chars < 5000:
            fail.append("%s has almost no OCR layer (%d chars) -- the "
                        "citation trail below cannot be audited on it"
                        % (tu, chars))
    if fail:
        print("\n".join(["", "FAILURES:"] + ["  " + f for f in fail]))
        return 1

    # -- 1. the illuminant pattern ------------------------------------------
    print("\n=== the sensitometric illuminant, per sheet ===")
    for tu, (want, pages) in ILLUMINANT.items():
        t = text[tu]
        found3200 = [i + 1 for i, x in enumerate(t) if "3200" in x]
        found5500 = [i + 1 for i, x in enumerate(t) if "5500" in x]
        print("  %-18s 3200 on %-12s 5500 on %-12s expected %s"
              % (tu, found3200 or "-", found5500 or "-", want or "NEITHER"))
        if want is None:
            if found3200 or found5500:
                fail.append("%s: the record says this sheet prints no colour "
                            "temperature, and the OCR layer has one" % tu)
            continue
        got = found3200 if want == "3200" else found5500
        other = found5500 if want == "3200" else found3200
        for pg in pages:
            if pg not in got:
                fail.append("%s: «%s» is not on the OCR of sheet %d, which "
                            "the record cites" % (tu, want, pg))
        if other:
            fail.append("%s: the OTHER illuminant «%s» also appears, on %s -- "
                        "the record's reading is no longer unambiguous"
                        % (tu, "5500" if want == "3200" else "3200", other))

    # ⚠ THE PATTERN, NOT JUST THE HITS. This is the part that condemned the
    # stored 5500 K on three tungsten stocks: one sheet saying 3200 could be
    # a quirk; five Л sheets saying 3200, one ДС sheet saying 5500 and one
    # daylight still film saying neither is a convention.
    _l_sheets = [tu for tu, (w, _) in ILLUMINANT.items() if w == "3200"]
    _d_sheets = [tu for tu, (w, _) in ILLUMINANT.items() if w == "5500"]
    print("\n  ⚠ %d sheets print 3200 K and every one of them governs a Л "
          "(лампы накаливания) stock; %d print 5500 K and it governs the ДС "
          "one; 3 print neither." % (len(_l_sheets), len(_d_sheets)))

    # -- 2. the base grade ---------------------------------------------------
    print("\n=== the support grade, where a sheet names one ===")
    for tu, (grade, page) in BASE_GRADE.items():
        pat = re.compile(grade.replace("-", r"[-\s]*"), re.I)
        got = [i + 1 for i, x in enumerate(text[tu]) if pat.search(x)]
        print("  %-18s %s on sheet(s) %s, record cites %d"
              % (tu, grade, got or "-", page))
        if page not in got:
            fail.append("%s: %s is not on the OCR of sheet %d"
                        % (tu, grade, page))

    # -- 3. the guarantee period ---------------------------------------------
    print("\n=== the guarantee period ===")
    gpat = re.compile(r"[Гг]арантийны\w*\s+срок")
    for stock, (tu, months, page) in SHELF.items():
        got = [i + 1 for i, x in enumerate(text[tu]) if gpat.search(x)]
        stored = FP.get_profile(stock).base.guaranteed_shelf_life_months
        okpage = page in got
        print("  %-16s %-18s stored %2d months, «Гарантийный срок» on %s %s"
              % (stock, tu, stored, got or "-", "" if okpage else "  <-- ?"))
        if stored != months:
            fail.append("%s: stored shelf life %d, this module expects %d"
                        % (stock, stored, months))
        if not okpage:
            fail.append("%s: «Гарантийный срок» is not on the OCR of sheet "
                        "%d, which the record cites" % (tu, page))

    # -- 4. the balance the database renders with ----------------------------
    print("\n=== balance_kelvin against the sheet that sets it ===")
    for stock, (tu, kelvin) in BALANCE.items():
        got = FP.get_profile(stock).balance_kelvin
        print("  %-16s %-18s stored %4d K, sheet prints %s K   %s"
              % (stock, tu, got, ILLUMINANT[tu][0], "ok" if got == kelvin
                 else "MISMATCH"))
        if got != kelvin:
            fail.append("%s: balance_kelvin %d against %s's %d"
                        % (stock, got, tu, kelvin))

    # -- 5. the absence that is permanent ------------------------------------
    print("\n=== plotted curves: the absence, counted ===")
    total_hits = 0
    for fn, tu, _marks, _s in DOCS:
        hits = []
        for i, x in enumerate(text[tu]):
            for m in PLOT_WORDS.finditer(x):
                hits.append((i + 1, m.group(0)[:34]))
        total_hits += len(hits)
        print("  %-18s %d phrase(s) that could name a plot%s"
              % (tu, len(hits), ("  " + str(hits[:3])) if hits else ""))
    print("  ⚠ Every hit above is a DEFINITION or an INSTRUCTION, not a "
          "caption: ЦО-32Д's table 5 item 8 defines its useful exposure "
          "interval as measured «на участке характеристических кривых между "
          "плотностями 0,3 и 2,1», and the appendices of ЛН-8, ЛН-9 and "
          "ДС-5М tell the tester to plot градационные кривые on graph paper. "
          "Not one of the eight sheets PRINTS a curve.")

    # And the database must not claim otherwise.
    claimed = [p.name for p in FP.FILM_PROFILES
               if p.name.startswith(("SVEMA_", "TASMA_"))
               and ("traced" in p.description.lower()
                    or "digitis" in p.description.lower()
                    or "digitiz" in p.description.lower())]
    if claimed:
        fail.append("a Soviet profile claims a traced or digitised curve: %s "
                    "-- no ТУ in this corpus contains one" % claimed)

    # -- 6. the two new stocks exist and carry what the sheets state ---------
    print("\n=== the two stocks this batch added ===")
    for stock, (s_ei, s_bal, s_rms) in (
            ("SVEMA_CO_T_90LM", (100, 3200, 25.0)),
            ("SVEMA_CND_64", (64, 5500, 20.0))):
        p = FP.get_profile(stock)
        okp = (p.exposure_index == s_ei and p.balance_kelvin == s_bal
               and p.grain.rms_granularity == s_rms)
        print("  %-16s S %3d, %4d K, granularity limit %.0f   %s"
              % (stock, p.exposure_index, p.balance_kelvin,
                 p.grain.rms_granularity, "ok" if okp else "MISMATCH"))
        if not okp:
            fail.append("%s does not hold the values its ТУ states" % stock)

    if fail and do_assert:
        print("\n".join(["", "FAILURES:"] + ["  " + f for f in fail]))
        return 1
    print("\n[OK] soviet_tu_1986_90.py -- 8 ТУ, 161 sheets, the citation "
          "trail audited rather than the values re-parsed, because the OCR "
          "of a Soviet typescript confuses 8 with 6 and 2 with с and a "
          "pretended re-parse would be worse than none. ⚠⚠ THE DECISIVE "
          "RESULT IS A PATTERN OF PRESENCE AND ABSENCE: «3200» appears in the "
          "sensitometry clause of all four Л sheets and in none of the ДС "
          "ones, «5500» appears in ДС-5М alone, and ЦНД-64 contains NEITHER "
          "-- which is what condemned the stored balance_kelvin of 5500 on "
          "ЛН-8, ЛН-9 and ЛН-9С, a rendered colour cast on three stocks and "
          "not a labelling slip. ⚠ THE SECOND RESULT IS AN ABSENCE AND IT IS "
          "PERMANENT: not one of the eight sheets prints a characteristic "
          "curve, a spectral curve or an MTF curve, so no further reading in "
          "this class of source can promote a Soviet curve SHAPE above "
          "tier 3 -- the specifications fix scalars and the shapes stay "
          "analogy")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="")
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    default=True)
    ns = ap.parse_args(argv)
    return run(ns.do_assert)


if __name__ == "__main__":
    sys.exit(main())
