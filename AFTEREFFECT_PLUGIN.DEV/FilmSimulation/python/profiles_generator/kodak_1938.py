#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""«Eastman Professional Films», 1938 -- the Kodak Research Laboratories speed
scale, and the conversion to the American Standard that queue P65 asked for.

WHAT THIS SOURCE IS
-------------------
**Eastman Kodak Company, «Eastman Professional Films», Rochester N.Y., 1938**
-- a 36-page trade booklet for the professional sheet-film line. Its entire
quantitative content is two tables: fifteen film speeds and a filter-factor
matrix. There is no characteristic curve, no spectral curve and no modulation
transfer anywhere in it.

⚠⚠ THE SPEEDS ARE ON A SCALE THAT PREDATES THE AMERICAN STANDARD, AND THE
BOOKLET SAYS SO IN ITS OWN WORDS. Page 8, immediately above the table:

    "The speed numbers of the various brands of Eastman Professional Films are
     shown in the following table. The figures given were obtained according to
     the standard system of speed evaluation employed in the Kodak Research
     Laboratories. They are for light of sunlight quality and are not valid
     when the materials are exposed to light of different quality, such as
     incandescent tungsten."

Two things follow from that sentence and both matter. The scale is Kodak's own
house scale, not ASA -- the American Standard z38.2.1 is five years away in
1943. And the numbers are SUNLIGHT ONLY, by Kodak's own warning, so the only
legitimate comparison is against a DAYLIGHT index and never against a tungsten
one.

WHAT QUEUE P65 ASKED FOR, AND WHAT IT GOT
------------------------------------------
The row's blocker was stated exactly: *"a single film appearing on BOTH the
Kodak Research Laboratories sunlight scale and the 1943 American Standard,
anywhere. One such pair makes the whole table usable."*

There are **three** such films, because the seventh edition of «Kodak Films»
(1956) publishes American Standard indexes for three emulsions this booklet
also rates. Two of them agree on a ratio of **exactly 4.00** and the third does
not:

    Super Panchro-Press        KRL 500   AS daylight 125    x 4.00
    Portrait Panchromatic      KRL 200   AS daylight  50    x 4.00
    Commercial                 KRL  40   AS daylight  25    x 1.60

⚠⚠ THE TWO THAT AGREE ARE THE EVIDENCE AND THE THIRD IS NOT AVERAGED INTO
THEM. Two independently-rated films landing on the same ratio to three
significant figures is not a coincidence a pair of emulsion changes would
produce: the 1956 Super Panchro-Press is a **Type B** re-coating, explicitly
renamed, while Portrait Panchromatic carries no type suffix at all, so the two
have different eighteen-year histories and the same ratio. That is the
signature of a SCALE, which is what the ratio has to be for the table to be
usable at all.

⚠ AND THE THIRD IS RECORDED, NOT DISCARDED. Commercial is the one
non-colour-sensitive film of the three -- its 1956 daylight-to-tungsten ratio
of 25/6 is the 4.2 only a blue-sensitive emulsion shows -- and it is the only
one of the three whose two ratings differ by anything other than 4.00. Either
the emulsion was slowed between the two editions, or the house scale treated
blue-sensitive stock differently, and this corpus contains nothing that decides
which. So the conversion below is adopted **for panchromatic sheet film only**
and `CONVERSION_REFUSED` names the case it does not cover.

⚠ ONE ARITHMETIC OBSERVATION, RECORDED AND NOT USED. The 1956 book's own page
25 states that the American Standard carries "a safety factor of 2.5". If the
Kodak Research Laboratories number had no safety factor at all, the ratio would
be 2.5 and not 4.0; the residual is 4.0 / 2.5 = 1.6, which is numerically the
Commercial ratio. That is very probably a coincidence of two round numbers and
it is written here so that nobody has to rediscover it before dismissing it.

WHAT IS STILL NOT ADOPTABLE, AND WHY THE ROW CLOSES ANYWAY
-----------------------------------------------------------
⚠ NONE OF THE FIFTEEN FILMS IS IN THE DATABASE, and none becomes one here: a
speed and a filter factor do not make a profile, and this booklet prints
nothing else. What closes P65 is that the SCALE is now convertible and the
table is stored rather than lost, so the day one of these emulsions acquires a
curve from another source, its 1938 speed can be carried onto it.
"""

from __future__ import annotations

import argparse
import os
import re

SHEET = os.path.join("KODAK", "kodak_prof_films-1938.pdf")

#: Page 8 of the printed book; PDF page 12. (name, class, KRL sunlight speed)
#: ⚠ TRANSCRIBED FROM THE RENDERED PAGE, NOT FROM THE TEXT LAYER. The scan's
#: OCR is poor enough to turn "12" into "L2" and "40" into ",40" on this very
#: table, so the values are read by eye and `check_text_layer` then requires
#: every one of them to appear as a token on that page -- a typo in the
#: transcription fails, while an OCR artefact somewhere else does not.
SPEEDS_1938: tuple[tuple[str, str, int], ...] = (
    ("Eastman Super Panchro-Press Safety Film",        "panchromatic", 500),
    ("Eastman Panchro-Press Safety Film",              "panchromatic", 300),
    ("Eastman Super Sensitive Panchromatic Film",      "panchromatic", 200),
    ("Eastman Portrait Panchromatic Film",             "panchromatic", 200),
    # ⚠ THE ASTERISK IS KODAK'S AND IT IS A CONDITION, NOT A FOOTNOTE MARKER
    # TO DROP: "*With development in D-76." The other fourteen speeds carry no
    # stated developer at all, which is a real limitation of the whole table.
    ("Eastman Safety Panatomic Film",                  "panchromatic", 150),
    ("Eastman Commercial Panchromatic Film",           "panchromatic", 120),
    ("Eastman Panchromatic Process Film",              "panchromatic",  16),
    ("Eastman Super Speed Ortho Portrait, Antihalation", "ortho",      220),
    ("Eastman Super Speed Ortho Portrait Film, Regular", "ortho",      200),
    ("Eastman Safety Ortho Press Film",                "ortho",        160),
    ("Eastman Par Speed Portrait Film",                "ortho",        120),
    ("Eastman Commercial Ortho Film",                  "ortho",        100),
    ("Eastman Commercial Film",                        "blue",          40),
    ("Eastman Commercial Matte Film",                  "blue",          40),
    ("Eastman Process Film",                           "blue",          12),
)

#: The developer the one asterisked speed was measured in.
PANATOMIC_DEVELOPER = "KODAK D-76"

#: Page 7; PDF page 11. Filter factors, for the two illuminant groups the
#: booklet prints. A dash in the original means the combination is not given.
#: (films sharing the row, {filter: factor} sunlight, {filter: factor} tungsten)
#: ⚠ THE ROWS ARE BRACED IN THE ORIGINAL -- four films share one line of
#: numbers, and that brace is the measurement's own scope. Splitting the row
#: into four would claim four measurements where Kodak published one.
FILTER_FACTORS_1938: tuple[tuple[tuple[str, ...], dict, dict], ...] = (
    (("Commercial Ortho",),
     {"K1": 3.0, "K2": 5.0}, {"K1": 2.5, "K2": 4.0}),
    (("Super Speed Ortho Portrait",),
     {"K1": 2.5, "K2": 3.5}, {"K1": 2.0, "K2": 2.5}),
    (("Ortho Press", "Super Speed Ortho Portrait, Antihalation"),
     {"K1": 2.0, "K2": 2.5}, {"K1": 1.5, "K2": 2.0}),
    (("Portrait Panchromatic",),
     {"K1": 1.5, "K2": 2.0}, {"K1": 1.5, "K2": 1.5, "X1": 3.0}),
    (("Super Sensitive Panchromatic", "Safety Panatomic",
      "Super Panchro-Press, Safety", "Panchro-Press, Safety"),
     {"K1": 1.5, "K2": 2.0, "X1": 5.0},
     {"K1": 1.5, "K2": 1.5, "X2": 5.0}),
    (("Commercial Panchromatic",),
     {"K1": 2.0, "K2": 3.0}, {"K1": 1.5, "K2": 2.0}),
)

#: The three films rated on BOTH scales.
#: (1938 booklet name, 1956 «Kodak Films» name, KRL sunlight, AS daylight)
BRIDGE: tuple[tuple[str, str, int, int], ...] = (
    ("Eastman Super Panchro-Press Safety Film",
     "KODAK SUPER PANCHRO-PRESS, TYPE B, SHEET FILM", 500, 125),
    ("Eastman Portrait Panchromatic Film",
     "KODAK PORTRAIT PANCHROMATIC SHEET FILM", 200, 50),
    ("Eastman Commercial Film",
     "KODAK COMMERCIAL SHEET FILM", 40, 25),
)

#: The adopted conversion, and the class it is adopted for.
KRL_TO_AMERICAN_STANDARD = 4.00
CONVERSION_CLASS = "panchromatic"
CONVERSION_TOL = 0.02        #: the two corroborating pairs must agree this far

#: The case the conversion does NOT cover, named so it cannot be forgotten.
CONVERSION_REFUSED = (
    "Eastman Commercial Film, the one non-colour-sensitive film of the three "
    "rated on both scales: KRL 40 against American Standard daylight 25 is a "
    "ratio of 1.60, not 4.00. Nothing in this corpus decides whether the "
    "emulsion was slowed between 1938 and 1956 or whether the house scale "
    "treated blue-sensitive stock differently, so the conversion is adopted "
    "for panchromatic sheet film only.")

SOURCE = (
    "Eastman Kodak Company, «Eastman Professional Films», Rochester "
    "N.Y., 1938 -- the professional sheet-film trade booklet, 36 pages. Its "
    "whole quantitative content is the speed table on printed page 8 and the "
    "filter-factor matrix on printed page 7; it contains no characteristic "
    "curve, no spectral curve and no modulation transfer. ⚠ THE SPEEDS "
    "ARE NOT ASA AND THE BOOKLET SAYS SO: page 8 states they \"were obtained "
    "according to the standard system of speed evaluation employed in the "
    "Kodak Research Laboratories\" and are \"for light of sunlight quality and "
    "are not valid when the materials are exposed to light of different "
    "quality, such as incandescent tungsten\" -- which is why the conversion "
    "below is derived against a DAYLIGHT index and never against a tungsten "
    "one.")


# ---------------------------------------------------------------------------
#  Re-derivation
# ---------------------------------------------------------------------------

def page_text(root=".", page=12):
    """The OCR text layer of one page, or None when the file is not staged."""
    import pymupdf
    path = os.path.join(root, "PDF", "PROFILES", SHEET)
    if not os.path.isfile(path):
        return None
    doc = pymupdf.open(path)
    t = doc[page - 1].get_text()
    doc.close()
    return t


#: The one transcribed value the page's text layer does not contain, and why.
#: \u26a0 NAMED RATHER THAN TOLERATED. "Eastman Process Film. L2" is what the
#: OCR produced: the digit 1 became the letter L, so the token "12" is simply
#: not on the page as far as a digit scan is concerned. Listing it by value
#: keeps the guard sharp -- a SECOND value going missing still fails, which is
#: what would happen if a transcription were mistyped.
OCR_LOST: frozenset = frozenset({12})


def check_text_layer(text) -> list[str]:
    """Every transcribed speed must appear as a token on the speed page.

    ⚠ THIS IS A TYPO GUARD AND NOT A READER, and the distinction is the whole
    reason it is written this way round. The scan's OCR turns "12" into "L2"
    and "40" into ",40" on this very table, so it cannot be trusted to PRODUCE
    the numbers -- but a digit sequence it did recover is still evidence that
    the sequence is on the page. Requiring every transcribed value to appear
    catches a mistyped transcription; it does not claim the OCR read the table.
    """
    if not text:
        return []
    tokens = set(re.findall(r"\d+", text))
    return [f"{n} ({nm})" for nm, _c, n in SPEEDS_1938
            if str(n) not in tokens and n not in OCR_LOST]


def bridge_ratios(ei_1956) -> list[tuple[str, float]]:
    """(1956 name, KRL / American-Standard-daylight) for the three bridges."""
    out = []
    for _n38, n56, krl, day in BRIDGE:
        have = ei_1956.get(n56)
        if have is None or not have[0]:
            continue
        out.append((n56, float(krl) / float(have[0])))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    text = page_text(args.root, 12)
    if text is None:
        print("[SKIP] kodak_1938.py -- kodak_prof_films-1938.pdf not staged")
        return 0

    # \u26a0 AND THE NAMED EXCEPTION IS ITSELF CHECKED, in the other direction:
    # a value listed in OCR_LOST that the text layer DOES contain means the
    # scan or the page number changed under the module, and the exception is
    # now hiding a real test rather than a known artefact.
    _tok = set(re.findall(r"\d+", text))
    _found = sorted(v for v in OCR_LOST if str(v) in _tok)
    if _found:
        print("[FAIL] kodak_1938.py -- OCR_LOST names %s as absent from the "
              "text layer and it is present; the exception is stale"
              % _found)
        return 1

    missing = check_text_layer(text)
    if missing:
        print("[FAIL] kodak_1938.py -- transcribed speeds that do not appear "
              "on the page at all: " + ", ".join(missing))
        return 1

    try:
        from kodak_1956 import EI_1956
    except ImportError:
        print("[FAIL] kodak_1938.py -- kodak_1956 is the other half of the "
              "conversion and is not importable")
        return 1

    ratios = bridge_ratios(EI_1956)
    pan = [(n, r) for n, r in ratios if "COMMERCIAL" not in n]
    if len(pan) != 2:
        print("[FAIL] kodak_1938.py -- the conversion rests on TWO "
              "panchromatic films rated on both scales and %d were found"
              % len(pan))
        return 1
    worst = max(abs(r - KRL_TO_AMERICAN_STANDARD) for _n, r in pan)
    if worst > CONVERSION_TOL:
        print("[FAIL] kodak_1938.py -- the two panchromatic bridges no longer "
              "agree on %.2f: %s"
              % (KRL_TO_AMERICAN_STANDARD,
                 ", ".join("%s %.3f" % (n, r) for n, r in pan)))
        return 1

    other = [(n, r) for n, r in ratios if "COMMERCIAL" in n]
    print("[OK] kodak_1938.py -- «Eastman Professional Films» 1938, "
          "15 film speeds on the KODAK RESEARCH LABORATORIES sunlight scale "
          "(the booklet names it in words) and 6 braced filter-factor rows, "
          "transcribed and checked against the page's own text layer. The "
          "scale converts: %d panchromatic films are rated on BOTH this scale "
          "and the 1956 American Standard and both give %.2f to within %.3f "
          "(%s), while %s -- recorded as a refusal, not averaged in"
          % (len(pan), KRL_TO_AMERICAN_STANDARD, worst,
             ", ".join("%s %.2f" % (n.split(",")[0].replace("KODAK ", ""), r)
                       for n, r in pan),
             ", ".join("%s gives %.2f" % (n.replace("KODAK ", ""), r)
                       for n, r in other) or "no third bridge was found"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
