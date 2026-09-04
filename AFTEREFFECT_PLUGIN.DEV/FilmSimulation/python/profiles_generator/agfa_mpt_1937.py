#!/usr/bin/env python3
"""Agfa Motion Picture Topics (1937-1940): the characteristic-curve panel and
the Goetz & Gould graininess measurements.

WHAT THIS SOURCE IS
-------------------
`AGFA/agfamotionpictur00ckin.pdf` -- **Agfa Motion Picture Topics**, C. King
Charney / Agfa Ansco, 314 pages, digitised by the Internet Archive
(`archive.org/details/agfamotionpictur00ckin`) and recoded by LuraDocument.
Volumes I-IV, **1937 to 1940**.

⚠ IT IS A TRADE MAGAZINE, NOT A DATA SHEET, and that governs what can be taken
from it. It carries technical articles by named Agfa Ansco and university
authors alongside photographs and product notices. There is no per-stock
sensitometric page anywhere in it. What it does carry is one characteristic
panel, one measured granularity series, and a set of printed facts about
emulsions the database already models.

⚠ AND IT IS THE FIRST SOURCE IN THIS CORPUS FROM THE ERA THE OWNER ASKED FOR.
`NotFound.md` records the corpus gap as **1929-1956**; the PS&E holding starts
in July 1957 and the Zeitschrift volumes stop in 1928. This document sits
inside the hole.

WHAT IS TRACED
--------------
**Page 13** (printed page 11), captioned *"Characteristic curves showing
comparison between direct duplicating film, positive 35 mm. film and Convira
paper."* Three D vs log E curves on one grid, D 0-3.0, log E 0-4.53.

  * **AGFA DIRECT DUPLICATING FILM** -- the DESCENDING curve. The article
    introduces it as "a completely novel type of film which, when exposed in a
    camera and normally developed, renders a positive instead of a negative",
    i.e. a direct-reversal material, and the curve falls with exposure as such
    a material must. Its identity is fixed by the printed label sitting beside
    it, verified against a 350 dpi render.
  * **CINE POSITIVE** -- 35 mm positive film, the steep rising curve.
    ⚠ IT LEAVES THE TOP OF THE FRAME at about log E 2.15, so its shoulder and
    Dmax are NOT in this figure and nothing here may be read as either.
  * **CONVIRA PAPER** -- the later rising curve, shouldering near D 1.63.

⚠ THE TWO PAIRS BEHAVE DIFFERENTLY AND BOTH WERE CHECKED ON THE PAGE, NOT
ASSUMED. Cine Positive and Direct Duplicating do **not** cross: at 350 dpi the
rising curve passes above the descending one's flat left end with clear white
between them. Convira and Direct Duplicating **do** merge, near D 1.6, for
about forty columns, which is why that pair is traced as a two-track pass with
`dashtrace.trace_predictive`'s merge coast and the other is not.

WHAT IS NOT TRACED, AND WHY
---------------------------
⚠ **Nothing here is adopted onto an existing profile.** None of the three
materials is in the database: it holds no Agfa positive stock, no Agfa direct
duplicating film and no Convira paper. Adding them is a decision about scope,
not a tracing problem, so this module READS the panel and stops there.

⚠ **The exposure axis is bare "Log E" with no units, no illuminant and no
development statement.** It therefore fixes SHAPE and RELATIVE placement only.
Two of the three curves cannot even be placed against each other in absolute
speed, because the article's own point is that they are different classes of
material read on one arbitrary axis.

THE OTHER HARVEST: GOETZ & GOULD, AND IT IS PRINTED NUMBERS
------------------------------------------------------------
**Pages 139-144**, *"The Graininess of Photographic Emulsions", Part IV*, by
**Dr. Alexander Goetz and W. O. Gould, California Institute of Technology**,
whose instrument was built "with the aid of the Agfa Ansco Research Fund".
Volume III No. 2, March-April 1939.

The article prints its results as numbers in the running text; `Fig. 3` merely
plots them. So this needs no tracing at all -- see `SUPERPAN_G_VS_D` and
`CLASS_G` below.

⚠ **G IS NOT rms GRANULARITY AND THIS MODULE REFUSES TO TREAT IT AS SUCH.**
Goetz's graininess constant G is defined in Parts I-III of the series, which
are **not in this volume**, and the only statement here about its scale is a
footnote: "The values of G are multiplied by the factor 1000 in order to avoid
the use of decimals." That is the same *presentation* convention this database
uses for `rms_granularity`, and it is precisely why the two must not be
conflated -- a shared factor of 1000 says nothing about what is being measured.
Until Parts I-III are in the corpus, G is a relative quantity on Goetz's own
scale.

WHAT THE SUPERPAN SERIES SAYS ANYWAY, AND WHY IT MATTERS
---------------------------------------------------------
Five densities on one named emulsion, **Agfa Superpan**:

    D    0.10   0.25   0.41   0.67   1.09
    G      58     75     93     92     57

⚠ **AN INTERIOR MAXIMUM, AND A FALL AT BOTH ENDS.** Normalised at its peak the
series is 0.62 / 0.81 / 1.00 / 0.99 / 0.61 -- graininess at D 1.09 is back
where it was at D 0.10. Schema v8 added `sigma_shape_peak` for exactly this
shape and 13 stocks carry a measured one.

⚠ **AND IT CONTRADICTS THE JONES 1958 CLASS SHAPE ADOPTED ON 2026-09-03.** That
shape, from four Kodak negatives, rises out of the toe and then FLATTENS:
0.507 / 1.000 / 1.016 at D 0.07 / 1.0 / 1.40. This one FALLS, hard. The two
are recorded side by side and neither is averaged into the other (method rule
4): they are different manufacturers, two decades apart, and -- decisively --
**different measurands**, since Jones tabulates microdensitometer sigma and
Goetz tabulates his own graininess constant. A conflict between a sigma and a
G is not yet a conflict about film.

⚠ AND A THIRD READING AGREES WITH GOETZ. `takano_1969_granularity` and
`ooue_1959_granularity` are already in this corpus, and the turnover Goetz
reports is the same phenomenon `RESULT_2026-09-01*` recorded from Lu & Torquato
as the mechanism behind a measured sigma(D) turnover. So the disagreement is
between Jones's four films and everyone else, not between this source and the
project.

THE CLASS LADDER, AND THE FOG FLOOR
------------------------------------
Six emulsion types at approximately matched density, from the same article --
the only class-by-class granularity comparison in the corpus:

    lithographic reproduction        D 0.46   G  39
    positive film                    D 0.17   G  57
    sound recording film             D 0.50   G  63
    process emulsion                 D 0.45   G  59
    panchromatic, medium speed       D 0.41   G  93
    panchromatic MP, very high speed D 0.47   G 105

⚠ AND A MEASURED FLOOR FOR SOMETHING THE PROJECT HAD WRITTEN OFF. `NotFound.md`
calls `fog_grain` a renderer parameter with "no photographic counterpart" that
"no source will ever publish". Page 143 publishes a bound on it: the graininess
contributed by celluloid base, gelatine and fog together is estimated "to be
between 15 and 30 so that the effect of base, gelatine, and fog can easily
produce 30 per cent to 50 per cent of the graininess of a fine grain emulsion
at low densities." That is a ratio, not a value for our parameter, and it is
recorded as a ratio.

THE EGGERT ARTICLE, pp 44-53 -- AND THIS ONE IS ADOPTED
--------------------------------------------------------
**"The New Agfacolor Process", by Prof. Dr. J. Eggert**, the process's own
inventor. The first pass through this document filed it in `ERA_FACTS` as a
provenance citation and moved on. That was a mistake worth naming: it is a
technical article, and it prints numbers.

⚠ **AND THE PROFILE IT DESCRIBES STATES IN ITS OWN PROVENANCE THAT NO SUCH
DOCUMENT EXISTS.** `AGFACOLOR_NEU_1936` carries tier 3 and
`fitted_from='analogy'`, with a PROVENANCE LIMIT reading "Neither carries a
photometric figure ... Every numeric value in this profile is therefore still
an analogy." True of the two citations it had; not true of the corpus.

  * **CONSTRUCTION, p48, adopted.** "three emulsion layers, one on top of each
    other, each .005 mm. thick, and separated by plain gelatine layers .002 mm.
    thick" -- 5 um emulsions, 2 um interlayers, **19 um total coated**. Page 53
    cross-checks it independently: the tripack is "only about" as thick as a
    normal single-layer film.
  * **COATING ORDER, p48, adopted.** Stated, not inferred: the yellow-dyed
    gelatine sits "between the top (blue-sensitive) emulsion and second
    (green-yellow sensitive) emulsion layer". Top to bottom blue / green / red.
  * **SPEED, p53, RECORDED AND NOT ADOPTED.** "In sunshine 16 mm. motion
    pictures may be made using a lens opening of F:4.5 to F:5.6." Sunny-16
    turns that into **ISO 2.5-3.9 at 16 fps or 4.0-6.1 at 24 fps**, and the
    article does not say which frame rate. ⚠ The stored EI 8 sits ABOVE the
    whole of both ranges -- the analogy is 0.4 to 1.7 stops fast, and the two
    readings agree on that even while they disagree with each other. The
    conflict is asserted; the stored value is not quietly moved towards it,
    because a speed inferred from exposure advice is not a measured speed.
  * **GRAIN, p53, RECORDED AND NOT ADOPTED.** "there is no silver in the
    finished film, and the dye image is practically grainless and does not show
    any clumping." A launch article's superlative, but the process fact inside
    it is real: the silver IS removed, so this stock's grain is dye-cloud grain.

THE 1937 SPEED CONVERSION TABLE, p38
-------------------------------------
Weston / H&D / Scheiner / DIN / relative sensitivity, thirteen rows, and the
only period speed-conversion table in the corpus. Stored because several stocks
here take a speed from a pre-1957 DIN, a Weston or a Scheiner and convert it
with a MODERN formula.

⚠ It validates itself against the note printed beneath it, and doing so found
**exactly one departure in the whole table**: H&D 24 -> 50 where the doubling
ladder wants 48. Every other pair on every column is exact. That single cell is
named in `SPEED_TABLE_ROUNDINGS` rather than hidden by a loose tolerance, so a
second departure still fails the check.

THE PRINTED ERA FACTS
---------------------
See `ERA_FACTS`. Each is a quotation with its page, and none of THOSE is
adopted by this module.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dashtrace as dt  # noqa: E402

SHEET = os.path.join("AGFA", "agfamotionpictur00ckin.pdf")

SOURCE = ("Agfa Motion Picture Topics, C. King Charney / Agfa Ansco, volumes "
          "I-IV, 1937-1940 -- PDF/PROFILES/AGFA/agfamotionpictur00ckin.pdf, "
          "digitised by the Internet Archive "
          "(archive.org/details/agfamotionpictur00ckin)")

#: The characteristic panel: PDF page (1-based) and the render resolution the
#: pinned pixel geometry below belongs to. Both are part of the calibration.
CURVE_PAGE = 13
CURVE_DPI = 350

#: ⚠ THE GRID IS FOUND, NOT ASSUMED, AND THEN CHECKED AGAINST THESE. The rules
#: are located by projection on every run (see `find_grid`); these values are
#: what a re-run must reproduce, so a change of renderer or of the scan cannot
#: move the calibration in silence.
#: Horizontal rules carry D = 3.0 / 2.0 / 1.0; the fourth, at the frame foot,
#: is NOT used -- it sits 5 px below where the three labelled ones extrapolate
#: D = 0, and fitting it would tilt the axis by 0.02 D to honour a frame line.
GRID_Y_PINNED = (1971.5, 2217.5, 2463.0)
GRID_Y_D = (3.0, 2.0, 1.0)
#: Vertical rules: the frame's left edge and then log E = 1, 2, 3, 4, then the
#: frame's right edge. Only the four labelled ones calibrate.
GRID_X_PINNED = (478.0, 717.0, 957.0, 1198.0, 1440.0, 1566.0)
GRID_X_LOGE = (1.0, 2.0, 3.0, 4.0)

#: Plot interior, in pixels of the pinned render.
BOX = (470, 1960, 1575, 2720)   # x0, y0, x1, y1

#: A component of the rule-free image no larger than this in either dimension
#: is a printed letter, not a curve.
#:
#: ⚠ THE MARGIN HERE IS THREE PIXELS AND THAT IS STATED RATHER THAN HIDDEN. On
#: this panel the letter components all fall at or under 75 px, and the
#: SMALLEST genuine curve fragment is 78 px wide -- the start of the Convira
#: curve lifting off the baseline at x 877..954, verified by eye at 4x. Ten of
#: the eleven curve fragments are 95 px or wider and would tolerate any cut in
#: a broad band; that one does not. A first draft of this module claimed the
#: split was clean by a factor of 1.3 and the run-time check caught it, which
#: is why the check now reports the true margin instead of a comfortable one.
#: If a future render moves that fragment below the cut, the Convira trace
#: loses its foot and the shape assertions below are what will say so.
TEXT_MAX_PX = 75

#: The margin the check above must still find between the largest letter and
#: the smallest surviving curve fragment. Positive, and known to be small.
TEXT_CUT_MARGIN_MIN = 2

#: Seeds for the three tracks: (x, density) in the panel's own units. Each is
#: on a stretch where the curve is alone in its column.
SEEDS = {
    "cine_positive":      (486, 0.052),
    "direct_duplicating": (1000, 2.522),
    "convira_paper":      (1000, 0.243),
}

#: ⚠ CONVIRA AND DIRECT DUPLICATING MERGE, CINE POSITIVE AND DIRECT DUPLICATING
#: DO NOT. Verified on a 350 dpi render of both regions. The merging pair is
#: traced together with the merge coast; the other two tracks are independent.
MERGE_PAIR = ("convira_paper", "direct_duplicating")
MERGE_PX = 9.0

#: Goetz & Gould Part IV, printed in the running text on page 142 (printed
#: p20). (density, G) for Agfa Superpan.
SUPERPAN_G_VS_D = ((0.10, 58), (0.25, 75), (0.41, 93), (0.67, 92), (1.09, 57))

#: Same article, page 139 (printed p17): six emulsion classes at approximately
#: matched density. (label, density, G).
CLASS_G = (
    ("lithographic reproduction", 0.46, 39),
    ("positive film", 0.17, 57),
    ("sound recording film", 0.50, 63),
    ("process emulsion", 0.45, 59),
    ("panchromatic, medium sensitivity", 0.41, 93),
    ("panchromatic motion picture, very high sensitivity", 0.47, 105),
)

#: Page 143: the base + gelatine + fog contribution, as the article states it.
FOG_FLOOR_G = (15, 30)
FOG_FLOOR_FRACTION = (0.30, 0.50)

# ---------------------------------------------------------------------------
# THE EGGERT HARVEST, 2026-09-04 -- and this one IS adopted.
# ---------------------------------------------------------------------------
# ⚠ WHY IT TOOK A SECOND PASS TO SEE IT. The first pass filed page 44 in
# `ERA_FACTS` as "provenance for AGFACOLOR_NEU_1936" -- a citation, nothing
# more. It is not a note: it is a technical article by **Prof. Dr. J. Eggert**,
# the process's own inventor, running pp 44-53, and it prints construction
# figures, an exposure recommendation and a grain statement.
#
# ⚠ AND THE PROFILE IT DESCRIBES SAYS IN ITS OWN PROVENANCE THAT NO SUCH
# DOCUMENT EXISTS. `AGFACOLOR_NEU_1936` is tier 3, `fitted_from='analogy'`, and
# its PROVENANCE LIMIT reads: "Neither carries a photometric figure: no speed,
# no gamma, no spectral sensitisation, no dmin or dmax. Every numeric value in
# this profile is therefore still an analogy." That was true of the two
# citations it had. It has not been true of the CORPUS since
# `agfamotionpictur00ckin.pdf` arrived on 2026-09-03.

#: Emulsion and interlayer thickness, page 48, in the inventor's own words:
#: "it took a great many experimental coatings before it was possible to coat
#: on a single film three emulsion layers, one on top of each other, each .005
#: mm. thick, and separated by plain gelatine layers .002 mm. thick."
AGFACOLOR_LAYER_UM      = 5.0
AGFACOLOR_INTERLAYER_UM = 2.0

#: Total coated thickness implied by those two figures: 3 x 5 + 2 x 2.
#:
#: ⚠ AND THE ARTICLE CROSS-CHECKS IT ITSELF on page 53 -- "the overall
#: thickness of the three emulsion layers being only about that of a normal
#: one-layer film". 19 micrometres sits inside the range a single-layer
#: emulsion of the period was coated at, so the two statements agree and
#: neither was derived from the other.
AGFACOLOR_COATED_UM = 3.0 * AGFACOLOR_LAYER_UM + 2.0 * AGFACOLOR_INTERLAYER_UM

#: Coating order, page 48, stated and not inferred: "A yellow-dyed gelatine
#: layer between the top (blue-sensitive) emulsion and second (green-yellow
#: sensitive) emulsion layer insures that no blue light reaches the two lower
#: emulsion layers." Top to bottom, by sensitisation.
AGFACOLOR_ORDER = ("blue", "green", "red")

#: The printed exposure recommendation, page 53: "In sunshine 16 mm. motion
#: pictures may be made using a lens opening of F:4.5 to F:5.6."
#:
#: ⚠ THIS IS THE ONLY SENTENCE IN THE ARTICLE THAT CAN BE TURNED INTO A SPEED,
#: AND ONLY BECAUSE IT NAMES THE LIGHT. The still-picture sentence beside it
#: ("1/50th to 1/100th of a second at F:3.5") does NOT say sunshine, so it is
#: recorded and not used: an exposure with no stated illuminant fixes nothing.
AGFACOLOR_SUN_F = (4.5, 5.6)

#: Shutter time for 16 mm cine, both plausible readings. 1936 amateur 16 mm ran
#: at 16 frames per second; 24 was the sound standard. With a 170-180 degree
#: shutter those are about 1/32 s and 1/50 s.
#:
#: ⚠ BOTH ARE CARRIED AND NEITHER IS CHOSEN, because the article does not say
#: which and the two differ by two thirds of a stop. They agree on the
#: DIRECTION of the finding below, so the ambiguity does not have to be
#: resolved for the finding to stand.
AGFACOLOR_SHUTTER_S = {"16 fps": 1.0 / 32.0, "24 fps": 1.0 / 50.0}

#: Sunny-16: correct exposure in bright sun for speed S is f/16 at 1/S second,
#: so S = (N^2 / t) / 256.
SUNNY16_K = 256.0

#: What the profile stores today, and it is an ANALOGY, not a measurement.
AGFACOLOR_STORED_EI = 8

#: Page 53, on grain: "there is no silver in the finished film, and the dye
#: image is practically grainless and does not show any clumping."
#:
#: ⚠ RECORDED, NOT ADOPTED. It is a claim about the finished DYE image while
#: `rms_granularity` is a claim about the rendered one, and the two coincide
#: only if the dye clouds carry no structure at all, which nothing here
#: establishes. It is also a manufacturer's launch article, and "practically
#: grainless" is what every one of them says. What it DOES establish, because
#: it is a process fact rather than a superlative, is that the silver is
#: removed -- so whatever grain this stock has is dye-cloud grain, not silver
#: grain, and `dye_cloud_um` rather than `clump_um` is where it lives.
AGFACOLOR_GRAIN_CLAIM = ("no silver in the finished film; dye image "
                         "practically grainless and shows no clumping")


def agfacolor_ei_range():
    """{shutter: (S at f/4.5, S at f/5.6)} from the printed sunshine advice."""
    return {label: tuple((n * n / t) / SUNNY16_K for n in AGFACOLOR_SUN_F)
            for label, t in AGFACOLOR_SHUTTER_S.items()}


# ---------------------------------------------------------------------------
# THE 1937 SPEED CONVERSION TABLE, page 38 (printed p16).
# ---------------------------------------------------------------------------
# "TABLE FOR APPROXIMATE COMPARISON OF FILM SPEED VALUES", thirteen rows.
#
# ⚠ WHY A CONVERSION TABLE BELONGS IN A FILM DATABASE. Several stocks here take
# their speed from a period rating -- a pre-1957 DIN, a Weston, a Scheiner --
# converted with a MODERN formula. This is the conversion as the era itself
# printed it, so a period rating can be read with a period table instead of an
# anachronism.
#
# ⚠ AND IT CARRIES ITS OWN CHECK, which is what makes it trustworthy after a
# text extraction that interleaved the columns. The printed note says:
# "Scheiner and DIN ratings increase by three units when the sensitivity of the
# film doubles. H & D ratings and Weston speeds are multiplied by 2 when the
# sensitivity of the film is doubled." `check_speed_table()` asserts exactly
# that on the transcribed columns, and passing it is what proves the
# de-interleaving is right rather than merely plausible.
SPEED_TABLE_1937 = {
    "weston":   (1.0, 1.2, 1.5, 2.0, 2.4, 3.0, 4.0, 4.8, 6.0, 8.0, 9.6, 12.0,
                 16.0),
    "hd":       (4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 50, 64),
    "scheiner": (150, 189, 238, 300, 378, 476, 600, 756, 952, 1200, 1512,
                 1904, 2400),
    "din":      (15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
    #: "Relative Sensitivity", printed as tenths: 6/10 through 18/10.
    "relative": (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
                 1.8),
}

#: How close the doubling test must hold, with ONE named exception below.
SPEED_TABLE_TOL = 0.03

#: ⚠ THE TABLE DEPARTS FROM ITS OWN PRINTED RULE IN EXACTLY ONE CELL, AND THAT
#: IS A FINDING ABOUT THE 1937 TABLE RATHER THAN A TRANSCRIPTION ERROR. Every
#: other pair on every column doubles exactly: Weston 1->2, 1.2->2.4, 1.5->3,
#: 2->4, 2.4->4.8, 3->6, 4->8, 4.8->9.6, 6->12, 8->16; Scheiner all ten pairs
#: exact; H&D 4->8, 5->10, 6->12, 8->16, 10->20, 12->24, 16->32, 20->40, 32->64.
#: The single departure is H&D row 8: **24 -> 50 where the ladder wants 48**, a
#: 4.2 per cent rounding to a round number.
#:
#: It is named here rather than absorbed by a looser tolerance, so a SECOND
#: departure -- which would mean the de-interleaving is wrong -- still fails.
SPEED_TABLE_ROUNDINGS = {
    ("hd", 8): "printed 50 where the doubling ladder gives 48 (+4.2%)",
}


def check_speed_table():
    """[(claim, ok, detail)] -- the printed note, asserted on the columns."""
    t = SPEED_TABLE_1937
    n = len(t["din"])
    out = [("all five columns carry %d rows" % n,
            all(len(v) == n for v in t.values()),
            ", ".join("%s=%d" % (k, len(v)) for k, v in t.items()))]

    din_ok = all(t["din"][i + 3] - t["din"][i] == 3 for i in range(n - 3))
    out.append(("DIN rises exactly 3 units every three rows", din_ok,
                "%d -> %d over %d rows" % (t["din"][0], t["din"][-1], n)))

    worst, at, extra = 0.0, "", []
    for key in ("weston", "hd", "scheiner"):
        for i in range(n - 3):
            r = t[key][i + 3] / float(t[key][i])
            e = abs(r - 2.0) / 2.0
            if (key, i) in SPEED_TABLE_ROUNDINGS:
                extra.append("%s row %d %s" % (key, i,
                                               SPEED_TABLE_ROUNDINGS[(key, i)]))
                continue
            if e > worst:
                worst, at = e, "%s row %d (%.3fx)" % (key, i, r)
    out.append(("Weston, H&D and Scheiner DOUBLE over those same three rows, "
                "everywhere but the %d named rounding(s)"
                % len(SPEED_TABLE_ROUNDINGS), worst <= SPEED_TABLE_TOL,
                "worst unnamed departure %.2f%% at %s; named: %s"
                % (100.0 * worst, at, "; ".join(extra) or "none")))
    out.append(("every named rounding is still present -- the table has not "
                "been silently corrected",
                all(abs(t[k][i + 3] / float(t[k][i]) - 2.0) / 2.0
                    > SPEED_TABLE_TOL for k, i in SPEED_TABLE_ROUNDINGS),
                ", ".join("%s[%d]=%s" % (k, i, t[k][i + 3])
                          for k, i in SPEED_TABLE_ROUNDINGS)))

    steps = {round(t["relative"][i + 1] - t["relative"][i], 6)
             for i in range(n - 1)}
    out.append(("Relative Sensitivity is a log ladder, 0.3 per doubling",
                steps == {0.1}, "steps %s" % sorted(steps)))
    return out


#: Printed statements worth citing, none of them adopted here.
ERA_FACTS = (
    (33, "gamma", "0.68 is named as the gamma a negative type shows "
     "'when developed at standard machine speed' -- an era anchor for "
     "release-negative development, not a property of any one stock"),
    (38, "speed", "TABLE FOR APPROXIMATE COMPARISON OF FILM SPEED VALUES: "
     "Weston / H&D / Scheiner / DIN / relative sensitivity"),
    (106, "Ultra-Speed vs Superpan", "P. H. Arnold, Agfa-Ansco Corporation, "
     "Binghamton: 'Ultra-Speed panchromatic film, compared to Superpan "
     "negative film, is much faster; slightly flatter in gradation; similar "
     "in color-sensitivity'"),
    (110, "speed ladder", "at 1/1000 second, ordinary Superpan calls for "
     "f/3.5, Supreme permits f/5.6 and Ultra Speed Pan f/7 -- 1.3 and 2.0 "
     "stops on Superpan"),
    (118, "speed listings", "Agfa Superpan Press is rated 100 for daylight "
     "under 'Rolls and Packs' and 125 under the 'Press' grouping; the "
     "difference is the development standard, not the emulsion"),
    (44, "Agfacolor", "Prof. Dr. J. Eggert, 'The New Agfacolor Process' -- "
     "the inventor's own account, provenance for AGFACOLOR_NEU_1936"),
    (13, "direct duplicating", "'Within the last year, Agfa Ansco has "
     "introduced a completely novel type of film which, when exposed in a "
     "camera and normally developed, renders a positive instead of a "
     "negative' -- and 'its sensitivity is only within the range of that of "
     "average contact printing papers'"),
)


# ---------------------------------------------------------------------------
# raster access
# ---------------------------------------------------------------------------
def page_gray(root=".", page=CURVE_PAGE, dpi=CURVE_DPI):
    """The page as float grayscale in [0, 1]. None when the PDF is absent."""
    import pymupdf
    path = os.path.join(root, "PDF", "PROFILES", SHEET)
    if not os.path.isfile(path):
        return None
    doc = pymupdf.open(path)
    pm = doc[page - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width)
    doc.close()
    return a.astype(np.float64) / 255.0


def find_grid(gray, box=BOX, thr=150 / 255.0):
    """Locate the drawn rules by projection. Returns (h_centres, v_centres).

    A rule is a row (column) whose ink covers most of the box. Adjacent rows
    are one rule and are averaged, which is what turns a 5 px drawn line into
    a sub-pixel centre.
    """
    x0, y0, x1, y1 = box
    ink = gray < thr
    m = np.zeros_like(ink)
    m[y0:y1, x0:x1] = ink[y0:y1, x0:x1]

    def centres(idx):
        out, cur = [], [idx[0]]
        for v in idx[1:]:
            if v - cur[-1] <= 2:
                cur.append(v)
            else:
                out.append(float(np.mean(cur)))
                cur = [v]
        out.append(float(np.mean(cur)))
        return out

    hs = [y for y in range(y0, y1) if m[y, x0:x1].sum() > 0.45 * (x1 - x0)]
    vs = [x for x in range(x0, x1) if m[y0:y1, x].sum() > 0.60 * (y1 - y0)]
    return centres(hs), centres(vs)


def calibrate(h_centres, v_centres):
    """(D_of_y, logE_of_x, y_of_D, x_of_logE) from the labelled rules."""
    hy = np.asarray(h_centres[:3], dtype=float)
    md, bd = np.polyfit(np.asarray(GRID_Y_D, dtype=float), hy, 1)
    vx = np.asarray(v_centres[1:5], dtype=float)
    ml, bl = np.polyfit(np.asarray(GRID_X_LOGE, dtype=float), vx, 1)
    return (lambda y: (y - bd) / md,
            lambda x: (x - bl) / ml,
            lambda d: md * d + bd,
            lambda l: ml * l + bl)


def trace_mask(gray, box=BOX, thr=150 / 255.0, text_max=TEXT_MAX_PX):
    """Ink with the grid rules and every printed letter removed.

    ⚠ THE RULES ARE REMOVED RATHER THAN TOLERATED, and the curves are cut into
    fragments as a result. That is deliberate: a 5 px rule crossing a curve
    contributes a spurious run at a FIXED density in every column it spans, and
    a tracker that has to reject those on thickness alone will eventually reject
    a thin part of a real curve instead. The gaps left behind are 5 px and
    `trace_predictive` bridges up to 26.
    """
    x0, y0, x1, y1 = box
    ink = gray < thr
    m = np.zeros_like(ink)
    m[y0:y1, x0:x1] = ink[y0:y1, x0:x1]
    clean = m.copy()
    for y in range(y0, y1):
        if m[y, x0:x1].sum() > 0.45 * (x1 - x0):
            clean[y, x0:x1] = False
    for x in range(x0, x1):
        if m[y0:y1, x].sum() > 0.60 * (y1 - y0):
            clean[y0:y1, x] = False
    lab, info = dt._components(clean)
    text = [n for n, (w, h, _c) in info.items()
            if w <= text_max and h <= text_max]
    # ⚠ RULE REMNANTS ARE NOT CURVE FRAGMENTS. Stripping a 5 px rule leaves
    # 1-2 px stubs where a curve met it at a shallow angle, and they survive
    # the letter cut on height while being nothing at all. They are excluded
    # from the width statistic so it reports the real gap between the smallest
    # curve fragment and the largest letter, which is what that check is for.
    # ⚠ RULE REMNANTS ARE NOT CURVE FRAGMENTS. Stripping the rules leaves five
    # degenerate slivers -- 1, 1, 1, 2 and 4 px wide, and two that are 1 and
    # 3 px TALL -- where a curve or a rule end met another rule. They carry at
    # most a one-pixel run and are excluded from the width statistic so that it
    # measures the real letter/curve margin rather than being pinned at 1 px by
    # a leftover.
    big = [(w, h) for n, (w, h, _c) in info.items()
           if n not in text and w > 5 and h > 5]
    clean &= ~np.isin(lab, text)
    return clean, len(text), (min((w for w, _h in big), default=0) if big else 0)


def trace_curves(gray, box=BOX):
    """Trace the three curves. Returns {name: [(logE, D), ...]} left to right."""
    hs, vs = find_grid(gray, box)
    D_of_y, logE_of_x, y_of_D, x_of_logE = calibrate(hs, vs)
    mask, _ntext, _minbig = trace_mask(gray, box)
    x0, y0, x1, y1 = box
    xlo, xhi = int(vs[0]) + 6, int(vs[-1]) - 4

    out = {}
    # Cine Positive: alone in its columns for its whole drawn extent. Traced
    # rightward from the left frame; it leaves the top of the frame on its own.
    sx, sd = SEEDS["cine_positive"]
    got = dt.trace_predictive(mask, gray, (xlo, xhi), y0, y1, sx,
                              {"cine_positive": y_of_D(sd)}, direction=+1,
                              max_bridge=26)
    out["cine_positive"] = got["cine_positive"]

    # Convira and Direct Duplicating share columns and merge near D 1.6, so
    # they are traced TOGETHER with the merge coast -- see the module docstring.
    seeds = {k: y_of_D(SEEDS[k][1]) for k in MERGE_PAIR}
    right = dt.trace_predictive(mask, gray, (1000, xhi), y0, y1, 1000, seeds,
                                direction=+1, max_bridge=26, merge_px=MERGE_PX)
    # ⚠ THE LEFTWARD PASS IS DELIBERATELY SHORT-SIGHTED, AND THAT IS THE ONLY
    # THING THAT STOPS IT INVENTING A CURVE. Direct Duplicating STOPS at its
    # drawn left end near log E 1.89 -- the flat stub in the figure -- and past
    # that there is no ink for it. With the rightward pass's settings the track
    # simply keeps missing, `tol_grow` widens the window by 0.7 px per missed
    # column, and after eighty columns it is 56 px wide: enough to capture the
    # Cine Positive curve, which at that point sits 0.23 D away. The trace then
    # came back running the full width of the frame, smooth and entirely wrong.
    # A tight window and a short bridge let the track DIE where the ink does.
    left = dt.trace_predictive(mask, gray, (xlo, 1000), y0, y1, 1000, seeds,
                               direction=-1, max_bridge=8, tol0=3.0,
                               tol_grow=0.25, merge_px=MERGE_PX)
    for k in MERGE_PAIR:
        pts = dict(left[k])
        pts.update(right[k])
        out[k] = pts

    return {k: [(logE_of_x(x), D_of_y(y)) for x, y in sorted(v.items())]
            for k, v in out.items()}


def resample(curve, lo, hi, step):
    """Uniform resample onto lo:hi:step; None outside the traced extent."""
    xs = np.asarray([p[0] for p in curve], dtype=float)
    ys = np.asarray([p[1] for p in curve], dtype=float)
    n = int(round((hi - lo) / step)) + 1
    out = []
    for i in range(n):
        g = lo + i * step
        if g < xs.min() - 1e-9 or g > xs.max() + 1e-9:
            out.append((g, None))
        else:
            out.append((g, float(np.interp(g, xs, ys))))
    return out


def class_ratios():
    """Goetz's class ladder as ratios to the medium-speed panchromatic entry."""
    ref = dict((lab, g) for lab, _d, g in CLASS_G)[
        "panchromatic, medium sensitivity"]
    return {lab: g / ref for lab, _d, g in CLASS_G}


def superpan_shape():
    """The Superpan series normalised at its own maximum: (D, G/Gmax)."""
    gmax = max(g for _d, g in SUPERPAN_G_VS_D)
    return tuple((d, g / gmax) for d, g in SUPERPAN_G_VS_D)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ns = ap.parse_args(argv)

    print("[i] %s" % SOURCE)
    bad = 0

    # ---- the printed numbers need no document -----------------------------
    print("\n  Goetz & Gould 1939 Part IV -- Agfa Superpan, G against density")
    sh = superpan_shape()
    for (d, g), (_d, r) in zip(SUPERPAN_G_VS_D, sh):
        print("      D %.2f   G %3d   %.2f of peak" % (d, g, r))
    peak_at = max(SUPERPAN_G_VS_D, key=lambda t: t[1])[0]
    ok = 0.3 < peak_at < 0.9
    bad += (not ok)
    print("    [%s] the maximum is INTERIOR, at D %.2f"
          % ("OK  " if ok else "FAIL", peak_at))
    ends = sh[0][1], sh[-1][1]
    ok = max(ends) < 0.70
    bad += (not ok)
    print("    [%s] and it falls at BOTH ends -- %.2f at D %.2f, %.2f at D "
          "%.2f" % ("OK  " if ok else "FAIL", ends[0], sh[0][0],
                    ends[1], sh[-1][0]))

    # ⚠ the conflict with Jones, asserted so it cannot be quietly reconciled
    try:
        import pse_jones_1958 as pj
        jones_high = pj.ADOPTED_DMAX / pj.ADOPTED_MID
        goetz_high = sh[-1][1] / max(r for _d, r in sh)
        ok = jones_high > 0.95 and goetz_high < 0.70
        bad += (not ok)
        print("    [%s] the two measured shapes still DISAGREE above D 1.0: "
              "Jones %.3f of its D 1.0 value at D 1.40, Goetz %.2f at D 1.09"
              % ("OK  " if ok else "FAIL", jones_high, goetz_high))
    except Exception as exc:            # pragma: no cover
        print("    [note] Jones comparison unavailable: %s" % exc)

    print("\n  The class ladder, as ratios to medium-speed panchromatic")
    for lab, r in sorted(class_ratios().items(), key=lambda t: t[1]):
        print("      %-52s %.2f" % (lab, r))
    print("    [note] base + gelatine + fog contribute G %d-%d, which the "
          "article puts at %d-%d %% of a fine-grain emulsion at low density"
          % (FOG_FLOOR_G[0], FOG_FLOOR_G[1],
             int(FOG_FLOOR_FRACTION[0] * 100), int(FOG_FLOOR_FRACTION[1] * 100)))

    # ---- the traced panel -------------------------------------------------
    gray = page_gray(ns.root)
    if gray is None:
        print("\n  [SKIP] source not present: %s" % SHEET)
        return 1 if (bad and ns.do_assert) else 0

    hs, vs = find_grid(gray)
    print("\n  Grid: %d horizontal rules, %d vertical rules"
          % (len(hs), len(vs)))
    ok = len(hs) >= 3 and len(vs) >= 6
    bad += (not ok)
    print("    [%s] the panel's rules are all found" % ("OK  " if ok else "FAIL"))
    dh = max(abs(a - b) for a, b in zip(hs[:3], GRID_Y_PINNED))
    dv = max(abs(a - b) for a, b in zip(vs[:6], GRID_X_PINNED))
    ok = dh < 1.5 and dv < 1.5
    bad += (not ok)
    print("    [%s] and they sit where they were pinned -- worst %.2f px "
          "horizontal, %.2f px vertical" % ("OK  " if ok else "FAIL", dh, dv))

    D_of_y, logE_of_x, y_of_D, x_of_logE = calibrate(hs, vs)
    # ⚠ the frame's foot is NOT a calibration point; this reports how far the
    # three labelled rules put D = 0 from where the frame line is drawn.
    if len(hs) >= 4:
        print("    [note] the three labelled rules extrapolate D = 0 to "
              "y %.1f; the drawn frame foot is at y %.1f, %.3f D away"
              % (y_of_D(0.0), hs[3], abs(D_of_y(hs[3]))))

    _mask, ntext, minbig = trace_mask(gray)
    margin = minbig - TEXT_MAX_PX
    ok = ntext > 100 and margin >= TEXT_CUT_MARGIN_MIN
    bad += (not ok)
    print("    [%s] %d letters removed; the smallest surviving curve fragment "
          "is %d px against a %d px cut -- a margin of %d px, which is thin "
          "and is meant to be visible"
          % ("OK  " if ok else "FAIL", ntext, minbig, TEXT_MAX_PX, margin))

    curves = trace_curves(gray)
    print("\n  Traced curves")
    for name in ("cine_positive", "convira_paper", "direct_duplicating"):
        c = curves[name]
        print("      %-20s n=%-4d  log E %.2f..%-5.2f  D %.3f..%.3f"
              % (name, len(c), c[0][0], c[-1][0],
                 min(p[1] for p in c), max(p[1] for p in c)))

    cp = curves["cine_positive"]
    dd = curves["direct_duplicating"]
    cv = curves["convira_paper"]

    # ⚠ SHAPE ASSERTIONS, because a swapped pair is two smooth curves and no
    # residual can see it. Each of these is a property the PAGE shows.
    # It exits through the D 3.0 rule, which `trace_mask` has removed, so the
    # last sample sits just below it rather than on it.
    ok = cp[-1][1] > cp[0][1] and cp[-1][1] > 2.70
    bad += (not ok)
    print("    [%s] Cine Positive RISES and leaves the frame near D 3.0 "
          "(%.2f at log E %.2f)"
          % ("OK  " if ok else "FAIL", cp[-1][1], cp[-1][0]))
    ok = dd[0][1] > dd[-1][1] and dd[0][1] > 2.2 and dd[-1][1] < 0.4
    bad += (not ok)
    print("    [%s] Direct Duplicating FALLS, %.2f down to %.2f -- a reversal "
          "material, as the article describes it"
          % ("OK  " if ok else "FAIL", dd[0][1], dd[-1][1]))
    ok = cv[-1][1] > cv[0][1] and 1.4 < max(p[1] for p in cv) < 1.9
    bad += (not ok)
    print("    [%s] Convira Paper RISES to a shoulder at D %.2f"
          % ("OK  " if ok else "FAIL", max(p[1] for p in cv)))
    # ⚠ and the pair that merges must come back SEPARATED again
    tail = [d for l, d in cv if l > 3.3]
    ok = bool(tail) and min(tail) > 1.3
    bad += (not ok)
    print("    [%s] Convira does not follow Direct Duplicating down after "
          "their merge -- it holds D %.2f past log E 3.3"
          % ("OK  " if ok else "FAIL", min(tail) if tail else float("nan")))

    # ---- the Eggert harvest, 2026-09-04 ----------------------------------
    print("\n  EGGERT 1937, pp 44-53 -- the inventor's own account of "
          "Agfacolor Neu")
    print("      three emulsions %.0f um each, gelatine interlayers %.0f um, "
          "total coated %.0f um (p48)"
          % (AGFACOLOR_LAYER_UM, AGFACOLOR_INTERLAYER_UM,
             AGFACOLOR_COATED_UM))
    print("      coating order top to bottom: %s (p48)"
          % " / ".join(AGFACOLOR_ORDER))

    # ⚠ THE SPEED FINDING, AND IT IS ARITHMETIC ON A PRINTED SENTENCE RATHER
    # THAN A PRINTED SPEED. The article never states an ASA, a DIN or a Weston;
    # it states an f-number in sunshine, which sunny-16 turns into one.
    ei = agfacolor_ei_range()
    for label in sorted(ei):
        lo, hi = ei[label]
        print("      sunshine f/%.1f-f/%.1f at %s -> ISO %.1f - %.1f"
              % (AGFACOLOR_SUN_F[0], AGFACOLOR_SUN_F[1], label, lo, hi))

    hi_all = max(v[1] for v in ei.values())
    lo_all = min(v[0] for v in ei.values())
    ok = hi_all < AGFACOLOR_STORED_EI
    bad += (not ok)
    print("    [%s] and the stored EI %d sits ABOVE the whole derived range "
          "%.1f - %.1f -- the analogy is %.1f to %.1f stops fast, and the two "
          "shutter readings agree on that even though they disagree by two "
          "thirds of a stop"
          % ("OK  " if ok else "FAIL", AGFACOLOR_STORED_EI, lo_all, hi_all,
             np.log2(AGFACOLOR_STORED_EI / hi_all),
             np.log2(AGFACOLOR_STORED_EI / lo_all)))

    # ⚠ WHAT IS ADOPTED FROM THIS IS THE GEOMETRY, NOT THE SPEED. The speed is
    # an inference from exposure advice with an unstated frame rate; the layer
    # thicknesses are printed numbers. Method rule 4: the conflict is recorded,
    # the stored value is not quietly averaged towards it.
    for name, want in (("AGFACOLOR_COATED_UM", 19.0),):
        got = globals()[name]
        ok = abs(got - want) < 1e-9
        bad += (not ok)
        print("    [%s] %s = %.1f um, which is 3 x %.0f + 2 x %.0f as the page "
              "prints them" % ("OK  " if ok else "FAIL", name, got,
                               AGFACOLOR_LAYER_UM, AGFACOLOR_INTERLAYER_UM))

    # ---- the 1937 speed table --------------------------------------------
    print("\n  SPEED TABLE, p38 -- checked against the note printed beneath it")
    for claim, ok, detail in check_speed_table():
        bad += (not ok)
        print("    [%s] %s   (%s)" % ("OK  " if ok else "FAIL", claim, detail))

    if ns.emit:
        print("\n  --- 0.1-decade resample ---")
        for name in ("cine_positive", "convira_paper", "direct_duplicating"):
            r = resample(curves[name], 0.0, 4.5, 0.1)
            print("  %s = (%s)" % (name, ", ".join(
                "None" if v is None else "%.3f" % v for _g, v in r)))

    print()
    if bad:
        print("  [FAIL] %d problem(s)" % bad)
        return 1 if ns.do_assert else 0
    print("  [OK  ] Agfa Motion Picture Topics: three 1937 characteristic "
          "curves traced, Goetz & Gould's graininess series read as printed, "
          "and its conflict with Jones 1958 asserted rather than resolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
