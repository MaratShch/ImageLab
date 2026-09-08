"""f50 and the adjacency overshoot from a VECTOR log-log MTF plot.

WHY
---
`MTFSpec.f50_*` is the sharpness parameter the renderer actually uses, and for
most stocks it is an estimate. Kodak prints the curve it comes from -- and on some
sheets prints it as VECTOR art, where the answer can be read rather than guessed.

The EASTMAN PLUS-X 5231 sheet (H-1-5231, February 1999) is the case this was
written for: page 3 carries the modulation-transfer curve (plot F002_0141AC) as a
single bezier path, and the whole page contains ZERO embedded images. E0's
re-verification of that profile found the sheet prints no numeric MTF value, so
the stored f50 could not be confirmed from text -- but it can be measured from the
path.

WHAT IT MEASURES, and the one thing that is NOT f50
---------------------------------------------------
  * f50: the frequency at which response falls back through 50 %. Taken at the
    LAST crossing, because the curve rises ABOVE 100 % at low frequency (see
    below) and a naive first-crossing search on a non-monotone curve can return
    the wrong branch.
  * the ADJACENCY OVERSHOOT: the peak response above unity, which is the
    development edge effect. `MTFSpec.adjacency` is documented as exactly that
    fraction, so the plot measures it directly.

⚠ THE OVERSHOOT'S FREQUENCY IS NOT `adjacency_um`. On 5231 the peak sits near
4-5 cycles/mm, a spatial scale of order 100-200 um, while the stored
`adjacency_um` is 16.0 (which corresponds to ~60 cycles/mm). The same
inconsistency appears on FUJI_F125_8530, whose Honjo-1989 overshoot peaks near
9 cycles/mm against a stored 13.0 um. Either the field means something narrower
than the overshoot period or the values are wrong; that depends on how the
renderer defines it, so this script REPORTS the peak frequency and changes
nothing. Recorded rather than resolved.

AXES: both are logarithmic, and both are least-squares fitted over every printed
decade and mantissa label with a residual test -- the same discipline as
dye_density.py and granularity_vector.py, for the same reason (a two-point span
cannot detect a misplaced label). 5231 gives 11 frequency ticks and 12 response
ticks, fitting to 0.66 and 0.82 pt.

C2b, 2026-08-23: THE COLOUR BATCH, AND FOUR DEFECTS IT FOUND IN THIS SCRIPT
--------------------------------------------------------------------------
Eight more Kodak colour sheets and the one non-Kodak MTF sheet in the corpus were
read. Every defect below produced PLAUSIBLE NUMBERS, which is the only reason
they are worth recording:

  1. ONE PATH, THREE CURVES. The 1990s technical sheets emit all three records as
     a single path object -- the same hazard the granularity panels have. Read as
     one curve, H-1-5218 gave "f50 69.7" off a trace that walks along blue, jumps
     to green and finishes on red. Fixed by splitting on
     `granularity_vector.subpaths` (imported, not copied).
  2. THE LOG GRID PASSES FOR A CURVE. On 5245 and 5246 the grid is one connected
     polyline; the letter matcher handed it back as the green record (f50 236.8,
     response to 190 %). Fixed by three shape tests -- single-valued in frequency,
     a real vertical extent, and near-monotone descent (total variation over span
     <= 2.0, measured 1.00-1.35 on every real curve and 12.9 on the grid).
  3. GREEDY LABEL MATCHING DOUBLE-CLAIMS, and ranking by height at a common
     abscissa gets the order wrong when one record stops earlier than another
     (5248 red stops at 115 cycles/mm, green runs to 191). Fixed by solving the
     3x3 assignment over all six permutations.
  4. A FRAGMENT HAS AN f50 AND IT IS MEANINGLESS. 5293's red survives as a
     30-125 cycles/mm arc starting at 53 % response and reports 32.0. Now refused
     with a stated reason instead of measured.

And the extractor gained the gate the other two plot readers already had:
`--overlay` draws every traced point back onto the page. All four defects above
were found by looking at it.

Run:
    python mtf_vector.py --root ../..
    python mtf_vector.py --root ../.. --assert
    python mtf_vector.py --root ../.. --overlay /tmp/ov     # LOOK AT THIS
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# The subpath splitter, imported rather than copied: one definition (C2b).
import granularity_vector as gv

#: tag -> (pdf under PDF/PROFILES -- bare name = KODAK, else "MAKER/name.pdf",
#: page, profile, frame hint x0,x1,y0,y1)
SHEETS = {
    # 2026-08-25, queue E0b-orig: the only colour REVERSAL sheet in the corpus
    # whose MTF panel is vector art. Frame read off the page rather than guessed;
    # the tuple order is (x0, x1, y0, y1), which is not the order PyMuPDF's Rect
    # prints in and is worth stating because getting it wrong finds no curves.
    "5285": ("Ektachrome_100d.pdf", 3, "KODAK_EKTACHROME_100D_5285",
             (362.6, 565.1, 348.3, 501.3)),
    "5231": ("5231-PLUS-X.pdf", 3, "EASTMAN_PLUS_X_5231", (87, 289, 293, 446)),
    # 2026-08-26, owner-supplied. H-1-5222 (July 2015) -- the FULL Kodak sheet
    # for EASTMAN DOUBLE-X, where the corpus previously held only the short
    # "technical information" extract. Black-and-white, so ONE curve, like its
    # sister sheet 5231. The stored f50 triple was the flat estimate 56/56/56.
    "5222": ("EASTMAN DOUBLE-X Negative Film 5222.pdf", 3,
             "EASTMAN_DOUBLE_X_5222", (72.7, 275.3, 189.9, 342.9)),
    # 2026-08-20, queue C2b's first addition. H-1-5201 p3 prints the same plot
    # type for a COLOUR negative, so it carries THREE curves -- one per record --
    # where 5231 (a black-and-white stock) carries one. That is the whole point:
    # MTFSpec has three f50 fields and until now every colour stock's three were
    # estimates in a fixed ratio.
    # ⚠ THE RED RECORD IS DRAWN TWICE, yellow under magenta, exactly as on the
    # same sheet's granularity panel. Handled by ink, see pick_curves().
    "5201": ("Kodak VISION2 50D 5201.pdf", 3, "KODAK_VISION2_50D_5201",
             (224, 350.5, 205, 286)),
    # 2026-08-20, added on the owner's re-upload of the four named PDFs. Three of
    # the four turned out to be documents the corpus already held -- V200T.pdf is
    # BYTE-IDENTICAL to 5274.pdf (md5 cf07db7d...) -- but its MTF panel had never
    # been traced, and 5274's stored f50 triple was an estimate.
    # ⚠ THIS SHEET LETTERS ITS CURVES INSTEAD OF COLOURING THEM. All three are
    # black; R / G / B are printed inside the frame's right edge. See
    # letter_assign().
    "5274": ("5274.pdf", 3, "KODAK_VISION_200T_5274",
             (362.7, 565.0, 155.3, 308.3)),
    # ---- C2b, 2026-08-23: the colour batch, and it is what C24 was waiting for.
    # Five more Kodak COLOUR negatives with all three records intact, spanning
    # 1989 (EXR) to 2005 (VISION2) -- plus two that yield green and blue only and
    # are registered anyway, because a REFUSED record is evidence too and this is
    # where the refusal is re-derived. Every one of these was inspected on the
    # --overlay render before being pinned; four extractor defects were found that
    # way, and all four produced plausible numbers (see the module docstring).
    "5217": ("5217-Vision2-200T.pdf", 3, "KODAK_VISION2_200T_5217",
             (222.7, 349.3, 205.0, 284.6)),
    "5218": ("5218-Vision2-500T-H-1-5218t.pdf", 3, "KODAK_VISION2_500T_5218",
             (368.5, 562.5, 385.6, 538.6)),
    "5245": ("5245.pdf", 3, "EASTMAN_EXR_50D_5245", (350.6, 553.1, 381.1, 534.1)),
    "5248": ("5248.pdf", 3, "EASTMAN_EXR_100T_5248", (362.6, 565.1, 284.5, 437.5)),
    # ⚠ THE LARGEST ADJACENCY OVERSHOOT IN THE CORPUS, AND IT IS PRINTED, NOT A
    # TRACE ERROR: this sheet's green and blue records rise to 142 % and 155 % at
    # 15 cycles/mm. Verified on the overlay -- the traced points sit on the
    # printed curves. It is also why this stock's ROLLOFF EXPONENT is not
    # adoptable: the carrier 1/(1+(f/f50)^q) is 1.0 at zero frequency by
    # construction and cannot represent a curve that starts at 1.42, so its fit
    # comes back at rms 0.25 against 0.02-0.10 everywhere else.
    "5279": ("5279.pdf", 2, "KODAK_VISION_500T_5279", (359.4, 541.5, 149.9, 287.6)),
    # ⚠ RED REFUSED ON BOTH OF THESE, by the fragment test rather than by hand:
    # the sheets emit the red record in pieces and the surviving piece starts at
    # 53 % (5293) and 77 % (5205) response. A piece has an f50 and it is
    # meaningless -- 5293's reads 32.0 cycles/mm off a 30-125 c/mm arc.
    "5293": ("5293.pdf", 3, "EASTMAN_EXR_200T_5293", (357.8, 560.4, 64.5, 215.1)),
    "5205": ("5205t.pdf", 3, "KODAK_VISION2_250D_5205",
             (359.7, 561.8, 396.2, 547.1)),
    # ---- the only NON-KODAK MTF sheet in the corpus, and the reason C2b went
    # ---- looking: C24 asks whether the per-record ratio can be derived from the
    # ---- layer stack, and seven measurements of one maker cannot answer that.
    # ⚠ IT DOES NOT ANSWER IT EITHER, because Agfa prints ONE curve: the panel is
    # captioned "Sharpness ... MTF (Modulation Transfer Function)" with
    # "Densitometry: visual filter (V-lambda)", i.e. a visual-weighted pooled
    # response, not three records. What it does establish is that the power-law
    # carrier and the overshoot are not Kodak habits: q = 2.63 at rms 0.039, and
    # a +11.7 % overshoot at 3.4 cycles/mm.
    # ⚠ ALSO NOT A NEW STOCK. This sheet's page 6 left column is Vista 200, which
    # the database holds; pages 5, 7 and 8 carry Vista 100/400/800, FUTURA II and
    # CTprecisa, which it does not.
    "vista200": ("AGFA/AGFACOLOR Vista 100, 200, 400, 800.pdf", 6,
                 "AGFA_VISTA_200", (53.4, 179.8, 316.1, 401.6)),
    # ---- queue T2, 2026-09-02e: the still-film sheets the row asked for ------
    # ⚠ ONLY EKTAR IS HERE, AND THAT IS THE ROW'S ANSWER, NOT AN OMISSION. T2
    # named E-4046 (EKTAR 100) and E-7022 (GOLD 200). Both 2016 sheets draw
    # their curves as VECTOR paths, but only E-4046 CARRIES an MTF panel at all
    # (E4046D, three records, 1-600 c/mm): E-7022 page 4 prints characteristic,
    # spectral-sensitivity and spectral-dye-density curves and NO modulation
    # transfer function, on any of its four pages or in any of the four copies
    # held. GOLD 200's f50 therefore stays an estimate and the reason is now on
    # record instead of being an unexplained gap.
    "ektar100": ("KODAK/e4046_ektar_100-2016.pdf", 4,
                 "KODAK_EKTAR_100", (346.0, 558.0, 352.0, 514.0)),
    # ---- 2026-09-06: the KODAK BLACK-AND-WHITE STILL sheets ------------------
    # Class A of the corpus audit: these panels were on the owner's disk all
    # along, indexed by `PDF/PROFILES/Index.md` with the M flag, and had never
    # been staged into the working corpus. Every one is VECTOR art with ZERO
    # embedded images on its page.
    #
    # ⚠ THE FRAME IS A SIGNATURE, AND FINDING IT IS WHAT MADE THE BATCH CHEAP.
    # Every Kodak modulation-transfer panel in this corpus is drawn inside a
    # rectangle of 202 x 153 pt. Selecting `re` items by that size and then
    # taking the one whose nearest caption above reads "Modulation Transfer"
    # located all ten frames without a single hand measurement -- and it also
    # caught the two pages where the caption sits BELOW the plot (F-32 p14), on
    # which a naive "caption is above" rule picks the spectral-sensitivity panel
    # and the y-tick scan then finds nothing.
    "tmax100": ("KODAK/f4016_TMax_100.pdf", 8, "KODAK_TMAX_100",
                (71.2, 273.7, 260.1, 413.1)),
    # ⚠ NOT AN INDEPENDENT CONFIRMATION OF tmax100 AND REGISTERED ANYWAY. The
    # 2007 edition of F-4016 carries the SAME VECTOR OBJECT as the 2016 one --
    # 143 points, bounding boxes equal, max deviation 0.0010 pt after an origin
    # shift. It is pinned so that the identity keeps being re-derived: the whole
    # reason P3200 was refused below is that this artwork travels between
    # publications, and a test that only ever looked at one edition would not
    # see it move.
    "tmax100_07": ("KODAK/f4016-TMAX-2007b.pdf", 14, "KODAK_TMAX_100",
                   (74.6, 277.1, 100.5, 253.5)),
    # ✅ THE REAL INDEPENDENT CONFIRMATION: F-32 (September 2001) is a different
    # drawing (112 points against 143) and reads 120.2 against 123.0, i.e. 2.3 %
    # apart on a stock whose stored value was the estimate 95.0.
    "tmax100_01": ("KODAK/f32-TMAX-200109.pdf", 14, "KODAK_TMAX_100",
                   (74.7, 277.2, 323.4, 476.4)),
    # ⚠ THREE INDEPENDENT DRAWINGS THAT DO NOT AGREE, AND THEY ARE ALL PINNED.
    # 95.8 (F-32, 2001, TMY) / 66.7 (F-4016, 2007) / 80.6 (F-4043, 2016, TMY-2).
    # Rule 4: recorded, not averaged. The adopted value is F-4043's, because
    # that is the publication `_PROVENANCE_SOURCES` already names for this
    # profile -- an edition choice, not a quality judgement between the three.
    "tmax400": ("KODAK/f4043_TMax_400.pdf", 7, "KODAK_TMAX_400",
                (360.9, 563.4, 248.6, 401.6)),
    "tmax400_07": ("KODAK/f4016-TMAX-2007b.pdf", 14, "KODAK_TMAX_400",
                   (348.8, 550.7, 115.7, 269.2)),
    "tmax400_01": ("KODAK/f32-TMAX-200109.pdf", 14, "KODAK_TMAX_400",
                   (350.7, 553.2, 331.3, 484.3)),
    # ✅ THE BEST-EVIDENCED MTF IN THE DATABASE: three independent drawings
    # across seventeen years reading 52.7 / 52.5 / 53.1, a spread of 1.1 %.
    # ⚠ THE PANEL BELONGS TO 400TX AND NOT TO 320TXP, which is why only one of
    # the two films this sheet covers gets a measurement. F-4017 (2016) p7 heads
    # its own right-hand column "KODAK PROFESSIONAL TRI-X 400 Film / 400TX,
    # 35 mm" and prints exactly one modulation-transfer frame on the page.
    "trix400": ("KODAK/f4017_TriX.pdf", 7, "KODAK_TRI_X_400TX",
                (84.8, 287.3, 76.0, 228.8)),
    "trix400_07": ("KODAK/f4017-400TX-2007.pdf", 10, "KODAK_TRI_X_400TX",
                   (348.5, 551.0, 99.8, 252.6)),
    "trix400_99": ("KODAK/f9-Tri-X_Pan-199906.pdf", 9, "KODAK_TRI_X_400TX",
                   (86.7, 289.0, 71.4, 224.4)),
    # ⚠ ADOPTS NOTHING -- E-4046 (2010) IS THE SAME ARTWORK AS THE 2016 EDITION
    # ALREADY REGISTERED ABOVE, on all three records (188 / 290 / 288 points,
    # max deviation 0.0010 pt). Pinned for the same reason as tmax100_07: the
    # identity is the evidence. It does mean the EKTAR measurement rests on ONE
    # drawing and not two, which the provenance note now says.
    "ektar100_10": ("KODAK/e4046-EKTAR-2010.pdf", 5, "KODAK_EKTAR_100",
                    (347.6, 550.1, 327.2, 480.2)),
    # ⚠ P3200's OWN CURVE, AND THE REASON THIS ENTRY EXISTS IS A MISTAKE OF MINE.
    # On 2026-09-06 the F-4001 (2018) panel was found to be T-MAX 100's artwork
    # and P3200 was written off as having no published MTF -- on the strength of
    # ONE edition. The owner asked the obvious question, and F-4001 (2019) p7
    # carries a DIFFERENT drawing: 40 bezier control points against 2018's 44,
    # bbox 106.79 x 104.85 against 112.97 x 113.22, peaking at 5.4 cycles/mm
    # instead of 18.8. Kodak put the wrong figure in the 2018 edition and
    # corrected it the following year; the 2018 page's resolving-power text was
    # P3200's own (40 / 125 lines/mm) all along, so only the plot was wrong.
    # ⚠ THE RULE THIS COST: never write a refusal from a single file when the
    # corpus holds another edition of the same publication.
    # Checked against every registered sheet -- this curve matches nothing else.
    "p3200_19": ("KODAK/f4001-P3200TMZ-2019.pdf", 7, "KODAK_TMAX_P3200",
                 (361.2, 563.7, 77.0, 230.0)),
}

#: Measured 2026-08-18/20. --assert fails if a sheet stops reproducing these.
#: A colour sheet pins one entry per record; a mono sheet pins the single curve
#: under the key "-".
EXPECTED = {
    # ✅ ADOPTED 2026-08-25 (queue E0b-orig). The FIRST measured MTF for a colour
    # REVERSAL stock in this database -- every other traced sheet is a negative.
    # ⚠ AND IT IS THE LARGEST MTF CORRECTION THE PROJECT HAS MADE: the stored
    # f50_g was the estimate 82.0 and the sheet measures 42.1, i.e. the estimate
    # was 1.95x TOO SHARP. Red and blue had no stored value of their own at all.
    # The layer order comes out R < G < B (27.2 / 42.1 / 60.9), which is the
    # order MTFSpec's docstring predicts -- blue on top, red at the bottom under
    # the most scattered light -- and is the second independent confirmation of
    # it, after 5201.
    # The power law beats the legacy Gaussian on all three records (3.5x, 1.9x
    # and 1.4x better in rms), so q is stored and mtf_measured is set.
    "5285": {
        "R": dict(f50=27.2, peak=1.040, peak_at=2.4),
        "G": dict(f50=42.1, peak=1.030, peak_at=7.8),
        "B": dict(f50=60.9, peak=1.022, peak_at=8.8),
    },
    "5231": {"-": dict(f50=41.3, peak=1.034, peak_at=4.6)},
    # ✅ ADOPTED 2026-08-26. EASTMAN DOUBLE-X, off the full H-1-5222 sheet the
    # owner supplied; the corpus had only a short extract before.
    # ⚠ THE STORED TRIPLE WAS THE FLAT ESTIMATE 56.0 / 56.0 / 56.0 -- 1.33x TOO
    # SHARP. The measurement also lands within 2 % of its SISTER STOCK: PLUS-X
    # 5231, the other Kodak black-and-white cine negative in this corpus, reads
    # 41.3 off its own sheet against DOUBLE-X's 42.2. Two independent sheets,
    # two independent traces, and the two films are two speeds of one design
    # family -- which is the sort of agreement that was NOT available while both
    # numbers were estimates (the old pair read 56.0 and 60.0).
    # ⚠ ITS OVERSHOOT IS +25 %, the third largest traced, and it is PRINTED --
    # verified on the --overlay render, where the traced points sit on the
    # printed curve over its whole 2.4-98.5 cycles/mm extent. q is still adopted
    # here, unlike 5279's +42 %: the power law fits at rms 0.076, inside the
    # 0.0095-0.132 band of every accepted curve, where 5279 returned 0.25-0.34.
    "5222": {"-": dict(f50=42.2, peak=1.250, peak_at=4.1)},
    "5201": {
        "R": dict(f50=32.1, peak=1.108, peak_at=2.5),
        "G": dict(f50=49.7, peak=1.157, peak_at=10.7),
        "B": dict(f50=55.5, peak=1.142, peak_at=12.7),
    },
    # ✅ ADOPTED 2026-08-20c (queue C13, owner-approved). 5274 stored the ESTIMATE
    # 56.0 / 64.0 / 72.0 with adjacency 0.09 and now carries these measurements.
    #   GREEN AND BLUE CONFIRMED the estimate to 7 % (68.8 vs 64, 74.0 vs 72).
    # THE RED RECORD DID NOT: 35.4 against 56.0, i.e. the estimate was 1.58x too
    # sharp. That is the estimating RULE, not this profile -- it puts f50_r/f50_b
    # at about 0.78 on 72 of the 92 colour stocks still carrying an estimate,
    # while both stocks measured per-record land at 0.478 (5274) and 0.578 (5201).
    # Whether to re-derive the rule for the remaining 92 is a separate decision
    # and is NOT taken here; this entry pins the measurement that raised it.
    "5274": {
        "R": dict(f50=35.4, peak=1.027, peak_at=2.4),
        "G": dict(f50=68.8, peak=1.162, peak_at=11.0),
        "B": dict(f50=74.0, peak=1.234, peak_at=16.1),
    },
    # ---- C2b, 2026-08-23 ----------------------------------------------------
    "5217": {
        "R": dict(f50=33.9, peak=1.058, peak_at=2.5),
        "G": dict(f50=58.1, peak=1.110, peak_at=13.7),
        "B": dict(f50=67.4, peak=1.154, peak_at=13.7),
    },
    "5218": {
        "R": dict(f50=37.6, peak=1.008, peak_at=2.4),
        "G": dict(f50=54.6, peak=1.014, peak_at=7.7),
        "B": dict(f50=69.7, peak=1.064, peak_at=18.4),
    },
    "5245": {
        "R": dict(f50=37.2, peak=0.984, peak_at=3.7),
        "G": dict(f50=83.8, peak=1.048, peak_at=12.9),
        "B": dict(f50=100.5, peak=1.089, peak_at=15.7),
    },
    "5248": {
        "R": dict(f50=37.4, peak=0.984, peak_at=3.7),
        "G": dict(f50=75.1, peak=1.069, peak_at=12.9),
        "B": dict(f50=111.2, peak=1.153, peak_at=20.4),
    },
    "5279": {
        "R": dict(f50=41.1, peak=1.088, peak_at=2.5),
        "G": dict(f50=73.1, peak=1.420, peak_at=15.1),
        "B": dict(f50=76.1, peak=1.554, peak_at=15.4),
    },
    # Green and blue only: red is refused as a fragment on both sheets.
    "5293": {
        "G": dict(f50=75.2, peak=1.065, peak_at=15.9),
        "B": dict(f50=114.6, peak=1.155, peak_at=18.3),
    },
    "5205": {
        "G": dict(f50=55.9, peak=1.032, peak_at=14.6),
        "B": dict(f50=59.3, peak=1.099, peak_at=14.6),
    },
    # One visual-weighted curve, so it pins the mono key.
    "vista200": {"-": dict(f50=50.0, peak=1.117, peak_at=3.4)},
    # ---- queue T2, 2026-09-02e ----------------------------------------------
    # ✅ THE FIRST MEASURED MTF FOR A STILL COLOUR NEGATIVE in this database --
    # every other traced sheet is a cine stock. ⚠ AND THE ESTIMATE IT REPLACES
    # WAS 1.5x TOO SHARP on every record: the stored triple was 74.3 / 80.0 /
    # 87.6 against a measured 35.5 / 52.7 / 54.8. That is the same direction and
    # very nearly the same size as the error found on 5285 (1.95x), 5222 (1.33x)
    # and 5231 (1.45x), so it is the estimating RULE showing through again, not
    # this profile. Layer order comes out R < G < B, which is the order
    # MTFSpec's docstring predicts and the fourth stock to confirm it.
    # Verified on the --overlay render: the traced points sit on the printed
    # curves over the whole 2.5-80.7 c/mm extent.
    "ektar100": {
        "R": dict(f50=35.5, peak=1.124, peak_at=9.0),
        "G": dict(f50=52.7, peak=1.183, peak_at=9.7),
        "B": dict(f50=54.8, peak=1.070, peak_at=9.0),
    },
    # ---- 2026-09-06: the KODAK black-and-white still sheets -----------------
    # ✅ ADOPTED: T-MAX 100 at 123.0 (was the estimate 95.0 -- the estimate was
    # 1.29x TOO SOFT, and it is the FIRST TRACED SHEET WHERE THE ESTIMATE ERRED
    # IN THAT DIRECTION. Every previous one was too sharp: 5285 by 1.95x, 5222
    # by 1.33x, 5231 by 1.45x, EKTAR by 1.5x. A T-grain stock outrunning the
    # estimating rule where conventional emulsions undershoot it is a fact about
    # the rule, and it is recorded rather than smoothed.)
    "tmax100": {"-": dict(f50=123.0, peak=1.145, peak_at=18.8)},
    "tmax100_07": {"-": dict(f50=123.0, peak=1.144, peak_at=18.8)},
    "tmax100_01": {"-": dict(f50=120.3, peak=1.111, peak_at=18.3)},
    # ✅ ADOPTED: T-MAX 400 at 80.6, from F-4043 (2016). ⚠ The curve reaches 50 %
    # AT ITS PRINTED ENDPOINT (81 c/mm, last plotted response 51 %), so this is
    # a crossing that coincides with the end of the drawing rather than one
    # inside it. Verified on the render: Kodak stops the curve at 50 %, which is
    # its convention on this sheet, so the value is read and not extrapolated.
    # ⚠ NO PIN FOR "tmax400" ITSELF -- IT IS IN `REFUSED`. F-4043's curve stops
    # 1.4 response-points above the crossing, which is a lower bound and not a
    # measurement; see REFUSED for why that settles the whole film.
    "tmax400_07": {"-": dict(f50=66.7, peak=1.198, peak_at=14.1)},
    "tmax400_01": {"-": dict(f50=95.9, peak=1.168, peak_at=7.5)},
    # ✅ ADOPTED: TRI-X 400TX at 52.7 (was the estimate 58.0, so 1.10x too sharp
    # -- the SMALLEST correction any traced sheet has produced, and the only one
    # under 1.3x).
    "trix400": {"-": dict(f50=52.7, peak=1.120, peak_at=8.6)},
    "trix400_07": {"-": dict(f50=52.6, peak=1.119, peak_at=8.3)},
    "trix400_99": {"-": dict(f50=53.1, peak=1.102, peak_at=8.5)},
    # ✅ ADOPTED: T-MAX P3200 at 84.3, from F-4001 (2019) -- the edition that
    # carries P3200's OWN drawing. ⚠ f50 ONLY: the rolloff fits at q = 2.03,
    # rms 0.0534 against the Gaussian's 0.0681, i.e. only 1.3x better. The
    # project's own threshold, set on the PORTRA NC/VC batch (2026-08-30), is
    # that 1.2-1.3x does NOT license switching the carrier -- so mtf_measured
    # stays False, no kernel row is needed, and the stock keeps the legacy
    # Gaussian at a measured f50.
    # ⚠ 84.3 AGAINST THE SHEET'S OWN 125 lines/mm at 1000:1 is a ratio of 1.35
    # on Tani's f50 ~ RP/2 relation, in family with T-MAX 100's 1.23. The 2018
    # sheet's misplaced figure would have implied 1.97, at the edge of the band
    # verify.py allows -- which is the physical reading of the same error.
    "p3200_19": {"-": dict(f50=84.3, peak=1.110, peak_at=5.4)},
    # Adopts nothing; pins the artwork identity with the 2016 edition.
    "ektar100_10": {
        "R": dict(f50=35.5, peak=1.124, peak_at=9.0),
        "G": dict(f50=52.7, peak=1.183, peak_at=9.7),
        "B": dict(f50=54.8, peak=1.070, peak_at=9.0),
    },
}

#: ⚠ SHEETS WHOSE MTF PANEL IS THE **WRONG FILM'S ARTWORK**, and the test that
#: proves it. Each entry is (the sheet carrying the wrong plot, the sheet it was
#: copied FROM, the profile). `--assert` re-derives the identity every build.
#:
#: ⚠ AN ENTRY HERE IS NOT A REFUSAL OF THE FILM, AND SAYING SO COST A REAL
#: MEASUREMENT. On 2026-09-06 F-4001 (2018)'s panel was found to be T-MAX 100's
#: artwork and KODAK_TMAX_P3200 was written off as having no published MTF --
#: from ONE edition. F-4001 (2019) carries P3200's own drawing (40 control
#: points against 44, f50 84.3) and is now adopted. What this table records is
#: that a SPECIFIC EDITION's plot cannot be read, not that the film has no data.
#: ⚠ THE RULE: before writing a refusal, check every edition of the publication
#: the corpus holds.
#:
#: ⚠ SECOND INSTANCE OF THIS DEFECT CLASS. Queue K6 found E-2468's characteristic
#: page carrying PORTRA 160VC's figure -- and there too the fix is to find the
#: right sheet, not to declare the parameter unobtainable.
#:
#: ⚠ THE KODAK FIGURE ID IS **NOT** EVIDENCE OF THIS, and it was nearly used as
#: such. Both sheets label the panel `F002_0542AC`, which looks conclusive until
#: the spectral panels are checked: those share `F002_0547AC` too and their
#: geometry differs by 6.0 pt. The IDs are template slots, not data identifiers.
#: Only the geometry proves the copy.
ARTWORK_REUSE = {
    # KODAK T-MAX P3200, F-4001 (2018) p7. The panel is byte-for-byte T-MAX
    # 100's: 143 points, equal bounding boxes, max deviation 0.0010 pt after an
    # origin shift, and the same annotation block (Tungsten / Small Tank / D-76
    # 68 F / Diffuse visual). An ISO 3200 push film cannot share a sharpness
    # curve with a 100-speed T-grain stock, so the panel is REFUSED and
    # KODAK_TMAX_P3200.mtf stays the estimate 50.0.
    # KODAK T-MAX P3200, F-4001 (2018) p7 -- 143 sampled points, equal bounding
    # boxes, max deviation 0.0010 pt after an origin shift, and the annotation
    # block travelled with it (Tungsten / Small Tank / D-76 68 F / Diffuse
    # visual). ⚠ THE PAGE'S RESOLVING-POWER TEXT IS P3200's OWN (40 / 125
    # lines/mm, against T-MAX 100's 63 / 200), so only the PLOT was misplaced --
    # which is why nothing else on the page looks wrong. Corrected by Kodak in
    # the 2019 edition; that edition is registered in SHEETS as "p3200_19" and
    # is what the profile now carries.
    "p3200": (("KODAK/F4001-P3200TMZ-2018.pdf", 7, (360.8, 563.3, 77.8, 230.8)),
              ("KODAK/f4016_TMax_100.pdf", 8, (71.2, 273.7, 260.1, 413.1)),
              "KODAK_TMAX_P3200"),
}
TOL_F, TOL_P = 1.0, 0.01

#: ⚠ SHEETS WHOSE PANEL IS REGISTERED PRECISELY SO THAT ITS REFUSAL KEEPS BEING
#: RE-DERIVED. A tag listed here is EXPECTED to fail the f50 read; the build
#: fails if one of them starts succeeding, because that would mean the drawing
#: changed and the profile's refusal should be revisited.
#:
#: This is the same discipline `spectral_sampling.py` applies to F3: a refusal
#: needs guarding MORE than an adoption does, because nothing downstream
#: consumes it and nothing would notice if its premise stopped holding.
REFUSED = {
    "tmax400": (
        "F-4043 (2016) stops T-MAX 400's curve at 81 cycles/mm with a last "
        "plotted response of 51.4 %, so the sheet gives f50 > 81 and NOT a "
        "value. ⚠ THAT LOWER BOUND CONTRADICTS THE OTHER TWO SHEETS -- F-4016 "
        "(2007) crosses at 66.7 and F-32 (2001) at 95.9 -- so the three Kodak "
        "publications for this film do not agree and no single number is "
        "defensible. KODAK_TMAX_400.mtf stays the estimate 72.0. Rule 4: the "
        "conflict is recorded, not averaged."),
}

TICK_RESID_PT = 1.5


def logfit(pairs, label, min_keep=6):
    """{decade value: pixel} -> (px per decade, intercept, residual, n).

    ⚠ ONE LABEL CAN BE MISPLACED, and on H-1-5274's MTF panel one is. Its
    response axis prints 1 2 3 5 7 10 20 30 50 70 100 **150**; the first eleven
    give 66.88 and 66.76 pt per decade -- agreeing to 0.2 % -- while "150" sits
    7.6 pt off the line they define. The axis is clipped at the frame top and the
    label was set at the frame edge rather than at its own value. A fit over all
    twelve is not collinear at 5.94 pt and refuses the sheet.
    Same outlier rejection as granularity_vector.fit(), and for the same reason:
    the give-away is that the SURVIVING ticks agree to a fraction of a point.
    Rejection stops while `min_keep` remain, so a sparse axis cannot be whittled
    down to a fabricated line. Verified: 5231 and 5201 drop nothing.
    """
    v = np.array([np.log10(k) for k in sorted(pairs)])
    px = np.array([pairs[k] for k in sorted(pairs)])
    keep = np.ones(len(v), bool)
    dropped = []
    while True:
        A = np.vstack([v[keep], np.ones(keep.sum())]).T
        m, c = np.linalg.lstsq(A, px[keep], rcond=None)[0]
        res = np.abs(m*v + c - px)
        worst = int(np.argmax(np.where(keep, res, -1.0)))
        if res[worst] <= TICK_RESID_PT or keep.sum() <= min_keep:
            break
        keep[worst] = False
        dropped.append((10.0**float(v[worst]), float(res[worst])))
    if dropped:
        print("    %s: DROPPED %s" % (label, ", ".join(
            "%g (%.2f pt off)" % d for d in dropped)))
    v, px = v[keep], px[keep]
    A = np.vstack([v, np.ones(len(v))]).T
    m, c = np.linalg.lstsq(A, px, rcond=None)[0]
    res = float(np.abs(m*v + c - px).max())
    if res > TICK_RESID_PT:
        raise SystemExit(f"[!] {label}: ticks not collinear, {res:.2f} pt "
                         f"over {len(v)}")
    return m, c, res, len(v)


def flatten(items, n=40):
    pts = []
    for it in items:
        if it[0] == "c":
            P = [it[1], it[2], it[3], it[4]]
            for k in range(n+1):
                t = k/n
                u = 1.0-t
                pts.append((
                    u**3*P[0].x + 3*u*u*t*P[1].x + 3*u*t*t*P[2].x + t**3*P[3].x,
                    u**3*P[0].y + 3*u*u*t*P[1].y + 3*u*t*t*P[2].y + t**3*P[3].y))
        elif it[0] == "l":
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
    return pts


IDEAL = {"R": (1.0, 0.0, 0.0), "G": (0.0, 1.0, 0.0), "B": (0.0, 0.0, 1.0)}

#: samples below this carry the adjacency overshoot, which is a separate effect
#: modelled separately; including them bends the rolloff to absorb a lift. The
#: same 8 cycles/mm cut C2 used on 5231.
ROLLOFF_FROM = 8.0


def score_carrier(f, r, f50, f_from):
    """Score the adopted power-law rolloff against the legacy Gaussian.

    C2 chose `1/(1+(f/f50)^q)` over `exp(-ln2 (f/f50)^2)` on ONE traced curve and
    said so in the result entry: "the one-curve basis is the weakest part of
    today's choice". Queue item C2b is to trace more and re-score. So every curve
    this file reads now reports the same comparison, in the same units, rather
    than leaving the re-scoring to a future ad-hoc script.

    Both forms pass through 0.5 at f50 by construction, so this compares SHAPE
    away from f50 and nothing else.
    """
    # ⚠ THE CUT IS THE OVERSHOOT PEAK, NOT A FIXED 8 cycles/mm. C2's 8 came from
    # 5231, whose overshoot peaks at 4.7 cycles/mm, so 8 was safely above it. On
    # 5201 the green record peaks at 10.7 and the blue at 12.7, so a fixed 8
    # leaves the lift inside the fitted band and the power law scores rms 0.095 --
    # a number that says nothing about the carrier and everything about fitting a
    # rolloff through an overshoot.
    m = f >= f_from
    if m.sum() < 6:
        return (f"rolloff: fewer than 6 samples above {f_from:.1f} cycles/mm "
                f"-- not scored")
    x = f[m] / f50
    y = r[m]
    gauss = np.exp(-np.log(2.0) * x**2)
    rms_g = float(np.sqrt(np.mean((gauss - y)**2)))
    best_q, best_r = None, None
    for q in np.arange(0.60, 6.001, 0.005):
        e = float(np.sqrt(np.mean((1.0/(1.0 + x**q) - y)**2)))
        if best_r is None or e < best_r:
            best_q, best_r = float(q), e
    return (f"rolloff over {int(m.sum())} samples >= {f_from:.1f} "
            f"cycles/mm: power law q = {best_q:.2f} at rms {best_r:.4f}, "
            f"Gaussian rms {rms_g:.4f} ({rms_g/best_r:.1f}x worse)")


def letter_assign(pg, cand, fx0, fx1, fy0, fy1):
    """Record identity from PRINTED R / G / B letters, as an exhaustive bijection.

    ⚠ KODAK PRINTS THE RECORD TWO DIFFERENT WAYS on the same plot type, and this
    is the second. The 2005 brochures state it in INK (see pick_curves); the 1997
    technical sheets draw all three curves in BLACK and letter them -- H-1-5274 p3
    puts R / G / B at x 494-500 inside the frame's right edge. A grey-ink sheet is
    therefore not necessarily a one-curve sheet, which is what the old
    "all grey -> take the thickest" branch assumed.

    Returns None -- and falls back to the single-curve rule -- unless the frame
    carries exactly one of each letter, the three letters are STACKED at one
    abscissa (which is what makes vertical order meaningful), and the resulting
    letter-to-curve map is a bijection.
    """
    letters = {}
    for a, b, c, d, t, *_ in pg.get_text("words"):
        if t not in ("R", "G", "B"):
            continue
        cx, cy = (a+c)/2.0, (b+d)/2.0
        if not (fx0-6 <= cx <= fx1+6 and fy0-6 <= cy <= fy1+6):
            continue
        letters.setdefault(t, []).append((cx, cy))
    if sorted(letters) != ["B", "G", "R"] or any(len(v) != 1
                                                for v in letters.values()):
        return None
    lab = {k: v[0] for k, v in letters.items()}
    # ⚠ THE THREE LETTERS ARE STACKED AT ONE x, so a nearest-point distance is
    # nearly the same for all three curves and an "is the winner clearly better
    # than the runner-up" gate refuses every time -- which it did. What the sheet
    # actually states is VERTICAL ORDER at the letters' abscissa, so each letter
    # is matched to the curve whose height AT THAT x is closest to it, and the
    # result must still be a bijection.
    xs = [c[0] for c in lab.values()]
    if max(xs) - min(xs) > 0.10 * (fx1 - fx0):
        return None                      # not a stacked legend -- refuse
    # ⚠ THE LABEL SITS BEYOND THE END OF ITS OWN CURVE ON SOME SHEETS, and that
    # is how Kodak labels a family: the letter is set just to the right of where
    # the curve stops. On H-1-5245 the three curves end at x 517 and the letters
    # are at x 520-528, so a strict "the label must be inside the curve's x range"
    # test refused all three and the sheet silently fell back to the single-curve
    # rule -- which is how it came to report ONE f50 for a colour negative. A
    # label is allowed to sit up to 8 % of the frame width past the terminus, and
    # is then compared against the curve's value AT that terminus.
    reach = 0.08 * (fx1 - fx0)

    # ⚠ SOLVED AS A MINIMUM-COST BIJECTION over the six permutations, and both
    # simpler rules were tried first and both broke on a real sheet.
    #   * NEAREST-CURVE, greedy: on H-1-5245 the label "G" sits 4.4 pt from the
    #     RED curve's end and 7.6 pt from its own, takes red, and the assignment
    #     then fails -- so the sheet silently fell back to the single-curve rule
    #     and reported ONE f50 for a colour negative.
    #   * RANK BY HEIGHT at a common abscissa: on 5248 the red record STOPS at
    #     115 cycles/mm while green runs to 191 and ends lower, so the vertical
    #     order of the three curve ENDS is B, R, G -- ranking swapped red and
    #     green and produced a red record SHARPER than green, which is
    #     physically impossible on any colour negative.
    # Three curves and three labels is six permutations, so the assignment can
    # simply be solved rather than approximated. Each label is compared against
    # the curve height at the point of that curve nearest the label's own x,
    # because Kodak sets the letter beside the curve's terminus.
    import itertools

    def height_at(pts, lx):
        px = [q[0] for q in pts]
        if not (min(px) - 2 - reach <= lx <= max(px) + 2 + reach):
            return None
        order = sorted(pts)
        xq = min(max(lx, min(px)), max(px))
        return float(np.interp(xq, [q[0] for q in order], [q[1] for q in order]))

    recs = sorted(lab)
    if len(cand) < 3:
        return None
    cost = {}
    for rec in recs:
        lx, ly = lab[rec]
        for i, (pts, *_rest) in enumerate(cand):
            h = height_at(pts, lx)
            cost[(rec, i)] = None if h is None else abs(h - ly)
    best, bestperm = None, None
    for perm in itertools.permutations(range(len(cand)), 3):
        vals = [cost[(rec, i)] for rec, i in zip(recs, perm)]
        if any(v is None for v in vals):
            continue
        tot = sum(vals)
        if best is None or tot < best:
            best, bestperm = tot, perm
    if bestperm is None:
        return None
    # A SANITY bound, not a fit: the letters are set beside their curves, so a
    # label further than a fifth of the frame height from the curve it was
    # assigned means the assignment is wrong and the sheet must be refused.
    if any(cost[(rec, i)] > 0.20 * (fy1 - fy0)
           for rec, i in zip(recs, bestperm)):
        return None
    return {rec: cand[i][0] for rec, i in zip(recs, bestperm)}


def pick_curves(pg, fx0, fx1, fy0, fy1):
    """The response curves inside the frame, keyed by record.

    ⚠ THE OLD RULE WAS "the thickest long path", and it only ever had to pick one
    curve out of one. A colour sheet draws three, at identical width, and prints
    the record in INK -- the same convention granularity_vector.colour_assign()
    reads on the brochures. It also draws the red record TWICE, once in yellow and
    once in magenta on top, so a naive by-colour grouping yields four curves and
    two of them are the same measurement.

    ⚠ AND A THIRD WAY, FOUND UNDER C2b ON 2026-08-23: the 1990s technical sheets
    emit ALL THREE response curves as ONE path object, exactly as the granularity
    panels do. Treating that as a single curve is not a small error -- on H-1-5218
    it produced one "curve" running 2.4 to 80 cycles/mm through 106 % down to
    20 %, i.e. a trace that walks along blue, jumps to green and finishes on red,
    and it still looks like a plausible MTF. So a drawing that splits into two or
    more frame-spanning subpaths is expanded into one candidate per subpath, using
    `granularity_vector.subpaths()` -- the SAME splitter, imported rather than
    copied, because it is the function that whole extraction turns on.
    A drawing that yields one wide subpath is left exactly as it was, which is
    what keeps 5231, 5201 and 5274 reproducing their pinned values.

    Returned as {record: points}. A sheet whose ink is black (5231) yields
    {"-": points} and is measured exactly as before -- verified: it reproduces
    f50 41.3 and the 3.4 % overshoot to the digit.
    """
    cand = []
    span_min = 0.20*(fx1-fx0)

    def wide(pts):
        return (len(pts) >= 8
                and max(x for x, _ in pts) - min(x for x, _ in pts) >= span_min)

    def falls(pts):
        """A response curve DESCENDS. A grid line and a tick row do not.

        ⚠ SECOND HALF OF THE SAME LESSON as single_valued(), and it is what the
        overlay showed on 5245: the letter matcher had claimed the row of tick
        marks along the 1 % gridline as the green record -- perfectly
        single-valued, perfectly flat -- while the actual green curve, drawn
        between red and blue and labelled G on the sheet, was never a candidate.
        An MTF curve crosses 50 %, so over its own frame it must cover a real
        fraction of the response axis; 15 % is far below the ~40 % that every
        traced curve in this corpus actually spans, and far above a gridline's 0.
        """
        ys = [q[1] for q in pts]
        return (max(ys) - min(ys)) >= 0.15 * (fy1 - fy0)

    def smooth(pts):
        """A response curve is nearly monotone; a comb of ticks is not.

        ⚠ THIRD AND LAST FILTER, and the one that finally separated the GRID from
        the curves on 5245 and 5246. Their grids are emitted as ONE connected
        polyline that walks the whole frame -- so it is wide, it spans the full
        response axis, and (because most of its vertices sit on the bottom rule)
        it even survives a per-bin spread test. What it cannot fake is SMOOTHNESS:
        walking a grid accumulates far more vertical travel than the height it
        covers. Measured on the eleven curves already traced in this corpus, the
        ratio of total vertical variation to vertical span is 1.00-1.35 -- a
        response curve descends, with at most a small rise over its overshoot.
        The 5245 grid comes in at 12.9. The cut at 2.0 is between the two by a
        wide margin in both directions, which is the only kind of threshold worth
        having.
        """
        ys = [q[1] for q in pts]
        span = max(ys) - min(ys)
        if span <= 0.0:
            return False
        tv = sum(abs(ys[i+1] - ys[i]) for i in range(len(ys)-1))
        return tv <= 2.0 * span

    def single_valued(pts):
        """A transfer function has ONE response per frequency. A grid does not.

        ⚠ THIS TEST IS WHY C2b's BATCH IS TRUSTWORTHY AT ALL, and it was added
        after the letter matcher confidently handed back the LOG GRID as the green
        record of 5245 (7.8-477 cycles/mm, response to 190 %, power-law rms 0.74)
        and of 5246 (1.0-608 cycles/mm). Both are absurd on inspection and both
        would have been adopted by a script that only checked "did three curves
        come back". The letters are printed near the grid lines too, so nearest-
        letter matching cannot separate them; the SHAPE can, and this is the
        cheapest true statement about the shape: bin the points by frequency and
        require the vertical spread inside a bin to be small. A grid has the full
        frame height in every bin.
        """
        xs = [q[0] for q in pts]
        lo, hi = min(xs), max(xs)
        if hi - lo <= 0.0:
            return False
        nb = 12
        spread = []
        for b in range(nb):
            a0 = lo + (hi - lo) * b / nb
            a1 = lo + (hi - lo) * (b + 1) / nb
            ys = [q[1] for q in pts if a0 <= q[0] <= a1]
            if len(ys) >= 2:
                spread.append(max(ys) - min(ys))
        if not spread:
            return False
        spread.sort()
        med = spread[len(spread)//2]
        return med <= 0.12 * (fy1 - fy0)

    for p in pg.get_drawings():
        r = p["rect"]
        if not (fx0-3 <= r.x0 and r.x1 <= fx1+3
                and fy0-3 <= r.y0 and r.y1 <= fy1+3):
            continue
        n_it = sum(1 for it in p["items"] if it[0] in ("l", "c"))
        if n_it < 1:
            continue
        col = p.get("color")
        col = tuple(round(float(c), 3) for c in col) if col else None
        w = p.get("width") or 0.0
        # ONE PATH, SEVERAL CURVES. Split first and only fall back to the whole
        # path when the split does not produce two or more frame-spanning pieces.
        # The subpath candidates inherit the parent's item count and stroke width,
        # because those describe the PEN and are what the mono branch reasons
        # about -- they are properties of the drawing, not of its pieces.
        parts = [q for q in (gv.subpaths(p["items"]))
                 if wide(q) and single_valued(q) and falls(q)
                 and smooth(q)]
        if len(parts) >= 2:
            for q in parts:
                cand.append((q, col, n_it, w))
            continue
        pts = flatten(p["items"])
        if len(pts) < 8:
            continue
        # a curve crosses a useful part of the frame; a tick or a legend rule
        # does not. 20 % of the frame width, the same floor granularity_vector
        # uses, and it is what rejects the three legend swatches on 5201.
        if max(x for x, _ in pts) - min(x for x, _ in pts) < span_min:
            continue
        if not (single_valued(pts) and falls(pts) and smooth(pts)):
            continue
        cand.append((pts, col, n_it, w))
    if not cand:
        return {}
    # mono sheet: everything is black ink, so there is no per-record identity to
    # read. ⚠ THE ORIGINAL RULE IS KEPT VERBATIM HERE -- >= 8 items, then the
    # THICKEST path -- and it has to be. Relaxing the item floor to 1 (which the
    # colour sheet needs, its curves being 2-4 beziers) and picking the longest
    # point list instead put 5231's f50 at 607.8 cycles/mm: the log grid is one
    # path with far more points than the curve, and it "falls through 50 %" at
    # the frame's right edge. A regression that only shows up as a plausible
    # number is exactly what EXPECTED exists to catch, and it did.
    def is_grey(c):
        return c is None or (max(c) - min(c) < 0.12)
    if all(is_grey(c) for _, c, _, _ in cand):
        thick = [t for t in cand if t[2] >= 8]
        if not thick:
            # ⚠ THE 8-ITEM FLOOR IS A TIE-BREAK NOW, NOT A GATE. It exists to
            # keep the log grid out of a mono sheet's single-curve choice, and
            # since C2b the shape filters above (single-valued, falls, smooth)
            # do that far better -- while the floor by itself refuses a
            # legitimate curve drawn in few beziers. Agfa's Vista sharpness
            # panel draws its whole transfer curve in FIVE, and this branch was
            # returning "no curve inside the frame" on the only non-Kodak MTF
            # sheet in the corpus.
            thick = cand
        # ⚠ GREY INK DOES NOT MEAN ONE CURVE. Try the printed letters first; only
        # a frame with no usable R / G / B triple falls back to the single-curve
        # rule, which is what 5231 (a black-and-white stock) needs.
        # ⚠ AND THE TEST IS ">= 3", NOT "== 3", SINCE C2b. The exact-three form
        # silently skipped every 1990s technical sheet: those frames also carry a
        # log grid drawn as a frame-spanning path, so the candidate list is four
        # and the letters were never consulted. `letter_assign` already refuses
        # anything it cannot map as a bijection at a stacked abscissa, so handing
        # it a longer list costs nothing and is what the extra path needs.
        if len(thick) >= 3:
            byletter = letter_assign(pg, thick, fx0, fx1, fy0, fy1)
            if byletter is not None:
                return byletter
        return {"-": max(thick, key=lambda t: t[3])[0]}
    cand = [(pts, col) for pts, col, _, _ in cand]
    out = {}
    for pts, col in cand:
        if is_grey(col):
            continue
        rec = min(IDEAL, key=lambda t: sum((col[k]-IDEAL[t][k])**2
                                           for k in range(3)))
        # yellow and magenta both land on R; keep the denser of the two, which is
        # the same overprint collapse the granularity extractor does
        if rec not in out or len(pts) > len(out[rec]):
            out[rec] = pts
    return out


def overlay(pg, got, fx, fy, path):
    """Draw every traced point back onto the rendered panel, per record.

    ⚠ THE OVERLAY IS THE GATE, and this extractor did not have one until C2b --
    which is exactly how the single-path defect above survived: the numbers it
    produced were plausible (an f50 of 69.7 for a 500T stock) and nothing showed
    that the "curve" they came from walked across all three records. Both other
    plot extractors in this project (`vision3_granularity`, `granularity_vector`)
    grew the same gate for the same reason, and their docstrings say so.

    Colours are the record's own, magenta for an unidentified single curve.
    """
    from PIL import Image, ImageDraw
    dpi = 200.0
    pix = pg.get_pixmap(dpi=int(dpi))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    dr = ImageDraw.Draw(img)
    k = dpi / 72.0
    cols = {"R": (220, 0, 0), "G": (0, 170, 0), "B": (0, 80, 255),
            "-": (255, 0, 200)}
    for rec, pts in got.items():
        c = cols.get(rec, (255, 140, 0))
        for x, y in pts:
            X, Y = x * k, y * k
            dr.ellipse([X - 1.6, Y - 1.6, X + 1.6, Y + 1.6], fill=c)
    img.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--overlay", metavar="DIR",
                    help="write ov_<tag>.png with every traced point drawn back "
                         "onto the page -- look at it before believing any number")
    ns = ap.parse_args()
    import pymupdf
    if ns.overlay:
        import os
        os.makedirs(ns.overlay, exist_ok=True)

    bad = 0
    for tag, (fn, pgno, prof, (fx0, fx1, fy0, fy1)) in SHEETS.items():
        # A sheet may name its maker's subdirectory ("AGFA/....pdf"); a bare
        # filename still means KODAK, which is where every sheet came from until
        # C2b went looking for a second manufacturer.
        base = Path(ns.root).resolve() / "PDF" / "PROFILES"
        pdf = (base / fn) if "/" in fn else (base / "KODAK" / fn)
        if not pdf.is_file():
            print(f"  [SKIP] {tag}: source not present: {fn}")
            continue
        pg = pymupdf.open(pdf)[pgno-1]
        xs, ys = {}, {}
        for a, b, c, d, t, *_ in pg.get_text("words"):
            if not re.fullmatch(r'\d+', t):
                continue
            v = float(t)
            cx, cy = (a+c)/2.0, (b+d)/2.0
            if fx0-12 <= cx <= fx1+12 and fy1 < cy <= fy1+22:
                xs[v] = cx                       # spatial frequency, below
            elif fx0-26 <= cx < fx0-1 and fy0-8 <= cy <= fy1+8:
                ys[v] = cy                       # response %, left
        if len(xs) < 4 or len(ys) < 4:
            print(f"  [FAIL] {tag}: ticks x={len(xs)} y={len(ys)}")
            bad += 1
            continue
        fx = logfit(xs, "spatial frequency")
        fy = logfit(ys, "response")
        got = pick_curves(pg, fx0, fx1, fy0, fy1)
        if not got:
            print(f"  [FAIL] {tag}: no curve inside the frame")
            bad += 1
            continue
        print(f"[i] {fn} p{pgno} -> {prof}")
        print(f"    freq axis {fx[0]:.2f} px/decade, residual {fx[2]:.2f} pt, "
              f"{fx[3]} ticks; response axis {abs(fy[0]):.2f} px/decade, "
              f"residual {fy[2]:.2f} pt, {fy[3]} ticks")
        pins = EXPECTED.get(tag, {})
        for rec in sorted(got, key=lambda k: "RGB-".index(k)):
            a = np.array(got[rec])
            f = 10.0 ** ((a[:, 0] - fx[1]) / fx[0])
            r = 10.0 ** ((a[:, 1] - fy[1]) / fy[0]) / 100.0
            o = np.argsort(f)
            f, r = f[o], r[o]
            # ⚠ A FRAGMENT IS REFUSED, NOT MEASURED. Some sheets emit a record
            # in pieces (a leader line, a label gap, a curve that leaves and
            # re-enters the frame), and the splitter then hands back a piece.
            # A piece has an f50 -- 5293's red fragment reports 32.0 cycles/mm
            # from a 30-125 c/mm arc that starts BELOW 53 % -- and it is
            # meaningless. A response curve must cover most of the plotted
            # frequency range and must start above 50 %, since that is what
            # "the frequency where it falls through 50 %" presumes.
            # ⚠ THE TEST IS "starts at full response over at least a decade",
            # NOT "covers most of the frame". Kodak draws these curves over the
            # 2-100 cycles/mm the film can actually resolve while the frame is
            # ruled to 600 or 1000, so a frame-coverage rule refuses every real
            # curve on the sheet -- it did, on all eight, before being corrected.
            # What a fragment cannot fake is where it STARTS: every intact curve
            # in this corpus begins at 96-110 % response, while 5293's red piece
            # begins at 53 % and 5205's at 77 %.
            fspan = np.log10(f.max()) - np.log10(f.min())
            if fspan < 1.0 or r[0] < 0.90:
                print(f"    [SKIP] {rec}: fragment, not a curve -- {fspan:.2f} "
                      f"decades starting at {r[0]*100:.0f} % response (an intact "
                      f"curve starts near 100 %). Refused rather than measured")
                continue

            # f50 at the LAST downward crossing of 0.5
            #
            # ⚠ A CURVE MAY END *AT* 50 % INSTEAD OF PASSING THROUGH IT, and
            # that is a reading, not a failure. F-4043 (2016) stops T-MAX 400's
            # curve at 81 cycles/mm with a last plotted response of 50.6 % --
            # Kodak's convention on that sheet is to draw the curve down to
            # half response and stop. Before 2026-09-06 this branch refused it
            # with "never falls through 50 %", which is true of the drawing and
            # false of the film.
            # ⚠ THE TOLERANCE IS DELIBERATELY TIGHT (1 % of response, i.e. the
            # curve must end between 50.0 and 51.0 %) AND THE VALUE IS FLAGGED.
            # A curve that stops at 60 % has genuinely not reached f50 and must
            # still be refused; widening this to "take the endpoint whenever the
            # crossing is missing" would silently turn every truncated plot into
            # a measurement, which is the failure mode the fragment test above
            # exists to prevent.
            ENDPOINT_TOL = 0.010
            above = np.where(r >= 0.5)[0]
            at_end = False
            if not len(above):
                print(f"    [FAIL] {rec}: the curve never reaches 50 %")
                bad += 1
                continue
            if above[-1]+1 >= len(f):
                if r[-1] - 0.5 <= ENDPOINT_TOL:
                    at_end = True
                    f50 = float(f[-1])
                elif tag in REFUSED:
                    print(f"    [i] {rec}: REFUSED AS EXPECTED -- the curve "
                          f"stops at {r[-1]*100:.1f} % response, "
                          f"{(r[-1]-0.5)*100:.1f} points above the crossing, so "
                          f"the sheet gives f50 > {f[-1]:.0f} and not a value")
                    print(f"        {REFUSED[tag]}")
                    continue
                else:
                    print(f"    [FAIL] {rec}: the curve never falls through "
                          f"50 % -- it stops at {r[-1]*100:.1f} % response, "
                          f"{(r[-1]-0.5)*100:.1f} points above the crossing")
                    bad += 1
                    continue
            else:
                i = above[-1]
                f50 = float(np.interp(0.5, [r[i+1], r[i]], [f[i+1], f[i]]))
            if at_end:
                print(f"    [i] {rec}: the curve ENDS at {r[-1]*100:.1f} % "
                      f"response, so f50 is read at the printed endpoint rather "
                      f"than interpolated inside the drawing")
            pk = int(np.argmax(r))
            print(f"    {rec}: {f.min():.1f}-{f.max():.1f} cycles/mm, response "
                  f"{r.min()*100:.1f}-{r.max()*100:.1f} %  ->  f50 = "
                  f"{f50:.1f} cycles/mm, overshoot {r[pk]-1.0:+.3f} "
                  f"(peak at {f[pk]:.1f} cycles/mm)")
            print("      " + score_carrier(
                f, r, f50, max(ROLLOFF_FROM, float(f[pk]))))
            w = pins.get(rec)
            if w:
                if abs(f50 - w["f50"]) > TOL_F:
                    print(f"    [FAIL] {rec} f50 moved: {f50:.1f} vs recorded "
                          f"{w['f50']:.1f}")
                    bad += 1
                if abs((r[pk]-1.0) - (w["peak"]-1.0)) > TOL_P:
                    print(f"    [FAIL] {rec} overshoot moved: {r[pk]:.3f} vs "
                          f"recorded {w['peak']:.3f}")
                    bad += 1
        missing = set(pins) - set(got)
        if missing:
            print(f"    [FAIL] records pinned but not found: {sorted(missing)}")
            bad += 1
        print("    (the overshoot FREQUENCY is reported, not stored; see the "
              "module note on adjacency_um)")
        if ns.overlay:
            from pathlib import Path as _P
            out = str(_P(ns.overlay) / f"ov_{tag}.png")
            overlay(pg, got, fx, fy, out)
            print(f"    overlay -> {out}")
    # ---- the artwork-reuse identities, RE-DERIVED rather than remembered ----
    # ⚠ THIS IS THE ONLY THING STANDING BETWEEN THE DATABASE AND A PHYSICALLY
    # IMPOSSIBLE NUMBER. F-4001's P3200 panel is clean vector art, on the right
    # page, under the right caption, in the film's own publication -- nothing
    # about it looks wrong. It is caught only by comparing it against the sheet
    # it was copied from, so that comparison runs on every build.
    for tag, (a, b, prof) in ARTWORK_REUSE.items():
        pts = []
        for fn, pgno, fr in (a, b):
            base = Path(ns.root).resolve() / "PDF" / "PROFILES"
            pdf = (base / fn) if "/" in fn else (base / "KODAK" / fn)
            if not pdf.is_file():
                pts = None
                print(f"  [SKIP] artwork-reuse {tag}: source not present: {fn}")
                break
            g = pick_curves(pymupdf.open(pdf)[pgno-1], *fr)
            if "-" not in g:
                pts = None
                print(f"  [FAIL] artwork-reuse {tag}: no mono curve in {fn}")
                bad += 1
                break
            pts.append(np.array(g["-"]))
        if not pts:
            continue
        p, q = pts
        if p.shape != q.shape:
            print(f"  [FAIL] artwork-reuse {tag}: point counts {p.shape[0]} vs "
                  f"{q.shape[0]} -- THE PANELS NO LONGER MATCH. {prof}'s "
                  f"refusal was based on them being the same drawing; if the "
                  f"publisher has redrawn one, re-read the sheet before "
                  f"trusting either")
            bad += 1
            continue
        d = float(np.abs((p - p.min(axis=0)) - (q - q.min(axis=0))).max())
        if d > 0.01:
            print(f"  [FAIL] artwork-reuse {tag}: max deviation {d:.4f} pt -- "
                  f"the two panels have diverged; re-read before trusting")
            bad += 1
            continue
        print(f"[i] MISPLACED FIGURE CONFIRMED for {prof}: {a[0]} p{a[1]} is "
              f"the same vector object as {b[0]} p{b[1]} -- {p.shape[0]} "
              f"points, max deviation {d:.4f} pt. That EDITION's panel is "
              f"unreadable; the profile's value comes from another edition.")

    print()
    if bad:
        print(f"[FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("[OK] MTF read from the sheet's vector path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
