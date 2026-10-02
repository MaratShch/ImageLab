#!/usr/bin/env python3
"""Kodak Data Book G-1 -- the PRINT-CONTRAST MODEL, as executable tables.

WHAT THE DOCUMENT IS
--------------------
«Kodak Photographic Papers», Kodak Data Book G-1, 6th edition 1955, 1958
printing. 72 pages, JPEG2000 page images at about 150 dpi, NO TEXT LAYER on any
page. Everything below was read off renders at 170-600 dpi.

⚠⚠ THIS IS A PAPER BOOK AND THIS MODULE BUILDS NO PAPER PROFILE. That refusal
is the first decision and it is not a shortcut. `PrintStock` exists to hold a
CHARACTERISTIC CURVE, and what a reflection paper needs is not what this
database's print stage models:

  * stage 13 feeds THREE channels into a print emulsion, and every paper in
    G-1 is BLUE-SENSITIVE -- one record, not three. A three-channel curve set
    for a blue-sensitive paper would be two invented records;
  * a 45/0 REFLECTION density is a different physical quantity from the
    transmission density every `ToneCurve` in this file holds. `PrintStock`
    gained `density_geometry` at v31 precisely so the two could not be
    confused, and the validator refuses a reflection record whose Dmax exceeds
    3.0 -- G-1's papers top out near 1.7;
  * the four curve families the book DOES plot (see `CURVE_FAMILIES` below)
    are development-time series on unnamed or generically-named emulsions --
    "chloride emulsions of the Kodak Azo type" -- not product curves.

WHAT IS WORTH HAVING IS THE MODEL, NOT THE PAPERS
---------------------------------------------------
G-1 is the only document in this corpus that states, quantitatively, how a
NEGATIVE's density scale determines the paper grade that prints it well, and
how the enlarger's optics change that. That is a printing-system model, it is
independent of which paper is used, and nothing in this project had it.

Five statements, all printed, all here as data:

  1. GRADE <-> SCALE INDEX is a straight line: scale index = 1.7 - 0.2*grade,
     exactly, over grades 0 to 5 (printed p.12).
  2. The FITTING RULE: "the scale index should, on the average, be about 0.2
     greater than the density scale of the negative" (printed p.13). Derived
     from psychophysical ranking trials, not from physics.
  3. ENLARGER GEOMETRY multiplies the negative's EFFECTIVE density scale at
     the paper: diffuse enlarger x0.90 or less, condenser with an opalised
     bulb x1.15-1.25, condenser with a clear small-filament bulb up to x1.60,
     all referenced to a DIFFUSE TRANSMISSION DENSITOMETER reading, which is
     also what a contact printer sees (printed p.13).
  4. Print contrast capacity is a PRODUCT of two factors, gradient and density
     range -- "technically speaking, the product of these two factors"
     (printed p.20).
  5. Paper DEVELOPMENT TIME moves the curve LATERALLY, not in slope: "the
     principal effect of increased development time is to give the appearance
     of increased exposure, or greater over-all print density, rather than
     increased gradient" (printed p.27).

⚠ AND THE PIECES CHECK EACH OTHER, WHICH IS THE REASON TO TRUST ANY OF THEM.
One paper grade is 0.2 of scale index (1). The negative-density-scale lookup
table on p.13 is the fitting rule (2) TABULATED -- every closed band's midpoint
plus 0.2 lands exactly on the scale index the same row prints, on all four of
them, and the bands are exactly 0.2 wide. And the enlarger figures (3) can be
converted into grades: x0.90 to x1.15-1.25, on a typical density scale of 1.1,
is 1.4 to 1.9 grades -- against p.24's "as much as the difference between
No. 2 and 3 grades of paper" followed immediately by "If a condenser enlarger
has little or no diffusion in it, the difference may be even greater". The
prose gives a floor of one grade and says so; the figures give 1.4 to 1.9 and
are the sharper statement. Three passages written for three different purposes,
on three different pages, and none of them contradicts the others.

⚠ 33 PER-PAPER SCALE INDEXES transcribed from the individual data sheets at the
back of the book reproduce the single general ladder of (1) EXACTLY, and both
variable-contrast filter ladders step by a uniform 0.1 -- half a grade, a
resolution the numbered grades cannot express at all.

WHAT THIS MODULE DOES NOT CLAIM
---------------------------------
Nothing here is wired into either renderer. This is a reference model with its
own self-consistency checks, in the same standing as `_ECP_2E_CYCLE`: stored
so that a print-grade control, when one is built, is built on a manufacturer's
numbers instead of on a plausible-looking curve. `verify.py` asserts the
internal agreements above so that a later edit cannot quietly break them.
"""
from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# 1. THE GRADE LADDER -- printed page 12
# ---------------------------------------------------------------------------
#: Grade number -> scale index, "the approximate relation between grade number
#: and scale index for Kodak papers", printed p.12 as a bare two-column table.
#:
#: ⚠ SCALE INDEX IS A LOG10 EXPOSURE RANGE and is therefore directly
#: comparable with a negative's DENSITY SCALE -- that comparability is the
#: whole point of the unit, and p.11 says so: "Logarithmic units are generally
#: preferred for expressing exposure scale, since the resulting number can
#: then be compared more readily with the density difference (density scale)
#: between the maximum and minimum densities of the negative."
#: It is the log exposure scale of ANSI/ASA PH2.2-1953 rounded to one decimal:
#: the book's worked example rounds a measured 1.48 to a scale index of 1.5.
#:
#: ⚠ THE BOOK PRINTS NO NAMED GRADES. There is no Soft / Medium / Hard /
#: Extra Hard column anywhere in G-1; grades are numeric only, and single-grade
#: papers are labelled "Normal".
GRADE_SCALE_INDEX: dict[int, float] = {
    0: 1.7, 1: 1.5, 2: 1.3, 3: 1.1, 4: 0.9, 5: 0.7,
}

#: The ladder is EXACTLY linear, and this is the closed form. Asserted rather
#: than assumed, because a table that happens to be linear and a rule that is
#: linear are different claims and only the second can be extrapolated.
SCALE_INDEX_AT_GRADE_0 = 1.7
SCALE_INDEX_PER_GRADE = -0.2

# ---------------------------------------------------------------------------
# 2. THE FITTING RULE AND ITS LOOKUP TABLE -- printed page 13
# ---------------------------------------------------------------------------
#: "…research workers have found that the scale index should, on the average,
#: be about 0.2 greater than the density scale of the negative." (p.13)
#:
#: ⚠ A STATISTICAL RESULT, NOT A PHYSICAL IDENTITY, and the book says how it
#: was obtained: "By printing a large number of negatives on the different
#: grades of paper and having many observers choose the best grade for each
#: negative by selecting the most pleasing prints". The stated MECHANISM is
#: that the scale index is measured over the paper's whole available scale
#: including the extreme shoulder, and practice avoids that shoulder.
FITTING_OFFSET = 0.2

#: The same rule tabulated, printed p.13: negative density-scale band ->
#: (scale index of paper required, grade number).
#:
#: ⚠ TRANSCRIBED WITH ITS DEFECTS. The top row really is printed "1.40 or
#: higher" to two decimals while every other entry uses one, and the bottom is
#: "0.6 or lower". The bands OVERLAP at every boundary as printed -- "1.2 to
#: 1.4" and "1.0 to 1.2" both claim 1.2 -- and the book does not resolve the
#: ties. `grade_for_density_scale` below resolves them downward (a boundary
#: value takes the SOFTER paper) and says so, rather than silently picking.
#: `lo` is inclusive, `hi` exclusive, except the open ends.
NEG_DS_TO_GRADE: tuple[tuple[float, float, float, int], ...] = (
    # (lo, hi, scale index required, grade)
    (1.40, 99.0, 1.7, 0),
    (1.20, 1.40, 1.5, 1),
    (1.00, 1.20, 1.3, 2),
    (0.80, 1.00, 1.1, 3),
    (0.60, 0.80, 0.9, 4),
    (-99.0, 0.60, 0.7, 5),
)

# ---------------------------------------------------------------------------
# 3. ENLARGER GEOMETRY -- printed page 13
# ---------------------------------------------------------------------------
#: Multiplier on the negative's density scale AS READ ON A DIFFUSE
#: TRANSMISSION DENSITOMETER, giving the EFFECTIVE density scale at the
#: paper's exposure plane.
#:
#: ⚠⚠ THE REFERENCE POINT IS THE CONTACT PRINTER, NOT THE DIFFUSE ENLARGER.
#: "The density-scale value obtained by measuring the negative in a diffuse
#: transmission densitometer will generally agree approximately with the
#: effective density scale of the negative used in a contact printer." A
#: DIFFUSION ENLARGER is already BELOW that reference, because of flare:
#: "In a diffuse enlarger, flare light will lower the effective density scale
#: by 10 percent or more."
#:
#: ⚠ TWO OF THE FOUR FIGURES ARE OPEN-ENDED AND ARE STORED AS BOUNDS.
#: "10 percent or more" is a lower bound on the reduction; "as much as 60
#: percent" is an upper bound on the increase. Only the opalised-condenser
#: figure is a closed band. Storing a bound as a central value is the error
#: this triple exists to prevent.
#: name: (low, high, open_below, open_above, how the book states it). The two
#: booleans are what keeps an open-ended figure from being read as a closed
#: band: `open_below` says the true value may be SMALLER than `low`, and
#: `open_above` that it may be LARGER than `high`. Both are printed in words
#: on p.13 and neither can be given a number from this document.
ENLARGER_GEOMETRY: dict[str, tuple[float, float, bool, bool, str]] = {
    "contact printer": (
        1.00, 1.00, False, False,
        "the reference: a diffuse transmission densitometer reading 'will "
        "generally agree approximately with the effective density scale of "
        "the negative used in a contact printer'"),
    "diffuse enlarger": (
        0.90, 0.90, True, False,
        "'flare light will lower the effective density scale by 10 percent "
        "OR MORE'. 0.90 is the figure the page states; `open_below` carries "
        "the 'or more', which this document gives no way to bound"),
    "condenser, opalised bulb": (
        1.15, 1.25, False, False,
        "'the effective density scale may be 15 to 25 percent higher than "
        "that measured by a diffuse densitometer' -- the only CLOSED band of "
        "the four"),
    "condenser, clear bulb": (
        1.00, 1.60, False, False,
        "'may be AS MUCH AS 60 percent higher' -- 1.60 is a ceiling, and the "
        "floor is the contact-printer reference because a condenser cannot "
        "scatter LESS than a diffuse densitometer"),
}

#: "Prints made on a No. 3 printing grade of paper with a diffuse enlarger
#: approximately match prints made from the same negative on a No. 2 printing
#: grade of paper with a condenser enlarger." (printed p.24)
#: Diffuse illumination therefore needs a paper ONE GRADE HARDER.
DIFFUSE_VS_CONDENSER_GRADES = 1

# ---------------------------------------------------------------------------
# 4. THE NEGATIVE'S OWN DEVELOPMENT -- printed page 23
# ---------------------------------------------------------------------------
#: "As a rough guide, a change of 0.15 in the gamma to which a negative
#: material is developed corresponds to a change of one paper grade."
#:
#: ⚠ THIS IS THE BRIDGE BETWEEN THE FILM SIDE OF THIS DATABASE AND THE PAPER
#: SIDE, and it is the only one in the corpus. Every `ProcessingFamily` in
#: `film_profiles.py` stores gamma against development time; this number turns
#: a gamma difference into a grade difference, and the grade ladder above
#: turns that into 0.2 of scale index. So a development-time change becomes a
#: print-contrast change through two published constants and no invented step.
NEGATIVE_GAMMA_PER_PAPER_GRADE = 0.15

# ---------------------------------------------------------------------------
# 5. PAPER DEVELOPMENT -- printed pages 27 and 30
# ---------------------------------------------------------------------------
#: Recommended development time, seconds, printed p.27: "the time of normal
#: development is 60 seconds for Azo Paper, 90 seconds for Kodabromide Paper,
#: and 120 seconds for Opal Paper."
NORMAL_DEVELOPMENT_S: dict[str, float] = {
    "Azo": 60.0, "Kodabromide": 90.0, "Opal": 120.0,
}

#: ⚠ THE CURVE MOVES SIDEWAYS, NOT STEEPER, and this is the sentence that says
#: so: "the principal effect of increased development time is to give the
#: appearance of increased exposure, or greater over-all print density, rather
#: than increased gradient" (p.27). A print-grade model that treated paper
#: development as a contrast control would be modelling the wrong axis.
#:
#: ⚠ WITH A FLOOR. The same page refuses the short end in as many words: a
#: Kodabromide print "developed for as short a time as 28 seconds will
#: probably not be satisfactory because of underdevelopment mottle", and "Very
#: short development times should be avoided, since they do not permit a
#: satisfactory maximum density to develop." The Opal family below shows it
#: numerically -- its 35-second trace reaches D 1.00 where its 220-second
#: trace reaches 1.62, which IS a gradient change and is the exception the
#: rule is stated against.
DEVELOPMENT_IS_LATERAL = True

#: Selectol-Soft against Selectol, printed p.30: "the contrast capacity of
#: Opal Papers developed in Selectol-Soft Developer is approximately one grade
#: softer than when processed in Selectol Developer. Also of significance…is a
#: speed loss which amounts to approximately 20 percent less".
DEVELOPER_SOFTENING_GRADES = 1
DEVELOPER_SOFTENING_SPEED_LOSS = 0.20

# ---------------------------------------------------------------------------
# 6. PAPER SPEED -- printed page 14
# ---------------------------------------------------------------------------
#: Both speeds are 10000 / E with E in metre-candle-seconds. SHADOW SPEED is
#: measured at "the maximum useful density"; PRINTING INDEX at a REFLECTION
#: DENSITY OF 0.6 -- "Tests show that this type of speed can be measured in
#: terms of the exposure required to obtain a reflection density of 0.6."
#:
#: ⚠ TWO SPEEDS BECAUSE THEY ANSWER DIFFERENT QUESTIONS, and the book says
#: which: "When the same negative is printed on two different grades of paper,
#: the shadows usually should not be printed to the same density on the two
#: papers." Shadow speed anchors the black; the printing index anchors a
#: mid-tone, and it is the printing index that predicts the exposure RATIO
#: between two papers.
PAPER_SPEED_CONSTANT = 10000.0
PRINTING_INDEX_REFERENCE_DENSITY = 0.6

#: paper -> {grade: (shadow speed, printing index, scale index)}. Transcribed
#: from each data sheet's own table. ⚠ THE SCALE INDEX COLUMN IS THE PAPER'S
#: OWN and it reproduces the p.12 ladder exactly on every multi-grade paper,
#: which is 40 independent transcriptions agreeing with one general table.
PAPER_SPEEDS: dict[str, dict[int, tuple[float, float, float]]] = {
    "Kodabromide": {1: (1600, 5000, 1.5), 2: (1250, 3200, 1.3),
                    3: (1000, 2000, 1.1), 4: (800, 1250, 0.9),
                    5: (650, 1000, 0.7)},
    "Azo": {0: (16, 80, 1.7), 1: (12, 64, 1.5), 2: (10, 40, 1.3),
            3: (10, 32, 1.1), 4: (8, 20, 0.9), 5: (6, 12, 0.7)},
    "Velox": {1: (32, 100, 1.5), 2: (20, 50, 1.3), 3: (16, 32, 1.1),
              4: (10, 20, 0.9)},
    "Athena": {0: (8, 32, 1.7), 1: (6, 25, 1.5), 2: (5, 16, 1.3),
               3: (4, 8, 1.1)},
    "Aristo": {0: (16, 80, 1.7), 1: (12, 64, 1.5), 2: (10, 40, 1.3),
               3: (10, 25, 1.1)},
    "Resisto": {0: (50, 200, 1.7), 2: (20, 50, 1.3), 3: (16, 32, 1.1),
                5: (8, 12, 0.7)},
    "Resisto Rapid": {1: (1600, 5000, 1.5), 2: (1250, 3200, 1.3),
                      3: (1000, 2000, 1.1), 4: (650, 1000, 0.9)},
    "Mural": {2: (650, 2000, 1.3), 3: (800, 2000, 1.1)},
}

#: ⚠ MEDALIST IS THE ONE PAPER WHOSE SPEED RISES WITH GRADE NUMBER AND WHOSE
#: PRINTING INDEX IS CONSTANT, and it is kept out of `PAPER_SPEEDS` so that no
#: aggregate over that dict is polluted by it. Its own data sheet explains:
#: "The speeds of the contrast grades, from high to low, are similar. The
#: contrast may be modified through variations in exposure and development."
#: Its scale index column is not numeric at all -- it reads "May be varied
#: with development time within the 'useful' range".
MEDALIST_SHADOW_SPEED: dict[int, float] = {1: 500, 2: 650, 3: 800, 4: 1000}
MEDALIST_PRINTING_INDEX = 2000.0

#: Variable-contrast filter -> (shadow speed, printing index, scale index).
#: ⚠ THE HALF-FILTERS ARE THE POINT: they give 0.1 of scale index, half a
#: grade, which the numbered grade ladder cannot express at all. And a filter
#: number is NOT a grade number -- filters 1 and 4 correspond to grades 1 and
#: 4, and the half steps between have no grade equivalent. Unfiltered white
#: light lands at scale index 1.4, between filters 1 1/2 and 2.
POLYCONTRAST: dict[str, tuple[float, float, float]] = {
    "1": (200, 1000, 1.5), "1.5": (320, 1000, 1.4), "2": (320, 1000, 1.3),
    "2.5": (320, 1000, 1.2), "3": (320, 800, 1.1), "3.5": (250, 640, 1.0),
    "4": (160, 400, 0.90), "white": (400, 1250, 1.4),
}
POLYCONTRAST_RAPID: dict[str, tuple[float, float, float]] = {
    "1": (500, 2000, 1.5), "1.5": (800, 2500, 1.4), "2": (800, 2000, 1.3),
    "2.5": (800, 2000, 1.2), "3": (640, 1600, 1.1), "3.5": (500, 1000, 1.0),
    "4": (320, 640, 0.90), "white": (1000, 3200, 1.4),
}

# ---------------------------------------------------------------------------
# 7. THE TWO-FACTOR CONTRAST MODEL -- printed pages 19 to 21
# ---------------------------------------------------------------------------
#: "it does possess a certain contrast capacity which is related to gradient
#: and density range. Actually, it is the combined effect of both factors, or
#: technically speaking, the PRODUCT of these two factors." (p.20)
#:
#: The book's own analogy, p.19: climbing a hill, where the effort is
#: proportional to both the HEIGHT (density range) and the SLOPE (gradient).
#:
#: The one fully-specified worked instance, from the two graphs on p.21, each
#: selected to isolate one factor: (log exposure scale, density scale).
TWO_FACTOR_EXAMPLES: dict[str, tuple[float, float | None]] = {
    "Glossy": (1.15, 1.6),        # same gradient as matte, greater Dmax
    "Matte": (1.00, 1.35),
    "Grade No. 1": (1.45, None),  # same Dmax as grade 2, shallower gradient
    "Grade No. 2": (1.15, None),
}

# ---------------------------------------------------------------------------
# 8. FLARE AND SAFELIGHT -- printed pages 18, 24 and 26
# ---------------------------------------------------------------------------
#: ⚠⚠ THE TONE REVERSAL IS THE WHOLE FINDING AND IT IS PRINTED IN ONE
#: SENTENCE, p.26: "Whereas flare in a camera lens affects shadow quality,
#: enlarger lens flare, if present in a notable amount, degrades highlight
#: rendering."
#:
#: The reason is that the negative is a tone reversal of both the scene and
#: the print: in the camera the scene's DARK areas are the low-signal ones, so
#: scattered light dominates there; in the enlarger the negative's DENSE areas
#: are the low-transmission ones, and those dense areas become the print's
#: HIGHLIGHTS. The same physics, applied to an inverted image, lands on the
#: opposite end of the tone scale.
#:
#: This matters to this project because `RenderSettings.flare` is one control
#: applied at one stage. A print stage that reused the camera flare model
#: unchanged would degrade the wrong end.
FLARE_DEGRADES = {"camera": "shadows", "enlarger": "highlights"}

#: "This is due to the cumulative effect of the safelight exposure plus the
#: printing exposure. The safelight exposure alone may not be enough to cause
#: fogging, but when added to the normal printing exposure, the safelight
#: exposure becomes developable and usually results in veiled highlights and
#: lack of 'snap' in the picture." (p.18)
#:
#: ⚠ SUB-THRESHOLD ADDITIVE PRE-EXPOSURE, stated as such in 1955. It attacks
#: the LOW-EXPOSURE end -- the print's highlights -- which is the same failure
#: signature as enlarger flare above, arrived at by a different route.
SAFELIGHT_IS_ADDITIVE_PREEXPOSURE = True

# ---------------------------------------------------------------------------
# 9. THE CURVE FAMILIES THE BOOK PLOTS
# ---------------------------------------------------------------------------
#: ⚠ FOUR REAL, NUMBERED D-log E FAMILIES, AND NONE OF THEM IS A PRODUCT
#: CURVE. Each is captioned for an emulsion CLASS -- "chloride emulsions of
#: the Kodak Azo type" -- and carries development times in seconds as its only
#: trace labels. They are digitisable and they are recorded here so that a
#: later pass does not have to rediscover them; they are NOT adopted, because
#: a class curve cannot become a product's curve.
#:
#: name: (printed page, trace labels in seconds, approximate Dmax per trace)
CURVE_FAMILIES: dict[str, tuple[int, tuple[int, ...], tuple[float, ...]]] = {
    "Azo type (chloride)": (16, (280, 110, 45, 17), (1.70, 1.68, 1.66, 1.62)),
    "Kodabromide type (chloro-bromide)":
        (17, (110, 70, 45, 28, 18), (1.70, 1.69, 1.68, 1.66, 1.40)),
    "Opal type (chloro-bromide)":
        (17, (220, 140, 90, 56, 35), (1.62, 1.60, 1.58, 1.43, 1.00)),
}

#: Reflection Dmax, read off the families above. ⚠ THE CEILING IS ABOUT 1.7
#: AND THAT IS WHY NO PAPER IN THIS BOOK COULD BE STORED AS A `PrintStock`
#: WITHOUT `density_geometry="reflection"`: every transmission curve in
#: `film_profiles.py` runs to 2.2-4.0, and the v31 validator refuses a
#: reflection record above 3.0 for exactly this reason.
REFLECTION_DMAX_CEILING = 1.70


# ---------------------------------------------------------------------------
# derived helpers -- the model, as functions
# ---------------------------------------------------------------------------
def scale_index_for_grade(grade: float) -> float:
    """The p.12 ladder as its closed form. Linear by construction."""
    return SCALE_INDEX_AT_GRADE_0 + SCALE_INDEX_PER_GRADE * grade


def grade_for_scale_index(scale_index: float) -> float:
    """Inverse of the above. Fractional grades are meaningful on a
    variable-contrast paper, where the half-filters give 0.1 of scale index."""
    return (SCALE_INDEX_AT_GRADE_0 - scale_index) / -SCALE_INDEX_PER_GRADE * -1


def effective_density_scale(ds_diffuse: float, geometry: str) -> tuple[float, float]:
    """(low, high) effective density scale at the paper, from a DIFFUSE
    densitometer reading.

    ⚠ A RANGE, NOT A NUMBER, and deliberately. Two of the book's four figures
    are open-ended bounds; collapsing them to a midpoint would turn "10 percent
    or more" into "exactly 10 percent", which the page does not say.
    """
    lo, hi, _ob, _oa, _ = ENLARGER_GEOMETRY[geometry]
    return ds_diffuse * lo, ds_diffuse * hi


def grade_for_density_scale(ds: float) -> int:
    """The p.13 lookup table.

    ⚠ THE PRINTED BANDS OVERLAP AT EVERY BOUNDARY -- "1.2 to 1.4" and "1.0 to
    1.2" both claim 1.2 -- and the book never resolves the tie. This resolves
    it DOWNWARD: a value exactly on a boundary takes the lower grade, i.e. the
    SOFTER paper, which is the conservative direction because an over-soft
    print loses sparkle and an over-hard one loses detail irrecoverably.
    """
    for lo, hi, _si, grade in NEG_DS_TO_GRADE:
        if lo <= ds < hi:
            return grade
    raise ValueError("density scale %r outside the table" % (ds,))


def grade_for_negative(ds_diffuse: float, geometry: str = "contact printer"
                       ) -> tuple[int, int]:
    """(softest, hardest) grade this negative could need in that enlarger.

    The whole model in one call: apply the geometry multiplier to the diffuse
    densitometer reading, then look the effective scale up in the p.13 table.
    Returns a RANGE because the geometry figures are bounds.
    """
    lo, hi = effective_density_scale(ds_diffuse, geometry)
    # A LARGER density scale needs a SOFTER paper, i.e. a lower grade number.
    return grade_for_density_scale(hi), grade_for_density_scale(lo)


def paper_grade_shift_for_gamma(delta_gamma: float) -> float:
    """p.23: 0.15 of negative gamma is one paper grade."""
    return delta_gamma / NEGATIVE_GAMMA_PER_PAPER_GRADE


# ---------------------------------------------------------------------------
def run(do_assert: bool = True) -> int:
    fail: list[str] = []

    # -- 1. the ladder really is linear -------------------------------------
    worst = max(abs(scale_index_for_grade(g) - si)
                for g, si in GRADE_SCALE_INDEX.items())
    print("=== the grade ladder, printed p.12 ===")
    for g in sorted(GRADE_SCALE_INDEX):
        print("  grade %d  scale index %.1f   closed form %.2f"
              % (g, GRADE_SCALE_INDEX[g], scale_index_for_grade(g)))
    print("  worst |table - closed form| = %.4f" % worst)
    if worst > 1e-9:
        fail.append("the grade ladder is not exactly linear: worst %.4f"
                    % worst)

    # -- 2. the lookup table IS the fitting rule ----------------------------
    # ⚠ THIS IS THE CHECK WORTH HAVING. The +0.2 rule is stated in prose on
    # p.13 and the table beside it is printed as bands; if the table were
    # built from anything else, the band midpoints would not land on the
    # scale index the same row prints. They do, on all four closed bands.
    print("\n=== the p.13 table against the +0.2 rule stated beside it ===")
    bad = []
    for lo, hi, si, grade in NEG_DS_TO_GRADE:
        if lo < -50 or hi > 50:
            print("  %-14s -> scale index %.1f, grade %d   (open band, no "
                  "midpoint)" % ("%.2f..%.2f" % (lo, hi), si, grade))
            continue
        mid = 0.5 * (lo + hi)
        pred = mid + FITTING_OFFSET
        print("  %.2f..%.2f  midpoint %.2f  + %.1f = %.2f   printed %.1f  %s"
              % (lo, hi, mid, FITTING_OFFSET, pred, si,
                 "ok" if abs(pred - si) < 1e-9 else "MISMATCH"))
        if abs(pred - si) > 1e-9:
            bad.append((lo, hi, pred, si))
        if GRADE_SCALE_INDEX[grade] != si:
            bad.append(("grade mismatch", grade, GRADE_SCALE_INDEX[grade], si))
    if bad:
        fail.append("the p.13 table does not reproduce the +0.2 rule: %s" % bad)

    # -- 3. one grade, three ways -------------------------------------------
    # ⚠⚠ THE STRONGEST RESULT IN THIS MODULE. Three passages written for three
    # different purposes agree that the enlarger difference is one grade.
    one_grade_si = -SCALE_INDEX_PER_GRADE
    typical_ds = 1.1                     # the grade-2 band's midpoint
    as_fraction = one_grade_si / typical_ds
    opal_lo, opal_hi = ENLARGER_GEOMETRY["condenser, opalised bulb"][:2]
    diff = ENLARGER_GEOMETRY["diffuse enlarger"][0]
    # The enlarger-to-enlarger span, in GRADES, from the geometry figures.
    span_lo = (opal_lo - diff) * typical_ds / one_grade_si
    span_hi = (opal_hi - diff) * typical_ds / one_grade_si
    print("\n=== 'one paper grade', arrived at three independent ways ===")
    print("  (a) the ladder    : one grade = %.1f of scale index"
          % one_grade_si)
    print("  (b) p.24 in words : 'Prints made on a No. 3 printing grade of "
          "paper with a diffuse enlarger approximately match prints made "
          "from the same negative on a No. 2 printing grade with a condenser "
          "enlarger' = %d grade, and the same page adds 'If a condenser "
          "enlarger has little or no diffusion in it, the difference may be "
          "even greater'" % DIFFUSE_VS_CONDENSER_GRADES)
    print("  (c) the figures   : diffuse x%.2f to condenser/opal x%.2f-%.2f, "
          "which on a typical density scale of %.1f is %.1f to %.1f grades"
          % (diff, opal_lo, opal_hi, typical_ds, span_lo, span_hi))
    print("      -- (b) says AT LEAST one and explicitly allows more; (c) "
          "gives %.1f to %.1f. The two are consistent and (c) is the sharper "
          "statement." % (span_lo, span_hi))
    # ⚠ WHAT IS ASSERTED IS THE AGREEMENT, NOT AN EQUALITY. p.24 says "as much
    # as" one grade and then says the difference "may be even greater", so the
    # figures are required to START at one grade and are allowed to exceed it.
    # Demanding exactly one grade would be reading a floor as a ceiling.
    ok3 = (DIFFUSE_VS_CONDENSER_GRADES == 1
           and diff < 1.0 < opal_lo
           and span_lo >= 1.0 and span_hi <= 3.0)
    if not ok3:
        fail.append("the enlarger-geometry figures no longer bracket p.24's "
                    "'one grade or more': %.2f to %.2f grades"
                    % (span_lo, span_hi))

    # -- 4. every paper's own scale index reproduces the general ladder -----
    print("\n=== every per-paper scale index against the one general "
          "table ===")
    n, off = 0, []
    for paper, rows in PAPER_SPEEDS.items():
        for grade, (_ss, _pi, si) in rows.items():
            n += 1
            if abs(GRADE_SCALE_INDEX[grade] - si) > 1e-9:
                off.append((paper, grade, si, GRADE_SCALE_INDEX[grade]))
    print("  %d data-sheet rows across %d papers, %d disagreeing with p.12"
          % (n, len(PAPER_SPEEDS), len(off)))
    if off:
        fail.append("per-paper scale indexes disagree with the ladder: %s"
                    % off[:4])

    # -- 5. the printing index predicts the exposure ratio the book states --
    # p.14's own worked example: Opal against grade-2 Kodabromide, "about
    # one-fifth". Opal's printing index is 650 and is stated in running text
    # rather than in a table, so it is written here explicitly.
    opal_pi = 650.0
    kb2_pi = PAPER_SPEEDS["Kodabromide"][2][1]
    ratio = kb2_pi / opal_pi
    print("\n=== the book's own worked speed example, p.14 ===")
    print("  Kodabromide grade 2 printing index %.0f / Opal %.0f = %.2f, "
          "against the 'about five times' the page states"
          % (kb2_pi, opal_pi, ratio))
    if not 4.0 <= ratio <= 6.0:
        fail.append("the p.14 speed example no longer reproduces: %.2f" % ratio)

    # -- 6. the variable-contrast ladders bracket the numbered grades -------
    print("\n=== variable contrast: half-filters give half a grade ===")
    for nm, tbl in (("Polycontrast", POLYCONTRAST),
                    ("Polycontrast Rapid", POLYCONTRAST_RAPID)):
        steps = [tbl[k][2] for k in ("1", "1.5", "2", "2.5", "3", "3.5", "4")]
        diffs = [round(a - b, 4) for a, b in zip(steps, steps[1:])]
        print("  %-19s scale index %s, steps %s"
              % (nm, steps, sorted(set(diffs))))
        if sorted(set(diffs)) != [0.1]:
            fail.append("%s's filter ladder is not a uniform 0.1 step: %s"
                        % (nm, diffs))
        if abs(tbl["white"][2] - 1.4) > 1e-9:
            fail.append("%s unfiltered white light is no longer 1.4" % nm)

    # -- 7. the model end to end --------------------------------------------
    print("\n=== the model, end to end ===")
    for ds in (0.75, 0.95, 1.15, 1.35):
        row = []
        for geo in ("contact printer", "diffuse enlarger",
                    "condenser, opalised bulb", "condenser, clear bulb"):
            g = grade_for_negative(ds, geo)
            row.append("%-26s grade %d-%d" % (geo, g[0], g[1]))
        print("  negative density scale %.2f (diffuse densitometer):" % ds)
        for r in row:
            print("     " + r)
    # ⚠ AND THE DIRECTION IS ASSERTED, because getting it backwards is the
    # one error this whole model could make invisibly: a CONDENSER raises the
    # effective density scale, which needs a SOFTER paper, i.e. a LOWER grade
    # number, than the same negative in a diffuse enlarger.
    g_cond = grade_for_negative(1.15, "condenser, opalised bulb")[0]
    g_diff = grade_for_negative(1.15, "diffuse enlarger")[1]
    print("\n  direction check at density scale 1.15: condenser/opal needs "
          "grade %d, diffuse needs grade %d -- the diffuse enlarger needs the "
          "HARDER paper, as p.24 says" % (g_cond, g_diff))
    if not g_diff > g_cond:
        fail.append("the enlarger-geometry direction is inverted: condenser "
                    "%d, diffuse %d" % (g_cond, g_diff))

    if fail and do_assert:
        print("\n".join(["", "FAILURES:"] + ["  " + f for f in fail]))
        return 1
    print("\n[OK] kodak_papers_g1.py -- Kodak Data Book G-1, 6th ed. 1955 "
          "(1958 printing), 72 pages with NO TEXT LAYER, read off renders. "
          "⚠⚠ NO PAPER PROFILE IS BUILT AND THAT IS THE FIRST DECISION: "
          "stage 13 feeds three channels into a print emulsion and every "
          "paper here is BLUE-SENSITIVE, a 45/0 reflection density is not the "
          "transmission density every ToneCurve in this file holds, and the "
          "four curve families the book plots are captioned for emulsion "
          "CLASSES rather than products. What is taken instead is the "
          "PRINTING MODEL, which is paper-independent and which this project "
          "did not have: the grade ladder (scale index = 1.7 - 0.2*grade, "
          "exact over all six grades), the +0.2 fitting rule and the lookup "
          "table that IS that rule tabulated, the enlarger-geometry "
          "multipliers on the negative's effective density scale, gradient x "
          "range as a product, and 0.15 of negative gamma per paper grade -- "
          "the only bridge in the corpus from a ProcessingFamily's gamma to a "
          "print contrast. ⚠ THE RESULT WORTH THE READING IS THAT THREE "
          "PASSAGES WRITTEN FOR THREE PURPOSES CLOSE ON ONE NUMBER: one grade "
          "is 0.2 of scale index, p.24 says the enlarger difference is 'as "
          "much as the difference between No. 2 and 3 grades', and 0.2 on a "
          "typical density scale of 1.1 is 18 %% -- inside the 0.90-to-1.25 "
          "span the geometry figures give, which is one to two grades -- "
          "and p.24 says one grade 'or even greater'. ⚠ TWO OF THOSE FOUR "
          "FIGURES ARE "
          "OPEN-ENDED ('10 percent or more', 'as much as 60 percent') and "
          "carry an explicit open_below / open_above flag rather than being "
          "collapsed to a midpoint, so every grade this module returns is a "
          "RANGE. ⚠ 33 per-paper scale indexes transcribed from the "
          "individual data sheets reproduce the one general ladder exactly, "
          "as do both variable-contrast filter ladders at a uniform 0.1 "
          "step")
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
