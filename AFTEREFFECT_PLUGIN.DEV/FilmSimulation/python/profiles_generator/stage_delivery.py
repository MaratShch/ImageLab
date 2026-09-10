#!/usr/bin/env python3
"""Stage and zip the five delivery archives. Strictly separated, by design.

⚠ FIVE ARCHIVES, NOT ONE, AND THE SEPARATION IS THE POINT. The owner integrates
each into a different place: the generator into `PYTHON/profile_generator`, the
database into `CPP/Algorithm/FilmProfile`, the two engines into
`CPP/Algorithm/Scalar` and `CPP/Algorithm/AVX2`, and the Markdown into the doc
tree. A single archive would make every delivery a merge.

⚠ AND THE TWO ENGINE ARCHIVES ARE NEVER MERGED INTO ONE GENERIC TREE. The
scalar twin computes in `double` and the AVX2 twin in `float`; they are two
files carrying one law, and the project's own rule is that they stay two files.
`interimage_parity.py` now compiles BOTH and compares each to the Python
reference, which is what makes keeping them separate safe rather than merely
tidy.

What each archive holds is listed in `MANIFEST` below with the reason, so a
future reader does not have to infer the layout from the zip.
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPP = Path("/root/work/tst")           # the live, editable engine tree
OUT = Path("/root/work/deliver9")
#: ⚠ SUFFIXED. A second delivery was cut on the same day at schema v30,
#: and two archives named for one date cannot be told apart on disk.
STAMP = date.today().isoformat() + 'e'

#: Generator sources: everything needed to REGENERATE the database.
#: ⚠ NOT everything needed to run every audit. Until 2026-09-10d this tuple's
#: output was joined by the generated C++, because `cpp_parity.py`,
#: `interimage_parity.py`, `spectral_mono_parity.py` and `field_coverage.py`
#: compile against it. The owner's directive that day was that the generator
#: archive carries no C++, so those four now need archive 2 unpacked beside
#: this one. See the note at the archive-1 assembly for the full reasoning.
#: ⚠ `*.md` IS ROOT-LEVEL ONLY and picks up Tasks.md. The doc/ TREE is
#: deliberately NOT in this archive: the owner's instruction is five
#: strictly separated components, and shipping the whole doc tree here as
#: well as in archive 5 made every delivery a two-place merge. Previous
#: deliveries did carry it; this one does not, on purpose.
PY_GLOBS = ("*.py", "*.txt", "*.lock", "*.md")

#: The generated database, in the layout the Visual Studio project expects.
DB_FILES = (
    "film_profiles.hpp", "film_profiles_detail.hpp", "film_profiles.cpp",
    "film_enum.hpp", "LoadFilmDataBase.h", "LoadFilmDataBase.cpp",
    "film_names.txt", "film_display_order.txt",
    # ⚠ SHIPS WITH THE DATABASE, 2026-09-08. The alphabetical re-sort moved 177
    # of 184 indices; this is the old -> new map, and it is the only thing that
    # can repair a project saved before the cutover. It belongs beside the
    # database it describes, not in a report.
    "film_id_migration.txt",
)

MANIFEST = """\
FIVE ARCHIVES -- {stamp}e
=========================

⚠ THIS SUPERSEDES BOTH EARLIER SETS OF THE SAME DAY. Schema v30; verify 669
PASS; three further patents harvested; and ARCHIVE 1 NO LONGER CONTAINS ANY
C++. The .vcxproj instruction below is UNCHANGED and still outstanding.

⚠⚠ ARCHIVE 1 HAS LOST ~2 MB OF C++, ON YOUR DIRECTIVE ("don't put C++ code
into archive with python"). It previously carried the 24
film_profiles_data_*.cpp slot files and every DB_FILES entry -- a byte-for-byte
duplicate of archive 2 -- so that four audits would run out of the box.
THE COST, so it is not a surprise: cpp_parity.py, interimage_parity.py,
spectral_mono_parity.py and field_coverage.py now need archive 2 unpacked
BESIDE archive 1 before they will run. They fail with a missing-file error --
loud and immediate, never a wrong result. build.py is unaffected; it generates
the C++ itself before running them. A guard in stage_delivery.py now REFUSES
to build archive 1 if any .cpp/.hpp/.h ever creeps back into it.


1_python_generator.zip     generator, schema, every reader, every audit.
                           ⚠ NO C++ -- see the note above; four audits need
                           archive 2 unpacked beside this one to run
2_generated_database.zip   the generated database in the FilmProfile/ layout
3_algorithm_scalar.zip     the scalar engine  -- AlgoType = double
4_algorithm_avx2.zip       the AVX2 engine    -- AlgoType = float
5_documentation_md.zip     every Markdown document, reviewed against this build

⚠⚠ READ FIRST: WHAT YOU MUST DO BEFORE YOU BUILD
=================================================

(1) ADD SIX FILES TO THE VS2015 .vcxproj BY HAND

        film_profiles_data_19.cpp      (outstanding from 2026-09-06f)
        film_profiles_data_20.cpp      (outstanding from 2026-09-08)
        film_profiles_data_21.cpp      NEW
        film_profiles_data_22.cpp      NEW
        film_profiles_data_23.cpp      NEW
        film_profiles_data_24.cpp      NEW

    Your tree at CPP/Algorithm/FilmProfile has 18. Archive 2 contains 24.
    Nothing in the repository can edit the project file. A missed file fails
    at LINK time with an unresolved AppendFilmProfiles_NN -- a loud failure,
    never a wrong render and never a shifted film index. CMake globs and needs
    no change.

(2) NOTHING ELSE. No film index moved, no rendered pixel moved, and the two
    engines are byte-for-byte the code of the previous delivery. Archives 3
    and 4 are included only so that all five components stay in step; if you
    have already integrated the previous delivery's engines you can ignore
    them entirely.

⚠⚠ ADDED IN THE e REVISION: THREE MORE PATENTS, NO NEW FIELD
=============================================================

US 5,449,592 (Konica, colour proof), US 3,701,783 (Barr et al., Eastman
Kodak, priority 1963 -- in substance the FOUNDATIONAL DIR COUPLER PATENT) and
US 4,199,363 (Chen, Eastman Kodak, loaded latex). All three read in full, all
three USEFUL, 45 findings between them. Schema stays at v30: what they
changed is two docstrings and one enum value, not the shape of any record.

1. A CORRECTION TO v29, AND IT IS THE MOST USEFUL THING IN THE THREE.
   `_GAMMA_CRITERIA` gained `chord_0.30_0.80`. At v29 the entry beside
   `chord_0.80_1.80` read "Konica colour paper", and the reading this project
   made was that the chord endpoints identify the MAKER. They do not.
   US 5,449,592 is also Konica and uses D 0.30 to 0.80 on a PDA-65
   densitometer. Both are correct: the endpoints bracket the working density
   range of the MATERIAL, so a proof film aiming at halftone dot densities
   reads 0.30-0.80 where a print paper aiming at a full reflection scale
   reads 0.80-1.80. ⚠ NOT A CONFLICT TO BE RESOLVED -- two correct criteria
   for two different materials. "Konica" is not a criterion and neither is
   any other maker's name; the criterion belongs to the MEASUREMENT and must
   be read off the document that printed the number, never inferred from
   assignee, era or sibling patents. That is exactly what this field exists
   to prevent, and its own first draft committed the error.
   ⚠ SCHEMA_VERSION DID NOT MOVE. Widening an enum changes no field, no
   offset and no emitted byte, and no stock uses the new value.

2. FOUR CONSTRAINTS ON THE DIR MODEL, from the patent that invented it.
   Recorded in the `CouplerSpec` docstring; none is a storable number and all
   four say something about the SHAPE of the model:
     * THE RANGE IS CAPTURE-LIMITED, NOT DIFFUSION-LIMITED. Example 2 places
       a silver chloride barrier at 1.08 g Ag/m2 for the stated purpose that
       "the silver chloride absorbs the inhibitor compound". A species that
       is consumed as it travels decays exponentially, not as a Gaussian.
       ⚠ The stage uses a Gaussian. Recorded as an argument against it and
       DELIBERATELY NOT ACTED ON: changing the kernel moves every colour
       render and no source in the corpus gives the decay constant.
     * IT IS DRIVEN BY COUPLING RATE, NOT DEVELOPED SILVER. Example 9's
       four-way control: grain and contrast identical in a black-and-white
       developer; the DIR effect appears only under colour development.
     * INHIBITION MUST BE IMAGEWISE. Example 14's equimolar bis-coupler
       control released inhibitor uniformly and "gave uniform transfer, no
       image".
     * IT IS A TOE-SHAPE CHANGE, NOT A GAMMA SCALE. FIG. 2 curve 3: the DIR
       coupler "lowered the contrast of the dye image and the speed of the
       toe region has been increased" where the competing non-DIR coupler
       LOSES toe speed. Direction only; the figure has no ordinate values.

3. A USABLE PRIOR FOR COEFFICIENTS THAT CANNOT BE MEASURED. Examples 4 and 6
   size the inhibitors to "subtract green density and correct for the
   unwanted green absorption of the cyan dye image" -- the interimage
   off-diagonals were deliberately built to MIRROR THE DYE-IMPURITY MATRIX.
   This is why `_dye(k)` has always had to stand in for the net of impurity
   and interimage and why dye_matrix_from_spectra.py refuses: the two are
   entangled IN THE FILM, BY DESIGN, not merely in this model. Where
   interimage is unmeasurable -- which since the c revision means everywhere
   outside a patent A/B coating set -- the dye-impurity matrix is the
   best-grounded prior available for its shape.

4. THE NEGATIVE RESULT, and it closes the most promising lead there was.
   The c revision established that interimage coefficients can come only
   from (a) a patent A/B coating set differing solely in DIR loading, or
   (b) an own separation measurement. A 1963-priority patent demonstrating a
   brand-new effect is the likeliest place in the whole corpus for an
   UNREBALANCED A/B pair, because there was no established design to
   rebalance against. Barr has the pairs -- Example 5 Films A/B and C/D
   (+/- 20 mg/sq ft of Coupler XI and XLIII), Example 4 Films A/B (DIR
   against non-DIR cyan coupler at equal weight) -- and Example 5 even uses
   the separation-exposure protocol. ⚠ AND PRINTS NO SENSITOMETRY FOR EITHER
   MEMBER OF ANY PAIR. There is no table. The entire quantitative yield is
   one scalar: "in Film B seven times as much silver was developed as in
   Film A" (about 0.85 log units), and one loading, 20 mg/sq ft ~ 1.1 mol% Ag.
   Route (a) is now checked in its strongest candidate and remains open.

5. A CONFLICT ON COUPLER DROPLET SIZE, recorded and not averaged. US 3,816,121
   measures 0.12 um against 2 um costing 70 % of sensitivity (-0.52 log E).
   US 4,199,363 measures SIX PAIRED COATINGS differing only in dispersion
   route -- loaded latex 0.02-0.2 um against oil 0.3-0.9 um -- and finds the
   speed moves only -0.041 to +0.177 log E, five of six inside +/-0.05.
   Gamma is what actually moves: +0.47, +1.20, +0.20, +0.14, +0.33, -0.40.
   The paired coatings are the stronger evidence, one variable changed, but
   neither figure is discarded. Both agree the droplet is 0.02-0.9 um while
   `dye_cloud_um` holds 1.5-2.5, so the dye cloud is set by diffusion AFTER
   coupling and not by the droplet it grew from.
   ⚠ THE `coupler_dispersion` FIELD STAYS REFUSED, re-examined against new
   evidence and upheld. US 4,199,363 supplies the missing physics and still
   makes no shipping stock's oil phase knowable; its own paired coatings then
   show the route is worth under 0.05 log E in five cases of six.

6. A THIRD MEDIUM FOR THE DYE-SPECTRUM QUESTION. At identical 0.20 g/m2 UV
   absorber and 0.54 g/m2 gelatin, D370 reads 1.6 (no solvent), 1.60 (dibutyl
   phthalate) and 3.00 (latex) while D415 reads 0.55, 0.42 and 0.11 -- peak
   1.92x up, edge-to-peak 0.33 -> 0.021. Directly comparable to US 6,110,658's
   solution-versus-in-film band narrowing already on file.

⚠⚠ ADDED IN THE c REVISION: SCHEMA v30, ONE FIELD
==================================================

InterimageSpec.gamma_ratio_criterion, and it was added BEFORE its first value
rather than after it -- which is the whole point of it.

A web sweep of the patent literature in English, Japanese, German, French and
Russian found SEVEN mutually incompatible published definitions of "the
interimage effect", from Kodak, Agfa, Fuji and Konica. They differ in what is
exposed, in what is measured, and in which way up the ratio goes, and two are
not ratios at all: Agfa's is a percentage aggregated over two records and
cannot be split back into per-record values, and Fuji's is a HORIZONTAL
log-exposure separation where the others are vertical. The three v29
interimage_gamma_ratio fields are populated on zero stocks, so the criterion
costs nothing now and prevents, before it can happen, the exact error
DyeImpurity.measurement_mode was added at v29 to prevent AFTER it happened.

⚠ ONE OF THE SEVEN IS A TRAP AND validate() REFUSES IT OUTRIGHT. Kodak's R --
red gamma over green gamma under a single white-light exposure -- is printed
in US 5,989,798 and EP 0 851 288 A1 with the statement that low R indicates
high interlayer interimage, and their coatings measure 0.70/0.85 with no DIR
coupler against 0.41-0.58 with one. It is correctly reported, and it does not
transfer to a finished product.

THE CHECK THAT SETTLED IT, run on this database rather than argued: all 84
colour negatives measure R = 0.856-1.127, median 0.972, and the 45 tier-1
datasheet-traced ones are indistinguishable from the tier-3 analogy estimates.
On Kodak's reading every shipping film here would carry LESS interimage than a
deliberately inhibitor-free control coating, which cannot be true of VISION3
or PORTRA.

⚠ THE FIRST EXPLANATION WAS WRONG AND IS RECORDED AS SUCH. The suspicion was
that the three records had been normalised together in the published curves --
an ingest defect. False: only 4 of 115 colour stocks share a mid_slope, 71
carry a real r < g < b dmin mask ladder above 0.2 D, and the toe_x spread
across records has a median of 0.12. The stored curves carry genuine
per-record differences.

The real reason is a property of the film, and it is THIS PROJECT'S REASONING
attributed to no source: a colour negative must hold a neutral across its
exposure scale or a grey ramp drifts in colour, so its layers are built to
matched contrast and a red record suppressed by interimage is given more
inherent contrast to compensate. R is about 1 by design on any finished
product, whatever the chemistry inside, and the patents' low values come from
experimental coatings that were never rebalanced.

⚠ THE RESEARCH CONSEQUENCE, worth more than the field: per-stock interimage
coefficients are STRUCTURALLY UNOBTAINABLE from manufacturer data. The signal
is engineered out before publication. They can come only from a patent A/B
coating set, or from a separation-exposure measurement made here. No further
datasheet harvesting will produce them -- which retires a search that would
otherwise have run indefinitely.

⚠ WHAT DID SURVIVE IS AN INVARIANT, now enforced as G-IIE-NEUTRAL: if real
layers are balanced, whatever this pipeline does to a NEUTRAL must leave them
balanced. Measured across the 80 stocks with an active interimage stage, R
moves 0.9697 -> 0.9717 median, worst -0.0205 and +0.0287, against a 0.05
bound. It is not a check on coefficient magnitude -- interimage is supposed to
move saturated colour hard; this pins only the neutral.

Unharvested leads recorded on the queue, all free and reachable: SU 1062640 A1
(ORWO/Wolfen, Russian, control 12 % vs 17 %, 2 % vs 35 %); DE 3711418 A1
(Agfa, prints the formula); DE 19749589 A1; US 5,283,163; US 4,528,263;
US 4,830,954; and the best single one, US 5,262,287 / EP 0 442 323 A2 (Fuji),
which runs the white-vs-separation protocol across 17 coatings.

⚠⚠ THE v29 HEADLINE, UNCHANGED: SCHEMA v28 -> v29, EVERY RENDER IDENTICAL
==========================================================================

This delivery is a DATA and SCHEMA delivery. It adds nineteen fields, fills
five of them across essentially the whole database, and does not move a single
pixel. `verify.py` asserts that last claim rather than merely stating it
(G-V29-INERT, below).

WHERE IT CAME FROM. 41 documents read in full on 2026-09-10:

    34 patents      CN, EP, RU, US -- 1941 to 2002
     1 thesis       an MSc on photographic emulsion preparation and
                    characteristics
     2 volumes      Pierre Glafkides, «Photographic Chemistry», 1022 pages

564 findings. Every document was read for its CONTENT: none was set aside for
its filename, title, abstract, country, classification, apparent subject or
date. That mattered -- several of the most useful numbers came out of patents
about coupler chemistry and colour paper, which a title-based triage would
have discarded.

Exactly ONE document carried nothing usable: US 2,249,541 (1941). Its content
is an organic-synthesis catalogue of splittable vat-dye coupling derivatives
cited through Berichte and Annalen, with no example, no sensitometry, no
spectra, no grain data and no coating data. That is a judgement about what is
printed in it, not about its age.

Two scanned patents had NO text layer at all and were OCR'd from page images:
CN 1135434 C (138 pages, Chinese) and RU 2172512 C1 (14 pages, Russian).

1. WHAT LANDED IN THE DATABASE
==============================

Newly populated, with the population and the evidence class:

    EmulsionSpec.base_um            19 -> 180 stocks    CLASS DEFAULT
    EmulsionSpec.grain_um                 170           CLASS DEFAULT
    EmulsionSpec.iodide_mol_pct           170           CLASS DEFAULT
    EmulsionSpec.size_sigma_log           170           DERIVED
    EmulsionSpec.coated_um          13 ->  92           DERIVED
    EmulsionSpec.antihalation              28           DOCUMENTED per stock
    EmulsionSpec.antihalation_undercoat_um  3           CLASS DEFAULT (2.0 um)
    ProcessingFamily rate law               4           FITTED to each stock's
                                                        own stored points
    ProcessingFamily.temp_q10               3           CLASS CONSTANT
    LayerStack per-layer resolving power    4           DOCUMENTED
    AgingSpec.shrinkage_pct                 2           DOCUMENTED
    ReseauSpec.angle_deg                    1           DOCUMENTED

ParamSource records 2114 -> 2131.

⚠ "CLASS DEFAULT" IS NOT "MEASURED", AND THE DISTINCTION IS CARRIED IN EVERY
CITATION. Glafkides measured a 1950s emulsion census; that is DOCUMENTED for
the class and an ENGINEERING ESTIMATE for any one product, and each emitted
`EmulsionSpec.source` says so in those words. Two fields were deliberately
NOT written from that census -- `habit` and `aspect_ratio` -- because for a
post-1980 tabular stock they would be flatly false rather than approximate:
measured C-41 fast-layer iodide runs 6-12 mol% against his 4-6 mol% ceiling,
and measured tabular aspect ratios run 6-24 against his extra-fast class at
7.7.

2. THE NINETEEN NEW FIELDS
==========================

SIX ATTACH A DEFINITION TO A NUMBER THAT WAS ALREADY STORED. This is the part
of the harvest that changed how the schema thinks, and each one exists because
the mistake it prevents was actually made during this harvest before the field
existed.

    GrainSpec.rms_aperture_um       an rms figure with no aperture is not a
                                    figure. Kodak read through 48 um, Konica
                                    through 25 um; treating the two as
                                    comparable overstates Konica grain by
                                    sqrt(48/25) = 39 %, which is more than the
                                    whole spread between a 100-speed and a
                                    400-speed stock. Default 48.0 -- the
                                    convention every stored figure was already
                                    entered under.
    DyeImpurity.measurement_mode    the magenta unwanted-blue absorption came
                                    back as 0.137/0.048 from one patent and
                                    0.36/0.20 from two others. NEITHER IS
                                    WRONG: the first is transmission on film,
                                    the second reflection on paper. Averaging
                                    them would have produced a dye_matrix
                                    describing no real material. US 6,110,658
                                    adds the third distinction and it is the
                                    largest: measured IN FILM the cyan dye's
                                    half bandwidth collapses 123 -> 73 nm and
                                    its peak moves 637 -> 619 nm against the
                                    same dye in SOLUTION.
    FilmProfile.gamma_criterion     nine mutually incompatible definitions of
                                    "gamma" appear in 41 documents. Konica's
                                    chord between D 0.80 and 1.80 and the ASA
                                    contrast index over a 1.5-decade window
                                    differ by tens of percent on one curve.
    FilmProfile.density_geometry    callier_q is a ratio between two of FOUR
                                    densities and never said which two.
    MTFSpec.turbidity_ref_log_e     the exposure at which the stored f50 was
                                    read.
    CoatingSpec.coverage_source     a coating weight is unreadable unless it
                                    says whether the silver-halide-to-metal
                                    conversion was applied (0.5745 for AgBr,
                                    0.7526 for AgCl).

THIRTEEN CARRY PHYSICS THE PIPELINE HAD NO REPRESENTATION FOR:

    ProcessingFamily.gamma_infinity / dev_rate_k / induction_t0_min
        gamma(t) = gamma_inf * (1 - exp(-k*(t - t0))), the Mees-Sheppard law
        (Glafkides Vol. 1 §211). Development points were being interpolated
        LINEARLY between sparse rows, which overstates gamma mid-bracket by
        about 4 % and worst exactly where sheets are thinnest. It also gives
        the first hard clamp on push processing this project has ever had:
        usable contrast tops out near 0.80 * gamma_infinity.
    ProcessingFamily.temp_q10
        2.378 for B&W (doubling per 8 degC) against 2.8-3.5 for colour
        reversal. Stored per family and NOT averaged.
    ReciprocitySpec.kron_a / kron_log_i0_rel / short_onset_s
        P = I*t*10^(-a*w), w = sqrt(1 + log10(I/I0)^2), a ~ 0.2 (§216).
        Schwarzschild's single exponent is wrong at BOTH ends -- unbounded
        loss at long times, unbounded GAIN at short ones. Kron's form has the
        right asymptotes, and its low-intensity limit p = 1/(1+a) = 0.8333
        independently reproduces the p ~ 0.85 the same book quotes from direct
        measurement. Two numbers from different pages agreeing to 2 % is the
        strongest internal cross-check in the whole harvest.
    MTFSpec.turbidity_gamma_um
        d = d0 + Gamma*log10(E), Gamma = 0 / 7 / 11 / 18 / 48 um per DECADE by
        material class (§§240-241, after Selwyn). Every spread function in
        this pipeline is exposure-invariant and the emulsion's is not.
    EmulsionSpec.antihalation / antihalation_undercoat_um
        ⚠ THE FIELD THAT EXPLAINS AN EXISTING DEFECT RATHER THAN ADDING A
        NUMBER. HalationSpec.gain_r is set on all 184 stocks and NOT ONE has a
        document behind it -- the largest wholly-estimated field in the
        schema. Glafkides §§402-404 supplies the missing variable: halation
        onset is 80x threshold on an unprotected support and above 3000x with
        an absorbing layer, a 37.5x split (6.32 against 11.55 stops) decided
        by the CONSTRUCTION and nothing else. It is also the field
        CINESTILL_800T has needed since v17: that stock is VISION3 500T with
        the rem-jet stripped -- identical emulsion, identical designation,
        famously different halation -- and until now the difference lived in a
        hand-tuned scalar with a prose comment.
    InterimageSpec.interimage_gamma_ratio_r/g/b
        ⚠ THAT RECORD'S OWN DOCSTRING SAYS ITS SIX COEFFICIENTS ARE "reasoned
        ... not measured" AND NAMES THE MISSING QUANTITY. The harvest found it
        printed: neutral gamma over separation gamma, per record.
    AgingSpec.dye_fade_low_density_factor
        fade is worse in the shadows, so it LOWERS CONTRAST -- which a uniform
        fraction cannot produce. US 5,104,782 is the only document in the
        corpus that ran its light-fade test at two initial densities on every
        sample: 68 % residual from D 1.0 against 50 % from D 0.5.
    CoatingSpec.silver_g_per_m2 / gelatin_g_per_m2
        seventeen complete layer build-ups were harvested, plus Glafkides'
        per-class figures and the covering-power constant (1 g Ag/m2 develops
        to about D 1.0) that turns a total into a checkable bound on Dmax.
    ReseauSpec.angle_deg
        Dufaycolor's reseau runs at 27 degrees on the NEGATIVE and 45 on the
        POSITIVE, and Glafkides Vol. 2 §497 says why: two regular grids at the
        same angle beat against each other, so the angles differ ON PURPOSE.
        The renderer has been laying it out axis-aligned, which is the one
        angle guaranteed to produce the moire the maker engineered around.

3. FOUR CARRIERS SHIP ON ZERO STOCKS, EACH FOR A STATED REASON
===============================================================

This is deliberate and each refusal is enforced by a guard, so that populating
one later requires the missing measurement rather than a quiet edit.

TURBIDITY -- the law is documented, the five per-class coefficients are
documented, and the CONVERSION between them and this pipeline is not printed
anywhere in the corpus. Glafkides' Gamma is a coefficient on a point-image
diameter whose own d0 is tens of micrometres, read under a microscope on 1950s
plates; the stored f50 values imply d0 = 1.9-7.8 um. The two are not the same
diameter. Feeding Gamma = 18 into one decade of exposure would take an f50 of
100 cycles/mm to 14.6 -- a 6.8x loss of sharpness that no photograph shows.
⚠ AN EARLIER DRAFT OF THE FIELD COMMENT CLAIMED THE EFFECT WAS "about 2 lp/mm
across the tone scale". That was wrong by more than an order of magnitude and
the correction is recorded at the field, because the arithmetic that exposed it
is the reason the field ships empty.

KRON -- no document prints an optimum intensity I0 for any named product, and
p = 1/(1+a) is the law's LOW-INTENSITY LIMIT, so the existing
schwarzschild_p_* cannot be inverted into (a, I0). Inventing an I0 to preserve
a p would be inventing a measurement.

INTERIMAGE GAMMA RATIOS -- neither source names a product. US 6,531,271 gives
0.90-1.05 as a scanning-oriented DESIGN BAND; EP 0 608 959 B1 measures the
print-oriented case an order of magnitude stronger. ⚠ The two quote the ratio
the OPPOSITE WAY UP; the field is always neutral-over-separation and validate()
refuses anything above 1.05 with the reciprocal in the message.

COATING COVERAGES -- AGFA_NEU_1936 is the only named target in the database,
and Glafkides gives its per-layer silver as 0.7-1.0 g/m2 in one place and about
2 g/m2 in another. Conflict recorded, not averaged, nothing written.

`gamma_criterion` is likewise empty on all 184: 95 stored gammas are traced
softplus coefficients and 86 are this project's own estimates, so almost
nothing has a recorded definition to state yet. That is the condition the field
exists to retire, one stock at a time.

4. THREE PROPOSALS EXAMINED AND REJECTED
=========================================

per-layer layer_stack[] coating arrays
    Seventeen complete real stacks are available and nothing in the 27 stages
    reads a coating weight. A 15-element array no stage consumes is inert data
    with a maintenance cost. The TOTAL is stored instead, because the
    covering-power constant turns a total into a predicted Dmax bound that the
    stored curve can be checked against.

tabularity_T
    Algebraically redundant with aspect_ratio and grain_um.

a Callier Q(gamma) field
    ⚠ ALREADY IMPLEMENTED, AND BETTER CALIBRATED THAN THE PROPOSAL KNEW.
    `_callier_beta_for` has derived Q from each monochrome stock's own mid
    slope since queue C43 on 2026-09-02, calibrated on Mees FIG. 179.
    Stricker's independently measured table (Glafkides §203) is therefore
    wired in as a CROSS-CHECK and not as a second carrier -- and the two
    agree: mean |difference| 0.1342, worst 0.1821, over all 69 monochrome
    stocks (G-STRICKER).

Also refused: the plan's resolving-power values. Glafkides §233 (Perrin &
Hoadley) publishes resolving power as a TAKING-LENS pair (Fuess against
apochromat), while resolving_power_lp_mm_lowc/_highc is a TEST-OBJECT-CONTRAST
pair (1.6:1 / 1000:1). Different axes, so the numbers do not map -- and three
of the five named stocks already carry a manufacturer figure that the proposed
values would have overwritten. Logged as needing a lens-regime carrier.

5. TWO CONFLICTS RECORDED OPEN
===============================

THE GOST SPEED CRITERION. Glafkides gives the Soviet criterion as D = fog + 0.2
with S = 1/E. ГОСТ 9160-91 gives fog + 0.85 with S = 20/H, and RU 2172512 C1's
own Table 3 column head reads S(0.85) in print. The patent settles what IT
used; it does not settle what applied in the 1950s-60s, so the conflict stays
open for the 18 Soviet stocks. Not averaged.

THE MAGENTA LIGHT-STABILITY RANK INVERTS between the 1950s and 1980s coupler
chemistries. Glafkides' 1950s survey ranks magenta the MOST light-stable of the
three dyes; the 1980s patents measure pyrazolotriazole magenta fading FASTER
than 5-pyrazolone while staining about 10x less. So `dye_fade_m` cannot have a
single era-independent default, and none was written.

6. ONE DATA RECOVERY WORTH RECORDING
=====================================

RU 2172512 C1's Table 3 -- 23 examples of speed, contrast, fog and resolving
power for the Soviet spectrozonal family -- was invisible to every OCR pass
because page 14 is printed LANDSCAPE. Rendered at 300 dpi and read from the
page image directly: S(0.85) 30-700, gamma 1.6-2.8, D0 0.15-0.30, R 68-145
lp/mm. The prose's claim checks out (R 100-145 for the invention against 68 for
the comparison), and row 1 disagrees with the prose's own description of the
1964 prototype -- recorded as a conflict. No stock in this database is that
material, so the values are family reference only and nothing was written from
them.

The same method recovered EP 0 264 730 B1 Tables 6/7/8 and EP 0 520 310 A1
Table 4, plus fifteen further tables on those pages, by re-rendering at 400 dpi
and re-OCR'ing.

7. TWO THINGS THAT GREW, AND ONE THAT WAS TRIMMED
==================================================

THE DATABASE NEEDED FOUR MORE SLOT FILES: 20 -> 24. Emitted total 1 999 173 ->
2 249 861 bytes, all of it per-stock citations for the newly populated
EmulsionSpec records. Measured feasibility at the 112 000-byte per-file limit:
22 slots leaves 223 bytes of headroom, 23 leaves 4 458, and 24 leaves 8 816.
Twenty-two is arithmetically feasible and practically useless -- 223 bytes is
less than one citation, so the next stock added would break the build again and
cost another manual .vcxproj edit. Twenty-four was chosen for the same reason
the 19 -> 20 bump chose its margin.

⚠ AND THE FIRST DRAFT WAS 2 513 004 BYTES, OF WHICH 309 116 WAS THE SAME PROSE
REPEATED. The harvest pass had written its full reasoning -- a 2 403-byte
argument about support gauges, a 1 000-byte derivation per crystal class -- into
the `source` FIELD of every profile it touched: 26 identical copies of one
essay, 170 of another. The slot packer refused it, and that refusal was correct
for a better reason than size: a generated database is a SHIPPED ARTEFACT, so
the emitted string is a CITATION and the argument for it belongs in a Python
comment that no consumer pays for. All four templates were split that way, with
the reasoning preserved verbatim beside them. That recovered 263 kB.

⚠ THE TWO LARGEST REQUIREMENT DOCUMENTS HAD NEVER SHIPPED. `stage_delivery.py`
globbed `doc/*.md` and both `FilmDatabase_Charecteristics.MD` and its Russian
twin are spelled with an UPPERCASE extension -- and the glob is case-sensitive
on Linux. They were silently absent from every documentation archive ever
delivered. Fixed; archive 5 now contains both.

8. DOCUMENTATION
================

FilmDatabase_Charecteristics.MD      4513 -> 5190 lines
FilmDatabase_Charecteristics_Rus.MD  4688 -> 5385 lines
    Both brought to v29 and reconciled AGAINST EACH OTHER section by section.
    They had drifted: the English said an earlier gap was closed for two of
    three consumers where the Russian said one, and the Russian was right --
    the English contradicted its own two other sections. The Russian was
    missing a paragraph the English had; the English was missing a whole
    dated entry the Russian had. Both now carry a supersession index mapping
    every dated figure to the present state, and both keep their
    vendor-neutrality: no manufacturer, product family or individual material
    is named anywhere in them.
NotFound.md          new dated section, the three new gaps, and a twelve-line
                     reference wishlist for the sources this harvest points at
                     but does not contain
DIGITIZATION_QUEUE.md  new pass section, 14 rows, 10 closed
PROGRESS.md          new current-state entry
FilmActiveProfiles.md / FilmCurves.md / PROJECT_STATE.md   regenerated

9. VERIFICATION
===============

verify.py          669 PASS / 1 FAIL (the baselined saturation-hierarchy
                   failure you instructed not to fix). Seven NEW guards:
                     G-V29-SIGMALOG   every crystal size_sigma_log is in
                                      log10-of-DIAMETER units. Six independent
                                      derivations of this quantity disagreed
                                      by factors of exactly ln(10) and exactly
                                      2 before the convention was written out.
                     G-V29-TURBIDITY  the carrier ships empty, and says why
                     G-V29-KRON       likewise
                     G-V29-APERTURE   every rms is on the 48 um convention
                     G-V29-RATELAW    no fitted asymptote puts a stock's own
                                      published gamma above the usable ceiling
                     G-V29-INERT      no v29 field is read on the render path
                     G-STRICKER       the project's Callier rule against an
                                      independently measured Q(gamma) table
                   And one more at v30:
                     G-IIE-NEUTRAL    the interimage stage leaves a neutral
                                      ramp's red/green balance intact
cpp_parity         OK -- grain, MTF, reciprocity, Callier, whole database
interimage_parity  OK -- stages 8b and 9, BOTH flavours
bromide_parity     OK -- scalar and AVX2
spectral_mono      OK -- 69/69 monochrome stocks agree exactly
doc_consistency    OK -- every registered documentation count matches
compile            OK -- g++ -std=c++14 -Wall -Wextra, 26 TUs, zero output
build.py           OK -- 0 failures, 0 warnings
"""


def _zip(name: str, root: Path, files: list[Path]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    z = OUT / name
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(files):
            zf.write(f, f.relative_to(root).as_posix())
    return z


def stage_engine(kind: str) -> Path:
    """One engine tree in the include/ + src/ layout the owner's CMake expects.

    ⚠ THE AVX2 TREE IS THE SCALAR TREE WITH THE 18 VECTOR TWINS OVERLAID, which
    is exactly how the project is organised: the twins replace the per-stage
    .cpp files and AlgoTypes.hpp, and every other header is shared. Copying the
    shared set first and the twins second is what makes the overlay correct --
    the same ordering `interimage_parity.stage_avx2_tree` relies on, and for the
    same reason.
    """
    dst = OUT / "stage" / kind
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "include").mkdir(parents=True)
    (dst / "src").mkdir(parents=True)
    skip = {"film_profiles.hpp", "film_profiles_detail.hpp", "film_enum.hpp",
            "LoadFilmDataBase.h", "LoadFilmDataBase.cpp", "film_profiles.cpp"}
    for p in sorted(CPP.iterdir()):
        if not p.is_file() or p.name in skip:
            continue
        if p.name.startswith("film_profiles_data_"):
            continue
        if p.suffix in (".hpp", ".h"):
            shutil.copy(p, dst / "include" / p.name)
        elif p.suffix == ".cpp":
            shutil.copy(p, dst / "src" / p.name)
    if kind == "AVX2":
        for p in sorted((CPP / "AVX2").iterdir()):
            if not p.is_file():
                continue
            sub = "include" if p.suffix in (".hpp", ".h") else "src"
            shutil.copy(p, dst / sub / p.name)
    for extra in ("CMakeLists.txt",):
        src = Path("/root/work/deliver") / kind / extra
        if src.is_file():
            shutil.copy(src, dst / extra)
    return dst


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "MANIFEST.txt").write_text(MANIFEST.format(stamp=STAMP),
                                      encoding="utf-8")

    # ---- 1. python generator ----------------------------------------------
    # ⚠ NO C++ IN THIS ARCHIVE. Owner directive, 2026-09-10d: "don't put C++
    # code into archive with python". Until that date this archive also
    # carried the 24 `film_profiles_data_*.cpp` slot files plus every entry
    # in DB_FILES -- about 2 MB of generated C++, byte-for-byte duplicating
    # archive 2 -- on the argument that `cpp_parity.py`,
    # `interimage_parity.py`, `spectral_mono_parity.py` and
    # `field_coverage.py` compile against them and so would run out of the
    # box.
    #
    # ⚠ THE COST, STATED SO IT IS NOT A SURPRISE: those four audits now need
    # archive 2 unpacked beside this one before they will run. They fail with
    # a missing-file error, which is loud and immediate rather than a wrong
    # result. `build.py` is unaffected -- it generates the C++ itself before
    # running them.
    #
    # The directive is right on the larger point. Five archives exist so that
    # each drops into exactly ONE place in the owner's tree, and an archive
    # carrying another archive's payload turns every delivery into a merge
    # and invites two divergent copies of the same generated file. The
    # convenience was the exception; the separation is the rule.
    py = [p for g in PY_GLOBS for p in HERE.glob(g)]
    py = [p for p in dict.fromkeys(py) if p.is_file()]
    _cxx = [p for p in py if p.suffix in (".cpp", ".hpp", ".h", ".hxx", ".cc")]
    if _cxx:
        raise RuntimeError(
            "archive 1 would ship C++ (%s ...). PY_GLOBS has widened, and "
            "the owner's 2026-09-10d directive is that the generator archive "
            "carries no C++ at all."
            % ", ".join(p.name for p in _cxx[:3]))
    z1 = _zip(f"1_python_generator_{STAMP}.zip", HERE, py)

    # ---- 2. generated database --------------------------------------------
    db_stage = OUT / "stage" / "FilmProfile"
    if db_stage.exists():
        shutil.rmtree(db_stage)
    (db_stage / "include").mkdir(parents=True)
    (db_stage / "src").mkdir(parents=True)
    for n in DB_FILES:
        p = HERE / n
        if not p.is_file():
            continue
        sub = "include" if p.suffix in (".hpp", ".h") else "src"
        if p.suffix == ".txt":
            sub = "src"
        shutil.copy(p, db_stage / sub / n)
    for p in sorted(HERE.glob("film_profiles_data_*.cpp")):
        shutil.copy(p, db_stage / "src" / p.name)
    cm = Path("/root/work/deliver/FilmProfile/CMakeLists.txt")
    if cm.is_file():
        shutil.copy(cm, db_stage / "CMakeLists.txt")
    z2 = _zip(f"2_generated_database_{STAMP}.zip", db_stage,
              [p for p in db_stage.rglob("*") if p.is_file()])

    # ---- 3 and 4. the two engines, never merged ----------------------------
    sc = stage_engine("Scalar")
    av = stage_engine("AVX2")
    z3 = _zip(f"3_algorithm_scalar_{STAMP}.zip", sc,
              [p for p in sc.rglob("*") if p.is_file()])
    z4 = _zip(f"4_algorithm_avx2_{STAMP}.zip", av,
              [p for p in av.rglob("*") if p.is_file()])

    # ---- 5. documentation --------------------------------------------------
    # ⚠ BOTH CASES. `FilmDatabase_Charecteristics.MD` and its Russian twin
    # are spelled with an UPPERCASE extension, and this glob is
    # case-sensitive on Linux -- so until 2026-09-10 the two largest
    # requirement documents in the project were silently absent from
    # every documentation archive ever shipped.
    docs = sorted(set((HERE / "doc").rglob("*.md"))
                  | set((HERE / "doc").rglob("*.MD"))) + \
        [p for p in (HERE / "doc").rglob("*.txt") if p.is_file()]
    # ⚠ ROOTED AT THE GENERATOR, NOT AT ITS PARENT, so entries read
    # `doc/NAME.md` and unzip straight over the owner's doc tree. Rooting at
    # the parent prefixed every entry with the working directory's own name,
    # which made the archive un-unzippable in place.
    man = HERE / "doc" / "MANIFEST.txt"
    shutil.copy(OUT / "MANIFEST.txt", man)
    docs += [man]
    z5 = _zip(f"5_documentation_md_{STAMP}.zip", HERE,
              [p for p in dict.fromkeys(docs) if p.is_file()])

    for z in (z1, z2, z3, z4, z5):
        with zipfile.ZipFile(z) as zf:
            print(f"  {z.name:44s} {z.stat().st_size/1024:9.1f} kB  "
                  f"{len(zf.namelist()):4d} files")
    print(f"[OK] five archives in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
