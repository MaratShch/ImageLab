"""Build and audit driver for the film-profile generator.

WHY THIS EXISTS
---------------
Before this file there was no entry point for the generator chain. `run.cmd`
contained one line -- `python film_sim.py Lady.png -p all` -- which renders an
image and regenerates nothing. The regeneration sequence existed only as a
loose command list in `doc/README.md`, and the extraction scripts existed only
as prose references. Every consequence of that has actually happened at least
once in this project:

  * the project-root and profile_generator copies of the generated C++ drifted a
    whole generation apart, and nothing noticed;
  * `film_names.txt` has two competing generators and the documented order runs
    the DEPRECATED one last, silently replacing the file the effect panel loads;
  * `sigma_shape_toe/mid/dmax` sat in the schema for weeks, populated and
    validated, while no renderer read it;
  * a delivery was reported as applied when the files had not in fact changed;
  * the hand-written status docs fell behind the data, so the owner had to read a
    sweep of twenty-two files to work out where things stood.

None of those is a coding mistake. They are all the same missing thing: no
single command that regenerates everything in the right order and fails loudly
when an invariant breaks. That is what this is.

    python build.py                 # full regeneration + audit
    python build.py --check         # READ-ONLY: audits, and reports drift
    python build.py --only verify   # one stage
    python build.py --skip compile  # everything but one stage
    python build.py --list          # stages and what each needs

Stdlib only, no third-party imports at module level, Windows and POSIX. The
stages themselves need numpy/Pillow (and PyMuPDF for the PDF audits); a stage
whose dependency or input is missing SKIPS with a reason and does not fail the
build.

STAGE ORDER IS NOT COSMETIC
---------------------------
  audit    re-derive adopted numbers from the source documents. First, because
           if the stored data no longer matches its own sources there is no
           point regenerating anything from it.
  verify   the check suite. Second, for the same reason.
  codegen  film_profiles.{hpp,cpp}, film_enum.hpp, film_names.txt. Must run
           before `sync`, and `film_names.txt` must be written by THIS step --
           see the note on gen_film_names.py below.
  sync     copy the four generated artefacts to the project root and assert
           both copies are byte-identical. Closes the drift trap.
  docs     FilmActiveProfiles.md, FilmCurves.md -- regenerated from the data,
           so they cannot describe a database that no longer exists. Also gates
           doc/PROGRESS.md, the hand-written status board: it carries a
           build-facts stamp (schema version, stock count, film_names digest) and
           the stage FAILS if the stamp disagrees with the live database. The
           generated docs cannot go stale; a hand-written one silently can, and
           the owner reads it to see where the project stands.
  compile  g++ -std=c++14 -Wall -Wextra on the generated table. Gated on the
           compiler's own exit code AND an empty stderr, not on a piped head:
           that mistake once reported "clean" while a string literal was broken.

DELIBERATELY NOT RUN: gen_film_names.py
---------------------------------------
`doc/README.md` line 548 records it as "Deprecated. Superseded by
cpp_codegen.write_film_names(), which derives order from the emitted .cpp
instead of from FILM_PROFILES." It is nonetheless still listed in the README's
own command block, AFTER cpp_codegen.py -- so following the documented sequence
overwrites `film_names.txt` with a different set of display names (19 of 154
differ: "KODAK T-MAX 100" against "KODAK TMAX 100", and so on). The owner's
in-service file is cpp_codegen's version. This driver never invokes the
deprecated script and asserts afterwards that the file still matches what
cpp_codegen produces.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: What identifies a directory as the project root: it holds the PDF corpus the
#: audits are pointed at. Cheap, and true on every layout this has run on.
_ROOT_SENTINEL = "PDF"


def _default_root() -> Path:
    """Project root, which holds the second copy of the generated C++.

    On the owner's layout HERE is <root>\\PYTHON\\profile_generator, so the root
    is two levels up. Overridable with --root (or FILMSIM_ROOT) so the driver
    can be exercised against a staged copy of the corpus.

    ⚠ THE ARITHMETIC ALONE IS NOT SAFE, AND THIS COST A DAY ON 2026-08-23.
    `HERE` is `Path(__file__).resolve().parent`, and `.resolve()` follows
    symlinks. Where `PYTHON/profile_generator` is a symlink -- which is how the
    tree was laid out in one working copy -- two-levels-up lands somewhere else
    entirely. The sync stage then wrote the generated C++ into a directory nobody
    compiles and reported "both copies identical" about files that did not
    matter, while the real root kept a schema-v10 `film_profiles.hpp` for five
    days. The plugin failed to compile with "HalationSpec has no member named
    radius_scale_r" and the header looked correct in the generator directory.
        So: try the candidates in order of trust and take the first that actually
    looks like the root. If none does, return the arithmetic answer anyway and
    let `stage_sync` say so loudly -- guessing silently is what caused the bug.
    """
    env = os.environ.get("FILMSIM_ROOT")
    if env:
        return Path(env).resolve()
    # ⚠ abspath, NOT resolve: abspath makes the path absolute against the CWD
    # without following symlinks, so a symlinked checkout keeps its own path.
    # ⚠ AND NOT `Path(__file__).parent.parent.parent` EITHER -- invoked as a bare
    # `python3 build.py` that degenerates, because Path('.').parent is Path('.'),
    # which is how the first version of this fix still landed on the wrong root
    # and tripped its own warning.
    here_unresolved = Path(os.path.abspath(__file__)).parent
    candidates = [here_unresolved.parent.parent,      # as invoked, symlink kept
                  HERE.parent.parent]                 # resolved, symlink followed
    # Last resort: walk up from the file's own directory looking for the corpus.
    for up in list(here_unresolved.parents):
        if up not in candidates:
            candidates.append(up)
    # ⚠ AND IT CAN STILL FAIL, LEGITIMATELY. If the generator directory lives
    # OUTSIDE the project tree -- a symlinked working copy, which is how this has
    # actually been run -- no arithmetic on __file__ can reach the root, because
    # Linux getcwd() returns the physical path and the symlink is already gone by
    # the time this code runs. That case is not solvable here and is not supposed
    # to be: FILMSIM_ROOT (or --root) is the answer, and stage_sync WARNS loudly
    # when the root it was handed has no corpus in it. A visible wrong answer is
    # the whole design goal; the silent one cost five days.
    for c in candidates:
        try:
            if (c / _ROOT_SENTINEL).is_dir():
                return c
        except OSError:                              # pragma: no cover
            continue
    return candidates[0]

ROOT = _default_root()

#: The machine-generated artefacts, and the two places each must exist.
#: 2026-08-18: the profile table is split across 16 data-slot TUs plus the
#: explicit-initialisation API (LoadFilmDataBase), because a single 676 KB
#: init-list function was beyond VS2015 SP3 / ICC -- see cpp_codegen.py's
#: "Split emission" banner. film_names.txt is unchanged in format and order.
#: film_display_order.txt joined this list on 2026-08-28, with the identifier
#: freeze. It is GENERATED presentation order -- database indices sorted by
#: natural name -- and the panel needs it beside film_names.txt, so it has to
#: reach the project root like every other artefact. Leaving it out of this
#: tuple would have made it the one generated file that silently went stale.
GENERATED = ("film_profiles.hpp", "film_profiles.cpp",
             "film_enum.hpp", "film_names.txt", "film_display_order.txt",
             "film_profiles_detail.hpp",
             "LoadFilmDataBase.h", "LoadFilmDataBase.cpp") + tuple(
    f"film_profiles_data_{i:02d}.cpp" for i in range(1, 17))

#: Audit scripts: re-derive adopted values from the original documents and exit
#: non-zero if they stop reproducing. ONE LINE PER SCRIPT -- this table is the
#: thing that was missing, and it is why a new extraction script now becomes
#: part of the build instead of an orphan.
#:   script, argv, the input it needs, what it guards
def audits(root: Path):
    return (
        ("vision3_granularity.py",
         ["--pdfdir", str(root / "PDF" / "PROFILES" / "KODAK")],
         root / "PDF" / "PROFILES" / "KODAK",
         "the four VISION3 sigma(D) triples, traced from the Kodak TI sheets"),
        ("mees_granularity.py",
         ["--root", str(root)],
         root / "PDF" / "PROFILES" / "RETRO"
              / "THE THEORY OF THE Photographic PROCESS.pdf",
         "B&W silver-negative sigma(D) shape, Mees Fig. 302"),
        # ⚠ THE ONLY AUDIT WHOSE SOURCE IS DELIBERATELY NOT ADOPTED. Every other
        # entry in this list re-derives a number the database stores. This one
        # re-derives a SHAPE the database cannot express -- Q rising from unity
        # at the toe -- from a figure whose caveats (no stated collection angle,
        # one motion-picture POSITIVE stock, "O. Sandvik, private
        # communication") disqualify it as a source for `callier_q`. It is
        # registered anyway because that shape is the standing argument for
        # replacing the constant-Q model, and an argument resting on a trace
        # nobody re-runs decays into a remembered impression.
        ("mees_callier_q.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "RETRO" / "mees_fig179_p643.png",
         "Callier Q against diffuse density at five development gammas, Mees "
         "FIG. 179 -- five curves traced by columns above D 0.25, the shared "
         "toe traced by rows below it. REFERENCE ONLY: it touches no profile, "
         "and must not until a source states a collection angle"),
        # ⚠ THE FIRST AUDIT THAT MAKES A STORED NUMBER *RENDER* RATHER THAN
        # CHECKING ONE THAT ALREADY DID. Every colour `dye_matrix` in this
        # database was built by `_dye(k)` from a single scalar and was therefore
        # SYMMETRIC -- so every colour stock crosstalked the same way and
        # differed only in how much, while twelve of them carried traced dye
        # spectra that nothing read. These two audits close that: the first
        # verifies the ISO 5-3 status responses, the second integrates each
        # stock's own dye panel against them and asserts the ten adopted
        # matrices, the two refused panels, and the sign pattern.
        # ⚠ GUARDED ON ITSELF, NOT ON THE PDF IT CAME FROM, AND THAT IS
        # DELIBERATE. The tables are literals transcribed once from the page
        # images; the check is a SELF-check of those literals -- peaks, unit
        # peaks, monotone flanks, status M longer than status A in every band.
        # Guarding it on the source PDF would make it SKIP on any tree without
        # the corpus, which is precisely where a mistranscribed constant would
        # otherwise travel unnoticed.
        ("iso_5_3_status.py",
         ["--root", str(root), "--assert"],
         HERE / "iso_5_3_status.py",
         "the ISO status A and status M spectral responses transcribed from "
         "ANSI/ISO 5-3-1995 tables 3 and 4 -- peaks at 440/530/620 and "
         "450/540/640 nm, unit peaks, monotone flanks. ⚠ Read off the PAGE "
         "IMAGES, because the scan's OCR floats two red entries free of their "
         "wavelength rows and shifts the status A red response by 10 nm"),
        ("dye_matrix_from_spectra.py",
         ["--root", str(root), "--assert"],
         HERE / "film_profiles.py",
         "the ten dye matrices derived by integrating each stock's traced "
         "spectral dye density against the ISO 5-3 response its own "
         "density_metric names, checked against the literals the database "
         "stores, against the sign pattern every real dye set obeys, and "
         "against four Soviet manufacturing specifications the derivation "
         "never saw -- plus the two panels it refuses by name"),
        # ⚠ AN AUDIT OF A LAW WE DO **NOT** SHIP, KEPT FOR THE SAME REASON as
        # mees_callier_q: the case for changing the Callier law rests on a
        # measured divergence, and a divergence nobody re-measures decays into a
        # remembered impression. It also pins the two things that must stay true
        # of any replacement -- exact inertness at specular 0, and exact
        # inertness for every colour stock at any setting.
        ("callier_silberstein_tuttle.py",
         ["--root", str(root), "--assert"],
         HERE / "film_profiles.py",
         "Silberstein & Tuttle's published specular/diffuse relation (Mees "
         "printed p644) against the linear law film_sim and AlgoCallier ship: "
         "both exactly inert at specular 0 and at Q = 1.0, identical at both "
         "ENDPOINTS, and diverging by up to 0.21 D in between -- the shipped "
         "law interpolates the multiplier, the published one interpolates "
         "transmittance, which is what light actually does"),
        # ⚠ QUEUE B4. Four plates that sat unread for a year behind a blocker
        # recorded as "axis calibration is not solved". It was two things and
        # neither was calibration: the outermost gridline on each axis is
        # fainter than the interior ones and fell under the ink threshold, and
        # the embedded raster is stored UPSIDE DOWN behind the page's own flip
        # transform. Registered because the adopted MTF, base+fog triple and
        # dye pair all come off these traces.
        ("ti0835_plates.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK" / "5247.pdf",
         "the four TI0835 plates for EASTMAN_5247_1983 -- MTF, characteristic, "
         "spectral and spectral dye density -- each calibrated on its own grid "
         "landing on round steps, traced by exact plate ink, and guarded on "
         "the properties an upside-down or legend-contaminated read breaks: "
         "records stacked blue > green > red, spectral peaks in order and "
         "within 18 nm of the stored set, and a D-min trace that FALLS towards "
         "the red because it is the orange mask"),
        ("dye_density.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK",
         "the 12 adopted spectral dye density sets, re-derived from the "
         "sheets' vector paths (5285 and 2383 are the validation pair; 7239, "
         "5217 and 5218 were recovered on 2026-08-18 from the FAILED list; "
         "5201 on 2026-08-25 by the ink-based family C)"),
        ("spectral_vector.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK" / "Kodak VISION2 50D 5201.pdf",
         "KODAK_VISION2_50D_5201's three spectral sensitivity curves, the "
         "first VECTOR-traced spectral set in the database -- layers assigned "
         "by Kodak's ink convention (the red record being a yellow-under-"
         "magenta overprint), peaks pinned at 470 / 540 / 650 nm and the "
         "absolute peak sensitivities the schema's per-layer normalisation "
         "throws away -- plus 12 further panels read as cross-checks against "
         "sets already adopted, and the three C37 disagreements C38 settled on "
         "2026-08-31: 5218 was the BROCHURE read against the technical sheet, "
         "5231 was this reader pairing a criterion caption with the wrong "
         "curve, and only 5245's blue tail was a defect in the data"),
        ("polaroid_spectral.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "POLAROID" / "664fds.pdf",
         "the four POLAROID spectral sensitivity panels, two of them new data "
         "(Type 52 and Type 55 P/N) and two cross-checks that reproduce "
         "hand-read sets from 1999 sheets to rms 0.034 and 0.027 decades. ⚠ "
         "Queue item E2 prescribed NEGATING these curves -- the sheets' prose "
         "says they plot 'the equivalent energy needed' -- and negating them is "
         "exactly the mirrored reading that row warned about. The axis is "
         "sensitivity: the 667 edition captions it 'Spectral Sensitivity "
         "(cm^2/erg)', area per unit energy. The decisive check is asserted "
         "here rather than argued: peak plotted value must RISE with exposure "
         "index across all four sheets (EI 50 -> 9.8, 100 -> 15.6, 400 -> 98.0, "
         "3000 -> 233.1), which under the inverted reading would have the ISO "
         "3000 film needing fifteen times more light than the ISO 100 one"),
        # ---- queue E3, 2026-08-31 -----------------------------------------
        ("konica_raster.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KONICA" / "IMP50.pdf",
         "the seven KONICA plot panels, which are the first RASTER-ONLY sheets "
         "in this corpus to be adopted from: every figure in IMP50.pdf and "
         "INF750.pdf is an embedded bitmap with no paths and no tick text, so "
         "calibration is geometric off the printed grid and every panel "
         "re-detects its own gridlines before a curve is traced. ⚠ The bitmaps "
         "are also stored UPSIDE DOWN -- rotating them 180 degrees leaves the "
         "text mirror-reversed, which is how the flip announces itself. What it "
         "adopted: IMPRESA 50's characteristic curves, whose Dmin triple was a "
         "family template shared with two other KONICA stocks and wrong in blue "
         "by 0.32 D, and its visual-filter MTF (f50 64.9, not the estimated 72; "
         "121.4 % overshoot at 6.88 c/mm; power-law rolloff q 2.20 beating the "
         "Gaussian 2x); and INFRARED 750's curve at the sheet's own standard "
         "condition, Konicadol DP 6 min at 20 C, which moved gamma 0.72 -> 1.70 "
         "because all FIFTEEN printed curves are steeper than the value held. "
         "The decisive check is asserted rather than argued: IMPRESA 50's Dmin "
         "is read off TWO figures on TWO pages -- the characteristic plateau and "
         "the minimum-density spectrum sampled at the status M band centres -- "
         "and they agree to 0.005-0.015 D"),
        ("granularity_vector.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK" / "Ektachrome_100d.pdf",
         "the EKTACHROME 100D 5285 sigma(D) triple -- the only MEASURED "
         "granularity shape for a colour REVERSAL stock in the database, and the "
         "one that contradicts _grain_v2's reversal heuristic in sign"),
        ("mtf_vector.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK" / "5231-PLUS-X.pdf",
         "PLUS-X 5231's f50 and adjacency overshoot, read off the sheet's own "
         "vector MTF path (41.3 cycles/mm and +0.034, against a stored estimate "
         "of 60.0 and 0.08)"),
        ("kodak_sensitometry.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK" / "Kodak VISION2 50D 5201.pdf",
         "KODAK_VISION2_50D_5201's three characteristic curves, least-squares "
         "fitted to the DENSE curves inside the sheet's granularity panel (100 / "
         "125 / 121 samples, rms 0.005-0.007 D) rather than to the six-vertex "
         "polylines the sensitometric panel actually draws -- plus the abscissa "
         "origin (+1.9932 decades) that only the coarse panel states, and the "
         "cross-check that the sheet's two printed abscissae agree"),
        ("agfa_vista.py",
         ["--root", str(root / "PDF" / "PROFILES"), "--assert"],
         root / "PDF" / "PROFILES" / "AGFA"
              / "AGFACOLOR Vista 100, 200, 400, 800.pdf",
         "AGFA_VISTA_200's spectral sensitivity, and the dash-pattern legend "
         "it depends on: the extractor re-checks the solid/dashed/dash-dot to "
         "green/blue/red mapping against Agfa's own printed labels and against "
         "the per-layer absorption bands, for all three films on the page"),
        ("gevaert_curves.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "GEVAERT"
              / "Verpoort_Stapp1980_NewGevacolNeg682.pdf",
         "GEVACOLOR_NEG_682's three characteristic curves, re-traced from the "
         "1980 SMPTE paper's Fig. 10 at native scan resolution (589/513/437 "
         "samples, fit rms 0.004-0.011 D) -- and re-checked against the gamma "
         "0.57 the figure itself prints, which the trace reproduces at 0.5677"),
        ("kodak_time_gamma.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK"
              / "EASTMAN DOUBLE-X Negative Film 5222.pdf",
         "EASTMAN DOUBLE-X's printed five-point time-gamma family, re-derived "
         "from the five DRAWN curves on H-1-5222 p2 rather than from the text "
         "labels that state it. Reproduces 0.500 / 0.558 / 0.652 / 1.060 "
         "against the printed 0.50 / 0.56 / 0.66 / 1.05, and records the one "
         "that does NOT reproduce -- 9 minutes, measured 0.798 against a "
         "printed 0.84 -- as a named exemption, so a NEW disagreement on any "
         "other curve fails instead of hiding inside a loose tolerance. Also "
         "measures base+fog per development time (0.231 / 0.233 / 0.233 / "
         "0.275 / 0.296), which is what corrected the profile's dmin"),
        ("agfa_2004_curves.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "AGFA" / "AGFA stocks.pdf",
         "Agfa F-PF-E4 (4th edition, 08/2004) read as VECTOR: 12 spectral "
         "curves and 12 colour-density curves across four film columns, with "
         "the three layers separated by DASH ARRAY and the keying checked "
         "against the sheet's own printed Blue/Green/Red words. Fits the "
         "corpus's six-parameter ToneCurve to each density curve at rms "
         "0.005-0.016 D and cross-checks every fitted gamma against an "
         "independent steepest-chord slope. ⚠ Reports the Sharpness panel's "
         "overshoot (109-114 %) as adoptable adjacency but REFUSES its f50: "
         "the abscissa says 'Lines per mm' and whether that is line pairs is "
         "open queue item G6"),
        ("kodak_1952_curves.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK" / "kodak-films-5.pdf",
         "All FOUR 1952 Data Book curve families -- VERICHROME, TRI-X SHEET, "
         "PANATOMIC-X SHEET, ORTHO-X SHEET -- re-derived from the page "
         "RASTER, because these plots are not vector: queue E1 said they were "
         "and the 30 'drawing objects' on the Tri-X page are all zero-height "
         "table rules left by the Acrobat Paper Capture OCR. Traces all 20 "
         "curves at 150 dpi scan grade and reproduces every printed gamma, 18 "
         "of 20 within 2 %. ⚠ Also pins the ESTIMATOR: gamma is the steepest "
         "0.6-decade chord, and the fixed net-density window that works on "
         "H-1-5222 reads 5 % low here because the 1952 toes are far longer"),
        ("di_2254.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK"
              / "KODAK-VISION3-2254-technical-information.pdf",
         "KODAK_VISION3_DI_2254's three characteristic curves, traced from the "
         "RASTER sensitometric figure on H-1-2254 p3 (474 samples per record, "
         "fit rms 0.006-0.012 D) -- and the physical check that comes free with "
         "an INTERMEDIATE film: nothing in the trace was told that this stock "
         "exists to change nothing, and the fitted gammas come out 1.05 / 0.96 "
         "/ 1.04. It also re-checks the origin placement (the exposure at which "
         "the green record reaches D-min + 1.0, which is the reference the "
         "sheet's own dye-stability table is quoted at)"),
        ("kodak_still_curves.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK"
              / "KODAK PROFESSIONAL PORTRA - 2003 year.pdf",
         "The 2026-08-26 KODAK still-film harvest, re-derived from all eleven "
         "E-series sheets: 30 characteristic dmin/gamma pairs, 12 MTF f50 "
         "readings and 10 dye-pair peaks. Three things in it are audits of the "
         "READER rather than of one number. (1) PORTRA 160NC is read from BOTH "
         "E-190 vintages -- two files, different md5 -- and must return the "
         "same six values from each, which exercises the tick fitting, the "
         "subpath splitting and the letter matching together. (2) E-2468's "
         "characteristic panel is pinned to PORTRA 160VC's figure "
         "F009_0154AC, because it IS that figure: the defect stays visible, and "
         "the day a corrected edition appears the assertion fires. (3) The two "
         "dye-pair REFUSALS are asserted as refusals, so a change that starts "
         "accepting a crossing pair has to say so rather than adopt it quietly. "
         "One f50 is asserted to remain CENSORED (E-190 2003 p9 blue is still "
         "at 55 % where the plot stops), which keeps 'never reaches 50 %' "
         "distinguishable from a number. ⚠ EXTENDED 2026-08-31 (queue K3) to "
         "the ten PUSH panels: two adopted pairs, the EI 800 anchor that makes "
         "them a push rather than an edition difference, and five readings "
         "pinned precisely because they are NOT adopted -- three editions of "
         "PORTRA 800 give red gamma at EI 1600 as 0.6883, 0.6100 and 0.6341, "
         "and an unpinned disagreement is a claim nobody can re-check. A "
         "sixth check asserts a panel stays UNREADABLE: E-4040 (2016) p4's EI "
         "800 axis is printed -4.0 / -2.0 / -3.0 / -1.0 / 0.0 / 1.0, two "
         "labels transposed in Kodak's artwork, and a tick fitter that ever "
         "accepts it would be wrong by a decade mid-plot"),
        ("kodak_aim_density.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK"
              / "KODAK PROFESSIONAL PORTRA - 2003 year.pdf",
         "the sixteen published AIM DENSITY tables -- the red Status M "
         "densities Kodak says a correctly exposed negative reads on a gray "
         "card, a paper gray scale and a lit forehead -- read off thirteen "
         "sheets by geometry rather than page order, because the two table "
         "layouts emit their forehead pairs in opposite orders. Five of the "
         "twenty-one checks are cross-document: PORTRA 400 must read the same "
         "in E-4050's 2010 and 2016 editions, ULTRA MAX 400 the same in E-7019 "
         "(2007) and E-7023 (2016), GOLD 200 the same in E-7022's 2007 "
         "two-column and 2022 one-column layouts, every pushed aim must RISE "
         "with the push, and E-190 (2003)'s PORTRA 800 table must keep showing "
         "BOTH of its defects -- an aim that falls as the film is pushed, and "
         "an EI 800 forehead pair copied verbatim from the 160NC/400NC column "
         "four inches above it. ⚠ That last assertion is inverted on purpose: "
         "it fails if the broken table ever starts agreeing, because that "
         "would mean the reader moved and not the document",),
        # ⚠ NOT A DOCUMENT AUDIT -- a CODE audit, and it belongs in this stage
        # anyway. Its "source" is the GENERATED header rather than a PDF, and what
        # it re-derives is that the Python reference law and the emitted C++ law
        # still agree. Added 2026-08-18 (C1b): that convention change had to be
        # made twice, in two languages, and the previous cross-check was a manual
        # one-off from a finished session -- i.e. it guarded nothing.
        ("cpp_parity.py",
         ["--assert", "--root", str(root)],
         HERE / "film_profiles.hpp",
         "Python grain_sigma() vs the generated FilmGrainSigma(): 4770 probes "
         "over 159 stocks x 3 channels x 10 densities, including net 1.0 and "
         "absolute 1.0 (equal only for an unmasked stock), with a coverage "
         "assertion that all 11 measured shapes really differ from the legacy "
         "law. SINCE 2026-08-23 (C8) also the RECIPROCITY law: "
         "film_sim.reciprocity_log_shift() against the plugin's own "
         "AlgoReciprocityLogShift, 159 stocks x 12 exposure times from 1e-5 s "
         "to 3600 s -- the inertness of exposure_time_s = 0, the held-flat ends "
         "outside every measured table, and the CC-filter chromatic branch. "
         "That third family SKIPS when the plugin tree is not on disk"),
        # ⚠ NOT A DOCUMENT AUDIT EITHER, and it probes the PLUGIN'S OWN C++ rather
        # than generated code -- the only audit that does. Added 2026-08-20: the
        # two DIR-coupler stages are the largest COLOUR effect in the chain
        # (disabling them moves Velvia's saturated patches by up to 143/255) and
        # they exist twice, in two languages, with nothing comparing them.
        # cpp_parity covers the grain and MTF laws only.
        ("interimage_parity.py",
         ["--root", str(root), "--assert"],
         root / "Algo_08_Sim.cpp",
         "Python apply_interimage()/apply_dir_couplers() vs the plugin's own "
         "AlgoStage08b_Interimage()/AlgoStage09_DirCoupler(): 5 stocks covering "
         "both interimage mechanisms plus a monochrome control, flat and ramp "
         "fields, at two pixel scales. Reads sizeof(AlgoType) from the compiled "
         "probe and picks its tolerance from it, so the switchable double/float "
         "typedef stays switchable"),
        # ⚠ THE THIRD CODE AUDIT, ADDED 2026-08-29, AND THE ONE THAT CAUGHT A
        # DIVERGENCE THAT WAS ALREADY SHIPPING. Algo_07_Sim.cpp derives the
        # monochrome collapse weights from the traced pan curve
        # unconditionally; film_sim gated the identical derivation behind
        # RenderSettings.spectral_mono, which defaulted to False. For the 24
        # stocks with a curve the plugin and the reference renderer therefore
        # produced different B&W images, worst case KODAK_PLUS_X_125 at blue
        # 0.110 against 0.502. Neither cpp_parity (grain, MTF, reciprocity) nor
        # interimage_parity (the DIR couplers) looks at stage 7.
        # ⚠ --allow-guard-gap WAS REMOVED 2026-08-30, NOT LEFT IN AS A SAFETY
        # NET. It accepted one known open defect: the gamut-reach guard existed
        # only in Python, so KONICA_INFRARED_750 derived to a blue-dominant
        # (0.1611, 0.1931, 0.6458) in the plugin against its authored and
        # correct red-dominant (0.55, 0.15, 0.30). Queue C40 closed that by
        # porting the two tests into AlgoSpectralMonoWeights(). Running WITHOUT
        # the flag is what keeps it closed: a re-opened gap now fails the build
        # instead of printing a line somebody has learned to skip.
        ("spectral_mono_parity.py",
         ["--algodir", str(root), "--assert"],
         root / "AlgoSpectralSensitivity.cpp",
         "film_sim.spectral_monochrome_weights() against the plugin's own "
         "AlgoSpectralMonoWeights(), all 68 monochrome stocks walked out of "
         "the real database, to 1e-9 -- including the gamut-reach guard, which "
         "must refuse and fall back on the same stocks in both engines"),
        # ⚠ NOT A DOCUMENT AUDIT EITHER: it re-derives the spectral_weights
        # PROVENANCE from the live database. Added 2026-08-29 after 48 colour
        # stocks were found labelled status='derived', 'integrated from the
        # traced log-sensitivity curves', while every one of them still stored
        # the (0.30, 0.59, 0.11) dataclass default -- Rec.601 luma, integrated
        # from nothing. _PARAM_SOURCES_DERIVED's own header says "REGENERATE,
        # do not hand-edit: the rules live in the task EM-A6 generator", and
        # that generator is not in the repository, so for this parameter the
        # rule now lives here and is checked.
        ("spectral_weight_provenance.py",
         ["--assert"],
         HERE / "film_profiles.py",
         "the spectral_weights ParamSource record of all 161 profiles against "
         "the rule that produces it: derived where a monochrome stock's curve "
         "is integrated at run time, refused-with-cause where the guard "
         "declines, inert where the stock is colour and the field is never "
         "read"),
        ("doc_consistency.py",
         ["--root", str(root / "PYTHON" / "profile_generator"), "--assert"],
         root / "PYTHON" / "profile_generator" / "doc",
         "every COUNT asserted in the documentation against the live database "
         "-- added 2026-08-25 after an audit found four hardcoded counts in the "
         "report generator wrong by up to 2.3x, a struct described as unread two "
         "days after it was wired, and queue rows still saying 'no profile' for "
         "stocks added the day before. A pattern that stops matching FAILS, "
         "because an unmatched pattern silently stops checking"),
        ("plot_inventory.py",
         ["--root", str(root / "PDF" / "PROFILES"), "--assert"],
         root / "PDF" / "PROFILES",
         "the corpus plot inventory: 191 vector dye-density pages (57 under the narrow title pattern), 199 MTF, "
         "101 granularity, 294 characteristic-curve, and the classifier's "
         "three known-answer pages"),
    )

#: verify.py exits 1 whenever ANY check fails, and two checks fail BY DESIGN --
#: the owner's instruction was explicit: "don't try to fix them". So the exit
#: code alone is a useless gate. Compare the FAIL SET against this baseline
#: instead: a new failure fails the build, and a baseline entry that starts
#: passing also reports, so the baseline is shrunk deliberately rather than by
#: accident.
#: ⚠ WAS TWO ENTRIES UNTIL 2026-08-20, AND ONE OF THEM WAS NOT A DATA PROBLEM.
#: "neighbour pairs couple harder than the far red-blue pair" asserted a
#: PER-DISTANCE interimage asymmetry that the database deliberately does not
#: store, because the evidence (US4725529A Table 1 -- inhibitor in the developer,
#: three separate single-layer coatings, no layer stack, asymmetry persists) says
#: the asymmetry is per RECEIVER. So it was unpassable by construction: a stale
#: assertion parked in the baseline as "known, leave alone", which is how it
#: survived. Replaced in verify.py with the assertion the evidence supports, and
#: this mechanism is what reported the change rather than swallowing it.
VERIFY_BASELINE = {
    "saturation hierarchy is ordered clean -> impure dyes",
}


class Result:
    __slots__ = ("name", "state", "detail")

    def __init__(self, name, state, detail=""):
        self.name, self.state, self.detail = name, state, detail

    def __repr__(self):
        return f"{self.state:4} {self.name}  {self.detail}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


#: cpp_codegen stamps a UTC generation time into the three C++ files (never into
#: film_names.txt -- that file is required to stay pure data). So a byte compare
#: of a fresh run against the file on disk ALWAYS differs, on the stamp alone.
#: Reporting that as drift would make --check warn every single time, and a gate
#: that always warns is a gate that gets ignored. Compare content instead, with
#: the stamp lines removed, so a real difference is the only thing that shows.
_STAMP = ("// generated:", "Generated by cpp_codegen.py")


def content_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if any(m in line for m in _STAMP):
                continue
            h.update(line.encode("utf-8", "replace"))
    return h.hexdigest()


def run(argv, cwd=HERE, timeout=3600):
    """Run a child process, capturing both streams. Never uses a shell."""
    try:
        p = subprocess.run(argv, cwd=str(cwd), timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        return None, "", f"{exc}"
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout}s"
    return p.returncode, p.stdout.decode("utf-8", "replace"), \
        p.stderr.decode("utf-8", "replace")


def py(*args):
    """This interpreter, not whatever 'python' happens to resolve to."""
    return [sys.executable, *args]


# ---------------------------------------------------------------- stages -----

def stage_audit(opts) -> list:
    """Re-derive adopted numbers from the source documents."""
    out = []
    for script, argv, needs, guards in audits(opts.root):
        if not (HERE / script).is_file():
            out.append(Result(script, "SKIP", "script not present"))
            continue
        if not Path(needs).exists():
            out.append(Result(script, "SKIP",
                              f"source not present: {Path(needs).name}"))
            continue
        rc, so, se = run(py(script, *argv))
        if rc is None:
            out.append(Result(script, "SKIP", se))
        elif rc == 0:
            tail = [ln for ln in so.splitlines() if ln.startswith("[OK]")]
            out.append(Result(script, "OK",
                              tail[-1] if tail else f"reproduces {guards}"))
        else:
            bad = [ln for ln in (so + se).splitlines()
                   if ln.startswith(("[!]", "[FAIL]"))]
            out.append(Result(script, "FAIL",
                              "; ".join(bad[:3]) or f"exit {rc}"))
    if not out:
        out.append(Result("audit", "SKIP", "no audit scripts registered"))
    return out


def stage_verify(opts) -> list:
    """Run verify.py and compare the FAIL set against VERIFY_BASELINE."""
    rc, so, se = run(py("verify.py"))
    if rc is None:
        return [Result("verify.py", "FAIL", se)]
    fails = {ln[len("FAIL"):].strip().split("   ")[0].strip()
             for ln in so.splitlines() if ln.startswith("FAIL")}
    npass = sum(1 for ln in so.splitlines() if ln.startswith("PASS"))
    res = [Result("verify.py", "OK" if fails == VERIFY_BASELINE else "FAIL",
                  f"{npass} PASS / {len(fails)} FAIL")]
    for extra in sorted(fails - VERIFY_BASELINE):
        res.append(Result("  NEW FAILURE", "FAIL", extra))
    for gone in sorted(VERIFY_BASELINE - fails):
        res.append(Result("  baseline entry now PASSES", "WARN",
                          f"{gone}  -- remove it from VERIFY_BASELINE"))
    if not so.strip():
        res.append(Result("  verify.py produced no output", "FAIL",
                          se.splitlines()[-1] if se.strip() else f"exit {rc}"))
    return res


def stage_codegen(opts) -> list:
    """Regenerate the C++ tables; assert film_names.txt is cpp_codegen's."""
    if opts.check:
        tmp = Path(tempfile.mkdtemp(prefix="fs_codegen_"))
        try:
            rc, so, se = run(py("cpp_codegen.py", "-o", str(tmp)))
            if rc != 0:
                return [Result("cpp_codegen.py", "FAIL", se.strip()[-200:])]
            res = [Result("cpp_codegen.py", "OK", "generated into a temp dir")]
            for name in GENERATED:
                fresh, live = tmp / name, HERE / name
                if not live.is_file():
                    res.append(Result(f"  {name}", "FAIL", "missing on disk"))
                elif content_sha256(fresh) == content_sha256(live):
                    same_bytes = sha256(fresh) == sha256(live)
                    res.append(Result(f"  {name}", "OK", "up to date"
                                      if same_bytes
                                      else "up to date (differs only in the "
                                           "generation timestamp)"))
                else:
                    res.append(Result(f"  {name}", "WARN",
                                      "CONTENT DIFFERS from a fresh run -- "
                                      "regenerate (run without --check)"))
            return res
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # Snapshot film_names.txt BEFORE regenerating. Checking it afterwards is
    # vacuous: cpp_codegen has already rewritten it, so the check could never
    # fail and gave a false all-clear. What is worth knowing is whether the file
    # on disk was NOT cpp_codegen's before this run -- i.e. whether something
    # (the deprecated gen_film_names.py) had clobbered the list the effect panel
    # loads. That can only be seen from the pre-run state.
    names = HERE / "film_names.txt"
    before = sha256(names) if names.is_file() else None

    rc, so, se = run(py("cpp_codegen.py", "-o", "."))
    if rc != 0:
        return [Result("cpp_codegen.py", "FAIL", se.strip()[-200:])]
    res = [Result("cpp_codegen.py", "OK",
                  f"wrote {len(GENERATED)} artefacts")]
    missing = [n for n in GENERATED if not (HERE / n).is_file()]
    if missing:
        res.append(Result("  artefacts", "FAIL", f"not written: {missing}"))

    after = sha256(names) if names.is_file() else None
    if before is None:
        res.append(Result("  film_names.txt", "OK", "created"))
    elif before == after:
        res.append(Result("  film_names.txt owner", "OK",
                          "was already cpp_codegen's output"))
    else:
        res.append(Result("  film_names.txt owner", "WARN",
                          "the file on disk was NOT cpp_codegen's output and "
                          "has been CORRECTED -- the deprecated "
                          "gen_film_names.py had probably been run over it"))
    return res


def stage_sync(opts) -> list:
    """Keep the project-root copy of the generated C++ identical."""
    res = []
    # ⚠ SAY IT OUT LOUD IF THE ROOT LOOKS WRONG. See _default_root: a root that
    # is not the project root makes every line below a lie -- the copies it
    # reports on are not the ones anybody compiles.
    if not (opts.root / _ROOT_SENTINEL).is_dir():
        res.append(Result("project root", "WARN",
                          f"{opts.root} has no {_ROOT_SENTINEL}/ -- this may not "
                          f"be the project root; the C++ copies below would go "
                          f"somewhere nothing compiles. Set FILMSIM_ROOT or pass "
                          f"--root."))
    for name in GENERATED:
        src, dst = HERE / name, opts.root / name
        if not src.is_file():
            res.append(Result(f"{name}", "FAIL", "not generated"))
            continue
        if not dst.exists():
            if opts.check:
                res.append(Result(f"{name}", "WARN",
                                  f"absent at {opts.root} -- would be created"))
                continue
            shutil.copy2(src, dst)
            res.append(Result(f"{name}", "OK", f"created at {opts.root}"))
            continue
        if sha256(src) == sha256(dst):
            res.append(Result(f"{name}", "OK", "both copies identical"))
        elif content_sha256(src) == content_sha256(dst):
            res.append(Result(f"{name}", "OK",
                              "same content, differs only in the timestamp"))
        elif opts.check:
            res.append(Result(f"{name}", "WARN",
                              "root copy DIFFERS -- run without --check"))
        else:
            shutil.copy2(src, dst)
            res.append(Result(f"{name}", "OK", "root copy refreshed"))
    return res


def stage_docs(opts) -> list:
    """Regenerate the reports that describe the database."""
    res = []
    if opts.check:
        tmp = Path(tempfile.mkdtemp(prefix="fs_docs_"))
        try:
            rc, so, se = run(py("gen_active_profiles.py",
                                "-o", str(tmp / "FilmActiveProfiles.md")))
            live = HERE / "doc" / "FilmActiveProfiles.md"
            if rc != 0:
                res.append(Result("gen_active_profiles.py", "FAIL",
                                  se.strip()[-200:]))
            elif not live.is_file():
                res.append(Result("FilmActiveProfiles.md", "FAIL", "missing"))
            else:
                same = sha256(tmp / "FilmActiveProfiles.md") == sha256(live)
                res.append(Result("FilmActiveProfiles.md",
                                  "OK" if same else "WARN",
                                  "up to date" if same else "stale"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        res.append(Result("gen_film_curves_md.py", "SKIP",
                          "no -o option, cannot run read-only"))
        return res

    for script, argv, produced in (
            ("gen_active_profiles.py", [], "doc/FilmActiveProfiles.md"),
            ("gen_film_curves_md.py", [], "doc/FilmCurves.md")):
        rc, so, se = run(py(script, *argv))
        if rc == 0:
            res.append(Result(script, "OK", produced))
        else:
            res.append(Result(script, "FAIL", (se or so).strip()[-200:]))
    res.extend(_progress_doc_check())
    return res


def _progress_doc_check() -> list:
    """Fail if doc/PROGRESS.md no longer states the live facts.

    WHY THIS IS A BUILD GATE AND NOT A GOOD INTENTION. The owner reads the
    markdown to see where the project stands, and the failure mode is silent: a
    document that was true last week reads exactly like one that is true now.

    ⚠ IT READS A MACHINE STAMP, NOT THE PROSE, and the first version of this
    check is why. Searching the body text for "v8" passed even after the status
    line was edited to say v7, because the string "v8" also occurs in a sentence
    about the struct layout. A gate that can be satisfied by an unrelated
    sentence is worse than no gate: it reports OK on a stale document. So
    PROGRESS.md carries one HTML-comment stamp line and this parses it:

        <!-- build-facts: schema=v8 stocks=155 names_md5=<32 hex> -->

    Formatting-independent, unambiguous, and trivially satisfiable -- update the
    stamp when the facts move. Fault-injected against all three fields.
    """
    doc = HERE / "doc" / "PROGRESS.md"
    if not doc.is_file():
        return [Result("doc/PROGRESS.md", "FAIL",
                       "missing -- the status board every task must update")]
    text = doc.read_text(encoding="utf-8", errors="replace")
    import re as _re
    m = _re.search(r"<!--\s*build-facts:\s*schema=v(\d+)\s+stocks=(\d+)\s+"
                   r"names_md5=([0-9a-fA-F]{32})\s*-->", text)
    if not m:
        return [Result("doc/PROGRESS.md", "FAIL",
                       "no build-facts stamp line (see _progress_doc_check)")]
    try:
        sys.path.insert(0, str(HERE))
        import film_profiles as _fp
        live_schema, live_stocks = _fp.SCHEMA_VERSION, len(_fp.FILM_PROFILES)
    except Exception as exc:                                # pragma: no cover
        return [Result("doc/PROGRESS.md", "SKIP", f"cannot import: {exc}")]
    names = HERE / "film_names.txt"
    live_md5 = (hashlib.md5(names.read_bytes()).hexdigest()
                if names.is_file() else "")
    bad = []
    if int(m.group(1)) != live_schema:
        bad.append(f"schema v{m.group(1)} != live v{live_schema}")
    if int(m.group(2)) != live_stocks:
        bad.append(f"stocks {m.group(2)} != live {live_stocks}")
    if live_md5 and m.group(3).lower() != live_md5:
        bad.append(f"names_md5 {m.group(3)[:8]}... != live {live_md5[:8]}...")
    if bad:
        return [Result("doc/PROGRESS.md", "FAIL", "STALE: " + "; ".join(bad))]
    return [Result("doc/PROGRESS.md", "OK",
                   f"stamp matches live: schema v{live_schema}, "
                   f"{live_stocks} stocks, names {live_md5[:8]}...")]


def stage_compile(opts) -> list:
    """Compile the generated table. Gate on exit code AND empty stderr."""
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        return [Result("compile", "SKIP",
                       "no g++/clang++ on PATH (set CXX to override)")]
    sources = [n for n in GENERATED if n.endswith(".cpp")]
    missing = [n for n in sources if not (HERE / n).is_file()]
    if missing:
        return [Result("compile", "SKIP", f"not generated: {missing[:3]}")]
    tmp = Path(tempfile.mkdtemp(prefix="fs_cc_"))
    try:
        res = []
        for n in sources:
            rc, so, se = run([cxx, "-std=c++14", "-Wall", "-Wextra",
                              "-c", str(HERE / n),
                              "-o", str(tmp / (n[:-4] + ".o"))], cwd=HERE)
            if rc is None:
                return [Result("compile", "SKIP", se)]
            noise = len(se.encode()) + len(so.encode())
            if rc != 0 or noise != 0:
                first = (se or so).strip().splitlines()
                res.append(Result(f"  {n}", "FAIL",
                                  f"exit {rc}, {noise} bytes: "
                                  f"{first[0][:140] if first else ''}"))
        if res:
            return res
        return [Result("compile", "OK",
                       f"{Path(cxx).name} -std=c++14 -Wall -Wextra on all "
                       f"{len(sources)} TUs: exit 0 AND zero bytes of output")]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ⚠ STAGE ORDER IS LOAD-BEARING AND WAS WRONG UNTIL 2026-08-24.
#
# It used to run audit first. Two of the audits (`cpp_parity`, `interimage_parity`)
# compile probes against the PLUGIN'S OWN C++ under `--root`, so on the first run
# after any schema change they compiled against the PREVIOUS schema and failed --
# a failure with nothing wrong in it, which passed on the next run and therefore
# taught everyone to re-run rather than to read. Worse, it hid a real one: the
# 2026-08-23 v11 bump reported `interimage_parity` "probe did not compile" while
# the actual problem was a stale root the sync stage had never corrected.
#
# Now: verify gates everything (it reads only the Python database, so it needs no
# artefacts), then codegen writes the C++, then sync places it, and only THEN do
# the audits compile against it. `docs` and `compile` are unchanged at the end.
#
# The one thing that must NOT move back above codegen/sync is `audit`.
STAGES = (
    ("verify",  stage_verify,  "verify.py, FAIL set compared to the baseline"),
    ("codegen", stage_codegen, "regenerate the four C++ artefacts"),
    ("sync",    stage_sync,    "keep the project-root C++ copy identical"),
    ("audit",   stage_audit,   "re-derive adopted numbers from source documents"),
    ("docs",    stage_docs,    "regenerate FilmActiveProfiles.md, FilmCurves.md"),
    ("compile", stage_compile, "g++ -std=c++14 -Wall -Wextra, gated strictly"),
)


def main() -> int:
    names = [n for n, _, _ in STAGES]
    ap = argparse.ArgumentParser(
        description="Regenerate and audit the film-profile database.",
        epilog="Stages run in the listed order; the order is load-bearing.")
    ap.add_argument("--only", action="append", metavar="STAGE", choices=names,
                    help="run only this stage (repeatable)")
    ap.add_argument("--skip", action="append", metavar="STAGE", choices=names,
                    help="skip this stage (repeatable)")
    ap.add_argument("--check", action="store_true",
                    help="READ-ONLY: audit and report drift, write nothing")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="project root holding PDF/ and the second C++ copy "
                         "(default: two levels above this script)")
    opts = ap.parse_args()
    opts.root = Path(opts.root).resolve()

    if opts.list:
        print("stages, in order:\n")
        for n, _, why in STAGES:
            print(f"  {n:8} {why}")
        print("\naudit scripts registered:\n")
        for script, _, needs, guards in audits(opts.root):
            have = "present" if Path(needs).exists() else "SOURCE MISSING"
            print(f"  {script:26} {guards}\n"
                  f"  {'':26} needs {Path(needs).name}  [{have}]")
        return 0

    chosen = [s for s in STAGES if (not opts.only or s[0] in opts.only)
              and (not opts.skip or s[0] not in opts.skip)]

    mode = "CHECK (read-only)" if opts.check else "BUILD"
    print(f"film-profile generator -- {mode}")
    print(f"  generator : {HERE}")
    print(f"  root      : {opts.root}")
    print(f"  python    : {sys.version.split()[0]}")

    failed = warned = 0
    for name, fn, why in chosen:
        print(f"\n=== {name} -- {why}")
        try:
            results = fn(opts)
        except Exception as exc:                       # a stage must not abort
            results = [Result(name, "FAIL", f"{type(exc).__name__}: {exc}")]
        for r in results:
            print(f"  [{r.state:4}] {r.name}"
                  + (f"  --  {r.detail}" if r.detail else ""))
            failed += r.state == "FAIL"
            warned += r.state == "WARN"

    print("\n" + "-" * 70)
    if failed:
        print(f"BUILD FAILED  --  {failed} failure(s), {warned} warning(s)")
        return 1
    print(f"OK  --  0 failures, {warned} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
