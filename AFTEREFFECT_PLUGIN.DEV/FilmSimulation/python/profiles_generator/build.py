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

import cpp_codegen

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
    f"film_profiles_data_{i:02d}.cpp"
    for i in range(1, cpp_codegen.N_DATA_SLOTS + 1))
# ⚠ THE SLOT COUNT IS READ FROM THE EMITTER, NOT REPEATED HERE. It was a literal
# 17 in this expression until 2026-09-02e -- a second copy of a constant that
# only ever changes together with the .vcxproj, and one that would have silently
# stopped syncing the newest slot file the moment they disagreed.

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
        # ---- queue G2, 2026-09-02 -----------------------------------------
        ("gevachrome_1968_raster.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "GEVAERT"
              / "Rens_vanBets1968Gevachr6.00.pdf",
         "J. E. Rens and K. Van Bets, «Gevachrome-Farbumkehrfilme für "
         "Farbfernsehen», KINO-TECHNIK 1968 Nr. 10, printed pp. 260-266 -- the "
         "four RASTER plot sets, the paper's text having been harvested by hand "
         "in G1. ⚠ EVERY PAGE IS ONE EMBEDDED JPEG at about 115 ppi with no "
         "curve paths and no tick text, and the sheet CURLS at its right edge, "
         "so Bild 1's abscissa decade width runs 174 px between 2 and 10 c/mm "
         "and about 99 px in the last stretch; the reader interpolates "
         "PIECEWISE between the nine printed gridlines rather than fitting one "
         "log scale, which is why an earlier pass that fitted one scale put "
         "100 c/mm off the panel and stopped. The ordinate does not curl -- all "
         "three panels independently return 176.5 px per decade -- which is "
         "what a sheet curling about a vertical axis does and is the reason to "
         "trust the abscissa anchors. ADOPTED: Bild 2a/2b spectral "
         "sensitisation for both types, Bild 4 image-dye absorption (ONE curve "
         "set captioned for both, so it is one measurement stored twice and "
         "verify.py asserts the two arrays are identical), and Bild 1a/b/c MTF "
         "in green / red / blue light. ⚠ THE MTF REPLACES [T3] CLASS ESTIMATES "
         "THAT WERE TWO TO THREE TIMES TOO HIGH -- measured 20.4/23.5/44.4 and "
         "15.8/20.3/35.9 c/mm against a stored 58/62/66 and 50/54/58 -- and "
         "what licenses that is three things the tracer was not told: blue "
         "comes out sharpest and red softest on both films, which is the "
         "printed Tab. I layer order at a ratio of 2.2 where the estimates had "
         "1.14; Typ 6.05, the faster film, comes out softer in every channel; "
         "and the adopted rolloff law fits all six curves at q 1.90-2.12, rms "
         "0.006-0.025. ⚠ L/mm IS CYCLES/mm: the text states the test object had "
         "a SINUSOIDAL density variation, which settles the unit question queue "
         "G6 raises for this paper. ⚠ BILDER 7a/7b ARE MEASURED AND NOT STORED: "
         "the interimage separation A - B on the cyan record is about 0.15 D at "
         "the foot, and CouplerSpec.strength has no published calibration "
         "against a measured Delta-D, so converting one into the other would be "
         "inventing the conversion rather than reading it"),
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
        ("agfa_p16c.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "AGFA" / "agfa_film_chem.pdf",
         "Agfa «Technical Data P-16-C» (08/1999), the processing companion "
         "`agfa_films.pdf` p11 names in its last line -- and which sat in the "
         "corpus under a filename that reads like a chemistry catalogue with "
         "nothing connecting it to the film sheet. Parses TEXT, not curves: 64 "
         "printed developing-time cells giving the time to reach gamma 0.55 / "
         "0.65 / 0.75 for each AGFAPAN film in each of six developers, for "
         "drum and small tank, plus 15 push rows. ⚠ It SUPERSEDES the traced "
         "gamma-time panel for the same three films -- same physics, printed "
         "instead of digitised -- and the two agree where they overlap, which "
         "is the check that makes the supersession safe rather than a swap. "
         "⚠ The audit's real work is the MONOTONICITY test: a longer "
         "development cannot give less contrast, so every (developer, method, "
         "film) triple must ascend with gamma. P-16-C passes on all of them; "
         "the 2004 handbook's RODINAL table fails, which is how a typesetting "
         "fault was told from a product revision. It also asserts the "
         "cross-document row RODINAL 1+25 / small tank / gamma 0.65 = 6 / 8 / "
         "7 min, which agfa_films.pdf p11 prints independently. ⚠ AND IT "
         "RECORDS A NEGATIVE RESULT: this document was read specifically to "
         "see whether it could fill `grain.clump_um*` or the sigma(D) shape, "
         "and it cannot -- no granularity plot, no aperture series, no Wiener "
         "spectrum. Those cells stay estimated for a checked reason"),
        ("agfa_1998_curves.py",
         ["--root", str(root)],
         root / "PDF" / "PROFILES" / "AGFA" / "agfa_films.pdf",
         "Agfa «Technical Data PF» (1st edition, 09/1998) read as VECTOR -- a "
         "DIFFERENT DOCUMENT from the F-PF-E4 sheet the audit above reads, "
         "though NotFound.md row 5 and queue G6 both recorded the four Agfa "
         "candidates as one publication. The byte-identical pair is real; "
         "agfa_films.pdf (md5 edb3dd17...) is not part of it, and it is the "
         "ONLY document in the corpus that plots AGFACOLOR ULTRA 50 or the "
         "AGFACHROME RSX II line. Reads all TWELVE film columns x four panels: "
         "spectral sensitivity, spectral density, sharpness and characteristic "
         "curves, plus the APX gamma-time families and SCALA's five push/pull "
         "steps. ⚠ Three things this reader had to do differently from the "
         "2004 one, each of which produced plausible output when done the 2004 "
         "way: the panel is calibrated from its own printed LABELS rather than "
         "a frame rect (the frame is an inner grid box a decade narrower than "
         "the plot, and frame containment returned 'no curve' on six of twelve "
         "columns); the label fit rejects outliers iteratively (the ordinate's "
         "single-glyph '0' bridges the 4 pt clustering gap to the abscissa's "
         "'-4.0' and dragged in 2.22 D of residual on a 3 D axis); and data is "
         "separated from furniture by SHAPE, not stroke width, because the "
         "Portrait spectral panel draws its curves at 0.503 pt -- THINNER than "
         "one of its own frames. Reversal records are fitted on a NEGATED "
         "abscissa as ToneCurve requires; fitted the sheet's way they return "
         "dmin 3.0 and gamma 2.0-2.5, in range for a slide film and wholly "
         "wrong. ⚠ Reports f50 and REFUSES to adopt it (queue G6) while "
         "adopting the overshoot, which is unit-free -- but files the evidence "
         "G6 asked for: this sheet prints an MTF and a resolving power for the "
         "same film on the same page, and f50/RP runs 0.19-0.52 with a median "
         "of 0.30 against Tani's predicted 0.5. ⚠ CORRECTED 2026-09-01b -- THIS "
         "AUDIT USED TO CONCLUDE \"so reading the axis as HALF-cycles would move "
         "it further from the relation, not closer\", WHICH OVERSTATED WHAT THE "
         "RATIO SHOWS. It is true of the one hypothesis it tested (the MTF "
         "ABSCISSA half-cycles, the resolving-power TABLE cycles -> 0.15) and it "
         "addressed neither of the other two: BOTH half-cycles leaves the ratio "
         "unchanged at 0.30, because it is a ratio of two quantities in the same "
         "unit, and the TABLE half-cycles with the abscissa in cycles gives 0.60, "
         "which is CLOSER to 0.5 than 0.30 is. The scale test in "
         "`agfa_2003_sheet.py` is what actually bounds it -- Agfa's "
         "RP x sqrt(RMS) invariant sits with every other maker's, so the table "
         "is on the line-pair scale and is kept as printed"),
        ("agfa_2003_curves.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "AGFA" / "agfa-aERRKF-Datenblatt_F_PF_D4.pdf",
         "Pages 8 and 9 of the 4th-edition Agfa sheet -- the six film columns "
         "`agfa_2004_curves.py` never read, because it stops at p7. RSX II 50 "
         "/ 100 / 200 as reversal, APX 100 / 400 as monochrome, SCALA 200x as "
         "B&W reversal. ⚠ ITS REAL WORK IS THE CROSS-EDITION COMPARISON, and "
         "the answer it returns is NEGATIVE IN THE USEFUL SENSE: the printed "
         "resolving power of the RSX II line was revised upward between 1998 "
         "and 2003 (125/125/110 -> 135/130/120 lines/mm) but the CURVES DID "
         "NOT MOVE. Once each edition's label offset is removed the two "
         "editions agree to 0.001-0.006 D rms on every density record and "
         "0.001-0.004 lg on every spectral one, because Agfa reused the "
         "identical artwork -- so the revision is a later measurement of an "
         "unchanged emulsion and a 2003 number may sit beside a 1998-traced "
         "curve. ⚠ THAT DE-BIAS IS ITSELF A FINDING. A text box's centre is "
         "not a digit's optical centre, and the two editions set their axis "
         "labels at different point sizes (7.69 pt box against 6.01 pt), so "
         "the error does not cancel between documents: measured against the "
         "RSX II 100 panel's own 0-to-4.0 axis rectangle the 1998 fit reads "
         "0.020 D LOW and the 2003 fit 0.015 D HIGH, and that 0.035 D is "
         "essentially the whole 0.038 D by which the editions appeared to "
         "disagree. ⚠ It also asserts the GERMAN TWIN relationship rather "
         "than assuming it -- pp7-8 must be byte-identical drawings in both "
         "files -- and turns Agfa's own typo into a check: the RSX II 50 "
         "spectral ordinate is printed «- 0.1» where the uniform 31.80 pt "
         "tick pitch and both sibling columns require -1.0, so that label is "
         "vetoed and the fit made without it is required to PREDICT -1.000 "
         "there"),
        ("agfa_2003_sheet.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "AGFA" / "agfa-aERRKF-Datenblatt_F_PF_D4.pdf",
         "Every PRINTED TABLE on the 4th-edition Agfa sheet, in BOTH "
         "languages, against the 1st edition's. ⚠ The English file has been "
         "in the corpus since 2026-08-29 and only its curves were ever read: "
         "not one of its tables had been harvested, although they carry the "
         "resolving power at two contrasts, the layer thickness, the base "
         "thickness per format, the DX and negative codes, the development-"
         "time matrices at 18/20/22/24 C and the exposure index per "
         "developer for all ten films. Reads all ten spec blocks in English "
         "and German and requires every numeric cell to agree across the two "
         "typesettings; keeps the measurement conditions in Agfa's own words "
         "(«Bezug: Energiegleiches Spektrum», «Diffuse Dichte 1,0; 48 µm "
         "Meßblende», «Densitometrie: Status A bzw. Status M»). ⚠ THE TWO "
         "ASSERTIONS THAT CARRY THE WEIGHT ARE A MATCHED PAIR: APX 100's "
         "processing tables must be IDENTICAL across the editions in all 7 "
         "comparable rows, and APX 400's must DIFFER in all 7. That is what "
         "turns «Neue Generation (ab 2003)» from a footnote into evidence, "
         "and it is why AGFA_APX_400 is left as the pre-2003 film. It also "
         "found the Optima 200 provenance defect -- the database stores RMS "
         "4.3 citing the 1998 sheet, which prints 4.5 in that column"),
        ("agfa_scala_sheet.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "AGFA" / "agfa_scala.pdf",
         "«AGFA SCALA 200x PROFESSIONAL -- Technical Data», F-SW12-E6, 6th "
         "edition 08/2000: the film's OWN four-page sheet, published between "
         "the two range sheets and new to the corpus. Reproduces all 17 "
         "stated values and checks them against the stored profile. What only "
         "this document gives: exposure latitude as a speed-dependent NUMBER "
         "(+-1/2 stop at ISO 200-1600, +-1 stop at ISO 100), the granularity "
         "viewing condition (\"equivalent to a 12-fold magnification\", and "
         "\"only in SCALA process\"), the five-layer emulsion design, the "
         "film base BY STANDARD -- safety film (acetyl cellulose) to DIN "
         "15551 -- which is the only per-film base material statement in the "
         "whole Agfa set, and the anti-halation construction in words. ⚠ ITS "
         "\"Total thickness: 12 um\" IS NOT THE RANGE SHEETS' "
         "«Schichtdicke 7 um» and the audit says so: 7 um is the emulsion "
         "layer, 12 um the whole coating including the retouchable gelatine "
         "backing. Two quantities, not a conflict, and not averaged. ⚠ AND A "
         "RECORDED NEGATIVE: no granularity plot, no aperture series, no "
         "Wiener spectrum, no gamma-time family, so this sheet cannot fill "
         "sigma(D) or clump size for SCALA either"),
        ("bbc_t101_2.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "RETRO" / "BBC Photographic film grain. 1964-04.pdf",
         "BBC Research Report T-101/2 (1964/4), K. Hacking -- the sequel to a "
         "T-101 this corpus already cites on three stocks, and the document "
         "that prints the whole measurement set in one table. Reads Table 1's "
         "four measured grain Wiener spectra (Ilford Pan F 0.10, Kodak Plus-X "
         "0.14, Kodak Tri-X 0.555, Ilford H.P.S. 0.62 square microns) and "
         "TRACES Fig. 8 from the page raster to check them. ⚠ THE TRACE IS NOT "
         "DECORATION: it reproduces the printed means to 0.1-0.7 %, and it "
         "measures the flatness the closed-form conversion depends on -- "
         "W(25.4 c/mm)/W(0) is 0.978-0.999, so sigma^2 = W/area holds by "
         "Parseval and no numerical integral is needed. ⚠ A WIENER SPECTRUM IS "
         "NOT AN rms GRANULARITY and this audit is where that stops being "
         "skated over: the BBC figures are at D 0.48 and gamma 1.0, not at the "
         "net density 1.0 and gamma 0.65 the project's rms convention means, "
         "and the three-step chain (Parseval, sqrt(gamma) from the report's own "
         "section 5.2, D^0.4 from Higgins and Stultz) is asserted against a "
         "CONTROL -- Kodak's own published 17.0 for Tri-X against the chain's "
         "18.9, 11 %. Plus-X and H.P.S. are adopted on that licence at -4.9 % "
         "and +5.3 %; ⚠ ILFORD PAN F IS REFUSED at +61 %, because its ASA 16 "
         "doubles under the table's own pre-revision footnote to about 32 "
         "against a profile rated EI 50, which is Pan F PLUS -- one trade name, "
         "two products, the same trap the corpus documents for TRI-X 5223"),
        ("flueckiger_2018.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "RETRO"
              / "flueckigeretal_investigationfilmmaterialscannerinteraction_2018_v_1-1b.pdf",
         "Flueckiger, Pfluger, Trumpy, Croci, Aydin and Smolic, «Investigation "
         "of Film Material-Scanner Interaction», University of Zurich / "
         "DIASTOR, v1.1 2018, 88 pages. ⚠ MOST OF IT IS A SCANNER STUDY AND "
         "IRRELEVANT HERE; four figures are not. §2.8.2 Fig. 16 gives the "
         "analytical densities of the THREE-STRIP TECHNICOLOR transfer dyes, "
         "measured on a 1949 SAMSON AND DELILAH print with a SHIMADZU UV-1800 "
         "and separated by the Ohta PCA method -- the FIRST measurement "
         "TECHNICOLOR_THREE_STRIP has ever carried, it being one of the nine "
         "stocks NotFound.md lists as having no source of any kind. ⚠ Figure "
         "16's ordinate has no scale, ticks or label, so only the SHAPE is "
         "taken, peak-normalised with the axis assumed to be zero; what "
         "validates it is the peak list printed in the running text and never "
         "seen by the trace -- 460 / 540 / 660 / 720 nm, returned exactly. "
         "§2.8.3 Figs. 21 and 22 give the Dufaycolor réseau: the transmittance "
         "of the three filter elements at 16 wavelengths, and the integral "
         "transmittance of the same sample on a bench spectrophotometer. "
         "⚠ THE CHECK THIS READER RESTS ON is that the two figures are "
         "separately calibrated and separately traced, and recomputing the "
         "report's own equation (7) -- 0.28 B + 0.32 G + 0.40 R -- from Fig. "
         "21 reproduces Fig. 22's markers to rms 0.29 transmittance points "
         "with no free parameter; the same inversion recovers the two markers "
         "Fig. 21 occludes. ⚠ FIGURE 21'S CAPTION SAYS «absorbance» AND THE "
         "FIGURE'S OWN ORDINATE SAYS «TRANSMITTANCE %» -- the figure is right. "
         "§4.1 Fig. 61 and Table 3 give the MEASURED MTF of eight film "
         "scanners at 10-40 lp/mm plus their sampling resolutions, checked "
         "against Fig. 62 through the report's equation (9); ⚠ THAT IS "
         "SCANNER DATA AND IS DELIBERATELY WRITTEN TO NO FILM PROFILE -- it "
         "lives in doc/SCANNER_CHARACTERISTICS.md. ⚠ AND Fig. 15 IS NOT "
         "COUNTED AS EVIDENCE: it is the same Callier artwork as "
         "trumpy_callier_q.py's Fig. 5 (shared author, both after Streiffert "
         "1947), used only as a pipeline reproducibility check -- two PDFs, "
         "two scans, one drawing, rms 0.0053 Q"),
        # ---- queue N1, 2026-09-02 -----------------------------------------
        ("fuji_neopan_ss.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "FUJI" / "SS35.pdf",
         "FUJIFILM DATA SHEET \"NEOPAN SS (135)\", Ref. No. AF3-411E(N) "
         "(EIGI-99.3-HB4-8), four pages, supplied by the owner 2026-09-02 -- "
         "and it closed a question three other documents could not. "
         "EMULSION_KNOWLEDGE_BASE.md §23k.8 had recorded, the same day, that "
         "FUJI NEOPAN could not be profiled because three papers here measure "
         "its GRAIN and nothing measured its TONE SCALE. This sheet is the "
         "tone scale: §3 ISO 100/21°, §4 «Orthopanchromatic», §7 a full "
         "development matrix over three Fuji and twelve non-Fuji developers at "
         "five temperatures, §8 a spectral sensitivity curve, §9 a "
         "characteristic-curve family and §10 time-Gbar curves. ⚠ §9 IS "
         "SELF-CHECKING BECAUSE IT PRINTS THE AVERAGE GRADIENT ON EVERY CURVE "
         "-- 4 min 0.28, 6 min 0.37, 8 min 0.45, 10 min 0.53, 12 min 0.61 -- "
         "and nothing in the trace is told them. ⚠ TWO CALIBRATION TRAPS, both "
         "found the hard way: the abscissa FRAME is logH -4.0 and the leftmost "
         "PRINTED label is -3.0, so reading the frame as the first label "
         "shifts every exposure by a decade while leaving every density and "
         "slope untouched -- the Gbar check still passes and nothing "
         "complains; and the ordinate's LABEL CENTROIDS disagree with its "
         "GRIDLINES (152.1 against 158.7 px per 0.5 D) because the axis title "
         "contaminates the label band, the gridlines being right because they "
         "make one density unit 317.5 px against one exposure decade at 318.4, "
         "the 1:1 aspect a sensitometric plot is drawn at. Five curves are "
         "followed TOGETHER with per-track coasting, because they converge at "
         "the toe and a single-track follower swaps them there; ⚠ THE TWO "
         "SHALLOWEST ARE THEN REFUSED, reconstructing Gbar at 0.67 and 0.87 of "
         "the printed value against 0.99-1.11 for the other three, and the GAP "
         "between those groups is what the 12 %% gate reads. ADOPTED: "
         "FUJI_NEOPAN_SS, stock 172, appended at frozen id 171 so no ListBox "
         "index moves -- the 10 min curve (the drawn member nearest the "
         "sheet's own 9 1/2 min recommendation for Microfine at 20 C, EI 100) "
         "fitted to rms 0.0234 D over 709 columns, whose model Gbar over dlogH "
         "2.0 is 0.552 against a printed 0.53; the §8 spectral curve at 10 nm "
         "pitch, peak-normalised at 410 nm with no absolute level claimed, "
         "reproducing the orthopanchromatic signature §4 states in words "
         "(blue peak 410, trough 490, secondary red lobe 590, cut past 650); "
         "and §7's Microfine 10 min at 20 C as the ProcessingSpec. ⚠ dmin "
         "0.245 is the sheet's printed «Base Density» rule, not a fitted "
         "value, and THE SHOULDER IS NOT MEASURED -- the panel stops at D 1.82 "
         "with the curve still straight, so Dmax is pinned at a class 2.70 and "
         "refitting at 2.5 or 3.0 moves the rms by 0.0002 D. ⚠ AND THE GRAIN "
         "BLOCK IS A FLAGGED CLASS ESTIMATE, NOT A MEASUREMENT, FOR A REASON "
         "THAT IS A DATE. The sheet has no image-structure section at all (no "
         "rms, no resolving power, no MTF, no reciprocity; all four pages "
         "searched), and this corpus DOES hold four granularity measurements "
         "of a film called Neopan SS -- Ooue 1959 Part 2 Fig. 26, Ooue's 23_7 "
         "Fig. 7 and Takano 1969 Fig. 8 -- which measure the coating sold in "
         "1959-1969 where this sheet is dated 1999 by its own printer's code. "
         "One trade name, two products, forty years apart: the trap already on "
         "file for EASTMAN_5247 (1974 against 1983) and ILFORD PAN F against "
         "PAN F PLUS. Grain and f50 sit in the band AGFA_APX_100 and "
         "KODAK_PLUS_X_125 occupy, and are labelled estimates"),
        # ---- queue E5, 2026-09-02 -----------------------------------------
        ("sehlin_kennel_1985.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "KODAK"
         / "Sehlin_Kennel_etal_1983_ChoosingECN5247or52941.pdf",
         "R. Sehlin, G. Kennel et al., \"Choosing between EASTMAN Color "
         "Negative Films 5247 and 5294\", SMPTE Journal 94(7) 724-731, JULY "
         "1985 -- ⚠ the file name in this corpus says 1983 and is wrong. "
         "NotFound.md row 2 has asked since 2026-08-17 for a "
         "granularity-against-density curve on a named stock, and every one "
         "found since has been a Kodak VENDOR SHEET for a VISION-family film. "
         "This is a JOURNAL plate, and its Fig. 8 puts DENSITY and RMS "
         "GRANULARITY on ONE shared log-exposure abscissa for EASTMAN_5294_1983 "
         "-- the sigma(D) construction complete on one figure, no second "
         "document needed. ⚠ THE TWO ORDINATES ARE NOT THE SAME SCALE (218 px "
         "per density unit against 242 px per granularity decade) so each is "
         "calibrated on its own labels, and ⚠ THE TWO CURVES CROSS near D = 1.0, "
         "which no single-track follower survives: they are walked TOGETHER and "
         "the candidate pair assigned to the prediction pair by least cost, "
         "which works because they cross with opposite slopes. 735 of 799 "
         "columns survive the |dD/dlogE| >= 0.25 conditioning gate, giving toe "
         "1.571 @ D 0.44 / mid 1.000 / dmax 0.703 @ D 2.08 with an interior "
         "peak 1.664 @ D 0.53 -- inside the eleven vendor-sheet negatives' "
         "1.20-1.62 and 0.50-0.90, on a plate none of them came from, which "
         "validates the trace. ⚠ AND NOTHING IS STORED. Written to the "
         "profile, cpp_parity.py rejected it in the next build at 5.7e-01 "
         "against a 2e-05 tolerance, and chasing that established a "
         "convention nobody had written down: sigma_shape_*_at are PER-LAYER "
         "ANALYTICAL densities, evidenced by toe_at equalling the GREEN "
         "curve's dmin on every measured stock (5219 0.59 against 0.58, 5201 "
         "0.62 against 0.62, 5245 0.57 against 0.64). Fig. 8's ordinate is "
         "the film's plotted density and its traced toe at D 0.44 sits BELOW "
         "5294's green dmin of 0.68 and far below its blue 1.09 -- the whole "
         "shape under the layer's own dmin. ⚠ WITHDRAWN, NOT PATCHED: "
         "re-anchoring needs a correspondence the paper does not print, and "
         "SHAPE AND SPACE ARE AS SEPARATE AS SHAPE AND LEVEL. ⚠ THE LEVEL IS "
         "REFUSED TOO: the ordinate is labelled \"RMS Granularity\" with no "
         "unit, no aperture and no densitometry. ⚠ AND SO IS FIG. 12. Its MTF "
         "for 5247 crosses 50 %% at 45-58 c/mm against a stored ESTIMATE of "
         "24/28/33, i.e. 1.6x to 2.1x, and is still not adopted: the running "
         "text calls it \"the SYSTEM modulation transfer function\" -- which "
         "carries the printer and the lens -- where the panel says \"5247 "
         "Film\", and the curve DOES NOT OVERSHOOT, staying at 100 %% to 18 "
         "c/mm where every colour-negative MTF traced from a vendor sheet here "
         "rises above 100 %% first. Three refusals on one document, each with "
         "its own cause"),
        # ---- queue TK1-TK5, 2026-09-02 --------------------------------------
        ("takano_1969_granularity.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "RETRO" / "JAPAN" / "23_13.pdf",
         "Kiyoshi Takano (高野潔), «写真フィルムの粒状性» / \"Granularity of "
         "Photographic Film\", テレビジョン / J. Inst. Telev. Engrs. Japan "
         "23(1) 13-23 (1969). A review like Ooue's, so none of its samples is "
         "a stock here, and FIVE ITEMS ARE STILL HARVESTED -- two of which the "
         "corpus was carrying as assumptions. Fig. 8 plots SELWYN GRANULARITY "
         "AGAINST SCANNING APERTURE for a colour negative and Neopan-SS, and "
         "⚠ IT IS THE FIRST MEASUREMENT THIS PROJECT HAS EVER CHECKED ITS "
         "APERTURE TERM AGAINST: film_sim.grain_reference_energy integrates "
         "(h*a)^2 with a Gaussian aperture of sigma = size/4, and that law, "
         "unchanged, with nothing tuned but one overall constant and "
         "clump_um, reproduces both traces to rms 0.007-0.020 in G over a "
         "0.2-1.04 range. Selwyn's constant is supposed to be "
         "aperture-independent; both curves saturate, which is why rms "
         "granularity replaced it. ⚠ WHAT THE FIT DOES NOT DETERMINE IS THE "
         "SIZE: across the corpus's clump_gain range 0.30-1.50 the fitted "
         "clump_um moves 6.20 -> 2.38 um while the residual moves only 0.020 "
         "-> 0.007 G. Fig. 13 measures the OPTICAL AUTOCORRELATION of two "
         "more named samples (Neopan-SSS D 2.0 in Minidol, cine positive D "
         "1.7 in D-16), half-widths 1.33 and 0.65 um, i.e. clump_um 1.77 and "
         "0.87 um -- and ⚠ NEITHER GOES NEGATIVE where Ooue's Fig. 24 does, a "
         "disagreement between two instruments that is left standing rather "
         "than resolved in the engine's favour. ⚠ TOGETHER THE TWO PAPERS NOW "
         "MAKE THE CENSUS QUEUE C45 WAS MISSING: every direct measurement of "
         "grain correlation length on file is 0.87, 1.77, 2.46, 3.22, 4.64 "
         "um, median 2.46, against 171 stored clump_um_g values with median "
         "13.0 -- THE STORED SCALE IS 5.3x EVERY MEASUREMENT IN THE CORPUS, "
         "and it is NOT changed here because clump_um moves a pixel on 168 "
         "stocks and C45 owns that decision. Fig. 9 traces sigma_D against "
         "integral colour density per layer and turns over -- a FOURTH "
         "independent confirmation of the 2026-08-17 correction to GrainSpec's "
         "docstring, on a Japanese colour negative rather than a Kodak sheet "
         "-- while ⚠ DISAGREEING ON WHERE: it peaks at D 1.04 at 1.00x where "
         "all eleven measured colour negatives here peak at D 0.65-0.80 at "
         "1.20-1.62x, which is what sigma_shape_measured already refuses to "
         "generalise. ⚠ AND TWO PRINTED EQUATIONS ARE THE PART THAT TOUCHES "
         "CODE. eq (2) gives sigma(D) from sigma(T) TO FOURTH ORDER and is "
         "exactly the correction that withdrew sigma_D = 0.648*D^0.665 -- BBC "
         "T-101 Fig. 26 runs sigma(T)/T from 0.39 to 1.64 where the "
         "first-order form the corpus was using is 1.3 % to 31 % low. ADOPTED "
         "as film_sim.sigma_density_from_transmittance plus its Newton "
         "inverse, ⚠ INERT: no render path calls either. eq (13) gives the "
         "print chain F_pr = F_pos + F_neg*R_pr^2*gamma^2 with R_pr the "
         "response of the printing optics AND the positive film; the engine "
         "satisfies all three terms by construction, and ⚠ DEPARTS IN ONE "
         "PLACE -- stage 14 band-limits the print stock's OWN grain by that "
         "same transfer, which eq (13) does not and which the duplication "
         "chain in the same function explicitly avoids. Recorded, not changed: "
         "it is correct whenever scanner_f50 is set and it moves a pixel on "
         "every print render"),
        # ---- owner addendum, 2026-09-03: the PS&E 1957-61 batch -------------
        ("pse_jones_1958.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "RETRO" /
         "Photographic Science and Engineering",
         "R. Clark Jones, «On the Quantum Efficiency of Photographic "
         "Negatives», PHOTOGRAPHIC SCIENCE AND ENGINEERING 2(2) 57-65, August "
         "1958. ⚠ THE FILE ON THE OWNER'S MACHINE IS MIS-NAMED "
         "`sim_journal-of-imaging-science_1958-08_2_2.pdf`; the Journal of "
         "Imaging Science is the SAME journal's name from 1987 on, and this "
         "issue's running foot reads `P S & E, Vol. 2, 1958`. Cite PS&E. "
         "Table C is Eastman Kodak's own sigma at four densities x three "
         "apertures for Royal-X, Tri-X, Plus-X and Pan-X, coatings of "
         "FEBRUARY 1957, 2000 density readings behind each of the 40 values. "
         "⚠ TWO THINGS COME OUT OF IT. (1) THE APERTURE LAW IS DIRECTLY "
         "TESTED FOR THE FIRST TIME IN THIS CORPUS: Selwyn needs sigma "
         "proportional to 1/DIAMETER, i.e. sigma10 = 2sigma20 = 4sigma40, "
         "which is how Jones tabulates it, and over 24 pairs the mean ratio "
         "is 0.929 -- the paper predicts the sign and size itself, from "
         "diffraction and finite layer thickness. That confirms "
         "`grain_reference_energy`'s aperture term from a third source and a "
         "different decade, and is NOT a licence to model the 7 %, which is "
         "an instrument effect. (2) THE sigma(D) CLASS SHAPE FOR B&W "
         "NEGATIVES, which queue row F2b opened for: normalised at D 1.0 the "
         "four films agree to sd 0.042-0.058 across roughly ASA 32 to 1250. "
         "⚠ ADOPTED AS A CLASS SHAPE AND NOT PER STOCK -- Royal-X and Pan-X "
         "are absent from this database, and its Tri-X and Plus-X are a "
         "modern still and a 1999 cine coating, not the February 1957 films. "
         "⚠ AND STILL INERT: `sigma_measured_usable` gates on "
         "`sigma_shape_measured`, which means measured on THAT stock. The "
         "stored numbers are now true; against the legacy sqrt(D) law these "
         "stocks actually render on, the measurement says -48 % at D 0.07 and "
         "+17 % at D 1.40"),
        # ---- owner addendum, 2026-09-03b: the two new AGFA documents -------
        ("agfa_mpt_1937.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "AGFA" / "agfamotionpictur00ckin.pdf",
         "Agfa Motion Picture Topics, Agfa Ansco, volumes I-IV 1937-1940, an "
         "Internet Archive scan the owner supplied on 2026-09-03. ⚠ IT IS THE "
         "FIRST SOURCE IN THIS CORPUS FROM INSIDE THE 1929-1956 GAP that "
         "NotFound.md records. Two harvests and one refusal. (1) TRACED: page "
         "13's characteristic panel, three D vs log E curves for AGFA DIRECT "
         "DUPLICATING (descending -- a direct-reversal material, and the "
         "article says so), CINE POSITIVE 35 mm and CONVIRA PAPER. Straight-"
         "line slopes +1.88, +2.05 and -2.01. ⚠ NOT ADOPTED, and the reason "
         "is structural rather than a data problem: of 175 profiles this "
         "database holds NOT ONE print or positive stock -- its highest "
         "non-reversal gamma is 1.70, on an infrared camera film -- so these "
         "are the only measured positive/print gradations in the corpus and "
         "there is nothing for them to correct. (2) READ AS PRINTED NUMBERS: "
         "Goetz & Gould, «The Graininess of Photographic Emulsions» Part IV, "
         "Caltech on the Agfa Ansco Research Fund, March-April 1939. G "
         "against density for AGFA SUPERPAN at five densities, and a six-way "
         "class ladder at matched density. ⚠ THE SUPERPAN SERIES HAS AN "
         "INTERIOR MAXIMUM AND FALLS AT BOTH ENDS -- 0.62 / 0.81 / 1.00 / "
         "0.99 / 0.61 of its own peak -- WHICH CONTRADICTS the Jones 1958 "
         "class shape adopted the same day, which FLATTENS above D 1.0 at "
         "1.016 of its D 1.0 value. Both are measured; the module asserts the "
         "disagreement persists rather than reconciling it. ⚠ AND G IS NOT "
         "rms GRANULARITY: its definition is in Parts I-III, which are not in "
         "this volume, and the only scale statement is that G is 'multiplied "
         "by the factor 1000 to avoid the use of decimals' -- the same "
         "PRESENTATION convention this database uses, which says nothing "
         "about the measurand. "
         "⚠ (3) EXTENDED 2026-09-04, AND THE SECOND PASS FOUND WHAT THE FIRST "
         "ONE FILED WITHOUT READING. Pages 44-53 are «The New Agfacolor "
         "Process» by Prof. Dr. J. EGGERT, the process's own inventor; the "
         "first pass recorded it in ERA_FACTS as a provenance citation. It is "
         "a technical article and it prints construction figures. ADOPTED onto "
         "AGFACOLOR_NEU_1936 -- the first measured numbers that profile has "
         "ever held, against a provenance that until today asserted no "
         "photometric figure for the film existed anywhere: three emulsion "
         "layers 0.005 mm each separated by plain gelatine layers 0.002 mm "
         "(p48) giving 19 um coated, cross-checked against p53's independent "
         "statement that the tripack is 'only about' as thick as a normal "
         "one-layer film; and the coating order blue/green/red, STATED rather "
         "than assumed from convention, by where p48 puts the yellow filter "
         "layer. ⚠ RECORDED AND REFUSED: p53's sunshine exposure 'F:4.5 to "
         "F:5.6' for 16 mm gives ISO 2.5-3.9 at 16 fps or 4.0-6.1 at 24 fps "
         "and the article does not say which -- the stored EI 8 sits above "
         "BOTH ranges, so the analogy is 0.4 to 1.7 stops fast and the two "
         "readings agree on that while disagreeing with each other. A speed "
         "inferred from exposure advice is not a measured speed, so 8 stands "
         "and the conflict is asserted (method rule 4). ⚠ (4) THE 1937 SPEED "
         "CONVERSION TABLE, p38: Weston / H&D / Scheiner / DIN / relative "
         "sensitivity, 13 rows, the only period speed-conversion table in the "
         "corpus, stored because several stocks here convert a pre-1957 DIN or "
         "a Weston with a MODERN formula. It validates against the note "
         "printed beneath it, and doing so found exactly ONE departure in the "
         "whole table -- H&D 24 -> 50 where the doubling ladder wants 48 -- "
         "which is named in SPEED_TABLE_ROUNDINGS so a second departure still "
         "fails"),
        # ---- owner addendum, 2026-09-03d -----------------------------------
        ("smpte_1953_hanson_kisner.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "RETRO" /
         "sim_smpte-motion-imaging-journal_1953-12_61_6.pdf",
         "W. T. Hanson, Jr. and W. I. Kisner, «Improved Color Films for Color "
         "Motion-Picture Production», Journal of the SMPTE 61(6) pp 667-701, "
         "December 1953 -- the primary paper for EASTMAN COLOR NEGATIVE 5248, "
         "which this database carries as EASTMANCOLOR_5248_1953. ⚠ IT WAS "
         "ACQUIRED ON A RECOMMENDATION THAT WAS WRONG ABOUT ITS CONTENTS AND "
         "THE MODULE SAYS SO: it was fetched to fill 5248's empty spectral and "
         "dye_density fields and contains NEITHER -- no spectral sensitivity "
         "panel anywhere, and its two spectral figures are the DENSITOMETER "
         "FILTERS, not the film's dyes. ⚠ AND ALL FOUR OF ITS D-log E FIGURES "
         "HAVE AN UNLABELLED EXPOSURE AXIS -- the abscissa reads 'Log E' with "
         "no numbers, verified at 400 dpi on each -- so gamma, speed and "
         "latitude are not in this paper and `refuse_gamma()` raises rather "
         "than letting a later caller fit one anyway. ⚠ Fig. 4 is additionally "
         "the author's own 'idealized set of curves', not a measurement; Figs "
         "7, 8 and 9 (5382 print, 5216 separation, 5245 internegative) carry "
         "conditions with no such qualifier, and the distinction is stored per "
         "figure. WHAT IS TRACED: Fig. 3, the printing-density filter set -- "
         "blue min D 0.849 at 437 nm, green 1.320 at 542, red 1.252 at 644, "
         "each clipped at D 3.0 by its own flanks because a filter passband is "
         "a MINIMUM. That is the era's answer to the `M_reader` question the "
         "database answers for exactly one stock while 164 of 165 render "
         "through SCAN_DI. ⚠ Green's pin was first taken from contaminated "
         "data -- the printed words 'Green combination' sit under the "
         "minimum -- and the label filter caught it. Tables I and III "
         "transcribed. Nothing adopted",),
        # ---- owner addendum, 2026-09-03c -----------------------------------
        ("smpte_1954_5382.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "RETRO" /
         "sim_smpte-motion-imaging-journal_1954-11_63_5.pdf",
         "Lovick and White, «Factors in Applying Color Soundtrack "
         "Developers», Journal of the SMPTE 63(5) p.189, November 1954, "
         "Figure 2: 'Spectral density of dye deposits of Eastman Color Print "
         "Film, Type 5382'. Three dye spectra traced 400-1000 nm -- yellow "
         "0.876 at 458 nm, magenta 0.870 at 553, cyan 1.118 at 667. ⚠ THE "
         "FIRST SPECTRAL DYE SET FOR A PRINT FILM IN THIS CORPUS, and it runs "
         "to 1000 nm because the paper's whole argument is that dye alone "
         "gives no infrared density for an S-1 phototube -- a product sheet "
         "would have stopped at 700. ⚠ ADOPTED 2026-09-03d onto the print "
         "stock EASTMANCOLOR_5382_1953, 410-700 nm at 5 nm -- the first "
         "spectral dye set on a PrintStock in this database -- WITH THE "
         "METROLOGY CAVEAT CARRIED IN THE RECORD: the caption says 'dye "
         "deposits' with no concentration, no reference density, no status "
         "and no illuminant (1954 predates Status A/M), so shape and the "
         "three peaks' ratio are what it fixes and the level must never be "
         "rescaled to a stated density. The paper carries no characteristic "
         "curve, speed or granularity for 5382 at all. ⚠ THE MERGE COAST HAD "
         "TO BE WIDENED to 55 px against a 32-36 px merged run, because the "
         "INK merges well before the PREDICTIONS do: at 10, 20 and 30 px "
         "magenta died at 502 nm and never reached its printed 553 nm peak. "
         "⚠ AND THAT SAME 55 px THEN BROKE THE TWO LOW CROSSINGS, which is "
         "why the module now runs two extra single-crossing tail passes: the "
         "joint trace ended yellow at 526.7 nm still at 0.158 D (it had "
         "re-acquired on the drawn axis after the cyan crossing at ~537 nm), "
         "and it missed the magenta/cyan crossing at ~430 nm entirely, "
         "holding cyan at a frozen 0.103 and 0.049 across 410-455 nm. Twelve "
         "column readings taken off the raw page by code that shares nothing "
         "with the tracker now pin both repaired tails to 0.001 D"),
        # ---- owner addendum, 2026-09-02e -----------------------------------
        ("takano_1968_mottle.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "RETRO" / "JAPAN" / "31_209.pdf",
         "Masao TAKANO (高野正雄), Fuji Photo Film Research Laboratories "
         "Ashigara, «写真像の粒状性(第2報)» / \"Granularity of Photographic "
         "Image (II): Wiener Spectrum at a Constant Density Level with Various "
         "Exposure and Time of Development\", J. Soc. Phot. Sci. Japan 31(4) "
         "209-214 (1968). ⚠ NOT THE TAKANO ALREADY HELD: 23_13.pdf is Kiyoshi "
         "Takano's REVIEW, this is Masao Takano's ORIGINAL EXPERIMENT, a "
         "different author and a different journal. Byte-compared against all "
         "96 PDFs: no duplicate; figure-compared against §23i, §23j and §23k: "
         "no overlap. ⚠ ITS SUBJECT IS A VARIABLE THIS ENGINE DOES NOT HAVE. "
         "One unnamed ASA 100 B&W negative is brought to the SAME DENSITY two "
         "ways -- [VTD] by developing longer at fixed exposure, [VE] by "
         "exposing more at fixed development -- and the two grain patterns "
         "differ. Fig. 11 is the only panel whose ordinate is a LENGTH, and it "
         "is traced: Expected mottle size 3.98-6.81 um over four developers, "
         "with the D=0 envelopes running [VE] 5.22 -> 7.29 um and [VTD] 3.35 "
         "-> 4.37 um. ⚠ REACHING A DENSITY BY DEVELOPING LONGER GIVES A "
         "36-40 %% SMALLER CLUMP than reaching it by exposing more -- on the "
         "envelopes; on the density-0.5-1.5 markers the same ratio is 10-28 %%, "
         "mean 17 %%, and the paper's prose quotes only the first. Developer "
         "ordering, finest first: para-phenylenediamine < PQ < Monol < MQ. ⚠ "
         "AND THE NUMBER THAT MATTERS TO THIS DATABASE: mottle is stated to be "
         "5-8x the mean developed grain size, so 3.98-6.81 um of mottle implies "
         "0.50-1.36 um of grain -- which lands INSIDE BBC T-101's independently "
         "measured 0.59-1.43 um band, from a different maker, country and "
         "decade. Two documents, one answer, and the stored clump_um median of "
         "13.0 is outside both. ⚠ NOTHING IS WRITTEN TO A PROFILE: the film is "
         "unnamed, so this is class evidence and is stored as inert reference "
         "constants only. Figs. 3-8 and 10 are REFUSED for adoption -- their "
         "ordinate is an unlabelled instrument unit with no printed constant"),
        # ---- queue J1 + J2, 2026-09-02 ------------------------------------
        ("ooue_1959_granularity.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "RETRO" / "JAPAN" / "22_91.pdf",
         "Shingo Ooue, Fuji Photo Film Research Laboratory, «写真感光材料の粒状"
         "性» Parts 1 and 2, J. Soc. Phot. Sci. Japan 22(1) 38-47 and 22(2) "
         "91-99 (1959) -- the two companion papers EMULSION_KNOWLEDGE_BASE.md "
         "§23i named as the ones that would settle the mean-square / rms "
         "ambiguity on his third. They do, in the author's own words: Part 2 "
         "§4.2.2 is headed «濃度変化の標準偏差による方法», the method using the "
         "STANDARD DEVIATION of density variation, and every granularity in "
         "the paper is built from it. ⚠ FOUR ITEMS ARE HARVESTED. Part 2 "
         "Fig. 26 is a measured WIENER SPECTRUM on three NAMED samples with "
         "stated developer, time and density -- Neopan SS / Minidol at D 1.03 "
         "and D 0.45 and Process Plate / D-72 at D 0.44 -- and it says two "
         "things about this engine's grain model. First, fitting a generalised "
         "Gaussian to the falling limbs returns exponents 0.71 / 0.89 / 1.36, "
         "ALL BELOW THE 2 THE MODEL ASSUMES, with a pure Gaussian fitting "
         "three to six times worse: real grain spectra have FATTER "
         "high-frequency tails than exp(-(f/f_hi)^2), the same defect MTFSpec "
         "already records for the MTF tail. Second, and needing no "
         "calibration at all, the SAME FILM at two densities gives f_half 45.6 "
         "against 70.8 with developer and time held fixed -- the clump grows "
         "with density and GrainSpec carries one clump size per stock. Part 2 "
         "Fig. 24 measures the AUTOCORRELATION of the same quantity "
         "independently (Neopan S, D 1.04, half-width 3.48 um, i.e. clump_um "
         "4.65 um under the engine's own law) and ⚠ REFUTES THE SHAPE FROM THE "
         "OTHER SIDE: past about 12 um it goes NEGATIVE, an anti-correlated "
         "ring that neither a Gaussian autocorrelation nor Sayanagi's Poisson "
         "placement can produce. Part 1 Fig. 2 measures MEAN DEVELOPED GRAIN "
         "AREA against density and it FALLS -- 1.10 to 0.93 um^2 at 32 min, "
         "1.16 to 0.57 at 1 min, equivalent diameters 1.21 down to 0.85 um, "
         "the same direction BBC T-101 Table 3 measures from another "
         "laboratory and ⚠ BELOW THE 1.3 um FLOOR OF ALL 17 STORED "
         "emulsion.grain_um VALUES, which come from one third-party "
         "aggregator. ⚠ NOTHING IS WRITTEN TO ANY PROFILE: none of the samples "
         "is a stock in this file, Fig. 26's ordinate is an unlabelled «POWER "
         "LEVEL» and its abscissa says «LINES/mm» without defining a line, so "
         "shape is usable and level is not. And Part 2 §4.2.1 supplies the "
         "EMPIRICAL half of queue C45 -- measured Q on commercial materials "
         "shows no correlation with graininess, because two emulsions of "
         "different grain size are coated as two layers"),
        # ---- queue C43 + C44, both closed 2026-09-02 ----------------------
        ("sayanagi_callier.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "RETRO" / "JAPAN" / "23_20.pdf",
         "Kazuo Sayanagi, Canon Camera Co., «Callier Q Factor と粒状» / \"A "
         "Theory on Callier Q Factor and Granularity\", J. Soc. Phot. Sci. "
         "Japan 23(1) 20-24 (1959). He derives Q from grain optics: base "
         "transmittance Ib, a developed grain of FINITE transmittance Ig "
         "(finite because the electron microscope shows it filamentary), "
         "circular grains Poisson-distributed at coverage 𝔅, and Savelli's "
         "Poisson averages for the mean intensity and the mean amplitude. His "
         "(10) is Q_II = 2/(1+Ig^½). ⚠ IT CONTAINS NO DENSITY -- not D, not the "
         "coverage, not the grain radius -- so on BASE-SUBTRACTED density "
         "Sayanagi's Q is flat, and the toe collapse this project had recorded "
         "as a MODEL DEFECT since 2026-09-01 has no mechanism in the only "
         "theory that derives Q from first principles. With the base left in "
         "it collapses exactly as measured, and the reader FITS THAT BASE TO "
         "BOTH MEASURED CURVES INDEPENDENTLY: Trumpy/Streiffert Fig. 5 gives "
         "Db 0.045 (whole-curve rms 0.019 Q against the shipped no-base fit's "
         "0.156, toe error +0.49 -> -0.014 with one parameter), and Mees "
         "FIG. 179 needs Db 0.050 to reconcile its five gamma curves with the "
         "shared toe stroke. ⚠ TWO LABORATORIES, TWO DECADES, THE SAME BASE "
         "DENSITY, and it explains the one feature of FIG. 179 nothing else "
         "could -- why five emulsions of five different contrasts are drawn as "
         "ONE stroke below D 0.25. ⚠ SO C44 CLOSES WITH A NULL CODE CHANGE: "
         "the engine's argument is NET density, the base is already gone, and "
         "a toe term fitted to base-inclusive data would remove it twice and "
         "darken the shadows -- the region C44 was opened to protect. ⚠ WHAT "
         "IT DOES CHANGE IS C43: refitting Mees's five curves with the base "
         "modelled gives beta 1.491/1.495/1.729/1.822/1.828 RISING WITH GAMMA, "
         "so `callier_q` is no longer the undocumented class constant 1.3 but "
         "beta(gamma) = 1 + 0.9706 g/(g+0.2558) evaluated on each stock's own "
         "mid slope, 1.64-1.87 across the 68 monochrome stocks. Colour stays "
         "1.0. The form was chosen for its ENDPOINTS -- beta(0)=1 exactly and "
         "beta(inf)=1.971, just under Sayanagi's own ceiling of 2 -- not for "
         "its residual. ⚠ THE CEILING IS A REAL TEST AND IT PASSES: inverting "
         "Ig = (2/beta-1)^2 turns the five betas into grain transmittances of "
         "11.7 % falling to 0.9 % as development proceeds, which is his "
         "assumption (II) and was not fitted. ⚠ BBC T-101 Fig. 25's 2.00-2.34 "
         "at 0.0016 sr sits ABOVE the ceiling and is recorded as an open "
         "tension, not explained away"),
        ("trumpy_callier_q.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "RETRO"
              / "Optical_Detection_of_Dust_and_Scratches_on_Photogr.pdf",
         "G. Trumpy and R. Gschwind, ACM JOCCH 8(2) Art. 7 (2015), Fig. 5 -- "
         "Callier Q against DIFFUSE DENSITY for a typical silver film, redrawn "
         "after Streiffert 1947 (J. SMPTE 49(6), not in this corpus). ⚠ WHAT IT "
         "IS FOR IS NOT WHAT THE PAPER IS ABOUT: the paper is a dust-and-"
         "scratch detection method, and Fig. 5 is the SECOND measured Q(D) in "
         "the corpus after Mees FIG. 179. Digitised 573 points, axes calibrated "
         "on the printed ticks (the frame then reads D 0.000-2.005 and Q "
         "0.996-1.698 against a printed 0-2 and 1.0-1.7), traced BY COLUMN "
         "where the stroke is thin and BY ROW on the near-vertical toe, with "
         "edge-touching runs rejected. Peak Q 1.5568 at D 0.354. ⚠ IT LETS THE "
         "PROJECT'S OWN LAW BE FITTED TO A MEASUREMENT FOR THE FIRST TIME: "
         "Silberstein and Tuttle, as film_sim.callier_net implements it, fits "
         "these points to rms 0.0087 Q over D 0.3-2.0 with E 0.1471 and beta "
         "1.6746, and E and beta are jointly identifiable (holding E at 0.10 "
         "or 0.20 raises the rms 2.4x and 2.9x). ⚠ AND IT QUANTIFIES THE KNOWN "
         "TOE DEFECT the callier_net docstring already admits: the law's "
         "small-D limit is the CONSTANT E+(1-E)*beta = 1.575, against a "
         "measured 1.081 at D 0.05 -- +0.491 Q. Mees FIG. 179 independently "
         "reads 1.042 at D 0.055, so two laboratories two decades apart witness "
         "the same collapse. ⚠ NOTHING IS CHANGED: beta IS FilmProfile."
         "callier_q, the 55 B&W negative stocks all carry the class constant "
         "1.3, this curve says 1.675 and BBC T-101 Fig. 25 says 2.0-2.34 at "
         "0.0016 sr where Q -> beta -- a real disagreement on a field that "
         "moves a pixel, recorded and left for a decision"),
        ("jp_jps_1965_269.py",
         ["--root", str(root), "--assert"],
         root / "PDF" / "PROFILES" / "RETRO" / "1965.1_269.pdf",
         "大上進吾 and 高野正雄 (Fuji Photo Film Research Labs), Physical "
         "Society of Japan abstract 10p-A-2 (1965) p269 -- the only source in "
         "the corpus that indexes granularity by CRYSTAL SIZE rather than by "
         "product: five emulsions at AgX 0.3 / 0.4 / 0.5 / 1.5 / 1.8 um, one "
         "apparatus, one density (測定濃度 D_H = 0.5). One handwritten page "
         "whose OCR layer is useless, so both figures are read from the "
         "raster. ⚠ ITS ORDINATE IS 相対値 -- RELATIVE -- SO NO rms "
         "GRANULARITY CAN COME FROM IT AND NONE IS TAKEN; what is absolute is "
         "the abscissa, and the half-power frequency is a measured BANDWIDTH "
         "in cycles/mm, which is exactly what GrainSpec.clump_um sets through "
         "grain_shape. ⚠ THE CHECK IS THAT THE PAGE DRAWS F(20,0) TWICE, as "
         "Fig. 1's plateau and as Fig. 3's solid curve, in two hand-drawn "
         "figures with independent axes: they agree to 0.0-4.5 %. The x "
         "calibration comes from the decade ticks and is validated by the "
         "markers reproducing the PRINTED crystal sizes to 0.4-10 %. ⚠ SAMPLE "
         "B IS A BRACKET, NOT A POINT -- the two curves cross there and its "
         "markers are one blob -- and the markers are classified by WHICH "
         "TRACED CURVE THEY SIT ON, because 'the upper one' swaps sides at "
         "that crossing. The abstract does not state its reading aperture, so "
         "the reader BOUNDS it instead: the curves run past 869 c/mm with no "
         "transfer zero, hence a circular aperture under 1.4 um whose MTF^2 "
         "is still 0.945 at 108 c/mm. ⚠ NOTHING IS ADOPTED AND THAT IS THE "
         "RESULT: both fitted laws are too flat to invert (1 % in the "
         "bandwidth is 8 % in crystal size), and the bandwidths it does "
         "measure -- clump_um 2.73-3.69 um -- disagree with BOTH the corpus's "
         "estimates (median 13.0 um, implying 23 c/mm) and the two "
         "BBC-derived measured values (PAN_F 0.655, HPS 1.431). Three "
         "sources, three answers, recorded as a conflict"),
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
