#!/usr/bin/env python3
"""Stage and zip the six delivery archives. Strictly separated, by design.

⚠ SIX ARCHIVES, NOT ONE, AND THE SEPARATION IS THE POINT. The owner integrates
each into a different place: the generator into `PYTHON/profile_generator`, the
database into `CPP/Algorithm/FilmProfile`, the two engines into
`CPP/Algorithm/Scalar` and `CPP/Algorithm/AVX2`, the Markdown into the doc
tree, and the mockup and the two parameter PDFs into the UI documentation. A
single archive would make every delivery a merge.

⚠ THE SIXTH IS NEW ON 2026-09-17b AND CARRIES NO CODE. The HTML control-surface
mockup and the English and Russian parameter references were previously handed
over outside the numbered set, which meant the one deliverable describing the
CONTROL SURFACE had no place in the delivery that changed it. It is archive 6
now, on the same rule as the other five.

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
OUT = Path("/root/work/deliver12")
#: ⚠ SUFFIXED. A second delivery was cut on the same day at schema v30,
#: and two archives named for one date cannot be told apart on disk. 2026-09-17
#: is the same case again: the 'a' set went out at schema v37 with 186 stocks,
#: this 'b' set has 191 and the six-archive layout.
STAMP = date.today().isoformat()

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
    # ⚠ NEW 2026-09-18e. The schema version and both of its accessors, in the
    # one generated header that is valid C as well as C++. film_profiles.hpp
    # includes it and restates no digits, so shipping the database without it
    # would not compile.
    "film_schema_version.h",
    "film_enum.hpp", "LoadFilmDataBase.h", "LoadFilmDataBase.cpp",
    "film_names.txt", "film_display_order.txt",
    # ⚠ SHIPS WITH THE DATABASE, 2026-09-08. The alphabetical re-sort moved 177
    # of 184 indices; this is the old -> new map, and it is the only thing that
    # can repair a project saved before the cutover. It belongs beside the
    # database it describes, not in a report.
    "film_id_migration.txt",
)

MANIFEST = """\
SIX ARCHIVES -- {stamp}
=======================

Schema v48. 191 film stocks, 11 print stocks, 2 colour-paper spectral records,
14 gauges. Verify 856 PASS / 1 baselined FAIL, build gate green end to end,
every engine parity audit green, all 28 translation units compiling at
-Wall -Wextra with zero bytes of output.

ONE SYNCHRONISED STATE, SIX DESTINATIONS
========================================
The archives are strictly separated because the owner integrates each into a
different place: the generator into PYTHON/profile_generator, the database into
CPP/Algorithm/FilmProfile, the two engines into CPP/Algorithm/Scalar and
CPP/Algorithm/AVX2, the Markdown into the doc tree, and the mockup and the two
parameter references into the UI documentation. A single archive would make
every delivery a merge.

EVERY ARCHIVE IS CUT FROM THE SAME STATE, and that is asserted rather than
intended: the generated C++ in archive 2 was written by the same build that ran
verify, the parity audits and the compile gate; archives 3 and 4 are the engine
tree that build synced and compiled; and the documents in archive 5 were
regenerated or re-derived in that same build, with doc_consistency.py checking
31 registered counts and the queue's live-row set against the database.

NO TEST CODE IN ANY CODE ARCHIVE
--------------------------------
Standing owner instruction. The engine archives exclude the thirteen
test_*.cpp files, profall.cpp and e2e.cpp; the Python archive carries no
test_*.py and no make_test_chart.py. build.py, verify.py and the parity
harnesses stay, because they are the build GATE rather than tests of it.

WHAT CHANGED IN THIS DELIVERY
=============================

THE FILM GRAIN MODEL IS REPLACED, TO FGS-DDS-001 Rev. A, AT SCHEMA v48
----------------------------------------------------------------------
Governing document: FGS-DDS-001 Rev. A, 51 pp., 2026-09-19, owner-approved in
full with one constraint -- Python, C++ scalar and C++ AVX2 must execute the
SAME ALGORITHM FLOW. That constraint decided the architecture.

⚠⚠ NOT ONE RENDER IS BIT-IDENTICAL TO A v47 ONE, ON PURPOSE. The mandatory F1
fix changes every pixel by construction and the spectral rebase changes the
texture on 186 stocks. What IS guarded: in `legacy_gaussian` mode the v48
spectrum reproduces the v47 spectrum to 4.4e-16, so the difference in that mode
is the generator and nothing else.

  1. THE SPECTRUM is the radius-averaged Boolean (jinc) form of the random-dot
     literature, h(f)^2 = E_r[r^4 b(f;r)^2]/E_r[r^4], carried as a FIVE-TERM
     GAUSSIAN MIXTURE. ⚠ Spec 19.1 asks for a per-frame 2-D FFT; neither C++
     engine has one, and writing one into two engines against a no-allocation
     policy would have produced three implementations of two algorithms. The
     mixture is what all three engines already execute. Worst fit error over
     the corpus 1.305e-03, against the aperture correction's own 1.0e-02 to
     5.3e-02 -- thirty times smaller than an error the model already accepts.
  2. THE GRAIN DIAMETER is derived from the published RMS granularity by a
     closed-form inversion with no fitted constant in it: 0.1835 - 2.6558 um,
     median 0.719. The five stocks whose diameter is a BBC T-101 measurement
     keep it and agree with the derivation at 1.04 / 1.05 / 1.21 / 1.62 / 1.91.
  3. THE 48 um APERTURE is the exact disk transfer, not a Gaussian stand-in.
  4. THE FIELD IS FRAME-KEYED through a counter-based generator (SplitMix64
     finaliser, seed/stage/ordinal counter), identical in all three engines.
  5. THE MARGINAL LEAVES GAUSSIAN where the grain count per resolution element
     is low, through the specification's own count gate driving a
     Cornish-Fisher skew.

FOUR PRE-EXISTING ENGINE DEFECTS, THREE OF THEM OUTSIDE GRAIN
--------------------------------------------------------------
  F1  the Python reference produced IDENTICAL grain on every frame of a clip
      while both C++ engines re-rolled it. The model of record and the shipping
      engines were simulating different physics, and every harness passed --
      because every harness compares ONE FRAME.
  kSigma, both Algo_11_Sim.cpp: 0.22508352815546 against 1/(pi*sqrt(2)) =
      0.22507907903927651.
  AlgoScanSigmaMm, both Algo_10_Sim.cpp: 0.18738564618678 against
      sqrt(ln2/2)/pi = 0.1873906251292776. This is STAGE 10 -- the whole image,
      not only the grain.
  ALGO_MTF_SIGMA_MM_PER_INV_F50, AlgoEmulsionMtf.hpp: the same wrong digits
      again, and this one is the EMULSION MTF read by stage 6, i.e. a property
      of every stock.
⚠ The last three hid for one structural reason worth carrying beyond grain:
the reference evaluates its transfers straight onto a frequency grid and
DERIVES NO SIGMA AT ALL, so each constant existed on the C++ side alone and had
nothing to disagree with. A quantity computed in one engine and not in the
other is not covered by parity testing, however thorough that testing is.

THE CONFORMANCE AUDIT, AND THE FIVE GAPS IT FOUND
-------------------------------------------------
All eighteen normative requirements were audited against the shipped code:
13 met, 3 partial, 2 not met. All five gaps are closed:

  R-S4(a)  the dispersion term was in the spectrum and not in the counts. With
           E[pi r^2] in the mean area the calibration loop closes from 1.0317x
           to 4.4e-16, and every derived diameter is 3-6 % smaller.
  R-N3     two EXACT factoring identities plus an identity-skip: AVX2, one
           channel at 4K, 398 -> 95 ms with a clustering lobe and 124 -> 62 ms
           without. ⚠ THE BUDGET IS 11.9 ms PER CHANNEL AND IS STILL MISSED,
           8x and 5x. See WHAT IS STILL OPEN.
  R-S6     the exact dot renderer was BUILT, measured at 25-100 ms per channel
           against an 8 ms budget, and refused on that measurement. The count
           gate ships; a Cornish-Fisher marginal replaces the sampler, with its
           coefficient clamped at 0.1 for monotonicity -- the clamp binds on
           2.04 % of the stock-density space, measured rather than waved at.
  R-S5     the saturating sigma(D) family was fitted and REFUSED BY ITS OWN
           GATE: 14.0 % mean / 25.3 % worst against the specification's 10 %.
           sigma_sat_droll and sigma_sat_q stay 0.0 on 191 of 191.
  R-T5     the specification contradicts itself -- 18.2 asks the stage to
           FREEZE the field for one class and R-T5 forbids any frame-locked
           component, which is what a frozen field is. The requirement wins;
           the field is declarative and validate_all refuses any other value.

FOUR ERRATA AGAINST THE SPECIFICATION BECAME SIX
------------------------------------------------
The two new ones were found by building the dot renderer that was then refused,
which is the argument for costing a requirement by building it: the
compound-Poisson skewness 1/sqrt(N) is low by exp(6 sigma_ln^2), and 12.4.1's
break densities were computed without the dispersion term 13 itself requires.

SCHEMA v47 -> v48
-----------------
GrainSpec gains grain_um_r/g/b, development_gamma_ref, grain_temporal_class,
sigma_sat_droll, sigma_sat_q and rho_layers; film_profiles.hpp gains
GrainSpectrumTerms, the factored spectrum both engines consume.
⚠ development_gamma_ref and rho_layers are populated on 0 of 191 and read by no
law. That is deliberate -- an unread carrier cannot move a pixel -- and it is
queue row P81 rather than a silent gap.

THE GATE ITSELF
---------------
verify.py gains 24 v48 guards and the specification's seven named validation
gates (V-CAL, V-NORM, V-TEMP, V-COMPAT, V-SAT, V-SPARSE, V-PARITY).
cpp_parity.py gains the first probe in this project that compares grain PIXELS
rather than grain statistics: RNG draws bit-exact against AlgoCounterRng.hpp,
mixture terms and the clustering lobe to 4.8e-08, the field to 2.7e-07 of field
RMS, and frames 0 and 7 sharing 0 of 3072 pixel values.
build.py's docs stage now also regenerates doc/FilmControlMatrix.md, which was
the one generated document nothing regenerated.

DOCUMENTATION
-------------
All eight named documents were REVIEWED, not appended to. Every figure derived
from a grain diameter was recomputed with the shipped code and restated in
place rather than annotated. PROJECT_STATE.md, FilmActiveProfiles.md,
FilmCurves.md and FilmControlMatrix.md are regenerated from the live module on
every build and cannot drift. Both FilmDatabase_Charecteristics documents carry
a new D.17.7 in English and Russian; FilmGrainSimulationModel.md carries a new
12.10; GRAIN_MODEL_ASSESSMENT.md a new 8.6; DIGITIZATION_QUEUE.md rows P74-P81,
with its census re-derived from the parse.

WHAT IS STILL OPEN
------------------
Twelve queue rows.

⚠ P80 IS THE ONE THAT NEEDS A DECISION FROM THE OWNER, and it is the honest
residue of this delivery. R-N3's budget is missed by 8x on the 186 stocks that
carry a clustering lobe. The lobe blur alone is 47 of the 95 ms, so setting
clump_gain to zero is a further 3.3x -- but that parameter was FITTED to
rendered results on those 186 stocks under queue C45, so dropping it is a
modelling decision and not an optimisation. Stated with the alternative: even
with clump_gain = 0 the budget is still missed 5x, so it is not reachable on
this architecture by dropping the lobe alone.

P81 asks for two quantities nothing in this corpus prints. C14, F1, K5, M1b,
P19, P20 and P38 need a document nobody here has; D1, D2a and D2b need scans
only the owner can make. The one baselined verify failure is unchanged and is
documented where it is asserted.
"""


def _zip(name: str, root: Path, files: list[Path]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    z = OUT / name
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(files):
            zf.write(f, f.relative_to(root).as_posix())
    return z


def _is_test_source(name: str) -> bool:
    """True for a file that is a test or a profiling harness, not production.

    ⚠ `profall.cpp` IS ON THIS LIST AND IT DOES NOT MATCH `test_`. It is a
    profiling driver with its own `main`, so shipping it in an archive whose
    stated contents are "the production algorithm source" would put a second
    entry point in the build. Matching on the prefix alone would have missed
    it, which is why this is a function rather than a `startswith` at the
    call site.
    """
    # ⚠ `e2e.cpp` JOINED THE LIST ON 2026-09-18e. It is the whole-chain dump
    # the Python reference is compared against -- a third entry point with its
    # own `main`, writing /tmp/e2e_cpp.bin -- and it had been shipping inside
    # both engine archives since the list was written, because it matches
    # neither `test_` nor `profall.cpp`. The owner's instruction this delivery
    # is that no test code appears in the C++ or Python packages at all.
    return (name.startswith("test_")
            or name in ("profall.cpp", "e2e.cpp"))


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
    # ⚠ film_schema_version.h JOINS THE SKIP SET ON 2026-09-18e for the same
    # reason every other name here is on it: it is the DATABASE's header and
    # ships in archive 2. film_profiles.hpp includes it, so an engine building
    # against archive 2 has it; a second copy inside archives 3 and 4 would be
    # two files carrying one version literal, which is the one thing that
    # header exists to prevent.
    skip = {"film_profiles.hpp", "film_profiles_detail.hpp", "film_enum.hpp",
            "film_schema_version.h",
            "LoadFilmDataBase.h", "LoadFilmDataBase.cpp", "film_profiles.cpp"}
    for p in sorted(CPP.iterdir()):
        if not p.is_file() or p.name in skip:
            continue
        if p.name.startswith("film_profiles_data_"):
            continue
        # ⚠ NO TEST SOURCES IN A PRODUCTION ARCHIVE. Owner directive
        # 2026-09-11: "Please do not include C++ or C/C++ header test files in
        # any of the five primary project archives." The engine tree carries
        # thirteen of them beside the production stages -- test_*.cpp plus
        # profall.cpp, which is a profiling harness rather than a stage -- and
        # they compile against headers this archive does ship, so a reader
        # would reasonably take them for part of the build. They go in the
        # separate optional archive instead.
        if _is_test_source(p.name):
            continue
        if p.suffix in (".hpp", ".h"):
            shutil.copy(p, dst / "include" / p.name)
        elif p.suffix == ".cpp":
            shutil.copy(p, dst / "src" / p.name)
        elif p.suffix == ".txt":
            # ⚠ TXT GOES IN include/, ON THE OWNER'S INSTRUCTION, and it is
            # not an arbitrary placement: film_names.txt and
            # film_display_order.txt are read at runtime beside the generated
            # database headers, so they belong where the headers are rather
            # than beside the sources that never open them.
            shutil.copy(p, dst / "include" / p.name)
    if kind == "AVX2":
        for p in sorted((CPP / "AVX2").iterdir()):
            if not p.is_file() or _is_test_source(p.name):
                continue
            sub = "include" if p.suffix in (".hpp", ".h", ".txt") else "src"
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
    # ⚠ AND NO TEST APPARATUS ON THE PYTHON SIDE EITHER, 2026-09-18e. There
    # are no test_*.py modules in this generator and never have been, but
    # `make_test_chart.py` writes the synthetic ramp-and-disc frame the engines
    # are rendered against. It is testing apparatus rather than a generator
    # source, and the owner's instruction this delivery admits no test code in
    # the Python package. `build.py`, `verify.py` and the parity harnesses
    # stay: they are the build GATE, invoked by build.py itself, and removing
    # them would ship a generator that cannot check its own output.
    py = [p for p in py
          if not (p.name.startswith("test_") or p.name == "make_test_chart.py")]
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

    # ---- 6. UI mockup and the two parameter references ---------------------
    # ⚠ NO SOURCE, NO DATABASE, NO PROJECT MARKDOWN, and the generator that
    # BUILDS the two PDFs is deliberately not here either: it imports
    # `algo_control_enums` and `film_profiles`, so shipping it would put a
    # dependency on archives 1 and 2 inside an archive whose whole point is
    # that it stands alone on the documentation shelf. What ships is the
    # rendered output and the mockup it describes.
    ui = Path("/root/work/ui")
    ui_files = [ui / n for n in
                ("FilmSimulator_Mockup_v5.html",
                 "FilmSimulation_EffectControls_EN.pdf",
                 "FilmSimulation_EffectControls_RU.pdf",
                 "README.txt")]
    missing = [f.name for f in ui_files if not f.is_file()]
    if missing:
        raise RuntimeError("archive 6 is missing %s" % ", ".join(missing))
    z6 = _zip(f"6_ui_mockup_and_pdf_{STAMP}.zip", ui, ui_files)

    for z in (z1, z2, z3, z4, z5, z6):
        with zipfile.ZipFile(z) as zf:
            print(f"  {z.name:44s} {z.stat().st_size/1024:9.1f} kB  "
                  f"{len(zf.namelist()):4d} files")
    print(f"[OK] six archives in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
