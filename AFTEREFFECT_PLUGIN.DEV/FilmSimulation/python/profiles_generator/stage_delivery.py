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
OUT = Path("/root/work/deliver11")
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

Schema v44. 191 film stocks, 11 print stocks, 2 colour-paper spectral records,
14 gauges. Verify 809 PASS / 1 baselined FAIL, build clean, every engine parity
audit green, all 28 translation units compiling at -Wall -Wextra with zero
bytes of output.

ONE SYNCHRONISED STATE, SIX DESTINATIONS
========================================
The archives are strictly separated because the owner integrates each into a
different place: the generator into PYTHON/profile_generator, the database into
CPP/Algorithm/FilmProfile, the two engines into CPP/Algorithm/Scalar and
CPP/Algorithm/AVX2, the Markdown into the doc tree, and the mockup and the two
parameter references into the UI documentation. A single archive would make
every delivery a merge.

NO TEST CODE IN ANY CODE ARCHIVE
--------------------------------
Owner instruction this delivery. The engine archives already excluded the
thirteen test_*.cpp files and profall.cpp; e2e.cpp JOINS THAT LIST NOW. It is
the whole-chain dump the Python reference is compared against -- a third entry
point with its own main() -- and it had been shipping in archives 3 and 4 since
the exclusion list was written, because it matches neither pattern. The Python
archive contains no test_*.py and never has; build.py, verify.py and the parity
harnesses stay, because they are the build gate rather than tests of it.

WHAT CHANGED IN THIS DELIVERY
=============================

TWENTY-TWO QUEUE ROWS CLOSED, NONE OPENED: THE LIVE SET FALLS 32 -> 11
----------------------------------------------------------------------
P11 P17 P18 P33 P35 P36 P45 P51 P52 P53 P54 P56 P66 P68, then M1a P12 P13 P14
P39 P40 P41, then P73 -- opened and closed the same day on four patents the
owner supplied. Not one document was acquired for the first twenty-one; they
closed on carriers built, defects found and refusals recorded.

FOUR SCHEMA VERSIONS: v40 -> v44
--------------------------------
v41  MTFSpec gains a LENS-REGIME resolving-power pair; DevelopmentLaw makes a
     ProcessingFamily hold a fitted gamma(t) per developer AND vessel;
     DyeImpurityRatio.quantity says what a ratio is a ratio OF; FilmProfile
     gains a daylight exposure index and both conversion filters.
v42  COLOUR PAPER STOPS BEING AN EMPTY CLASS. PaperSpectralRecord and
     PAPER_SPECTRA carry the Fujicolor Crystal Archive spectral dye density and
     spectral sensitivity panels -- TWO records covering THREE products,
     because the Supreme and Type CA bulletins publish byte-identical artwork
     and one measurement must be stored once. "reflection" joins
     _DENSITY_GEOMETRIES. No paper PROFILE was built: none of the three
     bulletins prints a characteristic curve, and four real spectral curves
     hung off an invented tone curve would look complete and be wrong.
v43  Six carriers. The GOST speed criteria become an EDITION-keyed table with
     9160-82 present and empty, because nobody has read it; the magenta
     light-fade default becomes coupler-keyed, because the ranking INVERTS
     between the 1950s and 1980s chemistries; SubLayerSet gives a colour record
     the offset sub-layers EP 0 083 377 A1 describes AND the law that sums
     them; PrintStock.reader_is_emulsion answers which stocks may serve as
     stage 12's M_reader; the NIKFI speed scale gets a carrier with NO
     conversion invented; and both Iofis 1981 summary tables are transcribed.
v44  THE 1931-1969 ANTIHALATION PATENT CHAIN. See below.

THE FOUR PATENTS, AND WHAT THEY SETTLED
---------------------------------------
US 1,908,527 (McMaster, Eastman Kodak, 1933) - US 2,182,794 (Dawson, Du Pont,
1939) - US 2,481,770 (Nadeau, Eastman Kodak, 1949) - US 3,445,231 (Nishio et
al., Fuji, 1969). A chain, not four opinions: the Fuji patent cites the Du Pont
one.

1. AntiHalationSpec.position is a PHYSICAL SYSTEM, not a label. A BACKING is
   capped at 0.10-0.30 D -- two Kodak patents eighteen years apart give the
   same reason, that positives are printed THROUGH it -- where an in-path
   absorber runs to 2.0 D. The bands do not overlap and ah_od_band_for()
   refuses every position the chain does not cover.
2. The Fuji patent states in prose the reason P71 made halation_gain_from_od
   refuse position == "backing": "the presence of a comparatively thick support
   layer between the light-sensitive layer and the anti-halation layer reduces
   the anti-halation effect of the layer itself."
3. THE FIRST QUANTITATIVE HALATION MEASUREMENT IN THE CORPUS. Halation latitude
   1.15 bare, 1.56 with a low-index sublayer alone, 1.65 / 1.74 / 1.81 as a
   0.07 / 0.099 / 0.1 D dye backing is added, 2.0-2.25 for an ordinary backing.
   The spatial model can now be CHECKED, not only fitted.
4. A SECOND CRITICAL ANGLE. base_fresnel() gives the base/air one the radii are
   derived from; emulsion_side_critical_angle() adds the emulsion/sublayer one
   -- 71.5 degrees at the patent's minimum 0.08 index break against acetate's
   42.5. It answers None for every stock here, which is honest: an ordinary
   gelatin subbing has the emulsion's own index.
5. The first measured SPEED COST of an in-pack antihalation layer: 50 per cent
   unmordanted against 20 per cent mordanted.
6. _V34_BASE_OPTICS gains a "nitrate" row at the measured 1.498 -- the support
   every 1930s stock in this database was actually coated on -- and the same
   1949 table corroborates the acetate index in use to 0.07 per cent.

No band was written onto a profile. 30 stocks name an antihalation position and
none carries an optical density; a midpoint stamped on nineteen backing stocks
would be nineteen inventions.

THIRTEEN STOCKS' RENDERED OUTPUT CHANGED, AND IT IS A CORRECTION
-----------------------------------------------------------------
Queue P51 pointed verify.py's overshoot probe at mtf_kernel_response -- the
kernel the engine actually convolves -- instead of the analytic law nothing
applies. Ten of the thirteen A4/T2 stocks then stopped reproducing their own
datasheets, because their adjacency had been fitted to the law. All thirteen
were re-solved against the kernel and every one is back on its printed
overshoot in both height and peak frequency. The corrections are small; the
largest is 5217, amplitude 0.151 -> 0.189.

Two MTF values moved with a better-calibrated axis: KODAK TECHNICAL PAN f50
72.3 -> 73.09 (the digit bank learned two more abscissa labels, so the log fit
runs on eleven printed labels instead of nine) and the T-MAX 400 reading was
restored to 98.7 after a bank addition briefly degraded it.

EVERY ENUMERATED CONTROL VALUE NOW STARTS AT ZERO
-------------------------------------------------
Owner requirement. ProcessVariantCtrl::eAS_SHIPPED was -1 and is 0; the
twenty-one developments run 1-21 and TOTAL_PROCESSES is 22. It was the only
negative enumerator in AlgoControl's enums; FilmFormatCtrl and PrintStockCtrl
already started at zero.

NOTHING BEHAVIOURAL CHANGED and the mechanism is worth stating: the inertness
of "as shipped" was never the range test, it is the EMPTY DATABASE KEY at entry
0 of ProcessVariantCtrlKey. Both resolvers look that key up in the selected
stock's own process_variants, find nothing and return the shipped profile --
exactly what the out-of-range sentinel did. A static_assert now pins that entry
empty so a future edit cannot make index 0 select a development.

⚠ A PRESET SAVED BEFORE THE CHANGE STORES DIFFERENT NUMBERS. -1 is no longer
legal and takes the same inert path; every other stored value is one LESS than
its new one, so a host migrating old presets adds one.

The float sentinels are NOT enumerations and are unchanged: flare, vignette,
developmentMinutes and developmentCelsius still default to -1.0 meaning "use
the stock's own value", because 0 is a legal value for each of them.

A C/C++ SCHEMA-VERSION API, IN A HEADER C CAN ACTUALLY INCLUDE
---------------------------------------------------------------
New generated file: film_schema_version.h, in archive 2 and synced to the
project root.

  enum {{ kFilmDatabaseSchemaVersionValue = 44 }};          /* the one literal */
  static inline int32_t FilmDatabaseSchemaVersion(void);  /* C and C++ */
  constexpr std::int32_t film::GetFilmDatabaseSchemaVersion() noexcept;

⚠ IT IS A SEPARATE HEADER ON PURPOSE. The request was for the API to live in
the generated database headers, and in film_profiles.hpp it could not be
C-callable at all: that header is C++ throughout -- <array>, <string>,
<vector>, classes, namespaces -- so a C translation unit cannot include it, and
an API a C compiler can never reach is not a C-compatible API. The new header
includes <stdint.h> and nothing else, compiles clean as C99 and as C++, and
film_profiles.hpp includes it rather than restating the number. THE VERSION
LITERAL APPEARS EXACTLY ONCE. Being constexpr, the C++ accessor also answers at
compile time, so a consumer can static_assert against it.

DOCUMENTATION
-------------
All eight named documents were reviewed rather than appended to.
PROJECT_STATE.md, FilmActiveProfiles.md, FilmCurves.md and FilmControlMatrix.md
are regenerated from the live module on every build, so they cannot drift.
DIGITIZATION_QUEUE.md, NotFound.md and PROGRESS.md carry the closures and the
corrections. Both FilmDatabase_Charecteristics documents had a header claiming
schema v34 and 184 stocks -- three weeks and ten versions stale -- which is
CORRECTED rather than annotated, and both gain a section D.14 covering v41-v44.

WHAT IS STILL OPEN
------------------
Eleven queue rows, and every one is blocked outside this project:
C14 F1 K5 K6 M1b P19 P20 P38 need a document nobody here has (K5, K6 and M1b
are proved absent; P20 is paywalled; P19 needs J-PlatPat PDFs); D1, D2a and D2b
need scans only the owner can make. The one baselined verify failure is
unchanged and is documented where it is asserted.
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
