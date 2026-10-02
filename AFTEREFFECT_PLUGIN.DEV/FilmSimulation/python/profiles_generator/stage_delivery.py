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

import os
import shutil
import zipfile
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPP = Path("/root/work/tst")           # the live, editable engine tree
OUT = Path(os.environ.get("FILMSIM_DELIVER_OUT", "/root/work/deliver23"))
#: ⚠ SUFFIXED. A second delivery was cut on the same day at schema v30,
#: and two archives named for one date cannot be told apart on disk. 2026-09-17
#: is the same case again: the 'a' set went out at schema v37 with 186 stocks,
#: this 'b' set has 191 and the six-archive layout.
#: ⚠ 2026-09-28c: overridable, because a third set was cut on 2026-09-28 and the
#: date alone would have overwritten the first set's names in DELIVERY.
STAMP = os.environ.get("FILMSIM_STAMP") or date.today().isoformat()

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
    # ⚠ NEW 2026-09-19d. Per-film control availability as a compile-time
    # bitmask, one uint64_t per film in database order. It ships with the
    # DATABASE rather than with either engine because it is indexed by
    # film::eFILM_PROFILE and static_asserts against eTOTAL_FILMS_PROFILES --
    # a copy in an engine archive would be a second file to keep in step with
    # the enumerators, which is the thing it exists to make impossible.
    "film_params_mask.hpp",
    "film_names.txt", "film_display_order.txt",
    # ⚠ SHIPS WITH THE DATABASE, 2026-09-08. The alphabetical re-sort moved 177
    # of 184 indices; this is the old -> new map, and it is the only thing that
    # can repair a project saved before the cutover. It belongs beside the
    # database it describes, not in a report.
    "film_id_migration.txt",
)

MANIFEST = """\
SEVEN ARCHIVES -- {stamp}
============================

Schema v61. 222 film stocks, 11 print stocks, 2 colour-paper spectral records,
14 gauges. Verify 1044 PASS / 4 baselined FAIL (saturation hierarchy + the
three G-SCRATCH guards whose harness is lost), all 40 translation units (38
data slots) compiling at -Wall -Wextra with zero bytes of output,
cpp_parity / chain_parity (222 stocks, Python = scalar = AVX2) /
spectral_mono_parity / interimage parity green. Build failures = the known
11-item baseline only.

ARCHIVES
--------
  1  python_generator      PYTHON/profile_generator  (no C++)
  2  generated_database    CPP/Algorithm/FilmProfile
  3  algorithm_scalar      CPP/Algorithm/Scalar
  4  algorithm_avx2        CPP/Algorithm/AVX2
  5  documentation_md      doc tree
  6  ui_mockup_and_pdf     UI documentation (mockup rev 13, EN/RU PDFs)
  7  cpp_tests             C++ test harness (test_chain_dump.cpp), flat

OWNER ACTION IN VISUAL STUDIO
-----------------------------
None: still 38 data slots (film_profiles_data_01..38.cpp), no new source file.
No film added since 2026-10-01e (KODAK_HIE, eKODAK_HIE = 122); the headers and
TXT files are regenerated anyway and unchanged apart from their timestamps.
AlgoControlEnums.hpp is unchanged since 2026-10-01e (ProcessVariantCtrl
TOTAL_PROCESSES 49).

WHAT CHANGED SINCE 2026-10-01
-----------------------------
  * Schema v61 (2026-10-01d): PrintingMatrix per negative (16 negatives print
    on 2383 through their own dyes), MTFSpec.combined_* (inert), the
    development_family rework (contrast-index families), temperature law
    keyed on contrast level; «Современные» held tables (786 points), 155
    kinetics curves (1139 points), dye panels, combined MTF on 7 stocks;
    Development Time on 33 stocks; SVEMA FOTO-250 gamma 0.795.
  * 2026-10-01e: NEW STOCK KODAK_HIE (eKODAK_HIE = 122; migrate projects with
    film_id_migration.txt) with six developer / format process variants;
    AGFA SCALA 200x push / pull ladder (five variants); Portra 160NC blue
    f50 60 -> 94; 27 of 28 book sensitivity panels corroborate the makers'
    records; SMPTE ST 2065-2 Academy Printing Density responsivities and 16
    per-negative scanner matrices stored INERT; book citations corrected to
    V. L. Likhachev, SLON-PRESS 2003.
  * Codegen: development points over 8 kB are emitted as helper functions in
    the same slot file (largest function 110 342 bytes, limit 112 000).
  * 2026-10-01f: stage 8b (Algo_08_Sim.cpp, Scalar and AVX2) on COLOUR
    NEGATIVES now leaves every neutral on the stored white-light curves (each
    layer measures a donor against that donor's density under a neutral
    exposure at the layer's own log E); interimage coefficients of all colour
    negatives re-solved. VISIBLE LOOK CHANGE on every colour negative: more
    contrast and saturation, much less grey-scale colour drift. Reversal
    unchanged. AGFA SCALA 200x Push 1-3 now use Agfa's own traced curves.
  * 2026-10-01g/h: FUJICOLOR_PRO_400H cyan (4th) record stored (inert); AVX2
    engine uses FMA at the last six multiply-then-add sites.
  * 2026-10-02: SCAN_DI is balanced PER COLOUR NEGATIVE (equal neutral system
    gamma per channel; Algo_13_Sim.cpp, AlgoDuplication.hpp) -- visible look
    change on colour negatives scanned through SCAN_DI: far less grey-scale
    colour drift. Small frames: the anchor solve and the print chain's mid
    grey now see stage 9's sub-pixel gate (AlgoDirCoupler.hpp,
    AlgorithmMain.cpp, Algo_13_Sim.cpp). Reversal stocks unchanged.

Full accounts: doc/RESULT_2026-10-01d_kinetics_dyes_mtf.md,
doc/RESULT_2026-10-01e_queue_followups_hie_scala_apd.md,
doc/RESULT_2026-10-01f_interimage_neutral_scala.md,
doc/RESULT_2026-10-02_scan_balance_smallframe_reversal_test.md, doc/PROGRESS.md.

Regenerated together from one build: film_profiles.*, film_schema_version.h,
film_enum.hpp, film_names.txt, film_display_order.txt, film_id_migration.txt,
film_params_mask.hpp, LoadFilmDataBase.*, film_profiles_data_01..38.cpp,
algo_control_enums.py, doc/FilmControlMatrix.md, doc/FilmActiveProfiles.md.

NO TEST CODE IN ARCHIVES 1-4 (owner instruction); the harness ships only in 7. AlgoControlEnums.hpp
carries the owner's Splice & Tear block verbatim. Films are in alphabetical
order in every generated list.
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
    # ⚠ THE MEASURED INFLUENCE TABLE SHIPS WITH ITS GENERATOR. It is the only
    # .json in this archive and it is not configuration: it is the result of a
    # ~15 minute render sweep that `gen_realism_score.py` refuses to run
    # without, so a generator archive lacking it cannot reproduce
    # doc/REALISM_SCORE.md at all.
    _infl = HERE / "realism_influence.json"
    if _infl.is_file():
        py.append(_infl)
    # The hold-out residuals travel with the generator too: the Markdown in
    # archive 5 prints the worst 40 rows and the full set lives here.
    _hold = HERE / "holdout_predictions.json"
    if _hold.is_file():
        py.append(_hold)
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
                ("FilmSimulator_Mockup_v6.html",
                 "FilmSimulation_EffectControls_EN.pdf",
                 "FilmSimulation_EffectControls_RU.pdf",
                 "README.txt")]
    missing = [f.name for f in ui_files if not f.is_file()]
    if missing:
        raise RuntimeError("archive 6 is missing %s" % ", ".join(missing))
    z6 = _zip(f"6_ui_mockup_and_pdf_{STAMP}.zip", ui, ui_files)

    # ---- 7. C++ test harnesses, optional ------------------------------------
    # ⚠ ADDED 2026-09-28. Archives 3 and 4 exclude every test_*.cpp by owner
    # directive, and until today the harnesses therefore reached the owner's
    # disk in no archive at all -- which is how `test_stage_dump.cpp` and
    # `test_scratch_guard.cpp` were lost between sessions, and why three verify
    # guards and stage_parity can no longer run. This archive holds ONLY the C++
    # harnesses, flat, and nothing else: it changes no other archive's layout
    # and mixes no Python into C++.
    tests = sorted(p for p in CPP.iterdir()
                   if p.is_file() and p.suffix == ".cpp"
                   and p.name.startswith("test_"))
    avx_tests = (sorted(p for p in (CPP / "AVX2").iterdir()
                        if p.is_file() and p.name.startswith("test_"))
                 if (CPP / "AVX2").is_dir() else [])
    zips = [z1, z2, z3, z4, z5, z6]
    # ⚠ 2026-10-01e: the owner asked for SIX archives; FILMSIM_NO_TESTS=1 skips 7.
    if (tests or avx_tests) and not os.environ.get("FILMSIM_NO_TESTS"):
        tstage = OUT / "stage" / "cpp_tests"
        if tstage.exists():
            shutil.rmtree(tstage)
        tstage.mkdir(parents=True)
        for p in tests:
            shutil.copy(p, tstage / p.name)
        if avx_tests:
            (tstage / "AVX2").mkdir()
            for p in avx_tests:
                shutil.copy(p, tstage / "AVX2" / p.name)
        zips.append(_zip(f"7_cpp_tests_{STAMP}.zip", tstage,
                         [p for p in tstage.rglob("*") if p.is_file()]))

    for z in zips:
        with zipfile.ZipFile(z) as zf:
            print(f"  {z.name:44s} {z.stat().st_size/1024:9.1f} kB  "
                  f"{len(zf.namelist()):4d} files")
    print(f"[OK] {len(zips)} archives in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
