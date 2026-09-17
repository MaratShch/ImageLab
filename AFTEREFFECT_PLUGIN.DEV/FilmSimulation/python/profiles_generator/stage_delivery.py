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
OUT = Path("/root/work/deliver10")
#: ⚠ SUFFIXED. A second delivery was cut on the same day at schema v30,
#: and two archives named for one date cannot be told apart on disk. 2026-09-17
#: is the same case again: the 'a' set went out at schema v37 with 186 stocks,
#: this 'b' set has 191 and the six-archive layout.
STAMP = date.today().isoformat() + 'b'

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
SIX ARCHIVES -- {stamp}
=======================

Schema v37. 191 film stocks, 11 print stocks, 14 gauges. Verify 749 PASS / 1
baselined FAIL, build clean, all six engine parity audits green, all 28
translation units compiling at -Wall -Wextra with zero bytes of output.

⚠⚠ A SIXTH ARCHIVE JOINS THE FIVE, AND IT IS THE ONE THAT CARRIES NO CODE
=========================================================================
The five production archives have always been strictly separated because the
owner integrates each into a different place in their tree. The UI mockup and
the two parameter PDFs are a sixth destination and were previously handed over
outside the numbered set; they are now archive 6, on the same rule -- one
archive, one place, never a merge.

WHAT CHANGED IN THIS DELIVERY
=============================

FIVE NEW FILM STOCKS AND ONE FROM EARLIER THE SAME DAY: 186 -> 191
------------------------------------------------------------------
185  EASTMAN_5293_250T_1982   the 1982 EI 250T emulsion, beside the 1992 EXR
                              200T film that reuses the catalogue number.
                              Traced from Kennel, Sehlin et al., SMPTE J.
                              91(10) 1982, 922-930: characteristic curves,
                              spectral sensitivity, spectral dye density,
                              push-1 sensitometry, MTF and granularity, each
                              cross-checked against the paper's own stated
                              numbers on every build.
186-190  the five 1956 Kodak sheet emulsions -- SUPER PANCHRO-PRESS TYPE B,
                              PORTRAIT PANCHROMATIC, ROYAL ORTHO, SUPER SPEED
                              ORTHO PORTRAIT and COMMERCIAL. Every number they
                              carry had been read before today; what was
                              missing was a reader for the two blockers.

THREE SCHEMA VERSIONS: v34 -> v37
---------------------------------
v35  DyeStabilitySpec reaches camera negatives, and AlgoControls gains
     storageYears. A published time to a 10 % dye loss becomes a state through
     f(t) = 1 - 0.9^(t/T) -- not fitted, and exactly 0.10 at t = T. ONLY THE
     DYE THE SOURCE NAMES IS FADED, because the differential between dyes IS
     the effect. Populated on 3 of 191 stocks; inert on the other 188 at any
     age.
v36  ProcessingFamily.reference_developer / reference_dilution. The field
     exists because a stock LOST a control by GAINING data: SUPER-XX PAN came
     out of the P61 harvest with 17 DK-50 points against 17 DK-60a, and
     development_family refuses a tie by design.
v37  ProcessVariant.variant_id -- the database half of the enumeration below.

THE PROCESS VARIANT CONTROL IS NO LONGER AN INDEX
--------------------------------------------------
AlgoControls::processVariant was an int32_t position in whichever stock
happened to be loaded, so the stored value 2 named "RODINAL 1+50" on an
AGFAPAN, "ECN-2, the base stock's native process" on CINESTILL 800T and
"EI 3200 (Push 2)" on PORTRA 800. Every one of them was in range, so no test
could tell a stale project from a correct one, and inserting a variant
re-pointed every saved selection in silence.

It is now ProcessVariantCtrl, a global enumeration in AlgoControlEnums.hpp,
which is the single authority for three lists that must be one list: the
enumerators, the pipe-separated ListBox strings, and the database keys the
resolver matches. Three static_asserts refuse to compile if the three ever
differ in length, and four verify.py guards bind the header, the generated
Python mirror, the database and every stamped variant_id together.

⚠ TOTAL_PROCESSES IS THE LAST ENUMERATOR AND THE COUNT (21). It sizes the
string table and bounds the range check and is NEVER a selectable item. A
value outside [0, TOTAL_PROCESSES), or one the selected stock does not offer,
resolves to eAS_SHIPPED rather than being clamped: a stale preset should render
the stock as shipped, not render some other development in its place.

FIVE QUEUE ROWS CLOSED
----------------------
P61  the 1956 time-gamma insets: 13 curves, 189 points, four stocks, six
     developer conditions new to the corpus.
P62  the thirty 1956 wedge spectrograms. The reader is validated on Kodak's own
     page-12 three-class reference stack (490 / 587 / 655 nm in that order)
     before any data-sheet plate is believed, and the vertical scale Kodak
     never published is recovered from the CIE 1924 V(lambda) curve drawn on
     that same plate.
P63  the five sheet emulsions above. 26 characteristic curves traced, checked
     against the gamma Kodak letters beside each one: 23 of 26 within 3.9 %,
     three pinned as refusals rather than covered by a wider tolerance.
P64  the dark-storage control: AlgoStorageAge.hpp turns the published rate into
     a state. Its second half -- a storage TEMPERATURE -- is refused, because
     three published temperatures do not define a continuous law.
P65  the 1938 Kodak Research Laboratories speed scale converts at exactly 4.00
     on two independently rated panchromatic films and at 1.60 on the one
     non-colour-sensitive film, so the conversion is adopted for panchromatic
     sheet film only and the third ratio is recorded as a refusal.

The live queue goes 43 -> 39 rows.

THE SIX ARCHIVES
================
  1  python_generator     the generator and its audits. NO C++ AT ALL.
  2  generated_database   include/ + src/, the layout the VS project expects.
  3  algorithm_scalar     include/ + src/, the double build.
  4  algorithm_avx2       include/ + src/, the float build.
  5  documentation_md     the whole doc/ tree.
  6  ui_mockup_and_pdf    the HTML control-surface mockup and the two parameter
                          references. No source, no database, no Markdown.

⚠ NO TEST SOURCE IN ANY OF THE SIX -- the test translation units, profall.cpp
and the two harnesses are in the separate optional archive. The .txt files ship
inside include/ in archives 3 and 4.

⚠ THE TWO ENGINE ARCHIVES ARE NEVER MERGED. Scalar computes in double and AVX2
in float; they are two files carrying one law, and the parity audits compile
BOTH and compare each to the Python reference, which is what makes keeping them
separate safe rather than merely tidy.
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
    return name.startswith("test_") or name == "profall.cpp"


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
                ("FilmSimulator_Mockup_v4.html",
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
