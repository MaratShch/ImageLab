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
FIVE ARCHIVES -- {stamp}
========================

⚠⚠ THIS SUPERSEDES 2026-09-11a. THAT SET DID NOT BUILD IN VISUAL STUDIO.
=========================================================================

Three build breaks, all found by the owner compiling the previous set. Two of
them were LONG-STANDING and had shipped silently more than once; one was mine
from the same day.

1. AlgoReciprocity.hpp AND AlgoProcessVariant.hpp WERE MISSING FROM THE TREE.
   AlgorithmMain.cpp includes both and calls both -- AlgoReciprocityLogShift
   before stage 8, AlgoResolveProcessVariant before anything reads a curve --
   and neither file existed. The driver therefore did not compile, in this
   tree or in the last two deliveries cut from it. Both are RECONSTRUCTED here
   from the Python reference, which never stopped applying either law, and
   both are verified numerically rather than by inspection:

       reciprocity      6624 probes, 184 stocks x 12 exposure times,
                        worst disagreement 1.01e-07 decades
       process variant  worst curve-parameter disagreement 4.53e-07

   ⚠ WHY IT WAS SILENT, WHICH MATTERS MORE THAN THE FILES. build.py's compile
   step covers the 26 GENERATED database translation units and nothing else,
   so AlgorithmMain.cpp -- the one file that includes thirty-odd headers and
   calls every stage -- was never compiled by the gate at all. And
   cpp_parity.py DID reference AlgoReciprocity.hpp, but SKIPPED when it was
   absent. One [SKIP] line in a long green log reads exactly like a pass. The
   guard that existed to protect the law is the reason its loss went unnoticed.

   Both holes are now closed: cpp_parity FAILS instead of skipping when the
   engine is present but the header is not, and a new verify guard,
   G-ENGINE-INCLUDES, reads every #include in all 122 engine sources and
   asserts the target exists. It needs no toolchain, and it is exactly the
   failure that got through -- not a bad expression, an absent file.

2. Exp2Accurate WAS PUT IN THE WRONG COMPONENT, AND THAT ONE IS MINE.
   The accurate exp2 that fixes the stage-14 defect went into
   FastAriphmeticsAVX.hpp, which is the obvious home -- it is where the other
   vector transcendentals live. But that header is a COMMON component: in your
   tree it is CPP\Common\include and it is shared by every project, while
   stage 14 ships in the AVX2 algorithm archive. The archives are deliberately
   separate, so the updated Common header never arrived, the stale copy on the
   include path won, and the compiler said

       error C2039: 'Exp2Accurate': is not a member of 'FastCompute::AVX2'

   The function now lives in AVX2/Algo_14_Sim.cpp itself, in the anonymous
   namespace. One consumer, one translation unit, no cross-archive coupling.
   FastAriphmeticsAVX.hpp is back to its original content and this delivery
   does not require you to touch Common at all. The measured accuracy is
   unchanged: 0.000326 % worst relative error against exact 10^-d, and exactly
   1.0 at D = 0, against the Schraudolph version's 2.98 % and 0.978161.

   ⚠ THE GENERAL LESSON IS IN THE TREE NOW: a stage may not depend on a change
   to a component it does not ship with.

NEW IN THE PYTHON RENDERER: -8bpp
==================================

`film_sim.py` gains `-8bpp` (also spelled `--8bpp`). It writes an 8-BIT RGBA
PNG with a constant opaque alpha.

⚠ IT IS NOT A SYNONYM FOR `--bits 8`. That option already existed and writes
an 8-bit THREE-channel file. The new flag adds the fourth channel, because
that is what the comparison tools want: a 16-bit render cannot be diffed
against an 8-bit source without a requantisation step that invents differences
of its own, and several viewers refuse a three-channel file outright. The
alpha is constant 255 and carries no information -- film is opaque and no
stage in the renderer produces coverage -- so anyone computing with it should
ignore it.

`-8bpp` OVERRIDES `--bits` rather than conflicting with it, so `-8bpp
--bits 16` is 8-bit and not an error. Without the flag nothing changes: the
default is still 16-bit RGB, byte for byte as before.

⚠ AND FIXING IT EXPOSED A SMALLER DEFECT WORTH KNOWING ABOUT. The output
encoding was being read in THREE places -- both `save_linear` call sites read
`args.bits` directly while `RenderSettings.bit_depth` was set from the same
value a few lines away. That is how a new flag gets honoured in one place and
silently ignored in the other two. It is now decided once and used
everywhere.

WHAT IS UNCHANGED FROM 2026-09-11a
====================================

Everything else in that manifest still stands and is not repeated here: schema
v33, the EP 0 083 377 A1 harvest, the AVX2 stage-14 accuracy fix, the grain
anisotropy wiring in all three implementations, the stage-17 NaN alignment,
the two false comments corrected, and the 349-field audit. Verify is now
691 PASS / 1 baselined FAIL -- one higher than the last set, which is
G-ENGINE-INCLUDES. Build clean, 0 warnings. cpp_parity green across grain,
MTF, reciprocity, process variant and Callier.

⚠ A v33 RENDER IS STILL NOT BIT-IDENTICAL TO A v32 ONE. Three engine defects
were fixed on the render path.

ARCHIVE CONTENTS
=================

  1  python_generator     the generator, its audits and its docs at root
                          level. NO C++. Includes png_compare.py and the
                          film_sim.py that now understands -8bpp.
  2  generated_database   the 26 generated C++/HPP files plus the name,
                          display-order and migration tables.
  3  algorithm_scalar     include/ + src/, the double build. Now carries
                          AlgoReciprocity.hpp and AlgoProcessVariant.hpp.
  4  algorithm_avx2       include/ + src/, the float build. Same two headers,
                          plus the self-contained Exp2Accurate.
  5  documentation_md     the whole doc/ tree.

⚠ NO TEST SOURCE APPEARS IN ANY OF THE FIVE. The thirteen test translation
units, and profall.cpp -- a profiling harness with its own main, which would
put a second entry point in the build -- are in the separate optional archive.
The .txt files ship inside include/ in archives 3 and 4, where the generated
database headers that read them at run time also live.
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

    for z in (z1, z2, z3, z4, z5):
        with zipfile.ZipFile(z) as zf:
            print(f"  {z.name:44s} {z.stat().st_size/1024:9.1f} kB  "
                  f"{len(zf.namelist()):4d} files")
    print(f"[OK] five archives in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
