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
STAMP = date.today().isoformat() + 'd'

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

Schema v33 unchanged. Verify 696 PASS / 1 baselined FAIL, build clean, all
four engine parity audits green.

⚠⚠ THE QUEUE WENT DOWN FOR THE FIRST TIME IN FIVE PASSES: 30 LIVE -> 28
=======================================================================

P41 through P44 were all opened on 2026-09-11 and the instruction the same day
was blunt: stop adding rows, start resolving them. P43 and P44 are CLOSED, and
⚠ NEITHER CLOSED THE WAY ITS OWN ROW PREDICTED. Both were blocked on their own
wrong premise rather than on the thing they named, which is the argument for
re-reading a blocker before going out and sourcing it.

P43 -- SETTLED FROM WITHIN THE DATABASE, AND THE DOCUMENT IT ASKED FOR WOULD
NOT HAVE SETTLED IT
============================================================================

The row wanted a SILVER-ONLY CLUMP MEASUREMENT so that `dye_cloud_um` would
have something to add to. That measurement is not needed: the silver-only law
can be derived here, because ⚠ the monochrome stocks carry dye_cloud_um = 0 BY
DEFINITION, so their clump IS silver-only. Fitting it against crystal size over
65 monochrome stocks gives

    clump = 3.93 x grain_um

⚠ THE TEST IS NOT CIRCULAR: clump_um is not derived from grain_um anywhere in
the generator, and the 170 stocks carrying both produce 130 distinct ratios.

THREE RESULTS, ALL AGREEING:

  1. Colour sits at the SAME ratio as monochrome. log(clump/grain) is 1.369 on
     65 monochrome against 1.297 on 103 colour -- t = -0.65, indistinguishable.
     A real extra dye spread demands colour sit systematically HIGHER.
  2. Adding the dye term is a COIN FLIP. Predicting each colour clump as
     sqrt((3.93*grain)^2 + dye^2) against the silver-only 3.93*grain improves
     51 stocks and worsens 52. It carries no information.
  3. ⚠ The two stocks with the LARGEST dye cloud contradict it hardest.
     KODAK_BW400CN and KODAK_T400CN are chromogenic black-and-white -- their
     image really is dye and nothing else, which is why they carry 9.0 um --
     and their clump/grain is 2.5, the LOWEST in the set, where the hypothesis
     needs the highest.

⚠⚠ AND THE FIELD IS NOT A MEASUREMENT ANYWAY: four distinct values across 115
stocks (1.5 / 2.0 / 2.5 and the two chromogenic 9.0s) with no per-stock
provenance. It is an era band. Pairing a real silver-only measurement with a
three-level guess would still be pairing a measurement with a guess.

So the answer is not "acquire a document" but "the test that document was
wanted for runs today and comes out null". G-DYECLOUD-INERT asserts the field
stays unread, and exists because the next reader will have the same good idea.

P44 -- A SCRATCH IS NOT BORN AND DOES NOT DIE, SO NO BIRTH/DEATH MODEL WAS
NEEDED. WHAT WAS MISSING WAS THE SCRATCH CLASS ITSELF
==========================================================================

The row assumed persistence had to be a lifetime on a static population. ⚠ The
field's own docstring says "mean lifetime of a RUNNING scratch" -- abraded by
continuous contact as the film passes, so it is a RUN LENGTH ALONG THE WEB, and
stage 9b already translates defects with the web. The coordinate was there.

⚠⚠ WHAT WAS ACTUALLY MISSING: stage 9b modelled three PARTICULATE classes --
dust, debris, fibres -- and there was NO SCRATCH GENERATOR ANYWHERE IN EITHER
ENGINE. The field had nothing to attach to, which the row read as a missing
model when it was a missing class.

NOW IMPLEMENTED IN BOTH TWINS, using only figures already measured and recorded
in the tree: width 26 um, straightness 0.98, the 3.5:1 longitudinal bias, median
contrast 3.5 %. Two populations, on the two controls that were already in the
layout and unconsumed:

    scratchTransport   a TRAMLINE locked to the transport axis that holds its
                       across-web position while the picture moves past it, for
                       scratch_persistence_frames frames
    scratchHandling    shorter unlocked single-event marks, no persistence

⚠ THE STRAIGHTNESS FIGURE VALIDATED THE EXISTING FIBRE CLASS ON THE WAY PAST.
Deriving the walk's persistence length from 0.98 as a 2-D worm-like chain gives
Lp = 2.5 mm, and a median 4 mm fibre then comes out at 0.87 -- mid-range of the
0.7-0.95 that `defectFibres` claims for itself. The two classes are now
separated by the one measured quantity rather than by assertion.

Controls consumed 9 -> 11, unconsumed 8 -> 6. Four guards:
G-SCRATCH-CONSTANTS / -GATE / -TRAMLINE / -POLARITY, the tramline one asserting
a run holds one column set for 41 consecutive frames.

⚠ SEVEN MODELLING CHOICES THE SOURCES DO NOT SETTLE are each named AS choices
at the constant rather than presented as fact -- among them that the transport
scratch is drawn exactly straight (0.98 over a multi-frame run implies a lateral
excursion wider than the web), that the cut-versus-burnish share is a
maximum-entropy 0.5 because nothing measures it, and that the 3.5 % contrast is
used unsolved in the negative-density domain because no scratch amplitude
distribution exists to solve against. The likely direction of that last error --
too strong, by roughly the print gamma -- is stated at the constant rather than
hidden.

VERIFICATION
=============

    verify.py            696 PASS / 1 baselined FAIL   (691 before; +5 guards)
    stage_parity.py      125/125 planes, 0 pinned exceptions
    interimage_parity    2 flavours, worst 5.221e-05
    bromide_parity       both builds, both directions, both scales
    spectral_mono_parity 69/69 monochrome stocks exact
    compile              all 26 TUs, exit 0 and zero bytes of output

⚠ The scratch class is LIVE at the default control values, so stage 9b renders
differently from the previous set on every stock. That is the point of the
change, but it is a visible difference and not a silent one.

ARCHIVE CONTENTS
=================

  1  python_generator     the generator, its audits and its docs at root
                          level. NO C++. Carries stage_parity.py,
                          png_compare.py and the film_sim.py with -8bpp.
  2  generated_database   the 26 generated C++/HPP files plus the name,
                          display-order and migration tables.
  3  algorithm_scalar     include/ + src/, the double build.
  4  algorithm_avx2       include/ + src/, the float build.
  5  documentation_md     the whole doc/ tree.

⚠ NO TEST SOURCE IN ANY OF THE FIVE -- the test translation units, profall.cpp
and the two harnesses are in the separate optional archive. The .txt files ship
inside include/ in archives 3 and 4.
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
