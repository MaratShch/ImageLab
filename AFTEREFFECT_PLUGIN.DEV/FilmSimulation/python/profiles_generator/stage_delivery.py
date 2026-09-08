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
OUT = Path("/root/work/deliver8")
STAMP = date.today().isoformat()

#: Generator sources. Everything needed to regenerate the database and to run
#: every audit out of the box -- which is why the GENERATED artefacts ship in
#: here too: `cpp_parity.py`, `interimage_parity.py`, `spectral_mono_parity.py`
#: and `field_coverage.py` all compile against them.
PY_GLOBS = ("*.py", "*.txt", "*.lock")

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

1_python_generator.zip     generator, schema, every reader, every audit, plus
                           the generated C++ so the audits run out of the box
2_generated_database.zip   the generated database in the FilmProfile/ layout
3_algorithm_scalar.zip     the scalar engine  -- AlgoType = double
4_algorithm_avx2.zip       the AVX2 engine    -- AlgoType = float
5_documentation_md.zip     every Markdown document, reviewed against this build

⚠⚠ TWO THINGS TO DO BEFORE YOU BUILD
=====================================

(1) ADD TWO FILES TO THE VS2015 .vcxproj BY HAND
        film_profiles_data_19.cpp
        film_profiles_data_20.cpp
    Your tree at CPP/Algorithm/FilmProfile currently has 18. Nothing in the
    repository can edit the project file. If either is missed the plugin fails
    at LINK time with an unresolved AppendFilmProfiles_19 / _20 -- a loud
    failure, never a wrong render. CMake globs and needs no change.

(2) REMAP EVERY SAVED PROJECT -- see film_id_migration.txt in archive 2.
    177 of 184 stocks have a different index than in the previous delivery.

⚠ WHY THE 20th SLOT WHEN NO DATA WAS ADDED. Byte-for-byte the same database,
1 999 173 bytes. The packer assigns CONSECUTIVE slices over indivisible
per-stock blocks, so feasibility at a fixed slot count depends on the ORDER:
    19 slots, alphabetical : minimum feasible maximum 113 878   OVER
    20 slots, alphabetical : minimum feasible maximum 105 250   OK
against a 112 000-byte limit. 19 is not tight, it is INFEASIBLE. Trimming
~1.9 kB of prose would have restored it with ~2 kB of margin, i.e. the next
stock would break it again, so the count was raised instead.

1. THE DATABASE IS NOW STORED ALPHABETICALLY
============================================
film_names.txt is the full natural-name sort of all 184 stocks, and line k IS
database index k IS the eFILM_PROFILE value. No runtime reorder file is needed.

    AGFA      18 stocks   rows   1- 18
    EASTMAN   17 stocks   rows  22- 38
    FUJI      20 stocks   rows  45- 64
    KODAK     54 stocks   rows  86-139
    SVEMA     15 stocks   rows 167-181

The stocks added over the last weeks are interleaved, not appended: AGFA VISTA
PLUS 200/400 sit at 16 and 17 directly after AGFA VISTA 200, and the AGFA RSX II
and FUJI PROVIA / SUPERIA / PORTRA additions likewise.

film_display_order.txt is still emitted but is now the IDENTITY permutation.
You do not need it at run time. It is kept as a checkable invariant: if it is
ever not 0,1,2,...,183 the sort has broken and generation stops.

⚠ THE COST, AND IT RECURS. The frozen-id scheme is retired by your decision, so
a future "AGFA APX 50" will take index 1 and shift 183 stocks. That is inherent
to storing a sorted order, not a bug. film_ids.lock is kept ONLY as the source
of the migration table; verify.py still pins its first 161 rows by SHA-256 so
the migration source itself cannot be edited unnoticed.

2. AGFACOLOR -> AGFA
====================
Three profile KEYS renamed, at your request:
    AGFACOLOR_NEG_TYPE_3       -> AGFA_NEG_TYPE_3         (index 3)
    AGFACOLOR_NEG_TYPE_B_1943  -> AGFA_NEG_TYPE_B_1943    (index 4)
    AGFACOLOR_NEU_1936         -> AGFA_NEU_1936           (index 5)
They now sort INSIDE the AGFA block instead of after it.

⚠ ONLY THE KEY CHANGED. Every citation still reads AGFACOLOR -- "Agfacolor
Neu", "AGFACOLOR Vista ... Technical Data AF" -- because that is what Agfa
printed and a citation that no longer matches its document is worthless. The
process constants in agfa_mpt_1937.py are untouched for the same reason: they
name the 1936 PROCESS, not a profile. The migration table marks these three
"# RENAMED from ..." so a rename is never read as a withdrawal.

3. THE COLOUR CAST YOU RENDERED -- FIXED, AND IT WAS MINE
==========================================================
You asked whether the new AGFA and FUJI metrics were captured correctly.
THEY WERE. The new negatives render at R/B 1.58-1.70 with zero clipping,
indistinguishable from AGFA VISTA 200 (1.63) and KODAK PORTRA 400 (1.57).

What was wrong was the engine I shipped that morning. Measured on your own
frame, mean R/B and share of pixels at clip:

    84 colour NEGATIVES                       R/B med 1.52  max 1.70  clip 0.0%
    31 REVERSAL, old sign                      R/B med 1.41  max 1.60  clip 2.6%
    31 REVERSAL, sign fixed / solver NOT        R/B med 1.63  max 3.37  clip 23.7%
    31 REVERSAL, both halves fixed             R/B med 1.62  max 2.88  clip 5.0%

    FUJI PROVIA 400F   23.7% of frame clipped -> 0.0%
    FUJI PROVIA 100F   R/B 3.37, 12.6% clip   -> 2.11, 0.1%
    FUJICHROME 64T II  R/B 2.76, 2.5% clip    -> 1.98, 0.0%

I had measured that overshoot on 2026-09-08 and shipped the sign fix anyway,
holding the solver half for your approval. That was the wrong call: a measured
side effect on named stocks is not something to hold while shipping the half
that causes it.

WHAT WAS ACTUALLY BROKEN -- three defects in _iie_measure, each hidden by the
next:
  (a) it modelled only the NEGATIVE branch, at density_weighting 0, while the
      renderer runs reversal with the negation and dw = 0.65;
  (b) the measurement window sat at logE 0, which is nowhere near a reversal
      stock's scale -- PROVIA 400F's toe_x is 1.473, so both ends of the window
      were below the toe and both gammas came out ~0. The reference is now the
      one US5273870A names: the log exposure putting the channel at
      dmin + 1.0, "density 1.0 over fog";
  (c) the return line read `if g_white > 1e-9`. A REVERSAL GAMMA IS NEGATIVE,
      so that test rejected every reversal channel and returned 0.00 for all of
      them -- a flat objective, on which the solver diverged to the clamp.

RESULT: all 106 colour stocks now land within 0.088 percentage points of their
published US5273870A target (median 0.055, none over 0.5). Before, five stocks
were over 100 pp out, worst FUJICHROME 64T II at 190/170/113 against 35/33/25.

⚠ THE NEGATIVES ARE NOT BIT-IDENTICAL, and I would rather say so than imply
otherwise. Their reference and weighting are pinned to the old behaviour on
purpose, but the solver damping was retuned (0.5 -> 0.35) and its iteration cap
raised (40 -> 200) because the overshooting stocks had 155 pp to travel. Both
settings stop at the same 0.15 pp tolerance, so negatives land at a slightly
different point in the same band: coefficient drift median 0.11%, max 0.52%,
and the rendered census above is unchanged to two decimal places.

⚠⚠ 4. TWO STORED CURVES THAT NO EMULSION CAN HAVE -- NOT FIXED
===============================================================
Found while chasing your cast. Of 115 colour profiles, median green gamma 0.62,
exactly two exceed 2.5:

    KODAK_EKTACHROME_100D_5285   gamma_g 15.43   <- 6x the next, 25x the median
    SUPER_ANSCOCHROME_1957       gamma_g  5.27

The steepest colour reversal film ever sold is about 2.5; AGFA RSX II 50 is
2.42; the steepest monochrome stock here is POLAROID 51 at 3.35. 15.43 is not a
characteristic curve.

⚠ BOTH WERE AMONG THE FIVE OVERSHOOTING STOCKS. The solver was being asked to
hit a published figure on a curve that cannot exist, and it obliged by driving
the coefficient to an extreme -- so part of what I diagnosed as a solver gap
was bad traced data wearing a solver defect's clothes. With the solver fixed
they now hit target ON AN IMPOSSIBLE CURVE, which is worse than failing.

NOTHING WAS EDITED. No gamma was clamped or nudged toward plausibility -- a
guessed curve is worse than a wrong one, because it stops looking wrong. The
remedy is a RE-TRACE of both characteristic curves; the sheets are already in
the corpus. verify.py G-GAMMA pins both values and fails if a third stock joins
them or if either moves, so the pair cannot drift or be forgotten.

RESIDUAL, reported not fixed: FUJI VELVIA 50 still renders R/B 2.88 with 5.0%
clipped against a reversal median of 1.62 -- the highest once the above is
fixed. Velvia is genuinely the most saturated reversal film ever sold, so a
high figure is expected; nothing here measures whether 1.8x the class median is
wrong. Recorded so it is not mistaken for a closed question.

5. CARRIED OVER FROM THE MORNING DELIVERY
=========================================
Stage 8b's reversal sign is corrected in all three engines -- film_sim.py,
Algo_08_Sim.cpp and AVX2/Algo_08_Sim.cpp (main body AND masked tail). The AVX2
twin of stages 8b and 9 is now compiled and compared numerically on every build
for the first time (it previously had only a textual token grep):
    scalar  AlgoType 8 bytes  tol 2e-06  worst 4.965e-05
    AVX2    AlgoType 4 bytes  tol 2e-03  worst 5.078e-05
field_coverage.py re-derives the database-to-algorithm census every build:
    273 computational fields | Python 131  C++ 136  both 129  neither 135
dye_matrix is settled as M_reader . M_status^-1; verify.py's false rationale is
replaced by the property that holds plus a pinned neutral-shift census. NO
stored matrix changed.

BUILD STATE
===========
build.py           OK -- 0 failures
verify.py          647 PASS / 1 FAIL  (the baselined saturation-hierarchy
                   failure you instructed not to fix)
G-FILMORDER        NEW -- the database IS in natural-name order, and
                   film_names.txt line k is database index k, checked against
                   the emitted file rather than a re-derivation
G-GAMMA            NEW -- the two impossible curves, pinned
G-FILMID           checks 3 and 5 replaced (the freeze's invariants are now
                   false by design); check 5 asserts the migration table is
                   complete and correct; the SHA-256 pin on the lock is intact
cpp_parity         OK -- grain, MTF, reciprocity, Callier, whole database
interimage_parity  OK -- stages 8b and 9, BOTH flavours
bromide_parity     OK -- scalar and AVX2
spectral_mono      OK -- 69/69 monochrome stocks agree exactly
doc_consistency    OK -- every registered documentation count matches
compile            OK -- g++ -std=c++14 -Wall -Wextra, 22 TUs, zero output
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
    py = [p for g in PY_GLOBS for p in HERE.glob(g)]
    py += sorted(HERE.glob("film_profiles_data_*.cpp"))
    py += [HERE / n for n in DB_FILES if (HERE / n).is_file()]
    py = [p for p in dict.fromkeys(py) if p.is_file()]
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
    docs = sorted((HERE / "doc").rglob("*.md")) + \
        [p for p in (HERE / "doc").rglob("*.txt") if p.is_file()]
    docs += [OUT / "MANIFEST.txt"]
    z5 = _zip(f"5_documentation_md_{STAMP}.zip", HERE.parent,
              [p for p in docs if p.is_file() and HERE.parent in p.parents])

    for z in (z1, z2, z3, z4, z5):
        with zipfile.ZipFile(z) as zf:
            print(f"  {z.name:44s} {z.stat().st_size/1024:9.1f} kB  "
                  f"{len(zf.namelist()):4d} files")
    print(f"[OK] five archives in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
