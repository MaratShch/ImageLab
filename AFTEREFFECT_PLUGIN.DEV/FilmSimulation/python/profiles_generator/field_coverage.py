#!/usr/bin/env python3
"""Database -> algorithm coverage census, derived rather than remembered.

⚠ WHY THIS EXISTS AS CODE AND NOT AS A MARKDOWN TABLE. `DB_ALGORITHM_COVERAGE`
was written by hand on 2026-09-03 at schema v24 / 175 stocks. It was correct,
it was useful, and by 2026-09-08 it was four schema versions and nine stocks out
of date with no way to tell WHICH of its rows had moved. A census whose only
copy is prose goes stale silently; one that is re-derived on demand cannot. The
prose document keeps its job -- the JUDGEMENT about what each gap means and what
it would cost -- and this module supplies the counts underneath it.

THE THREE PASSES, and the trap each one exists to avoid. All three are the same
method the hand census used, because that method was right; what is new is that
a machine runs it.

  1. ENUMERATION by `dataclasses.fields()` over every dataclass in
     `film_profiles.py`. Never by reading source with eyes, which is how a field
     added last week gets missed.

  2. EMISSION by parsing the generated `film_profiles.hpp` and
     `film_profiles_detail.hpp`. A field that is not in the struct CANNOT be
     read by any C++ stage whatever the engine does, so this pass is decisive in
     one direction and says nothing in the other.

  3. CONSUMPTION by spelling, across `film_sim.py` and every engine translation
     unit. This is a REACHABILITY FLOOR and is reported as such:

       * `.field` alone misses `->field`. The print chain reaches
         `pPrintStock->grain_rms` that way, and a dot-only scan once reported
         the whole print stock as unread -- six fields wrong from one omission.
         Both spellings are searched.
       * A field can be consumed WITHOUT ITS NAME APPEARING. `kind` is reached
         only through `profile.isReversal()`; `mtf_rolloff_q` only through
         `fp.mtf_response(spec, ...)`, which is handed the whole `MTFSpec`; the
         `sigma_shape_*` septet only through `fp.grain_sigma()` and its C++ twin
         `AlgoGrainAmpBuild`. Those are declared in `INDIRECT` below with the
         call site that reaches them, because no grep can find them and a census
         that reported them as gaps would send the next reader to fix code that
         is already correct.
       * A name in a COMMENT counts as a hit. That is the floor's known
         weakness, and it is why `INDIRECT` is a declaration list rather than a
         suppression list: every non-trivial verdict here was confirmed by
         reading the call site.

  A fourth number is reported beside those three and is not a pass at all:
  HOW MANY STOCKS ACTUALLY CARRY DATA in the field. An unconsumed field that is
  zero on all 184 stocks is a dormant generalisation; one populated on all 184
  is a live gap. Ranking without it produced the 2026-09-01 document's top item
  being a field nothing could ever fill.
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import inspect
import re
import sys
from pathlib import Path

import film_profiles as fp

HERE = Path(__file__).resolve().parent

#: Fields that carry no number a render could read: names, citations, prose,
#: provenance. Requirement 1 of the 2026-09-08 owner brief exempts exactly
#: these ("Informational fields and descriptive strings are the only
#: exceptions"), so they are excluded from the denominator rather than counted
#: as gaps.
#:
#: ⚠ MEMBERSHIP IS BY ROLE, NOT BY TYPE. `density_metric` is a string and is NOT
#: here: it decides how stage 12's matrix must be interpreted, which is what
#: settled the `dye_matrix` question on 2026-09-08. `name` is a string and IS
#: here. When in doubt the field stays OUT of this set, because a false
#: informational label hides a real gap and a false gap only costs a reading.
INFORMATIONAL: frozenset[str] = frozenset((
    "name", "manufacturer", "era", "notes", "note", "source", "sources",
    "citation", "conditions", "unit", "status", "tier", "confidence",
    "param", "designation", "emulsion_code", "process_name", "developer",
    "agitation", "description", "label", "comment", "provenance",
    "fitted_from", "display_name", "family", "figure", "page",
    # ⚠ THESE FOUR ARE REACHED BY PYTHON AND STILL BELONG HERE, because the
    # only thing that reads them is a REPORT, not a render. Counting them as
    # "consumed" inflated the Python column and counting them as gaps would
    # send someone to wire a debug string into the pixel path.
    #   aliases            film_sim.py:3460, a print() in the CLI listing
    #   criterion          film_sim.py:996, into an informational dict
    #   measured_through   never read; a note on how a curve was measured
    #   last_reviewed      a provenance date
    "aliases", "criterion", "measured_through", "last_reviewed",
    "speed_criterion", "normalisation",
    # ⚠ `param_sources` IS THE PROVENANCE REGISTER ITSELF -- 1958 records of
    # prose emitted into the C++ as untruncated std::string. It ships, it is
    # read by people, and no render reads a number out of it. It topped the
    # "consumed by neither" table on the first run purely because it is present
    # on 184/184 stocks, which is exactly the kind of false lead this set
    # exists to remove.
    "param_sources",
))

#: Fields reached WITHOUT their name appearing at the call site, with the call
#: that reaches them. Every entry was confirmed by reading that call, not by
#: grepping for it. Format: field -> (python_reached, cpp_reached, how).
INDIRECT: dict[str, tuple[bool, bool, str]] = {
    "kind": (True, True, "profile.is_reversal / profile.isReversal()"),
    "mtf_rolloff_q": (True, True,
                      "fp.mtf_response(spec, ...) takes the whole MTFSpec; "
                      "C++ FilmMtfKernel is keyed on the stored q"),
    "mtf_measured": (True, True, "same call as mtf_rolloff_q"),
    "mtf_tail_a": (True, True, "same call as mtf_rolloff_q"),
    "mtf_tail_f_exp": (True, True, "same call as mtf_rolloff_q"),
    "sigma_shape_toe": (True, True, "fp.grain_sigma() / AlgoGrainAmpBuild"),
    "sigma_shape_mid": (True, True, "fp.grain_sigma() / AlgoGrainAmpBuild"),
    "sigma_shape_dmax": (True, True, "fp.grain_sigma() / AlgoGrainAmpBuild"),
    "sigma_shape_peak": (True, True, "fp.grain_sigma() / AlgoGrainAmpBuild"),
    "sigma_shape_peak_at": (True, True, "fp.grain_sigma() / AlgoGrainAmpBuild"),
    "sigma_shape_toe_at": (True, True, "fp.grain_sigma() / AlgoGrainAmpBuild"),
    "sigma_shape_dmax_at": (True, True, "fp.grain_sigma() / AlgoGrainAmpBuild"),
    "sigma_shape_measured": (True, True,
                             "GrainSpec.sigma_measured_usable(), which "
                             "fp.grain_sigma() calls to pick the law"),
    "fog_grain": (True, True,
                  "fp.grain_sigma()'s LEGACY branch -- "
                  "sqrt(max(D-dmin,0)+fog)/sqrt(1+fog), the path 144 of 155 "
                  "stocks take; C++ AlgoGrainAmpBuild has the same branch"),
    # ⚠ THESE NINE WERE REPORTED AS C++-ONLY BY THE FIRST RUN OF THIS CENSUS
    # AND THAT WAS THE CENSUS BEING WRONG, NOT THE ENGINE. film_sim.py reaches
    # every one through an ACCESSOR METHOD on the spec, so the field name never
    # appears at the call site while the C++ reads members directly. Confirmed
    # by reading each call, with the line noted.
    "clump_um_r": (True, True, "GrainSpec.clumps() at film_sim.py:3031"),
    "clump_um_g": (True, True, "GrainSpec.clumps() at film_sim.py:3031"),
    "clump_um_b": (True, True, "GrainSpec.clumps() at film_sim.py:3031"),
    "rms_r": (True, True, "GrainSpec.rms_rgb() at film_sim.py:3048"),
    "rms_g": (True, True, "GrainSpec.rms_rgb() at film_sim.py:3048"),
    "rms_b": (True, True, "GrainSpec.rms_rgb() at film_sim.py:3048"),
    "gain_r": (True, True, "HalationSpec.gains() at film_sim.py:2781"),
    "gain_g": (True, True, "HalationSpec.gains() at film_sim.py:2781"),
    "gain_b": (True, True, "HalationSpec.gains() at film_sim.py:2781"),
    "f50_r": (True, True, "fp.mtf_response(spec, ...) at film_sim.py:1180, "
                          "handed the whole MTFSpec"),
    "f50_g": (True, True, "fp.mtf_response(spec, ...) at film_sim.py:1180"),
    "f50_b": (True, True, "fp.mtf_response(spec, ...) at film_sim.py:1180"),
    # The six interimage coefficients are read through InterimageSpec.matrix(),
    # which assembles the 3x3 with its structurally zero diagonal.
    # `film_sim.py:1579  m = iie.matrix()`.
    "a_rg": (True, True, "InterimageSpec.matrix() at film_sim.py:1579"),
    "a_rb": (True, True, "InterimageSpec.matrix() at film_sim.py:1579"),
    "a_gr": (True, True, "InterimageSpec.matrix() at film_sim.py:1579"),
    "a_gb": (True, True, "InterimageSpec.matrix() at film_sim.py:1579"),
    "a_br": (True, True, "InterimageSpec.matrix() at film_sim.py:1579"),
    "a_bg": (True, True, "InterimageSpec.matrix() at film_sim.py:1579"),
    # ⚠ THE FIRST RUN CALLED THESE TWO PYTHON-ONLY AND THAT WAS WRONG IN THE
    # OTHER DIRECTION -- the C++ reads them under a FLATTENED name. Python has
    # `profile.taking_filter.cut_on_nm`; the generated struct carries the
    # scalar `profile.taking_filter_cut_on_nm`, and
    # `AlgoSpectralMonoWeights()` reads it TWICE: once in
    # `panWithinBasisReach()` and once to zero the sub-cut-on samples on the
    # integration path. The header says why both are needed -- the guard and
    # the collapse must judge the same filtered emulsion.
    "cut_on_nm": (True, True,
                  "Py taking_filter_transmission(); C++ reads the flattened "
                  "profile.taking_filter_cut_on_nm in AlgoSpectralMonoWeights, "
                  "in the reach guard AND on the integration path"),
    "taking_filter": (True, True,
                      "same flattening: the struct is emitted, the scalar is "
                      "what the engine reads"),
}

#: Field names too short or too generic for a spelling search to mean anything.
#: ⚠ THIS IS NOT TIDINESS. `DyeImpurityRatio.lo` and `.hi` were reported as
#: C++-only by the first run on the strength of `lut.lo` in AlgoCallier.hpp and
#: AlgoCurveLut.hpp -- an unrelated LUT member. A two-letter name will collide
#: with something in 84 translation units every time, so the verdict is
#: withheld and the row is reported as UNRESOLVED rather than guessed. Resolve
#: one by reading the call site and moving it into INDIRECT.
TOO_SHORT_TO_GREP: frozenset[str] = frozenset(("lo", "hi", "d", "q", "a", "n"))

#: Fields that are build-time inputs, not render inputs, and must not be
#: counted as gaps. `features` is the documented case: film_profiles.py calls it
#: "a convenience summary of the numeric fields", and the flags are read only by
#: the schema helpers that SET those numeric fields at construction.
BUILD_TIME: frozenset[str] = frozenset(("features",))

#: Where the engine lives. `/root/work/sc` and `/root/work/av` are STALE PARTIAL
#: COPIES with no AlgoReciprocity.hpp at all -- scanning them reports
#: reciprocity as a C++ gap, which it stopped being on 2026-09-01. The synced
#: tree is the only valid target and the default reflects that.
DEFAULT_CPP_ROOT = Path("/root/work/proot")


def enumerate_fields() -> list[tuple[str, str, object]]:
    """(dataclass name, field name, type) for every dataclass in the schema."""
    out = []
    for cname, cls in sorted(vars(fp).items()):
        if not inspect.isclass(cls) or not dc.is_dataclass(cls):
            continue
        if cls.__module__ != fp.__name__:
            continue
        for f in dc.fields(cls):
            out.append((cls.__name__, f.name, f.type))
    # A field name can appear on more than one dataclass; dedupe on the pair.
    seen, uniq = set(), []
    for row in out:
        if (row[0], row[1]) in seen:
            continue
        seen.add((row[0], row[1]))
        uniq.append(row)
    return uniq


def emitted_names(root: Path) -> set[str]:
    """Every identifier declared as a struct member in the generated headers."""
    names: set[str] = set()
    for h in ("film_profiles.hpp", "film_profiles_detail.hpp"):
        p = HERE / h
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        # `<type> name;` and `<type> name[N];` -- member declarations only.
        for m in re.finditer(r"^\s+[A-Za-z_][\w:<>,\s\*&]*?\b(\w+)\s*(\[[^\]]*\])?\s*;",
                             text, re.M):
            names.add(m.group(1))
    return names


def _spellings(field: str) -> re.Pattern:
    """`.field`, `->field`, and the bare word inside a designated initialiser."""
    return re.compile(r"(?:\.|->)%s\b" % re.escape(field))


def consumed_in(paths: list[Path], field: str) -> bool:
    pat = _spellings(field)
    for p in paths:
        try:
            if pat.search(p.read_text(encoding="utf-8", errors="replace")):
                return True
        except OSError:
            continue
    return False


def populated(cls_name: str, field: str) -> tuple[int, int]:
    """(stocks carrying a non-default value, stocks examined).

    Walks each profile for a member of the named dataclass and compares the
    field against that dataclass's own default. A field equal to its default on
    every stock is dormant, which is a completely different thing from a gap.
    """
    cls = getattr(fp, cls_name, None)
    if cls is None or not dc.is_dataclass(cls):
        return (0, 0)
    default = None
    for f in dc.fields(cls):
        if f.name != field:
            continue
        if f.default is not dc.MISSING:
            default = f.default
        elif f.default_factory is not dc.MISSING:      # type: ignore[misc]
            try:
                default = f.default_factory()          # type: ignore[misc]
            except Exception:
                default = None
    # ⚠ THE PROFILES ARE `slots=True`, SO THERE IS NO `__dict__` TO WALK. The
    # first version of this used vars() and died on every stock. Members are
    # found through dataclasses.fields(), which is the only reliable walk over a
    # slotted frozen tree, and the search is one level deep plus tuples, which
    # is as deep as the schema nests a holder of a computational field.
    def holders(obj, depth=0):
        if isinstance(obj, cls):
            yield obj
        if depth > 2 or not dc.is_dataclass(obj):
            return
        for f in dc.fields(obj):
            v = getattr(obj, f.name, None)
            if dc.is_dataclass(v):
                yield from holders(v, depth + 1)
            elif isinstance(v, tuple):
                for item in v:
                    if dc.is_dataclass(item):
                        yield from holders(item, depth + 1)

    def differs(val) -> bool:
        if default is None:
            return val not in (0, 0.0, "", None, (), [], False)
        try:
            return bool(val != default)
        except Exception:
            return val is not default

    n = seen = 0
    for p in fp.FILM_PROFILES:
        got = None
        for holder in holders(p):
            if not hasattr(holder, field):
                continue
            got = getattr(holder, field)
            break
        if got is None and not any(hasattr(h, field) for h in holders(p)):
            continue
        seen += 1
        if differs(got):
            n += 1
    return (n, seen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(DEFAULT_CPP_ROOT),
                    help="engine tree holding the Algo_* sources")
    ap.add_argument("--gaps-only", action="store_true")
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    help="fail if the engine tree is missing (never on a gap: "
                         "a gap is a research state, not a defect -- rule 23)")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()

    py_paths = [HERE / "film_sim.py"]

    # ⚠ `Path.rglob` DOES NOT DESCEND THROUGH A SYMLINKED DIRECTORY on this
    # Python, and `proot/AVX2` IS a symlink to `tst/AVX2`. The first run of this
    # census therefore scanned 84 translation units and NONE of the vector
    # twins -- an audit of engine coverage that could not see half the engine.
    # The AVX2 tree is walked explicitly, and the count is printed split so the
    # omission cannot recur silently.
    def _sources(d: Path) -> list[Path]:
        return sorted(p for p in d.rglob("*")
                      if p.is_file() and p.suffix in (".cpp", ".hpp", ".inl")
                      and p.name not in ("film_profiles.hpp",
                                         "film_profiles_detail.hpp")
                      and not p.name.startswith("film_profiles_data_"))

    cpp_paths = _sources(root)
    avx_dir = (root / "AVX2").resolve()
    n_avx = 0
    if avx_dir.is_dir():
        avx = [p for p in _sources(avx_dir) if p not in cpp_paths]
        n_avx = len(avx)
        cpp_paths = sorted(cpp_paths + avx)
    if not cpp_paths:
        print(f"  [SKIP] no engine sources under {root}")
        if ns.do_assert:
            return 1
        return 0

    emitted = emitted_names(root)
    fields = enumerate_fields()

    rows = []
    for cls_name, field, _t in fields:
        if field in INFORMATIONAL:
            kind = "informational"
            py = cpp = None
        elif field in BUILD_TIME:
            kind = "build-time"
            py = cpp = None
        else:
            kind = "computational"
            if field in INDIRECT:
                py, cpp, _how = INDIRECT[field]
            elif field in TOO_SHORT_TO_GREP:
                kind = "unresolved"
                py = cpp = None
            else:
                py = consumed_in(py_paths, field)
                cpp = consumed_in(cpp_paths, field)
        have, seen = populated(cls_name, field)
        rows.append((cls_name, field, kind, py, cpp,
                     field in emitted, have, seen))

    comp = [r for r in rows if r[2] == "computational"]
    n_py = sum(1 for r in comp if r[3])
    n_cpp = sum(1 for r in comp if r[4])
    n_either = sum(1 for r in comp if r[3] or r[4])
    n_both = sum(1 for r in comp if r[3] and r[4])
    n_emit = sum(1 for r in comp if r[5])

    print(f"[i] schema v{getattr(fp, 'SCHEMA_VERSION', '?')}, "
          f"{len(fp.FILM_PROFILES)} stocks, "
          f"{len({r[0] for r in rows})} dataclasses, {len(rows)} fields "
          f"({len(comp)} computational, "
          f"{sum(1 for r in rows if r[2] == 'informational')} informational, "
          f"{sum(1 for r in rows if r[2] == 'build-time')} build-time)")
    print(f"[i] engine tree {root} -- {len(cpp_paths)} translation units "
          f"({len(cpp_paths) - n_avx} scalar + {n_avx} AVX2 twins)")
    _unres = [r for r in rows if r[2] == "unresolved"]
    if _unres:
        print(f"[i] {len(_unres)} field(s) UNRESOLVED -- name too short for a "
              f"spelling search to mean anything, verdict withheld rather than "
              f"guessed: "
              + ", ".join(f"{r[0]}.{r[1]}" for r in _unres))
    print()
    print(f"    computational fields          {len(comp):4d}")
    print(f"    emitted into the C++ headers  {n_emit:4d}")
    print(f"    consumed by Python            {n_py:4d}")
    print(f"    consumed by C++               {n_cpp:4d}")
    print(f"    consumed by BOTH              {n_both:4d}")
    print(f"    consumed by EITHER            {n_either:4d}")
    print(f"    consumed by NEITHER           {len(comp) - n_either:4d}")
    print()

    # ---- the asymmetries, which are the only true DEFECTS in this table -----
    # A field one engine reads and the other does not is a divergence: two
    # renderers, one name, two pictures. A field NEITHER reads is a research
    # state (rule 23) and is reported separately and never as a failure.
    only_py = [r for r in comp if r[3] and not r[4]]
    only_cpp = [r for r in comp if r[4] and not r[3]]
    print(f"[i] ASYMMETRIES -- Python only {len(only_py)}, C++ only {len(only_cpp)}")
    for tag, group in (("PY-ONLY", only_py), ("CPP-ONLY", only_cpp)):
        for cls_name, field, _k, _p, _c, emit, have, seen in group:
            print(f"  [{tag:8s}] {cls_name}.{field:32s} "
                  f"data {have}/{seen}"
                  + ("" if emit else "  ⚠ NOT EMITTED to C++"))
    print()

    unread = sorted((r for r in comp if not r[3] and not r[4]),
                    key=lambda r: (-r[6], r[0], r[1]))
    print(f"[i] CONSUMED BY NEITHER, ranked by stocks carrying data "
          f"({len(unread)} fields). ⚠ Rule 23: a gap here is a research state, "
          f"not a defect, and this audit never fails on one.")
    for cls_name, field, _k, _p, _c, emit, have, seen in unread:
        if ns.gaps_only and have == 0:
            continue
        flag = "" if emit else "  ⚠ not emitted"
        print(f"  {have:4d}/{seen:<4d} {cls_name}.{field}{flag}")
    dormant = sum(1 for r in unread if r[6] == 0)
    print()
    print(f"[i] of those {len(unread)}, {dormant} are DORMANT (zero stocks "
          f"carry a value) and {len(unread) - dormant} are LIVE gaps")
    # ⚠ ONE HONEST LIMIT OF THE POPULATION COLUMN, stated rather than papered
    # over: it walks FILM_PROFILES, and the PRINT STOCKS are a separate table.
    # Every `PrintStock.*` row therefore reports 0/0 -- "not examined", NOT
    # "no data". Their gaps are real and are tracked as the print-chain block
    # in DB_ALGORITHM_COVERAGE; do not read a 0/0 here as evidence of anything.
    print("[i] ⚠ PrintStock.* rows read 0/0 because this pass walks "
          "FILM_PROFILES only -- that is NOT EXAMINED, not 'no data'. The "
          "print chain is tracked separately.")
    print()
    print("[OK] census complete -- counts are a reachability floor; see the "
          "module docstring for what that does and does not prove")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
