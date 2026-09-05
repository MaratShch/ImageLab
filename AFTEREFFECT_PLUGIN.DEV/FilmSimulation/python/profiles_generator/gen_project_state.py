"""Generate `doc/PROJECT_STATE.md` -- the live state, derived, never written.

WHY THIS EXISTS
---------------
The owner's release requirement is that the Markdown be "a technical mirror of
the actual implementation, not an independent historical description". This tree
has ~90 Markdown files and most of them are correctly HISTORICAL: a dated
`RESULT_*` record states what was true on its date and must not be rewritten,
because it is the audit trail. Editing eighty narrative files to agree with the
tree would destroy that trail and would go stale again the next day.

⚠ SO THE MIRROR IS GENERATED AND THERE IS EXACTLY ONE OF IT. Every number in
`PROJECT_STATE.md` is read out of the live module, the live C++ sources or the
live queue at build time. It cannot drift, because nothing types it. The
narrative documents keep their job -- argument, history, reasoning -- and this
one carries the state they used to restate and get wrong.

The three bookkeeping failures this batch found were all the same shape: a
number restated in a second place and never re-derived (G5's row id, E4's
unstruck duplicate, five copies of the live-row count across four wordings).
This file is the structural answer to that class.

WHAT IT ASSERTS AS WELL AS REPORTS
----------------------------------
Generating is not enough on its own -- a mirror that reflects a broken tree is
still a broken tree. So the generator also runs the release gate and prints the
result: ordering identity across database / vector / enum / names file, the
scope-preservation census, and the provenance breakdown. `--assert` fails if any
of those checks fails, and `build.py` runs it that way.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import film_profiles as fp
from film_profiles import FILM_PROFILES, PRINT_STOCKS

HERE = Path(__file__).resolve().parent
OUT = HERE / "doc" / "PROJECT_STATE.md"

#: Carriers whose population is worth reporting: each is an OPTIONAL record that
#: a stock either has evidence for or does not. ⚠ A LOW COUNT HERE IS A RESEARCH
#: GAP AND NOT A DEFECT -- method rule 23. The column exists so the gap is
#: visible, not so it can be closed by inventing values.
CARRIERS = (
    ("spectral sensitivity", lambda p: p.spectral.has_data),
    ("spectral dye density (3 dyes)", lambda p: p.dye_density.has_data),
    ("dye density neutral + Dmin pair", lambda p: p.dye_density.has_neutral_pair),
    ("measured sigma(D) shape", lambda p: bool(getattr(p.grain, "sigma_shape_measured", False))),
    ("measured MTF", lambda p: bool(getattr(p.mtf, "mtf_measured", False))),
    ("reciprocity table", lambda p: p.reciprocity_table.has_data),
    ("processing family (time-gamma)", lambda p: bool(p.processing_family.points)),
    ("process variants", lambda p: bool(p.process_variants)),
    ("aim density", lambda p: bool(p.aim_density)),
    ("layer stack", lambda p: bool(p.layer_stack.order)),
    ("push spec", lambda p: bool(getattr(p.push, "has_data", False))),
    ("emulsion spec", lambda p: bool(getattr(p.emulsion, "has_data", False))),
    ("print grain index", lambda p: bool(getattr(p.print_grain_index, "has_data", False))),
    ("reseau (additive mosaic)", lambda p: p.reseau is not None),
    ("bromide drag (schema v27)", lambda p: p.processing.bromide_drag.has_data),
)


# ---------------------------------------------------------------------------
def ordering_check() -> tuple[list[str], list[str]]:
    """Database == emitted vector == enum == names file. (facts, failures)."""
    import cpp_codegen as cg
    facts, bad = [], []
    names = [p.name for p in FILM_PROFILES]
    try:
        vec = cg.parse_vector_names(HERE / "film_profiles.cpp")
    except Exception as e:                                   # pragma: no cover
        return [], [f"the emitted vector could not be read: {e}"]
    facts.append(f"`FILM_PROFILES` order == emitted `std::vector` order: "
                 f"**{vec == names}** ({len(vec)} entries)")
    if vec != names:
        bad.append("the emitted vector is not in FILM_PROFILES order")

    enum_txt = (HERE / "film_enum.hpp").read_text(encoding="utf-8")
    en = [(m.group(1), int(m.group(2)))
          for m in re.finditer(r"^\s*e([A-Z0-9_]+)\s*=\s*(\d+)", enum_txt, re.M)]
    body = [n for n, _ in en if n in set(vec)]
    sentinel = [(n, v) for n, v in en if n not in set(vec)]
    facts.append(f"`film_enum.hpp` order == vector order: **{body == vec}**, "
                 f"values consecutive from 0: "
                 f"**{[v for _n, v in en][:len(vec)] == list(range(len(vec)))}**")
    if body != vec:
        bad.append("film_enum.hpp is not in vector order")
    if sentinel:
        facts.append("plus the sentinel `e" + sentinel[0][0]
                     + f" = {sentinel[0][1]}`, which equals the profile count")
        if sentinel[0][1] != len(vec):
            bad.append("the enum sentinel does not equal the profile count")

    txt = [l for l in (HERE / "film_names.txt").read_text(encoding="utf-8")
           .splitlines() if l.strip()]
    norm = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper())
    same = (len(txt) == len(vec)
            and all(norm(t.strip('"').split("|")[0]) == norm(v)
                    for t, v in zip(txt, vec)))
    facts.append(f"`film_names.txt` is one line per vector entry, in vector "
                 f"order: **{same}** ({len(txt)} lines)")
    if not same:
        bad.append("film_names.txt does not match the vector line for line")

    ids = fp.FILM_IDS
    contiguous = sorted(ids.values()) == list(range(len(ids)))
    facts.append(f"`film_ids.lock` holds {len(ids)} frozen storage ids, "
                 f"contiguous from 0: **{contiguous}**  ⚠ these are STORAGE "
                 f"identity and are deliberately independent of display order")
    if not contiguous:
        bad.append("film_ids.lock is not contiguous")
    return facts, bad


def provenance_census() -> tuple[Counter, Counter, int]:
    st = Counter()
    par = Counter()
    for p in FILM_PROFILES:
        for s in p.param_sources:
            st[s.status] += 1
            par[s.param] += 1
    return st, par, sum(st.values())


def scope_check() -> tuple[list[str], list[str]]:
    """⚠ Nothing removed or disabled for want of data. (facts, failures)."""
    facts, bad = [], []
    n = len(FILM_PROFILES)
    ids = fp.FILM_IDS
    retired = [k for k in ids if str(k).startswith("RETIRED ")]
    facts.append(f"**{n} film stocks present**; `film_ids.lock` records "
                 f"**{len(retired)} retired** ids — a stock is never deleted, "
                 f"only appended")
    if retired:
        bad.append(f"a stock has been retired: {retired}")

    # every stock must still render: the four fields the renderer cannot do
    # without must be non-degenerate on every profile.
    broken = []
    for p in FILM_PROFILES:
        if p.grain.rms_granularity <= 0:
            broken.append(f"{p.name}: rms_granularity")
        if min(p.mtf.f50_r, p.mtf.f50_g, p.mtf.f50_b) <= 0:
            broken.append(f"{p.name}: f50")
        for ch in "rgb":
            if getattr(p.curves, ch).gamma <= 0:
                broken.append(f"{p.name}: {ch} gamma")
    facts.append(f"every stock carries the four fields the renderer cannot run "
                 f"without (rms, f50 triple, three gammas): "
                 f"**{len(broken) == 0}**")
    if broken:
        bad.append("a stock cannot render: " + ", ".join(broken[:4]))

    # ⚠ INERT CARRIERS MUST STILL BE WIRED. A carrier that nothing reads is a
    # research gap held open; a carrier that has been unwired is scope lost.
    fsrc = (HERE / "film_sim.py").read_text(encoding="utf-8")
    wired = {
        "stage 9c bromide drag": "apply_bromide_drag(dens, profile.processing.bromide_drag",
        "monochrome spectral weights": "spectral_monochrome_weights(profile)",
        "reciprocity": "reciprocity_log_shift(",
        "interimage (stage 8b)": "apply_interimage(",
        "DIR couplers (stage 9)": "apply_dir_couplers(",
    }
    for label, needle in wired.items():
        ok = needle in fsrc
        facts.append(f"{label} still called by the pipeline: **{ok}**")
        if not ok:
            bad.append(f"{label} is no longer wired")
    return facts, bad


def queue_rows() -> tuple[list[str], str]:
    """Live queue row ids, derived from the queue's own struck/done state."""
    import doc_consistency as dc
    q = HERE / "doc" / "DIGITIZATION_QUEUE.md"
    if not q.is_file():
        return [], ""
    text = q.read_text(encoding="utf-8", errors="ignore")
    return dc.queue_live_rows(text), text


def cpp_stage_entry_points() -> list[str]:
    """Stage entry points, read from AlgorithmMain.cpp -- the one file that must
    call every stage exactly once."""
    for root in (Path("/root/work/proot"), Path("/root/work/tst")):
        f = root / "AlgorithmMain.cpp"
        if f.is_file():
            txt = f.read_text(encoding="utf-8", errors="ignore")
            return sorted(set(re.findall(r"\bAlgoStage(\w+?)\s*\(", txt)))
    return []


# ---------------------------------------------------------------------------
def build(root: Path) -> tuple[str, list[str]]:
    bad: list[str] = []
    P = FILM_PROFILES
    kinds = Counter(p.kind.name for p in P)
    tiers = Counter(p.provenance.tier for p in P)
    st, par, n_par = provenance_census()
    ord_facts, ord_bad = ordering_check()
    sc_facts, sc_bad = scope_check()
    bad += ord_bad + sc_bad
    live, qtext = queue_rows()
    stages = cpp_stage_entry_points()

    w: list[str] = []
    a = w.append
    a("# PROJECT_STATE.md — the live state, generated")
    a("")
    a(f"**Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z by "
      f"`gen_project_state.py`. Do not edit by hand.**")
    a("")
    a("⚠ **THIS IS THE ONLY DOCUMENT IN `doc/` THAT STATES THE CURRENT STATE AS "
      "FACT.** Every number below is read out of the live module, the live C++ "
      "sources or the live queue at build time, so it cannot drift. Every other "
      "Markdown file in this tree is NARRATIVE OR HISTORICAL: a dated "
      "`RESULT_*` record states what was true on its date and is deliberately "
      "not rewritten, because it is the audit trail. Where a narrative file "
      "restates a count, this file is the authority.")
    a("")
    a("---")
    a("")
    a("## 1. Database")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| film stocks | **{len(P)}** |")
    a(f"| print stocks | **{len(PRINT_STOCKS)}** |")
    a(f"| gauges / formats | **{len(fp.FORMATS)}** |")
    a(f"| schema version | **v{fp.SCHEMA_VERSION}** |")
    a(f"| negative / reversal | {kinds.get('NEGATIVE', 0)} / "
      f"{kinds.get('REVERSAL', 0)} |")
    a(f"| monochrome | {sum(1 for p in P if p.is_monochrome)} |")
    a(f"| provenance tier 1 / 2 / 3 | {tiers.get(1,0)} / {tiers.get(2,0)} / "
      f"{tiers.get(3,0)} |")
    a("")
    a("## 2. Identity and ordering")
    a("")
    for f in ord_facts:
        a(f"- {f}")
    a("")
    a("## 3. Carrier census — what the database has evidence for")
    a("")
    a("⚠ **A LOW COUNT IS A RESEARCH GAP, NOT A DEFECT** (method rule 23). "
      "Every stock without a given carrier still renders; the field is absent "
      "because no source has been found, and its absence is recorded rather "
      "than filled with an invented value.")
    a("")
    a("| carrier | stocks | of |")
    a("|---|---|---|")
    for label, test in CARRIERS:
        try:
            n = sum(1 for p in P if test(p))
        except Exception as e:                               # pragma: no cover
            bad.append(f"carrier census failed on {label}: {e}")
            continue
        a(f"| {label} | **{n}** | {len(P)} |")
    a("")
    a("## 4. Parameter provenance")
    a("")
    a(f"**{n_par} `ParamSource` records** across "
      f"{sum(1 for p in P if p.param_sources)} profiles. The status vocabulary "
      f"is the project's own and maps onto the release requirement's "
      f"categories:")
    a("")
    a("| status | records | means |")
    a("|---|---|---|")
    meaning = {
        "measured": "an instrument reading, printed by the source as a number",
        "traced": "digitised off a published plot; the plot IS the source",
        "derived": "computed from another documented value by a stated rule",
        "stated": "a fact the source prints IN WORDS, not as a number",
        "spec_limit": "a TU / state-standard CEILING; the film was no worse",
        "estimated": "this project's own value, with a written rationale",
        "assumed": "this project's own value, WITHOUT one",
    }
    for k, v in st.most_common():
        a(f"| `{k}` | {v} | {meaning.get(k, '')} |")
    a("")
    a("⚠ **AN ABSENT RECORD IS NOT 'MEASURED'.** Outside the nine parameters "
      "`FilmActiveProfiles.md` prints, provenance is sparse on purpose: an "
      "entry means the provenance is known and stated, and an absence means "
      "the profile-level tier is the best statement available. That is a "
      "different claim from 'estimated' and must not be read as one.")
    a("")
    a("## 5. Engine")
    a("")
    a(f"- **{len(stages)} stage entry points** found in `AlgorithmMain.cpp`, "
      f"the one file that must call every stage exactly once.")
    if stages:
        a("- " + ", ".join(f"`{s}`" for s in stages))
    a("- Python (`film_sim.py`) is the model of record; scalar C++ is the "
      "high-accuracy reference; AVX2 is the optimised build of the same "
      "algorithm, not a second design.")
    a("- ⚠ The only permitted differences are the arithmetic type "
      "(`AlgoType` = double in the scalar project, float in AVX2; "
      "`HighPrecType` = double in both) and vectorisation. Every parity audit "
      "reads `sizeof(AlgoType)` from the compiled probe and picks its "
      "tolerance from it, so the typedef stays switchable.")
    a("")
    a("## 6. Live queue")
    a("")
    if live:
        a(f"**{len(live)} live rows**, derived from each row id's own "
          f"struck/done state: " + ", ".join(f"`{r}`" for r in live))
    a("")
    a("## 7. Scope preservation")
    a("")
    a("⚠ **Nothing has been removed, disabled or simplified for want of "
      "data.** Checked, not asserted:")
    a("")
    for f in sc_facts:
        a(f"- {f}")
    a("")
    if bad:
        a("## ⚠ GATE FAILURES")
        a("")
        for b in bad:
            a(f"- **{b}**")
        a("")
    return "\n".join(w) + "\n", bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--assert", dest="assert_", action="store_true")
    args = ap.parse_args()
    text, bad = build(Path(args.root))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"[OK] wrote {OUT.relative_to(HERE)} ({len(text.splitlines())} lines)")
    for b in bad:
        print(f"[FAIL] {b}")
    if bad:
        print(f"[FAIL] {len(bad)} gate check(s) failed")
        return 1 if args.assert_ else 0
    print("[OK] ordering, scope and carrier census all pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
