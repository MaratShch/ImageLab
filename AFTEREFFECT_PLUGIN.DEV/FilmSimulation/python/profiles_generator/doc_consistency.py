"""Fail the build when a COUNT asserted in the documentation stops being true.

WHY THIS EXISTS
---------------
On 2026-08-25 a documentation audit found FOUR hardcoded counts in the report
generator wrong by up to **2.3x** (ISO 6 said 27 against a live 51, ISO 5800 said
34 against 58, ISO 2240 said 13 against 17, manufacturer EI said 15 against 34),
plus a claim that a struct was "read by no renderer" two days after it was wired,
and a queue row still reading "no profile" for stocks added the previous day.

None of it was caught, because nothing compared prose to the database. `build.py`
already gates `doc/PROGRESS.md` on a build-facts stamp -- schema version, stock
count, film-names digest -- and that check has never fired, for the good reason
that it is the only claim being checked. This script generalises that one idea:
**a number asserted in a document is a testable claim, so test it.**

WHAT IT CHECKS, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------
It checks a REGISTRY of specific, load-bearing sentences -- each with the pattern
that finds it and the live expression that must equal it. It does NOT try to
parse every number in the corpus: most numbers in these documents are measured
values, residuals, dates and page references, and a checker that guessed at them
would produce noise and be switched off within a week.

⚠ THE REGISTRY IS THE POINT, AND IT MUST GROW. Every count added to a document
should be added here in the same edit. A count that is not here is not checked,
and the audit above is what that looks like after a few weeks.

⚠ AND IT CANNOT CATCH A WRONG *CLAIM*, only a stale *count*. "39 raster pages are
on disk and unread" was wrong in its second half, not its first; no arithmetic
detects that. Prose still needs reading.

Run:  python doc_consistency.py [--root ..] [--assert]
Exit non-zero under --assert if any registered claim no longer matches.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import film_profiles as fp  # noqa: E402


def _stage_count() -> int:
    """Stage entry points the engine actually calls, read from the driver.

    Not from the stage translation units: a stage whose body is an inline
    header function (12b Callier) has no `Algo_NN_Sim.cpp` to grep. The
    driver calls every stage exactly once, so it is the honest source.
    """
    import re as _re
    pat = r"AlgoStage\d+[a-c]?_[A-Za-z]+"
    for cand in (HERE / "AlgorithmMain.cpp",
                 HERE.parent / "tst" / "AlgorithmMain.cpp"):
        if cand.is_file():
            txt = cand.read_text(encoding="utf-8", errors="replace")
            return len(set(_re.findall(pat, txt)))
    return 0


def live() -> dict:
    """Every quantity the registry may refer to, computed from the database."""
    P = fp.FILM_PROFILES
    spd: dict[str, int] = {}
    for q in P:
        spd[q.speed_criterion] = spd.get(q.speed_criterion, 0) + 1
    crit: dict[str, int] = {}
    for q in P:
        c = q.spectral.criterion
        if c:
            crit[c] = crit.get(c, 0) + 1
    return {
        "stocks": len(P),
        "print_stocks": len(fp.PRINT_STOCKS),
        "sigma_measured": sum(1 for q in P if q.grain.sigma_shape_measured),
        # ⚠ ADDED 2026-08-27. The schema number in NotFound.md's headline
        # sentence read "v15" while the database was at v18 -- THREE VERSIONS
        # STALE -- for the mundane reason that this registry guarded the two
        # COUNTS in that sentence and not the version beside them. The counts
        # could not drift and the number next to them could. Registering it
        # closes the hole; a value in a guarded sentence is not itself guarded.
        "schema": fp.SCHEMA_VERSION,
        # ⚠ ADDED 2026-08-31, AFTER A DOCUMENTED COUNT WAS WRONG FOR AN HOUR.
        # Four documents and two archive manifests said the pipeline has 25
        # stage entry points. It has 26. The count had been derived by grepping
        # `AlgoStage[0-9]+[a-c]?_` across `Algo_*_Sim.cpp`, and
        # `AlgoStage12b_Callier` is defined INLINE IN A HEADER (AlgoCallier.hpp)
        # rather than in a stage translation unit, so the grep could not see it
        # -- while the running engine profiles it as "12b  Callier" every frame.
        # ⚠ THE FIX IS THE SOURCE, NOT THE PATTERN: count the stages the DRIVER
        # actually calls. AlgorithmMain.cpp must name every stage exactly once,
        # so it cannot omit one the engine runs.
        "stages": _stage_count(),
        "param_sources": sum(len(q.param_sources) for q in P),
        "developers": sum(1 for q in P if q.processing.developer),
        "mtf_measured": sum(1 for q in P if q.mtf.mtf_measured),
        "mtf_measured_q": sum(1 for q in P
                              if q.mtf.mtf_measured and q.mtf.mtf_rolloff_q > 0),
        "dye_sets": sum(1 for q in P if q.dye_density.has_data),
        "spectral_sets": sum(1 for q in P if q.spectral.has_data),
        "recip_tables": sum(1 for q in P if q.reciprocity_table.has_data),
        "iso6": spd.get("iso6", 0),
        "iso5800": spd.get("iso5800", 0),
        "iso2240": spd.get("iso2240", 0),
        "manufacturer_ei": spd.get("manufacturer_ei", 0),
        "crit_d02": sum(v for k, v in crit.items() if "D0.2_above_dmin" in k),
        "crit_d04": sum(v for k, v in crit.items() if "D0.4_above_dmin" in k),
        "mixed_tag": sum(1 for q in P
                         if re.match(r"\[T[123]/T[123]\]", q.description)),
        # ⚠ ADDED 2026-09-01 BECAUSE ALL FIVE WERE FOUND STALE AT ONCE, AND
        # THEY WERE STALE FOR THE REASON THIS MODULE EXISTS. NotFound.md's
        # headline sentence reads "170 film stocks, 11 print stocks, 14 gauges,
        # schema v22." on one line and "131 negative / 39 reversal; 68
        # monochrome. Provenance tiers: 84 T1, 45 T2, 41 T3." on the next. Only
        # the FIRST line was registered. KODAK_EKTAR_125 moved the negative
        # count and the T3 count on 2026-08-31, the build stayed green, and the
        # second line quietly described a database that no longer existed.
        # A number is only maintained if something fails when it drifts.
        "negative": sum(1 for q in P if q.kind is fp.StockKind.NEGATIVE),
        "reversal": sum(1 for q in P if q.kind is fp.StockKind.REVERSAL),
        "monochrome": sum(1 for q in P if q.is_monochrome),
        "tier1": sum(1 for q in P if q.provenance.tier == 1),
        "tier2": sum(1 for q in P if q.provenance.tier == 2),
        "tier3": sum(1 for q in P if q.provenance.tier == 3),
        # Carriers the AGFA harvest populated, registered in the same edit as
        # the sentence that reports them.
        "coated": sum(1 for q in P if q.emulsion.coated_um > 0.0),
        "proc_families": sum(1 for q in P if q.processing_family.has_data),
        "base_um": sum(1 for q in P if q.emulsion.base_um > 0.0),
        "neutral_pairs": sum(1 for q in P if q.dye_density.has_neutral_pair),
        "designations": sum(1 for q in P if q.emulsion.designation),
    }


#: (document, regex with ONE capturing group holding the number, live key, note)
#:
#: The regex must be specific enough that it matches the sentence it was written
#: for and nothing else. A loose pattern that drifts onto another sentence is
#: worse than no check: it will pass while asserting something nobody meant.
REGISTRY: tuple[tuple[str, str, str, str], ...] = (
    ("doc/NotFound.md",
     r"\*\*(\d+) film stocks, \d+ print stocks, \d+ gauges, schema v\d+\.\*\*",
     "stocks", "the headline database size"),
    ("doc/NotFound.md",
     r"\*\*\d+ film stocks, (\d+) print stocks, \d+ gauges, schema v\d+\.\*\*",
     "print_stocks", "the headline print-stock count"),
    ("doc/NotFound.md",
     r"\*\*\d+ film stocks, \d+ print stocks, \d+ gauges, schema v(\d+)\.\*\*",
     "schema", "the headline schema version -- see the note in _live()"),
    ("doc/FilmActiveProfiles.md",
     r"\*\*(\d+) parameters across \d+ profiles now carry `ParamSource`",
     "param_sources", "the ParamSource coverage claim"),
    ("doc/PROGRESS.md",
     r"52 \u2192 (\d+) entries \(1529 at the time",
     "param_sources", "PROGRESS item 17's ParamSource count"),
    ("doc/NotFound.md",
     # ⚠ THE TAIL OF THIS PATTERN CHANGED ON 2026-09-02c AND THE PATTERN HAD TO
     # CHANGE WITH IT. It used to read "measured, and every one is Kodak"; queue
     # E5 made one of them a JOURNAL plate rather than a vendor sheet, so the
     # sentence was rewritten and the anchor moved to the part that is about the
     # count. An unmatched pattern stops checking silently, which is why
     # doc_consistency fails on a pattern miss and not only on a stale number.
     r"\*\*(\d+) of \d+\*\* measured -- \d+ colour negatives",
     "sigma_measured", "the measured sigma(D) count in the one-screen table"),
    # ⚠ PATTERN UPDATED 2026-08-26 IN THE SAME EDIT AS THE SENTENCE, which is
    # the discipline this module exists to enforce. DOUBLE-X 5222 became the
    # SECOND monochrome stock with a measured MTF, so "the mono stock in the
    # **11**" stopped being true as English before it stopped being true as
    # arithmetic -- and the unmatched pattern failed the build, exactly as
    # designed, instead of passing on a sentence nobody had reread.
    ("doc/NotFound.md",
     r"MTF: 199 vector pages inventoried, (\d+) stocks measured",
     "mtf_measured", "the measured-MTF count in the one-screen table"),
    ("doc/DATASHEET_VERIFICATION_REPORT.md",
     r"\*\*(\d+) stocks now carry `mtf_measured`\*\*",
     "mtf_measured", "the measured-MTF count in the verification report"),
    ("doc/DATASHEET_VERIFICATION_REPORT.md",
     r"the database holds (\d+) film stocks",
     "stocks", "the verification report's status note"),
    ("doc/Found.md",
     r"now reports \*\*(\d+)\*\* stocks with a live measured/estimated split",
     "stocks", "Found.md's header claim"),
    # ---- 2026-08-25d. The carrier census, added in the same edit as the
    # sentence it checks, which is the rule this module's docstring states and
    # which the four counts below now enforce rather than merely recommend.
    # ⚠ REGISTERED 2026-08-31. `SCHEMA_VERSION` was found FOUR versions stale --
    # v19-v22 landed with their fields commented and the constant never bumped --
    # and every document repeating "schema v18" was wrong with it. Only
    # NotFound.md's headline had the version guarded; these two did not, so they
    # are registered now. The lesson is the one this registry exists for: a
    # number is only maintained if something fails when it drifts.
    ("doc/README.md",
     r"has \*\*(\d+) stage entry points\*\*",
     "stages", "the README pipeline stage count"),
    ("doc/PROGRESS.md",
     r"\*\*(\d+) stage entry points\*\*, and \*\*all \d+ exist",
     "stages", "the PROGRESS engine stage count"),
    ("doc/DATASHEET_VERIFICATION_REPORT.md",
     r"the database holds \d+ film stocks, \d+ print stocks, schema v(\d+)\.",
     "schema", "the verification report's schema version"),
    ("doc/PROGRESS.md",
     r"\*\*(\d+) film stocks, \d+ print stocks, \d+ gauges\*\*, schema \*\*v\d+\*\*",
     "stocks", "the PROGRESS build-facts stock count"),
    ("doc/PROGRESS.md",
     r"\*\*\d+ film stocks, \d+ print stocks, \d+ gauges\*\*, schema \*\*v(\d+)\*\*",
     "schema", "the PROGRESS build-facts schema version"),
    ("doc/NotFound.md",
     r"\*\*(\d+) stocks carry a spectral dye-density set\*\*",
     "dye_sets", "the carrier census: dye-density sets"),
    ("doc/NotFound.md",
     r"\*\*(\d+) carry a spectral sensitivity set\*\*",
     "spectral_sets", "the carrier census: spectral sensitivity sets"),
    ("doc/NotFound.md",
     r"\*\*(\d+) carry a measured .\(D\) shape\*\*",
     "sigma_measured", "the carrier census: measured sigma(D) shapes"),
    ("doc/NotFound.md",
     r"\*\*(\d+) carry a measured MTF\*\*",
     "mtf_measured", "the carrier census: measured MTF"),
    # ---- 2026-09-01. The second line of NotFound.md's headline, and the
    # PROGRESS build-facts row that repeats it. Both were stale before this
    # registration and neither failed anything.
    ("doc/NotFound.md",
     r"\*\*\d+ film stocks, \d+ print stocks, \d+ gauges, schema v\d+\.\*\* (\d+) negative",
     "negative", "the headline negative/reversal split: negatives"),
    ("doc/NotFound.md",
     r"schema v\d+\.\*\* \d+ negative / (\d+) reversal",
     "reversal", "the headline negative/reversal split: reversals"),
    ("doc/NotFound.md",
     r"Provenance tiers: \*\*(\d+) T1, \d+ T2, \d+ T3\*\*",
     "tier1", "the headline provenance tiers: T1"),
    ("doc/NotFound.md",
     r"Provenance tiers: \*\*\d+ T1, (\d+) T2, \d+ T3\*\*",
     "tier2", "the headline provenance tiers: T2"),
    ("doc/NotFound.md",
     r"Provenance tiers: \*\*\d+ T1, \d+ T2, (\d+) T3\*\*",
     "tier3", "the headline provenance tiers: T3"),
    ("doc/NotFound.md",
     r"\*\*(\d+) stocks carry a published coated thickness\*\*",
     "coated", "the carrier census: published coated thickness"),
    ("doc/NotFound.md",
     r"\*\*(\d+) carry a manufacturer reciprocity table\*\*",
     "recip_tables", "the carrier census: manufacturer reciprocity tables"),
    ("doc/NotFound.md",
     r"\*\*(\d+) carry a\s+processing family\*\*",
     "proc_families", "the carrier census: processing families"),
    ("doc/NotFound.md",
     r"\*\*(\d+) carry a published base thickness\*\*",
     "base_um", "the carrier census: published base thickness (schema v23)"),
    ("doc/NotFound.md",
     r"\*\*(\d+) carry a neutral \+ D-min pair\*\*",
     "neutral_pairs", "the carrier census: neutral + D-min pairs"),
    ("doc/NotFound.md",
     r"\*\*(\d+) carry the manufacturer's own\s+emulsion designation\*\*",
     "designations", "the carrier census: emulsion designations (schema v23)"),
    ("doc/PROGRESS.md",
     r"schema \*\*v\d+\*\* \(re-measured from the live module [\d-]+: `SCHEMA_VERSION`, "
     r"`len\(FILM_PROFILES\)`, `len\(PRINT_STOCKS\)`, `len\(FORMATS\)`\)\. (\d+) negative",
     "negative", "the PROGRESS build-facts negative count"),
)



# ---------------------------------------------------------------------------
#  The QUEUE's own live-row set -- derived, never read from a sentence.
# ---------------------------------------------------------------------------
#: ⚠ ADDED 2026-09-05b AFTER THE SAME BUG BIT THREE TIMES IN THREE DAYS.
#: `DIGITIZATION_QUEUE.md` states its live-row count in five places written on
#: four different dates, and nothing re-derived any of them. What that cost:
#:
#:   * G5 (2026-09-05)  -- prose said "closed" in two places; the row id was
#:                         never struck, so the live list counted it for 2 days
#:   * E4 (2026-09-05b) -- closed 2026-09-02e at one line, still listed as live
#:                         WORK at another, and in two group tables besides
#:   * the headline     -- read "96 closed, 16 live" against a real 105 and 9
#:
#: Every one was found by parsing the file instead of trusting it. So the parse
#: is now the check: the live set is DERIVED from each row id's own struck/✅
#: state, and the sentence that names it has to agree.
#:
#: ⚠ IT GUARDS A SET, NOT A COUNT, ON PURPOSE. A count agrees by accident when
#: one row closes and another opens; the names cannot.
QUEUE = "doc/DIGITIZATION_QUEUE.md"
_ROW_RE = re.compile(r'^\|\s*(~~)?(\*\*)?(~~)?([A-Z]{1,3}\d{1,2}[a-z]?)(~~)?(\*\*)?(~~)?\s*\|')


def queue_live_rows(text: str) -> list[str]:
    """Row ids with no struck-through and no ✅ occurrence anywhere in the file.

    A row id may appear several times -- its own row, a group table, a
    retrospective. ⚠ ONE ✅ ANYWHERE CLOSES IT, which is deliberate: a row whose
    prose says closed in one place and stays unstruck in another is exactly the
    E4 fault, and treating it as live would report a false positive forever
    instead of the real bug, which is the unstruck duplicate.
    """
    seen: dict[str, bool] = {}
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        rid = m.group(4)
        done = bool(m.group(1) or m.group(3)) or "\u2705" in line[:500]
        seen[rid] = seen.get(rid, False) or done
    return sorted(r for r, d in seen.items() if not d)


def check_queue(root: Path) -> int:
    """The derived live set against the sentence that claims it. 0 = agree."""
    path = root / QUEUE
    if not path.is_file():
        print(f"  [SKIP] {QUEUE}: not present")
        return 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    live = queue_live_rows(text)
    bad = 0

    m = re.search(r"\*\*The (?:nine|ten|eight|seven|six|five|\d+): ([^*]+?)\*\*", text)
    if not m:
        print(f"[FAIL] {QUEUE}: no sentence naming the live rows. Add one of the "
              f"form '**The nine: A1, B2, ...**' beside the headline count -- "
              f"an unnamed live set is one nobody re-derives")
        return 1
    named = sorted(x.strip() for x in m.group(1).split(",") if x.strip())
    if named != live:
        print(f"[FAIL] {QUEUE}: the named live set disagrees with the file's own "
              f"row states")
        print(f"         named:   {', '.join(named)}")
        print(f"         derived: {', '.join(live)}")
        for x in sorted(set(named) - set(live)):
            print(f"         ⚠ {x} is named live but is struck or ✅ somewhere")
        for x in sorted(set(live) - set(named)):
            print(f"         ⚠ {x} is live in the file and missing from the "
                  f"sentence")
        bad += 1

    # ⚠ ANY SENTENCE THAT NAMES A LIVE COUNT IS CHECKED, NOT JUST THE HEADLINE.
    # The first version of this guard matched one phrasing and a FOURTH stale
    # copy survived it, in §3, worded "114 rows total. 104 closed. 10 live." and
    # six rows out of date. A guard that checks one wording of a claim the file
    # makes in several wordings is a guard that teaches false confidence.
    for m2 in re.finditer(r"(\d+)\s+closed[.,]\s+(\d+)\s+live", text):
        nlive = int(m2.group(2))
        if nlive != len(live):
            print(f"[FAIL] {QUEUE}: a sentence says {nlive} live "
                  f"({m2.group(0)!r}), derived {len(live)}")
            bad += 1
    if not bad:
        print(f"[OK] {QUEUE}: {len(live)} live rows, derived from the file's own "
              f"row states and matching the sentence that names them "
              f"({', '.join(live)})")
    return bad



def check(root: Path, do_assert: bool) -> int:
    facts = live()
    bad = 0
    checked = 0
    missing = 0
    for rel, pattern, key, note in REGISTRY:
        path = root / rel
        if not path.is_file():
            print(f"  [SKIP] {rel}: not present")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = re.findall(pattern, text)
        if not hits:
            # ⚠ A PATTERN THAT STOPS MATCHING IS A FAILURE, NOT A PASS. The
            # sentence was edited or removed and the claim is now unchecked,
            # which is exactly the state this script exists to end.
            print(f"[FAIL] {rel}: the registered sentence for '{note}' no longer "
                  f"matches. Either restore it or update the pattern in the same "
                  f"edit -- an unmatched pattern silently stops checking")
            missing += 1
            bad += 1
            continue
        want = facts[key]
        for got in hits:
            checked += 1
            if int(got) != want:
                print(f"[FAIL] {rel}: {note} says {got}, database says {want}")
                bad += 1
    print(f"[i] doc consistency: {checked} registered count(s) across "
          f"{len({r for r, _p, _k, _n in REGISTRY})} document(s), "
          f"{bad - missing} stale, {missing} pattern(s) no longer matching")
    bad += check_queue(root)
    if bad == 0:
        print("[OK] every registered documentation count matches the database")
    return 1 if (bad and do_assert) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()
    return check(Path(ns.root).resolve(), ns.do_assert)


if __name__ == "__main__":
    raise SystemExit(main())
