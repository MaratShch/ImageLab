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
     r"52 \u2192 (\d+) entries, 26 \u2192 161 profiles",
     "param_sources", "PROGRESS item 17's ParamSource count"),
    ("doc/NotFound.md",
     r"\*\*(\d+) of \d+\*\* measured, and every one is Kodak",
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
)


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
