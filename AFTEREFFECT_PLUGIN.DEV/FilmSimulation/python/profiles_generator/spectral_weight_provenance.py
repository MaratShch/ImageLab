"""SPECTRAL_WEIGHTS PROVENANCE -- one rule, regenerated, never hand-edited.

WHY THIS FILE EXISTS
--------------------
`_PARAM_SOURCES_DERIVED` in `film_profiles.py` carries the header instruction
"REGENERATE, do not hand-edit: the rules live in the task EM-A6 generator".

⚠ THAT GENERATOR IS NOT IN THE REPOSITORY. It was a one-off in a session that
is over, so for eleven months the instruction has pointed at nothing and the
161 `spectral_weights` records could only ever be hand-edited -- the exact
thing it forbids. This file is the missing half for that one parameter: the
rule, written down, runnable, and asserted by the build.

WHAT THE PREVIOUS RULE GOT WRONG, MEASURED 2026-08-29
-----------------------------------------------------
1. **48 stocks claimed `status='derived'`, `conditions='integrated from the
   traced log-sensitivity curves'`. Every one of them stored (0.30, 0.59,
   0.11)** -- the `FilmProfile` dataclass default, which is Rec.601 video
   luma. Nothing had been integrated from anything. All 48 are colour stocks,
   where the field is never read, so no frame rendered wrong; but 48 cells
   printed PLAIN in `FilmActiveProfiles.md` on a false label.

2. **113 stocks carried the note "No traced spectral sensitivity for this
   stock".** For 28 of them that is simply untrue -- they carry a traced
   curve, `AGFA_APX_100` and `AGFA_APX_400` among them, extracted from
   `apx100.pdf` / `apx400.pdf` p2 on 2026-08-17 to 0.50 nm and 0.0034 log.
   The owner found this by reading the report and asking why his datasheets
   had been ignored. They had not been; the note was wrong.

THE RULE
--------
`spectral_weights` collapses scene RGB onto ONE silver record. It is read at
`film_sim` stage 7 and in `Algo_07_Sim.cpp`, and both read it ONLY under
`profile.is_monochrome`. So:

  A. monochrome + traced pan curve + passes the gamut-reach guard
     -> status 'derived'. The renderer does not read the stored triple at all;
        it integrates the curve. The report prints the DERIVED value.
  B. monochrome + traced pan curve + guard refuses
     -> status 'estimated'. Authored triple is what renders, and the guard's
        refusal is recorded with its measured cause.
  C. monochrome + no curve
     -> untouched. The existing note is true for these.
  D. colour, curve or no curve
     -> status 'estimated', recorded as INERT. The field exists on every
        profile because the struct is uniform, not because it means anything
        for a three-layer stock.

Run:
    python spectral_weight_provenance.py            # report, writes nothing
    python spectral_weight_provenance.py --write    # rewrite film_profiles.py
    python spectral_weight_provenance.py --assert   # non-zero exit on drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import film_profiles as fp
import film_sim as fs

HERE = Path(__file__).resolve().parent
TARGET = HERE / "film_profiles.py"

#: The dataclass default, spelled out so the check below cannot drift from it.
#: This IS Rec.601 luma, which is what makes it the wrong answer for film.
_LUMA_DEFAULT = (0.30, 0.59, 0.11)

_BASIS = ("pan curve integrated against the render primary basis "
          "(Gaussian lobes 600/540/460 nm, sigma 55 nm, unit area), "
          "renormalised to sum 1")

_INERT = (
    "⚠ INERT FOR THIS STOCK. spectral_weights collapses scene RGB onto "
    "ONE silver record and is read only where profile.is_monochrome "
    "(film_sim stage 7; Algo_07_Sim.cpp case 2). This is a three-layer "
    "colour stock, so no renderer ever reads it and its value cannot affect "
    "any frame. The stored triple is the FilmProfile dataclass default "
    "(0.30, 0.59, 0.11), which is Rec.601 video luma. "
    "⚠ CORRECTED 2026-08-29: 48 colour stocks previously carried "
    "status 'derived' with conditions 'integrated from the traced "
    "log-sensitivity curves'. That label was false on every one of them -- "
    "each still stored the untouched default. Nothing was integrated.")


def _tri(t) -> str:
    return "(%.3f, %.3f, %.3f)" % (float(t[0]), float(t[1]), float(t[2]))


def _spectral_tier(p) -> int:
    """1 when the curve names a manufacturer document, else 2."""
    src = (getattr(p.spectral, "source", "") or "").strip()
    return 1 if src else 2


def _derived_note(p, derived) -> str:
    note = (
        "⚠ THE STORED FilmProfile.spectral_weights TRIPLE IS NOT THIS "
        "VALUE AND IS NOT READ. Stored: %s, a class default. This cell "
        "prints %s, which both engines compute at run time from this stock's "
        "own traced pan curve -- Python via RenderSettings.spectral_mono (ON "
        "since 2026-08-29), C++ via AlgoSpectralMonoWeights(), which has "
        "never had a flag and has always derived. The stored triple survives "
        "only as the fallback for stocks with no curve. "
        "⚠ The lobe WIDTH (55 nm) is an assumption, not a measurement: "
        "the derivation is exact given the basis and the basis is a "
        "convention. A scene spectral model would remove that assumption; "
        "reprojecting the data the database already holds does not."
        % (_tri(p.spectral_weights), _tri(derived)))
    if p.name == "ROLLEI_INFRARED_400":
        note += (
            " ⚠ SPECIFIC TO THIS STOCK: the traced curve is the "
            "UNFILTERED sensitisation -- it peaks at 410 nm and puts only "
            "0.028 of its energy past 700 nm, so the gamut-reach guard "
            "cannot honestly refuse it. The authored (0.52, 0.20, 0.28) "
            "encodes an assumed deep-red/IR taking filter that NO FIELD IN "
            "THIS PROFILE RECORDS. The derived triple is right for the data "
            "on file and wrong for the way the film is used. Queue row C39.")
    return note


def _refused_note(p) -> str:
    peak = fs.spectral_peak_lambda(p)
    out = fs.spectral_out_of_reach(p)
    return (
        "Authored class triple, and it is what renders. The curve-based "
        "derivation is REFUSED by the gamut-reach guard: peak sensitisation "
        "%.0f nm against a %.0f nm basis limit, and %.3f of the emulsion's "
        "energy lies beyond that limit (measured on the curve's own samples "
        "to 830 nm, not on the renderer's 730 nm grid -- on the clipped grid "
        "the same figures read 730 nm and 0.203, low by a factor of two). "
        "Projected onto three visible lobes this stock derives to "
        "(0.161, 0.193, 0.646), BLUE-dominant, against an authored and "
        "correct red-dominant %s. That is a true statement about "
        "photographing a monitor and a nonsense one about photographing the "
        "world."
        % (peak, fs._SPECTRAL_BASIS_LAMBDA_MAX, out,
           _tri(p.spectral_weights)))


def plan() -> dict[str, dict]:
    """The intended ParamSource fields for every profile, by rule."""
    out: dict[str, dict] = {}
    for p in fp.FILM_PROFILES:
        if not p.is_monochrome:
            out[p.name] = dict(
                tier=2, status="estimated", unit="normalised weights",
                conditions="n/a -- field not read for a three-layer stock",
                source="", confidence="low", note=_INERT, case="D")
            continue

        if not p.spectral.has_data:
            out[p.name] = dict(case="C")          # leave exactly as found
            continue

        derived = fs.spectral_monochrome_weights(p)
        if derived is None:
            out[p.name] = dict(
                tier=2, status="estimated", unit="normalised weights",
                conditions="authored class triple; curve-based derivation "
                           "refused by the gamut-reach guard",
                source=(p.spectral.source or ""), confidence="medium",
                note=_refused_note(p), case="B")
        else:
            out[p.name] = dict(
                tier=_spectral_tier(p), status="derived",
                unit="normalised weights", conditions=_BASIS,
                source=(p.spectral.source or ""), confidence="high",
                note=_derived_note(p, derived), case="A")
    return out


# ---------------------------------------------------------------------------
#  Source rewriting.
#
#  The records live as literals inside `_PARAM_SOURCES_DERIVED`. The span of
#  one ParamSource(...) call is found by BRACKET MATCHING rather than by a
#  regex over its body: a note is free text and will eventually contain a
#  parenthesis, at which point a regex silently truncates the wrong record.
# ---------------------------------------------------------------------------
_ANCHOR = "param='spectral_weights',"


def _call_span(text: str, anchor_at: int) -> tuple[int, int]:
    """(start, end) of the ParamSource( ... ) call containing `anchor_at`."""
    start = text.rindex("ParamSource(", 0, anchor_at)
    i = text.index("(", start)
    depth = 0
    while True:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return start, i + 1
        elif ch in "\"'":
            quote = ch
            i += 1
            while text[i] != quote:
                i += 2 if text[i] == "\\" else 1
        i += 1


def _profile_of(text: str, at: int) -> str:
    """Name of the dict key whose block contains offset `at`."""
    line_start = text.rindex("\n    '", 0, at) + len("\n    '")
    return text[line_start:text.index("'", line_start)]


def _render(name: str, spec: dict) -> str:
    def lit(s: str) -> str:
        return repr(s)
    parts = [
        "ParamSource(",
        "\n            param='spectral_weights', tier=%d, status=%s,"
        % (spec["tier"], lit(spec["status"])),
        "\n            unit=%s," % lit(spec["unit"]),
        "\n            conditions=%s," % lit(spec["conditions"]),
    ]
    if spec["source"]:
        parts.append("\n            source=%s," % lit(spec["source"]))
    parts.append("\n            confidence=%s," % lit(spec["confidence"]))
    parts.append("\n            note=%s)" % lit(spec["note"]))
    return "".join(parts)


def rewrite(text: str, wanted: dict[str, dict]) -> tuple[str, int]:
    begin = text.index("_PARAM_SOURCES_DERIVED: dict")
    end = text.index("_PARAM_SOURCES_DEVELOPER: dict")
    head, body, tail = text[:begin], text[begin:end], text[end:]

    hits = []
    at = 0
    while True:
        at = body.find(_ANCHOR, at)
        if at < 0:
            break
        hits.append(at)
        at += len(_ANCHOR)

    changed = 0
    for at in reversed(hits):                    # right to left: offsets hold
        name = _profile_of(body, at)
        spec = wanted.get(name)
        if spec is None or spec.get("case") == "C":
            continue
        s, e = _call_span(body, at)
        new = _render(name, spec)
        if body[s:e] != new:
            body = body[:s] + new + body[e:]
            changed += 1
    return head + body + tail, changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="rewrite film_profiles.py in place")
    ap.add_argument("--assert", dest="assert_", action="store_true",
                    help="non-zero exit if the records are out of date")
    args = ap.parse_args()

    wanted = plan()
    cases = {}
    for spec in wanted.values():
        cases[spec["case"]] = cases.get(spec["case"], 0) + 1

    text = TARGET.read_text(encoding="utf-8")
    new, changed = rewrite(text, wanted)

    print("spectral_weights provenance rule")
    print("  A derived (mono, curve, guard passes) : %d" % cases.get("A", 0))
    print("  B refused (mono, curve, guard refuses): %d" % cases.get("B", 0))
    print("  C untouched (mono, no curve)          : %d" % cases.get("C", 0))
    print("  D inert (colour)                      : %d" % cases.get("D", 0))
    print("  records needing rewrite               : %d" % changed)

    # An independent restatement of the defect, so the audit fails loudly if
    # anyone ever re-labels an unmodified default as derived again.
    liars = [p.name for p in fp.FILM_PROFILES
             if not p.is_monochrome
             and tuple(round(float(v), 2) for v in p.spectral_weights)
             == _LUMA_DEFAULT
             and any(e.param == "spectral_weights" and e.status == "derived"
                     for e in p.param_sources)]
    if liars:
        print("  ⚠ colour stocks still labelled 'derived' while storing "
              "the luma default: %d" % len(liars))

    if args.assert_:
        if changed or liars:
            print("FAIL: spectral_weights provenance is out of date; "
                  "run with --write")
            return 1
        print("PASS: spectral_weights provenance matches the rule")
        return 0

    if args.write:
        if changed:
            TARGET.write_text(new, encoding="utf-8")
            print("WROTE %s (%d records)" % (TARGET.name, changed))
        else:
            print("no change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
