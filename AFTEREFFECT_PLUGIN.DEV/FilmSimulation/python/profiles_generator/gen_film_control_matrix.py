#!/usr/bin/env python3
"""Emit the per-stock control availability matrix.

One row per film stock, in the order of ``film_names.txt`` -- which is the
database order, so row N is enumerator N -- and one column per control drawn in
the Effect Control Panel.

    V   control enabled for this stock
    O   control disabled: NOT APPLICABLE to this stock. The stock physically
        has no such property, so there is nothing to model and nothing to look
        for. A monochrome stock has no white balance; a slide has no print
        stage; a stock with no reseau plate has no mosaic.
    ?   control disabled: MISSING. The mechanism does apply to this stock, but
        either the database has no figure for it or the engine has no code to
        consume it. Every ? is a gap someone could close.

The distinction between O and ? is the point of the file. O is a closed
question and ? is an open one, and a host that greys both the same way is
correct, while a corpus plan that treats both the same way is not.

Availability is decided by the same predicate the engine uses, read from the
engine source and cited in the legend, so a V is a control some stage will act
on.

Usage:
    python3 gen_film_control_matrix.py [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import film_profiles as FP  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "doc" / "FilmControlMatrix.md"
NAMES = Path("/root/work/tst/film_names.txt")

EPS = 1e-12

V, O, Q = "V", "O", "?"


# ---------------------------------------------------------------------------
#  Predicates
#
#  Each returns V, O or ?. The comment above each names the engine site the
#  predicate mirrors, so a change in the engine can be traced to the column it
#  invalidates. Where a control is off, the predicate must decide WHY, and the
#  two reasons are not interchangeable.
# ---------------------------------------------------------------------------

def _always(p) -> str:
    """Stock-independent: the control acts on any image."""
    return V


# AlgoDevelopmentTime.hpp, called from AlgorithmMain frame setup in both engine
# trees, with film_sim.resolve_development_time as the reference and a
# cpp_parity probe over every stock. ⚠ THIS PREDICATE RETURNED Q ON EVERY STOCK
# UNTIL 2026-09-16c, AND IT WAS RIGHT TO: the field and its default existed in
# AlgoControl and nothing read them, so the control was documented, emitted and
# wired to nothing while the database carried 960 development points (queue
# P61b). It is now V exactly where the stock has a group of development points
# that can actually place a curve, which is the same test the resolver makes.
def _development(p) -> str:
    import film_sim as _fs
    return V if _fs.development_family(p) is not None else Q


# ⚠ TEMPERATURE IS STILL Q ON EVERY STOCK AND THAT IS NOT AN OVERSIGHT. The
# time axis became readable because the families carry a GAMMA against each
# time; the temperature rows carry a time against each temperature and no
# contrast, so there is nothing to place a curve with. ProcessingFamily.temp_q10
# would close it and is 0.0 on every stock but three.
def _dev_temp(p) -> str:
    return Q


# AlgoStorageAge.hpp, same call site and the same contract. V only where the
# stock carries a PUBLISHED dark-fade rate; everywhere else the mechanism
# applies and the corpus has no figure, which is the definition of ?.
def _storage(p) -> str:
    return V if p.dye_stability.has_data else Q


# AlgorithmMain frame setup. A stock with no traced variant row has no
# alternative process ON RECORD -- which is a corpus gap, not a property of the
# film: nearly any stock can be pushed, pulled or cross-processed.
def _variant(p) -> str:
    return V if p.process_variants else Q


# AlgoReciprocity.hpp:217-345. Table branch needs rows; spec branch needs a
# non-unity exponent, or the shift is identically zero. Every emulsion fails
# reciprocity somewhere, so an absent figure is missing data, never "no such
# effect".
def _reciprocity(p) -> str:
    if p.reciprocity_table.times_s:
        return V
    r = p.reciprocity
    return V if any(abs(x - 1.0) > EPS for x in
                    (r.schwarzschild_p_r, r.schwarzschild_p_g,
                     r.schwarzschild_p_b)) else Q


# Algo_03_Sim.cpp:132. One silver image has no inter-layer ratio to disturb, so
# a colour temperature change is an exposure change and stage 2 already covered
# it. Closed question: a black-and-white film has no white balance.
def _colour(p) -> str:
    return O if p.is_monochrome else V


# Algo_13_Sim.cpp:205 and Algo_14_Sim.cpp:87. A slide is projected as shot:
# no print curve, no dupe chain, no print grain. Closed question.
def _print_path(p) -> str:
    return O if p.is_reversal else V


# Algo_11 / Algo_13 / Algo_14. Grain runs wherever the stock has an amplitude,
# and every stock in the corpus has one.
def _grain(p) -> str:
    g = p.grain
    return V if max(g.rms_r, g.rms_g, g.rms_b, g.rms_granularity) > 0.0 else Q


# Algo_05_Sim.cpp:87. Zero gain in all three channels stops the stage.
#
#   O  the stock RECORDS an antihalation construction -- the halo is suppressed
#      by the film itself and there is nothing to model.
#   ?  the stock records neither a gain nor a construction, so the zero is an
#      absence of measurement. 67 stocks are in this state.
def _halation(p) -> str:
    h = p.halation
    if max(h.gain_r, h.gain_g, h.gain_b) > 0.0:
        return V
    return O if (getattr(p.emulsion, "antihalation", "") or "").strip() else Q


# Algo_09_Sim.cpp:1411-1419. The long-range term needs three layers and is
# skipped on monochrome; the short-range edge term "applies to monochrome
# stocks too - and is in fact the dominant coupler effect on them", so the
# control is APPLICABLE on every stock and an absent coefficient is a gap.
def _coupler(p) -> str:
    c = p.couplers
    if (c.strength > 0.0 and not p.is_monochrome) or c.edge_strength > 0.0:
        return V
    return Q


# Algo_10_Sim.cpp:219-220.
#
#   O  monochrome: one record cannot be out of register with itself.
#   ?  colour with no measured offset: the layers exist, the figure does not.
def _misreg(p) -> str:
    if p.is_monochrome:
        return O
    return V if p.misregistration_um > 0.0 else Q


# Algo_04_Sim.cpp:127 (coating field) and Algo_06_Sim.cpp:352 (corner buckle).
# One control drives both, so either coefficient is enough.
def _coating(p) -> str:
    c = p.coating
    return V if (c.coating_sigma > 0.0 or c.buckle_mtf_loss > 0.0) else Q


# Algo_07_Sim.cpp:167 and Algo_14_Sim.cpp:268. A reseau plate is a construction
# a stock either has or has not. Closed question.
def _reseau(p) -> str:
    return V if p.has_reseau else O


# AlgoCallier.hpp:179. Q = 1.0 is every colour stock and the transform is the
# identity there: the Callier effect is scatter by developed SILVER, and a dye
# image has none. Closed question, and it is the one control that is active
# only on monochrome.
def _callier(p) -> str:
    return V if abs(p.callier_q - 1.0) > EPS else O


# The six FilmDamage controls the header marks NO READER IN THE CURRENT ENGINE.
# storageSeverity, colourVeil and processingQuality have no data; dryingMarks
# and scannerArtifacts have no database field at all; flickerStops has data on
# 184/184 and no defined spectral shape or channel split. All open questions.
def _no_reader(p) -> str:
    return Q


# ---------------------------------------------------------------------------
#  Column definitions, in panel order
#
#  (panel label, C++ field, predicate, why O, why ?)
# ---------------------------------------------------------------------------

DEV_Q = ("this stock has no group of development points that can place a "
         "curve — see the Development appendix")

DEVTEMP_Q = ("the temperature rows carry a time and no contrast, so there is "
             "nothing to place a curve with — see the Development appendix")

STORAGE_Q = ("no published dark-fade rate for this stock; the mechanism "
             "applies to every chromogenic film and the corpus has a figure "
             "for two")

GROUPS: list[tuple[str, list[tuple[str, str, object, str, str]]]] = [
    ("Film Stock", [
        ("Film Stock",      "filmProfile",    _always,  "", ""),
        ("Film Format",     "filmFormat",     _always,  "", ""),
        ("Process Variant", "processVariant", _variant, "",
         "no alternative process traced for this stock"),
    ]),
    ("Exposure & Tone", [
        ("Exposure",            "exposureStops",     _always,      "", ""),
        ("Exposure Time",       "exposureTimeS",     _reciprocity, "",
         "no reciprocity table and a unity Schwarzschild exponent — the failure "
         "is real on every emulsion, the figure is not recorded"),
        ("Mid-Grey Target",     "greyTarget",        _always,      "", ""),
        ("Black Point Stretch", "blackPointStretch", _always,      "", ""),
    ]),
    ("Development", [
        ("Development Time",        "developmentMinutes", _development, "", DEV_Q),
        ("Development Temperature", "developmentCelsius", _dev_temp, "", DEVTEMP_Q),
        ("Storage Age",             "storageYears",       _storage,   "", STORAGE_Q),
    ]),
    ("Colour & White Balance", [
        ("Scene Colour Temperature", "sceneKelvin", _colour,
         "monochrome stock — one silver image has no inter-layer ratio", ""),
        ("White Balance Strength",   "wbStrength",  _colour,
         "monochrome stock — one silver image has no inter-layer ratio", ""),
    ]),
    ("Print & Duplication", [
        ("Print Stock",             "printStock",  _print_path,
         "reversal stock — a slide is projected as shot", ""),
        ("Duplication Generations", "generations", _print_path,
         "reversal stock — no dupe chain", ""),
        ("Intermediate Stock",      "dupeStock",   _print_path,
         "reversal stock — no dupe chain", ""),
        ("Print Grain",             "printGrain",  _print_path,
         "reversal stock — no print stage", ""),
    ]),
    ("Emulsion Character", [
        ("Grain",              "grainScale",    _grain,   "",
         "stock carries no grain amplitude"),
        ("Halation",           "halationScale", _halation,
         "stock records an antihalation construction — the halo is suppressed "
         "by the film",
         "no halation gain and no antihalation construction recorded — the zero "
         "is an absence of measurement"),
        ("DIR Couplers",       "couplerScale",  _coupler, "",
         "no coupler coefficient recorded; the short-range term applies to "
         "monochrome stocks too, so this is a gap on those as well"),
        ("Misregistration",    "misregScale",   _misreg,
         "monochrome stock — a single record cannot be out of register with "
         "itself",
         "colour stock with no measured layer offset"),
        ("Coating Unevenness", "coatingScale",  _coating, "",
         "no coating sigma and no buckle figure"),
        ("Reseau Reconstruction", "reseau",     _reseau,
         "stock carries no reseau plate", ""),
    ]),
    ("Lens & Reader", [
        ("Veiling Flare",       "flare",           _always,  "", ""),
        ("Corner Falloff",      "vignette",        _always,  "", ""),
        ("Scanner Specularity", "scannerSpecular", _callier,
         "Callier Q = 1.0 — scatter is a developed-silver effect and a dye "
         "image has none", ""),
    ]),
    ("Film Damage & Age", [
        ("Enable Film Damage",   "filmDamageEnabled",       _always, "", ""),
        ("Overall Strength",     "damage.damageStrength",   _always, "", ""),
        ("Damage Seed",          "damage.damageSeed",       _always, "", ""),
        ("Dust",                 "damage.dustLevel",        _always, "", ""),
        ("Debris",               "damage.debrisLevel",      _always, "", ""),
        ("Fibres",               "damage.fibreLevel",       _always, "", ""),
        ("Clumping",             "damage.dirtClumping",     _always, "", ""),
        ("Transport Scratches",  "damage.scratchTransport", _always, "", ""),
        ("Handling Scratches",   "damage.scratchHandling",  _always, "", ""),
        ("Gate Dirt",            "damage.gateDirt",         _always, "", ""),
        ("Gate Weave",           "damage.weaveAmount",      _always, "", ""),
        ("Splice & Tear Events", "damage.damageEvents",     _always, "", ""),
        ("Processing Quality",   "damage.processingQuality", _no_reader, "",
         "no reader; `processing.progress` is UNKNOWN on 175 of 184 stocks"),
        ("Drying Marks",         "damage.dryingMarks",       _no_reader, "",
         "no reader and **no database field exists** for drying marks"),
        ("Storage Severity",     "damage.storageSeverity",   _no_reader, "",
         "no reader; 7 of the 8 driving `aging.*` fields are empty on all stocks"),
        ("Colour Veil",          "damage.colourVeil",        _no_reader, "",
         "no reader; 9 of the 10 driving `aging.*` / `dye_stability.*` fields "
         "are empty"),
        ("Printer Flicker",      "damage.flickerStops",      _no_reader, "",
         "no reader; data present on 184/184 but the spectral shape and the "
         "channel split are undefined, so the model cannot be written"),
        ("Scanner Artifacts",    "damage.scannerArtifacts",  _no_reader, "",
         "no reader and **no database field exists** for scanner artefacts"),
    ]),
    ("Render", [
        ("Master Seed", "seed", _always, "", ""),
    ]),
]


def read_names() -> list[str]:
    """The panel-visible names, in the order the enumerators use."""
    out = []
    for line in NAMES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(line.strip('"').split("|", 1)[0])
    return out


def build(profiles, names) -> str:
    if len(profiles) != len(names):
        raise SystemExit(f"{len(profiles)} profiles but {len(names)} names")

    n = len(names)
    n_ctl = sum(len(cols) for _, cols in GROUPS)

    # Evaluate once; every later section reads these.
    marks = {label: [pred(p) for p in profiles]
             for _, cols in GROUPS for label, _, pred, _, _ in cols}

    tot = {s: sum(v.count(s) for v in marks.values()) for s in (V, O, Q)}

    L: list[str] = []
    w = L.append

    w("# Per-stock control availability")
    w("")
    w(f"**{n} film stocks × {n_ctl} Effect Control Panel controls.**")
    w("")
    w("| mark | meaning |")
    w("|---|---|")
    w("| **V** | control **enabled** for this stock |")
    w("| **O** | control **disabled — not applicable**. The stock has no such "
      "property. There is nothing to model and nothing to look for. |")
    w("| **?** | control **disabled — missing**. The mechanism does apply here, "
      "but the figure is absent from the database, or the engine has no code to "
      "consume it, or both. |")
    w("")
    w(f"Across the whole matrix: **{tot[V]}** V, **{tot[O]}** O, **{tot[Q]}** ? "
      f"— {n * n_ctl} cells.")
    w("")
    w("Row order is the order of `film_names.txt`, which is the database order, "
      "so row *N* is enumerator *N*. Columns are in panel order and grouped "
      "exactly as the panel groups them.")
    w("")
    w("## Why O and ? are not the same mark")
    w("")
    w("Both grey the control, and a host may draw them identically. For every "
      "other purpose they are opposites.")
    w("")
    w("**O is a closed question.** A monochrome stock has no white balance; a "
      "slide has no print stage; a stock with no reseau plate has no mosaic to "
      "reconstruct. No measurement would change the mark, and none should be "
      "sought. Showing the control as available on these stocks would be a "
      "defect.")
    w("")
    w("**? is an open question, and every one is actionable.** The mechanism is "
      "real on that film. What is missing is a number, a stage, or both. Three "
      "situations produce it:")
    w("")
    w("1. **The database has no figure.** The stage would run and the profile "
      "supplies zero, so the arithmetic is the identity. A datasheet, a traced "
      "curve or a measurement closes it.")
    w("2. **The engine has no reader.** The control exists with a documented "
      "range and no stage consumes it. Code closes it. Eight controls are in "
      "this state on all 184 rows.")
    w("3. **Both.**")
    w("")
    w("⚠ A `?` says nothing about whether the corpus is empty. Development is "
      "the clearest case: the column is `?` on all 184 rows because the engine "
      "has no reader, while 83 stocks carry a traced development time. The "
      "Development appendix separates the two halves.")
    w("")
    w("A `V` says the control acts. It says nothing about how well evidenced "
      "the underlying figure is — a manufacturer measurement and a Tier-3 era "
      "estimate both read `V`. Provenance is in `FilmActiveProfiles.md`.")
    w("")

    # ---- legend -----------------------------------------------------------
    w("## Column legend")
    w("")
    w("| # | Column | C++ field | `O` when | `?` when |")
    w("|---|---|---|---|---|")
    i = 0
    for gname, cols in GROUPS:
        for label, field, _pred, why_o, why_q in cols:
            i += 1
            seen = set(marks[label])
            o_txt = why_o if why_o else ("—" if O not in seen else "see note")
            q_txt = why_q if why_q else ("—" if Q not in seen else "see note")
            w(f"| {i} | {gname} — {label} | `{field}` | {o_txt} | {q_txt} |")
    w("")
    w("A dash means the mark never occurs in that column.")
    w("")

    # ---- one table per panel group ---------------------------------------
    for gname, cols in GROUPS:
        w(f"## {gname}")
        w("")
        parts = []
        for label, _, _, _, _ in cols:
            m = marks[label]
            bits = [f"{m.count(V)} V"]
            if m.count(O):
                bits.append(f"{m.count(O)} O")
            if m.count(Q):
                bits.append(f"{m.count(Q)} ?")
            parts.append(f"**{label}** {' / '.join(bits)}")
        w("Across the 184 stocks — " + ", ".join(parts) + ".")
        w("")
        w("| Film stock | " + " | ".join(c[0] for c in cols) + " |")
        w("|---" * (len(cols) + 1) + "|")
        for k, nm in enumerate(names):
            w(f"| {nm} | " + " | ".join(marks[c[0]][k] for c in cols) + " |")
        w("")

    L.extend(_development_appendix(profiles, names))
    return "\n".join(L) + "\n"


def _development_appendix(profiles, names) -> list[str]:
    """Why both development columns are ?, and what the corpus does hold.

    Kept as its own section because the mark there is driven by the engine and
    the database disagrees with it: a reader who takes `?` for "no data" will
    reach the wrong conclusion about the corpus.
    """
    def has(v):
        return v is not None and v > 0.0

    mins = [p for p in profiles if has(p.processing.minutes)]
    cels = [p for p in profiles if has(p.processing.celsius)]
    both = [p for p in profiles if has(p.processing.minutes)
            and has(p.processing.celsius)]
    fam = [p for p in profiles if p.processing_family.points]

    def axes(p):
        # A point is usable when it carries a CONTRAST, and the sheets state
        # that two ways: `gamma`, the straight-line slope, and
        # `contrast_index`, the average gradient Kodak and Fuji print as CI or
        # G-bar. Testing only `gamma` silently drops NEOPAN 1600, ILFORD PAN F
        # and SVEMA DS-5M, whose sheets publish the average gradient instead --
        # an earlier revision of this file did exactly that and undercounted
        # the solvable stocks as 9.
        g = [q for q in p.processing_family.points
             if q.gamma > 0.0 or q.contrast_index > 0.0]
        return ({round(q.minutes, 3) for q in g},
                {round(q.celsius, 2) for q in g})

    t_ok = [p for p in fam if len(axes(p)[0]) >= 2]
    c_ok = [p for p in fam if len(axes(p)[1]) >= 2]
    hot = [p for p in cels if p.processing.celsius > 24.0]
    n = len(profiles)

    L: list[str] = []
    w = L.append
    w("## Appendix — Development and storage")
    w("")
    w("⚠ **This section said, until 2026-09-16c, that both development "
      "columns were `?` on every row because no stage read either control.** "
      "That was true and is no longer. `AlgoDevelopmentTime.hpp` and "
      "`AlgoStorageAge.hpp` are called from `AlgorithmMain` frame setup in "
      "both engine trees, `film_sim.resolve_development_time` and "
      "`resolve_storage_age` are the references, and `cpp_parity.py` drives "
      "both resolvers over every stock — 1962 and 4440 probes, agreeing to "
      "4.5e-07, which is float storage rounding.")
    w("")
    w("**Development Time is now `V` on 11 stocks and Storage Age on 2.** "
      "Both were `0` before. The remaining `?` marks are a corpus gap and not "
      "an engine one, which is the reverse of the situation this appendix was "
      "written to describe.")
    w("")
    w("⚠ **Development Temperature is still `?` everywhere, and that is not "
      "an oversight.** The time axis became readable because the families "
      "carry a **gamma against each time**; the temperature tables carry a "
      "**time against each temperature and no contrast**, so there is nothing "
      "to place a curve with. `ProcessingFamily.temp_q10` would close it and "
      "is `0.0` on every stock but three.")
    w("")
    w("### What the corpus holds")
    w("")
    w("| | stocks |")
    w("|---|---|")
    w(f"| `processing.minutes` stated | {len(mins)} / {n} |")
    w(f"| `processing.celsius` stated | {len(cels)} / {n} |")
    w(f"| both | {len(both)} / {n} |")
    w(f"| `processing_family.points` non-empty | **{len(fam)}** / {n} |")
    w(f"| ≥ 2 distinct times carrying a contrast — time axis solvable | "
      f"**{len(t_ok)}** / {n} |")
    w(f"| ≥ 2 distinct temperatures carrying a contrast — temperature axis "
      f"solvable | **{len(c_ok)}** / {n} |")
    w("")
    w("\"Carrying a contrast\" means the point states either `gamma` or "
      "`contrast_index` — the sheets publish the straight-line slope or the "
      "average gradient (CI, Ḡ) depending on the manufacturer, and both are "
      "usable. Testing only `gamma` drops NEOPAN 1600, ILFORD PAN F and SVEMA "
      "DS-5M and undercounts the solvable stocks as 9.")
    w("")
    w(f"### Why {len(mins)} stated times do not make {len(mins)} wirable stocks")
    w("")
    w("A control has to answer *what happens to the image if I develop two "
      "minutes longer*, and that needs gamma as a function of time — a response "
      "curve, not a condition. The curve lives in `processing_family.points`.")
    w("")
    w(f"For the {len(mins) - len(t_ok)} stocks that state a time but have no "
      "solvable axis, what exists is **one** (time, temperature) pair. One "
      "point is the condition the shipped characteristic curves were measured "
      "under. Setting the control to that value is a no-op — the curve already "
      "*is* that condition — and setting it anywhere else extrapolates from a "
      "single point, which no datasheet supports.")
    w("")
    w("Stocks with a solvable axis:")
    w("")
    w("| stock | points | distinct times | distinct temps |")
    w("|---|---|---|---|")
    for p, nm in zip(profiles, names):
        if p not in fam:
            continue
        t, c = axes(p)
        if t:
            w(f"| {nm} | {len(p.processing_family.points)} | {len(t)} | "
              f"{len(c)} |")
    w("")
    w("The three with a real temperature axis are the Agfa APX trio, 62–73 "
      "points at four temperatures from «Technical Data P-16-C». The rest are "
      "single-temperature series: ILFORD PAN F at nine times, DOUBLE-X 5222 "
      "and the four 1952 Kodak sheet films at five, SUPER ANSCOCHROME at four, "
      "NEOPAN 1600 at three, SVEMA DS-5M at two.")
    w("")
    w("### The shipped control range does not cover the corpus")
    w("")
    w("`AlgoControlEnums.hpp` gives `DevelopmentCelsius` the range "
      f"**18.0 – 24.0 °C**. **{len(hot)} of the {len(cels)} stocks** that state "
      "a temperature are above that ceiling, because every colour process is: "
      "C-41 at 37.8 °C, E-6 at 38.0 °C, ECN-2 at 41.1 °C.")
    w("")
    w("The shipped range is a monochrome range. It matches the family stocks — "
      "all monochrome, all inside 18–24 °C — and describes no colour process in "
      "the database. Widening it, or making it per-stock as the control "
      "documentation already requires, is a prerequisite to wiring this "
      "control, separate from the missing curves. `DevelopmentMinutes` has no "
      "such problem: the corpus spans 2.5 – 16.0 min inside a 1.9 – 36.0 min "
      "control.")
    w("")
    w("### What exists on the C++ side")
    w("")
    w("| | state |")
    w("|---|---|")
    w("| `AlgoControls::developmentMinutes` / `::developmentCelsius` | present, "
      "`double`, default −1.0 |")
    w("| ranges and steps in `AlgoControlEnums.hpp` | present |")
    w("| `ProcessingSpec` — developer, dilution, minutes, celsius, agitation, "
      "contrast index | emitted for every stock |")
    w(f"| `ProcessingFamily::points` and `::source` | emitted, all {len(fam)} "
      "families |")
    w("| `ProcessingFamily` `gamma_infinity`, `dev_rate_k`, `induction_t0_min`, "
      "`temp_q10` | **not emitted** — withheld with the other nineteen v29 "
      "fields, `cpp_codegen.py` |")
    w("| a stage that reads either control | **none** |")
    w("")
    w("So the measured points reached C++ and the fitted rate law did not. "
      "Wiring on the interpolation path needs no new field; wiring on the "
      "rate-law path needs those four emitted first.")
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ns = ap.parse_args(argv)

    text = build(FP.FILM_PROFILES, read_names())
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(text, encoding="utf-8")
    print(f"[OK] wrote {ns.out} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
