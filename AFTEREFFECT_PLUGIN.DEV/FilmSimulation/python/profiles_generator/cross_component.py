"""Fail the build when the SEVEN components stop describing one product.

WHY THIS EXISTS
---------------
This project ships the same simulator seven times over: the Python reference,
the generated database, the scalar engine, the AVX2 engine, the Markdown set,
the HTML control mockup and the two effect-control PDFs. Each of those is
checked internally by something -- `verify.py` for the database, `cpp_parity.py`
for the two engines, `doc_consistency.py` for counts asserted in prose -- and
until now NOTHING checked the seams between them on the CONTROL surface.

⚠⚠ THE FIRST RUN FOUND FOUR STALE FACTS IN THE SHIPPING UI, and every one of
them had survived a documentation review because reviewing prose does not
catch a number in a table cell:

  * the mockup's control table gave `filmProfile` the range **0 - 190** when
    the database holds 200 stocks, so the last nine stocks were outside the
    range the UI documents;
  * it gave `processVariant` a maximum of **TOTAL_PROCESSES-1 (21)** when the
    enumerator had reached 37, i.e. sixteen process variants the host is told
    do not exist;
  * it gave `storageYears` a step of **1** against the shipping 0.5;
  * and it did not list `batchPosition` at all -- a control live in all three
    engines since schema v52, absent from both the table and the parameter
    documentation, so a host integrating from these files would never have
    known to draw it.

⚠ A FIFTH WAS IN THE ENGINE HEADER'S INSTRUCTION TO THAT HOST. `AlgoControl.hpp`
tells a host to hide `batchPosition` on a stock with no acceptance band and
printed the population as «184 of 194» -- harmless as prose, wrong as an
instruction, and rotting because nothing computed it. It is now derived.

⚠ AND ONE IN THE ENGINE. `getAlgoControlsDefault` opens with the claim that it
"Mirrors film_sim.RenderSettings exactly", sets every neighbouring field from
the shared `...Def` constant, and then writes a bare literal `24.0` into
`frameRate` where the reference's `RenderSettings.frame_rate` is **0.0**. The
values differ for a real reason -- the defect stages' rates are tuned at 24 fps
and R-T3 only forbids a silent fps in the MOTION grain path, which the C++
engine does not implement -- but a mirror claim that is false for one field is
how the next person mirrors the wrong thing.

WHAT IT CHECKS
--------------
  1. every numeric control in the generated `algo_control_enums` appears in
     the HTML mockup's control table and in the effect-control `param_data`,
     and the mockup's printed min / max / default / step agree with it;
  2. the four control ENUMS' extents as the mockup states them;
  3. `filmProfile`'s documented range against the live stock count;
  4. EN and RU describe the same thing, measured as the multiset of numbers
     each text contains;
  5. the C++ default-fill against `film_sim.RenderSettings`, field by field,
     for the fields both name.

WHAT IT DELIBERATELY DOES NOT CHECK
-----------------------------------
Prose. A translated paragraph can be fluent and wrong and no arithmetic finds
that. What check 4 catches is the cheaper and commoner failure: a figure edited
on one side of the pair and not the other.

Run:  python cross_component.py [--assert]
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import algo_control_enums as ace  # noqa: E402
import film_profiles as fp  # noqa: E402

#: The UI tree. Both files are shipped in archive 6.
UI = Path("/root/work/ui")
MOCKUP = "FilmSimulator_Mockup_v6.html"

#: The engine trees. The scalar one is authoritative for the shared headers;
#: `cpp_parity.py` is what proves the AVX2 copy matches, so this module reads
#: the scalar copy only and does not duplicate that check.
PROOT = Path("/root/work/proot")

# ---------------------------------------------------------------------------
# 1. the control surface
# ---------------------------------------------------------------------------
#: Controls that exist in the reference and are deliberately NOT on the host
#: surface, with the reason. A control that vanishes from the mockup without
#: being listed here is a regression; one listed here that REAPPEARS also
#: reports, so the list shrinks deliberately.
NOT_ON_HOST = {
    # 'grain_temporal_mode' LEFT THIS LIST ON 2026-10-06 (owner decision G6):
    # the engines implement all three modes and the host draws it as
    # grainTemporalMode, availability bit 48.
    'curve_measured':
        "a reference-only escape hatch added with the schema v55 measured "
        "curve, so a render can be compared against the softplus fit the "
        "table replaced. Shipping it would let a host silently switch off "
        "measured data",
}

#: ⚠ NUMBERS SPELLED AS WORDS ON ONE SIDE OF A TRANSLATION PAIR. Each of these
#: is a checked exception, not a tolerance: the figure IS in both texts, one of
#: them writes it in letters, and a checker that cannot see that would be
#: noise within a week. Listed with the word that carries it.
WORD_NUMBERS = {
    ('grainScale', 'usr'): "EN «100 %» -- RU «стопроцентном»",
    ('couplerScale', 'usr'): "EN «a 1980s one» -- RU «материала 1980-х»",
    ('flare', 'usr'): "EN «a 1930s scene» -- RU «сцена 1930-х»",
    ('storageYears', 'usr'):
        "EN «roughly fourteen to twenty times» -- RU «примерно в 14-20 раз»",
}

_NUM = re.compile(r'(?<![\w.,])\d+(?:[.,]\d+)?(?![\w])')
_ISO_DATE = re.compile(r'\b(20\d\d)-(\d\d)-(\d\d)\b')
_RU_DATE = re.compile(r'\b(\d\d)\.(\d\d)\.(20\d\d)\b')
#: ⚠ A US PATENT NUMBER IS GROUPED DIFFERENTLY IN THE TWO LANGUAGES -- the
#: English text writes «US 2,481,770» and the Russian «US 2 481 770» -- and a
#: general rule that joined any digit triple would also swallow the Russian
#: DECIMAL COMMA in «1,481» (a refractive index, not one thousand four hundred
#: and eighty-one). So the separator is stripped only where a patent number is
#: actually named.
_PATENT = re.compile(r'\bUS\s*(\d[\d  ,]*\d)\b')


def _numbers(text: str) -> collections.Counter:
    """The multiset of numbers a text states, with both date forms levelled.

    ⚠ THE TWO DATE FORMS ARE THE WHOLE REASON THIS IS NOT A ONE-LINER. The
    English text writes 2026-09-17 and the Russian 17.09.2026; tokenised
    naively the first is three numbers and the second is one, so every dated
    note in the corpus would report a mismatch and the check would be useless.
    Both are rewritten to one canonical token before anything is counted.
    """
    text = _PATENT.sub(
        lambda m: "USPAT" + re.sub(r'[  ,]', '', m.group(1)), text)
    text = _ISO_DATE.sub(lambda m: "D%s%s%s" % m.groups(), text)
    text = _RU_DATE.sub(lambda m: "D%s%s%s" % (m.group(3), m.group(2),
                                               m.group(1)), text)
    out = collections.Counter()
    for t in _NUM.findall(text):
        out[t.replace(",", ".")] += 1
    return out


def mockup_table(path: Path) -> dict:
    """The mockup's own control table: key -> (field, type, range, def, step)."""
    html = path.read_text(encoding="utf-8")
    rows = {}
    for key, body in re.findall(r'<tr data-f="([A-Za-z0-9_]+)">(.*?)</tr>',
                                html, re.S):
        cells = [re.sub(r'<[^>]+>', ' ', c) for c in
                 re.findall(r'<td[^>]*>(.*?)</td>', body, re.S)]
        cells = [re.sub(r'\s+', ' ', c).strip() for c in cells]
        if len(cells) >= 6:
            rows[key] = dict(label=cells[0], field=cells[1], type=cells[2],
                             rng=cells[3], default=cells[4], step=cells[5])
    return rows


def _nums_in(s: str) -> list[float]:
    """Every number in a table cell, minus sign and unicode minus honoured."""
    s = s.replace("−", "-").replace("–", " ").replace("—", " ")
    s = re.sub(r'(\d)e(-?\d)', r'\1E\2', s)
    out = []
    for t in re.findall(r'-?\d+(?:\.\d+)?(?:E-?\d+)?', s):
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


#: Control keys whose printed range is prose rather than two numbers, with the
#: reason. Each is still checked -- by the numbers it DOES print -- but is not
#: required to yield exactly a min and a max.
RANGE_IS_PROSE = {
    'filmProfile', 'filmFormat', 'processVariant', 'printStock', 'dupeStock',
    'printGrain', 'reseau', 'filmDamageEnabled', 'damageSeed', 'seed',
    'frameIndex', 'exposureTimeS',
}


def control_surface(problems: list, report: list) -> None:
    names = [x for x in dir(ace) if not x.startswith('_')]
    bounds: dict[str, dict] = {}
    for x in names:
        m = re.match(r'^(.*?)(Min|Max|Def|Step|Sentinel)$', x)
        if m:
            bounds.setdefault(m.group(1), {})[m.group(2)] = getattr(ace, x)
    rows = mockup_table(UI / MOCKUP)
    report.append("control surface: %d numeric controls in the generated "
                  "enums, %d rows in the mockup table"
                  % (len(bounds), len(rows)))

    # header name -> mockup key: the header uses PascalCase, the host camelCase
    def camel(s: str) -> str:
        return s[0].lower() + s[1:]

    for cname, b in sorted(bounds.items()):
        key = camel(cname)
        row = rows.get(key)
        if row is None:
            problems.append("control %s is defined in AlgoControlEnums.hpp "
                            "and is absent from the mockup's control table"
                            % key)
            continue
        if key in RANGE_IS_PROSE:
            continue
        got = _nums_in(row['rng'])
        want = [v for k, v in (('Min', b.get('Min')), ('Max', b.get('Max')))
                if v is not None]
        if want and (len(got) != len(want)
                     or any(abs(a - c) > 1e-9 for a, c in zip(sorted(got),
                                                              sorted(want)))):
            problems.append("%s: the mockup prints the range %r, the header "
                            "says %s" % (key, row['rng'], want))
        if 'Def' in b and not isinstance(b['Def'], bool):
            gd = _nums_in(row['default'])
            if not gd or all(abs(x - float(b['Def'])) > 1e-9 for x in gd):
                problems.append("%s: the mockup prints the default %r, the "
                                "header says %s" % (key, row['default'],
                                                    b['Def']))
        if 'Step' in b:
            gs = _nums_in(row['step'])
            if not gs or all(abs(x - float(b['Step'])) > 1e-9 for x in gs):
                problems.append("%s: the mockup prints the step %r, the "
                                "header says %s" % (key, row['step'],
                                                    b['Step']))


def enum_extents(problems: list, report: list) -> None:
    """The four control enums, as the mockup states their size."""
    rows = mockup_table(UI / MOCKUP)
    live = {
        'filmFormat': len(list(ace.FilmFormatCtrl)) - 1,
        'printStock': len(list(ace.PrintStockCtrl)) - 1,
        'dupeStock': len(list(ace.DupeStockCtrl)) - 1,
        'processVariant': int(ace.ProcessVariantCtrl.TOTAL_PROCESSES),
    }
    report.append("control enums: " + ", ".join(
        "%s %d" % (k, v) for k, v in sorted(live.items())))
    # ⚠ «N enumerators» MEANS N GATE GEOMETRIES, NOT N MEMBERS OF THE ENUM.
    # The first version of this check counted eFILM_FORMAT_TOTAL_FORMATS as a
    # format and reported the mockup wrong when the mockup was right -- a
    # checker that cries wolf on its first run is the fastest way to get every
    # other finding in this module ignored.
    m = re.search(r'(\d+)\s+enumerators', rows['filmFormat']['rng'])
    if not m or int(m.group(1)) != live['filmFormat']:
        problems.append("filmFormat: the mockup says %r, the enum carries %d "
                        "gate geometries beside the TOTAL sentinel"
                        % (rows['filmFormat']['rng'], live['filmFormat']))
    # processVariant prints the last legal value in brackets
    got = _nums_in(rows['processVariant']['rng'])
    if not got or max(got) != live['processVariant'] - 1:
        problems.append("processVariant: the mockup prints %r; the last legal "
                        "value is %d (TOTAL_PROCESSES %d)"
                        % (rows['processVariant']['rng'],
                           live['processVariant'] - 1, live['processVariant']))
    # printStock prints "N stocks + sentinel"
    m = re.search(r'(\d+)\s+stocks', rows['printStock']['rng'])
    if not m or int(m.group(1)) != live['printStock'] - 1:
        problems.append("printStock: the mockup says %r; the enum carries %d "
                        "stocks beside the sentinel"
                        % (rows['printStock']['rng'], live['printStock'] - 1))


def film_profile_range(problems: list, report: list) -> None:
    """⚠ THE ONE CONTROL WHOSE RANGE IS THE DATABASE ITSELF."""
    rows = mockup_table(UI / MOCKUP)
    n = len(fp.FILM_PROFILES)
    got = _nums_in(rows['filmProfile']['rng'])
    report.append("filmProfile: %d stocks, so the legal index range is 0 - %d"
                  % (n, n - 1))
    if sorted(got) != [0.0, float(n - 1)]:
        problems.append("filmProfile: the mockup prints the range %r against "
                        "a live %d stocks (0 - %d)"
                        % (rows['filmProfile']['rng'], n, n - 1))


def tolerance_population(problems: list, report: list) -> None:
    """⚠ A COUNT THE ENGINE HEADER STATES TO THE HOST, SO IT HAS TO BE TRUE.

    `AlgoControl.hpp` tells a host that `batchPosition` must hide itself on a
    stock carrying no acceptance band, and prints how many that is. It said
    «184 of 194» while the database held 200 stocks -- harmless as prose,
    wrong as an instruction, and exactly the sort of number that rots because
    nothing computes it.
    """
    band = [p.name for p in fp.FILM_PROFILES if p.tolerance]
    n, tot = len(band), len(fp.FILM_PROFILES)
    report.append("batchPosition: %d of %d stocks carry an acceptance band"
                  % (n, tot))
    src = (PROOT / "AlgoControl.hpp").read_text(encoding="utf-8",
                                                errors="replace")
    m = re.search(r'THAT IS (\d+) OF (\d+) STOCKS', src)
    if not m:
        problems.append("AlgoControl.hpp no longer states the batchPosition "
                        "hide-rule population; the host instruction has lost "
                        "its number")
    elif (int(m.group(1)), int(m.group(2))) != (tot - n, tot):
        problems.append("AlgoControl.hpp tells the host to hide batchPosition "
                        "on %s of %s stocks; live it is %d of %d"
                        % (m.group(1), m.group(2), tot - n, tot))


#: Populations the mockup states in prose, each as (regex, live expression).
#: ⚠ THESE ARE THE COUNTS A HOST READS TO DECIDE WHETHER TO DRAW A CONTROL, so
#: a stale one is an instruction rather than a typo. The mockup said
#: «16 of 191» for development time on the day queue P61 landed and it was
#: true that day; it was still there at 200 stocks and 37 families.
MOCKUP_POPULATIONS = (
    # ⚠ CORRECTED 2026-09-30: this counted every stock holding ANY
    # development point (37 of 201 then, 53 of 221 now), but the control acts
    # only where the points carry a contrast -- film_sim.development_family,
    # the predicate film_params_mask.hpp's availability bit uses. The mockup
    # therefore told a host to draw the control on stocks the mask greys.
    (r'Development Time</b> answers on <b>(\d+) of (\d+)</b>',
     lambda P: (sum(1 for p in P
                    if __import__("film_sim").development_family(p) is not None),
                len(P)),
     "stocks whose development-time control has a family to move along"),
    (r'acts on the <b>(\d+)</b> stocks that publish a dye-fade\s+rate',
     lambda P: (sum(1 for p in P if p.dye_stability
                    and (p.dye_stability.loss_c or p.dye_stability.loss_m
                         or p.dye_stability.loss_y or p.dye_stability.loss_r
                         or p.dye_stability.loss_g or p.dye_stability.loss_b)),),
     "stocks with a published dark-fade rate"),
    (r'Batch Position</b> answers on <b>(\d+) of\s+(\d+)</b>',
     lambda P: (sum(1 for p in P if p.tolerance), len(P)),
     "stocks carrying a manufacturing acceptance band"),
)


def mockup_populations(problems: list, report: list) -> None:
    html = (UI / MOCKUP).read_text(encoding="utf-8")
    for pat, fn, what in MOCKUP_POPULATIONS:
        m = re.search(pat, html)
        want = tuple(fn(fp.FILM_PROFILES))
        if not m:
            problems.append("the mockup no longer states the population of %s "
                            "in the form this checks; an unmatched pattern "
                            "stops checking silently" % what)
            continue
        got = tuple(int(g) for g in m.groups())
        if got != want:
            problems.append("the mockup states %s as %s; live it is %s"
                            % (what, got, want))
    report.append("mockup populations: %d prose counts derived from the "
                  "database" % len(MOCKUP_POPULATIONS))


def translation_parity(problems: list, report: list) -> None:
    sys.path.insert(0, str(UI))
    import param_data as pdm  # noqa: E402
    rows = mockup_table(UI / MOCKUP)
    pairs = 0
    for rec in pdm.P:
        k = rec['k']
        if k not in rows:
            problems.append("param_data documents %r, which the mockup's "
                            "control table does not list" % k)
        for fld in ('dev', 'usr', 'dep', 'acc'):
            en, ru = rec.get(fld + '_en', ''), rec.get(fld + '_ru', '')
            if bool(en) != bool(ru):
                problems.append("%s.%s: present in one language only" % (k, fld))
                continue
            if not en:
                continue
            pairs += 1
            a, b = _numbers(en), _numbers(ru)
            if a == b:
                if (k, fld) in WORD_NUMBERS:
                    problems.append(
                        "%s.%s is listed in WORD_NUMBERS as a checked "
                        "exception and now matches outright -- remove the "
                        "entry rather than leaving a stale exemption"
                        % (k, fld))
                continue
            if (k, fld) in WORD_NUMBERS:
                continue
            problems.append("%s.%s: the two languages state different "
                            "numbers -- EN only %s, RU only %s"
                            % (k, fld, dict(a - b), dict(b - a)))
    report.append("translations: %d EN/RU text pairs compared, %d carry a "
                  "number written as a word and are listed" % (pairs,
                                                               len(WORD_NUMBERS)))
    # every documented control must still exist on the host surface
    documented = {r['k'] for r in pdm.P}
    missing = sorted(set(rows) - documented)
    report.append("param_data covers %d of the mockup's %d control rows; the "
                  "remainder are the damage group's individual levels, which "
                  "the PDFs document under one heading" % (len(documented),
                                                           len(rows)))
    if len(documented) < 20:
        problems.append("param_data has shrunk to %d records" % len(documented))
    del missing


def defaults_mirror(problems: list, report: list) -> None:
    """`getAlgoControlsDefault` against `film_sim.RenderSettings`.

    ⚠ THE FUNCTION CLAIMS TO MIRROR THE REFERENCE «EXACTLY», SO THE CLAIM IS
    WHAT IS TESTED. Where the two differ on purpose the difference is listed
    below WITH ITS REASON; an undeclared difference fails.
    """
    import film_sim as fs
    src = (PROOT / "AlgoControl.cpp").read_text(encoding="utf-8",
                                                errors="replace")
    body = src.split("getAlgoControlsDefault", 1)[-1]
    got = dict(re.findall(r'controls\.([A-Za-z0-9_.]+)\s*=\s*([^;]+);', body))

    #: C++ field -> reference field, for the scalars both sides name.
    SAME = {
        'exposureStops': 'exposure_stops', 'greyTarget': 'grey_target',
        'blackPointStretch': 'black_point_stretch',
        'sceneKelvin': 'scene_kelvin', 'wbStrength': 'wb_strength',
        'grainScale': 'grain_scale', 'halationScale': 'halation_scale',
        'couplerScale': 'coupler_scale', 'misregScale': 'misreg_scale',
        'coatingScale': 'coating_scale', 'storageYears': 'storage_years',
        'storageCelsius': 'storage_celsius',
        'scannerSpecular': 'scanner_specular',
        'scannerFixedPattern': 'scanner_fixed_pattern',
        'generations': 'generations',
    }
    #: ⚠ DECLARED DIFFERENCES. Each is a value the two sides hold apart on
    #: purpose, with the reason, so that an UNdeclared one has nowhere to hide.
    DECLARED = {
        'frameRate':
            "the C++ default-fill sets 24.0 and RenderSettings.frame_rate is "
            "0.0. R-T3 forbids a silent fps only in the MOTION grain path, "
            "which the reference implements and the engines do not; on the "
            "C++ side frameRate feeds flicker, negative defects, weave and "
            "gate defects, whose published rates are all tuned at 24 fps, so "
            "a zero there would divide by nothing. The reference refuses "
            "instead of defaulting because it is the only side that can.",
    }
    ref = fs.RenderSettings()
    n = 0
    for cfield, pyfield in sorted(SAME.items()):
        lit = got.get(cfield)
        if lit is None:
            continue
        nums = _nums_in(lit)
        if not nums:
            continue
        want = float(getattr(ref, pyfield))
        n += 1
        if abs(nums[0] - want) > 1e-9:
            problems.append("getAlgoControlsDefault sets %s = %s; "
                            "RenderSettings.%s is %s -- and the function "
                            "claims to mirror it exactly"
                            % (cfield, nums[0], pyfield, want))
    for cfield in DECLARED:
        if cfield not in got:
            problems.append("the declared default difference on %s can no "
                            "longer be checked: the field is not assigned in "
                            "getAlgoControlsDefault" % cfield)
    report.append("defaults: %d scalar fields mirrored, %d declared "
                  "differences" % (n, len(DECLARED)))
    # ⚠ AND THE LITERAL, WHICH IS THE PART THAT ROTS. Every other enumerated
    # default in that function reads from the shared header; frameRate wrote a
    # bare 24.0, so the header could move and this would not.
    if 'frameRate' in got and 'FrameRateDef' not in got['frameRate']:
        problems.append("getAlgoControlsDefault writes a literal into "
                        "frameRate (%s) instead of the shared FrameRateDef; "
                        "the header's value can then move without this "
                        "following it" % got['frameRate'].strip())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert", dest="hard", action="store_true")
    args = ap.parse_args(argv)
    problems: list[str] = []
    report: list[str] = []
    control_surface(problems, report)
    enum_extents(problems, report)
    film_profile_range(problems, report)
    tolerance_population(problems, report)
    mockup_populations(problems, report)
    translation_parity(problems, report)
    defaults_mirror(problems, report)
    if not args.hard:
        for line in report:
            print("  " + line)
    if problems:
        for p in problems:
            print("[FAIL] cross_component.py -- " + p)
        return 1
    print("[OK] cross_component.py -- " + "; ".join(report))
    return 0


if __name__ == '__main__':
    sys.exit(main())
