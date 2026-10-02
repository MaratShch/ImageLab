#!/usr/bin/env python3
"""FERRANIA P30 «BEST PRACTICES» v 2.5 — the maker's own processing chart.

    PDF/PROFILES/FERRANIA/FP3011_Datasheet.pdf
    Film Ferrania S.r.l., «FERRANIA P30 BEST PRACTICES», version 2.5,
    3 pages, undated, published at www.filmferrania.it/p30.

⚠⚠ THIS FILM HAD FIVE PROCESS VARIANTS AND NO MANUFACTURER PROCESSING TIME,
which is the gap this sheet closes. The five come from a named third party's
D-76 **1+1** sensitometric test, run on PRE-PRODUCTION stock the tester's own
notes call «Pellicola di pre-produzione, stage alfa, difettata» — pre-
production, alpha stage, defective. Good data, honestly tiered at 2, and not
the manufacturer. This chart is Ferrania's, covers eight developers, and uses
D-76 at **stock** strength, so the two sets do not overlap and neither
displaces the other.

WHAT IS TAKEN, AND THE TWO THINGS THAT ARE NOT
-----------------------------------------------
**Page 2, eleven rows → eleven `DevelopmentPoint`s.** Developer, dilution,
temperature, exposure index and minutes. ⚠ **NO GAMMA IS PRINTED ANYWHERE ON
THE SHEET**, so every point is time-only and none of them can confirm or
contradict the stored curve. That is what the schema's time-only path exists
for; it is not a shortfall in the reading.

⚠ **`vessel` IS EMPTY ON ALL ELEVEN, DELIBERATELY.** The table's caption is
«RECOMMENDED TECHNIQUES for Handheld and Rotary Tanks» and each row gives ONE
time for both, varying only the agitation column. Writing "small tank" would
invent a distinction Ferrania do not make.

⚠⚠ **PAGE 3 IS NOT ADOPTED AS POINTS.** Its thirteen rows sit under Ferrania's
own heading «Additional Community-Submitted Processing Techniques».
`DevelopmentPoint` has no evidence tier, so dropping them into the same tuple
makes a user submission indistinguishable from a manufacturer recommendation
to every consumer — `resolve_development_time` included. They are transcribed
in `film_profiles._FERRANIA_P30_COMMUNITY_TIMES`, which no engine reads, and
this module checks that they are still there and still thirteen.

THE UNIT DEFECT, WHICH THIS MODULE ASSERTS RATHER THAN REMEMBERS
-----------------------------------------------------------------
Two rows print **«24ºC/72.5ºF»** — Kodak TMAX on page 2 and Fuji Negastar on
page 3. **24 ºC is 75.2 ºF; 72.5 ºF is 22.5 ºC.** A third row, Ilford DD on
page 3, prints «24ºC/75ºF» correctly, and that is what identifies the other
two as the error rather than as a third temperature. 24.0 ºC is stored, being
T-MAX developer's own normal working temperature and consistent with the rest
of the Celsius column. The module re-reads all three strings on every build:
if a later version of the sheet fixes the typo, the pinned strings stop
matching and the record above has to be revisited rather than quietly aging.

Run:  python ferrania_p30_best_practices.py [--root .] [--assert]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PDF = "PDF/PROFILES/FERRANIA/FP3011_Datasheet.pdf"

#: (developer, dilution, celsius, exposure_index, minutes) -- page 2, in the
#: order the chart prints them.
EXPECTED = (
    ("Kodak D-76", "stock", 20.0, 50, 8.0),
    ("Kodak D-76", "stock", 20.0, 80, 7.0),
    ("Kodak D-96", "stock", 21.0, 50, 8.0),
    ("Kodak D-96", "stock", 21.0, 80, 8.0),
    ("Ilford Ilfosol 3", "1:9", 20.0, 80, 6.0),
    ("Kodak HC-110", "1:63 (dil. H)", 20.0, 80, 12.0),
    ("Kodak HC-110", "1:31 (dil. B)", 20.0, 80, 5.0),
    ("Kodak TMAX", "1:6", 24.0, 80, 7.0),
    ("R09 (Rodinal)", "1:100", 20.0, 80, 60.0),
    ("Tetenal Paranol S", "1:50", 20.0, 80, 14.0),
    ("FF No.1 Monobath", "stock", 21.0, 80, 6.0),
)

#: Sentences that must still be on the sheet. Each one is load-bearing for a
#: decision recorded in the database, so its disappearance has to be loud.
QUOTES = {
    "box speed": "box speed of 80 ISO",
    "not DX coded": "not DX coded",
    "both-vessel caption": "RECOMMENDED TECHNIQUES for Handheld and Rotary Tanks",
    "community heading": "Community-Submitted Processing Techniques",
    "D-96 is the 1960s chemistry":
        "most similar to the original P30 developer made by Ferrania",
    "semi-stand": "Semi-Stand Technique",
    "HC-110 volume floor": "at least 450ml water",
}

#: ⚠ THE TEMPERATURE CONTRADICTION, PINNED AS STRINGS. Two wrong, one right.
BAD_TEMP = "24ºC/72.5ºF"
GOOD_TEMP = "24ºC/75ºF"
BAD_TEMP_COUNT = 2


def _text(root: Path):
    import pymupdf
    doc = pymupdf.open(str(root / PDF))
    if doc.page_count != 3:
        return None, "the sheet is %d pages, not 3" % doc.page_count
    return [p.get_text() for p in doc], ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()
    if not (root / PDF).is_file():
        print("  [SKIP] source not present: %s" % (root / PDF))
        return 0

    print("FERRANIA P30 «BEST PRACTICES» v 2.5 -- the manufacturer's "
          "processing chart")
    pages, err = _text(root)
    if pages is None:
        print("  [FAIL] %s" % err)
        return 1 if ns.assert_ else 0
    whole = " ".join(" ".join(p.split()) for p in pages)
    bad = 0

    if "v 2.5" not in whole:
        print("  [FAIL] this is no longer version 2.5 -- every pinned value "
              "below is version-specific")
        bad += 1

    # ---- the sentences ----------------------------------------------------
    for label, quote in QUOTES.items():
        if quote not in whole:
            print("  [FAIL] the sheet no longer says %s (%r)" % (label, quote))
            bad += 1
    if not bad:
        print("  [OK  ] all %d load-bearing sentences still on the sheet"
              % len(QUOTES))

    # ---- the temperature contradiction ------------------------------------
    n_bad = whole.count(BAD_TEMP)
    n_good = whole.count(GOOD_TEMP)
    print("  temperature strings: %r x%d (WRONG -- 24 C is 75.2 F), "
          "%r x%d (right)" % (BAD_TEMP, n_bad, GOOD_TEMP, n_good))
    if n_bad != BAD_TEMP_COUNT or n_good < 1:
        print("  [FAIL] the sheet's temperature contradiction has changed. "
              "The stored 24.0 C for TMAX rests on the OTHER row being "
              "correct; re-read the record before trusting it.")
        bad += 1

    # ---- the eleven rows, against the database ----------------------------
    try:
        import film_profiles as fp
        all_pts = fp.get_profile("FERRANIA_P30").processing_family.points
        # ⚠⚠ THE FAMILY IS NO LONGER ONE SOURCE, AS OF 2026-09-25e, AND THIS
        # MODULE OWNS ONLY ITS OWN. Five points from the alfa TEST REPORT were
        # added beside this sheet's eleven -- D-76 at **1+1**, each carrying a
        # BTZS average gradient, which is the only contrast this stock has
        # ever had. They are a different document about a different dilution
        # of a different (pre-production) coating, so they are selected OUT
        # here rather than folded in: `ferrania_p30_alfa_report.py` owns them.
        # ⚠ THE SELECTOR IS THE DILUTION, not a count and not an index. A
        # count would pass silently if a chart row were replaced by a report
        # row, which is exactly the substitution this audit exists to catch.
        got = tuple(q for q in all_pts if q.dilution != "1+1"
                    or "D-76" not in q.developer)
        alfa = tuple(q for q in all_pts if q not in got)
        want = EXPECTED
        if len(got) != len(want):
            print("  [FAIL] the family holds %d chart points (%d in all), "
                  "the chart has %d" % (len(got), len(all_pts), len(want)))
            bad += 1
        elif len(alfa) != 5:
            print("  [FAIL] %d alfa-report points beside the chart, want 5"
                  % len(alfa))
            bad += 1
        elif not all(q.contrast_criterion == "btzs_avg_gradient" and q.gamma
                     for q in alfa):
            print("  [FAIL] an alfa-report point has lost its BTZS average "
                  "gradient or its criterion; a G-bar read as a gamma or a "
                  "Kodak contrast index is the exact confusion "
                  "`contrast_criterion` was added to prevent")
            bad += 1
        else:
            drift = []
            for q, w in zip(got, want):
                if (q.developer != w[0] or q.dilution != w[1]
                        or abs(q.celsius - w[2]) > 1e-9
                        or q.exposure_index != w[3]
                        or abs(q.minutes - w[4]) > 1e-9):
                    drift.append(q.developer)
                # ⚠ THE SHEET PRINTS NO CONTRAST. A point that acquires one
                # did not get it from here.
                if q.gamma or q.contrast_index:
                    drift.append(q.developer + " (has a contrast)")
                if q.vessel:
                    drift.append(q.developer + " (has a vessel)")
            if drift:
                print("  [FAIL] drift against the chart: %s"
                      % ", ".join(drift[:4]))
                bad += 1
            else:
                print("  [OK  ] all 11 chart points match, none carries a "
                      "contrast the sheet does not print, none claims a "
                      "vessel; the 5 alfa-report points beside them are "
                      "D-76 1+1 and carry a BTZS average gradient")

        # ---- the community table stays out of the family ------------------
        com_all = getattr(fp, "_FERRANIA_P30_COMMUNITY_TIMES", ())
        # ⚠ THE REGISTER HOLDS TWO POPULATIONS SINCE 2026-09-25e. Thirteen
        # rows are Ferrania's own page 3; five are analogica.it users, added
        # so that the forum's development times are in the database and
        # attributable instead of only in a report. This module owns the
        # thirteen, so it selects them and leaves the forum rows to
        # `ferrania_vendor_sheets.py`.
        com = tuple(c for c in com_all if "analogica.it" not in c[5])
        forum = tuple(c for c in com_all if "analogica.it" in c[5])
        if len(forum) != 5:
            print("  [FAIL] the register holds %d analogica.it rows, not 5"
                  % len(forum))
            bad += 1
        # ⚠ KEYED ON (developer, dilution), NOT ON THE DEVELOPER ALONE. R09
        # (Rodinal) is on BOTH pages -- 1:100 semi-stand is Ferrania's own,
        # 1:50 is a community submission -- so a name-only test reports a leak
        # that is not one, which is exactly what it did on first run.
        pairs = {(q.developer, q.dilution) for q in got}
        leaked = sorted(str(k) for k in ({(c[0], c[1]) for c in com} & pairs))
        print("  community table: %d rows, held outside the family" % len(com))
        if len(com) != 13 or leaked:
            print("  [FAIL] the community table is %d rows and %s leaked into "
                  "the processing family" % (len(com), leaked or "nothing"))
            bad += 1

        # ---- what the sheet confirms rather than supplies ------------------
        prof = fp.get_profile("FERRANIA_P30")
        if prof.exposure_index != 80:
            print("  [FAIL] the sheet states box speed 80 ISO and the profile "
                  "holds %d" % prof.exposure_index)
            bad += 1
        # ⚠ THE FIVE PRE-PRODUCTION VARIANTS MUST SURVIVE. They are D-76 1+1
        # and this chart is D-76 STOCK: different chemistry, not a newer
        # reading of the same one, and a later pass must not "tidy" one away.
        pv = fp._PROCESS_VARIANTS.get("FERRANIA_P30") or ()
        one_plus_one = [v for v in pv if "1+1" in v.name]
        stock = [v for v in pv if "stock" in v.name]
        # ⚠ SIX SINCE 2026-09-25, NOT FIVE, AND THE SIXTH IS A DIFFERENT
        # DOCUMENT. Five are the alfa report's D-76 **1+1** legs; the sixth is
        # Film Ferrania's own D-76 **STOCK** 8-minute drawing off «Curve
        # caratteristiche e sensibilita spettrali». THIS chart is also D-76
        # stock, at 8 min for EI 50 and 7 min for EI 80, so the sixth variant
        # and the chart's first two rows describe the same chemistry -- which
        # is why the count is checked by DILUTION and not as a bare total.
        if len(one_plus_one) != 5 or len(stock) != 1:
            print("  [FAIL] expected five D-76 1+1 variants and one D-76 "
                  "stock variant, found %d and %d"
                  % (len(one_plus_one), len(stock)))
            bad += 1
        else:
            print("  [OK  ] box speed 80 confirmed; five D-76 1+1 "
                  "pre-production variants plus the one D-76 stock curve")
    except Exception as exc:                                  # pragma: no cover
        print("  [WARN] could not consult film_profiles: %s" % exc)

    if ns.assert_ and bad:
        print("\n[FAIL] the P30 processing chart does not reproduce")
        return 1
    print("\n[OK] eleven manufacturer development points re-read, the "
          "thirteen community rows confirmed still outside the family, and "
          "the sheet's own 24 C / 72.5 F contradiction re-checked rather "
          "than remembered.")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
