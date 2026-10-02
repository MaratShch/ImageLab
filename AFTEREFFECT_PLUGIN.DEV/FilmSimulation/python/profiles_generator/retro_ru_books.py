#!/usr/bin/env python3
"""Two Russian photographic books, read for what the schema can actually hold.

    PDF/PROFILES/RETRO/753285859-V-A-Yashtold-Govorko-Pechat-fotosnimkov-1967.pdf
    В. А. Яштолд-Говорко, «Печать фотоснимков», 1967, 214 pp.
    23 tables, 125 figures. A printing manual: its subject is PAPER.

    PDF/PROFILES/RETRO/475255698-E-Mitchel-Fotografiya-1988.pdf
    Э. Митчел, «Фотография», Мир, Москва 1988, 225 pp. — the Russian
    translation of E. Mitchell's «Photographic Science». A textbook.

⚠⚠ MOST OF WHAT THESE TWO BOOKS CONTAIN CANNOT BE STORED, AND SAYING SO IS
HALF THE VALUE OF READING THEM. Between them they print 23 + 8 numbered
tables; four of those tables produced database changes. The rest is either
about a carrier this schema does not have (photographic PAPER as a graded
product), or is generic photographic science with no named stock attached, or
duplicates something already measured from a better source. Every one of those
refusals is recorded in `EMULSION_KNOWLEDGE_BASE.md` §23t so that neither book
is re-read for the same disappointment.

WHAT WAS ADOPTED — FOUR THINGS
---------------------------------
1. **KODACHROME_64 and EKTACHROME_64 resolving power**, off Mitchell p. 179.
   One sentence gives both films at both contrasts: identical at 80 lp/mm low,
   125 against 100 high. All four fields were 0.0 before.
2. **KODAK_PLUS_X_125's only contrast-carrying development points**, off
   Mitchell's Табл. 8.4 — five rungs of time, CONTRAST INDEX and EXPOSURE
   INDEX together. The 274 points this stock already had are Kodak's own
   developer × temperature tables, which print no gamma anywhere.

WHAT WAS CROSS-CHECKED RATHER THAN ADOPTED — AND IT IS THE STRONGER RESULT
----------------------------------------------------------------------------
3. **ГОСТ 5554—63's minimum resolving powers** for Фото-32 / 65 / 130 / 250 —
   116 / 92 / 75 / 70 lp/mm — are a FLOOR («не ниже»), not a measurement. The
   four SVEMA profiles already carry 135 / 110 / 100 / 82 from the makers' own
   specifications. Every one clears its floor, and the margins are orderly
   (16 %, 20 %, 33 %, 17 %). The standard is therefore stored as a GUARD and
   not as data: a later pass that lowers one of those four below the figure
   the state standard required now fails the build.
4. **ГОСТ 5554—63's recommended gamma, 0.8 for all four speeds.** The stored
   curves fit at 0.80 / 0.83 / 0.80 / 0.85 — a 1967 printing manual quoting a
   state standard, against four curves this project traced from elsewhere,
   agreeing to 0.05. Also a guard.

⚠ AND ONE COMPARISON THAT IS NOT A GUARD BECAUSE IT IS NOT THE SAME QUANTITY.
ГОСТ caps the FOG («вуаль») of the four films at 0.05 / 0.10 / 0.15 / 0.20.
`ToneCurve.dmin` is the fitted base+fog and includes the support's own
density, which the standard's figure excludes. Two of the four stored values
sit above their cap on a bare reading — Фото-32 at 0.12 against 0.05 — and
that is what a support of roughly 0.07–0.10 D looks like, not a defect. The
module prints the comparison every build and asserts only the loose form:
dmin ≤ cap + 0.10 D of support.

Run:  python retro_ru_books.py [--root .] [--assert]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

YASHTOLD = ("PDF/PROFILES/RETRO/"
            "753285859-V-A-Yashtold-Govorko-Pechat-fotosnimkov-1967.pdf")
MITCHEL = ("PDF/PROFILES/RETRO/"
           "475255698-E-Mitchel-Fotografiya-1988.pdf")
PAGES_Y, PAGES_M = 214, 225

#: Sentences that carry a decision. Whitespace-normalised before matching.
QUOTES_Y = {
    "GOST 5554-63 resolving-power floor":
        "Фото-32» —116, у «Фото-65» — 92",
    "GOST 5554-63 recommended gamma 0.8":
        "равное 0,8",
    "GOST fog ceilings":
        "0,05 — у пленок «Фото-32»",
    "paper Schwarzschild exponent":
        "для фотобумаг 0,65—0,7",
    "printing coefficient 0.8":
        "равным 0,8 при контактной печати",
    "grain diameters of the two speeds":
        "от 0,4 до 0,7 мк",
}
QUOTES_M = {
    "the Kodachrome / Ektachrome resolving-power sentence":
        "125 линия/мм против 100",
    "line/mm equals line pairs/mm":
        "цикл/мм",
    "the film resolving-power range":
        "\u043c\u043e\u0436\u0435\u0442 \u0434\u043e\u0441\u0442\u0438\u0433\u0430\u0442\u044c 1000 \u043b\u0438\u043d\u0438\u044f/\u043c\u043c",
    "the granularity law G = sigma*sqrt(2a)":
        "не меньше 10 диаметров зерна",
    "paper useful log-E ladder":
        "Полезный интервал логарифмов экспозиций",
}

#: ГОСТ 5554—63 minimum resolving power, lines per mm, «не ниже».
GOST_RESOLVING = {"SVEMA_FOTO_32": 116.0, "SVEMA_FOTO_65": 92.0,
                  "SVEMA_FOTO_130": 75.0, "SVEMA_FOTO_250": 70.0}
#: ГОСТ 5554—63 maximum fog, «не должна быть выше».
GOST_FOG = {"SVEMA_FOTO_32": 0.05, "SVEMA_FOTO_65": 0.10,
            "SVEMA_FOTO_130": 0.15, "SVEMA_FOTO_250": 0.20}
#: ⚠ The support's own density, which ГОСТ's fog figure excludes and
#: ToneCurve.dmin includes. Not a measurement — the allowance the guard makes.
SUPPORT_D = 0.10
GOST_GAMMA, GOST_GAMMA_TOL = 0.80, 0.10

#: Mitchell p. 179, one sentence, two films, two contrasts.
MITCHEL_RESOLVING = {"EKTACHROME_64": (80.0, 125.0),
                     "KODACHROME_64": (80.0, 100.0)}
#: Mitchell Табл. 8.4 — Plus-X Pan in D-76 1:1: minutes, CI, exposure index.
MITCHEL_PLUSX = ((22.0, 1.00, 200), (18.0, 0.85, 178), (12.0, 0.70, 145),
                 (8.0, 0.56, 122), (4.5, 0.40, 53))


def _text(path: Path, want_pages: int):
    import pymupdf
    doc = pymupdf.open(str(path))
    if doc.page_count != want_pages:
        return None, "%s is %d pages, not %d" % (path.name, doc.page_count,
                                                 want_pages)
    return " ".join(" ".join(p.get_text().split()) for p in doc), ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()
    if not (root / YASHTOLD).is_file() or not (root / MITCHEL).is_file():
        print("  [SKIP] the two Russian books are not in the corpus")
        return 0
    print("Two Russian books -- Яштолд-"
          "Говорко 1967 and "
          "Митчел 1988")
    bad = 0

    ytxt, err = _text(root / YASHTOLD, PAGES_Y)
    if ytxt is None:
        print("  [FAIL] %s" % err)
        bad += 1
        ytxt = ""
    mtxt, err = _text(root / MITCHEL, PAGES_M)
    if mtxt is None:
        print("  [FAIL] %s" % err)
        bad += 1
        mtxt = ""

    for label, quote in QUOTES_Y.items():
        if quote not in ytxt:
            print("  [FAIL] Яштолд-"
                  "Говорко no longer says "
                  "%s (%r)" % (label, quote))
            bad += 1
    for label, quote in QUOTES_M.items():
        if quote not in mtxt:
            print("  [FAIL] Митчел no longer "
                  "says %s (%r)" % (label, quote))
            bad += 1
    if not bad:
        print("  [OK  ] all %d load-bearing sentences still printed"
              % (len(QUOTES_Y) + len(QUOTES_M)))

    try:
        import film_profiles as fp
    except Exception as exc:                                  # pragma: no cover
        print("  [WARN] could not import film_profiles: %s" % exc)
        return 1 if ns.assert_ else 0

    # ---- 1. the GOST resolving-power FLOOR, as a guard -------------------
    print("  ГОСТ 5554-63 resolving power, «не "
          "ниже» — stored value against the floor:")
    for name, floor in GOST_RESOLVING.items():
        got = fp.get_profile(name).mtf.resolving_power_lp_mm_highc
        margin = 100.0 * (got / floor - 1.0) if floor else 0.0
        print("    %-16s stored %5.1f lp/mm, floor %5.1f, margin %+5.1f %%"
              % (name, got, floor, margin))
        if got < floor:
            print("  [FAIL] %s is below the resolving power ГОСТ 5554-63 "
                  "required of it" % name)
            bad += 1

    # ---- 2. the GOST recommended gamma, as a guard -----------------------
    gam = {n: fp.get_profile(n).curves.g.gamma for n in GOST_RESOLVING}
    worst = max(abs(v - GOST_GAMMA) for v in gam.values())
    print("  ГОСТ recommended γ %.2f against the "
          "stored fits: %s (worst %+.3f)"
          % (GOST_GAMMA, " ".join("%.2f" % v for v in gam.values()), worst))
    if worst > GOST_GAMMA_TOL:
        print("  [FAIL] a SVEMA curve has drifted more than %.2f from the "
              "gamma its own state standard recommends" % GOST_GAMMA_TOL)
        bad += 1

    # ---- 3. the fog ceiling, printed and only loosely asserted -----------
    # ⚠ NOT THE SAME QUANTITY. See the module docstring.
    over = []
    cells = []
    for name, cap in GOST_FOG.items():
        d = fp.get_profile(name).curves.g.dmin
        cells.append("%s %.2f/%.2f" % (name.split("_")[-1], d, cap))
        if d > cap:
            over.append(name)
        if d > cap + SUPPORT_D:
            print("  [FAIL] %s's base+fog %.3f exceeds the ГОСТ fog ceiling "
                  "%.2f by more than a support's worth (%.2f D)"
                  % (name, d, cap, SUPPORT_D))
            bad += 1
    print("  ГОСТ fog ceiling vs stored base+fog: %s"
          % "  ".join(cells))
    if over:
        print("    ⚠ %d of 4 sit above the bare ceiling (%s) -- expected: "
              "ToneCurve.dmin includes the support, the standard's "
              "«вуаль» does not"
              % (len(over), ", ".join(n.split("_")[-1] for n in over)))

    # ---- 4. Mitchell's resolving-power pair ------------------------------
    for name, (lo, hi) in MITCHEL_RESOLVING.items():
        m = fp.get_profile(name).mtf
        if (abs(m.resolving_power_lp_mm_lowc - lo) > 1e-9
                or abs(m.resolving_power_lp_mm_highc - hi) > 1e-9):
            print("  [FAIL] %s holds %.0f/%.0f lp/mm, Mitchell p.179 prints "
                  "%.0f/%.0f" % (name, m.resolving_power_lp_mm_lowc,
                                 m.resolving_power_lp_mm_highc, lo, hi))
            bad += 1
    e = fp.get_profile("EKTACHROME_64").mtf
    k = fp.get_profile("KODACHROME_64").mtf
    print("  Mitchell p.179: low contrast equal at %.0f lp/mm; high contrast "
          "Ektachrome %.0f against Kodachrome %.0f"
          % (e.resolving_power_lp_mm_lowc, e.resolving_power_lp_mm_highc,
             k.resolving_power_lp_mm_highc))
    # ⚠ THE SENTENCE'S SHAPE, NOT JUST ITS NUMBERS: equal low, Ektachrome
    # ahead high. A later edit that changed one film alone would keep both
    # numbers legal and destroy the comparison they came from.
    if not (abs(e.resolving_power_lp_mm_lowc
                - k.resolving_power_lp_mm_lowc) < 1e-9
            and e.resolving_power_lp_mm_highc > k.resolving_power_lp_mm_highc):
        print("  [FAIL] the two films no longer stand as Mitchell prints them "
              "-- equal at low contrast, Ektachrome ahead at high")
        bad += 1

    # ---- 5. Mitchell's Plus-X contrast ladder ----------------------------
    pts = [q for q in fp.get_profile("KODAK_PLUS_X_125").processing_family.points
           if q.contrast_index > 0.0]
    got = tuple((q.minutes, round(q.contrast_index, 2), q.exposure_index)
                for q in pts)
    print("  Plus-X Pan, D-76 1:1 at 20 C, the only contrast-carrying points "
          "this stock has: %d of %d"
          % (len(pts),
             len(fp.get_profile("KODAK_PLUS_X_125").processing_family.points)))
    if got != MITCHEL_PLUSX:
        print("  [FAIL] the Plus-X contrast ladder does not match Табл. 8.4: "
              "%s" % (got,))
        bad += 1
    else:
        # ⚠ AND THE DISAGREEMENT WITH KODAK'S OWN TIMES IS ASSERTED, because
        # it is the reason both sets are kept. Kodak put D-76 1:1 at 20 C
        # between 6.25 and 7.00 minutes; Mitchell needs 8 for the normal aim.
        kod = [q.minutes for q in
               fp.get_profile("KODAK_PLUS_X_125").processing_family.points
               if "76" in q.developer and q.dilution == "1:1"
               and abs(q.celsius - 20.0) < 1e-9 and q.contrast_index == 0.0]
        if kod:
            ratio = 8.0 / (sum(kod) / len(kod))
            print("    ⚠ Kodak's own 1:1 times at 20 C average %.2f min "
                  "against Mitchell's 8.0 for CI 0.56 -- %.0f %% more "
                  "development for the same aim, recorded and not averaged"
                  % (sum(kod) / len(kod), 100.0 * (ratio - 1.0)))
            if not 1.05 <= ratio <= 1.45:
                print("  [FAIL] the Kodak / Mitchell development disagreement "
                      "has changed size; the record of it is now wrong")
                bad += 1

    if ns.assert_ and bad:
        print("\n[FAIL] the two Russian books do not reproduce")
        return 1
    print("\n[OK] two books re-read: one state standard re-checked against "
          "four SVEMA curves it never measured, one textbook sentence holding "
          "two reversal films' resolving power, and the only contrast-carrying "
          "development ladder KODAK_PLUS_X_125 has.")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
