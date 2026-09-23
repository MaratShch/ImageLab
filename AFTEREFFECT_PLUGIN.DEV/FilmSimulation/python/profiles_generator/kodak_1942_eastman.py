#!/usr/bin/env python3
"""«Eastman Motion Picture Films for Professional Use» (Kodak, 1942), read whole.

WHAT THE DOCUMENT IS
--------------------
Eastman Kodak Company, © 1942, 98 PDF pages, **no text layer on any of them**
— every value below came from rendering the native 560 dpi JPEG and reading
it. Body sheets imprinted 12-44; a supplement of replacement sheets dated
SEPTEMBER 1943 is bound at the back, together with a separate insert booklet
«The Commercial Use of 16-mm. Kodachrome Film».

⚠ THE PAGE OFFSET IS NOT CONSTANT. Printed 1–51 sit at PDF+4; printed 53–72 at
PDF+3, because **the "EASTMAN POSITIVE FILMS" section-opening leaf at printed
page 52 is physically missing from this scan**. Anchors: PDF 49 = printed 45
(Super-XX), PDF 55 = printed 51, PDF 56 = printed 53.

⚠⚠ THE BOOKLET IS FULL OF PLOTTED CURVES, WHICH IS THE OPPOSITE OF WHAT THIS
PROJECT'S OTHER 1940s SOURCES GIVE. Every one of the 24 specification sheets
carries a characteristic-curve FAMILY with a time-gamma and a time-fog inset,
and three multi-film comparison panels sit at printed 40, 58 and 64. They are
letterpress line drawings, not halftones: measured at native resolution the
panels run 441 px per 1.0 density and the curve stroke is 5–7 px ≈ 0.013 D, so
the binding accuracy limit is the original draughtsman, not the scan. **The
time-gamma families below were read off those insets and are the substance of
this module.**

WHAT WAS WIRED INTO THE DATABASE, AND WHAT WAS NOT
-----------------------------------------------------
Exactly one profile in this database is covered by the booklet:
`EASTMAN_SUPER_XX_1938`, which the booklet calls **Type 1232**. It now carries
the four-point SD-21 family from printed page 45, and that family independently
locates the 0.65 aim the same page states in prose at 12 minutes.

⚠ THE OTHER THIRTY-ONE FAMILIES ARE HELD HERE AND ARE WIRED NOWHERE, AND THAT
IS A DECISION RATHER THAN AN OMISSION. They belong to films this database does
not model: 1942 Eastman picture negatives, bi-pack separation stocks,
release/duplicating positives and sound-recording stocks. Writing them onto a
modern profile of the same trade name would transfer a 1942 emulsion's
development response onto a coating made decades later. They are preserved in
full so that the reading survives the scan, and so that a later pass that adds
the profiles has the numbers without re-reading 98 renders.

⚠ AND THE BOOKLET REFUSES TO CALL THEM PROCESS TIMES. Printed page 17: «Times
of development cannot be specified because of the dissimilarity of various
types of continuous processing machines». Every minute figure in this module is
a laboratory-sensitometer time at 65 °F, which is a different object from a
lab's running time and must not be relabelled as one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import film_profiles as FP            # noqa: E402

SOURCE = ("«Eastman Motion Picture Films for Professional Use», Eastman Kodak "
          "Company, (c) 1942, body sheets imprinted 12-44 with a SEPTEMBER "
          "1943 supplement; 98 pages, no text layer, read off 560 dpi renders "
          "2026-09-23")

#: All sensitometric development in the booklet is at this one temperature.
SENSITOMETER_F = 65.0
SENSITOMETER_C = 18.3

#: ⚠ THE SPEED SCALES ARE NOT ASA AND THE BOOKLET NEVER OFFERS A CONVERSION.
#: Printed sheets give **Kodak Film Speed**, **Weston** and **G. E.** for the
#: camera negatives, and four unrelated arbitrary **Exposure Number** scales for
#: the positive, duplicating-negative, duplicating-positive and sound stocks.
#: Treating a Kodak Film Speed as an ASA figure is the single easiest mistake to
#: make with this document, so the scale is carried with every value.
SPEED_SCALES = ("Kodak Film Speed", "Weston", "G. E.",
                "Release Positive Exposure No.", "Duplicating Negative Exp. No.",
                "Duplicating Positive Exp. No.", "Sound Recording Exp. No.")

#: ⚠ TIME-GAMMA FAMILIES READ OFF THE PER-SHEET INSETS. Key is the booklet's own
#: type number; value is (developer, printed page, ((minutes, gamma), ...)).
#: All at 65 degF. These are the booklet's own labelled family members, not a
#: curve trace — the panels label each curve with its development time.
TIME_GAMMA: dict[str, tuple[str, int, tuple[tuple[float, float], ...]]] = {
    # -- picture negatives, all Kodak SD-21 -------------------------------
    "1210": ("Kodak SD-21", 41, ((3.0, 0.36), (4.0, 0.51), (5.5, 0.67),
                                 (7.0, 0.85), (9.0, 1.06))),
    "1213": ("Kodak SD-21", 42, ((4.0, 0.46), (5.5, 0.62), (7.0, 0.74),
                                 (9.0, 0.86), (12.0, 1.02))),
    "1230": ("Kodak SD-21", 43, ((7.0, 0.52), (9.0, 0.62), (12.0, 0.78),
                                 (18.0, 1.07))),
    "1231": ("Kodak SD-21", 44, ((7.0, 0.55), (9.0, 0.65), (12.0, 0.78),
                                 (18.0, 1.03))),
    "1232": ("Kodak SD-21", 45, ((7.0, 0.46), (9.0, 0.54), (12.0, 0.65),
                                 (18.0, 0.87))),
    "5240": ("Kodak SD-21", 46, ((7.0, 0.47), (9.0, 0.59), (12.0, 0.74),
                                 (18.0, 1.03))),
    "5242": ("Kodak SD-21", 47, ((7.0, 0.43), (9.0, 0.51), (12.0, 0.60),
                                 (18.0, 0.75))),
    # -- bi-pack -----------------------------------------------------------
    "1234": ("Kodak SD-21", 49, ((7.0, 0.64), (9.0, 0.70), (12.0, 0.76),
                                 (18.0, 0.81))),
    "1235 through 1234": ("Kodak SD-21", 50, ((7.0, 0.57), (9.0, 0.65),
                                              (12.0, 0.75), (18.0, 0.93))),
    "1235 through 1236": ("Kodak SD-21", 50, ((7.0, 0.59), (9.0, 0.70),
                                              (12.0, 0.82), (18.0, 1.04))),
    "1236": ("Kodak SD-21", 51, ((7.0, 0.51), (9.0, 0.55), (12.0, 0.60),
                                 (18.0, 0.66))),
    # -- positives ---------------------------------------------------------
    "1301": ("Kodak D-16", 53, ((3.0, 1.73), (4.0, 1.97), (5.5, 2.22),
                                (8.0, 2.43))),
    "1302": ("Kodak D-16", 55, ((2.5, 2.20), (3.0, 2.49), (4.0, 2.72),
                                (5.5, 2.99), (8.0, 3.24))),
    "1363": ("Kodak D-16", 56, ((3.0, 3.50), (4.0, 3.66), (5.5, 3.83),
                                (8.0, 4.00))),
    # -- duplicating -------------------------------------------------------
    "1203": ("Kodak SD-21", 59, ((3.0, 0.40), (4.0, 0.49), (5.5, 0.62),
                                 (7.0, 0.75), (9.0, 0.92))),
    "1505": ("Kodak SD-21", 60, ((4.0, 0.45), (5.5, 0.57), (7.0, 0.67),
                                 (9.0, 0.79), (12.0, 0.92))),
    "1355": ("Kodak D-16", 61, ((3.0, 1.47), (4.0, 1.61), (5.5, 1.73),
                                (8.0, 1.81))),
    "1362": ("Kodak D-16", 62, ((3.0, 1.67), (4.0, 1.92), (5.5, 2.15),
                                (8.0, 2.33))),
    "1365": ("Kodak SD-21", 63, ((4.0, 0.85), (5.5, 1.01), (7.0, 1.17),
                                 (9.0, 1.35), (12.0, 1.55))),
    # -- sound recording. ⚠ 1357 CARRIES TWO FAMILIES ON ONE SHEET, one per
    # -- recording method, and they are three orders of contrast apart.
    "1357 variable area (D-16)": ("Kodak D-16", 66, ((5.5, 2.03), (8.0, 2.24),
                                                     (12.0, 2.40),
                                                     (20.0, 2.50))),
    "1357 variable density (D-103)": ("Kodak D-103", 66,
                                      ((3.5, 0.14), (4.5, 0.23), (6.0, 0.36),
                                       (8.0, 0.56))),
    "1370": ("Kodak D-103", 67, ((3.5, 0.14), (4.5, 0.24), (6.0, 0.42),
                                 (8.0, 0.60))),
    "1301 sound": ("Kodak D-103", 68, ((3.5, 0.18), (4.5, 0.32), (6.0, 0.49),
                                       (8.0, 0.72))),
    "1302 sound": ("Kodak D-16", 69, ((2.5, 2.20), (3.0, 2.49), (4.0, 2.72),
                                      (5.5, 2.99), (8.0, 3.24))),
    # -- 1943 supplement ---------------------------------------------------
    "1372": ("Kodak D-16", 89, ((5.5, 2.98), (8.0, 3.05), (12.0, 3.14),
                               (20.0, 3.28))),
    "1373": ("Kodak SD-21", 90, ((4.0, 0.38), (5.5, 0.46), (7.0, 0.55),
                                 (9.0, 0.64), (12.0, 0.75))),
}

#: The three multi-film comparison panels, as (printed page, films, gammas).
#: ⚠ A COMPARISON PANEL IS NOT A FAMILY. Each draws several FILMS at ONE aim,
#: where a family draws one film at several times. Merging the two would
#: destroy both, which is why they are separate tables.
COMPARISON_PANELS = {
    "EASTMAN PICTURE NEGATIVE FILMS": (
        40, ("1232", "1231", "1230", "1213"), (0.65, 0.65, 0.65, 0.75),
        "Exposed to Sunlight (Type IIb Sensitometer), developed in Kodak "
        "SD-21 at 65 degF"),
    "EASTMAN DUPLICATING FILMS": (
        58, ("1362", "1355", "1203", "1505", "1365"),
        (2.10, 1.70, 0.65, 0.65, 1.40),
        "1355 and 1362 in Kodak D-16 at 65 degF; 1203, 1505 and 1365 in "
        "Kodak SD-21 at 65 degF"),
    "EASTMAN SOUND RECORDING FILMS": (
        64, ("1357", "1302", "1370", "1357", "1301"),
        (2.30, 2.70, 0.55, 0.35, 0.35),
        "1357 and 1302 in Kodak D-16 (variable area); 1357, 1370 and 1301 in "
        "Kodak D-103 (variable density), all at 65 degF"),
}

#: ⚠ THE FOURTH PANEL IS MISSING FROM THIS COPY. By the pattern of pages 40, 58
#: and 64 an EASTMAN POSITIVE FILMS comparison panel belongs on printed page 52
#: — the one leaf absent from the scan — and would carry 1301/1302/1363/1509.
#: Recorded so that a future reader knows to look for it rather than concluding
#: the positives were never drawn together.
MISSING_LEAF = (52, "EASTMAN POSITIVE FILMS comparison panel, presumed "
                    "1301 / 1302 / 1363 / 1509")

#: Resolving power, lines/mm, as printed. ⚠ NO TEST-OBJECT CONTRAST IS GIVEN
#: ANYWHERE IN THE BOOKLET, so none of these is comparable with a modern
#: two-contrast pair without saying so.
RESOLVING_POWER = {
    "1213": 60, "1230": 60, "1231": 55, "1232": 55, "5240": 60, "5242": 55,
    "1234": 110, "1235": 55, "1236": 110, "1301": 55, "1302": 90, "1363": 95,
    "1509": 95, "1203": 110, "1505": 75, "1355": 50, "1362": 55, "1365": 150,
    "1357": 50, "1370": 90, "1372": 150,
}

#: ⚠ HALATION IS CONTROLLED BY A TINTED SUPPORT AND BY DYE IN THE EMULSION, AND
#: THE WORD "ANTIHALATION" NEVER APPEARS. There is no backing layer, no rem-jet
#: and no AH undercoat anywhere in 98 pages. The negatives are coated on GRAY
#: nitrate (35 mm) or BLUE-GRAY acetate (16 mm); 1505, 1365 and 1509 carry a
#: YELLOW DYE IN THE EMULSION which the text credits with cutting internal
#: scatter; 1355 and the duplicating positives use a LAVENDER base.
HALATION_CONTROL = {
    "picture negatives": "gray nitrate base (35 mm) / blue-gray acetate "
                         "base (16 mm) -- a tinted SUPPORT, not a backing",
    "1505 / 1365 / 1509": "yellow dye IN THE EMULSION; the text credits it "
                          "with extending latitude and lowering maximum "
                          "contrast as well as cutting scatter",
    "duplicating positives": "lavender base",
}

#: Products the booklet itself declares to be the same emulsion as another, so
#: they must never become separate profiles. Quoted, because the booklet's own
#: wording is the evidence.
SAME_EMULSION_AS = {
    "1401 News Positive": ("1301", "«The same emulsion is furnished under the "
                           "name Eastman News Positive, Type 1401, on a "
                           "thinner nitrate base»"),
    "1402 Fine Grain News Positive": ("1302", "«The same emulsion … on a "
                                      "thinner nitrate base»"),
    "1301 for Sound Recording": ("1301", "«This is the regular Release "
                                 "Positive emulsion provided with footage "
                                 "numbers»"),
    "1302 for Sound Recording": ("1302", "«This is the regular Fine Grain "
                                 "Release Positive emulsion, provided with "
                                 "footage numbers»"),
    "Sonochrome 1316-1348": ("1301 and 1302", "«emulsion same as Release "
                             "Positive, Type 1301» / «emulsion same as Fine "
                             "Grain Release Positive, Type 1302» -- 28 "
                             "tinted-base variants of two emulsions"),
    "5240": ("1230", "«It is similar to Background-X Panchromatic, Type "
             "1230» -- same speed, same aim, same resolving power"),
    "5242": ("1232", "«It is similar to Super-XX Panchromatic, Type 1232» -- "
             "same speed, same aim, same resolving power; a 16 mm acetate "
             "gauge variant of the stock this database already holds"),
}

#: Distinct emulsions with a speed, a development aim and at least one
#: structural figure — i.e. enough to support a defensible profile if this
#: project ever models 1940s release and duplicating stocks.
PROFILE_READY = (
    "1213", "1230", "1231", "1234", "1236", "1301", "1302", "1363", "1509",
    "1203", "1505", "1355", "1362", "1365", "1357", "1370", "1372",
)
#: Justified but structurally thin, with the reason.
PROFILE_THIN = {
    "1210": "speed, aim and a five-point family, but NO resolving power and "
            "no graininess",
    "1210 Increased Speed (1943)": "supersedes 1210 rather than joining it -- "
                                   "«50 per cent faster than the previous "
                                   "Type 1210», and the developer changes "
                                   "from SD-21 to seasoned D-76",
    "1235": "aim, resolving power and TWO curve families, but NO SPEED OF ITS "
            "OWN -- the sheet points at 1234 and 1236",
    "1373": "curves only on the recovered supplement sheet; no speed, no "
            "resolving power",
}


def run(do_assert: bool = True) -> int:
    fail: list[str] = []

    # ---- 1. the family that was wired ------------------------------------
    print("=== the one family wired into the database ===")
    sx = FP.get_profile("EASTMAN_SUPER_XX_1938")
    want = TIME_GAMMA["1232"][2]
    got = tuple((p.minutes, p.gamma) for p in sx.processing_family.points)
    print("  EASTMAN_SUPER_XX_1938 (Type 1232), %s, printed p.%d"
          % (TIME_GAMMA["1232"][0], TIME_GAMMA["1232"][1]))
    for m, g in got:
        print("     %5.1f min -> gamma %.2f" % (m, g))
    if got != want:
        fail.append("the wired Super-XX family no longer matches the reading: "
                    "%s vs %s" % (got, want))

    # ⚠⚠ THE COROBORATION IS THE POINT AND IT IS CHECKED, NOT ASSERTED. The
    # page's PROSE gives the 0.65 aim; the page's PLOT puts 0.65 at 12 min.
    # Two transcriptions, two parts of one page, and they agree exactly.
    at12 = dict(want)[12.0]
    print("  ⚠ the prose aim %.2f and the inset's 12-minute member %.2f agree "
          "exactly -- independently transcribed from different parts of "
          "printed page 45" % (sx.processing.contrast_index, at12))
    if abs(sx.processing.contrast_index - at12) > 1e-9:
        fail.append("the stated aim and the plotted 12-minute gamma have "
                    "drifted apart: %.3f vs %.3f"
                    % (sx.processing.contrast_index, at12))
    if sx.processing.developer != "Kodak SD-21":
        fail.append("Super-XX's developer is no longer the SD-21 the family "
                    "is measured in")

    # ⚠ AND THE FAMILY MUST BE MONOTONE IN TIME. A development ladder that
    # falls somewhere is a transcription error, not a film.
    for key, (dev, pg, pts) in sorted(TIME_GAMMA.items()):
        ms = [m for m, _g in pts]
        gs = [g for _m, g in pts]
        if ms != sorted(ms) or gs != sorted(gs):
            fail.append("%s (p.%d) is not monotone in time and gamma: %s"
                        % (key, pg, pts))

    print("\n=== all %d time-gamma families held here ===" % len(TIME_GAMMA))
    for key, (dev, pg, pts) in sorted(TIME_GAMMA.items()):
        span = pts[-1][1] / pts[0][1] if pts[0][1] else float("nan")
        print("  %-30s p.%-3d %-13s %d pts, gamma %.2f -> %.2f (x%.2f)"
              % (key, pg, dev, len(pts), pts[0][1], pts[-1][1], span))

    # ⚠ THE SOUND STOCKS ARE THE EXTREME CASE AND THEY SHOULD LOOK LIKE ONE.
    # 1357 is drawn twice on one sheet: variable-AREA needs a near-binary
    # image and variable-DENSITY needs a near-linear one, so the SAME emulsion
    # is developed to gamma 2.03-2.50 in D-16 and to 0.14-0.56 in D-103. That
    # is a factor of fifteen in contrast from one coating, and it is the
    # strongest statement in this corpus that development, not emulsion,
    # decides contrast.
    va = dict(TIME_GAMMA["1357 variable area (D-16)"][2])
    vd = dict(TIME_GAMMA["1357 variable density (D-103)"][2])
    ratio = max(va.values()) / min(vd.values())
    print("  ⚠ Type 1357, ONE emulsion, TWO processes on one sheet: D-16 "
          "reaches gamma %.2f and D-103 starts at %.2f -- a factor of %.0f"
          % (max(va.values()), min(vd.values()), ratio))
    if ratio < 10:
        fail.append("the 1357 two-process spread has collapsed; it should be "
                    "more than tenfold")

    # ---- 2. comparison panels --------------------------------------------
    print("\n=== the three multi-film comparison panels ===")
    for name, (pg, films, gammas, legend) in COMPARISON_PANELS.items():
        print("  p.%-3d %-34s %s" % (pg, name,
                                     ", ".join("%s(g%.2f)" % (f, g)
                                               for f, g in zip(films, gammas))))
        print("        %s" % legend)
    print("  ⚠ MISSING LEAF: printed p.%d, %s" % MISSING_LEAF)
    # The picture-negative panel must agree with the per-film families where
    # they overlap: it draws 1231 and 1230 at 0.65, and both families reach
    # 0.65 inside their own printed range.
    for t in ("1231", "1230", "1232"):
        lo, hi = TIME_GAMMA[t][2][0][1], TIME_GAMMA[t][2][-1][1]
        if not lo <= 0.65 <= hi:
            fail.append("the p.40 panel draws %s at gamma 0.65 but that stock's"
                        " own family spans only %.2f-%.2f" % (t, lo, hi))
    print("  ⚠ the p.40 panel's 0.65 aim lies inside the printed family range "
          "of 1232, 1231 and 1230 -- panel and insets agree")

    # ---- 3. structure ------------------------------------------------------
    print("\n=== resolving power, lines/mm, NO test-object contrast anywhere ===")
    best = max(RESOLVING_POWER.items(), key=lambda kv: kv[1])
    worst = min(RESOLVING_POWER.items(), key=lambda kv: kv[1])
    print("  %d stocks; sharpest %s at %d, softest %s at %d"
          % (len(RESOLVING_POWER), best[0], best[1], worst[0], worst[1]))
    # The booklet's own claim: 1365 is "extremely low graininess and
    # exceptionally high resolving power". It should be the maximum.
    if best[0] != "1365" or best[1] != 150:
        fail.append("the booklet calls 1365 exceptionally high in resolving "
                    "power; the table no longer makes it the maximum")
    if RESOLVING_POWER["1232"] != FP.get_profile(
            "EASTMAN_SUPER_XX_1938").mtf.resolving_power_lp_mm_highc:
        fail.append("Super-XX's stored resolving power no longer matches the "
                    "booklet's 55 lines/mm")
    print("  ⚠ Super-XX 1232 = %d l/mm, and the database carries the same"
          % RESOLVING_POWER["1232"])

    print("\n=== halation control: a TINTED SUPPORT, never a backing ===")
    for k, v in HALATION_CONTROL.items():
        print("  %-24s %s" % (k, v))

    print("\n=== products the booklet declares to be another's emulsion ===")
    for k, (parent, quote) in SAME_EMULSION_AS.items():
        print("  %-30s -> %-16s %s" % (k, parent, quote[:64]))
    if "5242" not in SAME_EMULSION_AS:
        fail.append("the 5242 == 1232 identity has been lost; it is what "
                    "keeps a 16 mm gauge variant from becoming a second "
                    "Super-XX profile")

    print("\n=== profile support, if 1940s stocks are ever modelled ===")
    print("  ready (speed + aim + structure + own family): %d -- %s"
          % (len(PROFILE_READY), ", ".join(PROFILE_READY)))
    for k, why in PROFILE_THIN.items():
        print("  thin: %-30s %s" % (k, why))

    # ---- 4. absences -------------------------------------------------------
    print("\n=== recorded absences, 98 pages inspected ===")
    for line in (
            "no granularity or graininess NUMBER of any kind -- only prose "
            "adjectives, and printed p.9 says comparisons «can be made only "
            "by careful examination of screen tests»",
            "no MTF, sine-wave response, edge trace or acutance; printed p.9 "
            "declares sharpness «not so well adapted to numerical "
            "presentation as resolving power»",
            "no base or emulsion thickness for any camera, print, duplicating "
            "or sound stock -- thickness is printed only for leader",
            "no true spectral-sensitivity curves: colour sensitivity is shown "
            "ONLY by halftone wedge spectrograms whose wavelength numerals "
            "are UNREADABLE at this scan, and whose UV end the booklet itself "
            "says understates the truth",
            "no ASA, DIN, Scheiner or H&D speed, and no conversion offered",
            "no reciprocity, no latent-image keeping, no storage or fading "
            "data",
            "no printed latitude figure -- the word appears four times as a "
            "concept and never as a number",
            "no dye densities, no status densitometry, no colour negative or "
            "print stock beyond the two 16 mm Kodachrome insert sheets"):
        print("  * " + line)

    if fail and do_assert:
        print("\n".join(["", "FAILURES:"] + ["  " + f for f in fail]))
        return 1
    print("\n[OK] kodak_1942_eastman.py -- «Eastman Motion Picture Films for "
          "Professional Use» (Kodak, 1942), 98 pages with NO TEXT LAYER, read "
          "off 560 dpi renders. ⚠⚠ THE BOOKLET IS FULL OF PLOTTED CURVES, "
          "which is the opposite of what this project's other wartime sources "
          "give: every one of the 24 specification sheets carries a "
          "characteristic-curve family with a time-gamma inset, three "
          "multi-film comparison panels sit at printed 40, 58 and 64, and all "
          "are letterpress line drawings digitisable to about 0.013 D. %d "
          "time-gamma families were read off those insets. ⚠ EXACTLY ONE IS "
          "WIRED: Type 1232 is this database's EASTMAN_SUPER_XX_1938, which "
          "now carries 7/9/12/18 min -> 0.46/0.54/0.65/0.87 in Kodak SD-21 at "
          "65 degF, and whose 12-minute member reproduces the 0.65 aim the "
          "same page states in prose -- two transcriptions from one page that "
          "agree. ⚠ THE OTHER %d ARE HELD AND WIRED NOWHERE, deliberately: "
          "they belong to 1942 emulsions this database does not model, and "
          "attaching them to a modern profile of the same trade name would "
          "transfer a wartime coating's development response onto one made "
          "decades later. ⚠ AND THE BOOKLET REFUSES TO CALL THEM PROCESS "
          "TIMES -- printed p.17, «Times of development cannot be specified "
          "because of the dissimilarity of various types of continuous "
          "processing machines»"
          % (len(TIME_GAMMA), len(TIME_GAMMA) - 1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="")
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                    default=True)
    ns = ap.parse_args(argv)
    return run(ns.do_assert)


if __name__ == "__main__":
    sys.exit(main())
