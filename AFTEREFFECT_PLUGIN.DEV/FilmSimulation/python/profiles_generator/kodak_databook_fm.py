#!/usr/bin/env python3
"""The FM section of the «Kodak Data Book of Applied Photography», read whole.

WHAT THE DOCUMENT IS, AND THE THING THAT DECIDES EVERYTHING ELSE
-------------------------------------------------------------------
Pages 1140–1495 of `KODAK DATA BOOK.pdf`, held in this project as four chunks
under `PDF/PROFILES/KODAK/_databook_chunks/`. The FM section is the loose-leaf
film data sheets: 1152 to 1495, index pages carrying `PDDB-26/xWP9/11-67` and
`.../12-68`.

⚠⚠ IT IS **KODAK LIMITED, LONDON**, NOT EASTMAN KODAK OF ROCHESTER, AND THAT
FACT REWRITES THE TASK. Task #502 was written expecting development tables for
KODAK PLUS-X 125PX, VERICHROME PAN, TRI-X 400TX, ROYAL-X PAN 4166 and EASTMAN
PLUS-X 5231. An exhaustive text sweep of all 356 pages returns **zero**
occurrences of `5231`, `4166`, `125PX`, `400TX`, or `EASTMAN` as a film
designation. The word "Eastman" occurs twice and neither is a product: a
contents-list entry for 16 mm Eastman Fine-Grain Release Positive (p.1146) and
"made by the Eastman Kodak Company of the U.S.A." describing a Micro-File
machine (p.1472). This book uses British product naming and Kodak Ltd type
numbers throughout.

⚠⚠ SO THE DEVELOPMENT TABLES ARE **NOT WRITTEN ONTO THE DATABASE'S PROFILES**,
AND THE REASON IS NOT CAUTION -- IT IS THAT THEY DESCRIBE DIFFERENT EMULSIONS.
The database's `KODAK_PLUS_X_125` comes from Kodak F-4018 (2007),
`KODAK_TRI_X_400TX` from F-4017 (2005–2016), `KODAK_VERICHROME_PAN` from F-7
(1996). The sheets here are British coatings of the middle 1960s. The section
proves the gap in its own pages: FM-36 Issue B is "'Plus-X' Sheet Film" at ASA
**160** developed in D-61a to gamma 1.0/0.8/0.65, and FM-36 Issue C is
"'Plus-X' **PAN** Sheet Film" at ASA **125** developed in DK-50 to gamma
0.55/0.7. One FM number, one trade name, two emulsions a generation apart, with
a different principal developer and a different contrast regime. Transferring
either table onto a 2007 American coating would assert a development response
nobody measured.

WHAT THE SECTION *DID* CLOSE
-------------------------------
Three claims this project had been carrying unverified, all now settled against
page evidence; a quantified agitation law; an independent corroboration of the
Papers G-1 print model from a completely different Kodak document; and a
resolving-power figure for Super-XX that must be kept BESIDE the one the
database already holds rather than replacing it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import film_profiles as FP            # noqa: E402

try:
    import kodak_papers_g1 as G1      # noqa: E402
except Exception:                     # pragma: no cover
    G1 = None

SOURCE = ("«Kodak Data Book of Applied Photography», Kodak Limited, London, "
          "section FM, pp.1140-1495, read 2026-09-23 from "
          "PDF/PROFILES/KODAK/_databook_chunks/")

#: ⚠ THE PUBLISHER IS THE FINDING. Everything below is conditioned on it.
PUBLISHER = "Kodak Limited, London"
US_DESIGNATIONS_PRESENT = ()          # 5231, 4166, 125PX, 400TX: none occur

# ---------------------------------------------------------------------------
#  CLAIM 1 -- the D-76 / gamma 0.7 sentence
# ---------------------------------------------------------------------------
#: ⚠ AND IT IS A SPEED QUALIFIER, NOT A PROCESSING INSTRUCTION. The sentence
#: reads «These figures apply when this film is developed in 'Kodak' D-76
#: Developer to a gamma of 0·7» -- it says what the ASA number means, not what
#: the user should do. Reading it as an instruction is how it became a supposed
#: blanket rule in the first place.
D76_GAMMA_07_SHEETS = {
    "FM-45": (1400, "Tri-X Pan Roll Film, Issue C"),
    "FM-46": (1406, "Tri-X Pan Professional Roll Film, Issue B"),
    "FM-48": (1418, "Plus-X Pan Professional Roll Film, Issue A"),
}
#: ⚠ THE STRONGEST EVIDENCE THAT IT IS NOT BLANKET is a sheet that could have
#: carried it and does not: FM-49 Verichrome Pan is the same 125 ASA, the same
#: vintage, the same roll format, and its speed table has NO development
#: condition attached at all.
D76_GAMMA_07_ABSENT_FROM = {"FM-49": (1426, "Verichrome Pan, Issue C, 125 ASA "
                                            "-- same speed and vintage, no "
                                            "condition on its speed table")}

# ---------------------------------------------------------------------------
#  CLAIM 1b -- Royal-X contradicts it, and more sharply than expected
# ---------------------------------------------------------------------------
#: Both Royal-X sheets reject every developer but one, in a sentence that puts
#: D-76 among the rejected: «For the best results ONLY 'Kodak' Professional
#: Time-Standard Developer or a developer made up according to Kodak formula
#: DK-50 should be used. MOST OTHER DEVELOPERS do not give the fullest emulsion
#: speed, or they increase the fog, or both.»
ROYAL_X_DEVELOPER_RESTRICTION = {
    "FM-44": (1396, ("Kodak Professional Time-Standard", "DK-50"),
              "sheet film"),
    "FM-50": (1434, ("DK-50",), "roll film"),
}
#: ⚠ AND ROYAL-X IS THE ONLY FILM IN THE SECTION WITH NO SAFELIGHT AT ALL.
#: «Safelighting: NONE -- this film should be handled only in total darkness»,
#: plus a MANDATORY acid stop bath «to minimize the tendency to dichroic fog».
#: A 1250 ASA emulsion of 1960 is close enough to fogging that the manufacturer
#: withdrew the darkroom lamp, which is a physical statement about the coating.
ROYAL_X_NO_SAFELIGHT = True
ROYAL_X_ACID_STOP_MANDATORY = True

#: Royal-X's documented two-mode system -- a push expressed as a speed, which
#: is rare in this corpus. «With extended development … halve the exposure by
#: doubling the arithmetical speed, or by adding 3 to the logarithmic speeds.»
ROYAL_X_NORMAL_ASA = 1250
ROYAL_X_EXTENDED_ASA = 2500
ROYAL_X_EXTENDED_STOPS = 1.0
#: DK-50 undiluted at 20 degC, roll film (FM-50 p.1434): normal 7 min
#: continuous / 9 min intermittent; extended 12 min continuous.
ROYAL_X_DK50_MIN = {"normal continuous": 7.0, "normal intermittent": 9.0,
                    "extended continuous": 12.0}
#: The gamma ladder printed on FM-50's own curve plot, DK-50 undiluted at
#: 20 degC with INTERMITTENT agitation.
ROYAL_X_TIME_GAMMA = ((5.0, 0.58), (7.0, 0.65), (10.0, 0.75))
#: ⚠ AND A SEPARATE PUSH FOR ONE LIGHT SOURCE: «For exposures made with
#: high-voltage, electronic-flash, studio units, it may be desirable to develop
#: for 50 per cent longer. This does not normally apply when the low-voltage,
#: portable units are used.» A development correction keyed to the SHAPE of the
#: exposure pulse -- reciprocity behaviour named without the word.
ROYAL_X_ELECTRONIC_FLASH_DEV_FACTOR = 1.50

# ---------------------------------------------------------------------------
#  CLAIM 2 -- Weston
# ---------------------------------------------------------------------------
#: ⚠ CONFIRMED, AND THE PATTERN IS SHARPER THAN THE CLAIM. "Weston" occurs on
#: 24 pages, ALL of them colour sheets (1165-1257), and never as a speed scale:
#: every occurrence is the footnote «These figures should be used with the
#: Weston Master III [and IV] meter; with all earlier models, a figure
#: approximating to 4/5ths of these figures should be used» -- a METER-MODEL
#: correction, not a Weston speed. Zero occurrences on any page from 1258 to
#: 1495, which is every monochrome sheet in the section.
WESTON_PAGES_ALL_COLOUR = True
WESTON_FIRST_MONOCHROME_PAGE = 1258
MONOCHROME_SPEED_SCALES = ("B.S./ASA arithmetical", "B.S. logarithmic", "DIN")

# ---------------------------------------------------------------------------
#  CLAIM 3 -- Super-XX resolving power
# ---------------------------------------------------------------------------
#: ⚠⚠ FOUND, AND IT DISAGREES WITH WHAT THE DATABASE HOLDS -- SO IT IS RECORDED
#: BESIDE IT AND NOT OVER IT. FM-37 p.1358, «KODAK 'SUPER-XX' SHEET FILM»:
#: "Resolving power: 60 lines per millimetre when developed as recommended.
#: This is a figure relating to the emulsion only; the actual figure obtained
#: will depend largely upon other factors, such as the subject contrast and the
#: optical system used."
#:
#: ⚠ THEY ARE NOT THE SAME MEASUREMENT AND MAY NOT BE THE SAME FILM. The stored
#: 55 l/mm is Eastman Kodak's, from the 1942 US booklet, in Kodak SD-21. This
#: 60 is Kodak Ltd's, for a British sheet film, "as recommended" -- which on
#: FM-37 means D-61a or D-76 -- and with NO test-object contrast stated, the
#: sheet explicitly disclaiming contrast dependence instead. Averaging them
#: would produce a number neither company published.
SUPER_XX_FM37 = {
    "lines_per_mm": 60.0, "page": 1358, "sheet": "FM-37",
    "publisher": PUBLISHER,
    "developer": "as recommended -- D.61a or D.76 (p.1358 prose, p.1360 table)",
    "test_object_contrast": None,
    "adopted": False,
    "why_not_adopted": (
        "the database's 55 l/mm is an EASTMAN KODAK figure for the US 1942 "
        "Type 1232 in Kodak SD-21; this is a KODAK LIMITED figure for a "
        "British sheet film in D-61a/D-76 with no test-object contrast "
        "stated. Different company, different developer, possibly a different "
        "coating. Kept as a second reading, not a correction."),
}
#: FM-37's own development table, the one the 60 l/mm is keyed to. All 68 degF.
SUPER_XX_FM37_TIME_GAMMA = {
    ("D.61a", "1+1", "continuous"): ((10.0, 1.0), (5.5, 0.8)),
    ("D.61a", "1+3", "intermittent"): ((19.0, 1.0), (11.0, 0.8)),
    ("D.76", "undiluted", "continuous"): ((13.0, 0.8), (8.0, 0.65)),
    ("D.76", "undiluted", "intermittent"): ((16.0, 0.8), (10.0, 0.65)),
    ("Microdol", "undiluted", "continuous"): ((13.0, 0.8), (9.0, 0.65)),
    ("Microdol", "undiluted", "intermittent"): ((16.0, 0.8), (11.0, 0.65)),
}
#: ⚠ AND THE SHEET'S OWN SPEED TABLE IS PHYSICALLY OBSCURED. An orange
#: amendment slip is pasted across p.1358 giving NEW figures (ASA 200 / B.S.
#: 34° / DIN 24) and covering whatever was printed underneath. The original is
#: UNREADABLE -- not degraded, covered.
SUPER_XX_FM37_SPEED_SLIP = (200, "34 deg B.S.", 24)
SUPER_XX_FM37_ORIGINAL_SPEED = None   # obscured by the slip

#: ⚠ A THIRD SUPER-XX FIGURE EXISTS IN THE SECTION AND MUST NOT BE CONFLATED:
#: FM-59 «KODAK 'SUPER-XX' AERO FILM» p.1464 gives 35 lines/mm at test-object
#: contrast 1.5:1, speed 100. A different emulsion for a different job.
SUPER_XX_AERO = {"lines_per_mm": 35.0, "toc": "1.5:1", "asa": 100,
                 "page": 1464, "sheet": "FM-59"}

# ---------------------------------------------------------------------------
#  THE AGITATION LAW -- quantified two ways, and they agree
# ---------------------------------------------------------------------------
#: ⚠ THE SECTION STATES A BLANKET RULE AND ALSO PRINTS THE PAIRED ROWS THAT
#: TEST IT, which is unusual and worth keeping. The newer sheets say «For
#: continuous agitation the times given should be decreased by about 20 per
#: cent»; the older sheets instead print continuous and intermittent rows for
#: the same developer and gamma, and those pairs come out at +26 % to +31 %.
#: Same law from two directions, and the printed rule is the conservative end.
AGITATION_RULE_CONTINUOUS_REDUCTION = 0.20
AGITATION_MEASURED_PAIRS = (
    # (sheet, page, developer, gamma, continuous min, intermittent min)
    ("FM-53", 1456, "D.76", 0.8, 11.0, 14.0),
    ("FM-36 Issue B", 1349, "D-76", 0.8, 13.5, 17.0),
    ("FM-36 Issue B", 1349, "Microdol", 0.65, 13.0, 17.0),
    ("FM-50", 1434, "DK-50", None, 7.0, 9.0),
)
#: The two agitation definitions the section uses, verbatim, because a time
#: without one is not a time:
AGITATION_DEFINITIONS = {
    "intermittent (older sheets)": "thorough but brief agitation at "
                                   "one-minute intervals",
    "large tank": "in 3-gallon tanks with film on spiral reels in a "
                  "processing basket, or in deep tanks, thorough agitation "
                  "for 5 seconds at one-minute intervals",
    "small tank": "in small, daylight, spiral-reel or apron tanks, thorough "
                  "agitation for 5 seconds at half-minute intervals",
}

# ---------------------------------------------------------------------------
#  THE CROSS-DOCUMENT CORROBORATION -- and it is the best find in the section
# ---------------------------------------------------------------------------
#: ⚠⚠ TWO KODAK DOCUMENTS, DIFFERENT COUNTRIES, DIFFERENT DECADES, DIFFERENT
#: SUBJECTS, ONE NUMBER. The FM sheets print a uniform rubric beside their
#: two-gamma development tables: «The lower contrast level [gamma 0.55] is
#: intended for use with opal-lamp condenser enlargers, or for subjects which
#: have average to high contrast. The higher level [gamma 0.7] is more suitable
#: for subjects of lower contrast, or for enlargers with completely diffuse
#: illumination.» That is a negative-gamma offset of 0.15 between a condenser
#: and a diffuser printing chain.
#:
#: `kodak_papers_g1.py`, built from the US «Kodak Data Book G-1» on paper
#: grades, independently carries NEGATIVE_GAMMA_PER_PAPER_GRADE = 0.15 and
#: DIFFUSE_VS_CONDENSER_GRADES = 1. So the British film sheets and the American
#: paper book agree that the condenser-to-diffuser change is worth EXACTLY ONE
#: PAPER GRADE, and that one grade is 0.15 of negative gamma. Neither document
#: cites the other and neither was consulted while the other was read.
FM_CONDENSER_GAMMA = 0.55
FM_DIFFUSER_GAMMA = 0.70
FM_ENLARGER_DELTA_GAMMA = 0.15
FM_ENLARGER_RUBRIC_SHEETS = ("FM-36 Issue C", "FM-45", "FM-46", "FM-48",
                             "FM-52 Issue D")

#: Developer speed penalty, stated on every sheet that offers the developer.
MICRODOL_EXPOSURE_PENALTY_STOPS = 0.5
#: ⚠ AND A SPEED PENALTY FOR REDUCED DEVELOPMENT, which is the same physics
#: from the other side: «For lower degrees of contrast, it may be necessary to
#: increase the exposure by up to about 1/2 stop» (FM-45, FM-46, FM-48).
LOW_CONTRAST_EXPOSURE_PENALTY_STOPS = 0.5

# ---------------------------------------------------------------------------
#  GENERATION RENAMING -- a hard fact about which emulsion a name refers to
# ---------------------------------------------------------------------------
#: ⚠ THIS TABLE IS WHY NOTHING HERE IS WIRED. Each row is one trade name
#: carrying two different emulsions inside one bound section.
GENERATION_SPLITS = {
    "FM-36": (("'Plus-X' Sheet Film", 160, "D-61a", (1.0, 0.8, 0.65)),
              ("'Plus-X' PAN Sheet Film", 125, "DK-50", (0.7, 0.55))),
    "FM-52": (("'Plus-X' Miniature Film", 80, "D.76", (0.8, 0.65)),
              ("'Plus-X' PAN Miniature Film", 125, "D-76 1+1", (0.7, 0.55))),
    "FM-53 vs FM-45": (("'Tri-X' Miniature Film", 200, "D.76", (0.8, 0.65)),
                       ("'Tri-X' PAN Roll Film", 400, "D-76", (0.7, 0.55))),
}
#: ⚠ AND TWO FM NUMBERS WERE REASSIGNED TO DIFFERENT EMULSIONS ENTIRELY between
#: the 11-67 and 12-68 printings of the contents pages: FM-35 is 'Ortho-Royal'
#: in this binding and Tri-X Ortho in the later contents; FM-37 is Super-XX
#: here and Tri-X Pan Professional Sheet later. An FM number is not a stable
#: identifier for an emulsion, which is worth knowing before citing one.
FM_NUMBERS_REASSIGNED = ("FM-35", "FM-37")


def run(do_assert: bool = True) -> int:
    fail: list[str] = []

    print("=== the finding that conditions everything else ===")
    print("  publisher: %s" % PUBLISHER)
    print("  US designations present in 356 pages: %d (5231, 4166, 125PX, "
          "400TX all absent)" % len(US_DESIGNATIONS_PRESENT))
    if US_DESIGNATIONS_PRESENT:
        fail.append("a US designation has been claimed for a Kodak Ltd book")

    print("\n=== claim 1: «developed in D-76 to a gamma of 0.7» ===")
    for sheet, (pg, what) in sorted(D76_GAMMA_07_SHEETS.items()):
        print("  present: %-6s p.%d  %s" % (sheet, pg, what))
    for sheet, (pg, why) in D76_GAMMA_07_ABSENT_FROM.items():
        print("  ⚠ ABSENT: %-6s p.%d  %s" % (sheet, pg, why))
    print("  verdict: NOT a blanket instruction -- three sheets out of the "
          "whole section, and it qualifies a SPEED rather than prescribing a "
          "process")
    if len(D76_GAMMA_07_SHEETS) != 3:
        fail.append("the D-76/0.7 sheet list is no longer the three the sweep "
                    "found: %s" % sorted(D76_GAMMA_07_SHEETS))

    print("\n=== claim 1b: Royal-X contradicts it ===")
    for sheet, (pg, devs, fmt) in sorted(ROYAL_X_DEVELOPER_RESTRICTION.items()):
        print("  %-6s p.%d %-11s only %s" % (sheet, pg, fmt, " or ".join(devs)))
    print("  ⚠ D-76 is not among them -- it falls under «most other "
          "developers do not give the fullest emulsion speed, or they "
          "increase the fog, or both»")
    print("  ⚠ and Royal-X is the only film in the section with NO safelight "
          "(%s) and a MANDATORY acid stop bath (%s)"
          % (ROYAL_X_NO_SAFELIGHT, ROYAL_X_ACID_STOP_MANDATORY))
    if any("D-76" in d for _pg, devs, _f in
           ROYAL_X_DEVELOPER_RESTRICTION.values() for d in devs):
        fail.append("D-76 has appeared in Royal-X's permitted developer list, "
                    "which is the opposite of what both sheets say")

    # ⚠ THE PUSH IS ARITHMETIC AND IT SHOULD CHECK OUT. "Double the
    # arithmetical speed" and "add 3 to the logarithmic speeds" are the same
    # statement only if the log scale is 10*log10 in thirds -- 3 units = 1
    # stop. If those two disagree, one was transcribed wrong.
    stops = (ROYAL_X_EXTENDED_ASA / ROYAL_X_NORMAL_ASA)
    print("  extended development: ASA %d -> %d, which is %.0f stop, and the "
          "sheet's other form «add 3 to the logarithmic speeds» is the same "
          "stop on a 1/3-stop log scale"
          % (ROYAL_X_NORMAL_ASA, ROYAL_X_EXTENDED_ASA, ROYAL_X_EXTENDED_STOPS))
    if abs(stops - 2.0 ** ROYAL_X_EXTENDED_STOPS) > 1e-9:
        fail.append("Royal-X's two statements of the same push disagree")
    print("  FM-50 DK-50 ladder: %s"
          % ", ".join("%.0f min -> %.2f" % p for p in ROYAL_X_TIME_GAMMA))
    print("  ⚠ electronic-flash correction: develop %.0f %% longer for "
          "high-voltage studio units, and NOT for low-voltage portables -- a "
          "development correction keyed to the shape of the exposure pulse"
          % ((ROYAL_X_ELECTRONIC_FLASH_DEV_FACTOR - 1.0) * 100))
    _g = [g for _m, g in ROYAL_X_TIME_GAMMA]
    if _g != sorted(_g):
        fail.append("the Royal-X ladder is not monotone")

    print("\n=== claim 2: Weston ===")
    print("  every «Weston» in pp.1140-1495 is on a COLOUR sheet and is a "
          "meter-model correction, not a speed scale; zero from p.%d onward, "
          "which is every monochrome sheet" % WESTON_FIRST_MONOCHROME_PAGE)
    print("  monochrome scales actually used: %s"
          % ", ".join(MONOCHROME_SPEED_SCALES))
    if not WESTON_PAGES_ALL_COLOUR:
        fail.append("the Weston finding has been weakened")

    print("\n=== claim 3: Super-XX resolving power ===")
    stored = FP.get_profile("EASTMAN_SUPER_XX_1938").mtf \
        .resolving_power_lp_mm_highc
    print("  FM-37 p.%d (%s): %.0f l/mm, developer «%s», test-object "
          "contrast %s" % (SUPER_XX_FM37["page"], PUBLISHER,
                           SUPER_XX_FM37["lines_per_mm"],
                           SUPER_XX_FM37["developer"],
                           SUPER_XX_FM37["test_object_contrast"]))
    print("  database holds: %.0f l/mm (Eastman Kodak 1942, Kodak SD-21)"
          % stored)
    print("  adopted: %s -- %s" % (SUPER_XX_FM37["adopted"],
                                   SUPER_XX_FM37["why_not_adopted"][:96]))
    if SUPER_XX_FM37["adopted"] or stored == SUPER_XX_FM37["lines_per_mm"]:
        fail.append("the Kodak Ltd 60 l/mm has displaced the Eastman Kodak "
                    "55 l/mm; they are different measurements of possibly "
                    "different coatings and must both stand")
    print("  ⚠ a THIRD figure exists and is not the same film: FM-59 Super-XX "
          "AERO, %.0f l/mm at TOC %s, ASA %d, p.%d"
          % (SUPER_XX_AERO["lines_per_mm"], SUPER_XX_AERO["toc"],
             SUPER_XX_AERO["asa"], SUPER_XX_AERO["page"]))
    print("  ⚠ FM-37's own speed table is COVERED by a pasted amendment slip; "
          "the original is UNREADABLE and only the slip's %d / %s / DIN %d "
          "can be read" % SUPER_XX_FM37_SPEED_SLIP)
    if SUPER_XX_FM37_ORIGINAL_SPEED is not None:
        fail.append("a value has been supplied for the obscured original "
                    "speed table, which nobody can read")

    print("\n=== the agitation law, stated and measured ===")
    print("  printed rule: continuous agitation reduces the tabulated time by "
          "about %.0f %%" % (AGITATION_RULE_CONTINUOUS_REDUCTION * 100))
    for sheet, pg, dev, g, cont, inter in AGITATION_MEASURED_PAIRS:
        print("  %-15s p.%d %-9s gamma %-5s %.1f cont / %.1f inter = +%.0f %%"
              % (sheet, pg, dev, g if g else "-", cont, inter,
                 (inter / cont - 1.0) * 100))
    # ⚠ THE PRINTED RULE MUST BE THE CONSERVATIVE END OF THE MEASURED PAIRS.
    # If a pair came out BELOW 20 % the blanket rule would overshoot for that
    # developer, and the section would be contradicting itself.
    worst = min((inter / cont - 1.0) for _s, _p, _d, _g, cont, inter
                in AGITATION_MEASURED_PAIRS)
    print("  ⚠ the smallest measured pair is +%.0f %%, so the printed %.0f %% "
          "rule is the conservative end of its own evidence"
          % (worst * 100, AGITATION_RULE_CONTINUOUS_REDUCTION * 100))
    if worst < AGITATION_RULE_CONTINUOUS_REDUCTION:
        fail.append("a measured agitation pair now falls below the blanket "
                    "20 %% rule, which would make the section inconsistent")

    print("\n=== the cross-document corroboration ===")
    print("  FM rubric: condenser enlarger -> gamma %.2f, diffuse enlarger -> "
          "gamma %.2f, difference %.2f, printed on %d sheets"
          % (FM_CONDENSER_GAMMA, FM_DIFFUSER_GAMMA, FM_ENLARGER_DELTA_GAMMA,
             len(FM_ENLARGER_RUBRIC_SHEETS)))
    if abs((FM_DIFFUSER_GAMMA - FM_CONDENSER_GAMMA)
           - FM_ENLARGER_DELTA_GAMMA) > 1e-9:
        fail.append("the FM enlarger delta does not equal the difference of "
                    "its own two gammas")
    if G1 is not None:
        g1d = getattr(G1, "NEGATIVE_GAMMA_PER_PAPER_GRADE", None)
        g1g = getattr(G1, "DIFFUSE_VS_CONDENSER_GRADES", None)
        print("  Papers G-1 (US, a different book on a different subject): "
              "%s of negative gamma per paper grade, and %s grade between a "
              "diffuse and a condenser enlarger" % (g1d, g1g))
        ok = (g1d is not None and abs(g1d - FM_ENLARGER_DELTA_GAMMA) < 1e-9
              and g1g == 1)
        print("  ⚠⚠ %s: the British film sheets and the American paper book "
              "agree that the condenser-to-diffuser change is worth exactly "
              "ONE PAPER GRADE, and that one grade is %.2f of negative gamma. "
              "Neither document cites the other."
              % ("THEY AGREE" if ok else "THEY NO LONGER AGREE",
                 FM_ENLARGER_DELTA_GAMMA))
        if not ok:
            fail.append("the FM/G-1 agreement on 0.15 gamma per grade has "
                        "broken: G-1 says %s per grade and %s grades"
                        % (g1d, g1g))
    else:
        fail.append("kodak_papers_g1 could not be imported, so the "
                    "cross-document check did not run")

    print("\n=== why nothing here is wired onto a profile ===")
    for fm, (a, b) in sorted(GENERATION_SPLITS.items()):
        print("  %-16s %-28s ASA %-4d %-9s gammas %s"
              % (fm, a[0], a[1], a[2], a[3]))
        print("  %-16s %-28s ASA %-4d %-9s gammas %s"
              % ("", b[0], b[1], b[2], b[3]))
    print("  ⚠ one FM number, one trade name, two emulsions -- different "
          "speed, different principal developer, different contrast regime")
    print("  ⚠ and FM numbers were REASSIGNED between printings: %s"
          % ", ".join(FM_NUMBERS_REASSIGNED))
    # The database's own sources must still be the modern American ones.
    for nm, want in (("KODAK_PLUS_X_125", "F-4018"),
                     ("KODAK_TRI_X_400TX", "F-4017")):
        try:
            srcs = " ".join(FP.get_profile(nm).provenance.sources)
        except Exception:
            continue
        if PUBLISHER in srcs:
            fail.append("%s has acquired a Kodak Limited source; the FM "
                        "sheets describe a different emulsion" % nm)

    print("\n=== recorded absences, for the five films task #502 named ===")
    for line in (
            "no reciprocity of any kind -- «reciprocit*» occurs on four pages "
            "in 1140-1495 and all four are colour sheets",
            "no granularity number: «granular*» occurs once in the section, on "
            "a colour cine sheet; «RMS» and «grain index» zero times",
            "no resolving power for Plus-X Pan (any issue), Verichrome, Tri-X "
            "Pan or Royal-X -- the ONLY figure attaching to any of the five "
            "is 65 l/mm on FM-52 p.1444, and it carries no test-object "
            "contrast",
            "no characteristic curve for Tri-X Pan roll at all: FM-45 and "
            "FM-46 have no sensitometric-curve section, only time-temperature",
            "no gamma figure anywhere on FM-44, the Royal-X sheet film",
            "no storage, no latent-image keeping, no base thickness, no "
            "format list, and no Contrast Index on any FM sheet -- CI is only "
            "a pointer to data sheet SE-1A in the contents"):
        print("  * " + line)

    if fail and do_assert:
        print("\n".join(["", "FAILURES:"] + ["  " + f for f in fail]))
        return 1
    print("\n[OK] kodak_databook_fm.py -- the FM section of the «Kodak Data "
          "Book of Applied Photography», pp.1140-1495, read whole. ⚠⚠ IT IS "
          "KODAK LIMITED, LONDON, AND THAT REWRITES THE TASK: task #502 "
          "expected development tables for 125PX, 400TX, 4166 and EASTMAN "
          "5231, and none of those designations occurs in 356 pages. The "
          "sheets are British coatings of the middle 1960s, and the section "
          "proves the gap itself -- FM-36 carries 'Plus-X' at ASA 160 in "
          "D-61a and 'Plus-X PAN' at ASA 125 in DK-50 under ONE FM number. So "
          "no development table here is written onto a profile sourced from a "
          "2005-2007 American sheet. ⚠ WHAT IT DID CLOSE: the D-76/gamma-0.7 "
          "sentence is on exactly three sheets and qualifies a SPEED rather "
          "than prescribing a process, with Verichrome the sheet that could "
          "have carried it and does not; Royal-X contradicts it by naming "
          "DK-50 as the only acceptable developer; Weston is absent from every "
          "monochrome sheet and appears on colour sheets only as a "
          "meter-model correction; and Super-XX's 60 l/mm was found and "
          "DELIBERATELY NOT ADOPTED, because the database's 55 is Eastman "
          "Kodak's in SD-21 and this is Kodak Ltd's in D-61a with no "
          "test-object contrast. ⚠⚠ AND THE BEST FIND IS A COINCIDENCE THAT "
          "IS NOT ONE: the FM enlarger rubric puts 0.15 of negative gamma "
          "between a condenser and a diffuser chain, and kodak_papers_g1, "
          "built from a different book in a different country on a different "
          "subject, independently holds 0.15 per paper grade and one grade "
          "between those two enlargers")
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
