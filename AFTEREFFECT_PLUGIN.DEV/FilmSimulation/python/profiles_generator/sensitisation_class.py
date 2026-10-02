"""Fail the build when a monochrome stock's WEIGHTS contradict its CLASS.

WHY THIS EXISTS
---------------
A monochrome emulsion belongs to one of four sensitisation classes, and the
class is a statement about physics that its numbers have to honour:

    blue    non-colour-sensitised. Silver halide's own absorption and nothing
            else: dead past about 520 nm.
    ortho   orthochromatic. Sensitised to green, RED-BLIND: dead past about
            600 nm.
    pan     panchromatic. Sensitised across the visible, INCLUDING the red.
    ir      sensitised past the visible, where three visible lobes cannot
            describe it at all -- `spectral_monochrome_weights` refuses these
            and the authored triple stands.

⚠⚠ THIS FILE EXISTS BECAUSE A PANCHROMATIC FILM SHIPPED WITH LESS GREEN
RESPONSE THAN AN ORTHOCHROMATIC ONE AND NOTHING NOTICED. On 2026-09-27 the
owner rendered a ColorChecker through FERRANIA_P30 on the AVX2 engine and the
blue patch came out white. Measured: P30's derived weights were
(0.076, 0.166, 0.758) against FERRANIA_ORTO_50's (0.015, 0.544, 0.441) -- a
green weight a THIRD of a red-blind film's, on a film its maker sells as
panchromatic, with both numbers derived by the same function from curves in
the same database. An orthochromatic emulsion is green-sensitised BY
DEFINITION. No panchromatic film can be a third as green-sensitive as one.

That is a contradiction between two stored records, and a contradiction
between stored records is exactly what a build gate can catch without any new
document. Every check below is of that kind: it compares the corpus against
itself, or against the definition of a word the corpus already uses.

⚠ THE CLASS TABLE IS EXPLICIT AND IS NOT A STRING MATCH. The first draft of
this audit classified stocks by searching their descriptions, and it got
KODAK_ROYAL_ORTHO_1956 wrong (the word «pan» appears in the prose), and
KODAK_TECHNICAL_PAN wrong (its description mentions infrared). The same trap
is already recorded against the mono-weight work: matching «ortho» catches
FUJI NEOPAN SS and KODAK VERICHROME PAN, both panchromatic films whose
descriptions mention orthochromatic stock in order to contrast with it. Every
row below was read off the stock's own description and source.

⚠ AND ONE ROW CORRECTS A PUBLISHED CLAIM OF THIS PROJECT'S OWN. The 2026-09-26
mono-primary-width note named «the four stocks their makers call
ORTHOCHROMATIC» and listed KODAK_COMMERCIAL_1956 among them. It is not
orthochromatic. Its own description says so in terms -- "THE ONLY
NON-COLOUR-SENSITIZED FILM IN THIS DATABASE FROM THIS BOOK, and its wedge
spectrogram proves it: the measured response dies at 510 nm" -- so its near-
zero green weight is correct and the sentence that called it ortho was not.

Run:  python sensitisation_class.py [--assert]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import film_profiles as fp  # noqa: E402
import film_sim as fs  # noqa: E402

#: Stock -> sensitisation class, hand-checked one row at a time against each
#: stock's own description and its spectral source string. A monochrome stock
#: carrying a pan curve and absent from this table FAILS: an unclassified
#: stock is an unchecked stock, and the point of the file is that nothing
#: slips through unexamined.
CLASS = {
    # -- non-colour-sensitised ---------------------------------------------
    # Dies at 510 nm on its own wedge spectrogram, and its daylight and
    # tungsten indexes are four stops apart, which only a blue-sensitive
    # emulsion can be.
    "KODAK_COMMERCIAL_1956": "blue",

    # -- orthochromatic -----------------------------------------------------
    "FERRANIA_ORTO_50": "ortho",
    "KODAK_ROYAL_ORTHO_1956": "ortho",
    "KODAK_SUPER_SPEED_ORTHO_1956": "ortho",

    # -- sensitised past the visible; the derivation refuses these ----------
    "KODAK_HIE": "ir",                  # 2026-10-01e: «Современные» Рис. 3.365, to 920 nm
    "KONICA_INFRARED_750": "ir",
    "ROLLEI_INFRARED_400": "ir",

    # -- panchromatic -------------------------------------------------------
    "AGFA_APX_25": "pan", "AGFA_APX_100": "pan", "AGFA_APX_400": "pan",
    "AGFA_SCALA_200X": "pan",
    # 2026-09-29b: «panchromatic up to 750 nm» on its PE0 sheet (extended red,
    # not an infrared emulsion -- nothing past the visible is drawn).
    "AGFA_AVIPHOT_PAN_20": "pan",
    "EASTMAN_DOUBLE_X_5222": "pan", "EASTMAN_PLUS_X_5231": "pan",
    "FERRANIA_P30_MK2": "pan", "FERRANIA_P33_160": "pan",
    "FOMAPAN_400_ACTION": "pan",
    "FUJI_NEOPAN_1600": "pan", "FUJI_NEOPAN_ACROS_100": "pan",
    "FUJI_NEOPAN_400": "pan",   # 2026-09-29b, AF3-207U: «Panchromatic»
    "FUJI_NEOPAN_SS": "pan",
    "ILFORD_DELTA_3200": "pan", "ILFORD_HP5_PLUS_400": "pan",
    "KODAK_BW400CN": "pan", "KODAK_T400CN": "pan",
    "KODAK_PANATOMIC_X": "pan", "KODAK_PANATOMIC_X_SHEET_1952": "pan",
    "KODAK_PLUS_X_125": "pan",
    "KODAK_PORTRAIT_PANCHROMATIC_1956": "pan",
    "KODAK_ROYAL_PAN_4141": "pan", "KODAK_ROYAL_X_PAN_4166": "pan",
    "KODAK_SUPER_PANCHRO_PRESS_B_1956": "pan",
    "KODAK_SUPER_XX_PAN_4142": "pan",
    "KODAK_TECHNICAL_PAN": "pan",
    "KODAK_TMAX_100": "pan", "KODAK_TMAX_400": "pan",
    "KODAK_TMAX_P3200": "pan",
    "KODAK_TRI_X_400TX": "pan", "KODAK_TRI_X_REVERSAL_200": "pan",
    "KODAK_TRI_X_SHEET_1952": "pan", "KODAK_VERICHROME_PAN": "pan",
    "POLAROID_52": "pan", "POLAROID_55_PN_NEG": "pan",
    "POLAROID_664": "pan", "POLAROID_667": "pan",
    "ROLLEI_R3": "pan", "ROLLEI_RETRO_400": "pan",

    # -- panchromatic, and the maker says the red is weak --------------------
    # ⚠ A CLASS OF ONE, AND IT IS A CLAIM RATHER THAN A CONVENIENCE. Film
    # Ferrania describe the original P30 as having «bassa sensibilita al
    # rosso», and the only surviving measurement of how low is one
    # photographer's filter factor: +5 stops where the filter calls for +3,
    # i.e. 0.60 decade. The gates below state what that has to mean
    # numerically -- red BELOW every measured panchromatic stock, green
    # firmly inside the panchromatic range -- so the claim is falsifiable
    # rather than decorative.
    "FERRANIA_P30": "pan_weak_red",
}

#: The basis must have no hole: adjacent lobes cross at least at half maximum.
#: This is the whole justification for the 34 nm width and it is asserted here
#: rather than fitted anywhere.
HALF_MAX_K = 2.0 * np.sqrt(2.0 * np.log(2.0))     # 2.3548
#: and it must see most of what it is integrating.
CAPTURE_MIN = 0.60

#: Class bounds, each one read off the corpus's own MEASURED stocks and stated
#: as the loosest bound that still separates the classes.
BLUE_GREEN_MAX = 0.12   # a non-colour-sensitised film has next to no green
ORTHO_RED_MAX = 0.16    # an orthochromatic film is red-blind
ORTHO_GREEN_MIN = 0.25  # ... and is green-sensitised, which is the other half
PAN_RED_MIN = 0.15      # a panchromatic film responds in the red
#: ⚠ AND IN THE GREEN, WHICH IS THE GATE FERRANIA_P30 FAILED. Measured over
#: the 38 panchromatic stocks whose curve is traced, the weakest green weight
#: is POLAROID_52's 0.223 and the next is KODAK_PLUS_X_125's 0.246. The floor
#: is set at 0.20 -- under the whole measured population with a little air,
#: and well over the 0.166 the withdrawn P30 curve produced. It is a floor on
#: the class, NOT a comparison against the orthochromatic stocks: see the
#: note in `classes()` for why that comparison is not physics.
PAN_GREEN_MIN = 0.20
PAN_D600_MAX = 1.20     # its curve is alive at 600 nm, within this of peak


def _curve(profile):
    sp = profile.spectral
    if not sp.log_s_pan:
        return None, None
    v = np.asarray(sp.log_s_pan, float)
    lam = sp.lambda_start_nm + sp.lambda_step_nm * np.arange(v.size)
    return lam, v


def basis_geometry(problems: list, report: list) -> None:
    """⚠ THE WIDTH, ASSERTED FROM THE CENTRES AND NOTHING ELSE."""
    c = sorted(fs._PRIMARY_CENTRES_NM)
    spacing = max(b - a for a, b in zip(c, c[1:]))
    need = spacing / HALF_MAX_K
    got = float(fs._PRIMARY_WIDTH_NM)
    report.append("basis: centres %s, widest spacing %.0f nm, half-max width "
                  "%.2f nm, shipping %.1f nm"
                  % (tuple(int(x) for x in c), spacing, need, got))
    if got + 1e-9 < need:
        problems.append(
            "the primary lobes are narrower than half-max crossing: sigma "
            "%.1f nm against %.2f nm needed for centres %.0f nm apart, so the "
            "basis has a HOLE and discards sensitivity inside the visible"
            % (got, need, spacing))
    # and the coverage that the hole would cost
    grid = fs.spectral_grid()
    tot = fs._srgb_primary_spd().sum(axis=0)
    caps = []
    for p in fp.FILM_PROFILES:
        if not p.is_monochrome:
            continue
        s = fs.layer_sensitivities(p)
        if s is None or s.shape[0] != 1:
            continue
        den = float(np.trapezoid(s[0], grid))
        if den > 0:
            caps.append(float(np.trapezoid(s[0] * tot, grid)) / den / tot.max())
    cap = float(np.mean(caps)) if caps else 0.0
    report.append("basis sees %.1f %% of a typical emulsion's own integrated "
                  "sensitivity (n=%d)" % (100.0 * cap, len(caps)))
    if cap < CAPTURE_MIN:
        problems.append(
            "the basis sees only %.1f %% of a typical emulsion; below %.0f %% "
            "the weights are three samples of a curve rather than an integral "
            "of it" % (100.0 * cap, 100.0 * CAPTURE_MIN))


def classes(problems: list, report: list) -> None:
    seen = {}
    for p in fp.FILM_PROFILES:
        if not p.is_monochrome:
            continue
        lam, v = _curve(p)
        cls = CLASS.get(p.name)
        if lam is None and cls is None:
            continue            # no curve and no claim: nothing to contradict
        if cls is None:
            problems.append("%s carries a pan curve and is not in the class "
                            "table -- an unclassified stock is an unchecked "
                            "one" % p.name)
            continue
        w = fs.spectral_monochrome_weights(p)
        derived = w is not None
        if w is None:
            w = tuple(p.spectral_weights) if p.spectral_weights else None
        if w is None:
            problems.append("%s has neither a derivable nor an authored "
                            "weight triple" % p.name)
            continue
        seen.setdefault(cls, []).append((p.name, w, derived))
        R, G, B = w
        if cls == "blue":
            if G > BLUE_GREEN_MAX:
                problems.append("%s is non-colour-sensitised and carries "
                                "green %.3f (> %.2f)" % (p.name, G, BLUE_GREEN_MAX))
        elif cls == "ortho":
            if R > ORTHO_RED_MAX:
                problems.append("%s is orthochromatic -- red-blind -- and "
                                "carries red %.3f (> %.2f)"
                                % (p.name, R, ORTHO_RED_MAX))
            if G < ORTHO_GREEN_MIN:
                problems.append("%s is orthochromatic, which means GREEN-"
                                "SENSITISED, and carries green %.3f (< %.2f)"
                                % (p.name, G, ORTHO_GREEN_MIN))
        elif cls == "pan":
            if R < PAN_RED_MIN:
                problems.append("%s is panchromatic and carries red %.3f "
                                "(< %.2f)" % (p.name, R, PAN_RED_MIN))
            if G < PAN_GREEN_MIN:
                problems.append("%s is panchromatic and carries green %.3f "
                                "(< %.2f)" % (p.name, G, PAN_GREEN_MIN))
            if lam is not None and lam[0] <= 600.0 <= lam[-1]:
                d600 = float(v.max() - np.interp(600.0, lam, v))
                if d600 > PAN_D600_MAX:
                    problems.append("%s is panchromatic and its curve is "
                                    "%.2f decade below peak at 600 nm "
                                    "(> %.2f) -- that is not a red-sensitised "
                                    "emulsion" % (p.name, d600, PAN_D600_MAX))
        elif cls == "pan_weak_red":
            if G < PAN_GREEN_MIN:
                problems.append("%s is panchromatic and carries green %.3f "
                                "(< %.2f)" % (p.name, G, PAN_GREEN_MIN))
    # ⚠ THE CROSS-CLASS ORDERING IS THE CHECK THAT WOULD HAVE CAUGHT P30, and
    # it compares the corpus against itself rather than against a threshold.
    pan = [x for x in seen.get("pan", [])]
    ortho = [x for x in seen.get("ortho", [])]
    weak = [x for x in seen.get("pan_weak_red", [])]
    if pan and ortho:
        pan_g = min(w[1] for _, w, _ in pan)
        ort_g = max(w[1] for _, w, _ in ortho)
        report.append("green weight: weakest panchromatic %.3f, strongest "
                      "orthochromatic %.3f" % (pan_g, ort_g))
        # ⚠⚠ THERE IS DELIBERATELY NO CROSS-CLASS GREEN COMPARISON HERE, AND
        # THE FIRST VERSION OF THIS FILE HAD ONE. It asserted that a
        # panchromatic stock's green weight must not fall below every
        # orthochromatic stock's, which is how FERRANIA_P30's defect was
        # first described -- green 0.166 against ORTO 50's 0.448. Written as
        # a gate it failed THIRTEEN measured panchromatic stocks, KODAK TRI-X
        # 400TX among them, and thirteen measured curves are not all wrong at
        # once: the rule was. These weights are SHARES of one, and a
        # panchromatic film divides its response three ways where an
        # orthochromatic film divides it two, so the ortho film's green share
        # is larger for a reason that has nothing to do with how green-
        # sensitive either emulsion is. What P30's 0.166 actually violated is
        # the WITHIN-CLASS floor below, and that is where it is caught.
        # The cross-class comparison that IS physics is the red one, because
        # red-blindness is what the word orthochromatic means.
        pan_r = min(w[0] for _, w, _ in pan)
        ort_r = max(w[0] for _, w, _ in ortho)
        report.append("red weight: weakest panchromatic %.3f, strongest "
                      "orthochromatic %.3f" % (pan_r, ort_r))
        if pan_r <= ort_r:
            problems.append("the weakest panchromatic red weight (%.3f) no "
                            "longer exceeds the strongest orthochromatic one "
                            "(%.3f); the two classes have stopped separating"
                            % (pan_r, ort_r))
        for n, w, _ in weak:
            if w[0] >= pan_r:
                problems.append("%s is classed pan_weak_red on the maker's "
                                "own statement and its red weight %.3f is not "
                                "below the weakest ordinary panchromatic "
                                "stock's %.3f -- the class claims a deficit "
                                "the number does not show" % (n, w[0], pan_r))
    for c in sorted(seen):
        report.append("%s: %d stock(s), %d derived from a curve"
                      % (c, len(seen[c]), sum(1 for _, _, d in seen[c] if d)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert", dest="hard", action="store_true")
    args = ap.parse_args(argv)
    problems: list[str] = []
    report: list[str] = []
    basis_geometry(problems, report)
    classes(problems, report)
    if not args.hard:
        for line in report:
            print("  " + line)
    if problems:
        for p in problems:
            print("[FAIL] sensitisation_class.py -- " + p)
        return 1
    print("[OK] sensitisation_class.py -- " + "; ".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
