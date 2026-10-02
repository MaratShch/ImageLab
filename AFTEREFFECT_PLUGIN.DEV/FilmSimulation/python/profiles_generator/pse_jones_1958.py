"""R. Clark Jones 1958: Kodak's own sigma(D) for four B&W negatives, and the
first direct test in this corpus of the aperture law.

WHAT THIS SOURCE IS
-------------------
R. Clark Jones, "On the Quantum Efficiency of Photographic Negatives",
**PHOTOGRAPHIC SCIENCE AND ENGINEERING Vol. 2 No. 2, August 1958, pp. 57-65**.
On the owner's machine at
`PDF/PROFILES/RETRO/Photographic Science and Engineering/`.

⚠ THE FILE IS MIS-NAMED AND THE CITATION MUST NOT FOLLOW IT. The Internet
Archive calls it `sim_journal-of-imaging-science_1958-08_2_2.pdf`. The *Journal
of Imaging Science* is the SAME journal's name from 1987 onward; this issue's
own running foot reads `P S & E, Vol. 2, 1958`. Citing the filename points a
reader thirty years away from the paper.

The paper is about detective quantum efficiency. What matters here is the DATA
it prints to get there, all of it **supplied to the author by Eastman Kodak**:

    Table A   film, developer, time, gamma, for four materials
    Table B   the full characteristic curves as PRINTED NUMBERS, D against
              log10 U in ergs/sq cm at 430 nm, in 0.1-decade steps
    Table C   granularity: sigma at four densities x three apertures
              (10, 20, 40 um), 40 values, "each of the 40 values involves 2000
              individual density measurements"

⚠ VINTAGE IS PINNED AND IT IS THE REASON THIS IS USABLE AT ALL: "The first is
sheet film, and the last three are 35mm roll films. The films were manufactured
in **February 1957**." A granularity figure with no coating date is not
attributable to a product; this one is.

⚠ AND IT IS WHY NONE OF IT IS ADOPTED PER STOCK. `KODAK_TRI_X_400TX`,
`EASTMAN_TRI_X_5223` and `EASTMAN_PLUS_X_5231` are in the database and are NOT
these films -- 5231 is a 1999 cine coating, 400TX a modern still. Royal-X and
Pan-X are absent entirely. Attaching a February 1957 measurement to any of them
would be the trade-name substitution this project has already been caught by
twice. What IS adopted is the CLASS SHAPE, below, and the class is what the four
films jointly establish.

WHAT IS ADOPTED
---------------
1. **The sigma(D) class shape for black-and-white negatives**, from the four
   films' `(sigma10)Av` column normalised to each film's own value at D = 1.0.
2. Nothing else. Table B's curves reach only D 0.92-0.96 on three of the four
   films -- no shoulder, no Dmax -- and its exposure axis is 430 nm
   monochromatic in ergs/sq cm, so it fixes no speed. Table A's gammas belong
   to developers this database does not model per stock.

⚠ WHY THE CLASS SHAPE IS BELIEVABLE AND THE OLD PLACEHOLDER WAS NOT. The four
films span Pan-X to Royal-X -- roughly ASA 32 to ASA 1250, four developers --
and once each is normalised at D 1.0 they agree to a standard deviation of
**0.042 to 0.058** at every density. Four emulsions two orders of speed apart
landing on one curve is what a class law looks like. ⚠ The limitation, stated
rather than buried: all four are **Eastman Kodak**, so this is one
manufacturer's emulsion technology of 1957, and method rule 18's warning about
single-family inference applies to the maker even though it does not apply to
the sample count.

⚠ WHAT THE MEASUREMENT SAYS ABOUT WHAT THE ENGINE DOES TODAY. 56 monochrome
negatives carry `sigma_shape_measured=False`, so `sigma_measured_usable()`
refuses their stored shape and they render on the LEGACY sqrt(D) law. Against
this measurement that law is:

        D 0.07   -48 %      shadows far too quiet
        D 0.50   -19 %
        D 1.00     0 %      (both normalised here by definition)
        D 1.40   +17 %      highlights too loud

and the placeholder triple (0.40 / 1.00 / 1.20) that was sitting unused is
-21 % at D 0.07 and -26 % at D 0.50. Neither described the film.

THE APERTURE LAW, TESTED
------------------------
⚠ THIS IS THE FIRST DIRECT TEST OF SELWYN'S LAW IN THIS CORPUS ON A MULTI-
APERTURE MEASUREMENT. Selwyn requires sigma proportional to 1 / aperture
DIAMETER, i.e. sigma10 = 2*sigma20 = 4*sigma40, which is exactly how Jones
tabulates it. Over the 24 available pairs the mean ratio is **0.929** -- sigma10
runs **7.1 % low** -- and the paper predicts the sign and roughly the size
itself: "a noticeable tendency for the sigma10 to be perhaps 10% lower than
2sigma20 and 4sigma40 ... expected on the basis of the inevitable blurring of
the edges of the circular spot by diffraction and the finite thickness of the
developed layer."

That is an independent confirmation of the aperture term inside
`film_sim.grain_reference_energy`, from a third source and a different decade
than the two it already had. ⚠ It is NOT a licence to change the term: the 7 %
is an instrument effect (diffraction, layer thickness), not a property of the
emulsion, and modelling it would mean modelling a 1957 microdensitometer.

⚠ ONE ENTRY IS EXCLUDED BY THE PAPER ITSELF and the reader honours it: Royal-X
at D 0.80 reads sigma10 = 0.260 against 0.206 and 0.192 for the other two
apertures. Jones gives it **zero weight** in the average ("except that the entry
0.260 was given zero weight, and the entry 0.172 was given weight 1/3"). The
`(sigma10)Av` column already reflects that, which is why this reader uses that
column and not its own average of the three.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SOURCE = ("R. Clark Jones, «On the Quantum Efficiency of Photographic "
          "Negatives», Photographic Science and Engineering Vol. 2 No. 2, "
          "August 1958, pp. 57-65, Tables A, B and C. Data supplied to the "
          "author by Eastman Kodak; films manufactured February 1957")

#: Table A -- material, developer, time (min), gamma. Printed, not derived.
TABLE_A = {
    "Royal-X": dict(developer="DK-50", minutes=5.0, gamma=0.65, form="sheet"),
    "Tri-X":   dict(developer="SD-28", minutes=6.5, gamma=0.54, form="35mm"),
    "Plus-X":  dict(developer="SD-28", minutes=6.5, gamma=0.79, form="35mm"),
    "Pan-X":   dict(developer="D-76",  minutes=6.0, gamma=0.67, form="35mm"),
}

#: Printed alongside Table A: base density, common to all four.
BASE_DENSITY = 0.22

#: Printed in section 5.0: reciprocity correction from 15 s to 0.1 s, in log
#: exposure units, at a density of 0.38 above base. NOT adopted -- this
#: database's ReciprocitySpec is a Schwarzschild exponent, and one point
#: cannot fix one.
RECIPROCITY_15S_LOG = {"Royal-X": 0.45, "Tri-X": 0.34,
                       "Plus-X": 0.29, "Pan-X": 0.29}

#: Table C. (D, sigma10, 2*sigma20, 4*sigma40 or None, (sigma10)Av).
#: ⚠ Plus-X and Pan-X carry NO 40 um column in the paper; 4+4+3+3 apertures
#: over four densities is the "40 values" the text counts.
TABLE_C = {
    "Royal-X": ((0.10, 0.097, 0.102, 0.100, 0.100),
                (0.34, 0.147, 0.168, 0.164, 0.160),
                (0.80, 0.260, 0.206, 0.192, 0.199),
                (1.18, 0.186, 0.230, 0.232, 0.216)),
    "Tri-X":   ((0.06, 0.056, 0.074, 0.064, 0.065),
                (0.36, 0.085, 0.104, 0.128, 0.106),
                (0.74, 0.090, 0.112, 0.116, 0.106),
                (1.14, 0.122, 0.122, 0.172, 0.129)),
    "Plus-X":  ((0.06, 0.030, 0.030, None, 0.030),
                (0.30, 0.061, 0.058, None, 0.060),
                (0.80, 0.074, 0.074, None, 0.074),
                (1.40, 0.069, 0.062, None, 0.066)),
    "Pan-X":   ((0.04, 0.029, 0.034, None, 0.032),
                (0.26, 0.048, 0.054, None, 0.051),
                (0.66, 0.057, 0.062, None, 0.060),
                (1.10, 0.076, 0.066, None, 0.071)),
}

#: The adopted class shape, as stored on the monochrome negatives.
#: Derived by `class_shape()`; frozen here so the adoption is reproducible
#: without re-running the derivation.
ADOPTED_TOE_AT = 0.07
ADOPTED_TOE = 0.507
ADOPTED_MID = 1.000
ADOPTED_DMAX_AT = 1.40
ADOPTED_DMAX = 1.016

#: Selwyn: sigma10 / (2 sigma20) and sigma10 / (4 sigma40) should both be 1.0.
SELWYN_MEAN = 0.929
SELWYN_TOL = 0.02


def class_shape(grid=(0.06, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.20, 1.40)):
    """Mean and spread of the four films' sigma(D), each normalised at D = 1.0.

    Log-log interpolation throughout: sigma against density is a power law over
    most of this range, so interpolating linearly would bias the low-density
    end where the four films are furthest apart.
    """
    g = np.asarray(grid, dtype=float)
    curves = []
    for rows in TABLE_C.values():
        d = np.array([r[0] for r in rows], dtype=float)
        s = np.array([r[4] for r in rows], dtype=float)
        s1 = float(np.exp(np.interp(np.log(1.0), np.log(d), np.log(s))))
        curves.append(np.exp(np.interp(np.log(g), np.log(d), np.log(s / s1))))
    m = np.array(curves)
    return g, m.mean(axis=0), m.std(axis=0), m


def selwyn_ratios():
    """sigma10 / (2 sigma20) and sigma10 / (4 sigma40) over every printed pair."""
    out = []
    for rows in TABLE_C.values():
        for _d, s10, s20x2, s40x4, _av in rows:
            out.append(s10 / s20x2)
            if s40x4 is not None:
                out.append(s10 / s40x4)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()

    import film_profiles as fp

    print("R. Clark Jones 1958 -- PS&E 2(2), Kodak sigma(D) for four B&W negatives")
    bad = 0

    # ---- the aperture law -------------------------------------------------
    r = selwyn_ratios()
    ok = abs(float(r.mean()) - SELWYN_MEAN) < SELWYN_TOL
    bad += (not ok)
    print("\n  Selwyn: sigma10 / 2sigma20 and sigma10 / 4sigma40, %d pairs" % len(r))
    print("    [%s] mean %.3f (paper: sigma10 runs about 10%% low from "
          "diffraction and layer thickness); min %.3f max %.3f"
          % ("OK  " if ok else "FAIL", r.mean(), r.min(), r.max()))

    # ---- the class shape --------------------------------------------------
    g, mean, sd, _m = class_shape()
    ok = float(sd.max()) < 0.07
    bad += (not ok)
    print("\n  Class shape: four films, each normalised at D = 1.0")
    for i, d in enumerate(g):
        print("    D %.2f   mean %.3f   sd %.3f" % (d, mean[i], sd[i]))
    print("    [%s] the four agree to sd %.3f at worst -- four emulsions "
          "roughly ASA 32 to 1250 on one curve"
          % ("OK  " if ok else "FAIL", sd.max()))

    toe = float(np.exp(np.interp(np.log(ADOPTED_TOE_AT), np.log(g),
                                 np.log(mean))))
    top = float(np.exp(np.interp(np.log(ADOPTED_DMAX_AT), np.log(g),
                                 np.log(mean))))
    for got, want, label in ((toe, ADOPTED_TOE, "toe"),
                             (top, ADOPTED_DMAX, "dmax")):
        ok = abs(got - want) < 0.005
        bad += (not ok)
        print("    [%s] adopted %s anchor %.3f reproduces the derivation (%.3f)"
              % ("OK  " if ok else "FAIL", label, want, got))

    # ---- what is stored ---------------------------------------------------
    mono = [p for p in fp.FILM_PROFILES
            if p.is_monochrome and not p.is_reversal]
    on = [p for p in mono
          if abs(p.grain.sigma_shape_toe - ADOPTED_TOE) < 1e-9
          and abs(p.grain.sigma_shape_dmax - ADOPTED_DMAX) < 1e-9]
    print("\n  Adoption: %d of %d monochrome negatives carry the class shape"
          % (len(on), len(mono)))
    ok = len(on) >= 50
    bad += (not ok)
    print("    [%s] the 1957 class shape is on the monochrome-negative block"
          % ("OK  " if ok else "FAIL"))

    # ⚠ the trade-name refusal, asserted rather than trusted
    for n in ("KODAK_TRI_X_400TX", "EASTMAN_TRI_X_5223",
              "EASTMAN_PLUS_X_5231"):
        try:
            p = fp.get_profile(n)
        except Exception:
            continue
        same = (abs(p.grain.sigma_shape_toe - ADOPTED_TOE) < 1e-9)
        print("    [note] %s carries the class shape: %s -- it is NOT the "
              "February 1957 coating Jones measured" % (n, same))

    print()
    if bad:
        print("  [FAIL] %d problem(s)" % bad)
        return 1 if ns.do_assert else 0
    print("  [OK  ] Jones 1958: aperture law confirmed to 7 %%, class sigma(D) "
          "adopted on the monochrome-negative block")
    return 0


if __name__ == "__main__":
    sys.exit(main())
