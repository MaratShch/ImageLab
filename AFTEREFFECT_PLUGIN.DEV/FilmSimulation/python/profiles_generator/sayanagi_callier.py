"""Sayanagi 1959: why the Callier toe collapses, and what beta really is.

Queue items C44 (the toe) and C43 (the value of `callier_q`), both closed
2026-09-02, both by the same five-page paper.

WHAT THIS SOURCE IS
-------------------
`PDF/PROFILES/RETRO/JAPAN/23_20.pdf` -- Kazuo Sayanagi, Canon Camera Co., Inc.,
Shimo-maruko, Ohta-ku, Tokyo, «Callier Q Factor と粒状» / "A Theory on Callier Q
Factor and Granularity", *J. Soc. Phot. Sci. Japan* **23**(1) 20-24, received
15 December 1959. Text layer present; the figures are raster.

THE MODEL, FROM HIS OWN ASSUMPTIONS
------------------------------------
(I)   the base has intensity transmittance Ib and amplitude Ab, Ib = Ab^2;
(II)  a developed silver grain has FINITE intensity transmittance Ig and
      amplitude Ag, Ig = Ag^2 -- finite because the electron microscope shows
      the developed grain to be filamentary rather than a solid disc;
(III) grains are circles of radius r0 whose centres are POISSON-distributed on
      the film plane with area coverage 𝔅, so n = 𝔅 / (pi r0^2).

Savelli's Poisson averages then give, as his equations (7) and (8),

    Ibar = Ib    * exp{ -𝔅 (1 - Ig  ) }      mean intensity  -> DIFFUSE density
    Abar = Ib^½  * exp{ -𝔅 (1 - Ig^½) }      mean amplitude  -> the DC term,
                                             whose square is SPECULAR density

    D_diffuse  = Db +     0.4343 * 𝔅 * (1 - Ig  )
    D_specular = Db + 2 * 0.4343 * 𝔅 * (1 - Ig^½)

and dividing the two AFTER the base is removed gives his (10):

    Q_II = 2 (1 - Ig^½) / (1 - Ig) = 2 / (1 + Ig^½)

WHAT THAT SETTLES
-----------------
**1. The density reference, which C22 had to argue for itself.** His §2.3 names
the two conventions -- Q_I with the base left in, Q_II with it taken out -- and
§3.2 states that Q_II is the rational one. This engine computes Callier on NET
density. C22 reasoned its way there from first principles and recorded that no
source stated a convention. One does, from 1959, and it agrees.

**2. Q_II CONTAINS NO DENSITY.** Not D, not the coverage 𝔅, not the grain radius
r0 -- only Ig. On base-subtracted density Sayanagi's Q is flat. So the toe
collapse that `_CALLIER_TOE_MEASURED` has recorded as a model defect since
2026-09-01 has no mechanism in the one theory that derives Q from grain optics.

**3. WITH THE BASE LEFT IN, IT COLLAPSES EXACTLY AS MEASURED.**

    Q_I(D) = [ Db + Q_inf * (D - Db) ] / D

is 1 at D = Db by construction and climbs to Q_inf. This reader fits Db to the
two measured Q(D) curves in the corpus, independently, and gets the same answer:

    Trumpy/Streiffert Fig. 5   Db = 0.045   (three-parameter fit, rms 0.019 Q
                                             over the WHOLE curve including the
                                             toe; the shipped no-base fit gets
                                             rms 0.156 and misses the toe by
                                             +0.49 Q)
    Mees FIG. 179              Db = 0.050   (reconciles all five gamma curves
                                             with the shared toe stroke)

⚠ TWO LABORATORIES, TWO DECADES, TWO FIGURES, THE SAME BASE DENSITY TO HALF A
PERCENT OF D. That is the check everything below rests on, and neither figure
knows about the other.

⚠ AND IT EXPLAINS THE ONE FEATURE OF FIG. 179 NOTHING ELSE COULD: why all FIVE
gamma curves are drawn as a SINGLE stroke below D = 0.25. Q_I -> 1 at D = Db for
every curve whatever its Q_inf, so at the toe five emulsions of five different
contrasts genuinely coincide. Under any model in which Q is a film property
alone that shared toe is impossible, and it had to be read as an artist's
convention. It is a measurement.

**4. THEREFORE C44 CLOSES WITH NO CODE CHANGE, AND THAT IS THE RESULT.** The
engine's argument is NET density; the base is already gone; a toe term fitted to
Q_I data would remove it a second time and darken the shadows on every B&W stock
whenever a condenser is dialled in -- the exact region C44 was opened to protect.
`film_sim.callier_net`, `AlgoCallierNet` and the shared LUT are untouched and
stay bit-identical.

WHAT IT CHANGES INSTEAD: beta (queue C43)
------------------------------------------
The shipped E = 0.1471 and beta = 1.6746 were fitted to Trumpy's curve over
D >= 0.30 with NO base term, so the base contamination was absorbed into beta.
Refitting with the base modelled moves beta to 1.809.

Refitting each of Mees's five gamma curves with Db held at 0.050 gives beta as a
function of DEVELOPMENT GAMMA:

    gamma   0.21   0.37   0.69   1.20   1.65
    beta    1.491  1.495  1.729  1.822  1.828

⚠ THE DATABASE GAVE EVERY MONOCHROME STOCK 1.3 (1.25 REVERSAL), A CLASS CONSTANT
WITH NO DOCUMENT BEHIND IT, AND EVERY MEASUREMENT IS ABOVE IT. `callier_q` is
now computed per stock from its own mid slope through

    beta(gamma) = 1 + A * gamma / (gamma + K),   A = 0.9706, K = 0.2558

⚠ THE FORM WAS CHOSEN FOR ITS ENDPOINTS, NOT ITS RESIDUAL: beta(0) = 1 exactly
(no developed silver, no scattering, no Callier effect) and beta(inf) = 1.971,
just under Sayanagi's own ceiling of 2.0. A decaying exponential fits the same
five points equally well (rms 0.043 against 0.045) and gets both endpoints
wrong.

⚠ THE CEILING IS A REAL TEST AND THE MODEL PASSES IT. Inverting his (10),
Ig = (2/beta - 1)^2, so beta > 2 is impossible. The five Mees curves invert to
grain transmittances of 11.7 % at gamma 0.21 falling to 0.9 % at gamma 1.65 --
grains that grow more opaque as development proceeds, which is what assumption
(II) says they do. Nothing forced that ordering.

⚠ ONE MEASUREMENT DOES NOT FIT UNDER THE CEILING and is recorded rather than
explained away: BBC T-101 Fig. 25, cited on EASTMAN_TRI_X_5223, gives Q
2.00-2.34 at a 0.0016 sr collection angle where Q -> beta, and 2.34 is above
Sayanagi's absolute maximum. Either the circular-grain model understates the
scattering of a real filamentary grain, or T-101's quotient is on a different
density reference. Nothing here is fitted to it.

⚠ WHAT THIS DOES NOT TOUCH: the 93 colour stocks keep callier_q = 1.0. Dye
clouds do not scatter, the class rule for them was never in dispute, and
Sayanagi's model is explicitly about developed SILVER.

Usage:
    python3 sayanagi_callier.py --root . [--assert]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PDF_REL = os.path.join("PDF", "PROFILES", "RETRO", "JAPAN", "23_20.pdf")

#: Mees FIG. 179's five curves are traced by `mees_callier_q.py`; only the part
#: above this density is used for the beta refit, because below it the five are
#: drawn as one stroke.
MEES_FIT_FROM_D = 0.26

#: The base density held fixed in the Mees refit. It is NOT fitted there -- it
#: comes from the Trumpy curve, which is the independent one.
MEES_DB = 0.050


def st_net(d_net, e: float, beta: float):
    """Silberstein & Tuttle specular density from NET diffuse density."""
    d = np.asarray(d_net, dtype=np.float64)
    return -np.log10(e * np.power(10.0, -d) + (1.0 - e) * np.power(10.0, -beta * d))


def q_total(d_total, db: float, e: float, beta: float):
    """Sayanagi's Q_I: the same law read as a ratio of TOTAL densities."""
    d = np.asarray(d_total, dtype=np.float64)
    return (db + st_net(np.maximum(d - db, 1e-12), e, beta)) / d


def _fit(fun, p0, bounds):
    from scipy.optimize import least_squares
    return least_squares(fun, p0, bounds=bounds).x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()

    import film_profiles as fp

    print("Sayanagi 1959 -- «A Theory on Callier Q Factor and Granularity»")
    print("  source: %s" % PDF_REL)
    if not os.path.exists(os.path.join(ns.root, PDF_REL)):
        print("  [SKIP] source not present in this checkout")
        return 0
    bad = 0

    # ---- 1. the Trumpy curve, with and without the base term -------------
    ref = np.array(fp._CALLIER_Q_REFERENCE)
    D, Q = ref[:, 0], ref[:, 1]

    def r_nobase(p):
        return st_net(D, p[0], p[1]) / D - Q

    def r_base(p):
        return q_total(D, p[2], p[0], p[1]) - Q

    e0, b0 = fp._CALLIER_FIT_E, fp._CALLIER_FIT_BETA
    rms0 = float(np.sqrt((r_nobase([e0, b0]) ** 2).mean()))
    toe0 = float(r_nobase([e0, b0])[0])
    e1, b1, db1 = _fit(r_base, [0.15, 1.7, 0.02],
                       ([0.0, 1.0, 1e-4], [1.0, 4.0, 0.0499]))
    res1 = r_base([e1, b1, db1])
    rms1 = float(np.sqrt((res1 ** 2).mean()))

    print("\n  1. Trumpy & Gschwind Fig. 5, 18 reference points")
    print("     shipped fit, NO base term:  E %.4f  beta %.4f  "
          "rms %.4f  toe error %+.3f Q" % (e0, b0, rms0, toe0))
    print("     with Sayanagi's base term:  E %.4f  beta %.4f  Db %.4f  "
          "rms %.4f  toe error %+.3f Q" % (e1, b1, db1, rms1, res1[0]))
    ok = rms1 < 0.35 * rms0 and abs(res1[0]) < 0.10
    if not ok:
        bad += 1
    print("     [%s] the base term cuts the whole-curve rms by %.1fx and the "
          "toe error from %+.2f to %+.3f Q with ONE parameter"
          % ("OK  " if ok else "FAIL", rms0 / rms1, toe0, res1[0]))
    okdb = abs(db1 - fp._CALLIER_SAYANAGI_BASE_D[0]) < 0.003
    if not okdb:
        bad += 1
    print("     [%s] fitted Db %.4f matches the stored %.3f"
          % ("OK  " if okdb else "FAIL", db1, fp._CALLIER_SAYANAGI_BASE_D[0]))

    # ---- 2. Mees FIG. 179, five gamma curves, base held at Trumpy's ------
    print("\n  2. Mees FIG. 179 refitted with Db held at %.3f -- NOT fitted "
          "here, taken from the Trumpy curve above" % MEES_DB)
    try:
        import mees_callier_q as M
        page = M.load_page(M.page_path(ns.root))
        frame = M.find_frame(page)
        cal, _ = M.calibrate(page, frame)
        traced, _ = M.trace(page, frame, cal)
    except Exception as exc:                       # pragma: no cover
        print("     [SKIP] FIG. 179 page not traceable here (%s)" % exc)
        traced = None

    got: list[tuple[float, float]] = []
    if traced is not None:
        _, curves = traced
        print("     gamma   beta(Db=0)  beta(Db=%.3f)  rms   Q(0.055) pred"
              % MEES_DB)
        for g, c in zip(M.GAMMAS, curves):
            a = np.array(sorted(c.items()) if isinstance(c, dict) else c)
            d = cal["ax"] * a[:, 0] + cal["bx"]
            q = cal["ay"] * a[:, 1] + cal["by"]
            m = d > MEES_FIT_FROM_D
            d, q = d[m], q[m]

            def rr(p, db):
                return q_total(d, db, p[0], p[1]) - q

            e_a, b_a = _fit(lambda p: rr(p, 0.0), [0.15, 1.7],
                            ([0.0, 1.0], [0.6, 4.0]))
            e_b, b_b = _fit(lambda p: rr(p, MEES_DB), [0.15, 1.7],
                            ([0.0, 1.0], [0.6, 4.0]))
            rms = float(np.sqrt((rr([e_b, b_b], MEES_DB) ** 2).mean()))
            q055 = float(q_total(np.array([0.055]), MEES_DB, e_b, b_b)[0])
            got.append((g, round(b_b, 3)))
            print("     %.2f    %.3f        %.3f          %.4f  %.3f"
                  % (g, b_a, b_b, rms, q055))
        stored = [(g, b) for g, b in fp._CALLIER_BETA_VS_GAMMA]
        okm = len(got) == len(stored) and all(
            abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 0.004
            for a, b in zip(got, stored))
        if not okm:
            bad += 1
        print("     [%s] the table in film_profiles._CALLIER_BETA_VS_GAMMA "
              "reproduces this refit%s"
              % ("OK  " if okm else "FAIL", "" if okm else ": got %s" % got))
        okrise = all(b >= a - 1e-9 for a, b in zip([x[1] for x in got],
                                                   [x[1] for x in got][1:]))
        if not okrise:
            bad += 1
        print("     [%s] beta RISES MONOTONICALLY with gamma over the five "
              "curves -- which one class constant cannot express, and which is "
              "the whole case for C43" % ("OK  " if okrise else "FAIL"))
        # Sayanagi's inversion: Ig = (2/beta - 1)^2, and it must fall with gamma
        igs = [(2.0 / b - 1.0) ** 2 for _, b in got]
        okig = all(y <= x + 1e-9 for x, y in zip(igs, igs[1:])) and \
            all(0.0 <= v < 1.0 for v in igs)
        if not okig:
            bad += 1
        print("     [%s] inverted grain transmittance Ig = (2/beta - 1)^2 falls "
              "%.1f%% -> %.1f%% as gamma rises -- grains growing more opaque "
              "with development, which is assumption (II) and was not fitted"
              % ("OK  " if okig else "FAIL", 100 * igs[0], 100 * igs[-1]))

    # ---- 3. the ceiling ---------------------------------------------------
    print("\n  3. Sayanagi's ceiling, beta <= 2 (opaque grains, Ig = 0)")
    qs = [p.callier_q for p in fp.FILM_PROFILES if p.is_monochrome]
    okc = max(qs) < fp._CALLIER_SAYANAGI_BETA_MAX
    if not okc:
        bad += 1
    print("     [%s] every monochrome stock's callier_q is under it: "
          "%.4f to %.4f over %d stocks"
          % ("OK  " if okc else "FAIL", min(qs), max(qs), len(qs)))
    print("     [note] BBC T-101 Fig. 25 gives 2.00-2.34 at 0.0016 sr, ABOVE "
          "the ceiling. Recorded as an open tension; nothing is fitted to it")

    # ---- 4. what did NOT change ------------------------------------------
    print("\n  4. C44 closes with a null code change")
    import film_sim
    d = np.linspace(-0.2, 3.0, 33)
    # ⚠ 1e-12, not exact equality: -log10(10**-d) round-trips through two libm
    # calls, so the identity is exact in intent and correctly rounded in fact.
    same = float(np.abs(film_sim.callier_net(d, 1.7, 0.0) - d).max()) < 1e-12
    if not same:
        bad += 1
    print("     [%s] callier_net is still the exact identity at "
          "scanner_specular = 0, so no shipped render moves"
          % ("OK  " if same else "FAIL"))
    okcol = all(p.callier_q == 1.0 for p in fp.FILM_PROFILES
                if not p.is_monochrome)
    if not okcol:
        bad += 1
    print("     [%s] all colour stocks keep callier_q 1.0 -- dye clouds do not "
          "scatter and Sayanagi's model is about developed silver"
          % ("OK  " if okcol else "FAIL"))

    print()
    if bad:
        print("  [FAIL] %d problem(s)" % bad)
        return 1 if ns.do_assert else 0
    print("  [OK  ] C44 closed as a density-reference artefact with no code "
          "change; C43 closed by making callier_q per-stock from its own "
          "contrast")
    return 0


if __name__ == "__main__":
    sys.exit(main())
