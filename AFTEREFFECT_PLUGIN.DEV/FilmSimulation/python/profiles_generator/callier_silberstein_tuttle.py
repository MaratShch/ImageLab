"""Silberstein & Tuttle's relation, against the LEGACY law it replaced.

⚠ ADOPTED 2026-08-30 (queue M3). `film_sim.callier_net` and
`AlgoCallierNet` now implement the law below; `silberstein_tuttle` here is
the independent second implementation the comparison needs, and `legacy_linear`
is the multiplier interpolation that used to ship. This module is kept as the
RECORD OF THE CHANGE: it measures what moved, and it pins the two properties
the replacement had to preserve -- exact inertness at specular 0 and exact
inertness for every colour stock at any setting.

THE SOURCE
----------
Mees, *The Theory of the Photographic Process*, Chapter XVII "The Measurement of
Density", **printed page 644**, immediately after the Callier discussion that
FIG. 179 belongs to. Silberstein and Tuttle derive, and the book prints:

    10 ** -D_sp   =   E * 10 ** -D_diff   +   (1 - E) * 10 ** -(beta * D_diff)

with the book's own definitions, quoted:

  * **E** is "a constant expressing the fraction of the scattered light which
    emerges normally or quasi-normally; i.e., the amount accepted by the
    photometric field of a densitometer **or by the projection lens of such a
    device as an enlarging printer**".
  * **beta** is "unity plus the ratio of scattering to absorption coefficients".
  * "If E = 0, beta is numerically equal to Callier's Q. If E = 1.0,
    D_sp = D_diff."

⚠ THAT IS THE FILM x GEOMETRY SPLIT THIS PROJECT INVENTED FOR ITSELF IN C22, IN
PRINT SINCE 1942. `AlgoControls::scannerSpecular` is `1 - E` -- how little of the
scattered light the reader accepts -- and `FilmProfile.callier_q` is `beta`, a
property of the deposit. The C22 note records the split as a design decision
argued from first principles because no source stated one. A source states one.

WHAT THE CHANGE BOUGHT, AND IT IS NOT ACCURACY AT THE DEFAULT
--------------------------------------------------------------
Both laws are exactly inert at `specular = 0`, so the change moved no shipped
render. What it changed is what the control MEANS once it is turned up, and the
shape of the curve it sweeps:

  * The LEGACY law was `D_read = dmin + (D - dmin) * (1 + s*(Q-1))` -- a linear
    interpolation of the MULTIPLIER, chosen because it has the right endpoints.
  * Silberstein-Tuttle is not linear in `s`, because light adds and densities do
    not. Mixing an accepted and a rejected beam averages TRANSMITTANCES.
  * ⚠ So the two agree EXACTLY at s = 0 and s = 1 -- precisely where anyone
    would hand-check them -- and by up to 0.21 D everywhere in between.

⚠ AND THE TWO DISAGREE MOST WHERE THE PICTURE LIVES. The comparison below is run
over the real database; the divergence is reported as a density error at each
net density, so the size of the disagreement is a number rather than an opinion.

⚠ WHAT SILBERSTEIN-TUTTLE DOES NOT DO IS FIX THE TOE. Expanding for small D
gives `Q -> E + (1-E)*beta`, a CONSTANT: the model says Q is flat at low density
and falls as density rises. Mees FIG. 179, three pages earlier in the same
chapter, measures Q COLLAPSING to 1.04 at D 0.055. The two sources disagree
about the toe, and the measurement wins over the model. So this law is a better
interpolation between two readers; it is not a substitute for the traced shape,
and a future toe correction has to come from FIG. 179 either way.

Run:
    python callier_silberstein_tuttle.py            # the comparison
    python callier_silberstein_tuttle.py --assert   # non-zero exit on drift
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from film_profiles import FILM_PROFILES

#: Net densities the two laws are compared on. Spans a real tone scale.
NET_D = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50, 3.00)

#: Reader settings. 0 is the shipped default and must be inert in both laws.
SPECULAR = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Below this a density difference is invisible in an 8-bit render: 1/255 of a
#: stop is far smaller, but density is not luminance, and 0.002 D is about
#: a quarter of a 255th at mid grey. Used only for reporting.
VISIBLE_D = 0.002


def legacy_linear(d_net, q, s):
    """`D_read - dmin` under the law film_sim and AlgoCallier shipped until G3."""
    return d_net * (1.0 + s * (q - 1.0))


def silberstein_tuttle(d_net, beta, s):
    """`D_read - dmin` under 10^-Dsp = E*10^-Dd + (1-E)*10^-(beta*Dd), E = 1-s.

    ⚠ THE ARGUMENT IS NET DENSITY, NOT TOTAL, AND THAT IS THE PROJECT'S CHOICE
    RATHER THAN THE BOOK'S. Silberstein and Tuttle write plain D. Our law is
    referenced to dmin because the scattering scales with developed silver and
    clear base carries none -- C22's reasoning, unchanged. Feeding total density
    here would make a condenser darken the film base, which no densitometer
    measures. The substitution is recorded because it is a real difference
    between what the book states and what this function computes.
    """
    e = 1.0 - s
    t = e * np.power(10.0, -d_net) + (1.0 - e) * np.power(10.0, -beta * d_net)
    return -np.log10(np.maximum(t, 1e-300))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="accepted and unused")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    args = ap.parse_args(argv)

    mono = [p for p in FILM_PROFILES if p.is_monochrome]
    colour = [p for p in FILM_PROFILES if not p.is_monochrome]
    qs = sorted({p.callier_q for p in mono})
    bad = 0

    print("Silberstein & Tuttle against the legacy Callier law")
    print("  Mees printed p644; %d monochrome stocks carry Q in %s, "
          "%d colour stocks carry Q = 1.0"
          % (len(mono), ", ".join("%.2f" % q for q in qs), len(colour)))

    # ⚠ INERT AT THE DEFAULT, AS AN IDENTITY IN BOTH LAWS. Checked before
    # anything else, because a law that is not exactly inert at 0 changes every
    # render ever made and no amount of better physics excuses that.
    worst0 = 0.0
    for q in qs + [1.0]:
        for d in NET_D:
            a = legacy_linear(d, q, 0.0)
            b = silberstein_tuttle(d, q, 0.0)
            worst0 = max(worst0, abs(a - d), abs(b - d))
    print("  inert at specular 0: worst departure %.3e" % worst0)
    if worst0 > 1e-12:
        print("[!] a law is not exactly inert at specular = 0")
        bad += 1

    # Q = 1.0 must be inert at ANY setting -- a dye image does not scatter.
    worst1 = 0.0
    for s in SPECULAR:
        for d in NET_D:
            worst1 = max(worst1,
                         abs(legacy_linear(d, 1.0, s) - d),
                         abs(silberstein_tuttle(d, 1.0, s) - d))
    print("  Q = 1.0 inert at every setting: worst %.3e" % worst1)
    if worst1 > 1e-12:
        print("[!] Q = 1.0 is not inert -- all %d colour stocks would move"
              % len(colour))
        bad += 1

    print("")
    print("  density read, legacy vs Silberstein-Tuttle, at Q = 1.30 "
          "(the monochrome negative class value):")
    print("    %-7s %s" % ("net D", "".join("%12s" % ("s=%.2f" % s)
                                            for s in SPECULAR)))
    for d in NET_D:
        cells = []
        for s in SPECULAR:
            a = legacy_linear(d, 1.30, s)
            b = silberstein_tuttle(d, 1.30, s)
            cells.append("%6.3f/%5.3f" % (a, b))
        print("    %-7.2f %s" % (d, "".join("%12s" % c for c in cells)))

    # ⚠ THE HEADLINE NUMBER: how far apart the two laws are, over the real
    # database rather than over a chosen example.
    print("")
    worst, worst_at, n_visible, n_tot = 0.0, None, 0, 0
    for p in mono:
        q = float(p.callier_q)
        for s in SPECULAR:
            for d in NET_D:
                a = legacy_linear(d, q, s)
                b = silberstein_tuttle(d, q, s)
                n_tot += 1
                e = abs(a - b)
                if e > VISIBLE_D:
                    n_visible += 1
                if e > worst:
                    worst, worst_at = e, (p.name, q, s, d)
    print("  over %d monochrome stocks x %d settings x %d densities = %d points"
          % (len(mono), len(SPECULAR), len(NET_D), n_tot))
    print("  worst disagreement %.4f D at %s Q=%.2f s=%.2f netD=%.2f"
          % (worst, worst_at[0], worst_at[1], worst_at[2], worst_at[3]))
    print("  %d of %d points differ by more than %.3f D"
          % (n_visible, n_tot, VISIBLE_D))

    # ⚠ WHICH WAY, AND WHERE. A signed summary, because "they differ by 0.06"
    # is not actionable and "the shipped law over-darkens the shadows by up to
    # 0.06 D while agreeing in the highlights" is.
    print("")
    print("  signed difference, legacy minus Silberstein-Tuttle, Q = 1.30:")
    for s in (0.5, 1.0):
        row = [legacy_linear(d, 1.30, s) - silberstein_tuttle(d, 1.30, s)
               for d in NET_D]
        print("    s=%.2f  " % s + " ".join("%+.3f" % v for v in row))

    # ⚠ AND THE TOE, WHICH IS WHERE THIS LAW DOES NOT HELP. Both laws hold the
    # multiplier at its full value as density goes to zero; FIG. 179 measures it
    # collapsing to 1.04 by D 0.055. Asserted so nobody adopts this expecting
    # the toe fixed.
    q = 1.30
    for s in (1.0,):
        mult_toe = silberstein_tuttle(0.05, q, s) / 0.05
        mult_mid = silberstein_tuttle(1.00, q, s) / 1.00
        print("")
        print("  Silberstein-Tuttle multiplier at s=1, Q=%.2f: "
              "%.4f at net D 0.05, %.4f at net D 1.00" % (q, mult_toe, mult_mid))
        if mult_toe < mult_mid:
            print("[!] Silberstein-Tuttle now predicts a LOWER multiplier at "
                  "the toe than at mid scale -- it does not, and if it did "
                  "this module's conclusion about FIG. 179 would change")
            bad += 1

    # ⚠ AND THE SHIPPED IMPLEMENTATION MUST STILL BE THIS LAW. `silberstein_tuttle`
    # above is written from the book independently of `film_sim.callier_net`; if
    # they ever part company, one of them has been edited and this says so. That
    # is the difference between a module that recorded a decision once and one
    # that keeps the decision true.
    import film_sim as _fs
    worst_impl = 0.0
    for p in mono:
        q = float(p.callier_q)
        for s in SPECULAR:
            for d in NET_D:
                worst_impl = max(worst_impl, abs(
                    float(_fs.callier_net(d, q, s)) - silberstein_tuttle(d, q, s)))
    print("")
    print("  film_sim.callier_net against this module's own reading of the "
          "book: worst %.2e" % worst_impl)
    if worst_impl > 1e-12:
        print("[!] the shipped law is no longer Silberstein & Tuttle")
        bad += 1

    if args.assert_:
        if bad:
            print("[FAIL] the Silberstein-Tuttle comparison does not reproduce")
            return 1
        print("[OK] the shipped law IS Silberstein & Tuttle to %.1e; both it "
              "and the legacy law are exactly inert at specular 0 and at "
              "Q = 1.0; they diverge by at most %.4f D over the database, and "
              "neither reproduces FIG. 179's toe collapse"
              % (worst_impl, worst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
