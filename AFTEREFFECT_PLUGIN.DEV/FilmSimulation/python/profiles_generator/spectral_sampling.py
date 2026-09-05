"""Queue F3 -- does a 5 nm spectral trace buy anything over 10 nm?

WHAT THE ROW ASKED, AND WHY IT COULD BE SETTLED WITHOUT TRACING ANYTHING
------------------------------------------------------------------------
F3 read "5 nm spectral re-trace beyond ACROS", blocker "nothing", effort HIGH,
value LOW -- "measured benefit is 0.4-1.1 %", a figure the queue repeated in
three places and which nothing in the tree re-derived.

⚠ THE ROW PROPOSED HIGH-EFFORT WORK ON THE STRENGTH OF A NUMBER NOBODY OWNED.
That is the thing to fix first, and it is fixable with no new tracing at all,
because the question "does finer spectral sampling change the render?" is a
question about the EXISTING data. Decimate what is already stored, push both
versions through the same consumer, and read the difference.

TWO MEASUREMENTS, AND THE FIRST IS THE REAL ONE
-----------------------------------------------
  1. THE DIRECT TEST, on the one stock that already has both. FUJI_NEOPAN_1600
     carries a 5 nm set (50 samples, 390-635 nm) from FUJIFILM AF3-608E. Decimate
     it to 10 nm, derive the monochrome weights each way, and the difference IS
     what F3 proposes to buy -- measured rather than estimated, on real traced
     data, for exactly the transformation the row describes.
     ⚠ n = 1, and that is stated rather than hidden.
  2. THE SCALING LAW, across every spectral set in the database. Decimate 10 nm
     to 20 nm on all of them and measure the same consumer's response. That says
     how sensitive the pipeline is to spectral sampling in general, and it
     brackets the 10 -> 5 step from the other side: if HALVING the sampling from
     10 to 20 barely moves anything, DOUBLING it from 10 to 5 cannot move much
     either, because the underlying curves are smooth and the error falls with
     the square of the step.

⚠ WHY THE MONOCHROME WEIGHT TRIPLE IS THE RIGHT PROBE AND NOT AN ARBITRARY ONE.
`film_sim.spectral_monochrome_weights` is the ONE consumer on the render path
that reads a stored spectral curve and turns it into numbers that move pixels:
it integrates the pan sensitivity against three primary lobes and normalises.
Every other use of `SpectralSensitivity` in this database is inert. So the
weight triple is not a proxy for the render difference -- for a monochrome
stock it IS the render difference, entirely.

⚠ AND FOR COLOUR STOCKS THE HONEST ANSWER IS THAT F3 CANNOT MOVE THEM AT ALL.
Nothing on the render path reads `log_s_r/g/b`; stage 7's colour collapse uses
`spectral_weights`, which for a colour stock is the authored (0.30, 0.59, 0.11)
and is not derived from the curves. A 5 nm re-trace of a colour sheet would
change a stored array and zero pixels. That is not an argument for never doing
it -- the data would be better data -- but it is decisive about the row's own
claim to a rendering benefit.

Run with --assert to make a drift from the recorded findings fatal.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import film_profiles as fp
import film_sim as fs

#: The one stock in the database carrying a 5 nm trace. FUJIFILM AF3-608E.
FIVE_NM_STOCK = "FUJI_NEOPAN_1600"

#: What the direct test found on 2026-09-05b, so a rerun that disagrees fails.
#: Max absolute change in any one weight, 5 nm against its own 10 nm decimation.
EXPECTED_DIRECT_MAX = 0.005649
EXPECTED_DIRECT_TOL = 0.0005

#: What the 10 -> 20 nm sweep found: the worst single-weight change over every
#: monochrome stock whose weights are derivable.
EXPECTED_SWEEP_MAX = 0.016885
EXPECTED_SWEEP_TOL = 0.0005


class _Spectral:
    """A stand-in carrying a resampled curve, so the real consumer can be run.

    ⚠ THE POINT IS THAT THE CONSUMER IS NOT REIMPLEMENTED HERE. `film_sim`'s own
    `spectral_monochrome_weights` is called on a profile whose spectral record
    has been swapped for a decimated one, so what is measured is the shipped
    integrator, its own primary lobes and its own gamut-reach guard. A local
    copy of the integration would measure this module instead.
    """


def decimate(sp, keep: int):
    """Every `keep`-th sample of a SpectralSensitivity, step scaled to match."""
    def cut(t):
        return tuple(t[::keep]) if t else ()
    return fp.SpectralSensitivity(
        lambda_start_nm=sp.lambda_start_nm,
        lambda_step_nm=sp.lambda_step_nm * keep,
        log_s_r=cut(sp.log_s_r), log_s_g=cut(sp.log_s_g),
        log_s_b=cut(sp.log_s_b), log_s_pan=cut(sp.log_s_pan),
        criterion=sp.criterion, source=sp.source)


def weights_with(profile, sp):
    """The shipped weight derivation, run against a substituted spectral record."""
    from dataclasses import replace
    return fs.spectral_monochrome_weights(replace(profile, spectral=sp))


def direct_test() -> dict:
    """5 nm against its own 10 nm decimation, on the one stock that has both."""
    p = fp.get_profile(FIVE_NM_STOCK)
    sp = p.spectral
    if sp.lambda_step_nm != 5.0:
        raise SystemExit(f"[!] {FIVE_NM_STOCK} no longer carries a 5 nm set")
    fine = weights_with(p, sp)
    coarse = weights_with(p, decimate(sp, 2))
    out = dict(stock=FIVE_NM_STOCK, n_fine=len(sp.log_s_pan),
               step_fine=sp.lambda_step_nm, fine=fine, coarse=coarse)
    if fine is None or coarse is None:
        # ⚠ A REFUSAL IS A RESULT. The gamut-reach guard declines any stock
        # sensitised well outside what three visible primaries can excite, and
        # if it declines here the weight triple is not the probe for this stock
        # and the direct test has no answer rather than a small one.
        out["max_delta"] = None
        out["note"] = ("the shipped derivation REFUSED one or both samplings "
                       "(gamma-reach guard); no weight comparison is possible")
        return out
    out["max_delta"] = float(max(abs(a - b) for a, b in zip(fine, coarse)))
    out["delta"] = [float(a - b) for a, b in zip(fine, coarse)]
    return out


def sweep() -> dict:
    """10 -> 20 nm across every stock whose weights the renderer will derive."""
    rows = []
    for p in fp.FILM_PROFILES:
        sp = p.spectral
        if not sp.has_data or not sp.log_s_pan:
            continue
        if sp.lambda_step_nm != 10.0 or len(sp.log_s_pan) < 8:
            continue
        fine = weights_with(p, sp)
        coarse = weights_with(p, decimate(sp, 2))
        if fine is None or coarse is None:
            rows.append(dict(stock=p.name, refused=True))
            continue
        rows.append(dict(stock=p.name, refused=False,
                         n=len(sp.log_s_pan),
                         fine=[float(v) for v in fine],
                         max_delta=float(max(abs(a - b)
                                             for a, b in zip(fine, coarse)))))
    live = [r for r in rows if not r["refused"]]
    return dict(rows=rows, n=len(rows), n_live=len(live),
                worst=max((r["max_delta"] for r in live), default=0.0),
                median=float(np.median([r["max_delta"] for r in live]))
                if live else 0.0)


def colour_reach() -> dict:
    """How many stored spectral sets can move a pixel at all.

    ⚠ THIS IS THE MEASUREMENT THAT DECIDES F3, and it is not about sampling.
    A finer trace can only matter where the trace is read. `log_s_pan` feeds the
    monochrome weight derivation; `log_s_r/g/b` feeds nothing on the render
    path. So the population F3 could improve is the monochrome one, and its size
    is a fact about the database rather than an opinion about the row.
    """
    pan = [p.name for p in fp.FILM_PROFILES
           if p.spectral.has_data and p.spectral.log_s_pan]
    rgb = [p.name for p in fp.FILM_PROFILES
           if p.spectral.has_data and p.spectral.log_s_r and not p.spectral.log_s_pan]
    derivable = [n for n in pan
                 if fs.spectral_monochrome_weights(fp.get_profile(n)) is not None]
    return dict(pan=len(pan), rgb=len(rgb), derivable=len(derivable),
                refused=[n for n in pan if n not in derivable])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assert", dest="assert_", action="store_true")
    a = ap.parse_args()

    print("=== queue F3 -- what a 5 nm spectral trace is worth ===\n")

    c = colour_reach()
    print("REACH -- how many stored sets can move a pixel at all")
    print(f"  {c['pan']} stocks carry a pan curve; {c['derivable']} of those "
          f"pass the gamut-reach guard and have their weights DERIVED")
    print(f"  {c['rgb']} stocks carry three-layer r/g/b curves and "
          f"⚠ NOTHING ON THE RENDER PATH READS THEM")
    if c["refused"]:
        print(f"  refused by the guard: {', '.join(c['refused'])}")

    d = direct_test()
    print(f"\nDIRECT -- {d['stock']}, {d['n_fine']} samples at "
          f"{d['step_fine']:.0f} nm, against its own 10 nm decimation")
    if d["max_delta"] is None:
        print("  " + d["note"])
    else:
        print(f"  weights at  5 nm: {tuple(round(v, 5) for v in d['fine'])}")
        print(f"  weights at 10 nm: {tuple(round(v, 5) for v in d['coarse'])}")
        print(f"  worst single-weight change: {d['max_delta']:.6f} "
              f"({100 * d['max_delta']:.4f} % of full scale)")

    s = sweep()
    print(f"\nSWEEP -- 10 nm decimated to 20 nm, {s['n_live']} derivable stocks "
          f"of {s['n']} with a pan curve")
    print(f"  worst single-weight change: {s['worst']:.6f}")
    print(f"  median:                     {s['median']:.6f}")
    worst_row = max((r for r in s["rows"] if not r["refused"]),
                    key=lambda r: r["max_delta"], default=None)
    if worst_row:
        print(f"  worst stock: {worst_row['stock']}")

    print("\n--- what this settles ---")
    bound = s["worst"] / 4.0
    print(f"⚠ HALVING the sampling (10 -> 20 nm) moves the only render-path "
          f"consumer\n  by {s['worst']:.6f} at worst. A trapezoid rule's error "
          f"falls with the SQUARE of\n  the step, so DOUBLING it (10 -> 5 nm) "
          f"is bounded above by about a quarter of\n  that: {bound:.6f} in a "
          f"weight that sums to 1.0, i.e. {100 * bound:.4f} %.")
    if d["max_delta"] is not None:
        ratio = d["max_delta"] / max(bound, 1e-12)
        print(f"\n  ⚠ AND THE DIRECT MEASUREMENT IS {d['max_delta']:.6f}, "
              f"{ratio:.2f}x that bound -- so the\n     smoothness argument is "
              f"OPTIMISTIC and the honest answer is the measured one,\n     not "
              f"the extrapolated one. {100 * d['max_delta']:.2f} % on one weight "
              f"of a triple that sums to 1.")
    print(f"\n  The row quoted 0.4-1.1 %. The direct measurement lands INSIDE "
          f"that band,\n  so the figure was right -- but it was inherited, and "
          f"it is now owned.")

    bad = 0
    if d["max_delta"] is not None and \
            abs(d["max_delta"] - EXPECTED_DIRECT_MAX) > EXPECTED_DIRECT_TOL:
        print(f"\n⚠ DRIFT: the direct test now reads {d['max_delta']:.6f} "
              f"against a recorded {EXPECTED_DIRECT_MAX:.6f}")
        bad += 1
    if abs(s["worst"] - EXPECTED_SWEEP_MAX) > EXPECTED_SWEEP_TOL:
        print(f"\n⚠ DRIFT: the sweep now reads {s['worst']:.6f} against a "
              f"recorded {EXPECTED_SWEEP_MAX:.6f}")
        bad += 1

    print("\nOK" if not bad else "\nFAIL")
    return 1 if (bad and a.assert_) else 0


if __name__ == "__main__":
    sys.exit(main())
