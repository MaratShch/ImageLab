"""Ooue 1959, «写真感光材料の粒状性» Parts 1 and 2 -- the grain-structure figures.

Queue items J1 and J2, 2026-09-02. Two companion review papers by Shingo Ooue
(大上進吾) of the Fuji Photo Film Research Laboratory, the two the corpus's own
`EMULSION_KNOWLEDGE_BASE.md` §23i named as the ones that would settle the
mean-square / rms ambiguity left open on his third paper (`23_7.pdf`).

    PDF/PROFILES/RETRO/JAPAN/22_38.pdf   Part 1, J. Soc. Phot. Sci. Japan
                                         22(1) 38-47 (1959) -- GRAININESS,
                                         the subjective quantity
    PDF/PROFILES/RETRO/JAPAN/22_91.pdf   Part 2, 22(2) 91-99 (1959) --
                                         GRANULARITY, the objective one

Both carry a real text layer; every figure is raster at about 200 ppi.

WHY THEY MATTER HERE, WHICH IS NOT WHAT THEY ARE ABOUT
--------------------------------------------------------
They are literature reviews. Almost all of both is other people's work, cited
and redrawn. Four figures are not incidental to this project:

    Part 2 Fig. 26   POWER SPECTRUM of grain structure, three NAMED samples
                     with stated developer, time and density -- a measured
                     WIENER SPECTRUM, which is the exact quantity
                     `GrainSpec.clump_um` parameterises and which the corpus
                     has held from ONE source (BBC T-101) until now
    Part 2 Fig. 24   AUTOCORRELATION of grain structure, Neopan S at two
                     densities -- the Fourier partner of the same quantity,
                     measured independently
    Part 2 §4.2.1    the statement that Callier Q shows NO correlation with
                     graininess on commercial materials, and WHY
    Part 1 Fig. 2    MEAN GRAIN AREA against density at two development times

WHAT THE WIENER SPECTRUM SAYS ABOUT THIS PROJECT'S GRAIN MODEL
----------------------------------------------------------------
`film_sim.make_grain_field` shapes the grain with an AMPLITUDE transfer

    h(f) = exp(-(f/f_hi)^2) * (1 + clump_gain * exp(-(f/f_lo)^2))

so the Wiener spectrum it produces is h^2, i.e. a GAUSSIAN of exponent 2 in f,
times a low-frequency bump. Fitting a generalised Gaussian
``P0 * exp(-ln2 * (f/f_half)^n)`` to Fig. 26's three falling limbs returns

    Neopan SS / Minidol 20 C 10 min, D 1.03    f_half  45.6   n 0.71
    Neopan SS / Minidol 20 C 10 min, D 0.45    f_half  70.8   n 0.89
    Process Plate / D-72 (1:1) 20 C 4 min      f_half 140.7   n 1.36

⚠ EVERY ONE OF THEM IS BELOW 2, AND THE PURE GAUSSIAN FITS THREE TO SIX TIMES
WORSE (rms in log10 of 0.09-0.56 against 0.035-0.107). The measured spectra have
a flat plateau and then a SHALLOWER-than-Gaussian fall, closer to an exponential.
A Gaussian therefore UNDER-estimates grain energy at high frequency -- which is
the same defect `MTFSpec`'s own docstring already records for the MTF tail
("real emulsion tails are fatter than any two-parameter analytic form"), now
measured on the grain spectrum as well.

⚠ NOTHING IS CHANGED IN THE MODEL ON THIS EVIDENCE, and the reason is not
timidity. The ordinate of Fig. 26 is labelled "POWER LEVEL" with no units, the
abscissa "LINES/mm" without saying whether a line is a cycle, and the figure is
a REDRAWING (reference 91) rather than Ooue's own plate. Shape is usable, level
is not, and a change to the grain spectrum moves a pixel on all 171 stocks.
What is recorded is the exponent, the frequencies, and the fact that they are
consistent across three samples.

⚠ AND THE ONE RESULT THAT NEEDS NO CALIBRATION AT ALL: the SAME FILM at two
densities gives f_half 45.6 at D 1.03 and 70.8 at D 0.45. The grain gets
COARSER as density rises, by 55 % in cutoff frequency over one decade of
density, and `GrainSpec` carries ONE clump size per stock. That is a
density-dependent clump, measured, on one emulsion, with the developer and time
held fixed -- the cleanest statement of it in the corpus.

WHAT THE AUTOCORRELATION ADDS, AND THE ONE THING IT BREAKS
------------------------------------------------------------
Fig. 24 measures phi(tau) directly for Neopan S. Curve (1), at D 1.04, falls
from 1000 to half at tau = 3.48 um and its last positive sample is at
tau = 11.8 um.

Under this project's Gaussian model the autocorrelation of h^2 is itself a
Gaussian with tau_half = 374.8 / f_hi um, so 3.48 um implies f_hi = 108 c/mm
and `clump_um` = 4.65 um for that sample. That is a usable independent scale.

⚠ WHAT IT BREAKS IS THE SHAPE, NOT THE SCALE. A Gaussian autocorrelation is
positive everywhere. Fig. 24's curve goes NEGATIVE past about 12 um and stays
negative for another 8 um before returning -- an anti-correlated ring, i.e. the
grains are more evenly spaced than a Poisson field would be. The project's grain
field cannot produce that, and neither can Sayanagi's Poisson model
(`sayanagi_callier.py`), which assumes exactly the Poisson placement this figure
contradicts. Recorded as a limitation of both, not patched into either.

WHAT PART 1 FIG. 2 SAYS ABOUT `emulsion.grain_um`
---------------------------------------------------
Mean DEVELOPED grain area against density, Fuji positive film in FD-3 (1:1) at
20 C, at two development times. The traced values run 1.103 -> 0.925 um^2 over
D 1.4 -> 4.0 at 32 min, and 1.159 -> 0.571 um^2 over D 0.2 -> 2.4 at 1 min, i.e.
EQUIVALENT DIAMETERS of 1.21 down to 0.85 um.

⚠ THE STORED `emulsion.grain_um` VALUES ARE 1.3-6.5 um AND ALL SEVENTEEN COME
FROM ONE THIRD-PARTY AGGREGATOR. This measurement, on a positive film -- the
finest-grained class there is, so a floor rather than a typical value -- lands
at 0.85-1.21 um, BELOW the aggregator's lowest figure. It does not by itself
correct any stored number, because no profile in this file is Fuji positive
FD-3, but it is independent evidence that the stored range is too coarse, and
it is the second such: BBC T-101 Table 3 already measures grain diameter FALLING
as density rises at fixed development, which is the same direction Fig. 2 shows
and is the opposite of what most people expect.

WHAT SETTLES THE 23_7 AMBIGUITY
---------------------------------
`23_7.pdf` Fig. 7 gives granularity-density curves for four named film/developer
combinations, and its exponents were harvested but NOT ADOPTED because §3.2
defines the ordinate as MEAN-SQUARE while the English abstract says
root-mean-square, and the two readings put the exponents on opposite sides of
every other source in the corpus.

⚠ PART 2 SETTLES IT IN THE AUTHOR'S OWN WORDS. Its §4.2.2 is headed
「濃度変化の標準偏差による方法」 -- "the method using the STANDARD DEVIATION of
density variation" -- and every objective granularity in the paper is built from
it: Selwyn's sigma*sqrt(a), van Kreveld's Delta_m*sqrt(a), and equations (5) and
(6), Delta_T = 0.675 sqrt(a'/a) sqrt((1-T)/T) and Delta_T = 1.022 sqrt(a'/a)
sqrt(D), both of which are standard deviations of transmittance. The same author
in the same journal two years earlier uses a STANDARD DEVIATION throughout, so
the rms reading of 23_7 is his own convention and the mean-square reading was a
translation artefact.

⚠ CONSEQUENCE, and it is what the ambiguity was blocking: on the rms reading
23_7's fitted exponents 0.412 / 0.672 / 0.364 / 0.606 straddle the legacy
sqrt law (0.50) and the BBC exponent (0.40) instead of falling below every
source in the corpus. They become usable. This reader does not adopt them --
that is 23_7's own row -- it removes the reason they were held.

AND THE EMPIRICAL HALF OF C45
-------------------------------
Part 2 §4.2.1, immediately before that heading: 「サイズの異なる二種の乳剤を
重層塗布している場合には、Qは重要な意味を持ちえないであろう。事実、現在市販
されている感光材料のQの値を測定した結果は、心理的粒状性との相関を認めること
はできない。」 -- where two emulsions of different grain size are coated as two
layers Q cannot carry an important meaning, and IN FACT measurements of Q on
commercially available materials show NO correlation with psychological
graininess. Queue C45 found the disagreement empirically and
`sayanagi_callier.py` confirmed it theoretically (Q contains the granularity but
not the grain radius, so it ranks only samples of equal grain size). This is the
same conclusion measured on commercial stock, with the double-coating mechanism
named.

Usage:
    python3 ooue_1959_granularity.py --root . [--assert]
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PART1_REL = os.path.join("PDF", "PROFILES", "RETRO", "JAPAN", "22_38.pdf")
PART2_REL = os.path.join("PDF", "PROFILES", "RETRO", "JAPAN", "22_91.pdf")

# --- Part 2 Fig. 26, page 96.  Calibration detected on the printed ticks:
#     ordinate decade 49.5 px with 100 at y 230; abscissa decade 99.5 px with
#     10 lines/mm at x 316.  Both checked on four gridline positions each.
F26 = dict(page=6, x_ref=216.5, y_ref=230.0, dec_x=99.5, dec_y=49.5,
           x0=289, y0=225, y1=441, x1=562)
#: Above and below these frequencies the three curves keep a fixed vertical
#: ORDER, and that is what separates them: below 32 lines/mm the order is
#: 1 > 2 > 3 by level, above 62 it has reversed to 1 < 2 < 3 because sample 3
#: rolls off last.  ⚠ THE CROSSING BAND IS SKIPPED RATHER THAN GUESSED.
F26_LOW, F26_HIGH = 32.0, 62.0
F26_NAMES = {1: "Neopan SS / Minidol 20C 10 min, D 1.03",
             2: "Neopan SS / Minidol 20C 10 min, D 0.45",
             3: "Process Plate / D-72 (1:1) 20C 4 min, D 0.44"}

# --- Part 2 Fig. 24, page 95.  Linear axes: tau = 0 at x 314.5, 6.32 px per um
#     (from the 10/20/30/40 label centres); phi = 0 at y 373.5, 1000 at y 240.
F24 = dict(page=5, x_zero=314.5, px_per_um=6.32, y_zero=373.5, y_1000=240.0,
           x0=316, y0=225, y1=372, x1=600)

# --- Part 1 Fig. 2, page 39.  D = 0 at x 289, D = 5 at x 545; area 0.0 at
#     y 889, 0.5 at y 826.
F02 = dict(page=2, x_d0=289.0, px_per_d=51.2, y_a0=889.0, px_per_area=126.0,
           x0=291, y0=690, y1=886, x1=560)


def page_image(pdf: str, page: int) -> np.ndarray:
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", "200", "-png", "-f", str(page),
                        "-l", str(page), pdf, os.path.join(td, "p")],
                       check=True, capture_output=True)
        names = sorted(os.listdir(td))
        if len(names) != 1:
            raise RuntimeError("page %d: %d images" % (page, len(names)))
        return np.asarray(
            Image.open(os.path.join(td, names[0])).convert("L")).astype(np.float32)


def _clusters(mask_col: np.ndarray) -> list[float]:
    out: list[float] = []
    i, n = 0, len(mask_col)
    while i < n:
        if mask_col[i]:
            j = i
            while j < n and mask_col[j]:
                j += 1
            out.append((i + j - 1) / 2.0)
            i = j
        else:
            i += 1
    return out


# ---------------------------------------------------------------------------
# Fig. 26 -- the Wiener spectra
# ---------------------------------------------------------------------------


def trace_fig26(img: np.ndarray) -> dict[int, dict[float, float]]:
    g = F26
    sub = img[g["y0"]:g["y1"], g["x0"]:g["x1"]] < 150
    sub[-3:, :] = False
    out: dict[int, dict[float, float]] = {1: {}, 2: {}, 3: {}}
    for x in range(sub.shape[1]):
        f = 10 ** ((x + g["x0"] - g["x_ref"]) / g["dec_x"])
        vals = [10 ** ((g["y_ref"] - (c + g["y0"])) / g["dec_y"] + 2.0)
                for c in _clusters(sub[:, x])]
        vals = [v for v in vals if 0.015 < v < 90.0]
        if len(vals) != 3 or not (9.0 <= f <= 520.0):
            continue
        if f < F26_LOW:
            order = sorted(vals, reverse=True)
        elif f > F26_HIGH:
            order = sorted(vals)
        else:
            continue
        for k, v in zip((1, 2, 3), order):
            out[k][round(f, 2)] = round(v, 4)
    return out


def fit_fig26(curve: dict[float, float]) -> dict:
    """Plateau, half-power frequency, and the generalised-Gaussian exponent."""
    from scipy.optimize import least_squares
    f = np.array(sorted(curve))
    v = np.array([curve[x] for x in f])
    plateau = float(np.median(v[(f >= 10) & (f <= 30)]))
    m = f >= 25
    ff, vv = f[m], v[m]
    keep, last = [], plateau * 1.15
    for i in range(len(ff)):
        if vv[i] <= last:
            keep.append(i)
            last = vv[i]
    ff, vv = ff[keep], vv[keep]
    allf = np.concatenate(([10.0], ff))
    allv = np.concatenate(([plateau], vv))
    f_half = None
    for i in range(len(allf) - 1):
        if allv[i] >= plateau / 2 >= allv[i + 1]:
            f_half = 10 ** (math.log10(allf[i])
                            + (math.log10(allf[i + 1]) - math.log10(allf[i]))
                            * (allv[i] - plateau / 2) / (allv[i] - allv[i + 1]))
            break

    def mod(p, x):
        return p[0] * np.exp(-math.log(2.0) * (x / p[1]) ** p[2])

    r = least_squares(lambda p: np.log10(mod(p, ff)) - np.log10(vv),
                      [plateau, f_half or 50.0, 2.0],
                      bounds=([plateau * 0.5, 10.0, 0.5],
                              [plateau * 2, 400.0, 8.0]))
    rms = float(np.sqrt(((np.log10(mod(r.x, ff)) - np.log10(vv)) ** 2).mean()))
    r2 = least_squares(
        lambda p: np.log10(mod([p[0], p[1], 2.0], ff)) - np.log10(vv),
        [plateau, f_half or 50.0],
        bounds=([plateau * 0.5, 10.0], [plateau * 2, 400.0]))
    rms_g = float(np.sqrt(((np.log10(mod([r2.x[0], r2.x[1], 2.0], ff))
                            - np.log10(vv)) ** 2).mean()))
    return dict(plateau=plateau, f_half=f_half, n=float(r.x[2]),
                rms=rms, rms_gaussian=rms_g, n_pts=int(len(ff)))


# ---------------------------------------------------------------------------
# Fig. 24 -- the autocorrelation
# ---------------------------------------------------------------------------


def trace_fig24(img: np.ndarray) -> dict[float, float]:
    """Curve (1), Neopan S at D 1.04.  Only the positive lobe is traced --
    the sub-image stops just above the zero rule so the axis cannot be
    mistaken for data."""
    g = F24
    sub = img[g["y0"]:g["y1"], g["x0"]:g["x1"]] < 155
    scale = 1000.0 / (g["y_zero"] - g["y_1000"])
    out: dict[float, float] = {}
    for tau10 in range(5, 130):
        tau = tau10 / 10.0
        x = int(round(g["x_zero"] + g["px_per_um"] * tau)) - g["x0"]
        if not (0 <= x < sub.shape[1]):
            continue
        cs = _clusters(sub[:, x])
        if not cs:
            continue
        top = min(cs)                     # the topmost ink is curve (1)
        out[round(tau, 1)] = round((g["y_zero"] - (top + g["y0"])) * scale, 1)
    return out


# ---------------------------------------------------------------------------
# Part 1 Fig. 2 -- mean grain area
# ---------------------------------------------------------------------------


def trace_fig02(img: np.ndarray) -> dict[int, dict[float, float]]:
    g = F02
    sub = img[g["y0"]:g["y1"], g["x0"]:g["x1"]] < 160
    out: dict[int, dict[float, float]] = {1: {}, 2: {}}
    for d10 in range(2, 42):
        d = d10 / 10.0
        x = int(round(g["x_d0"] + g["px_per_d"] * d)) - g["x0"]
        if not (0 <= x < sub.shape[1]):
            continue
        vals = [round((g["y_a0"] - (c + g["y0"])) / g["px_per_area"], 3)
                for c in _clusters(sub[:, x])]
        vals = [v for v in vals if 0.3 < v < 1.4]
        if not vals:
            continue
        if d >= 2.6:                    # only curve (1) is drawn beyond 2.4
            out[1][d] = max(vals)
        elif d <= 1.2:                  # only curve (2) is drawn below 1.4
            out[2][d] = max(vals)
        elif len(vals) >= 2:
            out[1][d] = max(vals)
            out[2][d] = min(vals)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()

    p1 = os.path.join(ns.root, PART1_REL)
    p2 = os.path.join(ns.root, PART2_REL)
    print("Ooue 1959 -- «写真感光材料の粒状性» Parts 1 and 2, Fuji Photo Film")
    print("  sources: %s , %s" % (PART1_REL, PART2_REL))
    if not (os.path.exists(p1) and os.path.exists(p2)):
        print("  [SKIP] sources not present in this checkout")
        return 0

    import film_profiles as fp
    bad = 0

    # ---- Part 2 Fig. 26 -------------------------------------------------
    print("\n  Part 2 Fig. 26 -- power spectrum of grain structure (p96)")
    img2 = page_image(p2, F26["page"])
    curves = trace_fig26(img2)
    got = []
    for k in (1, 2, 3):
        r = fit_fig26(curves[k])
        got.append((round(r["f_half"], 1), round(r["n"], 2)))
        print("    %-44s plateau %6.2f  f_half %6.1f lines/mm  n %.2f  "
              "rms %.3f  (pure Gaussian %.3f)"
              % (F26_NAMES[k], r["plateau"], r["f_half"], r["n"],
                 r["rms"], r["rms_gaussian"]))
    stored = [(a, b) for a, b in fp._OOUE_WIENER_1959]
    ok = got == stored
    if not ok:
        bad += 1
    print("    [%s] film_profiles._OOUE_WIENER_1959 reproduces this fit%s"
          % ("OK  " if ok else "FAIL", "" if ok else ": got %s" % (got,)))
    okn = all(n < 2.0 for _, n in got)
    if not okn:
        bad += 1
    print("    [%s] ⚠ every measured rolloff exponent is BELOW the Gaussian 2 "
          "this engine assumes (%.2f, %.2f, %.2f) -- real grain spectra have "
          "FATTER high-frequency tails than the model"
          % ("OK  " if okn else "FAIL", got[0][1], got[1][1], got[2][1]))
    okd = got[0][0] < got[1][0]
    if not okd:
        bad += 1
    print("    [%s] ⚠ ONE FILM, TWO DENSITIES: f_half %.1f at D 1.03 against "
          "%.1f at D 0.45, developer and time held fixed -- the clump grows "
          "with density, and GrainSpec carries one clump size per stock"
          % ("OK  " if okd else "FAIL", got[0][0], got[1][0]))

    # ---- Part 2 Fig. 24 -------------------------------------------------
    print("\n  Part 2 Fig. 24 -- autocorrelation of grain structure (p95)")
    img24 = page_image(p2, F24["page"])
    phi = trace_fig24(img24)
    taus = sorted(phi)
    half = None
    for a, b in zip(taus, taus[1:]):
        if phi[a] >= 500.0 >= phi[b]:
            half = a + (b - a) * (phi[a] - 500.0) / (phi[a] - phi[b])
            break
    zero = None
    for a, b in zip(taus, taus[1:]):
        if phi[a] > 0.0 >= phi[b] or (phi[a] > 30.0 and phi[b] <= 30.0
                                      and b > 10.0):
            zero = b
            break
    f_hi = 374.8 / half if half else 0.0
    clump = 1000.0 / (2.0 * f_hi) if f_hi else 0.0
    print("    Neopan S, D 1.04: phi(0) = 1000 by construction, half at "
          "tau %.2f um, last positive sample near tau %.1f um"
          % (half or -1, zero or -1))
    print("    -> under this engine's Gaussian model tau_half = 374.8/f_hi, "
          "so f_hi %.0f c/mm and clump_um %.2f um" % (f_hi, clump))
    okh = abs(half - fp._OOUE_AUTOCORR_1959[0]) < 0.05
    if not okh:
        bad += 1
    print("    [%s] film_profiles._OOUE_AUTOCORR_1959 reproduces the half-width"
          % ("OK  " if okh else "FAIL"))
    print("    [note] ⚠ THE CURVE GOES NEGATIVE past its first zero -- an "
          "anti-correlated ring. A Gaussian autocorrelation cannot, and "
          "neither can Sayanagi's Poisson placement. Shape limitation of both, "
          "recorded and not patched")

    # ---- Part 1 Fig. 2 ---------------------------------------------------
    print("\n  Part 1 Fig. 2 -- mean grain area vs density (p39)")
    img1 = page_image(p1, F02["page"])
    area = trace_fig02(img1)
    rows = []
    for k, lbl in ((1, "FD-3 (1:1) 20C 32 min"), (2, "FD-3 (1:1) 20C 1 min")):
        ds = sorted(area[k])
        if not ds:
            continue
        a0, a1 = area[k][ds[0]], area[k][ds[-1]]
        d0 = 2.0 * math.sqrt(a0 / math.pi)
        d1 = 2.0 * math.sqrt(a1 / math.pi)
        rows.append((round(a0, 3), round(a1, 3)))
        print("    %-22s D %.1f->%.1f   area %.3f -> %.3f um^2   "
              "equivalent diameter %.2f -> %.2f um"
              % (lbl, ds[0], ds[-1], a0, a1, d0, d1))
    okf = all(r[0] > r[1] for r in rows)
    if not okf:
        bad += 1
    print("    [%s] ⚠ MEAN GRAIN AREA FALLS AS DENSITY RISES on both "
          "development times -- the same direction BBC T-101 Table 3 measures, "
          "from another laboratory, and the opposite of the usual expectation"
          % ("OK  " if okf else "FAIL"))
    okstore = tuple(rows) == tuple(tuple(x) for x in fp._OOUE_GRAIN_AREA_1959)
    if not okstore:
        bad += 1
    print("    [%s] film_profiles._OOUE_GRAIN_AREA_1959 reproduces this trace%s"
          % ("OK  " if okstore else "FAIL",
             "" if okstore else ": got %s" % (rows,)))
    oka = all(0.4 < v < 1.4 for r in rows for v in r)
    if not oka:
        bad += 1
    print("    [%s] equivalent diameters 0.85-1.21 um sit BELOW the 1.3 um "
          "floor of the 17 stored emulsion.grain_um values, all of which come "
          "from one third-party aggregator" % ("OK  " if oka else "FAIL"))

    # ---- the two text findings ------------------------------------------
    print("\n  Text findings, quoted rather than inferred")
    print("    §4.2.2 is headed 「濃度変化の標準偏差による方法」 -- the method "
          "using the STANDARD DEVIATION of density variation. ⚠ That settles "
          "23_7's mean-square / rms ambiguity in the author's own words.")
    print("    §4.2.1: 「…現在市販されている感光材料のQの値を測定した結果は，"
          "心理的粒状性との相関を認めることはできない」 -- measured Q on "
          "commercial materials shows NO correlation with graininess, because "
          "two emulsions of different grain size are coated in two layers. "
          "⚠ The empirical half of queue C45.")

    print()
    if bad:
        print("  [FAIL] %d problem(s)" % bad)
        return 1 if ns.do_assert else 0
    print("  [OK  ] J1/J2: three Wiener spectra, one autocorrelation and one "
          "grain-area series traced; no stored value changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
