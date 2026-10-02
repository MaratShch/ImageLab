"""Sehlin, Kennel et al. 1985 -- the granularity-versus-density pairing E5 asked
for, on a stock this database holds.

Queue item E5, 2026-09-02.  R. Sehlin, G. Kennel et al., "Choosing between
EASTMAN Color Negative Films 5247 and 5294", *SMPTE Journal* **94**(7) 724-731,
July 1985:

    PDF/PROFILES/KODAK/Sehlin_Kennel_etal_1983_ChoosingECN5247or52941.pdf

⚠ THE FILE NAME IS WRONG AND THE QUEUE ALREADY RECORDS IT: the paper is July
**1985**, not 1983.  Eleven pages, no text layer, one ~285 ppi raster per page.

WHY THIS DOCUMENT AND NOT ANOTHER
------------------------------------
`NotFound.md` row 2 has been asking for a granularity-against-density curve for
a named stock since 2026-08-17, and every one this corpus has found since has
been a Kodak *vendor sheet* for a VISION-family stock.  This is a JOURNAL paper,
by Kodak's own engineers, on two ECN stocks of 1983 that are both already in
this database -- `EASTMAN_5247_1983` and `EASTMAN_5294_1983` -- and its Fig. 8
plots DENSITY and RMS GRANULARITY against ONE shared log-exposure abscissa.
That is the σ(D) construction, on one plate, with no second document needed.

WHAT IS ADOPTED: ⚠ NOTHING, AND THAT IS THE RESULT
------------------------------------------------------
The σ(D) SHAPE was traced, validated, STORED, and then WITHDRAWN when
`cpp_parity.py` rejected it -- see "the density space" below.  What follows
was written when it was going to be adopted, and is kept because the reasoning
is what makes the refusal legible.  ⚠ The ordinate is
labelled "RMS Granularity" with **no unit, no aperture and no densitometry**, so
its LEVEL cannot be reconciled with this corpus's 48 µm diffuse-RMS convention
and is not stored -- the same split the project made for KODAK_TRI_X_REVERSAL_200
on 2026-08-25b ("shape and level are separate adoptions").

⚠ AND THE SHAPE VALIDATES ITSELF AGAINST A FAMILY IT WAS NOT FITTED TO.  Read
through this schema's own anchors it lands inside the range of the eleven Kodak
sheets already measured -- peak 1.20-1.62x at D 0.65-0.80, dmax 0.50-0.90 -- on
a plate none of them came from, drawn by different people for a different
purpose.  That is the check; the adoption is what the check licenses.

WHAT IS REPORTED AND REFUSED
-------------------------------
Fig. 12 is an MTF for 5247 from 1 to 100 c/mm at five exposures, and this
corpus stores an **estimated** f50 triple (24/28/33 c/mm) for that stock.  The
figure's 50 % crossings are 45-58 c/mm -- 1.7 to 2.1x the estimate.  ⚠ IT IS NOT
ADOPTED, for two reasons that are about the document and not about the
disagreement.  First, the running text calls it "the **system** modulation
transfer function", and a system MTF carries the printer and the lens; the panel
label says "5247 Film" and the two statements are not the same claim.  Second,
and independent of the wording: this curve **does not overshoot**.  Every
colour-negative MTF this project has traced from a vendor sheet rises above
100 % before it falls, because adjacency development does that; a curve pinned
at 100 % up to 10 c/mm has had that removed, normalised out, or never had it.
A f50 read off it would be a different quantity from the one `MTFSpec` stores.
Reported with the numbers, refused with the cause -- the same disposition
`agfa_1998_curves.py` takes on queue G6.

Fig. 11 (granularity against exposure, both films, three exposures each) and
Fig. 9 (five exposures on 5294) are read for their DIRECTION only, which is the
paper's own headline: overexposing decreases granularity.  That sentence is
already cited in `GrainSpec`'s docstring from this very paper; what is new is
that it is now traced rather than quoted.

⚠ THE DENSITY SPACE, WHICH IS WHY NOTHING IS STORED
------------------------------------------------------
`GrainSpec.sigma_anchors` reads its anchor densities as PER-LAYER ANALYTICAL
densities, and the corpus proves it: on every measured stock
`sigma_shape_toe_at` sits at that stock's own GREEN curve dmin -- 5219 0.59
against 0.58, 5201 0.62 against 0.62, 5245 0.57 against 0.64.  Fig. 8's
abscissa is scene placement and its ordinate is the film's plotted density, and
the traced toe at D 0.44 falls BELOW 5294's green dmin of 0.68 and far below
its blue dmin of 1.09.  Stored anyway, the whole shape sat under the layer's
dmin and `cpp_parity.py` rejected it at 5.7e-01 against a 2e-05 tolerance --
the guard catching the mistake before it shipped, which is what it is for.

Re-anchoring would need a correspondence between Fig. 8's density axis and this
film's per-layer curves, and the paper prints none.  SHAPE AND SPACE ARE AS
SEPARATE AS SHAPE AND LEVEL, and this document now carries three refusals: its
granularity LEVEL (no unit), its σ(D) SPACE (not per-layer), and Fig. 12's f50
(a system MTF that does not overshoot).

Usage:
    python3 sehlin_kennel_1985.py --root . [--assert]
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PDF_REL = os.path.join("PDF", "PROFILES", "KODAK",
                       "Sehlin_Kennel_etal_1983_ChoosingECN5247or52941.pdf")

#: 595 pt wide against 2350 px is 3.95 px/pt.  Clips are in POINTS so the
#: geometry survives a different rasteriser.
PX_PER_PT = 3.95

# --- Fig. 8, page 5 of the PDF (journal p728).  Frame x 165 / 998, y 41 / 841.
#     Left ordinate DENSITY: labels 3.0/2.0/1.0/0.0 centred at y 85.5/299.5/
#     523.5/739.0.  Right ordinate RMS GRANULARITY, log: 10.0 at y 316.5 and
#     1.0 at y 558.5.  ⚠ THE TWO ORDINATES ARE NOT THE SAME SCALE -- 218 px per
#     density unit against 242 px per granularity decade -- which is why each is
#     calibrated on its own labels and never on the other's gridline.
#     The printed |<-.30->| bar spans x 746-830, i.e. 0.30 log E in 84 px.
F08 = dict(page=4, clip=(85, 125, 400, 350),
           x0=170, x1=995, y0=45, y1=838,
           d_zero_y=739.0, px_per_d=218.0,
           g_one_y=558.5, px_per_decade=242.0,
           bar_px=84.0, bar_loge=0.30,
           seed_x=835, seed_density=343.0, seed_grain=542.0)

# --- Fig. 12, page 6 (journal p729).  Log-log.  Frame x 218.5 / 1101,
#     y 87 / 1045.  Abscissa ticks 1/5/10/20/50/100 c/mm at x 218.5/520.5/656.0/
#     785.5/958.0/1101.0; ordinate ticks 100/50/20/10/2 % at y 156.5/287.0/
#     463.0/594.0/903.0.
F12 = dict(page=5, clip=(235, 445, 530, 730),
           x_ticks=(218.5, 520.5, 656.0, 785.5, 958.0, 1101.0),
           x_vals=(1.0, 5.0, 10.0, 20.0, 50.0, 100.0),
           y_ticks=(156.5, 287.0, 463.0, 594.0, 903.0),
           y_vals=(100.0, 50.0, 20.0, 10.0, 2.0),
           xlo=224, xhi=1099)

#: The anchor densities this schema uses, and the density above which the
#: characteristic curve is too flat for the pairing to be conditioned.  The
#: 7266 trace discarded 22 of 52 points on exactly this rule.
MIN_SLOPE_D_PER_LOGE = 0.25


def page_gray(doc, spec) -> np.ndarray:
    import pymupdf
    p = doc[spec["page"]].get_pixmap(
        matrix=pymupdf.Matrix(PX_PER_PT, PX_PER_PT),
        clip=pymupdf.Rect(*spec["clip"]),
        colorspace=pymupdf.csGRAY)
    return np.frombuffer(p.samples, dtype=np.uint8).reshape(p.height, p.width)


def _runs(ink, x, lo, hi, max_len=26):
    out, s = [], None
    for y in range(lo, hi):
        if ink[y, x]:
            if s is None:
                s = y
        elif s is not None:
            if y - s <= max_len:
                out.append((s + y - 1) / 2.0)
            s = None
    if s is not None and hi - 1 - s <= max_len:
        out.append((s + hi - 1) / 2.0)
    return out


def _joint_walk(ink, g, x0, ya, yb, step):
    """Follow BOTH curves at once, assigning candidates to tracks jointly.

    ⚠ THE TWO CURVES CROSS, near D = 1.0, and no single-track follower survives
    that: within a few pixels of the crossing the nearest ink IS the other
    curve, and a tolerance tight enough to refuse it also refuses the right
    one.  What separates them is that they cross with OPPOSITE slopes, so the
    two tracks are followed together and the pair of candidates is assigned to
    the pair of predictions by whichever of the two pairings costs less.  The
    three dashed vertical arrows (Black / Gray / White) are dropped by the
    run-length cap in `_runs`.
    """
    A, B = {x0: ya}, {x0: yb}
    sa = sb = 0.0
    y_a, y_b = ya, yb
    miss = 0
    x = x0 + step
    while g["x0"] <= x < g["x1"]:
        cand = _runs(ink, x, g["y0"], g["y1"])
        pa = y_a + sa * step * (1 + miss)
        pb = y_b + sb * step * (1 + miss)
        tol = min(40.0, 6.0 + 2.5 * max(abs(sa), abs(sb))) * (1 + 0.6 * miss)
        best, cost = None, None
        for i, ca in enumerate(cand):
            for j, cb in enumerate(cand):
                if i == j:
                    continue
                c = abs(ca - pa) + abs(cb - pb)
                if abs(ca - pa) > tol or abs(cb - pb) > tol:
                    continue
                if cost is None or c < cost:
                    best, cost = (ca, cb), c
        if best is None:
            miss += 1
            if miss > 20:
                break
            x += step
            continue
        na, nb = best
        sa = 0.6 * sa + 0.4 * ((na - y_a) / (step * (1 + miss)))
        sb = 0.6 * sb + 0.4 * ((nb - y_b) / (step * (1 + miss)))
        y_a, y_b = na, nb
        A[x], B[x] = na, nb
        miss = 0
        x += step
    return A, B


def trace_fig08(img):
    ink = img < 150
    g = F08
    dl, gl = _joint_walk(ink, g, g["seed_x"], g["seed_density"],
                         g["seed_grain"], -1)
    dr, gr = _joint_walk(ink, g, g["seed_x"], g["seed_density"],
                         g["seed_grain"], +1)
    dens = dict(dl)
    dens.update(dr)
    grain = dict(gl)
    grain.update(gr)
    return dens, grain


def pair_fig08(dens, gr):
    """(log E, D, G) on the columns both curves reach, plus the local slope."""
    g = F08
    xs = sorted(set(dens) & set(gr))
    out = []
    for x in xs:
        d = (g["d_zero_y"] - dens[x]) / g["px_per_d"]
        gg = 10.0 ** ((g["g_one_y"] - gr[x]) / g["px_per_decade"])
        le = x * g["bar_loge"] / g["bar_px"]
        out.append((le, d, gg))
    # local dD/dlogE by central difference over a +/- 0.05 logE window
    keep = []
    for i, (le, d, gg) in enumerate(out):
        j = max(0, i - 12)
        k = min(len(out) - 1, i + 12)
        if out[k][0] <= out[j][0]:
            continue
        slope = (out[k][1] - out[j][1]) / (out[k][0] - out[j][0])
        keep.append((le, d, gg, slope))
    return keep


def fig08_anchors(pairs):
    """σ(D) in this schema's own anchor form, normalised at D = 1.0.

    ⚠ Points where the characteristic curve is flat are DISCARDED, not
    averaged in.  Where |dD/dlogE| is small one density maps to many σ and the
    pairing is ill-conditioned -- the rule `GrainSpec`'s docstring states and
    the 7266 trace applied.
    """
    ok = [(d, gg) for _, d, gg, s in pairs if s >= MIN_SLOPE_D_PER_LOGE]
    if len(ok) < 20:
        return None
    ok.sort()
    ds = np.array([p[0] for p in ok])
    gs = np.array([p[1] for p in ok])

    def at(dv):
        return float(np.interp(dv, ds, gs))

    mid = at(1.0)
    i = int(np.argmax(gs))
    return dict(d_lo=float(ds[0]), d_hi=float(ds[-1]),
                toe_at=round(float(ds[0]), 2), toe=at(ds[0]) / mid,
                dmax_at=round(float(ds[-1]), 2), dmax=at(ds[-1]) / mid,
                peak=float(gs[i]) / mid, peak_at=float(ds[i]),
                g_mid=mid, n=len(ok), n_all=len(pairs))


def fig12_crossings(img):
    ink = img < 150
    g = F12
    ax, bx = np.polyfit(np.log10(g["x_vals"]), g["x_ticks"], 1)
    ay, by = np.polyfit(np.log10(g["y_vals"]), g["y_ticks"], 1)
    out = {}
    for mtf in (90.0, 70.0, 50.0, 30.0):
        y = int(round(ay * math.log10(mtf) + by))
        xs, s = [], None
        for x in range(g["xlo"], g["xhi"]):
            if ink[y, x]:
                if s is None:
                    s = x
            elif s is not None:
                xs.append((s + x - 1) / 2.0)
                s = None
        out[mtf] = sorted(10.0 ** ((v - bx) / ax) for v in xs
                          if 10.0 ** ((v - bx) / ax) > 2.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        print("[!] pymupdf not installed:  pip install pymupdf")
        return 1
    import film_profiles as fp

    pdf = os.path.join(ns.root, PDF_REL)
    if not os.path.exists(pdf):
        print("[!] missing %s" % pdf)
        return 1
    import pymupdf
    doc = pymupdf.open(pdf)
    bad = 0

    print("Sehlin, Kennel et al., SMPTE Journal 94(7) 724-731, JULY 1985 --")
    print("  \"Choosing between EASTMAN Color Negative Films 5247 and 5294\"")
    print("  %s" % PDF_REL)
    print("  ⚠ the file name says 1983; the paper is July 1985")

    # ---- Fig. 8 ---------------------------------------------------------
    print("\n  Fig. 8 (p728) -- density AND RMS granularity on ONE log-E "
          "abscissa, 5294")
    img = page_gray(doc, F08)
    dens, gr = trace_fig08(img)
    pairs = pair_fig08(dens, gr)
    a = fig08_anchors(pairs)
    if a is None:
        print("    [FAIL] too few conditioned points")
        return 1
    print("    density   traced %d columns, D %.2f -> %.2f"
          % (len(dens), (F08["d_zero_y"] - dens[max(dens)]) / F08["px_per_d"],
             (F08["d_zero_y"] - dens[min(dens)]) / F08["px_per_d"]))
    print("    grain     traced %d columns, %d of %d pairs kept after the "
          "|dD/dlogE| >= %.2f gate" % (len(gr), a["n"], a["n_all"],
                                       MIN_SLOPE_D_PER_LOGE))
    print("    σ(D) over D %.2f-%.2f: toe %.3f @ %.2f / mid 1.000 / "
          "dmax %.3f @ %.2f / peak %.3f @ %.2f"
          % (a["d_lo"], a["d_hi"], a["toe"], a["toe_at"], a["dmax"],
             a["dmax_at"], a["peak"], a["peak_at"]))

    p94 = fp._BY_NAME["EASTMAN_5294_1983"]
    g94 = p94.grain
    # ⚠ THE ADOPTION WAS MADE AND THEN WITHDRAWN, AND THIS IS THE CHECK THAT
    # KEEPS IT WITHDRAWN. On every measured stock `sigma_shape_toe_at` sits at
    # the GREEN curve's own dmin, i.e. the anchors are PER-LAYER ANALYTICAL
    # densities. Fig. 8's abscissa is scene placement and its ordinate the
    # film's plotted density: the traced toe at D 0.44 falls below this stock's
    # green dmin and far below its blue one, so the whole shape would sit under
    # the layer's dmin. cpp_parity.py caught it at 5.7e-01 against a 2e-05
    # tolerance the first time it was stored.
    okwith = (not g94.sigma_shape_measured
              and a["toe_at"] < p94.curves.g.dmin)
    if not okwith:
        bad += 1
    print("    [%s] ⚠ WITHDRAWN, NOT STORED -- the anchors are in the wrong "
          "DENSITY SPACE: traced toe at D %.2f against this stock's green dmin "
          "%.2f and blue dmin %.2f, where every measured stock's toe_at IS its "
          "green dmin (5219 0.59 vs %.2f, 5201 0.62 vs %.2f)"
          % ("OK  " if okwith else "FAIL", a["toe_at"], p94.curves.g.dmin,
             p94.curves.b.dmin,
             fp._BY_NAME["KODAK_VISION3_500T_5219"].curves.g.dmin,
             fp._BY_NAME["KODAK_VISION2_50D_5201"].curves.g.dmin))

    kod = [(p.name, p.grain) for p in fp.FILM_PROFILES
           if p.grain.sigma_shape_measured and not p.is_reversal
           and p.grain.sigma_shape_peak > 0.0]
    pk = [g.sigma_shape_peak for _, g in kod]
    pka = [g.sigma_shape_peak_at for _, g in kod]
    dmx = [g.sigma_shape_dmax for _, g in kod]
    okfam = (min(pk) - 0.05 <= a["peak"] <= max(pk) + 0.05
             and min(dmx) - 0.05 <= a["dmax"] <= max(dmx) + 0.05)
    if not okfam:
        bad += 1
    print("    [%s] ⚠ AND IT VALIDATES ITSELF AGAINST A FAMILY IT WAS NOT "
          "FITTED TO: peak %.2fx vs the %d vendor-sheet negatives' %.2f-%.2f, "
          "dmax %.2f vs their %.2f-%.2f. A journal plate, drawn by different "
          "people for a different purpose, lands inside the sheet family"
          % ("OK  " if okfam else "FAIL", a["peak"], len(kod), min(pk),
             max(pk), a["dmax"], min(dmx), max(dmx)))
    okearly = a["peak_at"] < min(pka)
    print("    [%s] ⚠ its peak sits at D %.2f, EARLIER than all %d of them "
          "(%.2f-%.2f) -- the interior maximum moves with the stock and is not "
          "a constant of the class"
          % ("note" if okearly else "note", a["peak_at"], len(pka), min(pka),
             max(pka)))
    print("    [note] ⚠ THE LEVEL IS NOT ADOPTED. The ordinate is labelled "
          "\"RMS Granularity\" with no unit, no aperture and no densitometry, "
          "so it cannot be reconciled with this corpus's 48 um diffuse-RMS "
          "convention. Shape and level are separate adoptions (2026-08-25b); "
          "rms_granularity is untouched at %.1f. ⚠ AND THE SHAPE IS REFUSED "
          "TOO, on a SECOND ground found when it was stored and cpp_parity "
          "rejected it: shape and SPACE are as separate as shape and level"
          % g94.rms_granularity)

    # ---- Fig. 12 --------------------------------------------------------
    print("\n  Fig. 12 (p729) -- MTF vs exposure, 5247")
    img12 = page_gray(doc, F12)
    cr = fig12_crossings(img12)
    for m in (90.0, 70.0, 50.0, 30.0):
        print("    MTF %2.0f %% crossed at %s c/mm"
              % (m, ", ".join("%.1f" % v for v in cr[m]) or "-"))
    p47 = fp._BY_NAME["EASTMAN_5247_1983"]
    f50 = cr[50.0]
    ok12 = len(f50) >= 3 and min(f50) > 1.5 * p47.mtf.f50_g
    if not ok12:
        bad += 1
    print("    [%s] ⚠ THE 50 %% CROSSINGS ARE %.0f-%.0f c/mm AGAINST A STORED "
          "ESTIMATE OF %.0f (green) -- %.1fx to %.1fx"
          % ("OK  " if ok12 else "FAIL", min(f50), max(f50), p47.mtf.f50_g,
             min(f50) / p47.mtf.f50_g, max(f50) / p47.mtf.f50_g))
    ok_no_ovs = len(cr[90.0]) > 0 and min(cr[90.0]) > 10.0
    if not ok_no_ovs:
        bad += 1
    print("    [%s] ⚠ AND IT DOES NOT OVERSHOOT: 90 %% is not left until "
          "%.0f c/mm, where every colour-negative MTF traced from a vendor "
          "sheet in this corpus rises ABOVE 100 %% first, because adjacency "
          "development does that"
          % ("OK  " if ok_no_ovs else "FAIL", min(cr[90.0])))
    okref = not p47.mtf.mtf_measured
    if not okref:
        bad += 1
    print("    [%s] ⚠ NOT ADOPTED, AND EASTMAN_5247_1983.mtf_measured IS STILL "
          "FALSE. Two reasons about the DOCUMENT, not about the disagreement: "
          "the running text calls this \"the SYSTEM modulation transfer "
          "function\", which carries the printer and the lens, while the panel "
          "says \"5247 Film\" -- not the same claim; and a curve pinned at "
          "100 %% to 10 c/mm has had the overshoot removed, normalised out, or "
          "never had it, so its f50 is a different quantity from the one "
          "MTFSpec stores. Same disposition as G6" % ("OK  " if okref else "FAIL"))

    # ---- the paper's own sentence ---------------------------------------
    print("\n  The sentence GrainSpec's docstring already cites from this paper")
    rise, fall = a["toe"], a["dmax"]
    okdir = fall < rise
    if not okdir:
        bad += 1
    print("    [%s] \"overexposing either film significantly decreases "
          "granularity\" -- traced, not quoted: granularity falls to %.2f of "
          "its D = 1.0 value by D %.2f, from %.2f at D %.2f"
          % ("OK  " if okdir else "FAIL", fall, a["dmax_at"], rise,
             a["toe_at"]))
    print("    [note] Figs. 9 and 11 repeat it at five and three exposures on "
          "both films; they are read for DIRECTION only, because they share "
          "Fig. 8's unlabelled ordinate and add no density axis")

    print()
    if bad:
        print("  [FAIL] %d problem(s)" % bad)
        return 1 if ns.do_assert else 0
    print("  [OK  ] E5: a clean σ(D) shape traced from a journal plate and "
          "validated against the vendor-sheet family, then WITHDRAWN on a "
          "density-space mismatch; its level and Fig. 12's f50 refused too")
    return 0


if __name__ == "__main__":
    sys.exit(main())
