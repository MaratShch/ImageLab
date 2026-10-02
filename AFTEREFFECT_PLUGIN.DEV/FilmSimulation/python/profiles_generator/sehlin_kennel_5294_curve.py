#!/usr/bin/env python3
"""EASTMAN 5294's green characteristic curve, off Fig. 14 of the paper that was
already in the corpus for something else.

Queue 463, 2026-09-26. R. Sehlin, G. Kennel et al., "Choosing between EASTMAN
Color Negative Films 5247 and 5294", *SMPTE Journal* **94**(7) 724-731, July
1985, page 7 (PDF page 7, printed page 730):

    PDF/PROFILES/KODAK/Sehlin_Kennel_etal_1983_ChoosingECN5247or52941.pdf

⚠⚠ THIS IS THE "READ ALL OF IT" LESSON FOR THE THIRD TIME. `sehlin_kennel_1985
.py` has been in the build since 2026-09-02. It opened this paper for Fig. 8
(granularity against density), examined Fig. 9, 11 and 12, refused two of them
with measured reasons -- and never looked at Fig. 14, which is a CHARACTERISTIC
CURVE for a stock whose curve this database was carrying as pure analogy. The
AGFA SCALA 200x row in `CURVE_MISSING` records the same failure a month
earlier, in the same words: when you open a sheet, read all of it.

WHAT THE FIGURE IS, AND WHAT IT IS NOT
--------------------------------------
Two traces, both green Status M density: a SOLID one labelled G_N (neutral
exposure) and a DASHED one labelled G_G (green-light exposure). Only the solid
one is a characteristic curve in this schema's sense -- G_G is a colour
saturation diagnostic, the response to a monochromatic green wedge, and reading
it as a tone curve would store the film's response to light it will never see.
The two coincide below D 1.2 and separate above it, which is the paper's point.

⚠ THE ABSCISSA CARRIES NO NUMBERS. It is labelled "Relative Log Exposure" and
its only metric mark is a `|<-.30->|` scale bar in the bottom right. So this
figure fixes the curve's SHAPE, its GAMMA and its D-min in absolute density,
and says nothing whatever about where it sits on the exposure axis. The stored
toe position is therefore NOT adopted from here; it is kept from the analogy
the profile already carried, and the adoption note says so. A trace that
silently moved toe_x would be inventing a speed.

⚠ AND THE ORDINATE IS NOT LINEAR IN PIXELS. The four density ticks measure
1677.0 / 2047.5 / 2365.0 / 2677.5 px for 3.0 / 2.0 / 1.0 / 0.0 -- spacings of
370.5, 317.5 and 312.5, decreasing monotonically, which is vertical keystone in
a 1985 journal scan and not noise. A straight linear fit through them misplaces
mid-scale by about 0.1 D, which is six times this project's usual raster
tolerance. The mapping is therefore a QUADRATIC through the four ticks, which
removes the keystone and leaves one degree of freedom as a residual to report.

WHAT IS ADOPTED
---------------
The GREEN record's `dmin`, `gamma` and knee softness, fitted to the solid
trace. Red and blue keep the analogy shape they had and keep their own D-min:
this figure prints one layer, and spreading one measured green curve across
three records would claim a crossover the document does not show. That split
is the same one `cinestill_cs2.py` makes for the same reason.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PDF = "PDF/PROFILES/KODAK/Sehlin_Kennel_etal_1983_ChoosingECN5247or52941.pdf"
PAGE = 6            # 0-based; printed page 730
DPI = 400

#: Frame, in 400-dpi page pixels, for each of the two panels this module
#: reads. Fig. 14 is 5294 -- the film being adopted -- and Fig. 13 is 5247,
#: which is read ONLY as a calibration check: see CALIBRATION below.
FRAME = dict(left=748, right=1970, top=1671, bottom=2677)
FRAME_13 = dict(left=748, right=1961, top=464, bottom=1366)
FRAME_TOL = 6

#: Density ticks as measured: (pixel row, density). ⚠ FIG. 14's ARE NOT
#: EQUALLY SPACED and Fig. 13's ARE -- 370.5 / 317.5 / 312.5 against 293.5 /
#: 301.5 / 301.0 on the same page, in the same artwork style. So the keystone
#: the quadratic removes is a property of the lower panel's plate and not of
#: the scan as a whole, which is why the correction is fitted per panel
#: instead of once for the page.
DENSITY_TICKS = ((1677.0, 3.0), (2047.5, 2.0), (2365.0, 1.0), (2677.5, 0.0))
DENSITY_TICKS_13 = ((481.0, 3.0), (774.5, 2.0), (1076.0, 1.0), (1377.0, 0.0))

#: Fig. 13's own scale bar, same construction as Fig. 14's.
SCALE_BAR_PX_13 = (1719.5, 1841.0)

#: The `|<-.30->|` scale bar's two end caps, in the same pixels, and what it
#: says. This is the ONLY metric information the abscissa carries.
SCALE_BAR_PX = (1719.5, 1841.0)
SCALE_BAR_DECADES = 0.30

#: Regions inside the frame that are lettering rather than curve, as
#: (x0, x1, y0, y1) in page pixels: the "5294 Film" caption, the legend block,
#: and the two curve labels at the right.
MASKS = (
    (940, 1130, 1930, 1990),      # "5294 Film"
    (1450, 1960, 2320, 2500),     # the legend block
    (1830, 1970, 1930, 2010),     # G_G
    (1830, 1970, 2060, 2140),     # G_N
)

#: ⚠ FIG. 13 IS NOT FULLY TRACED, AND THE REASON IS STRUCTURAL RATHER THAN
#: EFFORT. It carries THREE curves -- G_G over-corrected, G_N, and G_G
#: under-corrected -- so the "lowest branch is the solid one" rule that works
#: on Fig. 14 picks the UNDER-CORRECTED green-exposure trace instead of the
#: neutral one over most of the plot. A trace attempted with that rule
#: returned 66 discontinuities and an rms of 0.134 D, i.e. it was reading two
#: curves at once. Fig. 13 is used only for the one measurement below, which
#: needs no disambiguation at all because all three traces COINCIDE at the
#: left-hand edge.
CAL_PROBE_13 = dict(x=FRAME_13["left"] + 20, ymin=FRAME_13["top"] + 45,
                    ymax=FRAME_13["bottom"] - 45)

#: What the trace must return. Pinned so a re-run that disagrees FAILS.
#: ⚠ `dmin` IS PINNED BUT NOT ADOPTED -- see CALIBRATION. It is here so that a
#: change in the figure's own zero is still detected.
EXPECTED = dict(dmin=0.3592, gamma=0.8008, toe_k=0.3018, separation=2.2808,
                rms=0.0093, d_at_right=2.008, span_decades=2.95)
TOL = dict(dmin=0.02, gamma=0.03, toe_k=0.03, separation=0.06, rms=0.003,
           d_at_right=0.04, span_decades=0.04)

#: The calibration probe's two readings and the datasheet they are judged
#: against. ⚠ THE 5247 ROW IS THE ONE THAT DECIDES: its film's green D-min is
#: tier 1 from Kodak's own TI0835, and the paper plots that film BELOW it.
EXPECTED_CAL = {"5247 (Fig. 13)": 0.409, "5294 (Fig. 14)": 0.434}
CAL_TOL = 0.02
CAL_DATASHEET_DMIN_5247 = 0.531


def _page():
    import pymupdf
    doc = pymupdf.open(str(HERE / PDF))
    pix = doc[PAGE].get_pixmap(dpi=DPI)
    a = np.frombuffer(pix.samples, np.uint8).reshape(
        pix.height, pix.width, pix.n)[..., :3].astype(int)
    return a


def _density_map(ticks=DENSITY_TICKS):
    """Quadratic pixel-row -> density, plus its worst residual on the ticks."""
    y = np.array([t[0] for t in ticks], float)
    d = np.array([t[1] for t in ticks], float)
    c = np.polyfit(y, d, 2)
    resid = float(np.max(np.abs(np.polyval(c, y) - d)))
    lin = np.polyfit(y, d, 1)
    lin_resid = float(np.max(np.abs(np.polyval(lin, y) - d)))
    return (lambda v: np.polyval(c, v)), resid, lin_resid


def trace(verbose=True, frame=None, ticks=None, masks=None, bar=None,
          label="Fig. 14, 5294"):
    frame = frame or FRAME
    ticks = ticks or DENSITY_TICKS
    masks = masks if masks is not None else MASKS
    bar = bar or SCALE_BAR_PX
    a = _page()
    g = a.mean(2)
    dark = g < 110

    # -- the frame, re-derived ---------------------------------------------
    sub = dark[1599:2879, 400:3000]
    cs = sub.sum(0)
    v = [j + 400 for j in range(len(cs)) if cs[j] > 700]
    got_left, got_right = (min(v), max(v)) if v else (0, 0)
    # `max` picks the outer yellow box; the plot's right rule is the next one
    # down, which is why this takes the largest value BELOW the box edge.
    inner = [x for x in v if x < got_right - 40]
    got_right = max(inner) if inner else got_right

    to_d, resid, lin_resid = _density_map(ticks)
    px_per_decade = ((bar[1] - bar[0]) / SCALE_BAR_DECADES)

    if verbose:
        print("Sehlin & Kennel 1985 p730 -- %s" % label)
        print("  frame left %d right %d (pinned %d / %d)"
              % (got_left, got_right, frame["left"], frame["right"]))
        print("  density axis: quadratic through 4 ticks, worst residual "
              "%.4f D; a LINEAR fit would be %.4f D out -- %.1fx worse"
              % (resid, lin_resid, lin_resid / max(resid, 1e-9)))
        print("  abscissa: the .30 scale bar spans %.1f px, so %.1f px per "
              "decade; NO ORIGIN, so no speed is read from this figure"
              % (bar[1] - bar[0], px_per_decade))

    # -- the solid trace ----------------------------------------------------
    # ⚠ THE MARGINS ARE 14 px AND NOT 4, AND THE FIRST DRAFT PROVED WHY. The
    # frame's own rules are dark, span the full width, and sit at density 3.0
    # and 0.0; a 4 px margin left them inside the search window and the
    # "lowest branch" rule then followed the BOTTOM RULE across the whole
    # plot, returning a flat curve at D 0.017 and a fitted gamma of 0.10. The
    # failure was loud, which is what the pinned expectations are for.
    x0, x1 = frame["left"] + 14, frame["right"] - 14
    # ⚠ THE VERTICAL MARGIN IS 45 px AND THE AXIS TICKS ARE WHY. Both rules
    # carry inward tick marks about 25 px long, and a tick is the lowest dark
    # thing in its column, so the "lowest branch" rule below followed a tick
    # to D 0.05 in twenty-three columns -- a 0.2 D rms floor that no choice of
    # softplus parameters could get under, and which looked like a bad fit
    # rather than bad data until the residual was plotted. 45 px clears the
    # ticks and still leaves 100 px of headroom under the traced curve's own
    # minimum (D 0.43, row 2530, against the cut at row 2632).
    y0, y1 = frame["top"] + 45, frame["bottom"] - 45
    pts = []
    for x in range(x0, x1):
        if any(mx0 <= x <= mx1 for mx0, mx1, _a, _b in masks):
            col_masked = [(my0, my1) for mx0, mx1, my0, my1 in masks
                          if mx0 <= x <= mx1]
        else:
            col_masked = []
        ys = [y for y in range(y0, y1) if dark[y, x]
              and not any(m0 <= y <= m1 for m0, m1 in col_masked)]
        if not ys:
            continue
        # cluster
        groups, cur = [], [ys[0]]
        for yy in ys[1:]:
            if yy - cur[-1] <= 4:
                cur.append(yy)
            else:
                groups.append(cur)
                cur = [yy]
        groups.append(cur)
        # ⚠ THE SOLID CURVE IS THE LOWER-DENSITY BRANCH WHERE TWO EXIST.
        # G_G (green exposure, dashed) runs ABOVE G_N on the plot, i.e. at a
        # SMALLER pixel row. Taking the largest row therefore takes G_N, and
        # where the two coincide there is only one group and the rule is a
        # no-op.
        pts.append((x, float(np.mean(groups[-1]))))

    if len(pts) < 400:
        raise SystemExit("[!] only %d columns traced; the frame or the masks "
                         "are wrong" % len(pts))

    px = np.array([p[0] for p in pts], float)
    py = np.array([p[1] for p in pts], float)
    dd = to_d(py)
    xx = (px - frame["left"]) / px_per_decade
    return xx, dd, dict(px_per_decade=px_per_decade, resid=resid,
                        lin_resid=lin_resid, n=len(pts),
                        left=got_left, right=got_right)


def calibration_probe():
    """The paper's own density level, against a datasheet on the same page.

    ⚠⚠ THIS IS THE MEASUREMENT THAT DECIDES WHAT IS TRANSFERABLE, and it is
    the reason this module adopts a SHAPE and not a density. Fig. 13 plots
    5247 and Fig. 14 plots 5294, side by side in one paper, in one ordinate
    labelled "Green Density (Status M)". EASTMAN_5247_1983's green record is
    tier 1, traced from Kodak's own TI0835 sheet, so the two can be compared
    on that film -- and they disagree in a direction that cannot be a
    disagreement about the FILM.

    The paper's leftmost plotted density for 5247 is BELOW that film's
    datasheet D-min. A characteristic curve cannot go under its own base plus
    fog, so the two ordinates are not the same quantity: the paper has
    normalised, or is plotting above some reference, or is reading a
    different densitometry than the sheet. Which of those it is cannot be
    settled from the paper, and does not need to be -- what follows is that
    its ABSOLUTE LEVEL must not be carried into this database, while its
    SHAPE is unaffected by any constant offset.
    """
    a = _page()
    dark = a.mean(2) < 110
    to_d13, _r, _l = _density_map(DENSITY_TICKS_13)
    to_d14, _r2, _l2 = _density_map(DENSITY_TICKS)
    out = {}
    for tag, x, frame, to_d in (
            ("5247 (Fig. 13)", CAL_PROBE_13["x"], FRAME_13, to_d13),
            ("5294 (Fig. 14)", FRAME["left"] + 20, FRAME, to_d14)):
        ys = [y for y in range(frame["top"] + 45, frame["bottom"] - 45)
              if dark[y, x]]
        if not ys:
            out[tag] = float("nan")
            continue
        # ⚠ THE LOWEST CLUSTER, NOT THE MEAN OF THE COLUMN. A plain mean
        # folds in the left rule's tick marks and reads 0.89 where the curve
        # is at 0.43 -- which is the same class of error the 45 px margin in
        # `trace` exists to stop, met again in a two-line helper.
        groups, cur = [], [ys[0]]
        for yy in ys[1:]:
            if yy - cur[-1] <= 4:
                cur.append(yy)
            else:
                groups.append(cur)
                cur = [yy]
        groups.append(cur)
        out[tag] = float(to_d(np.mean(groups[-1])))
    return out


def fit(xx, dd, shoulder_k):
    """Fit the curve, then throw the abscissa placement away.

    ⚠⚠ toe_x IS FITTED AND MUST NOT BE ADOPTED, and the distinction is the
    whole discipline of this trace. The figure has no exposure origin, so the
    optimiser needs a free horizontal offset or it cannot fit at all -- a
    first draft that held toe_x at the profile's analogy value converged at
    rms 0.2013 D, which is not a bad fit but a fit of the wrong thing. What
    that free parameter absorbs is the unknown origin, and in this database
    toe_x is read as a SPEED. So it is fitted, reported, and discarded.
    ⚠ THE SEPARATION shoulder_x - toe_x IS NOT DISCARDED. That is a property
    of the curve rather than of its placement, and the figure does show it, so
    it is fitted and adopted with the shoulder carried along under whatever
    toe_x the profile keeps.
    """
    from scipy.optimize import least_squares

    def sp(z, k):
        return k * np.log1p(np.exp(np.clip(z / k, -60, 60)))

    def model(p, x):
        dmin, gamma, tk, tx, sep = p
        return dmin + gamma * (sp(x - tx, tk) - sp(x - (tx + sep),
                                                   shoulder_k))

    r = least_squares(lambda p: model(p, xx) - dd,
                      [0.4, 0.7, 0.4, float(xx[0]) + 0.3, 3.0],
                      bounds=([0.0, 0.1, 0.05, float(xx[0]) - 3.0, 1.0],
                              [1.5, 3.0, 1.5, float(xx[-1]), 8.0]))
    rms = float(np.sqrt(np.mean((model(r.x, xx) - dd) ** 2)))
    return tuple(float(v) for v in r.x), rms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)

    if not (HERE / PDF).is_file():
        print("[SKIP] sehlin_kennel_5294_curve.py -- source not present")
        return 0

    import film_profiles as fp
    prof = fp.get_profile("EASTMAN_5294_1983")
    base = prof.curves.g

    xx, dd, info = trace()
    # The traced abscissa is relative to the frame's left edge. Anchor it on
    # the profile's own toe so the fit is about density, not about speed.
    (dmin, gamma, toe_k, toe_x_fit, sep), rms = fit(xx, dd, base.shoulder_k)

    bad = 0
    print("  %d columns traced over %.2f decades" % (info["n"], xx[-1] - xx[0]))
    print("  fitted  dmin %.4f  gamma %.4f  toe_k %.4f  separation %.4f "
          "decade   rms %.4f D" % (dmin, gamma, toe_k, sep, rms))
    print("  fitted toe_x %.4f -- DISCARDED, the figure has no origin"
          % toe_x_fit)
    print("  stored  dmin %.4f  gamma %.4f  toe_k %.4f  separation %.4f  "
          "(analogy, before this)"
          % (base.dmin, base.gamma, base.toe_k,
             base.shoulder_x - base.toe_x))
    print("  density at the right-hand edge %.3f" % dd[-1])

    got = dict(dmin=dmin, gamma=gamma, toe_k=toe_k, rms=rms,
               separation=float(sep), d_at_right=float(dd[-1]),
               span_decades=float(xx[-1] - xx[0]))
    for k, want in EXPECTED.items():
        if abs(got[k] - want) > TOL[k]:
            print("  [FAIL] %s = %.4f, pinned %.4f +- %.4f"
                  % (k, got[k], want, TOL[k]))
            bad += 1

    # ---- the calibration, and what it licenses ---------------------------
    cal = calibration_probe()
    print("  CALIBRATION -- the paper's own density level:")
    for tag, v in cal.items():
        print("    %-16s left-edge D %.3f" % (tag, v))
    print("    against EASTMAN_5247_1983's tier-1 green D-min %.3f, traced "
          "from Kodak's own TI0835" % CAL_DATASHEET_DMIN_5247)
    for tag, want in EXPECTED_CAL.items():
        if abs(cal.get(tag, float("nan")) - want) > CAL_TOL:
            print("  [FAIL] calibration probe %s reads %.3f, pinned %.3f"
                  % (tag, cal.get(tag, float("nan")), want))
            bad += 1
    _gap = CAL_DATASHEET_DMIN_5247 - cal["5247 (Fig. 13)"]
    print("    the paper plots 5247 %.3f D BELOW its own datasheet D-min, so "
          "the two ordinates are not the same quantity and the LEVEL is not "
          "transferable" % _gap)
    if _gap < 0.05:
        print("  [FAIL] the paper no longer sits below the datasheet D-min; "
              "the argument for adopting shape-only has gone and the "
              "adoption must be re-decided")
        bad += 1

    # ⚠ THE ADOPTED VALUES MUST BE LIVE, AND THE UNADOPTED ONE MUST NOT BE.
    # This is what makes the module an audit of the database rather than a
    # notebook about a figure.
    for k, v in (("gamma", gamma), ("toe_k", toe_k)):
        if abs(getattr(base, k) - v) > TOL[k]:
            print("  [FAIL] EASTMAN_5294_1983.curves.g.%s is %.4f; this "
                  "trace returns %.4f" % (k, getattr(base, k), v))
            bad += 1
    _sep_live = base.shoulder_x - base.toe_x
    if abs(_sep_live - sep) > TOL["separation"]:
        print("  [FAIL] the stored toe-to-shoulder separation is %.4f; this "
              "trace returns %.4f" % (_sep_live, sep))
        bad += 1
    if abs(base.dmin - dmin) < 0.05:
        print("  [FAIL] EASTMAN_5294_1983's green D-min has been set to the "
              "figure's own %.4f. The calibration above says that level is "
              "not this database's; D-min must stay at the analogy value."
              % dmin)
        bad += 1

    # ⚠ RED AND BLUE MUST NOT HAVE BEEN GIVEN THE GREEN MEASUREMENT. The
    # figure prints one layer; three records carrying one measured gamma
    # would assert a crossover nobody drew.
    if (abs(prof.curves.r.gamma - gamma) < 1e-9
            and abs(prof.curves.b.gamma - gamma) < 1e-9):
        print("  [FAIL] red and blue carry the green record's fitted gamma "
              "exactly; this figure documents ONE layer")
        bad += 1

    if bad:
        print("\n[FAIL] the 5294 green curve does not reproduce")
        return 1 if ns.assert_ else 0
    print("\n[OK] Fig. 14's solid G_N trace re-read at rms %.4f D, the "
          "keystoned density axis re-fitted, the paper's zero re-measured "
          "against a tier-1 datasheet on the facing figure, and "
          "EASTMAN_5294_1983's green record confirmed to carry the SHAPE and "
          "not the level: no D-min, no speed, red and blue still analogy."
          % rms)
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
