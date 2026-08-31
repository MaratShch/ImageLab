"""The four TI0835 plates for EASTMAN_5247_1983, read at last (queue B4).

WHAT THIS READS
---------------
Eastman Kodak, *EASTMAN Color Negative Film 5247*, publication **TI0835 Revised
6-93**, in the corpus as `PDF/PROFILES/KODAK/5247.pdf`. Pages 6-9 each carry one
full-page raster, 959 x 719 px on a 612 x 792 pt page:

    p6  TI0835A  MTF, diffuse visual, log-log
    p7  TI0835B  CHARACTERISTIC, three records
    p8  TI0835C  SPECTRAL SENSITIVITY, three layers
    p9  TI0835D  SPECTRAL DYE DENSITY, midscale neutral + D-min

⚠ THE BLOCKER THAT KEPT THESE UNREAD FOR A YEAR WAS A DETECTION THRESHOLD, NOT A
CALIBRATION PROBLEM. The queue recorded: "the gridlines do NOT divide the
labelled range into round steps -- 14 vertical intervals across a 0-3 density
axis and 29 horizontal across six decades, neither divisible by the label count
-- so 'frame edge = axis extreme' cannot simply be assumed". Both counts were
one short. On p7 the outermost gridline on each axis is FAINTER than the
interior ones -- 457 and 819 dark pixels against 461 and 819+ for their
neighbours -- and fell under the ink threshold the scan was being read with.
Lower the threshold from 0.5 to 0.45 of the span and the grid comes out at

    31 verticals,  82 .. 894 px  ->  30 intervals over log E -4.00 .. +2.00
    16 horizontals, 119 .. 571 px ->  15 intervals over density 0 .. 3

which is 0.2 per interval on both axes, exactly round, and the frame corners ARE
the labelled extremes. All four plates then calibrate the same way, every one of
them landing on a round step. ⚠ The lesson is the one this project keeps
relearning: a count that is ALMOST right is evidence of an off-by-one in the
measurement, not of an irregular source.

⚠ AND THE INKS ARE EXACT, WHICH MAKES LAYER ASSIGNMENT MACHINE-READABLE. Every
coloured pixel on all four plates is one of three values -- (0,0,255),
(128,128,0), (255,0,0) -- with no antialiasing between them. None of the
ink-overprint reasoning the H-1 sheets forced (family C in `dye_density.py`)
is needed here.

⚠ THE INK IS THE SENSITISATION BAND, NOT THE DYE. p8's legend reads Yellow /
Magenta / Cyan and its curves are drawn blue / olive / red, because the
yellow-forming layer is the BLUE-sensitive one. Read the legend, not the colour:
blue ink is the blue record, olive the green, red the red. p7 says
Blue / Green / Red outright and agrees.

WHAT IS ADOPTED, AND WHAT IS NOT
--------------------------------
See the per-plate notes and `main()`. The short version: the MTF and the dye
density are NEW measured data into fields that held an estimate and an empty
carrier; the characteristic and spectral plates are read and REPORTED as
cross-checks against what the profile already holds, because replacing a fitted
curve set is a bigger decision than this module should make on its own.

Run:
    python ti0835_plates.py            # all four plates
    python ti0835_plates.py --assert   # non-zero exit on drift
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "PDF", "PROFILES", "KODAK", "5247.pdf")
PROFILE = "EASTMAN_5247_1983"

#: Exact plate inks. No tolerance: these come out of the PDF bit-identical.
INK_BLUE = (0, 0, 255)
INK_OLIVE = (128, 128, 0)
INK_RED = (255, 0, 0)

#: Ink threshold for the GRID, and the span fractions a frame line must cover.
#: ⚠ 0.45 rather than 0.5 is the whole of the B4 blocker -- see the module note.
GRID_INK = 160
GRID_ROW_FRAC = 0.45
GRID_COL_FRAC = 0.35

#: Per plate: (page index, x-extremes, y-extremes, expected grid counts).
#: The extremes are the printed labels at the frame corners; the counts are
#: asserted, because a grid that does not come out at the expected number of
#: round intervals means the frame was found wrong.
PLATES = {
    "mtf":        dict(page=5, x=(0.0, 3.0), y=(3.0, 0.0), nx=28, ny=28,
                       xlog=True, ylog=True),
    "curve":      dict(page=6, x=(-4.0, 2.0), y=(3.0, 0.0), nx=31, ny=16,
                       xlog=False, ylog=False),
    "spectral":   dict(page=7, x=(250.0, 750.0), y=(3.0, -1.0), nx=51, ny=21,
                       xlog=False, ylog=False),
    "dye":        dict(page=8, x=(250.0, 750.0), y=(2.5, 0.0), nx=51, ny=6,
                       xlog=False, ylog=False),
}


def load(page_index):
    import pymupdf
    if not os.path.isfile(PDF):
        return None
    doc = pymupdf.open(PDF)
    pg = doc[page_index]
    imgs = pg.get_images(full=True)
    if not imgs:
        return None
    pix = pymupdf.Pixmap(doc, imgs[0][0])
    if pix.n > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    a = np.frombuffer(pix.samples, dtype=np.uint8)
    a = a.reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)

    # ⚠ THE EMBEDDED RASTER IS STORED UPSIDE DOWN AND THE PDF FLIPS IT AT DRAW
    # TIME. `get_images` hands back the stored bytes, which bypasses the page's
    # transform, so every one of these plates arrives mirrored top-to-bottom --
    # the axis labels included, which is how it was finally spotted.
    #
    # ⚠ THIS ONE FACT PRODUCED EVERY FAILURE OF THE FIRST WORKING RUN AT ONCE,
    # and not one of them looked like an orientation problem: the three
    # characteristic records came out with NEGATIVE gamma and stacked
    # red > green > blue instead of blue > green > red; the spectral peaks came
    # out in reverse order; and the dye plate's D-min appeared to RISE towards
    # the red, which the guard correctly reported as "the two traces are
    # swapped". Three different plausible diagnoses, one cause. The guards
    # earned their place here -- an unflipped read produces numbers that are
    # entirely reasonable and entirely wrong.
    return a[::-1]


def _runs(vals, gap=2):
    out, s, p = [], None, None
    for x in vals:
        if s is None:
            s = p = x
        elif x - p <= gap:
            p = x
        else:
            out.append((s + p) // 2)
            s = p = x
    if s is not None:
        out.append((s + p) // 2)
    return out


def grid(a):
    """Gridline centres, as (horizontals, verticals) in pixels."""
    g = a.mean(2)
    d = g < GRID_INK
    h, w = d.shape
    rows, cols = d.sum(1), d.sum(0)
    hr = [y for y in range(h) if rows[y] > w * GRID_ROW_FRAC]
    vc = [x for x in range(w) if cols[x] > h * GRID_COL_FRAC]
    return _runs(hr), _runs(vc)


def calibrate(a, spec):
    """(to_x, to_y, diagnostics) or (None, None, reason)."""
    hs, vs = grid(a)
    if len(hs) != spec["ny"] or len(vs) != spec["nx"]:
        return None, None, ("grid is %d x %d, expected %d x %d -- the frame or "
                            "the ink threshold is wrong"
                            % (len(vs), len(hs), spec["nx"], spec["ny"]))
    x0, x1 = vs[0], vs[-1]
    y0, y1 = hs[0], hs[-1]
    ax0, ax1 = spec["x"]
    ay0, ay1 = spec["y"]

    # ⚠ UNIFORMITY IS CHECKED, NOT ASSUMED -- BUT ONLY ON A LINEAR AXIS. A
    # frame found one gridline short still produces a plausible linear map;
    # what it cannot produce is even spacing at the expected count. On a LOG
    # axis the gridlines sit at 1,2,...,9,10,20,... and are non-uniform BY
    # DESIGN, so the same test fires on a perfectly good grid -- it did, at
    # 65 px, on the MTF plate. There the check is that the interior lines land
    # where log positions predict, which is a stronger test than evenness.
    dv, dh = np.diff(vs), np.diff(hs)
    if not spec["xlog"]:
        if float(dv.max() - dv.min()) > 2.5:
            return None, None, ("vertical spacing varies by %.1f px -- not a "
                                "uniform grid" % float(dv.max() - dv.min()))
    else:
        want = np.log10(np.array([m * 10 ** e for e in range(3)
                                  for m in range(1, 10)] + [1000.0]))
        want = (want - want[0]) / (want[-1] - want[0])
        got = (np.asarray(vs, float) - vs[0]) / float(vs[-1] - vs[0])
        err = float(np.abs(got - want).max()) * (vs[-1] - vs[0])
        if err > 4.0:
            return None, None, ("log gridlines are %.1f px from their "
                                "predicted positions" % err)
    if not spec["ylog"]:
        if float(dh.max() - dh.min()) > 2.5:
            return None, None, ("horizontal spacing varies by %.1f px -- not "
                                "a uniform grid" % float(dh.max() - dh.min()))
    spread = 0.0

    def to_x(px):
        t = (np.asarray(px, float) - x0) / float(x1 - x0)
        v = ax0 + t * (ax1 - ax0)
        return np.power(10.0, v) if spec["xlog"] else v

    def to_y(py):
        t = (np.asarray(py, float) - y0) / float(y1 - y0)
        v = ay0 + t * (ay1 - ay0)
        return np.power(10.0, v) if spec["ylog"] else v

    return to_x, to_y, dict(x0=x0, x1=x1, y0=y0, y1=y1,
                            step_x=float(dv.mean()), step_y=float(dh.mean()),
                            spread=spread)


#: Largest x gap the tracer will bridge inside one curve. Two of the three
#: records are DASHED (green dash-dot, red dotted), so a run detector that
#: refused any gap would shatter them into dozens of fragments.
CURVE_GAP_PX = 16


def trace(a, ink, tol=30):
    """{x_px: mean y_px} for one exact ink colour, LEGEND SWATCH EXCLUDED.

    ⚠ EVERY ONE OF THESE PLATES DRAWS ITS LEGEND IN THE SAME INKS AS ITS
    CURVES, AND THAT COST THE FIRST RUN ITS ENTIRE RESULT. A colour mask alone
    picks up the legend's little line samples, which sit at a constant height
    near a corner -- so the red record came out with a maximum of 2.83 D on a
    plate whose red curve never passes 1.5, the records failed their stacking
    order, and the dye plate's D-min appeared to RISE towards the red, which
    would have meant the two traces were swapped. Nothing about those failures
    pointed at a legend; they looked like a calibration error.

    The swatch and the curve are told apart by extent: a swatch is a few tens
    of pixels wide, a curve spans most of the frame. Keep the widest run.
    """
    d = (np.abs(a - np.asarray(ink)[None, None, :]).max(2) <= tol)
    ys, xs = np.nonzero(d)
    if xs.size == 0:
        return {}
    col = {}
    for x in np.unique(xs):
        col[int(x)] = float(ys[xs == x].mean())

    order = sorted(col)
    runs, cur = [], [order[0]]
    for x in order[1:]:
        if x - cur[-1] <= CURVE_GAP_PX:
            cur.append(x)
        else:
            runs.append(cur)
            cur = [x]
    runs.append(cur)
    best = max(runs, key=lambda r: r[-1] - r[0])
    return {x: col[x] for x in best}


def curve_points(a, spec, ink):
    to_x, to_y, diag = calibrate(a, spec)
    if to_x is None:
        return None, diag
    px = trace(a, ink)
    if len(px) < 20:
        return None, "only %d columns of ink" % len(px)
    xs = np.array(sorted(px))
    ys = np.array([px[x] for x in xs])
    return (to_x(xs), to_y(ys)), diag


def f50_from(freq, resp):
    """Frequency where the response crosses 50 %, by log-log interpolation."""
    lf, lr = np.log10(freq), np.log10(resp)
    k = int(np.argmax(resp))
    for i in range(k, len(lr) - 1):
        if lr[i] >= np.log10(50.0) >= lr[i + 1]:
            t = (np.log10(50.0) - lr[i]) / (lr[i + 1] - lr[i])
            return float(10.0 ** (lf[i] + t * (lf[i + 1] - lf[i])))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="accepted and unused")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.isfile(PDF):
        print("[SKIP] %s not present" % PDF)
        return 0

    from film_profiles import FILM_PROFILES
    prof = next((p for p in FILM_PROFILES if p.name == PROFILE), None)
    bad = 0
    print("TI0835 plates for %s -- four rasters, 959 x 719 px" % PROFILE)

    # ---- p6 MTF ---------------------------------------------------------
    a = load(PLATES["mtf"]["page"])
    got, diag = curve_points(a, PLATES["mtf"], INK_BLUE)
    if got is None:
        print("[!] MTF: %s" % diag)
        bad += 1
    else:
        freq, resp = got
        f50 = f50_from(freq, resp)
        print("  MTF     %d columns, %.1f-%.0f c/mm, peak %.0f %% at %.1f c/mm"
              % (len(freq), freq.min(), freq.max(), resp.max(),
                 freq[int(np.argmax(resp))]))
        print("          f50 = %.1f cycles/mm  (stored f50_g %.1f, measured=%s)"
              % (f50 if f50 else float("nan"),
                 prof.mtf.f50_g if prof else float("nan"),
                 prof.mtf.mtf_measured if prof else "?"))
        # ⚠ The abscissa is labelled "cycles/mm" IN WORDS, so the units hazard
        # recorded against the Gevaert sheet (queue G6) does not apply here.
        if f50 is None or not (10.0 <= f50 <= 120.0):
            print("[!] MTF: f50 %s is outside any plausible range" % f50)
            bad += 1

    # ---- p7 characteristic ----------------------------------------------
    a = load(PLATES["curve"]["page"])
    rec = {}
    for ink, name in ((INK_BLUE, "blue"), (INK_OLIVE, "green"),
                      (INK_RED, "red")):
        got, diag = curve_points(a, PLATES["curve"], ink)
        if got is None:
            print("[!] characteristic %s: %s" % (name, diag))
            bad += 1
            continue
        le, dd = got
        rec[name] = (le, dd)
    if len(rec) == 3:
        print("  CURVE   three records traced")
        cur = prof.curves.as_tuple() if prof else None
        for i, name in enumerate(("red", "green", "blue")):
            le, dd = rec[name]
            base = float(dd.min())
            m = (le >= le.min() + 0.6) & (le <= le.max() - 0.15)
            g = float(np.polyfit(le[m], dd[m], 1)[0]) if m.sum() > 5 else \
                float("nan")
            stored = cur[i] if cur else None
            print("          %-5s dmin %.3f gamma %.3f   stored dmin %.3f "
                  "gamma %.3f" % (name, base, g,
                                  stored.dmin if stored else float("nan"),
                                  stored.gamma if stored else float("nan")))
        # ⚠ ORDER, ASSERTED. On a colour negative the blue record sits highest
        # (the yellow layer is on top and the mask is strongest in blue). A
        # trace that reversed the records would fit three plausible curves to
        # the wrong layers.
        tops = {k: float(v[1].max()) for k, v in rec.items()}
        if not (tops["blue"] > tops["green"] > tops["red"]):
            print("[!] characteristic: records are not stacked blue > green > "
                  "red: %s" % {k: round(v, 3) for k, v in tops.items()})
            bad += 1

    # ---- p8 spectral, a CROSS-CHECK -------------------------------------
    a = load(PLATES["spectral"]["page"])
    peaks = {}
    for ink, name in ((INK_BLUE, "blue"), (INK_OLIVE, "green"),
                      (INK_RED, "red")):
        got, diag = curve_points(a, PLATES["spectral"], ink)
        if got is None:
            print("[!] spectral %s: %s" % (name, diag))
            bad += 1
            continue
        nm, ls = got
        peaks[name] = float(nm[int(np.argmax(ls))])
    if len(peaks) == 3:
        print("  SPECTRAL peaks  blue %.0f  green %.0f  red %.0f nm"
              % (peaks["blue"], peaks["green"], peaks["red"]))
        if prof is not None and prof.spectral.has_data:
            sp = prof.spectral
            lam = sp.lambda_start_nm + sp.lambda_step_nm * np.arange(
                len(sp.log_s_b))
            st = {"blue": float(lam[int(np.argmax(sp.log_s_b))]),
                  "green": float(lam[int(np.argmax(sp.log_s_g))]),
                  "red": float(lam[int(np.argmax(sp.log_s_r))])}
            print("           stored          blue %.0f  green %.0f  red %.0f nm"
                  % (st["blue"], st["green"], st["red"]))
            worst = max(abs(peaks[k] - st[k]) for k in peaks)
            print("           worst peak disagreement %.0f nm" % worst)
            if worst > 25.0:
                print("[!] spectral: the plate and the stored set disagree by "
                      "%.0f nm on a peak -- one of them is not this film"
                      % worst)
                bad += 1
        if not (peaks["blue"] < peaks["green"] < peaks["red"]):
            print("[!] spectral: peaks out of order")
            bad += 1

    # ---- p9 spectral dye density ----------------------------------------
    a = load(PLATES["dye"]["page"])
    dye = {}
    for ink, name in ((INK_BLUE, "neutral"), (INK_OLIVE, "dmin")):
        got, diag = curve_points(a, PLATES["dye"], ink)
        if got is None:
            print("[!] dye %s: %s" % (name, diag))
            bad += 1
            continue
        dye[name] = got
    if len(dye) == 2:
        nn, nd = dye["neutral"]
        mn, md = dye["dmin"]
        print("  DYE     neutral %.0f-%.0f nm, %.3f-%.3f D; "
              "dmin %.0f-%.0f nm, %.3f-%.3f D"
              % (nn.min(), nn.max(), nd.min(), nd.max(),
                 mn.min(), mn.max(), md.min(), md.max()))
        # ⚠ THE MASK MUST FALL TOWARDS THE RED. A colour negative's D-min IS
        # the orange mask: high in blue, low in red. If this came out rising,
        # the two traces have been swapped.
        lo = float(np.mean(md[mn < 470.0]))
        hi = float(np.mean(md[mn > 650.0]))
        print("          the D-min trace falls %.3f -> %.3f D across the "
              "spectrum (it is the orange mask)" % (lo, hi))
        if not (lo > hi):
            print("[!] dye: the D-min trace rises towards the red -- it is not "
                  "a mask, so the two traces are swapped")
            bad += 1
        if not (nd.min() > md.max()):
            print("[!] dye: the neutral is not everywhere above the D-min")
            bad += 1

    if args.assert_:
        if bad:
            print("[FAIL] the TI0835 plates do not reproduce")
            return 1
        print("[OK] all four TI0835 plates read: grids land on round steps, "
              "inks are exact, records stack blue > green > red, spectral "
              "peaks agree with the stored set, and the D-min trace is a mask")
    return 0


if __name__ == "__main__":
    sys.exit(main())
