"""ToneCurve parameters from the VECTOR characteristic curves on a Kodak sheet.

WHY THIS EXISTS
---------------
`RGBCurves` is the most load-bearing structure in the database -- the crossover
between its three curves *is* a stock's colour signature -- and for most stocks
the six numbers per channel are hand-fitted to a printed picture. Kodak's
brochures draw that picture as VECTOR paths, so on those sheets the curve can be
read and the fit can be least-squares rather than by eye.

`digitize_plot.py` already fits this model but only from pixels;
`gevaert_curves.py` does the same job for a SCANNED page. This is the vector road
in for Kodak's own house layout, and it shares the fitter
(`digitize_plot.fit_tonecurve`) so the model has ONE definition in the project.

⚠ THE SENSITOMETRIC PANEL IS NOT THE BEST SOURCE ON ITS OWN SHEET, and finding
that out is the whole design of this file. H-1-5201 p3 draws its "SENSITOMETRIC
CURVES" panel at brochure scale: the red and green records are **5-segment
POLYLINES -- six vertices each** -- spanning only -6..+6 camera stops, which is
+-1.8 decades. Neither dmin nor dmax is on the panel. Fitting six free
parameters to six points returned rms 0.0004 D, a number that describes an
interpolation and not a film: `shoulder_x` came out 1.73 with no shoulder
anywhere in the data. That is method rule 18 (reject a form the measurement
cannot support) arriving from the data side instead of the model side.

WHAT IT DOES INSTEAD -- two panels, each used for what it actually carries
-------------------------------------------------------------------------
  * SHAPE comes from the characteristic curves inside the **granularity** panel
    of the same sheet. Kodak draws those densely and over the whole scale: on
    5201, 100 / 125 / 121 samples per record across 4.1 decades, reaching both
    the toe plateau and the shoulder. `granularity_vector.py` already locates
    that panel, calibrates its three axes and splits the six curves into two
    families by ink, so this imports it rather than re-deriving any of it.
  * The ABSCISSA ORIGIN comes from the sensitometric panel, and nothing else
    does. The granularity panel's x axis is labelled "RELATIVE LOG EXPOSURE
    0.0..4.0" -- relative to an origin it does not state. The sensitometric
    panel states its origin in words: "'0' on the x-axis represents normal
    exposure of an 18-percent gray card", which is the origin
    `ToneCurve.toe_x` / `shoulder_x` are defined against. So the coarse
    panel is asked for ONE number -- a shift -- fitted over all three records at
    once. Six points per record is ample for one parameter and hopeless for six.

⚠ THE SHIFT IS FITTED ON THE STRAIGHT-LINE SECTION ONLY. A six-vertex polyline
is a set of CHORDS of the real curve, so in the toe and the shoulder it lies
systematically off it -- matching densities there biases the shift by the chord
error rather than measuring the offset. Restricting the match to vertices between
dmin + 0.3 and dmin + 1.2, where the curve is straight and a chord is the curve,
cuts the residual from 0.048 to 0.026 D. Measured sensitivity: the shift moves
from +1.976 to +1.993 decades, i.e. 0.017 decade or 0.06 stop, so the choice is
reported rather than agonised over -- but it is reported, and both numbers are
printed on every run.

⚠ DO NOT CHECK A FITTED toe_x AGAINST THE REST OF THE FAMILY. The stored VISION2
siblings carry toe_k 0.300 and shoulder_k 0.420 in ALL THREE channels, which is
the signature of numbers set by hand rather than measured; their toe_x and
shoulder_x are the same kind of value. 5201's are the family's first measured
pair and they do NOT match the hand-set ones (toe_x -0.93 against -1.42..-1.58).
That is a statement about the estimates, not about this fit.

⚠ THE TWO PANELS ARE NOT THE SAME MEASUREMENT, and the sheet says so on the same
page: "Sensitometric and Diffuse RMS Granularity curves are produced on
different equipment. A slight variation in curve shape may be noticed." The
residual of the shift fit is therefore reported per record; on 5201 it is 0.02 D
rms, which is the sheet's "slight variation" quantified rather than assumed.

⚠ THE TWO PRINTED ABSCISSAE ON THE SENSITOMETRIC PANEL CROSS-CHECK EACH OTHER.
Camera Stops gives 24.70 pt/decade and the LOG EXPOSURE (lux-seconds) ladder
along the top gives 24.47 pt/decade independently -- 0.9 % apart. Asserted,
because a stops axis mis-read by a factor of two produces a plausible-looking
curve with half the latitude and no other symptom.

⚠ THE PRINTED TICK LABELS ARE TYPOGRAPHICALLY JITTERED on this sheet, as they
are on its granularity panel. The density axis is taken from the frame span
carrying the labels' extreme values, admitted only when its slope agrees with
the label fit to 2 %. Same rule and same reason as granularity_vector.py.

⚠ THE FIT IS SEEDED FROM THE DATA, NEVER FROM THE PROFILE IT FEEDS. That is the
lesson of gevaert_curves.py: seeding with the stored profile makes the
"measurement" a function of its own previous output. Seeds here come from the
traced curve's own minimum, its steepest 0.6-decade slope and its 10 %/90 %
crossings, over a small grid of toe and shoulder softnesses.

Run:
    python kodak_sensitometry.py --root ../..
    python kodak_sensitometry.py --root ../.. --assert
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import digitize_plot as dp
import granularity_vector as gv

STOPS_PER_DECADE = 3.321928
TICK_RESID_PT = 1.5

#: the straight-line section, as (dmin + lo, dmin + hi) in density. See the
#: module note on chord bias -- this is where a 6-vertex polyline IS the curve.
STRAIGHT_BAND = (0.3, 1.2)

#: tag -> (pdf under PDF/PROFILES/KODAK, page, profile,
#:         sensitometric frame x0,x1,y0,y1)
#: The granularity panel on the same page is found by granularity_vector.py.
SHEETS = {
    "5201": ("Kodak VISION2 50D 5201.pdf", 3, "KODAK_VISION2_50D_5201",
             (226.68, 343.28, 70.20, 134.91)),
}

#: Measured 2026-08-20. --assert fails if a sheet stops reproducing these.
#: (dmin, gamma, toe_x, toe_k, shoulder_x, shoulder_k), then the fit's rms in D.
EXPECTED = {
    "5201": {
        # shoulder_k pinned at 1.4*toe_k -- see fit_pinned_shoulder()
        "R": (0.1662, 0.5129, -0.9426, 0.1289, 1.8751, 0.1804, 0.0053),
        "G": (0.6240, 0.5945, -0.9330, 0.1412, 1.9680, 0.1060, 0.0065),
        "B": (1.0149, 0.5954, -0.9741, 0.1997, 2.1490, 0.2773, 0.0064),
    },
}
TOL_P = 0.05          # per-parameter window
TOL_RMS = 0.008       # D

IDEAL = {"R": (1.0, 0.0, 0.0), "G": (0.0, 1.0, 0.0), "B": (0.0, 0.0, 1.0)}


def flatten(items, n=40):
    """Bezier + line items -> a dense polyline."""
    pts = []
    for it in items:
        if it[0] == "c":
            P = [it[1], it[2], it[3], it[4]]
            for k in range(n + 1):
                t = k / n
                u = 1.0 - t
                pts.append((
                    u**3*P[0].x + 3*u*u*t*P[1].x + 3*u*t*t*P[2].x + t**3*P[3].x,
                    u**3*P[0].y + 3*u*u*t*P[1].y + 3*u*t*t*P[2].y + t**3*P[3].y))
        elif it[0] == "l":
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
    return pts


def linfit(pairs, label, allow_frame=None):
    """{value: pixel} -> (slope, intercept, worst residual, n).

    `allow_frame` is (pixel_at_min_value, pixel_at_max_value); when the label fit
    is not collinear the axis is taken from those two, provided the two slopes
    agree to 2 %. See the module note on jittered labels.
    """
    v = np.array(sorted(pairs), dtype=float)
    px = np.array([pairs[k] for k in sorted(pairs)], dtype=float)
    A = np.vstack([v, np.ones(len(v))]).T
    m, c = np.linalg.lstsq(A, px, rcond=None)[0]
    worst = float(np.abs(m*v + c - px).max())
    if worst <= TICK_RESID_PT:
        return m, c, worst, len(v)
    if allow_frame is None:
        raise SystemExit(f"[!] {label}: ticks not collinear, {worst:.2f} pt "
                         f"over {len(v)}")
    p_lo, p_hi = allow_frame
    m2 = (p_hi - p_lo) / (v[-1] - v[0])
    if abs(m2 / m - 1.0) > 0.02:
        raise SystemExit(f"[!] {label}: ticks not collinear ({worst:.2f} pt) and "
                         f"the frame-span fallback disagrees: {m2:.3f} vs {m:.3f}")
    c2 = p_lo - v[0] * m2
    w2 = float(np.abs(m2*v + c2 - px).max())
    print(f"    {label}: FRAME-SPAN fallback, labels jitter up to {w2:.2f} pt "
          f"about a {m2:.4f} pt/unit axis")
    return m2, c2, 0.0, len(v)


def coarse_curves(pg, fr):
    """{record: polyline} for the sensitometric panel's three curves.

    Coarse by construction -- see the module note. Used only for the shift.
    """
    x0, x1, y0, y1 = fr
    out = {}
    for p in pg.get_drawings():
        r = p["rect"]
        if not (x0-3 <= r.x0 and r.x1 <= x1+3 and y0-3 <= r.y0 and r.y1 <= y1+3):
            continue
        if sum(1 for it in p["items"] if it[0] in ("l", "c")) < 1:
            continue
        pts = flatten(p["items"])
        if len(pts) < 4:
            continue
        if max(x for x, _ in pts) - min(x for x, _ in pts) < 0.20*(x1-x0):
            continue
        col = p.get("color")
        if not col:
            continue
        col = tuple(round(float(c), 3) for c in col)
        if max(col) - min(col) < 0.12:        # grid and frame ink
            continue
        rec = min(IDEAL, key=lambda t: sum((col[k]-IDEAL[t][k])**2
                                           for k in range(3)))
        # yellow and magenta both land on R and are the same geometry; the denser
        # point list survives, the same overprint collapse the granularity and
        # MTF extractors do on this sheet
        if rec not in out or len(pts) > len(out[rec]):
            out[rec] = pts
    return out


def dense_curves(pg):
    """{record: (x in decades of RELATIVE log E, density)} from the gran panel.

    Everything here is granularity_vector's, deliberately: the frame, the three
    axis fits, the family split and the ink-based record assignment are the
    machinery that file already pins against these PDFs, and duplicating any of
    it would create a second definition to keep in step.
    """
    fr, fx, fd, fs, xs, dens, sig = gv.frame_and_ticks(pg)
    if fx is None:
        raise SystemExit("[!] the granularity panel prints no numeric x labels, "
                         "so its abscissa cannot be put in decades")
    to_d = lambda py: (py - fd[1]) / fd[0]
    polys = gv.curves(pg, fr)
    ch, gr, gap = gv.split_families(polys, to_d, (3, 3))
    if ch is None:
        raise SystemExit("[!] the granularity panel's two curve families are not "
                         "separated -- refusing to guess")
    lab = gv.colour_assign(ch, gr)
    if lab is None:
        raise SystemExit("[!] the granularity panel's inks do not map one-to-one "
                         "onto R / G / B")
    out = {}
    for rec, idx in lab["ch"].items():
        a = np.array(sorted(ch[idx][0]))
        x = (a[:, 0] - fx[1]) / fx[0]
        d = to_d(a[:, 1])
        o = np.argsort(x)
        x, d = x[o], d[o]
        k = np.concatenate(([True], np.diff(x) > 1e-9))
        out[rec] = (x[k], d[k])
    return out, gap


def fit_shift(dense, coarse):
    """The one number the coarse panel is asked for: relative -> 18 %-grey x.

    Fitted over all three records at once by minimising density disagreement,
    then reported per record so the sheet's own "slight variation in curve shape
    may be noticed" is a measured number instead of a disclaimer.
    """
    def err(sh, recs=None, band=STRAIGHT_BAND):
        tot, n = 0.0, 0
        for rec in (recs or coarse):
            if rec not in dense:
                continue
            dx, dd = dense[rec]
            cx = np.array([p[0] for p in coarse[rec]])
            cd = np.array([p[1] for p in coarse[rec]])
            q = cx + sh                       # coarse x moved into relative space
            m = (q >= dx.min()) & (q <= dx.max())
            if band is not None:
                lo = float(dd.min())
                m &= (cd >= lo + band[0]) & (cd <= lo + band[1])
            if m.sum() < 2:
                continue
            r = np.interp(q[m], dx, dd) - cd[m]
            tot += float(np.sum(r*r))
            n += int(m.sum())
        return (tot/n if n else 1e9), n

    def solve(band):
        grid = np.arange(-1.0, 5.001, 0.002)
        b = min(grid, key=lambda s: err(s, band=band)[0])
        fine = np.arange(b-0.004, b+0.004, 0.0002)
        return float(min(fine, key=lambda s: err(s, band=band)[0]))

    best = solve(STRAIGHT_BAND)
    whole = solve(None)                       # the sensitivity, printed not used
    per = {}
    for rec in coarse:
        e, n = err(best, [rec])
        per[rec] = (np.sqrt(e) if n else float("nan"), n)
    return best, per, whole


def fit_pinned_shoulder(x, d, init):
    """Refit with shoulder_k pinned to exactly 1.4 * toe_k -- five free params.

    ⚠ THE FREE FIT CAN LAND JUST OUTSIDE THE PROJECT'S MONOTONICITY RULE, and it
    does here: 5201's red record wants shoulder_k = 1.4004 * toe_k, because
    `digitize_plot.fit_tonecurve` enforces the rule with a soft penalty rather
    than a barrier. A value 0.03 % over the line is not a measurement disagreeing
    with the model, it is an optimiser sitting on a constraint -- so the adopted
    numbers come from a refit with the constraint imposed exactly, and the cost
    of imposing it is printed. On 5201 it is 0.0000 D.
    """
    x = np.asarray(x, float)
    d = np.asarray(d, float)

    def loss(q):
        dmin, gamma, tx, tk, sx = q
        if gamma <= 0 or tk <= 0.02 or sx <= tx:
            return 1e9
        r = dp.softplus_curve(x, dmin, gamma, tx, tk, sx, 1.4*tk) - d
        return float(np.mean(r*r))

    q0 = np.array([init[0], init[1], init[2], init[3], init[4]], float)
    q, _ = dp._nelder_mead(loss, q0, [0.02, 0.03, 0.08, 0.04, 0.08])
    p = (float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]),
         float(1.4*q[3]))
    r = dp.softplus_curve(x, *p) - d
    return p, float(np.sqrt(np.mean(r*r))), float(np.max(np.abs(r)))


def seeds(x, d):
    """Data-derived starting points. Never the stored profile -- see module note."""
    lo, hi = float(d.min()), float(d.max())
    rng = max(hi - lo, 1e-6)
    g = 0.0
    for i in range(len(x)):
        j = int(np.searchsorted(x, x[i] + 0.6))
        if j >= len(x):
            break
        g = max(g, (d[j] - d[i]) / (x[j] - x[i]))
    g = g if g > 0.05 else rng / 3.0
    x10 = float(np.interp(lo + 0.10*rng, d, x))
    x90 = float(np.interp(lo + 0.90*rng, d, x))
    out = []
    for tk in (0.20, 0.30, 0.42):
        for sk in (0.28, 0.42, 0.55):
            if sk > 1.4*tk:
                continue
            out.append((lo, g, x10 - 0.25, tk, x90 + 0.25, sk))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()
    import pymupdf

    bad = 0
    for tag, (fn, pgno, prof, fr) in SHEETS.items():
        pdf = Path(ns.root).resolve() / "PDF" / "PROFILES" / "KODAK" / fn
        if not pdf.is_file():
            print(f"  [SKIP] {tag}: source not present: {fn}")
            continue
        pg = pymupdf.open(pdf)[pgno-1]
        x0, x1, y0, y1 = fr
        stops, dens, logh = {}, {}, {}
        for a, b, c, d, t, *_ in pg.get_text("words"):
            if not re.fullmatch(r'-?\d+(\.\d+)?', t):
                continue
            v = float(t)
            cx, cy = (a+c)/2.0, (b+d)/2.0
            if x0-14 <= cx <= x1+14 and y1+1 <= cy <= y1+12:
                stops[v] = cx
            elif x0-14 <= cx <= x1+14 and y0-12 <= cy < y0-1:
                logh[v] = cx
            elif x0-30 <= cx < x0-1 and y0-8 <= cy <= y1+8 and 0.0 <= v <= 6.0:
                dens[v] = cy
        if len(stops) < 4 or len(dens) < 3:
            print(f"  [FAIL] {tag}: ticks stops={len(stops)} density={len(dens)}")
            bad += 1
            continue
        print(f"[i] {fn} p{pgno} -> {prof}")
        fstop = linfit(stops, "camera stops")
        fd = linfit(dens, "density", allow_frame=(y1, y0))
        if len(logh) >= 3:
            fh = linfit(logh, "log exposure")
            per_dec = fstop[0] * STOPS_PER_DECADE
            if abs(fh[0] / per_dec - 1.0) > 0.02:
                print(f"    [FAIL] the two abscissae disagree: {per_dec:.2f} vs "
                      f"{fh[0]:.2f} pt/decade")
                bad += 1
                continue
            print(f"    abscissae agree: {per_dec:.2f} vs {fh[0]:.2f} pt/decade;"
                  f" 0 stops = log H {(fstop[1]-fh[1])/fh[0]:+.3f} lux-seconds")

        coarse = {}
        for rec, pts in coarse_curves(pg, fr).items():
            a = np.array(sorted(pts))
            coarse[rec] = list(zip(
                (a[:, 0] - fstop[1]) / fstop[0] / STOPS_PER_DECADE,
                (a[:, 1] - fd[1]) / fd[0]))
        print(f"    sensitometric panel: " + ", ".join(
            f"{r} {len(coarse[r])} vertices" for r in sorted(coarse)) +
            "  -- used ONLY for the abscissa shift, see the module note")

        dense, gap = dense_curves(pg)
        shift, per, whole = fit_shift(dense, coarse)
        print(f"    abscissa shift = {shift:+.4f} decades (relative log E of the "
              f"18 % grey point), from {sum(n for _, n in per.values())} coarse "
              f"vertices on the straight-line section over {len(per)} records; "
              f"all vertices would give {whole:+.4f} "
              f"({abs(whole-shift)*STOPS_PER_DECADE:.2f} stop)")
        print("    panel-to-panel density disagreement: " + ", ".join(
            f"{r} {per[r][0]:.3f} D" for r in sorted(per)))

        pins = EXPECTED.get(tag, {})
        for rec in ("R", "G", "B"):
            if rec not in dense:
                print(f"    [FAIL] {rec}: not present in the granularity panel")
                bad += 1
                continue
            x, d = dense[rec]
            xa = x - shift
            best = None
            for s in seeds(xa, d):
                p, rms, mx = dp.fit_tonecurve(xa, d, s)
                if best is None or rms < best[1]:
                    best = (p, rms, mx)
            p, rms, mx = best
            print(f"    {rec}: {len(xa)} samples over log E {xa.min():+.2f}.."
                  f"{xa.max():+.2f}, D {d.min():.2f}..{d.max():.2f}")
            if p[5] > 1.4*p[3]:
                p2, rms2, mx2 = fit_pinned_shoulder(xa, d, p)
                print(f"       free fit wants shoulder_k = {p[5]/p[3]:.4f}*toe_k, "
                      f"outside the project rule; refitted with it pinned at 1.4x "
                      f"-- rms {rms:.4f} -> {rms2:.4f} D")
                p, rms, mx = p2, rms2, mx2
            print(f"       ToneCurve({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}, "
                  f"{p[3]:.4f}, {p[4]:.4f}, {p[5]:.4f})   rms {rms:.4f} D, "
                  f"worst {mx:.4f} D, dmax {p[0]+p[1]*(p[4]-p[2]):.2f}, "
                  f"latitude {(p[4]-p[2])*STOPS_PER_DECADE:.1f} stops")
            if p[5] > 1.4*p[3]*(1.0 + 1e-6):
                print(f"    [FAIL] {rec}: shoulder_k {p[5]:.3f} exceeds "
                      f"1.4*toe_k {1.4*p[3]:.3f} -- the project monotonicity rule")
                bad += 1
            w = pins.get(rec)
            if w:
                if max(abs(p[i]-w[i]) for i in range(6)) > TOL_P:
                    print(f"    [FAIL] {rec} parameters moved beyond {TOL_P}")
                    bad += 1
                if rms > w[6] + TOL_RMS:
                    print(f"    [FAIL] {rec} fit degraded: rms {rms:.4f} vs "
                          f"recorded {w[6]:.4f}")
                    bad += 1
    print()
    if bad:
        print(f"[FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("[OK] characteristic curves fitted from the sheet's vector paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
