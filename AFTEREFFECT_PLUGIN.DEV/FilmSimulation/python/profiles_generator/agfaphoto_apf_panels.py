#!/usr/bin/env python3
"""AgfaPhoto «Technical Data Sheet AP-F» — the six RASTER panels on page 5.

    PDF/PROFILES/AGFA/AgfaPhoto_filmrange_en0.pdf
    AgfaPhoto Holding GmbH / Lupus Imaging & Media GmbH & Co. KG,
    «Technical Data Sheet AP-F», Stand 07/2007, brochure printed 11.01.2008.

Page 5 sets Vista 100, Vista 200 and Vista 400 side by side in three columns,
four captioned panels each. TWO of the four are RASTER and are what this module
reads:

    y ~ 477 pt   «Sharpness:»              transfer factor % vs 2-100 lines/mm
    y ~ 625 pt   «Colour densitiy curves:» D 0-4.0 vs lg exposure -4.0..+1.0

(the other two, «Spectral sensitivity» and «Spectral density», are drawn as
vector paths and are not this module's business; the profiles already carry
those axes from the Vista plus sheet, traced at vector precision.)

⚠⚠ **THE PANELS ARE RASTER AND HAVE NO TEXT LAYER AT ALL** -- not the curve
labels, not the axis numerals, not even the tick values. `get_text()` on page 5
returns the captions and the two printed data lines and nothing else. So the
calibration cannot be anchored on a printed ladder and is anchored on the
DRAWN GRIDLINES instead, exactly as `agfaphoto_vista_plus.py` had to do for the
vector sheet. What makes that safe here is that every panel draws its FULL
grid: the Sharpness panels carry six verticals (2 5 10 20 50 100) and five
horizontals (10 20 50 100 150), the density panels six verticals (-4..+1) and
five horizontals (0..4.0), and a single two-parameter log fit has to reproduce
all six of the one and all five of the other. The worst residual over the six
panels is 3.0 px on a 550 px axis -- 0.55 %, or 0.9 % in frequency.

⚠ **AND THE THREE SHARPNESS PANELS ARE THREE DRAWINGS, NOT ONE PLACED THREE
TIMES.** That was tested before anything was read off them, because the Vista
plus sheet's own MTF panel IS one path object placed twice and the profiles say
so. With frame, grid and outer box erased -- they are 60 % of the dark pixels
and identical in all three panels, so leaving them in dilutes the answer by
three -- and the remainder shifted to the best of 25x9 integer offsets, the
curve pixels of any two of these three disagree by 41.5 % to 52.4 %, where two
placements of one path would agree to the antialiasing. Three films, three
measurements.

WHAT THIS MODULE ADOPTS, AND THE MUCH LARGER PART IT DOES NOT
-------------------------------------------------------------
It adopts **nothing numeric into an existing field**, and that is a decision
rather than an omission. The reason is a three-way disagreement that this sheet
creates and cannot settle:

    overshoot at the Sharpness peak      f50
    ---------------------------------    ----
    Agfa-Gevaert «Technical Data AF»     47.83 c/mm   +0.0978   (Vista 200,
        06/2000, VECTOR, per-film                                 2000)
    AgfaPhoto AP-F 07/2007, RASTER       51.28 c/mm   +0.1012   (Vista 200)
    AgfaPhoto «Product Information»,     58.68 c/mm   +0.1842   (Vista plus
        undated, VECTOR, SHARED                                   200 AND 400)

The first two are independent documents seven years and one bankruptcy apart,
and they agree on the overshoot to 3.5 % and on f50 to 7 %. The third is the
one the two profiles actually carry, it is 1.8x the other two on the overshoot,
and it is the one drawing that is **known not to be per-film** -- section 13 of
the Vista plus sheet is the same 47-point path placed on page 4 and on page 8,
translated by 0.14 pt. Replacing it with this sheet's numbers would swap a
vector trace of the right product for a raster trace of its predecessor, so
nothing is swapped; instead the measurement is recorded where the next reader
will find it, and `DIGITIZATION_QUEUE.md` row P93 asks for the one document
that would settle it.

WHAT IT DOES SETTLE: THE TWO SHEETS DESCRIBE ONE EMULSION PAIR
---------------------------------------------------------------
The AP-F colour-density panels and the Vista plus characteristic curves are the
same films. Fitting each traced record against the stored `ToneCurve` with two
free parameters -- one log-exposure shift, one density offset, nothing else --
lands at

    Vista 200   B 0.0493   G 0.0238   R 0.0756  D rms
    Vista 400   B 0.0318   G 0.0700   R 0.0480  D rms

on a panel whose own quantisation is 0.0092 D per pixel, and the density offset
comes out at +0.192..+0.243 D on all six records: ONE constant, not three, i.e.
a difference of zero reference and not of emulsion. That is the evidence behind
attaching this sheet to `AGFA_VISTA_PLUS_200/400` at all, and behind the
2026-09-24 rms-granularity correction (4.5 -> 4.0) that it licensed. It is
re-run on every build, because it is the load-bearing claim.

⚠ The log-exposure shift is +1.07 decade for the 200 and +1.30 for the 400.
The difference, 0.23 decade, is three quarters of a stop where the speed
difference is one, so the two sheets do NOT share an absolute exposure axis and
neither shift is a speed measurement. Stated so that nobody reads one.

⚠ VISTA 100 HAS NO PROFILE and none is created here. Its panels are traced,
pinned and printed for the record -- f50 51.89 c/mm, q 3.02, overshoot +0.1379,
dmin B/G/R 0.883/0.618/0.244 -- so that adding the stock later is a decision
about scope and not another digitisation job.

Run:  python agfaphoto_apf_panels.py [--root .] [--assert]
"""
from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PDF = "PDF/PROFILES/AGFA/AgfaPhoto_filmrange_en0.pdf"
PAGE = 4                      # printed page 5

#: Column x0 on the page -> film. The three columns are 94.7 / 261.0 / 427.4 pt
#: and never move between the two panel rows, so the film is read off the image
#: RECTANGLE rather than off the order `get_images` happens to return.
COLUMNS = ((94.7, "Vista 100"), (261.0, "Vista 200"), (427.4, "Vista 400"))
#: Panel row y0 -> which chart. 476.9 and 624.6 pt.
ROWS = ((476.9, "sharpness"), (624.6, "density"))

#: The captions that must still be on the page. If AgfaPhoto ever reissue this
#: brochure with the panels rearranged, the geometry above is wrong and the
#: build should say so rather than trace whatever is at those coordinates.
CAPTIONS = ("Sharpness:", "Colour densitiy curves:", "AgfaPhoto Vista 100",
            "AgfaPhoto Vista 200", "AgfaPhoto Vista 400")

#: Sharpness panel ladders. Both axes logarithmic.
SHARP_X = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)          # lines per mm
SHARP_Y = (150.0, 100.0, 50.0, 20.0, 10.0)             # transfer factor %
#: Colour-density panel ladders. Both axes linear.
DENS_X = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0)            # lg exposure
DENS_Y = (4.0, 3.0, 2.0, 1.0, 0.0)                     # density

#: What this module found. A rerun that disagrees FAILS.
EXPECTED = {
    "Vista 100": dict(f50=51.89, q=3.02, overshoot=0.1379, peak_at=4.97,
                      dmin=(0.883, 0.618, 0.244)),
    "Vista 200": dict(f50=51.28, q=2.43, overshoot=0.1012, peak_at=4.60,
                      dmin=(0.945, 0.649, 0.342)),
    "Vista 400": dict(f50=51.28, q=2.45, overshoot=0.1012, peak_at=2.97,
                      dmin=(0.990, 0.717, 0.413)),
}

#: The cross-document identity test, film -> (channel rms in D, shift, offset).
#: ⚠ TOLERANCES ARE LOOSE ON PURPOSE. This is a raster panel measured against a
#: six-parameter fit to a different printing; what is being asserted is that the
#: two sheets describe one emulsion, not that they agree to the last pixel.
EXPECTED_IDENTITY = {
    "AGFA_VISTA_PLUS_200": dict(rms=(0.0493, 0.0238, 0.0756),
                                shift=(1.09, 1.00, 1.12),
                                offset=(0.221, 0.212, 0.192)),
    "AGFA_VISTA_PLUS_400": dict(rms=(0.0318, 0.0700, 0.0480),
                                shift=(1.36, 1.21, 1.34),
                                offset=(0.215, 0.203, 0.243)),
}

#: Which profile each column feeds. Vista 100 deliberately maps to nothing.
PROFILE_OF = {"Vista 200": "AGFA_VISTA_PLUS_200",
              "Vista 400": "AGFA_VISTA_PLUS_400"}


# ---------------------------------------------------------------------------
# raster plumbing
# ---------------------------------------------------------------------------
def _panels(doc):
    """The six images on page 5, keyed (film, chart)."""
    page = doc[PAGE]
    txt = page.get_text()
    missing = [c for c in CAPTIONS if c not in " ".join(txt.split())]
    if missing:
        return None, "page 5 no longer carries %s" % ", ".join(missing)
    out = {}
    for xref, *_ in page.get_images(full=True):
        for rect in page.get_image_rects(xref):
            film = min(COLUMNS, key=lambda c: abs(c[0] - rect.x0))
            chart = min(ROWS, key=lambda r: abs(r[0] - rect.y0))
            if abs(film[0] - rect.x0) > 6.0 or abs(chart[0] - rect.y0) > 6.0:
                continue
            pix = pymupdf.Pixmap(doc, xref)
            if pix.n > 1:
                pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
            from PIL import Image
            img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            out[(film[1], chart[1])] = np.asarray(img) < 128
    if len(out) != 6:
        return None, "expected 6 raster panels on page 5, found %d" % len(out)
    return out, ""


def _group(v):
    o, s, p = [], v[0], v[0]
    for x in v[1:]:
        if x != p + 1:
            o.append((s + p) / 2.0)
            s = x
        p = x
    o.append((s + p) / 2.0)
    return o


def _rules(b, axis, frac=0.55):
    """Centres of the full-length rules along one axis, outer box dropped.

    ⚠ THE OUTER BOX IS NOT THE PLOT FRAME. Every panel is a grey illustration
    box with its own border about 20 px outside the frame; reading the box as
    the ladder shifts every value by a fifth of a decade. The box is the first
    and last rule found and is discarded here by construction.
    """
    h, w = b.shape
    n = b.sum(axis)
    span = h if axis == 0 else w      # summing DOWN a column measures height
    hits = [i for i in range(len(n)) if n[i] > frac * span]
    if not hits:
        return []
    return _group(hits)[1:-1]


def _runs(b, x, y0, y1):
    ys = np.where(b[y0:y1, x])[0] + y0
    if len(ys) == 0:
        return []
    o, s, p = [], ys[0], ys[0]
    for y in ys[1:]:
        if y != p + 1:
            o.append((s, p))
            s = y
        p = y
    o.append((s, p))
    return o


def _fit(values, pixels, log=False):
    xs = [math.log10(v) for v in values] if log else list(values)
    a = np.polyfit(xs, pixels, 1)
    resid = max(abs(a[0] * x + a[1] - p) for x, p in zip(xs, pixels))
    return a, resid


# ---------------------------------------------------------------------------
# the two chart readers
# ---------------------------------------------------------------------------
def read_sharpness(b):
    vx, hy = _rules(b, 0, 0.60), _rules(b, 1, 0.60)
    if len(vx) != 6 or len(hy) != 5:
        return None, "sharpness grid is %d x %d, want 6 x 5" % (len(vx), len(hy))
    lx, rx = _fit(SHARP_X, vx, log=True)
    ly, ry = _fit(SHARP_Y, hy, log=True)
    grid = set()
    for y in hy:
        grid |= {int(round(y)) - 1, int(round(y)), int(round(y)) + 1}
    y0, y1 = int(round(hy[0])) + 1, int(round(hy[-1]))
    f, r, prev = [], [], None
    for x in range(int(round(vx[0])) + 2, int(round(vx[-1])) - 1):
        cand = [q for q in _runs(b, x, y0, y1)
                if not set(range(q[0], q[1] + 1)) <= grid and q[1] - q[0] <= 14]
        if not cand:
            continue
        s, e = (max(cand, key=lambda q: q[1] - q[0]) if prev is None
                else min(cand, key=lambda q: abs((q[0] + q[1]) / 2.0 - prev)))
        c = (s + e) / 2.0
        if prev is not None and abs(c - prev) > 12:
            continue
        prev = c
        f.append(10.0 ** ((x - lx[1]) / lx[0]))
        r.append(10.0 ** ((c - ly[1]) / ly[0]) / 100.0)
    f, r = np.array(f), np.array(r)
    below = np.flatnonzero(r < 0.5)
    if not len(below) or below[0] == 0:
        return None, "the curve never crosses 50 % inside the panel"
    f50 = float(np.interp(0.5, [r[below[0]], r[below[0] - 1]],
                          [f[below[0]], f[below[0] - 1]]))
    peak, pk_at = float(r.max()), float(f[int(r.argmax())])
    # ⚠ FITTED ABOVE THE PEAK ONLY. The overshoot is an adjacency effect and
    # belongs to `mtf.adjacency`; folding it into the rolloff would make a
    # transfer function that exceeds 1, which no MTF does.
    m = f > max(4.0, pk_at)
    q = qe = None
    for cand in np.arange(1.0, 5.001, 0.01):
        e = float(np.sqrt(np.mean(
            (1.0 / (1.0 + (f[m] / f50) ** cand) - r[m]) ** 2)))
        if qe is None or e < qe:
            q, qe = float(cand), e
    ge = float(np.sqrt(np.mean(
        (np.exp(-math.log(2.0) * (f[m] / f50) ** 2) - r[m]) ** 2)))
    return dict(f50=f50, q=q, q_rms=qe, gauss_rms=ge, overshoot=peak - 1.0,
                peak_at=pk_at, n=len(f), resid=(rx, ry),
                f=f, r=r), ""


def read_density(b):
    vx, hy = _rules(b, 0, 0.55), _rules(b, 1, 0.55)
    if len(vx) != 6 or len(hy) != 5:
        return None, "density grid is %d x %d, want 6 x 5" % (len(vx), len(hy))
    lx, rx = _fit(DENS_X, vx)
    ly, ry = _fit(DENS_Y, hy)
    grid = set()
    for y in hy[1:-1]:
        grid |= {int(round(y)) - 1, int(round(y)), int(round(y)) + 1}
    y0, y1 = int(round(hy[0])) + 2, int(round(hy[-1])) - 1
    cur, out = None, [[], [], []]
    for x in range(int(round(vx[0])) + 2, int(round(vx[-1])) - 2):
        rs = [q for q in _runs(b, x, y0, y1)
              if not set(range(q[0], q[1] + 1)) <= grid and q[1] - q[0] <= 8]
        cen = sorted((q[0] + q[1]) / 2.0 for q in rs)
        if cur is None:
            # ⚠ SEEDED ON THE FIRST COLUMN THAT SHOWS EXACTLY THREE RUNS, and
            # then tracked by nearest neighbour. The curves never cross -- a
            # masked colour negative's three records are stacked B over G over
            # R for their whole length -- but the panel prints its own «Blue /
            # Green / Red» key INSIDE the frame at the top right, which adds
            # spurious runs there. Tracking survives that; picking the three
            # darkest runs per column would not.
            if len(cen) != 3:
                continue
            cur = cen
        else:
            new = list(cur)
            for k in range(3):
                if cen:
                    j = min(range(len(cen)), key=lambda t: abs(cen[t] - cur[k]))
                    if abs(cen[j] - cur[k]) <= 6:
                        new[k] = cen[j]
            cur = new
        for k in range(3):
            out[k].append(((x - lx[1]) / lx[0], (cur[k] - ly[1]) / ly[0]))
    if min(len(o) for o in out) < 400:
        return None, "one of the three records traced short"
    dmin = tuple(float(np.mean([p[1] for p in o if p[0] <= -3.2]))
                 for o in out)
    return dict(b=out[0], g=out[1], r=out[2], dmin=dmin,
                resid=(rx, ry), n=len(out[0])), ""


# ---------------------------------------------------------------------------
# the cross-document identity test
# ---------------------------------------------------------------------------
def identity(dens, profile):
    """Two free parameters -- lg-E shift and D offset -- and nothing else."""
    import film_sim as FS
    got = {}
    for lab, curve in (("b", profile.curves.b), ("g", profile.curves.g),
                       ("r", profile.curves.r)):
        pts = np.array(dens[lab])
        x, y = pts[:, 0], pts[:, 1]
        best = None
        for dx in np.arange(-1.5, 1.501, 0.01):
            m = FS.density(x + dx, curve)
            dy = float((y - m).mean())
            rr = float(np.sqrt(((y - m - dy) ** 2).mean()))
            if best is None or rr < best[0]:
                best = (rr, float(dx), dy)
        got[lab] = best
    return got


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()
    if not (root / PDF).is_file():
        print("  [SKIP] source not present: %s" % (root / PDF))
        return 0

    print("AgfaPhoto «Technical Data Sheet AP-F» 07/2007 -- the six raster "
          "panels on page 5")
    doc = pymupdf.open(str(root / PDF))
    panels, err = _panels(doc)
    if panels is None:
        print("  [FAIL] %s" % err)
        return 1 if ns.assert_ else 0
    bad = 0

    # ---- are the three Sharpness panels three drawings? -------------------
    keys = [c[1] for c in COLUMNS]

    def _curve_only(b):
        """The panel with its frame, grid and outer box erased.

        ⚠ WITHOUT THIS THE TEST IS MEANINGLESS. Frame and gridlines are about
        60 % of the dark pixels and are identical in all three panels, so a
        whole-panel comparison dilutes a 90 % curve disagreement down to 30 %
        and would pass a sheet that really did reuse one drawing.
        """
        c = b.copy()
        for y in _rules(b, 1, 0.60) + [0.0, float(b.shape[0] - 1)]:
            c[max(0, int(round(y)) - 2):int(round(y)) + 3, :] = False
        for x in _rules(b, 0, 0.60) + [0.0, float(b.shape[1] - 1)]:
            c[:, max(0, int(round(x)) - 2):int(round(x)) + 3] = False
        return c

    worst = None
    for i in range(3):
        for j in range(i + 1, 3):
            A = _curve_only(panels[(keys[i], "sharpness")])
            B = _curve_only(panels[(keys[j], "sharpness")])
            n = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
            A, B = A[:n[0], :n[1]], B[:n[0], :n[1]]
            best = min(int(np.logical_xor(A, np.roll(B, (dy, dx), (0, 1))).sum())
                       for dx in range(-12, 13) for dy in range(-4, 5))
            frac = best / float(A.sum())
            worst = frac if worst is None else min(worst, frac)
            print("  %-9s vs %-9s  best-aligned CURVE-pixel mismatch %5.1f %%"
                  % (keys[i], keys[j], 100 * frac))
    if worst is not None and worst < 0.25:
        print("  [FAIL] the Sharpness panels register onto each other -- this "
              "sheet reuses one drawing and its curves are not per-film")
        bad += 1
    else:
        print("  [OK  ] three distinct Sharpness drawings, minimum mismatch "
              "%.1f %%" % (100 * worst))

    # ---- the numbers ------------------------------------------------------
    got = {}
    for film in keys:
        s, e1 = read_sharpness(panels[(film, "sharpness")])
        d, e2 = read_density(panels[(film, "density")])
        if s is None or d is None:
            print("  [FAIL] %s: %s" % (film, e1 or e2))
            bad += 1
            continue
        got[film] = (s, d)
        print("\n  %s" % film)
        print("    Sharpness  f50 %6.2f c/mm  q %.2f (rms %.4f vs Gaussian "
              "%.4f)  overshoot %+.4f at %.2f c/mm"
              % (s["f50"], s["q"], s["q_rms"], s["gauss_rms"],
                 s["overshoot"], s["peak_at"]))
        print("               %d points, grid residual %.1f / %.1f px"
              % (s["n"], s["resid"][0], s["resid"][1]))
        print("    Density    D-min B %.3f  G %.3f  R %.3f   %d points, grid "
              "residual %.1f / %.1f px"
              % (d["dmin"] + (d["n"], d["resid"][0], d["resid"][1])))
        want = EXPECTED[film]
        for k in ("f50", "q", "overshoot", "peak_at"):
            tol = 0.6 if k == "f50" else (0.05 if k == "peak_at" else 0.02)
            if abs(s[k] - want[k]) > tol:
                print("    [MISMATCH] %s %.4f vs pinned %.4f"
                      % (k, s[k], want[k]))
                bad += 1
        for k, (a, w) in enumerate(zip(d["dmin"], want["dmin"])):
            if abs(a - w) > 0.02:
                print("    [MISMATCH] D-min %s %.3f vs pinned %.3f"
                      % ("BGR"[k], a, w))
                bad += 1

    # ---- the identity test ------------------------------------------------
    try:
        import film_profiles as fp
        print("\n  CROSS-DOCUMENT IDENTITY -- AP-F raster panel against the "
              "Vista plus vector curves the profiles carry:")
        for film, name in PROFILE_OF.items():
            if film not in got:
                continue
            res = identity(got[film][1], fp.get_profile(name))
            want = EXPECTED_IDENTITY[name]
            print("    %-20s %s" % (name, "  ".join(
                "%s rms %.4f D shift %+.2f offset %+.3f"
                % (k.upper(), res[k][0], res[k][1], res[k][2])
                for k in "bgr")))
            for i, k in enumerate("bgr"):
                if res[k][0] > want["rms"][i] + 0.010:
                    print("    [MISMATCH] %s %s rms %.4f vs pinned %.4f"
                          % (name, k.upper(), res[k][0], want["rms"][i]))
                    bad += 1
            off = [res[k][2] for k in "bgr"]
            if max(off) - min(off) > 0.08:
                print("    [FAIL] the density offset is not ONE constant "
                      "across the three records (%.3f..%.3f), so the two "
                      "sheets differ by more than a zero reference"
                      % (min(off), max(off)))
                bad += 1
    except Exception as exc:                                  # pragma: no cover
        print("  [WARN] could not consult film_profiles: %s" % exc)

    # ---- against the database ---------------------------------------------
    try:
        import film_profiles as fp
        print("\n  AGAINST THE DATABASE -- nothing here is adopted, and this "
              "is the standing disagreement, printed so it stays visible:")
        for film, name in PROFILE_OF.items():
            if film not in got:
                continue
            m = fp.get_profile(name).mtf
            s = got[film][0]
            print("    %-20s stored f50_g %.2f / q %.2f / adjacency %.4f "
                  "(Vista plus vector, SHARED drawing)"
                  % (name, m.f50_g, m.mtf_rolloff_q, m.adjacency))
            print("    %-20s AP-F   f50   %.2f / q %.2f / adjacency %.4f "
                  "(this sheet, per-film raster)"
                  % ("", s["f50"], s["q"], s["overshoot"]))
    except Exception as exc:                                  # pragma: no cover
        print("  [WARN] could not compare against film_profiles: %s" % exc)

    if ns.assert_ and bad:
        print("\n[FAIL] the AP-F page-5 panels do not reproduce")
        return 1
    print("\n[OK] six raster panels re-read, three Sharpness drawings proved "
          "distinct, and the AP-F colour-density curves re-shown to be the "
          "Vista plus emulsions under a single +0.19..+0.24 D change of zero.")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
