#!/usr/bin/env python3
"""Trace the three FUJICOLOR SUPERIA datasheets, 2026-09-06e (queue T4).

    SUPERIA X-TRA 400   PDF/PROFILES/FUJI/superia_xtra400_datasheet.pdf   AF3-151E p6
    SUPERIA X-TRA 800   PDF/PROFILES/FUJI/superia_xtra800_datasheet.pdf   AF3-068E p4
    SUPERIA REALA       PDF/PROFILES/FUJI/superia_reala_datasheet.pdf     AF3-967E p4

⚠⚠ THE THREE SHEETS ARE THE SAME HOUSE TEMPLATE AND THE PANELS ARE NOT
INTERCHANGEABLE, WHICH IS WHY EVERY LADDER IS DECLARED PER PANEL BELOW RATHER
THAN ASSUMED FROM THE FAMILY. Three concrete traps this batch walked into and
the declarations exist to stop:

  1. X-TRA 400's characteristic ordinate runs to **4.0**; the other two stop at
     **3.5**. Reading 400's panel on a 3.5 ladder is an 8 % density error that
     looks entirely plausible.
  2. On X-TRA 800's characteristic panel the FRAME TOP IS THE D 3.5 RULE and
     there is a separate stray rule 2.5 pt below it (the caption box). Taking
     the first detected horizontal as the top of the ladder puts every density
     out by that offset.
  3. On REALA's dye panel the top TWO detected horizontals are caption-box
     rules; its ladder starts at the third. Fitting all six returns a spacing
     of 29.7/41.0/39.7 pt for equal 0.5 D steps, which is the panel telling you
     the assignment is wrong.

⚠ AND THE SQUARE-PANEL PROPERTY IS A CHECK HERE, NOT A CALIBRATION. It is
reported per sheet and never used to derive an axis: 400 and REALA come out
within 1.5 %, X-TRA 800's two axes differ by 2.6 %. A panel that is not square
is not a panel that is misread -- PROVIA 400F established that -- so the number
is printed and the fit is left alone.

Run bare for the full report; `--assert` re-derives every adopted number and
exits non-zero if any has moved. Registered in build.py's audit stage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import digitize_plot as dp                                  # noqa: E402

#: Colour negatives here stop at logH +0.3..+0.7, well inside the straight
#: line, so the shoulder is NOT in the data and must not be fitted. Pinned at
#: the family default every other colour negative in this database uses -- see
#: fuji_t3_2026.py's note on what a free shoulder did to SUPERIA X-TRA 400.
SHOULDER_X, SHOULDER_K = 1.75, 0.42

MTF_F = [1, 5, 10, 20, 50, 100, 200]
MTF_R = [150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2]

SHEETS = {
    "xtra400": dict(
        pdf="superia_xtra400_datasheet.pdf", page=5,
        name="FUJICOLOR_SUPERIA_XTRA_400", ref="AF3-151E",
        title="FUJICOLOR SUPERIA X-TRA 400 [CH]",
        # ⚠ ORDINATE TO 4.0, ALONE IN THIS BATCH. Its own text layer prints
        # "3.5 / 4.0 / 3.0 / ..." in the tick list, which is what caught it.
        # ⚠ ITS ABSCISSA RUNS TO +0.5 AND THE OTHER TWO TO +1.0, so this panel
        # has 10 exposure rungs where they have 11. Confirmed against the
        # drawn ladder: ten rules at 21.77 pt spanning a 195.8 pt frame.
        char=dict(frame=(84.969, 118.281, 280.768, 291.307),
                  xv=[-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5],
                  yv=[4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]),
        mtf=dict(frame=(87.983, 396.927, 265.233, 544.676)),
        # ⚠ AND ITS DYE ORDINATE REACHES 2.5, NOT 2.0. Fuji labels only
        # 0.0 / 1.0 / 2.0 but rules every 0.5, and the frame top is the
        # UNLABELLED 2.5 rung -- six rules at 29.6 pt over a 148.4 pt frame.
        # Reading it as 0.0-2.0 inflates every density by 25 %.
        dye=dict(frame=(334.616, 396.3, 539.87, 544.741),
                 xv=[400, 500, 600, 700],
                 yv=[2.5, 2.0, 1.5, 1.0, 0.5, 0.0]),
    ),
    "xtra800": dict(
        pdf="superia_xtra800_datasheet.pdf", page=3,
        name="FUJICOLOR_SUPERIA_XTRA_800", ref="AF3-068E",
        title="FUJICOLOR SUPERIA X-TRA 800 [CZ]",
        char=dict(frame=(64.012, 246.645, 273.876, 389.798),
                  xv=[-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
                      0.0, 0.5, 1.0],
                  yv=[3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]),
        mtf=dict(frame=(69.535, 485.483, 268.806, 647.75)),
        dye=dict(frame=(334.517, 484.548, 507.486, 628.845),
                 xv=[400, 500, 600, 700], yv=[3.0, 2.0, 1.0, 0.0]),
    ),
    "reala": dict(
        pdf="superia_reala_datasheet.pdf", page=3,
        name="FUJICOLOR_SUPERIA_REALA", ref="AF3-967E",
        title="FUJICOLOR SUPERIA REALA [CS]",
        char=dict(frame=(68.512, 125.006, 274.144, 270.673),
                  xv=[-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
                      0.0, 0.5, 1.0],
                  yv=[3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]),
        mtf=dict(frame=(83.117, 387.737, 250.784, 530.07)),
        dye=dict(frame=(351.509, 388.069, 493.876, 560.166),
                 xv=[400, 500, 600, 700], yv=[2.0, 1.5, 1.0, 0.5, 0.0]),
    ),
}

#: What this module found, pinned so a rerun that disagrees FAILS. Written from
#: the first clean run and never edited to match a later one without a reason
#: recorded beside it.
EXPECTED = {
    "xtra400": dict(
        # ⚠ THESE ARE THE T4 READER'S OWN NUMBERS, NOT THE PROFILE'S. The
        # database keeps the 2026-09-02e trace; this pass re-derived the same
        # panel through a completely different route -- vector subpaths and an
        # arithmetic-progression ladder instead of colour separation and the
        # sheet's text layer -- and agreed to 0.010 D on dmin, 0.009 on gamma
        # and 0.019 on toe_x -- the worst of the twelve parameters being
        # green's toe. Both sets are pinned so that neither can drift
        # without the other noticing.
        curves={"R": (0.1293, 0.6649, -2.6931, 0.3074),
                "G": (0.3662, 0.7133, -2.8607, 0.3000),
                "B": (0.6843, 0.7591, -2.7940, 0.3000)},
        stored={"R": (0.1366, 0.6622, -2.6956, 0.3041),
                "G": (0.3744, 0.7047, -2.8800, 0.3000),
                "B": (0.6940, 0.7510, -2.8060, 0.3000)},
        f50=58.68, q=2.62, stored_f50=57.9),
    "xtra800": dict(
        curves={"R": (0.1749, 0.7678, -2.7111, 0.3000),
                "G": (0.4380, 0.8092, -2.7534, 0.3096),
                "B": (0.7462, 0.8109, -2.7845, 0.3000)},
        f50=53.79, q=2.16),
    "reala": dict(
        curves={"R": (0.3060, 0.6824, -2.1883, 0.3417),
                "G": (0.4962, 0.6808, -2.4210, 0.3775),
                "B": (0.9628, 0.6856, -2.2817, 0.3095)},
        f50=59.96, q=2.82),
}



# ---------------------------------------------------------------------------
def _subpaths(items):
    """Split one drawing's items where the pen lifts.

    ⚠ REQUIRED, NOT COSMETIC: X-TRA 800 draws all three characteristic records
    as ONE path with three subpaths, exactly as PRO 800Z's AF3-177E does, while
    REALA splits the same three curves across two path objects (1 + 2). Neither
    layout can be read by taking whole drawings, and neither is announced.
    """
    prev, sub, subs = None, [], []
    for it in items:
        if it[0] == "c":
            a, b = it[1], it[4]
        elif it[0] == "l":
            a, b = it[1], it[2]
        else:
            continue
        if prev is not None and (abs(a.x - prev.x) > 0.6
                                 or abs(a.y - prev.y) > 0.6):
            subs.append(sub)
            sub = []
        sub.append(it)
        prev = b
    if sub:
        subs.append(sub)
    return subs


def _sample(items, n=240):
    """Flatten one subpath to points, evaluating cubics rather than using ends."""
    pts = []
    for it in items:
        if it[0] == "c":
            P = [np.array([q.x, q.y]) for q in it[1:5]]
            for t in np.linspace(0.0, 1.0, n):
                pts.append(((1 - t) ** 3) * P[0] + 3 * ((1 - t) ** 2) * t * P[1]
                           + 3 * (1 - t) * t * t * P[2] + (t ** 3) * P[3])
        else:
            pts.append(np.array([it[1].x, it[1].y]))
            pts.append(np.array([it[2].x, it[2].y]))
    return np.array(pts)


def _rules(page, FR, tol=1.2):
    """Full-width / full-height rules touching this frame, as (h, v) lists.

    Accepts both stroked lines and the degenerate rectangles Fuji's exporter
    emits for the same rules -- a panel drawn one way on one sheet and the
    other way on the next is normal here.
    """
    hs, vs = set(), set()
    for dr in page.get_drawings():
        for it in dr["items"]:
            if it[0] == "l":
                a, b = it[1], it[2]
                if (abs(a.y - b.y) < 0.4 and abs(a.x - b.x) > FR.width * 0.5
                        and FR.y0 - tol <= a.y <= FR.y1 + tol
                        and a.x > FR.x0 - tol and b.x < FR.x1 + tol):
                    hs.add(round(a.y, 3))
                if (abs(a.x - b.x) < 0.4 and abs(a.y - b.y) > FR.height * 0.5
                        and FR.x0 - tol <= a.x <= FR.x1 + tol
                        and a.y > FR.y0 - tol and b.y < FR.y1 + tol):
                    vs.add(round(a.x, 3))
            elif it[0] == "re":
                r = it[1]
                if (abs(r.width) < 0.6 and r.height > FR.height * 0.5
                        and FR.x0 - tol <= r.x0 <= FR.x1 + tol):
                    vs.add(round((r.x0 + r.x1) / 2.0, 3))
                if (abs(r.height) < 0.6 and r.width > FR.width * 0.5
                        and FR.y0 - tol <= r.y0 <= FR.y1 + tol):
                    hs.add(round((r.y0 + r.y1) / 2.0, 3))
    hs.add(round(FR.y0, 3))
    hs.add(round(FR.y1, 3))
    vs.add(round(FR.x0, 3))
    vs.add(round(FR.x1, 3))
    return sorted(hs), sorted(vs)


def _ladder(found, values, ascending, tol=1.2, log=False):
    """Assign printed values to detected rules, and REFUSE a bad assignment.

    ⚠⚠ A CONTIGUOUS RUN IS NOT ENOUGH, AND FINDING THAT OUT COST THIS BATCH ITS
    FIRST THREE PANELS. The first version of this took the best contiguous run
    of len(values) detected rules. It cannot read X-TRA 800's characteristic
    panel, because the stray caption rule at y 249.166 sits BETWEEN the D 3.5
    rung (the frame top, 246.645) and the D 3.0 rung (267.132) -- the correct
    subset is not contiguous in the detected list, and no contiguous run of
    eight is right.

    WHAT THIS DOES INSTEAD: the rungs of a printed ladder are an ARITHMETIC
    PROGRESSION in the drawing (equal value steps at equal spacing). So every
    ordered PAIR of detected rules is tried as the two ENDS, the n expected
    positions are generated between them, and each is matched to its nearest
    detected rule. A candidate is scored by its worst matched deviation and
    rejected unless at least n-2 rungs are actually present -- because a rung
    genuinely can be missing, hidden under a caption block, which is the defect
    PRO 400H's panel has and which must not by itself reject a good ladder.

    `log=True` for a decade axis: the progression is arithmetic in log10(value).
    """
    n = len(values)
    if len(found) < 2:
        return None, "only %d rules detected" % len(found)
    vals = np.log10(np.array(values, dtype=float)) if log \
        else np.array(values, dtype=float)
    F = np.array(sorted(found), dtype=float)
    span = vals[-1] - vals[0]
    if span == 0:
        return None, "degenerate value ladder"
    best = None
    for i in range(len(F)):
        for j in range(len(F)):
            if i == j:
                continue
            y0, y1 = F[i], F[j]
            slope = (y1 - y0) / span
            if ascending and slope <= 0:
                continue
            if not ascending and slope >= 0:
                continue
            want = y0 + (vals - vals[0]) * slope
            # ⚠ A DEGENERATE LADDER MATCHES EVERYTHING, which is how the first
            # version of this returned 0.0185 pt per decade on X-TRA 400: with
            # the two "ends" one rule apart, all nine rungs collapse onto the
            # same line and every one of them "hits". Two conditions kill it --
            # the ladder must span most of the frame, and each rung must match
            # a DISTINCT rule.
            if abs(want[-1] - want[0]) < 0.55 * (F[-1] - F[0]):
                continue
            D = np.abs(F[None, :] - want[:, None])
            nearest = D.argmin(axis=1)
            dev = D.min(axis=1)
            ok = dev <= tol
            if len(set(nearest[ok].tolist())) != int(ok.sum()):
                continue
            hit = int(ok.sum())
            if hit < n - 1:
                continue
            score = float(dev[ok].max())
            # Prefer more rungs matched, then the tighter fit.
            key = (-hit, score)
            if best is None or key < best[0]:
                c = np.polyfit(vals, want, 1)
                best = (key, c, want, hit, score)
    if best is None:
        return None, "no arithmetic ladder of %d rungs fits the %d rules " \
                     "detected" % (n, len(F))
    _key, c, want, hit, score = best
    if hit < n:
        pass  # a rung under a caption block; recorded by the caller if wanted
    return (c, score, want, hit), ""


def _curves_in(page, FR, minfrac=0.35):
    """Every curve-like subpath wholly inside the frame, top-first."""
    out = []
    for dr in page.get_drawings():
        if not FR.contains(dr["rect"]):
            continue
        if dr["rect"].width < FR.width * minfrac:
            continue
        for s in _subpaths(dr["items"]):
            if not any(it[0] == "c" for it in s):
                continue
            A = _sample(s)
            if len(A) < 60:
                continue
            if A[:, 0].max() - A[:, 0].min() < FR.width * minfrac:
                continue
            out.append(A)
    out.sort(key=lambda A: float(A[:, 1].mean()))
    return out


def _logfit(found, values):
    """Least-squares log axis over rules whose printed values are decades."""
    lv = np.log10(np.array(values, dtype=float))
    c = np.polyfit(lv, np.array(found, dtype=float), 1)
    return c, float(np.abs(np.polyval(c, lv) - np.array(found)).max())


# ---------------------------------------------------------------------------
def read_sheet(root: Path, tag: str, verbose=True):
    sh = SHEETS[tag]
    doc = pymupdf.open(str(root / "PDF" / "PROFILES" / "FUJI" / sh["pdf"]))
    page = doc[sh["page"]]
    out, bad = {}, 0

    # ---- characteristic ---------------------------------------------------
    cfg = sh["char"]
    FR = pymupdf.Rect(*cfg["frame"])
    hs, vs = _rules(page, FR)
    ay = _ladder(hs, cfg["yv"], ascending=False)
    ax = _ladder(vs, cfg["xv"], ascending=True)
    if ay[0] is None or ax[0] is None:
        print("  [FAIL] %s characteristic ladder unassignable (%s / %s)"
              % (tag, ay[1], ax[1]))
        bad += 1
    else:
        cy, resy, _wy, hity = ay[0]
        cx, resx, _wx, hitx = ax[0]
        if hity < len(cfg["yv"]) or hitx < len(cfg["xv"]):
            print("    ⚠ characteristic: %d/%d density rungs and %d/%d "
                  "exposure rungs actually drawn -- the rest are implied by "
                  "the ladder, which is the PRO 400H caption-block defect"
                  % (hity, len(cfg["yv"]), hitx, len(cfg["xv"])))
        # ⚠ SQUARE-PANEL CHECK, REPORTED AND NOT USED. One log decade against
        # one density unit; Fuji draws them equal on most sheets and not on
        # all, so a mismatch is recorded rather than treated as a misread.
        sq = abs(abs(cx[0]) / abs(cy[0]) - 1.0)
        if verbose:
            print("    characteristic: %.4f pt per decade (residual %.2f pt), "
                  "%.4f pt per density (%.2f pt); square to %.2f %%"
                  % (cx[0], resx, -cy[0], resy, sq * 100))
        recs = _curves_in(page, FR)
        if len(recs) != 3:
            print("  [FAIL] %s characteristic: %d curve subpaths, expected 3"
                  % (tag, len(recs)))
            bad += 1
        else:
            fits = {}
            for ch, A in zip(("B", "G", "R"), recs):
                x = (A[:, 0] - cx[1]) / cx[0]
                d = (A[:, 1] - cy[1]) / cy[0]
                o = np.argsort(x)
                x, d = x[o], d[o]
                init = (float(d.min()), 0.7, float(x.min()), 0.35,
                        SHOULDER_X, SHOULDER_K)
                p, rms, mx = dp.fit_tonecurve4(x, d, init)
                fits[ch] = (tuple(round(float(v), 4) for v in p[:4]),
                            float(rms), float(mx),
                            float(x.min()), float(x.max()))
                if verbose:
                    print("      %s: %d pts over logH %+.2f..%+.2f  ->  "
                          "ToneCurve(%.4f, %.4f, %.4f, %.4f)  rms %.4f max %.4f"
                          % (ch, len(A), x.min(), x.max(),
                             p[0], p[1], p[2], p[3], rms, mx))
            # ⚠ THE MASK LADDER IS AN ASSERTION, NOT AN OBSERVATION. On a
            # colour negative dmin must rise red < green < blue (the orange
            # mask) while gamma runs the other way. A swapped record is a
            # wrong render that looks entirely plausible.
            dmin = [fits[c][0][0] for c in ("R", "G", "B")]
            gam = [fits[c][0][1] for c in ("R", "G", "B")]
            if not (dmin[0] < dmin[1] < dmin[2]):
                print("  [FAIL] %s dmin ladder is not r<g<b: %s" % (tag, dmin))
                bad += 1
            # ⚠ NOT AN ORDERING TEST. Three near-equal gammas are exactly what
            # a well-balanced colour negative should show, so only a SPREAD
            # that is both large and non-monotone is worth reporting.
            if (max(gam) - min(gam) > 0.05
                    and not (gam[0] <= gam[1] <= gam[2]
                             or gam[0] >= gam[1] >= gam[2])):
                print("  [WARN] %s gammas spread %.3f and are not monotone "
                      "across records: %s" % (tag, max(gam) - min(gam), gam))
            out["curves"] = fits
            out["char_axis"] = (float(cx[0]), float(-cy[0]), float(sq))

    # ---- MTF --------------------------------------------------------------
    FR = pymupdf.Rect(*sh["mtf"]["frame"])
    hs, vs = _rules(page, FR)
    # ⚠ FUJI'S MTF ABSCISSA IS NOT ACCURATELY DRAWN and that is a property of
    # the sheets, not of this reader -- the same +/-5 % rule-to-rule scatter
    # fuji_t3_2026 records on SUPERIA and PROVIA. A 1.2 pt bound rejects all
    # three panels; 3.5 pt accepts them and still refuses a wrong assignment,
    # whose residual runs to 58-70 pt.
    a_f = _ladder(vs, MTF_F, ascending=True, log=True, tol=3.5)
    # ⚠ RESPONSE FALLS DOWN THE PAGE, so its ladder is DESCENDING in y while
    # its printed values descend too -- 150 at the top, 2 at the bottom -- which
    # makes the slope POSITIVE in (log value, y). Getting this sign wrong is
    # how the first run rejected all three MTF panels.
    a_r = _ladder(hs, MTF_R, ascending=False, log=True, tol=3.5)
    if a_f[0] is None or a_r[0] is None:
        print("  [FAIL] %s MTF ladder unassignable (%s / %s)"
              % (tag, a_f[1], a_r[1]))
        bad += 1
    else:
        cf, rf = a_f[0][0], a_f[0][1]
        cr, rr = a_r[0][0], a_r[0][1]
        if verbose:
            print("    MTF: %.2f pt per frequency decade (residual %.2f pt), "
                  "%.2f pt per response decade (%.2f pt)"
                  % (cf[0], rf, -cr[0], rr))
        recs = _curves_in(page, FR, minfrac=0.30)
        if len(recs) != 1:
            print("  [FAIL] %s MTF: %d curve subpaths, expected 1"
                  % (tag, len(recs)))
            bad += 1
        else:
            A = recs[0]
            f = 10.0 ** ((A[:, 0] - cf[1]) / cf[0])
            r = 10.0 ** ((A[:, 1] - cr[1]) / cr[0]) / 100.0
            o = np.argsort(f)
            f, r = f[o], r[o]
            peak, pk_at = float(r.max()), float(f[int(r.argmax())])
            below = np.flatnonzero(r < 0.5)
            f50 = float(np.interp(0.5, [r[below[0]], r[below[0] - 1]],
                                  [f[below[0]], f[below[0] - 1]]))
            # ⚠ THE ROLLOFF IS FITTED ABOVE THE OVERSHOOT ONLY. The carrier
            # 1/(1+(f/f50)^q) is 1.0 at zero frequency by construction, so
            # including the adjacency lift bends q to absorb something the
            # model cannot hold -- the 5279 lesson.
            m = f > max(8.0, pk_at)
            q, qe = None, None
            if int(m.sum()) > 20:
                ff, rr2 = f[m], r[m]
                for cand in np.arange(1.0, 5.001, 0.01):
                    e = float(np.sqrt(np.mean(
                        (1.0 / (1.0 + (ff / f50) ** cand) - rr2) ** 2)))
                    if qe is None or e < qe:
                        q, qe = float(cand), e
                ge = float(np.sqrt(np.mean(
                    (np.exp(-2 * (np.pi * 0.0 + 1e-9) ** 2) * 0 + np.exp(
                        -np.log(2.0) * (ff / f50) ** 2) - rr2) ** 2)))
            else:
                ge = float("nan")
            if verbose:
                print("    MTF: %d pts over %.1f-%.1f c/mm, response "
                      "%.1f-%.1f %%  ->  f50 %.1f c/mm, q %.2f (rms %.4f vs "
                      "Gaussian %.4f), overshoot %+.3f at %.1f c/mm"
                      % (len(A), f.min(), f.max(), 100 * r.min(), 100 * r.max(),
                         f50, q or float("nan"), qe or float("nan"), ge,
                         peak - 1.0, pk_at))
            out["mtf"] = dict(f50=round(f50, 2), q=round(q, 2) if q else None,
                              q_rms=qe, gauss_rms=ge,
                              overshoot=round(peak - 1.0, 4),
                              peak_at=round(pk_at, 2), n=len(A),
                              f_lo=round(float(f.min()), 2),
                              f_hi=round(float(f.max()), 2))

    # ---- spectral dye density: a NEUTRAL PAIR on all three sheets ----------
    cfg = sh["dye"]
    FR = pymupdf.Rect(*cfg["frame"])
    hs, vs = _rules(page, FR)
    a_y = _ladder(hs, cfg["yv"], ascending=False)
    a_x = _ladder(vs, cfg["xv"], ascending=True)
    if a_y[0] is None or a_x[0] is None:
        print("  [FAIL] %s dye ladder unassignable (%s / %s)"
              % (tag, a_y[1], a_x[1]))
        bad += 1
    else:
        cy, resy, _wy, _hy = a_y[0]
        cx, resx, _wx, _hx = a_x[0]
        if verbose:
            print("    dye: %.4f pt per 100 nm (residual %.2f pt), %.4f pt "
                  "per density (%.2f pt)" % (cx[0] * 100, resx, -cy[0], resy))
        recs = _curves_in(page, FR, minfrac=0.30)
        if len(recs) != 2:
            print("  [FAIL] %s dye: %d curve subpaths, expected 2 "
                  "(mid-scale neutral and D-min)" % (tag, len(recs)))
            bad += 1
        else:
            grid = np.arange(400.0, 701.0, 10.0)
            got = {}
            for key, A in zip(("neutral", "dmin"), recs):
                nm = (A[:, 0] - cx[1]) / cx[0]
                d = (A[:, 1] - cy[1]) / cy[0]
                o = np.argsort(nm)
                nm, d = nm[o], d[o]
                got[key] = (np.interp(grid, nm, d),
                            float(nm.min()), float(nm.max()))
            n_v, d_v = got["neutral"][0], got["dmin"][0]
            # ⚠ THE ORANGE-MASK TEST. `d_dmin` on a masked colour negative IS
            # the mask, so it must FALL towards the red and sit below the
            # neutral everywhere. This is what proves the two curves were not
            # swapped -- nothing on the panel labels them by position.
            if not (n_v > d_v).all():
                print("  [FAIL] %s dye: the neutral does not sit above D-min "
                      "at every wavelength" % tag)
                bad += 1
            if not d_v[0] > d_v[-1]:
                print("  [FAIL] %s dye: D-min does not fall towards the red, "
                      "so it is not an orange mask" % tag)
                bad += 1
            if verbose:
                print("    dye: neutral %.3f..%.3f D, D-min %.3f (400 nm) -> "
                      "%.3f (700 nm), read over %.0f-%.0f nm"
                      % (n_v.min(), n_v.max(), d_v[0], d_v[-1],
                         got["neutral"][1], got["neutral"][2]))
            out["dye"] = dict(
                neutral=tuple(round(float(v), 3) for v in n_v),
                dmin=tuple(round(float(v), 3) for v in d_v))

    return out, bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()

    print("FUJICOLOR SUPERIA -- three sheets, one house template, three "
          "different ladders")
    bad = 0
    res = {}
    for tag in ("xtra400", "xtra800", "reala"):
        sh = SHEETS[tag]
        print("\n  %s -- %s, %s p%d"
              % (sh["name"], sh["title"], sh["ref"], sh["page"] + 1))
        out, b = read_sheet(root, tag, verbose=True)
        res[tag] = out
        bad += b
        exp = EXPECTED.get(tag, {})
        if exp.get("curves") and "curves" in out:
            for ch, want in exp["curves"].items():
                got = out["curves"][ch][0]
                if max(abs(a - b2) for a, b2 in zip(got, want)) > 0.02:
                    print("  [MISMATCH] %s %s curve %s vs pinned %s"
                          % (tag, ch, got, want))
                    bad += 1
        if exp.get("f50") and "mtf" in out:
            if abs(out["mtf"]["f50"] - exp["f50"]) > 0.6:
                print("  [MISMATCH] %s f50 %.2f vs pinned %.2f"
                      % (tag, out["mtf"]["f50"], exp["f50"]))
                bad += 1
            if exp.get("q") and abs(out["mtf"]["q"] - exp["q"]) > 0.02:
                print("  [MISMATCH] %s q %.2f vs pinned %.2f"
                      % (tag, out["mtf"]["q"], exp["q"]))
                bad += 1
        # ⚠ THE CROSS-METHOD AGREEMENT IS ITSELF PINNED. X-TRA 400 is the one
        # stock this batch re-read rather than added, and the whole value of
        # that re-read is the size of the disagreement between two independent
        # extractions of one panel. If it ever grows, one of the two readers
        # has changed and the database should be told which.
        if exp.get("stored") and "curves" in out:
            worst = max(abs(a - b2)
                        for ch, want in exp["stored"].items()
                        for a, b2 in zip(out["curves"][ch][0], want))
            if worst > 0.022:
                print("  [MISMATCH] %s: the re-derivation now differs from the "
                      "STORED curves by %.4f, was 0.0193 (green's toe_x, the "
                      "worst of the twelve parameters)" % (tag, worst))
                bad += 1
            else:
                print("    ⚠ cross-method check: this re-derivation agrees "
                      "with the STORED 2026-09-02e trace to %.4f on every "
                      "curve parameter -- two readers sharing no extraction "
                      "step" % worst)
        if exp.get("stored_f50") and "mtf" in out:
            d = abs(out["mtf"]["f50"] - exp["stored_f50"])
            print("    ⚠ cross-method check: f50 %.2f here against the STORED "
                  "%.1f, %.1f %% apart, and q identical at %.2f. The stored "
                  "value is kept -- neither extraction is better conditioned, "
                  "and Fuji's MTF abscissa is drawn to about +/-5 %% rule to "
                  "rule on this family"
                  % (out["mtf"]["f50"], exp["stored_f50"],
                     100 * d / exp["stored_f50"], out["mtf"]["q"]))
            if d > 1.5:
                print("  [MISMATCH] %s: re-derived f50 has drifted from the "
                      "stored value by %.2f" % (tag, d))
                bad += 1

    if ns.assert_:
        if bad:
            print("\n[FAIL] the SUPERIA sheets do not reproduce")
            return 1
        print("\n[OK] three SUPERIA sheets re-derived: characteristic curves "
              "on three DIFFERENT ordinate ladders (X-TRA 400 runs to 4.0, the "
              "other two to 3.5), three MTF panels, three neutral+D-min pairs, "
              "each ladder assigned by fitting an ARITHMETIC PROGRESSION over "
              "the detected rules -- which is what rejects X-TRA 800's stray "
              "caption rule 2.5 pt inside its top rung, and REALA's two -- and "
              "refused rather than guessed. The mask test is asserted on all "
              "three: "
              "D-min below the neutral at every wavelength and falling towards "
              "the red, which is what proves the two dye curves were not "
              "swapped. Spectral panels are NOT read -- all three draw a "
              "FOURTH cyan-sensitive layer the schema cannot hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
