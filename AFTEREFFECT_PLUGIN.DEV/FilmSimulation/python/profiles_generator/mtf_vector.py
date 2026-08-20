"""f50 and the adjacency overshoot from a VECTOR log-log MTF plot.

WHY
---
`MTFSpec.f50_*` is the sharpness parameter the renderer actually uses, and for
most stocks it is an estimate. Kodak prints the curve it comes from -- and on some
sheets prints it as VECTOR art, where the answer can be read rather than guessed.

The EASTMAN PLUS-X 5231 sheet (H-1-5231, February 1999) is the case this was
written for: page 3 carries the modulation-transfer curve (plot F002_0141AC) as a
single bezier path, and the whole page contains ZERO embedded images. E0's
re-verification of that profile found the sheet prints no numeric MTF value, so
the stored f50 could not be confirmed from text -- but it can be measured from the
path.

WHAT IT MEASURES, and the one thing that is NOT f50
---------------------------------------------------
  * f50: the frequency at which response falls back through 50 %. Taken at the
    LAST crossing, because the curve rises ABOVE 100 % at low frequency (see
    below) and a naive first-crossing search on a non-monotone curve can return
    the wrong branch.
  * the ADJACENCY OVERSHOOT: the peak response above unity, which is the
    development edge effect. `MTFSpec.adjacency` is documented as exactly that
    fraction, so the plot measures it directly.

⚠ THE OVERSHOOT'S FREQUENCY IS NOT `adjacency_um`. On 5231 the peak sits near
4-5 cycles/mm, a spatial scale of order 100-200 um, while the stored
`adjacency_um` is 16.0 (which corresponds to ~60 cycles/mm). The same
inconsistency appears on FUJI_F125_8530, whose Honjo-1989 overshoot peaks near
9 cycles/mm against a stored 13.0 um. Either the field means something narrower
than the overshoot period or the values are wrong; that depends on how the
renderer defines it, so this script REPORTS the peak frequency and changes
nothing. Recorded rather than resolved.

AXES: both are logarithmic, and both are least-squares fitted over every printed
decade and mantissa label with a residual test -- the same discipline as
dye_density.py and granularity_vector.py, for the same reason (a two-point span
cannot detect a misplaced label). 5231 gives 11 frequency ticks and 12 response
ticks, fitting to 0.66 and 0.82 pt.

Run:
    python mtf_vector.py --root ../..
    python mtf_vector.py --root ../.. --assert
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

#: tag -> (pdf under PDF/PROFILES/KODAK, page, profile, frame hint x0,x1,y0,y1)
SHEETS = {
    "5231": ("5231-PLUS-X.pdf", 3, "EASTMAN_PLUS_X_5231", (87, 289, 293, 446)),
    # 2026-08-20, queue C2b's first addition. H-1-5201 p3 prints the same plot
    # type for a COLOUR negative, so it carries THREE curves -- one per record --
    # where 5231 (a black-and-white stock) carries one. That is the whole point:
    # MTFSpec has three f50 fields and until now every colour stock's three were
    # estimates in a fixed ratio.
    # ⚠ THE RED RECORD IS DRAWN TWICE, yellow under magenta, exactly as on the
    # same sheet's granularity panel. Handled by ink, see pick_curves().
    "5201": ("Kodak VISION2 50D 5201.pdf", 3, "KODAK_VISION2_50D_5201",
             (224, 350.5, 205, 286)),
    # 2026-08-20, added on the owner's re-upload of the four named PDFs. Three of
    # the four turned out to be documents the corpus already held -- V200T.pdf is
    # BYTE-IDENTICAL to 5274.pdf (md5 cf07db7d...) -- but its MTF panel had never
    # been traced, and 5274's stored f50 triple was an estimate.
    # ⚠ THIS SHEET LETTERS ITS CURVES INSTEAD OF COLOURING THEM. All three are
    # black; R / G / B are printed inside the frame's right edge. See
    # letter_assign().
    "5274": ("5274.pdf", 3, "KODAK_VISION_200T_5274",
             (362.7, 565.0, 155.3, 308.3)),
}

#: Measured 2026-08-18/20. --assert fails if a sheet stops reproducing these.
#: A colour sheet pins one entry per record; a mono sheet pins the single curve
#: under the key "-".
EXPECTED = {
    "5231": {"-": dict(f50=41.3, peak=1.034, peak_at=4.6)},
    "5201": {
        "R": dict(f50=32.1, peak=1.108, peak_at=2.5),
        "G": dict(f50=49.7, peak=1.157, peak_at=10.7),
        "B": dict(f50=55.5, peak=1.142, peak_at=12.7),
    },
    # ⚠ MEASURED AND PINNED, BUT NOT ADOPTED INTO THE PROFILE -- the owner has not
    # approved changing an existing stock's sharpness. 5274 stores the ESTIMATE
    # 56.0 / 64.0 / 72.0 with adjacency 0.09. Measured: 35.4 / 68.8 / 74.0 with a
    # green overshoot of 0.162.
    #   GREEN AND BLUE AGREE WELL (68.8 vs 64, 74.0 vs 72 -- within 8 %). THE RED
    # RECORD DOES NOT: 35.4 measured against 56.0 stored, i.e. the estimate is
    # 1.58x TOO SHARP in red. That is the same defect the estimating rule has by
    # construction -- it scales one number by a fixed layer-order ratio, and the
    # real red deficit on a colour negative is much larger than that ratio allows.
    # 5201 says the same thing independently: its measured spread is 32.1/49.7/55.5,
    # a red:blue ratio of 0.58 where the stored rule assumes about 0.78.
    # Recorded here so the reading is re-derived on every build and cannot drift
    # while the decision is pending, exactly as the 5219 brochure cross-check is.
    "5274": {
        "R": dict(f50=35.4, peak=1.027, peak_at=2.4),
        "G": dict(f50=68.8, peak=1.162, peak_at=11.0),
        "B": dict(f50=74.0, peak=1.234, peak_at=16.1),
    },
}
TOL_F, TOL_P = 1.0, 0.01

TICK_RESID_PT = 1.5


def logfit(pairs, label, min_keep=6):
    """{decade value: pixel} -> (px per decade, intercept, residual, n).

    ⚠ ONE LABEL CAN BE MISPLACED, and on H-1-5274's MTF panel one is. Its
    response axis prints 1 2 3 5 7 10 20 30 50 70 100 **150**; the first eleven
    give 66.88 and 66.76 pt per decade -- agreeing to 0.2 % -- while "150" sits
    7.6 pt off the line they define. The axis is clipped at the frame top and the
    label was set at the frame edge rather than at its own value. A fit over all
    twelve is not collinear at 5.94 pt and refuses the sheet.
    Same outlier rejection as granularity_vector.fit(), and for the same reason:
    the give-away is that the SURVIVING ticks agree to a fraction of a point.
    Rejection stops while `min_keep` remain, so a sparse axis cannot be whittled
    down to a fabricated line. Verified: 5231 and 5201 drop nothing.
    """
    v = np.array([np.log10(k) for k in sorted(pairs)])
    px = np.array([pairs[k] for k in sorted(pairs)])
    keep = np.ones(len(v), bool)
    dropped = []
    while True:
        A = np.vstack([v[keep], np.ones(keep.sum())]).T
        m, c = np.linalg.lstsq(A, px[keep], rcond=None)[0]
        res = np.abs(m*v + c - px)
        worst = int(np.argmax(np.where(keep, res, -1.0)))
        if res[worst] <= TICK_RESID_PT or keep.sum() <= min_keep:
            break
        keep[worst] = False
        dropped.append((10.0**float(v[worst]), float(res[worst])))
    if dropped:
        print("    %s: DROPPED %s" % (label, ", ".join(
            "%g (%.2f pt off)" % d for d in dropped)))
    v, px = v[keep], px[keep]
    A = np.vstack([v, np.ones(len(v))]).T
    m, c = np.linalg.lstsq(A, px, rcond=None)[0]
    res = float(np.abs(m*v + c - px).max())
    if res > TICK_RESID_PT:
        raise SystemExit(f"[!] {label}: ticks not collinear, {res:.2f} pt "
                         f"over {len(v)}")
    return m, c, res, len(v)


def flatten(items, n=40):
    pts = []
    for it in items:
        if it[0] == "c":
            P = [it[1], it[2], it[3], it[4]]
            for k in range(n+1):
                t = k/n
                u = 1.0-t
                pts.append((
                    u**3*P[0].x + 3*u*u*t*P[1].x + 3*u*t*t*P[2].x + t**3*P[3].x,
                    u**3*P[0].y + 3*u*u*t*P[1].y + 3*u*t*t*P[2].y + t**3*P[3].y))
        elif it[0] == "l":
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
    return pts


IDEAL = {"R": (1.0, 0.0, 0.0), "G": (0.0, 1.0, 0.0), "B": (0.0, 0.0, 1.0)}

#: samples below this carry the adjacency overshoot, which is a separate effect
#: modelled separately; including them bends the rolloff to absorb a lift. The
#: same 8 cycles/mm cut C2 used on 5231.
ROLLOFF_FROM = 8.0


def score_carrier(f, r, f50, f_from):
    """Score the adopted power-law rolloff against the legacy Gaussian.

    C2 chose `1/(1+(f/f50)^q)` over `exp(-ln2 (f/f50)^2)` on ONE traced curve and
    said so in the result entry: "the one-curve basis is the weakest part of
    today's choice". Queue item C2b is to trace more and re-score. So every curve
    this file reads now reports the same comparison, in the same units, rather
    than leaving the re-scoring to a future ad-hoc script.

    Both forms pass through 0.5 at f50 by construction, so this compares SHAPE
    away from f50 and nothing else.
    """
    # ⚠ THE CUT IS THE OVERSHOOT PEAK, NOT A FIXED 8 cycles/mm. C2's 8 came from
    # 5231, whose overshoot peaks at 4.7 cycles/mm, so 8 was safely above it. On
    # 5201 the green record peaks at 10.7 and the blue at 12.7, so a fixed 8
    # leaves the lift inside the fitted band and the power law scores rms 0.095 --
    # a number that says nothing about the carrier and everything about fitting a
    # rolloff through an overshoot.
    m = f >= f_from
    if m.sum() < 6:
        return (f"rolloff: fewer than 6 samples above {f_from:.1f} cycles/mm "
                f"-- not scored")
    x = f[m] / f50
    y = r[m]
    gauss = np.exp(-np.log(2.0) * x**2)
    rms_g = float(np.sqrt(np.mean((gauss - y)**2)))
    best_q, best_r = None, None
    for q in np.arange(0.60, 6.001, 0.005):
        e = float(np.sqrt(np.mean((1.0/(1.0 + x**q) - y)**2)))
        if best_r is None or e < best_r:
            best_q, best_r = float(q), e
    return (f"rolloff over {int(m.sum())} samples >= {f_from:.1f} "
            f"cycles/mm: power law q = {best_q:.2f} at rms {best_r:.4f}, "
            f"Gaussian rms {rms_g:.4f} ({rms_g/best_r:.1f}x worse)")


def letter_assign(pg, cand, fx0, fx1, fy0, fy1):
    """Record identity from PRINTED R / G / B letters, as an exhaustive bijection.

    ⚠ KODAK PRINTS THE RECORD TWO DIFFERENT WAYS on the same plot type, and this
    is the second. The 2005 brochures state it in INK (see pick_curves); the 1997
    technical sheets draw all three curves in BLACK and letter them -- H-1-5274 p3
    puts R / G / B at x 494-500 inside the frame's right edge. A grey-ink sheet is
    therefore not necessarily a one-curve sheet, which is what the old
    "all grey -> take the thickest" branch assumed.

    Returns None -- and falls back to the single-curve rule -- unless the frame
    carries exactly one of each letter, the three letters are STACKED at one
    abscissa (which is what makes vertical order meaningful), and the resulting
    letter-to-curve map is a bijection.
    """
    letters = {}
    for a, b, c, d, t, *_ in pg.get_text("words"):
        if t not in ("R", "G", "B"):
            continue
        cx, cy = (a+c)/2.0, (b+d)/2.0
        if not (fx0-6 <= cx <= fx1+6 and fy0-6 <= cy <= fy1+6):
            continue
        letters.setdefault(t, []).append((cx, cy))
    if sorted(letters) != ["B", "G", "R"] or any(len(v) != 1
                                                for v in letters.values()):
        return None
    lab = {k: v[0] for k, v in letters.items()}
    # ⚠ THE THREE LETTERS ARE STACKED AT ONE x, so a nearest-point distance is
    # nearly the same for all three curves and an "is the winner clearly better
    # than the runner-up" gate refuses every time -- which it did. What the sheet
    # actually states is VERTICAL ORDER at the letters' abscissa, so each letter
    # is matched to the curve whose height AT THAT x is closest to it, and the
    # result must still be a bijection.
    xs = [c[0] for c in lab.values()]
    if max(xs) - min(xs) > 0.10 * (fx1 - fx0):
        return None                      # not a stacked legend -- refuse
    out, used = {}, set()
    for rec, (lx, ly) in lab.items():
        best, bi = None, None
        for i, (pts, *_rest) in enumerate(cand):
            px = [q[0] for q in pts]
            if not (min(px) - 2 <= lx <= max(px) + 2):
                continue
            order = sorted(pts)
            y = float(np.interp(lx, [q[0] for q in order],
                                [q[1] for q in order]))
            if best is None or abs(y - ly) < best:
                best, bi = abs(y - ly), i
        if bi is None or bi in used:
            return None                  # unreachable or double-claimed
        used.add(bi)
        out[rec] = cand[bi][0]
    return out


def pick_curves(pg, fx0, fx1, fy0, fy1):
    """The response curves inside the frame, keyed by record.

    ⚠ THE OLD RULE WAS "the thickest long path", and it only ever had to pick one
    curve out of one. A colour sheet draws three, at identical width, and prints
    the record in INK -- the same convention granularity_vector.colour_assign()
    reads on the brochures. It also draws the red record TWICE, once in yellow and
    once in magenta on top, so a naive by-colour grouping yields four curves and
    two of them are the same measurement.

    Returned as {record: points}. A sheet whose ink is black (5231) yields
    {"-": points} and is measured exactly as before -- verified: it reproduces
    f50 41.3 and the 3.4 % overshoot to the digit.
    """
    cand = []
    for p in pg.get_drawings():
        r = p["rect"]
        if not (fx0-3 <= r.x0 and r.x1 <= fx1+3
                and fy0-3 <= r.y0 and r.y1 <= fy1+3):
            continue
        n_it = sum(1 for it in p["items"] if it[0] in ("l", "c"))
        if n_it < 1:
            continue
        pts = flatten(p["items"])
        if len(pts) < 8:
            continue
        # a curve crosses a useful part of the frame; a tick or a legend rule
        # does not. 20 % of the frame width, the same floor granularity_vector
        # uses, and it is what rejects the three legend swatches on 5201.
        if max(x for x, _ in pts) - min(x for x, _ in pts) < 0.20*(fx1-fx0):
            continue
        col = p.get("color")
        col = tuple(round(float(c), 3) for c in col) if col else None
        cand.append((pts, col, n_it, p.get("width") or 0.0))
    if not cand:
        return {}
    # mono sheet: everything is black ink, so there is no per-record identity to
    # read. ⚠ THE ORIGINAL RULE IS KEPT VERBATIM HERE -- >= 8 items, then the
    # THICKEST path -- and it has to be. Relaxing the item floor to 1 (which the
    # colour sheet needs, its curves being 2-4 beziers) and picking the longest
    # point list instead put 5231's f50 at 607.8 cycles/mm: the log grid is one
    # path with far more points than the curve, and it "falls through 50 %" at
    # the frame's right edge. A regression that only shows up as a plausible
    # number is exactly what EXPECTED exists to catch, and it did.
    def is_grey(c):
        return c is None or (max(c) - min(c) < 0.12)
    if all(is_grey(c) for _, c, _, _ in cand):
        thick = [t for t in cand if t[2] >= 8]
        if not thick:
            return {}
        # ⚠ GREY INK DOES NOT MEAN ONE CURVE. Try the printed letters first; only
        # a frame with no usable R / G / B triple falls back to the single-curve
        # rule, which is what 5231 (a black-and-white stock) needs.
        if len(thick) == 3:
            byletter = letter_assign(pg, thick, fx0, fx1, fy0, fy1)
            if byletter is not None:
                return byletter
        return {"-": max(thick, key=lambda t: t[3])[0]}
    cand = [(pts, col) for pts, col, _, _ in cand]
    out = {}
    for pts, col in cand:
        if is_grey(col):
            continue
        rec = min(IDEAL, key=lambda t: sum((col[k]-IDEAL[t][k])**2
                                           for k in range(3)))
        # yellow and magenta both land on R; keep the denser of the two, which is
        # the same overprint collapse the granularity extractor does
        if rec not in out or len(pts) > len(out[rec]):
            out[rec] = pts
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()
    import pymupdf

    bad = 0
    for tag, (fn, pgno, prof, (fx0, fx1, fy0, fy1)) in SHEETS.items():
        pdf = Path(ns.root).resolve() / "PDF" / "PROFILES" / "KODAK" / fn
        if not pdf.is_file():
            print(f"  [SKIP] {tag}: source not present: {fn}")
            continue
        pg = pymupdf.open(pdf)[pgno-1]
        xs, ys = {}, {}
        for a, b, c, d, t, *_ in pg.get_text("words"):
            if not re.fullmatch(r'\d+', t):
                continue
            v = float(t)
            cx, cy = (a+c)/2.0, (b+d)/2.0
            if fx0-12 <= cx <= fx1+12 and fy1 < cy <= fy1+22:
                xs[v] = cx                       # spatial frequency, below
            elif fx0-26 <= cx < fx0-1 and fy0-8 <= cy <= fy1+8:
                ys[v] = cy                       # response %, left
        if len(xs) < 4 or len(ys) < 4:
            print(f"  [FAIL] {tag}: ticks x={len(xs)} y={len(ys)}")
            bad += 1
            continue
        fx = logfit(xs, "spatial frequency")
        fy = logfit(ys, "response")
        got = pick_curves(pg, fx0, fx1, fy0, fy1)
        if not got:
            print(f"  [FAIL] {tag}: no curve inside the frame")
            bad += 1
            continue
        print(f"[i] {fn} p{pgno} -> {prof}")
        print(f"    freq axis {fx[0]:.2f} px/decade, residual {fx[2]:.2f} pt, "
              f"{fx[3]} ticks; response axis {abs(fy[0]):.2f} px/decade, "
              f"residual {fy[2]:.2f} pt, {fy[3]} ticks")
        pins = EXPECTED.get(tag, {})
        for rec in sorted(got, key=lambda k: "RGB-".index(k)):
            a = np.array(got[rec])
            f = 10.0 ** ((a[:, 0] - fx[1]) / fx[0])
            r = 10.0 ** ((a[:, 1] - fy[1]) / fy[0]) / 100.0
            o = np.argsort(f)
            f, r = f[o], r[o]
            # f50 at the LAST downward crossing of 0.5
            above = np.where(r >= 0.5)[0]
            if not len(above) or above[-1]+1 >= len(f):
                print(f"    [FAIL] {rec}: the curve never falls through 50 %")
                bad += 1
                continue
            i = above[-1]
            f50 = float(np.interp(0.5, [r[i+1], r[i]], [f[i+1], f[i]]))
            pk = int(np.argmax(r))
            print(f"    {rec}: {f.min():.1f}-{f.max():.1f} cycles/mm, response "
                  f"{r.min()*100:.1f}-{r.max()*100:.1f} %  ->  f50 = "
                  f"{f50:.1f} cycles/mm, overshoot {r[pk]-1.0:+.3f} "
                  f"(peak at {f[pk]:.1f} cycles/mm)")
            print("      " + score_carrier(
                f, r, f50, max(ROLLOFF_FROM, float(f[pk]))))
            w = pins.get(rec)
            if w:
                if abs(f50 - w["f50"]) > TOL_F:
                    print(f"    [FAIL] {rec} f50 moved: {f50:.1f} vs recorded "
                          f"{w['f50']:.1f}")
                    bad += 1
                if abs((r[pk]-1.0) - (w["peak"]-1.0)) > TOL_P:
                    print(f"    [FAIL] {rec} overshoot moved: {r[pk]:.3f} vs "
                          f"recorded {w['peak']:.3f}")
                    bad += 1
        missing = set(pins) - set(got)
        if missing:
            print(f"    [FAIL] records pinned but not found: {sorted(missing)}")
            bad += 1
        print("    (the overshoot FREQUENCY is reported, not stored; see the "
              "module note on adjacency_um)")
    print()
    if bad:
        print(f"[FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("[OK] MTF read from the sheet's vector path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
