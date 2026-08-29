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

C2b, 2026-08-23: THE COLOUR BATCH, AND FOUR DEFECTS IT FOUND IN THIS SCRIPT
--------------------------------------------------------------------------
Eight more Kodak colour sheets and the one non-Kodak MTF sheet in the corpus were
read. Every defect below produced PLAUSIBLE NUMBERS, which is the only reason
they are worth recording:

  1. ONE PATH, THREE CURVES. The 1990s technical sheets emit all three records as
     a single path object -- the same hazard the granularity panels have. Read as
     one curve, H-1-5218 gave "f50 69.7" off a trace that walks along blue, jumps
     to green and finishes on red. Fixed by splitting on
     `granularity_vector.subpaths` (imported, not copied).
  2. THE LOG GRID PASSES FOR A CURVE. On 5245 and 5246 the grid is one connected
     polyline; the letter matcher handed it back as the green record (f50 236.8,
     response to 190 %). Fixed by three shape tests -- single-valued in frequency,
     a real vertical extent, and near-monotone descent (total variation over span
     <= 2.0, measured 1.00-1.35 on every real curve and 12.9 on the grid).
  3. GREEDY LABEL MATCHING DOUBLE-CLAIMS, and ranking by height at a common
     abscissa gets the order wrong when one record stops earlier than another
     (5248 red stops at 115 cycles/mm, green runs to 191). Fixed by solving the
     3x3 assignment over all six permutations.
  4. A FRAGMENT HAS AN f50 AND IT IS MEANINGLESS. 5293's red survives as a
     30-125 cycles/mm arc starting at 53 % response and reports 32.0. Now refused
     with a stated reason instead of measured.

And the extractor gained the gate the other two plot readers already had:
`--overlay` draws every traced point back onto the page. All four defects above
were found by looking at it.

Run:
    python mtf_vector.py --root ../..
    python mtf_vector.py --root ../.. --assert
    python mtf_vector.py --root ../.. --overlay /tmp/ov     # LOOK AT THIS
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# The subpath splitter, imported rather than copied: one definition (C2b).
import granularity_vector as gv

#: tag -> (pdf under PDF/PROFILES -- bare name = KODAK, else "MAKER/name.pdf",
#: page, profile, frame hint x0,x1,y0,y1)
SHEETS = {
    # 2026-08-25, queue E0b-orig: the only colour REVERSAL sheet in the corpus
    # whose MTF panel is vector art. Frame read off the page rather than guessed;
    # the tuple order is (x0, x1, y0, y1), which is not the order PyMuPDF's Rect
    # prints in and is worth stating because getting it wrong finds no curves.
    "5285": ("Ektachrome_100d.pdf", 3, "KODAK_EKTACHROME_100D_5285",
             (362.6, 565.1, 348.3, 501.3)),
    "5231": ("5231-PLUS-X.pdf", 3, "EASTMAN_PLUS_X_5231", (87, 289, 293, 446)),
    # 2026-08-26, owner-supplied. H-1-5222 (July 2015) -- the FULL Kodak sheet
    # for EASTMAN DOUBLE-X, where the corpus previously held only the short
    # "technical information" extract. Black-and-white, so ONE curve, like its
    # sister sheet 5231. The stored f50 triple was the flat estimate 56/56/56.
    "5222": ("EASTMAN DOUBLE-X Negative Film 5222.pdf", 3,
             "EASTMAN_DOUBLE_X_5222", (72.7, 275.3, 189.9, 342.9)),
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
    # ---- C2b, 2026-08-23: the colour batch, and it is what C24 was waiting for.
    # Five more Kodak COLOUR negatives with all three records intact, spanning
    # 1989 (EXR) to 2005 (VISION2) -- plus two that yield green and blue only and
    # are registered anyway, because a REFUSED record is evidence too and this is
    # where the refusal is re-derived. Every one of these was inspected on the
    # --overlay render before being pinned; four extractor defects were found that
    # way, and all four produced plausible numbers (see the module docstring).
    "5217": ("5217-Vision2-200T.pdf", 3, "KODAK_VISION2_200T_5217",
             (222.7, 349.3, 205.0, 284.6)),
    "5218": ("5218-Vision2-500T-H-1-5218t.pdf", 3, "KODAK_VISION2_500T_5218",
             (368.5, 562.5, 385.6, 538.6)),
    "5245": ("5245.pdf", 3, "EASTMAN_EXR_50D_5245", (350.6, 553.1, 381.1, 534.1)),
    "5248": ("5248.pdf", 3, "EASTMAN_EXR_100T_5248", (362.6, 565.1, 284.5, 437.5)),
    # ⚠ THE LARGEST ADJACENCY OVERSHOOT IN THE CORPUS, AND IT IS PRINTED, NOT A
    # TRACE ERROR: this sheet's green and blue records rise to 142 % and 155 % at
    # 15 cycles/mm. Verified on the overlay -- the traced points sit on the
    # printed curves. It is also why this stock's ROLLOFF EXPONENT is not
    # adoptable: the carrier 1/(1+(f/f50)^q) is 1.0 at zero frequency by
    # construction and cannot represent a curve that starts at 1.42, so its fit
    # comes back at rms 0.25 against 0.02-0.10 everywhere else.
    "5279": ("5279.pdf", 2, "KODAK_VISION_500T_5279", (359.4, 541.5, 149.9, 287.6)),
    # ⚠ RED REFUSED ON BOTH OF THESE, by the fragment test rather than by hand:
    # the sheets emit the red record in pieces and the surviving piece starts at
    # 53 % (5293) and 77 % (5205) response. A piece has an f50 and it is
    # meaningless -- 5293's reads 32.0 cycles/mm off a 30-125 c/mm arc.
    "5293": ("5293.pdf", 3, "EASTMAN_EXR_200T_5293", (357.8, 560.4, 64.5, 215.1)),
    "5205": ("5205t.pdf", 3, "KODAK_VISION2_250D_5205",
             (359.7, 561.8, 396.2, 547.1)),
    # ---- the only NON-KODAK MTF sheet in the corpus, and the reason C2b went
    # ---- looking: C24 asks whether the per-record ratio can be derived from the
    # ---- layer stack, and seven measurements of one maker cannot answer that.
    # ⚠ IT DOES NOT ANSWER IT EITHER, because Agfa prints ONE curve: the panel is
    # captioned "Sharpness ... MTF (Modulation Transfer Function)" with
    # "Densitometry: visual filter (V-lambda)", i.e. a visual-weighted pooled
    # response, not three records. What it does establish is that the power-law
    # carrier and the overshoot are not Kodak habits: q = 2.63 at rms 0.039, and
    # a +11.7 % overshoot at 3.4 cycles/mm.
    # ⚠ ALSO NOT A NEW STOCK. This sheet's page 6 left column is Vista 200, which
    # the database holds; pages 5, 7 and 8 carry Vista 100/400/800, FUTURA II and
    # CTprecisa, which it does not.
    "vista200": ("AGFA/AGFACOLOR Vista 100, 200, 400, 800.pdf", 6,
                 "AGFA_VISTA_200", (53.4, 179.8, 316.1, 401.6)),
}

#: Measured 2026-08-18/20. --assert fails if a sheet stops reproducing these.
#: A colour sheet pins one entry per record; a mono sheet pins the single curve
#: under the key "-".
EXPECTED = {
    # ✅ ADOPTED 2026-08-25 (queue E0b-orig). The FIRST measured MTF for a colour
    # REVERSAL stock in this database -- every other traced sheet is a negative.
    # ⚠ AND IT IS THE LARGEST MTF CORRECTION THE PROJECT HAS MADE: the stored
    # f50_g was the estimate 82.0 and the sheet measures 42.1, i.e. the estimate
    # was 1.95x TOO SHARP. Red and blue had no stored value of their own at all.
    # The layer order comes out R < G < B (27.2 / 42.1 / 60.9), which is the
    # order MTFSpec's docstring predicts -- blue on top, red at the bottom under
    # the most scattered light -- and is the second independent confirmation of
    # it, after 5201.
    # The power law beats the legacy Gaussian on all three records (3.5x, 1.9x
    # and 1.4x better in rms), so q is stored and mtf_measured is set.
    "5285": {
        "R": dict(f50=27.2, peak=1.040, peak_at=2.4),
        "G": dict(f50=42.1, peak=1.030, peak_at=7.8),
        "B": dict(f50=60.9, peak=1.022, peak_at=8.8),
    },
    "5231": {"-": dict(f50=41.3, peak=1.034, peak_at=4.6)},
    # ✅ ADOPTED 2026-08-26. EASTMAN DOUBLE-X, off the full H-1-5222 sheet the
    # owner supplied; the corpus had only a short extract before.
    # ⚠ THE STORED TRIPLE WAS THE FLAT ESTIMATE 56.0 / 56.0 / 56.0 -- 1.33x TOO
    # SHARP. The measurement also lands within 2 % of its SISTER STOCK: PLUS-X
    # 5231, the other Kodak black-and-white cine negative in this corpus, reads
    # 41.3 off its own sheet against DOUBLE-X's 42.2. Two independent sheets,
    # two independent traces, and the two films are two speeds of one design
    # family -- which is the sort of agreement that was NOT available while both
    # numbers were estimates (the old pair read 56.0 and 60.0).
    # ⚠ ITS OVERSHOOT IS +25 %, the third largest traced, and it is PRINTED --
    # verified on the --overlay render, where the traced points sit on the
    # printed curve over its whole 2.4-98.5 cycles/mm extent. q is still adopted
    # here, unlike 5279's +42 %: the power law fits at rms 0.076, inside the
    # 0.0095-0.132 band of every accepted curve, where 5279 returned 0.25-0.34.
    "5222": {"-": dict(f50=42.2, peak=1.250, peak_at=4.1)},
    "5201": {
        "R": dict(f50=32.1, peak=1.108, peak_at=2.5),
        "G": dict(f50=49.7, peak=1.157, peak_at=10.7),
        "B": dict(f50=55.5, peak=1.142, peak_at=12.7),
    },
    # ✅ ADOPTED 2026-08-20c (queue C13, owner-approved). 5274 stored the ESTIMATE
    # 56.0 / 64.0 / 72.0 with adjacency 0.09 and now carries these measurements.
    #   GREEN AND BLUE CONFIRMED the estimate to 7 % (68.8 vs 64, 74.0 vs 72).
    # THE RED RECORD DID NOT: 35.4 against 56.0, i.e. the estimate was 1.58x too
    # sharp. That is the estimating RULE, not this profile -- it puts f50_r/f50_b
    # at about 0.78 on 72 of the 92 colour stocks still carrying an estimate,
    # while both stocks measured per-record land at 0.478 (5274) and 0.578 (5201).
    # Whether to re-derive the rule for the remaining 92 is a separate decision
    # and is NOT taken here; this entry pins the measurement that raised it.
    "5274": {
        "R": dict(f50=35.4, peak=1.027, peak_at=2.4),
        "G": dict(f50=68.8, peak=1.162, peak_at=11.0),
        "B": dict(f50=74.0, peak=1.234, peak_at=16.1),
    },
    # ---- C2b, 2026-08-23 ----------------------------------------------------
    "5217": {
        "R": dict(f50=33.9, peak=1.058, peak_at=2.5),
        "G": dict(f50=58.1, peak=1.110, peak_at=13.7),
        "B": dict(f50=67.4, peak=1.154, peak_at=13.7),
    },
    "5218": {
        "R": dict(f50=37.6, peak=1.008, peak_at=2.4),
        "G": dict(f50=54.6, peak=1.014, peak_at=7.7),
        "B": dict(f50=69.7, peak=1.064, peak_at=18.4),
    },
    "5245": {
        "R": dict(f50=37.2, peak=0.984, peak_at=3.7),
        "G": dict(f50=83.8, peak=1.048, peak_at=12.9),
        "B": dict(f50=100.5, peak=1.089, peak_at=15.7),
    },
    "5248": {
        "R": dict(f50=37.4, peak=0.984, peak_at=3.7),
        "G": dict(f50=75.1, peak=1.069, peak_at=12.9),
        "B": dict(f50=111.2, peak=1.153, peak_at=20.4),
    },
    "5279": {
        "R": dict(f50=41.1, peak=1.088, peak_at=2.5),
        "G": dict(f50=73.1, peak=1.420, peak_at=15.1),
        "B": dict(f50=76.1, peak=1.554, peak_at=15.4),
    },
    # Green and blue only: red is refused as a fragment on both sheets.
    "5293": {
        "G": dict(f50=75.2, peak=1.065, peak_at=15.9),
        "B": dict(f50=114.6, peak=1.155, peak_at=18.3),
    },
    "5205": {
        "G": dict(f50=55.9, peak=1.032, peak_at=14.6),
        "B": dict(f50=59.3, peak=1.099, peak_at=14.6),
    },
    # One visual-weighted curve, so it pins the mono key.
    "vista200": {"-": dict(f50=50.0, peak=1.117, peak_at=3.4)},
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
    # ⚠ THE LABEL SITS BEYOND THE END OF ITS OWN CURVE ON SOME SHEETS, and that
    # is how Kodak labels a family: the letter is set just to the right of where
    # the curve stops. On H-1-5245 the three curves end at x 517 and the letters
    # are at x 520-528, so a strict "the label must be inside the curve's x range"
    # test refused all three and the sheet silently fell back to the single-curve
    # rule -- which is how it came to report ONE f50 for a colour negative. A
    # label is allowed to sit up to 8 % of the frame width past the terminus, and
    # is then compared against the curve's value AT that terminus.
    reach = 0.08 * (fx1 - fx0)

    # ⚠ SOLVED AS A MINIMUM-COST BIJECTION over the six permutations, and both
    # simpler rules were tried first and both broke on a real sheet.
    #   * NEAREST-CURVE, greedy: on H-1-5245 the label "G" sits 4.4 pt from the
    #     RED curve's end and 7.6 pt from its own, takes red, and the assignment
    #     then fails -- so the sheet silently fell back to the single-curve rule
    #     and reported ONE f50 for a colour negative.
    #   * RANK BY HEIGHT at a common abscissa: on 5248 the red record STOPS at
    #     115 cycles/mm while green runs to 191 and ends lower, so the vertical
    #     order of the three curve ENDS is B, R, G -- ranking swapped red and
    #     green and produced a red record SHARPER than green, which is
    #     physically impossible on any colour negative.
    # Three curves and three labels is six permutations, so the assignment can
    # simply be solved rather than approximated. Each label is compared against
    # the curve height at the point of that curve nearest the label's own x,
    # because Kodak sets the letter beside the curve's terminus.
    import itertools

    def height_at(pts, lx):
        px = [q[0] for q in pts]
        if not (min(px) - 2 - reach <= lx <= max(px) + 2 + reach):
            return None
        order = sorted(pts)
        xq = min(max(lx, min(px)), max(px))
        return float(np.interp(xq, [q[0] for q in order], [q[1] for q in order]))

    recs = sorted(lab)
    if len(cand) < 3:
        return None
    cost = {}
    for rec in recs:
        lx, ly = lab[rec]
        for i, (pts, *_rest) in enumerate(cand):
            h = height_at(pts, lx)
            cost[(rec, i)] = None if h is None else abs(h - ly)
    best, bestperm = None, None
    for perm in itertools.permutations(range(len(cand)), 3):
        vals = [cost[(rec, i)] for rec, i in zip(recs, perm)]
        if any(v is None for v in vals):
            continue
        tot = sum(vals)
        if best is None or tot < best:
            best, bestperm = tot, perm
    if bestperm is None:
        return None
    # A SANITY bound, not a fit: the letters are set beside their curves, so a
    # label further than a fifth of the frame height from the curve it was
    # assigned means the assignment is wrong and the sheet must be refused.
    if any(cost[(rec, i)] > 0.20 * (fy1 - fy0)
           for rec, i in zip(recs, bestperm)):
        return None
    return {rec: cand[i][0] for rec, i in zip(recs, bestperm)}


def pick_curves(pg, fx0, fx1, fy0, fy1):
    """The response curves inside the frame, keyed by record.

    ⚠ THE OLD RULE WAS "the thickest long path", and it only ever had to pick one
    curve out of one. A colour sheet draws three, at identical width, and prints
    the record in INK -- the same convention granularity_vector.colour_assign()
    reads on the brochures. It also draws the red record TWICE, once in yellow and
    once in magenta on top, so a naive by-colour grouping yields four curves and
    two of them are the same measurement.

    ⚠ AND A THIRD WAY, FOUND UNDER C2b ON 2026-08-23: the 1990s technical sheets
    emit ALL THREE response curves as ONE path object, exactly as the granularity
    panels do. Treating that as a single curve is not a small error -- on H-1-5218
    it produced one "curve" running 2.4 to 80 cycles/mm through 106 % down to
    20 %, i.e. a trace that walks along blue, jumps to green and finishes on red,
    and it still looks like a plausible MTF. So a drawing that splits into two or
    more frame-spanning subpaths is expanded into one candidate per subpath, using
    `granularity_vector.subpaths()` -- the SAME splitter, imported rather than
    copied, because it is the function that whole extraction turns on.
    A drawing that yields one wide subpath is left exactly as it was, which is
    what keeps 5231, 5201 and 5274 reproducing their pinned values.

    Returned as {record: points}. A sheet whose ink is black (5231) yields
    {"-": points} and is measured exactly as before -- verified: it reproduces
    f50 41.3 and the 3.4 % overshoot to the digit.
    """
    cand = []
    span_min = 0.20*(fx1-fx0)

    def wide(pts):
        return (len(pts) >= 8
                and max(x for x, _ in pts) - min(x for x, _ in pts) >= span_min)

    def falls(pts):
        """A response curve DESCENDS. A grid line and a tick row do not.

        ⚠ SECOND HALF OF THE SAME LESSON as single_valued(), and it is what the
        overlay showed on 5245: the letter matcher had claimed the row of tick
        marks along the 1 % gridline as the green record -- perfectly
        single-valued, perfectly flat -- while the actual green curve, drawn
        between red and blue and labelled G on the sheet, was never a candidate.
        An MTF curve crosses 50 %, so over its own frame it must cover a real
        fraction of the response axis; 15 % is far below the ~40 % that every
        traced curve in this corpus actually spans, and far above a gridline's 0.
        """
        ys = [q[1] for q in pts]
        return (max(ys) - min(ys)) >= 0.15 * (fy1 - fy0)

    def smooth(pts):
        """A response curve is nearly monotone; a comb of ticks is not.

        ⚠ THIRD AND LAST FILTER, and the one that finally separated the GRID from
        the curves on 5245 and 5246. Their grids are emitted as ONE connected
        polyline that walks the whole frame -- so it is wide, it spans the full
        response axis, and (because most of its vertices sit on the bottom rule)
        it even survives a per-bin spread test. What it cannot fake is SMOOTHNESS:
        walking a grid accumulates far more vertical travel than the height it
        covers. Measured on the eleven curves already traced in this corpus, the
        ratio of total vertical variation to vertical span is 1.00-1.35 -- a
        response curve descends, with at most a small rise over its overshoot.
        The 5245 grid comes in at 12.9. The cut at 2.0 is between the two by a
        wide margin in both directions, which is the only kind of threshold worth
        having.
        """
        ys = [q[1] for q in pts]
        span = max(ys) - min(ys)
        if span <= 0.0:
            return False
        tv = sum(abs(ys[i+1] - ys[i]) for i in range(len(ys)-1))
        return tv <= 2.0 * span

    def single_valued(pts):
        """A transfer function has ONE response per frequency. A grid does not.

        ⚠ THIS TEST IS WHY C2b's BATCH IS TRUSTWORTHY AT ALL, and it was added
        after the letter matcher confidently handed back the LOG GRID as the green
        record of 5245 (7.8-477 cycles/mm, response to 190 %, power-law rms 0.74)
        and of 5246 (1.0-608 cycles/mm). Both are absurd on inspection and both
        would have been adopted by a script that only checked "did three curves
        come back". The letters are printed near the grid lines too, so nearest-
        letter matching cannot separate them; the SHAPE can, and this is the
        cheapest true statement about the shape: bin the points by frequency and
        require the vertical spread inside a bin to be small. A grid has the full
        frame height in every bin.
        """
        xs = [q[0] for q in pts]
        lo, hi = min(xs), max(xs)
        if hi - lo <= 0.0:
            return False
        nb = 12
        spread = []
        for b in range(nb):
            a0 = lo + (hi - lo) * b / nb
            a1 = lo + (hi - lo) * (b + 1) / nb
            ys = [q[1] for q in pts if a0 <= q[0] <= a1]
            if len(ys) >= 2:
                spread.append(max(ys) - min(ys))
        if not spread:
            return False
        spread.sort()
        med = spread[len(spread)//2]
        return med <= 0.12 * (fy1 - fy0)

    for p in pg.get_drawings():
        r = p["rect"]
        if not (fx0-3 <= r.x0 and r.x1 <= fx1+3
                and fy0-3 <= r.y0 and r.y1 <= fy1+3):
            continue
        n_it = sum(1 for it in p["items"] if it[0] in ("l", "c"))
        if n_it < 1:
            continue
        col = p.get("color")
        col = tuple(round(float(c), 3) for c in col) if col else None
        w = p.get("width") or 0.0
        # ONE PATH, SEVERAL CURVES. Split first and only fall back to the whole
        # path when the split does not produce two or more frame-spanning pieces.
        # The subpath candidates inherit the parent's item count and stroke width,
        # because those describe the PEN and are what the mono branch reasons
        # about -- they are properties of the drawing, not of its pieces.
        parts = [q for q in (gv.subpaths(p["items"]))
                 if wide(q) and single_valued(q) and falls(q)
                 and smooth(q)]
        if len(parts) >= 2:
            for q in parts:
                cand.append((q, col, n_it, w))
            continue
        pts = flatten(p["items"])
        if len(pts) < 8:
            continue
        # a curve crosses a useful part of the frame; a tick or a legend rule
        # does not. 20 % of the frame width, the same floor granularity_vector
        # uses, and it is what rejects the three legend swatches on 5201.
        if max(x for x, _ in pts) - min(x for x, _ in pts) < span_min:
            continue
        if not (single_valued(pts) and falls(pts) and smooth(pts)):
            continue
        cand.append((pts, col, n_it, w))
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
            # ⚠ THE 8-ITEM FLOOR IS A TIE-BREAK NOW, NOT A GATE. It exists to
            # keep the log grid out of a mono sheet's single-curve choice, and
            # since C2b the shape filters above (single-valued, falls, smooth)
            # do that far better -- while the floor by itself refuses a
            # legitimate curve drawn in few beziers. Agfa's Vista sharpness
            # panel draws its whole transfer curve in FIVE, and this branch was
            # returning "no curve inside the frame" on the only non-Kodak MTF
            # sheet in the corpus.
            thick = cand
        # ⚠ GREY INK DOES NOT MEAN ONE CURVE. Try the printed letters first; only
        # a frame with no usable R / G / B triple falls back to the single-curve
        # rule, which is what 5231 (a black-and-white stock) needs.
        # ⚠ AND THE TEST IS ">= 3", NOT "== 3", SINCE C2b. The exact-three form
        # silently skipped every 1990s technical sheet: those frames also carry a
        # log grid drawn as a frame-spanning path, so the candidate list is four
        # and the letters were never consulted. `letter_assign` already refuses
        # anything it cannot map as a bijection at a stacked abscissa, so handing
        # it a longer list costs nothing and is what the extra path needs.
        if len(thick) >= 3:
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


def overlay(pg, got, fx, fy, path):
    """Draw every traced point back onto the rendered panel, per record.

    ⚠ THE OVERLAY IS THE GATE, and this extractor did not have one until C2b --
    which is exactly how the single-path defect above survived: the numbers it
    produced were plausible (an f50 of 69.7 for a 500T stock) and nothing showed
    that the "curve" they came from walked across all three records. Both other
    plot extractors in this project (`vision3_granularity`, `granularity_vector`)
    grew the same gate for the same reason, and their docstrings say so.

    Colours are the record's own, magenta for an unidentified single curve.
    """
    from PIL import Image, ImageDraw
    dpi = 200.0
    pix = pg.get_pixmap(dpi=int(dpi))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    dr = ImageDraw.Draw(img)
    k = dpi / 72.0
    cols = {"R": (220, 0, 0), "G": (0, 170, 0), "B": (0, 80, 255),
            "-": (255, 0, 200)}
    for rec, pts in got.items():
        c = cols.get(rec, (255, 140, 0))
        for x, y in pts:
            X, Y = x * k, y * k
            dr.ellipse([X - 1.6, Y - 1.6, X + 1.6, Y + 1.6], fill=c)
    img.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--overlay", metavar="DIR",
                    help="write ov_<tag>.png with every traced point drawn back "
                         "onto the page -- look at it before believing any number")
    ns = ap.parse_args()
    import pymupdf
    if ns.overlay:
        import os
        os.makedirs(ns.overlay, exist_ok=True)

    bad = 0
    for tag, (fn, pgno, prof, (fx0, fx1, fy0, fy1)) in SHEETS.items():
        # A sheet may name its maker's subdirectory ("AGFA/....pdf"); a bare
        # filename still means KODAK, which is where every sheet came from until
        # C2b went looking for a second manufacturer.
        base = Path(ns.root).resolve() / "PDF" / "PROFILES"
        pdf = (base / fn) if "/" in fn else (base / "KODAK" / fn)
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
            # ⚠ A FRAGMENT IS REFUSED, NOT MEASURED. Some sheets emit a record
            # in pieces (a leader line, a label gap, a curve that leaves and
            # re-enters the frame), and the splitter then hands back a piece.
            # A piece has an f50 -- 5293's red fragment reports 32.0 cycles/mm
            # from a 30-125 c/mm arc that starts BELOW 53 % -- and it is
            # meaningless. A response curve must cover most of the plotted
            # frequency range and must start above 50 %, since that is what
            # "the frequency where it falls through 50 %" presumes.
            # ⚠ THE TEST IS "starts at full response over at least a decade",
            # NOT "covers most of the frame". Kodak draws these curves over the
            # 2-100 cycles/mm the film can actually resolve while the frame is
            # ruled to 600 or 1000, so a frame-coverage rule refuses every real
            # curve on the sheet -- it did, on all eight, before being corrected.
            # What a fragment cannot fake is where it STARTS: every intact curve
            # in this corpus begins at 96-110 % response, while 5293's red piece
            # begins at 53 % and 5205's at 77 %.
            fspan = np.log10(f.max()) - np.log10(f.min())
            if fspan < 1.0 or r[0] < 0.90:
                print(f"    [SKIP] {rec}: fragment, not a curve -- {fspan:.2f} "
                      f"decades starting at {r[0]*100:.0f} % response (an intact "
                      f"curve starts near 100 %). Refused rather than measured")
                continue

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
        if ns.overlay:
            from pathlib import Path as _P
            out = str(_P(ns.overlay) / f"ov_{tag}.png")
            overlay(pg, got, fx, fy, out)
            print(f"    overlay -> {out}")
    print()
    if bad:
        print(f"[FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("[OK] MTF read from the sheet's vector path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
