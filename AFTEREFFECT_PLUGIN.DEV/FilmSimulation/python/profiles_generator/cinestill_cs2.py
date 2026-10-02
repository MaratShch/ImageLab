#!/usr/bin/env python3
"""CineStill's second figure, identified at last — which curve is which process.

    doc/thirdparty/cinestill_cs41vscs2_raw_px.txt
    traced 2026-08-27 from `cs41vscs2curves2_PNG_480x480.png`, embedded in
    «What makes 800T the original and only true 800 speed tungsten balanced
    film for still photography», cinestillfilm.com/blogs/news/.

⚠⚠ THIS FIGURE WAS REFUSED FOR A MONTH AND THE REFUSAL WAS CORRECT AT THE
TIME. `doc/CINESTILL_800T_DATA_LEDGER_2026-08-27.md` classes it «VENDOR, but
UNIDENTIFIED -- which curve is which process is unknown, so no value can be
attributed», and `NotFound.md` §7.2 carries it. Two dashed curves, a caption
naming two processes, and nothing on the artwork saying which is which: a
50/50 guess dressed as a measurement is worse than no measurement.

WHAT SETTLES IT, AND IT COSTS NOTHING NEW
------------------------------------------
The FIRST figure from the same page is already adopted: three per-layer
characteristic curves, digitised at 480 samples per layer, fitted and stored
on `CINESTILL_800T`, with its abscissa shifted +1.51681 decade to reach this
database's mid-grey-at-zero convention. **One of the two curves in the second
figure must therefore BE one of those three**, because a comparison figure
whose point is «CS41 against CS2» plots the CS41 curve the first figure
already published.

So: put figure 2 on figure 1's abscissa -- the same +1.51681, not a fitted
shift, because both figures are the same publisher's plots of the same film --
and ask which of the six pairings is close. The answer is not marginal:

    curve                 vs stored r   vs stored g   vs stored b
    curve_red                0.4693        0.0848        0.3544   D rms
    curve_neutral            0.2878        0.2748        0.5332   D rms

**`curve_red` is the CS41 green record and nothing else is anything.** Its own
six-parameter fit returns gamma 0.6112 where the stored green record holds
0.6214 -- 1.6 % apart, on two independent traces of two different drawings --
and the next-nearest pairing is three times worse than the winner. `curve_red`
is therefore CS41, and by elimination `curve_neutral` is CS2.

⚠ AND THE DIRECTION OF THE RESULT IS ITS OWN CHECK. Cs2 is CineStill's
ECN-2 kit and Cs41 their C-41 kit; ECN-2 develops a colour negative to LOWER
contrast than C-41, and that is what the identification returns rather than
what it was told to return: CS2 comes out 17.7 % flatter (gamma 0.5029 against
0.6112). An identification that had landed the other way round would have had
to explain an ECN-2 process developing harder than C-41.

WHAT IS STORED, AND THE MUCH LARGER PART THAT IS NOT
-----------------------------------------------------
A `ProcessVariant` needs THREE curves and this figure prints ONE record per
process. Synthesising a red and a blue by carrying the green delta across is
exactly the move `_PROCESS_VARIANTS` exists to avoid -- process changes do not
move three layers by one factor, which is why the PJ800 push ladder stores
three separately fitted records per step. So the CS2 measurement goes into the
database as a documented delta on ONE record with its own provenance, not as a
process variant, and `NotFound.md` §7.2 changes from «unidentified» to «one
record of three».

⚠ THE TWO FIGURES DISAGREE ON D-MIN BY 0.075 D on the record they share
(0.6009 here against the 0.5258 stored from figure 1) while agreeing on gamma
to 1.6 %. Two drawings from one publisher, same film, same process; the base
of a plot is the easiest thing to redraw and the hardest to notice. THE DELTA
IS WHAT IS STORED, NOT THE ABSOLUTE: CS2 minus CS41 measured on ONE figure
cancels that figure's own base error, where CS2-from-figure-2 minus
CS41-from-figure-1 would carry it in whole.

Run:  python cinestill_cs2.py [--root .] [--assert]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RAW = "doc/thirdparty/cinestill_cs41vscs2_raw_px.txt"

#: The archive file's own calibration header, repeated so a silent edit to it
#: shows up as a mismatch here rather than as a quietly different answer.
X0, XS = 366.862, 75.186          # logE = (x - X0) / XS
Y0, YS = 433.5, 129.2             # D    = (Y0 - y) / YS
XPIX0 = 61                        # the first column the trace covers

#: The shift figure 1's adoption applied to reach mid-grey at zero. ⚠ NOT
#: FITTED HERE. Using a fitted shift would let the identification buy its own
#: answer; the whole force of the test is that the two figures are put on ONE
#: axis chosen in advance.
FIG1_SHIFT = 1.51681

#: What this module found, pinned so a rerun that disagrees FAILS.
EXPECTED = dict(
    cs41=(0.6009, 0.6112, -1.5216, 0.0956, 2.1293, 0.6470),
    cs2=(0.5645, 0.5029, -1.5122, 0.1465, 2.0957, 0.3653),
    cs41_rms=0.0061, cs2_rms=0.0056,
    winner_rms=0.0848, runner_up_rms=0.2748,
)


def _load(root: Path):
    out = {}
    for line in (root / RAW).read_text().splitlines():
        if not line.startswith("curve_"):
            continue
        key, vals = line.split("=", 1)
        xs, ys = [], []
        for i, s in enumerate(vals.split(",")):
            if not s.strip():
                continue
            xs.append(((XPIX0 + i) - X0) / XS + FIG1_SHIFT)
            ys.append((Y0 - float(s) / 2.0) / YS)
        out[key] = (np.array(xs), np.array(ys))
    return out


def _model(pr, x):
    d, g, tx, tk, sx, sk = pr
    sp = lambda z, k: k * np.logaddexp(0.0, z / k)
    return d + g * (sp(x - tx, tk) - sp(x - sx, sk))


def _fit(xs, ys):
    from scipy.optimize import least_squares
    f = least_squares(lambda pr: _model(pr, xs) - ys,
                      [0.55, 0.60, -1.6, 0.3, 2.0, 0.44],
                      bounds=([0, 0.1, -4, 0.02, 0.5, 0.02],
                              [2, 2, 1, 2, 6, 2]))
    return f.x, float(np.sqrt((f.fun ** 2).mean()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()
    if not (root / RAW).is_file():
        print("  [SKIP] archive not present: %s" % (root / RAW))
        return 0

    print("CineStill 800T -- `cs41vscs2curves2`, the figure NotFound.md 7.2 "
          "held as unidentified")
    import film_profiles as fp
    import film_sim as FS
    curves = fp.get_profile("CINESTILL_800T").curves
    stored = {"r": curves.r, "g": curves.g, "b": curves.b}
    data = _load(root)
    if set(data) != {"curve_red", "curve_neutral"}:
        print("  [FAIL] the archive no longer holds exactly two curves: %s"
              % sorted(data))
        return 1 if ns.assert_ else 0

    bad = 0
    table = {}
    for name, (xs, ys) in sorted(data.items()):
        row = {}
        for lab, c in stored.items():
            row[lab] = float(np.sqrt(((ys - FS.density(xs, c)) ** 2).mean()))
        table[name] = row
        print("  %-14s vs stored  r %.4f   g %.4f   b %.4f   D rms"
              % (name, row["r"], row["g"], row["b"]))
    flat = sorted(((v, n, l) for n, r in table.items() for l, v in r.items()))
    win, runner = flat[0], flat[1]
    print("  best pairing  %s <-> stored %s at %.4f D rms; next best %.4f -- "
          "%.1fx worse" % (win[1], win[2], win[0], runner[0],
                           runner[0] / max(win[0], 1e-9)))
    # ⚠ THE IDENTIFICATION IS ONLY AS GOOD AS ITS MARGIN. A 3x separation is
    # an identification; 1.2x would be a coin toss with extra steps, and the
    # module must fail rather than report one.
    if not (win[1] == "curve_red" and win[2] == "g"
            and runner[0] > 2.5 * win[0]):
        print("  [FAIL] the identification no longer separates: winner "
              "%s/%s, margin %.2fx" % (win[1], win[2],
                                       runner[0] / max(win[0], 1e-9)))
        bad += 1

    got = {}
    for key, name in (("cs41", "curve_red"), ("cs2", "curve_neutral")):
        pr, rms = _fit(*data[name])
        got[key], got[key + "_rms"] = pr, rms
        print("  %-5s (%s)  dmin %.4f gamma %.4f toe %.4f/%.4f "
              "shoulder %.4f/%.4f   fit rms %.4f"
              % ((key.upper(), name) + tuple(pr) + (rms,)))
        for i, w in enumerate(EXPECTED[key]):
            if abs(pr[i] - w) > 0.01:
                print("    [MISMATCH] %s parameter %d: %.4f vs pinned %.4f"
                      % (key, i, pr[i], w))
                bad += 1

    d_dmin = got["cs2"][0] - got["cs41"][0]
    d_gam = got["cs2"][1] - got["cs41"][1]
    print("  CS2 minus CS41, ON ONE FIGURE: d dmin %+.4f D, d gamma %+.4f "
          "(%.1f %%). ECN-2 flatter than C-41, which is the direction the "
          "chemistry predicts and is not what the test was told to find."
          % (d_dmin, d_gam, 100.0 * (got["cs2"][1] / got["cs41"][1] - 1.0)))
    if d_gam >= 0.0:
        print("  [FAIL] CS2 comes out no flatter than CS41 -- the "
              "identification and the chemistry disagree")
        bad += 1

    # ---- against the database ---------------------------------------------
    rec = [s for s in fp.get_profile("CINESTILL_800T").param_sources
           if s.param == "curves.g.gamma"]
    ok = bool(rec) and "cs41vscs2" in (rec[0].note or "").lower()
    print("\n  AGAINST THE DATABASE: the green record's provenance %s the "
          "second figure and the CS2 delta"
          % ("carries" if ok else "DOES NOT CARRY"))
    if not ok:
        bad += 1

    if ns.assert_ and bad:
        print("\n[FAIL] the CS41/CS2 identification does not reproduce")
        return 1
    print("\n[OK] `cs41vscs2curves2` identified -- curve_red is CS41, "
          "curve_neutral is CS2, and NotFound.md 7.2's «which curve is which» "
          "is answered by the figure the project had already adopted.")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
