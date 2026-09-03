"""Takano 1968 Part 2, Fig. 11 -- Expected mottle size by developer and route.

WHAT THIS PAPER IS, AND WHY IT IS NOT THE TAKANO ALREADY READ
-------------------------------------------------------------
`PDF/PROFILES/RETRO/JAPAN/31_209.pdf`. Masao TAKANO (高野正雄), Fuji Photo Film
Research Laboratories, Ashigara. 「写真像の粒状性(第2報) -- 一定の濃度をうる
露光量と現像時間の組合わせと Wiener Spectrum」 / *Granularity of Photographic
Image (II): Wiener Spectrum at a Constant Density Level with Various Exposure
and Time of Development*, J. Soc. Phot. Sci. Japan **31**(4), 209-214 (1968),
received 5 Aug 1968. Supplied by the owner 2026-09-02.

⚠ THE CORPUS ALREADY HELD A TAKANO AND THIS IS NOT IT. `23_13.pdf` is Takano's
解説 -- a REVIEW of granularity evaluation and measurement method, with a long
bibliography and no experiment of its own; §23k of the knowledge base is built
on it. This is an ORIGINAL EXPERIMENTAL PAPER, the second of a two-part series,
and its Part 1 is not in the corpus. Byte-compared against all 96 PDFs held: no
duplicate. Figure-by-figure against §23j (Ooue 1959) and §23k (Takano 1969): no
overlap of samples, panels or measurements. New material.

THE EXPERIMENT
--------------
One general-purpose black-and-white negative film, **ASA 100**, UNNAMED -- which
is why nothing here is written to a film profile. Grain size distribution
measured on a centrifugal sedimentation sizer, with an electron micrograph
(Fig. 1). Exposure by tungsten lamp through a DG filter at **6470 K**. Four
developers, the "Series-A" set of Part 1, each close to a single-agent solution
and all adjusted to **pH 8.5 with NaOH** so the developing agent's own behaviour
shows through: para-phenylenediamine (p-p), PQ, Monol, MQ. Wiener spectra on a
high-speed rotating microphotometer with a frequency analyser; 20 c/s-20 kc/s is
4.2-42000 lines/mm in principle but **500 lines/mm is the real ceiling**
(rotation irregularity, focus, circuit noise), which the author argues is ample
because 300-500 lines/mm already carries the granularity information.

⚠ THE TWO ROUTES TO ONE DENSITY, WHICH IS THE WHOLE POINT OF THE PAPER:
  [VTD] -- exposure held constant, DEVELOPMENT TIME varied to reach the density.
  [VE]  -- development time held constant, EXPOSURE varied to reach it.
Same film, same developer, same final density, two different grain patterns.
This is a variable the renderer does not have and this database does not store.

WHAT IS TRACED HERE, AND WHY ONLY THIS
--------------------------------------
**Fig. 11 only.** It is the one panel whose ordinate is a LENGTH IN MICROMETRES
-- "Expected mottle size (mu)" -- and therefore the only quantity in the paper
directly comparable with anything this project stores. Figs. 3-8 plot spectrum
level F(u,0) in an unlabelled instrument unit, and Fig. 10 plots that same unit
against gamma, so neither can be turned into a stored parameter without the
instrument constant, which is not printed. Fig. 9 (development time against
gamma) is a process curve for an unnamed film. All four are read for their
SHAPE and recorded in the knowledge base as such, not digitised.

Fig. 11 carries, for each of the four developers:
  * two ENVELOPE LINES labelled [VE] and [VTD], both at **D = 0** -- the legend
    "D=0" sits above the marker key and belongs to the lines, not the markers;
  * a filled pair (VE) and an open pair (VTD), each spanning **density 0.5 to
    1.5**: lower marker D 0.5, upper marker D 1.5.

⚠ THE PAPER'S TWO PERCENTAGES DESCRIBE THE LINES, NOT THE MARKERS, and reading
them onto the wrong object is the trap this file exists to prevent. Measured
here:
  * "[VTD] is 30-40 % smaller than [VE]" -- TRUE OF THE D = 0 ENVELOPES, which
    this trace measures at 36 % (p-p end) and 40 % (M-Q end). Measured on the
    density-0.5-1.5 MARKERS the same ratio is only 10-28 %, mean 17 %.
  * "mottle size grows 20-30 % with density" -- NOT reproduced by the markers,
    which give **+4 % to +16 %** from D 0.5 to D 1.5, mean +8 %. The prose
    presumably means a wider density span than the markers show, or the
    spectrum-level growth of Figs. 7-8 rather than the mottle size. Recorded as
    measured; the disagreement is recorded, not reconciled (method rule 4).

CALIBRATION, and the check that makes it believable
---------------------------------------------------
The ordinate is fitted over the five printed tick labels 0, 2, 4, 6, 8, located
as connected glyphs in the narrow column left of the axis (x within 32 px of it,
which separates them from the rotated axis title further left). Five points,
79.54 px per micrometre, residual 0.017 um. ⚠ THE INDEPENDENT CHECK: the panel's
own x-axis rule -- found separately, as the row of maximum ink -- lands at
y 1039 against the fitted zero at y 1038.8, i.e. **0.2 px, 0.003 um**, on a
quantity the fit never saw.

USAGE
    python3 takano_1968_mottle.py [--root .] [--assert] [--overlay DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

#: Page 5 of the PDF (printed page 213) carries Figs. 9, 10 and 11.
PAGE = 5
DPI = 300

#: Developer names left to right on the abscissa, as printed.
DEVELOPERS = ("p-p", "p-Q", "Monol", "M-Q")

#: Measured 2026-09-02e, micrometres. (D 0.5, D 1.5) per developer and route.
#: --assert fails if a re-trace stops reproducing these.
EXPECTED_MARKERS = {
    ("p-p",   "VE"):  (4.42, 4.80),
    ("p-p",   "VTD"): (3.98, 4.16),
    ("p-Q",   "VE"):  (4.98, 5.55),
    ("p-Q",   "VTD"): (4.37, 4.59),
    ("Monol", "VE"):  (5.78, 6.13),
    ("Monol", "VTD"): (4.62, 4.94),
    ("M-Q",   "VE"):  (6.32, 6.81),
    ("M-Q",   "VTD"): (4.57, 5.30),
}

#: The two D = 0 envelope lines, sampled at the p-p and M-Q abscissae.
EXPECTED_ENVELOPE = {"VE": (5.22, 7.30), "VTD": (3.35, 4.37)}

TOL_UM = 0.20


def _page_gray(root: Path) -> np.ndarray:
    import pymupdf
    pdf = root / "PDF" / "PROFILES" / "RETRO" / "JAPAN" / "31_209.pdf"
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    pm = pymupdf.open(pdf)[PAGE - 1].get_pixmap(dpi=DPI)
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)
    return a[:, :, :3].mean(axis=2).astype(float)


def _axes(g: np.ndarray) -> tuple[int, int]:
    """(x of the ordinate rule, y of the abscissa rule) for the Fig. 11 panel.

    Found by rule, not by hard-coded pixels: Fig. 11 owns the upper-right
    quadrant of the page, and within it the ordinate is the column carrying the
    most ink and the abscissa the row carrying the most. A page rendered at
    another DPI still resolves; a pixel box would not.
    """
    h, w = g.shape
    y0, x0 = 0, int(w * 0.50)
    sub = g[: int(h * 0.45), x0:] < 140
    return x0 + int(sub.sum(axis=0).argmax()), y0 + int(sub.sum(axis=1).argmax())


def _ordinate(g: np.ndarray, xax: int, ybot: int):
    """Least-squares micrometre-per-pixel over the printed tick labels."""
    from scipy import ndimage
    roi = np.zeros(g.shape, bool)
    roi[int(ybot * 0.28): ybot + 30, xax - 32: xax - 6] = True
    lab, n = ndimage.label((g < 150) & roi)
    cent = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        hgt = int(ys.max() - ys.min()) + 1
        wid = int(xs.max() - xs.min()) + 1
        if 14 <= hgt <= 30 and 6 <= wid <= 22:
            cent.append(float(ys.mean()))
    cent.sort()
    if len(cent) != 5:
        raise ValueError(f"found {len(cent)} ordinate labels, expected 5")
    fit = np.polyfit(cent, [8.0, 6.0, 4.0, 2.0, 0.0], 1)
    resid = float(np.max(np.abs(np.polyval(fit, cent) - np.array([8., 6., 4., 2., 0.]))))
    return fit, resid, cent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--overlay", metavar="DIR")
    ns = ap.parse_args()

    try:
        g = _page_gray(Path(ns.root).resolve())
    except FileNotFoundError as e:
        print(f"  [SKIP] source not present: {e}")
        return 0

    xax, ybot = _axes(g)
    fit, resid, cent = _ordinate(g, xax, ybot)
    zero = float(np.roots([fit[0], fit[1]])[0]) if fit[0] else float("nan")
    print(f"[i] 31_209.pdf p{PAGE} Fig. 11 -- ordinate rule x={xax}, abscissa rule y={ybot}")
    print(f"    ordinate: 5 tick labels, {abs(1.0/fit[0]):.2f} px per micrometre, "
          f"residual {resid:.3f} um")
    print(f"    independent check: the fitted zero lands at y {zero:.1f} against the "
          f"abscissa rule at y {ybot} -- {abs(zero-ybot):.1f} px, "
          f"{abs(zero-ybot)*abs(fit[0]):.3f} um, on a quantity the fit never saw")
    if resid > 0.05:
        print("  [FAIL] ordinate labels are not collinear -- refusing the panel")
        return 1

    def um(y_px: float) -> float:
        return float(np.polyval(fit, y_px))

    # ---- the markers -------------------------------------------------------
    # ⚠ EACH PAIR IS ONE CONNECTED BLOB, not two: the panel draws a vertical
    # rule joining the D 0.5 and D 1.5 markers of a pair. Detecting "circles"
    # therefore finds nothing and picks up the legend text instead, which is the
    # first thing this tracer got wrong. A pair is found as a TALL NARROW blob
    # and its two markers are its two ends, inset by the marker radius.
    from scipy import ndimage
    roi = np.zeros(g.shape, bool)
    roi[int(ybot * 0.32): int(ybot * 0.85), xax + 40: int(g.shape[1] * 0.91)] = True
    lab, n = ndimage.label((g < 120) & roi)
    pairs = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        hgt = int(ys.max() - ys.min()) + 1
        wid = int(xs.max() - xs.min()) + 1
        # a pair is 25-80 px tall and under 16 px wide; the [VE] and [VTD]
        # labels are 24 px tall but 17-19 px wide, which is what excludes them
        if 25 <= hgt <= 85 and wid <= 16:
            pairs.append((float(xs.mean()), int(ys.min()), int(ys.max()),
                          len(ys) / float(hgt * wid)))
    pairs.sort()
    # ⚠ THE [VE] AND [VTD] LABELS SURVIVE EVERY SHAPE TEST, because a square
    # bracket is exactly a tall narrow blob -- four of them, and two land on the
    # p-Q abscissa where they would be read as a third and fourth marker pair.
    # They are removed by the one property no data pair has: a bracket comes as
    # a TWIN, a second blob 40-110 px away in x with the same top and bottom to
    # within a few pixels. The two pairs at one developer are stacked, never
    # side by side, so this cannot take a real marker with it.
    def _twinned(a, rest):
        return any(abs(a[1] - b[1]) <= 6 and abs(a[2] - b[2]) <= 6
                   and 40 <= abs(a[0] - b[0]) <= 110 for b in rest)
    _brackets = [p for p in pairs if _twinned(p, [q for q in pairs if q is not p])]
    pairs = [p for p in pairs if p not in _brackets]
    print(f"    {len(pairs)} marker pairs found after dropping {len(_brackets)} "
          f"label brackets (8 expected: 4 developers x 2 routes)")

    if ns.overlay:
        Path(ns.overlay).mkdir(parents=True, exist_ok=True)
        from PIL import Image, ImageDraw
        im = Image.fromarray(g.astype(np.uint8)).convert("RGB")
        d = ImageDraw.Draw(im)
        for x, yt, yb, _f in pairs:
            d.rectangle([x - 14, yt - 3, x + 14, yb + 3], outline=(255, 0, 0), width=2)
        im.save(Path(ns.overlay) / "ov_takano1968_fig11.png")
        print(f"    overlay written to {ns.overlay}/ov_takano1968_fig11.png")

    # cluster by abscissa: four developers
    groups: list[list] = [[]]
    for p in pairs:
        if groups[-1] and p[0] - groups[-1][-1][0] > 45:
            groups.append([])
        groups[-1].append(p)
    groups = [gp for gp in groups if len(gp) == 2]
    if len(groups) != 4:
        print(f"  [FAIL] {len(groups)} developer clusters of two pairs, expected 4")
        return 1

    MARKER_R = 6          # px; the centre of an end marker is this far inside
    bad, got = 0, {}
    for name, gp in zip(DEVELOPERS, groups):
        # ⚠ [VE] IS THE UPPER PAIR AT EVERY DEVELOPER and that is how the two
        # routes are told apart, NOT by fill: the scan renders some filled discs
        # at fill 0.35 and some open circles at 0.36, so the disc/annulus test
        # is not separable here. The two envelopes never cross -- the paper's
        # whole finding is that [VTD] is smaller everywhere -- so ordering is
        # both the safer discriminator and the one the figure itself asserts.
        gp = sorted(gp, key=lambda p: p[1])
        for route, p in zip(("VE", "VTD"), gp):
            lo, hi = um(p[2] - MARKER_R), um(p[1] + MARKER_R)
            got[(name, route)] = (round(lo, 2), round(hi, 2))
            exp = EXPECTED_MARKERS[(name, route)]
            flag = ""
            if abs(lo - exp[0]) > TOL_UM or abs(hi - exp[1]) > TOL_UM:
                flag = f"   [MISMATCH] expected {exp[0]:.2f}/{exp[1]:.2f}"
                bad += 1
            print(f"    {name:6s} {route:3s}  D 0.5 {lo:.2f} um   D 1.5 {hi:.2f} um"
                  f"   ({100*(hi/lo-1):+.0f} % with density){flag}")

    # ---- the D = 0 envelope lines ------------------------------------------
    # Sampled at the first and last developer abscissa. At those columns the
    # markers are present too, so the envelopes are picked out as the THIN runs
    # (<= 4 px) -- a drawn line, against a marker's 11-25 px.
    env = {}
    for end, gp in (("lo", groups[0]), ("hi", groups[-1])):
        x = int(round(gp[0][0]))
        col = np.where(g[int(ybot * 0.32): int(ybot * 0.87), x] < 130)[0] \
            + int(ybot * 0.32)
        runs: list[list[int]] = []
        for y in col:
            if runs and y - runs[-1][-1] <= 3:
                runs[-1].append(int(y))
            else:
                runs.append([int(y)])
        thin = [r for r in runs if len(r) <= 4]
        if len(thin) >= 2:
            env[end] = (um(float(np.mean(thin[0]))), um(float(np.mean(thin[-1]))))
    if "lo" in env and "hi" in env:
        print(f"    D=0 envelopes: [VE] {env['lo'][0]:.2f} -> {env['hi'][0]:.2f} um, "
              f"[VTD] {env['lo'][1]:.2f} -> {env['hi'][1]:.2f} um")
        r0 = 1.0 - env['lo'][1] / env['lo'][0]
        r1 = 1.0 - env['hi'][1] / env['hi'][0]
        print(f"    [VTD] is {100*r0:.0f} % and {100*r1:.0f} % smaller than [VE] on "
              f"those lines -- THIS is the paper's \"30-40 %\", and it is a "
              f"statement about D = 0, not about the markers")
        for k, e in (("VE", EXPECTED_ENVELOPE["VE"]), ("VTD", EXPECTED_ENVELOPE["VTD"])):
            j = 0 if k == "VE" else 1
            if abs(env['lo'][j] - e[0]) > TOL_UM or abs(env['hi'][j] - e[1]) > TOL_UM:
                print(f"  [MISMATCH] {k} envelope expected {e[0]:.2f} -> {e[1]:.2f}")
                bad += 1

    ve = [got[(d, "VE")][0] for d in DEVELOPERS]
    vtd = [got[(d, "VTD")][0] for d in DEVELOPERS]
    ratio = [v / e for e, v in zip(ve, vtd)]
    print(f"    the same ratio on the D 0.5 MARKERS: "
          f"{', '.join('%.0f %%' % (100*(1-r)) for r in ratio)} "
          f"(mean {100*(1-float(np.mean(ratio))):.0f} %) -- not 30-40 %")
    order = [d for d, _ in sorted(zip(DEVELOPERS, ve), key=lambda t: t[1])]
    print(f"    developer ordering by mottle size, finest first: {' < '.join(order)}")

    if ns.do_assert and bad:
        print(f"[FAIL] {bad} value(s) no longer reproduce")
        return 1
    print("[OK] Fig. 11 traced" + (" and matches the pinned values" if ns.do_assert else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
