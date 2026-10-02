"""BBC Research Report T-101/2 (1964/4): four measured grain Wiener spectra.

WHAT THIS SOURCE IS
-------------------
`RETRO/BBC Photographic film grain. 1964-04.pdf` -- K. Hacking, «Photographic
film grain: an analysis of granularity in television», BBC Research Department
Report No. **T-101/2 (1964/4)**, 41 pages. Every page is a 2261x3029 grayscale
raster with an Acrobat Paper Capture OCR layer, so the tables read as text and
the figures have to be traced.

⚠ IT IS THE SEQUEL TO A REPORT THE CORPUS ALREADY HOLDS. `T-101 (1963/5)` and
BBC Engineering Monograph No. 54 (08/1964) are cited on ILFORD_HPS,
KODAK_TRI_X_400TX and EASTMAN_TRI_X_5223. This is the middle document of the
three and it is the one that prints the whole measurement set in one table.

WHAT IT GIVES THAT NOTHING ELSE IN THE CORPUS DOES
---------------------------------------------------
**Table 1** -- the mean level of the measured grain Wiener spectrum over
0-20 cycles/mm, in SQUARE MICRONS, for four named emulsions, beside their
rated speeds:

      Ilford Pan F     ASA 16    BS 23 deg    0.10 um^2
      Kodak Plus-X     ASA 64    BS 29 deg    0.14
      Kodak Tri-X      ASA 250   BS 35 deg    0.555
      Ilford H.P.S.    ASA 320   BS 36 deg    0.62

**Fig. 8** -- the same four spectra plotted against spatial frequency to
150 cycles/mm. That is a grain POWER SPECTRUM SHAPE, which this project has
nowhere and which no datasheet in the corpus publishes.

**Equation (4) and the exponents.** The report states the density law directly:
S/N = k * D^-0.5 for a direct-positive or negative record, i.e. sigma_D scales
as D^+0.5 -- this project's legacy square-root law, now with a citation. It then
says Higgins and Stultz measured the exponent "closer to -0.4" on a range of
Eastman Kodak emulsions, and that reversal processing needs "-0.6 to -0.7"
because reversal inverts the relation between effective grain size and density.
⚠ `NotFound.md` row 2 names Higgins and Stultz as the best lead for the sigma(D)
gap; this is the first hard number that lead has produced.

**Section 5.2** -- "for a given developed density, the mean grain diameter is
approximately proportional to the square root of the point gamma achieved at
that density", which is the same relation T-101 Table 3 measures directly and
which the ILFORD_HPS provenance already quotes.

THE CONVERSION CHAIN, AND WHY IT NEEDS THREE STEPS RATHER THAN ONE
-------------------------------------------------------------------
A Wiener spectrum is not an rms granularity, and the corpus already contains one
place where the difference was skated over: TRI-X's 0.555 um^2 was converted as
`sigma*1000 = 1000*sqrt(W/A)` = 17.5 and compared against a stored 17.0 "at net
1.0" -- but the BBC figure is measured at **D 0.48 above base at gamma 1.0**,
which is neither the density nor the development the stored number refers to.
Two errors happened to be small and cancelled.

The chain used here states all three steps:

  1. SPECTRUM TO VARIANCE.  sigma^2 = integral of W(f)|A(f)|^2 over the plane,
     and for a uniform aperture Parseval gives integral |A|^2 = 1/area. That
     reduces to sigma = sqrt(W/area) EXACTLY when W is flat across the
     aperture's passband. A 48 um circular aperture has its first transfer zero
     at 1.2197/48 um = 25.4 c/mm, and Fig. 8 is traced here specifically to
     check that W really is flat out to there -- it is, to a few per cent, which
     is what licenses the closed form instead of a numerical integral.

  2. DEVELOPMENT GAMMA.  Grain diameter goes as sqrt(point gamma) (this report
     section 5.2; T-101 Table 3 measures it to about 5 per cent), and granularity
     is proportional to grain diameter, so sigma scales as sqrt(gamma). The BBC
     measurement is at gamma 1.0; the reference development for a negative is
     gamma 0.65, which this report itself names as what a negative is developed
     to for negative-positive work.

  3. DENSITY.  sigma goes as D^0.4 -- Higgins and Stultz, via equation (4)'s
     discussion. ⚠ It is corroborated inside the corpus: T-101 Table 3 measures
     equivalent grain diameter FALLING with density at fixed development
     (2.40 um at D 0.23 against 2.12 um at D 0.54, at development gamma 0.56),
     which is d proportional to D^-0.145, and with sigma proportional to
     d*sqrt(D) that gives sigma proportional to D^0.355. Two independent routes,
     0.355 and 0.40.

  sigma(net 1.0, gamma 0.65) = 1000*sqrt(W/A) * sqrt(0.65/1.0) * (1.0/0.48)^0.4

⚠ **THE CONTROL IS TRI-X, AND IT IS WHAT MAKES THE OTHER THREE ADOPTABLE.**
KODAK_TRI_X_400TX carries 17.0 from Kodak's own published rms granularity. The
chain returns 18.9 from a BBC Wiener spectrum measured a different way, in a
different decade, by a different laboratory: 11 per cent. Plus-X lands 5 per
cent under its stored estimate and H.P.S. 5 per cent over. Three agreements at
that level are what license the method; a chain nobody could check against a
known answer would not be.

⚠ **AND PAN F IS REFUSED, ON THE EVIDENCE.** Its chain value is 61 per cent
above the stored estimate -- the only outlier in four -- and its speed does not
reconcile the way the other three do. The table footnote says these are
"earlier speed ratings (prior to the revised indices)", and the 1960 revision
roughly doubled nominal speeds: Plus-X ASA 64 -> the profile's EI 125 exactly,
Tri-X ASA 250 -> EI 400, H.P.S. ASA 320 -> EI 400. Pan F ASA 16 doubles to 32
and the profile is EI 50, which is Pan F PLUS -- a later emulsion. Same trap the
corpus already documents between TRI-X 5223 and the still Tri-X: one trade name,
two products. No value is taken for it.

Run:  python bbc_t101_2.py --root <corpus> [--assert]
Needs numpy + PyMuPDF.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

try:
    import pymupdf
except ImportError:                                       # pragma: no cover
    print("[!] pymupdf not installed:  pip install pymupdf")
    raise SystemExit(1)

SHEET = "RETRO/BBC Photographic film grain. 1964-04.pdf"

SOURCE = ("K. Hacking, «Photographic film grain: an analysis of granularity in "
          "television», BBC Research Department Report No. T-101/2 (1964/4) -- "
          "PDF/PROFILES/RETRO/BBC Photographic film grain. 1964-04.pdf, "
          "Table 1 p13 and Fig. 8 p13")

#: The 48 um circular reading aperture this project's rms granularity is
#: defined against, as area in square microns. Kodak, Ilford and Agfa all quote
#: rms through it, and it is the aperture the corpus's own convention names.
APERTURE_UM = 48.0
APERTURE_AREA = math.pi * (APERTURE_UM / 2.0) ** 2

#: First zero of a uniform circular aperture's transfer function, in cycles/mm.
#: 2*J1(pi*d*f)/(pi*d*f) vanishes at pi*d*f = 3.8317.
APERTURE_CUTOFF_CPMM = 1000.0 * 3.8317 / (math.pi * APERTURE_UM)

GAMMA_MEASURED = 1.0       # the report states gamma 1.0 on the Fig. 3 panel
GAMMA_REFERENCE = 0.65     # the negative development the report itself names
DENSITY_MEASURED = 0.48    # "D = 0.48" above base, stated on the figure
DENSITY_REFERENCE = 1.0    # this project's rms convention: NET density 1.0
DENSITY_EXPONENT = 0.40    # Higgins and Stultz, via equation (4)

#: (printed name, profile, Wiener um^2, ASA, BS degrees). ⚠ WRITTEN OUT SO THE
#: PARSE CAN FAIL: the OCR renders every decimal point as a hyphen, so "0-555"
#: has to become 0.555 and a reader that silently produced 555 would still look
#: like a number.
TABLE1 = (
    ("Ilford Pan.F",  "ILFORD_PAN_F",      0.10,   16, 23),
    ("Kodak Plus-X",  "KODAK_PLUS_X_125",  0.14,   64, 29),
    ("Kodak Tri-X",   "KODAK_TRI_X_400TX", 0.555, 250, 35),
    ("Ilford H.P.S.", "ILFORD_HPS",        0.62,  320, 36),
)

#: Adoption verdict per stock, and the reason. ⚠ THE REFUSAL IS PART OF THE
#: RESULT: three of four reconcile and the fourth does not, and recording which
#: is what keeps the method honest.
ADOPT = {
    "KODAK_PLUS_X_125": "adopt",
    "ILFORD_HPS": "adopt",
    "KODAK_TRI_X_400TX": "control",   # Kodak's own published rms outranks a chain
    "ILFORD_PAN_F": "refuse",         # speed says a different emulsion generation
}

#: Fig. 8 lives on PDF page index 16. The clip is derived from the figure's own
#: caption position rather than guessed -- see `_figure_clip`.
FIG8_PAGE = 16
FIG8_ZOOM = 6.0


def _figure_clip(pg):
    """The plot area, bounded by the Fig. 8 caption below and the page above."""
    cap_y = None
    ws = pg.get_text("words")
    for i, w in enumerate(ws):
        if w[4] == "Fig." and i + 3 < len(ws) and ws[i + 1][4] == "8":
            cap_y = w[1]
            break
    if cap_y is None:
        return None
    return pymupdf.Rect(55.0, 60.0, 300.0, cap_y - 18.0)


def _grid(a, frac=0.55):
    """Row and column positions of the plot's ruled grid."""
    dark = (a < 128)
    h, w = a.shape

    def runs(idx):
        out = []
        for i in idx:
            if out and i - out[-1][-1] <= 2:
                out[-1].append(i)
            else:
                out.append([i])
        return [sum(g) / len(g) for g in out]

    cx = runs([i for i, v in enumerate(dark.sum(0)) if v > frac * h])
    cy = runs([i for i, v in enumerate(dark.sum(1)) if v > frac * w])
    return cx, cy


def trace_fig8(doc):
    """{label: (freq c/mm, W um^2)} for the four curves, plus the calibration.

    ⚠ CALIBRATED ON THE RULED GRID, NOT ON THE PRINTED LABELS. The Paper Capture
    OCR reads this figure's axis labels as "().1" and "!So"; two of the six are
    unusable and one is wrong. The grid is not: 16 vertical lines over 0 to
    150 c/mm and 14 horizontal lines at 0.05 um^2 each, both regular to a
    fraction of a pixel, and the one label the OCR does get right ("100")
    lands on the eleventh vertical line to within 3 px, which is the check.
    """
    pg = doc[FIG8_PAGE]
    clip = _figure_clip(pg)
    if clip is None:
        return None, "Fig. 8 caption not found"
    pix = pg.get_pixmap(matrix=pymupdf.Matrix(FIG8_ZOOM, FIG8_ZOOM),
                        clip=clip, colorspace=pymupdf.csGRAY)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

    cx, cy = _grid(a)
    if len(cx) != 16 or len(cy) != 14:
        return None, f"grid is {len(cx)}x{len(cy)}, expected 16x14"

    x0, x1 = cx[0], cx[-1]                       # 0 and 150 cycles/mm
    ybot, ytop = cy[-1], cy[0]                   # 0 and 0.65 um^2
    fx = lambda px: (px - x0) * 150.0 / (x1 - x0)
    fy = lambda py: (ybot - py) * 0.65 / (ybot - ytop)

    dark = (a < 128)
    gridcols = {int(round(v)) for v in cx}
    # Curves start in a known order at the left edge: the table's own values.
    order = [("H.P.S.", 0.62), ("Tri-X", 0.555), ("Plus-X", 0.14), ("Pan.F", 0.10)]
    ypred = {nm: ybot - v / 0.65 * (ybot - ytop) for nm, v in order}
    out = {nm: [] for nm, _ in order}

    for col in range(int(x0) + 3, int(x1) - 2):
        if any(abs(col - g) <= 2 for g in gridcols):
            continue
        idx = np.flatnonzero(dark[:, col])
        idx = idx[(idx > ytop - 5) & (idx < ybot - 2)]
        if idx.size == 0:
            continue
        # Group into runs and keep the thin ones: a curve is 3-8 px thick here,
        # the in-plot text labels are far thicker and are rejected by width.
        segs, run = [], [idx[0]]
        for v in idx[1:]:
            if v - run[-1] <= 2:
                run.append(v)
            else:
                segs.append(run)
                run = [v]
        segs.append(run)
        cands = [sum(r) / len(r) for r in segs if len(r) <= 14]
        if not cands:
            continue
        for nm, _ in order:
            best = min(cands, key=lambda c: abs(c - ypred[nm]))
            if abs(best - ypred[nm]) <= 16:
                ypred[nm] = best
                out[nm].append((fx(col), fy(best)))

    return {nm: (np.array([p[0] for p in v]), np.array([p[1] for p in v]))
            for nm, v in out.items()}, None


def parse_table1(doc):
    """The four printed rows, with the OCR's hyphen-for-point undone."""
    txt = "\n".join(p.get_text() for p in doc)
    i = txt.find("TABLE 1.")
    if i < 0:
        return {}
    body = txt[i:i + 700]
    got = {}
    for printed, profile, _, _, _ in TABLE1:
        key = printed.split()[-1]
        m = re.search(re.escape(key) + r"\s*\n\s*(\d+)\s*\n\s*(\d+)[°o]?\s*\n\s*"
                      r"([\d]+)[-.]([\d]+)", body)
        if m:
            got[profile] = (int(m.group(1)), int(m.group(2)),
                            float(m.group(3) + "." + m.group(4)))
    return got


def chain(w_um2):
    """Wiener mean -> this project's rms granularity convention, x1000."""
    raw = 1000.0 * math.sqrt(w_um2 / APERTURE_AREA)
    fg = math.sqrt(GAMMA_REFERENCE / GAMMA_MEASURED)
    fd = (DENSITY_REFERENCE / DENSITY_MEASURED) ** DENSITY_EXPONENT
    return raw, raw * fg * fd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args()

    pdf = Path(ns.root).resolve() / "PDF" / "PROFILES" / SHEET
    if not pdf.is_file():
        print(f"  [SKIP] source not present: {pdf}")
        return 0
    doc = pymupdf.open(pdf)
    print(f"[i] {SOURCE}\n")

    bad = 0
    txt = "\n".join(p.get_text() for p in doc)
    if "T-IOl/2" not in txt and "T-101/2" not in txt and "T-IOI/2" not in txt:
        print("  [FAIL] this file does not identify itself as T-101/2")
        return 1
    print(f"  [OK  ] {doc.page_count} pages, raster with an OCR layer, "
          f"identifies as Research Report T-101/2 (1964/4)")

    # ---- Table 1 ----------------------------------------------------------
    got = parse_table1(doc)
    for printed, profile, w, asa, bs in TABLE1:
        g = got.get(profile)
        ok = g is not None and abs(g[2] - w) < 1e-9 and g[0] == asa and g[1] == bs
        print(f"  [{'OK  ' if ok else 'FAIL'}] Table 1 {printed:14s} "
              f"ASA {asa:4d}  BS {bs} deg  W {w:.3f} um^2"
              + ("" if ok else f"  <- read {g}"))
        if not ok:
            bad += 1

    # ---- Fig. 8, and the flatness the closed form depends on --------------
    traced, err = trace_fig8(doc)
    if traced is None:
        print(f"  [FAIL] Fig. 8: {err}")
        bad += 1
    else:
        print(f"  [OK  ] Fig. 8 traced, grid-calibrated, aperture cutoff "
              f"{APERTURE_CUTOFF_CPMM:.1f} c/mm")
        for printed, profile, w, _, _ in TABLE1:
            key = printed.split()[-1]
            key = {"Pan.F": "Pan.F", "Plus-X": "Plus-X", "Tri-X": "Tri-X",
                   "H.P.S.": "H.P.S."}[key]
            f, y = traced[key]
            if f.size < 50:
                print(f"  [FAIL] Fig. 8 {key}: only {f.size} traced points")
                bad += 1
                continue
            m20 = float(y[f <= 20.0].mean())
            w0 = float(y[f <= 3.0].mean())
            wc = float(np.interp(APERTURE_CUTOFF_CPMM, f, y))
            flat = wc / w0
            d = abs(m20 - w) / w
            print(f"        {key:8s} traced mean 0-20 {m20:.3f} vs printed "
                  f"{w:.3f} ({100*d:4.1f} %)   W(cutoff)/W(0) = {flat:.3f}")
            # The trace must reproduce the printed table: same measurement, two
            # renderings. 8 per cent is the scan grade, not the tolerance of the
            # conversion below.
            if d > 0.08:
                print(f"  [FAIL] traced Fig. 8 disagrees with Table 1 for {key}")
                bad += 1
            # And it must be flat enough for sigma^2 = W/area to hold.
            if flat < 0.85:
                print(f"  [FAIL] {key} is not flat across the aperture passband "
                      f"({flat:.3f}); the closed form does not apply and the "
                      f"integral has to be done numerically")
                bad += 1

    # ---- the chain, and the control ---------------------------------------
    print(f"\n  -- conversion to this project's convention "
          f"(net D {DENSITY_REFERENCE}, gamma {GAMMA_REFERENCE}, "
          f"{APERTURE_UM:.0f} um aperture)")
    try:
        import film_profiles as fp
    except Exception as exc:                              # pragma: no cover
        print(f"    [note] film_profiles unavailable ({exc})")
        return 1 if (bad and ns.do_assert) else 0

    by = {q.name: q for q in fp.FILM_PROFILES}
    for printed, profile, w, asa, bs in TABLE1:
        raw, conv = chain(w)
        p = by.get(profile)
        stored = p.grain.rms_granularity if p else float("nan")
        verdict = ADOPT[profile]
        delta = 100.0 * (conv - stored) / stored
        print(f"     {printed:14s} raw {raw:5.2f} -> chain {conv:5.2f}   "
              f"stored {stored:5.1f}   {delta:+6.1f} %   [{verdict}]")
        if verdict == "control" and abs(delta) > 15.0:
            print(f"  [FAIL] the control disagrees by {delta:+.1f} %, so the "
                  f"chain is not validated and nothing may be adopted from it")
            bad += 1
        if verdict == "adopt" and abs(delta) > 15.0:
            print(f"  [FAIL] {profile} moved {delta:+.1f} %, beyond what the "
                  f"control licenses")
            bad += 1
        if verdict == "refuse" and abs(delta) < 25.0:
            print(f"  [FAIL] {profile} was refused as an emulsion mismatch and "
                  f"now agrees to {delta:+.1f} % -- re-examine the refusal")
            bad += 1

    print(f"\n  [OK  ] ⚠ PAN F REFUSED: ASA 16 on the table's own "
          f"pre-revision scale doubles to about 32, and ILFORD_PAN_F is EI 50 "
          f"-- Pan F PLUS, a later emulsion. Plus-X ASA 64 -> EI 125, Tri-X "
          f"ASA 250 -> EI 400 and H.P.S. ASA 320 -> EI 400 all reconcile under "
          f"the same footnote, and Pan F is the only chain value that does not "
          f"land near its stored figure")

    print()
    if bad:
        print(f"  [FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("  [OK  ] T-101/2 Table 1 and Fig. 8 reproduced; chain validated on "
          "the Tri-X control")
    return 0


if __name__ == "__main__":
    sys.exit(main())
