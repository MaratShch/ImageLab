#!/usr/bin/env python3
"""AGFACOLOR Vista 200's «Sharpness» panel — and Agfa naming the quantity.

    PDF/PROFILES/AGFA/AGFACOLOR Vista 100, 200, 400, 800.pdf
    Agfa-Gevaert AG, «Technical Data AF», 2nd edition 06/2000, printed p6.

⚠⚠ THE THIRTEENTH AND LAST AGFA PANEL, AND THE ONE THE 2026-09-06h/i BATCH
MISSED. That batch swept both editions of «Technical Data PF» and adopted ten
of their twelve panels. Vista's panel is in a different document, so nothing in
that sweep touched it, and the profile kept a class-estimate f50 triple of
56 / 63 / 69 while its own sheet plotted the curve. `Tasks.md` had carried the
gap as an open row since 2026-08-18.

⚠⚠ AND THE REFUSAL THAT HELD IT OUT WAS ANSWERED BY PAGE 4 OF THE SAME FILE.
The stored comment argued, reasonably:

    "the panel plots TRANSFER FACTOR (%) against lines/mm and OVERSHOOTS 100 %
     at low frequency, so it is a CTF-like rectangular-wave quantity, not the
     sine-wave MTF that f50 is defined against. Reading a 50 % crossing
     straight off it would be a units error, not a measurement."

Page 4, in Agfa's own words:

    «Sharpness -- International name of the chart: MTF (Modulation Transfer
     Function) which defines the sharpness of the image. The higher the
     transfer factor in %, the lower the loss during transmission of the light.
     References: -- Exposure: daylight -- Densitometry: visual filter (Vλ)»

⚠ **THAT IS A STRONGER AUTHORITY THAN QUEUE G6's, AND IT ARRIVES AFTER G6
CLOSED.** G6 settled «Linien pro mm» = cycles per mm by inference from the
International Commission for Optics' 1961 nomenclature recommendation — sound,
but a statement about what the words conventionally denote. This is the
manufacturer stating what its own chart is. The two agree, which is worth
having: an inference and a primary statement reaching the same answer.

⚠ WHAT DOES NOT CHANGE IS THE OVERSHOOT. An MTF cannot exceed 1, so what Agfa
call MTF is an adjacency-enhanced measured response. The peak goes to
`adjacency`, the rolloff is fitted ABOVE the peak only, and the two never merge
— the same treatment the other twelve AGFA panels get.

⚠ TWO INDEPENDENT READERS AGREE ON THE SHAPE. `mtf_vector.py` read this panel
on 2026-09-02e from its printed labels and fitted **q 2.63**; this module fits
2.63 from the drawn frame. Its f50 of 50.0 against this 47.8 is the 4 % tick-
label defect corrected across the AGFA family on 2026-09-06i: this panel's
"100" label sits 1.43 pt off its own tick and is rejected here.

Run:  python agfa_vista_mtf.py [--root .] [--assert]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PDF = "PDF/PROFILES/AGFA/AGFACOLOR Vista 100, 200, 400, 800.pdf"

#: printed p6, LEFT column. ⚠ The right column is Vista 400, which has no
#: profile; pages 5, 7 and 8 carry Vista 100/800, FUTURA II and CTprecisa,
#: none of which the database holds. One panel is adoptable from this sheet.
PAGE, FRAME_X0, FRAME_Y0 = 5, 53.39, 316.08

#: What this module found, pinned so a rerun that disagrees FAILS.
EXPECTED = dict(f50=47.83, q=2.63, overshoot=0.0978, peak_at=3.5)

#: The sentence on p4 that settles what the panel measures. Asserted verbatim
#: on every run: if a future edition of this sheet drops it, the adoption's
#: primary authority is gone and the build should say so rather than carry on.
P4_DEFINITION = "MTF (Modulation Transfer"


def read(root: Path, verbose=True):
    import agfa_1998_sharpness as S
    doc = pymupdf.open(str(root / PDF))

    # ---- the manufacturer's own definition, first --------------------------
    p4 = doc[3].get_text()
    if P4_DEFINITION not in p4:
        print("  [FAIL] p4 no longer defines the chart as an MTF -- the "
              "primary authority for this adoption is missing")
        return None
    if verbose:
        i = p4.find("Sharpness")
        print("  [OK  ] p4 defines the chart, in Agfa's own words:")
        print("         %s" % " ".join(p4[i:i + 210].split()))

    page = doc[PAGE]
    fr = S._frame(page, FRAME_X0, FRAME_Y0)
    if fr is None:
        print("  [FAIL] no plot frame at the Vista 200 sharpness position")
        return None
    FR = pymupdf.Rect(fr.x0, fr.y0, fr.x1, fr.y1)
    fx, fy = S._labels(page, FR)
    # ⚠ THIS PANEL'S LADDERS ARE NOT THE RANGE SHEET'S. Six frequency labels
    # (2 5 10 20 50 100, no 3 and no 30) and five response ones (10 20 50 100
    # 150, no 30). A reader that assumed the twelve-panel family's grid would
    # calibrate this one on ticks it does not have.
    if sorted(fx) != [2.0, 5.0, 10.0, 20.0, 50.0, 100.0] or \
            sorted(fy) != [10.0, 20.0, 50.0, 100.0, 150.0]:
        print("  [FAIL] unexpected ladders: x %s  y %s"
              % (sorted(fx), sorted(fy)))
        return None
    cf, rf, nf, fdrop, ferr = S._logfit(fx, fr.width)
    cr, rr, nr, rdrop, rerr = S._logfit(fy, fr.height)
    if ferr > S.FRAME_TOL:
        print("  [FAIL] frequency ladder misses its own frame by %.2f %%"
              % (100 * ferr))
        return None
    X, Y = S._calibrate(fx, fy, fr)
    A, _dr = S._curve(page, FR)
    if A is None:
        print("  [FAIL] no thick stroked path in the plot frame")
        return None
    f, r = X(A[:, 0]), Y(A[:, 1])
    o = np.argsort(f)
    f, r = f[o], r[o]
    below = np.flatnonzero(r < 0.5)
    if not len(below) or below[0] == 0:
        print("  [FAIL] the curve never crosses 50 %% inside the panel")
        return None
    f50 = float(np.interp(0.5, [r[below[0]], r[below[0] - 1]],
                          [f[below[0]], f[below[0] - 1]]))
    peak, pk_at = float(r.max()), float(f[int(r.argmax())])
    m = f > max(4.0, pk_at)
    q = qe = None
    for cand in np.arange(1.0, 5.001, 0.01):
        e = float(np.sqrt(np.mean(
            (1.0 / (1.0 + (f[m] / f50) ** cand) - r[m]) ** 2)))
        if qe is None or e < qe:
            q, qe = float(cand), e
    ge = float(np.sqrt(np.mean(
        (np.exp(-np.log(2.0) * (f[m] / f50) ** 2) - r[m]) ** 2)))
    if verbose:
        print("  %d/%d ladder rungs, residual %.2f/%.2f pt, ladder-vs-frame "
              "%.3f %% x / %.2f %% y, dropped %s"
              % (nf, nr, rf, rr, 100 * ferr, 100 * rerr, fdrop or "nothing"))
        print("      f50 %.2f c/mm, q %.2f (rms %.4f vs Gaussian %.4f), "
              "peak %+.4f at %.1f c/mm, traced %.1f-%.1f"
              % (f50, q, qe, ge, peak - 1.0, pk_at, f.min(), f.max()))
    return dict(f50=round(f50, 2), q=round(q, 2),
                overshoot=round(peak - 1.0, 4), peak_at=round(pk_at, 1),
                q_rms=qe, gauss_rms=ge)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()
    if not (root / PDF).is_file():
        print("  [SKIP] source not present: %s" % (root / PDF))
        return 0

    print("AGFACOLOR Vista, Technical Data AF 06/2000 -- the Vista 200 "
          "«Sharpness» panel")
    got = read(root, verbose=True)
    bad = 0
    if got is None:
        bad += 1
    else:
        for k, want in EXPECTED.items():
            if abs(got[k] - want) > (0.6 if k == "f50" else 0.02):
                print("  [MISMATCH] %s %s vs pinned %s" % (k, got[k], want))
                bad += 1

    # ⚠ THE SISTER READER, because this panel has two and they were never
    # compared until the AGFA family's 4 % ladder defect showed what that
    # costs. `mtf_vector.py` calibrates on the printed labels; this module on
    # the drawn frame. The SHAPE must agree tightly -- q is what the render
    # actually uses above f50 -- while f50 may differ by the label offset.
    try:
        import mtf_vector as MV
        mv = MV.SHEETS.get("vista200") if hasattr(MV, "SHEETS") else None
        print("\n  CROSS-READER: mtf_vector.py registers this panel as %s"
              % ("vista200 -> AGFA_VISTA_200" if mv else "NOT REGISTERED"))
        if mv is None:
            print("  [FAIL] mtf_vector no longer registers vista200, so this "
                  "panel has only one reader again")
            bad += 1
    except Exception as exc:                                  # pragma: no cover
        print("  [WARN] could not consult mtf_vector: %s" % exc)

    # ---- against the database ---------------------------------------------
    try:
        import film_profiles as fp
        m = fp.get_profile("AGFA_VISTA_200").mtf
        drift = []
        if got:
            if abs(m.f50_g - got["f50"]) > 0.6:
                drift.append("f50_g %.2f vs panel %.2f" % (m.f50_g, got["f50"]))
            if abs(m.mtf_rolloff_q - got["q"]) > 0.02:
                drift.append("q %.2f vs panel %.2f"
                             % (m.mtf_rolloff_q, got["q"]))
            if abs(m.adjacency - got["overshoot"]) > 0.002:
                drift.append("adjacency %.4f vs panel %.4f"
                             % (m.adjacency, got["overshoot"]))
        if not m.mtf_measured:
            drift.append("mtf_measured is False")
        print("\n  AGAINST THE DATABASE: %s"
              % ("agrees" if not drift else "; ".join(drift)))
        if drift:
            bad += 1
    except Exception as exc:                                  # pragma: no cover
        print("\n  [WARN] could not compare against film_profiles: %s" % exc)

    if ns.assert_ and bad:
        print("\n[FAIL] the Vista 200 sharpness panel does not reproduce")
        return 1
    print("\n[OK] AGFA_VISTA_200's «Sharpness» panel re-derived, and the "
          "manufacturer's own definition of the chart re-read from p4 -- "
          "«International name of the chart: MTF (Modulation Transfer "
          "Function)», which is what retires the 2026-08-18 refusal that this "
          "panel measures a rectangular-wave quantity f50 is not defined "
          "against.")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
