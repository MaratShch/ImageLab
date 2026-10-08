#!/usr/bin/env python3
"""AUDIT: KODAK PROFESSIONAL EKTAR 100 Film, publication E-4046 (February 2016).

Re-reads the sheet and asserts that what KODAK_EKTAR_100 stores still follows
from it. Added 2026-10-07 with the owner's E-4046 re-read, which found the
2026-09-20 pass had taken the curves, the MTF and the spectral set off page 4
and left the rest of the document unread.

    page 1  «SIZES AVAILABLE»           base: 135 0.13 mm acetate, 120 0.10 mm
                                        acetate, sheets 0.19 mm ESTAR Thick Base
    page 2  «Adjustments for Long ...»  no correction 1/10,000 s to 1 s
    page 3  «JUDGING NEGATIVE EXPOSURES» four Status M red aim ranges
    page 3  «Print Grain Index»          135 / 120 / sheet tables
    page 4  E4046A characteristic        re-derived; must equal the stored curves
    page 4  E4046C spectral dye density  neutral + D-min pair, traced here

The vector panels are read with kodak_still_curves' own primitives
(find_panels / extract_panel / measure_char / resample), so this file adds no
second reader. ⚠ The one thing it does differently is the dye-pair extent
check: kodak_still_curves.assign_dye_pair refuses any panel whose traces do
not reach 700 nm, and E-4046's are drawn only to 684.93 nm. That is the
drawing, not a mis-calibrated axis -- a shifted axis moves BOTH ends, and the
starts sit at 400.06 / 399.99 nm -- so the pair is assigned here by the same
two rules (mean density orders them; they must never cross) plus a check that
both traces START at 400 nm and END at the SAME wavelength.

    python3 kodak_e4046_2016.py --root <tree> [--assert]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pymupdf                                   # noqa: E402
import kodak_still_curves as ksc                 # noqa: E402

PDF = "e4046_ektar_100-2016.pdf"
STOCK = "KODAK_EKTAR_100"
D_TOL = 0.002        # density, both the characteristic and the dye pair
G_TOL = 0.003        # gamma


def _text(doc, pno):
    return re.sub(r"\s+", " ", doc[pno - 1].get_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()
    path = root / "PDF" / "PROFILES" / "KODAK" / PDF
    if not path.is_file():
        print("SKIP: %s not on this tree" % path)
        return 0

    import film_profiles as fp
    prof = {p.name: p for p in fp.FILM_PROFILES}[STOCK]
    doc = pymupdf.open(str(path))
    bad = []

    def chk(ok, msg):
        print("[%s] %s" % ("OK  " if ok else "FAIL", msg))
        if not ok:
            bad.append(msg)

    # ---- page 4: the vector panels ------------------------------------------
    page = doc[3]
    seen = set()
    cwd = os.getcwd()
    os.chdir(str(root))
    try:
        panels = list(ksc.find_panels(page, PDF))
    finally:
        os.chdir(cwd)
    for kind, _txt, box, lx, ly, letters, exp in panels:
        pan = ksc.extract_panel(page, box, letters=letters, log_x=lx, log_y=ly,
                                expect=exp)
        if pan is None:
            continue
        if kind == "char":
            seen.add("char")
            for ch, tc in (("R", prof.curves.r), ("G", prof.curves.g),
                           ("B", prof.curves.b)):
                m = ksc.measure_char(pan.traces[ch])
                chk(abs(m["dmin"] - tc.dmin) <= D_TOL
                    and abs(m["gamma"] - tc.gamma) <= G_TOL,
                    "E4046A %s: traced dmin %.4f gamma %.4f, stored %.4f / %.4f"
                    % (ch, m["dmin"], m["gamma"], tc.dmin, tc.gamma))
        elif kind == "dye":
            seen.add("dye")
            tr = pan.unlabelled
            chk(len(tr) == 2, "E4046C: %d traces, expected the neutral + D-min "
                              "pair" % len(tr))
            if len(tr) != 2:
                continue
            hi, lo = sorted(tr, key=lambda t: sum(p[1] for p in t) / len(t),
                            reverse=True)
            cross = sum(1 for x, y in lo
                        if ksc._at_x(hi, x) is not None
                        and ksc._at_x(hi, x) < y - 0.02)
            chk(cross == 0, "E4046C: the midscale neutral sits above D-min at "
                            "every traced wavelength (%d crossings)" % cross)
            ends = [max(p[0] for p in t) for t in (hi, lo)]
            starts = [min(p[0] for p in t) for t in (hi, lo)]
            chk(all(abs(s0 - 400.0) <= ksc.DYE_NM_TOL for s0 in starts)
                and abs(ends[0] - ends[1]) < 0.05 and 680.0 <= ends[0] < 690.0,
                "E4046C: both traces start at 400 nm (%.2f / %.2f) and end "
                "TOGETHER at %.2f nm -- the drawing stops short of 700, the "
                "axis is not shifted" % (starts[0], starts[1], ends[0]))
            dd = prof.dye_density
            for key, stored, t in (("neutral", dd.d_neutral, hi),
                                   ("D-min", dd.d_dmin, lo)):
                got = [v for _x, v in ksc.resample(t, 405.0, 680.0, 5.0)]
                worst = max(abs(a - b) for a, b in zip(got, stored)) \
                    if len(got) == len(stored) else 9.9
                chk(dd.lambda_start_nm == 405.0 and dd.lambda_step_nm == 5.0
                    and len(stored) == 56 and worst <= D_TOL,
                    "E4046C %s: stored 405-680 nm array reproduces the trace "
                    "to %.4f D" % (key, worst))
            d_lo = min(dd.d_dmin)
            chk(dd.d_dmin[7] > dd.d_dmin[-1] and d_lo < 0.21,
                "E4046C: D-min falls from %.3f (440 nm) to %.3f -- the orange "
                "mask" % (dd.d_dmin[7], d_lo))
    chk(seen == {"char", "dye"}, "page 4: characteristic and dye panels both "
                                 "located (%s)" % sorted(seen))

    # ---- printed text --------------------------------------------------------
    t1, t2, t3 = _text(doc, 1), _text(doc, 2), _text(doc, 3)
    chk("0.13 mm (0.005 inch)" in t1 and "acetate" in t1
        and "KODAK ESTAR" in t1, "p1: 135 base 0.13 mm acetate, sheets ESTAR")
    em = prof.emulsion
    chk(em.base_um == 130.0 and em.base_material == "acetate",
        "stored base %.1f um %s" % (em.base_um, em.base_material))
    chk(re.search(r"No filter correction or exposure compensation is required "
                  r"for exposures from 1.10,000 second to 1 second", t2)
        is not None, "p2: the 1/10,000 s to 1 s reciprocity bound is printed")
    rt = prof.reciprocity_table
    chk(rt.times_s == (1.0,) and rt.stops_correction == (0.0,),
        "stored reciprocity table %s / %s" % (rt.times_s, rt.stops_correction))
    chk(all(s in t3 for s in ("0.77 to 0.87", "1.13 to 1.23", "1.08 to 1.18",
                              "0.93 to 1.03")),
        "p3: the four aim-density ranges are printed")
    ad = prof.aim_density[0] if prof.aim_density else None
    chk(ad is not None and ad.gray_card == (0.77, 0.87)
        and ad.gray_scale == (1.13, 1.23) and ad.forehead_light == (1.08, 1.18)
        and ad.forehead_dark == (0.93, 1.03) and ad.exposure_index == 100,
        "stored aim densities match p3")
    chk("38" in t3 and "66" in t3 and "less than 25" in t3,
        "p3: the Print Grain Index tables are printed")
    pg = prof.print_grain_index
    chk(pg.fmt_135 == (0.0, 38.0, 66.0) and pg.fmt_120 == (0.0, 0.0, 38.0)
        and pg.fmt_sheet == (0.0, 0.0, 0.0),
        "stored PGI 135 %s, 120 %s, sheet %s" % (pg.fmt_135, pg.fmt_120,
                                                pg.fmt_sheet))
    doc.close()

    if bad:
        print("\n[FAIL] %d E-4046 check(s) do not reproduce" % len(bad))
        return 1 if ns.do_assert else 0
    print("\n[OK] E-4046 (2016): characteristic curves re-derived to the stored "
          "values, the spectral-dye-density pair reproduced to %.3f D, and the "
          "printed base, reciprocity bound, aim densities and Print Grain Index "
          "found on the page and in the profile" % D_TOL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
