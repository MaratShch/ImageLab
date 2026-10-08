#!/usr/bin/env python3
"""AUDIT: Agfa-Gevaert AVIPHOT PAN 400S PE1/PE0 sheet (January 2006), 2026-10-07.

Checks that every printed number AGFA_AVIPHOT_PAN_400S stores is on the page,
that the stored curve is the adopted record of AVIPHOT_400S_CURVE_FITS (shift
applied), and re-measures the spectral identity with ROLLEI_INFRARED_400 that
the profile records (not used by the render).

    python3 agfa_aviphot_400s_2006.py --root <tree> [--assert]
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np      # noqa: E402
import pymupdf          # noqa: E402

PDF = "AVIPHOT PAN 400S PE1.pdf"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args(argv)
    path = Path(ns.root).resolve() / "PDF" / "PROFILES" / "AGFA" / PDF
    if not path.is_file():
        print("SKIP: %s not on this tree" % path)
        return 0
    import film_profiles as fp
    P = {p.name: p for p in fp.FILM_PROFILES}
    p = P["AGFA_AVIPHOT_PAN_400S"]
    doc = pymupdf.open(str(path))
    t = {i + 1: re.sub(r"\s+", " ", doc[i].get_text()) for i in range(len(doc))}
    bad = []

    def chk(ok, msg):
        print("[%s] %s" % ("OK  " if ok else "FAIL", msg))
        if not ok:
            bad.append(msg)

    chk("PE1: 0.10mm" in t[2] and p.emulsion.base_um == 100.0, "p1: PE1 base 0.10 mm stored as 100 um")
    chk("RMS = 14 at Density = 1" in t[4] and "50 µm spot" in t[4]
        and abs(p.grain.rms_granularity - 14 * 50 / 48) < 0.01, "p3: RMS 14 at 50 um, stored 14.58 at 48 um")
    chk("1000:1 = 161 line pairs" in t[4] and "1,6:1 = 40.3 line pairs" in t[4]
        and p.mtf.resolving_power_lp_mm_highc == 161.0 and p.mtf.resolving_power_lp_mm_lowc == 40.3,
        "p3: resolving power 161 / 40.3 lp/mm printed and stored")
    pts = {(q.developer, q.celsius, round(q.minutes, 2)): (q.exposure_index, q.contrast_index, q.base_fog)
           for q in p.processing_family.points}
    want = {("G 74 c", 30.0, 0.33): (200, 0.57, 0.08), ("G 74 c", 30.0, 0.7): (400, 0.90, 0.09),
            ("G 74 c", 30.0, 1.17): (500, 1.10, 0.12), ("G 74 c", 37.0, 0.33): (370, 0.73, 0.10),
            ("G 74 c", 37.0, 0.7): (550, 1.07, 0.19), ("G 74 c", 40.0, 0.33): (450, 0.87, 0.12),
            ("G 74 c", 40.0, 0.7): (580, 1.06, 0.30), ("G 74 c + AD 74", 30.0, 0.33): (355, 0.90, 0.10),
            ("G 74 c + AD 74", 30.0, 0.7): (515, 1.19, 0.15), ("G 74 c + AD 74", 30.0, 1.17): (630, 1.16, 0.24)}
    chk(pts == want, "p4: the ten (EAFS, average gradient, fog) cells stored as printed")
    nums = re.findall(r"\d+\.\d\d|\d{3}", t[5])
    chk(all(s in nums for s in ("0.57", "0.90", "1.10", "1.19", "630", "0.30")), "p4: table values found on the page")
    c = p.curves.g
    sh = fp.AVIPHOT_400S_CURVE_X_SHIFT
    q = fp.AVIPHOT_400S_CURVE_FITS[("G 74 c 30 C 42 s", "adopted")][0]
    st = (c.dmin, c.gamma, c.toe_x - sh, c.toe_k, c.shoulder_x - sh, c.shoulder_k)
    chk(all(abs(a - b) < 6e-4 for a, b in zip(st, q)), "stored curve = adopted 42 s record (shift %+.2f)" % sh)
    chk(p.exposure_index == 400 and p.speed_criterion == "manufacturer_ei", "EAFS 400 at 42 s, flagged manufacturer_ei")
    # spectral identity with ROLLEI_INFRARED_400 (recorded, not used)
    def arr(n):
        s = P[n].spectral
        lam = s.lambda_start_nm + s.lambda_step_nm * np.arange(len(s.log_s_pan))
        return lam, np.array(s.log_s_pan)
    la, va = arr("AGFA_AVIPHOT_PAN_400S")
    lb, vb = arr("ROLLEI_INFRARED_400")
    m = (lb >= 400) & (lb <= 800) & (vb > -3.9)
    a = np.interp(lb[m], la, va)
    r = a - np.mean(a - vb[m]) - vb[m]
    rms = float(np.sqrt(np.mean(r ** 2)))
    chk(rms < 0.10, "spectral: AVIPHOT 400S and ROLLEI_INFRARED_400 curves agree to rms %.3f log (400-800 nm)" % rms)
    doc.close()
    if bad:
        print("\n[FAIL] %d AVIPHOT 400S check(s)" % len(bad))
        return 1 if ns.do_assert else 0
    print("\n[OK] AVIPHOT PAN 400S (2006): base, RMS, resolving power and the ten processing cells found on "
          "the page and in the profile; stored curve = adopted trace; spectral identity with Rollei INFRARED %.3f log" % rms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
