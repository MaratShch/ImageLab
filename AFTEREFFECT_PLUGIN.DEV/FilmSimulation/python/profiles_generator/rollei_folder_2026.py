#!/usr/bin/env python3
"""AUDIT: the ROLLEI folder, re-read 2026-10-07 (owner request).

Re-derives what the database adopted from the Rollei / MACO documents and
asserts the stored values still follow from them:

    R3 product information (Oct 2004)  p9  characteristic curve (VECTOR)
                                       p12 reciprocity table
    SUPERPAN 200 R210701 (Jul 2021)    p3  spectral sensitivity (VECTOR)
                                       p3  Schwarzschild table
    INFRARED (Oct 2005) / R210701      p2  RMS 11, 160 lines/mm, 7.5 um, 100 um PET
    RETRO 100/400 (Jan 2008)           p2  base 120 / 95 um triacetate, 10 um
    PAN 25 (Oct 2005)                  p2  100 um polyester, ISO 25

Raster traces (INFRARED p2 curve, PAN 25 curves) are not re-traced here;
their fits are recorded in film_profiles.ROLLEI_CURVE_FITS and the stored
curves are checked against those records (shift applied).

    python3 rollei_folder_2026.py --root <tree> [--assert]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                               # noqa: E402
import pymupdf                                   # noqa: E402

DIR = ("PDF", "PROFILES", "ROLLEI")
R3 = "TARoR3_e.pdf"
SP = "SUPERPAN200_Data-Sheet_EN_R210701.pdf"
IR05 = "Rollei_Infrared.pdf"
IR21 = "INFRARED_Data-Sheet_EN_R210701.pdf"
RT = "TARRete.pdf"
P25 = "PAN25eng.pdf"


def _bez(p0, p1, p2, p3, n=60):
    out = []
    for i in range(n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
        out.append((a * p0.x + b * p1.x + c * p2.x + d * p3.x,
                    a * p0.y + b * p1.y + c * p2.y + d * p3.y))
    return out


def _pts(dr):
    pts = []
    for it in dr["items"]:
        if it[0] == "l":
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        elif it[0] == "c":
            pts += _bez(*it[1:5])
    return pts


def _sp(v, k):
    z = v / k
    return np.where(z > 60, v, k * np.log1p(np.exp(np.minimum(z, 60))))


def _model(c, x):
    return c.dmin + c.gamma * (_sp(x - c.toe_x, c.toe_k) - _sp(x - c.shoulder_x, c.shoulder_k))


def _txt(doc, pno):
    return re.sub(r"\s+", " ", doc[pno - 1].get_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ns = ap.parse_args(argv)
    d = Path(ns.root).resolve().joinpath(*DIR)
    if not (d / R3).is_file():
        print("SKIP: %s not on this tree" % (d / R3))
        return 0
    import film_profiles as fp
    P = {p.name: p for p in fp.FILM_PROFILES}
    bad = []

    def chk(ok, msg):
        print("[%s] %s" % ("OK  " if ok else "FAIL", msg))
        if not ok:
            bad.append(msg)

    # ---- R3 p9 characteristic curve ------------------------------------------
    doc = pymupdf.open(str(d / R3))
    cur = [x for x in doc[8].get_drawings() if len(x["items"]) == 5]
    chk(len(cur) == 1, "R3 p9: one 5-segment curve path (%d)" % len(cur))
    if cur:
        xy = np.array([((x - 104.4) / (331.2 - 104.4) * 4.2, (234.1 - y) / (234.1 - 52.7) * 3.0)
                       for x, y in _pts(cur[0])])
        c = P["ROLLEI_R3"].curves.g
        sh = fp.ROLLEI_CURVE_X_SHIFT["ROLLEI_R3"]
        e = _model(c, np.clip(xy[:, 0], 0, 4.2) + sh) - xy[:, 1]
        rms = float(np.sqrt(np.mean(e ** 2)))
        chk(rms <= 0.010 and float(np.abs(e).max()) <= 0.020,
            "R3 p9: stored curve reproduces the vector trace, rms %.4f D max %.4f D" % (rms, np.abs(e).max()))
    t12 = _txt(doc, 12)
    pairs = [(1, 2), (2, 4), (4, 10), (8, 20), (15, 60), (30, 150), (60, 350)]
    chk(all(re.search(r"\b%d %d\b" % pr, t12) for pr in pairs), "R3 p12: the seven printed reciprocity pairs")
    rt = P["ROLLEI_R3"].reciprocity_table
    want = [(b, math.log2(b / a)) for a, b in pairs]
    ok = rt.times_s[0] == 0.5 and rt.stops_correction[0] == 0.0 and all(
        abs(t - w[0]) < 1e-9 and abs(s - w[1]) < 6e-5 for t, s, w in zip(rt.times_s[1:], rt.stops_correction[1:], want))
    chk(ok, "R3: stored table = (actual, log2(actual/meter)) of the printed pairs")
    t8 = _txt(doc, 8)
    chk("Polyester, undyed, 100µm" in t8 and P["ROLLEI_R3"].emulsion.base_um == 100.0,
        "R3 p8: polyester 100 um printed and stored")
    doc.close()

    # ---- SUPERPAN 200 p3 spectral + Schwarzschild -----------------------------
    doc = pymupdf.open(str(d / SP))
    pg = doc[2]
    cur = [x for x in pg.get_drawings() if len(x["items"]) == 9 and all(i[0] == "c" for i in x["items"])]
    chk(len(cur) == 1, "SUPERPAN p3: one 9-Bezier spectral path (%d)" % len(cur))
    if cur:
        xt = np.array([332.4, 355.6, 378.8, 401.9, 425.1, 448.3, 471.4, 494.6, 517.8, 540.9, 564.1])
        ax = np.polyfit(xt, np.arange(300, 801, 50), 1)
        ay = np.polyfit([197.2, 168.6, 139.9, 111.3, 82.6], [0, 0.5, 1.0, 1.5, 2.0], 1)
        xy = np.array(sorted((float(np.polyval(ax, x)), float(np.polyval(ay, y))) for x, y in _pts(cur[0])))
        mx = xy[:, 1].max()
        sp = P["ROLLEI_SUPERPAN_200"].spectral
        got = [float(np.interp(l, xy[:, 0], xy[:, 1])) - mx for l in range(340, 771, 10)]
        stored = list(sp.log_s_pan[:-1])
        worst = max(abs(a - b) for a, b in zip(got, stored))
        chk(sp.lambda_start_nm == 340.0 and len(sp.log_s_pan) == 45 and worst <= 0.006 and sp.log_s_pan[-1] == -4.0,
            "SUPERPAN p3: stored 340-770 nm curve reproduces the trace to %.4f log (peak %.0f nm)"
            % (worst, xy[np.argmax(xy[:, 1]), 0]))
    t3 = _txt(doc, 3)
    chk(all(s in t3 for s in ("1 – 2 sec", "3 – 4 sec", "8 sec 24 sec", "15 sec 60 sec", "30 sec 180 sec")),
        "SUPERPAN p3: the Schwarzschild rows are printed")
    rt = P["ROLLEI_SUPERPAN_200"].reciprocity_table
    chk(rt.times_s[3:] == (8.0, 24.0, 60.0, 180.0) and abs(rt.stops_correction[4] - math.log2(3)) < 1e-4,
        "SUPERPAN: stored exact rows 8 / 24 / 60 / 180 s")
    t2 = _txt(doc, 2)
    g = P["ROLLEI_SUPERPAN_200"]
    chk("RMS (× 1000) = 14" in t2 and g.grain.rms_granularity == 14.0, "SUPERPAN p2: RMS 14 printed and stored")
    chk("180 lp/mm" in t2 and g.mtf.resolving_power_lp_mm_highc == 180.0, "SUPERPAN p2: 180 lp/mm printed and stored")
    chk("PET 100 micron" in t2 and g.emulsion.base_um == 100.0 and g.emulsion.coated_um == 10.0,
        "SUPERPAN p2: PET 100 um, 10 um layer printed and stored")
    doc.close()

    # ---- INFRARED, RETRO, PAN 25 text ----------------------------------------
    t = _txt(pymupdf.open(str(d / IR05)), 2)
    ir = P["ROLLEI_INFRARED_400"]
    chk("RMS (x1000) 11.0" in t and "7,5µm" in t and "100µm" in t and ir.grain.rms_granularity == 11.0
        and ir.emulsion.coated_um == 7.5 and ir.emulsion.base_um == 100.0,
        "INFRARED 2005 p2: RMS 11, 7.5 um, 100 um polyester printed and stored")
    t = _txt(pymupdf.open(str(d / IR21)), 2)
    chk("Granularity RMS (× 1000) = 11" in t and "160 lines/mm" in t, "INFRARED R210701 p2: RMS 11 and 160 lines/mm confirmed")
    t = _txt(pymupdf.open(str(d / RT)), 2)
    rr = P["ROLLEI_RETRO_400"]
    chk("120µm" in t and "cellulose triacetate" in t and "RETRO 400 – 10µm" in t
        and rr.emulsion.base_um == 120.0 and rr.emulsion.coated_um == 10.0,
        "RETRO 2008 p2: 120 um triacetate, 10 um printed and stored")
    t = _txt(pymupdf.open(str(d / P25)), 2)
    pp = P["ROLLEI_PAN_25"]
    chk("polyester, 100µm" in t and "ISO 25/15" in t and pp.exposure_index == 25 and pp.emulsion.base_um == 100.0,
        "PAN 25 p2: ISO 25, 100 um polyester printed and stored")

    # ---- every adopted record matches its profile ------------------------------
    for k, (q, rms, mx, gb) in fp.ROLLEI_CURVE_FITS.items():
        if k[3] != "adopted":
            continue
        c = P[k[0]].curves.g
        sh = fp.ROLLEI_CURVE_X_SHIFT[k[0]]
        st = (c.dmin, c.gamma, c.toe_x - sh, c.toe_k, c.shoulder_x - sh, c.shoulder_k)
        chk(all(abs(a - b) < 6e-4 for a, b in zip(st, q)), "%s: stored curve = adopted record (shift %+.2f)" % (k[0], sh))

    if bad:
        print("\n[FAIL] %d ROLLEI check(s) do not reproduce" % len(bad))
        return 1 if ns.do_assert else 0
    print("\n[OK] ROLLEI folder: R3 curve and SUPERPAN spectral re-traced to the stored values, "
          "both reciprocity tables re-derived, printed base / layer / RMS / resolving figures found on "
          "the page and in the profiles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
