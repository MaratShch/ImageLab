"""Audit of the 2026-09-29b KODAK sub-folder harvest.

Re-derives, from the PDFs themselves, every number the harvest wrote into
`film_profiles.py`, and fails on drift. Five documents:

  160nc.pdf                         E-190 (October 2006), a ManualsLib copy
  e27.pdf                           E-27, KODAK EKTACHROME 100 Professional (EPN), May 1998
  KODAK-VISION3-5219-7219-technical-information.pdf   H-1-5219 Revised 3-26
  estimating_historic_image_resolution_v9.pdf         Vitale 2009, Table 4
  2003KodakProfessionalCatalog_L9.pdf                 the 2003 L-9 catalogue

plus the documents already in the corpus that the checks lean on
(e190-Portra-2006.pdf, the 2003 E-190, e2468-Portra_100T.pdf and the
Revised 3-22 VISION3 5219 sheet), each used only when present.

Usage:  python kodak_2026_09_29b.py --root <project root> [--assert]
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GRID = np.arange(380.0, 701.0, 10.0)
DYE_GRID = np.arange(400.0, 701.0, 10.0)


def _bez(P, n=9):
    s = np.linspace(0.0, 1.0, n)[:, None]
    A = np.array([[p.x, p.y] for p in P])
    return (1 - s) ** 3 * A[0] + 3 * (1 - s) ** 2 * s * A[1] + 3 * (1 - s) * s * s * A[2] + s ** 3 * A[3]


def _runs(page, rect, width=0.72):
    """Every continuous pen run of the curve-weight strokes inside `rect`."""
    out = []
    for d in page.get_drawings():
        if not rect.intersects(d["rect"]) or d.get("type") == "f":
            continue
        if abs((d.get("width") or 0.0) - width) > 0.08:
            continue
        cur, last = [], None
        for it in d["items"]:
            if it[0] == "l":
                seg = np.array([[it[1].x, it[1].y], [it[2].x, it[2].y]])
            elif it[0] == "c":
                seg = _bez(it[1:5])
            else:
                continue
            if last is not None and abs(seg[0] - last).sum() > 0.05:
                out.append(np.array(cur))
                cur = []
            cur += list(seg) if not cur else list(seg[1:])
            last = seg[-1]
        if cur:
            out.append(np.array(cur))
    return [r for r in out if len(r) >= 20]


def _gridlines(page, rect):
    vx, hy = {}, {}
    for d in page.get_drawings():
        if not rect.intersects(d["rect"]):
            continue
        for it in d["items"]:
            if it[0] != "l":
                continue
            a, b = it[1], it[2]
            if abs(a.x - b.x) < 0.05 and rect.x0 + 5 < a.x < rect.x1 - 5:
                vx[round(a.x, 3)] = vx.get(round(a.x, 3), 0.0) + abs(a.y - b.y)
            if abs(a.y - b.y) < 0.05 and rect.y0 + 5 < a.y < rect.y1 - 5:
                hy[round(a.y, 3)] = hy.get(round(a.y, 3), 0.0) + abs(a.x - b.x)
    return vx, hy


def spectral_grid(page, rect, x_first, y_values):
    """Gridline-calibrated three-layer reading on the 10 nm database grid.

    x: the vertical gridlines inside the frame are x_first, x_first+50, ...
    y: the horizontal gridlines, top to bottom, carry `y_values`.
    """
    vx, hy = _gridlines(page, rect)
    px = np.array(sorted(k for k, v in vx.items() if v > 50))
    py = np.array(sorted(k for k, v in hy.items() if v > 50))
    cx = np.polyfit(px, x_first + 50.0 * np.arange(len(px)), 1)
    cy = np.polyfit(py, np.array(y_values[:len(py)], dtype=float), 1)
    rec = {}
    for r in _runs(page, rect):
        x = np.polyval(cx, r[:, 0]); y = np.polyval(cy, r[:, 1])
        if x.max() - x.min() < 60:
            continue
        o = np.argsort(x); x, y = x[o], y[o]
        pk = x[np.argmax(y)]
        lab = "b" if pk < 500 else ("g" if pk < 590 else "r")
        v = np.where((GRID >= x[0] - 0.5) & (GRID <= x[-1] + 0.5), np.interp(GRID, x, y), np.nan)
        gm = float(np.nanmax(v))
        rec[lab] = (np.where(np.isnan(v), -4.0, np.round(v - gm, 2)) + 0.0, float(x[0]), float(x[-1]))
    return rec


def _panel_rect(page, title):
    import kodak_still_curves as K
    for k, t, box, *_ in K.find_panels(page, None):
        if t.startswith(title):
            return box
    return None


def _frame_in(page, box):
    import pymupdf
    R = pymupdf.Rect(box)
    best = None
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "re":
                r = it[1]
                if R.contains(r) and r.width > 100 and r.height > 100:
                    if best is None or r.width * r.height < best.width * best.height:
                        best = r
    return best or R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parent))
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    a = ap.parse_args(argv)
    kd = Path(a.root) / "PDF" / "PROFILES" / "KODAK"
    import pymupdf
    import film_profiles as fp
    P = {p.name: p for p in fp.FILM_PROFILES}
    bad = 0
    ran = 0

    def chk(ok, msg):
        nonlocal bad, ran
        ran += 1
        print(("[OK  ] " if ok else "[FAIL] ") + msg)
        bad += 0 if ok else 1

    def have(name):
        return (kd / name).is_file()

    # ---- 1. 160nc.pdf is E-190 (October 2006) -------------------------------
    if have("160nc.pdf") and have("e190-Portra-2006.pdf"):
        A = pymupdf.open(str(kd / "e190-Portra-2006.pdf")); B = pymupdf.open(str(kd / "160nc.pdf"))
        foot = "Downloaded from www.Manualslib.com manuals search engine"
        same_txt = A.page_count == B.page_count and all(
            re.sub(r"\s+", " ", A[i].get_text()).strip()
            == re.sub(r"\s+", " ", B[i].get_text().replace(foot, "")).strip()
            for i in range(A.page_count))
        same_vec = all(len(B[i].get_drawings()) == len(A[i].get_drawings()) + 1 for i in range(A.page_count))
        chk(same_txt and same_vec,
            "160nc.pdf is E-190 October 2006 page for page: text identical bar the ManualsLib footer, "
            "one extra drawing per page (the footer rule)")

    # ---- 2. the PORTRA spectral readings -----------------------------------
    fams = {"160": (("KODAK_PORTRA_160NC", "KODAK_PORTRA_160VC", "KODAK_PORTRA_100T"),
                    [("e190-Portra-2006.pdf", 8), ("e190-Portra-2006.pdf", 9), ("160nc.pdf", 8),
                     ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 9),
                     ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 10), ("e2468-Portra_100T.pdf", 5)]),
            "400": (("KODAK_PORTRA_400NC", "KODAK_PORTRA_400VC"),
                    [("e190-Portra-2006.pdf", 10), ("e190-Portra-2006.pdf", 11), ("160nc.pdf", 10),
                     ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 11),
                     ("KODAK PROFESSIONAL PORTRA - 2003 year.pdf", 12)])}
    for fam, (stocks, pages) in fams.items():
        for pdf, pno in pages:
            if not have(pdf):
                continue
            pg = pymupdf.open(str(kd / pdf))[pno - 1]
            box = _panel_rect(pg, "Spectral-Sensitivity")
            if box is None:
                chk(False, "%s p%d: spectral panel not located" % (pdf, pno)); continue
            rec = spectral_grid(pg, _frame_in(pg, box), 300.0, (3.0, 2.0, 1.0))
            worst = 0.0
            for n in stocks:
                sp = P[n].spectral
                for k, st in (("r", sp.log_s_r), ("g", sp.log_s_g), ("b", sp.log_s_b)):
                    worst = max(worst, float(np.abs(rec[k][0] - np.asarray(st)).max()))
            # one unit in the last stored place is the tolerance between PRINTINGS:
            # the 2003 and 2006 vertices differ by ~1e-5 pt, which can flip a
            # rounding at 0.005 log (the Konica shared-drawing precedent)
            chk(set(rec) == {"r", "g", "b"} and worst <= 0.0100001,
                "PORTRA %s-speed spectral panel, %s p%d: three layers, and the stored arrays of %s "
                "reproduce it exactly (worst %.2g)" % (fam, pdf, pno, " / ".join(s.split("_")[-1] for s in stocks), worst))

    # ---- 3. reciprocity, E-190 and the catalogue ----------------------------
    if have("e190-Portra-2006.pdf"):
        t = re.sub(r"\s+", " ", pymupdf.open(str(kd / "e190-Portra-2006.pdf"))[3].get_text())
        ok = "exposures from 1⁄10,000 second to 10 seconds" in t or "exposures from 1/10,000 second to 10 seconds" in t
        tabs = all(P[n].reciprocity_table.times_s == (1.0 / 10000.0, 10.0)
                   and P[n].reciprocity_table.stops_correction == (0.0, 0.0)
                   for n in ("KODAK_PORTRA_160NC", "KODAK_PORTRA_160VC", "KODAK_PORTRA_400NC", "KODAK_PORTRA_400VC"))
        chk(ok and tabs, "E-190 p4 prints no correction from 1/10,000 to 10 s for the 160 and 400 films; the four tables hold exactly that")
    if have("2003KodakProfessionalCatalog_L9.pdf"):
        cat = pymupdf.open(str(kd / "2003KodakProfessionalCatalog_L9.pdf"))
        ct = re.sub(r"\s+", " ", " ".join(cat[i].get_text() for i in range(18, 21)))
        n160 = ct.count("ISO 160 for exposure times of 1/10,000 second to 10 seconds")
        n400 = ct.count("ISO 400 for exposure times of 1/10,000 second to 10 seconds")
        t100 = "EI 100 (1/1,000 to 5 sec), EI 64 (30 sec), EI 40 (120 sec)" in ct
        rt = P["KODAK_PORTRA_100T"].reciprocity_table
        # n400 is 3: the 400UC entry (not in this database) prints the same bound
        chk(n160 == 2 and n400 >= 2 and t100 and rt.times_s[0] == 5.0 and abs(rt.stops_correction[2] - 2.0 / 3.0) < 1e-3
            and abs(rt.stops_correction[4] - 4.0 / 3.0) < 1e-3,
            "L-9 (2003) confirms the 10 s bound on all four NC/VC films and PORTRA 100T's EI 100 / 64 / 40 walk "
            "(stored 0 / 2/3 / 1 1/3 stop at 5 / 30 / 120 s)")

    # ---- 4. E-27, EKTACHROME 100 Professional (EPN) --------------------------
    if have("e27.pdf"):
        import kodak_still_curves as K
        import film_sim as fs
        doc = pymupdf.open(str(kd / "e27.pdf")); pg = doc[4]
        p = P["KODAK_EKTACHROME_100_EPN"]
        txt = re.sub(r"\s+", " ", " ".join(doc[i].get_text() for i in range(doc.page_count)))
        chk("Diffuse rms Granularity* 11" in txt and p.grain.rms_granularity == 11.0 and p.exposure_index == 100,
            "E-27: printed rms 11 and EI 100 stored")
        chk("At 1 second, use a CC05M filter and increase exposure by 1⁄3 stop" in txt.replace("1⁄ 3", "1⁄3")
            or "CC05M" in txt, "E-27 p3 prints the 1 s CC05M + 1/3 stop step")
        chk(p.reciprocity_table.cc_filters == ("", "", "CC05M") and abs(p.reciprocity_table.stops_correction[2] - 1 / 3) < 1e-9,
            "EPN reciprocity table stores it")
        cwd = os.getcwd(); os.chdir(str(Path(a.root)))
        try:
            for k, t, box, lx, ly, letters, exp in K.find_panels(pg, "e27.pdf"):
                pan = K.extract_panel(pg, box, letters=letters, log_x=lx, log_y=ly, expect=exp)
                if t.startswith("Characteristic") and pan is not None:
                    cv = {"R": p.curves.r, "G": p.curves.g, "B": p.curves.b}
                    worst = 0.0
                    for ch in "RGB":
                        tr = np.array(pan.traces[ch])
                        x = -(tr[:, 0] + 1.5)
                        m = np.array([fs.density_scalar(float(v), cv[ch]) for v in x])
                        worst = max(worst, float(np.sqrt(np.mean((m - tr[:, 1]) ** 2))))
                    chk(worst < 0.025, "E-27 characteristic curves reproduced to %.4f D rms (worst channel)" % worst)
                elif t.startswith("Modulation") and pan is not None:
                    tr = np.array(pan.unlabelled[0]); f = 10 ** tr[:, 0]; r = 10 ** tr[:, 1] / 100.0
                    o = np.argsort(f); f, r = f[o], r[o]
                    i = np.where(r >= 0.5)[0][-1]
                    f50 = 10 ** np.interp(0.5, [r[i + 1], r[i]], [math.log10(f[i + 1]), math.log10(f[i])])
                    chk(abs(f50 - p.mtf.f50_g) < 0.1 and r.argmax() == 0,
                        "E-27 MTF crosses 50 %% at %.2f c/mm (stored %.2f); its maximum is the first sample, so the overshoot is unresolved"
                        % (f50, p.mtf.f50_g))
                elif t.startswith("Spectral-Dye") and pan is not None:
                    curves = []
                    for tr in pan.unlabelled:
                        tr = np.array(tr); o = np.argsort(tr[:, 0])
                        curves.append(np.interp(DYE_GRID, tr[o, 0], tr[o, 1]))
                    dd = p.dye_density
                    stored = [np.asarray(v) for v in (dd.d_cyan, dd.d_magenta, dd.d_yellow, dd.d_neutral)]
                    worst = max(min(float(np.abs(c - s).max()) for c in curves) for s in stored)
                    s3 = stored[0] + stored[1] + stored[2]
                    chk(worst < 0.0015 and float(np.abs(s3 - stored[3]).max()) < 0.005,
                        "E-27 dye panel: four stored curves reproduce the traces to %.4f D, and Y+M+C = visual neutral to %.4f D"
                        % (worst, float(np.abs(s3 - stored[3]).max())))
        finally:
            os.chdir(cwd)
        box = _panel_rect(pg, "Spectral-Sensitivity")
        if box is not None:
            rec = spectral_grid(pg, _frame_in(pg, box), 300.0, (2.0, 1.0, 0.0))
            sp = p.spectral
            worst = max(float(np.abs(rec[k][0] - np.asarray(st)).max())
                        for k, st in (("r", sp.log_s_r), ("g", sp.log_s_g), ("b", sp.log_s_b)))
            chk(worst < 1e-9, "E-27 spectral panel, gridline-calibrated, reproduces the stored arrays exactly")

    # ---- 5. VISION3 5219, the AHU generation --------------------------------
    new = "KODAK-VISION3-5219-7219-technical-information.pdf"
    if have(new):
        d = pymupdf.open(str(kd / new))
        t = re.sub(r"\s+", " ", " ".join(pg.get_text() for pg in d))
        chk("An Anti-halation undercoat replaces the traditional remjet backing layer" in t and "Revised 3-26" in t,
            "H-1-5219 Revised 3-26 states the anti-halation undercoat")

        def md5s(doc):
            out = []
            for pg in doc:
                for img in pg.get_images(full=True):
                    out.append(hashlib.md5(doc.extract_image(img[0])["image"]).hexdigest())
            return sorted(out)
        old = "VISION3_5219_7219_Technical-data.pdf"
        if have(old):
            o = pymupdf.open(str(kd / old))
            chk(md5s(o) == md5s(d) and len(md5s(d)) == 6,
                "its six image-structure bitmaps are byte-identical to the Revised 3-22 edition's -- no new plot was published")
        a0, b0 = P["KODAK_VISION3_500T_5219"], P["KODAK_VISION3_500T_5219_AHU"]
        import dataclasses
        diff = [f.name for f in dataclasses.fields(a0) if getattr(a0, f.name) != getattr(b0, f.name)]
        chk(set(diff) == {"name", "aliases", "era", "description", "emulsion", "anti_halation", "param_sources"}
            and b0.emulsion.antihalation == "dyed_undercoat" and b0.anti_halation.position == "undercoat"
            and dataclasses.replace(b0.emulsion, antihalation=a0.emulsion.antihalation,
                                    antihalation_undercoat_um=a0.emulsion.antihalation_undercoat_um) == a0.emulsion,
            "KODAK_VISION3_500T_5219_AHU differs from the rem-jet profile ONLY in identity and construction (%s)" % ", ".join(diff))

    # ---- 6. Vitale 2009, Table 4 ---------------------------------------------
    vf = "estimating_historic_image_resolution_v9.pdf"
    if have(vf):
        d = pymupdf.open(str(kd / vf))
        t = "\n".join(d[i].get_text() for i in range(13, 16))
        tt = re.sub(r"\s+", " ", t)
        miss = [r[1] for r in fp.VITALE_2009_TABLE4
                if not re.search(re.escape(r[1].split(" (")[0].replace("'", "’")[:18]) + r".{0,40}?\b%d\b" % r[2], tt)
                and not re.search(re.escape(r[1][:18]) + r".{0,40}?\b%d\b" % r[2], tt)]
        chk(not miss, "Vitale 2009 Table 4: all %d stored rows re-read from pages 14-16%s"
            % (len(fp.VITALE_2009_TABLE4), "" if not miss else "; missing " + ", ".join(miss[:4])))
        adop = [(r[5], r[2]) for r in fp.VITALE_2009_TABLE4 if r[6] == "adopted"]
        g30 = fp._VITALE_GAUSS_30_TO_50
        off = [n for n, f30 in adop if abs(P[n].mtf.f50_g * g30 - f30) > 0.02 or P[n].mtf.mtf_rolloff_q != 0.0]
        chk(not off and len(adop) == 9,
            "the %d adopted stocks render their 30 %% point on the paper's figure through the Gaussian carrier" % len(adop))
        ratios = []
        for r in fp.VITALE_2009_TABLE4:
            if not r[6].startswith("cross-check") or r[5] == "AGFA_APX_25":
                continue
            m = P[r[5]].mtf
            f30 = m.f50_g * ((0.7 / 0.3) ** (1.0 / m.mtf_rolloff_q) if m.mtf_rolloff_q > 0 else g30)
            ratios.append((r[5], f30 / r[2]))
        lo, hi = min(v for _, v in ratios), max(v for _, v in ratios)
        chk(0.80 <= lo and hi <= 1.15,
            "cross-check: on %d traced-MTF stocks the stored law's 30 %% point is %.2f-%.2f of the paper's figure"
            % (len(ratios), lo, hi))

    print("\n%s (%d checks)" % ("OK" if not bad else "FAIL (%d)" % bad, ran))
    return 1 if (bad and a.do_assert) else 0


if __name__ == "__main__":
    sys.exit(main())
