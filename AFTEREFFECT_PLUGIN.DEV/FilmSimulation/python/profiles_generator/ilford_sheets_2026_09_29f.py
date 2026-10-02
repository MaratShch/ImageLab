"""Audit of the 2026-09-29f ILFORD harvest (HP5 PLUS, DELTA 3200).

Re-derives from the sheets every number written by the batch and fails on
drift:

  ILFORD/HP5-Plus_201811.pdf, ILFORD/HP5+-200407.pdf
  ILFORD/Delta-3200_201811.pdf, ILFORD/DELTA 3200 technical data sheet F25.pdf,
  ILFORD/Delta_3200-200209.pdf

  * the development tables, parsed from the text layers, against the stored
    ILFORD_SHEET_POINTS rows (and the 2025 DELTA 3200 Ilford rows against
    2018's);
  * the vector characteristic curves (2004 HP5 p5, 2002 DELTA 3200 p6),
    re-traced and compared with the stored fits;
  * the DELTA 3200 contrast-time graphs (2002 p5) against the stored laws and
    G-bar points;
  * the reciprocity sentences against the stored Schwarzschild exponents.

Usage:  python ilford_sheets_2026_09_29f.py --root <project root> [--assert]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import film_profiles as fp  # noqa: E402

RESULTS: list[bool] = []


def chk(ok, label, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s   %s" % ("PASS" if ok else "FAIL", label, detail))


def _v(t):
    t = t.strip().replace("½", ".5").replace("¼", ".25").replace("¾", ".75")
    t = re.sub(r"^(\d+)1/2$", r"\1.5", t)
    t = re.sub(r"^(\d+)1/4$", r"\1.25", t)
    t = re.sub(r"^(\d+)3/4$", r"\1.75", t)
    if t in ("−", "–", "-", ""):
        return None
    return float(t)


def _bez(P, n=30):
    s = np.linspace(0, 1, n)[:, None]
    A = np.array([[p.x, p.y] for p in P])
    return (1 - s) ** 3 * A[0] + 3 * (1 - s) ** 2 * s * A[1] + 3 * (1 - s) * s * s * A[2] + s ** 3 * A[3]


def _strokes(page, rect, minw=1.3):
    out = []
    for g in page.get_drawings():
        if not (g.get("width") and g["width"] >= minw) or g.get("type") == "f":
            continue
        if not rect.intersects(g["rect"]):
            continue
        cur, last = [], None
        for it in g["items"]:
            if it[0] == "c":
                seg = _bez(it[1:5])
            elif it[0] == "l":
                seg = np.array([[it[1].x, it[1].y], [it[2].x, it[2].y]])
            else:
                continue
            if last is not None and np.abs(seg[0] - last).sum() > 0.3:
                out.append(np.vstack(cur))
                cur = []
            cur.append(seg)
            last = seg[-1]
        if cur:
            out.append(np.vstack(cur))
    return out


def _grid(page, y0, x0=None):
    for g in page.get_drawings():
        r = g["rect"]
        if g["items"] and g["items"][0][0] == "re" and abs(r.y0 - y0) < 0.3 and (x0 is None or abs(r.x0 - x0) < 0.3):
            vx = sorted({round(it[1].x, 2) for it in g["items"] if it[0] == "l" and abs(it[1].x - it[2].x) < 0.05})
            hy = sorted({round(it[1].y, 2) for it in g["items"] if it[0] == "l" and abs(it[1].y - it[2].y) < 0.05})
            return r, vx, hy
    raise LookupError(y0)


def _md(x, q):
    sp = lambda z, k: k * np.logaddexp(0, z / k)
    return q[0] + q[1] * (sp(x - q[2], q[3]) - sp(x - q[4], q[5]))


def _rows(text, names, ncol):
    """Rows of (developer, dilution, cells) from an Ilford table's text flow."""
    toks = [t.strip() for t in text.split("|") if t.strip()]
    return toks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--assert", dest="strict", action="store_true")
    a = ap.parse_args(argv)
    import pymupdf
    base = Path(a.root) / "PDF" / "PROFILES" / "ILFORD"
    hp18 = pymupdf.open(base / "HP5-Plus_201811.pdf")
    hp04 = pymupdf.open(base / "HP5+-200407.pdf")
    d18 = pymupdf.open(base / "Delta-3200_201811.pdf")
    d25 = pymupdf.open(base / "DELTA 3200 technical data sheet F25.pdf")
    d02 = pymupdf.open(base / "Delta_3200-200209.pdf")

    # -- reciprocity --------------------------------------------------------
    t_hp = hp18[1].get_text()
    t_d = d18[1].get_text()
    hp = fp.get_profile("ILFORD_HP5_PLUS_400").reciprocity
    dd = fp.get_profile("ILFORD_DELTA_3200").reciprocity
    chk("Ta = Tm1.31" in t_hp and "Ta = Tm1.33" in t_d
        and abs(hp.schwarzschild_p_g - 1 / 1.31) < 5e-4 and abs(dd.schwarzschild_p_g - 1 / 1.33) < 5e-4
        and hp.onset_s == 0.5 and dd.onset_s == 0.5,
        "reciprocity: Ilford's Ta = Tm^p (1.31 / 1.33) is the stored exponent at a 1/2 s onset")

    # -- development tables, from the text layer ------------------------------
    def cells_after(text, dev, dil, n):
        flat = text.replace("\n", "|")
        toks = [t.strip() for t in flat.split("|") if t.strip() != ""]
        for i in range(len(toks) - n):
            if toks[i] == dil and dev.lower() in " ".join(toks[max(0, i - 3):i + 1]).lower():
                vals = []
                for t in toks[i + 1:i + 1 + n]:
                    try:
                        vals.append(_v(t))
                    except ValueError:
                        vals = None
                        break
                if vals is not None:
                    return tuple(vals)
        return None
    stored = {}
    for q in fp.ILFORD_SHEET_POINTS["ILFORD_DELTA_3200"]:
        if not q.contrast_index:
            stored[(q.developer, q.dilution, q.celsius, q.exposure_index)] = q.minutes
    # DELTA 3200 2018 p3 (20 C) and p4 (24 C): spot the recommended developers
    ok = True
    probes = [("ILFOTEC DD-X", "1+4", 20.0, 3), ("ID-11", "stock", 20.0, 3), ("MICROPHEN", "stock", 20.0, 3),
              ("PERCEPTOL", "stock", 20.0, 3), ("Kodak Xtol", "stock", 20.0, 3), ("Kodak T-Max", "1+4", 20.0, 3)]
    got = []
    name_map = {"ILFOTEC DD-X": "Ilfotec DD-X", "ID-11": "ID-11", "MICROPHEN": "Microphen",
                "PERCEPTOL": "Perceptol", "Kodak Xtol": "XTOL", "Kodak T-Max": "T-MAX"}
    for dev, dil, t, pg in probes:
        v = cells_after(d18[2].get_text(), dev, dil, 6)
        want = tuple(stored.get((name_map[dev], dil, t, ei)) for ei in fp._D32_EI)
        got.append((dev, v == want))
        ok &= (v == want)
    chk(ok, "DELTA 3200 20 C table (2018 p3): six developers re-parsed equal the stored rows", str(got))
    ok25 = True
    for dev, dil in (("ILFOTEC DD-X *", "1+4"), ("ID-11*", "stock"), ("MICROPHEN *", "stock"), ("PERCEPTOL", "stock")):
        a18 = cells_after(d18[2].get_text(), dev.split()[0].rstrip("*"), dil, 6)
        a25 = cells_after(d25[2].get_text(), dev.split()[0].rstrip("*"), dil, 6)
        ok25 &= (a18 == a25 and a18 is not None)
    chk(ok25, "DELTA 3200: the 2025 sheet prints the 2018 Ilford rows unchanged")
    hst = {}
    for q in fp.ILFORD_SHEET_POINTS["ILFORD_HP5_PLUS_400"]:
        hst[(q.developer, q.dilution, q.exposure_index)] = q.minutes
    okh = True
    for dev, dil, sd in (("ILFOTEC DD-X", "1+4", "Ilfotec DD-X"), ("MICROPHEN", "stock", "Microphen"),
                         ("Kodak Xtol", "stock", "XTOL"), ("Kodak T-Max", "1+4", "T-MAX")):
        v = cells_after(hp18[2].get_text(), dev, dil, 7)
        want = tuple(hst.get((sd, dil, ei)) for ei in fp._HP5_EI)
        okh &= (v == want)
    chk(okh, "HP5 Plus 20 C table (2018 p3): four developers re-parsed equal the stored rows")
    # -- vector curves -------------------------------------------------------
    p = hp04[4]
    r, vx, hy = _grid(p, 313.6, 326.5)
    c = max(_strokes(p, r), key=len)
    X = (c[:, 0] - 326.565) / 41.102
    Y = (435.42 - c[:, 1]) / 40.2
    q = fp.ILFORD_SHEET_CURVE_FITS[("ILFORD_HP5_PLUS_400", "Ilfotec HC", "1+31", 6.5)][0]
    e_free = np.sqrt(np.mean((_md(X, q) - Y) ** 2))
    g = fp.get_profile("ILFORD_HP5_PLUS_400").curves.g
    gq = (g.dmin, g.gamma, g.toe_x, g.toe_k, g.shoulder_x, g.shoulder_k)
    e = np.sqrt(np.mean((_md(X + fp.ILFORD_CURVE_X_SHIFT["ILFORD_HP5_PLUS_400"], gq) - Y) ** 2))
    chk(e < 0.006 and e_free < 0.006 and fp.get_profile("ILFORD_HP5_PLUS_400").processing.developer == "Ilfotec HC",
        "HP5 Plus: the 2004 vector curve re-traced sits on the stored ToneCurve and on its free fit",
        "rms %.4f D stored / %.4f D free" % (e, e_free))
    p6 = d02[5]
    worst = 0.0
    for y0, dev, dil in ((99.61, "Ilfotec DD-X", "1+4"), (309.5, "Microphen", "stock")):
        r, vx, hy = _grid(p6, y0)
        kx, bx = np.polyfit([0.5 * (i + 1) for i in range(len(vx))], vx, 1)
        ky, by = np.polyfit([0.5 * (len(hy) - i) for i in range(len(hy))], hy, 1)
        st = sorted(_strokes(p6, r), key=lambda s: -s[:, 1].min())
        for s, t in zip(st, (7.0, 9.0, 12.0, 16.0)):
            X = (s[:, 0] - bx) / kx
            Y = (s[:, 1] - by) / ky
            q = fp.ILFORD_SHEET_CURVE_FITS[("ILFORD_DELTA_3200", dev, dil, t)][0]
            worst = max(worst, float(np.sqrt(np.mean((_md(X, q) - Y) ** 2))))
            if (dev, t) == ("Microphen", 9.0):
                g = fp.get_profile("ILFORD_DELTA_3200").curves.g
                gq = (g.dmin, g.gamma, g.toe_x, g.toe_k, g.shoulder_x, g.shoulder_k)
                e_st = float(np.sqrt(np.mean((_md(X + fp.ILFORD_CURVE_X_SHIFT["ILFORD_DELTA_3200"], gq) - Y) ** 2)))
    chk(worst < 0.03, "DELTA 3200: all eight 2002 vector curves re-traced sit on their stored fits", "worst rms %.4f D" % worst)
    chk(e_st < 0.025 and fp.get_profile("ILFORD_DELTA_3200").processing.developer == "Microphen",
        "DELTA 3200: the stored (monotone) ToneCurve sits on the re-traced Microphen 9 min curve", "rms %.4f D" % e_st)
    # -- contrast-time -------------------------------------------------------
    p5 = d02[4]
    fam = fp.get_profile("ILFORD_DELTA_3200").processing_family
    wl = 0.0
    for y0, dev, ticks in ((341.61, "Ilfotec DD-X", [6, 9, 12, 15, 18, 21, 24, 27]),
                           (532.61, "Microphen", [2, 4, 6, 8, 10, 12, 14, 16])):
        r, vx, hy = _grid(p5, y0)
        kx, bx = np.polyfit(ticks, vx, 1)
        ky, by = np.polyfit([1.0, 0.8, 0.6, 0.4, 0.2], hy, 1)
        s = np.vstack(_strokes(p5, r))
        T = (s[:, 0] - bx) / kx
        G = (s[:, 1] - by) / ky
        law = fam.law_for(dev, vessel="small tank")
        wl = max(wl, float(np.sqrt(np.mean((np.array([law.gamma_at(t) for t in T]) - G) ** 2))))
    chk(wl < 0.02, "DELTA 3200: the DD-X and Microphen laws reproduce the 2002 contrast-time graphs", "worst rms %.4f G" % wl)
    fits = fp.ILFORD_SHEET_CURVE_FITS
    dg = max(abs(fits[("ILFORD_DELTA_3200", d, dl, m)][3] - g)
             for d, dl, pts in fp._D32_GBAR for m, g in pts)
    chk(dg < 0.04, "DELTA 3200: each curve's Ilford G-bar agrees with the contrast-time graph", "worst %.3f G" % dg)
    n_bad = RESULTS.count(False)
    print("[%s] ilford_sheets_2026_09_29f.py -- %d checks, %d failed"
          % ("OK" if not n_bad else "FAIL", len(RESULTS), n_bad))
    return 1 if (a.strict and n_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
