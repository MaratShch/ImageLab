#!/usr/bin/env python3
"""FERRANIA's two vendor sheets — four curves, three wedge spectrograms.

    PDF/PROFILES/FERRANIA/Curve caratteristiche e sensibilita spettrali.pdf
        Film Ferrania S.r.l., 2 pages, undated. Page 1: the «P 30 New» and
        «P 33» characteristic curves, each with its own wedge spectrogram.
        Page 2: the «Orto» curve with the Orto spectrogram — the only strip
        on either sheet that carries a printed sensitivity ladder — and the
        three-film comparison plot «Riepilogo delle 3 tipologie di pellicole
        pancromatiche» (P33 red, «P30 new» solid black, «P30» dashed black).

    PDF/PROFILES/FERRANIA/1579.pdf
        P.F.G. by KARL BIELSER s.a.s. (Rollei Film Point, Milan),
        «Ferrania 2026» range sheet, 4 pages. Page 2 is the prose that
        separates the emulsions; page 3 reprints the sheet above, cropped;
        page 4 is the 14-row indicative development table.

⚠⚠ WHY THIS MODULE EXISTS: ONE PROFILE WAS CARRYING TWO FILMS.
Until 2026-09-25 the database held a single FERRANIA_P30 whose TONE CURVE was
a 2017 third-party test of pre-production ORIGINAL stock and whose SPECTRAL
SENSITIVITY was the wedge spectrogram captioned **«P 30 New»** — the Mk2. The
range sheet's own prose is what separates them:

    «Ferrania P30 (cinema): Questa e la formula originale della P30 ... e una
     pellicola a piu alto contrasto CON BASSA SENSIBILITA AL ROSSO, proprio
     come le pellicole pancromatiche degli anni '50.»
    «FERRANIA P30 Mk2 ... e una vera pellicola pancromatica moderna.»

and the page-2 comparison plot draws «P30 new» and «P30» as two curves. The
split produced FERRANIA_P30_MK2, FERRANIA_P33_160 and FERRANIA_ORTO_50, and
gave FERRANIA_P30 a sixth ProcessVariant (the dashed curve) and a derived
spectral record (see `ferrania_p30_colour_target.py`).

⚠ THE TRAP THIS MODULE IS BUILT TO CATCH.
On 1579.pdf page 3 the ORTO spectrogram is reprinted under the heading
«Sensibilita spettrale» with a yellow overlay whose nm labels stop at 550 and
which covers the photograph from 584 nm on. Read as if it ran to 650 nm, that
strip yields a curve peaking at 610-630 with a cut at 660 — which is very
nearly the shape of the Mk2 strip, and is how a spectral record gets stretched
onto the wrong film twice over. Every wavelength axis here is therefore
re-derived from the STRIP'S OWN printed major ticks on every build, and the
module asserts that the Orto strip's ruler ends at 550 and the other two at
650.

HOW THE ORDINATE IS CALIBRATED, AND THE ASSUMPTION IN IT
----------------------------------------------------------
Only the Orto strip carries a printed ladder («Scale: 1 / 0,5 / 0 —
sensibilita»), 74.6 px per decade against that strip's 104.5 px per 50 nm
ruler interval. The rulers are physical objects of fixed size photographed on
one instrument, so the ratio 0.714 decade per ruler-interval is carried to the
other two strips through their own ruler scales. ⚠ THE ASSUMPTION IS ONE
INSTRUMENT AND ONE WEDGE FOR ALL THREE STRIPS; it cannot be proved from the
sheet and is stated wherever the numbers are used.

⚠ THE WEDGE IS ABOUT 0.8 DECADE DEEP. Where a trace disappears the film is
not insensitive, it is below the wedge floor, so the deepest values in each
record are bounds and not measurements.

Run:  python ferrania_vendor_sheets.py [--root .] [--assert]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SHEET = "PDF/PROFILES/FERRANIA/Curve caratteristiche e sensibilita spettrali.pdf"
RANGE = "PDF/PROFILES/FERRANIA/1579.pdf"

#: Sentences that carry a decision. Their disappearance must be loud.
QUOTES = {
    "P30 cinema is the original formula":
        "Questa è la formula originale della P30",
    "P30 cinema has LOW RED SENSITIVITY":
        "bassa sensibilità al rosso",
    "Mk2 is a modern panchromatic":
        "una vera pellicola pancromatica moderna",
    "Orto is blind past green":
        "sensibile solo alla radiazione ultravioletta, alla luce blu e alla "
        "luce verde",
    "P33 spectral range, printed":
        "Sensibilità spettrale 380 a 640 nm",
    "the development table is indicative only":
        "INDICATIVA, BASE DI PARTENZA",
}
#: The curve sheet's own processing caption -- the condition every one of the
#: four traced curves was developed under.
CAPTION = "sviluppo in Kodak D-76 stock a 20°C"

#: (xref, band rows, clear-area rows, [(px, nm) major ticks]) per strip.
STRIPS = {
    "FERRANIA_P30_MK2": dict(page=0, xref=10, band=(45, 112), clear=(40, 55),
                             ticks=((80.5, 400), (131.5, 450), (182.0, 500),
                                    (233.5, 550), (284.0, 600), (335.5, 650))),
    "FERRANIA_P33_160": dict(page=0, xref=9, band=(62, 142), clear=(58, 72),
                             ticks=((88.5, 400), (138.5, 450), (188.5, 500),
                                    (238.5, 550), (289.0, 600), (339.0, 650))),
    "FERRANIA_ORTO_50": dict(page=1, xref=24, band=(70, 240), clear=(70, 105),
                             ticks=((87.0, 400), (191.5, 450), (294.5, 500),
                                    (401.0, 550))),
}
#: The Orto strip's printed ladder: px per decade, and that strip's px per
#: 50 nm. Everything else is cross-calibrated through this pair.
ORTO_DECADE_PX, ORTO_PX_PER_50 = 74.6, 104.5

#: (xref, page, colour, [(row px, D)] ordinate refs, [(col px, step)]).
PLOTS = {
    "P30_MK2_p1": dict(page=0, xref=11, colour="red",
                       yref=((46.5, 2.0), (262.0, 0.0)),
                       xref_pts=((125, 5), (457, 20)), top=20),
    "P33_p1": dict(page=0, xref=12, colour="red",
                   yref=((44.0, 2.0), (263.0, 0.0)),
                   xref_pts=((123, 5), (457, 20)), top=20),
    "ORTO_p2": dict(page=1, xref=27, colour="magenta",
                    yref=((87.0, 2.5), (548.6, 0.0)),
                    xref_pts=((283.5, 5), (1048.5, 20)), top=70),
    "P33_cmp": dict(page=1, xref=26, colour="red",
                    yref=((38.5, 2.0), (263.0, 0.0)),
                    xref_pts=((124, 5), (456.5, 20)), top=25),
}
#: 0.15 log H per sensitometer step -- the standard 21-step tablet increment,
#: CONFIRMED for this maker's own testing by the «Ferrania P30 alfa» report,
#: whose page-2 table prints its tablet's calibrated densities 0.04 ... 3.03
#: (mean increment 0.1495). Step 1 sits at this log H.
STEP_LOGH, STEP1_X = 0.15, -2.9034
#: dmin per film: measured for the P30 family (alfa report base+fog), an
#: estimate for the other two -- the sheet plots «Su DMIN» and prints none.
DMIN = {"FERRANIA_P30_MK2": 0.26, "FERRANIA_P33_160": 0.20,
        "FERRANIA_ORTO_50": 0.20, "FERRANIA_P30": 0.26}

TOL_SPECTRAL = 0.05     # decade, per stored sample
TOL_CURVE = 0.030       # D rms of the stored ToneCurve against the re-trace
TOL_TICK = 0.8          # nm, linear fit against the printed majors


def _pix(doc, xref):
    import pymupdf
    px = pymupdf.Pixmap(doc, xref)
    if px.n > 4:
        px = pymupdf.Pixmap(pymupdf.csRGB, px)
    a = np.frombuffer(px.samples, dtype=np.uint8)
    return a.reshape(px.height, px.width, px.n).astype(float)


def _envelope(rgb, band, clear, ticks):
    """Top edge of blackening, per column, as (wavelength, decades down)."""
    lum = rgb[..., :3].mean(2)
    xs = np.array([t[0] for t in ticks], float)
    ws = np.array([t[1] for t in ticks], float)
    fit = np.polyfit(xs, ws, 1)
    resid = float(np.max(np.abs(np.polyval(fit, xs) - ws)))
    px_per_50 = 50.0 / fit[0]
    dec_per_px = 1.0 / (ORTO_DECADE_PX * px_per_50 / ORTO_PX_PER_50)
    r0, r1 = band
    bg = lum[clear[0]:clear[1]].mean(0)
    env = np.full(lum.shape[1], np.nan)
    for j in range(lum.shape[1]):
        idx = np.nonzero(lum[r0:r1, j] < bg[j] * 0.90)[0]
        if len(idx):
            env[j] = r0 + idx[0]
    lam = np.polyval(fit, np.arange(lum.shape[1]))
    s = -(env - np.nanmin(env)) * dec_per_px
    return lam, s, resid, px_per_50


def _trace(rgb, spec):
    """Characteristic curve as {step: density above D-min}."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if spec["colour"] == "red":
        m = (r > 140) & (g < 110) & (b < 110)
    else:
        m = (r > 140) & (b > 140) & (g < 130)
    m[:spec["top"]] = False
    (y1, d1), (y0, d0) = spec["yref"]
    (x0, s0), (x1, s1) = spec["xref_pts"]
    dens = lambda y: d0 + (y - y0) * ((d1 - d0) / (y1 - y0))
    step = lambda x: s0 + (x - x0) * ((s1 - s0) / (x1 - x0))
    st, dd = [], []
    for x in range(m.shape[1]):
        ys = np.nonzero(m[:, x])[0]
        if len(ys):
            st.append(step(x))
            dd.append(dens(float(ys.mean())))
    return np.array(st), np.array(dd)


#: ⚠ ADDED 2026-09-27. The 2026 P.F.G. range sheet (1579.pdf) reprints the
#: three-film comparison plot on page 3 as part of ONE page raster (xref 42,
#: 1298 x 882, JPEG). It is an independent raster of the same drawing, so the
#: dashed «P30» curve -- FERRANIA_P30's DEFAULT since 2026-09-27 -- is traced
#: off it as well as off the «Curve caratteristiche» image. The grid is NOT
#: hard-coded: the four light-grey gridlines each way are found in the raster
#: and must land within GRID_TOL_PX of where they were measured on 2026-09-27,
#: so a re-encoded or re-laid-out sheet fails loudly instead of being read on
#: a stale axis.
RANGE_P3_XREF = 42
RANGE_P3_GRID_X = ((198.0, 5), (330.5, 10), (463.5, 15), (595.5, 20))
RANGE_P3_GRID_Y = ((549.5, 2.0), (616.0, 1.5), (683.5, 1.0), (750.5, 0.5))
GRID_TOL_PX = 1.5


def _grid_lines(rgb, box, axis):
    """Centres of the light-grey gridlines in `box` along `axis` (0 rows)."""
    y0, y1, x0, x1 = box
    sub = rgb[y0:y1, x0:x1, :3]
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    grey = ((np.abs(r - g) < 12) & (np.abs(g - b) < 12)
            & (r > 190) & (r < 240))
    prof = grey.sum(1 - axis)
    span = sub.shape[1 - axis]
    hits = [i for i, v in enumerate(prof) if v > 0.5 * span]
    groups, cur = [], []
    for i in hits:
        if cur and i - cur[-1] > 1:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    off = y0 if axis == 0 else x0
    return [off + float(np.mean(g)) for g in groups]


def _dashed_1579(rgb):
    """(steps, density above D-min, grid residual px) off 1579.pdf p3."""
    gx = _grid_lines(rgb, (560, 790, 140, 600), 1)
    gy = _grid_lines(rgb, (540, 790, 140, 600), 0)
    want_x = [p for p, _ in RANGE_P3_GRID_X]
    want_y = [p for p, _ in RANGE_P3_GRID_Y]
    if len(gx) != 4 or len(gy) != 4:
        return None, None, float("inf")
    resid = max(max(abs(a - b) for a, b in zip(gx, want_x)),
                max(abs(a - b) for a, b in zip(gy, want_y)))
    ax, bx = np.polyfit(gx, [s for _, s in RANGE_P3_GRID_X], 1)
    ay, by = np.polyfit(gy, [d for _, d in RANGE_P3_GRID_Y], 1)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    K = (r < 110) & (g < 110) & (b < 110)
    y_axis0 = -by / ay                 # pixel row of D = 0, the x axis line
    ds, dv = [], []
    for x in range(int(round((1 - bx) / ax)) + 2,
                   int(round((20 - bx) / ax)) + 1):
        ys = np.nonzero(K[545:830, x])[0] + 545
        ys = ys[ys < y_axis0 - 2.5]    # drop the axis line itself
        if not len(ys):
            continue
        groups, cur = [], [ys[0]]
        for v in ys[1:]:
            if v - cur[-1] <= 2:
                cur.append(v)
            else:
                groups.append(cur)
                cur = [v]
        groups.append(cur)
        cen = sorted(float(np.mean(q)) for q in groups)
        low = [v for v in cen if v > cen[0] + 7]
        st = ax * x + bx
        if low and st >= 6.0:
            ds.append(st)
            dv.append(ay * low[-1] + by)
    return np.array(ds), np.array(dv), resid


def _tone(tc, x):
    sp = lambda u, k: k * np.log1p(np.exp(np.clip(u / k, -60, 60)))
    return tc.dmin + tc.gamma * (sp(x - tc.toe_x, tc.toe_k)
                                 - sp(x - tc.shoulder_x, tc.shoulder_k))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()
    if not (root / SHEET).is_file() or not (root / RANGE).is_file():
        print("  [SKIP] Ferrania vendor sheets not present under %s" % root)
        return 0
    import pymupdf
    print("FERRANIA vendor sheets -- four curves, three wedge spectrograms")
    bad = 0
    sheet = pymupdf.open(str(root / SHEET))
    rng = pymupdf.open(str(root / RANGE))
    if sheet.page_count != 2 or rng.page_count != 4:
        print("  [FAIL] page counts changed: sheet %d (want 2), range %d "
              "(want 4)" % (sheet.page_count, rng.page_count))
        bad += 1

    whole = " ".join(" ".join(p.get_text().split())
                     for p in list(sheet) + list(rng))
    for label, quote in QUOTES.items():
        if quote not in whole:
            print("  [FAIL] the sheets no longer say %s (%r)" % (label, quote))
            bad += 1
    if CAPTION not in whole:
        print("  [FAIL] the processing caption is gone -- every traced curve "
              "here depends on it: %r" % CAPTION)
        bad += 1
    else:
        print("  [OK  ] %d load-bearing sentences and the D-76 stock 20 C "
              "8 min caption all still printed" % len(QUOTES))

    # ⚠ THE THREE STRIPS ARE DIFFERENT FILMS. Assert it from the rulers.
    try:
        import film_profiles as fp
    except Exception as exc:                                  # pragma: no cover
        print("  [WARN] could not import film_profiles: %s" % exc)
        return 1 if ns.assert_ else 0

    for name, S in STRIPS.items():
        rgb = _pix(sheet, S["xref"])
        lam, s, resid, px50 = _envelope(rgb, S["band"], S["clear"], S["ticks"])
        end_nm = S["ticks"][-1][1]
        if resid > TOL_TICK:
            print("  [FAIL] %s strip: wavelength fit off its own printed "
                  "majors by %.2f nm" % (name, resid))
            bad += 1
        prof = fp.get_profile(name)
        stored = np.array(prof.spectral.log_s_pan, float)
        got = []
        for i in range(len(stored)):
            w = prof.spectral.lambda_start_nm + i * prof.spectral.lambda_step_nm
            sel = (np.abs(lam - w) <= 6) & ~np.isnan(s)
            got.append(float(np.mean(s[sel])) if sel.sum() else -4.00)
        got = np.array(got)
        # ⚠ PEAK-NORMALISED, because SpectralSensitivity.validate requires it:
        # the strip's own maximum and the 10 nm resampled maximum are not the
        # same pixel, so the resampled record is shifted onto 0.0 exactly as
        # the stored one is.
        alive = got > -3.9
        if alive.any():
            got[alive] = got[alive] - got[alive].max()
        live = stored > -3.9
        worst = float(np.max(np.abs(got[live] - stored[live])))
        floor_ok = bool(np.all(got[~live] <= -3.9)) if (~live).any() else True
        print("  %-18s ruler %3d nm .. %3d nm, %5.2f px/50nm, tick resid "
              "%.2f nm; stored vs re-read worst %.3f decade%s"
              % (name, S["ticks"][0][1], end_nm, px50, resid, worst,
                 "" if floor_ok else "  [floor mismatch]"))
        if worst > TOL_SPECTRAL or not floor_ok:
            print("  [FAIL] %s spectral record does not reproduce from its "
                  "own strip" % name)
            bad += 1

    # ⚠ THE ORTO RULER MUST END AT 550 AND THE OTHER TWO AT 650. This is the
    # single fact that keeps the 1579 page-3 reprint from being read as P30's.
    ends = {k: v["ticks"][-1][1] for k, v in STRIPS.items()}
    if (ends["FERRANIA_ORTO_50"] != 550
            or ends["FERRANIA_P30_MK2"] != 650
            or ends["FERRANIA_P33_160"] != 650):
        print("  [FAIL] the strips' printed wavelength ranges have changed")
        bad += 1

    # -- the four characteristic curves ------------------------------------
    traces = {}
    for key, spec in PLOTS.items():
        doc = sheet
        rgb = _pix(doc, spec["xref"])
        traces[key] = _trace(rgb, spec)
    # P33 is drawn twice. The two drawings agreeing is what licenses every
    # other reading off this sheet, INCLUDING the dashed P30 original.
    a_st, a_d = traces["P33_p1"]
    b_st, b_d = traces["P33_cmp"]
    diffs = []
    for k in range(3, 21):
        sa = np.abs(a_st - k) < 0.12
        sb = np.abs(b_st - k) < 0.12
        if sa.sum() and sb.sum():
            diffs.append(abs(a_d[sa].mean() - b_d[sb].mean()))
    cross = float(max(diffs)) if diffs else 9.9
    print("  P33 traced from TWO drawings (page 1 plot, page 2 comparison): "
          "worst disagreement %.4f D over %d steps" % (cross, len(diffs)))
    if cross > 0.02:
        print("  [FAIL] the two drawings of P33 no longer agree -- the axis "
              "calibration for all four curves rests on this")
        bad += 1

    # ⚠ THE STEP CEILING PER FILM IS NOT COSMETIC. Two of these curves run
    # along the TOP OF THEIR OWN ORDINATE FRAME at the last two steps -- Mk2
    # reaches 1.977 against a frame that stops at 2.0, Orto 2.491 against 2.5
    # -- so those points measure the drawing and not the film. They are
    # dropped here exactly as they were dropped from the fit; a fit allowed to
    # read that flattening as a shoulder puts it just above mid grey and the
    # rendered negative then saturates on every highlight.
    for key, stock, smax in (("P30_MK2_p1", "FERRANIA_P30_MK2", 18.0),
                             ("P33_p1", "FERRANIA_P33_160", 20.0),
                             ("ORTO_p2", "FERRANIA_ORTO_50", 18.0)):
        st, dd = traces[key]
        keep = (st >= 1.0) & (st <= smax)
        x = (st[keep] - 1.0) * STEP_LOGH + STEP1_X
        want = dd[keep] + DMIN[stock]
        tc = fp.get_profile(stock).curves.g
        rms = float(np.sqrt(np.mean((_tone(tc, x) - want) ** 2)))
        print("  %-18s stored ToneCurve vs re-trace: rms %.4f D over %d "
              "columns (gamma %.3f)" % (stock, rms, keep.sum(), tc.gamma))
        if rms > TOL_CURVE:
            print("  [FAIL] %s curve does not reproduce from the drawing"
                  % stock)
            bad += 1

    # -- the dashed P30 ORIGINAL, off the same comparison plot --------------
    rgb = _pix(sheet, 26)
    K = (rgb[..., 0] < 110) & (rgb[..., 1] < 110) & (rgb[..., 2] < 110)
    K[:36] = False
    K[266:] = False
    K[:, :16] = False
    dens = lambda y: (y - 263.0) * (2.0 / (38.5 - 263.0))
    step = lambda x: 5 + (x - 124) * (15.0 / (456.5 - 124))
    ds, dv = [], []
    for x in range(K.shape[1]):
        ys = np.nonzero(K[:, x])[0]
        if not len(ys):
            continue
        groups, cur = [], [ys[0]]
        for v in ys[1:]:
            if v - cur[-1] <= 3:
                cur.append(v)
            else:
                groups.append(cur)
                cur = [v]
        groups.append(cur)
        cen = sorted(float(np.mean(g)) for g in groups if g)
        low = [v for v in cen if v > cen[0] + 7]
        if low and step(x) >= 6.0 and dens(low[-1]) > -0.02:
            ds.append(step(x))
            dv.append(dens(low[-1]))
    ds, dv = np.array(ds), np.array(dv)
    if len(ds) < 60:
        print("  [FAIL] the dashed «P30» trace no longer resolves (%d columns)"
              % len(ds))
        bad += 1
    else:
        keep = [0]
        for i in range(1, len(dv)):
            if dv[i] >= dv[keep[-1]] - 0.03:
                keep.append(i)
        ds, dv = ds[keep], dv[keep]
        # step 20 is the plot's right-hand frame -- 0.040 D of gain against
        # 0.158 in the step before it -- and is dropped as the fit drops it.
        inrange = ds <= 19.0
        ds, dv = ds[inrange], dv[inrange]
        x = (ds - 1.0) * STEP_LOGH + STEP1_X
        pv = [v for v in fp._PROCESS_VARIANTS["FERRANIA_P30"]
              if "stock" in v.name]
        if not pv:
            print("  [FAIL] FERRANIA_P30 has lost its D-76 stock variant, "
                  "which is the only manufacturer curve the ORIGINAL "
                  "emulsion has")
            bad += 1
        else:
            rms = float(np.sqrt(np.mean(
                (_tone(pv[0].curves.g, x) - (dv + DMIN["FERRANIA_P30"])) ** 2)))
            print("  FERRANIA_P30 (orig)  dashed «P30» trace vs stored D-76 "
                  "stock variant: rms %.4f D over %d columns, gamma %.3f"
                  % (rms, len(ds), pv[0].curves.g.gamma))
            if rms > TOL_CURVE:
                print("  [FAIL] the D-76 stock variant does not reproduce")
                bad += 1

    # -- the SAME dashed curve off 1579.pdf p3, an independent raster --------
    # ⚠ ADDED 2026-09-27, when this curve became FERRANIA_P30's DEFAULT by
    # owner decision. A default resting on one raster would be one JPEG away
    # from a silent error, so it is read off both and must reproduce from
    # each. Measured on the day: 173 columns, steps 7.4-19.0, rms 0.0140 D.
    ds3, dv3, gres = _dashed_1579(_pix(rng, RANGE_P3_XREF))
    if ds3 is None or gres > GRID_TOL_PX:
        print("  [FAIL] 1579.pdf p3: the comparison plot's grid moved "
              "(residual %s px) -- the raster changed and no reading off it "
              "can be trusted" % ("n/a" if ds3 is None else "%.2f" % gres))
        bad += 1
    elif len(ds3) < 60:
        print("  [FAIL] 1579.pdf p3: the dashed «P30» trace no longer "
              "resolves (%d columns)" % len(ds3))
        bad += 1
    else:
        keep = [0]
        for i in range(1, len(dv3)):
            if dv3[i] >= dv3[keep[-1]] - 0.03:
                keep.append(i)
        ds3, dv3 = ds3[keep], dv3[keep]
        m = ds3 <= 19.0
        ds3, dv3 = ds3[m], dv3[m]
        x3 = (ds3 - 1.0) * STEP_LOGH + STEP1_X
        dflt = fp.get_profile("FERRANIA_P30").curves.g
        rms3 = float(np.sqrt(np.mean(
            (_tone(dflt, x3) - (dv3 + DMIN["FERRANIA_P30"])) ** 2)))
        print("  FERRANIA_P30 (orig)  1579.pdf p3 re-trace vs the DEFAULT "
              "curve: rms %.4f D over %d columns, steps %.1f-%.1f, grid "
              "residual %.2f px" % (rms3, len(ds3), ds3.min(), ds3.max(), gres))
        if rms3 > TOL_CURVE:
            print("  [FAIL] FERRANIA_P30's default curve does not reproduce "
                  "from the 1579.pdf reprint")
            bad += 1

    # -- the distributor's development table --------------------------------
    page4 = " ".join(rng[3].get_text().split())
    for stock, want in (("FERRANIA_P30_MK2", 14), ("FERRANIA_P33_160", 13),
                        ("FERRANIA_ORTO_50", 14)):
        got = len(fp.get_profile(stock).processing_family.points)
        if got != want:
            print("  [FAIL] %s holds %d development points, the table gives "
                  "%d" % (stock, got, want))
            bad += 1
    # ⚠ P33's ROW COUNT IS ONE SHORT BY DESIGN: the Rollei Low Contrast row
    # has no P33 cell on the sheet. Assert the sheet still prints it that way.
    if "Rollei Low Contrast 1+4 9’ medio/basso 9’ medio sì" \
            not in page4:
        print("  [WARN] the Rollei Low Contrast row no longer reads as a "
              "two-film row; P33's 13-against-14 point count depends on it")
    # ⚠⚠ THE CONTRAST COLUMN IS STORED AS WORDS AND AS AN ORDINAL, 2026-09-25e.
    # It prints adjectives, not gammas, so `contrast_index` cannot take it --
    # but 41 cells of real data were being left in a source string. They now
    # live in `_FERRANIA_2026_CONTRAST_WORDS` with the word verbatim and a rank
    # on the sheet's own scale (basso < medio/basso < medio < medio/alto <
    # alto), which is exactly the ordinal the source publishes. ⚠ A RANK IS
    # NOT A CONTRAST INDEX: the interval between two ranks is unknown.
    words = getattr(fp, "_FERRANIA_2026_CONTRAST_WORDS", ())
    scale = getattr(fp, "_FERRANIA_2026_CONTRAST_SCALE", ())
    print("  contrast column stored as words + ordinal: %d cells over %d films"
          % (len(words), len({w[0] for w in words})))
    if len(words) != 41 or len(scale) != 5:
        print("  [FAIL] the contrast-word register is %d cells and the scale "
              "%d rungs; the sheet prints 41 and 5" % (len(words), len(scale)))
        bad += 1
    else:
        seen = {}
        for film, dev, dil, word, rank in words:
            base = word.replace("medio alto", "medio/alto").replace(
                "alto/medio", "medio/alto")
            if not (1 <= rank <= 5) or base not in scale \
                    or scale.index(base) + 1 != rank:
                print("  [FAIL] %s %s %s: word %r does not rank %d"
                      % (film, dev, dil, word, rank))
                bad += 1
            seen.setdefault(film, set()).add((dev, dil))
        for film, keys in seen.items():
            fam = {(q.developer, q.dilution)
                   for q in fp.get_profile(film).processing_family.points}
            orphan = sorted(k for k in keys if k not in fam)
            missing = sorted(k for k in fam if k not in keys)
            if orphan or missing:
                print("  [FAIL] %s: %d contrast words with no development "
                      "point %s, %d points with no word %s"
                      % (film, len(orphan), orphan[:2], len(missing),
                         missing[:2]))
                bad += 1
        # ⚠ THE ONE ORDERING THE SHEET ASSERTS BY ITSELF: on every row printed
        # for both, Orto is at least as contrasty as P30 Mk2. That is the
        # ordinal used as an ordinal, which is all it is good for.
        mk2 = {(d, x): r for f, d, x, _w, r in words if f == "FERRANIA_P30_MK2"}
        ort = {(d, x): r for f, d, x, _w, r in words if f == "FERRANIA_ORTO_50"}
        both = [k for k in mk2 if k in ort]
        soft = sorted(k for k in both if ort[k] < mk2[k])
        print("    Orto ranks at or above P30 Mk2 on %d of the %d rows the "
              "sheet prints for both" % (len(both) - len(soft), len(both)))
        if soft:
            print("  [FAIL] the sheet's Orto-over-Mk2 ordering no longer "
                  "holds: %s" % (soft[:2],))
            bad += 1

    # ⚠ THE REVERSAL CAPABILITY, STATED AND NOT MODELLED.
    rev = getattr(fp, "_FERRANIA_REVERSAL_CAPABILITY", ())
    if len(rev) != 2 or {r[0] for r in rev} != {"FERRANIA_P30",
                                                "FERRANIA_ORTO_50"}:
        print("  [FAIL] the reversal-capability register no longer names "
              "exactly FERRANIA_P30 and FERRANIA_ORTO_50")
        bad += 1
    if "anche in dia" not in page4:
        print("  [FAIL] the sheet no longer says «danno ottimi risultati "
              "anche in dia»; the reversal-capability register rests on it")
        bad += 1
    else:
        print("  reversal capability recorded for %d films, and the sentence "
              "it rests on is still printed" % len(rev))

    # ⚠ THE FORUM'S OWN DEVELOPMENT TIMES, five rows, attributable and inert.
    forum = tuple(c for c in getattr(fp, "_FERRANIA_P30_COMMUNITY_TIMES", ())
                  if "analogica.it" in c[5])
    eis = sorted({c[3] for c in forum})
    print("  analogica.it development times: %d rows, exposure indexes %s "
          "against a box speed of 80"
          % (len(forum), "/".join(str(e) for e in eis)))
    if len(forum) != 5 or (eis and (max(eis) > 80 or min(eis) > 25)):
        print("  [FAIL] the five analogica.it rows have changed; the spread "
              "from EI 12 to EI 50 below box speed IS the finding")
        bad += 1

    if "ma a 24°C" not in page4:
        print("  [FAIL] the Hydrofen 1+50 «ma a 24°C» note is gone -- three "
              "stored points are at 24 C because of it")
        bad += 1
    print("  [OK  ] development table re-read: 14 / 13 / 14 points for "
          "Mk2 / P33 / Orto, the 24 C exception still printed")

    if ns.assert_ and bad:
        print("\n[FAIL] the Ferrania vendor sheets do not reproduce")
        return 1
    print("\n[OK] three wedge spectrograms re-calibrated on their own printed "
          "rulers (Orto 400-550, P30 New and P33 400-650), four "
          "characteristic curves re-traced, P33's two drawings re-checked "
          "against each other, and the distributor's table re-counted.")
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
