#!/usr/bin/env python3
"""The FOUR-LAYER Fuji spectral sensitivity panels, 2026-09-06f (queue T5).

    PRO 800Z            PDF/PROFILES/FUJI/pro_800z_datasheet.pdf        AF3-177E p8 s19
    PORTRAIT NPZ 800    PDF/PROFILES/FUJI/NPZ.pdf                       AF3-100E p5 s18
    SUPERIA X-TRA 400   PDF/PROFILES/FUJI/superia_xtra400_datasheet.pdf AF3-151E p6 s18
    SUPERIA X-TRA 800   PDF/PROFILES/FUJI/superia_xtra800_datasheet.pdf AF3-068E p4 s16
    SUPERIA REALA       PDF/PROFILES/FUJI/superia_reala_datasheet.pdf   AF3-967E p4 s15

⚠⚠ EVERY ONE OF THESE PANELS DRAWS FOUR SENSITIVE LAYERS -- Blue, Green, Red
and CYAN -- and `SpectralSensitivity` carries three colour records plus pan.
These five stocks were REFUSED on that ground from 2026-09-06 to 2026-09-06f.

**OWNER DECISION, 2026-09-06f: store blue / green / red, and WRITE THE CYAN
CURVE OUT TO THE DOCUMENTATION rather than lose it.** That is what this module
implements, and the second half is not optional -- the objection to storing
three of four was never the three, it was losing the fourth silently. The cyan
trace is emitted as numbers, wavelength by wavelength, into
`doc/FUJI_FOURTH_LAYER.md`, and every one of the five profiles carries a
ParamSource saying its stored set is three of four by that decision.

⚠ THE CYAN LAYER IS DRAWN DASHED ON ALL FIVE SHEETS and that is what makes it
mechanically separable: it is a stroked path with a non-empty dash array while
blue, green and red are solid. So the four records are told apart by DASH
PATTERN, not by position or by guessing -- and that matters more than it
sounds. Cyan's measured peak is 516-519 nm on all four vector sheets, which
falls BETWEEN the blue and green records: sorting the four curves by wavelength
would name cyan "green" and push the real green record into red. Only the dash
array tells them apart.

⚠ THE ORDINATE IS A BRACKETED ARROW MARKED 1.0, NOT A NUMBERED LADDER. It gives
the SCALE and no absolute level, which is exactly what a peak-normalised store
needs and all this database has ever kept. Two horizontal rules one log unit
apart carry it. ⚠ ON REALA THAT RULE IS DRAWN IN TWO SEGMENTS, interrupted by
the "Green Sensitive Layer" label, so a full-width filter misses it and leaves
the caption-box rules looking like the ladder -- which would put the ordinate
out by 52 %.

Run bare for the report; `--assert` re-derives and fails on drift. `--emit-md`
regenerates the cyan documentation. Registered in build.py's audit stage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: The database's own spectral grid: 380 nm, 10 nm steps, 33 samples to 700 nm,
#: with -4.00 as the "no sensitivity measured here" floor every other set uses.
GRID = np.arange(380.0, 701.0, 10.0)
FLOOR = -4.00

WAVELENGTHS = [400, 500, 600, 700]

SHEETS = {
    "FUJICOLOR_PRO_800Z": dict(
        pdf="pro_800z_datasheet.pdf", page=7, ref="AF3-177E", section=19,
        frame=(341.102, 104.318, 520.091, 257.725),
        # the two rules the 1.0 bracket spans, and the four wavelength rules
        ordinate=(155.454, 206.590),
        abscissa=(351.017, 403.742, 456.087, 508.593)),
    "FUJICOLOR_SUPERIA_XTRA_400": dict(
        pdf="superia_xtra400_datasheet.pdf", page=5, ref="AF3-151E", section=18,
        frame=(339.07, 117.745, 533.614, 288.417),
        ordinate=(174.762, 231.947),
        abscissa=(350.424, 407.940, 464.883, 521.824)),
    "FUJICOLOR_SUPERIA_XTRA_800": dict(
        pdf="superia_xtra800_datasheet.pdf", page=3, ref="AF3-068E", section=16,
        frame=(332.214, 246.548, 512.118, 402.187),
        ordinate=(302.008, 351.959),
        abscissa=(344.868, 397.102, 448.891, 499.616)),
    "FUJICOLOR_SUPERIA_REALA": dict(
        pdf="superia_reala_datasheet.pdf", page=3, ref="AF3-967E", section=15,
        frame=(334.95, 124.07, 511.95, 275.605),
        # ⚠ 174.645 IS THE MEAN OF TWO SEGMENTS, 174.832 and 174.457, because
        # the "Green Sensitive Layer" label interrupts the rule. Taking only
        # full-width rules finds 148.426 and 225.594 instead -- the caption box
        # and the lower rule -- and returns 77.17 pt for one log unit against a
        # true 50.94, a 52 % error that every stored value would carry.
        ordinate=(174.645, 225.594),
        abscissa=(347.654, 399.117, 449.783, 499.783)),
}

#: ⚠⚠ NPZ 800 IS A RASTER TWIN, NOT A FIFTH VECTOR PANEL. Its sheet prints
#: this panel as a bilevel image with the cyan record dashed and the caption
#: block inside the frame, and a direct four-way separation was attempted on
#: 2026-09-06f and NOT achieved. What IS achieved -- and is re-derived on every
#: build by `overlay_npz()` below -- is the identity: PRO 800Z's four VECTOR
#: curves, mapped into this raster through the raster's OWN independent
#: calibration, land on ink at 98.1 % of sampled points, 100 % on each of the
#: three solid records and 89 % on the dashed one, which is what a dash pattern
#: gives. So NPZ 800 stores 800Z's extraction, exactly as it already stores
#: 800Z's MTF from the same pair of sheets.
NPZ = dict(
    pdf="NPZ.pdf", page=4, ref="AF3-100E", section=18, image_xref=19,
    # the raster's own ladder, in image pixels
    abscissa_px=(142.0, 361.0, 576.0, 792.0),   # 400 / 500 / 600 / 700 nm
    ordinate_px=(424.0, 212.0),                 # rel 0.0 and rel 1.0
    twin="FUJICOLOR_PRO_800Z")

#: What the overlay found, pinned. ⚠ THE NULL TESTS MATTER AS MUCH AS THE HIT
#: RATE: displacing the prediction by 10 nm or 0.10 log collapses it to 21-35 %,
#: so the 98.1 % is not "the panel is inky everywhere".
NPZ_OVERLAY = dict(hit_pct=98.1, solid_pct=100.0, worst_null_pct=35.0)

#: What this module found, pinned so a rerun that disagrees FAILS.
#: (blue, cyan, green, red) peak wavelengths in nm.
#: ⚠ CYAN LANDS AT 516-519 nm ON ALL FOUR SHEETS -- a 3 nm spread across four
#: different films and four separately calibrated panels. Blue spans 463-471,
#: green 529-557 and red 628-630, so the fourth layer is the most consistently
#: placed of the four. That is what a deliberately engineered interlayer looks
#: like, and it is also the best evidence that the dash-pattern separation is
#: picking out the same physical record on every sheet.
EXPECTED = {
    "FUJICOLOR_PRO_800Z": dict(peaks=(471, 519, 552, 629)),
    "FUJICOLOR_SUPERIA_XTRA_400": dict(peaks=(471, 516, 557, 628)),
    "FUJICOLOR_SUPERIA_XTRA_800": dict(peaks=(463, 517, 534, 630)),
    "FUJICOLOR_SUPERIA_REALA": dict(peaks=(465, 516, 529, 629)),
}


# ---------------------------------------------------------------------------
def _flatten(items, n=160):
    pts = []
    for it in items:
        if it[0] == "c":
            P = [np.array([q.x, q.y]) for q in it[1:5]]
            for t in np.linspace(0.0, 1.0, n):
                pts.append(((1 - t) ** 3) * P[0] + 3 * ((1 - t) ** 2) * t * P[1]
                           + 3 * (1 - t) * t * t * P[2] + (t ** 3) * P[3])
        elif it[0] == "l":
            pts.append(np.array([it[1].x, it[1].y]))
            pts.append(np.array([it[2].x, it[2].y]))
    return np.array(pts) if pts else np.zeros((0, 2))


def _subpaths(items):
    prev, sub, subs = None, [], []
    for it in items:
        if it[0] == "c":
            a, b = it[1], it[4]
        elif it[0] == "l":
            a, b = it[1], it[2]
        else:
            continue
        if prev is not None and (abs(a.x - prev.x) > 0.6
                                 or abs(a.y - prev.y) > 0.6):
            subs.append(sub)
            sub = []
        sub.append(it)
        prev = b
    if sub:
        subs.append(sub)
    return subs


def _dashed(dr) -> bool:
    """True when this stroked path carries a real dash array.

    PyMuPDF reports a solid stroke as the string '[] 0'. Anything with numbers
    between the brackets is dashed, and on these five sheets that is the CYAN
    record and nothing else -- '[ 2 1 ]', '[ 3 2 ]' and '[ 3.999 1.3 ]' across
    the four vector panels.
    """
    d = dr.get("dashes")
    if not d:
        return False
    inner = d[d.find("[") + 1: d.find("]")]
    return any(ch.isdigit() for ch in inner)


def read_panel(root: Path, name: str, verbose=True):
    cfg = SHEETS[name]
    doc = pymupdf.open(str(root / "PDF" / "PROFILES" / "FUJI" / cfg["pdf"]))
    page = doc[cfg["page"]]
    FR = pymupdf.Rect(*cfg["frame"])

    y0, y1 = cfg["ordinate"]
    pt_per_decade = abs(y1 - y0)
    ax = np.polyfit(np.array(WAVELENGTHS, dtype=float),
                    np.array(cfg["abscissa"], dtype=float), 1)
    res_x = float(np.abs(np.polyval(ax, WAVELENGTHS)
                         - np.array(cfg["abscissa"])).max())
    # ⚠ SQUARE-PANEL CHECK, REPORTED AND NEVER USED TO DERIVE AN AXIS. Fuji
    # draws 100 nm and one log decade at the same length on most of its sheets
    # and not on all; the number is a cross-check on the two independent
    # calibrations, not an input to either.
    square = abs(ax[0] * 100.0 / pt_per_decade - 1.0)
    if verbose:
        print("    %.3f pt per 100 nm (residual %.2f pt), %.3f pt per log "
              "decade; square to %.2f %%"
              % (ax[0] * 100.0, res_x, pt_per_decade, square * 100))

    solid, dash = [], []
    for dr in page.get_drawings():
        if not FR.contains(dr["rect"]):
            continue
        if dr.get("width") is None:          # a filled glyph, not a stroke
            continue
        if not any(it[0] == "c" for it in dr["items"]):
            continue
        if dr["rect"].width < FR.width * 0.12:
            continue
        if _dashed(dr):
            dash.append(dr)
        else:
            solid.extend(_subpaths(dr["items"]))

    recs = [_flatten(s) for s in solid]
    recs = [A for A in recs if len(A) > 40
            and A[:, 0].max() - A[:, 0].min() > FR.width * 0.10]
    if len(recs) != 3:
        raise ValueError("%s: %d solid records, expected 3" % (name, len(recs)))
    # ⚠ THE DASHED RECORD IS MERGED, NOT SPLIT. Its subpaths are DASH SEGMENTS,
    # so treating them the way the solid ones are treated would return a dozen
    # fragments instead of one curve.
    if len(dash) != 1:
        raise ValueError("%s: %d dashed paths, expected 1 (cyan)"
                         % (name, len(dash)))
    cyan = _flatten(dash[0]["items"])

    def to_curve(A):
        lam = (A[:, 0] - ax[1]) / ax[0]
        # y grows downward and sensitivity grows upward
        rel = (y1 - A[:, 1]) / pt_per_decade
        o = np.argsort(lam)
        return lam[o], rel[o]

    out = {}
    for key, A in zip(("s1", "s2", "s3"), recs):
        out[key] = to_curve(A)
    out["C"] = to_curve(cyan)

    # ⚠ RECORD ASSIGNMENT IS BY PEAK WAVELENGTH AND IS ASSERTED, NOT ASSUMED.
    # The three SOLID records must come out ascending B < G < R; anything else
    # means the panel was misread, and a swapped record is a wrong stored
    # sensitisation that looks entirely plausible.
    # ⚠ CYAN IS DELIBERATELY LEFT OUT OF THAT ORDERING. Its measured peak
    # (516-519 nm on all four sheets) falls BETWEEN blue's and green's, so
    # sorting all four by wavelength would name it "green" and demote the real
    # green record to red. The dash array is what keeps them apart, which is
    # exactly why the separation is mechanical and not geometric.
    peaks = {k: float(out[k][0][int(np.argmax(out[k][1]))])
             for k in ("s1", "s2", "s3", "C")}
    order = sorted(("s1", "s2", "s3"), key=lambda k: peaks[k])
    named = dict(zip(("B", "G", "R"), order))
    if not (peaks[named["B"]] < peaks[named["G"]] < peaks[named["R"]]):
        raise ValueError("%s: solid peaks do not ascend: %s" % (name, peaks))

    res = {}
    for band in ("B", "G", "R"):
        res[band] = out[named[band]]
    res["C"] = out["C"]

    grids, pk = {}, {}
    for band in ("B", "G", "R", "C"):
        lam, rel = res[band]
        pk[band] = float(lam[int(np.argmax(rel))])
        v = np.interp(GRID, lam, rel, left=-np.inf, right=-np.inf)
        v = v - np.nanmax(v[np.isfinite(v)])          # peak-normalise to 0.0
        v = np.where(np.isfinite(v), v, FLOOR)
        grids[band] = np.round(np.maximum(v, FLOOR), 2)

    if verbose:
        print("    peaks  B %.0f  C %.0f  G %.0f  R %.0f nm   (cyan sits "
              "between %s)"
              % (pk["B"], pk["C"], pk["G"], pk["R"],
                 "green and red" if pk["G"] < pk["C"] < pk["R"]
                 else "blue and green" if pk["B"] < pk["C"] < pk["G"]
                 else "NOWHERE EXPECTED"))
    return dict(grids=grids, peaks=pk, square=float(square),
                pt_per_decade=float(pt_per_decade), res_x=res_x,
                ref=cfg["ref"], section=cfg["section"], pdf=cfg["pdf"])


def overlay_npz(root: Path, verbose=True):
    """Is NPZ 800's raster spectral panel the SAME DRAWING as PRO 800Z's?

    ⚠ THE TEST IS AN OVERLAY, NOT A TRACE, and that is the point of it. Tracing
    a bilevel panel whose fourth record is dashed and whose caption sits inside
    the frame was attempted and failed; predicting where the twin's curves
    should fall and asking the raster whether there is ink there needs no
    separation at all. Each of PRO 800Z's four vector curves is sampled every
    2 nm, mapped into NPZ's pixel grid through NPZ's OWN ladder, and scored.

    ⚠⚠ AND IT IS SCORED AGAINST NULLS, because a hit rate alone proves nothing
    on a busy panel. Displacing the prediction by 10 nm, or by 0.10 in log
    sensitivity, must destroy it. It does: 98.1 % aligned against 21-35 % for
    every displacement tried.
    """
    doc = pymupdf.open(str(root / "PDF" / "PROFILES" / "FUJI" / NPZ["pdf"]))
    page = doc[NPZ["page"]]
    pix = pymupdf.Pixmap(doc, NPZ["image_xref"])
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, 0]
    ink = img > 128                      # this scan is INK-ON-BLACK, value 255
    H, W = ink.shape

    cfg = SHEETS[NPZ["twin"]]
    tdoc = pymupdf.open(str(root / "PDF" / "PROFILES" / "FUJI" / cfg["pdf"]))
    tp = tdoc[cfg["page"]]
    FR = pymupdf.Rect(*cfg["frame"])
    ty0, ty1 = cfg["ordinate"]
    tppd = abs(ty1 - ty0)
    tax = np.polyfit(np.array(WAVELENGTHS, dtype=float),
                     np.array(cfg["abscissa"], dtype=float), 1)
    solid, dash = [], []
    for dr in tp.get_drawings():
        if not FR.contains(dr["rect"]) or dr.get("width") is None:
            continue
        if not any(it[0] == "c" for it in dr["items"]):
            continue
        if dr["rect"].width < FR.width * 0.12:
            continue
        (dash if _dashed(dr) else solid).append(dr)
    curves = []
    for dr in solid:
        for sub in _subpaths(dr["items"]):
            A = _flatten(sub)
            if len(A) > 40 and A[:, 0].max() - A[:, 0].min() > FR.width * 0.10:
                curves.append((A, False))
    for dr in dash:
        curves.append((_flatten(dr["items"]), True))

    nx = np.polyfit(np.array(WAVELENGTHS, dtype=float),
                    np.array(NPZ["abscissa_px"], dtype=float), 1)
    ny0, ny1 = NPZ["ordinate_px"]

    def score(dnm=0.0, drel=0.0):
        tot = hit = 0
        per = []
        for A, is_dash in curves:
            lam = (A[:, 0] - tax[1]) / tax[0]
            rel = (ty1 - A[:, 1]) / tppd
            o = np.argsort(lam)
            lam, rel = lam[o], rel[o]
            g = np.arange(max(400.0, lam.min()) + 2,
                          min(700.0, lam.max()) - 2, 2.0)
            if len(g) < 10:
                continue
            rr = np.interp(g, lam, rel) + drel
            px = np.polyval(nx, g + dnm)
            py = ny0 + rr * (ny1 - ny0)
            n = k = 0
            for x, y in zip(px, py):
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < W and 0 <= yi < H):
                    continue
                n += 1
                if ink[max(0, yi - 4):yi + 5, max(0, xi - 2):xi + 3].any():
                    k += 1
            tot += n
            hit += k
            per.append((is_dash, k, n, float(g[0]), float(g[-1])))
        return (100.0 * hit / max(tot, 1)), tot, per

    aligned, npts, per = score()
    nulls = {lab: score(dn, dr)[0] for lab, dn, dr in
             (("+10 nm", 10.0, 0.0), ("-10 nm", -10.0, 0.0),
              ("+20 nm", 20.0, 0.0), ("+0.10 log", 0.0, 0.10),
              ("-0.10 log", 0.0, -0.10), ("+0.25 log", 0.0, 0.25))}
    solid_hits = sum(k for d, k, n, _a, _b in per if not d)
    solid_n = sum(n for d, k, n, _a, _b in per if not d)
    solid_pct = 100.0 * solid_hits / max(solid_n, 1)
    if verbose:
        print("    overlay of %s's four VECTOR curves onto this RASTER, "
              "through the raster's own ladder:" % NPZ["twin"])
        for is_dash, k, n, lo, hi in per:
            print("      %-6s %3d/%3d on ink (%3.0f %%) over %.0f-%.0f nm"
                  % ("dashed" if is_dash else "solid", k, n,
                     100.0 * k / max(n, 1), lo, hi))
        print("    ALIGNED %.1f %% of %d points (solid records %.1f %%); "
              "the dashed record scores lower because 11 %% of its length is "
              "gaps" % (aligned, npts, solid_pct))
        print("    ⚠ NULLS: " + ", ".join("%s %.0f %%" % (k, v)
                                          for k, v in nulls.items()))
    return dict(aligned=aligned, solid=solid_pct, npts=npts,
                worst_null=max(nulls.values()), nulls=nulls)


# ---------------------------------------------------------------------------
def emit_md(results, path: Path):
    """Write the CYAN curves out as numbers. This is the owner's condition."""
    L = []
    L.append("# The Fuji fourth colour layer — traced, and NOT stored\n")
    L.append("**Generated by `fuji_spectral4_2026.py --emit-md`. Do not hand-edit.**\n")
    L.append("""
⚠⚠ **THIS FILE EXISTS BECAUSE THE DATABASE CANNOT HOLD WHAT IS IN IT.**

Five Fuji colour negatives carry a **fourth sensitive layer** — cyan — in addition to the usual
blue, green and red. `SpectralSensitivity` has three colour records and a panchromatic one, so the
stored set on each of these stocks is **three of the four curves its datasheet prints**.

**Owner decision, 2026-09-06f:** store R/G/B, ignore cyan *in the schema*, and record the cyan
measurement here. The layer is therefore **measured, published and documented — just not stored**.
Nothing about it was guessed and nothing was thrown away.

**What this means for anyone reading the stored arrays:** they describe three of four layers. Do
not treat them as a complete sensitisation of these films, and do not enable
`RenderSettings.spectral_taking` on them expecting a correct answer — that path integrates the
stored records against the illuminant, and integrating three quarters of a film is wrong in a way
no tolerance will catch. It is off by default.

**Why cyan exists:** Fuji's "4th Colour Layer Technology". Measured here, its peak lands at
**516–519 nm on all four vector sheets** — a 3 nm spread across four different films and four
separately calibrated panels, where blue spans 463–471, green 529–557 and red 628–630. It sits in
the blue-green gap between the blue and green records, which is where fluorescent lamps put their
worst mercury spikes and where a three-layer film has its largest blind region. It is what these
films are sold on for mixed and fluorescent light, and the tightness of that 3 nm spread is also
the best evidence that the dash-pattern separation picks out the same physical record every time.

**How it was told apart from the other three:** it is drawn **dashed** on every sheet, and blue,
green and red are solid. The separation is mechanical — a dash array in the PDF — not a guess from
position, which matters because cyan's peak falls between two of the solid records and no
geometric rule could break that tie.

**Scale:** every one of these panels carries a bracketed arrow marked **1.0**, not a numbered
ladder, so the curves have a scale and no absolute level. All values below are **log sensitivity
relative to that record's own peak**, on the database's own 380–700 nm / 10 nm grid, floored at
−4.00 where the panel draws nothing. That is the same normalisation all 90 stored sets use.

---
""")
    for name in sorted(results):
        r = results[name]
        L.append("\n## %s\n" % name.replace("_", " "))
        L.append("`%s`, **%s** section %d — `PDF/PROFILES/FUJI/%s`\n"
                 % (name, r["ref"], r["section"], r["pdf"]))
        if r.get("shared_from"):
            L.append(
                "\n⚠⚠ **THESE NUMBERS ARE `%s`'s TRACE, NOT A SEPARATE READING "
                "OF THIS SHEET.** This panel is a bilevel **raster** and its "
                "twin's is vector, so the two were matched by **overlay** "
                "rather than traced twice: the twin's four curves, sampled "
                "every 2 nm and mapped through THIS panel's own independent "
                "ladder, land on ink at **%.1f %%** of %d points — **%.0f %%** "
                "on each of the three solid records. Displace the prediction "
                "and it collapses: %s. One drawing, read once.\n"
                % (r["shared_from"], r["overlay"]["aligned"],
                   r["overlay"]["npts"], r["overlay"]["solid"],
                   ", ".join("%s → %.0f %%" % (k, v)
                             for k, v in r["overlay"]["nulls"].items())))
        L.append("")
        L.append("| | peak (nm) |")
        L.append("|---|---|")
        for band, label in (("B", "blue"), ("C", "**cyan — NOT STORED**"),
                            ("G", "green"), ("R", "red")):
            L.append("| %s | %.0f |" % (label, r["peaks"][band]))
        L.append("")
        L.append("Calibration: %.3f pt per log decade, %.2f pt worst "
                 "wavelength residual, panel square to %.2f %%.\n"
                 % (r["pt_per_decade"], r["res_x"], r["square"] * 100))
        L.append("**Cyan-sensitive layer, log sensitivity relative to its own "
                 "peak, 380–700 nm in 10 nm steps:**\n")
        L.append("```python")
        L.append("# %s -- the FOURTH layer. Traced, documented, not stored." % name)
        L.append("log_s_cyan = (")
        v = r["grids"]["C"]
        for i in range(0, len(v), 8):
            L.append("    " + " ".join("%6.2f," % x for x in v[i:i + 8]))
        L.append(")")
        L.append("```")
        L.append("")
    L.append("\n---\n")
    L.append("**To close this file properly**, `SpectralSensitivity` needs a fourth colour record "
             "(`log_s_c`) and `AlgoSpectralSensitivity` needs to read it. That is a schema change, "
             "not a tracing job: every curve above is already measured and would drop straight in.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    ap.add_argument("--emit-md", action="store_true")
    ap.add_argument("--emit-py", action="store_true")
    ns = ap.parse_args(argv)
    root = Path(ns.root).resolve()

    print("The FOUR-LAYER Fuji spectral panels -- blue, green, red and CYAN")
    print("  ⚠ cyan is stored NOWHERE: schema has three colour records. It is "
          "traced here and written to doc/FUJI_FOURTH_LAYER.md")
    results, bad = {}, 0
    for name in SHEETS:
        print("\n  %s -- %s section %d"
              % (name, SHEETS[name]["ref"], SHEETS[name]["section"]))
        try:
            results[name] = read_panel(root, name, verbose=True)
        except ValueError as e:
            print("  [FAIL] %s" % e)
            bad += 1
            continue
        exp = EXPECTED.get(name, {}).get("peaks")
        if exp and exp[0] is not None:
            got = tuple(round(results[name]["peaks"][b], 0)
                        for b in ("B", "C", "G", "R"))
            if any(abs(a - b) > 6 for a, b in zip(got, exp)):
                print("  [MISMATCH] %s peaks %s vs pinned %s"
                      % (name, got, exp))
                bad += 1

    print("\n  FUJICOLOR_PORTRAIT_NPZ_800 -- %s section %d, a RASTER twin"
          % (NPZ["ref"], NPZ["section"]))
    ov = overlay_npz(root, verbose=True)
    if ov["aligned"] < NPZ_OVERLAY["hit_pct"] - 1.5:
        print("  [MISMATCH] NPZ overlay %.1f %% against a pinned %.1f"
              % (ov["aligned"], NPZ_OVERLAY["hit_pct"]))
        bad += 1
    if ov["solid"] < 99.0:
        print("  [MISMATCH] NPZ overlay: the three SOLID records score %.1f %%, "
              "under 99 -- the shared-drawing claim rests on them" % ov["solid"])
        bad += 1
    # ⚠ THE NULL BOUND IS THE LOAD-BEARING HALF. If a displaced prediction ever
    # scores as well as the aligned one, this panel has become inky enough that
    # the test proves nothing, and the identity must be re-argued.
    if ov["worst_null"] > 60.0 or ov["aligned"] - ov["worst_null"] < 40.0:
        print("  [MISMATCH] NPZ overlay: worst null %.1f %% against aligned "
              "%.1f -- the test no longer discriminates"
              % (ov["worst_null"], ov["aligned"]))
        bad += 1

    if ns.emit_md and results:
        # ⚠ NPZ 800 IS EMITTED FROM ITS TWIN'S TRACE, NOT FROM ITS OWN PANEL.
        # Its sheet prints the same drawing as a raster; the overlay above is
        # what licenses reusing PRO 800Z's numbers, and the entry says so.
        _twin = results.get(NPZ["twin"])
        if _twin is not None:
            results["FUJICOLOR_PORTRAIT_NPZ_800"] = dict(
                _twin, ref=NPZ["ref"], section=NPZ["section"], pdf=NPZ["pdf"],
                shared_from=NPZ["twin"], overlay=ov)
        out = HERE / "doc" / "FUJI_FOURTH_LAYER.md"
        emit_md(results, out)
        print("\n  wrote %s" % out)

    if ns.emit_py and results:
        for name in sorted(results):
            print("\n# ---- %s" % name)
            for band, field in (("R", "log_s_r"), ("G", "log_s_g"),
                                ("B", "log_s_b")):
                v = results[name]["grids"][band]
                print("            %s=(" % field)
                for i in range(0, len(v), 8):
                    print("                "
                          + " ".join("%.2f," % x for x in v[i:i + 8]))
                print("            ),")

    if ns.assert_:
        if bad:
            print("\n[FAIL] the four-layer Fuji panels do not reproduce")
            return 1
        print("\n[OK] %d four-layer Fuji spectral panels re-derived. Each "
              "separates its records by DASH PATTERN rather than position -- "
              "cyan is the only dashed stroke, and its peak at 516-519 nm "
              "falls BETWEEN blue's and green's, so sorting by wavelength "
              "would name it green and demote the real green record to red "
              "-- and each asserts that the three SOLID records ascend "
              "B < G < R. ⚠ Cyan lands within 3 nm on four different films "
              "and four separately calibrated panels, against 8 nm for blue "
              "and 28 for green, which is both what an engineered interlayer "
              "looks like and the best evidence the separation is picking out "
              "one physical record every time. ⚠ The cyan curve is traced on every sheet and "
              "stored on none: `SpectralSensitivity` has three colour records, "
              "so by the owner's decision of 2026-09-06f the stored set is "
              "three of four and the fourth lives in doc/FUJI_FOURTH_LAYER.md "
              "as numbers. Closing that needs a schema change, not a trace."
              % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
