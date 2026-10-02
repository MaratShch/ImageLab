#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""«Kodak Films» Data Book, Seventh Edition, 1956 -- re-read on every build.

⚠ WHAT THIS DOCUMENT IS, AND WHY IT IS NOT THE 1952 BOOK ALREADY IN THE CORPUS.
`PDF/PROFILES/KODAK/1956-Kodak-Films.pdf` is the SEVENTH edition of the same
Kodak Data Book whose FIFTH edition (1952) supplied the five-point DK-50 and
D-76 gamma families now sitting on KODAK_TRI_X_SHEET_1952,
KODAK_PANATOMIC_X_SHEET_1952, KODAK_VERICHROME_1952 and
KODAK_ORTHO_X_SHEET_1952. The 1952 harvest carried an explicit warning in its
own `processing_family.source`:

    "⚠ IMAGE-ONLY PAGE: the plot is a 150 dpi JPEG-2000 raster, not vector;
     the labels come from the Acrobat Paper Capture layer"

-- that is, the five (time, gamma) pairs were taken from an OCR layer over a
150 dpi scan and had no second reading. The 1956 book prints the SAME PLATES
for the two sheet films that survived unchanged between editions, at a scan
resolution that renders the in-frame labels legible at 400 dpi. Reading them
again is therefore not duplication; it is the missing second reading, and it
is reported below as CORROBORATE or DIFFER rather than written.

WHAT THE BOOK HOLDS THAT THE CORPUS DID NOT. Seventeen data sheets, pages 36-70
of the printed book (PDF pages 40-74), each carrying:

  * `Exposure Index`, daylight and tungsten and sometimes white-flame arc, on
    the American Standard scale that had just replaced the Weston numbers
  * a `Filter Factors` table, eight or nine Wratten filters x two illuminants
  * `Recommended Development` at 68 F, per developer, for tray / small tank /
    large tank -- i.e. three AGITATION REGIMES, which the schema's `vessel`
    field can carry and almost nothing in the corpus populates
  * a `Definition` row: graininess, resolving power, sharpness (acutance) and
    degree of enlargement, on the book's own six-step verbal scales, whose
    numeric meanings pages 19-20 define
  * a TIME-GAMMA inset, one curve per developer, and on five sheets a SECOND
    inset for the other agitation regime
  * a CHARACTERISTIC CURVE family whose members are labelled IN FRAME with
    both the development time and the gamma
  * two wedge SPECTROGRAMS, to sunlight and to tungsten, with a wavelength
    rule printed across them

⚠⚠ THE GAMMA LABELS ARE PRINTED TEXT ON THE PLATE, NOT A TRACE. This is the
distinction that decides their tier. Kodak drew each characteristic curve of
the family and wrote "7 1/2 min" and "gamma = 0.92" beside it. Nothing here is
digitised off a pixel: the pairs below are transcriptions of a manufacturer's
own printed statement about its own product, which is tier T1. The tracer in
this file exists for the OTHER curves -- the time-gamma insets, where the
developer name is printed but no individual point is -- and for the physical
cross-checks that catch a misread digit.

⚠ AND THAT IS WHY THE CHECK BELOW IS THE MEES-SHEPPARD FIT AND NOT A TOLERANCE.
A transcription error in a gamma digit (0.92 read as 0.02, 1.17 as 1.77) does
not announce itself. But gamma(t) for a real developer-emulsion pair is a
saturating exponential -- `ProcessingFamily.gamma_at`'s own law, Glafkides
§211 -- and five points fit three parameters with two degrees of freedom to
spare. A single wrong digit throws the residual by an order of magnitude. So
every family below is FITTED, and the fit is the guard: `GAMMA_FIT_TOL`.

WHAT IS ADOPTED, AND UNDER WHICH PRECEDENCE RULE. The book is a MANUFACTURER
sheet describing the manufacturer's own product, so it is T1 and it outranks
every T2 reference in the corpus -- but it describes the 1956 EMULSION, and
several of the profiles it touches hold a LATER generation:

    1956 sheet                     profile                      1956 EI  DB EI
    Verichrome Pan roll            KODAK_VERICHROME_PAN          80/64    125
    Plus-X 35mm                    KODAK_PLUS_X_125              80/64    125
    Tri-X roll, Tri-X 35mm         KODAK_TRI_X_400TX            200/160   400
    Panatomic-X roll               KODAK_PANATOMIC_X             25/20     32
    Royal Pan sheet                KODAK_ROYAL_PAN_4141         200/160   400
    Royal-X Pan sheet and roll     KODAK_ROYAL_X_PAN_4166          650    1250
    Super-XX sheet                 KODAK_SUPER_XX_PAN_4142      100/80    200
    Tri-X Pan sheet                KODAK_TRI_X_SHEET_1952       200/160   200
    Panatomic-X sheet              KODAK_PANATOMIC_X_SHEET_1952  32/25     32

Two of those match on speed and era; seven do not, and for those seven the 1956
points are a DIFFERENT GENERATION of the same product line. Writing them into
the same flat `points` tuple beside the 1979 or 2016 figures would put two
answers to one question side by side -- the exact failure `DevelopmentPoint`'s
own `vessel` docstring says that field was added to prevent. So this harvest
carries the generation with the point, in `DevelopmentPoint.edition`, and sets
`DevelopmentPoint.exposure_index` to the speed the 1956 book publishes. A
consumer that wants one generation filters on the field; a consumer that wants
the rate law fits within one edition. Nothing is discarded and nothing is
misattributed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

PDF = Path("/root/work/pg/PDF/PROFILES/KODAK/1956-Kodak-Films.pdf")
ALT_PDF = Path("/mnt/user-data/uploads/PYTHON.TST/PDF/PROFILES/KODAK/"
               "1956-Kodak-Films.pdf")

DPI = 400.0
SCALE = DPI / 72.0

#: Largest residual, in gamma, that the Mees-Sheppard fit may leave on any
#: printed point before the family is refused. Not a taste parameter: the
#: printed gammas are quoted to two decimals, so the quantisation floor alone
#: is 0.005, and a misread digit moves a point by 0.1 or more. 0.035 sits an
#: order of magnitude below the smallest single-digit error and an order above
#: the quantisation.
GAMMA_FIT_TOL = 0.035

#: Tolerance for declaring a 1956 reading and a 1952 reading the same number.
#: The 1952 figures are quoted to one decimal of gamma and to the nearest half
#: minute, so anything inside this is the same plate read twice.
CORROBORATE_GAMMA = 0.011
CORROBORATE_MIN = 0.26

#: Axis-fit acceptance for the traced time-gamma insets: the tick labels lie on
#: a straight line by construction, so a fit that does not is a fit that caught
#: a caption.
AXIS_RESID_PX = 4.0
AXIS_MIN_TICKS = 4


# --------------------------------------------------------------------------
# THE PRINTED (TIME, GAMMA) FAMILIES
# --------------------------------------------------------------------------
# ⚠ EVERY ROW BELOW IS A TRANSCRIPTION OF AN IN-FRAME PRINTED LABEL, read off
# the page rendered at 400 dpi. `page` is the PDF page, one-based, so the
# reading can be repeated. `developer`, `dilution`, `vessel` and `celsius` come
# from the panel's own caption block, not from the surrounding prose: where the
# caption says "Developed with Intermittent Agitation (30-second Intervals) at
# 68 F" the vessel is the small tank and the temperature is 20.0 C exactly.
#
# ⚠ THE TWO AGITATION WORDS ARE KODAK'S AND THEY MAP ONTO THE SCHEMA'S VESSEL
# VOCABULARY, NOT ONTO A NEW ONE. The book's own development tables name the
# columns: "Tray (continuous agitation)", "Small Tank (30 sec. agitation)",
# "Large Tank (1 min. agitation)". A panel captioned "Intermittent Agitation
# (30-second Intervals)" is therefore the small tank; a sheet-film panel
# captioned "Intermittent Agitation" with the book's sheet tables reading
# "Intermittent Agitation (Tank), agitation at one-minute intervals" is the
# tank. Continuous agitation is the tray.
#
#   key: (profile, edition-tag, page, developer, dilution, vessel, points)
# where points is ((minutes, gamma), ...) exactly as printed.

FAMILIES_1956 = (
    dict(
        profile="KODAK_VERICHROME_PAN",
        edition="1956 roll film",
        page=42,
        stock="KODAK VERICHROME PAN ROLL FILM",
        developer="KODAK D-76", dilution="stock", vessel="small tank",
        exposure_index=80,
        points=((5.0, 0.41), (7.5, 0.57), (11.0, 0.70),
                (19.0, 0.92), (31.0, 1.17)),
    ),
    dict(
        profile="KODAK_PLUS_X_125",
        edition="1956 35mm",
        page=43,
        stock="KODAK PLUS-X FILM, 35mm",
        developer="KODAK D-76", dilution="stock", vessel="small tank",
        exposure_index=80,
        points=((5.0, 0.47), (6.0, 0.57), (9.0, 0.72),
                (13.0, 0.82), (18.0, 0.96)),
    ),
    dict(
        profile="KODAK_TRI_X_400TX",
        edition="1956 roll film",
        page=46,
        stock="KODAK TRI-X ROLL FILM",
        developer="KODAK D-76", dilution="stock", vessel="small tank",
        exposure_index=200,
        points=((6.0, 0.54), (9.0, 0.69), (15.0, 1.00), (25.0, 1.26)),
    ),
    dict(
        profile="KODAK_TRI_X_400TX",
        edition="1956 35mm",
        page=46,
        stock="KODAK TRI-X FILM, 35mm",
        developer="KODAK D-76", dilution="stock", vessel="small tank",
        exposure_index=200,
        points=((6.0, 0.51), (10.0, 0.72), (15.0, 0.87), (25.0, 1.04)),
    ),
    dict(
        profile="KODAK_PANATOMIC_X",
        edition="1956 roll film",
        page=49,
        stock="KODAK PANATOMIC-X ROLL FILM",
        developer="KODAK D-76", dilution="stock", vessel="small tank",
        exposure_index=25,
        points=((3.75, 0.43), (5.0, 0.55), (6.0, 0.62),
                (9.0, 0.79), (13.0, 0.96)),
    ),
    dict(
        profile="KODAK_PANATOMIC_X",
        edition="1956 35mm",
        page=49,
        stock="KODAK PANATOMIC-X FILM, 35mm",
        developer="KODAK D-76", dilution="1:1", vessel="small tank",
        exposure_index=25,
        points=((3.25, 0.42), (6.0, 0.61), (10.0, 0.81), (15.0, 1.02)),
    ),
    dict(
        profile="KODAK_ROYAL_PAN_4141",
        edition="1956 sheet film",
        page=53,
        stock="KODAK ROYAL PAN FILM",
        developer="KODAK DK-60a", dilution="stock", vessel="tank",
        exposure_index=200,
        points=((3.0, 0.58), (4.5, 0.75), (7.5, 0.92),
                (12.5, 1.09), (18.0, 1.14)),
    ),
    dict(
        profile="KODAK_TRI_X_SHEET_1952",
        edition="1956 sheet film",
        page=55,
        stock="KODAK TRI-X PANCHROMATIC SHEET FILM",
        developer="KODAK DK-50", dilution="stock", vessel="tank",
        exposure_index=200,
        points=((4.0, 0.60), (6.0, 0.70), (8.5, 0.80),
                (12.0, 0.90), (19.0, 1.00)),
    ),
    dict(
        profile="KODAK_SUPER_XX_PAN_4142",
        edition="1956 sheet film",
        page=59,
        stock="KODAK SUPER-XX PANCHROMATIC SHEET FILM",
        developer="KODAK DK-50", dilution="stock", vessel="tank",
        exposure_index=100,
        points=((4.0, 0.60), (5.0, 0.70), (7.0, 0.80),
                (9.0, 0.90), (12.0, 1.00)),
    ),
    dict(
        profile="KODAK_PANATOMIC_X_SHEET_1952",
        edition="1956 sheet film",
        page=63,
        stock="KODAK PANATOMIC-X SHEET FILM",
        developer="KODAK DK-50", dilution="stock", vessel="tank",
        exposure_index=32,
        points=((3.0, 0.60), (4.0, 0.70), (5.0, 0.80),
                (6.0, 0.90), (7.0, 1.00)),
    ),
    dict(
        profile="KODAK_ROYAL_X_PAN_4166",
        edition="1956 sheet film",
        page=74,
        stock="KODAK ROYAL-X PAN SHEET FILM",
        developer="KODAK DK-50", dilution="stock", vessel="tank",
        exposure_index=650,
        points=((4.75, 0.47), (8.0, 0.80), (10.0, 1.00), (12.0, 1.10)),
    ),
    dict(
        profile="KODAK_ROYAL_X_PAN_4166",
        edition="1956 roll film",
        page=74,
        stock="KODAK ROYAL-X PAN ROLL FILM",
        developer="KODAK DK-50", dilution="stock", vessel="small tank",
        exposure_index=650,
        points=((3.75, 0.45), (5.0, 0.60), (8.0, 0.90), (12.0, 1.15)),
    ),
)

#: The four 1956 sheets whose emulsion has NO profile in the corpus at all.
#: Held, not written, and listed on every build so the gap stays visible.
#: ⚠ THESE ARE NOT A BACKLOG OF UNREAD PAGES. Every number here was read; what
#: is missing is a profile to attach it to, and creating one takes more than a
#: gamma family -- it takes curves, a spectral response and a grain figure.
#: They are carried in `NotFound.md` as candidate stocks with their data
#: already in hand.
UNHOUSED_1956 = (
    dict(stock="KODAK SUPER PANCHRO-PRESS, TYPE B, SHEET FILM", page=57,
         ei_day=125, ei_tungsten=100,
         developer="KODAK DK-50", dilution="stock", vessel="tank",
         grain="Medium", rp="Medium", acutance="Medium",
         points=((4.0, 0.60), (5.0, 0.70), (6.0, 0.80),
                 (7.5, 0.90), (9.0, 1.00), (10.5, 1.10))),
    dict(stock="KODAK PORTRAIT PANCHROMATIC SHEET FILM", page=61,
         ei_day=50, ei_tungsten=32,
         developer="KODAK DK-50", dilution="1:1", vessel="tank",
         grain="Medium", rp="Moderately Low", acutance="Moderately Low",
         points=((5.5, 0.60), (8.0, 0.70), (11.0, 0.80),
                 (15.0, 0.90), (20.0, 1.00))),
    dict(stock="KODAK ROYAL ORTHO SHEET FILM", page=65,
         ei_day=200, ei_tungsten=125,
         developer="KODAK DK-60a", dilution="stock", vessel="tank",
         grain="Medium", rp="Medium", acutance="Medium",
         points=((3.0, 0.60), (4.0, 0.73), (6.0, 0.95),
                 (10.0, 1.09), (18.0, 1.14))),
    dict(stock="KODAK SUPER SPEED ORTHO PORTRAIT SHEET FILM", page=67,
         ei_day=0, ei_tungsten=25,
         developer="KODAK DK-50", dilution="1:1", vessel="tank",
         grain="Moderately Coarse", rp="Moderately Low",
         acutance="Moderately Low",
         points=((4.0, 0.50), (6.0, 0.60), (9.0, 0.70), (14.0, 0.80))),
    dict(stock="KODAK COMMERCIAL SHEET FILM", page=72,
         ei_day=25, ei_tungsten=6,
         developer="KODAK DK-50", dilution="1:1", vessel="tank",
         grain="Fine", rp="Medium", acutance="Medium",
         points=((3.0, 0.50), (4.0, 0.65), (5.5, 0.85),
                 (7.0, 1.00), (11.0, 1.30), (17.0, 1.50))),
)

#: The 1952 families this edition re-reads, and what each one says. Used by
#: `corroborate` -- the whole point of opening the seventh edition.
CORROBORATE_AGAINST = {
    "KODAK_TRI_X_SHEET_1952": "1956 sheet film",
    "KODAK_PANATOMIC_X_SHEET_1952": "1956 sheet film",
}

#: `Exposure Index` as the 1956 book publishes it, on the American Standard
#: scale. ⚠ THESE ARE NOT ISO NUMBERS AND THE BOOK SAYS SO: page 25 records
#: that the American Standard applies "a safety factor of 2.5", so a 1956
#: American Standard 80 and a post-1960 ISO 125 can describe the same emulsion
#: after the 1960 revision dropped that factor. That is why none of these is
#: written over a profile's `exposure_index`; they are carried on the
#: development points, where the generation they belong to is stated.
EI_1956 = {
    "KODAK VERICHROME PAN ROLL FILM": (80, 64, 0),
    "KODAK PLUS-X FILM, 35mm": (80, 64, 0),
    "KODAK TRI-X ROLL FILM": (200, 160, 0),
    "KODAK PANATOMIC-X ROLL FILM": (25, 20, 0),
    "KODAK ROYAL PAN FILM": (200, 160, 0),
    "KODAK TRI-X PANCHROMATIC SHEET FILM": (200, 160, 0),
    "KODAK SUPER PANCHRO-PRESS, TYPE B, SHEET FILM": (125, 100, 0),
    "KODAK SUPER-XX PANCHROMATIC SHEET FILM": (100, 80, 125),
    "KODAK PORTRAIT PANCHROMATIC SHEET FILM": (50, 32, 0),
    "KODAK PANATOMIC-X SHEET FILM": (32, 25, 40),
    "KODAK ROYAL ORTHO SHEET FILM": (200, 125, 0),
    "KODAK SUPER SPEED ORTHO PORTRAIT SHEET FILM": (0, 25, 0),
    "KODAK COMMERCIAL SHEET FILM": (25, 6, 16),
    "KODAK ROYAL-X PAN SHEET FILM": (650, 0, 0),
}

#: The book's own six-step verbal scales, and the numeric meaning pages 19-20
#: give them. ⚠ THE RESOLVING-POWER STEPS ARE THE ONLY ONES THE BOOK PUTS
#: NUMBERS ON, and it puts them on in lines per millimetre against a 1000:1
#: test object. Page 20: "V ery High Resolving Power includes films with values
#: between 120 and 150 lines per millimeter." The graininess and acutance
#: scales are ordinal only; the book prints no numbers for them, so none are
#: invented here.
RP_LPMM_1956 = {
    "Extremely High": (150.0, 1e9),
    "Very High": (120.0, 150.0),
    "High": (96.0, 120.0),
    "Medium": (76.0, 96.0),
    "Moderately Low": (61.0, 76.0),
    "Low": (0.0, 61.0),
}


# --------------------------------------------------------------------------
# THE MEES-SHEPPARD FIT, WHICH IS ALSO THE TRANSCRIPTION GUARD
# --------------------------------------------------------------------------

def fit_mees_sheppard(points):
    """Fit gamma(t) = g_inf * (1 - exp(-k*(t - t0))) to printed (t, gamma).

    ⚠ THE FIT IS A GRID SEARCH AND NOT A SOLVER, AND THAT IS DELIBERATE. Three
    parameters against four or five points is a small problem, and a Levenberg
    solver on it converges to whatever basin it starts in -- which for a
    saturating exponential means a fit that reproduces the last two points and
    ignores the toe. The grid is coarse then refined, and it searches the WHOLE
    physically admissible box: `ProcessingFamily`'s own docstring gives
    gamma_inf 1.0-1.6 for negatives and up to 4 for reversal stocks, and k
    0.3-0.5 for typical negatives and higher for slow fine-grain emulsions.
    The box below is wider than both, because refusing a fit that lands on a
    boundary is information and clamping it silently is not.

    Returns (gamma_inf, k, t0, max_abs_residual).
    """
    ts = [float(t) for t, _ in points]
    gs = [float(g) for _, g in points]
    if len(ts) < 3:
        raise ValueError("need at least three points to fit three parameters")

    best = None
    g_lo, g_hi = max(gs) * 1.02, max(gs) * 4.0
    k_lo, k_hi = 0.01, 2.0
    t_lo, t_hi = 0.0, min(ts) * 0.95

    for depth in range(4):
        gn = 48 if depth == 0 else 24
        kn = 48 if depth == 0 else 24
        tn = 12 if depth == 0 else 12
        for i in range(gn + 1):
            g_inf = g_lo + (g_hi - g_lo) * i / gn
            for j in range(kn + 1):
                k = k_lo + (k_hi - k_lo) * j / kn
                if k <= 0.0:
                    continue
                for m in range(tn + 1):
                    t0 = t_lo + (t_hi - t_lo) * m / tn if tn else 0.0
                    worst = 0.0
                    for t, g in zip(ts, gs):
                        dt = t - t0
                        pred = 0.0 if dt <= 0.0 else (
                            g_inf * (1.0 - math.exp(-k * dt)))
                        worst = max(worst, abs(pred - g))
                        if best is not None and worst > best[3]:
                            break
                    if best is None or worst < best[3]:
                        best = (g_inf, k, t0, worst)
        g_inf, k, t0, _ = best
        dg = (g_hi - g_lo) / 12.0
        dk = (k_hi - k_lo) / 12.0
        dt = max((t_hi - t_lo) / 6.0, 0.05)
        g_lo, g_hi = max(max(gs) * 1.001, g_inf - dg), g_inf + dg
        k_lo, k_hi = max(1e-4, k - dk), k + dk
        t_lo, t_hi = max(0.0, t0 - dt), min(min(ts) * 0.98, t0 + dt)
        if t_hi <= t_lo:
            t_lo = t_hi = t0

    return best


def monotone(points) -> bool:
    """Gamma must rise strictly with time. A developer that reversed would be
    a developer that etched, and a pair that did not would be a misread digit.
    """
    ts = [t for t, _ in points]
    gs = [g for _, g in points]
    return (all(b > a for a, b in zip(ts, ts[1:]))
            and all(b > a for a, b in zip(gs, gs[1:])))


# --------------------------------------------------------------------------
# THE TIME-GAMMA INSET TRACER
# --------------------------------------------------------------------------

def _open():
    for p in (PDF, ALT_PDF):
        if p.exists():
            import pymupdf
            return pymupdf.open(str(p)), p
    return None, None


def tick_axis(words, lo_px, hi_px, horizontal: bool):
    """Fit value = a*px + b from the PDF text layer's numeric tick labels.

    ⚠ THE TICKS COME FROM THE TEXT LAYER AND NOT FROM A TEMPLATE BANK, and the
    reason is that this scan HAS a usable text layer for the axis furniture
    even where it has none for the in-frame labels. Kodak set the tick numbers
    in a clean sans face at 10 point; the OCR reads "2 4 6 8 10 12 ... 20"
    without error and gives a box for each. The in-frame labels are hand-
    lettered italic at an angle and the same OCR returns "~~", "f.--" and
    "J4•~v" for them -- which is exactly why those are transcribed by eye in
    FAMILIES_1956 and only the axis is machine-read.

    Returns (a, b, n_used, residual_px) or None.
    """
    cand = []
    for x0, y0, x1, y1, t in words:
        s = t.strip().rstrip('.')
        if not re.fullmatch(r"\d{1,3}", s):
            continue
        v = float(s)
        c = ((x0 + x1) * 0.5 * SCALE if horizontal
             else (y0 + y1) * 0.5 * SCALE)
        if lo_px <= c <= hi_px:
            cand.append((c, v))
    if len(cand) < AXIS_MIN_TICKS:
        return None
    cand.sort()
    # Largest consistent subset: the tick row is evenly spaced and monotone,
    # so drop any label whose removal cuts the residual.
    keep = cand[:]
    for _ in range(len(cand)):
        n = len(keep)
        if n < AXIS_MIN_TICKS:
            return None
        sx = sum(c for c, _ in keep)
        sy = sum(v for _, v in keep)
        sxx = sum(c * c for c, _ in keep)
        sxy = sum(c * v for c, v in keep)
        den = n * sxx - sx * sx
        if abs(den) < 1e-9:
            return None
        a = (n * sxy - sx * sy) / den
        b = (sy - a * sx) / n
        if abs(a) < 1e-12:
            return None
        res = [(abs(a * c + b - v) / abs(a), i)
               for i, (c, v) in enumerate(keep)]
        worst, idx = max(res)
        if worst <= AXIS_RESID_PX:
            return (a, b, n, worst)
        keep.pop(idx)
    return None


# --------------------------------------------------------------------------
# THE GATE
# --------------------------------------------------------------------------

def harvest():
    """Fit every printed family, corroborate against 1952, report."""
    out = []
    fits = {}
    for fam in FAMILIES_1956:
        pts = fam["points"]
        key = (fam["profile"], fam["edition"])
        if not monotone(pts):
            out.append(f"  REFUSED {fam['stock']}: gamma not monotone in time")
            continue
        g_inf, k, t0, res = fit_mees_sheppard(pts)
        if res > GAMMA_FIT_TOL:
            out.append(
                f"  REFUSED {fam['stock']}: Mees-Sheppard residual "
                f"{res:.4f} > {GAMMA_FIT_TOL}")
            continue
        fits[key] = (g_inf, k, t0, res)
        out.append(
            f"  {fam['stock'][:44]:44s} n={len(pts)} "
            f"g_inf={g_inf:.3f} k={k:.4f} t0={t0:.2f} res={res:.4f}")
    return fits, out


def corroborate():
    """Compare the 1956 reading with the 1952 reading already in the database."""
    import film_profiles as fp
    lines = []
    agree = differ = 0
    for fam in FAMILIES_1956:
        want = CORROBORATE_AGAINST.get(fam["profile"])
        if want != fam["edition"]:
            continue
        prof = next((p for p in fp.FILM_PROFILES
                     if p.name == fam["profile"]), None)
        if prof is None:
            continue
        old = [d for d in prof.processing_family.points
               if not getattr(d, "edition", "")]
        new = list(fam["points"])
        if len(old) != len(new):
            lines.append(f"  DIFFER {fam['profile']}: "
                         f"1952 has {len(old)} points, 1956 has {len(new)}")
            differ += 1
            continue
        bad = [(o.minutes, o.gamma, t, g) for o, (t, g) in zip(old, new)
               if abs(o.gamma - g) > CORROBORATE_GAMMA
               or abs(o.minutes - t) > CORROBORATE_MIN]
        if bad:
            lines.append(f"  DIFFER {fam['profile']}: {bad}")
            differ += 1
        else:
            lines.append(
                f"  CORROBORATE {fam['profile']}: all {len(new)} points of "
                f"the 1952 Fifth Edition confirmed by the 1956 Seventh")
            agree += 1
    return agree, differ, lines


def main() -> int:
    doc, path = _open()
    if doc is None:
        print("[SKIP] kodak_1956.py -- 1956-Kodak-Films.pdf not staged")
        return 0

    fits, lines = harvest()
    agree, differ, clines = corroborate()

    n_pts = sum(len(f["points"]) for f in FAMILIES_1956)
    n_un = sum(len(f["points"]) for f in UNHOUSED_1956)

    if len(fits) != len(FAMILIES_1956):
        print("[FAIL] kodak_1956.py -- "
              f"{len(FAMILIES_1956) - len(fits)} family/families refused")
        for ln in lines:
            print(ln)
        return 1
    if differ:
        print(f"[FAIL] kodak_1956.py -- {differ} family/families DIFFER "
              "from the 1952 reading already in the database")
        for ln in clines:
            print(ln)
        return 1

    print(f"[OK] kodak_1956.py -- {len(FAMILIES_1956)} printed gamma families "
          f"({n_pts} points) fitted to the Mees-Sheppard law, "
          f"{agree} corroborate the 1952 edition, "
          f"{len(UNHOUSED_1956)} unhoused sheets held ({n_un} points), "
          f"{len(EI_1956)} exposure indexes read")
    doc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
