"""Curve digitisation for the GEVAERT / Agfa-Gevaert journal scans (2026-08-19).

WHAT THESE SOURCES ARE, AND WHY THIS IS NOT dye_density.py OR granularity_vector.py
-----------------------------------------------------------------------------------
Both of those read VECTOR art: exact coordinates, axis ticks as text, no tracing.
The Gevaert material is the opposite end of the corpus -- three journal articles
scanned from paper, with **no text layer at all** (`pdftotext` yields 3-4 bytes per
file) and every plot a bitmap. So everything here is measured off pixels, and the
things that are free in a vector sheet have to be earned:

    * the page is SKEWED. Measured on the 1980 SMPTE scan: the y axis of Fig. 10
      drifts 9 px across 330 rows (~1.4 deg). A calibration that assumes a plumb
      axis puts the origin 9 px off, which on that plot is 0.05 D. Every axis here
      is therefore fitted as a LINE, not read as a column index.
    * the axis LABELS cannot be read as text, so the printed tick VALUES are
      supplied per figure below (read off the page by eye, quoted in the spec) and
      the tick POSITIONS are detected mechanically. Calibration is then a least
      squares fit of value against pixel with a reported residual -- the same
      discipline as the vector extractors, for the same reason.
    * the source resolution differs by a factor of two between documents and that
      is a hard limit on what can be claimed: the 1980 SMPTE scan is 1-bit at
      ~340 ppi (2277 x 3248 px), the 1968 Kino-Technik scan is JPEG colour at
      150 ppi (~940 x 1350 px). Six-parameter tone-curve fits are justified on the
      first; on the second the plots are ~300 px wide and the fit residual is
      reported so the difference is visible rather than hidden.

RESOLUTION POLICY (owner instruction, 2026-08-19): digitise at the highest
practical resolution and do NOT reduce a curve to a few representative points.
So each curve is traced at ONE SAMPLE PER PIXEL COLUMN -- 437 to 588 samples per
layer on the 1980 figures -- and the 6-parameter ToneCurve is then FITTED to all
of them by least squares, with the RMS and worst residual recorded in the profile.
That is the same method used for the Kodak VISION3 curves; the dense trace is the
measurement, the six parameters are how the database stores it, and the residual
is the honest statement of what the storage costs.

WHAT IS FITTED AND WHAT IS PINNED, per layer:
    dmin      PINNED to the measured left-plateau median (78-119 samples). It is
              measured directly; letting the optimiser move it lets curvature
              elsewhere pay for a wrong toe.
    gamma, toe_x, toe_k, shoulder_x, shoulder_k   fitted.
The project's monotonicity rule shoulder_k <= 1.4*toe_k is enforced as a penalty,
exactly as in digitize_plot.fit_tonecurve.

⚠ THE ABSCISSA HAS NO ABSOLUTE ANCHOR AND IS NOT INVENTED. Fig. 10's x axis reads
"LOG REL. EXP." 0 to 4.00 -- RELATIVE log exposure, with no statement of what 0
corresponds to in lux-seconds, and no printed speed point on the curve. The
database's tone-curve x is a log-exposure scale whose origin is the mid-grey
exposure. Those two origins cannot be related from anything printed in the paper,
so the traced curve is placed by INHERITING the existing profile's origin: the
offset is chosen so the green record reaches the same net density at x = 0 as the
profile already did. SHAPE, GAMMA and DMIN are therefore measured; the absolute
exposure placement is inherited and stays as uncertain as it was before.

VALIDATION THAT IS NOT SELF-REFERENTIAL: the paper prints "gamma = 0.57" on the
Fig. 10 green curve. The trace, calibrated only from tick positions, fits the green
record at gamma = 0.5712. That agreement to 0.001 is what licenses the rest of the
numbers off this figure.

Run:
    python gevaert_curves.py --root ../..            # extract + report
    python gevaert_curves.py --root ../.. --assert   # non-zero if a value moves
    python gevaert_curves.py --root ../.. --overlay /tmp/o.png
Needs numpy, Pillow and poppler's pdfimages (the embedded scans are extracted at
NATIVE resolution rather than re-rendered, so nothing is resampled).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import dashtrace as dt
import digitize_plot as dp
import film_profiles as fp

DARK = 0.5

#: The four scans, and which PDF page carries which figure.
DOCS = {
    "n682": ("GEVAERT/Verpoort_Stapp1980_NewGevacolNeg682.pdf",
             "Vervoort & Stappaerts, SMPTE Journal 89(9), September 1980, pp. 650-652"),
    "gev68": ("GEVAERT/Rens_vanBets1968Gevachr6.00.pdf",
              "Rens & Van Bets, Kino-Technik 1968 Nr. 10, pp. 260/262/264/266"),
}

#: ⚠ REGION BOXES ARE HAND-SUPPLIED AND THAT IS DELIBERATE. Each scanned page
#: carries three to five figures plus body text, and a bitmap page has no
#: structure to ask. The box only has to CONTAIN one figure with margin; the frame
#: lines, the ticks and the curves inside it are all found mechanically, and the
#: calibration residual would expose a box that had caught the wrong figure.
#: Coordinates are native-scan pixels, (x0, y0, x1, y1), page index from 0.
#:
#: TICK VALUES are read off the page by eye and quoted here verbatim from the
#: printed axis. They are the ONLY hand-read numbers in this file, they are
#: checked by the residual of the fit against the detected tick pixels, and on
#: Fig. 10 they are checked again by reproducing the printed gamma.
FIGURES = {
    # -- 1980 SMPTE, Type 682 -------------------------------------------------
    "fig10": dict(
        doc="n682", page=2, region=(120, 150, 790, 610),
        profile="GEVACOLOR_NEG_682",
        # ⚠ THE INHERITED ORIGIN IS A FROZEN CONSTANT, not a live lookup. It is the
        # net density above dmin that the PRE-2026-08-19 hand-fitted green curve
        # produced at x = 0, and it is what places this figure's "LOG REL. EXP."
        # axis on the database's mid-grey scale (the paper prints no absolute
        # anchor). Reading it from the live profile instead -- which the first
        # version did -- makes the extractor depend on the value it feeds, so the
        # offset drifted by 0.002 D on the run after adoption. Frozen here, the
        # extraction is reproducible from the PDF alone.
        origin_net_density=0.876,
        title='Fig. 10 "Sensitometric curves of the new Gevacolor negative film"',
        printed_page=652,
        # ⚠ TICK ANCHORS ARE MEASURED PIXEL POSITIONS, PINNED HERE, AND RE-VERIFIED
        # ON EVERY RUN. A fully automatic tick finder was written first and is kept
        # (`stub_ticks`); on this 1-bit skewed scan it cannot be trusted to pick the
        # right stubs unaided -- it caught the rule above the figure and the bottom
        # axis line as well as the real ticks, and "leftmost vertical ink" locks
        # onto the rotated "DENSITY" caption rather than the axis. So the anchors
        # below were located WITH that detector in a prototype run, checked against
        # the printed axis by eye, and frozen. `verify_anchor` then re-checks each
        # one against the pixels every run, so a drifting anchor fails loudly
        # instead of silently re-scaling a curve.
        # value -> pixel (row for density, column for log exposure)
        y_ticks={2.0: 246.0, 1.5: 328.5, 0.5: 495.5},
        x_ticks={0.0: 155.0, 1.00: 303.0, 2.00: 450.0, 3.00: 600.0, 4.00: 748.0},
        curves=("B", "G", "R"), kind="characteristic",
        seed_gap=25,
    ),
}

#: The DYE-DENSITY figures, kept in their own table because they are read by a
#: different procedure end to end: no tone-curve fit, no inherited abscissa, no
#: B/G/R ordering rule (these three curves cross five times), and a wavelength
#: axis instead of a log-exposure one. Sharing FIGURES' loop would have meant a
#: branch at every step of it.
#:
#: ⚠ ALL PIXEL COORDINATES ARE REGION-RELATIVE and were located with the same
#: detectors that re-verify them on every run (`fig8_ticks`). They are pinned for
#: the reason `fig10`'s are: an unaided detector on a 1-bit scan also finds the
#: rule above the figure and the axis captions, and a silently-wrong anchor
#: rescales a curve with nothing to disagree with it.
DYE_FIGURES = {
    "fig8": dict(
        doc="n682", page=1, region=(1450, 1880, 2250, 2420),
        profile="GEVACOLOR_NEG_682",
        title='Fig. 8 "Spectral density curves of the three dyes formed in '
              'the emulsions"',
        printed_page=651,
        # the plot frame's interior, region-relative
        frame=(143, 22, 718, 421),
        # WAVELENGTH ticks: the frame edges at 350 and 700 plus six interior
        # ticks at 50 nm. Only 400 / 500 / 600 / 700 are printed as numbers; the
        # rest are unlabelled ticks whose VALUES follow from the even 50 nm
        # spacing, and the calibration residual is what tests that reading.
        lam_ticks={350: 139.5, 400: 221.0, 450: 304.0, 500: 389.0,
                   550: 471.0, 600: 553.5, 650: 638.0, 700: 722.0},
        # DENSITY ticks: 2.0 is the top frame edge. 0.0 is NOT pinned -- the "0"
        # tick sits within 2 px of the bottom frame line and the two cannot be
        # told apart, so the axis is fitted from the four unambiguous anchors and
        # zero is where that fit puts it (row 423.8 against a frame line at
        # 425.5, i.e. 0.008 D apart -- recorded, not corrected).
        d_ticks={2.0: 17.0, 1.5: 120.0, 1.0: 221.5, 0.5: 321.5},
        # Seeded at the leftmost column clear of the tick stubs, where the three
        # curves are 0.11 D and 0.29 D apart. See `merge_px` in dashtrace for why
        # the crossings do not need to be seeded around.
        seed_x=152, merge_px=9.0,
    ),
}

#: Measured 2026-08-25. Peaks are the whole validation: the paper PRINTS its dye
#: peaks, so this is an external check in the same class as fig10's printed gamma.
EXPECTED_DYE = {
    "fig8": dict(
        # traced, in (cyan, magenta, yellow) order
        peaks_nm=(683.1, 522.1, 445.9),
        peaks_d=(1.459, 1.474, 1.462),
        # PRINTED in the paper, read 2026-08-19 (queue item G3/G7)
        printed_nm=(687.0, 525.0, 448.0),
        printed_d=(1.46, 1.48, 1.46),
        tol_nm=6.0, tol_d=0.03,
        # the in-figure glyph labels C / M / Y, by position only
        labels_nm=(683.0, 528.0, 448.0), tol_label_nm=8.0,
    ),
}

#: Measured 2026-08-19. --assert fails if a figure stops reproducing these.
EXPECTED = {
    "fig10": dict(
        dmin=(0.9137, 0.5863, 0.1356),      # B, G, R -- plateau medians
        gamma=(0.5396, 0.5677, 0.5056),
        # 0.008 is enough once the fit is multi-start and data-seeded: B 0.0063,
        # G 0.0040, R 0.0055. The earlier 0.012 existed only to accommodate a
        # local minimum the profile-seeded fit fell into (R at rms 0.0109).
        rms_max=0.008, tol=0.02,
        printed_gamma=0.57, printed_gamma_layer=1,   # the paper's own annotation
    ),
}


# --------------------------------------------------------------------------
def native_pages(pdf: Path, tmp: Path, tag: str) -> list[np.ndarray]:
    """Embedded page scans at NATIVE resolution, as float grey [0..1].

    ⚠ NOT a re-render. `page.get_pixmap(dpi=...)` would resample a bitmap that is
    already the only truth here; `pdfimages` hands back the stored raster
    untouched. Measured difference on Fig. 10: re-rendering at 200 dpi loses 40 %
    of the columns and with them 40 % of the samples the fit is claimed on.
    """
    out = subprocess.run(["pdfimages", "-png", str(pdf), str(tmp / tag)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"[!] pdfimages failed: {out.stderr[-500:]}")
    files = sorted(tmp.glob(f"{tag}-*.png"))
    return [np.asarray(Image.open(f).convert("L"), dtype=np.float64) / 255.0
            for f in files]


def fit_line(ys: np.ndarray, xs: np.ndarray, tol: float = 1.5, iters: int = 5):
    """Least-squares line with iterative outlier rejection. Returns (m, c, kept)."""
    ys, xs = np.asarray(ys, float), np.asarray(xs, float)
    keep = np.ones(len(ys), bool)
    m = c = 0.0
    for _ in range(iters):
        A = np.vstack([ys[keep], np.ones(int(keep.sum()))]).T
        m, c = np.linalg.lstsq(A, xs[keep], rcond=None)[0]
        r = np.abs(xs - (m * ys + c))
        keep = r < tol
        if keep.sum() < 8:
            break
    return float(m), float(c), keep


def axis_lines(ink: np.ndarray, region):
    """The left vertical and bottom horizontal axis, each as a fitted LINE.

    ⚠ FITTING A LINE RATHER THAN TAKING A COLUMN IS THE WHOLE POINT. Measured on
    Fig. 10: the y axis sits at x = 146 at the top of the plot and x = 155 at the
    bottom -- the scan is rotated about 1.4 deg. Reading "the darkest column" gives
    a single number for a line that moves by 9 px, i.e. 0.05 D of density error
    that no residual would reveal, because nothing else in the calibration would
    disagree with it.
    """
    x0, y0, x1, y1 = region
    sub = ink[y0:y1, x0:x1]
    # ⚠ FIND THE AXIS BY INK MASS FIRST, THEN REFINE PER ROW. An earlier version
    # took "the leftmost ink in the left third of the region" per row and fitted
    # that: on Fig. 10 the leftmost ink is the printed axis LABEL ("1.5" sits 15 px
    # left of the axis), so the fit chased text, kept 6 rows out of 460 and put the
    # tilt at 2.07 deg instead of the true 1.4. The column of maximum ink mass IS
    # the axis on every one of these figures -- it is the only near-full-height
    # vertical stroke inside the box -- and refining within +/-10 px of it then
    # measures the skew without ever seeing the labels.
    # ⚠ LEFTMOST AND LOWEST STRONG LINE, not the strongest one. Taking the column
    # of maximum ink mass picked the figure's RIGHT frame edge on Fig. 10 (745 px,
    # more ink than the left axis) and the row of maximum mass picked the rule
    # ABOVE the figure. Which axis is which is decided by POSITION -- the density
    # axis is the leftmost near-full-height vertical stroke, the exposure axis the
    # lowest near-full-width horizontal one -- and "strong" is 35 % of the region
    # extent, which every real axis clears and no curve or label does.
    # ⚠ THE AXIS IS THE LEFTMOST STRAIGHT LINE, NOT THE LEFTMOST VERTICAL INK.
    # Two earlier rules failed measurably on Fig. 10: "column of maximum ink mass"
    # picked the figure's RIGHT frame (745 px, straighter and denser than the
    # tilted left axis), and "leftmost column with ink down most of the height"
    # picked the rotated "DENSITY" caption at x ~ 100. What distinguishes an axis
    # from a caption is STRAIGHTNESS: candidates are scanned left to right and the
    # first one whose per-row ink positions fit a line to better than 1.5 px over
    # at least 60 % of the rows is the axis. The tilt is then a fitted output, not
    # an assumption.
    def straight_vertical(cx):
        rows_, xs_ = [], []
        for y in range(y0, y1):
            seg = np.where(ink[y, max(0, x0 + cx - 8):x0 + cx + 9])[0]
            if seg.size:
                first = seg[0]
                cl = [v for v in seg if v - first <= 3]
                rows_.append(y)
                xs_.append(max(0, x0 + cx - 8) + float(np.mean(cl)))
        if len(rows_) < 0.5 * (y1 - y0):
            return None
        m, c, keep = fit_line(np.array(rows_), np.array(xs_))
        if keep.sum() < 0.60 * (y1 - y0):
            return None
        return m, c, int(keep.sum())

    vhit = np.zeros(sub.shape[1], dtype=np.int32)
    for y in range(sub.shape[0]):
        xs_ = np.where(sub[y])[0]
        for x in xs_:
            lo, hi = max(0, x - 6), min(sub.shape[1], x + 7)
            vhit[lo:hi] += 1
    left_axis = None
    for cx in np.where(vhit > 0.60 * sub.shape[0])[0]:
        got = straight_vertical(int(cx))
        if got:
            left_axis = got
            break
    if left_axis is None:
        raise SystemExit(f"[!] no straight vertical axis in region {region}")
    # The horizontal axis needs the SAME straightness test, and for the same
    # measured reason: scanning bottom-up for "the lowest band with ink across the
    # width" locked onto the row of printed x-axis LABELS ("1.00 2.00 3.00 4.00"),
    # 15-20 px below the real axis, which pulled the bottom of the trace box past
    # the axis line and handed the tracer the axis itself as a curve.
    def straight_horizontal(cy):
        cols_, ys_ = [], []
        for x in range(x0, x1):
            seg = np.where(ink[max(0, y0 + cy - 8):y0 + cy + 9, x])[0]
            if seg.size:
                last = seg[-1]
                cl = [v for v in seg if last - v <= 3]
                cols_.append(x)
                ys_.append(max(0, y0 + cy - 8) + float(np.mean(cl)))
        if len(cols_) < 0.5 * (x1 - x0):
            return None
        m, c, keep = fit_line(np.array(cols_), np.array(ys_))
        if keep.sum() < 0.60 * (x1 - x0):
            return None
        return m, c, int(keep.sum())

    hhit = np.zeros(sub.shape[0], dtype=np.int32)
    for x in range(sub.shape[1]):
        ys_ = np.where(sub[:, x])[0]
        for y in ys_:
            lo, hi = max(0, y - 6), min(sub.shape[0], y + 7)
            hhit[lo:hi] += 1
    straights = []
    for cy in np.where(hhit > 0.60 * sub.shape[1])[0]:
        got = straight_horizontal(int(cy))
        if got:
            straights.append((int(cy), got))
    if not straights:
        raise SystemExit(f"[!] no straight horizontal axis in region {region}")
    # dedupe bands that describe the same line (within 12 px)
    uniq = []
    for cy, got in straights:
        if not uniq or cy - uniq[-1][0] > 12:
            uniq.append((cy, got))
    bottom_axis = uniq[-1][1]
    # A rule in the TOP quarter of the region is the figure's top frame, and the
    # trace box must start below it or the tracer adopts it as a curve.
    top_rule = None
    for cy, got in uniq[:-1]:
        if cy < 0.25 * sub.shape[0]:
            top_rule = got
    mv, cv, nv = left_axis
    mh, ch, nh = bottom_axis
    return (mv, cv, nv), (mh, ch, nh), top_rule


def stub_ticks(ink, axis, region, along="y", stub=(2, 14), clear=(30, 100),
               min_ink=5):
    """Tick marks as SHORT STUBS attached to an axis, returned as pixel centres.

    A tick and a curve both put ink next to the axis; what separates them is that
    a curve keeps going. So a stub is ink within `stub` px of the axis AND almost
    no ink in the `clear` band beyond it. Measured on Fig. 10: ticks are 7-9 px
    long and 2 rows tall, curves span 590 px, and this test picks up 3 of the 6
    printed density ticks -- the 0 tick is buried in the bottom axis line and the
    1.0 tick sits 17 px from the B plateau, whose ink fills the clear band. Both
    rejections are correct behaviour: an ambiguous tick is not used.
    """
    x0, y0, x1, y1 = region
    (mv, cv, _), (mh, ch, _) = axis
    hits = []
    if along == "y":
        for y in range(y0, y1):
            ax = mv * y + cv
            a, b = int(round(ax)) + stub[0], int(round(ax)) + stub[1]
            c0, c1 = int(round(ax)) + clear[0], int(round(ax)) + clear[1]
            if ink[y, a:b].sum() >= min_ink and ink[y, c0:c1].sum() <= 2:
                hits.append(y)
    else:
        for x in range(x0, x1):
            ay = mh * x + ch
            a, b = int(round(ay)) - stub[1], int(round(ay)) - stub[0]
            c0, c1 = int(round(ay)) - clear[1], int(round(ay)) - clear[0]
            if ink[a:b, x].sum() >= min_ink and ink[c0:c1, x].sum() <= 2:
                hits.append(x)
    groups = []
    for v in hits:
        if groups and v - groups[-1][-1] <= 5:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups if len(g) >= 2]


def verify_anchor(ink, axis, pixel, along, region):
    """Re-check that a pinned tick anchor still lands on an ink stub.

    ⚠ THIS IS THE PRICE OF PINNING PIXEL COORDINATES. A frozen anchor is only
    honest if it is checked: if the source file is ever replaced by a different
    scan of the same page -- a re-scan, a different crop, a deskewed export -- the
    numbers would still "work" and would silently describe the wrong rows. So each
    anchor must still have ink within a few pixels of the axis at that position,
    and the check runs on every extraction, not once.
    """
    x0, y0, x1, y1 = region
    (mv, cv, _), (mh, ch, _) = axis
    if along == "y":
        ax = int(round(mv * pixel + cv))
        band = ink[int(round(pixel)) - 2:int(round(pixel)) + 3, ax + 1:ax + 15]
    else:
        ay = int(round(mh * pixel + ch))
        band = ink[ay - 15:ay - 1, int(round(pixel)) - 2:int(round(pixel)) + 3]
    return int(band.sum())


def calibrate(pixels, values, what):
    """value = m*pixel + c by least squares, with the residual reported."""
    if len(pixels) < 2:
        raise SystemExit(f"[!] {what}: only {len(pixels)} ticks detected")
    p = np.asarray(pixels, float)
    v = np.asarray(values, float)
    A = np.vstack([p, np.ones(len(p))]).T
    m, c = np.linalg.lstsq(A, v, rcond=None)[0]
    res = v - (m * p + c)
    return float(m), float(c), float(np.max(np.abs(res)))


def trace_three(gray, ink, region, axis, seed_gap, top_rule=None):
    """The three layer curves, seeded where they are separated and non-crossing.

    ⚠ NO STYLE SEPARATION IS USED HERE, unlike the VISION3 granularity sheets, and
    the reason is measured rather than assumed: on these figures the three curves
    NEVER CROSS -- B stays above G stays above R across the full width, because
    they are the three records of one masked negative and the mask sets their
    order. Style separation exists to stop a trace migrating between two families
    that DO cross. Where nothing crosses, seeding at the left plateau (where the
    three are 25+ px apart) and tracing rightward is both simpler and stricter,
    and `dt.check_ordering` asserts the order held.
    """
    x0, y0, x1, y1 = region
    (mv, cv, _), (mh, ch, _) = axis
    left = int(round(mv * ((y0 + y1) / 2) + cv))
    bottom = int(round(mh * ((x0 + x1) / 2) + ch))
    top = y0
    if top_rule is not None:
        mt, ct, _ = top_rule
        top = int(round(mt * ((x0 + x1) / 2) + ct)) + 4
    box = np.zeros_like(ink)
    box[top:bottom - 3, left + 4:x1 - 2] = ink[top:bottom - 3, left + 4:x1 - 2]
    seed = None
    for x in range(left + 8, left + int(0.30 * (x1 - left))):
        cs = sorted(c for c, _t in dt.column_runs_weighted(box, gray, x, top, bottom - 3))
        if len(cs) == 3 and min(np.diff(cs)) > seed_gap:
            seed = (x, cs)
            break
    if seed is None:
        raise SystemExit("[!] no seed column with three separated curves")
    names = ("B", "G", "R")
    fwd = dt.trace_predictive(box, gray, (left + 4, x1 - 3), top, bottom - 3,
                              seed[0], dict(zip(names, seed[1])), direction=+1,
                              tol0=2.6, tol_grow=0.4, max_bridge=34, hist=16,
                              slope_cap=2.5)
    rev = dt.trace_predictive(box, gray, (left + 4, x1 - 3), top, bottom - 3,
                              seed[0], dict(zip(names, seed[1])), direction=-1,
                              tol0=2.2, tol_grow=0.25, max_bridge=12, hist=10,
                              slope_cap=0.4)
    return {k: {**rev[k], **fwd[k]} for k in names}, seed


#: The stored dye-density grid. `SpectralDyeDensity` defaults to 400 nm / 10 nm
#: and every adopted set in the database uses 31 samples to 700 nm.
DYE_GRID = np.arange(400.0, 701.0, 10.0)


def fig8_ticks(ink, spec):
    """Re-detect the pinned tick pixels and report how far each one moved.

    Same contract as `verify_anchor` on the characteristic figures: the anchors
    are pinned so the calibration is reproducible, and re-detected every run so a
    pinned anchor that no longer sits on ink fails loudly instead of quietly
    rescaling a curve. Returns (worst wavelength drift, worst density drift) in
    pixels, or raises if a tick cannot be found at all.
    """
    x0, y0, x1, y1 = spec["frame"]

    def _groups(mask, axis):
        s = mask.sum(axis)
        idx = np.where(s >= 4)[0]
        out, cur = [], []
        for i in idx:
            if cur and i - cur[-1] <= 3:
                cur.append(i)
            else:
                if cur:
                    out.append(float(np.mean(cur)))
                cur = [i]
        if cur:
            out.append(float(np.mean(cur)))
        return out

    lam_found = _groups(ink[y0 - 3:y0 + 10, :], 0)
    d_found = _groups(ink[:, x1 - 12:x1 + 4], 1) + _groups(ink[:, x0 - 3:x0 + 12], 1)
    worst_lam = worst_d = 0.0
    for val, px in spec["lam_ticks"].items():
        if not lam_found:
            raise SystemExit("[!] fig8: no wavelength ticks detected at all")
        near = min(lam_found, key=lambda f: abs(f - px))
        if abs(near - px) > 4.0:
            raise SystemExit(f"[!] fig8: pinned wavelength tick {val} nm at col "
                             f"{px} has no ink within 4 px (nearest {near:.1f})")
        worst_lam = max(worst_lam, abs(near - px))
    for val, px in spec["d_ticks"].items():
        near = min(d_found, key=lambda f: abs(f - px))
        if abs(near - px) > 4.0:
            raise SystemExit(f"[!] fig8: pinned density tick {val} at row {px} "
                             f"has no ink within 4 px (nearest {near:.1f})")
        worst_d = max(worst_d, abs(near - px))
    return worst_lam, worst_d


def fig8_label_glyphs(ink, spec, lam_of):
    """Wavelengths of the printed C / M / Y letters, ascending.

    ⚠ THIS IS AN INDEPENDENT CHECK AND NOT AN OCR. Which letter is which is never
    read; only WHERE the three letters are. The figure prints one letter above
    each peak, so three glyphs at three wavelengths that coincide with the three
    traced peaks is a statement by the paper that those peaks are the three dyes
    -- in the same class as the in-frame captions on the Kodak 7239 panel, and
    for the same reason: it is the source's own words about its own curves.
    """
    x0, y0, x1, y1 = spec["frame"]
    band = np.zeros_like(ink)
    band[y0 + 68:y0 + 103, x0:x1] = ink[y0 + 68:y0 + 103, x0:x1]
    lab, info = dt._components(band)
    out = []
    for i, (w, h, _c) in info.items():
        if w < 30 and h >= 8:
            ys, xs = np.where(lab == i)
            out.append(lam_of(float(xs.mean())))
    return sorted(out)


def extract_fig8(gray, spec):
    """The three dye-density curves of Gevacolor 682, on `DYE_GRID`.

    Returns (curves, diagnostics). `curves` is {"c"/"m"/"y": array on DYE_GRID}.

    ⚠ THE LAYER NAMES COME FROM THE PEAKS, NOT FROM THE SEED ORDER. The three
    tracks are seeded left to right by position and are then named by which
    absorption band their peak falls in -- cyan reddest, yellow bluest. That is
    the same rule the vector readers use, it is checked against the paper's own
    printed peak wavelengths, and it is checked again against the position of the
    printed C / M / Y letters. Naming by seed order would have been wrong here:
    at the left edge the CYAN curve is the top one and the MAGENTA the bottom.
    """
    region = spec["region"]
    sub = gray[region[1]:region[3], region[0]:region[2]]
    ink = sub < DARK
    x0, y0, x1, y1 = spec["frame"]

    drift_lam, drift_d = fig8_ticks(ink, spec)
    lm, lc, lres = calibrate(list(spec["lam_ticks"].values()),
                             list(spec["lam_ticks"].keys()), "wavelength axis")
    dm, dc, dres = calibrate(list(spec["d_ticks"].values()),
                             list(spec["d_ticks"].keys()), "density axis")
    lam_of = lambda px: lm * px + lc          # noqa: E731
    d_of = lambda px: dm * px + dc            # noqa: E731

    seed_x = spec["seed_x"]
    runs = sorted(c for c, _t in
                  dt.column_runs_weighted(ink, sub, seed_x, y0, y1 + 3))
    if len(runs) != 3:
        raise SystemExit(f"[!] fig8: the seed column {seed_x} shows {len(runs)} "
                         f"ink runs, not 3")
    tracks = dt.trace_predictive(ink, sub, (seed_x, x1 + 3), y0, y1 + 3, seed_x,
                                 {"t0": runs[0], "t1": runs[1], "t2": runs[2]},
                                 direction=+1, tol0=3.5, tol_grow=0.9,
                                 max_bridge=34, hist=16, slope_cap=2.5,
                                 merge_px=spec["merge_px"])
    traced = {}
    for k, t in tracks.items():
        if not t:
            raise SystemExit(f"[!] fig8: track {k} died at the seed")
        px = np.array(sorted(t), float)
        traced[k] = (lam_of(px), d_of(np.array([t[q] for q in px], float)))
    order = sorted(traced, key=lambda k: traced[k][0][int(np.argmax(traced[k][1]))])
    names = dict(zip(order, ("y", "m", "c")))     # ascending peak wavelength

    curves, peaks = {}, {}
    for k, (lam, d) in traced.items():
        n = names[k]
        i = int(np.argmax(d))
        peaks[n] = (float(lam[i]), float(d[i]))
        # ⚠ OUTSIDE THE TRACED SPAN, HOLD AT ZERO -- AND ONLY THE YELLOW NEEDS IT.
        # The yellow curve reaches the axis at about 583 nm and is thereafter
        # indistinguishable from the axis line itself, so the trace stops at 572.
        # The figure does show what happens next -- the curve runs along zero --
        # so 0.0 is a reading of the plot rather than an extrapolation of the
        # trace. Beyond 699 nm all three are extended by 1 nm to reach the grid's
        # last sample, which is inside the line width.
        v = np.interp(DYE_GRID, lam, d, left=float(d[0]), right=0.0)
        v[DYE_GRID > lam[-1] + 2.0] = 0.0
        if DYE_GRID[-1] - lam[-1] <= 2.0:
            v[-1] = float(d[-1])
        curves[n] = np.clip(v, 0.0, None)

    labels = fig8_label_glyphs(ink, spec, lam_of)
    diag = dict(lam_resid=lres, d_resid=dres, drift_lam=drift_lam,
                drift_d=drift_d, peaks=peaks, labels=labels,
                zero_row=float((0.0 - dc) / dm),
                n=(len(traced["t0"][0]), len(traced["t1"][0]),
                   len(traced["t2"][0])))
    return curves, diag


def fit_layer(x, d, *_ignored):
    """5-parameter fit with dmin PINNED to the measured plateau. See module head.

    ⚠ THE INITIAL GUESS MUST NOT COME FROM THE DATABASE, and the first version of
    this function proved why by breaking. It seeded Nelder-Mead with the stored
    profile's parameters -- which, once this extractor's own output had been
    adopted, meant seeding the fit with the answer it had previously produced. The
    red record (drawn DOTTED, so the sparsest trace) then jumped to a DIFFERENT
    local minimum on the very next run: gamma 0.5446 at rms 0.0109 D became gamma
    0.5059 at rms 0.0055 D, and the audit failed against its own pin. Neither
    number was wrong arithmetically; the procedure was not reproducible, which is
    worse than either.
    So the start points are now DERIVED FROM THE DATA and fixed:
      gamma_0    the steepest slope measured over a 1.0-decade window
      toe_x_0    where the curve first exceeds dmin + 0.10
      sh_x_0     where it last sits below (max - 0.10)
    and the optimiser is run from a small deterministic grid of softness values,
    keeping the lowest-loss solution. Multi-start rather than one start because
    a softplus toe traded against a shoulder has several minima; deterministic
    rather than random because an audit that cannot be re-run identically is not
    an audit.
    """
    plateau = d[x < (x.min() + 0.8)]
    dmin = float(np.median(plateau))

    def loss(p):
        gam, tx, tk, sx, sk = p
        if gam <= 0 or tk <= 0.02 or sk <= 0.02 or sx <= tx:
            return 1e9
        pen = 100.0 * max(0.0, sk - 1.4 * tk) ** 2
        r = dp.softplus_curve(x, dmin, gam, tx, tk, sx, sk) - d
        return float(np.mean(r * r)) + pen

    # data-derived starting geometry
    g0 = 0.5
    for i in range(len(x)):
        j = int(np.searchsorted(x, x[i] + 1.0))
        if j < len(x):
            g0 = max(g0, abs((d[j] - d[i]) / (x[j] - x[i])))
    hi = float(d.max())
    above = np.where(d > dmin + 0.10)[0]
    below = np.where(d < hi - 0.10)[0]
    tx0 = float(x[above[0]]) if above.size else float(x.min())
    sx0 = float(x[below[-1]]) if below.size else float(x.max())
    best = None
    for tk0 in (0.12, 0.20, 0.30):
        for sk0 in (0.16, 0.24, 0.36):
            p, v = dp._nelder_mead(loss, np.array([g0, tx0, tk0, sx0, sk0]),
                                   [0.03, 0.08, 0.04, 0.08, 0.04])
            if best is None or v < best[1]:
                best = (p, v)
    p = best[0]
    r = dp.softplus_curve(x, dmin, *p) - d
    return ((dmin,) + tuple(float(v) for v in p),
            float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r))), len(plateau))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="../..")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--overlay", metavar="PNG")
    ap.add_argument("--dump", action="store_true",
                    help="print the dye arrays in film_profiles.py form")
    ns = ap.parse_args()

    root = Path(ns.root).resolve() / "PDF" / "PROFILES"
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cache: dict[str, list[np.ndarray]] = {}
        for tag, spec in FIGURES.items():
            pdf = root / DOCS[spec["doc"]][0]
            if not pdf.is_file():
                print(f"  [SKIP] {tag}: source not present: {pdf.name}")
                continue
            if spec["doc"] not in cache:
                cache[spec["doc"]] = native_pages(pdf, tmp, spec["doc"])
            gray = cache[spec["doc"]][spec["page"]]
            ink = gray < DARK
            region = spec["region"]
            print(f"[i] {tag}: {spec['title']}, printed p{spec['printed_page']}, "
                  f"native scan {gray.shape[1]}x{gray.shape[0]} px")

            axis3 = axis_lines(ink, region)
            axis = (axis3[0], axis3[1])
            top_rule = axis3[2]
            (mv, cv, nv), (mh, ch, nh) = axis
            print(f"    axes fitted: x = {mv:+.5f}*y {cv:+.1f} ({nv} rows kept, "
                  f"tilt {np.degrees(np.arctan(mv)):+.2f} deg); "
                  f"y = {mh:+.5f}*x {ch:+.1f} ({nh} cols kept)")

            weak = []
            for val, px in spec["y_ticks"].items():
                if verify_anchor(ink, axis, px, "y", region) < 4:
                    weak.append(f"density {val} at row {px}")
            for val, px in spec["x_ticks"].items():
                if verify_anchor(ink, axis, px, "x", region) < 4:
                    weak.append(f"logE {val} at col {px}")
            if weak:
                print(f"    [FAIL] pinned tick anchors no longer sit on ink: "
                      f"{'; '.join(weak)}")
                bad += 1
                continue
            dm, dc, dres = calibrate(list(spec["y_ticks"].values()),
                                     list(spec["y_ticks"].keys()), "density axis")
            xm, xc, xres = calibrate(list(spec["x_ticks"].values()),
                                     list(spec["x_ticks"].keys()), "exposure axis")
            print(f"    density  {len(spec['y_ticks'])} anchors re-verified, "
                  f"{abs(1/dm):.1f} px per D, worst residual {dres:.4f} D")
            print(f"    exposure {len(spec['x_ticks'])} anchors re-verified, "
                  f"{abs(1/xm):.1f} px per decade, worst residual {xres:.4f} decade")

            tracks, seed = trace_three(gray, ink, region, axis, spec["seed_gap"],
                                       top_rule)
            dt.check_ordering(tracks, ("B", "G", "R"))
            data = {}
            for k in ("B", "G", "R"):
                xs = np.array(sorted(tracks[k]), float)
                ys = np.array([tracks[k][x] for x in xs], float)
                data[k] = (xm * xs + xc, dm * ys + dc)
                print(f"    {k}: {len(xs)} samples (one per pixel column), "
                      f"logE {data[k][0].min():+.2f}..{data[k][0].max():+.2f}, "
                      f"D {data[k][1].min():.3f}..{data[k][1].max():.3f}")

            # ⚠ PLACE THE ABSCISSA BY INHERITANCE, AND SAY SO. The figure's axis is
            # "LOG REL. EXP." with no absolute anchor -- no speed point, no
            # lux-seconds, nothing that ties its 0 to an exposure. The database's
            # tone-curve x has its origin at the mid-grey exposure. Nothing printed
            # in the paper relates the two, so the SHAPE is measured and the
            # PLACEMENT is inherited: the offset is the traced log exposure at
            # which the green record reaches the same NET density above dmin that
            # the existing profile already produced at x = 0. All three layers are
            # shifted by that one offset, because they share one exposure axis --
            # shifting them independently would invent a speed difference the
            # figure does not show.
            target_net = float(spec["origin_net_density"])
            le_g, d_g = data["G"]
            dmin_g = float(np.median(d_g[le_g < le_g.min() + 0.8]))
            off = float(le_g[int(np.argmin(np.abs((d_g - dmin_g) - target_net)))])
            print(f"    abscissa origin INHERITED: project x = traced logE "
                  f"- {off:.3f} (green reaches net {target_net:.3f} D there, "
                  f"which is what the stored curve gave at x = 0)")

            want = EXPECTED.get(tag)
            fits = {}
            for i, k in enumerate(("B", "G", "R")):
                le, d = data[k]
                le = le - off
                p, rms, mx, npl = fit_layer(le, d)
                fits[k] = (p, rms, mx)
                print(f"    {k}: dmin {p[0]:.4f} (plateau median, {npl} samples)  "
                      f"gamma {p[1]:.4f}  toe_x {p[2]:+.4f} toe_k {p[3]:.4f}  "
                      f"sh_x {p[4]:+.4f} sh_k {p[5]:.4f}  | fit rms {rms:.4f} D, "
                      f"worst {mx:.4f} D")
                if want:
                    if abs(p[0] - want["dmin"][i]) > want["tol"]:
                        print(f"    [FAIL] {k} dmin moved: {p[0]:.4f} vs recorded "
                              f"{want['dmin'][i]:.4f}")
                        bad += 1
                    if abs(p[1] - want["gamma"][i]) > want["tol"]:
                        print(f"    [FAIL] {k} gamma moved: {p[1]:.4f} vs recorded "
                              f"{want['gamma'][i]:.4f}")
                        bad += 1
                    if rms > want["rms_max"]:
                        print(f"    [FAIL] {k} fit rms {rms:.4f} exceeds "
                              f"{want['rms_max']:.4f}")
                        bad += 1
            # ⚠ THE EXTERNAL CHECK. The paper prints its own gamma on this figure;
            # the trace must reproduce it from tick positions alone.
            if want and "printed_gamma" in want:
                k = ("B", "G", "R")[want["printed_gamma_layer"]]
                got = fits[k][0][1]
                ok = abs(got - want["printed_gamma"]) <= 0.02
                print(f"    {'[OK]' if ok else '[FAIL]'} traced {k} gamma "
                      f"{got:.4f} vs the figure's printed gamma "
                      f"{want['printed_gamma']:.2f}")
                if not ok:
                    bad += 1

            if ns.overlay:
                out = (ns.overlay if len(FIGURES) == 1
                       else ns.overlay.replace(".png", f"_{tag}.png"))
                rgb = np.repeat((gray[region[1]:region[3], region[0]:region[2]]
                                 * 255).astype(np.uint8)[:, :, None], 3, axis=2)
                cols = {"B": (30, 60, 230), "G": (20, 160, 40), "R": (220, 30, 30)}
                for k in ("B", "G", "R"):
                    for x, y in tracks[k].items():
                        xi, yi = int(x) - region[0], int(round(y)) - region[1]
                        if 0 <= yi < rgb.shape[0] and 0 <= xi < rgb.shape[1]:
                            rgb[max(0, yi-1):yi+2, max(0, xi-1):xi+2] = cols[k]
                Image.fromarray(rgb).save(out)
                print(f"    overlay written to {out} (B blue, G green, R red)")

        # ---- the dye-density figures ------------------------------------
        for tag, spec in DYE_FIGURES.items():
            pdf = root / DOCS[spec["doc"]][0]
            if not pdf.is_file():
                print(f"  [SKIP] {tag}: source not present: {pdf.name}")
                continue
            if spec["doc"] not in cache:
                cache[spec["doc"]] = native_pages(pdf, tmp, spec["doc"])
            gray = cache[spec["doc"]][spec["page"]]
            print(f"[i] {tag}: {spec['title']}, printed p{spec['printed_page']}")
            curves, diag = extract_fig8(gray, spec)
            print(f"    wavelength {len(spec['lam_ticks'])} anchors re-verified "
                  f"(worst drift {diag['drift_lam']:.1f} px), fit residual "
                  f"{diag['lam_resid']:.2f} nm")
            print(f"    density    {len(spec['d_ticks'])} anchors re-verified "
                  f"(worst drift {diag['drift_d']:.1f} px), fit residual "
                  f"{diag['d_resid']:.4f} D; zero falls on row "
                  f"{diag['zero_row']:.1f}")
            print(f"    samples per track: {diag['n']} (one per pixel column)")
            want = EXPECTED_DYE.get(tag)
            for i, n in enumerate(("c", "m", "y")):
                lam, d = diag["peaks"][n]
                print(f"    {n}: peak {d:.3f} D at {lam:.1f} nm", end="")
                if want:
                    plam, pd = want["printed_nm"][i], want["printed_d"][i]
                    ok = (abs(lam - plam) <= want["tol_nm"]
                          and abs(d - pd) <= want["tol_d"])
                    print(f"   vs the paper's printed {pd:.2f} D at "
                          f"{plam:.0f} nm -- {'OK' if ok else 'FAIL'}")
                    if not ok:
                        bad += 1
                    if abs(lam - want["peaks_nm"][i]) > 1.0 or \
                       abs(d - want["peaks_d"][i]) > 0.01:
                        print(f"    [FAIL] {n} moved from the recorded "
                              f"{want['peaks_d'][i]:.3f} D at "
                              f"{want['peaks_nm'][i]:.1f} nm")
                        bad += 1
                else:
                    print()
            if want:
                got = diag["labels"]
                exp = sorted(want["labels_nm"])
                ok = (len(got) == 3 and
                      all(abs(a - b) <= want["tol_label_nm"]
                          for a, b in zip(got, exp)))
                print(f"    the printed C/M/Y letters sit at "
                      f"{' / '.join(f'{v:.0f}' for v in got)} nm -- "
                      f"{'OK' if ok else 'FAIL'}, one above each traced peak")
                if not ok:
                    bad += 1
            if ns.dump:
                for n, key in (("c", "d_cyan"), ("m", "d_magenta"),
                               ("y", "d_yellow")):
                    vals = ", ".join(f"{v:.3f}" for v in curves[n])
                    print(f"            {key}=({vals}),")

    print()
    if bad:
        print(f"[FAIL] {bad} problem(s)")
        return 1 if ns.do_assert else 0
    print("[OK] Gevaert curves reproduced from the native-resolution scans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
