#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""«Kodak Films», Seventh Edition 1956 -- the THIRTY WEDGE SPECTROGRAMS, read.

Queue P62. This is the only spectral data this corpus holds for the 1950s
Kodak black-and-white line, and it is the first reader in the project whose
measurement is the POSITION OF A DENSITY BOUNDARY IN A SCREENED GREY FIELD
rather than the centre of a drawn stroke.

WHAT A WEDGE SPECTROGRAM IS, IN KODAK'S OWN WORDS
--------------------------------------------------
Page 11 of the book states the whole instrument, and the reader is built on
that sentence and on nothing inferred:

    "These spectrograms are positive prints from films which have been exposed
     to a light spectrum through a neutral density wedge. This wedge is opaque
     at the top, decreasing in density or opacity until it is transparent at
     the bottom. As the transmitted light diminishes toward the top, the
     height of the film image at any point is an indication of the film's
     response to that particular wavelength."

Three consequences, all of which the reader depends on:

* The quantity is a HEIGHT measured from the wedge's transparent end, which is
  one horizontal line common to every wavelength -- so the baseline is a
  property of the plate and is measured, not assumed.
* Height is linear in LOG EXPOSURE, because a neutral wedge is linear in
  density. The shape therefore comes out in log units directly.
* The same page says the plates "show relative color sensitivity only, and
  give no indication of film speed", and warns that "due to the absorption of
  much of the ultraviolet by the lens system of the spectrograph, the
  indicated ultraviolet sensitivity of all films ... is lower than the true
  value". Both statements are carried into `criterion` rather than quietly
  dropped.

⚠⚠ WHAT KODAK DOES **NOT** PRINT IS THE WEDGE GRADIENT, AND WITHOUT IT THE
VERTICAL AXIS HAS NO SCALE. A shape known only up to a multiplicative constant
on the log axis is not a spectral sensitivity: "one log unit down at 650 nm"
and "three log units down at 650 nm" are the same picture. The peak
normalisation `SpectralSensitivity` requires does not remove that constant --
it removes the offset.

THE SCALE IS RECOVERED FROM A PUBLISHED FUNCTION PRINTED ON THE SAME WEDGE
---------------------------------------------------------------------------
Page 12 prints four spectrograms in one stack, to the same wedge and the same
spectrograph, labelled by Kodak: SENSITIVITY OF THE EYE, NON-COLOR-SENSITIZED,
ORTHOCHROMATIC, PANCHROMATIC. The first of them is a plot of a function that
is published elsewhere and exactly -- the CIE 1924 photopic luminous
efficiency V(lambda). Fitting its traced height against log10 V(lambda) over
410-690 nm gives

    59.5 px per log10 unit at 600 dpi,  r = 0.979,  rms 9.6 px (0.16 log)

which is the wedge gradient, in the only units the page can express it in.
Expressed scale-free as `LOG_PER_SPAN` -- the 400-to-600 nm axis length
divided by the pixels per log unit -- it is 12.79, and it then transfers to a
plate reproduced at any size.

⚠ THE TRANSFER IS CHECKED AND IT IS THE STRONGEST EVIDENCE HERE. If one wedge
made all thirty plates, then every plate's own HEIGHT, converted through its
own wavelength axis, must come out the same number of log units -- and it
does: 4.98 to 5.13 across fifteen pages and two page layouts, against 5.07 for
the page-12 reference. Nothing was fitted to make that happen; the heights and
the wavelength axes are independent measurements on unrelated pages.

⚠ AND THE UNCERTAINTY IS DECLARED RATHER THAN HIDDEN. The V(lambda) slope
depends on the window it is fitted over -- 51.3 px/log on 460-640 nm, 59.5 on
410-690 -- so the log DEPTHS carry roughly +/-10 %. Kodak also calls its own
eye curve "approximate". The wavelength axis is not affected by any of this.

THE WAVELENGTH AXIS
-------------------
Each data-sheet plate carries THREE long ticks in the white margin below it,
which the page labels in words: U.V. | BLUE | GREEN | RED. They are the
400 / 500 / 600 nm boundaries, and the page-12 reference proves it by printing
FOUR ticks with "400 m-mu", "500 m-mu", "600 m-mu", "700 m-mu" set under them.

Two checks make the identification a fact about the page rather than a habit:

* `(t500 - t400) / (t600 - t400)` is 0.5010 on the reference plate and lands
  in 0.4964-0.5078 on all thirty data-sheet plates. A misidentified tick
  cannot produce that.
* The dispersion is NOT linear. On the reference plate 400->500 spans 381.5 px
  and 600->700 spans 369.5, a 3 % compression toward the red that a straight
  line gets wrong by 2 nm at 700. The law is therefore taken as a quadratic in
  the normalised coordinate u = (x - x400) / (x600 - x400) from the reference
  plate's four ticks, and applied to every plate through its own two anchors.

THE VALIDATION QUEUE P62 DEMANDED
----------------------------------
The row's blocker: *"a boundary-finding reader with its own validation -- it
must reproduce the known long-wavelength cutoff of a panchromatic sheet versus
an orthochromatic one before any of its numbers are believed."* The book
supplies a three-way test rather than a two-way one, because page 12 prints
the canonical non-colour-sensitized, orthochromatic and panchromatic plates
side by side, and pages 65 and 67 print a blue-sensitive emulsion (Commercial)
directly above its own orthochromatic re-sensitizing (Commercial Ortho).

    NON-COLOR-SENSITIZED   380-490 nm, peak 447
    ORTHOCHROMATIC         380-587 nm, peak 565
    PANCHROMATIC           382-655 nm, peak 585
    EYE                    403-698 nm, peak 551   (V(lambda) peaks at 555)

⚠ NOTHING IS ADOPTED UNTIL ALL FOUR OF THOSE ARE REPRODUCED, and the eye
curve's 4 nm peak error is a check on the wavelength axis that costs nothing
and that a two-way cutoff test does not give.

WHAT IS NOT CLAIMED
-------------------
* No absolute sensitivity, and none is derivable: Kodak says so on page 11.
* Below 400 nm the values are Kodak's own understatement, by their warning.
* Beyond a plate's cutoff the boundary reaches the wedge's transparent end and
  the sensitivity is off-scale below it. The stored value there is the schema
  sentinel -4.0, meaning "at or below the floor of the source plot"; the REAL
  floor is each plate's own peak height, 2.0 to 2.8 log below the peak, and is
  reported per plate by `main()` rather than being silently implied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PDF = Path("/root/work/pg/PDF/PROFILES/KODAK/1956-Kodak-Films.pdf")
ALT_PDF = Path("/mnt/user-data/uploads/PYTHON.TST/PDF/PROFILES/KODAK/"
               "1956-Kodak-Films.pdf")

DPI = 600                 #: the halftone screen is ~110 lpi; 600 dpi resolves it.
INK = 150                 #: 8-bit threshold for "plate black" / "tick ink".
BLUR = 4.0                #: sigma, px. Kills the halftone dots, keeps the edge.
EDGE_FRAC = 0.30          #: boundary is bg + this fraction of (peak - bg).
MIN_RUN = 10              #: px of continuous light a column needs to count.
MIN_GROUP_PX = 20         #: ~8 nm. Narrower than this is a speck, not a lobe.
MIN_GROUP_PX_H = 12       #: and it must rise this far off the wedge's floor.
TICK_ROWS = 85            #: depth of the margin strip the ticks are sought in.
TICK_MIN = 45             #: inked rows a column needs to be a wavelength tick.

#: Reference plate, page 12 of the book (PDF page index 15). Kodak prints the
#: wavelength values under these four ticks in words, which is what makes the
#: three-tick data-sheet plates readable at all.
REF_PAGE = 15
REF_TICK_NM = (400.0, 500.0, 600.0, 700.0)
#: u = (x - x400) / (x600 - x400) for those four ticks on the reference plate,
#: measured at 600 dpi: 753.5, 1135.0, 1515.0, 1884.5.
REF_U = (0.0, 0.5010, 1.0, 1.4852)
#: Tolerance on (t500-t400)/(t600-t400) for a data-sheet plate. The reference
#: plate gives 0.5010 and the thirty plates span 0.4964-0.5078.
RATIO_NOMINAL = 0.5010
RATIO_TOL = 0.012

#: THE WEDGE GRADIENT, as the 400-600 nm axis length divided by the pixels per
#: log10 unit. Fitted on the page-12 eye plate against CIE 1924 V(lambda) over
#: 410-690 nm: 761.5 px of axis, 59.54 px per log unit. Dimensionless, so it
#: transfers to a plate printed at any reduction.
LOG_PER_SPAN = 12.79
#: The same fit run over narrower windows gives 51.3 (460-640) to 59.5
#: (410-690). Log DEPTHS therefore carry about this much relative uncertainty.
LOG_SCALE_UNCERTAINTY = 0.10

#: Full-scale log range every plate must come out at if one wedge made them
#: all. ⚠ THE BOUND IS NOT READ OFF THE PLATES. It is the page-12 reference
#: plate's own 5.07 log units widened by `LOG_SCALE_UNCERTAINTY`, which is the
#: uncertainty the V(lambda) fit already declares -- so the gate asks whether
#: the data-sheet plates agree with the reference wedge to within the error
#: that was admitted before they were measured, and it is not a tolerance
#: chosen to fit the answer. It earned its keep immediately: it caught page
#: 41's tungsten plate being measured 20 % short.
REF_FULL_SCALE_LOG = 5.07

#: ⚠⚠ A WEDGE SPECTROGRAM IS NOT A SPECTRAL SENSITIVITY, AND THIS IS THE STEP
#: THAT NEARLY WENT MISSING. The plate records the film's response to the
#: spectrograph's OWN light, so its height is log[S(lambda) x E(lambda)], and
#: the book prints each emulsion twice for exactly that reason -- "Spectrogram
#: to Sunlight" and "Spectrogram to Tungsten Light" are visibly different
#: pictures of one emulsion. `SpectralSensitivity` holds S, not S x E, so E
#: has to be divided out.
#:
#: Kodak does not publish either source's spectrum. Both are taken as Planck
#: radiators at the colour temperatures the rest of the book works in -- its
#: filter-factor tables are headed "Sunlight" and "Photoflood or
#: high-efficiency tungsten" -- and THE ASSUMPTION IS TESTED RATHER THAN
#: ASSERTED: the two plates of one emulsion are independent records of the
#: same S, so correcting them must bring them together. It does, from a mean
#: absolute disagreement of 0.425 log to 0.269, and the gate fails the build
#: if the correction ever stops helping. The residual 0.269 log is the honest
#: accuracy of this harvest and is stated in `source`.
#:
#: ⚠ The result is insensitive to the temperatures themselves -- 0.261 at
#: 6500/2850 K against 0.279 at 5000/2850 -- so the correction is carrying the
#: SHAPE of a blackbody ratio and not a fitted colour temperature.
SUN_K = 5500.0
TUNGSTEN_K = 2850.0
PAIR_RESIDUAL_MAX = 0.40     #: mean |sun - tungsten| after correction, log.
PAIR_RESIDUAL_WORST = 0.75   #: and the worst single emulsion.

#: Adopted sampling grid.
LAMBDA_START = 380.0
LAMBDA_STEP = 10.0
LAMBDA_N = 33             #: 380 .. 700 nm
FLOOR = -4.0              #: SpectralSensitivity's own off-scale sentinel.

SOURCE = ("Eastman Kodak Company, «Kodak Films», Seventh Edition, "
          "Rochester N.Y., 1956 -- wedge spectrograms, pages 12 and 37-70. "
          "Wavelength axis from the plates' own 400/500/600 nm boundary ticks, "
          "identified against the four labelled ticks of the page-12 reference "
          "plate. Vertical scale recovered from the page-12 "
          "SENSITIVITY OF THE EYE plate against CIE 1924 V(lambda) "
          "(59.5 px per log10 unit at 600 dpi, r=0.979); Kodak publishes no "
          "wedge gradient, so log depths carry about +/-10 %. Kodak states "
          "these plates give relative colour sensitivity only and warns that "
          "the indicated ultraviolet sensitivity is lower than the true value.")

CRITERION = "wedge_spectrogram_boundary_relative_log"

#: ⚠ TRANSCRIBED BY EYE, as every identification in this project is. The
#: geometry is all measured; this table says only WHICH FILM and WHICH
#: ILLUMINANT each plate belongs to, read off the page's own captions. Pages
#: 65 and 67 are the two that stack their plates vertically, and they carry
#: two DIFFERENT FILMS to one illuminant rather than one film to two.
#: Fields: PDF page index, plate order on the page, film as the page names it,
#: illuminant, database film_id or None when the emulsion has no profile yet.
PLATES = (
    (41, 0, "KODAK VERICHROME PAN FILM", "sunlight", "KODAK_VERICHROME_PAN"),
    (41, 1, "KODAK VERICHROME PAN FILM", "tungsten", "KODAK_VERICHROME_PAN"),
    (42, 0, "KODAK PLUS-X FILM (35mm)", "sunlight", "KODAK_PLUS_X_125"),
    (42, 1, "KODAK PLUS-X FILM (35mm)", "tungsten", "KODAK_PLUS_X_125"),
    (44, 0, "KODAK TRI-X FILM IN ROLLS", "sunlight", "KODAK_TRI_X_400TX"),
    (44, 1, "KODAK TRI-X FILM IN ROLLS", "tungsten", "KODAK_TRI_X_400TX"),
    (48, 0, "KODAK PANATOMIC-X FILM", "sunlight", "KODAK_PANATOMIC_X"),
    (48, 1, "KODAK PANATOMIC-X FILM", "tungsten", "KODAK_PANATOMIC_X"),
    (52, 0, "KODAK ROYAL PAN SHEET FILM", "sunlight", "KODAK_ROYAL_PAN_4141"),
    (52, 1, "KODAK ROYAL PAN SHEET FILM", "tungsten", "KODAK_ROYAL_PAN_4141"),
    (54, 0, "KODAK TRI-X PANCHROMATIC SHEET FILM", "sunlight",
     "KODAK_TRI_X_SHEET_1952"),
    (54, 1, "KODAK TRI-X PANCHROMATIC SHEET FILM", "tungsten",
     "KODAK_TRI_X_SHEET_1952"),
    (56, 0, "KODAK SUPER PANCHRO-PRESS TYPE B SHEET FILM", "sunlight", None),
    (56, 1, "KODAK SUPER PANCHRO-PRESS TYPE B SHEET FILM", "tungsten", None),
    (58, 0, "KODAK SUPER-XX PANCHROMATIC SHEET FILM", "sunlight",
     "KODAK_SUPER_XX_PAN_4142"),
    (58, 1, "KODAK SUPER-XX PANCHROMATIC SHEET FILM", "tungsten",
     "KODAK_SUPER_XX_PAN_4142"),
    (60, 0, "KODAK PORTRAIT PANCHROMATIC SHEET FILM", "sunlight", None),
    (60, 1, "KODAK PORTRAIT PANCHROMATIC SHEET FILM", "tungsten", None),
    (62, 0, "KODAK PANATOMIC-X SHEET FILM", "sunlight",
     "KODAK_PANATOMIC_X_SHEET_1952"),
    (62, 1, "KODAK PANATOMIC-X SHEET FILM", "tungsten",
     "KODAK_PANATOMIC_X_SHEET_1952"),
    (64, 0, "KODAK ROYAL ORTHO SHEET FILM", "sunlight", None),
    (64, 1, "KODAK ROYAL ORTHO SHEET FILM", "tungsten", None),
    (66, 0, "KODAK SUPER SPEED ORTHO PORTRAIT SHEET FILM", "sunlight", None),
    (66, 1, "KODAK SUPER SPEED ORTHO PORTRAIT SHEET FILM", "tungsten", None),
    (68, 0, "KODAK COMMERCIAL SHEET FILM", "tungsten", None),
    (68, 1, "KODAK COMMERCIAL ORTHO SHEET FILM", "tungsten", None),
    (70, 0, "KODAK CONTRAST PROCESS ORTHO SHEET FILM", "tungsten", None),
    (70, 1, "KODAK CONTRAST PROCESS PANCHROMATIC TYPE B SHEET FILM",
     "tungsten", None),
    (73, 0, "KODAK ROYAL-X PAN SHEET FILM", "sunlight",
     "KODAK_ROYAL_X_PAN_4166"),
    (73, 1, "KODAK ROYAL-X PAN SHEET FILM", "tungsten",
     "KODAK_ROYAL_X_PAN_4166"),
)

#: The page-12 reference stack, top to bottom, exactly as Kodak captions it.
REF_CLASSES = ("EYE", "NON-COLOR-SENSITIZED", "ORTHOCHROMATIC", "PANCHROMATIC")

#: What the reference plate must reproduce before any number here is believed.
#: (class, longest wavelength the boundary survives to, peak wavelength).
REF_GATE = {
    "NON-COLOR-SENSITIZED": ((460.0, 520.0), (430.0, 470.0)),
    "ORTHOCHROMATIC":       ((560.0, 615.0), (520.0, 585.0)),
    "PANCHROMATIC":         ((630.0, 690.0), (550.0, 610.0)),
    "EYE":                  ((670.0, 720.0), (541.0, 569.0)),
}
#: CIE 1924 V(lambda) peaks here; the traced eye plate must find it this close.
EYE_PEAK_NM = 555.0
EYE_PEAK_TOL = 14.0

#: CIE 1924 photopic luminous efficiency, 380-700 nm at 5 nm. The published
#: function the vertical scale is fitted against.
V_LAMBDA = {
    380: 0.000039, 385: 0.000064, 390: 0.000120, 395: 0.000217,
    400: 0.000396, 405: 0.000640, 410: 0.001210, 415: 0.002180,
    420: 0.004000, 425: 0.007300, 430: 0.011600, 435: 0.016840,
    440: 0.023000, 445: 0.029800, 450: 0.038000, 455: 0.048000,
    460: 0.060000, 465: 0.073900, 470: 0.090980, 475: 0.112600,
    480: 0.139020, 485: 0.169300, 490: 0.208020, 495: 0.258600,
    500: 0.323000, 505: 0.407300, 510: 0.503000, 515: 0.608200,
    520: 0.710000, 525: 0.793200, 530: 0.862000, 535: 0.914850,
    540: 0.954000, 545: 0.980300, 550: 0.994950, 555: 1.000000,
    560: 0.995000, 565: 0.978600, 570: 0.952000, 575: 0.915400,
    580: 0.870000, 585: 0.816300, 590: 0.757000, 595: 0.694900,
    600: 0.631000, 605: 0.566800, 610: 0.503000, 615: 0.441200,
    620: 0.381000, 625: 0.321000, 630: 0.265000, 635: 0.217000,
    640: 0.175000, 645: 0.138200, 650: 0.107000, 655: 0.081600,
    660: 0.061000, 665: 0.044580, 670: 0.032000, 675: 0.023200,
    680: 0.017000, 685: 0.011920, 690: 0.008210, 695: 0.005723,
    700: 0.004102,
}
#: Window the slope is fitted over, and the range the window dependence spans.
V_FIT_WINDOW = (410.0, 690.0)
V_FIT_R_MIN = 0.95


# --------------------------------------------------------------------------
# Raster access
# --------------------------------------------------------------------------

def _open():
    import pymupdf
    for p in (PDF, ALT_PDF):
        if p.exists():
            return pymupdf.open(str(p))
    return None


def page_gray(doc, pn, dpi=DPI):
    """One page as float32 grey at `dpi`."""
    import numpy as np
    import pymupdf
    pm = doc[pn].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pm.samples, dtype=np.uint8).reshape(
        pm.height, pm.width).astype(np.float32)


# --------------------------------------------------------------------------
# Plate geometry
# --------------------------------------------------------------------------

def raw_plates(g):
    """Candidate spectrogram rectangles: wide, dark, mostly solid.

    ⚠ THE RESULT IS DELIBERATELY TOO TALL. A morphological close joins the
    plate to the row of wavelength ticks 6 px below it, because the ticks are
    the same ink. `refine` separates them, and it does so on evidence -- the
    plate's left and right SIXTH-inch are unexposed emulsion and therefore
    black on every row of the plate and on no row below it.
    """
    import cv2
    import numpy as np
    dk = (g < INK).astype(np.uint8)
    dk = cv2.morphologyEx(dk, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    n, _lab, st, _cen = cv2.connectedComponentsWithStats(dk, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 900 < w < 1600 and 180 < h < 900 and a > 0.40 * w * h:
            out.append((int(x), int(y), int(w), int(h)))
    out.sort(key=lambda t: (t[1], t[0]))
    return out


def refine(g, box, min_h=150):
    """Split a candidate into the actual plate rectangles it contains.

    Returns one box per plate. A page that stacks two plates vertically comes
    back with two, which is how pages 65 and 67 are read without a special
    case anywhere else in the module.
    """
    import numpy as np
    x, y, w, h = box
    lip = max(8, int(w * 0.06))
    left = (g[y:y + h, x:x + lip] < 130).mean(1)
    right = (g[y:y + h, x + w - lip:x + w] < 130).mean(1)
    # ⚠ THE MAXIMUM OF THE TWO SIDES, NOT THE MEAN OVER BOTH. On page 41's
    # tungsten plate the wedge image runs off the LEFT edge, so the left lip
    # is bright for fifty rows and a mean over both sides cut the plate from
    # 270 px to 217 -- a 20 % error in the vertical scale, which the
    # full-scale gate caught. One black margin is enough to prove a row
    # belongs to the plate.
    dark = np.maximum(left, right)
    rows = np.where(dark > 0.85)[0]
    if not len(rows):
        return []
    seg, i = [], 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1] - rows[j] <= 4:
            j += 1
        seg.append((int(rows[i]), int(rows[j])))
        i = j + 1
    return [(x, y + a, w, b - a + 1) for a, b in seg if b - a + 1 >= min_h]


def ticks(g, box, want=3):
    """The wavelength ticks in the white margin below a plate.

    They are long -- `TICK_MIN` inked rows out of `TICK_ROWS` -- where the
    BLUE / GREEN / RED lettering beside them is at most 30. That single
    property separates them, and the ratio test in `axis` then confirms the
    three that are found really are 400, 500 and 600.
    """
    import numpy as np
    x, y, w, h = box
    strip = g[y + h + 5:y + h + 5 + TICK_ROWS, max(0, x - 60):x + w + 60]
    if strip.size == 0:
        return []
    col = (strip < 170).sum(0)
    idx = np.where(col >= TICK_MIN)[0]
    if not len(idx):
        return []
    grp, cur = [], [idx[0]]
    for v in idx[1:]:
        if v - cur[-1] <= 5:
            cur.append(v)
        else:
            grp.append(cur)
            cur = [v]
    grp.append(cur)
    xs = [float(np.mean(c)) + max(0, x - 60) for c in grp if len(c) >= 2]
    if want and len(xs) > want:
        xs = _best_triple(xs)
    return xs


def _best_triple(xs):
    """The three candidates whose spacing ratio is closest to 0.5010."""
    best, score = None, 1e9
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            for k in range(j + 1, len(xs)):
                span = xs[k] - xs[i]
                if span <= 0:
                    continue
                s = abs((xs[j] - xs[i]) / span - RATIO_NOMINAL)
                if s < score:
                    best, score = [xs[i], xs[j], xs[k]], s
    return best or xs[:3]


def axis(tk):
    """(lambda-of-u callable, px-per-log, ratio) for a three-tick plate.

    ⚠ THE DISPERSION IS QUADRATIC AND THAT IS NOT A FLOURISH. The reference
    plate's four ticks span 381.5, 380.0 and 369.5 px: the red end is
    compressed by 3 %, which a straight line through 400 and 600 misses by
    2 nm at 700 -- a fifth of a sample of the adopted grid.
    """
    import numpy as np
    if len(tk) != 3:
        return None
    span = tk[2] - tk[0]
    if span <= 0:
        return None
    ratio = (tk[1] - tk[0]) / span
    coef = np.polyfit(np.asarray(REF_U), np.asarray(REF_TICK_NM), 2)

    def lam(x):
        return np.polyval(coef, (np.asarray(x, dtype=float) - tk[0]) / span)

    return lam, span / LOG_PER_SPAN, ratio


# --------------------------------------------------------------------------
# The boundary
# --------------------------------------------------------------------------

def boundary(g, box):
    """Top and bottom of the exposed field, per column.

    The plate is a POSITIVE print, so the exposed part of the wedge image is
    LIGHT on a black surround. Per column the reader takes the longest
    continuous light run, which rejects the dust specks and the printer's
    register marks that a first-above-threshold rule picks up -- on the
    page-12 non-colour-sensitized plate that rule alone reported sensitivity
    at 732 nm on a blue-only emulsion.

    ⚠⚠ AND THE OBVIOUS SECOND RULE IS WRONG: "keep the single longest
    contiguous group of columns, because a spectral response has one support".
    It does not. Page 67's COMMERCIAL ORTHO plate is a blue hump, a gap that
    reaches the wedge's transparent end, and a separate green lobe -- the
    second lobe IS the orthochromatic sensitizing, and dropping it turned the
    ortho emulsion into a copy of the blue-sensitive one directly above it.
    The class gate caught that, and the rule is now a WIDTH test: a group of
    at least `MIN_GROUP_PX` columns rising at least `MIN_GROUP_PX_H` px is
    signal, anything narrower is a speck.
    """
    import cv2
    import numpy as np
    x, y, w, h = box
    sub = g[y:y + h, x:x + w]
    bl = cv2.GaussianBlur(sub, (0, 0), BLUR)
    bg = float(np.median(bl[3:15, :]))
    pk = float(np.percentile(bl, 99.5))
    thr = bg + EDGE_FRAC * (pk - bg)
    H, W = sub.shape
    top = np.full(W, -1, dtype=np.int32)
    bot = np.full(W, -1, dtype=np.int32)
    for c in range(W):
        idx = np.where(bl[:, c] > thr)[0]
        if len(idx) < MIN_RUN:
            continue
        runs, i = [], 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and idx[j + 1] - idx[j] == 1:
                j += 1
            runs.append((idx[i], idx[j]))
            i = j + 1
        a, b = max(runs, key=lambda r: r[1] - r[0])
        if b - a >= MIN_RUN:
            top[c], bot[c] = a, b
    keep = np.zeros(W, dtype=bool)
    i = 0
    while i < W:
        if top[i] >= 0:
            j = i
            while j < W and top[j] >= 0:
                j += 1
            if (j - i) >= MIN_GROUP_PX and (bot[i:j] - top[i:j]).max() >= \
                    MIN_GROUP_PX_H:
                keep[i:j] = True
            i = j
        else:
            i += 1
    if not keep.any():
        return None
    base = float(np.median(bot[keep]))
    height = np.where(keep, base - top, np.nan)
    return keep, height, base, thr


def read_plate(g, box, tk):
    """One plate -> (lambda array, log-sensitivity array, mask, facts)."""
    import numpy as np
    ax = axis(tk)
    b = boundary(g, box)
    if ax is None or b is None:
        return None
    lam_of, px_log, ratio = ax
    keep, height, base, _thr = b
    x, _y, w, h = box
    lam = lam_of(x + np.arange(w))
    i = int(np.nanargmax(height))
    logs = (height - height[i]) / px_log
    facts = {
        "ratio": ratio,
        "px_per_log": px_log,
        "full_scale_log": h / px_log,
        "peak_nm": float(lam[i]),
        "lo_nm": float(lam[keep][0]),
        "hi_nm": float(lam[keep][-1]),
        "floor_log": float(-height[i] / px_log),
        "cols": int(keep.sum()),
    }
    return lam, logs, keep, facts


def resample(lam, logs, keep):
    """Onto the adopted 380-700 nm / 10 nm grid, off-scale -> FLOOR.

    ⚠ INTERPOLATION IS PER SEGMENT. A plate whose support is in two pieces --
    the ortho emulsions of pages 67 and 69 -- must keep the hole between them,
    because the hole is a measurement: the boundary there has reached the
    wedge's transparent end. A single `interp` over the concatenated columns
    would draw a straight line across it and invent sensitivity.
    """
    import numpy as np
    grid = LAMBDA_START + LAMBDA_STEP * np.arange(LAMBDA_N)
    segs, i, W = [], 0, len(lam)
    while i < W:
        if keep[i]:
            j = i
            while j < W and keep[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    # ⚠ EACH SAMPLE IS THE MEAN OVER ITS OWN 10 nm BIN, NOT AN INTERPOLATION
    # AT ITS CENTRE. A bin is about 25 columns wide and the halftone screen
    # puts +/-0.15 log of ripple on a single column, so point sampling threw
    # away twenty-four measurements out of twenty-five and kept the noise.
    half = LAMBDA_STEP / 2.0
    out = []
    for L in grid:
        vals = []
        for a, b in segs:
            xs, ys = lam[a:b], logs[a:b]
            sel = (xs >= L - half) & (xs < L + half)
            if sel.any():
                vals.append(float(ys[sel].mean()))
        out.append(round(max(vals), 4) if vals else FLOOR)
    hi = max(out)
    return tuple(round(v - hi, 4) if v > FLOOR else FLOOR for v in out)


# --------------------------------------------------------------------------
# The page-12 reference plate, and the scale it carries
# --------------------------------------------------------------------------

def reference(doc):
    """Read the four reference spectrograms and the four labelled ticks."""
    import numpy as np
    g = page_gray(doc, REF_PAGE)
    band = g[2500:4300, 540:1990].mean(1)
    white = band > 215
    runs, i = [], 0
    while i < len(white):
        if not white[i]:
            j = i
            while j < len(white) and not white[j]:
                j += 1
            if j - i > 120:
                runs.append((i + 2500, j + 2500))
            i = j
        else:
            i += 1
    if len(runs) != 4:
        return None
    # the four labelled ticks, in the gap under the first plate
    y0 = runs[0][1] + 14
    strip = g[y0:y0 + 60, :]
    col = (strip < 140).mean(0)
    idx = np.where(col > 0.6)[0]
    grp, cur = [], [idx[0]]
    for v in idx[1:]:
        if v - cur[-1] <= 3:
            cur.append(v)
        else:
            grp.append(cur)
            cur = [v]
    grp.append(cur)
    tk = [float(np.mean(c)) for c in grp]
    if len(tk) != 4:
        return None
    x0, x1 = 530, 2003
    out = {}
    for (a, b), nm in zip(runs, REF_CLASSES):
        r = read_plate(g, (x0, a, x1 - x0, b - a), [tk[0], tk[1], tk[2]])
        if r is None:
            return None
        out[nm] = r
    return out, tk


def wedge_gradient(ref):
    """Fit the eye plate against V(lambda). Returns (log_per_span, r, rms, n).

    ⚠ THIS IS THE ONLY PLACE IN THE MODULE WHERE A PUBLISHED NUMBER FROM
    OUTSIDE THE BOOK ENTERS, and it enters as a CHECK on a constant that is
    frozen in `LOG_PER_SPAN` rather than as the constant itself -- so a change
    in the raster cannot silently move the scale of thirty adopted curves.
    """
    import numpy as np
    lam, logs, keep, _f = ref["EYE"]
    # heights back out of the normalised log, in px-independent log units
    wl = np.array(sorted(V_LAMBDA), dtype=float)
    vv = np.array([V_LAMBDA[int(k)] for k in wl], dtype=float)
    ys = np.interp(wl, lam[keep], logs[keep], left=np.nan, right=np.nan)
    m = (~np.isnan(ys)) & (wl >= V_FIT_WINDOW[0]) & (wl <= V_FIT_WINDOW[1])
    if m.sum() < 30:
        return None
    x = np.log10(vv[m])
    y = ys[m]
    a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    rms = float(np.sqrt(((y - np.polyval(a, x)) ** 2).mean()))
    return float(a[0]), r, rms, int(m.sum())


# --------------------------------------------------------------------------
# Harvest
# --------------------------------------------------------------------------

def harvest(doc):
    """Every plate in `PLATES`, read. Returns a list of records."""
    rows = []
    by_page = {}
    for pn, order, film, illum, fid in PLATES:
        by_page.setdefault(pn, []).append((order, film, illum, fid))
    for pn in sorted(by_page):
        g = page_gray(doc, pn)
        boxes = []
        for raw in raw_plates(g):
            boxes.extend(refine(g, raw))
        # left-to-right when side by side, top-to-bottom when stacked
        if len(boxes) >= 2 and abs(boxes[0][1] - boxes[-1][1]) < 60:
            boxes.sort(key=lambda t: t[0])
        else:
            boxes.sort(key=lambda t: t[1])
        for order, film, illum, fid in sorted(by_page[pn]):
            if order >= len(boxes):
                rows.append({"page": pn, "order": order, "film": film,
                             "illum": illum, "film_id": fid, "ok": False,
                             "why": "no plate rectangle at this position"})
                continue
            box = boxes[order]
            tk = ticks(g, box)
            if len(tk) != 3:
                rows.append({"page": pn, "order": order, "film": film,
                             "illum": illum, "film_id": fid, "ok": False,
                             "why": "%d wavelength ticks, need 3" % len(tk)})
                continue
            r = read_plate(g, box, tk)
            if r is None:
                rows.append({"page": pn, "order": order, "film": film,
                             "illum": illum, "film_id": fid, "ok": False,
                             "why": "no boundary"})
                continue
            lam, logs, keep, facts = r
            rec = {"page": pn, "order": order, "film": film, "illum": illum,
                   "film_id": fid, "ok": True, "box": box, "ticks": tk}
            rec.update(facts)
            rec["curve"] = resample(lam, logs, keep)
            rows.append(rec)
    return rows


def planck(nm, kelvin):
    """Relative spectral radiant exitance of a Planck radiator, peak 1.0."""
    import numpy as np
    m = np.asarray(nm, dtype=float) * 1e-9
    e = 1.0 / (m ** 5 * (np.exp(1.4388e-2 / (m * kelvin)) - 1.0))
    return e / e.max()


def illuminant_log(illum):
    """log10 E(lambda) on the adopted grid, peak 0.0."""
    import numpy as np
    grid = LAMBDA_START + LAMBDA_STEP * np.arange(LAMBDA_N)
    k = SUN_K if illum == "sunlight" else TUNGSTEN_K
    return np.log10(planck(grid, k))


def corrected(curve, illum):
    """One plate's curve with the spectrograph's own source divided out."""
    import numpy as np
    a = np.asarray(curve, dtype=float)
    live = a > FLOOR + 0.01
    out = np.where(live, a - illuminant_log(illum), FLOOR)
    if live.any():
        out = np.where(live, out - out[live].max(), FLOOR)
    return out


def merge(recs):
    """One emulsion's plates -> one spectral sensitivity.

    Both illuminants are independent records of the same S, so where both are
    on scale the mean is taken and the independent error is halved. Where only
    one is on scale that one stands; where neither is, the schema sentinel.
    """
    import numpy as np
    cs = [corrected(r["curve"], r["illum"]) for r in recs]
    A = np.vstack(cs)
    live = A > FLOOR + 0.01
    out = np.full(A.shape[1], FLOOR)
    for i in range(A.shape[1]):
        if live[:, i].any():
            out[i] = A[live[:, i], i].mean()
    hi = out.max()
    return tuple(round(float(v - hi), 4) if v > FLOOR else FLOOR for v in out)


def pair_residual(recs):
    """Mean |sunlight - tungsten| before and after the illuminant correction."""
    import numpy as np
    if len(recs) != 2:
        return None
    a, b = recs[0], recs[1]
    ra = np.asarray(a["curve"], dtype=float)
    rb = np.asarray(b["curve"], dtype=float)
    m = (ra > FLOOR + 0.01) & (rb > FLOOR + 0.01)
    if m.sum() < 8:
        return None
    raw = float(np.abs((ra[m] - ra[m].max()) - (rb[m] - rb[m].max())).mean())
    ca, cb = corrected(a["curve"], a["illum"]), corrected(b["curve"], b["illum"])
    fix = float(np.abs((ca[m] - ca[m].max()) - (cb[m] - cb[m].max())).mean())
    return raw, fix


def classify(rec):
    """'blue' / 'ortho' / 'pan' from the film's own printed name."""
    n = rec["film"]
    if "COMMERCIAL SHEET" in n:
        return "blue"
    if "ORTHO" in n:
        return "ortho"
    return "pan"


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def check_reference(ref):
    """Kodak's own three sensitizing classes, reproduced or not."""
    bad = []
    for nm, ((clo, chi), (plo, phi)) in REF_GATE.items():
        _lam, _logs, _keep, f = ref[nm]
        if not clo <= f["hi_nm"] <= chi:
            bad.append("%s cuts off at %.0f nm, not in %.0f-%.0f"
                       % (nm, f["hi_nm"], clo, chi))
        if not plo <= f["peak_nm"] <= phi:
            bad.append("%s peaks at %.0f nm, not in %.0f-%.0f"
                       % (nm, f["peak_nm"], plo, phi))
    order = [ref[k][3]["hi_nm"] for k in
             ("NON-COLOR-SENSITIZED", "ORTHOCHROMATIC", "PANCHROMATIC")]
    if not (order[0] < order[1] < order[2]):
        bad.append("the three sensitizing classes are not ordered "
                   "blue < ortho < pan: %s"
                   % ", ".join("%.0f" % v for v in order))
    return bad


def check_classes(rows):
    """Every orthochromatic plate must cut off below every panchromatic one."""
    good = [r for r in rows if r.get("ok")]
    blue = [r["hi_nm"] for r in good if classify(r) == "blue"]
    orth = [r["hi_nm"] for r in good if classify(r) == "ortho"]
    pan = [r["hi_nm"] for r in good if classify(r) == "pan"]
    bad = []
    if blue and orth and max(blue) >= min(orth):
        bad.append("a blue-sensitive plate reaches %.0f nm and an "
                   "orthochromatic one stops at %.0f" % (max(blue), min(orth)))
    if orth and pan and max(orth) >= min(pan):
        bad.append("an orthochromatic plate reaches %.0f nm and a "
                   "panchromatic one stops at %.0f" % (max(orth), min(pan)))
    return bad, (blue, orth, pan)


def check_wedge(rows, ref):
    """One wedge made all the plates, or the full-scale range says otherwise."""
    fs = [r["full_scale_log"] for r in rows if r.get("ok")]
    fs.append(ref["PANCHROMATIC"][3]["full_scale_log"])
    lo, hi = min(fs), max(fs)
    band = (REF_FULL_SCALE_LOG * (1.0 - LOG_SCALE_UNCERTAINTY),
            REF_FULL_SCALE_LOG * (1.0 + LOG_SCALE_UNCERTAINTY))
    if lo < band[0] or hi > band[1]:
        return ["plate full-scale log range %.2f-%.2f leaves %.2f-%.2f, so "
                "the plates were not all made on one wedge and the scale "
                "transfer is void" % (lo, hi, band[0], band[1])], (lo, hi)
    return [], (lo, hi)


def check_ratio(rows):
    bad = []
    for r in rows:
        if r.get("ok") and abs(r["ratio"] - RATIO_NOMINAL) > RATIO_TOL:
            bad.append("page %d plate %d tick ratio %.4f" %
                       (r["page"], r["order"], r["ratio"]))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.parse_args(argv)

    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        print("[SKIP] kodak_1956_spectro.py -- numpy/opencv not available")
        return 0
    doc = _open()
    if doc is None:
        print("[SKIP] kodak_1956_spectro.py -- 1956-Kodak-Films.pdf not staged")
        return 0

    ref = reference(doc)
    if ref is None:
        print("[FAIL] kodak_1956_spectro.py -- the page-12 reference plate, "
              "which is the only thing that calibrates the other thirty, did "
              "not resolve into four spectrograms and four labelled ticks")
        return 1
    ref, rtk = ref

    bad = check_reference(ref)
    if bad:
        print("[FAIL] kodak_1956_spectro.py -- the reader does not reproduce "
              "Kodak's own sensitizing classes, so nothing it reads is "
              "believed: " + "; ".join(bad))
        return 1

    grad = wedge_gradient(ref)
    if grad is None:
        print("[FAIL] kodak_1956_spectro.py -- the eye plate did not yield "
              "enough samples to check the wedge gradient against V(lambda)")
        return 1
    slope, rr, rms, nfit = grad
    if rr < V_FIT_R_MIN:
        print("[FAIL] kodak_1956_spectro.py -- the eye plate no longer tracks "
              "V(lambda) (r=%.3f < %.2f), so the vertical scale is not "
              "calibrated" % (rr, V_FIT_R_MIN))
        return 1
    if abs(slope - 1.0) > 0.20:
        print("[FAIL] kodak_1956_spectro.py -- the frozen wedge gradient "
              "LOG_PER_SPAN=%.2f disagrees with the eye plate by %.0f %%"
              % (LOG_PER_SPAN, abs(slope - 1.0) * 100.0))
        return 1

    rows = harvest(doc)
    good = [r for r in rows if r.get("ok")]
    if len(good) != len(PLATES):
        miss = ["page %d plate %d (%s)" % (r["page"], r["order"], r["why"])
                for r in rows if not r.get("ok")]
        print("[FAIL] kodak_1956_spectro.py -- %d of %d plates did not read: %s"
              % (len(PLATES) - len(good), len(PLATES), "; ".join(miss)))
        return 1

    bad = check_ratio(rows)
    if bad:
        print("[FAIL] kodak_1956_spectro.py -- the 400/500/600 tick "
              "identification fails its own spacing test on: "
              + ", ".join(bad))
        return 1

    bad, (blue, orth, pan) = check_classes(rows)
    if bad:
        print("[FAIL] kodak_1956_spectro.py -- " + "; ".join(bad))
        return 1

    bad, (fslo, fshi) = check_wedge(rows, ref)
    if bad:
        print("[FAIL] kodak_1956_spectro.py -- " + "; ".join(bad))
        return 1

    films = {}
    for r in good:
        films.setdefault(r["film"], []).append(r)
    raws, fixes, worst = [], [], ("", 0.0)
    for film, recs in films.items():
        pr = pair_residual(sorted(recs, key=lambda r: r["illum"]))
        if pr is None:
            continue
        raws.append(pr[0])
        fixes.append(pr[1])
        if pr[1] > worst[1]:
            worst = (film, pr[1])
    if not fixes:
        print("[FAIL] kodak_1956_spectro.py -- no emulsion carries both a "
              "sunlight and a tungsten plate, so the illuminant correction "
              "cannot be tested and the curves are not spectral sensitivities")
        return 1
    mraw = sum(raws) / len(raws)
    mfix = sum(fixes) / len(fixes)
    if mfix >= mraw:
        print("[FAIL] kodak_1956_spectro.py -- dividing out the spectrograph's "
              "own source does NOT bring an emulsion's two plates together "
              "(%.3f log before, %.3f after), so either the plates are not "
              "source-weighted or the sources are not what they are taken to "
              "be" % (mraw, mfix))
        return 1
    if mfix > PAIR_RESIDUAL_MAX or worst[1] > PAIR_RESIDUAL_WORST:
        print("[FAIL] kodak_1956_spectro.py -- the two illuminant plates of "
              "one emulsion still disagree by %.3f log on average and %.3f on "
              "%s, past the %.2f / %.2f this harvest is allowed"
              % (mfix, worst[1], worst[0], PAIR_RESIDUAL_MAX,
                 PAIR_RESIDUAL_WORST))
        return 1
    curves = {film: merge(recs) for film, recs in films.items()}

    housed = sorted({r["film_id"] for r in good if r["film_id"]})
    unhoused = sorted({r["film"] for r in good if not r["film_id"]})
    floors = [r["floor_log"] for r in good]
    print("[OK] kodak_1956_spectro.py -- «Kodak Films» Seventh Edition 1956, "
          "all %d wedge spectrograms on %d pages read as density boundaries "
          "in a halftone field. The page-12 reference plate calibrates "
          "everything and validates it first: Kodak's own "
          "NON-COLOR-SENSITIZED / ORTHOCHROMATIC / PANCHROMATIC stack comes "
          "back cutting off at %.0f / %.0f / %.0f nm in that order, and its "
          "SENSITIVITY OF THE EYE plate peaks at %.0f nm against V(lambda)'s "
          "%.0f. That eye plate is also what gives the vertical axis a scale "
          "Kodak never printed: %d samples of log10 V(lambda) against traced "
          "height, r=%.3f, rms %.2f log, and the frozen LOG_PER_SPAN=%.2f "
          "reproduces it to %.0f %%. One wedge made all %d plates -- their "
          "full heights come out %.2f-%.2f log units. The class test the "
          "queue asked for passes on the data sheets too: blue-sensitive "
          "stops by %.0f nm, orthochromatic by %.0f, panchromatic runs to "
          "%.0f. ⚠ AND A WEDGE SPECTROGRAM IS NOT A SPECTRAL SENSITIVITY "
          "UNTIL THE SPECTROGRAPH'S OWN LIGHT IS DIVIDED OUT: doing that "
          "brings each emulsion's two independent plates from %.3f log apart "
          "to %.3f, worst %.3f on %s, which is both the justification for the "
          "correction and the honest accuracy of the result. %d emulsions "
          "come out of it; %d gain a measured spectral response and %d more "
          "are read and held for the profiles that do not exist yet; the "
          "off-scale floor is %.1f to %.1f log below each plate's own peak "
          "and is stored as the schema sentinel, not as a measurement"
          % (len(good), len({r['page'] for r in good}),
             ref["NON-COLOR-SENSITIZED"][3]["hi_nm"],
             ref["ORTHOCHROMATIC"][3]["hi_nm"],
             ref["PANCHROMATIC"][3]["hi_nm"],
             ref["EYE"][3]["peak_nm"], EYE_PEAK_NM,
             nfit, rr, rms, LOG_PER_SPAN, abs(slope - 1.0) * 100.0,
             len(good), fslo, fshi,
             max(blue) if blue else 0.0,
             max(orth) if orth else 0.0,
             max(pan) if pan else 0.0,
             mraw, mfix, worst[1], worst[0].replace("KODAK ", ""),
             len(curves), len(housed), len(unhoused),
             min(floors), max(floors)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
