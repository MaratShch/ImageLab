#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""«Современные фотоматериалы и их обработка» -- the plot atlas, re-read on
every build.

⚠ WHY THIS READER EXISTS, AND THE MISTAKE IT CORRECTS. This 717-page volume was
reviewed on 2026-09-15 and written off as "a processing manual and nothing
else". That was wrong, and wrong for an instructive reason: the review searched
the text layer for TABLE captions containing the words granularity, resolving
power, spectral sensitivity and characteristic curve, found thirty-five word
occurrences and no table, and reported the null result as a property of the
document instead of a property of the query. The tables really are processing
tables -- 973 «табл» references across 381 pages, and not one granularity
figure among them. The SENSITOMETRY IS IN THE FIGURES, and the figures were
never looked at. There are 708 of them:

    Характеристические кривые              258
    Спектральная плотность красителей      117
    Спектральное поглощение красителями    115
    Функция передачи модуляции              81
    Кривые кинетики проявления              42
    Спектральная плотность (monochrome)     21
    Спектральная чувствительность           10
    Строение (layer structure)              19
    other (reciprocity, marking, process)   45

Chapter 2 gives a four-panel set -- characteristic curves, dye absorption, dye
spectral density, MTF -- for each Agfa and Konica amateur film. Chapter 3 does
the same for the Kodak and Fuji professional range including PUSH-PROCESSED
variants, and adds the development-kinetics family. Chapter 4 covers the print
and display materials.

WHAT THIS READER ADOPTS, AND THE RULE IT OBEYS. The book is [T2] documented
reference data, not a manufacturer sheet, so by the corpus's precedence rule it
FILLS a field the vendor sheet leaves empty and NEVER overwrites a figure that
came from the manufacturer. Every panel it reads is therefore sorted into one
of three outcomes, and all three are printed on every build:

  FILL         the database holds a placeholder or an era estimate and the book
               holds a measurement -- the book wins
  CORROBORATE  the database already holds manufacturer data and the book's
               independent trace agrees -- nothing is written, and the
               agreement is itself the check on this reader
  DIFFER       the two disagree by more than the tolerance -- nothing is
               written and the disagreement is reported, because a silent
               choice between two sources is exactly what the precedence rule
               forbids

HOW A PANEL IS READ. Every figure is an embedded bitmap of about 480x360 with
no vector paths, so the numbers come off pixels:

  1. The plot frame is the pair of rows and the pair of columns that run nearly
     the full width or height of the bitmap.
  2. The tick labels are one uniform font at two sizes, about 6x11 px on the
     characteristic curves and 9x13 on the kinetics and MTF panels. Tesseract
     misreads glyphs that small -- it drops the decimal comma and confuses 4
     with 1 -- so the reader cuts its OWN template bank out of ten axes in this
     same PDF whose values are legible to a human, and matches by normalised
     cross-correlation with an aspect-ratio prior and a margin requirement.
  3. The axis is then fitted value = a*px + b over the LARGEST CONSISTENT
     SUBSET of decoded labels, so one misread digit is dropped rather than
     averaged in, and the axis is refused unless at least three labels agree,
     they are at least 70 % of what was read, and every one lands within 2 % of
     the span.
  4. Curves are followed column by column with a slope-predicting gate.
  5. THE ADOPTION CROSS-CHECK IS PHYSICAL, NOT STATISTICAL. A masked colour
     negative's base density must come out an orange ladder -- blue above green
     above red, spread at least 0.15 D, the whole of it under D 1.6. A trace
     that puts the records in any other order has mis-assigned them and is
     refused. That single test is what makes an unattended run of 258 panels
     safe: a wrong trace cannot look right.

WHAT IS DELIBERATELY NOT ADOPTED, AND WHY.

  ⚠ THE 42 DEVELOPMENT-KINETICS PANELS ARE READ AND NOT STORED. They plot the
  contrast coefficient against development time at 20 degC for six to eight
  developers, which is precisely the shape `ProcessingFamily` wants and
  precisely the gap that leaves Development Time with no reader. The axes
  calibrate and the curves trace. What does not survive is the ATTRIBUTION: the
  panels name their curves either with a dashed-line legend or with labels
  dropped beside the strokes, and on the crowded panels -- Ektapan has four
  labels stacked over three curves that cross -- the nearest-curve assignment
  is not unambiguous. A DevelopmentPoint carries a developer name, and storing
  one against the wrong curve would be worse than storing nothing. The reader
  therefore reports the panels it can see and stores none of them; closing this
  needs dash-signature matching against the legend swatches, which is
  DIGITIZATION_QUEUE P49.

  ⚠ COLOUR MTF IS ONE COMBINED CURVE. The book prints a single modulation
  transfer curve per film, and `MTFSpec` wants f50 per record. One curve cannot
  be split into three, so an f50 off a colour panel is only ever reported as a
  bracket check against the stored triple, never written. The monochrome panels
  are a different matter -- there the stock has one record and the book's curve
  is that record.

  ⚠ THE DYE PANELS ARE READ AND NOT STORED. 232 spectral absorption and dye
  density panels calibrate, but `dye_matrix` is a 3x3 of coupling
  coefficients, not a spectrum, and no published calibration turns one into the
  other. Converting them would be inventing the conversion. DIGITIZATION_QUEUE
  P50.

Run standalone for the full report; ``--assert`` is the build gate form.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

try:
    import pytesseract
except Exception as exc:                                    # pragma: no cover
    raise SystemExit("sovremennye_2004.py needs pytesseract: %s" % exc)
try:
    import cv2
except Exception as exc:                                    # pragma: no cover
    raise SystemExit("sovremennye_2004.py needs OpenCV: %s" % exc)
try:
    import pymupdf
except Exception as exc:                                    # pragma: no cover
    raise SystemExit("sovremennye_2004.py needs PyMuPDF: %s" % exc)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PDF = Path("/root/work/tst/PDF/PROFILES/SOVIET/"
           "Современные фотоматериалы и их обработка.pdf")
ALT_PDF = Path("/mnt/user-data/uploads/PYTHON.TST/PDF/PROFILES/SOVIET/"
               "Современные фотоматериалы и их обработка.pdf")

# ---------------------------------------------------------------------------
# Tolerances. Each one is the number a panel has to beat, not a number tuned
# until the panels passed.
# ---------------------------------------------------------------------------
MATCH_MIN = 0.55        # cross-correlation floor for one digit template
MATCH_MARGIN = 0.02     # the winning digit must clear the runner-up by this
AXIS_RESID_TOL = 0.02   # every kept tick within 2 % of the axis span
AXIS_MIN_LABELS = 3     # fewer than three labels is not an axis
AXIS_MIN_KEEP = 0.70    # and at least this fraction of what was read
TRACE_MIN_COVER = 0.45  # a curve must span this fraction of the panel width
FAMILY_MIN_COVER = 0.06 # see FAMILY_MIN_DECADES -- the real test is in log E
FAMILY_MIN_DECADES = 0.80   # each curve of a family must span this much log E
GAMMA_WINDOW = 0.60         # and gamma is the steepest slope over this window
LADDER_MIN = 0.15       # a mask spreads the base densities by at least this
DMIN_MAX = 1.60         # and no record's base density exceeds this
DMIN_MIN = 0.02
AGREE_D = 0.06          # book vs database agreement band, density
AGREE_F50 = 0.15        # and MTF, as a fraction of the stored value
FLAT_LADDER = 0.05      # a stored ladder flatter than this is a placeholder

NORM = (10, 14)
_AR = {'0': .58, '1': .40, '2': .58, '3': .58, '4': .62,
       '5': .58, '6': .58, '7': .55, '8': .58, '9': .58}

# The ten axes whose printed values are legible to a human. The template bank
# is cut from these and every other axis in the book is decoded against it, so
# the bank is derived from the source on each run and is not a stored artefact.
BANK_SOURCES = [
    # (page, figure kind, film as the caption names it, axis, printed labels)
    (68, 'char', 'Ultra 100', 'x', ['-4,0', '-3,0', '-2,0', '-1,0', '0,0', '1,0']),
    (68, 'char', 'Ultra 100', 'y', ['4,0', '3,0', '2,0', '1,0', '0,0']),
    (69, 'mtf', 'Ultra 100', 'x', ['2', '5', '10', '20', '50', '100']),
    (69, 'mtf', 'Ultra 100', 'y', ['10', '20', '50', '100', '150']),
    (72, 'mtf', 'Vista 200', 'x', ['2', '5', '10', '20', '50', '100']),
    (72, 'mtf', 'Vista 200', 'y', ['10', '20', '50', '100', '150']),
    (334, 'kinet', 'Kodak Professional T-MAX 100', 'x',
     ['4', '6', '8', '10', '12', '14', '16', '18']),
    (334, 'kinet', 'Kodak Professional T-MAX 100', 'y',
     ['0,9', '0,8', '0,7', '0,6', '0,5', '0,4', '0,3']),
    (401, 'kinet', 'TRI-X 320 / 320TXP', 'x', ['0', '5', '10', '15', '20', '25']),
    (401, 'kinet', 'TRI-X 320 / 320TXP', 'y',
     ['0,9', '0,8', '0,7', '0,6', '0,5', '0,4', '0,3']),
    # -- queue P56, 2026-09-17d. FOUR PANELS FAILED CALIBRATION, NOT TRACING,
    # and the cause was a LABEL FORM the bank had never been shown: the ten
    # axes above all label WHOLE steps, while the Fujichrome reversal panels
    # label the density axis in HALF steps (0,0 / 0,5 / 1,0 ...) and the Konica
    # panel labels only every other gridline. Every value below was read off a
    # 200 dpi render of the page, which is how the first ten got here.
    (105, 'char', 'Konica Color VX 100', 'x',
     ['-3,0', '-2,0', '-1,0', '0,0', '1,0']),
    (105, 'char', 'Konica Color VX 100', 'y', ['3,0', '2,0', '1,0']),
    (277, 'char', 'Fujichrome Provia 100F Professional', 'x',
     ['-3,0', '-2,0', '-1,0', '0,0']),
    (277, 'char', 'Fujichrome Provia 100F Professional', 'y',
     ['4,0', '3,5', '3,0', '2,5', '2,0', '1,5', '1,0', '0,5', '0,0']),
    (280, 'char', 'Fujichrome Provia 400 Professional', 'x',
     ['-4,0', '-3,0', '-2,0', '-1,0', '0,0']),
    (280, 'char', 'Fujichrome Provia 400 Professional', 'y',
     ['3,5', '3,0', '2,5', '2,0', '1,5', '1,0', '0,5', '0,0']),
    # -- queue P56 finished, 2026-09-18b. The fourth panel the row named,
    # Fujichrome 64T Type II, was never a template gap at all: its frame is
    # printed in GREY, at levels 156-199 against the reader's 128 cutoff, so
    # `find_frame` saw no border and nothing downstream ran. With the
    # threshold ladder in `find_frame` it reads, and its axes are added here
    # because two more samples of the same half-step ladder are what the
    # remaining two panels need.
    (269, 'char', 'Fujichrome 64T Type II Professional', 'x',
     ['-3,0', '-2,0', '-1,0', '0,0', '1,0']),
    (269, 'char', 'Fujichrome 64T Type II Professional', 'y',
     ['4,0', '3,5', '3,0', '2,5', '2,0', '1,5', '1,0', '0,5', '0,0']),
    # and the dye-panel wavelength axis, which is where the bank's 6 and 7 are
    # thinnest -- four templates for 6 and two for 7 across the whole book.
    (68, 'dyeabs', 'Ultra 100', 'x', ['400', '500', '600', '700']),
    (71, 'dyeabs', 'Vista 200', 'x', ['400', '500', '600', '700']),
    # ⚠⚠ 2026-09-18c, AND THIS ONE IS HERE BECAUSE THE SIX ABOVE MADE THE BANK
    # WORSE ON ONE PAGE. Adding the dye-panel wavelength axes gave the bank
    # exemplars of «6» in the WAVELENGTH face; p341's MTF abscissa prints its
    # «600» and its «3» in the frequency face, and after the addition the bank
    # read that «600» as «0» and that «3» as nothing -- where before it read
    # the 600 correctly. A zero cannot sit on a log axis, so `fit_axis` then
    # dropped the label and f50 moved from 98.67 to 96.48: a real measurement
    # degraded by a change meant to recover four other panels.
    # ⚠ THE REMEDY IS MORE EXEMPLARS OF THE CONFUSED GLYPHS IN THIS FACE, not
    # fewer of the other. Every value below was read off a 150 dpi render of
    # the page, which is how all sixteen axes above got here.
    (341, 'mtf', 'Kodak Professional T-MAX 400', 'x',
     ['1', '2', '3', '4', '5', '10', '20', '50', '100', '200', '600']),
]

# ---------------------------------------------------------------------------
# Book film name -> database stock name.
#
# Hand-checked one line at a time against film_names.txt. A name is mapped only
# where the book and the database mean the SAME product. A near neighbour with
# a different speed or a different generation is left unmapped rather than
# guessed, and the book's still-format entries are never matched to the cine
# stocks of the same family (EASTMAN PLUS X 5231 is not Kodak Plus-X Pan).
# ---------------------------------------------------------------------------
STOCK_MAP = {
    # -- queue P54, 2026-09-18b. THE TABLE CAPTIONS SPELL THE FILM DIFFERENTLY
    # FROM THE FIGURE CAPTIONS this map was built from: a figure says «TRI-X
    # 400 / 400TX», the development table above it says «Kodak Professional
    # TRI-X 400 / 400TX». `sovremennye_dev_tables.stock_of` strips the house
    # prefix, and these are the remaining spellings that need naming outright.
    'Kodak Professional Plus-X 125 / 125PX': 'KODAK_PLUS_X_125',
    'Kodak Plus-X Pan / PX': 'KODAK_PLUS_X_125',
    'Kodak Plus-X Pan Professional / PXE': 'KODAK_PLUS_X_125',
    'Kodak Plus-X Pan Professional / PXT': 'KODAK_PLUS_X_125',
    'T-MAX P3200 Professional': 'KODAK_TMAX_P3200',

    'Vista 200': 'AGFA_VISTA_200',
    'Optima 100': 'AGFA_OPTIMA_100',
    'Optima 200': 'AGFA_OPTIMA_200',
    'Optima 400': 'AGFA_OPTIMA_400',
    'Portrait 160': 'AGFA_PORTRAIT_160',
    'RSX 50': 'AGFA_RSX_II_50',
    'RSX 100': 'AGFA_RSX_II_100',
    'RSX 200': 'AGFA_RSX_II_200',
    'APX 100': 'AGFA_APX_100',
    'APX 400': 'AGFA_APX_400',
    'Agfa Scala 200x': 'AGFA_SCALA_200X',
    'Portra 100Т': 'KODAK_PORTRA_100T',
    'Portra 160NC': 'KODAK_PORTRA_160NC',
    'Portra 160VC': 'KODAK_PORTRA_160VC',
    'Portra 400NC': 'KODAK_PORTRA_400NC',
    'Portra 400VC': 'KODAK_PORTRA_400VC',
    'Portra 800': 'KODAK_PORTRA_800',
    'Portra 400UC': 'KODAK_ULTRA_COLOR_400UC',
    'Vericolor III': 'KODAK_VERICOLOR_III_160',
    'PJ400': 'KODAK_EKTAPRESS_PJ400',
    'Pro 100T / PRT': 'KODAK_PRO_100T_PRT',
    'Profoto 100': 'KODAK_PROFOTO_100',
    'Kodak Professional T400 CN': 'KODAK_T400CN',
    'Portra 400BW': 'KODAK_BW400CN',
    'Ektachrome 160T': 'EKTACHROME_160T',
    'Kodak Ektachrome 64 Professional': 'EKTACHROME_64',
    'Kodachrome 64': 'KODACHROME_64',
    'Kodak Ektapan': 'KODAK_EKTAPAN_100',
    'Kodak Verichrome Pan': 'KODAK_VERICHROME_PAN',
    'Kodak Professional Technical Pan': 'KODAK_TECHNICAL_PAN',
    'Kodak Professional T-MAX 100': 'KODAK_TMAX_100',
    'Kodak T-MAX 100': 'KODAK_TMAX_100',
    'Kodak Professional T-MAX 400': 'KODAK_TMAX_400',
    'Kodak T-MAX 400': 'KODAK_TMAX_400',
    'Kodak Professional T-MAX P3200': 'KODAK_TMAX_P3200',
    'Kodak T-MAX Р3200': 'KODAK_TMAX_P3200',
    'Kodak Professional Plus-X 125': 'KODAK_PLUS_X_125',
    'Kodak Plus-X Pan': 'KODAK_PLUS_X_125',
    'Kodak Plus-X Pan Professional / PXP': 'KODAK_PLUS_X_125',
    'TRI-X 400 / 400TX': 'KODAK_TRI_X_400TX',
    'Kodak TRI-X Pan / TX': 'KODAK_TRI_X_400TX',
    'TRI-X 320 / 320TXP': 'KODAK_TRI_X_320TXP',
    'TRI-X 320 / 320TXP листового': 'KODAK_TRI_X_320TXP',
    'Kodak TRI-X Pan Professional / TXP': 'KODAK_TRI_X_320TXP',
    'Fujichrome Provia 100F Professional': 'FUJI_PROVIA_100F',
    'Fujichrome Provia 100F': 'FUJI_PROVIA_100F',
    'Fujichrome Provia 400 Professional': 'FUJI_PROVIA_400F',
    'Fujichrome Provia 400': 'FUJI_PROVIA_400F',
    'Fujichrome 64T Type II Professional': 'FUJICHROME_64T_II',
    'Fujichrome 64T Type II': 'FUJICHROME_64T_II',
    'Neopan 100 Acros': 'FUJI_NEOPAN_ACROS_100',
    'Neopan 100 Acros 135 формата': 'FUJI_NEOPAN_ACROS_100',
    'Neopan 100 Acros 120 формата': 'FUJI_NEOPAN_ACROS_100',
    'Neopan 100 Acros для всех форматов': 'FUJI_NEOPAN_ACROS_100',
    'Neopan 1600 Professional': 'FUJI_NEOPAN_1600',
    'Konica Color Centuria Super 400': 'KONICA_CENTURIA_SUPER_400',
    'Konica Color Centuria Super 1600': 'KONICA_CENTURIA_SUPER_1600',
    'Konica Chrome R-100': 'KONICA_CHROME_R100',
    'Konica Color Impresa 50 Professional': 'KONICA_IMPRESA_50',
    'Konica Color Impresa 50': 'KONICA_IMPRESA_50',
    'Konica Color VX 100': 'KONICA_VX_100',
    'Infrared 750': 'KONICA_INFRARED_750',
}

# Deliberately NOT mapped, with the reason, so the refusal is on the record and
# nobody has to re-derive it by hand later.
REFUSED_MAP = {
    'Ultra 100': 'the database holds AGFA ULTRA 50, a different speed',
    'Vista 100': 'no Vista 100 in the database',
    'Vista 400': 'the database holds AGFA VISTA PLUS 400, a later naming whose '
                 'emulsion identity with Vista 400 this source does not establish',
    'Vista 800': 'no Vista 800 in the database',
    'Ektachrome 100': 'the database entry KODAK EKTACHROME 100D 5285 is the '
                      'cine stock, not the still slide film this panel measures',
    'Kodachrome 25': 'no Kodachrome 25 in the database',
    'Kodachrome 200': 'no Kodachrome 200 in the database',
    'Konica Chrome Centuria 200': 'the database holds KONICA CHROME CENTURIA '
                                  '100, a different speed',
    'Fujicolor NPH 400 Professional': 'NPH 400 and the database FUJICOLOR PRO '
                                      '400H are separated by a renaming this '
                                      'source does not document',
    'Fujichrome Astia 100 Professional': 'no Astia in the database',
    'Agfapan APX 200 S': 'no APX 200 S in the database',
}


# ===========================================================================
# 1. figure index -- caption to bitmap
# ===========================================================================
CAPTION = re.compile(r'Рис\.\s*(\d+)\.(\d+)\s*\.?\s*(.{0,200})', re.S)
CARRIER = (r'(?:фотопленки|фотоплёнки|кинопленки|фотобумаги|фотоматериала|'
           r'материала|бумаги|пленки|плёнки|фотопленок)')
CAPTION_OFFSET = 45      # a caption block starts within this many pt of the
                         # bitmap it belongs to (measured: 12 pt, every time)
CAPTION_WRAP_GAP = 8.0   # a wrapped caption's second line starts this close
                         # below the first (measured: 0.0 pt, every time -- the
                         # blocks abut, and 8 pt is a full line of slack)
CAPTION_WRAP_MAX = 60    # and a continuation that is a FILM NAME is short; a
                         # longer block is body text and is not joined


def _kind(body):
    if re.search(r'Характеристические кривые', body): return 'char'
    if re.search(r'[Кк]ривые кинетики проявления', body): return 'kinet'
    if re.search(r'Функция передачи модуляции', body): return 'mtf'
    if re.search(r'Спектральное поглощение красител', body): return 'dyeabs'
    if re.search(r'Спектральная плотность красител', body): return 'dyeden'
    if re.search(r'Спектральная плотность фотопленки', body): return 'monospec'
    if re.search(r'[Кк]оррекци\w* длинных выдержек|График коррекции', body): return 'recip'
    if re.search(r'Строение', body): return 'struct'
    if re.search(r'Спектральная (чувствительность|сенсибилизация)', body): return 'spsens'
    return 'other'


def build_index(doc):
    """Every figure caption, tied to the bitmap it sits under."""
    rows = []
    for i in range(doc.page_count):
        page = doc[i]
        blocks = sorted((b[1], b[3], ' '.join(b[4].split()))
                        for b in page.get_text('blocks'))
        caps = []
        for bi, (by0, by1, btxt) in enumerate(blocks):
            m = CAPTION.search(btxt)
            if not m:
                continue
            # ⚠ A CAPTION THAT WRAPS LOSES THE FILM NAME, AND FOR ONE FIGURE
            # KIND IT LOSES IT ALMOST EVERY TIME (queue P53, 2026-09-17e).
            # «Спектральное поглощение красителями (оптическая плотность)
            # фотопленки» is long enough that the NAME lands in the next text
            # block, so 110 of 115 dye-absorption panels were indexed with
            # film=None and `harvest` skipped every one of them before it ever
            # reached a tracer. The rule is narrow on purpose: the carrier word
            # must END the block, and the continuation must be the very next
            # block, close below, and short enough to be a name rather than a
            # paragraph.
            if (re.search(CARRIER + r'\s*$', btxt) and bi + 1 < len(blocks)):
                ny0, _, ntxt = blocks[bi + 1]
                if ny0 - by1 < CAPTION_WRAP_GAP and len(ntxt) <= CAPTION_WRAP_MAX:
                    btxt = btxt + ' ' + ntxt
            caps.append((by0, int(m.group(1)), int(m.group(2)), btxt))
        if not caps:
            continue
        imgs = []
        for x in page.get_images(full=True):
            for r in page.get_image_rects(x[0]):
                imgs.append((r.y0, r.y1, x[0], x[2], x[3]))
        imgs.sort()
        used = set()
        for cy, ch, num, txt in sorted(caps):
            best, bd = None, 1e9
            for k, (_, iy1, _, _, _) in enumerate(imgs):
                if k in used:
                    continue
                d = abs(cy - iy1)
                if d < bd:
                    bd, best = d, k
            if best is None or bd > CAPTION_OFFSET:
                continue
            used.add(best)
            body = re.sub(r'^Рис\.\s*\d+\.\d+\s*\.?\s*', '', txt)
            m = re.search(CARRIER + r'\s+(.+)$', body)
            film = re.sub(r'\s+', ' ', m.group(1).strip().rstrip('.,;')) if m else None
            rows.append(dict(page=i + 1, ch=ch, num=num, kind=_kind(body),
                             film=film, xref=imgs[best][2], ordinal=best))
    return rows


def bitmap(doc, xref):
    pix = pymupdf.Pixmap(doc, xref)
    if pix.n - pix.alpha > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    return arr[:, :, 0]


# ===========================================================================
# 2. glyphs -- segmentation and the self-cut template bank
# ===========================================================================
def components(strip, dark=128):
    b = (strip < dark).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(st[i, k]) for k in range(5))
        if w > 42 or h > 28:
            continue
        out.append(dict(x=x, y=y, w=w, h=h, a=a,
                        bm=(lab[y:y + h, x:x + w] == i).astype(np.uint8)))
    out.sort(key=lambda c: c['x'])
    return out


def split_h(comps, gap=6):
    if not comps:
        return []
    groups = [[comps[0]]]
    for c in comps[1:]:
        if c['x'] - max(g['x'] + g['w'] for g in groups[-1]) <= gap:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def split_v(comps, gap=5):
    if not comps:
        return []
    cs = sorted(comps, key=lambda c: c['y'])
    groups = [[cs[0]]]
    for c in cs[1:]:
        if c['y'] - max(g['y'] + g['h'] for g in groups[-1]) <= gap:
            groups[-1].append(c)
        else:
            groups.append([c])
    for g in groups:
        g.sort(key=lambda c: c['x'])
    return groups


def normed(bm):
    r = cv2.resize(bm.astype(np.float32), NORM, interpolation=cv2.INTER_AREA)
    r = r - r.mean()
    n = np.linalg.norm(r)
    return r / n if n > 0 else r


def roles(group):
    """digit / minus / comma for each component of one printed number."""
    tall = [c for c in group if c['h'] >= 6]
    if not tall:
        return None
    y0 = min(c['y'] for c in tall)
    y1 = max(c['y'] + c['h'] for c in tall)
    dh = y1 - y0
    out = []
    for c in group:
        if c['h'] >= 0.55 * dh:
            out.append(('digit', c))
        else:
            rel = (c['y'] + c['h'] / 2.0 - y0) / max(1.0, dh)
            out.append(('comma' if rel > 0.72 else 'minus', c))
    return out


def band_filter(cs, axis):
    """Drop tick marks, the Cyrillic axis title and stray curve ends.

    Digits are 9-18 px tall and at most 14 wide; the axis titles are set
    smaller. On the abscissa two candidate bands can survive, and the one
    NEARER THE FRAME is the tick labels. On the ordinate the labels are
    right-aligned against the frame, so the rightmost 36 px is the label
    column and the rotated title falls outside it.
    """
    cand = [c for c in cs if 9 <= c['h'] <= 18 and 2 <= c['w'] <= 14]
    if not cand:
        return []
    if axis == 'x':
        cand.sort(key=lambda c: c['y'])
        bands, cur = [], [cand[0]]
        for c in cand[1:]:
            if c['y'] - cur[-1]['y'] <= 6:
                cur.append(c)
            else:
                bands.append(cur); cur = [c]
        bands.append(cur)
        bands = [b for b in bands if len(b) >= 2] or bands
        band = min(bands, key=lambda b: np.median([c['y'] for c in b]))
        lo = int(min(c['y'] for c in band)) - 4
        hi = int(max(c['y'] + c['h'] for c in band)) + 5
        return [c for c in cs if c['y'] >= lo and c['y'] + c['h'] <= hi]
    x1 = max(c['x'] + c['w'] for c in cand)
    return [c for c in cs if c['x'] + c['w'] >= x1 - 36]


def find_frame(im, dark=128):
    """The plot frame, tolerant of a border broken by the scan.

    ⚠ A BORDER THAT IS NOT CONTINUOUS IS STILL A BORDER, and requiring 55 % of
    a row to be ink lost whole panels to nothing worse than a faint scan --
    Fujichrome 64T Type II (p.269) among them, the last of queue P56's four.
    A rule is recognised by the LONGEST UNBROKEN RUN along it as well as by the
    total ink in it: a frame side is one continuous stroke over most of its
    length even when the scan has eaten pieces out of the rest of the row.
    """
    # ⚠ AND THE INK THRESHOLD IS A LADDER, NOT A CONSTANT. Some panels in this
    # book are printed with a GREY rule -- Fujichrome 64T Type II's frame sits
    # at levels 156-199 against a 128 cutoff -- so at `dark` the border is not
    # ink at all and the strongest row in the whole panel carries 37 % of the
    # width. Each step up is tried in turn and the FIRST that yields a frame is
    # taken, so a black-ruled panel is read exactly as before and a grey-ruled
    # one is read at all.
    for _cut in (dark, 170, 205):
        _f = _find_frame_at(im, _cut)
        if _f is not None:
            return _f
    return None


def _find_frame_at(im, dark):
    b = (im < dark).astype(np.uint8)
    h, w = b.shape

    def longest_run(v):
        best = cur = 0
        for q in v:
            cur = cur + 1 if q else 0
            best = max(best, cur)
        return best

    rc = [y for y in range(h)
          if b[y].sum() > 0.55 * w or longest_run(b[y]) > 0.45 * w]
    cc = [x for x in range(w)
          if b[:, x].sum() > 0.55 * h or longest_run(b[:, x]) > 0.45 * h]
    if not rc or not cc:
        return None

    def grp(a, tol=2):
        out, cur = [], [a[0]]
        for v in a[1:]:
            if v - cur[-1] <= tol:
                cur.append(v)
            else:
                out.append(cur); cur = [v]
        out.append(cur)
        return [int(round(np.mean(g))) for g in out]

    r, c = grp(rc), grp(cc)
    return dict(top=r[0], bot=r[-1], left=c[0], right=c[-1])


def strip_of(im, f, axis):
    h, w = im.shape
    if axis == 'x':
        yo, xo = f['bot'] + 4, max(0, f['left'] - 26)
        return im[yo:min(h, f['bot'] + 44), xo:min(w, f['right'] + 28)], (xo, yo)
    yo, xo = max(0, f['top'] - 12), max(0, f['left'] - 40)
    return im[yo:f['bot'] + 14, xo:max(1, f['left'] - 3)], (xo, yo)


def label_groups(im, f, axis):
    sub, org = strip_of(im, f, axis)
    cs = band_filter(components(sub), axis)
    return (split_h(cs) if axis == 'x' else split_v(cs)), org


class Bank:
    """The digit templates, cut from BANK_SOURCES in this same PDF."""

    def __init__(self, doc, index):
        store = {}
        want = {}
        for r in index:
            want[(r['page'], r['kind'], r['film'])] = r
        for page, kind, film, axis, truths in BANK_SOURCES:
            r = want.get((page, kind, film))
            if r is None:
                continue
            im = bitmap(doc, r['xref'])
            f = find_frame(im)
            if not f:
                continue
            groups, _ = label_groups(im, f, axis)
            groups = [g for g in groups if roles(g)]
            if len(groups) != len(truths):
                continue
            for g, t in zip(groups, truths):
                digits = [c for k, c in roles(g) if k == 'digit']
                chars = [ch for ch in t if ch.isdigit()]
                if len(digits) != len(chars):
                    continue
                for c, ch in zip(digits, chars):
                    store.setdefault(ch, []).append(normed(c['bm']))
        self.tpl = {k: np.stack(v) for k, v in store.items()}
        self.missing = [str(d) for d in range(10) if str(d) not in self.tpl]

    def digit(self, bm):
        v = normed(bm)
        ar = bm.shape[1] / max(1.0, bm.shape[0])
        sc = {}
        for d, stack in self.tpl.items():
            s = float(np.max(stack.reshape(len(stack), -1) @ v.reshape(-1)))
            s += 0.08 * (1.0 - min(1.0, abs(ar - _AR.get(d, ar)) / 0.45))
            sc[d] = s
        order = sorted(sc.items(), key=lambda kv: -kv[1])
        if len(order) < 2:
            return None
        (d0, s0), (_, s1) = order[0], order[1]
        if s0 < MATCH_MIN or (s0 - s1) < MATCH_MARGIN:
            return None
        return d0

    def label(self, group):
        r = roles(group)
        if not r:
            return None
        txt = ''
        for kind, c in r:
            if kind == 'digit':
                d = self.digit(c['bm'])
                if d is None:
                    return None
                txt += d
            elif kind == 'comma':
                txt += '.'
            elif not txt:
                txt = '-'          # a minus only ever precedes the digits
        if txt in ('', '-', '.', '-.'):
            return None
        try:
            return float(txt)
        except ValueError:
            return None


# ===========================================================================
# 3. axis calibration
# ===========================================================================
def _digits_of(value, places=1):
    """The multiset of decimal digits of a printed axis label, sign dropped.

    «-0,1» and «-1,0» both give ('0', '1'); «-0,1» and «-0,2» do not. That is
    the whole test the digit-swap repair below is allowed to make.
    """
    s = "%.*f" % (places, abs(float(value)))
    return tuple(sorted(ch for ch in s if ch.isdigit()))


def fit_axis(pairs, log=False):
    """value = a*px + b over the largest consistent subset of the labels."""
    if len(pairs) < AXIS_MIN_LABELS:
        return None
    v = np.array([p[0] for p in pairs], float)
    p = np.array([p[1] for p in pairs], float)
    if log:
        # ⚠⚠ A ZERO ON A LOG AXIS IS A MISREAD LABEL, NOT A REASON TO REFUSE
        # THE WHOLE PANEL, AND THE OLD `return None` COST A MEASUREMENT.
        # Found 2026-09-18c: p341's MTF abscissa runs 1 2 4 5 10 20 50 100 200
        # 500 and the bank reads that final «500» as «0» -- one glyph group
        # lost, one impossible value. Refusing the axis discarded the other TEN
        # labels and with them KODAK T-MAX 400's f50, which is the measurement
        # that broke a documented three-way Kodak conflict on that stock.
        # A logarithmic axis cannot carry a zero tick, so such a label is
        # known-bad by construction and dropping it keeps strictly more
        # information than dropping the panel. ⚠ THIS IS NOT A TOLERANCE:
        # only values a logarithm cannot take are removed, and what survives
        # still has to meet AXIS_MIN_LABELS and the residual gate below.
        _pos = v > 0
        if not _pos.all():
            if int(_pos.sum()) < AXIS_MIN_LABELS:
                return None
            v, p = v[_pos], p[_pos]
        v = np.log10(v)
    o = np.argsort(p)
    v, p = v[o], p[o]
    n = len(v)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            if p[j] - p[i] < 8:
                continue
            a = (v[j] - v[i]) / (p[j] - p[i])
            b = v[i] - a * p[i]
            span = float(np.ptp(v)) or 1.0
            keep = np.abs(a * p + b - v) <= AXIS_RESID_TOL * span
            if best is None or keep.sum() > best.sum():
                best = keep
    # ⚠⚠ THE DIGIT-SWAP REPAIR, AND IT RECOVERS FORTY-SEVEN PANELS (queue P56 /
    # P53, 2026-09-18b). One whole figure family in this book prints «-0,1»
    # where its own uniformly spaced ladder requires «-1,0» -- the same
    # misprint on every panel, a transposition around the comma. The bank reads
    # the glyphs correctly; the fit then drops the label as an outlier, which
    # leaves TWO labels on a three-label axis, and `AXIS_MIN_LABELS` refuses
    # the panel. The panel is perfectly legible and the axis is not in doubt.
    #
    # ⚠ THE REPAIR IS NOT A TOLERANCE AND MUST NEVER BECOME ONE. A dropped
    # label is repaired ONLY when the geometry predicts a value whose decimal
    # digits are a PERMUTATION of the digits actually printed -- so «-0,1» may
    # become «-1,0» and nothing may become anything else. A misread digit, a
    # mis-grouped label or a genuinely non-linear axis all fail that test,
    # because none of them produces an anagram of the right answer.
    if best is not None and best.sum() >= 2:
        _keep = best.copy()
        _a = (v[_keep][-1] - v[_keep][0]) / max(p[_keep][-1] - p[_keep][0], 1e-9)
        _b = float(v[_keep][0] - _a * p[_keep][0])
        for _i in range(n):
            if _keep[_i]:
                continue
            _pred = _a * p[_i] + _b
            if _digits_of(_pred) == _digits_of(v[_i]) and abs(_pred - v[_i]) > 1e-9:
                v[_i] = _pred
                _keep[_i] = True
        best = _keep
    if best is None or best.sum() < AXIS_MIN_LABELS or best.sum() < AXIS_MIN_KEEP * n:
        return None
    A = np.vstack([p[best], np.ones(int(best.sum()))]).T
    sol, *_ = np.linalg.lstsq(A, v[best], rcond=None)
    span = float(np.ptp(v[best])) or 1.0
    rel = float(np.max(np.abs(A @ sol - v[best]))) / span
    return dict(a=float(sol[0]), b=float(sol[1]), n=int(best.sum()), nread=n,
                log=log, resid=rel, ok=bool(rel <= AXIS_RESID_TOL))


def calibrate(im, bank, xlog=False, ylog=False):
    f = find_frame(im)
    if not f:
        return None
    out = {}
    for axis, log in (('x', xlog), ('y', ylog)):
        groups, org = label_groups(im, f, axis)
        pairs = []
        for g in groups:
            v = bank.label(g)
            if v is None:
                continue
            if axis == 'x':
                px = org[0] + float(np.mean([c['x'] + c['w'] / 2 for c in g]))
            else:
                px = org[1] + float(np.mean([c['y'] + c['h'] / 2 for c in g]))
            pairs.append((v, px))
        out[axis] = fit_axis(pairs, log)
    if not (out['x'] and out['y'] and out['x']['ok'] and out['y']['ok']):
        return None
    out['frame'] = f
    return out


# ===========================================================================
# 4. curve tracing
# ===========================================================================
def _thin_lines(idx, maxthick):
    """Keep only those indices whose contiguous run is at most ``maxthick``.

    A printed rule is one to three pixels thick; a traced curve lying flat is
    thicker. Used to stop the line-opening in `interior` from mistaking an
    MTF curve's 100 % plateau for a gridline -- see the note there.
    """
    out = set()
    run = []
    for i in sorted(idx) + [None]:
        if run and i is not None and i == run[-1] + 1:
            run.append(i)
            continue
        if run and len(run) <= maxthick:
            out.update(run)
        run = [] if i is None else [i]
    return out


def interior(im, f, dark=140, mend=True, grid=True):
    """The plot interior with the frame and any GRID removed, curves intact.

    ⚠⚠ THE GRID REMOVAL USED TO DESTROY THE FIGURE IT WAS CLEANING, and that is
    queue P53's fourth defect (found 2026-09-18). Blanking every row that is
    over 80 % ink removes the frame -- and on a panel drawn over a grid it
    removes the GRID too, which is correct, except that the grid CROSSES the
    curves. Each crossing left a one-pixel cut, so a curve a thousand pixels
    long arrived downstream as a dozen fragments and every component-based test
    rejected it: on p.139 the largest surviving piece was 221 px.
    ⚠ THE FIX IS TO MEND WHAT THE BLANKING CUT. A blanked row is one or two
    pixels tall and a curve is two to four wide, so closing ACROSS the blanked
    line in the perpendicular direction rejoins the curve and cannot bridge two
    different curves, which on these panels are tens of pixels apart. The mend
    is applied ONLY on the blanked lines, never over the whole mask, so no
    dashed stroke is silently closed along its own direction.
    """
    y0, y1 = f['top'] + 2, f['bot'] - 1
    x0, x1 = f['left'] + 2, f['right'] - 1
    sub = (im[y0:y1, x0:x1] < dark).astype(np.uint8)
    h, w = sub.shape
    # ⚠ THE LINES ARE FOUND BY OPENING, NOT BY COUNTING INK IN A ROW. A row
    # count only catches a gridline that runs the FULL width and is unbroken;
    # the same grid drawn a shade lighter, or interrupted where a curve crosses
    # it, falls under any threshold and survives as a one-pixel-wide component
    # that every downstream test then has to cope with. An opening with a long
    # line kernel finds a line by its GEOMETRY and catches both.
    lin_h = cv2.morphologyEx(sub, cv2.MORPH_OPEN,
                             np.ones((1, max(9, int(0.55 * w))), np.uint8))
    lin_v = cv2.morphologyEx(sub, cv2.MORPH_OPEN,
                             np.ones((max(9, int(0.55 * h)), 1), np.uint8))
    # ⚠ AND BOTH TESTS ARE KEPT, BECAUSE THEY CATCH DIFFERENT LINES. The
    # opening finds a faint or interrupted rule that no threshold on an ink
    # COUNT would reach; the count finds a rule so broken by curve crossings
    # that no single line kernel spans it. Either alone loses panels -- the
    # opening alone took p.139's largest surviving component DOWN from 800 px
    # to 202, because that panel's grid is cut at every crossing.
    # ⚠⚠ AND THE OPENING MUST BE THICKNESS-LIMITED OR IT EATS THE CURVE, which
    # cost a measured overshoot before it was caught (2026-09-18c). An MTF
    # curve is FLAT at 100 % across the low-frequency end of its own panel, so
    # a long horizontal line kernel finds it and the blanking deletes it: KODAK
    # TECHNICAL PAN's overshoot came back 0.1969 against the +15.1 % its own
    # book prints, and the mend was rebuilding a curve the grid removal had
    # just destroyed. A printed rule is ONE to THREE pixels thick and a traced
    # curve is three or more, so a candidate line that sits inside a thicker
    # band of ink is part of a curve and is left alone. The 80 %-ink test is
    # not filtered: a row that is four fifths ink across the full width is a
    # rule whatever its thickness.
    if not grid:
        # ⚠⚠ THE OPENING IS OFF FOR CONTRAST-TRANSFER PANELS, AND THE REASON IS
        # A MEASUREMENT IT DESTROYED. An MTF curve runs FLAT along its own
        # 100 % line at low frequency, and a long line kernel cannot tell that
        # plateau from a printed rule: on KODAK TECHNICAL PAN p372 the removal
        # cut the top of the curve and the tracer then latched onto the 120 %
        # gridline, reporting an overshoot of 0.1969 where the book prints
        # +15.1 %. The ink test alone reads 0.1564 and always did. ⚠ These
        # panels have no interior grid to remove in the first place, so the
        # opening was buying nothing here and costing a number.
        lin_h = np.zeros_like(sub)
        lin_v = np.zeros_like(sub)
    _oh = _thin_lines({y for y in range(h) if lin_h[y].any()}, 3)
    _ov = _thin_lines({x for x in range(w) if lin_v[:, x].any()}, 3)
    rows = sorted(_oh | {y for y in range(h) if sub[y].sum() > 0.80 * w})
    cols = sorted(_ov | {x for x in range(w) if sub[:, x].sum() > 0.80 * h})
    _keep_h = np.zeros(h, bool)
    _keep_h[list(_oh)] = True
    _keep_v = np.zeros(w, bool)
    _keep_v[list(_ov)] = True
    sub[(lin_h & _keep_h[:, None]) | (lin_v & _keep_v[None, :]) > 0] = 0
    for y in rows:
        sub[y] = 0
    for x in cols:
        sub[:, x] = 0
    if mend and (rows or cols):
        # vertical closing repairs a horizontal cut, horizontal closing a
        # vertical one; each is written back only into the blanked band, so a
        # dashed stroke is never closed along its own direction
        if rows:
            vert = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))
            for y in rows:
                sub[y] = np.maximum(sub[y], vert[y])
        if cols:
            horz = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8))
            for x in cols:
                sub[:, x] = np.maximum(sub[:, x], horz[:, x])
    return sub, (x0, y0)


def drop_text(sub, min_len=14):
    """Keep strokes, drop compact blobs -- the in-plot labels and legend keys."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    out = np.zeros_like(sub)
    for i in range(1, n):
        x, y, w, h, a = (int(st[i, k]) for k in range(5))
        if max(w, h) < min_len:
            continue
        if a > 0.55 * w * h and min(w, h) > 6:
            continue
        out[lab == i] = 1
    return out


def text_row_boxes(raw, min_members=5, min_span=55):
    """Zero the rows of PRINTED TEXT inside the plot.

    ⚠ `drop_text` REMOVES BLOBS AND A CAPTION IS NOT ONE BLOB. It is a row of
    them, and the taller Cyrillic letters clear the blob test individually, so a
    line like «Экспозиция: дневное освещение» survives into the curve mask and
    the tracer follows it -- on p.332 that produced a fourth 'curve' of gamma
    0.05 sitting at D 3.75 across four decades. What identifies text is not any
    one component but the ROW: five or more letter-sized components sharing a
    baseline over fifty-five pixels. No characteristic curve looks like that.

    ⚠ AND IT MUST BE COMPUTED ON THE RAW INK, NOT ON WHAT `drop_text` LEAVES.
    Run afterwards it sees the two or three letters that happened to survive,
    never reaches five members, and blanks nothing -- which is why the caption
    kept coming through. The rows are found on the full ink and the BOXES are
    then applied to the curve mask.
    """
    n, lab, st, _ = cv2.connectedComponentsWithStats(raw, 8)
    letters = []
    for i in range(1, n):
        x, y, w, h, a = (int(st[i, k]) for k in range(5))
        if 6 <= h <= 20 and 2 <= w <= 22 and a >= 0.22 * w * h:
            letters.append((y + h / 2.0, x, x + w, y, y + h))
    if not letters:
        return []
    letters.sort()
    boxes = []
    band, cur = [], [letters[0]]
    for L in letters[1:]:
        if L[0] - cur[-1][0] <= 5.0:
            cur.append(L)
        else:
            band.append(cur); cur = [L]
    band.append(cur)
    for b in band:
        if len(b) < min_members:
            continue
        x0 = min(q[1] for q in b); x1 = max(q[2] for q in b)
        if x1 - x0 < min_span:
            continue
        # ⚠ THE BASELINE TEST IS WHAT SEPARATES A CAPTION FROM A DASHED CURVE,
        # and without it this function deletes the curves it was written to
        # protect. Both are a row of small components; TEXT SITS ON A COMMON
        # BASELINE and a dashed curve steps steadily downward across the same
        # span. Requiring the bottom edges to agree within a couple of pixels
        # keeps every caption and releases every dash run.
        bots = np.array([q[4] for q in b], float)
        if float(np.std(bots)) > 2.5:
            continue
        y0 = min(q[3] for q in b); y1 = max(q[4] for q in b)
        boxes.append((max(0, x0 - 3), max(0, y0 - 3), x1 + 3, y1 + 3))
    return boxes


def drop_leaders(sub, boxes, resid=1.6, min_len=18, reach=14):
    """Delete the LEADER LINES that point from an in-plot label to its curve.

    ⚠ THE LABELS COME OFF AND THE LINES POINTING AT THEM DO NOT, which is the
    other half of queue P53's reader problem. `text_row_boxes` removes the
    words; each word on these panels is tied to its curve by a thin straight
    rule, and that rule survives into the curve mask, where it is long enough
    to pass every size test and steep enough to be mistaken for a crest by a
    flatness test -- measured on 7 of 19 Kodak-frame dye panels.

    A leader is identified by two properties together, neither sufficient
    alone: it is STRAIGHT along its whole length, which no spectral curve or
    characteristic curve is, and one of its ends REACHES a box that held text.
    """
    if not len(boxes):
        return sub
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    out = sub.copy()
    for i in range(1, n):
        x, y, w, h, a = (int(st[i, k]) for k in range(5))
        if max(w, h) < min_len:
            continue
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 8:
            continue
        # straightness: total least squares residual about the principal axis
        px = np.stack([xs - xs.mean(), ys - ys.mean()]).astype(float)
        u, s, _ = np.linalg.svd(px @ px.T)
        rms = float(np.sqrt(max(s[1], 0.0) / len(xs)))
        if rms > resid:
            continue
        near = False
        for bx0, by0, bx1, by1 in boxes:
            if (xs.min() <= bx1 + reach and xs.max() >= bx0 - reach
                    and ys.min() <= by1 + reach and ys.max() >= by0 - reach):
                near = True
                break
        if near:
            out[lab == i] = 0
    return out


def bridge_dashes(sub, k):
    """Rejoin a dashed stroke.

    Agfa and Konica draw the three records of a colour panel solid, dashed and
    dash-dot, so a dashed record presents as a broken column sequence. The
    kernel is wide and one pixel tall, so it closes along the stroke without
    merging records that are vertically apart.
    """
    return cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((1, k), np.uint8))


def column_runs(sub, max_run=14):
    h, w = sub.shape
    cols = []
    for x in range(w):
        col = sub[:, x]
        runs, y = [], 0
        while y < h:
            if col[y]:
                y0 = y
                while y < h and col[y]:
                    y += 1
                if 1 <= y - y0 <= max_run:
                    runs.append((y0 + y - 1) / 2.0)
            else:
                y += 1
        cols.append(runs)
    return cols


def trace(sub, n_curves, max_jump=7.0, min_cover=None):
    """Follow n non-crossing curves, seeded where they are furthest apart.

    The left end of a characteristic curve is where the toes crowd together and
    a dashed stroke drops out; a seed taken there assigns the records to the
    wrong strokes for the whole panel, so the seed is the column with n runs
    whose minimum separation is greatest, weighted towards the right half.
    """
    cols = column_runs(sub)
    w = len(cols)
    # ⚠ A SEED COLUMN NEED NOT HOLD EXACTLY n RUNS, AND REQUIRING THAT IS WHAT
    # LOST SIXTEEN PANELS. On the Agfa and Konica colour sheets the three
    # records are drawn solid, dashed and dash-dot, and a dashed record is
    # ABSENT from perhaps half the columns it crosses while a curve's own
    # anti-aliasing can split one stroke into two runs. So no column anywhere on
    # the panel shows exactly three, the seeder returned None, and a panel whose
    # curves are perfectly legible produced nothing at all.
    # What a seed actually has to be is a column where the n curves are present
    # and SEPARATED. A column with MORE than n runs is therefore usable: take
    # the n runs that are furthest apart and let the walker discard the rest.
    # The mask-ladder test downstream is unchanged and still refuses a seed that
    # picked the wrong n.
    def _pick(ys, n):
        if len(ys) == n:
            return list(ys)
        best_set, best_gap = None, -1.0
        for i in range(len(ys) - n + 1):
            cand = ys[i:i + n]
            g = float(np.min(np.diff(cand))) if n > 1 else 1.0
            if g > best_gap:
                best_gap, best_set = g, list(cand)
        return best_set

    seed, best, start = None, -1.0, None
    for x in range(w):
        if len(cols[x]) < n_curves:
            continue
        ys = _pick(sorted(cols[x]), n_curves)
        sep = float(np.min(np.diff(ys))) if n_curves > 1 else 1.0
        # exact-count columns are still preferred; an over-full column is
        # accepted at a discount so it only wins when nothing cleaner exists
        if len(cols[x]) != n_curves:
            sep *= 0.75
        sep *= 0.60 + 0.40 * x / max(1, w - 1)
        if sep > best:
            best, seed, start = sep, x, ys
    if seed is None:
        return None
    out = [{seed: start[k]} for k in range(n_curves)]
    for direction in (1, -1):
        p = list(start)
        v = [0.0] * n_curves
        miss = [0] * n_curves
        x = seed + direction
        while 0 <= x < w:
            runs = sorted(cols[x])
            taken = set()
            for k in range(n_curves):
                pred = p[k] + v[k]
                gate = max_jump + 1.3 * miss[k]
                pick, pd = None, 1e9
                for j, ry in enumerate(runs):
                    if j in taken:
                        continue
                    d = abs(ry - pred)
                    if d < pd and d <= gate:
                        pd, pick = d, j
                if pick is None:
                    miss[k] += 1
                    continue
                taken.add(pick)
                step = (runs[pick] - p[k]) / max(1, miss[k] + 1)
                v[k] = 0.55 * v[k] + 0.45 * step
                p[k] = runs[pick]
                miss[k] = 0
                out[k][x] = p[k]
            x += direction
    cover = min(len(o) for o in out) / float(w)
    if cover < (TRACE_MIN_COVER if min_cover is None else min_cover):
        return None
    return out, cover


def to_values(track, org, cal):
    xs = np.array(sorted(track))
    ys = np.array([track[x] for x in xs], float)
    xv = cal['x']['a'] * (xs + org[0]) + cal['x']['b']
    yv = cal['y']['a'] * (ys + org[1]) + cal['y']['b']
    if cal['x']['log']:
        xv = 10.0 ** xv
    if cal['y']['log']:
        yv = 10.0 ** yv
    o = np.argsort(xv)
    return xv[o], yv[o]


def base_density(yv):
    """The toe plateau -- the leftmost eighteenth of the traced curve."""
    return float(np.median(yv[:max(3, len(yv) // 18)]))


def f50_of(xv, yv):
    for i in range(1, len(yv)):
        if yv[i - 1] > 50.0 >= yv[i]:
            lx0, lx1 = np.log10(max(xv[i - 1], 1e-6)), np.log10(max(xv[i], 1e-6))
            t = (50.0 - yv[i - 1]) / (yv[i] - yv[i - 1])
            return float(10.0 ** (lx0 + t * (lx1 - lx0)))
    return None


def overshoot_of(xv, yv):
    """The adjacency peak, as the corpus stores it: a fraction above unity.

    These panels are contrast-transfer curves, so a development adjacency edge
    effect shows as the response rising ABOVE 100 % at low frequency before it
    falls. The corpus stores that peak in `MTFSpec.adjacency` and fits the
    rolloff above it -- see the +5.5 / +11.1 / +14.9 % entries already in
    film_profiles.py -- rather than folding it into f50.
    """
    y = np.asarray(yv, float)
    x = np.asarray(xv, float)
    if len(y) < 10:
        return 0.0, None
    k = int(np.argmax(y))
    peak = float(y[k])
    if peak <= 102.0:            # flat to within the trace's own noise
        return 0.0, None
    return (peak - 100.0) / 100.0, float(x[k])


def rolloff_q(xv, yv, f50, above=None):
    """Fit the corpus's MTF law M(f) = 1 / (1 + (f/f50)^q).

    f50 is already fixed by the crossing, so the law passes through 0.5 there
    by construction and the fit has ONE free parameter. Taking logs makes it a
    slope through the origin: log((100/M) - 1) = q * log(f/f50). Where the
    panel shows an adjacency peak the fit is taken ABOVE that peak only, which
    is the same convention the corpus already applies to its CTF panels -- the
    carrier is 1.0 at zero frequency by construction and cannot represent a
    curve that starts above it.
    """
    x = np.asarray(xv, float)
    y = np.asarray(yv, float)
    m = (y > 2.0) & (y < 98.0) & (x > 0)
    if above:
        m &= x >= above
    if m.sum() < 8:
        return None, None
    u = np.log(x[m] / f50)
    v = np.log(100.0 / y[m] - 1.0)
    keep = np.abs(u) > 1e-3
    if keep.sum() < 6:
        return None, None
    u, v = u[keep], v[keep]
    q = float(np.sum(u * v) / np.sum(u * u))
    pred = 100.0 / (1.0 + (x[m][keep] / f50) ** q)
    rms = float(np.sqrt(np.mean((pred - y[m][keep]) ** 2)))
    return q, rms


# ===========================================================================
# 4b. development families off the MONOCHROME characteristic panels
#
# ⚠ THIS IS WHAT CLOSED THE DEVELOPMENT GAP, AND IT IS NOT THE PANEL ANYONE
# EXPECTED. The 42 «Кривые кинетики проявления» panels plot contrast against
# time directly, which looks like the obvious source, and their curves cannot be
# attributed: the developer names sit in a dashed-line legend or in labels
# dropped over crossing strokes, and a DevelopmentPoint carries a developer
# name.
#
# The MONOCHROME CHARACTERISTIC panels give the same information in a form that
# attributes itself. Each plots several curves for ONE developer at ONE
# temperature, names the condition in a caption inside the frame -- «Проявление:
# малый бак, T-MAX, 20°C» -- and lists the DEVELOPMENT TIMES in a legend:
# 11 / 9 / 7 / 6 мин. Assignment is then physics rather than pattern matching:
# DEVELOPMENT TIME AND CONTRAST INCREASE TOGETHER, so the steepest curve is the
# longest time. The reader sorts the traced curves by gamma, sorts the legend
# times, pairs them, and REFUSES the panel unless the pairing is strictly
# monotonic -- which a mis-traced or mis-read panel will not be.
#
# Each panel therefore yields, per curve: the development time, the contrast the
# book measured at it, and the base+fog at that same condition -- the third of
# which `DevelopmentPoint.fog_at_dev` was added for and which no source in this
# corpus had populated.
# ===========================================================================
TIME_RE = re.compile(r'(\d{1,3})(?:[.,](\d))?\s*(?:мин|мип|мин\.|m[ий]n)', re.I)
TEMP_RE = re.compile(r'(\d{2})\s*[°*o]?\s*[CСc]')
DEV_LINE = re.compile(r'Проявл[ен]\w*\s*[:;]\s*(.+)')

# Cyrillic glyphs tesseract returns for digits it half-recognises in this font.
_DIGIT_FIX = {'Э': '9', 'З': '3', 'О': '0', 'о': '0', 'б': '6', 'В': '8',
              'Ч': '4', 'з': '3', 'І': '1', 'l': '1', 'I': '1'}

DEV_NAMES = ['T-MAX RS', 'T-MAX', 'D-76', 'D-19', 'D-23', 'DK-50', 'DEKTOL',
             'HC-110', 'XTOL', 'MICRODOL-X', 'TECHNIDOL', 'RODINAL', 'REFINAL',
             'STUDIONAL', 'ID-11', 'PERCEPTOL', 'MICROPHEN', 'ILFOTEC',
             'ATOMAL', 'NEOFIN', 'ID-68', 'FG-7', 'ACUFINE',
             'DIAFINE', 'HC-110 (Dil A)', 'HC-110 (Dil B)']
TANK_WORDS = {'малый бак': 'small tank', 'поддон': 'tray',
              'роторно-барабанный процессор': 'rotary drum processor',
              'большой бак': 'large tank', 'машинная': 'machine'}


def _fixdigits(t):
    return ''.join(_DIGIT_FIX.get(c, c) for c in t)


def plot_text(im, f):
    """Every line of text printed INSIDE the plot frame, read one at a time.

    OCR of the whole interior at once is unusable here: the curves themselves
    are read as characters and the legend's dash swatches merge into the words
    beside them. So the text is ISOLATED FIRST -- compact blobs are text, long
    thin strokes are curves -- then grouped into lines and each line is enlarged
    and read on its own. That is the difference between «Эмин» and «9 мин.»
    """
    y0, y1 = f['top'] + 2, f['bot'] - 1
    x0, x1 = f['left'] + 2, f['right'] - 1
    ink = (im[y0:y1, x0:x1] < 140).astype(np.uint8)
    h, w = ink.shape
    for y in range(h):
        if ink[y].sum() > 0.80 * w:
            ink[y] = 0
    for x in range(w):
        if ink[:, x].sum() > 0.80 * h:
            ink[:, x] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    txt = np.zeros_like(ink)
    for i in range(1, n):
        x, y, ww, hh, a = (int(st[i, k]) for k in range(5))
        if ww > 46 and hh <= 3:          # a legend dash swatch
            continue
        if max(ww, hh) >= 16 and not (a > 0.55 * ww * hh and min(ww, hh) > 6):
            continue                      # a curve stroke
        txt[lab == i] = 1
    lines = []
    d = cv2.dilate(txt, np.ones((1, 11), np.uint8))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(d, 8)
    for i in range(1, n2):
        x, y, ww, hh, a = (int(st2[i, k]) for k in range(5))
        if ww < 16 or hh < 7 or hh > 26:
            continue
        crop = 255 - (txt[y:y + hh, x:x + ww] * 255).astype(np.uint8)
        big = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 24, 24, 24, 24, cv2.BORDER_CONSTANT,
                                 value=255)
        for cfg in ('--psm 7 -l rus+eng', '--psm 7 -l eng'):
            t = pytesseract.image_to_string(big, config=cfg).strip()
            if t:
                lines.append(t)
                break
    return lines


def caption_lines(im, f):
    """The «Проявление:» / «Экспозиция:» caption, read as a whole block.

    The caption sits above the curves with clear space around it, so a single
    psm-6 pass over the interior reads it cleanly. The LEGEND does not survive
    that pass -- see `legend_times` -- but the caption always does.
    """
    y0, y1 = f['top'] + 2, f['bot'] - 1
    x0, x1 = f['left'] + 2, f['right'] - 1
    big = cv2.resize(im[y0:y1, x0:x1], None, fx=3, fy=3,
                     interpolation=cv2.INTER_CUBIC)
    txt = pytesseract.image_to_string(big, config='--psm 6 -l rus+eng')
    return [l.strip() for l in txt.splitlines() if l.strip()]


def legend_times(im, f):
    """Development times, read off the legend ROW BY ROW.

    ⚠ THE LEGEND CANNOT BE READ WITH THE REST OF THE PANEL. Its entries sit on
    top of the curves, and a whole-frame OCR returns the strokes as characters
    and merges each dash swatch into the word beside it -- «9 мин.» comes back
    as «Эмин», «11 мин.» as «И мин». What makes the legend findable is the
    SWATCH: every entry begins with a short horizontal rule, solid or dashed, at
    a common left edge. The reader locates those rules, groups them into rows,
    and reads only the strip to the RIGHT of each one. Digits then decode
    against the same template bank the axes use, which does not confuse 9 with
    the Cyrillic Э because it has never seen a letter.
    """
    y0, y1 = f['top'] + 2, f['bot'] - 1
    x0, x1 = f['left'] + 2, f['right'] - 1
    ink = (im[y0:y1, x0:x1] < 140).astype(np.uint8)
    h, w = ink.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    segs = []
    for i in range(1, n):
        x, y, ww, hh, a = (int(st[i, k]) for k in range(5))
        if hh <= 4 and 5 <= ww <= 130 and a >= 0.5 * ww * max(1, hh):
            segs.append((y + hh / 2.0, x, x + ww))
    if not segs:
        return [], None
    segs.sort()
    bands, cur = [], [segs[0]]
    for sg in segs[1:]:
        if sg[0] - cur[-1][0] <= 3.5:
            cur.append(sg)
        else:
            bands.append(cur); cur = [sg]
    bands.append(cur)
    rows = []
    for b in bands:
        span = max(e for _, _, e in b) - min(x for _, x, _ in b)
        if span < 18:                       # too short to be a legend rule
            continue
        yy = int(np.mean([q[0] for q in b]))
        xr = int(max(e for _, _, e in b))
        xl = int(min(q[1] for q in b))
        rows.append((yy, xr, xl))
    if len(rows) < 2:
        return [], None
    # A legend is a COLUMN: its rules share a right edge to within a few pixels.
    xs = np.array([r[1] for r in rows], float)
    med = float(np.median(xs))
    rows = [r for r in rows if abs(r[1] - med) <= 12]
    if len(rows) < 2:
        return [], None
    out = []
    for yy, xr, _xl in rows:
        a0, a1 = max(0, yy - 9), min(h, yy + 10)
        b0, b1 = xr + 2, min(w, xr + 96)
        if b1 - b0 < 10:
            continue
        crop = im[y0 + a0:y0 + a1, x0 + b0:x0 + b1]
        big = cv2.resize(crop, None, fx=7, fy=7, interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 22, 22, 22, 22, cv2.BORDER_CONSTANT,
                                 value=255)
        best = None
        for cfg in ('--psm 7 -c tessedit_char_whitelist=0123456789.,',
                    '--psm 7 -l rus'):
            t = pytesseract.image_to_string(big, config=cfg).strip()
            t = _fixdigits(t).replace(',', '.')
            m = re.search(r'(\d{1,3})(?:\.(\d))?', t)
            if m:
                v = float(m.group(1)) + (float(m.group(2)) / 10.0
                                         if m.group(2) else 0.0)
                if 0.5 <= v <= 90.0:
                    best = v
                    break
        if best is not None:
            out.append(best)
    seen, uniq = set(), []
    for v in out:
        if v in seen:
            continue
        seen.add(v); uniq.append(v)
    # The box the legend occupies, in INTERIOR coordinates, so the tracer
    # can blank it. A legend rule is a long horizontal stroke and the tracer
    # would otherwise follow one as if it were a characteristic curve -- which
    # is exactly what produced a 'base density' of 2.67 on the first run.
    ys = [r[0] for r in rows]
    # ⚠ THE BOX IS THE WHOLE LEGEND CARTOUCHE, NOT JUST ITS RULES. Most of
    # these legends are drawn inside a RECTANGLE, with a heading above the
    # entries («Условное обозначение»), and the rectangle's horizontal borders
    # are long clean strokes that the tracer follows as characteristic curves --
    # on p.332 that produced a 'curve' of gamma 0.05 sitting at D 2.97 across
    # four decades, and on p.333 two tracks locked onto the same border. The
    # margins below reach past the heading and past both borders.
    box = (max(0, int(min(r[2] for r in rows)) - 40),
           max(0, int(min(ys)) - 40),
           min(w, int(max(r[1] for r in rows)) + 130),
           min(h, int(max(ys)) + 26))
    return uniq, box


def _latinise(t):
    """Undo the Cyrillic/Latin homoglyph substitutions this OCR makes.

    The captions are set in a Latin face inside Russian text, so tesseract
    returns «Т-МАХ» for T-MAX and «0-76» or «Д-76» for D-76. Every character
    here looks identical in the two alphabets; none of them changes a Russian
    word this reader cares about, because the only Russian it matches is
    «Проявление» and the tank words, which are matched before this runs.
    """
    m = {'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H', 'К': 'K', 'М': 'M',
         'О': 'O', 'Р': 'P', 'Т': 'T', 'Х': 'X', 'У': 'Y', 'Ь': 'b',
         'а': 'a', 'с': 'c', 'е': 'e', 'о': 'o', 'р': 'p', 'х': 'x',
         'Д': 'D', 'Г': 'T', 'І': 'I', '—': '-', '–': '-', '−': '-'}
    return ''.join(m.get(c, c) for c in t)


def _match_dev(body):
    """Longest developer name present in the caption, after homoglyph repair."""
    up = _latinise(body).upper().replace(' ', '')
    up = up.replace('0-76', 'D-76').replace('0-19', 'D-19')
    for d in sorted(DEV_NAMES, key=len, reverse=True):
        if not d:
            continue
        if d.upper().replace(' ', '') in up:
            return d
    return None


def read_condition(lines):
    """(developer, tank, celsius) from the panel's own «Проявление:» caption."""
    dev = tank = None
    celsius = None
    for l in lines:
        m = DEV_LINE.search(l)
        if not m:
            continue
        body = m.group(1)
        t = TEMP_RE.search(_fixdigits(_latinise(body)))
        if t and 10 <= int(t.group(1)) <= 40:
            celsius = float(t.group(1))
        dev = dev or _match_dev(body)
        low = body.lower()
        for k, v in TANK_WORDS.items():
            if k in low:
                tank = v
                break
        if dev and celsius:
            break
    return dev, tank, celsius


def read_times(lines):
    """Development times, in minutes, from the panel's legend."""
    out = []
    for l in lines:
        for m in TIME_RE.finditer(_fixdigits(l)):
            v = float(m.group(1)) + (float(m.group(2)) / 10.0 if m.group(2) else 0.0)
            if 0.5 <= v <= 90.0:
                out.append(v)
    # one panel lists each time once; duplicates are OCR echoes of one line
    seen, uniq = set(), []
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        uniq.append(v)
    return uniq


def gamma_of(xv, yv, window=GAMMA_WINDOW):
    """Steepest sustained slope over a 0.60 log-E window -- the straight line.

    A 0.60 decade window is two stops, wide enough that a single traced pixel
    cannot set the slope and narrow enough to stay off the toe and the shoulder.
    """
    best = 0.0
    x = np.asarray(xv, float)
    y = np.asarray(yv, float)
    for i in range(len(x)):
        j = int(np.searchsorted(x, x[i] + window))
        if j >= len(x):
            break
        g = (y[j] - y[i]) / (x[j] - x[i])
        if g > best:
            best = float(g)
    return best


def dev_family(im, f, cal, n_curves):
    """One monochrome characteristic panel -> a development family."""
    lines = caption_lines(im, f)
    dev, tank, celsius = read_condition(lines)
    times, box = legend_times(im, f)
    if not times or celsius is None or dev is None:
        return None
    raw, org = interior(im, f)
    sub = drop_text(raw)
    for tx0, ty0, tx1, ty1 in text_row_boxes(raw):
        sub[ty0:ty1, tx0:tx1] = 0
    if box is not None:
        bx0, by0, bx1, by1 = box
        sub[max(0, by0):by1, max(0, bx0):bx1] = 0
    got = None
    for k in (1, 3, 5, 7):
        s = sub if k == 1 else bridge_dashes(sub, k)
        # ⚠ A LOWER COVERAGE FLOOR THAN THE SINGLE-CURVE PATH, AND IT IS NOT A
        # RELAXATION OF THE STANDARD. A development family's curves start at the
        # toe and the frame runs a decade further left, so four curves never
        # span the panel; what has to be covered is the STRAIGHT LINE, which is
        # where gamma is measured. The real check on this panel is not coverage
        # at all -- it is that the four gammas come out strictly ordered with
        # the four printed times, which a partial or crossed trace will fail.
        got = trace(s, n_curves or len(times), min_cover=FAMILY_MIN_COVER)
        if got:
            break
    if not got:
        return None
    tracks, cover = got
    rows = []
    for t in tracks:
        xv, yv = to_values(t, org, cal)
        # ⚠ FOG IS THE LOW PLATEAU, NOT THE LEFTMOST SAMPLE. A track can begin
        # on a stray -- a caption stroke, the frame edge -- and the leftmost
        # point then reports a base density of 3.6 on a film whose base is 0.25.
        # A characteristic curve is monotone, so the minimum density IS at the
        # left, and the median of the five lowest samples in the left 40 % is
        # that minimum without being one pixel's opinion of it.
        lo = np.sort(yv[:max(5, int(0.40 * len(yv)))])[:5]
        rows.append(dict(gamma=gamma_of(xv, yv), fog=float(np.median(lo)),
                         dmax=float(np.median(np.sort(yv)[-max(3, len(yv) // 18):])),
                         logE=(float(xv.min()), float(xv.max()))))
    if len(rows) != len(times):
        return None
    # ⚠ A PIXEL-COVERAGE FLOOR IS THE WRONG TEST FOR THESE PANELS AND THIS IS
    # THE RIGHT ONE. The frame runs five decades of log E and a development
    # family occupies about one of them, so three good curves cover 20 % of the
    # width and a fourth-order fraction says nothing. What gamma actually needs
    # is EXPOSURE SPAN: the slope is measured over a 0.60-decade window, so a
    # curve must carry at least 0.80 decades for that window to sit inside the
    # traced part rather than run off its end.
    if any((r['logE'][1] - r['logE'][0]) < FAMILY_MIN_DECADES for r in rows):
        return None
    rows.sort(key=lambda r: r['gamma'])
    ts = sorted(times)
    # THE CROSS-CHECK: contrast must rise strictly with development time, and
    # base+fog must not fall. A mis-traced or mis-read panel fails both.
    gs = [r['gamma'] for r in rows]
    if any(b <= a for a, b in zip(gs, gs[1:])):
        return None
    if gs[0] <= 0.05 or gs[-1] > 6.0:
        return None
    # ⚠ A PHYSICAL BOUND ON BASE+FOG, WHICH IS WHAT CATCHES A STRAY TRACK. A
    # monochrome camera film's base plus fog runs 0.05 to 0.60 D; anything
    # outside that is not a plateau this reader found, it is a stroke it
    # mistook for one. And fog RISES with development -- it never falls by more
    # than the trace's own noise -- which is the second thing a crossed pair of
    # tracks gets wrong.
    fo = [r['fog'] for r in rows]
    if min(fo) < 0.03 or max(fo) > 0.70:
        return None
    if any(b < a - 0.10 for a, b in zip(fo, fo[1:])):
        return None
    for r, t in zip(rows, ts):
        r['minutes'] = t
    return dict(developer=dev, tank=tank, celsius=celsius, cover=cover,
                points=rows, times=ts, lines=lines)


# ===========================================================================
# 4c. the dye panels
#
# ⚠ THE BLOCKER RECORDED AGAINST THESE WAS WRONG, AND IT WAS WRONG IN THE SAME
# WAY THE FIRST REVIEW OF THE WHOLE BOOK WAS. DIGITIZATION_QUEUE P50 said the
# 232 dye panels could not be stored because "dye_matrix is a 3x3 of coupling
# coefficients, not a spectrum, and no published calibration turns one into the
# other". Both halves are true and the conclusion does not follow: the database
# has held `SpectralDyeDensity` since schema v7, twenty-nine stocks already
# carry one, and `dye_matrix_from_spectra.py` has derived crosstalk from such
# curves since 2026-08-31. The carrier existed; the row asserted it did not.
#
# The book prints the two panel shapes the schema already knows:
#
#   «Спектральное поглощение красителями» -- THREE curves, labelled by the
#   SENSITISED LAYER (Синий / Зеленый / Красный). The dye each layer forms is
#   its complement, so blue -> yellow, green -> magenta, red -> cyan, and the
#   three go to `d_yellow` / `d_magenta` / `d_cyan`.
#
#   «Спектральная плотность красителей» -- TWO curves, D-min and D-nom. That is
#   the shape schema v14 added `d_dmin` for, and on a masked negative D-min IS
#   THE ORANGE MASK measured spectrally.
#
# Records are resampled onto the corpus's 10 nm grid, which is COARSER than the
# trace everywhere, so it decimates and never interpolates upward.
# ===========================================================================
SPECTRAL_GRID_NM = 10.0
SPECTRAL_MIN_NM, SPECTRAL_MAX_NM = 400.0, 700.0
SPECTRAL_MIN_SPAN = 240.0     # a panel must cover this much of the visible


def _resample(xv, yv, lo, hi, step):
    """Decimate a traced curve onto the corpus's wavelength grid."""
    grid = np.arange(lo, hi + 0.5 * step, step)
    o = np.argsort(xv)
    x, y = np.asarray(xv, float)[o], np.asarray(yv, float)[o]
    if x[0] > lo + step or x[-1] < hi - step:
        return None
    keep = np.concatenate(([True], np.diff(x) > 1e-9))
    return tuple(round(float(v), 4) for v in np.interp(grid, x[keep], y[keep]))


def dye_panel(im, f, cal, kind):
    """One dye panel -> the traces `SpectralDyeDensity` is shaped to hold."""
    lo_nm = cal['x']['a'] * (f['left'] + 2) + cal['x']['b']
    hi_nm = cal['x']['a'] * (f['right'] - 1) + cal['x']['b']
    # ⚠ THE LOWER BOUND WAS 300 nm AND IT REFUSED FIFTY LEGIBLE PANELS (queue
    # P53, 2026-09-17e). The book prints dye panels in two frames: an Agfa /
    # Konica one running 350-750 nm, and a Kodak one running 250-750. Measured
    # across the 115: the second frame calibrates to 249-251 nm at the left
    # edge on every one of the fifty panels that carry it, with an axis
    # residual under 0.5 %. A frame that starts at 250 nm is not a bad
    # calibration, it is a wider plot, and the gate has to admit it.
    if not (235.0 <= min(lo_nm, hi_nm) <= 460.0 and 620.0 <= max(lo_nm, hi_nm) <= 820.0):
        return None
    n = 3 if kind == 'dyeabs' else 2
    raw, org = interior(im, f)
    boxes = text_row_boxes(raw)
    sub = drop_text(raw)
    for tx0, ty0, tx1, ty1 in boxes:
        sub[ty0:ty1, tx0:tx1] = 0
    sub = drop_leaders(sub, boxes)
    got = None
    for k in (1, 3, 5, 7):
        t = sub if k == 1 else bridge_dashes(sub, k)
        got = trace(t, n, min_cover=0.55)
        if got:
            break
    if not got:
        return None
    curves = []
    for t in got[0]:
        xv, yv = to_values(t, org, cal)
        if float(xv.max() - xv.min()) < SPECTRAL_MIN_SPAN:
            return None
        g = _resample(xv, yv, SPECTRAL_MIN_NM, SPECTRAL_MAX_NM, SPECTRAL_GRID_NM)
        if g is None:
            return None
        curves.append((float(np.mean(yv)), g, xv, yv))
    # THE ASSIGNMENT, AND IT IS PHYSICS IN BOTH CASES.
    if n == 2:
        # D-nom lies above D-min at every wavelength, by construction: the
        # nominal density INCLUDES the base. Order by mean density.
        curves.sort(key=lambda q: q[0])
        dmin_t, dnom_t = curves[0][1], curves[1][1]
        if not all(b >= a - 0.05 for a, b in zip(dmin_t, dnom_t)):
            return None
        return dict(shape='neutral_pair', d_dmin=dmin_t, d_neutral=dnom_t)
    # Three dyes: each is named by WHERE IT ABSORBS. Yellow peaks blue,
    # magenta green, cyan red -- so the record is assigned by the position of
    # its own maximum, not by trace order, and a panel whose three maxima do
    # not fall in three different thirds of the visible is refused.
    grid = np.arange(SPECTRAL_MIN_NM, SPECTRAL_MAX_NM + 1, SPECTRAL_GRID_NM)
    peaks = [float(grid[int(np.argmax(q[1]))]) for q in curves]
    order = np.argsort(peaks)
    p_lo, p_mid, p_hi = (peaks[i] for i in order)
    if not (400.0 <= p_lo <= 500.0 and 500.0 < p_mid <= 600.0
            and 590.0 < p_hi <= 700.0):
        return None
    if min(p_mid - p_lo, p_hi - p_mid) < 40.0:
        return None
    return dict(shape='three_dye',
                d_yellow=curves[order[0]][1],
                d_magenta=curves[order[1]][1],
                d_cyan=curves[order[2]][1],
                peaks=(p_lo, p_mid, p_hi))


# ===========================================================================
# 5. the harvest
# ===========================================================================
def harvest(doc, index, bank):
    ladders, mtfs, kinetics, families, dyes = {}, {}, [], {}, {}
    for r in index:
        stock = STOCK_MAP.get(r['film'] or '')
        if not stock:
            continue
        if r['kind'] == 'char':
            im = bitmap(doc, r['xref'])
            cal = calibrate(im, bank)
            if not cal:
                continue
            fam = dev_family(im, cal['frame'], cal, None)
            if fam:
                families.setdefault(stock, []).append(dict(page=r['page'], **fam))
            raw, org = interior(im, cal['frame'])
            boxes = text_row_boxes(raw)
            got = None
            # ⚠ BRIDGE THE DASHES BEFORE DROPPING THE BLOBS, NEVER AFTER, AND
            # THIS ORDERING IS THE WHOLE OF QUEUE P55 (2026-09-17e). `drop_text`
            # deletes any component whose longest side is under `min_len` = 14
            # px, and ONE DASH OF A FLAT PLATEAU IS 8 px LONG. So on a colour
            # panel the dashed and dash-dot records were deleted wherever they
            # ran level -- which is exactly the D-min plateau the ladder is read
            # from -- while the solid record survived intact. The measurement
            # that opened P55 ("304 of 374 columns show a single run") was
            # therefore reading the mask this function had emptied, and the row's
            # diagnosis, that the three records COINCIDE, was wrong: on Optima
            # 100 they are 0.93 / 0.69 / 0.28 apart and never touch. Closing the
            # dashes first turns each broken plateau back into one long stroke,
            # which then clears `min_len` on its own merit.
            for k in (1, 3, 5, 7, 9, 11):
                s = drop_text(raw if k == 1 else bridge_dashes(raw, k))
                for tx0, ty0, tx1, ty1 in boxes:
                    s[ty0:ty1, tx0:tx1] = 0
                got = trace(s, 3)
                if got:
                    vv = [to_values(got[0][j], org, cal) for j in range(3)]
                    d = [base_density(v[1]) for v in vv]
                    if (d[0] > d[1] > d[2] and d[0] - d[2] >= LADDER_MIN
                            and d[0] <= DMIN_MAX and d[2] >= DMIN_MIN):
                        # the same three estimators `dev_family` uses, so a
                        # colour ladder and a development family report a Dmax
                        # and a gamma that mean the same thing
                        dmx = [float(np.median(np.sort(v[1])[-max(3, len(v[1]) // 18):]))
                               for v in vv]
                        gam = [gamma_of(*v) for v in vv]
                        span = min(float(v[0].max() - v[0].min()) for v in vv)
                        ladders.setdefault(stock, dict(
                            page=r['page'], cover=round(got[1], 3), bridge=k,
                            b=round(d[0], 4), g=round(d[1], 4), r=round(d[2], 4),
                            dmax_b=round(dmx[0], 4), dmax_g=round(dmx[1], 4),
                            dmax_r=round(dmx[2], 4),
                            gam_b=(round(gam[0], 4) if gam[0] else None),
                            gam_g=(round(gam[1], 4) if gam[1] else None),
                            gam_r=(round(gam[2], 4) if gam[2] else None),
                            span=round(span, 2)))
                        break
                    got = None
        elif r['kind'] == 'mtf':
            im = bitmap(doc, r['xref'])
            cal = calibrate(im, bank, xlog=True, ylog=True)
            if not cal:
                continue
            # grid=False: see the note in `interior`. A contrast-transfer
            # panel's own 100 % plateau is the thing the line opening mistakes
            # for a rule.
            sub, org = interior(im, cal['frame'], grid=False, mend=False)
            sub = drop_text(sub)
            got = trace(sub, 1) or trace(bridge_dashes(sub, 5), 1)
            if not got:
                continue
            xv, yv = to_values(got[0][0], org, cal)
            f = f50_of(xv, yv)
            if f is None or not (2.0 <= f <= 400.0):
                continue
            adj, fpk = overshoot_of(xv, yv)
            q, qrms = rolloff_q(xv, yv, f, above=fpk)
            mtfs.setdefault(stock, dict(page=r['page'], f50=round(f, 2),
                                        cover=round(got[1], 3),
                                        adj=round(adj, 4),
                                        q=(round(q, 3) if q else None),
                                        qrms=(round(qrms, 3) if qrms else None)))
        elif r['kind'] in ('dyeabs', 'dyeden'):
            im = bitmap(doc, r['xref'])
            cal = calibrate(im, bank)
            if not cal:
                continue
            d = dye_panel(im, cal['frame'], cal, r['kind'])
            if d:
                dyes.setdefault(stock, dict(page=r['page'], **d))
        elif r['kind'] == 'kinet':
            im = bitmap(doc, r['xref'])
            cal = calibrate(im, bank)
            if cal:
                kinetics.append((stock, r['page']))
    return ladders, mtfs, kinetics, families, dyes


def compare(ladders, mtfs):
    """Sort every harvested number into FILL / CORROBORATE / DIFFER."""
    import film_profiles as FP
    by = {p.name: p for p in FP.FILM_PROFILES}
    out = []
    for stock, h in sorted(ladders.items()):
        p = by.get(stock)
        if p is None or p.is_monochrome:
            continue
        db = (p.curves.b.dmin, p.curves.g.dmin, p.curves.r.dmin)
        bk = (h['b'], h['g'], h['r'])
        dev = max(abs(a - b) for a, b in zip(db, bk))
        if max(db) - min(db) < FLAT_LADDER:
            verdict = 'FILL'
        elif dev <= AGREE_D:
            verdict = 'CORROBORATE'
        else:
            verdict = 'DIFFER'
        out.append(dict(kind='ladder', stock=stock, page=h['page'],
                        book=bk, db=db, dev=round(dev, 4), verdict=verdict))
    for stock, h in sorted(mtfs.items()):
        p = by.get(stock)
        if p is None:
            continue
        db = (p.mtf.f50_r, p.mtf.f50_g, p.mtf.f50_b)
        f = h['f50']
        if not p.is_monochrome:
            inside = min(db) * 0.85 <= f <= max(db) * 1.15
            verdict = 'BRACKET-OK' if inside else 'BRACKET-OUT'
        elif p.mtf.mtf_measured:
            verdict = ('CORROBORATE' if abs(f - db[0]) <= AGREE_F50 * db[0]
                       else 'DIFFER')
        else:
            verdict = 'FILL'
        out.append(dict(kind='mtf', stock=stock, page=h['page'],
                        book=f, q=h.get('q'), qrms=h.get('qrms'),
                        adj=h.get('adj'),
                        db=db, verdict=verdict))
    return out


# ===========================================================================
# 6. report / gate
# ===========================================================================
def main(argv):
    hard = '--assert' in argv
    src = PDF if PDF.is_file() else ALT_PDF
    if not src.is_file():
        raise SystemExit("source PDF not found: %s nor %s" % (PDF, ALT_PDF))
    doc = pymupdf.open(src)
    index = build_index(doc)
    bank = Bank(doc, index)
    problems = []
    if bank.missing:
        problems.append("template bank is missing digits %s" % bank.missing)

    kinds = {}
    for r in index:
        kinds[r['kind']] = kinds.get(r['kind'], 0) + 1
    if len(index) < 700:
        problems.append("figure index found only %d captions, expected 708" % len(index))
    if kinds.get('char', 0) < 250:
        problems.append("only %d characteristic-curve panels indexed" % kinds.get('char', 0))

    ladders, mtfs, kinetics, families, dyes = harvest(doc, index, bank)
    rows = compare(ladders, mtfs)

    if not hard:
        print("«Современные фотоматериалы и их обработка» -- %d pages, %d figures"
              % (doc.page_count, len(index)))
        print("  figure kinds: " + ", ".join("%s %d" % (k, v)
                                             for k, v in sorted(kinds.items())))
        print("  template bank: " + ", ".join("%s x%d" % (k, len(v))
                                              for k, v in sorted(bank.tpl.items())))
        print("  mapped stocks: %d book names -> %d database stocks"
              % (len(STOCK_MAP), len(set(STOCK_MAP.values()))))
        print("  ladders traced %d, f50 traced %d, kinetics panels seen %d"
              % (len(ladders), len(mtfs), len(kinetics)))
        print("  dye panels traced %d (%d neutral pairs, %d three-dye)"
              % (len(dyes),
                 sum(1 for d in dyes.values() if d['shape'] == 'neutral_pair'),
                 sum(1 for d in dyes.values() if d['shape'] == 'three_dye')))
        nfp = sum(len(f['points']) for v in families.values() for f in v)
        print("  development families %d on %d stock(s), %d points"
              % (sum(len(v) for v in families.values()), len(families), nfp))
        for st in sorted(families):
            for f in families[st]:
                print("  FAMILY       %-26s p%-4d %-9s %-12s %.0f degC"
                      % (st, f['page'], f['developer'], f['tank'] or '-',
                         f['celsius']))
                for q in f['points']:
                    print("      %5.1f min  gamma %.3f  base+fog %.3f"
                          % (q['minutes'], q['gamma'], q['fog']))
        for r in rows:
            if r['kind'] == 'ladder':
                print("  %-12s %-26s p%-4d book B%.3f G%.3f R%.3f  db B%.3f G%.3f R%.3f"
                      % (r['verdict'], r['stock'], r['page'],
                         r['book'][0], r['book'][1], r['book'][2],
                         r['db'][0], r['db'][1], r['db'][2]))
            else:
                print("  %-12s %-26s p%-4d f50 %6.1f q %-6s rms %-6s adj %-7s db %.0f/%.0f/%.0f"
                      % (r['verdict'], r['stock'], r['page'], r['book'],
                         r.get('q'), r.get('qrms'), r.get('adj'),
                         r['db'][0], r['db'][1], r['db'][2]))

    # ---- the gate ---------------------------------------------------------
    n_corr = sum(1 for r in rows if r['verdict'] == 'CORROBORATE')
    n_brk = sum(1 for r in rows if r['verdict'] == 'BRACKET-OK')

    # The adopted set, pinned by name and by value. A reader that silently
    # stops finding one of these, or starts finding a different number for it,
    # fails the build rather than quietly changing the database's provenance.
    ADOPTED = {
        ('ladder', 'KODAK_EKTAPRESS_PJ400'): (1.020, 0.803, 0.382),
        ('ladder', 'KODAK_VERICOLOR_III_160'): (0.784, 0.568, 0.209),
        # ⚠ 72.3 -> 73.09 ON 2026-09-18c, AND THE AXIS GOT BETTER RATHER THAN
        # THE NUMBER GETTING LOOSER. Queue P56 taught the digit bank six more
        # axes; on this panel that takes the abscissa from NINE readable
        # labels to ELEVEN -- the «3» and the «600» were unreadable before --
        # and an eleven-point log fit is what moved f50 by 1.1 %. The panel,
        # the trace and the ordinate are unchanged.
        ('mtf', 'KODAK_TECHNICAL_PAN'): 73.09,
        # -- reviewed and ADOPTED 2026-09-17d. This panel only calibrated once
        # queue P56 taught the digit bank the four label forms it had never
        # seen, and it broke a documented three-way Kodak conflict on this
        # stock (F-32 95.9, F-4016 66.7, F-4043 > 81) by agreeing with F-32 to
        # 2.9 % -- see the profile comment. The stock held an estimate of 72.0
        # and now holds a measurement.
        # ⚠ UNCHANGED AT 98.7 THROUGH THE 2026-09-18c BANK CHURN, AND THE
        # ROUND TRIP IS THE LESSON. Adding six dye-panel axes to BANK_SOURCES
        # gave the bank «6» exemplars in the WAVELENGTH face and it then
        # misread this panel's «600» -- in the frequency face -- as «0». A
        # zero cannot sit on a log axis, `fit_axis` dropped the label, and the
        # nine-label fit gave 96.48 against the eleven-label 98.68. Adding
        # THIS axis to the bank restored both the «600» and the «3».
        ('mtf', 'KODAK_TMAX_400'): 98.7,
    }
    # and the two numbers the same panel supplies beside f50
    ADOPTED_MTF_EXTRA = {
        # ⚠ THE OVERSHOOT PIN IS THE GATE ON `interior`'s GRID REMOVAL and it
        # earned that job on 2026-09-18c. With the line-opening left on for
        # this panel the trace climbed onto the 120 % gridline and reported
        # 0.1969; the book prints +15.1 %. The MTF path now passes grid=False
        # and the reading is 0.1564, inside this pin's own 0.01 window.
        'KODAK_TECHNICAL_PAN': dict(q=1.071, adj=0.1506),
        'KODAK_TMAX_400': dict(q=2.163, adj=0.1929),
    }
    got = {}
    for r in rows:
        got[(r['kind'], r['stock'])] = r['book']
    for key, want in ADOPTED.items():
        if key not in got:
            problems.append("the adopted %s for %s is no longer produced" % key)
            continue
        have = got[key]
        if key[0] == 'ladder':
            if max(abs(a - b) for a, b in zip(have, want)) > 0.010:
                problems.append("%s ladder moved: %s against the adopted %s"
                                % (key[1], tuple(round(x, 3) for x in have), want))
        elif abs(have - want) > 0.5:
            problems.append("%s f50 moved: %.2f against the adopted %.2f"
                            % (key[1], have, want))
    # ⚠⚠ REVIEWED, CORROBORATING, AND DELIBERATELY NOT ADOPTED. `compare()`
    # calls an MTF row a FILL whenever the stored f50 is not flagged
    # `mtf_measured`, which is a proxy for "the database has only an estimate
    # here" -- and on ONE stock that proxy is wrong. KODAK T-MAX P3200's 84.3
    # is F-4001 (2019)'s own drawing; `mtf_measured` is False on it for a
    # different reason entirely (its q beats the Gaussian by only 1.3x, under
    # the threshold for switching the carrier -- see verify.py G-MTFBW3).
    # The book's p347 panel traces 85.81, which CORROBORATES Kodak to 1.8 %
    # and does not outrank it. Listed here so the gate keeps watching the
    # number without the harvest overwriting a manufacturer's figure with a
    # reference book's -- which it briefly did on 2026-09-18c.
    REVIEWED_NOT_ADOPTED = {
        ('mtf', 'KODAK_TMAX_P3200'): (85.81, 84.3),
    }
    for key, (book, db) in REVIEWED_NOT_ADOPTED.items():
        row = next((r for r in rows
                    if (r['kind'], r['stock']) == key), None)
        if row is None:
            problems.append("the reviewed-not-adopted %s for %s is no longer "
                            "produced" % key)
        elif abs(row['book'] - book) > 0.5:
            problems.append("%s moved: %.2f against the reviewed %.2f"
                            % (key[1], row['book'], book))
        elif abs(row['book'] - db) / db > 0.05:
            problems.append("%s no longer corroborates the stored %.1f "
                            "(book %.2f)" % (key[1], db, row['book']))
    for r in rows:
        if (r['verdict'] == 'FILL' and (r['kind'], r['stock']) not in ADOPTED
                and (r['kind'], r['stock']) not in REVIEWED_NOT_ADOPTED):
            problems.append("a NEW fill appeared that nothing has reviewed: "
                            "%s %s" % (r['kind'], r['stock']))
    for stock, want in ADOPTED_MTF_EXTRA.items():
        row = next((r for r in rows if r['kind'] == 'mtf' and r['stock'] == stock), None)
        if row is None:
            problems.append("%s no longer yields an MTF panel" % stock)
            continue
        if row.get('q') is None or abs(row['q'] - want['q']) > 0.05:
            problems.append("%s rolloff moved: %s against the adopted %.3f"
                            % (stock, row.get('q'), want['q']))
        if row.get('adj') is None or abs(row['adj'] - want['adj']) > 0.01:
            problems.append("%s adjacency overshoot moved: %s against the "
                            "adopted %.4f" % (stock, row.get('adj'), want['adj']))
    # The adopted development family, pinned the same way the fills are.
    ADOPTED_FAMILIES = {
        ('KODAK_TMAX_P3200', 'T-MAX'):
            [(8.0, 0.729), (9.0, 0.837), (11.0, 1.021), (13.0, 1.085)],
        ('KODAK_TMAX_P3200', 'T-MAX RS'):
            [(9.0, 0.549), (10.0, 0.604), (12.0, 0.771), (14.0, 0.886),
             (17.0, 1.038)],
    }
    seen_fam = {}
    for st, fs in families.items():
        for f in fs:
            seen_fam[(st, f['developer'])] = [(q['minutes'], q['gamma'])
                                              for q in f['points']]
    for key, want in ADOPTED_FAMILIES.items():
        have = seen_fam.get(key)
        if have is None:
            problems.append("the adopted development family %s / %s is no "
                            "longer produced" % key)
            continue
        if len(have) != len(want):
            problems.append("%s / %s now has %d points, adopted %d"
                            % (key[0], key[1], len(have), len(want)))
            continue
        for (t0, g0), (t1, g1) in zip(have, want):
            if abs(t0 - t1) > 0.01 or abs(g0 - g1) > 0.02:
                problems.append("%s / %s moved: %.1f min gamma %.3f against "
                                "the adopted %.1f / %.3f"
                                % (key[0], key[1], t0, g0, t1, g1))
    for key in seen_fam:
        if key not in ADOPTED_FAMILIES:
            problems.append("a NEW development family appeared that nothing "
                            "has reviewed: %s / %s" % key)
    # ⚠ THE DYE RECORDS ARE PINNED BY THE ONE PROPERTY A MIS-TRACE BREAKS.
    # D-nom includes the base, so it must lie above D-min at every wavelength;
    # and on a MASKED NEGATIVE the D-min trace must FALL from blue to red,
    # because that is what an orange mask is. Both are checked against the
    # database's own stored copy, so a reader that starts tracing something
    # else fails here rather than in a render.
    import film_profiles as _fp
    _by = {q.name: q for q in _fp.FILM_PROFILES}
    # ⚠ EKTACHROME_160T IS TRACED AND DELIBERATELY NOT ADOPTED. Its panel
    # calibrates and its two curves separate cleanly, but it is a REVERSAL
    # stock: it has no orange mask, so the falling-mask test that validates
    # every other pair has nothing to say about it, and the D-min trace came
    # out RISING 0.075 -> 0.773 toward the red. That may be the measurement --
    # a reversal's minimum density is residual dye, not a mask, and need not
    # fall -- or it may be the two curves assigned the wrong way round, and
    # nothing available here decides which. A record that cannot be checked is
    # not stored.
    ADOPTED_DYES = ['KONICA_CENTURIA_SUPER_400', 'KODAK_EKTAPRESS_PJ400',
                    'KODAK_VERICOLOR_III_160',
                    'KODAK_T400CN', 'KODAK_BW400CN']
    for st in ADOPTED_DYES:
        got = dyes.get(st)
        if got is None or got['shape'] != 'neutral_pair':
            problems.append("the adopted dye record for %s is no longer "
                            "produced" % st)
            continue
        stored = _by[st].dye_density
        if len(stored.d_dmin) != len(got['d_dmin']):
            problems.append("%s dye grid changed length" % st)
            continue
        worst = max(abs(a - b) for a, b in zip(stored.d_dmin, got['d_dmin']))
        worst = max(worst, max(abs(a - b) for a, b in
                               zip(stored.d_neutral, got['d_neutral'])))
        if worst > 0.01:
            problems.append("%s dye trace moved by %.3f D against the stored "
                            "copy" % (st, worst))
        if any(b < a - 0.05 for a, b in zip(got['d_dmin'], got['d_neutral'])):
            problems.append("%s D-nom falls below D-min" % st)
        p_ = _by[st]
        if (not p_.is_monochrome and not p_.is_reversal
                and got['d_dmin'][0] <= got['d_dmin'][-1]):
            problems.append("%s is a masked negative and its D-min trace does "
                            "not fall from blue to red" % st)
    if n_corr < 6:
        problems.append("only %d corroborations; this reader's own check is that "
                        "its traces reproduce manufacturer data already in the "
                        "database" % n_corr)
    if n_brk < 6:
        problems.append("only %d colour f50 brackets held" % n_brk)
    for r in rows:
        if r['kind'] == 'ladder' and r['verdict'] == 'FILL':
            b, g, rr = r['book']
            if not (b > g > rr):
                problems.append("%s adopted ladder is not ordered" % r['stock'])
    n_fill = len(ADOPTED)

    # ⚠ THE CAPTION-WRAP FIX IS PINNED BY COUNT, because a silent regression in
    # it costs no trace and raises no error -- it just makes a whole figure kind
    # anonymous again, and `harvest` then skips every panel without a word
    # (queue P53, 2026-09-17e). Before the fix five of the 115 dye-absorption
    # captions carried a film name; after it, 114 do, and 29 of those resolve to
    # a database stock. The one that does not is «Рис. 4.122 ... фотобумаги»,
    # a PAPER panel whose caption names no product.
    _dye = [r for r in index if r['kind'] == 'dyeabs']
    _named = [r for r in _dye if r['film']]
    _mapped = [r for r in _named if STOCK_MAP.get(r['film'])]
    if len(_dye) != 115 or len(_named) != 114 or len(_mapped) != 29:
        problems.append("dye-absorption index moved: %d panels, %d named, "
                        "%d mapped; expected 115 / 114 / 29"
                        % (len(_dye), len(_named), len(_mapped)))
    # ⚠ AND THE LADDER COUNT IS PINNED FOR THE SAME REASON, on the other side of
    # queue P55: bridging the dashes before dropping the blobs took the colour
    # panels that yield a three-record ladder from 2 to 18. A change that loses
    # them again fails here rather than quietly shrinking the harvest.
    if len(ladders) != 18:
        problems.append("%d ladders traced, expected 18" % len(ladders))

    if problems:
        for p in problems:
            print("[FAIL] sovremennye_2004.py -- " + p)
        return 1
    print("[OK] sovremennye_2004.py -- %d figures indexed, %d corroborations, "
          "%d brackets, %d fills, %d development families (%d points), "
          "%d dye records, %d kinetics panels read and deliberately not stored"
          % (len(index), n_corr, n_brk, n_fill, len(seen_fam),
             sum(len(v) for v in seen_fam.values()), len(dyes), len(kinetics)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
