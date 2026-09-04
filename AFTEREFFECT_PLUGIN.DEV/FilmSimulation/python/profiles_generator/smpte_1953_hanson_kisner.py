#!/usr/bin/env python3
"""Hanson & Kisner 1953: the Eastman Color system, and what its figures can and
cannot be read for.

WHAT THIS SOURCE IS
-------------------
**W. T. Hanson, Jr. and W. I. Kisner, "Improved Color Films for Color
Motion-Picture Production", Journal of the SMPTE vol. 61 no. 6, December 1953,
pp. 667-701** -- `RETRO/sim_smpte-motion-imaging-journal_1953-12_61_6.pdf`,
an Internet Archive microfilm scan, article at PDF pages 3-42.

It is the primary technical paper for **EASTMAN COLOR NEGATIVE FILM TYPE 5248**,
which this database carries as `EASTMANCOLOR_5248_1953`, and it is cited as
such by Flueckiger's timeline. It also carries the first published description
of **EASTMAN COLOR PRINT FILM 5382**, **EASTMAN PANCHROMATIC SEPARATION FILM
5216** and **EASTMAN COLOR INTERNEGATIVE FILM 5245**.

⚠ IT WAS ACQUIRED ON A RECOMMENDATION THAT WAS WRONG ABOUT ITS CONTENTS, AND
THAT IS RECORDED HERE RATHER THAN QUIETLY DROPPED. It was fetched to fill
5248's empty `spectral` and `dye_density` fields. **It contains neither.** There
is no spectral sensitivity panel anywhere in the paper, and the two spectral
figures it does print are the DENSITOMETER FILTERS, not the film's dyes. The
lesson is the obvious one: a citation that a stock's primary paper exists is not
evidence about which measurements are in it.

WHAT THE FOUR D-log E FIGURES CAN AND CANNOT GIVE
--------------------------------------------------
Figs 4, 7, 8 and 9 are D-log E curve sets for the four films, each with its
sensitometric conditions printed in full -- and **every one of them has an
UNLABELLED EXPOSURE AXIS**. The abscissa carries the words "Log E" and nothing
else: no numbers, no decade marks. Verified at 400 dpi on all four.

⚠ **SO NO GAMMA, NO SPEED AND NO LATITUDE CAN BE READ FROM THIS PAPER, AND NONE
IS.** Gamma is a density per decade of exposure; with the density axis
calibrated and the exposure axis not, the quantity is simply not in the figure.
A curve fitted to these plots would carry an invented horizontal scale, and its
gamma would be that invention rather than a measurement. `refuse_gamma()` exists
so that the refusal is executable rather than a comment.

⚠ WHAT THE DENSITY AXIS ALONE WOULD STILL SUPPORT -- per-layer base density, the
shoulder densities, and the vertical separation between the three curves, all of
which are calibrated -- IS **NOT** EXTRACTED HERE. It is a real remainder rather
than an oversight: those numbers are worth having only if the paper's films are
ever profiled, and none of the four is in the database. `FIG_CONDITIONS` records
what such a reading would mean when somebody does it.

⚠ AND FIG. 4 IS NOT A MEASUREMENT AT ALL. The body text introducing it reads
"An idealized set of curves is shown in Fig. 4." Figs 7, 8 and 9 carry
measurement conditions with no such qualifier. That distinction is preserved
per figure in `FIG_KIND`; it is exactly the sort of thing that vanishes when a
figure is traced and stored as data.

WHAT IS TRACED: THE PRINTING-DENSITY FILTER SET
------------------------------------------------
**Fig. 3**, captioned "Spectral density curves for filters designed to read
integral densities which approximate effective printing densities of Eastman
Color Negative Film and Eastman Color Internegative Film to Eastman Color Print
Film." Optical density 0-3.0 against wavelength 400-700 mmu, both axes
calibrated, three curves.

⚠ THIS IS THE ERA'S `M_reader`, WHICH IS A REAL GAP. The database defines
`M_reader` for exactly one stock, `KODAK_2383_RELEASE`, while 164 of 165 render
through `SCAN_DI`. Fig. 3 is the 1953 answer to the same question -- what
spectral windows the density of a colour negative was actually read through,
when that density was defined as "effective printing density to the print
stock". It is a filter set, not a scanner, and it is not interchangeable with
one; what it gives is the era's own reading basis, measured.

**Fig. 2** is the same kind of plot for "an arbitrary set of filters for
measuring red, green and blue densities of color films" -- the general-purpose
comparison the paper contrasts Fig. 3 against. Traced too, so the difference
between an arbitrary tricolor set and a printing-density set is on file as
numbers.

⚠ EACH CURVE IS CLIPPED BY ITS OWN PASSBAND AND THAT IS NOT A TRACING FAULT.
These are filter DENSITIES, so a passband is a MINIMUM and the flanks rise off
the top of a 0-3.0 frame. Each curve therefore exists in the figure only between
the two wavelengths where it crosses D 3.0, and outside that band the filter is
merely "denser than 3.0" -- which the figure states and does not quantify. The
band edges are reported as such.

WHAT ELSE IS PRINTED, AND TRANSCRIBED
--------------------------------------
`TABLE_I` -- the camera filter required for 5248 under each light source.
`TABLE_III` -- the twelve processing steps and their times.
Both are printed text, transcribed rather than traced.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dashtrace as dt  # noqa: E402

SHEET = os.path.join("RETRO",
                     "sim_smpte-motion-imaging-journal_1953-12_61_6.pdf")

SOURCE = ("W. T. Hanson, Jr. and W. I. Kisner, «Improved Color Films for Color "
          "Motion-Picture Production», Journal of the SMPTE 61(6), December "
          "1953, pp. 667-701 -- PDF/PROFILES/RETRO/"
          "sim_smpte-motion-imaging-journal_1953-12_61_6.pdf, Internet Archive "
          "microfilm scan")

DPI = 400

#: The two filter panels: PDF page and the plot frame in pixels of that render,
#: as (x0, x1, y0, y1). Frames located by projection and pinned here.
FILTER_PANELS = {
    "fig3_printing_density": dict(
        page=15, frame=(384.0, 1101.0, 416.0, 1474.0),
        caption=("Spectral density curves for filters designed to read "
                 "integral densities which approximate effective printing "
                 "densities of Eastman Color Negative Film and Eastman Color "
                 "Internegative Film to Eastman Color Print Film")),
    "fig2_arbitrary": dict(
        page=14, frame=None,
        caption=("Spectral density curves for an arbitrary set of filters for "
                 "measuring red, green and blue densities of color films")),
}

#: Both panels are drawn 400-700 mmu against optical density 0-3.0.
LAM_LO, LAM_HI = 400.0, 700.0
D_LO, D_HI = 0.0, 3.0

#: ⚠ 195, NOT THE 150 THE OTHER READERS USE. This is a 1953 microfilm scan and
#: its line art is grey rather than black; at 150 a fifth of the columns inside
#: the frame came back with no ink at all. Raising it to 195 leaves the genuine
#: gaps -- the wavelengths where a filter curve has left the top of the frame --
#: and removes the spurious ones.
INK_THR = 195 / 255.0

#: A component shorter than this is a printed label, not a curve. The three
#: in-frame text blobs on Fig. 3 are 15-17 px tall; the curve fragments are
#: 126-550. An order of magnitude, unlike the three-pixel margin the 1937 Agfa
#: panel forced.
CURVE_MIN_H = 40

#: What a re-run must reproduce for Fig. 3: (min density, wavelength of min).
#: ⚠ GREEN'S PIN WAS TAKEN FROM CONTAMINATED DATA THE FIRST TIME AND THE
#: LABEL FILTER CAUGHT IT. An exploratory probe at 543 nm returned two runs,
#: 1.323 and 1.239, and 1.239 was written down. It is not the curve: it is the
#: printed words "Green combination", which sit inside the frame right under
#: the passband minimum at D 1.13-1.26. With components shorter than
#: CURVE_MIN_H removed the green minimum reads 1.320, consistent with the 1.30
#: the plot shows by eye and with the other two bands' agreement to 0.002.
EXPECTED_FIG3 = {
    "blue": (0.848, 437.0),
    "green": (1.320, 542.0),
    "red": (1.250, 645.0),
}
FIG3_D_TOL = 0.03
FIG3_NM_TOL = 8.0

#: ⚠ WHICH FIGURES ARE MEASUREMENTS AND WHICH IS NOT, in the paper's own words.
FIG_KIND = {
    4: ("Eastman Color Negative Film, Type 5248", "IDEALIZED",
        "the body text introducing it reads 'An idealized set of curves is "
        "shown in Fig. 4'"),
    7: ("Eastman Color Print Film, Type 5382", "measured",
        "conditions printed, no qualifier"),
    8: ("Eastman Panchromatic Separation Film, Type 5216", "measured",
        "conditions printed, no qualifier"),
    9: ("Eastman Color Internegative Film, Type 5245", "measured",
        "conditions printed, no qualifier"),
}

#: The sensitometric conditions each D-log E figure prints, verbatim in
#: substance. Recorded because they are the part of those figures that IS
#: usable: they say what any future reading of the curves would mean.
FIG_CONDITIONS = {
    4: ("intensity-scale sensitometer 1/50 s; tungsten 3150 K; effective "
        "integral printing density to Eastman Color Print Film, read with the "
        "filters of Fig. 3; Eastman Electronic Color Densitometer Type 31A"),
    7: ("intensity-scale sensitometer 1/100 s; tungsten 3000 K; separate "
        "exposures through Kodak Wratten (1) No. 29, (2) No. 16 + No. 61, "
        "(3) No. 2B + No. 49; densities are (1) red of the cyan scale, "
        "(2) green of the magenta scale, (3) blue of the yellow scale, all "
        "through the filters of Fig. 2"),
    8: ("intensity-scale sensitometer 1/25 s; tungsten 3000 K plus Kodak "
        "Wratten (1) No. 70 + No. 96 (D = 0.40), (2) No. 16 + No. 61 + No. 96 "
        "(D = 0.10), (3) No. 47B + No. 2B; processed in Kodak Test Developer "
        "SD-28; DIFFUSE density"),
    9: ("intensity-scale sensitometer 1/25 s; tungsten 3000 K with "
        "superimposed exposures through Kodak Wratten (a) No. 29 + No. 96 "
        "(D = 0.60), (b) No. 16 + No. 61, (c) No. 34 + No. 38A + No. 96 "
        "(D = 0.20); effective integral printing density to Eastman Color "
        "Print Film"),
}

#: Table I, page 673: the camera filter 5248 needs under each light source.
TABLE_I = (
    ("Tungsten lamps, 3200 K", "None"),
    ("'CP' lamps, approx. 3350 K", "Straw-coloured gelatin filter such as "
                                   "Brigham Y-1"),
    ("Daylight (sunlight plus some skylight)", "Kodak Wratten No. 85"),
    ("M-R Type 170, 150-amp high-intensity arc", "Kodak Wratten No. 85"),
    ("M-R Type 40, 40-amp Duarc", "None"),
)

#: Table III, page 675: processing steps for 5248 and their times.
TABLE_III = (
    ("Prebath", "10 sec"), ("Spray rinse", "10-20 sec"),
    ("Color developer", "12 min"), ("Spray rinse", "10-20 sec"),
    ("First fixing bath", "4 min"), ("Wash", "4 min"),
    ("Bleach", "8 min"), ("Wash", "8 min"),
    ("Fix", "4 min"), ("Wash", "8 min"),
    ("Wetting agent", "5-10 sec"), ("Dry", "15-20 min"),
)


def page_gray(root=".", page=15, dpi=DPI):
    import pymupdf
    path = os.path.join(root, "PDF", "PROFILES", SHEET)
    if not os.path.isfile(path):
        return None
    doc = pymupdf.open(path)
    pm = doc[page - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width)
    doc.close()
    return a.astype(np.float64) / 255.0


def find_frame(gray, search):
    """The plot's frame rectangle, by projection. search = (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = search
    ink = gray < INK_THR
    w = ink[y0:y1, x0:x1]
    rows, cols = w.sum(axis=1), w.sum(axis=0)
    ys = [i + y0 for i, r in enumerate(rows) if r > 0.55 * (x1 - x0)]
    xs = [i + x0 for i, c in enumerate(cols) if c > 0.55 * (y1 - y0)]
    if len(ys) < 2 or len(xs) < 2:
        return None

    def edges(v):
        grp, cur = [], [v[0]]
        for q in v[1:]:
            if q - cur[-1] <= 3:
                cur.append(q)
            else:
                grp.append(float(np.mean(cur)))
                cur = [q]
        grp.append(float(np.mean(cur)))
        return grp
    ge, gx = edges(ys), edges(xs)
    return (gx[0], gx[-1], ge[0], ge[-1])


def trace_filter_panel(gray, frame):
    """Lower envelope of each filter curve: {band: [(nm, D), ...]}.

    ⚠ THE LOWER ENVELOPE IS THE CURVE HERE, and that is a property of the
    figure rather than a shortcut. A filter's spectral density is single-valued
    in wavelength, so one column holds one curve point; where a flank is near
    vertical the column holds a long run instead, and the lowest pixel of that
    run is the function's value at that column to within a pixel. Taking the
    bottom of the ink therefore reads the curve exactly where it is flat and
    conservatively where it is steep, which is the right way round -- the flat
    part is the passband minimum this figure exists to show.
    """
    fx0, fx1, fy0, fy1 = frame
    ink = gray < INK_THR
    m = np.zeros_like(ink)
    m[int(fy0) + 4:int(fy1) - 3, int(fx0) + 4:int(fx1) - 3] = \
        ink[int(fy0) + 4:int(fy1) - 3, int(fx0) + 4:int(fx1) - 3]
    lab, info = dt._components(m)
    keep = [n for n, (_w, h, _c) in info.items() if h >= CURVE_MIN_H]
    m &= np.isin(lab, keep)

    def nm(x):
        return LAM_LO + (x - fx0) * (LAM_HI - LAM_LO) / (fx1 - fx0)

    def dens(y):
        return (fy1 - y) * (D_HI - D_LO) / (fy1 - fy0)

    pts = []
    for x in range(int(fx0) + 5, int(fx1) - 4):
        col = np.flatnonzero(m[:, x])
        if col.size:
            pts.append((nm(x), dens(col.max())))
    # split into passbands wherever the wavelength gap exceeds one grid step
    bands, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[0] - cur[-1][0] > 4.0:
            bands.append(cur)
            cur = [p]
        else:
            cur.append(p)
    bands.append(cur)
    bands = [b for b in bands if len(b) > 12]
    names = ("blue", "green", "red")
    bands.sort(key=lambda b: min(p[0] for p in b))
    return {names[i]: b for i, b in enumerate(bands[:3])}


def refuse_gamma(_fig):
    """Always raises. The exposure axis of every D-log E figure here is bare.

    ⚠ EXECUTABLE, NOT ADVISORY. The temptation with four labelled curve sets
    from a manufacturer's own staff is to fit them anyway and note the caveat
    in a comment; comments do not survive a copy-paste into a profile. This
    raises instead.
    """
    raise ValueError(
        "Hanson & Kisner 1953 Figs 4/7/8/9 print 'Log E' with no numbers on "
        "the abscissa. Density is calibrated; exposure is not. Gamma, speed "
        "and latitude are therefore not in these figures and must not be "
        "derived from them.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ns = ap.parse_args(argv)

    print("[i] %s" % SOURCE)
    bad = 0

    print("\n  What the four D-log E figures are")
    for f in sorted(FIG_KIND):
        film, kind, why = FIG_KIND[f]
        print("      Fig %d  %-46s %-9s (%s)" % (f, film, kind, why))
    print("    [note] all four print 'Log E' with NO numbers on the abscissa; "
          "density is calibrated and exposure is not, so no gamma, speed or "
          "latitude is read from any of them")
    try:
        refuse_gamma(4)
        print("    [FAIL] refuse_gamma did not refuse")
        bad += 1
    except ValueError:
        print("    [OK  ] and the refusal is executable, not advisory")

    gray = page_gray(ns.root, page=FILTER_PANELS["fig3_printing_density"]["page"])
    if gray is None:
        print("\n  [SKIP] source not present: %s" % SHEET)
        return 1 if (bad and ns.do_assert) else 0

    frame = find_frame(gray, (300, 1250, 350, 1500))
    pin = FILTER_PANELS["fig3_printing_density"]["frame"]
    ok = frame is not None and max(abs(a - b) for a, b in zip(frame, pin)) < 2.0
    bad += (not ok)
    print("\n  [%s] Fig. 3's frame is where it was pinned: %s"
          % ("OK  " if ok else "FAIL",
             tuple(round(v, 1) for v in frame) if frame else None))

    bands = trace_filter_panel(gray, frame or pin)
    print("\n  Fig. 3 -- the printing-density filter set")
    for name in ("blue", "green", "red"):
        b = bands.get(name)
        if not b:
            print("      %-6s NOT FOUND" % name)
            bad += 1
            continue
        lo, hi = b[0][0], b[-1][0]
        mn = min(b, key=lambda p: p[1])
        print("      %-6s passband %6.1f..%-6.1f nm   minimum density %.3f at "
              "%.1f nm" % (name, lo, hi, mn[1], mn[0]))
        want = EXPECTED_FIG3[name]
        good = (abs(mn[1] - want[0]) < FIG3_D_TOL
                and abs(mn[0] - want[1]) < FIG3_NM_TOL)
        bad += (not good)
        print("        [%s] against the pinned %.3f at %.0f nm"
              % ("OK  " if good else "FAIL", want[0], want[1]))

    order = sorted(bands, key=lambda k: min(p[0] for p in bands[k]))
    ok = order == ["blue", "green", "red"]
    bad += (not ok)
    print("    [%s] the three passbands still run blue < green < red in "
          "wavelength" % ("OK  " if ok else "FAIL"))
    print("    [note] each curve is CLIPPED at D 3.0 by its own flanks -- "
          "these are filter densities, so a passband is a minimum and outside "
          "the quoted band the figure says only 'denser than 3.0'")

    print("\n  Table I -- camera filter for 5248 by light source")
    for src, filt in TABLE_I:
        print("      %-42s %s" % (src, filt))
    print("\n  Table III -- 5248 processing, %d steps, %s total wet time"
          % (len(TABLE_III), "about 49 min"))

    if ns.emit:
        print("\n  --- Fig. 3, 5 nm resample inside each passband ---")
        for name in ("blue", "green", "red"):
            b = bands.get(name)
            if not b:
                continue
            xs = [p[0] for p in b]
            ys = [p[1] for p in b]
            lo = 5 * int(np.ceil(min(xs) / 5))
            hi = 5 * int(np.floor(max(xs) / 5))
            g = list(range(lo, hi + 1, 5))
            print("  %-6s %d..%d nm = (%s)" % (
                name, lo, hi,
                ", ".join("%.3f" % float(np.interp(w, xs, ys)) for w in g)))

    print()
    if bad:
        print("  [FAIL] %d problem(s)" % bad)
        return 1 if ns.do_assert else 0
    print("  [OK  ] Hanson & Kisner 1953: the printing-density filter set "
          "traced, the four D-log E figures recorded as UNCALIBRATED in "
          "exposure and Fig. 4 as the author's own 'idealized', nothing "
          "adopted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
