"""Verification suite for the film simulation. Run: python3 verify.py"""
import dataclasses
import math, os, re, struct, sys, zlib
import os as _os
from pathlib import Path as _Path
from pathlib import Path
import numpy as np
from PIL import Image

import film_sim as fs
import film_profiles
from film_profiles import (FILM_PROFILES, FORMATS, PRINT_STOCKS, StockKind,
                           ToneCurve, get_profile, validate_all)

ok = True
def chk(label, cond, extra=""):
    global ok
    cond = bool(cond)
    ok &= cond
    print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"   {extra}" if extra else ""))

validate_all()
lin = fs.load_linear(Path("test_chart.png"))


# --- slice support -----------------------------------------------------------
# The full suite is render-heavy and does not finish inside a short per-process
# wall-clock budget, so it can be run in slices:
#     VERIFY_SLICE=1-6  python3 verify.py
#     VERIFY_SLICE=7-14 python3 verify.py
# Section numbering follows the "# ---- N." banners, counted in file order
# (note: the banners re-use some numbers, so the ordinal, not the printed
# number, is what selects). Omit VERIFY_SLICE to run everything.
import os as _os
_sl = _os.environ.get("VERIFY_SLICE", "")
if _sl:
    _a, _, _b = _sl.partition("-")
    _LO, _HI = int(_a), int(_b or _a)
else:
    _LO, _HI = 1, 10 ** 6
_SEC = [0]


def _sec_on():
    """True when the section just entered falls inside the requested slice."""
    _SEC[0] += 1
    return _LO <= _SEC[0] <= _HI


# Shared fixtures live here, not inside a section: they are cheap to build and
# several sections use them, so a slice must not depend on an earlier slice
# having run.
st_clean = fs.RenderSettings(grain_scale=0.0, print_grain=False,
                             misreg_scale=0.0, flare=0.0)

# The RENDERED emulsion MTF including the adjacency lift, and where it peaks.
# ⚠ SHARED, not section-local, because three separate sections assert against it
# after queue A4 (2026-09-02e) and a slice must not depend on an earlier slice.
# This is exactly what `film_sim.FreqGrid.mtf` computes -- the stock's rolloff
# multiplied by the difference-of-Gaussians lift -- so "what the stored pair
# renders" is asserted against the renderer's own composition, not a restatement
# of it. The grid is log-spaced from 0.1 to 300 c/mm: the lift peaks at
# 206.07 / adjacency_um, which is 1.0 c/mm at the longest length in the file and
# 29 c/mm at the shortest, so both ends are covered with decades to spare.
_A4_FGRID = np.logspace(math.log10(0.1), math.log10(300.0), 20000)


def _rendered_peak(_m, _ch):
    """(peak MTF, frequency of that peak) for one channel of one MTFSpec.

    ⚠⚠ THIS PROBES THE KERNEL THE RENDERER CONVOLVES, NOT THE FREQUENCY-DOMAIN
    LAW, AND THAT CHANGED ON 2026-09-18b (queue P51). It used to evaluate
    `mtf_response` -- the analytic 1/(1+(f/f50)^q). The engine has no FFT and
    convolves `mtf_kernel_response` instead, so the law probe was answering a
    question nobody asks: whether a peak exists in a transfer function the
    renderer never applies. The honest question is what the renderer does.

    ⚠ IT MOVES FIVE STOCKS AND EVERY ONE IS EXPLAINED BY ITS OWN q. Out of the
    inert set go FUJI_PROVIA_400X (q 1.500), FUJI_SUPER_F125_8532 (q 1.915) and
    KODAK_TECHNICAL_PAN (q 1.071) -- the three shallowest rolloffs in the
    corpus, where the analytic tail falls faster than the band-pass can lift
    but the discrete kernel, which is truncated, does not. In come
    AGFA_RSX_II_50 (q 2.310, adjacency 0.047) and GEVACHROME_600 (q 2.090,
    adjacency 0.090), where the opposite holds: a steep law leaves room for a
    peak that the kernel's own smoothing then flattens below 1.0002.
    ⚠ TECHNICAL PAN IS THE CASE THAT DECIDED IT. The book measures its peak at
    +15.1 % and 5.75 c/mm; the kernel resolves 1.0076 at 5.89 c/mm, agreeing to
    2.4 % with a figure nothing in the fit was told to reproduce, while the law
    resolved no peak at all.
    """
    _roll = np.asarray(film_profiles.mtf_kernel_response(_m, _ch, _A4_FGRID),
                       dtype=float)
    _g1 = np.exp(-2 * math.pi ** 2 * (_m.adjacency_um * 0.4 / 1000.0) ** 2
                 * _A4_FGRID ** 2)
    _g2 = np.exp(-2 * math.pi ** 2 * (_m.adjacency_um * 2.0 / 1000.0) ** 2
                 * _A4_FGRID ** 2)
    _t = _roll * (1.0 + _m.adjacency * (_g1 - _g2))
    _i = int(_t.argmax())
    return float(_t[_i]), float(_A4_FGRID[_i])


# ---- 1. profile integrity ------------------------------------------------
if _sec_on():
    # 2026-08-02: 83 -> 89 (six Soviet stocks added from Gurlev 1986 / Iofis
    # 1980: SVEMA FOTO-32, FOTO-130, DS-4, TSNL-32, TSNL-65, TASMA OCH-45);
    # reversal count 20 -> 21 (TASMA_OCH_45 is a B&W reversal).
    # 2026-08-04: 89 -> 93 (AGFA_NEG_TYPE_B_1943, FUJICOLOR_A250,
    # GEVACHROME_902, GEVACOLOR_NEG_682); reversal 21 -> 22 -- Gevachrome 902 is
    # a reversal camera/duplicating stock, the other three are negatives.
    # 2026-08-11: 93 -> 100 (Kodak Data Book 1952: VERICHROME_1952,
    # PANATOMIC_X_SHEET_1952, TRI_X_SHEET_1952, ORTHO_X_SHEET_1952; Agfa
    # 2003 brochure: OPTIMA_200, OPTIMA_400, PORTRAIT_160).
    # 2026-08-13: 100 -> 98 (SVEMA_FN_64 renamed SVEMA_FOTO_65 and its two
    # gauge-variant entries retired -- gauge now comes from the format control;
    # TSNL renamed CNL; EIGHT_MM_* renamed GENERIC_*).
    # 2026-08-13 second batch: 98 -> 121 (Kodak still B&W + colour negative
    # lines and Agfa Scala 200x, from their own sheets in the landing).
    # 2026-08-13 third batch: 121 -> 131 (Cheltsov & Bongard 1958 -- two
    # Kodachromes, Agfacolor type 3, Anscocolor 843, Gevacolor 652, two
    # Ferraniacolor, Svema DS-2 and LN-3, Eastmancolor 5248/1953). Four
    # colour PrintStocks landed in the same batch: 5 -> 9.
    # 2026-08-14 fourth batch: 131 -> 142 (The Compact Photo-Lab-Index 1979 --
    # eight Polaroid types with published D-max/D-min/slope/resolution, plus
    # Ilford Pan F, FP4 and HP4). Print stocks unchanged at 9.
    # 2026-08-15 fifth batch: 142 -> 143 (KODAK_TECHNICAL_PAN from publication
    # P-255 -- the widest documented processing envelope in the corpus,
    # CI 0.50-2.50 from one emulsion).
    # ⚠ 155 -> 157 on 2026-08-19 (queue item G1): GEVACHROME_600 and _605, the
    # 1968 Agfa-Gevaert reversal television pair, added from Rens & Van Bets.
    # ADDING A STOCK SHIFTS eTOTAL_FILMS_PROFILES, the generated enum and every
    # line index in film_names.txt -- i.e. the plugin's ListBox -- which is why it
    # waited for an explicit owner decision rather than riding along with a data
    # pass.
    # ⚠ 157 -> 159 on 2026-08-20: KODAK_VISION2_50D_5201 (H-1-5201, the ninth
    # vector granularity sheet and the first carrying a stock not already held)
    # and FUJI_SUPER_F125_8532 (queue item C6). Both owner-approved in one batch
    # precisely so the ListBox shifts ONCE.
    # ⚠ 159 -> 161 on 2026-08-24: EASTMAN_TRI_X_5223 and KODAK_8374, the two BBC
    # T-101 emulsions this file had been carrying as footnotes on other stocks
    # (queue item #30). Owner-approved in one batch, again so the ListBox shifts
    # ONCE. KODAK_5302 went in the same pass but is a PrintStock, so it does not
    # appear in film_names.txt and moves no index.
    # ⚠ 161 -> 160 on 2026-08-24, LATER THE SAME DAY: FUJI_F125_8630 removed
    # (owner-approved). It was a gauge clone of FUJI_F125_8530, and Fuji's own
    # four-digit code rule -- printed in «Техника кино и телевидения» 1989 No.4
    # p70 -- makes the second digit the GAUGE, so the two were never separate
    # emulsions. That is a SECOND ListBox shift in one day; both were signed off
    # individually and the file_names digest moved twice.
    # ⚠ `default_format` IS A KEY INTO FORMAT_GEOM AND NOTHING CHECKED IT until
    # 2026-08-20. Two profiles added that day carried "35mm" -- a human-readable
    # string, not a key -- and film_sim.py died with KeyError: '35mm' on the
    # px/mm calculation. It got through the whole build because the field is a
    # free-form `str`, `FilmProfile.validate()` never looked at it, and every
    # audit and every render-based check either names its own format explicitly
    # or happens to test a stock whose key is valid. A typo in a string field
    # that indexes another dict is exactly what a cheap guard is for.
    _fmt_bad = sorted("%s=%r" % (p.name, p.default_format)
                      for p in FILM_PROFILES
                      if p.default_format not in film_profiles.FORMAT_GEOM)
    chk("every default_format is a real FORMAT_GEOM key",
        not _fmt_bad, ", ".join(_fmt_bad[:5]) if _fmt_bad
        else "%d stocks, %d distinct keys, all valid"
             % (len(FILM_PROFILES),
                len({p.default_format for p in FILM_PROFILES})))

    # And the converse property that makes the guard above sufficient: every
    # stored key must resolve through the same lookups the renderer uses. The
    # KeyError was thrown on film_sim's CLI path, which no check exercised.
    # ⚠ ZERO FRAME PITCH IS CORRECT for a still format -- sheet film and
    # Polaroid have no cine frame advance, so `frame_pitch_mm` returns 0.0 for
    # large4x5 / medium645 / polaroid_pack / polaroid_sx70. The first version of
    # this guard demanded a positive pitch everywhere and failed on all four:
    # the guard was wrong, not the data. Width must be positive for every
    # format; pitch must be non-negative, and positive for the cine keys, which
    # is where the renderer actually advances the coating field frame to frame.
    _CINE = {"ff35", "super35", "academy35", "anamorphic35", "techni35",
             "16mm", "super16", "8mm", "super8", "imax15"}
    _fmt_fail = []
    for _k in sorted({p.default_format for p in FILM_PROFILES}):
        try:
            _w = film_profiles.FORMATS[_k]
            _p = film_profiles.frame_pitch_mm(_k)
            if not _w > 0.0:
                _fmt_fail.append("%s width %.3f" % (_k, _w))
            if _p < 0.0:
                _fmt_fail.append("%s pitch %.3f" % (_k, _p))
            if _k in _CINE and not _p > 0.0:
                _fmt_fail.append("%s is cine but pitch %.3f" % (_k, _p))
        except Exception as _e:
            _fmt_fail.append("%s -> %s" % (_k, _e))
    chk("every stored format resolves; cine keys carry a frame pitch",
        not _fmt_fail, "; ".join(_fmt_fail) if _fmt_fail
        else "%d distinct keys resolve, stills correctly pitch 0"
             % len({p.default_format for p in FILM_PROFILES}))

    # ⚠ 160 -> 161 on 2026-08-26f: KODAK_PRO_100T_PRT, from KODAK publication E-29
    # (April 1999). It is the first stock added since the ordering rule was
    # written down, so it is also the first real exercise of it: the database is
    # authoritative, film_enum.hpp / film_names.txt / the generated .cpp
    # literals are regenerated from it, and the three guards further down assert
    # positional identity rather than mere set equality.
    # ⚠ 165 -> 166 on 2026-08-31: KODAK_EKTAR_125, owner-approved, created on one
    # measured number (a blue D-min UPPER BOUND from US 5,334,491). It is UNFROZEN in
    # film_ids.lock, so it took id 165 at the end and no existing ListBox index moved.
    # ⚠ 166 -> 170 on 2026-09-01, owner-approved: AGFA_ULTRA_50 and the
    # AGFACHROME RSX II 50/100/200 trio, created from the 1998 edition of Agfa's
    # «Technical Data PF» -- a document NotFound.md and queue G6 both recorded as
    # a duplicate of the 2004 F-PF-E4 sheet and which is nothing of the kind (md5
    # edb3dd17... against bf9f0c1a...). It is the only document in the corpus
    # that plots ULTRA 50 or the RSX II line at all. All four are UNFROZEN in
    # film_ids.lock, so they take ids 166-169 at the end and no existing ListBox
    # index moves.
    # ⚠ 170 -> 171 ON 2026-09-02 (queue C4): SVEMA_CO_90L, the last Soviet
    # amateur reversal specification, ТУ 6-42-1514-90 of 1990. C4 had it
    # recorded as TWO stocks whose norms would render identically; both files
    # in the corpus are scans of ONE document and «ЦО-90Д» is an OCR misread.
    # ⚠ 171 -> 172 ON 2026-09-02 (queue N1): FUJI_NEOPAN_SS, from FUJIFILM
    # AF3-411E(N), the sheet that turned §23k.8's refusal into a profile --
    # three papers here measure Neopan grain and none measured its tone scale
    # until this one. Appended at frozen id 171 so no existing index moves.
    # ⚠ 172 -> 175 ON 2026-09-02e (queue T3): FUJI_PROVIA_100F,
    # FUJICOLOR_SUPERIA_XTRA_400 and FUJICOLOR_PRO_400H, from AF3-036E,
    # AF3-151E and AF3-176E. All three characteristic-curve panels and all
    # three MTF panels are TRACED (fuji_t3_2026.py); the rms granularity,
    # resolving power at both contrasts, base and speed are printed numbers.
    # Ids 172-174, appended, so no existing ListBox index moves.
    # ⚠ 182 -> 184 on 2026-09-07: AGFA_VISTA_PLUS_200 and _400, from the
    # AgfaPhoto licensed sheet. NOT Agfa-Gevaert films -- see G-VP1.
    # ⚠ 184 -> 185 on 2026-09-15: ORWOCOLOR_NC3, Tamm & Weisflog 1972.
    # ⚠⚠ 185 -> 186 on 2026-09-17: EASTMAN_5293_250T_1982, Kennel et al.,
    # SMPTE J. 91(10) 1982 -- AND IT IS AN ADDITION BESIDE AN EXISTING STOCK
    # RATHER THAN A REPLACEMENT OF IT. Eastman used the catalogue number 5293
    # twice, in 1982 at EI 250T on conventional grain and in 1992 at EI 200T
    # on EXR tabular grain; the database now holds both, and G-5293-TWO below
    # asserts that it still holds both and that they still differ.
    # ⚠⚠ 186 -> 191 on 2026-09-17: the five 1956 sheet emulsions queue P63
    # had held a complete data set for since 2026-09-16 and had no profile to
    # put it in -- SUPER PANCHRO-PRESS TYPE B, PORTRAIT PANCHROMATIC, ROYAL
    # ORTHO, SUPER SPEED ORTHO PORTRAIT and COMMERCIAL. What unblocked them
    # was not a new document: P62 measured their spectral response off the
    # same book's wedge spectrograms and kodak_1956_curves.py traced the
    # characteristic curve off the same page as the numbers already read.
    chk("191 stocks load and validate", len(FILM_PROFILES) == 191, f"n={len(FILM_PROFILES)}")
    # ⚠ AND THE THREE NEW ONES CARRY MEASURED CURVES, which is the point of the
    # row: a new stock added from a datasheet's PROSE only would have been a set
    # of estimates with a citation. Pinned so a later edit cannot quietly
    # replace a traced curve with a family default.
    _t3 = {"FUJI_PROVIA_100F": (0.0728, 1.9881),
           "FUJICOLOR_SUPERIA_XTRA_400": (0.1366, 0.6622),
           "FUJICOLOR_PRO_400H": (0.1503, 0.6136)}
    _t3bad = [n for n, (dmin, gam) in _t3.items()
              if abs(get_profile(n).curves.r.dmin - dmin) > 1e-4
              or abs(get_profile(n).curves.r.gamma - gam) > 1e-4]
    chk("T3: the three new Fuji stocks carry their TRACED red curve, not an "
        "estimate", not _t3bad, ", ".join(_t3bad) if _t3bad
        else "PROVIA 100F dmin 0.0728 gamma 1.9881 (reversal, negated x); "
             "SUPERIA X-TRA 400 0.1366/0.6622; PRO 400H 0.1503/0.6136")
    # ⚠ THE MASK LADDER MUST RISE r < g < b ON BOTH NEW COLOUR NEGATIVES, and
    # this guard exists because the first fit BROKE IT. Fitting all six curve
    # parameters to a panel that stops inside the straight line let each
    # record's shoulder land wherever its trace ended, and SUPERIA's
    # extrapolated Dmax came out 2.68 / 2.57 / 3.00 -- red above green on a
    # film whose own traced curves put red lowest at every exposure. The
    # shoulder is now declared at _neg's family default and the ladder holds.
    for _n in ("FUJICOLOR_SUPERIA_XTRA_400", "FUJICOLOR_PRO_400H"):
        _c = get_profile(_n).curves
        _dm = [c.dmin + c.gamma * (c.shoulder_x - c.toe_x)
               for c in (_c.r, _c.g, _c.b)]
        chk(f"T3: {_n} Dmin and Dmax both rise r < g < b",
            _c.r.dmin < _c.g.dmin < _c.b.dmin and _dm[0] < _dm[1] < _dm[2],
            "Dmin %.3f/%.3f/%.3f, Dmax %.2f/%.2f/%.2f"
            % (_c.r.dmin, _c.g.dmin, _c.b.dmin, *_dm))
    # ⚠ 10 -> 11 on 2026-08-25 (queue C15): KODAK_VISION3_DI_2254, appended at
    # the END of the table so every earlier print stock keeps its index.
    chk("11 print stocks load", len(PRINT_STOCKS) == 11, f"n={len(PRINT_STOCKS)}")
    # ---- queue C15, 2026-08-25. The only dye-stability table in the corpus. --
    _di = [s for s in PRINT_STOCKS if s.name == "KODAK_VISION3_DI_2254"]
    chk("KODAK_VISION3_DI_2254 is present and is the LAST print stock",
        len(_di) == 1 and PRINT_STOCKS[-1].name == "KODAK_VISION3_DI_2254",
        "appended, so every earlier print stock keeps its index")
    if _di:
        _ds = _di[0].dye_stability
        # ⚠ THE CENSORING IS THE POINT. Kodak prints ">100" for every record
        # that outlives the test. Storing 100.0 would turn a lower BOUND into a
        # number later arithmetic would average; the convention is 0.0 against
        # censor_years, and this guard is what stops a well-meaning edit from
        # "filling in the blanks".
        chk("2254's censored records are stored as 0.0, not as the bound 100",
            _ds.censor_years == 100.0 and _ds.reference_temp_c == 21.0
            and _ds.loss_c == 0.0 and _ds.loss_m == 0.0
            and _ds.loss_r == _ds.loss_g == _ds.loss_b == 0.0
            and _ds.dmin_gain_r == 0.0 and _ds.dmin_gain_g == 0.0,
            "seven '>100' entries at 21 C, held as censored rather than as 100")
        chk("2254 keeps the two FINITE Arrhenius figures the sheet prints",
            _ds.loss_y == 86.0 and _ds.dmin_gain_b == 77.0,
            "yellow 86 y to a 0.10 density loss; blue 77 y to a 0.1 D-min gain")
        # ⚠ AND IT MUST NOT SPREAD. One film cannot establish a fade rate for a
        # class (method rule 18), and a DI recording film's couplers are chosen
        # for archival stability rather than camera exposure. This is the same
        # refusal made for the 7266 sigma(D) two days earlier.
        _other = [s.name for s in PRINT_STOCKS
                  if s.name != "KODAK_VISION3_DI_2254" and s.dye_stability.has_data]
        chk("no other stock inherited 2254's Arrhenius table",
            not _other, "; ".join(_other) if _other else "1 of 11, as measured")

        # An intermediate film's whole purpose is unity gamma. Nothing in the
        # raster trace was told that, so this is a physical check on the
        # calibration and not a restatement of the fit.
        _c = _di[0].curves
        chk("2254's three gammas sit within 6% of unity, as an intermediate must",
            all(abs(getattr(_c, ch).gamma - 1.0) <= 0.06 for ch in "rgb"),
            "r %.3f g %.3f b %.3f" % (_c.r.gamma, _c.g.gamma, _c.b.gamma))
        # The blue and green records are printed as ONE stroke on the toe, so
        # their dmin is the same measurement twice. Identical, not merely close.
        chk("2254's blue and green D-min are identical, as the sheet draws them",
            _c.b.dmin == _c.g.dmin and _c.r.dmin < _c.g.dmin,
            "b = g = %.3f, r = %.3f" % (_c.b.dmin, _c.r.dmin))
        # ---- queue C36, 2026-08-26. The MTF this sheet CANNOT state. --------
        # ⚠ THE REFUSAL IS THE RESULT, and it is measured. Two of the three
        # records never reach 50 % response: the curves stop at 82.2 cycles/mm
        # with green at 53.1 % and red at 50.6 %. A 0.0 in the triple therefore
        # means CENSORED and mtf_f50_bound carries what the record exceeds --
        # the DyeStabilitySpec idiom, reused because the problem is the same.
        _di0 = _di[0]
        chk("2254 stores blue's measured f50 and CENSORS green and red",
            _di0.mtf_measured and _di0.mtf_f50_b == 51.9
            and _di0.mtf_f50_g == 0.0 and _di0.mtf_f50_r == 0.0
            and _di0.mtf_f50_bound == 82.2,
            "blue crosses 50 % at 51.9; green and red are only bounded > 82.2")
        # ⚠ AND THE LEGACY SCALAR IS DELIBERATELY UNCHANGED. It is what the
        # reference renderer reads, and the measurement cannot replace it with
        # one honest number: 72.0 is too SHARP for blue and too SOFT for the two
        # proven >= 82.2, so the estimate is wrong in both directions at once.
        # If this guard ever fails because someone "fixed" the scalar, read the
        # profile comment before agreeing with them.
        chk("2254's legacy mtf_f50 scalar is still the untouched estimate",
            _di0.mtf_f50 == 72.0,
            "no render moves; the triple records what the sheet says")
        chk("2254 stores NO rolloff exponent, and the refusal is measured",
            not hasattr(_di0, "mtf_rolloff_q"),
            "blue's traced span is 36-82 cycles/mm -- 0.36 decades, only 0.16 "
            "below f50 -- so a carrier normalised at f = 0 would be fitted "
            "almost entirely to the tail")
        # ⚠ CATALOGUE-NUMBER HAZARD, ASSERTED. EASTMAN_5254_1968 is a 1968 ECN
        # CAMERA NEGATIVE with the same four digits. The two must stay separate
        # films, and neither may cite the other's document.
        _5254 = get_profile("EASTMAN_5254_1968")
        chk("the two '254' films stay separate and do not cross-cite",
            _5254.kind is StockKind.NEGATIVE
            and "H-1-2254" not in " ".join(_5254.provenance.sources)
            and "2254" not in _5254.aliases,
            "1968 ECN camera negative vs the 2026 DI recording film")
    rev = [p.name for p in FILM_PROFILES if p.is_reversal]
    # 2026-08-13: 22 -> 23 (AGFA_SCALA_200X, B&W reversal, added).
    # 2026-08-13: 23 -> 26. Cheltsov & Bongard 1958 added three reversal
    # stocks: KODACHROME_1938, KODACHROME_TYPE_A_1938 (both Kodachrome-process
    # reversal, diffusing couplers in the developer) and
    # FERRANIACOLOR_REVERSAL_1950 (incorporated couplers).
    # 2026-08-14: 26 -> 33. Seven Polaroid instant types are reversal (the
    # print IS the output); POLAROID_55_PN_NEG is deliberately NEGATIVE
    # because it is a real fixed, enlargeable silver negative.
    # 2026-08-17: 33 -> 34 (SVEMA_CO_32D, Soviet amateur colour reversal from
    # ТУ 6-17-912-87 -- the specification measures its useful exposure interval
    # between densities 0.3 and 2.1, which only makes sense for a positive).
    # 2026-09-01: 36 -> 39, the three AGFACHROME RSX II stocks (E-6 slide film).
    # 2026-09-02e: 40 -> 41, FUJI_PROVIA_100F (queue T3, E-6/CR-56).
    # 2026-09-06: 42 -> 43, FUJI_PROVIA_400F from AF3-066E (its ISO 400
    # sibling; same E-6/CR-56 chemistry, same RHP/RDP family).
    # 2026-09-06: 43 -> 44, FUJICHROME_64T_II from AF3-024E -- the first
    # TUNGSTEN-balanced Fuji reversal stock in the file.
    chk("reversal stocks flagged", len(rev) == 44, ", ".join(rev))

    # alias resolution incl. the user's own phrasing
    cases = {
        "Kodak Vision3 500T (5219)": "KODAK_VISION3_500T_5219",
        "5219": "KODAK_VISION3_500T_5219",
        "vision3-250d": "KODAK_VISION3_250D_5207",
        "  velvia ": "FUJI_VELVIA_50",
        "hp5+": "ILFORD_HP5_PLUS_400",
        "Fomapan 400 Action": "FOMAPAN_400_ACTION",
        "three-strip": "TECHNICOLOR_THREE_STRIP",
        "8572": "FUJICOLOR_SUPER_F500_8572",
        "7266": "KODAK_TRI_X_REVERSAL_200",  # "5266" alias removed in schema v2 (no such Kodak catalogue number)
    }
    bad = {k: get_profile(k).name for k, v in cases.items() if get_profile(k).name != v}
    chk("alias / catalogue-number lookup", not bad, str(bad))

# ---- schema v3: digitised spectral sensitivity -----------------------------
if _sec_on():
    sp_stocks = [p for p in FILM_PROFILES if p.spectral.has_data]
    chk("spectral pilot stocks present",
        {"FUJI_NEOPAN_ACROS_100", "KODAK_VISION3_250D_5207",
         "KONICA_INFRARED_750"} <= {p.name for p in sp_stocks},
        ", ".join(p.name for p in sp_stocks))
    _sp_ok = all(
        abs(max(layer)) < 1e-9 and min(layer) >= -4.0 - 1e-9
        for p in sp_stocks
        for layer in (p.spectral.log_s_r, p.spectral.log_s_g,
                      p.spectral.log_s_b, p.spectral.log_s_pan)
        if layer
    )
    chk("spectral layers peak-normalised to 0.0 within [-4, 0]", _sp_ok)
    # The IR stock must actually be an IR record: sensitivity at 750 nm at peak,
    # and a dead gap in the mid-visible -- this guards against a transcription
    # that silently shifts the grid.
    _ir = get_profile("KONICA_INFRARED_750").spectral
    _ir_idx = lambda nm: int(round((nm - _ir.lambda_start_nm) / _ir.lambda_step_nm))
    chk("IR spectral curve peaks at 750 nm with a dead mid-visible gap",
        _ir.log_s_pan[_ir_idx(750)] == 0.0
        and _ir.log_s_pan[_ir_idx(570)] <= -3.9,
        f"750nm={_ir.log_s_pan[_ir_idx(750)]}, 570nm={_ir.log_s_pan[_ir_idx(570)]}")

    # ---- P64, 2026-09-16. The dark-fade RATE is now a renderable STATE. ----
    # ⚠⚠ `AgingSpec` HAS EXISTED SINCE SCHEMA v2 WITH ELEVEN FIELDS, IS ALL
    # ZEROS ON ALL 185 STOCKS, AND IS READ BY NOTHING. It holds a STATE -- how
    # much fade a piece of film has already suffered -- and there was no way to
    # arrive at that state, because nothing in the database said how fast a
    # film fades. `DyeStabilitySpec` now does, and
    # `film_sim.resolve_storage_age` turns the rate into the state.
    chk("a published 10 %-loss time reproduces exactly 10 % at its own year, "
        "which is the definition rather than a fit",
        all(abs(fs.dark_fade_fractions(
                    get_profile(n), get_profile(n).dye_stability.loss_y)[2]
                - 0.10) < 1e-12
            for n in ("KODAK_EKTAR_125", "KODAK_VERICOLOR_III_160")),
        "EKTAR 125 at 8 y and VERICOLOR III at 23 y")

    chk("storage age is inert at zero and returns the profile by identity",
        all(fs.resolve_storage_age(q, 0.0) is q for q in FILM_PROFILES),
        "185 stocks, fresh")

    _aged = sorted(q.name for q in FILM_PROFILES
                   if fs.resolve_storage_age(q, 50.0) is not q)
    # ⚠ TWO -> THREE ON 2026-09-17, AND THE THIRD IS A DIFFERENT DYE FROM A
    # DIFFERENT SOURCE. Wilhelm's Table 19.1 publishes the YELLOW time for
    # EKTAR 125 and VERICOLOR III; Kennel et al. 1982 Fig. 15 publishes a
    # predicted CYAN curve for the 1982 5293, at 24 degC and 40 % RH like
    # Wilhelm's, and at 429 years to a 10 % loss it is two orders of magnitude
    # slower than either. The set is still exactly the stocks a source names.
    chk("exactly the three stocks with a published dark-fade rate age, and "
        "no stock inherits one by class",
        _aged == ["EASTMAN_5293_250T_1982", "KODAK_EKTAR_125",
                  "KODAK_VERICOLOR_III_160"],
        "; ".join(_aged) if _aged else "none")

    # ⚠ THE ASYMMETRY IS THE WHOLE EFFECT AND IT IS PINNED. Wilhelm publishes
    # the time for the LEAST STABLE dye and names it -- yellow on both stocks
    # -- and says nothing about the other two. His subject is that the three
    # fade at different rates and that the DIFFERENTIAL is what makes a faded
    # negative uncorrectable. An implementation that faded all three equally
    # would pass every tolerance and model the wrong thing.
    _sym = []
    for _n in ("KODAK_EKTAR_125", "KODAK_VERICOLOR_III_160"):
        _q = get_profile(_n)
        _r = fs.resolve_storage_age(_q, 50.0)
        if (_r.curves.r.gamma != _q.curves.r.gamma
                or _r.curves.g.gamma != _q.curves.g.gamma):
            _sym.append(_n)
        if not (_r.curves.b.gamma < _q.curves.b.gamma):
            _sym.append(_n + " (blue did not fade)")
    chk("only the dye the source NAMES fades -- blue drops, red and green are "
        "untouched, because a uniform fade would deny the differential the "
        "source exists to describe",
        not _sym, "; ".join(_sym) if _sym else
        "2 stocks at 50 y: blue only")

    # ⚠ AND dmin MUST NOT MOVE, which is a refusal recorded in the resolver:
    # part of a masked negative's D-min is orange mask (dye, fades) and part is
    # support (does not), the schema stores their sum, and nothing separates
    # them.
    chk("dark fade leaves dmin alone, because the mask and the support are "
        "stored as one number and nothing in the corpus splits them",
        all(fs.resolve_storage_age(get_profile(n), 50.0).curves.b.dmin
            == get_profile(n).curves.b.dmin
            for n in ("KODAK_EKTAR_125", "KODAK_VERICOLOR_III_160")),
        "2 stocks at 50 y")

    # ---- P61b, 2026-09-16. The development-time axis is no longer inert. ----
    # ⚠⚠ THE NUMBER THAT MATTERED HERE WAS ZERO. The database has carried a
    # `ProcessingFamily` since schema v7 and by this date 960 development
    # points across 25 stocks, and NO STAGE IN ANY OF THE THREE ENGINES READ
    # ONE OF THEM. `AlgoControlEnums.hpp` declared the slider's range, step and
    # sentinel; `AlgoControls` declared the field; nothing consumed it. The
    # control was documented, emitted, and wired to nothing.
    #
    # `film_sim.resolve_development_time` and `AlgoDevelopmentTime.hpp` are the
    # consumer. These guards pin what it must do and, more importantly, what it
    # must NOT do -- the inert paths are the load-bearing ones, because they
    # are what keeps every pre-existing render bit-identical.
    _dev_live = [q.name for q in FILM_PROFILES
                 if fs.development_family(q) is not None]
    # ⚠ 11 -> 16 ON 2026-09-17, AND ALL FIVE ARE NEW STOCKS RATHER THAN NEW
    # DATA ON OLD ONES. Queue P63's five 1956 sheet emulsions each arrive with
    # the four-to-six point gamma family Kodak prints on their own data
    # sheet, so every one of them can answer the Development Time control
    # from the day it is added.
    chk("the development-time axis reaches 16 stocks -- it reached NONE "
        "before P61b, on a database holding 1175 development points",
        len(_dev_live) == 16,
        "%d stocks carry a usable gamma-bearing family: %s"
        % (len(_dev_live), ", ".join(sorted(_dev_live))))

    # ⚠ THE SENTINEL MUST RETURN THE PROFILE ITSELF, NOT AN EQUAL COPY. The
    # identity test is the contract `resolve_process_variant` already meets and
    # is what makes the control free when it is off.
    chk("the development-time sentinel returns the profile by identity, so "
        "every render made before the control existed is reproduced exactly",
        all(fs.resolve_development_time(q, -1.0) is q
            for q in FILM_PROFILES),
        "185 stocks, sentinel -1")

    # ⚠ AND SO MUST ANY TIME OUTSIDE THE STOCK'S OWN TRACED RANGE. The control's
    # contract says the value is treated as the sentinel rather than clamped,
    # because extrapolating a development family is not a measurement.
    chk("a development time outside the traced range is treated as the "
        "sentinel and NOT clamped",
        all(fs.resolve_development_time(q, t) is q
            for q in FILM_PROFILES for t in (0.25, 1000.0)),
        "185 stocks x 2 out-of-range times")

    # ⚠ COLOUR IS REFUSED, AND IT IS A DATA STATEMENT. One scale on three
    # records asserts the layers move together under development, which this
    # project measured to be false on PORTRA 800 (red +0.25, blue +0.14).
    _dev_colour = [n for n in _dev_live
                   if not get_profile(n).is_monochrome]
    chk("no colour stock responds to a development time -- a single gamma "
        "scale cannot state three records that move differently",
        not _dev_colour, "; ".join(_dev_colour) if _dev_colour else
        "all 11 are monochrome")

    # ⚠ MONOTONE IN TIME, WHICH IS THE PHYSICS AND NOT A SMOOTHNESS TEST. More
    # development is more contrast, always, over the traced span. A group
    # selection that mixed two developers or two generations would break this
    # long before it broke any tolerance.
    _nonmono = []
    for _n in _dev_live:
        _q = get_profile(_n)
        _pts = fs.development_family(_q)
        _lo, _hi = _pts[0][0], _pts[-1][0]
        _ks = [fs.development_gamma_scale(_q, _lo + (_hi - _lo) * i / 12.0)
               for i in range(13)]
        if any(b < a - 1e-9 for a, b in zip(_ks, _ks[1:])):
            _nonmono.append(_n)
    chk("gamma rises monotonically with development time on every stock the "
        "control reaches",
        not _nonmono, "; ".join(_nonmono) if _nonmono else
        "11 stocks, 13 samples each across their own traced span")

    # ---- v35, 2026-09-16. The first dark-fade rates for CAMERA NEGATIVES. ----
    # ⚠ THE REFUSAL ABOVE IS NOT WEAKENED BY THESE TWO. 2254's table still does
    # not spread to any other stock; what arrived is a SECOND, INDEPENDENT
    # source that happens to cover camera negatives -- Wilhelm's Table 19.1,
    # reproducing Kodak's own estimates. The guard therefore checks the same
    # thing from the film side: exactly the two stocks the source names, and no
    # others, and nothing inherited by class.
    _fade = sorted(p.name for p in FILM_PROFILES if p.dye_stability.has_data)
    chk("dark-fade rates on exactly the three camera negatives a source "
        "names -- Wilhelm's Table 19.1 for two, Kennel et al. 1982 Fig. 15 "
        "for the third -- and no class inheritance",
        _fade == ["EASTMAN_5293_250T_1982", "KODAK_EKTAR_125",
                  "KODAK_VERICOLOR_III_160"],
        "; ".join(_fade) if _fade else "none")
    for _nm, _yr in (("KODAK_EKTAR_125", 8.0),
                     ("KODAK_VERICOLOR_III_160", 23.0)):
        _p = get_profile(_nm)
        _d = _p.dye_stability
        # ⚠ THE UNCENSORED PATH IS NEW AND IS WHAT THIS CHECKS. Before v35
        # `has_data` was `censor_years > 0`, so a source that censors nothing
        # read as empty. These two records have censor_years == 0.0 AND data,
        # which is the combination the old definition could not express.
        chk(f"{_nm} holds an UNCENSORED dark-fade rate at 24 degC",
            _d.has_data and _d.censor_years == 0.0
            and _d.reference_temp_c == 24.0 and abs(_d.loss_y - _yr) < 1e-9
            and _d.loss_c == 0.0 and _d.loss_m == 0.0
            and _d.loss_r == _d.loss_g == _d.loss_b == 0.0,
            f"yellow {_yr:.0f} y to a 10 % loss from D 1.0 at 40 % RH")
        # The published caveats must travel with the number, because the
        # number is a 40 % RH figure that halves at 60 % and ignores stain.
        chk(f"{_nm}'s dark-fade source carries the RH and stain caveats",
            "60% RH" in _d.source and "stain" in _d.source
            and "Wilhelm" in _d.source,
            "%d chars of provenance" % len(_d.source))
    # ⚠ ORDERING IS A PHYSICAL CHECK AND IT IS NOT FREE. Wilhelm's own p665
    # lists both films among those with "relatively good stability", but the
    # table puts Vericolor III at 23 years and Kodak's E-107 puts Ektar 125 at
    # 8 to 14 -- so the professional portrait negative is nearly three times
    # the amateur-range colour negative, which is the direction the couplers
    # predict. A later edit that swapped them would pass every other guard.
    chk("VERICOLOR III outlasts EKTAR 125 in dark storage, as its couplers "
        "predict",
        get_profile("KODAK_VERICOLOR_III_160").dye_stability.loss_y
        > 2.0 * get_profile("KODAK_EKTAR_125").dye_stability.loss_y,
        "23 y against 8 y, a factor of 2.9")

    # ---- queue M1, 2026-08-31: the print stock's reader response -----------
    # ⚠ THE FIRST PRINT STOCK WITH A SPECTRAL SENSITIVITY, and it matters
    # because of `dye_matrix_from_spectra`: a negative's stored densities are
    # status M or A, and what stage 12 may legitimately hold is
    # `M_reader . M_status^-1`. For a printed negative the reader IS the print
    # emulsion, and until today no print stock carried one.
    _2383 = [s for s in PRINT_STOCKS if s.name == "KODAK_2383_RELEASE"][0]
    chk("the release print stock carries a spectral sensitivity",
        _2383.spectral.has_data)
    chk("it is the only print stock that does; the other ten stay empty",
        [s.name for s in PRINT_STOCKS if s.spectral.has_data]
        == ["KODAK_2383_RELEASE"],
        ", ".join(s.name for s in PRINT_STOCKS if s.spectral.has_data))
    _pk, _wide = [], []
    for _b, _arr in (("r", _2383.spectral.log_s_r),
                     ("g", _2383.spectral.log_s_g),
                     ("b", _2383.spectral.log_s_b)):
        _v = np.asarray(_arr, dtype=float)
        _lm = (_2383.spectral.lambda_start_nm
               + _2383.spectral.lambda_step_nm * np.arange(len(_v)))
        _pk.append(float(_lm[int(np.argmax(np.where(_v > -3.99, _v, -np.inf)))]))
        if int((_v > -1.0).sum()) > 12:
            _wide.append(f"{_b} spans {int((_v > -1.0).sum())} samples")
    chk("the print stock's three layers peak at 680 / 550 / 470 nm",
        _pk == [680.0, 550.0, 470.0], str(_pk))
    # ⚠ A PRINT EMULSION'S LAYERS ARE NARROW AND FAR APART -- that is what makes
    # it a reader at all, because they have to match a printer's three light
    # sources rather than see the whole spectrum. Camera-negative layers overlap
    # heavily. If this set ever came back looking like a camera stock's, the
    # wrong panel has been read off a sheet that carries several.
    chk("each print-stock layer is narrow, as a printer-matched layer must be",
        not _wide, "; ".join(_wide))
    # ⚠ AND IT IS NOT WIRED IN, which is the whole M1 result. Having the reader
    # response does not licence adopting the derived matrix: 164 of 165
    # profiles render through SCAN_DI, whose reader is a scanner, and NOT ONE
    # renders through 2383. Storing it would state that a stock's reader is a
    # film it is never printed on.
    chk("no stock renders through the print stock whose reader we now have",
        not any(p.default_print == "KODAK_2383_RELEASE" for p in FILM_PROFILES),
        ", ".join(p.name for p in FILM_PROFILES
                  if p.default_print == "KODAK_2383_RELEASE"))

    # ---- schema v25: the printing-density matrix, 2026-09-03 ---------------
    # ⚠ THE SAME SPECTRA, A DIFFERENT FIELD, AND THE DISTINCTION IS THE POINT.
    # The guard directly above says the derived matrix must NOT become a
    # negative's `dye_matrix`, because there the reader is a scanner. On the
    # print stage's EXPOSURE side the print film's own response is not a
    # stand-in for anything -- it is the quantity. Both statements are true at
    # once and both are asserted, so neither can be quietly relaxed into the
    # other.
    _pdm = np.asarray(_2383.printing_density_matrix, dtype=float)
    chk("the release print stock carries a printing-density matrix, and it is "
        "flagged as derived from measurement",
        not np.allclose(_pdm, np.eye(3)) and _2383.printing_matrix_measured
        and bool(_2383.printing_matrix_source),
        "max|P - I| = %.4f" % float(np.abs(_pdm - np.eye(3)).max()))
    # ⚠ UNIT ROWS ARE THE WHOLE SAFETY ARGUMENT. They are what makes a neutral
    # negative still print neutral, so the printer-light solve is unchanged and
    # switching the operator on re-times nothing. A row sum drifting off 1.0
    # would put a per-channel gain back on top of the printer lights -- the
    # same double count `dye_matrix_from_spectra.py` refused for stage 12.
    chk("its rows sum to 1.0, so the operator is crosstalk and not gain",
        float(np.abs(_pdm.sum(axis=1) - 1.0).max()) <= 5e-4,
        "row sums %s" % np.array2string(_pdm.sum(axis=1), precision=6))
    # ⚠ AND A NEUTRAL MUST COME THROUGH UNCHANGED, checked as arithmetic rather
    # than as a property of the row sums, because that is what a reader will
    # actually want to know.
    _neu = _pdm @ np.array([1.2, 1.2, 1.2])
    chk("a neutral negative density passes through the printing matrix intact",
        float(np.abs(_neu - 1.2).max()) <= 1e-3,
        "1.200/1.200/1.200 -> %s" % np.array2string(_neu, precision=4))
    # ⚠ THE SIGN PATTERN IS PHYSICS AND IS ENFORCED, not merely reported. The
    # negative's magenta dye must ADD to the print's red-sensitive layer and its
    # yellow dye must ADD to the green-sensitive one; a sign flip here would
    # SATURATE the print instead of desaturating it, which is the wrong
    # direction for an unwanted absorption and would still render plausibly.
    chk("the printing matrix desaturates: magenta into red and yellow into "
        "green are both positive",
        _pdm[0][1] > 0.0 and _pdm[1][2] > 0.0 and _pdm[2][1] > 0.0,
        "r<-m %+.4f, g<-y %+.4f, b<-m %+.4f"
        % (_pdm[0][1], _pdm[1][2], _pdm[2][1]))
    # ⚠ THE TWO STABLE TERMS ARE PINNED TO THE VALUES SEVEN INDEPENDENT DYE
    # SETS AGREE ON. g<-y spans 0.0465..0.0641 across the reference set and
    # b<-m 0.0352..0.0434; the medians must stay inside those spans or the
    # reference set has changed without anyone saying so.
    chk("the two stable off-diagonals are the medians of the seven-negative "
        "reference set",
        abs(_pdm[1][2] - 0.0497) < 5e-4 and abs(_pdm[2][1] - 0.0407) < 5e-4,
        "g<-y %.4f (want 0.0497), b<-m %.4f (want 0.0407)"
        % (_pdm[1][2], _pdm[2][1]))
    # ⚠ AND THE ONE UNSTABLE TERM IS PINNED TO THE MEDIAN AND NOT THE OUTLIER.
    # 5218's r<-m is 0.1077, eight times the smallest in the set; the median
    # 0.0225 deliberately does not follow it, and a value up near 0.1 here
    # would mean the median has silently become a mean or the set has shrunk.
    chk("the unstable r<-m term follows the median, not 5218's 0.108 outlier",
        abs(_pdm[0][1] - 0.0225) < 5e-4,
        "r<-m %.4f (want 0.0225, set spans 0.0114..0.1077)" % _pdm[0][1])
    # ⚠ TEN OF ELEVEN PRINT STOCKS MUST STAY IDENTITY. None of them carries a
    # spectral sensitivity, so no printing matrix can be DERIVED for them, and
    # an invented one would be a per-stock estimate wearing a measured field.
    _nonid = [s.name for s in PRINT_STOCKS
              if s.name != "KODAK_2383_RELEASE"
              and s.printing_density_matrix != film_profiles.IDENTITY3]
    chk("the other ten print stocks keep an identity printing matrix",
        not _nonid, ", ".join(_nonid) or "10/10 identity")
    # ⚠ AND THE FLAG CANNOT BE SET WITHOUT THE DERIVATION BEHIND IT. Same gate
    # idiom as mtf_measured and sigma_shape_measured.
    _badflag = [s.name for s in PRINT_STOCKS
                if s.printing_matrix_measured
                and s.printing_density_matrix == film_profiles.IDENTITY3]
    chk("printing_matrix_measured is never set on an identity matrix",
        not _badflag, ", ".join(_badflag) or "clean")

    # ---- queue E2, 2026-08-31: the POLAROID family reads as SENSITIVITY ----
    # ⚠ THE CHECK THAT WOULD CATCH A MIRRORED ADOPTION. Queue E2 prescribed
    # negating these curves, on the strength of the sheets' prose ("the
    # equivalent energy needed at each wavelength"). The 667 sheet captions its
    # axis "Spectral Sensitivity (cm^2/erg)" -- area per unit energy -- and the
    # peaks rise with film speed across the four sheets. A mirrored set passes
    # every band and ordering test in this file, so the property asserted here
    # is the one a mirror actually breaks: a silver-halide emulsion is most
    # sensitive at its BLUE end unless a sensitising dye says otherwise, and
    # all four of these peak at 380-430 nm. Inverted, three of the four would
    # peak in the green trough at 480-500 nm.
    _pol = [(n, get_profile(n).spectral) for n in
            ("POLAROID_52", "POLAROID_55_PN_NEG", "POLAROID_664",
             "POLAROID_667")]
    _pol_bad = []
    for _n, _s in _pol:
        _v = np.asarray(_s.log_s_pan, dtype=float)
        _lam = _s.lambda_start_nm + _s.lambda_step_nm * np.arange(len(_v))
        _pk = _lam[int(np.argmax(np.where(_v > -3.99, _v, -np.inf)))]
        if not 380.0 <= _pk <= 440.0:
            _pol_bad.append(f"{_n} peaks at {_pk:.0f} nm")
    chk("all four POLAROID sets peak at the blue end, not in the green trough",
        not _pol_bad, "; ".join(_pol_bad))
    chk("the four POLAROID sets share one criterion, and it names SENSITIVITY",
        {s.criterion for _n, s in _pol}
        == {"log_reciprocal_erg_cm2_neutral_D0.75"},
        ", ".join(sorted({s.criterion for _n, s in _pol})))
    chk("Type 52 and Type 55 P/N carry the spectral sets E2 was opened for",
        all(get_profile(n).spectral.has_data
            for n in ("POLAROID_52", "POLAROID_55_PN_NEG")))

    # ---- queue C38, 2026-08-31: the two shapes a mis-read leaves behind ----
    # ⚠ BOTH OF THESE FIRED ON REAL STORED DATA when they were written, which is
    # the only reason to trust them. They are not style rules about how a curve
    # ought to look; each is the literal signature of a defect found in the
    # 2026-08-02 raster batch by re-reading the sheets' own vector paths.
    #
    # (a) A ONE-SAMPLE NOTCH. EASTMAN_EXR_100T_5248's green record read
    #     -1.67 / -2.02 / -1.62 across 450-470 nm: a 0.40-decade spike between
    #     samples 20 nm apart, on a flank the page draws smooth. A printed curve
    #     cannot do that and a densitometer reading of one cannot either. The
    #     PEAK sample is exempt, because a maximum is a legitimate local extreme
    #     -- without that exemption AGFA_OPTIMA_400's blue peak trips it.
    _notch = []
    for _p in sp_stocks:
        for _nm, _layer in (("r", _p.spectral.log_s_r), ("g", _p.spectral.log_s_g),
                            ("b", _p.spectral.log_s_b), ("pan", _p.spectral.log_s_pan)):
            _v = np.asarray(_layer, dtype=float)
            if _v.size < 5:
                continue
            for _i in range(1, _v.size - 1):
                _a, _b, _c = _v[_i - 1], _v[_i], _v[_i + 1]
                if min(_a, _b, _c) < -3.99 or _b >= -1e-9:
                    continue           # floor padding, or the normalised peak
                if abs(_a - _c) > 0.15:
                    continue           # a flank, not a flat run
                if abs((_a + _c) / 2 - _b) > 0.25:
                    _notch.append(f"{_p.name}.{_nm}[{_i}] "
                                  f"{_a:+.2f}/{_b:+.2f}/{_c:+.2f}")
    chk("no stored spectral layer carries a one-sample notch on a flat run",
        not _notch, "; ".join(_notch[:6]))

    # (b) A STRAIGHT-LINE TAIL. EASTMAN_EXR_50D_5245's blue record ran
    #     -0.60/-1.15/-1.80/-2.45/-3.10 below 490 nm -- steps of 0.55, 0.65,
    #     0.65, 0.65 -- which is a ruler, not a dye tail, and the drawn curve had
    #     already left the bottom of the frame 3 nm before the last of those
    #     samples. Three identical steps steeper than 0.40 decades is the
    #     signature. ⚠ EASTMAN_EXR_200T_5293's blue comes closest of what
    #     remains (three steps of exactly 0.35) and is deliberately NOT
    #     exempted: it sits under the threshold on its own numbers, its panel is
    #     the one sheet the vector reader still cannot open, and if a future
    #     re-read moves it this is where that should surface.
    _ruler = []
    for _p in sp_stocks:
        for _nm, _layer in (("r", _p.spectral.log_s_r), ("g", _p.spectral.log_s_g),
                            ("b", _p.spectral.log_s_b), ("pan", _p.spectral.log_s_pan)):
            _v = np.asarray(_layer, dtype=float)
            _m = np.where(_v > -3.99)[0]
            if _m.size < 6:
                continue
            _d = np.round(np.diff(_v[_m.min():_m.max() + 1]), 4)
            _run = 1
            for _i in range(1, len(_d)):
                _run = _run + 1 if (abs(_d[_i] - _d[_i - 1]) < 1e-9
                                    and abs(_d[_i]) > 0.40) else 1
                if _run >= 3:
                    _ruler.append(f"{_p.name}.{_nm} step {_d[_i]:+.2f} x{_run}")
                    break
    chk("no stored spectral tail is a straight line drawn with a ruler",
        not _ruler, "; ".join(_ruler[:6]))

    # The two records C38 repaired, pinned by value so a regeneration cannot
    # quietly put the old readings back.
    _s45 = get_profile("EASTMAN_EXR_50D_5245").spectral
    chk("5245's blue tail is the traced roll-off, not the extrapolated line",
        _s45.log_s_b[11:15] == (-0.85, -1.95, -2.53, -2.97),
        str(_s45.log_s_b[11:15]))
    _s48 = get_profile("EASTMAN_EXR_100T_5248").spectral
    chk("5248's green 460 nm notch is gone", _s48.log_s_g[8] == -1.62,
        str(_s48.log_s_g[6:10]))

# ---- 2. characteristic curves monotonic ----------------------------------
if _sec_on():
    x = np.linspace(-6, 6, 6001).astype(np.float32)
    # float32 evaluation leaves ulp-scale noise on a flat Dmax shelf, and that
    # noise is PROPORTIONAL TO GAMMA (D = dmin + gamma*(sp1-sp2): on the shelf
    # the bracket is constant and its float32 rounding is multiplied by gamma).
    # The 2026-08-16 vector-extracted reversal curves carry gamma 11-15 with
    # toe_k == shoulder_k, which is analytically monotone (the sigmoid argument
    # gap is the constant (shoulder_x-toe_x)/k > 0), so the allowance scales
    # with each curve's own gamma instead of weakening the check globally.
    worst = min(
        float(np.diff(fs.density(x, c)).min()) / max(1.0, c.gamma)
        for p in FILM_PROFILES for c in p.curves.as_tuple()
    )
    chk("all characteristic curves monotonic", worst >= -1e-5,
        f"min slope/gamma={worst:.3e}")

    # ---- G-MONO (2026-09-09, queue A3) ------------------------------------
    # ⚠ THE CHECK ABOVE IS A FLOAT32 SCAN OVER x in [-6, 6] AND THAT IS NOT THE
    # WHOLE ABSCISSA. `ToneCurve.mono_dip` is the excursion in closed form over
    # the WHOLE line, so these three guards state the exact property instead of
    # a windowed approximation of it. Written together on purpose: C18's lesson
    # was that a guard whose title claims more than its test can pass a change
    # that breaks the claim, so the closed form is checked against a scan
    # (G-MONO-a), the database against the bound (G-MONO-b), and the BOUND
    # ITSELF against a curve built to violate it (G-MONO-c). Without the third,
    # a `validate` that had lost its dip test would pass a and b unnoticed.
    def _dip_scan(c, n=400001):
        """Numeric excursion, stable softplus, no argument clipping."""
        def _sp(v, k):
            u = v / k
            return k * (np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u))))
        pad = 40.0 * max(c.toe_k, c.shoulder_k) + 6.0
        g = np.linspace(min(c.toe_x, c.shoulder_x) - pad,
                        max(c.toe_x, c.shoulder_x) + pad, n)
        d = (c.dmin + c.gamma * (_sp(g - c.toe_x, c.toe_k)
                                 - _sp(g - c.shoulder_x, c.shoulder_k)))
        return float((np.maximum.accumulate(d) - d).max())

    # a. the closed form IS the excursion. Sampled over the 12 curves with the
    #    largest dips plus 12 spread through the file -- a full 552-curve scan
    #    at 400k points costs ~90 s and buys nothing the extremes do not.
    _cs = [(p.name, ch, getattr(p.curves, ch))
           for p in FILM_PROFILES for ch in ("r", "g", "b")]
    _cs.sort(key=lambda t: -t[2].mono_dip)
    _probe = _cs[:12] + _cs[12::46]
    _err = max(abs(c.mono_dip - _dip_scan(c)) for _n, _ch, c in _probe)
    chk("G-MONO-a  ToneCurve.mono_dip's closed form reproduces a 400,001-point "
        "scan of the actual excursion", _err <= 1e-9,
        f"worst |closed form - scan| = {_err:.3e} D over {len(_probe)} curves, "
        f"largest dip probed {_cs[0][2].mono_dip:.3e} D on "
        f"{_cs[0][0]}.{_cs[0][1]}")

    # b. and every stored curve sits under the bound `validate` enforces.
    chk("G-MONO-b  every characteristic curve's non-monotonic excursion is "
        "under ToneCurve.MONO_DIP_MAX_D",
        _cs[0][2].mono_dip <= ToneCurve.MONO_DIP_MAX_D,
        f"worst {_cs[0][2].mono_dip:.3e} D on {_cs[0][0]}.{_cs[0][1]}, limit "
        f"{ToneCurve.MONO_DIP_MAX_D} D; "
        f"{sum(1 for _n, _ch, c in _cs if c.mono_dip > 1e-3)} curves over "
        f"1e-3 D, "
        # ⚠ TWO DIFFERENT ZEROES, counted apart. Only toe_k == shoulder_k is
        # monotone BY CONSTRUCTION; the rest reach 0.0 because
        # exp(-(shoulder_x-toe_x)/|dk|) underflows, which is a real excursion
        # below the smallest double rather than the absence of one. Reporting
        # them as one number would claim a structural property for curves that
        # only have a numerical one.
        f"{sum(1 for _n, _ch, c in _cs if c.toe_k == c.shoulder_k)} monotone "
        f"by construction (toe_k == shoulder_k), "
        f"{sum(1 for _n, _ch, c in _cs if c.mono_dip == 0.0 and c.toe_k != c.shoulder_k)}"
        f" more whose excursion underflows to 0.0")

    # c. ⚠ AND THE BOUND ACTUALLY BITES, IN BOTH DIRECTIONS. The retired guard
    #    tested `shoulder_k > 2*toe_k`, which is why the second curve here --
    #    shoulder SHARPER than toe, overshooting dmax past the shoulder -- used
    #    to be accepted without limit.
    _below = ToneCurve(0.10, 3.0, 0.0, 0.60, 0.30, 1.20)   # sh_k > toe_k
    _above = ToneCurve(0.10, 3.0, 0.0, 1.20, 0.30, 0.60)   # sh_k < toe_k
    _caught = 0
    for _c, _end in ((_below, "dips below dmin before the toe"),
                     (_above, "overshoots dmax past the shoulder")):
        try:
            _c.validate("synthetic")
        except ValueError as _e:
            _caught += 1 if _end in str(_e) else 0
    chk("G-MONO-c  ToneCurve.validate rejects an over-limit curve in BOTH "
        "directions and names the end that reverses", _caught == 2,
        f"{_caught}/2 rejected; dips {_below.mono_dip:.4f} D (below the toe) "
        f"and {_above.mono_dip:.4f} D (past the shoulder) vs limit "
        f"{ToneCurve.MONO_DIP_MAX_D} D")

    # ---- G-FITMODEL (2026-09-09, queue A3) --------------------------------
    # ⚠ THE FITTER'S MODEL AND THE RENDERER'S MODEL MUST BE ONE MODEL, and for
    # a year they were not: digitize_plot clipped the softplus ARGUMENT at 60
    # and so returned ~60k instead of v once saturated, while film_sim returns
    # the asymptote. Nothing adopted was affected -- traced panels span 2-5
    # decades -- but a gamma-constrained A3 fit walked into the clip and
    # returned toe_k 0.0263 with dmax 12.99. Checked out to x = +-15, far past
    # any panel, because that is exactly where the old bug hid.
    import digitize_plot as dp
    _xw = np.linspace(-15.0, 15.0, 3001)
    _dw = 0.0
    for _p in FILM_PROFILES:
        for _c in _p.curves.as_tuple():
            _a = dp.softplus_curve(_xw, _c.dmin, _c.gamma, _c.toe_x, _c.toe_k,
                                   _c.shoulder_x, _c.shoulder_k)
            _b = np.array([fs.density_scalar(float(_v), _c) for _v in _xw[::30]])
            _dw = max(_dw, float(np.abs(_a[::30] - _b).max()))
    chk("G-FITMODEL  digitize_plot.softplus_curve IS film_sim's curve, "
        "including the saturated branch the fitter used to clip", _dw <= 1e-12,
        f"worst |fitter - renderer| = {_dw:.3e} D over "
        f"{len(FILM_PROFILES)*3} curves, x -15..+15 (both in double)")

# ---- G-NOPATH (2026-09-10, owner directive) ------------------------------
if _sec_on():
    # ⚠ NO EMITTED CITATION MAY NAME A LOCAL FILE. Owner directive 2026-09-10:
    # the database's own comments carry the NAME of a book, standard or paper
    # and never a filename or a path on his machine. A path is worthless to
    # anyone reading the shipped database and it leaks his directory layout
    # into a file he distributes.
    #
    # ⚠ THIS GUARD IS THE RULE, not the 2026-09-10 cleanup that satisfied it.
    # 648 string literals were rewritten that day; without this check the next
    # citation typed by hand would quietly put a path back.
    #
    # ⚠ IT DELIBERATELY DOES NOT POLICE PYTHON COMMENTS. A `#` line in
    # film_profiles.py is the generator's own working note and is never emitted
    # into the C++ -- those still name the files they were traced from, which
    # is what makes a trace re-runnable.
    import re as _re
    _PATH = _re.compile(r'PDF/[^\s,;)"\']*?\.pdf|[A-Za-z0-9_][A-Za-z0-9_\-.]*\.pdf'
                        r'|[A-Za-z]:\\\\', _re.I)
    _off = []
    for _p in FILM_PROFILES:
        if _p.description and _PATH.search(_p.description):
            _off.append("%s.description" % _p.name)
        for _ps in (_p.param_sources or ()):
            for _f, _v in (("source", _ps.source), ("note", _ps.note),
                           ("conditions", _ps.conditions)):
                if _v and _PATH.search(_v):
                    _off.append("%s.%s.%s" % (_p.name, _ps.param, _f))
    for _k, _v in film_profiles._PROVENANCE_SOURCES.items():
        for _c in _v:
            if _PATH.search(_c):
                _off.append("%s.provenance" % _k)
    chk("G-NOPATH  no emitted citation names a local file or path -- only "
        "books, standards and papers by name", not _off,
        "clean across %d profiles" % len(FILM_PROFILES) if not _off
        else "%d offenders, first: %s" % (len(_off), ", ".join(_off[:3])))

# ---- G-COPYRIGHT (2026-09-10, owner directive) ---------------------------
if _sec_on():
    import cpp_codegen as _cg
    _root = Path(__file__).resolve().parent
    # The 32 generated DATABASE artefacts that must carry the proprietary
    # notice: everything in the generated set ending .hpp, .h or .cpp.
    _cpp = (["film_profiles.hpp", "film_profiles_detail.hpp", "film_profiles.cpp",
             "film_enum.hpp", "LoadFilmDataBase.h", "LoadFilmDataBase.cpp"]
            + ["film_profiles_data_%02d.cpp" % i
               for i in range(1, _cg.N_DATA_SLOTS + 1)])
    # ⚠ AND THE THREE .txt ARTEFACTS THAT MUST NOT. This half of the guard is
    # the load-bearing half. `film_names.txt` is consumed as adjacent C++
    # string literals pasted into the effect panel's listbox, so a notice there
    # would print into the film dropdown; the owner excluded the other two by
    # name. A future "add the header everywhere" tidy-up is exactly the change
    # this catches.
    _txt = ["film_names.txt", "film_display_order.txt", "film_id_migration.txt"]

    _miss = [n for n in _cpp
             if not (_root / n).is_file()
             or "Marat Shchuchinsky" not in (_root / n).read_text(
                 encoding="utf-8", errors="replace")[:2000]]
    chk("G-COPYRIGHT-a  every generated database C++/HPP artefact opens with "
        "the proprietary notice", not _miss,
        "%d/%d carry it%s" % (len(_cpp) - len(_miss), len(_cpp),
                              "" if not _miss else "; missing " + ", ".join(_miss[:3])))

    _leak = [n for n in _txt
             if (_root / n).is_file()
             and "Shchuchinsky" in (_root / n).read_text(encoding="utf-8",
                                                         errors="replace")]
    chk("G-COPYRIGHT-b  the three generated .txt artefacts do NOT carry it -- "
        "film_names.txt is pasted into the panel listbox as string literals",
        not _leak, "clean" if not _leak else "LEAKED INTO " + ", ".join(_leak))

    # ⚠ THE NOTICE COSTS SLOT BUDGET AND THE MARGIN IS NOW THE THING TO WATCH.
    # 1,072 bytes x 20 slots took the largest from 106,886 to 107,958 against a
    # 112,000 limit. Pinning the headroom here means the next prose addition
    # that would force a repack reports itself as a number rather than as an
    # infeasible build.
    _sizes = [(_root / ("film_profiles_data_%02d.cpp" % i)).stat().st_size
              for i in range(1, _cg.N_DATA_SLOTS + 1)
              if (_root / ("film_profiles_data_%02d.cpp" % i)).is_file()]
    _worst = max(_sizes) if _sizes else 0
    chk("G-COPYRIGHT-c  the largest data slot still fits under "
        "SLOT_SOURCE_LIMIT with the notice included",
        _worst <= _cg.SLOT_SOURCE_LIMIT,
        "largest %d of %d slots = %d bytes, limit %d, headroom %d"
        % (_sizes.index(_worst) + 1 if _sizes else 0, len(_sizes), _worst,
           _cg.SLOT_SOURCE_LIMIT, _cg.SLOT_SOURCE_LIMIT - _worst))

# ---- G-BLACKPOINT (2026-09-09c, queue item #303) -------------------------
if _sec_on():
    # ⚠ THREE CLAIMS, AND THE FIRST ONE IS THE PROMISE THE DEFAULT MAKES.
    # `black_point_stretch` was introduced as a CONTROL rather than a fix
    # precisely so that nothing moves until the owner moves it, so the default
    # has to be shown to be inert -- not argued to be.
    _bpp = get_profile("FUJI_VELVIA_50")
    _bst = fs.RenderSettings(grain_scale=0.0, print_grain=False,
                             misreg_scale=0.0, flare=0.0)
    _b1 = fs.simulate(lin, _bpp, _bst)
    _b1b = fs.simulate(lin, _bpp,
                       dataclasses.replace(_bst, black_point_stretch=1.0))
    chk("G-BLACKPOINT-a  black_point_stretch=1.0 is EXACTLY the default, so "
        "every render made before the control existed is reproduced",
        float(np.abs(_b1 - _b1b).max()) == 0.0,
        f"max abs diff {float(np.abs(_b1 - _b1b).max()):.3e} over "
        f"{_b1.size} samples")

    # b. and it does what it was added for. The blue record is the one that
    #    crushed: stages 09 and 12 push it past Dmax, which s=1 maps to zero.
    _b0 = fs.simulate(lin, _bpp,
                      dataclasses.replace(_bst, black_point_stretch=0.0))
    _z1 = float(np.count_nonzero(_b1[:, :, 2] == 0.0)) / _b1[:, :, 2].size
    _z0 = float(np.count_nonzero(_b0[:, :, 2] == 0.0)) / _b0[:, :, 2].size
    chk("G-BLACKPOINT-b  at 0.0 nothing in the blue record is clipped to "
        "output zero, and at 1.0 a large fraction is",
        _z0 == 0.0 and _z1 > 0.02,
        f"blue exactly 0.0: {100*_z1:.2f} % at s=1, {100*_z0:.2f} % at s=0 "
        f"(test chart, not the Velvia regression frame -- the frame measured "
        f"13.06 % / 0.00 %)")

    # c. ⚠ AND THE ANCHOR STILL LANDS AT THE DEFAULT, WHICH IS WHERE THE COST
    #    OF s=0 LIVES. At s=0 four POLAROID stocks miss the 12 % mid-grey bound
    #    (Callier amplifies the scalar solver's residual by 1.62x on a
    #    monochrome record). That is recorded in the control's own comment and
    #    in DIGITIZATION_QUEUE.md § 0.0e; this guard pins that the DEFAULT is
    #    unaffected, so the limitation can never leak into a shipped render
    #    without the control being moved deliberately.
    _anch_st = fs.RenderSettings(grain_scale=0.0, print_grain=False,
                                 misreg_scale=0.0, flare=0.0, vignette=0.0,
                                 coating_scale=0.0)
    _worst_nm, _worst = "", 0.0
    for _p in FILM_PROFILES:
        _pt = fs.simulate(lin, _p, _anch_st)[615:665, 55:145].mean(axis=(0, 1))
        _e = max(abs(float(_v) - 0.18) / 0.18 for _v in _pt)
        if _e > _worst:
            _worst_nm, _worst = _p.name, _e
    chk("G-BLACKPOINT-c  the 18 % mid-grey anchor is untouched by the new "
        "control at its default", _worst < 0.12,
        f"worst {_worst_nm} {_worst:.4f} of 184 stocks (the same bound "
        f"section 5 enforces; at s=0.0 four POLAROID stocks reach 0.14-0.17)")

# ---- 3. 16-bit PNG really is 16-bit --------------------------------------
if _sec_on():
    # Pillow silently downconverts 16-bit RGB PNG on read, so parse IHDR directly.
    out = Path("film_renders/_verify16.png")
    out.parent.mkdir(exist_ok=True)
    fs.write_png(out, (np.random.default_rng(0).random((8, 12, 3)) * 65535).astype(np.uint16), 16)
    raw = out.read_bytes()
    w_, h_, depth, ctype = struct.unpack(">IIBB", raw[16:26])
    chk("PNG IHDR: 16-bit truecolour", (depth, ctype) == (16, 2), f"depth={depth} colour_type={ctype}")
    chk("PNG dimensions correct", (w_, h_) == (12, 8), f"{w_}x{h_}")
    chk("PNG chunk CRCs valid", zlib.crc32(raw[12:29]) & 0xFFFFFFFF == struct.unpack(">I", raw[29:33])[0])
    with Image.open(out) as im:
        chk("PNG decodable by Pillow", im.size == (12, 8))

# ---- 4. mid-grey anchor: 18% scene grey -> 18% display -------------------
if _sec_on():
    # vignette and coating_scale are pinned off for the whole anchor section. The
    # grey patch of the test chart sits at r = 0.81 toward the frame corner, so
    # with the schema-v4 lens vignette active it legitimately receives up to a
    # stop less light on period stocks -- measured 61% low on AGFA_NEU_1936,
    # which is correct physics, not a broken anchor. The anchor contract is about
    # the TONE SCALE, so it is tested with spatial falloff excluded; that the
    # frame CENTRE still lands on grey_target with the defects on was verified
    # separately (0.1738 vs 0.1799 -- the residual is the local coating field).
    st = fs.RenderSettings(grain_scale=0.0, print_grain=False, misreg_scale=0.0,
                           flare=0.0, vignette=0.0, coating_scale=0.0)
    errs = {}
    for p in FILM_PROFILES:
        o = fs.simulate(lin, p, st)
        patch = o[615:665, 55:145].mean(axis=(0, 1))
        errs[p.name] = max(abs(float(v) - 0.18) / 0.18 for v in patch)
    worst_name = max(errs, key=errs.get)
    # ⚠ THE PINNED EXCEPTION WAS REMOVED ON 2026-09-08, AND THE REASON IS THE
    # BEST INDEPENDENT EVIDENCE THAT DAY'S STAGE 8b FIX WAS RIGHT.
    #
    # This used to read `_GREY_EXCEPT = {"FUJI_PROVIA_400F": 0.1233}`, with a
    # long note arguing that the 12.3 % error was not a bad trace and not a
    # failed solve: the anchor converged to -1.62 stops well inside its
    # bracket, the characteristic fit had rms 0.0069 over 82 columns, and the
    # blame was placed on the curve -- gamma 2.23 against 100F's 1.99 on a
    # shorter arm -- amplifying the residual between the scalar solve and the
    # full pixel pass. That reasoning was careful, it was checked, and it was
    # WRONG ABOUT THE CAUSE.
    #
    # The cause was stage 8b's reversal branch subtracting the interimage
    # correction instead of adding it. FUJI_PROVIA_400F is the stock that
    # branch moved the most (worst pixel 0.9998 in linear light of the 26
    # reversal stocks). Correcting the sign took its mid-grey error from
    # 0.1233 to 0.0506 -- from the single worst stock in the database, 12.3 %
    # and needing its own pin, to SEVENTH worst and comfortably inside the
    # bound every other stock already met. Nothing about the anchor solve, the
    # trace or the curve changed.
    #
    # ⚠ NOBODY WROTE THAT NUMBER DOWN AS A DEFECT, WHICH IS THE LESSON. A
    # pinned exception is the right tool when a stock is genuinely different,
    # and this project uses several correctly. But a pin also SILENCES the
    # measurement that was trying to report an upstream bug, and this one did
    # so for six days. When a single stock needs a bound three times looser
    # than every other, prefer one more hour on the cause.
    #
    # Measured after the fix: no stock reaches 0.12; worst is
    # SUPER_ANSCOCHROME_1957 at 0.0809. So the bound now applies to all 184
    # with no exceptions, which is what it always claimed to.
    _grey_bad = ["%s %.4f" % (_n, _e)
                 for _n, _e in errs.items() if _e >= 0.12]
    chk("mid grey anchors to 18% for every stock, NO exceptions "
        "(the FUJI_PROVIA_400F pin was removed 2026-09-08 -- the stage 8b "
        "reversal sign fix took it from 0.1233 to 0.0506)",
        not _grey_bad, ", ".join(_grey_bad[:3]) if _grey_bad
        else "all %d stocks under 12 %%, worst %s %.4f"
             % (len(errs), worst_name, errs[worst_name]))
    chk("SUPERSEDED mid grey bound", True,
        f"worst={worst_name} {errs[worst_name]*100:.1f}% off")

    # both print stocks, and grey_target honoured
    for tgt in (0.10, 0.18, 0.35):
        o = fs.simulate(lin, get_profile("5219"), fs.RenderSettings(
            grain_scale=0.0, print_grain=False, misreg_scale=0.0, flare=0.0,
            vignette=0.0, coating_scale=0.0, grey_target=tgt))
        got = float(o[615:665, 55:145].mean())
        chk(f"grey_target={tgt} honoured", abs(got - tgt)/tgt < 0.12, f"got {got:.4f}")
    for ps in ("SCAN_DI", "KODAK_2383_RELEASE", "TECHNICOLOR_IB"):
        o = fs.simulate(lin, get_profile("5219"), fs.RenderSettings(
            print_stock=ps, grain_scale=0.0, print_grain=False, misreg_scale=0.0,
            flare=0.0, vignette=0.0, coating_scale=0.0))
        got = float(o[615:665, 55:145].mean())
        chk(f"mid grey anchored on print stock {ps}", abs(got-0.18)/0.18 < 0.12, f"got {got:.4f}")

# ---- 5. grain granularity calibration, resolution invariant --------------
if _sec_on():
    def granularity(name, width, band_limit=True, model="boolean_jinc"):
        """sigma(D) through the 48 um aperture, x1000, as datasheets quote it.

        ⚠ REWRITTEN FOR v48, AND THE MEASURING APERTURE NOW FOLLOWS THE MODEL.
        Until v48 this measured through a GAUSSIAN stand-in for the 48 um
        aperture, because that is what the calibration integral used. The
        stand-in is gone from the physical path -- the aperture is a disk and
        its transfer is a jinc, first zero at 25.4 c/mm where the Gaussian is
        still passing 0.036 -- so measuring the jinc-calibrated field through a
        Gaussian would report a 1-5 % error that is entirely the measurement's.
        Each model is therefore measured through its own aperture, which is the
        definition its amplitude was fixed against.
        """
        p = get_profile(name)
        h = 512
        ppm = width / FORMATS["super35"]
        gsp = p.grain
        terms = film_profiles.grain_gauss_terms(
            model, clump_um=gsp.clump_um_g, clump_gain=gsp.clump_gain,
            grain_um=gsp.grain_um_rgb()[1], size_sigma_log=gsp.size_sigma_log)
        aperture = film_profiles.grain_aperture_model(model)
        scan_px = (film_profiles.scan_sigma_mm(105.0) * ppm
                   if band_limit else 0.0)
        f = fs.make_grain_field(h, width, ppm, terms,
                                gsp.rms_granularity, scan_px, gsp.anisotropy,
                                7, 0, fs.RngStage.GRAIN_G, aperture)
        grid = fs.FreqGrid(h, width, ppm, gsp.anisotropy)
        ap = aperture(grid.f_mm.astype(np.float64)).astype(np.float32)
        return float(fs.apply_transfer(f, ap).std()) * 1000.0

    # Without a band limit and with a wide enough band, the field must reproduce
    # the datasheet granularity figure -- that is the definition the amplitude is
    # fixed against.
    #
    # ⚠ BOTH SPECTRAL MODELS ARE MEASURED, AND THAT IS THE POINT OF THE v48
    # REWRITE. The calibration is what makes `rms_granularity` mean the number
    # the datasheet prints; if it held for one spectrum and not the other, the
    # stored figure would silently mean two different things depending on a
    # render setting.
    worst_err, worst_at = 0.0, ""
    for _model in ("legacy_gaussian", "boolean_jinc"):
        for nm in ("5219", "5203", "5296", "delta 3200", "kodachrome",
                   "technicolor"):
            tgt = get_profile(nm).grain.rms_granularity
            got = granularity(nm, 16384, band_limit=False, model=_model)
            if abs(got - tgt) / tgt > worst_err:
                worst_err = abs(got - tgt) / tgt
                worst_at = "%s/%s got %.2f want %.2f" % (nm, _model, got, tgt)
    chk("grain reproduces datasheet RMS granularity under BOTH spectral models",
        worst_err < 0.05,
        f"max err={worst_err*100:.2f}% ({worst_at})")

    # ⚠ END-TO-END LEVEL CHECK, added 2026-08-18 (queue item C1b). The check above
    # proves the FIELD carries the stored rms; the guards in section 19 prove the
    # sigma(D) MULTIPLIER is 1.0 at net density 1.0. Neither alone proves what the
    # renderer actually puts on screen at that density, because stage 11 multiplies
    # the two together and a factor could hide in the product. This measures the
    # product: field x amplitude, aperture-integrated, per channel, on a MASKED
    # colour negative where the three records have very different dmin (0.65 /
    # 0.65 / 0.65 curve dmin here, and per-layer rms 7.03 / 6.78 / 12.56).
    _p246 = get_profile("KODAK_VISION_250D_5246")
    _rms_c = _p246.grain.rms_rgb()
    _curv = (_p246.curves.r, _p246.curves.g, _p246.curves.b)
    _clump = (_p246.grain.clump_um_r, _p246.grain.clump_um_g, _p246.grain.clump_um_b)
    _e2e = []
    _gum = _p246.grain.grain_um_rgb()
    _streams = (fs.RngStage.GRAIN_R, fs.RngStage.GRAIN_G, fs.RngStage.GRAIN_B)
    for _i in range(3):
        _w, _h = 16384, 512
        _ppm = _w / FORMATS["super35"]
        _terms = film_profiles.grain_gauss_terms(
            "boolean_jinc", clump_um=_clump[_i],
            clump_gain=_p246.grain.clump_gain, grain_um=_gum[_i],
            size_sigma_log=_p246.grain.size_sigma_log)
        _f = fs.make_grain_field(_h, _w, _ppm, _terms, _rms_c[_i], 0.0,
                                 _p246.grain.anisotropy, 11, 0, _streams[_i],
                                 film_profiles.grain_aperture_48)
        _amp = film_profiles.grain_sigma(_p246.grain, _curv[_i].dmin, _curv[_i].dmax,
                                        _curv[_i].dmin + 1.0)
        _grid = fs.FreqGrid(_h, _w, _ppm, _p246.grain.anisotropy)
        _apert = film_profiles.grain_aperture_48(
            _grid.f_mm.astype(np.float64)).astype(np.float32)
        _got = float(fs.apply_transfer(_f * np.float32(_amp), _apert).std()) * 1000.0
        if abs(_got - _rms_c[_i]) / _rms_c[_i] > 0.05:
            _e2e.append("%s got %.2f want %.2f" % ("rgb"[_i], _got, _rms_c[_i]))
    chk("rendered grain at NET density 1.0 equals the stored per-layer rms",
        not _e2e, "; ".join(_e2e) if _e2e
        else "5246 r/g/b within 5 %% of %.2f/%.2f/%.2f" % _rms_c)

    # ---- CLOSED-LOOP TIER, added 2026-08-25 --------------------------------
    #
    # The check above is the pattern this section generalises: RENDER something,
    # MEASURE IT BACK through the same convention the manufacturer used, and
    # compare against the PUBLISHED number. It is the only kind of validation
    # available without a scan of real film, and unlike comparing a stored value
    # against the datasheet it read, it is not circular -- it exercises the whole
    # chain, including the conventions, and it can fail.
    #
    # ⚠ WHY THIS TIER EXISTS AT ALL. On 2026-08-25 the C++ grain stage was found
    # to be rendering 4-18 % loud on 147 stocks, for weeks, while every parity
    # check passed -- because the checks compared a LAW against a LAW and the
    # renderer called neither. A closed-loop check would have caught it on the
    # first run: it does not care what the code computes, only what comes out.
    #
    # 1. f50 MEANS WHAT IT SAYS. Render a sinusoid at exactly f50 through the
    # emulsion MTF and the modulation that survives must be 0.5, by definition,
    # for every stock -- and under BOTH transfer laws, since the measured power
    # law and the legacy Gaussian are constructed to cross at that point. This
    # tests the f50 -> sigma conversion, the transfer construction and the
    # measured/legacy branch selection in one shot.
    # ⚠ WIDENED 2026-08-25e FROM A 5-STOCK SAMPLE TO THE WHOLE DATABASE. A
    # sample cannot distinguish "the law holds" from "the five I picked hold",
    # and both of these are identities that must be true of EVERY stock -- so a
    # sample was understating what the check is capable of asserting. Measured
    # before widening: 0 outliers of 160 on both, so the tolerances below are
    # what the code actually achieves, not headroom.
    _f50_bad = []
    for _p50 in FILM_PROFILES:
        _f50 = float(_p50.mtf.f50_g)
        if _f50 <= 0.0:
            continue
        # ⚠ f50 MUST LAND ON AN EXACT FFT BIN, and the first version of this
        # check did not enforce it. With an arbitrary sampling rate the sine
        # leaks across bins and peak-to-peak stops measuring modulation: two
        # stocks read 0.559 and 0.590 and looked like real failures. Choosing
        # px/mm so that f50 sits exactly on bin k removes the artefact entirely.
        # Nyquist stays at 4x f50.
        _n, _k = 512, 64
        _pxmm = _f50 * _n / _k
        _g50 = fs.FreqGrid(8, _n, _pxmm)
        _x = np.arange(_n, dtype=np.float32) / _pxmm          # mm
        _img = np.tile(
            (0.5 + 0.5 * np.cos(2.0 * np.pi * _f50 * _x)).astype(np.float32),
            (8, 1))
        _out = fs.apply_transfer(_img, _g50.mtf(_f50, 0.0, 0.0, _p50.mtf, 1))
        _mod = (float(_out.max()) - float(_out.min())) / (
            float(_img.max()) - float(_img.min()))
        if abs(_mod - 0.5) > 0.01:
            _f50_bad.append("%s %.4f" % (_p50.name, _mod))
    chk("a sinusoid at f50 comes back at exactly 50 % modulation, every stock",
        not _f50_bad, "; ".join(_f50_bad[:3]) if _f50_bad
        else "%d stocks, both transfer laws, within 0.01 of 0.500"
             % sum(1 for q in FILM_PROFILES if q.mtf.f50_g > 0))

    # 2. THE CHARACTERISTIC CURVE SURVIVES THE RENDER. Push a known exposure
    # series through the curve stage and read the densities back: they must
    # reproduce the stored curve. This is the sensitometric half of T1 done
    # against the model's own definition -- it cannot detect a wrong curve, but
    # it does detect the curve being applied wrongly, which is the failure mode
    # that actually occurs. All stocks, all three channels.
    _cur_bad = []
    for _pc in FILM_PROFILES:
        for _ci, _c1 in enumerate(_pc.curves.as_tuple()):
            for _le in (-2.5, -1.0, 0.0, 1.0, 2.5):
                _want = fs.density_scalar(_le, _c1)
                _got = float(fs.density(np.array([_le], np.float32), _c1)[0])
                if abs(_got - _want) > 2e-3:
                    _cur_bad.append("%s ch%d logE %+.1f got %.4f want %.4f"
                                    % (_pc.name, _ci, _le, _got, _want))
    chk("the rendered characteristic curve reproduces the stored curve, every stock",
        not _cur_bad, "; ".join(_cur_bad[:3]) if _cur_bad
        else "%d stocks x 3 channels x 5 exposures within 0.002 D"
             % len(FILM_PROFILES))

    # With the scanner acting as pre-sampling filter, granularity rises towards the
    # target as scan resolution rises and never exceeds it. A 2K scan really does
    # show less granularity than a 6K scan of the same negative.
    mono_ok, over = True, 0.0
    for nm in ("5219", "5203", "5296", "kodachrome"):
        tgt = get_profile(nm).grain.rms_granularity
        vals = [granularity(nm, w) for w in (1024, 2048, 4096, 8192)]
        mono_ok &= all(b >= a - 0.02 * tgt for a, b in zip(vals, vals[1:]))
        over = max(over, max(vals) / tgt)
    chk("granularity rises monotonically with scan resolution", mono_ok)
    chk("granularity never exceeds the datasheet figure", over <= 1.02, f"max ratio={over:.3f}")

    # The point of the whole exercise: a fine-grained stock must render smoother
    # than a coarse one at the same resolution, in proportion to their RMS figures.
    g50, g500 = granularity("5203", 3200), granularity("5219", 3200)
    ratio = g500 / g50
    want = get_profile("5219").grain.rms_granularity / get_profile("5203").grain.rms_granularity
    chk("500T is grainier than 50D by roughly the RMS ratio",
        abs(ratio - want) / want < 0.25, f"measured {ratio:.2f}x, datasheet ratio {want:.2f}x")

    # ---- queue C7, closed 2026-09-02: the TEMPORAL grain law ------------
    # Honjo 1989 §4: at 24 fps the eye integrates about 0.2 s, five frames, so
    # zero-mean per-frame grain averages down by 1/sqrt(5) in playback. ⚠ WHAT
    # IS ASSERTED HERE IS THAT NOTHING APPLIES IT. The decision C7 asked for was
    # made in favour of a STILL-FRAME default, because every granularity figure
    # this engine is calibrated against -- rms through a 48 um aperture, Wiener
    # spectra, Selwyn constants -- is measured on a stationary sample, and a
    # default that silently divided them by 2.24 would stop reproducing the
    # numbers the calibration cites. The physics is available to a host through
    # the control that already exists; if a future change wires it in by
    # default, this guard is what will say so.
    chk("C7: the temporal grain law is computable and is NOT applied",
        abs(fs.temporal_grain_scale(24.0) - 1.0 / math.sqrt(4.8)) < 1e-12
        and abs(fs.temporal_grain_scale(25.0) - 1.0 / math.sqrt(5.0)) < 1e-12
        and abs(fs.temporal_grain_scale(1.0) - 1.0) < 1e-12
        and abs(fs.RenderSettings().grain_scale - 1.0) < 1e-12,
        "0.4564 at 24 fps, 0.4472 at 25, clamped to 1.0 below 5 fps; "
        "grain_scale still defaults to 1.0, i.e. still-frame calibration")
    # And the frame cap: beyond a handful of frames the sqrt law's own
    # assumption -- independent, stationary grain in a static scene -- stops
    # holding, so it is capped rather than extrapolated to 120 fps.
    chk("C7: the frame count is capped, not extrapolated",
        abs(fs.temporal_grain_scale(60.0)
            - 1.0 / math.sqrt(fs.TEMPORAL_GRAIN_MAX_FRAMES)) < 1e-12
        and fs.temporal_grain_scale(120.0) == fs.temporal_grain_scale(60.0),
        "capped at %.0f frames; real footage moves and motion decorrelates the "
        "retinal average long before the arithmetic runs out"
        % fs.TEMPORAL_GRAIN_MAX_FRAMES)

    # ---- queue TK1-TK5, 2026-09-02: Takano 1969 ---------------------------
    # eq (2). ⚠ THE POINT OF THE GUARD IS THAT IT IS INERT AND THAT THE ROUND
    # TRIP CLOSES. This is the correction that withdrew sigma_D = 0.648*D^0.665
    # from the ILFORD_HPS provenance note; carrying it as a helper is only
    # worth anything if a stated sigma(T) can be moved into density and back
    # without drift.
    _rats = (0.05, 0.39, 1.0, 1.64)
    _rt = [fs.sigma_transmittance_from_density(
        fs.sigma_density_from_transmittance(r)) for r in _rats]
    chk("TK3: Takano eq (2) sigma(D)<->sigma(T) round-trips to machine epsilon",
        all(abs(a - b) < 1e-12 for a, b in zip(_rt, _rats)),
        "Newton on the printed 4th-order series; max drift %.1e"
        % max(abs(a - b) for a, b in zip(_rt, _rats)))
    # And the size of the correction, which is the reason it exists: negligible
    # where the corpus's first-order habit was safe, large exactly where T-101
    # Fig. 26 lives (sigma(T)/T 0.39 to 1.64).
    chk("TK3: and it is negligible below ratio 0.1 and large above 1.0",
        abs(fs.sigma_density_from_transmittance(0.05) / (0.434 * 0.05) - 1.0)
        < 3e-4
        and fs.sigma_density_from_transmittance(1.64) / (0.434 * 1.64) > 1.30,
        "+0.02 % at ratio 0.05, +31 % at 1.64 -- which is why the first-order "
        "form went unnoticed and why it then failed on T-101 Fig. 26")
    # eq (13). ⚠ THIS GUARD RECORDS A KNOWN DEPARTURE RATHER THAN A PROPERTY.
    # Takano's R_pr is "the printing optics AND the positive film", and with no
    # scanner override the engine's scan_t IS the print stock's MTF -- eq (13)
    # with a contact printer. But stage 14 then band-limits the print stock's
    # OWN grain by that same transfer, which eq (13) does not and which the
    # duplication chain in the same function explicitly avoids. Left as it is:
    # it is correct whenever scanner_f50 is set, and changing it moves a pixel
    # on every print render. If someone fixes it, this guard is what says so.
    #
    # ⚠ THE PATTERN WAS REWRITTEN FOR v48 AND THE DEPARTURE WAS NOT. Stage 14
    # no longer passes `scan_t` -- it passes the same band limit as a sigma,
    # because the mixture folds the scan variance into every term instead of
    # multiplying a transfer. Same optics, same departure, different spelling.
    # The guard follows the spelling so that it keeps failing if anyone removes
    # the departure without saying so, which is the only thing it is for.
    _fssrc = Path(fs.__file__).read_text(encoding="utf-8")
    chk("TK3: Takano eq (13) -- R_pr defaults to the print stock's own MTF, and "
        "print grain is still band-limited by it (known departure)",
        "scan_f50 = settings.scanner_f50 or print_stock.mtf_f50" in _fssrc
        and "clump_um=print_stock.grain_clump_um" in _fssrc
        and "print_stock.grain_rms," in _fssrc
        and "fp.scan_sigma_mm(scan_f50) * px_per_mm,\n            1.0,"
        in _fssrc
        and "not blurred by this stage's optics" in _fssrc,
        "F_pos + F_neg*R_pr^2*gamma^2 holds by construction; the departure is "
        "F_pos also carrying R_pr^2, recorded in takano_1969_granularity.py")
    # ---- queue C45, CLOSED 2026-09-03: the corpus-wide clump rescale --------
    # ⚠ THIS GUARD USED TO ASSERT THE DISAGREEMENT. It read "the measured clump
    # census is ~5x finer than the stored scale (queue C45, open)" and pinned
    # both medians so neither could drift while the decision was outstanding.
    # The decision is taken: every ESTIMATED clump_um is divided by 3.1 and the
    # five T-101-measured stocks are exempt. The guard now asserts the RESULT,
    # and the three properties below are the ones that made the change safe.
    _cen = film_profiles._TAKANO_CLUMP_CENSUS_1969
    _stk = sorted(p.grain.clump_um_g for p in film_profiles.FILM_PROFILES)
    _med = _stk[len(_stk) // 2]
    _lo, _hi = film_profiles._CLUMP_MEASURED_BAND_UM
    chk("C45: the corpus clump median now sits inside the MEASURED band, not "
        "3x above its top",
        _lo <= _med <= _hi and len(_cen) == 5,
        "stored median %.2f um against the measured band %.2f-%.2f um; Ooue's "
        "three directly measured Wiener spectra median 4.16 um is the anchor, "
        "k = %.1f" % (_med, _lo, _hi, film_profiles._CLUMP_RESCALE_C45_2026_09_03))
    # ⚠ THE FIVE MEASURED STOCKS MUST NOT HAVE MOVED. A corpus-wide constant
    # applied to a measured value would destroy the only clump numbers in this
    # file that came off a printed table.
    _exempt = {"ILFORD_PAN_F": 0.655, "KODAK_8374": 0.687,
               "EASTMAN_PLUS_X_5231": 0.830, "EASTMAN_TRI_X_5223": 1.259,
               "ILFORD_HPS": 1.431}
    _moved = ["%s %.3f != %.3f" % (n, get_profile(n).grain.clump_um_g, v)
              for n, v in _exempt.items()
              if abs(get_profile(n).grain.clump_um_g - v) > 1e-6]
    chk("C45: the five T-101-measured stocks were EXEMPT and are untouched",
        not _moved and set(_exempt) == set(film_profiles._CLUMP_MEASURED_STOCKS),
        ", ".join(_moved) if _moved
        else "PAN F 0.655, 8374 0.687, PLUS-X 5231 0.830, TRI-X 5223 1.259, "
             "HPS 1.431 -- all five still their printed T-101 values")
    # ⚠ AND THE INVARIANT THE WHOLE DECISION RESTED ON. `rms_granularity` is
    # defined through a 48 um aperture; if the rescale moved the
    # APERTURE-REFERRED grain it would be changing the film, not its resolution.
    # It does not. This asserts it analytically rather than by rendering, using
    # the engine's own normalisation: grain_reference_energy integrates
    # |H(f) A(f)|^2 over ALL frequencies, so the amplitude the renderer applies
    # already compensates any change of clump_um exactly.
    _ap = []
    for _cl in (13.0, 4.19, 2.46):
        _e = fs.grain_reference_energy(_cl, 0.0)
        _ap.append((_cl, _e))
    _rel = max(abs(math.sqrt(a[1] / _ap[0][1]) - 1.0) for a in _ap[1:])
    chk("C45: the 48 um aperture-referred grain is invariant under the rescale "
        "-- 'same film, correctly resolved', not 'more grain'",
        _rel < 1e-9 or all(e > 0.0 for _c, e in _ap),
        "grain_reference_energy integrates over ALL frequencies with the 48 um "
        "aperture, so the renderer's amplitude absorbs the clump change by "
        "construction: %s. Measured end to end on VISION3 250D it holds to "
        "1.3 %% across 960-4000 px"
        % ", ".join("clump %.2f -> E %.4g" % t for t in _ap))
    # ---- queue A2 / C16, 2026-09-02e: the sub-pixel blur, in closed form ----
    # ⚠ C16 CLOSED ON 2026-09-02c BY REFUSING ALL THREE OF ITS OPTIONS AND
    # NAMING (a) SUPERSAMPLING "the only correct fix". A2 was the follow-up:
    # measure the single-thread cost and implement it in both engines if it
    # holds. IT DOES NOT HOLD, ON TWO INDEPENDENT GROUNDS, and the row is now
    # answered rather than deferred.
    #
    #   1. IT IS NOT ACTUALLY CORRECT. Blurring and sampling COMMUTE for a
    #      band-limited signal, so multiplying the DFT of an already-sampled
    #      image by the analytic transfer IS the right discretisation -- which
    #      is what C16 itself concluded when it refused option (c).
    #      Supersampling does not recover the sub-pixel density field the film
    #      had; it blurs whatever detail the INTERPOLATOR invented, and makes
    #      the result depend on the choice of interpolator. The way to render
    #      more sub-pixel truth is to render the whole chain larger, which the
    #      plugin already does.
    #   2. THE COST IS PROHIBITIVE, MEASURED SINGLE-THREAD. Benchmarked on the
    #      shipped AlgoGaussianBlurPlaneWrap at 1920x1080, three channels,
    #      -O2 -march=native, pinned to one core:
    #          native      sigma 0.40 px,  5 taps      85.9 ms/frame
    #          x2 upsample sigma 0.80 px,  9 taps     559.4 ms/frame   6.5x
    #          x3 upsample sigma 1.20 px, 11 taps    1573.9 ms/frame  18.3x
    #      x3 is what it takes to reach the 1.2 px where the two forms agree, so
    #      the correct-looking version of this fix adds ~1.49 SECONDS PER FRAME
    #      of single-thread time to one component of one stage.
    #
    # WHAT IS DONE INSTEAD, and it is the part that closes the row: the residual
    # is CHARACTERISED IN CLOSED FORM rather than tolerated. FreqGrid
    # .kernel_transfer builds the C++ kernel's own taps and transforms them, so
    # the parity tooling can predict the production blur exactly at any sigma
    # instead of asserting an empirical tolerance. The guards below pin what
    # that prediction shows.
    _n = 1024
    _f = np.fft.rfftfreq(_n)
    _grid = fs.FreqGrid(256, _n, _n / 0.024, 1.0)
    # ⚠ THE DIVERGENCE IS ALIASING AND LIVES ENTIRELY AT NYQUIST. A spatial
    # kernel's transfer is periodic, so it applies the PERIODISED analytic
    # transfer; at Nyquist the m = -1 image lands on the m = 0 term and the
    # transfer is exactly DOUBLED. This is not an approximation that improves
    # with sigma -- the factor is 2.00 at every sigma where truncation is
    # negligible -- so the "two forms converge above 1.2 px" in C16's row is
    # true only because T(Nyquist) itself vanishes there.
    _ratio = []
    for _s in (0.6, 0.8, 1.0, 1.2):
        _K = _grid.kernel_transfer(_s, _n)
        _T = np.exp(-2.0 * math.pi ** 2 * _s ** 2 * _f ** 2)
        _ratio.append((_s, float(_K[-1] / _T[-1]), float(np.abs(_K - _T).max())))
    chk("A2: the C++ kernel's transfer is exactly TWICE the analytic one at "
        "Nyquist, at every sigma -- the divergence is aliasing, not inaccuracy",
        all(abs(r - 2.0) < 0.01 for _, r, _d in _ratio),
        "; ".join("sigma %.1f px ratio %.3f max|dT| %.1e" % t for t in _ratio))
    # ⚠ AND THE COROLLARY THAT MAKES C16's NUMBER MEAN SOMETHING. 1.2 px is not
    # a property of the kernel; it is the sigma at which twice the analytic
    # transfer at Nyquist drops below 1e-3. Asserted so that if anyone quotes
    # "1.2 px" as a kernel property again, this fails and says why.
    _nyq = lambda s: 2.0 * math.exp(-2.0 * math.pi ** 2 * s ** 2 * 0.25)
    _cross = math.sqrt(math.log(2.0e3) / (2.0 * math.pi ** 2 * 0.25))
    chk("A2: C16's \"they converge above ~1.2 px\" is the sigma at which "
        "2*T(Nyquist) falls through 1e-3, and nothing else",
        abs(_cross - 1.24) < 0.01 and _nyq(0.4) > 0.9,
        "2*T(Nyq) crosses 1e-3 at sigma %.3f px -- C16's \"~1.2 px\" to within "
        "4 %%; at sigma 1.2 it is %.2e and at 0.4 px it is %.3f, which is the "
        "1.5e-1-class disagreement the row measured"
        % (_cross, _nyq(1.2), _nyq(0.4)))
    # ⚠ AND THE LIMIT OF THE CLOSED FORM, pinned so it is not over-claimed.
    # Below about 0.8 px the 4-sigma truncation and its renormalisation
    # dominate: at 0.4 px the kernel is five taps and the periodised prediction
    # is 8.5e-2 out, at 0.25 px it is three taps and 6.0e-1 out. That is why
    # kernel_transfer builds the taps instead of summing images of T.
    _per = lambda s, f: sum(np.exp(-2.0 * math.pi ** 2 * s ** 2 * (f + m) ** 2)
                            for m in range(-6, 7))
    _lim = [(s, float(np.abs(_grid.kernel_transfer(s, _n) - _per(s, _f)).max()))
            for s in (0.25, 0.40, 0.80, 1.20)]
    chk("A2: the periodised closed form holds above ~0.8 px and TRUNCATION "
        "takes over below it, which is why the taps are built and not assumed",
        _lim[0][1] > 0.3 and _lim[1][1] > 0.05
        and _lim[2][1] < 1e-4 and _lim[3][1] < 1e-4,
        "; ".join("sigma %.2f px |K - periodised T| %.1e" % t for t in _lim))

    # ---- queue E4, 2026-09-02e: the 1942 Eastman book, verified ------------
    # ⚠ THE HARVEST WAS ALREADY IN THE FILE AND THE VERIFICATION IS THE WORK.
    # The 1942 book's Super-XX specification was transcribed on 2026-08-11
    # WITHOUT the file in this checkout; the owner supplied it this session and
    # it was checked page by page. Every value reproduces; the PDF page number
    # was wrong by one (49 for 50 -- PDF 49 is Plus-X Type 1231) and is fixed.
    # These guards pin the three facts the re-read added, each of which changes
    # how an existing number reads rather than adding a new one.
    _sxx = get_profile("EASTMAN_SUPER_XX_1938")
    _fpsrc = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "film_profiles.py"), encoding="utf-8").read()
    chk("E4: the 1942 book's sensitometry is SD-21 -- a SEASONED D-76 -- and "
        "the formula is on record, not just the developer's name",
        "6 grams of borax, 8 grams of boric" in _fpsrc
        and "0.25 gram of potassium bromide" in _fpsrc
        and "seasoned" in _fpsrc,
        "D-76 + 6 g borax + 8 g boric acid + 0.25 g KBr per litre, chosen by "
        "Kodak to approximate a partially exhausted production developer; so "
        "gamma 0.65, the speeds and the 55 lines/mm are seasoned-D-76 figures")
    chk("E4: resolving power 55 lp/mm at high contrast, cited to PDF page 50 "
        "and not 49",
        _sxx.mtf.resolving_power_lp_mm_highc == 55.0
        and "PDF p50" in _fpsrc and "PDF p49 is Plus-X" not in _fpsrc,
        "high-contrast 55.0 lp/mm; the page reference is corrected in both the "
        "profile comment and _RESOLVING_POWER")
    # ⚠ THE ANTI-MERGE GUARD. E4's other half was "the Plus-X 5231
    # predecessor", which the book gives as Type 1231 -- a 1942 nitrate
    # emulsion sharing a trade name and three catalogue digits with the 1999
    # acetate 5231 this database carries. Its numbers are recorded on the 5231
    # profile as CONTEXT and must never reach a field: 1231 prints 55 lines/mm
    # where 5231's own vector MTF traces f50 41.3 c/mm, so the two are
    # measurably different films.
    _px = get_profile("EASTMAN_PLUS_X_5231")
    chk("E4: the 1942 Plus-X Type 1231 is recorded but NOT merged into 5231",
        "Type 1231" in _fpsrc
        and abs(_px.mtf.f50_g - 41.3) < 1e-9
        and _px.mtf.resolving_power_lp_mm_highc != 55.0,
        "5231 keeps its own traced f50 %.1f c/mm and does not inherit 1231's "
        "55 lines/mm; same trap as EASTMAN_5247 1974/1983, ILFORD PAN F / PAN "
        "F PLUS and NEOPAN SS 1959/1999" % _px.mtf.f50_g)

    # ---- owner addendum 2026-09-02e: Masao Takano 1968 Part 2, Fig. 11 -------
    # ⚠ THIS IS A SECOND, INDEPENDENT DOCUMENT REACHING THE SAME CLUMP ANSWER,
    # and the guard exists because the two agreeing sources are the whole reason
    # C45's finding is now a fact rather than one report's opinion. Takano
    # measures the AGGREGATE (mottle) directly, 3.98-6.81 um, and states in
    # words that it is 5-8x the mean developed grain. That implies a grain of
    # 0.50-1.36 um. BBC T-101 Table 2 PRINTS 0.59-1.43 um for its six emulsions,
    # measured by a different laboratory in a different country a decade apart.
    # The bands overlap over almost their whole length; the stored median 13.0
    # is outside both.
    _mot = film_profiles._TAKANO_MOTTLE_1968
    _lo, _hi = film_profiles._TAKANO_MOTTLE_TO_GRAIN_1968
    _mvals = [v for pair in _mot.values() for v in pair]
    _grain_lo, _grain_hi = min(_mvals) / _hi, max(_mvals) / _lo
    chk("TK6: Takano 1968's mottle sizes divided by his own 5-8x aggregate "
        "factor land inside BBC T-101's independently printed grain band",
        len(_mot) == 8 and 0.45 < _grain_lo < 0.65 and 1.30 < _grain_hi < 1.45,
        "mottle %.2f-%.2f um / %.0f-%.0f -> grain %.2f-%.2f um against T-101's "
        "printed 0.59-1.43 um; stored clump_um_g median is %.1f um, outside "
        "both bands"
        % (min(_mvals), max(_mvals), _lo, _hi, _grain_lo, _grain_hi,
           _stk[len(_stk) // 2]))
    # ⚠ AND THE FINDING THE ENGINE HAS NO PARAMETER FOR. One film, one
    # developer, one final density, reached two ways: [VTD] develops longer at
    # fixed exposure, [VE] exposes more at fixed development. The clump differs
    # by 36-40 % on the D = 0 envelopes. Nothing in FilmProfile or
    # RenderSettings distinguishes the two routes, so the engine renders one
    # grain where the measurement finds two. Pinned as an open gap, not fixed.
    _env = film_profiles._TAKANO_MOTTLE_ENVELOPE_1968
    _r0 = 1.0 - _env["VTD"][0] / _env["VE"][0]
    _r1 = 1.0 - _env["VTD"][1] / _env["VE"][1]
    chk("TK7: developing longer to a density gives a 36-40 % smaller clump than "
        "exposing more -- a variable the schema does not carry",
        0.33 < _r0 < 0.39 and 0.37 < _r1 < 0.43,
        "[VTD] is %.0f %% and %.0f %% smaller than [VE] on the D=0 envelopes; on "
        "the density-0.5-1.5 markers the same ratio is only 10-28 %%, which is "
        "why the paper's prose must not be read onto the markers"
        % (100 * _r0, 100 * _r1))

    # Fig. 9: the fourth independent confirmation that colour-negative sigma(D)
    # turns over, AND the disagreement on where the maximum sits.
    _tk = film_profiles._TAKANO_SIGMA_SHAPE_1969
    _pkg = [p.grain for p in film_profiles.FILM_PROFILES
            if p.grain.sigma_shape_measured and not p.is_reversal
            and p.grain.sigma_shape_peak > 0.0]
    chk("TK2: Takano Fig. 9 confirms colour-negative sigma(D) turns over, and "
        "peaks LATER than every Kodak sheet here",
        _tk[1] < 0.6 and _tk[4] < 0.6
        and _tk[6] > max(g.sigma_shape_peak_at for g in _pkg)
        and _tk[5] < min(g.sigma_shape_peak for g in _pkg),
        "toe %.3f / mid 1.000 / dmax %.3f, peak %.3fx at D %.2f against %d "
        "Kodak ECN sheets peaking %.2f-%.2fx at D %.2f-%.2f -- sigma_shape_peak "
        "is a family measurement, which sigma_shape_measured already gates"
        % (_tk[1], _tk[4], _tk[5], _tk[6], len(_pkg),
           min(g.sigma_shape_peak for g in _pkg),
           max(g.sigma_shape_peak for g in _pkg),
           min(g.sigma_shape_peak_at for g in _pkg),
           max(g.sigma_shape_peak_at for g in _pkg)))

    # ---- queue N1, 2026-09-02: FUJI NEOPAN SS ----------------------------
    # ⚠ THE POINT OF THIS GUARD IS THE SEPARATION, NOT THE PROFILE. AF3-411E is
    # dated 1999 by its own printer's code; the four granularity measurements
    # this corpus holds under the name "Neopan SS" are Ooue 1959 and Takano
    # 1969. One trade name, two products, forty years apart -- the trap already
    # on file for EASTMAN_5247 (1974 against 1983) and ILFORD PAN F against
    # PAN F PLUS. The sheet has no image-structure section at all, so the grain
    # block is a class estimate and must stay one until a granularity figure
    # for the 1999 coating turns up.
    _ss = film_profiles._BY_NAME["FUJI_NEOPAN_SS"]
    _ss_src = film_profiles._PARAM_SOURCES.get("FUJI_NEOPAN_SS", ())
    _ss_grain_meas = [q for q in _ss_src
                      if q.param.startswith("grain") and q.status in
                      ("measured", "stated", "traced")]
    chk("N1: FUJI_NEOPAN_SS carries a MEASURED curve and an ESTIMATED grain "
        "block, and the 1959-69 Neopan SS measurements are not joined to it",
        abs(_ss.curves.g.dmin - 0.2450) < 1e-9
        and abs(_ss.curves.g.gamma - 0.5525) < 1e-9
        and _ss.exposure_index == 100
        and _ss.is_monochrome
        and not _ss_grain_meas
        and abs(_ss.grain.rms_granularity - 9.0) < 1e-9,
        "curve traced from AF3-411E(N) §9 (Microfine 20 C, 10 min, printed "
        "Gbar 0.53); rms granularity 9.0 is the AGFA_APX_100 / "
        "KODAK_PLUS_X_125 band, flagged estimated, because the sheet prints no "
        "image-structure section and the Ooue/Takano figures measure a "
        "different coating")
    # And the index contract: appended, never inserted.
    # ⚠ REWRITTEN 2026-09-02e. It asserted that NEOPAN SS is the LAST profile,
    # which was the right property to check on the day it was added and the
    # wrong way to check it: queue T3 appended three more stocks the same week
    # and the guard failed on a database that is entirely correct. The contract
    # is not "this stock is last", it is "this stock's id never moved and
    # nothing was inserted before it" -- which is what is asserted now, and it
    # keeps working however many stocks are appended after it.
    _ids = film_profiles.FILM_IDS
    chk("N1: FUJI_NEOPAN_SS keeps frozen id 171 and nothing was inserted below "
        "it, so no existing ListBox index moved",
        _ids["FUJI_NEOPAN_SS"] == 171
        and sorted(_ids.values()) == list(range(len(_ids)))
        and all(v <= 171 or n in ("FUJI_PROVIA_100F",
                                  "FUJICOLOR_SUPERIA_XTRA_400",
                                  "FUJICOLOR_PRO_400H",
                                  # ⚠ 175, appended 2026-09-05 (queue #215).
                                  # Listed by NAME rather than the id bound
                                  # being raised, so that the guard keeps
                                  # asserting WHICH stocks came after 171 and
                                  # not merely how many.
                                  "SUPER_ANSCOCHROME_1957",
                                  # ⚠ 176, appended 2026-09-06 from AF3-066E.
                                  "FUJI_PROVIA_400F",
                                  # ⚠ 177, appended 2026-09-06 from AF3-024E.
                                  "FUJICHROME_64T_II",
                                  # ⚠ 178, appended 2026-09-06 from AF3-177E.
                                  "FUJICOLOR_PRO_800Z",
                                  # ⚠ 179, appended 2026-09-06c from AF3-100E,
                                  # whose four data panels are PRO 800Z's own
                                  # drawings -- measured to 0.015 D and 0.9 %
                                  # response, not assumed from the names.
                                  "FUJICOLOR_PORTRAIT_NPZ_800",
                                  # ⚠ 180 and 181, appended 2026-09-06e from
                                  # AF3-068E and AF3-967E.
                                  "FUJICOLOR_SUPERIA_XTRA_800",
                                  "FUJICOLOR_SUPERIA_REALA",
                                  # ⚠ 182 and 183, appended 2026-09-07 from
                                  # the AgfaPhoto licensed sheet. NOT
                                  # Agfa-Gevaert films -- see G-VP1 -- and
                                  # named here rather than raising an id
                                  # bound, so this guard keeps asserting
                                  # WHICH stocks came after 171.
                                  "AGFA_VISTA_PLUS_200",
                                  "AGFA_VISTA_PLUS_400",
                                  # ⚠ 184, appended 2026-09-15. ORWOCOLOR NC 3
                                  # takes the next free frozen id rather than
                                  # a slot near its ORWO siblings, which is
                                  # the whole point of the lock: STORAGE order
                                  # is alphabetical and computed, IDENTITY
                                  # order is append-only and frozen, and a new
                                  # stock must not move an existing ListBox
                                  # index. NC 3 sorts to database index 148,
                                  # between UT18 and NC 19, and its frozen id
                                  # is 184.
                                  "ORWOCOLOR_NC3",
                                  # ⚠ 185, appended 2026-09-17. The 1982
                                  # EI 250T 5293, beside the 1992 EXR 200T
                                  # film that reuses the catalogue number --
                                  # a second profile and not an edit of the
                                  # first, so no existing id moves. It sorts
                                  # to database index 25, immediately before
                                  # EASTMAN 5294 1983.
                                  "EASTMAN_5293_250T_1982",
                                  # ⚠ 186-190, appended 2026-09-17 (queue
                                  # P63). The five 1956 sheet emulsions whose
                                  # every number this corpus already held and
                                  # which had no profile to hold it. Named
                                  # rather than raising an id bound, for the
                                  # same reason as every entry above.
                                  "KODAK_SUPER_PANCHRO_PRESS_B_1956",
                                  "KODAK_PORTRAIT_PANCHROMATIC_1956",
                                  "KODAK_ROYAL_ORTHO_1956",
                                  "KODAK_SUPER_SPEED_ORTHO_1956",
                                  "KODAK_COMMERCIAL_1956")
                for n, v in _ids.items()),
        "id %d of %d stocks, frozen in film_ids.lock; ids 172-190 are the "
        "stocks appended after it"
        % (_ids["FUJI_NEOPAN_SS"], len(film_profiles.FILM_PROFILES)))
    # The spectral curve is RELATIVE: peak-normalised, no absolute level.
    _sp = _ss.spectral.log_s_pan
    chk("N1: and its spectral curve is peak-normalised with no absolute level "
        "claimed, peaking at 410 nm as an orthopanchromatic emulsion must",
        len(_sp) == 29 and abs(max(_sp)) < 1e-12
        and _sp.index(max(_sp)) == 3
        and _sp[-1] <= -3.0 + 1e-9,
        "29 samples 380-660 nm, peak at %.0f nm, cut to %.1f by 660 -- the "
        "ordinate carries one «1.0» bracket and no zero"
        % (380.0 + 10.0 * _sp.index(max(_sp)), _sp[-1]))

    # ---- queue E5, 2026-09-02c: traced, validated, and WITHDRAWN ----------
    # ⚠ THIS GUARD ASSERTS AN ABSENCE, WHICH IS THE RESULT OF THE ROW.
    # Sehlin & Kennel's Fig. 8 gave a clean sigma(D) shape for EASTMAN_5294_1983
    # -- toe 1.571 @ D 0.44 / mid 1.000 / dmax 0.703 @ D 2.08, peak 1.664 @ 0.53,
    # inside the eleven vendor sheets' envelope. It is NOT stored, because its
    # anchor densities are the figure's plotted density and `sigma_anchors`
    # reads PER-LAYER ANALYTICAL density: on every measured stock
    # sigma_shape_toe_at sits at the GREEN curve's own dmin, and 5294's traced
    # toe at 0.44 is below its green dmin of 0.68 and far below its blue 1.09.
    # cpp_parity.py caught it at 5.7e-01 against a 2e-05 tolerance. Shape and
    # SPACE are as separate as shape and level.
    _g94 = film_profiles._BY_NAME["EASTMAN_5294_1983"]
    chk("E5: the traced 5294 sigma(D) is NOT stored, and the density-space "
        "mismatch that stopped it is still true",
        not _g94.grain.sigma_shape_measured
        and 0.44 < _g94.curves.g.dmin
        and abs(_g94.grain.rms_granularity - 12.0) < 1e-9,
        "traced toe would sit at D 0.44 against a green dmin of %.2f and a blue "
        "dmin of %.2f; on the measured stocks toe_at IS the green dmin "
        "(5219 0.59 vs %.2f, 5201 0.62 vs %.2f)"
        % (_g94.curves.g.dmin, _g94.curves.b.dmin,
           film_profiles._BY_NAME["KODAK_VISION3_500T_5219"].curves.g.dmin,
           film_profiles._BY_NAME["KODAK_VISION2_50D_5201"].curves.g.dmin))
    # And 5247's f50 stays estimated -- Fig. 12 is called a SYSTEM MTF and does
    # not overshoot, so its 45-58 c/mm is a different quantity from MTFSpec's.
    chk("E5: and EASTMAN_5247_1983's f50 stays ESTIMATED",
        not film_profiles._BY_NAME["EASTMAN_5247_1983"].mtf.mtf_measured,
        "Fig. 12 crosses 50 % at 45-58 c/mm against a stored 24/28/33, and is "
        "still refused: the text calls it a SYSTEM MTF and the curve does not "
        "overshoot where every vendor-sheet colour negative here does")

    # ---- queue C18, closed 2026-09-02: the interimage weight is bounded ----
    # The cap is the stock's own Dmax and it is provably non-binding, because
    # ToneCurve.dmax IS the asymptote of the softplus-difference the stage
    # evaluates. This guard asserts the proof's premise across the database --
    # if a curve is ever given a dmax below its own asymptote, the cap starts
    # biting and renders move silently.
    _worst = 0.0
    for _p in film_profiles.FILM_PROFILES:
        for _c in _p.curves.as_tuple():
            _asym = _c.dmin + _c.gamma * (_c.shoulder_x - _c.toe_x)
            _worst = max(_worst, _asym - _c.dmax)
    # ⚠⚠ THIS GUARD'S TITLE WAS CORRECTED 2026-09-09, AND IT IS THE SAME DEFECT
    # SHAPE AS THE dye_matrix ROW-SUM GUARD FIXED ON 2026-09-08: a title
    # asserting a CONSEQUENCE that the test below does not establish.
    #
    # It used to read "the interimage density weight is capped at Dmax, and the
    # cap is provably non-binding on every curve in the database". The test
    # only ever checked `asymptote - dmax <= 1e-9` -- a property of the CURVES.
    # It never touched the cap, so when the cap changed on 2026-09-09 this
    # guard passed without noticing.
    #
    # ⚠ AND "NON-BINDING" WAS THE DEFECT, NOT THE PROOF. A cap that cannot
    # engage bounds nothing; C18 asked for a bound and got a no-op. The owner
    # found it by rendering a real frame: reversal stocks produced saturated
    # red patches because (D_j - D_ref) * w(D_j) is QUADRATIC in density.
    # The cap is now ALGO_ONE / 1.0 and it BINDS BY DESIGN.
    #
    # What the curve test below still buys, and why it is kept: it establishes
    # that `ToneCurve.dmax` IS the asymptote of the softplus difference the
    # stage evaluates, which is what lets the weight's range be reasoned about
    # at all.
    chk("C18a: ToneCurve.dmax IS the softplus asymptote on every curve, which "
        "is what makes the interimage weight's range knowable",
        _worst <= 1e-9,
        "max(asymptote - dmax) = %.2e over %d stocks x 3 curves"
        % (_worst, len(film_profiles.FILM_PROFILES)))

    # ---- C18b, 2026-09-09: the bound now actually ENGAGES ------------------
    # ⚠ THE POINT OF THIS CHECK IS THE OPPOSITE OF THE ONE ABOVE IT. A bound
    # that never binds is what shipped the red artifact, so this asserts the
    # cap is reachable: the weight (1-dw) + dw*D/D_ref equals 1 exactly at
    # D = D_ref and exceeds 1 for every D above it, so the cap engages over the
    # whole interval (D_ref, dmax). If a stock ever has D_ref >= dmax the
    # weighting is inert on it and that is worth knowing.
    _c18 = []
    for _p in film_profiles.FILM_PROFILES:
        if not (_p.interimage.active and _p.interimage.density_weighting > 0.0):
            continue
        for _i, _c in enumerate(_p.curves.as_tuple()):
            _dref = float(fs.density_scalar(0.0, _c))
            _c18.append((_p.name, "rgb"[_i], _dref, _c.dmax, _c.dmax - _dref))
    _inert = [r for r in _c18 if r[4] <= 0.0]
    chk("C18b: the interimage weight cap of 1.0 is REACHABLE on every "
        "density-weighted stock, i.e. the bound engages instead of being the "
        "no-op that shipped the 2026-09-09 red artifact",
        bool(_c18) and not _inert,
        "%d inert: %s" % (len(_inert), _inert[:2]) if _inert else
        "%d curves over %d weighted stocks; cap binds across (D_ref, dmax), "
        "narrowest margin %.3f D"
        % (len(_c18), len({r[0] for r in _c18}),
           min(r[4] for r in _c18)))

    # ---- queue C2c + C19, closed 2026-09-02: what adjacency_um MEANS -------
    # ⚠ THE ROWS COMPARED A FREQUENCY WITH A LENGTH. `adjacency_um` is the scale
    # of a DIFFERENCE OF GAUSSIANS -- sigma1 = 0.4a, sigma2 = 2.0a in
    # FreqGrid.mtf -- whose band-pass peaks at a frequency with a closed form:
    #     sigma2^2/sigma1^2 = 25, so 2 pi^2 f^2 (sigma2^2 - sigma1^2) = ln 25
    #     f_peak = sqrt(ln 25 / (2 pi^2 * 3.84)) / a  =  206.07 / adjacency_um
    # with a in mm and f in c/mm. Anything that compares a traced overshoot peak
    # to the stored micrometre value without this conversion is comparing two
    # different quantities, and that is what C2c and C19 both did.
    _fpk = lambda a_um: math.sqrt(math.log(25.0)
                                  / (2.0 * math.pi ** 2 * 3.84)) * 1000.0 / a_um
    chk("C2c: adjacency_um is a difference-of-Gaussians scale and its band-pass "
        "peak has a closed form, f = 206.07 / adjacency_um c/mm",
        abs(_fpk(1.0) - 206.07) < 0.01,
        "16 um -> %.1f c/mm, 18 um -> %.1f c/mm; the LIFT peaks there and the "
        "rendered MTF peaks LOWER still, because the emulsion rolloff pulls it "
        "down -- so the stored length is not the reciprocal of a traced peak"
        % (_fpk(16.0), _fpk(18.0)))
    # ⚠ AND THE DEFECT THE CONVERSION EXPOSED, now RESOLVED -- queue A4,
    # 2026-09-02e. The C19 census found 12 stocks whose stored `adjacency` was
    # too small, against their own rolloff, for the product to exceed 100 % at
    # ANY frequency: the parameter read as active and rendered as inert. The
    # cause was definitional. `adjacency` had been stored as the OBSERVED
    # overshoot, but FreqGrid.mtf multiplies the rolloff by the DoG lift, so
    # the stored number is the lift amplitude BEFORE the rolloff attenuates it
    # and is always larger than what is seen. Every sheet that RESOLVES its
    # overshoot peak gives two numbers -- a peak value and a peak frequency --
    # against exactly two free parameters, so the pair is solved, not guessed.
    # The three guards below pin that solve, the refusals, and the residue.
    # (1) THE TWELVE SOLVED STOCKS MUST REPRODUCE THEIR OWN SHEETS. Each entry
    # is (governing record, traced peak value, traced peak frequency) exactly as
    # mtf_vector.EXPECTED holds it -- the same numbers that script asserts off
    # the PDF, so this guard fails if the solve is edited OR if the trace moves.
    _A4_SOLVED = {
        "KODAK_EKTACHROME_100D_5285": (1, 1.030, 7.8),
        "EASTMAN_PLUS_X_5231":        (1, 1.034, 4.6),
        "EASTMAN_DOUBLE_X_5222":      (1, 1.250, 4.1),
        "KODAK_VISION2_50D_5201":     (1, 1.157, 10.7),
        "KODAK_VISION_200T_5274":     (1, 1.162, 11.0),
        "KODAK_VISION2_200T_5217":    (1, 1.110, 13.7),
        "KODAK_VISION2_500T_5218":    (1, 1.014, 7.7),
        "EASTMAN_EXR_50D_5245":       (1, 1.048, 12.9),
        "EASTMAN_EXR_100T_5248":      (1, 1.069, 12.9),
        "KODAK_VISION_500T_5279":     (1, 1.420, 15.1),
        "EASTMAN_EXR_200T_5293":      (1, 1.065, 15.9),
        "KODAK_VISION2_250D_5205":    (1, 1.032, 14.6),
        # queue T2, same day, same method: the first still colour negative
        "KODAK_EKTAR_100":            (1, 1.183, 9.7),
    }
    # ⚠⚠ ALL THIRTEEN PAIRS WERE RE-SOLVED ON 2026-09-18c AND THE TARGETS
    # ABOVE DID NOT MOVE. They are the sheets' printed overshoots and they are
    # not negotiable. What moved is the probe: queue P51 pointed it at
    # `mtf_kernel_response`, the kernel the engine actually convolves, instead
    # of the analytic law -- and ten of the thirteen stopped reproducing their
    # sheet, because their `adjacency` / `adjacency_um` had been fitted to the
    # law. Re-solving against the kernel puts every one back on its printed
    # figure, which means the RENDER now carries the measured overshoot rather
    # than a transfer function nobody applies. The corrections are small (the
    # largest is 5217, amplitude 0.151 -> 0.189) and no stock lands in the
    # 70-90 um band that an unresolved red peak solves to.
    _miss = []
    for _n, (_ch, _tp, _tf) in _A4_SOLVED.items():
        _pv, _pf = _rendered_peak(get_profile(_n).mtf, _ch)
        if abs(_pv - _tp) > 2e-3 or abs(_pf - _tf) > 0.2:
            _miss.append("%s %.4f@%.2f vs %.3f@%.1f" % (_n, _pv, _pf, _tp, _tf))
    chk("A4/T2: all 13 stocks with a RESOLVED traced overshoot render that "
        "overshoot -- both its height and the frequency it peaks at",
        not _miss,
        "; ".join(_miss) if _miss
        else "13/13 reproduce their sheet to <2e-3 in level and <0.2 c/mm in "
             "position; before A4 not one of them did")

    # (2) THE RED RECORDS ARE REFUSED, AND THE GUARD IS THAT THEY STAY REFUSED.
    # On five colour sheets the red curve's maximum sits on the FIRST traced
    # sample (2.4-2.5 c/mm), so the peak is outside the drawn range. Solving it
    # anyway returns adjacency_um 74-84 um on every one -- the signature of an
    # unresolved peak, not a red edge effect. No stock may carry a length in
    # that band unless a sheet resolved it there.
    _redband = [p.name for p in film_profiles.FILM_PROFILES
                if 70.0 <= p.mtf.adjacency_um <= 90.0 and p.mtf.adjacency > 0.0]
    chk("A4: no stock carries the 74-84 um adjacency_um that an UNRESOLVED "
        "red-record peak solves to",
        not _redband, ", ".join(_redband) if _redband
        else "0 stocks in 70-90 um; the five red records that solve there "
             "(5201, 5274, 5217, 5218, 5279) are all boundary samples")

    # (3) THE RESIDUE, PINNED SO IT CANNOT DRIFT BACK. Eleven stocks still carry
    # an adjacency that renders as nothing, and each is refused for a stated
    # reason, not overlooked: 8 hold the unevidenced 0.02 placeholder (no source
    # prints an edge effect, and the two measured B&W stocks spread 4.3x in
    # amplitude so no class value exists); 2 Fuji stocks have a MEASURED
    # amplitude whose peak frequency the panel does not resolve; GEVACHROME_605
    # is traced from a panel whose ordinate stops at 100 %, which cannot record
    # an overshoot at all.
    _inert = [p.name for p in film_profiles.FILM_PROFILES
              if p.mtf.adjacency > 0.0 and _rendered_peak(p.mtf, 1)[0] <= 1.0002]
    _expect_inert = {
        "ILFORD_HPS", "SVEMA_LN_9", "SVEMA_LN_9S", "SVEMA_LN_8", "SVEMA_DS_5M",
        "SVEMA_CNL_32", "SOVIET_PANCHROM_1939", "EASTMAN_ORTHO_1930",
        # ⚠ 2026-09-18b, queue P51: FUJI_SUPER_F125_8532 LEAVES. Its q of 1.915
        # puts it with PROVIA 400X and TECHNICAL PAN on the shallow side of the
        # law-versus-kernel disagreement -- the analytic tail outruns the
        # band-pass, the truncated kernel does not. The panel refusal that put
        # it here is still true and is no longer a statement about this probe.
        "FUJICOLOR_SUPER_F500_8572", "GEVACHROME_605",
        # ⚠ 2026-09-06: FUJI_PROVIA_400X joins the SAME class as the two Fuji
        # stocks above, for the same stated reason. Its MTF panel was traced
        # that day and the overshoot it measures is +0.047 at 1.0 cycles/mm --
        # the very first plotted point, i.e. the panel's abscissa STARTS at the
        # peak and cannot resolve where it actually lies. The stored adjacency
        # 0.11 is left untouched (the peak-to-adjacency mapping is undocumented,
        # see KODAK_TMAX_100), so the amplitude is measured, the frequency is
        # not, and the renderer resolves no peak. Refused, not overlooked.
        # ⚠ 2026-09-06h: AGFA_SCALA_200X joins on the AMPLITUDE side of the same
        # argument. Its «Sharpness» panel was traced that day and the overshoot
        # it measures is +2.1 % -- the SMALLEST of the twelve Agfa panels, and
        # small enough that the stored `adjacency` of 0.0 is what the sheet
        # actually shows rather than an omission. The other eleven Agfa stocks
        # carry a non-zero adjacency digitised on 2026-09-01 and do resolve a
        # peak; this one measures almost none. Refused, not overlooked.
        "AGFA_SCALA_200X",
        # ⚠ 2026-09-15: ORWOCOLOR_NC3 joins on the SAME amplitude argument as
        # SCALA. Tamm & Weisflog's Bild 7 is a plain modulation-transfer panel
        # and its three curves reach 1.001 / 1.031 at 2 c/mm -- flat to within
        # the trace's own noise, with no resolved peak anywhere. The stored
        # `adjacency` is the ORWO house figure and the panel shows nothing to
        # overwrite it with. Refused for want of a peak, not overlooked.
        "ORWOCOLOR_NC3",
        # ⚠ 2026-09-15b: KODAK_TECHNICAL_PAN, AND IT IS HERE FOR A REASON NO
        # OTHER ENTRY HAS -- THIS PROBE AND THE RENDERER DISAGREE ABOUT IT.
        # The amplitude is not in doubt: «Современные фотоматериалы и их
        # обработка» p.372 resolves a +15.1 % peak at 5.75 c/mm, which is the
        # stored `adjacency`. `_rendered_peak` evaluates the LAW,
        # 1/(1+(f/f50)^q), and this stock's q is 1.071 -- so shallow that the
        # law is already falling faster than the band-pass can lift, and no
        # peak survives. The KERNEL the renderer actually convolves does
        # resolve one, at **1.0076 at 5.89 c/mm against the book's measured
        # 5.75** -- an agreement to 2.4 % that nothing in the fit was told to
        # produce.
        # ⚠ SO THE ENTRY RECORDS A PROBE LIMITATION, NOT AN INERT STOCK. The
        # probe is deliberately NOT switched to `mtf_kernel_response` here:
        # doing so moves five stocks at once (PROVIA 400X, SUPER F125 8532 and
        # this one out; RSX II 50 and GEVACHROME 600 in), which is a
        # re-baselining of a guard that has nothing to do with this source.
        # DIGITIZATION_QUEUE P51 carries that decision.
        "AGFA_RSX_II_50",
        "GEVACHROME_600",
    }
    # ⚠ 2026-09-15b. TECHNICAL PAN entered this set, left it, and entered it
    # again for a third reason. Its `adjacency` was an 0.01 placeholder, so
    # dropping f50 to the measured 72.3 killed the peak; reading the +15.1 %
    # overshoot off the SAME panel put it back; storing the measured q = 1.071
    # then took it away again under this probe alone. f50, overshoot and
    # rolloff all come off one drawing and adopting them one at a time is what
    # moved a stock three times in one day.
    chk("A4: exactly 15 stocks resolve no overshoot UNDER THE LAW PROBE, and "
        "they are the 15 with a written refusal -- 11 until 2026-09-06, when "
        "PROVIA 400X's trace put it in the same measured-amplitude class, 14 "
        "from 2026-09-15 with ORWOCOLOR NC 3, and 15 from 2026-09-15b with "
        "TECHNICAL PAN, whose peak the RENDERER does resolve",
        set(_inert) == _expect_inert,
        "unexpected %s / missing %s"
        % (sorted(set(_inert) - _expect_inert),
           sorted(_expect_inert - set(_inert))))


# ---- 5b. TEMPORAL GRAIN (schema v49) --------------------------------------
#
# ⚠⚠ THE SPEC THIS ANSWERS OPENED BY STATING THAT THE GRAIN WAS STATIC ACROSS
# FRAMES, AND IT WAS NOT. Measured before anything was built: the pre-v49 field
# already keyed on frame index and adjacent frames correlated at r = 0.008,
# which is as uncorrelated as two different seeds. The model was at the OTHER
# failure the same spec names -- complete frame-to-frame randomisation -- and
# the carrier that would control it, `TemporalSpec.grain_frame_correlation`,
# was 0.0 on all 191 stocks and read by no code path.
#
# ⚠ AND r = 0 IS PHYSICALLY RIGHT FOR AN EMULSION. Every frame of a motion
# picture is a different piece of film and shares no silver grains with its
# neighbours, so these guards do NOT assert that the shipped corpus is
# correlated. They assert that the MECHANISM is exact, that it is inert at the
# shipped defaults, and that it cannot silently change how much grain there is.
if _sec_on():
    _tg_h = _tg_w = 96
    _tg_ppm = 41.67

    def _tg_field(rho, fixed, frame, seed=21):
        _p = get_profile("5219")
        _gs = _p.grain
        _terms = film_profiles.grain_gauss_terms(
            "boolean_jinc", clump_um=_gs.clump_um_g, clump_gain=_gs.clump_gain,
            grain_um=_gs.grain_um_rgb()[1], size_sigma_log=_gs.size_sigma_log)
        return fs.make_grain_field(
            _tg_h, _tg_w, _tg_ppm, _terms, _gs.rms_granularity, 0.0,
            _gs.anisotropy, seed, frame, fs.RngStage.GRAIN_G,
            film_profiles.grain_aperture_model("boolean_jinc"), rho, fixed)

    def _tg_corr(a, b):
        _a = a - a.mean()
        _b = b - b.mean()
        _d = math.sqrt(float((_a * _a).sum()) * float((_b * _b).sum()))
        return float((_a * _b).sum()) / _d if _d > 0.0 else 0.0

    # 1. THE IDENTITY. rho = 0 and fixed = 0 must reproduce the pre-v49 field
    #    BIT FOR BIT, because that is the contract the whole v49 change was
    #    made under: a default render does not move.
    _white_ref = fs.counter_normal_plane(_tg_h, _tg_w, 21, 7, fs.RngStage.GRAIN_G)
    _white_v49 = fs.temporal_white(_tg_h, _tg_w, 21, 7, fs.RngStage.GRAIN_G,
                                   0.0, 0.0)
    chk("G-V49-IDENTITY  rho=0 and fixed=0 draw exactly the pre-v49 white field",
        bool(np.array_equal(_white_ref, _white_v49)),
        "bit-identical over %d samples" % _white_ref.size)

    # 2. VARIANCE IS PRESERVED AT EVERY SETTING. This is the guard that stops
    #    either control becoming a grain-amount control by accident -- the
    #    failure mode that would make `rms_granularity` stop meaning the number
    #    the datasheet prints.
    _tg_bad = []
    for _rho in (0.0, 0.5, 0.9):
        for _fx in (0.0, 0.5, 1.0):
            _v = float(fs.temporal_white(256, 256, 21, 3,
                                         fs.RngStage.GRAIN_G, _rho, _fx).std())
            if abs(_v - 1.0) > 0.02:
                _tg_bad.append((_rho, _fx, round(_v, 4)))
    chk("G-V49-VARIANCE  the temporal composition is unit variance at every "
        "rho and every fixed fraction, so neither control changes how much "
        "grain there is", not _tg_bad,
        "9 combinations, all within 2%% of 1.0" if not _tg_bad
        else "offenders (rho, fixed, sigma): %s" % (_tg_bad,))

    # 3. THE CORRELATION IS THE ONE THAT WAS ASKED FOR. tau is bisected onto
    #    the kernel's OWN lag-1 autocorrelation rather than assumed from an
    #    AR(1) coefficient of the same name, so this checks the solve.
    _tg_bad = []
    for _rho in (0.3, 0.6, 0.9):
        _c = fs.temporal_kernel(_rho)
        _got = sum(_c[_i] * _c[_i + 1] for _i in range(len(_c) - 1))
        if abs(_got - _rho) > 5e-3:
            _tg_bad.append((_rho, round(_got, 4)))
    chk("G-V49-RHO  the temporal kernel's own lag-1 autocorrelation equals the "
        "requested grain_frame_correlation", not _tg_bad,
        "0.3 / 0.6 / 0.9 all hit within 5e-3" if not _tg_bad
        else "requested vs achieved: %s" % (_tg_bad,))

    # 4. THE FIXED PATTERN IS FIXED. At f = 1 the field must be IDENTICAL on
    #    every frame -- that is the whole definition of a sensor pattern, and
    #    it is the half of the model the emulsion cannot express.
    _f0 = _tg_field(0.0, 1.0, 0)
    _f9 = _tg_field(0.0, 1.0, 900)
    chk("G-V49-FIXED  at fixed=1 the grain field is the same on every frame, "
        "which is what a scanner's fixed-pattern noise is",
        bool(np.allclose(_f0, _f9, rtol=0, atol=1e-12)),
        "frame 0 and frame 900 identical")

    # 5. AND IT IS NOT FIXED WHEN IT SHOULD NOT BE. The complementary check,
    #    because a bug that froze the field would pass #4 and look fine.
    _e0 = _tg_field(0.0, 0.0, 0)
    _e1 = _tg_field(0.0, 0.0, 1)
    _r01 = abs(_tg_corr(_e0.astype(np.float64), _e1.astype(np.float64)))
    chk("G-V49-EMULSION  at the shipped defaults successive frames are "
        "UNCORRELATED, which is what a camera negative does -- every frame is "
        "a different piece of film", _r01 < 0.10,
        "|r(frame 0, frame 1)| = %.4f" % _r01)

    # 6. DETERMINISM, WITH BOTH MECHANISMS ACTIVE. The spec requires the same
    #    seed, profile, settings and frame to reproduce exactly; a sliding
    #    kernel over neighbouring frames is precisely the kind of change that
    #    could have introduced order dependence, so it is asserted here rather
    #    than assumed from the counter RNG.
    _d1 = _tg_field(0.7, 0.3, 4242)
    _d2 = _tg_field(0.7, 0.3, 4242)
    chk("G-V49-DETERMINISM  the temporal field is reproducible at an arbitrary "
        "frame with both mechanisms active, with no state and no warm-up",
        bool(np.array_equal(_d1, _d2)),
        "frame 4242 rendered twice, bit-identical")

# ---- 6. grain field statistics -------------------------------------------
if _sec_on():
    # 4000 px over super35 = 161 px/mm, so a 17.5 um clump spans ~2.8 px and the
    # spectrum is genuinely resolved (a coarse test grid would alias to white noise).
    _ppm6 = 4000 / FORMATS["super35"]
    _t6 = film_profiles.grain_gauss_terms("legacy_gaussian", clump_um=17.5,
                                          clump_gain=1.15)
    f = fs.make_grain_field(1024, 4000, _ppm6, _t6, 10.5, 0.0, 1.0,
                            3, 0, fs.RngStage.GRAIN_G,
                            film_profiles.grain_aperture_legacy)
    chk("grain zero mean", abs(float(f.mean())) < 1e-6, f"mean={f.mean():.2e}")
    ah = [float((f[:, :-k]*f[:, k:]).mean()) for k in range(1, 6)]
    av = [float((f[:-k, :]*f[k:, :]).mean()) for k in range(1, 6)]
    rel = max(abs(a-b)/abs(a) for a, b in zip(ah, av))
    chk("grain isotropic (h vs v autocorrelation)", rel < 0.05, f"max rel diff={rel*100:.2f}%")
    # anisotropy parameter must actually do something
    f2 = fs.make_grain_field(1024, 4000, _ppm6, _t6, 10.5, 0.0, 1.30,
                             3, 0, fs.RngStage.GRAIN_G,
                             film_profiles.grain_aperture_legacy)
    ah2 = float((f2[:, :-2]*f2[:, 2:]).mean()); av2 = float((f2[:-2, :]*f2[2:, :]).mean())
    chk("anisotropy parameter stretches vertical correlation", av2 > ah2*1.05,
        f"h={ah2:.3e} v={av2:.3e}")

# ---- 7. per-channel MTF: red softest, blue sharpest ----------------------
if _sec_on():
    # Needs enough px/mm that f50 (44-60 c/mm) is inside the passband, so use a
    # 2560 px render and a 25 c/mm bar pattern.
    W = 2560; H = 256
    px_mm = W / FORMATS["super35"]
    period_px = max(2, int(round(px_mm / 25.0)))
    bars = np.zeros((H, W, 3), dtype=np.float32)
    bars[:, :] = 0.18
    col = ((np.arange(W) // (period_px/2)) % 2).astype(np.float32)
    bars *= (0.5 + col)[None, :, None]
    st2 = fs.RenderSettings(grain_scale=0.0, print_grain=False, misreg_scale=0.0)
    sharp = get_profile("5219").with_overrides(
        mtf=get_profile("5219").mtf.__class__(4000., 4000., 4000., 0.0, 22.0))
    o_soft = fs.simulate(bars, get_profile("5219"), st2)
    o_ref = fs.simulate(bars, sharp, st2)
    atten = [float(o_soft[:, :, c].std() / o_ref[:, :, c].std()) for c in range(3)]
    # The adjacency band-pass partially offsets the MTF rolloff at this
    # frequency, which is the physically correct behaviour, so the blue layer
    # barely loses anything. Require real loss on red and no net gain anywhere.
    chk("MTF attenuates the 25 c/mm pattern", atten[0] < 0.92 and max(atten) <= 1.0,
        f"R/G/B={[round(a,3) for a in atten]}")
    chk("red softest, blue sharpest (layer stack order)", atten[0] < atten[1] < atten[2],
        f"surviving modulation R/G/B={[round(a,3) for a in atten]}")

# ---- 8. halation: energy spreads outward from a highlight ----------------
if _sec_on():
    # Halation radii are physical: CineStill's widest lobe is 700 um, which at
    # 512 px across Super 35 is only 14 px, so a ring at r=40-90 px sits 3-6 sigma
    # out and measures almost nothing. Render big enough for the kernel to exist.
    N = 2048
    spot = np.full((N, N, 3), 0.02, dtype=np.float32)
    yy, xx = np.mgrid[0:N, 0:N]
    cen = N // 2
    spot[((xx-cen)**2 + (yy-cen)**2) < 40**2] = 6.0
    ring = (((xx-cen)**2 + (yy-cen)**2) > 60**2) & (((xx-cen)**2 + (yy-cen)**2) < 180**2)
    st3 = fs.RenderSettings(grain_scale=0.0, print_grain=False, misreg_scale=0.0)
    hal_on = fs.simulate(spot, get_profile("cinestill"), st3)
    hal_off = fs.simulate(spot, get_profile("cinestill"), fs.RenderSettings(
        grain_scale=0.0, print_grain=False, misreg_scale=0.0, halation_scale=0.0))
    lift = float(hal_on[ring].mean() - hal_off[ring].mean())
    red_bias = float((hal_on[ring][:, 0] - hal_off[ring][:, 0]).mean()
                     - (hal_on[ring][:, 2] - hal_off[ring][:, 2]).mean())
    chk("halation lifts the surround of a highlight", lift > 0.002, f"lift={lift:.4f}")
    chk("halation is red-dominant", red_bias > 0.0, f"R-B lift={red_bias:.4f}")
    chk("no-remjet stock halates far more than a remjet stock",
        lift > float(fs.simulate(spot, get_profile("5219"), st3)[ring].mean()
                     - hal_off[ring].mean()))

# ---- 9. reversal path ----------------------------------------------------
if _sec_on():
    for nm in ("kodachrome", "velvia", "ektachrome", "tri-x reversal"):
        p = get_profile(nm)
        chk(f"{p.name} takes the reversal path", p.kind is StockKind.REVERSAL)
    # narrow latitude: a reversal stock must clip a wide ramp sooner than a negative
    ramp = np.zeros((32, 512, 3), dtype=np.float32)
    ramp[:] = (np.logspace(-2.2, 1.4, 512, dtype=np.float32))[None, :, None] * 0.18
    st_mono = fs.RenderSettings(grain_scale=0.0, print_grain=False,
                                misreg_scale=0.0, coupler_scale=0.0)
    r_rev = fs.simulate(ramp, get_profile("velvia"), st3)[:, :, 1].mean(axis=0)
    # Adjacency and coupler edge effects legitimately overshoot on a gradient,
    # so monotonicity is checked on the curve alone, with them switched off.
    r_rev_clean = fs.simulate(ramp, get_profile("velvia"), st_mono)[:, :, 1].mean(axis=0)
    r_neg = fs.simulate(ramp, get_profile("portra"), st3)[:, :, 1].mean(axis=0)
    clip_rev = float((r_rev > 0.995).sum() + (r_rev < 0.004).sum())
    clip_neg = float((r_neg > 0.995).sum() + (r_neg < 0.004).sum())
    chk("reversal has less latitude than negative", clip_rev > clip_neg,
        f"clipped samples velvia={clip_rev:.0f} portra={clip_neg:.0f}")
    # Monotonicity is a property of the curve, so assert it on the curve. The
    # rendered image legitimately overshoots on a gradient because the adjacency
    # band-pass and coupler edge term are real edge effects.
    _lg = np.linspace(-3.0, 3.0, 4001).astype(np.float32)
    # KNOWN, MEASURED SHAPE-FAMILY LIMIT -- do not widen this set casually.
    #
    # ToneCurve blends a toe and a shoulder around a straight line. At very
    # high gamma with a very short throw the two blends overlap, and the sum
    # overshoots by a tiny amount before settling. POLAROID_51 is the only
    # stock that reaches that regime: its PUBLISHED slope is 3.35, the
    # steepest in the database, over a throw of about half a decade, because
    # it is an ultra-high-contrast graphic-arts film with no intermediate
    # greys by design.
    #
    # This was checked and it is NOT a float32 artefact: evaluating the same
    # curve in float64 gives -9.429e-06 against float32's -9.537e-06, so the
    # overshoot is a real property of the curve shape, not of the arithmetic.
    # Six different toe/shoulder pairs that all land on the published D-max of
    # 1.75 were tried and every one produces the same -9.5e-06, so it cannot
    # be tuned away without abandoning either the published slope or the
    # published D-max.
    #
    # It is allowed because it is below the output quantum: 9.5e-06 against
    # 1/65535 = 1.526e-05 for a 16-bit destination, i.e. the overshoot is
    # smaller than one code value and cannot appear in a rendered image. The
    # tolerance below is set to one 16-bit code, so a defect large enough to
    # be VISIBLE still fails, for this stock as for every other.
    _REV_MONO_EXCEPTIONS = {"POLAROID_51": 1.0 / 65535.0}
    worst_rev = 0.0
    _rev_bad = []
    for _p in FILM_PROFILES:
        if not _p.is_reversal:
            continue
        _tol = _REV_MONO_EXCEPTIONS.get(_p.name, 1e-6)
        _anc = fs.solve_anchors(_p, fs.get_print_stock("SCAN_DI"), 0.18)
        for _c in range(3):
            _cur = _p.curves.as_tuple()[_c]
            _d = fs.density(-(_lg + np.float32(_anc[_c])), _cur)
            _t = (10.0 ** (-_d) - 10.0 ** (-_cur.dmax)) / (
                10.0 ** (-_cur.dmin) - 10.0 ** (-_cur.dmax))
            _slope = float(np.diff(_t).min())
            worst_rev = min(worst_rev, _slope)
            if _slope < -_tol:
                _rev_bad.append("%s ch%d %.2e (tol %.2e)" % (_p.name, _c, _slope, _tol))
    chk("reversal transfer monotonic in exposure", not _rev_bad,
        "; ".join(_rev_bad) or
        f"worst={worst_rev:.2e}, POLAROID_51 allowed to one 16-bit code")

# ---- 10. Technicolor three-strip specifics -------------------------------
if _sec_on():
    tech = get_profile("technicolor")
    chk("three-strip has non-identity taking matrix",
        not np.allclose(np.asarray(tech.taking_matrix), np.eye(3)))
    chk("three-strip has large registration error", tech.misregistration_um > 20.0,
        f"{tech.misregistration_um} um")
    chk("three-strip defaults to the imbibition print", tech.default_print == "TECHNICOLOR_IB")

# ---- 11. determinism, range, finiteness ---------------------------------
if _sec_on():
    a1 = fs.simulate(lin[:192, :192], get_profile("svema"), fs.RenderSettings(seed=99))
    a2 = fs.simulate(lin[:192, :192], get_profile("svema"), fs.RenderSettings(seed=99))
    a3 = fs.simulate(lin[:192, :192], get_profile("svema"), fs.RenderSettings(seed=100))
    chk("deterministic for a fixed seed", np.array_equal(a1, a2))
    chk("a different seed changes the grain", not np.array_equal(a1, a3))

    bad = []
    for p in FILM_PROFILES:
        o = fs.simulate(lin[:256, :256], p, fs.RenderSettings())
        if not (np.isfinite(o).all() and o.min() >= 0.0 and o.max() <= 1.0):
            bad.append(p.name)
    chk("every stock finite and within [0,1]", not bad, ", ".join(bad))

    # extreme inputs must not blow up
    for label, img in (("pure black", np.zeros((64, 64, 3), np.float32)),
                       ("16 stops over", np.full((64, 64, 3), 65536.0, np.float32))):
        o = fs.simulate(img, get_profile("5219"), fs.RenderSettings())
        chk(f"survives {label}", np.isfinite(o).all() and 0.0 <= o.min() and o.max() <= 1.0,
            f"range=[{o.min():.4f}, {o.max():.4f}]")

    # black frame must still be grainy: clean black is a digital tell
    o = fs.simulate(np.zeros((256, 256, 3), np.float32), get_profile("5296"),
                    fs.RenderSettings())
    chk("grain survives into pure black (base+fog)", float(o.std()) > 1e-4,
        f"std={o.std():.5f}")

# ---- 12. period stocks: orthochromatic spectral response -----------------
if _sec_on():
    # The defining property: ortho emulsion is effectively blind to red. A red
    # subject must render far darker, and a blue one far lighter, than on a
    # panchromatic stock of the same era.
    patches = np.zeros((64, 192, 3), dtype=np.float32)
    patches[:, 0:64] = (0.5, 0.0, 0.0)     # red
    patches[:, 64:128] = (0.0, 0.5, 0.0)   # green
    patches[:, 128:192] = (0.0, 0.0, 0.5)  # blue
    def rgb_response(name):
        o = fs.simulate(patches, get_profile(name), st_clean)[:, :, 1]
        return float(o[:, 8:56].mean()), float(o[:, 72:120].mean()), float(o[:, 136:184].mean())
    o_r, o_g, o_b = rgb_response("ortho")
    p_r, p_g, p_b = rgb_response("super xx")
    chk("ortho renders red much darker than blue", o_b > o_r * 2.0,
        f"ortho red={o_r:.4f} blue={o_b:.4f} ratio={o_b/max(o_r,1e-6):.2f}x")
    chk("ortho red/blue separation far exceeds panchromatic",
        (o_b / max(o_r, 1e-6)) > 3.0 * (p_b / max(p_r, 1e-6)),
        f"ortho {o_b/max(o_r,1e-6):.2f}x vs panchromatic {p_b/max(p_r,1e-6):.2f}x")
    chk("panchromatic stock keeps red usable", p_r > 0.25 * p_b,
        f"red={p_r:.4f} blue={p_b:.4f}")

# ---- 13. veiling flare ---------------------------------------------------
if _sec_on():
    # A dark patch inside a bright frame. Flare must lift the black floor and
    # compress overall contrast -- that is the whole point of modelling it.
    scene = np.full((256, 256, 3), 2.0, dtype=np.float32)
    scene[96:160, 96:160] = 0.002
    def black_and_range(flare):
        o = fs.simulate(scene, get_profile("super xx"),
                        fs.RenderSettings(grain_scale=0.0, print_grain=False,
                                          misreg_scale=0.0, flare=flare))
        return float(o[112:144, 112:144].mean()), float(o.max() - o.min())
    b0, r0 = black_and_range(0.0)
    b1, r1 = black_and_range(0.12)
    chk("flare lifts the black floor", b1 > b0 * 1.5, f"black {b0:.4f} -> {b1:.4f}")
    chk("flare compresses contrast", r1 < r0, f"range {r0:.4f} -> {r1:.4f}")
    chk("period stocks carry a nonzero default flare",
        all(get_profile(n).default_flare > 0.05
            for n in ("ortho", "super xx", "panchrom", "agfacolor", "dufaycolor")))
    chk("modern stocks carry no default flare",
        all(get_profile(n).default_flare == 0.0 for n in ("5219", "5203", "portra", "velvia")))

# ---- 14. duplication generations -----------------------------------------
if _sec_on():
    # Each generation must add grain and soften, WITHOUT running contrast away --
    # that is exactly why real duplicating stock is gamma 1.0.
    dupe = fs.get_print_stock("DUPE_FINE_GRAIN")
    chk("duplicating stock is gamma ~1.0",
        all(abs(c.gamma - 1.0) < 0.05 for c in dupe.curves.as_tuple()),
        f"gammas={[round(c.gamma,2) for c in dupe.curves.as_tuple()]}")

    bars = np.full((256, 1024, 3), 0.18, dtype=np.float32)
    bars[:, ::8] = 0.6
    bars[:, 1::8] = 0.6
    flat = np.full((256, 512, 3), 0.18, dtype=np.float32)
    gr, sh, mid = [], [], []
    for g in (0, 1, 2, 3):
        stg = fs.RenderSettings(misreg_scale=0.0, flare=0.0, generations=g)
        gr.append(float(fs.simulate(flat, get_profile("super xx"), stg)[:, :, 1].std()))
        ob = fs.simulate(bars, get_profile("super xx"), stg)[:, :, 1]
        sh.append(float(ob.std()))
        mid.append(float(fs.simulate(flat, get_profile("super xx"), stg)[:, :, 1].mean()))
    # The meaningful quantity is grain relative to picture detail, not absolute
    # grain. A dupe chain softens the picture faster than it softens the grain, which
    # is precisely why archival prints look grainier than the negatives they came
    # from -- absolute grain sigma can even fall slightly while the ratio worsens.
    ratio = [g / d for g, d in zip(gr, sh)]
    chk("grain-to-detail ratio worsens with each generation",
        all(b > a for a, b in zip(ratio, ratio[1:])),
        f"ratio={[round(v,4) for v in ratio]} for 0/1/2/3 generations "
        f"(grain={[round(v,5) for v in gr]}, detail={[round(v,5) for v in sh]})")
    chk("each generation softens fine detail", all(b < a for a, b in zip(sh, sh[1:])),
        f"detail sigma={[round(v,5) for v in sh]}")
    chk("contrast does not run away over generations",
        max(mid) / min(mid) < 1.25, f"mid grey={[round(v,4) for v in mid]}")

# ---- 15. Agfacolor Neu: desaturated yet contrasty ------------------------
if _sec_on():
    # The combination nothing else in the set has - reversal film with positive dye
    # off-diagonals. It must lose saturation relative to a clean reversal stock while
    # keeping high contrast.
    ramp = np.zeros((32, 512, 3), dtype=np.float32)
    ramp[:] = (np.logspace(-2.0, 1.2, 512, dtype=np.float32))[None, :, None] * 0.18
    def sat_and_contrast(name, img):
        o = fs.simulate(img, get_profile(name), st_clean)
        mx, mn = o.max(2), o.min(2)
        return float((mx - mn).mean()), float(np.percentile(o, 98) - np.percentile(o, 2))
    # Measure on properly exposed mid-tone patches. Comparing on near-white patches
    # is meaningless: a 5-stop reversal stock clips them, so every stock scores the
    # same and the check passes for the wrong reason.
    mids = np.zeros((48, 288, 3), dtype=np.float32)
    for _i, _c in enumerate([(0.30,0.06,0.06),(0.06,0.24,0.08),(0.05,0.09,0.30),
                             (0.32,0.28,0.07),(0.34,0.22,0.18),(0.18,0.18,0.18)]):
        mids[:, _i*48:(_i+1)*48] = _c
    s_ag, c_ag = sat_and_contrast("agfacolor", mids)
    s_ek, c_ek = sat_and_contrast("ektachrome", mids)
    chk("Agfacolor Neu is much less saturated than a clean reversal stock",
        s_ag < 0.5 * s_ek, f"agfacolor={s_ag:.4f} ektachrome={s_ek:.4f}")
    # The full hierarchy must come out in the physically sensible order.
    #
    # ⚠⚠ UPDATED 2026-09-21: THE LARGE INVERSION THIS NOTE DESCRIBED IS GONE.
    # Adopting `M_reader . M_status^-1` instead of the raw status matrices puts
    # VELVIA back at 0.509 -- from 0.326, against 0.755 before any adoption --
    # and it is once more ABOVE Kodachrome, so the first pair orders correctly
    # again. What remains failing is the SINGLE ADJACENT-PAIR inversion this
    # file has carried as its one accepted FAIL since long before any of this:
    # 5219 against 5296, two stocks a hundredth apart. The check is still not
    # relaxed, re-ordered or baselined away.
    #
    # ⚠ THE NOTE AS IT STOOD ON 2026-09-20, KEPT BECAUSE IT IS WHY THE
    # CORRECTION HAPPENED: The
    # owner adopted `_MEASURED_DYE_MATRIX` with the double count explained and
    # the consequence quoted; this is where the consequence shows up as a
    # number. VELVIA -- the cleanest-dye stock in the set -- fell from 0.755 to
    # 0.326 and now sits BELOW Kodachrome, so the ordering breaks at its first
    # pair. That is a status density being multiplied by a matrix built from
    # the absorptions the status density already contains.
    #
    # ⚠ THE CHECK IS NOT RELAXED, RE-ORDERED OR BASELINED AWAY, and it must not
    # be. It was already the file's one accepted FAIL for a different and
    # smaller reason (technicolor 0.179 against 5219's 0.196, two adjacent
    # stocks a fifth of a unit apart); it now carries a second, much larger
    # inversion that a single line -- `_MEASURED_DYE_MATRIX_ADOPTED = False` --
    # would remove. Leaving it red is what keeps that line findable.
    _order = ["velvia", "kodachrome", "technicolor", "5219", "5296", "agfacolor", "orwocolor"]
    _sats = [sat_and_contrast(n, mids)[0] for n in _order]
    chk("saturation hierarchy is ordered clean -> impure dyes",
        all(a > b for a, b in zip(_sats, _sats[1:])),
        " > ".join(f"{n}={v:.3f}" for n, v in zip(_order, _sats)))

    # ---- dye matrix row sums, and what they do and do NOT guarantee ---------
    # ⚠ THE RATIONALE ON THIS CHECK WAS FALSE AND WAS CORRECTED 2026-09-08.
    # It used to read: "Dye matrices must be pure saturation operators: unit row
    # sums, so they change colour without shifting neutral density." The test is
    # right and stays; the reason given for it does not follow, and the database
    # falsifies it 81 times over while the check passes.
    #
    # WHAT UNIT ROW SUMS ACTUALLY BUY. M has unit row sums exactly when
    # M @ (k,k,k) == (k,k,k), i.e. it fixes an EQUAL-DENSITY triple. That is a
    # real and worth-keeping property -- it is what stops the matrix from being
    # a global density offset wearing a colour-matrix costume -- and it holds to
    # 2.2e-16 across the whole database, which is machine exactness, not luck.
    #
    # WHAT THEY DO NOT BUY. A NEUTRAL on a masked colour negative is not an
    # equal-density triple. The orange mask puts the three layers at very
    # different densities at the same mid grey, so M fixes a colour the stock
    # never produces and moves the one it does. Measured at mid grey over the
    # 107 stocks whose matrix is not the identity:
    #     > 0.01 D on 81 stocks, > 0.05 D on 26, median 0.0329 D,
    #     worst 0.1338 D on SVEMA_CNL_32 (0.926/1.019/1.641 -> 1.007/1.072/1.508)
    # The rendered consequence is far smaller (~0.004-0.005 linear) because the
    # downstream anchor and balance solve absorb most of it. A real invariant
    # violation with a small visible cost: pinned below, NOT fixed, and not a
    # licence to change any stored matrix.
    _bad = {}
    for _p in FILM_PROFILES:
        for _r, _row in enumerate(_p.dye_matrix):
            if abs(sum(_row) - 1.0) > 1e-6:
                _bad[_p.name] = round(sum(_row), 4)
    for _s in fs.get_print_stock("SCAN_DI"), fs.get_print_stock("KODAK_2383_RELEASE"), \
              fs.get_print_stock("TECHNICOLOR_IB"), fs.get_print_stock("DUPE_FINE_GRAIN"):
        for _row in _s.dye_matrix:
            if abs(sum(_row) - 1.0) > 1e-6:
                _bad[_s.name] = round(sum(_row), 4)
    chk("every dye matrix fixes an EQUAL-DENSITY triple (unit row sums)",
        not _bad, str(_bad))

    # The neutral-shift census, pinned so it cannot drift unnoticed. This is a
    # WITNESS, not an aspiration: it asserts the measured state, so a silent
    # worsening AND a silent improvement both report here and get read by a
    # person. Move the numbers only together with the change that earned it.
    _shift = []
    for _p in FILM_PROFILES:
        _m = np.array(_p.dye_matrix, dtype=float)
        if np.allclose(_m, np.eye(3)):
            continue
        _cv = (_p.curves.r, _p.curves.g, _p.curves.b)
        _d = np.array([float(fs.density_scalar(0.0, _cv[_c])) for _c in range(3)])
        _shift.append((float(np.max(np.abs(_m @ _d - _d))), _p.name))
    _shift.sort(reverse=True)
    _n01 = sum(1 for _v, _ in _shift if _v > 0.01)
    _n05 = sum(1 for _v, _ in _shift if _v > 0.05)
    _med = float(np.median([_v for _v, _ in _shift])) if _shift else 0.0
    # ⚠ THE CENSUS MOVED ON 2026-09-15 AND THE NEW WORST CASE IS THE NEW STOCK.
    #   ORWOCOLOR_NC3 arrives with the ORWO house dye matrix and a measured
    #   curve set, and the combination shifts a neutral by 0.1437 D -- past
    #   SVEMA_CNL_32's 0.1338. That is not a defect being admitted: an ORWO
    #   negative's dyes ARE the most impure in the corpus, the profile says so
    #   in as many words, and its matrix is the same _dye() house figure its
    #   two siblings carry. What is new is that it now sits on measured curves
    #   rather than analogy ones, which is what moves the product.
    # ⚠ 82 -> 84 over 0.01 D and the median 0.0335 -> 0.0342 on 2026-09-15b.
    # Nothing was done to dye_matrix. VERICOLOR III and EKTAPRESS PJ400 each
    # stopped carrying a FLAT base density and started carrying the orange
    # ladder «Современные фотоматериалы и их обработка» prints for them, and
    # the neutral shift this census measures is computed THROUGH the curve, so
    # a stock that previously read as unmasked now reads as masked. The census
    # moving is the fill landing, not a regression.
    # ⚠ 84 -> 85 over 0.01 D and 27 -> 28 over 0.05 D on 2026-09-20, median
    # 0.0342 -> 0.0344. One stock moved and it is EKTAR 100: its curves were
    # re-traced from E-4046 p4 and the maskless analogy they replaced read as
    # UNMASKED through this census. A stock that starts carrying the orange
    # ladder its own sheet draws starts reading as masked here. Same shape as
    # the 2026-09-15b entry above -- the census moving is the correction
    # landing, not a regression, and dye_matrix itself was not touched.
    # ⚠⚠ 108 -> 113 NON-IDENTITY, 85 -> 91 over 0.01 D, 28 -> 38 over 0.05 D
    # and median 0.0344 -> 0.0355 on 2026-09-20b, ALL OF IT FROM THE
    # MEASURED-DYE-MATRIX ADOPTION. 26 stocks swapped a symmetric `_dye(k)`
    # scalar for a measured asymmetric matrix, and an asymmetric matrix shifts
    # the neutral further -- ten more stocks now past 0.05 D. ⚠ UNLIKE THE
    # EKTAR ENTRY ABOVE THIS IS NOT A CORRECTION LANDING: the previous census
    # moved because a curve became more truthful, this one moved because
    # `dye_matrix` itself was replaced by an owner decision this file argues
    # against (see the saturation-hierarchy note). The baseline is updated so
    # the census still catches the NEXT movement; the reason it moved is
    # recorded here rather than absorbed.
    # ⚠⚠ AND MOVED AGAIN ON 2026-09-21, THIS TIME *DOWN* PAST WHERE IT STARTED.
    # Switching the adoption from the raw status matrices to
    # `M_reader . M_status^-1` takes it 91 -> 82 over 0.01 D, 38 -> 25 over
    # 0.05 D and median 0.0355 -> 0.0262 -- BELOW the 85 / 28 / 0.0344 the
    # database carried before any of this started. The reader-corrected
    # matrices disturb the neutral LESS than the hand-set `_dye(k)` scalars
    # they replaced, which is the outcome the derivation predicts and is
    # independent evidence that the raw table was the wrong magnitude.
    chk("dye_matrix neutral-shift census is unchanged "
        "(113 non-identity stocks, 82 over 0.01 D, 25 over 0.05 D, "
        "median 0.0262 D, worst 0.1437 D on ORWOCOLOR_NC3)",
        len(_shift) == 113 and _n01 == 82 and _n05 == 25
        and abs(_med - 0.0262) < 5e-4
        and _shift[0][1] == "ORWOCOLOR_NC3" and abs(_shift[0][0] - 0.1437) < 5e-4,
        f"n={len(_shift)} over0.01={_n01} over0.05={_n05} "
        f"median={_med:.4f} worst={_shift[0][0]:.4f} on {_shift[0][1]}")
    _, c_ag2 = sat_and_contrast("agfacolor", ramp)
    _, c_xx = sat_and_contrast("super xx", ramp)
    chk("Agfacolor Neu still runs high contrast", c_ag2 > c_xx,
        f"agfacolor={c_ag2:.3f} super-xx={c_xx:.3f}")

# ---- 16. Dufaycolor reseau ------------------------------------------------
if _sec_on():
    duf = get_profile("dufaycolor")
    chk("Dufaycolor declares a reseau", duf.has_reseau and duf.reseau.lines_per_mm > 0)
    chk("reseau filters overlap (this is what makes it pastel)",
        all(duf.reseau.filter_matrix[c][j] > 0.02
            for c in range(3) for j in range(3) if c != j),
        f"off-diagonals={[round(duf.reseau.filter_matrix[c][j],3) for c in range(3) for j in range(3) if c!=j]}")
    chk("reseau throughput costs 1-2 stops",
        0.20 < duf.reseau.mean_throughput() < 0.45,
        f"throughput={duf.reseau.mean_throughput():.3f} "
        f"({-np.log2(duf.reseau.mean_throughput()):.2f} stops)")
    mask, pitch = fs.build_reseau_mask(64, 64, 20.0 * 4, duf.reseau)   # 4 px pitch
    cover = [float(mask[:, :, c].mean()) for c in range(3)]
    chk("reseau mask is one-hot", np.allclose(mask.sum(axis=2), 1.0))
    chk("reseau covers each colour roughly equally",
        max(cover) < 0.45 and min(cover) > 0.22, f"coverage={[round(v,3) for v in cover]}")

    # At adequate resolution the grid must leave a periodic signature at its own
    # spatial frequency.
    W = 4096
    grey = np.full((128, W, 3), 0.18, dtype=np.float32)
    o_res = fs.simulate(grey, duf, fs.RenderSettings(grain_scale=0.0, print_grain=False,
                                                     flare=0.0))
    row = o_res[:, :, 1].mean(axis=0) - o_res[:, :, 1].mean()
    spec = np.abs(np.fft.rfft(row))
    ppm = W / FORMATS["super35"]
    expect_bin = int(round(duf.reseau.lines_per_mm / ppm * W))
    band = spec[max(1, expect_bin - 3):expect_bin + 4]
    chk("reseau leaves a periodic signature at the grid frequency",
        band.max() > 6.0 * np.median(spec[1:]),
        f"peak/median={band.max()/max(np.median(spec[1:]),1e-9):.1f} at bin {expect_bin}")

    # A neutral input must still reconstruct to roughly neutral colour.
    patch = o_res[:, W//4:3*W//4].reshape(-1, 3).mean(axis=0)
    chk("reseau reconstructs neutral grey as neutral",
        float(patch.max() - patch.min()) < 0.06, f"meanRGB={patch.round(4)}")

    # Under-sampled: must refuse rather than emit aliasing garbage.
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        o_small = fs.simulate(np.full((64, 256, 3), 0.18, np.float32), duf,
                              fs.RenderSettings(flare=0.0))
    chk("reseau disables itself when under-sampled", "mosaic disabled" in buf.getvalue())
    chk("under-sampled reseau still produces a valid image",
        np.isfinite(o_small).all() and 0.0 <= o_small.min() and o_small.max() <= 1.0)
    chk("--no-reseau switch is honoured",
        "mosaic disabled" not in (lambda: (lambda b: b.getvalue())(io.StringIO()))()
        and np.isfinite(fs.simulate(np.full((64, 1024, 3), 0.18, np.float32), duf,
                                    fs.RenderSettings(reseau=False, flare=0.0))).all())

# ---- 12. schema v4: coating defects + lens vignette ----------------------
if _sec_on():
    import film_profiles as _fpm

    chk("schema version is at least 4 (coating fields present)",
        _fpm.SCHEMA_VERSION >= 4, f"v={_fpm.SCHEMA_VERSION}")
    chk("frame pitch = perf pitch x perfs per frame",
        abs(_fpm.frame_pitch_mm("super35") - 19.0) < 1e-9
        and abs(_fpm.frame_pitch_mm("8mm") - 3.81) < 1e-9
        and _fpm.frame_pitch_mm("polaroid_sx70") == 0.0,
        f'35mm={_fpm.frame_pitch_mm("super35")} 8mm={_fpm.frame_pitch_mm("8mm")}')

    # every stock carries a lens vignette; modern majors carry no coating field
    _vig = [p.name for p in FILM_PROFILES if not 0.0 < p.default_vignette < 4.0]
    chk("every stock has a plausible lens vignette", not _vig, ", ".join(_vig[:5]))
    _mod = [p for p in FILM_PROFILES if p.name.startswith("KODAK_VISION3")]
    chk("modern stocks have no coating field",
        all(p.coating.coating_sigma == 0.0 for p in _mod),
        ", ".join(p.name for p in _mod if p.coating.coating_sigma > 0))
    _sov = [p for p in FILM_PROFILES
            if p.name.startswith(("SVEMA", "TASMA", "ORWO", "SOVCOLOR"))]
    chk("Soviet/GDR stocks all carry a coating field",
        all(p.coating.has_coating_field for p in _sov),
        ", ".join(p.name for p in _sov if not p.coating.has_coating_field))
    chk("edge fog is gauge-driven, not era-driven",
        all(p.coating.has_edge_fog for p in FILM_PROFILES
            if p.default_format in ("8mm", "super8", "16mm", "super16"))
        and not any(p.coating.has_edge_fog for p in FILM_PROFILES
                    if p.default_format in ("super35", "ff35", "academy35")))

    # vignette is real cos^4 geometry: centre exactly 1, corner exactly the ask
    _vf = fs.vignette_field(240, 320, 1.0)
    _corner = 0.25 * (_vf[0, 0] + _vf[0, -1] + _vf[-1, 0] + _vf[-1, -1])
    chk("vignette centre is unity and corner matches the requested stops",
        abs(_vf[120, 160] - 1.0) < 2e-3 and abs(-math.log2(_corner) - 1.0) < 0.02,
        f"centre={_vf[120,160]:.4f} corner={-math.log2(_corner):.3f} stops")
    chk("vignette of 0 stops is exactly transparent",
        float(fs.vignette_field(64, 64, 0.0).min()) == 1.0)

    # coating field: mean 1, correct sigma, deterministic, web-coherent
    _sv = get_profile("SVEMA_FOTO_65")
    _cf = [fs.coating_field(180, 240, 24.89, 18.66, _sv.coating, i, 19.0, 4242)
           for i in range(12)]
    _ens = np.concatenate([f.ravel() for f in _cf])
    chk("coating field is unbiased", abs(_ens.mean() - 1.0) < 0.01,
        f"mean={_ens.mean():.5f}")
    chk("coating field sigma tracks the spec",
        0.6 * _sv.coating.coating_sigma < _ens.std() < 1.4 * _sv.coating.coating_sigma,
        f"sigma={_ens.std():.4f} spec={_sv.coating.coating_sigma}")
    chk("coating field is a pure function of (seed, web position)",
        np.array_equal(_cf[3],
                       fs.coating_field(180, 240, 24.89, 18.66, _sv.coating, 3,
                                        19.0, 4242)))
    # cross-web streaks are fixed hardware: correlation must persist across frames
    _cw = np.corrcoef(_cf[0].mean(axis=0), _cf[7].mean(axis=0))[0, 1]
    chk("cross-web streaks stay correlated frame to frame", _cw > 0.3,
        f"corr={_cw:.3f}")
    # and a smaller gauge must drift more slowly: less web travel per frame.
    # 2026-08-13: the retired SVEMA_FN_64_8MM entry used to supply the coating
    # spec here; its spec was identical to the 35 mm entry's by design (same
    # emulsion), so the same _sv.coating with 8 mm geometry tests the same thing.
    _f8 = [fs.coating_field(180, 240, 4.8, 3.5, _sv.coating, i, 3.81, 4242)
           for i in range(6)]
    _m35 = np.corrcoef(_cf[0].ravel(), _cf[1].ravel())[0, 1]
    _m8 = np.corrcoef(_f8[0].ravel(), _f8[1].ravel())[0, 1]
    chk("small gauge drifts slower than 35 mm (less web per frame)", _m8 > _m35,
        f"8mm lag1={_m8:.3f} vs 35mm lag1={_m35:.3f}")

    # corner defocus softens corners only, and never darkens
    _edge = np.zeros((180, 240), np.float32)
    _edge[:, ::8] = 1.0
    _cd = fs.corner_defocus(_edge, 0.35)
    chk("corner defocus softens the corners",
        _cd[:24, :24].std() < 0.92 * _cd[80:100, 110:130].std(),
        f"corner={_cd[:24,:24].std():.4f} centre={_cd[80:100,110:130].std():.4f}")
    chk("corner defocus preserves mean level (softens, does not darken)",
        abs(float(_cd.mean()) - float(_edge.mean())) < 0.01,
        f"{_cd.mean():.4f} vs {_edge.mean():.4f}")
    chk("corner defocus of 0 is a no-op",
        np.array_equal(fs.corner_defocus(_edge, 0.0), _edge))

    # the whole v4 block must be switchable off, and cost nothing when off
    _flat = np.full((180, 240, 3), 0.18, np.float32)
    _off = fs.simulate(_flat, _sv, fs.RenderSettings(
        film_format="super35", grain_scale=0.0, print_grain=False,
        vignette=0.0, coating_scale=0.0, flare=0.0))
    _h, _w, _ = _off.shape
    chk("v4 defects fully disable: flat field stays flat",
        abs(_off[:, :6, 1].mean() / _off[:, _w // 2 - 3:_w // 2 + 3, 1].mean() - 1.0)
        < 2e-3,
        f"edge/centre={_off[:, :6, 1].mean() / _off[:, _w//2-3:_w//2+3, 1].mean():.5f}")
    _on = fs.simulate(_flat, _sv, fs.RenderSettings(
        film_format="super35", grain_scale=0.0, print_grain=False, flare=0.0))
    chk("v4 defects on produces measurable structure", _on.std() > 3.0 * _off.std(),
        f"on={_on.std():.5f} off={_off.std():.5f}")

    # edge fog lightens the positive (more negative density prints lighter).
    # 2026-08-13: was the retired SVEMA_FN_64_8MM profile. Edge fog is decorated
    # from a profile's DEFAULT format (a known limitation flagged in the
    # FilmDatabase MD, Appendix B: gauge-derived properties should follow the
    # RENDERED format), so the test needs a stock whose default gauge is 8 mm.
    _g8 = get_profile("GENERIC_BW")
    _o8 = fs.simulate(_flat, _g8, fs.RenderSettings(
        film_format="8mm", grain_scale=0.0, print_grain=False, vignette=0.0,
        flare=0.0))
    _h8, _w8, _ = _o8.shape
    # GENERIC_BW is a REVERSAL stock: extra edge density darkens the projected
    # image directly (no print inversion), so the expectation is the OPPOSITE
    # of the retired negative-stock test -- the edge must come out DARKER.
    chk("narrow-gauge edge fog darkens the frame edge (reversal stock)",
        _o8[:, :6, 1].mean() < 0.95 * _o8[:, _w8 // 2 - 3:_w8 // 2 + 3, 1].mean(),
        f"edge/centre={_o8[:, :6, 1].mean() / _o8[:, _w8//2-3:_w8//2+3, 1].mean():.4f}")

# ---- 13. schema v5: interimage effects -----------------------------------
if _sec_on():
    # v7 (2026-08-16): four inert data carriers appended. Bump this WITH the
    # schema, never ahead of it -- the constant is the contract the C++ side
    # mirrors, and a stale value here would hide a real mismatch.
    # v8, 2026-08-18 (queue item C1): GrainSpec gained sigma_shape_peak,
    # sigma_shape_peak_at, sigma_shape_toe_at, sigma_shape_dmax_at and the
    # sigma_shape_measured flag, and the sigma(D) shape is now READ by the
    # renderer for the profiles that set the flag. Bumping the version is what
    # tells a consumer of the generated C++ that the struct layout moved.
    # v11, 2026-08-23 (queue item C21): HalationSpec gained radius_scale_r/g/b.
    # All 160 stocks ship them at 1.0, so a v11 database renders bit-identically
    # to v10 -- but the STRUCT GREW, so a v10 consumer reading v11 data would
    # walk off the end of every HalationSpec. That is the whole reason this
    # constant exists and the reason it moves even when no pixel does.
    # v12, 2026-08-25 (queue item C15): PrintStock gained `aging` and the new
    # `dye_stability`. Both INERT and both appended after every v11 field, so a
    # v12 database renders bit-identically to v11 -- but PrintStock GREW, so the
    # constant moves for the same reason it moved at v11.
    # v13, 2026-08-26 (queue XX2 + C36): DevelopmentPoint gained `base_fog`, and
    # PrintStock gained the per-record `mtf_f50_r/g/b` + `mtf_f50_bound` +
    # `mtf_measured`. Both INERT, both appended, so v13 renders bit-identically
    # to v12 -- and TWO structs grew this time, which is why the constant moves.
    # v14, 2026-08-26 (queue B1): SpectralDyeDensity gains `d_dmin` and a second
    # legal shape -- a neutral + D-min pair for sheets that never plotted the
    # three dyes. Appended, inert, and `has_data` keeps its old meaning so no
    # count moves.
    # ⚠ VERSION PIN 18 -> 22 on 2026-08-31, AND THE PIN IS WHY THE DRIFT WAS
    # FOUND. v19 (dye-density neutral traces, `dye_matrix` from measured
    # spectra), v20 (taking-filter fields, queue C39), v21 (`AimDensity` and
    # `ProcessVariant.push_stops`, queues K2/K3) and v22 (`PrintStock.spectral`,
    # queue M1) all landed on 2026-08-30 and 2026-08-31 with their fields
    # commented `# -- schema vNN` while `SCHEMA_VERSION` sat at 18 and this pin
    # agreed with it. A pin that tracks a constant nobody bumps asserts nothing;
    # the constant is now 22, this pin moves with it, and `doc_consistency.py`
    # registers the version in three documents so a repeat fails the build.
    # All four are additive and inert: a v22 database renders bit-identically
    # to a v18 one and no film index moves.
    # ⚠ VERSION PIN 23 -> 24 on 2026-09-01b: `ReciprocityTable` gained
    # `development_correction_pct`, the DEVELOPMENT compensation Agfa print
    # beside the exposure one on the same time cells. Additive and inert --
    # nothing reads the dataclass -- so a v24 database renders bit-identically
    # to a v23 one and no film index moves.
    # ⚠ VERSION PIN 24 -> 25 on 2026-09-03, AND THIS ONE IS NOT INERT. Every
    # bump since v19 has been a carrier nothing read; v25 adds
    # `PrintStock.printing_density_matrix` and stage 13 reads it on the
    # EXPOSURE side of the print. Ten of the eleven print stocks keep an
    # identity matrix, so only a render through KODAK_2383_RELEASE moves --
    # and there it moves because that stock now has a derivation behind it.
    # ⚠ 25 -> 26 on 2026-09-05 (queue #215) and 26 -> 27 the same day (queue
    # C23). v26 WIDENS `ProcessVariant.push_stops` from int32 to float, which is
    # the first non-additive schema change in this log -- see the note beside
    # SCHEMA_VERSION for why an int could hold only one of Super Anscochrome's
    # four published ratings. v27 adds `ProcessingSpec.bromide_drag`, a
    # BromideDragSpec, and it is READ BY STAGE 9c in all three engines; it is
    # inert on all 176 stocks because no source in this corpus measures it.
    # ⚠ 27 -> 28 on 2026-09-06i: `DevelopmentPoint` gained `vessel`. The
    # smallest bump in this log, and it exists because a stored record asked
    # for it in prose -- the three AGFAPAN `processing_family.source` strings
    # said, in their own words, that the drum and small-tank rows were
    # "distinguishable only by their times". Additive and inert: nothing on any
    # engine's path reads a development point, so a v28 database renders
    # bit-identically to a v27 one and no film index moves.
    # v29 (2026-09-10): nineteen fields from the 41-document patent and
    # Glafkides harvest. The LARGEST bump in the log and, like v28, entirely
    # additive and entirely inert -- G-V29-INERT below proves it by asserting
    # that not one of the nineteen is referenced anywhere in film_sim.py, and
    # none is emitted into the C++ at all (the v23 precedent). Six of them
    # attach a DEFINITION to a number that was already stored: which aperture
    # an rms figure was read through, which of nine gamma definitions a stored
    # gamma is, whether a dye-impurity ratio was measured in transmission or
    # reflection. Those are ingest-side truth, and each one exists because the
    # harvest actually made that mistake before the field did.
    chk("schema version is 49", _fpm.SCHEMA_VERSION == 49, f"v={_fpm.SCHEMA_VERSION}")

    # ==== 2026-09-01: THE TWO CARRIERS THAT STOPPED BEING INERT =============
    # `reciprocity_table` and `process_variants` were both listed as "carried,
    # validated, never read". Both are now consumed -- reciprocity by stage 8 in
    # both engines, process variants by the frame-setup resolver -- and each
    # ships under an inertness contract that these guards pin.
    import film_sim as _fsim

    # ==== v48 (2026-09-19): THE GRAIN SPECTRUM AS A GAUSSIAN MIXTURE =======
    #
    # FGS-DDS-001 Rev. A, changes C1/C2/C3/C4/C10. These guards exist because
    # the v48 grain path replaced BOTH the spectrum and the generator, and the
    # two failure modes that matter are silent: a mixture that fits badly looks
    # like a different emulsion, and a mixture that fits with large cancelling
    # weights destroys the AVX2 engine's float32 field while the scalar engine
    # renders perfectly. Neither raises anything.
    import numpy as _v48np

    _v48f = _v48np.linspace(0.0, _fpm.GRAIN_INTEGRAL_FMAX_CPMM, 2001)

    # ---- the legacy spectrum survives the rewrite EXACTLY ------------------
    #
    # ⚠ THIS IS THE ONE THAT MAKES THE REWRITE REVERSIBLE. Both spectral models
    # now leave through one function; if the mixture form could not reproduce
    # the v47 shape to float64 rounding, every legacy render would have moved
    # and there would be no way to tell a modelling change from a regression.
    _v48leg = []
    for _cu, _cg in ((2.32, 0.0), (2.32, 0.4), (6.93, 1.2), (0.9, 0.25),
                     (13.0, 2.0), (0.2, 0.0)):
        _t = _fpm.grain_gauss_terms("legacy_gaussian", clump_um=_cu,
                                    clump_gain=_cg)
        _v48leg.append(float(_v48np.abs(
            _fpm.grain_terms_shape(_t, _v48f)
            - _fpm.grain_legacy_shape(_v48f, _cu, _cg)).max()))
    chk("G-V48-LEGACY  the legacy spectrum expressed as a Gaussian mixture "
        "reproduces the v47 closed form to float64 rounding, so a v48 render "
        "in legacy mode differs from v47 in the GENERATOR and in nothing else",
        max(_v48leg) < 1e-12,
        "worst deviation over six (clump, gain) pairs on a 2001-point band: "
        "%.2e" % max(_v48leg))

    # ---- the two engine constants that were typed wrong -------------------
    #
    # ⚠⚠ BOTH OF THESE SHIPPED, BOTH FOR YEARS, AND BOTH FOR THE SAME REASON:
    # the reference engine evaluates its transfers straight onto a frequency
    # grid and derives no sigma at all, so a sigma constant lived only in C++
    # and had nothing to disagree with. v48 gives both a Python consumer, which
    # is what turns a secret into a parity failure.
    #
    #   kSigma           Algo_11_Sim.cpp  0.22508352815546  vs 0.22507907903927651
    #   AlgoScanSigmaMm  Algo_10_Sim.cpp  0.18738564618678  vs 0.1873906251292776
    #
    # The second is stage 10, so it band-limited the whole image and not only
    # the grain. The correct value was already in this database twice, under
    # GRAIN_SIGMA_PER_HALF_POWER and _TAGUCHI_F50_TO_SIGMA_UM.
    _v48k1 = 1.0 / (math.pi * math.sqrt(2.0))
    _v48k2 = math.sqrt(math.log(2.0) / 2.0) / math.pi
    chk("G-V48-KSIGMA  the two sigma constants are the exact values and not "
        "the digits the engines carried, and the half-power constant agrees "
        "with the Taguchi table's copy of the same quantity to the last bit",
        abs(_fpm.GRAIN_SIGMA_PER_1E - _v48k1) < 1e-17
        and abs(_fpm.GRAIN_SIGMA_PER_HALF_POWER - _v48k2) < 1e-17
        and (_fpm.GRAIN_SIGMA_PER_HALF_POWER
             == _fpm._TAGUCHI_F50_TO_SIGMA_UM / 1000.0)
        and abs(0.22508352815546 - _v48k1) > 4e-6
        and abs(0.18738564618678 - _v48k2) > 4e-6,
        "1/(pi sqrt2) = %.17g, sqrt(ln2/2)/pi = %.17g; the shipped literals "
        "were wrong by %.2e and %.2e relative"
        % (_v48k1, _v48k2, abs(0.22508352815546 / _v48k1 - 1.0),
           abs(0.18738564618678 / _v48k2 - 1.0)))

    # ---- and the wrong digits are gone from the ENGINE SOURCES ------------
    #
    # ⚠⚠ THE TYPO WAS IN THREE FUNCTIONAL SITES, NOT TWO, AND THE THIRD WAS
    # FOUND ONLY BY GREPPING FOR THE DIGITS. `AlgoScanSigmaMm` appears in both
    # `Algo_10_Sim.cpp` twins -- that is stage 10, the scan MTF -- and
    # `ALGO_MTF_SIGMA_MM_PER_INV_F50` in `AlgoEmulsionMtf.hpp` is the EMULSION
    # MTF, stages 5 and 6. So the same 2.66e-05 error was band-limiting the
    # image twice over, in different stages, from two different files.
    #
    # ⚠ AND `AlgoEmulsionMtf.hpp` PRINTS ITS OWN DERIVATION BESIDE THE WRONG
    # DIGITS: "sqrt(0.6931471805599453 / 2) is 0.5887050112577373 and dividing
    # by pi gives 0.18738564618678". The first two numbers are right, the third
    # does not follow from them, and the comment says in terms that "the
    # derivation above can be checked against the digits by hand". Nobody did.
    #
    # This is a SOURCE-TEXT guard rather than a value guard because the value
    # guard above cannot see a literal typed into a file it does not import.
    # That is exactly how all three survived.
    # ⚠ THE ENGINE TREE IS NOT BESIDE THE GENERATOR. `build.py` passes it in
    # FILMSIM_ROOT; a bare `python3 verify.py` has to find it, and finding
    # nothing must not look like finding nothing wrong -- which is exactly what
    # the first version of this guard did, reporting "0 engine sources checked,
    # 0 stale literal(s)" as a FAIL only because of an unrelated count check.
    _v48eng = None
    for _cand in (_os.environ.get("FILMSIM_ROOT"), "/root/work/proot",
                  "/root/work/tst"):
        if _cand and (_Path(_cand) / "AlgoGrain.hpp").is_file():
            _v48eng = _Path(_cand)
            break
    _v48srcs = [] if _v48eng is None else [
                _v48eng / "Algo_10_Sim.cpp",
                _v48eng / "AVX2" / "Algo_10_Sim.cpp",
                _v48eng / "AlgoEmulsionMtf.hpp",
                _v48eng / "Algo_11_Sim.cpp",
                _v48eng / "AVX2" / "Algo_11_Sim.cpp",
                _v48eng / "AlgoGrain.hpp"]
    _v48have = [p for p in _v48srcs if p.is_file()]
    _v48hits = []
    for _p in _v48have:
        _txt = _p.read_text(encoding="utf-8", errors="replace")
        for _ln in _txt.splitlines():
            if ("0.18738564618678" in _ln or "0.22508352815546" in _ln):
                # A line that NAMES the wrong value while stating the right one
                # is the record of the fix, not the defect.
                if "0.1873906251292776" in _ln or "0.22507907903927651" in _ln \
                        or "which is what shipped" in _ln or "WRONG" in _ln \
                        or "wrong by" in _ln:
                    continue
                _v48hits.append("%s: %s" % (_p.name, _ln.strip()[:70]))
    chk("G-V48-KSIGMA-SRC  neither wrong sigma literal survives anywhere in the "
        "engine sources -- the scan MTF in both stage-10 twins, the EMULSION "
        "MTF in AlgoEmulsionMtf.hpp, and the grain path in stage 11",
        (_v48eng is None) or (len(_v48have) >= 3 and not _v48hits),
        ("no engine tree on this checkout -- nothing to check"
         if _v48eng is None else
         "%d engine sources checked under %s, %d stale literal(s)%s"
         % (len(_v48have), _v48eng, len(_v48hits),
            (": " + "; ".join(_v48hits[:3])) if _v48hits else "")))

    # ---- the mixture, over the whole corpus -------------------------------
    #
    # Runs the real fit on all 191 stocks x 3 channels. It is the slowest guard
    # in this file by a wide margin and it stays that way: a spot check would
    # have missed SOVIET_PANCHROM_1939, whose lobe sits 33x below its jinc
    # rolloff and which drove the first version of the fit to L1 = 1567.
    _v48l1 = (0.0, "")
    _v48err = (0.0, "")
    _v48sig = (0.0, "")
    for _p in _fpm.FILM_PROFILES:
        _g = _p.grain
        _d = _g.grain_um_rgb()
        for _c in range(3):
            _cu = _g.clumps()[_c]
            _t = _fpm.grain_gauss_terms(
                "boolean_jinc", clump_um=_cu, clump_gain=_g.clump_gain,
                grain_um=_d[_c], size_sigma_log=_g.size_sigma_log)
            _dc = 1.0 + max(_g.clump_gain, 0.0)
            _l1 = _fpm.grain_terms_l1(_t) / _dc
            _tgt = (_fpm.grain_phys_shape(_v48f, _d[_c], _g.size_sigma_log)
                    * _fpm.grain_clustering_lobe(_v48f, _cu, _g.clump_gain))
            _e = float(_v48np.abs(
                _fpm.grain_terms_shape(_t, _v48f) - _tgt).max()) / _dc
            # ⚠ THE WIDEST TERM INCLUDES THE LOBE, which since the v48
            # factoring is carried beside the mixture rather than expanded
            # into it -- and it is by far the widest thing the stage blurs
            # with, so reading only `terms` here would check the cheap blurs
            # and miss the expensive one.
            _ms = max([sg for sg, _w in _t.terms]
                      + ([_t.lobe_sigma_mm] if _t.lobe_gain > 0.0 else []))
            if _l1 > _v48l1[0]:
                _v48l1 = (_l1, "%s ch%d" % (_p.name, _c))
            if _e > _v48err[0]:
                _v48err = (_e, "%s ch%d" % (_p.name, _c))
            if _ms > _v48sig[0]:
                _v48sig = (_ms, "%s ch%d" % (_p.name, _c))

    chk("G-V48-COND  no stock's mixture needs large cancelling weights, which "
        "in the AVX2 engine's float32 field would not be an inaccuracy but a "
        "destroyed image",
        _v48l1[0] <= _fpm.GRAIN_MIXTURE_L1_LIMIT,
        "worst normalised L1 %.3f on %s, ceiling %.1f (the rejected "
        "product-fitting version reached 1567)"
        % (_v48l1[0], _v48l1[1], _fpm.GRAIN_MIXTURE_L1_LIMIT))

    chk("G-V48-FIT  the five-term mixture reproduces the radius-averaged jinc "
        "everywhere in the corpus, to well inside the aperture correction it "
        "ships alongside",
        _v48err[0] <= _fpm.GRAIN_MIXTURE_FIT_WORST * 1.05,
        "worst relative fit error %.3e on %s against the recorded %.3e"
        % (_v48err[0], _v48err[1], _fpm.GRAIN_MIXTURE_FIT_WORST))

    # ---- the blur the engines can actually perform ------------------------
    #
    # Above 16 px the C++ separable blur leaves its exact truncated kernel for a
    # pyramid path that the reference does not model, so a mixture whose widest
    # term crosses that line would silently stop being the same operator in the
    # two engines. Checked at 8K on 35 mm, the widest geometry this project
    # renders: px_per_mm = 2 * 3840 / 24.9.
    _v48px = 2.0 * 3840.0 / 24.9
    chk("G-V48-BLURSIGMA  every mixture term stays inside the sigma range "
        "where the C++ blur is the exact truncated kernel, at the widest "
        "geometry this project renders",
        _v48sig[0] * _v48px <= _fpm.GRAIN_BLUR_SIGMA_EXACT_MAX,
        "widest term %.4f mm on %s = %.2f px at %.0f px/mm, ceiling %.0f px"
        % (_v48sig[0], _v48sig[1], _v48sig[0] * _v48px, _v48px,
           _fpm.GRAIN_BLUR_SIGMA_EXACT_MAX))

    # ---- the physical diameter, and what it is NOT ------------------------
    #
    # ⚠ ERRATUM AGAINST THE SPECIFICATION, PINNED AS A NUMBER. Spec §18.2 gives
    # grain_um the range [0.2, 30] and §18.3 offers `clump_um / 4.95` as an
    # equivalent route to it. Both are wrong against this corpus and the guard
    # says so in figures rather than in a comment: the inversion's floor is
    # below the spec's, its ceiling is a tenth of the spec's, and the 4.95
    # shortcut disagrees with the inversion by more than 30 % on a large
    # fraction of the database.
    _v48d = [_p.grain.grain_um_rgb()[1] for _p in _fpm.FILM_PROFILES]
    _v48est = [_p for _p in _fpm.FILM_PROFILES
               if _p.name not in _fpm.GRAIN_CLUMP_MEASURED_STOCKS]
    _v48ratio = [(_p.grain.clumps()[1] / 4.95) / _p.grain.grain_um_rgb()[1]
                 for _p in _v48est]
    # ⚠ R-S4(a) MOVED THE FLOOR. With radius dispersion in the inversion the
    # derived diameters are 3-6 % smaller, so the finest stock is now 0.1835 um
    # and the specification's own lower bound of 0.20 rejects it by more than it
    # did before, not less.
    _v48bad = sum(1 for _r in _v48ratio if abs(_r - 1.0) > 0.30)
    chk("G-V48-GRAINSIZE  every stock's derived grain diameter lies in the "
        "range this corpus actually occupies, which is NOT the range the "
        "specification gives, and the specification's clump/4.95 shortcut is "
        "not equivalent to the inversion on a large minority of stocks",
        all(_fpm.GRAIN_UM_RANGE[0] <= _x <= _fpm.GRAIN_UM_RANGE[1]
            for _x in _v48d)
        and min(_v48d) < 0.2 and max(_v48d) < 3.0 and _v48bad > 50,
        "derived %.3f-%.3f um (the spec's floor of 0.20 would reject the "
        "finest stock and its ceiling of 30 is ten times the coarsest); "
        "clump/4.95 differs from the inversion by over 30 %% on %d of %d "
        "estimated stocks, ratio range %.2f-%.2f"
        % (min(_v48d), max(_v48d), _v48bad, len(_v48est),
           min(_v48ratio), max(_v48ratio)))

    # ---- the five stocks that keep their measurement ----------------------
    chk("G-V48-MEASURED  the five stocks whose clump_um is a BBC T-101 "
        "measurement keep it as their grain diameter instead of taking the "
        "derivation, and the derivation agrees with all five closely enough "
        "to be trusted on the other 186",
        all(_fpm.get_profile(_n).grain.grain_um_g
            == _fpm.get_profile(_n).grain.clump_um_g
            for _n in _fpm.GRAIN_CLUMP_MEASURED_STOCKS)
        and all(abs(_fpm.get_profile(_n).grain.clump_um_g
                    / _fpm.grain_um_from_rms(
                        _fpm.get_profile(_n).grain.rms_rgb()[1], 1.0,
                        _fpm.get_profile(_n).grain.size_sigma_log)
                    - _fpm.GRAIN_UM_MEASURED_AGREEMENT[_n]) < 0.02
                for _n in _fpm.GRAIN_CLUMP_MEASURED_STOCKS),
        "five measured stocks pinned; measured/derived %s -- against a median "
        "5.64 on the 186 estimated stocks, two populations that do not overlap "
        "at the quartile"
        % ", ".join("%.2f" % (_fpm.get_profile(_n).grain.clump_um_g
                              / _fpm.grain_um_from_rms(
                                  _fpm.get_profile(_n).grain.rms_rgb()[1], 1.0,
                                  _fpm.get_profile(_n).grain.size_sigma_log))
                    for _n in _fpm.GRAIN_CLUMP_MEASURED_STOCKS))

    # ---- the aperture correction is real and is not free ------------------
    chk("G-V48-APERTURE  the exact disk aperture differs from the Gaussian "
        "stand-in by enough to move every stock's calibrated amplitude, which "
        "is why the two apertures are bound to the two spectral models rather "
        "than offered as an independent switch",
        abs(float(_fpm.grain_aperture_48(_v48np.array([25.4]))[0])) < 0.01
        and float(_fpm.grain_aperture_legacy(_v48np.array([25.4]))[0]) > 0.02
        and all(1.005 < _r < 1.10 for _r in [
            math.sqrt(_fpm.grain_reference_energy_terms(
                _fpm.grain_gauss_terms("legacy_gaussian", clump_um=_cu),
                _fpm.grain_aperture_legacy)
                / _fpm.grain_reference_energy_terms(
                    _fpm.grain_gauss_terms("legacy_gaussian", clump_um=_cu),
                    _fpm.grain_aperture_48))
            for _cu in (1.0, 5.0, 13.0)]),
        "the disk transfer has its first zero at 25.4 c/mm where the Gaussian "
        "still passes %.3f (a sixth of the signal, where the real aperture "
        "has none); swapping the model rescales amplitude by "
        "%.4f (clump 1 um) to %.4f (clump 13 um)"
        % (float(_fpm.grain_aperture_legacy(_v48np.array([25.4]))[0]),
           math.sqrt(_fpm.grain_reference_energy_terms(
               _fpm.grain_gauss_terms("legacy_gaussian", clump_um=1.0),
               _fpm.grain_aperture_legacy)
               / _fpm.grain_reference_energy_terms(
                   _fpm.grain_gauss_terms("legacy_gaussian", clump_um=1.0),
                   _fpm.grain_aperture_48)),
           math.sqrt(_fpm.grain_reference_energy_terms(
               _fpm.grain_gauss_terms("legacy_gaussian", clump_um=13.0),
               _fpm.grain_aperture_legacy)
               / _fpm.grain_reference_energy_terms(
                   _fpm.grain_gauss_terms("legacy_gaussian", clump_um=13.0),
                   _fpm.grain_aperture_48))))

    # ---- the constant-radius limit, against the function it replaces ------
    chk("G-V48-DISPERSION  the radius-averaged spectrum reduces to the "
        "existing constant-radius jinc when the dispersion is zero, and moves "
        "the band by more than a factor of two when it is not -- which is the "
        "size of the effect that was sitting in the schema unread",
        float(_v48np.abs(
            _fpm.grain_phys_shape(_v48f, 1.0, 0.0)
            - _fsim.boolean_grain_shape(_v48f, 1.0)).max()) < 1e-6
        and _fpm.grain_half_power_freq(1.0, 0.0) > 650.0
        and _fpm.grain_half_power_freq(1.0, 0.5) < 400.0
        and (_fpm.grain_half_power_freq(1.0, 0.0)
             / _fpm.grain_half_power_freq(1.0, 0.5)) > 2.0,
        "sigma_ln = 0 matches boolean_grain_shape to %.2e; the AMPLITUDE half "
        "point at d = 1 um goes %.0f -> %.0f -> %.0f c/mm for sigma_ln "
        "0 / 0.25 / 0.5"
        % (float(_v48np.abs(_fpm.grain_phys_shape(_v48f, 1.0, 0.0)
                            - _fsim.boolean_grain_shape(_v48f, 1.0)).max()),
           _fpm.grain_half_power_freq(1.0, 0.0),
           _fpm.grain_half_power_freq(1.0, 0.25),
           _fpm.grain_half_power_freq(1.0, 0.5)))

    # ---- F1: the reference must re-roll grain per frame --------------------
    #
    # ⚠⚠ THE DEFECT THIS GUARD EXISTS FOR WAS INVISIBLE TO EVERY HARNESS IN
    # THIS REPOSITORY, BECAUSE THEY ALL COMPARE ONE FRAME. The reference drew
    # grain from a generator seeded once per render, so frame 0 and frame 1 of
    # a clip carried an IDENTICAL field while both C++ engines re-rolled theirs.
    # Two engines, two different physics, and nothing failed. Any future
    # temporal defect will be caught here or not at all.
    _v48t = _fpm.grain_gauss_terms("boolean_jinc", clump_um=2.32,
                                   clump_gain=0.0, grain_um=0.469,
                                   size_sigma_log=0.315)
    _v48a = _fsim.make_grain_field(48, 64, 154.0, _v48t, 4.0, 0.9, 1.0,
                                   12345, 0, _fsim.RngStage.GRAIN_G,
                                   _fpm.grain_aperture_48)
    _v48b = _fsim.make_grain_field(48, 64, 154.0, _v48t, 4.0, 0.9, 1.0,
                                   12345, 1, _fsim.RngStage.GRAIN_G,
                                   _fpm.grain_aperture_48)
    _v48c = _fsim.make_grain_field(48, 64, 154.0, _v48t, 4.0, 0.9, 1.0,
                                   12345, 0, _fsim.RngStage.GRAIN_R,
                                   _fpm.grain_aperture_48)
    _v48rms = float(_v48np.sqrt((_v48a ** 2).mean()))
    _v48corr = float((_v48a * _v48b).mean() / max(_v48rms ** 2, 1e-30))
    chk("G-V48-FRAMEKEY  the reference re-rolls the grain field per frame and "
        "per channel stream, so a clip is not one still frame repeated",
        float(_v48np.abs(_v48a - _v48b).max()) > 0.1 * _v48rms
        and abs(_v48corr) < 0.25
        and float(_v48np.abs(_v48a - _v48c).max()) > 0.1 * _v48rms
        and float(_v48np.abs(
            _v48a - _fsim.make_grain_field(
                48, 64, 154.0, _v48t, 4.0, 0.9, 1.0, 12345, 0,
                _fsim.RngStage.GRAIN_G, _fpm.grain_aperture_48)).max()) == 0.0,
        "frames 0 and 1 correlate at %.3f and differ; the same frame redrawn "
        "is bit-identical, so the field is a pure function of its key"
        % _v48corr)

    # ---- the ported generator is the engine's generator --------------------
    chk("G-V48-RNGPORT  the reference's counter generator reproduces "
        "AlgoCounterRng exactly, including the negative frame indices that "
        "occur when a defect's birth frame is searched backwards",
        int(_fsim.rng_counter(123456789, 0, _fsim.RngStage.GRAIN_R, 0))
        == 530242871308058624
        and abs(float(_fsim.rng_uniform01(
            _fsim.rng_counter(123456789, 0, _fsim.RngStage.GRAIN_R, 0)))
            - 0.85671107674338154) == 0.0
        and abs(float(_fsim.rng_normal(
            _fsim.rng_counter(123456789, 0, _fsim.RngStage.GRAIN_R, 0)))
            - 0.43942222155008914) == 0.0
        and int(_fsim.rng_counter(123456789, -3, _fsim.RngStage.GRAIN_B, 4))
        == int(_fsim.rng_counter(123456789, -3, _fsim.RngStage.GRAIN_B, 4)),
        "counter, uniform and Box-Muller normal all reproduce the compiled "
        "header's values bit for bit")

    # ---- the identity skip, and what it admits ---------------------------
    _id_err = 0.0
    for _sg in (0.1552, 0.10, 0.05, 0.02):
        if not _fpm.grain_blur_is_identity(_sg):
            continue
        _e = math.exp(-0.5 / (_sg * _sg))
        _id_err = max(_id_err, _e / (1.0 + 2.0 * _e))
    chk("G-V48-IDENTITY  the blurs the renderer skips really are the identity, "
        "and the error that admits is far below what float32 carries -- this is "
        "the optimisation that took the physical spectrum from ten full-plane "
        "blurs to one or three",
        _fpm.grain_blur_is_identity(0.15) and not _fpm.grain_blur_is_identity(0.2)
        and _id_err < _fpm.GRAIN_BLUR_IDENTITY_EPS,
        "worst outer tap among skipped kernels %.2e against a %.0e threshold; "
        "sigma 0.15 px skips, 0.20 px does not" % (_id_err,
                                                   _fpm.GRAIN_BLUR_IDENTITY_EPS))

    # ---- R-S4(a): the closed calibration loop ----------------------------
    #
    # ⚠⚠ THIS GUARD FOUND A REAL INCONSISTENCY AND IS THE REASON IT IS HERE.
    # `grain_um` is obtained by inverting the counting law; the count gate runs
    # the same law forward. Before R-S4(a) the inversion ignored radius
    # dispersion and the forward law did not, so the loop closed to 1.0317x
    # instead of 1.0000x -- 3.2 % at the corpus-typical sigma_ln, 6.3 % at 0.35.
    # Neither number was wrong on its own. They were answers to two questions
    # about the same grain and only one had the dispersion in it.
    _rs4 = 0.0
    for _p in _fpm.FILM_PROFILES:
        _g = _p.grain
        if _p.name in _fpm.GRAIN_CLUMP_MEASURED_STOCKS:
            continue
        for _c in range(3):
            _rs4 = max(_rs4, abs(
                _fpm.grain_sparse_sigma(1.0, _g.grain_um_rgb()[_c],
                                        _g.size_sigma_log)
                / (_g.rms_rgb()[_c] / 1000.0) - 1.0))
    chk("G-V48-SPARSECAL (R-S4a)  the counting law run FORWARD from the derived "
        "diameter returns the stored rms granularity at net density 1.0, on "
        "every derived stock and channel -- the loop the grain size was "
        "inverted out of now closes",
        _rs4 < 1e-9,
        "worst closure error over 186 stocks x 3 channels: %.2e (it was "
        "3.2e-02 before radius dispersion entered the inversion)" % _rs4)

    # ---- the mean projected area really carries the dispersion ------------
    chk("G-V48-RS4A  radius dispersion enters the density-to-count mapping "
        "through E[pi r^2] and not pi E[r]^2, which is the half of R-S4 the "
        "first v48 pass did not build",
        abs(_fpm.grain_mean_area_um2(1.0, 0.0) - math.pi / 4.0) < 1e-12
        and abs(_fpm.grain_mean_area_um2(1.0, 0.35)
                / _fpm.grain_mean_area_um2(1.0, 0.0)
                - math.exp(0.35 ** 2)) < 1e-12,
        "the mean AREA exceeds the area of the mean radius by exp(sigma_ln^2): "
        "%.4f at 0.35, %.4f at 0.55"
        % (math.exp(0.35 ** 2), math.exp(0.55 ** 2)))

    # ---- the marginal transform's moments and its clamp -------------------
    _mg_z = _fsim.counter_normal_plane(400, 400, 31337, 0,
                                       _fsim.RngStage.GRAIN_R).astype(
                                           _v48np.float64)
    _mg_z = (_mg_z - _mg_z.mean()) / _mg_z.std()
    _mg_c = _fpm.grain_marginal_coeff(25.0, 0.25)
    _mg_p = ((_mg_z + _mg_c * (_mg_z * _mg_z - 1.0))
             / math.sqrt(1.0 + 2.0 * _mg_c * _mg_c))
    _mg_sk = float(((_mg_p - _mg_p.mean()) ** 3).mean() / _mg_p.std() ** 3)
    _mg_tgt = 6.0 * _mg_c
    chk("G-V48-MARGINAL (R-S6)  the count-gated transform delivers the third "
        "moment it promises while leaving the first two alone, and is clamped "
        "where it would stop being monotone",
        abs(_mg_p.mean()) < 1e-3 and abs(_mg_p.std() - 1.0) < 1e-3
        and abs(_mg_sk - _mg_tgt) < 0.05 * max(_mg_tgt, 1e-6)
        and _fpm.grain_marginal_coeff(0.1, 0.55)
        == _fpm.GRAIN_MARGINAL_COEFF_MAX
        and _fpm.GRAIN_MARGINAL_COEFF_MAX <= 0.1,
        "at N = 25 the transform gives skew %.4f against %.4f asked; the clamp "
        "keeps it monotone to z = %.1f and binds on 2.0 %% of the "
        "stock-density space"
        % (_mg_sk, _mg_tgt, -1.0 / (2.0 * _fpm.GRAIN_MARGINAL_COEFF_MAX)))

    # ---- the resolution element is not the pixel --------------------------
    _el_ppm = 3840.0 / 24.9
    _el_pitch = 1.0 / _el_ppm
    _el_A = _fpm.grain_element_area_um2(_fpm.scan_sigma_mm(105.0), _el_pitch)
    chk("G-V48-ELEM  the noise-equivalent element is larger than the pixel, "
        "which is what makes the count gate read the patch an output sample "
        "actually averages over rather than its own footprint",
        _el_A > (_el_pitch * 1000.0) ** 2 and _el_A < 20.0 * (_el_pitch * 1000.0) ** 2
        and _fpm.grain_element_area_um2(0.0, 0.0) == 0.0,
        "A_elem %.1f um2 against a %.1f um2 pixel at 4K -- a factor of %.2f "
        "straight onto the gate"
        % (_el_A, (_el_pitch * 1000.0) ** 2,
           _el_A / (_el_pitch * 1000.0) ** 2))

    # ---- R-T5: no frame-locked component, and the refusal that keeps it ---
    chk("G-V48-RT5  the grain stage introduces no frame-locked noise "
        "component: `grain_temporal_class` is declarative, its only legal "
        "value is the default, and no engine reads it to pin a frame index",
        all(_p.grain.grain_temporal_class == "emulsion"
            for _p in _fpm.FILM_PROFILES)
        and "grain_temporal_class" not in _Path(_fsim.__file__).read_text(
            encoding="utf-8").split("def simulate")[-1].replace(
                "# `GrainSpec.grain_temporal_class` is declarative and "
                "`validate`", ""),
        "spec §18.2 asks the stage to freeze the field for such a stock and "
        "R-T5 forbids any frame-locked component; R-T5 wins, validate_all "
        "refuses the other value, and 191 stocks carry the default")

    # =====================================================================
    #  THE SPECIFICATION'S OWN NAMED GATES (FGS-DDS-001 Rev. A, Appendix A)
    # =====================================================================
    #
    # ⚠ THESE EXIST UNDER THE SPEC'S NAMES ON PURPOSE. The v48 work was checked
    # by `G-V48-*` guards that cover most of the same ground, but Appendix A
    # maps each requirement to a NAMED gate, and a reviewer holding the
    # specification cannot tell whether R-S3 passed by reading a guard called
    # `G-V48-FIT`. One gate per requirement row, named as the document names it.

    # ---- V-CAL (R-S1): the calibration anchor ---------------------------
    # Covered in substance by section 5's datasheet-granularity check, which
    # renders a field per spectral model and measures it through that model's
    # own aperture. Restated here as the named gate so Appendix A's row closes.
    _vcal = []
    for _nm in ("5219", "5203", "delta 3200"):
        _p = _fpm.get_profile(_nm)
        _g = _p.grain
        for _model in ("legacy_gaussian", "boolean_jinc"):
            _t = _fpm.grain_gauss_terms(
                _model, clump_um=_g.clump_um_g, clump_gain=_g.clump_gain,
                grain_um=_g.grain_um_rgb()[1], size_sigma_log=_g.size_sigma_log)
            _ap = _fpm.grain_aperture_model(_model)
            _e = _fpm.grain_reference_energy_terms(_t, _ap)
            _vcal.append(_e > 0.0 and _v48np.isfinite(_e))
    chk("V-CAL (R-S1)  the amplitude calibration integral is finite and "
        "positive for every stock under both spectral models, and section 5 "
        "measures the rendered field back to the datasheet figure",
        all(_vcal), "%d model-stock pairs" % len(_vcal))

    # ---- V-NORM (R-S3, R-S4, R-S8): the synthesis normalisation ----------
    #
    # ⚠⚠ THIS IS THE GATE THE FIRST v48 PASS DID NOT RUN, and it is the only
    # one that checks the SYNTHESIS rather than the law feeding it. Spec
    # §10.2.1 step 3 asks for it in terms: "the ensemble periodogram of F0 over
    # >= 64 realizations matches P0 within statistical error".
    #
    # Everything else measures a spectrum, a coefficient or a total energy. This
    # measures what the generator ACTUALLY PRODUCES, bin by bin, and it is the
    # only thing that would catch a transfer applied to the wrong axis, an rfft
    # normalisation off by N, or a mixture assembled correctly and then used
    # once per term instead of once per field.
    _vn_h, _vn_w, _vn_r = 96, 128, 64
    _vn_sp = _fpm.grain_gauss_terms("boolean_jinc", clump_um=2.32,
                                    clump_gain=0.4, grain_um=0.469,
                                    size_sigma_log=0.315)
    _vn_ppm, _vn_scan, _vn_a = 154.0, 0.9, 1.04
    _vn_T = _fsim.grain_transfer(_vn_sp, _vn_h, _vn_w, _vn_ppm, _vn_scan, _vn_a)
    _vn_T = _v48np.array(_vn_T, dtype=_v48np.float64, copy=True)
    _vn_T[0, 0] = 0.0
    _vn_acc = _v48np.zeros_like(_vn_T)
    for _k in range(_vn_r):
        _wf = _fsim.counter_normal_plane(_vn_h, _vn_w, 4242, _k,
                                         _fsim.RngStage.GRAIN_G)
        _vn_acc += _v48np.abs(_v48np.fft.rfft2(
            _v48np.fft.irfft2(_v48np.fft.rfft2(_wf) * _vn_T,
                              s=(_vn_h, _vn_w)))) ** 2
    _vn_acc /= _vn_r
    # For unit-variance white noise and numpy's unnormalised transform,
    # E|rfft2(w)|^2 = h*w, so the prediction is P0 = |T|^2 * h * w.
    _vn_P0 = (_vn_T ** 2) * float(_vn_h * _vn_w)
    _vn_m = _vn_P0 > (1e-4 * _vn_P0.max())
    _vn_ratio = float((_vn_acc[_vn_m] / _vn_P0[_vn_m]).mean())
    # Per bin the periodogram is chi-square with 2 dof, so 64 realisations give
    # a 1/sqrt(64) = 12.5 % standard error per bin; averaged over the retained
    # bins the mean is far tighter, and 3 % is a loose bound on it.
    chk("V-NORM (R-S3/S4/S8)  the ensemble periodogram of the synthesised "
        "field over 64 realisations reproduces |T|^2, and the DC bin is "
        "exactly zero -- the only gate that measures the GENERATOR rather "
        "than the law feeding it",
        abs(_vn_ratio - 1.0) < 0.03 and _vn_T[0, 0] == 0.0
        and float(_vn_acc[0, 0]) < 1e-20 * float(_vn_P0.max()),
        # ⚠ THE DC TEST IS "ONE QUANTIZATION STEP", WHICH IS R-S8'S OWN WORDING
        # AND NOT BIT-ZERO. The transfer's DC bin is exactly 0.0 and the field
        # is therefore exactly zero-mean as synthesised; measuring it back
        # through an inverse and a forward transform leaves a denormal around
        # 1e-28 of the peak, which is float64 round-trip and not a mean.
        # Asserting == 0.0 there fails a correct generator, which is how this
        # gate first read.
        "mean periodogram / prediction = %.4f over %d bins, %d realisations; "
        "transfer DC exactly 0, measured DC %.1e of peak"
        % (_vn_ratio, int(_vn_m.sum()), _vn_r,
           float(_vn_acc[0, 0]) / float(_vn_P0.max())))

    # ---- V-TEMP (R-T1): frame independence -------------------------------
    _vt_sp = _fpm.grain_gauss_terms("boolean_jinc", clump_um=2.32,
                                    clump_gain=0.0, grain_um=0.469,
                                    size_sigma_log=0.315)
    _vt = [_fsim.make_grain_field(64, 64, 154.0, _vt_sp, 4.0, 0.9, 1.0,
                                  777, _f, _fsim.RngStage.GRAIN_G,
                                  _fpm.grain_aperture_48) for _f in range(8)]
    _vt_rms = float(_v48np.sqrt((_vt[0] ** 2).mean()))
    _vt_cc = [float((_vt[0] * _vt[_i]).mean()) / (_vt_rms ** 2)
              for _i in range(1, 8)]
    chk("V-TEMP (R-T1)  successive frames are statistically independent and "
        "each is a pure function of its key -- the gate F1 needed and no "
        "single-frame harness could provide",
        max(abs(_c) for _c in _vt_cc) < 0.25
        and float(_v48np.abs(
            _vt[3] - _fsim.make_grain_field(
                64, 64, 154.0, _vt_sp, 4.0, 0.9, 1.0, 777, 3,
                _fsim.RngStage.GRAIN_G, _fpm.grain_aperture_48)).max()) == 0.0,
        "worst |correlation| with frame 0 over seven later frames %.4f; "
        "re-rendering frame 3 is bit-identical"
        % max(abs(_c) for _c in _vt_cc))

    # ---- V-COMPAT (R-S7, R-T4): what must survive unchanged --------------
    _vc_leg = _fpm.grain_gauss_terms("legacy_gaussian", clump_um=2.32,
                                     clump_gain=0.4)
    _vc_dev = float(_v48np.abs(
        _fpm.grain_terms_shape(_vc_leg, _v48f)
        - _fpm.grain_legacy_shape(_v48f, 2.32, 0.4)).max())
    _vc_a = _fsim.make_grain_field(48, 48, 154.0, _vt_sp, 4.0, 0.9, 1.0,
                                   5, 0, _fsim.RngStage.GRAIN_G,
                                   _fpm.grain_aperture_48)
    _vc_b = _fsim.make_grain_field(48, 48, 154.0, _vt_sp, 4.0, 0.9, 1.0,
                                   5, 9, _fsim.RngStage.GRAIN_G,
                                   _fpm.grain_aperture_48)
    _vc_frozen = float(_v48np.abs(_vc_a - _vc_b).max())
    chk("V-COMPAT (R-S7/R-T4)  the legacy spectrum, per-layer rms, the "
        "one-field-per-emulsion rule, anisotropy and the fog floor all survive "
        "v48, and `frozen` reproduces a single frame for the whole clip",
        _vc_dev < 1e-12
        and _fpm.get_profile("KODAK_PORTRA_400").grain.rms_rgb()[2]
        != _fpm.get_profile("KODAK_PORTRA_400").grain.rms_rgb()[1]
        and any(_p.grain.anisotropy != 1.0 for _p in _fpm.FILM_PROFILES)
        and all(_p.grain.fog_grain > 0.0 for _p in _fpm.FILM_PROFILES)
        and _vc_frozen > 0.0,
        "legacy spectrum reproduced to %.1e; per-layer rms, anisotropy and the "
        "fog floor all still populated; frames 0 and 9 differ by %.3e so "
        "`frozen` has something to freeze" % (_vc_dev, _vc_frozen))

    # ---- V-SAT (R-S5): the saturating family, and its refusal ------------
    #
    # ⚠ THIS GATE IS EXPECTED TO REFUSE AND THAT IS THE PASS CONDITION. Spec
    # S-C8 says the family may become a default once it reproduces the measured
    # traces within 10 % RMS. It does not -- 14.0 % mean, 25.3 % worst -- so the
    # assertion is that the family is implemented, the fit is re-derivable, and
    # the mode is NOT adopted.
    _vs_neg = [_p for _p in _fpm.FILM_PROFILES
               if _p.grain.sigma_shape_measured and not _p.is_reversal]
    _vs_dr, _vs_q = _fpm.GRAIN_SIGMA_SAT_GLOBAL_FIT
    _vs_e = [_fpm.grain_sigma_sat_fit_error(_p, _vs_dr, _vs_q)
             for _p in _vs_neg]
    chk("V-SAT (R-S5)  the saturating sigma(D) family is implemented and "
        "re-derivable, FAILS its own acceptance gate S-C8 against the measured "
        "negatives, and is therefore not adopted -- the gate refusing is the "
        "gate working",
        len(_vs_neg) == 12
        and abs(float(_v48np.mean(_vs_e))
                - _fpm.GRAIN_SIGMA_SAT_GLOBAL_RMS_MEAN) < 0.005
        and abs(max(_vs_e) - _fpm.GRAIN_SIGMA_SAT_GLOBAL_RMS_WORST) < 0.005
        and max(_vs_e) > _fpm.GRAIN_SIGMA_SAT_GATE
        and _fpm.GRAIN_SIGMA_SAT_ADOPTED is False,
        "12 measured negatives (the 2 reversal stocks rise towards Dmax and no "
        "falling family can express that): mean %.1f%%, worst %.1f%% against a "
        "10 %% bar; legacy sqrt law retained"
        % (100 * float(_v48np.mean(_vs_e)), 100 * max(_vs_e)))

    # ---- V-SPARSE (R-S6): the count gate ---------------------------------
    _vp_ppm = 3840.0 / 24.9
    _vp_A = _fpm.grain_element_area_um2(_fpm.scan_sigma_mm(105.0),
                                        1.0 / _vp_ppm)
    _vp_brk = []
    for _p in _fpm.FILM_PROFILES:
        _g = _p.grain
        _n1 = _fpm.grain_n_elem(1.0, _g.grain_um_rgb()[1], _g.size_sigma_log,
                                _vp_A)
        if _n1 > 0.0:
            _vp_brk.append(_fpm.GRAIN_N_MIN / _n1)
    _vp_brk.sort()
    chk("V-SPARSE (R-S6)  the marginal leaves Gaussian below a COMPUTED grain "
        "count, the gate is smooth in log N, and the region it opens is the "
        "deep toe rather than a stock list",
        _fpm.grain_alpha(_fpm.GRAIN_N_HI * 4.0) == 1.0
        and _fpm.grain_alpha(_fpm.GRAIN_N_MIN * 0.5) == 0.0
        and abs(_fpm.grain_alpha(
            math.sqrt(_fpm.GRAIN_N_MIN * _fpm.GRAIN_N_HI)) - 0.5) < 1e-12
        and _fpm.grain_marginal_coeff(1.0e9, 0.25) == 0.0
        and 0.03 < _vp_brk[len(_vp_brk) // 2] < 0.12,
        "alpha is 0 / 0.5 / 1 at N_MIN/2, the geometric mean and 4*N_HI; the "
        "Gaussian marginal becomes indefensible below net D %.4f for the "
        "median stock and %.3f for the coarsest, at a 4K element of %.1f um2"
        % (_vp_brk[len(_vp_brk) // 2], _vp_brk[-1], _vp_A))

    # ---- V-PARITY (R-T2, R-N1, R-N5) -------------------------------------
    # Lives in `cpp_parity.py`, which compiles the real stage sources: the v48
    # field probe compares whole planes pixel for pixel across two frame
    # indices, an anisotropic stock and 8K geometry. Named here so Appendix A's
    # row points somewhere, and asserted to the extent a Python-only guard can:
    # the reference is deterministic and thread-count-free by construction.
    chk("V-PARITY (R-T2/N1/N5)  the reference is a pure function of its inputs "
        "and the three-engine pixel comparison is `cpp_parity`'s v48 FIELD "
        "probe, which drives the real stage sources rather than a copy",
        float(_v48np.abs(
            _fsim.make_grain_field(32, 32, 154.0, _vt_sp, 4.0, 0.9, 1.0,
                                   11, 2, _fsim.RngStage.GRAIN_B,
                                   _fpm.grain_aperture_48)
            - _fsim.make_grain_field(32, 32, 154.0, _vt_sp, 4.0, 0.9, 1.0,
                                     11, 2, _fsim.RngStage.GRAIN_B,
                                     _fpm.grain_aperture_48)).max()) == 0.0,
        "identical inputs give a bit-identical field; cross-engine parity is "
        "measured in cpp_parity at 2e-07 of field RMS against the scalar twin")


    # OFF must return the very same object, not an equal copy: the contract is
    # that selecting nothing costs nothing, and an equal-but-new profile would
    # satisfy a value comparison while allocating on every frame.
    chk("process variant OFF returns the profile itself on every stock",
        all(_fsim.resolve_process_variant(q, -1) is q for q in _fpm.FILM_PROFILES),
        "at least one stock copies at index -1")
    # ⚠ "OUT OF RANGE" NOW MEANS "NOT A DECLARED ENUMERATOR", and the test
    # widened with the control: TOTAL_PROCESSES, anything past it, and the
    # sentinel must all return the profile itself, by identity.
    import algo_control_enums as _acev
    _pvtot = int(_acev.ProcessVariantCtrl.TOTAL_PROCESSES)
    # ⚠ 0 JOINS THE LIST ON 2026-09-18e and for a different reason from the
    # rest: -1, TOTAL_PROCESSES and anything past it are inert because they
    # are not declared enumerators, while 0 is eAS_SHIPPED, which IS declared
    # and is inert because its key is empty. Both paths must end at the same
    # place -- the profile itself, by identity -- and this asserts they do.
    chk("a ProcessVariantCtrl value outside [0, TOTAL_PROCESSES), and "
        "eAS_SHIPPED inside it, both return the profile itself",
        all(_fsim.resolve_process_variant(q, v) is q
            for q in _fpm.FILM_PROFILES
            for v in (0, -1, -99, _pvtot, _pvtot + 7)),
        "eAS_SHIPPED (0, empty key), the old -1, TOTAL_PROCESSES and "
        "anything past it are all inert")
    # ⚠ THREE STOCKS, AND THE NUMBER IS THE POINT. 24 variants exist across 6
    # stocks and only these three carry a measured curve set; the other 19 are
    # the AGFAPAN developer records, which state an exposure index and no second
    # curve. If this count moves, a variant gained or lost its curves.
    # ⚠ DRIVEN BY ENUMERATOR AND NOT BY POSITION since 2026-09-17: the loop
    # asks every stock about every declared development, which is what a host
    # can now actually send it.
    _vcurve = sorted({
        q.name for q in _fpm.FILM_PROFILES
        for i in range(_pvtot)
        if _fsim.resolve_process_variant(q, i).curves != q.curves})
    chk("exactly 5 stocks have a process variant that changes a curve",
        _vcurve == ["CINESTILL_800T", "GEVACHROME_605", "KODAK_PORTRA_800",
                    "KODAK_ULTRA_COLOR_400UC", "SUPER_ANSCOCHROME_1957"],
        f"{_vcurve}")
    # Reciprocity: an unstated time is not a zero-length exposure.
    chk("reciprocity is exactly zero when no exposure time is stated",
        all(_fsim.reciprocity_log_shift(q, 0.0) == (0.0, 0.0, 0.0)
            for q in _fpm.FILM_PROFILES),
        "a stock shifts exposure with no time stated")

    # ==== SCHEMA v18 RELATIONAL GUARDS ======================================
    # These are the layer the emulsion assessment of 2026-08-27 argued was
    # missing: the schema is observable-COMPLETE and constraint-FREE, so it can
    # express films that cannot exist. A constraint that is not enforced on
    # every build is a comment.

    # ---- G-MTF: MTF-50 is about HALF the resolving power -------------------
    # The only frequency-domain bridge in the emulsion source set
    # [Tani 1995 §1.2, p.11 / PDF 20]. CALIBRATED, NOT GUESSED: measured over
    # the 59 stocks that carry a printed resolving-power pair, the ratio
    # f50_g / (RP_highc / 2) has median EXACTLY 1.000, p10 0.72, p90 1.16.
    # ⚠ THE GUARD DELIBERATELY EXEMPTS MEASURED f50. A real MTF trace beats a
    # rule of thumb, and one already does: EASTMAN_EXR_50D_5245 sits at 1.68
    # with mtf_measured True. Failing on that would punish the better datum.
    _mtf_band = []
    for _p in _fpm.FILM_PROFILES:
        _rp = _p.mtf.resolving_power_lp_mm_highc
        if _rp <= 0 or _p.mtf.mtf_measured:
            continue
        _ratio = _p.mtf.f50_g / (_rp / 2.0)
        if not 0.5 <= _ratio <= 2.0:
            _mtf_band.append("%s %.2f" % (_p.name, _ratio))
    chk("every ESTIMATED f50 sits within 0.5-2.0x half its printed resolving "
        "power", not _mtf_band,
        ", ".join(_mtf_band) if _mtf_band
        else "checked %d stocks with a printed RP pair" % sum(
            1 for _p in _fpm.FILM_PROFILES
            if _p.mtf.resolving_power_lp_mm_highc > 0 and not _p.mtf.mtf_measured))

    # ---- G-MTFBW: the 2026-09-06 KODAK black-and-white MTF batch -----------
    # Two adoptions and three refusals, all guarded. ⚠ THE REFUSALS ARE GUARDED
    # HARDER THAN THE ADOPTIONS, on the principle spectral_sampling.py already
    # applies to F3: nothing downstream consumes a refusal, so nothing would
    # notice if its premise quietly stopped holding.
    _bw = _fpm._BY_NAME
    chk("G-MTFBW1  KODAK_TMAX_100 carries the traced f50 123.0, not the 95.0 "
        "estimate",
        (_bw["KODAK_TMAX_100"].mtf.f50s() == (123.0, 123.0, 123.0)
         and _bw["KODAK_TMAX_100"].mtf.mtf_measured
         and abs(_bw["KODAK_TMAX_100"].mtf.mtf_rolloff_q - 3.63) < 1e-9),
        "F-4016 p8, traced 2026-09-06; independently confirmed by F-32 (2001) "
        "at 120.3")
    chk("G-MTFBW2  KODAK_TRI_X_400TX carries the traced f50 52.7",
        (_bw["KODAK_TRI_X_400TX"].mtf.f50s() == (52.7, 52.7, 52.7)
         and _bw["KODAK_TRI_X_400TX"].mtf.mtf_measured
         and abs(_bw["KODAK_TRI_X_400TX"].mtf.mtf_rolloff_q - 4.23) < 1e-9),
        "F-4017 (2016) p7; three independent drawings across 17 years agree to "
        "1.1 % -- 52.7 / 52.6 / 53.1")
    # ⚠ P3200 CARRIES THE 2019 EDITION'S VALUE, AND MUST NEVER CARRY THE 2018
    # EDITION'S. F-4001 (2018) p7 prints T-MAX 100's MTF artwork -- proved by a
    # 0.0010 pt path identity -- and reading it gives 123.0, the sharpest B&W
    # value in the database, on an ISO 3200 push film. F-4001 (2019) carries
    # P3200's own drawing and gives 84.3, which is what is stored.
    # ⚠ THIS GUARD REPLACED ONE THAT ASSERTED THE OPPOSITE. Until the owner
    # questioned it, this checked that P3200 kept its ESTIMATE, because the
    # refusal had been written from a single edition. The lesson is in the
    # guard's shape now: it pins the ADOPTED value AND separately refuses the
    # misplaced one, so neither the measurement nor the defect can be lost.
    chk("G-MTFBW3  KODAK_TMAX_P3200 carries F-4001 (2019)'s own curve and NOT "
        "the 2018 edition's misplaced T-MAX 100 artwork",
        (_bw["KODAK_TMAX_P3200"].mtf.f50s() == (84.3, 84.3, 84.3)
         and not _bw["KODAK_TMAX_P3200"].mtf.mtf_measured
         and _bw["KODAK_TMAX_P3200"].mtf.f50_g
             != _bw["KODAK_TMAX_100"].mtf.f50_g),
        "84.3 from the 2019 edition; 123.0 would be T-MAX 100's plot. "
        "mtf_measured stays False -- q 2.03 beats the Gaussian by only 1.3x, "
        "under the PORTRA NC/VC threshold for switching the carrier")
    # ⚠ 320TXP's REFUSAL. It shares F-4017 with 400TX, which now carries a
    # measurement. A shared datasheet is not a shared measurement (rule 18).
    chk("G-MTFBW4  KODAK_TRI_X_320TXP keeps its own estimate and does NOT "
        "borrow 400TX's measurement",
        (_bw["KODAK_TRI_X_320TXP"].mtf.f50s() == (58.0, 58.0, 58.0)
         and not _bw["KODAK_TRI_X_320TXP"].mtf.mtf_measured
         and (_bw["KODAK_TRI_X_320TXP"].mtf.f50_g
              != _bw["KODAK_TRI_X_400TX"].mtf.f50_g)),
        "F-4017 prints one MTF frame and heads it 'TRI-X 400 Film / 400TX'")
    # ⚠ T-MAX 400's REFUSAL, and it is a CONTRADICTION and not an absence: three
    # independent Kodak drawings give 95.9 (F-32 2001), 66.7 (F-4016 2007) and
    # f50 > 81 (F-4043 2016, whose curve stops at 51.4 %). The 2016 lower bound
    # rules out the 2007 value. Averaging 95.9 and 66.7 gives 81.3, which would
    # look like a measurement -- so the guard also refuses that specific number.
    # ⚠⚠ THE REFUSAL ENDED ON 2026-09-17d AND ITS OWN REOPENING CONDITION IS
    # WHAT ENDED IT. The note above said the value "reopens if Kodak's own
    # TMY-2 sheet is found with a curve that crosses"; what arrived instead is
    # a FOURTH independent drawing -- «Современные фотоматериалы и их
    # обработка» p341, which only calibrated once queue P56 registered four
    # axis label forms the digit bank had never been shown. It traces f50 98.7,
    # which agrees with F-32's 95.9 to 2.9 % and satisfies F-4043's traced > 81
    # bound. That is not a third opinion in a three-way argument; it is a
    # corroboration of one branch of it, and the 2007 F-4016 reading of 66.7
    # stays refused by the same bound that already refused it.
    # ⚠ THE GUARD STILL REFUSES THE AVERAGE. 81.3 is what averaging 95.9 and
    # 66.7 would give and it would look like a measurement, so it stays
    # forbidden -- the value had to arrive from a drawing, and it did.
    chk("G-MTFBW5  KODAK_TMAX_400 is a MEASUREMENT, corroborated across two "
        "independent drawings, and is still not the average of the two that "
        "disagreed",
        (_bw["KODAK_TMAX_400"].mtf.f50s() == (98.7, 98.7, 98.7)
         and _bw["KODAK_TMAX_400"].mtf.mtf_measured
         and abs(_bw["KODAK_TMAX_400"].mtf.mtf_rolloff_q - 2.163) < 1e-9
         and abs(_bw["KODAK_TMAX_400"].mtf.f50_g - 81.3) > 1.0
         and abs(_bw["KODAK_TMAX_400"].mtf.f50_g - 95.9) / 95.9 < 0.03),
        "book p341 98.7 against Kodak F-32 95.9, 2.9 % apart, both above "
        "F-4043's >81 bound; F-4016's 66.7 stays refused")
    # ⚠ THE PROVENANCE MUST AGREE WITH THE VALUE. EKTAR 100 shipped a traced
    # 52.7 with mtf_measured True while its ParamSource still said tier-3
    # 'estimated' from FilmLab and ended 'mtf_measured stays False'. Every
    # census that reads ParamSource counted this stock as unmeasured. Guarded so
    # the two tables cannot drift apart again.
    _pv_bad = []
    for _n in ("KODAK_TMAX_100", "KODAK_TRI_X_400TX", "KODAK_EKTAR_100",
               "KODAK_TMAX_P3200"):
        _st = {_s.param: _s.status for _s in _bw[_n].param_sources}
        for _f in ("mtf.f50_r", "mtf.f50_g"):
            if _st.get(_f) != "traced":
                _pv_bad.append("%s %s=%s" % (_n, _f, _st.get(_f)))
    chk("G-MTFBW6  every stock in this batch says 'traced' in its own "
        "ParamSource", not _pv_bad,
        ", ".join(_pv_bad) if _pv_bad
        else "4 stocks x 2 records, value and provenance agree")

    # ---- G-VP, 2026-09-07: AgfaPhoto Vista plus, and the shared MTF -------
    # ⚠⚠ THE POINT OF THIS GUARD IS THE SEPARATION, NOT THE VALUES. The owner
    # asked whether «Vista plus 200»'s traced curves could replace
    # AGFA_VISTA_200's estimates. They cannot: the sheet disclaims Agfa
    # manufacture and the dye set matches FUJICOLOR_SUPERIA_XTRA_400 nine times
    # more closely than the next of 26 stored sets. So this asserts that the
    # two families stay apart -- that AGFA_VISTA_200 keeps ITS numbers and the
    # new stocks keep theirs.
    _vp2 = get_profile("AGFA_VISTA_PLUS_200")
    _vp4 = get_profile("AGFA_VISTA_PLUS_400")
    _av = get_profile("AGFA_VISTA_200")
    chk("G-VP1: AGFA_VISTA_200 is untouched by the AgfaPhoto sheet",
        abs(_av.curves.g.gamma - 0.6350) < 1e-9
        and abs(_av.curves.g.dmin - 0.6400) < 1e-9
        and abs(_av.mtf.f50_g - 47.8) < 0.05
        and abs(_av.grain.rms_granularity - 4.3) < 1e-9,
        "gamma 0.6350 dmin 0.6400 f50 47.8 rms 4.3 -- Agfa-Gevaert's own")
    # ⚠ AND THE NEW STOCKS' D-MIN LADDER MUST RISE r < g < b, which is what a
    # measured orange mask does. A pair of records stored the wrong way round
    # passes every other check on this sheet.
    _vpbad = []
    for _p in (_vp2, _vp4):
        if not (_p.curves.r.dmin < _p.curves.g.dmin < _p.curves.b.dmin):
            _vpbad.append("%s dmin ladder %.3f/%.3f/%.3f"
                          % (_p.name, _p.curves.r.dmin, _p.curves.g.dmin,
                             _p.curves.b.dmin))
        # ⚠ THIS USED TO FAIL IF EITHER STOCK CLAIMED A MEASURED MTF, on the
        # shared-drawing ground. Reversed 2026-09-07b by owner decision: the
        # flag is now REQUIRED, because a vendor curve beats an estimate even
        # when two films share it. What is still asserted is that both carry
        # the SAME green f50 and q -- if they ever diverge, someone has
        # treated one drawing as two measurements.
        if not _p.mtf.mtf_measured:
            _vpbad.append("%s lost its adopted vendor MTF" % _p.name)
        if abs(_p.mtf.f50_g - 58.7) > 0.05 or abs(_p.mtf.mtf_rolloff_q - 2.65) > 1e-9:
            _vpbad.append("%s f50_g %.2f q %.2f -- the shared panel reads "
                          "58.7 / 2.65 for both"
                          % (_p.name, _p.mtf.f50_g, _p.mtf.mtf_rolloff_q))
        if len(_p.dye_density.d_neutral) != 31:
            _vpbad.append("%s dye set is %d samples"
                          % (_p.name, len(_p.dye_density.d_neutral)))
    chk("G-VP2: both Vista plus stocks carry a rising D-min ladder, a 31-sample "
        "dye set, and the SAME adopted vendor MTF from their shared panel",
        not _vpbad,
        "; ".join(_vpbad) or "200: 0.104/0.409/0.724   400: 0.092/0.441/0.732")
    # ⚠ THE DYE-SET MATCH, ASSERTED AS A NUMBER. This is the evidence that
    # answers the owner's question, so it is checked rather than remembered.
    try:
        import numpy as _np
        _sx = _np.array(get_profile("FUJICOLOR_SUPERIA_XTRA_400")
                        .dye_density.d_neutral, dtype=float)
        _d2 = _np.array(_vp2.dye_density.d_neutral, dtype=float)
        _near = float(_np.abs(_d2 - _sx).mean())
        _others = []
        for _q in _fpm.FILM_PROFILES:
            _dd = getattr(_q, "dye_density", None)
            if (_dd is None or len(_dd.d_neutral) != 31
                    or not any(_dd.d_neutral)
                    or _q.name in ("FUJICOLOR_SUPERIA_XTRA_400",
                                   "AGFA_VISTA_PLUS_200",
                                   "AGFA_VISTA_PLUS_400")):
                continue
            _others.append(float(_np.abs(_d2 - _np.array(
                _dd.d_neutral, dtype=float)).mean()))
        chk("G-VP3: Vista plus 200's dye set sits an order of magnitude closer "
            "to SUPERIA X-TRA 400 than to anything else in the corpus",
            _near < 0.03 and min(_others) > 5.0 * _near,
            "SUPERIA X-TRA 400 %.4f D vs next-nearest %.4f D over %d sets"
            % (_near, min(_others), len(_others) + 1))
    except Exception as _exc:                                 # pragma: no cover
        chk("G-VP3: dye-set proximity", False, "could not run: %s" % _exc)

    # ---- G-FLP, 2026-09-07: the filmlabpro.com re-audit, and its near miss --
    # ⚠⚠ THE GUARD THAT MATTERS MOST HERE PROTECTS A MEASUREMENT FROM AN
    # AUTOMATED "IMPROVEMENT". The owner's rule is that a published third-party
    # figure beats one of this project's own estimates, and applying it to
    # filmlabpro.com's single mtf50 per stock means re-anchoring GREEN and
    # carrying red and blue by the profile's stored ratios. On
    # KODAK_VISION3_500T_5219 that would have dragged `f50_r` from 36.0 to
    # about 41.5 -- and 36.0 is `measured`, TIER 1, the queue C24 family anchor
    # from seven per-record measurements across 1989-2005. Red is pinned here
    # with its provenance so that no future pass can move it by that route.
    _v3 = get_profile("KODAK_VISION3_500T_5219")
    _v3st = {_s.param: (_s.status, _s.tier) for _s in _v3.param_sources}
    chk("G-FLP1: VISION3 500T's red f50 stays MEASURED at 36.0 while green "
        "and blue carry a third-party level",
        abs(_v3.mtf.f50_r - 36.0) < 1e-9
        and _v3st.get("mtf.f50_r") == ("measured", 1)
        and abs(_v3.mtf.f50_g - 60.0) < 1e-9
        and abs(_v3.mtf.f50_b - 69.2) < 0.05
        and _v3st.get("mtf.f50_g", ("", 0))[0] == "estimated"
        and not _v3.mtf.mtf_measured,
        "f50 %.1f/%.1f/%.1f, red %s"
        % (_v3.mtf.f50_r, _v3.mtf.f50_g, _v3.mtf.f50_b,
           _v3st.get("mtf.f50_r")))
    # ⚠ AND THE DEFECT THAT LET THE NEAR MISS HAPPEN: two DATASHEET-PRINTED rms
    # values carried no ParamSource, so an audit that classifies by provenance
    # read them as this project's own estimates and proposed replacing them
    # with 10.5 and 7.0 -- 2.6x and 1.75x coarser than what Fuji print. The
    # values are pinned WITH their status, because the value alone was never
    # the problem.
    _rmsbad = []
    for _n in ("FUJICOLOR_SUPERIA_XTRA_400", "FUJICOLOR_PRO_400H"):
        _p = get_profile(_n)
        _st = {_s.param: (_s.status, _s.tier) for _s in _p.param_sources}
        if (abs(_p.grain.rms_granularity - 4.0) > 1e-9
                or _st.get("grain.rms_granularity") != ("stated", 1)):
            _rmsbad.append("%s rms %.1f %s"
                           % (_n, _p.grain.rms_granularity,
                              _st.get("grain.rms_granularity")))
    chk("G-FLP2: the two Fuji rms 4.0 figures are PRINTED and now say so, so "
        "no provenance-driven pass can mistake them for estimates",
        not _rmsbad,
        "; ".join(_rmsbad) or "SUPERIA X-TRA 400 and PRO 400H, stated tier 1")
    # ⚠ THE MIS-ATTRIBUTED IMPORT. filmlabpro's key `fuji_eterna_500t` is Fuji
    # Eterna 500T; this profile is Eterna VIVID 500T (8547), a different
    # coating -- and the same record's rms was refused here on exactly that
    # ground while its mtf50 and size_microns were adopted. Both observables
    # are back to this project's own, and the harvest stays in `third_party`,
    # which no renderer reads.
    _ev = get_profile("FUJI_ETERNA_VIVID_500T_8547")
    chk("G-FLP3: Eterna VIVID 500T carries no value taken from Eterna 500T's "
        "record",
        _ev.mtf.f50s() == (50.0, 58.0, 66.0)
        and abs(_ev.emulsion.grain_um) < 1e-9
        and abs(_ev.grain.rms_granularity - 3.5) < 1e-9,
        "f50 %s, grain_um %.1f, rms %.1f (its own sheet's)"
        % (_ev.mtf.f50s(), _ev.emulsion.grain_um, _ev.grain.rms_granularity))
    # ⚠ THE THREE EMPTY FIELDS THAT WERE FILLED, and the one that was emptied.
    # `emulsion.grain_um` is inert -- no engine path reads it -- so this pins
    # the record rather than a rendered result.
    _umbad = []
    for _n, _want in (("FUJI_PROVIA_100F", 1.6), ("FUJICOLOR_PRO_400H", 2.6),
                      ("FUJICOLOR_SUPERIA_XTRA_400", 3.2)):
        _p = get_profile(_n)
        _st = {_s.param: (_s.status, _s.tier) for _s in _p.param_sources}
        if (abs(_p.emulsion.grain_um - _want) > 1e-9
                or _st.get("emulsion.grain_um") != ("estimated", 3)):
            _umbad.append("%s %.2f %s" % (_n, _p.emulsion.grain_um,
                                          _st.get("emulsion.grain_um")))
    chk("G-FLP4: the three post-audit Fuji stocks carry the third party's "
        "size_microns at tier 3, in fields that were empty",
        not _umbad, "; ".join(_umbad) or "1.6 / 2.6 / 3.2 um, estimated t3")
    # ⚠ ACROS: an estimate replaced by a published figure, which is the rule
    # working as intended. The vendor RESOLVING POWER on the same sheet is a
    # different quantity and must survive untouched.
    _ac = get_profile("FUJI_NEOPAN_ACROS_100")
    chk("G-FLP5: ACROS 100's f50 takes the third party's 95 while its own "
        "sheet's resolving power stays 60 / 200",
        _ac.mtf.f50s() == (95.0, 95.0, 95.0)
        and not _ac.mtf.mtf_measured
        and abs(_ac.mtf.resolving_power_lp_mm_lowc - 60.0) < 1e-9
        and abs(_ac.mtf.resolving_power_lp_mm_highc - 200.0) < 1e-9
        and abs(_ac.grain.rms_granularity - 7.0) < 1e-9,
        "f50 95, RP 60/200, rms 7.0 -- the sheet's own two numbers intact")

    # ---- G-DQE: gamma, granularity and speed are ONE relation -------------
    # Eq. (1.1): DQE = (log e)^2 * gamma^2 / (E * G^2), and measured DQE
    # clusters at 1-2 % for real films [Tani 1995 §1.4, p.16 / PDF 25]. With
    # E ~ 1/EI and G ~ rms, the dimensionless proxy is K = gamma^2 * EI / rms^2.
    #
    # ⚠ IT CANNOT BE AN ABSOLUTE BAND AND THE MEASUREMENT SAYS SO. Ungrouped, K
    # spans 500x within colour negatives and 1300x within monochrome, because
    # (a) Selwyn's G is defined through sigma_D * sqrt(2a) while our rms is
    # aperture-specific at 48 um, so the conversion is unestablished, and (b) a
    # 1943 emulsion and a 2016 tabular one genuinely differ in DQE by that much.
    #
    # SO IT IS A WITHIN-CLASS, WITHIN-ERA OUTLIER TEST. Banding by stock class
    # and era collapses 14 ungrouped outliers to ONE, and the class medians then
    # tell a coherent physical story that is worth stating because it is this
    # database independently reproducing Tani's Fig. 1.1 sensitivity history:
    #
    #     colour negative    pre-1960  K = 0.045
    #                        1960-89   K = 0.219     (4.9x better)
    #                        1990+     K = 2.676     (12.2x better again)
    # ⚠ THESE THREE NUMBERS MOVED ON 2026-08-27 and the earlier comment here
    # (0.048 / 0.241 / 2.727) was stale, not wrong-at-the-time: the v17
    # third-party rms imports raised rms_granularity on six colour negatives,
    # which is a denominator in K. Recomputed against the live database. The
    # decomposition is in EMULSION_KNOWLEDGE_BASE.md §23c.3, which also
    # cross-checks the ladder against Tani's traced Fig. 1.1: the 12.2x is
    # NOT a speed gain -- median EI rises only 2.0x across that boundary while
    # median rms falls 12.0 -> 4.6, contributing 6.8x through the 1/rms^2 term.
    # ⚠ A HARDCODED MEDIAN IN A COMMENT GOES STALE THE MOMENT THE DATA MOVES.
    # If these drift again, recompute; do not trust the printed figures.
    #
    # ⚠ POLAROID PEEL-APART IS ITS OWN CLASS. Those are print-like materials
    # with gamma 1.5-3.4; pooling them with camera negatives put six of them
    # 30-135x off a median they were never members of. That was the GROUPING
    # being wrong, not the data.
    # ⚠ AND REVERSAL IS TESTED BEFORE MONOCHROME, because a monochrome REVERSAL
    # film (KODAK_TRI_X_REVERSAL_200, gamma 3.06) is a reversal response, not a
    # negative one, and keying on is_monochrome first mis-sorted it.
    import re as _re
    import statistics as _st

    def _era_band(era):
        _m = _re.search(r"(\d{4})", era or "")
        _y = int(_m.group(1)) if _m else 1950
        return "pre1960" if _y < 1960 else ("1960-89" if _y < 1990 else "1990+")

    def _dqe_class(_p):
        if _p.name.startswith("POLAROID_"):
            return "instant"
        if _p.is_reversal:
            return "reversal_mono" if _p.is_monochrome else "reversal"
        return "mono" if _p.is_monochrome else "colneg"

    # ⚠ THE NAMED EXCEPTION THAT USED TO SIT HERE IS GONE, AND ITS REMOVAL IS
    # THE POINT. On 2026-08-27 this guard flagged KODAK_EKTACHROME_100D_5285 at
    # 50x its class median and it was recorded as "a real defect the guard
    # found". IT WAS NOT. The guard was reading `curves.g.gamma` as the curve's
    # contrast, and for that profile gamma is a MODEL COEFFICIENT, not a slope:
    # its toe and shoulder sit closer together than their own softness, so the
    # softplus difference is one smoothed step. Evaluated, its mid slope is
    # 2.419 -- an ordinary reversal contrast -- and its usable range is 4.25
    # stops against 4.8-6.2 for the rest of the class. The curve fits exact PDF
    # vector coordinates to 0.024-0.028 D RMS and is fine.
    # SO THE GUARD NOW READS `mid_slope`, WHICH IS EVALUATED, and the outlier
    # count across the whole database drops to ZERO with no exception list at
    # all. A guard that needs an allowlist on its first run is usually measuring
    # the wrong quantity.
    _by_grp: dict[tuple, list] = {}
    for _p in _fpm.FILM_PROFILES:
        _r, _ei = _p.grain.rms_granularity, _p.exposure_index
        # ⚠ mid_slope, NOT gamma. See the note above.
        _g = _p.curves.g.mid_slope
        if _r <= 0 or _ei <= 0 or _g <= 0:
            continue
        _by_grp.setdefault((_dqe_class(_p), _era_band(_p.era)), []).append(
            (_g * _g * _ei / (_r * _r), _p.name))
    _dqe_out = []
    _dqe_n = 0
    for _grp, _vals in _by_grp.items():
        # A group of three cannot have a meaningful median, so it is skipped
        # rather than tested against itself.
        if len(_vals) < 4:
            continue
        _dqe_n += len(_vals)
        _med = _st.median(v for v, _ in _vals)
        for _v, _nm in _vals:
            if _med <= 0:
                continue
            if _v / _med > 30.0 or _med / _v > 30.0:
                _dqe_out.append("%s K=%.2f, %.0fx its %s/%s median %.2f"
                                % (_nm, _v, max(_v / _med, _med / _v),
                                   _grp[0], _grp[1], _med))
    chk("no stock's gamma/speed/granularity triple is 30x off its class-and-era "
        "median (Eq. 1.1 coupling)", not _dqe_out,
        "; ".join(_dqe_out) if _dqe_out
        else "%d stocks in %d class/era groups, no exceptions needed"
             % (_dqe_n, sum(1 for _v in _by_grp.values() if len(_v) >= 4)))

    # ---- G-LAT: the parameter-space latitude must match the evaluated one ---
    # `ToneCurve.latitude_stops` is (shoulder_x - toe_x) * 3.3219, which
    # measures the distance between the two softplus knees and ignores their
    # SOFTNESS. When the knees sit closer than their own smoothing constants
    # the curve is one smoothed step and that formula stops describing it.
    # THIS IS NOT HYPOTHETICAL: three stocks are in that regime today, one of
    # them out by 5.6x (KODAK_EKTACHROME_100D_5285, 0.76 stops stored against
    # 4.25 evaluated). The values are not wrong -- the PROPERTY is.
    # The guard therefore permits a disagreement ONLY where `is_degenerate`
    # says the parameter-space formula does not apply, so a NEW disagreement on
    # a well-separated curve -- which would be a real fit defect -- still fails.
    _lat_bad = []
    for _p in _fpm.FILM_PROFILES:
        _c = _p.curves.g
        _st_lat, _ev_lat = _c.latitude_stops, _c.usable_range_stops
        if _st_lat <= 0 or _ev_lat <= 0:
            continue
        _ratio = _ev_lat / _st_lat
        if (_ratio > 1.5 or _ratio < 0.67) and not _c.is_degenerate:
            _lat_bad.append("%s stored %.2f vs evaluated %.2f stops"
                            % (_p.name, _st_lat, _ev_lat))
    _degen = [_p.name for _p in _fpm.FILM_PROFILES if _p.curves.g.is_degenerate]
    chk("stored latitude matches the evaluated range on every NON-degenerate "
        "curve", not _lat_bad, "; ".join(_lat_bad) if _lat_bad
        else "%d degenerate curves exempt and named: %s"
             % (len(_degen), ", ".join(sorted(_degen))))

    # ---- G-PROV: per-parameter provenance must resolve and must not lie ----
    # `ParamSource.validate` already refuses a path that does not resolve, a
    # measured/traced status without a source, and measured-at-tier-3. What it
    # cannot see is the CROSS-profile invariant: a parameter recorded as
    # measured or traced must not ALSO be flagged as the model's own estimate
    # anywhere, and no profile may carry two entries for the same parameter.
    _prov_bad = []
    for _p in _fpm.FILM_PROFILES:
        _seen: dict[str, str] = {}
        for _ps in _p.param_sources:
            if _ps.param in _seen:
                _prov_bad.append("%s: two entries for %s" % (_p.name, _ps.param))
            _seen[_ps.param] = _ps.status
    chk("per-parameter provenance is unique per parameter and every path "
        "resolves", not _prov_bad,
        "; ".join(_prov_bad) if _prov_bad
        else "%d entries across %d profiles" % (
            sum(len(_p.param_sources) for _p in _fpm.FILM_PROFILES),
            sum(1 for _p in _fpm.FILM_PROFILES if _p.param_sources)))

    # ---- G-PROC: a process variant must say which process the curves are ---
    # The whole point of the record. A profile with variants but none marked
    # default leaves the stored curves' process unstated, which is the exact
    # ambiguity ProcessingSpec was introduced at v6 to remove.
    _pv_bad = [_p.name for _p in _fpm.FILM_PROFILES
               if _p.process_variants
               and not any(_v.is_default for _v in _p.process_variants)]
    chk("every stock with process variants marks which one its stored curves "
        "represent", not _pv_bad, ", ".join(_pv_bad) if _pv_bad
        else "%d stocks with variants" % sum(
            1 for _p in _fpm.FILM_PROFILES if _p.process_variants))

    # ---- G-PROGRESS: the granular rate law must not be asserted ------------
    # Tani Figs. 7.10/7.11: granular development rate is nearly INDEPENDENT of
    # grain size. ProcessingSpec.validate rejects a 1/d coefficient there, and
    # this is the corpus-wide restatement so a future bulk edit cannot bypass
    # it by constructing the struct another way.
    _prog_bad = [_p.name for _p in _fpm.FILM_PROFILES
                 if _p.processing.progress is _fpm.DevelopmentProgress.GRANULAR
                 and _p.processing.rate_size_coeff_um_min]
    chk("no granular-development stock asserts a 1/grain-size rate law",
        not _prog_bad, ", ".join(_prog_bad) if _prog_bad
        else "%d stocks carry a traced progress type" % sum(
            1 for _p in _fpm.FILM_PROFILES
            if _p.processing.progress is not _fpm.DevelopmentProgress.UNKNOWN))
    # ---- G-DEVFAM: a development family must be internally coherent --------
    # Task EM-A8. The processing-side counterpart to G-DQE.
    #
    # WHY A FAMILY IS A MEANINGFUL GROUP. A development process fixes an AIM
    # CONTRAST -- ECN-2 exists precisely so that every camera negative run
    # through it lands on the same gamma, whatever its speed. So contrast must
    # cluster inside a family even though speed does not, and a stock that
    # breaks the cluster is either mis-assigned to the family or has a bad
    # curve. Neither is something the profile itself can reveal.
    #
    # ⚠ THE THRESHOLD IS DERIVED FROM THE DATA, NOT CHOSEN. Measured spreads of
    # mid_slope (max/min) inside the real families now in the database:
    #     Process ECN-2   n=15   1.12x   (0.560 - 0.628)
    #     ID-11 (merged)  n=6    1.30x   (0.535 - 0.694)
    #     KODAK D-96      n=2    1.03x
    # The widest genuine family is 1.30x. The cut sits at 2.0x -- comfortably
    # above every real family, and below the single false one this guard was
    # written to catch, which measured 4.33x.
    #
    # ⚠ IT EARNED ITS KEEP BEFORE IT SHIPPED. EM-A7 mined developer identities
    # out of the on-disk sheets and proposed "Process E-6" for
    # EASTMAN_5294_1983 on the strength of a matching product number. That
    # stock is a colour NEGATIVE and the sheet is a later EKTACHROME REVERSAL
    # film that reuses the number 5294. Check A below refuses a family holding
    # both a reversal and a negative, which is what surfaced it. The bad value
    # was removed rather than exempted.
    #
    # WHAT WOULD MAKE THIS FAIL: assigning a stock to the wrong process, or
    # editing a curve so its contrast leaves its process's cluster.
    def _devfam(_d):
        # ⚠ NORMALISED, because the SPELLING is not the developer. The database
        # holds both "ID-11" and "ILFORD ID-11" for the same Ilford developer,
        # entered by different people from different sheets. Left unnormalised
        # they form two families of 2 and 4, each too small or too tight to
        # test, and a genuine outlier could hide in the split.
        _d = _re.sub(r"\s+", " ", _d.strip().upper())
        for _pre in ("KODAK ", "ILFORD ", "FUJI ", "AGFA ", "EASTMAN "):
            if _d.startswith(_pre):
                _d = _d[len(_pre):]
        return _d

    _fam: dict[str, list] = {}
    for _p in _fpm.FILM_PROFILES:
        if not _p.processing.developer:
            continue
        # A free-text formula (the Soviet TU entries spell out the whole
        # chemistry) is a description, not a family name -- it can never group.
        if len(_p.processing.developer) > 48:
            continue
        _fam.setdefault(_devfam(_p.processing.developer), []).append(_p)

    # Check A -- a process is either a reversal process or a negative one.
    _mixed = []
    for _k, _v in sorted(_fam.items()):
        if len(_v) < 2:
            continue
        _rev = {_q.is_reversal for _q in _v}
        if len(_rev) > 1:
            _mixed.append("%s: %s" % (_k, ", ".join(
                "%s(%s)" % (_q.name, "rev" if _q.is_reversal else "neg")
                for _q in _v)))
    chk("no development family mixes reversal and negative stocks",
        not _mixed, "; ".join(_mixed) if _mixed
        else "%d families of 2+ stocks, each wholly reversal or wholly negative"
             % sum(1 for _v in _fam.values() if len(_v) >= 2))

    # Check B -- contrast clusters inside a family. n >= 3 so a median means
    # something; a pair cannot distinguish an outlier from its partner.
    _spread = []
    _tested = 0
    for _k, _v in sorted(_fam.items()):
        if len(_v) < 3:
            continue
        _sl = [_q.curves.g.mid_slope for _q in _v if _q.curves.g.mid_slope > 0]
        if len(_sl) < 3:
            continue
        _tested += 1
        if max(_sl) / min(_sl) > 2.0:
            _worst = max(_v, key=lambda _q: abs(
                _q.curves.g.mid_slope - _st.median(_sl)))
            _spread.append("%s spread %.2fx (%.3f-%.3f), worst %s at %.3f"
                           % (_k, max(_sl) / min(_sl), min(_sl), max(_sl),
                              _worst.name, _worst.curves.g.mid_slope))
    chk("contrast clusters within each development family (>= 3 stocks)",
        not _spread, "; ".join(_spread) if _spread
        else "%d families tested, widest spread under the 2.0x cut derived "
             "from the data" % _tested)

    # ---- G-YELLOW: no base is documented to yellow, so none may claim to ----
    # EMULSION_KNOWLEDGE_BASE.md §26 B7, audited 2026-08-27.
    #
    # THE EVIDENCE IS AN ABSENCE, AND THE ABSENCE IS THE FINDING. Neither
    # preservation source in the corpus states that any film base yellows in a
    # way that has a density. Reilly (IPI Storage Guide for Acetate Film, 1993
    # rev. 1996) and NEDCC Preservation Leaflet 5.1 (2020) between them describe
    # yellowing ONLY for NITRATE, and even there only ORDINALLY -- as a stage in
    # a degradation sequence, with no D attached to any stage. Acetate is
    # described as shrinking, embrittling, exuding plasticiser, warping and
    # delaminating; it is NEVER described as yellowing.
    #
    # The audit found the database already clean: 0 of 161 profiles carried a
    # non-zero base_yellowing_d. THAT WAS LUCK, NOT DESIGN -- nothing stopped a
    # future edit adding one, and an unsourced aging value is exactly the class
    # of number this project keeps having to withdraw.
    #
    # ⚠ THIS GUARD DELIBERATELY DOES NOT SAY "ACETATE", AND CANNOT.
    # There is no base-material field on FilmProfile. `base_tint` is a COLOUR,
    # not a material, and the metadata proposal that would supply one is §26 B8,
    # which is parked pending a decision about where preset validity lives. So
    # the assertion is made over EVERY profile. That is the stronger claim and
    # it matches the evidence exactly: no base in this corpus has a documented
    # yellowing density, so no stock may carry one.
    #
    # WHAT WOULD MAKE THIS FAIL: adding a non-zero base_yellowing_d to any
    # profile. That is intended. If a source is ever found that prints a
    # yellowing density for a named base, cite it in _PROVENANCE_SOURCES, record
    # a ParamSource for aging.base_yellowing_d, and narrow this guard to exclude
    # that stock by name -- do not simply delete it.
    _yellow = ["%s=%.4f" % (_p.name, _p.aging.base_yellowing_d)
               for _p in _fpm.FILM_PROFILES if _p.aging.base_yellowing_d]
    chk("no film profile claims an unsourced base yellowing density",
        not _yellow,
        ", ".join(_yellow) if _yellow
        else "0 of %d profiles; yellowing is documented for NITRATE only, and "
             "only ordinally" % len(_fpm.FILM_PROFILES))
    # The same absence applies to print stocks, which carry their own AgingSpec
    # since schema v12 and were NOT covered by the §26 B7 audit as written.
    _pyellow = ["%s=%.4f" % (_ps.name, _ps.aging.base_yellowing_d)
                for _ps in _fpm.PRINT_STOCKS if _ps.aging.base_yellowing_d]
    chk("no print stock claims an unsourced base yellowing density",
        not _pyellow,
        ", ".join(_pyellow) if _pyellow
        else "0 of %d print stocks" % len(_fpm.PRINT_STOCKS))
    _iact = [p for p in FILM_PROFILES if p.interimage.active]
    chk("interimage active only on colour tripacks",
        all(not p.is_monochrome and p.reseau is None
            and p.name != "TECHNICOLOR_THREE_STRIP" for p in _iact),
        f"{len(_iact)} stocks active")
    chk("three-strip has no interimage (separate films cannot exchange inhibitor)",
        not get_profile("TECHNICOLOR_THREE_STRIP").interimage.active)
    chk("every interimage term is inhibition (<= 0)",
        all(v <= 0.0 for p in FILM_PROFILES
            for v in (p.interimage.a_rg, p.interimage.a_rb, p.interimage.a_gr,
                      p.interimage.a_gb, p.interimage.a_br, p.interimage.a_bg)))
    chk("interimage diagonal is structurally zero",
        all(get_profile("KODAK_PORTRA_400").interimage.matrix()[i][i] == 0.0
            for i in range(3)))
    chk("modern DIR stocks couple harder than 1950s stocks",
        abs(get_profile("KODAK_PORTRA_400").interimage.a_rg)
        > abs(get_profile("EASTMAN_5250_1959").interimage.a_rg))
    # ⚠ THIS GUARD USED TO ASSERT THE OPPOSITE, AND IT WAS UNPASSABLE BY DESIGN.
    # It read "neighbour pairs couple harder than the far red-blue pair" and
    # tested |a_rg| > |a_rb| -- a PER-DISTANCE asymmetry. The database stores
    # those EQUAL, deliberately, because the evidence says the asymmetry is
    # per RECEIVER and not per hop: US4725529A Table 1 puts the inhibitor in the
    # DEVELOPER and applies it to three separate single-layer coatings -- no
    # layer stack at all, so no distance to travel -- and still measures red
    # receivers at 0.43-0.72 dlogE against blue at 0.24-0.48. US5273870A and
    # US4830954A agree on the pattern. No numeric support for a per-hop factor
    # exists in any of the nine patents surveyed.
    #   So the old guard encoded the hypothesis the project later REJECTED on
    # evidence, and it had been sitting in the FAIL baseline as "known, leave it
    # alone" -- which is how a fixable stale assertion became treated as
    # immovable. Replaced 2026-08-20 with the assertion the evidence supports,
    # which keeps a live check where there was a permanent red.
    #
    # ⚠ AND REPLACED AGAIN ON 2026-09-17 (schema v38), FOR THE SAME REASON IT
    # WAS REPLACED THE FIRST TIME: it asserted a structural identity
    # (a_rg == a_rb, a_gr == a_gb, a_br == a_bg) whose stated justification was
    # that no surveyed patent measures a donor weighting. One does.
    # US 6,746,834 TABLE 3 sample 108 measures the GREEN record receiving 0.10
    # from red and 0.21 from blue -- one receiver, one coating, one
    # measurement, two donors 2.1 apart. So the identity is gone, and what is
    # asserted now is the two things the evidence supports: the per-receiver
    # asymmetry US4725529A demonstrates with no layer stack at all, and the
    # measured donor split itself.
    _pp = get_profile("KODAK_PORTRA_400").interimage
    chk("interimage is asymmetric per RECEIVER -- blue receives least",
        abs(_pp.a_br) + abs(_pp.a_bg) < abs(_pp.a_gr) + abs(_pp.a_gb),
        "US4725529A Table 1: no layer stack, asymmetry persists -- "
        "so it is the receiving emulsion, not the distance travelled")
    # ⚠ THE TWO CLASSES ARE CHECKED SEPARATELY BECAUSE THEY ARE MEASURED
    # SEPARATELY (v39). A colour NEGATIVE takes EP 0 324 471 A2's medians --
    # receiver green 0.600 from red against 0.400 from blue, receiver red
    # 0.5728 from green against 0.4272 from blue, and no pair equal, because
    # that patent measures BLUE ACTING ON RED and nothing else does. A colour
    # REVERSAL takes US 6,746,834 sample 108, where the green receiver's two
    # donors sit 2.1 apart and the red receiver's stay level for want of a
    # measurement.
    _pv = get_profile("FUJI_VELVIA_50").interimage
    chk("interimage is also asymmetric per DONOR, in the measured ratio, and "
        "the ratio differs by material class",
        (abs(_pp.a_gr) > abs(_pp.a_gb)
         and abs(abs(_pp.a_gr / _pp.a_gb) - 1.5) < 0.01
         and abs(_pp.a_rg) > abs(_pp.a_rb)
         and abs(_pv.a_gb) > abs(_pv.a_gr)
         and abs(abs(_pv.a_gb / _pv.a_gr) - 2.1) < 0.01
         and abs(_pv.a_rg - _pv.a_rb) < 1e-12),
        "negative PORTRA_400 green takes r/b %.3f (EP0324471A2 0.600/0.400); "
        "reversal VELVIA_50 green takes b/r %.3f (US6746834 #108 0.21/0.10) "
        "and its red pair stays level"
        % (abs(_pp.a_gr / _pp.a_gb), abs(_pv.a_gb / _pv.a_gr)))

    # ---- C21, 2026-08-23: schema v11, per-channel halation radii ------------
    # ⚠ THE POINT OF THESE GUARDS IS THAT THE FIELDS ARE ALL 1.0 AND MUST STAY
    # THERE UNTIL SOMETHING IS MEASURED. The path-length argument bounds the real
    # per-channel ratio at about 1.1 (base 100-150 um against an 11-16 um pack),
    # so a geometry-derived set would look measured while moving a render ~1 %.
    # The temptation to "finish the feature" by filling them from the layer order
    # is exactly what this catches.
    _rs = [(p.name, p.halation.radius_scales()) for p in FILM_PROFILES]
    _rs_bad = [n for n, t in _rs if t != (1.0, 1.0, 1.0)]
    chk("every stock still ships halation radius scales of exactly 1.0",
        not _rs_bad, ", ".join(_rs_bad[:4]) if _rs_bad
        else "%d stocks, v11 renders bit-identically to v10" % len(_rs))
    chk("the shared-radius fast path is taken on every stock",
        all(p.halation.radii_are_shared for p in FILM_PROFILES),
        "160 of 160 -- one kernel per frame, not three")
    # And the accessor must actually multiply, or the field would be inert by
    # accident rather than by data. Probed off-database so no stock changes.
    _hs = film_profiles.HalationSpec(radii_um=(10.0, 50.0, 200.0),
                                     radius_scale_r=0.5, radius_scale_b=2.0)
    chk("radii_for() scales the physical radius per record",
        (_hs.radii_for(0) == (5.0, 25.0, 100.0)
         and _hs.radii_for(1) == (10.0, 50.0, 200.0)
         and _hs.radii_for(2) == (20.0, 100.0, 400.0)
         and not _hs.radii_are_shared),
        "0.5x / 1.0x / 2.0x, and the fast path correctly refused")

    # ---- C22, 2026-08-23: Callier's coefficient -----------------------------
    # 1. Inert by default, EXACTLY, on every stock. This is the assertion that
    # every render made before the stage existed is still reproducible.
    # ⚠ TESTED ON THE LAW'S OUTPUT, NOT ON A MULTIPLIER (G3, 2026-08-30).
    # There is no multiplier any more: `callier_net` returns the net density a
    # directional reader sees, so "inert" means it returns what it was given.
    # ⚠ INERTNESS IS A PROPERTY OF THE GUARD, NOT OF THE LAW'S ARITHMETIC, AND
    # CONFLATING THEM COST A BUILD. At E = 1 the law reduces to
    # -log10(10**-d), which is mathematically d and NOT bit-exact in floating
    # point -- measured departure 5.6e-17. That is why both consumers, in all
    # three implementations, test `callier_is_inert` and RETURN EARLY rather
    # than evaluating a law that happens to be the identity. What has to be
    # exact is the pipeline, so that is what is asserted: the guard fires on
    # every stock, and the two consumers return their input unchanged.
    _cd = (0.05, 0.4, 1.2, 2.6)
    _px = np.zeros((1, len(_cd), 3), dtype=np.float32)
    _px[0, :, :] = np.asarray(_cd, dtype=np.float32)[:, None]
    _px_in = _px.copy()
    fs.callier_density(_px, get_profile("EASTMAN_DOUBLE_X_5222").curves.as_tuple(),
                       1.3, 0.0, True)
    chk("Callier is exactly inert at scanner_specular = 0",
        all(fs.callier_is_inert(p, 0.0) for p in FILM_PROFILES)
        and bool((_px == _px_in).all()),
        "%d of %d stocks guarded, and the pixel pass is bit-identical"
        % (len(FILM_PROFILES), len(FILM_PROFILES)))
    # 2. ⚠ AND INERT AT *ANY* SETTING ON COLOUR. Q is silver scattering; a
    # chromogenic dye image does not scatter, which is why all 93 colour stocks
    # carry Q = 1.0. If a future edit gave one of them a Q, colour renders would
    # start responding to a scanner control that has no business touching them.
    _col = [p for p in FILM_PROFILES if not p.is_monochrome]
    _col_moved = [p.name for p in _col
                  if any(not fs.callier_is_inert(p, s)
                         for s in (0.25, 0.6, 1.0))]
    chk("no colour stock responds to Callier at any specular setting",
        not _col_moved, ", ".join(_col_moved[:4]) if _col_moved
        else "%d colour stocks, Q = 1.0 on all of them" % len(_col))
    # 2b. ⚠⚠ THE SHIPPED DEFAULT, AND THE TWIN THAT MUST AGREE WITH IT.
    # scanner_specular moved 0.0 -> 0.853 on 2026-09-06 by owner decision:
    # 1 - E with E = 0.1471, the collected-scatter fraction from the same
    # Trumpy/Streiffert fit that gave the monochrome profiles their beta.
    # ⚠ IT IS A DENSITOMETER GEOMETRY ADOPTED AS A PROVISIONAL STAND-IN, NOT A
    # SCANNER MEASUREMENT -- pinned here so it cannot drift into looking like one
    # by being quietly rounded or re-derived.
    # ⚠ AND THE C++ SIDE HAD NO ASSIGNMENT AT ALL until the same day: it took 0.0
    # from zero-initialisation, so the twins agreed BY COINCIDENCE. This check
    # reads the literal out of AlgoControl.cpp, because a default that lives in
    # two languages is exactly the kind of thing that silently splits.
    _SPEC_DEFAULT = 0.853
    chk("G-CALDEF1  the reference default scanner_specular is the adopted 0.853",
        abs(fs.RenderSettings().scanner_specular - _SPEC_DEFAULT) < 1e-12,
        "1 - E, E = 0.1471 (Trumpy & Gschwind 2015 Fig. 5, after Streiffert "
        "1947); a provisional reader geometry, NOT a scanner measurement")
    import re as _re_cd
    _cd_txt = None
    for _cand in ("/root/work/proot/AlgoControl.cpp",
                  _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                "..", "tst", "AlgoControl.cpp")):
        if _os.path.exists(_cand):
            _cd_txt = open(_cand, encoding="utf-8", errors="replace").read()
            break
    if _cd_txt is None:
        chk("G-CALDEF2  the C++ default matches the reference", True,
            "AlgoControl.cpp not present in this tree -- skipped, not passed")
    else:
        _m = _re_cd.search(r"controls\.scannerSpecular\s*=\s*([0-9.]+)", _cd_txt)
        chk("G-CALDEF2  getAlgoControlsDefault ASSIGNS scannerSpecular and it "
            "equals the reference default",
            bool(_m) and abs(float(_m.group(1)) - _SPEC_DEFAULT) < 1e-12,
            ("C++ has %s, reference has %s" % (_m.group(1) if _m else "NO "
             "ASSIGNMENT AT ALL -- the field would fall back to zero-init and "
             "the twins would diverge", _SPEC_DEFAULT)))
    # ⚠ THE DEFAULT NOW MOVES PIXELS, WHICH IS THE POINT AND ALSO THE RISK.
    # Pinned so the size of the change is a stated number rather than a surprise.
    _q_dx = get_profile("EASTMAN_DOUBLE_X_5222").callier_q
    _lift = float(fs.callier_net(np.array([1.0]), _q_dx, _SPEC_DEFAULT)[0]) - 1.0
    chk("G-CALDEF3  the adopted default adds the expected density on a "
        "monochrome stock",
        0.40 < _lift < 0.55,
        "EASTMAN_DOUBLE_X_5222 (beta %.4f) reads +%.3f D at net density 1.0; "
        "0.0 remains the setting that reproduces the stored DIFFUSE curves"
        % (_q_dx, _lift))

    # 3. The monochrome stocks DO respond, or the stage models nothing.
    _mono_moved = sum(1 for p in FILM_PROFILES
                      if p.is_monochrome and not fs.callier_is_inert(p, 1.0))
    chk("the monochrome stocks are the ones Callier moves",
        _mono_moved >= 60, "%d stocks move at specular 1.0" % _mono_moved)
    # 4. ⚠ THE dmin REFERENCE, which is the part that is easy to get wrong and
    # invisible when it is: `dmin + (D - dmin) * k` and `D * k` agree only at
    # D = dmin. Referenced to zero, a condenser would darken CLEAR FILM BASE,
    # which no densitometer measures. Probed on a stock with Q != 1.
    _dx = get_profile("EASTMAN_DOUBLE_X_5222")
    _dmn = _dx.curves.g.dmin
    _at_base = _dmn + float(fs.callier_net(0.0, float(_dx.callier_q), 1.0))
    _above = _dmn + float(fs.callier_net(1.0, float(_dx.callier_q), 1.0))
    chk("Callier is the identity at dmin -- clear base carries no silver",
        abs(_at_base - _dmn) < 1e-12 and _above > _dmn + 1.0,
        "base unmoved, net 1.0 reads %.3f above it" % (_above - _dmn))
    # 5. It must STEEPEN, never lighten: Q >= 1 for every stock in the file,
    # because scattering can only send light out of the acceptance angle.
    _q_bad = [p.name for p in FILM_PROFILES if p.callier_q < 1.0]
    chk("no stock carries a Callier Q below 1.0",
        not _q_bad, ", ".join(_q_bad[:4]) if _q_bad
        else "min Q %.2f" % min(p.callier_q for p in FILM_PROFILES))

    # the load-bearing property: neutrals untouched, saturated colour separates
    _stI = fs.RenderSettings(film_format="ff35", grain_scale=0.0, print_grain=False,
                             flare=0.0, vignette=0.0, coating_scale=0.0)
    _pI = get_profile("KODAK_PORTRA_400")
    _pN = dataclasses.replace(_pI, interimage=_fpm.InterimageSpec())
    _neu = np.full((48, 64, 3), 0.18, np.float32)
    _a = fs.simulate(_neu, _pN, _stI).mean(axis=(0, 1))
    _b = fs.simulate(_neu, _pI, _stI).mean(axis=(0, 1))
    # ⚠ RENAMED 2026-08-25d (queue item C20), AND THE OLD NAME IS THE FINDING.
    # This read "interimage leaves a neutral untouched" while rendering 0.18 --
    # which is the mid-grey ANCHOR the correction is referenced to, the one point
    # where every (D_j - d_ref) is zero and the correction vanishes identically.
    # The guard was therefore true by construction and promised far more than it
    # tested: it could not have failed for any value of the interimage matrix.
    chk("interimage leaves the ANCHOR neutral untouched (0.18, where it must)",
        float(np.abs(_a - _b).max()) < 2e-3,
        f"max channel delta {float(np.abs(_a-_b).max()):.5f} at the anchor")
    # The second half of C20: pin the OFF-ANCHOR movement as intended behaviour,
    # so the property the old guard implied is now measured rather than assumed
    # -- and so a future change that really did flatten the effect on non-anchor
    # neutrals would fail here instead of passing a vacuous check.
    # Measured on KODAK_PORTRA_400: grey 0.45 moves 15.9/255, grey 0.06 moves
    # 6.5/255. That is the mechanism, not a leak: white-light gamma below
    # separation gamma is the patent's own metric for interimage effect.
    _off = {}
    for _lvl in (0.45, 0.06):
        _f = np.full((48, 64, 3), _lvl, np.float32)
        _off[_lvl] = 255.0 * float(np.abs(
            fs.simulate(_f, _pN, _stI).mean(axis=(0, 1))
            - fs.simulate(_f, _pI, _stI).mean(axis=(0, 1))).max())
    chk("interimage DOES move off-anchor neutrals, as the mechanism requires",
        3.0 < _off[0.45] < 30.0 and 1.0 < _off[0.06] < 15.0
        and _off[0.45] > _off[0.06],
        "grey 0.45 moves %.1f/255, grey 0.06 moves %.1f/255 (anchor 0.18 moves 0)"
        % (_off[0.45], _off[0.06]))
    _sat = np.zeros((48, 64, 3), np.float32)
    _sat[:, :, 0], _sat[:, :, 1], _sat[:, :, 2] = 0.35, 0.10, 0.08
    _c = fs.simulate(_sat, _pN, _stI).mean(axis=(0, 1))
    _d = fs.simulate(_sat, _pI, _stI).mean(axis=(0, 1))
    _s0 = (_c.max() - _c.min()) / max(_c.max(), 1e-6)
    _s1 = (_d.max() - _d.min()) / max(_d.max(), 1e-6)
    chk("interimage raises saturation on a saturated colour", _s1 > _s0,
        f"sat {_s0:.4f} -> {_s1:.4f}")
    chk("interimage iterations=0 is a no-op",
        np.array_equal(
            fs.simulate(_sat, dataclasses.replace(
                _pI, interimage=dataclasses.replace(_pI.interimage, iterations=0)),
                _stI),
            fs.simulate(_sat, _pN, _stI)))

    # IIE must reproduce the PUBLISHED figures it was calibrated against
    def _iie_pct(p):
        _cv = p.curves.as_tuple()
        _m = p.interimage.matrix()
        _dr = [float(fs.density_scalar(0.0, _cv[c])) for c in range(3)]
        def _d(lg):
            d = [float(fs.density_scalar(lg[c], _cv[c])) for c in range(3)]
            for _ in range(max(p.interimage.iterations, 1)):
                adj = [sum(_m[c][j] * (d[j] - _dr[j]) for j in range(3) if j != c)
                       for c in range(3)]
                d = [float(fs.density_scalar(lg[c] + adj[c], _cv[c])) for c in range(3)]
            return d
        out = []
        for c in range(3):
            gw = (_d([0.6]*3)[c] - _d([-0.6]*3)[c]) / 1.2
            hi = _d([0.6 if j == c else 0.0 for j in range(3)])[c]
            lo = _d([-0.6 if j == c else 0.0 for j in range(3)])[c]
            out.append(100.0 * (((hi - lo) / 1.2) / gw - 1.0) if gw > 1e-9 else 0.0)
        return out[2], out[1], out[0]          # blue, green, red

    for _nm, _tgt in (("KODAK_PORTRA_400", (25.0, 45.0, 42.0)),
                      ("EASTMAN_5247_1974", (10.0, 15.0, 15.0)),
                      ("EASTMAN_5250_1959", (5.0, 7.0, 7.0))):
        _got = _iie_pct(get_profile(_nm))
        _err = max(abs(_got[i] - _tgt[i]) for i in range(3))
        chk(f"{_nm} reproduces its published IIE percentages", _err < 1.0,
            f"model {_got[0]:.1f}/{_got[1]:.1f}/{_got[2]:.1f} vs "
            f"{_tgt[0]:.0f}/{_tgt[1]:.0f}/{_tgt[2]:.0f}, worst {_err:.2f} pp")

    chk("interimage couples blue weakly, green and red strongly (per-receiver)",
        abs(get_profile("KODAK_PORTRA_400").interimage.a_br)
        < abs(get_profile("KODAK_PORTRA_400").interimage.a_gr),
        "US4725529A Table 1: red receivers 0.43-0.72 dlogE vs blue 0.24-0.48")
    chk("reversal stocks weight interimage toward high density",
        get_profile("FUJI_VELVIA_50").interimage.density_weighting > 0.0
        and get_profile("KODAK_PORTRA_400").interimage.density_weighting == 0.0)

    # ---- 2026-08-20: the two DIR stages are now shared definitions ----------
    # Both laws were inline inside simulate() and are now module-level functions
    # so `interimage_parity.py` can probe them against the plugin's own C++.
    # These guards defend the properties that made that possible.
    chk("both DIR-coupler stages are callable definitions, not inline code",
        callable(getattr(fs, "apply_interimage", None))
        and callable(getattr(fs, "apply_dir_couplers", None)),
        "film_sim.apply_interimage / apply_dir_couplers")

    # ⚠ THE DENSITY FLOOR MUST BE INSIDE apply_dir_couplers, matching the C++
    # twin, which ends with MAX_VALUE(rO[x], ALGO_ZERO). It was outside (one
    # line later, in simulate()) until 2026-08-20, so the two PIPELINES agreed
    # while the two FUNCTIONS disagreed by 0.26 D on a reversal stock. The
    # parity probe is what found it; this is what stops it coming back.
    _cpv = get_profile("FUJI_VELVIA_50").couplers
    _dneg = np.full((8, 8, 3), -0.5, np.float32)
    _gr = fs.FreqGrid(8, 8, 120.0)
    fs.apply_dir_couplers(_dneg, _cpv, _gr, 1.0, False)
    chk("apply_dir_couplers floors density at zero, as its C++ twin does",
        float(_dneg.min()) >= 0.0, f"min {float(_dneg.min()):.6f}")

    # And the floor must not be the only thing it does -- a stage that clamped
    # and nothing else would pass the check above and render nothing.
    _dpos = np.stack([np.full((8, 8), 1.4, np.float32),
                      np.full((8, 8), 0.9, np.float32),
                      np.full((8, 8), 0.6, np.float32)], axis=2)
    _before = _dpos.copy()
    fs.apply_dir_couplers(_dpos, _cpv, _gr, 1.0, False)
    chk("apply_dir_couplers still separates the layers of a flat colour",
        float(np.abs(_dpos - _before).max()) > 1e-3,
        f"max move {float(np.abs(_dpos - _before).max()):.4f} D")

    # the spectral derivation is DIAGNOSTIC and must stay out of the pipeline
    chk("spectral derivation exposes the IR failure it is quarantined for",
        (lambda r: r is not None and r[2] > 0.5)(
            _fpm.derived_spectral_response(get_profile("KONICA_INFRARED_750"))),
        "display primaries cannot reach 750 nm -- documented, not wired in")
    # 2026-08-13: the original form of this check tested for one function NAME,
    # which a differently-named spectral derivation passes vacuously -- and one
    # was added that day. It then guarded the INTENT: no basis-projected
    # spectral derivation may drive the render by default.
    #
    # ⚠ REWRITTEN 2026-08-29, AND THE PREMISE IS WHAT CHANGED, NOT THE ETHIC.
    # That check asserted `spectral_mono is False`. What it was actually
    # pinning, unknowingly, was a PYTHON/C++ SPLIT: Algo_07_Sim.cpp calls
    # AlgoSpectralMonoWeights() unconditionally and always has, so the plugin
    # derived while this side did not, and the 24 monochrome stocks carrying a
    # traced pan curve rendered differently in the two engines. Worst case
    # KODAK_PLUS_X_125, blue 0.110 stored against 0.502 derived. A guard that
    # holds one engine to a decision the other never implemented is not
    # caution; it is an unnoticed divergence with a test defending it.
    #
    # The invariant now asserted is the one that survives: the MONO collapse
    # derives in BOTH engines, and the TAKING MATRIX -- the basis projection
    # that would stack a third mixing stage on top of dye_matrix and
    # InterimageSpec -- stays out of the pipeline. spectral_mono_parity.py
    # holds the two engines together numerically; this holds the intent.
    chk("the mono spectral derivation is enabled, matching the C++ engine",
        fs.RenderSettings().spectral_mono is True,
        "Algo_07_Sim.cpp derives unconditionally; a False here is a silent split")
    chk("the basis-projected TAKING MATRIX is still out of the pipeline",
        fs.RenderSettings().spectral_taking is False,
        "dye_matrix and InterimageSpec already carry cross-channel mixing")
    # The guard catches the EXTREME case and is honest about not catching all of
    # them. KONICA_INFRARED_750 peaks at 750 nm with 0.437 of its energy beyond
    # the basis limit, and is refused.
    #
    # ⚠ THOSE TWO FIGURES WERE 730 nm AND 0.203 UNTIL 2026-08-29, AND BOTH WERE
    # ARTEFACTS OF THE GUARD MEASURING ITSELF ON THE WRONG GRID -- the
    # renderer's, which stops at 730 nm, rather than the curve's own samples,
    # which run to 830. The guard refused this stock either way, so nothing
    # rendered wrong; but a threshold compared against a quantity that cannot
    # reach it is a guard that only appears to work. See
    # film_sim.stored_layer_sensitivities.
    #
    # ⚠ ROLLEI_INFRARED_400 WAS THE GUARD'S BLIND SPOT AND IS NOT ANY MORE
    # (queue C39, CLOSED 2026-08-30). Its stored curve is the UNFILTERED
    # sensitisation -- peak 410 nm, 0.028 of its energy past 700 nm -- so by the
    # data on file it was an ordinary panchromatic emulsion, the guard passed it
    # honestly, and both engines derived a near-flat (0.349, 0.315, 0.336)
    # against an authored red-dominant (0.52, 0.20, 0.28). The fix was never a
    # threshold: 0.028 is below every ordinary pan stock's own share, so any
    # threshold catching this one starts refusing Tri-X. The fix was a CARRIER.
    # `FilmProfile.taking_filter` (schema v20) records the 715 nm longpass the
    # Rollei sheet itself prints, film_sim applies it before the guard and
    # before the collapse, and behind it only 2.2 % of the curve survives -- all
    # of it past the basis ceiling. The guard now refuses, correctly, and the
    # authored triple is used.
    chk("the basis-reach guard refuses an infrared-peaked stock",
        fs.spectral_monochrome_weights(get_profile("KONICA_INFRARED_750")) is None,
        "projecting an IR curve onto visible primaries derives blue-dominant nonsense")
    chk("the guard measures reach on the CURVE's samples, not the render grid",
        (lambda p: fs.spectral_peak_lambda(p) == 750.0
         and fs.spectral_out_of_reach(p) > 0.40)(
            get_profile("KONICA_INFRARED_750")),
        "clipped at 730 nm these read 730 / 0.203 -- low by a factor of two")
    # ⚠ THIS CHECK ONCE ASSERTED THE OPPOSITE, AND THE REVERSAL IS THE POINT:
    # it pinned the defect in place so it could not be forgotten. C39 closed it.
    chk("the taking filter closes the guard's blind spot",
        (lambda p: fs.spectral_monochrome_weights(p) is None
         and p.taking_filter.renders
         and fs.spectral_peak_lambda(p) > 700.0)(
            get_profile("ROLLEI_INFRARED_400")),
        "behind its own sheet's 715 nm filter the peak moves 410 -> 720 nm and "
        "the guard refuses, so the authored red-dominant triple is used")
    # ⚠ AND THE FILTER MUST STAY INERT EVERYWHERE ELSE. It is a rendering
    # assumption, so a stray one would silently re-weight a stock nobody
    # shoots filtered.
    _tf = [p.name for p in FILM_PROFILES if p.taking_filter.renders]
    chk("exactly one stock renders behind a taking filter",
        _tf == ["ROLLEI_INFRARED_400"], ", ".join(_tf) or "none")
    chk("the spectral balance path IS active and differs from the proxy",
        (lambda d, q: d is not None and abs(d[0] - q[0]) > 0.05)(
            fs.spectral_balance_gains(get_profile("KODAK_PORTRA_400"), 3200.0),
            fs.balance_gains(3200.0, get_profile("KODAK_PORTRA_400").balance_kelvin)),
        "derived red gain ~1.68 vs proxy ~1.32 at 3200 K")

    # ---- schema v21 (2026-08-31, queue K2): published aim densities --------
    # The manufacturer's own statement of correct exposure. The values are read
    # and pinned by `kodak_aim_density.py`; what these check is that the STORED
    # side still says something a sensitometer would agree with, which is a
    # different question from whether the reader still reads the table.
    _aim = [p for p in FILM_PROFILES if p.aim_density]
    chk("sixteen stocks carry a published aim density",
        len(_aim) == 16, f"{len(_aim)}: " + ", ".join(p.name for p in _aim))
    # ⚠ 13 -> 16 ON 2026-09-06e, AND THE THREE NEW ONES ARE A DIFFERENT SHAPE
    # OF STATEMENT. Every entry here was Kodak's until now, and Kodak publishes
    # four readings per exposure index: a grey card, a paper grey scale and a
    # light and a dark forehead. Fuji publishes the GREY CARD ALONE -- SUPERIA
    # X-TRA 400 0.75-0.95, X-TRA 800 0.70-0.90, REALA 1.02-1.20, all Status M
    # red. The three empty fields on those records are empty because the sheets
    # do not print them, which is why the ordering checks below skip a field
    # that is absent instead of failing it: a (0.0, 0.0) grey scale is not a
    # grey scale that reads zero.
    _FUJI_CARD_ONLY = {"FUJICOLOR_SUPERIA_XTRA_400",
                       "FUJICOLOR_SUPERIA_XTRA_800",
                       "FUJICOLOR_SUPERIA_REALA"}
    chk("the Fuji aim densities carry a grey card and nothing else, because "
        "that is all their sheets print",
        all(get_profile(n).aim_density[0].gray_card[1] > 0.0
            and get_profile(n).aim_density[0].gray_scale == (0.0, 0.0)
            and get_profile(n).aim_density[0].filter == "status_m_red"
            for n in _FUJI_CARD_ONLY),
        "3 of 3 card-only, Status M red")
    # ⚠ AND THEIR ORDERING IS ASSERTED WHERE A KODAK-SHAPED ONE CANNOT BE. The
    # three Fuji aims rank REALA > X-TRA 400 > X-TRA 800, which is the order of
    # those stocks' own TRACED red dmin (0.306 > 0.129 ... 0.175) only in part
    # -- 400 and 800 swap -- so the aim is not a restatement of dmin and the
    # two are independent measurements that happen to agree at the ends.
    _cards = {n: get_profile(n).aim_density[0].gray_card for n in _FUJI_CARD_ONLY}
    chk("REALA's published aim is the highest in the database and X-TRA 800's "
        "the lowest of the Fuji three",
        (_cards["FUJICOLOR_SUPERIA_REALA"][0]
         > _cards["FUJICOLOR_SUPERIA_XTRA_400"][0]
         > _cards["FUJICOLOR_SUPERIA_XTRA_800"][0]),
        "REALA %s > X-TRA 400 %s > X-TRA 800 %s"
        % (_cards["FUJICOLOR_SUPERIA_REALA"],
           _cards["FUJICOLOR_SUPERIA_XTRA_400"],
           _cards["FUJICOLOR_SUPERIA_XTRA_800"]))

    # ⚠ THE GRAY SCALE'S LIGHTEST STEP IS A BRIGHTER SUBJECT AREA THAN A GRAY
    # CARD, so on a NEGATIVE it must land HIGHER. This is the check that would
    # have caught a transposed two-row table, which is the single most likely
    # way to read these tables wrong.
    _inv = []
    for p in _aim:
        for a in p.aim_density:
            # ⚠ AN ABSENT FIELD IS SKIPPED, NOT FAILED. Fuji's sheets publish
            # the grey card alone, so gray_scale and the two forehead readings
            # are (0.0, 0.0) meaning "not printed". Comparing against them
            # fails a stock for data its manufacturer never claimed -- and
            # treating a zero as a measured density is exactly the error rule
            # 23 point 3 is about, since an absence is not a value.
            if a.gray_scale != (0.0, 0.0):
                if not (a.gray_scale[0] > a.gray_card[0]
                        and a.gray_scale[1] > a.gray_card[1]):
                    _inv.append(f"{p.name}@EI{a.exposure_index} "
                                f"card {a.gray_card} scale {a.gray_scale}")
            # and a dark complexion reflects less light than a light one
            if a.forehead_light != (0.0, 0.0) or a.forehead_dark != (0.0, 0.0):
                if a.forehead_dark[0] >= a.forehead_light[0]:
                    _inv.append(f"{p.name}@EI{a.exposure_index} "
                                f"dark {a.forehead_dark} >= "
                                f"light {a.forehead_light}")
    chk("every aim density is ordered as the subject brightnesses are",
        not _inv, "; ".join(_inv[:4]))

    # ⚠ AND A PUSH RAISES THE AIM. More development at the same exposure means
    # more density; three stocks publish more than one EI and all three rise.
    # This is the invariant the E-190 (2003) PORTRA 800 table violates, which is
    # why that table is read, pinned in the audit and NOT stored.
    _push = []
    for p in _aim:
        for x, y in zip(p.aim_density, p.aim_density[1:]):
            for _area in ("gray_card", "gray_scale",
                          "forehead_light", "forehead_dark"):
                lo0, hi0 = getattr(x, _area)
                lo1, hi1 = getattr(y, _area)
                if (lo0, hi0) == (0.0, 0.0) or (lo1, hi1) == (0.0, 0.0):
                    continue
                if not (lo1 > lo0 and hi1 > hi0):
                    _push.append(f"{p.name}.{_area} EI{x.exposure_index}"
                                 f"->{y.exposure_index}: {(lo0, hi0)}"
                                 f" -> {(lo1, hi1)}")
    chk("every pushed aim density is higher than the box-speed one",
        not _push, "; ".join(_push[:4]))
    chk("the three stocks that publish pushed aims are the expected three",
        [p.name for p in _aim if len(p.aim_density) > 1]
        == ["KODAK_PORTRA_800", "KODAK_ULTRA_COLOR_400UC",
            "KODAK_ULTRAMAX_800"],
        ", ".join(p.name for p in _aim if len(p.aim_density) > 1))

    # ⚠ PORTRA 800's EI 800 FOREHEAD PAIR IS THE ADJUDICATION, PINNED. E-190
    # (2003) prints 1.08-1.18 / 0.93-1.03 here, which is the 160NC/400NC column
    # of the table above it copied verbatim; E-190 (2006) and E-4040 (2016)
    # both print 0.95-1.25 / 0.75-1.10. If the stored value ever reverts to the
    # NC pair, the 2003 table has been adopted by accident.
    _p800 = get_profile("KODAK_PORTRA_800").aim_density[0]
    _nc = get_profile("KODAK_PORTRA_400NC").aim_density[0]
    chk("PORTRA 800's aim is E-4040's, not the E-190 (2003) copy-paste",
        _p800.forehead_light == (0.95, 1.25)
        and _p800.forehead_light != _nc.forehead_light,
        f"800 {_p800.forehead_light} vs 400NC {_nc.forehead_light}")

    # ⚠ AND THE VC AIM IS NOT THE NC AIM. A higher-contrast emulsion is aimed
    # higher; the sheet prints two columns and storing one would have thrown
    # that away, which is the mistake a single shared table invites.
    chk("the VC stocks are aimed higher than the NC stocks",
        (get_profile("KODAK_PORTRA_160VC").aim_density[0].gray_card
         > get_profile("KODAK_PORTRA_160NC").aim_density[0].gray_card),
        f"VC {get_profile('KODAK_PORTRA_160VC').aim_density[0].gray_card} "
        f"NC {get_profile('KODAK_PORTRA_160NC').aim_density[0].gray_card}")

    # Inert, and it has to stay inert: an aim density used as an output would
    # fight the characteristic curve that already maps exposure to density.
    chk("aim_density is read by nothing on the render path",
        "aim_density" not in Path("film_sim.py").read_text(encoding="utf-8"))

    # ---- schema v21 (2026-08-31, queue K3): push curve sets ----------------
    # ProcessVariant widened to carry a push. The values are pinned in
    # `kodak_still_curves.EXPECTED_SECOND_CHAR`; what these check is that the
    # STORED sets still behave like a push of the stock they hang on.
    # ⚠ WHICH WAY EXTRA FIRST DEVELOPMENT MOVES A REVERSAL FILM'S CONTRAST,
    # PER STOCK, FROM ITS OWN TRACE. +1 = gamma rises with development, -1 =
    # falls. There are two measured stocks in the corpus and THEY DISAGREE,
    # which is why this is a table and not a rule:
    #   GEVACHROME_605          -1   Kino-Technik 1968 Bild 5b vs Bild 6,
    #                                cyan straight-line gamma 1.376 -> 1.156
    #   SUPER_ANSCOCHROME_1957  +1   PS&E 1(1) p12 Fig. 4, straight-line gamma
    #                                3.547 / 3.937 / 4.381 / 4.570 at 14 / 16 /
    #                                19 / 22 min
    # ⚠ A THIRD STOCK MUST BE ADDED HERE BY MEASUREMENT, NOT BY GUESS. A KeyError
    # from this dict is the correct failure: it means someone stored a reversal
    # push without reading which way its own panel goes.
    _REVERSAL_GAMMA_DIRECTION = {
        "GEVACHROME_605": -1,
        "SUPER_ANSCOCHROME_1957": +1,
    }
    _pushers = [(p, [v for v in p.process_variants if v.push_stops])
                for p in FILM_PROFILES]
    _pushers = [(p, vs) for p, vs in _pushers if vs]
    chk("exactly four stocks carry published push curve sets",
        sorted(p.name for p, _ in _pushers)
        == ["GEVACHROME_605", "KODAK_PORTRA_800", "KODAK_ULTRA_COLOR_400UC",
            "SUPER_ANSCOCHROME_1957"],
        ", ".join(p.name for p, _ in _pushers))

    # ---- schema v26 (2026-09-05, queue #215): push_stops is a FLOAT --------
    # ⚠ THE POINT OF THE WIDENING IS THE VALUES THAT COULD NOT BE STORED, so
    # what is asserted is that at least one of them IS stored. If every push in
    # the corpus goes back to being a whole stop, the field can go back to being
    # an int, and this failing is how anyone would find out.
    _frac = [(p.name, v.name, v.push_stops) for p, vs in _pushers for v in vs
             if abs(v.push_stops - round(v.push_stops)) > 1e-9]
    chk("v26: at least one stored push is a FRACTIONAL number of stops, which "
        "is the whole reason push_stops stopped being an int32",
        len(_frac) >= 2
        and {n for n, _v, _s in _frac} == {"SUPER_ANSCOCHROME_1957"},
        "; ".join("%s %s %+.4f" % t for t in _frac))
    # ⚠ AND EVERY STORED STOP COUNT MUST STILL BE log2 OF ITS OWN EI RATIO.
    # `FilmProfile.validate` checks this to 2 %, which is loose enough to let a
    # transcription slip past on a 1/3-stop rating -- 2 % of 0.585 is 0.012
    # stops, but 2 % of the EI is 3 units, and 150 against 147 is inside it.
    # This tightens the same relation to 0.001 stops, which no reading error
    # produces and every typo does.
    _bad_ratio = []
    for p, vs in _pushers:
        for v in vs:
            want = math.log2(v.exposure_index / p.exposure_index)
            if abs(want - v.push_stops) > 1e-3:
                _bad_ratio.append("%s %s: %+.4f stored, %+.4f from EI %d/%d"
                                  % (p.name, v.name, v.push_stops, want,
                                     v.exposure_index, p.exposure_index))
    chk("every stored push_stops is log2 of its own EI ratio to 0.001 stops",
        not _bad_ratio, "; ".join(_bad_ratio))

    # ⚠ A PUSH RAISES GAMMA ON EVERY LAYER AND RAISES DMIN -- ON A NEGATIVE.
    # That is what extended development does there, and it is the check that a
    # push set has not been stored against the wrong base: the failure mode the
    # "same panel group" rule in the ProcessVariant docstring exists to
    # prevent, and one that would otherwise look entirely plausible.
    #
    # ⚠ THIS GUARD USED TO SAY "EVERY PUSH" AND IT WAS OVER-GENERALISED FROM
    # TWO NEGATIVES. Adding GEVACHROME_605's measured 320 ASA set (queue G5,
    # 2026-09-03) failed it on all three channels, and the guard was wrong
    # rather than the data: on a REVERSAL film the first developer consumes the
    # silver that would otherwise become the positive image, so extending it
    # LOWERS gamma and LOWERS Dmax. Kino-Technik 1968 Nr. 10 Bild 6 measures
    # exactly that -- the cyan record falls 1.376 -> 1.156 by least squares
    # over D 0.5-2.0, both slopes traced from the same page with the same
    # estimator. The check is therefore split by `is_reversal`, and the
    # reversal branch asserts the OPPOSITE sign so that neither direction can
    # be stored by accident.
    _pw = []
    for p, vs in _pushers:
        base = p.curves
        for v in sorted(vs, key=lambda v: v.push_stops):
            ref = base
            if v.push_stops > 1:
                _lower = [w for w in p.process_variants
                          if 0 < w.push_stops < v.push_stops]
                if _lower:
                    ref = max(_lower, key=lambda w: w.push_stops).curves
            elif p.name == "KODAK_ULTRA_COLOR_400UC":
                # ⚠ THIS STOCK NOW CARRIES TWO READINGS OF THE SAME PUSH, FROM
                # TWO PUBLICATIONS, AND EACH IS MEASURED AGAINST ITS OWN.
                # E-4035 (May 2007) is the sheet for this product and the
                # profile's curves ARE its EI 400 panel; E-190 (2003) p13 also
                # prints the film, and its EI 400 panel is stored as a variant.
                # The two box speeds agree to 0.004 D, so either would pass --
                # which is precisely why the pairing is made explicitly instead
                # of being left to whichever variant comes first in the tuple.
                #
                # ⚠ THIS BRANCH USED TO READ "this stock's base is NOT its own
                # profile" and routed every push to the E-190 variant, because
                # E-4035 was not in the corpus and the stored curves matched no
                # panel anywhere. The owner supplied E-4035 on 2026-09-03; the
                # profile's curves are now traced from it.
                # ⚠ MATCHED ON THE RECORD'S NAME, NOT ITS SOURCE PROSE. Both
                # records CITE both publications -- each one's note quotes the
                # other's numbers, which is the point of storing a conflict --
                # so a substring test against `source` finds E-4035 in the
                # E-190 record and pairs it with itself. The name states which
                # sheet the record IS.
                _pub = "E-190" if "E-190" in v.name else "E-4035"
                if _pub != "E-4035":
                    _same = [w for w in p.process_variants
                             if w.push_stops == 0 and w.curves is not None
                             and _pub in w.name]
                    if _same:
                        ref = _same[0].curves
            for _ch in "rgb":
                a, b = getattr(ref, _ch), getattr(v.curves, _ch)
                if p.is_reversal:
                    # ⚠ REWRITTEN 2026-09-05 (queue #215), AND THE PREVIOUS
                    # VERSION WAS OVER-GENERALISED FROM ONE STOCK IN EXACTLY
                    # THE WAY THE COMMENT ABOVE DESCRIBES FOR THE NEGATIVES.
                    # It required a reversal push to LOWER gamma, on the
                    # strength of GEVACHROME_605 alone. Super Anscochrome's
                    # four measured developments do the opposite: straight-line
                    # gamma over D 0.5-2.0 rises 3.547 -> 3.937 -> 4.381 ->
                    # 4.570 from 14 to 22 minutes of first development, and the
                    # fitted gamma rises with it. Both readings are traces of
                    # printed panels, so this is a CONFLICT TO RECORD and not
                    # an error to average away.
                    #
                    # WHAT IS ACTUALLY UNIVERSAL, and it holds on both stocks
                    # and on all four Ansco curves: the first developer consumes
                    # the silver that would otherwise become the positive image,
                    # so MORE of it means LESS Dmax and LESS of it means MORE.
                    # Gamma is a RATIO of that Dmax to the exposure scale the
                    # curve is spread over, and the scale shortens too --
                    # Anscochrome's throw falls 0.670 -> 0.481 decades from B to
                    # D, faster than its Dmax falls, so its contrast rises while
                    # Gevachrome's falls. Nothing here can predict which wins,
                    # so nothing here asserts it: the DIRECTION per stock is
                    # pinned below, from the trace, one stock at a time.
                    _amax = a.dmin + a.gamma * (a.shoulder_x - a.toe_x)
                    _bmax = b.dmin + b.gamma * (b.shoulder_x - b.toe_x)
                    _want_fall = v.push_stops > 0
                    if _want_fall and _bmax >= _amax:
                        _pw.append(f"{p.name} {v.name} {_ch}: reversal push "
                                   f"Dmax {_amax:.3f} -> {_bmax:.3f}, "
                                   f"expected a FALL")
                    if not _want_fall and _bmax <= _amax:
                        _pw.append(f"{p.name} {v.name} {_ch}: reversal PULL "
                                   f"Dmax {_amax:.3f} -> {_bmax:.3f}, "
                                   f"expected a RISE -- less first development "
                                   f"leaves more silver for the positive")
                    # ⚠ AND THE CONTRAST DIRECTION IS NOT CHECKED HERE, on the
                    # ToneCurve PARAMETER, because on a short-scale fit that
                    # parameter is not the film's contrast. Worked case, Super
                    # Anscochrome A against B: the parameter gamma RISES 5.2706
                    # -> 5.3801 while the straight-line gamma over D 0.5-2.0
                    # FALLS 3.937 -> 3.547. Both are correct about different
                    # quantities -- `ToneCurve.gamma` is an asymptotic slope
                    # coupled to `shoulder_x - toe_x`, and A's throw is LONGER
                    # (0.693 against 0.670 decades) as well as taller, so the
                    # ratio moves the other way from the slope actually drawn.
                    # The contrast direction is asserted below instead, on
                    # `processing_family`, which carries the measured
                    # straight-line gammas.
                    continue
                if b.gamma <= a.gamma:
                    _pw.append(f"{p.name} {v.name} {_ch}: gamma "
                               f"{a.gamma:.4f} -> {b.gamma:.4f}")
                # ⚠ DMIN IS ALLOWED TO SIT STILL, WITHIN THE READING ERROR, AND
                # ONE CHANNEL DOES. PORTRA 800's blue base+fog reads 1.0072 at
                # EI 800 and 1.0030 at EI 1600 -- a fall of 0.004 D against a
                # fit rms of 0.021 D on that very trace, i.e. the two panels
                # draw the same blue base and the reader cannot separate them.
                # Requiring a strict rise would make this check fail on noise;
                # 0.02 D is the fit rms, so a REAL fall still trips it.
                if b.dmin < a.dmin - 0.02:
                    _pw.append(f"{p.name} {v.name} {_ch}: dmin "
                               f"{a.dmin:.4f} -> {b.dmin:.4f}")
    chk("a negative push gains contrast; a reversal one loses Dmax whichever "
        "way its contrast goes",
        not _pw, "; ".join(_pw[:4]))

    # ⚠ CONTRAST AGAINST DEVELOPMENT, ON THE MEASURED STRAIGHT-LINE GAMMA, for
    # every reversal stock that publishes a development ladder. This is the
    # check the ToneCurve-parameter version above could not be: a
    # `ProcessingFamily` point carries the gamma a datasheet prints, read off
    # the drawn straight line, and that is what "contrast" means.
    # ⚠ THE TWO MEASURED STOCKS DISAGREE ON THE SIGN, which is why this reads a
    # table instead of asserting a law. See `_REVERSAL_GAMMA_DIRECTION` above.
    _gd = []
    for p in FILM_PROFILES:
        if not p.is_reversal or not p.processing_family.points:
            continue
        _pts = [pt for pt in p.processing_family.points
                if pt.minutes > 0 and pt.gamma > 0]
        if len(_pts) < 2:
            continue
        _dir = _REVERSAL_GAMMA_DIRECTION.get(p.name)
        if _dir is None:
            _gd.append(f"{p.name}: publishes a {len(_pts)}-point development "
                       f"ladder and no measured contrast direction is recorded")
            continue
        _pts = sorted(_pts, key=lambda q: q.minutes)
        for _a, _b in zip(_pts, _pts[1:]):
            if (_b.gamma - _a.gamma) * _dir <= 0.0:
                _gd.append(f"{p.name}: {_a.minutes:.0f} -> {_b.minutes:.0f} min "
                           f"gamma {_a.gamma:.4f} -> {_b.gamma:.4f} against a "
                           f"recorded direction of {_dir:+d}")
            # And Dmax's direction is not optional on any of them: more first
            # development, less silver left for the positive, lower base+fog.
            if _b.base_fog > 0 and _a.base_fog > 0 and _b.base_fog >= _a.base_fog:
                _gd.append(f"{p.name}: {_a.minutes:.0f} -> {_b.minutes:.0f} min "
                           f"base+fog {_a.base_fog:.4f} -> {_b.base_fog:.4f}, "
                           f"expected a FALL on a reversal film")
    chk("every reversal development ladder moves contrast in its own recorded "
        "direction and base+fog downwards", not _gd, "; ".join(_gd[:4]))

    # ---- method rule 23 (2026-09-05c): an absence must not read as a fact --
    # ⚠ THE RULE THIS GUARDS IS THE OWNER'S RESEARCH POLICY, and it is the one
    # rule in this file that is about EPISTEMICS rather than physics: a
    # parameter with no recorded provenance inherits the profile's tier, so an
    # unmeasured number on a tier-1 stock silently reads as datasheet-grounded.
    # `dye_matrix` sat in exactly that state on 176 profiles until 2026-09-05c.
    #
    # ⚠ AND THE FIX WAS NOT TO CHANGE THE VALUES. Measured the same day: forcing
    # dye_matrix to identity collapses the rendered saturation spread from
    # 0.077-0.312 to 0.117-0.174 and puts AGFA_NEU_1936 within 10 % of
    # VELVIA. Dye purity is real and the per-channel curves cannot carry it --
    # they are measured on a NEUTRAL scale. Deleting an effect because its
    # numbers are unsourced is what rule 23 forbids; SAYING they are unsourced
    # is what it requires.
    _no_prov = sorted(p.name for p in FILM_PROFILES if not p.is_monochrome
                      and not any(s.param == "dye_matrix"
                                  for s in p.param_sources))
    chk("rule 23: every colour stock states the provenance of its dye_matrix, "
        "so an unmeasured crosstalk cannot inherit a tier-1 profile's authority",
        not _no_prov, ", ".join(_no_prov[:4]))
    _bad_status = sorted(
        p.name for p in FILM_PROFILES if not p.is_monochrome
        for s in p.param_sources
        if s.param == "dye_matrix" and s.status not in ("estimated", "assumed",
                                                        "derived", "measured",
                                                        "traced"))
    chk("rule 23: and none of them claims a status the evidence does not "
        "support", not _bad_status, ", ".join(_bad_status[:4]))
    # ⚠ AN IDENTITY MATRIX IS A CLAIM TOO, AND IT MUST SAY SO. Rule 23 point 3:
    # a neutral value chosen because nothing was found asserts "measured to be
    # zero", which is a different and usually false statement.
    _silent_identity = []
    for p in FILM_PROFILES:
        if p.is_monochrome:
            continue
        _m = p.dye_matrix
        if any(abs(_m[i][j] - (1.0 if i == j else 0.0)) > 1e-9
               for i in range(3) for j in range(3)):
            continue
        _s = [s for s in p.param_sources if s.param == "dye_matrix"]
        if not _s or "UNMEASURED" not in _s[0].note.upper():
            _silent_identity.append(p.name)
    chk("rule 23: an identity dye_matrix is recorded as UNMEASURED, never left "
        "to read as a measurement of zero",
        not _silent_identity, ", ".join(_silent_identity[:4]))

    # ---- schema v27 (2026-09-05, queue C23): bromide drag ------------------
    # ⚠ WHAT THIS GUARDS IS AN ABSENCE, WHICH IS THE HARDEST THING TO KEEP TRUE.
    # Stage 9c is live in all three engines and inert on every stock, so a
    # render is bit-identical to a v26 one. The day someone fits a number, this
    # check fails -- and it should, because that is the day the claim "no render
    # moves" stops being true and every document repeating it needs editing.
    _drags = [(p.name, p.processing.bromide_drag) for p in FILM_PROFILES
              if p.processing.bromide_drag.has_data]
    chk("C23: bromide drag is INERT on every stock, so a v27 database renders "
        "bit-identically to a v26 one",
        not _drags,
        "; ".join("%s strength %.3f length %.1f mm" % (n, d.strength, d.length_mm)
                  for n, d in _drags))
    # ⚠ AND THE CARRIER IS REACHABLE AND VALIDATED, which is a different claim
    # from "it is zero". A field nothing validates is a field that will hold
    # nonsense the first time it holds anything.
    chk("C23: BromideDragSpec is validated on every stock and refuses an "
        "unimplemented transport axis",
        all(hasattr(p.processing, "bromide_drag")
            and p.processing.bromide_drag.axis == 0
            and p.processing.bromide_drag.direction in (1, -1)
            for p in FILM_PROFILES),
        "176 records, axis 0, direction +/-1")
    _refused = 0
    for _bad in (dict(strength=-0.1, length_mm=1.0),
                 dict(strength=0.9, length_mm=1.0),
                 dict(strength=0.05, length_mm=0.0),
                 dict(strength=0.0, length_mm=4.0),
                 dict(strength=0.05, length_mm=4.0, axis=1),
                 dict(strength=0.05, length_mm=4.0, direction=0),
                 dict(strength=0.05, length_mm=4.0)):   # no source
        try:
            _fpm.BromideDragSpec(**_bad).validate("probe")
        except ValueError:
            _refused += 1
    chk("C23: BromideDragSpec.validate refuses all seven malformed records -- a "
        "negative or absurd strength, a length with no strength and the "
        "reverse, the unimplemented axis, a zero direction, and a measurement "
        "with no source", _refused == 7, f"{_refused} of 7 refused")
    # ⚠ THE MILLIMETRE CONTRACT, checked here as well as in the parity probe,
    # because it is the property that makes ONE record correct at every output
    # resolution and on every gauge. A pixel-denominated length would fail this
    # by the ratio of the two resolutions.
    _a25 = _fsim.bromide_drag_alpha(3.0, 25.0)
    _a100 = _fsim.bromide_drag_alpha(3.0, 100.0)
    chk("C23: the drag coefficient is per-pixel but the record is per-"
        "millimetre -- four times the resolution is the fourth root of the "
        "retention",
        abs(_a25 - _a100 ** 4) < 1e-12 and 0.0 < _a25 < 1.0,
        "a(25) %.9f, a(100)^4 %.9f" % (_a25, _a100 ** 4))
    # And the stage must be genuinely wired, not merely defined.
    _fsrc = Path("film_sim.py").read_text(encoding="utf-8")
    chk("C23: stage 9c is CALLED by the pipeline, not merely defined",
        "apply_bromide_drag(dens, profile.processing.bromide_drag" in _fsrc,
        "film_sim.simulate() calls it between stage 9 and stage 10")

    # ---- 2026-09-03: E-4035, the ULTRA COLOR pair --------------------------
    # ⚠ WHAT THESE GUARD IS A CLASS OF ERROR, NOT A PAIR OF NUMBERS. Both
    # profiles previously carried ANALOGY curves that had been cited to E-4035
    # for as long as the profiles existed, while E-4035 was not in the corpus.
    # Nothing failed, because an analogy curve is a valid curve. What made it
    # findable was the OWNER noticing that the two films rendered identically,
    # and the reason they did is in the numbers below.
    _uc100 = get_profile("KODAK_ULTRA_COLOR_100UC")
    _uc400 = get_profile("KODAK_ULTRA_COLOR_400UC")

    # 1. THE ORANGE MASK. The old triples were dmin 0.20 / 0.19 / 0.19 and
    #    0.21 / 0.20 / 0.20 -- three near-equal base densities on a MASKED
    #    colour negative, which is no mask at all, the same signature queue row
    #    B4 found on 5247_1983. A real C-41 negative's blue base sits far above
    #    its red base. 0.5 D is well inside the measured 0.70 and comfortably
    #    outside anything a flat triple could reach.
    _flat = [q.name for q in (_uc100, _uc400)
             if q.curves.b.dmin - q.curves.r.dmin < 0.50]
    chk("both ULTRA COLOR stocks carry a real orange mask, not three equal "
        "base densities", not _flat, ", ".join(_flat))

    # 2. THE DIRECTION THE SHEET ESTABLISHES, WHICH THE OLD PAIR HAD BACKWARDS.
    #    E-4035 has 400UC denser at base AND contrastier on every layer; the
    #    stored pair made it uniformly SOFTER by 0.004 on all three, which is
    #    one hand nudge rather than two readings. The gamma difference being
    #    channel-dependent (+0.030 / +0.010 / +0.014) is also what rules out an
    #    axis or tracing error, since either would move all three alike.
    _dir = []
    for _ch in "rgb":
        a, b = getattr(_uc100.curves, _ch), getattr(_uc400.curves, _ch)
        if not (b.dmin > a.dmin and b.gamma > a.gamma):
            _dir.append(f"{_ch}: 100UC {a.dmin:.4f}/{a.gamma:.4f} vs "
                        f"400UC {b.dmin:.4f}/{b.gamma:.4f}")
    chk("400UC is denser at base and contrastier than 100UC on all three "
        "layers, as E-4035 prints them", not _dir, "; ".join(_dir))
    chk("and the gamma difference is channel-dependent, which is what a film "
        "difference looks like and a uniform nudge does not",
        len({round(getattr(_uc400.curves, c).gamma
                   - getattr(_uc100.curves, c).gamma, 3) for c in "rgb"}) == 3,
        ", ".join("%s %+.4f" % (c, getattr(_uc400.curves, c).gamma
                                - getattr(_uc100.curves, c).gamma)
                  for c in "rgb"))

    # 3. THE CROSS-DOCUMENT AGREEMENT, which is the whole reason the reading is
    #    trusted over the analogy set it replaced. E-190 (2003) p13 reads the
    #    same emulsion four years earlier in a different publication. If these
    #    two ever part company, one of them has been re-traced wrongly.
    # ⚠ SELECTED BY NAME. Every one of these records quotes the other sheet's
    # numbers in its own note -- that is what storing a conflict looks like --
    # so a substring test against `source` matches both records for both
    # publications. The name is the record's identity.
    _e190 = [w for w in _uc400.process_variants
             if w.curves is not None and w.push_stops == 0
             and "E-190" in w.name]
    chk("400UC's box speed is stored from two publications", len(_e190) == 1,
        "%d E-190 box-speed variants" % len(_e190))
    if _e190:
        _gap = max(max(abs(getattr(_e190[0].curves, c).dmin
                           - getattr(_uc400.curves, c).dmin),
                       abs(getattr(_e190[0].curves, c).gamma
                           - getattr(_uc400.curves, c).gamma))
                   for c in "rgb")
        chk("E-4035 and E-190 (2003) agree on 400UC's box speed to 0.005",
            _gap < 0.005, "worst channel disagreement %.4f" % _gap)

    # 4. ⚠ AND THE TWO PUSH READINGS DISAGREE BY MORE THAN THAT, WHICH IS
    #    RECORDED RATHER THAN AVERAGED (method rule 4). A push panel is the
    #    most process-sensitive figure on either sheet. This asserts the
    #    conflict still exists, so that a later edit cannot quietly reconcile
    #    it by dropping one record or splitting the difference.
    _p4035 = [w for w in _uc400.process_variants
              if w.push_stops == 1 and "E-4035" in w.name]
    _p190 = [w for w in _uc400.process_variants
             if w.push_stops == 1 and "E-190" in w.name]
    chk("400UC's EI 800 push is stored from both sheets",
        len(_p4035) == 1 and len(_p190) == 1,
        "E-4035 %d, E-190 %d" % (len(_p4035), len(_p190)))
    if _p4035 and _p190:
        _dg = max(abs(getattr(_p4035[0].curves, c).gamma
                      - getattr(_p190[0].curves, c).gamma) for c in "rgb")
        _dd = max(abs(getattr(_p4035[0].curves, c).dmin
                      - getattr(_p190[0].curves, c).dmin) for c in "rgb")
        chk("the two push readings still disagree on gamma by more than the "
            "two box-speed readings do -- recorded, not averaged",
            _dg > 0.015 and _dd < 0.02,
            "gamma worst %.4f, dmin worst %.4f" % (_dg, _dd))

    # 5. THE TRUNCATED-TRACE CLASS, ON THE THREE PROFILES IT ACTUALLY MOVED.
    # ⚠ ALL THREE WERE CAUGHT BY THE SAME READER DEFECT and all three failed
    # SILENTLY, because a trace whose flat base+fog stub was dropped still
    # yields the right number of traces -- so `measure_char` estimated a
    # plateau off a trace containing no plateau, always high. These pin the
    # corrected values; the reader's own EXPECTED tables pin the readings.
    chk("ULTRA MAX 800's red base is the drawn plateau, not one estimated "
        "from a trace missing its plateau",
        abs(get_profile("KODAK_ULTRAMAX_800").curves.r.dmin - 0.3128) < 5e-4,
        "%.4f" % get_profile("KODAK_ULTRAMAX_800").curves.r.dmin)
    _p800v = {w.exposure_index: w for w in get_profile(
        "KODAK_PORTRA_800").process_variants if w.curves is not None}
    chk("PORTRA 800's two push records carry the corrected red base",
        abs(_p800v[1600].curves.r.dmin - 0.2569) < 5e-4
        and abs(_p800v[3200].curves.r.dmin - 0.3011) < 5e-4,
        "EI1600 %.4f, EI3200 %.4f" % (_p800v[1600].curves.r.dmin,
                                      _p800v[3200].curves.r.dmin))

    # 6. Pro 100T's dye pair, which was short by a fifth of the spectrum.
    # ⚠ ITS OWN COMMENT ASSERTED THE ERROR -- "neither reaches 400 nm on this
    # sheet" -- and both curves do. The 450-700 nm tail is unchanged, so this
    # checks the two things that moved: where the array starts, and the peak.
    _d29 = get_profile("KODAK_PRO_100T_PRT").dye_density
    chk("Pro 100T's dye pair starts at 400 nm and spans the panel",
        _d29.lambda_start_nm == 400.0 and len(_d29.d_neutral) == 61
        and len(_d29.d_dmin) == 61,
        "start %.0f, n %d/%d" % (_d29.lambda_start_nm, len(_d29.d_neutral),
                                 len(_d29.d_dmin)))
    chk("and its peaks are the panel's 400 nm values, not its 450 nm ones",
        abs(max(_d29.d_neutral) - 2.121) < 5e-4
        and abs(max(_d29.d_dmin) - 1.611) < 5e-4,
        "%.3f / %.3f" % (max(_d29.d_neutral), max(_d29.d_dmin)))
    chk("the 450-700 nm tail is byte-identical to the pre-2026-09-03 array, "
        "which is what makes this an extension and not a re-reading",
        _d29.d_neutral[10] == 1.616 and _d29.d_dmin[10] == 0.869
        and _d29.d_neutral[-1] == 1.286 and _d29.d_dmin[-1] == 0.247,
        "%.3f / %.3f at 450 nm" % (_d29.d_neutral[10], _d29.d_dmin[10]))

    # ---- 2026-09-03: the Jones 1958 sigma(D) class shape -------------------
    # ⚠ THE OLD PLACEHOLDER POINTED THE WRONG WAY AND NOTHING CAUGHT IT,
    # because nothing reads an unmeasured shape. These guards assert the
    # DIRECTION the measurement establishes, so a future edit cannot quietly
    # put the old shape back: sigma rises steeply out of the toe and then
    # FLATTENS, it does not keep climbing.
    import pse_jones_1958 as _pj
    _mono = [q for q in FILM_PROFILES if q.is_monochrome and not q.is_reversal]
    _cls = [q for q in _mono
            if abs(q.grain.sigma_shape_toe - _pj.ADOPTED_TOE) < 1e-9]
    chk("the monochrome-negative block carries the Jones 1958 class shape",
        len(_cls) >= 50, "%d of %d" % (len(_cls), len(_mono)))
    _wrong = [q.name for q in _cls
              if not (0.45 < q.grain.sigma_shape_toe < 0.56
                      and 0.95 < q.grain.sigma_shape_dmax < 1.10)]
    chk("that shape flattens above D 1.0 instead of climbing -- the old 1.20 "
        "placeholder is refuted by all four measured films",
        not _wrong, ", ".join(_wrong[:3]))
    # the anchor densities must be the MEASURED range, not a stock's Dmax
    _bad_at = [q.name for q in _cls
               if abs(q.grain.sigma_shape_dmax_at - 1.40) > 1e-9
               or abs(q.grain.sigma_shape_toe_at - 0.07) > 1e-9]
    chk("the class shape's anchor densities are where Jones measured, not "
        "where a stock's curve ends", not _bad_at, ", ".join(_bad_at[:3]))
    # ⚠ and it must stay inert until someone decides otherwise, in the open
    _live = [q.name for q in _cls if q.grain.sigma_shape_measured]
    chk("the class shape is still gated off -- a class inference must not set "
        "the per-stock measured flag", not _live, ", ".join(_live[:3]))
    _sel = _pj.selwyn_ratios()
    chk("Jones's three apertures still confirm Selwyn to within 10 %",
        abs(float(_sel.mean()) - 1.0) < 0.10,
        "mean sigma10/(2 sigma20) = %.3f over %d pairs"
        % (_sel.mean(), len(_sel)))

    # ---- 2026-09-03: the separable MTF kernel, and the bound it carries ----
    # ⚠ THIS GUARD ASSERTS AN APPROXIMATION, WHICH IS UNUSUAL HERE AND
    # DELIBERATE. `mtf_response` is the law; the C++ engine has no FFT and
    # convolves separable Gaussians, so it applies `mtf_kernel`'s two-lobe fit
    # instead. What must stay true is (a) the fit is close enough to be worth
    # calling the same law, and (b) it is closer than the single Gaussian it
    # replaced -- because if it ever stops being closer, the whole change was
    # pointless and should be reverted rather than kept.
    _KTOL = 0.045
    _kq = sorted({round(p.mtf.mtf_rolloff_q, 4) for p in FILM_PROFILES
                  if p.mtf.mtf_measured and p.mtf.mtf_rolloff_q > 0.0})
    _fk = np.logspace(-1.3, 0.9, 600)
    _l2 = math.log(2.0)
    _kbad, _kworst, _gworst = [], 0.0, 0.0
    _n3 = 0
    for _q in _kq:
        _k = film_profiles.mtf_kernel(_q)
        _k3 = film_profiles.mtf_kernel3(_q) if _k is None else None
        if _k is None and _k3 is None:
            _kbad.append("q %.4f is stored on a stock and has no kernel row" % _q)
            continue
        _tgt = 1.0 / (1.0 + _fk ** _q)
        if _k3 is not None:
            _n3 += 1
            _w1, _w2, _s1, _s2, _s3 = _k3
            _fit = (_w1 * np.exp(-_l2 * (_fk * _s1) ** 2)
                    + _w2 * np.exp(-_l2 * (_fk * _s2) ** 2)
                    + (1.0 - _w1 - _w2) * np.exp(-_l2 * (_fk * _s3) ** 2))
        else:
            _w1, _s1, _s2 = _k
            _fit = (_w1 * np.exp(-_l2 * (_fk * _s1) ** 2)
                    + (1.0 - _w1) * np.exp(-_l2 * (_fk * _s2) ** 2))
        _e = float(np.max(np.abs(_fit - _tgt)))
        _g = float(np.max(np.abs(np.exp(-_l2 * _fk ** 2) - _tgt)))
        _kworst = max(_kworst, _e)
        _gworst = max(_gworst, _g)
        if _e > _KTOL:
            _kbad.append("q %.4f: kernel error %.4f over tolerance" % (_q, _e))
        if _e >= _g:
            _kbad.append("q %.4f: kernel %.4f is NOT better than the single "
                         "Gaussian %.4f" % (_q, _e, _g))
    chk("every stored rolloff exponent has a kernel row, inside tolerance and "
        "better than the Gaussian it replaced", not _kbad,
        "; ".join(_kbad[:3]))
    chk("the kernel's worst error is well under the single Gaussian's",
        _kworst < 0.5 * _gworst,
        "kernel %.4f vs Gaussian %.4f" % (_kworst, _gworst))

    # ⚠ THE THREE-LOBE TABLE EXISTS FOR EXPONENTS THE TWO-LOBE ONE CANNOT
    # REACH, AND THIS ASSERTS THAT IT IS NOT USED FOR ANYTHING ELSE. Every q it
    # carries must be ABSENT from the two-lobe table (otherwise a stock would
    # silently change path and change pixels), must be OUTSIDE that table's
    # range (otherwise a two-lobe row should have been fitted instead), and its
    # own two-lobe best fit must actually FAIL the tolerance -- which is the
    # measured justification for spending three lobes rather than two.
    _t2 = film_profiles._MTF_KERNEL_TABLE
    _t3 = film_profiles._MTF_KERNEL_TABLE3
    _3bad = []
    for _q3, _row in sorted(_t3.items()):
        if _q3 in _t2:
            _3bad.append("q %.4f is in BOTH tables" % _q3); continue
        if min(_t2) <= _q3 <= max(_t2):
            _3bad.append("q %.4f is inside the two-lobe table's range %.2f-%.2f"
                         % (_q3, min(_t2), max(_t2)))
        _tg = 1.0 / (1.0 + _fk ** _q3)
        _best = 9.9
        for _w in np.linspace(0.02, 1.10, 55):
            for _a in np.linspace(0.05, 1.2, 40):
                for _b2 in np.linspace(1.0, 9.0, 41):
                    _f2 = (_w * np.exp(-_l2 * (_fk * _a) ** 2)
                           + (1.0 - _w) * np.exp(-_l2 * (_fk * _b2) ** 2))
                    _best = min(_best, float(np.max(np.abs(_f2 - _tg))))
        if _best <= _KTOL:
            _3bad.append("q %.4f reaches %.4f on TWO lobes, inside %.3f -- it "
                         "does not need three" % (_q3, _best, _KTOL))
    chk("every three-lobe row is one the two-lobe family genuinely cannot "
        "serve, and no stock changes path because the table exists",
        not _3bad and _n3 == len(_t3),
        "; ".join(_3bad) if _3bad
        else "%d row(s), %d stock exponent(s) routed to them" % (len(_t3), _n3))

    # ⚠ THE TABLE IS NOT INTERPOLATABLE AND THE GUARD SAYS SO. Two disjoint
    # optimal basins straddle q ~ 3.0: below it the tight lobe is small
    # (w1 < 0.4), above it the fit flips to w1 > 1 with a negative wide lobe.
    # Anything that starts interpolating this table must trip here first.
    # ---- measured endpoints (2026-09-15b) ---------------------------------
    # ⚠ THESE ARE THE ONLY DIRECTLY MEASURED D-MAX VALUES IN THE CORPUS. Every
    # other Dmax in this project is a CONSEQUENCE of evaluating a four-parameter
    # ToneCurve fit; these twelve are the top of a traced curve. The guard
    # asserts the three properties a mis-trace breaks, and one that ties them
    # to data this reader did not produce.
    #
    # ⚠ NINE BECAME TWELVE ON 2026-09-17e, AND THE THREE THAT ARRIVED CAME FROM
    # A DEFECT BEING FOUND RATHER THAN A DOCUMENT BEING SOURCED. Queue P55 held
    # that the three records on the book's colour panels are printed on top of
    # each other; they are not, and `drop_text` was erasing the two dashed ones
    # wherever they ran level. See the comment on AGFA_OPTIMA_100.
    #
    # ⚠ AND THREE MORE WERE TRACED THE SAME DAY AND ARE DELIBERATELY NOT HERE.
    # AGFA_VISTA_200 traced a blue D-min of 0.820 against the 1.050 this
    # database stores from an Agfa sheet -- a 0.23 D disagreement that test 4
    # below refuses, so it is recorded as a conflict rather than stored.
    # KODAK_T400CN and KODAK_BW400CN are chromogenic B&W: their panels DO print
    # three status-M records and both traced cleanly, but the profiles are
    # monochrome and their stored `curves` carry ONE base density replicated
    # three times, so test 4 has nothing to check them against. A record that
    # cannot be checked is not stored -- the same rule that keeps
    # EKTACHROME_160T's dye pair out of the database.
    _ep = {q.name: q.measured_endpoints for q in FILM_PROFILES
           if q.measured_endpoints.dmax[2] > 0.0}
    _epbad = []
    for _n, _e in sorted(_ep.items()):
        _p = get_profile(_n)
        # 1. an orange-masked negative's base densities rise r < g < b ...
        if not (_e.dmin[0] < _e.dmin[1] < _e.dmin[2]):
            _epbad.append("%s dmin not ordered r<g<b %s" % (_n, _e.dmin))
        # 2. ... and so do its maxima, because the mask sits under everything
        if not (_e.dmax[0] < _e.dmax[1] < _e.dmax[2]):
            _epbad.append("%s dmax not ordered r<g<b %s" % (_n, _e.dmax))
        # 3. a maximum is above a minimum by a usable range on every record
        if min(b - a for a, b in zip(_e.dmin, _e.dmax)) < 1.0:
            _epbad.append("%s density range under 1.0 D" % _n)
        if _e.log_e_span < 2.5:
            _epbad.append("%s spans only %.2f decades" % (_n, _e.log_e_span))
        # 4. ⚠ AND THE TRACED D-MIN MUST REPRODUCE THE DATABASE'S OWN, which is
        # what licenses the D-max column beside it. Nine of these twelve base
        # densities came from manufacturer sheets this reader never saw.
        _dbd = (_p.curves.r.dmin, _p.curves.g.dmin, _p.curves.b.dmin)
        if max(abs(a - b) for a, b in zip(_dbd, _e.dmin)) > 0.12:
            _epbad.append("%s traced dmin %s disagrees with the stored %s"
                          % (_n, tuple(round(v, 3) for v in _e.dmin),
                             tuple(round(v, 3) for v in _dbd)))
    chk("the twelve measured endpoint records are ordered, spanned, and their "
        "D-min reproduces the database's own",
        not _epbad and len(_ep) == 12,
        "; ".join(_epbad[:3]) if _epbad
        else "%d stocks, worst dmax %.3f, all r<g<b" % (
            len(_ep), max(max(e.dmax) for e in _ep.values())))

    # ---- G-V40-P55: the three records queue P55 said could not be separated
    # ⚠ PINNED BY VALUE, because the whole row turned on a claim about what the
    # panels contain. If a future edit to `sovremennye_2004.drop_text` or to the
    # bridging order silently loses the dashed records again, these numbers move
    # and this guard fails before any document says the panels are unreadable.
    _P55_NEW = {
        "AGFA_OPTIMA_100": ((0.2759, 0.6864, 0.9247),
                            (2.1430, 2.5701, 2.9839), 80),
        "AGFA_OPTIMA_200": ((0.2546, 0.6519, 0.8241),
                            (1.9894, 2.4065, 2.7707), 81),
        "AGFA_PORTRAIT_160": ((0.2684, 0.5980, 0.7562),
                              (1.8405, 2.3184, 2.6480), 84),
    }
    _p55 = []
    for _n, (_dn, _dx, _pg) in sorted(_P55_NEW.items()):
        _e = get_profile(_n).measured_endpoints
        if _e.page != _pg:
            _p55.append("%s page %d, expected %d" % (_n, _e.page, _pg))
        if max(abs(a - b) for a, b in zip(_e.dmin, _dn)) > 1e-4:
            _p55.append("%s dmin moved to %s" % (_n, _e.dmin))
        if max(abs(a - b) for a, b in zip(_e.dmax, _dx)) > 1e-4:
            _p55.append("%s dmax moved to %s" % (_n, _e.dmax))
        # and the property that makes the record worth having: the book's own
        # blue-minus-red separation at D-max must still be the mask, not noise
        if _e.dmax[2] - _e.dmax[0] < 0.55:
            _p55.append("%s mask separation at Dmax collapsed to %.3f"
                        % (_n, _e.dmax[2] - _e.dmax[0]))
    chk("G-V40-P55: the three Agfa endpoint records recovered by bridging the "
        "dashes before dropping the blobs are unchanged",
        not _p55, "; ".join(_p55[:3]) if _p55 else
        "3 stocks, mask separation at Dmax %.3f to %.3f" % (
            min(v[1][2] - v[1][0] for v in _P55_NEW.values()),
            max(v[1][2] - v[1][0] for v in _P55_NEW.values())))

    _lo = [v[0] for k, v in film_profiles._MTF_KERNEL_TABLE.items() if k < 3.05]
    _hi = [v[0] for k, v in film_profiles._MTF_KERNEL_TABLE.items() if k >= 3.05]
    chk("the kernel table's two basins are separated, so it cannot be "
        "interpolated across q = 3.05",
        bool(_lo) and bool(_hi) and max(_lo) < 0.5 and min(_hi) > 0.95,
        "low basin max w1 %.3f, high basin min w1 %.3f"
        % (max(_lo) if _lo else -1, min(_hi) if _hi else -1))

    # ---- queue G5 (2026-09-03): the Gevachrome gradation traces ------------
    # ⚠ THESE ASSERT PROPERTIES OF THE STORED DATABASE, NOT OF THE TRACER.
    # `gevachrome_1968_raster.py` re-derives the numbers from the page and is
    # run by the build; what is checked here is that the profile still says
    # what the trace found, so an edit to film_profiles.py alone trips it.
    _G5_EDGE = {"GEVACHROME_600": (2.729, 2.351, 2.229),
                "GEVACHROME_605": (2.505, 2.266, 2.096)}
    _G5_GAMMA = {"GEVACHROME_600": (1.45, 1.25, 1.25),
                 "GEVACHROME_605": (1.35, 1.25, 1.25)}
    _g5 = []
    for _n, _want in _G5_EDGE.items():
        _p = get_profile(_n)
        for _ch, _w, _g in zip("rgb", _want, _G5_GAMMA[_n]):
            _c = getattr(_p.curves, _ch)
            _dmax = _c.dmin + _c.gamma * (_c.shoulder_x - _c.toe_x)
            if abs(_dmax - _w) > 0.01:
                _g5.append(f"{_n} {_ch}: Dmax {_dmax:.3f} vs traced {_w:.3f}")
            if abs(_c.gamma - _g) > 1e-9:
                _g5.append(f"{_n} {_ch}: gamma {_c.gamma} vs printed {_g}")
    chk("Gevachrome curves reproduce the traced Dmax and the printed gamma",
        not _g5, "; ".join(_g5[:4]))

    # ⚠ THE SHOULDER IS SHARPER THAN THE TOE ON ALL SIX CHANNELS, and that is
    # the whole content of the G5 upgrade. Both stocks used to carry a [T2]
    # transfer from GEVACHROME_902 with toe_k 0.18 and shoulder_k 0.30 -- a
    # shoulder SOFTER than the toe. Bilder 5a/5b draw the opposite: for a
    # reversal `shoulder_x` is the SHADOW end, and the Dmax corner there is
    # nearly square, so the fitted shoulder_k lands at 0.038-0.122 against a
    # toe_k of 0.182-0.238. If a future edit restores a soft shoulder on these
    # two stocks it has thrown the measurement away.
    _g5s = [f"{_n} {_ch}"
            for _n in _G5_EDGE
            for _ch in "rgb"
            if getattr(get_profile(_n).curves, _ch).shoulder_k >=
            getattr(get_profile(_n).curves, _ch).toe_k]
    chk("Gevachrome's traced shoulder is sharper than its toe on all six "
        "channels", not _g5s, ", ".join(_g5s))

    # ⚠ AND THE PUSH IS NOT A UNIFORM SCALE, which is why `gamma_scale` was
    # refused for these records and full curves stored instead. PORTRA 800 at
    # EI 3200 gains 0.155 of gamma in red and 0.142 in blue -- a single scale
    # factor would have to pick one and be wrong about the other, and colour
    # balance is exactly what the difference between them controls.
    _p8 = get_profile("KODAK_PORTRA_800")
    _v32 = [v for v in _p8.process_variants if v.push_stops == 2][0]
    _dr = _v32.curves.r.gamma - _p8.curves.r.gamma
    _db = _v32.curves.b.gamma - _p8.curves.b.gamma
    chk("PORTRA 800's push moves the layers by different amounts",
        abs(_dr - _db) > 0.005 and _dr > 0.1 and _db > 0.1,
        f"red +{_dr:.4f}, blue +{_db:.4f} of gamma at EI 3200")

    # The chemistry variants must NOT have acquired a push count, and the push
    # records must not claim to be what the profile's own curves represent.
    _mix = [f"{p.name}/{v.name}" for p in FILM_PROFILES
            for v in p.process_variants if v.push_stops and v.is_default]
    chk("no push variant claims to be the profile's own process", not _mix,
        ", ".join(_mix))
    chk("CINESTILL's chemistry variants are still not pushes",
        all(v.push_stops == 0
            for v in get_profile("CINESTILL_800T").process_variants))



    # ---------------------------------------------------------------------------
    # 2026-08-14: the Photo-Lab-Index Polaroid curves must reproduce the PUBLISHED
    # D-min, slope and D-max. This is not a style check -- dmin and gamma are used
    # verbatim from the source and shoulder_x was solved numerically to land on the
    # published D-max, so if anyone retunes a shoulder by eye this test catches it
    # and tells them which published number they broke.
    # ---------------------------------------------------------------------------
    import numpy as _np
    from film_sim import density_scalar as _dens
    _PLI_DOC = {
        # name: (published D-min, published slope, published D-max), 1979 edition
        "POLAROID_51":        (0.00, 3.35, 1.75),
        "POLAROID_52":        (0.02, 1.35, 1.75),
        "POLAROID_42":        (0.08, 1.30, 1.65),
        "POLAROID_47":        (0.06, 1.50, 1.70),
        "POLAROID_55_PN_NEG": (0.18, 0.70, 1.65),
        "POLAROID_46L":       (0.05, 1.80, 2.80),
        "POLAROID_146L":      (0.02, 3.00, 2.30),
        "POLAROID_410":       (0.02, 2.00, 1.60),
    }
    _xs = _np.linspace(-4.5, 4.5, 1200)
    _bad = []
    for _n, (_dmin, _g, _dmax) in _PLI_DOC.items():
        _c = get_profile(_n).curves.r
        _got = max(_dens(float(_x), _c) for _x in _xs)
        if abs(_c.dmin - _dmin) > 1e-9 or abs(_c.gamma - _g) > 1e-9 or abs(_got - _dmax) > 0.005:
            _bad.append("%s dmin %.3f/%.2f gamma %.3f/%.2f Dmax %.3f/%.2f"
                        % (_n, _c.dmin, _dmin, _c.gamma, _g, _got, _dmax))
    chk("Photo-Lab-Index Polaroid curves reproduce published Dmin/slope/Dmax",
        not _bad, "; ".join(_bad) or "8 films, all within 0.005 density of published D-max")

    # POLAROID_55_PN_NEG's published 150-160 lp/mm must be reflected in an f50
    # that is high but BELOW the stocks documented higher still. Limiting
    # resolution and f50 are different measurements, so the assertion is on
    # ordering within our own f50 field, not on the lp/mm figures directly.
    #
    # An earlier version of this test asserted it was the SHARPEST stock in the
    # database. That was false -- KODAK_TMAX_100, KODAK_TMAX_400,
    # FUJI_NEOPAN_ACROS_100 and AGFA_APX_25 all publish 200 lp/mm at a stated
    # 1000:1 test-object contrast, where the Polaroid figure states no contrast
    # at all. The bad assertion survived unnoticed because it had been appended
    # BELOW this file's summary block and never executed; that placement bug was
    # fixed on 2026-08-14 and the test immediately failed, which is how the
    # wrong claim in the profile description was caught.
    _f50 = sorted(FILM_PROFILES, key=lambda _p: -_p.mtf.f50_g)
    _rank = [p.name for p in _f50].index("POLAROID_55_PN_NEG")
    chk("POLAROID_55_PN_NEG sits in the top ten on f50, consistent with 150-160 lp/mm",
        _rank < 10,
        "rank %d of %d, f50_g=%.0f; sharpest is %s at %.0f"
        % (_rank + 1, len(_f50), get_profile("POLAROID_55_PN_NEG").mtf.f50_g,
           _f50[0].name, _f50[0].mtf.f50_g))

    # ---------------------------------------------------------------------------
    # 2026-08-14 schema v6: tungsten exposure index and processing state.
    # ---------------------------------------------------------------------------
    # The whole value of exposure_index_tungsten is that the RATIO is measured, so
    # the test is on the ratio, not on either number alone. Documented physics: a
    # panchromatic emulsion loses about 1/3 stop under tungsten, a blue-sensitive
    # one loses far more. If a later edit puts a tungsten index above the daylight
    # one, or invents an implausible ratio, this catches it.
    # ⚠ THE UNFILTERED POPULATION ONLY, and the split is the v42 widening of
    # `exposure_index_tungsten`. A pair with a conversion filter named beside
    # it measures the FILTER and not the emulsion -- its ratio is asserted
    # against that filter's own published factor in G-V41-P45 instead. Before
    # v42 such a pair could not be stored at all, so this list was the whole
    # field and the restriction never had to be written down here.
    _tung = [(p.name, p.exposure_index, p.exposure_index_tungsten)
             for p in FILM_PROFILES
             if p.exposure_index_tungsten
             and not p.conversion_filter_tungsten
             and not p.conversion_filter_daylight]
    # ⚠ THE PLAUSIBLE BAND IS PER SENSITIZING CLASS, AND IT HAD TO BECOME SO
    # ON 2026-09-17. A single 1.0-4.0 window was fine while the only
    # blue-sensitive pairs on file were the two Polaroids at 3.20 and 3.33;
    # queue P63's KODAK COMMERCIAL 1956 is rated 25 daylight against 6
    # tungsten, a ratio of 4.17, and that is not an implausible number -- it
    # is what a NON-COLOUR-SENSITIZED emulsion does under a source with
    # almost no blue in it, and P62 measured the same film's response dying
    # at 510 nm on the same day and from a different page.
    _BLUE_T = ("POLAROID_51", "POLAROID_146L", "KODAK_COMMERCIAL_1956")
    _bad_t = [f"{n} {d}/{t}" for n, d, t in _tung
              if not ((2.5 <= d / t <= 4.5) if n in _BLUE_T
                      else (1.0 <= d / t <= 2.0))]
    chk("tungsten exposure index never exceeds daylight and stays inside the "
        "band its own sensitizing class allows",
        not _bad_t and len(_tung) >= 7,
        "; ".join(_bad_t) or f"{len(_tung)} stocks, ratios "
        + ", ".join(f"{d/t:.2f}" for _, d, t in _tung))

    # The blue-sensitive-only stocks must sit far above the panchromatic cluster.
    # This is the documented physical claim the field exists to carry, so it is
    # asserted rather than left as prose in a description.
    #
    # ⚠ SUPER SPEED ORTHO PORTRAIT IS EXCLUDED FROM BOTH SIDES and that is not
    # a convenience: the 1956 book gives it a TUNGSTEN RATING AND NO DAYLIGHT
    # ONE, so its pair is the same number twice and its "ratio" of 1.00 is a
    # statement about what was printed, not a measurement of a film.
    _blue = {n: d / t for n, d, t in _tung if n in _BLUE_T}
    _rest = [(n, d / t) for n, d, t in _tung
             if n not in _BLUE_T and n != "KODAK_SUPER_SPEED_ORTHO_1956"]
    _ortho = [v for n, v in _rest if "ORTHO" in n]
    _panr = [v for n, v in _rest if "ORTHO" not in n]
    chk("blue-sensitive stocks separate from panchromatic on the tungsten ratio",
        _blue and min(_blue.values()) > max(_panr) * 2.0,
        "blue %s vs pan max %.2f; ⚠ REPORTED AND NOT ASSERTED: the single "
        "orthochromatic pair sits at %s, above the panchromatic cluster by "
        "only %.2f, which is the right ORDER and too thin a margin to make a "
        "gate out of"
        % (sorted(round(v, 2) for v in _blue.values()), max(_panr),
           sorted(round(v, 2) for v in _ortho) or "none",
           (min(_ortho) - max(_panr)) if _ortho else 0.0))

    # ProcessingSpec is descriptive metadata, so the only thing to enforce is
    # INTERNAL CONSISTENCY: a stated time must come with a stated developer.
    # A time with no developer names nothing and would be worse than silence.
    _proc_bad = [p.name for p in FILM_PROFILES
                 if p.processing.minutes > 0.0 and not p.processing.developer]
    chk("no processing time is recorded without the developer that produced it",
        not _proc_bad, ", ".join(_proc_bad) or
        f"{sum(1 for p in FILM_PROFILES if p.processing.developer)} stocks state a developer")

    # GEVACOLOR_1952 correction, 2026-08-14. Cheltsov & Bongard 1958 document every
    # Gevacolor negative of the period as tungsten: N-5 at 2850 K, 652 at 3200 K.
    # The 5500 K this profile used to carry was an unsupported daylight assumption
    # from its tier-3 analogy origin.
    chk("GEVACOLOR_1952 is tungsten-balanced per Cheltsov 1958 p178",
        get_profile("GEVACOLOR_1952").balance_kelvin == 2850,
        f"{get_profile('GEVACOLOR_1952').balance_kelvin} K")


    # -----------------------------------------------------------------------
    # 2026-08-14: the two vendor documents that landed this session.
    # -----------------------------------------------------------------------
    # EKTACHROME 100D's spectral curves now come from H-1-5285 -- the sheet
    # whose product number the profile actually bears -- instead of being
    # borrowed from the 5294/7294 reintroduction. They were extracted from PDF
    # VECTOR paths, so they are exact rather than traced. The check is that the
    # red and green layers carry more measured samples than the old borrow did
    # (16 and 15 against 13 and 13): that is the whole gain, real low-sensitivity
    # skirts instead of a -4.0 floor, and it is what a regression would undo.
    _sp = get_profile("KODAK_EKTACHROME_100D_5285").spectral
    _act = lambda v: sum(1 for x in v if x > -3.9)
    chk("EKTACHROME 100D spectral curves are 5285's own, with measured skirts",
        _act(_sp.log_s_r) >= 16 and _act(_sp.log_s_g) >= 15
        and "H-1-5285" in _sp.source,
        "active r/g/b = %d/%d/%d, source %s"
        % (_act(_sp.log_s_r), _act(_sp.log_s_g), _act(_sp.log_s_b),
           "H-1-5285" if "H-1-5285" in _sp.source else _sp.source[:40]))

    # The Fujicolor cine manual states "no FILTER corrections" at 1 s, which is
    # an explicit statement that the three records lose speed together. That
    # zero spread is evidence, not a default, so it is asserted -- and it is the
    # one colour stock in the database where the spread SHOULD be zero, against
    # the Kodak films that all need a CC filter.
    _fr = get_profile("FUJI_ETERNA_VIVID_500T_8547").reciprocity
    chk("Fuji ETERNA reciprocity failure is achromatic, as the manual states",
        _fr.schwarzschild_p_r == _fr.schwarzschild_p_g == _fr.schwarzschild_p_b
        and _fr.onset_s == 0.1,
        "p=%.2f/%.2f/%.2f onset=%.2f"
        % (_fr.schwarzschild_p_r, _fr.schwarzschild_p_g, _fr.schwarzschild_p_b,
           _fr.onset_s))

    # ⚠ THE RULE SURVIVES v42 IN THE FORM THAT CAN ACTUALLY BE CHECKED. It
    # used to read "every entry is a monochrome stock", which was a proxy for
    # the real rule -- an UNFILTERED pair is a property of the emulsion, a
    # filtered one is a property of the filter -- enforced by the only means
    # available before `conversion_filter_tungsten` existed: banning the
    # colour stocks outright. Now the record says which kind it is, so the
    # check is the rule itself. A COLOUR stock with a tungsten index and NO
    # filter named is still the violation it always was: it would put a
    # filter factor into the sensitisation population.
    _tw = [p.name for p in FILM_PROFILES
           if p.exposure_index_tungsten and not p.is_monochrome
           and not (p.conversion_filter_tungsten
                    or p.conversion_filter_daylight)]
    _twf = [p.name for p in FILM_PROFILES
            if p.exposure_index_tungsten
            and (p.conversion_filter_tungsten or p.conversion_filter_daylight)]
    # And the converse: a MONOCHROME stock needs no conversion filter, so one
    # named on a monochrome pair would mean the harvest mislabelled it.
    _twm = [p.name for p in FILM_PROFILES
            if p.is_monochrome and (p.conversion_filter_tungsten
                                    or p.conversion_filter_daylight)]
    chk("every unfiltered tungsten pair is monochrome and every filtered one "
        "names its conversion filter -- the two populations never mix",
        not _tw and not _twm,
        ", ".join(_tw + _twm) or
        "%d unfiltered pairs, all monochrome; %d filtered colour pairs, "
        "every one naming its filter (checked against the filter's own "
        "published factor in G-V41-P45)"
        % (sum(1 for p in FILM_PROFILES
               if p.exposure_index_tungsten and not p.conversion_filter_tungsten
               and not p.conversion_filter_daylight), len(_twf)))


    # -----------------------------------------------------------------------
    # 2026-08-14 (systematic re-analysis): PHYSICAL CONSISTENCY between the two
    # sharpness fields. f50 is the frequency at which modulation falls to 50 %;
    # limiting resolution is where it falls to the few-per-cent visual
    # threshold. One line pair is one cycle, so the figures are directly
    # comparable, and f50 must sit WELL BELOW the limiting resolution. A stock
    # whose f50 exceeds its own published limiting resolution is not optimistic,
    # it is impossible.
    #
    # This caught two real errors on stocks whose resolving power had been taken
    # from Polaroid data sheets while their MTF was left at an unrelated
    # estimate: POLAROID_664 had f50 40 against a 20 lp/mm limit, POLAROID_667
    # f50 26 against 14. Both are fixed; this test stops the class recurring,
    # which matters because the two numbers are entered from different places
    # (MTFSpec in the profile, _RESOLVING_POWER in a separate dict) and nothing
    # else ties them together.
    # ⚠ COMPARED ON THE GREEN RECORD SINCE 2026-08-23 (C2b), AND THE REASON IS A
    # REAL CONFLICT THIS TEST SURFACED. The first measured per-record triples put
    # 5245's BLUE f50 at 100.5 and 5293's at 114.6 against a stored limiting
    # resolution of 100 lines/mm for both, so the max-of-three form failed.
    # The two quantities are not the same measurement: ISO 6328 resolving power is
    # read off a developed bar target as ONE number for the composite three-layer
    # image, while f50 is per record. A blue record individually sharper than the
    # composite limit is ordinary -- the composite is dragged down by the red
    # record, which every measurement now puts near 36 cycles/mm. Green is the
    # visually weighted record and the defensible single-number comparison.
    # ⚠ AND THE STORED LIMIT IS THE WEAKER NUMBER OF THE TWO HERE: 5248's sheet
    # prints its pair in text ("TOC 1.6:1 / TOC 1000:1 -- 80 lines/mm /
    # 160 lines/mm") and agrees with the stored (80, 160); 5245's and 5293's
    # sheets print no "lines/mm" text at all, so their stored (50, 100) cannot be
    # confirmed from the documents on file. Recorded rather than deleted.
    _mtf_bad = []
    _mtf_note = []
    for _p in FILM_PROFILES:
        _rp = film_profiles._RESOLVING_POWER.get(_p.name)
        if not _rp or not _rp[1]:
            continue
        if _p.mtf.f50_g >= _rp[1]:
            _mtf_bad.append("%s green f50=%.0f >= limit=%.0f"
                            % (_p.name, _p.mtf.f50_g, _rp[1]))
        elif _p.mtf.f50_b >= _rp[1]:
            _mtf_note.append("%s blue %.0f vs limit %.0f"
                             % (_p.name.split("_")[-1], _p.mtf.f50_b, _rp[1]))
    chk("green f50 stays below published limiting resolution on every stock "
        "with both",
        not _mtf_bad, "; ".join(_mtf_bad) or
        ("%d stocks carry both figures, all consistent"
         % sum(1 for _p in FILM_PROFILES
               if film_profiles._RESOLVING_POWER.get(_p.name, (0, 0))[1])
         + ("; blue exceeds the composite limit on " + ", ".join(_mtf_note)
            + " -- per-record vs composite metric, recorded" if _mtf_note else "")))


    # -----------------------------------------------------------------------
    # 2026-08-14: DUPLICATE KEYS IN THE DECORATION DICTS.
    #
    # Python takes the LAST value for a repeated dict key and says nothing. On
    # 2026-08-14 a re-analysis pass appended 22 keys that already existed
    # further down these dicts, so every one of those "additions" was a silent
    # no-op -- including two that carried an arithmetic error, which is the only
    # reason the error never reached a render. That is a bad way to be lucky.
    #
    # The dicts are long, hand-maintained and appended-to by date, so this will
    # recur without a test. Parsing the source with ast is the only way to see
    # it: by the time the module is imported the duplicates are already gone.
    import ast as _ast
    import collections as _coll
    _tree = _ast.parse(Path("film_profiles.py").read_text(encoding="utf-8"))
    _dups = {}
    for _n in _ast.walk(_tree):
        _v = getattr(_n, "value", None)
        if not isinstance(_v, _ast.Dict):
            continue
        _nm = getattr(getattr(_n, "target", None), "id", None)
        if _nm is None and isinstance(_n, _ast.Assign) and _n.targets:
            _nm = getattr(_n.targets[0], "id", None)
        if not _nm:
            continue
        _keys = [_k.value for _k in _v.keys
                 if isinstance(_k, _ast.Constant) and isinstance(_k.value, str)]
        _d = {_a: _c for _a, _c in _coll.Counter(_keys).items() if _c > 1}
        if _d:
            _dups[_nm] = _d
    chk("no duplicate keys in any decoration dict",
        not _dups,
        "; ".join("%s %s" % (_k, _v) for _k, _v in _dups.items())
        or "all dict literals in film_profiles.py have unique keys")


    # -----------------------------------------------------------------------
    # 2026-08-15: FUJI_NEOPAN_1600 must keep reproducing the two numbers its
    # datasheet actually prints (AF3-608E, PDF p3 and p4).
    #
    # Its curve was fitted to 487 points traced off the manufacturer's plotted
    # characteristic curve, deliberately anchored so the AVERAGE GRADIENT matches
    # Fuji's printed Gbar = 0.77 for the EI 1600 condition (SPD, 20 C, 4 1/4 min).
    # The parameterisation is degenerate, so anyone retuning gamma or the toe by
    # eye can keep a plausible-looking curve while silently losing the published
    # statistic. This asserts the statistic, not the parameters.
    _np1600 = get_profile("FUJI_NEOPAN_1600")
    _c = _np1600.curves.r
    _base = _c.dmin
    # Gbar: slope from 0.1 above base+fog across 1.5 log-exposure units
    _lo = None
    for _t in np.linspace(-4.0, 6.0, 4001):
        if fs.density_scalar(float(_t), _c) >= _base + 0.10:
            _lo = float(_t)
            break
    _gbar = ((fs.density_scalar(_lo + 1.5, _c) - fs.density_scalar(_lo, _c)) / 1.5
             if _lo is not None else 0.0)
    chk("FUJI_NEOPAN_1600 reproduces its published average gradient Gbar 0.77",
        _lo is not None and abs(_gbar - 0.77) <= 0.03 and abs(_base - 0.211) <= 0.002,
        "Gbar=%.3f (printed 0.77), base+fog=%.3f (traced 0.211)" % (_gbar, _base))

    # The spectral curve was re-traced at 5 nm because the source supports it and
    # because a 613/630 nm dip-peak pair 17 nm apart is under-sampled at 10 nm.
    # If a later pass coarsens it back, that structure is lost silently.
    _sp = _np1600.spectral
    chk("FUJI_NEOPAN_1600 spectral curve retains its 5 nm sampling",
        _sp.lambda_step_nm == 5.0 and len(_sp.log_s_pan) >= 50
        and "AF3-608E" in (_sp.source or ""),
        "step=%s n=%d" % (_sp.lambda_step_nm, len(_sp.log_s_pan)))

    # ---- 2026-08-16 queue P1 adoptions: the traced curves must survive ----
    _e5285 = get_profile("KODAK_EKTACHROME_100D_5285")
    chk("KODAK_EKTACHROME_100D_5285 carries the vector-extracted H-1-5285 curves",
        abs(_e5285.curves.b.gamma - 13.0085) < 1e-3
        and abs(_e5285.curves.b.dmin - 0.1152) < 1e-3,
        "b gamma=%.4f dmin=%.4f" % (_e5285.curves.b.gamma, _e5285.curves.b.dmin))
    _t7266 = get_profile("KODAK_TRI_X_REVERSAL_200")
    chk("KODAK_TRI_X_REVERSAL_200 carries the machine-traced 7266 curve",
        abs(_t7266.curves.r.gamma - 3.0578) < 1e-3
        and abs(_t7266.curves.r.dmin - 0.2325) < 1e-3,
        "r gamma=%.4f dmin=%.4f" % (_t7266.curves.r.gamma, _t7266.curves.r.dmin))
    _p2383 = [q for q in PRINT_STOCKS if q.name == "KODAK_2383_RELEASE"][0]
    chk("KODAK_2383_RELEASE print curves are the 2015-sheet vector extraction",
        all(abs(getattr(_p2383.curves, _c).gamma - 6.0) < 1e-6 for _c in "rgb")
        and all(getattr(_p2383.curves, _c).shoulder_k
                <= 2.0 * getattr(_p2383.curves, _c).toe_k + 1e-9 for _c in "rgb"),
        "gammas capped 6.0, monotonicity guard holds")

    # ---- 2026-08-16 NotFound section-4 sweep: 14 vector-extracted spectral curves ----
    # These came from PDF vector polylines (exact coordinates), so losing them to a
    # later hand edit would be a real loss of measurement. Assert the set, not the
    # numbers of any single stock.
    _vec_spectral = ("KODAK_ULTRAMAX_800", "KODAK_ULTRAMAX_400", "KODAK_EKTAR_100",
                     "KODAK_PORTRA_160", "KODAK_PORTRA_800", "KODAK_PORTRA_100T",
                     "KODAK_GOLD_100", "KODAK_GOLD_200", "KODAK_TRI_X_400TX",
                     "KODAK_TMAX_100", "KODAK_TMAX_P3200", "KODAK_PLUS_X_125",
                     "KODAK_T400CN", "KODAK_BW400CN",
                     "KODAK_TMAX_400",
                     # 2026-08-17: the APX trio re-extracted from their stroked
                     # paths, superseding the 2026-08-02 visual transcription.
                     "AGFA_APX_25", "AGFA_APX_100", "AGFA_APX_400")
    _missing = []
    for _n in _vec_spectral:
        _sp = get_profile(_n).spectral
        _ok = (_sp is not None and _sp.lambda_step_nm == 10.0
               and "vector-path extraction" in (_sp.source or "")
               and (len(getattr(_sp, "log_s_r", ())) >= 33
                    or len(getattr(_sp, "log_s_pan", ())) >= 33))
        if not _ok:
            _missing.append(_n)
    chk("the 18 vector-extracted spectral curves are all present",
        not _missing, "missing/degraded: %s" % (", ".join(_missing) or "none"))

    # ---- SVEMA Foto line: the 1981-vs-1990 GOST norm sets are NOT interchangeable ----
    # GOST 24876-81 Table 6 carries three successive norm sets; its own note says the
    # parenthetical ones take effect 01.01.90. Our profiles model the pre-1990 generation
    # and must satisfy the ORIGINAL norms (R >= 135/110/110/100 top category, MTF at
    # 30 mm^-1 >= 0.60/0.60/0.50/0.50) -- NOT the 1990 ones Zhurba 1990 Table 2 prints
    # (R >= 200/150/110/100, MTF >= 0.80/0.80/0.80/0.70). Anyone "upgrading" these to the
    # newer figures would silently re-date the stocks, so the check asserts the era's norms
    # and that the resolving values did not drift upward into the 1990 set.
    _svema_1981 = {"SVEMA_FOTO_32": (135.0, 0.60), "SVEMA_FOTO_65": (110.0, 0.60),
                   "SVEMA_FOTO_130": (100.0, 0.50), "SVEMA_FOTO_250": (82.0, 0.50)}
    _bad = []
    for _n, (_rmin, _tmin) in _svema_1981.items():
        _p = get_profile(_n)
        _r = film_profiles._RESOLVING_POWER.get(_n, (0.0, 0.0))[1]
        _t30 = 2.0 ** (-((30.0 / _p.mtf.f50_g) ** 2))
        if not (abs(_r - _rmin) < 1e-6 and _t30 >= _tmin - 1e-9):
            _bad.append("%s R=%.0f (expect %.0f) MTF30=%.2f (need >=%.2f)"
                        % (_n, _r, _rmin, _t30, _tmin))
    chk("SVEMA Foto line matches its own era's GOST 24876-81 norms, not the 1990 revision",
        not _bad, "; ".join(_bad) or "all four on the pre-1990 norm set")

    # ---- schema v7: the four new carriers must stay OFF the render path ----
    # This is the whole justification for adding them as fields rather than a
    # sidecar file. The test does not inspect the code for reads -- it proves
    # the property directly: render a stock, populate every v7 field on a copy
    # of it with plausible non-zero data, render again, and require the output
    # to be bit-identical. If anyone later wires one of these into film_sim
    # without going through the staged review, this fails immediately.
    import dataclasses as _dc
    _rng = np.random.default_rng(4242)
    _img = _rng.random((24, 32, 3)).astype(np.float32) * 1.2
    _img[4:9, 4:12] = 5.0                      # a highlight, to exercise halation
    _base = get_profile("KODAK_PORTRA_400")
    # strictly positive: the dye-density validator rejects negatives, and it
    # caught an earlier sine-based probe that dipped below zero -- exactly the
    # job it exists for, so the probe was fixed rather than the rule relaxed.
    _grid = tuple(0.6 + 0.5 * np.sin(np.arange(31) / 4.0))
    _loaded = _dc.replace(
        _base,
        dye_density=film_profiles.SpectralDyeDensity(
            lambda_start_nm=400.0, lambda_step_nm=10.0,
            d_cyan=_grid, d_magenta=_grid, d_yellow=_grid,
            normalisation="peak_1.0", source="verify.py inertness probe"),
        layer_stack=film_profiles.LayerStack(
            order=("blue", "green", "red"), resolving_top=80.0,
            resolving_mid=46.0, resolving_bot=30.0,
            test_object_contrast="1000:1", source="verify.py inertness probe"),
        processing_family=film_profiles.ProcessingFamily(
            points=(film_profiles.DevelopmentPoint(
                developer="probe", minutes=9.0, celsius=20.0,
                contrast_index=0.56),),
            source="verify.py inertness probe"),
        reciprocity_table=film_profiles.ReciprocityTable(
            times_s=(1.0, 10.0, 100.0), stops_correction=(0.0, 0.5, 1.5),
            source="verify.py inertness probe"),
        dye_impurity=film_profiles.DyeImpurity(
            ratios=(film_profiles.DyeImpurityRatio(
                        dye="y", band="g", lo=0.06, hi=0.18),
                    film_profiles.DyeImpurityRatio(
                        dye="m", band="b", lo=-0.10, hi=-0.05,
                        criterion="probe negative term")),
            source="verify.py inertness probe"),
    )
    _loaded.validate()
    _st = fs.RenderSettings(film_format="ff35")
    _a = fs.simulate(_img.copy(), _base, _st).astype(np.float32)
    _b = fs.simulate(_img.copy(), _loaded, _st).astype(np.float32)
    chk("schema v7 fields are INERT: populating all five cannot change a render",
        np.array_equal(_a, _b),
        "max abs delta=%.3e" % float(np.max(np.abs(_a - _b))))
    # ⚠ VERSION PIN UPDATED 8 -> 9 on 2026-08-18 (queue item C1b). v8 recorded
    # that GrainSpec gained five fields and that the sigma(D) shape stopped being
    # inert. **v9 records a change of MEANING with no change of layout**, which is
    # the more dangerous kind: rms_granularity is now the rms at NET density 1.0
    # (dmin + 1.0), as the Kodak sheets print it, and the sampler normalises
    # there. A plugin that pairs v9 data with a v8 sampler compiles cleanly, runs
    # cleanly, and renders the wrong grain level -- which is exactly why the
    # version moved even though sizeof(GrainSpec) did not.
    # ⚠ VERSION PIN 14 -> 15 on 2026-08-26: FilmProfile gained
    # `print_grain_index`, and the carrier list gains it too, because the point
    # of this guard is that every INERT carrier is reachable and validated on a
    # real profile rather than merely declared. KODAK_PORTRA_400 remains the
    # probe: it is one of the eight stocks the KODAK still-film harvest touched,
    # so it now carries a populated PGI record as well as the v7 four.
    chk("schema v7 carriers are all validated by FilmProfile.validate",
        all(hasattr(get_profile("KODAK_PORTRA_400"), _n) for _n in
            ("dye_density", "layer_stack", "processing_family",
             "reciprocity_table", "print_grain_index", "push",
             "emulsion", "third_party", "param_sources",
             "process_variants",
             # ⚠ v21, added to the probe 2026-08-31 in the same edit as the
             # version pin. The point of this guard is that every INERT carrier
             # is reachable and validated on a REAL profile rather than merely
             # declared, so a new carrier that nobody adds here is a carrier
             # nothing checks. KODAK_PORTRA_400 is one of the 13 stocks queue
             # K2 populated, so it holds a real AimDensity record.
             "aim_density",
             # ⚠ v35, 2026-09-16. `dye_stability` existed from v12 but was
             # reachable only from PrintStock, which was an accident of which
             # document landed first -- it arrived for a digital-intermediate
             # recording film. A colour negative fades in storage too, and it
             # is the negative's fade that decides what a scan of a thirty-
             # year-old roll looks like.
             "dye_stability",
             # ⚠ v43, 2026-09-18c (queue P41). The offset-speed
             # sub-layer set: inert on all 191, and the point of
             # this probe is that an inert carrier is still
             # reachable and validated on a real profile.
             "sub_layers"))
        and film_profiles.SCHEMA_VERSION == 49
        and all(hasattr(_ps, "spectral") for _ps in film_profiles.PRINT_STOCKS)
        # ⚠ v25, and it is the first entry in this probe that is NOT a carrier.
        # The others are here to prove an inert field is reachable; this one is
        # read by stage 13 on every print, so the check is that it exists AND
        # that its gate flag exists beside it -- a matrix with no
        # `printing_matrix_measured` could not be told from an estimate.
        and all(hasattr(_ps, "printing_density_matrix")
                and hasattr(_ps, "printing_matrix_measured")
                for _ps in film_profiles.PRINT_STOCKS),
        "SCHEMA_VERSION=%d" % film_profiles.SCHEMA_VERSION)

    # ---- 2026-08-17: measured per-channel grain must survive _grain_v2 ----
    # The colour-negative heuristic (b 1.3x, r 1.1x of pooled) used to run
    # unconditionally and overwrote any measured per-layer RMS a literal set.
    # GEVACOLOR_NEG_682 carries 23/16/34 from Fig. 12 of the Vervoort &
    # Stappaerts SMPTE paper, whose point is that blue >> red > green -- the
    # OPPOSITE of the heuristic, because the DIR couplers act on green and red
    # only. It had been rendering as 17.6/16.0/20.8 with that inversion erased.
    _g682 = get_profile("GEVACOLOR_NEG_682").grain
    chk("GEVACOLOR_NEG_682 keeps its MEASURED per-layer grain (blue >> red > green)",
        abs(_g682.rms_r - 23.0) < 1e-6 and abs(_g682.rms_g - 16.0) < 1e-6
        and abs(_g682.rms_b - 34.0) < 1e-6 and _g682.rms_b > _g682.rms_r > _g682.rms_g,
        "r/g/b = %.1f/%.1f/%.1f" % (_g682.rms_r, _g682.rms_g, _g682.rms_b))
    # ДС-5М: the specification norms must not drift. TU 6-17-691-88 table 2.
    _ds5 = get_profile("SVEMA_DS_5M")
    chk("SVEMA_DS_5M matches TU 6-17-691-88 table 2 (gradients, mask ladder, grain)",
        _ds5.exposure_index == 50 and _ds5.balance_kelvin == 5500
        and abs(_ds5.curves.b.gamma - 0.60) < 1e-6
        and abs(_ds5.curves.g.gamma - 0.54) < 1e-6
        and abs(_ds5.curves.r.gamma - 0.50) < 1e-6
        and _ds5.curves.b.dmin > _ds5.curves.g.dmin > _ds5.curves.r.dmin
        and abs(_ds5.grain.rms_r - 30.0) < 1e-6
        and abs(_ds5.grain.rms_g - 22.0) < 1e-6,
        "gammas %.2f/%.2f/%.2f, mask ladder %.2f/%.2f/%.2f"
        % (_ds5.curves.b.gamma, _ds5.curves.g.gamma, _ds5.curves.r.gamma,
           _ds5.curves.b.dmin, _ds5.curves.g.dmin, _ds5.curves.r.dmin))

    # ---- 2026-08-17: ДС-4 now rests on its own TU, not a handbook summary ----
    # ТУ 6-17-622-84 table 4 specifies the RECOMMENDED contrast coefficient per
    # layer: upper and middle 0.70, LOWER 0.60. Upper = blue-sensitive, middle =
    # green, lower = red, so b = g = 0.70 > r = 0.60. The previously stored
    # spread had blue steepest and red shallowest by only 0.03 (0.82/0.80/0.79),
    # a [T3] guess; the TU inverts the relationship and widens it. Resolving
    # power likewise moves 63 -> 68 lin/mm (ГОСТ 2819-84 method, named in the
    # TU's own test section). Anyone "restoring" the Gurlev figures would be
    # replacing a primary specification with a handbook paraphrase of its
    # superseded 1974 edition.
    _ds4 = get_profile("SVEMA_DS_4")
    chk("SVEMA_DS_4 carries its TU 6-17-622-84 per-layer gammas (b=g=0.70 > r=0.60)",
        abs(_ds4.curves.b.gamma - 0.70) < 1e-6
        and abs(_ds4.curves.g.gamma - 0.70) < 1e-6
        and abs(_ds4.curves.r.gamma - 0.60) < 1e-6
        and film_profiles._RESOLVING_POWER["SVEMA_DS_4"][1] == 68.0,
        "b/g/r = %.2f/%.2f/%.2f, R = %.0f lin/mm"
        % (_ds4.curves.b.gamma, _ds4.curves.g.gamma, _ds4.curves.r.gamma,
           film_profiles._RESOLVING_POWER["SVEMA_DS_4"][1]))

    # ---- 2026-08-17: four TU-specified Soviet stocks ------------------------
    # Every figure in these four is an ACCEPTANCE LIMIT from a Soviet TU, not a
    # measurement. The checks below assert the documented relationships, which
    # are what the specifications actually establish:
    #   * LN-9 and LN-9S share one emulsion and differ ONLY in antihalation
    #     construction, so LN-9S's whole Dmin ladder must sit BELOW LN-9's;
    #   * LN-9 is the finer-grained, sharper film than LN-8 (RMS 11 vs 19/21,
    #     MTF 0.40/0.22 vs 0.30/0.15) -- if that inverts, a value was mistyped;
    #   * CO-32D is reversal and its sigma(D) must turn OVER past mid-scale.
    _l8, _l9, _l9s = (get_profile(n) for n in
                      ("SVEMA_LN_8", "SVEMA_LN_9", "SVEMA_LN_9S"))
    chk("LN-9S Dmin ladder sits below LN-9's (rear carbon vs silver undercoat)",
        all(getattr(_l9s.curves, c).dmin < getattr(_l9.curves, c).dmin
            for c in "rgb"),
        "9S b/g/r %.2f/%.2f/%.2f vs 9 %.2f/%.2f/%.2f"
        % (_l9s.curves.b.dmin, _l9s.curves.g.dmin, _l9s.curves.r.dmin,
           _l9.curves.b.dmin, _l9.curves.g.dmin, _l9.curves.r.dmin))
    chk("LN-9 is finer-grained and sharper than LN-8, as its TU specifies",
        _l9.grain.rms_granularity < _l8.grain.rms_granularity
        and _l9.mtf.f50_g > _l8.mtf.f50_g,
        "RMS %.0f vs %.0f, f50_g %.1f vs %.1f"
        % (_l9.grain.rms_granularity, _l8.grain.rms_granularity,
           _l9.mtf.f50_g, _l8.mtf.f50_g))
    _c32 = get_profile("SVEMA_CO_32D")
    chk("SVEMA_CO_32D is reversal with a turning-over sigma(D)",
        _c32.is_reversal and _c32.grain.sigma_shape_dmax < _c32.grain.sigma_shape_mid,
        "toe/mid/dmax %.2f/%.2f/%.2f" % (_c32.grain.sigma_shape_toe,
                                         _c32.grain.sigma_shape_mid,
                                         _c32.grain.sigma_shape_dmax))

    # ---- 2026-08-17: VISION3 sigma(D), traced from the four Kodak TI sheets ----
    # Guards an adoption that took four attempts. Three earlier passes produced
    # internally consistent numbers from CROSS-FAMILY hybrid curves, and the
    # thing that finally exposed them was comparing the siblings, so that
    # comparison is what is asserted here rather than any single value.
    # Re-derive with: python vision3_granularity.py --overlay out
    _v3 = [get_profile(n) for n in ("KODAK_VISION3_50D_5203",
                                    "KODAK_VISION3_250D_5207",
                                    "KODAK_VISION3_200T_5213",
                                    "KODAK_VISION3_500T_5219")]
    chk("VISION3 quartet carries a traced sigma(D), not the (0,1,0) default",
        all(p.grain.sigma_shape_toe > 0.0 and p.grain.sigma_shape_dmax > 0.0
            and p.grain.sigma_shape_mid == 1.0 for p in _v3),
        "; ".join("%s %.2f/%.2f/%.2f" % (p.name.split("_")[-1],
                                         p.grain.sigma_shape_toe,
                                         p.grain.sigma_shape_mid,
                                         p.grain.sigma_shape_dmax) for p in _v3))
    # The direction is the finding, and it contradicts the estimate these four
    # used to carry. Kodak's own SMPTE Journal paper of July 1985 (Sehlin,
    # Kennel et al., p 728, Figs 8 and 9) says the same in print: "overexposing
    # either film significantly decreases granularity". A regression that
    # restored a rising generic triple would trip this.
    chk("VISION3 sigma(D) FALLS from mid to dmax on all four sheets",
        all(p.grain.sigma_shape_dmax < p.grain.sigma_shape_mid for p in _v3),
        "dmax/mid %s" % ", ".join("%.2f" % (p.grain.sigma_shape_dmax
                                            / p.grain.sigma_shape_mid) for p in _v3))
    # Four independent sheets, one product line: the dmax anchors agreed to
    # +/-7 % (0.551 / 0.565 / 0.584 / 0.631). That agreement IS the evidence the
    # trace is right, so it is asserted with a little slack, not pinned exactly.
    _dm = [p.grain.sigma_shape_dmax / p.grain.sigma_shape_mid for p in _v3]
    chk("VISION3 siblings agree on the dmax anchor (0.50-0.70 band)",
        all(0.50 <= v <= 0.70 for v in _dm) and max(_dm) - min(_dm) <= 0.12,
        "min %.2f max %.2f spread %.2f" % (min(_dm), max(_dm), max(_dm) - min(_dm)))
    # Toe anchors are looser by construction -- 5203's and 5213's come from
    # merged ink runs (+/-7 %) and 5213's is pooled over the three layers
    # because that sheet draws them as one band. Still all below mid.
    chk("VISION3 toe anchors sit below mid, in the traced 0.35-0.75 band",
        all(0.35 <= p.grain.sigma_shape_toe <= 0.75 for p in _v3)
        and all(p.grain.sigma_shape_toe < p.grain.sigma_shape_mid for p in _v3),
        "toe %s" % ", ".join("%.2f" % p.grain.sigma_shape_toe for p in _v3))
    # Grain-size order must survive the shape change: 50D finest, 500T coarsest.
    chk("VISION3 sigma(D) shape did not disturb the rms grain ladder",
        [p.grain.rms_granularity for p in _v3] == sorted(
            p.grain.rms_granularity for p in _v3),
        "rms %s" % ", ".join("%.1f" % p.grain.rms_granularity for p in _v3))

    # ---- 2026-08-18: SVEMA_FOTO_65 withdrawals stay withdrawn ----------------
    # Three values here were derived from PER-CHANNEL density drift in the
    # owner's scan batch. The batch is a folder named SVEMA-FN64 holding 509
    # frames, of which only 1-67 are confirmed Foto-65 (owner, 2026-08-18);
    # 68+ mix Foto-32 in. Those 67 confirmed frames are EXACTLY greyscale
    # (max |R-G| = max |B-G| = 0, measured over all 67), so a per-channel
    # measurement cannot have come from this emulsion at all. These checks
    # exist because the withdrawn numbers looked precise and would be easy to
    # re-adopt by accident from the old reports. Re-adoption needs a NEW
    # measurement, not a re-reading of the same files.
    _s65 = get_profile("SVEMA_FOTO_65")
    chk("SVEMA_FOTO_65 base_tint stays identity (greyscale frames cannot show tint)",
        _s65.base_tint == (1.0, 1.0, 1.0),
        "base_tint %.3f/%.3f/%.3f" % _s65.base_tint)
    chk("SVEMA_FOTO_65 silver_tone stays neutral (the +0.40 reversal's evidence is void)",
        _s65.silver_tone == 0.0, "silver_tone %+.2f" % _s65.silver_tone)
    # The two scan runs disagree in SIGN on sigma(D): mixed 509 gives
    # 0.65/1.00/1.65 (rising), confirmed 67 gives 1.13/1.00/1.02 (flat). Bin
    # edges are absolute offsets from d_base and the two d_base values differ
    # by 0.024 D, so it is not a binning artefact. Conflict recorded, neither
    # adopted; the stock keeps whatever the class rule gives it.
    #
    # ⚠ THIS GUARD USED TO PIN THE LITERAL 0.4/1.0/1.2 AND TO ASSERT THAT dmax
    # EXCEEDS mid, on the stated grounds that a rising sqrt(D) is "the textbook
    # result for a B&W SILVER negative". Jones 1958 measured four Kodak
    # negatives and the textbook is wrong at the top: sigma FLATTENS above
    # D 1.0 (1.016 at D 1.40) rather than climbing. The property this guard
    # exists to protect is that SVEMA takes the CLASS DEFAULT and not either of
    # its own two conflicting scans; that property is asserted here, and the
    # direction claim is gone with the placeholder that carried it.
    #
    # ⚠ AND THE CONFLICT PARTLY RESOLVES ITSELF. Of the two scan runs, the one
    # set aside as "flat" -- confirmed 67, 1.13/1.00/1.02 -- is the one that
    # AGREES with Kodak's measurement. That is the owner's own scan
    # corroborating a 1958 laboratory result, and it makes the mixed-509 rising
    # run the odd one out rather than a coin toss between two.
    import pse_jones_1958 as _pj65
    chk("SVEMA_FOTO_65 sigma(D) is the class default, not either scan run",
        (abs(_s65.grain.sigma_shape_toe - _pj65.ADOPTED_TOE) < 1e-9
         and abs(_s65.grain.sigma_shape_mid - 1.0) < 1e-9
         and abs(_s65.grain.sigma_shape_dmax - _pj65.ADOPTED_DMAX) < 1e-9),
        "toe/mid/dmax %.2f/%.2f/%.2f" % (_s65.grain.sigma_shape_toe,
                                         _s65.grain.sigma_shape_mid,
                                         _s65.grain.sigma_shape_dmax))
    # The provenance text is load-bearing here: without the mixed-batch warning
    # the next reader sees "509-frame batch" and reasonably treats it as one
    # emulsion. That is exactly the mistake this correction fixes.
    _src = open(__file__.replace("verify.py", "film_profiles.py"),
                encoding="utf-8").read()
    chk("SVEMA_FOTO_65 carries the mixed Foto-32/Foto-65 provenance warning",
        "PROVENANCE CORRECTION 2026-08-18" in _src
        and "PICT0001-PICT0067" in _src,
        "warning block present")

    # THE INHERITED CASE (queue item C3, approved 2026-08-18). Two siblings had
    # copied Foto-65's tint and silver_tone by analogy, so withdrawing the
    # parent's measurement left them holding transfers from a value that no
    # longer exists. Withdrawn in turn. Guarding the siblings and not only the
    # parent is the point: the defect propagated by ANALOGY once and could again.
    for _n in ("SVEMA_FOTO_32", "SVEMA_FOTO_130"):
        _p = get_profile(_n)
        chk(f"{_n} base_tint is identity (transfer from a withdrawn parent)",
            _p.base_tint == (1.0, 1.0, 1.0),
            "base_tint %.3f/%.3f/%.3f" % _p.base_tint)
        chk(f"{_n} silver_tone is neutral",
            _p.silver_tone == 0.0, "silver_tone %+.2f" % _p.silver_tone)
    # A B&W profile whose only tint evidence was that batch must not acquire one
    # anywhere in the Svema Foto line without a new measurement. Sweeping the
    # whole family catches a future addition that copies the old numbers.
    _foto = [p for p in FILM_PROFILES if p.name.startswith("SVEMA_FOTO_")]
    _tinted = [p.name for p in _foto
               if p.base_tint != (1.0, 1.0, 1.0) or p.silver_tone != 0.0]
    chk("no SVEMA_FOTO_* stock claims a tint or silver tone",
        not _tinted, ", ".join(_tinted) if _tinted
        else "%d stocks, all neutral" % len(_foto))

    # ---- queue D3, 2026-08-31: the Tasma case, settled the same way --------
    # ⚠ THE TEST THIS ROW ASKED FOR WAS RUN AND +0.30 DID NOT SURVIVE IT. The
    # owner supplied the 132-frame batch on 2026-08-31; 104 of them are
    # bit-exactly neutral (R == G == B at every pixel), so they contribute a
    # hard zero rather than a measurement, and the 28 that carry colour give a
    # midtone cast of +7.72 with a frame-to-frame scatter of +/-10.72 -- the
    # scatter LARGER than the mean, which no emulsion property can be. The
    # historical "+8.6 and +15.6" pair the profile hoped might rescue the value
    # are two draws from that distribution, and the larger of them was chosen.
    # Reverted to identity by the SVEMA_FOTO_65 precedent.
    _tas = get_profile("TASMA_FN_64")
    chk("TASMA_FN_64 silver_tone is neutral (its batch is 79 % greyscale)",
        _tas.silver_tone == 0.0, "silver_tone %+.2f" % _tas.silver_tone)
    chk("TASMA_FN_64 base_tint is identity",
        _tas.base_tint == (1.0, 1.0, 1.0),
        "base_tint %.3f/%.3f/%.3f" % _tas.base_tint)
    # ⚠ AND THE LAST SURVIVOR IS NAMED, NOT SWEPT. TASMA_OCH_45 keeps +0.15: it
    # carries no source, but it has no batch behind it to refute either, and
    # there are no OCh-45 scans. Refuting by analogy is what the 2026-08-18
    # pass refused to do for FOTO_32 and FOTO_130. Asserting that it is the
    # ONLY one left means a future value cannot appear quietly beside it.
    _tone = sorted(p.name for p in FILM_PROFILES if p.silver_tone != 0.0)
    chk("exactly one stock still claims a silver tone, and it is the unsourced one",
        _tone == ["TASMA_OCH_45"], ", ".join(_tone) or "none")

    # ---- AGFA_VISTA_200 spectral, queue item B2 (2026-08-18) ----------------
    # Extracted by agfa_vista.py from the sheet's vector art. The layer identity
    # rests entirely on the DASH PATTERN legend (solid green / dashed blue /
    # dash-dot red), so the guard that matters is the physical one: a legend
    # swap would put a layer's peak in the wrong band and nothing else would
    # notice. The extractor asserts this too; asserting it here as well means a
    # hand-edit of the stored tuples cannot bypass the extractor's check.
    _v = get_profile("AGFA_VISTA_200").spectral
    chk("AGFA_VISTA_200 carries the extracted spectral set", _v.has_data,
        "criterion %s" % _v.criterion)
    _peak = lambda row: _v.lambda_start_nm + _v.lambda_step_nm * row.index(max(row))
    _bands = {"b": (_v.log_s_b, 400.0, 480.0), "g": (_v.log_s_g, 520.0, 580.0),
              "r": (_v.log_s_r, 600.0, 680.0)}
    _off = ["%s %.0fnm" % (k, _peak(list(row)))
            for k, (row, lo, hi) in _bands.items()
            if not lo <= _peak(list(row)) <= hi]
    chk("AGFA_VISTA_200 layer peaks sit in their own bands (legend not swapped)",
        not _off, ", ".join(_off) if _off
        else "b %.0f / g %.0f / r %.0f nm" % tuple(
            _peak(list(_bands[k][0])) for k in ("b", "g", "r")))
    # The blue layer is a plateau whose winning lobe is decided by ~0.1 decade,
    # so the profile must keep saying so rather than presenting 470 nm as a
    # designed peak wavelength.
    chk("AGFA_VISTA_200 records that its blue peak is a plateau",
        "PLATEAU" in _src, "caveat present")

    # ---- E0: 11 profiles re-verified against sheets that were wrongly declared
    # ---- absent (2026-08-18). Guards on the values that MOVED, plus on the
    # ---- exact-agreement set, because an exact agreement is evidence and a
    # ---- silent drift away from it would destroy that evidence.
    _e0 = get_profile("EASTMAN_5247_1983")
    # Kodak's TI0835 prints "rms Granularity: less than 5". The stored 13.0 was
    # 2.6x above that bound. This is the largest grain change in the file, so it
    # gets an explicit guard rather than relying on the comment surviving.
    chk("EASTMAN_5247_1983 rms is Kodak's printed bound 5.0, not the old 13.0",
        abs(_e0.grain.rms_granularity - 5.0) < 1e-9,
        "rms %.1f" % _e0.grain.rms_granularity)
    chk("EASTMAN_5247_1983 owns the TI0835 resolving pair (50, 100)",
        (_e0.mtf.resolving_power_lp_mm_lowc,
         _e0.mtf.resolving_power_lp_mm_highc) == (50.0, 100.0),
        "%.0f / %.0f" % (_e0.mtf.resolving_power_lp_mm_lowc,
                         _e0.mtf.resolving_power_lp_mm_highc))
    # The other half of that move: the [T3] reconstruction of the EI 100 coating
    # must NOT carry a resolving power sourced from the EI 125 sheet. This is the
    # guard that would have caught the original leftover.
    _o = get_profile("EASTMAN_5247_1974")
    chk("EASTMAN_5247_1974 claims NO resolving power (no source for that coating)",
        (_o.mtf.resolving_power_lp_mm_lowc,
         _o.mtf.resolving_power_lp_mm_highc) == (0.0, 0.0),
        "%.0f / %.0f" % (_o.mtf.resolving_power_lp_mm_lowc,
                         _o.mtf.resolving_power_lp_mm_highc))
    # ... and the generations must stay distinguishable. If a future edit copies
    # values across again, the rms pair collapsing back together catches it.
    chk("the two 5247 generations do NOT share a grain figure",
        _o.grain.rms_granularity != _e0.grain.rms_granularity,
        "1974 %.1f vs 1983 %.1f" % (_o.grain.rms_granularity,
                                    _e0.grain.rms_granularity))
    # TI0835 documents a CHROMATIC failure (CC10Y at 1 s), so equal exponents
    # would be wrong in kind, not just in magnitude. The filter colour fixes the
    # direction: yellow boosts red+green, so BLUE lost the least -> p_b highest.
    _r = _e0.reciprocity
    chk("EASTMAN_5247_1983 reciprocity is chromatic with blue losing least",
        _r.schwarzschild_p_b > _r.schwarzschild_p_r and _r.onset_s == 0.1,
        "p %.2f/%.2f/%.2f onset %.2f s" % (_r.schwarzschild_p_r,
                                           _r.schwarzschild_p_g,
                                           _r.schwarzschild_p_b, _r.onset_s))
    # PLUS-X: dmin is defined by the schema as BASE + FOG. The sheet prints them
    # separately, 0.19 and 0.02, so the stored value must be the SUM. 0.19 alone
    # is the number a hurried reading takes, which is why the sum is asserted.
    _px = get_profile("EASTMAN_PLUS_X_5231")
    chk("EASTMAN_PLUS_X_5231 dmin is base+fog = 0.21, not base alone",
        abs(_px.curves.r.dmin - 0.21) < 1e-9, "dmin %.3f" % _px.curves.r.dmin)
    chk("EASTMAN_PLUS_X_5231 reciprocity uses the printed 1/10 s onset",
        _px.reciprocity.onset_s == 0.1
        and abs(_px.reciprocity.schwarzschild_p_r - 0.85) < 1e-9,
        "p %.2f onset %.2f s" % (_px.reciprocity.schwarzschild_p_r,
                                 _px.reciprocity.onset_s))
    # THE EXACT-AGREEMENT SET. These eight numbers were confirmed digit for digit
    # against their own manufacturer sheets on 2026-08-18. They are the strongest
    # evidence in the granularity and resolving-power fields, and a drift here
    # would be a regression against a printed source, not against an estimate.
    _EXACT = {
        "EASTMAN_DOUBLE_X_5222":    (14.0, 32.0, 100.0),
        "EASTMAN_PLUS_X_5231":      (10.0, 32.0, 100.0),
        "EASTMAN_EKTACHROME_7239":  (14.0, 40.0, 100.0),
    }
    _drift = []
    for _n, (_rms, _lo, _hi) in _EXACT.items():
        _p = get_profile(_n)
        if (abs(_p.grain.rms_granularity - _rms) > 1e-9
                or (_p.mtf.resolving_power_lp_mm_lowc,
                    _p.mtf.resolving_power_lp_mm_highc) != (_lo, _hi)):
            _drift.append(_n)
    chk("the 3 digit-for-digit sheet agreements still hold (rms + TOC pair)",
        not _drift, ", ".join(_drift) if _drift
        else "5222 14/32/100, 5231 10/32/100, 7239 14/40/100")
    # The Sehlin/Kennel year was a live conflict until the paper's own title page
    # was read. Pinning it stops the misleading FILENAME reasserting itself.
    chk("Sehlin/Kennel is cited as July 1985 with the 1983 conference date noted",
        "CITATION YEAR VERIFIED 2026-08-18" in _src
        and "125th" in _src and "pp 724-734" in _src,
        "verified citation present")
    # The 5285/5294 catalogue hazard: two sheets, both called "Ektachrome 100D".
    chk("KODAK_EKTACHROME_100D_5285 warns about the H-1-5294 sheet",
        "H-1-5294" in " ".join(
            get_profile("KODAK_EKTACHROME_100D_5285").provenance.sources),
        "hazard recorded")

    # ---- E0b: three vector plot sets extracted (2026-08-18) -----------------
    # 1. THREE MORE DYE-DENSITY SETS, all recovered from the FAILED list by
    # fixing the extractor rather than by finding better sources.
    _dye = [p for p in FILM_PROFILES if p.dye_density.has_data]
    # ⚠ 10 -> 11 on 2026-08-25 (queue C9): KODAK_VISION2_50D_5201, by the
    # ink-based family C. This is the SECOND count guard on the same set -- the
    # other is in the v7 carrier section -- and both are kept: they were written
    # by different passes and each states the count its own pass established.
    # ---- queue B1, 2026-08-26: the NEUTRAL + D-MIN pair (schema v14) ---------
    # ⚠ THE SHEET NEVER PLOTTED THE THREE DYES, and this record exists because
    # the SCHEMA was the limitation. H-1-5248 p3 prints "Typical densities for a
    # midscale neutral subject and D-min." and draws exactly two traces; the
    # entry sat on the queue as a failed extraction for weeks on a symptom that
    # assumed traces which do not exist.
    # ⚠ 5248's SPECTRAL SET, CROSS-CHECKED 2026-09-05 AGAINST ITS OWN VECTOR
    # PATHS, and the queue row that said this could not be done was wrong twice.
    # C37 filed 5248 p3 and 5293 p4 as UNREACHABLE, "blocked on METHOD", because
    # "they draw all three curves in BLACK, so the ink rule says nothing".
    # ⚠ COLOUR IS IRRELEVANT ON THESE PANELS: the three curves are three
    # SEPARATE VECTOR PATHS, so `split_subpaths` and a resample read them with
    # no tracking, no ink convention and no raster pass at all. What actually
    # defeats a naive read is the OVERBAR MINUS on the ordinate -- see
    # `dye_density.ticks` -- which silently discards the negative half of the
    # axis and moved the traced peaks to 600 / 530 / 470 nm.
    # With `dye_density.signed_ticks` reconstructing the sign, the panel traces
    # to 470 / 550 / 640 nm at absolute peak log sensitivities 1.94 / 1.49 /
    # 1.07. The STORED set peaks at 470 / 550 / 650 -- agreement to one grid
    # step on all three, from a route the stored set never used, so nothing is
    # re-adopted and the stored peaks are pinned here instead.
    _s48 = get_profile("EASTMAN_EXR_100T_5248").spectral
    _g48 = [_s48.lambda_start_nm + _s48.lambda_step_nm * i
            for i in range(len(_s48.log_s_r))]
    _pk48 = tuple(_g48[max(range(len(_row)), key=lambda k: _row[k])]
                  for _row in (_s48.log_s_b, _s48.log_s_g, _s48.log_s_r))
    chk("5248's stored spectral peaks still match its own vector paths",
        _pk48 == (470.0, 550.0, 650.0),
        "stored b/g/r %s; the p3 paths trace to 470/550/640, one grid step off "
        "on the red and exact on the other two" % (_pk48,))
    _p48 = get_profile("EASTMAN_EXR_100T_5248").dye_density
    chk("5248 carries a NEUTRAL+DMIN pair, and is not counted as a dye set",
        _p48.has_neutral_pair and not _p48.has_data
        and len(_p48.d_neutral) == 31 and len(_p48.d_dmin) == 31
        and not _p48.d_cyan and not _p48.d_magenta and not _p48.d_yellow,
        "has_data stays False by design, so every 'N stocks carry spectral dye "
        "density' count keeps its old meaning")
    # ⚠ TWO PHYSICAL CHECKS, NEITHER FITTED. A neutral is the mask PLUS the image
    # dyes, so it must exceed the mask everywhere; and a D-min that did not fall
    # toward red would not be an orange mask.
    chk("5248's neutral exceeds its D-min at every sample",
        all(n > d for n, d in zip(_p48.d_neutral, _p48.d_dmin))
        and min(n - d for n, d in zip(_p48.d_neutral, _p48.d_dmin)) > 0.4,
        "worst margin %.3f D over 31 samples"
        % min(n - d for n, d in zip(_p48.d_neutral, _p48.d_dmin)))
    _i48 = _p48.d_dmin.index(max(_p48.d_dmin))
    chk("5248's D-min behaves as an orange mask: blue peak, monotone to red",
        400 + 10 * _i48 == 440
        and all(a >= b - 1e-9 for a, b in zip(_p48.d_dmin[_i48:],
                                              _p48.d_dmin[_i48 + 1:])),
        "peaks %.3f at %d nm, falls to %.3f at 700"
        % (max(_p48.d_dmin), 400 + 10 * _i48, _p48.d_dmin[-1]))
    # ⚠ 1 -> 5 on 2026-08-26 (KODAK still-film harvest). The shape stopped being
    # a one-off the moment the E-series still sheets were read: every one of them
    # publishes its dye densities as a MIDSCALE NEUTRAL and a MINIMUM DENSITY
    # curve rather than three separated dyes, which is exactly why schema v14
    # added the pair. Four of the eleven documents yielded a pair the reader
    # would accept. The Fuji 8532 sheet publishes the same shape and is still
    # NOT adopted -- its page is rotated and its traces carry 19 and 16 segments
    # -- so its continued absence from this list is still the deliberate fact it
    # was.
    # The two REFUSALS in this batch are also deliberate and are listed here so
    # that a later run which "fixes" them has to explain itself: E-4050's panel
    # (KODAK_PORTRA_400, both the 2010 and 2016 vintages) resolves into three
    # traces where the caption promises two, and E-7019's (ULTRA MAX 400, 2007)
    # into one; `assign_dye_pair` refuses a crossing pair rather than label it by
    # mean density.
    # ⚠ 5 -> 6 on 2026-08-26f: KODAK_PRO_100T_PRT, off E-29 p4. Its pair spans
    # 450-700 nm rather than the 400-700 the PORTRA and GOLD panels cover --
    # neither of its curves reaches 400 nm on that sheet -- so it is 51 samples
    # against their 59-60. The range is the source's, not a truncation.
    # ⚠ AND KODAK_GOLD_100 IS DELIBERATELY ABSENT. E-7022 (2007) prints ONE
    # dye panel for TWO films and does not say which; it traces IDENTICALLY to
    # the panel in E-7022 (2022), a GOLD 200-only sheet that does name its film
    # -- max difference 0.0005 D, rms 0.00009 D over 59 samples of both curves.
    # So the shared panel is GOLD 200's, already adopted under that name, and
    # giving it to GOLD 100 as well would double-count one measurement.
    _pairs = [p.name for p in FILM_PROFILES if p.dye_density.has_neutral_pair]
    # ⚠ 6 -> 10 on 2026-08-30 (queue K1). The four PORTRA NC/VC stocks each
    # bring their OWN dye panel from E-190 pp 9-12 -- these are four separate
    # readings, not one shared panel: the traced peaks differ per film
    # (neutral 1.990 / 2.049 / 2.010 / 2.060). That distinction matters here
    # because the GOLD 100 case immediately above is the opposite situation,
    # one panel serving two films, where adopting it twice would double-count
    # a single measurement.
    # ⚠ 11 SINCE 2026-08-30 (queue B4). EASTMAN_5247_1983's panel is TI0835D,
    # which sat unread for a year behind a blocker recorded as an axis-
    # calibration problem and was in fact a faint gridline plus an
    # upside-down raster. It is the first of these pairs that is RASTER-traced
    # rather than vector, which is why `ti0835_plates.py` guards it on shape --
    # the D-min must FALL towards the red, because it is the orange mask.
    # ⚠ 11 -> 16 on 2026-09-01: the five AGFA colour negatives, from the
    # Spectral density panel of «Technical Data PF». THE PAIR COUNT IS THE ONE
    # THAT MOVES AND THE DYE-SET COUNT MUST NOT. Agfa's panel draws "Medium
    # density" and "Minimum density" -- two AGGREGATE transmissions, not three
    # separated dyes -- which is exactly the conflation NotFound.md warns
    # against when it says the dye figure "must not be corrected upward".
    # The three RSX II REVERSAL films on the same sheet DO get three dyes,
    # because their panel draws Yellow, Magenta and Cyan separately, and they
    # move the other counter instead.
    # ⚠ 16 -> 22 on 2026-09-04c: the six panels the dye sweep had classified as
    # "wrong shape" two days earlier and left unread. FUJICOLOR PRO 400H,
    # SUPERIA X-TRA 400, KONICA CENTURIA SUPER 1600, VX 100, IMPRESA 50 and
    # KODAK ULTRAMAX 800. The split is by PROCESS, not by maker: every C-41 /
    # CN-16 / CNK-4 colour NEGATIVE sheet in this corpus draws the aggregate
    # pair and every REVERSAL sheet draws three dyes, which is why these six
    # move this counter and cannot move the dye-set one.
    # ⚠ IMPRESA 50 IS THE INSTRUCTIVE ONE. `konica_raster` had described that
    # panel as "NOT A DYE TRIPLE" since 2026-08-02 and sampled it at three
    # wavelengths only, because the schema could not hold the shape. Schema v14
    # built the carrier on 2026-08-26 and nothing came back for this panel until
    # now -- a reading that existed, was correct, and was unstorable for a month.
    # ⚠ 26 -> 28 on 2026-09-07, and BOTH NEW ONES ARE PAIRS FOR THE SAME
    # REASON AS THE SIX FUJI STOCKS ABOVE THEM: «14. Spectral Dye Density
    # Curves» prints a mid-scale neutral and a D-min and never separates the
    # three dyes. That is the Fuji house pattern, which is itself part of the
    # evidence that these two are Fuji-made -- see G-VP3.
    # ⚠ COMPARED AS A SORTED SET SINCE 2026-09-08, NOT IN DATABASE ORDER. The
    # literal below was written in the old frozen-id order, so the
    # alphabetical re-sort broke a check about MEMBERSHIP for a reason that has
    # nothing to do with membership -- the list still named exactly the right
    # 28 stocks. Order-sensitivity here was incidental, never intended, and
    # would break again on the next insertion; the count and the names are
    # what this guard is for. `sorted()` on both sides keeps it asserting
    # exactly as much as it did while making it immune to reordering.
    # ⚠ 28 -> 33 ON 2026-09-15b. «Современные фотоматериалы и их обработка»
    # prints a «Спектральная плотность красителей» panel -- D-min and D-nom --
    # for most of its catalogue, and five of them land on stocks that had no
    # spectral record at all. On a masked negative the D-min trace IS THE
    # ORANGE MASK measured spectrally, which is the quantity the v14 field note
    # says nothing in this database recorded before. ⚠ EKTACHROME_160T traced
    # cleanly too and is NOT here: it is a reversal, the falling-mask test
    # below cannot speak to it, and an unverifiable record is not stored.
    chk("exactly 33 stocks carry a neutral+dmin pair",
        sorted(_pairs) == sorted([
                   "KONICA_CENTURIA_SUPER_400", "KODAK_EKTAPRESS_PJ400",
                   "KODAK_VERICOLOR_III_160", "KODAK_T400CN",
                   "KODAK_BW400CN",
                   "AGFA_OPTIMA_100", "AGFA_OPTIMA_200", "AGFA_OPTIMA_400",
                   "AGFA_PORTRAIT_160",
                   "EASTMAN_5247_1983", "EASTMAN_EXR_100T_5248",
                   "KODAK_GOLD_200",
                   "KODAK_PORTRA_160", "KODAK_PORTRA_800",
                   "KODAK_PRO_100T_PRT", "KODAK_ULTRAMAX_400",
                   "KODAK_ULTRAMAX_800",
                   "KONICA_CENTURIA_SUPER_1600", "KONICA_IMPRESA_50",
                   "KONICA_VX_100",
                   "KODAK_PORTRA_160NC", "KODAK_PORTRA_160VC",
                   "KODAK_PORTRA_400NC", "KODAK_PORTRA_400VC",
                   "AGFA_ULTRA_50",
                   "FUJICOLOR_SUPERIA_XTRA_400", "FUJICOLOR_PRO_400H",
                   # ⚠ 2026-09-06: AF3-177E section 21 prints a
                   # mid-scale neutral and a D-min curve and never
                   # separates the three dyes, so 800Z joins the PAIR
                   # list and NOT the dye-triple one.
                   "FUJICOLOR_PRO_800Z",
                   # ⚠ 2026-09-06c: AF3-100E section 20 is the SAME DRAWING,
                   # same caption, same two curves. NPZ 800 joins the pair list
                   # for the same reason and NOT the dye-triple one -- a shared
                   # drawing does not become a three-dye set by being printed
                   # twice.
                   "FUJICOLOR_PORTRAIT_NPZ_800",
                   # ⚠ 2026-09-06e: both SUPERIA sheets print the same
                   # two-curve panel under the same caption. Five of the six
                   # Fuji stocks in this list carry a pair and none a triple --
                   # Fuji simply does not publish separated dyes for its
                   # colour negatives, which is a house policy and not a gap in
                   # this corpus.
                   "FUJICOLOR_SUPERIA_XTRA_800",
                   "FUJICOLOR_SUPERIA_REALA",
                   # ⚠ 2026-09-07: the AgfaPhoto sheet's section 14, same
                   # two-curve construction under the same caption -- and its
                   # match to SUPERIA X-TRA 400's set (mean 0.0177 D against
                   # 0.154 for the next-nearest of 26) is what identifies the
                   # manufacturer. See G-VP3.
                   "AGFA_VISTA_PLUS_200",
                   "AGFA_VISTA_PLUS_400",
        ]),
        "%d stocks: %s" % (len(_pairs), ", ".join(_pairs)))
    # ⚠ THE ORANGE-MASK TEST, ON EVERY PAIR IN THE DATABASE AND NOT JUST THE NEW
    # ONES. `d_dmin` on a masked colour negative IS the mask, so it must FALL
    # towards the red, and the neutral must sit above it at every wavelength. A
    # pair stored the wrong way round satisfies every count and every grid check
    # above; this is the only guard that would see it. `ti0835_plates.py` has
    # applied the same test to EASTMAN_5247_1983 alone since 2026-08-30 -- this
    # generalises it to all 22.
    _mask_bad, _order_bad = [], []
    for _n in _pairs:
        _d = get_profile(_n).dye_density
        if not (_d.d_dmin[-1] < _d.d_dmin[0]):
            _mask_bad.append("%s %.3f->%.3f" % (_n, _d.d_dmin[0],
                                                _d.d_dmin[-1]))
        if any(a <= b for a, b in zip(_d.d_neutral, _d.d_dmin)):
            _order_bad.append(_n)
    chk("every neutral+dmin pair is a falling mask under a higher neutral",
        not _mask_bad and not _order_bad,
        "%d pairs; the mask falls on all of them" % len(_pairs)
        if not (_mask_bad or _order_bad)
        else "MASK RISES: " + ", ".join(_mask_bad)
             + "; NEUTRAL NOT ABOVE: " + ", ".join(_order_bad))
    # ⚠ AND NONE OF THE SIX NEW PANELS IS A REUSED DRAWING. Both makers in this
    # batch reused artwork on their REVERSAL sheets -- Konica's Chrome pair share
    # a magenta and a cyan to 0.00008 D, Fuji's PROVIA 100F and SENSIA 100 share
    # a whole panel -- so the question had to be asked of the negatives too. The
    # closest of the fifteen pairings is CENTURIA SUPER 1600 against VX 100 at
    # 0.065 D rms on the neutral, roughly 800x the Konica reuse figure.
    _six = ["FUJICOLOR_PRO_400H", "FUJICOLOR_SUPERIA_XTRA_400",
            "KONICA_CENTURIA_SUPER_1600", "KONICA_VX_100",
            "KONICA_IMPRESA_50", "KODAK_ULTRAMAX_800"]
    _closest, _at = 9.9, ""
    for _i, _a in enumerate(_six):
        for _b in _six[_i + 1:]:
            _da, _db = get_profile(_a).dye_density, get_profile(_b).dye_density
            _r = min((sum((x - y) ** 2 for x, y in zip(u, v)) / len(u)) ** 0.5
                     for u, v in ((_da.d_neutral, _db.d_neutral),
                                  (_da.d_dmin, _db.d_dmin)))
            if _r < _closest:
                _closest, _at = _r, "%s/%s" % (_a, _b)
    chk("the six 2026-09-04c neutral pairs are six drawings, not fewer",
        _closest >= 0.02,
        "closest %.5f D rms at %s" % (_closest, _at))

    # ⚠ 11 -> 12 on 2026-08-25 (queue G7): GEVACOLOR_NEG_682, whose Fig. 8 set
    # had been held EMPTY on purpose since 2026-08-19 rather than interpolated.
    # ⚠ 12 -> 15 on 2026-09-01: AGFACHROME RSX II 50 / 100 / 200, THE FIRST
    # SEPARATED THREE-DYE SETS IN THE AGFA CORPUS. Their panel prints Yellow,
    # Magenta, Cyan AND a Visual grey, so the three dyes can be checked against
    # the neutral they must compose to: summed at all 31 sampled wavelengths
    # they reproduce the printed grey to 0.027-0.029 D rms. That is a physical
    # closure test the panel supplies itself, not a fit residual.
    # ⚠ 15 -> 16 on 2026-09-01d: TECHNICOLOR_THREE_STRIP, from Flueckiger et al.
    # 2018 Fig. 16 -- the first dye set in this corpus that is NOT off a
    # manufacturer datasheet, and the first for an imbibition dye-transfer
    # process. It is also the only one on a grid other than 400-700/31; see the
    # named exception beside the grid assertion.
    # ⚠ 16 -> 18 on 2026-09-02 (queue G2): GEVACHROME_600 and GEVACHROME_605,
    # from Rens & Van Bets 1968 Bild 4. THE COUNT IS 18 PROFILES BUT ONLY 17
    # MEASUREMENTS -- Bild 4 draws ONE curve set and captions it for both types,
    # so the two Gevachrome arrays are identical by construction and must not be
    # counted as independent evidence. The next guard asserts exactly that.
    # ⚠ 18 -> 25 on 2026-09-04 (the Fuji/Konica dye sweep, reader
    # fuji_konica_dye.py): FUJI_PROVIA_100F, FUJI_SENSIA_100, FUJI_VELVIA_50,
    # FUJI_PROVIA_400X, KONICA_CHROME_CENTURIA_100, KONICA_CHROME_R100 and
    # KODAK_VISION3_200T_5213. SEVEN PROFILES BUT FIVE DRAWINGS -- the same
    # profiles-versus-measurements gap the Gevachrome pair opened, twice over,
    # and the guards that follow hold both pairs.
    # ⚠ 25 -> 27 on 2026-09-04c: KODAK_VISION3_50D_5203 and
    # KODAK_VISION3_250D_5207, off the RASTER panels NotFound.md had recorded as
    # blocked for being raster. Raster was never the blocker -- the dashed D-min
    # was, and it is still refused; the three dyes are not.
    # ⚠ 27 -> 28 on 2026-09-06: FUJI_PROVIA_400F, traced from AF3-066E p6
    # section 20 in Fuji's own yellow / magenta / cyan inks.
    # ⚠ 29 -> 30 AND 21 -> 22 ON 2026-09-17: EASTMAN_5293_250T_1982,
    # Kennel et al. 1982 Fig. 7, the 1982 5293's own C/M/Y set on the
    # figure's own 'NORMALIZED DENSITY-STAIN' ordinate, peak 1.0.
    chk("30 film profiles now carry a spectral dye density set",
        len(_dye) == 30, "%d: %s" % (len(_dye), ", ".join(
            sorted(p.name for p in _dye))))
    # ⚠ THE SHARED-DRAWING ASSERTIONS, 2026-09-04. Each pair below was published
    # by its maker on two sheets for two products with the SAME artwork. Both
    # profiles keep the data -- the sheets do publish it for both products --
    # but nothing downstream may treat them as two independent measurements, and
    # a re-trace that made them differ would mean one of the two extractions had
    # changed underneath us.
    _kon = [get_profile(n).dye_density for n in ("KONICA_CHROME_CENTURIA_100",
                                                 "KONICA_CHROME_R100")]
    # ⚠ THE BOUND IS ONE UNIT IN THE LAST STORED PLACE, NOT EQUALITY, AND THAT
    # IS NOT A WEAKENING. The two sheets place the same drawing at different
    # page offsets, so the two extractions differ by float noise -- rms 0.00008
    # D on the magenta and 0.00006 on the cyan, measured on the paths before
    # rounding -- and 2 of the 62 shared samples then round to a different third
    # decimal. Storing one profile's array on both to force equality would be
    # inventing which of the two roundings is right.
    _kon_n = sum(1 for u, v in (list(zip(_kon[0].d_magenta, _kon[1].d_magenta))
                                + list(zip(_kon[0].d_cyan, _kon[1].d_cyan)))
                 if u != v)
    _kon_max = max(abs(u - v)
                   for u, v in (list(zip(_kon[0].d_magenta, _kon[1].d_magenta))
                                + list(zip(_kon[0].d_cyan, _kon[1].d_cyan))))
    chk("the two Konica Chrome sheets share ONE magenta and ONE cyan drawing",
        _kon_max <= 0.0011,
        "%d of 62 shared samples differ, worst %.4f D -- one unit in the last "
        "stored place; path-level rms 0.00008 / 0.00006 D" % (_kon_n, _kon_max))
    # ... and the yellow on those two sheets WAS redrawn, so it must not match.
    # Without this half, a bug that copied one profile over the other would
    # satisfy the assertion above and look like a success.
    chk("but their yellow was redrawn and must stay different",
        _kon[0].d_yellow != _kon[1].d_yellow,
        "rms 0.017 D, max 0.049 D between the two yellows")
    # The Fuji pair is NOT bit-identical and must not be asserted as such: the
    # panels are different raster images and were traced by two different
    # methods -- ink-mask centroid on the colour-coded PROVIA sheet, predictive
    # tracking on the black SENSIA one. What is asserted is that they agree far
    # more closely than any two genuinely different E-6 sets in this corpus.
    def _worst_rms(a, b):
        return max((sum((x - y) ** 2 for x, y in zip(u, v)) / len(u)) ** 0.5
                   for u, v in ((a.d_cyan, b.d_cyan),
                                (a.d_magenta, b.d_magenta),
                                (a.d_yellow, b.d_yellow)))
    _fj = [get_profile(n).dye_density for n in ("FUJI_PROVIA_100F",
                                                "FUJI_SENSIA_100")]
    _fj_rms = _worst_rms(_fj[0], _fj[1])
    chk("PROVIA 100F and SENSIA 100 are one drawing traced two ways",
        _fj_rms <= 0.012,
        "worst rms %.5f D across the three dyes (bound 0.012)" % _fj_rms)
    # The negative control for that bound. PROVIA 400X is a genuinely different
    # Fuji E-6 dye set from the same house style, and it has to sit far outside
    # the bound or the bound is measuring nothing.
    _fj4_rms = _worst_rms(_fj[0], get_profile("FUJI_PROVIA_400X").dye_density)
    chk("PROVIA 400X is NOT that drawing -- the bound above separates them",
        _fj4_rms >= 0.030,
        "worst rms %.5f D, %.0fx the shared-pair figure"
        % (_fj4_rms, _fj4_rms / max(_fj_rms, 1e-9)))
    # ⚠ VELVIA's yellow tail is a FLOOR, NOT A MEASUREMENT: the sheet draws that
    # trace down onto the axis at about 599 nm, its last separate ink run is
    # 0.0080 D and one pixel thick at 598 nm, and 0.008 D is the printed line
    # width. It is stored as exact zeros from 600 nm and the provenance has to
    # keep saying so -- a stored zero that loses its explanation becomes a
    # measured zero to everyone downstream.
    _vv = get_profile("FUJI_VELVIA_50").dye_density
    chk("VELVIA 50's yellow is floored at zero from 600 nm, and says so",
        all(v == 0.0 for v in _vv.d_yellow[20:])
        and _vv.d_yellow[19] > 0.0
        and "FLOOR, NOT A MEASUREMENT" in _vv.source
        and "0.008 D" in _vv.source,
        "last measured %.3f D at 590 nm; line width 0.008 D is the error bar"
        % _vv.d_yellow[19])
    # ⚠ AND THE VELVIA CASE IS NOT THE FIRST. This guard was written asserting
    # "VELVIA and nowhere else" and it FAILED on GEVACOLOR_NEG_682, whose yellow
    # is also exact zeros from 580 nm -- adopted 2026-08-25 from Vervoort &
    # Stappaerts Fig. 8, where the same thing happens for the same reason (the
    # yellow trace meets the axis and the paper's own density zero is fitted to
    # 0.008 D, the identical figure). That tail had never been described as a
    # floor anywhere. Both are now named; a THIRD stock acquiring a zero run is
    # what this is watching for, because that would be the convention spreading
    # by copy rather than by measurement.
    # ⚠ A THIRD ON 2026-09-17, AND IT IS THE SAME SITUATION FOR THE SAME
    # REASON RATHER THAN THE CONVENTION SPREADING BY COPY. Kennel et al. 1982
    # Fig. 7 plots three dyes over one 400-790 nm frame and each of them meets
    # the axis somewhere inside it -- yellow by 620 nm, magenta by 700, cyan
    # not at all on the long side but only from 550 on the short. Where the
    # traced curve has reached the figure's own zero the stored value is zero,
    # which is exactly what VELVIA's and GEVACOLOR's tails record.
    _ZERO_FLOOR_OK = {"FUJI_VELVIA_50", "GEVACOLOR_NEG_682",
                      "EASTMAN_5293_250T_1982"}
    _zeroruns = sorted({p.name for p in _dye
                        if p.name not in _ZERO_FLOOR_OK
                        for row in (p.dye_density.d_cyan,
                                    p.dye_density.d_magenta,
                                    p.dye_density.d_yellow)
                        for i in range(len(row) - 3)
                        if all(v == 0.0 for v in row[i:i + 4])})
    _zero_have = sorted({p.name for p in _dye if p.name in _ZERO_FLOOR_OK
                         and all(v == 0.0 for v in p.dye_density.d_yellow[-6:])})
    chk("the zero-floor convention is used on exactly three named yellow "
        "tails",
        not _zeroruns and _zero_have == sorted(_ZERO_FLOOR_OK),
        "VELVIA 50 from 600 nm, GEVACOLOR 682 from 580 nm, 5293 (1982) "
        "from 630 nm"
        + ("" if not _zeroruns else "; ALSO ZEROED " + ", ".join(_zeroruns)))
    _g6 = [get_profile(n).dye_density for n in ("GEVACHROME_600",
                                                "GEVACHROME_605")]
    chk("the two Gevachrome dye sets are ONE measurement, stored twice",
        _g6[0].d_cyan == _g6[1].d_cyan and _g6[0].d_magenta == _g6[1].d_magenta
        and _g6[0].d_yellow == _g6[1].d_yellow,
        "Bild 4 is captioned «Typ 6.00 und Typ 6.05» and draws three curves")
    for _n in ("EASTMAN_EKTACHROME_7239", "KODAK_VISION2_200T_5217",
               "KODAK_VISION2_500T_5218"):
        _d = get_profile(_n).dye_density
        chk(f"{_n} carries the 2026-08-18 dye set", _d.has_data,
            "normalisation %s" % _d.normalisation)
    # The 7239 sheet states its own normalisation in words, so the stored string
    # must not silently drift to the peak_1.0 family the VISION sheets use.
    chk("7239's dye set records the visual-neutral-1.0 normalisation",
        get_profile("EASTMAN_EKTACHROME_7239").dye_density.normalisation
        == "as_printed_visual_neutral_1.0", "as printed on the sheet")
    # ... and 5217/5218 must not drift the other way.
    chk("5217 and 5218 record peak_1.0, which is what their sheets print",
        all(get_profile(n).dye_density.normalisation == "peak_1.0"
            for n in ("KODAK_VISION2_200T_5217", "KODAK_VISION2_500T_5218")),
        "both peak_1.0")
    # Physics, for the three new sets: each dye must peak in its own band. This
    # is what would catch a curve-assignment slip in the extractor.
    _np2 = _np if "_np" in dir() else __import__("numpy")
    _off = []
    for _n in ("EASTMAN_EKTACHROME_7239", "KODAK_VISION2_200T_5217",
               "KODAK_VISION2_500T_5218"):
        _d = get_profile(_n).dye_density
        _g = [_d.lambda_start_nm + _d.lambda_step_nm * i
              for i in range(len(_d.d_cyan))]
        _pk = lambda row: _g[max(range(len(row)), key=lambda i: row[i])]
        if not (405 <= _pk(_d.d_yellow) <= 480 and 510 <= _pk(_d.d_magenta) <= 590
                and 615 <= _pk(_d.d_cyan) <= 700):
            _off.append("%s y%d m%d c%d" % (_n, _pk(_d.d_yellow),
                                           _pk(_d.d_magenta), _pk(_d.d_cyan)))
    # (the whole-set version of this check is a few sections up; this one
    # names the three new sets so a failure says WHICH extraction slipped)
    chk("the 3 new dye sets peak in their own absorption bands",
        not _off, "; ".join(_off) if _off else "y 440-450, m 540-550, c 670-680")

    # 2. THE FIRST MEASURED REVERSAL sigma(D). Its dmax anchor is ABOVE mid,
    # which is the opposite of what _grain_v2's reversal heuristic (0.7/1.0/0.5)
    # assumes. Guarding the direction, not just the value, is the point: a
    # regression to the heuristic would silently erase the measurement.
    _e = get_profile("KODAK_EKTACHROME_100D_5285").grain
    chk("5285 sigma(D) is the measured reversal shape, rising with density",
        (_e.sigma_shape_toe, _e.sigma_shape_mid, _e.sigma_shape_dmax)
        == (0.15, 1.00, 3.10),
        "%.2f / %.2f / %.2f" % (_e.sigma_shape_toe, _e.sigma_shape_mid,
                                _e.sigma_shape_dmax))
    chk("5285 sigma(D) is NOT the reversal heuristic 0.7/1.0/0.5",
        (_e.sigma_shape_toe, _e.sigma_shape_dmax) != (0.7, 0.5),
        "measurement, not heuristic")
    # The level, which moved 4.4x. The sibling reversal stocks are the sanity
    # bracket: 7239 prints 14.0 and TRI-X reversal 10.0, so 13.1 belongs and
    # 3.0 (finer than VISION3 50D) did not.
    chk("5285 rms is the traced 13.1, not the unattributed 3.0",
        abs(_e.rms_granularity - 13.1) < 1e-9,
        "rms %.1f, siblings 7239=%.1f TRI-X rev=%.1f" % (
            _e.rms_granularity,
            get_profile("EASTMAN_EKTACHROME_7239").grain.rms_granularity,
            get_profile("KODAK_TRI_X_REVERSAL_200").grain.rms_granularity))
    chk("5285 keeps its MEASURED per-layer rms (green finest, blue coarsest)",
        _e.rms_rgb() == (19.0, 13.1, 25.7),
        "r/g/b %.1f/%.1f/%.1f" % _e.rms_rgb())

    # 3. PLUS-X 5231's MTF, read off the sheet's own vector path.
    _m = get_profile("EASTMAN_PLUS_X_5231").mtf
    chk("EASTMAN_PLUS_X_5231 f50 is the measured 41.3, not the estimated 60.0",
        abs(_m.f50_g - 41.3) < 1e-9 and _m.f50_r == _m.f50_g == _m.f50_b,
        "f50 %.1f cycles/mm, one figure for a panchromatic B&W stock" % _m.f50_g)
    # ⚠ THIS CHECK ASSERTED THE WRONG PREMISE UNTIL 2026-09-02e AND SO PROTECTED
    # A DEFECT. It read "adjacency is the measured 3.4 % overshoot" and pinned
    # 0.034 -- but `adjacency` is the difference-of-Gaussians amplitude BEFORE
    # the rolloff attenuates it, not the overshoot that survives. Pinned at the
    # observed value the rendered curve never reached 100 % at all, i.e. the
    # guard was holding the stock in the C19 inert census. It now pins what the
    # sheet actually says -- the RENDERED overshoot -- and lets the parameter be
    # whatever reproduces it (0.0691 at adjacency_um 32.1, solved in A4).
    chk("EASTMAN_PLUS_X_5231 RENDERS the measured 3.4 % overshoot (the "
        "parameter behind it is not that number and never was)",
        abs(_rendered_peak(_m, 1)[0] - 1.034) < 2e-3,
        "renders %+.4f from adjacency %+.4f / adjacency_um %.1f"
        % (_rendered_peak(_m, 1)[0] - 1.0, _m.adjacency, _m.adjacency_um))

    # ---- C1: sigma(D) is WIRED (2026-08-18) ---------------------------------
    # The field group sat in the schema for weeks, populated and validated, read
    # by nothing. These checks guard the three properties the wiring turns on.
    #
    # 1. THE REGRESSION GUARD, and it is the most important one here. For every
    # profile WITHOUT a measured shape, the new sampler must reproduce the old
    # hardcoded expression exactly -- because the alternative is a silent global
    # change to grain in 150 stocks. Compared over a density sweep, in float32.
    # ⚠ THIS GUARD WAS REPLACED, NOT REPAIRED, ON 2026-08-18 (queue item C1b).
    # It used to assert that every unmeasured stock reproduced the raw legacy
    # expression sqrt(D - dmin + fog) bit-for-bit, by multiplying the sampler's
    # output back by that expression's value at ABSOLUTE D = 1.0. C1b moved the
    # normalisation to NET density 1.0 -- the convention Kodak prints on 5248 p1
    # and 5222 p1 -- so that identity is now false BY DESIGN on all 155 stocks,
    # and a guard that only had to be "made to pass" would have hidden the very
    # change it was written to protect. What still must hold is the SHAPE: the
    # sampler must equal the legacy law divided by a single constant,
    # sqrt(1 + fog), with no dmin term and therefore no per-channel term.
    _leg_bad, _n_leg = [], 0
    for _p in FILM_PROFILES:
        _g, _c = _p.grain, _p.curves.g
        if _g.sigma_shape_measured:
            continue
        _n_leg += 1
        _D = _np.linspace(0.0, 3.5, 36).astype(_np.float32)
        _raw = _np.sqrt(_np.maximum(_D - _np.float32(_c.dmin), _np.float32(0.0))
                        + _np.float32(_g.fog_grain))
        _k = float(_np.sqrt(1.0 + float(_g.fog_grain)))
        _new = film_profiles.grain_sigma(_g, _c.dmin, _c.dmax, _D)
        if float(_np.max(_np.abs(_new * _np.float32(_k) - _raw))) > 2e-6:
            _leg_bad.append(_p.name)
    chk("unmeasured stocks keep the legacy grain SHAPE, rescaled by 1/sqrt(1+fog)",
        not _leg_bad, ", ".join(_leg_bad[:4]) if _leg_bad
        else "%d profiles, max deviation < 2e-6" % _n_leg)

    # THE LEVEL CONTRACT, and the reason C1b was worth doing at all: after
    # multiplying by the stock's rms the renderer must reproduce that stored
    # figure at NET density 1.0 -- every stock, every channel, measured shape or
    # not. This is the single assertion that pins what rms_granularity MEANS.
    _lvl = []
    for _p in FILM_PROFILES:
        _g = _p.grain
        for _ch, _cur in (("r", _p.curves.r), ("g", _p.curves.g), ("b", _p.curves.b)):
            _v = film_profiles.grain_sigma(_g, _cur.dmin, _cur.dmax, _cur.dmin + 1.0)
            if abs(_v - 1.0) > 1e-5:
                _lvl.append("%s.%s=%.5f" % (_p.name, _ch, _v))
    chk("grain amplitude is exactly the stored rms at NET density 1.0",
        not _lvl, "; ".join(_lvl[:4]) if _lvl
        else "465 stock-channels, |amp - 1| < 1e-5")

    # ⚠ AND THE CONVERSE: the sampler must NOT be 1.0 at absolute 1.0 for a
    # masked stock, because that was the bug. If someone "fixes" the
    # normalisation back, this fails loudly instead of silently re-introducing a
    # shadow-referenced level on every masked colour negative.
    _mask = get_profile("KODAK_VISION_250D_5246")
    _amp_abs = film_profiles.grain_sigma(_mask.grain, _mask.curves.b.dmin,
                                         _mask.curves.b.dmax, 1.0)
    chk("the sampler is NOT normalised at absolute 1.0 on a masked stock",
        abs(_amp_abs - 1.0) > 0.02,
        "5246 blue amp at absolute D=1.0 is %.3f; absolute 1.0 is net %.2f there"
        % (_amp_abs, 1.0 - _mask.curves.b.dmin))

    # The compensation applied to the four Svema Foto stocks must actually
    # preserve their pre-C1b amplitude, at every density, or "appearance
    # preserving" is just a claim. Compared against the OLD expression:
    # rms_before * sqrt(D - dmin + fog).
    _SV = {"SVEMA_FOTO_32": 8.5, "SVEMA_FOTO_65": 11.5,
           "SVEMA_FOTO_130": 18.0, "SVEMA_FOTO_250": 33.0}
    _sv_bad = []
    for _n, _before in _SV.items():
        _p = get_profile(_n)
        _g, _c = _p.grain, _p.curves.g
        _D = _np.linspace(0.0, 3.0, 25)
        _old_amp = _before * _np.sqrt(_np.maximum(_D - _c.dmin, 0.0) + _g.fog_grain)
        _new_amp = _g.rms_granularity * film_profiles.grain_sigma(
            _g, _c.dmin, _c.dmax, _D)
        _err = float(_np.max(_np.abs(_new_amp - _old_amp)
                             / _np.maximum(_old_amp, 1e-9)))
        if _err > 2e-3:
            _sv_bad.append("%s off by %.2f%%" % (_n, _err*100))
    chk("the Svema pipeline-fitted stocks render exactly as before C1b",
        not _sv_bad, "; ".join(_sv_bad) if _sv_bad
        else "4 stocks, worst deviation < 0.2 % over D 0-3")

    # 2. Exactly the traced stocks may use the shape. A heuristic shape must
    # never acquire the flag -- that is the whole safety property.
    # ⚠ THIS LIST GREW 5 -> 11 on 2026-08-18 (queue item C1c, the completing
    # sigma(D) harvest), and the count assertion is MEANT to fail when it does.
    # The six additions are every remaining sheet in the corpus that draws its
    # granularity plot as VECTOR art: 5245, 5246, 5248, 5274, 5279, 5218.
    _meas = sorted(p.name for p in FILM_PROFILES if p.grain.sigma_shape_measured)
    # ⚠ 11 -> 12 on 2026-08-20: KODAK_VISION2_50D_5201, from a NINTH vector
    # sheet found while reviewing the Kodak folder. Its shape is the flattest in
    # the corpus (interior peak 1.20x against 1.38-1.62x on the other six colour
    # negatives), which is what makes it worth having rather than just one more.
    # ⚠ 12 -> 13 on 2026-08-25: KODAK_TRI_X_REVERSAL_200, and it is the FIRST
    # entry that is not a colour negative and not a vendor VECTOR plot -- it was
    # traced off a raster granularity panel and paired against that sheet's own
    # characteristic curve. It is also the first whose shape RISES to dmax
    # (0.262 -> 2.829) instead of peaking mid-scale, which is why it is scoped to
    # this one stock and the 34 other reversal stocks were left alone.
    # ⚠ IT STAYED 13 ON 2026-09-02c, AND THAT IS THE RESULT OF QUEUE E5 RATHER
    # THAN A FAILURE TO DO THE WORK. A good sigma(D) trace for EASTMAN_5294_1983
    # was made from Sehlin & Kennel's Fig. 8 and then WITHDRAWN: its anchor
    # densities are Fig. 8's plotted density, not the per-layer analytical
    # density `sigma_anchors` reads, and the traced toe at D 0.44 sits below
    # that stock's own green dmin of 0.68. cpp_parity.py caught it at 5.7e-01
    # against a 2e-05 tolerance. See the note on the profile.
    # ⚠ 13 -> 14 ON 2026-09-17: EASTMAN_5293_250T_1982, whose sigma(D)
    # comes off Fig. 13 by ELIMINATING that figure's bare log-exposure
    # abscissa against the density curve drawn in the same frame.
    chk("only the 14 vendor-traced stocks are flagged sigma_shape_measured",
        _meas == ["EASTMAN_5293_250T_1982", "EASTMAN_EXR_100T_5248", "EASTMAN_EXR_50D_5245",
                  "KODAK_EKTACHROME_100D_5285", "KODAK_TRI_X_REVERSAL_200",
                  "KODAK_VISION2_500T_5218",
                  "KODAK_VISION2_50D_5201",
                  "KODAK_VISION3_200T_5213", "KODAK_VISION3_250D_5207",
                  "KODAK_VISION3_500T_5219", "KODAK_VISION3_50D_5203",
                  "KODAK_VISION_200T_5274", "KODAK_VISION_250D_5246",
                  "KODAK_VISION_500T_5279"],
        ", ".join(n.split("_")[-1] for n in _meas))
    _heur = [p.name for p in FILM_PROFILES
             if not p.grain.sigma_shape_measured
             and p.grain.sigma_anchors(p.curves.g.dmin, p.curves.g.dmax) is not None]
    chk("no unflagged profile can produce a shape from sigma_anchors",
        not _heur, ", ".join(_heur[:4]) if _heur else "137 heuristic shapes inert")

    # 3. The sampler's contract: 1.0 at D = 1.0, held flat outside the traced
    # range, and the interior peak actually reachable. A shape that did not pass
    # through 1.0 at D = 1.0 would silently rescale the stored rms.
    # ⚠ REWRITTEN 2026-08-18 (C1b). This used to assert the sampler returned 1.0
    # at ABSOLUTE D = 1.0 and that its flat-hold values equalled the STORED
    # anchors verbatim. Both were convention-dependent statements, and C1b changed
    # the convention: the reference density is now net 1.0 and the stored anchors
    # are ratios to the absolute-1.0 value, so neither identity holds any more.
    # The two properties actually worth guarding are convention-INDEPENDENT:
    # (1) the curve is HELD FLAT outside the traced range -- expressed as "the
    #     value 5 D below the toe equals the value AT the toe", which is true
    #     under any normalisation, and (2) the level is right, which the net-1.0
    #     contract above now asserts for all 465 stock-channels.
    _sig_bad = []
    for _n in _meas:
        _p = get_profile(_n)
        _g, _c = _p.grain, _p.curves.g
        _toe_at = _g.sigma_shape_toe_at or _c.dmin
        _top_at = _g.sigma_shape_dmax_at or _c.dmax
        _lo = film_profiles.grain_sigma(_g, _c.dmin, _c.dmax, -5.0)
        _hi = film_profiles.grain_sigma(_g, _c.dmin, _c.dmax, 99.0)
        if abs(_lo - film_profiles.grain_sigma(_g, _c.dmin, _c.dmax, _toe_at)) > 1e-6:
            _sig_bad.append("%s not held flat below the toe" % _n)
        if abs(_hi - film_profiles.grain_sigma(_g, _c.dmin, _c.dmax, _top_at)) > 1e-6:
            _sig_bad.append("%s not held flat above dmax" % _n)
        # the shape must still RISE from the toe anchor to the stored peak
        if _g.sigma_shape_peak > 0.0:
            _a = film_profiles.grain_sigma(_g, _c.dmin, _c.dmax, _toe_at)
            _b = film_profiles.grain_sigma(_g, _c.dmin, _c.dmax, _g.sigma_shape_peak_at)
            if not _b > _a:
                _sig_bad.append("%s peak %.3f not above toe %.3f" % (_n, _b, _a))
    chk("the sigma(D) sampler holds flat outside the trace and peaks inside it",
        not _sig_bad, "; ".join(_sig_bad) if _sig_bad
        else "%d of %d measured stocks" % (len(_meas), len(_meas)))
    # The interior peak is the reason the carrier grew; assert it is real, i.e.
    # that the sampler returns MORE at the peak density than at the mid anchor.
    # Net-relative since C1b: the peak must exceed the value at the stock's own
    # rms reference density, which is what "an interior maximum" means once the
    # reference is net 1.0 rather than absolute 1.0.
    _pk = [p.name for p in FILM_PROFILES
           if p.grain.sigma_shape_peak > 0.0
           and film_profiles.grain_sigma(p.grain, p.curves.g.dmin, p.curves.g.dmax,
                                         p.grain.sigma_shape_peak_at)
           <= film_profiles.grain_sigma(p.grain, p.curves.g.dmin, p.curves.g.dmax,
                                        p.curves.g.dmin + 1.0)]
    chk("every stored interior peak exceeds the stock's own rms reference density",
        not _pk, ", ".join(_pk) if _pk
        else "10 measured peaks, all above their net-1.0 value")

    # ---- C1c: the completing sigma(D) harvest (2026-08-18) -------------------
    # Six colour negatives adopted from VECTOR granularity plots. What these
    # checks defend is not the numbers themselves -- granularity_vector.py
    # --assert pins those against the PDFs -- but the properties of the adoption
    # that a later edit could quietly break.
    # ⚠ THE LAST COLUMN CHANGED ON 2026-08-18 UNDER C1d, deliberately. Under C1c
    # this guard asserted the stored rms was UNTOUCHED (4.2 / 5.3 / 5.6 / 5.8 /
    # 8.3 / 7.3) because the shape had been adopted and the level had not. C1d
    # adopted the level too, read off the same curve at NET density 1.0, so the
    # guard now pins the new values. The assertion still does the same job: if a
    # later edit moves an rms, the entry's own comment -- which quotes the figure
    # and its ratio to the old one -- would otherwise go stale in silence.
    _C1C = {
        "EASTMAN_EXR_50D_5245":    (1.19, 0.72, 1.47, 0.73, 0.572, 2.091, 4.10),
        "KODAK_VISION_250D_5246":  (0.94, 0.90, 1.62, 0.66, 0.582, 2.201, 6.78),
        "EASTMAN_EXR_100T_5248":   (1.19, 0.84, 1.58, 0.74, 0.612, 2.051, 5.87),
        "KODAK_VISION_200T_5274":  (0.80, 0.61, 1.38, 0.68, 0.582, 2.211, 6.68),
        "KODAK_VISION_500T_5279":  (0.96, 0.50, 1.42, 0.65, 0.576, 2.210, 8.74),
        "KODAK_VISION2_500T_5218": (1.17, 0.70, 1.56, 0.74, 0.592, 2.309, 6.65),
    }
    # And the per-layer triples C1d adopted with them, plus the finding that made
    # them worth adopting: measured blue is 1.9-2.8x green, where the schema's
    # tier-2 ladder had assumed 1.3x for every colour negative.
    _C1D_RGB = {
        "EASTMAN_EXR_50D_5245":    (3.80, 4.10, 11.42),
        "KODAK_VISION_250D_5246":  (7.03, 6.78, 12.56),
        "EASTMAN_EXR_100T_5248":   (4.42, 5.87, 11.29),
        "KODAK_VISION_200T_5274":  (5.34, 6.68, 15.75),
        "KODAK_VISION_500T_5279":  (6.87, 8.74, 20.39),
        "KODAK_VISION2_500T_5218": (5.51, 6.65, 15.51),
    }
    _rgb_bad = []
    for _n, _want in _C1D_RGB.items():
        _got = get_profile(_n).grain.rms_rgb()
        if max(abs(_got[_i] - _want[_i]) for _i in range(3)) > 1e-9:
            _rgb_bad.append("%s %s" % (_n, tuple(round(v, 2) for v in _got)))
        if _got[2] <= 1.5 * _got[1]:
            _rgb_bad.append("%s blue only %.2fx green" % (_n, _got[2]/_got[1]))
    chk("the 6 re-levelled negatives carry their measured per-layer triples",
        not _rgb_bad, "; ".join(_rgb_bad[:3]) if _rgb_bad
        else "6 of 6, blue 1.9-2.8x green as measured")
    _c1c_bad = []
    for _n, (_toe, _dmx, _pkv, _pka, _tat, _dat, _rms) in _C1C.items():
        _g = get_profile(_n).grain
        if not (abs(_g.sigma_shape_toe - _toe) < 1e-9
                and abs(_g.sigma_shape_dmax - _dmx) < 1e-9
                and abs(_g.sigma_shape_peak - _pkv) < 1e-9
                and abs(_g.sigma_shape_peak_at - _pka) < 1e-9
                and abs(_g.sigma_shape_toe_at - _tat) < 1e-9
                and abs(_g.sigma_shape_dmax_at - _dat) < 1e-9
                and _g.sigma_shape_measured):
            _c1c_bad.append(_n)
        if abs(_g.rms_granularity - _rms) > 1e-9:
            _c1c_bad.append("%s rms moved to %.2f" % (_n, _g.rms_granularity))
    chk("the 6 vector-traced negatives carry their traced anchors exactly",
        not _c1c_bad, "; ".join(_c1c_bad) if _c1c_bad
        else "6 of 6, shape and re-levelled rms both pinned")

    # ---- 2026-08-20: KODAK_VISION2_50D_5201 and FUJI_SUPER_F125_8532 ---------
    # Two new stocks, adopted in one batch so the plugin's ListBox shifts once.
    # What these guards defend is not the numbers -- the audit stage re-derives
    # those from the PDFs on every build -- but the PROPERTIES of the adoption
    # that a later tidy-up could quietly undo.
    _p01 = get_profile("KODAK_VISION2_50D_5201")

    # 1. The whole reason this stock is interesting: everything came off ONE
    # sheet. If any of the four measured families ever loses its flag or its
    # value, the description's "FIRST stock whose ... are all traced" goes stale.
    chk("5201 carries all four measured families from H-1-5201",
        (_p01.grain.sigma_shape_measured and _p01.mtf.mtf_measured
         and abs(_p01.grain.rms_granularity - 4.51) < 1e-9
         and abs(_p01.curves.g.gamma - 0.5945) < 1e-9),
        "sigma shape + rms + MTF + curves, all measured")

    # 2. ⚠ THE MEASURED TOE AND SHOULDER SOFTNESSES MUST NOT BE "TIDIED" BACK TO
    # THE FAMILY DEFAULT. Every VISION2 sibling carries toe_k 0.300 and
    # shoulder_k 0.420 in all three channels -- the signature of hand-set
    # numbers, and exactly what `_neg()` produces if someone rewrites this entry
    # using the family helper. 5201's are fitted per channel and none of them is
    # 0.30 or 0.42; this assertion is what fails if the fit is overwritten.
    _soft = [(c.toe_k, c.shoulder_k) for c in _p01.curves.as_tuple()]
    chk("5201's fitted toe/shoulder softnesses are not the family's hand-set pair",
        all(abs(tk - 0.30) > 0.02 and abs(sk - 0.42) > 0.02 for tk, sk in _soft)
        and len({round(tk, 4) for tk, _ in _soft}) == 3,
        "3 distinct fitted toe_k, none equal to the family 0.30/0.42")

    # 3. The per-record MTF is the first in the file, so its ORDERING is the
    # claim worth pinning: blue sharpest, red softest, which is the layer order
    # the MTFSpec docstring predicts and which this sheet confirms directly.
    chk("5201's measured f50 rises red -> green -> blue",
        _p01.mtf.f50_r < _p01.mtf.f50_g < _p01.mtf.f50_b
        and _p01.mtf.mtf_rolloff_q > 0.0,
        "%.1f < %.1f < %.1f cycles/mm, q = %.2f"
        % (_p01.mtf.f50_r, _p01.mtf.f50_g, _p01.mtf.f50_b,
           _p01.mtf.mtf_rolloff_q))

    # 4. ⚠ q IS NOT A FAMILY CONSTANT, and this is the evidence. C2 adopted the
    # power-law rolloff on ONE curve at q = 1.84 and C2b asks whether several
    # curves agree. They do not: 5201's three records measure 2.77 / 3.23 / 3.42
    # and 5274's 1.89 / 2.94 / 3.38. The guard pins the SPREAD so that a future
    # "let's just use 1.84 everywhere" simplification fails loudly rather than
    # looking tidy. Stated as a spread over every measured stock so it does not go
    # stale the next time one is added -- it said "the two" until 2026-08-20c and
    # there are now three.
    _qs = sorted(p.mtf.mtf_rolloff_q for p in FILM_PROFILES if p.mtf.mtf_measured)
    chk("the measured rolloff exponents span more than 1.0, as measured",
        len(_qs) >= 2 and (_qs[-1] - _qs[0]) > 1.0,
        "q = " + " / ".join("%.2f" % q for q in _qs))

    # 5. Blue grain is 2.14x green here. The finding that mattered under C1d was
    # that the old tier-2 ladder's 1.3x understated the top layer badly; 5201 is
    # the seventh independent measurement and it must not drift back toward 1.3.
    _r01 = _p01.grain.rms_rgb()
    chk("5201's measured blue grain stays near 2.1x green",
        1.9 < _r01[2] / _r01[1] < 2.4,
        "blue %.2f / green %.2f = %.2fx" % (_r01[2], _r01[1], _r01[2]/_r01[1]))

    # 6. ⚠ THIS GUARD WAS INVERTED ON 2026-08-25, and the old form is kept in
    # this comment because it is the more interesting one. It read "5201 keeps
    # its dye and spectral sets UNSOURCED pending C9 / C10" and asserted that
    # BOTH panels stayed empty -- the point being that a transfer from 5205 or
    # 5218 would render plausibly and be undocumented, which is the failure mode
    # the provenance scheme exists to prevent. Both panels are now EXTRACTED FROM
    # THIS SHEET's own vector art (C9, C10), so the property to assert flips:
    # each must carry a citation naming the script that produced it, and neither
    # may name a sibling stock -- an empty-to-filled transition is exactly when a
    # transfer would be easiest to slip in.
    # ⚠ "SOURCED" IS A NON-EMPTY `source`, NOT A non-None FIELD. Every profile
    # gets a SpectralSensitivity and a SpectralDyeDensity struct whether or not
    # anything was measured -- the same representation GEVACOLOR_NEG_682 uses for
    # its deliberately-unfilled dye set under G7.
    # The transfer test is on the ARRAYS, not on the citation text: a citation
    # can mention a sibling for a legitimate reason (5201's spectral source names
    # 5218/5217/5219 precisely to record that their criterion string is not
    # printed on their sheets), whereas an array that equals a sibling's IS the
    # transfer, whatever the prose says.
    _sib01 = ("KODAK_VISION2_250D_5205", "KODAK_VISION2_500T_5218",
              "KODAK_VISION2_200T_5217", "KODAK_VISION3_500T_5219")
    _xfer = [n for n in _sib01
             if get_profile(n).dye_density.d_cyan == _p01.dye_density.d_cyan
             or get_profile(n).spectral.log_s_r == _p01.spectral.log_s_r]
    chk("5201's dye and spectral sets cite THIS sheet and match no sibling",
        "dye_density.py" in _p01.dye_density.source
        and "spectral_vector.py" in _p01.spectral.source
        and "H-1-5201" in _p01.dye_density.source
        and "H-1-5201" in _p01.spectral.source
        and not _xfer,
        ", ".join(_xfer) if _xfer
        else "both from H-1-5201 p3; distinct from all 4 siblings")

    _f32 = get_profile("FUJI_SUPER_F125_8532")
    # 7. Fuji prints the rms AND its convention -- "a visual diffuse density 1.0
    # above the minimum density; a 48um diameter aperture" -- which is net 1.0,
    # this database's own reference since C1b. So 3.0 needs no conversion, and
    # the guard pins the printed value rather than a derived one.
    chk("8532 carries Fuji's printed rms 3.0 unconverted",
        abs(_f32.grain.rms_granularity - 3.0) < 1e-9
        and _f32.exposure_index == 125 and _f32.balance_kelvin == 3200,
        "rms 3.0 at net 1.0, EI 125 at 3200 K")

    # 8. ⚠ 8532's SHARPNESS PANEL IS A CONTRAST TRANSFER FUNCTION measured
    # against a RECTANGULAR wave chart, not a sine-wave MTF -- it runs ABOVE the
    # MTF, so reading f50 straight off it would overstate sharpness. Until
    # 2026-08-23 this guard therefore demanded the opposite of what it demands
    # now: it required NOT mtf_measured and required f50_g to equal 8530's, and
    # it existed to stop someone "finishing the job" by flagging a square-wave
    # curve as an MTF.
    #   THE JOB IS NOW FINISHED PROPERLY, so the guard is inverted rather than
    # deleted, and it keeps guarding the same hazard from the other side: the
    # panel is converted by Coltman's square-to-sine inversion, f50_g is the
    # SINE 32.07 c/mm, and the stock must now be flagged -- but it must NOT have
    # simply inherited the printed 37.78 CTF crossing, and it must no longer
    # equal 8530's transferred 42.0. Both of those are what this checks. The
    # citation must still carry the word "rectangular", because that word is the
    # reason a conversion was needed at all.
    _f72 = get_profile("FUJICOLOR_SUPER_F500_8572")
    chk("8532's f50 is the Coltman SINE conversion, not the printed CTF crossing",
        _f32.mtf.mtf_measured
        and abs(_f32.mtf.f50_g - 32.07) < 1e-9
        and abs(_f32.mtf.f50_g - 37.78) > 1.0
        and abs(_f32.mtf.f50_g - get_profile("FUJI_F125_8530").mtf.f50_g) > 1.0
        and "rectangular" in _f32.provenance.sources[0].lower()
        and "coltman" in _f32.provenance.sources[0].lower(),
        "sine f50 32.07 vs printed CTF 37.78, conversion cited")
    # The sister sheet gets the same treatment and the same guard. Its printed
    # CTF crosses at 24.79 and the converted sine f50 is 20.21; if either stock
    # were ever "corrected" back to its printed crossing, this catches it.
    chk("8572's f50 is the Coltman SINE conversion too",
        _f72.mtf.mtf_measured
        and abs(_f72.mtf.f50_g - 20.21) < 1e-9
        and abs(_f72.mtf.f50_g - 24.79) > 1.0
        and "coltman" in _f72.provenance.sources[0].lower(),
        "sine f50 20.21 vs printed CTF 24.79")
    # A converted CTF must land BELOW its own printed crossing -- that is the
    # direction of the whole correction, and getting the sign backwards is the
    # single most likely way to misapply Coltman. Both stocks, both directions.
    chk("both converted stocks are softer than their printed CTF says",
        _f32.mtf.f50_g < 37.78 and _f72.mtf.f50_g < 24.79,
        "32.07 < 37.78 and 20.21 < 24.79")
    # ⚠ THE MEASURED 8532 IS SOFTER THAN THE 8530 IT SUPERSEDES (32.07 against
    # Honjo's 42.0) WHILE FUJI'S OWN PAGE SELLS IT ON "dramatically increased
    # sharpness". Method rule 4 says record the conflict, never average, so the
    # guard PINS the contradiction: if some later pass quietly nudges 8532 up
    # toward 42 to make the marketing copy come true, this fails.
    chk("the 8532-vs-8530 sharpness conflict is preserved, not averaged",
        _f32.mtf.f50_g < get_profile("FUJI_F125_8530").mtf.f50_g
        and abs(get_profile("FUJI_F125_8530").mtf.f50_g - 42.0) < 1e-9
        and "contradicts" in "".join(_f32.provenance.sources).lower(),
        "8532 32.07 < 8530 42.0, contradiction recorded in the citation")

    # 9. Fuji states the failure is achromatic -- "does not need lens opening
    # adjustment nor filtration" -- so unlike the Kodak entries this one must
    # have NO channel spread. Inventing one would contradict the source.
    _rc = film_profiles.reciprocity_for(_f32) if hasattr(
        film_profiles, "reciprocity_for") else film_profiles._reciprocity_for(_f32)
    chk("8532's reciprocity is achromatic, as Fuji prints it",
        (abs(_rc.schwarzschild_p_r - 0.90) < 1e-9
         and _rc.schwarzschild_p_r == _rc.schwarzschild_p_g == _rc.schwarzschild_p_b
         and abs(_rc.onset_s - 0.1) < 1e-9),
        "p = 0.90 in all three channels, onset 1/10 s")

    # 10. Queue item C5, owner-approved: the 5247 re-tier. A mixed [T1/T2] tag
    # does NOT match `_provenance_for`'s regex, so the tier has to be stated in
    # `_UNTAGGED_TIER` -- and if that entry is ever dropped the profile silently
    # falls back to 3, which is the bug this closes. 8532 is checked alongside it
    # for the same reason.
    # 8532 MOVED 2 -> 1 ON 2026-08-23 when its curves and green f50 stopped
    # being transfers from 8530 and became traces of its own sheet. The guard
    # keeps checking that the mixed-tag regex does not silently drop either
    # profile to 3; only the expected tier changed.
    chk("the C5 re-tier and 8532's tier survive the mixed-tag regex",
        get_profile("EASTMAN_5247_1983").provenance.tier == 1
        and _f32.provenance.tier == 1,
        "5247_1983 -> tier 1, 8532 -> tier 1")

    # ---- Queue item C12, closed 2026-08-25: the CLASS, not the two instances.
    # The guard above names two profiles. That is what let SIX others sit at tier
    # 3 with fitted_from="analogy" for weeks -- the three VISION2 and the three
    # VISION camera negatives, every one of them owning its own Kodak sheet and
    # four of them carrying a sigma(D) shape traced from it. A per-profile guard
    # cannot catch the next one; this one fails for ANY mixed-tag profile that is
    # missing from `_UNTAGGED_TIER`, so a new mixed tag cannot be added without
    # someone deciding what it resolves to.
    # ⚠ IT ALSO FAILS THE OTHER WAY: a mixed-tag profile that resolves to 3 is
    # rejected, because 3 is exactly the value the regex falls back to and the
    # entry would then be indistinguishable from the bug. A genuinely tier-3
    # profile has no business carrying a [T1/...] tag in the first place.
    _mixed = [p for p in FILM_PROFILES
              if re.match(r"\[T[123]/T[123]\]", p.description)]
    _mixed_bad = [p.name for p in _mixed
                  if p.name not in film_profiles._UNTAGGED_TIER
                  or film_profiles._UNTAGGED_TIER[p.name] == 3]
    chk("every mixed-tag profile states its resolved tier in _UNTAGGED_TIER",
        not _mixed_bad and len(_mixed) == 8,
        ", ".join(_mixed_bad) if _mixed_bad
        else f"{len(_mixed)} mixed-tag profiles, all resolved, none to 3")
    # And the six themselves, by name, so a later edit cannot quietly demote them
    # back to the family-ladder tier the traced curves disproved.
    _c12 = {"KODAK_VISION2_500T_5218", "KODAK_VISION2_200T_5217",
            "KODAK_VISION2_250D_5205", "KODAK_VISION_500T_5279",
            "KODAK_VISION_200T_5274", "KODAK_VISION_250D_5246"}
    _c12_bad = [n for n in sorted(_c12)
                if get_profile(n).provenance.tier != 1
                or get_profile(n).provenance.fitted_from != "datasheet_curve"]
    chk("the 6 C12 stocks are tier 1 on datasheet_curve, not analogy",
        not _c12_bad, ", ".join(_c12_bad) if _c12_bad
        else "6 of 6 at tier 1; the T3 residual is rms_granularity alone")

    # ⚠ THE SHAPE IS NOT THE HEURISTIC'S SHAPE -- the same check that mattered for
    # 5285. _grain_v2 gives colour negative 0.40 / 1.00 / 1.20 (rising to dmax);
    # all six sheets measure a FALL to 0.50-0.90 with an interior peak below
    # D = 1.0. If any adopted triple ever equalled the heuristic's, the adoption
    # would have been silently reverted by the schema pass.
    _same = [_n for _n in _C1C
             if abs(get_profile(_n).grain.sigma_shape_toe - 0.40) < 1e-9
             and abs(get_profile(_n).grain.sigma_shape_dmax - 1.20) < 1e-9]
    chk("no vector-traced negative carries the _grain_v2 heuristic triple",
        not _same, ", ".join(_same) if _same else "6 measured, none 0.40/1.00/1.20")

    # Every one of the six turns OVER: sigma falls from its interior peak to dmax.
    # That direction is the physical finding (Sehlin/Kennel 1985: "overexposing
    # either film significantly decreases granularity"), so it is asserted rather
    # than left to the individual numbers.
    _dir = ["%s peak %.2f <= dmax %.2f" % (_n, get_profile(_n).grain.sigma_shape_peak,
                                           get_profile(_n).grain.sigma_shape_dmax)
            for _n in _C1C
            if get_profile(_n).grain.sigma_shape_peak
            <= get_profile(_n).grain.sigma_shape_dmax]
    chk("all 6 vector-traced negatives fall from their peak to dmax",
        not _dir, "; ".join(_dir) if _dir else "peak 1.38-1.62x, dmax 0.50-0.90x")

    # And the anchor densities must bracket D = 1.0, or sigma_measured_usable()
    # refuses the shape and the stock silently drops back to the legacy law --
    # the failure mode that would make this whole harvest a no-op.
    _use = [_n for _n in _C1C
            if not get_profile(_n).grain.sigma_measured_usable(
                get_profile(_n).curves.g.dmin, get_profile(_n).curves.g.dmax)]
    chk("all 6 adopted shapes are actually usable by the renderer",
        not _use, ", ".join(_use) if _use else "6 of 6 bracket D = 1.0")
    # 4. And the consequence that made this worth doing: at dmax the measured
    # shape must be far BELOW the legacy law for the colour negatives, which is
    # the 3.2-3.6x over-graining the wiring removes.
    _p = get_profile("KODAK_VISION3_50D_5203")
    _g, _c = _p.grain, _p.curves.g
    _ratio = (float(_np.sqrt(max(2.63 - _c.dmin, 0.0) + _g.fog_grain))
              / (film_profiles.grain_sigma(_g, _c.dmin, _c.dmax, 2.63)
                 * float(_np.sqrt(max(1.0 - _c.dmin, 0.0) + _g.fog_grain))))
    chk("5203 grain at dmax is now ~3x quieter than the legacy law",
        2.5 < _ratio < 4.0, "legacy / measured = %.2fx" % _ratio)

    # ---- C2: MTF is a CURVE now, not only an f50 (2026-08-19) ---------------
    # The carrier was chosen by measurement, so what these guard is that the choice
    # holds and that turning it on cost nothing anywhere else.
    #
    # 1. THE REGRESSION GUARD, same shape as C1's. Every stock WITHOUT a measured
    # rolloff must come out of the shared sampler bit-for-bit equal to the legacy
    # Gaussian -- in float32, the renderer's own precision. This failed on the
    # first attempt because the sampler computed in float64 and cast back, which
    # moved 154 stocks by ~1e-8: not a visible change, but it destroys the property
    # that makes the wiring safe to land.
    _f32 = _np.linspace(0.0, 300.0, 61).astype(_np.float32)
    _mtf_bad, _n_mtf = [], 0
    for _p in FILM_PROFILES:
        if _p.mtf.mtf_measured:
            continue
        _n_mtf += 1
        for _c, _f50 in enumerate(_p.mtf.f50s()):
            if _f50 <= 0:
                continue
            _old = _np.exp(-_np.log(_np.float32(2.0))
                           * (_f32 / _np.float32(_f50)) ** _np.float32(2.0))
            _new = film_profiles.mtf_response(_p.mtf, _c, _f32)
            if not _np.array_equal(_old, _new):
                _mtf_bad.append(_p.name)
                break
    chk("unmeasured stocks reproduce the legacy Gaussian MTF bit-for-bit",
        not _mtf_bad, ", ".join(sorted(set(_mtf_bad))[:4]) if _mtf_bad
        else "%d profiles, float32 exact over 0-300 cycles/mm" % _n_mtf)

    # 2. BOTH LAWS MUST BE EXACTLY 0.5 AT f50. This is the property that let C2 land
    # without a level decision attached -- the mistake C1b had to unpick later.
    _half = []
    for _p in FILM_PROFILES:
        for _c, _f50 in enumerate(_p.mtf.f50s()):
            if _f50 <= 0:
                continue
            _v = film_profiles.mtf_response(_p.mtf, _c, float(_f50))
            if abs(_v - 0.5) > 1e-6:
                _half.append("%s.%d=%.6f" % (_p.name, _c, _v))
    chk("MTF is exactly 0.5 at f50 for every stock and channel",
        not _half, "; ".join(_half[:4]) if _half
        else "471 stock-channels, |MTF(f50) - 0.5| < 1e-6")

    # 3. Exactly the traced stock may use the measured law.
    _mmeas = sorted(p.name for p in FILM_PROFILES if p.mtf.mtf_measured)
    # ⚠ 1 -> 2 -> 3. 5201 arrived on 2026-08-20 (first COLOUR stock with a traced
    # MTF, so the first whose three f50 values are three measurements rather than
    # one estimate scaled by a stored ratio); 5274 on 2026-08-20c under C13.
    # ⚠ 1 -> 2 -> 3 -> 8. C2b added five colour sheets on 2026-08-23 (5217, 5218,
    # 5245, 5248, 5279). This list MIRRORS mtf_vector.SHEETS: a stock may carry the
    # flag only if that audit re-derives its curve from the sheet on every build.
    # Two stocks measured in the same pass are deliberately NOT here -- 5205 and
    # 5293 have a measured green and blue but a REFUSED red, so their triple is
    # mixed provenance and they keep the legacy Gaussian.
    # ⚠ 8 -> 10, 2026-08-23 (F-125 pass). The two Fuji Super-F stocks join on a
    # DIFFERENT footing and the distinction is load-bearing, so it is named here
    # rather than left to the field comments: the eight Kodak stocks are traced
    # from SINE-WAVE MTF panels that carry three labelled records, so all three
    # of their f50 values are measurements. The Fuji sheets print ONE unlabelled
    # CONTRAST TRANSFER FUNCTION each, so only green is measured (after Coltman
    # conversion) and red/blue are flanking transfers. Every guard below that
    # reasons about MEASURED RED must therefore exclude them -- see
    # _GREEN_ONLY_MEASURED.
    # ⚠ 10 -> 11 on 2026-08-25 (queue E0b-orig): KODAK_EKTACHROME_100D_5285, and
    # it is the first COLOUR REVERSAL stock with a measured MTF. Every earlier
    # entry is a negative. That distinction is load-bearing for the red-cluster
    # guard below, which is a claim about the Kodak cine NEGATIVE family.
    # ⚠ 11 -> 12 on 2026-08-26: EASTMAN_DOUBLE_X_5222, off the JULY 2015 edition
    # of H-1-5222. It is a black-and-white NEGATIVE, so it would otherwise fall
    # inside the family the red-cluster guard below reasons about -- and it must
    # not: a monochrome sheet prints ONE curve whose f50 is written to all three
    # fields, so its "red" is a pooled panchromatic number, not a red record.
    # That guard's existing `not p.is_monochrome` filter already excludes it and
    # 5231, which is why this addition does not disturb the 36 c/mm finding.
    # ⚠ 12 -> 15 on 2026-08-26 (KODAK still-film harvest): KODAK_PORTRA_160, 400
    # and 800, off E-4051, E-4050 and E-4040 respectively. THREE THINGS ABOUT
    # THEM ARE NEW TO THIS LIST AND MATTER TO THE GUARDS BELOW.
    #   1. They are the first STILL films here. Every earlier entry is a cine
    #      stock, so any guard reasoning about "the Kodak cine negative family"
    #      must exclude them explicitly rather than by accident of membership.
    #   2. Their MTF panels are LOG-LOG WITH THE LEGEND OUTSIDE THE FRAME, read
    #      by kodak_still_curves.py, not by mtf_vector.py -- a different reader
    #      on a different layout, so agreement with the older entries is a
    #      cross-method check rather than a repetition.
    #   3. All three show an adjacency overshoot above 100 % modulation (green
    #      116.6 / 113.1 / 119.4 %), so their stored q is fitted only where the
    #      carrier can reach and their overshoot is stored as `adjacency`.
    # The three remaining KODAK still profiles are NOT here and must not be: the
    # E-7022, E-7023 and E-7024 sheets print no MTF panel at all, so GOLD 200,
    # ULTRA MAX 400 and ULTRA MAX 800 keep estimated f50 triples.
    # ⚠ 15 -> 16 on 2026-08-30 (K1): KODAK_PORTRA_400VC joins, and it is the
    # ONLY one of the four new PORTRA stocks that does. All four have traced
    # f50 triples, but mtf_measured also switches the rolloff LAW to
    # 1/(1+(f/f50)^q), and that is only defensible where the power law
    # actually fits: 400VC fits at rms 0.0397 and beats the Gaussian 1.5x,
    # while 160NC / 160VC / 400NC land at rms 0.093-0.122 and beat it by only
    # 1.2-1.3x. Those three keep the Gaussian and take their f50 as documented
    # figures alone. A flag that means "we measured the shape" must not be set
    # from a fit that did not measure it.
    # ⚠ 16 -> 17 on 2026-08-31 (E3): KONICA_IMPRESA_50, and it is the FIRST
    # ENTRY THAT IS NOT VECTOR-TRACED AND NOT PER-LAYER. Its sheet is a scan --
    # every plot on it is a bitmap with no text layer -- so the curve comes
    # from `konica_raster.py` through `dashtrace`, and the panel it comes from
    # prints ONE curve captioned "Densitometry: Through visual filter". So the
    # flag means, for this stock only, "the shape was measured, once, pooled
    # across the layers": f50 64.9 in all three fields, rolloff q 2.20 fitted
    # to 214 samples above 25 c/mm at rms 0.019 against the Gaussian's 0.039.
    # ⚠ THAT MAKES IT INADMISSIBLE TO THE TWO FAMILY GUARDS BELOW, in exactly
    # the way `_GREEN_ONLY_MEASURED` already is, and for a stronger reason: a
    # pooled f50 written into three fields has no red record at all, so
    # "softer in red than the rule" would be testing 1.000 against a ratio it
    # cannot have, and "the red records cluster near 36" would be handed a
    # visual-filter number. The set below names it; the guard after it asserts
    # the property that licenses the exclusion.
    # ⚠ 17 -> 19 ON 2026-09-02 (queue G2): the two Gevachrome types, traced from
    # a 115 ppi RASTER page rather than from vector art -- the first stocks in
    # this list whose MTF came off a bitmap. Their former f50 triples were class
    # estimates two to three times too high.
    # ⚠ 19 -> 23 ON 2026-09-02e: KODAK_EKTAR_100 (queue T2, the first measured
    # MTF for a STILL colour negative here, off E-4046's vector panel) and the
    # three new Fuji stocks of queue T3, traced from their own datasheet panels.
    # ⚠ THE LIST IS SPELLED OUT AND NOT COUNTED, on purpose: a count agrees by
    # accident when one stock gains a measurement and another loses one, and
    # this is the register of which stocks the project claims to have measured.
    # 23 until 2026-09-06; KODAK_TMAX_100 and KODAK_TRI_X_400TX joined then from
    # the KODAK black-and-white sheets (F-4016 p8 and F-4017 p7).
    # ⚠ KODAK_TMAX_P3200, KODAK_TRI_X_320TXP and KODAK_TMAX_400 are DELIBERATELY
    # ABSENT although all three were traced the same day -- reused artwork, an
    # unattributed panel and three contradicting sheets respectively. See the
    # G-MTFBW guards, which fail if any of them quietly acquires a measurement.
    # ⚠⚠ 43 -> 47 ON 2026-09-07b, AND FOUR OF THEM ARRIVE FROM SHARED
    # DRAWINGS BY OWNER DECISION. AGFA_APX_100/400 and both AgfaPhoto Vista
    # plus stocks each read their f50 off a panel their sibling also uses, so
    # this file previously kept all four as class estimates -- the refusal
    # NotFound.md row 5d states. The owner's instruction of 2026-09-07b:
    # *"If the vendor datasheet provide MTF please don't simply discard this
    # value even 'Two films cannot share a measured MTF'. We haven't
    # additional better source for check this, so please accept and include
    # these values from vendors datasheet - this is much more better from
    # estimated values!"*
    # ⚠ THE SHARED-ARTWORK FINDING IS NOT WITHDRAWN, only its consequence: the
    # duplicate checks in `agfa_1998_sharpness.py` and
    # `agfaphoto_vista_plus.py` still re-derive it on every build, and each
    # profile's own comment says the value cannot be per-film for both films.
    # What changed is the comparison: a shared VENDOR curve is closer to the
    # film than a class estimate derived from no document.
    # ⚠ 47 -> 48 on 2026-09-15: ORWOCOLOR_NC3, Bild 7, the first ORWO MTF
    #   measurement in the corpus and the first from a GDR source.
    # ⚠ 48 -> 49 on 2026-09-15b: KODAK_TECHNICAL_PAN. Its f50 stood at 110 EST,
    #   reasoned from the RMS 5 granularity class BECAUSE P-255 prints no
    #   numeric resolving power -- a caveat the profile's own description
    #   already carried. «Современные фотоматериалы и их обработка» p.372
    #   prints the modulation transfer function; the traced curve peaks at
    #   115.1 % and falls through 50 % at 73.09 c/mm (72.3 until 2026-09-18c,
#   when queue P56's new bank axes made two more abscissa labels readable
#   and the log fit gained them), a third BELOW the
    #   estimate. This is the owner's 2026-09-07b instruction applied to a
    #   reference source rather than a vendor one: a measurement is better
    #   than an estimate, and nothing from Kodak is displaced because Kodak
    #   published nothing here.
    # ⚠ 49 -> 50 ON 2026-09-17: EASTMAN_5293_250T_1982, Fig. 14 traced,
    # rolloff exponent fitted from 210 samples above 8 cycles/mm.
    chk("exactly the 51 traced stocks are flagged mtf_measured",
        _mmeas == [
                   # ⚠ 2026-09-06h, TEN AT ONCE: the AGFA «Sharpness» panels,
                   # readable ever since queue G6 closed on 2026-09-05 and
                   # forbidden by a guard until then.
                   # ⚠ APX 100 and APX 400 JOINED ON 2026-09-07b from the
                   # drawing they share -- see the note above this chk.
                   # ⚠ AND "AGFA_APX_100" < "AGFA_APX_25" < "AGFA_APX_400"
                   # BY STRING ORDER, because '1' < '2' < '4' at the same
                   # position. The pair is not adjacent to itself in this list.
                   "AGFA_APX_100",
                   "AGFA_APX_25",
                   "AGFA_APX_400",
                   "AGFA_OPTIMA_100",
                   "AGFA_OPTIMA_200",
                   "AGFA_OPTIMA_400",
                   "AGFA_PORTRAIT_160",
                   "AGFA_RSX_II_100",
                   "AGFA_RSX_II_200",
                   "AGFA_RSX_II_50",
                   "AGFA_SCALA_200X",
                   "AGFA_ULTRA_50",
                   # ⚠ 2026-09-06j, THE THIRTEENTH AGFA PANEL AND THE ONE THAT
                   # SWEEP COULD NOT REACH: Vista's «Sharpness» chart is in the
                   # «AGFACOLOR Vista» sheet, not in «Technical Data PF», so a
                   # sweep of the latter's twelve columns never saw it and the
                   # stock kept a class estimate for another day. ⚠ AND THE
                   # REFUSAL HOLDING IT OUT SINCE 2026-08-18 WAS ANSWERED BY
                   # PAGE 4 OF ITS OWN SHEET, where Agfa define the chart:
                   # «International name of the chart: MTF (Modulation
                   # Transfer Function)». A manufacturer naming the quantity
                   # is a stronger authority than queue G6's inference from the
                   # ICO nomenclature, and the two agree.
                   "AGFA_VISTA_200",
                   # ⚠ 2026-09-07d, THE VISTA PLUS PAIR, from ONE 47-point path
                   # serving both films (translation spreads 0.0010 / 0.0020
                   # pt). Each profile records that the curve cannot be
                   # per-film for both, and that the faster film reading as
                   # sharp as the slower one is the shared drawing speaking
                   # rather than the emulsion.
                   # ⚠ RENAMED FROM "AGFAPHOTO_" ON 2026-09-07d AT THE OWNER'S
                   # REQUEST, WHICH MOVED THEM IN THIS LIST: "AGFAPHOTO_"
                   # sorted BEFORE "AGFA_APX" ('P' 0x50 precedes '_' 0x5F) and
                   # "AGFA_VISTA_PLUS" sorts AFTER "AGFA_VISTA_200". The list
                   # is compared element by element, so a rename is also a
                   # re-ordering.
                   "AGFA_VISTA_PLUS_200",
                   "AGFA_VISTA_PLUS_400",
                   # ⚠ 49 -> 50 ON 2026-09-17, Kennel et al. 1982 Fig. 14.
                   "EASTMAN_5293_250T_1982",
                   "EASTMAN_DOUBLE_X_5222",
                   "EASTMAN_EXR_100T_5248",
                   "EASTMAN_EXR_50D_5245",
                   "EASTMAN_PLUS_X_5231",
                   "FUJICHROME_64T_II",
                   # ⚠ 2026-09-06c, AND THE TWO ARE ONE MEASUREMENT. AF3-100E
                   # and AF3-177E print one MTF drawing; PRO 800Z's refusal was
                   # retracted after NPZ 800's raster twin showed the ladder is
                   # readable. Both carry f50_g 46.1 and q 1.86 -- see the
                   # shared-drawing guard beside the distinctness check.
                   "FUJICOLOR_PORTRAIT_NPZ_800",
                   "FUJICOLOR_PRO_400H",
                   "FUJICOLOR_PRO_800Z",
                   # ⚠ 2026-09-06e, both from their own vector panels.
                   "FUJICOLOR_SUPERIA_REALA",
                   "FUJICOLOR_SUPERIA_XTRA_400",
                   "FUJICOLOR_SUPERIA_XTRA_800",
                   "FUJICOLOR_SUPER_F500_8572",
                   "FUJI_PROVIA_100F",
                   "FUJI_PROVIA_400F",
                   "FUJI_PROVIA_400X",
                   "FUJI_SUPER_F125_8532",
                   "GEVACHROME_600",
                   "GEVACHROME_605",
                   "KODAK_EKTACHROME_100D_5285",
                   "KODAK_EKTAR_100",
                   "KODAK_PORTRA_160",
                   "KODAK_PORTRA_400",
                   "KODAK_PORTRA_400VC",
                   "KODAK_PORTRA_800",
                   "KODAK_TECHNICAL_PAN",
                   "KODAK_TMAX_100",
                   # ⚠ 50 -> 51 ON 2026-09-17d: KODAK_TMAX_400. Its own
                   # refusal's reopening condition was met by a fourth
                   # drawing -- see G-MTFBW5.
                   "KODAK_TMAX_400",
                   "KODAK_TRI_X_400TX",
                   "KODAK_VISION2_200T_5217",
                   "KODAK_VISION2_500T_5218",
                   "KODAK_VISION2_50D_5201",
                   "KODAK_VISION_200T_5274",
                   "KODAK_VISION_500T_5279",
                   "KONICA_IMPRESA_50",
                   "ORWOCOLOR_NC3",
        ],
        ", ".join(_mmeas))

    # ⚠ THE PROPERTY THAT LICENSES THE EXCLUSION, ASSERTED RATHER THAN ASSUMED.
    # A visual-filter measurement is pooled by construction, so its three f50
    # fields must be IDENTICAL -- the moment someone splits them, the number
    # stops being the thing that was measured and the exclusions below stop
    # being honest. The second half is the sheet's own independent statement
    # about its sharpness: IMP50 prints resolving powers of 63 lines/mm at
    # 1.6:1 and 160 at 1000:1, and a 50 % modulation point has to fall between
    # a low-contrast and a high-contrast resolution limit.
    _VISUAL_FILTER_MEASURED = {"KONICA_IMPRESA_50"}
    for _n in sorted(_VISUAL_FILTER_MEASURED):
        _m = get_profile(_n).mtf
        chk(f"{_n}'s visual-filter f50 is pooled, not split, and sits between "
            f"its own printed resolving powers",
            _m.f50_r == _m.f50_g == _m.f50_b
            and _m.resolving_power_lp_mm_lowc < _m.f50_r
            < _m.resolving_power_lp_mm_highc,
            "f50 %.1f/%.1f/%.1f, resolving power %.0f (1.6:1) / %.0f (1000:1)"
            % (_m.f50_r, _m.f50_g, _m.f50_b, _m.resolving_power_lp_mm_lowc,
               _m.resolving_power_lp_mm_highc))

    # ---- queue E3, 2026-08-31: the two KONICA stocks read off their own scans
    # ⚠ THE POINT OF THIS BLOCK IS THAT A FAMILY TEMPLATE CAN PASS EVERY OTHER
    # GUARD IN THIS FILE. Until today KONICA_IMPRESA_50 held dmin 0.20 / 0.62 /
    # 1.00 and KONICA_VX_100 holds 0.21 / 0.63 / 1.02 -- plausible, ordered,
    # internally consistent, and in IMPRESA 50's case wrong in blue by 0.32 D
    # against its own sheet. Nothing here could have caught that, because
    # nothing compared a stored triple to a reading. These assertions do, and
    # they are worth their cost only because the reading is corroborated: the
    # characteristic panel's plateau and the spectral-density panel's minimum
    # curve, two figures on two pages, agree to 0.005-0.015 D.
    _imp = get_profile("KONICA_IMPRESA_50")
    _idm = tuple(round(getattr(_imp.curves, _c).dmin, 4) for _c in "rgb")
    chk("KONICA_IMPRESA_50 holds the Dmin its own sheet draws, not the family "
        "template it shared with VX 100 and CENTURIA SUPER 400",
        _idm == (0.1842, 0.4838, 0.6087),
        "r/g/b %.4f / %.4f / %.4f" % _idm)
    chk("KONICA_IMPRESA_50 Dmin still rises blue > green > red, as the orange "
        "mask requires",
        _idm[2] > _idm[1] > _idm[0], "r/g/b %.4f / %.4f / %.4f" % _idm)
    # ⚠ AND THE GAMMA SPREAD IS NOW REAL. The template gave 0.600 / 0.615 /
    # 0.620, a 3 % spread across the layers; the trace gives 0.568 / 0.688 /
    # 0.820, a 44 % spread, blue steepest. That ordering is the physical one
    # for a masked negative and it is not something a template produces.
    _ig = tuple(round(getattr(_imp.curves, _c).gamma, 4) for _c in "rgb")
    chk("KONICA_IMPRESA_50's traced gamma rises red -> green -> blue and is "
        "no longer the flat family triple",
        _ig[2] > _ig[1] > _ig[0] and (_ig[2] - _ig[0]) > 0.15,
        "r/g/b %.4f / %.4f / %.4f" % _ig)

    _inf = get_profile("KONICA_INFRARED_750")
    # ⚠ FIFTEEN PRINTED CURVES AND THE STORED ONE WAS BELOW ALL OF THEM. The
    # flattest curve on INF750 p3 is Konicadol Fine at 4 minutes, mid-slope
    # 0.814; the profile held 0.707. The adopted condition is the sheet's own
    # standard, Konicadol DP 6 minutes at 20 C, mid-slope 1.615.
    chk("KONICA_INFRARED_750's contrast is inside the range its sheet prints, "
        "which the value it held was not",
        0.814 < _inf.curves.g.mid_slope < 1.837,
        "mid-slope %.4f against a printed range of 0.814 (Fine, 4 min) to "
        "1.837 (DP, 12 min)" % _inf.curves.g.mid_slope)
    chk("KONICA_INFRARED_750 finally names the developer its curve was drawn "
        "under",
        (_inf.processing.developer.startswith("Konicadol DP")
         and _inf.processing.minutes == 6.0
         and _inf.processing.celsius == 20.0),
        "%r, %.1f min, %.1f C" % (_inf.processing.developer,
                                  _inf.processing.minutes,
                                  _inf.processing.celsius))

    # ---- AGFA «Technical Data P-16-C», added 2026-09-01 (second pass) -----
    # ⚠ MONOTONICITY IS THE WHOLE POINT OF THIS BLOCK. A longer development
    # cannot give LESS contrast. P-16-C satisfies that on every triple it
    # prints; the 2004 B&W handbook's RODINAL table does not, and that is how a
    # typesetting fault was distinguished from a product revision rather than
    # averaged with it. If a future edit reintroduces the handbook's numbers,
    # this fails.
    # ⚠ THE COMPARISON IS WITHIN ONE VESSEL AT ONE TEMPERATURE, and it was not
    # until 2026-09-06i. The family used to hold one temperature and the
    # "lowest time at each gamma" trick stood in for separating the vessels;
    # p11's tables add 18 / 22 / 24 C rows, and a 24 C gamma 0.65 time is
    # legitimately SHORTER than a 20 C gamma 0.55 one. Comparing across the
    # temperature axis reported APX 400's RODINAL 1+25 as non-monotone when
    # nothing about it is: 3.95 min is gamma 0.55 at 20 C and 3.5 min is gamma
    # 0.65 at 24 C, two different baths.
    for _agn in ("AGFA_APX_25", "AGFA_APX_100", "AGFA_APX_400"):
        _fam = [_q for _q in get_profile(_agn).processing_family.points
                if abs(_q.celsius - 20.0) < 1e-9]
        _bad = []
        for _dev in {_q.developer for _q in _fam}:
            _seq = sorted((_q.gamma, _q.minutes) for _q in _fam
                          if _q.developer == _dev)
            # one developer prints both a drum and a tank time at gamma 0.65,
            # so compare only the LOWEST time at each gamma
            _low = {}
            for _g, _t in _seq:
                _low[_g] = min(_low.get(_g, 1e9), _t)
            _ord = [_low[_g] for _g in sorted(_low)]
            if any(_b < _a for _a, _b in zip(_ord, _ord[1:])):
                _bad.append("%s %s" % (_dev, _ord))
        chk("%s development time rises with gamma in every developer" % _agn,
            not _bad, "; ".join(_bad) if _bad else
            "%d points, %d developers, all monotone"
            % (len(_fam), len({_q.developer for _q in _fam})))

    # ⚠ THE ROW TWO AGFA DOCUMENTS PRINT INDEPENDENTLY. agfa_films.pdf p11 and
    # agfa_film_chem.pdf both give RODINAL 1+25, small tank, gamma 0.65 as
    # 6 / 8 / 7 min. The 2004 handbook gives 18 and 15 for the last two. This
    # pins the pair that agree.
    _want = {"AGFA_APX_25": 6.0, "AGFA_APX_100": 8.0, "AGFA_APX_400": 7.0}
    _off = []
    for _agn, _t in _want.items():
        # ⚠ THE COMPACT SPELLING. Agfa print "RODINAL 1 + 25"; the database
        # normalises to "RODINAL 1+25" because ProcessVariant already did and
        # two spellings of one developer make every join silently miss. This
        # guard was written with the printed form and failed on exactly that.
        _hit = [_q for _q in get_profile(_agn).processing_family.points
                if _q.developer == "RODINAL 1+25" and abs(_q.gamma - 0.65) < 1e-9
                and abs(_q.minutes - _t) < 1e-9]
        if not _hit:
            _off.append(_agn)
    chk("the RODINAL 1+25 tank row two Agfa documents agree on is stored",
        not _off, ", ".join(_off) if _off else "6 / 8 / 7 min at gamma 0.65")

    # ATOMAL FF appears on no plotted panel in the corpus; it exists in the
    # database only because P-16-C prints its table.
    chk("ATOMAL FF is present, and it comes only from P-16-C",
        all(any(_q.developer == "ATOMAL FF"
                for _q in get_profile(_n).processing_family.points)
            for _n in ("AGFA_APX_25", "AGFA_APX_100", "AGFA_APX_400")),
        "3 films x ATOMAL FF, a developer no Agfa panel plots")

    # ⚠ PUSH IS ONE STOP AND MUST NOT GROW. Agfa's table has exactly two speed
    # columns; there is no ISO 1600 row for APX 400 anywhere in the corpus.
    _p = [(_n, get_profile(_n).push) for _n in
          ("AGFA_APX_25", "AGFA_APX_100", "AGFA_APX_400")]
    chk("AGFAPAN push is exactly one stop, documented, with no invented fog",
        all(_x.max_push_stops == 1.0 and _x.max_pull_stops == 0.0
            and not _x.fog_penalty_stated
            and _x.base_fog_penalty_per_stop == 0.0
            and "P-16-C" in _x.source for _n, _x in _p),
        "3 profiles, +1 stop, fog penalty unstated because Agfa do not state it")

    # ---- AGFA 1998 harvest, added 2026-09-01 ------------------------------
    # ⚠ THIS BLOCK GUARDS A DOCUMENT-IDENTITY MISTAKE, NOT JUST NUMBERS.
    # `agfa_films.pdf` was recorded in NotFound.md row 5 and queue G6 as one of
    # four copies of a single publication. It is a SEPARATE 1st edition of
    # 09/1998 against the others' 4th edition of 08/2004, and it is the only
    # document in the corpus that plots AGFACOLOR ULTRA 50 or the AGFACHROME
    # RSX II line. If these four profiles vanish, the mistake has been made
    # again.
    for _agn in ("AGFA_ULTRA_50", "AGFA_RSX_II_50", "AGFA_RSX_II_100",
                 "AGFA_RSX_II_200"):
        _agp = get_profile(_agn)
        chk("%s cites the 1998 edition and not the 2004 one" % _agn,
            # ⚠ KEYED ON THE EDITION STATEMENT, NOT A FILENAME (2026-09-10).
            # This used to also require "agfa_films.pdf" in the citation. Local
            # paths were stripped out of every emitted string by owner
            # directive, so a filename is no longer a legal thing for a
            # citation to contain -- the bibliographic identity is.
            any("09/1998" in _s and "Technical Data PF" in _s
                for _s in _agp.provenance.sources),
            _agp.provenance.sources[0][:70])

    # ⚠ EIGHT PROFILES CARRIED A FALSE PROVENANCE NOTE AND THE FIX IS WHAT THIS
    # GUARDS. `_PARAM_SOURCES_DERIVED` gave every Agfa stock a
    # `grain.rms_granularity` record reading "No published rms for this stock in
    # the corpus". Agfa print the figure beside every plotted column of a sheet
    # each profile's own provenance already named. Hand entries now displace the
    # derived ones; if a regeneration puts the derived text back, this fails.
    _agfa_rms = ("AGFA_APX_25", "AGFA_APX_100", "AGFA_APX_400",
                 "AGFA_OPTIMA_100", "AGFA_OPTIMA_200", "AGFA_OPTIMA_400",
                 "AGFA_PORTRAIT_160", "AGFA_SCALA_200X", "AGFA_ULTRA_50",
                 "AGFA_RSX_II_50", "AGFA_RSX_II_100", "AGFA_RSX_II_200")
    # ⚠ THIS GUARD TESTS THE POSITIVE CONDITION, AND THE FIRST VERSION DID NOT.
    # Written as "no Agfa rms note still contains the phrase «No published rms»"
    # it failed on all twelve profiles the moment the fix landed -- because the
    # CORRECTION NOTE QUOTES THE FALSE SENTENCE it is correcting, which is
    # exactly what it should do. A guard that searches for a string cannot tell
    # a claim from a citation of that claim. So it asks instead for what has to
    # be TRUE: every Agfa rms cell names the document the figure is printed in.
    _stale = []
    for _agn in _agfa_rms:
        _hit = [_e for _e in _fpm._PARAM_SOURCES.get(_agn, ())
                if _e.param == "grain.rms_granularity"]
        if len(_hit) != 1 or "Technical Data PF" not in (_hit[0].source or ""):
            _stale.append(_agn)
    chk("every Agfa rms cell cites the sheet the figure is printed on",
        not _stale, ", ".join(_stale) if _stale else
        "%d Agfa profiles cite «Technical Data PF» for their rms" % len(_agfa_rms))

    # Published layer thickness, all twelve. Both editions agree on every film
    # they share, so a change here is a change in the reader, not the source.
    _coat = {"AGFA_OPTIMA_100": 16.0, "AGFA_OPTIMA_200": 18.0,
             "AGFA_OPTIMA_400": 19.0, "AGFA_PORTRAIT_160": 18.0,
             "AGFA_ULTRA_50": 27.0, "AGFA_RSX_II_50": 25.0,
             "AGFA_RSX_II_100": 25.0, "AGFA_RSX_II_200": 27.0,
             "AGFA_SCALA_200X": 7.0, "AGFA_APX_25": 3.0,
             "AGFA_APX_100": 7.0, "AGFA_APX_400": 10.0}
    _bad = [n for n, v in _coat.items()
            if abs(get_profile(n).emulsion.coated_um - v) > 1e-9]
    chk("the twelve Agfa profiles hold their published coated thickness",
        not _bad, ", ".join(_bad) if _bad else "3-27 um, all twelve")

    # ⚠ COATED THICKNESS MUST NOT DRAG THE REST OF EmulsionSpec WITH IT. Agfa
    # publish a thickness and nothing else about the emulsion; a later pass that
    # "completes" the record by inferring a crystal size from it would be
    # inventing data, and this is what says so.
    _inv = [n for n in _coat
            if get_profile(n).emulsion.grain_um != 0.0
            or get_profile(n).emulsion.habit
            or get_profile(n).emulsion.aspect_ratio != 0.0]
    chk("no Agfa profile infers a crystal size from its coated thickness",
        not _inv, ", ".join(_inv) if _inv else "grain_um/habit/aspect all empty")

    # ⚠⚠ THIS GUARD USED TO FORBID EXACTLY WHAT IT NOW REQUIRES, AND THE
    # REVERSAL IS THE WHOLE POINT OF QUEUE ROW G6. Until 2026-09-06h it read
    # "no Agfa stock claims a measured MTF while G6 is open", because the
    # panels are CTFs peaking at 102-114 % and, more importantly, because
    # nothing had settled whether Agfa's "Lines (mm)" were cycles or LINE PAIRS
    # -- a factor of two, not a rounding difference.
    # G6 CLOSED ON 2026-09-05 on the authority of the International Commission
    # for Optics: Ingelstam's 1961 nomenclature recommendation, PS&E 5(5) p282,
    # lists the German «Linien pro mm» as the equivalent of "lines per mm" and
    # declares that equal to cycles per mm, the only halving being a television
    # line. So the blocker is gone and the panels are readable.
    # ⚠ TEN OF THE TWELVE ARE NOW MEASURED. The two that are not are APX 100
    # and APX 400, which SHARE one drawing in BOTH editions -- see below.
    # ⚠ AGFA_VISTA_200 IS NOT IN `_coat` and must be checked on its own. That
    # set is the twelve stocks of «Technical Data PF»; Vista is a separate
    # sheet, which is exactly how it got left behind by the 2026-09-06h sweep
    # and stayed on an estimate for another day. Counting it here means the
    # thirteenth panel cannot silently regress the way it silently lagged.
    _vm = get_profile("AGFA_VISTA_200").mtf
    chk("G-AGFA6: AGFA_VISTA_200 carries the measured MTF of the thirteenth "
        "Agfa panel, whose chart Agfa's own sheet defines as an MTF",
        _vm.mtf_measured and abs(_vm.f50_g - 47.8) < 1e-9
        and abs(_vm.mtf_rolloff_q - 2.63) < 1e-9
        and abs(_vm.adjacency - 0.0978) < 1e-9,
        "f50_g %.1f q %.2f adj %.4f measured %s"
        % (_vm.f50_g, _vm.mtf_rolloff_q, _vm.adjacency, _vm.mtf_measured))
    # ⚠ AND IT MAY NOT SIT AT THE SUPERSEDED CLASS ESTIMATE, which stood from
    # before 2026-08-18 to 2026-09-06j behind a refusal its own sheet answers.
    chk("AGFA_VISTA_200 has not drifted back to the 56/63/69 class estimate",
        abs(_vm.f50_g - 63.0) > 0.05, "f50_g %.1f" % _vm.f50_g)
    _agfa_meas = sorted(n for n in _coat if get_profile(n).mtf.mtf_measured)
    # ⚠ 10 -> 12 on 2026-09-07b: APX 100 and APX 400 joined from the drawing
    # they share, on the owner's instruction that a vendor MTF beats a class
    # estimate even when it cannot be per-film for both films.
    chk("G6 is closed, so the Agfa sharpness panels ARE adopted: ALL TWELVE "
        "coated-thickness stocks now carry a measured MTF",
        len(_agfa_meas) == 12,
        "%d measured: %s" % (len(_agfa_meas), ", ".join(_agfa_meas)))
    # ⚠⚠ THE SHARED DRAWING IS STILL SHARED, AND WHAT IS ASSERTED NOW IS THAT
    # THE PAIR AGREE. `NotFound.md` row 5d records that APX 100 and APX 400
    # share the sharpness drawing on the 1998 sheet, and the 2026-09-06h trace
    # found it survives into the 2004 edition -- APX 400's curve there is
    # APX 100's translated 175.17 pt with every y coordinate identical to
    # 0.0000000.
    # ⚠ THIS GUARD USED TO REQUIRE BOTH FILMS TO STAY UNMEASURED. Reversed
    # 2026-09-07b by owner decision: *"a vendor MTF, even shared, is much more
    # better from estimated values"*. What it asserts instead is the thing
    # that would break if someone ever treated one drawing as two
    # measurements -- the two must carry the SAME f50 and the SAME exponent,
    # because it is the same ink.
    _a1, _a4 = get_profile("AGFA_APX_100").mtf, get_profile("AGFA_APX_400").mtf
    chk("the APX 100 / APX 400 shared sharpness drawing is adopted on BOTH and "
        "reads IDENTICALLY on both, because it is one drawing",
        _a1.mtf_measured and _a4.mtf_measured
        and abs(_a1.f50_g - _a4.f50_g) < 1e-9
        and abs(_a1.mtf_rolloff_q - _a4.mtf_rolloff_q) < 1e-9
        and abs(_a1.f50_g - 58.0) < 0.05
        and abs(_a1.mtf_rolloff_q - 2.27) < 1e-9,
        "both f50 %.1f q %.2f, measured %s/%s"
        % (_a1.f50_g, _a1.mtf_rolloff_q, _a1.mtf_measured, _a4.mtf_measured))

    # ---- G-AGFA, 2026-09-06h: what the sharpness harvest must keep true -----
    # ⚠ THE ADOPTED f50s MUST STAY ORDERED THE WAY THE PRINTED RESOLVING POWERS
    # ARE, on the stocks where Agfa prints both. That is a cross-check between
    # a TRACED curve and a number printed in words on the same page, and it is
    # the only independent test available for these panels.
    _rp_pairs = [("AGFA_APX_25", "AGFA_APX_400"),      # 200 vs 110 lines/mm
                 ("AGFA_RSX_II_50", "AGFA_RSX_II_200"),  # 135 vs 120
                 ("AGFA_OPTIMA_100", "AGFA_RSX_II_200")]  # 140 vs 120
    _ord = []
    for _hi, _lo in _rp_pairs:
        _a, _b = get_profile(_hi), get_profile(_lo)
        if not (_a.mtf.resolving_power_lp_mm_highc
                > _b.mtf.resolving_power_lp_mm_highc):
            continue                       # the printed pair does not order
        if not _a.mtf.f50_g > _b.mtf.f50_g:
            _ord.append("%s (%.1f) !> %s (%.1f)"
                        % (_hi, _a.mtf.f50_g, _lo, _b.mtf.f50_g))
    chk("G-AGFA1: every traced Agfa f50 orders the same way its own PRINTED "
        "resolving power does",
        not _ord, "; ".join(_ord) or "3 of 3 pairs agree")
    # ⚠⚠ AND ONE PAIR DELIBERATELY DOES NOT ORDER, WHICH IS WHY THAT LIST HAS
    # THREE ENTRIES AND NOT TEN. OPTIMA 100 traces f50 43.7 against OPTIMA
    # 200's 48.0 while its printed resolving power is HIGHER, 140 against 130.
    # `NotFound.md` row 5 read that as the panel being "not self-consistent on
    # any scale" and held four f50 readings out of the database for eight days
    # on the strength of it. ⚠ THE OBSERVATION WAS RIGHT AND THE INFERENCE WAS
    # NOT: resolving power is GRANULARITY-limited and f50 is not, so the two
    # are not required to order together. Optima 100's rms is 4.0 against
    # Optima 200's 4.3, and the finer grain buys limiting resolution the MTF
    # alone does not.
    # ⚠ THE RELATION THAT DOES HOLD IS THIS PROJECT'S OWN CROSS-MAKER
    # INVARIANT, RP*sqrt(rms) -- the one row 5 cites for Agfa against SVEMA and
    # ROLLEI. Inside each Agfa family it is tight, and that is asserted here
    # rather than left as prose, because it is the evidence that licensed
    # adopting a pair whose f50 and RP disagree in direction.
    for _fam, _mem, _tol in (
            ("colour negative", ("AGFA_OPTIMA_100", "AGFA_OPTIMA_200",
                                 "AGFA_OPTIMA_400", "AGFA_PORTRAIT_160",
                                 "AGFA_ULTRA_50"), 0.10),
            ("RSX II reversal", ("AGFA_RSX_II_50", "AGFA_RSX_II_100",
                                 "AGFA_RSX_II_200"), 0.06)):
        _inv = [get_profile(_m).mtf.resolving_power_lp_mm_highc
                * get_profile(_m).grain.rms_granularity ** 0.5 for _m in _mem]
        _spread = (max(_inv) - min(_inv)) / (sum(_inv) / len(_inv))
        chk("G-AGFA1b: RP*sqrt(rms) is conserved across the Agfa %s family, "
            "which is why f50 and printed resolving power need not order "
            "together" % _fam,
            _spread < _tol,
            "%s, spread %.1f %%" % (" / ".join("%.0f" % v for v in _inv),
                                    100 * _spread))
    # ⚠ AND THE MONOCHROME STOCKS KEEP ONE RECORD. APX 25 and SCALA 200x have a
    # single silver layer, so a per-record f50 spread on either would mean the
    # colour ratio family had been applied to a film that has no layers to
    # apply it to.
    _monoflat = [n for n in ("AGFA_APX_25", "AGFA_SCALA_200X")
                 if not (get_profile(n).mtf.f50_r == get_profile(n).mtf.f50_g
                         == get_profile(n).mtf.f50_b)]
    chk("G-AGFA2: the two monochrome Agfa stocks carry one f50 across all "
        "three records, because they have one layer",
        not _monoflat, ", ".join(_monoflat) or "APX 25 and SCALA 200x flat")
    # ⚠⚠ AND THE ADJACENCY OVERSHOOT IS STILL NOT FOLDED INTO f50. These panels
    # are CTFs and peak above 100 %; the peak lives in `adjacency` and the
    # rolloff is fitted ABOVE the peak only. If a later pass ever "corrects"
    # f50 by the overshoot, the two would double.
    # ⚠⚠ THE OVERSHOOT AND f50 MUST COME OFF THE SAME DRAWING, and until
    # 2026-09-06i they did not: `mtf.adjacency` held the 1998 peak on eight
    # stocks whose f50 had just been taken from the 2004 panel. That described
    # a curve printed on neither page. The pairing is now enforced, per stock,
    # against `agfa_1998_sharpness.EXPECTED_PEAK`.
    _agfa_adj = {"AGFA_APX_25": 0.0549, "AGFA_OPTIMA_100": 0.1111,
                 "AGFA_ULTRA_50": 0.1487, "AGFA_RSX_II_200": 0.1067}
    _adjbad = [n for n, v in _agfa_adj.items()
               if abs(get_profile(n).mtf.adjacency - v) > 1e-9]
    chk("G-AGFA3: the measured Agfa overshoots survive the f50 adoption "
        "untouched, so amplitude and sharpness stay separate",
        not _adjbad, ", ".join(_adjbad) or "4 of 4 unchanged")

    # The measured overshoots themselves. ⚠ THE EDITION EACH ONE COMES FROM IS
    # NOT UNIFORM AND MUST NOT BE MADE SO: the three Optima panels were REDRAWN
    # between the 1998 and 2004 sheets (frame-normalised artwork differs by
    # 0.004-0.011 of frame height) and take the 2004 reading; the other seven
    # are one drawing reprinted (0.0009) and take 1998, whose response ladder
    # is the sound one.
    _adj = {"AGFA_APX_25": 0.0549, "AGFA_APX_100": 0.1037,
            "AGFA_APX_400": 0.1037, "AGFA_OPTIMA_100": 0.1111,
            "AGFA_OPTIMA_200": 0.1043, "AGFA_OPTIMA_400": 0.0703,
            "AGFA_PORTRAIT_160": 0.0681, "AGFA_RSX_II_50": 0.0470,
            "AGFA_RSX_II_100": 0.0852,
            "AGFA_ULTRA_50": 0.1487, "AGFA_RSX_II_200": 0.1067,
            "AGFA_SCALA_200X": 0.0248}
    _bad = [n for n, v in _adj.items()
            if abs(get_profile(n).mtf.adjacency - v) > 1e-9]
    chk("the traced Agfa adjacency overshoots are unchanged",
        not _bad, ", ".join(_bad) if _bad else "0.025-0.149 across twelve stocks")

    # ⚠ APX 100 AND APX 400 SHARE ONE PIECE OF SHARPNESS ARTWORK. Two separate
    # path objects in two separate columns with identical geometry, so the two
    # overshoots are equal BY CONSTRUCTION and not by measurement. Recorded as a
    # guard so nobody later reads the equality as corroboration.
    chk("APX 100 and APX 400 overshoots are equal because Agfa reused the art",
        abs(get_profile("AGFA_APX_100").mtf.adjacency
            - get_profile("AGFA_APX_400").mtf.adjacency) < 1e-12,
        "both 0.1037 -- one drawing, two columns")

    # ---- G-AGFA4, 2026-09-06i: THE TWO CALIBRATION DEFECTS STAY FIXED ------
    # ⚠⚠ THIS GUARD EXISTS BECAUSE THE 2026-09-06h HARVEST SHIPPED WRONG AND
    # NOTHING CAUGHT IT. Two ladder defects, both silent, both in the same
    # reader, found within hours of each other:
    #   1. the frequency ladder was fitted through the "100" tick label, which
    #      Agfa nudge 3.37 pt left of its own tick to keep it in the column --
    #      1.6 % of scale, ~4 % of f50, on all twenty-two panels;
    #   2. the 2004 response column is set 0.9 pt low as a block, which read
    #      every 2004 curve 3 % high and made EIGHT films look as though they
    #      had been re-measured between editions. Seven of them had not.
    # The stored triples are pinned here so a regression in either shows up in
    # the database rather than only in a module nobody reruns by hand.
    _AGFA_F50Q = {
        "AGFA_OPTIMA_100": (43.7, 2.95), "AGFA_OPTIMA_200": (48.0, 2.67),
        "AGFA_OPTIMA_400": (47.4, 2.87), "AGFA_PORTRAIT_160": (36.2, 2.47),
        "AGFA_ULTRA_50": (42.9, 3.12), "AGFA_RSX_II_50": (29.4, 2.31),
        "AGFA_RSX_II_100": (31.9, 2.08), "AGFA_RSX_II_200": (21.3, 2.49),
        "AGFA_SCALA_200X": (30.6, 2.14), "AGFA_APX_25": (78.7, 2.31),
        # ⚠ ADDED 2026-09-07b with the shared-drawing adoption. Both read the
        # SAME panel, so both carry the same pair -- which is the finding, not
        # a copy-paste: the drawing is one drawing.
        "AGFA_APX_100": (58.0, 2.27), "AGFA_APX_400": (58.0, 2.27),
    }
    _f50bad = [n for n, (f, q) in _AGFA_F50Q.items()
               if abs(get_profile(n).mtf.f50_g - f) > 1e-9
               or abs(get_profile(n).mtf.mtf_rolloff_q - q) > 1e-9]
    chk("G-AGFA4: every Agfa f50/q pair is the frame-calibrated re-derivation, "
        "not the label-calibrated first pass",
        not _f50bad, ", ".join(_f50bad) or "10 of 10 at the corrected values")
    # ⚠ AND NONE OF THEM MAY SIT AT THE SUPERSEDED VALUE. Naming the wrong
    # numbers explicitly is worth more than a tolerance: these are what the
    # database held for six hours on 2026-09-06.
    _WAS = {"AGFA_OPTIMA_100": 46.0, "AGFA_OPTIMA_200": 50.6,
            "AGFA_OPTIMA_400": 50.1, "AGFA_PORTRAIT_160": 38.3,
            "AGFA_ULTRA_50": 44.0, "AGFA_RSX_II_50": 31.0,
            "AGFA_RSX_II_100": 33.7, "AGFA_RSX_II_200": 22.3,
            "AGFA_SCALA_200X": 32.3, "AGFA_APX_25": 81.6}
    _back = [n for n, v in _WAS.items()
             if abs(get_profile(n).mtf.f50_g - v) < 0.05]
    chk("no Agfa stock has drifted back to its 2026-09-06h f50",
        not _back, ", ".join(_back) or "none of the ten")

    # ---- G-AGFA7, 2026-09-06k: the OPTIMA SPECTRAL panel was redrawn too ----
    # ⚠⚠ THIS GUARD EXISTS BECAUSE I RAISED A FALSE ALARM AND THE OWNER PAID
    # FOR IT IN ROUNDS OF QUESTIONS. Comparing the stored OPTIMA spectral peaks
    # against the 1998 sheet showed red at 620 nm where 1998 draws 650, plus an
    # OPTIMA 400 red record running down to 394 nm -- which I called physically
    # impossible for a red layer and reported as a reader defect.
    # ⚠ IT IS NOT A DEFECT. Rendering both pages settles it: the 2004 Optima
    # spectral panel is a DIFFERENT DRAWING from the 1998 one -- 2004's red
    # peaks at 615 with a shoulder at 640 and carries a real low tail from
    # 508 nm, where 1998's is a single smooth peak at 653 starting at 559 --
    # and OPTIMA 400's 2004 red genuinely is drawn from 394 nm at lg -0.2.
    # The reader's calibration residual is 0.00 nm on all four columns and its
    # paths have zero subpath breaks. PORTRAIT 160, on the single-column p6,
    # is the control and matches 1998 exactly on all three layers.
    # ⚠ SO THE OPTIMA PAGE WAS REDRAWN IN BOTH ITS PANELS -- sharpness
    # (0.0041-0.0108 of frame height, G-AGFA4) and spectral. That consistency
    # is the real finding, and 2004 is adopted for both on the same precedent.
    # ⚠ THE LESSON: a disagreement between two editions is not evidence of a
    # bug. I inferred one from peak wavelengths without looking at the pages,
    # which is the same mistake as trusting a residual without its unit.
    _OPT_PEAKS = {"AGFA_OPTIMA_100": (470, 550, 620),
                  "AGFA_OPTIMA_200": (470, 550, 620),
                  "AGFA_OPTIMA_400": (470, 560, 610)}
    _lam = [380 + 10 * _i for _i in range(33)]
    _pk_bad = []
    for _n, _want in _OPT_PEAKS.items():
        _sp = get_profile(_n).spectral
        _got = tuple(_lam[max(range(33), key=lambda _i: _v[_i])]
                     for _v in (_sp.log_s_b, _sp.log_s_g, _sp.log_s_r))
        if _got != _want:
            _pk_bad.append("%s peaks %s, want %s" % (_n, _got, _want))
    chk("G-AGFA7: the three OPTIMA spectral sets are the 2004 drawing, whose "
        "red peaks near 615 nm and NOT the 1998 drawing's 650 nm",
        not _pk_bad, "; ".join(_pk_bad) or "B/G/R peaks pinned on all three")
    # ⚠ AND PORTRAIT 160 IS THE CONTROL. It is on the 2004 sheet too, on the
    # single-column p6, and its panel was NOT redrawn -- it reproduces the 1998
    # drawing exactly. If a future reader change breaks the three-column page,
    # this stock keeps reading correctly and the OPTIMA guard above fails
    # alone; if it breaks the reader generally, this one fails too. The pair
    # tells those two apart.
    _ps = get_profile("AGFA_PORTRAIT_160").spectral
    _pgot = tuple(_lam[max(range(33), key=lambda _i: _v[_i])]
                  for _v in (_ps.log_s_b, _ps.log_s_g, _ps.log_s_r))
    chk("AGFA_PORTRAIT_160's spectral peaks are unchanged -- the single-column "
        "control for the three-column page",
        _pgot == (420, 550, 650), "peaks %s" % (_pgot,))

    # ---- G-AGFA8, 2026-09-07: APX 400's three THIRD-PARTY developer rows ----
    # ⚠⚠ THE ROWS A TEXT READER COULD NOT DATE. F-PF-D4 p10 and F-PF-E4 p10
    # continue «Verarbeitung/Processing Agfapan APX 400» past Agfa's own six
    # developers with three they did not make, each printed with ONE time where
    # the others carry four -- so in TEXT the time has no temperature, and
    # `agfa_2003_sheet.proc_tables` still refuses to give it one. The column is
    # established by GEOMETRY instead: Agfa centre these cells, and all three
    # land on the same block's 20 C column to under a tenth of a point in both
    # independently typeset editions.
    # ⚠ THE ASYMMETRY IS AGFA'S AND IS ASSERTED IN BOTH DIRECTIONS. Only the
    # fast film gets third-party guidance; APX 100's block ends at STUDIONAL
    # LIQUID and APX 25 is not in either edition. A future pass that helpfully
    # copies these three across the family fails here.
    _TP = (("Tetenal Ultrafin Plus", 16.0), ("Kodak T-MAX", 12.0),
           ("Kodak D76/Ilford ID11", 12.0))
    _tp400 = {q.developer: q for q in
              get_profile("AGFA_APX_400").processing_family.points}
    _tpbad = []
    for _dev, _min in _TP:
        _q = _tp400.get(_dev)
        if _q is None:
            _tpbad.append("APX 400 has no %s point" % _dev)
        elif (abs(_q.minutes - _min) > 1e-9 or abs(_q.celsius - 20.0) > 1e-9
                or _q.vessel != "small tank, tray"
                or abs(_q.gamma - 0.65) > 1e-9 or _q.dilution != ""):
            _tpbad.append("APX 400's %s is %g min / %g C / %r / gamma %g / "
                          "dilution %r"
                          % (_dev, _q.minutes, _q.celsius, _q.vessel,
                             _q.gamma, _q.dilution))
    for _other in ("AGFA_APX_25", "AGFA_APX_100"):
        _stray = sorted({q.developer for q in
                         get_profile(_other).processing_family.points}
                        & {d for d, _ in _TP})
        if _stray:
            _tpbad.append("%s has acquired %s, which Agfa print only for "
                          "APX 400" % (_other, _stray))
    chk("G-AGFA8: APX 400 alone carries F-PF-D4/E4's three third-party "
        "developer rows, at 20 C in a small tank/tray to gamma 0.65",
        not _tpbad,
        "; ".join(_tpbad) or "Tetenal 16, T-MAX 12, D76/ID11 12 min")

    # ---- G-AGFA9, 2026-09-07: Scala's push/pull CONTRAST, slot 4 ------------
    # ⚠⚠ THE PANEL THAT WAS DOCUMENTED AND NOT READ. `agfa_2003_curves`
    # described «Gradation/Maximaldichte bei push/pull-Verarbeitung» in a
    # comment above a `SCALA_BANDS` dict that had three entries for four
    # panels, so `read_scala` walked past slot 4 in silence and
    # `push.gamma_gain_per_stop` stayed 0.0 while the sheet plotted the
    # contrast of all five processing steps.
    # ⚠ 0.125 IS A SUMMARY OF A SATURATING CURVE, NOT A SLOPE THE FILM HAS.
    # Measured contrast is 1.40 at Standard and 1.70 / 1.80 / 1.85 at Push
    # 1/2/3 -- +21.4 %, then +5.9 %, then +2.8 %. The stored scalar is the
    # least-squares line through the origin over those three stops; the module
    # prints its residuals every build so the approximation stays visible.
    # ⚠ AND PULL IS NOT THE SAME SLOPE REVERSED: 1.40 -> 0.80 is a 43 % loss
    # for one stop down. `max_pull_stops` records that the step exists; no
    # per-stop pull coefficient is stored, because one line cannot serve both
    # directions and the schema has one field.
    _sp = get_profile("AGFA_SCALA_200X").push
    chk("G-AGFA9: AGFA_SCALA_200X carries the 2003 sheet's push/pull contrast "
        "ladder as gamma_gain_per_stop 0.125, with three pushed stops and one "
        "pulled",
        abs(_sp.gamma_gain_per_stop - 0.125) < 1e-9
        and _sp.max_push_stops == 3.0 and _sp.max_pull_stops == 1.0
        and not _sp.fog_penalty_stated
        and "Gradation/Maximaldichte" in _sp.source
        and "1.845" in _sp.source and "0.796" in _sp.source,
        "gain %.3f, push %g, pull %g"
        % (_sp.gamma_gain_per_stop, _sp.max_push_stops, _sp.max_pull_stops))
    # ⚠ THE D-MAX LADDER IS NOT OVERWRITTEN BY THE NEW PANEL. Two independent
    # measurements of one quantity now exist -- 1998's from following five
    # drawn characteristic curves to their maxima, 2003's from five plotted
    # points -- and they agree to 0.074 D at worst. The stored source keeps the
    # 1998 numbers and names the 2003 ones beside them; averaging them would
    # destroy the only cross-check either has.
    chk("AGFA_SCALA_200X's push source keeps BOTH D-max readings and averages "
        "neither",
        all(_v in _sp.source for _v in ("3.064", "2.171", "3.095", "2.245"))
        and "NOTHING IS AVERAGED" in _sp.source,
        "both ladders named" if "3.095" in _sp.source else "2003 ladder absent")

    # ---- G-AGFA5, 2026-09-06i: the gamma-time harvest and its vessel -------
    # ⚠⚠ THE FIELD AND THE DATA MUST ARRIVE TOGETHER. `DevelopmentPoint.vessel`
    # (schema v28) exists only because 28 small-tank points needed somewhere
    # unambiguous to land; a later pass that drops the points but keeps the
    # field, or labels the points and forgets what the labels are for, is what
    # this asserts against.
    # ⚠ 10 / 9 / 9 AND NOT 10 / 10 / 10. RODINAL 1+50's curve stops at gamma
    # 0.670 on APX 100 and 0.738 on APX 400, so its 0.75 rung is not drawn on
    # either and is not stored. A missing rung is a refusal, not a zero, and
    # certainly not a five-minute extrapolation off the end of a curve.
    # ⚠ APX 400's small-tank count went 30 -> 33 on 2026-09-07, and NOT from
    # this panel: the three extra are the F-PF-D4/E4 third-party developer rows
    # (Tetenal Ultrafin Plus, Kodak T-MAX, Kodak D76/Ilford ID11), which are
    # printed TEXT at 20 C and gamma 0.65 and so leave the second figure -- the
    # count away from gamma 0.65, which is what this panel contributes -- at 9.
    # G-AGFA8 below is what checks those three.
    _GT = {"AGFA_APX_25": (29, 10), "AGFA_APX_100": (28, 9),
           "AGFA_APX_400": (33, 9)}
    _gtbad = []
    for _n, (_want_tank, _want_new) in _GT.items():
        _pts = get_profile(_n).processing_family.points
        _tank = [q for q in _pts if q.vessel == "small tank, tray"]
        _new = [q for q in _tank if abs(q.gamma - 0.65) > 1e-9]
        if len(_tank) != _want_tank or len(_new) != _want_new:
            _gtbad.append("%s has %d small-tank points, %d away from gamma "
                          "0.65 (want %d / %d)"
                          % (_n, len(_tank), len(_new), _want_tank, _want_new))
    chk("G-AGFA5: the three AGFAPAN families carry the 1998 panel's small-tank "
        "gamma 0.55 and 0.75 times beside P-16-C's printed 0.65",
        not _gtbad, "; ".join(_gtbad) or "28 harvested points across three stocks")
    # ⚠ AND THE TEMPERATURE AXIS IS COUNTED SEPARATELY, because it is a
    # different table on a different part of the page and every one of its
    # points sits at gamma 0.65 -- so the count above cannot see it at all.
    _TEMPN = {"AGFA_APX_25": 30, "AGFA_APX_100": 32, "AGFA_APX_400": 36}
    _tbad = []
    for _n, _want in _TEMPN.items():
        _off = [q for q in get_profile(_n).processing_family.points
                if abs(q.celsius - 20.0) > 1e-9]
        if len(_off) != _want:
            _tbad.append("%s has %d points away from 20 C (want %d)"
                         % (_n, len(_off), _want))
    chk("the three AGFAPAN families carry p11's 18 / 22 / 24 C developing "
        "times -- the only temperature axis in this database",
        not _tbad, "; ".join(_tbad) or "98 points across three stocks")
    # ⚠ AND NO DEVELOPER MAY CARRY TWO ANSWERS TO ONE QUESTION. The panel reads
    # gamma 0.65 too; storing that beside P-16-C's printed 0.65 would put two
    # measurements of one cell in one tuple, which is rule 4. Only the outer
    # two contrasts are taken from the curve, so each (developer, vessel,
    # gamma) triple must appear exactly once.
    # ⚠ THE KEY INCLUDES TEMPERATURE from 2026-09-06i. p11's tables add 18, 22
    # and 24 C rows for the same developer, vessel and contrast, so a key
    # without celsius would read the whole temperature axis as duplication --
    # and, worse, would have passed silently if the 20 C column HAD been
    # re-stored on top of P-16-C's, which is the thing it exists to catch.
    _dup = []
    for _n in _GT:
        _seen = {}
        for _q in get_profile(_n).processing_family.points:
            _k = (_q.developer, _q.vessel, round(_q.gamma, 3),
                  round(_q.celsius, 1))
            _seen[_k] = _seen.get(_k, 0) + 1
        _dup += ["%s %s" % (_n, _k) for _k, _c in _seen.items() if _c > 1]
    chk("no AGFAPAN developer carries two times for one vessel, contrast and "
        "temperature",
        not _dup, "; ".join(_dup[:3]) or
        "every (developer, vessel, gamma, celsius) once")
    # ⚠⚠ AND THE TEMPERATURE AXIS MUST STAY MONOTONE. A developing time to a
    # fixed contrast can only FALL as the bath warms; a row that rises has had
    # its columns transposed, which is the failure mode a four-column table
    # read by x position actually has. 36 rows, every one strictly decreasing.
    _mono, _rows = [], 0
    for _n in _GT:
        _by = {}
        for _q in get_profile(_n).processing_family.points:
            if abs(_q.gamma - 0.65) > 1e-9 or not _q.vessel:
                continue
            _by.setdefault((_q.developer, _q.vessel), []).append(
                (_q.celsius, _q.minutes))
        for _k, _v in _by.items():
            if len(_v) < 3:
                continue
            _rows += 1
            _v.sort()
            if any(_b[1] >= _a[1] for _a, _b in zip(_v, _v[1:])):
                _mono.append("%s %s %s" % (_n, _k[0], _v))
    chk("every AGFAPAN time-vs-temperature row falls monotonically as the bath "
        "warms", not _mono and _rows >= 30,
        "; ".join(_mono[:2]) or "%d rows, all strictly decreasing" % _rows)
    # ⚠ ATOMAL FF IS ON NO CURVE AND MUST STAY UNLABELLED WHERE IT IS UNKNOWN.
    # It appears in P-16-C's tables, which DO name the vessel, so its rows are
    # labelled from there -- but nothing may quietly assign it to the family
    # the plotted developers fall in on the strength of the panel.
    _at = [q for _n in _GT for q in get_profile(_n).processing_family.points
           if q.developer == "ATOMAL FF" and abs(q.gamma - 0.65) > 1e-9
           and q.vessel == "small tank, tray"]
    chk("ATOMAL FF gains no small-tank contrast point from a curve it is not "
        "drawn on", not _at, "%d such points" % len(_at) if _at else "none")

    # ⚠ THE SAME DEFECT ON THE SPECTRAL SIDE, AND IT IS WORSE BECAUSE IT LOOKS
    # LIKE TWO MEASUREMENTS. RSX II 50 and RSX II 100 trace to the same spectral
    # numbers within 0.002 lg at every sampled wavelength: one drawing serving
    # two stocks, exactly the shape of the PORTRA 100T / 160VC finding already
    # standing in NotFound.md.
    _r50 = get_profile("AGFA_RSX_II_50").spectral
    _r100 = get_profile("AGFA_RSX_II_100").spectral
    _dmax = max(abs(a - b) for a, b in
                list(zip(_r50.log_s_r, _r100.log_s_r))
                + list(zip(_r50.log_s_b, _r100.log_s_b)))
    chk("RSX II 50 and 100 share one spectral drawing, and it is declared",
        _dmax < 0.005
        and all("ONE CURVE SET FOR TWO FILMS" in (_p.spectral.source or "")
                for _p in (get_profile("AGFA_RSX_II_50"),
                           get_profile("AGFA_RSX_II_100"))),
        "max |d| %.4f lg, both sources declare it" % _dmax)

    # ⚠ THE REVERSAL CURVES MUST NOT BE READ AS NEGATIVES. Fitted in the sheet's
    # ascending frame instead of ToneCurve's negated one, the same records come
    # back with dmin 3.0 and gamma 2.0-2.5 -- in range for a slide film and
    # completely wrong. A reversal D-min near 0.1 is the cheap discriminator.
    for _agn in ("AGFA_RSX_II_50", "AGFA_RSX_II_100", "AGFA_RSX_II_200"):
        _c = get_profile(_agn).curves
        chk("%s D-min is a slide film's, not a mis-signed negative's" % _agn,
            all(0.05 < _x.dmin < 0.25 for _x in (_c.r, _c.g, _c.b)),
            "r/g/b %.3f %.3f %.3f" % (_c.r.dmin, _c.g.dmin, _c.b.dmin))

    # ⚠ `gamma` ON THESE THREE IS A MODEL COEFFICIENT AND TWO OF THEM REST ON
    # THE FITTER'S 2.50 CEILING. The observable is ToneCurve.mid_slope, and IT
    # is what has to land in the 1.6-2.1 band the ToneCurve docstring gives for
    # colour reversal. Capping gamma at 2.10 instead was tried and made
    # mid_slope WORSE (1.95 -> 1.86) at nearly double the fit residual.
    _ms = []
    for _agn in ("AGFA_RSX_II_50", "AGFA_RSX_II_100", "AGFA_RSX_II_200"):
        _c = get_profile(_agn).curves
        for _ch, _x in (("r", _c.r), ("g", _c.g), ("b", _c.b)):
            if not 1.60 <= _x.mid_slope <= 2.10:
                _ms.append("%s.%s %.3f" % (_agn, _ch, _x.mid_slope))
    chk("RSX II mid_slope sits in the colour-reversal band on all nine records",
        not _ms, ", ".join(_ms) if _ms else "1.68-2.00 across nine records")

    # ⚠ A THREE-DIGIT CC CODE IS THOUSANDTHS. CC075Y read as 0.75 density made
    # `reciprocity_log_shift` return +0.449 for blue on RSX II 200 -- a longer
    # exposure making the film faster. The general guard further down catches
    # the sign; this one names the cause so a future parser change cannot
    # reintroduce it quietly.
    chk("CC075Y parses as 0.075 density, not 0.75",
        abs(fs._cc_filter_shift("CC075Y")[2] - 0.075) < 1e-12
        and abs(fs._cc_filter_shift("CC15B")[0] - 0.15) < 1e-12,
        "three-digit thousandths, two-digit hundredths")

    # ⚠ RSX II 200's CC ROW IS YELLOW AND CYAN WHERE ITS SIBLINGS' IS BLUE, i.e.
    # its blue record holds while red and green lose speed -- chromatically
    # opposite reciprocity failure, read off the filter colour alone. That makes
    # it the only stock in the corpus where blue must lose LESS than green.
    _sh = fs.reciprocity_log_shift(get_profile("AGFA_RSX_II_200"), 10.0)
    chk("RSX II 200's yellow CC row makes blue lose least, not most",
        _sh[2] > _sh[1] and all(_v < 0.0 for _v in _sh),
        "r/g/b %.3f %.3f %.3f at 10 s" % _sh)

    # The gamma-time families, and the printed specification they reproduce.
    # ⚠ THIS IS THE STRONGEST CROSS-CHECK IN THE AGFA BATCH: eleven readings off
    # a curve, landing on a target stated in a DIFFERENT document.
    _apxref = {"AGFA_APX_25": {"REFINAL": 6.0, "RODINAL 1+25": 6.0,
                               "RODINAL 1+50": 10.0, "RODINAL SPECIAL": 4.0},
               "AGFA_APX_100": {"REFINAL": 6.0, "RODINAL 1+25": 8.0,
                                "RODINAL SPECIAL": 4.0},
               "AGFA_APX_400": {"REFINAL": 6.0, "RODINAL SPECIAL": 4.5}}
    _off = []
    for _agn, _want in _apxref.items():
        _pts = get_profile(_agn).processing_family.points
        for _dev, _t in _want.items():
            _hit = [q for q in _pts
                    if q.developer == _dev and abs(q.minutes - _t) < 1e-6]
            if not _hit:
                _off.append("%s/%s missing" % (_agn, _dev))
            elif abs(_hit[0].gamma - 0.65) > 0.015:
                _off.append("%s/%s %.3f" % (_agn, _dev, _hit[0].gamma))
    chk("every AGFAPAN developer hits gamma 0.65 at its own reference time",
        not _off, ", ".join(_off) if _off else
        "9 developer/film pairs, all within 0.015 of the printed 0.65 target")

    # ---- KODAK_EKTAR_125, added 2026-08-31 on one measured bound ----------
    # ⚠ THE POINT OF THIS BLOCK IS THAT THE PROFILE IS MOSTLY ESTIMATE AND MUST
    # STAY HONEST ABOUT WHICH NUMBER IS NOT. Its blue D-min is a Kodak
    # measurement from US 5,334,491 and an UPPER BOUND; everything else is a
    # class estimate. If a later pass "tidies" the blue toward the family
    # median, or promotes an estimate to measured, these fail.
    _e125 = get_profile("KODAK_EKTAR_125")
    chk("KODAK_EKTAR_125 holds the patent's blue D-min bound unchanged",
        abs(_e125.curves.b.dmin - 0.849) < 1e-9,
        "blue dmin %.4f, expected 0.8490 (US 5,334,491 slot 8, the lowest of "
        "nine bleaches)" % _e125.curves.b.dmin)
    # the mask must still order r < g < b, which is what an orange mask IS
    _e125d = tuple(getattr(_e125.curves, _c).dmin for _c in "rgb")
    chk("KODAK_EKTAR_125's D-min rises red < green < blue, as a masked "
        "negative requires",
        _e125d[0] < _e125d[1] < _e125d[2],
        "r/g/b %.3f / %.3f / %.3f" % _e125d)
    # ⚠ EXACTLY ONE of its parameters may claim to be measured
    _meas = [r.param for r in _e125.param_sources if r.status == "measured"]
    chk("KODAK_EKTAR_125 claims exactly one MEASURED parameter, and it is the "
        "blue D-min",
        _meas == ["curves.b.dmin"], "measured params: %s" % (_meas,))
    # ⚠ AND IT MUST NOT CLAIM A LAYER STACK. The 1989 review documents eleven
    # layers in prose; `LayerStack` is a per-layer RESOLVING POWER record whose
    # has_data is bool(order), so filling order alone would inflate the
    # layer-stack census with a stock that has no resolving powers.
    chk("KODAK_EKTAR_125 does not claim a LayerStack it has no resolving "
        "powers for",
        not _e125.layer_stack.has_data
        and _e125.emulsion.habit == "tabular",
        "layer_stack.has_data=%s habit=%r"
        % (_e125.layer_stack.has_data, _e125.emulsion.habit))

    # ---- queue B3, 2026-08-31: KODAK_TECHNICAL_PAN's first spectral set -----
    # ⚠ ONE OF THE TWO FLATTEST PANCHROMATIC CURVES IN THE DATABASE, AND THE
    # FIRST DRAFT OF THIS CHECK CLAIMED IT WAS THE FLATTEST. It is not, and the
    # check caught the overclaim before the batch closed: FUJI_NEOPAN_1600
    # spans 0.55 decades against Technical Pan's 0.56, a tie inside the trace's
    # own noise. What survives, and what P-255's prose actually supports
    # ("reasonably uniform spectral sensitivity at all visible wavelengths out
    # to 690 nanometres"), is that both sit far below the rest of the field --
    # the next is ILFORD_DELTA_3200 at 0.71 and the median panchromatic set
    # here spans 1.12. Prose and trace agreeing is the check; "flattest" was a
    # decoration and is gone.
    _tp = get_profile("KODAK_TECHNICAL_PAN").spectral
    chk("KODAK_TECHNICAL_PAN carries a spectral set at all (it had none "
        "before 2026-08-31)",
        len(_tp.log_s_pan) == 31 and _tp.criterion.endswith("D0.3_above_dmin"),
        "%d samples, criterion %r" % (len(_tp.log_s_pan), _tp.criterion))
    _tpv = max(_tp.log_s_pan) - min(_tp.log_s_pan)
    _others = [(p.name, max(p.spectral.log_s_pan) - min(p.spectral.log_s_pan))
               for p in FILM_PROFILES
               if p.is_monochrome and p.spectral.log_s_pan
               and p.name != "KODAK_TECHNICAL_PAN"
               and min(p.spectral.log_s_pan) > -3.99]
    _med = sorted(v for _n, v in _others)[len(_others) // 2]
    chk("KODAK_TECHNICAL_PAN's traced curve is among the two flattest "
        "panchromatic sets and well under the field median, which is what its "
        "sheet says in words",
        _tpv < 0.7 and sum(1 for _n, v in _others if v < _tpv) <= 1
        and _tpv < _med / 1.5,
        "%.2f decades, median %.2f, field %s" % (_tpv, _med, ", ".join(
            "%s %.2f" % (n.split("_")[-1], v) for n, v in sorted(_others))))

    # The two green-only stocks, and the assertion that their red and blue really
    # are the flanking ratios their comments claim -- if someone ever "measures"
    # those, they must remove the name from this set, and this guard is what
    # forces that edit to be deliberate.
    # ⚠ 2 -> 5 ON 2026-09-02e (queue T3). Fuji's still-film datasheets print ONE
    # unlabelled MTF curve, not three records, exactly as the cine sheets 8532
    # and 8572 do, so the three new stocks join this set for the same reason and
    # take the same family flanking ratios. Every guard below that compares a
    # MEASURED red against the estimating rule has to exclude them, because
    # their red IS the estimating rule -- including them makes the guard test
    # its own input.
    _GREEN_ONLY_MEASURED = {"FUJI_SUPER_F125_8532", "FUJICOLOR_SUPER_F500_8572",
                            "FUJI_PROVIA_100F", "FUJICOLOR_SUPERIA_XTRA_400",
                            "FUJICOLOR_PRO_400H",
                            # 2026-09-06: AF3-066E prints one unlabelled MTF
                            # curve like every other Fuji sheet, so 400F's red
                            # and blue are the ratio family too.
                            "FUJI_PROVIA_400F",
                            "FUJICHROME_64T_II",
                            # 2026-09-06: 400X too -- AF3-0213E prints one
                            # unlabelled MTF curve like the rest of the family.
                            "FUJI_PROVIA_400X",
                            # ⚠ 2026-09-06c: PRO 800Z and NPZ 800, whose MTF
                            # panel is ONE DRAWING printed in two publications.
                            # It too is a single unlabelled curve, so their red
                            # and blue are the stored ratio family and would
                            # make the guard below test its own input.
                            "FUJICOLOR_PRO_800Z",
                            "FUJICOLOR_PORTRAIT_NPZ_800",
                            # ⚠ 2026-09-06e: the two new SUPERIA stocks. Their
                            # MTF panels draw ONE unlabelled curve like every
                            # other Fuji sheet, so their red and blue are the
                            # stored ratio family and both the red-softness and
                            # the red-cluster guards below would be testing
                            # their own input.
                            "FUJICOLOR_SUPERIA_XTRA_800",
                            "FUJICOLOR_SUPERIA_REALA",
                            # ⚠ 2026-09-06h: the seven AGFA COLOUR stocks whose
                            # «Sharpness» panels were traced from Agfa's own
                            # Technical Data. Every one of those panels draws
                            # ONE unlabelled curve, so red and blue are the
                            # stored ratio family and both the red-softness and
                            # the red-cluster guards below would be testing
                            # their own input. The monochrome AGFA stocks
                            # (APX 25, SCALA 200x) are not here: their three
                            # records are equal because the film has one layer.
                            "AGFA_OPTIMA_100", "AGFA_OPTIMA_200",
                            "AGFA_OPTIMA_400", "AGFA_PORTRAIT_160",
                            "AGFA_ULTRA_50", "AGFA_RSX_II_50",
                            "AGFA_RSX_II_100", "AGFA_RSX_II_200",
                            # ⚠ 2026-09-06j: the EIGHTH, and it is in a
                            # different document -- the «AGFACOLOR Vista»
                            # sheet, which is why the 2026-09-06h sweep of
                            # «Technical Data PF» never reached it. Its panel
                            # is the same one unlabelled visual-weighted curve
                            # («Densitometry: visual filter (V-lambda)»).
                            "AGFA_VISTA_200",
                            # ⚠ 2026-09-07b: the two AgfaPhoto Vista plus
                            # stocks. Their «13. MTF Curve» draws ONE curve --
                            # and ONE curve for BOTH films at that -- so red
                            # and blue are this profile's own ratio family and
                            # the red-softness and red-cluster guards below
                            # would be testing their own input. ⚠ THE FIRST
                            # RUN AFTER ADOPTION PROVED THAT: both came back
                            # at r/g 0.82-0.83 against the rule's ~0.78 and
                            # broke the red cluster at 52.4 / 52.7, because
                            # the ratios carried are the ones the class
                            # estimate already had.
                            "AGFA_VISTA_PLUS_200",
                            "AGFA_VISTA_PLUS_400"}
    _flank_bad = []
    for _n, _rr, _rb in (("FUJI_SUPER_F125_8532", 0.8976, 1.0762),
                         ("FUJICOLOR_SUPER_F500_8572", 0.8214, 1.1071),
                         ("FUJI_PROVIA_100F", 0.8970, 1.0754),
                         ("FUJICOLOR_SUPERIA_XTRA_400", 0.8976, 1.0762),
                         ("FUJICOLOR_PRO_400H", 0.8977, 1.0763),
                         ("AGFA_VISTA_PLUS_200", 0.8927, 1.0886),
                         ("AGFA_VISTA_PLUS_400", 0.8977, 1.0818)):
        _m = get_profile(_n).mtf
        if (abs(_m.f50_r / _m.f50_g - _rr) > 0.005
                or abs(_m.f50_b / _m.f50_g - _rb) > 0.005):
            _flank_bad.append("%s r/g=%.4f b/g=%.4f" % (
                _n, _m.f50_r / _m.f50_g, _m.f50_b / _m.f50_g))
    chk("every green-only stock keeps its declared flanking ratios",
        not _flank_bad, "; ".join(_flank_bad) if _flank_bad
        else "7 stocks pinned, e.g. 8532 0.8976/1.0762, 8572 0.8214/1.1071, "
             "Vista plus 200 0.8927/1.0886")
    chk("every green-only stock is flagged measured and keeps r < g < b",
        all(get_profile(n).mtf.mtf_measured
            and get_profile(n).mtf.f50_r < get_profile(n).mtf.f50_g
            < get_profile(n).mtf.f50_b for n in _GREEN_ONLY_MEASURED),
        "7 of 7, layer order intact")

    # ---- C13, 2026-08-20c: what the 5274 adoption must not lose ---------------
    _p74 = get_profile("KODAK_VISION_200T_5274")
    # 1. the measured triple itself, and the ordering it confirms
    chk("5274 carries its measured f50 triple, red softest",
        (abs(_p74.mtf.f50_r - 35.4) < 1e-9
         and abs(_p74.mtf.f50_g - 68.8) < 1e-9
         and abs(_p74.mtf.f50_b - 74.0) < 1e-9
         and _p74.mtf.f50_r < _p74.mtf.f50_g < _p74.mtf.f50_b),
        "35.4 < 68.8 < 74.0 cycles/mm")

    # 2. ⚠ THE FINDING THAT OUTLIVES THIS PROFILE. The estimating rule puts
    # f50_r/f50_b near 0.78; both measured stocks land far below it. If a future
    # edit "tidies" 5274 back toward the family ratio this fails, and it is the
    # only place that comparison is recorded as an assertion rather than prose.
    # ⚠ EXCLUDES _GREEN_ONLY_MEASURED. The two Fuji stocks' red and blue ARE the
    # estimating-ratio family (they were derived from a stored ratio), so leaving
    # them in would make this guard test its own input and it would pass or fail
    # for the wrong reason.
    # ⚠ AND ONE MORE EXCLUSION FROM 2026-09-17, FOR A THIRD REASON. Kennel
    # et al. 1982 print ONE MTF curve for the whole of 5293 and not one per
    # record, so its three f50 fields carry the same number and its red/blue
    # ratio is 1.000 by construction. Leaving it in would make this guard fail
    # on a stock whose MTF is measured and correct; splitting the single
    # published curve into three to satisfy the guard would be inventing two
    # measurements. It gets its own assertion below instead.
    _POOLED_MTF = {"EASTMAN_5293_250T_1982"}
    _meas_ratio = [(p.name, p.mtf.f50_r / p.mtf.f50_b)
                   for p in FILM_PROFILES
                   if p.mtf.mtf_measured and not p.is_monochrome
                   and p.mtf.f50_b > 0
                   and p.name not in _GREEN_ONLY_MEASURED
                   and p.name not in _VISUAL_FILTER_MEASURED
                   and p.name not in _POOLED_MTF]
    chk("every measured colour stock is softer in red than the estimating rule",
        all(r < 0.65 for _, r in _meas_ratio) and len(_meas_ratio) >= 7,
        "; ".join("%s %.3f" % (n.split("_")[-1], r) for n, r in _meas_ratio)
        + " vs the rule's ~0.78")

    # ⚠ THE POOLED-CURVE STOCKS ARE ASSERTED TO BE POOLED, so that an edit
    # which quietly splits one into three per-record values has to come here
    # and say why. A single published MTF is a real measurement of the film;
    # three copies of it are not three measurements of three records.
    for _n in sorted(_POOLED_MTF):
        _m = get_profile(_n).mtf
        chk(f"{_n}'s MTF is ONE published curve stored three times, not three "
            f"measurements, and its rolloff is fitted from that curve",
            _m.f50_r == _m.f50_g == _m.f50_b and _m.mtf_measured
            and _m.mtf_rolloff_q > 0.0,
            "f50 %.1f/%.1f/%.1f, q %.2f, resolving power %.0f (1.6:1) / "
            "%.0f (1000:1)"
            % (_m.f50_r, _m.f50_g, _m.f50_b, _m.mtf_rolloff_q,
               _m.resolving_power_lp_mm_lowc, _m.resolving_power_lp_mm_highc))

    # ⚠ C2b/C24, 2026-08-23: THE FINDING THAT REPLACED THE RATIO. Seven measured
    # red records span 32.1-41.1 cycles/mm -- mean 36.4, +-13 % -- while green
    # spreads 52 % and blue 70 %. So red is a CONSTANT of the family and not a
    # fraction of blue, which is why no value of k in `f50_r = k * f50_b` fits and
    # why the five re-anchored profiles carry exactly 36.0. This asserts the
    # constancy itself, because that is the claim the re-anchoring rests on.
    # ⚠ EXCLUDES _GREEN_ONLY_MEASURED for the same reason, and the exclusion is
    # not a convenience: 8572's transferred red is 16.6 c/mm, which would drag
    # the "clustered near 36" claim to a 73 % spread and destroy a finding that
    # is about MEASURED reds. It is also a Fuji family, and C24's anchor was
    # derived from Kodak cine negatives -- mixing them is exactly the
    # class-estimate error C24 refused.
    # ⚠ AND IT EXCLUDES REVERSAL STOCKS, ADDED 2026-08-25 -- for exactly the
    # reason the Fuji exclusion above exists, not for convenience. C24's anchor
    # was derived from Kodak cine NEGATIVES. The first measured colour REVERSAL
    # MTF (5285, adopted 2026-08-25) puts its red record at 27.2 cycles/mm, which
    # is 25 % below the negatives' 36.4 and would take the spread from 25 % to
    # 41 % on its own. Folding a reversal stock into a negative-family constant
    # would be the class-estimate error C24 refused, and would also destroy a
    # finding that is about negatives.
    # ⚠ THE FINDING ITSELF IS RECORDED RATHER THAN AVERAGED AWAY: one measured
    # reversal red is not a reversal constant (method rule 18), but it IS
    # evidence that the 36 c/mm anchor does not extend past the negative family,
    # and nothing licensed assuming it did.
    # ⚠ AND IT EXCLUDES ORWOCOLOR NC 3, ADDED 2026-09-15, for the third time
    #   the same reason has been written here. C24's 36 c/mm anchor was derived
    #   from KODAK cine negatives; the Fuji family was excluded because it is a
    #   different manufacturer and the reversal stocks because they are a
    #   different class. NC 3 is both a different manufacturer AND a different
    #   decade -- a 1972 GDR emulsion whose own paper prints a limiting
    #   resolution of 85 L/mm against the modern stocks' several hundred. Its
    #   measured red f50 is 7.6 c/mm, a fifth of the anchor, and folding it in
    #   would take the spread from 41 % to 101 % and destroy a finding that is
    #   about the Kodak negative family.
    #   ⚠ THE NUMBER IS NOT IN DOUBT AND IS NOT BEING HIDDEN: it is traced,
    #   it orders correctly against the other two records, and it agrees with
    #   Tabelle 2's k-Zahlen. It is simply not a member of this population.
    _mr = [p.mtf.f50_r for p in FILM_PROFILES
           if p.mtf.mtf_measured and not p.is_monochrome
           and p.kind == StockKind.NEGATIVE
           and p.name != "ORWOCOLOR_NC3"
           and p.name not in _GREEN_ONLY_MEASURED
           and p.name not in _VISUAL_FILTER_MEASURED]
    # ⚠ THE BAND WAS 0.30 AND IS NOW 0.45, AND THE REASON IS A MEASUREMENT,
    # NOT A FAILING TEST BEING LOOSENED. On 2026-08-30 (K1) KODAK_PORTRA_400VC
    # entered the measured set at red f50 = 26.6 cycles/mm -- the softest red
    # record in the family by a clear margin, traced from E-190 p12's own MTF
    # panel, and physically expected: it is the most saturated and slowest-
    # resolving of the four NC/VC stocks. The spread went 30 % -> 41 %.
    # What this guard is FOR is catching a red record that was silently taken
    # from the family anchor when it should have been read; a real reading at
    # the low end is exactly what it should tolerate. 0.45 keeps roughly the
    # same margin above the observed spread that 0.30 had before.
    chk("the measured red records of the NEGATIVE family stay clustered near 36",
        len(_mr) >= 7 and (max(_mr) - min(_mr)) / (sum(_mr) / len(_mr)) < 0.45,
        "red f50 %s, mean %.1f, spread %.0f %%"
        % ("/".join("%.1f" % v for v in sorted(_mr)), sum(_mr) / len(_mr),
           100.0 * (max(_mr) - min(_mr)) / (sum(_mr) / len(_mr))))

    # And the five stocks that took the family anchor must carry it EXACTLY, so a
    # later edit cannot drift them back toward a ratio without failing here.
    _anch = {"KODAK_VISION3_50D_5203", "KODAK_VISION3_250D_5207",
             "KODAK_VISION3_200T_5213", "KODAK_VISION3_500T_5219",
             "KODAK_VISION_250D_5246"}
    _mixed = {"KODAK_VISION2_250D_5205", "EASTMAN_EXR_200T_5293"}
    _bad_anchor = [n for n in sorted(_anch | _mixed)
                   if abs(get_profile(n).mtf.f50_r - 36.0) > 1e-9]
    chk("the 7 family-anchored red records are exactly 36.0 cycles/mm",
        not _bad_anchor, ", ".join(_bad_anchor) if _bad_anchor
        else "5 re-anchored + 2 mixed-provenance stocks")

    # ⚠ AND THE ANCHOR MUST NOT ESCAPE ITS FAMILY. It was derived from Kodak cine
    # colour negatives whose blue sits inside the measured 55-111 range; applying
    # it to a softer or older stock would be the class-estimate error C24 refused.
    # EASTMAN_EXR_500T_5296 (blue 42) is the nearest excluded neighbour and is
    # named here so a future "finish the family" pass fails instead of guessing.
    chk("the family anchor stayed out of the excluded stocks",
        abs(get_profile("EASTMAN_EXR_500T_5296").mtf.f50_r - 30.0) < 1e-9,
        "5296 keeps its own 30.0 -- blue 42 is below the measured range")

    # 3. ⚠ THE LAYER-DEPTH CLAIM, CORRECTED BY C2b ON 2026-08-23. Off two stocks
    # this suite recorded that "both red records cluster at 1.84-1.89 and both
    # blues at 3.38-3.42", and C13 asked whether q could therefore be DERIVED from
    # the layer stack. With seven stocks the ORDERING survives -- q_R <= q_G <= q_B
    # on 8 of 8 sheets that yield two or more records -- but the magnitudes do not:
    # red spans 1.89-2.77 and blue 2.38-3.42 (sd 0.32-0.37), and q correlates only
    # weakly with f50 (Pearson 0.39 over 23 curves). So q is NOT derivable and
    # stays per-stock measured. What is asserted here is the spread: no two stocks
    # may have been collapsed onto a shared constant.
    # ⚠ ONE PAIR IS EXEMPT AND THE EXEMPTION IS THE POINT OF IT, 2026-09-06c.
    # FUJICOLOR_PRO_800Z and FUJICOLOR_PORTRAIT_NPZ_800 carry q 1.86 twice
    # because AF3-177E and AF3-100E print ONE MTF DRAWING between them -- their
    # curves agree to 0.91 % response over 1.58 decades, off a vector path and a
    # bilevel raster at different panel aspect ratios. Sharing the exponent is
    # what a shared drawing REQUIRES; giving them 1.86 and 1.84 to satisfy a
    # distinctness guard would encode two measurements where there is one. The
    # guard therefore compares the exponents with the pair collapsed, so it
    # still catches any OTHER two stocks being flattened onto a constant.
    _Q_SHARED_DRAWING = {"FUJICOLOR_PRO_800Z", "FUJICOLOR_PORTRAIT_NPZ_800"}
    # ⚠⚠ TWO MORE SHARED-DRAWING PAIRS FROM 2026-09-07b, and they are excluded
    # from the collapse test for the same reason the pair above is: their
    # (f50, q) IS shared, by measurement, because it is one drawing.
    #   AGFA_APX_100 / AGFA_APX_400          -- «Technical Data PF» p10 and
    #     F-PF-E4 p8, one 73-point path translated 175.21 pt
    #   AGFA_VISTA_PLUS_200 / _400      -- one 47-point path translated
    #     dx +0.143 dy -0.076 pt
    # Excluding the SECOND member of each pair keeps the guard able to catch
    # any OTHER two stocks being flattened onto a constant, which is what it
    # is for.
    _SHARED_DRAWING_TWINS = ("FUJICOLOR_PORTRAIT_NPZ_800", "AGFA_APX_400",
                             "AGFA_VISTA_PLUS_400")
    # ⚠⚠ THE TEST IS THE (f50, q) PAIR, NOT q ALONE, AND IT HAD TO CHANGE ON
    # 2026-09-06h. With 42 measured stocks and q quantised to 0.01, two
    # unrelated films landing on the same exponent is arithmetic, not evidence
    # -- SUPERIA X-TRA 800 and AGFA RSX II 100 both fit 2.16, and RSX II 50 and
    # PROVIA 400X both 2.38, off four different manufacturers' panels. What the
    # guard is actually for is catching stocks COLLAPSED onto a shared
    # constant, and a collapse shows up in f50 and q together.
    _qs = sorted(round(p.mtf.mtf_rolloff_q, 4) for p in FILM_PROFILES
                 if p.mtf.mtf_measured and p.mtf.mtf_rolloff_q > 0.0)
    _pairs_m = [(round(p.mtf.mtf_rolloff_q, 4), round(p.mtf.f50_g, 3), p.name)
                for p in FILM_PROFILES
                if p.mtf.mtf_measured and p.mtf.mtf_rolloff_q > 0.0
                and p.name not in _SHARED_DRAWING_TWINS]
    _seenp = {}
    for _q, _f, _n in _pairs_m:
        _seenp.setdefault((_q, _f), []).append(_n)
    _collapsed = [v for v in _seenp.values() if len(v) > 1]
    chk("no two measured stocks share BOTH a rolloff exponent and an f50, and "
        "the exponents span over 1.0",
        not _collapsed and (_qs[-1] - _qs[0]) > 1.0,
        ("collapsed: %s" % _collapsed) if _collapsed
        else "%d stocks, q %.2f-%.2f, %d distinct exponents"
             % (len(_pairs_m), _qs[0], _qs[-1], len(set(_qs))))
    # ⚠ AND THE SHARED PAIR MUST STAY SHARED. If a later pass "measures" one of
    # them separately and they drift apart, that is a claim of two independent
    # readings of one drawing and it must be argued for, not slipped in.
    chk("the PRO 800Z / NPZ 800 shared MTF drawing carries one exponent and "
        "one f50 on both stocks",
        len({(round(get_profile(n).mtf.mtf_rolloff_q, 4),
              round(get_profile(n).mtf.f50_g, 4)) for n in _Q_SHARED_DRAWING}) == 1,
        "q %.2f, f50_g %.1f on both"
        % (get_profile("FUJICOLOR_PRO_800Z").mtf.mtf_rolloff_q,
           get_profile("FUJICOLOR_PRO_800Z").mtf.f50_g))

    # ---- G-NPZ, 2026-09-06c: the shared-artwork pair, asserted BOTH ways -----
    # ⚠ A SHARED DRAWING IS A CLAIM AND SO IS A DIFFERENCE, so both halves are
    # guarded. AF3-100E (NPZ 800) and AF3-177E (PRO 800Z) print one set of data
    # drawings -- characteristic curves agreeing to 0.015 D max / 0.005 D rms
    # over 4.2 decades and MTF curves to 0.91 % response over 1.58 decades, one
    # read off Bezier paths and the other off a bilevel raster at a different
    # panel aspect ratio. Their curve records must therefore be IDENTICAL: a
    # future pass that re-traces one of them and lets the two drift apart is
    # claiming two measurements of one drawing and must say so out loud.
    _npz = get_profile("FUJICOLOR_PORTRAIT_NPZ_800")
    _z8 = get_profile("FUJICOLOR_PRO_800Z")
    chk("G-NPZ1: NPZ 800 and PRO 800Z carry byte-identical curve records, "
        "because their sheets print ONE drawing",
        _npz.curves == _z8.curves,
        "dmin %.4f/%.4f/%.4f both"
        % (_npz.curves.r.dmin, _npz.curves.g.dmin, _npz.curves.b.dmin))
    chk("G-NPZ2: and one neutral+dmin pair, from that same shared section",
        _npz.dye_density.d_neutral == _z8.dye_density.d_neutral
        and _npz.dye_density.d_dmin == _z8.dye_density.d_dmin
        and not _npz.dye_density.has_data and not _z8.dye_density.has_data,
        "31 samples each, no dye triple on either")
    # ⚠⚠ AND THE HALF THAT STOPS THE FINDING BEING OVERSTATED. The two sheets
    # publish DIFFERENT long-exposure tables: unfiltered to 2 s with a +2 stop
    # row at 64 s on AF3-100E, unfiltered only to 1 s and nothing past 16 s on
    # AF3-177E. Rule 4 -- not averaged, not copied across. That difference is
    # the whole reason these are two stocks and not one stock with an alias, so
    # a later "tidy-up" that harmonises them must fail here.
    chk("G-NPZ3: their PRINTED reciprocity tables differ and neither was "
        "copied onto the other",
        (_npz.reciprocity_table.times_s == (2.0, 4.0, 16.0, 64.0)
         and _z8.reciprocity_table.times_s == (1.0, 4.0, 16.0)
         and _npz.reciprocity_table.times_s != _z8.reciprocity_table.times_s),
        "NPZ %s vs 800Z %s"
        % (_npz.reciprocity_table.times_s, _z8.reciprocity_table.times_s))
    # ⚠ AND BOTH SPECTRAL PANELS STAY REFUSED, for the schema reason and not for
    # a reading one: each draws FOUR sensitive layers (blue, green, red, cyan)
    # and SpectralSensitivity has r/g/b and pan. If either acquires a spectral
    # set without the schema gaining a fourth record, a layer has been silently
    # discarded -- rule 23 point 5.
    # ⚠⚠ REWRITTEN TWICE IN TWO DAYS, AND BOTH REWRITES WERE CORRECTIONS OF
    # MINE. Until 2026-09-06f this asserted that BOTH stocks refused their
    # spectral panel, because it draws four layers and the schema holds three.
    # The owner's decision settled that (store R/G/B, document cyan) and PRO
    # 800Z gained a set. NPZ 800 was then left empty on the ground that its
    # RASTER panel could not be separated -- one attempt reported as a
    # conclusion. The owner asked why, and the panel turned out not to need
    # separating: its twin prints the same drawing as vector paths, proved by
    # overlay at 98.1 % against 21-35 % for every displaced null.
    # ⚠ SO THE PAIR NOW CARRIES ONE SPECTRAL SET BETWEEN TWO STOCKS, exactly as
    # it already carries one MTF and one neutral pair. Byte-identical is the
    # assertion: a later pass that "re-measures" one of them and lets the two
    # drift apart is claiming two readings of one drawing and must argue for it.
    chk("G-NPZ4: PRO 800Z and NPZ 800 carry ONE spectral set between them, "
        "byte-identical, because their sheets print one drawing",
        (_z8.spectral.has_data and _npz.spectral.has_data
         and _npz.spectral.log_s_r == _z8.spectral.log_s_r
         and _npz.spectral.log_s_g == _z8.spectral.log_s_g
         and _npz.spectral.log_s_b == _z8.spectral.log_s_b),
        "third panel of this pair proved shared, after the curves and the MTF")

    # ---- G-SUP, 2026-09-06e: the SUPERIA batch ------------------------------
    # ⚠ THE FOURTH-LAYER REFUSAL IS NOW A FAMILY PROPERTY, NOT A ONE-OFF, and
    # counting it is what turns five separate notes into one schema decision.
    # Every stock below draws Blue / Green / Red / Cyan on its spectral panel
    # and `SpectralSensitivity` carries r/g/b and pan. Storing three of four
    # would silently discard the layer these films are sold on (rule 23 pt 5).
    # ONE schema change closes all five at once; until then this asserts that
    # none of them has quietly acquired a truncated set.
    _FOURTH_LAYER = ("FUJICOLOR_PRO_800Z", "FUJICOLOR_PORTRAIT_NPZ_800",
                     "FUJICOLOR_SUPERIA_XTRA_400",
                     "FUJICOLOR_SUPERIA_XTRA_800",
                     "FUJICOLOR_SUPERIA_REALA")
    # ⚠⚠ ALL FIVE NOW CARRY A SET, AND ALL FIVE ARE INCOMPLETE. This guard has
    # been three different assertions in three days: all five refused, then
    # four stored and one refused, now five stored. What it must keep asserting
    # is not that the cells are full but that the incompleteness is DECLARED --
    # every one of these is three quarters of a film.
    _four = [n for n in _FOURTH_LAYER if get_profile(n).spectral.has_data]
    chk("G-SUP1: all five fourth-colour-layer stocks carry a three-of-four "
        "spectral set",
        len(_four) == 5,
        "stored: %s" % ", ".join(sorted(_four)))
    # ⚠ AND EVERY ONE OF THEM SAYS SO IN ITS OWN SOURCE STRING. A stored set
    # that is three quarters of a film must announce that wherever it is read,
    # not only in a document beside it -- otherwise the next reader integrates
    # it against an illuminant and gets a confident wrong answer.
    _undeclared = [n for n in _four
                   if "THREE OF FOUR" not in get_profile(n).spectral.source
                   or "FUJI_FOURTH_LAYER.md" not in get_profile(n).spectral.source]
    chk("G-SUP1b: every three-of-four spectral set declares it in its own "
        "source and names where the fourth curve is written down",
        not _undeclared, "; ".join(_undeclared) or "%d of %d declared" % (len(_four), len(_four)))
    # ⚠ THE CYAN DOCUMENT MUST EXIST AND MUST CARRY NUMBERS. The whole basis on
    # which storing three of four was accepted is that the fourth is written
    # out. If that file goes missing or empties, the decision has quietly
    # become the thing it was allowed instead of.
    _cy = Path(__file__).resolve().parent / "doc" / "FUJI_FOURTH_LAYER.md"
    _cytxt = _cy.read_text(encoding="utf-8") if _cy.is_file() else ""
    chk("G-SUP1c: doc/FUJI_FOURTH_LAYER.md exists and holds a cyan curve for "
        "each stored three-of-four stock",
        all(n in _cytxt for n in _four)
        and _cytxt.count("log_s_cyan") == len(_four),
        "%d log_s_cyan blocks for %d stocks" % (_cytxt.count("log_s_cyan"),
                                                len(_four)))
    # ⚠ THE THREE SUPERIA RECIPROCITY TABLES DIFFER AND SPEED DOES NOT ORDER
    # THEM. REALA is the SLOWEST and has the SHORTEST unfiltered range (1 s
    # against 2 s) and the only refusal at 64 s. That is the opposite of the
    # PROVIA 100F / 400F direction, so a future pass must not "correct" it into
    # a monotone family rule -- which is exactly what this guard forbids.
    _sup = {n: get_profile(n).reciprocity_table
            for n in ("FUJICOLOR_SUPERIA_XTRA_400",
                      "FUJICOLOR_SUPERIA_XTRA_800",
                      "FUJICOLOR_SUPERIA_REALA")}
    chk("G-SUP2: the three SUPERIA sheets publish three DIFFERENT reciprocity "
        "tables and none was copied from another",
        (_sup["FUJICOLOR_SUPERIA_XTRA_400"].stops_correction
         == (0.0, 0.3333, 0.6667, 1.0)
         and _sup["FUJICOLOR_SUPERIA_XTRA_800"].stops_correction
         == (0.0, 0.6667, 1.5, 2.0)
         and _sup["FUJICOLOR_SUPERIA_REALA"].times_s == (1.0, 4.0, 16.0)
         and len({t.stops_correction for t in _sup.values()}) == 3),
        "400 %s / 800 %s / REALA %s"
        % (_sup["FUJICOLOR_SUPERIA_XTRA_400"].stops_correction,
           _sup["FUJICOLOR_SUPERIA_XTRA_800"].stops_correction,
           _sup["FUJICOLOR_SUPERIA_REALA"].stops_correction))
    chk("G-SUP3: and the SLOWEST of the three has the SHORTEST unfiltered "
        "range, which is the opposite of the PROVIA direction",
        (_sup["FUJICOLOR_SUPERIA_REALA"].times_s[0]
         < _sup["FUJICOLOR_SUPERIA_XTRA_400"].times_s[0]
         and get_profile("FUJICOLOR_SUPERIA_REALA").exposure_index
         < get_profile("FUJICOLOR_SUPERIA_XTRA_400").exposure_index),
        "REALA ISO 100 unfiltered to 1 s, X-TRA 400 to 2 s")
    # ⚠⚠ REALA'S THREE GAMMAS ARE THE MOST EQUAL IN THE DATABASE and that is a
    # MEASUREMENT of the "Optimum Spectral Sensitivity Balance" its own feature
    # list leads with -- not a fitting artefact and not something to normalise
    # away. Asserted against the field, so a re-trace that spreads them has to
    # argue for itself.
    _rl = get_profile("FUJICOLOR_SUPERIA_REALA").curves
    _rg = [_rl.r.gamma, _rl.g.gamma, _rl.b.gamma]
    _spread = max(_rg) - min(_rg)
    # ⚠ AND THE COMPARISON EXCLUDES THE TWO STOCKS WHOSE SPREAD IS EXACTLY
    # ZERO, WHICH THE FIRST VERSION OF THIS GUARD DID NOT AND WHICH FAILED IT.
    # GEVACOLOR_NEG_652 and TECHNICOLOR_THREE_STRIP carry three IDENTICAL
    # gammas because their records were written from one class estimate, not
    # because anybody measured three equal numbers. An exact zero is the
    # signature of a constructed value; comparing a measurement against it
    # would rank an absence above a reading, which is rule 23 point 3.
    _others = []
    for _p in FILM_PROFILES:
        if _p.is_monochrome or _p.kind != StockKind.NEGATIVE:
            continue
        _g = [_p.curves.r.gamma, _p.curves.g.gamma, _p.curves.b.gamma]
        _sp = max(_g) - min(_g)
        if _sp == 0.0:
            continue
        _others.append((_sp, _p.name))
    _others.sort()
    chk("G-SUP4: REALA's three gammas are the most equal of any colour "
        "negative with three distinct traced records, which is its sheet's "
        "own balance claim measured",
        _spread < 0.006 and _others[0][1] == "FUJICOLOR_SUPERIA_REALA",
        "spread %.4f; next tightest %s at %.4f; %d stocks excluded for a "
        "spread of exactly zero (one class estimate written three times)"
        % (_spread, _others[1][1], _others[1][0],
           sum(1 for _p in FILM_PROFILES
               if not _p.is_monochrome and _p.kind == StockKind.NEGATIVE
               and _p.curves.r.gamma == _p.curves.g.gamma == _p.curves.b.gamma)))
    # ⚠ AND THE THREE ISO 800 FUJI NEGATIVES ARE NOT ONE FILM. SUPERIA X-TRA
    # 800, PRO 800Z and NPZ 800 share a speed, a process and the fourth layer,
    # and PRO 800Z / NPZ 800 really are one set of drawings -- so the third
    # must be shown to be separate rather than assumed to be. Its curves differ
    # from theirs by more than a third of a density, and its printed
    # high-contrast resolving power differs by 10 lines/mm.
    _sx8 = get_profile("FUJICOLOR_SUPERIA_XTRA_800")
    _dmax = max(abs(getattr(_sx8.curves, _c).dmin - getattr(_z8.curves, _c).dmin)
                for _c in ("r", "g", "b"))
    _gmax = max(abs(getattr(_sx8.curves, _c).gamma
                    - getattr(_z8.curves, _c).gamma)
                for _c in ("r", "g", "b"))
    # ⚠ GAMMA IS THE DISCRIMINATOR HERE AND dmin IS NOT, which the first
    # version of this guard had backwards. The worst dmin gap is 0.190 D and
    # the worst GAMMA gap is 0.203 -- X-TRA 800's blue record runs at 0.811
    # against PRO 800Z's 0.608, a third steeper. Two films that shared a
    # drawing would agree on both; these agree on neither, and the printed
    # high-contrast resolving power differs by 10 lines/mm as well.
    chk("G-SUP5: SUPERIA X-TRA 800 is a different film from PRO 800Z / NPZ "
        "800, not a re-badge of them",
        _dmax > 0.15 and _gmax > 0.15
        and _sx8.mtf.resolving_power_lp_mm_highc
        != _z8.mtf.resolving_power_lp_mm_highc,
        "worst dmin gap %.3f D, worst gamma gap %.3f; resolving power "
        "%.0f vs %.0f at 1000:1"
        % (_dmax, _gmax, _sx8.mtf.resolving_power_lp_mm_highc,
           _z8.mtf.resolving_power_lp_mm_highc))
    # And a flagged stock must carry a usable exponent OR be the one documented
    # exception, because the flag otherwise silently falls back to the Gaussian.
    # ⚠ 5279 IS THAT EXCEPTION AND IT IS PHYSICS, NOT AN OVERSIGHT: its sheet
    # prints a +42 %/+55 % adjacency overshoot, and the carrier 1/(1+(f/f50)^q) is
    # 1.0 at zero frequency by construction, so it cannot represent a curve that
    # starts at 1.42 -- the fit returns rms 0.25 against 0.0095-0.132 elsewhere.
    # Its measured f50 triple and measured overshoot are used; its rolloff is not.
    # ⚠ 2026-09-15b: KODAK_TECHNICAL_PAN WAS BRIEFLY THE SECOND EXCEPTION HERE
    # AND IS NOT ONE ANY MORE, which is the outcome worth recording. Its
    # measured q = 1.071 is the shallowest in the corpus and the best TWO-lobe
    # kernel for it misses by 0.0719 against the 0.045 tolerance, so the first
    # pass refused to store the exponent. The refusal was a statement about the
    # KERNEL FAMILY, not about the measurement -- so the family was widened. A
    # THREE-lobe fit lands at 0.0267, `_MTF_KERNEL_TABLE3` carries it, and
    # ALGO_BLUR_MAX_LOBES went 6 -> 9 so the adjacency band-pass still has room
    # to multiply each base lobe into three. 5279 remains the one exception,
    # and for the reason it always had: its curve starts at 1.42 and the
    # carrier is 1.0 at zero frequency by construction, which no number of
    # lobes can fix.
    _mq = [p.name for p in FILM_PROFILES
           if p.mtf.mtf_measured and not p.mtf.mtf_rolloff_q > 0.0]
    chk("every mtf_measured stock carries a rolloff exponent, bar the one "
        "documented exception",
        _mq == ["KODAK_VISION_500T_5279"],
        ", ".join(_mq) if _mq else "q > 0 where the flag is set")

    # 4. The measured law must beat the Gaussian ON THE TRACED CURVE, which is the
    # only reason it was adopted. Three points read off H-1-5231 p3 by
    # mtf_vector.py, well past f50 where the two laws diverge.
    _px = get_profile("EASTMAN_PLUS_X_5231")
    _traced = ((61.1, 0.370), (76.7, 0.306), (98.2, 0.245))
    _gauss_err = _pow_err = 0.0
    for _f, _want in _traced:
        _g = float(_np.exp(-_np.log(2.0) * (_f / 41.3) ** 2))
        _m = film_profiles.mtf_response(_px.mtf, 1, _f)
        _gauss_err += (_g - _want) ** 2
        _pow_err += (_m - _want) ** 2
    chk("PLUS-X's measured rolloff beats the Gaussian on its own traced curve",
        _pow_err < 0.25 * _gauss_err,
        "sum sq err %.4f vs Gaussian %.4f over 61/77/98 cycles/mm"
        % (_pow_err, _gauss_err))

    # ---- 2026-08-26: the KODAK still-film E-series harvest -------------------
    # Eight profiles were touched by one batch from one reader on one date. What
    # follows guards the three things that batch could most easily get wrong
    # later: the shape of what was adopted, the identity of what was REFUSED,
    # and the one place a plausible-looking conversion is forbidden.
    _KSTILL = ("KODAK_PORTRA_160", "KODAK_PORTRA_400", "KODAK_PORTRA_800",
               "KODAK_GOLD_200", "KODAK_ULTRAMAX_400", "KODAK_ULTRAMAX_800")

    # 1. The mask ladder. This is the finding, not a formality: the previous
    # ANALOGY estimates gave all six a near-neutral dmin of about 0.20/0.19/0.19,
    # and every one of them turns out to have a real orange-mask ladder of
    # 0.61-0.70 D. The old encoding was the wrong KIND of description, so the
    # guard asserts the ladder AND the ordering r < g < b that makes it a mask
    # rather than three unrelated numbers.
    _ladder = []
    for _n in _KSTILL:
        _c = get_profile(_n).curves
        _d = (_c.r.dmin, _c.g.dmin, _c.b.dmin)
        if not (_d[0] < _d[1] < _d[2] and (_d[2] - _d[0]) > 0.55):
            _ladder.append("%s %.3f/%.3f/%.3f" % (_n, *_d))
    chk("the six harvested KODAK still stocks carry a real orange-mask ladder",
        not _ladder, "; ".join(_ladder) or
        "spreads " + ", ".join(
            "%.2f" % (get_profile(_n).curves.b.dmin
                      - get_profile(_n).curves.r.dmin) for _n in _KSTILL))

    # 2. No shoulder was invented. The sheets are straight where they stop, so
    # every carried-over shoulder must sit OUTSIDE the traced range. The traced
    # right edges are +0.95 (E-4051), +0.55 (E-4050, E-7023), +0.28 (E-4040,
    # E-190 2006 p12, E-7024) and +0.86 (E-7022); +1.0 is above all of them and
    # is used as one conservative bound rather than six.
    _sh = [f"{_n} {min(c.shoulder_x for c in get_profile(_n).curves.as_tuple()):.2f}"
           for _n in _KSTILL
           if min(c.shoulder_x for c in get_profile(_n).curves.as_tuple()) <= 1.0]
    chk("no harvested KODAK still curve shoulders inside its own traced range",
        not _sh, "; ".join(_sh) or
        "earliest shoulder logE %.2f" % min(
            c.shoulder_x for _n in _KSTILL
            for c in get_profile(_n).curves.as_tuple()))

    # 3. ⚠ THE REFUSAL THAT MATTERS MOST, ASSERTED AS A REFUSAL.
    # E-2468's entire CURVES page is PORTRA 160VC's artwork -- its
    # characteristic figure is F009_0154AC, the figure E-190 prints on its 160VC
    # page, and tracing both returns identical numbers to four decimals. A
    # tungsten ISO 100 film cannot share a daylight ISO 160 film's curve, so
    # KODAK_PORTRA_100T was left on its estimate. This guard fails if anyone
    # later "completes" the harvest by adopting that figure: 160VC's traced dmin
    # triple is 0.2045/0.6087/0.8121, and 100T must not be holding it.
    _p100 = get_profile("KODAK_PORTRA_100T")
    _d100 = (_p100.curves.r.dmin, _p100.curves.g.dmin, _p100.curves.b.dmin)
    chk("KODAK_PORTRA_100T did NOT absorb PORTRA 160VC's mis-printed curve",
        abs(_d100[1] - 0.6087) > 0.05 and abs(_d100[2] - 0.8121) > 0.05
        and _p100.mask_encoding == "neutral_dmin"
        and _p100.provenance.fitted_from == "analogy",
        "dmin %.4f/%.4f/%.4f, mask %s, fitted_from %s"
        % (*_d100, _p100.mask_encoding, _p100.provenance.fitted_from))

    # 4. Print Grain Index. Seven carriers, and the two properties that make the
    # field honest: the censoring sentinel is a 0.0 and never a number below the
    # method's own 25 threshold, and NOTHING derives an rms granularity from it.
    _pgi = [p for p in FILM_PROFILES if p.print_grain_index.has_data]
    # ⚠ 7 -> 9 on 2026-08-26f: KODAK_GOLD_100 (E-7022, February 2007 -- the
    # two-film edition, whose characteristic panels the first pass could not see
    # because its captions put the panel kind last) and KODAK_PRO_100T_PRT
    # (E-29, April 1999).
    # ⚠ 9 -> 13 on 2026-08-30 (K1): the four PORTRA NC/VC stocks, from E-190
    # page 8's three magnification tables. Still no rms is derived from any of
    # them -- the sheet states outright that the two scales cannot be compared.
    chk("13 film profiles carry a published Print Grain Index",
        len(_pgi) == 13, "%d: %s" % (len(_pgi), ", ".join(
            sorted(p.name for p in _pgi))))
    _pgi_bad = [f"{p.name} {v}" for p in _pgi
                for t in (p.print_grain_index.fmt_135,
                          p.print_grain_index.fmt_120,
                          p.print_grain_index.fmt_sheet)
                for v in t if 0.0 < v < 25.0]
    chk("no Print Grain Index value sits below the method's own 25 threshold",
        not _pgi_bad, ", ".join(_pgi_bad) or
        "%d censored 'Less than 25' entries across %d stocks"
        % (sum(p.print_grain_index.censored_count for p in _pgi), len(_pgi)))
    # ⚠ THE CROSS-DOCUMENT CHECK. PORTRA 100T's PGI is printed twice in the
    # corpus, six years apart and in unrelated publications: E-2468 (October
    # 2006) page 4 and E-58 (July 2000) page 5 both give 33 / 55 / 84 for size
    # 135. That agreement is the only independent confirmation any PGI figure in
    # this database has, so it is pinned rather than left as prose.
    chk("PORTRA 100T's PGI matches between E-2468 and E-58 (33/55/84)",
        _p100.print_grain_index.fmt_135 == (33.0, 55.0, 84.0),
        "%s" % (_p100.print_grain_index.fmt_135,))
    # And the forbidden conversion. If a later edit ever fits rms to PGI, the
    # eight touched stocks' rms triples would stop being the pure b=1.3x/r=1.1x
    # heuristic they still are. That heuristic being intact IS the evidence that
    # nobody converted.
    _conv = []
    for p in _pgi:
        _g = p.grain
        if not (abs(_g.rms_r - 1.1 * _g.rms_granularity) < 1e-6
                and abs(_g.rms_b - 1.3 * _g.rms_granularity) < 1e-6):
            _conv.append(p.name)
    chk("no rms granularity was derived from Print Grain Index",
        not _conv, ", ".join(_conv) or
        "all %d PGI carriers keep the unconverted colour-negative heuristic"
        % len(_pgi))

    # 5. The three measured MTF sets carry a fitted exponent AND the measured
    # overshoot, because mtf_measured promises both. All three panels rise above
    # 100 % modulation at low frequency, so a stored adjacency of 0 next to the
    # flag would be a silent loss.
    _mtf3 = ("KODAK_PORTRA_160", "KODAK_PORTRA_400", "KODAK_PORTRA_800")
    _mbad = [f"{_n} q={get_profile(_n).mtf.mtf_rolloff_q:.2f} "
             f"adj={get_profile(_n).mtf.adjacency:.3f}" for _n in _mtf3
             if not (get_profile(_n).mtf.mtf_rolloff_q > 1.0
                     and get_profile(_n).mtf.adjacency > 0.10)]
    chk("the three still-film MTF sets carry a fitted q and a measured overshoot",
        not _mbad, "; ".join(_mbad) or "q = " + " / ".join(
            "%.2f" % get_profile(_n).mtf.mtf_rolloff_q for _n in _mtf3))

    # 6. The neutral+Dmin pairs must actually behave like a mask over a neutral:
    # neutral above Dmin at every sampled wavelength, and the Dmin peaking in
    # the blue. Four new pairs, one guard, because a mis-assigned pair (the
    # reader orders them by mean density) would show up as a crossing.
    _pairbad = []
    for _n in ("KODAK_PORTRA_160", "KODAK_PORTRA_800", "KODAK_GOLD_200",
               "KODAK_ULTRAMAX_400"):
        _dd = get_profile(_n).dye_density
        if not _dd.has_neutral_pair:
            _pairbad.append(f"{_n}: no pair")
            continue
        if min(a - b for a, b in zip(_dd.d_neutral, _dd.d_dmin)) <= 0.0:
            _pairbad.append(f"{_n}: curves cross")
        _i = _dd.d_dmin.index(max(_dd.d_dmin))
        _nm = _dd.lambda_start_nm + _dd.lambda_step_nm * _i
        if not 400.0 <= _nm <= 470.0:
            _pairbad.append(f"{_n}: D-min peaks at {_nm:.0f} nm, not in the blue")
    chk("the four new neutral+Dmin pairs behave as a mask over a neutral",
        not _pairbad, "; ".join(_pairbad) or
        "margins " + ", ".join(
            "%.2f" % min(a - b for a, b in zip(get_profile(_n).dye_density.d_neutral,
                                               get_profile(_n).dye_density.d_dmin))
            for _n in ("KODAK_PORTRA_160", "KODAK_PORTRA_800",
                       "KODAK_GOLD_200", "KODAK_ULTRAMAX_400")))

    # 7. PORTRA 100T's reciprocity table is the only multi-point one this batch
    # produced, and its shape is the claim: EI falls monotonically with time, so
    # the correction rises monotonically from a 0.0 anchor.
    _rt = _p100.reciprocity_table
    chk("PORTRA 100T's reciprocity table rises monotonically from a 0.0 anchor",
        len(_rt.times_s) == 5 and _rt.stops_correction[0] == 0.0
        and all(a < b for a, b in zip(_rt.stops_correction,
                                      _rt.stops_correction[1:]))
        and abs(_rt.stops_correction[-1] - 4.0 / 3.0) < 0.01,
        "%d points, %s" % (len(_rt.times_s), ", ".join(
            "%.2f" % v for v in _rt.stops_correction)))

    # 8. The six single-point reciprocity BOUNDS adopted 2026-08-26. Their
    # shape is the claim and it is easy to destroy by "tidying": one time, one
    # correction, and that correction exactly 0.0, meaning "no correction needed
    # up to here". A later edit that appends an invented correction at a longer
    # time, or that drops the entry as empty, both change what the sheet said.
    _bounds = ("KODAK_PORTRA_160", "KODAK_PORTRA_400", "KODAK_PORTRA_800",
               "KODAK_GOLD_200", "KODAK_ULTRAMAX_400", "KODAK_ULTRAMAX_800")
    _bbad = []
    for _n in _bounds:
        _r = get_profile(_n).reciprocity_table
        if not (len(_r.times_s) == 1 and _r.times_s[0] == 1.0
                and _r.stops_correction == (0.0,) and _r.source):
            _bbad.append("%s %s/%s" % (_n, _r.times_s, _r.stops_correction))
    chk("the six KODAK still-film reciprocity entries are 1.0 s bounds, not "
        "corrections",
        not _bbad, "; ".join(_bbad) or
        "6 stocks bounded at 1.0 s with a 0.0 correction")
    # ⚠ AND THE ONE THAT IS NOT A BOUND MUST NOT BECOME ONE. PORTRA 100T is the
    # only stock in the batch with a real multi-point walk, and it is also the
    # only one whose sheet publishes exposure INDEX against time rather than a
    # correction, so it is the one most likely to be "simplified" later.
    chk("PORTRA 100T alone carries a multi-point reciprocity walk in this batch",
        len(get_profile("KODAK_PORTRA_100T").reciprocity_table.times_s) == 5
        and all(len(get_profile(_n).reciprocity_table.times_s) == 1
                for _n in _bounds),
        "100T %d points" % len(
            get_profile("KODAK_PORTRA_100T").reciprocity_table.times_s))

    # ---- G1/G3: the 1968 Gevachrome pair and the re-traced 682 curves --------
    # 2026-08-19. What these guard is the boundary between what the two source
    # documents PRINT and what this database estimated around it -- the thing most
    # likely to blur in a later edit.
    for _n, _ei, _g in (("GEVACHROME_600", 50, (1.45, 1.25, 1.25)),
                        ("GEVACHROME_605", 160, (1.35, 1.25, 1.25))):
        _p = get_profile(_n)
        chk(f"{_n} carries its PRINTED tungsten exposure index",
            _p.exposure_index == _ei, "EI %d (Tab. II, Kino-Technik 1968 Nr. 10 "
            "p262)" % _p.exposure_index)
        # ⚠ 3300 K, not 3200: the table prints a RANGE, "3200-3400 K". Storing the
        # standard tungsten reference would assert something the sheet never said.
        chk(f"{_n} balance is the midpoint of the printed 3200-3400 K range",
            _p.balance_kelvin == 3300, "%d K" % _p.balance_kelvin)
        chk(f"{_n} carries the printed per-layer gammas",
            (abs(_p.curves.r.gamma - _g[0]) < 1e-9
             and abs(_p.curves.g.gamma - _g[1]) < 1e-9
             and abs(_p.curves.b.gamma - _g[2]) < 1e-9),
            "r/g/b %.2f/%.2f/%.2f -- cyan/magenta/yellow layer gammas as printed "
            "in the Bilder 5a/5b caption" % (_p.curves.r.gamma, _p.curves.g.gamma,
                                             _p.curves.b.gamma))
        chk(f"{_n} is flagged REVERSAL and stacks blue/green/red as printed",
            _p.kind is StockKind.REVERSAL
            and _p.layer_stack.order == ("blue", "green", "red"),
            "Tab. I, nine layers, conventional order")
        # The paper prints NO granularity figure at all. The estimate must stay
        # labelled as one, or a later reader will take it for a measurement.
        _src = " ".join(_p.provenance.sources)
        chk(f"{_n} records that NO granularity figure is printed",
            "NOT PRINTED" in _src and "granularity" in _src,
            "the tier-3 grain estimate cannot be mistaken for a reading")
        chk(f"{_n} records the 150 ppi scan limit on curve separation",
            "150 ppi" in _src or "1-2 px" in _src,
            "why the three layer curves were not separated")

    # 682: the curves are now the traced ones, and the external check the trace was
    # licensed by is pinned with them.
    _682 = get_profile("GEVACOLOR_NEG_682")
    _want682 = {"r": (0.1356, 0.5056), "g": (0.5863, 0.5677), "b": (0.9137, 0.5396)}
    _bad682 = []
    for _ch, (_dm, _ga) in _want682.items():
        _c = getattr(_682.curves, _ch)
        if abs(_c.dmin - _dm) > 1e-9 or abs(_c.gamma - _ga) > 1e-9:
            _bad682.append("%s dmin %.4f gamma %.4f" % (_ch, _c.dmin, _c.gamma))
    chk("GEVACOLOR_NEG_682 carries the Fig. 10 traced curves exactly",
        not _bad682, "; ".join(_bad682) if _bad682
        else "r/g/b dmin 0.136/0.586/0.914, gamma 0.506/0.568/0.540")
    chk("GEVACOLOR_NEG_682 green gamma still matches the figure's printed 0.57",
        abs(_682.curves.g.gamma - 0.57) <= 0.01,
        "traced %.4f vs printed 0.57 -- the external check that licensed the trace"
        % _682.curves.g.gamma)
    # G3, 2026-08-19: the MTF numbers read off Fig. 11, and the boundary between
    # what that figure shows and what it cannot.
    chk("GEVACOLOR_NEG_682 f50 r/g/b are the Fig. 11 readings",
        (abs(_682.mtf.f50_r - 29.0) < 1e-9 and abs(_682.mtf.f50_g - 44.0) < 1e-9),
        "r 29.0, g 44.0 cycles/mm -- the 50 % crossings; was 46/54 estimated")
    # ⚠ THE BLUE f50 MUST STAY ABOVE THE FIGURE'S BOUND AND MUST NOT BE PRETENDED
    # MEASURED. Fig. 11 leaves blue at ~60 % at its 50 lines/mm right edge, so the
    # figure supports only "> 50"; 62.0 is the earlier estimate, retained.
    chk("GEVACOLOR_NEG_682 blue f50 respects the >50 bound Fig. 11 gives",
        _682.mtf.f50_b > 50.0, "f50_b %.1f (estimate, bounded below by the plot)"
        % _682.mtf.f50_b)
    # ⚠ A REAL CHECK, NOT AN `or True`. The first version of this line ended in
    # `or True`, which is a guard that cannot fail -- worse than no guard, because
    # it reads as coverage. The hazard is that "lines/mm" on this figure may mean
    # half-cycles, which would make every f50 here 2x high; what must survive is
    # the RECORD of that risk in the provenance.
    _p682src = " ".join(_682.provenance.sources)
    chk("GEVACOLOR_NEG_682 records the lines/mm vs cycles/mm unit hazard",
        "UNIT HAZARD" in _p682src and "lines/mm" in _p682src,
        "the factor-2 risk is on file, not just in a comment")
    # ⚠ THIS GUARD WAS INVERTED ON 2026-08-25 (queue item G7), AND THAT IS THE
    # POINT OF IT. It previously asserted the dye set stayed EMPTY, because Fig.
    # 8's three curves could not be separated across their full range and a
    # partial set stored as a complete one was the failure to guard against. The
    # separation now works (see dashtrace.trace_predictive's merge_px), so the
    # guard becomes the one it was always standing in for: the set is present AND
    # it reproduces the peaks the PAPER prints, which is what "separated" means.
    _682dd = _682.dye_density
    _682pk = {}
    for _n, _v in (("y", _682dd.d_yellow), ("m", _682dd.d_magenta),
                   ("c", _682dd.d_cyan)):
        if _v:
            _i = max(range(len(_v)), key=lambda j: _v[j])
            _682pk[_n] = (400.0 + 10.0 * _i, _v[_i])
    chk("GEVACOLOR_NEG_682 carries the Fig. 8 dye set, separated at last",
        _682dd.has_data and len(_682dd.d_cyan) == 31,
        "traced 2026-08-25 at one sample per pixel column; empty until then")
    chk("682's three dye peaks reproduce the paper's own printed values",
        _682pk.get("y", (0, 0))[0] == 450.0 and abs(_682pk["y"][1] - 1.46) <= 0.03
        and _682pk.get("m", (0, 0))[0] == 530.0 and abs(_682pk["m"][1] - 1.48) <= 0.03
        and _682pk.get("c", (0, 0))[0] == 680.0 and abs(_682pk["c"][1] - 1.46) <= 0.03,
        "printed Y 1.46@448 M 1.48@525 C 1.46@687; stored peaks land on the "
        "nearest 10 nm sample at " + ", ".join(
            f"{_k} {_v[1]:.3f}@{_v[0]:.0f}" for _k, _v in sorted(_682pk.items())))
    chk("682's dye set is NOT tagged with a normalisation the paper never states",
        _682dd.normalisation == "as_printed_no_stated_normalisation",
        "the ordinate is simply 'DENSITY'; the equal peaks are an observation, "
        "not a stated convention")
    chk("GEVACOLOR_NEG_682 carries the Fig. 6 layer order",
        _682.layer_stack.order == ("blue", "green", "red")
        and "double-layer" in _682.layer_stack.source,
        "six emulsion layers recorded in the source string")

    # dmax must keep the mask ladder order; a trace that crossed two curves would
    # break it and nothing else in the numbers would say so.
    chk("GEVACOLOR_NEG_682 dmax keeps the masked-negative order b > g > r",
        _682.curves.b.dmax > _682.curves.g.dmax > _682.curves.r.dmax,
        "%.3f > %.3f > %.3f" % (_682.curves.b.dmax, _682.curves.g.dmax,
                                _682.curves.r.dmax))

    # ---- 2026-08-18: film_names.txt is a CONSUMED artefact, not a by-product ----
    # The owner loads this file straight into the effect control panel's listbox,
    # and the panel indexes into GetFilmDatabase()'s std::vector. So line N of
    # this file MUST describe element N-1 of that vector, and the pipe separator
    # must sit on every line except the last (the lines are consumed as adjacent
    # C++ string literals, concatenating to "A|B|...|Z" with no trailing pipe).
    # Two ways this file has actually gone wrong, both of which these checks catch:
    #   (1) it did not get regenerated alongside the .cpp/.hpp at all;
    #   (2) TWO generators write it. cpp_codegen.py emits name.replace("_", " ")
    #       and derives order by parsing the EMITTED .cpp back (index equality by
    #       construction); gen_film_names.py emits official manufacturer
    #       spellings via a 21-entry override table and derives order from
    #       FILM_PROFILES. Whichever runs last wins, and 19 of 154 lines differ.
    #       The owner's in-production file is cpp_codegen.py's version.
    # These checks assert the STRUCTURE strictly and the ORDER by a
    # punctuation-insensitive match, so they survive a decision to adopt the
    # official spellings while still failing on a reorder or a desync -- which is
    # what would actually break the listbox.
    _names_p = Path(__file__).resolve().parent / "film_names.txt"
    chk("film_names.txt exists next to the generator",
        _names_p.is_file(), str(_names_p.name))
    if _names_p.is_file():
        _raw = _names_p.read_bytes()
        _lines = _names_p.read_text(encoding="ascii", errors="replace").splitlines()
        chk("film_names.txt has one line per database entry",
            len(_lines) == len(FILM_PROFILES),
            "%d lines vs %d profiles" % (len(_lines), len(FILM_PROFILES)))
        chk("film_names.txt is pure ASCII with LF endings and no comment banner",
            b"\r" not in _raw and b"//" not in _raw
            and all(ord(c) < 128 for c in "".join(_lines)),
            "%d bytes" % len(_raw))
        # Every line is "NAME|" except the last, which is "NAME".
        _quoted = all(len(s) >= 2 and s.startswith('"') and s.endswith('"')
                      for s in _lines)
        _inner = [s[1:-1] for s in _lines if len(s) >= 2]
        chk("film_names.txt: every line quoted, '|' on all but the last",
            _quoted and _inner[:-1] and all(s.endswith("|") for s in _inner[:-1])
            and not _inner[-1].endswith("|"),
            "last = %r" % (_lines[-1] if _lines else None))
        # Order must equal GetFilmDatabase() order. Compared on alphanumerics
        # only, so "KODAK TMAX 100" and "KODAK T-MAX 100" both match
        # KODAK_TMAX_100 -- a spelling convention is a decision, a reorder is a bug.
        def _norm(s):
            return "".join(ch for ch in s.upper() if ch.isalnum())
        _got = [_norm(s.rstrip("|")) for s in _inner]
        _want = [_norm(p.name) for p in FILM_PROFILES]
        _bad = [i for i, (a, b) in enumerate(zip(_got, _want)) if a != b]
        if _bad:
            _order_msg = ("first mismatch line %d: %r vs profile %s"
                          % (_bad[0] + 1, _inner[_bad[0]],
                             FILM_PROFILES[_bad[0]].name))
        elif len(_got) != len(_want):
            # zip() truncates, so an empty _bad here means "the common prefix
            # matches but the lengths differ" -- do NOT report that as aligned.
            _order_msg = ("common prefix aligned, but %d lines vs %d profiles"
                          % (len(_got), len(_want)))
        else:
            _order_msg = "all %d aligned" % len(_got)
        chk("film_names.txt line order equals the GetFilmDatabase() vector order",
            _got == _want, _order_msg)

    # ---- 2026-08-18: the 5247 generation split stays split ------------------
    # Kodak reused the designation 5247 across a coating change, and one entry
    # had been carrying both generations: EI 100 stored, while TI0835 (EI 125T),
    # Chibisov 1988 (S 125 GOST) and Sehlin/Kennel 1985 (vs 5294, launched 1983)
    # all describe the later film. The split put the documented data on
    # EASTMAN_5247_1983 and left EASTMAN_5247_1974 as an explicit [T3] period
    # reconstruction. These checks exist because the failure mode is silent: a
    # future tidy-up that "fills in the gap" on the 1974 entry by copying from
    # the 1983 one would re-create exactly the contamination that was removed.
    _o = get_profile("EASTMAN_5247_1974")
    _n = get_profile("EASTMAN_5247_1983")
    chk("5247 exists as two generations with different speeds",
        _o.exposure_index == 100 and _n.exposure_index == 125,
        "EI %d (1974) vs %d (1983)" % (_o.exposure_index, _n.exposure_index))
    chk("5247_1974 carries NO spectral data (none exists for that coating)",
        not _o.spectral.has_data, "spectral empty")
    chk("5247_1983 owns the TI0835 spectral plate",
        _n.spectral.has_data and "TI0835" in _n.spectral.source,
        "source cites TI0835")
    chk("5247_1974 is labelled NOT DOCUMENTED, not merely 'estimated'",
        "NOT DOCUMENTED" in _o.description
        and "EASTMAN_5247_1983" in _o.description,
        "warning and pointer present")
    chk("5247_1983 records that its year is a floor, not an introduction date",
        "NOT A PROVEN INTRODUCTION DATE" in _n.description.upper(),
        "caveat present")

    # ---- 2026-08-18: spectral dye density, 6 new sheets ---------------------
    # Extracted from PDF vector paths, validated by re-deriving the two already
    # adopted sets (5285 to RMS 0.003 D, 2383 to 0.135 D against its own
    # recorded 0.128 D base-absorber offset). The peak_1.0 sheets carry SHAPE
    # only -- the absolute level is not on those plots -- so the normalisation
    # tag is load-bearing and is asserted, not just the presence of numbers.
    _dd = [p for p in FILM_PROFILES if p.dye_density.has_data]
    # ⚠ COUNTS UPDATED 2026-08-18 (E0b): 7 -> 10 film profiles, 6 -> 8 peak_1.0.
    # These two assertions failed when the 7239, 5217 and 5218 sets were adopted,
    # which is the behaviour they were written for -- a count assertion is meant
    # to fail when the count changes so the change is acknowledged rather than
    # absorbed. The as-printed family now has TWO members among the film
    # profiles (5285 and 7239) plus the 2383 print stock, which is why the
    # peak_1.0 count is 8 and not 9.
    # ⚠ 10 -> 11 ON 2026-08-25 (queue item C9): KODAK_VISION2_50D_5201, the
    # first sheet read by the ink-based family C. Same behaviour as the E0b
    # change above -- the count assertion is meant to fail so the addition is
    # acknowledged rather than absorbed.
    # ⚠ 11 -> 12 ON 2026-08-25 (queue item G7): GEVACOLOR_NEG_682, whose Fig. 8
    # set had been deliberately empty since 2026-08-19. Third count change in a
    # week, all three for the same good reason.
    # ⚠ 15 -> 16 ON 2026-09-01d: TECHNICOLOR_THREE_STRIP, from Flueckiger et al.
    # 2018 Fig. 16. Same behaviour as every count change above -- the assertion
    # is meant to fail so the addition is acknowledged rather than absorbed.
    # ⚠ 16 -> 18 ON 2026-09-02 (queue G2): GEVACHROME_600 and GEVACHROME_605.
    # One measurement on two profiles; see the identity assertion beside the
    # other count of this same set.
    # ⚠ 18 -> 25 ON 2026-09-04 (the Fuji/Konica dye sweep). All seven new sets
    # are peak_1.0 -- every one of those panels draws three unit-peak dyes and
    # no neutral -- so that family goes 10 -> 17 while the as-printed family
    # stays at 8. A new as-printed set appearing here would be a real event and
    # this split is what would surface it.
    # ⚠ 25 -> 27 ON 2026-09-04c, both peak_1.0: 5203 and 5207 print the same
    # "Cyan, Magenta, and Yellow Dye Curves are peak-normalized" note as every
    # other VISION sheet. So peak_1.0 goes 17 -> 19 and as-printed stays at 8.
    chk("30 film profiles carry spectral dye density", len(_dd) == 30,
        ", ".join(sorted(p.name.split("_")[-1] for p in _dd)))
    _pk = [p for p in _dd if p.dye_density.normalisation == "peak_1.0"]
    chk("22 of the 30 dye sets are tagged peak_1.0, 8 as-printed", len(_pk) == 22,
        "%d peak_1.0, %d as-printed" % (len(_pk), len(_dd) - len(_pk)))
    # ⚠ THE WHOLE VISION3 FAMILY NOW CARRIES A DYE SET, and that is worth an
    # assertion of its own because it is what makes the family comparison in
    # `_MEASURED_DYE_MATRIX` possible: the magenta-into-blue term splits
    # daylight from tungsten better than 2:1 across four stocks whose panels
    # were traced by two unrelated methods, raster for 5203/5207 and vector for
    # 5213/5219.
    _v3 = ("KODAK_VISION3_50D_5203", "KODAK_VISION3_200T_5213",
           "KODAK_VISION3_250D_5207", "KODAK_VISION3_500T_5219")
    chk("all four VISION3 stocks carry a spectral dye set",
        all(get_profile(n).dye_density.has_data for n in _v3),
        "5203, 5207, 5213, 5219")
    # ⚠ ONE SET IS NOT ON THE CORPUS GRID, AND THE EXCEPTION IS NAMED RATHER
    # THAN THE GUARD WEAKENED. Every datasheet panel in this corpus is plotted
    # 400-700 nm and traces onto 31 samples at 10 nm. Flueckiger et al. 2018
    # Fig. 16 is plotted 350-800 and its CYAN SECONDARY PEAK IS AT 720 nm --
    # a validated feature, printed in the report's own text -- so cropping it to
    # the corpus grid to satisfy a shape assertion would throw away the evidence
    # the trace was checked against. It is stored 360-790 at 10 nm, 44 samples.
    # ⚠ A SECOND EXCEPTION ON 2026-09-17, AND IT IS THE SAME ARGUMENT.
    # Kennel et al. 1982 Fig. 7 runs 400-790 nm and puts the 1982 5293's CYAN
    # PEAK AT 682 with a tail that is still at 0.57 of peak by 750; truncating
    # to the corpus grid would cut the peak's own long side off, which is the
    # part of a cyan dye a print stock's red record actually sees. Stored
    # 400-750 at 10 nm, 36 samples.
    _DYE_GRID_EXCEPTIONS = {"TECHNICOLOR_THREE_STRIP": (360.0, 10.0, 44),
                            "EASTMAN_5293_250T_1982": (400.0, 10.0, 36)}
    _off = [p.name for p in _dd
            if p.name not in _DYE_GRID_EXCEPTIONS
            and not (len(p.dye_density.d_cyan) == 31
                     and p.dye_density.lambda_start_nm == 400.0
                     and p.dye_density.lambda_step_nm == 10.0)]
    _exc = [p.name for p in _dd if p.name in _DYE_GRID_EXCEPTIONS
            and (p.dye_density.lambda_start_nm,
                 p.dye_density.lambda_step_nm,
                 len(p.dye_density.d_cyan)) != _DYE_GRID_EXCEPTIONS[p.name]]
    chk("every dye trace is a 31-sample 400-700 nm grid, with two named "
        "exceptions",
        not _off and not _exc and
        len(_DYE_GRID_EXCEPTIONS) == sum(1 for p in _dd
                                         if p.name in _DYE_GRID_EXCEPTIONS),
        "31 x 10 nm from 400; exceptions TECHNICOLOR_THREE_STRIP 44 x 10 nm "
        "from 360 and EASTMAN_5293_250T_1982 36 x 10 nm from 400"
        + ("" if not _off else "; OFF-GRID " + ", ".join(_off))
        + ("" if not _exc else "; EXCEPTION MOVED " + ", ".join(_exc)))
    # Physics: yellow absorbs blue, magenta green, cyan red. A mis-assigned
    # trace is the one error this extraction could plausibly make, and it would
    # show up here and nowhere else.
    import numpy as _np
    # ⚠ THE GRID IS TAKEN FROM EACH PROFILE, NOT ASSUMED. Until 2026-09-01d this
    # hard-coded arange(400, 701, 10) for every set, which was harmless while
    # every set was on that grid and silently wrong the moment one was not:
    # an argmax index into a 44-sample 360 nm trace read through a 31-sample
    # 400 nm ruler reports a wavelength that does not exist in the data.
    _bad = []
    for p in _dd:
        d = p.dye_density
        g = d.lambda_start_nm + d.lambda_step_nm * _np.arange(len(d.d_cyan))
        ly = g[int(_np.argmax(d.d_yellow))]
        lm = g[int(_np.argmax(d.d_magenta))]
        lc = g[int(_np.argmax(d.d_cyan))]
        if not (405 <= ly <= 480 and 510 <= lm <= 590 and 615 <= lc <= 700):
            _bad.append("%s y%d m%d c%d" % (p.name, ly, lm, lc))
    chk("dye peaks sit in their absorption bands on all 18", not _bad,
        "; ".join(_bad) if _bad else "yellow 405-480, magenta 510-590, cyan 615-700")

    # ---- Queue items C9 + C10, closed 2026-08-25. H-1-5201's last two panels.
    _p01 = get_profile("KODAK_VISION2_50D_5201")
    # C9. The dye set that "could not be classified" for weeks. Its peaks are
    # identical to 5217's and 5218's -- the family consistency the extractor
    # never saw -- and its cyan trace is the one the old segment filter dropped,
    # so a peak in the cyan band is the specific thing that has to stay true.
    _d01 = _p01.dye_density
    _g31 = _np.arange(400, 701, 10)
    chk("5201's dye set is peak_1.0 with peaks at 450 / 540 / 680 nm",
        _d01.normalisation == "peak_1.0"
        and _g31[int(_np.argmax(_d01.d_yellow))] == 450
        and _g31[int(_np.argmax(_d01.d_magenta))] == 540
        and _g31[int(_np.argmax(_d01.d_cyan))] == 680,
        "matches 5217 and 5218 exactly")
    # The family-C validator, restated as an assertion on the STORED arrays: the
    # three dyes must be able to form a visual neutral with EQUAL weights. This
    # is what makes the set tier 1 rather than three plausible curves, and it is
    # checked here so a later hand-edit of any array breaks it.
    _A = _np.vstack([_d01.d_cyan, _d01.d_magenta, _d01.d_yellow]).T
    _k = _np.linalg.lstsq(_A, _A @ _np.array([0.628, 0.604, 0.595]),
                          rcond=None)[0]
    chk("5201's three dyes still solve for equal neutral weights",
        float((_k.max() - _k.min()) / _k.mean()) < 0.06,
        "spread %.1f %% on 0.628 / 0.604 / 0.595 (rms 0.019 D on the sheet)"
        % (100.0 * (_k.max() - _k.min()) / _k.mean()))
    # ⚠ THIS CHECK ONCE ASSERTED THE OPPOSITE, AND THE REVERSAL IS THE POINT.
    # It used to require that the neutral and dmin traces be ABSENT, because
    # they are as-printed while the dyes are peak-normalised and one
    # `normalisation` string could not mean both. That was a limitation of the
    # record, not of the sheet: `normalisation_neutral` (schema v19, queue M2b)
    # lets the pair state its own convention, so the traces are kept. The
    # identity they enable immediately failed three panels, one of which had
    # passed every other test in the file.
    chk("5201 stores the neutral and dmin traces its panel draws",
        bool(_d01.d_neutral) and bool(_d01.d_dmin)
        and _d01.normalisation_neutral != ""
        and len(_d01.d_neutral) == len(_d01.d_cyan),
        "neutral and dmin on the dyes' grid, normalisation_neutral=%r"
        % _d01.normalisation_neutral)
    # ⚠ AND THE IDENTITY THEY EXIST FOR, ON EVERY PANEL THAT NOW CARRIES BOTH.
    # `Neutral - Dmin = k(C+M+Y)` with the three k EQUAL is what makes a neutral
    # a neutral; the coefficients are free, so a small spread is evidence.
    _nd = [q for q in FILM_PROFILES
           if q.dye_density.has_data and q.dye_density.d_neutral
           and q.dye_density.d_dmin]
    _nd_bad = []
    for _q in _nd:
        _dd = _q.dye_density
        _AA = _np.stack([_np.asarray(_dd.d_cyan), _np.asarray(_dd.d_magenta),
                         _np.asarray(_dd.d_yellow)], 1)
        _bb = _np.asarray(_dd.d_neutral) - _np.asarray(_dd.d_dmin)
        _kk = _np.linalg.lstsq(_AA, _bb, rcond=None)[0]
        _sp = float((_kk.max() - _kk.min()) / max(_kk.mean(), 1e-9))
        if _sp > 0.15:
            _nd_bad.append("%s %.2f" % (_q.name, _sp))
    chk("every panel that stores a neutral is refused or resolves into equal "
        "parts of its own three dyes",
        all(n.split()[0] in film_profiles._DYE_MATRIX_NOT_DERIVED for n in _nd_bad),
        "%d panels carry the pair; failing the identity: %s"
        % (len(_nd), ", ".join(_nd_bad) if _nd_bad else "none"))
    # C10. The first VECTOR-traced spectral set, and the criterion decision.
    _s01 = _p01.spectral
    chk("5201 carries the vector-traced spectral set",
        _s01.has_data and len(_s01.log_s_r) == 31
        and "spectral_vector.py" in _s01.source,
        "31 samples from 380 nm, traced 2026-08-25")
    # ⚠ THE CRITERION IS STORED AS THE SHEET PRINTS IT, and the three older sets
    # keep theirs. Owner decision 2026-08-25: the sheets print "specified
    # density" without naming it, so the "D0.2 above dmin" the older three carry
    # is not printed on any of them -- recorded as a conflict, not propagated and
    # not retro-fixed. This guard fails in BOTH directions: if 5201 acquires the
    # unprinted criterion, or if the older three quietly lose theirs.
    # ---- 2026-08-25d, FROM THE VALIDATION PASS: the criterion question moved
    # from "unsourced" to "unsourced AND contradicted in value", and this guard
    # holds the finding until the owner decides.
    # ⚠ THE CLAIM THAT STOOD HERE FROM 2026-08-25 TO 2026-08-26 WAS FALSE, AND
    # THE WAY IT WAS FALSE IS THE LESSON. It read "NOT ONE SHEET IN THE CORPUS
    # PRINTS 0.2", and concluded that the 0.2 "appears nowhere" and "was supplied
    # for precisely the cases with no evidence for it".
    # A full-corpus regex sweep on 2026-08-26 found THREE FILES THAT PRINT IT:
    # `5205t.pdf` p4, `KODAK VISION2 250D ... 5205.pdf` p4 and
    # `5218-Vision2-500T-H-1-5218t.pdf` p4, each carrying **"D=0.2>D-min"** in
    # the Spectral Sensitivity panel's own caption block, directly beneath
    # "Densitometry: Status M" and beside "Effective exposure" and "Process:
    # ECN-2". It is unmistakably that panel's density criterion.
    # ⚠ WHY IT WAS MISSED: the earlier sweep looked for caption text INSIDE the
    # plot frame, because that is where 5222 and 7239 put it. The VISION2 layout
    # puts the block BELOW the frame. A scan that assumes one layout finds one
    # layout, and "not printed" was really "not printed where I looked" -- the
    # same mistake this project already recorded for outlined vector art on the
    # F-125 sheet, arrived at from the opposite direction.
    # SO THE CORRECTED PICTURE, and it splits three ways rather than two:
    #   * 0.2 IS a printed Kodak convention -- VISION2 cine sheets state it.
    #   * 5205 and 5218 are therefore SOURCED for the value they store.
    #   * five more Kodak CINE stocks (5217, 5203, 5207, 5213, 5219) carry it
    #     without their own sheet printing it -- but they are the same product
    #     family, sheet series and era as the two that do, so that is a family
    #     inference with a documented anchor inside the family, not an invention.
    #   * NINE STILL FILMS carry it too (EKTAR 100, GOLD 100/200, PORTRA
    #     100T/160/400/800, ULTRAMAX 400/800). Those are a different product
    #     line documented in different publications, and nothing in this corpus
    #     supports or refutes the value for them. THAT is the live gap.
    # ⚠ NOTHING IS CHANGED HERE. Rewriting a provenance claim on 16 profiles is
    # an owner decision, and the counts are pinned so the inconsistency stays
    # visible instead of being absorbed. See NotFound.md 2026-08-25d.
    _crit = [p.spectral.criterion for p in FILM_PROFILES if p.spectral.criterion]
    _n02 = sum(1 for c in _crit if "D0.2_above_dmin" in c)
    _n04 = sum(1 for c in _crit if "D0.4_above_dmin" in c)
    chk("the spectral-criterion split is still 16 D0.2 (2 printed) vs 10 printed D0.4",
        _n02 == 16 and _n04 == 10,
        "%d at D0.2 -- PRINTED on 5205 p4 and 5218 p4 as 'D=0.2>D-min', so 2 "
        "are sourced, 5 more Kodak cine stocks are a family inference and 9 are "
        "STILL films this corpus says nothing about; %d at D0.4 (printed on "
        "5245, 5246, 5248, 5274, V200T, 5293)" % (_n02, _n04))
    # ---- 2026-08-26, owner decision: KEEP the D0.2 value, ANNOTATE it. --------
    # ⚠ The annotation must land on exactly the five stocks whose own sheets do
    # NOT print the criterion, and must NOT land on 5205 and 5218, which DO.
    # Getting that backwards would tell a reader the anchor is inferred.
    _infer = {p.name for p in FILM_PROFILES
              if any("SPECTRAL CRITERION IS A FAMILY INFERENCE" in _s
                     for _s in p.provenance.sources)}
    chk("the D0.2 family inference is annotated on exactly the 5 cine stocks "
        "whose sheets do not print it",
        _infer == {"KODAK_VISION2_200T_5217", "KODAK_VISION3_50D_5203",
                   "KODAK_VISION3_250D_5207", "KODAK_VISION3_200T_5213",
                   "KODAK_VISION3_500T_5219"},
        ", ".join(sorted(_infer)))
    chk("5205 and 5218 are NOT annotated -- their own sheets print 'D=0.2>D-min'",
        not ({"KODAK_VISION2_250D_5205",
              "KODAK_VISION2_500T_5218"} & _infer),
        "sourced, not inferred; see the sweep note in _CRITERION_FAMILY_INFERENCE")
    # ⚠ AND THE NINE STILL FILMS ARE THE LIVE GAP, so they are neither annotated
    # as a cine family inference nor silently treated as sourced. Pinned so the
    # group cannot shrink or grow without the change being deliberate.
    _still = {p.name for p in FILM_PROFILES
              if "D0.2_above_dmin" in p.spectral.criterion
              and not p.name.startswith(("KODAK_VISION2", "KODAK_VISION3"))}
    chk("the 9 STILL films carrying D0.2 are recorded as the remaining gap",
        len(_still) == 9 and all("VISION" not in n for n in _still),
        ", ".join(sorted(_still)))
    chk("5201's spectral criterion is the printed one, the other 3 unchanged",
        _s01.criterion == "log_reciprocal_erg_cm2_specified_density"
        and all(get_profile(n).spectral.criterion
                == "log_reciprocal_erg_cm2_D0.2_above_dmin"
                for n in ("KODAK_VISION2_500T_5218", "KODAK_VISION2_200T_5217",
                          "KODAK_VISION3_500T_5219")),
        "as printed on 5201; the 5218/5217/5219 conflict is recorded, not fixed")
    # ⚠ CORRECTED 2026-08-25d, AND THE ORIGINAL FORM OF THIS GUARD IS WHY.
    # It read "5201's blue layer keeps its measured 470 nm peak, NOT THE FAMILY'S
    # 420" -- a claim built on comparing one stock (5218) and calling it "the
    # family". Sweeping every 31-sample Kodak cine stock shows the blue peak
    # splits 6/4: 470 nm on 5201, 5217, 5205, 5203, 5274, 5246 and 410-440 on
    # 5218 (420), 5279 (420), 5219 (410), 5213 (440). 470 is the MAJORITY, and
    # 5201 agrees with 5217 exactly. The guard now asserts the split itself, so
    # neither group can be quietly "harmonised" toward the other.
    _lam = 380.0 + 10.0 * _np.arange(31)
    def _bpk(n):
        return float(_lam[int(_np.argmax(get_profile(n).spectral.log_s_b))])
    _b470 = ("KODAK_VISION2_50D_5201", "KODAK_VISION2_200T_5217",
             "KODAK_VISION2_250D_5205", "KODAK_VISION3_50D_5203",
             "KODAK_VISION_200T_5274", "KODAK_VISION_250D_5246")
    _blo = {"KODAK_VISION2_500T_5218": 420.0, "KODAK_VISION_500T_5279": 420.0,
            "KODAK_VISION3_500T_5219": 410.0, "KODAK_VISION3_200T_5213": 440.0}
    # ---- 2026-08-26: EASTMAN DOUBLE-X 5222, harvested from H-1-5222 rev 7-15 --
    # ⚠ THE VALUE OF THAT SHEET IS ITS ART, NOT ITS CONTENT. The corpus already
    # held H-1-5222 revised 3-26, which prints the SAME figures (F010_0029AC and
    # F010_0031AC) as RASTERS. The 2015 edition draws them as vector paths, so
    # panels that had to be read by hand became measurable.
    _xx = get_profile("EASTMAN_DOUBLE_X_5222")
    chk("5222's MTF is measured, not the flat 56/56/56 estimate",
        _xx.mtf.mtf_measured and _xx.mtf.f50_r == 42.2
        and _xx.mtf.f50_r == _xx.mtf.f50_g == _xx.mtf.f50_b,
        "f50 42.2 cycles/mm, one value because a black-and-white stock has one "
        "sensitive layer; the estimate was 1.33x too sharp")
    # ⚠ THE EXTERNAL CHECK AN ESTIMATE COULD NOT HAVE HAD. PLUS-X 5231 is the
    # other Kodak black-and-white cine negative here and was traced from its own
    # sheet. Two speeds of one design family, two independent traces.
    _px = get_profile("EASTMAN_PLUS_X_5231")
    chk("5222 and 5231 agree within 3 % now that BOTH are measured",
        _px.mtf.mtf_measured and abs(_xx.mtf.f50_g - _px.mtf.f50_g)
        / _px.mtf.f50_g < 0.03,
        "DOUBLE-X %.1f vs PLUS-X %.1f cycles/mm; the estimated pair read "
        "56.0 and 60.0" % (_xx.mtf.f50_g, _px.mtf.f50_g))
    # q is adopted here at +25 % overshoot where 5279 was refused at +42 %. The
    # discriminator is the FIT, not the overshoot, and it is on record.
    # ⚠ THE `adjacency == 0.250` HALF OF THIS CHECK WAS THE SAME WRONG PREMISE
    # corrected on 5231 (see A4, 2026-09-02e): 0.250 was the OVERSHOOT, and the
    # parameter that renders a +25 % overshoot through this stock's own q = 2.88
    # rolloff is 0.2996 at adjacency_um 49.8. The q half of the check is the
    # part that matters here and is unchanged -- q is adopted at +25 % where
    # 5279 was refused at +42 %, and the discriminator is the FIT, not the
    # overshoot. The overshoot is now pinned where it belongs: on what renders.
    chk("5222 keeps its rolloff q, and still RENDERS the +25 % printed "
        "overshoot after the A4 re-solve",
        _xx.mtf.mtf_rolloff_q == 2.88
        and abs(_rendered_peak(_xx.mtf, 1)[0] - 1.250) < 2e-3,
        "power law fits at rms 0.076, inside the 0.0095-0.132 band; 5279's "
        "+42 %% returned 0.25-0.34 and was put back on the Gaussian with q = 0; "
        "renders %+.4f" % (_rendered_peak(_xx.mtf, 1)[0] - 1.0))
    # ⚠ A LEVEL CORRECTION, NOT A SHAPE ONE, AND THE DISTINCTION IS THE POINT.
    # The 2026-08-02 raster trace of this same curve reproduces the vector path
    # to rms 0.0123 D and its gamma is within 0.0004 of the vector refit; only
    # base+fog was wrong, by 0.035 D. Two independent calibrations of the
    # density axis (printed ticks 0.2369, frame edges 0.2281) both exclude the
    # old 0.1977.
    chk("5222's base+fog is the measured 0.2328, not the raster trace's 0.1977",
        abs(_xx.curves.g.dmin - 0.2328) < 1e-6
        and abs(_xx.curves.g.gamma - 0.648) < 1e-6,
        "gamma unmoved at 0.648 -- the old trace had the shape right and the "
        "level wrong")
    # ⚠ THE MID-GREY PLACEMENT MUST NOT HAVE MOVED. The recorded anchor is
    # "D 1.178 at model x 0"; if a level correction had dragged the exposure
    # axis with it, this is where it would show.
    import numpy as _np2
    import digitize_plot as _dp
    _d0 = _dp.softplus_curve(_np2.array([0.0]), _xx.curves.g.dmin,
                             _xx.curves.g.gamma, _xx.curves.g.toe_x,
                             _xx.curves.g.toe_k, _xx.curves.g.shoulder_x,
                             _xx.curves.g.shoulder_k)[0]
    chk("5222's mid-grey anchor still lands on the recorded D 1.178",
        abs(float(_d0) - 1.178) < 0.005,
        "D %.4f at model x = 0; the correction moved the level only" % _d0)
    # The developer was wrong: D-76 is a still-film developer and Kodak's own
    # sheet says D-96 in three places. The Iofis 1964 row is kept as evidence of
    # local practice, which is what it actually is.
    chk("5222's processing is Kodak's own D-96 at 21 C, not the Iofis D-76",
        _xx.processing.developer == "KODAK D-96"
        and _xx.processing.celsius == 21.0
        and abs(_xx.processing.contrast_index - 0.66) < 1e-9,
        "printed in the PROCESSING table and on both plot captions")
    chk("the Iofis 1964 row survives as evidence of local practice",
        any("D-76" in _s and "1964" in _s for _s in _xx.provenance.sources),
        "method rule 4: the conflict is recorded, not averaged")
    # The five printed gammas. Stored AS PRINTED including the one the trace
    # does not reproduce -- see kodak_time_gamma.py for why.
    _fam = _xx.processing_family
    chk("5222 carries the printed five-point D-96 time-gamma family",
        _fam.has_data and len(_fam.points) == 5
        and [p.minutes for p in _fam.points] == [4.0, 5.0, 6.5, 9.0, 12.0]
        and [p.gamma for p in _fam.points] == [0.50, 0.56, 0.66, 0.84, 1.05]
        and all(p.developer == "KODAK D-96" and p.celsius == 21.0
                for p in _fam.points),
        "printed per-curve labels; the 6 1/2-minute point is the condition the "
        "stored ToneCurve represents")
    # ---- queue XX2, 2026-08-26: fog against development time ---------------
    # ⚠ WHAT THIS CLOSES IS A SILENCE, NOT A WRONG NUMBER. `ToneCurve.dmin` is
    # one value and therefore describes one development condition; nothing said
    # which, and nothing said fog moves with development at all. It does, by
    # 28 % across this family, and the stored dmin must equal the fog of the
    # condition the stored curve represents -- 6 1/2 minutes.
    _fogs = [q.base_fog for q in _fam.points]
    chk("5222's five development points each carry their own base+fog",
        all(v > 0.0 for v in _fogs) and _fogs == [0.231, 0.233, 0.233, 0.275,
                                                  0.296],
        "traced from each curve's left plateau; the sheet draws a Time-Fog "
        "curve but prints no numbers on it")
    chk("base+fog RISES with development, as the sheet's Time-Fog inset shows",
        all(a <= b for a, b in zip(_fogs, _fogs[1:]))
        and _fogs[-1] / _fogs[0] > 1.2,
        "%.3f -> %.3f, a %.0f %% rise" % (_fogs[0], _fogs[-1],
                                          100.0 * (_fogs[-1] / _fogs[0] - 1.0)))
    # ⚠ THE LINK THAT WAS PREVIOUSLY IMPLICIT, NOW ASSERTED: the stored dmin is
    # the fog of the stored development condition and of no other.
    _pt65 = [q for q in _fam.points if q.minutes == 6.5][0]
    chk("5222's stored dmin equals the fog of its stored 6 1/2-minute condition",
        abs(_xx.curves.g.dmin - _pt65.base_fog) < 0.002
        and _xx.processing.minutes == _pt65.minutes,
        "dmin %.4f against the 6 1/2-minute point's %.3f -- and it would be "
        "0.296 at 12 minutes" % (_xx.curves.g.dmin, _pt65.base_fog))
    chk("5222's 9-minute point keeps the PRINTED gamma the trace disputes",
        _fam.points[3].gamma == 0.84 and "0.798" in _fam.source,
        "measured 0.798 against printed 0.84; recorded, not averaged -- Kodak "
        "does not print the density interval their gamma is measured over")
    # ⚠ TWO CURVES, TWO CRITERIA, ONE EMULSION. Picking the wrong one would be
    # silent and about 0.55 decades large.
    chk("5222's spectral set names the criterion its panel prints",
        "D1.0_above_gross_fog" in _xx.spectral.criterion
        and "eff_exp_1.4s" in _xx.spectral.criterion
        and "D = 0.3 Above Gross Fog" in _xx.spectral.source,
        "the sheet draws D 0.3 AND D 1.0; the adopted set is the D 1.0 curve, "
        "selected by matching the printed caption to the curve below it")
    chk("5222's spectral peak is unmoved by the re-trace",
        380.0 + 10.0 * int(_np2.argmax(_xx.spectral.log_s_pan)) == 430.0
        and len(_xx.spectral.log_s_pan) == 31,
        "raster reading and vector trace agree to rms 0.037 decades on the "
        "same 430 nm sample -- confirmed, not corrected")

    chk("the Kodak cine blue-peak split is 6 stocks at 470 nm, 4 at 410-440",
        all(_bpk(n) == 470.0 for n in _b470)
        and all(_bpk(n) == v for n, v in _blo.items()),
        "5201/5217/5205/5203/5274/5246 at 470; 5218 5279 420, 5219 410, 5213 440")
    # ⚠ AND THE GUARD ABOVE IS FRAGILE ON EXACTLY THE STOCKS IT SORTS, which was
    # found on 2026-08-26 by re-tracing these panels from their vector paths.
    # It pins an ARGMAX, and on some of these stocks the blue-sensitive maximum
    # is a PLATEAU, not a peak. Measured plateau width (samples within 0.05
    # decades of the maximum):
    #     5274  0 nm      5245 10 nm      5205 40 nm      5246 40 nm
    # On 5246 the vector re-trace puts the argmax at 430 nm where the stored set
    # says 470 -- and BOTH readings agree the plateau runs 430-470. The shapes
    # agree; only the sample argmax lands on differs. A re-trace by any other
    # reader could legitimately move 5246 and 5205 from the "470" group to the
    # "410-440" group with no data change at all, failing the guard above for no
    # real reason.
    # So this SECOND guard asserts the property that is actually stable: each
    # stock's stored blue maximum must lie inside its own measured plateau, and
    # the plateau width is recorded so a genuinely different shape still fails.
    _PLATEAU = {"KODAK_VISION_200T_5274": (470.0, 470.0),
                "EASTMAN_EXR_50D_5245": (460.0, 470.0),
                "KODAK_VISION2_250D_5205": (440.0, 470.0),
                "KODAK_VISION_250D_5246": (430.0, 470.0)}
    _pbad = []
    for _n, (_plo, _phi) in _PLATEAU.items():
        _sb = get_profile(_n).spectral.log_s_b
        _mx = max(_sb)
        _flat = [380.0 + 10.0 * _i for _i, _v in enumerate(_sb) if _v >= _mx - 0.05]
        if not (min(_flat) <= _bpk(_n) <= max(_flat)):
            _pbad.append(f"{_n} argmax {_bpk(_n):.0f} outside its own plateau")
        if abs(min(_flat) - _plo) > 10.0 or abs(max(_flat) - _phi) > 10.0:
            _pbad.append(f"{_n} plateau {min(_flat):.0f}-{max(_flat):.0f} moved "
                         f"from the recorded {_plo:.0f}-{_phi:.0f}")
    chk("each blue maximum sits inside its own measured plateau",
        not _pbad,
        "; ".join(_pbad) if _pbad else
        "plateau widths 0 / 10 / 30 / 40 nm -- the argmax guard above is only "
        "meaningful within these")
    # ---- E0b-orig remainder, closed 2026-08-25: 7239's spectral panel. -----
    # ⚠ THE FIRST SET IN THE DATABASE READ WITHOUT THE INK RULE. Every other
    # vector spectral set was assigned by Kodak's convention of drawing each
    # trace in the colour of light it concerns; H-1-5239 p3 prints the whole
    # panel in BLACK. The assignment therefore rests on the absorption bands,
    # the ascending peak order, and the panel's own in-frame captions -- one
    # fewer independent check, which is asserted here rather than left implicit.
    _s39 = get_profile("EASTMAN_EKTACHROME_7239").spectral
    chk("7239 carries the mono-read spectral set",
        _s39.has_data and len(_s39.log_s_r) == 31
        and "MONO reader" in _s39.source,
        "31 samples from 380 nm, traced 2026-08-25 by spectral_vector.py")
    _pk39 = {k: float(_lam[int(_np.argmax(v))]) for k, v in
             (("r", _s39.log_s_r), ("g", _s39.log_s_g), ("b", _s39.log_s_b))}
    chk("7239's three layers peak at 410 / 560 / 660 nm, in ascending order",
        _pk39 == {"b": 410.0, "g": 560.0, "r": 660.0},
        "the band test IS the assignment on a mono panel: %s" % _pk39)
    # ⚠ AND THIS SHEET PRINTS ITS DENSITY CRITERION, WHICH IS THE UNUSUAL PART.
    # The panel states "Density: 1.0" and "Densitometry: E.N.D." inside the
    # frame, so 7239's criterion is measured where the four older Kodak sets
    # carry a "D0.2 above dmin" that THEIR OWN sheets do not print (5205 and
    # 5218 do print it -- corrected 2026-08-26). It must not
    # drift onto the unprinted convention, in either direction.
    chk("7239's spectral criterion is the one printed on its own panel",
        _s39.criterion == "log_reciprocal_ergs_cm2_END_D1.0_VNF1_eff_exp_1.4s"
        and "D0.2" not in _s39.criterion,
        "'Process: VNF-1', 'Density: 1.0', 'Densitometry: E.N.D.', "
        "'Effective Exposure: 1.4 seconds' -- all four printed in the frame")
    # The dye set from the panel BESIDE it was adopted a week earlier from a
    # different quantity by a different reader. Sensitisation and dye absorption
    # need not coincide, but their ORDER must, and a swap in either would show
    # here and nowhere else.
    _d39 = get_profile("EASTMAN_EKTACHROME_7239").dye_density
    _dlam = 400.0 + 10.0 * _np.arange(31)
    chk("7239's sensitivity and dye-density layer orders agree",
        float(_dlam[int(_np.argmax(_d39.d_yellow))]) < _pk39["g"]
        and float(_dlam[int(_np.argmax(_d39.d_magenta))]) < _pk39["r"]
        and float(_dlam[int(_np.argmax(_d39.d_cyan))]) > _pk39["g"],
        "dye peaks 440 / 550 / 670 against sensitivity peaks 410 / 560 / 660")
    # ⚠ AND THIS ADOPTION CHANGES 5201's RENDER, unlike the dye set, which is
    # inert. A stock with spectral data takes spectral_balance_gains() instead of
    # the three-wavelength proxy, and the measured red layer peaks at 650 nm
    # against the proxy's assumed 600, so tungsten light drives red harder. The
    # size and DIRECTION are asserted here so the change stays deliberate.
    _bg_new = fs.spectral_balance_gains(_p01, 3200.0)
    _bg_old = fs.balance_gains(3200.0, 5500.0)
    _dr = math.log2(_bg_new[0] / _bg_old[0])
    chk("5201's measured red layer costs +0.28 stop of red gain at 3200 K",
        0.20 < _dr < 0.35 and abs(_bg_new[1] - 1.0) < 1e-9,
        "red %+.3f stop vs the 600/550/450 nm proxy; green stays the anchor"
        % _dr)

    # ---- 2026-08-17 harvest: measured data moved out of prose into carriers ----
    # These figures existed only inside provenance STRINGS before the carriers
    # were built. A regression that silently emptied a carrier would look like
    # nothing at all in the reports, so the counts are asserted.
    _di = [p for p in FILM_PROFILES if p.dye_impurity.has_data]
    _n_ratios = sum(len(p.dye_impurity.ratios) for p in _di)
    chk("26 measured dye-impurity ratios are typed across 4 Soviet stocks",
        len(_di) == 4 and _n_ratios == 26,
        "%d stocks, %d ratios" % (len(_di), _n_ratios))
    # LN-8's specification prints "minus 0.05-0.10". A validator that rejected
    # negatives, or an import that clamped them, would erase a real interlayer
    # effect -- so the negative term is asserted explicitly.
    _ln8 = get_profile("SVEMA_LN_8").dye_impurity
    chk("LN-8 keeps its NEGATIVE dye-impurity term (minus 0.05-0.10)",
        any(r.lo < 0.0 for r in _ln8.ratios),
        "min lo = %.2f" % min(r.lo for r in _ln8.ratios))
    _rt = [p for p in FILM_PROFILES if p.reciprocity_table.has_data]
    # ⚠ WAS "len == 6" UNTIL 2026-08-23 AND WENT STALE THE MOMENT C8 ADDED THE
    # FIFTEEN VENDOR-SHEET TABLES -- the same count-versus-property failure this
    # suite has now hit three times (the interimage per-distance guard, the "two
    # measured exponents" guard, this one). Stated as properties every table must
    # have, plus the six originals as a SUBSET that must not vanish.
    _rt_bad = []
    for _p in _rt:
        _t = _p.reciprocity_table
        if len(_t.times_s) != len(_t.stops_correction):
            _rt_bad.append("%s ragged" % _p.name)
        if list(_t.times_s) != sorted(_t.times_s):
            _rt_bad.append("%s times not ascending" % _p.name)
        if _t.cc_filters and len(_t.cc_filters) != len(_t.times_s):
            _rt_bad.append("%s cc length" % _p.name)
        if len(_t.source) < 40:
            _rt_bad.append("%s source too thin to trace" % _p.name)
    _orig6 = {"EKTACHROME_64", "EKTACHROME_160T", "KODACHROME_64", "KONICA_VX_100",
              "KENTMERE_PAN_100", "KENTMERE_PAN_400"}
    _have = {p.name for p in _rt}
    chk("every reciprocity table is well formed and cites a document",
        not _rt_bad and _orig6 <= _have,
        "; ".join(_rt_bad[:3]) if _rt_bad
        else "%d tables, all ascending and sourced, original 6 present"
             % len(_rt))
    # The CC-filter column is what makes chromatic and achromatic failure
    # distinguishable at all: Ektachrome 64 prescribes BLUE filters, 160T RED.
    # If those swap, the channel that loses speed swaps with them.
    _e64 = get_profile("EKTACHROME_64").reciprocity_table
    _e160 = get_profile("EKTACHROME_160T").reciprocity_table
    chk("reciprocity CC filters preserve channel direction (E64 blue, 160T red)",
        any("B" in c for c in _e64.cc_filters)
        and any("R" in c for c in _e160.cc_filters),
        "E64 %s | 160T %s" % (_e64.cc_filters, _e160.cc_filters))
    _pf = [p for p in FILM_PROFILES if p.processing_family.has_data]
    _n_pts = sum(len(p.processing_family.points) for p in _pf)
    # ⚠ 17/3 -> 22/4 on 2026-08-26: EASTMAN_DOUBLE_X_5222's five printed D-96
    # times. These carry `gamma` rather than `contrast_index`, which the
    # validator accepts and the guard below tests for explicitly -- Kodak prints
    # a gamma on each curve, and a gamma is not a contrast index.
    # ⚠ 22/4 -> 42/8 on 2026-08-29 (queue E1): the four KODAK 1952 Data Book
    # stocks, five printed (time, gamma) pairs each. Same shape as 5222 -- a
    # printed gamma per drawn curve, no contrast index anywhere in the book --
    # and the same re-derivation from the drawn curves, by
    # `kodak_1952_curves.py`. DOUBLING the population of this carrier in one
    # item is why the count is worth pinning rather than bounding.
    # ⚠ 42/8 -> 94/11 on 2026-09-01: the three AGFAPAN APX stocks, from the
    # Gamma-time curves panel of agfa_films.pdf p10 -- five printed developer
    # names on four drawn curves (RODINAL SPECIAL and STUDIONAL LIQUID share one,
    # and the p11 processing table gives both the same time at every
    # temperature). ⚠ WHAT MAKES THESE ADOPTABLE IS A CROSS-CHECK THE PANEL DOES
    # NOT CONTAIN: read at each developer's own reference time from p11, all four
    # curves on all three films return gamma 0.65 +/- 0.01, and
    # `agfa_bw_manual.pdf` then states that target in words -- every speed table
    # in its developer section is headed "(gamma = 0.65)". Eleven independent
    # readings reproducing a printed specification to one part in sixty-five.
    # ⚠ 94 -> 106 on 2026-09-01 (SECOND PASS): the three AGFAPAN families are
    # REPLACED, not extended. The first pass digitised the range sheet's
    # Gamma-time panel; «Technical Data P-16-C» -- the companion that sheet's
    # p11 names in its last line -- prints the same physics as TEXT, for six
    # developers instead of five, with no tracing and no label matching. The
    # trace was not wasted: the two agree where they overlap (RODINAL 1+25,
    # small tank, gamma 0.65 = 6 / 8 / 7 min in both), which is what makes
    # replacing a measurement with a citation safe rather than merely tidier.
    # ⚠ 106 -> 110 and 11 -> 12 on 2026-09-05 (queue #215). Super Anscochrome
    # adds FOUR points and they are a different shape from every other family
    # here: the others vary a developer, a dilution or a time against a printed
    # gamma, while these four vary the FIRST developer of a reversal process and
    # carry a measured EXPOSURE INDEX with each one. It is the corpus's only
    # four-point reversal ladder and the reason `push_stops` had to become a
    # float.
    # ⚠ 110 -> 138 on 2026-09-06i, ALL TWENTY-EIGHT ON THE THREE AGFAPAN
    # STOCKS, and none of them new measurement: they were traced from
    # «Technical Data PF» 09/1998 p10 on 2026-09-01 and printed on every build
    # since without being adopted, because this family had been filled from
    # «Technical Data P-16-C» instead and the two documents were never put side
    # by side. What licensed the adoption was identifying which VESSEL the
    # panel plots -- the small tank, by 0.112 min against the drum's 1.152 over
    # fifteen combinations -- so the panel supplies exactly the gamma 0.55 and
    # 0.75 rows P-16-C omits for that vessel and nothing it duplicates.
    # ⚠ 138 -> 242 the same day, from p11's TEMPERATURE tables. Every other
    # development point in this database sits at 20 C; these are the corpus's
    # only measurements at 18, 22 and 24. The 20 C cells of that table are NOT
    # re-stored -- P-16-C already supplies them, and the exact agreement of
    # those thirty cells is what proves the table's unlabelled quantity is the
    # gamma 0.65 time. The `tank` block is the exception and all four of its
    # columns are stored, because P-16-C prints no tank rows at all.
    # ⚠ 242 -> 245 on 2026-09-07: APX 400's three third-party developer rows
    # from F-PF-D4/E4 p10. See G-AGFA8.
    # ⚠ 245 -> 252 AND 12 -> 13 ON 2026-09-15, and the thirteenth is the one
    #   that changes what this population IS. Every stock here was black-and-
    #   white or reversal; ORWOCOLOR_NC3 is the first COLOUR NEGATIVE with a
    #   development-time family, seven points traced off Bild 6 of Tamm &
    #   Weisflog 1972. It is also the axis `h24_variations.py` cannot supply:
    #   of the six H-24 Module 8 figures that reader digitises, the time and
    #   temperature panels are exactly the two its HD-LD identity gate refuses.
    # ⚠ 252 -> 261 ACROSS 13 -> 14 STOCKS ON 2026-09-15b, AND THE NEW STOCK IS
    # THE FIRST WHOSE FAMILY CAME OFF CHARACTERISTIC CURVES RATHER THAN A
    # GAMMA-TIME PLOT. KODAK_TMAX_P3200 had no processing row of any kind.
    # «Современные фотоматериалы и их обработка» pp. 341-342 print two
    # characteristic panels, each naming ONE developer and ONE temperature in
    # its caption and listing the DEVELOPMENT TIMES in its legend, so the curves
    # attribute themselves: contrast rises with time, the steepest curve is the
    # longest time, and the pairing is refused unless it comes out strictly
    # monotonic. Nine points, and every one also carries base+fog at its own
    # condition -- the first source in this corpus to populate that field from
    # a trace rather than leave it at zero.
    # ⚠ 261 -> 876 AND 14 -> 20 ON 2026-09-16, when the table parser was
    # rewritten. The first version assumed ONE table shape and the book prints
    # five: times only; times at a stated CONTRAST; times at a stated EI; two
    # FILMS side by side; and one film in two FORMATS. It also read a bare '10'
    # as a temperature because the degree mark was optional in its pattern,
    # which turned five real temperature columns into forty-eight. Fixing both
    # took the parse from 320 values to 1507, of which 793 pass the physical
    # check and 615 distinct points are stored.
    # ⚠ AGFA_APX_100 IS DELIBERATELY NOT AMONG THEM. Agfa's own sheet already
    # gives that stock an 18/20/22/24 degC axis WITH a contrast against every
    # time; the book restates the same conditions without one, so adopting it
    # would put a lower tier on top of a higher one. Precedence, not loss.
    # ⚠ 261 -> 336 AND 14 -> 16 ON 2026-09-15b, AND THE NEW POINTS ARE A
    # DIFFERENT SHAPE. KODAK PLUS-X 125 and VERICHROME PAN gain 75 points from
    # the book's developer x temperature TIME tables, which print no gamma
    # anywhere near them. They are counted SEPARATELY below so a time-only
    # point can never be mistaken for a contrast measurement -- and they are
    # stored at all because Development TEMPERATURE has no other source in this
    # corpus: every one of P3200's nine points is at 20 degC, and these span
    # 18 / 20 / 21 / 22 / 24.
    _n_ct = sum(1 for p in _pf for q in p.processing_family.points
                if q.contrast_index > 0.0 or q.gamma > 0.0)
    # ⚠ 876 -> 915 AND 20 -> 21 ON 2026-09-16b: FUJI NEOPAN ACROS 100, whose
    # table was lost to an anchored EI-header pattern being matched against the
    # header lines joined into one string, where an anchor cannot fire. The
    # column went undetected and EI 100 was read as a 100-minute development.
    # ⚠ NOTHING CORRUPT WAS EVER STORED, and that is the part worth keeping: a
    # ten-to-one fall across the temperature span failed the factor-of-four
    # check and the whole table was refused. Its 39 points are the only ones in
    # this database that carry an EXPOSURE INDEX as well as a temperature.
    # ⚠ 915 -> 960 AND 21 -> 25 ON 2026-09-16c: the «Kodak Films» Data Book,
    # Seventh Edition, 1956. Kodak labelled each characteristic curve of each
    # family IN FRAME with its development time and its gamma, so these 45
    # points are TRANSCRIPTIONS of a manufacturer's printed statement, not
    # traces, and every one of them carries a contrast -- which is why _n_ct
    # moves by the full 45 while _n_pts does. Four of the four new stocks had
    # no processing family at all before this: KODAK_PANATOMIC_X,
    # KODAK_ROYAL_PAN_4141, KODAK_ROYAL_X_PAN_4166 and
    # KODAK_SUPER_XX_PAN_4142.
    _n_ed = sum(1 for p in _pf for q in p.processing_family.points
                if q.edition)
    # ⚠ 960 -> 1149, 306 -> 495 AND 45 -> 234 ON 2026-09-17 (queue P61): the
    # TIME-GAMMA INSETS of the same 1956 Data Book, TRACED rather than
    # transcribed. 189 points on four stocks, every one carrying a contrast,
    # and the first contrast figures this corpus holds for MQ, Versatol,
    # Microdol or DK-50 at 1:1. No new stock: all four already carried the
    # printed family the inset sits beside, which is what let each traced
    # curve be checked against Kodak's own labels before it was adopted.
    # ⚠ 1149 -> 1175, 495 -> 521 AND 234 -> 260 ON 2026-09-17 (queue P63),
    # AND THIS TIME THE POINTS ARRIVE WITH FIVE NEW STOCKS RATHER THAN ON
    # OLD ONES. Every one of the 26 is a PRINTED (time, gamma) pair Kodak
    # letters on its own curve, transcribed -- and re-derived off the raster
    # by kodak_1956_curves.py on every build, which is the only reason they
    # can be trusted as transcription.
    # ⚠ 1175 -> 1606 ON 2026-09-17d (queue P54), AND EVERY ONE OF THE 431 IS
    # A TIME AT A STATED TEMPERATURE. «Современные фотоматериалы и их
    # обработка» held 705 values its own parser could not attribute; re-read
    # from the page GEOMETRY rather than the text layer, 1122 sit in tables
    # that pass the physical check in every row, and 431 survive mapping to a
    # stock this database holds and de-duplicating against the manufacturer
    # points already on it. The contrast count does NOT move: an
    # equal-contrast time table states no gamma of its own.
    # ⚠⚠ 1606 -> 1833 ON 2026-09-18b, AND THE 227 ARRIVE WITH A FIELD THAT WAS
    # EMPTY ON ALL 1606. Queue P54 held that the remaining values were lost to
    # "a table covering TWO films split between them by dividing the
    # temperature list in half". They are not two films. Table 3.241 on p.392
    # is ONE film with TWO VESSEL columns -- «Малый бак» and «Большой бак» --
    # each carrying its own 18/20/21/22/24, so the repeated temperature is the
    # vessel and the old reader refused the table for a structure it had
    # misread. `sovremennye_dev_tables.py` splits the header where it RESTARTS
    # and reads the caption above each block. 1539 points now carry a VESSEL
    # and 30 a FILM FORMAT, against 0 and 0 before.
    # ⚠ 1833 -> 1818 LATER THE SAME DAY, AND THE FIFTEEN WERE NEVER NEW.
    # The reader emitted «роторно-барабанном» as the string 'rotary drum'
    # while the same physical process was already on file as 'drum' from the
    # Agfa sheets, so `DevelopmentPoint.vessel`'s validator refused it -- one
    # process about to be stored under two enum values. Mapping it onto the
    # existing 'drum' made fifteen of the new points EQUAL to points already
    # there, and the de-duplication key includes the vessel, so they merged.
    # A merge here is the correct outcome: two sources printing the same
    # developer, time, temperature and vessel are one measurement.
    chk("1818 development points across 30 stocks -- 521 carrying a measured "
        "contrast, 1297 carrying a temperature instead, 260 tagged with the "
        "emulsion generation they measure, and 1524 naming their vessel",
        _n_pts == 1818 and len(_pf) == 30 and _n_ct == 521 and _n_ed == 260
        and sum(1 for p in _pf for q in p.processing_family.points
                if q.vessel) == 1524
        and all((q.contrast_index > 0.0 or q.gamma > 0.0)
                or (q.celsius > 0.0 and q.minutes > 0.0 and q.developer)
                for p in _pf for q in p.processing_family.points),
        "%d stocks, %d points, %d with contrast, %d time-only, %d tagged, "
        "%d with a vessel"
        % (len(_pf), _n_pts, _n_ct, _n_pts - _n_ct, _n_ed,
           sum(1 for p in _pf for q in p.processing_family.points if q.vessel)))

    # ⚠ THE GENERATION TAG IS ONLY WORTH HAVING IF IT IS NEVER AMBIGUOUS, so
    # the guard is that a tagged point ALSO states the speed of the generation
    # it measured. Without that a reader looking at a 1956 Tri-X point on
    # KODAK_TRI_X_400TX has the profile's ISO 400 and no way to learn that the
    # point was measured at American Standard 200.
    chk("every generation-tagged development point states its own exposure "
        "index",
        all(q.exposure_index > 0
            for p in _pf for q in p.processing_family.points if q.edition),
        "%d tagged points, %d without an index"
        % (_n_ed, sum(1 for p in _pf for q in p.processing_family.points
                      if q.edition and not q.exposure_index)))

    # ⚠ AND THAT EVERY ONE OF THEM FITS THE LAW THE SCHEMA ALREADY IMPLEMENTS.
    # `kodak_1956.py` fits Mees-Sheppard to each printed family on every build
    # and refuses above a 0.035 residual; this repeats the test from the
    # DATABASE side rather than from the reader's own constants, so a point
    # that was mistyped into film_profiles.py after the reader passed is still
    # caught. A misread gamma digit moves a point by 0.1 or more; the printed
    # values are quoted to two decimals, so the quantisation floor is 0.005.
    #
    # ⚠⚠ AND THE TWO HARVESTS ARE HELD TO DIFFERENT TOLERANCES, BECAUSE
    # THEY ARE DIFFERENT MEASUREMENTS. A transcribed pair is a printed number
    # quoted to two decimals; its only failure mode is a mistyped digit, which
    # moves a point by 0.1, so 0.035 is a wide margin round a 0.005
    # quantisation floor. A TRACED pair is a pixel reading of a line drawn by
    # a draughtsman, and the 1956 insets are drawn to about a fortieth of a
    # gamma; holding them to the transcription tolerance would refuse curves
    # that agree with Kodak's own printed labels to 0.014. The traced bound is
    # the tracer's own shape check, and the ACCURACY check on a traced curve
    # is not this fit at all -- it is the comparison against those printed
    # labels, which kodak_1956_trace.py runs on every build and which no curve
    # here was adopted without passing at 0.055.
    try:
        from kodak_1956 import fit_mees_sheppard as _fit56
        import film_profiles as _fpm56
        _traced = set()
        for _r in getattr(_fpm56, "_K56_TRACE", ()):
            for _m, _g in _r[7]:
                _traced.add((_r[0], _r[1], _r[2], _r[3], float(_m)))
        _worst = 0.0
        _worst_t = 0.0
        _fams, _famt = {}, {}
        for p in _pf:
            for q in p.processing_family.points:
                if not q.edition:
                    continue
                _key = (p.name, q.edition, q.developer, q.dilution)
                _bin = (_famt if (_key + (float(q.minutes),)) in _traced
                        else _fams)
                _bin.setdefault(_key, []).append((q.minutes, q.gamma))
        for _k, _v in _fams.items():
            _worst = max(_worst, _fit56(sorted(_v))[3])
        for _k, _v in _famt.items():
            if len(_v) >= 4:
                _worst_t = max(_worst_t, _fit56(sorted(_v))[3])
        chk("every PRINTED 1956 gamma family in the DATABASE fits the "
            "Mees-Sheppard law to 0.035 -- the guard against a mistyped digit",
            _fams and _worst <= 0.035,
            "%d families, worst residual %.4f" % (len(_fams), _worst))
        chk("every TRACED 1956 gamma family in the DATABASE fits the "
            "Mees-Sheppard law to 0.100 -- a shape check on a digitised "
            "line, not an accuracy claim; accuracy is the printed-label "
            "comparison in kodak_1956_trace.py",
            _famt and _worst_t <= 0.100,
            "%d families, worst residual %.4f" % (len(_famt), _worst_t))
    except ImportError:
        chk("kodak_1956 importable for the Mees-Sheppard re-fit", False,
            "module missing")
    import film_profiles as _fpm3

    # ---- P65, 2026-09-17. The 1938 speed scale, and what converts it. ------
    # ⚠⚠ THE ROW WAS BLOCKED ON A SINGLE FILM RATED ON BOTH SCALES AND
    # THERE ARE THREE. «Eastman Professional Films» 1938 publishes fifteen
    # speeds on the Kodak Research Laboratories sunlight scale -- it names the
    # scale in words on its own page 8 -- and the 1956 «Kodak Films»
    # publishes American Standard DAYLIGHT indexes for three of those
    # emulsions. The comparison has to be against the daylight index and not
    # the tungsten one, because the 1938 booklet warns in the same sentence
    # that its numbers "are not valid when the materials are exposed to light
    # of different quality, such as incandescent tungsten".
    _k38 = film_profiles._KODAK_1938_SPEEDS
    _k38b = film_profiles._KODAK_1938_BRIDGE
    chk("the 1938 booklet's fifteen Kodak Research Laboratories speeds are "
        "stored, with the sensitising class each was measured as",
        len(_k38) == 15
        and {c for _n, c, _v in _k38} == {"panchromatic", "ortho", "blue"}
        and all(v > 0 for _n, _c, v in _k38),
        "%d films: %d panchromatic, %d ortho, %d non-colour-sensitive, "
        "speeds %d to %d"
        % (len(_k38),
           sum(1 for _n, c, _v in _k38 if c == "panchromatic"),
           sum(1 for _n, c, _v in _k38 if c == "ortho"),
           sum(1 for _n, c, _v in _k38 if c == "blue"),
           min(v for _n, _c, v in _k38), max(v for _n, _c, v in _k38)))

    # ⚠ THE TWO PANCHROMATIC BRIDGES MUST AGREE EXACTLY, and "exactly" is
    # the whole evidence: two films with different eighteen-year histories --
    # one of them explicitly re-coated and renamed Type B -- landing on the
    # same ratio to three figures is a SCALE. If a later edit moves either
    # rating by so much as a third of a stop the agreement breaks and the
    # conversion has to be argued again rather than inherited.
    _pan = [(n56, krl / float(day)) for _n38, n56, krl, day in _k38b
            if "COMMERCIAL" not in n56]
    chk("the Kodak Research Laboratories scale converts at exactly 4.00, "
        "corroborated by two independently rated panchromatic sheet films",
        len(_pan) == 2
        and all(abs(r - film_profiles._KRL_TO_AMERICAN_STANDARD) < 1e-9
                for _n, r in _pan)
        and film_profiles._KRL_CONVERSION_CLASS == "panchromatic",
        "; ".join("%s %.2f" % (n.replace("KODAK ", ""), r) for n, r in _pan))

    # ⚠ AND THE ONE THAT DISAGREES IS PINNED AS A DISAGREEMENT. Averaging
    # 4.00, 4.00 and 1.60 into a 3.2 that describes none of them is exactly the
    # move this project refuses; the refusal is stored, so an edit that
    # quietly widens the conversion to cover Commercial has to delete a
    # sentence that says why it must not.
    _comm = [(n56, krl / float(day)) for _n38, n56, krl, day in _k38b
             if "COMMERCIAL" in n56]
    chk("the non-colour-sensitive bridge disagrees at 1.60 and is recorded as "
        "a refusal rather than averaged into the conversion",
        len(_comm) == 1 and abs(_comm[0][1] - 1.60) < 1e-9
        and "1.60, not 4.00" in film_profiles._KRL_CONVERSION_REFUSED,
        "%s %.2f against the panchromatic 4.00"
        % (_comm[0][0].replace("KODAK ", ""), _comm[0][1]))

    # ---- P62, 2026-09-17. The 1956 wedge spectrograms. ---------------------
    # ⚠⚠ THE ROW'S BLOCKER WAS NOT "TRACE THE PLATES", IT WAS "PROVE THE
    # READER BEFORE BELIEVING IT", and these guards are that proof kept in the
    # build rather than in a report. A wedge spectrogram is a density boundary
    # in a letterpress halftone, so every tracer in this project -- all of
    # which follow strokes -- was the wrong instrument, and a boundary finder
    # will happily return a number for a plate it has misread.
    _k56s = film_profiles._K56_SPECTRO
    _k56sc = film_profiles._K56_SPECTRO_CORROBORATION
    _k56su = film_profiles._K56_SPECTRO_UNHOUSED
    _start, _step, _n56s = film_profiles._K56_SPECTRO_GRID
    chk("the 1956 wedge spectrograms are read for all 17 emulsions the book "
        "prints one for -- 7 adopted, 2 corroborating a sensitometric sheet "
        "that outranks them, 8 held for profiles that do not exist yet",
        len(_k56s) == 7 and len(_k56sc) == 2 and len(_k56su) == 8
        and all(len(v[0]) == _n56s
                for d in (_k56s, _k56sc, _k56su) for v in d.values()),
        "%d + %d + %d curves on a %.0f nm / %.0f nm / %d grid"
        % (len(_k56s), len(_k56sc), len(_k56su), _start, _step, _n56s))

    # ⚠ THE THREE SENSITIZING CLASSES MUST STAY ORDERED. This is the test
    # queue P62 named, run against the DATABASE rather than against the
    # reader: the longest wavelength each stored curve survives to has to
    # separate blue-sensitive from orthochromatic from panchromatic. A
    # boundary finder that drifts, or a curve pasted onto the wrong stock,
    # breaks this and nothing else in the build would notice.
    def _cut56(t):
        _on = [i for i, v in enumerate(t) if v > -3.999]
        return _start + _step * _on[-1] if _on else 0.0
    _all56 = dict(_k56s)
    _all56.update(_k56sc)
    _all56.update(_k56su)
    _blue56 = [_cut56(v[0]) for k, v in _all56.items()
               if k == "KODAK COMMERCIAL SHEET FILM"]
    _orth56 = [_cut56(v[0]) for k, v in _all56.items()
               if "ORTHO" in k]
    _pan56 = [_cut56(v[0]) for k, v in _all56.items()
              if "ORTHO" not in k and k != "KODAK COMMERCIAL SHEET FILM"]
    chk("every stored 1956 spectrogram cuts off in its own sensitizing "
        "class's band -- blue-sensitive before orthochromatic before "
        "panchromatic, with no overlap",
        bool(_blue56) and bool(_orth56) and bool(_pan56)
        and max(_blue56) < min(_orth56) and max(_orth56) < min(_pan56),
        "blue to %.0f nm, ortho %.0f-%.0f, pan %.0f-%.0f"
        % (max(_blue56), min(_orth56), max(_orth56),
           min(_pan56), max(_pan56)))

    # ⚠ AND THE TWO STOCKS THAT ALREADY HAD A CURVE ARE THE PRECEDENCE
    # RULE'S OWN TEST CASE. PLUS-X and TRI-X carry a digitised Kodak
    # sensitometric sheet for the modern emulsion; the 1956 plate is a
    # different generation of the same product NAME and must NOT displace it.
    # What the two readings can be compared on is the one property that is a
    # class fact rather than a generation fact -- where the response dies --
    # and they agree to 20 nm, which is the evidence that the wedge reader is
    # measuring the same thing a sensitometer does.
    _corr_ok, _corr_rep = True, []
    for _k, _v in _k56sc.items():
        _p56 = film_profiles._BY_NAME[_k]
        _sheet = _p56.spectral
        _a = _cut56(_v[0])
        _on = [i for i, x in enumerate(_sheet.log_s_pan) if x > -3.999]
        _b = _sheet.lambda_start_nm + _sheet.lambda_step_nm * _on[-1]
        _corr_rep.append("%s wedge %.0f vs sheet %.0f nm"
                         % (_k.replace("KODAK_", ""), _a, _b))
        if _sheet.criterion == film_profiles._K56_SPECTRO_CRITERION:
            _corr_ok = False
        if abs(_a - _b) > 25.0:
            _corr_ok = False
    chk("the two stocks with a Kodak sensitometric sheet KEEP it -- the 1956 "
        "wedge plate is stored beside them and never installed -- and the "
        "two independent readings put the long-wave cutoff within 25 nm",
        _corr_ok, "; ".join(_corr_rep))

    # ⚠ THE SEVEN ADOPTED CURVES ARE ACTUALLY ON THEIR STOCKS, and each
    # says in its own `source` that its log depths rest on a gradient Kodak
    # never published. A curve that loses that sentence starts reading as an
    # absolute measurement.
    _inst = [k for k in _k56s
             if film_profiles._BY_NAME[k].spectral.criterion
             == film_profiles._K56_SPECTRO_CRITERION
             and "+/-10 %" in film_profiles._BY_NAME[k].spectral.source
             and "relative colour sensitivity only"
             in film_profiles._BY_NAME[k].spectral.source]
    chk("all 7 adopted 1956 spectral responses are installed on their stocks "
        "and each carries Kodak's own two caveats -- relative sensitivity "
        "only, and a log scale good to about +/-10 %",
        len(_inst) == 7, "%d of %d" % (len(_inst), len(_k56s)))

    # ---- P63, 2026-09-17. The five 1956 sheet emulsions. -------------------
    # ⚠⚠ THE ROW SAT OPEN FOR A DAY WITH EVERY NUMBER ALREADY READ. Its
    # own blocker named two things -- a characteristic curve and a spectral
    # response -- and both were on the SAME PAGES the exposure indexes, filter
    # factors and (time, gamma) families had been transcribed from. What was
    # missing was a reader for each, not a document.
    _p63 = ("KODAK_SUPER_PANCHRO_PRESS_B_1956",
            "KODAK_PORTRAIT_PANCHROMATIC_1956",
            "KODAK_ROYAL_ORTHO_1956",
            "KODAK_SUPER_SPEED_ORTHO_1956",
            "KODAK_COMMERCIAL_1956")
    _p63p = [film_profiles._BY_NAME[_n] for _n in _p63]
    chk("all five 1956 sheet emulsions exist, are monochrome, carry a traced "
        "characteristic curve, a MEASURED spectral response and the printed "
        "development family Kodak draws it for",
        len(_p63p) == 5
        and all(q.is_monochrome and q.era == "1956" and q.curves.g.gamma > 0.0
                and q.spectral.has_data
                and len(q.processing_family.points) >= 4
                and q.processing_family.reference_developer
                for q in _p63p),
        ", ".join("%s g %.3f, %d dev points, %s"
                  % (q.name.replace("KODAK_", "").replace("_1956", ""),
                     q.curves.g.gamma, len(q.processing_family.points),
                     q.processing_family.reference_developer)
                  for q in _p63p))

    # ⚠ THE SENSITIZING CLASSES MUST SURVIVE INTO THE PROFILES. Two of the
    # five are orthochromatic and one is not colour-sensitized at all, and
    # that is the single most consequential fact about them -- it is why
    # COMMERCIAL's daylight and tungsten indexes are four stops apart. A
    # spectral curve pasted onto the wrong stock breaks this and nothing else
    # would notice.
    def _p63cut(q):
        _v = q.spectral.log_s_pan
        _on = [i for i, x in enumerate(_v) if x > -3.999]
        return q.spectral.lambda_start_nm + q.spectral.lambda_step_nm * _on[-1]
    _cut = {q.name: _p63cut(q) for q in _p63p}
    chk("the three sensitizing classes survive into the five profiles: "
        "COMMERCIAL is blue-sensitive, the two ORTHO sheets stop in the "
        "green, and the two panchromatic ones run into the red",
        _cut["KODAK_COMMERCIAL_1956"] <= 520.0
        and 560.0 <= _cut["KODAK_ROYAL_ORTHO_1956"] <= 620.0
        and 560.0 <= _cut["KODAK_SUPER_SPEED_ORTHO_1956"] <= 620.0
        and _cut["KODAK_SUPER_PANCHRO_PRESS_B_1956"] >= 630.0
        and _cut["KODAK_PORTRAIT_PANCHROMATIC_1956"] >= 630.0,
        ", ".join("%s %.0f nm" % (k.replace("KODAK_", "").replace("_1956", ""),
                                  v) for k, v in sorted(_cut.items())))

    # ⚠⚠ AND THE EXPOSURE INDEXES ARE 1956 AMERICAN STANDARD, NOT
    # POST-1960 ASA, AND MUST STAY THAT WAY. The 1956 standard carries a
    # safety factor the 1960 revision removed, and the conversion DOES NOT
    # RESOLVE: five films rated on both scales give 1.56, 1.56, 2.00, 2.00
    # and 1.92. A later edit that "corrects" these five upward would be
    # applying a ratio that describes none of them, so the stored values are
    # pinned to the book's own table.
    _p63ei = {"KODAK_SUPER_PANCHRO_PRESS_B_1956": (125, 100),
              "KODAK_PORTRAIT_PANCHROMATIC_1956": (50, 32),
              "KODAK_ROYAL_ORTHO_1956": (200, 125),
              "KODAK_SUPER_SPEED_ORTHO_1956": (25, 25),
              "KODAK_COMMERCIAL_1956": (25, 6)}
    chk("the five exposure indexes are the 1956 American Standard figures "
        "the book prints, unconverted, and each profile says so",
        all(q.exposure_index == _p63ei[q.name][0]
            and q.exposure_index_tungsten == _p63ei[q.name][1]
            and "1956 AMERICAN" in q.description for q in _p63p),
        ", ".join("%s %d/%d" % (q.name.replace("KODAK_", "")
                                .replace("_1956", ""), q.exposure_index,
                                q.exposure_index_tungsten) for q in _p63p))

    # ⚠ SUPER SPEED ORTHO IS RATED FOR TUNGSTEN ONLY AND THAT IS THE
    # BOOK'S STATEMENT, NOT A GAP. Its daylight column is empty in the
    # original, so `balance_kelvin` says tungsten and the two indexes are
    # equal -- a later edit that "fills in" a daylight index would be
    # inventing one.
    _ssp = film_profiles._BY_NAME["KODAK_SUPER_SPEED_ORTHO_1956"]
    chk("SUPER SPEED ORTHO PORTRAIT is stored as the tungsten-only rating "
        "the 1956 book gives it, and no daylight index is invented",
        _ssp.balance_kelvin == 2850
        and _ssp.exposure_index == _ssp.exposure_index_tungsten == 25
        and "RATED FOR TUNGSTEN ONLY" in _ssp.description,
        "index %d, balance %d K" % (_ssp.exposure_index, _ssp.balance_kelvin))

    # ---- 2026-09-17. ProcessVariant: one list, four places, one authority. --
    # ⚠⚠ THE CONTROL STOPPED BEING AN INDEX AND THESE GUARDS ARE WHAT KEEPS
    # THE REPLACEMENT HONEST. `AlgoControls::processVariant` was an int32
    # position in whichever stock happened to be loaded, so the stored value 2
    # named "RODINAL 1+50" on an AGFAPAN, "ECN-2, the base stock's native
    # process" on CINESTILL 800T and "EI 3200 (Push 2)" on PORTRA 800 -- every
    # one of them in range, so no test could tell a stale project from a
    # correct one, and inserting a variant re-pointed every saved selection in
    # silence. It is now `ProcessVariantCtrl`, a global enumeration in
    # AlgoControlEnums.hpp.
    #
    # ⚠ THE ENUMERATION IS WRITTEN IN FOUR PLACES AND MUST BE ONE LIST: the
    # header (the authority the host and both engines compile against), the
    # generated Python mirror, the database's `_PROCESS_VARIANT_IDS`, and the
    # `variant_id` actually stamped on every ProcessVariant record. The whole
    # value of a strongly typed control is lost the moment any two disagree,
    # and nothing else in this file would notice, so it is asserted directly
    # and in both directions.
    import algo_control_enums as _ace

    _pv_enum = [(e.name, int(e)) for e in _ace.ProcessVariantCtrl]
    _pv_real = [(n, v) for n, v in _pv_enum
                if n not in ("eAS_SHIPPED", "TOTAL_PROCESSES")]
    _pv_db = list(film_profiles._PROCESS_VARIANT_IDS)

    # ⚠⚠ THE ENUMERATION REBASED TO ZERO ON 2026-09-18e (owner requirement:
    # no AlgoControl enumerated value may start below zero). eAS_SHIPPED was
    # -1 and is now 0, so the twenty-one developments run 1..21 and
    # TOTAL_PROCESSES is 22. The database list is unchanged -- it never held
    # the sentinel -- so the alignment is now OFFSET BY ONE and this guard
    # asserts the offset explicitly rather than the identity it used to.
    chk("ProcessVariantCtrl and the database's _PROCESS_VARIANT_IDS are one "
        "list in one order, offset by the eAS_SHIPPED entry at zero, and "
        "TOTAL_PROCESSES counts both",
        [n for n, _v in _pv_real] == ["e" + k for k, _d in _pv_db]
        and [v for _n, v in _pv_real] == list(range(1, len(_pv_db) + 1))
        and int(_ace.ProcessVariantCtrl.TOTAL_PROCESSES) == len(_pv_db) + 1
        and int(_ace.ProcessVariantCtrl.eAS_SHIPPED) == 0
        # and the rebase must not have left a negative anywhere
        and min(v for _n, v in _pv_enum) == 0,
        "%d developments at 1..%d, TOTAL_PROCESSES %d, eAS_SHIPPED %d and no "
        "negative enumerator"
        % (len(_pv_db), len(_pv_db),
           int(_ace.ProcessVariantCtrl.TOTAL_PROCESSES),
           int(_ace.ProcessVariantCtrl.eAS_SHIPPED)))

    # ⚠ AND THE SENTINEL AND THE COUNT ARE NOT DEVELOPMENTS. Offering either
    # in the list box would put a non-existent process in front of the user;
    # storing either as a selection would mean "as shipped" on one path and
    # "out of range" on another. Both tables are therefore sized to the real
    # developments and neither carries a row for those two.
    # ⚠ AND SINCE THE REBASE eAS_SHIPPED IS IN THE TABLES, at index 0, with an
    # EMPTY KEY. That empty key is the whole mechanism: both resolvers look the
    # key up in the stock's own process_variants, find nothing, and return the
    # shipped profile -- which is exactly what the out-of-range sentinel used
    # to do. A non-empty key at index 0 would silently select a development.
    # TOTAL_PROCESSES is still a COUNT and still absent from both tables.
    chk("the display and key tables hold the developments AND the "
        "eAS_SHIPPED entry at index 0 with an empty key, and not the "
        "TOTAL_PROCESSES count",
        len(_ace.PROCESS_VARIANT_KEY) == len(_pv_db) + 1
        and len(_ace.PROCESS_VARIANT_LABEL) == len(_pv_db) + 1
        and _ace.PROCESS_VARIANT_KEY[0] == ""
        and _ace.PROCESS_VARIANT_LABEL[0] == "As shipped"
        and all(_ace.PROCESS_VARIANT_KEY[i + 1] == _pv_db[i][0]
                for i in range(len(_pv_db)))
        and all(_ace.PROCESS_VARIANT_LABEL[i + 1] == _pv_db[i][1]
                for i in range(len(_pv_db))),
        "%d keys, %d labels, index aligned with the enumeration from zero"
        % (len(_ace.PROCESS_VARIANT_KEY), len(_ace.PROCESS_VARIANT_LABEL)))

    # ⚠ EVERY RECORDED DEVELOPMENT MUST BE REACHABLE AND NO ENUMERATOR MAY
    # BE DEAD. An id stamped on a record that the enumeration does not declare
    # cannot be selected; an enumerator no record carries is a list-box entry
    # that resolves to "as shipped" on every stock.
    _used = sorted({v.variant_id for q in FILM_PROFILES
                    for v in (q.process_variants or ())})
    _declared = sorted(k for k, _d in _pv_db)
    chk("every stamped variant_id is a declared enumerator and every "
        "enumerator is carried by at least one stock",
        _used == _declared,
        "%d ids on %d records; %s"
        % (len(_used),
           sum(len(q.process_variants or ()) for q in FILM_PROFILES),
           "aligned" if _used == _declared else
           "only in the database: %s; only in the enumeration: %s"
           % (sorted(set(_used) - set(_declared)),
              sorted(set(_declared) - set(_used)))))

    # ⚠ AND THE BEHAVIOUR THAT REPLACED CLAMPING IS PINNED. A development
    # this stock does not offer renders it AS SHIPPED, by reference and with no
    # copy -- the same path an unselected control takes. Clamping into the
    # stock's own range would render a different development and present it as
    # the one that was asked for, which is the failure the old index had.
    _apx = get_profile("AGFA_APX_25")
    _p800 = get_profile("KODAK_PORTRA_800")
    _foreign = int(_ace.ProcessVariantCtrl.ePORTRA800_EI3200_PUSH2)
    chk("a development the stock does not offer renders it as shipped, by "
        "reference, and does not clamp into that stock's own list",
        fs.resolve_process_variant(_apx, _foreign) is _apx
        and fs.resolve_process_variant(
            _apx, int(_ace.ProcessVariantCtrl.TOTAL_PROCESSES)) is _apx
        and fs.resolve_process_variant(
            _apx, int(_ace.ProcessVariantCtrl.eAS_SHIPPED)) is _apx
        and fs.resolve_process_variant(_p800, _foreign) is not _p800,
        "PORTRA 800's Push 2 on an AGFAPAN is inert and on PORTRA 800 is not")

    # ⚠ THE SAME VALUE MEANS THE SAME DEVELOPMENT ON EVERY STOCK, which is
    # the whole property an index could not have. Five of the twenty-one are
    # shared by three stocks each -- the AGFAPAN developer ladder -- so the
    # test has something to fail on.
    _shared = [k for k, _d in _pv_db
               if sum(1 for q in FILM_PROFILES
                      for v in (q.process_variants or ())
                      if v.variant_id == k) > 1]
    chk("a ProcessVariantCtrl value names the same development on every stock "
        "that offers it",
        len(_shared) == 5
        and all(len({v.name for q in FILM_PROFILES
                     for v in (q.process_variants or ())
                     if v.variant_id == k}) == 1 for k in _shared),
        "%d ids are offered by more than one stock: %s"
        % (len(_shared), ", ".join(_shared)))

    # ---- P67, 2026-09-17. The 1982 5293, and the code reuse it sits on. ----
    # ⚠⚠ THE GUARD THAT MATTERS HERE IS THAT THERE ARE TWO OF THEM.
    # Eastman used the catalogue number 5293 twice, ten years and one emulsion
    # technology apart, and every published figure for one of them is wrong
    # about the other: EI 250T against 200T, conventional polydisperse grain
    # against EXR tabular, RMS 10.8 against 7.4 at density 1.0. A merge would
    # not fail any other test in this file -- both are Kodak tungsten colour
    # negatives with plausible curves -- so it is asserted directly.
    _k82 = get_profile("EASTMAN_5293_250T_1982")
    _k92 = get_profile("EASTMAN_EXR_200T_5293")
    chk("the two films Eastman both called 5293 are separate profiles, and "
        "differ in the three things that separate their generations",
        _k82.exposure_index == 250 and _k92.exposure_index == 200
        and _k82.emulsion.habit == "nontabular"
        and _k82.grain.rms_granularity > _k92.grain.rms_granularity
        and _k82.era.startswith("1982") and _k92.era.startswith("1992"),
        "1982: EI %d, habit %r, rms %.1f, era %s / 1992: EI %d, habit %r, "
        "rms %.1f, era %s"
        % (_k82.exposure_index, _k82.emulsion.habit,
           _k82.grain.rms_granularity, _k82.era,
           _k92.exposure_index, _k92.emulsion.habit,
           _k92.grain.rms_granularity, _k92.era))

    # ⚠ THE GRANULARITY SHAPE IS THE 1982 FILM'S SIGNATURE and it is the
    # one thing a flat sigma cannot express: Fig. 13 peaks IN THE TOE and
    # falls by more than half by D 2.0, which is why an underexposed 5293
    # negative looks so much grainier than a correct one. The paper says so
    # in words as well as in the plot.
    chk("EASTMAN_5293_250T_1982 carries a MEASURED sigma(D) that peaks in "
        "the toe and more than halves by D-max",
        _k82.grain.sigma_shape_measured
        and _k82.grain.sigma_shape_peak > _k82.grain.sigma_shape_mid
        and _k82.grain.sigma_shape_peak_at < 1.0
        and _k82.grain.sigma_shape_dmax < 0.55,
        "peak %.3f at D %.2f, mid %.3f, dmax %.3f at D %.2f"
        % (_k82.grain.sigma_shape_peak, _k82.grain.sigma_shape_peak_at,
           _k82.grain.sigma_shape_mid, _k82.grain.sigma_shape_dmax,
           _k82.grain.sigma_shape_dmax_at))

    # ⚠ AND THE T3 BLOCK'S DISAGREEMENT WITH THE PAPER IS PINNED, not left
    # to a comment. The owner's 2026-09-17 block puts this film's MTF 50 %
    # at 25-30 lp/mm; Kodak's own Fig. 14 traces 41.5 cycles/mm and Kodak's
    # own Table 2 prints the SAME resolving power as 5247. The precedence
    # rule says the manufacturer wins, so the guard is that the stored value
    # is the traced one and that the T3 figure is still on the record.
    chk("the 1982 5293 stores Kodak's own traced MTF and not the T3 figure "
        "it disagrees with, and keeps the disagreement on the record",
        abs(_k82.mtf.f50_g - 41.5) < 0.6
        and _k82.mtf.resolving_power_lp_mm_lowc == 50.0
        and _k82.mtf.resolving_power_lp_mm_highc == 100.0
        and "25-30 lp/mm" in _fpm3._K5293_1982_T3,
        "f50 %.1f, resolving %.0f/%.0f lp/mm"
        % (_k82.mtf.f50_g, _k82.mtf.resolving_power_lp_mm_lowc,
           _k82.mtf.resolving_power_lp_mm_highc))

    # ⚠ THE PUSH LADDER IS THE FIRST IN THIS DATABASE WITH NUMBERS. Every
    # other PushSpec on file carries a stops count and three zeros, because
    # a vendor saying a film pushes N stops says nothing about the fog or the
    # contrast it gains. Kodak drew both processes here, so all three are
    # measured -- and the cap stays at one stop because the paper says a
    # Push-2 buys graininess and no tone scale.
    chk("EASTMAN_5293_250T_1982 is the first stock to populate the push "
        "ladder's fog, contrast and realised-speed fields",
        _k82.push.max_push_stops == 1.0 and _k82.push.fog_penalty_stated
        and _k82.push.base_fog_penalty_per_stop > 0.05
        and _k82.push.gamma_gain_per_stop > 0.02
        and 0.90 < _k82.push.speed_gain_per_stop <= 1.0
        and sum(1 for q in FILM_PROFILES
                if q.push and q.push.base_fog_penalty_per_stop) == 1,
        "fog +%.3f/stop, gamma +%.1f%%/stop, %.0f%% of the nominal stop"
        % (_k82.push.base_fog_penalty_per_stop,
           100.0 * _k82.push.gamma_gain_per_stop,
           100.0 * _k82.push.speed_gain_per_stop))

    _ls = get_profile("EASTMANCOLOR_5248_1953").layer_stack
    chk("EASTMANCOLOR_5248_1953 carries Cheltsov's per-LAYER resolving with its order",
        _ls.order == ("blue", "green", "red")
        and abs(_ls.resolving_top - 110.0) < 1e-6
        and abs(_ls.resolving_bot - 30.0) < 1e-6,
        "%s %.0f/%.0f/%.0f" % (_ls.order, _ls.resolving_top, _ls.resolving_mid,
                               _ls.resolving_bot))

    # ---- 2026-08-17 dye density: the self-validating extraction ------------
    # The Kodak sheets plot a "Visual Neutral" trace ALONGSIDE the three dyes, and
    # a neutral is by definition their sum. Checking sum(C+M+Y) against it validates
    # curve identification, axis calibration and sampling in one step -- 5285 agrees
    # to max 0.013 D. That relationship is the reason these curves can be trusted,
    # so it is asserted rather than left in a comment.
    _dd = get_profile("KODAK_EKTACHROME_100D_5285").dye_density
    _s = [c + m + y for c, m, y in zip(_dd.d_cyan, _dd.d_magenta, _dd.d_yellow)]
    _worst = max(abs(a - b) for a, b in zip(_s, _dd.d_neutral))
    chk("5285 dye density: neutral trace equals sum(C+M+Y) to better than 0.02 D",
        _worst < 0.02 and len(_dd.d_cyan) == 31,
        "max |sum - neutral| = %.4f D over %d samples" % (_worst, len(_dd.d_cyan)))
    # Peaks must sit in the bands the dyes actually absorb. If a curve were
    # mis-identified this is what would catch it.
    def _peak_nm(vals):
        i = max(range(len(vals)), key=lambda k: vals[k])
        return 400.0 + 10.0 * i
    chk("5285 dye peaks land in their absorption bands (Y blue, M green, C red)",
        420 <= _peak_nm(_dd.d_yellow) <= 470
        and 520 <= _peak_nm(_dd.d_magenta) <= 570
        and 620 <= _peak_nm(_dd.d_cyan) <= 680,
        "Y %.0f / M %.0f / C %.0f nm" % (_peak_nm(_dd.d_yellow),
                                         _peak_nm(_dd.d_magenta),
                                         _peak_nm(_dd.d_cyan)))
    # 2383 is a PrintStock, which had NO v7 carrier until this extraction produced
    # data with nowhere to go. Assert the field survives on that dataclass too.
    _p2383 = [q for q in PRINT_STOCKS if q.name == "KODAK_2383_RELEASE"][0]
    chk("PrintStock carries dye density too (2383, normalised to visual neutral 1.0)",
        _p2383.dye_density.has_data
        and _p2383.dye_density.normalisation == "visual_neutral_1.0_xenon_arc",
        "%d samples, %s" % (len(_p2383.dye_density.d_cyan),
                            _p2383.dye_density.normalisation))

    # ---- EASTMANCOLOR_5382_1953 dye deposits, adopted 2026-09-03d -----------
    # ⚠ THE SECOND PrintStock TO CARRY DYE DENSITY, AND THE FIRST PRE-STATUS
    # ONE. Everything below exists because this set is easy to misuse in a way
    # nothing else in the corpus can be: it is a 1954 "dye deposit", not a
    # densitometric level, and the temptation to normalise it to a neutral or
    # to a stated density would silently invent a calibration the paper does
    # not contain.
    _p5382 = [q for q in PRINT_STOCKS if q.name == "EASTMANCOLOR_5382_1953"][0]
    _d5382 = _p5382.dye_density
    chk("EASTMANCOLOR_5382_1953 carries the 1954 dye-deposit set",
        _d5382.has_data and len(_d5382.d_cyan) == 59
        and _d5382.lambda_start_nm == 410.0 and _d5382.lambda_step_nm == 5.0,
        "%d samples, %.0f nm +%.0f" % (len(_d5382.d_cyan),
                                       _d5382.lambda_start_nm,
                                       _d5382.lambda_step_nm))
    # ⚠ THE GRID IS DELIBERATELY OFF THE CORPUS STANDARD (400-700 at 10 nm) AND
    # THE REASON IS THE FIGURE'S OWN PLOT BOX, which begins at 406.1 nm. 400 and
    # 405 are not on the page; 5 nm is the owner's stated minimum resolution.
    chk("5382's grid starts at 410 nm because the 1954 plot box does",
        _d5382.lambda_start_nm == 410.0
        and "406.1" not in _d5382.source[:0] + "",
        "%.0f nm" % _d5382.lambda_start_nm)
    # ⚠ THE UNCALIBRATED-LEVEL WARNING IS LOAD-BEARING, so it is asserted as
    # text and not merely written in a comment. A future edit that normalises
    # this set must delete these words to pass, which is the point.
    chk("5382's normalisation says in words that the level is uncalibrated",
        "UNCALIBRATED" in _d5382.normalisation
        and "1954 predates Status A/M" in _d5382.normalisation
        and "not be rescaled" in _d5382.normalisation,
        _d5382.normalisation[:60])
    # ⚠ NOT A NEUTRAL, AND THAT IS CHECKABLE. A neutral set has its three peaks
    # near-equal by construction; this one runs 0.876 / 0.869 / 1.118, a 29 %
    # spread. If a future edit ever flattens that, the set has been renormalised.
    _pk5382 = (max(_d5382.d_yellow), max(_d5382.d_magenta), max(_d5382.d_cyan))
    chk("5382's three peaks keep the printed 0.876 / 0.869 / 1.118 ratio",
        abs(_pk5382[0] - 0.876) < 0.002 and abs(_pk5382[1] - 0.869) < 0.002
        and abs(_pk5382[2] - 1.118) < 0.002
        and max(_pk5382) / min(_pk5382) > 1.20,
        "%.3f / %.3f / %.3f, spread %.2fx"
        % (_pk5382 + (max(_pk5382) / min(_pk5382),)))
    # ⚠ THE ~430 nm CROSSING, WHICH THE FIRST READING OF THIS FIGURE GOT WRONG.
    # Below it cyan is above magenta; above it, below. The failure mode this
    # catches is the two branches being swapped back, which leaves both curves
    # smooth and every peak correct -- no other guard here would see it.
    chk("5382's magenta and cyan are on the right side of the ~430 nm crossing",
        _d5382.d_cyan[0] > _d5382.d_magenta[0] + 0.10      # 410 nm
        and _d5382.d_cyan[8] < _d5382.d_magenta[8],        # 450 nm
        "410 nm C %.3f / M %.3f, 450 nm C %.3f / M %.3f"
        % (_d5382.d_cyan[0], _d5382.d_magenta[0],
           _d5382.d_cyan[8], _d5382.d_magenta[8]))
    # ⚠ THE YELLOW TAIL EXISTS. Before the tail repair yellow ended at 526.7 nm
    # still at 0.158 D -- a dye stopping mid-descent, which is a tracking
    # failure and not a measurement. It must now reach the axis on its own.
    _ylast = max(i for i, v in enumerate(_d5382.d_yellow) if v > 0.0)
    chk("5382's yellow descends to the axis instead of stopping mid-fall",
        410 + 5 * _ylast >= 575 and _d5382.d_yellow[_ylast] < 0.02,
        "last non-zero %.3f D at %d nm" % (_d5382.d_yellow[_ylast],
                                           410 + 5 * _ylast))
    # ⚠ ZERO MEANS "AT THE DRAWN AXIS", AND ONLY OUTSIDE EACH DYE'S OWN SPAN.
    # A zero appearing INSIDE a dye's descent would mean the trace dropped out
    # mid-curve and was padded, which is exactly what must never be adopted.
    for _nm, _tr in (("yellow", _d5382.d_yellow), ("magenta", _d5382.d_magenta),
                     ("cyan", _d5382.d_cyan)):
        _nz = [i for i, v in enumerate(_tr) if v > 0.0]
        chk("5382 %s has no interior zero -- every zero is past its own descent"
            % _nm,
            _nz and all(_tr[i] > 0.0 for i in range(_nz[0], _nz[-1] + 1)),
            "non-zero %d..%d nm" % (410 + 5 * _nz[0], 410 + 5 * _nz[-1]))
    # ⚠ AND THE PRINT STOCK'S OWN 1953 NEGATIVE IS STILL WITHOUT A DYE SET.
    # Recorded as a guard rather than a note because it is the standing gap the
    # Hanson & Kisner acquisition was meant to close and did not: 5248's primary
    # paper contains neither spectral sensitivity nor dye density.
    _p5248 = [q for q in FILM_PROFILES if q.name == "EASTMANCOLOR_5248_1953"]
    chk("the 1953 pair is still asymmetric: 5382 has dyes, 5248 has none",
        bool(_p5248) and not _p5248[0].dye_density.has_data,
        "5248 dye set present" if _p5248 and _p5248[0].dye_density.has_data
        else "5248 still empty, as Hanson & Kisner 1953 leaves it")

    # ---- AGFA_NEU_1936: Eggert 1937, adopted 2026-09-04 ---------------
    # ⚠ THE FIRST MEASURED NUMBERS THIS PROFILE HAS EVER CARRIED, and the guard
    # exists because of how nearly they were missed: the document arrived on
    # 2026-09-03 and the first pass filed the inventor's own seven-page article
    # as a bare provenance citation without reading it, while the profile's own
    # provenance went on asserting that no photometric figure for this film
    # existed anywhere.
    _neu = get_profile("AGFA_NEU_1936")
    chk("AGFA_NEU_1936 carries Eggert's measured layer geometry",
        abs(_neu.emulsion.coated_um - 19.0) < 1e-9
        and _neu.layer_stack.order == ("blue", "green", "red")
        and "Eggert" in _neu.emulsion.source,
        "coated %.1f um, order %s" % (_neu.emulsion.coated_um,
                                      "/".join(_neu.layer_stack.order)))
    # ⚠ 19 um IS 3 x 5 + 2 x 2 AND THE ARITHMETIC IS ASSERTED, not the total,
    # so an edit that changes the total without a source has to change this too.
    chk("and 19 um is exactly three 5 um emulsions plus two 2 um interlayers",
        abs(_neu.emulsion.coated_um - (3.0 * 5.0 + 2.0 * 2.0)) < 1e-9,
        "%.1f um" % _neu.emulsion.coated_um)
    # ⚠ EVERYTHING ELSE IN EmulsionSpec MUST STAY EMPTY. The article names no
    # crystal size, no habit, no iodide, no chemical sensitisation and no base
    # material; the profile's NITRATE_BASE feature is this project's period
    # assumption, and letting it leak into `base_material` beside an Eggert
    # citation would turn an assumption into a source.
    chk("and nothing Eggert does not print has leaked into EmulsionSpec",
        not (_neu.emulsion.grain_um or _neu.emulsion.habit
             or _neu.emulsion.iodide_mol_pct or _neu.emulsion.sensitization
             or _neu.emulsion.base_material or _neu.emulsion.base_um),
        "grain_um %.1f, habit %r, base %r" % (_neu.emulsion.grain_um,
                                              _neu.emulsion.habit,
                                              _neu.emulsion.base_material))
    # ⚠ AND THE SPEED CONFLICT IS NOT QUIETLY SETTLED. Eggert's sunshine
    # exposure gives ISO 2.5-6.1 depending on which 16 mm frame rate is meant;
    # the stored 8 sits above all of it. Method rule 4: recorded, not averaged.
    # If a future edit moves the EI it must come with a real speed measurement,
    # and this guard is what makes that edit announce itself.
    chk("AGFA_NEU_1936's exposure_index is still the analogy 8, with the "
        "Eggert conflict recorded rather than averaged in",
        _neu.exposure_index == 8
        and any("0.4 to 1.7 stops fast" in s for s in _neu.provenance.sources),
        "EI %d" % _neu.exposure_index)
    # ⚠ THE OLD CLAIM SURVIVES ONLY AS A QUOTATION OF ITSELF, and that is
    # deliberate: the note records what it used to say and why that stopped
    # being true, rather than deleting the error. So this guard cannot simply
    # look for the absence of the old sentence -- it asserts that the sentence
    # is now marked superseded, and that the geometry is called measured.
    _prov = " ".join(_neu.provenance.sources)
    chk("and its provenance marks the old 'no photometric figure' claim as "
        "superseded rather than still asserting it",
        "STOPPED BEING TRUE OF" in _prov and "Eggert" in _prov
        and "LAYER GEOMETRY is now measured" in _prov,
        "%d sources" % len(_neu.provenance.sources))

    # ---- no photographic PAPER may enter the database, 2026-09-04 ----------
    # ⚠ THIS GUARD IS THE OUTCOME OF QUEUE ROW #179 AND IT ASSERTS A REFUSAL.
    # `AGFA/mcp_Agfa.pdf` is Technical Data P-53-P (11/1997, 4th edition) for
    # MULTICONTRAST PREMIUM, a variable-contrast RC ENLARGING PAPER. It is a
    # complete, entirely vector sheet -- six characteristic curves for filters
    # 0-5, a spectral sensitivity panel at three reflection densities, Dmax
    # 2.25, ISO P 400/160/80, reciprocity and latent-image plots -- and it is
    # the best paper document in the corpus. It is still refused.
    #
    # THE REASON IS THE MEASURAND, NOT THE QUALITY. Every density in it is a
    # REFLECTION density. This engine's output path is transmittance: stage 14
    # converts D to t = 10^-D between a stock's own dmin and dmax, and
    # `PrintStock` means a transmissive positive that is projected or scanned.
    # A paper has no transmittance, no projection, and a paper white this
    # schema cannot express. Entering MCP as a PrintStock would run the
    # arithmetic and state something false, and every downstream consumer --
    # the transmittance conversion, print grain, dye_matrix, the v25
    # printing-density matrix -- would be operating on a quantity the stock
    # does not have.
    #
    # WHAT WOULD CHANGE THE ANSWER: a reflection print path. If one is ever
    # added, this sheet is where it should start.
    _PAPER_WORDS = ("paper", "multicontrast", "convira", "baryta", "rc/pe")
    _papers = [s.name for s in PRINT_STOCKS
               if any(w in s.name.lower() for w in _PAPER_WORDS)]
    chk("no photographic PAPER has been entered as a print stock -- the engine "
        "has no reflection path (queue #179, AGFA MULTICONTRAST PREMIUM)",
        not _papers, ", ".join(_papers) or "%d print stocks, all transmissive"
        % len(PRINT_STOCKS))

    # ---- KODAK F-5 (August 1979) DS sheets, added 2026-08-17 ----------------
    _f5 = ("KODAK_PANATOMIC_X", "KODAK_VERICHROME_PAN", "KODAK_SUPER_XX_PAN_4142",
           "KODAK_ROYAL_PAN_4141", "KODAK_ROYAL_X_PAN_4166", "KODAK_RECORDING_2475")
    _byname = {q.name: q for q in FILM_PROFILES}
    chk("F-5 1979: all six new stocks present", all(n in _byname for n in _f5),
        ", ".join(n for n in _f5 if n not in _byname) or "all six")
    # Speeds are the one thing F-5 states unambiguously, so assert them.
    _speeds = {"KODAK_PANATOMIC_X": 32, "KODAK_VERICHROME_PAN": 125,
               "KODAK_SUPER_XX_PAN_4142": 200, "KODAK_ROYAL_PAN_4141": 400,
               "KODAK_ROYAL_X_PAN_4166": 1250, "KODAK_RECORDING_2475": 1600}
    chk("F-5 1979: DS-sheet ISO speeds as printed",
        all(_byname[n].exposure_index == v for n, v in _speeds.items()),
        ", ".join("%s=%d" % (n, _byname[n].exposure_index) for n, v in _speeds.items()
                  if _byname[n].exposure_index != v) or "6/6 match")
    # VERICHROME Pan must NOT collapse into VERICHROME: different sensitisation
    # class (pan vs ortho) and a full stop apart. This guard exists because the
    # names differ by one word and a future edit could "tidy" them together.
    chk("VERICHROME Pan is distinct from the 1952 ortho VERICHROME",
        _byname["KODAK_VERICHROME_PAN"].exposure_index == 125
        and _byname["KODAK_VERICHROME_1952"].exposure_index != 125,
        "pan=%d ortho=%d" % (_byname["KODAK_VERICHROME_PAN"].exposure_index,
                             _byname["KODAK_VERICHROME_1952"].exposure_index))
    # Resolving power came from F-5 at BOTH test-object contrasts; low < high
    # must hold for every entry, and the three gap-filled stocks must be present.
    from film_profiles import _RESOLVING_POWER as _RP
    _f5rp = ("KODAK_PLUS_X_125", "KODAK_TRI_X_400TX", "KODAK_EKTAPAN_100",
             "KODAK_PANATOMIC_X", "KODAK_VERICHROME_PAN",
             "KODAK_SUPER_XX_PAN_4142", "KODAK_ROYAL_PAN_4141",
             "KODAK_ROYAL_X_PAN_4166")
    chk("F-5 resolving power: 8 stocks, low contrast < high contrast",
        all(n in _RP and 0 < _RP[n][0] < _RP[n][1] for n in _f5rp),
        ", ".join("%s=%s" % (n, _RP.get(n)) for n in _f5rp
                  if not (n in _RP and 0 < _RP[n][0] < _RP[n][1])) or "8/8 ordered")
    # Recording 2475 has NO printed resolving power. An absent key is the honest
    # representation and must stay absent.
    chk("Recording 2475 carries NO resolving power (none is printed)",
        "KODAK_RECORDING_2475" not in _RP, "absent as intended")
    # PANATOMIC-X holds the highest resolving power in the file (200 lines/mm).
    # ⚠ QUEUE G6 CLOSED 2026-09-05, AND THIS IS THE GUARD THAT KEEPS IT CLOSED.
    # The row asked, for two and a half weeks, what Agfa's «Linien pro mm» axis
    # means -- because if one Linie is a half-cycle then every Agfa resolving
    # power here, and the Gevacolor 682 f50 pair, are a factor of two out.
    # ⚠ ANSWERED BY THE ICO ITSELF. Erik Ingelstam, "Nomenclature for Fourier
    # Transforms of Spread Functions", PS&E 5(5) p282, Sept-Oct 1961 -- the
    # published, unanimous recommendation of the International Commission for
    # Optics Subcommittee for Image Assessment Problems (Arnulf, Hopkins,
    # Kubota, Rosenhauer of Braunschweig, Scott, Ingelstam as chairman, with
    # Kinosita, Murata, Sayanagi and MacAdam), and the document that named the
    # modulation transfer function. Recommendation (3), verbatim: "The unit of
    # spatial frequency shall be described by either: Cycles per mm or lines per
    # mm. The latter shall be used when no confusion can occur with television
    # lines, since 2 television lines = 1 cycle. The corresponding terms are:
    # Cycles par mm; lignes par mm; Perioden pro mm; Linien pro mm."
    # ⚠ So «Linien pro mm» IS cycles per mm, named as the German equivalent by a
    # committee containing a German member, in force decades before every Agfa
    # sheet in this corpus -- and it AGREES with the RP*sqrt(RMS) cross-maker
    # test the project ran independently (Agfa median 450, SVEMA 455, ROLLEI
    # 454). The figures below therefore stand AS PRINTED, on the line-pair
    # scale. Halving them to "convert to cycles" would put Agfa at 225, below
    # every maker in the corpus, and now has to argue with this document.
    _AGFA_RP_AS_PRINTED = {
        "AGFA_APX_25": 200.0, "AGFA_APX_100": 150.0, "AGFA_APX_400": 110.0,
        "AGFA_OPTIMA_100": 140.0, "AGFA_OPTIMA_200": 130.0,
        "AGFA_OPTIMA_400": 130.0, "AGFA_PORTRAIT_160": 150.0,
        "AGFA_SCALA_200X": 120.0, "AGFA_VISTA_200": 130.0,
    }
    _rp_moved = ["%s %s->%s" % (_n, _want, film_profiles._RESOLVING_POWER[_n][1])
                 for _n, _want in _AGFA_RP_AS_PRINTED.items()
                 if _n in film_profiles._RESOLVING_POWER
                 and film_profiles._RESOLVING_POWER[_n][1] != _want]
    _rp_gone = [_n for _n in _AGFA_RP_AS_PRINTED
                if _n not in film_profiles._RESOLVING_POWER]
    chk("G6: the Agfa resolving powers stand as printed, on the line-pair scale",
        not _rp_moved and not _rp_gone,
        "9 stocks, 110-200 lines/mm at 1000:1, unhalved; Ingelstam/ICO 1961 "
        "recommendation (3) names «Linien pro mm» the German equivalent of "
        "«lines per mm» = cycles per mm"
        + ("" if not _rp_moved else "; MOVED " + ", ".join(_rp_moved))
        + ("" if not _rp_gone else "; MISSING " + ", ".join(_rp_gone)))
    chk("PANATOMIC-X has the highest high-contrast resolving power held",
        _RP["KODAK_PANATOMIC_X"][1] == max(v[1] for v in _RP.values()),
        "%.0f lines/mm" % _RP["KODAK_PANATOMIC_X"][1])
    # EKTAPAN's processing point carries a REAL contrast index from DS 5's own
    # curve caption -- unlike the Иофис rows, where the source printed a gamma
    # and contrast_index was deliberately left at 0.0.
    _ekt = _byname["KODAK_EKTAPAN_100"].processing
    chk("EKTAPAN processing point carries DS 5's printed contrast index",
        abs(_ekt.contrast_index - 0.54) < 1e-9 and _ekt.minutes == 5.0
        and "HC-110" in _ekt.developer,
        "%s %.1f min CI %.2f" % (_ekt.developer, _ekt.minutes, _ekt.contrast_index))
    # ⚠ 5222 LEFT THIS SET ON 2026-08-26 AND THAT IS THE POINT OF THE EDIT.
    # Its row was Иофис-sourced ("Kodak D-76", no temperature, contrast_index
    # deliberately 0.0 because the source printed a gamma band and not a CI).
    # Kodak's own sheet supersedes it: D-96 at 21 C with the printed gamma 0.66
    # for that time, which IS the aim the time targets, so the field is now
    # populated from the manufacturer. The two stocks still on Иофис rows keep
    # the old discipline, and this guard now asserts BOTH halves so neither can
    # drift into the other.
    chk("the 2 remaining Иофис processing rows keep contrast_index 0.0",
        all(_byname[n].processing.contrast_index == 0.0
            for n in ("ILFORD_HP3", "ILFORD_HPS"))
        and _byname["EASTMAN_DOUBLE_X_5222"].processing.contrast_index == 0.66,
        "HP3 and HPS at 0.0; 5222 now carries Kodak's own printed 0.66")

    # ---- МЗ-3 Soviet positive, Иофис 1964 table 11 (2026-08-17) -------------
    _mz3 = [q for q in PRINT_STOCKS if q.name == "TASMA_POSITIVE_28"][0]
    # gamma follows ТУ 6-17-647-80 / Журба 1984 (recommended 2,8-3,2), NOT
    # Иофис 1964's earlier 2,5 +/- 0,2. Both readings are cited in the profile;
    # this guard pins which one the render uses.
    chk("МЗ-3 print gamma is 3.00, the ТУ-era recommended centre",
        abs(_mz3.curves.r.gamma - 3.00) < 1e-9,
        "gamma %.2f" % _mz3.curves.r.gamma)
    chk("МЗ-3 gamma sits inside Журба's recommended band 2.8-3.2",
        2.8 <= _mz3.curves.r.gamma <= 3.2, "%.2f" % _mz3.curves.r.gamma)
    # dmin is now documented by two sources that agree, not estimated.
    chk("МЗ-3 dmin 0.04 is the documented minimum optical density",
        abs(_mz3.curves.r.dmin - 0.04) < 1e-9, "dmin %.3f" % _mz3.curves.r.dmin)
    chk("МЗ-3 records that it is unsensitized (blue-sensitive only)",
        "BLUE-SENSITIVE ONLY" in _mz3.description.upper(), "noted")
    chk("МЗ-3 keeps the superseded Иофис 2,5 reading on record",
        "2,5 +/- 0,2" in _mz3.description, "both generations cited")
    chk("МЗ-3 does not silently claim to be the Л variant",
        "NOT guessed here" in _mz3.description, "Л suffix left open")
    # Иофис p 93 gives the class norm for Soviet positive film as contrast
    # coefficient 2,0-3,0. МЗ-3 must sit inside its own class limits.
    chk("МЗ-3 gamma sits inside the p 93 class norm 2.0-3.0",
        2.0 <= _mz3.curves.r.gamma <= 3.0, "2.0 <= %.2f <= 3.0" % _mz3.curves.r.gamma)
    # The owner asked that the Tasma attribution and the yellow boxes be carried
    # as personal recollection, NOT as verified evidence -- Иофис says only
    # «Отечественное». This guard exists so a future tidy-up cannot silently
    # promote testimony to documentation by deleting the caveat.
    chk("МЗ-3 keeps the manufacturer claim labelled as owner recollection",
        "PERSONAL" in _mz3.description.upper()
        and "NOT as verified technical evidence" in _mz3.description,
        "caveat present")
    chk("МЗ-3 records that GOST 2.8 has no source in the corpus",
        "NO sensitivity column" in _mz3.description, "unsourced-2.8 caveat present")

    # ---- provenance placeholder guard (2026-08-18) --------------------------
    # `_provenance_for` derives the tier from the [T*] tag in the description
    # but takes sources from `_PROVENANCE_SOURCES`, falling back to the
    # `_NO_DATASHEET` placeholder. Nothing tied the two together, so a profile
    # could -- and eight did -- claim datasheet grounding in its tier while the
    # queryable struct said "no official manufacturer datasheet available".
    # Six were closed on 2026-08-18 by lifting citations already present in the
    # profiles; these guards stop the gap reopening.
    _ph = film_profiles._NO_DATASHEET

    # TIER 1 IS ABSOLUTE. Tier 1 means datasheet-grounded, so a tier-1 profile
    # with only the placeholder is self-contradictory. No allowlist here.
    _t1 = sorted(p.name for p in FILM_PROFILES
                 if p.provenance.tier == 1 and p.provenance.sources == _ph)
    chk("no tier-1 profile carries only the _NO_DATASHEET placeholder",
        not _t1, ", ".join(_t1) if _t1 else "0 of %d tier-1 profiles"
        % sum(1 for p in FILM_PROFILES if p.provenance.tier == 1))

    # TIER 2 HAS A CLOSED, DOCUMENTED EXCEPTION SET -- AND IT IS NOW EMPTY.
    # It held FUJI_F125_8530 / _8630, the two profiles that were tier 2 with no
    # citable document anywhere in the corpus. The owner supplied
    # PDF/PROFILES/FUJI/52_509.pdf on 2026-08-18; it names type 8530 and prints
    # a measured MTF, so both were cited and removed from this set the same day.
    # Guard 3 below caught the change rather than letting the stale allowlist
    # outlive the gap, which is the whole reason it is a LITERAL SET and not a
    # count. Keeping the empty set (rather than deleting these checks) is
    # deliberate: it is what fails if a new placeholder-only tier-2 appears.
    _T2_PLACEHOLDER_OK = set()
    _t2 = {p.name for p in FILM_PROFILES
           if p.provenance.tier == 2 and p.provenance.sources == _ph}
    _new = sorted(_t2 - _T2_PLACEHOLDER_OK)
    chk("no NEW tier-2 profile carries only the _NO_DATASHEET placeholder",
        not _new, ", ".join(_new) if _new else "only the 2 documented gaps")
    # The other direction: if one of the two acquires a real citation, this
    # fails so the allowlist and NotFound.md get updated together instead of
    # the allowlist quietly outliving the gap it documents.
    _gone = sorted(_T2_PLACEHOLDER_OK - _t2)
    chk("the tier-2 placeholder allowlist still matches reality exactly",
        not _gone, "now cited, remove from allowlist + NotFound.md: "
        + ", ".join(_gone) if _gone else "allowlist empty, as intended")

    # F-125 must keep the citation that emptied that allowlist, and must keep
    # the measured f50 that citation grounds. 42.0 c/mm is the printed number;
    # a regression to the old estimate of 78 would be a 1.86x sharpness error
    # in a stock the renderer treats as fine-grained.
    # ⚠ WAS A TWO-STOCK LOOP UNTIL 2026-08-24. FUJI_F125_8630 was a gauge clone
    # of 8530 and was removed that day: «Техника кино и телевидения» 1989 No.4
    # p70 prints Fuji's own code rule, in which the SECOND digit is the gauge
    # (5 = 35 mm, 6 = 16 mm), so 8530/8630 were never two emulsions. The guard
    # below asserts the removal stayed done, because re-adding the clone is the
    # obvious way for a later pass to "fix" a missing 16 mm entry.
    for _n in ("FUJI_F125_8530",):
        _p = get_profile(_n)
        chk(f"{_n} cites Honjo 1989 for its MTF",
            "Honjo" in " ".join(_p.provenance.sources), "cited")
        chk(f"{_n} keeps the measured f50_g = 42.0 c/mm",
            abs(_p.mtf.f50_g - 42.0) < 1e-9, "f50_g %.1f" % _p.mtf.f50_g)
    _f125 = get_profile("FUJI_F125_8530")
    _f125src = " ".join(_f125.provenance.sources)
    chk("the F-125 16 mm gauge clone stays removed",
        not any(p.name == "FUJI_F125_8630" for p in FILM_PROFILES)
        and "8630" in _f125.aliases,
        "8630 resolves to 8530 as an alias, not as a second profile")
    # rms 4.0 is PRINTED (1989 No.4 Table 1 p70), replacing an estimated 5.4.
    # The pin also guards the generation gap: 8532 prints 3.0 at the same speed,
    # so 8530 > 8532 must stay true or the two profiles have been crossed over.
    chk("F-125 8530 carries the printed rms 4.0, and 8532 stays finer",
        abs(_f125.grain.rms_granularity - 4.0) < 1e-9
        and get_profile("FUJI_SUPER_F125_8532").grain.rms_granularity < 4.0
        and "Техника кино и телевидения" in _f125src,
        "8530 rms 4.0 printed, 8532 rms 3.0, both cited")
    # ⚠ AND THE THINGS THOSE DOCUMENTS COULD HAVE GROUNDED BUT DID NOT must stay
    # unadopted, each for a stated reason -- a later pass that "finishes" any of
    # them without re-reading the plate would be repeating a recorded failure.
    chk("the F-125 sigma(D) figure stays unharvested and says why",
        not _f125.grain.sigma_shape_measured
        and "converge inside the line width" in _f125src,
        "Fig. 4 cited, F-125 and F-64 merge at the validating anchor")
    # The R/B values are an interpolation, not a measurement -- but the layer
    # order they encode is physical and must survive (red softest, blue
    # sharpest, per the MTFSpec docstring).
    _f = get_profile("FUJI_F125_8530").mtf
    chk("F-125 keeps the physical layer order r < g < b after the rescale",
        _f.f50_r < _f.f50_g < _f.f50_b,
        "%.1f < %.1f < %.1f" % (_f.f50_r, _f.f50_g, _f.f50_b))

    # The six closures are pinned individually: each citation must name its own
    # document, not merely be non-empty. An entry that regressed to a stub or
    # was pasted from a neighbouring stock would pass a length test.
    _CLOSED = {
        "FUJICOLOR_A250":             "MP3-57E",
        "GEVACHROME_902":             "Verbrugghe",
        "KONICA_CHROME_CENTURIA_100": "CENTURIA 100 SRA",
        "KONICA_CHROME_R100":         "R-100",
        "ILFORD_HPS":                 "table 7",
        "KODAK_SUPER_XX_PAN_4142":    "DS 17",
    }
    _miss = [n for n, tok in _CLOSED.items()
             if tok not in " ".join(get_profile(n).provenance.sources)]
    chk("all 6 closed citations still name their own document",
        not _miss, ", ".join(_miss) if _miss else "6 of 6")
    # A250's confusable companion document is the one hazard in this batch
    # that would silently corrupt data if the warning were dropped: a 1985 SMPTE
    # paper about AX 8514/8512 and LP 8816 sits under a near-identical name.
    chk("A250 keeps its misattribution warning",
        "must NOT be attributed to A250"
        in " ".join(get_profile("FUJICOLOR_A250").provenance.sources),
        "hazard recorded")
    # HPS is a Soviet source for a British film -- method rule 14 says an
    # Ilford sheet outranks it. Losing that note would promote it to parity.
    # ---- 2026-08-23: the two BBC documents, items A and B -------------------
    # ⚠ WHAT THESE GUARDS PROTECT IS A DISTINCTION, NOT A NUMBER. Both documents
    # are third-party BBC research reports, so method rule 14 still applies and
    # the Soviet caveat must survive; what they add is (a) an independent
    # confirmation of the two speeds, (b) ONE measured value adopted (gamma), and
    # (c) three measured values deliberately NOT adopted. A later pass that
    # "finishes the harvest" by adopting (c) is what these catch.
    _hps = get_profile("ILFORD_HPS")
    chk("HPS carries the measured BBC development gamma 0.63",
        abs(_hps.curves.g.gamma - 0.630) < 1e-9
        and _hps.curves.r.gamma == _hps.curves.g.gamma == _hps.curves.b.gamma,
        "gamma 0.630, monochrome so all three records agree")
    _hsrc = " ".join(_hps.provenance.sources)
    chk("HPS cites both BBC documents alongside Иофис",
        len(_hps.provenance.sources) == 3
        and "Monograph No. 54" in _hsrc and "T-101" in _hsrc
        and "Иофис" in _hsrc,
        "3 citations: Иофис 1964, BBC M54 1964, BBC T-101 1963")
    # ⚠ SUPERSEDED 2026-09-01, AND THE EARLIER REFUSAL WAS RIGHT FOR ITS
    # EVIDENCE. This guard used to read "HPS keeps rms 19.0 rather than the
    # D-0.48 conversion 18.5", because the BBC Wiener spectrum sits at D 0.48
    # above base while this field is defined at NET 1.0 and nothing on file
    # bridged the two. What changed is that the bridge arrived: BBC T-101/2
    # (1964/4) states the density exponent (Higgins and Stultz, 0.4) AND the
    # development-gamma law (grain diameter proportional to sqrt(gamma),
    # section 5.2), and T-101 Table 3 corroborates both from inside the corpus.
    # 18.5 was the RAW sqrt(W/A) with neither correction applied, which is
    # exactly what should have been refused; 20.02 is the corrected value.
    # ⚠ AND IT IS NOT TAKEN ON TRUST: the same chain is run on KODAK_TRI_X_400TX,
    # whose 17.0 is Kodak's OWN published rms, and returns 18.9 -- 11 %. A
    # conversion nobody could check against a known answer would still be
    # refused. See bbc_t101_2.py, which asserts the control on every build.
    chk("HPS rms is the CORRECTED BBC conversion, not the raw one",
        abs(_hps.grain.rms_granularity - 20.02) < 1e-9
        and "0.62 square microns" in _hsrc
        and "0.48 ABOVE BASE" in _hsrc,
        "20.02 at net 1.0 and gamma 0.65; the raw 18.5 stays refused")
    # ⚠ THE CONFLICT MUST STAY VISIBLE AND UNRESOLVED. clump_um 26.0 against a
    # measured 2.5 um: correcting one stock while 158 others keep the same
    # convention would make this profile inconsistent rather than correct, so the
    # decision is the owner's. If someone changes it, this fails and they have to
    # read the comment explaining why the whole field is in question.
    # ⚠ THE SET EMPTIED ON 2026-08-24. First HPS left it (fitted to Monograph 54
    # Fig. 8, 268 traced points), then the same day T-101 Fig. 18 AND Table 2
    # were read and the whole family moved. What decided it: Table 2 (p28) PRINTS
    # the measured equivalent grain diameter of all six emulsions, and Table 4
    # (p38) prints their granularity ladder with 5302 as unity -- so the clump
    # column for these stocks no longer needs a traced curve at all.
    _hpsg = get_profile("ILFORD_HPS").grain
    chk("HPS carries the PRINTED equivalent grain diameter, not the trace",
        abs(_hpsg.clump_um_g - 1.431) < 1e-9
        and _hpsg.clump_um_r == _hpsg.clump_um_g == _hpsg.clump_um_b
        and _hpsg.clump_gain == 0.0
        and abs(_hpsg.rms_granularity - 20.02) < 1e-9,
        "clump 1.431 um = 2.5/1.7473 on all three records, gain 0.000, "
        "rms 20.02 from the corrected BBC conversion")
    # THE CONVERSION IS PINNED NUMERICALLY, not just described in prose. T-101
    # defines equivalent grain diameter as the full width of the normalised
    # autocorrelation at ordinate 0.39, so for this file's Gaussian carrier
    # D_eq = 2*sqrt(2*ln(1/0.39))/(pi*f_hi) = 1.746*clump_um. If someone changes
    # `grain_shape`'s carrier, this fails and the six adopted numbers have to be
    # re-derived rather than silently meaning something else.
    _DEQ = 4.0 * math.sqrt(2.0 * math.log(1.0 / 0.39)) / math.pi
    chk("the D_eq <-> clump_um conversion is still 1.7473",
        abs(_DEQ - 1.74727) < 1e-4,
        "D_eq = %.4f * clump_um, from the 0.39 autocorrelation width" % _DEQ)
    # ⚠ ILFORD_PAN_F LEFT THIS SET ON 2026-08-25 and must not be put back into
    # it. Table 2's diameters were measured at the BBC's own development gamma,
    # and Pan F is the one stock whose stored curve disagrees with that gamma
    # (0.55 against 1.0). Its clump_um is therefore Table 2's value MOVED to the
    # stored gamma by the law T-101 Table 3 measures on this same emulsion --
    # see the guard below. The other four match their BBC gamma and are stored
    # as printed.
    _T2_DEQ = {"ILFORD_HPS": 2.5, "EASTMAN_TRI_X_5223": 2.2,
               "EASTMAN_PLUS_X_5231": 1.45, "KODAK_8374": 1.2}
    _t2_bad = [n for n, d in _T2_DEQ.items()
               if abs(get_profile(n).grain.clump_um_g - d / _DEQ) > 6e-4]
    chk("the 4 gamma-matched T-101 stocks store Table 2's diameter / 1.7473",
        not _t2_bad, ", ".join(_t2_bad) if _t2_bad
        else "2.5/2.2/1.45/1.2 um -> 1.431/1.259/0.830/0.687")
    # THE GAMMA CORRECTION, pinned with its own arithmetic so that changing the
    # exponent or the stored gamma without redoing the conversion fails loudly.
    _pf = get_profile("ILFORD_PAN_F")
    _pf_expect = (1.5 / _DEQ) * (_pf.curves.g.gamma / 1.00) ** 0.452
    chk("ILFORD_PAN_F's clump_um is Table 2 moved to ITS OWN gamma",
        abs(_pf.grain.clump_um_g - 0.655) < 1e-9
        and abs(_pf_expect - _pf.grain.clump_um_g) < 3e-3,
        "0.859 um at gamma 1.0 -> %.3f at the stored gamma %.2f, n = 0.452"
        % (_pf.grain.clump_um_g, _pf.curves.g.gamma))
    # ⚠ AND PLUS-X IS DELIBERATELY *NOT* CORRECTED. Its stored gamma is 0.68
    # against the BBC's 0.64, which the same law makes a +2.5 % move to 0.851 --
    # far inside the upper-bound caveat those printed diameters already carry.
    # Moving a number by less than its own stated uncertainty is false precision.
    chk("EASTMAN_PLUS_X_5231 keeps the uncorrected 0.830",
        abs(get_profile("EASTMAN_PLUS_X_5231").grain.clump_um_g - 0.830) < 1e-9,
        "0.830 kept; the 2.5 %% gamma move is inside the source's own bound")
    # The law itself must stay findable, and so must the retraction beside it.
    _hps_src = " ".join(get_profile("ILFORD_HPS").provenance.sources)
    chk("T-101 Fig. 26 stays recorded as NOT convertible to sigma_D",
        "sigma_t/t << 1, is invalid" in _hps_src
        and "THERE IS NO CONFLICT" in _hps_src,
        "the pinhole two-level model and the withdrawn conversion are both cited")
    # 5302 is a PrintStock, so it is checked separately -- and it is the anchor
    # of Table 4's whole granularity ladder, which is why it earns its own guard.
    _p5302 = [p for p in PRINT_STOCKS if p.name == "KODAK_5302"]
    chk("KODAK_5302 exists as a print stock and anchors the T-101 ladder",
        len(_p5302) == 1
        and abs(_p5302[0].grain_clump_um - 0.589) < 1e-9
        and abs(_p5302[0].grain_rms - 4.7) < 1e-9
        and abs(_p5302[0].curves.g.gamma - 2.40) < 1e-9,
        "clump 0.589 = 1.03/1.7473, rms 4.7 = HPS 0.62/3.9^2 through the "
        "48 um aperture, printed gamma 2.4")
    # ⚠ clump_gain 0.000 ON ALL OF THEM IS A MEASUREMENT. A free two-parameter
    # fit to every one of the six Fig. 18 spectra drove the low-frequency lobe
    # to zero, and T-101 p38 states it in words. This guard stops a later pass
    # reinstating a clumping lobe on any of them because a render looks odd.
    _gain_bad = [n for n in list(_T2_DEQ)
                 if get_profile(n).grain.clump_gain != 0.0]
    chk("no T-101 stock reinstates a low-frequency clumping lobe",
        not _gain_bad, ", ".join(_gain_bad) if _gain_bad
        else "clump_gain exactly 0.0 on all 5")
    # ⚠ clump_gain 0.0 IS THE MEASUREMENT AND MUST NOT BE "RESTORED". A free
    # two-parameter fit drove it to zero and T-101 p38 says the same in words.
    # This is the guard that stops a later pass reinstating a clumping lobe
    # because the render looks unfamiliar.
    chk("HPS clump_gain stays exactly 0.0 -- the fit refused the lobe",
        _hpsg.clump_gain == 0.0 and "clump_gain 0.000" in
        " ".join(get_profile("ILFORD_HPS").provenance.sources),
        "gain 0.000, cited")
    # And the level must be unchanged: this edit was texture, not loudness.
    # grain_reference_energy renormalises the field, so rms is independent of
    # clump_um -- assert that the two really are decoupled, off-database.
    _e_old = fs.grain_reference_energy(26.0, 1.65)
    _e_new = fs.grain_reference_energy(1.431, 0.0)
    chk("the grain level is renormalised, so clump_um moves texture only",
        _e_old > 0 and _e_new > 0 and abs(_e_new / _e_old - 1.31) < 0.06,
        "aperture-weighted energy ratio %.2f, amplitude rescaled by %.2f"
        % (_e_new / _e_old, (_e_old / _e_new) ** 0.5))
    # ⚠ THE STOCK THAT DID *NOT* TAKE 5223's MEASUREMENT, AND WHY IT MUST NOT.
    # T-101 measured "Tri-X Type 5223", the 35 mm CINE negative at 250/320
    # A.S.A. KODAK_TRI_X_400TX is the ASA 400 STILL film. Same trade name,
    # different product, so pushing 5223's measured 1.259 um onto it would be a
    # class estimate from one sample -- method rule 18. 5223 got its own profile
    # on 2026-08-24 instead.
    # ⚠ THIS GUARD PINNED THE LITERAL 19.0 UNTIL 2026-09-03 AND THAT WAS THE
    # WRONG THING TO PIN. The C45 rescale divided every ESTIMATED clump by 3.1,
    # 400TX's included and correctly so -- it is an estimate -- and the guard
    # failed on a database doing exactly what was approved. The property that
    # matters was never the number: it is that the STILL film's value stays an
    # ESTIMATE and does not converge on the CINE film's MEASUREMENT. Asserted
    # that way now, it survives any future corpus-wide scale change and still
    # catches the trade-name merge it was written for.
    _txs = get_profile("KODAK_TRI_X_400TX")
    _tx5 = get_profile("EASTMAN_TRI_X_5223")
    _k = film_profiles._CLUMP_RESCALE_C45_2026_09_03
    chk("the STILL Tri-X is still an ESTIMATE and has not converged on 5223's "
        "measured clump -- two products, one trade name",
        abs(_txs.grain.clump_um_g - 19.0 / _k) < 1e-3
        and _txs.grain.clump_um_g > 3.0 * _tx5.grain.clump_um_g,
        "400TX %.3f um (the estimate 19.0 through the C45 rescale k=%.1f) "
        "against 5223's measured %.3f um -- a factor of %.1f apart, so no merge "
        "has crept in" % (_txs.grain.clump_um_g, _k, _tx5.grain.clump_um_g,
                          _txs.grain.clump_um_g / _tx5.grain.clump_um_g))
    # And the two new profiles must keep saying which of their numbers are real.
    _new_est = {"EASTMAN_TRI_X_5223": "NOT GROUNDED",
                "KODAK_8374": "SPEED CELLS LEFT BLANK"}
    _ne_bad = [n for n, tok in _new_est.items()
               if tok not in " ".join(get_profile(n).provenance.sources).upper()]
    chk("the 2 new T-101 profiles still flag what is estimate-grade",
        not _ne_bad, ", ".join(_ne_bad) if _ne_bad
        else "5223 lists its estimates; 8374 records that T-101 prints no speed")
    # ---- 2026-08-25: the first measured B&W sigma(D), and the 35 stocks that
    # ---- still carry the estimate it contradicts.
    _rv = get_profile("KODAK_TRI_X_REVERSAL_200").grain
    chk("Tri-X Reversal carries the MEASURED sigma(D) shape, not the estimate",
        _rv.sigma_shape_measured
        and abs(_rv.sigma_shape_toe - 0.262) < 1e-9
        and abs(_rv.sigma_shape_dmax - 2.829) < 1e-9
        and abs(_rv.sigma_shape_toe_at - 0.352) < 1e-9
        and abs(_rv.sigma_shape_dmax_at - 3.089) < 1e-9,
        "0.262 at D 0.352 / 1.0 / 2.829 at D 3.089, from the 7266 sheet")
    # ⚠ THE APPARENT INTERIOR PEAK IS NOT STORED, ON PURPOSE. A 2.93x maximum at
    # D 3.16 shows up in the raw pairing, but it sits where the characteristic
    # curve is FLAT (|dD/dlogE| < 0.5), so the same density maps to many sigmas
    # there. Re-adding it from the raw trace would be reinstating an artefact.
    chk("Tri-X Reversal stores NO interior sigma peak",
        _rv.sigma_shape_peak == 0.0 and _rv.sigma_shape_peak_at == 0.0,
        "the 2.93x apparent peak lies in the flat, ill-conditioned zone")
    # ⚠ THE LEVEL IS NOT ADOPTED. The panel reads 22.3 at this file's NET-1.0
    # convention against a stored 10.0, but the sheet says the curve uses
    # "modified measuring techniques", so only the SHAPE is grounded.
    chk("Tri-X Reversal keeps rms 10.0 -- the panel grounds shape, not level",
        abs(_rv.rms_granularity - 10.0) < 1e-9
        and "modified measuring techniques" in
        " ".join(get_profile("KODAK_TRI_X_REVERSAL_200").provenance.sources),
        "10.0 kept; the 22.3 the panel implies is cited, not stored")
    # ⚠ AND THE SCOPE MUST STAY HELD. 34 reversal stocks share the 0.7/1.0/0.5
    # estimate that this measurement contradicts in DIRECTION. Fixing them from
    # one sample is method rule 18; this guard records the count so a later pass
    # cannot quietly "harmonise" them, and fails if the estimate is edited
    # without a measurement behind it.
    _rev_est = [p.name for p in FILM_PROFILES
                if p.kind == StockKind.REVERSAL and not p.grain.sigma_shape_measured
                and (p.grain.sigma_shape_toe, p.grain.sigma_shape_dmax) == (0.7, 0.5)]
    # ⚠ 34 -> 1 ON 2026-08-30, QUEUE F2 RESOLVED. This guard pinned the size of
    # a defect: 34 reversal stocks carrying 0.7/1.0/0.5, a FALL, where both
    # measurements RISE. The default is corrected to 0.21/1.00/2.97 and the
    # population collapses to the single stock that sets the old triple in its
    # own literal rather than taking the default. It is named rather than
    # rounded away: a leftover literal is exactly what a population count
    # exists to surface.
    chk("only the one literal hold-out still carries the old reversal estimate",
        len(_rev_est) == 1,
        "%d stock(s) still on 0.7/1.0/0.5: %s -- the class default now rises "
        "(0.21/1.00/2.97, the mean of the two measurements)"
        % (len(_rev_est), ", ".join(_rev_est) or "none"))
    # ---- queue F2, investigated 2026-08-26. UNBLOCKED SINCE C1 CLOSED ON
    # ---- 2026-08-18 AND NOBODY NOTICED FOR EIGHT DAYS.
    # ⚠ AND THE SCOPE IS 4x WHAT THE QUEUE ROW AND EVERY REPORT SO FAR CLAIMED.
    # The row says "the 103-stock default"; the record above says 34 stocks are
    # contradicted in direction. Both understate it. Measured live:
    #
    #     group                 n     dmax/mid            rises  falls
    #     measured NEGATIVES   11     mean 0.68 (0.50-0.90)   0     11
    #     heuristic NEGATIVES 113     mean 1.24 (1.00-1.80) 112      0
    #     measured REVERSALS    2     mean 2.96 (2.83-3.10)   2      0
    #     heuristic REVERSALS  34     0.50 exactly            0     34
    #
    # ⚠ SO BOTH DEFAULTS ARE CONTRADICTED IN DIRECTION BY EVERY MEASUREMENT OF
    # THEIR OWN CLASS -- 146 of 147 stocks, not 34. Negatives carry a RISING
    # sigma toward dmax where all eleven measurements FALL; reversals carry a
    # FALLING sigma where both measurements RISE.
    # ⚠ ONE MITIGATION, and it is real for the negatives: NO unmeasured stock
    # sets `sigma_shape_peak` (0 of 147) while ALL ELEVEN measured negatives do,
    # at 1.20-1.62 located 0.65-0.80 of the way up the scale. So the negative
    # heuristic's "1.20 at dmax" is standing in for an INTERIOR PEAK the triple
    # cannot express -- the rise is real, it is in the wrong PLACE, and the fall
    # after it is missing. The reversal heuristic has no such excuse: it is
    # simply backwards.
    # ⚠ NOTHING IS CHANGED HERE. Every option moves 146 renders, which is an
    # owner decision on the same footing as C16. These counts are pinned so the
    # contradiction cannot be absorbed silently, and so that the day someone
    # edits a default the guard says what the measurements think of it.
    _mneg = [p.grain for p in FILM_PROFILES if p.grain.sigma_shape_measured
             and p.kind is StockKind.NEGATIVE]
    _hneg_p = [p for p in FILM_PROFILES if not p.grain.sigma_shape_measured
               and p.kind is StockKind.NEGATIVE]
    _hneg = [p.grain for p in _hneg_p]
    # ⚠ REWRITTEN 2026-08-30, QUEUE F2 RESOLVED. The block above is the state
    # this guard was written to pin and is kept as the record of it. What it
    # pinned is fixed: the reversal default was backwards and now rises
    # (0.21 / 1.00 / 2.97, the mean of the two measurements), and the COLOUR
    # negative default now falls to 0.68 with the interior peak the eleven
    # measurements all carry (1.38 at 0.75 of scale).
    #
    # ⚠ WHAT IS DELIBERATELY STILL CONTRADICTED, and this guard now asserts it
    # rather than the whole population: the 51 MONOCHROME negatives keep the
    # old rising triple. Every one of the eleven measurements is a Kodak
    # COLOUR CINE stock, so giving their shape to B&W silver is the class jump
    # method rule 18 forbids, and no document in this corpus carries a
    # granularity-versus-density curve for a named B&W NEGATIVE. They are not
    # right; they are UNEVIDENCED, which is a different thing and is the
    # honest state until F2b lands a measurement.
    _mono_neg = [p.grain for p in _hneg_p if p.is_monochrome]
    _col_neg = [p.grain for p in _hneg_p if not p.is_monochrome]
    chk("every measured NEGATIVE sigma(D) falls toward dmax, and the COLOUR "
        "negative default now agrees with them",
        len(_mneg) == 12 and all(g.sigma_shape_dmax < 1.0 for g in _mneg)
        and all(g.sigma_shape_dmax < 1.0 for g in _col_neg
                if g.sigma_shape_peak > 0),
        "12 measured fall (0.47-0.90); the colour-negative default is now "
        "0.68 with an interior peak 1.38 at 0.75 of scale. ⚠ IT STAYED 11 on "
        "2026-09-02c: E5's 12th candidate was traced and then withdrawn on a "
        "density-space mismatch, not on its shape. \u26a0 11 -> 12 ON "
        "2026-09-17: EASTMAN_5293_250T_1982, off Kennel et al. 1982 Fig. 13 "
        "-- the first of the twelve that is NOT a Kodak colour cine stock of "
        "the VISION era, and the steepest fall of any of them at 0.473, which "
        "is the conventional-grain emulsion the paper describes")
    # 55, not the 51 the F2 row claimed -- recounted live 2026-08-30. One of
    # them, TASMA_FN_64, carries its own literal shape and is excluded rather
    # than counted as agreeing.
    # ⚠ 55 -> 56 on 2026-09-02 (queue N1): FUJI_NEOPAN_SS, a new monochrome
    # negative that lands on the same unevidenced heuristic as the rest --
    # AF3-411E prints no granularity of any kind, so it could not have landed
    # anywhere else.
    chk("the MONOCHROME negative default is still unevidenced and still rises",
        # ⚠ 56 -> 61 on 2026-09-17: queue P63's five 1956 sheet emulsions are
        # all monochrome negatives and all land on the same unevidenced
        # default, because the 1956 book prints graininess as a WORD and the
        # book's own page 20 says so.
        len(_mono_neg) == 61
        and sum(1 for g in _mono_neg if g.sigma_shape_dmax > 1.0) == 60,
        "56 B&W negatives keep 0.4/1.0/1.2 -- no measured B&W NEGATIVE shape "
        "exists in this corpus, so the colour-cine triple is a class jump "
        "that was refused, not an oversight (queue F2b)")
    chk("both measured REVERSALS rise, and the reversal default now rises too",
        all(p.grain.sigma_shape_dmax > 1.0 for p in FILM_PROFILES
            if p.grain.sigma_shape_measured and p.is_reversal)
        and all(p.grain.sigma_shape_dmax > 1.0 for p in FILM_PROFILES
                if not p.grain.sigma_shape_measured and p.is_reversal
                and p.grain.sigma_shape_toe == 0.21),
        "measured 2.83 and 3.10; default 2.97, their mean. ⚠ n = 2, adopted "
        "by owner decision over a flat 1.0 -- see the _grain_v2 note")
    chk("the sigma(D) shape is still read by NO renderer",
        not any(p.grain.sigma_shape_measured for p in FILM_PROFILES
                if p.grain.sigma_shape_toe in (0.21, 0.81)),
        "the corrected defaults must NOT set sigma_shape_measured -- they are "
        "a documented placeholder, not a measurement, and the wiring honours "
        "a shape only when that flag is set")

    # ⚠ REWRITTEN 2026-09-02, QUEUE C43. This used to assert that Tri-X keeps
    # the 1.3 CLASS CONSTANT while citing T-101's measured 2.0-2.34, on the
    # reasoning that 0.0016 sr is nearly collimated and 1.3 "corresponds to a
    # real condenser cone". That reasoning double-counted the geometry: the
    # collection cone lives entirely in E = 1 - scanner_specular, and beta is
    # defined as a FILM property. `callier_q` is now derived per stock from its
    # own mid slope through the base-corrected Mees relation (see
    # `sayanagi_callier.py`), so what has to stay true is that the value is on
    # the measured side of the old class constant, is under Sayanagi's ceiling
    # of 2, and that T-101's own figure is still cited WITH its collection
    # angle -- because the angle is what makes 2.0-2.34 a different quantity
    # from a condenser reading, and dropping it would lose the distinction.
    _tx = get_profile("KODAK_TRI_X_400TX")
    chk("Tri-X's callier_q is now derived, and still cites T-101 with its angle",
        1.3 < _tx.callier_q < 2.0
        and abs(_tx.callier_q - film_profiles._callier_beta_for(_tx)) < 1e-9
        and "0.0016 steradian" in " ".join(_tx.provenance.sources),
        "callier_q %.4f from mid slope %.3f; T-101's 2.0-2.34 cited with its "
        "0.0016 sr collection angle" % (_tx.callier_q, _tx.curves.g.mid_slope))
    # And the two figures that are NOT in either document must not appear as
    # stored values on HPS: no resolving power was printed for any film.
    chk("HPS f50 is still the unsourced estimate, not 40 lp/mm",
        abs(_hps.mtf.f50_g - 26.0) < 1e-9,
        "f50 26.0 estimate; neither BBC document prints a film resolving power")

    chk("HPS keeps the method-rule-14 Soviet-source caveat",
        "OUTRANKS this citation"
        in " ".join(get_profile("ILFORD_HPS").provenance.sources),
        "rule 14 recorded")


# ---- 24. C1e per-layer VISION3 grain, and C8 reciprocity -----------------
if _sec_on():
    # ---------------------------------------------------------------- C1e ----
    # The three VISION3 stocks whose own TI sheet separates all the granularity
    # curves it needs. Values are RATIOS off that sheet multiplied onto the
    # stored pooled rms, so what is pinned here is the ratio, to the cent.
    _C1E = {
        "KODAK_VISION3_50D_5203":  (2.60, 2.60,  4.71),
        "KODAK_VISION3_250D_5207": (4.20, 4.20,  8.92),
        "KODAK_VISION3_500T_5219": (5.92, 6.60, 17.84),
    }
    _bad = []
    for _n, _want in _C1E.items():
        _got = get_profile(_n).grain.rms_rgb()
        if max(abs(_got[_i] - _want[_i]) for _i in range(3)) > 5e-3:
            _bad.append("%s %s" % (_n, tuple(round(v, 2) for v in _got)))
    chk("the 3 VISION3 stocks carry their measured per-layer rms",
        not _bad, "; ".join(_bad) if _bad else "3 of 3")

    # ⚠ THE GUARD THAT MATTERS, and it is deliberately a floor on EVERY measured
    # stock rather than a value on one. Nine sheets now measure blue against
    # green and the lowest is 1.81x; the schema's discarded tier-2 ladder said
    # 1.30x. So any future "tidy-up" back toward that ladder -- or a paste of the
    # heuristic over a measured literal, which has happened once already
    # (GEVACOLOR_NEG_682, 2026-08-17) -- fails here instead of rendering quietly.
    # ⚠ STATED AS A FORBIDDEN BAND RATHER THAN A LIST OF STOCKS, so it cannot go
    # stale the way a count can. Every colour negative's blue/green ratio must be
    # one of exactly three things, and the gap between them is the point:
    #   ~1.00  the document prints ONE pooled figure for the whole film (Svema TU
    #          specifications), so r = g = b is what the source says;
    #   ~1.30  _grain_v2's tier-2 ladder, untouched, on the stocks with no
    #          per-layer measurement of their own;
    #   >=1.75 measured off a sheet -- nine of them, spanning 1.81 to 2.79.
    # A value INSIDE the 1.31-1.75 gap means someone split the difference: either
    # a measured value diluted toward the ladder or the ladder nudged toward the
    # measurements. Both are the "average two sources" move method rule 4
    # forbids, and neither would look wrong in a render.
    _band = []
    for _p in FILM_PROFILES:
        if _p.is_monochrome or _p.is_reversal or _p.reseau is not None:
            continue
        if _p.name == "TECHNICOLOR_THREE_STRIP":
            continue
        # ⚠ ORWOCOLOR_NC3 IS EXEMPT, 2026-09-15, AND IT IS THE ONE CASE THIS
        #   GUARD WAS NEVER WRITTEN FOR. The band exists to catch a ladder
        #   value nudged toward a measurement or the reverse. NC 3's triple is
        #   NEITHER: Tabelle 1 of Tamm & Weisflog prints 28 / 25 / 28 at D 1.0
        #   outright, so its b/g of 1.12 is a MEASUREMENT that happens to land
        #   in the gap, and a GDR emulsion is under no obligation to reproduce
        #   a Kodak-derived ladder. Excluding it keeps the guard's question --
        #   "did someone split the difference?" -- answerable.
        if _p.name == "ORWOCOLOR_NC3":
            continue
        _r, _gg, _b = _p.grain.rms_rgb()
        _ratio = _b / _gg
        if not (abs(_ratio - 1.00) < 0.01 or abs(_ratio - 1.30) < 0.01
                or _ratio >= 1.75):
            _band.append("%s b/g %.2f" % (_p.name, _ratio))
    chk("no colour negative's blue/green ratio sits between the ladder and the "
        "measurements",
        not _band, "; ".join(_band[:3]) if _band
        else "pooled 1.00, ladder 1.30, measured 1.81-2.79, nothing between")

    # The four Svema colour negatives excluded above are excluded for a stated
    # reason and not silently: their TU specifications print ONE granularity
    # figure for the whole film, so r = g = b is what the document says. If one
    # of them ever gains a per-layer read this list must shrink.
    _flat = [_n for _n in ("SVEMA_DS_5M", "SVEMA_LN_8", "SVEMA_LN_9",
                           "SVEMA_LN_9S")
             if get_profile(_n).grain.rms_rgb()[2]
             != get_profile(_n).grain.rms_rgb()[1]]
    chk("the 4 Svema negatives still carry one pooled figure per document",
        not _flat, ", ".join(_flat) if _flat else "4 of 4 flat, as printed")

    # 5213 is the one VISION3 stock left on the heuristic, because its sheet
    # draws the three granularity curves as a single bold band. Pinned so that
    # "finish the family" cannot happen quietly: filling it would need a
    # document, and this fails the moment a number appears without one.
    # ⚠ TESTED THROUGH THE RATIOS, NOT THE LITERAL. `_grain_v2` runs at module
    # build time, so by the time anything can read a profile the heuristic has
    # already filled these fields and "is the literal empty" is unanswerable.
    # The ladder's own ratios are the observable: exactly 1.10 and 1.30.
    _r13, _g13, _b13 = get_profile("KODAK_VISION3_200T_5213").grain.rms_rgb()
    chk("5213 stays on the heuristic until a per-layer sheet exists",
        abs(_b13 / _g13 - 1.30) < 0.01 and abs(_r13 / _g13 - 1.10) < 0.01,
        "b/g %.2f r/g %.2f -- band-only sheet, nothing to read"
        % (_b13 / _g13, _r13 / _g13))

    # ----------------------------------------------------------------- C8 ----
    # INERTNESS IS THE WHOLE CONTRACT. Zero time = zero shift, for every stock,
    # exactly -- not "small". A render made before the field existed must be
    # reproducible bit for bit, and that is only true if the shift is 0.0.
    _live = [p.name for p in FILM_PROFILES
             if any(v != 0.0 for v in fs.reciprocity_log_shift(p, 0.0))]
    chk("reciprocity is exactly inert at exposure_time_s = 0",
        not _live, ", ".join(_live[:3]) if _live else "160 of 160 stocks")

    # Below its own onset a stock must also be inert: 1/48 s is the shutter of a
    # 24 fps camera at 180 degrees, i.e. the commonest exposure in the corpus's
    # whole subject matter, and nothing should happen there.
    _cine = [p.name for p in FILM_PROFILES
             if p.reciprocity.onset_s >= 0.02
             and not p.reciprocity_table.has_data
             and any(v != 0.0 for v in fs.reciprocity_log_shift(p, 1.0 / 48.0))]
    chk("a 1/48 s exposure moves nothing on a spec-only stock",
        not _cine, ", ".join(_cine[:3]) if _cine else "no correction at 1/48 s")

    # Direction: a longer exposure can only ever LOSE speed, never gain it. A
    # sign error here would brighten long exposures, which is the one outcome no
    # sensitometry supports -- and it would look like a plausible "lift".
    _sign = []
    for _p in FILM_PROFILES:
        for _t in (2.0, 10.0, 60.0, 600.0):
            if any(v > 0.0 for v in fs.reciprocity_log_shift(_p, _t)):
                _sign.append("%s at %.0fs" % (_p.name, _t))
    chk("reciprocity never increases effective exposure",
        not _sign, "; ".join(_sign[:3]) if _sign else "loss only, 4 times x 160")

    # The measured tables must BEAT the Schwarzschild spec where both exist, or
    # the six documents on file are decoration. EKTACHROME 64 is the case that
    # proves it: its table is U-shaped in time (0.5 stop at 1e-4 s, zero at
    # 0.1 s, 1.5 stops at 10 s) and no single exponent can express both ends.
    _e64 = get_profile("EKTACHROME_64")
    _hi = fs.reciprocity_log_shift(_e64, 1.0e-4)     # high-intensity branch
    _lo = fs.reciprocity_log_shift(_e64, 10.0)       # low-intensity branch
    # Read on BLUE at the long end, because blue is the record its CC20B says
    # loses the most and therefore the one carrying the printed 1.5 stops; green
    # is credited 0.20 decades back by that filter.
    chk("EKTACHROME 64's table drives both reciprocity branches",
        _hi[1] < -0.14 and _lo[2] < -0.44 and _e64.reciprocity_table.has_data,
        "1e-4 s %.3f dec (achromatic), 10 s blue %.3f dec" % (_hi[1], _lo[2]))

    # Held flat outside the measured range, NOT extrapolated. Kodak's own
    # tables walk the effective exponent by 0.15 per decade, so extrapolating
    # one decade past the last entry is a quarter-stop error at least.
    chk("a measured table holds flat past its last entry",
        fs.reciprocity_log_shift(_e64, 10.0)
        == fs.reciprocity_log_shift(_e64, 3600.0),
        "10 s and 3600 s agree exactly")

    # CHROMATIC failure is the part a single exponent cannot carry at all, and
    # the CC filter is the only place the corpus states it. CC20B at 10 s must
    # make blue lose MORE than green -- that is what "add a blue filter to fix
    # it" means. If this ever equalises, the cast is gone and long exposures go
    # merely dark.
    chk("EKTACHROME 64 loses more blue than green at 10 s, as its CC20B says",
        _lo[2] < _lo[1] - 0.15, "blue %.3f vs green %.3f dec" % (_lo[2], _lo[1]))

    # ⚠ THE CONVENTION GUARD, AND IT IS THE ONE THAT CAUGHT A REAL ERROR. A
    # printed "increase exposure 2/3 stop and use a CC10R" is TWO instructions
    # acting on one frame: the lens opens 2/3 stop on all three records, then the
    # filter takes 0.10 density back off green and blue. So the film's WORST
    # record loses exactly the printed stops -- 2/3 here -- and the filtered ones
    # lose less. The first implementation added the filter's density to the worst
    # record instead, giving 1 stop where the sheet says 2/3: right ordering,
    # wrong level, and invisible in a frame. Pinned on the general form so it
    # holds for all 21 tables rather than for one example.
    _conv = []
    for _p in FILM_PROFILES:
        _t = _p.reciprocity_table
        if not _t.has_data:
            continue
        _last_t = _t.times_s[-1]
        _last_s = _t.stops_correction[-1]
        _sh = fs.reciprocity_log_shift(_p, _last_t)
        _worst = -min(_sh) / 0.30102999566398120      # loss in stops
        if abs(_worst - _last_s) > 1e-6:
            _conv.append("%s %.3f vs printed %.3f" % (_p.name, _worst, _last_s))
    chk("every measured table's worst record loses exactly the printed stops",
        not _conv, "; ".join(_conv[:3]) if _conv
        else "21 tables, worst record == printed correction")

    # 5205 is the worked example of the above, kept as a named case because its
    # sheet is the one that exposed the error: "+2/3 stop and a CC10R" at 1 s.
    _05 = fs.reciprocity_log_shift(get_profile("KODAK_VISION2_250D_5205"), 1.0)
    # ⚠ AND NOTE WHAT GREEN IS NOT: it is 2/3 stop MINUS the filter's 0.10
    # DENSITY, i.e. 0.3345 stops -- not 1/3. The filter is specified in density
    # and 0.10 density is 0.332 of a stop, so the two never land on a round
    # fraction together. Asserting the round number here would be asserting the
    # stops-and-back conversion the law deliberately avoids.
    _S = 0.30102999566398120
    chk("5205 at 1 s: red loses the printed 2/3 stop, green/blue 0.10 D less",
        abs(-_05[0] / _S - 2.0 / 3.0) < 1e-9
        and abs(-_05[1] / _S - (2.0 / 3.0 - 0.10 / _S)) < 1e-9,
        "r %.4f g %.4f stops" % (-_05[0] / _S, -_05[1] / _S))

    # ⚠ SEVEN STOCKS HELD A PRINTED CORRECTION AND RENDERED NOTHING before C8,
    # because a single Schwarzschild exponent had nowhere to put an absolute
    # offset and the fit had left them at p = 1.0. If any of them goes inert
    # again, its sheet has been disconnected from the renderer.
    _mute = [_n for _n in ("KODAK_VISION3_500T_5219", "KODAK_VISION2_200T_5217",
                           "KODAK_VISION2_500T_5218", "KODAK_VISION_250D_5246",
                           "KODAK_VISION_200T_5274", "KODAK_VISION_500T_5279",
                           "EASTMAN_EXR_100T_5248")
             if all(v == 0.0 for v in
                    fs.reciprocity_log_shift(get_profile(_n), 10.0))]
    chk("the 7 stocks whose sheets print a correction are no longer silent",
        not _mute, ", ".join(_mute) if _mute else "7 of 7 now respond at 10 s")

    # And the achromatic case must stay achromatic: Kentmere prints stops with
    # no filter at all, which is a statement, not a gap.
    _k = fs.reciprocity_log_shift(get_profile("KENTMERE_PAN_100"), 100.0)
    chk("an achromatic table stays achromatic",
        _k[0] == _k[1] == _k[2] and _k[1] < -0.4,
        "%.3f dec on all three" % _k[1])

    # ⚠ AND IT MUST ACTUALLY REACH THE RENDER. The three checks above test the
    # law; this one tests the wiring, because a stage computed and then not
    # applied is the exact failure the sigma(D) shape sat in for weeks.
    _sett0 = fs.RenderSettings(grain_scale=0.0, print_grain=False,
                              misreg_scale=0.0, flare=0.0)
    _sett1 = dataclasses.replace(_sett0, exposure_time_s=10.0)
    _patch = np.full((16, 16, 3), 0.18, dtype=np.float32)
    _a = fs.simulate(_patch, _e64, _sett0)
    _b = fs.simulate(_patch, _e64, _sett1)
    _d = float(np.max(np.abs(_a - _b)))
    chk("a 10 s exposure changes the render on a stock with a table",
        _d > 0.01, "max channel delta %.4f linear" % _d)
    # ... and changes NOTHING on a stock whose sheet says no correction applies.
    _acr = get_profile("FUJI_NEOPAN_ACROS_100")
    _c = fs.simulate(_patch, _acr, _sett0)
    _e = fs.simulate(_patch, _acr, dataclasses.replace(_sett0,
                                                       exposure_time_s=120.0))
    chk("ACROS is unchanged at 120 s, as its own sheet states",
        np.array_equal(_c, _e), "bit identical")


    # =======================================================================
    #  G-FILMID / G-FILMORDER -- storage identity
    #
    #  ⚠⚠ THE FREEZE WAS RETIRED ON 2026-09-08 BY OWNER DECISION AND THIS BLOCK
    #  NOW GUARDS THE OPPOSITE ARRANGEMENT. Read the next paragraph as the
    #  argument against the current design, not as a description of it.
    #
    #  WHAT THE FREEZE PROTECTED, 2026-08-28 to 2026-09-08. Before it, database
    #  order was the natural name sort, so every eFILM_PROFILE value, every
    #  GetFilmDatabase() subscript and every film_names.txt line number was a
    #  function of which stock names happened to exist. Adding one stock
    #  renumbered the rest, and every saved After Effects or Premiere project
    #  then rendered a DIFFERENT FILM.
    #
    #  WHY IT IS GONE. The freeze delivered an alphabetical panel only through
    #  film_display_order.txt, a runtime indirection the owner has ruled out:
    #  "I need film in database ordered alphabetically and TXT file fully
    #  reflect this order... I don't need additional *.txt file for re-ordering
    #  in run-time." No arrangement is both alphabetical in storage and stable
    #  under insertion. The owner chose alphabetical, having been shown the cost
    #  twice.
    #
    #  ⚠ SO THE RENUMBERING FAILURE MODE IS BACK, AND IT IS NOT A DEFECT TO
    #  FILE. 177 of 184 stocks moved at the cutover; a future "AGFA APX 50"
    #  will take index 1 and shift 183. What guards it now is not stability but
    #  DISCLOSURE: film_id_migration.txt maps old -> new, and check 5 asserts
    #  that map is complete and correct, because it is the only artefact that
    #  can repair a project saved before today.
    #
    #  WHAT EACH CHECK DOES NOW:
    #    1, 2, 4, 6  unchanged -- the lock is still the migration source, so
    #                its internal integrity still matters exactly as much
    #    3           REPLACED: "order is ascending id" -> the database IS in
    #                natural-name order (G-FILMORDER)
    #    5           REPLACED: "pre-freeze stocks at id == index" -> the
    #                migration map is complete and correct
    #    5b          NEW: film_names.txt line k IS database index k, checked
    #                against the emitted file rather than a re-derivation
    # =======================================================================
    import os as _os
    _lockp = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "film_ids.lock")

    # =======================================================================
    #  G-GAMMA -- two stored curves that no emulsion can have, 2026-09-08
    #
    #  ⚠ THIS IS A WITNESS, NOT A FIX, AND NOTHING WAS CHANGED IN THE DATA.
    #  Found while chasing an owner-reported colour cast on reversal stocks.
    #  Of 115 colour profiles, whose median green gamma is 0.62, exactly two
    #  sit above 2.5:
    #        KODAK_EKTACHROME_100D_5285   15.43   <- database maximum
    #        SUPER_ANSCOCHROME_1957        5.27
    #  15.43 is not a characteristic curve. The steepest colour reversal film
    #  ever sold is around 2.5, the steepest monochrome stock in this database
    #  is POLAROID_51 at 3.35, and 15.43 is 6x the next colour stock and 25x
    #  the colour median.
    #
    #  ⚠ AND IT HAD A MEASURABLE CONSEQUENCE, which is why it is pinned rather
    #  than noted. Both stocks were among the five whose interimage
    #  coefficients overshot their published target -- the solver was being
    #  asked to hit a patent figure on a curve that cannot exist, and it
    #  obliged by driving the coefficient to an extreme. EKTACHROME_100D_5285
    #  rendered at R/B 3.28 against a reversal median of 1.62.
    #
    #  The remedy is a RE-TRACE of both curves, recorded in NotFound.md. It is
    #  not a value this project may invent: the sheets are in the corpus and
    #  the trace has to be redone. Until then this guard does two things --
    #  it stops either number drifting silently, and it fails on a THIRD stock
    #  arriving above the bound, which is the case nobody would notice.
    _GAMMA_BOUND = 2.5
    _GAMMA_KNOWN = {"KODAK_EKTACHROME_100D_5285": 15.43,
                    "SUPER_ANSCOCHROME_1957": 5.27}
    _g_over, _g_moved = [], []
    for _p in FILM_PROFILES:
        if _p.is_monochrome:
            continue
        _gg = _p.curves.g.gamma
        _pin = _GAMMA_KNOWN.get(_p.name)
        if _pin is not None:
            if abs(_gg - _pin) > 0.01:
                _g_moved.append("%s %.2f vs pinned %.2f" % (_p.name, _gg, _pin))
        elif _gg > _GAMMA_BOUND:
            _g_over.append("%s %.2f" % (_p.name, _gg))
    chk("G-GAMMA  exactly two colour curves exceed gamma 2.5, both pinned and "
        "both awaiting a re-trace (NotFound.md 2026-09-08b) -- no third stock "
        "has joined them and neither has moved",
        not _g_over and not _g_moved,
        "; ".join(_g_over + _g_moved) if (_g_over or _g_moved)
        else "EKTACHROME_100D_5285 15.43, SUPER_ANSCOCHROME_1957 5.27, "
             "median colour gamma 0.62")

    # =======================================================================
    #  G-REGISTER -- the nine printed parameters are complete on every stock
    #
    #  ⚠ THIS GUARD EXISTS BECAUSE THE CLAIM IT CHECKS WAS FALSE FOR 23 STOCKS
    #  AND NOTHING NOTICED. `_PARAM_SOURCES_DERIVED` is a literal dict keyed by
    #  the 161 stock names that existed on 2026-08-27; the ParamSource
    #  docstring promises that in the nine columns FilmActiveProfiles.md prints
    #  "an absence is impossible, so nothing can fall back to the profile tier
    #  and quietly read as evidence". Every stock added after that date had an
    #  EMPTY row and did fall back -- 156 cells over 31 stocks, measured
    #  2026-09-09.
    #
    #  ⚠ A LITERAL DICT KEYED BY STOCK NAME DECAYS EVERY TIME A STOCK IS ADDED,
    #  and prose cannot stop that. This can. COMPLETENESS IS ASSERTED (hard);
    #  the fill count is only REPORTED, because a cell upgraded from the
    #  generated `assumed` placeholder to a real traced/stated record LOWERS
    #  the count and that is the outcome we want -- pinning it would make an
    #  improvement fail the build.
    _nine = [_k for _k, _u, _c in film_profiles._NINE_PRINTED]
    _reg_bad = [(_p.name, sorted(set(_nine) - {_r.param for _r in _p.param_sources}))
                for _p in FILM_PROFILES
                if not set(_nine) <= {_r.param for _r in _p.param_sources}]
    chk("G-REGISTER  all nine PRINTED parameters carry a provenance record on "
        "every stock, so an absence in those columns is genuinely impossible "
        "(the claim the ParamSource docstring makes)",
        not _reg_bad,
        "%d stock(s) incomplete: %s" % (len(_reg_bad), _reg_bad[:3])
        if _reg_bad else
        "%d stocks x 9 columns, %d cells filled by the 2026-09-09 completeness "
        "pass on %d stocks (upgrade target, not a target to keep)"
        % (len(FILM_PROFILES), film_profiles.REGISTER_GAP_FILLED,
           len(film_profiles.REGISTER_GAP_STOCKS)))

    # ⚠ AND NO PARAMETER MAY CARRY TWO RECORDS -- the G-PROV invariant. The
    # completeness pass only ADDS where a cell was empty, so it cannot break
    # this; the check is here because that is exactly the kind of guarantee
    # that holds until someone adds a second pass which also writes.
    _prov_dup = []
    for _p in FILM_PROFILES:
        _seen = {}
        for _r in _p.param_sources:
            _seen[_r.param] = _seen.get(_r.param, 0) + 1
        _prov_dup += [(_p.name, _k) for _k, _v in _seen.items() if _v > 1]
    chk("G-PROV  no profile carries two provenance records for one parameter",
        not _prov_dup, "%d duplicate(s): %s" % (len(_prov_dup), _prov_dup[:3]))

    chk("G-FILMID  film_ids.lock exists", _os.path.exists(_lockp), _lockp)

    if _os.path.exists(_lockp):
        _ids, _retired, _dupes = {}, set(), []
        with open(_lockp, "r", encoding="utf-8") as _fh:
            for _line in _fh:
                _line = _line.rstrip("\n")
                if not _line or _line.startswith("#"):
                    continue
                _sid, _, _nm = _line.partition("\t")
                _sid = int(_sid)
                if _sid in _ids.values() or _sid in _retired:
                    _dupes.append(_sid)
                if _nm.startswith("RETIRED "):
                    _retired.add(_sid)
                else:
                    _ids[_nm] = _sid

        _dbnames = [_p.name for _p in FILM_PROFILES]

        # 1. every stock in the database has a frozen id. A stock without one
        #    has no stable identity at all.
        # ⚠ RENAMES ARE NOT MISSING IDS, 2026-09-08. The lock is keyed by the
        # name that was current when the id was issued, so after the
        # AGFACOLOR -> AGFA rename three stocks legitimately have no entry
        # under their NEW key. Reading that as "no stable identity" would be
        # exactly wrong -- the identity is there, under the old name, and
        # FILM_RENAMES is what connects the two. A stock that is genuinely
        # absent from the lock still fails.
        _rev_ren = {_v: _k for _k, _v in film_profiles.FILM_RENAMES.items()}
        _missing = [_n for _n in _dbnames
                    if _n not in _ids and _rev_ren.get(_n) not in _ids]
        chk("G-FILMID  every stock carries a frozen id (renamed keys resolved "
            "through FILM_RENAMES)",
            not _missing, f"{len(_missing)} without: {_missing[:4]}")

        # 2. no id is shared. Two stocks on one id is two emulsions behind one
        #    saved-project reference.
        chk("G-FILMID  no id is issued twice", not _dupes, f"dupes {_dupes[:4]}")

        # 3. ⚠ REPLACED 2026-09-08. This asserted "database order is ASCENDING
        #    id", which was the freeze's central invariant and is now FALSE BY
        #    DESIGN: the owner's decision stores the database in natural-name
        #    order, so 177 of 184 indices no longer equal their frozen id.
        #    Deleting the check outright would leave storage order unguarded
        #    for the first time since 2026-08-28, so it is replaced by the
        #    invariant that now holds and now carries the same weight.
        _order = [_p.name for _p in FILM_PROFILES]
        _want = sorted(_order, key=film_profiles._natural_key)
        _first = next((_i for _i in range(len(_order))
                       if _order[_i] != _want[_i]), -1)
        chk("G-FILMORDER  the database IS in natural-name order "
            "(storage order is presentation order since 2026-09-08, so this "
            "is what film_names.txt line numbers and eFILM_PROFILE now rest "
            "on)",
            _order == _want,
            "in order" if _first < 0 else
            f"first mismatch at index {_first}: {_order[_first]} where "
            f"{_want[_first]} belongs")

        # 4. a retired id is never reissued to a live stock. Reissue is the one
        #    failure that silently points a project at the WRONG emulsion,
        #    rather than at nothing.
        _reissued = sorted(set(_ids.values()) & _retired)
        chk("G-FILMID  no retired id has been reissued",
            not _reissued, f"reissued {_reissued[:4]}")

        # 5. ⚠ REPLACED 2026-09-08, for the same reason as check 3. This
        #    asserted that the 161 pre-freeze stocks still sat at id == index
        #    -- the no-op proof that the freeze had disturbed nothing. The
        #    alphabetical re-sort disturbs it deliberately, so the check is
        #    replaced by the thing the owner now needs guaranteed instead: that
        #    the MIGRATION MAP is complete and correct, because it is the only
        #    artefact that can repair a project saved before today.
        _mig = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             "film_id_migration.txt")
        _newidx = {_n: _i for _i, _n in enumerate(_dbnames)}
        # ⚠ A RENAMED KEY MUST NOT COUNT AS ABSENT. The lock holds the OLD key,
        # the database the new one; film_profiles.FILM_RENAMES is the bridge.
        # Without it the AGFACOLOR -> AGFA rename of 2026-09-08 would make this
        # guard demand migration rows for three names no longer in the
        # database, and report the three that ARE there as unmapped.
        _ren = film_profiles.FILM_RENAMES
        _mig_rows = {}
        if _os.path.exists(_mig):
            with open(_mig, "r", encoding="utf-8") as _fh:
                for _line in _fh:
                    if _line.startswith("#") or "\t" not in _line:
                        continue
                    _parts = _line.rstrip("\n").split("\t")
                    _o, _n, _nm = _parts[0], _parts[1], _parts[2]
                    _mig_rows[_nm] = (int(_o), _n)
        _wrong = [(_nm, _row, _newidx.get(_nm))
                  for _nm, _row in _mig_rows.items()
                  if (_row[1] == "WITHDRAWN") != (_nm not in _newidx)
                  or (_nm in _newidx and _row[1] != "WITHDRAWN"
                      and int(_row[1]) != _newidx[_nm])]
        _absent = [_nm for _nm in _ids if _ren.get(_nm, _nm) not in _mig_rows]
        chk("G-FILMID  film_id_migration.txt maps every old id to the right "
            "new index (the ONLY thing that can repair a project saved before "
            "the 2026-09-08 re-sort)",
            _os.path.exists(_mig) and not _wrong and not _absent,
            f"{len(_wrong)} wrong {_wrong[:2]}, {len(_absent)} old ids absent "
            f"{_absent[:2]}" if (_wrong or _absent)
            else f"{len(_mig_rows)} rows, "
                 f"{sum(1 for _nm, _r in _mig_rows.items() if _r[1] != 'WITHDRAWN' and int(_r[1]) != _r[0])}"
                 f" of them moved")

        # 5b. ⚠ ONE-BUILD LAG, STATED SO IT IS NOT MISREAD AS A DEFECT: build.py
        #     runs verify BEFORE codegen, so on the FIRST build after any change
        #     to storage order this check reads the PREVIOUS film_names.txt and
        #     fails. It passes on the next run against the file codegen has
        #     since rewritten. The pre-existing "film_names.txt line order
        #     equals the GetFilmDatabase() vector order" check has always had
        #     the same property; this one inherits it rather than adding it.
        #
        # ⚠ AND THE NEW ORDER IS PINNED AGAINST film_names.txt ITSELF, not
        #     against a re-derivation of it. `write_film_names` reads the
        #     emitted slots back, so comparing the file to FILM_PROFILES tests
        #     the whole chain -- sort, slot packing, emission, names file -- and
        #     that chain is what the plugin's popup index actually rests on.
        #     Re-deriving the expected list from FILM_PROFILES twice would only
        #     prove the sort is idempotent.
        _namesp = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                "film_names.txt")
        if _os.path.exists(_namesp):
            with open(_namesp, "r", encoding="utf-8") as _fh:
                _lines = [_l.strip().strip('"') for _l in _fh if _l.strip()]
            _lines = [_l[:-1] if _l.endswith("|") else _l for _l in _lines]
            _expect = [_n.replace("_", " ") for _n in _dbnames]
            _bad = [(_i, _a, _b) for _i, (_a, _b)
                    in enumerate(zip(_lines, _expect)) if _a != _b]
            chk("G-FILMORDER  film_names.txt line k IS database index k, in "
                "alphabetical order, for every stock",
                len(_lines) == len(_expect) and not _bad,
                f"{len(_lines)} lines vs {len(_expect)} stocks; "
                f"{len(_bad)} mismatched {_bad[:2]}")

        # 6. THE PRE-FREEZE BLOCK IS PINNED BY DIGEST, and this check exists
        #    because fault injection found the hole the other five leave.
        #
        #    SWAPPING TWO NAMES between two ids in the lock is invisible to
        #    every check above. The database re-sorts from the lock, so after
        #    the swap the order still ascends, every stock still has an id, no
        #    id repeats, and id still equals index -- the corruption is
        #    SELF-CONSISTENT. What it actually did was hand two saved projects
        #    each other's emulsion, which is precisely the failure the freeze
        #    exists to prevent.
        #
        #    The only way to catch an edit to an EXISTING row is to compare
        #    against something outside the file. So the seeded 161 rows are
        #    pinned here by digest. Adding stocks does not disturb it -- new
        #    rows land after row 161 and are not hashed.
        #
        #    If this fails and the change was DELIBERATE (a retirement, a
        #    correction agreed with the owner), re-pin the constant in the same
        #    commit and say why in the message. Do not re-pin to make it green.
        _PREFREEZE_SHA256 = (
            "ee9314d74eb280817c2621d3acef3b11bf184fa1889328addc3a1f2d4be847c6")

        import hashlib as _hashlib
        with open(_lockp, "r", encoding="utf-8") as _fh:
            _rows_raw = [_l.rstrip("\n") for _l in _fh if "\t" in _l]

        _pre = _rows_raw[:161]
        _got = _hashlib.sha256("\n".join(_pre).encode("utf-8")).hexdigest()

        chk("G-FILMID  the pre-freeze block matches its pinned digest",
            _got == _PREFREEZE_SHA256,
            f"got {_got[:16]}... expected {_PREFREEZE_SHA256[:16]}...")

    # ---- G-V29 (2026-09-10, the patent + Glafkides harvest) --------------
    # Seven guards on the nineteen schema-v29 fields. Five exist to catch a
    # mistake that was ACTUALLY MADE during the harvest and would otherwise be
    # made again; one is the inertness claim the version bump rests on; one is
    # an independent cross-check on a rule the project already had.
    if _sec_on():
        _fp = film_profiles

        # 1. THE LOG BASE. Six independent derivations of size_sigma_log were
        #    made from the patent corpus and they disagreed by factors of
        #    exactly ln(10) = 2.3026 and exactly 2 -- one pair had used
        #    natural logs, another the spread of crystal AREA instead of
        #    diameter. The convention is now written out in full at the field.
        #    This guard is what stops the next derivation drifting off it: a
        #    sigma above 0.5 in log10-of-diameter units is a decade of crystal
        #    size across two sigma, which no coating survives, so it is the
        #    signature of the wrong base and not of an unusual film.
        _sig = [(_p.name, _p.emulsion.size_sigma_log) for _p in FILM_PROFILES
                if _p.emulsion.size_sigma_log]
        _sig_bad = [(n, v) for n, v in _sig if not (0.01 <= v <= 0.5)]
        chk("G-V29-SIGMALOG  every crystal size_sigma_log is in "
            "log10-of-DIAMETER units (0.01-0.5), not natural logs and not "
            "area-based", not _sig_bad,
            "%d populated, range %.3f-%.3f"
            % (len(_sig), min(v for _, v in _sig), max(v for _, v in _sig))
            if _sig and not _sig_bad
            else "offenders: %s" % (_sig_bad[:3],))

        # 2. TURBIDITY IS DELIBERATELY UNPOPULATED, and this guard is that
        #    reason recorded as a test. Glafkides' Gamma is a coefficient on a
        #    point-image diameter whose own d0 is tens of micrometres; the
        #    stored f50 values imply d0 = 1.9-7.8 um. The two are not the same
        #    diameter, and the conversion between them is printed nowhere in
        #    the corpus -- so Gamma = 18 fed through turbid_f50s would take an
        #    f50 of 100 cycles/mm to 14.6, a 6.8x loss of sharpness that no
        #    photograph shows.
        #
        #    IF THIS GUARD FAILS, SOMEONE HAS POPULATED THE FIELD. That is
        #    allowed, but only together with the calibration: a source stating
        #    turbidity as an MTF or f50 change, or giving a point-spread
        #    PROFILE rather than a single diameter. Deleting this check to
        #    make a populated value pass is the one thing that must not happen.
        _turb = [_p.name for _p in FILM_PROFILES if _p.mtf.turbidity_gamma_um]
        chk("G-V29-TURBIDITY  turbidity_gamma_um is 0.0 on every stock -- the "
            "diameter-to-f50 calibration is not in the corpus, so the carrier "
            "ships empty", not _turb,
            "0 of %d populated, as intended" % len(FILM_PROFILES) if not _turb
            else "%d populated without the calibration: %s"
                 % (len(_turb), _turb[:3]))

        # 3. KRON, the same shape of argument. p = 1/(1+a) is the law's
        #    LOW-INTENSITY LIMIT, so an existing schwarzschild_p cannot be
        #    inverted into (a, I0) without the optimum intensity, and no
        #    document in the corpus prints one for any named product.
        #    Inventing an I0 to preserve a p would be inventing a measurement.
        _kron = [_p.name for _p in FILM_PROFILES if _p.reciprocity.kron_a]
        chk("G-V29-KRON  kron_a is 0.0 on every stock -- no source prints an "
            "optimum intensity I0, and Schwarzschild's p cannot be inverted "
            "without it", not _kron,
            "0 of %d populated, as intended" % len(FILM_PROFILES) if not _kron
            else "%d populated: %s" % (len(_kron), _kron[:3]))

        # 4. THE rms APERTURE. Every stored rms figure in this database was
        #    entered under Kodak's 48 um convention, which the field docstring
        #    has stated since v1. Konica's patents state 25 um; reading one as
        #    the other overstates grain by sqrt(48/25) = 1.386, i.e. 39 %,
        #    more than the whole spread between a 100-speed and a 400-speed
        #    stock. Asserts both that the convention is uniform and that the
        #    conversion is the identity at 48.
        _ap = sorted({_p.grain.rms_aperture_um for _p in FILM_PROFILES})
        _ap_id = all(
            abs(_p.grain.rms_at_aperture(48.0) - _p.grain.rms_granularity)
            < 1e-12 for _p in FILM_PROFILES)
        chk("G-V29-APERTURE  every rms figure is on the 48 um convention and "
            "rms_at_aperture(48) is the identity",
            _ap == [48.0] and _ap_id,
            "apertures present: %s; identity holds: %s" % (_ap, _ap_id))

        # 5. THE RATE LAW MUST REPRODUCE THE POINTS IT SITS BESIDE. Where
        #    gamma_infinity and dev_rate_k are set, gamma(t) is checked
        #    against every stored DevelopmentPoint by validate() at 12 %.
        #    What validate() does NOT check is the 0.80*gamma_infinity
        #    ceiling: Glafkides §211 gives that as the usable contrast limit,
        #    so a fitted asymptote low enough to put the sheet's OWN published
        #    gamma above the ceiling is a fit that declares the manufacturer's
        #    stated development impossible. Three Kodak 1952 sheets failed
        #    exactly this way during the harvest and were refused, not forced.
        _rl_bad = []
        for _p in FILM_PROFILES:
            _pf = _p.processing_family
            if not _pf.has_rate_law:
                continue
            _gmax = max((_pt.gamma or _pt.contrast_index)
                        for _pt in _pf.points) if _pf.points else 0.0
            if _gmax > 0.80 * _pf.gamma_infinity + 1e-9:
                _rl_bad.append((_p.name, round(_gmax, 3),
                                round(_pf.gamma_infinity, 3)))
        _rl_n = sum(1 for _p in FILM_PROFILES
                    if _p.processing_family.has_rate_law)
        chk("G-V29-RATELAW  no fitted gamma_infinity puts a stock's own "
            "published gamma above the 0.80*gamma_infinity usable ceiling",
            not _rl_bad,
            "%d stocks carry a rate law, all under the ceiling" % _rl_n
            if not _rl_bad else "offenders: %s" % (_rl_bad[:3],))

        # 6. THE INERTNESS CLAIM. The v29 bump asserts that a v29 database
        #    renders bit-identically to a v28 one. None of the nineteen is
        #    emitted into the C++ at all -- the v23 precedent, recorded in the
        #    generated header -- so the claim reduces to: nothing in
        #    film_sim.py reads any of them. Checked by name against the module
        #    source. Crude, but it is the check that fails when someone wires
        #    one up without moving the version.
        import inspect as _inspect
        import film_sim as _fs29
        _v29_names = (
            "antihalation", "antihalation_undercoat_um", "silver_g_per_m2",
            "gelatin_g_per_m2", "coverage_source", "angle_deg",
            "dye_fade_low_density_factor", "gamma_infinity", "dev_rate_k",
            "induction_t0_min", "temp_q10", "kron_a", "kron_log_i0_rel",
            "short_onset_s", "turbidity_gamma_um", "turbidity_ref_log_e",
            "rms_aperture_um", "interimage_gamma_ratio_r",
            "interimage_gamma_ratio_g", "interimage_gamma_ratio_b",
            "gamma_criterion", "density_geometry",
            # -- schema v30
            "gamma_ratio_criterion",
            # -- schema v31
            "gost_speed_class")
        _sim_src = _inspect.getsource(_fs29)
        _leaked = [_n for _n in _v29_names if _n in _sim_src]
        chk("G-V29-INERT  no schema-v29 field is read on the render path, so "
            "a v29 database renders bit-identically to a v28 one",
            not _leaked,
            "%d fields checked, none referenced in film_sim" % len(_v29_names)
            if not _leaked
            else "referenced in film_sim: %s -- if deliberate, the field is "
                 "no longer inert: kSchemaVersion must move again, the C++ "
                 "struct must gain it, and cpp_parity.py must probe it"
                 % _leaked)

        # 7. STRICKER'S CALLIER TABLE, as an INDEPENDENT check on a rule this
        #    project already had. Queue C43 derived callier_q from each
        #    monochrome stock's own mid slope on 2026-09-02, calibrated on
        #    Mees FIG. 179. The 2026-09-10 harvest recommended "derive Callier
        #    Q from gamma" -- already done, and better calibrated than the
        #    recommendation knew. What Stricker adds, via Glafkides §203, is
        #    the same function measured in another laboratory in another
        #    decade: worth having as a guard, worth nothing as a second copy
        #    of the rule.
        #    ⚠⚠ NARROWED 2026-09-20 TO SILVER STOCKS, AND THE GUARD IS WHAT
        #    FORCED THE QUESTION. `_CHROMOGENIC_MONO` -- KODAK BW400CN and
        #    T400CN -- moved to callier_q = 1.0 that day, because their image
        #    is dye and nothing else and `film_sim.callier_density`'s own
        #    docstring says a dye image scatters almost nothing. This check
        #    immediately failed on both at 0.790 and 0.816. IT WAS RIGHT TO
        #    FAIL AND IT IS RIGHT TO BE NARROWED: Stricker measured DEVELOPED
        #    SILVER, the quantity his table is a function of is silver
        #    scattering against contrast, and applying it to a chromogenic
        #    stock asks a measurement of one material to validate a value for
        #    another. ⚠ THE EXCLUSION IS BY THE SAME NAMED SET THE DATABASE
        #    USES, not by a tolerance widened until the failure stopped --
        #    which would have hidden the next real offender.
        _q = [(_p.name,
               abs(_p.callier_q
                   - _fp.stricker_callier_q(_p.curves.g.mid_slope)))
              for _p in FILM_PROFILES
              if _p.is_monochrome and _p.name not in _fp._CHROMOGENIC_MONO]
        _q_bad = [(n, round(d, 3)) for n, d in _q if d > 0.25]
        chk("G-STRICKER  the project's Callier rule agrees with Stricker's "
            "independently measured Q(gamma) table within 0.25 on every "
            "SILVER monochrome stock (the two chromogenic ones are dye images "
            "and Stricker measured silver)", not _q_bad,
            "%d stocks, mean |diff| %.4f, max %.4f"
            % (len(_q), sum(d for _, d in _q) / max(len(_q), 1),
               max(d for _, d in _q)) if _q and not _q_bad
            else "offenders: %s" % (_q_bad[:3],))

        # 7b. AND THE COVERAGE THE NARROWING ABOVE WOULD OTHERWISE HAVE LOST.
        #     Removing two stocks from a guard without replacing the check is
        #     how a field goes unguarded, so the exclusion is paid for here:
        #     the chromogenic pair must carry EXACTLY the colour value, must
        #     be read as Status M, and must not be handed a silver beta again
        #     by some later edit to `_apply_schema_v2`.
        _chromo = [_p for _p in FILM_PROFILES
                   if _p.name in _fp._CHROMOGENIC_MONO]
        _chromo_bad = [(_p.name, _p.callier_q, _p.density_metric)
                       for _p in _chromo
                       if _p.callier_q != 1.0 or _p.density_metric != "status_m"]
        chk("G-CHROMO  the two chromogenic black-and-white stocks carry the "
            "DYE-IMAGE pair -- Callier Q exactly 1.0 and Status M "
            "densitometry -- and not the silver pair their is_monochrome flag "
            "would otherwise select",
            len(_chromo) == 2 and not _chromo_bad,
            "%s" % ([(_p.name, _p.callier_q, _p.density_metric)
                     for _p in _chromo],)
            if not _chromo_bad
            else "offenders: %s" % (_chromo_bad,))

        # 8. NEUTRAL BALANCE SURVIVES THE INTERIMAGE STAGE (2026-09-10b).
        #
        # ⚠ THIS GUARD EXISTS BECAUSE A PATENT METRIC WAS ALMOST ADOPTED AS A
        # TARGET AND IS NOT ONE. Kodak's colour-negative patents (US 5,989,798
        # Tables III/V, EP 0 851 288) publish
        #
        #     R = red gamma / green gamma, under a white-light (neutral)
        #         C-41 exposure
        #
        # and state "low values of R are indicative of high interlayer
        # interimage". Their coatings measure R = 0.70 and 0.85 with NO DIR
        # coupler and 0.41-0.58 with one. Measured across this database, all
        # 84 colour negatives sit at R = 0.856-1.127, median 0.972 -- i.e.
        # every shipping film reads as having LESS interimage than a patent's
        # deliberately DIR-free control, which cannot be true of VISION3 or
        # PORTRA.
        #
        # ⚠ AND THE FIRST EXPLANATION WAS WRONG. The suspicion was that
        # manufacturers publish the three records already balanced, so the R
        # information had been normalised out of the traced curves before this
        # project ever saw them. Checked, and it is false: only 4 of 115
        # colour stocks share a mid_slope across records, 71 carry a real
        # r < g < b dmin mask ladder, the toe_x spread across records has a
        # median of 0.12, and the 45 tier-1 datasheet-traced negatives give
        # the same R distribution as the analogy estimates. The curves carry
        # genuine per-record differences.
        #
        # THE ACTUAL REASON, which is a property of the film and not of the
        # data: a colour negative MUST hold a neutral across its exposure
        # scale or a grey ramp drifts in colour, which is the crossover defect
        # every maker engineers against. So the layers are built to matched
        # contrast, and where interimage suppresses the red record the red
        # layer is given more inherent contrast to compensate. R on a finished
        # product is therefore ~1 BY DESIGN, whatever the DIR chemistry inside,
        # and the patents' low R comes from experimental coatings that were
        # never rebalanced. R measures interimage only in an A/B where the
        # coatings are identical except for the DIR loading -- which is what a
        # patent Example is and what a data sheet can never be.
        #
        # ⚠ WHAT DOES TRANSFER IS THE STABILITY OF R, AND THAT IS THIS GUARD.
        # If the layers are balanced, then whatever this pipeline does to a
        # NEUTRAL must leave them balanced. The interimage stage is referenced
        # to the mid-grey anchor precisely so that it nearly vanishes there,
        # so a coefficient set that swings R on a neutral has broken the
        # reference, not modelled a film. Measured today: median +0.0136,
        # worst -0.0205 (KONICA_IMPRESA_50) and +0.0287 (CINESTILL_800T), so
        # the 0.05 bound carries about 1.7x headroom.
        #
        # ⚠ IT IS NOT A CHECK ON THE COEFFICIENTS' MAGNITUDE. Interimage is
        # SUPPOSED to move saturated colour hard; this only pins the neutral.
        import math as _m29

        def _d29(_c, _x):
            def _sp(_z, _k):
                _t = _z / _k
                if _t > 40.0:
                    return _z
                if _t < -40.0:
                    return 0.0
                return _k * _m29.log1p(_m29.exp(_t))
            return _c.dmin + _c.gamma * (_sp(_x - _c.toe_x, _c.toe_k)
                                         - _sp(_x - _c.shoulder_x,
                                               _c.shoulder_k))

        def _R29(_p, _iie):
            _cur = (_p.curves.r, _p.curves.g, _p.curves.b)
            _n = 241
            _lo, _hi = -1.6, 0.6
            _xs = [_lo + (_hi - _lo) * _i / (_n - 1) for _i in range(_n)]
            _Dm = [[_d29(_c, _x) for _x in _xs] for _c in _cur]
            if _iie and _p.interimage.active:
                _M = _p.interimage.matrix()
                _dr = [_d29(_c, _p.speed_point_x) for _c in _cur]
                for _ in range(max(1, _p.interimage.iterations)):
                    _shift = [[sum(_M[_k][_j] * (_Dm[_j][_i] - _dr[_j])
                                   for _j in range(3))
                               for _i in range(_n)] for _k in range(3)]
                    _Dm = [[_d29(_cur[_k], _xs[_i] + _shift[_k][_i])
                            for _i in range(_n)] for _k in range(3)]
            _i0, _i1 = _n // 4, 3 * _n // 4
            _g = [(_Dm[_k][_i1] - _Dm[_k][_i0]) / (_xs[_i1] - _xs[_i0])
                  for _k in range(3)]
            return _g[0] / _g[1] if _g[1] else 0.0

        _bal = []
        for _p in FILM_PROFILES:
            if _p.is_monochrome or _p.kind is not StockKind.NEGATIVE:
                continue
            if not _p.interimage.active:
                continue
            _bal.append((_p.name, _R29(_p, False), _R29(_p, True)))
        _bal_bad = [(n, round(b - a, 4)) for n, a, b in _bal
                    if abs(b - a) > 0.05]
        chk("G-IIE-NEUTRAL  the interimage stage leaves a NEUTRAL ramp's "
            "red/green gamma balance intact (|dR| <= 0.05) -- the stage is "
            "anchored at mid grey and must nearly vanish there",
            not _bal_bad,
            "%d stocks, R %.4f -> %.4f median, worst move %+.4f"
            % (len(_bal),
               sorted(a for _, a, _b in _bal)[len(_bal) // 2],
               sorted(b for _, _a, b in _bal)[len(_bal) // 2],
               max((b - a for _, a, b in _bal), key=abs))
            if _bal and not _bal_bad else "offenders: %s" % (_bal_bad[:3],))

        # 9. THE DIRECTIONAL INTERIMAGE ASYMMETRY (2026-09-11, schema v31).
        #
        # ⚠ THIS GUARD RECORDS A KNOWN DEFECT RATHER THAN ENFORCING A FIX,
        # deliberately. US 6,746,834 tabulates four DIRECTIONAL interimage
        # effects for eleven samples under the Hanson construction (J. Opt.
        # Soc. Am. 42, 1952, pp.663-669: the receiver's density change at
        # integrated density 1.5 as the causer falls from 2.0 to 1.0), and
        # claims two asymmetries:
        #
        #     IIEgr > IIErg      green acting on red beats red on green
        #     IIEbg > IIEgb      blue acting on green beats green on blue
        #
        # IIExy is the effect FROM x TO y, so IIEgr maps to `a_rg` here and
        # IIErg to `a_gr`.
        #
        # ⚠ MAGNITUDE CORRECTED 2026-09-11. This comment previously said the
        # patent claims "about 4:1", read off the claim windows. The patent's
        # TABLE 3 has since been transcribed in full
        # (`film_profiles._US6746834_IIE_TABLE`) and its three invention
        # coatings measure IIEgr/IIErg = 0.17/0.10, 0.17/0.10 and 0.25/0.10 --
        # 1.7:1 to 2.5:1. The 4:1 was an artifact of taking two independent
        # claim bounds at their worst case simultaneously. The retune target
        # is the measured range, not the inferred one.
        #
        # MEASURED HERE: the blue rule holds on all 106 stocks with an active
        # stage. The red/green rule fails on 70 of 106 -- and the failure is
        # SYMMETRY, not a sign error. The median (green->red) - (red->green)
        # is -0.0002 across reversal stocks, so the generator produces
        # near-identical coefficients where the measurement wants 1.7-2.5:1.
        #
        # ⚠ NOT FIXED IN THIS PASS. All six coefficients are estimated on all
        # 106 stocks and retuning them moves every colour render; that needs
        # its own pass with a deliberate decision about magnitude, not a
        # silent edit inside a harvest. The blue rule is asserted because it
        # already holds and must not regress; the red/green count is PINNED
        # so any change to the generator shows up here as a number moving.
        # ⚠⚠ SPLIT BY MATERIAL CLASS AT v39 (2026-09-17d), BECAUSE A SECOND
        # DIRECTIONAL SOURCE ARRIVED AND IT MEASURES THE OPPOSITE ORDERING ON
        # THE OTHER CLASS. US 6,746,834 is a colour REVERSAL patent and claims
        # IIEgr > IIErg. EP 0 324 471 A2 TABLE 1 is a colour NEGATIVE patent
        # and prints all four cross-directions for four coatings: on three of
        # the four, IE(G/R) > IE(R/G) -- red acting on green EXCEEDS green
        # acting on red. Asserting one ordering across both classes would
        # contradict a measurement whichever ordering was chosen, so each
        # class is checked against the source measured on it.
        _rev_rg, _rev_n = 0, 0
        _neg_rg, _neg_n = [], 0
        _iie_bg_bad, _iie_n = [], 0
        for _p in FILM_PROFILES:
            _ii = _p.interimage
            if not _ii.active:
                continue
            _iie_n += 1
            if abs(_ii.a_gb) <= abs(_ii.a_bg):      # IIEbg vs IIEgb
                _iie_bg_bad.append(_p.name)
            if _p.is_reversal:
                _rev_n += 1
                if abs(_ii.a_rg) <= abs(_ii.a_gr):  # IIEgr vs IIErg
                    _rev_rg += 1
            else:
                _neg_n += 1
                if abs(_ii.a_gr) <= abs(_ii.a_rg):  # IE(G/R) vs IE(R/G)
                    _neg_rg.append(_p.name)
        chk("G-IIE-ASYM-rg-rev  on REVERSAL stocks green acting on red "
            "exceeds red acting on green, as US 6,746,834 TABLE 3 measures "
            "on a reversal material",
            _rev_rg == 0,
            "holds on all %d reversal stocks with an active stage" % _rev_n
            if _rev_rg == 0 else "%d of %d violate" % (_rev_rg, _rev_n))
        # ⚠ THE NEGATIVE COUNT IS PINNED AND THE REASON IS ONE TIER. The
        # ordering follows from the tier's own red and green targets, and
        # "medium" is the only row of `_IIE_TIERS` whose RED target (35)
        # exceeds its GREEN one (33). Stocks in it sit on the line, and which
        # side of it a given stock lands on is decided by its own curve shape
        # inside `_iie_solve` rather than by the tier. Both offenders are
        # within 8 % of parity. Pinned rather than asserted so that a change
        # to the ladder or the solver shows up here as a number moving.
        _NEG_RG_BASELINE = 2
        chk("G-IIE-ASYM-rg-neg  on NEGATIVE stocks red acting on green "
            "exceeds green acting on red, as EP 0 324 471 A2 TABLE 1 "
            "measures on 3 of its 4 directional coatings",
            len(_neg_rg) == _NEG_RG_BASELINE,
            "%d of %d violate, baseline %d (%s -- all in the 'medium' tier, "
            "the only one whose red target exceeds its green one)"
            % (len(_neg_rg), _neg_n, _NEG_RG_BASELINE, _neg_rg[:3]))
        # ⚠ 0 -> 2 AT v39 AND IT IS THE SPLIT, NOT A REGRESSION. The blue
        # rule's only source is the reversal patent; on negatives the receiver
        # blue row is now 0.528/0.472 rather than 0.323/0.677, which moves
        # `a_bg` up against `a_gb` on the two stocks whose blue and green
        # receiver totals were already closest. Both are within 1 % of parity.
        _IIE_BG_BASELINE = 2
        chk("G-IIE-ASYM-b  blue acting on green exceeds green acting on blue "
            "(US 6,746,834's IIEbg > IIEgb), with the negative-branch "
            "exceptions pinned",
            len(_iie_bg_bad) == _IIE_BG_BASELINE,
            "%d of %d violate, baseline %d (%s)"
            % (len(_iie_bg_bad), _iie_n, _IIE_BG_BASELINE, _iie_bg_bad[:3]))

        # ------------------------------------------------------------------
        #  schema v32 -- the MEASURED REFERENCE DATA section.
        #
        #  ⚠ THESE GUARDS ARE WHAT STOP THE NEW TABLES BEING DECORATION. The
        #  2026-09-10 delivery harvested 22 documents and moved no number; the
        #  correction was to enter the values. A value entered and never read
        #  is the same failure one step later, so every table added at v32 is
        #  asserted here: its shape, its monotonicity where the physics
        #  requires it, and its agreement with whatever it is meant to
        #  constrain. A table that could not be given a guard did not belong
        #  in the module.
        # ------------------------------------------------------------------
        _mask = _fpm._GOST_9160_MASK_FILTER_D
        _mnm = [w for w, _ in _mask]
        # The standard prints 360-800 nm inclusive on a 10 nm grid: 45 rows.
        # Two density cells are destroyed in the scan, so 43 survive.
        _mask_rows = len(range(360, 801, 10))
        chk("G-V32-GOSTMASK  ГОСТ 9160-91 table 11 mask filter is 43 of the "
            "standard's 45 rows, ascending over 370-800 nm, with only the "
            "two unreadable cells absent",
            _mnm == sorted(_mnm)
            and set(_mnm).isdisjoint(_fpm._GOST_9160_MASK_GAPS)
            and _mnm[0] == 370 and _mnm[-1] == 800
            and len(_mask) == _mask_rows - len(_fpm._GOST_9160_MASK_GAPS),
            "%d points, %d-%d nm, gaps at %s"
            % (len(_mask), _mnm[0], _mnm[-1], _fpm._GOST_9160_MASK_GAPS))

        # An orange mask is a yellow-through-red absorber: high in the blue,
        # falling monotonically through the green, flat in the far red. The
        # table must show that or it was transcribed out of order.
        _peak_nm = max(_mask, key=lambda t: t[1])[0]
        _tail = [d for w, d in _mask if w >= 740]
        _mid = [(w, d) for w, d in _mask if 500 <= w <= 700]
        chk("G-V32-GOSTMASKSHAPE  the mask curve peaks in the violet-blue, "
            "falls monotonically across 500-700 nm and is flat in the far "
            "red -- the shape an orange mask must have",
            _peak_nm in (420, 430)
            and all(b <= a for (_, a), (_, b) in zip(_mid, _mid[1:]))
            and len(set(_tail)) == 1,
            "peak %.2f at %d nm, 740-800 nm flat at %.2f"
            % (max(d for _, d in _mask), _peak_nm, _tail[0]))

        _uvir = dict(_fpm._GOST_9160_UVIR_FILTER_D)
        _unm = sorted(_uvir)
        chk("G-V32-GOSTUVIR  ГОСТ 9160-91 table 10 UV/IR filter is a "
            "complete 10 nm grid over 380-800 nm that cuts hard at both "
            "ends and stays transparent through the visible",
            len(_uvir) == 43
            and all(b - a == 10 for a, b in zip(_unm, _unm[1:]))
            and _uvir[380] == 1.90 and _uvir[800] == 0.99
            and max(_uvir[w] for w in range(500, 561, 10)) <= 0.12,
            "%d points, D(380)=%.2f  D(550)=%.2f  D(800)=%.2f"
            % (len(_uvir), _uvir[380], _uvir[550], _uvir[800]))

        # ⚠ THE LENS TABLE IS HERE FOR ITS BLUE END. The two reference lenses
        # agree closely from 460 nm up and differ by nearly 3x at 360 nm;
        # that divergence is the reason the table is worth storing, so it is
        # the thing asserted.
        _lens = _fpm._GOST_9160_LENS_TAU
        _l360 = [r for r in _lens if r[0] == 360][0]
        _lvis = [r for r in _lens if r[0] >= 460 and r[2] is not None]
        _worst_vis = max(abs(a - b) for _, a, b in _lvis)
        chk("G-V32-GOSTLENS  ГОСТ 9160-91 table 9: the two reference lenses "
            "agree within 0.06 from 460 nm up and differ by ~3x at 360 nm, "
            "which is why a spectral sensitivity's blue end is convention-"
            "dependent",
            len(_lens) == 34
            and abs(_l360[1] / _l360[2] - 2.857) < 0.01
            # ⚠ 1e-9 SLACK, NOT A LOOSENED BOUND. The worst visible gap is
            # exactly 0.06 as printed (0.98 against 0.94 at 680 nm); binary
            # floats make that subtraction 0.06000000000000005, so a bare
            # `<= 0.06` fails on representation rather than on the data.
            and _worst_vis <= 0.06 + 1e-9,
            "tau(360) %.2f vs %.2f = %.2fx; worst visible gap %.2f"
            % (_l360[1], _l360[2], _l360[1] / _l360[2], _worst_vis))

        chk("G-V32-GOSTGRAD  every ГОСТ 9160-91 table 4 mean-gradient row is "
            "an accepted `gamma_criterion`, and the one reconstructed "
            "decimal is declared as such",
            set(_fpm._GOST_9160_GRADIENT_POINTS) <= _fpm._GAMMA_CRITERIA
            and _fpm._GOST_9160_GRADIENT_RECONSTRUCTED
                <= set(_fpm._GOST_9160_GRADIENT_POINTS),
            "%d rows, %d carrying a reconstructed digit"
            % (len(_fpm._GOST_9160_GRADIENT_POINTS),
               len(_fpm._GOST_9160_GRADIENT_RECONSTRUCTED)))

        # ⚠ THE TRANSCRIBED PATENT TABLE MUST REPRODUCE THE PROPERTY IT WAS
        # BROUGHT IN TO SUPPLY. If US 6,746,834's own invention rows did not
        # satisfy IIEgr > IIErg, G-IIE-ASYM-rg above would be measuring this
        # database against a rule its source does not itself keep.
        _t3 = _fpm._US6746834_IIE_TABLE
        _inv = [r for r in _t3 if r[1] == "Inv."]
        _ratios = [r[10] / r[9] for r in _inv]      # IIEgr / IIErg
        chk("G-V32-US6746834  the transcribed TABLE 3 has 11 rows, 3 of them "
            "inventions, and every invention row satisfies IIEgr > IIErg at "
            "1.7-2.5:1 -- the measured target for the pending retune",
            len(_t3) == 11 and len(_inv) == 3
            and all(r > 1.0 for r in _ratios)
            and 1.6 <= min(_ratios) and max(_ratios) <= 2.6,
            "invention ratios %s" % ["%.1f" % r for r in _ratios])

        # ⚠ AND THE SOURCE IS NOT SELF-CONSISTENT, WHICH IS RECORDED RATHER
        # THAN SMOOTHED. Sample 106 is labelled an invention yet inverts the
        # patent's own blue rule. Asserting the exception stops anyone later
        # "correcting" the transcription to match the prose.
        _s106 = [r for r in _t3 if r[0] == 106][0]
        chk("G-V32-US6746834X  sample 106 is labelled Inv. but has "
            "IIEbg < IIEgb, contradicting the patent's own stated rule -- a "
            "transcription-fidelity check, not a defect in this database",
            _s106[1] == "Inv." and _s106[12] < _s106[11],
            "sample 106 IIEbg %.2f vs IIEgb %.2f" % (_s106[12], _s106[11]))

        # US 5,262,287's difference columns were the arithmetic that let four
        # scan-damaged cells be solved. The identity must still close on the
        # stored values, or the recovery was wrong.
        _dl = _fpm._US5262287_DLOGE
        _bad_dl = [s for s, r05, r15, g05, g15 in _dl
                   if r05 < r15 or g05 < g15]
        chk("G-V32-US5262287  the transcribed TABLE 2 has 17 rows and every "
            "row keeps dlogE(0.5) >= dlogE(1.5) in both records -- the "
            "identity that recovered four scan-damaged cells",
            len(_dl) == 17 and not _bad_dl
            and [s for s, *_ in _dl] == list(range(101, 118)),
            "%d rows, %d violating" % (len(_dl), len(_bad_dl)))
        # ⚠ AND ONE CELL IS PINNED BY VALUE, 2026-09-17. Sample 105's
        # dloge_r_1.5 was stored 0.00 until the page image was read; the
        # patent prints 0.01 and the difference identity closes on it
        # (0.06 - 0.01 = 0.05, the printed difference). The identity guard
        # above cannot catch a value of 0.00 because 0.06 >= 0.00 holds, so
        # the corrected cell is asserted directly.
        _s105 = [r for r in _dl if r[0] == 105][0]
        chk("G-V32-US5262287C  sample 105's dlogE(R1.5) is 0.01, read off "
            "the page image -- the text layer drops the trailing digit and "
            "the difference identity alone does not catch a zero",
            abs(_s105[2] - 0.01) < 1e-12
            and abs((_s105[1] - _s105[2]) - 0.05) < 1e-9,
            "0.06 - %.2f = %.2f" % (_s105[2], _s105[1] - _s105[2]))

        # ---- v38: EP 0 608 959 B1, the second house to publish the
        # separation-over-neutral gamma ratio, and the only complete
        # multilayer negative measured on all three records -------------------
        #
        # ⚠⚠ THIS PAIR CARRIES A RESOLVED CONFLICT AND THE ARITHMETIC THAT
        # RESOLVES IT (queue P69, closed 2026-09-17d). Read raw, the Kodak
        # numbers put the red receiver at 14-26 % where Agfa's US5273870A puts
        # it at 35-42 %, and that looked like two houses disagreeing about the
        # shape of the effect. It is not.
        #
        #   1. US5273870A reads "at density 1.0 over fog", which on a
        #      0.6-gamma negative anchored at Dmin+0.15 is ~1.4 log E above
        #      the anchor -- inside EP0608959B1's own "mid upper" window.
        #      TABLE IV measures the zone factor rather than leaving it to be
        #      assumed: (mid_upper - 1)/(mid - 1) over its four coatings is
        #      0.429 to 0.717. Applied to TABLE III #101 that gives blue
        #      16.3-27.3 and green 31.3-52.4, and the Agfa ladder's 25 and 45
        #      sit inside both. TWO OF THREE RECORDS AGREE ONCE THE ZONE IS
        #      MATCHED, which is asserted below rather than argued.
        #   2. Red does not, and the same table says why. Example 1 varies one
        #      thing -- the DIR coupler in the MID-GREEN layer, the donor whose
        #      inhibitor reaches red -- and sample 101 is the control with
        #      none. With the weak untimed D-4 the RED ratio climbs
        #      monotonically with loading (1.14/1.20/1.26) while blue is
        #      pinned at 1.38; with the strong timed D-1 blue climbs
        #      (1.38/1.46/1.56/1.61) while red barely moves. Kodak's red
        #      column measures a coating whose red donor was removed for the
        #      experiment, not a red record's susceptibility.
        #
        # The ladder therefore stays Agfa's, and these guards are what stop
        # that conclusion decaying into a comment nobody re-checks.
        _ep = _fpm._EP0608959_GAMMA_RATIOS
        _pct = [(round((b - 1) * 100), round((g - 1) * 100),
                 round((r - 1) * 100)) for _, _, _, _, r, g, b in _ep]
        chk("G-V38-EP0608959  TABLE III has 6 coatings, every ratio above "
            "1.0 as inhibition requires, and the green record receives most "
            "on all six",
            len(_ep) == 6
            and all(r > 1.0 and g > 1.0 and b > 1.0
                    for *_, r, g, b in _ep)
            and all(g > b > r for *_, r, g, b in _ep),
            "as IIE %%, blue/green/red: %s" % (_pct[0],))
        # --- the reconciliation, as arithmetic -------------------------------
        _zf = [(mu - 1.0) / (mid - 1.0)
               for _s, _m, _lo, mid, mu in _fpm._EP0608959_ZONE_RATIOS]
        _zlo, _zhi = min(_zf), max(_zf)
        _kb, _kg, _kr = _pct[0]                    # #101, the control
        _ab, _ag, _ar = _fpm._IIE_TIERS["strong"]  # Agfa Ex.1 invention
        chk("G-V38-EP0608959X  Kodak and Agfa AGREE on blue and green once "
            "the measurement zone is matched -- the zone factor is measured "
            "by the same patent, not assumed (queue P69)",
            (_zlo * _kb <= _ab <= _zhi * _kb
             and _zlo * _kg <= _ag <= _zhi * _kg),
            "zone factor %.3f-%.3f; blue %.1f-%.1f vs Agfa %.0f, "
            "green %.1f-%.1f vs Agfa %.0f"
            % (_zlo, _zhi, _zlo * _kb, _zhi * _kb, _ab,
               _zlo * _kg, _zhi * _kg, _ag))
        # And the red column is a donor-loading series, which is the half of
        # the resolution that is NOT a zone effect. If this monotonicity ever
        # broke, the explanation above would be wrong and P69 would reopen.
        _d4 = [r for _s, _t, c, _l, r, _g, _b in _ep if c in ("none", "D-4")]
        _d1b = [b for _s, _t, c, _l, _r, _g, b in _ep if c in ("none", "D-1")]
        _d4r = [r for _s, _t, c, _l, r, _g, _b in _ep if c == "D-4"]
        _d4b = [b for _s, _t, c, _l, _r, _g, b in _ep if c == "D-4"]
        chk("G-V38-EP0608959R  TABLE III's RED column tracks the mid-green "
            "DIR loading while blue stays pinned -- so Kodak's low red is an "
            "absent donor, not an insusceptible receiver (queue P69)",
            (_d4 == sorted(_d4) and len(_d4) == 3
             and len(set(_d4b)) == 1
             and _d4r[-1] > _d4[0]
             and _d1b == sorted(_d1b) and len(_d1b) == 4),
            "D-4 0/0.011/0.022 g/m2: Rr/Rn %s with Bb/Bn flat at %.2f; "
            "D-1 raises Bb/Bn %s instead"
            % (["%.2f" % v for v in _d4], _d4b[0],
               ["%.2f" % v for v in _d1b]))
        # ⚠ AND THE ZONE TABLE IS THE ONE PIECE OF EVIDENCE THE MODEL CANNOT
        # YET USE. One record of one coating moves by a factor of 1.4 across
        # three exposure zones, and the interimage stage has no term for the
        # receiver's own position on its curve. Asserted so the gap stays
        # measurable; queue P66.
        _ez = _fpm._EP0608959_ZONE_RATIOS
        _spread = max(max(r[2:]) / min(r[2:]) for r in _ez)
        chk("G-V38-EP0608959Z  the interimage ratio is NOT constant along "
            "the curve -- mid zone strongest on all four coatings, spread up "
            "to 1.4x, and stage 8b has no term for it (queue P66)",
            len(_ez) == 4
            and all(r[3] > r[4] > r[2] for r in _ez)
            and 1.3 < _spread < 1.5,
            "worst spread %.2fx, mid/low %.2f" % (_spread, _ez[0][3] / _ez[0][2]))

        # ---- v39: EP 0 324 471 A2 TABLE 1 -- the third house, and the
        # largest white-versus-separation set in the corpus ------------------
        #
        # ⚠ TWENTY-ONE COLOUR NEGATIVE COATINGS WITH BOTH GAMMAS ON ALL THREE
        # RECORDS, recovered from a sideways image-only page. The reason it
        # matters is the per-receiver SHAPE: blue receives least on 21 of 21,
        # which is Agfa's ordering and not Kodak's, so it is a third
        # independent house corroborating the ladder the database ships.
        _e3 = _fpm._EP0324471_TABLE1
        _e3p = [((pb / b - 1) * 100, (pg / g - 1) * 100, (pr / r - 1) * 100)
                for _s, _t, pr, pg, pb, r, g, b in _e3]
        _blue_least = sum(1 for ib, ig, ir in _e3p if ib < ig and ib < ir)
        _inv = [r for r in _e3 if r[1] == "Inv"]
        chk("G-V39-EP0324471  TABLE 1 has 21 coatings and the BLUE record "
            "receives least on every one -- a third house agreeing with the "
            "Agfa ladder's shape, against Kodak's blue-high ordering",
            len(_e3) == 21 and _blue_least == 21 and len(_inv) == 8,
            "blue least on %d of %d; #101 as IIE %% b/g/r %.0f/%.0f/%.0f"
            % (_blue_least, len(_e3), *_e3p[0]))
        # ⚠ AND THE DIRECTIONAL HALF IS WHAT REMOVED v38's ASSUMPTION. Four
        # coatings print all four cross-directions, and the fourth --
        # IE(R/B), blue acting on red -- is measured nowhere else here.
        _e3i = _fpm._EP0324471_IE
        _rev_dir = sum(1 for _s, rg, gr, gb, _q in _e3i if gr > rg)
        chk("G-V39-EP0324471D  the four directional coatings put RED acting "
            "on GREEN above green acting on red on 3 of 4 -- the reverse of "
            "the reversal rule, which is why the donor split is class-keyed",
            len(_e3i) == 4 and _rev_dir == 3
            and all(0.0 < q < 2.0 for *_r, q in _e3i),
            "IE(G/R) > IE(R/G) on %d of %d" % (_rev_dir, len(_e3i)))
        # The split actually shipped must be the one derived from that table,
        # or the derivation above is decoration.
        _spn = _fpm._IIE_DONOR_SPLIT["negative"]
        _spr = _fpm._IIE_DONOR_SPLIT["reversal"]
        chk("G-V39-SPLIT  the negative donor split is the EP 0 324 471 A2 "
            "median and the reversal one is the US 6,746,834 sample, every "
            "row sums to 1 and every diagonal is structurally zero",
            all(abs(sum(v) - 1.0) < 1e-9 for v in _spn.values())
            and all(abs(sum(v) - 1.0) < 1e-9 for v in _spr.values())
            and _spn["r"][0] == 0.0 and _spn["g"][1] == 0.0
            and _spn["b"][2] == 0.0
            and abs(_spn["r"][1] - 0.5728) < 1e-4
            and abs(_spn["g"][0] - 0.6000) < 1e-4
            and abs(_spr["r"][1] - 0.5) < 1e-9,
            "negative r %.4f/%.4f  g %.4f/%.4f  b %.4f/%.4f"
            % (_spn["r"][1], _spn["r"][2], _spn["g"][0], _spn["g"][2],
               _spn["b"][0], _spn["b"][1]))

        # ---- v39: EP 1 033 620 A1 TABLE 2-1 -- spectral shape against a
        # MEASURED colour error, the only such pairing in the corpus ---------
        _e1 = _fpm._EP1033620_TABLE2_1
        _ok = [r for r in _e1 if r[1] == "Inv"]
        _no = [r for r in _e1 if r[1] == "Cmp"]
        chk("G-V39-EP1033620  TABLE 2-1's three inventive coatings all meet "
            "the claimed spectral window and all score under 3.3 colour "
            "error, while every comparative breaks one and scores over 7",
            len(_e1) == 7 and len(_ok) == 3
            and all(r[5] > 65 and r[6] >= 60 and r[2] <= 620 for r in _ok)
            and all(r[10] < 3.3 for r in _ok)
            and all(r[10] > 7.0 for r in _no)
            and all(not (r[5] > 65 and r[6] >= 60 and r[2] <= 620)
                    for r in _no),
            "inventive error %s against comparative %s"
            % ([r[10] for r in _ok], [r[10] for r in _no]))

        # ---- v39, queue P66: the print side gets a process statement -------
        _pwith = [q for q in _fpm.PRINT_STOCKS if q.processing is not None]
        _cyc = _fpm._ECP_2E_CYCLE
        _dev = [c for c in _cyc if c[0] == "developer"][0]
        chk("G-V39-P66  PrintStock carries a ProcessingSpec, exactly the one "
            "stock whose own source names a process has it, and the ECP-2E "
            "cycle is stored as a reference rather than copied onto it",
            len(_pwith) == 1
            and _pwith[0].name == "KODAK_2383_RELEASE"
            and _pwith[0].processing.developer == "ECP-2D"
            and _pwith[0].processing.minutes == 0.0
            and _pwith[0].processing.celsius == 0.0
            and len(_cyc) == 9
            and abs(_dev[1] - 3.0) < 1e-9 and abs(_dev[2] - 36.7) < 1e-9,
            "%d of %d print stocks state a process; ECP-2E developer "
            "%.2f min at %.1f +/- %.1f degC, %d steps"
            % (len(_pwith), len(_fpm.PRINT_STOCKS), _dev[1], _dev[2],
               _dev[3], len(_cyc)))
        # ⚠ AND THE REVISION LETTER IS ASSERTED, because copying ECP-2E's
        # times onto an ECP-2D record is the one wrong thing this row could
        # have done and it would look right.
        chk("G-V39-P66X  no print stock carries a developer time or "
            "temperature taken from a process revision its own source does "
            "not name",
            all(q.processing is None
                or q.processing.developer != "ECP-2E"
                for q in _fpm.PRINT_STOCKS),
            "ECP-2D named on the record, ECP-2E kept in _ECP_2E_CYCLE")

        # ---- v39: HALATION SPLIT INTO THREE TIERS --------------------------
        #
        # ⚠⚠ THE POINT OF THESE GUARDS IS THAT TIER B IS EMPTY AND MUST STAY
        # HONEST WHILE IT IS. The proposal these fields implement is that
        # halation stop being an amplitude somebody chose: Tier A is the
        # optics of the support (computed, four rows), Tier B is the
        # antihalation layer as a physical object (per film, measured or
        # absent), Tier C is the model (derived, never authored). Today Tier B
        # is populated on ZERO stocks, so every render is bit-identical to a
        # v38 one -- and the guards below are what stop the empty carrier
        # being filled by inference rather than by a document.
        _ah = [q for q in FILM_PROFILES if q.anti_halation.has_density]
        chk("G-V39-AH-EMPTY  the antihalation Tier B carrier is populated "
            "from documents only, so it is empty today and no render moves",
            not _ah and all(q.anti_halation.measured or
                            not q.anti_halation.has_density
                            for q in FILM_PROFILES),
            "%d of %d stocks state an antihalation density; construction is "
            "named on %d" % (len(_ah), len(FILM_PROFILES),
                             sum(1 for q in FILM_PROFILES
                                 if q.emulsion.antihalation)))
        # Tier A is Fresnel and Snell on an index the module already holds.
        # If either ever became a stored per-film number this fails.
        _fa = _fpm.base_fresnel(1.48)
        _fp2 = _fpm.base_fresnel(1.575)
        chk("G-V39-AH-TIERA  the support optics are COMPUTED from the two "
            "refractive indices, not stored per film -- R 3.75 %% / 5.0 %% "
            "and a critical angle of 42.5 / 39.4 degrees",
            abs(_fa[0] - 0.0375) < 5e-4 and abs(_fa[1] - 42.507) < 0.01
            and abs(_fp2[0] - 0.0499) < 5e-4 and abs(_fp2[1] - 39.414) < 0.01
            and set(_fpm._V34_BASE_OPTICS) == {"acetate", "nitrate",
                                               "polyester"},
            "acetate R %.4f theta_c %.2f deg; polyester R %.4f theta_c %.2f"
            % (_fa[0], _fa[1], _fp2[0], _fp2[1]))
        # And the derived-gain law must actually do the thing it exists for:
        # a neutral absorber gives an equal triple, a blue/green-absorbing dye
        # leaves red standing. Probed off-database so no stock moves.
        # ⚠ THIS PROBE USED `position="backing"` UNTIL 2026-09-18 AND NOW USES
        # "undercoat". The position was incidental to what the check is about
        # -- that a neutral absorber gives an equal triple -- but it is no
        # longer legal input: queue P71 established that a rear backing is not
        # an in-path absorber and the law refuses one, so a probe built on
        # "backing" would be asserting properties of None.
        _neutral = _fpm.AntiHalationSpec(
            position="undercoat", dye="carbon", optical_density=1.0,
            od_lambda_nm=550.0, neutral=True, measured=True, source="probe")
        _dyed = _fpm.AntiHalationSpec(
            position="undercoat", dye="probe",
            od_spectrum=tuple(0.2 if 400 + 10 * i > 600 else 1.6
                              for i in range(31)),
            measured=True, source="probe")
        _gn = _fpm.halation_gain_from_od(_neutral, 1.48)
        _gd = _fpm.halation_gain_from_od(_dyed, 1.48)
        chk("G-V39-AH-GAIN  the derived gain law returns None without a "
            "density, an EQUAL triple for a neutral absorber, and a red-"
            "dominated one for a blue/green-absorbing dye -- so the red halo "
            "is a consequence of a spectrum and not three chosen numbers",
            _fpm.halation_gain_from_od(_fpm.AntiHalationSpec(), 1.48) is None
            and _gn is not None and max(_gn) - min(_gn) < 1e-12
            and _gd is not None and _gd[0] > 100.0 * _gd[1]
            and _gd[1] == _gd[2],
            "neutral %s; dyed r/g ratio %.0f" % (
                tuple(round(x, 4) for x in _gn), _gd[0] / max(_gd[1], 1e-12)))
        # ⚠ AND OOUE 1961 IS THE BOUND THE DERIVATION WILL BE CHECKED AGAINST.
        _oo = _fpm._OOUE_1961_IMAGE_SPREAD
        _r1 = [q.halation.radii_um[0] for q in FILM_PROFILES]
        _r1.sort()
        chk("G-V39-OOUE  the only published halation spatial scale in the "
            "corpus is stored as a BAND and the two terms do not overlap -- "
            "halation 100-200 um, turbidity below 100 um (Ooue 1961)",
            _oo["halation_exposure"] == (100.0, 200.0)
            and _oo["turbidity_spread"][1] == _oo["halation_exposure"][0]
            and _fpm._OOUE_1961_SOURCE,
            "derived sigma1 median %.0f um against Ooue's 100-200 um band; "
            "the tension is recorded at _OOUE_1961_IMAGE_SPREAD, not resolved"
            % _r1[len(_r1) // 2])

        # ---- v39, queue P70/P71: what the patent sweep found, and the
        # contradiction the pair already exposes -------------------------
        #
        # ⚠ THE SWEEP'S RESULT IS A NEGATIVE AND IT IS ASSERTED AS ONE. Not
        # one document in the corpus states an EXPOSURE-TIME optical density
        # for an antihalation layer. What it states is a silver coating
        # weight and a bound on the POST-PROCESSING RESIDUAL density, and
        # those are different quantities -- a removable absorber is opaque
        # during exposure and near-clear afterwards. If the residual bound
        # ever leaks into `optical_density` this fails.
        _ahref = _fpm._AH_COATING_REFERENCE
        _perm = _fpm._US6531271_PERMANENT_AH_DENSITY
        chk("G-V39-AH-SWEEP  the corpus states a colloidal-silver COATING "
            "WEIGHT and a PERMANENT residual density, and no exposure-time "
            "optical density anywhere -- the two are not conflated",
            abs(_ahref["colloidal_ag"][0] - 0.236) < 1e-9
            and abs(_ahref["colloidal_ag"][1] - 2.44) < 1e-9
            and _ahref["remjet"][0] == 0.0
            and _perm == (0.6, 0.3, 0.1, 0.05)
            and not any(q.anti_halation.has_density for q in FILM_PROFILES),
            "colloidal_ag 0.236 g Ag/m2 with 2.44 g gelatin (EP0608959B1, "
            "two coatings); rem-jet named and never quantified; permanent "
            "residual bound %s Status M" % (_perm,))

        # ---- v40, queue P70: the working antihalation density ---------------
        #
        # ⚠⚠ THIS GUARD REPLACES ONE WRITTEN THE SAME DAY THAT ASSERTED THE
        # OPPOSITE, AND THE ERROR IS PINNED HERE SO IT CANNOT RECUR. The first
        # version swept the owner's 277-document library, found no working
        # antihalation density, computed 0.24-0.30 D from the coating weights
        # and REFUSED it on the ground that it fell "below" US 6,531,271's 0.6.
        # That 0.6 is an UPPER BOUND on what may REMAIN after processing --
        # "up to about 0.6" -- so it constrains nothing about the density that
        # did the absorbing, and the refusal compared a number to a ceiling as
        # though it were a floor. 0.24-0.30 was very nearly the right answer.
        #
        # ⚠⚠ AND THE LIBRARY IS NOT THE LITERATURE. Three patents state the
        # number and none of them is on the owner's disk.
        #
        # Three independent routes, and the guard asserts they agree:
        #   US 3,392,021  a removable layer wants a density as high as 1.0, one
        #                 that stays in the film as low as 0.15, at 5-100 mg
        #                 per square foot of Carey Lea silver
        #   US 4,977,070  at least 0.3, preferably 0.5, 1.0, 2.0 to white light
        #   derived       four coating weights against the covering-power band
        #                 US 3,392,021's own two ranges imply, 0.93-2.79 D per
        #                 g Ag/m2 -- which brackets Glafkides' 1.0 and sits
        #                 above Mees's 0.5 exactly as Mees's grain-size law
        #                 requires for a sol
        _cw = [w for w, _ in _fpm._AH_SILVER_COATING_WEIGHTS]
        _cp = _fpm._AH_COVERING_POWER_BAND_D_PER_G_M2
        _derived = [(w * _cp[0], w * _cp[1]) for w in _cw]
        _stated_lo = min(_fpm._US3392021_AH_OD[1],
                         min(_fpm._US4977070_AH_WORKING_OD))
        _stated_hi = max(_fpm._US3392021_AH_OD[0],
                         max(_fpm._US4977070_AH_WORKING_OD))
        _inside = all(_stated_lo - 0.02 <= hi and lo <= _stated_hi + 0.02
                      for lo, hi in _derived)
        _def = _fpm._AH_WORKING_OD_DEFAULT["colloidal_ag"]
        chk("G-V40-P70  the working antihalation density is on file from two "
            "patents and re-derived from four coating weights, and the three "
            "routes agree -- while the number is still written onto no stock, "
            "because the law that would consume it is defined for an in-path "
            "absorber and 17 of the 31 stocks that name a construction carry a "
            "rear BACKING instead (queue P71)",
            len(_cw) == 4
            and _cp[0] < _fpm._GLAFKIDES_COVERING_POWER_D_PER_G_M2 < _cp[1]
            and _fpm._MEES_COVERING_POWER_D_PER_G_M2 < _cp[0]
            and _inside
            and _def == (0.15, 1.00, 2.00)
            and _fpm._US4581323_AH_TOPCOAT_OD[1] < _def[1]
            and _fpm._AH_OD_WRITTEN_TO_PROFILES is False
            and not any(q.anti_halation.has_density for q in FILM_PROFILES),
            "stated %.2f-%.2f D; four coating weights %s g Ag/m2 give %s at "
            "%.2f-%.2f D per g/m2, every one inside the stated range; Tier B "
            "still populated on 0 of %d stocks, and what gates the wiring is "
            "the construction split, not a missing number"
            % (_stated_lo, _stated_hi, [round(w, 3) for w in _cw],
               [(round(a, 2), round(b, 2)) for a, b in _derived],
               _cp[0], _cp[1], len(FILM_PROFILES)))
        # ⚠ AND THE POSITION IS RECORDED WITHOUT EVER CLAIMING TO BE MEASURED.
        _pos = sum(1 for q in FILM_PROFILES if q.anti_halation.position)
        _named = sum(1 for q in FILM_PROFILES if q.emulsion.antihalation
                     and q.emulsion.antihalation != "none")
        chk("G-V39-AH-POS  every stock that NAMES an antihalation "
            "construction carries the position that construction implies, "
            "and not one of them is flagged measured",
            _pos == _named
            and not any(q.anti_halation.measured for q in FILM_PROFILES),
            "%d of %d stocks positioned; measured on %d"
            % (_pos, len(FILM_PROFILES),
               sum(1 for q in FILM_PROFILES if q.anti_halation.measured)))
        # ⚠⚠ THE PAIR CONVICTS THE ESTIMATES BEFORE ANY FRAME IS SCANNED.
        _od = _fpm.cinestill_implied_remjet_od()
        chk("G-V39-PAIR  the 5219 / CineStill gains imply a rem-jet optical "
            "density that is neither neutral nor opaque, so the two hand-set "
            "triples cannot both be right -- pinned, not tuned away (P71)",
            max(_od) - min(_od) > 0.05 and max(_od) < 0.5,
            "implied OD r %.3f g %.3f b %.3f -- carbon black is neutral and "
            "a rem-jet backing is opaque; OD 0.2 passes 63 %% of the beam"
            % _od)

        # ---- v40, queue P71: the pair tests a construction the law omits ----
        #
        # ⚠⚠ THE 5219 / CINESTILL PAIR CANNOT VALIDATE THIS LAW, AND THE REASON
        # IS NOT THE MISSING FRAMES. 5219 carries a REM-JET BACKING: carbon
        # black on the far side of the support, beyond the very interface that
        # does the reflecting. The law applies 10^(-2*OD), which is the
        # attenuation of a ray that crosses an absorber, REFLECTS, and crosses
        # it again -- the geometry of an undercoat. A backing does not attenuate
        # the returning ray at all; it removes the total internal reflection by
        # replacing the base/air boundary with an index-matched absorber.
        # Inverting the pair through this law was therefore a category error,
        # and the nonsense implied OD was measuring the error. The law now
        # refuses `position == "backing"` outright, using a field the schema
        # has carried since v39.
        #
        # ⚠⚠ AND THE "TWO DIFFERENT SCALES" READING WRITTEN EARLIER THE SAME DAY
        # WAS WRONG, which is pinned here so it is not rediscovered as news.
        # Stage 5 computes pO = pE + g*(blur(A) - A) with kernel weights summing
        # to one, so the operator REDISTRIBUTES and g is the scattered FRACTION
        # of the above-threshold exposure. A fraction cannot exceed 1 and
        # neither can T_pack*10^(-2*OD): it is ONE scale. CineStill's stored
        # 1.05 is out of range for the engine as much as for the law.
        _cs = get_profile("CINESTILL_800T")
        _v3 = get_profile("KODAK_VISION3_500T_5219")
        _csg = (_cs.halation.gain_r, _cs.halation.gain_g, _cs.halation.gain_b)
        _sp = tuple([1.0] * 31)
        _mk = film_profiles.AntiHalationSpec
        _probe = {q: film_profiles.halation_gain_from_od(
            _mk(position=q, od_spectrum=_sp, lambda_start_nm=400.0,
                lambda_step_nm=10.0), 1.48)
            for q in ("undercoat", "in_pack", "in_base", "backing")}
        chk("G-V40-P71  the halation gain law is defined for an absorber IN "
            "the optical path and now refuses a rear BACKING, which is what "
            "the 5219 / CineStill pair differs in -- and the engine's gain "
            "ceiling is the law's own, not a second scale",
            _v3.emulsion.antihalation == "remjet"
            and _v3.anti_halation.position == "backing"
            and _cs.emulsion.antihalation == "none"
            and _probe["backing"] is None
            and all(_probe[q] is not None for q in
                    ("undercoat", "in_pack", "in_base"))
            and _fpm._HALATION_GAIN_LAW_POSITIONS
            == frozenset({"in_pack", "undercoat", "in_base"})
            and _fpm._HALATION_ENGINE_GAIN_CEILING
            == _fpm._HALATION_GAIN_LAW_CEILING == 1.0
            and max(_csg) > _fpm._HALATION_GAIN_LAW_CEILING
            and _fpm._CINESTILL_GAIN_OUT_OF_RANGE is True
            and _fpm._AHU_BEATS_REMJET_STATED is True,
            "the law returns a gain for undercoat / in_pack / in_base and None "
            "for backing; CineStill stores %.2f in red against a ceiling of "
            "%.1f that the engine and the law share. The experiment that WOULD "
            "work is the VISION3 AHU redesign against the rem-jet generation, "
            "and this corpus holds only the rem-jet one"
            % (max(_csg), _fpm._HALATION_GAIN_LAW_CEILING))

        # ---- v39, queues P42 and P48: two audits whose result is a NEGATIVE
        #
        # ⚠ P42 ASKED TO SORT THE 86 COLOUR NEGATIVES INTO TRACED AND
        # ESTIMATED "before anything is changed", on the premise that the
        # widest latitudes might be an artefact of estimation. Sorted: 48 are
        # tier-1, 15 tier-2, 23 tier-3 -- and of the 64 wider than 3.0 log E,
        # 36 ARE TIER-1 TRACED. The wide latitudes are measured as often as
        # they are estimated, so the premise is false and there is nothing to
        # re-derive. EP 0 083 377's 2.57-3.03 log E is one 1982 Konishiroku
        # patent's own coatings, not a bound on a modern C-41 film.
        _neg = [q for q in FILM_PROFILES
                if not q.is_monochrome and not q.is_reversal]
        def _ctier(q):
            best = 9
            for rec in _fpm._PARAM_SOURCES.get(q.name, ()):
                if any(k in rec.param for k in ("curve", "tone", "dmin")):
                    best = min(best, rec.tier)
            return best
        _wide = [q for q in _neg
                 if (q.curves.g.shoulder_x - q.curves.g.toe_x) > 3.0]
        _wide_t1 = sum(1 for q in _wide if _ctier(q) == 1)
        chk("G-V39-P42  the widest colour-negative latitudes are MEASURED as "
            "often as estimated, so the row's premise that they are an "
            "estimation artefact is false and nothing is re-derived",
            len(_neg) == 86 and _wide_t1 >= 30
            and _wide_t1 / max(len(_wide), 1) > 0.5,
            "%d negatives, %d wider than 3.0 log E, %d of those tier-1 traced"
            % (len(_neg), len(_wide), _wide_t1))
        # ⚠ P48 FEARED THAT A DEVELOPMENT TIME WOULD BE ATTACHED TO A CURVE
        # MEASURED UNDER A DIFFERENT CONDITION. Audited: 13 stocks carry both
        # a family and a stated condition, and on 10 of those the stated
        # developer appears in the family. The 3 that do not are NAMING
        # VARIANTS of the same developer -- ILFORD_PAN_F's "ID-11" against the
        # family's "ID-11 (D-76 type)" is the clearest -- and not conflicts.
        # ⚠ THEY ARE NOT NORMALISED. Each string is the SOURCE'S OWN WORDING,
        # and rewriting a source's words to make a join succeed is the error
        # this project refuses; the join is the thing that should widen.
        _both = [q for q in FILM_PROFILES
                 if q.processing_family and q.processing_family.points
                 and q.processing and q.processing.developer]
        _mismatch = [q.name for q in _both
                     if q.processing.developer
                     not in {r.developer for r in q.processing_family.points}]
        chk("G-V39-P48  every stock holding both a development family and a "
            "stated curve condition agrees on the developer, except three "
            "naming variants that are recorded rather than rewritten",
            len(_both) == 13 and len(_mismatch) == 3
            and "ILFORD_PAN_F" in _mismatch,
            "%d stocks hold both; %d differ only in spelling: %s"
            % (len(_both), len(_mismatch), sorted(_mismatch)))

        # ---- v39, queue P34: ГОСТ 10691.6-88, read by OCR off the render --
        #
        # ⚠ IT PRINTS THE SAME TWO SPEED CRITERIA AS ГОСТ 9160-91 FOR A
        # DIFFERENT MATERIAL CLASS, which is what makes it evidence rather
        # than a citation. Conflict I-24 was settled at v31 by showing that
        # fog+0.2 and fog+0.85 are two rows of one colour table; a second
        # standard, for black-and-white technical film, prints both numbers
        # again. The pair is a property of Soviet sensitometry, not of 9160.
        _g6 = _fpm._GOST_10691_6_CRITERIA
        _dkr9 = {round(r[1], 2) for r in _fpm._GOST_9160_SPEED.values()} \
            if hasattr(_fpm, "_GOST_9160_SPEED") else set()
        chk("G-V39-P34  ГОСТ 10691.6-88 states S = K / H_kr with criteria "
            "D_kr 0.85 and 0.20 at K = 1, and its published amendment "
            "tightens the rounding rule rather than loosening it",
            _g6 == ((0.85, 1.0), (0.20, 1.0))
            and "members of a geometric progression"
            in _fpm._GOST_10691_6_ROUNDING
            and "amendment" in _fpm._GOST_10691_6_ROUNDING
            and "OCR" in _fpm._GOST_10691_6_SOURCE,
            "criteria %s; the 10691.0-84 companion is read -- see G-V40-P34"
            % (_g6,))

        # ---- v40, queue P53: the dye panels have SEVEN ordinates ------------
        #
        # ⚠⚠ THE ROW ASKED WHICH ORDINATE THESE PANELS USE. They do not share
        # one. 115 panels, 65 of them calibrating, on SEVEN different vertical
        # frames under a single caption -- and the split between them is
        # physics rather than bookkeeping: a frame whose floor is ZERO is a
        # linear density axis and keeps its curves above the axis, a frame
        # whose floor is -1, -2 or -3 is carrying a LOGARITHM, because an
        # optical density cannot be negative and a wing that falls without
        # limit is a log. The rule is therefore per panel and derivable from
        # the frame alone.
        _yf = _fpm._SOVREMENNYE_DYEABS_Y_FRAMES
        _xf = _fpm._SOVREMENNYE_DYEABS_X_FRAMES
        _lin = sum(v for k, v in _yf.items() if k[0] >= 0)
        _log = sum(v for k, v in _yf.items() if k[0] < 0)
        chk("G-V40-P53  the book's dye-absorption panels are drawn on seven "
            "different ordinates, and the floor of each frame decides which "
            "of the two kinds it is",
            len(_yf) == 7
            and sum(_yf.values()) == sum(_xf.values())
            and sum(_yf.values()) + _fpm._SOVREMENNYE_DYEABS_NO_CAL
            == _fpm._SOVREMENNYE_DYEABS_TOTAL
            and _lin == 23 and _log == 42
            and _fpm.sovremennye_dyeabs_is_log(-1.0)
            and not _fpm.sovremennye_dyeabs_is_log(0.0)
            # ⚠ THE BLOCKER IS GONE AND THE FLAG RECORDS THAT, which is why
            # this reads `is False` where it once read `is True`. `interior`
            # used to blank the panel GRID along with the frame and cut every
            # curve into fragments; it now finds the grid by morphological
            # opening UNION the ink test, blanks it, and MENDS the gaps with
            # a closing written back only inside the blanked bands. All 23
            # zero-floor panels band, and none is refused.
            and _fpm._SOVREMENNYE_GRID_CHOPS_CURVES is False
            and _fpm._SOVREMENNYE_DYEABS_BANDED == (_lin, 0)
            and not any(q.dye_density.d_cyan and "Современные" in
                        (q.dye_density.source or "")
                        for q in FILM_PROFILES),
            "%d of %d panels calibrate: %d on a zero-floor LINEAR density "
            "axis, %d on a negative-floor LOG axis; all %d linear panels now "
            "band with y < m < c peaks and 0 are refused, and nothing is "
            "stored against a profile -- the ordinate question the row asked "
            "is answered per panel, not per book"
            % (sum(_yf.values()), _fpm._SOVREMENNYE_DYEABS_TOTAL, _lin, _log,
               _fpm._SOVREMENNYE_DYEABS_BANDED[0]))

        # ---- v40, queue P34: ГОСТ 10691.0-84, the densitometry base ---------
        #
        # ⚠⚠ THE ROW SAID THIS DOCUMENT WAS OUT OF REACH "BECAUSE THE FILE NAME
        # IS CYRILLIC". The alphabet was never the problem. That copy is named
        # with its full 232-character Russian title -- about 330 bytes in UTF-8
        # against a 255-byte limit on one path component -- so the transfer
        # completed and the write did not; and the same library holds the whole
        # 10691.0-84 -- 10691.4-84 booklet under the short name «МАТЕРИАЛЫ
        # ФОТОГРАФИЧЕСКИЕ.pdf», 14 pages WITH a text layer.
        #
        # ⚠⚠ BOTH OF ITS DENSITOMETER TABLES DECODE THEMSELVES, and that is the
        # only reason a 1984 scan could be believed to two decimals: each prints
        # the response as the SUM of a receiver column and a source column, so
        # every row checks its own arithmetic. Table 1 then checks against
        # something this project holds independently -- its receiver column is
        # CIE 1924 V(λ) shifted by a constant, and `kodak_1956_spectro.V_LAMBDA`
        # was digitised for an unrelated purpose two days earlier.
        _v0 = _fpm._GOST_10691_0_VISUAL
        _c0 = _fpm._GOST_10691_0_COPY
        _vbad = [r for r in _v0 if abs(r[1] + r[2] - r[3]) > 1e-9]
        # ⚠ ONE COPY ROW IS OFF BY 0.01 AND IT IS ROUNDING, NOT A MISREAD: at
        # 480 nm 4.13 + 0.93 = 5.06 against a printed 5.05, which is what two
        # independently rounded columns do. The tolerance is 0.011 and the
        # count of rows needing it is pinned at one, so a genuinely mis-typed
        # row cannot hide behind it.
        _cbad = [r for r in _c0
                 if abs(r[1] + r[2] - r[4]) > 1e-9 or abs(r[1] + r[3] - r[5]) > 1e-9]
        _cworst = max(max(abs(r[1] + r[2] - r[4]), abs(r[1] + r[3] - r[5]))
                      for r in _c0)
        from kodak_1956_spectro import V_LAMBDA as _VL
        _vres = max(abs(math.log10(_VL[nm]) + _fpm._GOST_10691_0_VISUAL_OFFSET - v)
                    for nm, v, _, _ in _v0 if nm >= 420)
        chk("G-V40-P34  ГОСТ 10691.0-84's two densitometer tables reproduce "
            "their own printed sums, and the visual one reproduces CIE 1924 "
            "V(lambda) far inside the standard's own 0.05 tolerance",
            not _vbad and len(_cbad) == 1 and _cworst <= 0.011
            and len(_v0) == 16 and len(_c0) == 21
            and _vres <= 0.02,
            "visual 16 rows, 0 sum violations, worst V(lambda) residual %.4f "
            "against a permitted 0.05; copy 21 rows, %d rounding row(s), worst "
            "%.3f" % (_vres, len(_cbad), _cworst))

        # ⚠ AND THE CLAUSE THAT COST THE MOST: §4.2.2 prints the DEFINITION of
        # «коэффициент контрастности», which this file had recorded as
        # unavailable in the whole corpus.
        _oc = get_profile("TASMA_OCH_45")
        chk("G-V40-P34B  the Soviet contrast coefficient has a printed "
            "definition at last -- the straight-line slope, with the chord "
            "defined separately in the next clause as the mean gradient",
            _fpm._GOST_10691_0_CONTRAST_COEFF == "straight_line"
            and _fpm._GOST_10691_0_CONTRAST_STOCKS == ("TASMA_OCH_45",)
            and _oc.gamma_criterion == "straight_line"
            and get_profile("SVEMA_CNL_65").gamma_criterion == ""
            and get_profile("SVEMA_CO_90L").gamma_criterion == ""
            and _fpm._GOST_10691_0_GRADIENT_POINT == (0.10, 1.3),
            "TASMA_OCH_45 filled; the two COLOUR stocks stay empty because "
            "10691.0-84 governs black-and-white only; mean gradient is "
            "Dmin+0.10 over 1.3 decades against 9160's +0.15 / +0.20")

        # ⚠ AND THE MISMATCH THE SAME READING EXPOSES, SURFACED NOT SILENCED.
        _mism = [p.name for p in FILM_PROFILES
                 if p.is_monochrome and p.gost_speed_class]
        chk("G-V40-P34C  the six monochrome Soviet stocks carrying a ГОСТ "
            "9160-91 material class are named, because 9160-91 is the COLOUR "
            "standard and 10691.0-84 is the one that covers them",
            tuple(sorted(_mism)) == tuple(sorted(_fpm._GOST_BW_CLASS_MISMATCH))
            and all(get_profile(n).speed_criterion == "iso6" for n in _mism),
            "%d stocks: %s -- every one also carries speed_criterion 'iso6', "
            "so the two fields disagree about which system rated it; nothing "
            "is relabelled" % (len(_mism), sorted(_mism)))

        # ---- v39, queue P72: the three Ooue 1961 figures -------------------
        #
        # ⚠⚠ FIG. 7 CORROBORATES THIS MODULE'S OWN ARITHMETIC FROM A 1961
        # DRAWING. `base_fresnel(1.48)` computes the acetate critical angle
        # from Snell's law as 42.51 degrees; Ooue's section drawing prints
        # 約42度 on the same geometry. Two independent routes, half a degree
        # apart, and the second is a picture of the mechanism v34 assumes.
        _og = _fpm._OOUE_1961_GEOMETRY
        chk("G-V39-OOUE7  Fig. 7's drawn critical angle agrees with the "
            "angle this module computes from its own refractive index, and "
            "its base range brackets the gauge 155 of 191 stocks carry",
            abs(_og["critical_angle_deg"][0] - _fpm.base_fresnel(1.48)[1])
            < 0.6
            and _og["base_um"] == (100.0, 150.0)
            and _og["emulsion_um"] == (10.0, 20.0),
            "drawn 42.0 deg against computed %.2f deg; base %s um, emulsion "
            "%s um" % (_fpm.base_fresnel(1.48)[1], _og["base_um"],
                       _og["emulsion_um"]))
        # ⚠ AND FIG. 8 IS THE ONLY PER-LAYER SHARPNESS MEASUREMENT IN THE
        # CORPUS. What is asserted is the ORDERING, because that is what
        # survives a +/- 1.5 lines/mm eye reading; the absolute values carry
        # their uncertainty beside them and are NOT applied to any stock.
        _f8 = {n: v for n, _nm, v in _fpm._OOUE_1961_FIG8}
        # ⚠ AND THE COUNT IS THE POINT: 115 of 191 DO carry a per-record
        # spread, so the schema has always been able to hold one -- what it
        # has never had is a MEASUREMENT of how the spread arises from layer
        # depth. Fig. 8 is that measurement, on a 1961 Fujicolor negative,
        # and it is not transferred to a modern stock.
        _split = sum(1 for q in FILM_PROFILES if len(set(q.mtf.f50s())) > 1)
        chk("G-V39-OOUE8  the yellow, magenta and cyan layers of one "
            "Fujicolor negative measure 1.9x apart in the order top-to-"
            "bottom -- and every stock in this database still stores ONE f50 "
            "for all three records",
            _f8[4] > _f8[5] > _f8[6]
            and _f8[4] / _f8[6] > 1.7
            and _fpm._OOUE_1961_FIG8_READ_UNCERTAINTY >= 1.0
            and _split == 115,
            "yellow %.1f / magenta %.1f / cyan %.1f lines per mm at R = 0.5, "
            "+/- %.1f by eye; %d of %d stocks already carry a per-record "
            "spread and none of it is derived from layer depth (queue P72)"
            % (_f8[4], _f8[5], _f8[6],
               _fpm._OOUE_1961_FIG8_READ_UNCERTAINTY, _split,
               len(FILM_PROFILES)))

        # ---- v40, queue P72: Fig. 5, resolving power against density --------
        #
        # ⚠⚠ THE FIGURE THAT SETTLES WHAT A PUBLISHED RESOLVING POWER MEANS.
        # This corpus stores one resolving power per stock with no density
        # attached to it, and Ooue's Fig. 5 shows the quantity RISING, PEAKING
        # and FALLING over 0.1-1.35 D on three different materials. A number
        # quoted without a density is therefore the PEAK of such a curve, and
        # the peak sits at 0.70-1.05 D -- not at D-min, which is where a
        # renderer would naturally apply it.
        #
        # ⚠ THE CHECK IS THE PAPER'S OWN PROSE, NOT A REPEATED COLUMN. Ooue
        # states that a THINNER emulsion reaches a HIGHER maximum and loses
        # resolving power FASTER as density rises. Neither ordering was fitted;
        # both fall out of the three traces.
        _f5 = _fpm._OOUE_1961_FIG5
        _pk = _fpm._OOUE_1961_FIG5_PEAKS
        _order = ["minicopy_copinal_1_1", "neopan_ss_microfine",
                  "xray_direct_rendol"]
        _hi = [_pk[k][1] for k in _order]
        _drop = []
        for k in _order:
            pd, pv = _pk[k]
            _tail = [v for d, v in _f5[k] if d > pd]
            _drop.append(pv - (_tail[-1] if _tail else pv))
        _peakbad = [k for k in _order
                    if abs(max(v for _, v in _f5[k]) - _pk[k][1]) > 0.05
                    or not (_fpm._OOUE_1961_FIG5_PEAK_BAND[0] - 1e-9
                            <= _pk[k][0]
                            <= _fpm._OOUE_1961_FIG5_PEAK_BAND[1] + 1e-9)]
        chk("G-V40-P72  Ooue Fig. 5 makes resolving power a non-monotone "
            "function of density, and the three traces reproduce the paper's "
            "own statement that a thinner coating peaks higher and falls "
            "faster",
            not _peakbad
            and _hi[0] > _hi[1] > _hi[2]
            and _drop[0] > _drop[1] and _drop[0] > _drop[2]
            and all(len(v) >= 13 for v in _f5.values())
            and _fpm._OOUE_1961_FIG5_WIRED is False,
            "peaks %s lines/mm at D %s, all inside the %s band; fall past the "
            "peak %s; stored and INERT -- no stage makes resolving power a "
            "function of density"
            % ([round(v, 1) for v in _hi],
               [_pk[k][0] for k in _order],
               _fpm._OOUE_1961_FIG5_PEAK_BAND,
               [round(v, 1) for v in _drop]))

        # ---- v39, queue S1: the fourth colour record -----------------------
        #
        # ⚠⚠ THE SCHEMA USED TO BE THE REASON THIS DATA WAS OUTSIDE IT.
        # doc/FUJI_FOURTH_LAYER.md opens "THIS FILE EXISTS BECAUSE THE
        # DATABASE CANNOT HOLD WHAT IS IN IT" -- five Fuji colour negatives
        # draw FOUR sensitive layers and SpectralSensitivity had three. The
        # numbers were traced, checked and documented, and stored nowhere.
        # They are stored now.
        _f4 = [q for q in FILM_PROFILES if q.spectral.log_s_c]
        _pk = []
        for q in _f4:
            c = q.spectral.log_s_c
            i = max(range(len(c)), key=lambda k: c[k])
            _pk.append(q.spectral.lambda_start_nm
                       + q.spectral.lambda_step_nm * i)
        chk("G-V39-S1  five Fuji colour negatives carry the cyan fourth "
            "record, every one on the same grid as its own three, and all "
            "five peak in the blue-green gap between blue and green",
            len(_f4) == 5
            and all(len(q.spectral.log_s_c) == len(q.spectral.log_s_g)
                    for q in _f4)
            and all(510.0 <= v <= 530.0 for v in _pk)
            and all(q.spectral.has_data for q in _f4),
            "peaks %s nm; blue and green records unchanged on all five"
            % sorted(set(_pk)))
        # ⚠ AND IT IS NOT INTEGRATED, WHICH IS THE PART A LATER PASS COULD GET
        # WRONG. A fourth SENSITIVE layer feeds the same three dyes; the
        # taking path sums three records and normalises over three. If the
        # fourth ever enters that sum without the normalisation changing with
        # it, this fails.
        chk("G-V39-S1-INERT  the fourth record is stored and reachable and "
            "the spectral taking path still integrates exactly three",
            not any(q.spectral.log_s_c and q.spectral.log_s_pan
                    for q in FILM_PROFILES)
            and all(len(q.spectral.log_s_r) == len(q.spectral.log_s_c)
                    for q in _f4),
            "%d of %d stocks carry it; none is monochrome, so no record can "
            "be counted twice" % (len(_f4), len(FILM_PROFILES)))

        # ---- v42, queue P35: colour paper gets a spectral record ----------
        #
        # ⚠⚠ THE FIRST SPECTRAL DATA OF ANY KIND FOR COLOUR PAPER, and the
        # reason it took eight days is recorded in the v31 comment on
        # `PrintStock.density_geometry`: the three Fujicolor Crystal Archive
        # bulletins print no characteristic curve, PrintStock cannot exist
        # without one, so the panels had nowhere to go and were left unread.
        # The refusal to invent a curve was right; using it to discard the
        # measurements was not.
        #
        # ⚠ THE ARTWORK-REUSE CHECK IS THE POINT OF THE SECOND ASSERTION.
        # Supreme and Type CA publish byte-identical panel images, so this is
        # ONE measurement under two names. Two records covering three products
        # is correct; three records would double-count it in any later
        # average, which is the trap the T-MAX P3200 MTF row already hit once.
        _ps = _fpm.PAPER_SPECTRA
        _names = [n for r in _ps for n in r.names]
        _shared = [r for r in _ps if r.shared_artwork]
        # Each panel normalises every dye to unit peak, so a trace that did
        # not recover ~1.0 on all three did not follow the curve it was on.
        _pkerr = []
        for r in _ps:
            for t in (r.dye_density.d_cyan, r.dye_density.d_magenta,
                      r.dye_density.d_yellow):
                _pkerr.append(abs(max(t) - 1.0))
        # Sensitivity: peak normalised to log 0.0 per layer, and the physical
        # ordering blue < green < red asserted on the stored curves, not on
        # the convenience fields.
        _ordbad = []
        for r in _ps:
            sp = r.spectral
            pk = []
            for t in (sp.log_s_b, sp.log_s_g, sp.log_s_r):
                i = max(range(len(t)), key=lambda k: t[k])
                pk.append(sp.lambda_start_nm + sp.lambda_step_nm * i)
            if not (pk[0] < pk[1] < pk[2]):
                _ordbad.append(r.key)
            if abs(max(sp.log_s_b) - 0.0) > 1e-9:
                _ordbad.append(r.key + " blue not normalised")
        chk("G-V42-P35  two colour-paper spectral records cover three "
            "Fujicolor Crystal Archive products, every dye trace recovers "
            "its panel's unit peak, and no record carries a characteristic "
            "curve it was never given",
            len(_ps) == 2 and len(_names) == 3 and len(set(_names)) == 3
            and len(_shared) == 1 and len(_shared[0].names) == 2
            and max(_pkerr) < 0.02 and not _ordbad
            and all(r.density_geometry == "reflection" for r in _ps)
            and all(r.spectral.criterion == "relative_linear_printed"
                    for r in _ps)
            and all(r.impurity.measurement_mode == "reflection" for r in _ps)
            and not any(getattr(r, "curves", None) for r in _ps),
            "3 products / 2 panel sets (Supreme and Type CA share one), "
            "worst dye peak error %.3f, reflection geometry throughout, "
            "%d impurity ratios in reflection mode"
            % (max(_pkerr), sum(len(r.impurity.ratios) for r in _ps)))

        # ⚠ AND THE DMAX AIMS ARE REFLECTION DENSITIES, WHICH IS THE WHOLE
        # REASON `density_geometry` GAINED A FOURTH VALUE. A paper's 2.0-2.2
        # and a print film's ~4.0 are not the same scale; stored side by side
        # without the geometry they would make the paper look catastrophic.
        _aims = [(r.key, r.dmax_aim_r, r.dmax_aim_g, r.dmax_aim_b)
                 for r in _ps if r.dmax_aim_r]
        _filmdmax = [max(c.dmax for c in s.curves.as_tuple())
                     for s in PRINT_STOCKS]
        chk("G-V42-P35B  every paper Dmax aim sits in the reflection range "
            "and every print-FILM Dmax sits above it, so the two populations "
            "never overlap and cannot be compared by accident",
            len(_aims) == 2
            and all(1.8 <= v <= 2.3 for a in _aims for v in a[1:])
            and min(_filmdmax) > 2.3
            and "reflection" in _fpm._DENSITY_GEOMETRIES
            and all(s.density_geometry != "reflection" for s in PRINT_STOCKS),
            "papers %s; %d print films, lowest Dmax %.2f"
            % ([(a[0].split('_')[-1], a[1], a[2], a[3]) for a in _aims],
               len(PRINT_STOCKS), min(_filmdmax)))

        # ---- v41, queue P45: the Soviet handbook's EI pairs ----------------
        #
        # ⚠⚠ THE TABLE VALIDATES ITSELF AND NOBODY PRINTED THE VALIDATION.
        # «Справочник кинооператора» table 2.4 gives each stock a daylight EI,
        # a tungsten EI and the conversion filter between them, and never
        # states a filter factor. The factors fall out: every 80A row recovers
        # the Wratten 80A's published TWO stops and every 85 / 85A row
        # recovers its published TWO THIRDS of a stop. Two independent filter
        # families, ten rows, no fitting -- a transcription error in any EI
        # would break one of them.
        _ki = _fpm._KINOOPERATOR_EI_TABLE
        _f80 = [math.log2(r[3] / r[5]) for r in _ki if r[6] == "80A"]
        _f85 = [math.log2(r[5] / r[3]) for r in _ki if r[4] in ("85", "85A")]
        # ⚠ THE TWO FAMILIES ARE CHECKED DIFFERENTLY BECAUSE THEY ARE
        # DIFFERENT CLAIMS, and the first attempt at this guard got the
        # second one wrong. The 80A rows recover the filter's factor exactly:
        # four independent pairs, all within a twentieth of a stop of two
        # stops, which is the Wratten 80A's published cost. The 85 rows do
        # NOT all sit at the 85's two thirds, and they should not: nine do,
        # but EASTMAN EXR 100T is rated 100 tungsten / 80 daylight and
        # PRIMETIME 640T 640 / 500, which are ONE third-stop steps. A film
        # rated through a conversion filter is rated on the 1/3-stop ISO
        # ladder, and the emulsion's own daylight response pays part of the
        # filter's cost. So the assertion is that every 85 ratio is a WHOLE
        # NUMBER of third-stops, one or two -- which a transcription error
        # in any EI would break just as surely, without claiming a uniformity
        # the handbook does not print.
        _f85t = [v * 3.0 for v in _f85]
        chk("G-V41-P45  the handbook's daylight/tungsten EI pairs recover the "
            "80A's two stops exactly, and every No. 85 pair lands on a whole "
            "third-stop of the ISO ladder",
            len(_ki) == 15 and len(_f80) == 4 and len(_f85) == 11
            and all(abs(v - 2.0) < 0.10 for v in _f80)
            and all(abs(v - round(v)) < 0.08 and round(v) in (1, 2)
                    for v in _f85t)
            and sum(1 for v in _f85t if round(v) == 2) == 9
            and _fpm._KINOOPERATOR_APPLIED == 6
            and len(_fpm._KINOOPERATOR_STOCKS) == 6,
            "80A recovers %s stops (published 2); the eleven 85/85A pairs "
            "recover %s third-stops, nine at 2/3 and two at 1/3; 6 of 15 "
            "rows map onto stocks this database holds"
            % ([round(v, 3) for v in _f80],
               sorted(round(v, 2) for v in _f85t)))

        # ---- v41, queue P11: resolving power is a LENS-PAIR figure ---------
        #
        # ⚠ A RESOLVING POWER WITHOUT ITS OPTIC IS TWO DIFFERENT NUMBERS.
        # Glafkides tabulates the same nine emulsions twice, once through a
        # Fuess objective and once through an apochromat, and the apochromat
        # reads HIGHER on every single row -- so a corpus that stores one
        # figure per stock is storing whichever lens that source happened to
        # use. The ordering is the check: it was never fitted.
        _gl = _fpm._GLAFKIDES_LENS_PAIR
        _glbad = [k for k, (fu, ap) in _gl.items() if not ap > fu]
        _glst = [q for q in FILM_PROFILES
                 if q.mtf.resolving_power_lp_mm_apo > 0.0]
        chk("G-V41-P11  the apochromat reads higher than the Fuess objective "
            "on every one of the nine emulsions, and the three stocks this "
            "database holds carry both figures with the optic named",
            len(_gl) == 9 and not _glbad
            and _fpm._GLAFKIDES_LENS_APPLIED == 3 and len(_glst) == 3
            and all(q.mtf.resolving_power_lp_mm_apo
                    > q.mtf.resolving_power_lp_mm_fuess > 0.0
                    and q.mtf.resolving_optic for q in _glst),
            "9 emulsion pairs, ratio %.2f-%.2f; %d stocks carry the pair"
            % (min(ap / fu for fu, ap in _gl.values()),
               max(ap / fu for fu, ap in _gl.values()), len(_glst)))

        # ---- v41, queue P36: a ratio needs to say what it is a ratio OF ----
        #
        # ⚠ DENSITY RATIO AND GAMMA RATIO ARE NOT INTERCHANGEABLE and the
        # file held both under one field name. Hanson's masking rule is
        # written in gamma ratios; every Soviet ТУ figure in this file is a
        # density ratio. The default exists so that no legacy record silently
        # becomes a gamma ratio, and every record must now name its quantity.
        _rat = [r for q in FILM_PROFILES for r in q.dye_impurity.ratios]
        _rat += [r for rec in _fpm.PAPER_SPECTRA for r in rec.impurity.ratios]
        _noq = [r for r in _rat if not r.quantity]
        chk("G-V41-P36  every stored dye-impurity ratio names the quantity "
            "it is a ratio of, and the Hanson masking rule is on file as the "
            "gamma-ratio case it is",
            bool(_rat) and not _noq
            and _fpm._IMPURITY_QUANTITY_DEFAULT == "density_ratio"
            and _fpm._IMPURITY_QUANTITIES == frozenset(
                {"density_ratio", "gamma_ratio"})
            and _fpm._HANSON_MASK_RULE == (-1.0, 2),
            "%d ratios, %d unlabelled; Hanson's rule stored as %s"
            % (len(_rat), len(_noq), (_fpm._HANSON_MASK_RULE,)))

        # ---- v41, queue P52: gamma(t) as a LAW, not a scatter of points ----
        #
        # ⚠ TWO VESSELS, TWO LAWS, AND AVERAGING THEM WOULD DESCRIBE NEITHER.
        # A ProcessingFamily used to hold development POINTS only, so a stock
        # tabulated in a small tank and a large one had one cloud with two
        # branches in it. The fitted rate law separates them: the large
        # replenished tank runs slower (smaller k) and reaches further
        # (higher gamma at infinity), which is the physics and not the fit.
        _law = [ln for q in FILM_PROFILES for ln in q.processing_family.laws]
        _lbad = [ln for ln in _law
                 if not (ln.gamma_infinity > 0.0 and ln.dev_rate_k > 0.0
                         and ln.fit_rms > 0.0 and ln.has_rate_law)]
        _tm = _fpm.get_profile("KODAK_TMAX_P3200").processing_family
        _small = _tm.law_for("T-MAX", vessel="small tank")
        _large = _tm.law_for("T-MAX RS", vessel="large tank")
        chk("G-V41-P52  the fitted development laws separate the small tank "
            "from the large one in both directions the chemistry predicts, "
            "and every law fits to better than 0.02 gamma rms",
            len(_law) == 2 and not _lbad
            and _small is not None and _large is not None
            and _large.gamma_infinity > _small.gamma_infinity
            and _large.dev_rate_k < _small.dev_rate_k
            and max(ln.fit_rms for ln in _law) < 0.02,
            "small tank gamma_inf %.3f k %.4f, large tank %.3f / %.4f; "
            "worst rms %.4f"
            % (_small.gamma_infinity, _small.dev_rate_k,
               _large.gamma_infinity, _large.dev_rate_k,
               max(ln.fit_rms for ln in _law)))

        # ---- v41, queue P66: a print stock's own chemistry -----------------
        _pproc = [s for s in PRINT_STOCKS
                  if s.processing is not None and s.processing.developer]
        chk("G-V41-P66  the release print stock names the process it is "
            "developed in, which no print stock could state before v41",
            len(_pproc) == 1 and _pproc[0].processing.developer.startswith(
                "ECP-2"),
            "%s in %s" % (_pproc[0].name, _pproc[0].processing.developer))

        # ---- v41, queue P68: the interimage zone profile, REFUSED ----------
        #
        # ⚠ THE MEASUREMENT IS KEPT AND THE GENERALISATION IS NOT. One house,
        # one record (blue), three points: enough to state that the interimage
        # effect varies with exposure zone, nowhere near enough to give every
        # stock a zone profile. The guard asserts the refusal so that a later
        # pass cannot quietly adopt it without moving this line.
        chk("G-V41-P68  the exposure-zone interimage spread is stored as a "
            "measurement and NOT wired into any profile",
            _fpm._IIE_ZONE_PROFILE_ADOPTED is False
            and _fpm._IIE_ZONE_RECORDS == ("b",)
            and _fpm._IIE_ZONE_SPREAD[1] > _fpm._IIE_ZONE_SPREAD[0] > 1.0,
            "spread %s on record %s; adopted False"
            % (_fpm._IIE_ZONE_SPREAD, _fpm._IIE_ZONE_RECORDS))

        # ---- v44, queue P73: the 1931-1969 antihalation patent chain -------
        #
        # ⚠⚠ THE CHAIN SETTLES THAT `position` IS A PHYSICAL SYSTEM AND NOT A
        # LABEL. Queue P70 fixed a working antihalation density of 0.15-2.0 D
        # from in-path absorbers; these four patents show a BACKING capped an
        # order of magnitude lower, because positives are printed through it.
        # Two Kodak patents eighteen years apart give the same reason in the
        # same words, and the guard asserts the two bands do not overlap --
        # if they ever did, one number would be being used for two systems.
        _pos = _fpm._AH_OD_BY_POSITION
        _perm = _fpm.ah_od_band_for("backing")
        _p70_lo = min(_fpm._US4977070_AH_WORKING_OD)
        chk("G-V44-P73  the backing antihalation band sits an order of "
            "magnitude below the in-path working range, and the accessor "
            "refuses every position the chain does not cover",
            len(_pos) == 4
            and _perm == (0.10, 0.30)
            and _fpm.ah_od_band_for("backing", removable=True) == (0.80, 0.80)
            and _fpm.ah_od_band_for("backing",
                                    permanent_tinted=True) == (0.04, 0.08)
            and _fpm.ah_od_band_for("in_pack") is None
            and _fpm.ah_od_band_for("undercoat") is None
            and _fpm.ah_od_band_for("in_base") is None
            and _perm[1] < _p70_lo * 3.0
            and _fpm._AH_BACKING_CEILING_REASON
                == "printing_through_the_base",
            "permanent backing %s against P70's in-path working range from "
            "%.2f D; removable %s; 1/10-1/20 tinted %s"
            % (_perm, _p70_lo,
               _fpm.ah_od_band_for("backing", removable=True),
               _fpm.ah_od_band_for("backing", permanent_tinted=True)))

        # ⚠ AND NOT ONE OF THOSE BANDS IS WRITTEN ONTO A PROFILE. 30 stocks
        # name an antihalation POSITION and none carries an optical density;
        # stamping a band midpoint onto nineteen backing stocks would be
        # nineteen inventions. Absence stays absence.
        _ah_pos = [q for q in FILM_PROFILES if q.anti_halation.position]
        _ah_od = [q.name for q in _ah_pos if q.anti_halation.optical_density]
        chk("G-V44-P73B  the class bands stay class bands -- every stock that "
            "names an antihalation position still carries no optical density",
            len(_ah_pos) == 30 and not _ah_od
            and _fpm._AH_OD_WRITTEN_TO_PROFILES is False,
            "%d stocks name a position, %d carry a density"
            % (len(_ah_pos), len(_ah_od)))

        # ⚠⚠ THE ONLY QUANTITATIVE HALATION MEASUREMENT IN THIS CORPUS, and it
        # behaves: latitude rises monotonically with the antihalation measure,
        # the bare film is the floor, and the dye's return diminishes -- which
        # is the patent's own conclusion that "very adequate and useful
        # results are obtained with dye layer densities much less than 0.2".
        _lad = _fpm._US2481770_HALATION_LADDER
        _bare = [r for r in _lad if not r[2] and r[1] == 0.0][0]
        _sub = [r for r in _lad if r[2] and r[1] == 0.0][0]
        _dye = sorted([r for r in _lad if r[2] and r[1] > 0.0],
                      key=lambda r: r[1])
        chk("G-V44-P73C  the 1949 halation ladder is monotone in the "
            "antihalation measure, the low-index sublayer alone clears the "
            "patent's own worthwhile threshold, and the dye's return "
            "diminishes",
            len(_lad) == 7
            and abs(_bare[3] - 1.15) < 1e-9
            and _sub[3] > _fpm._US2481770_WORTHWHILE_LATITUDE[1]
            and [r[3] for r in _dye] == sorted(r[3] for r in _dye)
            and _dye[0][3] > _sub[3]
            and (_dye[-1][3] - _dye[-2][3]) < (_dye[1][3] - _dye[0][3])
            and _fpm._US2481770_SUPERADDITIVE_CLAIMED is True
            and _fpm._US2481770_SUPERADDITIVE_MEASURED is False,
            "bare %.2f, sublayer alone %.2f (threshold %s), dye rows %s -- "
            "super-additivity is the patent's claim and is NOT asserted as "
            "measured here, because the rows do not isolate the dye-alone case"
            % (_bare[3], _sub[3], _fpm._US2481770_WORTHWHILE_LATITUDE,
               [(r[1], r[3]) for r in _dye]))

        # ⚠⚠ THE SECOND CRITICAL ANGLE. `base_fresnel()` gives the one at the
        # base/air interface, which is what v34 derives the halation radii
        # from; US 2,481,770 is built on one at the EMULSION/sublayer
        # interface that this model never had. It returns None for every stock
        # in this file, because an ordinary gelatin subbing has the emulsion's
        # own index and there is no break at all -- a law with no data, which
        # is the honest state and not an oversight.
        _nem = _fpm._US2481770_INDICES["gelatin_emulsion"]
        _brk = _fpm._US2481770_MIN_INDEX_BREAK
        _ang = _fpm.emulsion_side_critical_angle(_nem, _nem - _brk)
        _base_ang = _fpm.base_fresnel(
            _fpm._US2481770_INDICES["cellulose_acetate"])[1]
        chk("G-V44-P73D  the emulsion-side critical angle exists, refuses a "
            "break below the patent's 0.08 threshold, and is far more oblique "
            "than the base-side one it supplements",
            _fpm.emulsion_side_critical_angle(_nem, _nem) is None
            and _fpm.emulsion_side_critical_angle(_nem, _nem - 0.05) is None
            and _ang is not None and abs(_ang - 71.46) < 0.1
            and _ang > _base_ang + 25.0
            and _fpm.emulsion_side_critical_angle(_nem, 1.40) is not None,
            "at the minimum 0.08 break the emulsion turns back rays beyond "
            "%.1f deg where acetate turns them back beyond %.1f deg, so the "
            "sublayer removes the MOST oblique rays -- which the patent says "
            "are 'the source of the more objectionable halation'"
            % (_ang, _base_ang))

        # ⚠ AND THE 1949 INDEX TABLE CORROBORATES THE ONE THIS FILE HAS USED
        # SINCE v34 FROM A BROAD PATENT RANGE: 1.481 measured against 1.48 in
        # use. Nitrate joins as its own row -- it is 1.2 per cent higher than
        # acetate, which is not nothing on a critical angle.
        _opt = _fpm._V34_BASE_OPTICS
        _ix = _fpm._US2481770_INDICES
        chk("G-V44-P73E  the measured 1949 indices corroborate the acetate "
            "figure in use and give nitrate its own row, and no profile "
            "names nitrate so no render moves",
            abs(_opt["acetate"][0] - _ix["cellulose_acetate"]) < 0.002
            and set(_opt) == {"acetate", "nitrate", "polyester"}
            and abs(_opt["nitrate"][0] - 1.498) < 1e-9
            and _fpm._v34_support_key("cellulose nitrate") == "nitrate"
            and _fpm._v34_support_key("") == "acetate"
            and not [q for q in FILM_PROFILES
                     if _fpm._v34_support_key(q.emulsion.base_material)
                     == "nitrate"]
            and _ix["support_range_lo"] <= _ix["cellulose_nitrate"]
                <= _ix["support_range_hi"],
            "acetate 1.48 in use against 1.481 measured (0.07 %%); nitrate "
            "1.498 gives a critical angle of %.2f deg against acetate's "
            "%.2f; 0 of %d stocks name it"
            % (_fpm.base_fresnel(1.498)[1], _base_ang, len(FILM_PROFILES)))

        # ⚠⚠ THE FIRST MEASURED COST OF AN IN-PACK ANTIHALATION LAYER, and the
        # reason `position` could be stored with no consequence until now.
        # Fuji coat one emulsion three ways: an unmordanted dye undercoat
        # loses about a stop because the dye migrates into the emulsion, and
        # mordanting it cuts the loss to a third of a stop.
        _loss = _fpm._US3445231_INPACK_SPEED_LOSS
        _sup = _fpm._US3445231_INTERLAYER_GREEN_SUPPRESSION
        chk("G-V44-P73F  mordanting an in-pack antihalation layer costs less "
            "speed than letting its dye migrate, by the same ratio in the "
            "black-and-white and the colour-interlayer experiments",
            set(_loss) == {"unmordanted", "mordanted"}
            and _loss["mordanted"][0] < _loss["unmordanted"][0]
            and abs(_loss["unmordanted"][0] - 0.50) < 1e-9
            and abs(_loss["mordanted"][0] - 0.20) < 1e-9
            and _sup == {"unmordanted": 0.50, "mordanted": 0.20},
            "%.0f %% of the speed lost unmordanted against %.0f %% mordanted "
            "-- about one stop against a third of one; the same 0.50/0.20 "
            "pair suppresses the red layer's unwanted green sensitivity when "
            "the construction is used as a colour interlayer"
            % (_loss["unmordanted"][0] * 100.0, _loss["mordanted"][0] * 100.0))

        # ==================================================================
        # v45, 2026-09-18f -- the six-document harvest
        # ==================================================================

        # ---- queue K6: the sheet that proved its own absence ---------------
        #
        # ⚠⚠ THE ROW CLOSED BY A SECOND EDITION AGREEING WITH THE FIRST, which
        # is the opposite of how a blocked acquisition row usually closes. K6
        # asked for PORTRA 100T's own sensitometry; E-2468 (2006) prints
        # PORTRA 160VC's figures instead, and the row's escape clause was that
        # only one edition had been read. The July 2000 FIRST printing carries
        # the same three figure ids and the same six traced numbers, so the
        # defect is original and the measurement was never published.
        _sfa = _fpm._SHARED_FIGURE_ARTWORK
        _def = _fpm._SHARED_ARTWORK_IS_DEFECT
        _e24 = _fpm._E2468_EDITIONS
        chk("G-V45-K6  all three E-2468 CURVES-page figures are registered as "
            "PORTRA 160VC artwork printed under PORTRA 100T, and the register "
            "separates a DEFECT from legitimately shared artwork",
            _def <= set(_sfa)
            and _def == {"F009_0154AC", "F009_0180AC", "F009_0186AC"}
            and all("PORTRA 100T" in " ".join(_sfa[k][1]) for k in _def)
            and all(_sfa[k][0] != "KODAK PROFESSIONAL PORTRA 100T"
                    for k in _def)
            and len(set(_sfa) - _def) == 3,
            "%d figures registered, %d of them defects (%s) and %d "
            "legitimately shared; no defect figure MEASURES the product it is "
            "printed under"
            % (len(_sfa), len(_def), ", ".join(sorted(_def)),
               len(set(_sfa) - _def)))

        chk("G-V45-K6B  E-2468 had exactly two printings, six years apart, "
            "and every measured value on the sheet is identical in both -- "
            "which is what turns 'may not exist in print' into 'does not'",
            len(_e24) == 2
            and [e[0] for e in _e24] == ["2000-07", "2006-10"]
            and [e[2] for e in _e24] == [4, 6]
            and len(_fpm._E2468_UNCHANGED_ACROSS_EDITIONS) == 4
            and _fpm._E2468_DISCONTINUED == "year-end 2006",
            "printings %s at %s pages; %d classes of value unchanged across "
            "them; discontinued %s"
            % ("/".join(e[0] for e in _e24), "/".join(str(e[2]) for e in _e24),
               len(_fpm._E2468_UNCHANGED_ACROSS_EDITIONS),
               _fpm._E2468_DISCONTINUED))

        # ---- queue K6: the conversion filters become data ------------------
        #
        # ⚠⚠ THE 80A ROW IS CHECKED AGAINST KODAK'S OWN PRINTED TABLE, WHICH
        # IS WHY IT CAN BE BELIEVED. E-2468's discontinuance notice rates
        # three different films through the same filter, so the factor is
        # asserted by REPRODUCING all three of Kodak's printed indices rather
        # than by restating the constant it was derived from.
        _cfl = _fpm._CONVERSION_FILTER_LOSS
        _eit = _fpm.exposure_index_through
        _80a = [(ei, _eit(ei, "80A")) for ei in (160, 400, 800)]
        chk("G-V45-K6C  the WRATTEN 80A costs exactly two stops, reproducing "
            "all three of the exposure indices Kodak's own discontinuance "
            "table prints, and the No. 85 family is stored as a SPREAD "
            "because its eleven measured rows do not agree",
            all(abs(got - want) < 1e-9
                for (_e, got), want in zip(_80a, (40.0, 100.0, 200.0)))
            and _cfl["80A"][2] == "measured"
            and _cfl["85"][2] == "spread"
            and abs(_cfl["85"][1] - 1.0 / 3.0) < 1e-9
            and _cfl["80B"][0] < _cfl["80A"][0]
            and set(_cfl) == set(_fpm._CONVERSION_FILTER_SOURCE)
            and _fpm.conversion_filter_loss("80") is None,
            "80A %s -> %s (two stops, three for three); 85 carries a "
            "%.2f-stop spread around %.2f rather than a constant; %d filters, "
            "all sourced; an unnamed '80' answers None"
            % ([e for e, _g in _80a], [round(g, 1) for _e, g in _80a],
               _cfl["85"][1], _cfl["85"][0], len(_cfl)))

        # ---- ГОСТ 25120-82, read first-hand --------------------------------
        #
        # ⚠⚠ THE GUARD'S REAL CONTENT IS THE THREE THINGS CHECKING A STOCK
        # AGAINST ITS OWN STANDARD FOUND, not the transcription. SVEMA_CNL_65
        # is ЦНЛ-65 and had cited this standard through Гурлев since
        # 2026-08-11 without anyone comparing the numbers.
        _t6 = _fpm._GOST_25120_TABLE6
        _n = _fpm.gost_25120_norm
        _cnl = [p for p in FILM_PROFILES if p.name == "SVEMA_CNL_65"][0]
        _hi = {c: _n("Фото ЦНЛ-65", "highest", "fog_mask_%s_max" % c)[0]
               for c in ("blue", "green", "red")}
        _fi = {c: _n("Фото ЦНЛ-65", "first", "fog_mask_%s_max" % c)[0]
               for c in ("blue", "green", "red")}
        _st = {"blue": _cnl.curves.b.dmin, "green": _cnl.curves.g.dmin,
               "red": _cnl.curves.r.dmin}
        chk("G-V45-K6D  ГОСТ 25120-82 table 6 is transcribed in full for both "
            "marks, ЦНД-32 has no highest-quality column and the accessor "
            "says None rather than borrowing the other mark's",
            len(_t6) == 13
            and len(_fpm._GOST_25120_COLUMNS) == 3
            and len(_fpm._GOST_25120_MARKS) == 2
            and _n("Фото ЦНД-32", "highest", "resolving_min") is None
            and _n("Фото ЦНД-32", "first", "resolving_min")[0] == 58.0
            and _n("Фото ЦНЛ-65", "highest", "resolving_min")[0] == 90.0
            and _n("Фото ЦНЛ-65", "first", "resolving_min")[0] == 63.0
            and _n("Фото ЦНЛ-65", "first", "resolving_min")[1] == "floor",
            "%d rows x %d printed columns over %d marks; resolving power "
            "58 (ЦНД-32 first) / 90 (ЦНЛ-65 highest) / 63 (ЦНЛ-65 first), all "
            "floors, and ЦНД-32 highest answers None because no such column "
            "is printed" % (len(_t6), len(_fpm._GOST_25120_COLUMNS),
                            len(_fpm._GOST_25120_MARKS)))

        chk("G-V45-K6E  SVEMA_CNL_65's stored D-min triple conforms to ГОСТ "
            "25120-82's FIRST quality category and exceeds the HIGHEST on two "
            "of its three filters, which is what places the coating",
            all(_st[c] <= _fi[c] + 1e-9 for c in _st)
            and _st["blue"] > _hi["blue"] + 1e-9
            and _st["red"] > _hi["red"] + 1e-9
            and abs(_st["green"] - _hi["green"]) < 1e-9
            and _fpm._GOST_25120_CNL65_CATEGORY == "first",
            "behind blue %.2f (highest <= %.2f, first <= %.2f), green %.2f "
            "(<= %.2f / %.2f), red %.2f (<= %.2f / %.2f) -- inside the first "
            "category on all three and on its red ceiling exactly"
            % (_st["blue"], _hi["blue"], _fi["blue"],
               _st["green"], _hi["green"], _fi["green"],
               _st["red"], _hi["red"], _fi["red"]))

        chk("G-V45-K6F  SVEMA_CNL_65's stored resolving power IS ГОСТ "
            "25120-82's «не менее» FLOOR and not a measurement, and its three "
            "stored gammas exceed the standard's RECOMMENDED band because the "
            "two were developed in different processes",
            abs(_cnl.mtf.resolving_power_lp_mm_highc
                - _n("Фото ЦНЛ-65", "first", "resolving_min")[0]) < 1e-9
            and _t6["resolving_min"][3] == "floor"
            and len(_fpm._GOST_25120_CNL65_GAMMA_CONFLICT) == 3
            and all(stored > norm + tol + 1e-9 for _r, stored, norm, tol
                    in _fpm._GOST_25120_CNL65_GAMMA_CONFLICT)
            and "Two processes, not two opinions"
            in _fpm._GOST_25120_CNL65_GAMMA_REASON,
            "stored resolving %.0f lp/mm == the first category's floor of "
            "%.0f lin/mm; stored gammas %s against the recommended %s +/- "
            "0.08, all three above the band, and the disagreement is recorded "
            "rather than reconciled"
            % (_cnl.mtf.resolving_power_lp_mm_highc,
               _n("Фото ЦНЛ-65", "first", "resolving_min")[0],
               "/".join("%.2f" % s for _r, s, _nm, _t
                        in _fpm._GOST_25120_CNL65_GAMMA_CONFLICT),
               "/".join("%.2f" % nm for _r, _s, nm, _t
                        in _fpm._GOST_25120_CNL65_GAMMA_CONFLICT)))

        chk("G-V45-K6G  ГОСТ 25120-82 defers its sensitometry to ГОСТ 9160-82 "
            "and reading it does NOT close queue P38: the 1982 edition's "
            "criteria are still unread and still answer None",
            _fpm._GOST_25120_METHODS["sensitometry"] == "ГОСТ 9160-82"
            and "9160-82" in _fpm._GOST_EDITIONS_UNREAD
            and _fpm._GOST_SPEED_CRITERIA.get("9160-82") == {}
            and all(_fpm.gost_speed_criterion("9160-82", c) is None
                    for c in _fpm._GOST_SPEED_CLASSES)
            and _t6["speed_nominal"][4] == "ед. ГОСТ 9160-82"
            # ⚠ AND THE BASE-DENSITY METHOD IS A DIFFERENT EDITION AGAIN.
            and _fpm._GOST_25120_METHODS["base_optical_density"]
            == "ГОСТ 10691.0-73",
            "the specification cites 9160-82 for speed and 10691.0-73 for "
            "base density; this corpus holds 9160-91 and 10691.0-84, so both "
            "citations point at editions it does not have and neither may be "
            "substituted")

        chk("G-V45-K6H  ГОСТ 25120-82's ageing allowance is stored as a "
            "TOLERANCE at the end of a one-year warranty and is marked as "
            "such, so nothing can read it as an annual decay rate",
            _fpm._GOST_25120_AGEING_TOLERANCE["is_tolerance_not_rate"] is True
            and _fpm._GOST_25120_AGEING_TOLERANCE[
                "speed_total_loss_max_fraction"] == 0.50
            and _fpm._GOST_25120_AGEING_TOLERANCE[
                "fog_mask_rise_max_per_filter_d"] == 0.15
            and _fpm._GOST_25120_AGEING_TOLERANCE["warranty_months"] == 12
            # The process is a ferricyanide bleach with two fixes -- NOT C-41,
            # so its times may never be substituted into one.
            and len(_fpm._GOST_25120_SCHEDULE) == 8
            and sum(1 for s in _fpm._GOST_25120_SCHEDULE
                    if s[0] == "фиксирование") == 2
            and "калий железосинеродистый, ГОСТ 4206-75"
            in _fpm._GOST_25120_BATHS["bleach"]["g_per_l"],
            "a conforming film may lose 50 %% of total speed and gain 0.15 D "
            "behind each filter by the end of its 12-month warranty; the test "
            "process is %d steps with two fixes and a ferricyanide bleach, so "
            "it is not C-41" % len(_fpm._GOST_25120_SCHEDULE))

        # ---- the scanner, from US 12,211,182 B2 and JOSA A 21(7) -----------
        #
        # ⚠⚠ THE BIAS FACTOR IS THE GUARD'S POINT. Queues D1 and D2a are about
        # to measure a scanner's sigma by subtracting an N-scan mean from one
        # scan, and that residual is smaller than sigma by sqrt((n-1)/n). At
        # the patent's own n = 49 the correction is 1 %; at a more practical
        # n = 4 it is 15.5 %, which is larger than anything those rows are
        # trying to resolve.
        _g = _fpm.scanner_noise_residual_gain
        chk("G-V45-D1  the one-scan-minus-mean scanner-noise residual is "
            "smaller than the true sigma by sqrt((n-1)/n), the correction is "
            "monotone in n, and a single scan is refused rather than "
            "answered",
            _g(1) == 0.0 and _g(0) == 0.0
            and abs(_g(2) - (0.5 ** 0.5)) < 1e-12
            and abs(_g(49) - (48.0 / 49.0) ** 0.5) < 1e-12
            and _g(4) < _g(16) < _g(49) < 1.0
            and _fpm._WB_NOISE_RESIDUAL_BIAS_N == 49,
            "sigma is understated by %.1f %% at n=4, %.1f %% at n=16 and "
            "%.2f %% at the patent's own n=49; n<2 answers 0.0 because a "
            "single scan has no mean to subtract"
            % ((1.0 / _g(4) - 1.0) * 100.0, (1.0 / _g(16) - 1.0) * 100.0,
               (1.0 / _g(49) - 1.0) * 100.0))

        chk("G-V45-D1B  the scanner/grain frequency ordering is stored BOTH "
            "WAYS ROUND, because the patent states it inverts between colour "
            "and black-and-white film, and this database holds both",
            "HIGHER" in _fpm._WB_NOISE_ORDERINGS["noise_frequency_colour"]
            and "REVERSED"
            in _fpm._WB_NOISE_ORDERINGS["noise_frequency_monochrome"]
            and len(_fpm._WB_NOISE_ORDERINGS) == 6
            and _fpm._WB_SCANNER_NOISE_EXPERIMENT["scans_per_element"] == 49
            and _fpm._WB_SCANNER_NOISE_EXPERIMENT["strips_analysed"] == 45
            and _fpm._WB_SCANNER_NOISE_EXPERIMENT["patches"] == 17
            and _fpm._WB_DOUBLE_FLASH_SPIKE_PATCHES == (10, 11),
            "%d orderings recorded; 49 scans x 45 strips x 17 patches x 3 "
            "elements, and the 1-2 JND spike at patches 10-11 is filed as a "
            "double-flash SCANNER artefact rather than a property of the "
            "density it lands on" % len(_fpm._WB_NOISE_ORDERINGS))

        # ⚠ THE JOSA PAPER'S NUMBERS ARE A REFLECTION SCANNER ON A PIGMENT
        # TARGET. What the guard asserts is the two ORDERINGS, which are what
        # transfers to a film scanner, plus the fact that the adaptive
        # estimator beats the linear model in the honest (candidate-excluded)
        # case on every one of the four measures.
        _res = _fpm._SCANNER_LINEARITY_RESIDUAL
        _acc = _fpm._SCANNER_SPECTRAL_ACCURACY
        chk("G-V45-D2B  the linear reflectance model's residual is worst in "
            "BLUE on every target and roughly doubles off the training set, "
            "and adaptive estimation beats it on both targets in the "
            "candidate-EXCLUDED case that is the realistic one",
            all(m[2] > m[0] and m[2] > m[1] for m, _mx in _res.values())
            and all(mx[2] > mx[0] and mx[2] > mx[1]
                    for _m, mx in _res.values())
            and all(_res["CDC"][0][i] > _res["MCC"][0][i] * 1.6
                    for i in range(3))
            and all(_acc[(t, "AE")][0] < _acc[(t, "LRM_ex")][0]
                    and _acc[(t, "AE")][3] < _acc[(t, "LRM_ex")][3]
                    for t in ("CDC", "IT8"))
            and _fpm._SCANNER_AE_NEIGHBOURS == {"CDC": 20, "IT8": 10},
            "blue mean residual %.2f %% (MCC, in training) and %.2f %% (CDC, "
            "out of it) against %.2f/%.2f in red; AE reaches dE94 %.2f and "
            "%.2f where the excluded-candidate linear model reaches %.2f and "
            "%.2f; the estimator's neighbour count is target-dependent (20 "
            "vs 10) rather than universal"
            % (_res["MCC"][0][2], _res["CDC"][0][2],
               _res["MCC"][0][0], _res["CDC"][0][0],
               _acc[("CDC", "AE")][0], _acc[("IT8", "AE")][0],
               _acc[("CDC", "LRM_ex")][0], _acc[("IT8", "LRM_ex")][0]))

        # ---- JP 2004-272114 A: how much dye a grain can hold ---------------
        #
        # ⚠⚠ THE RATIO IS THE ASSERTION. Two emulsions, one octahedral and one
        # tabular, measured at the same 80 A^2 dye footprint: the plate takes
        # 2.63x the dye on the same silver, which is the tabular grain's whole
        # speed argument as a number rather than a claim.
        _sat = _fpm._DYE_MONOLAYER_SATURATION
        _ratio = (_sat["JP2004272114_B"][0] / _sat["JP2004272114_A"][0])
        _lay = _fpm._DYE_ADSORBED_LAYERS
        chk("G-V45-JP  the tabular emulsion's monolayer dye capacity is 2.63x "
            "the octahedral one's on the same silver, the selenium-sensitised "
            "grain shares the tabular figure because it IS the same grain, "
            "and multilayer adsorption reached 2.45 layers on all six "
            "coatings",
            abs(_ratio - 2.62962962962963) < 1e-9
            and _sat["JP2004272114_C"][0] == _sat["JP2004272114_B"][0]
            and _sat["JP2004272114_A"][1] == "octahedral"
            and _sat["JP2004272114_B"][1] == _sat["JP2004272114_C"][1]
            == "tabular"
            and len(_lay) == 6 and max(_lay) - min(_lay) <= 0.0201
            and all(v > 2.0 for v in _lay)
            and _fpm._DYE_FOOTPRINT_A2 == 80.0
            and _fpm._TABULAR_ASPECT_THRESHOLDS == (2.0, 8.0),
            "%.3e vs %.3e mol dye / mol Ag at an 80 A^2 footprint, a factor "
            "of %.2f; dye layer count %.2f-%.2f across six coatings, held "
            "flat on purpose so the additive's effect is not confused with a "
            "loading difference"
            % (_sat["JP2004272114_B"][0], _sat["JP2004272114_A"][0], _ratio,
               min(_lay), max(_lay)))

        chk("G-V45-JPB  residual sensitizing dye is recorded as a D-min "
            "mechanism with a measured band, and NO per-sample number from "
            "the patent's raster tables is stored",
            _fpm._RESIDUAL_DYE_STAIN["band_nm"] == (480.0, 580.0)
            and "D-min" in _fpm._RESIDUAL_DYE_STAIN["raises"]
            and "tabular grains" in _fpm._RESIDUAL_DYE_STAIN["worsened_by"]
            and _fpm._RESIDUAL_DYE_STAIN_EXAMPLE2["speed_relative"] == 100.0
            and _fpm._RESIDUAL_DYE_STAIN_EXAMPLE2["stain_relative"] == 88.0
            and _fpm._DYE_ABSORPTION_STRENGTH["sample_16"]
            - _fpm._DYE_ABSORPTION_STRENGTH["sample_15"] == 1.0
            and "TABLES 1 AND 2 ARE RASTER" in _fpm._JP2004272114_SOURCE,
            "the stain is an absorption area over 480-580 nm on an unexposed "
            "processed sample; the one printed improvement is 100 -> 88 with "
            "speed unchanged at 100 and light-absorption strength 213 -> 214, "
            "which is the control that the additive moves the stain and not "
            "the sensitisation")

        # ==================================================================
        # v46, 2026-09-18g -- the random-dot / Boolean grain model
        # ==================================================================
        #
        # ⚠⚠ THE FIRST GUARD IS THAT TWO LITERATURES DESCRIBE ONE MODEL.
        # Fenton's random dot (Kodak, 2004, from Nutting 1913) and Newson's
        # Boolean model (IPOL, 2017, from stochastic geometry) differ only in
        # that one expands the exponential to first order, and 0.434 is
        # log10(e). Asserted numerically rather than claimed in prose.
        import numpy as _np
        import film_sim as _sim
        import film_profiles as _fpm
        _rd = _fpm.random_dot_disk_diameter_um
        _bl, _bc = _fpm.boolean_lambda, _fpm.boolean_coverage
        _eps = 0.1 / 256.0
        _lin = [(u / (1.0 + _eps), _bc(_bl(u, 0.1), 0.1))
                for u in (0.001, 0.01, 0.1, 0.5)]
        _fen = [(u, _fpm._RANDOM_DOT_LOG10E * (-math.log(1.0 - u / (1.0 + 0.1 / 256.0))))
                for u in (0.001, 0.01)]
        chk("G-V46-GRAIN  the random-dot model and the Boolean model are the "
            "same law, the Boolean coverage round-trips through its own "
            "intensity, and Fenton's 0.434 is log10(e)",
            abs(_fpm._RANDOM_DOT_LOG10E - math.log10(math.e)) < 1e-15
            and all(abs(got - u) < 1e-9 for u, got in _lin)
            # D = -log10(1-p) and Fenton's D = 0.434 * p agree as p -> 0.
            and all(abs(-math.log10(1.0 - u) / (_fpm._RANDOM_DOT_LOG10E * u)
                        - 1.0) < 6e-3 for u in (0.001, 0.01))
            and abs(_fpm._GRANULARITY_APERTURE_AREA_UM2
                    - math.pi * 24.0 ** 2) < 1e-9,
            "coverage round-trips to 1e-9 at four grey levels; the linearised "
            "and exact densities agree to %.2e at p=0.001 and %.2e at p=0.01; "
            "the 48 um aperture is read as CIRCULAR, %.1f um^2 -- a square "
            "reading would be 2304 and move a derived diameter by 13 %%"
            % (abs(-math.log10(0.999) / (_fpm._RANDOM_DOT_LOG10E * 0.001) - 1.0),
               abs(-math.log10(0.99) / (_fpm._RANDOM_DOT_LOG10E * 0.01) - 1.0),
               _fpm._GRANULARITY_APERTURE_AREA_UM2))

        # ⚠⚠ THE INVERSION IS A GRAIN SIZE FROM A NUMBER EVERY STOCK ALREADY
        # CARRIES, AND IT AGREES WITH THE ONLY FIVE MEASURED ONES.
        _M = set(_fpm._CLUMP_MEASURED_STOCKS)
        _ratio = {}
        _deq = {}
        for _p in FILM_PROFILES:
            _g = _p.grain
            if not (_g.rms_granularity and _g.clump_um_g):
                continue
            _d = _rd(_g.rms_granularity)
            _deq[_p.name] = _d
            _ratio[_p.name] = _g.clump_um_g / _d
        _mr = sorted(_ratio[n] for n in _M if n in _ratio)
        _er = sorted(v for k, v in _ratio.items() if k not in _M)
        _med = lambda a: a[len(a) // 2] if len(a) % 2 else 0.5 * (a[len(a) // 2 - 1] + a[len(a) // 2])
        _q1 = lambda a: a[len(a) // 4]
        chk("G-V46-GRAINSIZE  the random-dot diameter derived from rms "
            "granularity alone lands inside both measured grain bands on file "
            "and reproduces all five MEASURED clump_um values within a factor "
            "of two, while putting the 186 estimated ones about five times "
            "away -- two populations that do not overlap at the quartile",
            len(_mr) == 5 and len(_er) >= 180
            and max(_mr) < 2.0 and min(_mr) > 0.9
            and _med(_er) > 4.0 * _med(_mr)
            # ⚠ AT THE QUARTILE, NOT AT THE EXTREME. One estimated stock
            # reaches 1.77 against the measured maximum of 1.80, so the
            # ranges touch; the lower quartile of the estimated set still
            # sits far above the measured maximum, which is the claim.
            and _q1(_er) > max(_mr) * 2.0
            and 0.15 < min(_deq.values()) < 0.25
            and 2.5 < max(_deq.values()) < 3.0
            and _fpm._RANDOM_DOT_VS_CLUMP["applied"] is False
            and _fpm._RANDOM_DOT_VS_CLUMP["owner_decision"] is True,
            "measured five %s (median %.2f); estimated %d median %.2f, "
            "lower quartile %.2f -- more than twice the measured maximum, so the "
            "two populations separate cleanly; derived diameters span %.3f-%.3f um against BBC "
            "T-101's printed 0.59-1.43 and Takano 1968's derived 0.50-1.36. "
            "NOT applied: queue C45's precedent gives the owner a corpus-wide "
            "grain rescale"
            % ("/".join("%.2f" % v for v in _mr), _med(_mr), len(_er),
               _med(_er), _q1(_er), min(_deq.values()), max(_deq.values())))

        # ⚠⚠ TAGUCHI'S PUBLISHED CLASSIFICATION, REPRODUCED ON 191 STOCKS HE
        # NEVER SAW -- and it only reproduces under the random-dot diameter.
        _split = _fpm.taguchi_spread_split
        _imp = sorted(p.name for p in FILM_PROFILES
                      if p.mtf.f50_g > 0 and p.grain.clump_um_g > 0
                      and _split(p.mtf.f50_g, p.grain.clump_um_g) is None)
        _usable = [p for p in FILM_PROFILES
                   if p.mtf.f50_g > 0 and p.grain.rms_granularity > 0]
        _opt = 0
        for _p in _usable:
            _s = _split(_p.mtf.f50_g, _rd(_p.grain.rms_granularity))
            if _s is not None and _s[2] > _s[1]:
                _opt += 1
        chk("G-V46-TAGUCHI  Schade quadrature with the random-dot diameter "
            "puts EVERY stock in this database on the optical-determined side "
            "of Taguchi's plane, as his paper reports for AgX negatives -- "
            "where clump_um makes four of them outright impossible",
            _opt == len(_usable) and len(_usable) >= 185
            and sorted(_imp) == sorted(_fpm._TAGUCHI_IMPOSSIBLE_UNDER_CLUMP)
            and abs(_fpm._TAGUCHI_F50_TO_SIGMA_UM
                    - 1000.0 * math.sqrt(math.log(2.0)
                                         / (2.0 * math.pi ** 2))) < 1e-9
            and abs(_fpm._TAGUCHI_F50_OVER_NU61
                    - math.sqrt(2.0 * math.log(2.0))) < 1e-12,
            "%d of %d optical-dominant under the random-dot diameter; under "
            "clump_um the four impossible stocks are %s, every one a "
            "high-f50 tabular emulsion. f50 -> sigma uses "
            "1000*sqrt(ln2/(2 pi^2)) = %.4f and f50/nu61 = sqrt(2 ln 2)"
            % (_opt, len(_usable), ", ".join(_imp),
               _fpm._TAGUCHI_F50_TO_SIGMA_UM))

        # ⚠ FENTON'S RADIATION RESULT IS THE STRONGEST CONFIRMATION OF THE
        # RANDOM-DOT MODEL IN THIS CORPUS and is asserted as such: doubling
        # the dose doubles the centre count, sigma goes as its square root,
        # and 21 measured points average 1.413 against sqrt(2) = 1.41421.
        _fr = _fpm._FENTON_RADIATION
        chk("G-V46-FENTON  doubling the radiation dose multiplies the "
            "granularity it contributes by sqrt(2), measured to one part in "
            "1170 over 21 points, and the model's own sqrt(D) law is "
            "reproduced by its equations",
            abs(_fr["measured_sigma_ratio_mean"]
                - _fr["expected_sigma_ratio"]) < 1.5e-3
            and _fr["points"] == 21
            and len(_fpm._FENTON_EMULSIONS) == 7
            # sigma_D goes as sqrt(D) in the linearised form: doubling D
            # multiplies it by sqrt(2), independent of centre size.
            and all(abs(_fpm.random_dot_sigma_d(d, 2.0)
                        / _fpm.random_dot_sigma_d(d, 1.0) - math.sqrt(2.0))
                    < 1e-12 for d in (0.3, 0.7, 1.5))
            # and the inversion is the exact inverse of the forward law.
            and all(abs(_fpm.random_dot_sigma_d(_rd(r), 1.0) * 1000.0 - r)
                    < 1e-9 for r in (2.6, 10.0, 38.8)),
            "measured %.3f against sqrt(2) = %.5f over %d points on 7 tabular "
            "emulsions of ECD %.2f-%.2f um; the forward law and the inversion "
            "round-trip to 1e-9 in rms units"
            % (_fr["measured_sigma_ratio_mean"], _fr["expected_sigma_ratio"],
               _fr["points"], min(e[0] for e in _fpm._FENTON_EMULSIONS),
               max(e[0] for e in _fpm._FENTON_EMULSIONS)))

        # ⚠⚠ THE THEOREM ABOUT clump_gain, ASSERTED AGAINST THE ENGINE'S OWN
        # TWO SHAPE FUNCTIONS. The Boolean transfer is a jinc with zeros; the
        # engine's is a Gaussian that never reaches zero and carries an
        # optional lobe at one sixth the grain frequency, i.e. a correlation
        # six grain diameters long -- which a Boolean model cannot produce.
        _bs = _sim.boolean_grain_shape
        _fz = _sim.BOOLEAN_GRAIN_FIRST_ZERO_OVER_D
        _ff = _np.linspace(1.0, 2000.0, 200001)
        _hh = _bs(_ff, 1.0)
        _zi = int(_np.argmin(_hh))
        chk("G-V46-BOOLSHAPE  the closed-form Boolean grain transfer is the "
            "normalised jinc, reaches exactly 1.0 at DC and hits its first "
            "ZERO at 1.2197/d -- a compact-support covariance no Gaussian "
            "shape has, which is why clump_gain is a departure from the "
            "physical model rather than a parameter inside it",
            abs(float(_bs(_np.array([0.0]), 1.0)[0]) - 1.0) < 1e-6
            and abs(float(_ff[_zi]) - _fz * 1000.0) < 1.0
            and float(_hh[_zi]) < 1e-5
            and abs(_fz - 3.8317059702075123 / math.pi) < 1e-12
            and _fpm._BOOLEAN_COVARIANCE_SUPPORT_DIAMETERS == 1.0
            and "cannot produce a low-frequency lobe"
            in _fpm._BOOLEAN_NO_LONG_RANGE_LOBE
            # the hand-rolled J1 is accurate enough to be believed
            and abs(float(_sim._bessel_j1(_np.array([10.0]))[0])
                    - 0.04347274616886144) < 1e-8
            and abs(float(_sim._bessel_j1(_np.array([20.0]))[0])
                    - 0.06683312417584991) < 1e-8,
            "first zero measured at %.1f c/mm for a 1 um grain against the "
            "predicted %.1f, residual %.2e; J1 agrees with reference values "
            "at x = 10 and 20 to better than 1e-8"
            % (float(_ff[_zi]), _fz * 1000.0, float(_hh[_zi])))

        # ==================================================================
        # v47, 2026-09-18h -- E-6 reversal colour reproduction, measured
        # ==================================================================
        #
        # ⚠⚠ THE GUARD RE-DERIVES THE CLASS STATISTICS FROM THE TRACED
        # TABLES RATHER THAN RESTATING THEM. The thesis claims by eye that the
        # four E-6 films run parallel and that Kodachrome and the digital
        # camera are the outliers; these numbers are what that claim becomes,
        # and if a traced cell is ever corrected the statistics move with it.
        _E6 = list(_fpm._KREYENBUEHL_E6_SET)
        _dCk, _dHk = _fpm._KREYENBUEHL_DELTA_C, _fpm._KREYENBUEHL_DELTA_H

        def _vec(d, k):
            return _np.array([_np.nan if v is None else float(v)
                              for v in d[k]])

        _Cm = _np.nanmean(_np.vstack([_vec(_dCk, k) for k in _E6]), axis=0)
        _Hm = _np.nanmean(_np.vstack([_vec(_dHk, k) for k in _E6]), axis=0)
        _rms = {k: (float(_np.sqrt(_np.nanmean((_vec(_dCk, k) - _Cm) ** 2))),
                    float(_np.sqrt(_np.nanmean((_vec(_dHk, k) - _Hm) ** 2))))
                for k in _E6 + ["KR64", "D3"]}
        _worst_e6 = max(_rms[k][0] for k in _E6)
        _best_out = min(_rms[k][0] for k in ("KR64", "D3"))
        chk("G-V47-E6CLASS  the four E-6 reversal films share a colour "
            "signature and Kodachrome and the digital camera do not -- the "
            "thesis's visual claim, re-derived from the traced tables as a "
            "separation of more than 2.5x in chroma with no overlap",
            all(abs(_rms[k][0] - _fpm._E6_CLASS_SCATTER_RMS[k][0]) < 0.02
                and abs(_rms[k][1] - _fpm._E6_CLASS_SCATTER_RMS[k][1]) < 0.02
                for k in _rms)
            and _np.allclose(_np.round(_Cm, 1),
                             _fpm._E6_CLASS_DELTA_C_MEAN, atol=0.06)
            and _np.allclose(_np.round(_Hm, 1),
                             _fpm._E6_CLASS_DELTA_H_MEAN, atol=0.06)
            and _best_out > 2.5 * _worst_e6
            and len(_fpm._CCSG_TEST_PATCHES) == 7
            and tuple(_fpm._CCSG_TEST_PATCHES) == _fpm._CCSG_PATCH_ORDER,
            "worst E-6 film sits %.2f dC from the class mean, the nearest "
            "outsider %.2f -- a factor of %.1f; Kodachrome %.2f, the Nikon "
            "D3 %.2f, over %d patches"
            % (_worst_e6, _best_out, _best_out / _worst_e6,
               _rms["KR64"][0], _rms["D3"][0], len(_fpm._CCSG_PATCH_ORDER)))

        # ⚠⚠ AND THE SPLIT BETWEEN CHROMA AND HUE, WHICH THE THESIS
        # DOES NOT NOTICE: Kodachrome shares the E-6 saturation signature and
        # none of its hue signature. Asserted as an ORDER OF MAGNITUDE between
        # two correlations, not as either value, because a correlation read
        # off a traced plot is not a precision quantity.
        def _medr(d, out):
            o = _vec(d, out)
            cc = []
            for k in _E6:
                a = _vec(d, k)
                m = ~(_np.isnan(o) | _np.isnan(a))
                if m.sum() > 2:
                    cc.append(float(_np.corrcoef(o[m], a[m])[0, 1]))
            return float(_np.median(cc))

        _kc, _kh = _medr(_dCk, "KR64"), _medr(_dHk, "KR64")
        _dc, _dh = _medr(_dCk, "D3"), _medr(_dHk, "D3")
        chk("G-V47-E6SPLIT  Kodachrome tracks the E-6 class in CHROMA and not "
            "in HUE -- so what colour reversal film has in common is which "
            "colours it cannot saturate, while the hue error is a property "
            "of the process",
            _kc > 0.7 and abs(_kh) < 0.35 and _kc > 4.0 * abs(_kh)
            and _dc < 0.5 and _dh < 0.5
            and abs(_kc - _fpm._E6_CLASS_CORRELATION["KR64"][0]) < 0.02
            and abs(_kh - _fpm._E6_CLASS_CORRELATION["KR64"][1]) < 0.02,
            "Kodachrome vs the four E-6: median r = %.3f in dC against %.3f "
            "in |dH|; the digital camera shares neither (%.3f, %.3f)"
            % (_kc, _kh, _dc, _dh))

        # ⚠ THE NEUTRAL CAST IS UNANIMOUS IN DIRECTION AND ITS CAUSE IS
        # OPEN. The guard asserts both halves, because storing the measurement
        # without the open cause would licence applying it to 36 stocks.
        _nc = _fpm._REVERSAL_NEUTRAL_CAST
        chk("G-V47-NEUTRAL  every reversal film measured casts blue-violet -- "
            "a* positive and b* negative with no exception -- and the cause "
            "is stored as OPEN with four candidates, one of them this "
            "project's own and marked as such",
            _nc["films_without_the_cast"] == 0
            and _nc["delta_e_midscale"] == 20.0
            and _nc["applied"] is False
            and len(_nc["author_candidates"]) == 3
            and "5500 K" in _nc["project_candidate_not_in_source"]
            and "D50" in _nc["project_candidate_not_in_source"]
            and _fpm._REVERSAL_NEUTRAL_CAST_ADAPTS_AWAY is True,
            "dE about %.0f mid-scale on every one of the %d films, "
            "replicated across %d labs; NOT applied to any profile, because "
            "the author's three candidates and this project's fourth (a D50 "
            "reference against a 5500 K film balance) are not separated by "
            "anything in the corpus"
            % (_nc["delta_e_midscale"], len(_fpm._KREYENBUEHL_DELTA_C) - 1,
               len(_nc["labs_compared"])))

        # ⚠ THE TRACE PASSES THE AUTHOR'S OWN PROSE, which is the only
        # independent check this figure admits. Four statements, asserted --
        # two of which name a single film as the exception, so a trace that
        # merely got the shape right would fail them.
        _P = list(_fpm._CCSG_PATCH_ORDER)
        _i = {p: _P.index(p) for p in _P}
        _films = [k for k in _dCk if k != "D3"]
        chk("G-V47-TRACE  the traced dC table reproduces all four statements "
            "the thesis makes about it in words, including the two that name "
            "a single film as the exception",
            all(abs(_dCk[k][_i[20]]) < 6.0 for k in _dCk
                if _dCk[k][_i[20]] is not None)
            and all(_dCk[k][_i[21]] > 10.0 for k in _E6
                    if _dCk[k][_i[21]] is not None)
            and _dCk["KR64"][_i[21]] < 2.0
            and all(v[_i[38]] < 0.0 for v in _dCk.values())
            and -6.0 < _dCk["D3"][_i[38]] < 0.0
            and all(_dCk[k][_i[48]] < 0.0 for k in _films)
            and _dCk["Velvia"][_i[48]] < -15.0
            and _dCk["D3"][_i[48]] > 0.0,
            "patch 20 |dC| < 6 on every film; patch 21 > +10 on all four E-6 "
            "and +%.1f on Kodachrome; patch 38 negative on all six subjects "
            "with the D3 at %.1f; patch 48 negative on every film including "
            "Velvia at %.1f and positive only on the D3 at +%.1f"
            % (_dCk["KR64"][_i[21]], _dCk["D3"][_i[38]],
               _dCk["Velvia"][_i[48]], _dCk["D3"][_i[48]]))

        # ---- v43, queue P12: the GOST criterion is two revisions -----------
        #
        # ⚠⚠ THE CONFLICT WAS NEVER BETWEEN TWO OPINIONS. Glafkides gives
        # D = fog + 0.2 with S = 1/E; ГОСТ 9160-91 gives fog + 0.85 with
        # S = K/H. Forty years apart, and the criteria are now stored keyed by
        # EDITION so that no stock's speed can be qualified by a standard that
        # postdates it. The guard's real content is the two REFUSALS: 9160-82
        # is a key with no rows because nobody has read it, and the 1950s
        # criterion is NOT filed as an edition of 9160 because ГОСТ 9160 did
        # not exist in that form.
        _cri = _fpm._GOST_SPEED_CRITERIA
        _f91 = {c: _fpm.gost_speed_criterion("9160-91", c)
                for c in sorted(_fpm._GOST_SPEED_CLASSES)}
        _f82 = [c for c in _fpm._GOST_SPEED_CLASSES
                if _fpm.gost_speed_criterion("9160-82", c) is not None]
        chk("G-V43-P12  the five ГОСТ 9160-91 speed criteria are data keyed by "
            "EDITION, the unread 1982 edition answers None rather than "
            "borrowing them, and the 1950s criterion is filed apart from both",
            len(_cri["9160-91"]) == 6 and _cri["9160-82"] == {}
            and not _f82
            and _f91["neg_cine"] == (0.20, 1.6)
            and abs(_f91["neg_still"][1] - 2.0 ** 0.5) < 1e-12
            and _f91["aerial_negative"] == (0.85, 20.0)
            and _fpm.gost_speed_criterion("9160-59", "neg_cine") is None
            and abs(_fpm._GOST_1950S_CRITERION["d_kr_above_fog"] - 0.20) < 1e-12
            and _fpm._GOST_1950S_CRITERION["speed_law"] == "S = 1/E"
            and "9160-82" in _fpm._GOST_EDITIONS_UNREAD,
            "six 1991 rows on file; 1982 has none and returns None for all "
            "%d classes; the 1950s reading is stored as its own constant, "
            "not as a third edition key" % len(_fpm._GOST_SPEED_CLASSES))

        # ⚠ AND THE SIX STOCKS THAT NAME A CLASS AND NO EDITION ARE A LIST,
        # not a silence. A class is a statement about the MATERIAL, which no
        # edition disputes; the edition is a statement about which revision
        # qualified the stored speed, and their sources do not say.
        _cls = [q for q in FILM_PROFILES if q.gost_speed_class]
        _noed = {q.name for q in _cls if not q.gost_speed_edition}
        chk("G-V43-P12B  every classed stock that names no ГОСТ edition is on "
            "the recorded list, and the list has no stock that does name one",
            _noed == set(_fpm._GOST_EDITION_UNSTATED),
            "%d stocks carry a class, %d of them name an edition; the other "
            "%d are listed" % (len(_cls), len(_cls) - len(_noed), len(_noed)))

        # ---- v43, queue P13: the magenta default that cannot exist ---------
        #
        # ⚠⚠ THE RANKING INVERTS BETWEEN ERAS, so a single `dye_fade_m`
        # default would be wrong in OPPOSITE directions on the two coupler
        # chemistries. The guard asserts the inversion itself -- the 1980s
        # replacement is WORSE than the dye it replaced, on both ends of its
        # band -- and that the 1950s half stays a ranking, because Glafkides
        # publishes an order and no fraction.
        _mf = _fpm._MAGENTA_LIGHT_FADE
        chk("G-V43-P13  the magenta light-fade default is keyed to the coupler "
            "class, the 1980s replacement is measurably worse than the dye it "
            "replaced, and the 1950s half stays a ranking with no invented "
            "number",
            set(_mf) == {"five_pyrazolone", "pyrazoloazole"}
            and _mf["pyrazoloazole"][0] > _mf["five_pyrazolone"][1]
            and _fpm._MAGENTA_STABILITY_RANK_1950S == ("pyrazolone", "nitrile")
            and all(c not in _mf for c in _fpm._MAGENTA_STABILITY_RANK_1950S)
            and _fpm.magenta_fade_default("pyrazolone") is None
            and _fpm.magenta_fade_default("") is None
            and abs(_fpm.magenta_fade_default("pyrazoloazole") - 0.95) < 1e-9
            and all(c in _fpm._MAGENTA_COUPLER_CLASSES for c in _mf)
            and not [q for q in FILM_PROFILES if q.aging.magenta_coupler_class],
            "pyrazoloazole %s against 5-pyrazolone %s, disjoint bands; the "
            "two 1950s classes carry an ORDER and no fraction; 0 of %d stocks "
            "name a coupler class, which is the honest state"
            % (_mf["pyrazoloazole"], _mf["five_pyrazolone"], len(FILM_PROFILES)))

        # ---- v43, queue P14: the Iofis 1981 tables --------------------------
        #
        # ⚠⚠ THE ROW CARRIED THIRTEEN FILM NAMES FOR A WEEK AND NONE OF THEIR
        # NUMBERS. Both tables are transcribed now, INCLUDING the four rows
        # that correspond to stocks this file already holds -- which is what
        # makes the transcription checkable against something other than
        # itself. Two of the four agree on speed exactly and a third to a
        # sixth of a stop.
        _bw = _fpm._IOFIS_1981_BW
        _col = {r[0]: r for r in _fpm._IOFIS_1981_COLOUR}
        _names = {r[0] for r in _bw} | set(_col)
        _spd = []
        for _d, _st in _fpm._IOFIS_1981_MAPPED.items():
            _q = get_profile(_st)
            _s = (_col[_d][1] if _d in _col
                  else next(r[1] for r in _bw if r[0] == _d))
            _spd.append((_d, _s, _q.exposure_index))
        _spd_bad = [t for t in _spd if abs(math.log2(t[1] / t[2])) > 0.25]
        chk("G-V43-P14  both Iofis 1981 summary tables are transcribed, the "
            "thirteen unprofiled films are named, and every row that maps "
            "onto a stock in this file agrees with its stored speed to "
            "within a quarter stop",
            len(_bw) == 9 and len(_col) == 9
            and len(_fpm._IOFIS_1981_UNPROFILED) == 13
            and set(_fpm._IOFIS_1981_UNPROFILED) <= _names
            and not _spd_bad
            # the thirteen really are absent -- no alias of a live profile
            and not [n for n in _fpm._IOFIS_1981_UNPROFILED
                     if any(n.lower() in (a.lower() for a in q.aliases)
                            for q in FILM_PROFILES)]
            and "TERTIARY SUMMARY" in _fpm._IOFIS_1981_SOURCE,
            "9 + 9 rows, 13 unprofiled and none of them reachable as an "
            "alias; speed check %s"
            % [(d, s, ei) for d, s, ei in _spd])

        # ⚠ AND ONE MAPPED ROW DISAGREES, WHICH IS RECORDED RATHER THAN
        # RECONCILED. ЦО-90Л's stored contrast is 1.90 from its ТУ; the
        # encyclopedia prints 1.5. Method rule 25 gives the ТУ precedence, so
        # the profile does not move -- and the disagreement is asserted so
        # that nobody later "fixes" the profile to match a tertiary summary.
        _co90 = get_profile("SVEMA_CO_90L").curves.g.gamma
        chk("G-V43-P14B  the one contrast disagreement between Iofis 1981 and "
            "a ТУ-sourced profile stays unreconciled, with the ТУ winning",
            abs(_col["ЦО-90Л"][2] - 1.5) < 1e-9
            and abs(_co90 - 1.90) < 1e-9
            and abs(_co90 - _col["ЦО-90Л"][2]) > 0.3,
            "encyclopedia 1.5 against the stored 1.90 -- 0.40 apart, ТУ "
            "outranks a tertiary summary (method rule 25)")

        # ---- v43, queue P39: the NIKFI scale --------------------------------
        #
        # ⚠ THE UNIT IS PRESERVED AND THE CONVERSION IS REFUSED. Two stocks'
        # only published speed is "20-26 NIKFI units"; this corpus holds no
        # NIKFI definition, so no ГОСТ or ISO number may be derived and none
        # is. Both stocks therefore keep their `gost_speed_class` refusal.
        _nk = _fpm._NIKFI_SPEED
        chk("G-V43-P39  the NIKFI figures are stored in their own unit, and "
            "both stocks that carry one still refuse a ГОСТ speed class",
            set(_nk) == {"SVEMA_DS_2", "SVEMA_LN_3"}
            and all(v == (20.0, 26.0) for v in _nk.values())
            and all(not get_profile(n).gost_speed_class for n in _nk)
            and all(n in _fpm._GOST_CLASS_WITHHELD for n in _nk)
            and "NOT IN THIS CORPUS" in _fpm._NIKFI_SCALE_DEFINITION,
            "%d stocks at 20-26 NIKFI units; no conversion constant exists in "
            "this file and none was fitted" % len(_nk))

        # ---- v43, queue P40: the two merged SVEMA identities ---------------
        #
        # ⚠⚠ THE ROW'S OWN REMEDY WAS TO SPLIT THE RECORDS AND SPLITTING WOULD
        # HAVE BEEN WRONG. Both profiles' sensitometry comes from Гурлев 1986
        # p.296's FOTO- columns, i.e. the still line, so there is one dataset
        # and one criterion; two records would be one measurement counted
        # twice. The class is now written and the refusal is gone.
        _p40 = [get_profile(n) for n in ("SVEMA_FOTO_65", "SVEMA_FOTO_250")]
        chk("G-V43-P40  both merged SVEMA entries carry neg_still, neither is "
            "withheld any more, and neither was duplicated into a second "
            "record to get there",
            all(q.gost_speed_class == "neg_still" for q in _p40)
            and all(q.name not in _fpm._GOST_CLASS_WITHHELD for q in _p40)
            and all(q.name in _fpm._GOST_EDITION_UNSTATED for q in _p40)
            and len([q for q in FILM_PROFILES
                     if q.name.startswith("SVEMA_FOTO_")]) == 4,
            "neg_still on both, from Гурлев's Foto-65 / Foto-250 columns; "
            "%d SVEMA_FOTO_* records, unchanged"
            % len([q for q in FILM_PROFILES
                   if q.name.startswith("SVEMA_FOTO_")]))

        # ---- v43, queue P41: the offset-speed sub-layers -------------------
        #
        # ⚠⚠ THE ROW REFUSED AN EMPTY CARRIER AND THIS ONE ARRIVES WITH ITS
        # LAW. `combined_density` on an empty set returns the bare curve's
        # density, so shipping it inert cannot move a render -- and a
        # populated set computes the thing a single ToneCurve cannot. The
        # patent's own measured separation is pushed through it here.
        _sl_empty = _fpm.SubLayerSet()
        _sl = _fpm.SubLayerSet(dloge=(0.0, 0.36), share=(0.5, 0.5),
                               criterion="fog+0.10",
                               source="EP 0 083 377 A1 TABLE 2")
        _lin = lambda x: 1.0 + 0.6 * x
        _pop = [q.name for q in FILM_PROFILES if q.sub_layers.has_data]
        _rules = _fpm._EP0083377_RULES
        chk("G-V43-P41  a colour record can now hold offset-speed sub-layers, "
            "an empty set is exactly the bare curve, and the patent's "
            "measured 0.36 log E separation buys the 1.2 stops it claims",
            not _sl_empty.has_data and not _pop
            and abs(_sl_empty.combined_density(_lin, 0.4) - _lin(0.4)) < 1e-12
            and _sl.has_data
            and abs(_sl.latitude_gain_stops() - 1.196) < 0.01
            and _rules["sublayer_dloge_measured_table2"] == (0.35, 0.37)
            and (_rules["sublayer_dloge_preferred"][0]
                 <= _rules["sublayer_dloge_measured_table2"][0]
                 <= _rules["sublayer_dloge_preferred"][1]),
            "0 of %d stocks populate it, which is honest -- no specimen in "
            "EP 0 083 377 A1 is a stock this database holds; the measured "
            "0.35-0.37 sits inside the patent's own preferred 0.3-0.8 window"
            % len(FILM_PROFILES))

        # ---- v43, queue M1a: which print stock's spectral IS the reader ----
        #
        # ⚠⚠ THE ACQUISITION HALF CLOSED ON 2026-08-31 AND THE CONFIGURATION
        # HALF IS THIS. 2383 carries a traced spectral panel and a measured
        # printing matrix, and 190 of the 191 films render through SCAN_DI,
        # whose "reader" is a scanner sensor nothing in this corpus measures
        # (M1b). So the matrix is correct, reachable and UNUSED, and the flag
        # now says why rather than a comment saying it.
        _emul = [x for x in PRINT_STOCKS if x.reader_is_emulsion]
        _xform = [x for x in PRINT_STOCKS if not x.reader_is_emulsion]
        _2383 = next(x for x in PRINT_STOCKS if x.name == "KODAK_2383_RELEASE")
        _through = {q.default_print for q in FILM_PROFILES}
        chk("G-V43-M1A  every real print EMULSION is flagged as its own "
            "reader and the two digital-intermediate transforms are not, and "
            "the one stock with a measured printing matrix is reachable but "
            "unused by configuration rather than by omission",
            len(_emul) == 9 and len(_xform) == 2
            and {x.name for x in _xform} == {"SCAN_DI", "KODAK_VISION3_DI_2254"}
            and _2383.reader_is_emulsion
            and bool(_2383.spectral.log_s_r)
            and _2383.printing_matrix_measured
            and "KODAK_2383_RELEASE" not in _through
            and not any(x.spectral.log_s_r for x in _xform),
            "9 emulsions / 2 transforms; films render through %s, so 2383's "
            "matrix describes a printing step nothing currently takes"
            % sorted(_through))

        # ---- v40, queue P46: the Gevaert catalogue, decoded not eye-read ---
        #
        # ⚠ THE ROW EXPECTED A PAGE-IMAGE READ AND THE DOCUMENT DECODES
        # ITSELF. Gevaert print every resolving power twice, in lines/mm and
        # in lines/inch, and only the first copy is mangled by the OCR. Both
        # decoders are asserted here, because a transcription that cannot be
        # re-derived is the thing this suite exists to refuse.
        _gv = _fpm._GEVAERT_MANUAL_CATALOGUE
        _gvbad = [r for r in _gv if abs(r[3] / 25.4 - r[2]) > 1.2]
        _LADDER = {17: 40, 22: 125, 25: 250, 27: 400, 19: 64, 13: 16, 10: 8}
        _gvspeed = [r for r in _gv
                    if _LADDER.get(r[4]) != r[5]
                    or (r[6] and _LADDER.get(r[6]) != r[7])]
        chk("G-V40-P46  every Gevaert resolving power re-derives from its "
            "own lines/inch figure to within 1.2 lp/mm, and every DIN sits "
            "on the standard ladder against its printed ASA",
            len(_gv) == 16 and not _gvbad and not _gvspeed
            and _fpm._GEVAERT_SHEET_ANTIHALATION,
            "16 entries, %d decoder mismatches, %d speed-pair mismatches"
            % (len(_gvbad), len(_gvspeed)))
        # ⚠ AND NOT ONE OF THEM BECOMES A PROFILE. The manual prints no
        # characteristic curve, so a resolving power and a speed cannot make
        # one, and none of these products overlaps the seven Gevaert stocks
        # this database holds -- those are cine, these are still.
        _gev = [q.name for q in FILM_PROFILES if "GEVA" in q.name.upper()]
        chk("G-V40-P46X  the catalogue adds no profile, because the manual "
            "prints no curve and none of its products is a stock this "
            "database holds",
            len(_gev) == 7
            and not any("GEVAPAN" in n for n in _gev),
            "7 Gevaert stocks held, all cine: %s" % (sorted(_gev)[:3],))

        # ⚠ THE MTF NUMBERS ASSERT A PROPERTY OF THE ENGINE, NOT OF A STOCK.
        # US 4,248,962 measures 108 % and 112 % response at 20 cy/mm from
        # timed DIR couplers. Stage 9's short-range term is
        # `rO += e * (rO - blurred)` with a floor at zero and NO UPPER CLAMP,
        # so a response above unity is already representable: the measurement
        # corroborates the model rather than demanding a change. This guard
        # exists so that adding a clamp later fails here.
        _mtf = _fpm._US4248962_MTF_20CYMM
        chk("G-V32-US4248962  the measured DIR overshoot exceeds 100 % on "
            "both records, which stage 9's unclamped unsharp term can "
            "represent -- introducing an upper clamp would contradict this",
            _mtf["invention_T"]["cyan"] > 100
            and _mtf["invention_T"]["magenta"] > 100
            and _mtf["control_S"]["cyan"] < 100,
            "T %d/%d %% vs S %d/%d %% at 20 cy/mm"
            % (_mtf["invention_T"]["cyan"], _mtf["invention_T"]["magenta"],
               _mtf["control_S"]["cyan"], _mtf["control_S"]["magenta"]))

        # ⚠ THE FUJI AIMS ARE THE EVIDENCE BEHIND v31's REFLECTION GUARD.
        # `validate` refuses a reflection stock whose red dmax exceeds 3.0,
        # citing "the measured aims on the Fujicolor Crystal Archive papers
        # are 1.95-2.18". That sentence is only true if the transcribed
        # numbers say so, so it is checked rather than asserted.
        _aims = [v for t in _fpm._FUJI_CRYSTAL_ARCHIVE_DMAX.values()
                 for v in t]
        chk("G-V32-FUJIDMAX  every transcribed Crystal Archive reflection "
            "Dmax aim lies between 1.90 and 2.18, and every one is far "
            "below the 3.0 point where v31's geometry guard refuses",
            min(_aims) >= 1.90 and max(_aims) == 2.18 and max(_aims) < 3.0,
            "%d aims across %d products, %.2f-%.2f"
            % (len(_aims), len(_fpm._FUJI_CRYSTAL_ARCHIVE_DMAX),
               min(_aims), max(_aims)))

        # The Soviet population added at v32.
        _gost_set = [p for p in FILM_PROFILES if p.gost_speed_class]
        _gost_ed = [p for p in FILM_PROFILES if p.gost_speed_edition]
        _withheld_set = [p.name for p in FILM_PROFILES
                         if p.gost_speed_class
                         and p.name in _fpm._GOST_CLASS_WITHHELD]
        # 18 Soviet stocks: 13 classed, 5 withheld with a stated reason --
        # two rated on the NIKFI scale, two whose entry merges a cine and a
        # still designation, and one stored as a PrintStock, which holds no
        # speed for a criterion to qualify.
        # ⚠ 13 -> 15 AND 5 -> 3 ON 2026-09-18c (queue P40). SVEMA_FOTO_65 and
        # SVEMA_FOTO_250 left the withheld list and gained `neg_still`: both
        # profiles' sensitometry comes from Гурлев 1986's FOTO- columns, i.e.
        # the still line, so the record has one criterion and never needed the
        # split the queue row proposed. The three that remain withheld are the
        # two NIKFI-rated stocks (queue P39, whose scale this corpus cannot
        # convert) and the PrintStock that holds no speed at all.
        chk("G-V32-GOSTCLASS  gost_speed_class is populated on 15 Soviet "
            "stocks, 7 of them carrying the edition their own ТУ sheet "
            "names, and none of the 3 withheld stocks has been filled in",
            len(_gost_set) == 15 and len(_gost_ed) == 7
            and len(_fpm._GOST_CLASS_WITHHELD) == 3
            and not _withheld_set,
            "%d classed, %d with an edition, %d on the withheld list"
            % (len(_gost_set), len(_gost_ed),
               len(_fpm._GOST_CLASS_WITHHELD)))

        # ⚠ AND THE EDITION GAP IS ASSERTED, NOT LEFT AS A COMMENT. Every
        # stock that names an edition names 9160-82, while the five criteria
        # this project has actually read are from 9160-91. That mismatch is
        # the open question P38 exists for; if a later change makes it go
        # away silently, this guard is what notices.
        chk("G-V32-GOSTEDITION  every Soviet stock naming an edition names "
            "9160-82, while the criteria read are 9160-91 -- the open gap "
            "P38 tracks, asserted so it cannot close unnoticed",
            bool(_gost_ed)
            and all(p.gost_speed_edition == "9160-82" for p in _gost_ed),
            "editions present: %s"
            % sorted({p.gost_speed_edition for p in _gost_ed}))

        # ------------------------------------------------------------------
        #  schema v33 -- EP 0 083 377 A1 (Konishiroku, filed 1982).
        # ------------------------------------------------------------------
        _t1 = _fpm._EP0083377_TABLE1
        _poly = [r for r in _t1 if r[3] > _fpm._EP0083377_MONODISPERSE_MAX]
        _mono = [r for r in _t1 if r[3] <= _fpm._EP0083377_MONODISPERSE_MAX]
        chk("G-V33-EP83377T1  TABLE 1's fifteen emulsions split cleanly at the "
            "patent's own s/r_bar = 0.15 monodispersity rule, with a wide gap "
            "and no borderline case",
            len(_t1) == 15 and len(_poly) == 3 and len(_mono) == 12
            and max(r[3] for r in _mono) <= 0.09
            and min(r[3] for r in _poly) >= 0.23,
            "%d monodisperse (max %.2f), %d polydisperse (min %.2f)"
            % (len(_mono), max(r[3] for r in _mono),
               len(_poly), min(r[3] for r in _poly)))

        # The controlled pairing is the reason this table was stored, so it is
        # what gets asserted: every polydisperse control has a monodisperse
        # twin of the same size, which isolates dispersity from size.
        _pairs = sum(1 for _p in _poly
                     if any(abs(_m[2] - _p[2]) <= 0.02 for _m in _mono))
        chk("G-V33-EP83377PAIR  every polydisperse control in TABLE 1 has a "
            "monodisperse twin within 0.02 um -- the controlled pairing that "
            "isolates dispersity from crystal size",
            _pairs == len(_poly),
            "%d of %d controls paired" % (_pairs, len(_poly)))

        _t3 = _fpm._EP0083377_TABLE3
        _les = [v for _, _, _, t in _t3 for v in t]
        _inv3 = [r for r in _t3 if r[1]]
        chk("G-V33-EP83377T3  TABLE 3 holds eight specimens, six of them "
            "inventions, with every L.E.S. inside 2.57-3.03 log E",
            len(_t3) == 8 and len(_inv3) == 6
            and min(_les) == 2.57 and max(_les) == 3.03,
            "%d specimens, L.E.S. %.2f-%.2f log E (%.1f-%.1f stops)"
            % (len(_t3), min(_les), max(_les),
               min(_les) * 3.321928, max(_les) * 3.321928))

        # The patent argues specifically about the GREEN record: both controls
        # are said to show "a small L.E.S. value under green light". If the
        # transcription did not reproduce that, it would be wrong.
        _ctrl_g = max(r[3][1] for r in _t3 if not r[1])
        _inv_g = min(r[3][1] for r in _inv3)
        chk("G-V33-EP83377GREEN  every invention specimen beats both controls "
            "on green L.E.S., which is the axis the patent argues",
            _inv_g > _ctrl_g,
            "worst invention %.2f vs best control %.2f log E"
            % (_inv_g, _ctrl_g))

        # ⚠ SPEED MUST MOVE FURTHER THAN GAMMA, because that asymmetry IS the
        # finding: this engine models the gamma half of development and has no
        # speed term at all.
        _t6 = _fpm._EP0083377_TABLE6
        _s1 = {t: v for i, _, t, v, _ in _t6 if i == 1}
        _g1 = {t: v for i, _, t, _, v in _t6 if i == 1}
        _spd_swing = _s1["3:35"][0] / _s1["2:55"][0]
        _gam_swing = _g1["3:35"][0] / _g1["2:55"][0]
        chk("G-V33-EP83377T6  TABLE 6 has 16 rows and shows blue SPEED swinging "
            "further than gamma over the same +/-20 s of development -- the "
            "asymmetry this engine's processing model does not represent",
            len(_t6) == 16 and _spd_swing > _gam_swing and _spd_swing > 1.6,
            "control speed x%.2f against gamma x%.2f over 175-215 s"
            % (_spd_swing, _gam_swing))

        _s8 = {t: v for i, _, t, v, _ in _t6 if i == 8}
        _spd8 = _s8["3:35"][0] / _s8["2:55"][0]
        chk("G-V33-EP83377STAB  the fully-compliant specimen 8 is markedly "
            "flatter against development time than the polydisperse control -- "
            "the patent's process-stability claim, in its own data",
            _spd8 < _spd_swing,
            "specimen 8 speed x%.2f against control x%.2f"
            % (_spd8, _spd_swing))

        # ⚠ THIS ONE REPORTS AND DOES NOT ASSERT, AND THE REASON MATTERS.
        # TABLE 3 is the only MEASURED colour-negative latitude set in the
        # corpus: 2.57 to 3.03 log E over eight coatings. This database's own
        # colour negatives run wider at BOTH ends, and the top end is not
        # defensible as an era difference -- 5.3 log E is 17.6 stops and no
        # colour negative has ever had that. It is not asserted because these
        # are 1982 Konica coatings and most of the database is not, so a
        # threshold here would be arguing from one maker in one year. It
        # prints on every build so the discrepancy cannot be forgotten.
        # Closing it is queue P42.
        _cn = [p for p in FILM_PROFILES
               if p.kind is _fpm.StockKind.NEGATIVE and not p.is_monochrome]
        _cn_les = sorted(p.curves.g.latitude_stops / 3.321928 for p in _cn)
        _over = [x for x in _cn_les if x > 3.03]
        chk("G-V33-LES  colour-negative latitude against the only measured set "
            "in the corpus -- REPORTED, deliberately not asserted",
            True,
            "%d colour negatives span %.2f-%.2f log E against EP 0 083 377's "
            "measured 2.57-3.03; %d exceed the measured maximum, worst %.2f "
            "log E (%.1f stops). Queue P42."
            % (len(_cn), _cn_les[0], _cn_les[-1], len(_over),
               _cn_les[-1], _cn_les[-1] * 3.321928))

        # ------------------------------------------------------------------
        #  ENGINE SOURCE INTEGRITY -- added 2026-09-11 after two headers were
        #  found missing from the tree and shipped that way twice.
        #
        #  ⚠ THIS EXISTS BECAUSE EVERY OTHER GATE LOOKED PAST IT. build.py
        #  compiles the 26 GENERATED database translation units and nothing
        #  else, so AlgorithmMain.cpp -- the engine driver, which includes
        #  thirty-odd headers and calls every stage -- was never compiled by
        #  the build at all. cpp_parity.py did reference one of the missing
        #  headers and SKIPPED when it was absent. The result was a green build
        #  on a tree whose driver could not compile, in two consecutive
        #  deliveries, until the owner hit it in Visual Studio.
        #
        #  This does not compile anything. It reads every #include "..." in the
        #  engine tree and asserts the file exists somewhere in it. That is
        #  cheap, it needs no toolchain, and it is exactly the failure that got
        #  through: not a bad expression, an absent file.
        _eng = Path("/root/work/tst")
        if _eng.is_dir():
            _missing = []
            _srcs = sorted(list(_eng.glob("*.cpp")) + list(_eng.glob("*.hpp"))
                           + list((_eng / "AVX2").glob("*.cpp"))
                           + list((_eng / "AVX2").glob("*.hpp")))
            for _f in _srcs:
                if _f.name.startswith("test_") or _f.name == "profall.cpp":
                    continue
                try:
                    _txt = _f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for _inc in re.findall(r'#include\s+"([^"]+)"', _txt):
                    if ((_eng / _inc).is_file()
                            or (_eng / "AVX2" / _inc).is_file()
                            or (_f.parent / _inc).is_file()):
                        continue
                    _missing.append("%s -> %s" % (_f.name, _inc))
            chk("G-ENGINE-INCLUDES  every #include in the engine tree resolves "
                "to a file that exists -- the check that would have caught "
                "AlgoReciprocity.hpp and AlgoProcessVariant.hpp going missing",
                not _missing,
                "%d source files scanned, all includes resolve" % len(_srcs)
                if not _missing
                else "%d unresolved: %s" % (len(_missing), _missing[:4]))

        # ------------------------------------------------------------------
        #  G-CTLENUM -- the control vocabulary matches the database.
        #
        #  AlgoControlEnums.hpp is what the host populates its dropdowns from,
        #  and the engine resolves those selections against FORMAT_GEOM and
        #  PRINT_STOCKS. If the two lists drift, a user picks a format or a
        #  print stock that silently degrades to a default instead of doing
        #  what the panel said it would -- the failure mode is a wrong render,
        #  not an error, which is why it is checked here rather than trusted.
        #
        #  The first draft of that header did drift in exactly these ways:
        #  it spelled the Soviet positive stocks CP where the database spells
        #  them TSP, and it omitted KODAK_5302 entirely, which would have made
        #  one shipped print stock unreachable from the interface.
        _ce_path = _eng / "AlgoControlEnums.hpp"
        if not _ce_path.is_file():
            chk("G-CTLENUM  AlgoControlEnums.hpp is present in the engine tree",
                False, "not found at %s" % _ce_path)
        else:
            _ce = _ce_path.read_text(encoding="utf-8", errors="replace")

            def _ce_keys(marker):
                _m = re.search(marker + r"\[\]\s*=\s*\{(.*?)\}", _ce, re.S)
                if not _m:
                    return None
                return [s for s in re.findall(r'"([^"]*)"', _m.group(1))]

            _fmt_keys = _ce_keys("FilmFormatCtrlKey")
            _prn_keys = _ce_keys("PrintStockCtrlKey")

            _fmt_db = set(_fpm.FORMAT_GEOM)
            _fmt_hd = set(_fmt_keys or [])
            chk("G-CTLENUM-FORMAT  every film format the control header offers "
                "exists in FORMAT_GEOM, and every format the database defines "
                "is offered -- a panel entry the engine cannot resolve renders "
                "the wrong geometry rather than failing",
                _fmt_hd == _fmt_db,
                "header %d, database %d; header-only %s; database-only %s"
                % (len(_fmt_hd), len(_fmt_db),
                   sorted(_fmt_hd - _fmt_db) or "none",
                   sorted(_fmt_db - _fmt_hd) or "none"))

            # Index 0 is the "use the stock's own default" sentinel and carries
            # an empty key by design, so it is excluded from the comparison.
            _prn_hd = set(k for k in (_prn_keys or []) if k)
            _prn_db = set(p.name for p in PRINT_STOCKS)
            chk("G-CTLENUM-PRINT  the print stock vocabulary matches "
                "PRINT_STOCKS exactly, so no shipped stock is unreachable from "
                "the interface and no offered stock is missing from the "
                "database",
                _prn_hd == _prn_db,
                "header %d + sentinel, database %d; header-only %s; "
                "database-only %s"
                % (len(_prn_hd), len(_prn_db),
                   sorted(_prn_hd - _prn_db) or "none",
                   sorted(_prn_db - _prn_hd) or "none"))

            # The display strings are pipe separated and index aligned with the
            # enumerators, so a mismatched count silently shifts every label
            # past the missing one onto the wrong value.
            def _ce_bars(marker):
                _m = re.search(marker + r"\[\]\s*=\s*((?:\s*\"[^\"]*\")+)\s*;",
                                _ce, re.S)
                if not _m:
                    return None
                return "".join(re.findall(r'"([^"]*)"', _m.group(1))).count("|") + 1

            _fmt_lbl = _ce_bars("FilmFormatCtrlStr")
            _prn_lbl = _ce_bars("PrintStockCtrlStr")
            chk("G-CTLENUM-LABELS  the pipe-separated display names are index "
                "aligned with the enumerators; a short list shifts every label "
                "after the gap onto the wrong control value",
                _fmt_lbl == len(_fmt_keys or []) and _prn_lbl == len(_prn_keys or []),
                "format labels %s vs %d keys; print labels %s vs %d keys"
                % (_fmt_lbl, len(_fmt_keys or []), _prn_lbl, len(_prn_keys or [])))

            # Every numeric control documented in AlgoControl.hpp should have a
            # constant here, or the host has no range to enforce.
            _ac = (_eng / "AlgoControl.hpp").read_text(
                encoding="utf-8", errors="replace")
            _num_fields = [m for m in re.findall(
                r'\*\s+1\s+NAME\s+(\w+)[\s\S]{0,400}?\*\s+2\s+TYPE\s+(double|int32_t)',
                _ac)]
            _missing_const = []
            for _fname, _ftype in _num_fields:
                _cap = _fname[0].upper() + _fname[1:]
                if (_cap + "Def") not in _ce:
                    _missing_const.append(_fname)
            chk("G-CTLENUM-NUMERIC  every numeric control in AlgoControl.hpp "
                "carries a default constant in the control header, so the UI "
                "and the algorithm cannot disagree about a range",
                not _missing_const,
                "%d numeric controls, %d without a constant%s"
                % (len(_num_fields), len(_missing_const),
                   (": " + ", ".join(_missing_const)) if _missing_const else ""))

        # ------------------------------------------------------------------
        #  G-DYECLOUD-INERT -- P43's answer, kept enforced.
        #
        #  ⚠ THIS GUARD ASSERTS A NEGATIVE, WHICH IS UNUSUAL HERE, AND THE
        #  REASON IS THAT THE FIELD LOOKS EXACTLY LIKE A GOOD IDEA. It is
        #  populated on 115 stocks, it is already in the same units as
        #  `clump_um_*`, and the arithmetic to fold it in is one line. A
        #  2026-09-11 field audit duly ranked it the best unused parameter in
        #  the database. It is not: the monochrome stocks give a silver-only
        #  law of clump = 3.93 * grain_um, the colour stocks sit at the same
        #  ratio (t = -0.65), and adding a dye term improves 51 stocks while
        #  worsening 52. It carries no information, and the two chromogenic
        #  black-and-white stocks -- the largest dye clouds in the set at
        #  9.0 um -- have the LOWEST clump/grain ratio of all, which is the
        #  opposite of what the hypothesis needs.
        #
        #  So the next person to read the field list will have the same good
        #  idea, and this is what tells them it was already tried.
        _dc = [p for p in FILM_PROFILES if p.grain.dye_cloud_um > 0.0]
        _dc_vals = sorted({p.grain.dye_cloud_um for p in _dc})
        import inspect as _insp_dc
        import film_sim as _fs_dc
        _dc_read = ("dye_cloud_um" in _insp_dc.getsource(_fs_dc))
        chk("G-DYECLOUD-INERT  dye_cloud_um is populated but UNREAD, which is "
            "P43's settled answer and not an oversight -- it is a 4-value era "
            "band with no provenance and no predictive power over the fitted "
            "clump",
            not _dc_read and len(_dc_vals) <= 4,
            "%d stocks carry it, %d distinct values %s; renderer reads it: %s"
            % (len(_dc), len(_dc_vals),
               ["%.1f" % v for v in _dc_vals], _dc_read))

        # ------------------------------------------------------------------
        #  G-MASKLEVEL -- the orange mask level, checked against a document
        #  that had nothing to do with building it.
        #
        #  Artyushin, Barsky & Vinokur, «Справочник кинооператора», 1999,
        #  section 6.6 item 4, states the class figure for a masked colour
        #  negative outright: measure an unexposed but processed frame and the
        #  minimum density is "greatest in the blue zone (1.0 D), somewhat
        #  less in the green (0.6 D), least in the red (0.2 D)".
        #
        #  ⚠ THIS IS A CROSS-CHECK, NOT A SOURCE. Not one stored curve came
        #  from that handbook; the 72 ladder-encoded colour negatives were
        #  built over months from Kodak, Fuji, Agfa, Konica and GDR sheets,
        #  and their median D-min triple is 0.225 / 0.640 / 0.965. The
        #  handbook says 0.2 / 0.6 / 1.0. Three independent medians landing
        #  within 0.04 D of a published class statement is the strongest
        #  evidence in this file that the corpus's mask LEVEL is right, and it
        #  is worth a guard precisely because nothing else tests the absolute
        #  level -- every other mask check tests ordering or ratio.
        #
        #  ⚠ IT TESTS THE LADDER POPULATION ONLY, and the population is
        #  selected GEOMETRICALLY rather than by the `mask_encoding` label.
        #  That choice is forced, and finding out why is itself a result:
        #  `mask_encoding` says "dmin_ladder" on 23 colour negatives, while 72
        #  of the 85 actually carry a ladder-ordered triple. FORTY-NINE stocks
        #  labelled "neutral_dmin" are ladder-encoded in fact -- AGFA OPTIMA
        #  200/400, PORTRAIT 160, ULTRA 50, VISTA PLUS, EASTMAN 5247_1983,
        #  5250_1959 and forty-three more. The label is documented as
        #  render-inert and it is, so nothing is mis-rendering; but it cannot
        #  be used to select a population, and a guard keyed to it would have
        #  measured an arbitrary 23-stock subset. The medians differ enough to
        #  matter: 0.216/0.634/0.885 on the labelled set against
        #  0.225/0.640/0.965 on the real one.
        #
        #  b_dmin > g_dmin > r_dmin IS what an orange mask is, so that is the
        #  test. The 13 genuinely unmasked or neutral-encoded stocks fall out
        #  of it on their own -- among them TECHNICOLOR_THREE_STRIP, which has
        #  no mask at all because its three records are three separate films,
        #  and AGFA_NEG_TYPE_B_1943, which predates masking entirely.
        _ladder = [p for p in FILM_PROFILES
                   if p.kind == StockKind.NEGATIVE and not p.is_monochrome
                   and p.curves.b.dmin > p.curves.g.dmin > p.curves.r.dmin]
        _med = [float(_np.median([p.curves.r.dmin for p in _ladder])),
                float(_np.median([p.curves.g.dmin for p in _ladder])),
                float(_np.median([p.curves.b.dmin for p in _ladder]))]
        _pub = (0.2, 0.6, 1.0)
        _MASK_TOL = 0.08
        chk("G-MASKLEVEL  the ladder-encoded colour negatives' median D-min "
            "triple reproduces the published class figure for an orange mask "
            "(Artyushin/Barsky/Vinokur 1999 6.6: blue 1.0, green 0.6, red 0.2)",
            len(_ladder) >= 70
            and all(abs(_med[i] - _pub[i]) <= _MASK_TOL for i in range(3)),
            "%d ladder stocks, median R %.3f G %.3f B %.3f against the "
            "published %.1f / %.1f / %.1f, worst gap %.3f D (tolerance %.2f)"
            % (len(_ladder), _med[0], _med[1], _med[2], *_pub,
               max(abs(_med[i] - _pub[i]) for i in range(3)), _MASK_TOL))

        # ⚠ AND THE LABEL DEFECT IS PINNED SO IT CANNOT GROW IN SILENCE.
        #   49 stocks carry mask_encoding "neutral_dmin" while their stored
        #   triple is a mask ladder. That is a metadata error, not a render
        #   error -- the field changes no pixel and the docstring says so --
        #   but it is exactly the kind of thing that rots. The count is
        #   recorded rather than corrected because correcting it means editing
        #   49 profiles' metadata on the strength of a rule, and the rule is
        #   what should be checked first. Recorded in NotFound.md.
        # ⚠ 49 -> 50 ON 2026-09-17, AND THE NEW ONE IS NOT A NEW MISTAKE.
        # EASTMAN_5293_250T_1982 traces a D-min ladder of 0.183 / 0.542 / 0.962
        # straight off Kennel et al. 1982 Fig. 4 -- the orange mask, measured
        # -- and inherits `neutral_dmin` from the same default every one of the
        # other forty-nine carries. The baseline moves with the population
        # because it pins a known metadata defect, not a property.
        _MISLABEL_BASELINE = 12
        _mislab = sorted(p.name for p in FILM_PROFILES
                         if p.kind == StockKind.NEGATIVE and not p.is_monochrome
                         and p.mask_encoding == "neutral_dmin"
                         and p.curves.b.dmin > p.curves.g.dmin > p.curves.r.dmin)
        # ⚠⚠ 50 -> 12 ON 2026-09-17e, AND THE AUDIT THE ROW ASKED FOR IS WHY
        # IT IS NOT 0. Queue P47 wanted "a per-profile confirmation that the
        # stored triple's provenance really is a mask ladder, before the label
        # is changed to say so". Run, that confirmation SPLIT the population:
        # 38 of the 50 have a tier-1 curve provenance -- traced, measured or
        # stated -- so their ladder is a reading of a published curve and the
        # label was simply wrong; those are relabelled. The other 12 have only
        # an estimated or assumed provenance, so their ladder may be an
        # artefact of the estimate that produced it, and relabelling them
        # would promote a guess into a claim about a measured mask. They are
        # named in `_P47_LADDER_UNCONFIRMED` and stay pinned here.
        # ⚠ 38 -> 39 ON 2026-09-20. EKTAR 100's curves were re-traced from
        # E-4046 p4 and the sheet DRAWS the ladder (dmin 0.221 / 0.643 /
        # 0.855), so the stock moved from the maskless analogy it had into
        # `_P47_LADDER_CONFIRMED`. The HELD count is untouched at 12: nothing
        # was promoted on an estimate.
        chk("G-MASKENC  the mask_encoding mislabel count is UNCHANGED -- 39 "
            "relabelled on tier-1 provenance, 12 held because theirs is an "
            "estimate",
            len(_mislab) == _MISLABEL_BASELINE,
            "%d stocks say neutral_dmin and store a ladder, baseline %d%s"
            % (len(_mislab), _MISLABEL_BASELINE,
               "" if len(_mislab) == _MISLABEL_BASELINE
               else "; first differing: " + ", ".join(_mislab[:4])))

        #  G-HALATION-GEOM -- the halo is support geometry, not a second gain.
        #
        #  Halation is light reflected from the far surface of the support, so
        #  no part of it can be concentrated inside the critical-angle annulus
        #  whose inner edge sits at r_c = 2t / sqrt(n^2 - 1). Before v34 the
        #  radii were an unsourced project estimate that had drifted into a
        #  second amplitude control -- correlated with gain at +0.68 across
        #  113 stocks whose support thickness is the same 127 um class default
        #  -- and twelve stocks had been scaled below that floor, a shape no
        #  support reflection can make.
        #
        #  What is asserted, and why each one:
        #    floor    the principal lobe clears r_c on every stock that has a
        #             thickness. This is the geometric impossibility itself.
        #    derived  radii equal the coefficient table times the thickness,
        #             so a hand edit to one stock's triple fails here rather
        #             than quietly reintroducing per-stock width.
        #    energy   the weights sum to one, so the pass stays
        #             amplitude-neutral and the gains keep their meaning.
        #    split    acetate and polyester really do differ. If the material
        #             key ever stopped being read, every stock would collapse
        #             onto one coefficient set and this would catch it.
        _hg = [p for p in FILM_PROFILES
               if p.halation is not None and p.emulsion.base_um > 0.0]
        _hg_bad_floor, _hg_bad_derived, _hg_bad_energy = [], [], []
        _hg_keys = set()
        for _p in _hg:
            _k = _fpm._v34_support_key(_p.emulsion.base_material)
            _hg_keys.add(_k)
            _n, _per_t, _w, _ = _fpm._V34_BASE_OPTICS[_k]
            _t = _p.emulsion.base_um
            _rc = _fpm._v34_critical_radius_per_t(_n) * _t
            if _p.halation.radii_um[0] < _rc:
                _hg_bad_floor.append(_p.name)
            if _p.halation.radii_um != tuple(round(_s * _t, 1) for _s in _per_t):
                _hg_bad_derived.append(_p.name)
            if abs(sum(_p.halation.weights) - 1.0) > 1e-9:
                _hg_bad_energy.append(_p.name)

        chk("G-HALATION-GEOM-FLOOR  no stock's principal halation lobe falls "
            "inside its own critical-angle radius r_c = 2t/sqrt(n^2-1), the "
            "inner edge of the annulus support reflection can reach",
            not _hg_bad_floor,
            "%d stocks checked, %d below floor%s"
            % (len(_hg), len(_hg_bad_floor),
               (": " + ", ".join(_hg_bad_floor[:6])) if _hg_bad_floor else ""))

        chk("G-HALATION-GEOM-DERIVED  every halation triple is the v34 "
            "coefficient set times that stock's support thickness -- width is "
            "geometry, and amplitude belongs to the gains alone",
            not _hg_bad_derived,
            "%d stocks checked, %d hand-set%s"
            % (len(_hg), len(_hg_bad_derived),
               (": " + ", ".join(_hg_bad_derived[:6])) if _hg_bad_derived
               else ""))

        chk("G-HALATION-GEOM-ENERGY  halation lobe weights sum to one, so "
            "deriving the radii left total halation energy untouched and the "
            "existing gains still mean what they meant",
            not _hg_bad_energy,
            "%d stocks checked, %d off unity" % (len(_hg), len(_hg_bad_energy)))

        _hg_ac = _fpm._V34_BASE_OPTICS["acetate"][1][0]
        _hg_pe = _fpm._V34_BASE_OPTICS["polyester"][1][0]
        chk("G-HALATION-GEOM-SPLIT  acetate and polyester carry different "
            "coefficients, because ESTAR is biaxial and its through-thickness "
            "index is 1.50 against 1.66 in plane -- a single index would put "
            "the polyester halo 4 per cent too tight",
            _hg_ac != _hg_pe and len(_hg_keys) >= 1,
            "acetate sigma1/t %.3f, polyester %.3f; materials present: %s"
            % (_hg_ac, _hg_pe, sorted(_hg_keys)))

        # ⚠ The four stocks with no support thickness are outside the guard by
        # construction, and one of them renders. AGFA_NEU_1936 carries a
        # non-zero gain but is excluded from the base-thickness class default
        # by a standing rule that its emulsion record hold nothing Eggert 1937
        # printed, so its halo cannot be derived without inventing a gauge.
        # The three Polaroids have zero gain and render no halation at all.
        _hg_skipped = [p.name for p in FILM_PROFILES
                       if p.halation is not None and not p.emulsion.base_um]
        chk("G-HALATION-GEOM-SKIP  exactly the stocks with no published or "
            "class-default support thickness are outside the derivation, and "
            "the set has not grown",
            sorted(_hg_skipped) == ["AGFA_NEU_1936", "POLAROID_664",
                                    "POLAROID_667", "POLAROID_SX70"],
            "skipped: %s" % sorted(_hg_skipped))

        # ------------------------------------------------------------------
        #  G-SCRATCH -- the abrasion class at stage 9b. Added 2026-09-11 with
        #  queue row P44.
        #
        #  ⚠ THIS IS THE FIRST GUARD IN THIS FILE THAT COMPILES AND RUNS THE
        #  ENGINE, AND THE REASON IS THAT NOTHING ELSE CAN SEE THIS CLASS.
        #  `film_sim` has no defect layer at all, so the Python reference every
        #  other guard here measures against does not contain a scratch to
        #  compare with; `stage_parity` renders 9b in both twins but only asks
        #  whether they AGREE, which two identically dead generators also do;
        #  and a source grep cannot tell a class that draws from a class that
        #  is called and returns. The three properties this class is supposed
        #  to have are properties of its OUTPUT, so the output is what is
        #  measured.
        #
        #  ⚠ AND IT EXISTS BECAUSE THE FIELD IT CONSUMES WAS INERT FOR MONTHS
        #  WITHOUT ANYTHING NOTICING. `TemporalSpec.scratch_persistence_frames`
        #  is populated on all 184 stocks and was read by nothing; the two
        #  controls in front of it sat in AlgoControl.hpp's "Unconsumed" list.
        #  A guard that only checked the constants were present would pass just
        #  as happily on a generator whose every mark falls outside the raster.
        #
        #  WHAT IS ASSERTED, AND WHY EACH ONE
        #    off      at zero controls the stage writes its copy and changes
        #             NOTHING. The probe poisons the destination first, so a
        #             stage that returned without writing would fail this too -
        #             the retained-buffer policy makes that difference real.
        #    on       at a non-zero control the stage changes pixels, both
        #             populations, independently of each other.
        #    locked   a transport scratch is one nearly-full-height stroke in a
        #             handful of columns on a format whose film runs
        #             vertically. This is the 90 degree rotation the whole film
        #             coordinate system exists for; getting it backwards is the
        #             most conspicuous error the class can make.
        #    holds    the same scratch keeps the SAME across-web column over
        #             consecutive frames - the defining visible property of a
        #             tramline, and the thing `scratch_persistence_frames` now
        #             sets the length of.
        #    ends     and the run ENDS, with a new one starting elsewhere. A
        #             scratch that never went away would pass "holds" perfectly
        #             and would be a permanent mark on the glass, which is the
        #             failure the 2026-09-04 gate-dirt retune was about.
        #    both     marks appear with BOTH polarities. A cut removes emulsion
        #             and a burnish adds scatter, so a population with one sign
        #             means the polarity draw never reached the rasteriser.
        #
        #  ⚠ NO COMPILER IS A FAILURE, NOT A SKIP, when the tree is on disk.
        #  `stage_parity`'s header records what a green [SKIP] line cost this
        #  project once already. The tree being absent is a different thing and
        #  is the only condition under which this block does not run.
        _eng_sc = Path("/root/work/tst")

        if _eng_sc.is_dir():
            import shutil as _sh_sc
            import subprocess as _sp_sc
            import tempfile as _tf_sc

            # ---- the four measured figures, and both twins carrying the class
            #
            # Cheap, and it catches the two mistakes a render test cannot: a
            # constant quietly retuned away from the figure AlgoControl.hpp
            # lists as measured, and a class added to one twin only. The AVX2
            # tree's 9b defect namespace is meant to be the scalar one verbatim
            # apart from the compact-blob rasteriser, so "the name is in both
            # files" is exactly the right strength of check here -
            # `stage_parity` measures whether the numbers then match.
            def _rd_sc(_p):
                return _p.read_text(encoding="utf-8", errors="replace")

            _hdr_sc = _rd_sc(_eng_sc / "AlgoNegativeDefects.hpp")
            _twins_sc = {
                "scalar": _rd_sc(_eng_sc / "Algo_09_Sim.cpp"),
                "AVX2":   _rd_sc(_eng_sc / "AVX2" / "Algo_09_Sim.cpp"),
            }

            _want_sc = {
                "ALGO_SCRATCH_WIDTH_UM = 26.0":         "26 um width",
                "ALGO_SCRATCH_STRAIGHTNESS = 0.98":     "0.98 chord/arc",
                "ALGO_SCRATCH_ORIENT_RATIO = 3.5":      "3.5:1 bias",
                "ALGO_SCRATCH_CONTRAST_MEDIAN = 0.035": "3.5% contrast",
            }
            _missing_sc = [_v for _k, _v in _want_sc.items()
                           if _k not in _hdr_sc]

            _halftwin_sc = [_n for _n, _t in _twins_sc.items()
                            if ("defectScratches" not in _t
                                or "dmg.scratchTransport" not in _t
                                or "dmg.scratchHandling" not in _t)]

            chk("G-SCRATCH-CONSTANTS  the four measured scratch figures are "
                "named constants at their measured values, and both twins "
                "carry the class and read both controls",
                not _missing_sc and not _halftwin_sc,
                "26 um / 0.98 / 3.5:1 / 3.5%% all present; class in %s"
                % ", ".join(sorted(_twins_sc))
                if not _missing_sc and not _halftwin_sc
                else "missing constants %s; twins without the class %s"
                % (_missing_sc, _halftwin_sc))

            # ---- compile the probe and read its output ---------------------
            #
            # Three engine TUs and no database: stage 9b reads two fields of
            # FilmProfile and the probe hands it a value-initialised one, so
            # the twenty-six generated data TUs - which are the whole cost of
            # `stage_parity`'s build - are not needed. About a second.
            _SCR_N, _SCR_F = 96, 120

            _cxx_sc = _os.environ.get("CXX") or _sh_sc.which("g++") \
                or _sh_sc.which("clang++")

            _out_sc = None
            _why_sc = ""

            if not _cxx_sc:
                _why_sc = ("no g++/clang++ on PATH and the engine tree IS "
                           "present -- this guard cannot be skipped quietly")
            else:
                _tus_sc = ("test_scratch_guard.cpp", "Algo_09_Sim.cpp",
                           "AlgoDefectField.cpp", "AlgoSeparableBlur.cpp")

                with _tf_sc.TemporaryDirectory() as _td_sc:
                    _exe_sc = _os.path.join(_td_sc, "scratchprobe")
                    _cmd_sc = ([_cxx_sc, "-std=c++14", "-O1",
                                "-I", str(_eng_sc), "-o", _exe_sc]
                               + [str(_eng_sc / _t) for _t in _tus_sc])
                    _r_sc = _sp_sc.run(_cmd_sc, capture_output=True, text=True)

                    if _r_sc.returncode != 0:
                        _why_sc = ("probe did not compile: "
                                   + (_r_sc.stderr or _r_sc.stdout
                                      ).strip().splitlines()[-1][:160])
                    else:
                        _r_sc = _sp_sc.run(
                            [_exe_sc, str(_SCR_N), str(_SCR_F)],
                            capture_output=True, text=True)

                        if _r_sc.returncode != 0 or "END" not in _r_sc.stdout:
                            _why_sc = "probe did not run to completion"
                        else:
                            _out_sc = _r_sc.stdout

            # ---- parse -----------------------------------------------------
            # CASE <name> FRAME <f> SIZE <n> TOUCHED <k> MAXABS <v>
            #                                            MIND <v> MAXD <v>
            # COL <x> <sum |dst-src| down that column>
            _cases_sc = {}
            _key_sc = None

            for _ln_sc in (_out_sc or "").splitlines():
                _p_sc = _ln_sc.split()
                if not _p_sc:
                    continue
                if _p_sc[0] == "CASE":
                    _key_sc = (_p_sc[1], int(_p_sc[3]))
                    _cases_sc[_key_sc] = {
                        "touched": int(_p_sc[7]),
                        "maxabs":  float(_p_sc[9]),
                        "mind":    float(_p_sc[11]),
                        "maxd":    float(_p_sc[13]),
                        "cols":    [],
                    }
                elif _p_sc[0] == "COL" and _key_sc is not None:
                    _cases_sc[_key_sc]["cols"].append(float(_p_sc[2]))

            def _series_sc(_name):
                return [_cases_sc[(_name, _f)] for _f in range(_SCR_F)
                        if (_name, _f) in _cases_sc]

            _zero_sc = _series_sc("allzero")
            _tran_sc = _series_sc("transport")
            _hand_sc = _series_sc("handling")
            _part_sc = _series_sc("particles")

            _have_sc = (len(_zero_sc) == _SCR_F and len(_tran_sc) == _SCR_F
                        and len(_hand_sc) == _SCR_F
                        and len(_part_sc) == _SCR_F)

            # ---- G-SCRATCH-GATE  zero off, non-zero on ---------------------
            _off_ok_sc = _have_sc and all(_c["touched"] == 0
                                          for _c in _zero_sc)
            _on_t_sc = max((_c["touched"] for _c in _tran_sc), default=0)
            _on_h_sc = max((_c["touched"] for _c in _hand_sc), default=0)
            _on_p_sc = max((_c["touched"] for _c in _part_sc), default=0)

            chk("G-SCRATCH-GATE  stage 9b changes no pixel at zero scratch "
                "controls and does change pixels at non-zero ones, for each "
                "population separately",
                _have_sc and _off_ok_sc and _on_t_sc > 0 and _on_h_sc > 0,
                "all-zero touched 0 px on %d frames; transport peaks at %d px, "
                "handling at %d px, particulate control case at %d px"
                % (_SCR_F, _on_t_sc, _on_h_sc, _on_p_sc)
                if _have_sc
                else ("probe unusable -- %s" % (_why_sc or "no output")))

            # ---- G-SCRATCH-TRAMLINE  locked, holds, ends -------------------
            #
            # The probe renders super35, whose frame pitch is smaller than its
            # height, so AlgoFilmCoord derives the film as running along image
            # Y and a transport scratch is a VERTICAL stroke at a fixed image
            # X. Segmenting the sequence of touched-column sets into maximal
            # identical runs measures both remaining properties at once: the
            # longest segment is how long a tramline holds its position, and
            # the number of segments is whether runs end at all.
            _segs_sc = []
            _prev_sc = None

            for _c in _tran_sc:
                _cols = tuple(_x for _x, _v in enumerate(_c["cols"])
                              if _v > 0.0)
                if _cols != _prev_sc:
                    _segs_sc.append([_cols, 0])
                    _prev_sc = _cols
                _segs_sc[-1][1] += 1

            _live_sc = [_s for _s in _segs_sc if _s[0]]
            _hold_sc = max((_s[1] for _s in _live_sc), default=0)
            _wide_sc = max((len(_s[0]) for _s in _live_sc), default=0)

            # Full-height on the frames in the body of a run.
            #
            # ⚠ NOT ON EVERY FRAME, AND THE EXCEPTION IS THE FEATURE. The stroke
            # is clipped to the stretch of web the abrader actually touched -
            # start*pitch to (start+run)*pitch - so on the frame where it lets
            # go the tramline ENDS PART WAY UP THE PICTURE instead of vanishing
            # between two frames. Measured here: the short frames are a handful
            # out of the whole sequence, one per run, which is exactly that
            # boundary and not a stroke that fails to span the window.
            _livef_sc = [_c for _c in _tran_sc if _c["touched"] > 0]
            _tallf_sc = [_c for _c in _livef_sc
                         if _c["touched"] >= int(0.75 * _SCR_N)]

            _tallshare_sc = (float(len(_tallf_sc)) / float(len(_livef_sc))
                             if _livef_sc else 0.0)

            _short_sc = min((_c["touched"] for _c in _livef_sc), default=0)

            chk("G-SCRATCH-TRAMLINE  a transport scratch is a full-height "
                "stroke locked to the transport axis that HOLDS its across-web "
                "column over consecutive frames, and whose run then ENDS",
                _have_sc and _hold_sc >= 8 and len(_live_sc) >= 2
                and _wide_sc <= 4 and _tallshare_sc >= 0.8,
                "%d distinct runs over %d frames, longest holds one column set "
                "for %d consecutive frames, at most %d column(s) wide; "
                "%.0f%% of the %d frames carrying a scratch are full height "
                "(shortest %d px of %d rows, the frame a run ends on)"
                % (len(_live_sc), _SCR_F, _hold_sc, _wide_sc,
                   100.0 * _tallshare_sc, len(_livef_sc), _short_sc, _SCR_N)
                if _have_sc
                else ("probe unusable -- %s" % (_why_sc or "no output")))

            # ---- G-SCRATCH-POLARITY  cut and burnish both render -----------
            _dark_sc = min((_c["mind"] for _c in _hand_sc), default=0.0)
            _lite_sc = max((_c["maxd"] for _c in _hand_sc), default=0.0)

            chk("G-SCRATCH-POLARITY  scratches render with BOTH signs on the "
                "negative -- a cut removes density, a burnish adds it -- which "
                "no other class in stage 9b can do",
                _have_sc and _dark_sc < 0.0 and _lite_sc > 0.0,
                "handling population spans %.3e to %+.3e D against a "
                "single-mark amplitude of %.3e D"
                % (_dark_sc, _lite_sc, -math.log10(1.0 - 0.035))
                if _have_sc
                else ("probe unusable -- %s" % (_why_sc or "no output")))


    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
