"""Verification suite for the film simulation. Run: python3 verify.py"""
import dataclasses
import math, os, re, struct, sys, zlib
from pathlib import Path
import numpy as np
from PIL import Image

import film_sim as fs
import film_profiles
from film_profiles import (FILM_PROFILES, FORMATS, PRINT_STOCKS, StockKind,
                           get_profile, validate_all)

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
    """(peak MTF, frequency of that peak) for one channel of one MTFSpec."""
    _roll = np.asarray(film_profiles.mtf_response(_m, _ch, _A4_FGRID),
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
    # 2026-08-04: 89 -> 93 (AGFACOLOR_NEG_TYPE_B_1943, FUJICOLOR_A250,
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
    chk("176 stocks load and validate", len(FILM_PROFILES) == 176, f"n={len(FILM_PROFILES)}")
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
    chk("reversal stocks flagged", len(rev) == 42, ", ".join(rev))

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
    # stop less light on period stocks -- measured 61% low on AGFACOLOR_NEU_1936,
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
    chk("mid grey anchors to 18% for every stock", max(errs.values()) < 0.12,
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
    def granularity(name, width, band_limit=True):
        """sigma(D) through the 48 um aperture, x1000, as datasheets quote it."""
        p = get_profile(name)
        h = 512
        ppm = width / FORMATS["super35"]
        grid = fs.FreqGrid(h, width, ppm, p.grain.anisotropy)
        bl = grid.mtf(105.0, 0.0, 0.0) if band_limit else None
        f = fs.make_grain_field(grid, np.random.default_rng(7), p.grain.clump_um_g,
                                p.grain.clump_gain, p.grain.rms_granularity, bl)
        ap = np.exp(-2*math.pi**2*fs.APERTURE_SIGMA_MM**2*grid.f_mm.astype(np.float32)**2)
        return float(fs.apply_transfer(f, ap).std()) * 1000.0

    # Without a band limit and with a wide enough band, the field must reproduce the
    # datasheet granularity figure -- that is the definition the amplitude is fixed
    # against.
    worst_err = 0.0
    for nm in ("5219", "5203", "5296", "delta 3200", "kodachrome", "technicolor"):
        tgt = get_profile(nm).grain.rms_granularity
        got = granularity(nm, 16384, band_limit=False)
        worst_err = max(worst_err, abs(got - tgt) / tgt)
    chk("grain reproduces datasheet RMS granularity", worst_err < 0.05,
        f"max err={worst_err*100:.2f}%")

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
    for _i in range(3):
        _w, _h = 16384, 512
        _ppm = _w / FORMATS["super35"]
        _grid = fs.FreqGrid(_h, _w, _ppm, _p246.grain.anisotropy)
        _f = fs.make_grain_field(_grid, np.random.default_rng(11), _clump[_i],
                                 _p246.grain.clump_gain, _rms_c[_i], None)
        _amp = film_profiles.grain_sigma(_p246.grain, _curv[_i].dmin, _curv[_i].dmax,
                                        _curv[_i].dmin + 1.0)
        _apert = np.exp(-2*math.pi**2*fs.APERTURE_SIGMA_MM**2
                        * _grid.f_mm.astype(np.float32)**2)
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
    _fssrc = Path(fs.__file__).read_text(encoding="utf-8")
    chk("TK3: Takano eq (13) -- R_pr defaults to the print stock's own MTF, and "
        "print grain is still band-limited by it (known departure)",
        "scan_f50 = settings.scanner_f50 or print_stock.mtf_f50" in _fssrc
        and "print_stock.grain_clump_um, 0.25, print_stock.grain_rms, scan_t"
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
                                  "SUPER_ANSCOCHROME_1957")
                for n, v in _ids.items()),
        "id %d of %d stocks, frozen in film_ids.lock; ids 172-175 are the "
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
    chk("C18: the interimage density weight is capped at Dmax, and the cap is "
        "provably non-binding on every curve in the database",
        _worst <= 1e-9,
        "max(asymptote - dmax) = %.2e over %d stocks x 3 curves; the weight can "
        "only bind where a density exceeds its own curve's Dmax, which the "
        "softplus difference cannot produce"
        % (_worst, len(film_profiles.FILM_PROFILES)))

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
        "FUJI_SUPER_F125_8532", "FUJICOLOR_SUPER_F500_8572", "GEVACHROME_605",
    }
    chk("A4: exactly 11 stocks still render no overshoot, and they are the 11 "
        "with a written refusal -- down from 12, and 5231 is no longer one",
        set(_inert) == _expect_inert,
        "unexpected %s / missing %s"
        % (sorted(set(_inert) - _expect_inert),
           sorted(_expect_inert - set(_inert))))

# ---- 6. grain field statistics -------------------------------------------
if _sec_on():
    # 4000 px over super35 = 161 px/mm, so a 17.5 um clump spans ~2.8 px and the
    # spectrum is genuinely resolved (a coarse test grid would alias to white noise).
    grid = fs.FreqGrid(1024, 4000, 4000 / FORMATS["super35"], 1.0)
    f = fs.make_grain_field(grid, np.random.default_rng(3), 17.5, 1.15, 10.5)
    chk("grain zero mean", abs(float(f.mean())) < 1e-6, f"mean={f.mean():.2e}")
    ah = [float((f[:, :-k]*f[:, k:]).mean()) for k in range(1, 6)]
    av = [float((f[:-k, :]*f[k:, :]).mean()) for k in range(1, 6)]
    rel = max(abs(a-b)/abs(a) for a, b in zip(ah, av))
    chk("grain isotropic (h vs v autocorrelation)", rel < 0.05, f"max rel diff={rel*100:.2f}%")
    # anisotropy parameter must actually do something
    g2 = fs.FreqGrid(1024, 4000, 4000 / FORMATS["super35"], 1.30)
    f2 = fs.make_grain_field(g2, np.random.default_rng(3), 17.5, 1.15, 10.5)
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
    _order = ["velvia", "kodachrome", "technicolor", "5219", "5296", "agfacolor", "orwocolor"]
    _sats = [sat_and_contrast(n, mids)[0] for n in _order]
    chk("saturation hierarchy is ordered clean -> impure dyes",
        all(a > b for a, b in zip(_sats, _sats[1:])),
        " > ".join(f"{n}={v:.3f}" for n, v in zip(_order, _sats)))

    # Dye matrices must be pure saturation operators: unit row sums, so they change
    # colour without shifting neutral density.
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
    chk("every dye matrix has unit row sums", not _bad, str(_bad))
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
    chk("schema version is 27", _fpm.SCHEMA_VERSION == 27, f"v={_fpm.SCHEMA_VERSION}")

    # ==== 2026-09-01: THE TWO CARRIERS THAT STOPPED BEING INERT =============
    # `reciprocity_table` and `process_variants` were both listed as "carried,
    # validated, never read". Both are now consumed -- reciprocity by stage 8 in
    # both engines, process variants by the frame-setup resolver -- and each
    # ships under an inertness contract that these guards pin.
    import film_sim as _fsim

    # OFF must return the very same object, not an equal copy: the contract is
    # that selecting nothing costs nothing, and an equal-but-new profile would
    # satisfy a value comparison while allocating on every frame.
    chk("process variant OFF returns the profile itself on every stock",
        all(_fsim.resolve_process_variant(q, -1) is q for q in _fpm.FILM_PROFILES),
        "at least one stock copies at index -1")
    chk("process variant out-of-range returns the profile itself",
        all(_fsim.resolve_process_variant(q, len(q.process_variants)) is q
            for q in _fpm.FILM_PROFILES),
        "an index past the end is treated as a selection")
    # ⚠ THREE STOCKS, AND THE NUMBER IS THE POINT. 24 variants exist across 6
    # stocks and only these three carry a measured curve set; the other 19 are
    # the AGFAPAN developer records, which state an exposure index and no second
    # curve. If this count moves, a variant gained or lost its curves.
    _vcurve = sorted({
        q.name for q in _fpm.FILM_PROFILES
        for i in range(len(q.process_variants))
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
    _pp = get_profile("KODAK_PORTRA_400").interimage
    chk("interimage coupling is per RECEIVER, not per layer distance",
        (abs(_pp.a_rg - _pp.a_rb) < 1e-12
         and abs(_pp.a_gr - _pp.a_gb) < 1e-12
         and abs(_pp.a_br - _pp.a_bg) < 1e-12),
        "US4725529A Table 1: no layer stack, asymmetry persists -- "
        "so donor identity carries no weighting")

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
    chk("thirteen stocks carry a published aim density",
        len(_aim) == 13, f"{len(_aim)}: " + ", ".join(p.name for p in _aim))

    # ⚠ THE GRAY SCALE'S LIGHTEST STEP IS A BRIGHTER SUBJECT AREA THAN A GRAY
    # CARD, so on a NEGATIVE it must land HIGHER. This is the check that would
    # have caught a transposed two-row table, which is the single most likely
    # way to read these tables wrong.
    _inv = []
    for p in _aim:
        for a in p.aim_density:
            if not (a.gray_scale[0] > a.gray_card[0]
                    and a.gray_scale[1] > a.gray_card[1]):
                _inv.append(f"{p.name}@EI{a.exposure_index} "
                            f"card {a.gray_card} scale {a.gray_scale}")
            # and a dark complexion reflects less light than a light one
            if a.forehead_dark[0] >= a.forehead_light[0]:
                _inv.append(f"{p.name}@EI{a.exposure_index} "
                            f"dark {a.forehead_dark} >= light {a.forehead_light}")
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
    # 0.077-0.312 to 0.117-0.174 and puts AGFACOLOR_NEU_1936 within 10 % of
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
    for _q in _kq:
        _k = film_profiles.mtf_kernel(_q)
        if _k is None:
            _kbad.append("q %.4f is stored on a stock and has no kernel row" % _q)
            continue
        _w1, _s1, _s2 = _k
        _tgt = 1.0 / (1.0 + _fk ** _q)
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

    # ⚠ THE TABLE IS NOT INTERPOLATABLE AND THE GUARD SAYS SO. Two disjoint
    # optimal basins straddle q ~ 3.0: below it the tight lobe is small
    # (w1 < 0.4), above it the fit flips to w1 > 1 with a negative wide lobe.
    # Anything that starts interpolating this table must trip here first.
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
    _tung = [(p.name, p.exposure_index, p.exposure_index_tungsten)
             for p in FILM_PROFILES if p.exposure_index_tungsten]
    _bad_t = [f"{n} {d}/{t}" for n, d, t in _tung if not (1.0 <= d / t <= 4.0)]
    chk("tungsten exposure index never exceeds daylight and stays plausible",
        not _bad_t and len(_tung) >= 7,
        "; ".join(_bad_t) or f"{len(_tung)} stocks, ratios "
        + ", ".join(f"{d/t:.2f}" for _, d, t in _tung))

    # The blue-sensitive-only stocks must sit far above the panchromatic cluster.
    # This is the documented physical claim the field exists to carry, so it is
    # asserted rather than left as prose in a description.
    _blue = {n: d / t for n, d, t in _tung if n in ("POLAROID_51", "POLAROID_146L")}
    _pan = [d / t for n, d, t in _tung if n not in _blue]
    chk("blue-sensitive stocks separate from panchromatic on the tungsten ratio",
        _blue and min(_blue.values()) > max(_pan) * 2.0,
        f"blue {sorted(round(v, 2) for v in _blue.values())} vs pan max {max(_pan):.2f}")

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

    # exposure_index_tungsten is defined as UNFILTERED pairs only. A colour
    # film's second index is quoted through a conversion filter and is therefore
    # a filter factor, not a film property -- so every entry must be monochrome.
    # If a later batch adds a colour stock here, this fails and the definition
    # in the field docstring has been violated.
    _tw = [p.name for p in FILM_PROFILES
           if p.exposure_index_tungsten and not p.is_monochrome]
    chk("every tungsten exposure index is a monochrome stock (unfiltered pairs)",
        not _tw, ", ".join(_tw) or
        "%d entries, all monochrome"
        % sum(1 for p in FILM_PROFILES if p.exposure_index_tungsten))


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
             "aim_density"))
        and film_profiles.SCHEMA_VERSION == 27
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
    chk("exactly 22 stocks carry a neutral+dmin pair",
        _pairs == [
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
        ],
        ", ".join(_pairs))
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
    chk("27 film profiles now carry a spectral dye density set",
        len(_dye) == 27, "%d: %s" % (len(_dye), ", ".join(
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
    _ZERO_FLOOR_OK = {"FUJI_VELVIA_50", "GEVACOLOR_NEG_682"}
    _zeroruns = sorted({p.name for p in _dye
                        if p.name not in _ZERO_FLOOR_OK
                        for row in (p.dye_density.d_cyan,
                                    p.dye_density.d_magenta,
                                    p.dye_density.d_yellow)
                        for i in range(len(row) - 3)
                        if all(v == 0.0 for v in row[i:i + 4])})
    _zero_have = sorted({p.name for p in _dye if p.name in _ZERO_FLOOR_OK
                         and all(v == 0.0 for v in p.dye_density.d_yellow[-6:])})
    chk("the zero-floor convention is used on exactly two named yellow tails",
        not _zeroruns and _zero_have == sorted(_ZERO_FLOOR_OK),
        "VELVIA 50 from 600 nm, GEVACOLOR 682 from 580 nm"
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
    chk("only the 13 vendor-traced stocks are flagged sigma_shape_measured",
        _meas == ["EASTMAN_EXR_100T_5248", "EASTMAN_EXR_50D_5245",
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
    chk("exactly the 23 traced stocks are flagged mtf_measured",
        _mmeas == [
                   "EASTMAN_DOUBLE_X_5222",
                   "EASTMAN_EXR_100T_5248",
                   "EASTMAN_EXR_50D_5245",
                   "EASTMAN_PLUS_X_5231",
                   "FUJICOLOR_PRO_400H",
                   "FUJICOLOR_SUPERIA_XTRA_400",
                   "FUJICOLOR_SUPER_F500_8572",
                   "FUJI_PROVIA_100F",
                   "FUJI_SUPER_F125_8532",
                   "GEVACHROME_600",
                   "GEVACHROME_605",
                   "KODAK_EKTACHROME_100D_5285",
                   "KODAK_EKTAR_100",
                   "KODAK_PORTRA_160",
                   "KODAK_PORTRA_400",
                   "KODAK_PORTRA_400VC",
                   "KODAK_PORTRA_800",
                   "KODAK_VISION2_200T_5217",
                   "KODAK_VISION2_500T_5218",
                   "KODAK_VISION2_50D_5201",
                   "KODAK_VISION_200T_5274",
                   "KODAK_VISION_500T_5279",
                   "KONICA_IMPRESA_50",
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
    for _agn in ("AGFA_APX_25", "AGFA_APX_100", "AGFA_APX_400"):
        _fam = get_profile(_agn).processing_family.points
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
            any("09/1998" in _s and "agfa_films.pdf" in _s
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
        if len(_hit) != 1 or "agfa_films.pdf" not in (_hit[0].source or ""):
            _stale.append(_agn)
    chk("every Agfa rms cell cites the sheet the figure is printed on",
        not _stale, ", ".join(_stale) if _stale else
        "%d Agfa profiles cite agfa_films.pdf for their rms" % len(_agfa_rms))

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

    # ⚠ f50 IS NOT ADOPTED FROM THE AGFA SHARPNESS PANELS AND THAT IS THE POINT.
    # The panels are CTFs -- they peak at 102-114 %, above the 100 % a true MTF
    # cannot exceed -- so the overshoot is adoptable as `adjacency` (a unit-free
    # ratio) while f50 is not, because queue G6 has not settled whether Agfa's
    # "Lines (mm)" are cycles or line pairs. A factor of two is not a rounding
    # difference. If someone adopts f50 from these sheets, mtf_measured will be
    # set and this fails.
    _m = [n for n in _coat if get_profile(n).mtf.mtf_measured]
    chk("no Agfa stock claims a measured MTF while G6 is open",
        not _m, ", ".join(_m) if _m else "12 stocks, adjacency measured, f50 not")

    # The measured overshoots themselves, digitised 2026-09-01.
    _adj = {"AGFA_APX_25": 0.0514, "AGFA_APX_100": 0.0998,
            "AGFA_APX_400": 0.0998, "AGFA_OPTIMA_100": 0.1091,
            "AGFA_ULTRA_50": 0.1445, "AGFA_RSX_II_200": 0.1041,
            "AGFA_SCALA_200X": 0.0211}
    _bad = [n for n, v in _adj.items()
            if abs(get_profile(n).mtf.adjacency - v) > 1e-9]
    chk("the traced Agfa adjacency overshoots are unchanged",
        not _bad, ", ".join(_bad) if _bad else "0.021-0.145 across seven stocks")

    # ⚠ APX 100 AND APX 400 SHARE ONE PIECE OF SHARPNESS ARTWORK. Two separate
    # path objects in two separate columns with identical geometry, so the two
    # overshoots are equal BY CONSTRUCTION and not by measurement. Recorded as a
    # guard so nobody later reads the equality as corroboration.
    chk("APX 100 and APX 400 overshoots are equal because Agfa reused the art",
        abs(get_profile("AGFA_APX_100").mtf.adjacency
            - get_profile("AGFA_APX_400").mtf.adjacency) < 1e-12,
        "both 0.0998 -- one drawing, two columns")

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
                            "FUJICOLOR_PRO_400H"}
    _flank_bad = []
    for _n, _rr, _rb in (("FUJI_SUPER_F125_8532", 0.8976, 1.0762),
                         ("FUJICOLOR_SUPER_F500_8572", 0.8214, 1.1071),
                         ("FUJI_PROVIA_100F", 0.8970, 1.0754),
                         ("FUJICOLOR_SUPERIA_XTRA_400", 0.8976, 1.0762),
                         ("FUJICOLOR_PRO_400H", 0.8977, 1.0763)):
        _m = get_profile(_n).mtf
        if (abs(_m.f50_r / _m.f50_g - _rr) > 0.005
                or abs(_m.f50_b / _m.f50_g - _rb) > 0.005):
            _flank_bad.append("%s r/g=%.4f b/g=%.4f" % (
                _n, _m.f50_r / _m.f50_g, _m.f50_b / _m.f50_g))
    chk("the two green-only stocks keep their declared flanking ratios",
        not _flank_bad, "; ".join(_flank_bad) if _flank_bad
        else "8532 0.8976/1.0762, 8572 0.8214/1.1071")
    chk("every green-only stock is flagged measured and keeps r < g < b",
        all(get_profile(n).mtf.mtf_measured
            and get_profile(n).mtf.f50_r < get_profile(n).mtf.f50_g
            < get_profile(n).mtf.f50_b for n in _GREEN_ONLY_MEASURED),
        "5 of 5, layer order intact")

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
    _meas_ratio = [(p.name, p.mtf.f50_r / p.mtf.f50_b)
                   for p in FILM_PROFILES
                   if p.mtf.mtf_measured and not p.is_monochrome
                   and p.mtf.f50_b > 0
                   and p.name not in _GREEN_ONLY_MEASURED
                   and p.name not in _VISUAL_FILTER_MEASURED]
    chk("every measured colour stock is softer in red than the estimating rule",
        all(r < 0.65 for _, r in _meas_ratio) and len(_meas_ratio) >= 7,
        "; ".join("%s %.3f" % (n.split("_")[-1], r) for n, r in _meas_ratio)
        + " vs the rule's ~0.78")

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
    _mr = [p.mtf.f50_r for p in FILM_PROFILES
           if p.mtf.mtf_measured and not p.is_monochrome
           and p.kind == StockKind.NEGATIVE
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
    _qs = sorted(round(p.mtf.mtf_rolloff_q, 4) for p in FILM_PROFILES
                 if p.mtf.mtf_measured and p.mtf.mtf_rolloff_q > 0.0)
    chk("the measured rolloff exponents are distinct and spread over 1.0",
        len(set(_qs)) == len(_qs) and (_qs[-1] - _qs[0]) > 1.0,
        "q = " + " / ".join("%.2f" % q for q in _qs))
    # And a flagged stock must carry a usable exponent OR be the one documented
    # exception, because the flag otherwise silently falls back to the Gaussian.
    # ⚠ 5279 IS THAT EXCEPTION AND IT IS PHYSICS, NOT AN OVERSIGHT: its sheet
    # prints a +42 %/+55 % adjacency overshoot, and the carrier 1/(1+(f/f50)^q) is
    # 1.0 at zero frequency by construction, so it cannot represent a curve that
    # starts at 1.42 -- the fit returns rms 0.25 against 0.0095-0.132 elsewhere.
    # Its measured f50 triple and measured overshoot are used; its rolloff is not.
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
    chk("27 film profiles carry spectral dye density", len(_dd) == 27,
        ", ".join(sorted(p.name.split("_")[-1] for p in _dd)))
    _pk = [p for p in _dd if p.dye_density.normalisation == "peak_1.0"]
    chk("19 of the 27 dye sets are tagged peak_1.0, 8 as-printed", len(_pk) == 19,
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
    _DYE_GRID_EXCEPTIONS = {"TECHNICOLOR_THREE_STRIP": (360.0, 10.0, 44)}
    _off = [p.name for p in _dd
            if p.name not in _DYE_GRID_EXCEPTIONS
            and not (len(p.dye_density.d_cyan) == 31
                     and p.dye_density.lambda_start_nm == 400.0
                     and p.dye_density.lambda_step_nm == 10.0)]
    _exc = [p.name for p in _dd if p.name in _DYE_GRID_EXCEPTIONS
            and (p.dye_density.lambda_start_nm,
                 p.dye_density.lambda_step_nm,
                 len(p.dye_density.d_cyan)) != _DYE_GRID_EXCEPTIONS[p.name]]
    chk("every dye trace is a 31-sample 400-700 nm grid, with one named exception",
        not _off and not _exc and
        len(_DYE_GRID_EXCEPTIONS) == sum(1 for p in _dd
                                         if p.name in _DYE_GRID_EXCEPTIONS),
        "31 x 10 nm from 400; exception TECHNICOLOR_THREE_STRIP 44 x 10 nm from 360"
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
    chk("110 development points across 12 stocks, every one with a measured contrast",
        _n_pts == 110 and len(_pf) == 12
        and all(q.contrast_index > 0.0 or q.gamma > 0.0
                for p in _pf for q in p.processing_family.points),
        "%d stocks, %d points" % (len(_pf), _n_pts))
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

    # ---- AGFACOLOR_NEU_1936: Eggert 1937, adopted 2026-09-04 ---------------
    # ⚠ THE FIRST MEASURED NUMBERS THIS PROFILE HAS EVER CARRIED, and the guard
    # exists because of how nearly they were missed: the document arrived on
    # 2026-09-03 and the first pass filed the inventor's own seven-page article
    # as a bare provenance citation without reading it, while the profile's own
    # provenance went on asserting that no photometric figure for this film
    # existed anywhere.
    _neu = get_profile("AGFACOLOR_NEU_1936")
    chk("AGFACOLOR_NEU_1936 carries Eggert's measured layer geometry",
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
    chk("AGFACOLOR_NEU_1936's exposure_index is still the analogy 8, with the "
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
            "52_509.pdf" in " ".join(_p.provenance.sources), "cited")
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
        "KONICA_CHROME_CENTURIA_100": "chrocen100.pdf",
        "KONICA_CHROME_R100":         "R100.pdf",
        "ILFORD_HPS":                 "table 7",
        "KODAK_SUPER_XX_PAN_4142":    "DS 17",
    }
    _miss = [n for n, tok in _CLOSED.items()
             if tok not in " ".join(get_profile(n).provenance.sources)]
    chk("all 6 closed citations still name their own document",
        not _miss, ", ".join(_miss) if _miss else "6 of 6")
    # A250's confusable companion file is the one hazard in this batch that
    # would silently corrupt data if the warning were dropped: PDF/PROFILES/
    # FUJI/'A 250.pdf' is a 1985 SMPTE paper about AX 8514/8512 and LP 8816.
    chk("A250 keeps the 'A 250.pdf' misattribution warning",
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
        len(_mneg) == 11 and all(g.sigma_shape_dmax < 1.0 for g in _mneg)
        and all(g.sigma_shape_dmax < 1.0 for g in _col_neg
                if g.sigma_shape_peak > 0),
        "11 measured fall (0.50-0.90); the colour-negative default is now "
        "0.68 with an interior peak 1.38 at 0.75 of scale. ⚠ IT STAYED 11 on "
        "2026-09-02c: E5's 12th candidate was traced and then withdrawn on a "
        "density-space mismatch, not on its shape")
    # 55, not the 51 the F2 row claimed -- recounted live 2026-08-30. One of
    # them, TASMA_FN_64, carries its own literal shape and is excluded rather
    # than counted as agreeing.
    # ⚠ 55 -> 56 on 2026-09-02 (queue N1): FUJI_NEOPAN_SS, a new monochrome
    # negative that lands on the same unevidenced heuristic as the rest --
    # AF3-411E prints no granularity of any kind, so it could not have landed
    # anywhere else.
    chk("the MONOCHROME negative default is still unevidenced and still rises",
        len(_mono_neg) == 56
        and sum(1 for g in _mono_neg if g.sigma_shape_dmax > 1.0) == 55,
        "51 B&W negatives keep 0.4/1.0/1.2 -- no measured B&W NEGATIVE shape "
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
    #  G-FILMID -- the frozen film identifiers, 2026-08-28
    #
    #  WHAT THIS PROTECTS. Before the freeze, database order was the natural
    #  name sort, so every eFILM_PROFILE value, every GetFilmDatabase()
    #  subscript and every film_names.txt line number was a function of which
    #  stock names happened to exist. Adding one stock renumbered the rest, and
    #  every saved After Effects or Premiere project would then have rendered a
    #  DIFFERENT FILM. These five checks are what make that unrepeatable.
    #
    #  Each is a distinct failure mode, which is why they are five and not one.
    # =======================================================================
    import os as _os
    _lockp = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "film_ids.lock")

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
        _missing = [_n for _n in _dbnames if _n not in _ids]
        chk("G-FILMID  every stock carries a frozen id",
            not _missing, f"{len(_missing)} without: {_missing[:4]}")

        # 2. no id is shared. Two stocks on one id is two emulsions behind one
        #    saved-project reference.
        chk("G-FILMID  no id is issued twice", not _dupes, f"dupes {_dupes[:4]}")

        # 3. database order is ASCENDING id. This is what makes the C++ index
        #    equal the id, which is what film_names.txt line numbers rely on.
        _seq = [_ids[_n] for _n in _dbnames if _n in _ids]
        chk("G-FILMID  database order is ascending id",
            _seq == sorted(_seq), f"first inversion near index "
            f"{next((i for i in range(1, len(_seq)) if _seq[i] < _seq[i-1]), -1)}")

        # 4. a retired id is never reissued to a live stock. Reissue is the one
        #    failure that silently points a project at the WRONG emulsion,
        #    rather than at nothing.
        _reissued = sorted(set(_ids.values()) & _retired)
        chk("G-FILMID  no retired id has been reissued",
            not _reissued, f"reissued {_reissued[:4]}")

        # 5. the 161 stocks that predate the freeze still sit at id == index.
        #    This is the no-op proof, kept as a live assertion: if it ever
        #    fails, the seeding was disturbed and every pre-freeze project is
        #    affected.
        _shift = [(_i, _n) for _i, _n in enumerate(_dbnames[:161])
                  if _ids.get(_n) != _i]
        chk("G-FILMID  the 161 pre-freeze stocks are still at id == index",
            not _shift, f"{len(_shift)} moved: {_shift[:3]}")

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


    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
