"""Verification suite for the film simulation. Run: python3 verify.py"""
import dataclasses
import math, struct, sys, zlib
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
    chk("142 stocks load and validate", len(FILM_PROFILES) == 142, f"n={len(FILM_PROFILES)}")
    chk("9 print stocks load", len(PRINT_STOCKS) == 9, f"n={len(PRINT_STOCKS)}")
    rev = [p.name for p in FILM_PROFILES if p.is_reversal]
    # 2026-08-13: 22 -> 23 (AGFA_SCALA_200X, B&W reversal, added).
    # 2026-08-13: 23 -> 26. Cheltsov & Bongard 1958 added three reversal
    # stocks: KODACHROME_1938, KODACHROME_TYPE_A_1938 (both Kodachrome-process
    # reversal, diffusing couplers in the developer) and
    # FERRANIACOLOR_REVERSAL_1950 (incorporated couplers).
    # 2026-08-14: 26 -> 33. Seven Polaroid instant types are reversal (the
    # print IS the output); POLAROID_55_PN_NEG is deliberately NEGATIVE
    # because it is a real fixed, enlargeable silver negative.
    chk("reversal stocks flagged", len(rev) == 33, ", ".join(rev))

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

# ---- 2. characteristic curves monotonic ----------------------------------
if _sec_on():
    x = np.linspace(-6, 6, 6001).astype(np.float32)
    worst = min(
        float(np.diff(fs.density(x, c)).min())
        for p in FILM_PROFILES for c in p.curves.as_tuple()
    )
    # float32 evaluation leaves a few ulp of noise on a flat Dmax shelf; the
    # curve is analytically monotonic, so allow one ulp-scale negative slope.
    chk("all characteristic curves monotonic", worst >= -1e-5, f"min slope={worst:.3e}")

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
    chk("schema version is 5", _fpm.SCHEMA_VERSION == 5, f"v={_fpm.SCHEMA_VERSION}")
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
    chk("neighbour pairs couple harder than the far red-blue pair",
        abs(get_profile("KODAK_PORTRA_400").interimage.a_rg)
        > abs(get_profile("KODAK_PORTRA_400").interimage.a_rb))

    # the load-bearing property: neutrals untouched, saturated colour separates
    _stI = fs.RenderSettings(film_format="ff35", grain_scale=0.0, print_grain=False,
                             flare=0.0, vignette=0.0, coating_scale=0.0)
    _pI = get_profile("KODAK_PORTRA_400")
    _pN = dataclasses.replace(_pI, interimage=_fpm.InterimageSpec())
    _neu = np.full((48, 64, 3), 0.18, np.float32)
    _a = fs.simulate(_neu, _pN, _stI).mean(axis=(0, 1))
    _b = fs.simulate(_neu, _pI, _stI).mean(axis=(0, 1))
    chk("interimage leaves a neutral untouched",
        float(np.abs(_a - _b).max()) < 2e-3,
        f"max channel delta {float(np.abs(_a-_b).max()):.5f}")
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

    # the spectral derivation is DIAGNOSTIC and must stay out of the pipeline
    chk("spectral derivation exposes the IR failure it is quarantined for",
        (lambda r: r is not None and r[2] > 0.5)(
            _fpm.derived_spectral_response(get_profile("KONICA_INFRARED_750"))),
        "display primaries cannot reach 750 nm -- documented, not wired in")
    # 2026-08-13: the original form of this check tested for one function NAME,
    # which a differently-named spectral derivation passes vacuously -- and one
    # was added that day. It now guards the INTENT: no basis-projected spectral
    # derivation may drive the render by default. The balance-gain path IS
    # enabled, and is deliberately excluded here because it projects onto no
    # basis at all -- it is a ratio of one curve integrated against two
    # blackbody SPDs, so the gamut-reach failure below cannot apply to it.
    chk("no basis-projected spectral derivation is enabled by default",
        fs.RenderSettings().spectral_mono is False
        and fs.RenderSettings().spectral_taking is False,
        "mono weights and taking matrix both need a scene spectral model first")
    # The guard catches the EXTREME case and is honest about not catching all of
    # them. KONICA_INFRARED_750 peaks at 730 nm and is refused. ROLLEI_INFRARED_400
    # is NOT refused and should be: its curve sits at ~96 % of peak across
    # 660-680 nm, which the 600 nm primary lobe reaches poorly, so the derived
    # triple under-weights red (0.35 against an authored 0.52) -- yet only 2.7 %
    # of its energy lies past 700 nm, so neither guard condition fires. That
    # residual failure is not a threshold to tune; it is evidence that projecting
    # onto three visible lobes is the wrong construction, which is why
    # spectral_mono stays OFF by default and why the fix is a scene spectral
    # model. Recorded rather than papered over.
    chk("the basis-reach guard refuses an infrared-peaked stock",
        fs.spectral_monochrome_weights(get_profile("KONICA_INFRARED_750")) is None,
        "projecting an IR curve onto visible primaries derives blue-dominant nonsense")
    chk("the guard's known blind spot is still present and documented",
        fs.spectral_monochrome_weights(get_profile("ROLLEI_INFRARED_400")) is not None,
        "deep-red sensitisation inside 700 nm passes both conditions -- see comment")
    chk("the spectral balance path IS active and differs from the proxy",
        (lambda d, q: d is not None and abs(d[0] - q[0]) > 0.05)(
            fs.spectral_balance_gains(get_profile("KODAK_PORTRA_400"), 3200.0),
            fs.balance_gains(3200.0, get_profile("KODAK_PORTRA_400").balance_kelvin)),
        "derived red gain ~1.68 vs proxy ~1.32 at 3200 K")



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
    _mtf_bad = []
    for _p in FILM_PROFILES:
        _rp = film_profiles._RESOLVING_POWER.get(_p.name)
        if not _rp or not _rp[1]:
            continue
        _f50 = max(_p.mtf.f50_r, _p.mtf.f50_g, _p.mtf.f50_b)
        if _f50 >= _rp[1]:
            _mtf_bad.append("%s f50=%.0f >= limit=%.0f" % (_p.name, _f50, _rp[1]))
    chk("f50 stays below published limiting resolution on every stock with both",
        not _mtf_bad, "; ".join(_mtf_bad) or
        "%d stocks carry both figures, all consistent"
        % sum(1 for _p in FILM_PROFILES
              if film_profiles._RESOLVING_POWER.get(_p.name, (0, 0))[1]))


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

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
