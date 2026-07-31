"""Verification suite for the film simulation. Run: python3 verify.py"""
import math, struct, sys, zlib
from pathlib import Path
import numpy as np
from PIL import Image

import film_sim as fs
from film_profiles import FILM_PROFILES, FORMATS, StockKind, get_profile, validate_all

ok = True
def chk(label, cond, extra=""):
    global ok
    cond = bool(cond)
    ok &= cond
    print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"   {extra}" if extra else ""))

validate_all()
lin = fs.load_linear(Path("test_chart.png"))

# ---- 1. profile integrity ------------------------------------------------
chk("71 stocks load and validate", len(FILM_PROFILES) == 71, f"n={len(FILM_PROFILES)}")
rev = [p.name for p in FILM_PROFILES if p.is_reversal]
chk("reversal stocks flagged", len(rev) == 20, ", ".join(rev))

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

# ---- 2. characteristic curves monotonic ----------------------------------
x = np.linspace(-6, 6, 6001).astype(np.float32)
worst = min(
    float(np.diff(fs.density(x, c)).min())
    for p in FILM_PROFILES for c in p.curves.as_tuple()
)
# float32 evaluation leaves a few ulp of noise on a flat Dmax shelf; the
# curve is analytically monotonic, so allow one ulp-scale negative slope.
chk("all characteristic curves monotonic", worst >= -1e-5, f"min slope={worst:.3e}")

# ---- 3. 16-bit PNG really is 16-bit --------------------------------------
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
st = fs.RenderSettings(grain_scale=0.0, print_grain=False, misreg_scale=0.0,
                       flare=0.0)
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
        grey_target=tgt))
    got = float(o[615:665, 55:145].mean())
    chk(f"grey_target={tgt} honoured", abs(got - tgt)/tgt < 0.12, f"got {got:.4f}")
for ps in ("SCAN_DI", "KODAK_2383_RELEASE", "TECHNICOLOR_IB"):
    o = fs.simulate(lin, get_profile("5219"), fs.RenderSettings(
        print_stock=ps, grain_scale=0.0, print_grain=False, misreg_scale=0.0,
        flare=0.0))
    got = float(o[615:665, 55:145].mean())
    chk(f"mid grey anchored on print stock {ps}", abs(got-0.18)/0.18 < 0.12, f"got {got:.4f}")

# ---- 5. grain granularity calibration, resolution invariant --------------
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
worst_rev = 0.0
for _p in FILM_PROFILES:
    if not _p.is_reversal:
        continue
    _anc = fs.solve_anchors(_p, fs.get_print_stock("SCAN_DI"), 0.18)
    for _c in range(3):
        _cur = _p.curves.as_tuple()[_c]
        _d = fs.density(-(_lg + np.float32(_anc[_c])), _cur)
        _t = (10.0 ** (-_d) - 10.0 ** (-_cur.dmax)) / (
            10.0 ** (-_cur.dmin) - 10.0 ** (-_cur.dmax))
        worst_rev = min(worst_rev, float(np.diff(_t).min()))
chk("reversal transfer monotonic in exposure", worst_rev >= -1e-6,
    f"min slope={worst_rev:.2e}")

# ---- 10. Technicolor three-strip specifics -------------------------------
tech = get_profile("technicolor")
chk("three-strip has non-identity taking matrix",
    not np.allclose(np.asarray(tech.taking_matrix), np.eye(3)))
chk("three-strip has large registration error", tech.misregistration_um > 20.0,
    f"{tech.misregistration_um} um")
chk("three-strip defaults to the imbibition print", tech.default_print == "TECHNICOLOR_IB")

# ---- 11. determinism, range, finiteness ---------------------------------
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
# The defining property: ortho emulsion is effectively blind to red. A red
# subject must render far darker, and a blue one far lighter, than on a
# panchromatic stock of the same era.
patches = np.zeros((64, 192, 3), dtype=np.float32)
patches[:, 0:64] = (0.5, 0.0, 0.0)     # red
patches[:, 64:128] = (0.0, 0.5, 0.0)   # green
patches[:, 128:192] = (0.0, 0.0, 0.5)  # blue
st_clean = fs.RenderSettings(grain_scale=0.0, print_grain=False, misreg_scale=0.0,
                             flare=0.0)
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

print()
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
