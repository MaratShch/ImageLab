#!/usr/bin/env python3
"""Measure what each MEASUREMENT in the database is worth, in visible pixels.

⚠⚠ THIS IS NOT A STAGE-OFF ABLATION AND THE DIFFERENCE IS THE WHOLE POINT.
Turning a stage off answers "how much does this effect matter", which nobody is
asking: the effect is in the model because it matters. What the realism score
needs is the other question -- **"how much does HAVING THE MEASUREMENT change
the picture, against what this stock would render with no measurement at all?"**
So every axis below is ablated by SUBSTITUTING ITS OWN NO-EVIDENCE FALLBACK:
the traced curve becomes the generic class curve, the measured sigma(D) shape
becomes the legacy square-root law, the measured MTF becomes the class median,
and so on. The delta is what the document bought.

That number is the weight. A stock that lacks a carrier is scored as having
lost the delta that carrier is worth on the stocks that do have it -- stated as
an assumption in the generated report rather than hidden in a constant.

⚠ IT IS SLOW AND IT IS CACHED FOR THAT REASON. The influence table is written
to `realism_influence.json` with a MODEL HASH over every source file that can
change a rendered pixel. `gen_realism_score.py` reads the cache on every build
and refuses it when the hash no longer matches, so a model change cannot leave
stale weights in place silently.

Usage:
    python3 realism_ablation.py                 # measure and write the cache
    python3 realism_ablation.py --check         # fail if the cache is stale
    python3 realism_ablation.py --stocks 8      # smaller sample, for a smoke run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import film_profiles as fp          # noqa: E402
import film_sim as fs               # noqa: E402

CACHE = HERE / "realism_influence.json"

#: Files whose contents can move a rendered pixel. The hash of these is what
#: makes a cached influence table trustworthy or stale.
MODEL_FILES = ("film_sim.py", "film_profiles.py", "realism_ablation.py")

#: Scene size. Large enough that px_per_mm is in the range a real scan sits in
#: (Super-35 is 24.9 mm wide, so 1536 px is ~62 px/mm -- a 2K scan), small
#: enough that the
#: whole sweep runs in minutes. ⚠ GRAIN AND MTF ARE SCALE-DEPENDENT: measuring
#: their influence at 64 px/mm would report a different number, which is why
#: the sampling rate is recorded in the cache beside the results.
SCENE_W, SCENE_H = 1536, 864

SEED = 20260920


# ---------------------------------------------------------------------------
#  Scenes
# ---------------------------------------------------------------------------

def _chart() -> np.ndarray:
    """Ramp, colour patches, blown discs and bar targets: the instrument scene."""
    h, w = SCENE_H, SCENE_W
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = xx / (w - 1)
    img = np.repeat(ramp[:, :, None], 3, axis=2) * 1.6

    for i, col in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1),
                             (1, 1, 0), (0, 1, 1), (1, 0, 1)]):
        y0 = 20 + i * 14
        img[y0:y0 + 12, 40:w - 40] = (np.array(col, np.float32)
                                      * ramp[y0:y0 + 12, 40:w - 40, None])

    # Blown discs: halation and the shoulder both live here.
    for cx, cy, r in [(180, 300, 22), (384, 312, 12), (590, 290, 34)]:
        img[((xx - cx) ** 2 + (yy - cy) ** 2) < r * r] = 6.0

    # Bar targets across four periods: MTF, adjacency and the grain band.
    for k, period in enumerate([3, 6, 12, 24]):
        y0 = 150 + k * 18
        bars = ((xx // (period / 2)) % 2).astype(np.float32)
        img[y0:y0 + 14] = bars[y0:y0 + 14, :, None] * 0.6 + 0.05

    img[360:400, 40:120] = 0.18     # mid grey
    return np.clip(img, 0.0, None).astype(np.float32)


def _pictorial() -> np.ndarray:
    """A scene with the statistics of a photograph rather than of a chart.

    ⚠ THE CHART ALONE WOULD OVERWEIGHT SHARPNESS AND UNDERWEIGHT TONE. It is
    mostly edges and saturated primaries; a real frame is mostly smooth
    midtones in a narrow gamut, which is where a tone curve earns its keep and
    where grain is actually visible. Both scenes are measured and the influence
    is their mean.
    """
    rng = np.random.default_rng(SEED)
    h, w = SCENE_H, SCENE_W
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Sky gradient, foliage band, skin-toned subject, a small specular.
    img = np.zeros((h, w, 3), np.float32)
    sky = 1.0 - yy / h
    img[..., 0] = 0.35 * sky + 0.05
    img[..., 1] = 0.55 * sky + 0.06
    img[..., 2] = 0.95 * sky + 0.08

    foliage = yy > h * 0.55
    img[foliage] = np.array([0.055, 0.12, 0.035], np.float32)

    skin = ((xx - w * 0.35) ** 2 / (w * 0.13) ** 2
            + (yy - h * 0.55) ** 2 / (h * 0.30) ** 2) < 1.0
    img[skin] = np.array([0.42, 0.28, 0.21], np.float32)

    shadow = ((xx - w * 0.75) ** 2 + (yy - h * 0.70) ** 2) < (h * 0.22) ** 2
    img[shadow] = np.array([0.012, 0.011, 0.014], np.float32)

    img[((xx - w * 0.30) ** 2 + (yy - h * 0.40) ** 2) < 7 ** 2] = 9.0

    # Low-amplitude 1/f texture, so smooth areas are not literally flat.
    noise = rng.standard_normal((h, w, 1)).astype(np.float32)
    k = np.fft.rfftfreq(w)[None, :] + np.fft.fftfreq(h)[:, None] * 0 + 1e-3
    tex = np.fft.irfft(np.fft.rfft(noise[..., 0], axis=1) / (k ** 0.9), n=w, axis=1)
    tex = (tex / (np.abs(tex).max() + 1e-9)).astype(np.float32)
    img *= (1.0 + 0.06 * tex[..., None])

    return np.clip(img, 0.0, None).astype(np.float32)


SCENES = {"chart": _chart, "pictorial": _pictorial}


# ---------------------------------------------------------------------------
#  Difference metric
# ---------------------------------------------------------------------------

_M_RGB_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _lab(linear: np.ndarray) -> np.ndarray:
    """CIE L*a*b* from linear sRGB, D65. Clipped at the display ceiling first.

    ⚠ CLIPPED ON PURPOSE. The renderer returns display-referred light and a
    viewer's screen cannot show more than 1.0, so a difference above white is a
    difference nobody sees. Measuring it would let a stage that only moves
    blown highlights outrank one that moves skin.
    """
    x = np.clip(linear.astype(np.float64), 0.0, 1.0)
    xyz = x @ _M_RGB_XYZ.T / _WHITE_D65
    e, k = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > e, np.cbrt(xyz), (k * xyz + 16.0) / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0,
                     500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """Mean CIE76 delta-E over the frame.

    CIE76 rather than CIEDE2000 deliberately: this is a WEIGHT, not a
    perceptual verdict, and 76 is a plain Euclidean distance that nobody has to
    re-derive to check a number. The ranking of the axes is the same either way.
    """
    d = _lab(a) - _lab(b)
    return float(np.sqrt((d * d).sum(-1)).mean())


# ---------------------------------------------------------------------------
#  The axes: each is a measurement, and each knows its own no-evidence fallback
# ---------------------------------------------------------------------------

def _kind_median_f50(profile) -> tuple[float, float, float]:
    """Median f50 of the stocks of the same kind that have NO measured MTF."""
    peers = [p for p in fp.FILM_PROFILES
             if p.is_monochrome == profile.is_monochrome
             and not p.mtf.mtf_measured]
    if not peers:
        peers = [p for p in fp.FILM_PROFILES if not p.mtf.mtf_measured]
    return tuple(float(np.median([getattr(p.mtf, f) for p in peers]))
                 for f in ("f50_r", "f50_g", "f50_b"))


def _generic_curves(profile):
    """The curve a stock with no traced sheet would render through."""
    g = fp.get_profile("GENERIC_BW" if profile.is_monochrome
                       else "GENERIC_COLOR")
    return g.curves


def _has_curve(p):
    return any(s.param.startswith("curves") and s.tier == 1
               for s in p.param_sources)


def _sub_curves(p):
    return replace(p, curves=_generic_curves(p))


def _sub_sigma_shape(p):
    """Drop the traced sigma(D) anchors; grain_sigma falls back to the sqrt law."""
    return replace(p, grain=replace(p.grain, sigma_shape_measured=False,
                                    sigma_shape_toe=0.0, sigma_shape_mid=0.0,
                                    sigma_shape_dmax=0.0,
                                    sigma_shape_peak=0.0))


def _sub_mtf(p):
    r, g, b = _kind_median_f50(p)
    return replace(p, mtf=replace(p.mtf, f50_r=r, f50_g=g, f50_b=b,
                                  mtf_measured=False))


def _sub_adjacency(p):
    return replace(p, mtf=replace(p.mtf, adjacency=0.0))


def _sub_spectral(p):
    """Clear the traced curves; the render falls back to spectral_weights."""
    return replace(p, spectral=replace(
        p.spectral, log_s_r=(), log_s_g=(), log_s_b=(), log_s_pan=(),
        log_s_c=()))


def _sub_dye_density(p):
    """Remove what the traced dye spectra actually contribute to a render.

    ⚠⚠ CLEARING THE SPECTRA IS NOT ENOUGH AND FOR A YEAR IT WAS ALL THIS DID.
    The renderer never reads `dye_density`; it reads `dye_matrix`. While the
    measured matrices were inert that made this axis a true 0.000 -- a traced
    carrier on 30 stocks worth nothing because nothing consumed it. After the
    2026-09-20 adoption it would STILL have read 0.000, for a different and
    much worse reason: the substitution was removing an object the render path
    does not touch, so the axis would have reported "the measurement is
    worthless" about a matrix that had just started moving every pixel on 26
    stocks.

    ⚠ SO THE SUBSTITUTION IS THE FALLBACK THAT WOULD ACTUALLY BE USED: the
    `_dye(k)` scalar stand-in each stock carried before adoption, kept by
    `film_profiles._DYE_MATRIX_PRE_ADOPTION`. That makes the axis measure the
    real question -- what the spectra are worth OVER the aesthetic scalar they
    replaced -- and it measures it in whichever direction the answer falls.
    """
    q = replace(p, dye_density=replace(
        p.dye_density, d_cyan=(), d_magenta=(), d_yellow=(), d_neutral=()))
    pre = getattr(fp, "_DYE_MATRIX_PRE_ADOPTION", {}).get(p.name)
    if pre is not None:
        q = replace(q, dye_matrix=pre)
    return q


def _sub_dye_matrix(p):
    return replace(p, dye_matrix=fp.IDENTITY3)


_IIE_FIELDS = ("a_rg", "a_rb", "a_gr", "a_gb", "a_br", "a_bg")


def _sub_interimage(p):
    return replace(p, interimage=replace(
        p.interimage, **{f: 0.0 for f in _IIE_FIELDS}))


def _sub_halation(p):
    return replace(p, halation=replace(p.halation, gain_r=0.0, gain_g=0.0,
                                       gain_b=0.0))


def _sub_callier(p):
    return replace(p, callier_q=1.0)


def _sub_rms(p):
    peers = [q.grain.rms_granularity for q in fp.FILM_PROFILES
             if q.is_monochrome == p.is_monochrome
             and q.grain.rms_granularity > 0.0]
    med = float(np.median(peers))
    g = p.grain
    k = med / g.rms_granularity if g.rms_granularity > 0 else 1.0
    return replace(p, grain=replace(g, rms_granularity=med,
                                    rms_r=g.rms_r * k, rms_g=g.rms_g * k,
                                    rms_b=g.rms_b * k))


def _sub_reciprocity(p):
    return replace(p, reciprocity_table=replace(
        p.reciprocity_table, times_s=(), stops_correction=()))


#: (axis key, what it is, carrier test, substitution, extra render settings)
#:
#: ⚠ THE CARRIER TEST DECIDES WHERE THE AXIS CAN BE MEASURED AT ALL. Removing a
#: measurement a stock never had changes nothing and would drag the median to
#: zero, so an axis is only measured on the stocks that carry it.
AXES = (
    ("tone_curve", "traced characteristic curve",
     _has_curve, _sub_curves, {}),
    ("grain_amplitude", "published rms granularity",
     lambda p: p.grain.rms_granularity > 0.0, _sub_rms, {}),
    ("grain_shape", "measured sigma(D) shape",
     lambda p: bool(p.grain.sigma_shape_measured), _sub_sigma_shape, {}),
    ("sharpness_mtf", "measured MTF",
     lambda p: bool(p.mtf.mtf_measured), _sub_mtf, {}),
    ("edge_effects", "measured adjacency overshoot",
     lambda p: p.mtf.adjacency > 0.0, _sub_adjacency, {}),
    # ⚠ AND THE SCENE IS TUNGSTEN. At 5500 K against a daylight-balanced
    # stock the derivation and the stored triple agree by construction, so the
    # axis would report zero for the same reason a thermometer in a thermostat
    # reports no change.
    ("spectral_sensitivity", "traced spectral sensitivity",
     lambda p: bool(p.spectral.has_data), _sub_spectral,
     # ⚠ THE CURVES ARE READ ONLY WHEN A DERIVATION ASKS FOR THEM. At the
     # defaults the balance triple and the monochrome collapse are the stored
     # scalars, so clearing the curves would move nothing and the axis would
     # report zero -- true of a render nobody makes.
     {"spectral_balance": True, "spectral_mono": True,
      "scene_kelvin": 3200.0, "wb_strength": 1.0}),
    ("dye_density", "traced spectral dye density",
     lambda p: bool(p.dye_density.d_cyan or p.dye_density.d_neutral),
     _sub_dye_density, {}),
    ("dye_matrix", "measured reader/status matrix",
     lambda p: any(s.param == "dye_matrix" for s in p.param_sources),
     _sub_dye_matrix, {}),
    ("interimage", "interimage coefficients",
     lambda p: any(abs(getattr(p.interimage, f, 0.0)) > 0.0
                   for f in _IIE_FIELDS),
     _sub_interimage, {}),
    ("halation", "halation gain",
     lambda p: max(p.halation.gain_r, p.halation.gain_g,
                   p.halation.gain_b) > 0.0, _sub_halation, {}),
    ("callier", "measured Callier Q",
     lambda p: abs(p.callier_q - 1.0) > 1e-12, _sub_callier, {}),
    # ⚠ RECIPROCITY IS INERT UNTIL A TIME IS STATED, so its scene states one.
    # Measuring it at the default would report zero and be true only of a
    # render nobody makes.
    ("reciprocity", "published reciprocity table",
     lambda p: bool(p.reciprocity_table.times_s), _sub_reciprocity,
     {"exposure_time_s": 8.0}),
)


# ---------------------------------------------------------------------------
#  The sweep
# ---------------------------------------------------------------------------

def sample_stocks(n: int) -> list:
    """A spread over kind, era and speed -- not the first n alphabetically.

    ⚠ THE SAMPLE IS STRATIFIED AND NAMED IN THE CACHE. An influence measured on
    twelve colour negatives would be an influence for colour negatives, and the
    report has to be able to say which stocks produced its weights.
    """
    P = list(fp.FILM_PROFILES)
    buckets = {
        "colour_neg": [p for p in P if not p.is_monochrome and not p.is_reversal],
        "reversal": [p for p in P if p.is_reversal and not p.is_monochrome],
        "mono": [p for p in P if p.is_monochrome],
    }
    out = []
    for key, group in buckets.items():
        # Best-evidenced first: an axis can only be measured where the carrier
        # exists, so a sample of bare stocks would measure almost nothing.
        group = sorted(group, key=lambda p: -sum(
            (bool(p.grain.sigma_shape_measured), bool(p.mtf.mtf_measured),
             bool(p.spectral.has_data),
             bool(p.dye_density.d_cyan or p.dye_density.d_neutral),
             bool(p.reciprocity_table.times_s))))
        take = max(2, round(n * len(group) / len(P)))
        out.extend(group[:take])
    return out[:n]


def model_hash() -> str:
    h = hashlib.sha256()
    for name in MODEL_FILES:
        h.update((HERE / name).read_bytes())
    return h.hexdigest()[:16]


def _baseline_settings() -> list[dict]:
    """Every distinct render setting an axis needs, the plain default first."""
    out = [{}]
    for _k, _w, _c, _s, extra in AXES:
        if extra and extra not in out:
            out.append(dict(extra))
    return out


def measure(stocks, verbose=True) -> dict:
    scenes = {k: f() for k, f in SCENES.items()}
    per_axis: dict[str, list] = {a[0]: [] for a in AXES}

    for i, p in enumerate(stocks, 1):
        base = {}
        for skey, img in scenes.items():
            # ⚠ THE BASELINES ARE DERIVED FROM THE AXES, NOT LISTED HERE.
            # An axis that needs a non-default setting to be live needs its
            # BASELINE rendered at that same setting; a hand-kept list of
            # settings silently compares the wrong pair the day an axis is
            # added.
            for extra in _baseline_settings():
                tag = (skey, tuple(sorted(extra.items())))
                base[tag] = fs.simulate(
                    img, p, fs.RenderSettings(seed=SEED, **extra))
        for key, _what, carries, sub, extra in AXES:
            if not carries(p):
                continue
            q = sub(p)
            tag_extra = tuple(sorted(extra.items()))
            vals = []
            for skey, img in scenes.items():
                out = fs.simulate(img, q, fs.RenderSettings(seed=SEED, **extra))
                vals.append(delta_e(base[(skey, tag_extra)], out))
            per_axis[key].append((p.name, float(np.mean(vals))))
        if verbose:
            print(f"  [{i:2d}/{len(stocks)}] {p.name}")

    axes = {}
    for key, what, _c, _s, extra in AXES:
        rows = per_axis[key]
        vals = sorted(v for _n, v in rows)
        axes[key] = {
            "what": what,
            "stocks_measured": len(rows),
            "delta_e_median": float(np.median(vals)) if vals else 0.0,
            "delta_e_max": float(vals[-1]) if vals else 0.0,
            "worst_stock": max(rows, key=lambda r: r[1])[0] if rows else None,
            "render_settings": dict(extra),
        }
    return {
        "schema_version": fp.SCHEMA_VERSION,
        "model_hash": model_hash(),
        "scene_px": [SCENE_W, SCENE_H],
        "px_per_mm": round(SCENE_W / 24.892, 2),
        "scenes": sorted(SCENES),
        "metric": "mean CIE76 delta-E, display-referred, clipped at 1.0",
        "sample": [p.name for p in stocks],
        "axes": axes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stocks", type=int, default=18)
    ap.add_argument("--check", action="store_true",
                    help="write nothing; fail if the cache is missing or stale")
    ns = ap.parse_args()

    if ns.check:
        if not CACHE.is_file():
            print("[FAIL] realism_influence.json is missing -- run "
                  "`python3 realism_ablation.py`")
            return 1
        got = json.loads(CACHE.read_text(encoding="utf-8"))
        if got.get("model_hash") != model_hash():
            print("[FAIL] realism_influence.json was measured against a "
                  "different render model -- re-run `python3 "
                  "realism_ablation.py`")
            return 1
        print(f"[OK] realism_influence.json matches the live render model "
              f"({len(got['axes'])} axes, sample of {len(got['sample'])})")
        return 0

    stocks = sample_stocks(ns.stocks)
    print(f"[i] measuring {len(AXES)} axes over {len(stocks)} stocks "
          f"x {len(SCENES)} scenes at {SCENE_W}x{SCENE_H}")
    data = measure(stocks)
    CACHE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {CACHE.name}")
    for k, v in sorted(data["axes"].items(),
                       key=lambda kv: -kv[1]["delta_e_median"]):
        print(f"    {k:22s} dE {v['delta_e_median']:7.3f}  "
              f"(n={v['stocks_measured']}, worst {v['delta_e_max']:.3f} "
              f"on {v['worst_stock']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
