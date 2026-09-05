"""Queue D1 + D2a -- what repeat scans of ONE untouched frame can measure.

WHY THIS FILE EXISTS BEFORE THE FILES DO
----------------------------------------
Queue D2 asked for a step-wedge scan. The owner declined the purchase on
2026-09-05b, and the free replacement turns out to measure the RIGHT quantity
rather than a cheaper approximation of the wrong one: scan ONE untouched film
frame about sixteen times and everything that differs between the scans IS the
scanner, because the film did not move.

  sigma_repeat(level)   the scanner, measured on the actual emulsion at the
                        actual densities -- not on a grainless target at
                        densities the film never reaches
  sigma_total(level)    the single frame's own local statistics: emulsion+scanner
  sigma_emulsion        sqrt(max(sigma_total^2 - sigma_repeat^2, 0))

That split is the whole point of the row, and it is what blocks the STRENGTH
half of C18 and C19.

This module is written and self-tested BEFORE the scans arrive so that the
answer to "I sent the files" is a run and not a week of coding. Run
`python scan_repeat.py --synthetic` to exercise every estimator against data
whose truth is known by construction.

⚠ THE SCANNER IS JPEG-ONLY, AND THAT IS RECORDED BEFORE THE SCAN RATHER THAN
DISCOVERED AFTER IT (owner, 2026-09-05b). Three consequences, each handled:

  1. JPEG QUANTISATION IS DETERMINISTIC. Two sensor readouts whose DCT
     coefficients land in the same bin encode to BIT-IDENTICAL bytes. A repeat
     test therefore sees only the noise that survives quantisation, and it reads
     LOW in exactly the smooth areas where the floor matters most. So
     `identical_fraction` is measured DIRECTLY and reported beside every sigma:
     it turns the codec floor from an unknown into a number, and a high value is
     the signal that sigma_repeat is a LOWER BOUND rather than a measurement.
  2. THE QUANTISATION TABLES ARE IN THE FILE. `jpeg_facts` reads them, plus the
     chroma subsampling, so the codec's own floor is bounded analytically at
     zero cost -- no scan required, just the files that are being sent anyway.
  3. ⚠ THE PHOTON-TRANSFER ROUTE IS DEAD AND IS NOT ATTEMPTED. Plotting variance
     against signal level would have given the transfer curve from these same
     scans with no target at all -- for a LINEAR sensor, variance is proportional
     to signal. JPEG's quantisation is level- and block-dependent, so the curve
     would be the encoder's and not the sensor's. Queue row D2b owns the
     transfer curve and asks for a known-density target instead.

⚠ AND THE PIPELINE NOISE IS THE RIGHT TARGET ANYWAY. The 546 archived frames are
JPEGs. What corrupts the sigma(D) estimator on them is the whole chain including
the codec, not the bare sensor, so a floor that includes the codec is the floor
that applies.

WHAT ELSE FALLS OUT OF THE SAME FILES, FREE
-------------------------------------------
  * D1's OWN QUESTION -- does the UF15 re-expose per frame? The per-frame mean
    level across the repeats answers it directly, and it is the question the 50
    archived base strips raised (235.8 to 252.9 on ONE piece of base) and could
    not settle.
  * ANISOTROPY, from the 90-degree rotation. Grain correlation that SWAPS AXES
    with the film is the film; correlation that stays put in image coordinates is
    the scanner or the codec. ⚠ JPEG's 8x8 grid is fixed in image coordinates, so
    this test will find it -- and that is a finding, not a contaminant.
  * THE BLOCK GRID ITSELF. `block_periodicity` measures the 8-pixel component of
    the difference image's autocorrelation, which says how much of what looks
    like grain is actually the codec.

⚠ WHAT THIS CANNOT DO, STATED PLAINLY. It cannot give an absolute density scale.
sigma is measured in CODE VALUES, and converting to density needs the transfer
curve that D2b is still open for. Every number here is therefore a noise figure
in the scan's own units, and the split between emulsion and scanner is a RATIO,
which is scale-free and is the part that matters for C18/C19.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
#: Frames whose per-pixel spread is measured. Sixteen is not arbitrary: the
#: standard deviation of a sample standard deviation is about sigma/sqrt(2(N-1)),
#: so N=16 knows sigma to about 18 %, N=4 to 41 %, N=64 to 9 %. Sixteen buys a
#: useful number in about five minutes of the owner's time; sixty-four would buy
#: half the error for four times the tedium, and the tedium is the binding
#: constraint on a protocol somebody has to sit through.
WANT_REPEATS = 16
MIN_REPEATS = 6

#: Level bins for the sigma-against-level curves, in 8-bit code values.
LEVEL_EDGES = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128,
                        144, 160, 176, 192, 208, 224, 240, 256], dtype=float)

#: Half-width of the local window used for the single-frame sigma_total. 3 gives
#: a 7x7 window: 49 samples, enough for a stable local variance, and small
#: enough that real picture structure inside it is rare once the flatness test
#: below has rejected the textured windows.
LOCAL_HALF = 3

#: A window counts as FLAT -- i.e. its variance is noise rather than picture --
#: when the range of the HEAVILY SMOOTHED image across it is under this many
#: code values.
#: ⚠ WITHOUT THIS TEST sigma_total IS THE PICTURE, NOT THE GRAIN. An edge inside
#: the window contributes far more variance than any emulsion, and a frame of an
#: ordinary scene is mostly edges. This is the single assumption in the module
#: that a bad choice would silently corrupt, which is why it is a named constant
#: and why `--flatness` exposes it.
#: ⚠ AND IT IS MEASURED ON A 41x41 SMOOTHED COPY, NOT ON 3x3 BOX MEANS. Measured
#: on the first version of this module, which used 3x3: with grain at 3 code
#: values the 3x3 means themselves swing by more than the threshold, so the test
#: rejected EVERY window in the frame and the split silently produced no bins at
#: all. The smoothing box has to be much larger than the grain and much smaller
#: than the picture, and 41 px is comfortably both.
FLATNESS_MAX = 2.0
FLATNESS_SMOOTH = 41

#: Lag range over which grain correlation length is fitted, in pixels.
CORR_LAGS = 12


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_gray(path: Path) -> tuple[np.ndarray, dict]:
    """One scan as float32 luminance, plus what the container says about itself.

    ⚠ THE GREEN CHANNEL IS USED WHEN THE FILE IS COLOUR, not a luma mix. JPEG
    subsamples chroma and leaves luma alone, and green carries most of luma, so
    green is the channel the codec has damaged least. Mixing R and G and B would
    average a full-resolution channel with two half-resolution ones and report a
    noise figure for a signal the file does not contain.
    """
    from PIL import Image
    im = Image.open(path)
    facts = jpeg_facts(im, path)
    a = np.asarray(im)
    if a.ndim == 3:
        g = a[:, :, 1].astype(np.float32)
    else:
        g = a.astype(np.float32)
    return g, facts


def jpeg_facts(im, path: Path) -> dict:
    """What the file's own headers say -- the codec floor, for free.

    ⚠ THIS COSTS THE OWNER NOTHING AND BOUNDS THE THING THAT WORRIES ME MOST.
    A JPEG's quantisation table IS its noise floor: a DCT coefficient quantised
    with step q carries an error uniform on +/- q/2, i.e. an rms of q/sqrt(12).
    The DC term's step (`q_dc`) is the one that matters for a flat patch, because
    a flat 8x8 block is almost pure DC. Reading it turns "JPEG adds some
    unknown error" into a number, before a single pixel is compared.
    """
    out: dict = {"path": str(path), "format": getattr(im, "format", "?"),
                 "mode": im.mode, "size": tuple(im.size)}
    q = getattr(im, "quantization", None)
    if q:
        t0 = list(q.values())[0]
        arr = np.asarray(t0, dtype=float).ravel()
        out["q_dc"] = float(arr[0])
        out["q_mean"] = float(arr.mean())
        out["q_tables"] = len(q)
        # rms of a uniform quantiser of step q, applied to the DC term
        out["dc_quant_rms"] = float(arr[0] / np.sqrt(12.0))
    lay = getattr(im, "layer", None)
    if lay:
        out["subsampling"] = [tuple(int(v) for v in c) for c in lay]
        # ⚠ (1,2,2) on the luma component means 4:2:0 -- chroma at half
        # resolution in both axes. On a MONOCHROME original that is harmless;
        # on a colour one it means the R-G difference the 2026-08-31 review
        # looked for was thrown away by the encoder before anyone measured it.
        out["chroma_subsampled"] = any(c[1] > 1 or c[2] > 1 for c in lay)
    return out


def align_int(ref: np.ndarray, other: np.ndarray, search: int = 4):
    """Integer-pixel shift of `other` against `ref`, by brute-force SAD.

    ⚠ THE PROTOCOL SAYS DO NOT TOUCH THE FILM, and if it is obeyed this returns
    (0, 0) every time. It exists because a shift of even one pixel turns an
    edge into a difference and inflates sigma_repeat by far more than the noise
    it is trying to measure -- so a non-zero result here is not a correction to
    apply quietly, it is a WARNING that the run measured registration and not
    the scanner. The caller reports it.
    """
    h, w = ref.shape
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    base = ref[y0:y1, x0:x1]
    best, bdy, bdx = None, 0, 0
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            cut = other[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            if cut.shape != base.shape:
                continue
            sad = float(np.abs(cut - base).mean())
            if best is None or sad < best:
                best, bdy, bdx = sad, dy, dx
    return bdy, bdx, best


# ---------------------------------------------------------------------------
# the estimators
# ---------------------------------------------------------------------------
def exposure_jitter(frames: list[np.ndarray]) -> dict:
    """D1's OWN QUESTION: does the scanner re-expose between frames?

    The 50 archived base strips read 235.8 to 252.9 on ONE physical piece of
    base, which said the UF15 re-exposes but could not prove it -- those frames
    hold different pictures, and a different picture is a different auto-exposure
    decision for an honest reason. Repeats of ONE frame remove that confound
    entirely: the scanner sees the same thing every time, so any movement of the
    mean is the machine changing its mind.
    """
    means = [float(f.mean()) for f in frames]
    m = float(np.mean(means))
    return dict(per_frame_mean=means, mean=m,
                spread=float(max(means) - min(means)),
                std=float(np.std(means)),
                # a 1 % swing of the mean level is about 2.5 code values at
                # mid-grey and is far above anything read noise produces
                re_exposing=bool(max(means) - min(means) > 0.01 * max(m, 1.0)))


def repeat_sigma(frames: list[np.ndarray]) -> dict:
    """sigma_repeat against level, and the codec floor beside it.

    Per pixel: the standard deviation across the repeats, and whether every
    repeat gave the identical byte. Binned by mean level.

    ⚠ `identical_fraction` IS REPORTED BESIDE EVERY BIN AND IS NOT A CURIOSITY.
    Where it is high the scanner's noise did not survive quantisation, so
    sigma_repeat there is a LOWER BOUND and the emulsion/scanner split computed
    from it over-credits the emulsion. A bin at 0.95 identical is not a
    measurement of a quiet scanner; it is a measurement of a coarse quantiser.
    """
    stack = np.stack(frames, axis=0)
    mean = stack.mean(axis=0)
    sd = stack.std(axis=0, ddof=1)
    same = (stack.max(axis=0) == stack.min(axis=0))

    rows = []
    for lo, hi in zip(LEVEL_EDGES[:-1], LEVEL_EDGES[1:]):
        m = (mean >= lo) & (mean < hi)
        n = int(m.sum())
        if n < 500:
            continue
        rows.append(dict(level=float(0.5 * (lo + hi)), n=n,
                         sigma=float(sd[m].mean()),
                         sigma_p90=float(np.percentile(sd[m], 90)),
                         identical_fraction=float(same[m].mean())))
    return dict(bins=rows,
                sigma_overall=float(sd.mean()),
                identical_overall=float(same.mean()),
                mean_map=mean, sd_map=sd)


def local_sigma(frame: np.ndarray, half: int = LOCAL_HALF,
                flatness: float = FLATNESS_MAX) -> dict:
    """sigma_total against level, from ONE frame's flat windows.

    Emulsion grain plus everything the scanner added, which is what a single
    archived frame offers and what the sigma(D) estimator has always been
    reading without knowing what fraction was which.
    """
    k = 2 * half + 1
    # box mean and box mean-of-squares by summed-area table
    f = frame.astype(np.float64)
    s1 = _boxsum(f, k) / (k * k)
    s2 = _boxsum(f * f, k) / (k * k)
    var = np.maximum(s2 - s1 * s1, 0.0)

    # flatness, on a copy smoothed far past the grain scale so that only
    # PICTURE structure survives to be tested against the threshold
    flat = flat_mask(frame, half, flatness)

    rows = []
    for a, b in zip(LEVEL_EDGES[:-1], LEVEL_EDGES[1:]):
        m = flat & (s1 >= a) & (s1 < b)
        n = int(m.sum())
        if n < 500:
            continue
        rows.append(dict(level=float(0.5 * (a + b)), n=n,
                         sigma=float(np.sqrt(var[m].mean()))))
    return dict(bins=rows, flat_fraction=float(flat.mean()))


def _boxsum(a: np.ndarray, k: int) -> np.ndarray:
    """Sliding k x k sum, edge-replicated, via a summed-area table."""
    p = np.pad(a, k // 2 + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    h, w = a.shape
    y0 = np.arange(h)
    x0 = np.arange(w)
    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    return (c[Y0 + k, X0 + k] - c[Y0, X0 + k]
            - c[Y0 + k, X0] + c[Y0, X0])


def _boxmax(a: np.ndarray, k: int) -> np.ndarray:
    """Sliding k x k maximum, edge-replicated. Separable, so 2k comparisons."""
    h = k // 2
    p = np.pad(a, h, mode="edge")
    o = np.maximum.reduce([p[i:i + p.shape[0] - k + 1, :] for i in range(k)])
    return np.maximum.reduce([o[:, i:i + o.shape[1] - k + 1] for i in range(k)])


def split_sigma(total: dict, repeat: dict) -> list[dict]:
    """sigma_emulsion = sqrt(max(sigma_total^2 - sigma_repeat^2, 0)), per level.

    ⚠ THE CLAMP AT ZERO IS NOT COSMETIC AND ITS HITS ARE COUNTED. A negative
    difference means the measured scanner noise exceeded the measured total,
    which is impossible physically and therefore says one of the two estimates is
    wrong -- most likely sigma_total, because a flat window that was not flat
    inflates it, or sigma_repeat, because a registration shift inflated that. A
    run with clamped bins is a run to distrust, so the caller reports how many.
    """
    rp = {r["level"]: r for r in repeat["bins"]}
    out = []
    for t in total["bins"]:
        r = rp.get(t["level"])
        if r is None:
            continue
        v = t["sigma"] ** 2 - r["sigma"] ** 2
        out.append(dict(level=t["level"],
                        sigma_total=t["sigma"], sigma_scanner=r["sigma"],
                        sigma_emulsion=float(np.sqrt(max(v, 0.0))),
                        clamped=bool(v < 0.0),
                        scanner_share=float(min(1.0, r["sigma"] ** 2
                                                / max(t["sigma"] ** 2, 1e-9))),
                        identical_fraction=r["identical_fraction"]))
    return out


def autocorr_1d(resid: np.ndarray, axis: int, lags: int = CORR_LAGS,
                mask: np.ndarray | None = None):
    """Normalised autocorrelation of a residual field along one axis.

    ⚠ `mask` IS NOT OPTIONAL IN PRACTICE AND OMITTING IT INVERTED THE ANISOTROPY
    VERDICT ON SYNTHETIC DATA WHOSE TRUTH WAS KNOWN. A picture's EDGES survive
    any high-pass, and an edge is a long correlated run along its own direction.
    On the self-test frame two rectangular plateaus were enough to push the x
    correlation length past the 12-lag ceiling while the true grain was
    stretched along y, so the measured ratio came out 0.58 where the truth is
    about 2. Restricting the sum to pairs where BOTH pixels lie in a flat region
    is what makes this a measurement of grain rather than of composition.
    """
    r = resid - (resid[mask].mean() if mask is not None else resid.mean())
    if mask is not None:
        r = np.where(mask, r, 0.0)
    w = mask.astype(np.float64) if mask is not None else np.ones_like(r)
    n0 = float(w.sum())
    if n0 <= 0:
        return [1.0] + [0.0] * lags
    v = float((r * r).sum() / n0)
    if v <= 0:
        return [1.0] + [0.0] * lags
    out = [1.0]
    for L in range(1, lags + 1):
        if axis == 0:
            num = float((r[:-L, :] * r[L:, :]).sum())
            den = float((w[:-L, :] * w[L:, :]).sum())
        else:
            num = float((r[:, :-L] * r[:, L:]).sum())
            den = float((w[:, :-L] * w[:, L:]).sum())
        out.append((num / den) / v if den > 0 else 0.0)
    return out


#: Half-width of the box whose mean is subtracted to leave the grain. 20 gives
#: a 41x41 box.
#:
#: ⚠ IT MUST BE MUCH LARGER THAN THE GRAIN AND THIS WAS MEASURED, NOT GUESSED.
#: The first version of this module used a 9x9 box and the anisotropy verdict
#: came out INVERTED on synthetic data whose truth was known: grain stretched
#: 2:1 along y reported a ratio of 0.50 instead of ~2.0. The reason is that a
#: fixed high-pass attenuates the two axes DIFFERENTLY -- the more-correlated
#: axis carries more low-frequency energy, so it loses more, and with a cutoff
#: near the correlation length the loss is large enough to flip the ratio. A box
#: far wider than the grain removes the picture and leaves the grain's own
#: spectrum intact.
GRAIN_HIGHPASS_HALF = 20


def grain_residual(frame: np.ndarray,
                   half: int = GRAIN_HIGHPASS_HALF) -> np.ndarray:
    """Frame minus its local mean -- the high-frequency part, grain plus noise."""
    k = 2 * half + 1
    return frame.astype(np.float64) - _boxsum(frame.astype(np.float64), k) / (k * k)


def flat_mask(frame: np.ndarray, half: int = LOCAL_HALF,
              flatness: float = FLATNESS_MAX) -> np.ndarray:
    """Where the picture is smooth enough that the residual is grain, not edges."""
    f = frame.astype(np.float64)
    k = 2 * half + 1
    ms = _boxsum(f, FLATNESS_SMOOTH) / float(FLATNESS_SMOOTH ** 2)
    return (_boxmax(ms, k) + _boxmax(-ms, k)) <= flatness


def block_periodicity(resid: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """How much of the residual sits on the JPEG 8-pixel grid.

    ⚠ THIS IS THE MEASUREMENT THAT SEPARATES CODEC FROM GRAIN, and neither the
    sigma split nor the rotation test can make it. Blocking artefacts are
    periodic at exactly 8 px and locked to image coordinates; emulsion grain has
    no period at all. The ratio of the autocorrelation at lag 8 to the mean of
    its neighbours at 7 and 9 is 1.0 for grain and rises above it for a residual
    the encoder shaped.
    """
    out = {}
    for name, axis in (("y", 0), ("x", 1)):
        ac = autocorr_1d(resid, axis, lags=12, mask=mask)
        nb = 0.5 * (ac[7] + ac[9])
        # ⚠ THE RATIO IS A RATIO OF TWO SMALL NUMBERS AND IS MEANINGLESS WHEN
        # BOTH ARE NOISE. Where the grain's own correlation has already decayed
        # by lag 7 -- which it has, on any emulsion whose correlation length is
        # a few pixels -- ac(7), ac(8) and ac(9) are all near zero and their
        # ratio wanders freely. Measured on the self-test: quality 92 with no
        # chroma subsampling gives 1.52 on one axis and 0.83 on the other, from
        # the same encoder and the same grid, which is the ratio being noise and
        # not the encoder being anisotropic. So the ratio is reported only when
        # lag 8 clears a floor, and otherwise the honest answer is "not
        # detectable", which is different from "not present".
        detectable = abs(ac[8]) >= 0.02
        out[name] = dict(lag8=ac[8], neighbours=nb, detectable=bool(detectable),
                         ratio=(float(ac[8] / nb)
                                if detectable and abs(nb) > 1e-9
                                else float("nan")),
                         curve=[round(v, 5) for v in ac])
    return out


def corr_length(resid: np.ndarray, axis: int,
                mask: np.ndarray | None = None) -> float:
    """Lags until the autocorrelation first falls below 1/e. Sub-lag interpolated."""
    ac = autocorr_1d(resid, axis, lags=CORR_LAGS, mask=mask)
    thr = 1.0 / np.e
    for i in range(1, len(ac)):
        if ac[i] < thr:
            a, b = ac[i - 1], ac[i]
            return float((i - 1) + (a - thr) / max(a - b, 1e-9))
    return float(CORR_LAGS)


def anisotropy(frame0: np.ndarray, frame90: np.ndarray | None) -> dict:
    """Is the directional grain correlation the FILM or the SCANNER?

    ⚠ ONE FRAME CANNOT ANSWER THIS AND THE DATABASE HAS BEEN ASSUMING AN ANSWER.
    `GrainSpec.anisotropy` is a film property in this schema, and every stored
    value is an estimate. Measure the correlation length along x and along y on
    one scan and the ratio is real but unattributed: it could be the emulsion's
    coating direction or the sensor's readout direction, and those are different
    facts with the same signature.

    Rotating the FILM 90 degrees separates them, because only one of the two
    turns with it:

      ratio(0 deg) * ratio(90 deg) ~= 1     the anisotropy followed the film
      ratio(0 deg) ~= ratio(90 deg)         it stayed in image coordinates,
                                            i.e. it is the scanner or the codec
    """
    r0 = grain_residual(frame0)
    m0 = flat_mask(frame0)
    a0 = dict(cy=corr_length(r0, 0, m0), cx=corr_length(r0, 1, m0),
              flat=float(m0.mean()))
    a0["ratio"] = a0["cy"] / max(a0["cx"], 1e-9)
    out = dict(deg0=a0, block0=block_periodicity(r0, m0))
    if frame90 is None:
        out["verdict"] = "no 90-degree scan supplied; anisotropy unattributed"
        return out
    r9 = grain_residual(frame90)
    m9 = flat_mask(frame90)
    a9 = dict(cy=corr_length(r9, 0, m9), cx=corr_length(r9, 1, m9),
              flat=float(m9.mean()))
    a9["ratio"] = a9["cy"] / max(a9["cx"], 1e-9)
    out["deg90"] = a9
    out["block90"] = block_periodicity(r9, m9)
    prod = a0["ratio"] * a9["ratio"]
    same = abs(a0["ratio"] - a9["ratio"])
    out["product"] = float(prod)
    out["difference"] = float(same)
    if abs(prod - 1.0) < 0.15 and same > 0.15:
        out["verdict"] = ("the anisotropy TURNED WITH THE FILM: it is the "
                          "emulsion, and GrainSpec.anisotropy is the right home")
    elif same < 0.10:
        out["verdict"] = ("the anisotropy STAYED IN IMAGE COORDINATES: it is the "
                          "scanner or the codec, and storing it on a film "
                          "profile would be a category error")
    else:
        out["verdict"] = ("mixed -- neither test is clean; both a film and a "
                          "scanner component are present, or one scan moved")
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def analyse(repeats: list[Path], rot90: Path | None, gate: Path | None,
            dark: Path | None, flatness: float) -> dict:
    frames, facts = [], []
    for p in repeats:
        g, f = load_gray(p)
        frames.append(g)
        facts.append(f)
    if len(frames) < MIN_REPEATS:
        raise SystemExit(f"[!] {len(frames)} repeats; {MIN_REPEATS} is the "
                         f"minimum and {WANT_REPEATS} is the ask -- the "
                         f"standard deviation of a standard deviation is "
                         f"sigma/sqrt(2(N-1)), so fewer than six knows the "
                         f"floor to worse than 32 %")
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise SystemExit(f"[!] the repeats are not all the same size: {shapes}")

    shifts = [align_int(frames[0], f)[:2] for f in frames[1:]]
    moved = [s for s in shifts if s != (0, 0)]

    res = dict(
        n_repeats=len(frames),
        jpeg=facts[0],
        jpeg_consistent=all(f.get("q_dc") == facts[0].get("q_dc")
                            for f in facts),
        shifts=shifts,
        moved=len(moved),
        exposure=exposure_jitter(frames),
    )
    rep = repeat_sigma(frames)
    tot = local_sigma(frames[0], flatness=flatness)
    res["repeat"] = {k: v for k, v in rep.items() if not k.endswith("_map")}
    res["total"] = tot
    res["split"] = split_sigma(tot, rep)

    f90 = load_gray(rot90)[0] if rot90 else None
    res["anisotropy"] = anisotropy(frames[0], f90)

    for name, p in (("gate", gate), ("dark", dark)):
        if p:
            g, f = load_gray(p)
            res[name] = dict(mean=float(g.mean()), std=float(g.std()),
                             p01=float(np.percentile(g, 1)),
                             p99=float(np.percentile(g, 99)), jpeg=f)
    return res


def report(r: dict) -> int:
    bad = 0
    print("=== scan repeat analysis -- queue D1 + D2a ===")
    j = r["jpeg"]
    print(f"container: {j['format']} {j['mode']} {j['size'][0]}x{j['size'][1]}"
          f"  q_dc={j.get('q_dc', '?')}  DC quantiser rms="
          f"{j.get('dc_quant_rms', float('nan')):.3f} code values"
          if "q_dc" in j else f"container: {j['format']} {j['mode']}")
    if j.get("chroma_subsampled"):
        print("  ⚠ CHROMA IS SUBSAMPLED. On a monochrome original that is "
              "harmless; on a colour one the encoder halved the chroma "
              "resolution before anybody measured a colour cast.")
    if not r["jpeg_consistent"]:
        print("  ⚠ the repeats do not share one quantisation table -- the "
              "encoder changed settings mid-session, so the codec floor is not "
              "one number")
        bad += 1

    print(f"\nrepeats: {r['n_repeats']}")
    if r["moved"]:
        print(f"  ⚠ {r['moved']} of {r['n_repeats'] - 1} repeats are SHIFTED "
              f"against the first: {r['shifts']}")
        print("     the film moved. sigma_repeat now contains registration and "
              "is an upper bound, not the scanner. Re-shoot without touching "
              "the holder.")
        bad += 1
    else:
        print("  all repeats register to the first at integer precision")

    e = r["exposure"]
    print(f"\nD1 -- auto-exposure: mean level {e['mean']:.2f}, spread "
          f"{e['spread']:.2f}, sd {e['std']:.3f}")
    if e["re_exposing"]:
        print("  ⚠ THE SCANNER RE-EXPOSES BETWEEN FRAMES. A gate frame is a "
              "white point for ITSELF and cannot calibrate a batch; the "
              "archived 546 cannot be made absolute retroactively.")
    else:
        print("  ✅ the exposure HELD across the repeats. A gate frame taken in "
              "this session IS a valid white point for it.")

    print("\nD2a -- the split, per level (code values):")
    print("  level     sigma_tot  sigma_scan  sigma_emul  scanner share  "
          "identical")
    nclamp = 0
    for s in r["split"]:
        nclamp += bool(s["clamped"])
        print("  %6.1f  %9.3f  %10.3f  %10.3f  %12.1f%%  %8.1f%%%s"
              % (s["level"], s["sigma_total"], s["sigma_scanner"],
                 s["sigma_emulsion"], 100 * s["scanner_share"],
                 100 * s["identical_fraction"],
                 "  ⚠ CLAMPED" if s["clamped"] else ""))
    print(f"  flat windows used: {100 * r['total']['flat_fraction']:.1f}% of "
          f"the frame")
    print(f"  bit-identical pixels across all repeats: "
          f"{100 * r['repeat']['identical_overall']:.1f}%")
    if r["repeat"]["identical_overall"] > 0.5:
        print("  ⚠ MORE THAN HALF THE FRAME IS BIT-IDENTICAL ACROSS REPEATS. "
              "The codec quantised the scanner's noise away, so every "
              "sigma_scanner above is a LOWER BOUND and the split OVER-credits "
              "the emulsion. Read the DC quantiser rms as the floor instead.")
    if nclamp:
        print(f"  ⚠ {nclamp} bin(s) clamped at zero -- measured scanner noise "
              f"exceeded measured total, which is impossible, so one of the two "
              f"estimates is wrong in those bins. Distrust this run.")
        bad += 1

    a = r["anisotropy"]
    print("\nD2a -- anisotropy:")
    print("  0 deg : corr length y %.2f px, x %.2f px, ratio %.3f"
          % (a["deg0"]["cy"], a["deg0"]["cx"], a["deg0"]["ratio"]))
    if "deg90" in a:
        print("  90 deg: corr length y %.2f px, x %.2f px, ratio %.3f"
              % (a["deg90"]["cy"], a["deg90"]["cx"], a["deg90"]["ratio"]))
        print("  product %.3f, difference %.3f" % (a["product"], a["difference"]))
    print("  verdict: " + a["verdict"])
    b = a["block0"]
    det = [k for k in ("y", "x") if b[k]["detectable"]]
    if not det:
        print("  JPEG 8-px grid: NOT DETECTABLE -- the autocorrelation at lag 8 "
              "is under 0.02 on both axes, i.e. the grain has already decayed "
              "there and no block structure rises above it. That is not the "
              "same as absent.")
    else:
        print("  JPEG 8-px grid in the residual: "
              + ", ".join("%s ratio %.3f" % (k, b[k]["ratio"]) for k in det)
              + "  (1.0 = no block structure)")
        if max(b[k]["ratio"] for k in det) > 1.2:
            print("  ⚠ THE RESIDUAL CARRIES THE CODEC'S BLOCK GRID. Part of "
                  "what a single-frame grain estimator reads as emulsion is "
                  "the encoder.")

    for name in ("gate", "dark"):
        if name in r:
            g = r[name]
            print(f"\n{name}: mean {g['mean']:.2f}, sd {g['std']:.3f}, "
                  f"1st pct {g['p01']:.1f}, 99th {g['p99']:.1f}")

    print("\n⚠ EVERY SIGMA ABOVE IS IN CODE VALUES, NOT DENSITY. Converting "
          "needs the transfer curve, which is queue D2b and still open. The "
          "SPLIT is a ratio and is scale-free, which is the part C18/C19 need.")
    return bad


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def synthetic(tmp: Path, n: int = WANT_REPEATS, seed: int = 7) -> dict:
    """Build scans whose truth is known, so every estimator can be checked today.

    ⚠ THE POINT IS THAT THE ANSWERS ARE KNOWN BY CONSTRUCTION. A grain field
    with a chosen correlation length and a chosen amplitude, plus per-scan noise
    of a chosen sigma, written through a real JPEG encoder -- so the codec's
    contribution is the real codec's and not a model of it.
    """
    from PIL import Image
    rng = np.random.default_rng(seed)
    h, w = 512, 640
    # a picture: broad ramps and a few flat plateaus, nothing high-frequency
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    pic = 40 + 150 * (0.5 + 0.5 * np.sin(xx / 160.0)) * (0.4 + 0.6 * yy / h)
    pic[100:200, 100:300] = 90.0
    pic[300:400, 350:600] = 190.0

    # Grain: white noise smoothed isotropically, then stretched along y by a
    # 3-tap moving average. ⚠ THE TRUTH IS A CORRELATION-LENGTH RATIO OF ABOUT
    # 1.14, not of 3 -- a 3-tap average along one axis of a field already
    # correlated over ~3 px moves the 1/e crossing only a little, and quoting
    # the tap count as if it were the anisotropy would set this test an
    # impossible target and then call the estimator broken for missing it. The
    # measured raw field gives cy 3.58 against cx 3.14.
    g = rng.normal(0.0, 1.0, (h, w))
    g = _boxsum(g, 5) / 25.0 * 5.0
    gy = (g + np.roll(g, 1, 0) + np.roll(g, 2, 0)) / np.sqrt(3.0)
    grain = 3.0 * gy / max(gy.std(), 1e-9)

    truth = dict(sigma_grain=3.0, sigma_scan=1.5)
    base = np.clip(pic + grain, 0, 255)

    paths = []
    for i in range(n):
        f = np.clip(base + rng.normal(0.0, truth["sigma_scan"], (h, w)), 0, 255)
        p = tmp / f"rep{i:02d}.jpg"
        Image.fromarray(f.astype(np.uint8)).save(p, "JPEG", quality=92,
                                                 subsampling=0)
        paths.append(p)
    # the same FILM rotated: rotate base, add fresh scanner noise
    rot = np.rot90(base).copy()
    rp = tmp / "rot90.jpg"
    Image.fromarray(np.clip(rot + rng.normal(0, truth["sigma_scan"],
                                             rot.shape), 0, 255).astype(np.uint8)
                    ).save(rp, "JPEG", quality=92, subsampling=0)
    truth["repeats"] = paths
    truth["rot90"] = rp
    return truth


def self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        t = synthetic(tmp)
        r = analyse(t["repeats"], t["rot90"], None, None, FLATNESS_MAX)
        bad = report(r)
        print("\n--- self-test verdict ---")
        print("truth: grain sigma %.2f, scanner sigma %.2f"
              % (t["sigma_grain"], t["sigma_scan"]))
        flat = [s for s in r["split"] if s["n" if "n" in s else "level"]]
        got_s = np.median([s["sigma_scanner"] for s in r["split"]])
        got_e = np.median([s["sigma_emulsion"] for s in r["split"]])
        print("recovered (median over bins): scanner %.2f, emulsion %.2f"
              % (got_s, got_e))
        # ⚠ THE SHORTFALL IS THE CODEC AND IT IS THE HEADLINE OF THIS TEST, not
        # a tolerance to be widened until it passes. At quality 92 with a DC
        # quantiser step of 3, a scanner noise of 1.50 code values reads as
        # about 1.26 -- the encoder absorbed roughly a third of the variance.
        # A real UF15 JPEG will be coarser than quality 92, so the shortfall on
        # the owner's files will be LARGER than this, and every sigma_scanner
        # the report prints is a lower bound by about this much.
        print("  JPEG absorbed %.0f%% of the scanner variance at quality 92 "
              "(%.2f measured against %.2f injected) -- the real files will be "
              "coarser, so treat sigma_scanner as a lower bound"
              % (100 * (1 - (got_s / t["sigma_scan"]) ** 2),
                 got_s, t["sigma_scan"]))
        ok = True
        # ⚠ THE SCANNER FIGURE IS THE ONE THAT MUST BE TIGHT. It is measured
        # directly from the repeats and nothing else stands between the
        # estimator and the truth except the encoder.
        if not (0.6 * t["sigma_scan"] <= got_s <= 1.6 * t["sigma_scan"]):
            print("  ⚠ scanner sigma is outside 0.6-1.6x of truth")
            ok = False
        # The emulsion figure passes through the flatness test and the encoder,
        # so it is allowed a wider band -- but the ORDER must be right.
        if got_e <= got_s:
            print("  ⚠ recovered emulsion sigma is not above the scanner's, "
                  "and in this synthetic it is twice it")
            ok = False
        av = r["anisotropy"]
        if "deg90" in av and av["deg0"]["ratio"] <= 1.05:
            print("  ⚠ the synthetic grain is stretched along y and the 0-degree "
                  "correlation ratio did not exceed 1: got %.3f"
                  % av["deg0"]["ratio"])
            ok = False
        if "deg90" in av and abs(av["product"] - 1.0) > 0.20:
            print("  ⚠ the two ratios should be reciprocal on a film-borne "
                  "anisotropy; product %.3f" % av["product"])
            ok = False
        print("SELF-TEST " + ("PASS" if ok and bad == 0 else "FAIL"))
        return 0 if (ok and bad == 0) else 1


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", nargs="*", default=[],
                    help="the repeat scans of ONE untouched frame")
    ap.add_argument("--dir", help="a directory; every D2a_rep*.* in it is a repeat")
    ap.add_argument("--rot90", help="the same frame rotated 90 degrees")
    ap.add_argument("--gate", help="the empty-gate frame (D1)")
    ap.add_argument("--dark", help="the opaque/dark frame (D1)")
    ap.add_argument("--flatness", type=float, default=FLATNESS_MAX,
                    help="max range of 3x3 box means for a window to count as "
                         "flat, code values (default %(default)s)")
    ap.add_argument("--json", help="also write the full result here")
    ap.add_argument("--synthetic", action="store_true",
                    help="run the self-test on data whose truth is known")
    a = ap.parse_args()

    if a.synthetic:
        return self_test()

    reps = [Path(p) for p in a.repeats]
    if a.dir:
        d = Path(a.dir)
        reps += sorted(p for p in d.iterdir()
                       if p.name.lower().startswith("d2a_rep"))
    if not reps:
        ap.error("no repeats given; use --repeats, --dir, or --synthetic")

    r = analyse(reps, Path(a.rot90) if a.rot90 else None,
                Path(a.gate) if a.gate else None,
                Path(a.dark) if a.dark else None, a.flatness)
    bad = report(r)
    if a.json:
        Path(a.json).write_text(json.dumps(
            {k: v for k, v in r.items() if k != "repeat" or True},
            indent=1, default=float), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
