# analyze_film_scans.py v2.1

v2.0 + the movie-team review, honestly implemented. numpy + Pillow,
Python 3.10+. Output format and value types unchanged from v2.0.

## The three review blocks

### 1. [SpectralResponse] -- with one hard correction to the premise

The review asked for a "red index" computed from ordinary frames. That is
not physically possible and v2.1 says so instead of inventing numbers:
developed silver is spectrally near-neutral, so the scan of a B&W negative
carries NO memory of which wavelengths exposed it. An ortho film's
dark-red-lips signature lives in the scene rendering, not in scan channel
ratios (channel ratios measure base tint + silver tone -- different
sections already).

What v2.1 adds instead is the measurable protocol: shoot uniform RED /
GREEN / BLUE patches on the film at one exposure, scan, name files red*,
green*, blue*, pass --spectral-dir DIR. Density each patch produced =
response to that band; green-normalised weights land in exactly the
spectral_weights convention of the simulation profiles. Validated on
synthetic ortho truth (0 / 0.524 / 0.476): recovered 0.000 / 0.523 /
0.477, "ORTHOCHROMATIC response confirmed" fires.

### 2. [Crossover]

Median (D_r - D_g) and (D_b - D_g) per density bin (12 bins, streaming
2D histograms, per-bin medians so coloured scene objects cannot drag the
estimate), plus toe/dense divergence relative to mid. Green-shadow /
magenta-highlight faults of old colour processes show here; on B&W it
reads silver-tone curvature. Stated limit: a batch with no neutral
content biases it.

### 3. [CONTAMINATED-BY-AGING]

Base tint strongly yellow/brown (tint_b < 0.94, or warm+blue-weak combo)
now flags the roll: likely acetate degradation, not emulsion design.
Advice printed: simulate as roll aging, keep the stock's base_tint clean.

## Precision upgrade (review's "highest accuracy" request)

* float64 end to end for every intermediate (v2.0 mixed float32 arrays);
* 16-bit TIFF/PNG scans read at full depth -- v2.0 flattened everything
  through 8-bit RGB, a 256x precision loss on exactly the scans a movie
  team would make;
* histogram percentiles interpolated within bins, 8192 bins (quantisation
  ~4e-4 D, interpolation below that);
* box filters keep float64 (no downcast to input dtype).

Output TXT numbers keep their original formats/rounding.

## Review's code notes, checked

* _corr_len None path: confirmed safe -- every report line that depends
  on it is conditional; sensor-noise-dominated ACFs simply omit the line.
* BaseTint exponent-of-density-difference: confirmed correct, unchanged.

## Regression

Full v2.0 validation suite re-run: wedge gamma/toe/shoulder identical;
batch dmin/gamma/tint/grain/halation identical to within the interpolated
percentile refinement (<=2e-4 D). Nothing regressed.

## Does film_profiles need extending? (short answer: no)

* spectral_weights -- already exists; [SpectralResponse] feeds it as-is.
* crossover -- COLOUR stock: already representable, the three ToneCurves
  are independent per channel, so differing toe/shoulder/gamma per layer
  IS crossover; fit the [Crossover] table into them. B&W: silver_tone is
  linear-only, but measured curvature so far is +-0.01 D -- below
  visibility. If a stock ever measures >0.03 D, add a two-float
  silver_tone_curve (toe, dense) then, not now.
* aging -- AgingSpec hooks already exist and profiles deliberately ship
  fresh; the [CONTAMINATED-BY-AGING] flag routes roll damage to aging
  controls instead of polluting the stock's base_tint. No change.
