# analyze_film_scans.py v2.0

Rewrite of the frame-scan analyzer. numpy + Pillow only, Python 3.10+.
Same TXT output style; every value now carries an honesty tier:
[MEASURED] / [ESTIMATE] / [LOWER-BOUND] / [CONTAMINATED] / [DEFAULT].

## What the old script got wrong (and v2 fixes)

1. **No density conversion.** v1 reported raw sRGB pixel percentiles and
   called them dmin/dmax. Your SVEMA64_MY file: "dmin 0.1746" was a dark
   PIXEL VALUE = density 1.59 (scene highlights); "dmax 0.8123" was the
   base = density 0.20. v2 decodes sRGB, converts to density, and labels
   the base as the base. **Correction to my previous adoption: your
   batch's real base+fog is ~0.20 D, not 0.174 -- see note below.**
2. **gamma = 2.2/contrast heuristic, clamped to [1,3]** -- that is where
   your 3.0 came from. v2: real gamma from an exposure wedge, or an
   honest ESTIMATE with the assumption printed.
3. **Grain block was hardcoded defaults.** v2 measures it, at native
   resolution, in density space, per channel, per tone region.
4. **Resize to 1024x1024** destroyed both grain and aspect ratio. v2
   analyzes tone on smoothed subsamples and grain on native pixels.

## What it measures

Tone: percentile ladder, base+fog, Dmax lower bound, toe/shoulder spread
ratios, dynamic range. Gamma: wedge mode (real) or batch mode (estimate).
Grain: RMS-48 granularity and sigma(D) per channel in THIN / MID / DENSE
tone bins, grain size (correlation length, um) per tone bin, anisotropy.
Halation: strength (D) and 1/e radius (um) per channel. Base tint;
density-weighted image tone drift (warm/cold with density); field
unevenness (vignette vs coating mottle); spectral sharpness proxy.

## Validation against synthetic ground truth

| Quantity | Truth | Recovered |
|---|---|---|
| gamma (batch mode, span matched) | 0.85 | 0.84-0.85 |
| gamma (wedge mode, mid slope 0.814) | 0.814 | 0.793 |
| toe / shoulder onset (wedge) | -1.0 / +1.4 | -0.90 / +1.20 (onset = before the bend; correct) |
| base+fog dmin | 0.200 | 0.190 |
| base tint r/b | 1.021 / 0.979 | 1.020 / 0.979 |
| grain sigma(D), 3 px clumps | 0.030 | 0.031-0.032 mid/dense, 0.040 toe (upper bound) |
| grain corr length | ~6 px | 5.6 px |
| halation strength | 0.12 D | 0.09-0.11 (documented -15..20% bias) |
| halation radius | 20 px | ~28 px (ESTIMATE, biases long) |

Three estimator designs failed validation before this one and are
documented in the code: far-window line fit (ate the halo tail),
quadratic extrapolation (overshot, rectified negative), joint exp+poly
fit (collinear when the window is short). Final design: masked-box local
background + ring profile + hard far-ring flatness gate.

## How to shoot for the best profile

1. **Empty gate**: one shot of the light source with no film (or clear
   rebate), same settings. Pass as --empty-gate. Absolute densities.
2. **Exposure wedge**: ONE scene (grey card best), bracketed -4..+4 EV in
   1 EV steps, filenames like `frame_-2EV.jpg`. Run with --wedge.
   This replaces the gamma estimate with a real measured curve including
   toe and shoulder. Ten such frames beat 300 unknown ones.
3. **Resolution**: >= 63 px/mm (1600 dpi over 35 mm) or the 48 um
   granularity aperture is under-resolved (script warns).
4. **For halation**: include night/indoor frames with small bright lamps.
5. Always pass --frame-width-mm (or --px-per-mm), else no um units.
6. Positive/inverted scans: add --positive.

## Usage

    python3 analyze_film_scans.py SCANS_DIR -o profile.txt \
        --frame-width-mm 36 --empty-gate gate.jpg
    python3 analyze_film_scans.py WEDGE_DIR -o curve.txt --wedge \
        --frame-width-mm 36 --empty-gate gate.jpg

## IMPORTANT correction to the previous FN-64 update

The dmin=0.174 I adopted into SVEMA_FN_64 last time came from v1's
mislabeled output and is a misreading: 0.1746 was a pixel value, not the
base density. Your batch's base+fog is ~0.20 D IF the scan is sRGB and
scanner white = no film -- neither is guaranteed without --empty-gate.
Recommendation: rerun your 290 frames through v2 with an empty-gate
frame, then adopt its dmin. Until then the honest profile value is the
old estimate 0.16 or the new ~0.20; say the word and I will set either.
