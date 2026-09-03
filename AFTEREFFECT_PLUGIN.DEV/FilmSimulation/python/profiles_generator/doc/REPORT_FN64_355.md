# SVEMA FN-64 -- adoption report for the 355-frame v2 batch

> ⚠ **Status note 2026-08-18 — READ THIS FIRST. The batch this report is
> about is NOT one emulsion, and three of the values adopted below (or in the
> 509-frame run that superseded it) have since been WITHDRAWN.**
>
> The analyzer was pointed at a folder named `SVEMA-FN64`. The owner confirms
> only frames `PICT0001`–`PICT0067` are certainly Foto-65; frames 68+ are a
> **mixture of Foto-32 and Foto-65**, and Foto-32 was chosen deliberately for
> finer grain, so the mixed batch reads **finer and sharper** than Foto-65
> alone. A confirmed-subset re-run shows all 67 frames are **exactly
> greyscale** (`max |R−G| = max |B−G| = 0`), so every per-channel figure in
> this report — `base_tint`, `silver_tone`, the crossover bins — measured
> scanner white-balance drift across the contaminated tail, not the film.
>
> **Withdrawn:** `base_tint` → identity; `silver_tone` → 0.0;
> `sigma_shape` → schema default (the two runs disagree in **sign**).
> **Kept:** gamma 0.830 (now re-based on the printed Gurlev 1986 p296 figure,
> γ_rec 0.8, per method rule 14 — the batch statistics 0.677/0.834 are a
> consistency bracket only), clump 23 µm, halation.
> **Also corrected:** the "Bayer-demosaiced DSLR scan" described in the
> *Rejected* section below is wrong — EXIF reads `Make=GCMC`,
> `Model=Scanner`, `Software=UF15 16/08/20 v0.69`. That invalidates the
> stated *reason* for rejecting `anisotropy 0.621`, and the value is
> reproducible (0.658 mixed / 0.634 confirmed). Still not adopted, but now an
> open question rather than a settled rejection.
>
> Full record: `DATASHEET_VERIFICATION_REPORT.md`, addendum 2026-08-18, and
> `RESULT_2026-08-18_svema_clean67.md`. Everything below is kept **verbatim**
> as the historical adoption record. Do not adopt values from it directly.

> **Status note 2026-08-02:** historical adoption record. Values below were
> later superseded by the 509-frame batch (gamma 0.79 → 0.83 adopted, see
> comments in `film_profiles.py`), and the profile is now also corroborated
> by a printed source — Gurlev 1986 p296, Foto-65 column (γ_rec 0.8, D0
> 0.05, R 110 lin/mm, Δλ_S 665 nm); tier raised to 2.

Implied scan resolution: 4416 px / 36 mm = 122.7 px/mm (3:2 aspect check
passed). You did not pass --frame-width-mm; next run, do -- it unlocks um
units and RMS-48 in the analyzer output directly.

## Adopted

| Field | Old | New | Basis |
|---|---|---|---|
| gamma | 0.86 [T3] | **0.79** [T2] | interdecile span 1.494 / 1.9 logE; sits in GOST cine range 0.65-0.8; aged film loses contrast |
| sigma_shape toe/mid/dmax | 0.4/1.0/1.2 [T3] | **0.70/1.0/1.35** [T2] | measured 0.021/0.028/0.037, 0.01 scanner floor removed in quadrature |
| clump_um | 15 [T3] | **23** [T2] | corr length 3.48 px = 28 um raw, ~23 um after 2 px PSF deconvolution; rendered amplitude unchanged (RMS calibration re-normalises) |
| base_tint | (.996,1,.979) | **(.992,1,.988)** [T2] | 355 frames beat 290; still scanner-WB contaminated |
| silver_tone | -0.25 | **-0.10** | measured density drift near neutral (dD < 0.01 over 3 D); weak cold kept for the crow-wing look |
| halation | OFF | **ON: gains 0.09, radii (12,69,320) um, weights (.30,.55,.15)** [T2] | 58 highlight frames: 0.24 D excess (~0.28 bias-corrected), 1/e 69 um |
| dmin | 0.174 | **0.16 (reverted)** | 0.174 was my misreading of v1 output (pixel value, not density); v2 batch shows base at scanner white -> absolute base+fog UNKNOWABLE without --empty-gate |

## Rejected, with reasons

**anisotropy 0.621** -- grain is physically isotropic; a green-channel value
that far from 1 on a Bayer-demosaiced DSLR scan is the sensor mosaic, not
the film. Profile keeps 1.10.

**dmax 3.4996** -- your scan's 8-bit encoding ceiling, warned by the
analyzer itself. Not the emulsion.

**rms_granularity change** -- not needed: the new native-res mid sigma
(0.028) confirms the existing [T1] fit (which produced 0.030). Good news,
not a change.

**scene_stops 14.7** -- span/gamma arithmetic on a ceiling-clipped span;
reported, not a film property.

## Halation notes

Verified through the render pipeline at 41 px/mm: decay length matches the
measured 69 um; edge amplitude lands at roughly 0.11 D-equivalent vs your
0.24 measured. Three stacked assumptions separate them (highlight overshoot
stops, ring-1 scanner edge-smear inflation, print-gamma translation), all
documented in the profile comment. If renders read weaker than your scans,
the plugin's halationStrength control at 1.5-2.0x closes the gap -- that
knob exists precisely for this.

## Gauge variants (16 mm, 8 mm; "64 mm" read as 35 mm)

All emulsion numbers mirrored verbatim -- same coating, slit narrower.
Density does NOT scale with gauge (per-unit-area property). Apparent grain
DOES, and the pipeline derives it from default_format automatically;
flat-field verified after this update: 16 mm renders 1.22x, 8 mm 1.39x the
35 mm grain amplitude with identical GrainSpec values.

## Upgrade path (repeats, because it matters)

One --empty-gate frame makes dmin absolute. One +-4 EV bracket of a grey
card with --wedge makes gamma, toe and shoulder MEASURED instead of
ESTIMATE. Ten frames, biggest remaining accuracy win.
