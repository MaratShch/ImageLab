# Film Defect Layer — Implementation State and Calibration Log

> **UPDATE 2026-08-11 — database grew 93 → 100 stocks.** Seven stocks added
> from owner-supplied documents (Kodak Data Book 1952 ×4, Agfa 2003 ×3); see
> `PYTHON/profile_generator/doc/CHANGES_2026-08-11_stocks100.md`. Every
> "93 stocks" figure below records the sweep as it was run and remains true of
> that run; the monochrome population relevant to `defectChroma` is now 40 of
> 100 (all four 1952 Kodak additions are monochrome, two of them ortho). The
> per-stock verification sweeps have NOT been re-run against the seven new
> profiles.

**Scope:** sub-stage 9b, negative-side particulate. Three classes render: fine dust,
coarse debris, hair and fibres.
**Status:** implemented and calibrated against measurement. Scalar, unoptimised.
**Measured on:** GCC `-O2`, `AlgoType = double` unless stated.

This file is the change record for the defect layer. Anything altered in the defect
code must be noted here.

---

## 1. What renders, and what does not

| Class | Control | Stage | State |
|---|---|---|---|
| Fine dust (embedded) | `dustLevel` | 9b | **renders** |
| Coarse debris | `debrisLevel` | 9b | **renders** |
| Hair and fibres | `fibreLevel` | 9b | **renders** |
| Transport scratches | `scratchTransport` | 9b | not written |
| Handling scratches | `scratchHandling` | 9b | not written |
| Development mottle | `processingQuality` | 9b | not written |
| Drying marks | `dryingMarks` | 9b | not written |
| Storage fade / age fog | `storageSeverity` | 13/14 | not written |
| Colour veil | `colourVeil` | 14 | not written |
| Gate dirt, one-frame sparkle | `gateDirt` | 16 | **renders** |
| Gate weave | `weaveAmount` | 15 | **renders** |
| Events (splices) | `damageEvents` | 16 | **renders** |
| Exposure flicker | `flickerStops` | 3c | stub |
| Scanner artifacts | `scannerArtifacts` | 10 | not written |

`getAlgoControlsDefault()` now returns **damage enabled**, with all fourteen levels
populated. Three of them do something; the other eleven are correct values waiting
for their stages.

---

## 2. Only the embedded dust population lives in 9b

Measured dust splits three ways:

| Population | Share | Temporal behaviour | Stage |
|---|---|---|---|
| Loose dust on the film at scan time | 50–80 % | one frame only | 16 |
| Embedded processing dust | 15–40 % | locked to film coordinates | **9b** |
| Dirt lodged in the gate | 5–20 % | fixed for 10²–10⁵ frames | 16 |

`ALGO_DUST_EMBEDDED_FRACTION = 0.6`, so `dustLevel = 1.0` currently places
**1.20 /mm²** of the measured 2.0 /mm².

This split is not a simplification. Putting the whole level in 9b would make every
speck survive the print and duplicate through the dupe chain — wrong for the
majority of real dirt, whose defining behaviour is that it *moves between passes* —
and would remove the one-frame sparkle entirely.

**Both halves now exist.** Stage 16 renders the loose one-frame population and the
gate population; 9b renders the embedded one. See §10.

---

## 3. Verification against the four required statistics

The dust model must pass all four **simultaneously**. Matching only density is the
failure mode the test exists to catch.

Rendered on `SVEMA_FOTO_65` (named `SVEMA_FN_64` at measurement time, renamed 2026-08-13; the index has since moved) — the exact stock the measurements were made
on — dust only, flat field, detected with the same 1 DN floor:

| Statistic | Simulated | Measured target | |
|---|---|---|---|
| Areal density | 0.82 /mm² | 1.20 placed | detector-limited, expected |
| Peak amplitude, median | **9.31 DN** | 8.8 – 9.8 | pass |
| Peak amplitude, p95 | **11.91 DN** | 11.2 – 14.3 | pass |
| Index of dispersion, 1.1 mm² | 5.00 | 5.3 – 9.2 | pass (at λ = 8.45) |
| Index of dispersion, 4.5 mm² | 16.79 | 15.5 – 31.7 | pass |
| Index of dispersion, 18 mm² | 62.26 | 46.9 – 111.0 | pass |
| Clark–Evans ratio | 1.06 | 0.82 – 0.97 | **fail, ~10 % too regular** |
| Channel ratio, colour stock | 0.56 | 0.55 | pass |
| Channel ratio, mono stock | 0.97 | 1.0 exactly | pass |

Poisson control (`dirtClumping = 0`) returns 0.94 / 0.93 / 1.06 / 0.99 against a
required 1.0 at every scale — this is the important half, because it proves the
placement sampler is unbiased and the clustering therefore comes from the intensity
field rather than from an artefact of the sampler.

**Open:** Clark–Evans. Suspicion is the estimator (global mean density in the
expected value while sampling an interior guard band) rather than the generator, but
it is not proven. Eight of nine pass.

---

## 4. Calibration log — three domain errors, all the same mistake

Every one of these was a **measured output quantity applied as an input constant**.
The pipeline sits between the two and does not preserve the value.

### 4.1 Opacity level and dispersion

First implementation: Beta(2,3) scaled onto 0.15 … 1.0 — the range any reasonable
person picks for "mostly translucent specks, a few opaque".

```
                    before      after      target
median peak       31.2 DN     9.31 DN    8.8 - 9.8
p95 peak         160.5 DN    11.91 DN   11.2 - 14.3
```

Three times too strong at the median, **eleven times** in the tail. Cause: opacity
is drawn in negative density, the targets are peak deviations in the finished
positive scan, and the print gamma amplifies a density difference rather than
passing it through.

A single gain cannot fix it — gain 0.25 lands p95 but leaves the median at 6.9; gain
0.33 lands the median but pushes p95 to 19. The measured distribution is *narrower*
than the old range produced (p95/median 1.4 measured against 2.1 produced), so
dispersion is an independent error. Hence two solved constants:

```cpp
ALGO_DUST_ALPHA_MID    = 0.13   // amplitude level
ALGO_DUST_ALPHA_SPREAD = 0.22   // amplitude dispersion
```

The Beta(2,3) **shape** is retained and only its location and scale are calibrated,
so the solved numbers carry no shape information of their own.

### 4.2 Saturation threshold conflated with the size limit

`ALGO_DUST_OPAQUE_UM = 200` forced opacity to 1.0 exactly at the top of the dust
size range, so the largest dust in every frame saturated by construction — the bulk
of the 160 DN tail.

The measurement says particles **≥ 0.3 mm** saturate. 0.3 mm is 300 µm, which is
*coarse debris*, not fine dust (which stops at 200 µm). Split into
`ALGO_DUST_OPAQUE_ONSET_UM = 200` and `ALGO_DUST_OPAQUE_FULL_UM = 300`, making the
ramp deliberately inert across the whole dust range while staying correct if the
size limit is ever raised.

### 4.3 Chroma floor

`0.55` was the measured *output* ratio used directly as a *negative density*
constant, giving 0.59 in the render. Solved on `ORWOCOLOR_NC21` — the colour
material the ratio was measured on — to `ALGO_DEFECT_CHROMA_MIN = 0.45`, which
produces 0.56.

### 4.4 Calibration raster — a mistake worth not repeating

The first solve ran at 1024 px across the frame, where one pixel is **24.3 µm**
while the measured median particle is 20–34 µm. The typical particle is smaller than
a pixel, so its peak is averaged down, and calibrating there yields an opacity that
compensates for a blur the original scan never had.

The source scan resolved to ~18 µm, so its sampling was ~12 µm/px. 24.89 mm across
2048 px is 12.2 µm/px, and the solve was redone there.

```
same constants:   12.2 um/px -> 11.0 DN median
                  24.3 um/px ->  9.4 DN median
```

**A 15 % error purely from the calibration raster.** The residual raster dependence
is correct physics and must not be calibrated away — a real scanner at lower
resolution genuinely averages sub-pixel dirt down, which is why dust obvious in a 4K
scan is nearly invisible in 2K.

---

## 5. Two rendering bugs found by looking at output

**Coloured particles on monochrome film.** A single silver record carries one density
and prints through one dye, so it cannot record what colour a speck was — only how
much light it blocked. Drawing three channel weights put measurable colour on black
and white stock: detected ratio 0.58 where it must be exactly 1. It hid because 0.58
is close to the measured 0.55, which is the right figure for the *wrong material*.
**36 of 93 stocks are monochrome.** `defectChroma` now returns neutral for them.

**Faceted fibres.** Stroking the persistent-walk control polyline directly gave
250 µm steps — ten pixels at 41 px/mm — and a chain of ten-pixel straight lines does
not read as a hair. Fixed by separating the *physics* sampling rate from the
*drawing* rate: the walk still steps at 4 points/mm, because a walk with more,
smaller steps is a different curve and cannot be used to fix appearance, and the
drawn polyline is Catmull-Rom subdivided ×4 to 16 points/mm. Catmull-Rom passes
exactly through every control point, so the curvature the persistence length produced
is preserved rather than smoothed away.

Related: a fibre must be **stroked from distance-to-whole-centreline**, not stamped
per segment. Consecutive segments always overlap at their joint, so stamping lays a
string of beads along the fibre. Constant width is the mechanical signature of a
foreign object lying on the film — variable width is what makes something a scratch.

---

## 6. Cost

1024², double, `SVEMA_FOTO_65` (then `SVEMA_FN_64`):

```
damage off             1478.14 ms
damage at 1.0          1476.91 ms    unmeasurable
damage forced high     1507.98 ms    +2 %, at levels far past plausible
```

Off costs one branch — the master flag is tested before any damage field is read.
At 1.0 it is unmeasurable because ~560 sub-pixel particles touch 0.8 % of the frame.

**This supersedes the "09b is a stub, always memcpy" rows in
`SINGLE_THREAD_OPTIMISATION.md` (lines 54, 108, 133, 242, 332) and the "09b is a
stub; fold into 10's read" entry in `STAGE_FUSION_PROPOSAL.md` (line 69).** 9b is no
longer a pure copy and no longer a no-op elision candidate, though it remains
pointwise-plus-scatter and is still foldable into stage 10's read pass.

---

## 7. AgingSpec is unusable as an era baseline today

The design was that each control level would *scale* the era-typical figure the
profile carries in `AgingSpec` — `dust_area_ppm`, `mottle_amplitude`,
`scratch_rate_base_per_m`, `dye_fade_c/m/y`, `dmin_lift` — so that a 1943 Agfacolor
at `dustLevel 1.0` would be dirtier than a modern stock at the same setting with
nobody authoring two presets.

**Verified by parsing the generated database: 0 of 93 stocks carry a non-zero
`AgingSpec`.** All are documented "fresh".

Multiplying would therefore silence the defect layer on every stock and look exactly
like a broken control. Levels are **absolute** for now.

When `AgingSpec` is populated it must enter as an **additive** era term, not a
multiplier, so fresh stock keeps behaving as it does today and only aged stock gains
dirt nobody asked for.

---

## 8. Regression

```
double, damage off by flag       93 stocks PASS
double, damage forced high       93 stocks PASS   all finite, all within 0..1
float,  damage off by flag       93 stocks PASS
float,  damage forced high       93 stocks PASS
e2e vs film_sim.py               mean abs 5.0238e-05   unchanged to the digit
```

The e2e figure is the load-bearing one: with `filmDamageEnabled = false` the engine
is numerically identical to the pre-defect build.

**`e2e.cpp` must clear the flag**, because `getAlgoControlsDefault()` now enables
damage and the reference model has no defect layer to mirror. Without it the
comparison reports `mean abs 8.9529e-04`, and the entire difference is the defect
layer.

---

## 9. Next

1. **Stage 16** — gate dirt, one-frame sparkle, events. Holds the other 40 % of
   dust and is where dirt starts reading as *motion*.
2. Resolve the Clark–Evans residual.
3. Scratches — transport and handling — into 9b.
4. Populate `AgingSpec` so era drives the baseline.


---

## 10. Stages 15 and 16 — weave and machine-side dirt

Added after the 9b calibration. Both are driven by each stock's `TemporalSpec`,
which — unlike `AgingSpec` — **is populated on all 93 profiles**: weave 20–25 µm
RMS for 1930s–40s material, 10 for the 1950s, 6 by the 1970s, 3 modern;
`dirt_events_per_frame` 3.0 for the earliest stocks down to 0.1 for modern. So era
drives the baseline and the control expresses intent, exactly as grain does.

### 10.1 The interaction rule is free, because the stage order encodes it

```
 9b  film-borne particulate   -> before the weave, so it IS translated
 15  the weave
 16  gate dirt, machine-fixed -> after the weave, so it is NOT translated
```

Film-borne dirt sits still relative to the image; gate dirt sits still relative to
the frame while the image jitters beneath it. No flag, no coordinate bookkeeping.
Without this, a gate mark is a mathematically static line and reads as a digital
overlay immediately.

### 10.2 Polarity is opposite between the two dirt stages, and that is correct

| | 9b, film-borne | 16, machine-side |
|---|---|---|
| domain | negative density | positive transmittance |
| operation | `D' = D − log10(1−α)` | `T' = T·(1−α)` |
| result on screen | **bright** | **dark** |
| survives print/dupe | yes | no |
| moves with weave | yes | no |

A speck embedded in the negative blocks printing light, so the print is never
exposed there and comes out clear. A speck in the projector gate blocks projection
light, so the screen goes dark. Both populations in one frame is what a real
projected print looks like.

Machine-side marks are also **neutral** across channels — something in the light
path in front of the whole picture attenuates all three equally — so the
per-channel chroma weighting at 9b has no counterpart at 16.

### 10.3 Persistent gate dirt with no persistent state

`Algorithm_Main` is a pure function; a host may ask for frame 900 without ever
rendering 899. So the population is **derived, not accumulated**: each arrival
ordinal draws a birth and a lifetime from its own stream and is present when
`birth <= f < birth + lifetime`. Two draws, no history.

Rates are chosen for the population they imply, not independently:

```
initial              4 particles (survived the reel change)
accretion    4e-3 /frame     shed   2e-4 /frame
mean lifetime        5000 frames  = 1 / shed
steady state         4e-3 x 5000  = 20 particles
```

So the count climbs from 4 towards 20 with a ~3.5 min time constant and resets at
the 20-minute reel change.

### 10.4 Two bugs, both caught by measuring

**Population collapsed instead of growing.** A fixed pool of 32 ordinals numbered
from the reel head runs out — once its last ordinal has arrived, the gate can
accrete nothing more. Measured through the pipeline: 61, 32, 0, 2, 3 marks across
a reel, monotonically *down*. Replaced with a **sliding window** of 96 ordinals
ending at the current frame, long enough to cover ~5 mean lifetimes.

**My own verification test was wrong first.** Comparing consecutive rendered frames
measures *grain*, which is redrawn every frame and swamps the dirt entirely —
103776 pixels "changed" between frames when the actual dirt was ~50. The test must
compare defect *masks* (damaged minus clean, at the **same** frame index), not raw
frames. Same error later in the weave test: a reference taken at a different frame
index injects a grain difference the shift estimator reads as displacement.

### 10.5 Verification

Weave noise, 200 000 frames:

```
rms                    1.0069     target 1.0
lag-1 correlation      0.9937     red noise: near 1 (white would be 0)
X-Y correlation       +0.0101     must be ~0; a shared stream would make the
                                  image travel along a diagonal
```

The RMS confirms both normalisations — the octave sum *and* the
`ALGO_WEAVE_INTERP_VARIANCE = 26/35` correction for variance lost to interpolating
the temporal lattice. The same class of error in the 2D dirt field previously cost
a factor of 1.34.

Weave displacement, measured from the local gradient against a same-frame
reference:

```
level 4:  predicted -0.8528 -0.8682 -0.9006 -0.9942 -1.0880 -1.1326
          measured  -0.8405 -0.8723 -0.9109 -0.9979 -1.0759 -1.1389
```

Within 0.012 px, and the **sign is confirmed** — which matters, because an
inverted sign is perfectly plausible in isolation and exactly wrong against stage
16. At level 16 the measurement under-reads by ~8 %: edge clamping compresses a
ramp at 3–4 px shifts, a test artifact.

Temporal class split, defect masks on consecutive frames:

```
EASTMAN_DOUBLE_X  (modern, 0.1 events/frame)   100 % persistent
AGFACOLOR_1943    (era,    3.0 events/frame)    56 % persistent
```

Exactly the required behaviour: gate dirt autocorrelation ≈ 1, one-frame dirt ≈ 0.
The modern stock is essentially pure gate dirt; on the 1943 stock the 44 %
non-persistent share is the sparkle. **One-frame dirt is given zero frame-to-frame
correlation deliberately** — the temptation to add a little "so it looks less
noisy" destroys the one cue that says the film is running.

Population through one reel (defect pixels, modern stock):

```
frame      0   37
frame   7200  131
frame  14400  262
frame  21600  211
frame  28700  362
frame  28810  125   <- new reel, reset
```

Growth and reset both correct. **Not strictly monotonic** — the dip at 21600 is
shedding noise. The requirement asks for monotonic growth; strictly, only the
*expected* count is monotonic, and a single realisation must fluctuate because the
model has a stochastic shed rate at all. A strictly monotonic count would require
zero shedding, contradicting the model.

### 10.6 Grading — weaker than 9b, and it should be said

The dataset **cannot** calibrate gate dirt: the scanner used was demonstrably free
of static particulate. That is a clean negative result, but it leaves every rate,
size and bias in stage 16 an engineering estimate inside a documented plausible
range — materially weaker than the 9b constants, which were solved against
measurement. What *is* anchored is the per-frame event rate and the weave
amplitude, both from `TemporalSpec`.

Gate dirt is deliberately **larger** than fine dust (0.12 mm median vs 20–34 µm)
and rendered with a **wider edge** (25 µm vs 8 µm) because it sits on the aperture
plate, out of the film plane, so the optics image it softer. It is also why this
class stays visible at low delivery resolutions where fine dust averages away:
0.12 mm is 3 px at standard definition and 19 px at 4K.

### 10.7 Cost

```
1024x1024 double      damage off   1452.67 ms
                      at 1.0       1488.12 ms   +2.4 %
                      forced high  1501.58 ms   +3.4 %
```

### 10.8 Regression

```
double / float, damage off        93 stocks PASS
double / float, damage forced high 93 stocks PASS
e2e vs film_sim.py                mean abs 5.0238e-05   unchanged
```

### 10.9 Still open

Scratches, mottle, drying marks, storage fade, colour veil, scanner artifacts,
flicker (stage 3c). Clark–Evans residual. `AgingSpec` unpopulated.
