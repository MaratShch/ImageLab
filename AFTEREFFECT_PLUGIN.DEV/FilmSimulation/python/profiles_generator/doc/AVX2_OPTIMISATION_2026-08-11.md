# AVX2 optimisation pass — 2026-08-11

> **FOURTH PASS — accumulate-mode blur (C1 completed): 332.5 → 321.3 ms
> (min of 3), cumulative 702.4 → 321.3 ms, 2.19×.** Predicted 12-15 ms,
> delivered 11.2. `AlgoMultiGaussianBlurPlaneWrap` no longer clears the
> destination or runs a per-lobe accumulate pass: the blur's own final store
> emits `w*result` for lobe 0 and `dst += w*result` thereafter, so a lobe
> result never becomes an intermediate plane. 16-22 full-plane traversals per
> three-lobe call become 3 blurs. Verified against an independent scalar
> weighted-sum-of-2D-convolutions reference (5 lobe configurations × 3 odd
> plane sizes, incl. a negligible-sigma lobe): worst 1.44e-02, which is the
> pyramid's own shape error. End-to-end sweep unchanged at 3.20e-02, no
> failures. DIR coupler 22.8 → 9.2 ms cumulative (2.47×).
>
> **THIRD PASS — `pyramidUpsample` vectorised: 358.5 → 332.5 ms (min of 3),
> cumulative 702.4 → 332.5 ms, 2.11×.** It was the last fully scalar loop in
> the blur core — two `std::floor` and two `wrapIndex` calls per
> full-resolution pixel, 2.07 M iterations per call, three calls per frame —
> and it hid because the pyramid path was only ever measured as a whole and
> looked cheap next to the direct kernel it replaced. Isolated: σ 27.8 blur
> 4.61 → 1.62 ms (2.85×), σ 5.4 6.44 → 3.40 (1.89×), σ 3.9 6.11 → 3.38
> (1.81×). Interior gathers with no wrap (xb = xa+1 by construction), wrapped
> edges left scalar. Verified against an independent 2-D reference; end-to-end
> sweep unchanged at 3.20e-02, no failures. Halation 140.9 → 105.2 ms overall.
>
> **SECOND PASS (same day) — P1 fused blur sweep + P3 negligible-sigma skip
> added: 392.7 → 358.5 ms, a further 1.10×. Cumulative 702.4 → 358.5 ms,
> 1.96×.** Error unchanged at 3.20e-02 worst over the sweep, no failures. See
> the section "P1 and P3" at the end, including the part of P1's premise that
> the measurement falsified.

**Result: an HD frame goes 702.4 ms → 392.7 ms, 1.79×, measured min-of-3 on one
machine.** Accuracy against the scalar reference *improved* at the same time:
worst-case error over the pipeline fell from `4.37e-02` to `2.71e-02`.

Three files changed. Nothing outside the `AVX2/` folder was touched, so the
scalar reference path is bit-for-bit what it was.

---

## Measured, per stage (same machine, same stock, 1920 × 1080)

| stage | before | after | speedup |
|---|---|---|---|
| 08b interimage | 164.5 | **8.7** | **18.97×** |
| 11 grain | 169.5 | **52.2** | **3.25×** |
| 14 print grain + transmit | 51.0 | **14.4** | **3.53×** |
| 08 characteristic curve | 6.8 | **4.0** | 1.69× |
| 12 dye impurity | 3.2 | 2.7 | 1.16× |
| 05 halation | 140.9 | 137.5 | 1.02× |
| 06 emulsion MTF | 34.4 | 38.2 | 0.90× |
| everything else | — | — | within noise |
| **TOTAL** | **702.4** | **392.7** | **1.79×** |

Sandbox run-to-run spread was measured at 6.7 %, so every entry between 0.93×
and 1.07× is noise and is reported as such rather than claimed.

---

## Change 1 — the characteristic curve becomes a table (`Algo_08_Sim.cpp`)

**The finding that drove it:** `AlgoStage08b_Interimage` was **still scalar in
the AVX2 build**, deliberately, on a comment claiming it "measures 1.38 ms —
two tenths of one per cent — because it is inactive on most stocks". That
measurement was taken on a stock where interimage is inactive. On a stock where
it *is* active it measured **164.5 ms, 23 % of the frame** — bigger than
halation. The comment was wrong and has been replaced.

**Why a table and not just vectorisation:** the stage re-evaluates the curve
inside a fixed-point loop — 3 channels × `iterations`, up to twelve curve
evaluations per pixel — and each one was a difference of two softplus ramps,
so **four transcendentals**. A 2048-entry table built once per channel per
frame collapses all four into one gather plus one FMA.

**It is also more accurate, which is the unusual part.** The vector path's `Exp`
is the Schraudolph bit trick at ~3 % relative error, and essentially all of the
pipeline's 4.37e-02 error came from it *inside the curve*. Table entries are
computed the scalar way in `HighPrecType` with `log1p`; only the interpolation
between them is approximate.

**Where a table would be wrong, and is not used:** any stage evaluating ONE
exponential per sample. Benchmarked, a table lookup with its gather costs
0.54 ns/sample against Schraudolph's 0.22 ns — **2.4× slower per call** — and
only wins when it replaces several calls at once. Stage 14's single `Exp` is
deliberately untouched.

**Clamping is exact, not defensive:** both ends of the curve are asymptotically
flat (below the toe both ramps vanish → `dmin`; above the shoulder both are
linear with unit slope → their difference is the constant `shoulder_x − toe_x`).
The domain covers the transition plus ten knee widths, where a softplus is
within `k·exp(−10)` of its asymptote.

Stage 8 itself uses the same table (1.69×). Its remaining `Log` cannot be
tabled — the argument is scene-linear exposure over eight decades.

## Change 2 — vector counter-based normal generator (`Algo_11_Sim.cpp`)

**Measured in isolation:** the scalar counter-RNG cost 41.6 ms per HD plane
(20.1 ns/px) and a colour stock draws **three** independent fields, so 125 ms
of the frame was one scalar loop — 18 % of the engine. It was scalar because
SplitMix64 needs a 64-bit multiply and AVX2 has none.

`_mm256_mul_epu32` plus the identity `a·b = al·bl + ((al·bh + ah·bl) << 32)`
gives the low 64 bits in three multiplies. The omitted `ah·bh` term contributes
only above bit 63, so this is **exact** — the same value the scalar multiply
produces, not an approximation.

**Preserved exactly:** the counter construction (every sample still a pure
function of seed, frame, stage and pixel ordinal — so render order, tiling,
threading and backward scrubbing are all still safe), the SplitMix64 constants
and shifts, and the Box–Muller transform with its real Gaussian tails. A
bounded generator would have been cheaper and would have passed the variance
calibration, but it would have removed the rare bright and dark specks a
developed emulsion genuinely has.

**Differs deliberately:** uniforms formed from the top 24 bits rather than 53
(the destination is a 32-bit float); `Log` and a new 6-term `cos(2πu)` series
are the vector approximations.

**Verified statistically, never by differencing two images** — though the two
turned out close enough to difference anyway:

```
                     scalar        AVX2 vector
mean               -0.000358      -0.000354     (target 0)
variance            1.000844       1.000852     (target 1)
skew               -0.003316      -0.003297     (target 0)
excess kurtosis    +0.003981      +0.004011     (target 0)
min / max          -4.663/+5.028  -4.663/+5.028 (tails intact)
determinism (repeat call)  0.000e+00 both
cross-stage correlation   +0.00229 both         (target 0)
per-sample difference vs scalar: max 3.69e-03, mean 6.64e-06
```

Full grain field including its blur: **50.99 ms → 5.61 ms, 9.1×.** One fix
serves stages 11, 13 and 14, which all call `AlgoMakeGrainField`.

## Change 3 — pyramid decimation vectorised, threshold lowered (`AlgoSeparableBlur.cpp`)

**Instrumented finding:** the blur core is **314 of ~412 Mcycles per HD frame,
76 %** — eight stages call into it. `pyramidDownsample` was scalar with a
`wrapIndex` call per sample (two branches per sample, unvectorisable because
consecutive outputs read non-overlapping strided blocks). Because of that, the
pyramid threshold had to sit at σ ≥ 6, which left the **expensive σ 4–6 band on
the direct path with a 41-tap kernel — 28.7 Mcycles per call, more than TWICE
the 12.3 Mcycles a σ = 27 pyramid call cost.**

Restructured to two passes per output row: accumulate k source rows
(contiguous, vectorised, wrap only on the row index), then decimate that
accumulation horizontally in place (front-to-back, safe because `lx ≤ lx·k`).
Same box average, same circular boundary, same result — a pure win, not a trade.
Threshold then lowered 6.0 → 3.5.

Isolated blur cost, ms per HD plane:

| σ px | before | after |
|---|---|---|
| 1.0 | 2.13 | 2.14 |
| 2.0 | 4.15 | 4.57 |
| 3.9 | 9.89 | **5.56** |
| 4.0 | 10.41 | **5.52** |
| 8.0 | 7.82 | **5.83** |
| 16.0 | 6.40 | **4.69** |
| 34.0 | 5.78 | **4.25** |

Blur total per frame: 314 → 251 Mcycles.

**Honest limitation: this did not move the frame.** Halation went 140.9 → 137.5,
inside noise. A finer histogram shows why — the frame's blur calls are
concentrated at **σ 0.2–1.1 (13 calls, 50.6 Mcycles)** and σ 2.7–2.8, i.e.
*below* the band that was fixed. Those sub-pixel blurs are two full passes over
the plane each and are **memory-bandwidth-bound** (≈11 GB/s achieved), not
arithmetic-bound, so no amount of arithmetic improvement helps them. The change
is kept because it is free and correct, and it pays on stocks whose lobes land
in σ 3.5–6.

---

## Items 4 and 5 — measured, bounded, NOT implemented

**Item 4 (box cascade for the MTF/DIR/scan stages) was withdrawn on the
measurement.** It was proposed to buy speed for ~6 % lobe shape error. The
instrumentation shows those stages' blurs are memory-bound sub-pixel kernels; a
3-pass box cascade is *six* streaming passes and would move **more** memory than
the two passes it replaces. It would have cost accuracy and bought nothing.
This is a proposal I made and the numbers killed.

**Item 5 (stage fusion) is the remaining lever and is genuinely structural.**
The measured target is precise: 13 sub-pixel blur calls doing 2 memory passes
each. Fusing horizontal and vertical into one sweep with a rolling row window
halves the traffic — estimated 25 Mcycles, ~12 ms — and fusing the pointwise
chains (02+02b, 14b+14c, 16+17) removes further full-plane touches. Estimated
total 30–50 ms. Not attempted here: it needs its own verification pass, and
shipping it half-tested inside a five-item change would have made every number
above unattributable.

**Realistic remaining ceiling: ~340–360 ms**, from fusion alone. The 230 ms
target is not reachable without either multithreading or an accuracy trade that
the measurements above do not currently justify.

---

## Verification performed

* **Correctness sweep**, 15 profiles across the 100-stock database at 384²,
  AVX2 vs scalar reference: worst error `3.20e-02` (8.17 code values at 8-bit),
  no failures, all values finite. Stochastic stages disabled on both sides —
  the generators cannot agree sample for sample by construction.
* **Pipeline error before this pass: `4.37e-02`. After: `2.71e-02`.** Improved,
  because the curve LUT displaced the Schraudolph `Exp` that produced most of it.
* **Generator verified statistically** (table above) plus determinism and
  cross-stage independence.
* One scare investigated and dismissed: profile index 98 segfaulted in *both*
  builds. Cause was a stale 93-stock `film_profiles.cpp` in the scratch tree,
  not an engine fault; re-run against the current 100-stock table passes.
* Every file compiles clean at `-O3 -mavx2 -mfma`, and with
  `ALGO_PROFILE_STAGES=0`.

## Corrections to earlier claims in this project's notes

* "08b measures 1.38 ms, inactive on most stocks" — **false on an active
  stock**: 164.5 ms. Sampling error; the comment is replaced in the source.
* "grain RNG is ~28 % of the frame" — the per-plane figure is 6 %; the correct
  figure is **18 %**, because a colour stock draws three planes. Both of my
  earlier statements were wrong in different directions.
* "the LUT is worth ~1 % of the frame" — that counted stage 8 only (14 ms). The
  curve's real cost was in 08b, and the LUT is worth **156 ms**.


---

# P1 and P3 — fused blur sweep and negligible-sigma skip (second pass)

**Result: 392.7 → 358.5 ms (min of 3), a further 30.7 ms.** Cumulative for the
day: **702.4 → 358.5 ms, 1.96×.**

| stage | items 1-3 | +P1+P3 | gain |
|---|---|---|---|
| 09 DIR coupler lateral | 24.3 | **12.1** | **2.00×** |
| 11 grain | 52.2 | **40.3** | 1.29× |
| 08b interimage | 8.7 | 7.8 | 1.12× |
| 05 halation | 137.5 | **124.8** | 1.10× |
| 09b negative defects | 8.2 | 7.5 | 1.09× |
| 06 emulsion MTF | 38.2 | 37.1 | 1.03× |
| **TOTAL** | **392.7** | **355.9** | **1.10×** |

## P3 — skip blurs below sigma 0.20

At sigma 0.20 the kernel is three taps with a side weight of exp(-12.5) =
3.7e-06 of the centre, below the quantisation of a 16-bit channel. Replaced by
the copy the result already is. Measured in isolation: **1.02 → 0.41 ms per
plane**, the largest single-call ratio in the whole blur table. Threshold set
at 0.20 rather than 0.25 deliberately - at 0.25 the side weight is 3.4e-04,
which IS representable at 12 bits, so the claim would have been arguable.

## P1 — fused single sweep, and the part of my reasoning that was wrong

The two-pass form traverses the plane four times (read src, write intermediate,
read intermediate, write dst). The fused form keeps 2*half+1 horizontally
blurred rows in a rolling window in the caller's existing scratch and traverses
twice. Applied only for half <= 8 (sigma <= 2.0), where the window is 17 rows =
130 KB and fits L2.

**I predicted this would nearly halve those calls because they were "bandwidth
bound at 11 GB/s against 24-27 GB/s available". The first measurement said
otherwise** - fusion came out SLOWER at sigma 0.3 to 1.1:

```
sigma    two-pass    fused (first attempt)
 0.30      1.29           1.38     slower
 0.60      1.79           2.02     slower
 0.90      2.13           2.59     slower
 1.10      2.54           3.03     slower
```

Two things were wrong with the premise. First, an 8.3 MB intermediate plane at
HD fits in a large L3, so the traffic I costed as DRAM was never going to DRAM -
the saving I was buying did not exist at this size. Second, the rotation
arithmetic `(base+t) mod win` sat inside the x loop, putting a compare and a
select on the critical path of every FMA.

The second cause was fixable: slot pointers and tap weights are now resolved
into tap order once per output row. After that:

```
sigma    two-pass    fused (hoisted)
 0.19      1.02           0.41    <- P3
 0.60      1.79           1.53
 1.60      4.66           4.18
 2.00      4.94           4.76
 2.80      7.40           6.99
 3.40      8.73           8.19
 5.40      6.74           6.12
27.80      4.71           4.46
 0.30      1.29           1.38    still slower
 1.10      2.54           2.84    still slower
```

So P1 is a modest win overall rather than the near-2× I projected, and it is
still a small LOSS at two sigmas. It is kept because the frame-level measurement
is unambiguous (30.7 ms including P3) and because the wide-sigma calls improved
too - the pyramid's internal low-resolution blur now takes the fused path.

The one large per-stage gain, **DIR coupler 2.00×**, was not predicted at all.
That stage's lobes evidently sit in the band where fusion pays best.

**Correctness:** the fused path was verified against an INDEPENDENT scalar 2-D
wrapped convolution - not against the two-pass path it replaces - across 8
plane sizes (odd widths and heights, from 61x37 to 1440x1094) and 9 sigmas from
0.21 to 3.4. All agree to within 2e-05. The end-to-end sweep over 15 profiles
is unchanged at worst 3.20e-02 with no failures, confirming the change is
traffic-only and not numerical.

## What remains, after both passes

Blur is still the frame: halation 124.8 + emulsion MTF 37.1 + scan MTF 25.9 +
gate weave 17.7 + corner defocus 16.6 + DIR 12.1 = **234 ms of 358.5, 65 %.**

* **P2 (cutoff 4.0 sigma -> 3.0 sigma)** is still on the table, still unbuilt,
  still awaiting a decision on whether AVX2 may diverge from the scalar
  reference systematically rather than by rounding. Estimated 5-10 ms.
* **P4 (pointwise stage fusion)** deferred deliberately: it would break the
  documented one-buffer-per-stage inspectability the project relies on for
  debugging, and it is worth only 8-12 ms.
* **P5 (pre-touch the arena)** removes ~20 ms from a one-shot render's
  measurement without removing any real work.
* **Single-thread is now close to its floor.** 24-27 GB/s is measured; two
  thirds of the frame is blur; the remaining factor of ~1.5 to reach 230 ms
  needs row-band multithreading - for which the engine is already stateless and
  reentrant - or a GPU path.


---

# Third pass — `pyramidUpsample` vectorised (C1 partial)

**358.5 → 332.5 ms, min of 3. Cumulative for the day: 702.4 → 332.5, 2.11×.**

Cumulative per stage against the original AVX2 build:

| stage | original | now | total |
|---|---|---|---|
| 08b interimage | 164.5 | **7.7** | **21.31×** |
| 11 grain | 169.5 | **33.1** | **5.13×** |
| 14 print grain + transmit | 51.0 | **13.2** | 3.88× |
| 09 DIR coupler lateral | 22.8 | **10.3** | 2.22× |
| 05 halation | 140.9 | **105.2** | 1.34× |
| 06 emulsion MTF | 34.4 | 30.5 | 1.13× |
| 10 scan MTF | 25.8 | 26.4 | within noise |
| 15 gate weave | 17.2 | 17.6 | within noise |
| **TOTAL** | **702.4** | **332.5** | **2.11×** |

## What was found

`pyramidUpsample` was **entirely scalar** — `std::floor` twice and
`wrapIndex` twice per FULL-RESOLUTION pixel. At HD that is 2.07 million
scalar iterations per pyramid call and three such calls per frame.

It hid for a reason worth recording as a method lesson: the pyramid path was
only ever timed *as a whole*, and as a whole it looked good — it had just
replaced a 41-tap direct kernel, so its measured cost was an improvement and
nobody opened it. Rule A4 (instrument the callee) catches the case where one
function serves many callers; this is the complementary case, where one
function contains a fast part and a slow part and the aggregate hides the
slow one. **Corollary to add to A4: when a path is adopted because it beat
something worse, re-profile its internals afterwards — "faster than before"
is not "fast".**

## What was done

Interior gathers, wrapped edges scalar. The interior is defined by
`x0i >= 0` and `x0i + 1 <= loW - 1`, i.e. `centre <= x <= centre + (loW-2)·k`,
and inside it `xb` is simply `xa + 1` — so no `wrapIndex` at all and the two
horizontal neighbours come from one pair of gathers per row. Gather rather
than load because consecutive output pixels do **not** read consecutive
source samples: `x0i` advances once every `k` output pixels, a repeating
staircase, which is precisely gather's use case.

Isolated blur cost, ms per HD plane:

| σ px | after P1/P3 | + upsample | gain |
|---|---|---|---|
| 3.9 | 6.11 | **3.38** | 1.81× |
| 5.4 | 6.44 | **3.40** | 1.89× |
| 27.8 | 4.61 | **1.62** | 2.85× |
| ≤ 2.8 | — | unchanged | direct path |

Verified per A3 against an independent brute-force 2-D wrapped convolution on
5 odd-sized planes × 5 sigmas in the pyramid band (worst 1.83e-02, which is
the pyramid's own shape approximation, not a defect), plus the 15-profile
end-to-end sweep: **unchanged at 3.20e-02 worst, no failures**.

## C1's remaining half — accumulate-mode blur, NOT implemented

The other half of C1 is folding the per-lobe accumulate into the blur's own
write, so `AlgoMultiGaussianBlurPlaneWrap` stops doing a destination clear
plus three extra traversals per lobe.

**Properly derived estimate, now that the traffic count is right:** per lobe
the accumulate is read-lobe + read-dst + write-dst = 3 traversals ≈ 25 MB at
HD ≈ 1.25 ms at the measured ~20 GB/s. Halation makes 9 lobe calls, so ~11 ms,
plus ~3.7 ms from the lobe-plane read disappearing on the fused path.
**Total 12–15 ms, about 4 %.**

**Deliberately stopped short of it.** It requires refactoring
`AlgoGaussianBlurPlaneWrap` — the single most-called function in the engine,
serving eight stages — into an implementation taking a weight and an
accumulate flag, with four separate write sites to convert (negligible-sigma
copy, fused store, two-pass store, upsample store). 4 % is not worth a subtle
fault in that function at the end of a long session; it is worth doing with a
fresh verification pass in front of it. Prototype unchanged either way — the
accumulating entry point would be file-local, so constraint D2 holds.

## Where the frame stands

Blur is now ~206 ms of 332.5 (62 %): halation 105, emulsion MTF 31, scan 26,
gate weave 18, corner defocus 17, DIR 10. The three items still open are
accumulate-mode (12–15 ms, above), C2 IIR (10–25 ms, **accuracy trade
awaiting a decision**) and C3 pointwise fusion (8–12 ms, **design trade,
recommended last**). All three together would land near **290–300 ms**.


---

# Fourth pass — accumulate-mode blur (C1 completed)

**332.5 → 321.3 ms, min of 3. Cumulative: 702.4 → 321.3, 2.19×.**
Estimate was 12–15 ms; actual 11.2 ms. First estimate this session that came
in slightly under rather than over.

## What changed

`AlgoGaussianBlurPlaneWrap`'s body became a file-local
`template <bool ACC> blurPlaneWrapT(..., wAcc)`. Every write site — the
negligible-sigma exit, the fused-sweep store, the two-pass vertical store and
all three upsample sites — now goes through `blurEmit<ACC>`, which is
`dst = w*res` when `ACC` is false and `dst += w*res` when true.

**Templated on the mode rather than branched on it**, so the dead half
disappears at compile time and the non-accumulating path keeps exactly the
instruction sequence it had. The weight multiply is applied in both modes and
the public entry point passes 1.0; multiplying by `1.0f` is exact in
IEEE-754, so **that path is bit-identical to before, not merely equivalent** —
which is what makes this refactor of the engine's most-called function safe to
make in one step.

`AlgoMultiGaussianBlurPlaneWrap` collapsed from

```
clear dst (1 traversal)
per lobe: blur into scratch (2 fused / 4 two-pass)
          + accumulate pass (read lobe, read dst, write dst = 3)
        = 16 traversals for 3 fused lobes, 22 if none fuse
```

to three blur calls and nothing else. Order of summation is preserved
(k = 0, 1, 2 …), so the floating-point result is the same sequence of
operations — the accumulation simply happens in the store that produced the
value. `pScratchB` is now unused; the parameter stays because the prototype is
shared with the scalar build and must not change (constraint D2).

## Cumulative per stage, against the original AVX2 build

| stage | original | now | total |
|---|---|---|---|
| 08b interimage | 164.5 | **12.2** | **13.48×** |
| 11 grain | 169.5 | **39.6** | **4.28×** |
| 14 print grain + transmit | 51.0 | **15.0** | 3.40× |
| 09 DIR coupler lateral | 22.8 | **9.2** | **2.47×** |
| 08 characteristic curve | 6.8 | 4.0 | 1.69× |
| 06 emulsion MTF | 34.4 | 30.0 | 1.15× |
| 05 halation | 140.9 | **124.9** | 1.13× |
| 06b corner defocus | 16.5 | 15.3 | 1.08× |
| **TOTAL (min of 3)** | **702.4** | **321.3** | **2.19×** |

Stages showing worse-than-1.0 in a single-run table (09b, 10, 15, 04) were not
touched by any of the four passes and are noise — the per-stage spread is
±40 % on one sample, which is exactly why the frame total is quoted min-of-3
and the per-stage column is only trusted where the change was deliberate.

## Verification

* **Multi-lobe path** against an independent scalar weighted-sum-of-2-D-
  convolutions reference: 5 lobe configurations (halation-like, wide-lobe,
  two-lobe, single-lobe, and one including a negligible-sigma lobe) × 3 odd
  plane sizes. Worst 1.44e-02 — the pyramid's shape approximation on the wide
  lobe, not a defect.
* **Single-lobe paths** unchanged: the σ 0.21–3.4 and pyramid-band harnesses
  both still pass at 2e-05 and 1.83e-02 respectively.
* **End-to-end**, 15 profiles: **3.20e-02 worst, no failures, all finite** —
  identical to before the change, confirming it is traffic-only.

## What is left

Blur is now ~205 ms of 321.3 (64 %). The two remaining items both need an
owner decision, not more engineering:

* **C2 IIR recursive Gaussian** — 10–25 ms, but makes AVX2 diverge from the
  scalar reference systematically rather than by rounding. **Undecided.**
* **C3 pointwise stage fusion** — 8–12 ms, costs per-stage buffer
  inspectability. **Recommended last, after the stage set is frozen.**

Everything that could be taken without a trade has now been taken. The
measured streaming floor for this traversal count is ~50 ms; the remaining
factor on one thread is not there.

---

# Fifth pass — resampler fidelity fix: bilinear → Catmull-Rom

**Fixed a real fidelity defect AND got 30 ms faster than before the fix.**
Frame: 297.7 → **267.1 ms** (min of 5). Cumulative for the day:
**702.4 → 267.1 ms, 2.63×.**

## The defect

Both sub-pixel resamplers in the engine — the gate weave in stage 15 and the
channel registration shift in stage 10 — used **bilinear** interpolation.
Bilinear is a low-pass filter: at a half-pixel shift its amplitude transfer at
Nyquist is 0.5. But **gate weave and misregistration are TRANSLATIONS.** A
projector moves the frame; its optics do not change. Softening the image is an
artefact of the interpolator, not a property of the film — and it
**double-counted** the digitisation loss, because the scanner's own MTF is
modelled explicitly in stage 10.

Measured on `Lady.png` through `EASTMAN_EKTACHROME_5239`, Laplacian variance of
the green record, against the Python reference (which has no weave — damage is
C++-only by design):

| | lapvar | vs source | vs Python |
|---|---|---|---|
| source | 216.5 | 1.000 | 0.484 |
| **Python reference** | 447.3 | 2.066 | 1.000 |
| C++ bilinear | 131.9 | 0.609 | **0.295** |
| C++ Catmull-Rom | 317.0 | 1.464 | **0.709** |
| C++ Catmull-Rom, damage off | 397.8 | 1.837 | **0.889** |

Note the reference is *sharper than its own source* (2.066×) — gamma 1.45
amplifies detail, which is what a reversal stock does. The bilinear build was
going the wrong way entirely.

**Sharpness recovered 2.40×.** The gap to the reference closed from 0.295 to
0.709 with weave on, and from 0.610 to 0.889 on matched feature sets.

## Why Catmull-Rom and not Lanczos

Interpolating (passes exactly through the samples, so an integer shift is the
identity), C1 continuous, four taps, and ~0.87 transfer at Nyquist for a
half-pixel shift against bilinear's 0.5. Lanczos-3 is slightly flatter at six
taps and considerably more ringing.

**Overshoot is clamped at zero only.** The cubic has negative lobes and can
undershoot at a hard edge; these are exposures and transmittances, where
negative is meaningless and would poison the downstream logarithm. Positive
overshoot is deliberately kept — it is the edge acutance the film has and that
bilinear was removing. Verified: 0 negatives, 0 non-finite values in the linear
output with damage on and off.

## The performance trap, and the way out

The first implementation evaluated the cubic basis per pixel and wrapped each of
the sixteen tap indices with a modulo. Result:

```
                     bilinear   scalar Catmull-Rom   vectorised Catmull-Rom
stage 10 scan+misreg   26.3 ms         88.5 ms              4.05 ms
stage 15 gate weave    13.0 ms         70.3 ms              2.64 ms
HD frame              297.7 ms        422.4 ms            267.1 ms
```

A **125 ms regression** — half of a day's optimisation spent to buy sharpness.
The way out was rule A1: classify before optimising. **The displacement is a
frame constant**, so all sixteen weights are frame constants too, and what
remains is a fixed 4×4 separable convolution at an integer offset. The four
horizontal taps are then *contiguous* — four overlapping unaligned loads, not a
gather — and the interior of the plane needs no wrapping at all.

So the vectorised cubic is **6.5× faster than the bilinear it replaced**, because
the bilinear versions were themselves scalar per-pixel loops doing modulo
arithmetic. Better physics and less time.

## Applied to BOTH flavours

Scalar and AVX2 both changed. This is a **physics fix, not a vector-path
optimisation**, so leaving it out of the scalar path would knowingly keep the
accuracy reference wrong. The scalar copies are the straightforward
per-pixel form (correctness over speed, as befits a reference); only the AVX2
copies carry the vectorised frame-constant version.

## Verification

* End-to-end sweep, 15 profiles: **3.40e-02 worst (8.67 DN), no failures, all
  finite** — up marginally from 3.20e-02, expected, because a cubic amplifies
  edge contrast and therefore amplifies the AVX2-vs-scalar difference at edges.
* Tone and colour unmoved: mean luminance 100.42 → 100.38, saturation 0.2962 →
  0.2985 against the reference's 0.2946.
* C++ vs Python direct difference: mean 3.83 DN, p95 11.78 (was 3.88 / 12.23) —
  slightly *closer* to the reference than the bilinear build.
