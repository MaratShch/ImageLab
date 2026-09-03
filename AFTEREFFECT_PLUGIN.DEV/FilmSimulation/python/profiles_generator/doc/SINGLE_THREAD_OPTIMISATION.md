# Single-Thread Optimisation Proposals — CPU Scalar/AVX2 Engine

> ## UPDATE 2026-08-28 — RE-MEASURED, BOTH PATHS, AND THREE PROPOSAL STATUSES CORRECTED
>
> **Every figure below this banner that predates 2026-08-28 is superseded.** The
> cost landscape moved twice this session: the wide-sigma blur now computes the
> lobes it was silently collapsing to a 37 px box, and the characteristic curve
> is tabulated in the scalar path.
>
> ### Proposal statuses, corrected
>
> | Proposal | Status as this document had it | Actual, 2026-08-28 |
> |---|---|---|
> | **O1** wide-sigma blur | AVX2 only; scalar successor withdrawn | **Implemented in BOTH paths**, and converged onto one shared planner and one shared set of cell weights in `AlgoSeparableBlur.hpp`. |
> | **O2** curve via LUT | AVX2 only | **Implemented in BOTH paths**, from one shared header `AlgoCurveLut.hpp`, serving stages 08, 08b and 13. |
> | **O4** no-op stage elision | proposed | **Redefined.** It cannot be done alone: the arena ping-pongs two physical triples across 25 logical stages, so forwarding a pointer past an elided stage aliases the next stage's source and destination. It is now part of the driver restructure, not a standalone step. |
> | `AlgoTypes` contradiction | flagged in the 2026-08-25 review banner | **Resolved.** Selector + `AlgoTypes_Scalar.hpp` + `AVX2/AlgoTypes.hpp`, one project-wide `ALGO_TARGET_AVX2`. |
>
> ### Per-stage, 1024 x 1024, best of three, milliseconds
>
> Four reference stocks, both paths. `sc` = scalar (`AlgoType = double`),
> `av` = AVX2 (`AlgoType = float`).
>
| stage | APX 25 sc | APX 25 av | 800T sc | 800T av | 5219 sc | 5219 av | 1939 sc | 1939 av |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 02 relative exposure | 4.0 | 2.8 | 4.0 | 2.8 | 4.2 | 2.8 | 4.1 | 2.8 |
| 02b taking filters | 5.8 | 3.2 | 5.8 | 3.2 | 6.0 | 3.1 | 6.0 | 3.0 |
| 03 stock colour balance | 5.7 | 2.9 | 5.9 | 3.0 | 6.0 | 2.9 | 5.9 | 2.7 |
| 03b veiling flare | 5.8 | 2.8 | 5.7 | 2.9 | 5.8 | 2.5 | 69.2 | 25.6 |
| 03c flicker  STUB | 6.1 | 2.8 | 5.8 | 2.9 | 5.7 | 2.4 | 5.9 | 3.2 |
| 04 coating + vignette | 16.6 | 12.4 | 9.5 | 5.5 | 9.3 | 4.8 | 18.8 | 13.7 |
| 05 halation | 5.7 | 3.0 | 668.2 | 177.1 | 1559.8 | 295.2 | 678.0 | 175.2 |
| 06 emulsion MTF | 169.4 | 14.5 | 253.1 | 24.8 | 256.7 | 23.4 | 301.0 | 36.8 |
| 06b corner defocus | 38.0 | 40.8 | 34.8 | 41.4 | 36.2 | 40.8 | 35.5 | 41.9 |
| 07 emulsion record | 5.3 | 2.5 | 5.7 | 3.1 | 6.3 | 3.2 | 5.5 | 2.9 |
| (anchor solve) | 0.2 | 0.3 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| 08 characteristic curve | 54.1 | 11.3 | 53.5 | 11.5 | 54.4 | 11.1 | 52.9 | 11.7 |
| 08b interimage | 5.8 | 3.1 | 31.9 | 15.6 | 32.0 | 16.1 | 6.2 | 3.4 |
| 09 DIR coupler | 5.7 | 3.0 | 149.7 | 27.1 | 149.4 | 26.4 | 5.9 | 3.3 |
| 09b neg defects STUB | 21.7 | 16.5 | 21.4 | 16.7 | 22.2 | 16.5 | 21.9 | 17.3 |
| 10 scan MTF | 10.0 | 5.6 | 130.4 | 8.4 | 133.3 | 8.3 | 10.5 | 5.6 |
| 10b edge fog | 6.2 | 2.8 | 5.8 | 3.9 | 6.2 | 2.8 | 5.7 | 2.9 |
| 11 grain | 91.1 | 15.1 | 289.9 | 43.7 | 284.7 | 39.7 | 128.9 | 21.4 |
| 12 dye impurity | 6.1 | 2.8 | 10.0 | 4.3 | 9.4 | 3.9 | 6.1 | 3.1 |
| 13 dupe + print | 27.5 | 13.8 | 33.9 | 13.6 | 35.1 | 13.7 | 27.2 | 14.0 |
| 14 transmittance | 102.8 | 16.0 | 101.4 | 15.8 | 105.4 | 16.1 | 105.8 | 15.9 |
| 14b reseau recon | 5.8 | 3.0 | 7.9 | 4.6 | 8.6 | 4.5 | 6.3 | 2.7 |
| 14c silver tone | 5.7 | 2.6 | 10.0 | 3.3 | 5.7 | 2.7 | 6.0 | 2.4 |
| 15/16 STUB + 17 clamp | 160.9 | 10.8 | 163.2 | 13.8 | 161.3 | 10.9 | 159.0 | 9.3 |
| **TOTAL 02..17** | **765.8** | **194.4** | **2007.8** | **448.9** | **2903.8** | **554.0** | **1672.5** | **421.0** |
>
> ### Reading this table
>
> **Do not compare it against the 2026-08-11 figures further down.** Those were
> taken on different hardware in a different container, and this one has roughly
> ±10 per cent run-to-run variance of its own. A cross-session delta here would
> be measuring the machine.
>
> **Halation dominates the scalar frame** — 1559.8 ms of 2903.8 on
> `KODAK_VISION3_500T_5219`, 668.2 of 2007.8 on `CINESTILL_800T`. That is the
> honest cost of computing wide lobes correctly. The previous code was faster
> because it was wrong: past sigma 16 it collapsed every lobe onto the same
> ~37 px box regardless of the physics requested. Speed bought by not doing the
> work is not speed.
>
> **Stages 15/16/17 remain the flat scalar cost** — about 160 ms on every stock,
> against 9-14 ms in the vector path. That is the largest single scalar/AVX2
> ratio in the table and the clearest remaining target.
>
> **06b corner defocus is the one stage where AVX2 is SLOWER than scalar**
> (about 41 ms against 35-38). Worth an investigation; it is not what one would
> predict and it has not been explained.
>
> ### What changed in cost this session, measured before and after
>
> | | before | after |
> |---|---|---|
> | scalar stage 08 | 223.25 ms | 53.24 ms |
> | scalar stage 08b (`CINESTILL_800T`) | 172.46 ms | 30.98 ms |
> | scalar stage 13 | 167.20 ms | 26.82 ms |
> | scalar stage 14 | 166.39 ms | 111.97 ms |
>
> Stage 14 is the cheapest win in the file: `std::pow(10.0, -d)` became
> `std::exp2(-d * log2(10))`, which is **31-33 per cent faster and produces
> BIT-IDENTICAL output on all 161 stocks**. It is not an approximation — the
> base is a compile-time constant, so the general two-argument `pow` was doing
> edge-case work that a single `exp2` does not need.
>
> ### One optimisation was measured and REJECTED
>
> Tabulating the print curve at stage 13 in the **AVX2** path. It was
> implemented and reverted: the fast approximate softplus is 15-23 per cent
> faster there, because a gather only pays where it replaces several
> transcendentals at once. Stage 08 re-evaluates the curve through the
> interimage fixed point and wins; stage 13 evaluates once per sample and loses.
> The resulting cross-path difference, 2.4e-03 in plane mean, is below one part
> in 255. Recorded at the site as a deliberate mode difference.
>
> Full detail: `doc/RESULT_2026-08-28_engine_convergence.md`.


> **UPDATE 2026-08-11 — AVX2 optimisation pass executed: 702.4 → 392.7 ms at
> HD, 1.79×, with accuracy IMPROVING from 4.37e-02 to 2.71e-02.** Three files
> changed (`AVX2/Algo_08_Sim.cpp`, `AVX2/Algo_11_Sim.cpp`,
> `AVX2/AlgoSeparableBlur.cpp`). Headlines: interimage 164.5 → 8.7 ms (18.97×,
> it was still scalar), grain 169.5 → 52.2 ms (3.25×, vector counter-RNG),
> print grain 51.0 → 14.4 ms. **Proposal O2 (curve via LUT) is now implemented
> and its saving was ~20× larger than this document estimated**, because the
> curve's real cost was in 8b's fixed-point loop, not in stage 8.
> **Proposal O1's box-cascade successor was WITHDRAWN on measurement** — the
> frame's blur calls are memory-bound sub-pixel kernels, so a 6-pass cascade
> would move more memory than the 2 passes it replaces. Full detail, including
> what was measured and what was not attempted, in
> `AVX2_OPTIMISATION_2026-08-11.md`.
>
> **Second pass, same day: fused blur sweep + negligible-sigma skip take it to
> 358.5 ms — cumulative 702.4 → 358.5, 1.96×.** DIR coupler 2.00×, grain a
> further 1.29×, halation 1.10×. Note for the record that the fused sweep's
> stated premise (DRAM bandwidth) was FALSIFIED by measurement — an 8.3 MB
> intermediate plane fits L3 — and it survives on a smaller, differently-caused
> gain. Blur is still 65 % of the frame; single thread is near its floor.

> **UPDATE — `Algorithm_Main` now measures itself, per stage.** Every stage boundary
> in `AlgorithmMain.cpp` is timestamped with `RDTSC()` from `Common.hpp`, and a table
> of 27 rows — setup, the anchor solve and all 25 stage calls — is written to
> **stderr** as the function returns. Columns: Mcycles, milliseconds, per cent of
> frame, nanoseconds per pixel.
>
> - **Switch:** `ALGO_PROFILE_STAGES`, defaulted to `1` at the top of
>   `AlgorithmMain.cpp`. Set to `0` — or pass `-DALGO_PROFILE_STAGES=0`, which wins —
>   and every macro expands to nothing: no timestamps, no stack, no `<cstdio>`.
> - **Cost when enabled:** 28 `RDTSC` instructions and two `steady_clock` reads per
>   frame, together under 1 µs against a frame measured in hundreds of milliseconds.
> - **Milliseconds are derived, not assumed.** `RDTSC` gives no rate, so the frame is
>   bracketed by one `steady_clock` pair and the cycles-to-ms factor comes from the
>   frame itself. Correct on any host; relies on an invariant TSC, which every
>   targeted x86 part has.
> - **No shared state.** The timer is a stack local, so the engine's reentrancy
>   guarantee is intact and concurrent instances do not interfere.
> - **Time attributes to the caller.** Eight stages spend most of their time inside
>   `AlgoSeparableBlur.cpp`; that time appears against the stage that requested the
>   blur.
> - **Read a sub-0.05 ms row as "too cheap to measure".** `RDTSC` does not serialise,
>   so boundary placement carries tens of cycles of skew.
>
> First measurement it produced, AVX2, 1920 × 1920, `EASTMAN_5294_1983`: halation
> 255.6 ms (30.7 %), **interimage 304.4 ms (36.5 %)**, emulsion MTF 69.2 ms (8.3 %),
> DIR coupler 43.9 ms (5.3 %), gate weave 30.6 ms (3.7 %). The interimage figure is
> new information — the fixed-point iteration at stage 8b is the largest single cost
> on a stock with active interimage, ahead of halation, and none of the proposals
> below address it.

> **UPDATE — stage 9b is no longer a stub.** Fine dust, coarse debris and fibres now
> render there, so every "09b is a pure copy / always memcpy / stub" statement below
> is superseded. 9b remains pointwise-plus-scatter and is still foldable into stage
> 10's read pass, but it is no longer a no-op elision candidate and no longer free.
> Measured cost at 1024x1024 double: unmeasurable at default levels, +2 % at levels
> far past plausible. See `DEFECT_LAYER.md`.



**Status:** proposals only. No C++ or Python source has been modified.
**Scope:** single thread, CPU only. No GPU, no multithreading.
**Measured on:** Intel i7-14700, GCC `-O2`, `AlgoType = double`, 1024 × 1024,
median of 3–7 runs, stock `AGFACOLOR_NEG_TYPE_B_1943` unless stated.

---

## 1. Executive summary

Four proposals are pure wins with no design cost. Two are architectural and trade
something real. One is not worth doing.

| # | Proposal | 1024² saving | Cost to the design |
|---|---|---|---|
| **O1** | Wide-lobe blur at reduced resolution | **−725 ms** | none |
| **O2** | Characteristic curve via per-frame LUT | **−200 ms** | none |
| **O3** | Grain generator: paired Box–Muller | **−77 ms** | none |
| **O4** | No-op stage elision by pointer forwarding | **−16 ms** | none |
| **O5** | AVX2 intrinsics, selectively | −100 to −150 ms | maintenance |
| **O6** | Stage fusion into shared memory passes | −60 to −90 ms | loses per-stage inspection |
| **O7** | Recursive IIR Gaussian | −80 ms | perturbs every blur's numerics |

O1 through O4 together take **1408 ms → ~390 ms** and touch four files. Everything
after that fights a memory-bandwidth floor that has been measured, not assumed.

**The honest ceiling.** Even with all seven applied, single-threaded:

- **HD: ~200–250 ms.** The 50 ms figure is the *floor with the arithmetic deleted*.
- **4K: ~850–1000 ms.** Not reachable at 100 ms on one thread, by a wide margin.

---

## 2. Measured baseline

### 2.1 Per-stage cost, 1024 × 1024, `AlgoType = double`

```
02  relative exposure         1.59 ms    0.1 %
02b taking filters            2.14 ms    0.2 %     <- identity on 92/93 stocks
03  stock colour balance      1.98 ms    0.1 %
03b veiling flare             2.08 ms    0.1 %     <- 529 ms on the 11 flare stocks
03c flicker  STUB             2.06 ms    0.1 %     <- pure copy, always
04  coating + vignette        8.14 ms    0.6 %
05  halation                745.24 ms   52.9 %     <-- O1
06  emulsion MTF            119.85 ms    8.5 %     <-- O5, O7
06b corner defocus           15.44 ms    1.1 %
07  emulsion record           1.98 ms    0.1 %
    (anchor solve)            0.10 ms    0.0 %
08  characteristic curve    111.17 ms    7.9 %     <-- O2
08b interimage                2.03 ms    0.1 %     <- 80 ms on CINESTILL_800T
09  DIR coupler              41.92 ms    3.0 %     <-- O5
09b neg defects  STUB         2.22 ms    0.2 %     <- pure copy, always
10  scan MTF                 15.27 ms    1.1 %
10b edge fog                  1.94 ms    0.1 %     <- pure copy on 86/93 stocks
11  grain                   166.72 ms   11.8 %     <-- O3, O5
12  dye impurity              3.28 ms    0.2 %
13  dupe + print             83.91 ms    6.0 %     <-- O2
14  transmittance            68.74 ms    4.9 %     <-- O2
14b reseau recon              3.12 ms    0.2 %
14c silver tone               1.90 ms    0.1 %     <- pure copy on 86/93 stocks
15/16 STUB + 17 clamp         6.91 ms    0.5 %
                          ----------
TOTAL 02..17               1407.97 ms
```

Six stages are 92% of the frame. Worst case over the database is **1840 ms**
(`SOVIET_PANCHROM_1939`, where stage 3b activates).

### 2.2 The bandwidth floor — measured, not derived

The same 24 stages, each doing nothing but `AlgoCopyImage`. Zero arithmetic. No
optimisation can go below this without changing how much memory is touched.

```
                    time      traffic    achieved
1024²  float      23.9 ms      576 MB    25.2 GB/s
720p   float      19.9 ms      506 MB    26.7 GB/s
HD     float      48.2 ms     1139 MB    24.8 GB/s
QHD    float      88.1 ms     2025 MB    24.1 GB/s
4K UHD float     ~193 ms     ~4600 MB    (linear extrapolation)

1024²  double    49.5 ms     1152 MB    24.4 GB/s
HD     double   103.1 ms     2278 MB    23.2 GB/s
```

Linear across four sizes at a steady 24–27 GB/s. **This is the number that decides
what is achievable**, and it is why several otherwise attractive optimisations
return nothing.

### 2.3 Access pattern per stage — what can and cannot fuse

Fusion requires consecutive stages that read only the pixel they write
(*pointwise*). A stage that reads neighbours needs the whole plane, or at least a
row band, before it can produce output.

| stage | pattern | fusible with neighbours? |
|---|---|---|
| 02, 02b, 03, 03c | **pointwise** | yes — one run of four |
| 03b veiling flare | neighbourhood (3-lobe blur) | no |
| 04 coating + vignette | **pointwise** given the field | yes, into 05's source build |
| 05 halation | neighbourhood (3-lobe blur) | no |
| 06 emulsion MTF | neighbourhood (3-lobe blur) | no |
| 06b corner defocus | neighbourhood, **5-tap only** | yes, via a 5-row band |
| 07, 08, 08b | **pointwise** | yes — one run of three |
| 09 DIR coupler | neighbourhood | no |
| 09b defects | **pointwise** (stub) | yes |
| 10 scan MTF + shift | neighbourhood | no |
| 10b edge fog | **pointwise** | yes, into 11's add |
| 11 grain | field build is neighbourhood; **the add is pointwise** | partly |
| 12 dye impurity | **pointwise** | yes |
| 13 dupe + print | neighbourhood per generation | no |
| 14 transmittance | **pointwise** (after the grain add) | yes |
| 14b reseau | neighbourhood — **but only 1/93 stocks** | yes for 92 stocks |
| 14c, 15, 16, 17 | **pointwise** | yes — one run of four |

### 2.4 How often each optional stage does real work

This is the basis for proposal O4 and it is more lopsided than expected.

```
02b taking filters      1 / 93     <- identity, i.e. memcpy, on 92 stocks
03b veiling flare      11 / 93
05  halation           63 / 93
06b corner defocus     93 / 93
08b interimage         51 / 93
09  DIR coupler        54 / 93
10b edge fog            7 / 93
12  dye impurity       52 / 93
14b reseau              1 / 93
14c silver tone         7 / 93
03c / 09b / 15 / 16     0 / 93     <- stubs, always memcpy
```

Also: **22 of 93 stocks are reversal**, and those skip the whole of stage 13.
**36 are monochrome**, and those collapse three grain fields to one at stage 11.

---

## 3. Proposal O1 — wide-lobe blur at reduced resolution

**Target:** stage 05 (745 ms), stage 03b (529 ms on the 11 stocks that use it).
Together up to **1274 ms**, which is 69% of a worst-case frame.

**Cause.** Halation's widest lobe is 440 µm, which at 41 px/mm is σ ≈ 18 px.
Truncated at 4σ that is a 145-tap kernel, two separable passes, three channels —
about 870 taps per pixel. Veiling flare is worse: its widest lobe is 20 000 µm,
σ ≈ 820 px. Both are correct and both are being done as direct convolution.

**Proposal.** For any lobe whose σ exceeds a threshold (≈ 4 px), decimate the
plane by 2^k until σ/2^k lands near 2 px, blur there, and bilinearly upsample.
Work falls by 4^k in the blur and the decimate/upsample passes are cheap.

Why this loses nothing: a halo and a veiling haze are by definition
low-frequency. The information a σ = 820 px kernel produces has no content above
the decimated Nyquist. **Stage 04 already uses exactly this trick** for the
coating field — the low-resolution grid clamped to 24…192 samples, then
`AlgoBilinearUpsample`. This extends a pattern already in the codebase rather
than introducing one.

**Expected:** 05 → ~20 ms, 03b → ~15 ms. Saving **~725 ms** typical,
**~1240 ms** worst case.

**Verification required.** Keep the direct path as reference. Compare across
resolutions and across all 93 stocks; report max and mean error. Accept only if
the error sits well under the float32 precision of the profile data (~1e-7
relative). The pyramid is *not* bit-identical and must not be presented as such.

---

## 4. Proposal O2 — characteristic curve via per-frame lookup table

**Target:** stages 08 (111 ms), 13 (84 ms), 14 (69 ms), 08b (2–80 ms).
Together **264–340 ms**.

**Cause.** `AlgoSoftplus` is `k · log1p(exp(x/k))` — one `exp` and one `log1p`
per call, two calls per pixel per channel, so **~15 transcendental calls per
pixel**. Stage 14 adds a `pow(10, −d)` per pixel per channel.

**Proposal.** The curve is a one-dimensional, monotonic, smooth function of log
exposure with six frame-constant parameters. Build a table once per frame per
channel — a few thousand entries over the usable logE range — and make the pixel
loop a clamp, an index and a linear interpolation.

Table error is bounded by the second derivative times the square of the step, and
the curve's curvature is modest. A 4096-entry table over 8 decades gives an
interpolation error far below the float32 precision of `dmin`, `gamma`, `toe_x`
and the rest — i.e. below the accuracy of the input data. **The monotonicity
guarantee survives**, because a linear interpolation between samples of a
monotonic function is monotonic.

Stage 14's `pow(10, −d)` needs no table at all: `exp2f(−d · log2(10))` is a
single hardware-assisted call and is much cheaper than a general `pow`.

**Expected:** 08 → ~20 ms, 13 → ~30 ms, 14 → ~15 ms. Saving **~200 ms**.

**Note on stage 8b.** It re-evaluates the curve once per fixed-point iteration.
Nothing in the current database exceeds one iteration, but the field allows more,
and on CINESTILL_800T the stage already costs 80 ms. The LUT makes the iteration
count almost free, which removes a latent cliff rather than just a cost.

---

## 5. Proposal O3 — paired Box–Muller in the grain generator

**Target:** stage 11 (167 ms typical; 141 ms on CINESTILL, 58 ms on monochrome).

**Cause.** `AlgoRngNormal` draws two uniforms, then computes
`sqrt(−2·log(u1)) · cos(2π·u2)` — and **discards the `sin` half of the
transform**. Every normal costs a `log`, a `sqrt` and a `cos`. A tripack needs
three independent fields, so that is 3 M normals per frame at 1024².

**Proposal.** Box–Muller naturally produces *two* independent normals per
transcendental group. Return both and consume both. The transcendental cost per
sample halves. The counter-based determinism guarantee is preserved: the pair is
still a pure function of `(seed, frameIndex, stage, ordinal)` provided the
ordinal indexes *pairs* rather than samples.

**Expected:** 167 → ~90 ms. Saving **~77 ms**.

**Verification note.** This changes which numbers land at which pixels, so grain
will not match the current build sample-for-sample. It must be validated
*statistically* — mean ≈ 0, variance matching the RMS granularity target, and the
spectral shape unchanged — not by pixel comparison. Say so up front or it will
look like a regression.

---

## 6. Proposal O4 — no-op stage elision by pointer forwarding

**Target:** ~16 ms at 1024², **~133 ms at 4K**. Free.

**Cause.** The retained-buffer policy makes every inactive stage perform a full
`AlgoCopyImage` so its own buffer holds valid data. For the default stock that
is **eight of the twenty-four stages**:

```
02b taking filters   identity matrix
03b veiling flare    default_flare = 0
03c flicker          stub
09b neg defects      stub
10b edge fog         no edge fog on 35 mm
14c silver tone      not monochrome
15  gate weave       stub
16  gate defects     stub
```

**One third of the pipeline is `memcpy`,** and at 4K that is ~133 ms of pure
memory traffic producing nothing.

**Proposal.** Let `Algorithm_Main` hold a *current triple* of pointers rather
than hard-wiring each stage's source. When a stage's activity test fails, forward
the pointers instead of calling it:

```
if (stageIsActive) { call stage; current = stageBuffer; }
else               { /* current unchanged — no copy, no call */ }
```

The activity tests already exist inside each stage; they would need exposing as
small predicate functions so the caller can ask before dispatching.

**What this costs.** An elided stage's buffer holds stale data, so
"dump S10b and look at it" stops being universally valid. The debugging property
is preserved *for stages that ran*, which is the case that matters — an inactive
stage's output is by definition identical to its input. A debug flag could force
the copies back on.

**Why this is the best value in the list:** no numerical change whatsoever. The
output is bit-identical. There is nothing to verify beyond that.

---

## 7. Proposal O5 — AVX2 intrinsics, selectively

**Not uniform, and the wins do not stack with O1–O3.** Applying AVX2 to a stage
already fixed algorithmically returns much less than the raw width suggests.

| stage class | state after O1–O3 | realistic AVX2 gain | worth it? |
|---|---|---|---|
| pointwise (02, 02b, 03, 07, 12, 14c, 17) | bandwidth-bound at ~1.9 ms | **1.0–1.2×** | **no** |
| small-kernel blurs (05, 06b, 09, 10) | bandwidth-bound | 1.3–1.8× | marginal |
| emulsion MTF (06) | still compute-bound, 120 ms | **3–4×** | **yes** |
| curve stages (08, 13, 14) after LUT | gather-bound | 1.5–2× | marginal |
| grain RNG (11) | see below | **~1.3×** | **no, as written** |

**The pointwise stages will not improve.** Measured: 1.9 ms each, moving 48 MB,
which is 25 GB/s — already at the roofline. `-O3 -march=native -ffast-math`
changed them by nothing. Hand-vectorising them is effort spent for no gain.

**Stage 06 is the one clear AVX2 target.** Nine separable passes per frame,
compute-bound on kernel taps, and the taps are a simple multiply-accumulate over
contiguous floats. This is the textbook case: `_mm256_fmadd_ps` over the tap
loop, with the wrap boundary handled by peeling the first and last `radius`
columns out of the vector body. Expect 120 → ~35 ms.

**SplitMix64 does not vectorise on AVX2, and this matters.** The finaliser needs
a 64 × 64 → 64 multiply. AVX2 provides only `vpmuludq` (32 × 32 → 64).
Synthesising the 64-bit product takes three multiplies plus shifts and adds, so
4-wide at ~3× the operations is barely better than scalar. `vpmullq` is AVX-512
only.

If grain throughput matters after O3, the fix is to **change generator, not to
vectorise this one**: Philox 4×32 or a 32-bit mixing function is counter-based —
so the determinism contract is unchanged — and vectorises cleanly 8-wide. That is
a design decision with a validation cost (grain becomes different again), so it
should be taken deliberately and only if measurement justifies it.

**A caution on `-ffast-math`.** It gave a real gain in the earlier stage-4
measurements, but it relicenses the compiler to reassociate floating point. The
engine has several places where the *order* of operations is load-bearing —
the energy-conserving `blur(above) − above` in halation, the zero-mean
subtraction in the grain field, the difference-of-softplus in the curve. Enabling
it globally is not safe here. Per-file, with verification against the reference,
is defensible.

---

## 8. Proposal O6 — stage fusion into shared memory passes

**The architectural one.** Biggest remaining win after O1–O4, and the only one
that moves the *floor* rather than the compute.

From §2.3, the fusible runs are:

| group | stages | passes now | passes fused |
|---|---|---|---|
| G1 | 02, 02b, 03, 03c | 4 | 1 |
| G2 | 04 → 05's source build | 2 | 1 |
| G3 | 06b, 07, 08, 08b (5-row band for 06b) | 4 | 1 |
| G4 | 09b, then 10 | — | fold 09b into 10's read |
| G5 | 10b, 11's add | 2 | 1 |
| G6 | 12, then 13's first blur | — | fold 12 into 13's read |
| G7 | 14, 14c, 15, 16, 17 (92/93 stocks) | 5 | 1 |

Group G3 is the interesting one: **corner defocus is only a 5-tap kernel**, so it
does not need the whole plane — a rolling 5-row band suffices, which lets it join
the pointwise run that follows it.

Traffic falls from **144 plane-touches to roughly 66**, a factor of 2.2. That
takes the floor from 193 ms to **~88 ms at 4K**, and 48 ms to **~22 ms at HD**.
It also cuts the arena from 2.90 GB to roughly 1 GB at 4K, which is a separate
and serious benefit for an AE plugin.

**What it costs.** The retained-buffer-per-stage property, which was chosen
deliberately so any stage's output could be dumped and inspected without
re-running the chain. That property is what made it possible to validate this
engine against the Python reference stage by stage.

**Therefore: do this last.** The numerics are now settled — stages 7 and 8 agree
with the reference to ~2e-7, the whole chain to ~1e-6 mean at adequate
resolution. Fusion is safe to do *because* that validation is already banked. Do
it before the validation is complete and there is no way to localise a
discrepancy.

**Suggested hedge:** keep the unfused path behind a compile-time switch, so a
future discrepancy can be bisected against the per-stage version.

---

## 9. Proposal O7 — recursive IIR Gaussian

Replaces the separable FIR blur with a Deriche or Young–van Vliet recursion:
**O(1) per pixel regardless of σ**, so stage 06's 120 ms drops sharply and O1
becomes unnecessary for the wide lobes.

It would also fix a known accuracy gap. A sampled Gaussian kernel and an analytic
Gaussian *transfer* stop agreeing once σ falls below a pixel: at σ = 0.311 px —
which is 35 mm academy at 1024 px wide — the sampled kernel passes 97.7% at
Nyquist while the true transfer passes 62%. Measured deviation from the reference
is 5.4e-2 there, converging to 3.9e-6 by σ = 0.206 px at higher resolution.

**Why it is listed last despite being the most powerful.** It changes the
numerics of *every* blur in the engine simultaneously — stages 03b, 05, 06, 09,
10, 11, 13, 14b — so a regression cannot be localised. And an IIR filter's
boundary handling is materially harder than an FIR's; the engine relies on wrap
boundaries to match the reference's circular convolution, and getting that right
in a recursion requires care.

Recommended only after O1–O6 are in and validated, and only with the FIR path
retained as reference.

---

## 10. Smaller items

Each small, each free, none individually worth a release.

1. **`std::pow(2.0, x)` for frame constants.** Stage 05's threshold and stage 02's
   exposure gain call `pow` once per frame. `exp2` is cheaper and exact for
   integral arguments. Negligible, but it is the correct function.
2. **Stage 09 rebuilds its Gaussian kernel three times** — once per channel for
   the edge term, with identical σ. Build once, reuse.
3. **`pScrBlurB` is unused in stage 09.** Documented as reserved for a future
   second lobe. Harmless; noted so it is not mistaken for an oversight.
4. **`AlgoPlaneMean` is a separate full pass** over the grain field. It could be
   accumulated during the field's final scaling pass, saving one read of a plane
   per channel per frame.
5. **Stage 06's three lobes each re-read the source plane.** If the three kernels
   were applied in one pass over each row — accumulating three running sums —
   the source would be read once instead of three times. Saves ~2/3 of that
   stage's read traffic; interacts with O5, so do them together.
6. **`HighPrecType` inside pixel loops.** 22 sites across the engine use double
   per pixel where float would do (stage 04's vignette term, 06b's radial blend,
   10's shift weights, 14's transmittance). On CPU this costs bandwidth and
   halves SIMD width. Worth converting *with verification*, since a few of them
   were chosen deliberately.

---

## 11. Projected outcome

**Measured** rows are measured. **Projected** rows carry real uncertainty — the
AVX2 and fusion figures are estimates, and I would not defend them to better than
±30% until they exist.

| step | 1024² | HD | 4K UHD | basis |
|---|---|---|---|---|
| current, double | **1408 ms** | ~2900 ms | ~11 700 ms | measured / scaled |
| + O1 pyramid blur | ~683 ms | ~1400 ms | ~5700 ms | projected |
| + O2 curve LUT | ~484 ms | ~1000 ms | ~4000 ms | projected |
| + O3 grain RNG | ~407 ms | ~840 ms | ~3400 ms | projected |
| + O4 no-op elision | ~391 ms | ~810 ms | ~3200 ms | projected |
| + `AlgoType = float` | ~290 ms | ~600 ms | ~2400 ms | projected |
| + O5 AVX2 (stage 06 etc.) | ~165 ms | ~340 ms | ~1400 ms | projected |
| + O6 fusion | ~110 ms | ~230 ms | ~920 ms | projected |
| **bandwidth floor, fused** | ~11 ms | ~22 ms | **~88 ms** | derived from measurement |

### What this means for the original question

- **HD at 100 ms single-threaded: plausible but tight.** It needs essentially the
  whole list. 200–250 ms is the confident answer.
- **4K at 100 ms single-threaded: no.** The fused floor alone is 88 ms with the
  arithmetic removed. Reaching it needs threading — the pipeline is tile-parallel
  apart from the blur wrap boundaries and the anchor solve, and an i7-14700 has
  20 cores.

---

## 12. Recommended order, and why

1. **O4** — free, bit-identical output, nothing to verify. Do it first to
   establish the harness discipline on a change that cannot break anything.
2. **O1** — largest single win, self-contained, extends a pattern already in
   stage 04.
3. **O2** — second largest, and it removes a latent cliff in stage 8b as well as
   a cost.
4. **O3** — smaller, but it also halves the cost of the print and dupe grain.
5. **Re-measure.** Everything after this point is projection; the four above will
   have changed which stages matter.
6. **`AlgoType = float`** — already verified to pass every test; the accuracy
   question was settled earlier.
7. **O5** on stage 06 only, guided by fresh measurement rather than by this
   document.
8. **O6** last, with the unfused path retained behind a switch.
9. **O7** only if 06 is still limiting after O5.

## 13. Verification requirements — non-negotiable for O1, O3, O5, O6, O7

Each of those changes the numbers. The existing harnesses must be used, not
replaced:

- `test_full.cpp` — all 93 stocks, finite, in range, `Dst_*` written.
- `test_chain_02_08.cpp` — chain output bit-identical to hand-called stages.
- `e2e.cpp` + `e2e.py` — the whole chain against `film_sim.simulate()`, with the
  stochastic stages disabled on both sides.

Current agreement, to be preserved or explicitly re-budgeted:

```
stage 7, all 93 stocks     max abs < 1e-5
stage 8, all 93 stocks     max abs ~2e-7
whole chain, N=768         mean abs 9.3e-7   p99 3.9e-6
```

**O3 is the exception** and must be validated statistically rather than by
comparison — see §5.

**One trap to avoid.** `test_chain_02_08.cpp` earlier reported two false failures
because the *test* resolved the gauge from `profile.default_format` while
`Algorithm_Main` correctly honours `algoCtrl.filmFormat` (default `"super35"`).
When a harness disagrees with the engine, suspect the harness first.
