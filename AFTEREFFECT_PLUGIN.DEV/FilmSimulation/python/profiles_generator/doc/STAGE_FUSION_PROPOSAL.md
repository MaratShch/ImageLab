# Stage Fusion — Reducing Memory Traffic by Combining Passes

> ## STATUS UPDATE 2026-08-28
>
> **O4 (no-op stage elision) is redefined, not merely pending.** It cannot be
> implemented as the cheap bit-identical first step this document assumes. The
> production arena maps 25 logical stage triples onto **two physical triples**
> (`offStage[s] = offPair[s & 1]` in `AlgoMemHandler.cpp`), so forwarding a
> pointer past an elided stage makes the NEXT stage's source and destination the
> same plane — and every stage signature declares them `RESTRICT`, with eleven
> stages reading a neighbourhood of their source.
>
> Elision therefore requires a dynamic plane cursor in the driver: hold a current
> triple and a free triple, execute into the free one, swap **only when a stage
> actually runs**. It is now part of the driver restructure rather than a
> standalone proposal.
>
> **Stages that must NOT be elided even when they look inactive:** stage 10 (its
> zero-floor pass runs unconditionally, so the inactive case is
> `dst = max(src, 0)`, not `dst = src`); stage 14b (the copy is only the reseau
> fallback, a base-tint multiply follows unconditionally); stage 11 with
> `grainScale > 0` and zero granularity; stage 13, whose skip is safe *only*
> because the driver pre-initialises `finalCurves = profile.curves`. Stages 08
> and 08b share `Scr_LogE_R/G/B`, the single cross-stage buffer lifetime in the
> engine, and density is not invertible through the shoulder.
>
> **Also corrected:** this document's premise that the two paths share a blur and
> a curve implementation. They did not, until 2026-08-28 — the wide-sigma blur
> and the curve table each existed in one path only. Both are now shared, from
> `AlgoSeparableBlur.hpp` (`namespace AlgoBlurDetail`) and `AlgoCurveLut.hpp`.
> Any fusion proposal costed against the old asymmetry needs re-costing.
>
> See `doc/RESULT_2026-08-28_engine_convergence.md`.


> **UPDATE 2026-08-11 — fusion is now the LARGEST remaining lever, and its
> target is measured rather than assumed.** After the AVX2 pass (see
> `AVX2_OPTIMISATION_2026-08-11.md`) the frame is 392.7 ms and the blur core
> is 251 of ~412 Mcycles. Instrumentation shows the cost concentrated in **13
> sub-pixel blur calls (sigma 0.2-1.1, 50.6 Mcycles)** that are
> memory-bandwidth-bound at ~11 GB/s, each doing TWO full passes over the
> plane. Fusing horizontal and vertical into one sweep with a rolling row
> window halves that traffic (~25 Mcycles, ~12 ms); fusing the pointwise
> chains 02+02b, 14b+14c and 16+17 removes further full-plane touches.
> Estimated total 30-50 ms, taking the frame to ~340-360 ms. Not yet
> implemented.

> **UPDATE — stage 9b is no longer a stub.** Fine dust, coarse debris and fibres now
> render there, so every "09b is a pure copy / always memcpy / stub" statement below
> is superseded. 9b remains pointwise-plus-scatter and is still foldable into stage
> 10's read pass, but it is no longer a no-op elision candidate and no longer free.
> Measured cost at 1024x1024 double: unmeasurable at default levels, +2 % at levels
> far past plausible. See `DEFECT_LAYER.md`.



**Status:** proposal. No engine source modified. The prototype lives in a sandbox
build directory only.
**Measured on:** Intel i7-14700, GCC `-O2`, `AlgoType = double`, 1024 × 1024.

---

## 1. It works, and here is the measurement

I prototyped the first fusible group — stages **02, 02b, 03, 03c** — as one pass
and compared it against the four separate stages.

```
four separate stages :   7.68 ms
fused, one pass      :   2.13 ms   -> 3.60x faster, saves 5.55 ms
traffic separate     :  180.0 MB
traffic fused        :   36.0 MB   -> 5.00x less
numerical difference :  0.000e+00  on ALL 93 stocks
```

Bit-identical, at `wbStrength = 1` and `sceneKelvin = 3200` so that every branch in
the group is actually exercised. The mechanism is proven, not projected.

**But read §4 before writing any of it.** My first two attempts at this same group
were both wrong, in ways that only measurement caught.

---

## 2. Why fusion is the right lever

The engine's floor is memory traffic, measured by running all 24 stages as pure
copies with the arithmetic deleted:

```
1024²  float    23.9 ms     576 MB    25.2 GB/s
HD     float    48.2 ms    1139 MB    24.8 GB/s
4K UHD float   ~193 ms    ~4600 MB    (linear, confirmed over 4 sizes)
```

Twenty-four stages, each reading a triple and writing a triple, is **144
plane-touches per frame**. Nothing else in the optimisation list reduces that
number — AVX2 makes arithmetic faster and leaves traffic untouched, which is why
the pointwise stages measure 1.9 ms each and do not improve under `-O3
-march=native`.

Fusion is the only proposal that moves the floor itself.

---

## 3. The segment map

A stage can join the pass in front of it if it reads **only the pixel it writes**.
A stage that reads neighbours needs the whole plane first, and becomes a *barrier*.

Barriers, verified by counting blur and resample calls per translation unit:
**3b, 5, 6, 9, 10, 11** (field build only), **13, 14b**.

Everything between two barriers collapses into one pass:

| segment | stages | passes now | fused | note |
|---|---|---|---|---|
| **S1** | 02, 02b, 03, 03c | 4 | **1** | **measured, 3.60×** |
| S2 | 04 | 1 | fold into S3 | pointwise once the field exists |
| S3 | 05 halation | barrier | — | wide blur |
| S4 | 06 emulsion MTF | barrier | — | wide blur |
| **S5** | 06b, 07, 08, 08b | 4 | **1** | 06b needs only a 5-row band |
| S6 | 09 DIR coupler | barrier | — | |
| S7 | 09b, 10 | 2 | **1** | 09b is a stub; fold into 10's read |
| S8 | 10b, 11 add | 2 | **1** | fold 10b into the grain add |
| S9 | 12 | 1 | fold into S10 | pointwise |
| S10 | 13 dupe + print | barrier | — | blur per generation |
| **S11** | 14, 14c, 15, 16, 17 | 5 | **1** | 92/93 stocks; 14b is a barrier only for Dufaycolor |

**24 stages become about 10–11 passes.** Plane-touches fall from 144 to roughly 60
— a **2.4× traffic reduction**, which puts the HD floor near 20 ms and the 4K floor
near 80 ms.

### The two segments worth the most

**S5 — 06b, 07, 08, 08b.** The interesting one. Corner defocus is a **5-tap**
kernel, so it does not need the whole plane: a rolling 5-row band is enough, which
lets it join the three pointwise stages behind it. That converts four passes into
one over the most expensive pointwise stage in the chain (08 at 110 ms).

**S11 — 14, 14c, 15, 16, 17.** Five consecutive pointwise stages for every stock
except Dufaycolor. Stage 14b sits between 14 and 14c and is a barrier, but it is a
pass-through on 92 of 93 stocks, so the segment is fusible almost always and the
mosaic case can simply take the unfused path.

### Also fold, and separately worth doing

`09b`, `10b`, `12` and `04` are each a single pointwise stage sitting next to a
barrier. Folding each into its neighbour's read or write pass costs nothing and
removes a round trip apiece.

---

## 4. Two traps, both found by measuring rather than reasoning

This is the substance of the proposal. Fusion is **not** concatenating loop bodies.

### 4.1 Operation order does not commute

My first prototype folded the stage 3 balance gains into the stage 2 exposure gain,
so the pixel loop had one multiply per channel instead of two. Correct on 92
stocks. On `TECHNICOLOR_THREE_STRIP` at `wbStrength = 1` it was wrong by
**7.3e-02**.

Cause: the real chain is 02 exposure → **02b matrix** → 03 balance. A per-channel
scale and a 3 × 3 matrix commute only when the matrix is **diagonal**, and
Technicolor is the one stock whose taking matrix is not. Folding the gains ahead of
the matrix silently reordered the operations.

The fix is to apply the gains *after* the matrix inside the fused loop — no cost,
but it has to be seen. **Every fused segment needs its operation order checked
against the unfused chain, on a stock that exercises the non-trivial path.**

### 4.2 Activity tests must be carried into the fused loop

Second attempt, order corrected. Now `AGFA_APX_100` was wrong by **1.468**.

Cause: stage 3 is a pass-through when `wbStrength == 0` **or the stock is
monochrome** — a single silver record has no colour balance to mismatch. My fused
loop applied the gains unconditionally. Thirty-six of the 93 stocks are
monochrome, so this was not an edge case.

**Every stage's activity predicate has to be evaluated and honoured inside the
fused loop.** Those predicates currently live inside each stage function; fusion
requires exposing them, which is the same refactor proposal **O4** (no-op elision)
needs. That is a real argument for doing O4 first.

### 4.3 What made both bugs visible

Comparing the fused result against the unfused result, per stock, over the whole
database, with controls set so the interesting branches run. At the default
`wbStrength = 0` **both bugs produce a difference of exactly zero** — the first
prototype looked perfect.

Any fusion work must be validated that way or it will ship broken.

---

## 5. Sequencing — fusion is not the first thing to do

Fusion saves memory traffic. The frame is currently **compute**-bound, so the
saving is real but small against today's total:

| fused segment | now | after fusion | saving |
|---|---|---|---|
| S1 (02, 02b, 03, 03c) | 7.68 ms | 2.13 ms | **5.55 ms** measured |
| S5 (06b, 07, 08, 08b) | ~129.8 ms | ~112 ms | ~18 ms projected |
| S11 (14, 14c, 15/16/17) | ~85.7 ms | ~72 ms | ~14 ms projected |
| folds (09b, 10b, 12, 04) | ~8 ms | — | ~8 ms projected |
| | | | **~45 ms of 1467 ms** |

Three per cent. Because halation (748 ms), grain (176 ms), MTF (142 ms) and the
curve (110 ms) are arithmetic, and fusion does not touch arithmetic.

**Fusion becomes the dominant win only after O1, O2 and O3**, when the frame is
bandwidth-bound and the floor is what remains. Recommended order is unchanged:

1. **O4** no-op elision — free, bit-identical, and it produces the activity
   predicates fusion needs anyway.
2. **O1** pyramid blur — halation is 51% of the frame.
3. **O2** curve LUT.
4. **O3** grain RNG.
5. **Re-measure.** Everything below is projection until this point.
6. **Fusion**, starting with S1 (already prototyped and proven) then S11, S5.

---

## 6. What fusion costs

The retained-buffer-per-stage property — chosen deliberately so any intermediate
could be dumped without re-running the chain, which is what made stage-by-stage
validation against the Python reference possible.

A fused segment has no intermediate buffers. `S02`, `S02b` and `S03` stop existing
as inspectable results; only `S03c` remains.

**Two mitigations, both cheap:**

- Keep the unfused path behind a compile-time switch. A future discrepancy can then
  be bisected against the per-stage version, which is exactly how the two bugs in
  §4 were localised.
- Fuse only where the intermediates have no diagnostic value left. `S02` and `S02b`
  are a scale and a matrix multiply — nobody will ever need to look at them again.
  `S08` density, by contrast, is worth keeping visible.

**A second benefit worth stating:** fewer retained buffers means a smaller arena.
At 4K the current footprint is 2.90 GB float / 5.62 GB double. Fusing the segments
above removes roughly a third of the retained triples, taking 4K float towards
2 GB — which matters for an AE plugin independently of speed.

---

## 7. Verification requirements

For each fused segment, before it is accepted:

1. **Bit-comparison against the unfused chain, per stock, all 93.** Not a spot
   check. Both §4 bugs affected a minority of stocks.
2. **With non-default controls.** At minimum `wbStrength = 1`,
   `sceneKelvin = 3200`, `generations = 2`. Both bugs were invisible at defaults.
3. **`test_full`** — all 93 stocks finite, in range, `Dst_*` written.
4. **`e2e.cpp` + `e2e.py`** — the whole chain against `film_sim.simulate()`, to
   confirm the fused engine still agrees with the reference and not merely with its
   own previous self.

Bit-identical is the standard here, and it is achievable — the S1 prototype reaches
it. Unlike the pyramid blur (O1), which is an approximation and needs an error
budget, fusion changes nothing but the order in which memory is touched. If a fused
segment is not bit-identical, something is wrong with it.
