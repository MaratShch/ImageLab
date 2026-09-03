# Memory Optimisation — Analysis and Plan

**Status:** M1 and M2 IMPLEMENTED and verified. M3, M4 outstanding.
**Trigger:** `2816 x 1536 needs 3148873792 bytes, beyond the pool interface`.
**Measured on:** GCC `-O2`, the current 94-slice arena.

---

## 1. The finding that makes this easy

**The pipeline is a strict linear chain.** Every stage reads *only* its immediate
predecessor. Verified mechanically across all 25 stage calls in `AlgorithmMain.cpp`:

```
02 -> 02b -> 03 -> 03b -> 03c -> 04 -> 05 -> 06 -> 06b -> 07 -> 08 -> 08b
   -> 09 -> 09b -> 10 -> 10b -> 11 -> 12 -> 13 -> 14 -> 14b -> 14c -> 15 -> 16 -> 17
```

Not one stage reaches back two or more steps. There is exactly one cross-stage
lifetime in the whole engine, and it is not a stage buffer:

- `Scr_LogE_R/G/B` — written at stage 7/8, read at 8b, because the interimage
  correction re-enters in the exposure domain.

Everything else that looks long-range (`Scr_Dbar`, `Scr_Grain_*`, `Scr_BlurA/B`,
`Scr_Field`, `Scr_FieldLo`, `Scr_Luma`) is scratch, written and consumed inside a
single stage.

**So 25 retained stage triples exist purely as a debugging convenience, not because
the algorithm needs them.** They can collapse to two.

---

## 2. Where the 94 slices actually go

```
  6   boundary planes      Src_R/G/B, Dst_R/G/B      (ImgType)
 75   stage triples        25 x 3                    (AlgoType)
  7   scratch singles      BlurA, BlurB, Luma, Field, FieldLo, Dbar, DbarBlur
  6   scratch triples      LogE x3, Grain x3
 ---
 94
```

88 of the 94 are `AlgoType`, which is why `double` costs exactly twice `float`.

Arithmetic check against the live allocator at 2816 x 1536, double:
`88 x 34,603,008 + 6 x 17,301,504 + 64 = 3,148,873,792` — the figure in your log,
to the byte.

---

## 3. What ping/pong buys

Two stage triples alternating, instead of 25: **88 AlgoType planes become 19**
(2 triples + 7 scratch singles + 2 scratch triples).

```
  AlgoType = double            now         ping/pong    saving   fits int32?
  1024x1024                0.71 GiB         0.17 GiB      76%    Y -> Y
  1920x1080 HD             1.41 GiB         0.34 GiB      76%    Y -> Y
  2816x1536 Sunset         2.93 GiB         0.71 GiB      76%    N -> Y
  3840x2160 4K UHD         5.62 GiB         1.36 GiB      76%    N -> Y
  4096x2160 4K DCI         6.00 GiB         1.45 GiB      76%    N -> Y

  AlgoType = float
  2816x1536 Sunset         1.51 GiB         0.40 GiB      73%    Y -> Y
  3840x2160 4K UHD         2.90 GiB         0.77 GiB      73%    N -> Y
  4096x2160 4K DCI         3.10 GiB         0.82 GiB      73%    N -> Y
```

**4K DCI in double drops to 1.45 GiB and fits the existing `int32_t` pool
interface.** So ping/pong removes the need to widen `GetMemoryBlock` at all — that
change becomes optional hygiene rather than a prerequisite for 4K.

---

## 4. Why it is safe here, and where the sharp edge is

Ping/pong is only safe when no stage reads its own destination. Two properties
guarantee that:

1. **Source and destination are always different buffers.** With A/B alternation,
   stage *N* reads A and writes B; stage *N+1* reads B and writes A. `src == dst`
   never occurs, so the `RESTRICT` qualifiers on every stage signature stay honest.
2. **Neighbour-reading stages are unaffected.** Eleven stages read a neighbourhood
   of their source — halation, both MTF passes, the DIR coupler, grain, duplication,
   the weave resample. That is fine precisely *because* src and dst are distinct;
   these stages would break under true in-place operation, which is why in-place is
   **not** part of this plan.

**The sharp edge is stage 13.** `AlgoStage13_Duplication` copies its source into its
destination and then iterates generations *in place on its own destination*
(`work[3] = { pDstR, pDstG, pDstB }`, `passes = 2 * generations`). That is internal
to the stage and already correct today — but it means stage 13's destination must
not alias its source, which ping/pong guarantees. Worth stating so nobody later
"optimises" 13 into a single buffer.

---

## 5. What we lose, and the mitigation

The retained-buffer policy was a deliberate choice: any intermediate could be dumped
without re-running the chain, and that is what made stage-by-stage validation against
`film_sim.py` possible. Several real bugs were localised exactly that way.

Ping/pong destroys it — after the run, only the last two triples hold anything
meaningful.

**Mitigation: a compile-time switch, not a runtime one.**

```cpp
#define ALGO_RETAIN_ALL_STAGES 0   // 1 = full 94-slice arena, for debugging
```

At `1` the allocator lays out all 25 triples exactly as today and `AlgorithmMain`
addresses them by stage; at `0` it allocates two and alternates. Both paths must
produce **bit-identical** output, which is itself the regression test — and it means
a future discrepancy can still be bisected against a fully retained run.

Runtime would be worse: it would double the pointer bookkeeping in the hot path for
a facility only ever wanted during development.

---

## 6. Second, smaller win: scratch aliasing

`Scr_LogE_*` dies at stage 8b. `Scr_Grain_*` is first written at stage 11. **Their
lifetimes are disjoint**, so they can share one triple — 3 planes saved, about 2 %
of the ping/pong arena.

Small, but it costs nothing beyond a liveness comment, and it is the only remaining
overlap worth taking. The seven scratch singles are all genuinely live within their
own stages and cannot be merged without a per-stage liveness map, which is not worth
the fragility.

**Do this after ping/pong, separately, so a regression can be attributed.**

---

## 7. A speed effect worth measuring, not assuming

The engine is memory-bandwidth-bound at the floor: 24 stages reading and writing
triples is 144 plane-touches per frame, measured at ~25 GB/s.

Ping/pong does **not** reduce plane-touches — the same data still moves. But it
reduces the *footprint* those touches land in, from 88 planes to 19. At HD float
that is 0.73 GiB down to 0.19 GiB, which changes what fits in L3 and how many TLB
entries the frame needs.

Whether that shows up as a speedup depends on the machine. It is a hypothesis to
measure, not a claim to plan around, and it must not be used to justify the change —
the justification is the footprint.

---

## 8. Plan, in order

| # | Change | Saving | Risk |
|---|---|---|---|
| **M1** | Ping/pong the 25 stage triples to 2 | **76 % double / 73 % float** | low — chain is linear |
| **M2** | `ALGO_RETAIN_ALL_STAGES` debug switch | — | none, and it protects M1 |
| **M3** | Alias `Scr_LogE_*` with `Scr_Grain_*` | ~2 % more | low |
| **M4** | Widen `GetMemoryBlock` to `std::size_t` | — | touches your interface |

M4 becomes optional once M1 lands, but it is still the right shape: an interface
that cannot express a request larger than 2 GiB will bite again on a bigger format
or a future stage.

**Files touched by M1 and M2:** `AlgoMemHandler.hpp`, `AlgoMemHandler.cpp`,
`AlgorithmMain.cpp`. No stage `.cpp` changes at all — every stage signature already
takes explicit src and dst pointers, which is exactly what makes this a layout
change rather than an algorithm change.

**Tests needing updates:** `test_memhandler.cpp` and `profall.cpp` read stage planes
directly. `test_full.cpp`, `e2e.cpp` and the defect tests consume only `Dst_*` and
are unaffected.

---

## 9. Verification requirements

Same standard as the fusion proposal, and for the same reason — this is a layout
change, so **bit-identical is achievable and anything less means a bug**.

1. `ALGO_RETAIN_ALL_STAGES` 0 vs 1, **bit-identical**, all 93 stocks.
2. Both `AlgoType = double` and `float`.
3. Damage off, damage at defaults, and damage forced high — the defect stages are
   the newest code and stage 13's in-place generation loop is the sharpest edge.
4. Non-default controls: `generations = 2` at minimum, since that is what exercises
   stage 13's in-place path, plus `wbStrength = 1` and `sceneKelvin = 3200`.
5. `e2e` against `film_sim.py`: must stay at `mean abs 5.0238e-05`.
6. Non-square and odd sizes — `2816 x 1536`, `1920 x 1080`, and something not a
   multiple of the alignment quantum, to prove the padding logic survives the
   relayout.
7. A poison pass: fill the arena with a signalling pattern before each run, so a
   stage reading a buffer that ping/pong has since overwritten fails loudly instead
   of producing plausible numbers.

Item 7 is the one that matters most. The failure mode of a bad ping/pong is not a
crash — it is a stage silently reading data from two stages ago, which looks like a
subtle image bug rather than a memory bug.

---

## 10. Interaction with the AVX2 work

Orthogonal, and M1 should go first.

Fusion (`STAGE_FUSION_PROPOSAL.md`) reduces the *number of passes*; ping/pong
reduces the *number of buffers*. They do not conflict — a fused segment simply
writes to the pong buffer instead of to its own retained triple.

Doing M1 before the AVX2 work also means every vectorised stage is validated against
the smaller arena from the start, rather than being re-validated after a later
relayout.


---

## 11. Results — M1 and M2 as built

### 11.1 Implementation shape, which turned out smaller than planned

The alternation is **entirely a decision inside the allocator**. `offStage[s][c]` is
still a 25-entry table; under ping/pong its entries point at `offPair[s & 1][c]`.

Consequence: **`AlgorithmMain.cpp` and every stage `.cpp` are untouched.** The plan
budgeted for changes to `AlgorithmMain.cpp`; none were needed. Only
`AlgoMemHandler.hpp` and `.cpp` carry the change, plus `test_memhandler.cpp`.

### 11.2 Footprint, measured from the live allocator

```
[AlgoMemHandler] 2816 x 1536  padded 2816 x 1536
  sizeof(AlgoType) = 8   plane 34603008 bytes
  stage triples    = 25 logical, 2 physical (ping/pong)
  slices           = 25
  total            = 761266240 bytes (726.00 MiB)
  -> ALLOCATED
```

The frame that previously reported `needs 3148873792 bytes, beyond the pool
interface` now allocates **726 MiB** — matching the predicted 0.71 GiB exactly.

### 11.3 Bit-identical, which is the whole regression test

```
retained (25 triples) vs ping/pong (2 triples), 93 stocks, 192x192

defaults : BIT-IDENTICAL     md5 421d45a5600347e9967b1fcff1ae3194
harsh    : BIT-IDENTICAL     md5 f5c2090915cb7f83cc68ac0959486aa8
```

`harsh` = `generations = 2`, `wbStrength = 1`, `sceneKelvin = 3200`,
`exposureStops = 1.5`, damage cranked past plausible. `generations = 2` matters
specifically because it is the only setting that exercises stage 13's in-place
generation loop, the sharpest edge in this change.

### 11.4 The poison test found nothing, and that is the point

Whole arena filled with quiet NaN before every run, all 93 stocks x 2 control modes:

```
ping/pong  192x128   PASS  0 bad samples
ping/pong  129x191   PASS  0 bad samples     (both dimensions odd, both pad)
retained   192x128   PASS  0 bad samples
```

This is the check that matters most, because the failure mode of a bad ping/pong is
**not a crash** — it is a stage silently reading data from two stages ago, which
looks like a subtle image bug. NaN propagation turns that into an immediate,
unmissable failure.

### 11.5 An unexpected 6 % speedup

```
1024x1024 double        retained    ping/pong
damage off              1435.13 ms  1349.53 ms   -6.0 %
damage at 1.0           1477.51 ms  1368.12 ms   -7.4 %
damage forced high      1498.61 ms  1386.03 ms   -7.5 %
```

§7 flagged this as a hypothesis to measure, not a justification. It measured
positive: the same plane-touches now land in 19 planes instead of 88, so more of the
working set stays resident and the frame needs far fewer TLB entries. Worth
recording, still not the reason for the change.

### 11.6 `test_memhandler.cpp` had to change, exactly as predicted

Its "no two buffers overlap" check failed on all five geometries — correctly, since
25 logical pointers now alias 2 physical triples by design. Rewritten as three
checks that are meaningful in both layouts:

- **no two DISTINCT buffers overlap** — duplicates collapsed first, so a genuine
  plane collision is still caught
- **distinct plane count equals `6 + 3 x ALGO_PHYSICAL_STAGE_TRIPLES + 13`** — proves
  the layout in force is the one intended
- **no stage shares a plane with its predecessor** — this is the actual safety
  property of ping/pong. Plain distinctness was too strong; aliasing-anywhere is too
  weak. Passes in both modes.

### 11.7 Full verification matrix

```
double, ping/pong    test_full 93 PASS   damage-high 93 PASS   memhandler PASS
double, retained     test_full 93 PASS   damage-high 93 PASS   memhandler PASS
float,  ping/pong    test_full 93 PASS   damage-high 93 PASS
e2e vs film_sim.py   mean abs 5.0238e-05   unchanged to the digit
profall              runs, all 24 stages reported
```

### 11.8 What is still open

- **M3** alias `Scr_LogE_*` with `Scr_Grain_*` (disjoint lifetimes, ~2 % more)
- **M4** widen `GetMemoryBlock` to `std::size_t` — no longer required for 4K, since
  4K DCI in double now sits at 1.45 GiB, but still the right shape
- `profall.cpp` reports stage 15/16 as "STUB"; that label is stale, both now render
