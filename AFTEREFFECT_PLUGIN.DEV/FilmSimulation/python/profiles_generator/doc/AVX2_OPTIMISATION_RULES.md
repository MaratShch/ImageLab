# AVX2 Single-Thread Optimisation Rules — intrinsics-level playbook

**Scope:** one thread, one core, manual `_mm256_*` intrinsics, C++14. No GPU,
no OpenMP, no auto-vectoriser reliance. Written 2026-08-11 after the two
optimisation passes that took an HD frame from 702.4 → 358.5 ms (sandbox) /
694.3 → 381.1 ms (owner's machine); every rule below is tagged with its
evidence level:

* **[M]** — measured in this engine, numbers quoted.
* **[D]** — derived arithmetically from measured numbers in this engine.
* **[L]** — standard practice from architecture literature, NOT yet measured
  here. Treat as a hypothesis to test, not a fact.

The single most important lesson of the two passes is not any individual
trick. It is that **two of my own confidently-stated premises were falsified
by measurement in one day** (the "grain is 28 %" figure and the "sub-pixel
blurs are DRAM-bound" premise behind the fused sweep). The rules are
therefore split into *method* rules — how to decide — and *technique* rules —
what to do. The method rules outrank the technique rules.

---

## Part A — Method rules (these outrank everything below)

### A1. Classify the loop before touching it. [M]

Every loop is limited by exactly one of three things, and the optimisation
that helps one class does nothing — or harm — to the others:

| class | symptom | what helps | what does NOT help |
|---|---|---|---|
| **arithmetic-bound** | ns/px falls when taps/terms are removed | fewer/cheaper ops, FMA, LUT | traffic reduction |
| **bandwidth-bound** | ns/px tracks bytes touched, not ops | fewer passes, fusion, smaller types | more/cheaper ALU work |
| **latency-bound** | neither; serial dependency chain visible | break the chain, unroll accumulators | both of the above |

How to classify cheaply — **CORRECTED 2026-08-11; the first version of this
rule was wrong.** It said "halve the traffic by running on half the rows",
which halves traffic AND arithmetic together and isolates nothing. Valid
procedures:

* **Vary arithmetic, hold footprint:** drop half the taps, same plane. Time
  moves => arithmetic-bound.
* **Vary footprint, hold arithmetic:** run the identical kernel on a plane
  small enough to sit in L2, repeated to the same total sample count. Time
  per sample falls sharply => bandwidth-bound.
* **Preferred — counters, no code edit:** `perf stat -e cycles,instructions,
  cache-misses,LLC-load-misses`. High IPC with few LLC misses => arithmetic-
  bound; the reverse => bandwidth-bound; low IPC with few misses =>
  latency-bound.

Measured reference points for this machine class: streaming copy sustains
**24–27 GB/s** (measured across four frame sizes); the engine's blur calls
achieve ~11 GB/s because of pass structure; a plane at HD float is 8.3 MB —
**which fits a large L3**, so "it's DRAM-bound" is usually false at HD and
must be *measured*, not assumed. That false assumption cost this project a
half-day of fusion work that first came out *slower*.

### A2. Min-of-N per stage, or the number is noise. [M]

Run-to-run spread measured 6.7 % at frame level and **up to ±40 % at stage
level on single runs** — a single-run per-stage comparison produced a
claimed 2.00× on the DIR stage that was actually 1.10×. Rules:

* Frame totals: min of ≥3 runs.
* Per-stage claims: min of ≥3 runs **per stage**, not per frame.
* One-frame harnesses include first-touch page faults (~20 ms at HD,
  concentrated in stages touching fresh planes — stage 02 measures
  3.3 ns/px on frame one vs 0.74 on frame three). Loop ≥3 frames on one
  arena and read the last table.
* Anything between 0.93× and 1.07× is reported as "within noise", never as
  a win or a loss.

### A3. Verify against an INDEPENDENT reference, chosen by change type. [M]

* **Traffic/structure change** (fusion, pyramid, pass reorder): result must
  match — the fused sweep was verified against a brute-force scalar 2-D
  wrapped convolution (not against the code it replaced) across 8 odd-sized
  planes × 9 sigmas, to 2e-05.
* **Approximation change** (LUT, polynomial, fast exp/log): quantify error
  against the scalar reference over the full profile sweep, in DN at 8-bit,
  and record it. Current pipeline worst: 3.20e-02 = 8.2 DN.
* **Generator change** (RNG): NEVER difference images. Verify mean,
  variance, skew, kurtosis, extreme values, determinism on repeat call and
  cross-stream correlation. (The vector RNG passed all seven; it also
  happened to be per-sample close, but that was luck of construction, not
  the test.)

### A4. Instrument the callee, not only the caller. [M]

The per-stage table attributes blur time to eight different stages; only a
histogram *inside* `AlgoGaussianBlurPlaneWrap` keyed on sigma revealed that
the frame's cost concentrates at σ 0.2–1.1 (13 calls) — information no
stage-level view can produce, and the fact that killed the box-cascade
proposal before it wasted accuracy. When one function serves many callers,
put a temporary counter in the function.

### A4b. "Faster than before" is not "fast" — re-profile any path you adopted. [M]

`pyramidUpsample` stayed fully scalar for the whole project because the
pyramid path was only ever timed as a WHOLE, and as a whole it was a big
improvement over the 41-tap direct kernel it replaced. Vectorising that one
loop later gave another 1.8x to 2.85x on every wide lobe. When a path is
adopted because it beat something worse, put the profiler INSIDE it
afterwards - the aggregate win hides whatever is still slow within.

### A5. Estimate → measure → keep or revert, in writing. [M]

Two optimisations this project attempted made things slower (both upsample
"improvements"; the first fused-sweep attempt at σ 0.3–1.1). The discipline
that kept them from shipping: every change is benchmarked in isolation
before it is benchmarked in the frame, and a change that loses is reverted
*with the numbers recorded in a comment at the site*, so the next person
does not re-attempt it.

---

## Part B — Technique rules, ordered by measured value in this engine

### B1. Collapse transcendental CLUSTERS into a per-frame LUT + gather; never a single call. [M]

Measured costs per sample: Schraudolph `Exp` 0.22 ns, `Log` 0.23 ns,
2048-entry LUT with `_mm256_i32gather_ps` + FMA lerp 0.54 ns.

* One `exp` per sample → LUT is **2.4× slower**. Keep the polynomial/trick.
* Two softplus (2 Exp + 2 Log) per sample → LUT is ~1.5× faster **and three
  orders of magnitude more accurate** if entries are built the scalar way.
* Same expression inside an iteration loop (stage 8b: up to 12 curve
  evaluations/px) → LUT is transformative: **164.5 → 8.7 ms**.

**Caveat on the 0.54 ns figure:** measured in a tight loop with the table hot
in L1 and nothing competing. In a real stage the gather shares L1 with
streaming plane data, so treat 0.54 ns as a FLOOR and re-measure in place.

Requirements that made it safe here and must hold elsewhere: the function is
a frame constant (table built once per frame, stack-local, reentrant); both
ends of the domain are asymptotically flat so clamping is *exact*; index is
clamped in float before conversion so no lane can form an out-of-range
gather address.

### B2. Emulate missing integer ops exactly rather than dropping to scalar. [M]

AVX2 has no 64×64 multiply, which kept the counter-RNG scalar at
20.1 ns/px. `_mm256_mul_epu32` + the identity
`a·b mod 2^64 = al·bl + ((al·bh + ah·bl) << 32)` reproduces the scalar
result bit-exactly in three multiplies. Full grain field: 51.0 → 5.6 ms.
The general rule: before accepting "AVX2 can't do this", write the missing
op from narrower primitives and check whether the emulation is still ≥4×
the scalar throughput. It usually is, because eight lanes forgive a 3×
per-op emulation cost.

### B3. Nothing branches, selects, or indexes-with-modulo inside the x-loop. [M]

The fused sweep's first version computed `(base+t) mod win` per tap per
vector — a compare and select on the critical path of every FMA — and was
*slower* than the code it replaced. Hoisting the slot resolution to once
per row flipped it to a win. Corollaries, all already standard in this
codebase and to be kept: broadcast constants once per channel
(`_mm256_set1_ps` outside the loop); branch on frame constants (`reversal`)
outside the loop and let the compiler hoist; guarded reciprocals computed
once per frame so the loop multiplies instead of divides.

### B4. Fuse separable passes only when the working window fits L2, and re-measure the premise. [M]

The two-pass separable blur traverses the plane four times; a rolling
window of 2·half+1 horizontally-blurred rows makes it two. **But** the
saving only exists where the intermediate would have missed cache: at HD an
8.3 MB plane sits in L3, so the measured gain was 1.10× at frame level, not
the ~1.8× DRAM arithmetic promised. Apply with: window ≤ ~130 KB
(half ≤ 8 at HD width = fits L2), plane taller than the window, and slot
pointers hoisted per row (B3). Expect the gain to GROW with frame size —
at 4K a plane is 33 MB and genuinely leaves L3, so the DRAM arithmetic
starts holding. **[D]** for the 4K claim: predicted, not yet measured.

### B5. Do wide kernels at reduced resolution; compensate variance exactly. [M]

Product-of-Gaussians ⇒ variances add ⇒ a box-decimation of factor k (its own
variance (k²−1)/12) followed by a narrower Gaussian and a cell-centred
upsample reproduces the wide blur. Measured: σ 34 direct 5.9 ms → 4.3 ms,
and *more* faithful, because the direct kernel was truncating at 1.9 σ on
wide lobes. Threshold placement is a measured crossover, not a guess —
it moved from 6.0 to 3.5 only after the decimation stopped being scalar.
The decimation itself must follow B3 (two contiguous passes, wrap on row
index only).

### B6. Skip work below representable precision — with the arithmetic written down. [M]

σ < 0.20 blur: side tap exp(−12.5) = 3.7e-06 of centre, below 16-bit
quantisation (1.5e-05) → the blur IS a copy; replacing it measured
1.02 → 0.41 ms per call. The rule generalises (identity taking matrices are
already skipped on 141/142 stocks; zero-gain halation channels are skipped)
but each skip must carry the numeric argument in a comment, with the
threshold placed where the claim is *unarguable*, not merely plausible —
0.20 was chosen over 0.25 because at 0.25 the discarded term is
representable at 12 bits.

### B7. Unaligned access everywhere on pool memory; masked tails everywhere. [M]

History, twice over: an aligned store on an interior row offset faulted
(interior starts at x = half, an arbitrary integer), and the arena base
itself arrived 16 mod 32 from the host pool. On Haswell+ an unaligned op on
aligned data costs nothing extra, and the only real penalty — cache-line
splits — is set by the base alignment regardless of instruction choice. So:
`loadu`/`storeu` on all plane data, `maskload`/`maskstore` for tails (keeps
row padding untouched, which the NaN-poison arena test depends on), aligned
ops only on compiler-aligned file-local tables (`AVX2_ALIGN`).

### B8. Keep integer work in the integer domain end-to-end. [M]

The vector RNG forms counters, mixes and extracts bits entirely in
`__m256i`, converting to float exactly once (`_mm256_cvtepi32_ps` on the
final 24-bit uniform). Uniform width is matched to the DESTINATION type —
24 bits for float, since a float mantissa cannot hold more — not to the
source's 53-bit convention. Round-tripping through float mid-pipeline costs
conversions and loses bits; both are avoidable by design.

### B9. Break dependency chains in reductions with several accumulators. [L]

An FMA has ~4 cycles latency and ~0.5 cycles throughput, so a reduction
written with ONE accumulator runs at latency speed and wastes roughly 8x the
available throughput. Any `acc = fma(a, b, acc)` over a long run wants 4-8
independent accumulators summed at the end.

Candidates here: `AlgoPlaneMean` (whole-plane sum), the tap accumulation in
both blur passes (n up to 129), the horizontal decimation in
`pyramidDownsample`. **Not measured here** — the compiler may already be
splitting these, which is exactly why this is [L] and needs an isolated
benchmark before any rewrite.

### B9b. If a transform parameter is constant over the frame, its KERNEL is too. [M]

Both sub-pixel resamplers derived their interpolation weights per pixel, and the
first Catmull-Rom rewrite kept that structure - costing 125 ms and turning a
fidelity fix into a net loss. The gate displacement and the registration error
are FRAME CONSTANTS, so the four tap weights per axis are frame constants: what
looks like a general resample is a fixed separable convolution at an integer
offset, whose taps are contiguous loads rather than gathers, and whose interior
needs no boundary handling at all.

Measured: stage 10 88.5 -> 4.05 ms, stage 15 70.3 -> 2.64 ms. The vectorised
four-tap cubic ended up **6.5x faster than the two-tap bilinear it replaced**,
because the bilinear version was itself a scalar per-pixel loop doing modulo
arithmetic. Ask of any per-pixel computation: which of these quantities actually
varies per pixel?

### B10. Guarantee no subnormals reach a hot loop. [L]

Subnormals can cost 100+ cycles per operation on several x86 generations, and
the symptom is a stage that is inexplicably slow while profiling normally in
instruction count. Three places in this engine legitimately approach zero:
the grain field, the exposure floor at stage 8, and halation's
energy-conserving difference clamped at zero.

1. **Prove they cannot occur** — the exposure floor and the zero clamps exist
   for physical reasons and sit far above the float subnormal threshold of
   1.2e-38.
2. **Flush to zero** — FTZ/DAZ in MXCSR.

**Why this is a decision, not a free win:** MXCSR is thread-global, so setting
it inside `Algorithm_Main` would change the HOST application's floating-point
behaviour on that thread and break the engine's "touches no global" contract.
Route 1 (audit and prove) is correct here; route 2 belongs to the host and
should be documented, not taken.

---

## Part C — Concrete remaining proposals for THIS engine, with estimates

Current frame (owner's machine, single run): 381.1 ms, of which blur ≈ 239 ms
(63 %): halation 118, MTF 41, scan 26, DIR 24, weave 17, defocus 13.

### C1. Multi-kernel single sweep: N lobes of the SAME source in one pass. [D — est. −15 to −30 ms]

`AlgoMultiGaussianBlurPlaneWrap` currently runs `lobeCount` complete
independent blurs of the *same source plane* and accumulates. Halation does
this 3 lobes × 3 channels = 9 full blurs; grain blurs the same noise plane
twice (narrow + wide). A multi-kernel sweep shares the source reads and the
horizontal window across all kernels of one call: horizontal pass produces
per-lobe blurred rows from ONE read of the source row (the window's row
storage grows ×lobes but stays L2-sized for the narrow lobes), and the
vertical accumulation emits the weighted sum directly — eliminating both
the destination-clear pass and the per-lobe accumulate pass.

**Traffic recounted against the code, 2026-08-11 — the first version of this
estimate ("~10 traversals to ~4") was asserted, not derived, and was wrong.**
Actual: clear destination (1 write) + per lobe [ blur = 2 traversals fused or
4 two-pass, then accumulate = read lobe + read dst + write dst = 3 ]. A
three-lobe call is **16 traversals if every lobe fuses, 22 if none does, ~17
in practice** with the wide lobe on the pyramid. Target ~5-6, so the available
reduction is about 3x on blur traffic — larger than first claimed, but the
original figure should not have been quoted. The wide lobe (σ ≥ 3.5)
stays on the pyramid path; sharing applies to the 1–2 narrow lobes plus the
accumulate. Grain's narrow+wide pair is the cleanest first target: same
source, two kernels, and the wide one is pyramid anyway, so the win is the
shared read + fused accumulate.

**STATUS 2026-08-11: IMPLEMENTED, both halves.** Upsample vectorisation gave
358.5 -> 332.5 ms; accumulate-mode gave 332.5 -> 321.3 ms (11.2 ms against a
12-15 ms estimate). `AlgoMultiGaussianBlurPlaneWrap` now issues one blur per
lobe and nothing else. Implementation note worth reusing: the mode is a
TEMPLATE parameter, not a runtime flag, and the weight multiply is applied in
both modes with 1.0 on the public path - exact in IEEE-754, so the historic
path stayed bit-identical through a refactor of the engine's most-called
function.

### C2. IIR recursive Gaussian (Deriche / van Vliet) for the mid band. [L — est. −10 to −25 ms, ACCURACY TRADE]

Constant cost per pixel regardless of σ: ~4 multiply-adds per direction
against 2·(2·half+1) for FIR — at σ 2.8 (half 12) that is 4 vs 50. The
vertical pass vectorises trivially (8 columns per vector, sequential rows).

**CORRECTED: the horizontal pass needs neither a transpose nor "lane
rotation" — that phrasing was muddled.** The recurrence runs along x, so the
correct form is EIGHT ROWS IN PARALLEL, one per lane: lane j holds row y+j and
all lanes march x together, with the 2-4 previous outputs per lane held in
registers.

Costs that must be paid before adoption: the impulse response is
only approximately Gaussian (~1e-3 relative); the circular boundary needs the
filter state initialised from the periodic extension, a derivation this
document has NOT done and must not hand-wave as a "steady-state trick"; and
AVX2-vs-scalar divergence
becomes systematic rather than rounding-level — the same policy question as
the σ-cutoff proposal (P2), so it needs the owner's explicit sign-off. The
old proposal document's O7 estimated it and deferred it; the fused-sweep
experience (B4) says: build the horizontal-pass prototype and measure before
believing the 12× tap arithmetic.

### C3. Pointwise stage fusion — 02+02b, 14b+14c, 16+17, 9b into 10. [D — est. −8 to −12 ms, DESIGN TRADE]

Each fusion removes one full read+write traversal (~8 MB at HD float).
Deferred to date because it breaks the documented one-buffer-per-stage
inspectability that debugging the physics depends on. Recommendation
unchanged: do this LAST, when the stage set is frozen, and keep a
compile-time switch (`ALGO_RETAIN_ALL_STAGES` already exists and is the
natural flag) so the unfused path remains available for debugging.

### C4. Non-temporal stores for final-write planes. [L — est. −2 to −5 ms]

`_mm256_stream_ps` bypasses the cache for stores whose data is never
re-read — candidates are stage 17's `Dst_*` planes and the last write of
any ping/pong pair before reuse distance exceeds L3. Requirements: 32-byte
aligned destinations (NOT guaranteed by the pool — would need the write
loop to peel to alignment), full-line writes, and an `_mm_sfence` at stage
end. Given the alignment complication and the modest bound, measure first
on a standalone kernel; drop if <2 ms.

### C5. Software prefetch: do not bother. [L, negative recommendation]

Every hot loop in this engine walks rows sequentially; hardware prefetchers
on anything Haswell+ already track that pattern. `_mm_prefetch` pays only on
irregular access — the LUT gather is the sole irregular consumer, and its
table is 8 KB, permanently L1-resident. Listed to prevent the time being
spent.

### C6. Pre-touch the arena at allocation. [M — −20 ms on one-shot renders, 0 on sequences]

Not an AVX2 rule but recorded here because it distorts every AVX2
measurement: first-touch page faults land in whichever stages first write
fresh planes. One streaming pass over the arena at alloc time (or
`MADV_POPULATE_WRITE` where available) moves the cost to setup where it is
visible and amortised.

### C7. The ceiling, restated. [M]

With C1–C4 landed and lucky, HD single-thread sits near **320–340 ms**.
The measured copy floor for the engine's traversal count is ~50 ms; the
distance between those numbers is arithmetic that is already vectorised.
There is no further 2× on one thread. The next real factor is row-band
threading (the engine is stateless and reentrant by design — it is already
thread-safe) or GPU. Any future proposal claiming otherwise should be
required to name which of the three A1 classes it accelerates and show the
isolated measurement.

---

## Part D — Standing constraints these rules operate under

Unchanged, and every rule above respects them:

1. Float32 lanes; `HighPrecType` only for setup-time scalars and wide
   accumulators (`AlgoPlaneMean`).
   **AUDITED AND LARGELY ENFORCED 2026-08-11** — seven files were in breach with
   per-pixel `double`, including a divide per pixel in the vignette and
   `atan2`/`log10` in double per defect pixel. Converted: vignette, coating-field
   upsample, corner defocus, edge fog, gate-weave sampler, dust rasteriser, plus
   a loop-invariant of mine hoisted out of the blur upsample. **321.3 -> 297.7 ms.**
   Sanctioned exceptions documented at each site (coating-field phase grows
   unbounded along a clip; AlgoPlaneMean accumulator; particle placement; anchor
   bisection; planckRadiance; kernel build). Still outstanding: the fibre/polygon
   rasteriser in 09 and the blob rasteriser in 16. Full record:
   `D1_TYPE_ALIGNMENT_2026-08-11.md`.
   **Method lesson: violation COUNT is not a cost proxy - call FREQUENCY is.**
   Algo_09 had 36 in-loop occurrences and gave nothing measurable; Algo_15 had
   two and gave 1.95x on its stage, because its sampler ran 6.2 million times
   per frame.
2. Same file names, function names, prototypes as the scalar build — the
   AVX2 folder is a drop-in TU set, no header divergence.
3. The scalar path is the accuracy reference and is never modified for the
   vector path's convenience.
4. No mutable shared state — every table/window is stack-local; reentrancy
   across concurrent frames is a contract, not a preference.
5. Tails by mask, padding untouched, NaN-poison test stays meaningful.
6. Determinism: every stochastic value a pure function of
   (seed, frame, stage, ordinal), independent of render order and tiling.
