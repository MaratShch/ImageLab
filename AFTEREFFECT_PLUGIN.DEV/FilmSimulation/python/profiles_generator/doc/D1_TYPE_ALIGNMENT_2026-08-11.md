# Rule D1 type alignment — AVX2 folder audit and conversion, 2026-08-11

**Rule D1:** *float32 lanes; `HighPrecType` only for setup-time scalars and wide
accumulators.*

The audit found **real violations in seven files** — per-pixel `double`
arithmetic in the vector build, including a double divide per pixel and a
`double atan2` per pixel of every defect particle. This is the record of what
was measured, what was converted, what was deliberately left, and what is
still outstanding.

---

## Result

**321.3 → 297.7 ms** at HD, min of 3, measured during a quiet interval
(samples 300.2 / 299.2 / 297.7). Cumulative for the day: **702.4 → 297.7 ms,
2.36×**.

Accuracy did not move: end-to-end sweep over 15 profiles stays at
**3.20e-02 worst (8.17 DN at 8-bit), no failures, all values finite** — before
`3.2039e-02`, after `3.2038e-02`.

**Honesty note on the second batch.** The stage-9b rasteriser conversion was
made after that measurement, and by then the sandbox had become noisy —
five samples spread 303–335 ms against the earlier run's 298–300. **The 9b
change is therefore verified correct but its speed effect is unmeasured; it is
not counted in the 297.7 figure.** It is kept for rule compliance, not on a
performance claim.

---

## What was converted

| file | site | why float32 is sufficient |
|---|---|---|
| `Algo_04_Sim.cpp` | vignette cos⁴, per pixel | pure geometry; pixel coords exact in float to 16.7 M; result is a smooth multiplier at ~1e-07 relative, four orders below the 16-bit destination. **Was a scalar double divide per pixel** and made the `ALGO_VECTOR_HINT` above it unhonourable. |
| `Algo_04_Sim.cpp` | coating-field bilinear upsample | widened **float plane samples** to double, interpolated, narrowed back — the two roundings in and out equal the one avoided. Bought nothing. |
| `Algo_06_Sim.cpp` | corner-defocus blend weight | same as the vignette: normalised radius used as a blend weight. |
| `Algo_10_Sim.cpp` | edge-fog distance in mm | float resolves a position across a 35 mm frame to ~4e-06 mm, a thousandth of a pixel. |
| `Algo_15_Sim.cpp` | gate-weave bilinear sampler | **the clearest violation**: `HighPrecType` throughout a function called once per pixel per channel — 6.2 M calls at HD — that only ever reads float samples and returns float. Stage measured **25.3 → 13.0 ms, 1.95×**. |
| `Algo_09_Sim.cpp` | dust/debris particle rasteriser | `sqrt`, `atan2`, `cos`, `log10` all in double per pixel of every particle bounding box. Shape mask only; float gives ~1e-07 on a quantity that becomes an opacity. |
| `AlgoSeparableBlur.cpp` | upsample interior bounds | **two defects in one**: `HighPrecType` in a repeated computation AND a loop invariant sitting inside the row loop, recomputed 1080× per call with a `ceil` and a `floor` for a value that never changes. Hoisted; the `HighPrecType` is retained at the hoisted site because it now runs once per call and the ceil/floor must land on the right integer even for a centre of 3.4999999 — exactly the case float would get wrong. **This was my own code from the third pass.** |

## Added 2026-08-13 — a new sanctioned HighPrecType site

`AlgoSpectralSensitivity.cpp` (the measured-spectral-sensitivity consumers) is
`HighPrecType` throughout and is **not** a D1 violation. Two independent
mechanisms from the precision policy apply, either one sufficient:

* **M5, exponent-range exhaustion.** Planck's law here has λ⁵ for a wavelength
  in metres (~1e-32) divided by an exponential whose argument reaches 53. The
  quotient is ~3e-55 and **flushes to zero in float32** — the result would not
  be imprecise, it would be wrong. This is the same hazard the existing
  `planckRadiance` entry documents.
* **M3, cancellation.** `expm1` is used rather than `exp(x) − 1`: at long
  wavelength and low colour temperature the argument becomes small and the naive
  form loses every significant digit.

It is also setup domain in the D1 sense: about sixty integrals per frame, never
per pixel, so the cost of holding it at double is unmeasurable. Consistent with
the rule that "per-pixel does not imply float32 and once-per-call does not imply
double — the mechanism decides".

## What was deliberately NOT converted, and why it is not a violation

* **`Algo_04_Sim.cpp` coating-field synthesis.** Its own precision note already
  explained this: the web offset grows without bound along a clip — a thousand
  frames of 35 mm is ~19 m — and is multiplied by a spatial frequency to form a
  cosine argument reaching thousands of radians. In float the low bits are gone
  and the field would decorrelate frame to frame. **This is a setup-domain
  quantity in the D1 sense even though it is evaluated per pixel.** Reading that
  comment before editing is why it survived.
* **`AlgoPlaneMean` accumulator.** The wide accumulator D1 explicitly sanctions.
* **Particle placement in 9b/16** — positions, radii, alphas, phases from the
  log-Gaussian Cox process. Computed once per particle; setup-time by D1, and
  what keeps a particle in the same place on every render of a frame.
* **`AlgoSolveAnchors` bisection.** Sixty steps whose bracket shrinks below
  float resolution; documented at the site.
* **`planckRadiance`.** λ⁵ for a wavelength in metres is ~1e-32 while the
  exponential argument reaches 53 — sixty decades of span, six calls per frame.
* **`buildGaussianKernel`.** Per-call setup; deriving taps identically to the
  scalar build is what makes the two paths comparable at all.

## Still outstanding — NOT full alignment yet

Honest status: **the biggest per-pixel offenders are converted, the tail is
not.** Remaining in-loop `HighPrecType`, by file:

| file | count | what it is | verdict |
|---|---|---|---|
| `Algo_09_Sim.cpp` | ~22 | the **fibre / polygon rasteriser** — point lists and distance-to-segment per pixel | real violation, unconverted |
| `Algo_16_Sim.cpp` | 8 | gate-defect blob rasteriser, same pattern as 9b's dust | real violation, unconverted; stage is 1.9 ms |
| `Algo_14_Sim.cpp` | 1 | single site | unaudited individually |
| `Algo_04_Sim.cpp` | ~10 | coating-field synthesis | **sanctioned exception**, see above |
| `AlgoSeparableBlur.cpp` | 6 | hoisted bounds + `AlgoPlaneMean` | **sanctioned** |

So the claim to make is: **all per-pixel-of-frame arithmetic in the vector
build is now float32, except the coating-field phase (documented exception) and
two defect rasterisers that are per-particle-bounding-box rather than
per-frame-pixel.** Full alignment needs those two, and neither is worth doing
without a quiet machine to measure it on.

## Verification performed

* **End-to-end**, 15 profiles: 3.20e-02 worst, no failures, all finite.
  Unchanged from before the conversion.
* **Damage path checked separately**, because the e2e harness disables it and
  the 9b rasteriser change would otherwise be untested: damage forced on with
  dust/debris/fibre at 2.0, AVX2 vs scalar reference — **max 2.50e-02
  (6.38 DN), mean 3.22e-03, no non-finite values**, output mean 0.16722 against
  the reference's 0.16762. The single largest deviation is a defect edge pixel,
  which is where a smoothstep in float differs from one in double.
* **All three blur harnesses** re-run after the upsample hoist: single-lobe
  2e-05, pyramid band 1.83e-02, multi-lobe 1.44e-02 — all unchanged.

## Method note worth keeping

The gains were **not** where the violation count was highest. `Algo_09` had 36
in-loop occurrences and produced no measurable change; `Algo_15` had **two**
and produced 1.95× on the stage. Violation count is not a cost proxy —
call frequency is. The weave sampler was called 6.2 million times per frame;
the fibre rasteriser is called per particle bounding box.
