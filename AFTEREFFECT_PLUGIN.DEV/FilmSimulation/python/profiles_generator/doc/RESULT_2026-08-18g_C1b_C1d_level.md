# RESULT 2026-08-18g — C1b + C1d: the grain LEVEL, and the convention that decides it

> ⚠ **ITS §5 DEFERRAL IS NOW RESOLVED, AND ITS PREMISE WAS WRONG (2026-08-23, C1e).**
> §5 declined to re-level the four VISION3 rms values because the raster extractor was
> suspected of reading its σ axis ~1.3× high. **It is not:** the 5219 panel's own 15-tick
> comb reproduces 0.001–0.100 within 1 % on the stored calibration. The 1.32× came from
> comparing the two documents at ABSOLUTE D 1.0; read at NET 1.0 they differ by 1.12× in
> green and 1.25× in blue. The deferral's *conclusion* still stands — the VISION3 green
> levels were left alone — but the *reason* is corrected, and the per-layer half is adopted:
> 5219 5.92/6.60/17.84, 5207 rms_b 8.92, 5203 rms_b 4.71. See `RESULT_2026-08-23_c1e_c8.md`.

**Task:** complete C1b (normalise the grain law so `rms_granularity` means what it says) together
with C1d (re-level the traced stocks from their own curves). Owner approved both, plus policy
**(a) preserve appearance** for the rms values that had been fitted by rendering.

**Outcome:** done, and the scope came out very different from the scope I proposed — because the
proposal rested on an assumption the documents contradict. That correction is the main content of
this entry.

Build after the work: **`build.py --root <corpus>` OK**, 7/7 audits green, `verify.py`
**240 PASS / 2 FAIL** (the two known baseline failures), C++ clean on 18 TUs. Schema **v8 → v9**.

---

## 1. The correction: a level is meaningless until its reference density is named

I scoped C1b as "the legacy law's value at D = 1.0 is `sqrt(1 − dmin + fog)`, so the stored rms
carries an accidental per-stock factor", and reported a **per-channel spread of up to 2.8× on 43
stocks** as a channel-balance defect. Both statements silently assumed `rms_granularity` means σ at
**absolute** density 1.0.

Two sheets in this corpus print the convention, in a footnote, unambiguously:

> **Kodak 5248 p1** — "Diffuse RMS Granularity\* Less than 5"
> "\* Read at a **net** diffuse visual density of 1.0, using a 48-micrometre aperture."
>
> **Kodak 5222 (DOUBLE-X) p1** — "Diffuse rms granularity\* 14", identical footnote.

Net, not absolute. And at net density 1.0 the legacy law is `sqrt(1 + fog)` — **dmin cancels**:

| stock | factor at net 1.0 (r/g/b) | factor at absolute 1.0 (r/g/b) |
|---|---|---|
| 5246 | 1.086 / 1.086 / 1.086 | 0.985 / 0.728 / 0.424 |
| KONICA_IMPRESA_50 | 1.058 / 1.058 / 1.058 | 0.959 / 0.707 / 0.346 |
| DOUBLE-X (unmasked) | 1.105 / 1.105 / 1.105 | 1.011 / 1.011 / 1.011 |

**So there was never a channel imbalance.** The 2.8× spread I reported was an artefact of measuring
a net-referenced law against an absolute-referenced expectation. The defect was in the question.

What C1b actually is, therefore: divide by `sqrt(1 + fog)` — **a uniform 4–8 % amplitude drop per
stock, identical in all three channels, shape untouched.** Not ±30 % with a rebalance.

⚠ The old code had the contradiction written into it. `grain_sigma`'s docstring said the multiplier
is "1.0 at D = 1.0" so that rms "keeps meaning what the datasheets say it means: the rms at a net
diffuse density of 1.0" — two different densities in consecutive sentences. The code implemented the
first. **Method rule 21** now covers this class.

## 2. What changed in the code

| file | change |
|---|---|
| `film_profiles.py` | `grain_sigma()` normalises at `dmin + 1.0` instead of `1.0`, for both laws. `sigma_measured_usable()` now requires `dmax > dmin + 1.0` — the real requirement, and strictly harder than the old `dmin < 1.0 < dmax` (all 11 measured stocks pass). `SCHEMA_VERSION` 8 → 9 |
| `film_sim.py` | stage 11 drops the `legacy_mid` factor; amplitude is now `rms × grain_sigma(...)` and nothing else |
| `cpp_codegen.py` | `FilmGrainSigma()` mirrors the net normalisation. ⚠ **Its documented calling convention is now the opposite of what the v8 header said** — the old text told an existing renderer to multiply by its own `sqrt(D − dmin + fog)` at D = 1.0 to preserve level; following that now double-counts the correction. Called out in the header comment |

**Why the schema version moved with no layout change.** `sizeof(GrainSpec)` is unchanged; the
*meaning* of a field changed. A plugin pairing v9 data with a v8 sampler compiles clean, runs clean,
and renders the wrong grain level. That is precisely the case a version number exists for.

## 3. C1d: the six re-levelled stocks, read at net 1.0

Once the convention is net, the re-levelling is much smaller than I reported — and its per-layer
content is the interesting part. Values read at **one exposure point**: the abscissa where the
visual (green) record reaches net 1.0. Reading each record at its own net 1.0 instead agrees within
5 %, so the choice does not matter numerically; it is stated because it could have.

| stock | rms was | rms now (green) | ratio | r / g / b adopted | blue ÷ green |
|---|---|---|---|---|---|
| 5245 | 4.2 | **4.10** | 0.98× | 3.80 / 4.10 / 11.42 | 2.8× |
| 5246 | 5.3 | **6.78** | 1.28× | 7.03 / 6.78 / 12.56 | 1.9× |
| 5248 | 5.6 | **5.87** | 1.05× | 4.42 / 5.87 / 11.29 | 1.9× |
| 5274 | 5.8 | **6.68** | 1.15× | 5.34 / 6.68 / 15.75 | 2.4× |
| 5279 | 8.3 | **8.74** | 1.05× | 6.87 / 8.74 / 20.39 | 2.3× |
| 5218 | 7.3 | **6.65** | 0.91× | 5.51 / 6.65 / 15.51 | 2.3× |

Two findings here:

1. **The tier-3 family ladder was roughly right** — 0.91–1.28×, median 1.05×. My earlier
   "1.3–1.6× understated" claim was the same absolute-vs-net error, and the 2026-08-18g README
   entry is corrected on that point.
2. **The per-layer ladder was not.** `GrainSpec`'s docstring assumed blue ≈ 1.3× green as a tier-2
   estimate; measured blue is **1.9–2.8× green on all six sheets**. That is the substantive data
   change in this pass, and it is measured rather than assumed for the first time.

⚠ **One conflict recorded, not resolved, not averaged.** 5248 is the only one of the six that prints
a scalar: "Less than 5", at net visual 1.0. The traced green at net 1.0 is **5.87** — 17 % above the
printed bound; the traced **red** is 4.42, inside it. So either Kodak's single figure names the red
record, or it is a rounded marketing bound, or this sheet's σ axis reads ~15 % high. The traced
value ships (it is the only per-layer evidence), the conflict is on file, and nothing was averaged.

## 4. Policy (a): four Svema stocks, changed on paper, identical on screen

`SVEMA_FOTO_65` ("rms 11.5 kept: still [T1] (fitted through the full pipeline)") and
`SVEMA_FOTO_250` ("Tuned through the FULL PIPELINE … swept against rendered output") had their rms
**fitted against pre-C1b rendered output**. `SVEMA_FOTO_32` and `_130` are sqrt-speed scalings *of*
those fits, so they inherit the same calibration. All four were multiplied by their own
`sqrt(1 + fog)`:

| stock | rms was | rms now | factor |
|---|---|---|---|
| SVEMA_FOTO_32 | 8.5 | 9.617 | 1.1314 |
| SVEMA_FOTO_65 | 11.5 | 13.212 | 1.1489 |
| SVEMA_FOTO_130 | 18.0 | 20.680 | 1.1489 |
| SVEMA_FOTO_250 | 33.0 | 38.766 | 1.1747 |

`verify.py` asserts these render **identically to before C1b** — worst deviation < 0.2 % over
D 0–3, compared against the literal old expression. Their fog values differ (0.28–0.38), so the
*ladder's* internal ratios shift by up to 4 %: that is the price of preserving each stock's own
appearance rather than the ladder's, and appearance is what was chosen.

Two stale claims were also corrected while auditing which values were render-fitted: both the
profile comment and the provenance citation for `EASTMAN_5247_1983` still said *"rms 13.0 is
pipeline-calibrated"*, defending a value that **queue item E0 had already replaced with 5.0** the
previous day on the strength of TI0835's own "rms Granularity: less than 5". Chibisov's printed 5 is
therefore corroborating evidence for an adopted value, not a rejected alternative to a fitted one.

## 5. Deferred, on evidence

**The four VISION3 rms values were NOT re-levelled.** Their traced net-1.0 values imply 1.36–1.70×
(5203 2.6→4.36, 5207 4.2→7.14, 5213 4.6→6.38, 5219 6.6→8.98) while their vector-traced siblings
cluster at 0.91–1.28×. Those four come from the **raster** extractor, and two independent signals
now point at its σ axis rather than at the film:

* the 5219 brochure (vector) reads a near-uniform **1.32× below** the 5219 technical sheet (raster);
* the raster family's implied correction is ~1.5× the vector family's, in the same direction.

Adopting them would bake a suspected ~1.3× calibration error into four profiles. Queue item **C1e**
opens to re-derive `vision3_granularity.py`'s σ ladder against the brochure's vector ladder first.

## 6. Verify: what was replaced rather than repaired

* **The bit-for-bit legacy regression guard is gone by design.** It asserted every unmeasured stock
  reproduced the raw `sqrt(D − dmin + fog)` exactly — false on all 155 stocks after C1b. It was
  replaced with the shape-only form: the sampler must equal that law divided by a single constant
  `sqrt(1 + fog)`, with no dmin term and therefore no per-channel term. A guard that had merely been
  "made to pass" would have hidden the change it existed to protect.
* **New: the level contract.** `grain_sigma` must return exactly 1.0 at net 1.0 for **every stock in
  every channel** — 465 stock-channels, |amp − 1| < 1e-5. This one assertion is what pins the
  meaning of `rms_granularity`.
* **New: the converse.** On a masked stock the sampler must *not* be 1.0 at absolute 1.0, so a
  future "fix" back to the old reference fails loudly instead of quietly re-introducing a
  shadow-referenced level.
* **New: end-to-end, empirical.** Field × amplitude, aperture-integrated at 48 µm, per channel on
  5246 — measured 7.03 / 6.78 / 12.56 within 5 % of the stored triple. The two algebraic guards
  each prove half; only this measures the product the renderer actually puts on screen.
* Two C1-era guards were rewritten to be convention-independent (flat-hold expressed as "the value
  5 D below the toe equals the value at the toe"; the interior peak compared against the stock's own
  reference density rather than absolute 1.0).

## 7. Owner action

* **Rebuild the plugin.** Schema v9. If the C++ grain path calls `FilmGrainSigma()`, re-read its
  calling-convention comment — the v8 instruction to multiply by `sqrt(D − dmin + fog)` at D = 1.0
  is now wrong and double-counts.
* Expect **4–8 % less grain** on 149 stocks, **no channel-balance change anywhere**, the four Svema
  stocks **identical**, and the six re-levelled negatives changed per the table in §3 — mostly in
  the blue record, which is where the measurement disagreed with the old estimate.
