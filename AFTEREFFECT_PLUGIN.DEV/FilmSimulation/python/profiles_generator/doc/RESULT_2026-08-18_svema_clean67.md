# SVEMA_FOTO_65 — confirmed-subset re-run, and three withdrawals

**Date:** 2026-08-18
**Trigger:** the owner disclosed that the scan folder named `SVEMA-FN64` does not
hold one film.
**Outcome:** three adopted `[T2]` values withdrawn, one re-based on a printed
source, two documentation errors corrected, one reusable method rule added.
**Reproduce with:**

```
python analyze_film_scans.py <67 confirmed frames> -o clean67.txt \
       --name "SVEMA-FOTO65-CLEAN67" --px-per-mm 122.7
```

---

## 1. What the owner disclosed, and why it matters

`analyze_film_scans.py v2.1` was pointed at a directory named `SVEMA-FN64` and
analysed **all 509 frames as a single emulsion**. Its output
(`PDF/PROFILES/SVEMA/SVEMA-FN64_generated_film_profile.txt`, header
`Analyzed Frames: 509`, `Native Resolution: 4416x2944`) is a genuine run — the
numbers were never fabricated, and the header matches the values quoted in
`film_profiles.py` down to `dmin_r = 0.0080` / `dmin_g = 0.0041`. **The defect
is the label on the folder, not the analyzer and not the arithmetic.**

The owner confirms:

* frames `PICT0001`–`PICT0067` are **certainly Foto-65**;
* frames 68 onward are a **mixture of Foto-32 and Foto-65** which cannot be
  resolved frame by frame;
* Foto-32 was chosen deliberately, at the time, when finer grain and higher
  resolution were wanted for large prints;
* all frames are the owner's own scans of film he shot and processed between
  1982 and 1994.

The third point is what makes this worse than a small sample. The
contamination is **one-directional**: the mixed batch reads *finer and
sharper* than Foto-65 alone, because the contaminant was selected for exactly
those properties. A random mixture would add variance that averages out; this
one adds bias that does not.

**Independent confirmation of the direction, before the re-run:** grain
correlation length measured directly over the two pools gave frames 1–67 at
**4.34 px (35.4 µm)** against frames 68+ at **3.67 px (29.9 µm)** — the
confirmed pool is ~18 % coarser, as predicted. There is **no bimodality**
(within-pool scatter ±1.3 px versus 0.67 px between pools), so individual
frames in the tail cannot be classified after the fact. The mixture is
permanent.

---

## 2. The measurement that decides most of it

Over **all 67** confirmed frames:

```
max |R − G| = 0        max |B − G| = 0
```

They are **exactly greyscale**.

Every per-channel quantity in the 509-frame output therefore originates
entirely in the contaminated 68+ tail:

* `base_tint` (0.991, 1.000, 0.991)
* `tone_slope_r` −0.0205, `tone_slope_b` +0.0079
* all twelve `Crossover` bins (up to ±0.0016)
* the per-channel gamma spread 0.806 / 0.834 / 0.850

None of it is attributable to this emulsion. The analyzer says the same thing
from first principles in its own `[SpectralResponse]` section: developed
silver is spectrally near-neutral, so **a scan of a B&W negative carries no
memory of which wavelengths exposed it**, and any channel asymmetry is the
scanner's illuminant and white balance.

This check costs seconds. It voided four apparently-precise parameter
adoptions. It is now method rule 17's corollary in `DIGITIZATION_QUEUE.md`.

---

## 3. Confirmed 67 versus mixed 509, parameter by parameter

Same script, same version, `--px-per-mm 122.7` (the mixed run was executed
*without* `--px-per-mm`, so its µm figures were converted by hand at 122.7 —
legitimate arithmetic, but worth knowing).

| Parameter | mixed 509 | confirmed 67 | reproduced? |
|---|---|---|---|
| `gamma_g` | 0.834 | **0.677** | no — −19 %, both `[ESTIMATE]` |
| σ(D) green toe/mid/dense | 0.0191 / 0.0292 / 0.0482 | **0.0479 / 0.0425 / 0.0435** | **no — sign flips** |
| → normalised shape | 0.65 / 1.00 / **1.65** rising | 1.13 / 1.00 / **1.02** flat | — |
| `corr_len_px` | 3.42 | 3.63 | direction yes (+6 %) |
| `clump_um` (raw) | 28.0 (hand-converted) | 29.6 | direction yes |
| `base_tint` | (0.991, 1.000, 0.991) | (1.000, 1.000, 1.000) | **no** |
| `tone_slope_r` / `_b` | −0.0205 / +0.0079 | **0.0000 / 0.0000** | **no** |
| crossover, 12 bins | up to ±0.0016 | exactly 0.0000 | **no** |
| `anisotropy` | 0.658 | 0.634 | **yes** |
| halation `strength_g` | 0.1992 | 0.1656 | direction only (−17 %) |
| `coating_sigma_d` | 0.0643 | 0.1272 | see §5 |
| `vignette_d` | 0.1000 | 0.1317 | see §5 |
| `dmin_g` (relative) | 0.0041 | 0.0281 | n/a — relative to scanner white |

### The σ(D) disagreement is not a binning artefact

The obvious objection is that the two runs' density bins might sample
different densities. They do not. `analyze_film_scans.py` defines them as
**absolute offsets from `d_base`**:

```python
bins = {"toe":   (d_base + 0.05, d_base + 0.35),
        "mid":   (d_base + 0.35, d_base + 0.95),
        "dense": (d_base + 0.95, DENSITY_CEILING - 0.4)}
```

`d_base` is 0.0041 on the mixed run and 0.0281 on the confirmed run — a
difference of **0.024 D**, against bin widths of 0.30 and 0.60 D. The two runs
integrate over essentially the same absolute density windows. The
disagreement is real.

### Stated as unexplained

Confirmed-67 σ_toe (0.0479) is **2.5×** the mixed batch's σ_toe (0.0191).
Adding 442 frames should not move a toe-bin statistic that far. Candidate
explanations were considered and none survives: the bins are comparable
(above); the confirmed pool has *fewer* near-base pixels, not more; block
counts are of the right order (10,720 flat blocks used of 92,460 seen). It
may be that the tail contributes a large population of very-low-noise thin
blocks that dominates a pooled statistic, but that is a hypothesis, not a
finding.

**This is precisely why the shape was withdrawn rather than replaced.**
Neither run has earned adoption. The estimator is scanner-noise dominated by
the analyzer's own admission — *"Scanner noise adds in quadrature: treat as
upper bounds"* — and the method rule is that a conflict is **recorded, never
averaged**.

---

## 4. Adopted changes

| Field | Was | Now | Tier | Basis |
|---|---|---|---|---|
| `base_tint` | (0.991, 1.000, 0.991) | **(1.0, 1.0, 1.0)** | — | confirmed frames are exactly greyscale |
| `silver_tone` | **+0.40** | **0.0** | `[T3]` | the sign reversal rested on `tone_slope_r −0.0205`, which is 0.0000 on the confirmed frames |
| `sigma_shape_toe` / `_dmax` | 0.65 / 1.65 | **withdrawn → 0.4 / 1.0 / 1.2** | `[T3]` | the two runs disagree in sign; conflict recorded |
| `gamma` **basis** | 509-frame batch | **Gurlev 1986 p296** (γ_rec 0.8) | `[T2]` | method rule 14. **Value 0.830 unchanged** |
| `anisotropy` **rejection reason** | "Bayer mosaic, DSLR scan" | **open question** | `[T3]` | device is a GCMC/UF15 scanner. **Value 1.10 unchanged** |
| `_FITTED_FROM` tier note | measured batch **and** printed source | printed source alone carries tier 2 | 2 | the batch is demoted to a consistency bracket |

Unchanged, and why:

* **gamma 0.830.** Gurlev 1986 (book p296) prints γ_rec **0.8 (CT-2)** for
  Svema Foto-65. Under method rule 14 a printed source outranks a derived
  estimate, so it becomes the primary basis and the scan statistics become a
  bracket — a wide one: **0.677** confirmed against **0.834** mixed, both
  resting on an *assumed* 1.90 logE interdecile scene span, which is an
  assumption and not a measurement. 0.830 is kept because it is the value
  nearest Gurlev inside that bracket. **Tier 2 survives on the printed source
  alone**, which is the point of having ranked the sources.
* **clump 23 µm.** Mixed 3.48 px → 28 µm raw → ~23 µm after deconvolving a
  ~2 px scanner PSF. Confirmed 3.63 px → 29.6 µm raw → ~24.7 µm. The
  contamination bias is visible and in the predicted direction, but the shift
  is inside this figure's stated uncertainty.
* **halation** gains 0.09, radii (12, 69, 320) µm. The confirmed subset (19
  usable highlight frames) measures 0.166 D excess against 0.199 D — 17 %
  lower, one direction, and far inside the 4-to-7-stop highlight-overshoot
  assumption that already sets this parameter's `[T2]` tier. It is also a
  **single-channel scalar**, so unlike `base_tint` and `silver_tone` it never
  depended on the per-channel structure that turned out to be artefact.
* **rms 11.5** stays `[T1]` — fitted through the full pipeline, and both runs'
  native-resolution mid σ bracket the 0.030 that fit produced (0.0292 mixed,
  0.0425 confirmed).

### Two decisions inside the withdrawals worth stating explicitly

**`silver_tone` was set neutral, not restored to the earlier −0.10.** The
whole trail — −0.25 from one frame, −0.10 from the 355-frame batch, +0.40
from the 509 — rests on per-channel density drift in scans from the same rig,
which is the same class of artefact throughout. Restoring −0.10 would swap
one unsupported number for another. Image tone of a developed silver negative
is a **real physical effect** (developer chemistry, grain size, sulphiding),
and this is not a claim that Foto-65 is neutral. It records the **absence of
any admissible measurement**.

**The σ(D) fallback 0.4 / 1.0 / 1.2 is the defensible default here, not an
embarrassment.** `_grain_v2` fills a *rising* triple for non-reversal stocks,
and for a B&W **silver** negative σ ∝ √D is the textbook Poisson-counting
result — it rises. The *falling* triples adopted for the four Vision3 stocks
on 2026-08-17 are measured on **chromogenic** colour negatives, a different
mechanism (see the `GrainSpec` docstring). **That sign must not be transferred
to this stock.**

---

## 5. Two figures that got worse, and are not evidence of anything

`coating_sigma_d` 0.0643 → 0.1272 and `vignette_d` 0.1000 → 0.1317 both
roughly doubled on the confirmed subset. This does **not** mean Foto-65 has
worse coating unevenness than the mixed batch suggested.

Both are `[UPPER-BOUND]` figures, and the analyzer says why in its own output:
*"UPPER BOUNDS: scene content leaks in unless the batch has many varied
compositions."* 67 frames from a single shooting period are far less varied
than 509 frames spanning twelve years. **The bound got looser, not the film
worse.** No change was made to either parameter, in either direction.

---

## 6. The second documentation error: the scanner

Four documents — `DATASHEET_VERIFICATION_REPORT.md`, `REPORT_FN64_355.md`,
`FilmDatabase_Charecteristics.MD` and the `film_profiles.py` comments —
described the source as a **"Bayer-demosaiced DSLR"** rig. EXIF on the owner's
frames reads:

```
Make      GCMC
Model     Scanner
Software  UF15 16/08/20 v0.69
```

4416 px / 36 mm = **122.7 px/mm = 3116 dpi**, 1 px = **8.15 µm**; 8-bit JPEG,
q≈90, 4:2:2 chroma subsampling.

This is not cosmetic. The stated **reason** for rejecting `anisotropy`
0.62–0.66 was that it must be the sensor mosaic of a Bayer-demosaiced DSLR
scan. There is no established Bayer pattern to blame, and the measurement is
**reproducible** — 0.658 on the mixed frames, 0.634 on the confirmed. Something
anisotropic is real in these files. Candidates are scanner line/transport
structure and film transport smear, and nothing distinguishes them from these
files alone. The stored 1.10 (a transport-smear estimate) stands as `[T3]`,
and the question is now logged as **open**, not settled.

Note also that 4:2:2 subsampling means the chroma planes are horizontally
half-resolution — a further reason not to read per-channel structure out of
these JPEGs even where it is non-zero.

---

## 7. Knock-on caveats recorded, not re-fitted

Two `SVEMA_FOTO_250` values lean on the contaminated batch as a comparator.
Neither is changed, because in both cases the bias pushes in the direction the
existing adoption already went, and re-fitting on a 26-frame batch would trade
a documented weakness for an undocumented one.

* **gamma 0.85** — adopted on the reasoning that FN250's batch (0.844)
  measures the *same* slope as FN-64's 0.834, "not higher", contradicting a
  recollection of more contrast. That comparator is the mixed batch; the
  Foto-65-only subset gives 0.677. The conclusion holds directionally but is
  weaker than it reads.
* **rms 25.0** — fitted from a flat-region σ ratio, FN250 0.0502 against
  "SVEMA_FOTO_65's 0.0299" over 3 supplied scans. Which 3 frames were used is
  not recorded, so the denominator may be biased **low** (Foto-32 is finer),
  which would make the true ratio **smaller** than 1.68×, not larger. The
  shipped value was already capped far below what a literal fit demanded
  (~70), so this bias pushes the same way the cap did.

---

## 8. Regression guards

`verify.py` gains 4 checks. Result **169 PASS / 2 FAIL**; the 2 failures are
the long-standing saturation-hierarchy and neighbour-pair-coupling ones,
byte-identical before and after, and were not touched.

| Check | Guards against |
|---|---|
| `base_tint` stays identity | re-adopting (0.991, 1, 0.991) from the old reports |
| `silver_tone` stays 0.0 | re-adopting the +0.40 reversal |
| `sigma_shape` is the B&W default and rises | re-adopting *either* scan run's triple |
| the provenance-warning block is present in `film_profiles.py` | the warning being edited away |

The last check asserts **prose**, deliberately. Without the mixed-batch
warning the next reader sees "509-frame batch" and reasonably treats it as one
emulsion — which is exactly the mistake this correction fixes. A comment that
load-bearing deserves a test.

---

## 9. What would actually settle these — three scans, none requiring new film

Ranked by value per unit of effort. The owner has no film camera and no
accessible processing lab, so nothing here involves shooting.

1. **One `--empty-gate` frame** — a scan of the empty gate with no film in it.
   Makes every density **absolute** instead of relative to scanner white, which
   is the blocker on `dmin` (still `[T3]` 0.16) for every stock scanned on this
   device. **Free, one minute, highest value on the list.**
2. **One step-wedge scan** (Stouffer T2115 or Kodak Q-13) — characterises the
   scanner's own transfer curve and noise floor. This is what currently makes
   the σ(D) estimator uninterpretable, and a uniform patch on a known target
   would decide the anisotropy question outright, which no photographic frame
   can do. Cost: the price of one wedge.
3. **A ±4 EV grey-card bracket with `--wedge`** (filenames carrying the offset,
   e.g. `frame_-2EV.jpg`) — replaces the entire gamma bracket (0.677–0.834,
   both resting on an assumed 1.90 logE scene span) with one **MEASURED**
   gamma, toe and shoulder. Ten frames.

---

## 10. The transferable lesson

Recorded as method rule 17 in `DIGITIZATION_QUEUE.md`:

> **A folder name is not an emulsion identity, and a batch statistic inherits
> every film in the batch.** Ask which frames are certainly which stock, and
> write the frame **range** into the provenance, not the folder name. A mixture
> is worse than a small sample, because its bias is invisible and — where the
> films differ deliberately — one-directional rather than noise that averages
> out.
>
> **Corollary, the cheap half:** check the channels before trusting any
> per-channel number. One line of code voided four adoptions here.

The failure was not in the analyzer, the arithmetic, or the owner's memory.
Every quoted number was a real measurement of a real thing. The chain broke at
the one link nobody tested: **what the measured thing actually was.**
