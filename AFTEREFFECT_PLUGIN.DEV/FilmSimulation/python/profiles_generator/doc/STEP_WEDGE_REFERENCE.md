# STEP_WEDGE_REFERENCE.md — the calibration ladders D1/D2 will be read against

**Created 2026-09-02e.** Manufacturer target densities for the Stouffer transmission and
reflection step tablets, harvested from `stouffer.net` at the owner's direction so that a future
scanner-characterisation pass does not have to assume its own step values.

Source: **Stouffer Industries**, `https://www.stouffer.net/Specifications.htm` and the per-product
pages linked from it, fetched 2026-09-02. Reproduced here verbatim.

---

## 0. ⚠ READ THIS BEFORE USING ANY NUMBER BELOW

**This file does not close queue D2, and it was never going to.** D2 asks for *"one step-wedge
scan"* — an actual scan of a physical wedge on the owner's GCMC/UF15 at the film settings. What
this file supplies is the **ruler**, not the measurement. Without a scan there is still no way to
separate emulsion σ from scanner σ, which is the whole point of the row.

What it *does* close is a smaller thing that would otherwise have been guessed at scan time: the
step ladder, its increment, and the base convention.

Three traps, each of which would corrupt a scan analysis silently:

1. ⚠ **THE «% LIGHT TRANSMISSION» COLUMN IS NOT A SECOND MEASUREMENT.** It is exactly
   `100 · 10^−D` restated. Checked on all 21 T2115 rows: worst disagreement **0.033 %**, mean under
   0.01 %, which is rounding to four significant figures and nothing else. Treating it as
   independent data would double-count one number.
2. ⚠ **T3110 PRINTS A SECOND TRANSMISSION COLUMN THAT IS A DIFFERENT SERIES ENTIRELY** — 100, 80,
   64, 50, 40.5, 32, 25, 20, 16, 12.5, 10 … That is a *nominal convenience* ladder (×0.8 per step,
   then rounded "nice" numbers), **not** the wedge's transmittance. At step 2 it reads 80 % where
   the density says 70.79 % — a **13 % error** if mistaken for the real thing. Only the first
   transmission column corresponds to the target density.
3. ⚠ **THE DENSITIES ARE BASE-INCLUSIVE, AND THE PAGES SAY SO OBLIQUELY.** Every table carries the
   footnote *".05 film base is assumed to have 100% transmission"*. Step 1 is therefore **D 0.05
   = the base**, and a wedge step's density *above base* is `D_step − 0.05`. Any fit of scanner
   response against these numbers must state which of the two it is using; this project's own
   density convention is net of base almost everywhere, so the 0.05 must come off.

⚠ **AND THESE ARE TARGET DENSITIES, NOT A CALIBRATION CERTIFICATE.** The pages print no tolerance,
no measurement wavelength, no densitometry status (Visual? ISO Visual Diffuse?), no step or overall
dimensions, and no traceability statement. Stouffer sell individually-calibrated wedges with a
measured certificate as a separate product; **if a wedge is bought for D2, buy the calibrated
version and use its certificate in place of this table.** These figures are the design intent, and
a design intent is a tier-3 number.

---

## 1. Transmission step wedges

Every one of these is the same ladder at a different increment and length, so the choice is only
about increment and range.

| product | steps | increment | density range | best for |
|---|---|---|---|---|
| **T2115** | 21 | 0.15 | 0.05 – 3.05 | ⚠ **the D2 candidate** — one full frame, covers a negative's whole scale |
| T4110 | 41 | 0.10 | 0.05 – 4.05 | finer sampling; the top ten steps are not tabulated |
| T3110 | 31 | 0.10 | 0.05 – 3.05 | finer sampling over a negative's range |
| T4105 | 41 | 0.05 | 0.05 – 2.05 | finest increment, shortest range |
| T2120 | 21 | 0.20 | 0.05 – 4.05 | widest range, coarsest steps |
| T1415 | 14 | 0.15 | 0.05 – 2.00 | short |
| T1015 | 10 | 0.15 | 0.05 – 1.40 | short |

### 1.1 T2115 — 21 steps, 0.15 increment

⚠ The `%T` column is `100 · 10^−D` and is reproduced only because the page prints it.

| step | target D | %T | step | target D | %T |
|---|---|---|---|---|---|
| 1 | 0.05 | 89.13 | 12 | 1.70 | 1.995 |
| 2 | 0.20 | 63.10 | 13 | 1.85 | 1.413 |
| 3 | 0.35 | 44.67 | 14 | 2.00 | 1.000 |
| 4 | 0.50 | 31.62 | 15 | 2.15 | 0.7079 |
| 5 | 0.65 | 22.39 | 16 | 2.30 | 0.5012 |
| 6 | 0.80 | 15.85 | 17 | 2.45 | 0.3548 |
| 7 | 0.95 | 11.22 | 18 | 2.60 | 0.2512 |
| 8 | 1.10 | 7.943 | 19 | 2.75 | 0.1778 |
| 9 | 1.25 | 5.623 | 20 | 2.90 | 0.1259 |
| 10 | 1.40 | 3.981 | 21 | 3.05 | 0.0891 |
| 11 | 1.55 | 2.818 | | | |

*".05 film base is assumed to have 100% transmission"*

### 1.2 T3110 — 31 steps, 0.10 increment

⚠ The page's **third** column is the nominal ×0.8 convenience ladder described in §0 trap 2. It is
reproduced here **only so it can be recognised and refused**.

| step | target D | %T = 10^−D | ⚠ nominal (do NOT use) |
|---|---|---|---|
| 1 | 0.05 | 89.13 | 100 |
| 2 | 0.15 | 70.79 | 80 |
| 3 | 0.25 | 56.23 | 64 |
| 4 | 0.35 | 44.67 | 50 |
| 5 | 0.45 | 35.48 | 40.5 |
| 6 | 0.55 | 28.18 | 32 |
| 7 | 0.65 | 22.39 | 25 |
| 8 | 0.75 | 17.78 | 20 |
| 9 | 0.85 | 14.13 | 16 |
| 10 | 0.95 | 11.22 | 12.5 |
| 11 | 1.05 | 8.913 | 10 |
| 12 | 1.15 | 7.079 | 8 |
| 13 | 1.25 | 5.623 | 6.25 |
| 14 | 1.35 | 4.467 | 5 |
| 15 | 1.45 | 3.548 | 4 |
| 16 | 1.55 | 2.818 | 3.13 |
| 17 | 1.65 | 2.239 | 2.5 |
| 18 | 1.75 | 1.778 | 2 |
| 19 | 1.85 | 1.413 | 1.57 |
| 20 | 1.95 | 1.122 | 1.25 |
| 21 | 2.05 | 0.8913 | 1 |
| 22 | 2.15 | 0.7079 | 0.79 |
| 23 | 2.25 | 0.5623 | 0.63 |
| 24 | 2.35 | 0.4467 | 0.50 |
| 25 | 2.45 | 0.3548 | 0.40 |
| 26 | 2.55 | 0.2818 | 0.31 |
| 27 | 2.65 | 0.2239 | 0.25 |
| 28 | 2.75 | 0.1778 | 0.20 |
| 29 | 2.85 | 0.1413 | 0.16 |
| 30 | 2.95 | 0.1122 | 0.13 |
| 31 | 3.05 | 0.0891 | 0.10 |

*".05 film base is assumed to have 100% transmission"*

### 1.3 T4110 — 41 steps, 0.10 increment

Steps 1–31 are **identical to T3110's first two columns** (0.05 → 3.05). ⚠ **Steps 32–41 are
printed as `na`** — the page tabulates the densities 3.15 → 4.05 but gives no transmission, which
is Stouffer declining to specify above D 3.05. Treat the top ten steps as unspecified, not as
`10^−D` extrapolation.

### 1.4 T4105 — 41 steps, 0.05 increment

Densities **0.05 → 2.05 in steps of 0.05**, transmission 89.13 % → 0.8913 %. ⚠ **This entry is a
GENERATING RULE, not a transcription**: the fetched page returned the ladder as a described
progression rather than row by row, and it was not transcribed cell by cell. The rule is exact and
self-consistent with every other table here, but if a value from this wedge is ever used, re-read
the page first.

### 1.5 T2120 — 21 steps, 0.20 increment

| step | target D | step | target D | step | target D |
|---|---|---|---|---|---|
| 1 | 0.05 | 8 | 1.45 | 15 | 2.85 |
| 2 | 0.25 | 9 | 1.65 | 16 | 3.05 |
| 3 | 0.45 | 10 | 1.85 | 17 | 3.25 |
| 4 | 0.65 | 11 | 2.05 | 18 | 3.45 |
| 5 | 0.85 | 12 | 2.25 | 19 | 3.65 |
| 6 | 1.05 | 13 | 2.45 | 20 | 3.85 |
| 7 | 1.25 | 14 | 2.65 | 21 | 4.05 |

⚠ The transmission column was not returned by the fetch for this product; it is `100 · 10^−D` by
the same rule as every other table, and is deliberately not written out here rather than computed
and presented as if it had been read.

### 1.6 T1415 — 14 steps, 0.15 increment

Steps 1–14 of T2115 exactly: 0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 1.10, 1.25, 1.40, 1.55,
1.70, 1.85, 2.00, with the same transmissions.

### 1.7 T1015 — 10 steps, 0.15 increment

Steps 1–10 of T2115 exactly: 0.05 → 1.40.

---

## 2. Reflection step tablets

Listed on the index page and **not fetched**, because nothing in this project measures reflection
density: `R1215` (12-step), `R1415` (14-step), `R2110` (21-step), `R3705` (37-step). Recorded so a
later reader knows they exist and that skipping them was a decision.

---

## 3. What this is for

**Queue D2** — *"scanner transfer + noise floor: one step-wedge scan (Stouffer T2115 / Kodak
Q-13)"*, still open. When a scan arrives, the analysis is:

1. Locate each step patch and take its mean and standard deviation over a flat interior region.
2. Fit **scanner code value against the target densities above**, base-corrected by −0.05 — that is
   the scanner's opto-electronic transfer.
3. The per-patch **standard deviation is the scanner's own noise floor at that density**, with no
   emulsion in the path at all. ⚠ **This is the number the project has never had**, and it is what
   lets every stored rms granularity be split into emulsion σ and scanner σ.
4. Cross-check against **D1**'s empty-gate frame, which gives the same quantity at D = 0 with no
   wedge in the path.

⚠ **And the reason it matters beyond D1/D2**: queue **C18** wants a saturating form with a
*measured* asymptote for `density_weighting`, the largest undocumented number in the colour path
(0.65 on 36 reversal stocks). Its row states the requirement as *"the §D measurement below"* — that
is this scan. The Dmax cap adopted on 2026-09-02c bounded the expression; only a wedge measurement
can give it the right shape.

---

## 4. Provenance

| field | value |
|---|---|
| source | Stouffer Industries, `stouffer.net/Specifications.htm` and per-product pages |
| fetched | 2026-09-02 |
| tier | **3** — manufacturer *target* densities, no tolerance, no wavelength, no densitometry status, no traceability |
| status | reference data; **written to no film profile and read by no renderer** |
| supersedes | nothing — this is the first calibration-target reference in the corpus |

⚠ Nothing in this file is a property of any film, and nothing in it may be stored on one — the same
rule `SCANNER_CHARACTERISTICS.md` states for the scanner MTF data.
