# RESULT 2026-08-30e — the dye matrices are synthetic, the measured ones are the wrong quantity, and ISO 5-3 was in the corpus all along

**Task:** owner reframed the objective — maximum visual fidelity to a real scanned frame — and asked which
missing parameters most improve it. Answer: not the ones being worked on. Then: wire `dye_density`,
the highest-return inert field.

**Outcome:** the derivation is built, audited and **deliberately not adopted**. Build
**0 failures / 0 warnings**, `verify.py` 424 PASS / 1 FAIL (baseline), 23 audits registered.
**Database unchanged.**

---

## 1. ⚠ Every colour `dye_matrix` in this database is synthetic, and symmetric

Not "approximate" — generated. All 97 come from `_dye(k)`, one scalar per stock:

```
KODAK_VISION2_50D_5201   1.0667  -0.0333  -0.0333
GEVACOLOR_NEG_682        1.0600  -0.0300  -0.0300     40 distinct values, one shape
```

⚠ **Symmetric is the defect, not the magnitude.** Real dye crosstalk is asymmetric and always in the
same direction — cyan bleeds heavily into green and blue, magenta into blue, yellow into almost
nothing. A symmetric matrix cannot express any of it, so **before this work every colour stock in the
file crosstalked the same way and differed only in how much.** That asymmetry is precisely the
per-emulsion colour signature, and the schema's own `SpectralDyeDensity` docstring had predicted the
consequence in writing: *"a matrix cannot express an unwanted absorption that peaks off-band, which
is exactly what makes Gevacolor's 550 nm magenta look different from Agfacolor's 540 nm one."*

---

## 2. ISO 5-3 was already in the corpus

I said the Status M and Status A response tables were missing and that I would not invent them. They
were sitting in `PDF/PROFILES/aimm.it2.18.1996.pdf` — **ANSI/ISO 5-3-1995**, public domain, Table 3
(status A) and Table 4 (status M), with extrapolation slopes for the tails. Now transcribed into
`iso_5_3_status.py` and self-checked.

⚠ **Read off the page images, not the text layer, and that was not pedantry.** The scan's OCR floats
two status A red entries free of their wavelength rows, which lands the red peak on 630 nm instead of
620 and shifts the entire red response by 10 nm. The OCR also invents a minus sign in the caption
("−log₁₀"), which if believed would invert every response. Both were caught by rendering the pages
at 200 dpi and reading them.

Half-power widths come out at **20–35 nm**. The project's own spectral basis — Gaussian lobes,
σ = 55 nm — is about 130 nm. That is why the first attempt, which used the project basis as the
analysing response, produced a matrix with a **diagonal of 0.58**: a reader taking 42 % of its red
density from magenta and yellow. Refused. It would have made colour worse while looking principled.

---

## 3. The derivation, and the reason to believe it

For each dye, solve for the amount giving exactly 1.00 density in its own band, then read what that
same amount gives in the other two. Transmittance is integrated and *then* logged — a densitometer
averages light, not logarithms.

⚠ Solving for the amount is what makes the eight `peak_1.0` panels usable at all: their absolute
levels are gone, and any construction needing them would be inventing numbers.

**Validated three independent ways:**

| check | result |
|---|---|
| Four Soviet manufacturing specifications, never used as input | early emulsions land **inside** the m→b band 0.15–0.25 (0.176 / 0.239 / 0.171); later stocks average **1.89× cleaner** |
| The published **neutral** trace on the two panels that carry one | three dyes sum to it at amounts 1.01 / 1.01 / 1.01, fit rms **0.0012** |
| Matrix against the full nonlinear integration of that neutral | agrees to **0.013** density |

⚠ **And it found defects in the source data.** Two of the twelve panels are refused by name:
`EASTMAN_EXR_50D_5245` derives cyan 0.009 into green against yellow 0.030 into red — inverted against
every other panel and all four specifications; `KODAK_VISION2_500T_5218` derives magenta 0.363 into
red, more than twice any other. Both have their dye peaks in the right windows, so the layers are not
misnamed — the traced shapes are wrong at the red end. **Using the data is what exposed it.** The
refusal list is pinned, so a third failure or a silent recovery has to announce itself.

---

## 4. ⚠⚠ And then it must not be adopted

The ten matrices are right about the dyes and are **the wrong quantity for `dye_matrix`.**

`density_metric` says these curves are status M or status A. A status density is what a densitometer
reads from the **whole developed film** — unwanted absorptions included. So the crosstalk is
**already in the stored characteristic curves**, and multiplying those densities by a matrix built
from the same absorptions applies it twice.

⚠ **Which inverts the conclusion of §1: the near-identity `_dye(k)` matrices are structurally RIGHT
and the measured table is structurally WRONG**, however much better sourced it is. What stage 12 may
legitimately hold is

```
    dye_matrix  =  M_reader · M_status⁻¹
```

near identity whenever the reader resembles the declared status — which is the shape the hand-set
scalars already have. They are an aesthetic stand-in for a real quantity, not a placeholder for this
one.

⚠ **`verify.py` caught the double count before the reasoning did.** Adopting the table made *"Agfacolor
Neu is much less saturated than a clean reversal stock"* fail: measured Ektachrome 100D came out
nearly as muddy as the 1936 stock that exists in this file as the muddiness reference. That is what a
double count looks like from outside. A guard written for another purpose turned a shipped defect
into a caught one, and the unit-row-sum contract caught the other half.

The refusal is **enforced, not merely written down**: `_MEASURED_DYE_MATRIX_ADOPTED = False`, and
`dye_matrix_from_spectra.py --assert` fails if anyone flips it. The table looks exactly like something
ready to wire in and is one line away from being adopted by a reader in a hurry.

---

## 5. What this converts into a small named gap

**`M_reader`: the spectral sensitivity of whatever reads the negative.** Zero of the eleven print
stocks carry one; no scanner response is on file. With one, the table above stops being unusable and
becomes half of a correct derivation — and the colour signature of ten stocks becomes real.

That is now the highest-value acquisition in the project, and it is specific: **one spectral
sensitivity curve set for a colour print stock** (Kodak 2383 or 5383 would cover the release path),
or a scanner's channel responses.

---

## 6. Priority ranking, measured rather than argued

Against the owner's objective, from the live database:

| tier | item | state |
|---|---|---|
| 0 | `dye_density` 12 sets | **inert** — this work; blocked on `M_reader` |
| 0 | `reciprocity_table` 30, `emulsion` 17, `print_grain_index` 13, `dye_impurity` 4, `layer_stack` 4 | inert |
| 1 | spectral sensitivity | 76/165 have curves — 89 stocks fall back to `spectral_weights` |
| 1 | grain σ(D) shape | **13/165 measured, and only one is monochrome** |
| 1 | MTF | 16/165 measured |
| 2 | Callier | correct, audited, and inert at the shipped default |

⚠ **And the thing that outranks all of it: there is no ground-truth harness.** Every audit checks the
database against its documents; none checks the render against a photograph. The 90–95 % target is
currently unfalsifiable — no audit here can say whether we are at 60 % or 92 %. Owner has no scans
available today, so priority stays an argument from physics rather than a measurement.

---

## 7. Files

**New:** `iso_5_3_status.py` (ISO 5-3 tables + self-check), `dye_matrix_from_spectra.py` (derivation,
validation, and the enforced refusal), `PDF/PROFILES/iso_5_3_it2_18_1996.txt`.

**Changed:** `film_profiles.py` (`_MEASURED_DYE_MATRIX` table and the block explaining why it is not
applied — **no profile value changed**), `build.py` (two audits registered, 21 → 23).

**Not changed:** every profile, `film_sim.py`, and all C++.
