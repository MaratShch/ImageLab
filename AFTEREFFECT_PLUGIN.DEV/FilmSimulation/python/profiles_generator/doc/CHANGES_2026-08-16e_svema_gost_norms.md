# SVEMA Foto line — Zhurba Table 2 vs the stored values, investigated and resolved

**Date:** 2026-08-16. **Trigger:** the owner asked whether the vertical table in the
supplied Zhurba-1990 screenshot had been used to fine-tune the SVEMA films.

## The honest starting answer

**No.** Table 2 (book p46) had been recorded in `provenance.sources` as consistency data
only, on the judgement that it describes a later generation. No SVEMA parameter was
changed by it. When the owner asked, that judgement was re-examined rather than defended —
and the numbers turned out to disagree in a way that needed explaining:

| | stored | Zhurba Table 2 norm | first reading |
|---|---|---|---|
| Resolving power, lin/mm | 135 / 110 / 100 / 82 | ≥ 200 / 150 / 110 / 100 | all four *below* the minimum |
| MTF at 30 mm⁻¹ (from stored f50 42/35/31/30) | 0.70 / 0.60 / 0.52 / 0.50 | ≥ 0.80 / 0.80 / 0.80 / 0.70 | all four fail |
| RMS granularity (1000·σ_D) | 8.5 / 11.5 / 18 / 33 | ≤ 20 / 25 / 25 / 35 | consistent |
| Average gradient | 0.80 / 0.83 / 0.80 / 0.85 | 0.8–1.1 | consistent |

A film cannot typically be *below* the minimum of its own standard, so one of the two
readings had to be wrong.

## The resolution, from a primary source already on disk

`PDF/PROFILES/SOVIET STANDARDS/ГОСТ 24876-81.pdf` — the technical-conditions standard for
these very films. Its Table 6 exists in **three successive versions** inside the one
document, and the standard states the rule itself:

> «Нормы, указанные в скобках, вводятся с 01.01.90.»
> *(The norms shown in parentheses take effect from 1990-01-01.)*

| Table 6 version | Фото-32 | Фото-64/65 | Фото-125/130 | Фото-250 |
|---|---|---|---|---|
| **Original, 1981** (p8) — high/first quality category | R ≥ **135**/110 | 110/100 | 110/100 | 100/90 |
| **Amended** (p23), new-scale names | R ≥ 145 **(200)**; MTF ≥ 0.60 **(0.80)**; RMS ≤ 35 **(20)** | 110 **(150)**; 0.60 **(0.80)**; 45 **(25)** | 100 **(110)**; 0.50 **(0.80)**; 45 **(25)** | 90 **(100)**; 0.50 **(0.70)**; 50 **(35)** |
| **Later amendment** (p27), renamed ФН-32/64/125/250 | R ≥ 195; MTF ≥ 0.80; RMS ≤ 20 | 145; 0.80; 25 | 110; 0.80; 30 | 100; 0.70; 35 |

**Zhurba 1990 Table 2 prints the parenthetical set exactly** — 200/150/110/100,
MTF 0.80/0.80/0.80/0.70, RMS 20/25/25/35 — because the book is the 1990 edition and those
norms came into force on 1 January 1990.

**Our stored values are the original 1981 top-category norms**, which is correct for the
generation these profiles model. Stored R 135 for Фото-32 *is* the 1981 high-category
minimum; stored f50 42 gives MTF 0.70 at 30 mm⁻¹ against the 1981 requirement of ≥ 0.60 —
compliant.

## Why this is not a test-object-contrast artefact

Page 14 of ГОСТ 24876-81 names the measurement method: «Разрешающую способность плёнок
определяют по ГОСТ 2819—84». The **same method standard governs both norm sets**, so the
figures are like-for-like. The difference is a raised requirement for a new emulsion
generation — not a difference in how resolving power was measured. (ГОСТ 2818-91, also on
disk, is the *spectrosensitometric* method and is unrelated to this question.)

## What changed

- **No film parameter was modified.** The investigation confirmed the stored values.
- Provenance on `SVEMA_FOTO_32/65/130/250` rewritten: the vague "post-1987 successor
  generation" wording is replaced by the full three-tier norm history with the 01.01.90
  effective date, the page numbers, and the explicit statement that no value was changed.
- **New permanent guard** in `verify.py`: the Foto line must satisfy its own era's norms
  (R = 135/110/100/82, MTF at 30 mm⁻¹ ≥ 0.60/0.60/0.50/0.50) and must not drift upward
  into the 1990 set — because doing so would silently re-date the stocks.

## What Table 2 additionally documents (recorded, no schema home)

Development time to γ = 0.80 (6–10 / 6–10 / 8–10 / 8–11 min) and to Ḡ = 0.62 (4–8 / 4–8 /
6–9 / 6–9 min); general sensitivity ranges at both contrast criteria; effective sensitivity
through ЖС-18 / ОС-14 / КС-14 filters (filter factors — per project rule not film
properties); latitude minima 1.8/1.6/1.6/1.5; fog ≤ 0.02/0.04/0.06/0.08 (note: GOST fog is
density *above base*, which is not the same quantity as our `dmin`, so it is not compared);
maximum contrast coefficient 1.0–1.3; emulsion deformation temperature ≥ 60 °C; guaranteed
shelf life 30 / 24 / 24 / 12 months.

## Verification

`verify.py` **126 PASS / 2 FAIL** (the two long-standing failures). C++ regenerated;
reports regenerated; `validate_all()` green.
