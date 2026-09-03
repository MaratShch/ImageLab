# Soviet TU batch — extraction record and pending queue

**Date:** 2026-08-17. **Source folder:** `PDF/PROFILES/SOVIET STANDARDS` (read-only).

## Inventory: 11 files, **10 unique documents**

`tu_61762284_..._d.pdf` and `..._d-1.pdf` are **byte-identical duplicates** (md5
`290c906b…`). The two `tu_642151490_…` files are **different documents** (md5 `d5766093…`
vs `b2d5e20d…`) covering ЦО-90Д and ЦО-90Л respectively.

| TU | Pages | Text layer | Film(s) covered | In our database? |
|---|---|---|---|---|
| **ТУ 6-17-622-84** | 9 | **none** (183 MB scan) | **ДС-4** | ✅ held — **entered, see below** |
| ТУ 6-17-691-88 | 22 | OCR | ДС-5М | ✅ added 2026-08-17 (earlier today) |
| ТУ 6-17-1371-86 | 8 | OCR | Фото-65, ЦНЛ-65, ДС-4 (export) | ✅ all held — **provenance entered** |
| ТУ 6-17-1109-88 | 21 | OCR | **ЛН-8** | ❌ not held |
| ТУ 6-17-1443-88 | 23 | OCR | **ЛН-9, ЛН-9С** | ❌ not held |
| ТУ 6-17-1453-89 | 26 | OCR | **ЦНД-64** | ❌ not held |
| ТУ 6-17-912-87 | 23 | OCR | **ЦО-32Д** | ❌ not held |
| ТУ 6-17-1000-88 | 18 | OCR | **ЦО-Т-90ЛМ** | ❌ not held |
| ТУ 6-42-1514-90 (a) | 20 | OCR | **ЦО-90Д** | ❌ not held |
| ТУ 6-42-1514-90 (b) | 20 | OCR | **ЦО-90Л** | ❌ not held |

## Entered into the database

### ДС-4 — `SVEMA_DS_4`, from ТУ 6-17-622-84 (primary specification)

The 183 MB file has **no text layer at all**; every figure was read visually from page
renders. Confirmed on sheet 1: «ПЛЕНКА ФОТОГРАФИЧЕСКАЯ ЦВЕТНАЯ НЕГАТИВНАЯ ДС-4», Свема
Shostka, in force 04.02.1985 → 1990, superseding ТУ 6-17-622-74.

**This replaces a handbook paraphrase with the film's own specification.** Everything
previously stored came from Gurlev 1986, which cites the *superseded 1974 edition*.

| Parameter | Was | Now (ТУ 6-17-622-84, table 4, sheet 4) | Class |
|---|---|---|---|
| Per-layer gamma | 0.82 / 0.80 / 0.79 (b/g/r), a [T3] guess at "Soviet practice" | **b = g = 0.70, r = 0.60** (upper+middle 0.70 ± 0.05, lower 0.60 ± 0.05) | **corrected, documented** |
| Resolving power | 63 lin/mm (Gurlev) | **≥ 68 lin/mm** | **corrected, documented** |
| Development time | "5–8 min" (Gurlev) | **6–8 min**, midpoint 7.0 entered with the developer formula | **refined, documented** |
| Sensitivity balance | not stored | **≤ 2.2** | **new** |
| Fog per spectral zone | 0.25 typical (Gurlev) | ceiling **≤ 0.28 in every zone** — equal across zones, confirming the film is unmasked | **new (ceiling), 0.25 retained** |
| Densitometry | assumed `status_m` | **Status M confirmed** («Макбет», sheets 7–8) | **estimate → documented** |
| Resolving-power method | unstated | **ГОСТ 2819-84**, resolvometer РП-2М | **new** — same method as ГОСТ 24876-81, so like-for-like with our other Soviet figures |
| Deformation temperature | not stored | ≥ 33 °C | new (no schema home; in provenance) |
| Shelf-life allowance | not stored | sensitivity −40 %, fog +50 % | new (provenance) |

**The gamma correction matters and is not cosmetic.** The stored spread had blue steepest
and red shallowest by 0.03; the TU inverts and widens it — blue and green equal at 0.70 with
red a full 0.10 lower. On the apparent conflict with Gurlev's single "gamma 0.8": two things
differ and both are recorded rather than reconciled by fiat — the *edition* (1974 vs 1984,
same designation) and the *quantity* (Gurlev prints one overall figure; the TU specifies the
recommended development **aim** per layer, which is what our curves represent). A permanent
`verify.py` guard now asserts the TU values so nobody "restores" the handbook figures.

### Фото-65, ЦНЛ-65, ДС-4 — from ТУ 6-17-1371-86

⚠ **This document contains no photographic norms table.** It is an export packaging,
marking, acceptance and transport specification for delivery to the Mongolian People's
Republic, and it explicitly **defers every photographic characteristic** to other documents.
Entered as provenance:

- **Guaranteed shelf life: Фото-65 two years; ДС-4 and ЦНЛ-65 twelve months** (sheet 6)
- 35 mm perforated, roll 300 ± 15 m; storage 50–70 % RH, 14–22 °C
- **A lead worth chasing:** ЦНЛ-65's photographic characteristics are governed by
  **ГОСТ 25130-82**, which is *not in this corpus*. That standard would be the primary
  source for the ЦНЛ line, exactly as ТУ 6-17-622-84 turned out to be for ДС-4.

## NOT entered — extracted at OCR level, pending visual verification

Eight stocks are fully specified in these documents and **none is in the database**. Their
figures are recorded below so the reading is not lost, but they are **OCR-level only**: the
scans are typewritten and the OCR repeatedly detaches values from their row labels — on the
ЛН-9 table it interleaved two product columns. Every number entered today (ДС-4, ДС-5М) was
verified visually against the page image first, and these have not been. **They must not be
entered until read visually**, one table at a time.

### ЛН-8 — ТУ 6-17-1109-88, masked colour negative, professional cine
S ≥ 100; balance ≤ 2.0; mean gradients ≈ blue 0.60 / green 0.54 / red 0.50 (+0.06 −0.04);
Dmin blue 0.70–1.05, green 0.25–0.60, red ≤ 0.25; latitude ≥ 1.5; filter-layer efficiency
≥ 1.0; MTF@30 green ≥ 0.30, red ≥ 0.15; **RMS ≤ 19 green, ≤ 21 red**; **red-layer
sensitisation limit ≤ 690 nm**; seven dye-impurity ratios; colour separation 75/40/20,
10/80/30, −5/10/135 (±10, ±20 on the last); any single layer ≥ 60; base ОТБ-14.

### ЛН-9 and ЛН-9С — ТУ 6-17-1443-88, masked colour negative, professional cine
Two variants distinguished by antihalation construction: **ЛН-9 has a colloidal-silver
undercoat, ЛН-9С a carbon-black lacquer backing**, both removed during processing. S ≥ 100
both; gradients blue 0.60 / green 0.54 / red 0.50; **Dmin ceilings differ by variant** —
ЛН-9 1.10 / 0.60 / 0.30, ЛН-9С 1.00 / 0.55 / 0.25; any single layer ≥ 80; uniformity
≤ 15 %; shelf allowance S −30 %, Dmin +0.15 D; colour separation and dye-impurity blocks
present but column-scrambled by OCR.

### ЦНД-64 — ТУ 6-17-1453-89, colour negative photo film, daylight, automated processing
S ≥ 64; balance ≤ 2.0; **recommended contrast coefficient 0.80 ± 0.10 for all layers**;
contrast balance ≤ 0.10; Dmin blue 0.65–0.85, green 0.25–0.60, red ≤ 0.30; latitude ≥ 1.50;
RMS ≤ 20 green and ≤ 20 red; MTF@30 green ≥ 0.30, red ≥ 0.15.

### ЦО-32Д — ТУ 6-17-912-87, colour reversal cine and photo, amateur
Nominal S 32, general S by the reversed image 32–63; balance 1.3–1.8; **gamma upper layer
2.2–2.6, middle and lower 1.8–2.2**; contrast balance lower-to-middle ≤ 0.3; Dmax ≥ 2.2 and
Dmin ≤ 0.25 per layer; useful exposure interval measured between D 0.3 and 2.1; resolving
≥ 68 lin/mm.

### ЦО-Т-90ЛМ — ТУ 6-17-1000-88, colour reversal cine for television, tungsten 3200 K
S ≥ 100; balance ≤ 1.6; **gamma 1.40–1.60 each layer**; latitude ≥ 0.6; Dmin ≤ 0.25 and
Dmax ≥ 2.20 per layer; **resolving ≥ 75 mm⁻¹**; RMS ≤ 25; MTF@30 ≥ 0.27. Notes that the
red-sensitive layer must be the least sensitive and the blue the most.

### ЦО-90Д and ЦО-90Л — ТУ 6-42-1514-90 (two documents), colour reversal, tungsten 3200 K
S ≥ 80; balance ≤ 2.0; overall gamma 1.6–2.2; Dmin ≤ 0.25 (a 0.4 figure also appears —
column ambiguity to resolve visually); Dmax ≥ 2.0 per layer; resolving ≥ 75 mm⁻¹; base
ОТБ-14; supplied as both cine (15/30/60/120 m) and photo film.

## Recommended next step

Read the six norms tables visually, one document per pass, then add the eight stocks. They
are unusually well specified — per-layer gammas, Dmin ladders, MTF, RMS and dye-impurity
ratios all from primary specifications — and would make the Soviet colour section the
best-documented part of the database. Queued in `next_week_task.md`.
