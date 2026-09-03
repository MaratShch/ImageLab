# Fujifilm cine manual, Kodak H-1-5285, and two Fujifilm-simulation websites

**Date:** 2026-08-14 (second session)

**Sources landed:**
1. `PDF/PROFILES/FUJI/fujifilm_motion_picture_film_manual.pdf` — *FUJIFILM MOTION
   PICTURE FILM MANUAL*, ref. **KB-1101E**, © 2011 FUJIFILM Corporation. 44 pages,
   true PDF with text layer.
2. `PDF/PROFILES/KODAK/Ektachrome_100d.pdf` — *KODAK EKTACHROME 100D Color Reversal
   Film 5285 / 7285*, publication **H-1-5285**, February 2010. 5 pages, true PDF.
   *(Landed at `KODAK/Ektachrome_100d.pdf`, not the `Ectacthome_100D.pdf` path given —
   found by a modification-time sweep.)*

**Websites assessed:** imaging-resource.com "Fujifilm Film Simulations: Definitive
Guide" and fujipic.com "Complete Guide to Fujifilm Film Simulations".

---

## 1. Headline: the first curve in this database that needed no tracing

The Kodak sheet draws its curves as **PDF vector polylines, not as a raster figure.**
The spectral sensitivity plot is three 56-point paths. That means the coordinates are
*exact* — the only estimated step is the axis calibration, and that was fitted by least
squares to the printed tick centres, closing to **0.63 nm** on wavelength and **0.009
log** on sensitivity.

Every other curve in this project came through `digitize_plot.py` with the error budget
that implies. This one did not. Worth remembering as a pre-flight check on any future
Kodak PDF: **look for vector paths before reaching for the tracer.**

## 2. What changed

### 2.1 `KODAK_EKTACHROME_100D_5285` — spectral curves replaced, cross-product borrow removed

The stored curves came from the **5294/7294** sheet — a *different product*, the 2018
Ektachrome reintroduction — borrowed under a declared same-family assumption because no
5285 sheet was on file. The source comment was honest about it, so this was a declared
transfer, not a silent one. H-1-5285 has now landed, so the profile's curves come from
the sheet whose product number it actually bears.

**The old borrow was validated, not merely replaced.** Comparing both peak-normalised on
the same 10 nm grid:

| layer | 5285 peak | 5294 peak (old) | agreement through main lobe |
|---|---|---|---|
| Blue / yellow-forming | 420 nm | 430 nm | within 0.1–0.3 log |
| Green / magenta-forming | 550 nm | 570 nm | within 0.05–0.2 log |
| Red / cyan-forming | 650 nm | 650 nm | within 0.1–0.2 log |

The green "20 nm shift" is not real: that lobe is flat-topped (−0.06, 0.00, −0.02, −0.07
across 540–570 nm), so `argmax` is noise-sensitive. **The declared 5294 → 5285 family
equivalence was sound.**

What 5285's own plot adds is **measured skirts**: active samples go 13/13/13 → **16/15/13**.
The earlier digitisation had truncated real low-sensitivity tails on the red and green
layers to the −4.0 floor.

**Render effect, measured:** derived balance gains at 3200 K move
1.5886/1.0/0.4380 → **1.5731/1.0/0.4256** — about −1 % red, −2.8 % blue. Zero change at
5500 K, the film's own balance, by construction.

Also added from the sheet: `ProcessingSpec(developer="Process E-6", agitation="cine
machine only")`. A colour reversal film has no developer *choice*, so naming the process
fully specifies the condition in a way no black-and-white curve can. Time and temperature
stay "not stated" — they are fixed by the E-6 specification, which is a different
document, and copying them in would be sourcing from something not on file.

### 2.2 `FUJI_ETERNA_VIVID_500T_8547` — verified, and reciprocity documented

The manual **confirms** every value we already held, against a second independent
Fujifilm publication: exposure index 500, balance 3200 K, Status M densitometry.
Nothing needed correcting.

It also supplies reciprocity, stated verbatim and identically for all nine cine stocks:

> "requires no filter corrections or exposure adjustments for shutter speeds of
> 1/1000 to 1/10 second. For exposures of 1 second, open the lens 1/3 of a stop."

Two things follow, and the second is the interesting one:

* `onset_s = 0.1` is **documented**, not inferred — the no-correction range is stated to
  end at 1/10 s.
* **The failure is achromatic.** "No *filter* corrections" is an explicit statement that
  the three records lose speed together. That is the opposite of every Kodak colour film
  in the H-1 reciprocity master table, each of which needs a CC filter (CC10R, CC10M,
  CC15B…). The per-channel spread is therefore set to exactly **zero, and that zero is
  evidence rather than a default.** `verify.py` asserts it.

The exponent is *not* well determined — one non-zero point with onset only bracketed in
(0.1, 1]. Taking the documented 0.1 s gives 1−p = ⅓·ln2/ln10 = 0.100, so **p = 0.90**,
and that is used with the single-point weakness recorded as [C2].

### 2.3 `exposure_index_tungsten` — definition tightened

Both new documents forced a clarification of the field added earlier today. **It is
unfiltered pairs only.**

A monochrome film needs no conversion filter, so its daylight/tungsten pair is a pure
statement about the emulsion. A **colour** film's second index is quoted *with* a
conversion filter, and then the number is dominated by the filter:

- KODAK EKTACHROME 100D: 100 daylight / **25 tungsten with an 80A**. 100→25 is two stops,
  which is simply what an 80A costs.
- Every Fujicolor cine stock, the same way: a **No. 85** for tungsten-type films used in
  daylight, an **80A** for daylight-type films used under tungsten.

Those are **filter factors, not film properties**, and storing them would make a filter
look like a sensitisation difference. Consequently every entry in
`_EXPOSURE_INDEX_TUNGSTEN` is a monochrome stock — a property of the definition, not an
accident of reading order. `verify.py` now asserts it.

---

## 3. The full Fujicolor cine exposure-index table (manual p1)

Extracted from word coordinates. **Recorded here, not entered** — see §5.

**Tungsten type** (secondary index through a Kodak No. 85):

| Film | 35 mm | 16 mm | Tungsten E.I. | Daylight E.I. | Sideprint |
|---|---|---|---|---|---|
| ETERNA Vivid 160 | 8543 | 8643 | 160 | 100 | FN43 |
| ETERNA 250 | 8553 | 8653 | 250 | 160 | FN53 |
| ETERNA 400 | 8583 | 8683 | 400 | 250 | FN83 |
| **ETERNA Vivid 500** | **8547** | **8647** | **500** | **320** | FN47 |
| ETERNA 500 | 8573 | 8673 | 500 | 320 | FN73 |

**Daylight type** (secondary index through a Kodak 80A):

| Film | 35 mm | 16 mm | Daylight E.I. | Tungsten E.I. | Sideprint |
|---|---|---|---|---|---|
| F-64D | 8522 | 8622 | 64 | 16 | FN22 |
| ETERNA Vivid 250D | 8546 | 8646 | 250 | 64 | FN46 |
| ETERNA 250D | 8563 | 8663 | 250 | 64 | FN63 |
| REALA 500D | 8592 | 8692 | 500 | 125 | FN92 |

Also in the manual: intermediate ETERNA-CI (8503/4503/8603), recording ETERNA-RDI
(8511/4511 PET), positives ETERNA-CP 3512/3612, 3514DI/3614DI, 3523XD; exposure
conditions (3200 K or 5400 K for 1/50 s through a Fuji SC-41; 2854 K for 1/100 s for the
positives); Status M densitometry; full edge-marking specifications for 35 mm and 16 mm;
raw-stock storage and X-ray dose data; and standing times after refrigeration.

**Note on our other Fuji entries:** `FUJI_F125_8530`/`8630` and
`FUJICOLOR_SUPER_F500_8572` are **not** in this manual — 8522 is F-64D and 8573 is
ETERNA 500, different products with adjacent numbers. No data was transferred to them.

**Fuji curves are raster**, unlike the Kodak sheet: 20–21 images per page with only axis
frames as vector. Exact extraction is not available for them.

---

## 4. The two websites — verdict

**Neither contains usable physical film data.** Assessed in detail; the reasoning
matters more than the conclusion.

Fujifilm "Film Simulations" (Provia, Velvia, Astia, Classic Chrome, PRO Neg Hi/Std,
Acros, Eterna, Classic Neg, Nostalgic Neg, Reala Ace) are **in-camera JPEG colour
processing presets in Fuji digital cameras.** They are named after films but they are
tone and colour transforms, not emulsions. Nothing in them is a measurement of a physical
film.

| Page | Verdict |
|---|---|
| imaging-resource.com | **Usable only as UI/vocabulary reference.** Explains film physics qualitatively and *shows* spectral-sensitivity and dye-density plots — but as **unlabelled JPEG images**, with no values in the text. Published 2020-08-18, revised 2025-10-02. **Sponsored by Fujifilm** — the author states "I proposed a sponsored article to Fuji in late 2019, and they accepted." Its measurements are Imatest on an X-Rite ColorChecker, i.e. *digital output*, not film. |
| fujipic.com | **No usable data.** Pure recipe blog; every number is a camera menu setting. **Factually unreliable** — contradicts itself on Reala Ace's launch year (2023 in its table, 2024 in its text), misstates a camera's sensor generation, and inflates the count to "18 simulations" by counting monochrome filter variants. Zero citations. **Recommend not citing it at all.** |

The only physical numbers on either page: *"film grains isolated from Fujicolor SUPERIA
100 film; they're only about one micron, or 0.001 mm across"* — which is consistent with
the grain sizes we already model — and structural remarks ("a complex stack of 9 or more
layers"; "NS 160 also had a unique fourth, cyan-sensitive emulsion layer").

**What is genuinely worth keeping** is the mapping from simulation name to real film,
because it prevents a category error. Only three are tied to a specific emulsion with any
confidence: **Acros → Neopan Acros**, **PRO Neg. Std → NS 160**, **Classic Neg →
Superia 100 (1998)**. And explicitly:

> "Fuji's CLASSIC CHROME profile doesn't match any specific film emulsion"

— it merely evokes Kodachrome. Anyone modelling Classic Chrome *as* Kodachrome would be
building on a false premise.

**Four control-design observations** (UI, not physics), recorded because they bear on how
this engine might expose controls rather than on what it computes:

1. **Hue-dependent tone curves.** Classic Neg is described as having a tone curve that
   "varies more between colors in this profile than any of the others" — i.e. per-hue
   tone mapping, not one global curve. Our `dye_matrix` plus a global curve cannot express
   that.
2. **Exposure-dependent grain.** Acros grain is said to change "as you go from light to
   dark areas" — grain amplitude varying with density, which is exactly what our three
   `sigma_shape_*` scalars parameterise and which nothing currently reads.
3. **Grain strength and grain size as separate axes** (Weak/Strong × Small/Large). We
   already store `rms_granularity` and `clump_um` separately; this confirms the split is
   the one users expect.
4. **Independent highlight and shadow shaping** as two controls rather than one contrast
   slider.

Item 2 is the notable one: an independent source describing what a film *looks* like
converges on a mechanism our schema already has data for and does not consume.

---

## 5. Deferred, and why

Eight new Fujicolor cine stocks are fully documented **for exposure index and balance
only** — the manual gives no gamma, D-max, granularity or resolving power, and its curves
are raster. Entering them would create eight profiles where one documented number is
surrounded by estimates. Recorded in `next_week_task.md` with all figures transcribed so
it is authoring work, not research, whenever it is wanted.

Not entered from the Kodak sheet, though now available:

- **Spectral dye density curves** (400–700 nm, peak-normalised, cyan/magenta/yellow plus
  visual neutral) — also vector, also exactly extractable. **We have no field for them**;
  a 3×3 `dye_matrix` stands in. This is the §A.5 L2 gap the Addendum records as "dye
  spectral densities absent", and it is now blocked only by the schema.
- **MTF and diffuse rms granularity curves** — vector, extractable, but our schema holds
  three scalars each rather than curves.
- Characteristic curves (vector) — the profile already has datasheet-derived curves.

---

## 6. Verification

- `verify.py`: **117 PASS / 2 FAIL** — the same two pre-existing failures (saturation
  hierarchy ordering, neighbour-pair coupling). Three new tests: 5285's curves are its
  own with measured skirts; Fuji reciprocity is achromatic; every tungsten index is a
  monochrome stock.
- All 142 profiles and 9 print stocks load and pass `validate_all()`.
- `film_profiles.cpp` and `AlgoSpectralSensitivity.cpp` compile clean, `-std=c++14`.
- `film_names.txt`: 142 lines, 141 pipe separators.
- Generated reports: 142 stocks, 124 citing documents.

## 7. Files changed

| File | Change |
|---|---|
| `film_profiles.py` | 5285 spectral curves replaced; `ProcessingSpec` for 5285; Fuji 8547 reciprocity; tungsten-EI definition tightened; 1 provenance row |
| `verify.py` | +3 tests (above the summary block) |
| `film_profiles.hpp/.cpp`, `film_enum.hpp`, `film_names.txt` | regenerated, both copies |
| `doc/FilmActiveProfiles.md`, `doc/FilmCurves.md` | regenerated |
| `doc/NotFound.md` | 5285 row resolved — the sheet is on file |
| `doc/Found.md` | both documents logged |
| `doc/DIGITIZATION_QUEUE.md` | vector-extraction note; dye-density and MTF curves queued |
| `doc/next_week_task.md` | eight Fuji cine candidates with figures |
| `doc/FilmDatabase_Charecteristics.MD` + Russian mirror | follow-up entry |
| `doc/README.md`, `Readme!.txt` | status entry |
