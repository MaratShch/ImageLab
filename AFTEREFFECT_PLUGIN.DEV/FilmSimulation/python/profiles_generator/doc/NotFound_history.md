# NotFound_history.md — superseded revisions of `NotFound.md`

⚠ **Nothing here is current.** `NotFound.md` is the live gap analysis; this file exists so that
its front matter could be rewritten on 2026-08-31 instead of appended to, without losing the
record of how each correction was arrived at. Read it for provenance of a fix, never for state.

---

## Archived 2026-08-31 — the previous header block of NotFound.md

# NotFound.md — verified gap analysis and data-acquisition guide

**State as of 2026-08-25.** ⚠ This header read "2026-08-20" until a documentation audit on 2026-08-25 while the file already carried 2026-08-25/25b/25c sections — the audit's own first finding, and a reminder that a hand-maintained date line is the least reliable thing in a file. This file is a **research checklist**, not a history: it exists so
that someone can write an effective search query, or a precise request to a manufacturer,
archive, museum, laboratory or standards body, without first re-investigating what the film
*is*. Everything here is either **still missing** or a **hazard that will mislead the next
search**. Anything that has since been found is removed from the body and recorded in
§0 with the date and the source, so this list shortens over time instead of growing. The
layered pre-2026-08-16 history is preserved verbatim in `NotFound_history.md`.

**READ FIRST — what is still open, in one screen:**

| # | Still missing | Why it matters | Best next move |
|---|---|---|---|
| 1 | **9 stocks with no documentation of any kind** (§1) | every parameter is a class estimate | §1 lists a specific query per stock. ⚠ This row said "13" while §1 itself said 9 — the header was stale, corrected 2026-08-20 |
| 2 | **σ(D) for anything outside the Kodak vector corpus** (§0.4) | **13** stocks measured (`sigma_shape_measured`), all Kodak — **11 colour negatives, 1 colour reversal (`KODAK_EKTACHROME_100D_5285`) and 1 B&W reversal (`KODAK_TRI_X_REVERSAL_200`, 2026-08-25b)**. ⚠ Corrected 2026-08-25: this row briefly called TRI-X "the first REVERSAL one", which §0.3.1 of this same file contradicts — 5285 was the first measured σ(D) for a colour reversal stock. TRI-X is the first **black-and-white** one, and the only one whose σ RISES toward dmax; every other σ(D) is an estimate whose heuristic is **wrong in sign** | raster granularity pages already on disk (39), or any non-Kodak σ(D) curve |
| 3 | **MTF curves: 199 vector pages inventoried, 26 curves traced on 12 sheets** (queue C2b) | ⚠ **REVERSED 2026-08-23 — q is NOT a layer-depth constant.** The power law still beats the Gaussian on all 26 traced curves, and the ordering q_R ≤ q_G ≤ q_B holds 8/8 stocks, but the magnitudes are far too spread to be per-layer constants (red 1.89–2.77, blue 2.38–3.42). q therefore **cannot be derived** and stays per-stock measured | trace more COLOUR sheets — each one buys its own stock's q and nothing more; there is no depth constant to harvest |
| 3b | ⚠ **REVERSED 2026-08-23 — the estimated f50 triple is wrong in FORM, not merely in scale** (queue C13) | the old rule scales one number by a fixed layer-order ratio (`f50_r ≈ 0.78 × f50_b`). Measured red f50 is effectively **constant at 36.4 cycles/mm** (range 32.1–41.1, ±13 %) while green spreads 52 % and blue 70 %, so no fixed ratio can be right | partly applied: five modern Kodak cine stocks (5203, 5207, 5213, 5219, 5246) have their RED re-anchored to exactly 36.0, green/blue left at estimates. All other makers and all pre-1990 stocks untouched, and **63 colour stocks still carry an estimated f50 triple** |
| 4 | **A 300+ ppi re-scan of Kino-Technik 1968 Nr. 10, pp. 260/262/264** (queue G5) | the 150 ppi scan cannot separate the three Gevachrome layer curves (they sit 1–2 px apart) | owner holds the source |
| 5 | **One Agfa-Gevaert MTF or resolving-power sheet** (queue G6) | decides whether "lines/mm" on the 682 plot means cycles or half-cycles — a **factor of 2** on 4 stocks | any Agfa-Gevaert image-structure sheet |
| 6 | **Absolute base+fog** (§ D1, one `--empty-gate` frame) | makes density absolute for every stock on that scanner | owner, one minute |
| 6b | ⚠ **NEW 2026-08-20 — a Kodak datasheet for EKTAR 125 (1989–1994)** (queue C14) | the stock is unrepresented and the only document on file is a *magazine review* (PHOTOgraphic, Sept 1989) with **no sensitometry at all**. It documents the eleven-layer construction in detail and not one measured number | a Kodak publication for Ektar 125; the review names no publication code. ⚠ `KODAK_EKTAR_100` (E-4046) is a DIFFERENT, LATER film |
| 7 | **Callier coefficient** — populated on all 160 stocks from three assumed values (§ below) | a documented provenance defect | any densitometer specification stating diffuse-vs-specular ratio |

**Database: 165 film stocks, 11 print stocks, schema v18.** ⚠ 161 -> 165 on 2026-08-30 (queue K1): the four PORTRA NC/VC stocks. Existing indices did NOT move -- the new names are unfrozen and append at 161-164. ⚠ **This line read "schema v15" until
2026-08-27 and was three versions stale**, because `doc_consistency.py` registers the two COUNTS in
this sentence and not the schema number — so the stock counts were guarded and the schema was not.
v15 → v18 across 2026-08-27: v16 `PushSpec`, v17 `EmulsionSpec` + `ThirdPartyObservations`, v18
`ParamSource` + `ProcessVariant` + `DevelopmentProgress` and four `ProcessingSpec` fields. All
additive and inert; each renders bit-identically to its predecessor. ⚠ **The stock count moved on 2026-08-26f, for the first time since the ordering rule was written down:** `KODAK_PRO_100T_PRT` was added from KODAK publication E-29 (April 1999), so `film_enum.hpp`, `film_names.txt` and the generated `.cpp` literals were all regenerated from the database and re-checked for POSITIONAL identity, not merely set equality. The schema moved 12 -> 15 across 2026-08-26 (v13 `DevelopmentPoint.base_fog`, v14 the `SpectralDyeDensity` neutral+D-min pair, v15 `PrintGrainIndex`). ⚠ The print-stock count and the schema moved on 2026-08-25d (queue C15): `KODAK_VISION3_DI_2254` was added and `PrintStock` gained `aging` and `dye_stability`. ⚠ This line read "159 / 9 / v10" until 2026-08-25 and had been stale since 2026-08-23; the live figures come from `build.py`'s build-facts stamp, and `doc_consistency.py` now fails the build if this sentence disagrees with the database. (Was 161 before 2026-08-24, when `FUJI_F125_8630` was removed as a gauge clone; was 159 before 2026-08-24, when `EASTMAN_TRI_X_5223`, `KODAK_8374` and the `KODAK_5302` print stock were added; was 157 before 2026-08-20, when
`KODAK_VISION2_50D_5201` and `FUJI_SUPER_F125_8532` were added; was 155 before 2026-08-19, when
`GEVACHROME_600` and `GEVACHROME_605` were added from Kino-Technik 1968 Nr. 10; was 143 when this
file was first rebuilt on 2026-08-16.)

**Carrier census, 2026-08-26 (KODAK still-film harvest)** — how much of the database is measured
rather than estimated, for the four carriers that are filled one sheet at a time. Every number here
is checked against the live database by `doc_consistency.py`, so a stale figure fails the build
instead of quietly misleading a search:
**12 stocks carry a spectral dye-density set**,
**79 carry a spectral sensitivity set**,
**13 carry a measured σ(D) shape** and
**17 carry a measured MTF**.

⚠ **78 → 79 and 16 → 17 on 2026-08-31 (queues B3 and E3).** `KODAK_TECHNICAL_PAN` gained its first
spectral set from P-255 p9 (B3). `KONICA_IMPRESA_50` became the **17th** measured MTF and the first
that is neither vector-traced nor per-layer: its sheet is a scan end to end, so the curve comes off
the bitmap through `konica_raster.py`, and the panel prints ONE curve captioned *"Densitometry:
Through visual filter"* — so its f50 64.9 is pooled across the layers and written to all three
fields. `verify.py` names it in `_VISUAL_FILTER_MEASURED`, excludes it from the two family guards
that reason about red records, and asserts the pooling that licenses the exclusion.

⚠ **76 → 78 on 2026-08-31 (queue E2).** `POLAROID_52` and `POLAROID_55_PN_NEG` gained vector-traced
pan sets from their own 1999 data sheets. ⚠ Not counted here, because it is not a `FilmProfile`:
**`KODAK_2383_RELEASE` became the first PRINT STOCK to carry a spectral sensitivity** (queue M1),
which is the `M_reader` `dye_matrix_from_spectra.py` had been waiting on.

⚠ **73 → 76 on 2026-08-29 (queue E1).** `AGFA_PORTRAIT_160`, `AGFA_OPTIMA_200` and
`AGFA_OPTIMA_400` gained vector-traced sets from Agfa F-PF-E4 p6–p7. `AGFA_OPTIMA_100` was already
counted and is not counted twice; its set was **replaced** in the same edit, because the 2004 vector
page draws its red record peaking at 615–620 nm where the 2026-08-02 raster reading put it at 650 —
which is where the vector page draws a *shoulder*. Blue and green agree exactly between the two
readings, which is the reason the disagreement reads as a mis-read peak rather than a different
emulsion.

⚠ **THE DYE-DENSITY FIGURE OF 12 IS NOT STALE AND MUST NOT BE "CORRECTED" TO 16.** Four stocks
gained dye data on 2026-08-26 — `KODAK_PORTRA_160`, `KODAK_PORTRA_800`, `KODAK_GOLD_200` and
`KODAK_ULTRAMAX_400` — and none of them counts here, by design. They carry the schema-v14
**neutral + D-min pair**, not three separated dyes, because that is the only shape the KODAK
E-series still sheets publish: one *Midscale Neutral* curve and one *Minimum Density* curve.
`has_data` still means "three dyes"; `has_neutral_pair` reports the pair, and it now stands at
**5 stocks** (was 1). Two different facts, two different counters, deliberately.

The last additions were `GEVACOLOR_NEG_682`'s dye set and `EASTMAN_EKTACHROME_7239`'s sensitivity
set on 2026-08-25d; `EASTMAN_DOUBLE_X_5222`'s MTF on 2026-08-26; and `KODAK_PORTRA_160`,
`KODAK_PORTRA_400` and `KODAK_PORTRA_800`'s MTF on 2026-08-26 — the first STILL films in the
measured-MTF set, and the first read by `kodak_still_curves.py` rather than `mtf_vector.py`.

**Basis.** Corpus reviewed: all PDFs under `PDF/PROFILES` plus 14 at `PDF/` root (~2.7 GB); the
systematic sweep of 2026-08-14 (`reanalysis_2026-08-14/`); the 2026-08-15 reference batch; the
machine inventory (`plot_inventory.py`, which classifies every plot page vector-or-raster and is
re-runnable); and the database's own `provenance.sources` and `description` fields re-queried live
on 2026-08-20.

⚠ **Working-copy caveat, 2026-08-23 — some paths cited below do not resolve here.** The corpus
described above is the full archive. **This working copy holds only `AGFA`, `FERRANIA`, `FUJI`,
`GEVAERT`, `KODAK`, `RETRO` and `SVEMA` under `PDF/PROFILES`**, so every `KONICA/…` path and the
`SOVIET STANDARDS/` folder cited in this file (§0.1, §0.2, §2, §4.1) is **not openable from this
checkout** — the "(present)" annotations record where those files were read, not what is on this
disk. Re-verification of any KONICA or Soviet-standards claim needs the full archive re-staged
first.


---

## Archived 2026-08-31 — the previous "§0 CORRECTIONS TO THE PREVIOUS REVISION"

⚠ **400 lines of corrections to a revision three weeks dead**, in a file whose purpose is to say
what is missing now. The findings that still bind were carried forward into the new §0 of
`NotFound.md` as standing rules; the working is here.

## Rules of this file

1. **"Not found in an earlier pass" is not "not in the corpus."** Every absence claim
   states which documents were read before concluding it.
2. **Nothing is estimated to make a row disappear.**
3. **Uncertainty is marked, not smoothed.** Where a designation, a manufacturer or a
   production date is not established by a document, it says so and says what would
   settle it. A profile name is *not* evidence of a product's existence.
4. Filter-derived exposure indices are filter factors, not film properties; their absence
   is not a gap.
5. **Read §0 before using §1.** Five films were listed as undocumented in the previous
   revision and are not.

---

## 0. CORRECTIONS TO THE PREVIOUS REVISION — read this first

### 0.1 Five films were wrongly listed as having no documentation. They have datasheets, on disk.

The previous revision's §1 said "no manufacturer sheet, standard, book chapter or journal
article anywhere in the corpus names a single measured value for these" and then listed
five films whose citations are sitting in their own profile `description` fields. **Anyone
using the old file would have gone hunting for documents already owned.** Corrected:

| Stock | The source that exists | Where |
|---|---|---|
| `FUJICOLOR_A250` | **Fuji Film Data Sheet MP3-57E**, Fuji Photo Film Co., printed 1980.08 | `PDF/PROFILES/FUJI/FUJICOLOR NEGATIVE FILM A 250.pdf` (1.4 MB, present) |
| `GEVACHROME_902` | **Verbrugghe, R. G. L., "A Sharp Reversal Color Print Film", Journal of the SMPTE** | `PDF/PROFILES/AGFA/Gevachrome902.pdf` (present) |
| `KONICA_CHROME_CENTURIA_100` | Konica datasheet: ISO 100, RMS 11 (48 µm, net D 1.0), reciprocity to 64 s | `PDF/PROFILES/KONICA/chrocen100.pdf` (present) |
| `KONICA_CHROME_R100` | Konica datasheet: ISO 100/32(80B)/25(80A), CRK-2/E-6 | `PDF/PROFILES/KONICA/R100.pdf` (present) |
| `KONICA_CENTURIA_SUPER_400` | Own sheet located 2026-08-16 | `PDF/PROFILES/KONICA/csuper400.pdf` (present) |

These five move to §2 (partial documentation). **What is actually missing for them is
named per film in §2, not "everything".**

### 0.2 A provenance data-integrity defect, 20 profiles — ✅ **RESOLVED 2026-08-18** (2 residual gaps, below)

**The defect.** Twenty profiles carried a `provenance.sources` tuple whose **only** entry was
the placeholder string *"No official manufacturer datasheet available — values estimated from
secondary/historical sources."* `_provenance_for()` derives the tier from the `[T*]` tag in the
description but takes sources from a separate `_PROVENANCE_SOURCES` dict, and nothing tied the
two together — so a profile could claim datasheet grounding in its tier while the queryable
struct said no source existed. Eight did.

| `provenance.tier` / `fitted_from` | count | status |
|---|---|---|
| tier 3 / `analogy` | 12 | ✅ correct — genuinely undocumented, placeholder is the true answer |
| tier 2 / `secondary_sources` | 4 | 2 **fixed**, 2 **unsupported tier claim** — see below |
| tier 1 / `datasheet_curve` | 4 | ✅ all 4 **fixed** — the four in §0.1 |

**Six citations supplied.** Each was lifted from material already inside the profile
(description prose, field comments, or a `spectral.source=` string) — no new research, no new
values. All six documents were re-confirmed present on disk on 2026-08-18 before the citation
was written, so the file's local-archive caveat does not apply to them:

| Stock | Tier | Citation now carried | Document |
|---|---|---|---|
| `FUJICOLOR_A250` | T1 | Fuji Data Sheet MP3-57E, 1980.08 | `FUJI/FUJICOLOR NEGATIVE FILM A 250.pdf` |
| `GEVACHROME_902` | T1 | Verbrugghe, *J. SMPTE* 76(12) 1967, pp. 1198–1201 | `AGFA/Gevachrome902.pdf` |
| `KONICA_CHROME_CENTURIA_100` | T1 | Konica TDS (PDF 2002) | `KONICA/chrocen100.pdf` |
| `KONICA_CHROME_R100` | T1 | Konica TDS (PDF 1999) | `KONICA/R100.pdf` |
| `ILFORD_HPS` | T2 | Иофис 1964 table 7 p79 **+ BBC Monograph 54 (1964) Table I p12 + BBC Report T-101 (1963) Tables 1/2/4** | `SOVIET/Киноплёнки и их обработка.pdf`, `ILFORD/AN ANALYSIS OF FILM GRANULARITY.pdf`, `ILFORD/1963-05.pdf` |
| `KODAK_SUPER_XX_PAN_4142` | T2 | KODAK F-5, August 1979, DS 17 | `KODAK/kodak-professional-b-w-film/` |

Two of the six are worth reading in full rather than trusting the one-line summary, because
each carries a hazard that the citation exists partly to prevent:

* **`FUJICOLOR_A250`** — the corpus contains a *confusable second file*,
  `PDF/PROFILES/FUJI/A 250.pdf`, which is **not** this film's datasheet. It is Yamaryo,
  Ishimaru and Takemura, *SMPTE Journal*, July 1985, about **AX 8514/8512 and LP 8816**. Its
  granularity, CTF and exposure figures — including its 40×40 µm granularity numbers — must
  not be attributed to A250. If you are searching for A250 material, this file will look like
  a hit and is not one.
* **`ILFORD_HPS`** — a **Soviet source for a British film**, admitted under method rule 14
  only because the value it replaced (EI 800) was a self-declared estimate. It grounds
  **exactly one thing**: the speed pair ASA 400 daylight / 320 tungsten. An Ilford sheet for
  HP-S, if one is ever located, **outranks it**. The 1942 Ilford Manual does not count: it
  predates the product, and its "Hypersensitive Panchromatic" is a separate, slower emulsion.

**`KODAK_SUPER_XX_PAN_4142` was a genuine omission, not a gap.** It is the only one of the ten
F-5 stocks in the database whose F-5 citation was missing while its own description named DS 17.
The 2026-08-17 F-5 pass entered it into `_RESOLVING_POWER` and skipped `_PROVENANCE_SOURCES`.
No stored value changed; only the citation was supplied. **Acquisition consequence: nothing to
look for.**

#### 0.2.1 Residual: ✅ **CLOSED the same day — the owner supplied the missing document**

This section was written to record that `FUJI_F125_8530` and `FUJI_F125_8630` (the latter removed
2026-08-24, see §1.5) were deliberately
**not** given a citation, because — as it was worded at the time — "no Fuji F-125 document
existed anywhere in the corpus and neither profile named one". **That held for about an hour.**
The owner then added
`PDF/PROFILES/FUJI/52_509.pdf`, which is the only document in the corpus that names type 8530 —
and it prints a measured MTF for it.

> **本庄 知 [Honjo, Satoru], «動画と静止画 / Moving and Still Images», 日本写真学会誌
> [J. Soc. Photogr. Sci. Technol. Japan] 52(6), 1989, pp. 509–516.** Author affiliation printed
> on p509: 富士写真フイルム(株)足柄研究所 — **Fuji Photo Film Co., Ltd., Ashigara Research
> Laboratories.** So this is a *manufacturer-authored* figure in a learned journal, not a
> third-party estimate. It is a **review essay** on the psychology of viewing moving versus
> still images, **not a datasheet**, and must never be cited as one — but F-125 8530 is its
> worked example.

**What it grounds.** Table 1.2 (printed p511) is headed *Motion Picture Master Nega Film ⇒ Dupe
Nega Film* and footnotes its master negative as `* Fujicolor Negative Film F-125 (8530)`, its
intermediate as `** Fujicolor Intermediate Film FCI (8213)` and its frame as
`*** Frame size: 16.5 mm × 22 mm`. Its single row reads **ν₅₀ = 42 c/mm (Vis.), pel size
11.9 µm, 2.56×10⁶ pels/frame**, with the dupe negative at ν₅₀ 26 c/mm. Fig. 3 (printed p512)
plots five MTF curves, of which **curve 2 is labelled *Shooting Nega Film (E.I. 125)*** — it
overshoots to ≈113 % near 8–10 c/mm and crosses 50 % at the tabulated 42 c/mm. The body text
states the film is represented by the **magenta (G-density) record**, "which occupies the centre
of the layer structure and is visually the most important".

**The correction this forced.** The stored `f50` triple was `70 / 78 / 84` c/mm — an estimate.
It is now **37.7 / 42.0 / 45.2**: a **1.86× reduction**, and the largest sharpness correction in
the database. Only the green figure is measured; red and blue are the previous estimated layer
ratio rescaled to it, and are labelled as such in the profile. ⚠ **That layer ratio is now known
to be wrong in form** (header row 3b): measured red f50 is effectively constant near 36.4 c/mm
rather than a fixed fraction of blue, so 8530/8630's red and blue remain estimates of a
discredited shape. They were **not** re-anchored — the 36.0 re-anchoring is confined to five
modern Kodak cine stocks, and this is Fuji.

**Both tier-2 claims are now supported, the `verify.py` allowlist is empty**, and guard 3 from
§0.2 — "the allowlist must match the placeholder set exactly" — is what caught the change
instead of letting a stale allowlist outlive the gap it documented. That is the guard working as
designed on its first real test.

**Still missing for these two** (see §1.5): everything except the MTF and the speed. A proper
Fuji data sheet for **type 8530/8630 specifically** remains worth acquiring.

> ##### ⚠ 2026-08-23 — the wording above was wrong, and the owner was right to challenge it
>
> The sentence "no Fuji F-125 document existed anywhere in the corpus" is **false as written**,
> and it stayed in this file for five days. `PDF/PROFILES/FUJI/Fujifilm-Super-F-125-8532-35mm-
> Motion-Picture-Film.pdf` is a **complete two-page Fuji product sheet**, Ref. No. **KB-913E**,
> **©1999**, and its own printed title is **FUJICOLOR NEGATIVE FILM F-125** — the *same product
> designation*. It is pure vendor documentation, it was in the corpus the whole time, it is
> cited in full under `FUJI_SUPER_F125_8532`, and since 2026-08-23 its characteristic curves,
> its contrast transfer function and its spectral sensitivity curves are all **traced and
> adopted** (see §1.5a below and the closure note on row `FUJI_SUPER_F125_8532` in header row 3).
>
> **What is actually missing is narrower, and this file must say it at that width:** no sheet for
> **type 8530 / 8630**. 8532/8632 is the *successor generation of the same product line* and
> Fuji's own page says so in as many words — "Announcing the new Fujifilm F-125", "the newly
> upgraded F-125", with a new SUFG hexagonal grain, new Two-Stage Timing DIR couplers and "a
> more linear response curve". A sheet documenting a **different emulsion under the same trade
> name** cannot be lifted onto 8530/8630 without asserting the two are the same film, which that
> page explicitly denies. So the acquisition target is unchanged; only the *claim about the
> corpus* was wrong.
>
> **Why the error was easy to make, which is the part worth keeping.** Two habits combined.
> First, the type number was read as the identity — "8530 ≠ 8532, therefore not an F-125
> document" — when the *product name* on the sheet is the thing a reader looking for "Fuji F-125
> documentation" would search for. Second, and more mechanically: the 8532 sheet's footer,
> product name and logotype are **outlined vector art, not text**, so `get_text()` never returns
> "F-125", "KB-913E" or "©1999" from page 1. A text-layer reading of that file genuinely looks
> like it has no product name and no date — which is also how this file came to state, in header
> row 3, that the sheet carries "no printed date anywhere". Both statements came from the same
> blind spot. **Rule adopted: on a sheet whose typography is outlined, "not in the text layer" is
> not "not printed" — render the page and look.**

**Guards added to `verify.py` (2026-08-18), fault-injection tested:**

1. tier 1 + placeholder-only → hard FAIL, no exceptions
2. tier 2 + placeholder-only → FAIL unless in the two-name allowlist
3. the allowlist must match the placeholder set **exactly** — if one of the two acquires a real
   citation, the check fails, so the allowlist and this section get updated together
4. each of the six closed citations must **name its own document** (a plausible-looking stub
   such as `("Konica, Technical Data Sheet",)` passes guard 1 and is caught here)
5. the A250 misattribution warning and the HPS method-rule-14 caveat must both still be present

All five were verified by injecting the corresponding fault and confirming each tripped its own
guard with no crosstalk.

A **seventh** citation was registered in the same pass, outside the tier ≤ 2 class:
`EASTMAN_5247_1983`, tier 3, which owns Kodak publication **TI0835** (revised 6-93, spectral
plate TI0835C 6-83). It was the only other profile in the database with a citation available to
lift, and leaving exactly one behind would have been worse than closing it. **Its tier was not
changed** — the description carries no `[T*]` tag so `_UNTAGGED_TIER` leaves it at 3, and
whether a profile that owns a manufacturer spectral plate and a printed EI should stay at tier 3
is an owner judgement, not a side effect of registering a citation. **Open question for the
owner, no research needed:** should `EASTMAN_5247_1983` be re-tiered?

### 0.3 ⚠ The "no copy on file" list was wrong: 11 of 12 documents are on disk

**This is the most consequential correction in this revision, because it changes what you
should go looking for.** `film_profiles.py` carried a LOCAL-ARCHIVE CAVEAT, written 2026-07-31,
naming twelve entries whose documents were "not on file" and stating the corpus held 270 PDFs. A
file-by-file re-check on 2026-08-18 found **the corpus holds 448 PDFs (559 files), and 11 of the
12 documents are present.**

| Declared absent | Actually on disk | Text layer |
|---|---|---|
| VISION3 5203 | `KODAK/KODAK-VISION3-50D-5203-7203-technical-information.pdf` | yes |
| VISION3 5207 | `KODAK/KODAK-VISION3-250D-5207-7207-technical-information.pdf` | yes |
| VISION3 5213 | `KODAK/KODAK-VISION3-200T-5213-7213-technical-information.pdf` (+ a duplicate `… (1).pdf`) | yes |
| VISION3 5219 | `KODAK/VISION3_5219_7219_Technical-data.pdf` + a brochure | yes |
| DOUBLE-X 5222 | `KODAK/EASTMAN-DOUBLE-X-technical-information.pdf` | 15 KB |
| PLUS-X 5231 | `KODAK/5231-PLUS-X.pdf` | yes |
| TRI-X Reversal 7266 | `KODAK/KODAK-TRI-X-7266-technical-information.pdf` | yes |
| EKTACHROME 7239 | `KODAK/Kodak Eastman EKTACHROME Film (Daylight) 7239.pdf` | yes |
| EKTACHROME 100D 5285 | `KODAK/Ektachrome_100d.pdf` — **confirmed to be publication H-1-5285**, naming 5285/7285 | yes |
| 5247 | `KODAK/5247.pdf` — **confirmed to be TI0835 "Revised 6-93"**, printing `Tungsten (3200 K) … 125/22` | yes |
| NEOPAN 1600 | `FUJI/datasheet_neopan1600superpresto_en_01.pdf` (19 KB text). The caveat described only `FUJI/Neopan1600.pdf`, which **is** a text-less scan (8 bytes) — but it is not this stock's only file | yes |
| ETERNA Vivid 500 | `FUJI/eterna_vivid500.pdf` (28 KB text) | yes |
| FERRANIA P30 | `FERRANIA/FP3011_Datasheet.pdf` (9 KB text) | yes |
| **CINESTILL 800T** | **no PDF under any spelling — and there never will be.** CineStill publish no technical data sheet for this emulsion at all. ⚠ **DO NOT READ THIS ROW AS A DATA GAP.** This table audits the local PDF archive, nothing else. As of 2026-08-27 CineStill's own sensitometric plot is digitized and `CINESTILL_800T`'s three characteristic curves are **traced, not estimated** — see §7.2c. What is still missing on that profile is listed at §7.2f, and it is not "everything" | — |

Two of these refutations came from inside this project's own work, which is what makes the
caveat's error hard to excuse:

* **5285** — `Ektachrome_100d.pdf` is the very file the adopted dye-density extraction reads
  (`dye_density.py`, page 4). The caveat declared absent a document the code opens.
* **5247** — `KODAK/5247.pdf` **is** TI0835, and its printed `125/22` is the evidence the
  2026-08-18 split of 5247 into `_1974` / `_1983` turns on. The caveat declared absent the
  load-bearing document of a decision made while it was in force.

**✅ The re-verification is DONE, same day.** All eleven were read digit for digit against their
own sheets; full record in `RESULT_2026-08-18c_E0_reverify.md`. Five values changed, three exact
agreements were confirmed and are now pinned by `verify.py`, and two plausible-looking
"corrections" were rejected after checking. What matters for **acquisition** is the last column
below — what these sheets do and do not print:

| Sheet | Prints a numeric rms + resolving pair? | Plots | Acquisition consequence |
|---|---|---|---|
| VISION3 5203 / 5207 / 5213 / 5219 | **No** — method only ("48-micrometre aperture", "multiply by 1000") | **raster** | Nothing left to find. Their granularity/gamma/dmin came off raster curves; improving them means **re-tracing**, not acquiring |
| DOUBLE-X 5222 | **Yes** — 14, 32/100 | raster | Fully grounded. Nothing to ask for |
| PLUS-X 5231 | **Yes** — 10, 32/100 | **vector** | Nothing to ask for; its f50 **has since been traced** — 41.3 cycles/mm, §0.3.1, and it is one of the TWO mono stocks in the **17** that now carry `mtf_measured` -- DOUBLE-X 5222 joined it on 2026-08-26 at 42.2 cycles/mm, 2 % away from PLUS-X's 41.3, which is the first cross-check either of them has ever had (was 8 until `FUJI_SUPER_F125_8532` and `FUJICOLOR_SUPER_F500_8572` were traced 2026-08-23; count corrected 2026-08-25) |
| TRI-X 7266 | **No** — no TOC block at all | raster | Its gamma, rms, MTF and Dmin have **no printed source anywhere**. A different Kodak document would be needed |
| EKTACHROME 7239 | **Yes** — 14, 40/100 | **vector** | Fully grounded |
| EKTACHROME 100D 5285 | **No** — curves only | **vector** | Nothing to ask for; trace the curves |
| 5247 (TI0835) | **Yes** — "less than 5", 50/100 | raster | See §1.1 — the plates are 1983, the text 1993 |
| NEOPAN 1600 | **No** | raster | No reciprocity section exists at all in AF3-608E |

**The single biggest correction it produced:** `EASTMAN_5247_1983`'s stored rms granularity was
**13.0 against Kodak's own printed "less than 5"** — 2.6× above the manufacturer's upper bound, on
the same 48 µm convention. Two independent sources (Kodak TI0835 and Chibisov 1988) now agree on
~5. An earlier decision *not* to adopt Chibisov's figure was correct at the time and is now
superseded, because the objection was about Chibisov's metric and this is Kodak's.

**Only CINESTILL 800T remains absent from the twelve — absent as a FILE.** No PDF exists because CineStill issue no data sheet. That is a permanent property of the vendor, not an open acquisition task, and it is **not** the same as having no documentation: their product page and their published sensitometric figure are both on file and both cited, and the figure grounds the profile's curves. See §7.2.

#### 0.3.1 ✅ What re-reading those sheets then produced (E0b, same day)

Three of them turned out to carry **vector** plots nobody had read. Extracting them changed seven
values across four profiles and, more usefully, showed that the corpus was not the bottleneck:

* **Three more spectral dye-density sets adopted** — `EASTMAN_EKTACHROME_7239`,
  `KODAK_VISION2_200T_5217`, `KODAK_VISION2_500T_5218`. All three had been on a FAILED-extraction
  list, and **none of them needed a better source**: the failures were three defects in our own
  extractor (a caption anchor that merged two stacked plots, an axis calibration with nothing
  checking it, and a stroke-width filter referenced to the thickest path rather than to the
  curves). Film profiles with a dye set: **7 → 10** (→ **11** on 2026-08-25 when `KODAK_VISION2_50D_5201` was read by the ink-based family C; plus the `KODAK_2383_RELEASE` print stock, which this count has never included). Sheets still failing: **2** (5246, 5248).
* **The first measured σ(D) for a colour REVERSAL stock** (5285): 0.15 / 1.00 / 3.10, i.e. σ rises
  ~20× from dmin to dmax — the **opposite sign** to the heuristic that fills 21 reversal profiles.
* **`EASTMAN_PLUS_X_5231`'s f50 measured**: 41.3 cycles/mm against a stored estimate of 60.0.
* **`5285`'s rms granularity**: 3.0 → **13.1**, a 4.4× correction. The old figure had no comment and
  no source, and .003 is where that sheet's granularity curve sits at *dmin* — so the likely origin
  was the curve's toe read as if it were the mid-scale anchor. Sibling reversal stocks whose figures
  *are* printed bracket it: 7239 at 14.0, TRI-X Reversal at 10.0.

**Acquisition consequence:** nothing new to ask for. Full record in
`RESULT_2026-08-18d_E0b_vector.md`.

**⚠ Adjacent-product hazard found in the same sweep, recorded before it can cause a graft:**
`KODAK/KODAK-EKTACHROME-100D-5294-7294-technical-information.pdf` is publication **H-1-5294**
and documents EKTACHROME 100D **5294/7294** — a *different catalogue number* from the held
5285/7285. Both files are named "Ektachrome 100D" and will both look like hits when searching.
**This database holds 5285 only. Do not read the two sheets as one product.**

**✅ The Sehlin/Kennel citation-year conflict is SETTLED — the year is 1985, and the filename is
wrong.** The paper has no text layer, so page 1 was rendered and read as an image:

> Sehlin, R. C., Kennel, G. L., Ortman, E. F., Reinking, F. R., "Choosing Eastman Color Negative
> Film 5247 or Eastman Color High-Speed Negative Film 5294", *SMPTE Journal*, **July 1985,
> pp. 724–734**.

Every page footer prints "SMPTE Journal, July 1985", and the first-page footnote both confirms the
year and **explains where 1983 came from**: "Presented at the 125th SMPTE Technical Conference in
Los Angeles (paper No. 125-40) on November 2, 1983 … This article was received in final form on
September 14, 1984. Copyright © 1985 …". So **1983 is the conference date**, the file
`Sehlin_Kennel_etal_1983_…pdf` is misnamed, and this database's existing 1985 citation was right.
No volume or issue number is printed anywhere in the scan — do not invent one. The
granularity-versus-exposure data queue item E5 wants is in **Figs 7, 8, 9** plus a referenced
Fig. 11.

### 0.4 The VECTOR granularity corpus — ⚠ CORRECTED 2026-08-20: it was NOT exhausted

**2026-08-18, queue item C1c**, claimed **8 sheets** and called the vector corpus exhausted:
5285, 5245, 5246, 5248, 5274, 5279, 5218, and the VISION3 500T brochure as a cross-check. Found by
sweeping every staged Kodak PDF for a rotated GRANULARITY caption on a page carrying zero embedded
images.

⚠ **A NINTH SHEET WAS FOUND ON 2026-08-20** — `Kodak VISION2 50D 5201.pdf` p3, i.e. a sheet that had
been sitting in the same folder throughout. **The sweep did not miss it; the EXTRACTOR refused it**,
and refused it silently enough that "exhausted" looked true:

* its panels are **89 × 90 pt**, below the extractor's 110 × 100 pt frame floor;
* lowering that floor alone let a full-page **308 × 808 pt background box** win the
  "widest qualifying frame" contest, producing a frame with zero density ticks;
* its density tick labels are **typographically jittered** — "1.0" sits 5.0 pt (0.17 D) off its own
  gridline — so the axis fit was not collinear at any usable tolerance;
* it draws its **red record twice**, yellow under magenta, so the panel presents 8 curves where the
  physics says 6 and the count gate refused it.

**The lesson for this file: "the corpus is exhausted" is a claim about a TOOL, not about a corpus,
until the tool has been shown to refuse loudly.** Four independent guards each rejected this sheet
correctly and none of them said "there is a granularity panel here that I cannot read". The sweep
now records square-ness as a property of the figure (measured 172×173 … 89×90 across nine sheets),
and the frame floor is 80 pt.

So **σ(D) is no longer blocked on tooling** — it is blocked on documents. There are three ways
to extend it, in increasing cost:

0. ⚠ **RE-RUN THE VECTOR SWEEP WITH THE FIXED EXTRACTOR FIRST.** One sheet was recovered by
   accident, from a folder review rather than from a sweep. The three defects above were all
   size- or style-related, so any other small-panel brochure in the corpus is a candidate. This is
   now the cheapest σ(D) lead there is and it comes before every item below.
1. **Raster granularity plots already on disk.** `plot_inventory.py` counts **39 raster
   granularity pages** corpus-wide against 101 vector ones. `vision3_granularity.py` already
   reads four of them, so the method exists; each additional sheet is a per-sheet job because
   raster tracing does not generalise the way vector reading does.
2. **Sheets whose granularity is a printed NUMBER but no curve.** These give the level and say
   nothing about the shape. Adopting a shape from a sibling stock would be a class estimate,
   and method rule 18's second half forbids rendering from one.
3. **Documents not held.** A σ(D) curve for any pre-1980 stock, any Soviet stock, or any Agfa
   stock would be new information of a kind the archive currently has none of: every measured
   σ(D) in this database is a Kodak plot.

⚠ **And one thing no document will fix.** The toe anchor is ill-posed in principle: below the
toe the characteristic curve is flat, so density holds at dmin while σ keeps changing, and
σ(D) is multivalued exactly there. Two traces of 5219 — raster technical sheet vs vector
brochure — disagree by 1.7× on the toe while agreeing to 0.02 on dmax and to 0.03 D on the
peak location. **That is not a missing source, and searching for a better one will not settle
it.** It is a limitation of the three-anchor question, not of the answer.

Recorded here rather than in the queue because the actionable part is an **acquisition ask**,
which is what this file is for.

### 0.5 The GEVAERT folder, reviewed 2026-08-19 — three usable documents, one with nothing

The owner added five files under `PDF/PROFILES/GEVAERT`. Reviewed page by page; all quoted values
were read from the page IMAGES, because none of the three journal scans has a text layer at all
(`pdftotext` yields 3–4 bytes each; measured 1 embedded image and 0 curve paths per page, so every
plot in them is raster).

| file | identity as PRINTED in the document | yields |
|---|---|---|
| `Rens_vanBets1968Gevachr6.00.pdf` | J. E. Rens und K. Van Bets, "Gevachrome-Farbumkehrfilme für Farbfernsehen", **KINO-TECHNIK 1968 Nr. 10**, printed pp. **260, 262, 264, 266** (recto pages only — 261/263/265 are not in the scan). Authors at Agfa-Gevaert AG, Mortsel | **two undocumented reversal camera stocks with full specs** — queue G1/G2 |
| `Verpoort_Stapp1980_NewGevacolNeg682.pdf` | A. **Vervoort** and H. Stappaerts, "A New Gevacolor Negative Film Type 682", **SMPTE Journal September 1980, Volume 89, pp. 650–652**. ⚠ Three dates are printed and all are true: presented 22 Oct **1979** (121st Technical Conference), first published April **1980** (BKSTS Journal), reprinted Sept **1980** (SMPTE). The filename's author spelling "Verpoort" is wrong; the paper prints **Vervoort**, which is what the existing citation already says | already cited and partly mined; 3 unmined figures — queue G3 |
| `Webers_Westendorp1979Umkehr.pdf` | Johannes Webers und Kurt Westendorp, "Einführung in die Kopierwerktechnik (XIV)", **FERNSEH- UND KINO-TECHNIK 33. Jahrgang Nr. 7/1979, pp. 245–247**, continuing issue 6/1979 p. 213 | the Gevachrome II process and four type numbers, **no sensitometry at all** — queue G4 |
| `Gevachrome902.pdf` | ⚠ **NOT A NEW DOCUMENT.** MD5 `80ce5885ca35572dc8b458a2a7bcab59`, byte-identical to `PDF/PROFILES/AGFA/Gevachrome902.pdf`, already the cited source for `GEVACHROME_902` (Verbrugghe, J. SMPTE 76(12) 1967) | nothing new |
| `enticknap_2013_film_restoration.pdf` | Leo Enticknap, *Film Restoration: The Culture and Science of Audiovisual Heritage*, 2013 | ⚠ **no Gevaert film-stock data of any kind.** The strings "Gevaert", "Gevacolor" and "Gevachrome" do not occur in the body text; the only hit is a subject-index heading "Agfa-Gevaert, N.V." pointing at printed pp. 66 and 221, where the word actually printed is plain "Agfa" — once in a five-manufacturer list, once in a French bibliography title. Recorded so nobody re-reads 126 pages hoping otherwise |

**⚠ UPDATED 2026-08-19, same day: the first three rows are now DONE or PARTLY DONE.** G1 adopted
`GEVACHROME_600` and `GEVACHROME_605`; G3 traced and adopted the 682 characteristic curves. What
remains from this folder is recorded in the queue as G2 (1968 MTF + spectral, blocked on scan
quality), G3 remainder (1980 Figs 7 / 8 / 11), G4 (Gevachrome II process, no sensitometry to adopt)
and **G5, an acquisition ask: a 300+ ppi grayscale re-scan of printed pages 260 / 262 / 264**. The
150 ppi scan on file is what prevents separating Bild 5a/5b into three layer curves — they sit within
1–2 px of one another — so both new profiles carry a `[T2]` softness transfer and a measured *lower
bound* for Dmax rather than traced values. See `RESULT_2026-08-19_gevaert.md`.

**Coverage note, not an error:** the 1980 paper prints Gevacolor negative history as 16 ASA (1948) →
**Type 655** (first 100 ASA masked, 1968) → **Type 680** (early 1974, ECN-1) → Type 682 (ECN-2). The
database holds `GEVACOLOR_1952` (16 ASA, from the 1948 line) and `GEVACOLOR_NEG_652` (EI 32, from
Cheltsov & Bongard 1958 pp. 179–180) but nothing for **655** or **680**. `652` and `655` are
different types from different decades and different sources — not a transcription error of one
another — so this is a gap in coverage, and neither type has a document on file.

---

### 0.6 The KODAK folder review, 2026-08-20 — five files, three of them already mined

The owner named four new KODAK PDFs and re-uploaded them mid-session; five arrived. **Identity was
checked before anything was read as new evidence, and that check is the main result:**

| file | verdict |
|---|---|
| `KODAK VISION2 50D Color Negative Film 5201 _ 7201 - 125px.pdf` | the **same publication** as `Kodak VISION2 50D 5201.pdf` (H-1-5201, New 10-2005): identical 1719-word text layer and identical vector geometry, 67 paths on p3 matching to 0.01 pt |
| `V200T.pdf` | **byte-identical** to `5274.pdf`, md5 `cf07db7d…` |
| `EASTMAN Color Negative Film 5247 - 125px.pdf` | the same 9-page TI0835 as `5247.pdf` |
| `Kodak Ektar 125 - Jack and Sue Drafahl.pdf` | **new**, and a magazine review — see the §0.6 row below and queue **C14** |
| `bringing enhanced performance to the digital workflow. - Kodak.pdf` | **new**, VISION3 DI 2254/5254 — queue **C15** |

⚠ **A duplicate document is worse than no document if it is not recognised as one**: re-reading a
sheet the database was already built from would have "confirmed" its own numbers. Three of five
here. **Check identity — bytes, then text, then vector geometry — before treating a file as a new
source.**

**What these five leave still missing:**

| For | Still missing | Best next move |
|---|---|---|
| ~~`KODAK_VISION2_50D_5201`~~ **spectral sensitivity + spectral dye density** | ✅ **CLOSED 2026-08-25 (queue C9 + C10), and the acquisition estimate was right — it was extractor work, on a document already held.** Both panels traced off the sheet's own vector paths by identifying each curve by its INK (Kodak draws every trace in the colour of light it concerns; the red record is a yellow-under-magenta overprint of two bit-identical 7-segment paths, which is what the old `n < 8` segment filter had been silently dropping). Dye set peak_1.0 at 450 / 540 / 680 nm, identical to 5217 and 5218, validated by `Neutral − Dmin = k·(C+M+Y)` solving to 0.628 / 0.604 / 0.595 at rms 0.019 D. Spectral set on the 380–680 nm grid, peaks 470 / 540 / 650 nm — the **first vector-traced spectral set in the database**, by the new `spectral_vector.py`. ⚠ Still open on this sheet: **which density** its sensitivity criterion means (§ 2026-08-25c) | for the criterion only: Kodak publication **H-1** *Image Structure*, cited by name on the sheet and absent from the corpus |
| ~~`FUJI_SUPER_F125_8532`~~ **characteristic curves, spectral sensitivity** | ✅ **CLOSED 2026-08-23.** Both panels traced off the sheet's own vector art. Curves fitted to rms 0.005–0.009 D (replacing the 8530 transfer, whose Dmin was ~0.25 D high on every layer); spectral sensitivity stored on the 380–700 nm grid, peaks 469/553/645 nm. Queue **C11** closed for this stock | — |
| ~~`FUJI_SUPER_F125_8532`~~ **printed date** | ✅ **CLOSED 2026-08-23, and the old entry was simply wrong.** The sheet DOES print a date: footer of page 1, `Ref. No. KB-913E (SK·99·05·DT·MW) Printed in Japan ©1999 Fuji Photo Film Co., Ltd.` The footer is **outlined vector art**, so it never appeared in the text layer, and the earlier pass concluded "no printed date anywhere" and dated the document from the PDF creation stamp. `era` moved 2001-2000s → **1999-2000s** | — |
| ~~`FUJI_SUPER_F125_8532` sharpness~~ | ✅ **CLOSED 2026-08-23 by conversion, not acquisition.** The old entry said the square-to-sine conversion "needs the chart's duty cycle". It does not: a *rectangular wave chart* is 1:1, and Coltman's inversion `MTF(f) = (π/4)[C(f) + C(3f)/3 − C(5f)/5 + C(7f)/7 + C(11f)/11 − C(13f)/13 − …]` needs nothing but the curve. Printed CTF crosses 0.5 at **37.78 c/mm**; the converted **sine f50 is 32.07 c/mm** (30.98–33.49 over a ±30–40 % swing of the extrapolated tail), overshoot +9.0 %. ⚠ Red and blue remain flanking transfers — the panel is one unlabelled curve | still worth having: any Fuji sheet printing a **per-layer** MTF for the Super F generation |
| `FUJI_SUPER_F125_8532` **spectral dye density** | ⚠ **NOT a missing document — a schema-shape mismatch.** The sheet's spectral-density panel *is* present and *was* read (both traces are quoted in the profile comment), but Fuji plots only "Typical Densities for a Mid-scale Neutral Subject" and "Minimum Densities" and never separates the three dyes, while `SpectralDyeDensity.validate()` requires cyan **and** magenta **and** yellow. So nothing can be stored under the present schema. The minimum-density trace is the orange mask, falling 0.85 → 0.03 D across 400–700 nm | either a Fuji sheet that plots the three dyes separately, **or** a schema decision to carry an as-printed neutral+Dmin pair (owner's call — it is a struct change) |
| **EKTAR 125** | **everything measured.** The review has no rms, gamma, Dmin/Dmax, MTF, resolving power, spectral data, reciprocity or processing table. It does document the eleven-layer construction in full | a Kodak publication for Ektar 125 (1989–1994). The review names no publication code |
| **VISION3 DI 2254/5254** | nothing, for now — its full technical sheet (`KODAK-VISION3-2254-technical-information.pdf`) is **already on disk and unread**. What the 2-pager adds is the Arrhenius dye-stability table, the first source-backed `AgingSpec` data in the project | read the TI sheet; then decide whether an intermediate film's fade rates may transfer to camera stocks (probably not) |

⚠ **A catalogue-number collision to carry forward:** the database's `EASTMAN_5254_1968` is a 1968 ECN
camera negative. VISION3 **5254** is a digital-intermediate recording film of the same number, forty
years later. Same hazard class as 5294/5285 and as the two 5247 generations.

### 0.2 ⚠ Kodak printed PORTRA 160VC's curves on the PORTRA 100T datasheet. `KODAK_PORTRA_100T`'s figures are not its own. (New 2026-08-26.)

`KODAK_PORTRA_100T` cites **KODAK Publication E-2468** and has done since 2026-07-30. Reading
E-2468's CURVES page on 2026-08-26 established that **none of the figures on it belong to PORTRA
100T**:

| E-2468 panel | Figure id | Also printed on |
|---|---|---|
| Characteristic Curves | `F009_0154AC` | E-190, the **PORTRA 160VC** page |
| Spectral-Sensitivity Curves | `F009_0180AC` | E-190, shared across the whole 160-speed family |
| Spectral-Dye-Density Curves | (unnumbered) | traces identically to 160VC's pair |

Traced independently from the two documents, the characteristic panels return the **same numbers to
four decimals** — dmin 0.2045 / 0.6087 / 0.8121 and gamma 0.5809 / 0.6050 / 0.6691. A tungsten ISO
100 emulsion and a daylight ISO 160 emulsion cannot share a characteristic curve, so this is a
copy-paste defect in Kodak's own publication and not a discovery about the film.

**Consequence, and it is a real gap rather than a completed task:** PORTRA 100T's curves, spectral
sensitivity and dye densities remain ESTIMATES, and the profile still reads `fitted_from="analogy"`.
`verify.py` asserts that it has not silently absorbed 160VC's numbers. What E-2468 *does* supply
uniquely is text, and that is harvested: a five-point reciprocity table (the only multi-point one in
the eleven-document batch), the Status M red aim densities, and a Print Grain Index of 33 / 55 / 84
which **KODAK E-58 (July 2000) page 5 independently confirms**.

**Still wanted:** any Kodak publication that prints PORTRA 100T's own sensitometry. E-2468 is the
only publication code the film has, so this may not exist in print.

