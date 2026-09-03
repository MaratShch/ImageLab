# next_week_task.md — stocks to add next week (owner decision 2026-08-13)

Owner: "Rest of the films - add on next week." This is the remainder of the
Tier-A candidate list (own datasheet in the archive, text-readable) after
today's batch, plus the queued siblings from today's sheets.

## 0d. ⚠ STATUS 2026-08-31 — this file is a CANDIDATE POOL and nothing in it is a commitment

Read `PROGRESS.md` first, then `DIGITIZATION_QUEUE.md` §0. **This file has not driven the work
since 2026-08-13** and every count in the sections below predates six weeks of change. It is kept
because the per-stock *candidacy* reasoning is still useful when a new profile is proposed; it is
not a plan, and no list in it should be read as outstanding work.

* **Stock count is 172** (re-read 2026-09-02), and the live per-stock coverage lives in
  `FilmActiveProfiles.md`, which is regenerated on every build. ⚠ **THE «EVERY ADDITION SHIFTS THE
  LISTBOX» WARNING BELOW IS NO LONGER TRUE, AND HAS NOT BEEN SINCE `film_ids.lock` FROZE THE ORDER.**
  A new stock is appended at the next free frozen id and no existing index moves — the four AGFA
  stocks (166–169), `SVEMA_CO_90L` (170) and `FUJI_NEOPAN_SS` (171) all landed that way, and
  `verify.py` asserts the append on each. What still holds is that additions land one owner-approved
  batch at a time, because each is a decision about evidence rather than about indices.
* ⚠ **THE LIVE NEW-PROFILE CANDIDATES ARE IN QUEUE ROW T3, NOT HERE.** As of 2026-08-31 the corpus
  sweep found datasheets on the owner's machine for **PROVIA 100F**, **SUPERIA X-TRA 400** and
  **PRO 400H**, none of which exists in the database. Genuinely absent and still wanted: FUJI ETERNA
  500T and 250D, and KODAK EKTACHROME E100 (2018 relaunch).
* ⚠ **ADDED 2026-09-02: FUJI NEOPAN S and NEOPAN SSS are now candidates too, and they are cheaper
  than they look.** The Fuji sales guide the owner supplied with AF3-411E(N) prints **ASA 50 / 100 /
  200** for S / SS / SSS and carries a per-film γ-versus-development-time panel and a per-film
  characteristic-curve family with numeric axes, all Minidol 20 °C tank — enough for tier 2 on both.
  `FUJI_NEOPAN_SS` itself was added from its own datasheet on the same day (queue N1).
  ⚠ **What must NOT happen when they are added**: the four granularity measurements this corpus
  holds under the names Neopan S / SS / SSS are from Ooue 1959 and Takano 1969 and describe the
  coatings of that era. Joining them to a modern sheet is the trade-name trap already on file for
  EASTMAN_5247 and ILFORD PAN F. See `EMULSION_KNOWLEDGE_BASE.md` §23m.3.

## 0c. ⚠ STATUS 2026-08-23 — superseded by 0d above, kept for its reasoning

* **Stock count is 159**, not the figure any list below implies. Two stocks were added on
  2026-08-20 (5201, F-125 8532) and none since — every addition shifts the plugin's ListBox, so
  they land one owner-approved batch at a time.
* ✅ **MTF is no longer "one estimate scaled by a ratio"** (C2b/C24): 8 stocks carry a measured
  f50 triple, 26 curves on 12 sheets are traced, and the old estimating rule was measured wrong
  in FORM — red f50 is effectively constant at 36.4 cycles/mm, so five modern Kodak cine reds are
  now anchored at 36.0. 63 colour stocks still carry an estimated triple.
* ✅ **Reciprocity is wired and live** (C8): `exposure_time_s` / `exposureTimeS`, 0 = inert,
  21 measured tables (up from 6) plus 105 Schwarzschild exponents. It was the last inert data
  family; nothing in this file's wish-lists about it is outstanding.
* ✅ **Per-layer grain rms is measured on 11 stocks** (C1e), and the tier-2 stack ladder
  (blue 1.30×, red 1.10× of green) is contradicted by all nine sheets that measure it —
  deliberately not rescaled, because all nine are Kodak cine negatives.
* ⚠ **The corpus in this working copy is partial**: only AGFA, FERRANIA, FUJI, GEVAERT, KODAK,
  RETRO and SVEMA are present under `PDF/PROFILES`. Any entry below that depends on KONICA or
  the Soviet-standards folder cannot be actioned here at all.

## 0b. ⚠ SCOPE NOTE 2026-08-20 — read with `DIGITIZATION_QUEUE.md`, which now outranks this file

This file was written on **2026-08-13** as a one-week stock-addition plan. Since then the work has
been driven by the numbered queue in `DIGITIZATION_QUEUE.md`, and stocks have been added **one
owner-approved batch at a time** precisely because each addition shifts the plugin's ListBox. Two
were added on 2026-08-19 (`GEVACHROME_600`, `_605`) and two on 2026-08-20
(`KODAK_VISION2_50D_5201`, `FUJI_SUPER_F125_8532`) — the latter closing **queue C6**, which is the
item that this file's §3 "siblings deferred" line became.

**So the lists below are a candidate pool, not a plan.** Nothing in them is scheduled, and the
method requirements in §"Method requirements" still bind. Before adding anything from here, check
the queue: several entries have since been argued down (C4) or superseded.

## 0a. ✅ CLOSED 2026-08-18 — both σ(D) items below are DONE, read this first

**The schema decision is made and the wiring exists.** Queue item C1, same day:

* **Carrier: three anchors + an interior peak (5 floats), chosen by measurement.** Against
  the seven measured σ(D) samples per VISION3 sheet: legacy √ law 245 % max / 127 % rms,
  three anchors 41 % / 18 %, **three anchors + peak 20 % / 8.6 %**, a 12-sample array
  3.8 % / 2.0 %. The array was **rejected as over-parameterised against seven measured
  points**. The 4th anchor sits at D = 0.80, which also beat 0.70 / 0.75 / 0.90 on test.
* **σ(D) is now READ by the renderer** — `film_sim.py` stage 11 and `FilmGrainSigma()` in the
  generated C++, agreeing to 5.4e-07. Schema **v7 → v8**.
* **The heuristic below was still NOT touched, and the reason got stronger, not weaker.**
  The B&W measurement this file asks for was found (Mees Fig. 302, 2026-08-18b) — and then
  EKTACHROME 100D 5285's own sheet showed the **reversal** branch is wrong in sign too, in
  the opposite direction (σ rises ~20× to dmax, the heuristic says it falls). So both
  branches now have a measured counter-example, and the fix is a per-class evidence pass, not
  a sign flip. Instead of guessing, the heuristic's output was made **explicitly inert**: the
  renderer honours a shape only where `sigma_shape_measured` is set, which is the five
  vendor-traced stocks (**twelve** as of 2026-08-23). 150 stocks verified bit-for-bit unchanged.

`RESULT_2026-08-18e_C1_sigma_wiring.md`. **What remains from item 0 is only the per-class
evidence for the heuristic's signs** — one measured σ(D) per class (B&W silver has Mees now;
chromogenic negative has four sheets; reversal has one), which is a data task, not a decision.

## 0. ADDED 2026-08-17 — not a stock, and ahead of the list below

**Split the `_grain_v2` σ(D) heuristic on `is_monochrome`.** The VISION3 granularity
adoption established that colour-negative granularity **falls** toward Dmax (four Kodak TI
sheets, dmax/mid 0.55–0.63; plus 5247 and 5294 in Kodak's SMPTE Journal paper of July 1985).
`_grain_v2` still fills 0.4/1.0/**1.2** — rising — for all 103 non-reversal stocks, so its
sign is wrong for every colour negative it touches. It was left alone on purpose: the
approval covered four stocks, and all the evidence is *chromogenic*, while the same branch
also fills **B&W silver** negatives where σ ∝ √D is the textbook result.

What is needed before touching it: **one measured σ(D) for a B&W silver negative.** Two
candidates are already in the archive — `PDF/2017_Newson_film_grain.pdf` and the Mees 1942
material. This is worth more than any single stock on the list below, because it decides a
grain property for 103 stocks rather than one. See `DIGITIZATION_QUEUE.md` §3 and
`RESULT_2026-08-17f_vision3_granularity.md` §8c.

Second, smaller item from the same pass: the σ(D) schema is three anchors, and every curve
measured so far has an **interior peak** (D ≈ 0.78, at 1.24–1.32× the D = 1.0 value) that
three anchors cannot represent — they understate the maximum by about a quarter. A fourth
anchor, or a short σ(D) array, is a schema decision for the owner.

## 1. Kodak still reversal (sheets in PDF/PROFILES/KODAK/)
E100G, E100GX, E100VS, E200, EPP (Ektachrome 100 Plus), Elite Chrome
100/200/400, Ektachrome 64T (7280 sheet also present), 320T, P1600, EDUPE
duplicating.

## 2. Kodak motion picture
Vision2 50D 5201, Vision2 200T 7212 (16 mm doc), Ektachrome 100D 5294/7294,
Internegative II 5272, Vision Premier print 2393, B&W print 2302, sound
2374, Tri-X reversal 7266 (modern sheet for the existing profile),
aerographic Double-X 2405 / Tri-X 2403.

## 3. Siblings deferred from today's sheets
Plus-X Pan Professional PXE/PXT (rms 14 [C1] already extracted, F-8),
Ektapress PJ100 and PJ800 (E-116), Portra 2006 NC/VC generation
(160NC/160VC/400NC/400VC from e190), earlier Portra/Ektar/TMax editions as
formulation-revision records (MF.5), Royal Gold / GA100 (e2328).

## 4. Fuji (18 sheets)
Provia 100F/400F, Velvia 100/100F, Astia 100F, Sensia 200/400, Superia
100/200/1600, X-TRA 400/800, Reala, Pro 160S/160C/400H/800Z, T64,
True Definition.

## 5. Konica (~10 sheets)
VX 200/400, VX-S 100/200/400, Centuria Pro 160/400, Chrome Centuria 200,
csuper line, R100 (paper — decide if in scope).

## 6. Polaroid (~40 sheets — decide scope first)
Type 55 P/N, 52/53/54/57/59, 665, 669, 600, Time-Zero, Spectra, ID/UV.
Recommend selecting ~6 iconic types rather than all.

## 7. Foma / ORWO / Rollei / Maco / Kentmere
Fomapan 100, 200, R100 reversal; ORWO Wolfen NC400, NC500, UN54, NP100,
DP31, PF2, P400; Rollei Superpan 200, Pan 25; Maco IR820c Aura (owner
excluded infrared from today's batch — confirm whether IR stays excluded);
Kentmere Pan 200.

## 8. F-5-era 1970s Kodak professional B&W line (decision pending)
Panatomic-X, Plus-X Pan/Professional, Tri-X Pan/Professional, Verichrome
Pan, Royal Pan 4141, Royal-X Pan, Ektapan 4162 (1970s ed.), Contrast
Process Ortho 4154, Recording 2475. Unique value: full curve families over
development from the F-5 scans — the only line where the processing axis is
documented end-to-end. Add as distinct-era stocks, NOT as data grafts onto
the _1952 profiles.

## Method requirements (bind, per project rules)
Every addition: own-sheet provenance in _PROVENANCE_SOURCES; rms only where
the sheet prints it (colour negs = PGI = [C4] estimates, stated); resolving
power into _RESOLVING_POWER only as printed; reciprocity fitted with the
established convention; aliases must not collide (run the loader);
verify.py counts updated; C++ regenerated + compiled; READMEs, Found.md and
a CHANGES doc updated in the same pass; ZIP delivered.

---

## Added 2026-08-13 — deferred stocks from Cheltsov & Bongard 1958

Ten stocks from this source landed today (see `CHANGES_2026-08-13_cheltsov1958.md`).
The following were read and deliberately **not** entered, because the book gives fewer
engine-relevant fields for them. All numbers below are transcribed and cited, so entering
them is authoring work, not research work.

Source for every line: Чельцов В. С., Бонгард С. А., «Цветное проявление трёхслойных
светочувствительных материалов», М.: Искусство, 1958.

### Reversal stocks — the 1958 resolving-power ladder is internally comparable (p152)

| Candidate | Speed | Resolving power | Other documented | Page |
|---|---|---|---|---|
| `ILFORDCOLOR_1950` | 10 ASA (12/10 DIN) | **36 lp/mm** | AgHal + Ag₂S interlayers between sensitive layers — unique to Ilford; two types, daylight and artificial | p152 |
| `AGFACOLOR_REVERSAL_1950` | 20 ASA (15/10 DIN) | **32 lp/mm** | types T (daylight) and K (artificial); dye curves labelled "до 1940 г."; full process, amidol 35 min @18 °C | p159–160, p152 |
| `ANSCOCOLOR_REVERSAL_1950` | 12 ASA (13/10 DIN) | — | cyan max 660 nm, magenta max 540 nm; less unwanted blue in magenta than Agfacolor; grey antihalation underlayer | p160–161 |
| `GEVACOLOR_REVERSAL_1950` | 12 ASA (13/10 DIN) | **24 lp/mm** | magenta max **550 nm** (bluish, heavy unwanted blue); gelatin interlayers; protective supercoat; 7-bath, 2-hour amateur process | p163–166, p152 |

Note: entering all four completes the ladder Kodachrome 40 / Ilfordcolor 36 / Agfacolor
32 / Ferraniacolor 30 / Gevacolor 24, of which only Kodachrome and Ferraniacolor are in
the database now. The ladder is worth completing precisely because one measurement set
lets relative sharpness be set from documentation rather than taste.

### Negative stocks

| Candidate | Balance | Speed | Other documented | Page |
|---|---|---|---|---|
| `AGFACOLOR_NEG_B_333` | **5800 K** | 14/10 DIN ≈ 16 ASA | arc/daylight type, a.k.a. Agfacolor T; dye curves fig. 57 (developer Agfa T55) | p172 |
| `AGFACOLOR_NEG_C_334` | 3200 K | 16/10 DIN = 20–25 ASA | tungsten type, a.k.a. Agfacolor K | p172 |
| `GEVACOLOR_NEG_N5` | **2850 K** | 14/10 DIN ≈ 16 ASA | gelatin interlayers vs Agfacolor; magenta 550, cyan 660 nm | p178–179 |
| `FERRANIACOLOR_NEG_51` | 3200 K | 8 ASA effective in daylight w/ filter | low speed, raised contrast — the location stock of the set | p180 |
| `FERRANIA_STILL_NEG_1950` | 13/10 DIN = 12 ASA | — | still-photo negative, daylight and artificial types | p181 |
| `TELCOLOR_NEG_1950` | universal | 14/10 DIN ≈ 16 ASA | **red-sensitive MIDDLE layer, green-sensitive BOTTOM** — cannot be represented today | p181–182 |
| `PAKOLOR_NEG_1950` | — | 13/10 DIN = 12 ASA | daylight and artificial; conventional structure | p181 |
| `AGFA_ULTRA_NEG` | — | 17/10 DIN = 32 ASA | no yellow filter layer, emulsion dyed yellow instead; red sensitisation narrowed to fix deep blues rendering red | p174 |
| `AGFA_ULTRA_REVERSAL` | — | 16/10 DIN ≈ 25 ASA | same family | p174 |
| `PLENACOLOR_NEG` | — | 16/10 DIN ≈ 25 ASA | Ansco still negative, daylight only | p178 |

### Print stocks

| Candidate | Resolving power | Other documented | Page |
|---|---|---|---|
| `ANSCOCOLOR_848_POSITIVE` | — | dye maxima 440 / 540 / 660 nm; narrow sensitisation bands | p177–178 |
| `ANSCOCOLOR_846_DUPE` | **66 lp/mm** | grey antihalation base; dyes identical to the negative's | p178 |
| `EASTMANCOLOR_5245_DUPE` | — | bands 380–450 / 510–570 / 560–700; magenta top, cyan middle, yellow bottom; yellow fluorescing dye as filter layer | p224–225 |
| `EASTMANCOLOR_5216_PAN` | — | B&W separation-positive stock; sensitisation zones 520–560 and 680–700 nm; bluish-magenta washing dye plus a non-washing blue-green dye | p222–223 |
| `GEVACOLOR_952_POSITIVE` | — | inverted order: green-sensitive top, blue-sensitive bottom | p180 |
| `DUPONCOLOR_275_POSITIVE` | 185 / 52 / 35 lp/mm per layer | **polymer couplers** — coupler and dye are functional groups on polyvinyl acetal macromolecules replacing gelatin; blue top, red middle, green bottom; magenta antihalation layer that bleaches in the developer | p234–236, p200 |
| `SOVIET_CONTRATYPE_1950` | **70 lp/mm** | **gamma = 1.0 (documented)**; magenta masking coupler; reversal-processed dupe | p176–177 |

`DUPONCOLOR_275_POSITIVE` is the most interesting of these historically — a 1950s
attempt to abolish gelatin entirely — and the least representable, since neither the
polymer coupler chemistry nor the layer inversion has a field.

### Open questions to settle before entering the above

1. **NIKFI → ISO.** Soviet speeds are S = 20/H at D 0.85 above maximum fog. Not GOST,
   not ASA, no conversion published. `SVEMA_DS_2` and `SVEMA_LN_3` currently carry the
   printed midpoint as an `exposure_index` and label that a class-4 assumption. If a
   conversion can be sourced, both should be revisited — and `SVEMA_DS_1` (10–15 NIKFI)
   should not be entered until it is, because its range straddles a factor of 1.5.
2. **`GEVACOLOR_1952` balance.** Recommend 5500 K → 2850 K; awaiting approval, see
   `CHANGES_2026-08-13_cheltsov1958.md` §5.1. If approved, `GEVACOLOR_NEG_N5` above
   becomes redundant — our 1952 entry *is* N-5.
3. **Infrared exclusion.** Still open from the previous list: confirm whether Maco IR820c
   and other infrared stocks stay excluded.

---

## Added 2026-08-14 — deferred from The Compact Photo-Lab-Index (1979)

Eleven stocks from this source landed today (`CHANGES_2026-08-14_photo_lab_index.md`).
The following were read and deliberately **not** entered. All figures below are
transcribed and cited; entering them is authoring work, not research.

Source for every line: Pittaro, E. M. (ed.), *The Compact Photo-Lab-Index*, Morgan &
Morgan, 2nd Compact Edition 1979.

### Kodak motion-picture stocks — exposure indices only

The source gives dual daylight/tungsten indices but **no curve, gamma, granularity or
resolution** for any of these, so each would be one documented number plus a profile of
estimates. Enter only if that trade is acceptable.

| Candidate | Daylight | Tungsten | Page |
|---|---|---|---|
| `EASTMAN_4X_NEG_5224` / `7224` | 500 | 400 | p285, p289 |
| `KODAK_PLUS_X_REVERSAL_7276` | 50 | 40 | p289, p290 |
| `KODAK_4X_REVERSAL_7277` | 400 | 320 | p289, p290 |

Corroborations for stocks we already hold: Plus-X Negative 5231 at 80/64 and Double-X
5222 at 250/200 (p285, p289) — both match our stored `exposure_index`, so no change.

### Fuji — illuminant-conditioned exposure-index ladders

Our Fuji stocks are modern products, not these 1979 ones, so these are candidates for
NEW entries rather than corrections.

* **Fujicolor F-II 400** (p382): ASA 400 / 27 DIN daylight, no filter; ASA 125 / 22 DIN
  tungsten with LBB-12 or Wratten 80B; ASA 200 / 24 DIN with CC-20M + CC-20B; ASA 250 /
  25 DIN with CC-20B; ASA 200 / 24 DIN with CC-30M + CC-10R.
* **Fujichrome R100** (p388–389): ASA 100 / DIN 21 daylight; ASA 32 / DIN 16 tungsten.
  Indices referenced to 1/250 s; exposures of 1 s or longer need an increase.

### Polaroid — remaining types

| Candidate | Data available | Page |
|---|---|---|
| Type 57 | plot D-max 1.70 / D-min .09 / slope 1.40, 2500 ASA block shared with 47 | p589, p591 |
| Type 107 | plot D-max 1.60 / D-min .10 / slope 1.40 | p600 |
| Type 87 | plot D-max 1.65 / D-min .11 / slope 1.45 | p591 |
| Type 665 P/N | ASA 75; the roll-film P/N sibling of Type 55 | p574, p578 |
| Types 20, 32, 37, 38, 88, 108, 668, Polapan | speed only, no technical block | p574–578 |

Types 57, 107 and 87 are **formats of the same 2500-speed emulsion** already represented
by `POLAROID_47`; entering them would add format variants, not new emulsions. Decide
whether format variants belong in this database at all — the same question the gauge
collapse settled for cine stocks on 2026-08-13.

### Ilford — remaining

* **HP5** (p~482): in the section but not extracted this pass.
* **Commercial Ortho** (p488) and **Lith** (p496): wedge spectrograms present, speeds
  not extracted.
* **FP3, Selochrome, Special Portrait**: appear only in the cross-manufacturer
  development tables (p700–711) with speeds FP3 125, Selochrome ?, Special Portrait 125.

### Open questions

1. **`exposure_index_tungsten`** — decide whether to add the field. Documented for most
   of the corpus; the Polaroid Type 51 ratio of 3.2 versus panchromatic 1.25 is a real
   spectral statement that we currently discard.
2. **`ProcessingSpec`** — decide whether to add. Without it, no stored curve in the
   database states which developer, dilution, time and temperature it represents.
3. **Multi-segment reciprocity** — our single Schwarzschild exponent is provably the
   wrong form for 3 of 4 measured Kodak films. Decide whether to extend the model or to
   document the limitation permanently.
4. **PDF 700 sheet-film table** — column heading could not be established; it carries
   800 and 1000 against HP3 and HP4 where the roll-film tables give 400. Resolve before
   using anything from that page.
5. **Polaroid Type 42's ASA numeral** is missing from the page. Currently inferred from
   two agreeing figures (plot Speed 200, and 24 DIN → ASA 200). Confirm from another
   source if one appears.
6. Still open from earlier lists: confirm whether infrared stays excluded (Maco IR820c);
   `GEVACOLOR_1952` balance 5500 K → 2850 K awaiting approval.

---

## Added 2026-08-14 (second session) — Fujicolor cine line

From the FUJIFILM MOTION PICTURE FILM MANUAL (ref. KB-1101E, 2011). Every figure below is
transcribed from the master table on p1 and the per-film pages. **Deferred because the
manual gives exposure index and balance only** — no gamma, D-max, granularity or resolving
power, and its curves are raster. Entering them means one documented number surrounded by
estimates; that is a scope decision, not a research gap.

Our `FUJI_ETERNA_VIVID_500T_8547` is the one stock already held and it was **confirmed**,
not changed.

**Tungsten type** (secondary index through a Kodak No. 85 filter — a filter factor, so it
must NOT go in `exposure_index_tungsten`):

| Candidate | 35 mm | 16 mm | Tungsten E.I. | Daylight E.I. | Sideprint |
|---|---|---|---|---|---|
| `FUJI_ETERNA_VIVID_160_8543` | 8543 | 8643 | 160 | 100 | FN43 |
| `FUJI_ETERNA_250_8553` | 8553 | 8653 | 250 | 160 | FN53 |
| `FUJI_ETERNA_400_8583` | 8583 | 8683 | 400 | 250 | FN83 |
| *(held)* `FUJI_ETERNA_VIVID_500T_8547` | 8547 | 8647 | 500 | 320 | FN47 |
| `FUJI_ETERNA_500_8573` | 8573 | 8673 | 500 | 320 | FN73 |

**Daylight type** (secondary index through a Kodak 80A):

| Candidate | 35 mm | 16 mm | Daylight E.I. | Tungsten E.I. | Sideprint |
|---|---|---|---|---|---|
| `FUJI_F64D_8522` | 8522 | 8622 | 64 | 16 | FN22 |
| `FUJI_ETERNA_VIVID_250D_8546` | 8546 | 8646 | 250 | 64 | FN46 |
| `FUJI_ETERNA_250D_8563` | 8563 | 8663 | 250 | 64 | FN63 |
| `FUJI_REALA_500D_8592` | 8592 | 8692 | 500 | 125 | FN92 |

REALA 500D is described as "the world's first high-speed (E.I. 500) daylight-type motion
picture film" with a **4th colour layer**; ETERNA Vivid 250D likewise mentions a fourth
layer. Our schema is a three-layer tripack, so a fourth colour layer has no
representation — worth noting alongside the layer-order gap from the Cheltsov batch.

Shared by all nine (documented, uniform, verbatim per film): reciprocity — no correction
1/1000 to 1/10 s, **+1/3 stop at 1 s, and no filter correction**, i.e. achromatic failure.
Exposure conditions 3200 K (tungsten type) or 5400 K (daylight type) for 1/50 s through a
Fuji SC-41; Status M densitometry.

Also in the manual, not yet catalogued: intermediate ETERNA-CI (8503/4503/8603), recording
ETERNA-RDI (8511/4511 PET), positives ETERNA-CP 3512/3612, 3514DI/3614DI, 3523XD (exposed
2854 K for 1/100 s).

**⚠ Do not confuse with stocks we already hold:** `FUJI_F125_8530`/`8630` and
`FUJICOLOR_SUPER_F500_8572` are NOT in this manual. 8522 is F-64D and 8573 is ETERNA 500 —
different products with adjacent numbers. No data was transferred.

### Not a stock task — the two Fujifilm-simulation websites

Assessed 2026-08-14, **nothing entered.** Fujifilm "Film Simulations" are in-camera JPEG
presets in digital cameras, not emulsions. Only three map to a specific real film with any
confidence: Acros → Neopan Acros, PRO Neg. Std → NS 160, Classic Neg → Superia 100.
**Classic Chrome matches no specific emulsion** — it merely evokes Kodachrome, so modelling
it as Kodachrome would be a category error. The imaging-resource page is Fujifilm-sponsored
and its charts are unlabelled JPEGs; the fujipic page contradicts itself and should not be
cited. Four control-design observations worth revisiting if a UI is ever designed are
recorded in `CHANGES_2026-08-14b_fuji_kodak_websites.md` §4 — the notable one being
exposure-dependent grain, which converges on the three `sigma_shape_*` scalars we already
store and never read.

---

## Added 2026-08-15 — from the new reference batch

1. **Portra NC/VC generation** (E-190, on file, vector curves): `KODAK_PORTRA_160NC`,
   `160VC`, `400NC`, `400VC` as new stocks if the 2006 generation is wanted alongside our
   2010s stocks. PGI 36/40/44/48; reciprocity none to 10 s; C-41 Status M.
2. **KODAK DATA BOOK volume 5 (FILMS), pp 1150–1495** — dedicated pass. ~1948–1968 UK
   sheets for Plus-X/Super-XX/Tri-X/Panatomic-X/Verichrome/Royal-X/Kodachrome families.
   ⚠ Post-1960 ASA: halve before comparing with our 1952-era stocks.
3. **`EASTMAN_5366_DUPE_POSITIVE` as a PrintStock**: γ aim 1.20–1.60, RMS 9,
   RP 100/200 lp/mm (TI0265) — better documented than our generic `DUPE_FINE_GRAIN`.
4. **Trace queue additions (vector)**: 8572 sheet (all 4 curve sets), Vista brochure
   (all films), eterna_vivid500, E-190 per-film curves, 5205/2383(2015) curves.
5. **Zhurba 1990 pp 44–131**: request a local PDF/DjVu copy — online page images are
   webp binaries that web_fetch cannot retrieve.

## Added 2026-08-16b — Zhurba 1990 screenshot follow-ups
- Candidate new stocks from Table 66 (speed+balance documented): ORWO NC19 (64, 4200 K),
  ORWOCHROM UT-15/UT-20/UT-21/UT-23, UK-17 (40, 3200 K), Orwocolor Typ L (40, 3200 K).
- Table 21 (pp 72-73): ЛН-7/8/9, ДС-5М per-layer RMS + MTF — enter if stocks are added.
- ОЧ-Т line (Table 16 p69). Rename aliases Фото-64/Фото-125/ЦНЛ-64 if successor stocks ever modelled.
- Obtain local PDF/DjVu of Zhurba 1990 for the remaining pp 44-131 spreads.

## Note 2026-08-16c — KODAK DATA BOOK.pdf moved offline
Owner moved `PDF/PROFILES/KODAK/KODAK DATA BOOK.pdf` (1495 pp) to an external drive to
save local space. **Before running the queued volume-5 Films pass (pp 1150-1495), ask the
owner to copy the file back to the local PDF/PROFILES/KODAK folder.**

## Added 2026-08-16c — official-source hunt, next legs
- archive.org fulltext: "ORWO Handbuch", "ORWO Farbenfibel", "VEB Filmfabrik Wolfen" — NC24/UT family.
- Konica reversal (Chrome Centuria 100, Chrome R100): 125px mirror has negatives only; look for Konica Japan archived pages via Wayback.
- Re-fetch csuper400.pdf data page by another route (text extraction dropped it).
- Never-published parameter classes: Google Patents class G03C, assignee Eastman Kodak / Fuji Photo Film (DIR coupler coefficients, layer thicknesses, worked emulsion examples); Photographic Science and Engineering / Journal of Photographic Science via archive.org + HathiTrust (Callier Q, interimage, granularity-vs-density); Mees & James 4th ed. 1977.

## Added 2026-08-17 — Soviet TU batch follow-up
- VISUALLY VERIFY then enter eight fully specified stocks: ЛН-8 (ТУ 6-17-1109-88),
  ЛН-9 + ЛН-9С (ТУ 6-17-1443-88), ЦНД-64 (ТУ 6-17-1453-89), ЦО-32Д (ТУ 6-17-912-87),
  ЦО-Т-90ЛМ (ТУ 6-17-1000-88), ЦО-90Д + ЦО-90Л (ТУ 6-42-1514-90 x2). All figures already
  captured at OCR level in RESULT_2026-08-17_SOVIET_TU_BATCH.md -- one document per pass.
- OBTAIN ГОСТ 25130-82: named by ТУ 6-17-1371-86 as governing ЦНЛ-65's photographic
  characteristics; not in the corpus. Would do for the TsNL line what TU 6-17-622-84 did
  for ДС-4.
- ДС-5М dye-impurity ratios (7 measured terms) cannot be expressed by the single-scalar
  _dye() helper -- revisit when a dye-impurity carrier exists.
