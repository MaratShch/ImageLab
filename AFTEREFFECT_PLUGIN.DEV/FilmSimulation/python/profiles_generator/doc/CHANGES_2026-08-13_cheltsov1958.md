# Cheltsov & Bongard 1958 — extraction, verdict and database changes

**Date:** 2026-08-13
**Source:** `PDF/PROFILES/cheltsov_vs_bongard_sa_tsvetnoe_proiavlenie_trekhsloinykh_sv.pdf`
Чельцов В. С., Бонгард С. А., «Цветное проявление трёхслойных светочувствительных
материалов», М.: Искусство, 1958, 250 с.
*(Cheltsov V. S., Bongard S. A., "Colour Development of Three-Layer Light-Sensitive
Materials", Moscow: Iskusstvo, 1958.)*

**Document state:** true PDF with an OCR text layer over a scan. 250 pages, ~405 kB of
extractable text. OCR quality is mediocre — Cyrillic arrives with inter-letter spacing
("В С . Н Е Л Ь Ц О В" for "В. С. ЧЕЛЬЦОВ") and Latin transliterations are corrupted
("АЗА" for "ASA", "тр" for "mµ"). Text was normalised before searching; every number
quoted below was then re-read in context, and the one table whose column order the
flat text scrambled was rebuilt from word coordinates.

---

## 1. Verdict

**Yes — and the most valuable thing in it is exactly what you asked about: colour
temperature.**

The book is a chemistry monograph, so most of its 250 pages are developing-agent
structure, coupler synthesis and dye stability, which this database has no field for.
But chapters VIII–XII are a systematic survey of every three-layer colour film in
production in 1958, and that survey prints, film by film, the three things the engine
actually reads: **balance colour temperature, speed, and in several cases gamma.**

Before this pass the database had exactly four distinct balance points in use across
121 stocks: 5500, 4500, 4200, 3200 K, plus one at 2950. The book supplies **eight
balance temperatures that were measured and published by the manufacturers of the
period, six of which existed nowhere in our data**:

| Film | Balance | Note |
|---|---|---|
| Kodachrome, ordinary | **5900 K** | highest in the whole database |
| Anscocolor negative 843 | **5400 K** | |
| Agfacolor negative B-333 / T | **5800 K** | post-war arc/daylight |
| Soviet DS-2 | **5000 K** | Soviet studio carbon arc |
| Agfacolor negative **type 3** | **4000 K** | deliberate mid-point — first universal negative |
| Kodachrome Type A | **3450 K** | between studio 3200 and photoflood 3400 |
| Soviet LN-3 | **3300 K** | Soviet incandescent practice, not 3200 |
| Gevacolor negative N-5 | **2850 K** | |

Not one of these is 5500 K. The reflex assumption that "daylight film = 5500 K,
tungsten film = 3200 K" is a *modern* convention; in 1958 manufacturers balanced to
whatever their market's lamps and arcs actually produced, and they published the
number. Six new balance points is a bigger gain than six new stocks, because balance
temperature is a documented physical property whereas most of what we author for an
old stock is estimate.

**Second-most valuable:** an internally comparable **resolving-power ladder**. The book
measures five reversal stocks by one method and prints them together (p152):
Kodachrome 40, Ilfordcolor 36, Agfacolor 32, Ferraniacolor 30, Gevacolor 24 lines/mm.
Absolute lines/mm figures from 1958 are not directly comparable with a modern MTF, but
a *ratio within one measurement set* is, and it lets the relative sharpness ordering of
five stocks be set from documentation instead of taste.

**Third:** the book confirms, rather than supplies, a modelling assumption we had been
carrying unsupported — see §4.

---

## 2. What was added

Ten `FilmProfile` entries (121 → **131**) and four `PrintStock` entries (5 → **9**).

### 2.1 Film profiles

| Name | Balance | Speed | Documented extras |
|---|---|---|---|
| `KODACHROME_1938` | 5900 K | ASA 12 | res 40 lp/mm; couplers in developer, not emulsion |
| `KODACHROME_TYPE_A_1938` | 3450 K | ASA 20 | res 40 lp/mm |
| `AGFACOLOR_NEG_TYPE_3` | 4000 K | ASA 20 | soft gradation is the documented mechanism |
| `ANSCOCOLOR_NEG_843` | 5400 K | — | layer peaks 440 / 555 / 655 nm; res 48; grey base |
| `GEVACOLOR_NEG_652` | 3200 K | ASA 32 | **gamma 0.65 published**; magenta 550 nm, cyan 660 nm |
| `FERRANIACOLOR_NEG_82` | 3200 K | ASA 20 | structure stated equivalent to Agfacolor negative |
| `FERRANIACOLOR_REVERSAL_1950` | daylight | ASA 20 | res 30 lp/mm; magenta identical to Agfacolor's |
| `SVEMA_DS_2` | 5000 K | 20–26 NIKFI | gamma 0.60–0.85; res 65 lp/mm (SR-13) |
| `SVEMA_LN_3` | 3300 K | 20–26 NIKFI | gamma 0.60–0.85; res 65 lp/mm (SR-13) |
| `EASTMANCOLOR_5248_1953` | 3200 K | ASA 32 | bands 380–500 / 490–580 / 580–680 nm; antihalation D 1.3 |

### 2.2 Print stocks

| Name | Resolving power | Why it differs |
|---|---|---|
| `TSP_1_POSITIVE` | 70 lp/mm | colloidal-silver yellow filter layer |
| `TSP_3_POSITIVE` | 150 lp/mm | filter layer replaced by a washing-out yellow dye |
| `TSP_6_POSITIVE` | **>200 lp/mm** | layer order inverted, emulsion stained, no filter layer |
| `EASTMANCOLOR_5382_1953` | 200 / 97 / 37 lp/mm per layer | same inversion |

The TsP-1 → TsP-3 → TsP-6 ladder is the single most instructive measurement in the
book: three stocks from one factory, one dye set, one curve family, differing **only**
in how blue light is kept out of the lower layers — and resolving power nearly triples.
It is a clean demonstration that print sharpness is dominated by intra-layer scatter,
not by the emulsion's intrinsic resolution.

### 2.3 Name-collision hazards handled

Two of these numbers are reused by their own manufacturers for later, unrelated films.
Both are called out in the profile descriptions so nobody merges them:

- **5248.** `EASTMANCOLOR_5248_1953` (this book) vs `EASTMAN_EXR_100T_5248` (1989 EXR).
  Thirty-six years apart. Same number, different film, no shared data.
- **652 / 682.** `GEVACOLOR_NEG_652` (this book, gamma 0.65) vs `GEVACOLOR_NEG_682`
  (SMPTE 1980, gamma 0.57). Same manufacturer, twenty-two years apart. The two gammas
  are a pleasing independent consistency check on both entries, not evidence they are
  the same stock.

---

## 3. On Ferrania — no split needed

You asked whether our Ferrania entry should be split into modern and 1940s versions.
I checked before acting, and the answer is that there is nothing to split.

`FERRANIA_P30` is a **black-and-white** stock whose every number comes from the 2017
Film Ferrania manufacturer specification. This book documents **Ferraniacolor**, a
*colour* reversal and colour negative line. Different chemistry, different product
family, no shared data — so splitting P30 would create two entries where the evidence
supports one, which is the opposite of what you want.

What was warranted instead was **adding** the period Ferrania stocks the book actually
documents, which is what §2 did: `FERRANIACOLOR_REVERSAL_1950` (ASA 20, 30 lp/mm) and
`FERRANIACOLOR_NEG_82` (ASA 20, tungsten). `FERRANIA_P30` was left untouched.

One thing I did *not* fix but should flag: `FERRANIA_P30` carries
`era="1960s / revived 2017"` and a description crediting it to Italian neorealist
cinema, while all of its numbers come from the 2017 sheet. The 1960s cine P30 and the
2017 revival share a name and a nominal EI, but the 2017 emulsion was coated on new
equipment with modern materials. The era string therefore claims a span the data does
not cover. Left alone pending your call, since it is not what this book is about.
**ACTIONED 2026-08-14:** era narrowed to `"2017 revival"`. The neorealist history
stays in the description as history; the era field now states only what the data
covers. See `CHANGES_2026-08-14_photo_lab_index.md`.

---

## 4. What the book CONFIRMS (assumption → citation)

`DUPE_FINE_GRAIN` has always been modelled at **gamma ≈ 1.0**, with a comment arguing
from first principles that unity gamma is a design necessity because a release print is
three or four generations from the negative and contrast would otherwise compound.
That argument was sound but uncited.

The book states it as fact for the Soviet duplicating stock (p177):
*«Коэффициент контрастности её равен единице»* — "its contrast coefficient equals
unity" — alongside a resolving power of 70 lp/mm and a magenta masking coupler. A
first-principles argument in our source comments is now a documented manufacturer
specification. The engine behaviour does not change; its justification improves.

---

## 5. What the book contradicts — checked, and mostly it does not

Four existing entries looked like they conflicted with the book. Three survived the
check. Reporting all four, including the ones where I was wrong to suspect them:

| Entry | Suspicion | Outcome |
|---|---|---|
| `AGFACOLOR_NEG_TYPE_B_1943` | balance 4200 K vs book's 5800 K for "type B" | **No change.** Our 4200 K is already cited as German wartime *Schneeweisskohle* carbon arc. The book's B-333 is the post-war Wolfen/Leverkusen product. Two different arcs, two different eras. |
| `GEVACOLOR_NEG_682` | book documents type **652** | **No change.** Genuinely different films, 1958 vs 1980. |
| `SVEMA_DS_4` | book's DS family is 5000 K arc, ours is 5500 K daylight | **No change.** DS-1 is the daylight member, DS-2 the arc one; DS-4 is documented as daylight by Gurlev 1986. Independent corroboration instead: Gurlev gives DS-4 **63 lin/mm**, this book gives the 1958 DS family **65 lp/mm** — two sources, three decades apart, agreeing to 3 %. |
| `GEVACOLOR_1952` | balance 5500 K | **Was flagged here, ACTIONED 2026-08-14**: changed to 2850 K. See `CHANGES_2026-08-14_photo_lab_index.md` and the §D follow-up in the requirements document. |

### 5.1 `GEVACOLOR_1952` — recommend 5500 K → 2850 K, awaiting your approval

Our entry: colour negative, 1948–1960s, EI 16, balance 5500 K, provenance **tier 3,
`fitted_from="analogy"`** — i.e. no documentary basis for any of it.

The book documents **every** Gevacolor negative of that era as tungsten-balanced:
type N-5 at **2850 K** (14/10 DIN) and type 652 at 3200 K. There is no daylight-balanced
Gevacolor negative in the survey at all.

The identification is tight on the one number we do carry. 14/10 DIN converts to
**16 ASA** on the book's own DIN↔ASA pairs — and our entry's EI is 16, arrived at
independently. Same speed, same kind, same era, same manufacturer. Our stock is
almost certainly N-5, and N-5 is a 2850 K tungsten film.

**Why I did not just change it:** this alters an existing stock's render output rather
than adding a new one. At `wb_strength > 0` the blue gain under a 5500 K scene moves
from 1.00 to roughly 2.4 — a large, visible shift. That is your call, not mine. Say the
word and it is a one-line change plus a provenance upgrade from tier 3 to tier 1.

---

## 6. Extracted but NOT entered — and why

Recorded here so the reading is not lost. Full numbers are in `DIGITIZATION_QUEUE.md`
and `next_week_task.md`.

**6.1 Per-layer resolving power (Table 24, p200) — schema cannot hold it.**
The book publishes resolving power *per emulsion layer*:

| Film | yellow | magenta | cyan |
|---|---|---|---|
| Agfacolor negative | 80 (top) | 34 (mid) | 27 (bot) |
| Agfacolor positive | 102 (top) | 42 (mid) | 30 (bot) |
| Eastmancolor negative | 110 (top) | 46 (mid) | 30 (bot) |
| Eastmancolor positive | 37 (bot) | **200 (top)** | 97 (mid) |
| Duponcolor positive 275 | 35 (bot) | 185 (top) | 52 (mid) |

A **3.7× spread inside a single film**, and the ordering tracks physical depth: the
layer light reaches first resolves best, every time, in all five films. Our `MTFSpec`
carries three per-record numbers, so this *could* in principle be represented — but our
three numbers are R/G/B *records*, whereas these are *physical layers*, and the mapping
between them depends on the layer order, which we also do not store. Entering them as
if they were record MTFs would silently assert natural layer order for films that do not
have it. Anchored the single MTF on the documented middle-layer figure instead and
recorded the spread in the descriptions.

*Verification note:* the flat OCR text scrambled this table's cyan column, and my first
reading assigned 27 lp/mm to the Eastmancolor negative. Rebuilding the table from word
coordinates showed 27 belongs to the **Agfacolor** negative and Eastmancolor's is 30.
The printed layer tags (в./с./н. = top/middle/bottom) confirm the corrected order. The
first reading was wrong and did not reach the database.

**6.2 Non-natural layer orders — no field exists.**
Five films in the book deliberately permute their layers: TsP-6, Eastmancolor positive
5382, Gevacolor positive 952, Duponcolor 275 (blue top, **red middle, green bottom**),
and Telcolor negative (**red middle, green bottom**). The motive is documented and
physical — put the magenta record, which carries most perceived sharpness, in the layer
light strikes first. This is a real structural property with a measured consequence and
our schema has nowhere to put it. Appendix A already requires it; this is the first
corpus that would exercise it.

**6.3 Processing chemistry — the PRC axis, still empty.**
The book gives complete published recipes for Kodachrome, Agfacolor reversal,
Anscocolor, Gevacolor, Ektachrome and Eastmancolor: every bath's composition in g/L,
plus times and temperatures (e.g. Agfacolor reversal: amidol first developer 35 min at
18 °C, colour developer 11 min at 18 °C). Also Table 14, measuring how silver and dye
gamma diverge with development time — dye gamma rises from 0.33 to 2.09 while silver
gamma goes 0.25 to 0.73, i.e. the ratio climbs from 1.32 to 2.86. That is a
*quantitative* statement about a mechanism the engine models qualitatively, and it is
exactly the kind of data Appendix A's processing axis was specified for. Nothing was
entered because there is no field to enter it into.

**6.4 Deferred stocks (thinner data).** Ilfordcolor (10 ASA, 36 lp/mm), Agfacolor
reversal T/K (15/10 DIN, 32 lp/mm), Anscocolor reversal (12 ASA, dye maxima 540/660 nm),
Gevacolor reversal (12 ASA, magenta 550 nm, 24 lp/mm), Ferraniacolor negative 51
(3200 K, 8 ASA effective in daylight), Anscocolor positive 848 and dupe 846, Gevacolor
N-5 and positive 952, Telcolor negative, Pakolor, Agfa Ultra (17/10 DIN negative,
16/10 DIN reversal), Duponcolor 275, Plenacolor, Fujicolor 1950s.

---

## 7. Unit conversions — checked, not assumed

**7.1 DIN → ASA is verified against the source, not inferred.**
The book quotes pre-1957 DIN, usually paired with ASA. Those pairs are internally
consistent with the standard 10·log₁₀ relation anchored at 12 DIN = 10 ASA:

| printed DIN | printed ASA | 10 × 1.259^(DIN−12) |
|---|---|---|
| 12/10 | 10 | 10.0 |
| 13/10 | 12 | 12.6 |
| 15/10 | 20 | 19.9 |
| 16/10 | 20–25 | 25.1 |
| 17/10 | 32 | 31.6 |

Five independent pairs, all consistent. The conversion is therefore **class 2**
(documented equivalence) rather than an assumption. Note these are *period* ASA values
carrying the old safety factor; no attempt was made to restate them as modern ISO, so
`exposure_index` here means "the number the manufacturer printed".

**7.2 NIKFI → ISO is NOT solved, and is the weakest link in the Soviet entries.**
Soviet speeds are given in NIKFI units defined as S = 20/H at density 0.85 above maximum
fog (p175). NIKFI is neither GOST nor ASA and the book gives no conversion. The printed
range (20–26) and the criterion are class 1; carrying the midpoint across as an
`exposure_index` because the schema has no NIKFI field is a **class 4 assumption**, and
it is labelled as such in both profiles.

---

## 8. One thing measured that is not about this book

While verifying that the new balance temperatures reach the render, I found that
**`balance_kelvin` has no effect at default settings.** `RenderSettings.wb_strength`
defaults to `0.0`, and the entire balance block is gated on `wb_strength > 0.0`.
`SVEMA_DS_2` (5000 K) and `SVEMA_LN_3` (3300 K) render byte-identically out of the box,
as do the two Kodachromes.

This is **by design, not a bug** — the docstring on `solve_anchors` states that
colour-temperature mismatch is treated as a creative control rather than something a
real lab would leave in, and that per-channel anchoring deliberately does not cancel it
once enabled. At `wb_strength=1.0` all ten new stocks produce distinct casts in the
correct direction: Kodachrome at 5900 K goes warm (R 0.364 / B 0.336), Kodachrome Type A
at 3450 K goes strongly blue (R 0.273 / B 0.594), Anscocolor at 5400 K is near neutral.

Worth stating plainly because it changes how this batch should be read: the six new
balance temperatures are *correct data that is inert until the user asks for it*. They
do not silently change anybody's existing renders.

---

## 9. Files changed

| File | Change |
|---|---|
| `film_profiles.py` | +10 `FilmProfile`, +4 `PrintStock`, +10 `_PROVENANCE_SOURCES` rows |
| `verify.py` | count 121 → 131; print stocks 5 → 9; reversal count 23 → 26; `PRINT_STOCKS` import |
| `gen_active_profiles.py` | **bug fix** — block slicer now bounded at the end of each profile literal (see §10) |
| `film_profiles.hpp` / `.cpp` | regenerated |
| `film_enum.hpp` | regenerated, +10 enumerators |
| `film_names.txt` | regenerated, 131 lines, pipe separator preserved |
| `doc/FilmActiveProfiles.md` | regenerated — 131 stocks, 112 citing documents |
| `doc/FilmCurves.md` | regenerated |
| `doc/FilmDatabase_Charecteristics.MD` | Addendum follow-up note |
| `doc/FilmDatabase_Charecteristics_Rus.MD` | same note, Russian |
| `DIGITIZATION_QUEUE.md` | fifth batch — the four unrepresentable data classes |
| `next_week_task.md` | deferred stocks from §6.4 with their numbers |

## 10. Generator bug found and fixed

`gen_active_profiles.blocks_from_source()` sliced per-profile source text between
successive `name="..."` matches, so the **textually last profile in each tuple absorbed
every following line to the next match** — for the last `FilmProfile`, that meant the
whole tail of the module including `_PROVENANCE_SOURCES`. The citation scanner then
harvested other stocks' document numbers.

Observed concretely: `EASTMANCOLOR_5248_1953` was credited with Kodak publications
F-4016, F-4043 and F-4001 — the T-MAX sheets, nothing to do with a 1953 Eastmancolor
negative. Fixed by cutting each block at the first following line that starts in
column 0.

**This bug predates today's work** and simply moved from stock to stock as the last
profile changed, so earlier revisions of `FilmActiveProfiles.md` may carry the same
contamination on whichever stock happened to be last at the time. Citing-document count
is unchanged at 112 after the fix, so no legitimate citation was lost.

## 11. Verification

- `verify.py`: **108 PASS / 2 FAIL** — both failures pre-existing and unrelated
  (saturation hierarchy ordering, neighbour-pair coupling).
- All 131 profiles and 9 print stocks load and pass `validate_all()`.
- All 10 new stocks and all 4 new print stocks render finite through `simulate()`.
- `film_profiles.cpp` and `AlgoSpectralSensitivity.cpp` compile clean, `-std=c++14`.
- All 10 new stocks correctly fall back to the authored balance proxy: the book gives
  band edges and peak wavelengths, not curves, and **no spectral sensitivity curve was
  synthesised from them.** Three peak wavelengths are not a curve; fitting Gaussians
  through them would manufacture a shape nobody measured.
