# RESULT 2026-09-01b — the German F-PF-D4 twin and the SCALA F-SW12-E6 sheet

Two documents were supplied by the owner and staged into
`PDF/PROFILES/AGFA/`:

| file | what it is | md5 | new? |
|---|---|---|---|
| `agfa-aERRKF-Datenblatt_F_PF_D4.pdf` | «Technische Daten — Agfa Professional Filmsortiment», **F-PF-D4, 4. Auflage, Stand 07/2003**, 12 pp, vector | `eba47f2f…` | yes |
| `agfa_scala.pdf` | «AGFA SCALA 200x PROFESSIONAL — Technical Data», **F-SW12-E6, 6th edition, 08/2000**, 4 pp, vector | `70410719…` | yes |

Three new readers, all registered in the audit stage; build green,
`0 failures, 0 warnings`, **33 audits**, `verify.py` 489 PASS / 1 FAIL
(the known saturation-hierarchy baseline).

---

## 1. What F-PF-D4 turned out to be, and the finding that came with it

D4 is the **German twin of `AGFA stocks.pdf` (F-PF-E4)** — the same PageMaker
job, distilled 55 minutes apart on 2003-07-18. That was asserted, not assumed:
`agfa_2003_curves.py` compares every drawn coordinate page by page and **pp7–8
are byte-identical**, pp6/9 differ only in label placement (5.65 pt on p9).

The consequence is the uncomfortable one. **The English twin has been in the
corpus since 2026-08-29 and only its curves were ever read.**
`agfa_2004_curves.py` stops at p7. Pages 8 and 9 carry six more plotted
columns, and **not one printed table on the sheet had been harvested in either
language** — although those tables carry the resolving power at two contrasts,
the layer thickness, the base thickness per format, the DX and negative codes,
the development-time matrices at 18/20/22/24 °C and the exposure index per
developer, for all ten films.

So the German file's contribution is not chiefly German. It is that it made
somebody read the English one properly.

---

## 2. The RSX II resolving power was revised, and the curves were not

| | 1998 «Technical Data PF» | 2003/04 F-PF-D4/E4 |
|---|---|---|
| RSX II 50 | 125 lines/mm @ 1000:1 | **135** |
| RSX II 100 | 125 | **130** |
| RSX II 200 | 110 | **120** |

RMS, the 1.6:1 figure and the layer thickness are unchanged on all three, so
this is a targeted revision and not a re-typeset. The owner's instruction was
to trace both editions before adopting anything, and that is what decided it:

> once each edition's label-box offset is removed, the two editions' curves
> agree to **0.001–0.006 D rms** on every colour-density record and
> **0.0006–0.0035 lg** on every spectral one.

Agfa reused the identical artwork. A **later measurement of an unchanged
emulsion** is therefore what the three profiles now store, with the 1998
figure recorded in each `ParamSource`, and the traced curve beside it stays
the 1998 one it has always been — there is no mixing of editions, because
there is only one drawing.

### 2a. The de-bias is itself a result

The raw comparison said the editions disagreed by a suspiciously constant
0.027 lg and 0.038 D — the same magnitude on every film and every quantity,
which is the signature of a systematic error rather than a product change.

**A text box's centre is not a digit's optical centre.** `_axis_labels` takes
each label's y as its bounding-box centre; the box runs ascender to descender
while the digits sit on the baseline. It is a pure offset, invisible inside
one document — and the two editions set their ordinates at **different point
sizes**, 7.69 pt against 6.01 pt, so it does not cancel between documents.
Measured against the RSX II 100 panel's own axis rectangle, which must span
exactly 0.0 to 4.0 D:

```
1998   top 3.9808   bottom -0.0199    ->  0.020 D LOW
2003   top 4.0153   bottom +0.0154    ->  0.015 D HIGH
```

0.035 D combined — essentially the whole 0.038 D by which the editions
appeared to differ. `axis_box_bias()` finds the rectangle rather than assuming
it, and rejects it if its mapped extremes miss the outermost printed labels by
more than 0.10 in axis units.

⚠ **This bias is present in every absolute density either Agfa reader has ever
produced**, at roughly 0.02 D. It is below any adopted tolerance and no stored
value was changed for it, but it is now quantified rather than unknown.

---

## 3. APX 400 is two films, and APX 100 is the control

D4/E4 p1 marks the entry **«Agfapan APX 400 * … * Neue Generation (ab 2003)»**.
Its headline numbers do not move (RMS 14.0, 110 lines/mm, 10 µm) and neither
does its characteristic curve — the 2003 panel reproduces the 1998 one to
0.0010 D rms, because Agfa reprinted the same drawing. **The processing tables
move in every row:**

| APX 400, small tank / tray, 18/20/22/24 °C | 1998 | 2003 |
|---|---|---|
| REFINAL | 8 / 6 / 4½ / 4 | 7 / 5 / 4 / 3 |
| RODINAL 1+25 | 8 / 7 / 5½ / 4 | 11½ / 10 / 9 / 8 |
| RODINAL 1+50 | 13 / 11 / 9 / 8 | – / 30 / 27½ / 25 |
| RODINAL SPECIAL | 5 / 4½ / 3½ / 3 | 7 / 6 / 4½ / 4 |
| STUDIONAL LIQUID | 5 / 4½ / 3½ / 3 | 7 / 6 / 4½ / 4 |
| ATOMAL FF (tanks) | 10 / 8 / 6 / 5 | 12½ / 10 / 6 / 6 |
| REFINAL (tanks) | 9 / 7 / 5 / 4 | 6½ / 5 / 4 / 3 |

**7 of 7 changed.** And **APX 100's tables are identical across the two
editions in all 7 comparable rows** — that control is what turns a footnote
into evidence. `agfa_2003_sheet.py` asserts both halves: APX 100 must match,
APX 400 must not.

`AGFA_APX_400` is deliberately left as the **pre-2003** film, per the owner's
decision, because its curve, processing spec and P-16-C data are all from
1998/1999. The 2003 tables and the generation statement are recorded in its
`exposure_index` `ParamSource`.

⚠ APX 100 was **re-rated without being re-formulated**: at unchanged
developing times, RODINAL 1+25 at 8 min goes ISO 100/21° → **125/22°** and
RODINAL 1+50 at 17 min goes ISO 125/22° → **160/23°**.

---

## 4. Two defects found, both now corrected

**4a. `AGFA_OPTIMA_200`'s RMS citation named the wrong document.** The stored
4.3 is right; the 1998 sheet it cited prints **4.5** in that column (verified
by word coordinates — `RMS` at x 347.0, the value at x 367.6, inside that
column's own band). 4.3 is the 4th edition's figure. This was collateral from
the 2026-09-01 rms correction, which fixed eight profiles' values and gave
them all the same citation; Optima 200 is the one film of the ten whose RMS
moved between editions. Citation of record corrected, both values recorded.

**4b. `AGFA_RSX_II_200` was carrying SCALA's push table.** It held
`max_push_stops 3.0 / max_pull_stops 1.0`, cited to the 1998 sheet p9
«Push/pull behaviour». That table sits at x 396.8 on a page whose three
columns are RSX II 100 (x 42–200), RSX II 200 (x 219–375) and **AGFA SCALA
200x (x 425–575)** — it is SCALA's, and `AGFA_SCALA_200X` already carries the
identical numbers from it. RSX II 200 is ISO 200 like SCALA, so a
400/800/1600 push ladder looked entirely plausible on it. Column adjacency,
not a reading error.

What Agfa actually publish for the RSX II line is smaller and is prose, on
D4/E4 p4: *«Bis zu einer Empfindlichkeitsanpassung von ± 1 Blende (!) bleibt
die Neutralität der Farbwiedergabe voll erhalten. Und selbst eine
Empfindlichkeitssteigerung von bis zu 2 Blenden beeinflußt die Farbbalance und
die Maximaldichte nur sehr gering.»* Stored as `max_push 2.0 / max_pull 1.0` —
asymmetric because the sentence is: ±1 stop with full neutrality, and only the
push direction extends to 2.

---

## 5. A refusal with evidence: the AGFAPAN sharpness panel is not per-film

A duplicate-artwork scan over all ten columns × four panels of **both**
editions found exactly two duplications, and the same two in each:

1. **RSX II 50 and RSX II 100 share the spectral-density panel** — already
   documented in `_MEASURED_DYE_MATRIX` and in RSX II 100's `dye_density`
   source. Independently reconfirmed here, and now known to hold in the 2003
   edition too.
2. **APX 100 and APX 400 share the sharpness panel** — new. On the 4th edition
   p9 the two are the same 73-point path translated 175.21 pt: every y offset
   identical to **0.0000 pt**, every x offset agreeing to 0.096 pt. The 1998
   sheet p10 does the same, returning f50 57.6 lines/mm for both films where
   the 4th returns 59.6 for both.

Two different films cannot share a measured MTF. The panel is a line-generic
illustration, and an f50 read off it would be a fabricated per-film number.
Both AGFAPAN `mtf.f50_g` cells therefore **stay estimated on purpose**, and
now say so in a `ParamSource` instead of merely being red.

---

## 6. What F-SW12-E6 adds that nothing else in the corpus has

* **Exposure latitude as a speed-dependent number**: ±½ stop at ISO 200–1600,
  **±1 stop at ISO 100**. The range sheets say only "flatter".
* **The granularity viewing condition**: "equivalent to a 12-fold
  magnification", and "(only in SCALA process)".
* **The film base by standard** — *"Film base: safety film (acetyl cellulose)
  to DIN 15551"*, with polyester named separately and only for the 175 µm
  sheet-film base, plus an extra NC layer on roll and sheet backs. This is the
  **only per-film base-material statement in the whole Agfa set**, and it is
  why `AGFA_SCALA_200X.base_material` is now populated while the other nine
  Agfa profiles correctly stay empty: the range sheets say only «Die
  Filmunterlage besteht aus Acetylzellulose oder Polyester», an either/or with
  no per-film assignment.
* **The five-layer emulsion design** and **"Total thickness: 12 µm"**.
  ⚠ This is *not* the range sheets' «Schichtdicke 7 µm» and the two are not
  averaged: 7 µm is the emulsion layer, 12 µm the whole coating including the
  retouchable gelatine backing. Two quantities, one recorded in `coated_um`
  and the other in its `ParamSource`.
* **The anti-halation construction in words**: 35 mm is a clear base with an
  AHU layer decolorised in the developer; roll and sheet add a dark green
  gelatine back, also decolorised.
* **Pulled granularity as a figure**: "− 10 % at ISO 100/21°".
* *"Contrast matched to AGFACHROME RSX 100 (basis ISO 200/24°)."*

It **confirms** rather than adds on two counts, and both are asserted against
the stored profile: the four-point reciprocity table is now printed in three
independent documents six years apart, and the push/pull speed ladder in two.

⚠ **Recorded negative**: no granularity plot, no aperture series, no Wiener
spectrum, no gamma-time family. This sheet cannot fill σ(D) or clump size for
SCALA either, which is the same checked absence `agfa_p16c.py` records for the
AGFAPAN films.

---

## 7. Page 2, which the owner asked to be read closely

D4 p2 «Agfacolor Optima mit EYE VISION-Technologie» is the one page of the
document carrying an embedded raster, and it prints two panels: *spektrale
Empfindlichkeit des Auges* and *spektrale Empfindlichkeit der Film-Emulsionen*,
each showing Blau/Grün/Rot humps for OPTIMA against «herkömmlich».

**No number was digitised from it, because there is none to digitise
honestly.** Agfa's own sentence is *«Der Effekt ist schematisch in den
folgenden Grafiken dargestellt»*, and the panels carry 400–650 nm tick labels
and no ordinate scale at all.

What it does carry is a documented design statement, recorded as such: Optima's
sensitisation is deliberately matched to the eye's response, and Agfa name the
four errors it is meant to remove — *unangenehmer Grünstich bei
Fluoreszenz-Licht*, *Rotverschiebung bei bestimmten blauen Blütenfarben
(Hortensien, Klematis, Rittersporn)*, *bräunliche Wiedergabe bestimmter grüner
Textilfarben*, and *fehlende Struktur in bestimmten roten Farben (z. B. Rosen)*.

---

## 8. Measurement conditions, in Agfa's own words (D4 p5)

Kept verbatim, as the owner asked, and now asserted by
`agfa_2003_sheet.py` rather than transcribed once:

| Grafik | Bezug |
|---|---|
| Spektrale Empfindlichkeiten | Energiegleiches Spektrum; Meßdichte 1,0 über Minimaldichte |
| Absorption der Schichtfarbstoffe | Neutrales Objekt mittlerer Helligkeit; Minimaldichte |
| Farbdichtekurven | Belichtung Tageslicht 1/100 sec.; Prozeß AP 70/C-41 bzw. AP 44/E-6; Densitometrie Status A bzw. Status M |
| Schärfe | Belichtung Tageslicht; Densitometrie Visuelles Filter (Vλ) |
| Körnigkeit | Belichtung Tageslicht; Visueller Filter (Vλ); Diffuse Dichte 1,0; 48 µm Meßblende |
| Auflösungsvermögen | Linien pro mm bei Kontrastumfang 1.6 : 1 bzw. 1000 : 1 |

Production tolerances, D4 p1: **Empfindlichkeit ± 0,5 DIN = ± 1/6 Blende**;
**Farbabstimmung ± 5 CC-Filtereinheiten**.

`AGFA_OPTIMA_100` now carries the only documented **coating order** in the
Agfa set, from D4 p5 «Schichtaufbau … am Beispiel des Optima 100»:
Schutzschicht / UV-Filterschicht / Blauempfindliche Gelbschichten /
Gelbfilterschicht / Grünempfindliche Purpurschichten / Rotfilterschicht /
Rotempfindliche Blaugrünschichten / Lichthofschutzschicht / Unterlage,
«Gesamtschichtdicke: 16 µm» — which equals its stored `coated_um`, so the
diagram and the spec block agree and the stack is this film's. The three
sensitised layers run blue/green/red from the top, which is the conventional
order; `LayerStack` exists because it is not always (Duponcolor 275 is
blue/red/green), so storing it says *checked* rather than *assumed*.

---

## 9. Read and deliberately not stored

⚠ The development-correction row moved out of this list on the owner's word —
see §11, schema v24.

* **Taking filters**, D4 p3: 81A +⅓ at 5700 K, 82A +⅓ at 5300 K, 80B +1⅓ at
  3400 K, 80A +2 at 3200 K; fluorescent 50R +1 (D), 40M +⅔ (W), 20C+40M +1
  (KW), 40M+10Y +1 (warm weiß). `TakingFilter` holds one designation and a
  transmission curve per stock, not a filter *table*, so these are line-wide
  advice rather than a per-profile value.
* **Per-format base thicknesses and sheet-film materials** — RSX II 100
  «Planfilm = Azetat 190 µm», APX 100 / SCALA / OPTIMA 100 «Planfilm = PET
  175 µm». `base_um` is one float and the corpus's stocks are 135; recorded in
  each emulsion source.
* **DX cartridge codes and Agfa negative codes** — the negative codes were
  already the `emulsion.designation` values (49-14, 49-15, 49-10, 49-02); the
  DX cartridge codes are packaging identifiers and are **not** emulsion
  designations, so they were not written into that field.
* **ISO 3664 viewing standard** (5000 K, 1400 cd/m², uniformity ≥ 75 %),
  storage limits, and the UV absorber statement *«eine UV-Sperrschicht bereits
  in der Emulsionsschicht eingelagert»*.

---

## 10. Files

| new | what |
|---|---|
| `agfa_2003_curves.py` | pp8–9 of F-PF-E4 traced; the twin check; the cross-edition comparison and `axis_box_bias` |
| `agfa_2003_sheet.py` | every printed table, both languages, both editions; the APX 100/400 matched assertion |
| `agfa_scala_sheet.py` | F-SW12-E6 in full, checked against the stored profile |

Changed: `film_profiles.py` (three resolving powers, one push spec, one
citation, two layer stacks, one base material, seven new `ParamSource`
records), `build.py` (three audits), `doc/PROGRESS.md`, `doc/NotFound.md`,
`doc/DIGITIZATION_QUEUE.md`, `doc/FilmActiveProfiles.md` (generated).


---

## 11. Second pass, same day — schema v24 and the G6 narrowing

The owner's instruction was *"use values from German document"* and *"if you
need update schema version — do it"*. Both done.

### 11a. `ReciprocityTable.development_correction_pct` — schema v24

Reciprocity failure does not only cost speed: a long exposure develops to a
higher contrast, and Agfa quantify the compensation on the same four time
cells as the exposure correction. The database held only the speed half. The
field is additive and inert like the rest of the dataclass, and validates
against `times_s` for length and refuses a positive value (a development
correction is a reduction; a positive one would mean the sign was misread).

| film | times (s) | exposure (stops) | **developing (%)** |
|---|---|---|---|
| APX 25 | 0.5 / 1 / 10 / 100 | 0 / +½ / +1 / +2 | **0 / 0 / 0 / 0** |
| APX 100 | 0.5 / 1 / 10 / 100 | 0 / +1 / +2 / +3 | **0 / −10 / −25 / −35** |
| APX 400 | 0.5 / 1 / 10 / 100 | 0 / +1 / +2 / +3 | **0 / −10 / −25 / −35** |

⚠ **APX 25's four zeros are a stated null, not an absent row.** The row spans
all three AGFAPAN columns at once on `agfa_films.pdf` p6 and reads 0/0/0/0
under APX 25 while its neighbours read 0/−10/−25/−35. Agfa are saying this
emulsion's contrast does *not* climb with a long exposure, and the slowest film
being the one that needs no correction is the expected direction. Storing `()`
would have thrown that statement away.

Three editions agree cell for cell — «Developing adjustment (%)» in
`agfa_films.pdf` 09/1998 and F-PF-E4 08/2004, «Entwicklungskorrektur (%)» in
F-PF-D4 07/2003 — and `agfa_2003_sheet.py` asserts all three plus the stored
values. ⚠ My first pass reported this row as German-only; that was wrong, and
the English label is simply different words in the same place.

**A typo caught on the way.** F-PF-E4 and F-PF-D4 both head APX 400's first
time cell «1/10 000-1» where APX 100's and SCALA's read «1/10 000-½». The glyph
is genuine — U+00BD on those two, a plain `1` here, in both languages — so it
is the document, not the extraction. As printed it contradicts itself: the same
1 s would carry both the end of the zero-correction interval and the +1 stop
column beside it. The 1st edition prints «1/10 000 - ½» for all three AGFAPAN
films, and the 4th edition's own layout has the 3-column *colour* blocks ending
at 1 s and the 4-column *B&W* blocks ending at ½ — a cut-and-paste from the
colour block. The stored 0.5 s was right and is now right *for a written
reason*; the audit asserts the typo so that a future edition which fixes it is
noticed rather than silently changing the meaning.

### 11b. G6: the German definition, and an argument of ours that was overstated

The German edition defines the figure better than the English twin does:

> DE «kennzeichnet die Auflösungsgrenze bei der Wiedergabe benachbarter,
> feinster Details (z. B. **Striche eines Linienrasters**). Bezug: **Linien pro
> mm** bei Kontrastumfang 1.6 : 1 bzw. 1000 : 1»
>
> EN "It indicates the resolution limit in the rendition of adjacent finest
> details (e.g. **lines in a matrix**)."

*Striche eines Linienrasters* is **the strokes of a line grating** — Agfa name
the test object. "Lines in a matrix" names nothing; it is a mistranslation, and
the German is now the record of choice.

**It still does not close G6.** The sentence says what you look at; the unit
line says «Linien pro mm» and not «Linienpaare pro mm», the standard German
term Agfa did not use. `Linienpaare`, `Lp/mm`, `Perioden`, "line pairs" and
"cycles" appear in **no Agfa file in this corpus**; `Strich` appears exactly
once, in that sentence.

⚠ **And this row's own recorded evidence proved one third of what it claimed.**
The note in `agfa_1998_curves.py` and in queue G6 said f50/RP = 0.30 against
Tani's 0.5, "so reading the axis as HALF-cycles would move it further from the
relation, not closer". That holds for the one hypothesis it tested — the MTF
*abscissa* half-cycles while the *table* is cycles, giving 0.15. It addressed
neither of the other two: if **both** are half-cycles the ratio is unchanged at
0.30, being a ratio of two quantities in one unit; and if the **table** is
half-cycles with the abscissa in cycles it becomes 0.60, which is *closer* to
0.5 than 0.30 is. Corrected in both places.

**What does bound it** is a cross-maker test, now in `agfa_2003_sheet.py`.
Resolving power falls as grain rises, and over the monochrome stocks that
publish both figures `RP·√RMS` is roughly invariant within a maker:

| maker | n | median RP·√RMS |
|---|---:|---:|
| SVEMA | 4 | 455 |
| ROLLEI | 3 | 454 |
| **AGFA** | **4** | **450** |
| KODAK | 15 | 361 |
| EASTMAN | 3 | 316 |

Agfa sit with everyone else. Halving Agfa's printed figures to reach cycles/mm
would put them at 225 — below every maker in the corpus. The direct pairs say
it too: **APX 25 prints 200 at RMS 7.0, KODAK PANATOMIC-X prints 200 at RMS
7.0, FUJI NEOPAN ACROS 100 prints 200 at RMS 7.0** — three makers, one grain
figure, one resolving power.

**Conclusion: the resolving-power TABLE is on the line-pair scale and the
stored values stand as printed.** What remains open is only the MTF panel's own
abscissa — and the German edition cannot settle that one, because the plot
artwork is in English in both files.
