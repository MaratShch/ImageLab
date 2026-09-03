# RESULT 2026-08-29 — G4: the Gevachrome II process, and four Gevaert types the corpus did not know existed

**Task (queue G4):** read Webers & Westendorp, *"Einführung in die Kopierwerktechnik (XIV)"*,
**Fernseh- und Kino-Technik 33(7), 1979, pp. 245–247**, populate a process note for the
Gevachrome II process, and establish that the four named types existed. The queue entry states, and
this reading confirms, that the document **carries no sensitometry** — so **no profile was created
and no stored number was changed.**

**Outcome:** documentation only. **0 profiles added, 0 profiles modified, 0 schema changes.**
Database stays at **161 stocks**. This is the intended result, not a shortfall.

---

## 0. Source, and what kind of scan it is

`PDF/PROFILES/GEVAERT/Webers_Westendorp1979Umkehr.pdf`, 3 pages = printed pp. 245, 246, 247.

⚠ **IMAGE-ONLY SCAN, NO TEXT LAYER.** `pdftotext` returns **3 bytes**. Every figure below was read
from page images rendered at **200 dpi** with pymupdf. Bilevel, clean, no bleed-through — this is a
substantially better scan than the 1968 Gevachrome paper (150 ppi JPEG colour with show-through) that
`GEVACHROME_600` / `GEVACHROME_605` were traced from, and the table is unambiguous throughout.

Header, verbatim: `FERNSEH- UND KINO-TECHNIK · 33. Jahrgang · Nr. 7/1979`, running head
`Für den technischen Nachwuchs`, continued from `Heft 6/1979, S. 213`, and ending
`(Fortsetzung folgt)`. Authors printed as `Von JOHANNES WEBERS und KURT WESTENDORP`.

⚠ **A printing inconsistency in the source, recorded rather than silently corrected.** The page's
first section is numbered `D. 3.3.2. Umkehrverfahren`; the Gevachrome section three columns later is
numbered `D. 3.2.2.1. Der Gevachrome-II-Prozeß` — a *lower* number after a higher one. Transcribed as
printed. It does not affect any content.

---

## 1. The four types — the substantive finding

Printed on p245, verbatim, under `Agfa-Gevaert brachte in den letzten Jahren folgende Umkehrfilme
auf den Markt:`

| type, as printed | stated application | in the corpus? |
|---|---|---|
| `Gevachrome S – Typ 700` | `(für Studioaufnahmen)` — studio origination | **no** |
| `Gevachrome – Typ 710` | `(Reportagefilm)` — news / reportage | **no** |
| `Gevachrome D – Typ 720` | `(Material für Tageslicht)` — daylight-balanced | **no** |
| `Gevachrome Print-Typ 780` | `(Kopiermaterial)` — print stock | **no** |

followed by `Diese Materialien sind im Gevachrome-II-Prozeß zu verarbeiten, der entsprechend
Tabelle VIII abläuft.`

⚠ **THIS IS A SECOND GENERATION, NOT A RENAMING OF THE STOCKS ALREADY HELD, AND THE PROCESS PROVES
IT.** The corpus holds `GEVACHROME_600` (Typ 6.00), `GEVACHROME_605` (Typ 6.05) and
`GEVACHROME_902` (Print T.9.02) — the 1967–68 generation. Those run the **first** Gevachrome
process: **12 steps**, two temperature columns at **21 °C / 25 °C**, first developer **GP 110**
(Rens & Van Bets 1968, Tab. IV). The 1979 types run **15 steps**, a single **25 °C** column, first
developer **GP 112**. Different step count, different developer designation, different temperature
regime, and the 1979 paper names the process `Gevachrome II` explicitly.

**Consequence, and the reason this entry is worth its length:** the three stored Gevachrome profiles
**must not** be relabelled or re-cited as Gevachrome II. The naming similarity is exactly the trap a
future editor would fall into. The 700/710/720/780 family maps onto the 6.00/6.05/9.02 family by
*role* — reportage/studio camera stock plus a print stock — and by nothing else that this document
establishes.

**What is NOT stated anywhere in the document, for any of the four:** exposure index or DIN/ASA
rating, gamma, Dmin, Dmax, density range, granularity of any kind, resolving power or MTF, spectral
sensitivity, layer count or coating order, interimage behaviour, reciprocity, base tint, dye set.
**No characteristic curve is drawn for any of them.** There is therefore nothing to trace, and a
profile built from this document would be a profile built from a product name and an application
word. It was not built.

---

## 2. Tabelle VIII — `Entwicklungsprozeß Gevachrome II`, transcribed complete

Column headings as printed: `Bearbeitungsstufe und Funktion` · `Temperatur` · `Einwirkzeit` ·
`Bestandteile des Bades / Substanz` · `Mengen: Maschinen-Tank | Regenerator` ·
`Regeneriermengen ml/m 16-mm-Film`.

Quantities below are given **tank | regenerator**; `—` means the printed cell is a dash, i.e. the
substance is absent from that make-up. All pH values are `bei 25 °C`.

| # | step | T | time | replenish |
|---:|---|---|---:|---:|
| 1 | `Vorbad (GP 602)` — soften the rear anti-halation layer | 25 ± 1 °C | 10 s | 12 ml/m |
| 2 | `Backing-removal` — remove it, spray jets + wiper | 23 ± 2 °C | 10 s | 250 ml/m |
| 3 | `Schwarz-Weiß-Entwicklung (GP 112)` — first developer | 25 ± 0.2 °C | **180 s** | 15 ml/m |
| 4 | `Stoppbad` | 25 ± 0.5 °C | 45 s | 15 ml/m |
| 5 | `Wässerung` | 23 ± 2 °C | 45 s | 250 ml/m |
| 6 | `Zweite Belichtung` — light re-exposure | — | — | — |
| 7 | `Farbentwicklung` — colour developer | 25 ± 0.2 °C | **255 s** | 20 ml/m |
| 8 | `Erstes Fixierbad` | 25 ± 0.5 °C | 30 s | 15 ml/m |
| 9 | `Wässerung` | 23 ± 2 °C | 60 s | 250 ml/m |
| 10 | `Bleichbad` | 25 ± 0.5 °C | 120 s | 15 ml/m |
| 11 | `Wässerung` | 23 ± 2 °C | 60 s | 250 ml/m |
| 12 | `Zweites Fixierbad` — composition as the first | 25 ± 0.5 °C | 60 s | 15 ml/m |
| 13 | `Schlußwässerung` — `Gegenstromwässerung` | 23 ± 2 °C | 90 s | 250 ml/m |
| 14 | `Stabilisator` | 25 ± 2 °C | 10 s | 12 ml/m |
| 15 | `Trocknung` | 40–50 °C, 20–50 % rel. Feuchte | 180–300 s | — |

**Derived, not printed:** wet time steps 1–14 sums to **975 s = 16 min 15 s**; total replenishment
**1369 ml per metre of 16 mm film**, of which **1250 ml (91 %) is wash water** in five stages. Both
are arithmetic on the printed column, marked here as derived so neither can later be quoted as a
Gevaert figure.

### Bath formulae, as printed

**1. `Vorbad (GP 602)`** — `Wasser` 600 | 600 ml · `E.D.T.A. Na₄` (`Tetra-Natriumsalz von
Äthylendiamin-Tetra-Essigsäure`) 2 | 2 g · `Natriumsulfat (wasserfrei)` 100 | 100 g ·
`Borax (Na₂B₄O₇, 10 H₂O)` 15 | 15 g · `Natriumhydroxid` 0.8 | 0.8 g · `mit Wasser auffüllen auf`
1000 | 1000 ml · **pH 9.30 ± 0.15**.

**2. `Backing-removal`** — `Weichwasser über Sprühdüsen`. No formula.

**3. `Schwarz-Weiß-Entwicklung (GP 112)`** — `Wasser` 600 | 600 ml ·
`Natriumhexametaphosphat` 2 | 2 g · `Natriumsulfit (wasserfrei)` 50 | 58 g ·
**`Hydrochinon` 6 | 9.5 g** · **`Phenidon B` 0.5 | 0.6 g** · `Natriumkarbonat (wasserfrei)` 25 |
27.5 g · `Kaliumbromid` 2.3 | 0.5 g · `Kaliumthiocyanat` 3 | 3.6 g · `Kaliumjodid` 6 | 2 mg ·
`Additiv GP 112 AD` 5 | 5 ml · `Natriumhydroxid` — | 1.5 g · `auffüllen auf` 1000 | 1000 ml ·
**pH 10.20 ± 0.1 | 10.30 ± 0.1**.

Function, as printed: `Reduktion der belichteten Silberhalogenid-Körner in den drei
lichtempfindlichen Schichten der Emulsion.`

**4. `Stoppbad`** — `Wasser` 700 | 700 ml · `Kalialaun` 15 | 20 g · `Eisessig` 10 | 8 ml ·
`Borax (10 H₂O)` 21 | 8 g · `auffüllen bis` 1000 | 1000 ml · **pH 4.2 ± 0.2 | 3.65 ± 0.15**.
Hardening stop (potassium alum), not a plain acid stop.

**5. `Wässerung`** — `Wasser`.

**6. `Zweite Belichtung`**, verbatim: `Belichtung der bei der Aufnahme unbelichteten und bisher
unentwickelten Silberhalogenidkristalle. Die Zweitbelichtung geschieht zweckmäßig unter
Flüssigkeitsniveau im Wässerungstank. Es ist weißes Licht mit einer Lichtmenge von 100 000 lx/s zu
verwenden. Beide Seiten des Films müssen vom Licht erfaßt werden.`
⚠ Printed as `lx/s`; the physical quantity is an **exposure**, lx·s. Transcribed as printed, read as
lx·s. White light, **both sides**, **under the liquid surface** of the wash tank.

**7. `Farbentwicklung`** — `Wasser` 500 | 500 ml · `Natriumhexametaphosphat` 2 | 2 g ·
`Natriumsulfit (wasserfrei)` 4 | 4.75 g · **`N,N-Diäthyl-p-Phenylendiamin-Sulfat (TSS)` 3.6 | 5.35 g
`oder` `N,N-Diäthyl-p-Phenylendiamin-Chlorhydrat (Gevadiamin-C, CD-1)` 2.7 | 4 g** ·
`Natriumbikarbonat (wasserfrei)` 25 | 29 g · `Kaliumbromid` 0.75 | — g · `Natriumkarbonat` 0.3 | — g ·
`Kaliumjodid` 4 | — mg · `Natriumhydroxid` — | 0.9 g · `Additiv GP 52` 2.5 | 3 ml ·
`auffüllen bis` 1000 | 1000 ml · **pH 10.70 ± 0.1 | 11.30 ± 0.1**.

⚠ The developing agent is named **two ways in one cell** — the Gevaert trade name `Gevadiamin-C` and
the Kodak designation `CD-1` — as alternatives at different weights (sulfate vs chlorhydrate), not as
two ingredients. Both spellings retained.

**8. `Erstes Fixierbad`** — `Wasser` 700 | 600 ml · `Natriumsulfit (wasserfrei)` 10 | 12 g ·
`Natriummetabisulfit` 8.75 | 10.5 g · `Borsäure` 6.25 | 7.5 g · `Natriumazetat (3 H₂O)` 6 | 7.3 g ·
`Eisessig` 10 | 11.5 ml · `Aluminiumchlorid (6 H₂O)` 10 | 12 g · `Ammoniumthiosulfat` 175 | 212 g ·
`Natriumbisulfat` — | 9 g · `auffüllen bis` 1000 | 1000 ml · **pH 4.30 ± 0.15 | 3.80 ± 0.15**.

**10. `Bleichbad`** — `Wasser` 600 | 600 ml · **`Kaliumferricyanid` 40 | 75 g** ·
`Kaliumbromid` 30 | 40 g · `Natriumazetat (3 H₂O)` 5 | 6.5 g · `Eisessig` 5 | 6.5 ml ·
`Natriumbisulfat` 6 | 7.5 g · `E.D.T.A. Na₄` 10 | 13 g · `auffüllen bis` 1000 | 1000 ml ·
**pH 4.10 ± 0.2** both. Ferricyanide bleach, not a ferric-EDTA bleach.

**12. `Zweites Fixierbad`** — `Zusammensetzung wie Erstfixierbad`.

Steps 10 and 12 both carry the note `Bei elektrolytischer Rezyklierung kann die Regeneriermenge auf
etwa 1,25 ml reduziert werden` — a **12× reduction** from 15 ml/m with electrolytic silver recovery.

**14. `Stabilisator`** — `Wasser` 250 | 250 ml · `Formalin (40 %-Lösung)` 12.5 ml | (cell blank) ·
`Netzmittel (Saponine/Merck)` 1.8 | 1.8 ml · `auffüllen bis` 1000 | 1000 ml · **pH 7.60 ± 0.3** both.
Function, as printed: `Verhindert die Bildung von Wasserflecken und dient zur Stabilisierung der
Farbstoffe. Verringert die Neigung zum Ausbleichen.`

⚠ The formalin cell in the **regenerator** column is **blank**, not dashed, and the two are different
claims. Transcribed as blank; not read as zero.

---

## 3. Why nothing was stored, checked against each carrier the schema offers

This is the part of the task that had a real decision in it. Four carriers could plausibly have taken
this material, and each was rejected for a stated reason rather than by omission.

| carrier | why it does not fit |
|---|---|
| a new **profile** for any of the four types | no sensitometry of any kind. The queue entry forbids it and the reading confirms the ground for the prohibition. A profile is 40+ numbers; the document supplies **zero** of them |
| **`ProcessingFamily`** | it holds `DevelopmentPoint` rows, and a `DevelopmentPoint` is `(developer, dilution, minutes, celsius, contrast_index, gamma, base_fog)`. This table prints developer, minutes and celsius but **no gamma, no contrast index and no fog at any time point** — it is a single fixed condition, not a time-gamma series. Filling the row with a fabricated gamma is the only way to use it, so it was not used. ⚠ Note the queue entry itself suggests `ProcessingFamily`; the queue was written before the table had been read, and the table shows the suggestion does not survive contact with the schema |
| **`ProcessingSpec`** | it is a **member of a profile**, and there is no profile for Typ 700/710/720/780 to hang it on. Attaching it to `GEVACHROME_600/605/902` would assert those stocks run Gevachrome II, which §1 shows is **false** |
| **`ProcessVariant`** | it describes the *same emulsion* under a different chemistry, keyed to a profile whose curves exist. Neither condition holds: no emulsion of ours is documented here, and no curves are printed |

`layer_stack` was considered and rejected on the same ground: Bild 74 does draw a full reversal
coating order (`Schutzschicht / blauempfindlich / Gelbfilter / grünempfindlich / rotempfindlich /
Lichthofschutzschicht (Silber-Schwarz-Schicht) / Schichtträger`, with `Gelb / Purpur / Blaugrün`
dye layers) — but its caption is `Prinzipieller Aufbau eines Umkehrfarbfilms`, a **generic textbook
diagram in a tutorial series**, not a measurement of any Gevachrome type. Adopting it would attach a
fabricated coating order to a real product name.

**Net effect on the render path: none.** No stored value moved, so no image can change.

---

## 4. What the document does contribute

Three things, all documentation:

1. **Four product identities are now on record with a citable source**, so a future Gevaert
   datasheet naming Typ 700/710/720/780 can be recognised and matched instead of being read as an
   unknown. This is the whole value of the item and it is worth having.
2. **The generation boundary is pinned** — 1967–68 Gevachrome (12 steps, GP 110, 21/25 °C) against
   1979 Gevachrome II (15 steps, GP 112, 25 °C) — which protects the three stored profiles from
   being mislabelled.
3. **A complete reversal process specification** joins the corpus's process material, alongside the
   1968 Gevachrome 12-step table already noted as unstorable in
   `RESULT_2026-08-19_gevaert.md` §1. Both are on file; neither has a carrier.

---

## 5. Verification

- `pdftotext` byte count re-checked: **3 bytes**, no text layer — the transcription is from images
  and is labelled as such.
- Every quantity above was read twice from the 200 dpi renders, once per column.
- Step numbering 1–15 is continuous across the page break (p246 ends inside step 7, p247 opens with
  its continuation, then step 8).
- Database unchanged: **161 stocks**, `film_ids.lock` untouched, no generated file regenerated.

**Queue:** `G4` closed. It produced no data because the source contains none — which is the
conclusion the queue entry predicted, now confirmed by reading rather than assumed.
