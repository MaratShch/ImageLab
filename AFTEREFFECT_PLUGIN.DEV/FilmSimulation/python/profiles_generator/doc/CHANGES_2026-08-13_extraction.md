# CHANGES 2026-08-13 — first extraction pass from the new document landing

Follows PDF_LANDING_2026-08-13.md priorities. Owner-directed, with special
attention to «Современные фотоматериалы и их обработка» as requested.

## 1. EASTMAN EKTACHROME 5239 / 7239 — datasheet found for a "no datasheet" stock

Kodak publication **H-1-5239** (the 7239 Daylight sheet; the publication code
is the 5239 family code, so it documents both gauge entries):

* `rms_granularity` **10.4 → 14.0 [C1]** on both entries. The sheet prints
  "Diffuse RMS Granularity 14", read at net diffuse visual density 1.0
  through a 48 µm aperture — exactly the field's stored convention. The old
  10.4 was an estimate with no recorded source. Visible effect: the stock
  renders noticeably grainier, which the profile's own description ("grainy
  in a way 1960s television flattered") already expected.
* `_RESOLVING_POWER` += (40.0, 100.0) lp/mm at 1.6:1 / 1000:1 [C1],
  ISO 6328-1982 method, Process VNF-1.
* `_PROVENANCE_SOURCES`: 5239 no longer claims "no official manufacturer
  datasheet" — both entries now cite H-1-5239.
* Sheet also documents: reciprocity flat 1 s – 1/10 000 s, Status A
  densitometry, LAD control values — recorded here, not yet representable
  as fields.
* NOTE: 5239/7239 remain a duplicated gauge pair of the FN-64 kind
  (identical numbers, separate entries for default_format only). Collapse
  candidate under the same owner decision; not touched today.

## 2. Reciprocity from «Современные фотоматериалы» (2002–03) — 7 profiles [C1]

Vendor-compiled reference, era matches the profiles. Fitted with the
established convention (t_a^p · onset^(1−p) = t_m at printed points):

| profile | printed points | fit |
|---|---|---|
| AGFA_OPTIMA_100 | flat→1 s; +½ @10 s; +1½ @100 s | p 0.84, onset 1.0 |
| AGFA_OPTIMA_200/400, PORTRAIT_160 | +1 @10 s; +2 @100 s | p 0.77 (both points agree exactly), onset 1.0 |
| AGFA_VISTA_200 | +1 @10 s | p 0.77, onset 1.0 |
| AGFA_APX_100/400 | flat→½ s; +1 @1 s; +2 @10 s; +3 @100 s | p 0.77, onset 0.5 (pairwise spread 0.72–0.81 recorded) |

APX table also prints development-time reductions for pushed exposures
(−10 % at +1, −25 % at +2, −35 % at +3) — processing-axis data the schema
cannot hold; recorded in the override comment for the future PRC group.

Replaces heuristic defaults (colour 1.0; APX 0.95). Fields are stored per
the gap-analysis plan; the engine's reciprocity consumption awaits the
exposure-duration control input (Appendix B.3.2).

## 3. Kodak Publication F-5 — indexed and catalogued (extraction queued)

88-page scan OCR-indexed (headers + DS footers). Data-sheet insert spans
pages 33–57: Panatomic-X, Plus-X Pan and Professional, Tri-X Pan (DS 18)
and Professional (DS 20), Verichrome Pan (DS 22), Royal Pan 4141 (DS 16),
Royal-X Pan, Ektapan 4162 (DS 5), Contrast Process Ortho 4154 (DS 3),
High Speed Infrared 4443, Recording 2475, plus developer-property tables
(p 76) and a film-base section (p 3).

**Deliberately NOT written into the database today:** these are ~1970s
formulations (ISO-era speed markings); the existing KODAK_*_1952 profiles
document the 1952 formulations. Transferring 1970s numbers onto 1952 stocks
would be a cross-era C3 transfer presented as C1. The sheets become C1 data
if/when the 1970s versions are added as their own stocks, and are the
archive's only curve-families-over-development source for this line either
way. Queued in DIGITIZATION_QUEUE.md.

## 4. Verification

* verify.py after edits: **104 PASS / 2 FAIL** — the same two pre-existing
  failures (saturation hierarchy, red-blue pair); zero new.
* C++ regenerated (film_profiles.hpp/cpp, film_enum.hpp, film_names.txt),
  copied to root, compile-checked.
* One transcription incident during the edit, caught by immediate re-read:
  a regex aimed at the provenance table first matched the new
  `_RESOLVING_POWER` entry and overwrote it; repaired in the same session
  and values re-verified from the loaded module. Recorded per the honesty
  rule.

## Sources registered

* Kodak H-1-5239 (EASTMAN EKTACHROME Daylight 7239), 4 pp text PDF.
* «Современные фотоматериалы и их обработка», 717 pp text PDF, reference on
  2002–03 Agfa/Fuji/Kodak/Konica materials.
* Kodak Publication F-5, ~1979 edition, 88 pp JPG scan (indexed only).
