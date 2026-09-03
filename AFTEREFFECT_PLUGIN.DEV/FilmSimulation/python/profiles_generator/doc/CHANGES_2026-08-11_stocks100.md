# Changes 2026-08-11 — seven new stocks from owner-supplied documents (93 → 100)

Owner supplied three new document sets and requested seven additions:

* `PDF/PROFILES/KODAK/kodak-films-5.pdf` — **Kodak Films, Data Book, Fifth
  Edition, Eastman Kodak Company, 1952** (72 pages, ~20 data sheets, text
  layer present, characteristic curves as vector line art).
* `PDF/PROFILES/AGFA/AGFA stocks.pdf` — **Technical Data, Agfa Professional
  Films**, ~2003 (10 stocks; ISO, RMS granularity ×1000, resolving power at
  both 1.6:1 and 1000:1, emulsion layer thickness, base thickness).
* `PDF/PROFILES/SOVIET STANDARDS/` — 20 GOST documents (19 with text
  layers; `gost_10691.6-88.pdf` is a scan without OCR but duplicates
  `ГОСТ 10691.6-88.pdf`, which has one).

## New profiles

| Name | Tier | Key published data | Source page |
|---|---|---|---|
| `KODAK_VERICHROME_1952` | T2 | EI 64/32 (1952 ASA), ortho, R 95 l/mm @30:1 | kodak-films-5.pdf p33–34 |
| `KODAK_PANATOMIC_X_SHEET_1952` | T2 | EI 32/25, pan Type B, "very fine grain", R 100 @30:1 | p57–58 |
| `KODAK_TRI_X_SHEET_1952` | T2 | EI 200/160, pan Type C, R 65 @30:1 | p49–50 |
| `KODAK_ORTHO_X_SHEET_1952` | T2 | EI 125/64, ortho, R 85 @30:1 | p59–60 |
| `AGFA_OPTIMA_200` | T1 | ISO 200, RMS 4.3, R 130/50 l/mm, layer 18 µm | AGFA stocks.pdf p6 |
| `AGFA_OPTIMA_400` | T1 | ISO 400, RMS 4.5, R 130/50 l/mm, layer 19 µm | p6 |
| `AGFA_PORTRAIT_160` | T1 | ISO 160, RMS 3.5, R 150/60 l/mm, layer 18 µm | p5 |

## Conventions and caveats recorded in the source

* **1952 exposure indexes are stored AS PRINTED** (pre-1960 American
  Standard, ~2.4× safety factor). The same emulsions were renumbered ~2×
  higher in 1960 with no emulsion change. Documented at the profile block;
  do not "correct" them.
* **Kodak resolving power is at 30:1 test-object contrast** — a third
  convention alongside 1.6:1 and 1000:1. Stored in `_RESOLVING_POWER`'s
  high-contrast slot AS PRINTED with a caveat comment (Foma precedent).
* **Kodak 1952 publishes no granularity figures** — grain rms values are
  [T2] interpolations against SUPER_XX_1938 / APX_25 anchors and the book's
  qualitative classes. The book's D-logE curve families are queued for
  digitisation (see `DIGITIZATION_QUEUE.md`).
* **Agfa granularity and resolving power are transcriptions [T1]**; curve
  shapes, couplers and dye purity are carried from the measured
  `AGFA_OPTIMA_100` family fit, scaled by the published deltas. Per-channel
  rms uses the same tier-2 stack rule as Optima 100.
* Two Tri-X clash traps documented in the profile text: the 1952 SHEET film
  is neither `KODAK_TRI_X_REVERSAL_200` (cine reversal) nor the 1954+ roll
  film.

## SVEMA_FN_64 / Фото-65 (GOST corroboration, no value changes)

Owner confirms **Фото-65 and FN-64 are the same emulsion** — the database
already carried this equivalence (aliases `foto-65` etc., Gurlev 1986
citation). Added **ГОСТ 24876-81 Table 6** as a third provenance source:
R ≥ 110 lin/mm (corroborates Gurlev's 110 over Chibisov's 92), gamma
0.8–1.1, latitude ≥ 1.5 logH, fog D₀ ≤ 0.04/0.06. The GOST granularity
limit (σ_D×1000 ≤ 45, GOST aperture) is **not comparable** to the 48 µm
diffuse-RMS convention used here and was deliberately not imported.

## Files touched

* `PYTHON/profile_generator/film_profiles.py` — 7 profiles,
  `_RESOLVING_POWER` (+7 with 30:1 caveat), `_PROVENANCE_SOURCES` (+7 new,
  +1 GOST citation on SVEMA_FN_64).
* `PYTHON/profile_generator/gen_film_names.py` — hyphenated official-name
  overrides for the four 1952 Kodak stocks.
* Regenerated: `film_profiles.hpp`, `film_profiles.cpp`, `film_enum.hpp`,
  `film_names.txt` (root, listbox order re-locked to the .cpp),
  `PYTHON/profile_generator/film_names.txt`,
  `doc/FilmActiveProfiles.md` (100 stocks, 79 citing documents),
  `doc/FilmCurves.md`.
* `doc/DIGITIZATION_QUEUE.md` — queued the 1952 curve families and the
  Agfa 2003 spectral/MTF/density plots.

`validate_all()` passes with no warnings. `film_profiles.cpp` compiles
(`g++ -std=c++14 -fsyntax-only`). Enum indexes shifted (insertion is
alphabetical): any saved preset storing a raw profile index from the
93-stock build resolves to a different stock; presets storing names are
unaffected.
