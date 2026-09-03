# CHANGES 2026-08-13 — 23 new stocks (owner selection from the landing)

Database **98 → 121**. Owner selected: all Kodak still B&W (infrared
excluded), all Kodak still colour negative, Agfa Scala 200x. Remaining
families deferred to next week (next_week_task.md).

## Added — Kodak still B&W (9)

| stock | rms [C1] | RP lp/mm [C1] | source |
|---|---|---|---|
| KODAK_TMAX_100 | 8 | 63/200 | F-4016 2018 |
| KODAK_TMAX_400 (TMY-2) | 10 | 50/200 | F-4043 2016 |
| KODAK_TMAX_P3200 | 18 | 40/125 | F-4001 2019 |
| KODAK_TRI_X_400TX | 17 | not published | F-4017 2016 |
| KODAK_TRI_X_320TXP | 16 | not published | F-4017 2016 |
| KODAK_PLUS_X_125 | 10 | not published | F-8 1997 |
| KODAK_T400CN (chromogenic) | 9 | not published | F-2350 |
| KODAK_BW400CN (chromogenic) | 9.5 **[C4]** | — | F-4036 (PGI only) |
| KODAK_EKTAPAN_100 | 12 | not published | F-10 |

All rms figures at D=1.0 / 48 µm — the field's stored convention.
Chromogenic pair: dye image, silver_tone 0, low clump_gain, dye_cloud set;
BW400CN carries the orange printing mask in base_tint.

## Added — Kodak still colour negative (13)

EKTAR_100, PORTRA_160, PORTRA_800, PORTRA_100T (tungsten 3200 K), GOLD_100,
GOLD_200, ULTRAMAX_400, ULTRAMAX_800, VERICOLOR_III_160, EKTAPRESS_PJ400,
PROFOTO_100, ULTRA_COLOR_100UC, ULTRA_COLOR_400UC.

**Granularity honesty:** every colour-neg sheet publishes Print Grain Index
only, which Kodak states "cannot be compared to rms granularity" and
publishes no conversion for. All colour rms values are **[C4] engineering
estimates anchored to the PGI ordering**, the KODAK_PORTRA_400 policy.
Curves are [T3] class-shaped; the sheets' plotted curves are queued for
digitisation.

**Reciprocity [C1 where printed]:** Ektar/Portra 160/800/Gold/UltraMax/UC —
flat 1/10 000–1 s (statement only, no failure points: defaults kept, onset
evidence in comments). Three earned overrides: VERICOLOR_III onset **0.1 s**
(fails a decade before its successors — period-correct behaviour),
PROFOTO_100 flat to **10 s**, SCALA (below).

## Added — Agfa Scala 200x (1)

B&W **reversal**. rms 11 (×1000, D=1.0, 48 µm, Vλ, Scala process) and RP
50/120 [C1] from the sheet. Reciprocity: flat to ½ s; +½ @1 s, +1 @10 s,
+2 @100 s → p 0.80, onset 0.5 [C1]. Sheet documents push/pull ISO 100–1600 —
processing-axis data, queued.

## Naming and collisions

Ambiguous aliases kept with their prior owners: "plus-x" stays on
EASTMAN_PLUS_X_5231, "trix" on KODAK_TRI_X_REVERSAL_200; the new stocks take
qualified aliases. Official hyphenated names (T-MAX, TRI-X, PLUS-X PAN)
added to gen_film_names.py overrides.

## Breaking

eFILM_PROFILE renumbered again (23 insertions into natural order). Same
caveat as the morning's rename pass; ID-freeze still queued.

## Verification

verify.py **104 PASS / 2 FAIL** — the same two pre-existing failures, zero
new (count and reversal-count checks updated 98→121, 22→23). C++
regenerated, compiles; film_names.txt 121 lines.
