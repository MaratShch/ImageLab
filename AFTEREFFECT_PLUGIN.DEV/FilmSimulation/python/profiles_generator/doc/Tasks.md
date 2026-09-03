Open items, ranked by impact on simulation accuracy.

| # | ID | Name | What it needs (≤10 words) |
|---|---|---|---|
| 1 | **D2** | Scanner transfer + noise floor | Your one step-wedge scan; splits emulsion σ from scanner σ |
| 2 | **C18** | `density_weighting` saturating asymptote | Blocked on D2; largest undocumented number in colour path |
| 3 | **D1** | Absolute base+fog | One empty-gate scan, no film, same settings |
| 4 | **—** | [VTD]/[VE] route grain field | New inert `GrainSpec` scalar; your schema approval |
| 5 | **F2b** | B&W negative σ(D) plot | ⚠ Higgins & Stultz no longer the only route — see below |
| 6 | **M1** | Print-stock spectral configuration | Choose a profile rendering through a held print stock |
| 7 | **G6** | Agfa "lines/mm" axis meaning | Unambiguous statement; unlocks twelve MTF/resolving pairs |
| 8 | **—** | AGFA_VISTA_200 f50 | Adopt traced 50.0 over estimate 56/63/69 |
| 9 | **—** | Spectral for three new Fuji stocks | Fuji sheet with numbered sensitivity ordinate |
| ~~10~~ | ~~**G5**~~ | ~~Kino-Technik re-scan~~ | **CLOSED 2026-09-03 — needed no re-scan and no owner action** |
| 11 | **C23** | Bromide drag / directional exhaustion | Needs a processing-side spec that does not exist |
| 12 | **K6** | PORTRA 100T sensitometry | Acquire; E-2468 prints 160VC's artwork instead |
| 13 | **K5** | rms granularity, eight Kodak stills | Acquire; proved unobtainable across 201 Kodak files |
| 14 | **C14** | EKTAR 125 Kodak publication | Acquire; one measured number already recovered from patent |
| 15 | **F1** | Bayer/Wilder/Trabka JOSA papers | ⚠ Shrunk — Bayer's closed form in hand; Wilder doubtful |
| 16 | **F3** | 5 nm spectral re-trace | Nothing — self-declared near-worthless, 0.4–1.1 % benefit |

⚠ **Updated 2026-09-03.** **G5 is closed** — it asked for a re-scan it did not need, and three of its four blockers had already been discharged by G2; six channel shapes went [T2] → [T1] on GEVACHROME_600/605 plus the database's only measured reversal push. **F2b and F1 both moved** on Lu & Torquato 1990: all 11 measured colour negatives already show the σ(D) turnover that paper predicts (peak at D 0.65–0.80, then a fall to ×0.35–0.74 where √D predicts a rise of ×1.66–1.84), so F2b has a principled one-parameter shape available without Higgins & Stultz, and F1's Bayer target is in hand while its Wilder citation looks wrong.

Rows 4, 8, 9 are not formal queue rows — open gaps recorded yesterday. Everything above **row 10** either changes rendered accuracy directly or unblocks something that does; **11–16** are acquisitions or dead ends.

Fastest real gains available today without anything from you: **#8** (one stock, measured value replacing an estimate) and **#4** (inert field, bit-exact by default).
