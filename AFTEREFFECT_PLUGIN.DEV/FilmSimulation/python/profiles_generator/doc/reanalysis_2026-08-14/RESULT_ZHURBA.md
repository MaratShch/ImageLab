# RESULT — Zhurba data extraction (2026-08-16)

## Sources
- **[Zh84]** Журба Ю.И., «Лабораторная обработка фотоматериалов», Искусство, М., 1984 (local PDF, 178 pp). Tables 2–5 (book pp. 10–17 = PDF pp. 11–18) are printed ROTATED 90°; OCR there is garbage. All table pages were rendered at 200 dpi, rotated (PIL ROTATE_270) and read visually. Renders kept in outputs/: z84_p11_r270.png … z84_p18_rot.png, z84_p61.png.
- **[Zh90]** Журба Ю.И., «Краткий справочник по фотографическим процессам и материалам», 3-е изд., 1990, via https://djvu.online/file/tBoewNhmlnFMB — see access report at end.

Units: sensitivity everywhere below is **ед. ГОСТ** (общая светочувствительность). Contrast = коэффициент контрастности (γ). Resolving = разрешающая способность, лин/мм. Latitude = фотографическая широта (log units as printed). Fog = оптическая плотность вуали. Dev time at 20°C unless stated.

---

## (a) CONFIRMS existing holdings
[Zh84] Табл. 2, book p.10 (read from image, stated table values):
| Film | S, ед. ГОСТ | γ recommended | γ max | Fog D0 | Latitude | Resolving, лин/мм | Dev (developer / min) |
|---|---|---|---|---|---|---|---|
| «Фото-32» | 32 | 0,8 | 1,0–1,4 | 0,05–0,1 | 1,5 | 116 | №2 ГОСТ 10691.2-73, 6–10 |
| «Фото-65» | 65 | 0,8 | 1,0–1,4 | 0,1–0,16 | 1,5 | 92 | №2, 6–10 |
| «Фото-130» | 130 | 0,8 | 1,0–1,4 | 0,16–0,25 | 1,5 | 75 | №2, 8–14 |
| «Фото-250» | 250 | 0,8 | 1,0–1,4 | 0,2–0,3 | 1,5 | 70 | №2, 8–14 |
(Confirms the Фото family we hold; note book has Фото-130, not "Фото-125".)

## (b) NEW numbers for held stocks
[Zh84] Табл. 2 cont. (book p.11) — **ОЧ-45** (we hold TASMA_OCH_45), reversal:
S=45 ед. ГОСТ; γ 1,2–1,6; **Dmin 0,06**; latitude **1,05**; resolving **100–110**; sensitization limit **660 нм**; melt temp 70°; first dev 6–12 min. (Sister data: ОЧ-180: S180, γ1,2–1,6, Dmin 0,08, lat 0,9, res 82–95, 660–680 нм; ОЧ-Т-45: S45, γ1,0–1,25, Dmin 0,07, lat 1,05, res 85, 650 нм, 8–10 мин; ОЧ-Т-180: S180, γ1,2–1,5, Dmin 0,12, lat 0,9, res 73, 690–720 нм; ОЧ-Т-45М: S45, γ1,1–1,4, Dmin 0,08, lat 0,9, res 85, 650 нм.)

[Zh84] Табл. 4 (book p.15) — colour negatives (we hold CNL_32/65, DS_4):
| Film | S | γ all layers | γ mid+bottom | γ top | Balance ≤ | Fog+mask B | G | R ≤ | Latitude | Res | Dev 20°, мин |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ДС-4** | 45 | 0,7–0,85 | — | — | 0,12 | 0,25 | 0,25 | 0,25 | **1,2** | **63** | 5–8 |
| **ЦНД-32** | 32 | 0,6–0,8 | 0,7±0,1 | (0,7±0,1)+0,1–0,2 | 0,1 | 0,75–1,1 | 0,25–0,45 | 0,3 | **0,9** | **58** | 5–8 |
| **ЦНЛ-32** | 32 | 0,6–0,8 | 0,7±0,1 | (0,7±0,1)+0,1–0,2 | 0,1 | 0,75–1,1 | 0,25–0,45 | 0,3 | **0,9** | **58** | 5–8 |
| **ЦНЛ-65** | 65 | — | 0,7±0,1 | (0,7±0,1)+0,15 | 0,1 | 0,75–1,1 | 0,4–0,6 | 0,3 | **1,5** | **63** | 5–8 |
New: per-layer gamma structure (top layer runs +0.1–0.2 hotter), interlayer balance limit, blue/green/red fog+mask densities, latitude & resolving per film.

[Zh84] Fig. 14, book p.60 (GRAPH, read visually): family of characteristic curves of **«Фото-65»** in standard developer for t = 2, 4, 6, 8, 10, 12, 16, 20, 24, 30 min; axes D 0–3,0 vs lgH −3,0…+1,0. Gives gamma-time behaviour: 2 min curve tops ~D1,1; 30 min reaches D~2,9 with toe near lgH≈−2,7. Render saved: outputs/z84_p61.png.

[Zh84] p.83: sensitometric-test dev times: Фото-32/65 4–10 мин, Фото-130/250 6–14 мин in №2; КН-1/2/3 7–13 мин, НК-1/2 4–8 мин, НК-4 8–12 мин in №5 (ГОСТ 10691.3-73, recipe given: метол 1,6 г, сульфит 100 г, гидрохинон 2 г, бура 2 г, KBr 0,4 г).

## (c) Films we do NOT hold
[Zh84] Табл. 2 (pp.10–12, images):
- Кинопленки негативные: КН-1 S11, γ0,65, fog 0,1–0,13, res 135, 650 нм, №5 7–13 мин; КН-2 S32, fog 0,12–0,15, res 100; КН-3 S90, fog 0,15–0,20, res 78; НК-1 S32, fog 0,06, res 120, 670 нм, 4–10; НК-2 S90, fog 0,10, res 110; НК-3 S250, fog 0,12, res 90, 5–11; НК-4 S350–500, fog 0,20, res 75, 8–14. All γ=0,65.
- ЗТ-8 (sound): S16–32, γ3,6, fog 0,05–0,08, res 240, 570 нм, 3,5–5 мин.
- МЗ-3/МЗ-3М (positive): S2,8–5,5, γ2,6 (max 2,8–3,2), fog 0,04, res 110, dev ГОСТ 10691.4-73 2–4 мин.
- Plates «Фото-90/130/180/250/350»: S=90…350, γ0,9–1,7 (max 1,2–2,0), lat 1,2/0,9, res 70…55, №1 ГОСТ 10691.1-73, 4–8 мин.
- Фототехнические ФТ-10…ФТФ-2 (p.12 image): S0,2–130, γ1,0±0,1…10,0, res 73–250, own developers (№2, ФТ-2, ИП-3).
[Zh84] Табл. 4 (pp.15–16, images):
- ЦНЛ-90: S90, γ0,65±0,05, bal 0,12, fog 0,9±0,15 / ~0,5±0,1 / 0,3, lat 1,3, res 63, melt 30°, 5–8 мин.
- ЦОД-16/ЦОД-32: γ1,6–2,2, bal 0,2, res 45, melt 30°.
- ДС-5М: S22, ср. градиент 0,6±0,1, bal 0,12, lat 1,05, res 58, 5–7 мин.
- ЛН-7: S65, 0,6±0,05 (all layers), lat 1,5, res 63, melt 33°, 5–7 мин. ЛН-8: S100, same γ, lat 1,5, res 70.
- КП-М S0,2 / КП-6 S0,1, γ1,0–1,15, lat 1,2, res 73/—.
- Positives: ЦП-8Р S0,2–0,75, γ3,3±0,3, bal 0,4, fog 0,18, res 75, 9 мин; ЦП-11 S0,3, γ3,0±0,3, fog 0,15, 7–8 мин.
- Reversal (Dmax/Dmin per layer, useful exposure interval, 1st B&W dev): ЦО-22: S22, γ1,8–2,2, bal 0,3, Dmax 2,2, Dmin 0,25, interval 1,2, res 70, melt 50°, 8–14; ЦО-32Д: S32, 1,8–2,2, 2,2/0,25, 1,2, 53, 35°, 8–12; ЦО-65: S65, γ1,9–2,4, 2,3/0,25, 1,2, 68, 50°, 7–11; ЦО-90Л: S90, 1,8–2,2, 2,2/0,25, 1,2, 53, 8–12; ЦО-180Л: S180, γ1,6–1,8, 2,1/0,25, 1,2, 58, 60°, 10–12; ЦО-Т-90Л: S90, γ1,6, 2,3/0,2, 1,1, res 82, 70°, 8–14; ЦО-6 (dup): S0,4, γ0,9–1,15, 2,0/0,25, 0,9, 68, 34°, 8–10.
- Табл. 5 (p.17): colour papers Фотоцвет-2 (S5–25) / Фотоцвет-4 (S3–12), γ glossy 1,8–2,4 (норм.) / 2,5–3,3 (контр.), Dmax≥2,0, fog ≤0,15–0,20 per R/G/B.
- Processing: ЦО-22 full recipes + Табл. 26 accelerated regimes (30°/40°: 1st dev 5,5–7,5 мин @30° or 1,5–2,2 мин @40°), book pp.169–170; ЦО-65 Табл. 28 @25°: 1st dev 7–11 мин, colour dev 8–12 мин (book p.171).

## (d) ORWO findings
**NONE.** The 1984 book contains zero occurrences of ОРВО/ORWO (full-text grep over the whole OCR layer = 0 hits); it covers only domestic (СВЕМА/ТАСМА) materials. The reachable fragment of the 1990 handbook (see below) also contains no ORWO text. ORWOCOLOR NC21/NC24 and ORWOCHROM UT18 remain undocumented from these two sources.

## Source B access report (djvu.online, 1990 handbook)
- Landing page fetched OK (via web_fetch; base URL was dedup-locked, variant `?x=1` worked). It embeds OCR text of the book **only up to book page ~24–25** (title/annotation, intro, «Состав и строение светочувствительного…», developer/fixer classification, diffusion process; text cuts mid-sentence on p.25).
- The target section «ФОТОГРАФИЧЕСКИЕ МАТЕРИАЛЫ» (pp. 44–131) is **NOT reachable**: page images exist as 178 numbered webp files (https://djvu.online/jpg1/t/B/o/tBoewNhmlnFMB/NNN.webp) but web_fetch returns empty for binary content; `/text/…` route returns empty; `?page=60` returns the identical landing text; `/download1/…` (captcha-gated binary) returns empty. No per-page text route exists on the server side without JS.
- Per instruction, no curl/wget/requests were used.

## Conflicts with Gurlev/GOST
None verifiable in this pass: the local Gurlev PDF has no searchable text layer for these film names (grep over extracted text = 0 hits), so a numeric cross-check was not possible here. Flag for the parent: [Zh84] resolving power Фото-32 = 116 лин/мм and ЦНЛ-65 latitude = 1,5 should be checked against the Gurlev-derived values in the engine.
