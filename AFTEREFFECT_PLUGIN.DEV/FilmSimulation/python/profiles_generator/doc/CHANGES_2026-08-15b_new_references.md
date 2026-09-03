# New reference batch: 15 files + KODAK DATA BOOK + Zhurba 1984/1990

**Date:** 2026-08-15 (second session)
**Full extraction reports:** `doc/reanalysis_2026-08-14/` gained
`RESULT_KODAK_NEW.md`, `RESULT_FUJI_AGFA_NEW.md`, `RESULT_ZHURBA.md` (copied from /tmp).

## 1. Summary

| | before | after |
|---|---|---|
| Film stocks | 142 | **143** (`KODAK_TECHNICAL_PAN`) |
| verify.py | 121 PASS / 2 FAIL | **121 PASS / 2 FAIL** (same two pre-existing) |
| Citing documents | 124 | **127** |

**Three RMS granularity corrections, all ~2× off, all manufacturer-settled:**

| Stock | was (estimate) | now (documented) | Source |
|---|---|---|---|
| `FUJICOLOR_SUPER_F500_8572` | 7.4 | **4.0** | `FUJI/F500 - 8572.pdf` p2, 48 µm, D=1.0 |
| `AGFA_VISTA_200` | 9.4 | **4.3** | Vista brochure (AF 06/2000), 48 µm |
| `FUJI_ETERNA_VIVID_500T_8547` | 6.8 | **3.5** | `FUJI/eterna_vivid500.pdf`, 48 µm |

The first two stocks were in the gap chapter's "genuinely absent" list — **both now
settled by their own sheets.** Also: `AGFA_VISTA_200` resolving power (50, 130) entered;
reciprocity for 8572 (achromatic, onset 0.1 s, p 0.90 [C2]) and for
`KODAK_VISION2_250D_5205` (chromatic CC10R, **two-point fit** p 0.90 with red at 0.87,
onset 0.1 s) entered.

## 2. New stock: KODAK_TECHNICAL_PAN (publication P-255, 12 pp)

The most extreme processing-dependent film Kodak sold: **CI 0.50 → 2.50 from one
emulsion** (Technidol LC at EI 16–25 through Dektol at EI 200), RMS **5** in Technidol /
**8** in HC-110 D, **extended red to 690 nm**, 4-mil ESTAR-AH with 0.1 ND tint. Entered
at the pictorial condition (Technidol LC, CI 0.50, EI 25) with `ProcessingSpec` naming it.
**Honest caveat, in the profile:** P-255 prints *no* numeric resolving power — the famous
320+ lp/mm circulates from literature not in this corpus, so the MTF is EST from the
granularity class and `_RESOLVING_POWER` gets no entry.

## 3. Kodak group findings (RESULT_KODAK_NEW.md)

- **Portra gap refuted:** `E-190` (2006, 14 pp, vector curves) documents 160NC/160VC/
  400NC/400VC/800 — PGI 36/40/44/48/48, reciprocity none to 10 s (NC/VC) / 1 s (800),
  C-41 Status M. **NOT merged into our `KODAK_PORTRA_160/800`**: those model the 2010s
  reformulation; NC/VC is a different generation. Candidates recorded in
  `next_week_task.md`.
- **2383 print stock:** numeric **LAD 1.09/1.06/1.03** (Status A) from the 2015 sheet;
  2022 sheet is ECP-2E, raster-only. 2242/3242/5242 intermediate: LAD MP 1.15/1.60/1.70,
  DN 1.00/1.45/1.55 ± 0.10.
- **5366/7366 dupe positive:** γ aim 1.20–1.60 (Status M blue), **RMS 9,
  RP 100/200 lp/mm** — the only numeric RP in the batch; PrintStock candidate.
- **5205, 7213 sheets:** agree with our stored EI/balance; 7213 = 16 mm of our 5213.
- **KODAK DATA BOOK** (1495 pp): UK "Data Books of Applied Photography", 5 loose-leaf
  volumes, sheets ~1948–1968. **Volume 5 FILMS = pp 1150–1495.** Documents Plus-X,
  Super-XX, Tri-X, Panatomic-X, Verichrome Pan, Royal-X, Kodachrome/Ektachrome families.
  ⚠ Speeds are post-1960 ASA (no safety factor) — **halve before comparing with our
  1952-era stocks.** Queued for a dedicated pass in `next_week_task.md`.

## 4. Fuji/Agfa group (RESULT_FUJI_AGFA_NEW.md)

- **`F125 - 8532.png`** (screenshot): Fujicolor F-125 **Type 8532/8632 — the successor
  generation** to our 8530/8630, same film family. RMS 3.0 (48 µm) **applies to the 8532
  generation and was NOT back-applied.** Values transcribed from image, marked as such.
- `eterna_vivid500.pdf`: all curves vector; ECN-2 with persulfate bleach; light-cyan
  tinted triacetate.
- **Enticknap 2013 (126 pp): discard as a data source** — zero per-product numbers;
  restoration-workflow prose. Recorded so nobody re-reads it.

## 5. Zhurba 1984 (Лабораторная обработка фотоматериалов)

Tables 2–5 are **rotated 90°** and their OCR is useless — read visually from rotated
renders (kept in outputs/ as `z84_p11_r270.png`…). Yield: full B&W and colour tables for
the SVEMA/TASMA lines, Фото-65 γ–time curve family (p 60), ЦО processing recipes.
Highlights against held stocks (book pp 10–15, tables — corroboration of Gurlev/GOST):
ОЧ-45 res 100–110 л/мм, γ 1.2–1.6, 660 nm limit (**confirms** ГОСТ 20945-80); ДС-4
γ 0.7–0.85 (**consistent** with Gurlev's 0.8); ЦНЛ-65 latitude 1.5, res 63; ЦНЛ-32 res 58;
Фото-65 latitude 1.5, res 92, fog 0.10–0.16. **ORWO: zero occurrences in the whole book**
— NC21/NC24/UT18 stay undocumented.

## 6. Online Zhurba 1990 (djvu.online) — honest access report

Landing page reachable (via a `?x=1` variant); its embedded OCR text covers **book pages
~3–25 only** (front matter and chemistry) and stops. The target section «ФОТОГРАФИЧЕСКИЕ
МАТЕРИАЛЫ» pp 44–131 exists as **178 webp page images that web_fetch returns empty**;
`/text/`, `?page=N` and download routes yield nothing further. Per the web-content rules
no other retrieval method was attempted. **pp 44–131 remain unread** — if a PDF/DjVu of
this book can be placed under `PDF/PROFILES/SOVIET/`, it becomes a normal local
extraction.

## 7. Files changed

`film_profiles.py` (3 RMS corrections + Tech Pan + 2 reciprocity + resolving + 3
provenance rows + ProcessingSpec), `verify.py` (count 143), regenerated
`film_profiles.hpp/.cpp`, `film_enum.hpp`, `film_names.txt` (143/142), regenerated
reports, `NotFound.md` (8572 + Vista rows settled; Portra row narrowed; Zhurba access
recorded), `Found.md`, `next_week_task.md` (Portra NC/VC candidates, DATA BOOK vol-5
pass, 5366 PrintStock), `README.md`, master doc + Russian mirror follow-up, all copies
synced.
