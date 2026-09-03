# RESULT: FUJI + AGFA new extractions (2026-08-16)
Paths relative to /sessions/vigilant-wonderful-dijkstra/mnt/PYTHON.TST/PDF/PROFILES/

## 1. FUJICOLOR (SUPER) F-500, 35mm Type 8572 / 16mm Type 8672
FILE: FUJI/F500 - 8572.pdf (2pp, true text)
- EI: 500 tungsten 3200K; 320 daylight w/ Fuji LBA-12 or Kodak No.85 (p2)
- Balance: colour-balanced for 3200K tungsten, no filter at 3200K (p2)
- Full CC filter table (p2): Tungsten 3200K none/500; Daylight LBA-12|No.85/320; Metal halide (HMI) LBA-12|No.85/320; Fluorescent white CC-30R/250; fluorescent daylight LBA-12|No.85/320; 3-band 5000K CC-30R/250; 3-band 6700K CC-40R/200
- RMS granularity: 4.0 (x1000, visual diffuse density 1.0 above Dmin, 48 um aperture) (p1)
- Sharpness: Contrast Transfer Function graph 1-100 cycles/mm (no numeric resolving-power TOC values given; CTF replaces resolving power on Fuji cine sheets) (p1)
- Gamma: not stated numerically; characteristic curves (density 0-3.0 vs log H lux.s), 3200K 1/50s, SC-41 filter, status M (p1, graph only)
- Reciprocity: no filter or aperture correction 1/1000-1/10 s; at 1 s open 1/3 stop; NO CC filter needed (p2)
- Process: NOT stated in this 2pp file (no ECN-2 paragraph, no base/safelight section - sheet is curves page + marketing/tech page). Sensitometry says "specified standard conditions" only.
- Base: NOT stated in this file.
- Tech: SUFG (Super Uniform Fine Grain) hexagonal-tabular grain, Super/Two-Stage-Timing DIR couplers; edge code FN72, film name FUJI F-500 (p2)
- Vector inventory: p1 = 282 drawings / 3 images -> characteristic, spectral sensitivity, spectral density, CTF curves are ALL VECTOR. p2 = 106 drawings / 6 images (photos/diagrams).
- vs stored FUJICOLOR_SUPER_F500_8572 (film_profiles.cpp line 2157): EI 500 / 3200K MATCH. Tier can be upgraded from T2 "no manufacturer datasheet" - this IS the datasheet.

## 2. FUJICOLOR ETERNA Vivid 500, 35mm Type 8547 / 16mm Type 8647
FILE: FUJI/eterna_vivid500.pdf (2pp, Ref. No. KB-0901E, (c)2009 FUJIFILM)
- NEW NUMBER: RMS granularity 3.5 (x1000, D=1.0 above Dmin visual diffuse, 48 um aperture) (p1)
- CTF graph 1-100 cycles/mm, density 1.3 visual diffuse (p1); no numeric resolving power, no numeric gamma (graph only; char. curve x-axis in CAMERA STOPS -6..+6, 1/50s 3200K, SC-41, status M)
- EI 500 tungsten / 320 daylight (LBA-12/No.85); same 7-row filter table as F-500 (p2)
- Reciprocity: none 1/1000-1/10 s; +1/3 stop at 1 s (p2)
- Process: ECN-2 (persulfate, ferricyanide or PDTA-ferric/UL bleach OK) (p2)
- Base: triacetate safety base, tinted light cyan (p2); edge code FN47, name FUJI V500
- Vector inventory: p1 = 228 drawings / 24 images -> all four curve sets VECTOR. p2 = 168 drawings / 16 images.

## 3. FUJICOLOR NEGATIVE FILM F-125, 35mm Type 8532 / 16mm Type 8632
SOURCE: SCREENSHOT IMAGE (FUJI/F125 - 8532.png) - raster only, no vector curves; all values transcribed from image.
- Branding: "SUPER F" logo, 2002 FIFA World Cup sponsor artwork -> ca. 2001-2002 generation. Doc code "C41E S EN RI-SGNS".
- RMS granularity: 3.0 (0.001 units, visual diffuse density 1.0 above Dmin, 48 um aperture)
- Characteristic curves: Exposure 3200K tungsten 1/30 s; Developing "Super-FUJI Standard"; densitometry status M; density axis 0-3.0 vs log lx-seconds -2.0..+2.0; approx Dmin B~0.75, G~0.5, R~0.2
- CTF graph 1-100 cycles/mm, density 1.0 visual diffuse; spectral sensitivity plotted on LINEAR % scale (unusual), 380-700nm spectral density curves
- No EI/reciprocity/base panel visible in the screenshot (curves side only)
- Relation to stored FUJI_F125_8530/8630: SAME film family (Fujicolor F-125 tungsten camera negative). 8532/8632 are the later "Super F-series" type numbers succeeding 8530/8630 (same 35mm/16mm pairing convention). NOT a different base/format - it is a generation successor. Our stored EI 125 / 3200K remains consistent; RMS 3.0 applies to the 8532 generation, do not back-apply blindly to 8530.

## 4. AGFACOLOR Vista 100/200/400/800
FILE: AGFA/AGFACOLOR Vista 100, 200, 400, 800.pdf (8pp, Agfa Technical Data AF, 06/2000 2nd ed., true text, curves ALL VECTOR - 0 images on pp5-8)
Common: process AP 70/C-41 (p3); daylight 5500K balance (p3); reciprocity flat 1/10000-1 s (p3,p5); base acetyl cellulose 120 um (35mm), 110 um pocket (p4); RMS at diffuse D=1.0, 48 um aperture, visual filter (p4); resolving power at TOC 1000:1 and 1.6:1 (p4); artificial light: photo lamp 80B +1 2/3 stops, bulbs 80A +2 stops (p3).
| Film | ISO | RMS(x1000) | RP 1000:1 | RP 1.6:1 | Latitude | Layer thk | Recip @10s | PAGE |
|---|---|---|---|---|---|---|---|---|
| Vista 100 | 100/21 | 4.0 | 130 l/mm | 60 l/mm | -2..+3 | 16 um | +1/2 stop | p5 |
| Vista 200 | 200/24 | 4.3 | 130 l/mm | 50 l/mm | -1.5..+3 | 18 um | +1 stop | p5(recip)/p6 |
| Vista 400 | 400/27 | 4.5 | 130 l/mm | 50 l/mm | -1..+3 | 19 um | +1 stop | p5/p6 |
| Vista 800 | 800/30 | 5.0 | 110 l/mm | 40 l/mm | -1..+3 | 22 um | +1/2 stop | p5/p6 |
No CC filter needed for long exposure on Vista (only slide CTprecisa gets CC). Spectral sensitivity + spectral density + MTF + colour density curves present per film, VECTOR (p5-p7). Bonus: Futura II 100/200/400 and CTprecisa 100/200 data on p7-p8 (CTprecisa 100: RMS 10.0, 130/50 l/mm, 25 um; CTprecisa 200: RMS 12.0, 120/50 l/mm, 27 um).
- vs stored AGFA_VISTA_200 (film_profiles.cpp line 393): ISO 200 / 5500K MATCH; "no datasheet" claim REFUTED.

## 5. Enticknap 2013 (GEVAERT/enticknap_2013_film_restoration.pdf, 126pp)
VERDICT: purely historical/archival prose. Keyword sweep: zero RMS, zero resolving power, zero ASA/DIN/ISO+number, zero spectral-sensitivity data. "gamma" (6 pp) and "granularity" (1 p) appear only in restoration-workflow discussion (scanner settings, duplication grading). Kelvin values are projector-lamp colour temperatures (carbon arc ~5000K, xenon 4000-4500K, p15/p88 glossary). Gevacolor/Gevachrome hits: index only (p120). NO per-product data - not a profile source.
