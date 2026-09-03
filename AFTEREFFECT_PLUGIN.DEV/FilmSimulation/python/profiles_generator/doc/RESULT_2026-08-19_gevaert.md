# RESULT 2026-08-19 — GEVAERT: two new stocks, one re-traced curve set

**Task:** add the fully specified stock(s) found in the GEVAERT folder; digitise the curves in the
1980 SMPTE paper at the highest practical resolution; fill missing GEVA parameters from
source-supported values only; keep provenance to the page; distinguish measured from estimated.

**Build after the work:** `build.py --root <corpus>` **OK**, **9 audits green** (new:
`gevaert_curves.py`), `verify.py` **255 PASS / 2 FAIL** (the two known baseline failures), C++ clean
on 18 TUs, database **155 → 157 stocks**.

---

## 1. Two stocks added: `GEVACHROME_600` and `GEVACHROME_605`

Source: **Rens, J. E. / Van Bets, K., "Gevachrome-Farbumkehrfilme für Farbfernsehen",
KINO-TECHNIK 1968 Nr. 10, printed pp. 260 / 262 / 264 / 266.** Agfa-Gevaert's colour **reversal
camera** films for colour television. The database already held their recommended *print* stock
(`GEVACHROME_902`, T.9.02) but no Gevaert reversal camera stock at all.

| | Typ 6.00 | Typ 6.05 | source |
|---|---|---|---|
| EI, tungsten 3200–3400 K | **18 DIN (50 ASA)** | **23 DIN (160 ASA)** | Tab. II, p262 |
| EI, daylight 5500–6500 K + CTO 12 | 15 DIN (25 ASA) | 20 DIN (80 ASA) | Tab. II — **filter-derived, deliberately not stored** |
| push | — | **26 DIN (320 ASA)**, first developer +≈45 s | Tab. II footnote 1 + p264 |
| γ yellow / magenta / cyan | **1.25 / 1.25 / 1.45** | **1.25 / 1.25 / 1.35** | Bilder 5a/5b caption, p264 |
| layer stack | 9 layers, blue/green/red, yellow **and magenta** filter layers, black anti-halation | same | Tab. I, p262 |
| projection balance | 5000–6000 K | same | p262 |
| Dmin | **0.10** (traced 0.09–0.12) | same | Bild 5a/5b right plateau |

Stored per the standard chromogenic mapping — yellow layer records blue, magenta records green,
cyan records red — so `curves.r.gamma = 1.45` for 6.00 and `1.35` for 6.05.

**What is NOT in the source, and is therefore tagged `[T3]` in the profile:** any granularity figure
(no σ_D, no rms, not even a curve), any resolving-power number, Dmax, reciprocity, base thickness.
The grain and MTF entries are class estimates and their comments say so; `verify.py` asserts the
provenance string still contains the "NOT PRINTED … granularity" statement, so the estimate cannot
later be mistaken for a reading.

⚠ **The 1968 scan limits what could be traced, and that is measured rather than assumed.** Its
embedded page images are **JPEG colour at 150 ppi** (940 × 1345 px) with ink bleed-through from the
reverse of the sheet — against the 1980 paper's **1-bit 340 ppi** (2277 × 3248 px). On Bild 5a the
three layer curves lie **within 1–2 px of each other** over most of their length and coincide
entirely at the right-hand end, which is consistent with the paper printing the *same* γ 1.25 for
two of them. A three-way separation would have been fabricated, so it was not attempted. What the
figure did yield, traced off its own gridlines (94.0 px per D, 92.5 px per decade, both measured):
the lower curve pair runs D **2.20 → 1.00 → 0.26 → 0.09** at lg i·t 0.1 / 1.0 / 2.0 / 2.9.

⚠ **Dmax is not determined by the figure.** The curves are still climbing at the plot's left edge, so
2.2 is a measured **lower bound**, not a shoulder. Toe and shoulder softness are transferred from
`GEVACHROME_902` — same manufacturer, same process family, adjacent years — and that transfer is
labelled `[T2]`, not presented as a measurement of these films.

**Also on file but not stored, because no carrier fits:** the complete 12-step process (Tab. IV, two
temperature columns) with full formulae and pH for GP 110, GP 332, GP 26, GP 308, GP 446, GP 660; the
required illumination table (Tab. III, lux and foot-candles for f/1.4–f/11); storage temperatures.
`ProcessingFamily` wants gamma-against-time points and the paper prints curves, so inventing rows
would have been the only way to use it.

⚠ **This changed the ListBox.** Two stocks inserted → `eTOTAL_FILMS_PROFILES`, the generated enum and
every `film_names.txt` line index after `GEVACHROME_600` shifted. `film_names.txt` MD5
`ae9e4be3…` → **`c37a188b…`**.

## 2. `GEVACOLOR_NEG_682`: characteristic curves re-traced from Fig. 10

Source: **Vervoort, A. / Stappaerts, H., "A New Gevacolor Negative Film Type 682", SMPTE Journal
89(9), September 1980, pp. 650–652**, Fig. 10 on printed p652, annotated "STATUS M MEASUREMENT".
(The filename says *Verpoort*; the paper prints *Vervoort*, which the existing citation already had
right. Three dates are printed and all are true: presented 22 Oct 1979, published April 1980 in the
BKSTS Journal, reprinted September 1980 in the SMPTE Journal.)

**Digitised at one sample per pixel column** — the owner's instruction was not to reduce curves to a
few representative points — from the **native-resolution embedded scan**, extracted with `pdfimages`
rather than re-rendered, because re-rendering at 200 dpi would discard 40 % of the columns:

| layer | line style | samples | fitted dmin | fitted γ | fit rms | worst |
|---|---|---|---|---|---|---|
| B | solid | **589** | 0.9137 | 0.5396 | 0.0063 D | 0.0177 D |
| G | dash-dot | **513** | 0.5863 | **0.5677** | 0.0040 D | 0.0165 D |
| R | dotted | **437** | 0.1356 | 0.5056 | 0.0055 D | 0.0161 D |

`dmin` is **pinned to the measured left-plateau median** (119 / 101 / 78 samples), not fitted. The
other five ToneCurve parameters are least-squares fitted to all samples. Calibration: density axis
**166.4 px per D** (3 printed ticks, worst residual 0.0026 D), exposure axis **148.3 px per decade**
(5 printed ticks, worst residual 0.0081 decade).

**The external check that licenses all of it:** the figure prints **"γ = 0.57"** on the green curve.
The trace, calibrated only from tick positions, fits green at **0.5677** — agreement to 0.002 on a
number the extractor never saw. `verify.py` pins that comparison.

Two things the old entry had that the trace corrects: the stored γ was the printed 0.57 applied to
all three layers (the plot shows a per-layer spread, 0.506–0.568), and dmax was 2.08/2.51/2.75 where
the figure gives **1.48 / 2.01 / 2.26**.

⚠ **The abscissa origin is inherited, not measured, and is now frozen.** Fig. 10's axis reads
"LOG REL. EXP." 0 → 4.00 with no absolute anchor — no speed point, no lux-seconds — so it cannot be
tied to this database's mid-grey origin from anything printed. The offset was chosen so the green
record reaches the same net density at x = 0 that the previous hand-fitted curve gave (0.876 D),
frozen as a constant in the extractor, and applied to **all three layers equally** because they
share one exposure axis. Shape, gamma and dmin are measured; absolute exposure placement is exactly
as uncertain as before.

## 3. The scan is skewed, and pretending otherwise costs 0.05 D

Measured on Fig. 10: the density axis sits at x = 146 at the top of the plot and x = 155 at the
bottom — a **1.40° rotation**. Both axes are therefore fitted as *lines*, not read as a column
index. Three axis-finding rules were tried and two failed measurably before the third worked:

* *column of maximum ink mass* → picked the figure's **right frame edge** (denser and straighter
  than the tilted axis);
* *leftmost column with ink down most of the height* → picked the rotated **"DENSITY" caption**;
* *leftmost near-full-height stroke whose per-row positions fit a line to <1.5 px over ≥60 % of
  rows* → the axis. Straightness is what separates an axis from a caption.

The same correction was needed horizontally: "lowest band with ink across the width" locked onto the
row of printed **x-axis labels**, 15–20 px below the real axis, which pulled the trace box past the
axis and handed the tracer the axis itself as a curve.

**Tick anchors are pinned pixel positions, re-verified every run.** A fully automatic tick finder is
in the file and is not trusted unaided on a 1-bit skewed scan — it also caught the rule above the
figure and the bottom axis line. The anchors were located with it, checked against the printed axis
by eye, frozen, and `verify_anchor()` re-checks each one against the pixels on every extraction, so
a replaced or re-cropped scan fails loudly instead of silently rescaling a curve.

## 4. Two self-referential loops found and closed

Both would have produced numbers that looked fine and could not be reproduced.

1. **The fit was seeded with its own previous answer.** `fit_layer` originally took the stored
   profile parameters as the Nelder-Mead start. Once this extractor's output had been adopted, that
   meant seeding the fit with what it had produced — and the red record (sparsest trace) moved to a
   *different local minimum* on the next run: γ 0.5446 at rms 0.0109 D became **0.5056 at rms
   0.0055 D**. The lower-residual solution is what ships. Start points are now derived from the data
   (steepest 1-decade slope; where the curve leaves dmin + 0.10 and reaches max − 0.10) and the
   optimiser runs a **fixed multi-start grid** — deterministic, because an audit that cannot be
   re-run identically is not an audit.
2. **The inherited origin was read live from the profile it feeds**, so it drifted by 0.002 D on the
   run after adoption. Now a frozen constant in the figure spec.

## 5. `cpp_parity.py` corrected — it was testing data freshness, not the law

The new parity audit failed with a **1.5e-02** disagreement the first time a curve was re-traced, and
the law was not at fault: `build.py` runs audits **before** codegen, so the C++ copy still held the
previous `dmax` while Python held the new one. Two implementations of one law were being fed
different data and blamed for it. The probe now **passes identical GrainSpec fields, dmin, dmax and
density into the C++ side as literals**, so it tests the function; data freshness is already covered
by the sync stage and by `verify.py`. Side effects: no database translation units to compile, so the
audit is ~20× faster, and a positional-initialiser guard (`check_field_order`) fails if a `GrainSpec`
field is ever inserted upstream. Worst disagreement across **4710 probes: 2.67e-07**.

Also fixed while there: `%.9g` renders 12.0 as `12`, and `12f` is not a C++ float literal — g++
reports it as *"unable to find numeric literal operator"*, which is an obscure way to be told that a
clump diameter is a whole number.

## 6. What the other two documents yielded

| document | outcome |
|---|---|
| Webers & Westendorp, "Einführung in die Kopierwerktechnik (XIV)", *Fernseh- und Kino-Technik* 33(7) 1979, pp. 245–247 | The **Gevachrome II process**, 15 steps with bath formulae, pH, temperature tolerances and replenishment rates, and four type numbers (`Gevachrome S 700`, `710`, `Gevachrome D 720`, `Print 780`). ⚠ **No sensitometry whatever** — no speed, gamma, Dmin/Dmax, granularity, MTF or spectral data — so it cannot support profiles. Queue G4 |
| Enticknap, *Film Restoration*, 2013 | ⚠ **No Gevaert stock data.** "Gevaert", "Gevacolor" and "Gevachrome" appear nowhere in the body text; the only hit is a subject-index heading "Agfa-Gevaert, N.V." pointing at printed pp. 66 and 221, where the printed word is plain "Agfa" — once in a five-manufacturer list, once in a French bibliography title. Both checked against the page image, not just the OCR layer |
| `Gevachrome902.pdf` | ⚠ **Byte-identical duplicate** (MD5 `80ce5885…`) of the AGFA-folder copy already cited for `GEVACHROME_902`. Not a new source |

## 7. Deliberately not done, with reasons

* **Fig. 7 (spectral sensitivity), Fig. 8 (spectral dye density), Fig. 11 (MTF) of the 1980 paper.**
  All three are traceable and all three are queued (**G3**). They need a different
  curve-identification rule from Fig. 10: those are overlapping *humps* with no consistent
  top-to-bottom ordering, where Fig. 10's three curves never cross and can be seeded at a plateau
  and traced rightward. The database already holds spectral and dye-density sets for 682 from earlier
  work, so the gain there is **validation**, not new data — unlike the characteristic curves, which
  were the explicit priority and were estimates before today.
* **Bild 1a–c (MTF) and Bild 2a/2b (spectral sensitivity) of the 1968 paper** — queued as **G2**, and
  worth a better scan first: at 150 ppi with bleed-through, a traced spectral curve would carry more
  scanning artefact than film. **Acquisition ask: a 300+ ppi grayscale re-scan of printed pages 260,
  262 and 264** would make G2 straightforward and would let Bild 5a/5b be separated into three
  layers, which is the one thing this pass could not do.
* **`ProcessingFamily` for any of the three stocks** — the sources print process *curves* and *bath
  recipes*, not the gamma-against-time points the carrier stores.
