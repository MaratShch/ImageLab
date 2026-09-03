# RESULT 2026-08-31c — the corpus reconciliation, B3 and E3. Seven of thirteen "acquisitions" were not

**Task:** owner approved *"reconciliation + B3 + E3"* in one batch. **All three done.** Nothing new
queued.

**The headline is the reconciliation, not the two closures.** This working copy holds **56 PDFs**
under `PDF/PROFILES`; the owner's machine holds **475**, in twenty directories, of which this
checkout has six. Thirteen queue rows were filed under *acquisition*. **Six survive.**

---

## 1. The reconciliation — every live row against the real corpus

| row | recorded | found | outcome |
|---|---|---|---|
| **B3** | "P-255 and F-4043, not in this checkout" | both, in two editions each | **closed** |
| **E3** | "the KONICA files are not openable from this working copy" | all three, plus sixteen more | **closed** |
| **G5** | filed acquisition | its own text says *"the owner holds the source"* | → owner action |
| **M1** | filed acquisition | document arrived 2026-08-31 | → configuration |
| **T1** | "Kodak **E-4046** — VISION3 500T / 250D data sheet" | ⚠ **E-4046 is EKTAR 100.** The VISION3 sheets are **H-1-5219** and **H-1-5207**, both held | → stage-and-cite |
| **T2** | "**H-1-40295** (EKTAR 100) and **F-4017** (GOLD 200)" | ⚠ **F-4017 is TRI-X 320/400.** H-1-40295 exists nowhere and cannot be EKTAR 100's code — H-1 is Kodak's motion-picture series and EKTAR 100 is a still film (**E-4046**, five copies). GOLD 200 is **E-7022**, four copies | → stage-and-cite |
| **T3** | Fuji **AF3-0076E5**, **AF3-058E3** | both films held under different revisions — **AF3-036E** and **AF3-151E** — and **PRO 400H** (AF3-176E), which the row lists as absent | → half dissolved |
| **E4** | "ready now, source on disk" | ⚠ **the opposite error** — the 1942 book is on the owner's machine, not here | → stage first |
| **G2** | "trace the four raster plot sets" | same: the 1968 paper is not in this checkout | → stage first |
| **E5** | listed as *acquisition* in the tier-3 line | ⚠ held, and **misnamed**: `Sehlin_Kennel_etal_1983_…pdf` is **SMPTE Journal, July 1985, p. 724** — read off page 1 of a render, since all eleven pages are raster | → method, as its own category cell already said |

**Re-proved absent, against 475 files rather than 56:** **C14** (no Kodak EKTAR 125 publication —
only the 1989 *PHOTOgraphic* review), **F1** (no JOSA paper of any kind), **G6** (⚠ worse than it
looked: `agfa_films.pdf`, `AGFA stocks.pdf` and `FPD1e.pdf` are the same publication as
`Datasheet_F_PF_E4.pdf`, two of them byte-identical, and all four GEVAERT papers are 100 % raster),
**K5**, **K6**, **F2b**.

⚠ **K5's proof got STRONGER.** The row said "not obtainable from any of the thirteen". All **201**
KODAK files were searched for "diffuse rms granularity": **more than eighty** print it — every cine
stock, most EKTACHROMEs, and the 1996–1998 E-55/E-88 reversal editions that predate Print Grain
Index. **Not one is a PORTRA, GOLD or ULTRA MAX colour negative still film.** The eight stocks K5
names are exactly the population Kodak moved to PGI.

⚠ **A named lead for F2b, which is not a closure.** `ILFORD/AN ANALYSIS OF FILM GRANULARITY.pdf` is
**BBC Engineering Monograph No. 54, August 1964** — a different document from the BBC T-101 the row
discusses, and absent from this checkout. Its Fig. 8 is at a fixed density, so it yields no shape.
What it carries is a law: S/N ∝ D^−0.5 for negatives, "closer to −0·4" per **Higgins and Stultz**,
and **−0.6 to −0.7 for reversal** — the first independent support for the thing F2 refused to
assume, that negatives and reversals do not share a shape. The real lead is Higgins & Stultz.

**Count: 26 live → 24.** 102 rows, 78 struck.

---

## 2. B3 — one stock's first spectral set, one stock's first validation

**`KODAK_TECHNICAL_PAN`** had no spectral set at all and now has one: P-255 p9, vector paths, the
D=0.3-above-D-min curve of a printed pair, 31 samples, absolute peak log sensitivity 1.03.

⚠ **It is one of the two flattest panchromatic curves in the database** — 0.56 decades across
380–680 nm against a field median of 1.12 — which is the trace agreeing with P-255's own prose,
*"reasonably uniform spectral sensitivity at all visible wavelengths out to 690 nanometres"*.

⚠ **The first draft of that verify check said "flatter than every other panchromatic set" and the
check caught it.** `FUJI_NEOPAN_1600` spans 0.55. The claim is now "among the two flattest and well
under the field median", which is what the data supports.

⚠ **380 nm is the GRID EDGE, not the peak.** The printed panel runs to 250 nm and is still climbing
where the stored grid begins — consistent with the panel's own note, 1.4 s visible against 0.2 s
ultraviolet. Recorded at the profile so nobody reports a 380 nm peak as a property of the emulsion.

### `KODAK_TMAX_400` did not want what the row said it wanted

The row asked for "its second criterion". Its set was adopted 2026-08-16 from F-4043 (2016), and
`NotFound.md` had already recorded that as closed; the sheet's second criterion is a *different
measurement of the same emulsion* and the schema holds one set per stock. What it actually lacked,
and now has, is a **cross-edition validation**: the **2007** edition is a file the profile has never
read and reproduces the stored set to **rms 0.005 decades** — closer than the 2016 edition now does.

### Two reader assumptions, each one line

- The caption test was `"ABOVE" in txt and "=" in txt`. It fits H-1-5222 and matches **neither** new
  sheet: F-4043 prints "D=0.3 greater than D-min" (no "above"), and P-255 splits its caption across
  two text lines — the line with the number has no `=`, the line with the `=` has no number.
- The mono reader committed to the first frame that calibrated. F-4043 (2007) p11's first
  calibrating frame yields **one** curve where the caption pair needs two, so the sheet failed
  outright while the right frame sat next in the list. Same lesson `extract_sheet` learned on 7239.

### The second criterion is now measured, not merely named

The gap between a panel's two criterion curves is the log-exposure interval between two densities,
so its **sign must follow the criteria** and it must be **wavelength-independent**:

| sheet | adopted | other | gap (decades) | dD/dlogE |
|---|---|---|---|---|
| H-1-5222 | D=1.0 | D=0.3 | **−0.992 ± 0.064** | 0.71 |
| H-1-5231 | D=0.3 | D=1.0 | **+1.068 ± 0.088** | 0.66 |
| F-4043 | D=0.3 | D=1.0 | **+1.059 ± 0.078** | 0.66 |
| P-255 | D=0.3 | D=1.0 | **+0.408 ± 0.036** | **1.72** |

⚠ Swapping a pair — the exact C38 failure, which survived every band, ordering and peak check the
reader had — **flips the sign**. And the magnitudes are legible: Technical Pan is the only film here
whose sheet prints a contrast index above 2, and it is the only one whose criteria sit 0.4 decades
apart instead of 1.0. Its panel's printed CI is 2.00 for HC-110 (Dil D) 8 min; 1.72 is the
toe-region slope, which must be lower, and is pinned as an interval between two printed criteria
rather than as a reconstruction of CI.

⚠ **P-255's two editions are NOT an independent check** — they agree at rms **0.0000** because the
artwork is bit-identical. Recorded as a redraw guard, not as a cross-validation.

---

## 3. E3 — the first adoption from a corpus sheet that is raster end to end

`professional_160.pdf`, `IMP50.pdf` and `INF750.pdf` carry a text layer for the **prose** and
bitmaps for every **plot**: no paths, no tick text. `konica_raster.py` calibrates geometrically off
the printed grid, and all seven panels **re-detect their own gridlines** before a curve is traced.

⚠ **The bitmaps are stored upside down.** Rotating them 180° fixes the picture and leaves the text
mirror-reversed, which is how the flip announces itself; `FLIP_TOP_BOTTOM` is the right transform
and the grid assertions passing afterwards are the proof.

### `KONICA_IMPRESA_50` — a family template, caught by its own sheet

It held Dmin **0.20 / 0.62 / 1.00** with gamma 0.600 / 0.615 / 0.620. `KONICA_VX_100` holds
0.21 / 0.63 / 1.02 with 0.615 / 0.625 / 0.630. `KONICA_CENTURIA_SUPER_400` holds 0.22 / 0.65 / 1.05
with 0.62 / 0.63 / 0.635. **Three stocks, one shape, round numbers, all marked
`fitted_from='datasheet_curve'`.** Nothing in `verify.py` could catch that, because nothing compared
a stored triple to a reading.

Traced: Dmin **0.199 / 0.557 / 0.676**, softplus fits at rms 0.009–0.015 D, gamma
**0.568 / 0.688 / 0.820** — a 44 % layer spread against the template's 3 %, blue steepest, which is
the physical ordering for a masked negative.

⚠ **And a second figure on a second page corroborates the number that moved most.** p3's
**minimum-density spectrum**, sampled at the ISO 5-3 status M band centres 640 / 540 / 450 nm, reads
**0.190 / 0.552 / 0.691** — agreeing to **0.009 / 0.005 / 0.015 D** and jointly refuting the stored
blue of 1.00.

**MTF:** f50 **64.9 c/mm**, not the estimated 72; overshoot to **121.4 % at 6.88 c/mm**; power-law
rolloff **q 2.20** at rms 0.019 against the Gaussian's 0.039 (at 90 c/mm: drawn 35.2 %, power
32.7 %, Gaussian 26.3 %). f50 falls between the sheet's own printed resolving powers, 63 lines/mm at
1.6:1 and 160 at 1000:1 — the only independent statement it makes about its sharpness.

⚠ **The per-layer 72 / 80 / 88 had to go, and that loses something real.** The sheet prints ONE
curve captioned *"Densitometry: Through visual filter"*, so a pooled f50 is what was measured. The
layer ordering `MTFSpec`'s own docstring argues for is physically true and had no source here.
`verify.py` names the stock in `_VISUAL_FILTER_MEASURED`, excludes it from the two family guards
that reason about red records, and **asserts the pooling that licenses the exclusion** — the three
f50 fields must stay identical, and the value must sit between the printed resolving powers.

### `KONICA_INFRARED_750` — the stored gamma was below all fifteen printed curves

| developer | 4 min | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| Konicadol DP | 1.153 | **1.563** | 1.764 | 1.804 | 1.837 |
| Konicadol Super | 1.036 | 1.321 | 1.440 | 1.410 | 1.425 |
| Konicadol Fine | 0.814 | 1.087 | 1.244 | 1.418 | 1.546 |

The profile held **0.720** (mid-slope 0.707) with an **empty `ProcessingSpec`** — no developer, no
time, no temperature. The flattest curve on the page is 0.814.

Adopted: **Konicadol DP, 6 min, 20 °C** — the sheet's own standard time for the developer its
footnote equates to **KODAK D-76** — fitted to 167 samples at rms **0.0049 D**. Base+fog **0.234**,
shared by all fifteen curves and measured on the four traces that still resolve at log H −1.95
(0.2303 ± 0.0148), against 0.150 held before. The first non-empty `ProcessingSpec` this stock has
had, with the agitation printed in the sheet's own words.

### What the sheets cannot give, recorded rather than deferred

- **No dye triple.** p3 draws two **neutral** spectra — minimum and midscale — not three separated
  dye curves. It is the cross-check above and it can never be a `DyeDensity`.
- **No per-layer Konica MTF.** One visual-filter curve.
- **No absolute spectral level for INF750.** Its p1 panel is 1440×276 and carries **no y tick labels
  at all**; it can give the band shape the stored relative set already has, and never a criterion.
- **`professional_160.pdf` closed as unusable.** All four pages extract **zero characters**, so
  there is no caption, axis label or legend to calibrate against — and `NotFound.md` had already
  established that its one technical page matches no stock in the database.

⚠ **A third data point for C2c**, recorded and deliberately not acted on: IMPRESA 50's overshoot
peaks at 6.88 c/mm against a stored `adjacency_um` of 14.0. That is 5231 (4.7 vs 16.0), FUJI F-125
(~9 vs 13.0) and now IMPRESA 50 — three stocks, two manufacturers, one direction. The overshoot
**amplitude** was adopted (0.10 → 0.214); the **length** is C2c's decision, not E3's.

---

## 4. The pattern, again

Every closure this week has overturned its own recorded diagnosis, and this batch adds three more:
**B3** ("not in this checkout" — it was, twice over, and the T-MAX half was already closed in
`NotFound.md`), **E3** ("not openable from this working copy" — nineteen Konica sheets on the
owner's machine), and **T1 / T2 / T3**, which between them name four publication codes and get
**three of them wrong about which film they describe**.

The rule in §0.4 is now sharpened: **check the owner's corpus, and check the publication code
against page 1 of the file rather than the filename.**

## 5. Files

**New:** `konica_raster.py`, this document.
**Changed:** `spectral_vector.py` (`CRIT_RE`, `_mono_from_frame`, frame continuation, four new mono
registry entries, `SEP_EXPECTED`, `MONO_EDITIONS`), `film_profiles.py`
(`KODAK_TECHNICAL_PAN.spectral`, `KONICA_IMPRESA_50` curves and MTF, `KONICA_INFRARED_750` curve and
`ProcessingSpec`, two provenance sources, regenerated `spectral_weights` provenance), `verify.py`
(`_VISUAL_FILTER_MEASURED` and eight new checks), `build.py` (`konica_raster.py` registered, 27 →
28), `doc/NotFound.md`, `doc/DATASHEET_VERIFICATION_REPORT.md`, `doc/DIGITIZATION_QUEUE.md`, and the
regenerated C++ in both trees.
**Staged into the checkout from the owner's machine:** `KODAK/p255.pdf`, `p255-2003_06.pdf`,
`f4043_TMax_400-2016.pdf`, `f4043-TMAX_400-2007.pdf`,
`Sehlin_Kennel_etal_1983_ChoosingECN5247or52941.pdf`, and `KONICA/{professional_160,IMP50,INF750}.pdf`.
