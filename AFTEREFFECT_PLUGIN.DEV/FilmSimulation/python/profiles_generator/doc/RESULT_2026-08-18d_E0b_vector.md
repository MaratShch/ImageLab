# RESULT 2026-08-18d — E0b: three vector plot sets, and what fixing the extractor recovered

E0 (the re-verification of eleven sheets wrongly declared absent) ended by noting three plot sets
that were **vector** and unread. This is what came out of reading them — plus three extractor
defects found on the way, which turned out to matter more than any single sheet.

## 1. Three dye-density sets adopted, and none of them needed a better source

`dye_density.py` had five sheets on a FAILED list. **Three of the five now extract cleanly, and
the sources were never at fault** — all three failures were defects in the script:

| Defect | What it did | Fix |
|---|---|---|
| The y-axis caption anchor grouped rotated words by **x-centre only** | On H-1-5239 p3 that merged the spectral-sensitivity plot's "DENSITY" with the dye plot's "DIFFUSE SPECTRAL DENSITY" into one band spanning y 127–440, so the extractor measured **the wrong plot** | `LABEL_STACK_GAP` — split a column into captions on a vertical gap |
| Axis calibration used the **lowest and highest tick only** | Two points always define a line and nothing checked it. On the 5218 sheet it produced curve maxima of **2.14–2.24 D inside a frame whose axis stops at 1.8 D** — impossible, since curves are clipped to the frame | `_fit_axis` — least squares over every tick with outlier rejection |
| Stroke-width filter was `width < 0.6 × max(width in frame)` | The thickest path in the 5248 frame is a **rule, not a curve**, so four of six traces were discarded | reference the **median** width of long paths, and let physics reject non-curves downstream |

Adopted:

| Profile | Sheet | Family | Residual | Peaks (y/m/c) |
|---|---|---|---|---|
| `EASTMAN_EKTACHROME_7239` | H-1-5239 p3 | as-printed + neutral | 0.040 D | 440 / 550 / 670 nm |
| `KODAK_VISION2_200T_5217` | 5217-Vision2-200T p3 | peak 1.0 | 0.0097 | 450 / 540 / 680 nm |
| `KODAK_VISION2_500T_5218` | H-1-5218t p4 | peak 1.0 | 0.0195 | 450 / 540 / 680 nm |

All three sheets **label their own curves** Yellow / Magenta / Cyan, so the peak-based assignment
was checked against Kodak's words rather than trusted. On 5217 there is a second, independent
check: that page draws the dyes in colour, and the colour names the *absorbed band* rather than
the dye — the **blue** trace is the yellow dye, **green** is magenta, **pink** is cyan. Both
methods agree.

Film profiles with a dye set: **7 → 10**. B1's remainder: **5 sheets → 2** (5246, 5248), and
their measured near-misses are now in the docstring instead of a shrug.

### 1.1 The calibration change was adopted on independent evidence, not on preference

Switching to a fitted axis moved every residual, so the question was which method is right. The
answer came from the two sheets whose arrays had been adopted by a **separate earlier pass**:

| Sheet | New fit vs stored arrays | Old fit vs stored arrays |
|---|---|---|
| 2383 | **RMS 0.0005 D** | RMS 0.0185 D |
| 5285 | **RMS 0.0003 D** | RMS 0.0029 D |

The new method reproduces numbers it did not derive 10–40× more closely. On the peak-normalised
family the independent test (|max − 1.0|, which must be 0 because those sheets normalise to unit
peak) is a **wash** — new better on three sheets, old better on three, all within 0.003 — so
**those stored arrays were not re-adopted.** Re-adopting on a wash would be churn dressed as
progress. The two calibrations agreeing to ≤ 0.015 D is itself a useful error bar, now recorded.

### 1.2 A sheet whose own note contradicts its own plot

H-1-5285 p4 states *"Note: Cyan, Magenta, and Yellow Dye Curves are peak-normalized."* The
plotted maxima are **0.921 / 0.895 / 0.907** — confirmed against a 200 dpi render, where the peaks
sit visibly below the printed 1.0 gridline. The curves are as-printed; the caption is boilerplate.

That matters beyond one sheet, because **the same sentence is true on the VISION2/VISION3
sheets.** So the phrase cannot decide a sheet's normalisation — only the measured maxima can.

## 2. The first measured σ(D) for a colour reversal stock — and the heuristic has the sign backwards

`KODAK_EKTACHROME_100D_5285`, from H-1-5285 p4 (plot F002_1047AC), green record:

| Anchor | Density | rms (= 1000·σ) | ratio to mid |
|---|---|---|---|
| toe | 0.141 | 1.9 | **0.15** |
| mid | 1.000 | 13.1 | 1.00 |
| dmax | 3.514 | 40.6 | **3.10** |

**σ rises by a factor of ~20 from dmin to dmax.** `_grain_v2`'s reversal heuristic fills
`0.7 / 1.0 / 0.5` — i.e. it assumes σ *falls*, reasoning that "the densest regions of a slide
received the least exposure". The premise is true; the conclusion does not follow, and the sheet
settles it.

**The heuristic was not flipped.** It fills 21 reversal stocks and this is one sheet — flipping
all of them on a single measurement is the "plausible therefore adopted" move this project
forbids. That is the same reasoning already recorded in `_grain_v2` for the colour-negative
branch, which is wrong in the *other* direction. **Both branches now have a measured
counter-example and neither has been changed.** That is a decision for C1, and it is now much
better evidenced than it was this morning.

Getting there needed three geometry facts, each of which broke a naive read:

* **All three granularity curves live in ONE drawing object** — 105 segments across three
  disjoint subpaths. Treating it as one curve produces a trace that teleports between layers and
  still looks plausible plotted.
* **Kodak draws each layer only where it is distinguishable**, so the six curves have six
  different left-hand starts (R from log E −3.0, G from −2.5, B from −2.1). That the granularity
  subpaths' x-extents match the characteristic subpaths' pairwise is an *independent* confirmation
  of the R/G/B label assignment.
* **The two curves of a pair don't start at exactly the same x** — G's characteristic reaches dmax
  at −2.54 while G's granularity curve begins at −2.50, which returned NaN until the anchors were
  clamped to the overlap and the clamped density printed.

One σ-axis tick is genuinely misplaced: fitting all nine labels leaves eight under 1.1 pt and
**".010" at −1.94 pt**. The give-away is the decade spacing — at face value the labels put
.001→.010 at 42.5 px and .010→.100 at 39.2 px, which cannot both be a decade. Dropping ".010"
gives 40.6 and 41.1. Trusting every label would have carried a 5 % σ error into an adopted number.

### 2.1 The level moved 4.4×, and that is the visible change

`5285` stored `rms_granularity = 3.0` with **no comment and no source**. E0 had already established
that this sheet prints no rms number, only curves plus "multiply by 1000 for the rms value". Doing
exactly that on the green record at D = 1.0 gives **13.1**.

Two cross-checks say 13 is right and 3 was not:

* **.003 is where the curve ends up at the far right of the plot — at dmin.** That is the likely
  origin of the old figure: the toe of the granularity curve read as if it were the mid anchor.
* The sibling reversal stocks, whose figures *are* printed on their sheets, are **7239 at 14.0**
  and **TRI-X Reversal at 10.0**. A 100-speed E-6 reversal at 3.0 would have been finer than
  VISION3 50D (2.6), which is not credible.

Per-layer values are measured too — **19.0 / 13.1 / 25.7** for R/G/B, i.e. green finest and blue
coarsest. The heuristic would have produced 14.4 / 13.1 / 17.0: right order, far too flat.

## 3. PLUS-X 5231's MTF, measured instead of estimated

From H-1-5231 p3 (plot F002_0141AC), a single bezier path on a page with zero embedded images.
Both axes logarithmic, fitted over 11 frequency and 12 response labels to 0.66 and 0.82 pt.

| | Was | Now | Evidence |
|---|---|---|---|
| `f50` | 60.0 | **41.3 cycles/mm** | curve falls back through 50 % at 41.3; spans 2.4–98.2 c/mm |
| `adjacency` | 0.08 | **0.034** | peak response 103.4 %, i.e. a 3.4 % overshoot |

The stored f50 overstated this stock's sharpness by **45 %**, and the stored adjacency claimed an
edge effect **2.4× stronger** than Kodak prints. One f50 for all three channels, because a
panchromatic B&W negative has no layer stack to soften red. The 50 % crossing is taken at the
**last** one: the curve rises above 100 % at low frequency, so it is not monotone and a
first-crossing search can return the wrong branch.

**`adjacency_um` was flagged, not changed.** The measured overshoot peaks at 4.7 cycles/mm — a
scale of order 100–200 µm — while the stored 16.0 µm corresponds to roughly 60 cycles/mm. The same
contradiction appeared the same day on `FUJI_F125_8530` (peak near 9 cycles/mm, stored 13.0 µm), so
it is **systematic, not a typo in one profile**. Resolving it needs the renderer's definition of
the field. Guessing would replace one unexamined number with another.

## 4. State

`225 PASS / 2 FAIL` (both baseline), compile clean on 18 TUs, `film_names.txt` MD5 unchanged.

Two stale count assertions failed on the way and were updated deliberately, not absorbed —
"7 film profiles carry spectral dye density" → 10, and "6 new dye sets are peak_1.0" → 8 of 10.
A count assertion is *meant* to fail when the count changes.

**New audit scripts, both registered in `build.py`:** `granularity_vector.py` and `mtf_vector.py`.
Every number above is re-derivable by running them, per method rule 13.

## 5. What this run set up

* **C1 is now the best-evidenced open decision in the queue.** Both halves of `_grain_v2`'s σ(D)
  heuristic have a measured counter-example, in opposite directions, and `sigma_shape_*` is still
  read by no renderer.
* **7239's dye set proves the extractor was the bottleneck, not the corpus.** Two sheets remain on
  the failed list; the same class of fix may reach them.
* `KODAK_EKTACHROME_100D_5285` p3 also carries a **vector characteristic curve** family that has
  never been traced — its stored gamma of 11.6–15.4 is a softplus artefact, per its own comment.
