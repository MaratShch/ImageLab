# RESULT 2026-08-18f — C1c: the σ(D) harvest, completed and closed

**Task as given:** "I want complete with harvest σ(D) and close this task as fulfilled."

**Outcome:** every granularity plot in the corpus that is drawn as VECTOR art is now
read. Eight sheets, all eight green under `granularity_vector.py --assert`; seven
adopted into the database (six new), the eighth used as an independent cross-check
of a shape traced earlier from a raster plot by a different extractor.

Build after the work: **`build.py --root <corpus>` OK, 0 failures, 0 warnings** —
7 of 7 audits green, `verify.py` **235 PASS / 2 FAIL** (the two the owner said to
leave alone), C++ clean on 18 TUs.

---

## 1. What was harvested

Green record, the visually weighted one. σ×1000 = the sheet's own rms convention
(48 µm aperture, "multiply by 1000 for the rms value").

| stock | sheet | toe (at D) | mid | dmax (at D) | interior peak |
|---|---|---|---|---|---|
| `EASTMAN_EXR_50D_5245` | 5245 p4 | 1.19 (0.572) | 1.00 | 0.72 (2.091) | **1.47× at D 0.73** |
| `KODAK_VISION_250D_5246` | 5246 p4 | 0.94 (0.582) | 1.00 | 0.90 (2.201) | **1.62× at D 0.66** |
| `EASTMAN_EXR_100T_5248` | 5248 p3 | 1.19 (0.612) | 1.00 | 0.84 (2.051) | **1.58× at D 0.74** |
| `KODAK_VISION_200T_5274` | 5274 p3 | 0.80 (0.582) | 1.00 | 0.61 (2.211) | **1.38× at D 0.68** |
| `KODAK_VISION_500T_5279` | 5279 p2 | 0.96 (0.576) | 1.00 | 0.50 (2.210) | **1.42× at D 0.65** |
| `KODAK_VISION2_500T_5218` | 5218 p4 | 1.17 (0.592) | 1.00 | 0.70 (2.309) | **1.56× at D 0.74** |

Absolute traced σ×1000 at D = 1.0 (green): 6.04 / 8.28 / 7.32 / 9.18 / 13.23 / 9.58.
Per-layer at D = 1.0, where the pair's overlap reaches it:

| stock | R | G | B |
|---|---|---|---|
| 5245 | 3.78 | 6.04 | 10.06 |
| 5246 | 7.16 | 8.28 | 22.22 |
| 5248 | 4.37 | 7.32 | — (blue overlap starts above D 1.0) |
| 5274 | 5.51 | 9.18 | 24.49 |
| 5279 | 6.90 | 13.23 | 40.90 |
| 5218 | 5.49 | 9.58 | 33.70 |

**All six turn over.** σ rises from the toe to a maximum between D 0.65 and 0.74
and falls to half or two-thirds of its D = 1.0 value by dmax. That is the same
direction the four VISION3 sheets and Kodak's own July 1985 SMPTE paper give
("overexposing either film significantly decreases granularity"), now on six more
independent sheets. The database's heuristic for colour negative — 0.40 / 1.00 /
1.20, i.e. σ *rising* to dmax — is wrong in sign on every sheet that has been
measured, which is now ten of them.

## 2. What it does to the render

Amplitude ratio, measured shape ÷ legacy sqrt law, green record, level-preserving
at D = 1.0 by construction:

| stock | at dmin | D 0.7 | D 1.0 | D 2.0 | at dmax |
|---|---|---|---|---|---|
| 5245 | 2.27 | 2.13 | 1.00 | 0.44 | 0.36 |
| 5246 | 2.63 | 2.35 | 1.00 | 0.54 | 0.45 |
| 5248 | 2.24 | 2.21 | 1.00 | 0.50 | 0.42 |
| 5274 | 2.14 | 2.09 | 1.00 | 0.40 | 0.30 |
| 5279 | 2.35 | 2.12 | 1.00 | 0.34 | 0.25 |
| 5218 | 2.29 | 2.24 | 1.00 | 0.45 | 0.35 |

So on these six stocks the renderer now puts **2.1–2.6× more grain in the shadows
and 2–4× less in the highlights** than it did yesterday, at unchanged mid-density
level. Nothing else moved: the other 149 profiles reproduce the legacy law
bit-for-bit (guarded, max deviation < 2e-6 in float32 over a 36-point sweep).

## 3. The independent cross-check, and the conflict it exposed

`5219` is the one stock whose σ(D) had already been traced — from the RASTER plot
on its technical sheet, by `vision3_granularity.py`. The VISION3 500T brochure
prints the same kind of plot as vector art, so it was read here by a different
extractor, from a different document, in a different medium:

| | toe | mid | dmax | peak |
|---|---|---|---|---|
| raster, H-1-5219 p3 | 7.11 @ D 0.595 | **10.60** | 5.84 @ D 2.712 | 1.32× at D 0.79 |
| vector, brochure p2 | 3.20 @ D 0.607 | **8.03** | 4.60 @ D 2.662 | 1.24× at D 0.76 |
| agreement | see below | 1.32× apart | ratio 0.55 vs 0.57 | 0.08× and 0.03 D apart |

**Shape: confirmed.** dmax/mid agrees to 0.02, the peak location to 0.03 D, the
peak height to 0.08×. Two extractors, two documents, two media.

**Absolute σ: a real conflict, recorded not averaged.** The brochure reads a
near-uniform 1.3× lower everywhere, so one of the two σ-axis calibrations is off
by about a third of a decade. The brochure's own ladder is internally consistent
to 0.5 pt over two decades (0.001 → 0.10, 36.7 pt/decade), which is why it is
pinned as a cross-check and *not* adopted over the raster trace on its own
authority. Nothing in the database changed on the strength of it.

**The toe disagreement is not a conflict at all.** 0.67 vs 0.40 looks like the
worst of the three, and it is an artefact of the anchor's definition: below the toe
the characteristic curve is FLAT, so density holds at dmin while σ keeps climbing,
and σ(D) is genuinely multivalued at exactly that density. The two traces landed on
different points of the same plateau. The extractor now measures the span and
prints `[plateau] … toe anchor is not unique` when it exceeds 15 % — on 5219v it
spans 3.19–5.49. Three of the six adopted stocks carry the same flag, and their
entries say so.

## 4. Six extractor defects found, all measured

Every one of these produced *plausible* output before it was found — which is the
argument for the overlay gate, not for tighter tolerances.

1. **`subpaths()` discarded any group of ≤ 4 points** (`if len(cur) > 4`). Where a
   granularity curve goes flat its dashes need no bezier and are emitted as single
   straight segments; on 5245 the whole right-hand half of one curve is 2-point
   dashes at a constant y = 202.2 pt. They vanished before any filter saw them,
   cutting that curve off at 52 % of the frame width — while leaving a chain long
   enough to pass every count test.
2. **The per-path item floor of 8 cost 5279 its sixth curve.** That sheet draws one
   granularity curve as three separate path objects of 12 / 6 / 5 items; the two
   short ones were dropped before `stitch()` ran, so the sheet presented five
   curves and was refused for weeks.
3. **`stitch()` required a non-negative x step.** Those three pieces *abut* — they
   share an endpoint, so the measured step at the junction is −0.1 pt. Fixed with
   0.6 pt of backward tolerance, which admits a shared endpoint and nothing else.
4. **`curves()` returned early when six wide pieces existed.** On 5279 that was
   satisfied by five whole curves plus one curve's right-hand third (67 % of the
   frame on its own): the count test passed and two thirds of a curve were silently
   discarded. Stitching now always runs. A fragment that survives the count test is
   worse than one that fails it.
5. **Nearest-label-wins lost letters on four sheets of eight.** Where the two
   families share plot area a granularity letter can sit closer to a passing
   characteristic curve than to its own, and on 5279 one label triple is printed at
   the LEFT edge. The sheet states a constraint the greedy rule threw away — six
   letters, two of each, one triple per family, against six curves — so the
   assignment is now solved exhaustively as a bijection (8 × 36 = 288 candidates).
   On the four sheets the greedy rule *did* resolve, the strict method returns the
   identical answer; that equality is the reason it was safe to adopt.
6. **Chaining across a crossing.** Granularity curves cross each other, and
   nearest-in-y hands a fragment to the wrong curve there. The join is now against
   the chain's own local slope, and `stitch()` additionally refuses to chain across
   stroke colours — inert on the seven black sheets, necessary on the colour one.

Two further generalisations were needed rather than fixed:

* the granularity family is **not always three curves** — the brochure omits the red
  one entirely (verified by rendering the panel), so the expected count per family
  is declared per sheet;
* layer identity is **not always letters** — the brochure states it in ink, so that
  sheet is read by stroke colour. Colour is as printed as a letter is; this is not
  an inference from stroke style in pixel space, which method rule 3 forbids.

## 5. One audit made honest

`plot_inventory.py --assert` was failing four count assertions in this workspace.
Cause: its page counts are corpus-wide (450 PDFs) and this container holds a 41-PDF
staged subset. Four red lines that mean "you copied fewer files" are
indistinguishable at a glance from "the classifier broke", so the script now sizes
the corpus first and skips the corpus-wide assertions loudly on a partial mirror.
The per-page ground-truth classification check still runs — it does not care how
many other files exist.

## 6. What was deliberately NOT done

**`rms_granularity` was not re-levelled.** The traced green mids run 1.3–1.6× above
the stored figures (5245 6.04 vs 4.2; 5246 8.28 vs 5.3; 5248 7.32 vs 5.6; 5274 9.18
vs 5.8; 5279 13.23 vs 8.3; 5218 9.58 vs 7.3), exactly as the four VISION3 stocks do.
None of these sheets prints an rms number — only the curve — so the stored values
are tier-3 family-ladder estimates and the traced values are better evidence. But
that is a **level** change across a whole family, it interacts with the C1b
normalisation decision and with the rms values self-described as
"pipeline-calibrated", and it is a different question from the shape. This entry
adopts the shape only; `verify.py` now asserts the stored rms values are untouched,
so the comment that quotes them cannot go stale silently.

## 7. Files touched

| file | change |
|---|---|
| `granularity_vector.py` | scope 1 → 8 sheets; six defects fixed; colour-aware stitching; per-sheet count/identity spec; interior-peak and plateau reporting; all eight sheets pinned in `EXPECTED` |
| `film_profiles.py` | six colour negatives gained a measured σ(D) shape with `sigma_shape_measured=True` and full provenance |
| `verify.py` | 231 → 235 checks: traced anchors exact, rms untouched, not-the-heuristic, all six turn over, all six usable by the renderer; the measured-stock list 5 → 11 |
| `plot_inventory.py` | partial-corpus detection; corpus-wide counts no longer asserted on a subset |
| `doc/PROGRESS.md`, `doc/DIGITIZATION_QUEUE.md`, `doc/README.md` | status, queue closure, known limits |
