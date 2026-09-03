# Q1 — VISION3 granularity σ(D): extraction succeeded, four sheets now agree

> ⚠ **EXTENDED ON 2026-08-23 (C1e): these sheets also yield PER-LAYER rms, which this pass
> did not read.** The σ(D) shapes below are unchanged and still reproduce. What was added
> later is the per-record level: reading each record at its own NET 1.0 gives 5219
> 6.43/7.16/19.37 (ratios 0.897/1.000/2.703), 5207 b/g 2.123 and 5203 b/g 1.813 — adopted as
> RATIOS onto the frozen pooled rms. 5213 is the one member that cannot be done this way:
> its three granularity curves are printed as a single bold band. ⚠ And the σ-axis suspicion
> that this pass's numbers later attracted was DISPROVED by the panel's own tick comb (15
> ticks within 1 %). See `RESULT_2026-08-23_c1e_c8.md`.

**Date 2026-08-17. APPROVED AND APPLIED.** Owner approved after review of the
overlays; §8 below records what was changed and the verification that followed.
Source: Eastman Kodak Technical Information sheets, **page 3, "Diffuse rms Granularity
Curves"**, one per stock:

| Stock | File | Plot image (native raster) |
|---|---|---|
| VISION3 50D 5203 | `KODAK-VISION3-50D-5203-7203-technical-information.pdf` | 587 × 562 px |
| VISION3 250D 5207 | `KODAK-VISION3-250D-5207-7207-technical-information.pdf` | 587 × 557 px |
| VISION3 200T 5213 | `KODAK-VISION3-200T-5213-7213-technical-information.pdf` | 587 × 538 px |
| VISION3 500T 5219 | `VISION3_5219_7219_Technical-data.pdf` | 587 × 574 px |

Method rule 1 applied: `get_drawings()` returns **2 paths, 0 with ≥30 items** on all four
pages — these plots are **raster**, confirming the §3 correction. Worked at the embedded
image's native resolution (extracted by xref) rather than re-rendering the page, so the
pixel grid is the publisher's own and no interpolation is introduced.

---

## 1. Root cause: confirmed by overlay, and the direction matters

The overlay was run **first**, as recommended at the end of batch 14. It reproduced the §3
diagnosis exactly and added one fact the write-up did not have.

Painting a **right-seeded, leftward** trace onto 5203 shows all three density tracks
derailing at log E ≈ 1.0–1.6: `Db` leaves the thin B density curve and continues down the
**bold granularity** curve, `Dg` ends on the R density curve, `Dr` ends on a granularity
curve. Same class of failure on 5213.

**The added fact — the crossing is tangential, and only in one direction.** At log E ≈ 1.1
on 5203 the B density curve is rising through the B granularity curve *at that granularity
curve's maximum*. Traced **leftward**, both branches descend from the junction with similar
slope: no slope test and no proximity test can choose, which is why three attempts failed
there. Traced **rightward** from the left plateau, the density branch is rising while the
granularity branch is at slope ≈ 0: the two separate cleanly. So the §3 prescription
("seed at the LEFT EDGE and trace RIGHTWARD only") is not merely convenient — it is the
only direction in which the junction is decidable. Confirmed on all four sheets.

## 2. Where the §3 plan needed changing

**Seeding the density curves by identity at the left edge is not sufficient on its own.**
Two of the four sheets do not present six separated runs at the left edge, because the
granularity curves lie *on top of* the density curves there:

| Sheet | runs at left edge (x = L+12), density units | verdict |
|---|---|---|
| 5203 | 0.906/t2, 0.615/t2, 0.519/t4, 0.282/**t8**, 0.177/t3 | **5 runs for 6 curves** |
| 5207 | 1.029/t1, 1.013/t2, 0.720/t2, 0.667/t2, 0.593/t2, 0.227/t2 | 6 runs, B pair 0.016 D apart |
| 5213 | 0.907/t2, 0.617/t2, 0.469/**t9**, 0.200/t3 | **4 runs for 6 curves** |
| 5219 | 1.353/t3, 0.901/t3, 0.824/t2, 0.690/t2, 0.598/t3, 0.212/t2 | 6 runs |

(`tN` = vertical run thickness in px. A t8 or t9 run is two or three curves merged.)

**What does work: separate the two families by DRAWING STYLE before tracing anything.**
The sheets themselves distinguish the families by style, not by position, and 5219 prints a
legend that says so in words — *Blue Density / Green Density / Red Density / Blue Grain /
Green Grain / Red Grain*. Two style tests, one per sheet pair:

* **5207, 5219 — dashed/dotted vs solid.** Connected-component labelling of the ink gives
  exactly **three components wider than 35 px** (438/438/424 px on 5207, 442/442/442 on
  5219) — the three solid density curves — and everything else ≤ 11 px. Keeping only the
  narrow components isolates the granularity family completely, with no density ink left.
* **5203, 5213 — bold vs thin.** Granularity strokes run 4–9 px thick, density strokes 1–3.
  Keeping only vertical runs ≥ 4 px isolates the granularity family; its complement isolates
  the density family.

With the families physically separated, **cross-family capture is structurally impossible
rather than checked afterwards** — which is the stronger form of the fix §3 asked for. The
`check_cross_family` assertion is still added (below), as a regression guard.

**Third change: slope-predictive stepping.** Each track predicts its next position from a
linear fit over its last 10–16 accepted points, extrapolated from the last *real* point
(not from the previous prediction — that double-counts and was the reason early runs died
mid-plot). Nearest-neighbour stepping cannot survive the crossings; this can, in the
decidable direction.

## 3. Calibration — and a correction to `dashtrace.py`'s docstring

Density axis: frame top = 3.0, frame bottom = 0.0. Validated against the left-hand minor
tick comb at 0.2 D: 29.6 px per 0.2 D on 5207 → **147.9 px/D**, against 443/3 = 147.67 from
the frame. Agreement 0.16 %.

Granularity axis: the tick comb outside the right frame line resolves as a clean
1,2,3…9,10,20…100 log ladder with the frame bottom at σ = 0.001.

| Sheet | px per decade, lower decade | upper decade | adopted |
|---|---|---|---|
| 5203 | 139.0 | 139.0 | 139.00 |
| 5207 | 140.0 | 139.5 | 139.75 |
| 5213 | 139.0 (0.001→0.01) | — | 139.00 |
| 5219 | 141.0 | 140.0 | 140.25 |

Within-decade tick positions reproduce log₁₀ to ≤ 0.5 px.

⚠ **`dashtrace.py`'s docstring is half wrong and should be corrected.** It says
"`sigma_shape_*` are RATIOS normalised at D = 1.0, so a multiplicative error in that log
calibration cancels exactly. Absolute sigma accuracy is therefore not the blocker."
With σ = C·10^((y₀−y)/P), a ratio is 10^((y₂−y₁)/P): an error in **C or y₀ cancels**, an
error in **P does not**. P had to be measured, and was. (The sensitivity is mild — ±1 % on
P moves the dmax/mid ratio by 0.5 % — but the claim as written would licence not measuring
it at all.)

## 4. Density-track identity — the check §3 asked for, and it passes

Each granularity plot's own left-edge density plateau against the [T1] H&D dmin ladder
already in the profiles. All twelve agree, and every offset has the same sign:

| Stock | R plot / profile | G plot / profile | B plot / profile |
|---|---|---|---|
| 5203 | 0.176 / 0.1341 (+0.042) | 0.614 / 0.5688 (+0.045) | 0.906 / 0.8434 (+0.063) |
| 5207 | 0.226 / 0.1539 (+0.072) | 0.593 / 0.5708 (+0.022) | 1.002 / 0.8392 (+0.163) |
| 5213 | 0.199 / 0.1681 (+0.031) | 0.614 / 0.5813 (+0.033) | 0.907 / 0.8510 (+0.056) |
| 5219 | 0.206 / 0.1867 (+0.019) | 0.595 / 0.5811 (+0.014) | 0.891 / 0.8374 (+0.054) |

Mean offset **+0.051 D, all positive** — the granularity plot reads slightly higher than
the sensitometric plot on every layer of every sheet. That is exactly what the sheets' own
footnote predicts: *"Sensitometric and Diffuse RMS Granularity curves are produced on
different equipment. A slight variation in curve shape may be noticed."* A systematic
offset is a signature of a calibration difference; a random one would have been a signature
of mis-traced identity. This is evidence for the trace, not against it.

## 5. Result — green record, σ×1000 (multiply by 1000 = the sheet's own rms convention)

5213's three granularity curves are drawn as one overlapping bold band across the whole
plot, so its row is the **band centre, pooled over the three layers**, not green alone.

| Stock | dmin | dmax* | σ(dmin) | σ(0.8) | σ(1.0) | σ(1.5) | σ(2.0) | σ(2.5) | σ(dmax*) | **toe/mid** | **dmax/mid** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5203 | 0.614 | 2.630 | 1.97 | 6.50 | 5.01 | 4.47 | 4.67 | 3.64 | 3.16 | **0.393** | **0.631** |
| 5207 | 0.593 | 2.695 | 4.95 | 10.70 | 8.35 | 6.22 | 7.10 | 5.78 | 4.71 | **0.593** | **0.565** |
| 5213 † | 0.614 | 2.663 | 3.04 | 9.21 | 7.48 | 5.42 | 5.74 | 4.87 | 4.37 | **0.406** | **0.584** |
| 5219 | 0.595 | 2.712 | 7.11 | 13.89 | 10.60 | 7.42 | 8.32 | 7.01 | 5.84 | **0.671** | **0.551** |

\* dmax = the highest density covered by **both** the density and the granularity trace on
that sheet, not an assumed film dmax. † pooled band.

**The four siblings now agree**, which is what batch 14 could not achieve:

* `dmax/mid` = 0.551, 0.565, 0.584, 0.631 — spread ±7 % about 0.583, four independent sheets.
* `toe/mid` = 0.393, 0.406, 0.593, 0.671 — same sign and order of magnitude, spread ±27 %.
* σ peaks at **D ≈ 0.77–0.80** on all four, at **1.24–1.32 ×** the D = 1.0 value.

Compare batch 14's contradictory 1.00/1.00/**2.56**, 0.65/1.00/**0.70**, 0.83/1.00/**0.67**.

**Robustness.** Re-running at ink threshold 0.40 and 0.50 changes every ratio by ≤ 0.002.
At 0.60 the 5207 trace derails (toe/mid → 1.000): the operating window is
**dark ∈ [0.40, 0.50]** and 0.50 sits inside a flat plateau, not on an edge.

## 6. The physics premise used to reject batch 14 was itself wrong

Batch 14 rejected its own numbers on the ground that "for a colour negative sigma_D should
RISE with density, since more developed silver means more grain". The measurement says the
opposite, and so does an official Kodak source, which under method rule 14 outranks an
a-priori expectation:

> **R. G. Sehlin, G. L. Kennel et al., "Choosing between ECN 5247 and 5294",
> _SMPTE Journal_, July 1985, p. 728** (in the archive as
> `Sehlin_Kennel_etal_1983_ChoosingECN5247or52941.pdf`, p. 5).
>
> * **Fig. 8**, "Granularity related to picture placement — 100:1 brightness ratio scene,
>   normal exposure": the 5294 density curve rises across relative log exposure while the
>   RMS Granularity curve on the same frame **falls monotonically**, from ≈ 2.5 to ≈ 1.0,
>   crossing it. Same two-family, two-axis layout as the VISION3 sheets.
> * **Fig. 9**, "Granularity vs. under- and overexposures": five granularity curves, **all
>   monotonically decreasing** with relative log exposure.
> * Body text: *"It is clear that overexposing either 5294 or 5247 film significantly
>   **decreases** granularity and that underexposing either film significantly **increases**
>   granularity."*

Two independent official Kodak sources, twenty-four years apart, agree that colour-negative
rms granularity **falls** as exposure and density rise. Classification:

* **Verified fact** — the σ(D) values in §5, and the falling trend at high density.
* **Engineering conclusion** — the green record is the right single curve to carry into a
  one-triple-per-stock schema field, being the middle of the stack and the layer with the
  best-conditioned dmin→dmax span on all four sheets.
* **Refuted assumption** — "σ rises with density for a colour negative". This is the
  premise behind the docstring's "Negatives are monotone (~0.4 / 1.0 / 1.2)" and behind
  batch 14's rejection. It should be corrected in both places rather than left standing.
* **Not established** — the physical mechanism for the fall. Dye-cloud overlap at high
  density is the usual explanation; no source in the corpus states it, so it stays out.

## 7. Limitations that must travel with the numbers

1. **The three-anchor model cannot represent the measured shape.** σ peaks at D ≈ 0.78,
   *below* the D = 1.0 mid anchor, at 1.24–1.32 × mid. Piecewise interpolation through
   (dmin, 1.0, dmax) rises then falls and **understates the maximum by ~25–30 %**. The
   triples below are the best three-point projection of a curve with an interior peak, not
   a faithful description of it. A fourth anchor, or storing σ(D) as a short array, would
   fix it — a schema question, not a tracing one.
2. **5213 is pooled, not per-layer.** Its three granularity curves overlap along the whole
   plot; the band half-width is ±4 px ⇒ ±7 % in σ. Per-layer separation is not available
   from this sheet at this resolution and should not be attempted.
3. **5203's toe anchor comes from a merged run.** At the left edge its G and R granularity
   curves share one 8-px run; ±4 px ⇒ ±7 %, plus an unresolved G/R assignment. This is the
   main reason the `toe/mid` column is looser than `dmax/mid`.
4. **Blue and red records are incomplete or ill-conditioned.** Blue dmin ≈ 0.9, so D = 1.0
   sits 0.1 D above dmin inside the steep rise and the blue toe/mid ratio is meaningless.
   The red granularity trace covers only log E ≥ 2.3 (5203) and ≥ 2.7 (5207).
5. **No printed summary statistic exists on these four sheets** (method rule 7 cannot be
   satisfied here — grep for a numeric rms returns nothing on all four). At D = dmin + 1.0
   the traced green σ×1000 is 4.55 / 6.32 / 5.60 / 7.16 against the profiles' stored
   `rms_granularity` of 2.6 / 4.2 / 4.6 / 6.6 — ratios 1.75 / 1.51 / 1.22 / 1.09. The
   agreement improves monotonically with grain size, consistent with an additive floor, but
   **the stored rms figures are not moved by this work** and the discrepancy is recorded,
   not averaged. Only the *shape* is being adopted; ratios are immune to any multiplicative
   error in the σ axis (§3).
6. **5203's green density trace has a 0.6-decade gap** (log E 1.01–1.64) where the bold
   granularity curve overdraws it. Nothing was interpolated across it. It is not needed:
   the toe anchor sits left of it and D = 1.0 is reached at log E ≈ 1.70, right of it. The
   two branches are consistent — 0.315 D over 0.63 decades = 0.50 D/decade, matching the
   film's own gamma of 0.579.

---

## 8. What was changed, and the verification

### 8a. `film_profiles.py` — four `GrainSpec` triples, with provenance

```
5203  sigma_shape_toe=0.39, sigma_shape_mid=1.00, sigma_shape_dmax=0.63
5207  sigma_shape_toe=0.59, sigma_shape_mid=1.00, sigma_shape_dmax=0.57
5213  sigma_shape_toe=0.41, sigma_shape_mid=1.00, sigma_shape_dmax=0.58
5219  sigma_shape_toe=0.67, sigma_shape_mid=1.00, sigma_shape_dmax=0.55
```

Each carries a comment naming the sheet and page, the raster/native-resolution method, the
measured px/D and px/decade, the green-record basis (pooled for 5213), the interior peak the
three anchors cannot represent, and the per-stock uncertainty. `rms_granularity` untouched.

### 8b. `film_profiles.py` — `GrainSpec` docstring corrected

The claim "negatives are monotone (~0.4 / 1.0 / 1.2)" is replaced with the measured result
and the SMPTE 1985 citation, and the statement that a turning-over triple is **not** a
reversal signature — what distinguishes reversal is the reason, not the direction.

### 8c. `film_profiles.py` — `_grain_v2` heuristic flagged, NOT changed

The heuristic still fills 0.4/1.0/**1.2** for 103 non-reversal stocks and its sign is now
known to be wrong for the colour negatives among them. Left in place deliberately, with a
comment giving both reasons: the approval covered four stocks, and every source is a
*chromogenic* negative while the same branch also fills **B&W silver** negatives, where
σ ∝ √D is the textbook result and nothing in the corpus contradicts it. Queued as the
follow-on task with the blocker named — a measured σ(D) for a B&W silver negative.

### 8d. `dashtrace.py` — three additions, and a correction

`family_split_by_style` (the structural fix: separate the families by dash/bold before
tracing), `trace_predictive` (slope-predictive, one direction only), `check_cross_family`
(the assertion `check_ordering` structurally cannot make), `column_runs_weighted`.
`trace_tracks` and `check_ordering` are unchanged. The docstring records the direction rule,
the measured px/decade figures, and the correction from §3 above.

### 8e. `vision3_granularity.py` — new, the re-runnable audit trail

Reads the four PDFs, drives `dashtrace`'s public API, writes the overlays, and **fails
loudly** if it stops reproducing the adopted triples. Current output:

```
sheet | dmin  dmax  | s_toe  s_mid  s_dmax | toe/mid dmax/mid | peak s @ D
5203  | 0.607 2.630 |   1.94   5.01   3.16 |   0.387   0.631  |  6.61 @ 0.80 (1.32x)
5207  | 0.593 2.695 |   4.95   8.35   4.71 |   0.593   0.565  | 10.81 @ 0.78 (1.30x)
5213  | 0.614 2.663 |   3.04   7.48   4.37 |   0.406   0.584  |  9.28 @ 0.77 (1.24x)
5219  | 0.595 2.712 |   7.11  10.60   5.84 |   0.671   0.551  | 13.95 @ 0.79 (1.32x)
Reproduces the adopted triples.
```

### 8f. `verify.py` — five regression checks

Triples populated and not the default; σ falls from mid to dmax on all four; the four dmax
anchors agree inside a 0.50–0.70 band with spread ≤ 0.12; toe anchors below mid in
0.35–0.75; and the rms grain ladder (2.6 < 4.2 < 4.6 < 6.6) undisturbed.

**Result: 165 PASS / 2 FAIL.** Baseline before the change was 160 / 2 on this machine; the
five new checks account for the difference. The two failures are the long-standing
saturation-hierarchy and neighbour-pair-coupling ones, byte-identical before and after.

### 8g. `cpp_codegen.py` — the generated header carried the same wrong claim

`HPP_TEMPLATE`'s `sigma_shape_toe` doc comment repeated "negatives ~0.4/1.0/1.2 monotone",
so the generated C++ would have contradicted the corrected Python docstring. Corrected there
too, with the citation and the interior-peak limitation.

### 8h. `gen_active_profiles.py` — the traceability report reports the new property

New "Grain sigma(D) Shape" column and a `sigma_shape` evidence key whose regex matches only
the *traced* provenance wording, so an estimated triple still prints red. Coverage reads
**4 of 154 (3 %)**. Without this the report would have silently omitted a newly documented
[T1] property, which is the drift that file exists to prevent. The four VISION3 provenance
comments also gained an explicit `SOURCE` citation so the evidence scanner can see them.

### 8i. Verification actually performed

| Gate | Result |
|---|---|
| `verify.py` | 165 PASS / 2 FAIL (the two known) |
| `cpp_codegen.py` | regenerated `film_profiles.{hpp,cpp}`, `film_names.txt`, `film_enum.hpp` |
| `g++ -std=c++14 -Wall -Wextra -c film_profiles.cpp` | **exit 0 AND zero bytes of output** |
| compile + link + run a TU that reads the values back | 154 entries; all four triples read 0.39/1.00/0.63, 0.59/1.00/0.57, 0.41/1.00/0.58, 0.67/1.00/0.55; enum count 154 |
| `vision3_granularity.py` | reproduces all four adopted triples from the PDFs |
| overlays | inspected on all four sheets, all six tracks correct |

The compile gate was checked on **g++'s exit code together with an empty output stream**,
not through a pipe — the mistake that once reported "clean" while a string literal was broken
cannot recur here, and the link-and-run step is the stronger form of the same check.

### 8j. Documentation

`DIGITIZATION_QUEUE.md` (entry moved to DONE; rule 10 amended; rules 15 and 16 added; the
σ(D)-heuristic follow-on opened), `DIGITIZATION_QUEUE_history.md` (batch 15),
`doc/README.md` (status entry), `FilmActiveProfiles.md` and `FilmCurves.md` regenerated,
`film_names.txt` regenerated, and this file.
