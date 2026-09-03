# PROGRESS.md — one screen, current state

**Updated 2026-08-29.** This file exists because status was previously
spread across twenty-two documents and the owner had to read a sweep to see where things
stood. **Rule (method rule 13): this file is updated at the end of every task, before the task is
reported done.** It is partly enforced rather than promised — the stamp at the bottom carries
the schema version, stock count and `film_names.txt` digest, and **`build.py`'s docs stage
FAILS if the stamp disagrees with the live database**. If this file disagrees with a dated
`RESULT_*` document, that document is the detail and this is the index.

---

## CURRENT STATE — 2026-09-03

**Build green.** `verify.py` **509 PASS / 1 FAIL** (the saturation-hierarchy baseline, left failing
on the owner's instruction). **29 audits registered, all green**; SKIPs are for sources on
the owner's machine. `cpp_parity` green across the whole database, `interimage_parity` worst
**5.335e-05**, `spectral_mono_parity` **68/68 with zero guard gaps**, `doc_consistency` **31/31**,
`g++ -std=c++14 -Wall -Wextra` clean on all 18 TUs with zero bytes of output.

⚠ **2026-09-02b, queue TK1–TK5 — Takano 1969 (`JAPAN/23_13.pdf`).** Three things worth the index:
(1) **Fig. 8 is the first measurement this project's aperture term has ever been checked against,
and it fits** — `grain_reference_energy` unchanged reproduces both Selwyn saturation curves to
rms 0.007–0.020 in G. (2) ⚠ **The clump census queue C45 was missing now exists and disagrees:**
five direct measurements, 0.87–4.64 µm, **median 2.46 against a stored median of 13.0 — 5.3×.**
C45's document blocker is discharged and the row is now an **owner decision**. (3) **eq (2), σ(D)
from σ(T) to fourth order, is adopted** as `film_sim.sigma_density_from_transmittance` — the
correction that withdrew `σ_D = 0.648·D^0.665`; **inert**, no render path calls it. ⚠ **A stock was
also asked for and refused**: FUJI NEOPAN S cannot be profiled from this corpus — three papers
measure its grain and nothing measures its tone scale (`EMULSION_KNOWLEDGE_BASE.md` §23k.8).
Detail in `RESULT_2026-09-02b_takano_1969.md`. **No stored profile value changed; codegen output is
byte-identical apart from timestamps.**

⚠ **2026-09-02c, batch E5 + C45 + C46 + C16 + C18 + C19 + C2c — one trace, six decisions, and
THREE OF THE ROWS WERE WRONG.** (1) **E5** traced a clean σ(D) for `EASTMAN_5294_1983`
from Sehlin & Kennel's Fig. 8 (density and RMS granularity on ONE shared log-exposure abscissa),
validated it against the eleven vendor sheets — and then ⚠ **WITHDREW it: `cpp_parity` rejected the
adoption at 5.7e-01 against a 2e-05 tolerance**, and the cause established a convention nobody had
written down — **`sigma_shape_*_at` are PER-LAYER ANALYTICAL densities**, evidenced by `toe_at`
equalling the green curve's dmin on every measured stock. Measured count stays 13. Level and
Fig. 12's f50 refused too. (2) **C18** bounded the
largest undocumented number in the colour path at the curve's own Dmax — **provably non-binding, so
renders are bit-identical**. (3) ⚠ **C2c and C19 were comparing a frequency with a length**;
`adjacency_um` has a closed form, **f = 206.07/adjacency_um c/mm**, and the "systematic" claim
becomes +8 %/−41 %/+102 % once converted — while the conversion exposed **12 stocks whose
`adjacency` is inert**. (4) ⚠ **C46's invented ×4.05 cancels** under the row normalisation that
follows it. (5) ⚠ **C16's recommended option was arithmetically wrong** — a 1.0 px gate kills the
edge term on every 35 mm render under ~4000 px. (6) **C45 refused** with the cost of the alternative
measured. ⚠ **THE OWNER-DECISION CATEGORY IS NOW EMPTY**: queue **113 rows / 95 closed / 16 live**,
and not one live row is a judgement call. Detail in
`RESULT_2026-09-02c_batch_E5_C45_C46_C16_C18_C19_C2c.md`.

⚠ **2026-09-02e — the D2 ruler, not the D2 measurement.** Stouffer's target-density ladders for all seven
transmission step wedges are harvested into **`doc/STEP_WEDGE_REFERENCE.md`**, so a future scanner-characterisation
scan no longer has to assume its own step values. ⚠ **D2 stays open** — it asks for a scan, and without one there is
still no way to split emulsion σ from scanner σ. Three silent-corruption traps are recorded in that file: the
«% transmission» column is `100·10^−D` restated and carries no independent information; T3110's third column is a
×0.8 nominal ladder off by 13 % at step 2; and the densities are base-inclusive at 0.05.

⚠ **2026-09-02d, queue N1 — a stock refused in the morning and added in the afternoon.**
`EMULSION_KNOWLEDGE_BASE.md` §23k.8 had recorded that FUJI NEOPAN could not be profiled; the owner
supplied **FUJIFILM AF3-411E(N)**, the NEOPAN SS (135) data sheet, and it is the missing tone
scale. **`FUJI_NEOPAN_SS` added as stock 172**, appended at frozen id 171 so no ListBox index
moves. ⚠ Its §9 **prints the average gradient on every curve**, so the trace is self-checking: the
stored 10 min member fits to **0.0234 D** with a model Ḡ of **0.552** against a printed 0.53.
⚠ **Two calibration traps recorded, and a lesson**: the abscissa frame is logH −4.0 where the
leftmost printed label is −3.0, and *a gradient check cannot catch that* because a gradient is a
ratio of differences. ⚠ **Grain and MTF are flagged class estimates and must stay so** — the sheet
has no image-structure section, and the four "Neopan SS" granularity measurements on file are of
the 1959–1969 coating against a 1999 sheet. §23k.8 is **corrected, not deleted**: its inventory is
still right for NEOPAN S and SSS. Detail in `EMULSION_KNOWLEDGE_BASE.md` §23m.

⚠ **2026-09-02e — one batch, three parts, and the item scheduled as housekeeping was the one
that mattered.** Full account in `RESULT_2026-09-02e_batch_A1_A4_A2_T1_T2_T3_E4_takano1968.md`.

* ⚠ **A4 — `adjacency` was stored wrong on EVERY stock traced from a sheet, and a guard was
  holding it in place.** The field was set equal to the overshoot the sheet prints, but the
  renderer multiplies the rolloff by a difference-of-Gaussians lift, so the stored number is
  the amplitude BEFORE attenuation. **Across the thirteen sheets in `mtf_vector.EXPECTED` the
  stored pair reproduced the traced peak on ZERO records** — PLUS-X 5231 printed +3.4 % at
  4.6 c/mm and rendered no peak at all, and `verify.py` pinned the value that guaranteed it.
  Fixed by SOLVING the two-parameter system against each sheet's own peak value and peak
  frequency: **13 of 13 now reproduce to < 2e-3 in level and < 0.2 c/mm in position.**
  ⚠ **Five red records REFUSED** — their maxima sit on the first traced sample, and solving
  anyway returns `adjacency_um` 74–84 µm on every one; a guard now forbids that band.
  Eleven stocks still render no overshoot, **each with a written refusal**.
* ⚠ **A2 / C16 — supersampling refused, measured.** Single-thread, 1920×1080, three channels:
  native σ 0.40 px **85.9 ms/frame**, ×2 **559.4 ms (6.51×)**, ×3 **1573.9 ms (18.33×)**. And
  it is not correct anyway — blurring and sampling commute for a band-limited signal.
  ⚠ **The residual is now closed-form: the divergence is ALIASING at Nyquist, where a spatial
  kernel is exactly 2.00× the analytic transfer at every sigma.** "They converge above 1.2 px"
  is not a property of the kernel — it is where 2·T(Nyquist) falls below 1e-3.
* **A1 — corpus sweep closed as a negative result**: 90 PDFs, **0 unaccounted for**.
* **T3 — three new stocks**, ids 172–174: `FUJI_PROVIA_100F`, `FUJICOLOR_SUPERIA_XTRA_400`,
  `FUJICOLOR_PRO_400H`, curves and MTF traced. ⚠ Four tracing defects found on the way, each
  producing plausible numbers; the sharpest is **a fitted shoulder on a panel that stops
  inside the straight line**, which put SUPERIA's red Dmax above its green.
* **T2 — `KODAK_EKTAR_100` gains the first measured MTF for a still colour negative here**,
  and ⚠ **the estimate was 1.5× too sharp on every record** (74.3/80.0/87.6 → 35.5/52.7/54.8).
  `KODAK_GOLD_200` gains a documented absence: **E-7022 carries no MTF panel at all.**
* **T1 — both VISION3 stocks cited to H-1-5219 and H-1-5207**; closes as a citation because
  both sheets' panels are rasters. ⚠ 5219 is acetate **with** rem-jet, 5207 **without**.
* **E4 — the 1942 Eastman book verified.** Every value reproduces; ⚠ **the PDF page number did
  not** (49 → 50). Three additions: SD-21 is a deliberately **seasoned** D-76 (+6 g borax,
  8 g boric acid, 0.25 g KBr/litre), so every printed figure is a seasoned-developer figure;
  Type 5242 is the same emulsion on acetate; and the book's one quantitative graininess
  statement is why rms 12.0 is kept. ⚠ **Plus-X Type 1231 recorded and deliberately NOT merged
  into 5231** — 1942 nitrate against 1999 acetate, guarded.
* **Owner addendum — `31_209.pdf` reviewed**: Masao Takano 1968 Part 2, **new material, not a
  duplicate** of the Kiyoshi Takano review §23k is built on. ⚠ **Reaching a density by
  developing longer gives a 36–40 % smaller clump than reaching it by exposing more** — a
  variable this schema does not carry. ⚠ **And a second independent source for C45**: its
  mottle sizes over its own stated 5–8× aggregate factor give a grain of 0.50–1.36 µm against
  BBC T-101's printed 0.59–1.43 µm, while the stored `clump_um_g` median is **13.0**.
  Knowledge base §23n.

⚠ **2026-09-03, queue C45 — THE CORPUS-WIDE `clump_um` RESCALE, owner-approved, and the previous
day's refusal did not survive measurement.** Every ESTIMATED `clump_um_r/g/b` divided by **3.1**;
the five T-101-measured stocks exempt. **510 literals across 170 profiles; median 13.00 → 4.19 µm,
maximum 40.0 → 12.90.** Knowledge base §23k.9.

* ⚠ **WHAT HAD BEEN CONFUSING THE ROW: `clump_um` is the GRAIN scale, not the clump.** It sets
  `f_hi = 500/clump_um`; `clump_gain` sits at `f_lo = f_hi/6`, a length **six times** longer, and
  **Takano 1968 measures the developed aggregate at 5–8× the grain** — corroborating the model's
  hard-coded 6 from a 1968 document. Every source now converts into one parameter instead of three
  different physical scales being compared with each other.
* ⚠ **THE REFUSAL'S REASON, TESTED.** C45 was refused on 2026-09-02c because the rescale makes grain
  more resolution-dependent. But `rms_granularity` is **defined** through a 48 µm aperture, and the
  aperture-referred rms is **invariant to 1.3 %** across the whole clump range and every render size
  (5.263/5.203/5.228 at clump 13.0 against 5.194/5.145/5.164 at 2.46). **"Same film, correctly
  resolved", not "more grain"** — and `grain_reference_energy`'s docstring had already ruled on the
  direction years earlier.
* ⚠ **AND IT IS NOT A PARAMETER-BASIS ARGUMENT.** A lobe can only push the half-power DOWN — 22.6
  c/mm at gain 0, 13.7 at 0.25 — against Ooue's measured **45.6 / 70.8 / 140.7**. No `clump_gain`
  reaches the measurements, so the estimates really were a different spectrum.
* **Anchor: Ooue Fig. 26 alone** (three directly measured Wiener spectra on named stocks, median
  4.16 µm), not the full census's 2.46 — the conservative choice, still inside the band every other
  source brackets.
* **`clump_gain` deliberately untouched.** T-101 says there is no lobe; Takano 1968 measures one.
  Method rule 4: record the conflict, do not average it. Consequence stated: each rescaled value is
  conditional on that stock's stored gain.
* **Visible effect** +3 % at 960 px, +27 % at 2000 px, +63 % at 4000 px. **Cost none**: 716.6 →
  689.6 ms single-thread through the full `simulate()` at 2000 px.

⚠ **2026-09-03, `Photographic granularity. Mathematical formulation.pdf` — Lu & Torquato,
JOSA A 7(4) 717–724 (April 1990), assessed, NOT a duplicate, and NOTHING GOES INTO A PROFILE.**
Pure theory: no named film, no emulsion, no measured stock. Its value is method and validation, and
it changes three existing rows rather than opening any.

* ⚠ **QUEUE F1 SHRINKS AND ONE OF ITS CITATIONS LOOKS WRONG.** Eqs (4.1)+(4.2) reproduce **Bayer's
  random-dot closed form** — the paper says so in words, "agrees with the expression given by Bayer
  (apart from a trivial constant...)" — so F1's first acquisition target is **in hand without the
  original**. Its reference list carries Bayer JOSA 54 1485 (1964), Trabka JOSA 61 800 (1971),
  Hamilton/Lawton/Trabka 1972 and Castro/Kemperman/Trabka, and **contains no Wilder at all**; F1
  lists "Wilder JOSA 62 (1972)" as a target, and a 1990 granularity review citing four
  Trabka-school papers while skipping it is evidence that row is wrong or marginal. F1 also
  records "no JOSA paper of any kind is in the corpus" — **false as of today**.
* **SELWYN'S LAW DERIVED, WITH THE CONSTANT IDENTIFIED, AND IT IS OUR APERTURE INTEGRAL.**
  Eq (3.8) `G = K·Aa^(−1/2)`; Eq (3.9) makes `K` the square root of `∫[S₂(r) − φ₂²]dr`, i.e. the
  **zero-frequency limit of the autocovariance**. That is what `grain_reference_energy` integrates
  through the 48 µm aperture. Independent theoretical confirmation of the mechanism C45 established
  empirically the same day (1.3 % aperture-referred invariance).
* **TABLE 1 IS A FREE NUMERIC VALIDATOR** — G for impenetrable disks, 23 scaled aperture areas
  (Aa/A₁ 0.0–25.0) × 7 disk area fractions (φ₂ 0.1–0.7), 161 numbers. Our aperture law can be
  checked against it with no film involved.
* ⚠ **IT SUPPLIES THE MISSING MECHANISM FOR THE T-101 / TAKANO-1968 CONFLICT IN §23n.**
  Impenetrable grains cannot overlap, so they "constitute a less random distribution than the
  random-dot model" — short-range **order**, i.e. **NEGATIVE** low-frequency correlation and
  granularity **below** random-dot, growing with packing fraction (their Fig. 4: penetrable G rises
  monotonically with φ₂, impenetrable rises, peaks, then falls). T-101 fitting `clump_gain` → 0.000
  on six emulsions and Takano 1968 measuring a real 5–8× aggregate can therefore **both be true** of
  different emulsions. ⚠ **Our model can only express the clustering half**: `film_sim.py:1248`
  `if clump_gain > 0.0:`, `Algo_11_Sim.cpp:69` `if (clumpGain > 0.0)` and its two
  `MAX_VALUE(clumpGain, ALGO_ZERO)` clamps at 191/234. **Checked both — Python and C++ agree, so
  there is NO PARITY BUG**, but a negative gain is unrepresentable by construction in either.
  Their **penetrable-concentric-shell model**, one impenetrability index λ ∈ [0,1] spanning
  λ=0 random-dot to λ=1 hard disks, is the principled one-parameter replacement if the ordered
  side is ever wanted.
* **Nothing new was queued** — the four actions it enables (shrink F1, correct F1's Wilder line,
  add Table 1 as a numeric guard, record the §23n mechanism) all land on rows that already exist.

⚠ **2026-09-03, THE PYTHON/C++ PARITY BATCH — the one true law divergence is closed, and two
schema gaps with it.** Full account in `PARITY_PLAN_2026-09-03.md` and
`DB_ALGORITHM_COVERAGE_2026-09-03.md`, both of which had to be corrected the same day because the
batch invalidated their headline findings.

* ⚠ **THE MEASURED MTF ROLLOFF NOW RENDERS IN C++, ON BOTH TWINS.** 22 stocks carry a measured
  `1/(1+(f/f50)^q)`; Python applied it and the C++ engines could not, because the law is a
  frequency-domain form and neither twin has an FFT. They rendered the legacy single Gaussian —
  correct at f50 by construction and **up to 3.8× too much modulation at 2× f50**. Closed by
  `FilmMtfKernel`, a **two-Gaussian separable fit** that stage 6 convolves as two more lobes of the
  multi-Gaussian blur it already ran for the adjacency band-pass, so **no new machinery and no FFT**.
  Worst max|error| **0.0384** against **0.1737** for the Gaussian it replaces; every affected stock
  improves, by **1.4× to 10.0×**.
* ⚠ **THE TABLE IS KEYED ON THE EXACT STORED q AND MUST NEVER BE INTERPOLATED.** The two-lobe family
  has **two disjoint optimal basins** straddling q ≈ 3.0: below it a small tight lobe on a wide one
  (w1 0.005–0.36), above it a slightly over-weighted narrow lobe minus a very wide one (w1 1.00–1.08).
  A table interpolated across that switch fits neither side. A guard asserts the two basins stay
  separated so that anyone who starts interpolating trips first.
* ⚠ **A closed-form fit was tried first and REFUSED.** Matching the law at f50 and 2·f50 with a fixed
  sigma ratio gives 0.117 — barely better than the single Gaussian's 0.1325 — because matching two
  points is not minimising the error. The pinned minimax fit is what earns the change.
* **`cpp_parity` reorganised honestly**: `FilmMtfResponse` left `LAW_BYPASS_BASELINE` **because it
  gained a caller**, which the file states is the only admissible reason. ⚠ Its entry had also gone
  stale — it said **9 stocks** where the database has **22**. The indirection is now declared in
  `LAW_EQUIVALENT_IMPL`, ⚠ **flagged as a different kind of entry from `FilmGrainSigma`'s**: that one
  is the same law with arithmetic hoisted and agrees exactly; this one is an approximation with a
  stated bound.
* **Two schema gaps closed**: the spectral `TakingFilter` struct (only a bare `cut_on_nm` scalar
  crossed before) and `ReciprocityTable.development_correction_pct`. Empty and 3 stocks
  respectively, so nothing renders differently — they are emitted so the first stock to carry one is
  not a silent divergence.
* **Python gained `RenderSettings.mtf_use_kernel`, off by default.** ⚠ Additive: Python keeps the
  exact law as the reference, and the switch renders it through the same kernel when the three
  implementations must be compared on identical arithmetic. Turning it on makes Python *less*
  accurate, which is why it is not the default.
* ⚠ **`exposure_index` REFUSED as an exposure-placement input, and this is not a gap.**
  `solve_anchors` exists to land 18 % scene grey on target, so any EI-derived global exposure shift
  is exactly the quantity it removes: wiring it before the solve is inert, wiring it after breaks
  the grey landing on all 175 stocks. EI is not missing from the renderer — it is incompatible with
  automatic neutral anchoring. It would matter in a fixed-photographic-exposure mode, which does not
  exist and is a product decision rather than a data one. EI **is** consumed where it belongs, in
  `ProcessVariant` resolution.
* ⚠ **Stage 3c temporal flicker REFUSED in both languages.** `AlgoTemporalFlicker.hpp` records that
  the model cannot be written yet: the spectral shape is unspecified (1/f or 1/f², "the two look
  quite different") and there is no control for the common/per-channel split. The header even carries
  a correction where two controls were previously *invented*. Implementing it would mean choosing a
  spectrum with nothing behind the choice.
* **Still open and stated plainly**: Python has no port of C++ stages **9b, 15 and 16** (negative
  defects, gate weave, machine defects). ⚠ They are the least-verified consumed code in the system —
  they read `weave_amp_x_um`, `weave_amp_y_um`, `weave_hz_corner` and `dirt_events_per_frame` and
  nothing independent checks them. A bit-exact port needs the counter-based RNG reproduced in Python;
  a statistical port would not satisfy `cpp_parity`'s tolerance. Not attempted rather than half done.
* ⚠ **The two pipelines' stage numbers diverge after 14** — Python 15 is sRGB encode, C++ 15 is gate
  weave — and Python's encode must NOT be added to C++, where the host owns the transfer function.
* `ALGO_BLUR_MAX_LOBES` raised **4 → 6**: two base lobes each carry their own inner and outer
  adjacency partner. Both twins compile clean at `-Wall -Wextra`. `verify.py` **529 PASS / 1 FAIL**
  (the saturation baseline, left failing on instruction).

⚠ **2026-09-03, queue G5 — CLOSED, and it was never the acquisition the row said it was.**
`GEVAERT/Rens_vanBets1968Gevachr6.00.pdf`, Gevachrome Typ 6.00 / 6.05, Kino-Technik 1968 Nr. 10.
The row asked the owner for **a 300+ ppi re-scan of three printed pages** and rated it *high impact*
because it "unblocks G2 and upgrades two profiles from [T3] estimates". Detail in
`RESULT_2026-09-03_G5_gevachrome_gradation.md`.

* ⚠ **THREE OF THE ROW'S FOUR BLOCKERS WERE ALREADY DEAD WHEN IT WAS READ.** G2 traced and adopted
  Bild 2a/2b (spectral) and Bild 1a-c (MTF) on 2026-09-02 from the scan the row calls unusable. Only
  the Bilder 5a/5b layer curves were actually outstanding. ⚠ **And the row's own facts were wrong**:
  the source is **grayscale in an RGB wrapper** (chroma exactly 0.0), so the "separate the layers by
  colour to defeat bleed-through" plan was never available, and the embedded images are **150 ppi**,
  not the 150-is-not-enough the row implies.
* ⚠ **THE RE-SCAN WAS NEVER NEEDED. THE FIRST TRACE WAS RUN AT 114 ppi BY MISTAKE** — a page render,
  not the embedded image. Rendering the same JPEG at 300 dpi separates all three curves cleanly.
  The blocker was the reader's, not the source's, and the row spent a day of owner action on it.
* ⚠ **THE γ CONVENTION WAS THE REAL OBSTACLE, AND IT IS NOW SETTLED.** Gevaert print γ 1.45 / 1.25 /
  1.25 (Typ 6.00) and 1.35 (6.05's cyan) and **never define how they measured it**. A sliding-window
  max-slope estimator returns **1.895 against a printed 1.45, +31 %**, and no window width fixes it.
  **Least squares over the straight-line portion, D 0.5–2.0, returns all four printed values: +1.8 %,
  −0.2 %, +1.9 %, −3.6 %.** That identifies the convention and turns the caption from an obstacle
  into a validator. ⚠ A band defined as a *fraction of each curve's own throw* was tried and is
  **worse** (+9 % on 5a's curve a), which says the printed figure is a fixed density interval.
* **Adopted: six channel shapes upgraded [T2] → [T1] on `GEVACHROME_600` and `GEVACHROME_605`.**
  γ from the caption, `dmin` from the right-hand plateau, Dmax from the left edge, span =
  (Dmax − dmin)/γ, and only the two softness parameters fitted. ⚠ **The transfer they replace was
  half right and half backwards**: `GEVACHROME_902`'s toe_k 0.18 sits inside the traced 0.18–0.24,
  but its shoulder_k 0.30 is **2.5 to 8× too soft** — on a reversal `shoulder_x` is the *shadow* end
  and Bild 5a's Dmax corner is nearly square.
* ⚠ **BILD 5b DOES NOT SHARE BILD 5a's ABSCISSA** although the panels are the same size, in the same
  column, one above the other. 5b's frame starts at **lg i·t 0.45**. Reading it on 5a's grid shifts
  every point 0.40 decades and silently rescales the throw.
* **New: the only measured reversal push in the database.** Bild 6 is Typ 6.05 exposed at **26 DIN
  (320 ASA)**, one stop over box speed, stored as a `ProcessVariant` with its own curves. ⚠ **It
  moves the opposite way from a negative's push** — extra first development *spends* the silver that
  would have become the positive image, so γ falls **1.376 → 1.156 (−16 %)** and Dmax **2.505 →
  2.223**. ⚠ **A verify guard said "every push set is contrastier" and this failed it on all three
  channels. The guard was wrong, not the data** — it had been generalised from two C-41 negatives —
  and is now split by `is_reversal` with the reversal branch asserting the opposite sign.
* ⚠ **REFUSED: Bild 6's dmin.** Its right-hand decade is under the page-curl shadow. A per-column
  background normalisation recovers ink to about lg i·t 2.7, where the merged tail reads 0.096 and is
  still falling, but the lg i·t 3 gridline never comes back, so the tail cannot be placed on the
  abscissa. dmin is **inherited from the box-speed record and labelled as inherited**.
* ⚠ **A DEVELOPMENT TIME CONFLICTS WITH ITSELF INSIDE ONE PAPER.** p262 prints 3.5 min for Bild 6's
  first development; p264's Tab. IV footnote prints 2.5 min + about 45 s = 3.25 min. **Method rule 4:
  both recorded, neither averaged.**
* ⚠ **A BUG IN THE NEW READER THAT PRODUCED PLAUSIBLE NUMBERS.** `page_images` returns 0–255;
  the new ink threshold was written as 0.55 as though it were 0–1, selecting only pure black. It
  returned γ 1.337 / 1.292 against printed 1.45 / 1.25 — **wrong by 8 % and entirely believable**.
  Caught by the fit rms tripling to 0.10 and the column count falling 93 → 71, not by the gammas.
* **Row reclassified**: acquisition → worked-with-what-exists, impact **high → low**, owner action
  **none**. `verify.py` **526 PASS / 1 FAIL** (the saturation baseline, left failing on instruction).

⚠ **2026-09-03, `Photographic granularity. Mathematical formulation.pdf` — THE FIRST VERDICT WAS
WRONG AND THE OWNER CAUGHT IT.** Lu & Torquato, JOSA A 7(4) 717–724 (1990). It was assessed as
"pure theory, nothing goes into a profile, zero database rows change". The owner's objection — that
a 1990 statistical-mechanics paper should carry more weight against 1950s–60s empiricism, not less
— is right, and the test that settles it was already in the database.

* **The paper predicts granularity RISES with coverage, peaks, then FALLS**, because impenetrable
  grains cannot overlap and so fluctuate less than a random-dot field. Selwyn's √D law rises forever.
* ⚠ **ALL ELEVEN MEASURED COLOUR NEGATIVES IN THIS DATABASE ALREADY SHOW THE PREDICTED TURNOVER.**
  `sigma_shape_peak` sits at D 0.65–0.80 on every one, and σ then **falls to 0.35–0.74× of peak** at
  Dmax. A √D law predicts a **rise of 1.66–1.84×** over the same interval. **The disagreement is a
  factor of 2.3 to 5.3, in direction as well as size**, and until now the interior peak was an
  empirical shape with no mechanism behind it.
* Converting Table 1 to density (D = −log₁₀(1−φ₂)) makes the same point numerically: at every real
  aperture the √D law overshoots the theory by **1.8× to 2.7×** by D 0.52.
* **What actually changes**: **F2b** — 55 monochrome negatives on a placeholder σ(D), blocked on an
  unobtainable Higgins & Stultz — now has a **principled one-parameter shape** available instead of
  a placeholder. **F1** shrinks (Bayer's closed form is reproduced as Eqs 4.1/4.2) and its **Wilder
  citation looks wrong** (absent from a 1990 review that cites four Trabka-school papers).
* Still true: no named film, so it can set a **shape and never a level**; monolayer disks, table
  capped at D 0.52. ⚠ And the mechanism is **negative** low-frequency correlation, which
  `clump_gain` cannot express — `film_sim.py:1248` and `Algo_11_Sim.cpp:69`/191/234 clamp it at zero.
  **Checked both: Python and C++ agree, so there is no parity bug**, only a missing half of the model.

**Database.** 175 film stocks, 11 print stocks, 14 gauges, **schema v24**. 134 negative / 41
reversal; 69 monochrome. Provenance **88 T1 / 45 T2 / 42 T3**. *(Re-counted from the live module
2026-09-02e; `doc_consistency.py` registers the headline numbers in `NotFound.md` and this file and
fails the build if either drifts.)*

**Coverage** (live; the authority is `FilmActiveProfiles.md`, regenerated on every build):

| carrier | measured | of 172 |
|---|---|---|
| spectral sensitivity curve | **87** | 85 remaining |
| resolving power, printed pair | **59** | 113 |
| manufacturer reciprocity table | **42** | 130 |
| measured MTF | **19** | 153 |
| spectral dye density (three dyes) | **18** | 154 |
| neutral + D-min pair | **16** | 156 |
| σ(D) grain shape | **13** | 159 |
| published coated thickness | **12** | 160 |
| published base thickness | **12** | 160 |

⚠ **σ(D) STAYED AT 13 ON 2026-09-02c AND THAT WAS A RESULT, NOT AN OMISSION.** A fourteenth was
traced from a journal plate and then withdrawn on a density-space mismatch `cpp_parity` caught —
see `RESULT_2026-09-02c…` §1 and `EMULSION_KNOWLEDGE_BASE.md` §23l.1.

**9 stocks carry no source of any kind** (plus `GENERIC_BW` / `GENERIC_COLOR`, which are classes,
not gaps). Per-stock acquisition plans in `NotFound.md` §1.

**Engine.** **26 stage entry points**, and **all 26 exist in both the scalar reference and the AVX2
production path.** ⚠ Read from `AlgorithmMain.cpp` and registered in `doc_consistency.py` — a count
grepped from the stage `.cpp` files says 25, because 12b Callier is an inline header function. Scalar is the high-accuracy reference (`AlgoType = double`); AVX2 is production
(`AlgoType = float`); `HighPrecType = double` in both. Unaligned SIMD loads/stores on image buffers,
scalar tail on every kernel, no AVX-512. Full table and the two known C++ gaps: `README.md`,
*Implementation status*.

**Queue.** **102 rows, 78 closed, 24 live.** Not one live row is blocked on tracing harder:

| in the way | rows |
|---|---|
| nothing but the work | 5 — E4, F3, T1, T2, T3 |
| an owner decision | 6 — C4, C7, C16, C18, C19, C2c |
| a small owner action | 3 — D1, D2, G5 |
| G1 approval, then work | 1 — G2 |
| a model or schema decision | 2 — C23, E5 |
| configuration, not acquisition | 1 — M1 |
| a document proved absent | 6 — C14, F1, F2b, G6, K5, K6 |

⚠ **The largest gap in the project is not a document.** Every audit checks the database against its
sources or one engine against the other; **nothing checks a render against a photograph.** Closing
every queue row would leave that unanswered.

---

## What changed on 2026-08-31c — the corpus reconciliation, B3 and E3

**Two queue rows closed, seven "acquisitions" reclassified, nothing opened.**
Detail: `RESULT_2026-08-31c_reconcile_B3_E3.md`.

⚠ **The reconciliation is the headline.** This working copy holds **56 PDFs** under
`PDF/PROFILES`; the owner's machine holds **475**, in twenty directories, of which this checkout
has six. **Thirteen rows were filed under acquisition and six survive.** B3 and E3 had their
documents (closed); G5 is an owner action by its own description; M1 became configuration when the
2383 sheet arrived; and **T1, T2 and T3 name four publication codes between them and get three
wrong about which film they describe** — E-4046 is EKTAR 100, not the VISION3 sheet; F-4017 is
TRI-X 320/400, not GOLD 200; H-1-40295 exists nowhere and cannot be a still film's code.
⚠ Two rows had the **opposite** error: E4 and G2 were filed "source on disk" and their sources are
on the owner's machine, not here. And **E5's file is misnamed** — `Sehlin_Kennel_etal_1983_…pdf` is
SMPTE Journal, **July 1985**, p. 724.

| item | outcome |
|---|---|
| **B3** ✅ | `KODAK_TECHNICAL_PAN`'s **first spectral set** (P-255 p9, 31 samples) — one of the two flattest panchromatic curves here, 0.56 decades, which is the trace agreeing with the sheet's own prose. `KODAK_TMAX_400` did not want "its second criterion": its set closed 2026-08-16 and what it lacked was a **cross-edition validation**, now rms **0.005** against the 2007 edition it had never read. ⚠ Two reader assumptions broke on these sheets — a caption test that required both "ABOVE" and "=", and committing to the first frame that calibrated. ⚠ The second criterion is now **measured** on every mono sheet, which is the check that would have caught C38: swapping a criterion pair flips the sign of the gap |
| **E3** ✅ | The **first adoption from a corpus sheet that is raster end to end**. `konica_raster.py` calibrates geometrically off the printed grid; all seven panels re-detect their own gridlines. ⚠ The bitmaps are stored **upside down**. `KONICA_IMPRESA_50`'s Dmin triple was a **family template** shared with two other KONICA stocks and **wrong in blue by 0.32 D** — refuted by two figures on two pages agreeing to 0.005–0.015 D. Its MTF f50 is **64.9**, not 72, with a 121.4 % overshoot and a power-law rolloff beating the Gaussian 2×. `KONICA_INFRARED_750`'s gamma moved **0.72 → 1.70**: all fifteen printed curves are steeper than the value held, the flattest being 0.814 |
| **queue census** ✅ | **26 live → 24**, 78 struck of 102. Category table, §0 header and the tier lists all re-derived from the parse |

⚠ **Not adopted, recorded as unobtainable rather than deferred:** a Konica dye triple (p3 draws two
NEUTRAL spectra, not three dye curves), any per-layer Konica MTF (one visual-filter curve), INF750's
absolute spectral level (its panel has no y tick labels at all), and `professional_160.pdf`, whose
four pages extract **zero characters**.

⚠ **A third data point for C2c, deliberately not acted on:** IMPRESA 50's overshoot peaks at
6.88 c/mm against a stored `adjacency_um` of 14.0 — after 5231 (4.7 vs 16.0) and FUJI F-125 (~9 vs
13.0). Three stocks, two manufacturers, one direction. The amplitude was adopted; the length is
C2c's decision.

---

## What changed on 2026-08-29

Four queue items closed and two opened. ⚠ **Three of the four closed rows were factually wrong
about their own scope, and one of the two new rows exists because a finding was sitting in a
docstring where nothing tracked it.** The recurring lesson, which this file has now recorded
three separate times: **a readiness label decays, and the only way to know is to open the
document.**

| item | outcome |
|---|---|
| **G4** ✅ | Gevachrome II process transcribed from Webers & Westendorp 1979 (⚠ image-only scan, `pdftotext` yields 3 bytes). **0 profiles created**, as the row required. The real finding is a GENERATION BOUNDARY: the corpus's `GEVACHROME_600/605/902` run the FIRST Gevachrome process (12 steps, 21/25 °C, GP 110) and these four 1979 types run 15 steps at 25 °C with GP 112 — the three stored profiles must not be relabelled |
| **E1** ✅ | 8 profiles changed, 2 new audits, `ProcessingFamily` **doubled** 22 points/4 stocks → 42/8. Kodak 1952: gammas 0.780→0.744, 0.700→**0.852**, 0.680→**0.832**, 0.720→0.800. Agfa 2004: 12 tone curves fitted at rms 0.005–0.016 D, 3 new spectral sets, `AGFA_OPTIMA_100`'s red peak corrected 650→615-620 nm. Render impact up to **0.45 D / 12 eight-bit codes** |
| **C37** ✅ | **Yielded no new data, and that is the finding** — the row promised "up to 13 new sets" and every stock behind its 15 panels already had one. Delivered instead: `spectral_vector.py`'s registry 4 → **11 sheets**, every agreement pinned, `--assert` now fails on drift. Also corrected the comparison itself (see `_core_rms`) and overturned a recorded diagnosis |
| **B1** — | Already closed 2026-08-26. Its residue was one sentence inside a closed row, tracking four unread plates; promoted to **B4** |
| **C38** 🆕 | Three vector re-reads disagree with adopted raster sets by more than reading error — 5245 blue 0.335, 5218 0.241/0.210/0.138, 5231 pan 0.213. Pinned so they cannot drift; adjudication is an owner call |
| **B4** 🆕 | `EASTMAN_5247_1983`'s four TI0835 plates, all colour-coded with printed legends. Two are new data — a measured MTF whose abscissa says **cycles/mm in words** (so G6's units question does not apply) and a dye-density neutral pair for the carrier schema v14 created. Blocked on axis calibration: 113 dpi, no text layer |

**The build reached 0 failures / 0 warnings**, and not by changing any audit:
`vision3_granularity.py` and `kodak_still_curves.py` had been reported as failing for weeks and
were both missing PDFs, proven by re-running them against the pristine archive.

⚠ **THAT "FIRST FULLY GREEN BUILD" CLAIM IS WITHDRAWN, SAME DAY, BY THE WORK BELOW.** It was run
against a root holding no algorithm sources, and `cpp_parity.py` **skips** its twin-consistency,
law-reachability and grain-stage probes when those are absent — five `[SKIP]` lines and exit 0,
verified. Green partly because three checks were not looking. Run against a root that carries the
algorithm tree, the long-recorded C30/C33 grain-level failure appears exactly as this file already
described it on 2026-08-27. Nothing regressed; the claim was simply broader than the evidence.

---

## What changed on 2026-08-29, second session

**Not a queue row.** The owner asked why `AGFA_APX_100` and `AGFA_APX_400` showed a red,
estimate-marked *Spectral Sensitivity* cell when Agfa's sheets carry the curves at good
resolution. They had not been ignored — both were vector-traced 2026-08-17 to 0.50 nm and
0.0034 log. **The column simply prints a different parameter than its heading suggests, and it was
printing a value the renderer does not read.** Detail:
`RESULT_2026-08-29c_spectral_weights.md`.

| finding | outcome |
|---|---|
| ⚠ **Python and C++ rendered different monochrome images** | `Algo_07_Sim.cpp` derives the collapse weights from the traced pan curve **unconditionally**; `film_sim` gated the same derivation behind `spectral_mono`, default **False**. 24 stocks affected; worst `KODAK_PLUS_X_125`, blue **0.110 vs 0.502**. Flag now defaults True. ⚠ `verify.py` had a check *defending* the split — rewritten to assert the invariant that survives |
| ⚠ **48 records claimed a derivation nobody ran** | `status='derived'`, `'integrated from the traced log-sensitivity curves'` — all 48 stored the (0.30, 0.59, 0.11) dataclass default, i.e. Rec.601 luma. All colour stocks, where the field is never read, so no frame was wrong; 48 report cells were plain on a false label. A second note, *"No traced spectral sensitivity for this stock"*, was untrue for **28** |
| ⚠ **A guard measured itself on the wrong grid** | `spectral_out_of_reach` / `spectral_peak_lambda` read the curve off the renderer's 730 nm grid while testing for sensitisation beyond it. `KONICA_INFRARED_750` read **730 nm / 0.203**; on its own samples to 830 nm it is **750 nm / 0.437**. Refused either way, but on a figure low by 2x |
| **C39** 🆕 | `ROLLEI_INFRARED_400`: its stored curve is the *unfiltered* sensitisation (peak 410 nm, 0.028 past 700 nm) so no honest guard refuses it, while its authored triple encodes an IR taking filter **no field records**. Needs a `taking_filter` carrier, not a threshold |
| **C40** 🆕 | ⚠ **A wrong render that ships today.** The gamut-reach guard exists only in Python. `AlgoSpectralMonoWeights()` derives for any stock with `log_s_pan`, so `KONICA_INFRARED_750` renders in the plugin at **(0.161, 0.193, 0.646)** — blue-dominant — against the correct **(0.55, 0.15, 0.30)**. ~20 lines to port; left open because algorithm sources were out of scope |

**Two new audits**, both green: `spectral_mono_parity.py` (compiles the plugin's own translation
unit; **67/68 monochrome stocks agree exactly**, the one gap being C40, named aloud in its `[OK]`
line every run) and `spectral_weight_provenance.py` (161 `spectral_weights` records re-derived
from one rule — ⚠ written because `_PARAM_SOURCES_DERIVED` says *"REGENERATE, do not hand-edit:
the rules live in the task EM-A6 generator"* **and that generator is not in the repository**).

⚠ **No `spectral_weights` value changed.** What changed is which value the renderer reads, and
what the records say about it.

---

## What changed on 2026-08-30

**One batch, two queue items, both algorithm-source edits** — owner-approved on 2026-08-29,
started only after he said go. Detail: `RESULT_2026-08-30_C30_C33_C40.md`.

| item | outcome |
|---|---|
| **C30/C33** ✅ | ⚠ **The remainder was bigger than "partially closed" suggested.** `AlgoAddGrain` took a loose `fogGrain` value, so the stage **could not reach the measured `sigma(D)` anchors at all** — a signature that cannot express the law. It now takes `dmin[3]`, `dmax[3]` and the `GrainSpec`; the law lives once in `AlgoGrain.hpp` (`AlgoGrainAmpBuild`/`AlgoGrainAmpAt`) and both twins call it. Worst disagreement against the Python reference **2.52e-07** over 2415 probes; `\|amp − 1\|` at NET density 1.0 is **exactly zero** on all 161 stocks × 3 channels |
| **C40** ✅ | The gamut-reach guard is ported into `AlgoSpectralMonoWeights()`, measured on the profile's own stored samples. `KONICA_INFRARED_750` no longer renders at a blue-dominant (0.1611, 0.1931, 0.6458); both engines use its authored (0.55, 0.15, 0.30). **68/68, no guard gaps.** `--allow-guard-gap` removed from the build rather than left in as a safety net |
| ⚠ a trap avoided | Stages 13 and 14 use the same weighting and the reference does **not** normalise them — print and dupe stocks carry no published rms. A second entry point, **`AlgoAddGrainRaw`**, keeps them unpinned. Pinning them would have been the same error in the opposite direction, and `cpp_parity`'s twin check now requires both names in both twins so the distinction cannot be collapsed |

⚠ **THIS CHANGES THE PICTURE.** Every stock renders **3.9–15.5 % quieter** (mean 9.2 %) — the factor
it was wrong by. And the **13 stocks with a measured shape** now render that shape instead of a
square root: **×1.41 to ×2.82 in the shadows, ×0.37 to ×0.61 in the highlights**. That is what the
granularity plots on those sheets show and what the square root could never produce. It is a
correction, not a refinement.

**Single thread, isolated, best of 12 at 1920×1080, one core.** Scalar legacy branch
2.253 → **2.489 ns/px** (+10.5 %); scalar measured branch 2.239 → **2.129 ns/px** (−4.9 %, the
piecewise-linear lookup beats a square root). AVX2 legacy 0.817 → **0.834 ns/px** (+2.1 %, one extra
`_mm256_mul_ps`); AVX2 measured 0.757 → **1.010 ns/px**, ⚠ **not a regression on equal work** — the
old figure is the cost of computing the wrong answer, and there was no correct measured branch to
compare against. End to end both engines: **PASS, 0 failures over 161 stocks.**

⚠ **`FilmGrainSigma` left `LAW_BYPASS_BASELINE`, and the reachability check had to grow a concept to
allow it honestly.** The stage reaches the law through a hoisted evaluator, so a symbol grep still
finds nothing; declaring that "reached" unilaterally would be the convenient reinterpretation that
dict exists to forbid. The indirection is now declared in `LAW_EQUIVALENT_IMPL` **and licensed by a
numeric assertion** — both spellings evaluated in the same compiled program on the same rows, worst
**2.40e-07**.

---

## What changed on 2026-08-30, second batch

**Four approved, three landed, one halted.** Detail: `RESULT_2026-08-30b_K1_F2_queue.md`.

| item | outcome |
|---|---|
| **K1** ✅ | Four profiles — PORTRA **160NC / 160VC / 400NC / 400VC** from E-190 (May 2003) pp 9-12. Database **161 → 165**, `film_names.txt` md5 `696c4c26…`, ⚠ **zero existing indices moved** (verified by simulating the sort first). ⚠ The row was wrong FOUR times, every error optimistic: five films not four, "renumbers" when it does not, "all traced" when half was not, and an unrecorded page mapping |
| **queue reconcile** ✅ | Three counts, none maintained by the others (26 / 28 / 34). ⚠ The category table **omitted eight real rows**, and **two closed rows were never struck** so every parser counted them live. Dashboard now derived from the parse, with the rule written down |
| **F2** ✅ | Reversal default reversed (0.5 → **2.97**, it was backwards); colour-negative default 1.2 → **0.68**. ⚠ **Monochrome negatives deliberately left wrong-facing** — all 11 measurements are Kodak colour cine and no B&W negative shape exists in this corpus (**F2b**) |
| **C41** ✅ | **DONE later the same day, owner chose option A.** Halted first (see below), then wired once the header had been read properly: `AlgoSolveAnchors` gains `scannerSpecular` and applies the factor at `film_sim`'s own two points; new pointwise **stage 12b** in place between 12 and 13; both twins moved together. ⚠ **The header named THREE consumers and two do not exist** — `AlgoNeutralMidDensity` applies nothing in Python and wiring it would have CREATED a divergence, and Callier is its own stage rather than part of dye impurity. Corrected. ⚠ And the one measurement the argument rests on is recorded twice, differently (+54/255 vs +48/255) — no exact figure is quoted anywhere now |

⚠ **I OVERSTATED F2 WHEN I PROPOSED IT** and the correction belongs here, not only in the RESULT.
I called it "the largest render-quality item left: 146 stocks carry a shape every measurement
contradicts". The shape part is true; the **render** part is not. The wiring honours a shape only
when `sigma_shape_measured` is set, the heuristic never sets it, and that has been true since
2026-08-18. **F2 changed no frame.** It made a placeholder truthful.

⚠ **A guard caught me overreaching inside F2 and was right.** The first attempt wrote the measured
interior peak (1.38 at 0.75) into the heuristic; `verify.py` failed it on all 55 affected stocks,
because `sigma_anchors()` returns None for an unmeasured stock and `grain_sigma` never sees a peak.
It would have stored a number the data model cannot honour.

⛔ **WHY C41 STOPPED FIRST, AND WHAT READING IT PROPERLY CHANGED.** The stop was right and the
reason survives; what it bought was the discovery in the row above. Reading the two sides line by
line showed the job was **two** consumers, not three, and the insertions were one line each — so the
halt cost a conversation and saved two invented call sites. Callier has two halves and they cannot ship separately. The pixel pass is
easy — Python runs it as stage 12b, in place, and the C++ planes are already there. The **anchor
solve** is not: `film_sim` applies the same factor at two points inside it, and the call site
records why — *"mid grey moved +48/255"* on Double-X at specular = 1 when the solve was left blind.
Wiring only the pixel pass reproduces that regression at a fifth of the output range. Half of C41 is
worse than none of it, and I would not start a numerical solve I had not read line by line at the
end of a long batch.

**C41 guard, and it is the point of the task:** `cpp_parity` gains a Callier **STAGE** family that
drives `AlgoStage12b_Callier` and `AlgoSolveAnchors` themselves rather than the law beside them —
2475 probes, worst stage **3.83e-07**, worst solve **2.77e-07**, inert at specular 0 as an
*identity*, **0 colour stocks moved**, **272 monochrome rows moving** at full specular. ⚠ The
existing law family had passed for a week while nothing called either function; the SOLVE assertion
is the load-bearing one, because a pixel pass without it moves mid grey by more than it changes
contrast and a stage-only probe would pass on exactly that.

⚠ **Still not right, and it is the data half:** the film side of the product (1.3 negative, 1.25
reversal) is a generator rule with no document behind it. That is why the control still ships at
zero. C41 made the mechanism correct, not the number.

**Verified:** build **0 failures / 0 warnings**; `verify.py` **424 PASS / 1 FAIL** (baseline);
ordering identical across all four representations at 165; `film_ids.lock` first 161 entries
unchanged; both engines end to end — **scalar PASS 0 failures over every stock**, **AVX2 PASS 0
failures over every stock**. *(The figure read "165 stocks" until 2026-09-02d; it is the whole
database at whatever size it is, so the count is no longer written into the sentence.)*

---

## Build state, right now

| | |
|---|---|
| `verify.py` | **422 PASS / 1 FAIL** (re-run 2026-08-29, second session). The one failure is the saturation-hierarchy ordering the owner said to leave alone; `build.py` compares the FAIL *set* against a baseline, so a NEW failure fails the build. History: 391 → 409 with the six relational guards of 2026-08-27 (G-MTF, G-DQE, G-LAT, G-PROV, G-PROC, G-PROGRESS), 409 → 413 with **G-YELLOW** and **G-DEVFAM** (2 checks each, all fault-injection tested), 413 → 419 with **G-FILMID** (6 checks pinning the frozen identifier block, including a SHA-256 digest of the 161 pre-freeze rows — added because fault injection showed a swapped pair was otherwise undetectable, the database re-sorting from the lock and making the corruption self-consistent), and 419 → 420 on 2026-08-29 when the `ProcessingFamily` census guard moved 22 points/4 stocks → **42/8** rather than being loosened |
| `build.py --root <corpus>` | ✅ **0 failures / 0 warnings, 2026-08-30 — and this one is real.** Of **20 registered audits**, **15 pass, 5 skip** for sources not staged in this checkout, **0 fail**. ⚠ Unlike the 2026-08-29 claim that had to be withdrawn, this run used a root **carrying the algorithm tree**, so `cpp_parity`'s twin-consistency, law-reachability, grain-stage and Callier probes all executed. The Callier family runs here for the first time (11592 probes, worst 1.43e-07) because the owner supplied `AlgoCallier.hpp` on 2026-08-29. Still skipping: 5 audits whose PDFs are not staged. **Always check the SKIP count and the ROOT, not just the failure count.** |
| ⚠ **THE BUILD ROOT, AND WHY THE SKIP COUNT MATTERS AS MUCH AS THE FAILURE COUNT** | Discovered 2026-08-27, and it bit again in a different way on 2026-08-29. `build.py` derives every audit path from `--root`. A root without `PDF/` makes audits report `[SKIP] source not present` while the build still says "OK, 0 failures" — **a skipped audit and a passing audit look the same in the summary line.** ⚠ On 2026-08-29 two audits had been reported as FAILING for weeks — `vision3_granularity.py` and `kodak_still_curves.py` — and BOTH turned out to be missing PDFs, not defects: their guard file differs from the file they actually open, so they fail instead of skipping. Proven by re-running them against the pristine `1_python.zip`, where they fail identically. Staging the sources fixed both. ⚠ **AND THEN THE SAME TRAP CAUGHT THE CLAIM ITSELF, HOURS LATER.** That run reported "0 failures / 0 warnings with 18 audits registered and 11 running — the first fully green build in this project's recorded history". It was run against a root with **no algorithm sources**, where `cpp_parity.py` skips five of its probes and exits 0. With the tree present the build is **20 registered, 14 pass, 5 skip, 1 fail**, and the failure is the C30/C33 one this file had already recorded on 2026-08-27. **The "first fully green build" claim is withdrawn.** Anyone re-running must check the SKIP count and the ROOT, not just the failure count |
| ✅ `cpp_parity.py` — **GREEN 2026-08-30, with the algorithm tree present, which it never was before** | *"the Python and C++ grain, MTF, reciprocity and Callier laws agree on the whole database"* — and this time the grain-stage, twin-consistency, law-reachability and Callier probes all actually ran. The three sub-failures below are **fixed**, not skipped: queue C30/C33 closed 2026-08-30. Worst rendered-amplitude disagreement **2.52e-07** against a 2e-5 tolerance; `\|amp − 1\|` at NET density 1.0 **exactly zero**. ⚠ The history under this row is kept because the lesson is worth more than the pass. | The previous state of this row said *"now GREEN — the Python and C++ grain, MTF, reciprocity and Callier laws agree on the whole database"*. ⚠ **Withdrawn 2026-08-29, second session.** That run used a root with no algorithm sources, where the audit prints `[SKIP] law reachability`, `[SKIP] twin consistency`, `[SKIP] grain stage`, `[SKIP] reciprocity`, `[SKIP] Callier` and exits 0. Against a root that carries the tree, the **same three sub-failures recorded on 2026-08-27 are still there, unchanged**: `ampScale` absent from `Algo_11_Sim.cpp` and its AVX2 twin, rendered grain amplitude off by **1.83e-01** at `('S', ILFORD_HPS, 0, 3)` against a 2e-05 tolerance, and the grain stage not returning 1.0 at NET density 1.0. The number is exactly `sqrt(1 + fog_grain) - 1` — `AlgoAddGrain` computes `sqrt(max(D - dmin, 0) + fog)` with no net-1.0 normalisation, so **`rms_granularity` does not mean the printed figure in the C++ render**. That is queue **C30/C33**, the recorded `FilmGrainSigma` bypass. Re-confirmed today by running the pristine `1_python.zip` against the same tree: identical failures. ⚠ The lesson is now recorded twice on this page and it is the same one: **a skipped audit and a passing audit look the same in the summary line** |
| C++ compile | clean on **18 TUs**, `g++ -std=c++14 -Wall -Wextra`, gated on exit 0 **and** zero bytes of output. ⚠ That gate covers the GENERATED database only. The plugin's own stage TUs are compiled by hand; on 2026-08-23 the scalar set and the **AVX2 set (now that `FastAriphmeticsAVX.hpp` is on disk)** both build clean at `AlgoType = double` and `= float` respectively. Pre-existing warnings NOT introduced by this session and NOT touched without approval: six unused locals in `AVX2/Algo_08_Sim.cpp` (`dm`/`gm`/`tx`/`tk`/`sx`/`sk`, ~line 1407) left dead when that path moved to `AlgoCurveLut` |
| `film_names.txt` | MD5 **`696c4c26c0df83359e80f75850c2d215`**, **165 lines** — moved 2026-08-30 (queue K1: PORTRA 160NC/160VC/400NC/400VC). ⚠ **NO EXISTING LISTBOX INDEX MOVED**: the four names are absent from `film_ids.lock`, so they sort with `_UNFROZEN` and append at 161-164, which was verified by simulating the sort before the profiles were written. Previous state, still true of everything before line 161: MD5 `41e0bc5d2c7db82324529e773f2fd5ee`, 161 lines — re-measured 2026-08-27; **unchanged by the v16/v17/v18 work**, which added fields and values but no stock, so the plugin's ListBox indices did not move again. Earlier history, still true: MD5 was `2de4536b80602d38bea4a48e46533df1` at 160 lines and ⚠ CHANGED TWICE on 2026-08-24. First `e8dc2cb9…` → `c2b9e17e…` (queue C26: `EASTMAN_TRI_X_5223` inserted at line 31, `KODAK_8374` at line 71). Then `c2b9e17e…` → `2de4536b…` (queue C27: `FUJI F125 8630` **deleted** from line 41). Net effect on the plugin's ListBox: indices 30 and below unchanged, 31–39 +1, 40 onward net 0 for one stock's width then +1 again — **do not hand-patch; regenerate**. The plugin's ListBox indices move with it |
| Audit scripts | **28 registered** (re-counted 2026-08-31 from `build.py`'s `audits()`), **23 running** in this checkout and **all 23 green**; **5 SKIP** for sources that are on the owner's machine and not staged here — `mees_granularity`, `granularity_vector`, `agfa_vista`, `gevaert_curves`, `di_2254`. ⚠ **A SKIPPED AUDIT AND A PASSING AUDIT LOOK THE SAME IN A SUMMARY LINE**, which is why the two numbers are printed separately. Added since the last recount: `kodak_aim_density.py` (16 published aim-density tables off 13 sheets, five of the twenty-one checks cross-document), `polaroid_spectral.py` (four Polaroid panels, two of them new data), and `konica_raster.py` (2026-08-31, queue E3 — the first audit reading sheets that are RASTER end to end, seven panels calibrated geometrically off the printed grid). Earlier additions: `spectral_mono_parity.py`, `spectral_weight_provenance.py`, `kodak_1952_curves.py`, `agfa_2004_curves.py`. ⚠ `spectral_vector.py` now registers **19 sheets**, up from 4 on 2026-08-29: the cross-checks a 2026-08-26 sweep had run BY HAND lived only in a docstring and were pinned by nothing |
| Database | **175 film stocks, 11 print stocks, 14 gauges**, schema **v24** (re-measured from the live module 2026-09-02: `SCHEMA_VERSION`, `len(FILM_PROFILES)`, `len(PRINT_STOCKS)`, `len(FORMATS)`). 134 negative / 41 reversal; 69 monochrome; provenance tiers 88 T1 / 45 T2 / 42 T3. ⚠ **170 → 171 on 2026-09-02 (queue C4): `SVEMA_CO_90L`, ТУ 6-42-1514-90, appended at id 170 with no existing index moved.** ⚠ **AND THE QUEUE ROW THAT ASKED FOR IT NAMED A FILM THAT DOES NOT EXIST** — «ЦО-90Д» is an OCR misread of Л as Д on a typewritten page, and both files in the corpus are scans of the SAME ТУ for ЦО-90Л. One stock, not two. ⚠ **166 → 170 on 2026-09-01**, owner-approved: AGFA_ULTRA_50 and AGFACHROME RSX II 50/100/200, from the 1998 edition of Agfa's «Technical Data PF» — a file `NotFound.md` row 5 and queue G6 both recorded as a duplicate of the 2004 F-PF-E4 sheet and which is a separate edition five years older, 100 % vector, and the only document in the corpus that plots those four films. All four appended at ids 166–169, no existing index moved. ⚠ **THE SPLIT AND THE TIER LINE WERE THEMSELVES STALE BEFORE THIS EDIT** — they read 129/36 and 80/45/40 against a live 130/36 and 80/45/41, because `doc_consistency.py` registered the stock count in this sentence and not the two beside it. All three are registered now. ⚠ **`SCHEMA_VERSION` READ 18 UNTIL 2026-08-31 AND WAS FOUR VERSIONS STALE.** v19 (dye-density neutral traces and the measured-spectra `dye_matrix`), v20 (taking-filter fields, queue C39), v21 (`AimDensity` and `ProcessVariant.push_stops`, queues K2/K3) and v22 (`PrintStock.spectral`, queue M1) all landed on 2026-08-30 and 2026-08-31 with `# -- schema vNN` comments on their fields while the constant sat at 18 — and `doc_consistency.py` guards the two COUNTS in the headline sentence and not the version beside them. All four are additive and inert: a v22 database renders bit-identically to a v18 one and no film index moves. Earlier, still true: 161 → 165 on 2026-08-30 (queue K1, the four PORTRA NC/VC stocks, appended at 161–164 with no existing index moved) ⚠ **SECOND AGFA PASS, same day:** «Technical Data P-16-C» (`agfa_film_chem.pdf`, 08/1999) — the processing companion `agfa_films.pdf` p11 names in its closing line, already in the corpus under a filename that hid it. 64 printed developing-time cells SUPERSEDE the gamma-time panel digitised earlier the same day: same physics as text, six developers instead of five, including ATOMAL FF which no Agfa panel plots. The trace and the table agree where they overlap (RODINAL 1+25, small tank, γ 0.65 = 6 / 8 / 7 min in both), which is what makes replacing a measurement with a citation safe. It also supplied the first documented `PushSpec` on the AGFAPAN line (+1 stop, printed times) and named the characteristic curves' development condition as Agfa's γ 0.75 aim. |
| Measured data now LIVE in the render | σ(D) shape (**13** stocks), **MTF f50 + rolloff (12 measured — 8 from sine-wave panels, 2 Coltman-converted from square-wave CTFs, 1 reversal added 2026-08-25g — +2 mixed-provenance, +5 red-re-anchored)**, interimage (all), per-layer rms (**11**), **reciprocity (21 measured tables + 105 exponents)**, **characteristic curves traced from the sheet (+2 stocks 2026-08-23: the two Fuji Super-F negatives; +1 print stock 2026-08-25d: the 2254 DI film, from a RASTER figure)**, spectral sensitivity (**76 sets** — re-counted live 2026-08-29 second session; the 73 recorded here was stale). ⚠ **And "live in the render" now means something stronger for the monochrome half than it did**: since 2026-08-29 the 24 mono stocks carrying a traced pan curve take their RGB collapse weights from that curve in **both** engines, where previously only the C++ plugin did |
| ⚠ **The two rows below this one are NOT re-derived** | Their per-stock counts date from before schema v16 and were **not** re-checked on 2026-08-27, so treat them as a floor, not a census. Naive queries against the live module disagree with several of them — 30 stocks with a non-empty `reciprocity_table.times_s` against the row's "21 measured tables", 65 with an explicit per-layer rms against its "11", 48 with a non-empty `spectral.log_s_g` against its "73 sets" — but those queries use *my* definition of "measured", not the row's, and the difference is definitional, not necessarily an error. ⚠ **DO NOT copy either set of numbers forward as fact.** Re-derive them from `FILM_PROFILES` with the same predicate the row's author used, or replace the row with generated cells (the standing method rule: every count in a document should be computed in the same expression that fills the cells beside it) |
| Carried but INERT — nothing on the render path reads it | spectral dye density (**12** stocks + 1 print stock), `layer_stack`, `processing_family`, `reciprocity_table`, `aging`, and since v12 `dye_stability` (**1** stock, the only Arrhenius table in the corpus) |

⚠ **Owner action outstanding: REBUILD THE PLUGIN.** Three reasons; the second is the dangerous one.
1. Schema **v8** added five `GrainSpec` fields, so the struct layout moved.
2. Schema **v9 changed a MEANING with no change of layout**: `rms_granularity` is now the rms at
   **NET** density 1.0 — `dmin + 1.0`, the convention Kodak prints on 5248 p1 and 5222 p1 — and
   `FilmGrainSigma()` normalises there. v9 data paired with a v8 sampler compiles clean, runs clean
   and renders the wrong grain level. ⚠ **If the C++ grain path calls `FilmGrainSigma()`, re-read its
   calling-convention comment: the v8 instruction to multiply by your own `sqrt(D − dmin + fog)` at
   D = 1.0 is now WRONG and double-counts.**
3. Schema **v10** added `MTFSpec.mtf_rolloff_q` + `mtf_measured` (layout moved again) and a second
   mirrored law, `FilmMtfResponse()`. Optional to call; if you do, multiply nothing extra — both laws
   are exactly 0.5 at f50 by construction.
4. ⚠ **NEW 2026-08-23, and it is a COMPILE-BREAKING signature change on purpose:**
   `AlgoStage08_CharacteristicCurve` takes a fifth trailing argument,
   `const HighPrecType logEShift[3]` — the reciprocity shift from the new
   `AlgoReciprocity.hpp`. The declaration, both definitions (scalar and AVX2) and both call
   sites in the tree are updated; any other caller in your working copy will fail to compile,
   which is intended rather than silently ignoring the shift. `AlgoControls` also gains
   `exposureTimeS` (default 0 = inert) — every caller uses `getAlgoControlsDefault()`, so no
   positional initialiser breaks. ⚠ **The AVX2 caveat printed here on 2026-08-23 is retired:**
   the owner supplied `FastAriphmeticsAVX.hpp` and `FastAriphmetics.hpp`, and the AVX2 TUs now
   compile clean at `AlgoType = float`, so that edit is verified by the compiler rather than by
   inspection.
19. ⚠ **NEW 2026-08-27 (tasks #77-#81) — TWO NEW GUARDS, BOTH FAULT-INJECTION TESTED, AND ONE OF
   THEM CAUGHT A DEFECT BEFORE IT SHIPPED.**
   **G-YELLOW** (2 checks) pins `AgingSpec.base_yellowing_d` to zero on all 161 films **and all 11
   print stocks** — the print stocks were not in the §26 B7 item as written and have carried their
   own `AgingSpec` since v12. The evidence is an ABSENCE: neither preservation source describes any
   base yellowing except **nitrate**, and for nitrate only ordinally, as a stage with no density
   attached. The audit found the database already clean at 0 of 161. ⚠ **That was luck, not design,
   and the guard converts it into design.** It deliberately does NOT say "acetate" — there is no
   base-material field to select on (§26 B8) — so it asserts zero everywhere, which is the stronger
   claim and matches the evidence exactly.
   **G-DEVFAM** (2 checks). (a) No development family may mix reversal and negative stocks. (b)
   Within a family of ≥ 3, `mid_slope` must not spread more than 2.0×.
   ⚠ **THE 2.0× IS DERIVED, NOT CHOSEN.** Measured spreads inside the real families: ECN-2 n=15 →
   **1.12×**, ID-11 n=6 → **1.30×**, D-96 n=2 → **1.03×**. The widest genuine family is 1.30×; the
   cut sits above every real one and below the single false one, which measured 4.33×.
   ⚠ **THE FAMILY KEY IS NORMALISED, AND IT HAS TO BE.** The database holds both `"ID-11"` and
   `"ILFORD ID-11"` for the same Ilford developer, entered by different people from different
   sheets. Unnormalised they form two groups of 2 and 4, both under the n ≥ 3 floor, and a genuine
   outlier could hide in the split.
   **Fault injection, because a guard that cannot fail is documentation wearing a test's clothes:**
   moving a reversal stock into the ECN-2 family trips check (a) and drives the spread to 4.32×;
   multiplying one ECN-2 stock's gamma by 4 trips check (b) at 3.99×; setting any
   `base_yellowing_d` non-zero trips G-YELLOW. All three verified.

18. ⚠ **NEW 2026-08-27 (task EM-A7) — developer identity mined, 13 → 29 stocks, and THIRTEEN OF
   TWENTY-NINE CANDIDATES WERE FALSE POSITIVES.** Development progress type is a property of the
   DEVELOPER, not the emulsion — Tani gets both types out of one emulsion with CP-20 and D72 — so
   `DevelopmentProgress` could not reach past 9 stocks while 148 profiles recorded no developer.
   A keyword sweep of the 74-PDF corpus proposed 29 identities. **Sixteen were accepted, each
   quoted from the stock's OWN datasheet.**
   ⚠ **THE REJECTIONS ARE THE INTERESTING PART, BECAUSE THEY LOOKED RIGHT.** A general film-
   restoration monograph on disk mentions "ECN-2" twice, and matching on it would have assigned
   **ECN-2 to AGFACOLOR NEU 1936, DUFAYCOLOR 1937, SOVIET PANCHROM 1939 and EASTMANCOLOR 5248
   (1953)** — every one decades before ECN-2 existed (1974). Two more assigned Portra's C-41 sheet
   to the ULTRA COLOR stocks, which are C-41 films but are not that document.
   **Rule adopted: a process is accepted ONLY from the stock's own datasheet. A mention in a third
   document is not evidence about this film.**
   ⚠ **ONE FALSE POSITIVE SURVIVED THE RULE AND WAS CAUGHT BY G-DEVFAM.** `EASTMAN_5294_1983` was
   matched to `KODAK-EKTACHROME-100D-5294-7294-technical-information.pdf` — its **own** sheet by
   product number — and would have been written as "Process E-6". It is a colour NEGATIVE (Eastman
   High Speed 5294, 400T, 1983) and **Kodak reused the number 5294 much later for an EKTACHROME
   reversal film**. A four-digit product code is not a unique key across decades. Removed, not
   exempted; its real process is almost certainly ECN-2 but no sheet for it is in this corpus, so it
   stays UNSET rather than inferred.
   ⚠ **Six of the sixteen are tier 2 on purpose.** Their sheets have a PROCESSING section and name
   the process on the page, but the two-column text layer interleaves and the instruction sentence
   cannot be quoted intact. The identification is sound; the QUOTATION is not. Rendering those six
   pages would upgrade them (EM-A7b).

17. ⚠ **NEW 2026-08-27 (task EM-A6) — per-parameter provenance went 52 → 1511 entries, and 197
   DOCUMENT CELLS TURNED OUT TO BE LYING.** The hand audit that motivated `ParamSource` found 22.
   Covering all nine parameters `FilmActiveProfiles.md` prints, across all 161 profiles, found
   **197 cells that read as "documented" and were not**, plus 6 that were backed and read as
   unbacked. ⚠ **THE NUMBER OF UNDOCUMENTED CELLS WENT UP, AND THAT IS THE FIX WORKING.** The
   database did not get worse; it stopped overstating itself.
   **Census, re-counted 2026-09-01: 863 estimated, 302 assumed, 225 traced, 27 derived, 65 stated, 23 measured. ⚠ 1463 → 1470 and 161 → 162 profiles when KODAK_EKTAR_125 was created on 2026-08-31 with seven hand-written records — its blue D-min is the 11th MEASURED entry in the whole file. ⚠ 1470 → 1498 and 162 → 166 profiles on 2026-09-01, the AGFA harvest: 36 hand-written records across twelve stocks, of which twelve DISPLACE a derived entry rather than adding one. That is the first use of the precedence rule for its stated purpose — `_PARAM_SOURCES_DERIVED` had given every Agfa stock a `grain.rms_granularity` cell reading «No published rms for this stock in the corpus», and Agfa print the figure beside every plotted column of a sheet each profile's own provenance already named. ⚠ 1498 → 1505 later the same day, the SECOND AGFA pass: the F-PF-D4 German twin and the SCALA F-SW12-E6 sheet. Seven more hand-written records, and three of them are REFUSALS with evidence -- the two AGFAPAN f50 cells stay estimated because Agfa printed ONE sharpness curve under both APX columns, and RSX II 200's push spec was corrected after the 1998 sheet's push table turned out to belong to the SCALA column two columns to its right. ⚠ 1505 → 1509 on 2026-09-01, the US 4,495,277 pass: `emulsion.sensitization` = "S" on the four Agfa B&W stocks, tier 3 / `assumed`, stored at the owner's instruction as an inference from an ASSIGNEE AND A DATE and labelled as one in every note. Nothing numeric was taken from that patent -- it names no product, permits every crystal habit and a 0.3-2 um size range, and its worked example is a COLOUR coating with a yellow coupler. EmulsionSpec is inert, so the four entries cannot move a pixel. ⚠ 1509 → 1511 on 2026-09-01d, the Flueckiger et al. 2018 pass: `dye_density` on TECHNICOLOR_THREE_STRIP -- the FIRST measurement that profile has ever carried, it having been one of the nine stocks with no source of any kind -- and `reseau.filter_matrix` on DUFAYCOLOR_1937, which records a SECOND measurement that disagrees with the stored one rather than replacing it (queue C46).**
   ⚠ **AN 'estimated' OR 'assumed' ENTRY IS A RESULT, NOT A BLANK.** It is the statement "we looked,
   and this number is ours" — precisely what was missing.
   **Precedence is ordered and load-bearing:** hand-written `_PARAM_SOURCES` beats mined
   `_PARAM_SOURCES_DEVELOPER` beats derived `_PARAM_SOURCES_DERIVED`. ⚠ **This order was wrong once
   and the symptom was SILENT** — with the derived merge first, its blanket "no developer recorded"
   entry claimed `processing.developer` on all 161 profiles and quietly discarded all 16 mined
   identities, each carrying a real quotation. The developer count rose and the provenance did not.
   **A new status, `"stated"`, was added to `_PARAM_STATUS`** for a fact a source prints IN WORDS —
   a developer name, a process name. "measured" was defined as a number and would have been a lie;
   "estimated" would have been a worse one. `validate` demands a source for it, like the other
   evidence-bearing statuses.
   ⚠ **CODEGEN HIT ITS OWN CAPACITY STOP AND THE STOP WAS RIGHT.** Slot 01 reached **120 961 bytes
   against a 112 000 ceiling**; `N_DATA_SLOTS` was NOT raised, because that forces new
   `film_profiles_data_NN.cpp` files into the Visual Studio project by hand. The real problem was
   that **642 KB of provenance text, 324 KB of it prose notes, was heading into a runtime database
   that never reads it** — a grep of the whole C++ tree finds `param_sources` only in the generated
   header that declares it. The `note` field is now emitted EMPTY while the struct keeps it, so the
   aggregate-initialiser layout does not move, `SCHEMA_VERSION` stays 18 and no plugin rebuild is
   forced. Classification (param, tier, status, unit, conditions, source, confidence) all survives.
   ⚠ **Slot 01 now sits at 108 515 of 112 000 — a 3 % margin.** The next material addition to the
   generated database will need `N_DATA_SLOTS` raised, and that is an owner decision because of the
   .vcxproj step.

16. ⚠ **NEW 2026-08-27 (task A5) — the 1920 American market harvested, and OCR was measured at
   67 % on the column that mattered.** Davis & Walters Part III is 86 pages, one anonymous emulsion
   each, with a three-field data block: B.S. speed, scale, resolution number. The pages are rotated
   90°, so the harvest went through `pdftotext -bbox` word boxes clustered into rotated lines. That
   worked, and **the result was wrong on a third of the speeds.**
   ⚠ **THE OCR FAILURE MODE HERE IS THE DANGEROUS ONE: SILENT LEADING-DIGIT DROPS.** 330 → 30,
   560 → 20, 650 → 10, 175 → 75, 355 → 55, 280 → 80. Not garbage — plausible values that pass any
   range or sanity check. Measured against 300 dpi renders of all 86 pages: speed **67.4 %**
   correct, resolution 79.7 %, scale 98.8 %. **All three numeric fields in the shipped dataset are
   image-read, not OCR-read.** The knowledge base's existing warning that this scan's digits are its
   least reliable content was qualitative; it is now measured, and it was an understatement.
   ⚠ **A UNIT TRAP THAT INVERTS THE RESULT.** The "resolution number" is **not** lines/mm — the
   paper defines it as the line-centre separation in units of 0.001 mm, so **smaller is better**. A
   harvest that captured only the number would have concluded the opposite of the truth.
   **What it gives us:** 86 records (73 plates, 13 films), speed spread **300 : 1**, and the
   speed/sharpness exchange rate of 1920 material regressed for the first time — resolving power
   ∝ speed^**−0.14**, r = +0.735 over 79 emulsions, and it **survives within class** (ortho r =
   +0.529, ordinary r = +0.360), which is what rules out class composition as the whole story. The
   paper states the rule qualitatively and adds "but this is not strictly true"; this is that caveat
   made quantitative. Median resolution number 20 → f50 ≈ 25 c/mm against our fleet median of 50.0
   (measured-only 54.8) — ⚠ **with a factor-of-two convention ambiguity that the paper does not
   resolve**, so the direction is usable and the level is not.
   **No database change.** The emulsions are anonymous by the paper's design; our oldest stock is
   1930. The 258 plotted curve families on those same pages are **recommended into Category C** for
   the same reason. `doc/thirdparty/davis_walters_1920_survey.txt`; write-up in §23d.
   ⚠ **§28.1 of the knowledge base is now fully resolved — 9 of 9 rows**: six traced or harvested,
   one superseded, two struck as having no route into the renderer.

15. ⚠ **NEW 2026-08-27 (task A4a) — Wall's coating-weight table verified, and OUR OWN derived table
   was wrong.** The Tappen & Rekaschow analyses needed a page render and a folio confirmation.
   Neither turned out to need a render: the text layer carries the table intact, PDF 164's running
   head reads `PREPARATION FOR COATING 157` so the page is confirmed by the book's own folio, and
   **C + D = 100.000 on all ten rows** — an identity the column key requires and a single flipped
   digit would break, which proves those two columns undamaged.
   ⚠ **THE ERROR WAS OURS, NOT THE SOURCE'S.** Our derived areal table applied C and D — which are
   percentages of the **anhydrous** emulsion — directly to A, which is the coating weight
   **including** its residual water. Every mass in it was ~9 % high and the three of them summed to
   more than the coating weight. Corrected: silver halide 18.85 g/m², gelatine 25.64, water 4.09,
   summing to 48.58 exactly; metallic silver equivalent **10.83 g Ag/m²**, not 11.9.
   ⚠ **A second book error, larger than the one already recorded.** §3.3.1 knew Wall's mean divides
   a sum of nine by ten. His *composition* is worse: because C and D are complementary in every row,
   their means must sum to 100 for any subset and any divisor — and his headline "42.6 % halide /
   50 % gelatine" sums to **92.6**. No selection and no divisor produces it. True pair 42.363 /
   57.637. The one headline figure that is right is the ratio, 1.3 : 1 — **because a ratio is immune
   to the divisor error**, which is a fair warning about how little one consistent-looking number
   proves.
   **What it buys:** the first layer thickness the corpus can support — **21.9 µm dry, 26.0 µm as
   coated, 13.3 % silver-halide volume fraction** — which lands inside Duffin's stated 2–40 µm dry
   envelope near its thick end, exactly where a 1929 plate should sit. That closes a gap §3.3.1
   itself recorded as unclosable ("the two cannot be compared"). ⚠ The densities are assumed, not
   sourced; the value is the order of magnitude, not the third digit. **No database change** — per-
   stock coated thickness is Category C. Write-up in §3.3.1a.

14. ⚠ **NEW 2026-08-27 (task A3) — Tani Fig. 1.1 traced, and it cross-checks the DQE guard.**
   The sensitivity-versus-year curve, ~150 years on one log axis, traced from a 600 dpi render of
   PDF page 13 by column-wise dark-run tracking; 2246 columns, both axes calibrated by least
   squares on the printed ticks (residuals 0.17 yr and 0.029 decade). **The curve is piecewise
   linear and a 7-knot fit recovers it to 0.017 decade RMS**, which is *below* the calibration
   systematic — so it is a straight-segment construction, and the knots are the technology dates.
   **Three plateaux at ×1.2 per decade, separated by two steps**: ×56 in 4.5 years at 1847–51.5
   (development) and ×51 in 8.5 years at 1871–79.5 (gelatin, then spectral sensitization). The
   modern era **decelerates**: +0.371 decade/10 yr for 1912–35, +0.224 for 1935–91.
   ⚠ **THE ORDINATE IS NOT A SPEED SOURCE.** The caption says only "photographic materials" — it
   never states whether it plots the fastest material of the year or a typical one, never states a
   measuring standard (which cannot have been ISO before 1974), and its pre-1880 values are
   necessarily reconstruction, since nobody measured ISO 0.0013 in 1840. **No profile may take an
   `exposure_index` from it.** Recorded [Q] in shape, [F] at best in level.
   **What it bought us: an external check the DQE guard did not have.** Our fastest *conventional*
   stock per decade (instant materials excluded) tracks the traced curve to a median factor of
   2.03 in level and +0.30 vs +0.25 decade/10 yr in slope — and the factor of 2 is explained, not
   fudged: our `exposure_index` is the rated speed, and the stocks topping each decade are the
   push-rated ones whose true ISO is about half their name. ⚠ **Nothing was adjusted to make this
   agree.** The substantive finding is the *divergence*: our colour-negative K ladder rises ×12.2
   after 1990 while Tani's curve rises only ×2, and our own component medians say why — median EI
   ×2.0, median mid slope flat, **median rms 12.0 → 4.6**, worth ×6.8 through the 1/rms² term.
   That is the tabular-grain transition, and it means the two measures are answering different
   questions after 1990: Fig. 1.1 asks how fast the fastest material got, K asks how clean a given
   speed got. **No database change follows.** Written up in `EMULSION_KNOWLEDGE_BASE.md` §23c;
   raw trace in `doc/thirdparty/tani_fig1_1_raw_px.txt`.
   ⚠ **A stale K ladder was found and corrected while doing this.** `verify.py`'s guard comment and
   this file both printed 0.048 / 0.241 / 2.727. Those were right when written and wrong within the
   same day: the v17 third-party rms imports raised `rms_granularity` on six colour negatives, and
   rms is a denominator in K. Live values are **0.045 / 0.219 / 2.676**. A hardcoded median in a
   comment goes stale the moment the data moves, and nothing was comparing them.

13. ⚠ **NEW 2026-08-27 — the knowledge base's PDF page citations were audited, and the offset map
   it was built on was wrong in two of four books.** §0.2 of `EMULSION_KNOWLEDGE_BASE.md` promises
   every claim carries both the printed page and the PDF page. The offsets were recorded from spot
   checks, with change-over points stated as approximations ("up to ~p.55", "from ~p.80"), which is
   not good enough to validate a citation against. Every page of all four books was run through
   `pdftotext -layout`, the printed folio read from the running head or footer, and the offset
   computed **per page**.
   **What the measurement found:** Tani is +9 uniform across all 264 pages, and Davis & Walters is
   +1 uniform — neither has a boundary. **Duffin has two regimes** (+4 to p.71, +6 from p.73; two
   unnumbered plate leaves and a blank are bound in at PDF 76–78). **Wall has three** (+8 to p.65,
   +7 for pp.70–252, +6 from p.254) because it is missing two pages, not one.
   **2242 citation pairs checked; 9 occurrences in 6 distinct pairs were wrong** — all corrected,
   all landing within one page of the right place, none pointing at unrelated content.
   ⚠ **SIX OF THE NINE WERE PAGE RANGES THAT WERE RIGHT AT THE LOW END AND WRONG AT THE HIGH END**,
   because the range was extended by counting rather than by reading the last page's folio.
   **Validate a range citation at BOTH ends.**
   ⚠ **An automated folio check on Davis & Walters produces a false Δ ≈ +35 on ~86 pages** — its
   plate captions carry their own numbers, which are not folios. Recorded in §29.1 so the next
   person to run this does not "fix" 300 correct citations.
   **Two knowledge-base entries corrected in the same pass.** (a) **Wall p.253 is not an OCR
   failure — the page is not in the scan**, so it moved from "digitise later" to *permanently
   unavailable from this copy*. The proof is the scan's own annotation: PDF 259 ends with the
   literal text `(Table on 253 missing)`, PDF 260 opens on folio 254, and the offset drops +7 → +6
   at exactly that point, which is the arithmetic signature of one absent leaf. Nothing in the
   knowledge base depends on it. (b) The **Reilly prediction-tables row was struck from §28.1's
   digitisation worklist**, because §27 already places the whole IPI storage-prediction apparatus
   in Category C on the grounds that there is no path from those numbers to a pixel — a figure with
   no route into the renderer does not belong on a worklist however well quantified it is.

12. ⚠ **NEW 2026-08-27 — schema v18: per-parameter provenance, process variants,
   development progress type, and FIVE RELATIONAL GUARDS.** The three items the colorist's brief
   and the emulsion assessment both converged on. All additive and inert; a v18 database renders
   bit-identically to v17.
   **① `ParamSource` — provenance attached to the PARAMETER, not the profile.** 52 entries across
   26 profiles: dotted path, tier, status (`measured` / `traced` / `derived` / `spec_limit` /
   `estimated` / `assumed`), unit, measurement conditions, confidence, note. ⚠ **Every `param`
   path is validated against the live object**, so provenance cannot drift from the schema the way
   a comment can. ⚠ **Sparse on purpose — absence is NOT a claim**: it means only the profile tier
   applies, which is a different statement from "estimated". `FilmActiveProfiles.md` now reads its
   marking from this record where one exists, which is the structural fix for the 22 lying cells the hand audit found — and, once EM-A6 extended coverage to every reported parameter on every profile the same day, for the **197** it turned out to be (item 17)
   found earlier the same day.
   **② `ProcessVariant` — the same emulsion under a different chemistry.** Distinct from
   `ProcessingFamily` (time-gamma points *within* one process). Two variants on `CINESTILL_800T`:
   C-41 as shipped (`is_default`, which is what finally states which process the stored curves
   belong to) and ECN-2, the base stock's native process, **deliberately carrying no curves** —
   5219's published curves are for 5219 *with* its remjet.
   **③ `DevelopmentProgress` on `ProcessingSpec`, TRACED not assumed.** Traced from
   `Tani Figs. 7.8/7.9/7.12` on 2026-08-27. For D72 (granular) density/fully-developed-fraction is
   0.0160 at 3 min and 0.0152 at 9 min — constant to 5 %, so D ∝ the grain count; for CP-20
   (parallel) the same ratio moves 0.12 → 0.043, a factor of three, so it is not. Also traced:
   `partial_fill_fraction` ~0.45 (CP-20) vs ~0.04 (D72) — the number a developer-aware grain-noise
   model needs — and `rate_size_coeff_um_min` = 0.591 µm/min for parallel development, omitted for
   granular because that rate is size-INDEPENDENT and `validate` rejects asserting otherwise.
   Set on **9 stocks**, assigned only where the record names the developing AGENTS and they match
   Tani's examples by chemistry (metol-hydroquinone ≡ D72; 4-amino-N-dialkylaniline ≡ CP-20).
   ⚠ **Deliberately NOT inferred from process family** — two worked examples are not a survey.
   **④ Six relational guards in `verify.py`** (409 PASS now): MTF-50 within 0.5–2.0× half the
   printed resolving power for *estimated* f50 only (calibrated: median ratio **exactly 1.000**
   over 59 stocks, and measured f50 is exempt because a real trace beats a rule of thumb); the
   Eq. (1.1) γ/speed/granularity coupling as a within-class, within-era outliner test; provenance
   uniqueness; process-variant default; and the granular-rate-law prohibition.
   ⚠ **THE COUPLING GUARD FLAGGED `KODAK_EKTACHROME_100D_5285`, AND THE FIRST DIAGNOSIS WAS
   WRONG — CORRECTED SAME DAY (A1).** It was recorded here as "a real defect the guard found":
   gamma 15.43 across 0.228 decades = 0.76 stops of latitude, against 4.8–6.1 for every other
   reversal stock. **That reading was mine, and it was mistaken.** The profile's own comment
   already said so — *"gamma 11–15 is the softplus straight-line slope of a model whose toe and
   shoulder nearly coincide; it reads only together with toe_x/shoulder_x. The printed curve's
   actual mid slope is ~1.8–2.2"* — and evaluating the stored curve confirms it: **mid slope
   2.419, usable range 4.25 stops**, both ordinary for reversal. The curve fits exact PDF vector
   coordinates to 0.024–0.028 D RMS. **No refit was needed and none was done.**
   **What WAS defective, and is now fixed, is different and broader.** Two properties presented
   parameter-space quantities as if they were film properties:
   * `ToneCurve.latitude_stops` = `(shoulder_x − toe_x) × 3.3219` ignores knee SOFTNESS, so it
     breaks when the knees sit closer than their own smoothing. **Four stocks are in that regime**
     — 5285 (out by 5.6×), POLAROID_51 (1.8×), POLAROID_146L and POLAROID_410 (1.5×). For the
     other 157 it is accurate to 1 % on 139 and 5 % on 154. It has a domain; nothing said so.
   * the guard itself read `curves.g.gamma` as contrast, which is a model coefficient in that
     regime, not a slope.
   **Fixes:** `ToneCurve` gains evaluated `mid_slope`, `usable_range_stops` and `is_degenerate`;
   the coupling guard now reads `mid_slope`, and **its outlier count across the whole database
   drops to zero with no exception list at all** — a guard that needs an allowlist on its first
   run is usually measuring the wrong quantity. A new **G-LAT** guard fails on any stored/evaluated
   latitude disagreement that is NOT explained by degeneracy, so a genuine fit defect still trips.
   ⚠ **The degeneracy threshold is DERIVED, not tuned**: sorting all 161 stocks by
   `(shoulder_x − toe_x) / max(toe_k, shoulder_k)` against the latitude ratio shows a clean break —
   ratio 1.248 at sep/k 3.32, then 1.475 at sep/k 2.47. The cut sits at **2.5 k**, which is also
   what the softplus geometry predicts (a knee is smoothed over ≈ ±2 k, so two knees need ≈ 2.5 k
   between them to read as two).
   ⚠ **The guard's class/era medians independently reproduce Tani's Fig. 1.1 sensitivity history**
   from our own data: colour-negative K = **0.045** (pre-1960) → **0.219** (1960–89) → **2.676**
   (1990+). ⚠ The figures first recorded here (0.048 / 0.241 / 2.727) were stale within the same
   day — the v17 third-party rms imports raised `rms_granularity` on six colour negatives, and rms
   is a denominator in K. Corrected 2026-08-27 against the live database, here and in the guard's
   own comment. Fig. 1.1 has since been traced (task A3, item 14 below) and the comparison is now
   quantitative rather than a resemblance: see `EMULSION_KNOWLEDGE_BASE.md` §23c.3.

11. ⚠ **NEW 2026-08-27 — schema v17, and `FilmProfile` GREW BY TWO RECORDS.** It gains
   `emulsion` (a new `EmulsionSpec`) and `third_party` (a new `ThirdPartyObservations`). Both
   **INERT**, both appended after every v16 field, so **a v17 database renders bit-identically to
   v16** and no film index moves — but a v16 reader would walk off the end of every `FilmProfile`.
   Owner instruction: where a parameter is **our own estimate** and no T1 datasheet or T2 book
   figure exists, prefer the one published third-party number over an in-house analogy.
   **What actually moved, on 6 stocks:** `rms_granularity` on PORTRA 400 (4.0→6.5), PORTRA 800
   (5.6→11.0), EKTAR 100 (3.4→5.5), GOLD 200 (5.2→9.0), ULTRAMAX 400 (6.4→10.0), HP5 PLUS
   (9.0→12.0); `f50` re-anchored on 6 stocks (EKTAR, GOLD, HP5, ETERNA VIVID, KODACHROME 64,
   ULTRAMAX). **These change rendered output.** `emulsion.grain_um` and `third_party` populated on
   17 stocks; both inert.
   ⚠ **AN ATTEMPT ON THE TWO VISION3 STOCKS WAS MADE AND REVERTED THE SAME DAY.** `verify.py`
   failed four pinned invariants and was right: 5219/5207 carry **measured** per-layer rms (a ratio
   adoption from the sheet's own granularity curves) and a **measured** red f50 anchor of 36.0
   (seven per-record measurements, mean 36.4, queue C24). Both are T1, so the owner rule excludes
   them. This is the guard machinery doing exactly the job §26 B2 of the knowledge base argued for.
   ⚠ **`EmulsionSpec.grain_um` IS A MEAN CRYSTAL DIAMETER AND IS NOT `GrainSpec.clump_um_*`**,
   which is a mean DEVELOPED CLUMP diameter and depends on development gamma and density. Never
   alias them. ⚠ **Every value in `third_party` is TIER 3** and is never evidence for the matching
   observable. `NotFound.md` §7.1 and §7.2 carry the full ledger — including a correction: the
   claim that E-4050 prints an rms of 4 for PORTRA 400 was **wrong** (Kodak print Print Grain Index
   for that film and no rms at all), so that stock was never a contradiction and its 4.0 was our
   own unattributed estimate.

10. ⚠ **NEW 2026-08-27 — schema v16, and `FilmProfile` GREW.** It gains `push`, a new
   `PushSpec`, **INERT** and appended after every v15 field. **A v16 database renders
   bit-identically to v15** and no film index moves — 160 of the 161 stocks take an all-zero
   default with an empty source — but a v15 reader would walk off the end of every `FilmProfile`,
   which is the whole reason the constant moved. It exists because a *manufacturer* statement
   arrived that v15 had nowhere to put: CineStill's own page says 800T "could even be push
   processed up to 3 stops further without any base fog issues". ⚠ **`base_fog_penalty_per_stop`
   stores an AMBIGUOUS zero and `fog_penalty_stated` is what resolves it** — 0.0 means either "no
   source states one" or "a source states there is none", and CineStill's claim is precisely the
   negative one. Same class of problem as the v15 PGI censoring sentinel, solved the same way.
   Read the struct comment before consuming it. It is deliberately NOT a `ProcessingSpec` field
   (that describes the ONE condition the stored curve represents) and NOT a `ProcessingFamily`
   point (that carries measured time-gamma pairs). See `NotFound.md` §7.2b.
   ⚠ **Same day, and larger than the schema bump: `CINESTILL_800T`'s three characteristic curves
   are now TRACED**, 480 points per layer, from CineStill's own published sensitometric figure.
   `dmin` moves from a flat 0.22/0.20/0.19 to the real orange-mask ladder 0.187/0.526/0.876, so
   the stock joins `_DMIN_LADDER` and its `mask_encoding` flips to `dmin_ladder`. **This changes
   rendered output for that one stock** — unlike the schema bump, which changes none. Gamma barely
   moved (0.602/0.621/0.609 traced against 0.610/0.630/0.652 estimated). Tier stays 2;
   `fitted_from` becomes `datasheet_curve` through the new `_VENDOR_TRACED_CURVES` set.
   `NotFound.md` §7.2c has the calibration, the residuals, and the 3.7 % inconsistency between the
   two abscissae CineStill print on their own chart.

6. ⚠ **NEW 2026-08-23 (C21) — schema v11, and the struct GREW.** `HalationSpec` gains
   `radius_scale_r/g/b`. **Every stock ships 1.0, so a v11 database renders bit-identically to
   v10** — but a v10 reader would walk off the end of every `HalationSpec`, which is the whole
   reason the version moved. `Algo_05_Sim.cpp` (both flavours) now builds the halation kernel
   **once per record** instead of once per frame, with a per-record resolvability skip.
9. ⚠ **NEW 2026-08-26 (XX2 + C36) — schema v13, and TWO structs grew.** `DevelopmentPoint` gains
   `base_fog`; `PrintStock` gains `mtf_f50_r/g/b`, `mtf_f50_bound` and `mtf_measured`. Both INERT
   and appended after every v12 field, so **a v13 database renders bit-identically to v12** and no
   film index moves — but a v12 reader would walk off the end of both structs. ⚠ **`mtf_f50_r/g/b`
   store CENSORED values the same way `DyeStabilitySpec` does**: a 0.0 means "this sheet does not
   reach 50 % response for that record" and `mtf_f50_bound` carries what it is known to exceed.
   Never zero, never unknown. The legacy `PrintStock::mtf_f50` scalar is unchanged and is still what
   the reference renderer reads; prefer the triple when it is populated and fall back on a 0.0.

8. ⚠ **NEW 2026-08-25d (C15) — schema v12, and `PrintStock` GREW.** It gains `aging` (an
   `AgingSpec`) and `dye_stability` (a new `DyeStabilitySpec`), both **INERT** and both appended
   after every v11 field. **A v12 database renders bit-identically to v11**, and no film index
   moves — the new `KODAK_VISION3_DI_2254` is appended to `PRINT_STOCKS`, which is a separate table
   from `film_names.txt`. But a v11 reader would walk off the end of every `PrintStock`, which is
   the whole reason the constant moved. ⚠ **`DyeStabilitySpec` stores CENSORED figures**: a field at
   0.0 against `censor_years > 0` means "greater than the bound", not "no data" and certainly not
   "zero years" — read the struct comment before consuming it.

7. ⚠ **NEW 2026-08-23 (C22) — TWO MORE COMPILE-BREAKING SIGNATURES, on purpose.**
   `AlgoSolveAnchors` and `AlgoNeutralMidDensity` each take a new trailing
   `const HighPrecType scannerSpecular` before their out-parameter, and there is a new stage
   `AlgoStage12b_Callier` (declared in `AlgoDyeImpurity.hpp`, defined in both `Algo_12_Sim.cpp`
   flavours) called from `AlgorithmMain.cpp` between stages 12 and 13. `AlgoControls` gains
   `scannerSpecular` (default 0 = inert). Call sites updated in the tree: `AlgorithmMain.cpp`,
   `Algo_13_Sim.cpp`, `AVX2/Algo_13_Sim.cpp`, `profall.cpp`. ⚠ **THE FACTOR MUST REACH ALL THREE
   CONSUMERS OR MID GREY MOVES INSTEAD OF CONTRAST CHANGING** — the anchor solve, the print
   chain's own mid-grey reference and the per-pixel pass. Measured on EASTMAN DOUBLE-X at
   specular 1: with only two of the three, mid grey shifted **+54/255**; with all three it holds
   to 0.2/255 while contrast rises x1.21–1.23 (mono negative) and x1.11 (mono reversal), and
   every colour stock is bit-identical at any setting.
5. **159 stocks, not 155**: `film_names.txt` and the enum shifted after `GEVACHROME_600` (2026-08-19)
   and again after `FUJI_SENSIA_100` and `KODAK_VISION_250D_5246` (2026-08-20). ⚠ **Schema v10 is
   unchanged by the 2026-08-20 work** — the two new stocks add rows, not fields, so if you already
   rebuilt for v10 this is a data-only refresh.

⚠ **The FAIL baseline is 1, not 2, since 2026-08-20b.** The entry that left was
`neighbour pairs couple harder than the far red-blue pair` -- an assertion of a PER-DISTANCE
interimage asymmetry the database deliberately does not store (the evidence says per RECEIVER:
US4725529A Table 1). It was unpassable by construction and had been treated as immovable
because it sat in the baseline. Replaced with the assertion the evidence supports. **Deleting a
guard changes zero pixels** -- this was signal hygiene, not fidelity.

**What the render does differently after 2026-08-20:**

* **Two new stocks:** `KODAK_VISION2_50D_5201` — the first with curves, σ(D), grain level *and*
  per-record MTF all measured from one sheet, and the flattest σ(D) in the file (interior peak 1.20×
  against 1.38–1.62× on its siblings) — and `FUJI_SUPER_F125_8532`, printed scalars only.
* **Nothing existing changed.** 5274's measured MTF (red f50 35.4 against a stored 56.0) is pinned
  in the audit but **not applied**: it needs one owner decision, and it is the tip of a rule-level
  defect — every colour stock with an estimated f50 triple is too sharp in red.

**What the render did differently after 2026-08-19:**

* **Grain level, every stock:** −4 to −8 % amplitude (`1/sqrt(1+fog)`, uniform per stock, no
  channel-balance change). The four Svema Foto stocks are byte-identical by design.
* **Grain shape, 11 stocks:** measured σ(D) — roughly 2.1–2.6× more shadow grain and 2–4× less
  highlight grain than the legacy √density law.
* **Grain level, 6 stocks:** re-levelled from their own curves, mostly in the **blue** record, where
  measurement gives blue 1.9–2.8× green against the old 1.3× estimate.
* **Sharpness, 1 stock:** PLUS-X 5231 keeps f50 41.3 but gains a measured rolloff — response beyond
  f50 rises from 0.020 to 0.169 at 98 cycles/mm (the sheet reads 0.245).
* **Sharpness, 1 stock:** Gevacolor 682's f50 moves 46/54/62 → **29/44/62** cycles/mm, read off its
  own MTF plot; the blue figure is still an estimate, bounded below by the plot.
* **Curves, 3 stocks:** 682's three characteristic curves re-traced (per-layer γ 0.506/0.568/0.540,
  dmax 1.48/2.01/2.26); two new Gevachrome profiles added.

---

## Emulsion-characteristics harvest — status panel (2026-08-27)

⚠ **READ THE ID WARNING FIRST.** Four *different* A/B numbering schemes exist in this project and
they do not refer to each other:

| Namespace | Where it lives | What its `A1`/`B1` mean |
|---|---|---|
| `DIGITIZATION_QUEUE.md` A1–A11 / B1–B3 / C*/E*/F*/G* | the queue | film-datasheet digitisation work |
| `EMULSION_KNOWLEDGE_BASE.md` §23 A1–A8 / B1–B4 | the KB | **aging effects**, not tasks |
| `EMULSION_KNOWLEDGE_BASE.md` §26 B1–B9 | the KB | emulsion Category B proposals |
| the working list used in conversation | nowhere on disk | this session's harvest tasks |

The last one was never written down, which is why this panel exists. Its items are given **`EM-`
prefixes here** and that is now their permanent name.

### Closed

| ID | Task | Outcome | Landed in |
|---|---|---|---|
| **EM-A1** | Ektachrome 100D 5285 — investigate the flagged curve defect | ⚠ **No defect. My claim was withdrawn.** The profile's own comment already explained it; mid slope 2.418, usable range 4.25 stops, both normal. The real bug was `latitude_stops` and a guard reading `gamma`. `ToneCurve` gained evaluated `mid_slope`, `usable_range_stops`, `is_degenerate`; guard **G-LAT** added | `film_profiles.py`, `verify.py`, NotFound.md §6b |
| **EM-A2** | Tani Figs. 7.10 / 7.11 — sulfur-sensitization sweeps | Traced. Rate: CP-20 flat to ±3 %, D72 falls 4.2×. Sensitivity falls −0.30 vs −0.88 log. Fog 0.107 vs 0.351 | KB §23b |
| **EM-A3** | Tani Fig. 1.1 — sensitivity over ~150 years | Traced, 2246 columns. Piecewise linear, 7 knots, RMS 0.017 decade. Three plateaux, two steps (×56, ×51), modern era decelerating +0.371 → +0.224 decade/10 yr | KB §23c, `doc/thirdparty/tani_fig1_1_raw_px.txt` |
| **EM-A4a** | Wall — Tappen & Rekaschow coating weights | Verified without a render; page confirmed by folio; C + D = 100.000 on all ten rows. **Two errors found: one ours (~9 % high, wrong base), one the book's second (a composition pair summing to 92.6)**. First thickness the corpus supports: 21.9 µm dry | KB §3.3.1 + §3.3.1a |
| **EM-A5** | Davis & Walters Part III — 86-emulsion survey | Harvested and image-verified. **OCR measured at 67.4 % on speed**, failing by silent leading-digit drops. Resolving power ∝ speed^−0.14, surviving within class | KB §23d, `doc/thirdparty/davis_walters_1920_survey.txt` |
| **EM-CIT** | PDF↔book offset audit across all four books | Offsets re-derived per page. Duffin has two regimes, Wall three. 2242 pairs checked, 9 wrong, all fixed | KB §29.1 |
| **EM-253** | Wall p.253 | Closed as **permanently unavailable from this copy** — the page is absent from the scan, not un-OCR'd | KB §29.3 |
| **EM-B7** | Audit acetate profiles for unsourced `base_yellowing_d` | ✅ **CLOSED AND GUARDED 2026-08-27.** Audit found **0 of 161** — a clean negative, nothing to withdraw. It was clean by accident, so **G-YELLOW** now pins it, and it also covers the **11 print stocks**, which the §26 B7 item as written had not thought of. ⚠ The guard cannot say "acetate" — there is no base-material field (§26 B8) — so it asserts zero on every profile, which is the stronger claim and matches the evidence: no base in the corpus has a documented yellowing density | `verify.py` G-YELLOW |
| **EM-A6** | Per-parameter provenance for everything the active-profiles document prints | ✅ **DONE 2026-08-27. 52 → 1529 entries, 26 → 167 profiles**, covering all nine reported parameters. ⚠ **THIS ROW CARRIES A LIVE COUNT, NOT THE ONE THE TASK LANDED WITH** — `doc_consistency.py` registers the number here, so it tracks the database rather than the day. EM-A6 itself landed 1463/161 on 2026-08-27; +7/+1 for KODAK_EKTAR_125 on 2026-08-31 and +28/+12 for the AGFA harvest on 2026-09-01, and +4/+0 for the US 4,495,277 sensitization defaults the same day, and +2/+0 for the Flueckiger 2018 Technicolor and Dufaycolor records. ⚠ **197 CELLS FLIPPED FROM "DOCUMENTED" TO "UNDOCUMENTED"** — the hand audit had found 22; the systematic pass found 197. Six flipped the other way. Census: 840 estimated, 311 assumed, 225 traced, 48 derived, 29 stated, 10 measured | `_PARAM_SOURCES_DERIVED` |
| **EM-A7** | Developer identity mined from the on-disk sheets | ✅ **DONE 2026-08-27. 13 → 29 stocks.** 16 identities accepted, each quoted from the stock's OWN datasheet. ⚠ **13 of 29 candidates were REJECTED as false positives**, including one that would have shipped — see item 18 | `_PROCESSING_DEVELOPER_MINED` |
| **EM-A8** | Development-family speed and contrast guard | ✅ **DONE 2026-08-27 as G-DEVFAM**, two checks, threshold derived from the data (widest real family 1.30×, cut at 2.0×). It caught a real defect before it shipped | `verify.py` G-DEVFAM |

**Also closed, and they were §26 proposals rather than session tasks:**

| §26 ID | Proposal | Actually implemented as |
|---|---|---|
| **B1** | Development progress type | schema v18 `DevelopmentProgress` + `partial_fill_fraction` + `rate_size_coeff_um_min`, set on **9 stocks**, guard **G-PROGRESS** (rejects a 1/d coefficient on GRANULAR) |
| **B2** | γ / granularity / speed coupling guard | **G-DQE**, class-and-era banded, reading `mid_slope`. **Zero outliers over the whole database with no allowlist** |
| **B5** | `EmulsionSpec` as a new record | schema v17, populated on **17 stocks** |
| **B9** | Digitisation targets | **§28.1 is now 9 of 9 resolved** — six traced or harvested, one superseded, two struck as having no route into the renderer |

⚠ **THE KNOWLEDGE BASE STILL DESCRIBES ALL FOUR AS UNBUILT.** §26 B1 says "deliberately held for
the combined schema revision", B2 says "the conversion must be established before the guard can be
numeric", B5 says "Why it is not implemented". Those sentences were true when written and are false
now. **§26 currently misreports the state of four of its nine items** — that is the same class of
defect as the FilmActiveProfiles cells and the stale K ladder. **Not rewritten without your
approval; say the word and it is a small edit.**

### Open — effort only, no decision needed

⚠ **EM-A6, EM-A7 and EM-A8 were the whole of this list and are now CLOSED** (see above). What is left
is blocked on evidence or on a design decision, not on effort — which is a different situation and is
worth saying plainly rather than leaving three empty-looking rows.

| ID | Task | Why it is still open | Effort |
|---|---|---|---|
| **EM-A7b** | Widen developer identity past 29 stocks | ⚠ **THE CORPUS CANNOT CLOSE THIS.** 74 PDFs for 161 stocks, and most citations are prose, not filenames. Of the stocks whose own sheet IS on disk, the ones that name a developer or process in recoverable text have now been taken. Six more are recorded at tier 2 because the sheet's two-column text layer interleaves and the sentence cannot be quoted intact — **rendering those six pages would upgrade them to tier 1** | small, and it is the only cheap win left |
| **EM-A8b** | Widen G-DEVFAM past two testable families | Only `Process ECN-2` (15 stocks) and `ID-11` (6, after normalising two spellings) reach the n ≥ 3 floor. Every further family needs EM-A7b first | follows A7b |
| **§26 B3** | Per-record grain scaling from the volume/area distinction | ⚠ **Source-blocked, not effort-blocked.** Tani gives both scaling laws and the absorption figures but **no granularity ratio**. Needs a derivation with stated assumptions, or a measurement | blocked on evidence |
| **§26 B4** | Two-component Dmin (emulsion fog vs developer fog) | EM-A2 **sharpened it without closing it**: direction and developer dependence are now quantified, the **rates** are not. Needs a measured Dmin-versus-time series for at least one stock | blocked on evidence |
| **§26 B6** | Reciprocity from intensity-dependent nucleation | Source constrains the **shape** and supplies **no coefficient** — no Schwarzschild exponent, no absolute threshold intensity, and no temperature dependence anywhere in the book | blocked on evidence |
| **§26 B8** | Base material as a preset-validity constraint | Metadata, never a pixel operation. Needs a decision about **where preset validity lives** before it can be built | needs a design decision |

### Open — blocked on you

| ID | Ask | Why no document can close it |
|---|---|---|
| **EM-B1** | Halation radius and magnitude in µm, 156 stocks | Requires a measurement on your own frames. No datasheet in any corpus publishes it |
| **EM-B2** | CineStill 800T figure-2 curve assignment | Two curves on the vendor page, no legend. Attach the PNG and it is settled in minutes |
| **EM-B3** | Document acquisition | Kodak E-4046, H-1-40295, F-4017; Fuji AF3-0076E5, AF3-058E3. Cited but not in this checkout |
| **EM-B4** | Three schema decisions | `aim_density` shape; alternate-EI curve sets; F2 option 1/2/3. Held together deliberately — landing three overlapping schema changes separately would make three migrations where one would do |

### Permanently closed — recorded so they are not reopened

Per-stock crystal habit, iodide content, aspect ratio, sensitization type and coated thickness;
grain size → rms granularity as a derivation; gelatin batch → speed/fog magnitude; turbidity → MTF
coefficient; CineStill rms / MTF / spectral; vinegar syndrome as an image effect (contradicted by
the source itself); the IPI storage-prediction apparatus; and, new on 2026-08-27, **Davis & Walters'
258 plotted curve families** — real measurements of emulsions the paper deliberately anonymised,
with no stock in our database older than 1930 to attach them to.

### One-line summary

**Emulsion harvest from the six-source corpus is finished, and so is the provenance work that came
out of it.** §28.1 is 9 of 9. EM-A6, EM-A7, EM-A8 and EM-B7 all closed on 2026-08-27.

What remains under "emulsion" is **four §26 items blocked on evidence or on a design decision rather
than on effort** (B3 needs a granularity ratio no source prints; B4 needs measured Dmin-versus-time;
B6 needs a Schwarzschild coefficient the book does not give; B8 needs a decision about where preset
validity lives), **two cheap follow-ons that the corpus itself limits** (EM-A7b/A8b), and **four asks
on your side**.

**No further reading of Tani, Wall, Duffin, Davis & Walters, Reilly or NEDCC is scheduled**, and the
KB records why for every item not taken.

⚠ **The most useful thing this area produced is not a value, it is a count.** Extending provenance to
every parameter the active-profiles document prints flipped **197 cells from "documented" to
"undocumented"**. The database did not get worse; it stopped overstating itself. Nine of those cells
per stock now say what they actually are, and the four new guards mean the next wrong one fails a
build instead of being found by hand a month later.

---

## What needs a decision from the owner

Nothing below is blocked on work. Each is one answer.

| # | Decision | Why it matters | Recommendation |
|---|---|---|---|
| **C7** | Apply the temporal grain law? Honjo 1989: at 24 fps the eye integrates ≈0.2 s ≈ 5 frames, so perceived granularity falls by **1/√5 ≈ 0.447** versus a frozen frame | This engine renders motion for an AE/PR plugin. Taken literally, still-frame grain is **2.24× too strong in playback**. Whether that is a defect depends on whether you judge frame-by-frame or in motion — a product decision, not physics | Ask first whether you grade on stills or in playback |
| ~~**C8**~~ | ✅ **Answered and done 2026-08-23: the owner chose seconds (option a).** `exposure_time_s` / `exposureTimeS`, 0 = not stated and inert. Shutter angle was rejected on evidence, not taste: angle / frame rate only ever spans 1/1000–1/24 s and every sheet prints *no correction needed* there. ⚠ My own entry's model claim was wrong — all measured data is TIME-only, so the honest form is a per-channel global shift, not a per-pixel intensity effect. **15 measured tables read from the sheets** (6 → 21); seven stocks had a printed correction and were rendering nothing | | |
| ~~**C13**~~ | ✅ Answered 2026-08-20c: adopted. Raised **C24**, which C2b answered on 2026-08-23: the rule was wrong in FORM, not in its constant, and the correction is now adopted (five measured triples, five re-anchored reds) | | |
| ~~**C1e**~~ | ✅ **Answered and done 2026-08-23: adopted, three stocks.** The item's premise was disproved first — the raster σ axis is NOT 1.32× high (its own 15-tick comb reproduces 0.001–0.100 within 1 %); that figure came from comparing at absolute D 1.0, and at net 1.0 the two documents differ 1.12× in green. So 5219 = 5.92 / 6.60 / 17.84, 5207 rms_b 8.92, 5203 rms_b 4.71, greens frozen. 5213 stays on the heuristic (band-only sheet) and is pinned there | | |
| ~~**C6**~~ | ✅ Done 2026-08-20 — `FUJI_SUPER_F125_8532` added, batched with 5201 so the ListBox shifted once | | |
| ~~**C5**~~ | ✅ Done 2026-08-20 — `EASTMAN_5247_1983` re-tiered to 1, tagged `[T1/T2]`. ⚠ Surfaced C12: a mixed tag falls to tier 3 unless listed, which is where 5218 and 5217 have been sitting | | |
| ~~**C24**~~ | ✅ **Answered and adopted 2026-08-23.** The rule's FORM was wrong, not its constant: measured red f50 is effectively fixed at 36.4 c/mm (±13 %) while green spreads 52 % and blue 70 %, so `f50_r = k·f50_b` fails at every k. Five measured triples adopted; five modern Kodak cine reds re-anchored to 36.0; 5205/5293 given a mixed triple and left unflagged; every other maker and every pre-1990 stock untouched. ⚠ Render impact up to **45.7/255** on 5203 | | |
| **C4** | ЦО-90Д / ЦО-90Л — add or not? | Two documents with near-identical norms would render identically | Argued against |

---

## Done recently, newest first — 2026-08-18 to 2026-08-26

### 2026-08-26g — two more E-series sheets: GOLD 100 closed, a new stock, and a caption matcher that had been skipping whole figures

`E7022-Gold_100_200.pdf` (E-7022, February 2007) and `e29-Pro_100T_PRT.pdf` (E-29, April 1999). Both
read in full, both vector, **neither publishes rms granularity** — the string appears on page 3 of
each only inside the boilerplate explaining that Print Grain Index replaces it. The rms gap stays
open and the pre-1997-edition hunt is still the only answer to it.

**⚠ THE CAPTION MATCHER HAD BEEN SKIPPING COMPLETE FIGURES IN SILENCE.** `find_panels` tested
`low.startswith(k)`, which is fine for eleven single-film sheets and wrong for a two-film one: E-7022
(2007) captions its panels **"KODAK GOLD 100 Film Characteristic Curves"** and **"KODAK GOLD 200 Film
Characteristic Curves"**, with the panel kind at the END of the line. Neither matched. The page
reported only its spectral and dye panels while **two complete three-channel characteristic figures
sat there unread**, and nothing failed — the sheet simply looked like it had no curves. Matching on
substring (longest key wins, so a line naming two kinds cannot be claimed by the shorter) found both.
**The lesson is the shape of the bug, not the fix: a locator that finds fewer panels than exist
reports success.** The panel-count expectation added in the previous pass guards traces WITHIN a
panel; nothing guarded the number of panels. Both new panels are now pinned in the audit, which is
the closest available substitute.

**One dye panel, two films — and the identity had to be proved rather than assumed.** The 2007 sheet
prints a single dye panel and never says which film it belongs to. Assigning it to both would
double-count one measurement. Traced and compared against the panel in E-7022 (March 2022), a GOLD
200-ONLY sheet that does name its film: **the same artwork, max difference 0.0005 D and rms 0.00009 D
over 59 resampled points of both curves.** So the panel is GOLD 200's, already adopted under that
name, and **KODAK_GOLD_100 stays honestly empty of dye data.** The audit pins the comparison so a
later pass cannot quietly hand the panel to GOLD 100 as well.

**The strongest cross-check the uncaptioned-panel override has had.** GOLD 200's curves now come from
the 2007 sheet by CAPTION and the 2022 sheet by GEOMETRY OVERRIDE — two different location
mechanisms, fifteen years apart, on one emulsion. They agree to 0.002 D in dmin and 0.008 in gamma,
and both sheets print `Log H Ref: -1.14`.

**⚠ Kodak omitted a minus sign, and this one is NOT the macron.** The 2007 sheet prints
`Log H Ref: 0.84` for GOLD 100 and `-1.14` for GOLD 200. Verified against a 450 dpi render of both
captions: the 200's minus is a real glyph and the 100's is simply **absent**, so this is a different
defect from the overbar-minus on the E-190 axis labels. A positive Log H Ref is implausible for a
daylight ISO 100 negative when every other sheet in the corpus is negative, and **-0.84 against -1.14
is exactly 0.301 decades — one stop — which is the ISO 100 versus 200 difference.** Recorded as an
inference; nothing adopted from it, because Log H Ref is not a stored field.

**KODAK_PRO_100T_PRT: the 161st stock, and the first added since the ordering rule was written down.**
E-29 covers KODAK Pro 100T Film / PRT, a discontinued tungsten ISO 100 negative sold in 120 and sheets
only, absent from the database. Adopted: characteristic curves (fit rms 0.0066-0.0133 D), a
neutral+D-min dye pair over **450-700 nm** — narrower than the PORTRA and GOLD panels because neither
of its curves reaches 400 nm, so 51 samples rather than 59 — Print Grain Index for both formats it
was actually sold in, and a **five-point reciprocity walk**, only the second in the database. Grain
and MTF are estimates by analogy to PORTRA 100T, with rms nudged 4.0 -> 4.2 because the two sheets'
120-format PGI rows are directly comparable and PRT is the grainier (<25/35/58 against <25/33/55).

Adding a stock renumbers `film_enum.hpp`, so the ordering rule got its first real exercise: the
database is authoritative, all four representations were regenerated from it, and positional identity
was re-verified rather than set equality. `film_names.txt` md5 moved 2de4536b -> 41e0bc5d as it must.

**⚠ E-29 NAMES PORTRA 100T AS ITS REPLACEMENT, AND THAT LICENSES NOTHING.** The note cites E-2468 by
number, so the two profiles are a documented succession — but PRT's curves are its own and are not
applied to PORTRA 100T, whose plots remain Kodak's copy of PORTRA 160VC's. **Their reciprocity tables
are numerically identical entry for entry.** Given that E-2468's figures are demonstrably copied
artwork, a carried-over table is at least as likely as two films measuring the same; each profile
cites the publication that prints it and nothing is merged. PORTRA 100T's provenance now also records
that its stored [T1] spectral set is the shared 160-family figure F009_0180AC — a family curve Kodak
attributed to the film, not a per-film measurement. Its traced layer spans (blue 368-509, green
438-589, red 539-689 nm) match the stored array, which is how the identification was confirmed.

verify.py 403 PASS / 1 baseline FAIL; audit 67 values across 13 sheets; C++ clean on 18 TUs.

### 2026-08-26f — reciprocity bounds for six stocks, and spectral sensitivity closed as unobtainable rather than deferred

Two follow-ups to the E-series harvest, chosen because one was free and the other stops a future pass
wasting effort.

**Reciprocity: six empty fields filled from text that was already read.** Every one of the six
harvested stocks publishes a bound, not a walk — "You do not need to make any exposure or filter
adjustments for exposure times of 1/10,000 second to 1 second" — and `ReciprocityTable` already had
the right shape for it: **a one-point entry whose only correction is 0.0**, which says "no correction
is required up to here and nothing is stated beyond it". That is the censoring idiom the MTF fields
use, applied to TIME instead of frequency, and the docstring now says so, because both plausible
misreadings lose information: "the correction is zero" overstates, an empty field understates. All
six read 1.0 s.

**⚠ Kodak's own page prints "to i second", and it is not an OCR artefact.** E-4051 page 2 reads "for
exposures from 1/10,000 second to i second" — verified against a 400 dpi render of the line, so the
lowercase i is in the typesetting. It is a defect for "1": the five sibling sheets print "1 second"
in the identical sentence and no bound of "i" seconds exists. Stored as 1.0 with the defect quoted in
the citation rather than silently normalised.

**⚠ ULTRA MAX 400's bound disagrees between vintages by a factor of ten** — 10 s in E-7019 (February
2007) against 1 s in E-7023 (February 2016), same film, same sentence. The later figure is stored and
the earlier one cited (method rule 4). The sheets do not say whether Kodak tightened the claim or the
emulsion changed.

**Spectral sensitivity: CLOSED, not deferred, and the difference matters.** The previous entry said
the layer curves cross and left it there, which invites a later pass to try again. So it was tested
properly: every panel run through a plausibility window on the assigned peaks (blue 415-485 nm, green
525-565, red 595-665), which every real colour-negative tripack satisfies.

**Zero of the seven database targets yields a usable reading. Exactly one panel in the batch passes,
and it belongs to PORTRA 400NC — a film that is not in the database.** Four panels are REFUSED
outright (2 or 4 traces where three are drawn); PORTRA 400's returns a "blue peak" at **257 nm**,
which is the axis's own left edge and the signature of a chain welded across a crossing rather than a
sensitivity maximum. The asymmetry is structural, not luck: the E-190 family pages draw the three
layers as three separate long subpaths, and the later single-film sheets fragment them into pieces
that only a chainer can reassemble — and a chainer cannot be trusted where curves cross. Recorded as
a per-panel table in `NotFound.md` §4.9.1 so the next reader sees the evidence rather than the
conclusion. **What would change it is a different kind of source or a different kind of reader, not
more effort on these eleven files.**

Two new guards pin the reciprocity shapes: the six bounds must stay one-point 1.0 s / 0.0, and
PORTRA 100T must remain the only multi-point walk in the batch — it is the likeliest to be
"simplified" later, being the only one whose sheet publishes exposure INDEX against time rather than
a correction. verify.py 401 -> 403 PASS.

### 2026-08-26e — eleven KODAK still-film E-series sheets harvested; schema v15; a sixth curve reader; and a copy-paste defect in Kodak's own publication

**What arrived.** Eleven documents in `PDF/PROFILES/KODAK`, covering eight products already in the
database and five that are not: E-190 in two vintages (May 2003, six films; October 2006, five),
E-2468 (PORTRA 100T), E-4040 / E-4050 (two vintages) / E-4051 (the 2011+ PORTRA line), E-7019 and
E-7023 (ULTRA MAX 400, 2007 and 2016), E-7024 (ULTRA MAX 800), E-7022 (GOLD 200, 2022). All eleven
readable, none encrypted, all VECTOR. **Six of the eleven are byte-identical duplicates of files
already in that directory** (md5-verified on the source machine) — which did not make them redundant,
because the container's PDF mirror held only about 35 of the ~200 files there and most had never been
opened.

**A sixth plot reader, `kodak_still_curves.py`, and why none of the five existing ones could be
extended instead.** Every panel in all eleven sheets is MONOCHROME: E-190 (2003) draws its entire
figure set in one ink, and the 2016 Alaris re-issues use three near-blacks that encode PDF layer
order, not channel identity. **Kodak's cine ink convention — the whole basis of
`spectral_vector.py` — does not hold on the still sheets.** So channels are read from the printed
`R`/`G`/`B` letters instead, matched by proximity and assigned globally rather than in list order
(on E-190 p11 the MTF panel's G and B letters are 6.8 pt apart, closer to each other than either is
to its curve). 77 panels located, zero skipped.

**Four defects the reader had to be built around, each found by a wrong answer first:**

* **The macron minus, on the X axis this time.** Same defect `spectral_vector._sign_y_ticks` was
  written for, and worse here because the exposure axis is mostly negative: PyMuPDF returns
  `4.0 3.0 2.0 1.0 0.0 1.0` for a run that reads −4 to +1. Signing by position about the zero tick
  fixes it — **but the first version of that fix accepted any monotonic run, and E-190 p13's
  "EI 800 (Push 1)" panel calibrated to a perfectly collinear, perfectly MIRRORED axis
  (x 4.000..0.003), putting its traces at logE −0.555..3.438 instead of −3.44..0.57.** A mirror is
  linear, so no fit-quality test can see it. What sees it is physics: exposure, wavelength and
  spatial frequency increase left-to-right and density increases upward, so the axis slope has a
  required sign. Guarded now.
* **Label centres are not tick positions.** On E-7019 p4 the drawn x ticks sit at an exact 36.9 pt
  pitch while the "−2.0" label is misplaced by 2.8 pt, which failed a 1.5 pt collinearity test and
  SKIPPED a real curve set. Snapping labels onto drawn tick geometry fixes it — **and snapping
  unconditionally then broke the spectral-sensitivity and MTF panels**, whose dense rulings let a
  6 pt snap window pull a label onto its neighbour's tick. Snapped reading is tried first and the
  labels are the fallback, which is the correct order because a drawn tick *is* the position.
* **The 2016 re-issues draw traces as BEZIERS, not polylines.** E-4051 p4 holds each characteristic
  curve as 20 cubic segments. Skipping non-line items reported "0 traces" from a good three-curve
  figure — the same class of silent zero as the `MONO_MAX_CHANNEL` bug on 5222.
* **Four sheets do not draw a curve as a curve.** E-7019's spectral panel is 60-odd separate paths
  spanning TWO POINTS of x each; E-7022's is 41 of four points. All report `dashes = '[] 0'`, i.e.
  solid, so the dash attribute is useless for grouping. The same fragmentation split E-7019's and
  E-7024's RED characteristic curve in two and the first run silently reported `named=['B','G']` on
  both. A chainer fixes it — **and chaining is only applied when the unchained reading comes up
  short**, because on the spectral panels, whose three layer curves genuinely cross, chaining welds
  the blue layer's descent onto the green layer's ascent and returns two traces for three curves.

**⚠ THE CHARACTERISTIC PANELS DRAW NO SHOULDER, WHICH CHANGES WHAT MAY BE ADOPTED.** Local-slope
profiling shows the slope rising through the toe to a plateau that holds dead flat to the right edge
— E-190 p9 reads 0.527, 0.528, 0.528, 0.529, 0.528, 0.527 across its last six red samples. A free
six-parameter `ToneCurve` fit therefore invents three of its six numbers, and corrupts a fourth: run
unconstrained on 160NC's red it put the shoulder at logE 1.276 on a plot ending at 0.874, and
reported gamma 0.601 where the drawn line is 0.528 — **a 14 % error in the single most important
curve parameter, caused by a phantom shoulder pulling the slope.** `dmin`, `gamma`, `toe_x` and
`toe_k` are measured; `shoulder_x` is carried over unchanged and Dmax follows from it.
**The obvious alternative was tried and is wrong:** preserving the old DMAX instead put
KODAK_PORTRA_160's blue shoulder at logE 0.115 — inside a traced range that runs to +0.95 and is
straight the whole way — because the old Dmax figures were chosen to sit above an estimated
near-neutral dmin of 0.19, not above the real masked 0.857. An assertion now refuses any carried-over
shoulder that would fall inside the traced range.

**Adopted.** Six profiles' characteristic curves replaced by traced ones (`KODAK_PORTRA_160`, `_400`,
`_800`, `KODAK_GOLD_200`, `KODAK_ULTRAMAX_400`, `_800`), worst residual 0.012–0.154 D with 20 of 24
channels under 0.06. All six move from `mask_encoding` "neutral_dmin" to "dmin_ladder" — **the
previous analogy estimates had flattened a real 0.61–0.70 D mask ladder into a near-neutral
0.20/0.19/0.19 triple, so the old encoding was not merely imprecise but the wrong KIND of
description.** Four schema-v14 neutral+D-min dye pairs at 5 nm (the pair count goes 1 → 5). Three
measured MTF sets with fitted rolloff exponents and measured adjacency overshoot (measured-MTF count
12 → 15; the first STILL films in that set). Seven Print Grain Index records, on a new schema-v15
field. One five-point reciprocity table.

**⚠ E-2468 CONTAINS NO ORIGINAL FIGURES. Its entire CURVES page is PORTRA 160VC's artwork.** Its
characteristic figure is `F009_0153AC`'s sibling `F009_0154AC` — the same figure id E-190 prints on
its 160VC page — and tracing both independently returns identical numbers to four decimals (dmin
0.2045/0.6087/0.8121, gamma 0.5809/0.6050/0.6691). Its spectral-sensitivity figure is `F009_0180AC`,
the shared 160-family plot, and its dye pair traces identically to 160VC's too. A tungsten ISO 100
film and a daylight ISO 160 film cannot share a characteristic curve. **`KODAK_PORTRA_100T` therefore
gets nothing from its own datasheet's plots** — only its text, which is sound and unique (the
reciprocity table is the only multi-point one in the batch).

**Refused, each for a stated reason.** No rms granularity from Print Grain Index: the sheets say PGI
"cannot be compared to rms granularity" and E-58, which defines the method, declines to publish the
transformation — its first step alone depends on four properties of the print PAPER. No spectral
sensitivity re-derived: the layer curves cross and on 7 of 11 panels the separation is ambiguous, and
these eight profiles already carry sets cited to these same publications. No dye pair for
`KODAK_PORTRA_400` or 2007-vintage ULTRA MAX 400: their panels resolve into three traces and one
where the caption promises two. `KODAK_GOLD_100`: not covered by any of the eleven documents.

**⚠ A GUARD CAUGHT ME PUTTING A FILTER FACTOR IN A FILM FIELD.** Five sheets publish a tungsten
exposure index, and writing them into `exposure_index_tungsten` looked like free gap closure.
`verify.py` rejected it: the field is defined as UNFILTERED pairs, and 160/40 is exactly 4× — the
80A filter's own transmission loss, identical for any daylight film. It measures the filter, not the
emulsion, and it would have filled the panchromatic cluster with filter factors and broken the
blue-versus-panchromatic physics claim the field exists to carry. Discarded; the numbers stay in the
citation text where a filter factor belongs.

**Version conflicts recorded, not averaged (method rule 4).** PORTRA 800's PGI is published at
50/72/101 (E-190, 2003) and 48/70/99 (E-190 2006 and E-4040 2016), and its traced curves differ with
it — red dmin 0.3168 / gamma 0.5638 in 2003 against 0.2200 / 0.5372 in 2006. ULTRA MAX 400's
reciprocity bound is 10 s in E-7019 (2007) and 1 s in E-7023 (2016).

**Independent validation of the whole pipeline:** `KODAK_PORTRA_160NC` traced from E-190 (2003) p9 and
from E-190 (2006) p8 — two different files, different md5 — returns dmin 0.2044/0.6089/0.8116 and
gamma 0.5279/0.5501/0.6078 from both, identical to four decimals. E-4050's 2010 and 2016 vintages
agree to 0.008 D and 0.008 in gamma. And `PORTRA 100T`'s PGI 33/55/84 in E-2468 is confirmed by
E-58 (2000) p5 independently.

**Also settled in passing:** the D-0.2 spectral criterion. E-190 (2003) p9 prints
**"Density: 0.2 above D-min"** inside the panel, and E-4051, E-4050, E-4040, E-7022 and E-7023 print
`Density: 0.2>D-min`. The criterion string those profiles carry is sourced after all.

### 2026-08-26d — C37 closed with no adoption, and F2 turns out to be 146 stocks

`verify.py` **391 PASS / 1 baseline FAIL**, 14 audits green, C++ clean, schema v14, nothing adopted.

**C37 — the honest outcome is that it found no defect.** Eleven panels re-derived from vector paths
against the 2026-08-02 raster batch. Both classes of apparent disagreement were chased down and both
are artefacts of the comparison:

* **Level** (5245 blue "rms 0.340 decades") — comparing a *truncated* trace against a complete one
  after per-layer peak normalisation. Two curves that stop at different wavelengths, each normalised
  to its own maximum, disagree by construction.
* **Peak** (5246 blue traced 430 vs stored 470) — **argmax noise on a plateau**. Measured plateau
  width, samples within 0.05 decades of the maximum: 5274 **0 nm**, 5245 **10 nm**, 5205 **40 nm**,
  5246 **40 nm**. On 5246 *both* readings agree the plateau runs 430–470.

Traces verified the way this project verifies traces — rendered back onto the page; on 5245 and 5246
the points lie on the printed curves. So readers sound, adopted data sound, **nothing re-adopted**: a
wash is not a reason to churn adopted data.

⚠ **But it found a fragile guard.** The blue-peak 6/4 split pins an *argmax* on stocks whose maximum
is a 40 nm plateau — a re-trace by any reader could move 5246 and 5205 between its two groups with no
data change. A second guard now asserts the stable property: each stored blue maximum must lie inside
its own measured plateau.

⚠ **And caption-based assignment was tried and rejected on evidence.** These panels do print
"Yellow-/Magenta-/Cyan-Forming Layer" inside the frame, so assigning by caption looked strictly
better than assigning by absorption band. It is not: on 5245 the "Magenta-" caption sits 43 nm from
one peak and 47 nm from another. The band test is the sound rule here — the reverse of the 7239 mono
panel, where the captions were the key.

**F2 — investigated, and it is four times the size the queue said.**

⚠ Unblocked since **C1 closed on 2026-08-18** ("depends on C1, and inert until then"), so it sat
actionable for eight days. The row says "the 103-stock default"; the live count is **147**.

| group | n | dmax/mid | rises | falls |
|---|---|---|---|---|
| measured NEGATIVES | 11 | 0.50–0.90 (mean 0.68) | 0 | **11** |
| heuristic NEGATIVES | **113** | 1.00–1.80 (mean 1.24) | **112** | 0 |
| measured REVERSALS | 2 | 2.83–3.10 | **2** | 0 |
| heuristic REVERSALS | **34** | 0.50 exactly | 0 | **34** |

⚠ **Both defaults are contradicted in direction by every measurement of their own class — 146 of 147
stocks, not the 34 previously recorded.**

⚠ **One real mitigation, negatives only:** no unmeasured stock sets `sigma_shape_peak` (0 of 147)
while **all eleven** measured negatives do, at **1.20–1.62 located 0.65–0.80 up the scale**. So the
negative default's "1.20 at dmax" stands in for an *interior peak* the triple cannot express — the
rise is real, it is in the wrong place, and the fall after it is missing. The reversal default has no
such excuse: it is backwards.

**Nothing changed.** Every option moves 146 renders, which puts this on the same footing as C16. Two
guards pin the contradiction so it stays visible.

### 2026-08-26c — the B group worked through, and two of my own claims corrected

**Schema v13 → v14.** `verify.py` **388 PASS / 1 baseline FAIL**, 14 audits green, C++ clean on
18 TUs, no film index moves.

⚠ **CORRECTION 1 — "no sheet in the corpus prints 0.2" was FALSE, and it had been in the guards,
`NotFound.md` and the queue since 2026-08-25.** A full-corpus sweep found **three files that print
it**: `5205t.pdf` p4, `KODAK VISION2 250D … 5205.pdf` p4 and `5218-Vision2-500T-H-1-5218t.pdf` p4,
each carrying **`D=0.2>D-min`** in the Spectral Sensitivity panel's own caption block, beneath
`Densitometry: Status M`. **Why it was missed:** the earlier sweep searched only *inside* the plot
frame — where 5222 and 7239 put the caption. The VISION2 layout puts it *below*. "Not printed" was
"not printed where I looked", the same lesson as the F-125 outlined-vector-art case arrived at from
the opposite direction.

**So the 16 D0.2 profiles split three ways, not two:** 2 **sourced** (5205, 5218); 5 Kodak cine a
**family inference with a documented anchor inside the family** (5217, 5203, 5207, 5213, 5219); and
**9 STILL films** (EKTAR 100, GOLD 100/200, PORTRA 100T/160/400/800, ULTRAMAX 400/800) which are the
live gap — a different product line in different publications. Owner decision: keep the values,
annotate them. Done via `_CRITERION_FAMILY_INFERENCE`, and `verify.py` asserts the annotation lands
on exactly the five inferred stocks and **not** on the two that print it.

⚠ **CORRECTION 2 — I told the owner C37 was "up to 13 new spectral sets". It is ZERO.** All eleven
stocks behind those 15 panels already carry a set. I inferred the number from "15 panels became
findable" without checking which stocks had data — the same unchecked inference this file keeps
catching in others. What the panels *did* deliver is the criterion finding above.

**B1 — worked through in full: one adoption, three row errors.**

* **5248 ADOPTED, and the SCHEMA was the blocker.** Its p3 panel prints "Typical densities for a
  midscale neutral subject and D-min." and draws exactly **two** traces; `SpectralDyeDensity`
  demanded cyan AND magenta AND yellow, so a clean panel could never be stored — a published
  measurement discarded for having the wrong *shape*. **v14** adds `d_dmin` and a second legal shape.
  ⚠ `has_data` keeps its old meaning, so no count or document moves; `has_neutral_pair` reports the
  new one. Two physical checks, neither fitted: the neutral exceeds the D-min at all 31 samples by
  ≥ **0.463 D**, and the D-min behaves as an orange mask must — peaking **1.011 in the blue at
  440 nm**, falling monotonically to **0.169 at 700**.
* **5246 REFUSED, alternatives excluded by measurement.** Seven solid traces coexist at 480–580 nm
  against five legend entries; labels sit in whitespace so no positional rule can work; and "two
  products on one plate" — attractive, since the header names 5246 *and* 7246 — is dead: no traces
  pair off (closest sd 0.103 D over a 0.330 D range, next closest a *crossing*).
* **5247 p4 has no plot on it.** It is a plate index; the row was matching a line of contents. The
  real plates are p6–9 and they are **rasters**.

**B3 reclassified — the sources are not in this checkout.** The work is real (Technical Pan has no
spectral set at all) but **P-255** and **F-4043** are not under `PDF/PROFILES` here, though both are
cited in `_PROVENANCE_SOURCES`, so they were read once. A checkout gap, not a research gap.

**One acquisition item closed for free:** `Basic-Photographic-Sensitometry-Workbook.pdf` — Kodak
**H-740**, November 2006 — was already on disk. I had sent the owner hunting for it. ⚠ It does **not**
contain the spectral criterion: it documents ANSI/ISO *speed* (0.10 above D-min), average gradient
and contrast index. Worth recording so nobody later confuses the 0.10 speed point with the 0.2/0.4
spectral-panel criterion.

### 2026-08-26b — XX2 and C36: two schema gaps closed, and a measured refusal

**Schema v12 → v13, both additions INERT and appended, so a v13 database renders bit-identically
to v12.** `verify.py` **381 PASS / 1 baseline FAIL**, 14 audits green, C++ clean on 18 TUs, no film
index moves.

**XX2 — `DevelopmentPoint.base_fog`.** ⚠ What this closes is a **silence, not a wrong number**.
`ToneCurve.dmin` is one value and therefore describes ONE development condition — but nothing in the
schema said which, and nothing said fog moves with development at all. It does: DOUBLE-X 5222's five
traced curves give base+fog **0.231 / 0.233 / 0.233 / 0.275 / 0.296** at 4 / 5 / 6½ / 9 / 12 minutes
in D-96, a **28 % rise**. These are **traced, not printed** — the sheet draws a Time-Fog curve and
puts no numbers on it. `verify.py` now asserts the link that used to be implicit: **the stored dmin
equals the fog of the stored development condition and of no other** (0.2328 at 6½ min; it would be
0.296 at 12). Same gap `ProcessingFamily` closed for contrast, closed for the other quantity the
same plot measures.

**C36 — and the result is a refusal.** Traced H-1-2254's MTF panel (694×605 raster, log-log axes
from 10 frequency and 12 response gridlines, worst residual **0.008 decades**). ⚠ **Two of the three
records never reach 50 % response:** the curves stop at **82.2 cycles/mm** with green still at
53.1 % and red at 50.6 %. Only blue crosses, at **51.9**.

⚠ **So the stored 72.0 estimate is wrong in both directions at once** — too sharp for blue, too soft
for the two proven ≥ 82.2 — and no single number is right about a set spanning a factor of 1.6.
`PrintStock` gains `mtf_f50_r/g/b` + `mtf_f50_bound` + `mtf_measured`, with **0.0 meaning CENSORED**
and the bound carried beside it: the same idiom `DyeStabilitySpec` introduced at v12, because it is
the same problem. ⚠ **The legacy scalar is left at 72.0 on purpose** — it is what the reference
renderer reads, and changing it would move a render on the strength of a number the sheet does not
state.

⚠ **No rolloff exponent stored, and that refusal is measured too.** Blue's traced span reaches only
36–82 cycles/mm, so a carrier normalised at f = 0 would be fitted over 0.36 decades with just
**0.16 of them below f50**. The fit is good where it sits (q 1.78 at rms 0.026, 2.8× better than the
Gaussian) and says nothing about the knee, which is what q means.

⚠ **A tracing trap found on the way, and it is the reusable part.** A log-log MTF panel is ruled at
1/2/3/5/7/10/20…; those rules are **1 px thick where the curves are 3**, and through the flat
low-frequency half of every MTF curve a tracker cannot tell them apart by position. It followed the
**100 % and 70 % rules** instead of the traces. Measured cost before `_strip_gridlines`: blue's q
came back **0.74 at rms 0.063** against **1.78 at rms 0.026**. Both look like fits. Only one is of
the curve. Thickness is the discriminator, and where a curve crosses a rule the merged run is
thicker still, so the curve survives its own crossings.

**Still estimated on 2254 and unfixable from this sheet:** `grain_rms`. H-1-2254 publishes no rms
figure at all — only that granularity is "similar to VISION Color Intermediate 2242", a stock this
database does not hold.

### 2026-08-26 — EASTMAN DOUBLE-X harvested from a second edition of a sheet we already had

**Owner supplied `EASTMAN DOUBLE-X Negative Film 5222.pdf` = H-1-5222 revised 7-15 (July 2015).**
`verify.py` **375 PASS / 1 baseline FAIL**, 14 audits green, C++ clean on 18 TUs.

⚠ **THE VALUE OF THAT FILE IS ITS ART, NOT ITS CONTENT.** The corpus already held H-1-5222
revised 3-26, and the two print the *identical figures* — same plot numbers, F010_0029AC and
F010_0031AC. But the 2026 edition draws them as **rasters** and the 2015 edition draws them as
**vector paths**. Panels that had to be read by hand became measurable. The lesson generalises and
is worth keeping: *when a sheet resists, look for another printing of the same sheet.*

**MTF measured for the first time.** f50 **42.2** cycles/mm, rolloff q **2.88**, printed adjacency
overshoot **+25 %** peaking at 4.1 cycles/mm — replacing the flat estimate 56/56/56, which was
**1.33× too sharp**.

⚠ **And it gains a cross-check no estimate could have had.** PLUS-X 5231 is the corpus's other
Kodak black-and-white cine negative and was traced from its own sheet: **41.3** against DOUBLE-X's
**42.2**. Two sheets, two independent traces, two speeds of one design family, **2 % apart**. The
old *estimated* pair read 56.0 and 60.0 — which agreed with each other for no reason at all.

q is adopted here at +25 % overshoot where 5279 was refused at +42 %. The discriminator is the
**fit**, not the overshoot: rms 0.076, inside the 0.0095–0.132 band of every accepted curve, where
5279 returned 0.25–0.34 and was put back on the Gaussian.

**Spectral sensitivity re-traced, and the criterion read off the panel.** ⚠ The panel draws **two**
curves — "D = 0.3 Above Gross Fog" and "D = 1.0 Above Gross Fog", about **0.55 decades apart** — so
a reader that took "the curve" would have stored whichever the page emitted first, silently. The
adopted set is matched to its printed caption by geometry. It agrees with the 2026-08-02 raster
reading to **rms 0.037 decades** on the same 430 nm peak: confirmed rather than corrected, but now
machine-derived with its residuals on record.

**A level error found in a figure this project had already traced.** ⚠ Base+fog **0.1977 → 0.2328**.
The 2026-08-02 raster trace of this same curve had the *shape* right — its gamma is within
**0.0004** of the vector refit and it reproduces the vector path to rms 0.0123 D — and the *level*
wrong by 0.035 D. Confirmed two independent ways before changing anything: calibrating the density
axis from the five printed **ticks** gives 0.2369, calibrating from the **frame edges** gives
0.2281, and the stored 0.1977 sits outside both. The mid-grey anchor was re-checked afterwards
(D 1.1786 against the recorded 1.178), so this is a level correction and the exposure axis has not
moved. Shoulder parameters were deliberately **not** re-fitted: no shoulder exists in the plotted
range, and refitting would have produced new numbers from the same absence of data.

⚠ **The developer was wrong, and that is the substantive correction.** `_PROCESSING` said **Kodak
D-76**, from Иофис 1964 table 7. Kodak's own sheet says **D-96 at 21 °C** in three separate places:
the PROCESSING table, the sensitometric caption and the MTF caption. D-76 is a still-film
developer; 5222 is a motion-picture stock. The Иофис row is **kept** for what it actually evidences
— 1963-64 local practice, plus its independent confirmation of the ASA 250/200 pair and the 0,6–0,7
gamma band — rather than deleted. Method rule 4.

**ProcessingFamily populated** from the five printed per-curve gamma labels (4 min 0.50, 5 min 0.56,
6½ min 0.66, 9 min 0.84, 12 min 1.05, D-96 at 21 °C) — the fourth stock in the database to carry a
processing axis. New audit **`kodak_time_gamma.py`** re-derives them from the *drawn* curves and
reproduces four of five within 2 %. ⚠ The **9-minute** point does not: measured **0.798** against a
printed **0.84**. It is also the most window-sensitive of the five (0.744–0.813 as the fitted
density interval moves, where every other curve moves by less than 0.012), which says the curve is
not straight over the interval being fitted — and Kodak does not print the interval *their* gammas
were measured over. Stored **as printed**, disagreement recorded as a named exemption so that a new
disagreement on any other curve fails instead of hiding in a loose tolerance.

**One new gap found and queued (XX2):** the sheet prints a **Time-Fog** curve, and the trace
measures base+fog rising 0.231 → 0.296 across the five times. The schema has no carrier for fog
against development time, so every stock's `dmin` is silently a statement about one development
condition. Same class of gap `ProcessingFamily` closed for contrast.



### 2026-08-25h — task 5 closed: G7, the panel finder, and C15 unblocked by schema v12

**Queue G7, C15 and E0b-orig's remainder all closed. Three defects found in this project's own
readers, one of which would have been invisible.** `verify.py` **364 PASS / 1 baseline FAIL**,
12 audits green, C++ clean on 18 TUs.

**G7 — Gevacolor 682's dye set, empty on purpose since 2026-08-19, is now measured.** The entry had
recorded "the dotting merges into components the solid/dashed threshold classes as solid". ⚠ **That
diagnosis was not the blocker.** The real one: the dotted CYAN curve and the dash-dot MAGENTA curve
**cross at about 425 nm**, and for roughly twelve pixel columns they are one ink run. The tracer
accepted that merged run into its slope history; the merged ink is nearly flat, so the descending
track's fitted slope collapsed from +0.75 to +0.3 px/column, and when the curves separated the
flattened prediction landed on the wrong branch. **The two curves came back swapped with every
residual still small** — two smooth curves, just not the two that were printed.

The fix is `merge_px` in `dashtrace.trace_predictive`: at any column where two live tracks predict
within `merge_px` of each other, **neither may claim ink** — both coast on the slope measured before
the merge. Refusing to decide is the correct answer at a crossing; the ink genuinely does not say
which curve it belongs to. Default 0.0, so no existing caller changes.

The paper validates the result from outside, which is why it is adopted:

| | traced | printed in the paper |
|---|---|---|
| yellow | **1.462 D @ 445.9 nm** | 1.46 @ 448 |
| magenta | **1.474 D @ 522.1 nm** | 1.48 @ 525 |
| cyan | **1.459 D @ 683.1 nm** | 1.46 @ 687 |

and the figure's own C / M / Y letters sit at 448 / 528 / 683 nm, one above each traced peak. ⚠ The
layer names come from the **peaks**, never the seed order: at the left edge the cyan curve is the
**top** one. ⚠ `verify.py`'s guard was **inverted** in the same edit — it used to assert the set
stayed empty; it now asserts the set is present and reproduces the printed peaks.

**The panel finder — 6 pages reachable became 21.** `EASTMAN_EKTACHROME_7239`'s spectral set is
adopted, and getting there cost three independent fixes, none of them in the source:

1. **The caption finder could not see short words.** `rot_labels` calls a word rotated when
   `(y1-y0) > 1.6*(x1-x0)` — true of "SENSITIVITY", **false of "LOG"**, which is three characters
   tall. The pair never met. Replaced by PyMuPDF's per-line writing direction, which is not a
   heuristic at all. Corpus sweep over all 2159 pages: **6 pages found by the old rule, 21 by the
   new one** — 15 more sensitivity panels are now reachable (5231, 5245, 5246, 5248, 5274, 5279,
   5293, V200T, both 5205 sheets, 5218, the 5219 brochure, 8532, eterna_vivid500, 7239).
2. **The frame picker took the nearest frame and stopped.** On 7239 two rects qualify and the tick
   labels sit **between** them, so the nearer frame is uncalibratable. Candidates are now tried in
   order until one calibrates.
3. ⚠ **The y axis has a minus sign that is not in the text layer**, and this was the dangerous one.
   The axis runs 2.0 / 1.0 / 0.0 / −1.0 / −2.0 with the negatives set as **overbars**, so PyMuPDF
   returns "1.0" and "2.0" twice each. The old code keyed ticks by value with `setdefault` and
   silently dropped the duplicates — **and happened to keep the right branch**. An identical sheet
   emitting the lower branch first would have calibrated **mirrored**, still perfectly collinear,
   still inside tolerance, with every stored sensitivity carrying the wrong sign and nothing able to
   see it. Ticks are now signed by position about the zero tick, and the five-tick collinearity
   test is what confirms it (0.46 pt worst residual).

⚠ **7239 is also the first set read WITHOUT the ink rule.** Its panel is printed entirely in black,
so Kodak's colour-of-light convention says nothing about it. Assignment rests on the absorption
bands, the ascending peak order, and the panel's own in-frame captions ("Yellow-Forming Layer" at
394 nm, "Magenta-" at 550, "Cyan-" at 702) — **one fewer independent check than an inked panel
gets**, which the profile states rather than leaves to be inferred. Peaks 410 / 560 / 660 nm.
⚠ And this sheet **prints its density criterion** — "Process: VNF-1", "Density: 1.0",
"Densitometry: E.N.D.", "Effective Exposure: 1.4 seconds", all inside the frame — where the four
older Kodak sets carry a "D 0.2 above dmin" that appears on **no sheet in this corpus**.

**C15 — unblocked, and the owner lifted the schema freeze. Schema v11 → v12.**

* `PrintStock` gains **`aging`** (an `AgingSpec`). It had none: the struct existed only on
  `FilmProfile`, so a positive stock had nowhere to record storage damage. A gap, not a decision —
  the same shape the v7 bump closed for `dye_density` on this same struct.
* `PrintStock` gains **`dye_stability`**, a **new** `DyeStabilitySpec`. ⚠ **A new struct on
  purpose.** `AgingSpec` stores how much fade a stock has *suffered* (fractions, zero = fresh); an
  Arrhenius prediction states how *long* a fade takes. Writing "86 years" into a field documented as
  a 0–1 fraction is the same category error that stalled this item once already.
* ⚠ **The published figures are CENSORED and are stored as censored.** Kodak prints ">100" for every
  record that outlives the test. `censor_years = 100.0` with the field at `0.0` means "greater than
  the bound"; storing the number 100 would let later arithmetic average a bound as if it were a
  measurement. Exactly **two** entries at 21 °C are finite: **yellow 86 years** to a 0.10 density
  loss (colour separations) and **blue 77 years** to a 0.1 D-min *gain*. The 7 °C column is entirely
  censored and is therefore not stored at all.
* Both fields are **INERT** and appended after every v11 field, so a v12 database renders
  bit-identically to v11 — but `PrintStock` **grew**, which is why the version moved.

**`KODAK_VISION3_DI_2254` added** as the 11th print stock, appended so no existing index moves. Its
curves are **traced, not estimated**: `di_2254.py` is a fifth curve reader, for a case none of the
other four covers — a *modern Kodak brochure whose plots are nonetheless raster*. 474 samples per
record off the 680×704 px figure, axis residuals 0.0015 decade and 0.0000 D (the plot is
axis-aligned to the pixel and its ticks fall on exactly uniform 135 px spacings, so the calibration
is arithmetic rather than a fit), fit rms 0.006–0.012 D.

⚠ **The gammas are the check that costs nothing and proves the calibration:** an *intermediate* film
exists to change nothing, and the fit — told none of that — returns **1.05 / 0.96 / 1.04**.
⚠ **The origin is the sheet's, not mine:** the abscissa is absolute ("LOG EXPOSURE (lux-seconds)")
while `ToneCurve`'s x is relative, and the sheet quotes its dye-stability table at "1.0 Above
D-min", so x = 0 is where the green record reaches D-min + 1.0 (traced logE −1.962).
⚠ **Blue and green share one D-min (0.711) because the sheet draws them as one stroke** below
logE −3.0 — the source's statement, not a tracing failure.
⚠ **Catalogue-number hazard, now asserted by `verify.py`:** `EASTMAN_5254_1968` is a **1968 ECN
camera negative** with the same four digits. Two different films, one number; neither may cite the
other's document.
⚠ **The table is stored on this film and nowhere else.** One film cannot establish a fade rate for a
class (method rule 18), and a DI recording film's couplers are chosen for archival stability rather
than camera exposure — the same refusal made for the 7266 σ(D) two days earlier.

**Still open on this stock, and small:** `mtf_f50` and `grain_rms` are **estimates**. H-1-2254 p5
*does* print MTF curves, but they stop at ~80 cycles/mm with red and green still at roughly 50 %, so
the sheet does not reach f50 for two of three layers; what it bounds is blue crossing 50 % between
50 and 60. The sheet publishes no rms figure at all — only that granularity is "similar to VISION
Color Intermediate 2242", a stock this database does not hold. Queue item **C36**.

### 2026-08-25g — the first reversal MTF, a 1.95x correction, and a blocked adoption

**Task 5 started. Queue E0b-orig partly closed; C15 stopped on a schema finding I should have
checked before recommending a course of action.**

**`KODAK_EKTACHROME_100D_5285`'s MTF traced and adopted** — the first measured MTF for a colour
REVERSAL stock here; every other traced sheet is a negative.

| | stored (estimate) | measured |
|---|---|---|
| f50 R / G / B | 74.0 / **82.0** / 90.0 | **27.2 / 42.1 / 60.9** |
| rolloff q | — | 1.87 / 2.39 / 2.52 |

⚠ **The largest MTF correction the project has made: the stored green was 1.95x too sharp.** Red and
blue were the estimating rule's fixed ratios about that wrong centre. Layer order comes out
R < G < B, the second independent confirmation of the docstring's prediction after 5201, and the
power law beats the legacy Gaussian on all three records.

⚠ **AND IT BOUNDS A FAMILY CONSTANT.** C24 found measured red clustered at 36.4 c/mm across the
Kodak cine negatives. This reversal red is **27.2** — 25 % below, and it would have taken the
cluster's spread from 25 % to 41 %. The guard now excludes reversal stocks, the same refusal C24
made for the Fuji sheets. **Nothing had licensed assuming that anchor reached past the negative
family, and it does not.**

**⚠ E0b-orig was mostly finished a week ago and never struck.** 7239's dye set, 5231's MTF and
5285's sigma(D) were all adopted on or before 2026-08-18, while the entry still described them as
to-do — so it sat on the ready-now list for a week. A closed item that still reads open costs the
same as a wrong one.

**⚠ C15 IS BLOCKED, AND ON SOMETHING I SHOULD HAVE CHECKED BEFORE RECOMMENDING IT.** The owner
answered three questions on the strength of my proposal to give the VISION3 DI film its own
profile. Reading the schema afterwards: `StockKind` has only NEGATIVE and REVERSAL — there is **no
intermediate category** — and the DI/dupe concept already lives in `PRINT_STOCKS` (`SCAN_DI`,
`DUPE_FINE_GRAIN`). But **`PrintStock` has no `aging` field**; `AgingSpec` exists only on
`FilmProfile`. So the only way to attach the measured Arrhenius table today is to declare a
digital-intermediate recording film a camera negative, which would put it in the camera ListBox and
shift 160 indices — a category error committed to obtain a field. Stopped rather than forced.

**The data itself is confirmed** from the full 5-page sheet (H-1-2254, March 2026), which agrees
with the 2-pager: colour separations yellow **86 years**, D-min blue **77 years** to 0.1 gain, at
21 C; everything else >100, and everything >100 at 7 C.

### 2026-08-25f — four gates, and the bypass rate on the shared-law surface was 2 of 2

**Queue C32–C35. No new film data; four checks that make yesterday's class of failure reportable.**

**C32 — the sweep.** After C30, the question was how many other laws were bypassed. The shared-law
surface is exactly two functions: `film_sim.py` calls `fp.grain_sigma` and `fp.mtf_response` and
nothing else, and the generator emits exactly those two. **Both were unreachable from the stages.**
Not a sample — the whole surface. `check_law_reachability()` now fails the build on any published
law with no caller, and equally on a recorded bypass that quietly gains one.

⚠ **Its first run passed on a comment.** `FilmGrainSigma` reported "reached from 1 stage source" —
the source being the comment explaining that it is *not* called. Comments are stripped now. A gate
that passes on prose about its own failure is the defect it exists to catch.

**C33 — the twins diverged within hours, and I caused it.** Fixing grain on the scalar side only
left the two C++ paths **1.039×–1.183× apart on amplitude** — a difference in the model, not the
vectorisation, which this project's own AVX2 rules forbid. Mirrored into the AVX2 twin at zero
inner-loop cost (it folds into `gain` before the broadcast), verified by compiling under
`AlgoType = float` with `-mavx2 -mfma`, header restored to `double` immediately.
`check_twin_consistency()` now asserts that law-bearing tokens appear on both sides.

**C34 — a documentation gate.** `build.py` had gated one claim (the `PROGRESS.md` stamp) and it had
never fired. `doc_consistency.py` generalises it to a registry of load-bearing sentences checked
against live expressions. ⚠ A pattern that stops matching **fails** — an unmatched pattern silently
stops checking, which is the state it exists to end.

**C35 — the project-root `doc/` folder, reviewed for the first time and never once delivered.**
Seven files, 2661 lines, outside every archive ever shipped. Three still called stages 15, 16 and
09b stubs when all three render. `STAGE_FUSION_PROPOSAL`'s central 4K memory argument is superseded
**by its own project** — it quotes the pre-ping/pong footprint, and M1 already went further than
fusion would have. ⚠ And neither AVX2 document mentions that the vector build requires
`AlgoType = float`, while the shipped header says `double` and 17 AVX2 units static_assert against it.

**Also: the closed-loop tier widened from a 5-stock sample to the whole database** — f50 modulation
and characteristic-curve reproduction now run on all 160 stocks (0 outliers, so the tolerances are
what the code achieves rather than headroom).

### 2026-08-25e — the C++ was rendering grain 4–18 % loud, and every check passed

**Queue C30 and C31. The largest single accuracy defect found in this project, and it was in the
shipped plugin.**

`film_profiles.hpp` defines `FilmGrainSigma()` as THE ONE DEFINITION — the legacy law **divided by
`sqrt(1 + fog_grain)`**, so the shape is exactly 1.0 at NET density 1.0, plus the measured-anchor
branch. **It had zero callers.** `AlgoAddGrain` inlined its own square root without the
normalisation.

| population | C++/Python ratio |
|---|---|
| 147 legacy-branch stocks | **exactly `sqrt(1 + fog_grain)`** — 1.0392 to 1.1832, mean **1.1027**, reproduced to 3.0e-08 |
| 13 measured-shape stocks | 0.39× in shadow to 2.2× at depth — **inverted** on the two reversal stocks |

The 13 are the serious half: not "slightly loud" but **grain distributed wrongly across the tone
scale, in opposite directions for negative and reversal**, all crossing 1.0 near net density 1.0 —
which is exactly why a mid-grey spot check looked fine.

⚠ **WHY IT SURVIVED FOR WEEKS.** `cpp_parity.py` called `FilmGrainSigma()` **directly**. It agreed
with Python on every stock and never touched the function that renders. Third instance this month of
a guard aimed at the wrong subject, after C20 and the AVX2 compile gate.

**Fixed** by applying the normalisation in the stage, hoisted out of both loops — it depends only on
`fogGrain`, which the stage already receives, **so no shared signature changed and the AVX2 twin
still compiles untouched**. The stage now returns exactly 1.0 at net density 1.0 on all 160 stocks ×
3 channels; the 147 legacy stocks agree with the reference to **4.3e-09**.

⚠ **The measured SHAPE is still unreachable from the stage** (13 stocks). It needs the `GrainSpec`
and `dmax`, i.e. a shared signature change with the AVX2 twin moving in the same commit — scoped as
its own change. Those 13 now get the correct level and the legacy shape: worst error falls from
0.39×–2.2× to a pinned 1.73 at net 2.5, with net 1.0 exact. Strictly better, still not right.

**C31 — two validation tiers, both of which would have caught this on their first run:**

* **A stage-level parity family** that compiles and calls `AlgoAddGrain` itself, 2400 probes, with
  the amplitude recovered exactly (`amp = out − D` at unit field and unit gain). It asserts the
  net-1.0 identity and judges the two populations separately, so the scoped shape gap can neither
  grow nor silently close.
* **A closed-loop tier** in `verify.py`: render, measure back through the manufacturer's own
  convention, compare to the published number. Added a sinusoid at f50 returning exactly 50 %
  modulation, and the rendered characteristic curve reproducing the stored curve.

⚠ The f50 check needed its sampling fixed before it was trustworthy: at an arbitrary rate the sine
leaks across FFT bins and peak-to-peak stops measuring modulation — two stocks read 0.559 and 0.590
and looked like real failures. px/mm is now chosen so f50 lands on an exact bin.

### 2026-08-25d — a validation pass, a renderer parity fix, and a documentation audit

**Queue C17, C20 and 5248's half of B1 closed; C16 narrowed to one number. And the pass corrected
work done hours earlier the same day.**

**The cross-validation.** `KODAK_VISION2_200T_5217` already carried a spectral set from the
2026-08-02 RASTER batch. Re-deriving it from the same sheet's VECTOR paths agrees to rms
**0.109 / 0.091 / 0.049** decades (r/g/b) with peaks within one 10 nm step. Neither side corrected:
a wash is not a reason to churn adopted data. Both methods are now credible on evidence.

⚠ **AND IT CORRECTED 2026-08-25c IMMEDIATELY.** That entry said 5201's blue layer "peaks at 470
where its siblings peak at 410–420" — from comparing **one** sibling. The family splits **6/4**:
470 nm on 5201, **5217**, 5205, 5203, 5274, 5246; 410–440 on 5218, 5279, 5219, 5213. 470 is the
majority and 5201 matches 5217 exactly.

⚠ **THE CRITERION QUESTION GOT WORSE, AND IS THE ONE DECISION WAITING.** A sweep for a printed
density criterion found five sheets that print one: **5246 p5, 5274 p4 and V200T p4 all say
"0.4 above D-min"**; the DI/intermediate sheets say 1.0. **No sheet prints 0.2** — yet 16 profiles
store `D0.2_above_dmin` and 10 store a printed D0.4. The split follows the sheets exactly: sheets
that print a number → 0.4 stored; sheets that say only "specified density" → the 0.2 that appears
nowhere. Nothing changed (16 provenance claims is the owner's call); the counts are pinned.

**C17 — the one-sided gate.** C++ has always gated both coupler components below 0.25 px; Python had
no gate, so below that scale one renderer ran the stage and the other did not. Python now carries the
same gate at the same threshold — **adopted from the shipped constant, not chosen**, which keeps a
fidelity judgement out of a parity fix. Parity unchanged at worst 5.335e-05.

**C16 — narrowed, not closed.** The two blurs are still different forms (analytic transfer vs
truncated separable kernel), agreeing to 6e-5 only above ~1.2 px and diverging to 1.5e-1 at 0.4 px,
while stored `edge_um` sits at 0.36–0.60 px at 40 px/mm. What C17 removed was the "whether"; what
remains is the "how", i.e. the shared threshold's value. Recommendation on file: raise it to ~1.0 px,
which is also the honest model statement — a 9–13 µm feature at 25 µm/px is below the sampling limit
and rendering it anyway aliases a sub-pixel feature. It changes every render, so it is the owner's.

**C20 — a guard that could not fail.** "interimage leaves a neutral untouched" rendered 0.18, the
anchor, where the correction vanishes identically for *any* interimage matrix. Renamed; a second
guard now pins the off-anchor movement (grey 0.45 → 15.9/255, grey 0.06 → 6.5/255) as the mechanism
it is. The docstring is qualified: vanishes **at the anchor**, not on neutrals in general.

**B1 — 5248 was never a failed extraction.** Its panel prints "Typical densities for a midscale
neutral subject and D-min." and draws exactly those two traces. There are no separate dye curves on
that sheet, so the schema's cyan-AND-magenta-AND-yellow requirement can never be met from it — the
same mismatch already recorded for `FUJI_SUPER_F125_8532`, now with a second instance. **5246 stays
open with a sharper reason:** 7 traces for 5 labels, the label-nearest Cyan peaking **0.943** against
the sheet's own "peak-normalized" claim, and two unlabelled traces unaccounted for.

**The documentation audit, because two of today's errors were in prose rather than data.** Four
hardcoded counts in `gen_active_profiles.py` were wrong — ISO 6 **27 → 51**, ISO 5800 **34 → 58**,
ISO 2240 **13 → 17**, manufacturer EI **15 → 34** — and are now derived from the database. Also
corrected: "7 curves on 3 sheets" → 26 on 12; a claim that `ReciprocitySpec` is read by no renderer
(C8 closed 2026-08-23); "39 raster granularity pages … unread"; "all 395 documents in `PDF/PROFILES`"
against a measured 448. `NotFound.md` lost nine stale claims, including three rows still saying "no
profile" for stocks added on 2026-08-24 and a row listing four already-closed dye sheets as failures.
⚠ `gen_film_curves_md.py`'s `QUEUED_PLOT_ON_FILE` had been **empty since 2026-08-02**, so the report
printed "no plot in archive" for five stocks whose plots are in the archive with page numbers in
`NotFound.md` §4.1.

### 2026-08-25c — H-1-5201 finished, and two recorded facts turned out to be wrong

**Queue items C9, C10 and C12, all closed against documents already on disk. No acquisition.**

**C9 — the dye panel, and the queue's own diagnosis was the obstacle.** C9 blamed
`dye_density.py`'s family classifier ("3 dyes, or 3 dyes + neutral, not 3 + neutral + dmin"). It was
never that: family B takes any three of however many curves it is given. **The cyan trace never
reached the classifier** — Kodak draws it as a yellow-under-magenta overprint, two bit-identical
paths of **7 segments each**, and the `n < 8` segment filter dropped both, so nothing was left in
the 615–700 nm band for a triple to pass the band test on.

**Fix: identify traces by INK.** Kodak's rule is physical, not decorative — each trace is drawn in
the colour of light it concerns. Yellow dye (absorbs blue) drawn BLUE, magenta drawn GREEN, cyan
drawn RED via the overprint. Read off the panel's own legend swatches. Lowering the segment
threshold instead would admit gridline stubs on every other sheet.

**And a new validator, which is why the set is tier 1.** With the dyes peak-normalised and the
neutral as-printed, family A's `neutral = C+M+Y` cannot hold. What must hold is

| fit | coefficients c / m / y | rms |
|---|---|---|
| `Neutral − Dmin = k_c·C + k_m·M + k_y·Y` | **0.628 / 0.604 / 0.595** (5.4 % spread) | **0.019 D** |
| same without the Dmin term | 0.855 / 1.220 / 1.609 | 0.085 D |

Equal coefficients are what make the result a *visual* neutral, and they came out of an
unconstrained fit. The 4.5×-worse alternative is what identifies which dark trace is the neutral and
which the dmin. Adopted peaks **450 / 540 / 680 nm — identical to 5217 and 5218.**

**C10 — the first VECTOR-traced spectral set in the database.** New `spectral_vector.py`, in
`build.py`'s audit stage. Same ink rule, assignment checked three ways that are not the ink: legend
swatches, absorption bands (**470 / 540 / 650 nm**, ascending), and the sibling sheets — red and
green agree with 5217/5218 to rms 0.05–0.14 decades.

⚠ **The blue layer peaks at 470 nm where its siblings peak at 410–420.** That is the whole of the
cross-check disagreement (blue rms 0.24–0.42) and it is printed: a narrow cusp above log S 2.0,
confirmed on a 26× render. Pinned, because "correcting" it toward the family would undo a
measurement.

⚠ **THE CRITERION IS PRINTED ON NO SHEET IN THIS FAMILY.** 5201 says only "reciprocal of exposure
(erg/cm²) required to produce **specified density**". The three sets already stored carry
`log_reciprocal_erg_cm2_D0.2_above_dmin` — and 5218 and 5217 print the same unspecified wording
while 5219's footnote is not in its text layer at all. 5201 stores what its sheet prints; the other
three are left alone and the conflict is recorded (method rule 4).

⚠ **AND THIS ONE MOVES A RENDER.** The dye set is inert; spectral sensitivity is not. A stock
carrying it takes `spectral_balance_gains()` instead of the 600/550/450 nm proxy, and a red layer
peaking at 650 rather than 600 means tungsten drives it harder: **+0.28 stop of red gain at
3200 K**, −0.17 at 10000 K, green the unchanged anchor. Asserted in size and direction.

**C12 — filed against two profiles, and there were six.** A sweep for `[T1/T3]`-style tags found
`KODAK_VISION2_500T_5218`, `_200T_5217`, `_250D_5205`, `KODAK_VISION_500T_5279`, `_200T_5274` and
`_250D_5246` all resolving to **tier 3 on `fitted_from="analogy"`** — every one owning its own Kodak
sheet, four with a σ(D) shape traced from it, and in all six the T3 half is one scalar
(`rms_granularity`, because from VISION onward Kodak prints curves and no rms number). All six to
tier 1, owner-approved. ⚠ **Closed by a class guard, not by loosening the regex:** a mixed tag must
now be listed in `_UNTAGGED_TIER` and may not resolve to 3, or the build fails.

### 2026-08-25b — the first measured B&W σ(D), and it reverses a stored sign

**Kodak TRI-X Reversal 7266's granularity panel traced and paired against its own characteristic
curve. Queue item C29; full account in `NotFound.md`.**

**Why this sheet and no other:** both plots share one log-E abscissa, so σ and D pair without a
second document and without transferring a calibration between sheets. 52 columns paired, **30
kept** — the restriction is |dD/dlogE| > 0.5, because on the flat parts of a characteristic curve one
density maps to many σ and the pairing is ill-conditioned. On a reversal stock the flat part is
**dmax**, not the toe.

**ADOPTED on `KODAK_TRI_X_REVERSAL_200` only:**

| anchor | measured | at D | stored estimate |
|---|---|---|---|
| `sigma_shape_toe` | **0.262** | 0.352 | 0.70 |
| `sigma_shape_mid` | 1.000 | 1.000 | 1.00 |
| `sigma_shape_dmax` | **2.829** | 3.089 | 0.50 |

`σ_D ∝ D^1.078`, rms 0.038 decades. ⚠ **The estimate was pointing the wrong way** — it had
granularity falling 2× toward dmax where the sheet shows it rising 2.8×. On reversal film dmax is
the *unexposed, fully developed* silver, so rising is the physical direction; the estimate was a
negative film's shape pasted onto a positive film.

⚠ **THE LEVEL IS NOT ADOPTED.** The panel implies **22.3** at this file's NET-1.0 convention against
a stored **10.0**. The sheet says the curve uses "modified measuring techniques" and does not define
them, so only the SHAPE is grounded. rms stays 10.0; the 22.3 is cited in the profile.

⚠ **The apparent 2.93× interior peak at D 3.16 is discarded** — it sits inside the ill-conditioned
flat zone. Re-adding it from the raw trace would be storing an artefact.

⚠ **Scope held to one stock.** The other **34** reversal stocks stay on the contradicted estimate —
one measured sample is not a class (method rule 18). The 68 monochrome **negative** stocks are
untouched for the stronger reason that a reversal emulsion's rising shape must not be transferred to
them at all. `verify.py` gained four guards: the shape is pinned, no interior peak, rms 10.0 kept
with the "modified measuring techniques" citation required, and the 34-stock count recorded so a
later pass cannot quietly harmonise them.

### 2026-08-25 — a retraction, and grain size turns out to depend on development

**T-101 Figs. 20/21/23/24/26 read. Queue item C28; full account in `NotFound.md`.**

⚠ **RETRACTED, same session it was produced.** Fig. 26 was extracted cleanly — `log10(t̄/σ) =
-0.6648·log10(D) - 0.1738`, 1039 columns, rms 0.0063 decades, self-validated by five ✕ markers
landing within 2.2 % of densities known independently from §B.2's printed transmissions, and
cross-checked against Fig. 21 on linear axes (0.668 vs 0.665). **And it is still not convertible to
σ_D.** T-101 §2 builds its σ from a two-level opaque-grain model, σ = √(t̄(1−t̄)), approached *as the
aperture becomes vanishingly small*; the measured σ_t/t̄ runs 0.39 → **1.64**, so the small-signal
linearisation σ_D = 0.4343·σ_t/t̄ is invalid everywhere on the plate. The mid-session result
"σ_D = 0.648·D^0.665" is withdrawn.

✅ **The Mees Fig. 302 conflict never existed.** Mees is Goetz–Gould G at a fixed densitometer
aperture — the Selwyn regime, which is where this file's 48 µm `rms_granularity` lives. Fig. 26 is
the pinhole limit. Not commensurable. The σ(D) question is exactly where it was.

**ADOPTED, from printed Table 3 — no tracing needed:** `D_eq ∝ γ^n`, n = **0.452** (Pan F, rms
0.0035 µm), 0.396 (Tri-X), 0.425 pooled. The table's own last column normalises by √γ and
reproduces to three decimals. Validated at 2 % against Table 2.

**That exposed a condition mismatch shipped the day before.** `clump_um` came from Table 2's
diameters, measured at *the BBC's* development gamma; `ILFORD_PAN_F` stores γ 0.55 against their
1.0. **0.859 → 0.655 µm.** `EASTMAN_PLUS_X_5231` (0.68 vs 0.64) deliberately not moved — +2.5 % is
inside the source's own upper-bound caveat.

⚠ **Both dependences are now in the `GrainSpec` docstring:** `clump_um` varies with development
gamma (γ^0.42–0.45) *and* with density (−20 % across the tone scale, T-101 Fig. 21). The schema
stores one scalar, so every stored value is a mid-scale representative at one development
condition.

### 2026-08-24 (second pass) — the F-125 family, settled by a printed sentence

**`FUJI_F125_8630` removed; `FUJI_F125_8530` and `FUJI_SUPER_F125_8532` kept as independent
profiles.** Queue item **C27**, full account in `NotFound.md` §1.5. Two issues of «Техника кино и
телевидения» (1989 №4, 1990 №1 — the latter a translation of Fuji's own symposium paper) print
**Fuji's four-digit code rule in words**: first digit 8 = colour negative, **second digit = gauge**
(5 = 35 mm, 6 = 16 mm), last two digits = the film. So 8530/8630 were one emulsion slit two ways —
a gauge is `default_format`, and `8630` is now an alias. The same rule keeps 8530 and **8532**
apart: they differ in the *last two* digits and measure differently, rms **4.0** vs **3.0** at
identical speed.

**Adopted:** 8530 `rms_granularity` 5.4 (estimate) → **4.0**, printed in 1989 №4 Table 1 p70, read
off the page image; 48 µm at visual diffuse D 1.0, the same definition the 8532 sheet uses.
Cross-checked against 1990 №1 Fig. 4 (0.0036–0.0040 at D 1.0).

⚠ **A third MTF measurement arrived and nothing was changed.** 1990 Fig. 3 traces to f50 ≈ **33**
mm⁻¹ against the 1989 table's 0.60 at 30 and Honjo's **ν₅₀ = 42**. It reframes the conflict already
on record for 8532 (Coltman 32.07): two of three sources now cluster at 32–33. Method rule 4.
⚠ **Three figures left unharvested with stated reasons** — σ(D) (curves merge at the validating
anchor), gammas (F-125 drawn *superimposed* on type A, and no numeric abscissa), spectral
sensitivity (no density criterion printed).

### 2026-08-24 — the grain-size column, and it was wrong by an order of magnitude

**T-101 Fig. 18 digitised, then set aside in favour of a printed table.** Four `clump_um` values
corrected and three profiles added; full account in `NotFound.md` and queue items **C25 / C26**.
Stored values come from T-101 **Table 2 p28**, which prints the measured equivalent grain diameter
of all six emulsions, through `D_eq = 1.7473 · clump_um`:

| | HPS | Tri-X 5223 | Plus-X | Pan F | 8374 | 5302 |
|---|---|---|---|---|---|---|
| stored now | **1.431** | **1.259** | **0.830** | **0.859** | **0.687** | **0.589** |
| stored before | 1.90 | *new* | 11.0 | 5.0 | *new* | *new* |

**HPS was not special.** All six land in 0.59–1.43 µm against 3.2–40 µm across the file (median
13). `f_hi = 500/clump_um`, so a stored 19 µm puts grain rolloff at **26 c/mm** where Fig. 18 shows
Tri-X still at half power at **290**. ⚠ The other **155 stocks were not touched** — six 1963 B&W
emulsions do not license rewriting the colour column; only the error's *direction* is now on
record. ⚠ All six values are **upper bounds** (p38: diameters "expected to be greater than the true
values"). Renders change **texture only** — `grain_reference_energy` renormalises, so
`rms_granularity` still means what it meant.

**New:** `EASTMAN_TRI_X_5223` (T2), `KODAK_8374` (T3), `KODAK_5302` (T2, a PrintStock — no ListBox
shift). Their grain blocks are measured; tone curve shape, f50 and spectral weights are flagged
estimates, and 8374's `exposure_index` is an acknowledged placeholder because T-101 leaves its
speed cells blank. ⚠ `KODAK_TRI_X_400TX` deliberately did **not** move: T-101 measured the cine
5223 at 250/320 A.S.A., not the ASA 400 still film, and `verify.py` pins the non-move.

Three working days. Older work is in the dated `RESULT_*` and `CHANGES_*` documents; this table is
the index, not the archive.

| Item | What changed | Detail |
|---|---|---|
| **HPS grain spectrum** (owner-approved, HPS only) | **The `clump_um` conflict is settled by measurement, and it was an order of magnitude.** BBC Monograph 54 Fig. 8 traced off the page image — 268 points, 1.9–116 c/mm — fits **clump_um 1.900 µm, clump_gain 0.000** at rms 0.0018 µm² against the stored 26.0 / 1.65, whose own rms is 0.862. The old pair predicted W(20)/W(0) = 0.016 where the measurement is 0.985. ⚠ **The trace validates itself three ways, none used to set a parameter:** 13 gridlines landing on a ladder that ends at exactly 0.000 and 0.701; W(60)/W(0) = 0.900 against T-101 p38's printed "falls by only about 10 %" over 0–60 c/mm; traced W(0) 0.610 µm² against Table I's printed 0.62. ⚠ **`clump_gain` 0.000 is measured, not defaulted** — the free two-parameter fit drives it to zero and T-101 p38 says the same in words (correlation "substantially confined to about ± one equivalent grain diameter"), the C2 signature again. ⚠ **A √2 units trap, and a real defect found:** `grain_shape`'s docstring called itself a power spectrum while the code uses it as an **amplitude** transfer (`apply_transfer` multiplies the noise FFT; `grain_reference_energy` squares it). Fitting the mislabelled reading gives 2.69 µm. Code was always self-consistent — **docstring corrected**. ⚠ **Texture, not level:** rms stays 19.0 because the field is renormalised; correlation length 41.7 → 20.8 µm at 48 px/mm, and at 24 px/mm in-band energy drops 86 % → 56 %, so 2K renders of this stock get finer *and* quieter. ⚠ **Scope held to HPS (rule 18):** Tri-X and Plus-X were attempted from the same figure and **rejected** — the Tri-X tracker re-followed HPS, caught by its fitted W(0) of 0.611 against a printed 0.555, and Plus-X kept only 28 points after gridline masking. The family needs T-101 Fig. 18 | `NotFound.md` §1.12 |
| **ILFORD / BBC** (owner-supplied, items A+B) | **HPS stops being estimate-grade.** Two contemporaneous BBC research documents — Monograph 54 (1964) and Research Report T-101 (1963) — measure it directly, and T-101's speed table is headed MANUFACTURERS' DATA so its ASA figures are Ilford's own relayed. **Adopted:** development gamma **0.63** (replacing the 0.62 estimate), plus citations on `ILFORD_HPS`, `ILFORD_PAN_F`, `EASTMAN_PLUS_X_5231` and `KODAK_TRI_X_400TX`. **`rms_granularity` 19.0 kept and now corroborated rather than asserted:** the measured Wiener spectrum 0.62 µm² converts through the 48 µm aperture to σ 18.5 at D 0.48 above base, and the same conversion reproduces Plus-X's and Tri-X's published values — so the agreement is the result and the stored number stays, because the field is defined at NET 1.0. ⚠ **One conflict recorded and left open:** `clump_um` 26.0 against a measured 2.5 µm, with three independent lines of evidence (and `film_sim`'s own `f_hi` formula) agreeing on 2.5–2.7. Not corrected on one stock because the field runs 3.2–40 µm file-wide while the six measured emulsions span 1.03–2.5 — if the measurement is right the error is systematic. ⚠ **Two widely quoted HPS figures are NOT in either document and are not stored:** 800 ASA / 30 DIN, and 40 lp/mm — T-101's "0 to 40 cycles/mm" is an assumed system bandwidth, and reading it as a resolving power would repeat the CTF-vs-MTF unit error C11 existed to avoid. ⚠ **The C22 Callier gap is closed as a document:** T-101 Fig. 25 measures Q vs density on Tri-X 5223 (2.34 → 2.00) at a 0.0016 sr collection angle — nearly collimated, so it is the upper bound and supports the film × geometry split rather than replacing the 1.3. `callier_q` unchanged. ⚠ **Deliberately not used:** Monograph 54 Figs. 12–16, which the author states are *computed* from Fig. 8 plus six system assumptions and whose tone dependence is an assumed D^−0.6 law. ⚠ **A repair, recorded because it nearly shipped:** an ad-hoc edit script mis-bounded a string literal and moved citations onto four wrong profiles; caught by regenerating the C++ and diffing it stock-by-stock against the pre-edit snapshot, which named the four exactly. Only intended stocks differ now | `NotFound.md` §1.12, §1.12a, §1.12b |
| **F-125** (owner-raised, highest priority) | **A wrong sentence in `NotFound.md` turned into the largest single-stock harvest in the file.** The file said "no Fuji F-125 document exists in this corpus"; the owner pointed at `Fujifilm-Super-F-125-8532-...pdf`, which is a complete Fuji sheet, **Ref. No. KB-913E, ©1999**, titled *FUJICOLOR NEGATIVE FILM F-125*. ⚠ **Root cause of the error, and it generalises:** that sheet's footer, product name and logotype are **outlined vector art**, so `get_text()` returns no product name and no date — which is also why this project had recorded "no printed date anywhere on the sheet" and dated the document from its PDF stamp. **Rule adopted: on a sheet with outlined typography, "not in the text layer" is not "not printed" — render the page.** Harvested and adopted: **three characteristic curves traced** (rms 0.005–0.009 D, replacing an 8530 transfer whose dmin was ~0.25 D high on every layer), **spectral sensitivity traced** (peaks 469/553/645 nm), **f50_g 32.07 c/mm by Coltman square-to-sine conversion** of the CTF panel (printed CTF crossing 37.78), `era` 2001→**1999**, tier 2→**1**, queue **C11 closed for this stock**. ⚠ **Two hazards recorded, not smoothed:** the sheet's exposure axis is **mis-labelled and non-monotonic** (`−4.5 −3.0 −3.5 −2.0 …`), settled at first-gridline −4.5 by physics and cross-checked against the F-500 sheet's own speed point to 0.08 stop; and the converted 32.1 c/mm **contradicts** Honjo's 42.0 for the 8530 it replaced while Fuji sells it on "dramatically increased sharpness" — both left on record. **Same method applied to the sister sheet `FUJICOLOR_SUPER_F500_8572`** in the same pass (it was the calibration source): curves, spectral sensitivity, and f50_g 56 → **20.21** c/mm; its "cyan shadows" description was **retracted as unsupported** — measured, the toes fall green, blue, red, not blue/green ahead of red | `RESULT_2026-08-23c_f125_c21_c22.md`; `NotFound.md` §0.2.1, §1.5a |
| **C21 + C22** | **Two structural gaps closed, both inert by default.** C21 (schema **v11**): `HalationSpec` gains `radius_scale_r/g/b` so halo SIZE can differ per record, not just halo strength — the kernel is now built once per record in both C++ flavours. ⚠ **All 159 stocks ship 1.0 and `verify.py` pins them there:** the path-length geometry (base 100–150 µm against an 11–16 µm pack) bounds the real ratio near 1.1, so a geometry-derived set would look measured while moving a render ~1 %. C22: **Callier's coefficient**, `D_read = dmin + (D − dmin)·(1 + specular·(Q − 1))`, with the film's scattering on the profile and the READER's directionality on a new `scanner_specular` / `scannerSpecular` control. ⚠ **The field alone was the wrong shape** — Q is a property of film × measuring geometry, not of film. ⚠ **The factor must reach three consumers or mid grey moves instead of contrast changing** (+54/255 on DOUBLE-X when it reached only two). Colour is bit-identical at any setting: a dye image does not scatter. Fourth `cpp_parity` family added, 11448 probes | `RESULT_2026-08-23c_f125_c21_c22.md` |
| **C24** | **The f50 estimating rule replaced where the measurements reach.** Five measured triples adopted (5217, 5218, 5245, 5248, 5279) with their adjacency overshoots and rolloff exponents — `mtf_measured` 3 → **8**. Two stocks (5205, 5293) take a **mixed** triple: measured green and blue, family-anchored red, flag deliberately unset. Five modern Kodak cine stocks (5203, 5207, 5213, 5219, 5246) had **red re-anchored to 36.0 c/mm**; green and blue left at estimates because the measured blues run 0.96–1.43× theirs with no consistent factor. ⚠ Scope held to stocks whose blue lies inside the measured 55–111 range — 5296 (blue 42) and all pre-1990 stocks excluded, and `verify.py` asserts 5296 keeps its 30.0. ⚠ **Render impact 7.8–45.7 per 255**, visible at preview size unlike C13. ⚠ A conflict recorded: 5245/5293 measured blue (100.5, 114.6) exceeds a stored limiting resolution of 100 that their own sheets do not print — the guard now compares green | `RESULT_2026-08-23b_c2b.md` |
| **C2b** | **Nine more colour MTF sheets: 3 sheets / 7 curves became 12 / 26, and Agfa joined Kodak.** ⚠ Most of the work was repairing the extractor: **four defects, every one returning plausible numbers** — three records emitted as ONE path (5218 read as a single curve walking blue→green→red), the log GRID handed back as 5245's green record at f50 236.8, label matching that double-claimed and then mis-ranked when a record stops early (5248 came out with red SHARPER than green), and a red fragment reporting f50 32.0 off an arc starting at 53 % response. All four found on the new `--overlay` render. **C13's layer-depth hypothesis is half refuted**: `q_R ≤ q_G ≤ q_B` holds 8/8 but the magnitudes are not constants (red 1.89–2.77 where two samples had suggested 1.84–1.89), so **q stays per-stock measured**. **C24 is answered**: red f50 is effectively constant at ~36 c/mm over seven stocks and three families, so a ratio rule cannot express it. Five measured triples are waiting on an owner yes | `RESULT_2026-08-23b_c2b.md` |
| **C1e + C8** | **The last inert data family is wired, and the per-layer grain ladder is measured on three more stocks.** C1e: the 1.32×-high suspicion the item rested on is **disproved by the sheet's own tick comb** (15 ticks within 1 %); adopted 5219 5.92/6.60/17.84, 5207 rms_b 8.92, 5203 rms_b 4.71 by RATIO with green frozen; 5219's blue grain rises ×1.52 on screen. C8: `exposureTimeS` in both languages, inert at 0, applied inside stage 8 on the log exposure; **15 measured reciprocity tables read from the sheets (6 → 21)** because seven stocks held a printed correction and rendered nothing. ⚠ Two errors of mine caught and recorded: the C8 queue entry's intensity-dependent model (impossible — all data is time-only) and the CC-filter arithmetic (added the filter density to the worst record, giving 1 stop where 5205's sheet says 2/3). ⚠ Two defects recorded but NOT patched, because the documents are not in this corpus copy: two different conventions behind the Agfa/Konica exponents (24 % light), and Kentmere's tables not reproducing their own stated formula (~40 % light) | `RESULT_2026-08-23_c1e_c8.md` |
| **C13** | `KODAK_VISION_200T_5274`'s MTF panel traced and **adopted**: f50 **35.4 / 68.8 / 74.0** c/mm against the estimate 56/64/72, adjacency 0.09 -> 0.162, q = 2.94. Green and blue confirmed the estimate to 7 %, **red was 1.58x too sharp**. ⚠ Render impact is **scale-dependent**: 3.9/255 at 48 px/mm, 7.1 at 96, 11.1 at 193 -- f50 lives above the frequencies a 2K render reaches, so the correction pays at scan resolution. ⚠ My first impact measurement said 0.1/255 and was wrong: the target was 3.9 px/mm, which cannot resolve the effect. Raised **C24** (92 estimated triples) | `RESULT_2026-08-20c_c13_mtf.md` |
| **DIR parity** | The two DIR-coupler stages -- the largest COLOUR effect in the chain -- now have a cross-language audit, and it found **two real defects on its first run**. (1) FIXED: the density floor was inside the C++ stage and outside the Python one, so the PIPELINES agreed while the FUNCTIONS differed by **0.26 D** on a reversal stock. (2) OPEN: the adjacency blur is **implementation-dependent below ~1 px sigma** (1.5e-1 apart at 0.4 px, 6e-5 above 1.2 px) and the stored 9-13 um edge scale sits there at ordinary render sizes. Also: the missing `CHANGES_2026-08-03_v5_interimage.md` written, the interimage tier contradiction fixed, the stale guard converted. **No profile data changed** | `RESULT_2026-08-20b_dir_parity.md` |
| **KODAK review** | **2 stocks added** — `KODAK_VISION2_50D_5201` (curves, σ(D), grain level and per-record MTF **all traced from one sheet**; f50 32.1/49.7/55.5 is the first per-record MTF in the file) and `FUJI_SUPER_F125_8532` (C6). **C5** closed. ⚠ **Three of the five PDFs the owner supplied were documents already mined** — one byte-identical. ⚠ **Method rule 22**: the panel titled "Sensitometric" draws six-vertex polylines and fitted six parameters to six points at rms 0.0004 D; the shape came from the *granularity* panel instead. ⚠ **"The vector σ(D) corpus is exhausted" was wrong** — a ninth sheet was in the same folder, refused by four guards that never said so | `RESULT_2026-08-20_kodak_5201_c5_c6.md` |
| **C1e / C2b** | Both moved on evidence the new code produced for free. **C1e:** 5219's brochure net-1.0 green is 6.37 vs a stored 6.6, so the raster extractor does read ~1.3× high **and** the stored VISION3 green levels are right — deferral vindicated; its blue is 1.8× low and that is now the open half. **C2b:** 1 traced MTF curve → **7**; the power law wins on all seven and **q clusters by layer depth, not by film** (both reds 1.84–1.89, both blues 3.38–3.42) | same |
| **C2** | **MTF is a curve now.** Carrier chosen by measurement and the schema's reserved form LOST: `1/(1+(f/f50)^q)` at rms 0.0375 with one parameter beats `mtf_tail_a/_f_exp` at 0.0583 and the Gaussian at 0.0878. Wired both sides, parity extended to 8478 probes, **both laws exactly 0.5 at f50** so no level decision rides along. Schema v10 | `RESULT_2026-08-19b_C2_mtf_curve.md` |
| **G3** | 682's remaining figures: **f50 read from Fig. 11** (r 29 / g 44, replacing estimates 46/54; blue bounded >50 only), layer stack from Fig. 6 with the six-emulsion double-layer construction, Fig. 7 used as a check on the stored spectral set, Fig. 8 peaks measured but the **dye set left EMPTY rather than interpolated** | `RESULT_2026-08-19_gevaert.md` |
| **GEVAERT harvest** | **2 stocks added** — `GEVACHROME_600` / `_605`, the 1968 Agfa-Gevaert reversal television pair (printed DIN/ASA for two illuminants, per-layer gammas 1.25/1.25/1.45 and 1.35, nine-layer stack, documented 320 ASA push). **`GEVACOLOR_NEG_682`'s three characteristic curves re-traced** at one sample per pixel column (589/513/437 samples, fit rms 0.004–0.006 D) and validated against the γ 0.57 the figure prints (traced 0.5677). ⚠ Two self-referential loops found and closed — a fit seeded with its own previous output, and an inherited origin read live from the profile it feeds | `RESULT_2026-08-19_gevaert.md` |
| **C1b + C1d** | **the grain LEVEL is now defined by the documents.** `rms_granularity` means σ at **net** density 1.0 (Kodak's own printed footnote), the sampler normalises there in Python and C++, schema v8→v9. ⚠ **A finding I reported earlier was wrong and is corrected here:** the "per-channel 2.8× imbalance on 43 stocks" was an artefact of assuming *absolute* 1.0 — at net 1.0 the legacy law is `sqrt(1+fog)` and dmin cancels, so there was no imbalance. Real cost: a uniform 4–8 % drop. Six negatives re-levelled from their curves (0.91–1.28×, **not** the 1.3–1.6× I first claimed), with measured per-layer triples showing **blue 1.9–2.8× green** vs the old 1.3× estimate. Four Svema stocks preserved exactly. VISION3's four deferred on evidence → C1e | `RESULT_2026-08-18g_C1b_C1d_level.md` |
| **C1c** | **the σ(D) harvest is closed.** All 8 vector granularity sheets read; **6 colour negatives adopted** (11 measured stocks now); 5219's shape independently confirmed from a second document by a second extractor, with a 1.3× absolute conflict recorded not averaged; **6 extractor defects** found, all of which had been producing plausible numbers | `RESULT_2026-08-18f_C1c_sigma_harvest.md` |
| **C1** | σ(D) carrier chosen **by measurement** and **wired** into both renderers; schema v7→v8; 5 stocks measured, 150 bit-for-bit unchanged | `RESULT_2026-08-18e_C1_sigma_wiring.md` |
| **E0b** | 3 more dye-density sets (7 → 10 profiles), first measured **reversal** σ(D), PLUS-X f50 measured. Three extractor defects found — the sources were never at fault | `RESULT_2026-08-18d_E0b_vector.md` |
| **E0** | 11 profiles re-verified against sheets wrongly declared absent: 5 values changed, 3 digit-for-digit agreements pinned, 2 plausible "corrections" rejected, Sehlin/Kennel settled as **July 1985** | `RESULT_2026-08-18c_E0_reverify.md` |
| **B2** | `AGFA_VISTA_200` spectral sensitivity adopted; the "legend" was a **dash pattern**, not colour. Latitude corrected (a value read one column over) | `agfa_vista.py`, queue A11 |
| **C3** | `SVEMA_FOTO_32` / `_130` tint + silver_tone withdrawn — transfers from a parent measurement that was itself void | queue A10 |
| **2 new FUJI docs** | F-125 8530's `f50` corrected **78 → 42 c/mm** (Honjo 1989, the only document naming that type); the complete 8532 successor sheet read and indexed | `Index.md`, queue A9 |
| **Provenance** | **Zero** profiles now claim tier ≤ 2 with no source (was 8). 7 citations registered; guard 3 fired on its first real test | `NotFound.md` §0.2 |
| **C++ split** | 676 KB single function → 16 slots + `LoadFilmDataBase()`; compiles in VS2015 SP3 | `CHANGES_2026-08-18_cpp_split.md` |

---

## Open work, ranked — no decisions needed, just effort

| # | Item | Effort | Value |
|---|---|---|---|
| **C2b** | ⚠ **The weakest part of today's work.** The MTF rolloff carrier was chosen against **one** traced curve. 199 vector MTF pages are inventoried; trace 5–10 across makers and eras, then re-score with the 12-sample array back in the running (it scored 30× better and was rejected only as over-parameterised against a single film) | medium | **high — it either confirms the choice or replaces it, and nothing else can tell** |
| **C1e** | Re-derive `vision3_granularity.py`'s σ axis. Two independent signals say it reads ~1.3× high: the 5219 brochure reads 1.32× below the 5219 raster trace, and the raster family's implied rms correction is ~1.5× its vector-traced siblings'. Until settled, the four VISION3 rms values are deliberately NOT re-levelled | small–medium | **high — gates 4 stocks and casts doubt on 4 adopted σ(D) shapes** |
| **G7** | Finish the Gevacolor 682 dye-density trace (Fig. 8). Peaks measured (Y 448 / M 525 / C 687 nm) but the three curves are not separated: at 340 ppi the cyan curve's dotting merges into components the style split classes as solid. `dye_density` was left EMPTY rather than interpolated | medium | medium — 682 is the only masked negative with no dye set |
| **C2c** | `adjacency_um` disagrees with the measured overshoot frequency on both stocks checked (PLUS-X 4.7 c/mm vs stored 16.0 µm; F-125 ~9 c/mm vs 13.0 µm). Split out of C2 rather than left buried in it | small | medium — controls visible edge crispness |
| **G2** | The 1968 Gevachrome MTF (Bild 1a–c) and spectral sensitivity (Bild 2a/2b) curves — **blocked on scan quality, see G5** | medium | medium |
| **C8** | Same treatment for `ReciprocitySpec` — now the **last** carrier family read by nothing, C1 and C2 having wired the other two | small, the pattern is established twice | medium |
| **B1** | 2 dye sheets left (5246, 5248) + `5247` p4 visual pass. Near-misses are measured and recorded, not shrugged at | medium | high |
| **E0c** | `5285` p3's **vector characteristic curves** — its stored gamma of 11.6–15.4 is a softplus artefact by its own comment | medium | medium |
| **E1** | Kodak 1952 Data Book + Agfa 2003 → 7 stocks `[T2]`→`[T1]` | medium | medium–high |
| **E2–E5** | Polaroid spectral; Gevacolor 682 rasters; Eastman 1942; Sehlin/Kennel Figs 7–9 | medium each | medium |
| **D1–D3** | Owner measurements: one `--empty-gate` frame (free), one step-wedge scan (~$30–50), `max abs(R−G)` over the Tasma batch | your side | **high** — D1/D2 make density absolute and separate emulsion σ from scanner σ |
| **G5** | ⚠ **Owner acquisition ask:** a 300+ ppi grayscale re-scan of Kino-Technik 1968 Nr. 10, printed pp. 260 / 262 / 264. The 150 ppi scan on file cannot separate the three Gevachrome layer curves (1–2 px apart) and blocks G2 | your side | **high — unblocks G2 and upgrades two profiles from [T3] estimates** |
| **G6** | ⚠ **One document settles a factor of 2.** Gevacolor 682's MTF abscissa says "lines/mm"; the database stores cycles/mm. Any Agfa-Gevaert MTF or resolving-power sheet decides it for the whole Gevaert family | a document not held | medium–high |
| **F1–F3** | Blocked on material not held | — | medium |

---

## Standing hazards — read before trusting a search

* **A sheet's own boilerplate can contradict its own plot.** H-1-5285 says its dye curves are
  "peak-normalized"; the plotted maxima are 0.921 / 0.895 / 0.907. The same sentence *is* true
  on the VISION2/VISION3 sheets, so the phrase cannot decide a normalisation — only measured
  maxima can.
* **Kodak reused catalogue numbers, and two sheets can share a product name.**
  `H-1-5294` documents EKTACHROME 100D **5294/7294**; the database holds **5285** only. Both
  files are called "Ektachrome 100D".
* **Searching the corpus for `H-1-5247` hits the 7239 file** — stray template text on its
  page 1. It is not a 5247 source.
* **A filename's year can be the conference, not the publication.** Sehlin/Kennel is
  **July 1985**; the file says 1983.
* **The second exposure index on a Kodak colour sheet is usually filter-derived** (80A, 85,
  85B) — a filter factor, not a film speed, and deliberately not stored.
* **`5213` has no rem-jet; `5219` does.** Printed on both sheets.
* **A LEVEL IS MEANINGLESS UNTIL ITS REFERENCE DENSITY IS NAMED** (method rule 21). Kodak's rms
  figures are read at **net** density 1.0, not absolute. On a masked colour negative absolute 1.0 is
  net 0.42 in green and net 0.16 in blue — a shadow, not a midtone. Measuring a net-referenced law
  against an absolute-referenced expectation manufactures a 2.8× "channel imbalance" that does not
  exist; it was written up as a defect before the footnote was read.
* **A count test is not a coverage test** (method rule 19). Five whole curves plus one curve's
  right-hand third satisfied "six curves" on the 5279 sheet and the extractor reported success.
* **σ(D) is multivalued at the toe.** Below the toe the characteristic curve is flat, so density
  holds at dmin while σ keeps changing. Two traces of the SAME stock can disagree by 1.7× on the
  toe anchor while agreeing to 0.02 on dmax — that is one ill-posed anchor, not two bad sources.
* **A folder name is not an emulsion identity** (method rule 17) — and a carrier is chosen by
  measurement, not elegance (method rule 18).
* **THE BEST PANEL ON A SHEET IS NOT ALWAYS THE ONE WITH THE RIGHT TITLE** (method rule 22, added
  2026-08-20). H-1-5201's panel headed "SENSITOMETRIC CURVES" draws its red and green records as
  **six-vertex polylines** over ±1.8 decades with neither dmin nor dmax on the plot; the
  characteristic curves inside the *granularity* panel on the same page are 100–125 samples over
  4.1 decades and reach both ends. Six free parameters against six points fitted to rms 0.0004 D —
  an interpolation wearing a measurement's error bar.
* **A "CONTRAST TRANSFER FUNCTION" IS NOT AN MTF.** Fuji's Super F-125 sheet measures response
  against a **rectangular** wave chart, which runs up to 4/π above the sine-wave MTF at low
  frequency. Reading f50 off it would overstate sharpness on every Fuji stock in the corpus.
* **Printed tick labels can be typographically jittered.** On H-1-5201 the density label "1.0" sits
  5.0 pt (0.17 D) off its own gridline in two different panels. The axis is fine; the *label* is
  loose. Recovered by pinning the axis to the frame span and checking its slope against the
  (bad) label fit — never by loosening the collinearity tolerance.
* **An alias is normalised to alphanumerics only.** `"vision2 50d"` collapses to `VISION250D`,
  which is already `KODAK_VISION_250D_5246`'s alias `"vision 250d"`. Two different films, one
  lookup key; the index raises, which is how it was caught.
* **READ EVERY TABLE BEFORE TRACING ANY CURVE** (2026-08-24, and it cost a full digitisation).
  T-101 Fig. 18 was cracked properly — bow tracking, a dash-period classifier, an arc-length
  walker — and validates three independent ways. Then **Table 2 on p28 turned out to print the
  measured equivalent grain diameter of all six emulsions outright**, which is the quantity the
  trace was for. That table had already been cited on four profiles since 2026-08-23 for its
  *other* columns. The trace was not wasted (it is what validates the printed ladder and what
  proves `clump_gain` = 0 on all six), but it is not what is stored.
* **A PAGE CAN BE BOWED RATHER THAN SKEWED, and the two need opposite fixes.** On T-101 p30 one
  gridline sits at y 325 near the left frame and 310 near the right, and the W = 0 line moves
  **97 px** across the plate. No rotation corrects that, and a whole-row fill test cannot find the
  grid at all. Track every line independently; the residual scale error otherwise reaches 6 % of
  the smallest curve's own W(0).
* **A GRIDLINE TRACKER CAN FOLLOW A CURVE AND THEN DELETE IT.** T-101's W = 0.075 line runs almost
  along the 8374 spectrum; the tracker locked onto the curve, wandered 83 px, and its removal band
  erased the emulsion while leaving the real gridline behind. The emulsion looked *absent from the
  plate*. A trimmed fit across the whole ladder index is what caught and repaired it.
* **A TANGENT WALKER WILL DIVE DOWN A GRIDLINE STUB.** Monotonicity is not enough of a constraint:
  Pan F and 5302 both fell to W = 0 down residual vertical stubs while satisfying "y never
  decreases". The constraint that works is a **turn-rate limit** — the real curves never turn
  faster than ~0.6 °/px, a stub turns instantly.
* **AN UNDER-DETERMINED FIT LOOKS EXACTLY LIKE A GOOD ONE.** Pan F's spectrum falls only 18 %
  across its resolvable range, so it fits `clump_um` 0.613 with a Gaussian carrier and 1.168 with
  a free exponent **at equal residual** — 1.9× apart. The residual was small in both cases. What
  exposes it is varying a parameter you were not fitting, not looking at rms.
* **A PRINTED NUMBER CAN CARRY ITS OWN ERROR DIRECTION, and that is worth more than a tighter
  number without one.** T-101 p38 states its equivalent grain diameters are "expected to be
  greater than the true values". So every `clump_um` adopted from them is an **upper bound** — and
  the traced fits, sitting 4–15 % higher, are looser bounds in the same direction. Knowing the sign
  of an error beats not knowing it.
* **THE SAME TRADE NAME IS NOT THE SAME PRODUCT** (method rule 18 again, 2026-08-24). T-101
  measured "Tri-X Type 5223", the 35 mm cine negative at 250/320 A.S.A. `KODAK_TRI_X_400TX` is the
  ASA 400 still film. The measurement got its **own** profile rather than being pushed onto the
  neighbour, and `verify.py` now pins the non-move so a later pass cannot "finish the job".
* **A FIGURE CAN BE DRAWN SO THAT IT CANNOT BE READ, ON PURPOSE** (2026-08-24). Fuji's own
  characteristic-curve plot for F-125 superimposes it on the previous emulsion *at matched speed*
  — that is the POINT of the figure, it shows the shapes differ while the speeds do not — so each
  visible track carries two films and neither can be traced. The plate is not defective; it is
  answering a different question. Check what a figure was drawn to show before trying to digitise it.
* **AN AXIS CAN CARRY A SCALE WITHOUT CARRYING AN ORIGIN.** The same plots label the abscissa only
  with a 0.5-decade span bar. Gamma (a slope) is therefore measurable in principle and the speed
  point is not, ever — no amount of care recovers an origin that was never printed.
* **THE DECIDING EVIDENCE FOR A DATABASE-STRUCTURE QUESTION WAS A SENTENCE, NOT A MEASUREMENT.**
  Whether 8530/8630 are one film or two was settled by Fuji's printed code rule — second digit is
  the gauge — after two days of treating them as separate profiles. Read the prose around the
  tables, not only the tables.
* **CHECK THE UNITS REGIME, NOT JUST THE UNITS** (2026-08-25, and it cost a retraction). σ_t and σ_D
  differ by a linearisation that is only valid for small fluctuations. T-101 measures grain at a
  scale where the fractional fluctuation reaches **164 %**, so the conversion this project reaches
  for by habit is simply wrong there. Two granularity measurements can use the same words, the same
  symbols and the same film and still not be the same quantity.
* **A "CONFLICT" BETWEEN TWO SOURCES IS SOMETIMES A CONFLICT BETWEEN TWO CONVENTIONS.** Before
  recording Mees against T-101 under method rule 4, the question "are these even commensurable?"
  dissolved it entirely. Rule 4 is for real disagreements; reaching for it too early would have
  parked a permanent false conflict in the file.
* **A NUMBER IS ONLY VALID AT THE CONDITION IT WAS MEASURED AT.** `clump_um` was adopted from a
  table whose samples were developed to the BBC's gamma, not to the gamma each profile stores. Five
  of six matched by luck. The sixth was 27 % out.
* **PAIRING TWO PLOTS IS ONLY AS GOOD AS THE SLOPE BETWEEN THEM** (2026-08-25b). Reading σ(D) from a
  granularity panel plus a characteristic curve means inverting D(log E). Wherever that curve is
  flat the inverse does not exist, and the raw pairing there produces smooth, plausible, entirely
  fictional structure — the discarded 2.93× "interior peak" was exactly that. **Condition the
  pairing on |dD/dlogE| before fitting, not after.** 22 of 52 points had to go.
* **AN ESTIMATE CAN BE WRONG IN DIRECTION, NOT JUST IN MAGNITUDE** (2026-08-25b). 36 reversal stocks
  carried granularity *falling* toward dmax. The one stock that got measured shows it *rising* 2.8×,
  because on reversal film dmax is the unexposed fully-developed silver. A plausible number,
  extrapolated across a whole class from the wrong film TYPE, survived months of checks — none of
  which could catch it, because nothing in the file said which direction to expect.
* **CHECK THE SCHEMA CAN HOLD THE ANSWER BEFORE ASKING WHICH ANSWER TO GIVE** (2026-08-25g). Three
  questions were put to the owner about how to store a film's aging data, and answered. Only
  afterwards did reading `StockKind` and `PrintStock` show that the schema has no intermediate
  category and no aging field outside `FilmProfile` -- so two of the three answers could not be
  acted on. **A decision request is only as good as the feasibility check behind it**, and asking
  costs someone else's time, which makes an unchecked question worse than an unasked one.
* **STRIKE A CLOSED ITEM THE DAY IT CLOSES** (2026-08-25g). E0b-orig described three plot sets as
  to-do; all three had been adopted a week earlier. It stayed on the ready-now list the whole time.
  An item that reads open when it is closed wastes exactly as much attention as one that is wrong.
* **WHEN YOU FIND ONE INSTANCE, MEASURE THE WHOLE SURFACE** (2026-08-25f). C30 looked like one
  bypassed law. Enumerating the surface took ten minutes and showed the rate was **2 of 2** — every
  shared law between the two implementations was unreachable from the code that renders. A single
  instance is a bug; a rate is a design fault, and you cannot tell which you have without counting.
* **FIXING ONE SIDE OF A MIRRORED PAIR CREATES A DIVERGENCE** (2026-08-25f, and I did it). Correcting
  grain on the scalar path alone put the two C++ implementations 1.039–1.183× apart within the hour.
  **Either fix both sides or fix neither** — a half-applied law is worse than the original error,
  because now two things are true at once and no test says which is running.
* **A NEW GATE'S FIRST PASS DESERVES MORE SUSPICION THAN ITS FIRST FAILURE** (2026-08-25f). The
  law-reachability check passed immediately — on a comment describing the very failure it was built
  to detect. **Fault-inject a new gate before trusting a green result from it.**
* **A PARITY CHECK MUST EXERCISE THE CODE THAT RUNS, NOT A FUNCTION BESIDE IT** (2026-08-25e, and
  it cost weeks of shipped error). The generated header held a correct grain law; the harness
  evaluated that law; the renderer called neither. Every number in the report was true and the
  rendered image was still 4–18 % wrong. **Ask of any parity harness: does this call the same entry
  point the product calls?** If it re-derives, re-implements or reaches past the stage, it is
  measuring agreement between two things that both sit outside the pipeline.
* **A SPOT CHECK AT THE ANCHOR PROVES THE LEAST** (2026-08-25e). The grain error crossed 1.0 near
  net density 1.0 on every affected stock — the one density anybody would sample by hand, and the
  one the stored figure is quoted at. Errors that are normalised at a reference point are invisible
  exactly where you are most likely to look.
* **BEFORE BELIEVING A NUMERICAL TEST, CHECK ITS SAMPLING** (2026-08-25e). The new f50 check
  reported two stocks at 0.559 and 0.590 against an expected 0.500. Both were real code and false
  failures: f50 did not land on an FFT bin, so the sine leaked and peak-to-peak stopped measuring
  modulation. **A new test's first failures are more likely to be the test's than the code's.**
* **A CROSS-CHECK AGAINST ONE SAMPLE IS NOT A CROSS-CHECK AGAINST A FAMILY** (2026-08-25d, and it
  caught an error made the same day). "5201's blue peaks at 470 where its siblings peak at 420" came
  from comparing one sibling. Sweeping all ten showed 470 is the majority, 6 to 4. **When writing
  "unlike its siblings", enumerate them.**
* **AN EMPTY HAND-MAINTAINED SET IS NOT A STATEMENT ABOUT THE WORLD** (2026-08-25d).
  `QUEUED_PLOT_ON_FILE` had been empty since 2026-08-02, and the generator read that as "no stock has
  an un-digitised plot on file" — printing the stronger claim on its behalf for five stocks whose
  plots are catalogued elsewhere in the same project. Empty means nobody refilled it.
* **A HARDCODED CENSUS GOES STALE SILENTLY; A DERIVED ONE CANNOT** (2026-08-25d). Four ISO-standard
  counts in the report generator were hand-typed and all four were wrong — by up to 2.3×. Nothing
  could have caught them, because nothing was comparing them to anything. Every count in a generated
  document should be computed from the database in the same expression that fills the cells beside it.
* **A GUARD THAT CANNOT FAIL IS WORSE THAN NO GUARD** (2026-08-25d, queue C20). "Interimage leaves a
  neutral untouched" tested the anchor — the single point where the effect is zero by construction —
  so it passed for any possible interimage matrix while reading like a strong invariant. **Ask of
  every new guard: what edit would make this fail?** If nothing plausible would, it is documentation
  wearing a test's clothes.
* **A DIAGNOSIS RECORDED IN THE QUEUE IS NOT EVIDENCE** (2026-08-25c). C9 carried a specific,
  plausible, *wrong* explanation of why a sheet would not extract, and it survived a fortnight
  because nobody re-derived it — the real cause was a segment-count filter three lines away. When
  picking up a deferred item, **reproduce the failure before believing the note about it.**
* **IDENTIFY A TRACE BY WHAT THE PUBLISHER USED TO DRAW IT** (2026-08-25c). Kodak's brochures draw
  every curve in the colour of the light it concerns, which makes ink a *physical* label, not a
  cosmetic one — and it survives the case that defeats geometry: a curve drawn as two overprinted
  paths of 7 segments each. Segment counts, stroke widths and dash patterns are proxies for
  identity; the ink, where a publisher is consistent, nearly is identity.
* **WHEN A NORMALISATION BREAKS THE OLD IDENTITY, LOOK FOR THE GENERALISED ONE** (2026-08-25c).
  `neutral = C+M+Y` fails the moment the dyes are peak-normalised. `Neutral − Dmin = k·(C+M+Y)` with
  the three k EQUAL holds instead, and it is a *stronger* test: the equality was not imposed, it
  fell out of a free three-parameter fit to 5.4 %.
* **CHECK WHETHER THE STRING YOU ARE COPYING IS PRINTED ANYWHERE** (2026-08-25c). Three profiles
  carried a spectral criterion naming a density ("D 0.2 above dmin") that appears on none of their
  three source sheets. It was almost certainly copied from a fourth document and then propagated by
  each new adoption matching its siblings. A convention worth storing is worth quoting.
* **SHAPE AND LEVEL ARE SEPARATE ADOPTIONS** (2026-08-25b). The same panel that grounded the shape
  implies a level 2.2× the stored one, and its caption disqualifies the level ("modified measuring
  techniques") without disqualifying the shape. Taking both because they came off one plot would
  have imported an undefined convention; taking neither would have thrown away a real measurement.

---

<!-- build-facts: schema=v24 stocks=175 names_md5=faf861bcb7523155324bc875ed67c1c8 -->
<!-- build.py's docs stage parses the line above and FAILS if it disagrees with the live
     database. Update it when the facts move; do not delete it. -->
