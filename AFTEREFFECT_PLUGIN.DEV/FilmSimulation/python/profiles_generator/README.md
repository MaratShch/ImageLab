# Photochemical film simulation

A rewrite of the original grain-overlay script as an actual photochemical model.
Python 3.12, 64-bit, Windows and Linux/WSL2. Dependencies: **numpy and Pillow only** —
no OpenCV, no SciPy. 16-bit PNG writing uses stdlib `zlib`.

⚠ **That rule covers the RENDERER and the GENERATOR, which is where it matters. The audit
scripts additionally need PyMuPDF**, because they re-derive adopted numbers from the source
PDFs — an audit that cannot open the document cannot check anything. `agfa_2004_curves.py`
will also use **SciPy** for one six-parameter curve re-fit if it is installed, and skips that
one check with a printed message if it is not; every other check in it runs either way. No
audit dependency reaches the render path.

> ## → For current status, read **`PROGRESS.md`** first
>
> One screen: build state, what is done, what is open and ranked, and the short list of
> things waiting on an owner decision. It is updated at the end of every task. The dated
> status entries below are the **history** — they are kept verbatim, including where a later
> entry supersedes an earlier one, because the reasoning is the audit trail.

> ⚠ **HOW TO READ THIS SECTION.** Every `Status <date>` entry below is a **snapshot of that
> day** and is left as written, including its counts and its wrong turns -- the reasoning is the
> audit trail. **Nothing in a dated entry should be read as current state.** For current state use,
> in order: `PROGRESS.md` (one screen), `FilmActiveProfiles.md` (per-stock coverage, regenerated
> from the database on every build), `NotFound.md` (what is still missing), and the **Files** and
> **Known limits** sections at the bottom of this README, which ARE maintained as current.
>
> **Status 2026-08-25g (task 5 started, queue E0b-orig / C15):** `verify.py` **351 PASS / 1 FAIL**,
> 14 audits green, schema **v11 unchanged**, **160 stocks unchanged** -- the new profile C15 would
> have added is NOT there, deliberately; see below.
>
> **THE FIRST MEASURED MTF FOR A COLOUR REVERSAL STOCK, AND THE LARGEST MTF CORRECTION THIS PROJECT
> HAS MADE.** `KODAK_EKTACHROME_100D_5285`, traced from H-1-5285 p3 by `mtf_vector.py`. Stored
> f50_g was the estimate **82.0**; the sheet measures **42.1** -- the estimate was **1.95x too
> sharp**. Red and blue had no measurement of their own either: 74.0 / 90.0 were the estimating
> rule's fixed ratios about that wrong centre. Measured **R 27.2 / G 42.1 / B 60.9**, layer order
> R < G < B exactly as `MTFSpec`'s docstring predicts (second independent confirmation, after 5201),
> and the power law beats the legacy Gaussian on all three records (q 1.87 / 2.39 / 2.52).
>
> ⚠ **IT ALSO BOUNDS A FAMILY CONSTANT.** Queue C24 found measured red clustered at 36.4 cycles/mm
> across the Kodak cine NEGATIVES, and five profiles were re-anchored to exactly 36.0 on that
> finding. This REVERSAL stock's red is **27.2** -- 25 % below, and it alone would take the cluster's
> spread from 25 % to 41 %. The guard now excludes reversal stocks, which is the same class-estimate
> refusal C24 already made for the Fuji sheets. **Nothing had licensed assuming that anchor reached
> past the negative family, and it does not.** The five re-anchored negatives are untouched.
>
> ⚠ **QUEUE ITEM E0b-orig HAD BEEN FINISHED FOR A WEEK AND NEVER STRUCK.** Its three plot sets --
> 7239 dye density, 5231 MTF, 5285 sigma(D) -- were all adopted on or before 2026-08-18 while the
> entry still described them as to-do, so it sat on the ready-now list the entire time. One real
> leftover remains: **7239's spectral sensitivity**, blocked because its panel carries no rotated
> caption for the panel finder to anchor on (the known limit recorded in `spectral_vector.py`: 5
> pages reachable against 24 for dye panels).
>
> ⚠ **AND C15 IS STOPPED, ON SOMETHING I SHOULD HAVE CHECKED BEFORE PROPOSING IT.** Three questions
> were put to the owner and answered -- do not transfer the fade rates, treat the DI film as a
> separate film, store `AgingSpec` inert -- all on the strength of my recommendation to give it its
> own profile. Reading the schema afterwards: **`StockKind` has only NEGATIVE and REVERSAL**, there
> is no intermediate category, and the DI/dupe concept already lives in `PRINT_STOCKS` (`SCAN_DI`,
> `DUPE_FINE_GRAIN`) -- but **`PrintStock` carries no `aging` field**, because `AgingSpec` exists
> only on `FilmProfile`. So attaching the measured Arrhenius table today would mean declaring a
> digital-intermediate recording film a camera negative, putting it in the camera ListBox and
> shifting 160 indices: a category error committed to obtain a field. **Stopped rather than forced.**
> The data is confirmed from the full sheet (H-1-2254, March 2026): colour separations yellow **86
> years**, D-min blue **77 years**, at 21 C; everything else >100, and everything >100 at 7 C.
>
> **Status 2026-08-25f (four gates, queue C32-C35):** `verify.py` **351 PASS / 1 FAIL**, 14 audits
> green (two new), schema **v11 unchanged**, **160 stocks unchanged**. No new film data -- this pass
> adds checks, not values.
>
> **THE BYPASS RATE ON THE SHARED-LAW SURFACE WAS 2 OF 2.** After C30 found one law with no callers,
> the surface was enumerated rather than sampled: `film_sim.py` calls exactly `fp.grain_sigma` and
> `fp.mtf_response` from the database module and nothing else, and the generator emits exactly those
> two into the header. **Both were unreachable from the stages.** `check_law_reachability()` now
> fails the build on any published law with no caller, and equally on a recorded bypass that quietly
> acquires one.
>
> ⚠ **THE NEW GATE'S FIRST RUN PASSED -- ON A COMMENT.** `FilmGrainSigma` reported "reached from 1
> stage source", the source being the comment in `Algo_11_Sim.cpp` explaining that it is NOT called.
> C++ comments are stripped before the search now. A gate that passes on prose about its own failure
> is the defect it exists to catch, and a new gate's first PASS deserves more suspicion than its
> first failure.
>
> ⚠ **AND THE TWO C++ PATHS DIVERGED WITHIN HOURS, BECAUSE OF YESTERDAY'S FIX.** Applying the
> net-1.0 grain normalisation to the scalar stage alone left scalar and AVX2 **1.039x to 1.183x
> apart on grain amplitude** -- a difference in the MODEL, not the vectorisation, which this
> project's own AVX2 rules forbid. Mirrored into `AVX2/Algo_11_Sim.cpp` at **zero inner-loop cost**
> (it folds into `gain` before the broadcast, since `gain * (amp * scale) == (gain * scale) * amp`),
> and verified by compiling the twin under `AlgoType = float` with `-mavx2 -mfma`. **The header was
> flipped for that test and restored to `double` immediately -- the shipped alias is untouched.**
> `check_twin_consistency()` now asserts that law-bearing tokens appear on both sides.
>
> ⚠ **`FilmMtfResponse` IS STILL BYPASSED, AND IS NOW RECORDED WITH ITS COST.** The measured
> `1/(1+(f/f50)^q)` rolloff is a frequency-domain form and **the C++ side has no FFT at all** --
> `AlgoSeparableBlur.hpp` opens by arguing why not. So the 9 stocks with a measured q render on the
> legacy Gaussian: exact at f50 by construction, **up to 3.8x too much modulation at 2x f50**.
> Closing it needs an architecture decision, not a patch.
>
> **A DOCUMENTATION GATE (C34).** An audit found four hardcoded counts wrong by up to **2.3x**, and
> nothing had compared prose to the database -- `build.py` gated exactly one claim, and that check
> had never fired. `doc_consistency.py` checks a registry of load-bearing sentences against live
> expressions, from the audit stage. ⚠ A pattern that stops matching **fails**: an unmatched pattern
> silently stops checking, which is the state it exists to end.
>
> **THE PROJECT-ROOT `doc/` FOLDER, REVIEWED FOR THE FIRST TIME (C35) -- and it has never been
> delivered.** Seven documents, 2661 lines, outside every archive ever shipped, because every
> archive covered `PYTHON/profile_generator/doc` only. Three of them still described stages **15, 16
> and 09b as stubs** when all three fully render. `STAGE_FUSION_PROPOSAL`'s central 4K memory
> argument is **superseded by its own project**: it quotes the pre-ping/pong 2.90 GB float footprint,
> but M1 shipped and 4K UHD is now ~0.77 GiB, so the case for fusing to reach 2 GB is moot. ⚠ And
> **neither AVX2 document states that the vector build requires `AlgoType = float`** while the
> shipped header sets `double` and 17 AVX2 units static_assert against it. Not corrected: every
> performance figure is a 2026-08-11 measurement never repeated, now with two 2026-08-25 engine
> changes underneath the tables that time them.
>
> **The closed-loop tier widened from a 5-stock sample to the whole database** -- f50 modulation and
> characteristic-curve reproduction across all 160 stocks, 0 outliers.
>
> **Status 2026-08-25e (the grain bypass, queue C30/C31):** `verify.py` **351 PASS / 1 FAIL**,
> 12 audits green, schema **v11 unchanged**, **160 stocks unchanged**, `film_names.txt` unchanged.
> ⚠ **THE C++ RENDER CHANGES ON EVERY STOCK, DELIBERATELY** -- this is a correction, not a tuning.
>
> **THE LARGEST SINGLE ACCURACY DEFECT FOUND IN THIS PROJECT, AND IT WAS IN THE SHIPPED PLUGIN.**
> `film_profiles.hpp` defines `FilmGrainSigma()` as THE ONE DEFINITION -- the legacy law divided by
> `sqrt(1 + fog_grain)`, so the shape is exactly 1.0 at NET density 1.0, plus the measured-anchor
> branch. **It had zero callers.** `AlgoAddGrain` inlined its own square root with no normalisation.
>
> Measured across the whole database: on the **147** legacy-branch stocks the C++/Python ratio was
> **exactly `sqrt(1 + fog_grain)`** -- 1.0392 to 1.1832, mean **1.1027** -- reproduced to
> **3.0e-08**, so one missing divisor was the entire level error. On the **13** stocks carrying a
> measured sigma(D) shape it was worse and not a constant: **0.39x in shadow to 2.2x at depth**, and
> **inverted** on the two reversal stocks. That is grain distributed wrongly across the tone scale,
> in opposite directions for negative and reversal -- and every affected stock crossed 1.0 near net
> density 1.0, which is exactly why a mid-grey spot check looked correct.
>
> ⚠ **IT SURVIVED BECAUSE THE PARITY HARNESS CALLED THE LAW DIRECTLY.** `cpp_parity.py` evaluated
> `FilmGrainSigma()` itself, agreed with Python on every stock, and never touched the function that
> renders. Every number in its report was true and the image was still wrong. Third instance this
> month of a check aimed at the wrong subject, after C20's vacuous interimage guard and a compile
> gate that has never covered the AVX2 path.
>
> **Fixed** by applying the normalisation inside the stage, hoisted out of both loops -- it depends
> only on `fogGrain`, which the stage already receives, **so no shared signature changed and the
> AVX2 twin still compiles untouched**. Verified: the stage returns **exactly 1.0** at net density
> 1.0 on all 160 stocks x 3 channels, and the 147 legacy stocks now agree with the reference to
> **4.3e-09**.
>
> ⚠ **STILL OPEN, SCOPED, AND PINNED: the measured sigma(D) SHAPE cannot be reached from the stage**
> (13 stocks). It needs the `GrainSpec` and `dmax`, i.e. a shared signature change with the AVX2
> twin moving in the same commit. Those 13 now get the correct LEVEL and the legacy SHAPE -- worst
> error falls from 0.39x-2.2x to a pinned **1.73** at net 2.5, with net 1.0 exact. Strictly better,
> still not right.
>
> **C31, two validation tiers, both of which would have caught this on their first run.** A
> **stage-level parity family** that compiles and calls `AlgoAddGrain` itself -- 2400 probes, the
> amplitude recovered exactly rather than fitted (`amp = out - D` at unit field and unit gain) --
> asserting the net-1.0 identity and judging the legacy and measured-shape populations separately,
> so a scoped gap can neither grow nor silently close. And a **closed-loop tier** in `verify.py`:
> render, measure back through the manufacturer's own convention, compare to the published number.
> Two new checks -- a sinusoid at f50 returning exactly 50 % modulation under both transfer laws,
> and the rendered characteristic curve reproducing the stored curve to 0.002 D.
>
> ⚠ The f50 check needed its sampling corrected before it could be trusted: at an arbitrary rate the
> sine leaks across FFT bins and peak-to-peak stops measuring modulation, so two stocks read 0.559
> and 0.590 and looked like real failures. px/mm now places f50 on an exact bin.
>
> **Status 2026-08-25d (validation pass + renderer parity + documentation audit, queue
> C17/C20/B1-part):** `verify.py` **349 PASS / 1 FAIL**, 12 audits green, schema **v11 unchanged**,
> **160 stocks unchanged**, `film_names.txt` unchanged -- no ListBox shift. One renderer behaviour
> changed; no stored value moved.
>
> **THE VALIDATION PASS CORRECTED WORK DONE EARLIER THE SAME DAY, which is what it was for.**
> `KODAK_VISION2_200T_5217` already carried a spectral set from the 2026-08-02 RASTER batch -- a
> different image, a different method, a different author -- and it is the only stock in the corpus
> with both an adopted set and a vector panel the new reader can reach. Re-deriving it agrees to rms
> **0.109 / 0.091 / 0.049** decades (r/g/b) with peaks within one 10 nm grid step. Inside the reading
> error of a printed plot, so **neither side was corrected**: a wash is not a reason to churn adopted
> data. Both methods are now credible on evidence rather than on assertion.
>
> ⚠ **AND IT IMMEDIATELY FALSIFIED A CLAIM FROM THE 2026-08-25c ENTRY BELOW.** That entry says
> 5201's blue layer "peaks at 470 nm where its siblings peak at 410-420" -- written after comparing
> exactly **one** sibling. Sweeping every 31-sample Kodak cine stock shows the blue peak splits
> **6 / 4**: 470 nm on 5201, **5217**, 5205, 5203, 5274 and 5246; 410-440 nm on 5218 (420), 5279
> (420), 5219 (410) and 5213 (440). **470 is the majority and 5201 matches 5217 exactly.** Corrected
> in the profile comment, the source string and the guard, which now asserts the split itself so
> neither group can be "harmonised" toward the other.
>
> ⚠ **THE SPECTRAL CRITERION IS NOW CONTRADICTED IN VALUE, NOT MERELY UNSOURCED -- AND IT IS THE ONE
> DECISION WAITING.** A sweep of every short Kodak sheet for a printed density criterion found five
> that print one: **`5246.pdf` p5 ("Density: 0.4 above D-min", Status M, .013 sec), `5274.pdf` p4 and
> `V200T.pdf` p4 all say 0.4**; the VISION3 DI sheet and the digital-workflow sheet say 1.0 (a
> different product class). **No sheet in the corpus prints 0.2.** Yet **16 profiles store
> `log_reciprocal_erg_cm2_D0.2_above_dmin`** while **10 store a printed D0.4 variant** -- and the
> split follows the sheets exactly: the profiles whose sheets print a number carry 0.4, the profiles
> whose sheets say only "specified density" carry the 0.2 that appears nowhere. The 0.2 was supplied
> for precisely the cases with no evidence for it, then propagated by each new adoption matching its
> siblings. **Nothing was changed** -- rewriting a provenance claim on 16 profiles is the owner's
> call -- and the counts are pinned so the inconsistency cannot be absorbed.
>
> **QUEUE C17 CLOSED: a gate that existed on one side only.** `AlgoDirCoupler.hpp` has always
> disabled both coupler components below `ALGO_COUPLER_MIN_SIGMA_PX` = 0.25 px; the Python reference
> had **no gate at all**, so below that scale the two renderers were not approximating each other --
> one ran the stage and the other did not. `apply_dir_couplers` now carries the same gate at the same
> threshold. **The threshold was ADOPTED, not chosen:** taking the shipped and reviewed C++ constant
> is what keeps a fidelity judgement out of a parity fix. The crossovers are not exotic scales -- the
> long term switches off below **3.1 px/mm** (`EASTMAN_5247_1974`, radius 80 um) and the edge term
> below **27.8 px/mm** (`KODACHROME_64`, edge 9 um), a 35 mm frame about 670 px wide.
> `interimage_parity.py` unchanged at worst **5.335e-05**.
>
> ⚠ **C16 IS NARROWED, NOT CLOSED, AND IT IS NOW A ONE-NUMBER DECISION.** The two blurs are still
> different FORMS -- analytic Gaussian transfer in Python, truncated separable spatial kernel in C++
> -- agreeing to 6e-5 only above ~1.2 px and diverging to **1.5e-1 at 0.4 px**, while stored
> `edge_um` of 9-13 um is 0.36-0.60 px at 40 px/mm: inside that band and above the gate. C17 removed
> the question of WHETHER the stage runs; what remains is HOW, i.e. the shared threshold's value.
> Recommendation on file: **raise it to ~1.0 px**, where the forms converge -- which is also the
> honest model statement, since a 9-13 um feature at 25 um/px is below the sampling limit and
> rendering it anyway aliases a sub-pixel feature. It changes every render, so it is the owner's.
>
> **QUEUE C20 CLOSED: a guard that could not fail.** `verify.py`'s "interimage leaves a neutral
> untouched" rendered **0.18** -- the mid-grey ANCHOR the correction is referenced to, the one point
> where every (D_j - d_ref) is zero. It was true by construction for **any** value of the interimage
> matrix while reading like a strong invariant. Renamed to what it tests, and a second guard now pins
> the off-anchor movement as intended behaviour: on `KODAK_PORTRA_400` with the stage disabled, grey
> 0.45 moves **15.9/255** and grey 0.06 moves **6.5/255**. That is the mechanism, not a leak --
> white-light gamma below separation gamma is the patent's own metric. `InterimageSpec`'s docstring is
> qualified to match: the correction vanishes **at the anchor**, not on neutrals in general.
>
> **B1: 5248 WAS NEVER A FAILED EXTRACTION.** Its recorded symptom -- "only 2 curves survive inside
> the frame, so the other traces are being lost" -- assumed traces that do not exist. The sheet
> prints "Typical densities for a midscale neutral subject and D-min." and draws exactly those two.
> There are no separate dye curves on it, so `SpectralDyeDensity.validate()` (cyan AND magenta AND
> yellow) can never be satisfied from it. That is the **same schema-shape mismatch already recorded
> for `FUJI_SUPER_F125_8532`**, and 5248 is its second instance -- which is the evidence the pending
> schema decision needs, not an extractor bug. **5246 stays open with a sharper reason:** 7 traces
> for 5 labels; nearest-label assignment resolves Yellow (peak 1.008 at 446 nm) and Magenta (1.006 at
> 542) cleanly, but the trace nearest "Cyan" peaks at **0.943** against the sheet's own
> "peak-normalized" claim, and two unlabelled traces are unaccounted for.
>
> **A DOCUMENTATION AUDIT, because two of today's errors were in prose rather than in data.** Four
> hardcoded counts in `gen_active_profiles.py` were checked against the live database and all four
> were wrong: **ISO 6 27 -> 51, ISO 5800 34 -> 58, ISO 2240 13 -> 17, manufacturer EI 15 -> 34.** All
> four are now derived from the database. Also corrected there: "7 curves on 3 sheets traced" -> 26 on
> 12; a claim that `ReciprocitySpec` "is still read by no renderer (queue C8)" when C8 closed
> 2026-08-23; "39 raster granularity pages are on disk and unread"; and "all 395 documents in
> `PDF/PROFILES`" against a measured 448 PDFs / 559 files. `NotFound.md` lost nine stale claims,
> including three rows still reading "no profile" for stocks added on 2026-08-24 and a row listing
> four already-closed dye sheets as failures. ⚠ `gen_film_curves_md.py`'s `QUEUED_PLOT_ON_FILE` set
> had been **empty since 2026-08-02**, which made the report print "no plot in archive (text/table
> data)" for five stocks whose plots are in the archive *with page numbers listed in `NotFound.md`
> section 4.1* -- an empty hand-maintained set means nobody refilled it, not that the world is empty.
>
> **Status 2026-08-25c (H-1-5201's last two panels + the tier bug, queue C9/C10/C12):**
> `verify.py` **347 PASS / 1 FAIL**, 12 audits green (one new), schema **v11 unchanged**,
> **160 stocks unchanged**, `film_names.txt` unchanged -- no ListBox shift. One stock gained two
> measured data sets and six changed the tier they report.
>
> **TWO THINGS THIS PROJECT HAD WRITTEN DOWN TURNED OUT TO BE WRONG, and both were wrong in the
> same way: a plausible note went unchecked.**
>
> ⚠ **QUEUE ITEM C9 CARRIED THE WRONG DIAGNOSIS FOR A FORTNIGHT.** It recorded that H-1-5201's dye
> panel could not be read because `dye_density.py`'s family classifier "handles 3 dyes, or 3 dyes +
> neutral, not 3 + neutral + dmin". It never was that -- family B takes any three of however many
> curves it is offered, so two extra traces cost it nothing. **The cyan trace never reached the
> classifier.** Kodak draws it as a yellow-under-magenta overprint, two bit-identical paths of
> **7 segments each**, and `extract`'s `n < 8` segment filter dropped both, leaving nothing in the
> 615-700 nm band for any triple to pass the band test on. The recorded reason was a true statement
> about a curve list that was missing the curve.
>
> **THE FIX IS TO IDENTIFY A TRACE BY THE INK IT IS DRAWN IN**, and Kodak's rule is physical rather
> than decorative: each trace is drawn in the colour of light it concerns. The yellow dye, which
> absorbs blue, is drawn in BLUE ink; magenta in GREEN; and cyan, which absorbs red, in RED -- not
> one of the four process inks, so Kodak overprints yellow under magenta. Read off the panel's own
> legend swatches (green sits on "Magenta Dye", amber on "Cyan Dye"). A lower segment threshold was
> rejected: it would admit gridline stubs on every other sheet, whereas the ink already knows what
> it is looking at.
>
> **AND A NEW VALIDATOR, which is why the set is tier 1 rather than three plausible curves.** With
> the dyes peak-normalised and the neutral as-printed, family A's `neutral = C+M+Y` cannot hold. The
> generalisation that must hold is `Neutral - Dmin = k_c*C + k_m*M + k_y*Y` with the three
> coefficients **EQUAL**, because equal contributions are what make the result a *visual* neutral.
> Unconstrained least squares returns **0.628 / 0.604 / 0.595** -- a 5.4 % spread on three numbers
> that were free to be anything -- at rms **0.019 D**. Drop the Dmin term and the fit is 4.5x worse
> (rms 0.085) with the coefficients scattered over 0.86-1.61, which is what identifies which of the
> two dark traces is the neutral and which the minimum density. Adopted peaks **450 / 540 / 680 nm,
> identical to 5217 and 5218** -- a family check the extractor never saw.
>
> **QUEUE ITEM C10: the FIRST VECTOR-TRACED SPECTRAL SENSITIVITY SET in the database**, by a new
> script `spectral_vector.py` now in `build.py`'s audit stage. Every earlier spectral set came from
> the 2026-08-02 raster batch or `agfa_vista.py`'s dash-legend reader. Same ink rule -- and C10's own
> prediction that Kodak draws the red record as yellow under magenta was right, the two paths
> bit-identical. Assignment checked three ways, **none of them the ink**: the legend swatches, the
> absorption bands (peaks 470 / 540 / 650 nm, ascending), and the independently-adopted 5217/5218
> sets, which agree in red and green to rms 0.05-0.14 decades.
>
> ⚠ **5201's BLUE LAYER PEAKS AT 470 nm WHERE ITS SIBLINGS PEAK AT 410-420.** That is the whole of
> the cross-check disagreement (blue rms 0.24-0.42 decades) and it is **printed**: a narrow cusp
> above log S 2.0 at 470, higher than the 445 nm bump, then a cliff to zero by 500 -- confirmed on a
> 26x render before adoption. It is pinned by a guard, because a later "correction" toward the family
> shape would be undoing a measurement.
>
> ⚠ **THE SENSITIVITY CRITERION IS PRINTED ON NO SHEET IN THIS FAMILY.** 5201's footnote says only
> "Sensitivity = reciprocal of exposure (erg/cm2) required to produce **specified density**" -- it
> never names the density. The three sets already stored (5218, 5217, 5219) carry
> `log_reciprocal_erg_cm2_D0.2_above_dmin`, and checking their sources: 5218 and 5217 print the same
> unspecified wording, and 5219's footnote is not in its text layer at all. So the "D 0.2 above
> dmin" half is printed on none of them. 5201 stores what its sheet prints; the other three are
> **left alone** and the conflict is recorded with a two-way guard (method rule 4). Best next move
> for it: Kodak publication **H-1** *Image Structure*, cited by name on the sheet, absent from the
> corpus.
>
> ⚠ **AND THIS ADOPTION MOVES A RENDER**, unlike the dye set, which is inert schema-v7 storage. A
> stock carrying spectral data takes `spectral_balance_gains()` instead of the 600/550/450 nm proxy,
> and 5201's measured red layer peaks at **650** nm rather than the assumed 600, so tungsten light
> drives it harder: **+0.28 stop of red gain at 3200 K**, -0.17 at 10000 K, green the unchanged
> anchor. Both size and direction are asserted, so the change stays deliberate.
>
> **QUEUE ITEM C12 WAS FILED AGAINST TWO PROFILES AND THERE WERE SIX.** A sweep for mixed
> `[T1/T3]`-style tags found `KODAK_VISION2_500T_5218`, `_200T_5217`, `_250D_5205`,
> `KODAK_VISION_500T_5279`, `_200T_5274` and `_250D_5246` all resolving to **tier 3 with
> `fitted_from="analogy"`** -- every one owning its own Kodak sheet, **four with a sigma(D) shape
> traced from it**, and in all six the T3 half is one flagged scalar: `rms_granularity`, because from
> VISION onward Kodak prints granularity CURVES and no rms number. All six moved to **tier 1**
> (owner-approved), matching the two precedents already in `_UNTAGGED_TIER` -- `EASTMAN_5247_1983`
> is tier 1 with hand-fitted tone curves and `FUJI_SUPER_F125_8532` is tier 1 with a transferred
> red/blue f50, both larger residuals. ⚠ **The mechanism was closed by a CLASS guard, not by
> loosening the regex:** a mixed tag must now be listed in `_UNTAGGED_TIER` and may not resolve to 3
> (3 being exactly the value the regex falls back to, so an entry resolving there is
> indistinguishable from the bug). The strict regex is the feature -- it forces a decision on every
> future mixed tag instead of quietly picking a number.
>
> **Status 2026-08-25b (Kodak 7266 granularity panel, queue C29):** `verify.py` **338 PASS / 1
> FAIL**, 11 audits green, schema **v11 unchanged**, **160 stocks unchanged**, `film_names.txt`
> unchanged -- no ListBox shift. One stock gained a measured grain shape.
>
> **THE FIRST MEASURED sigma(D) ON A BLACK-AND-WHITE STOCK IN THIS FILE -- AND IT POINTS THE
> OPPOSITE WAY FROM WHAT WAS STORED.** Kodak TRI-X Reversal 7266's sheet draws its granularity panel
> and its characteristic curve against the SAME log-exposure abscissa, which is what makes the
> pairing possible without transferring a calibration between documents. 52 columns paired; **30
> kept.** The restriction is |dD/dlogE| > 0.5: where the characteristic curve is flat, one density
> maps to many sigma values and the inversion is ill-conditioned. On a reversal stock the flat zone
> is **dmax**, not the toe -- the opposite end from a negative.
>
> **ADOPTED on `KODAK_TRI_X_REVERSAL_200` and nothing else:** `sigma_shape_toe` **0.262** (at D
> 0.352), `sigma_shape_mid` 1.000, `sigma_shape_dmax` **2.829** (at D 3.089),
> `sigma_shape_measured=True`. The law is `sigma_D ~ D**1.078`, rms 0.038 decades.
>
> ⚠ **THE ESTIMATE IT REPLACES WAS WRONG IN DIRECTION, NOT MERELY IN SIZE.** The stored triple was
> 0.70 / 1.00 / 0.50 -- granularity FALLING 2x toward dmax. The sheet shows it RISING 2.8x. On
> reversal film dmax is the unexposed, fully developed silver, so rising is the physical direction;
> the estimate was a negative film's shape applied to a positive film. Nothing in the file recorded
> which direction to expect, so no existing check could have caught it.
>
> ⚠ **THE LEVEL IS DELIBERATELY NOT ADOPTED.** The same panel implies **22.3** at this file's
> NET-density-1.0 convention against the stored **10.0**. The sheet states the curve was obtained
> with "modified measuring techniques" and never defines them, so the level rests on an undefined
> convention while the SHAPE does not. `rms_granularity` stays 10.0 and the 22.3 is cited in the
> profile with that caveat as the reason.
>
> ⚠ **AN APPARENT 2.93x INTERIOR PEAK AT D 3.16 IS DISCARDED.** It appears in the raw pairing, it is
> smooth, and it lies entirely inside the ill-conditioned flat zone. Storing it would be storing an
> artefact of the inversion.
>
> ⚠ **SCOPE HELD.** The other **34** reversal stocks keep the contradicted 0.7/1.0/0.5 estimate --
> one measured sample is not a class (method rule 18) -- and the contradiction is now written into
> the `GrainSpec` docstring, `NotFound.md` and a counted `verify.py` guard instead of being
> "harmonised" away. The 68 monochrome NEGATIVE stocks are untouched for a stronger reason: 7266 is
> a reversal emulsion, and its rising shape is precisely what must not be transferred to negatives.
> Four new guards pin the shape, the absent peak, the kept rms and that 34-stock count.
>
> **Status 2026-08-25 (T-101 sigma(D) figures, queue C28):** `verify.py` **334 PASS / 1 FAIL**,
> 11 audits green, schema **v11 unchanged**, **160 stocks unchanged**, `film_names.txt` unchanged --
> no ListBox shift. One stored value moved.
>
> **THE MAIN RESULT IS A RETRACTION, AND IT IS RECORDED RATHER THAN QUIETLY FIXED.** T-101 Fig. 26
> (mean-signal-to-r.m.s.-noise ratio against mean optical density, log-log) was extracted cleanly:
> `log10(t/sigma) = -0.6648*log10(D) - 0.1738`, 1039 columns, rms 0.0063 decades. It self-validates
> twice on quantities the fit never sees -- sec. B.2 prints the five samples' mean transmissions, so
> their densities are known before tracing and four of five markers land within 2.2 %, and Fig. 21
> plots the same quantity on linear axes giving exponent 0.668 against 0.665. **And it still cannot
> be converted to sigma_D.** T-101 sec. 2 builds its sigma from a two-level model -- grains
> "uniformly opaque" with "infinitely sharp edges", t taking only the eigenvalues 0 and 1, giving
> sigma = sqrt(t(1-t)) and the eq. (4) lower limit drawn on the figure itself, approached "as the
> scanning aperture becomes vanishingly small". The measured sigma_t/t runs 0.39 to **1.64**, so the
> small-signal linearisation sigma_D = 0.4343*sigma_t/t is invalid across the whole plate. A
> mid-session result of "sigma_D = 0.648*D^0.665" is **withdrawn**.
>
> ✅ **A CONFLICT THIS PROJECT WAS ABOUT TO RECORD DOES NOT EXIST.** Mees Fig. 302 (Goetz-Gould G on
> a trace evaluator at a fixed densitometer aperture -- grains unresolved, Selwyn regime, which is
> where this file's 48 um `rms_granularity` lives) and T-101 Fig. 26 (the pinhole limit) are
> different regimes of the same physics. The apparent disagreement about whether B&W silver-negative
> grain turns over at high density was an artefact of the bad conversion. Method rule 4 is for real
> disagreements; asking "are these commensurable?" first is now a step in the method.
>
> **WHAT WAS ADOPTED, from PRINTED Table 3 (p35) with no tracing at all: grain size depends on
> DEVELOPMENT.** The table gives equivalent grain diameter against point gamma at two densities for
> two emulsions, and its own last column normalises by sqrt(point gamma). Refitting its eight rows:
> `D_eq ~ gamma**n`, n = **0.452** (Pan F, rms 0.0035 um), 0.396 (Tri-X), 0.425 pooled -- so the
> printed sqrt is marginally steep. Validated at 2 % against a number the fit never saw (Table 2's
> printed 1.5 um at gamma 1.0, D 0.43).
>
> ⚠ **That exposed a condition mismatch shipped the previous day.** `clump_um` was taken from Table
> 2's diameters, which were measured at the BBC's OWN development gamma. Five of six stocks happen
> to match their stored gamma; `ILFORD_PAN_F` does not -- stored 0.55 (Ilford's ID-11 contrast
> index) against the BBC's 1.0. **0.859 -> 0.655 um.** `EASTMAN_PLUS_X_5231` (0.68 vs 0.64) is
> deliberately NOT corrected: +2.5 % is far inside the upper-bound caveat those printed diameters
> already carry, and moving a number by less than its own stated uncertainty is false precision.
>
> ⚠ **Two dependences are now recorded in the `GrainSpec` docstring, neither of which the schema can
> express:** `clump_um` varies with development gamma (above) AND with density -- T-101 Fig. 21
> measures Pan F falling 1.726 -> 1.384 um across the tone scale, -20 %, with its t = 0.37 point
> reading 1.484 against Table 2's printed 1.5. Every stored `clump_um` is therefore a MID-SCALE
> REPRESENTATIVE AT ONE DEVELOPMENT CONDITION, and that is now written down where it will be read.
>
> **Status 2026-08-24b (F-125 family restructured, queue C27):** `verify.py` **331 PASS / 1 FAIL**,
> 11 audits green, schema **v11 unchanged**, database **161 -> 160 film stocks** (10 print stocks),
> so `film_names.txt` changed again and **the ListBox shifts a second time today** -- rebuild once,
> after both passes.
>
> **`FUJI_F125_8630` REMOVED. `FUJI_F125_8530` and `FUJI_SUPER_F125_8532` kept as independent
> profiles.** The deciding evidence is a sentence, not a curve: two issues of «Техника кино и
> телевидения» (1989 No.4, 1990 No.1 -- the latter a translation of Fuji's own 1988 symposium
> paper) print **Fuji's four-digit code rule in words** -- first digit 8 = colour negative, SECOND
> DIGIT = GAUGE (5 = 35 mm, 6 = 16 mm), last two digits = the film -- applied consistently in three
> tables across all five F-series stocks and matched by Fuji's own Super F-125 sheet ("35mm Type
> 8532 / 16mm Type 8632"). So 8530/8630 were one emulsion slit two ways; a gauge is
> `default_format`, and `8630` is now an alias. ⚠ The SAME rule keeps 8530 and 8532 apart: they
> differ in the LAST TWO digits, the part that names the film, and they measure differently --
> **rms 4.0 against 3.0 at identical speed** (125 tungsten / 80 daylight).
>
> **Adopted:** 8530 `rms_granularity` 5.4 (estimate) -> **4.0**, printed in 1989 No.4 Table 1 p70
> and verified against the page image; the convention is confirmed on the plate as 48 um at visual
> diffuse D 1.0, which is how the 8532 sheet defines its own 3.0.
>
> ⚠ **A third MTF measurement arrived and nothing was changed.** 1990 Fig. 3 traces to f50 ~33
> mm^-1, against the 1989 table's 0.60 at 30 mm^-1 and Honjo's nu_50 = 42. The other two traces on
> that plate agree with the table to 2-3 %, so the 8 % disagreement is specific to this stock.
> Method rule 4. It does reframe the conflict already recorded on 8532: that profile's
> Coltman-converted 32.07 looked like a regression against 42.0, and this third figure lands at 33
> -- two of three sources now cluster at 32-33 and Honjo's 42 is the outlier.
>
> ⚠ **Three figures left UNHARVESTED with stated reasons**, so a later pass does not re-litigate
> them: sigma(D) (1990 Fig. 4 -- F-125 and F-64 converge inside the drawn line width exactly where
> the only validating anchor sits), gammas (Fig. 1 draws F-125 SUPERIMPOSED on type A at matched
> speed, and the abscissa carries a 0.5-decade span bar and no numbers at all), spectral
> sensitivity (Fig. 6 states no density criterion). `NotFound.md` §1.5 carries the full account.
>
> **Status 2026-08-24 (T-101 Fig. 18 + the grain-size column, queue C25/C26):** `verify.py`
> **331 PASS / 1 FAIL**, 11 audits green, schema **v11 unchanged**, but the database moved
> **159 -> 161 film stocks and 9 -> 10 print stocks**, so `film_names.txt` MD5 changed to
> `c2b9e17e…` and **the ListBox shifts** -- rebuild.
>
> **The headline is a correction, not a harvest: `clump_um` was wrong by an order of magnitude on
> every stock that could be checked.** BBC Report T-101 measures six 1963 emulsions; four already
> had profiles here. Their stored grain-clump sizes were 1.90 / 11.0 / 5.0 / 19.0 µm against
> measured 1.43 / 0.83 / 0.86 / 1.26. Since `f_hi = 500/clump_um`, a stored 19 µm puts the grain
> rolloff at **26 c/mm** -- Fig. 18 shows Tri-X still at half power at **290**. The stored column
> was making grain low-frequency and blobby on stocks whose real spectra are broadband.
>
> **And the method lesson is the expensive one: READ EVERY TABLE BEFORE TRACING ANY CURVE.**
> Fig. 18 was digitised properly -- six dashed curves separated by **dash period**, a page that is
> **bowed rather than skewed** (97 px of movement on the W=0 line, so no rotation fixes it), two
> gridlines that are untrackable because one of them *runs along the 8374 curve* and its removal
> band deleted that emulsion, and an arc-length walker with a turn-rate limit because without one
> Pan F and 5302 dive down gridline stubs to W=0. It validates three independent ways. **Then
> Table 2 on p28 turned out to print the measured equivalent grain diameter of all six emulsions
> outright** -- the quantity the trace was for -- in a table already cited on four profiles since
> 2026-08-23 for its other columns. Stored values are the printed ones, through
> `D_eq = 1.7473 · clump_um`; the traces are what validate them and what proves `clump_gain` fits
> to **exactly 0.000 on all six** independently.
>
> ⚠ **Three limits recorded rather than papered over.** (1) Every adopted value is an **upper
> bound** -- p38 says the printed diameters are "expected to be greater than the true values",
> instrumental weighting uncorrected. (2) **The other 155 stocks were not touched:** six 1963 B&W
> emulsions do not license rewriting the colour negative and reversal column, so only the error's
> direction is on record. (3) The carrier **shape** is wrong in a second way -- a free exponent
> fits n = 1.80 (HPS), 2.01 (Tri-X), 2.43 (Plus-X), 2.4-4.1 (fine grain) against the file's fixed
> n = 2. That is a **renderer** change and was not attempted.
>
> **New profiles, all from the same document:** `EASTMAN_TRI_X_5223` (tier 2, the 35 mm cine
> negative whose numbers had been parked in `KODAK_TRI_X_320TXP`'s citation with a note that they
> belonged to a profile that did not exist), `KODAK_8374` (tier 3, 16 mm TV recording film, blue
> and U.V. sensitive) and `KODAK_5302` (tier 2, **a PrintStock**, so no ListBox shift -- and it is
> the *unity* of Table 4's granularity ladder that every T-101 grain number is anchored on).
> ⚠ `KODAK_TRI_X_400TX` deliberately did **not** move: T-101 measured the cine 5223 at 250/320
> A.S.A., not the ASA 400 still film. `verify.py` pins that non-move so a later pass cannot
> "finish the job". Renders change **texture only** -- `grain_reference_energy` renormalises, so
> `rms_granularity` still means what it meant. `NotFound.md` carries the full account.
>
> **Status 2026-08-23e (F-125 harvest + C21/C22, schema v11):** `verify.py` **316 PASS / 1 FAIL**,
> 11 audits green, schema **v10 -> v11** (`HalationSpec` grew three fields, all shipping 1.0, so
> renders are bit-identical but **a v10 reader would walk off the end of every HalationSpec** --
> rebuild), `film_names.txt` **unchanged**, so no ListBox shift.
>
> **The F-125 item came from the owner catching a wrong sentence in `NotFound.md`** -- "no Fuji
> F-125 document exists in this corpus", when a complete Fuji sheet (**Ref. No. KB-913E, (C)1999**,
> titled *FUJICOLOR NEGATIVE FILM F-125*) had been on disk all along. The root cause generalises and
> is now a rule: that sheet's footer, product name and logotype are **outlined vector art**, so
> `get_text()` returns neither the product name nor the date, and a text-layer reading of the file
> looks like it has neither. **"Not in the text layer" is not "not printed" -- render the page.**
>
> **Harvested from it, all traced from the sheet's own vector panels:** three characteristic curves
> (rms 0.005-0.009 D), spectral sensitivity (peaks 469/553/645 nm), and **f50_g 32.07 c/mm by
> Coltman's square-to-sine conversion** of the contrast transfer function -- an item this project had
> previously recorded as unusable "without the chart's duty cycle", which a rectangular wave chart
> does not require. `FUJI_SUPER_F125_8532` moves tier 2 -> **1**; queue **C11 closed for it**.
> The same method was applied in the same pass to the sister sheet `FUJICOLOR_SUPER_F500_8572`
> (f50_g 56 -> **20.21** c/mm; its "cyan shadows" description **retracted as unsupported**).
>
> **Two hazards recorded rather than smoothed:** both Super-F sheets carry a **mis-labelled,
> non-monotonic exposure axis** (`-4.5 -3.0 -3.5 -2.0 ...`), settled at first-gridline -4.5 by
> physics and cross-checked between the two sheets to 0.08 stop; and the converted 32.1 c/mm
> **contradicts** Honjo's 42.0 for the 8530 it replaced, while Fuji sells 8532 on "dramatically
> increased sharpness". Both figures stay on record.
>
> **C21/C22:** per-channel halation radii (all 1.0, pinned by guard -- the geometry bounds the real
> ratio near 1.1, so derived values would look measured while moving a render ~1 %) and Callier's
> coefficient as **film x scanner geometry**, `D_read = dmin + (D - dmin)*(1 + specular*(Q - 1))`,
> inert at the shipped `scanner_specular = 0` and inert on colour at any setting. ⚠ The factor has
> to reach the anchor solve, the print chain's mid-grey reference AND the pixel pass: with only two
> of the three, mid grey moved **+54/255** on DOUBLE-X instead of contrast changing.
>
> **Status 2026-08-23c (C24 adopted: the f50 estimating rule is replaced where the
> measurements reach):** `verify.py` **303 PASS / 1 FAIL**, 11 audits green, schema **v10
> unchanged**, `film_names.txt` **unchanged** -- data-only rebuild, no ListBox shift.
>
> **Five measured triples adopted** -- 5217 `33.9/58.1/67.4`, 5218 `37.6/54.6/69.7`,
> 5245 `37.2/83.8/100.5`, 5248 `37.4/75.1/111.2`, 5279 `41.1/73.1/76.1` -- each with its
> measured adjacency overshoot and, except on 5279, its measured rolloff exponent.
> **mtf_measured stocks: 3 -> 8.**
>
> **Two stocks take a MIXED triple, and it is labelled as such:** 5205 and 5293 have a
> measured green and blue but a red their sheets emit in fragments, so their red carries the
> family anchor and `mtf_measured` stays unset -- two measured records and one class estimate
> is not a measured stock.
>
> **Five modern Kodak cine stocks had their RED re-anchored to 36.0 cycles/mm** (5203, 5207,
> 5213, 5219, 5246), replacing the fixed-ratio rule. Green and blue were left alone
> deliberately: the measured blues run 0.96-1.43x their stored values with no consistent
> factor, so only red is a constant to anchor on. ⚠ **Scope is narrow on purpose** -- modern
> Kodak cine colour negatives whose stored blue lies inside the measured 55-111 cycles/mm
> range. `EASTMAN_EXR_500T_5296` (blue 42) and every pre-1990 stock are excluded, as is every
> other manufacturer, and `verify.py` now asserts 5296 keeps its own 30.0 so a later
> "finish the family" pass fails instead of guessing. **63 colour stocks still carry an
> estimated triple.**
>
> ⚠ **RENDER IMPACT IS LARGE, and larger than C13's, because red moves by up to 2.2x.**
> Bar-sweep target, grain and flare off, worst channel delta: **5203 45.7/255**,
> 5248 22.9-26.8, 5217 21.8-26.1, 5219 7.8-8.8 -- measured at both 48 and 193 px/mm, and
> unlike C13 the effect is visible at preview size because red now differs by tens of
> cycles/mm, not a few.
>
> ⚠ **A CONFLICT SURFACED AND IS RECORDED, NOT PAPERED OVER.** 5245's and 5293's measured
> BLUE records reach 50 % modulation at 100.5 and 114.6 cycles/mm against a stored limiting
> resolution of 100 lines/mm. ISO resolving power is a COMPOSITE three-layer reading while
> f50 is per record, so a sharp blue record above the composite limit is not impossible -- and
> the stored pair is the weaker number of the two: 5248's sheet prints "TOC 1.6:1 / TOC
> 1000:1 ... 80 / 160 lines/mm" and matches what is stored, while a text search of the 5245
> and 5293 sheets finds no "lines/mm" at all. `verify.py` now compares the GREEN record and
> reports the blue exceedances as information. `RESULT_2026-08-23b_c2b.md`.
>
> **Status 2026-08-23b (C2b: nine more colour MTF sheets, and the extractor that had to be
> repaired to read them):** `verify.py` **300 PASS / 1 FAIL**, 11 audits green, **no profile
> data changed** — schema v10 and `film_names.txt` untouched.
>
> **3 traced sheets / 7 curves became 12 / 26**, and Agfa joined Kodak. ⚠ **Most of this pass
> was four defects in `mtf_vector.py`, and every one of them returned numbers that looked like
> MTF measurements:** (1) the 1990s technical sheets emit all three records as ONE path
> object, so H-1-5218 reported f50 69.7 off a trace running along blue, jumping to green and
> finishing on red; (2) the log GRID passed for a curve and was handed back as 5245's GREEN
> record at f50 236.8 with response to 190 %; (3) label matching by nearest curve
> double-claimed on 5245, and ranking by height instead swapped red and green on 5248 -- whose
> red STOPS at 115 c/mm while green runs to 191 -- producing a red record sharper than green,
> which no colour negative can be; (4) a fragment has an f50 and it is meaningless, 5293's red
> arc starting at 53 % response reporting 32.0. **All four were found on the new `--overlay`
> render and none by reading the numbers** -- the other two plot readers in this project
> already had that gate and this one did not.
>
> **C13's layer-depth hypothesis: the ORDERING is real, the MAGNITUDES are not.** `q_R <= q_G
> <= q_B` holds on 8 of 8 stocks with two or more records, but red spans 1.89-2.77 and blue
> 2.38-3.42 (sd 0.32-0.37), so C13's "both reds cluster at 1.84-1.89" was a two-sample
> illusion. **q therefore cannot be derived for the 156 unmeasured stocks and stays per-stock
> measured**, which confirms the `mtf_measured` design rather than replacing it. The power law
> still beats the Gaussian on all 26 curves (1.1x-5.8x, rms 0.0095-0.132).
>
> **C24 is answered, and not the way it was framed.** Seven stocks now have a complete
> per-record measurement. Red f50 reads **32.1 33.9 35.4 37.2 37.4 37.6 41.1** -- mean 36.4,
> spread +-13 % -- against green spreading 52 % and blue 70 %. **Red f50 is effectively a
> CONSTANT ~36 cycles/mm across 1989-2005 and three product families**, so it does not scale
> with the stock's sharpness and no rule of the form `f50_r = k * f50_b` can express it at any
> k. The estimates were 1.12-1.72x too sharp in red AND 0.70-0.83x too soft in blue.
>
> ⚠ **The cross-maker check C24 asked for cannot be made from this corpus.** The Agfa Vista
> sheet is the only non-Kodak MTF on file and it prints ONE visual-weighted curve
> ("Densitometry: visual filter (V-lambda)"), not three records: f50 50.0, q 2.63 at rms
> 0.039, +11.7 % overshoot -- itself 1.26x softer than that stock's own estimate.
>
> **Five measured triples (5217, 5218, 5245, 5248, 5279) are traced, pinned and NOT adopted**
> -- that is one owner decision, with the rule question beside it. `RESULT_2026-08-23b_c2b.md`.
>
> **Status 2026-08-23 (C1e per-layer grain, and C8 wires the last inert data family):**
> `verify.py` **300 PASS / 1 FAIL**, 11 audits green, schema **v10 unchanged**, `film_names.txt`
> unchanged -- no ListBox shift. ⚠ **But the plugin DOES need a rebuild: stage 8's signature
> changed** (see below).
>
> **C1e was unlocked by disproving its own premise.** The item existed because the raster
> granularity extractor was suspected of reading ~1.32x high on the sigma axis. It is not: the
> 5219 panel's own right-hand tick comb, fitted on the stored calibration, reproduces
> **0.001-0.100 at all 15 ticks within 1 %** (5203 within 1.3 %). The 1.32x was an ABSOLUTE-D
> comparison; read at NET 1.0 the two documents differ by 1.12x in green and 1.25x in blue --
> real, a third the size, and not an axis error. So the raster family's per-layer RATIOS are
> usable, and **three stocks were adopted rather than one**: 5219 r/g/b **5.92 / 6.60 / 17.84**,
> 5207 **rms_b 8.92**, 5203 **rms_b 4.71**, every green frozen as agreed. 5219's blue grain rises
> **x1.52 on screen** (sigma 2.33 -> 3.60 per 255); red falls slightly; green does not move.
>
> ⚠ **5213 stays on the heuristic and a guard pins it there** -- its sheet prints the three
> granularity curves as one bold band, so there is no blue track to read. And the corpus-wide
> ladder (`b = 1.30x`, `r = 1.10x` of green, still filling **54** colour negatives) is
> **measured to be wrong in magnitude for blue and in SIGN for red on all nine sheets that carry
> a per-layer read** (b/g 1.81-2.79, r/g 0.75-1.05) -- and deliberately NOT rescaled, because all
> nine are Kodak cine negatives and the blue ratio tracks stock SPEED, not any constant.
> Recorded as a settled refusal with its numbers, not as an open item.
>
> **C8: reciprocity now renders.** `RenderSettings.exposure_time_s` and
> `AlgoControls::exposureTimeS`, both **0.0 = not stated and inert**, so every earlier render is
> bit-identical (asserted over 159 stocks x 3 channels). New `film_sim.reciprocity_log_shift()`
> and its C++ twin **`AlgoReciprocity.hpp`**, applied **inside stage 8 on the log exposure** --
> after everything optical, before the curve, and onto the RETAINED log-E plane so stage 8b sees
> the same effective exposure. `cpp_parity.py` gains a **third family** (5724 probes, 159 stocks
> x 12 times from 1e-5 s to 3600 s, worst 1.0e-07 decades) and it probes the plugin's own header.
>
> **Seconds, not shutter angle, and that was decided on evidence:** angle / frame rate spans only
> 1/1000-1/24 s, and every sheet in the corpus prints *no correction needed* across exactly that
> span -- an angle control would provably never do anything.
>
> ⚠ **The data was the bigger half. 15 measured `ReciprocityTable` entries** were read from the
> stocks' own sheets here (5205, 5217, 5218, 5219-**brochure**, 5201, 5246, 5274, 5279, 5248,
> 5231, 5247, F-125 8532, F-500 8572, ETERNA Vivid 8547, VISTA 200), taking the total **6 -> 21**.
> The reason it mattered: **seven of them carried `p = 1.0` and rendered NO reciprocity while
> their own sheets print a correction**, and the rest rendered about half of theirs -- a single
> Schwarzschild exponent has nowhere to put an offset the film has already lost by 1 s (5205's
> sheet prints +2/3 stop at 1 s; its exponent delivered +1/3).
>
> ⚠ **An error of mine that the 5205 sheet caught.** The CC-filter arithmetic first ADDED the
> filter's density to the worst-losing record, giving 1 stop where the sheet says 2/3: right
> channel ordering, wrong level, entirely plausible in a frame. Both instructions act on one
> frame -- the lens opens by the printed stops on all three records and the filter takes part of
> it back -- so the record the filter does NOT attenuate loses exactly the printed stops. A guard
> now asserts that over all 21 tables.
>
> ⚠ **Two defects recorded and NOT patched, because the documents are not in this corpus copy.**
> (1) The Agfa/Konica exponents were fitted under a different reading of a printed correction
> ("the film needs a longer exposure" rather than "the loss at the stated time"), which delivers
> **0.766 stop where the source prints 1.0 -- 24 % light**; the corrected values are written out
> next to `_RECIPROCITY_TABLES`, and `PDF/PROFILES/KONICA/*` and `SOVIET STANDARDS/*` are absent
> here. (2) Kentmere's stored stops (0.517 / 0.599 at 10 s) do not reproduce the formula their own
> source string quotes (0.864 / 0.997). AGFA_VISTA_200 is the one that COULD be fixed and was --
> its printed point is on file in English and it now carries a measured table.
>
> ⚠ **REBUILD NOTE.** `AlgoStage08_CharacteristicCurve` gained a trailing
> `const HighPrecType logEShift[3]`. Declaration, both definitions and both call sites are
> updated; any other caller fails to compile, which is intended. The **AVX2 TU could not be
> compiled here** (`FastAriphmeticsAVX.hpp` is not in the uploaded set), so that three-line edit
> is verified by inspection against its scalar twin.
>
> **State:** **300 PASS / 1 known FAIL**, 11 audits green, compile clean on 18 TUs, schema v10,
> 159 stocks. `RESULT_2026-08-23_c1e_c8.md`.
>
> **Status 2026-08-20c (C13: 5274's MTF adopted, and the finding is bigger than the profile):**
> `verify.py` **284 PASS / 1 FAIL**, 11 audits green, schema **v10 unchanged**, `film_names.txt`
> unchanged -- **no ListBox shift, data-only rebuild**.
>
> **`KODAK_VISION_200T_5274` now carries its measured MTF** from H-1-5274 p3 (plot F010_0006AC),
> a panel that had never been traced: **f50 35.4 / 68.8 / 74.0 cycles/mm** against the stored
> estimate 56.0 / 64.0 / 72.0, adjacency **0.162** (was 0.09), rolloff **q = 2.94**. Third stock
> with a traced MTF.
>
> **Green and blue confirmed the estimate to 7 %. Red was 1.58x too sharp** -- and that is a defect
> in the ESTIMATING RULE, not in this profile. The rule puts `f50_r / f50_b` near **0.78**; across
> the 92 colour stocks that still carry an estimate the stored ratios sit **72 in 0.75-0.85**, while
> both stocks measured per-record land at **0.478** (5274) and **0.578** (5201). New queue item
> **C24**, and it explicitly refuses to rescale 92 profiles from two measurements of one Kodak
> family -- that is the method-rule-18 error. It becomes answerable if C2b's next sheets agree on
> the layer-depth pattern, in which case the ratio can be DERIVED from the layer stack.
>
> ⚠ **RENDER IMPACT IS SCALE-DEPENDENT, and smaller than 1.58x sounds.** Measured on a bar-sweep
> target: worst **3.9/255 at 48 px/mm** (a 2K-ish 35 mm frame), **7.1/255 at 96 px/mm**,
> **11.1/255 at 193 px/mm**. f50 lives at 35-74 cycles/mm and a 2K render never reaches those
> frequencies, so at normal preview sizes most of the visible change comes from the ADJACENCY term,
> not from f50. **The f50 correction earns its keep at scan resolution.** Worth saying before anyone
> judges the change on a 1080p preview and concludes it did nothing.
>
> ⚠ **And a measurement error of my own, recorded because it is the instructive part:** the first
> impact test reported 0.1/255 and I nearly published it. The test image was 96 px wide for a
> 24.9 mm frame -- about 3.9 px/mm -- where a 35-versus-56 cycles/mm difference cannot show at all.
> A null result from a target that cannot resolve the effect is not a null result.
>
> **C2b's remit is now narrower on purpose:** trace COLOUR sheets. A monochrome sheet has one
> record and therefore cannot test a per-record ratio, which is what C24 needs.
>
> **State:** **284 PASS / 1 known FAIL**, 11 audits green, compile clean on 18 TUs, schema v10,
> 159 stocks. `RESULT_2026-08-20c_c13_mtf.md`.
>
> **Status 2026-08-20b (the DIR-coupler stages get a parity test):** no profile data
> changed -- **159 stocks, schema v10, `film_names.txt` unchanged**, so no ListBox shift and
> no plugin rebuild. `verify.py` **279 PASS / 1 FAIL**: the FAIL baseline is **down from 2
> to 1**. 11 audits green.
>
> **The question that started it was "are we modelling inter-image effects?" and the answer
> was yes** -- both halves, live in both renderers: `InterimageSpec` / stage 8b /
> `AlgoStage08b_Interimage` for the vertical (cross-layer) half, `CouplerSpec` / stage 9 /
> `AlgoStage09_DirCoupler` for the lateral (adjacency) half, called from `AlgorithmMain.cpp`.
> The coefficients are not guessed from stock generation: they are solved per stock against
> published patent measurements and `verify.py` re-derives the published IIE percentages to
> < 1 pp. Measured contribution, by disabling the stages and re-rendering: up to **143/255**
> on Velvia's saturated patches, 23/255 on Portra's reds.
>
> **What was actually missing was a cross-check.** `cpp_parity.py` covers the grain and MTF
> laws only, so the largest COLOUR effect in the chain existed twice, in two languages, with
> nothing comparing the two. That is the configuration that produced the C1b bug.
> **`interimage_parity.py`** now compiles a probe against the plugin's OWN
> `Algo_08_Sim.cpp` / `Algo_09_Sim.cpp` -- the only audit that tests shipped C++ rather than
> generated code -- and it found two real defects on its first run.
>
> ⚠ **Defect 1, FIXED: the density floor was on the wrong side of the function boundary.**
> C++ stage 9 ends with `MAX_VALUE(rO[x], ALGO_ZERO)`; Python clamped one line later, inside
> `simulate()`. The two PIPELINES agreed and the two FUNCTIONS did not -- a **0.26 D**
> disagreement on Velvia, a reversal stock whose ramp drives density negative. The floor now
> lives inside `apply_dir_couplers`. Rendering unchanged: max(max(x,0),0) is max(x,0).
>
> ⚠ **Defect 2, OPEN: the adjacency term is not the same effect in the two renderers at
> ordinary render sizes.** Python blurs by the analytic Gaussian transfer, C++ by a truncated
> separable kernel. Measured across sigma: they agree to 6e-5 above ~1.2 px and diverge to
> **1.5e-1 at 0.4 px**. Stored `edge_um` is 9-13 um, so at 40 px/mm -- a 35 mm frame about
> 960 px wide -- the edge sigma is **0.36-0.60 px** and the outputs differ by up to 2.6e-2 D.
> Queue **C16**. Separately, C++ disables both coupler terms below 0.25 px and Python does
> not, which parts the two renderers below **27.8 px/mm** on the edge term (queue **C17**).
>
> **The stale guard was CONVERTED, not deleted.** "neighbour pairs couple harder than the far
> red-blue pair" asserted a PER-DISTANCE asymmetry the database deliberately does not store,
> because US4725529A Table 1 -- inhibitor in the developer, three separate single-layer
> coatings, no layer stack, asymmetry persists -- says it is per RECEIVER. It was unpassable by
> construction and had been parked in the FAIL baseline as "known". Deleting it would have
> changed zero pixels; it now asserts the property the evidence supports.
>
> **Also closed:** `doc/CHANGES_2026-08-03_v5_interimage.md` **was cited from four places and
> was not on disk** -- written, reconstructed from the code of record, with a section stating
> plainly what could not be recovered. And the interimage tier contradiction: the generator
> said "tier 3 for every stock without exception" while `InterimageSpec` had said tier 2 since
> 2026-08-03. Generator corrected.
>
> **State:** **279 PASS / 1 known FAIL**, **11 audits green**, compile clean on 18 TUs, schema
> v10, 159 stocks. `RESULT_2026-08-20b_dir_parity.md`.
>
> **Status 2026-08-20 (the KODAK folder review: one stock measured end to end, C5, C6, and three
> answered queue items):** database **157 -> 159 stocks**, so `film_names.txt` MD5 moved
> `c37a188b...` -> **`e8dc2cb9...`**; **schema v10 is UNCHANGED**, so no sampler or calling-convention
> change rides along this time.
>
> ⚠ **The four PDFs the owner named were not on disk at the start of the session, and when they were
> re-uploaded, THREE OF THE FIVE turned out to be documents this database had already mined** --
> `V200T.pdf` is byte-identical to `5274.pdf`, and the 5201 and 5247 files are the same publications
> as the copies already held (identical text layers; for 5201, identical vector geometry to 0.01 pt).
> Re-reading a sheet the database was built from would have "confirmed" its own numbers. **Check
> identity before treating a file as a new source.**
>
> **Added: `KODAK_VISION2_50D_5201`** from Kodak H-1-5201 (New 10-2005) -- and it is the **first stock
> in this database whose characteristic curves, sigma(D) shape, grain level AND per-record MTF are all
> traced from one sheet**. The sheet prints no scalar for grain, sharpness or gamma; on grain it says
> only "the measured granularity is exceptionally low". Measured: rms r/g/b **4.36 / 4.51 / 9.63** at
> net 1.0; sigma(D) **0.54 / 1.00 / 0.89** peaking at only **1.20x** (the flattest in the corpus, where
> the other six colour negatives run 1.38-1.62x); f50 **32.1 / 49.7 / 55.5 cycles/mm** -- the first
> per-record MTF in the file and the first direct confirmation of the blue-sharpest layer order.
>
> ⚠ **And the panel titled "SENSITOMETRIC CURVES" was refused** (now method rule 22). At brochure
> scale it draws the red and green records as **six-vertex polylines** over +-1.8 decades with neither
> dmin nor dmax on the plot; six free parameters fitted those six points at **rms 0.0004 D**, an
> interpolation wearing a measurement's error bar. The curves used are the dense ones inside the
> *granularity* panel on the same page (100-125 samples over 4.1 decades); the coarse panel is asked
> only for the abscissa origin, which is one parameter and which only it states.
>
> **Added: `FUJI_SUPER_F125_8532`** (queue C6), tier 2: printed scalars are `[T1]` -- EI 125 at 3200 K,
> **rms 3.0 with its net-1.0 convention printed on the sheet**, achromatic reciprocity as stated -- while
> the curves and f50 are a **flagged `[T2]` transfer from 8530**. ⚠ Its sharpness panel is a **contrast
> transfer function against a rectangular wave chart, not an MTF**, and was not read.
>
> **`EASTMAN_5247_1983` re-tiered 3 -> 1** (queue C5). ⚠ Doing it surfaced a latent bug class: a mixed
> `[T1/T2]` tag does not match the tier regex, so it falls to 3 unless listed explicitly -- which is
> why `KODAK_VISION2_500T_5218` and `_200T_5217` (`[T1/T3]`) have been sitting at tier 3 while owning
> their own sheets. Queue C12.
>
> **Three queue items moved on their own evidence.** **C1e** is answered in the direction opposite to
> the one it was opened in: the 5219 brochure's net-1.0 green reads **6.37 against a stored 6.6**
> (0.97x) where the raster sheet implied 8.98 (1.36x), so the raster extractor does read ~1.3x high
> and **the four VISION3 rms values should NOT be re-levelled** -- but the same reading puts 5219's
> blue at 15.45 against a stored 8.58, so the per-layer ladder is 1.8x low and that needs a decision.
> **C2b** went from 1 traced MTF curve to 7: the power law wins on all seven, q spans 1.84-3.42, and
> it clusters by **layer depth** rather than by film (both reds 1.84-1.89, both blues 3.38-3.42).
> **C13 (new)**: 5274's MTF panel, never traced before, says its stored **red f50 is 1.58x too sharp**
> -- a defect in the estimating rule rather than in one profile, since it affects every colour stock
> whose triple is an estimate. Measured and pinned, **not adopted**: it needs one owner yes/no.
>
> **Also reviewed, nothing adopted:** `Kodak Ektar 125 - Jack and Sue Drafahl.pdf` is *PHOTOgraphic*,
> September 1989 -- a **magazine review with no sensitometry at all**, though it documents the
> eleven-layer construction in full (queue C14, and Ektar 125 remains unrepresented); and a VISION3
> **DI 2254/5254** 2-pager whose Arrhenius table is the **first source-backed `AgingSpec` data in the
> project** (queue C15). ⚠ `EASTMAN_5254_1968` in the database is a 1968 camera negative -- VISION3
> 5254 is a different film with a reused number.
>
> **State:** **275 PASS / 2 known FAIL**, **10 audits green**, compile clean on 18 TUs, schema v10,
> 159 stocks. `RESULT_2026-08-20_kodak_5201_c5_c6.md`.
>
> **Status 2026-08-19 (GEVAERT: two stocks added, one curve set re-traced):** queue items G1 and
> G3, owner-approved. Database **155 -> 157 stocks**, so `film_names.txt` MD5 moved
> `ae9e4be3...` -> **`c37a188b...`** and **every ListBox line index after `GEVACHROME_600` shifted**.
>
> **Added: `GEVACHROME_600` and `GEVACHROME_605`** -- Agfa-Gevaert's 1968 colour REVERSAL *camera*
> films for television, from Rens & Van Bets, *Kino-Technik* 1968 Nr. 10, printed pp. 260/262/264/266
> (German, image-only scan, read from the page images). Printed and adopted: **18 DIN (50 ASA)** and
> **23 DIN (160 ASA)** tungsten at 3200-3400 K; per-layer gammas **1.25 / 1.25 / 1.45** and
> **1.25 / 1.25 / 1.35** (yellow/magenta/cyan -> blue/green/red records); the nine-layer stack with
> both a yellow *and* a magenta filter layer; the documented push to **26 DIN (320 ASA)** on +45 s of
> first development. The daylight indices (25 / 80 ASA with CTO 12) are **filter-derived and
> deliberately not stored**, the same treatment already applied to the Kodak and Konica stocks.
> ⚠ The paper prints **no granularity figure of any kind**, no resolving power, no Dmax and no
> reciprocity -- those fields are `[T3]` estimates, their comments say so, and `verify.py` asserts the
> provenance still records the absence so the estimate cannot later read as a measurement.
>
> **Re-traced: `GEVACOLOR_NEG_682`'s three characteristic curves** from Vervoort & Stappaerts,
> *SMPTE Journal* 89(9) September 1980, printed p652, Fig. 10. Digitised at **one sample per pixel
> column** off the native-resolution embedded scan (`pdfimages`, not a re-render -- re-rendering at
> 200 dpi throws away 40 % of the columns): **589 / 513 / 437 samples** for B / G / R, then the
> 6-parameter ToneCurve fitted to all of them at **rms 0.0063 / 0.0040 / 0.0055 D**. dmin is pinned to
> the measured plateau median rather than fitted. **The figure prints "gamma = 0.57" and the trace
> reproduces it at 0.5677** -- an external check on a number the extractor never saw, now pinned in
> `verify.py`. The old entry carried the printed 0.57 for all three layers; the plot shows a spread
> (0.506-0.568), and dmax moves from 2.08/2.51/2.75 to **1.48 / 2.01 / 2.26**.
>
> ⚠ **These are paper scans and the pixels are the only truth, which cost three axis-finding rules
> before one worked.** The 1980 page is skewed **1.40 deg**, so both axes are fitted as *lines*; taking
> "the darkest column" would have cost 0.05 D. *Maximum ink mass* picked the figure's right frame;
> *leftmost tall ink column* picked the rotated "DENSITY" caption; **leftmost near-full-height stroke
> whose per-row positions fit a line to <1.5 px** is the axis. Horizontally, "lowest band with ink
> across the width" locked onto the printed x-label row and fed the tracer the axis as a curve.
>
> ⚠ **Two self-referential loops were found and closed** -- both would have produced plausible numbers
> that could not be reproduced. (1) The curve fit was seeded with the stored profile, i.e. with its
> own previous output; the red record then jumped to a different local minimum on the next run
> (gamma 0.5446 at rms 0.0109 -> **0.5056 at rms 0.0055**, the better fit, now stored). Start points are
> now data-derived with a fixed multi-start grid. (2) The inherited abscissa origin was read live from
> the profile it feeds and drifted 0.002 D after adoption; it is now a frozen constant.
>
> ⚠ **What the 150 ppi 1968 scan could NOT give.** Its three layer curves lie within **1-2 px** of
> each other and coincide at the right end -- consistent with the paper printing the same gamma 1.25 for
> two of them -- so no three-way separation was attempted. Dmax is a measured **lower bound** (2.2);
> toe/shoulder softness is a declared `[T2]` transfer from `GEVACHROME_902`. **Acquisition ask G5: a
> 300+ ppi grayscale re-scan of printed pages 260/262/264** unblocks the MTF and spectral traces (G2)
> and would upgrade both new profiles.
>
> **`cpp_parity.py` corrected on its first real test.** It reported a 1.5e-02 disagreement the first
> time a curve was re-traced -- and the law was innocent: audits run *before* codegen, so the C++ copy
> still held the previous dmax. It now passes **identical GrainSpec fields, dmin, dmax and density as
> literals** into the C++ side, so it tests the function rather than codegen freshness (worst
> disagreement over **4710** probes: 2.67e-07), needs no database TUs to compile, and guards the
> positional initialiser against an upstream field insertion.
>
> **Also reviewed, nothing adopted:** Webers & Westendorp, *FKT* 33(7) 1979 pp. 245-247 -- the
> Gevachrome II process in 15 steps with formulae and pH, plus four type numbers, but **no
> sensitometry at all**; Enticknap 2013 -- **no Gevaert stock data anywhere** in the body text;
> `Gevachrome902.pdf` -- a **byte-identical duplicate** of the AGFA copy already cited.
>
> **State:** **255 PASS / 2 known FAIL**, **9 audits green**, compile clean on 18 TUs, schema v9,
> 157 stocks. `RESULT_2026-08-19_gevaert.md`.
>
> **Status 2026-08-18h (the grain LEVEL, and a wrong finding corrected):** queue items **C1b +
> C1d**, both approved by the owner. `rms_granularity` now means what the manufacturer's footnote
> says it means, and the sampler normalises there.
>
> **The convention is printed, and it is NET density 1.0.** Kodak 5248 p1: *"Diffuse RMS
> Granularity\* Less than 5 / \* Read at a **net** diffuse visual density of 1.0, using a
> 48-micrometre aperture"*; identical footnote on DOUBLE-X 5222 p1. `grain_sigma()` and the generated
> `FilmGrainSigma()` now normalise at `dmin + 1.0`, and `film_sim.py` stage 11 multiplies by the rms
> field alone. **Schema v8 → v9: a change of MEANING with no change of layout** — v9 data paired with
> a v8 sampler compiles clean, runs clean and renders the wrong level, which is exactly what a
> version number is for. ⚠ The generated header's calling convention is now the **opposite** of v8's:
> multiply by rms and nothing else; the old "multiply by your own `sqrt(D − dmin + fog)` at D = 1.0"
> double-counts.
>
> ⚠ **A finding reported in the 2026-08-18g entry below was WRONG, and this is the correction.** That
> entry (and the C1b queue item) described a per-channel grain factor spanning **2.8× on 43 stocks**
> as a channel-balance defect. It was an artefact of assuming *absolute* density 1.0. At **net** 1.0
> the legacy law is `sqrt(1 + fog)` — dmin cancels — so the factor is **identical in all three
> channels** and there was never an imbalance:
>
> | stock | factor at net 1.0 (r/g/b) | factor at absolute 1.0 (r/g/b) |
> |---|---|---|
> | 5246 | 1.086 / 1.086 / 1.086 | 0.985 / 0.728 / 0.424 |
> | KONICA_IMPRESA_50 | 1.058 / 1.058 / 1.058 | 0.959 / 0.707 / 0.346 |
>
> **What C1b actually costs: a uniform 4–8 % amplitude drop** (`1/sqrt(1+fog)`) on the 149 stocks
> without a render-fitted rms, shape and channel balance untouched.
>
> **C1d, also smaller than claimed.** Re-levelling the six vector-traced negatives at net 1.0 gives
> **0.91–1.28×** (median 1.05) — 5245 4.2→4.10, 5246 5.3→6.78, 5248 5.6→5.87, 5274 5.8→6.68,
> 5279 8.3→8.74, 5218 7.3→6.65. The earlier "1.3–1.6× understated" was the same error. **The real
> data change is per-layer:** measured r/g/b triples give **blue 1.9–2.8× green**, where the
> `GrainSpec` docstring's tier-2 ladder assumed 1.3× for every colour negative. ⚠ One conflict is on
> file and was not averaged: 5248 is the only one of the six printing a scalar — "Less than 5" — and
> traces to 5.87 green / 4.42 red at that same net density.
>
> **Policy (a) for the render-fitted values.** The audit C1b required found exactly one family:
> `SVEMA_FOTO_65` ("still [T1] (fitted through the full pipeline)") and `SVEMA_FOTO_250` ("Tuned
> through the FULL PIPELINE … swept against rendered output"), plus `_32` and `_130` which are
> sqrt-speed scalings of those fits. All four had their rms multiplied by their own `sqrt(1+fog)`
> (8.5→9.617, 11.5→13.212, 18.0→20.680, 33.0→38.766) and **render identically to before** — asserted
> to < 0.2 % over D 0–3. Two stale claims were cleaned up in passing: the profile comment *and* the
> provenance citation for `EASTMAN_5247_1983` still defended "rms 13.0 is pipeline-calibrated", a
> value queue item E0 had replaced with 5.0 the day before.
>
> **Deferred on evidence:** the four VISION3 rms values. Their implied corrections (1.36–1.70×) sit
> ~1.5× above their vector-traced siblings', and the 5219 brochure reads a uniform 1.32× below the
> 5219 raster trace — two independent signs that `vision3_granularity.py`'s σ axis, not the film, is
> off. Queue item **C1e** re-derives it first.
>
> **New audit: `cpp_parity.py`,** registered in the build. C1b had to change one law in two
> languages — numpy `interp` on one side, a hand-rolled four-anchor insertion sort on the other —
> and the only previous cross-check was a manual one-off from a finished session. It now compiles
> a probe against the generated header, walks the real `GetFilmDatabase()`, and compares 4650
> values: **worst disagreement 2.7e-07**. It also asserts the measured branch is genuinely
> exercised, so the check cannot pass by both sides falling back to the legacy law. ⚠ Its first
> version expected 11 stocks to differ from the legacy law at their stored interior peak and got
> 10 — not a bug: 5285 is a reversal stock whose σ(D) rises monotonically, so its maximum IS the
> dmax anchor and `sigma_shape_peak` is legitimately 0. The probe moved to the traced dmax anchor.
>
> **Verify: 240 PASS / 2 known FAIL.** The bit-for-bit legacy guard was **replaced, not repaired** —
> it asserted an identity C1b makes false by design on all 155 stocks. In its place: the shape-only
> form (legacy law ÷ one constant, no dmin term), **the level contract** (amplitude is exactly 1.0 at
> net 1.0 for all 465 stock-channels), its converse (a masked stock must *not* be normalised at
> absolute 1.0, so re-introducing the old reference fails loudly), and an **empirical end-to-end**
> measurement — grain field × amplitude, aperture-integrated at 48 µm, reproducing 5246's stored
> 7.03 / 6.78 / 12.56 within 5 %. `RESULT_2026-08-18g_C1b_C1d_level.md`.
>
> **Status 2026-08-18g (the σ(D) HARVEST is COMPLETE — every vector granularity plot in the
> corpus is now read):** queue item C1c. `granularity_vector.py` went from **one sheet to
> eight**, all eight green under `--assert`, and **six colour negatives** gained a measured
> σ(D) shape: `EASTMAN_EXR_50D_5245`, `KODAK_VISION_250D_5246`, `EASTMAN_EXR_100T_5248`,
> `KODAK_VISION_200T_5274`, `KODAK_VISION_500T_5279`, `KODAK_VISION2_500T_5218`. Eleven
> stocks now carry `sigma_shape_measured=True`; the other 144 keep the legacy law bit-for-bit.
>
> | stock | toe (at D) | mid | dmax (at D) | interior peak | traced σ×1000 at D 1.0 (stored rms) |
> |---|---|---|---|---|---|
> | 5245 | 1.19 (0.572) | 1.00 | 0.72 (2.091) | 1.47× at D 0.73 | 6.04 (4.2) |
> | 5246 | 0.94 (0.582) | 1.00 | 0.90 (2.201) | 1.62× at D 0.66 | 8.28 (5.3) |
> | 5248 | 1.19 (0.612) | 1.00 | 0.84 (2.051) | 1.58× at D 0.74 | 7.32 (5.6) |
> | 5274 | 0.80 (0.582) | 1.00 | 0.61 (2.211) | 1.38× at D 0.68 | 9.18 (5.8) |
> | 5279 | 0.96 (0.576) | 1.00 | 0.50 (2.210) | 1.42× at D 0.65 | 13.23 (8.3) |
> | 5218 | 1.17 (0.592) | 1.00 | 0.70 (2.309) | 1.56× at D 0.74 | 9.58 (7.3) |
>
> **Render effect on those six:** grain amplitude ×2.1–2.6 at dmin, ×2.0–2.4 at D 0.7,
> **×1.00 at D 1.0 by construction**, ×0.34–0.54 at D 2.0, ×0.25–0.45 at dmax. All six turn
> OVER — σ peaks between D 0.65 and 0.74 and falls by half or more to dmax — so the
> heuristic's colour-negative branch (0.40/1.00/1.20, *rising*) is now measured wrong in sign
> on **ten** sheets rather than four.
>
> **An independent cross-check, and one conflict recorded.** `5219`'s shape had been traced in
> August from the **raster** plot on its technical sheet. The VISION3 500T brochure prints the
> same plot as **vector** art, so it was re-read here by a different extractor from a different
> document: shape **confirmed** (dmax/mid 0.55 vs 0.57, peak 1.32× at D 0.79 vs 1.24× at
> D 0.76), absolute σ **conflicts by a near-uniform 1.3×** (mid 10.60 vs 8.03). Conflict
> recorded, not averaged; nothing in the database moved on the brochure's authority.
> ⚠ The apparent third disagreement — toe/mid 0.67 vs 0.40 — is **not a document conflict**:
> below the toe the characteristic curve is FLAT, so σ(D) is genuinely multivalued at the toe
> anchor's density and the two traces landed on different points of one plateau. The extractor
> now prints `[plateau] … toe anchor is not unique` where the span exceeds 15 %.
>
> **Six extractor defects, every one of which had been producing plausible numbers:** groups of
> ≤ 4 points discarded before any filter saw them (flat dashes are 2-point segments — this
> truncated a 5245 curve at 52 % of frame width); a per-path item floor of 8 that dropped two
> of a curve's three pieces on 5279; a `>= 0.0` x-step test that refused to join pieces which
> abut (measured step −0.1 pt); an early return that accepted five whole curves **plus one
> curve's right-hand third** as "six curves"; nearest-label-wins losing letters on four sheets
> of eight, replaced by an exhaustive bijection under the two-family partition; and chaining
> across a curve crossing, now constrained by the chain's own local slope. New **method rules
> 19 and 20**: a count test must be paired with a coverage test, and an assertion that can only
> mean "your copy is incomplete" must not fail (`plot_inventory.py` now sizes the corpus and
> skips its corpus-wide counts loudly on a partial mirror).
>
> **NOT done, deliberately:** `rms_granularity` was **not** re-levelled, though all ten traced
> stocks store a family-ladder estimate 1.3–1.6× below their own plot's value. ⚠ **That 1.3–1.6×
> figure is SUPERSEDED and was wrong** — it compared the stored values against σ at *absolute*
> density 1.0, and the manufacturer's convention is *net* 1.0. The corrected range is 0.91–1.28×;
> see the 2026-08-18h entry above. That is a level
> change across whole families and it interacts with C1b, so it is queued as **C1d** to be
> decided together with C1b — grain level should move once, not twice.
>
> **State:** **235 PASS / 2 known-baseline FAIL**, `build.py --root <corpus>` OK with all
> **7 audits green**, compile clean on 18 TUs, `film_names.txt` MD5 unchanged (`ae9e4be3…`),
> schema still v8 — but the DATA moved on six stocks, so **rebuild the plugin**.
> `RESULT_2026-08-18f_C1c_sigma_harvest.md`.
>
> **Status 2026-08-18f (σ(D) WIRED — schema v8 — and the sharpness/grain corrections that
> came with it):** ⚠ **its "five vendor-traced stocks" is superseded by the 2026-08-18g entry
> above: there are now eleven.** The reasoning below stands as written.
> the field group that the 2026-08-18b entry below reports as "read by
> nothing" is now **read**, and the entry below is superseded on exactly that point.
>
> **C1 closed.** Carrier chosen by measurement, not by preference: scored against the seven
> measured σ(D) samples per VISION3 sheet, the legacy `sqrt(D − dmin + fog)` law errs by
> **245 % max / 127 % rms**, three anchors by 41 % / 18 %, and **three anchors plus one
> interior peak by 20 % / 8.6 %** — the form adopted (5 floats). A 12-sample array scores
> 3.8 % / 2.0 % and was **rejected as over-parameterised against seven measured points**.
> The 4th anchor's density was tested too, not assumed: 0.80 wins over 0.70/0.75/0.90.
> **Wired into `film_sim.py` stage 11 and mirrored in the generated C++** as
> `FilmGrainSigma()`, cross-checked against Python to 5.4e-07 by compiling the emitted
> header. **Honoured for the FIVE vendor-traced stocks only**, gated by a new
> `sigma_shape_measured` flag: `_grain_v2` fills the anchors for 137 profiles and **both
> branches are wrong in sign**, so wiring wholesale would have replaced one wrong law with
> another on 137 stocks to fix 5. **150 stocks verified bit-for-bit unchanged** (36-point
> density sweep, float32, max deviation < 2e-6). Level preserved, shape changed: amplitude
> at D = 1.0 is identical, so this is not a global grain edit.
> ⚠ **The dense-end figure is bigger than the 1.48×/2.54× quoted below.** Measured with the
> legacy law normalised at D = 1.0, it runs **3.2–3.6× too grainy at dmax** on the four
> VISION3 sheets, and **~1.6× too quiet at D ≈ 0.8**, where the real maximum sits.
> ⚠ **NOT done, and it is a decision (C1b):** the legacy law's value at D = 1.0 is
> `sqrt(1 − dmin + fog)` ≈ 0.77–0.95, so `rms_granularity` has never meant "rms at D = 1.0"
> in the renderer. Normalising that is a **level** change of up to +30 % on every stock and
> collides with the rms values self-described as "pipeline-calibrated".
> `RESULT_2026-08-18e_C1_sigma_wiring.md`.
>
> **The same day, upstream of C1 — E0 and E0b.** The LOCAL-ARCHIVE CAVEAT claimed twelve
> documents were not on file; **eleven of the twelve are**, including the file
> `dye_density.py` already opens and the TI0835 sheet the 5247 split turned on. Re-reading
> all eleven changed five values and produced three digit-for-digit agreements; three of the
> sheets then turned out to carry **unread vector plots**. Net data changes across the day:
>
> | Stock | Field | Was | Now | Source |
> |---|---|---|---|---|
> | `EASTMAN_5247_1983` | rms | 13.0 | **5.0** | TI0835 prints "less than 5" |
> | `EASTMAN_5247_1983` | resolving | (0, 0) | **(50, 100)** | printed ISO pair |
> | `EASTMAN_PLUS_X_5231` | dmin | 0.120 | **0.210** | printed base 0.19 + fog 0.02 |
> | `EASTMAN_PLUS_X_5231` | f50 | 60.0 | **41.3** | traced from the vector MTF path |
> | `EASTMAN_PLUS_X_5231` | adjacency | 0.08 | **0.034** | peak response 103.4 % |
> | `KODAK_EKTACHROME_100D_5285` | rms | 3.0 | **13.1** | traced; siblings print 14.0 and 10.0 |
> | `KODAK_EKTACHROME_100D_5285` | σ(D) | heuristic | **0.15/1.00/3.10** | first measured REVERSAL shape |
> | `FUJI_F125_8530` / `_8630` | f50 | 78.0 | **42.0** | Honjo 1989 Table 1.2, ν₅₀ = 42 c/mm |
> | `AGFA_VISTA_200` | spectral | none | **adopted** | dash-pattern legend, 0.00 nm residual |
> | `SVEMA_FOTO_32` / `_130` | tint, silver_tone | inherited | **withdrawn** | parent measurement was void |
>
> **Three more dye-density sets** (7239, 5217, 5218) — 7 → 10 profiles — recovered by fixing
> **three defects in our own extractor**, not by finding better sources. Five verify guards
> plus 13 more from E0/E0b. **Two new audit scripts** registered in `build.py`:
> `granularity_vector.py`, `mtf_vector.py`; plus `agfa_vista.py`.
> `RESULT_2026-08-18c_E0_reverify.md`, `RESULT_2026-08-18d_E0b_vector.md`.
>
> **Provenance: zero profiles now claim tier ≤ 2 with no source** (was 8). Seven citations
> registered, and `verify.py` guard 3 fired on its first real test when the owner supplied a
> document for the last two. `NotFound.md` §0.2–0.3.
>
> **State:** 231 PASS / 2 known-baseline FAIL, compile clean on 18 TUs, `film_names.txt`
> MD5 unchanged (`ae9e4be3…`). **Schema v7 → v8: `GrainSpec` gained five fields, so the
> generated struct changed size — rebuild the plugin.**
>
> **Status 2026-08-18c (C++ database split + explicit initialisation):** the generated
> `film_profiles.cpp` — 676 KB, 97 % of it ONE function — could not be compiled reliably
> by the owner's VS2015 SP3 + ICC toolchain, whose hard limits are per-function. The
> table is now emitted as **16 size-balanced data-slot TUs** (one
> `v.push_back(FilmProfile{...})` statement per profile — bounded functions, moves not
> copies, no initializer_list stack array), plus an explicit-initialisation API:
> `film::LoadFilmDataBase()` (called from the effect's GlobalSetup; `std::call_once`;
> failure does not latch) and `film::GetFilmProfile(id)` (O(1), noexcept, no lock) for
> the render path. `GetFilmDatabase()` and the print/format accessors now return
> **const&** (owner decision). `film_names.txt` is **byte-identical** — order still
> equals database order, guarded by construction, by verify.py and by a runtime
> enum==index test over all 155 entries. Slot count is fixed so the `.vcxproj` never
> changes as the database grows; outgrowing it fails generation loudly. Measured: the
> monolith compiled in 18.9 s on g++, the worst slot in 0.6 s. One-time owner actions
> (add files to the project, call LoadFilmDataBase in GlobalSetup):
> `CHANGES_2026-08-18_cpp_split.md`.
>
> **Status 2026-08-18b (a build entry point, and σ(D) is read by nothing):** two
> findings, and the second one reprioritises the queue.
>
> ⚠ **THE σ(D) HALF OF THIS ENTRY IS SUPERSEDED by the 2026-08-18f entry above: it was
> wired on 2026-08-18 (queue item C1).** What still stands is the diagnosis — the field
> really was inert, the "103 stocks" item really would have changed zero pixels, and the
> useful unit of work really was the wiring. What changed is the answer to "then set the
> defaults": the defaults were NOT set, because both heuristic branches turned out to be
> wrong in sign, so they are now explicitly inert behind a `measured` flag instead.
>
> **`sigma_shape_toe/mid/dmax` is consumed by NO renderer.** Not `film_sim.py`, not
> `Algo_11_Sim.cpp`, not the AVX2 port — all three hardcode
> `amp = sqrt(max(D − dmin, 0) + fog_grain)`. The field is populated, validated,
> emitted into the generated C++ and printed in the reports, and never read. So the
> queue's largest item by stock count ("σ(D) heuristic sign, 103 stocks") would change
> **zero pixels**, and the VISION3 adoption above is likewise inert at render time: the
> stored data is correct, the rendering benefit is still pending. The useful unit of
> work is to **wire σ(D) into stage 11 and then set the defaults** — one change, once,
> in three implementations. Awaiting a decision; nothing has been wired.
>
> Quantified while checking: because the √ law is applied to every stock, grain
> amplitude at the dense end is **1.48× too high against the B&W measurement** and
> **2.54× too high against the VISION3 colour-negative measurement**. On a negative,
> dense is a scene *highlight*, so renders carry too much grain exactly where real film
> reads clean. That is an algorithm defect across all 154 stocks, not a data one. What
> the √ law gets *right* is the toe: it gives 0.420 at D = dmin against a measured
> 0.41–0.55, for the right reason.
>
> **The B&W evidence the queue said did not exist.** Mees, *The Theory of the
> Photographic Process*, **Figure 302** (printed p866, in `PROFILES/RETRO/`) plots
> granularity against density for **four B&W negative emulsions**, data from Goetz and
> Gould. Digitised: toe/mid **0.41–0.55** at D ≈ 0.07–0.14, peak at D ≈ 0.59–1.17, and
> **0.87–1.01** at D = 1.0–1.5 — flat to gently falling, never rising. dmax (D ≈ 2.0–2.5)
> is **not measured**; the curves stop at 1.02–1.51. Three checks were load-bearing: the
> ordinate is in *relative transparency*, so G ∝ σ_D (Mees states the 10⁻ᴰ conversion
> explicitly on p863) — absolute units would have inverted the conclusion; calibration
> is fitted per panel because gridline spacing is non-uniform and one assumed scale
> missed a line by 3.4 px; and families are separated by **marker style**, which
> overturned a first pass that split them by position — the two curves in a panel are
> plotted against *different* abscissae. New tool `mees_granularity.py`. See
> `RESULT_2026-08-18b_bw_sigma_d.md`.
>
> **`build.py` — the entry point that did not exist.** `run.cmd` was one line that
> rendered an image and regenerated nothing; the regeneration sequence lived only as a
> command list in this README, and that list ran the **deprecated**
> `gen_film_names.py` last, which silently rewrites 19 of 154 names in the
> `film_names.txt` the effect panel loads. Both extraction scripts were orphans,
> referenced in prose and executed by nothing. `build.py` runs audit → verify →
> codegen → sync → docs → compile, gated: the verify gate compares the FAIL **set**
> against a baseline rather than trusting an exit code that is always 1, and the
> compile gate demands exit 0 **and** zero bytes of output. All five gates were tested
> by fault injection, and one of those tests found a bug in the gate itself — the
> `film_names.txt` check was vacuous in build mode because codegen had already repaired
> the file before it ran.
>
> **Status 2026-08-18 (SVEMA_FOTO_65 — three adopted values WITHDRAWN after a
> provenance correction):** the owner's scan batch that four parameters here were
> derived from is **not one emulsion**. The analyzer was pointed at a folder named
> `SVEMA-FN64` and analysed all **509 frames as one film**; the owner confirms only
> frames `PICT0001–PICT0067` are certainly Foto-65, and frames 68+ mix in **Foto-32**,
> which was chosen at the time precisely for finer grain and higher resolution. The
> contamination is therefore **one-directional**: the mixed batch reads finer and
> sharper than Foto-65 alone. A confirmed-subset re-run (same `analyze_film_scans.py`
> v2.1, the 67 frames, `--px-per-mm 122.7`) settles what survives.
>
> **The measurement that decides most of it takes one line to state: over all 67
> confirmed frames, `max |R−G| = max |B−G| = 0`.** Those frames are exactly
> greyscale. So every per-channel quantity in the 509-frame output — `base_tint`,
> `tone_slope_r/_b`, all twelve crossover bins, the 0.806/0.834/0.850 gamma spread —
> originates entirely in the contaminated tail and cannot be attributed to this
> emulsion. `base_tint` (0.991, 1.000, 0.991) → **identity**; `silver_tone`
> **+0.40 → 0.0**, because the sign reversal that produced +0.40 rested specifically
> on `tone_slope_r −0.0205`, which is 0.0000 on the confirmed frames. Neutral rather
> than restored to the earlier −0.10: that figure came from the same rig and the same
> class of artefact, so restoring it would swap one unsupported number for another.
> This is the *absence of an admissible measurement*, not a claim that Foto-65 is
> neutral.
>
> `sigma_shape` **0.65/1.00/1.65 → withdrawn to the schema default.** The two runs
> disagree in **sign**, not just size: mixed 509 gives 0.0191/0.0292/0.0482 (rising),
> the confirmed 67 give 0.0479/0.0425/0.0435 (flat, mildly toe-peaked). Bin edges are
> absolute offsets from `d_base` and the two `d_base` values differ by 0.024 D, so it
> is not a binning artefact — and the toe disagreement is a factor of **2.5** and is
> **unexplained**. Conflict recorded, neither adopted. The fallback 0.4/1.0/1.2 is the
> defensible one here — but for a narrower reason than first written. σ ∝ √D is the
> correct *low-density* limit (sparse non-overlapping grains are Poisson), and the
> falling Vision3 triples are measured on *chromogenic* stock, a different mechanism.
> ⚠ CORRECTED 2026-08-18b: the claim that √D is "the textbook result" for a B&W silver
> negative **over the whole range** is wrong. Mees Fig. 302 measures four B&W negative
> emulsions and they are flat-to-falling above D = 1.0. See the entry below.
>
> **Kept, with the caveat stated:** gamma 0.830, now re-based on a **printed** source
> (Gurlev 1986 p296, γ_rec 0.8 for Foto-65) per method rule 14 — the batch statistics
> are demoted to a consistency bracket, and that bracket is wide (0.677 confirmed vs
> 0.834 mixed, both resting on an *assumed* 1.90 logE scene span). clump 23 µm
> (confirmed subset gives 3.63 px → ~24.7 µm deconvolved, inside the stated
> uncertainty). Halation, at 0.166 D vs 0.199 D — 17 % lower, far inside the 4-to-7-stop
> overshoot assumption that already sets its T2 tier, and a single-channel scalar, so
> unlike the two withdrawals it never depended on the per-channel structure.
>
> **A second documentation error, corrected in four places:** the scanner was
> described throughout as a "Bayer-demosaiced DSLR" rig. EXIF reads `Make=GCMC`,
> `Model=Scanner`, `Software=UF15 16/08/20 v0.69` (3116 dpi, 8.15 µm/px). That
> invalidates the stated *reason* for rejecting `anisotropy` 0.62–0.66 as a sensor
> mosaic — and the value is **reproducible**, 0.658 mixed and 0.634 confirmed. Still
> not adopted, but now logged as an open question rather than a settled rejection.
> `verify.py` gains 4 checks that keep the withdrawals withdrawn (169 PASS / 2 FAIL;
> the 2 are the long-standing saturation-hierarchy and neighbour-pair ones).
> See `RESULT_2026-08-18_svema_clean67.md`.
>
> **Status 2026-08-17 (VISION3 granularity σ(D) — adopted at the fourth attempt):**
> the four KODAK VISION3 stocks (5203/5207/5213/5219) now carry a **traced**
> `sigma_shape_toe/mid/dmax` instead of a tier-3 estimate: 0.39/1.00/0.63,
> 0.59/1.00/0.57, 0.41/1.00/0.58, 0.67/1.00/0.55, read off the "Diffuse rms
> Granularity Curves" plot on page 3 of each sheet. Four independent sheets agree on
> the dmax anchor within ±7 %, against the mutually contradictory 2.56 / 0.70 / 0.67
> that the third attempt produced.
>
> Two findings matter more than the numbers. **First, the plots' two curve families
> are distinguished by DRAWING STYLE, not position** — dashed on 5207/5219, bold on
> 5203/5213, and 5219 prints a legend saying exactly that. Splitting on style before
> tracing makes a cross-family swap *impossible* rather than merely detectable, which
> is what three earlier passes needed and lacked. **Second, the physics premise used
> to reject the third attempt was itself wrong.** That premise — colour-negative
> granularity rises with density — is contradicted by these four sheets and, in
> print, by Kodak's own SMPTE Journal paper of July 1985 (Sehlin/Kennel, p 728,
> Figs 8–9: "overexposing either film significantly decreases granularity").
> Granularity *falls* toward Dmax on a colour negative.
>
> Consequence deliberately NOT acted on: `_grain_v2`'s heuristic still fills
> 0.4/1.0/**1.2** (rising) for 103 non-reversal stocks, so its sign is now known to
> be wrong for the colour negatives among them. It was left alone and queued —
> the approval covered four stocks, and every source is a chromogenic negative while
> that branch also fills B&W silver negatives. ⚠ The second half of that reasoning did
> not survive checking — see the 2026-08-18b entry: the B&W measurement went the same
> way as the colour one, and the field turned out to be read by nothing.
> Known limitation of the schema, recorded with the values: σ peaks at D ≈ 0.78,
> *below* the mid anchor, at 1.24–1.32× the D = 1.0 value, and three anchors cannot
> represent an interior peak. New tool `vision3_granularity.py` re-derives all of it
> from the PDFs and fails loudly if it stops reproducing. See
> `RESULT_2026-08-17f_vision3_granularity.md`.
>
> **Status 2026-08-13 (sixth entry):** spectral **integration grid 5 nm → 2 nm**
> after measuring that 5 nm was adequate only because every illuminant in the
> engine is a blackbody — against a narrow-line source a 5 nm grid is 1.5 % wrong
> and a 10 nm grid 52.7 % wrong, while 2 nm matches a 1 nm reference exactly. No
> stored curve was altered: resampling stored arrays would interpolate and destroy
> the record of what was measured. A supervised re-trace campaign is queued
> instead, priority-ordered by measured benefit. See
> `CURVE_RESOLUTION_ANALYSIS_2026-08-13.md`.
>
> **Status 2026-08-13 (fifth entry, CORRECTED 2026-08-29):** the
> spectral-sensitivity gap is **partly** closed. The digitised per-layer curves
> (**76 of 161** stocks) were read by nothing; they now drive colour-temperature
> balance in both Python and C++. The monochrome-collapse and taking-matrix
> derivations were also built, then found to reproduce a failure a 2026-08-03
> analysis had already quarantined (projecting a sensitisation onto three visible
> primaries derives blue-dominant nonsense for an IR stock). New:
> `AlgoSpectralSensitivity.hpp/.cpp` and a standalone function block in
> `film_sim.py`. Stocks without curves render bit-identically. The derived taking
> matrix is computed and reported but deliberately NOT wired in — it would
> double-count mixing already carried by `dye_matrix` and `InterimageSpec`. See
> `CHANGES_2026-08-13_spectral_path.md`.
>
> ⚠ **TWO CLAIMS IN THE ORIGINAL OF THIS ENTRY WERE WRONG. They are corrected in
> place above rather than left standing with a note under them.** It said the
> monochrome collapse *"ships OFF by default"* and that *"Python and C++ agree to
> four decimals"*. Neither described what shipped. `Algo_07_Sim.cpp` calls
> `AlgoSpectralMonoWeights()` **unconditionally**, and always has; only
> `film_sim` carried a flag, and it defaulted to False. So the collapse shipped
> ON in the plugin and OFF in the reference renderer, and for the 24 monochrome
> stocks carrying a traced pan curve the two engines rendered **different
> images** — worst case `KODAK_PLUS_X_125`, blue weight 0.110 against 0.502. The
> "agree to four decimals" was true of the two *functions* and false of the two
> *pipelines*, and that distinction is what hid it for sixteen days. Since
> 2026-08-29 `RenderSettings.spectral_mono` defaults to **True**, and
> `spectral_mono_parity.py` compiles the plugin's own translation unit and
> compares all 68 monochrome stocks on every build. Full account:
> `RESULT_2026-08-29c_spectral_weights.md`.
>
> **Status 2026-08-15 (tenth entry): 143 stocks.** Fifteen new references processed
> plus the 1495-page KODAK DATA BOOK (vol 5 FILMS located, pp 1150–1495, queued) and
> Zhurba 1984 (rotated tables read visually; zero ORWO content). New stock:
> `KODAK_TECHNICAL_PAN` (P-255) — CI 0.50–2.50 from one emulsion, the widest
> documented processing envelope in the corpus. Two "genuinely absent" gaps settled
> by their own sheets, both with ~2× RMS-granularity corrections: 8572 (7.4→4.0)
> and Vista 200 (9.4→4.3); the ETERNA Vivid sheet gave a third (6.8→3.5). That
> consistent 2× bias matters when reading remaining [C3] grain estimates. Portra
> E-190 documents the 2006 NC/VC generation — recorded, NOT merged into our 2010s
> stocks. Online Zhurba 1990 pp 44–131 unreachable (webp page images); local copy
> requested. Full record: `CHANGES_2026-08-15b_new_references.md`.
>
> **Status 2026-08-15 (ninth entry):** FUJI NEOPAN 1600's own manufacturer
> datasheet (AF3-608E, true digital PDF) extracted in full. Its curves are 300 dpi
> **rasters** despite the digital origin — the vector paths on those pages are the
> footer logo. Both were traced: the spectral curve **re-traced at 5 nm**, the
> finest sampling in the corpus, justified because a 613/630 nm dip-peak pair sits
> 17 nm apart and is under-sampled at 10 nm; it agrees with the independent 2026-08-02
> trace to 0.016 log. The characteristic curve was **refitted to 487 traced points**
> and this was a real correction: base+fog 0.170 → **0.211**, and the curve now
> reproduces Fuji's printed average gradient **Ḡ 0.77 to 0.001** where the old
> gamma 0.610 matched neither the straight-line slope nor the published Ḡ.
>
> Two lessons landed in `DIGITIZATION_QUEUE.md`: tracks sharing a plot need **mutual
> exclusion** or two will collapse onto one stroke (it produced identical Ḡ for two
> development times), and a fitted statistic must be measured **on the model, not on
> the trace** — taking the threshold from the traced points shifted Ḡ by 0.04 and the
> new regression test caught it. Full record: `CHANGES_2026-08-15_neopan1600.md`.
>
> **Status 2026-08-14 (eighth entry):** two vendor documents extracted; corpus
> unchanged at **142 stocks** because this pass corrected and verified rather than
> added. **Method finding worth acting on: check for PDF vector paths before
> tracing.** Kodak H-1-5285 draws its curves as vector polylines, so they were
> extracted EXACTLY — axes calibrated to 0.63 nm and 0.009 log, an order of
> magnitude better than any trace, and `digitize_plot.py` was not needed at all. A
> two-line check now heads `DIGITIZATION_QUEUE.md`. The Fujifilm cine manual, by
> contrast, is raster.
>
> `KODAK_EKTACHROME_100D_5285`'s spectral curves had been borrowed from the
> **5294/7294** sheet (a different product) under a declared same-family
> assumption. Replaced with 5285's own — and the borrow was **validated** on the
> way out: peaks agree to one sample, values within 0.1-0.3 log through every lobe.
> The gain is real measured skirts, 13/13/13 → 16/15/13 active samples. Fuji's
> ETERNA line gave the corpus its **first documented achromatic reciprocity
> failure** ("no filter corrections" at 1 s), so its per-channel spread is zero as
> evidence rather than as a default. `exposure_index_tungsten` was tightened to
> **unfiltered pairs only** — a colour film's second index is a filter factor
> (100→25 through an 80A is just what an 80A costs), not a film property.
>
> Two websites on Fujifilm "Film Simulations" assessed, **nothing entered**: those
> are in-camera JPEG presets, not emulsions. Only three map to a real film, and
> **Classic Chrome matches no emulsion at all** — modelling it as Kodachrome would
> be a category error. Full record: `CHANGES_2026-08-14b_fuji_kodak_websites.md`.
>
> **Status 2026-08-14 (seventh entry):** **schema v6.** Acted on the three gaps the
> Photo-Lab-Index pass identified instead of deferring them. Two new fields, the
> first the PRC axis has ever had: `exposure_index_tungsten` (7 stocks — the RATIO
> is the datum, and it separates panchromatic 1.25 from blue-sensitive 3.2/3.3 on
> documented physics) and `processing` / `ProcessingSpec` (2 of 142 — that count IS
> the measurement: almost no datasheet names the developer behind its curve).
> Two corrections: `GEVACOLOR_1952` balance 5500 → **2850 K** (Cheltsov 1958 shows
> every period Gevacolor negative is tungsten; changes the render at
> `wb_strength > 0`), and `FERRANIA_P30` era narrowed to what its 2017-only data
> covers.
>
> **`verify.py` had six dead tests.** They sat below the file's summary block and
> `sys.exit`, so they never ran. Moving them up took the pass count 108 → **114**
> with no new work — and one resurrected test failed at once, catching a **false
> claim** that `POLAROID_55_PN_NEG` was the sharpest stock. It is not: TMAX 100/400,
> ACROS and APX 25 are documented at 200 lp/mm *with* a stated 1000:1 test-object
> contrast, where the Polaroid figure states none. Claim corrected in four
> documents. A test that cannot fail is worse than no test.
>
> **Status 2026-08-14 (sixth entry):** **142 film stocks, 9 print stocks** — The
> Compact Photo-Lab-Index (Pittaro, ed., 2nd Compact Edition 1979) contributed
> eight Polaroid types whose D-max, D-min, curve slope, speed and resolving power
> are printed as numerals, plus Ilford Pan F, FP4 and HP4. `POLAROID_55_PN_NEG` —
> the peel-apart negative at a **published 150-160 lines/mm**, gamma 0.70 — is now
> sixth of 142 on `f50` and exceptional for an instant material (it is NOT the
> sharpest stock — TMAX 100/400, ACROS and APX 25 are documented at 200 lp/mm;
> an earlier draft claimed otherwise and the claim was caught when a dead
> `verify.py` block was made live). Reciprocity for
> `EKTACHROME_64`, `EKTACHROME_160T` and `KODACHROME_64` moved from a shared family
> default to documented onset and channel ordering, rebuilt from a table that does
> not survive flat text extraction. Two findings worth reading: the published Kodak
> data **prove a single Schwarzschild exponent cannot fit** three of four measured
> films (the exponent steepens 0.85 -> 0.70 per decade), and `ToneCurve` cannot stay
> strictly monotonic at the gamma 3.35 of `POLAROID_51` — verified in float64, so a
> shape-family limit, not a precision artefact. Full record:
> `CHANGES_2026-08-14_photo_lab_index.md`; survey in
> `SURVEY_2026-08-14_photo_lab_index.md`.
>
> **Status 2026-08-13 (fifth entry):** **131 film stocks, 9 print stocks** —
> Cheltsov & Bongard 1958 (Soviet monograph on colour development of three-layer
> materials) contributed ten camera stocks and four colour positives. Its value is
> disproportionately *documented* rather than authored: **six balance colour
> temperatures that existed nowhere in the corpus** (5900, 5400, 5000, 4000, 3450,
> 3300 K), one published gamma, and one internally comparable five-stock
> resolving-power ladder. Before this batch the whole database used four distinct
> balance points. It also promotes `DUPE_FINE_GRAIN`'s unity gamma from a
> first-principles argument to a manufacturer citation. Three schema gaps are now
> exercised by real data and remain unimplementable: per-layer MTF, layer order,
> and the processing axis. A generator bug was found and fixed in passing
> (`gen_active_profiles` credited the textually-last stock with other stocks'
> document numbers). Full record: `CHANGES_2026-08-13_cheltsov1958.md`.
>
> **Status 2026-08-13 (fourth entry):** **121 stocks** — owner-selected batch
> from the landing: 9 Kodak still B&W (T-MAX 100/400/P3200, TRI-X 400/320,
> PLUS-X 125, T400CN, BW400CN, Ektapan; every rms and resolving-power figure
> [C1] from its own sheet), 13 Kodak still colour negatives (Ektar, Portra
> 160/800/100T, Gold, UltraMax, Vericolor III, Ektapress PJ400, Profoto,
> 100UC/400UC — rms [C4] estimates, sheets print PGI only), and AGFA SCALA
> 200x [C1]. Remaining families: `next_week_task.md`. BREAKING: enum
> renumbered. See `CHANGES_2026-08-13_new_stocks.md`.
>
> **Status 2026-08-13 (third entry):** first extraction from the landing —
> EASTMAN_EKTACHROME_5239/7239 upgraded from their own datasheet (H-1-5239:
> rms 10.4→14.0 [C1], resolving power 40/100 added, "no datasheet" claim
> retired); reciprocity for 7 Agfa profiles [C1] from «Современные
> фотоматериалы» (2002-03); Kodak F-5 indexed and catalogued, extraction
> queued (1970s formulations ≠ our 1952 profiles). See
> `CHANGES_2026-08-13_extraction.md`.
>
> **Status 2026-08-13 (second entry):** large document landing in
> PDF/PROFILES — ~447 files not previously registered, headlined by Kodak
> Publication F-5 (88-page scan, curve families over development time for the
> professional B&W line), ~200 true-text Kodak datasheets covering the entire
> Vision line, 57 Polaroid sheets, measured spectral-density plates for
> Dufaycolor and Agfacolor Neu, and the Wilhelm dye-fade reference. Inventory,
> extractability and priorities: `PDF_LANDING_2026-08-13.md`. Nothing
> extracted yet; scans queued in `DIGITIZATION_QUEUE.md`.
>
> **Status 2026-08-13:** **98 film stocks** after owner-requested renames —
> `SVEMA_FN_64` is now `SVEMA_FOTO_65` (same film per the USSR standard; its
> two gauge-variant entries retired, gauge is the format control's job, their
> transport data preserved in `_GAUGE_TRANSPORT_PRESERVED`), `TSNL` → `CNL`,
> `EIGHT_MM_*` → `GENERIC_*`, and the stock list is now in NATURAL numeric
> order (FOTO-32 < 65 < 130 < 250). All old names remain as aliases. BREAKING:
> `eFILM_PROFILE` values renumbered. See `CHANGES_2026-08-13_rename.md`.
> Historical sections below keep the names in use at the time they were
> written.
>
> **Status 2026-08-11:** the database has grown to **100 film stocks, 5 print
> stocks, 14 gauges** — seven stocks added from owner-supplied documents
> (Kodak Data Book 5th ed. 1952: Verichrome, Panatomic-X / Tri-X / Ortho-X
> sheet films; Agfa 2003 brochure: Optima 200/400, Portrait 160), and
> ГОСТ 24876-81 added as a corroborating source on SVEMA_FN_64 (= Foto-65).
> See `CHANGES_2026-08-11_stocks100.md`. Same day, second pass: GOST norms
> extracted into the Soviet profiles — two MTF floors raised where the old
> estimates violated the films' own state standard (SVEMA_FN_64 f50 34→35,
> SVEMA_FOTO_250 26→30), TsNL-65 mask/fog/latitude corroborated by
> ГОСТ 25120-82, per-layer gamma conflict Gurlev-vs-GOST recorded not
> adopted. See `CHANGES_2026-08-11_gost_extraction.md`. Third pass: the 1942
> Eastman MP book (scan, OCR-indexed) — first Kodak motion-picture document
> in the archive; EASTMAN_SUPER_XX_1938 documented [C1] (Type 1232 sheet,
> tier 3→2), PLUS_X_5231 predecessor context [C3]. See
> `CHANGES_2026-08-11_kodak1942.md`. Fourth pass: SMPTE July-1985
> Sehlin/Kennel paper registered — measured granularity-vs-exposure and MTF
> plots for 5247/5294, digitisation queued. See
> `CHANGES_2026-08-11_smpte1985.md`.
>
> **Status 2026-08-02:** the database had grown since this write-up to
> **93 film stocks, 5 print stocks, 14 gauges** (Soviet reference-book pass:
> six Svema/Tasma stocks added from Gurlev 1986 / Iofis 1980, ORWO_UT18
> renamed ORWO_CHROM_UT18). Counts quoted in the body and in the historical
> sections below are what was true when each section was written. See
> `SOVIET_EXTRACTION_2026-08-02.md` and `CHANGES_2026-08-02_soviet.md`.
>
> **Schema v3 (same day):** digitised spectral sensitivity curves — new
> `SpectralSensitivity` struct on every profile (inert when empty), generated
> C++ carries a generation timestamp + schema version, spectral tables are
> `std::vector<double>`. Test suite is now 70 checks. After the batch
> digitization passes, **35 of 93 stocks carry per-emulsion spectral curves**
> (per-stock table: `FilmCurves.md`; all Kodak MP stocks covered, plus the
> 2026-08-04 Agfa/Fuji/Gevaert additions); H&D curves machine-traced and refitted
> for VISION3 250D and ACROS 100 (`digitize_plot.py`). Details in the final
> sections of `Readme!.txt` and `CHANGES_2026-08-02_soviet.md`.

```bash
python film_sim.py photo.jpg --list                 # what stocks exist
python film_sim.py photo.jpg -p 5219                # one stock, by catalogue number
python film_sim.py photo.jpg -p "Kodak Vision3 500T (5219)"   # or by full name
python film_sim.py photo.jpg -p all -o renders      # everything
python film_sim.py photo.jpg -p velvia -f ff35      # 35 mm still, not Super 35
python film_sim.py photo.jpg -p ortho                # red-blind 1930s B&W
python film_sim.py photo.jpg -p "super xx" -g 3      # 1938 stock, 3 dupe generations
python film_sim.py photo.jpg -p dufaycolor           # additive mosaic (render big!)
python film_sim.py photo.jpg -p 5219 --flare 0.10    # force period lens flare
python film_sim.py photo.jpg -p technicolor --emit-cpp
```

### Regenerating the database

**Use `build.py`. Do not run the generators by hand in the order this README
used to list.** That list ended with `python gen_film_names.py`, *after*
`cpp_codegen.py` — and that script is deprecated (see the file table below).
Running it last silently replaces `film_names.txt`, the list the effect control
panel loads, with a different set of display names: 19 of 154 differ
(`KODAK T-MAX 100` against `KODAK TMAX 100`, `SVEMA FOTO-65` against
`SVEMA FOTO 65`, and so on). The in-service file is `cpp_codegen`'s.

```bash
python build.py                 # audit + regenerate everything, in order, gated
python build.py --check         # READ-ONLY: audit and report drift, write nothing
python build.py --list          # the stages, and which sources each audit needs
python build.py --only verify   # one stage
run.cmd  |  run.cmd check  |  run.cmd build  |  run.cmd render     (Windows)
```

Stages run in a load-bearing order: **audit** (re-derive adopted numbers from
the source documents) → **verify** → **codegen** → **sync** (assert the
project-root copy of the generated C++ is identical) → **docs** → **compile**.

Two gates are deliberately stricter than they look:

* **`verify.py`'s exit code is not the gate.** It exits 1 whenever any check
  fails, and two fail *by design*. `build.py` compares the FAIL **set** against
  a baseline, so a new failure fails the build and a baseline entry that starts
  passing is also reported, rather than quietly absorbed.
* **The compile gate requires exit 0 AND zero bytes of output.** A warning with
  a zero exit code once reported "clean" while a string literal was broken;
  `build.py` fails on 216 bytes of `-Wunused-variable` even though `g++`
  returns 0.

An audit whose source document is absent **SKIPs with the reason** and does not
fail the build, so the sequence still works on a machine without the 449-PDF
corpus.

Individual generators, for reference — `build.py` runs these for you:

```bash
python verify.py                                    # 304-check suite (303 PASS / 1 FAIL by design)
python cpp_codegen.py -o .                          # the 23 C++ artefacts
python vision3_granularity.py --overlay out         # re-derive the VISION3 sigma(D)  [raster]
python mees_granularity.py --root ../.. --overlay out   # re-derive the B&W sigma(D)
python granularity_vector.py --root ../.. --assert  # 5285 sigma(D)                   [vector]
python mtf_vector.py --root ../.. --assert          # PLUS-X 5231 f50 + overshoot
python dye_density.py --root ../.. --assert         # the 11 dye-density sets
python agfa_vista.py --root ../../PDF/PROFILES --assert   # Vista 200 spectral + legend
python plot_inventory.py --root ../../PDF/PROFILES --assert
```

## Why the original script could not get there

The original added spectrally-unshaped noise to gamma-encoded sRGB pixels. Grain is
maybe 20% of what makes an eye say "film", and everything else was missing. The four
structural problems, worst first:

1. **No characteristic curve.** Film's identity is its density-vs-log-exposure curve —
   toe, straight line, shoulder. Without it there is no latitude, no highlight rolloff,
   no shadow compression, and output clips digitally hard at 255. Real negative rolls
   off over 4+ stops above diffuse white.
2. **Wrong domain.** Halation, scatter and grain statistics are all linear-light
   phenomena. Applied to gamma-encoded values they come out the wrong shape.
3. **No resolution awareness.** `grain_size` and `GaussianBlur(radius=15)` were in
   pixels, so the same profile looked like a different film at 1080p and 4K.
4. **No MTF.** The image stayed digitally razor-sharp with grain pasted on top — the
   single loudest tell.

## Pipeline

Order is not cosmetic; several steps give visibly wrong results if moved.

| # | Step | Domain |
|---|------|--------|
| 1 | Decode sRGB → linear light | — |
| 2 | Relative exposure (18% grey = 1.0), exposure offset, taking filters | linear |
| 3 | Stock colour balance, then **veiling flare** from the taking lens | linear |
| 4 | Large-scale coating unevenness | linear |
| 5 | Halation: multi-radius, all channels, energy conserving | linear exposure |
| 6 | Emulsion MTF — light scatter inside the gelatin | linear exposure |
| 7 | Collapse to one record: spectral sensitivity, or the **réseau grid** | linear |
| 8 | Characteristic curve → density | density |
| 9 | DIR coupler inter-image effects | density |
| 10 | Scan: MTF + per-channel misregistration (pre-sampling filter) | density |
| 11 | Grain: per-channel RMS, spectrally shaped, amplitude from **`grain_sigma()`** — the MEASURED σ(D) shape where a vendor plot was traced (5 stocks), otherwise the legacy √density law, bit-for-bit as before (150 stocks) | density |
| 12 | Dye impurity / scanner crosstalk matrix | density |
| 13 | **Duplication generations**, then print | density |
| 14 | Print grain, transmittance → display linear, réseau reconstruction | — |
| 15 | Encode sRGB, dither, quantise to 16 or 8 bit | — |

Reversal stocks skip step 13 entirely: the film *is* the positive, so there is no print.

Details that matter and are easy to get wrong:

- **Red is the softest channel.** In colour negative the blue-sensitive layer is on top,
  green in the middle, red at the bottom. Light reaching the red layer has been
  scattered by two layers of gelatin. That per-channel softness is a strong signature.
- **Grain goes in before the scan MTF, not last.** Adding grain at the end is what makes
  it read as digital noise sitting on a sharp picture.
- **Halation conserves energy.** Light scattered away from a point is removed from it.
  Adding `blur(highlights)` alone injects a flat brightness lift — at CineStill's gain
  it lifted an 18% grey card by 16%.
- **Adjacency is band-pass.** A plain unsharp term settles at `1 + a` for all high
  frequencies, i.e. permanent global sharpening. Real adjacency peaks at the inhibitor
  diffusion scale and returns to unity at both ends.
- **Grain never fully vanishes.** `fog_grain` keeps it alive in deep shadow. Perfectly
  clean blacks are a digital tell.
- **The print anchor is solved, not guessed.** `logE_print = offset − D_neg`, with the
  offset solved per channel so 18% scene grey lands on 18% display — exactly what a lab
  does with printer lights. The naive `offset = D_mid` puts mid grey around 2% display,
  three stops too dark. The solve includes the taking matrix, both dye matrices, the
  coupler flat-field term and the base tint. Originally the dye matrices had row sums
  as far off as 1.27, which threw the mid tone out by more than a stop; they are now
  unit-row-sum by construction (see the `_dye` fix below), so the matrix contribution
  to the anchor is exactly neutral and only the taking matrix and couplers move it.

## The 1930s-40s block, and what it needed

Adding period *emulsions* alone would have underdelivered, because the emulsion is
maybe a third of what makes archival footage read as archival. Two pipeline
additions carry the rest, and both help the modern stocks too:

**Veiling flare (`--flare`).** A *lens* effect, not an emulsion one. Uncoated
pre-1940 glass scattered 6-14% of incoming light into a broad haze across the
frame; anti-reflection coating cut that below 1%. It lifts the black floor and
compresses contrast globally, and nothing in the emulsion model substitutes for
it — without it a 1930s stock still renders with modern blacks. Each stock carries
an era-appropriate `default_flare`; modern stocks are 0. Measured effect on a dark
patch in a bright frame: black level 0.007 → 0.180, overall range 0.69 → 0.52.

**Duplication generations (`-g N`).** Nobody projected the camera negative. A
release print is three or four generations away: negative → interpositive → dupe
negative → print. Each intermediate adds grain and MTF loss. Duplicating stock
runs at gamma 1.0 by design, so contrast does not compound over the chain — only
grain and softness do. Measured over 0/1/2/3 generations, mid grey holds at 0.1801
throughout while the grain-to-detail ratio climbs 0.080 → 0.114. Absolute grain σ
actually falls slightly; what worsens is grain *relative* to picture detail, which
is exactly why dupes look grainier than the negatives they came from.

**Additive colour (Dufaycolor).** This one needed a genuinely new code path, not
parameters. A microscopic grid of colour filters (a réseau) is ruled onto the base
with one panchromatic emulsion behind it; colour resolution is capped by the grid
pitch rather than the emulsion, there is exactly one grain field and no
inter-layer effects of any kind, and the grid stays faintly visible as texture.

Two things about it were worth getting right. The filters must *overlap* — real
ruled gelatin has broad passbands, so a cell under the red filter still records a
lot of green. Model them as pure and the process comes out more saturated than
Kodachrome, which is precisely backwards; the overlap is where the pastel comes
from. And the grid is physical at 20 lines/mm, so it needs at least 3 pixels per
cell to exist at all: below that the mosaic disables itself with a warning rather
than emit aliasing noise. Render Dufaycolor at 2500 px wide or more.

Measured saturation, mid-tone patches, showing the dye hierarchy falls out of the
matrices rather than a saturation control:

| Velvia | Kodachrome | Technicolor | VISION3 500T | EXR | Dufaycolor | Agfacolor Neu | ORWOcolor |
|---|---|---|---|---|---|---|---|
| 0.718 | 0.412 | 0.227 | 0.195 | 0.156 | 0.152 | 0.118 | 0.076 |

Orthochromatic response, which is the loudest single period cue, is pure
parameters — the machinery was already there. Red renders 16× darker than blue on
the ortho stock, against 1.2× on a panchromatic stock of the same era. That is why
silent-era makeup was so extreme: ordinary red lipstick photographed black.

### One model fix this forced

Working on the period dye matrices exposed a flaw in all of them. A hand-written
crosstalk matrix tends to have row sums away from 1 — 1.27 for a "muddy" stock,
0.92 for a "clean" one — which means it shifts neutral *density* as well as
colour. Two unrelated effects on one knob: the anchor solve then has to undo the
density part, and a stock's black level ends up depending on its saturation
setting. All 18 matrices are now generated by a `_dye(k)` helper with row sums
pinned to exactly 1.0, so they change colour and nothing else, leaving `dmin` and
`gamma` solely responsible for level. Verified as a test.

## Everything spatial is physical

Grain clump size, halation radii, MTF cutoffs and channel registration error are in
micrometres or cycles/mm, converted to pixels at render time from
`px_per_mm = image_width_px / format_width_mm`. Change `-f/--format` and the physics
follows the gauge.

Consequence worth understanding: **rendered granularity legitimately depends on scan
resolution.** The scanner MTF is the pre-sampling filter, so a 2K render shows less
grain than a 6K render of the same negative, converging upward. That is not an artefact
— it is why 4K rescans of old negatives look grainier than the 2K masters people
remember. The *stock parameters* are resolution independent; the *rendered result*
correctly is not.

`film_sim.py` warns below 60 px/mm. At 1280 px across Super 35 (51 px/mm) a 4.6 µm
clump is a tenth of a pixel, so fine-grained stocks cannot show their structure. For
judging grain, render at 3000 px wide or more.

## Stocks

All 13 you asked for, plus 13 chosen to stress different parts of the model.

**Colour negative, motion picture**
`KODAK_VISION3_50D_5203`, `KODAK_VISION3_250D_5207`, `KODAK_VISION3_200T_5213`,
`KODAK_VISION3_500T_5219`, `EASTMAN_EXR_500T_5296`, `FUJICOLOR_SUPER_F500_8572`,
`ORWOCOLOR_NC21`, plus `EASTMAN_5247_1974` (the 1970s look) and
`FUJI_ETERNA_VIVID_500T_8547`.

**Colour negative, still**
`KODAK_PORTRA_400`, and `CINESTILL_800T` — VISION3 500T with the remjet stripped, which
makes it the most extreme halation in production and a good stress test of that model.

**Colour reversal** (no print stage)
`KODACHROME_64`, `KODAK_EKTACHROME_100D_5285`, `FUJI_VELVIA_50`.

**Black and white negative**
`ILFORD_HP5_PLUS_400`, `FOMAPAN_400_ACTION`, `SVEMA_FOTO_65` (ex FN-64), plus
`EASTMAN_DOUBLE_X_5222` (Manhattan, Raging Bull) and `ILFORD_DELTA_3200` — tabular
crystals, so enormous grain that is nonetheless *even* rather than clumpy, which
demonstrates that grain size and grain character are independent parameters.

**Black and white reversal**
`KODAK_TRI_X_REVERSAL_200`.

**1930s-1940s** (see the section above)
`EASTMAN_ORTHO_1930` (red-blind), `EASTMAN_SUPER_XX_1938` (film noir),
`SOVIET_PANCHROM_1939`, `AGFACOLOR_NEU_1936` (first integral tripack, muddy dyes
on a reversal stock — a combination nothing else here covers), `DUFAYCOLOR_1937`
(additive réseau mosaic).

**Special**
`TECHNICOLOR_THREE_STRIP` — beam-splitter camera, three separate B&W records, imbibition
dye transfer print. Three things make the look and none is grain: broad overlapping
taking filters (the famous reds), very pure transfer dyes, and 26 µm registration error
between the strips, which is why its edges fringe.

Any stock resolves by name, alias or catalogue number: `5219`, `vision3-500t`,
`Kodak Vision3 500T (5219)` all work.

Print stocks: `SCAN_DI` (digital intermediate, system gamma ≈ 1.0),
`KODAK_2383_RELEASE` (theatrical, contrasty), `TECHNICOLOR_IB` (dye transfer),
`DUPE_FINE_GRAIN` (gamma 1.0, used automatically by `-g`).

### One catalogue-number caveat

You asked for "Kodachrome Tri-X 200 (5266)". Tri-X Reversal ships as **7266** in 16 mm;
I could not establish a 5266 Tri-X reversal product, so the profile is built as the 7266
emulsion and answers to both numbers. Correct it if you have a datasheet that says
otherwise.

## Calibration honesty

**Numeric values tagged `# EST` in `film_profiles.py` are engineering estimates, not
datasheet transcriptions.** They produce a convincing and internally consistent result —
grain, sharpness and latitude all scale correctly across each family — but they are not
authoritative, and the older and more obscure the stock the rougher the estimate.
Kodachrome and Technicolor parameters are reconstructions from published descriptions,
not measurements: treat them as artistic targets.

**The 1930s-40s block is weaker still, and differently so.** For the modern stocks
the numbers are estimates anchored to datasheets I could reason about. For the
period stocks there are no datasheets I can consult at all: the figures are inferred
from how surviving footage looks, from the emulsion technology of the era, and from
internal consistency with the rest of the database. Super-XX is the firmest of the
five because it stayed in production for decades. Agfacolor Neu and the Soviet stock
are the softest. Dufaycolor's réseau pitch is the only figure in that block I would
defend within a factor of two.

On the Soviet stock specifically: it is modelled as a late-1930s Shostka-factory
panchromatic negative. Note that the "Svema" brand name postdates this era, so the
profile is deliberately not called that. Its defining trait here is inconsistency,
which is historically well attested — domestic stock of the period was variable
enough that major productions often preferred imported Agfa or Kodak when available.

To make this a true emulation rather than a good-looking approximation, replace them with
digitised datasheet data:

- Kodak publishes D-logE curves, MTF curves, spectral sensitivity and RMS granularity
  for every current VISION3 stock in its Technical Data sheets.
- Fujifilm published equivalents for ETERNA / SUPER F while the stocks shipped.
- ORWO and Svema data survives mostly in scanned GDR/USSR technical handbooks.

Digitise with WebPlotDigitizer or similar, fit the six `ToneCurve` parameters to the real
curves, and the "can a colourist tell?" answer changes from *probably* to *no*. The
structure is built to accept real data; only the numbers are provisional.

## Files

| File | Purpose |
|------|---------|
| `build.py` | **The entry point.** Ordered, gated regeneration + audit: audit → verify → codegen → sync → docs → compile. `--check` is read-only. Registers the audit scripts, so a new extraction script becomes part of the build instead of an orphan. Stdlib only |
| `run.cmd` | Windows wrapper: `run.cmd` / `check` / `build` / `render` |
| `film_profiles.py` | Physical parameters, **159 film stocks, 9 print stocks**, 14 gauges. **Schema v10** (v8 added `GrainSpec.sigma_shape_peak`/`_peak_at`/`_toe_at`/`_dmax_at`/`_measured`; **v9 redefined `rms_granularity` as the rms at NET density 1.0** — same layout, different meaning; **v10 added `MTFSpec.mtf_rolloff_q` / `mtf_measured`, so the MTF rolloff shape is stored and read**). Holds the ONE definition of both the grain-σ law (`grain_sigma`) and the MTF law (`mtf_response`), mirrored in the generated C++ as `FilmGrainSigma` / `FilmMtfResponse` |
| `film_sim.py` | The pipeline, 16-bit PNG writer, CLI. Holds the ONE definition of the reciprocity law (`reciprocity_log_shift`, `_cc_filter_shift`) and of the two DIR-coupler stages (`apply_interimage`, `apply_dir_couplers`), each mirrored in the plugin's own C++ and covered by a parity audit |
| `cpp_codegen.py` | Emits `film_profiles.hpp` / `.cpp`, then `film_names.txt` and `film_enum.hpp` for a C++ port |
| `film_profiles.hpp/.cpp` | Generated C++ tables, with the reference formulae in the header |
| `film_names.txt` | Generated. One display name per line, quoted, spaces not underscores, ASCII, no comments — feeds the effect-panel listbox. Line *N* is vector element *N−1* |
| `film_enum.hpp` | Generated. `enum class eFILM_PROFILE : int32_t`, from 0, ending `eTOTAL_FILMS_PROFILES`. **Values shift when a stock is inserted** — see the compatibility warning in `CHANGES_2026-08-04_stocks93.md` |
| `test_film_enum.cpp` | C++14 cross-check that enum values, vector order and `film_names.txt` lines all agree |
| `gen_film_names.py` | **Deprecated, and actively harmful if run.** Superseded by `cpp_codegen.write_film_names()`, which derives order from the emitted `.cpp` instead of from `FILM_PROFILES`. Running it after `cpp_codegen.py` rewrites 19 of 154 display names in `film_names.txt`. `build.py` never invokes it and reports if the file on disk was not `cpp_codegen`'s. Kept only for reference |
| `vision3_granularity.py` | Audit: re-derives the four VISION3 σ(D) triples from the Kodak TI sheets, exits non-zero if it stops reproducing. Run by `build.py`'s audit stage |
| `mees_granularity.py` | Audit: re-derives the B&W silver-negative σ(D) shape from Mees Fig. 302 (printed p866), four negative emulsions. Run by `build.py`'s audit stage. See `RESULT_2026-08-18b_bw_sigma_d.md` |
| `dye_density.py` | Audit: re-derives all **11 adopted spectral dye-density sets** from the sheets' vector paths; 5285 and 2383 are the validation pair. ⚠ Its docstring records the three defects that had put 7239/5217/5218 on a "failed" list when the sources were fine |
| `granularity_vector.py` | Audit: σ(D) from a **vector** granularity plot — **9 sheets**: EKTACHROME 100D 5285 (the only measured σ(D) for a colour **reversal** stock), the six colour negatives adopted 2026-08-18 (5245, 5246, 5248, 5274, 5279, 5218), **VISION2 50D 5201 added 2026-08-20**, and the VISION3 500T brochure as an independent cross-check of 5219's raster trace. Also reports each sheet's **net-1.0 rms triple**, which reproduces all six values adopted under C1d to within 0.7 % — and which answered queue C1e as a by-product. Composes the characteristic and granularity curves at shared abscissa, so the log-exposure axis cancels; `--overlay` draws every traced point back onto the panel and is the adoption gate. See `RESULT_2026-08-18f_C1c_sigma_harvest.md` |
| `gevaert_curves.py` | Audit: characteristic curves from the **paper scans** of the Agfa-Gevaert journal articles -- `GEVACOLOR_NEG_682`'s Fig. 10 at one sample per pixel column off the native 340 ppi 1-bit scan, validated against the gamma 0.57 the figure prints. Fits both axes as LINES because the page is skewed 1.40 deg, and re-verifies its pinned tick anchors against the pixels on every run |
| `interimage_parity.py` | Audit: **the Python vs C++ DIR-coupler stages — the only audit that probes the PLUGIN'S OWN translation units** rather than generated code. Compiles against `Algo_08_Sim.cpp` / `Algo_09_Sim.cpp` and compares `apply_interimage()` / `apply_dir_couplers()` against `AlgoStage08b_Interimage()` / `AlgoStage09_DirCoupler()` over 5 stocks × 2 fields × 2 pixel scales. Reads `sizeof(AlgoType)` from the compiled probe and picks its tolerance from it, so the deliberately switchable double/float typedef stays switchable. ⚠ Asserts only where the blur is resolved; the sub-pixel scale is probed and reported, because below ~1 px the two blur implementations are not the same operator |
| `AlgoReciprocity.hpp` | **Plugin C++, NEW 2026-08-23 (C8).** The reciprocity law: the per-channel shift of log exposure a stated shutter time implies. Header only, computed once per frame, consumed by stage 8 as three constants added to the logarithm. Mirrors `film_sim.reciprocity_log_shift()`; `cpp_parity.py` compares the two |
| `cpp_parity.py` | Audit: **the Python reference laws vs the C++ ones — grain, MTF and (since 2026-08-23) RECIPROCITY**, 8586 + 5724 probes over every stock × 3 channels × 10 densities, walking the real `GetFilmDatabase()`. Worst disagreement 2.7e-07 against a 2e-5 tolerance. Carries a coverage assertion so the check cannot pass by both sides silently falling back to the legacy law. ⚠ Written because C1b changed one law in two languages and the previous cross-check was a manual one-off |
| `mtf_vector.py` | Audit: `f50`, the adjacency overshoot and the **rolloff exponent** from a vector log-log MTF plot — **12 sheets, 26 curves** (2026-08-23): PLUS-X 5231 mono, seven Kodak colour negatives with a complete per-record triple (5201, 5274, 5217, 5218, 5245, 5248, 5279), two with green and blue only and a **refused** red (5205, 5293), and the **Agfa Vista 200** panel — the only non-Kodak MTF in the corpus, and a single visual-weighted curve rather than three records. Record identity comes from **ink** on the brochures and from **printed R/G/B letters** on the technical sheets, solved as a minimum-cost bijection rather than nearest-neighbour; the single path the 1990s sheets use for all three records is split; a fragment is refused rather than measured. **`--overlay` is the gate** — it found all four extractor defects fixed under C2b |
| `kodak_sensitometry.py` | Audit: `ToneCurve` parameters least-squares fitted to a Kodak sheet's **vector** characteristic curves, sharing `digitize_plot.fit_tonecurve` so the model has one definition. ⚠ Takes the SHAPE from the dense curves inside the *granularity* panel and asks the panel titled "Sensitometric" only for the abscissa origin — see method rule 22 |
| `agfa_vista.py` | Audit: `AGFA_VISTA_200`'s spectral sensitivity, and the **dash-pattern legend** it depends on — solid = green, dashed = blue, dash-dot = red, cross-checked against Agfa's own printed labels |
| `plot_inventory.py` | Audit: the corpus plot inventory (191 vector dye-density pages, 199 MTF, 101 granularity, 294 characteristic), with three known-answer pages as the classifier's ground truth. ⚠ Those counts are **corpus-wide (450 PDFs)**, so they are reported but NOT asserted when fewer PDFs are present — a partial mirror used to fail all four (method rule 20) |
| `verify.py` | **304-check suite (303 PASS / 1 FAIL by design)**: curves, calibration, anchors, isotropy, PNG, flare, generations, réseau, spectral data, provenance guards, the grain LEVEL contract (amplitude = stored rms at net density 1.0, all 465 stock-channels, plus an empirical aperture-integrated end-to-end check), edge cases. Render-heavy. ⚠ **Slicing is currently broken for any slice that does not start at 1** — a later section references a name bound in an earlier one (`NameError: _fpm`). Full runs are unaffected |
| `make_test_chart.py` | Synthetic chart (ramp, patches, MTF bars, specular discs) |
| `make_period_chart.py` | Larger chart for the period stocks and the réseau |
| `contact_sheet.png` | All stocks on the small chart |
| `period_sheet.png` | The period stocks, plus a 3-generation dupe comparison |
| `dufay_crop.png` | Dufaycolor réseau at 1:1, so the grid is visible |
| `doc/` | Audit trail: datasheet verification (Found/NotFound), Soviet book extraction, dated changelogs, measurement adoption reports |

## Verification

⚠ **THIS SECTION WAS BADLY STALE AND IS REWRITTEN AS OF 2026-08-23.** It said "70 checks,
all passing (67 original + 3 schema-v3 spectral checks)", which described the suite as it
stood before schema v4. `python verify.py` runs a **304-check suite: 303 PASS and 1 FAIL by
design** — the saturation-hierarchy ordering, which the owner instructed to leave alone.
`build.py` compares the FAIL *set* against a baseline, so a NEW failure fails the build while
the known one does not.

The list below says what the suite asserts in kind; the file itself is the specification:

- every characteristic curve is monotonic
- grain reproduces the datasheet RMS granularity to within 1.3%
- granularity rises monotonically with scan resolution and never exceeds the figure
- 500T renders 2.54× grainier than 50D — the datasheet ratio is 2.54
- 18 % grey anchors to 18 % display for **all 159 stocks and all 9 print stocks**
- red is softest and blue sharpest through a 25 c/mm target
- halation is red-dominant and CineStill halates far more than a remjet stock
- reversal stocks clip a wide ramp far sooner than negative stocks
- B&W output is exactly neutral (R = G = B)
- ortho renders red 16× darker than blue, against 1.2× on a panchromatic stock
- flare lifts the black floor and compresses contrast
- each dupe generation worsens grain-to-detail while mid grey holds to 4 decimals
- the réseau leaves a periodic signature 1334× the noise floor at exactly the grid
  frequency, reconstructs neutral grey as neutral, and refuses to run under-sampled
- every dye matrix has unit row sums, and the saturation hierarchy is correctly ordered
- deterministic for a fixed seed; survives pure black and 16 stops of overexposure
- the emitted C++ reproduces the Python characteristic curve to 6 decimals

The generated C++ tables are compiled by `build.py`'s last stage with `g++ -std=c++14
-Wall -Wextra` over all 18 translation units, gated on exit 0 **and** zero bytes of compiler
output. Four laws are cross-checked against the Python reference on every build: grain and
MTF against the generated header and **reciprocity against the plugin's own
`AlgoReciprocity.hpp`** (`cpp_parity.py`), plus the two DIR-coupler stages against the
plugin's own `Algo_08_Sim.cpp` / `Algo_09_Sim.cpp` (`interimage_parity.py`).

## Known limits

- **Reciprocity is a per-channel GLOBAL shift, not a per-pixel effect** (C8, 2026-08-23). Wired and
  live, inert until `exposure_time_s` / `exposureTimeS` is set. 21 stocks carry a measured table
  and 105 a Schwarzschild exponent. ⚠ **Real reciprocity failure is intensity dependent** -- the
  darkest parts of a frame fail first, which is why a long exposure loses shadow separation as well
  as speed -- and **no source in the corpus carries an intensity axis**: all 21 measured tables are
  functions of time alone. The intensity term is therefore absent rather than estimated. ⚠ Outside a
  table's measured range the correction is **held flat, not extrapolated**, because Kodak's own
  tables walk the effective exponent from ~0.85 to ~0.70 across successive decades. ⚠ Only ONE
  stock (EKTACHROME 64, from 1e-4 s) measures the high-intensity branch at all, so a flash duration
  lands on the held-flat first entry everywhere else.
- **The per-layer grain ladder is measured on 11 stocks and estimated on 54** (C1e, 2026-08-23).
  Where measured, blue is **1.81-2.79x** green and red **0.75-1.05x**; `_grain_v2`'s estimate says
  1.30x and 1.10x, i.e. wrong in magnitude for blue and in sign for red. It is deliberately not
  rescaled: all nine measured sheets are Kodak cine negatives, and the blue ratio tracks stock
  SPEED (500T 2.70x, 250D 2.12x, 50D 1.81x) rather than any constant a class estimate could carry.

- **Display-referred input.** A JPEG or PNG has already had its highlights clipped by the
  camera, so the film's shoulder has nothing to roll off. Feed scene-referred data (EXR,
  or a raw file developed to linear) for a real improvement — this is the biggest
  remaining quality lever after datasheet calibration.
- **MTF is measured on EIGHT stocks and estimated on the rest, and the estimating RULE was
  measured wrong** (C2b/C24, 2026-08-23). Eight stocks carry `mtf_measured`: PLUS-X 5231
  (mono) and seven Kodak colour negatives with a complete traced triple. Two more (5205,
  5293) have a measured green and blue and a refused red, so they stay on the legacy law.
  ⚠ **The old estimating rule scaled all three records from one number by a fixed layer
  ratio (`f50_r ≈ 0.78 × f50_b`), and seven measurements say that FORM is wrong:** red f50
  reads 32.1 33.9 35.4 37.2 37.4 37.6 41.1 — mean **36.4, spread ±13 %** — while green
  spreads 52 % and blue 70 %. Red does not scale with the stock's sharpness at all, which
  reads as the bottom record being limited by scatter through the two layers above it. The
  estimates were **1.12–1.72× too sharp in red and 0.70–0.83× too soft in blue**.
  Five modern Kodak cine stocks (5203, 5207, 5213, 5219, 5246) had their **red re-anchored
  to the family's 36.0 cycles/mm**; green and blue were left alone because the measured blues
  run 0.96–1.43× their estimates with no consistent factor. ⚠ **Every other maker and every
  pre-1990 stock keeps the old rule** — the corpus holds no per-record MTF outside Kodak
  cine, so there is nothing to re-derive from. **63 colour stocks still carry an estimated
  triple.**
- **The rolloff exponent `q` is per-stock and cannot be derived.** The power law
  `1/(1+(f/f50)^q)` beats the legacy Gaussian `exp(-ln2·(f/f50)²)` on **all 26 traced
  curves** (1.1×–5.8×, rms 0.0095–0.132), and both are exactly 0.5 at f50 so the choice
  carries no level change. ⚠ **C13's layer-depth hypothesis is half refuted:** the ORDERING
  `q_R ≤ q_G ≤ q_B` holds on 8 of 8 sheets, but the magnitudes are not per-layer constants
  (red 1.89–2.77, blue 2.38–3.42, sd 0.32–0.37) and q correlates only weakly with f50
  (Pearson 0.39 over 23 curves). The two-stock clustering that suggested a derivation was a
  two-sample illusion. ⚠ One stock's exponent is deliberately **not** fitted: 5279's sheet
  prints a +42 %/+55 % adjacency overshoot and the carrier is 1.0 at zero frequency by
  construction, so it cannot represent a curve that starts at 1.42.
  `mtf_tail_a` / `mtf_tail_f_exp` remain the form C2 rejected on evidence and are inert.
- ⚠ **Every colour stock whose f50 triple is an ESTIMATE is too sharp in the red record** (queue
  **C24**, confirmed twice as of 2026-08-20c). The estimating rule scales one measured number by a fixed layer-order ratio. Both stocks
  measured per-record contradict it: 5274 measures red 35.4 against a stored 56.0 (**1.58× too
  sharp**) while its green and blue confirm to 7 %, and 5201's measured red:blue is 0.58 where the
  rule assumes ~0.78. This is a defect in the rule, not in one profile. Measured and pinned in
  `mtf_vector.EXPECTED`. 5274 was **adopted** under C13 on 2026-08-20c; the remaining **92**
  estimated triples are queue **C24**, deliberately NOT rescaled from two measurements of one Kodak
  family.
- **Grain σ(D) is measured for twelve stocks only** (all Kodak). The other 147 use the legacy √density law, normalised at **net** density
  1.0 since schema v9. The heuristic that fills the shape for 137 profiles is **wrong in sign
  in both branches** and is deliberately inert; fixing its signs needs its own evidence pass,
  per-class, not an assumption that one measurement generalises. 39 raster granularity pages
  remain unread.
  ⚠ **"The vector corpus is exhausted" was WRONG, and the correction is instructive.** C1c claimed
  it at 8 sheets on 2026-08-18; a **ninth** was found on 2026-08-20 in the same folder — H-1-5201,
  whose 89 × 90 pt panels sat below the extractor's frame floor, whose density labels are jittered by
  0.17 D, and which draws its red record twice so the panel presented 8 curves where the physics says
  6. Four guards refused it correctly and **none of them said "there is a panel here I cannot
  read"**. "Exhausted" is a claim about a tool, not a corpus, until the tool refuses loudly. Re-run
  the sweep with the fixed extractor before reaching for raster tracing — see `NotFound.md` §0.4.
- **Grain level is now pinned to the printed convention** (net density 1.0), and the six
  vector-traced negatives carry rms values and per-layer triples read off their own curves.
  ⚠ The four VISION3 stocks are **not** re-levelled, and as of 2026-08-20 that is settled rather than
  pending: the 5219 brochure's own net-1.0 green reads **6.37 against a stored 6.6** where the raster
  sheet implied 8.98, confirming both that the raster extractor reads ~1.3× high **and** that the
  stored green levels are about right. ⚠ Their **per-layer** ladder is a different matter — the same
  brochure puts 5219's blue at 15.45 against a stored 8.58 (1.8× low), which is the 1.3×-ladder error
  again and which needs an owner decision (C1e). Any future rms work
  must state its reference density before comparing anything (method rule 21).
- **The toe anchor of any σ(D) trace is ill-posed.** Below the toe the characteristic curve
  is flat, so density holds at dmin while σ keeps changing and σ(D) is multivalued there.
  Measured consequence: two traces of 5219 from two documents disagree 1.7× on the toe while
  agreeing to 0.02 on dmax. The extractor flags it; the affected entries say so.
- ⚠ **The DIR-coupler chemistry is modelled in both halves, and one half has no provenance.**
  `InterimageSpec` (vertical, stage 8b) is patent-derived, solved per stock against
  US5273870A's published IIE percentages using each stock's own gamma, and `verify.py`
  re-derives those percentages to < 1 pp. `CouplerSpec` (lateral, stage 9) is **87 hand-typed
  literals** with no registry, no citation and no tier — and its `adjacency_um` is contradicted
  by the traced MTF overshoot on all four stocks measured. Queue **C19**, merged with C2c.
- ⚠ **The adjacency term is implementation-dependent at ordinary render sizes.** The two
  renderers' blurs agree to 6e-5 above ~1.2 px sigma and diverge to 1.5e-1 below 1 px, and the
  stored 9–13 µm edge scale puts them at 0.36–0.60 px at 40 px/mm. Queue **C16**/**C17**.
- ⚠ **`density_weighting` is unbounded and its 0.65 is tier 3.** On Velvia the per-donor weight
  reaches 1.82 at D 3.2, a worst-case −0.58 logE ≈ 1.9 stops. 36 reversal stocks ride on it.
  Queue **C18**.
- **`ReciprocitySpec` is stored and read by nothing.** Emitted to C++, consumed by no
  renderer. Two datasheet-derived entries were added on 2026-08-18 and neither changes a
  rendered frame.
- **No temporal behaviour or physical damage.** Single frames only: no gate weave, no
  processing flicker, no frame-to-frame grain animation, no dust or scratches. For the
  period stocks this is the largest remaining gap — flare and dupe generations cover
  the optical and photochemical side of the archival look, but not the mechanical one.
- **Memory.** `numpy.fft` computes in double precision, so a 6K frame needs a few hundred
  MB. Use `--max-dim` to work smaller.
- Requires Python 3.12 as specified; the modules also import on 3.10 (plain `Enum`
  rather than `StrEnum`), which is how the test suite was run.

---

## Expansion set: 26 → 55 stocks

29 stocks added. Database now holds **55 film stocks, 4 print stocks, 12 gauges**
(Super 8 added at 5.79 mm).

| Group | Stocks |
|---|---|
| Agfa B&W | APX 25, APX 100, APX 400 |
| Agfa colour | Optima 100, Vista 200 |
| Eastman reversal | Ektachrome EF 5239 (35 mm), 7239 (16 mm) |
| Ektachrome stills | 64 daylight, 160T tungsten |
| Fuji | F-125 8530, F-125 8630, Neopan Acros 100, Neopan 1600, Provia 400X, Sensia 100 |
| Polaroid | SX-70, 664, 667 |
| USSR | Svema Foto-250, Tasma FN-65 |
| 8 mm gauges | generic B&W reversal, generic colour reversal |
| Indian cinema 1940–60 | Gevacolor 1952, Gevaert Panchro 1950, Eastman Plus-X 5231 |
| Britain | Ilford HP3, Ilford HPS |
| France | Lumière Lumichrome |
| Italy / Latin America | Ferrania P30 |

### Confidence tiers

The original block carries one blanket `# EST`. The new block is graded, because
the sources vary enormously and you should be able to see which is which:

- **[T1] Datasheet-grounded.** Published speed, granularity and resolution exist;
  numbers fitted to them. Good to roughly 10 %.
- **[T2] Partially grounded.** Speed and reputation documented; grain and MTF
  interpolated from siblings in the same family and era.
- **[T3] Reconstruction.** No datasheet available. Built from era, speed class,
  process type and written descriptions. Plausible and internally consistent —
  **not** measurements.

`[T3]` set: Svema Foto-250, Tasma FN-65, both 8 mm entries, Gevacolor 1952,
Gevaert Panchro 1950, Lumière Lumichrome. Lumichrome is the weakest of the lot
and says so in its own description.

### Gauge pairs

`5239`/`7239` and `8530`/`8630` are the same emulsion on different base, so their
numbers are **deliberately identical**. The visible difference is magnification,
which the renderer derives from `--format`, not from the profile. Render the 16 mm
member with `--format 16mm` or `--format super16` or the distinction is lost.
Same for the two 8 mm entries: use `--format 8mm` or `--format super8`.

### Two honest notes

**South America.** No South American country manufactured raw film at scale in
1940–1980; its studios shot on imports, Ferrania prominently among them. So
`FERRANIA_P30` is labelled as the Italian stock it is rather than dressed up as
something it isn't.

**India.** Indian studios also shot imports across the whole 1940–60 window.
Domestic manufacture began 1960 with Hindustan Photo Films at Ootacamund
("Indu" stock), just outside the window. Gevacolor is documented on *Aan* (1952)
and *Mother India* (1957).

### Monotonicity bound corrected

`ToneCurve`'s docstring previously claimed monotonicity is guaranteed for
`shoulder_k <= 2 * toe_k`. That is the analytic bound on the second derivative,
but measured on the actual transfer, ratios above about **1.4** produce a
reversal of order 1e-6 near the shoulder asymptote. Harmless visually, but
`verify.py` checks for it. Four of the new low-Dmax reversal stocks tripped it and
were retuned; the docstring now states the empirical bound.

### Known limitation: low-Dmax stocks don't yet look low-Dmax

Instant film's defining property is a low Dmax — SX-70 reaches 1.87 where
Kodachrome reaches 3.20, so its blacks are open and slightly milky however you
expose it. **That is currently not visible in the render.**
`_normalised_transmittance()` rescales each curve's own `dmin..dmax` to `1..0`,
so the stock's own Dmax is the divisor, every stock is stretched to fill the
output range, and the difference is normalised away. The profiles and the C++
tables carry the correct Dmax; the Python renderer flattens it. On the test chart
SX-70 and Kodachrome both bottom out at display 0.000.

For negatives this is correct — the negative is an intermediate and the print
stock sets the final range. For reversal it is wrong, because the film *is* the
viewed image.

**Proposed fix, not applied:** normalise reversal against a fixed viewing-black
reference (Dmax 3.40) instead of each stock's own Dmax. Predicted sRGB floors:

| Stock | Floor | Stock | Floor |
|---|---|---|---|
| POLAROID_SX70 | 0.159 | KODACHROME_64 | 0.005 |
| POLAROID_667 | 0.151 | EKTACHROME_64 | 0.006 |
| POLAROID_664 | 0.129 | FUJI_VELVIA_50 | 0.006 |
| EIGHT_MM_BW | 0.103 | FUJI_PROVIA_400X | 0.008 |
| AGFACOLOR_NEU_1936 | 0.052 | EIGHT_MM_COLOR | 0.009 |
| DUFAYCOLOR_1937 | 0.035 | EASTMAN_EKTACHROME_* | 0.012 |

Polaroids and 8 mm B&W get their real floor; Kodachrome, Velvia and modern E-6
are essentially untouched. Agfacolor Neu and Dufaycolor also read more correctly
for their era as a side effect.

This changes rendered output for all 17 reversal stocks, so it awaits your
decision.


## 2026-07-31 second pass

71 stocks (83 after the 2026-08-01 pass). Soviet/ORWO profiles re-fitted from the owner's real scan batches
(see DATASHEET_VERIFICATION_REPORT.md addendum for per-field verdicts) and 13
stocks added: ORWO UT18 (measured, aged slides), Konica Infrared 750 /
Impresa 50 / VX 100 / Centuria Super 400 / Centuria Super 1600 / Chrome
Centuria 100 / Chrome R100, Rollei R3 / Infrared 400 / Retro 400, Kentmere
Pan 100 / Pan 400 (datasheets). Reciprocity for the new stocks is fitted from
the printed correction tables, not defaulted. `_grain_v2` no longer
overwrites author-set sigma shapes (bug fix).


## 2026-08-01 pass

83 stocks. Added 12 Kodak motion-picture negatives spanning 1959-2010
(5250, 5254, 5294, EXR 5245/5248/5293, Vision 5246/5274/5279, Vision2
5205/5217/5218) so the classic-film and franchise looks map to real
library entries. Movie-stock claims fact-checked in
MOVIE_STOCK_VERIFICATION.md -- four systematic errors found in the source
list (code reuse, anachronisms, 5295/5296 conflation, the Rogue One
film-out myth, which belongs to Dune 2021).


## Schema v4 (2026-08-03): coating, gate and lens defects

Four effects behind what viewers read as "old film edges". The premise was
corrected during review: coating unevenness **cannot** produce a
frame-corner-locked defect, because film is coated as a wide web and slit
afterwards -- the coating never knew where the frames would fall. Corner
darkening is the LENS.

| Effect | Field | Driver | Eras |
|---|---|---|---|
| Lens vignette | `FilmProfile.default_vignette` (stops) | lens | all -- modern glass still loses 0.3-0.5 stop |
| Coating field | `CoatingSpec.coating_sigma` + 2 correlation lengths | plant QC, not date | Soviet/GDR/budget any era; modern majors zero |
| Gate buckling | `CoatingSpec.buckle_mtf_loss` | base stiffness x gate size | pre-1955 worst; 8 mm into the 1980s |
| Edge fog | `CoatingSpec.edge_fog_density/_mm` | gauge only | 8/16 mm; never 35 mm |

New render controls: `vignette` (stops, <0 = era default), `coating_scale`
(scales all three coating defects, 0 disables), `frame_index` (slides the
coating field by one frame pitch; any frame renders independently and out of
order). `frame_pitch_mm(fmt)` and `PERFS_PER_FRAME` are exported.

Measured cost at HD, worst-case stock with all four active: **+19%**, of which
corner defocus is +11% (a 3-channel separable pass, the only neighbourhood
operation), vignette +5.6%, coating field and edge fog +2.5% together. Against
the pre-v4 code (which ran a full-resolution FFT pair per frame for the old
isotropic mottle) it is **+11%**. Modern stocks pay +11% for gate buckling
alone and nothing for the rest.

Full rationale, verification and known limits:
`doc/CHANGES_2026-08-03_v4_coating.md`.


## Schema v5 (2026-08-03): interimage effects

`InterimageSpec` -- cross-layer development inhibition, the vertical half of
the DIR-coupler chemistry whose lateral half is `CouplerSpec`. Applied to log
exposure before the curve, referenced to the mid-grey density so a neutral is
untouched (verified: 0.00000 delta) while saturated colour separates further --
saturation rising without gamma rising, which no per-channel curve can produce.

Active on 51/93 stocks. Excluded with reasons: monochrome, the additive-mosaic
stocks, and Technicolor three-strip (three separate films cannot exchange
inhibitor).

**Tier 3 throughout, unavoidably.** All 395 datasheets were searched: none
publishes interimage data, because camera negative is characterised with a
single white-light exposure series. To measure, shoot a neutral step wedge plus
the same wedge through W25/W58/W47B filters on one roll with an empty-gate
reference.

`derived_spectral_response()` was built, measured, and **quarantined** -- it
returns blue-dominant weights for a 750 nm infrared film because display
primaries stop at 630 nm, and near-identity matrices for colour stocks. Not
wired into the renderer; a verify check asserts it stays out. Details:
`doc/CHANGES_2026-08-03_v5_interimage.md`.

Additive-only, proved: 0 existing field values changed across 93 stocks.


## Interimage upgraded to tier 2 (2026-08-03, later same day)

No datasheet publishes interimage effects -- but PATENTS do, because claiming
improved interimage requires demonstrating it. US5273870A defines the metric
("percentage steepening of color gradation during color separation exposure ...
in relation to ... white light", citing James 4th ed. pp. 574/614) and
tabulates it with a genuine DIR-free control: invention 25/45/42 % (B/G/R)
against control 10/15/15 %. Corroborated by US4830954A and US4725529A.

Two things the survey settled:

* **The asymmetry is per RECEIVER, not per distance.** Blue receives weakly,
  green and red strongly. US4725529A Table 1 proves it is emulsion chemistry
  rather than geometry -- inhibitor in the developer, three separate
  single-layer coatings, no stack at all, and red receivers still take
  0.43-0.72 dlogE against blue 0.24-0.48.
* **The conversion must be solved numerically.** The closed form matched the
  DIR-free control to 0.9 pp but overshot strong DIR by 23 pp. `_IIE_TIERS`
  now stores the patent percentages and the coefficients are fitted against
  the model, reproducing published figures to **0.05 pp** on stocks of
  differing contrast.

Reversal stocks additionally carry `density_weighting=0.65`, because
US4729943A places their interimage in high dye-density areas via first-developer
iodide rather than colour-developer DIR.


## ISO 5-3 densitometry complete (2026-08-03)

`iso5_3_density.py` now carries all nine spectral-product tables: visual,
Type 1, Type 2 (ISO 5-3:1995 Table 2) and **Status M / Status A blue-green-red**
(Tables 4 and 3, recovered from ANSI/NAPM IT2.18-1996 after the ISO copy turned
out to be a preview that stops mid-sentence at the point of naming Table 4).

Peaks: visual 570; Type 1 400; Type 2 430; Status M 450/540/640; Status A
440/530/620 nm. Status M red peaks longer than Status A red, which is the
documented reason both exist -- M matches colour negative responses, A matches
transparency.

One subtlety worth knowing: Tables 3 and 4 do NOT print "< 1,000" out of range,
they print a SLOPE and an arrow, so the response continues linearly in log10.
Truncating to zero would narrow every channel skirt and bias derived densities;
`weights()` applies the printed slopes instead.

This clears the blocker on deriving `dye_matrix` from measured dye curves.
Details: `doc/ISO_5_3_STATUS.md`.


## Archived source files (2026-08-03)

Mees 1942 (356 MB) and the American Cinematographer Manual (117 MB) moved to
external storage. Mees findings preserved with page citations in
`doc/MEES_1942_EXTRACTION.md`; the ASC manual is an image-only scan with no
text layer, so nothing was extractable. **`The Permanence and Care of Color
Photographs` (34 MB) stays** — canonical source for the still-unpopulated
`AgingSpec` dye-fade fields.


## AlgoControl.hpp (2026-08-03)

Real controls struct, replacing the `int dummy` placeholder. 21 live fields
mirroring `film_sim.RenderSettings` one-for-one (**verified 21/21**), plus
`bool filmDamageEnabled` gating a nested `FilmDamage` sub-struct of 17
specified-but-inert fields. Sentinel convention: `flare` and `vignette` are
−1.0 for "use the stock default", 0.0 for "genuinely none". Damage rates are
per second, not per frame. See `doc/ALGOCONTROL_NOTES.md`.
