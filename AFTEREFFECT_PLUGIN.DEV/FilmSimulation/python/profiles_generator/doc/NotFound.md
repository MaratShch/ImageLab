# NotFound.md — verified gap analysis and data-acquisition guide

**State as of 2026-09-01e.** This file is a **research checklist**, not a history: it exists so that
someone can write an effective search query, or a precise request to a manufacturer, archive,
museum, laboratory or standards body, without first re-investigating what the film *is*. Everything
here is either **still missing** or a **hazard that will mislead the next search**.

⚠ **Rewritten from the front on 2026-08-31 rather than appended to.** The header and §0 had become
a record of corrections to a revision three weeks dead — 400 lines of "this used to say X" in a file
whose whole purpose is to say what is missing *now*. All of it is preserved verbatim in
**`NotFound_history.md`**; the findings that still bind are carried forward as rules in §0 below.

---


## 2026-09-02e — what the three new sheets and the 1942 book do NOT print

**`FUJI_PROVIA_100F`, `FUJICOLOR_SUPERIA_XTRA_400`, `FUJICOLOR_PRO_400H` — spectral
sensitivity.** All three datasheets print a spectral-sensitivity panel and the curves are
perfectly legible. ⚠ **The ORDINATE is the blocker, not the curves**: on AF3-151E and
AF3-176E it is a bracketed arrow annotated "1.0", not a numbered tick ladder, so the decade
scale rests on an annotation. A peak-normalised `log_s` built on a misread bracket would be
wrong by a factor and look entirely plausible. `spectral.has_data` is False on all three.
*What would settle it*: any Fuji publication printing these same curves against a numbered
sensitivity axis.

**`FUJI_PROVIA_100F`, `FUJICOLOR_SUPERIA_XTRA_400`, `FUJICOLOR_PRO_400H` — per-record MTF.**
Fuji prints ONE unlabelled black curve, so each sheet yields one f50. Assigned to green;
red and blue take the family flanking ratios and remain [T2].

**`FUJI_PROVIA_100F`, `FUJICOLOR_SUPERIA_XTRA_400`, `FUJICOLOR_PRO_400H` — the adjacency
pair.** Each panel prints an overshoot (+15.5 %, +21.1 %, +11.0 %) whose **peak frequency it
does not resolve** — the drawn curve begins at its maximum. Two unknowns, one measurement, so
`adjacency` is 0.0 by refusal, exactly as on 8532 and 8572. *What would settle it*: an MTF
panel drawn from below 1 c/mm.

**`FUJICOLOR_SUPERIA_XTRA_400`, `FUJICOLOR_PRO_400H` — clump size and everything downstream
of it.** The sheets print rms granularity and resolving power and no other image-structure
data. The clump triples are [T3] family estimates.

**`KODAK_GOLD_200` — MTF.** ⚠ **E-7022 has no MTF panel**, verified across all three copies
in the corpus. Its curve page prints characteristic, spectral-sensitivity and
spectral-dye-density panels and nothing else. The f50 estimate is not a gap in the reading;
it is a gap in the document.

**`KODAK_VISION3_500T_5219`, `KODAK_VISION3_250D_5207` — anything traceable.** H-1-5219 and
H-1-5207 are held and cited, but ⚠ **pages 3 and 4 of both carry three embedded images each**,
so the granularity, MTF and spectral panels are rasters. The absolute granularity level on
both stocks stays the project's pooled anchor.

**`EASTMAN_PLUS_X_5231`'s 1942 predecessor, Type 1231 — a tone curve.** The 1942 book prints
its H&D family, time-gamma and time-fog curves as a **raster on a letterpress scan**, so a
curve for it would be a trace of a halftone of a printed plot, and its speed is on the
pre-ASA Kodak scale that PH2.5-1960 does not convert by a constant. Its printed numbers are
recorded on the 5231 profile as context and reach no field.

**The unnamed ASA 100 negative of Takano 1968.** ⚠ **The film is not named anywhere in the
paper**, so its measured mottle sizes are class evidence and cannot be written to a stock.
Figs. 3–8 and 10 additionally plot spectrum level in an **unlabelled instrument unit with no
printed constant** and are refused for adoption on that ground alone.

## What is still missing, in one screen

| # | Still missing | Why it matters | Best next move |
|---|---|---|---|
| 1 | **8 stocks with no documentation of any kind** (§1) ⚠ **9 → 8 on 2026-09-01d: TECHNICOLOR_THREE_STRIP now carries a measurement** -- Flueckiger et al. 2018 §2.8.2 Fig. 16, the analytical densities of the three imbibition transfer dyes off a 1949 SAMSON AND DELILAH print, SHIMADZU UV-1800 plus the Ohta PCA separation. Shape only: that figure's ordinate has no scale, so the curves are peak-normalised with the axis assumed to be zero | every parameter is a class estimate | §1 lists a specific query per stock |
| 1b | ⚠ **GRAIN CANNOT BE FILLED FROM ANY AGFA DOCUMENT, AND THIS IS NOW CHECKED** | `grain.clump_um*`, `clump_gain`, `fog_grain` and the σ(D) shape print red on all twelve Agfa stocks. All four Agfa documents were searched on 2026-09-01: **no granularity plot, no aperture series, no Wiener spectrum, no granularity-vs-density**. Agfa publish ONE rms number at ONE density. ⚠ **`clump_um` is estimated or assumed on EVERY stock — no manufacturer in this corpus publishes it** ⚠ **UPDATED 2026-09-02 (queue TK1/TK4): AN APERTURE SERIES NOW EXISTS, ON SOMEBODY ELSE'S FILM.** Takano 1969 Fig. 8 gives Selwyn G at thirteen apertures 3-90 µm for a colour negative and Neopan-SS, and Fig. 13 adds two optical autocorrelations. With Ooue Fig. 24 there are now **five direct measurements of grain correlation length: 0.87, 1.77, 2.46, 3.22, 4.64 µm, median 2.46** against a stored median of **13.0**. ⚠ That does not fill an Agfa cell — none of the five is an Agfa stock, and the whole point of this row is that a class estimate from another maker's emulsion is method rule 18. What it does is turn the Agfa blank from an unquantified guess into a guess with a known bias: **the estimating scale itself is about 5x every measurement on file.** Queue C45 owns that | it needs rms at **two or more apertures** (Selwyn's law holds only for uncorrelated grain; the departure from √A *is* the correlation length), a Wiener spectrum, or direct microscopy. ⚠ It is also development-dependent, D_eq ∝ γ^0.42 (BBC T-101), so a figure measured at another gamma must be converted first. `clump_gain` and `fog_grain` are renderer parameters with no photographic counterpart and no source will ever publish them. ⚠ **A THIRD MEASUREMENT ARRIVED 2026-09-01c AND IT WIDENED THE PROBLEM RATHER THAN CLOSING IT.** JPS 1965 abstract 10p-A-2 (`jp_jps_1965_269.py`) measures the granularity Wiener spectrum of five emulsions indexed by CRYSTAL SIZE and gives half-power frequencies 80–108 c/mm, i.e. `clump_um` **2.73–3.69 µm** through this project's own `grain_shape` law. Set beside the two BBC-derived MEASURED values (ILFORD_PAN_F 0.655, ILFORD_HPS 1.431 → 449 and 206 c/mm) and the 168 estimates (median 13.0 µm → 23 c/mm), the three sources span a factor of twenty and **just one stock lies inside the band this new measurement brackets**. The corpus median implies grain markedly blobbier than any measurement on file supports. Nothing was adopted: a one-page abstract in RELATIVE units cannot move 168 stocks. ⚠ Its two fitted laws are also **not invertible** — a six-fold change in crystal size moves the bandwidth 35 %, so 1 % in u½ is 8 % in d. ⚠ **AND A SECOND MAKER JOINED IT ON 2026-09-02d (queue N1).** FUJIFILM **AF3-411E(N)**, the NEOPAN SS (135) data sheet, prints speed, colour sensitivity, a full development matrix, a spectral sensitivity curve, a characteristic-curve family and time-Ḡ curves — and **no image-structure section at all**: no rms granularity, no resolving power, no MTF, no reciprocity, no base thickness, all four pages searched. `FUJI_NEOPAN_SS` therefore ships with a flagged class grain block for exactly the reason the twelve Agfa stocks do. ⚠ And it could not borrow the four granularity measurements this corpus holds under its own name: Ooue 1959 and Takano 1969 measured the **1959–1969** coating, this sheet is dated **1999**. |
| 2 | **σ(D) for anything outside the Kodak vector corpus** | **13 of 172** measured -- 11 colour negatives, 1 colour reversal (5285), 1 B&W reversal (TRI-X Reversal 200), and every one is a Kodak vendor sheet. ⚠ **A TWELFTH WAS TRACED AND WITHDRAWN ON 2026-09-02c (queue E5), AND THE REASON IS A NEW ONE FOR THIS ROW.** Sehlin & Kennel, *SMPTE Journal* 94(7), July 1985, Fig. 8 puts density and RMS granularity on ONE shared log-exposure abscissa for `EASTMAN_5294_1983`, and the trace is good: toe 1.571 @ D 0.44 / mid 1.000 / dmax 0.703 @ D 2.08, peak 1.664 @ D 0.53, inside the eleven sheets' 1.20-1.62 and 0.50-0.90. ⚠ **But its anchor densities are the FIGURE'S plotted density and `sigma_anchors` reads PER-LAYER ANALYTICAL density** — on every measured stock `sigma_shape_toe_at` IS the green curve's dmin, and 5294's traced toe at 0.44 sits below its green dmin of 0.68 and far below its blue 1.09. Stored once, `cpp_parity.py` rejected it at 5.7e-01 against a 2e-05 tolerance. ⚠ **SHAPE AND SPACE ARE AS SEPARATE AS SHAPE AND LEVEL** — so this row now asks for two things, a named stock AND an anchor density it can place. ⚠ **CORRECTED 2026-09-01: THE OLD TEXT HERE WAS WRONG.** It said the 55 monochrome negatives carry a heuristic "measured to be wrong in direction — the one measured B&W shape RISES toward dmax and the default falls". Both rise. Measured against `KODAK_TRI_X_REVERSAL_200`, the only B&W stock with a traced shape, the legacy √(D−dmin+fog) law gives 0.218 / 1.000 / **1.446** at toe / net 1.0 / D-max where the measurement gives 0.218 / 1.000 / **1.994**. The default is not backwards — it is **≈38 % too shallow at D-max**, which is a different defect and sends a reader somewhere useful. ⚠ Note also that the stored triple on an unmeasured stock is **INERT**: `sigma_anchors` returns `None` unless `sigma_shape_measured` is set, and the renderer uses the √ law, so filling that cell without the flag changes no pixel ⚠ **AND A FOURTH INDEPENDENT MEASUREMENT ARRIVED 2026-09-02 (queue TK2), THE FIRST FROM OUTSIDE KODAK.** Takano 1969 Fig. 9 traces σ_D against mean integral colour density for a Japanese colour negative at a 16 x 16 µm aperture, one curve per layer: the magenta record gives **toe 0.301 @ D 0.30 / mid 1.000 / dmax 0.301 @ D 2.50, peak 1.002x at D 1.04**. It **turns over**, confirming the 2026-08-17 correction on a non-Kodak emulsion measured a generation earlier — and ⚠ **it disagrees on WHERE the maximum sits**: all eleven measured colour negatives here peak at D 0.65-0.80 at 1.20-1.62x, this one at D 1.04 at 1.00x with no interior peak at all. So `sigma_shape_peak` is confirmed as a Kodak-FAMILY measurement, which `sigma_shape_measured` already refuses to generalise. ⚠ **NOT WRITTEN TO ANY PROFILE** — Takano names his sample only 「カラーネガフィルム」. This row stays open and its ask is unchanged: the missing thing is a named stock, not another anonymous one | queue **F2b** — one granularity-vs-density plot for a named B&W negative at a stated aperture. Best lead **WAS** Higgins & Stultz, cited by BBC Engineering Monograph No. 54. ⚠ **THAT LEAD WAS RUN DOWN 2026-09-01 AND IT DOES NOT FIT.** BBC T-101/2 prints the number the lead promised — σ ∝ D^0.4 — and `bbc_t101_2.py` adopted it for the DENSITY step of a granularity conversion, where it is corroborated at 0.355 by T-101 Table 3's own grain-diameter pairs. But as a σ(D) SHAPE it is the worst of the candidates: the one measured B&W reversal shape wants an exponent near **0.92–1.28**, and 0.4 is further from it than the legacy √ law. The lead is spent; ⚠ **AND ON 2026-09-01e FOUR SUCH PLOTS ARRIVED.** Ooue 1959 (`JAPAN/23_7.pdf`, Fig. 7) gives granularity against density for **Neopan S / D-76, Neopan SS / Microfine, Neopan SSS / Pandol and an X-ray film / Rendol**, at a STATED 10 µm aperture chosen to match the eye at 14x. Fitted exponents 0.412 / 0.672 / 0.364 / 0.606 (R² 0.965-0.998). ⚠ **NOT ADOPTED, ON AN AMBIGUITY THE PAPER ITSELF CREATES**: §3.2 defines the ordinate v as the MEAN-SQUARE output while the English abstract says the meter reads the ROOT-MEAN-SQUARE. On the rms reading the three named negatives straddle the legacy √ law and the BBC 0.40; on the mean-square reading they fall below every other source here, which is itself the argument for rms. Ooue's companion papers (本誌 22, 38 and 91, 1959) would settle it and are not in this corpus. ⚠ What the four DO settle is the shape of the question: the √ law is about right for B&W NEGATIVES, and the measured 0.92-1.28 reversal shape is reversal behaviour, not a defect in the negative law -- which is what BBC T-101/2 said in words |
| 3 | **MTF: 199 vector pages inventoried, 23 stocks measured** | ⚠ **q is NOT a layer-depth constant** (reversed 2026-08-23). The power law beats the Gaussian on every traced curve and the ordering q_R ≤ q_G ≤ q_B holds 8/8, but the magnitudes are far too spread to derive — red 1.89–2.77, blue 2.38–3.42. q stays per-stock measured | trace more colour sheets; each buys its own stock's q and nothing more |
| 3b | **63 colour stocks still carry an estimated f50 triple** | ⚠ **The estimating rule is wrong in FORM, not scale** (queue C13). Measured red f50 is effectively **constant at ~36 c/mm** while green spreads 52 % and blue 70 %, so no fixed `f50_r ≈ 0.78 × f50_b` ratio can be right | five modern Kodak cine stocks have their red re-anchored to 36.0; all other makers and all pre-1990 stocks untouched |
| 4 | **A 300+ ppi re-scan of Kino-Technik 1968 Nr. 10, pp. 260/262/264** (queue G5) | the 150 ppi scan cannot separate the three Gevachrome layer curves — they sit 1–2 px apart | **the owner holds the source** |
| 5 | **An unambiguous statement of what Agfa's "lines/mm" axis means** (queue G6) | decides whether the 682 plot's axis is cycles or half-cycles — **a factor of 2 on 4 stocks** | ⚠ **NARROWED, NOT CLOSED, 2026-09-01 — and the 2026-08-31 entry here was wrong.** It said "the four Agfa candidates are the same publication, two of them byte-identical". The byte-identical PAIR is real (`AGFA stocks.pdf` = `FPD1e.pdf`, md5 `bf9f0c1a…`) but **`agfa_films.pdf` is md5 `edb3dd17…`, a separate 1st edition of 09/1998** against the others' 4th edition of 08/2004. It prints an MTF **and** a resolving power for the same film on the same page, for twelve films. Measured f50/RP runs **0.19–0.52, median 0.30** against Tani's predicted 0.5; reading the axis as half-cycles halves every ratio, i.e. moves it *further* from the relation. **Evidence favours lines/mm = cycles/mm. A 2.7× spread is not proof.** |
| 5b | ⚠ **NOT MISSING — RESOLVED 2026-09-01, kept here as a warning** | `agfa_films.pdf` was read as RASTER in the 2026-08-02 batch and recorded as a duplicate in row 5. It has **zero embedded images on all twelve pages**. Four profiles (ULTRA 50, RSX II 50/100/200) and twelve films' worth of tabulated data were sitting behind a one-line misfiling | **before believing an absence, run `get_images()` and `md5sum`.** Two lines of code against two years of a stock being "undocumented" |
| 5c | ⚠ **NOT MISSING — READ 2026-09-01b, kept here because the pattern repeated** | `AGFA stocks.pdf` (F-PF-E4) had been in the corpus since 2026-08-29 and **only its curves were ever read, and only pages 6-7 of them**. Pages 8 and 9 carry six more plotted columns — RSX II 50/100/200, APX 100/400, SCALA — and **not one printed table on the twelve-page sheet had been harvested**, although those tables carry the resolving power at two contrasts, the layer thickness, the base thickness per format, the DX and negative codes, the development-time matrices at 18/20/22/24 °C and the exposure index per developer for all ten films. It took the owner supplying the GERMAN twin (`agfa-aERRKF-Datenblatt_F_PF_D4.pdf`, F-PF-D4 07/2003) for anybody to open the English one properly. ⚠ Row 5b's lesson was "before believing an absence, check the file". This one is narrower and worse: **a document already harvested is not a document already read**. A reader that stops at p7 leaves no trace saying so | when a reader is registered, record WHICH PAGES it covers, and audit the remainder as an explicit gap rather than an implicit one |
| 5d | **A statement of what Agfa's per-film sharpness panel actually measures** | ⚠ **AND ONE THING IT DOES NOT: it is not per-film on the AGFAPAN pages.** A duplicate-artwork scan over all ten columns × four panels of BOTH Agfa editions (2026-09-01b) found exactly two duplications. One was already known — RSX II 50 and RSX II 100 share the spectral-density panel, documented in `_MEASURED_DYE_MATRIX`. The other is new: **APX 100 and APX 400 share the sharpness panel**, the same 73-point path translated 175.21 pt, every y offset identical to 0.0000 pt, on the 1998 sheet and the 2003 one alike (f50 57.6 for both films in 1998, 59.6 for both in 2003). Two different films cannot share a measured MTF, so **both AGFAPAN `mtf.f50_*` triples stay estimated ON PURPOSE** and now say so in a ParamSource | an Agfa MTF that names one film. Until then the AGFAPAN f50 cells are red for a checked reason, not an unexamined one |
| ~~5e~~ | ✅ **CLOSED 2026-09-01b — schema v24.** «Developing adjustment (%)» / «Entwicklungskorrektur (%)» is now `ReciprocityTable.development_correction_pct`, populated 0 / −10 / −25 / −35 on APX 100 and APX 400 and **0 / 0 / 0 / 0 on APX 25**, which is a stated null and the only one in the set — Agfa are saying the slowest AGFAPAN emulsion's contrast does not climb with a long exposure while its two faster siblings' does. Three editions across six years agree cell for cell, and `agfa_2003_sheet.py` asserts all three plus the stored values. Additive and inert | — |
| 6 | **Absolute base+fog** (queue D1, one empty-gate frame) | makes density absolute for every stock on that scanner | the owner, one minute — ⚠ **but it must be shot in the same session as the scans it calibrates**; the UF15 re-exposes per frame, so no later frame can rescue an old batch |
| 6b | **A Kodak datasheet for EKTAR 125 (1989–1994)** (queue C14) | ⚠ **the stock is NO LONGER UNREPRESENTED and this row's old premise was false.** `KODAK_EKTAR_125` was created 2026-08-31 on **one measured number** — a blue D-min **upper bound ≤ 0.849** from US 5,334,491, slot 8 of nine bleaches. The *magazine review* still prints no sensitometry; the patent, in the same corpus, prints one density. Everything else on the profile is tier 3 | ⚠ a *datasheet* re-proved absent 2026-08-31 across all 475 files; the patent was found 2026-09-01 in the same corpus. `KODAK_EKTAR_100` (E-4046) is a DIFFERENT, LATER film |
| 7 | **Callier coefficient** — ⚠ **NARROWED HARD 2026-09-01c, AND THE GAP IS NOW A DECISION RATHER THAN AN ABSENCE.** The old text read "populated on every stock from three assumed values" with "any densitometer specification" as the ask. Two measured Q(D) curves are now in the corpus and the ask is answered. **Trumpy & Gschwind 2015 Fig. 5** (after Streiffert 1947, `trumpy_callier_q.py`, 573 points) fits the shipped Silberstein & Tuttle law to **rms 0.0087 Q over D 0.3–2.0** with E 0.1471 and **β 1.6746**, the two jointly identifiable. **BBC T-101 Fig. 25**, already cited on EASTMAN_TRI_X_5223, gives β 2.0–2.34 at 0.0016 sr where Q → β. The database gives all **55 B&W negatives the class constant 1.3** and all reversal 1.25 — **below both measurements** | the film half of C22's split is no longer undocumented, so the reason the `scanner_specular` control ships at zero has expired. What is left is a judgement on ~90 stocks' worth of pixels | ⚠ **NOT a search — an owner decision.** Either raise `callier_q` toward the measured β, or record why 1.3 is kept. ⚠ Weigh it with the model in mind: β is defined as a **film** property and the collection cone lives entirely in E, so a β held down "because a real condenser is not collimated" counts the geometry twice. A named-emulsion densitometer specification would still be worth having, but it is no longer the blocker |
| 7b | **A toe correction for the Callier law** | ⚠ **THE MODEL IS WRONG BELOW D ≈ 0.2 AND IT IS NOW MEASURED TWICE.** Silberstein & Tuttle's small-D limit is the CONSTANT E + (1−E)·β = **1.575** at the fitted parameters; the measurement is **1.081 at D 0.05** — **+0.49 Q**. Independently witnessed: Mees FIG. 179 reads **1.042 at D 0.055** (`mees_callier_q.py`), Trumpy/Streiffert **1.081 at D 0.058** (`trumpy_callier_q.py`). Two laboratories, two decades, two emulsions, no shared calibration | shadows and clear base are exactly where a condenser must NOT add contrast, and the law currently says it does | ⚠ **A FUNCTIONAL FORM ARRIVED 2026-09-01e.** Sayanagi 1959 (`JAPAN/23_20.pdf`) derives Q(D) from a Poisson grain model in which the developed grain has FINITE transmittance T_g, and states that his theory fits at LOW density and fails at high — the exact complement of Silberstein & Tuttle, which fits above D 0.3 and fails below. He also gives the method: T_g is estimated from the measured Q at D → 0. ⚠ And he independently states the convention this project chose without a source: Q must be defined on BASE-SUBTRACTED density, «Q_II is the rational Q factor». What is still missing is a parity plan. Neither figure names a film, so any correction is a CLASS shape, not a per-stock one |
| 8 | **rms granularity for eight KODAK still stocks** (queue K5) | eight stocks' grain is estimated and will stay so | ⚠ **proved unobtainable**: 201 KODAK files searched, 80+ print "diffuse rms granularity", **none for a PORTRA / GOLD / ULTRA MAX colour negative**. Kodak moved exactly that population to Print Grain Index and E-58 declines to publish the inverse |
| 8b | **The scanner half of the imaging chain is unmodelled, and the data to model it properly does not exist in the corpus** | every reference image of every stock that anyone has seen came through a scanner or a telecine, and the renderer assumes that stage is the identity. Flueckiger et al. 2018 measures how wrong that is: contrast reproduction at 20 lp/mm ranges **0.13 to 0.83** across eight machines on one test target | ⚠ **PARTLY ANSWERED AND PARTLY REFUSED, 2026-09-01d.** What the report supplies for all eight scanners -- MTF at seven spatial frequencies, sampling resolution in px/mm, light-source class, CFA presence -- is digitised in `SCANNER_CHARACTERISTICS.md`. What it does not supply for ANY of them is the **RGB spectral sensitivity**, and §2.8 says so in as many words; Figure 6 is captioned «**Typical** … spectral sensitivities of a color imaging device», a textbook illustration and not a measurement. So a scanner preset could model sharpness and sampling but not colour. ⚠ The verdict is to DEFER: see `SCANNER_CHARACTERISTICS.md` §5 -- with no ground-truth harness (row 9) a scanner stage adds free parameters to a system that cannot measure whether they help |
| 9 | **A ground-truth harness** | ⚠ **the largest gap in the project and it is not a document.** Every audit checks the database against its sources; **none checks a RENDER against a photograph** | scanned frames from stocks we model, with the scanner named |

## Database and corpus, live

**175 film stocks, 11 print stocks, 14 gauges, schema v24.** 134 negative / 41 reversal; 68
monochrome. Provenance tiers: **88 T1, 45 T2, 42 T3**.

⚠ **170 → 171 on 2026-09-02 (queue C4): `SVEMA_CO_90L`, ТУ 6-42-1514-90, the last Soviet amateur
reversal specification.** ⚠ **THE ROW THAT ASKED FOR IT WAS WRONG ABOUT ITS OWN SOURCE:** C4 read
«ЦО-90Д / ЦО-90Л — two documents with near-identical norms». There is ONE document, for ЦО-90Л, and
«ЦО-90Д» is an OCR misread of Л as Д on a typewritten page — which the same OCR performs
inconsistently *within a single file*. So one stock, not two, and the objection dissolved rather
than being overruled.

⚠ **166 → 170 on 2026-09-01, and THREE OF THESE FIVE NUMBERS WERE STALE BEFORE THE CHANGE.** The
negative/reversal split read 129/36 and the tier line read 80/45/40 while the database held
130/36 and 80/45/41 — the EKTAR 125 addition of 2026-08-31 moved them and nothing failed, because
`doc_consistency.py` registered the stock COUNT in the sentence above and not the split or the
tiers beside it. Registered 2026-09-01, so this cannot recur. The stock change itself is the four
AGFA profiles below.

⚠ **`SCHEMA_VERSION` read 18 until 2026-08-31 and was four versions stale.** v19–v22 were added on
2026-08-30 and 2026-08-31 — their fields carry `# -- schema v19/v20/v21/v22` comments throughout
`film_profiles.py` — while the constant, and therefore every document repeating it, said 18.
`doc_consistency.py` registers the two stock COUNTS in this sentence and not the schema number, so
the counts were guarded and the version was not. All four are additive and inert: a v22 database
renders bit-identically to a v18 one and no film index moves.

**Carrier census** — how much of the database is measured rather than estimated. Every number here
is checked against the live database by `doc_consistency.py`, so a stale figure fails the build
instead of quietly misleading a search:
**18 stocks carry a spectral dye-density set**,
**87 carry a spectral sensitivity set**,
**13 carry a measured σ(D) shape** and
**23 carry a measured MTF**.

Five carriers the census did not track until 2026-09-01, all populated by the AGFA harvest:
**12 stocks carry a published coated thickness**, **12 carry a published base thickness** (schema
v23 — the field did not exist before), **42 carry a manufacturer reciprocity table**, **11 carry a
processing family**, **16 carry a neutral + D-min pair** and **45 carry the manufacturer's own
emulsion designation**.

⚠ **`ReciprocityTable.development_correction_pct` IS NEW ON 2026-09-01 (schema v24) and closes what row 5e asked for**: Agfa print, on the same four time cells as the exposure correction, «Developing adjustment (%)» / «Entwicklungskorrektur (%)» = 0 / −10 / −25 / −35 for APX 100 and APX 400 — and **0 / 0 / 0 / 0 for APX 25**, which is a stated null and not an absent row. Three editions across six years agree cell for cell. Reciprocity failure raises development contrast as well as costing speed, and until now the database held only the speed half. Additive and inert, like the rest of `ReciprocityTable`.

⚠ **`EmulsionSpec.designation` IS NEW ON 2026-09-01 (schema v23) AND ITS EMPTY VALUE IS A FINDING,
NOT A BLANK.** It holds the maker's own identifier for the emulsion behind a product, copied
verbatim — Kodak/Eastman cine codes (`5219/7219`), Kodak still B&W letter codes (`TMX`, `TMZ`),
Agfa's printed colour-negative Negative code (`49-14`). A pair is ONE emulsion slit two ways, 35 mm
and 16 mm. `FilmActiveProfiles.md` prints **`-`** for the other **125** stocks, which is a checked
statement that no reliable identifier is published, not an unfilled cell.

⚠ **NO VALUE IS DERIVED FROM A PROFILE NAME.** A first attempt that read four-digit groups out of
names returned `1936` for `AGFACOLOR_NEU_1936`, `3200` for `ILFORD_DELTA_3200` and `1952` for four
different Kodak sheet films — two years, a speed and a Data Book edition, nineteen junk values in
fifty-five. Every stored value is attested in that profile's own cited source text, and the table
is a literal so it cannot drift. The positive half of the same argument: `CINESTILL_800T` carries
`5219/7219` although its name contains no code at all, because it is VISION3 500T with the remjet
stripped and CineStill say so in print — which is the correlation the field exists to record.

⚠ **78 → 79 and 16 → 17 on 2026-08-31 (queues B3 and E3).** `KODAK_TECHNICAL_PAN` gained its first
spectral set from P-255 p9. `KONICA_IMPRESA_50` became the 17th measured MTF and the first that is
neither vector-traced nor per-layer: its sheet is a scan end to end, so the curve comes off the
bitmap through `konica_raster.py`, and the panel prints ONE curve captioned *"Densitometry: Through
visual filter"* — so its f50 64.9 is pooled across the layers and written to all three fields.
`verify.py` names it in `_VISUAL_FILTER_MEASURED`, excludes it from the two family guards that
reason about red records, and asserts the pooling that licenses the exclusion.

⚠ **12 → 15 ON 2026-09-01, AND THE FIVE AGFA COLOUR NEGATIVES ON THE SAME SHEET DID NOT MOVE IT.**
The three **AGFACHROME RSX II** films are the first separated three-dye sets in the Agfa corpus:
their Spectral density panel prints **Yellow, Magenta, Cyan and a Visual grey**, so the dyes can be
checked against the neutral they must compose to — summed at all 31 sampled wavelengths they
reproduce the printed grey to **0.027–0.029 D rms**, which is a physical closure test the panel
supplies itself rather than a fit residual. The five colour NEGATIVES on the same sheet draw two
AGGREGATE curves instead ("Medium density", "Minimum density"), so they moved the **pair** counter
11 → 16 and left this one alone. That is the distinction the paragraph below exists to protect,
exercised for the first time.

⚠ **THE DYE-DENSITY FIGURE IS NOT STALE AND MUST NOT BE "CORRECTED" UPWARD.** Several stocks
carry the schema-v14 **neutral + D-min pair** rather than three separated dyes, because that is the
only shape the KODAK E-series still sheets and the KONICA sheets publish: one *Midscale Neutral*
curve and one *Minimum Density* curve. `has_data` still means "three dyes"; `has_neutral_pair`
reports the pair. Two different facts, two different counters, deliberately.

**Corpus.** ⚠ **57 PDFs as of 2026-09-01** — `agfa_films.pdf` was copied into the checkout so its
audit runs here rather than SKIPping; `agfa_bw_manual.pdf` was copied too, as the independent third reading of the APX and SCALA density curves; `agfa_film_chem.pdf` (P-16-C) followed on the same day, once the range sheet's own closing sentence was noticed to name it. ⚠ **This working
copy holds 59 PDFs under `PDF/PROFILES`, in seven directories —
`AGFA`, `FUJI`, `GEVAERT`, `KODAK`, `KONICA`, `POLAROID`, `RETRO`. The owner's machine holds 475, in
twenty.** Absent here and present there: `ILFORD` (24), `SOVIET STANDARDS` (32), `SOVIET` (13),
`ORWO` (9), `FOMACOLOR` (8), `ROLLEI` (7), `KENTMERE` (7), `DUFAYCOLOR` (4), `MACO` (4), `FERRANIA`
(3), `MISC` (2), plus 163 further KODAK sheets and 22 further FUJI ones. **Every path cited in this
file that does not resolve here resolves there** — the "(present)" annotations record where a file
was read, not what is on this disk. Five of `build.py`'s 28 audits SKIP for exactly this reason.

⚠ **This is the single biggest source of wrong blockers in the project.** On 2026-08-31 the
reconciliation found **seven of thirteen** rows filed as "acquisition" were nothing of the kind.
Before concluding a document is missing, check the owner's corpus, and check the publication code
against **page 1 of the file** rather than the filename.

**Basis.** Corpus reviewed: all PDFs under `PDF/PROFILES` plus 14 at `PDF/` root (~2.7 GB); the
systematic sweep of 2026-08-14; the 2026-08-15 reference batch; the machine inventory
(`plot_inventory.py`, which classifies every plot page vector-or-raster and is re-runnable); the
2026-08-31 full-corpus sweep of the owner's machine; and the database's own `provenance.sources`
and `description` fields, re-queried live.

## Rules of this file

1. **"Not found in an earlier pass" is not "not in the corpus."** Every absence claim states which
   documents were read before concluding it — and, since 2026-08-31, **which corpus** was searched.
2. **Nothing is estimated to make a row disappear.**
3. **Uncertainty is marked, not smoothed.** Where a designation, a manufacturer or a production date
   is not established by a document, it says so and says what would settle it. A profile name is
   *not* evidence of a product's existence.
4. Filter-derived exposure indices are filter factors, not film properties; their absence is not a
   gap.
5. **A publication code is not a document.** Three of this project's queue rows named codes that
   describe a different film than the row claimed. Verify against page 1.

---

## 0. STANDING CORRECTIONS — the findings that still bind

⚠ **The 400-line correction log that used to live here is in `NotFound_history.md`.** These are the
ones a reader still needs before using the sections below.

- **A stored value that passes every guard can still be a template.** `KONICA_IMPRESA_50` held a
  Dmin triple of 0.20 / 0.62 / 1.00 that was ordered, plausible, marked `fitted_from='datasheet_curve'`
  and **wrong in blue by 0.32 D**. `KONICA_VX_100` holds 0.21 / 0.63 / 1.02 and
  `KONICA_CENTURIA_SUPER_400` holds 0.22 / 0.65 / 1.05. Three stocks, one shape, round numbers.
  Nothing in `verify.py` could catch it, because nothing compared a stored triple to a reading.
  ⚠ **The other two are still unread.**
- **Kodak printed PORTRA 160VC's curves on the PORTRA 100T datasheet.** E-2468's entire CURVES page
  is figure `F009_0154AC`, the figure E-190 prints on its 160VC page; tracing both documents
  independently returns identical numbers to four decimals. A tungsten ISO 100 emulsion and a
  daylight ISO 160 emulsion cannot share a characteristic curve. `KODAK_PORTRA_100T`'s curves,
  spectral set and grain are therefore **not its own** — what E-2468 supplies uniquely is text.
- **Five films listed as undocumented in an old revision are not**, and four more were listed as
  having no source while holding traced curves: the PORTRA **160NC / 160VC / 400NC / 400VC**
  quartet was created from E-190 pp 9–12 on 2026-08-30 and no `_PROVENANCE_SOURCES` entry was
  written, so it fell through to the `_NO_DATASHEET` placeholder. ⚠ **That is the reverse of the
  usual defect — a measurement wearing a placeholder that says it has none.** Corrected 2026-08-31;
  the no-source count went 13 → 9.
- ⚠ **"AGFA FILMS USE SILVER HALIDE" IS NOT A LINK BETWEEN A PATENT AND A PRODUCT, and this
  one nearly became one.** US 4,495,277 is Agfa-Gevaert's own emulsion patent (Becker, Klötzer and
  Moisar, filed 1983-08-01) and it is genuinely Agfa house practice — but it **names no product**:
  no AGFAPAN, no APX, no ISO, DIN or ASA in six pages, and its speeds 290 / 100 / 445 are a
  relative scale. It also declines, in its own words, to constrain the two fields a reader reaches
  for — *"the mean grain size may vary within wide limits"* (0.3 to 2 µm, the entire field) and
  grains *"may assume the known forms, cubic, octahedral or … tetrahedral and decahedral"* (all
  four). ⚠ And its worked example is a **colour** coating: a yellow coupler, a CD-3 developer and a
  bleach-fix, on an iodide-free AgBr/AgCl/AgBr core-shell grain. Every emulsion in this corpus,
  from every maker, is silver halide; the phrase identifies nothing. **Only `sensitization = "S"`
  was taken, tier 3 / `assumed`, on the four Agfa B&W stocks, and it is labelled an inference from
  an assignee and a date in every note.** ⚠ Also: claim 1's **R ≥ 3 is a density ratio**, ripened
  over unripened, not an aspect ratio — an earlier note in this project used the phrase loosely.
- ⚠ **`emulsion.grain_um` is probably holding the wrong physical quantity on all 17 stocks that
  carry it.** Every one comes from a single third-party aggregator and the magnitudes are
  1.3–6.5 µm — Tri-X 4.5, TMAX P3200 6.5. **Two independent sources put real AgX crystals at
  0.2–2 µm**: JPS 1965's five measured emulsions (0.3 / 0.4 / 0.5 / 1.5 / 1.8 µm) and Trumpy &
  Gschwind §4 citing Vitale 2009 (*"silver particles typically span from 0.2 μm to 2 μm"*). The
  field is very likely a developed-CLUSTER figure under a crystal-size name. ⚠ Nothing was
  corrected — the point is that **nothing may be derived from that field** until it is.
- **ISO 5-3 was in the corpus the whole time.** `aimm.it2.18.1996.pdf` — the Status A and Status M
  spectral response tables — sat unread while the queue recorded them as missing.
- **The VECTOR granularity corpus was never exhausted.** An early pass concluded it was; 39 raster
  granularity pages remain on disk.
- **A raster sheet is not automatically unreadable.** `IMP50.pdf` and `INF750.pdf` are scans end to
  end — no paths, no tick text — and both were adopted from on 2026-08-31 by calibrating
  geometrically off the printed grid. ⚠ Their embedded bitmaps are also stored **upside down**;
  rotating 180° leaves the text mirror-reversed, which is how the flip announces itself.
- **`professional_160.pdf` is closed as unusable, not deferred.** All four pages extract **zero
  characters**, so there is no caption, axis label or legend to calibrate against — and its one
  technical page matches no stock in the database.
- ⚠ **THE AGFA CORPUS WAS MIS-CATALOGUED AND FOUR FILMS WERE LOST BEHIND IT** (2026-09-01). This
  file's row 5 recorded four Agfa candidates as "the same publication, two of them byte-identical".
  Two are; `agfa_films.pdf` is not — md5 `edb3dd17…`, **«Technical Data PF», 1st edition, 09/1998**,
  against the other three's F-PF-E4 of 08/2004. The 1998 edition plots **twelve** films where the
  2004 one plots four, and it is the only document in the corpus that plots **AGFACOLOR ULTRA 50**
  or the **AGFACHROME RSX II** line at all. All four are now profiles.
- ⚠ **THE SAME FILE WAS RECORDED AS RASTER AND HAS ZERO EMBEDDED IMAGES.** `agfa_2004_curves.py`'s
  docstring said AGFA_OPTIMA_100's spectral set came "from a RASTER page of the older 1998
  brochure". `page.get_images()` returns empty on all twelve pages; `get_drawings()` returns 116–172
  stroked objects on pp7–10. It was transcribed by eye in the 2026-08-02 batch because nobody
  checked — the same defect the APX spectral sets were corrected for on 2026-08-17.
- ⚠ **EIGHT PROFILES CLAIMED THEIR PUBLISHED GRANULARITY WAS UNPUBLISHED** (2026-09-01).
  `_PARAM_SOURCES_DERIVED` gave every Agfa stock a `grain.rms_granularity` cell reading «No
  published rms for this stock in the corpus». Agfa print the figure beside **every** plotted column
  — with its developer, time and temperature on the APX films — on a sheet each profile's own
  provenance string already named. ⚠ **THE VALUES WERE ALL CORRECT.** Only the provenance was
  wrong, which is the harder defect to find: nothing renders differently, and a search for
  "unsourced parameters" returns the same list before and after.
- ⚠ **AGFA'S GRANULARITY CONDITIONS NEARLY MATCH THIS PROJECT'S OWN CONVENTION, AND THE GAP IS ONE
  WORD.** The sheet's p5 states: daylight exposure, visual filter (Vλ) densitometry, **diffuse
  density 1.0, 48 µm reading aperture**. The project convention is 48 µm at **NET** density 1.0.
  Agfa do not say whether base+fog is subtracted, which is the whole reason these twelve cells are
  tier 2 and not tier 1.
- ⚠ **A DOCUMENT THE CORPUS ALREADY HELD WAS NAMED BY ANOTHER AND NOTHING CONNECTED THEM**
  (2026-09-01, second pass). `agfa_films.pdf` p11 ends: *"Further processing details are given in
  the Technical Data P-16-C."* P-16-C **is in the corpus** — as `agfa_film_chem.pdf`, a filename
  that reads like a chemistry catalogue. It prints, as TEXT, the developing time to reach γ 0.55 /
  0.65 / 0.75 for each AGFAPAN film in each of six developers, for drum and small tank: **64 cells,
  no tracing**. It supersedes the digitised Gamma-time panel adopted the same day, and it covers
  **ATOMAL FF**, which no Agfa panel in the corpus plots. **When a held document names another by
  publication code, resolve the reference.**
- ⚠ **THE 2004 HANDBOOK'S RODINAL TABLE IS DEFECTIVE, AND MONOTONICITY IS WHAT PROVED IT.** A longer
  development cannot give less contrast. Every (developer, method, film) triple in P-16-C ascends
  with γ; the handbook's γ 0.55 column falls against its own γ 0.65 column and clusters at
  10.4–10.8 min. The decisive row — RODINAL 1+25, small tank, γ 0.65 — reads **6 / 8 / 7 min** in
  *both* the 1998 range sheet and the 1999 P-16-C, and **18 / 15** in the handbook. Two independent
  documents against one. Recorded as defective, not averaged.
- ⚠ **THE AGFAPAN CHARACTERISTIC CURVES' DEVELOPMENT CONDITION IS NOW NAMED.** They were adopted at
  mid-slope 0.70–0.74 with a note saying the sheet "states no development at all" — true of that
  sheet, false of the corpus. P-16-C §3.4 states Agfa's three standard aims in words, and 0.74 is
  the **γ 0.75 aim**, with a printed time in every developer.
- ⚠ **ONE DEVELOPER, TWO SPELLINGS, AND EVERY JOIN SILENTLY MISSES.** Agfa print `RODINAL 1 + 25`
  in the range sheet and P-16-C, and `RODINAL 1+25` in the handbook. `ProcessVariant` already used
  the compact form; adopting the spaced one into `ProcessingFamily` gave one developer two names
  and broke a `verify.py` guard on the first build after adoption. The database normalises to the
  compact form; the reader reports as printed.
- ⚠ **AGFA REUSED ARTWORK BETWEEN FILMS, TWICE, AND BOTH TIMES IT LOOKS LIKE CORROBORATION.**
  APX 100 and APX 400 share one **sharpness** curve — two path objects in two columns with identical
  geometry, so their equal overshoots are equal by construction. RSX II 50 and RSX II 100 share one
  **spectral** drawing, tracing to within 0.002 lg at every sampled wavelength. Same shape as the
  PORTRA 100T / 160VC finding above; both are declared in the stored source strings and guarded.

---

## 1. STOCKS WITH NO SOURCE OF ANY KIND — **9 films** (re-queried live, 2026-08-31)

⚠ **RE-QUERIED 2026-08-31 AND THE COUNT IS 9 AGAIN, BY A DIFFERENT ROUTE.** Eleven profiles carry
nothing but the `_NO_DATASHEET` placeholder; **two of the eleven are `GENERIC_BW` and
`GENERIC_COLOR`, which are generic classes and not gaps**, leaving 9 real ones. The count had been
13 that morning: the PORTRA **160NC / 160VC / 400NC / 400VC** quartet was created from E-190
pp 9–12 on 2026-08-30 with traced characteristic curves and **no `_PROVENANCE_SOURCES` entry**, so
it fell through to the placeholder. Corrected in the same pass — a measurement must not wear a
label that says it has none.

The nine, and the query that would close each:

`EASTMAN_5250_1959`, `EASTMAN_5254_1968`, `EASTMAN_ORTHO_1930`, `GEVAERT_PANCHRO_1950`,
`LUMIERE_LUMICHROME`, `ORWOCOLOR_NC24`, `SOVIET_PANCHROM_1939`, `TASMA_FN_64`,
`TECHNICOLOR_THREE_STRIP`.

**Four entries below have MOVED OUT of this category and their subsections are kept only as
acquisition guides for a *better* source:**

| stock | now cites | when |
|---|---|---|
| `EASTMAN_5247_1974` | Kodak TI0835 **plus** Chibisov & al. 1988 Table VIII | 2026-08-18 |
| `FUJI_F125_8530` | Honjo 1989, J. Soc. Photogr. Sci. Technol. Japan **plus** «Техника кино и телевидения» 1989 №4 and 1990 №1 | 2026-08-18, extended 2026-08-24 |
| ~~`FUJI_F125_8630`~~ | **removed 2026-08-24** — a gauge clone, not a second emulsion; see §1.5 | — |
| `ILFORD_HPS` | Иофис 1964 table 7 p79; BBC M54 1964; BBC T-101 1963 | 2026-08-18, extended 2026-08-23 |

So the honest statement of the remaining gap is narrower than it was: those nine films have **no
manufacturer sheet, standard, book chapter or journal article in the corpus naming a single measured
value**, and everything they render with is an era estimate or an analogy. `GENERIC_BW` and
`GENERIC_COLOR` also carry only the placeholder and are excluded on purpose — they are generic
classes, undocumentable by definition and not actionable gaps.

**Corpus-wide gaps that apply to every film in this section** (listed once rather than repeated):
`LayerStack` and `ReciprocityTable` are empty for all of them and are absent corpus-wide for these
eras — see §3. (Corpus-wide the reciprocity carrier is no longer thin: **21 measured
`ReciprocityTable` entries** as of 2026-08-23, 15 of them read from vendor sheets, plus a
Schwarzschild exponent on 105 stocks — but none of that reaches the nine films in this section.)

---

### 1.1 `EASTMAN_5247_1974` — ✅ **SOURCED 2026-08-18** (Kodak TI0835 + Chibisov 1988); kept as a guide to a better source

* **Designation:** EASTMAN Color Negative Film 5247, **original 1974 coating, EI 100T**.
* **Manufacturer:** Eastman Kodak Company, Rochester NY.
* **Period:** 1974 – c. 1982 (stored `era`). Superseded by a later coating that **kept the
  same 5247 number** at EI 125T, then by 5294 in 1983.
* **Type / application:** colour negative, motion picture camera film, 35 mm (Super 35
  geometry), process ECN-2, clear acetate base with rem-jet backing.
* **Aliases in the database:** `5247`, `eastman 5247`. ⚠ The bare alias `5247` resolves
  **here**, to the undocumented generation — see the generation note.
* **Missing:** spectral sensitivity, spectral dye density, dye impurity / interimage,
  reciprocity. Curves, grain, MTF and halation are era estimates, **not measurements of
  this emulsion**.
* **⚠ GENERATION IDENTIFICATION — THE CRITICAL POINT FOR ANY SEARCH.** Kodak reused the
  designation 5247 across a coating change, and **every 5247 document in this corpus
  describes the LATER film, not this one**:
  * TI0835 (rev. 6-93) prints "Tungsten (3200K) — 125/22";
  * Chibisov 1988, table VIII p 165, prints "Kodak 5247: S **125** GOST";
  * Sehlin/Kennel, SMPTE Journal July 1985, compares 5247 against 5294 (launched 1983).

  Those sources now sit on `EASTMAN_5247_1983`. **When requesting data for this film, ask
  explicitly for the EI 100T / 1974–1979 coating and state that you do *not* want the
  EI 125T revision** — otherwise you will be sent TI0835 again.
* **What to ask for:** the original 1974 5247 technical data sheet (the TI0835 predecessor,
  publication number unknown to this corpus). Kodak's own chronology page is the only
  official record found for the 1974 introduction.
* **Likely holders:** Eastman Kodak historical archives; SMPTE Journal 1974–1976; George
  Eastman Museum (Rochester); the Kodak Research Laboratories records at the University of
  Rochester — expected to reopen early 2027 (see `ROADMAP_2026-08-17_fidelity.md` Priority 4).

### 1.2 `EASTMAN_5250_1959`

* **Designation:** Eastman Color Negative Film 5250.
* **Manufacturer:** Eastman Kodak Company.
* **Period:** 1959 – 1962 (official: Kodak chronology page). Replaced 5248 (EI 25);
  replaced by 5251 in 1962.
* **Type / application:** colour negative, motion picture camera, 35 mm, process **ECN**
  (the original, *not* ECN-2).
* **Aliases:** `5250`, `ecn 5250`, `eastman 5250`.
* **Documented:** EI 50 tungsten / 32 daylight, process, introduction and replacement
  dates — all from Kodak's chronology page (official, but a chronology, not a datasheet).
* **Missing:** everything sensitometric — curves/gamma, granularity, resolving power, MTF,
  spectral sensitivity, spectral dye density, dye impurity, reciprocity.
* **Generation note:** ⚠ modern files named "5250" are unrelated; Kodak has reused
  four-digit numbers extensively. Verify any document's date before use.
* **What to ask for:** the period Eastman Color Negative 5250 data sheet (print only).
* **Likely holders:** Kodak historical archives; SMPTE Journal 1959–1962; AMPAS Margaret
  Herrick Library; George Eastman Museum.

### 1.3 `EASTMAN_5254_1968`

* **Designation:** Eastman Color Negative Film 5254.
* **Manufacturer:** Eastman Kodak Company.
* **Period:** 1968 – **discontinued March 1977** (official: Kodak chronology). Replaced
  5251. Image structure stated equal to 5251. Academy Award.
* **Type / application:** colour negative, motion picture camera, 35 mm, ECN.
* **Aliases:** `5254`, `ecn 5254`, `eastman 5254`.
* **Documented:** EI 100T, dates, replacement lineage — Kodak chronology only.
* **Missing:** all sensitometry, as 5250 above.
* **⚠ Generation note:** files named "5254" in this corpus are **code reuses** (VISION3 DI
  stock), not this film. Same for "5294" (Ektachrome 100D). Check the publication date.
* **Likely holders:** as 5250. The 5251/5254 pair is well covered in SMPTE literature of
  the period, which is the most promising single lead in this section.

### 1.4 `EASTMAN_ORTHO_1930`

* **Designation:** ⚠ **not an exact product designation.** A generic Eastman
  orthochromatic B&W cine negative of the period, not a named catalogue product.
* **Manufacturer:** Eastman Kodak Company (assumed from the profile name; **unverified** —
  no document in the corpus ties this profile to a specific Kodak product).
* **Period:** 1920s – early 1930s.
* **Type / application:** B&W orthochromatic negative, motion picture camera, 35 mm.
  Blue-and-green sensitive, effectively red-blind.
* **Aliases:** `ortho`, `orthochromatic`, `1930 ortho`, `eastman ortho`.
* **Missing:** everything. EI 25 and the 3400 K balance are era estimates.
* **⚠ Before searching, the profile needs re-scoping.** Because this is a generic class
  rather than a product, no datasheet can match it. The actionable move is to pick a
  *named* period product (e.g. a specific Eastman Ortho Negative catalogue number) and
  re-scope the profile to it — otherwise any document found will be a graft. **Foma ortho
  sheets are a different manufacturer and have never been substituted.**
* **Likely holders:** pre-datasheet era; Kodak archives and the trade press
  (*American Cinematographer* 1925–1935) are the realistic routes.

### 1.5 `FUJI_F125_8530` — ✅ **PARTLY SOURCED 2026-08-18** (Honjo 1989: f50 corrected 78 → 42 c/mm), **granularity measured 2026-08-24**; still no vendor sheet **for type 8530 itself**

> **⚠ `FUJI_F125_8630` NO LONGER EXISTS, removed 2026-08-24, owner-approved.** It was a byte-for-byte clone of 8530 differing only in `default_format`. What made keeping it indefensible rather than merely redundant is that **Fuji's own code rule is printed in words** in «Техника кино и телевидения» 1989 №4, journal p70: *«первая цифра 8 свидетельствует о том, что кинопленка цветная негативная, вторая обозначает размер кинопленки (5 — 35 мм, 6 — 16 мм), а последние две цифры обозначают код каждой кинопленки»*. The **second digit is the gauge** and carries no photographic meaning. The rule is applied consistently across all five F-series stocks in three separate tables (1989 №4 Table 2 p70; 1990 №1 Tables 1 and 3, pp 56 and 59), and Fuji's own Super F-125 sheet prints the same pairing for the later generation — *"35mm Type 8532 / 16mm Type 8632"*. A gauge is `default_format`, which this file already models.
>
> **⚠ AND THE SAME RULE SAYS 8530 AND 8532 MUST STAY SEPARATE.** They differ in the **last two** digits — the part of the code that identifies the film — and they measure differently: rms **4.0** against **3.0** at identical speed (125 tungsten / 80 daylight). A 25 % grain difference at the same speed is a difference this renderer expresses, so merging them would lose a real photographic distinction. `8630` and `8632` resolve as **aliases** of their 35 mm profiles.

> **⚠ Read the heading precisely (reworded 2026-08-23, owner-reported).** It used to end "still
> no Fuji datasheet", which invited exactly the wrong conclusion — that this corpus holds no
> vendor documentation for Fuji F-125 at all. It holds a complete one: the **8532/8632 sheet**,
> Ref. No. KB-913E, ©1999, titled *FUJICOLOR NEGATIVE FILM F-125*, fully exploited since
> 2026-08-23 (§1.5a). What is missing is a sheet for the **earlier emulsion, type 8530/8630**.

* **Designation:** Fujicolor **F-125**, type **8530** (35 mm) and type **8630** (16 mm).
* **Manufacturer:** Fuji Photo Film Co., Ltd., Japan.
* **Period:** 1980s – 1990s (stored `era`; **exact introduction year unverified**).
* **Type / application:** colour negative, tungsten-balanced (3200 K), motion picture
  camera film. 8530 and 8630 are the **same emulsion in two gauges**.
* **Aliases:** `f125`, `f-125`, `8530`, `8630`.
* **Missing:** curves/gamma, granularity, resolving power, spectral sensitivity, spectral dye
  density, dye impurity, reciprocity. ✅ **MTF and speed are no longer missing** — see below.
* ✅ **PARTIALLY DOCUMENTED 2026-08-18.** `PDF/PROFILES/FUJI/52_509.pdf` (Honjo, *J. Soc.
  Photogr. Sci. Technol. Japan* **52(6), 1989, pp. 509–516**, written at Fuji's own Ashigara
  Research Laboratories) names **type 8530 explicitly** and prints **ν₅₀ = 42 c/mm** for it
  (Table 1.2, visual density, magenta/G record) plus an MTF curve labelled *Shooting Nega Film
  (E.I. 125)* (Fig. 3). Both gauges' `f50` were corrected 78 → 42 c/mm on the strength of it.
  Full reading in §0.2.1. ⚠ It is a **review essay, not a datasheet** — it grounds the MTF and
  corroborates the speed, and nothing else.
* **⚠ Generation and adjacency traps, all verified:** 8522 = F-64D and 8573 = ETERNA 500 —
  adjacent type numbers, different films. The file `F125 - 8532.png` in the corpus
  documents the **successor generation 8532/8632** (RMS 3.0) and was deliberately **not**
  back-applied. So when requesting data, give the type number **and** the generation:
  "F-125 type 8530/8630, not 8532/8632".
* ✅ **GRANULARITY MEASURED 2026-08-24, and the missing list above shrinks by one.** The owner
  added two issues of **«Техника кино и телевидения»** (1989 №4, 1990 №1) which carry the Fuji
  F-series material from the **1988 Moscow Fuji symposium** — the 1990 one as a *translation of
  Fuji's own paper* (Kozo Noguchi, Yukihide Urata, Koichi Murai; prepared by A. V. Redko), the
  1989 one as a review by Дьяконов & Редько citing the same symposium plus SMPTE J.
  ⚠ **They are one source in two renderings, not two independent confirmations** — where they
  agree numerically that is a transcription check.
  * **ADOPTED: `rms_granularity` 5.4 (estimate) → 4.0 (printed).** 1989 №4 Table 1, journal p70,
    read off the page image at 300 dpi. Convention matches this field exactly — 1990 №1 Fig. 4
    states 48 µm aperture at visual diffuse density 1.0, which is how Fuji's own 8532 sheet
    defines its 3.0. Cross-checked against the graph (0.0036–0.0040 at D 1.0) and **coherent with
    the successor**: 4.0 → 3.0 is the 25 % improvement the 8532 sheet claims.
  * ⚠ **A THIRD MTF MEASUREMENT ARRIVED AND DISAGREES. Nothing was changed.** 1990 №1 Fig. 3
    (journal p57) plots response to 60 mm⁻¹ at visual density 1.0, F-125 **and F-250 as one
    curve**. Traced off the 600 dpi image it gives **f50 ≈ 33 mm⁻¹** and T(30) = **0.552**, while
    the 1989 table prints **0.60** at 30 mm⁻¹ and Honjo prints **ν₅₀ = 42**. The other two traces
    on the same plate agree with the table to 2–3 % (A8511/8521 traced 0.435 vs printed 0.42;
    F-64 traced 0.774 against the table's separate 0.80/0.76 pair, which one drawn curve for two
    stocks explains). So the disagreement is specific to F-125 and is 8 %. **Method rule 4** —
    recorded, not averaged. What it does do is **reframe the conflict on `FUJI_SUPER_F125_8532`**:
    that profile's Coltman-converted **32.07** looked like a regression against 42.0, and this
    third figure lands at **33** — two of three sources now cluster at 32–33 and Honjo's 42 is the
    outlier. Grounds to re-examine what ν₅₀ means in Honjo, not to overwrite it.
  * ⚠ **MTF vs CTF is still unresolved on this plate, and it is worth ~4/π.** Fig. 3's caption
    says *Функция передачи модуляции* (MTF); the body text on the same page says *«резкость
    выражена через физический параметр CTF (Contrast Transfer Function)»*. Evidence for the
    caption: all three traces overshoot to ≈1.06 near ν = 8–12, matching the DIR adjacency
    overshoot Honjo reports independently (113 % at 8–10 c/mm) and which `adjacency` 0.13 already
    encodes — so the overshoot needs no square wave to explain it.
  * ⚠ **THREE FIGURES DELIBERATELY LEFT UNHARVESTED, each for a stated reason**, so a later pass
    does not re-litigate them:
    **(1) σ(D)** — 1990 №1 Fig. 4 is a full measured σ(D) curve, 48 µm, D 0.35–1.3, and
    `sigma_shape_*` exists for it. Rejected because F-125 and F-64/F-64D **converge inside the
    drawn line width above D ≈ 0.9**, which is exactly where the only validating anchor (the
    printed 4.0 at D 1.0) sits; a column read there returns 0.0034 and 0.0036 as two runs that
    may be either curve. Same failure class as VISION3 batch 8 and Fig. 18's Pan F.
    It does establish one thing qualitatively, against the legacy √D law: **σ peaks near D ≈ 0.47
    and falls monotonically to D 1.3 on all six emulsions.**
    **(2) Characteristic curves** — 1990 №1 Fig. 1 draws F-125 and type A **superimposed at
    matched speed** (*«совпадающих по светочувствительности»*), so each of the three visible
    tracks carries two films; and the abscissa has **no numeric labels at all**, only a
    0.5-decade scale bar. Fig. 2 (F-64/F-125/F-250, nine curves) has the same abscissa limit but
    not the superposition, and is the better candidate if gammas are ever wanted.
    **(3) Spectral sensitivity** — 1990 №1 Fig. 6 plots this emulsion against type A over
    400–700 nm, but states **no density criterion**, so `SpectralSensitivity.criterion` would
    have to be invented.
  * **CORROBORATED, not adopted:** EI **125** at 3200 K and **80** in daylight with Fuji LBA-12 /
    Wratten 85 (1990 Table 2, p56) — identical to the 8532 sheet, so the stored 125 now has two
    documents behind it; edge marking **№ 30**, yellow packaging (1990 Table 3, p59); emulsion
    thinner by *«более чем на 10 %»* than type A, with the **green record in three sub-layers**
    against two for blue and red (1990 Fig. 5, p57).
  * ⚠ **One transcription fault in the 1989 article, recorded so it cannot be copied:** its
    Table 2 gives F-500 as **8415/8525** where Table 1 of the same article, the 1990
    Fuji-authored table and the footnote *«Код кинопленки F-500 не изменился»* all say
    **8514/8524**. The F-125 row is identical in every table. Prefer the 1990 rendering for codes.
* **Reviewed:** all 37+ Fuji PDFs including the 2011 cine manual, plus the two
  «Техника кино и телевидения» issues (2026-08-24).
* **What to ask for:** the Fuji Film Data Sheet for F-125 8530/8630 (Fuji's MP-series
  bulletins; the A250 sheet in this corpus is MP3-57E, so the F-125 sheet will carry a
  comparable MP number).
* **Likely holders:** Fujifilm Corporation technical archives; older Fujifilm cine
  bulletins; Japanese motion-picture industry libraries.
* ✅ **The tier-justification gap logged here earlier on 2026-08-18 is closed.** These were the
  last two profiles in the database whose tier claim had no citation behind it; Honjo 1989 (above)
  supplied one, and the `verify.py` allowlist that tracked the gap is now empty. **Everything
  except the MTF and the speed is still class-estimate grade** — a real Fuji data sheet would
  still be the single most valuable acquisition for this stock.
* **⚠ A NEW, SHARPER ASK, because we now know what the successor sheet looks like.** The owner
  also supplied `FUJI/Fujifilm-Super-F-125-8532-35mm-Motion-Picture-Film.pdf`, the **complete**
  2-page sheet for the *next* generation (**35 mm type 8532 / 16 mm type 8632**, Ref. No.
  **KB-913E**, ©1999). It is exactly the document class we want for 8530/8630, and it proves
  Fuji issued such sheets in this layout: four vector plots plus a printed block giving EI,
  RMS granularity, reciprocity, resolving-class data and edge markings. **So the request to make
  is concrete:** the equivalent Super-F / F-series sheet for **type 8530/8630** — a `KB-`prefixed
  Ref. No. earlier than KB-913E, or the MP-series bulletin of the 1980s (the A250 sheet in this
  corpus is MP3-57E, so an F-125 8530 bulletin should carry a comparable MP number). Ask for it
  by **Ref. No. and type number together**, and state the generation explicitly:
  *"F-125 type 8530/8630, the generation BEFORE 8532/8632."*

#### 1.5a The 8532/8632 sheet — fully exploited 2026-08-23, and what it may *not* be used for

The successor sheet is no longer just evidence that such sheets exist. Everything printed or
plotted on it is now in the database under `FUJI_SUPER_F125_8532`:

| From the 8532 sheet | Status |
| --- | --- |
| EI 125 (3200 K) + the seven-row illuminant/filter/EI table | stored (filter rows as factors, not second speeds) |
| RMS granularity 3.0, net D 1.0, 48 µm | stored as printed — the sheet states the convention explicitly |
| Reciprocity: flat 1/1000–1/10 s, +1/3 stop at 1 s, achromatic | stored as a measured table |
| Characteristic curves, 3 layers | ✅ **traced 2026-08-23**, rms 0.005–0.009 D |
| Contrast transfer function | ✅ **Coltman-converted 2026-08-23**, sine f50 32.07 c/mm |
| Spectral sensitivity, 3 layers | ✅ **traced 2026-08-23**, peaks 469/553/645 nm |
| Spectral density (neutral + Dmin) | read, **not storable** — the schema wants three separated dyes |
| Edge markings: MR code, mark FN32, name FUJI F-125, frame marks 5/8/15 (65 mm), 4 (35 mm), none (16 mm) | recorded in the citation |
| Ref. No. KB-913E, ©1999, print code SK·99·05 | recorded; `era` corrected to 1999-2000s |

**⚠ NONE OF IT TRANSFERS TO 8530/8630, and the sheet is the reason why.** Fuji's own page sells
8532 as a *new emulsion under the old name* — "Announcing the new Fujifilm F-125", "the newly
upgraded F-125", **SUFG** flat hexagonal grain "just 1/3 the size of conventional grain",
**Two-Stage Timing DIR Couplers**, and "a more linear response curve … minimal 'blocking up' of
dark tones". Grafting 8532's curves onto 8530 would assert the opposite of what the vendor
prints. The measured numbers make the gap concrete rather than rhetorical: 8532 reads RMS **3.0**
against 8530's estimated 5.4, and Dmin **0.19/0.44/0.78** against the 0.20/0.60/0.98 that the
old transfer *in the other direction* had put on 8532.

**⚠ And one conflict is left standing rather than resolved** (method rule 4). 8532's own panel
converts to f50 32.1 c/mm while Honjo measures **42.0** for the 8530 it replaced — so the
measurement says the successor is *softer*, while Fuji's page sells it on "dramatically increased
sharpness". The two are not the same measurement (Honjo: visual-density MTF of a master negative
inside a duplication chain; the sheet: square-wave response at visual diffuse density 1.0), and
both are on record. A per-layer sine-wave MTF for either generation would settle it.

**⚠ Extraction hazard worth carrying to any future Super-F sheet.** The exposure axis on both
Super-F sheets in this corpus (8532 and 8572) is **mis-labelled**: the ten uniformly spaced
gridlines read `−4.5 −3.0 −3.5 −2.0 −2.5 −1.0 −1.5 0.0 0.5 1.0`, which is **not monotonic**, so
the sheet contradicts itself and at least four labels are wrong. The gridlines are exact, so only
the origin is in doubt; it was settled at −4.5 by requiring the fitted toe and mid-grey density to
land inside the range the traced Kodak stocks occupy, and cross-checked by the two sheets' own
speed points, which sit 0.577 decades apart against the 0.602 their printed EIs (125 and 500)
demand — 0.08 stop. **Do not read those labels literally.**

### 1.6 `GEVAERT_PANCHRO_1950`

* **Designation:** ⚠ **generic, not a product designation.** "Gevaert panchromatic B&W cine
  negative, around EI 32."
* **Manufacturer:** Gevaert Photo-Producten N.V., Belgium (from 1964, Agfa-Gevaert).
* **Period:** 1940s – 1960s.
* **Type / application:** B&W panchromatic negative, motion picture camera, 35 mm.
  Registers red weakly.
* **Aliases:** `gevaert`, `gevaert panchro`, `panchro 1950`, `geva bw`.
* **Missing:** everything; EI 32 is an era estimate.
* **⚠ Needs re-scoping to a named product before any search can succeed** — same problem
  as §1.4. Gevaert's B&W cine negatives carried names such as the Gevapan series; picking
  one and re-scoping is the prerequisite.
* **Reviewed:** all Agfa/Gevaert files; Cheltsov 1958; Enticknap 2013 (swept — zero
  product numbers).
* **Likely holders:** Agfa-Gevaert company archive (Mortsel, Belgium); FOMU (Fotomuseum
  Antwerpen); Belgian and Indian film-industry archives — this stock was heavily used in
  Indian production of the 1940s–50s.

### 1.7 `LUMIERE_LUMICHROME`

* **Designation:** ⚠ **uncertain.** "Lumichrome" is carried as the product name; the
  corpus contains **no document naming it**. Treat as unconfirmed.
* **Manufacturer:** Société Lumière, Lyon, France. Manufactured independently **until
  Ilford absorbed the company in 1961**.
* **Period:** 1940s – 1961.
* **Type / application:** B&W negative, still photography, 35 mm.
* **Aliases:** `lumiere`, `lumichrome`.
* **Missing:** everything. **The profile's own description calls it "the most speculative
  profile in this database".** EI 40 and the curve shape are inference from the general
  behaviour of French B&W negative of the period.
* **⚠ First verify the product exists under this name.** Lumière's B&W range used names
  such as Lumipan and Opta; "Lumichrome" may be a colour-process name or a
  misremembering. **Confirm the designation before requesting technical data.**
* **Reviewed:** whole corpus; Wall 1929 mentions Lumière historically with no product data.
* **Likely holders:** Institut Lumière (Lyon); Ilford/HARMAN historical records (post-1961
  ownership); French photographic-society journals.

### 1.8 `ORWOCOLOR_NC24`

* **Designation:** ⚠⚠ **THE DESIGNATION IS NOT CONFIRMED TO EXIST.** The profile's own
  description states: *"I could not confirm 'NC 24' as a shipped ORWO product designation.
  The documented NC series runs NC 3, NC 5, NC 16, NC 19, NC 21."* This is a **family
  interpolation, not a product.**
* **Manufacturer:** ORWO — VEB Filmfabrik Wolfen, GDR (post-1990, Filmotec GmbH).
* **Period:** 1980s – 1990s (assumed for the interpolated position; unverified).
* **Type / application:** colour negative, motion picture, 35 mm.
* **Aliases:** `nc24`, `nc-24`, `orwocolor nc24`, `orwo nc 24`.
* **Missing:** everything.
* **⚠ THE FIRST QUESTION IS NOT TECHNICAL.** Ask ORWO/Filmotec or the Wolfen archive
  **whether NC 24 ever shipped**. If it did not, the correct action is to retire the
  profile or re-scope it to a documented member of the series — not to keep searching for
  data that cannot exist. If a real speed or datasheet surfaces, the profile is to be
  refitted.
* **Reviewed:** all 9 ORWO PDFs; Zhurba 1984 (**zero ORWO content**); Zhurba 1990 Table 66
  (owner scans) lists NC19 / NC21 / Typ L and **no NC24**.
* **Likely holders:** Filmotec GmbH (Wolfen); Industrie- und Filmmuseum Wolfen;
  **ORWO Handbuch** (the company's own technical handbook, not in this corpus — the single
  most promising document for all ORWO stocks).

### 1.9 `SOVIET_PANCHROM_1939`

* **Designation:** ⚠ **generic.** "Soviet panchromatic negative of the late 1930s."
* **Manufacturer:** the **Shostka film factory** (Ukrainian SSR), per the profile
  description. ⚠ The brand name **"Svema" postdates this era** and must not be used in a
  search for this period; the plant is the searchable entity.
* **Period:** 1930s – 1940s.
* **Type / application:** B&W panchromatic negative, motion picture, 35 mm. Known for
  severe batch-to-batch and within-roll sensitivity variation.
* **Aliases:** `panchrom`, `sovkino`, `shostka`, `soviet 1939`, `kinoplenka`.
* **Missing:** everything; EI 45 is an era estimate.
* **⚠ A near-miss that is deliberately NOT used, and the reason matters.** Gorokhovskii,
  «Методы спектральной сенситометрии», *УФН* XVI(4) 1936, **Fig. 7** plots a measured
  spectral curve for **«Изопанхром ФОКХТ»** (ГОИ Leningrad, 1934–35, criterion D = 1.0
  above fog) — the only measured spectral data for any 1930s Soviet material in the
  corpus. **Not attached:** Изопанхром is a *named product* and this profile is a generic
  class; grafting one onto the other is the FP4-vs-FP4-Plus error. **If this profile were
  re-scoped to Изопанхром ФОКХТ, that curve becomes immediately usable** — this is the
  single highest-value re-scoping decision in this file.
* **Reviewed:** all Soviet folders and standards.
* **Likely holders:** ГОСТ archives; ГОИ (State Optical Institute) publications;
  Госфильмофонд; the Шостка plant's own technical literature.

### 1.10 `TASMA_FN_64`

* **Designation:** ⚠ **ambiguous.** Both **ФН-64 (FN-64)** and **ФН-65 (FN-65)** circulate
  for this stock; the database carries both as aliases.
* **Manufacturer:** **Tasma**, Kazan (Tatar ASSR) — Svema's rival supplier to Soviet
  studios.
* **Period:** 1960s – 1990s.
* **Type / application:** B&W negative, motion picture camera, 35 mm.
* **Aliases:** `tasma`, `fn64t`, `tasma fn64`, `fn65`, `fn-65`, `tasma fn65`.
* **Missing:** spectral sensitivity, resolving power, reciprocity; granularity and curve
  numbers are analogy from the Svema equivalents.
* **⚠ Designation and maker traps, both real in this database.** (a) Svema also made an
  FN-64 — the aliases `fn64`/`fn-64` overlap with `SVEMA_FOTO_65`, whose cine designation
  was FN-64. **State the plant (Kazan) as well as the mark when requesting data.**
  (b) A related caution from `TASMA_POSITIVE_28`: the Tasma attribution there rests on
  *owner recollection*, not on a document — Иофис lists Soviet positive stock only as
  «Отечественное» (domestic) and names no factory. Do not assume Tasma attribution is
  documented for any Soviet stock unless a standard names it.
* **Reviewed:** all Soviet folders; 15 of 16 GOST files are method/dimension standards;
  Zhurba 1984 tables read visually.
* **Standards to request by number:** **ГОСТ 24876-81** (B&W negative cine film,
  technical conditions — held, covers other marks); ГОСТ 10691 series (sensitometry
  methods); the plant's own **ТУ** (technical conditions) documents, which are the level at
  which per-mark norms are actually published.
* **Likely holders:** Rosstandart / ГОСТ archive; Tasma (Kazan) successor entity;
  Госфильмофонд; Иофис and Журба later editions.

### 1.11 `TECHNICOLOR_THREE_STRIP` — ✅ **PARTLY SOURCED 2026-09-01d**: the three imbibition transfer dyes are now MEASURED (Flueckiger et al. 2018 §2.8.2 Fig. 16, a 1949 SAMSON AND DELILAH print on a SHIMADZU UV-1800, Ohta PCA separation). ⚠ SHAPE ONLY — that figure's ordinate carries no scale, no ticks and no label, so the curves are stored peak-normalised with the axis assumed to be zero absorbance. Everything else on the profile is still tier 3, and no characteristic curve, granularity or MTF for Technicolor exists anywhere in this corpus.

* **Designation:** Technicolor **Process 4** three-strip.
* **Manufacturer:** Technicolor Motion Picture Corporation (the camera negatives
  themselves were Eastman Kodak B&W stock).
* **Period:** 1932 – 1955.
* **Type / application:** three separate B&W negative records exposed through a
  beam-splitter camera behind broad overlapping filters; printed by imbibition dye
  transfer. Motion picture, 35 mm.
* **Aliases:** `technicolor`, `three strip`, `three-strip`, `process 4`, `ib tech`.
* **Missing:** everything measured. EI 5 and the taking-filter behaviour are estimates.
* **⚠ What to ask for is unusual here, and asking for the wrong thing wastes the enquiry.**
  This is a *system*, not one emulsion. Four distinct data sets are needed and should be
  requested separately: (a) the three camera-negative emulsions' sensitometry; (b) the
  **taking-filter transmittances** of the beam splitter; (c) the **imbibition transfer dye
  spectral densities**; (d) registration tolerances between the three records. (b) and (c)
  are what actually produce the look.
* **Reviewed:** whole corpus.
* **Likely holders:** SMPTE Journal / *Journal of the SMPE* 1932–1955 (Technicolor
  published extensively); AMPAS Margaret Herrick Library; the George Eastman Museum
  Technicolor collection; Technicolor corporate records.

### 1.12 `ILFORD_HPS` — ✅ **MEASURED 2026-08-23** (two contemporaneous BBC research reports; Иофис 1964 is no longer the only source)

* **Designation:** Ilford **HPS**, ASA 400 daylight / 320 tungsten.
* **Manufacturer:** Ilford Limited, Britain.
* **Period:** **1954 – 1960s** (HPS dates from 1954).
* **Type / application:** B&W negative, still photography, 35 mm; also bulk-spliced into
  cine lengths in period practice (Coutard, *Breathless*, 1960).
* **Aliases:** `hps`, `ilford hps`, `nouvelle vague`, `hps 800` (⚠ "hps 800" is a *push
  rating*, not a box speed — the profile's speed was corrected 800 → 400 on 2026-08-17).
* **Missing:** spectral sensitivity, **characteristic curve**, resolving power / MTF, Dmax,
  reciprocity. ⚠ **RMS granularity is no longer missing** — see below.
* **Partial / insufficient:** the toe, shoulder, `dmin` 0.21 and `fog_grain` 0.40 remain
  declared **rendering intent** for the pushed Nouvelle-Vague look, not measurement. `f50`
  26.0 is still an unsourced estimate. The Иофис citation stays on record and method rule 14
  still applies to all three sources — an Ilford sheet would outrank every one of them.
* **⚠ Two figures often quoted for HPS are NOT in any document in this corpus, and are not
  stored:** a later **800 ASA / 30 DIN** rating, and a **resolving power of 40 lp/mm**. The
  nearest thing to either is Monograph 54's own footnote to Table I — *"Earlier speed ratings
  (prior to the revised indices)"* — which is a plausible origin for a later 800 figure and a
  reason not to treat 320/400 and 800 as contradictory. The "40 cycles/mm" in T-101 p38 is an
  assumed **system bandwidth** for a granularity comparison, not a film resolving power, and
  reading it as one would repeat the CTF-versus-MTF unit error that queue item C11 existed to
  avoid.

#### ✅ What the two BBC documents ground (added 2026-08-23)

Both are third-party BBC research reports, so **method rule 14 still applies**. Two things
nevertheless raise them above the Soviet handbook: they are **primary measurements** rather
than a compilation, and T-101's speed table is headed **MANUFACTURERS' DATA**, so its ASA
figures are Ilford's own relayed. Both are **image-only scans with an OCR layer**; every value
below was read from the page image at 170 dpi, the OCR used only to find the pages.

| Quantity | Value | Source | Status |
| --- | --- | --- | --- |
| Speed, tungsten | **320 ASA** | T-101 T1 p27; M54 TI p12 | ✅ confirmed, independently of Иофис |
| Speed, daylight | **400 ASA** | T-101 T1 p27 | ✅ confirmed |
| BS logarithmic | 36° | M54 TI p12 | consistent with 320 ASA (10·log₁₀ASA + 10 → 36.1°) |
| **Development gamma** | **0.63** | T-101 T2 p28, T4 p38 | ✅ **ADOPTED**, replacing the 0.62 estimate |
| Grain Wiener spectrum, 0–20 c/mm | **0.62 µm²** at D 0.48 above base | M54 TI p12 | ✅ recorded; corroborates `rms_granularity` |
| Equivalent grain diameter | **2.5 µm**, stated as an upper bound (T-101 p38) | T-101 T2 p28 | ✅ consistent with the **1.900 µm** fitted from Fig. 8, below |
| Relative granularity, 5302 = 1 | 3.9 over 0–40 c/mm | T-101 T4 p38 | recorded |
| Spectrum flatness | falls ≈10 % over 0–60 c/mm | T-101 p38 | recorded |
| Description / use, as published | "Panchromatic film of extreme speed" / "special cine-camera work" | T-101 T1 p27 | recorded |

**`rms_granularity` 19.0 is now corroborated rather than asserted — and deliberately not
changed.** For a spatially white grain field σ²·A = W(0), so through this file's own 48 µm
aperture (A = π·24² = 1809.6 µm²) the measured 0.62 µm² gives **σ×1000 = 18.5**. That
conversion is worth something only because the same table carries three films whose RMS is
published elsewhere, so it can be checked rather than trusted: Pan F 0.10 → 7.4, Plus-X
0.14 → **8.8** (published 10 at net 1.0), Tri-X 0.555 → **17.5** (published 17). All three
land on or just below their net-1.0 values, which is the correct direction because grain rises
with density. ⚠ **And that is exactly why 18.5 does not replace 19.0:** the BBC measurement
sits at D 0.48 above base while this field is defined at **net 1.0**. Swapping it in would
trade a confirmed estimate at the right density for a measurement at the wrong one.

**✅ `clump_um` and `clump_gain` — RESOLVED 2026-08-24 by tracing Fig. 8.** The profile stored
26.0 µm and a clumping gain of 1.65; the measured spectrum fits **1.900 µm and 0.000**.

| | clump_um | clump_gain | rms against the 268 traced points |
| --- | --- | --- | --- |
| stored estimate | 26.0 µm | 1.65 | **0.862 µm²** |
| **fitted to Fig. 8** | **1.900 µm** | **0.000** | **0.0018 µm²** |

The old pair was not slightly wrong: it predicts W(20)/W(0) = 0.016 where the measurement is
0.985, because it put the modelled rolloff at 19 c/mm against a measured 263.

**Why the trace is trusted, on a 1-bit scan of a 1964 print.** Three checks, none of which set a
parameter: the 13 horizontal gridlines land on a uniform ladder terminating at exactly 0.000 and
0.701, fixing the y calibration; **W(60)/W(0) came out 0.900** against T-101 p38's printed
statement that the HPS spectrum "falls by only about 10 %" over 0–60 c/mm; and the traced W(0) is
0.610 µm² against Table I's printed **0.62**.

⚠ **The page has ~0.8° of skew and broken gridlines**, so neither pixel-count nor run-length frame
detection works — the frame was recovered by fitting the bottom axis per column. Worth knowing
before the next scanned figure.

⚠ **`clump_gain` 0.000 is a measurement, not a default.** A free two-parameter fit drives it to
zero and the one-parameter fit is identical — the same signature the MTF carrier showed in C2.
T-101 p38 states it independently in words: grain correlation is *"substantially confined to about
plus or minus one equivalent grain diameter"*, with only small components outside, and the measured
autocorrelation even undershoots slightly negative. A low-frequency lobe **is** long-range
correlation, and the document says there is none.

⚠ **A units trap that nearly stored 2.69 µm.** `film_sim.grain_shape` is an **amplitude** transfer —
`make_grain_field` multiplies the noise FFT by it and `grain_reference_energy` squares it — so a
datasheet Wiener spectrum must be fitted as (h/h₀)². Its own docstring claimed it was already a
power spectrum. Fitting that reading gives 2.69 µm, exactly √2 out. The code was always
self-consistent; only the comment lied, which is the worst case because nothing fails. **Docstring
corrected in the same pass.**

**What changed in the render, and what did not.** `rms_granularity` stays 19.0 and the loudness
stays with it: `grain_reference_energy` renormalises the field, so this edits **texture, not
level**. The 48 µm measuring aperture is 1/e at 13.3 c/mm and dominates the visible band, so at
48 px/mm and above both parameter sets put >96 % of aperture-weighted energy in band. Measured
correlation length falls 41.7 µm → 20.8 µm at 48 px/mm; at 24 px/mm the in-band fraction drops
86 % → 56 %, so a 2K-scale render of this stock gets finer **and** quieter.

⚠ **Scope held to HPS (method rule 18) — FOR ONE DAY.** The same documents measure five more
emulsions and the field runs 3.2–40 µm across the file (median 13), so if this is systematic the
error is systematic — but one measured emulsion is not a class. **Tri-X and Plus-X were attempted
from Fig. 8 in this pass and both were rejected, not adopted:** the Tri-X tracker re-followed the
HPS curve (caught because its fitted W(0) returned 0.611 against the printed 0.555) and Plus-X
kept only 28 points after gridline masking, leaving the rolloff at the search floor. **The family
still needs T-101 Fig. 18**, which plots all six emulsions on its own axes and out to 600 c/mm.

### 2026-08-24 — the family question, answered, and it was not answered by a curve

**T-101 Fig. 18 was digitised, and then not used for the stored numbers.** The plate was cracked
(method below, because three earlier attempts produced *plausible but wrong* results and that is
worth recording), it validates against three independent printed quantities, and then **Table 2 on
p28 turned out to print the measured equivalent grain diameter of all six emulsions outright** —
so the adopted values need no traced curve at all. This is the sequence in which the work
actually happened, and the lesson is the older one restated: **read every table before tracing any
curve.** Table 2 had been cited on four profiles since 2026-08-23 for its gamma and its
signal-to-noise column; its last column was the answer to the question Fig. 18 was traced for.

**What is stored, and from what.** `D_eq = 4·√(2·ln(1/0.39))/π · clump_um = 1.7473 · clump_um`,
derived from the file's own Gaussian carrier and the report's definition of equivalent grain
diameter (full width of the normalised autocorrelation at ordinate 0.39, p38), and verified
numerically by Hankel-transforming that carrier.

| emulsion | T-101 Table 2 `D_eq` | `clump_um` stored | was | Fig. 18 trace (not stored) |
|---|---|---|---|---|
| HPS | 2.5 µm | **1.431** | 1.90 (M54 Fig. 8 fit) | 1.638 |
| Tri-X 5223 | 2.2 µm | **1.259** | *new profile* | 1.454 |
| Plus-X 4231/5231 | 1.45 µm | **0.830** | 11.0 | 0.867 |
| Pan F | 1.5 µm | **0.859** | 5.0 | 0.613 / 1.168 ⚠ |
| 8374 | 1.2 µm | **0.687** | *new profile* | 0.606 |
| 5302 | 1.03 µm | **0.589** | *new print stock* | 0.543 |

⚠ **Every one of those is an UPPER BOUND, and the report says so:** p38 states the printed
diameters are "expected to be greater than the true values" because the instrumental weighting of
the measuring microscope is uncorrected. It does not state that aperture, so no correction is
possible. The traced fits sit **+4 % to +15 %** above the printed values on the three stocks whose
spectra are well determined — i.e. looser bounds still, in the same direction, which is what a
blurred autocorrelation predicts.

⚠ **Pan F is why the printed column won.** Its spectrum falls only 18 % across the range where the
plate resolves it from its neighbours, so clump and shape trade off freely: the same 132 traced
points fit **0.613** with the Gaussian carrier and **1.168** with a free exponent (n = 4.11) at
equal residual. A number that moves by 1.9× depending on a shape you are not measuring is not a
measurement. 8374 and 5302 have the same problem to a lesser degree.

**The verdict on the field.** HPS was **not** special. All six land in 0.59–1.43 µm against
3.2–40 µm stored across the file — the stored column is systematically an order of magnitude
high, and `f_hi = 500/clump_um` says what that means: a stored 19 µm puts the grain rolloff at
**26 c/mm**, where Fig. 18 shows Tri-X still at half power at **290 c/mm**. ⚠ **The remaining 155
stocks were NOT touched.** Six 1963 B&W emulsions do not license rewriting colour negative and
reversal stocks; what is now on record is the direction of the error, not a licence to apply it.

**Three independent validations of the digitisation**, none of which set any parameter:

1. **W(0) against a different document.** Traced W(0) for HPS is 0.617 µm² against M54's printed
   0.62 (0.5 %), and for Tri-X 0.552 against M54's printed 0.555 (0.5 %) — different instrument,
   different year.
2. **Table 4's printed granularity ladder**, 5302 as unity over 0–40 c/mm. Granularity goes as
   √W(0) over a band this flat, and the traced spectra reproduce the printed ladder: HPS
   3.94 vs **3.9**, Tri-X 3.72 vs **3.5**, Pan F 1.98 vs **1.9**, Plus-X 1.86 vs **1.8**, 8374
   1.37 vs **1.3**.
3. **`clump_gain` fits to exactly 0.000 on all six, independently** — the same result the HPS fit
   gave from a different figure, and what p38 states in words ("substantially confined to about
   plus or minus one equivalent grain diameter").

**Why the plate needed real work.** Recorded because the failure modes were silent, not loud:

* **The page is bowed, not skewed.** One gridline sits at y 325 near the left frame and 310 near
  the right; the W = 0 line moves **97 px** across the plate. A whole-row fill test therefore
  cannot remove the grid at all, and a single global y→W map carries up to 0.0024 of W — **6 % of
  the 5302 curve's own W(0)**. Fixed by tracking all 29 horizontal and 17 vertical lines
  independently.
* **Two of the 29 lines are untrackable, and one of them ate an emulsion.** W = 0.075 runs almost
  exactly along the 8374 curve, so the tracker followed the *curve*, wandered 83 px, and its
  removal band deleted 8374 while leaving the real gridline in place — which is why 8374 first
  looked absent from the plate. W = 0.000 merges with the converged bundle. A trimmed quadratic
  fit across the ladder index repairs both and reproduces every clean line to 1.7 px.
* **Column tracing is impossible here.** HPS and Tri-X reach ~11 y-px per x-px. Replaced with an
  arc-length tangent walker with a 0.57 °/px turn limit; **without the turn limit Pan F and 5302
  both dive down residual vertical gridline stubs to W = 0**, which is the exact shape of the
  earlier wrong results.
* **Separation is by DASH PERIOD**, measured off the six legend swatches at 600 dpi: HPS even
  38–52 px ink / 8–14 gap; Tri-X long 132–142 alternating with short 14–18; Plus-X short 22–30;
  Pan F long 80–124; 8374 long ~133 with **two** shorts between; 5302 solid. Grid removal cuts
  dashes, so any run touching a gridline is discarded before classifying — and merging across a
  removal band has to be done carefully, because over-merging turns Tri-X's short+long pair into
  one false long run.
* **HPS and Tri-X cross, at f ≈ 130**, and the crossing is decided by **two methods that agree**:
  the dash signature on each branch above f = 155 (lower branch 42,39,42,43,52,56,53,61,44,55 =
  HPS; upper 95,131,84,58,79,154,93 with 13–16 px shorts = Tri-X), and extrapolation of the
  f < 105 fits (HPS predicts 0.114 at f = 270, Tri-X 0.176; observed 0.132 and 0.165).
  **f ∈ [105, 160] is dropped for both**, not interpolated.

### 2026-08-25d — a validation pass that corrected yesterday's own work

**Two adopted claims checked, three queue items closed (C17, C20, and 5248's half of B1), and one
finding that supersedes a decision made hours earlier.**

**1. `spectral_vector.py` cross-validated against an independently adopted set.**
`KODAK_VISION2_200T_5217` already carried a spectral set from the **2026-08-02 raster batch** — a
different image, a different method, a different author. Re-deriving it from the same sheet's
**vector** paths is the only cross-validation this corpus permits, and it holds:

| layer | rms agreement | peak (raster) | peak (vector) |
|---|---|---|---|
| red | 0.109 dec | 650 nm | 640 nm |
| green | 0.091 dec | 550 nm | 540 nm |
| blue | **0.049 dec** | 470 nm | 470 nm |

Inside the reading error of a printed plot, peaks within one 10 nm grid step. **Neither side is
corrected** — a wash is not a reason to churn adopted data. Both methods are now credible on
evidence rather than on assertion.

⚠ **AND IT IMMEDIATELY CORRECTED 2026-08-25c.** That section claimed 5201's blue layer "peaks at
470 nm where its siblings peak at 410–420", on the strength of comparing **one** sibling. Sweeping
every 31-sample Kodak cine stock shows the blue peak splits **6 / 4**:

* **470 nm** — 5201, **5217**, 5205, 5203, 5274, 5246
* **410–440 nm** — 5218 (420), 5279 (420), 5219 (410), 5213 (440)

470 is the **majority**, and 5201 agrees with 5217 exactly. The larger blue residual in the
cross-check lies in the 480–500 nm cliff, not at the peak. Corrected in the profile comment, the
source string and the `verify.py` guard, which now asserts the split itself so neither group can be
"harmonised" toward the other.

⚠ **KNOWN LIMIT, and it bounds the tool honestly:** a corpus sweep finds a rotated `LOG SENSITIVITY`
caption on **5 pages**, against **24** pages carrying a readable dye panel. The ink rule generalises;
the **panel finder does not yet**. Measured failures: `Ektachrome_100d.pdf` p4 (5285) — the caption
sits inside a decorative outer box whose x0 is *left* of the label, so the "frame right of the label"
rule rejects it and the real plot frame is not a separate path; `2383` p6 — only 2 sensitivity ticks
against the frame. Most other sheets draw the caption as **outlined vector art**, so there is no
rotated text to find at all. Neither failure blocks anything (5285's set is already adopted, 2383 is
a print stock).

**2. THE SENSITIVITY CRITERION IS NOW WORSE THAN UNSOURCED — IT IS CONTRADICTED IN VALUE.**
2026-08-25c recorded that no sheet prints the density behind "reciprocal of exposure (erg/cm²)
required to produce specified density". A full sweep of every short Kodak sheet in the corpus for a
printed criterion found **five that do print one**:

| sheet | printed criterion |
|---|---|
| `5246.pdf` p5 | **Density: 0.4 above D-min**, Densitometry: Status M, Effective Exposure: .013 sec |
| `5274.pdf` p4 | **0.4 above D-min** |
| `V200T.pdf` p4 | **0.4 above D-min** |
| `KODAK-VISION3-2254-technical-information.pdf` p4 | 1.0 above D-min *(intermediate/DI film — different product class)* |
| `bringing enhanced performance to the digital workflow.pdf` p2 | 1.0 above D-min |

⚠ **CORRECTED 2026-08-26 — AND THE CORRECTION REVERSES THE HEADLINE.** This paragraph read "NOT ONE
SHEET IN THE CORPUS PRINTS 0.2 … the 0.2 appears nowhere … supplied for precisely the cases with no
evidence for it." **That was false.** A full-corpus regex sweep found **three files that print it**:

| file | page | printed |
|---|---|---|
| `5205t.pdf` | p4 | **`D=0.2>D-min`** |
| `KODAK VISION2 250D Color Negative Film 5205.pdf` | p4 | **`D=0.2>D-min`** |
| `5218-Vision2-500T-H-1-5218t.pdf` | p4 | **`D=0.2>D-min`** |

On each, the string sits in the **Spectral Sensitivity** panel's own caption block, directly beneath
`Densitometry: Status M` and beside `Effective exposure` and `Process: ECN-2`. It is unmistakably
that panel's density criterion.

⚠ **WHY IT WAS MISSED, because the mechanism generalises.** The 2026-08-25c sweep looked for caption
text **inside the plot frame** — which is where 5222 and 7239 put it. The VISION2 layout puts the
block **below** the frame. A scan that assumes one layout finds one layout, and "not printed" was
really "not printed where I looked". Same failure as the F-125 outlined-vector-art case, reached from
the opposite direction: there the text was invisible to `get_text()`, here it was visible and outside
the search box.

**So the 16 split three ways, not two:**

| group | count | status |
|---|---|---|
| `KODAK_VISION2_250D_5205`, `KODAK_VISION2_500T_5218` | **2** | ✅ **SOURCED** — their own sheets print `D=0.2>D-min` |
| `5217`, `5203`, `5207`, `5213`, `5219` (Kodak cine) | **5** | family inference — same product line, sheet series and era as the two above, which is now a documented anchor *inside* the family rather than an invention |
| `EKTAR 100`, `GOLD 100/200`, `PORTRA 100T/160/400/800`, `ULTRAMAX 400/800` | **9** | ⚠ **THE LIVE GAP.** STILL films, a different product line in different publications; nothing in this corpus supports or refutes 0.2 for them |

**The 0.2 is a real printed Kodak convention.** What remains unexamined is whether it carries across
from the VISION2 cine line to the nine still films — which is a smaller and much better-defined
question than the one this section asked yesterday.

⚠ **NOTHING WAS CHANGED.** Rewriting a provenance claim on 16 profiles is an owner decision. The
counts are pinned in `verify.py` so the inconsistency stays visible instead of being absorbed.
**This is the one decision waiting.** Options: (a) correct the 16 to 0.4 on the strength of three
sheets of the same maker, process and product class; (b) change them to an explicitly unnamed
criterion, as 5201 now carries; (c) leave them and keep the conflict recorded. Best next move for
settling it outright: Kodak publication **H-1** *Image Structure*, cited by name on every one of
these sheets and absent from the corpus.

**3. Queue C17 closed — the one-sided sub-pixel gate.** `AlgoDirCoupler.hpp` has always gated BOTH
coupler components below `ALGO_COUPLER_MIN_SIGMA_PX` = 0.25 px; the Python reference had **no gate**,
so below that scale one renderer ran the stage and the other did not. `apply_dir_couplers` now
carries the same gate at the same threshold. **The threshold was adopted, not chosen** — taking the
shipped C++ constant keeps this a pure parity fix with no fidelity judgement folded in. The
crossovers are not exotic: the long term switches off below **3.1 px/mm** (`EASTMAN_5247_1974`,
radius 80 µm) and the edge term below **27.8 px/mm** (`KODACHROME_64`, edge 9 µm) — a 35 mm frame
about 670 px wide. Parity unchanged at worst **5.335e-05** over 5 stocks × 2 fields × 5760 values.

⚠ **C16 IS STILL OPEN AND IS A DIFFERENT QUESTION.** The two blurs remain different *forms* —
analytic Gaussian transfer in Python, truncated separable spatial kernel in C++ — agreeing to 6e-5
only above ~1.2 px and diverging to 1.5e-1 at 0.4 px. Stored `edge_um` of 9–13 µm is 0.36–0.60 px at
40 px/mm: **inside that divergent band and above the gate**. So what remains is the shared
threshold's **value** (0.25 px versus the ~1.0 px where the forms converge), and raising it changes
every render.

**4. Queue C20 closed — a guard that could not fail.** `verify.py`'s "interimage leaves a neutral
untouched" rendered **0.18**, which is the mid-grey anchor the correction is referenced to and the
one point where every `(D_j − d_ref)` is zero. The guard was true by construction for any interimage
matrix. Renamed to what it tests, and a second guard now pins the off-anchor movement as intended
behaviour: on `KODAK_PORTRA_400`, grey 0.45 moves **15.9/255** and grey 0.06 moves **6.5/255** with
the stage disabled — the mechanism, not a leak, since white-light gamma below separation gamma is
the patent's own metric. `InterimageSpec`'s docstring is qualified to match: the correction vanishes
**at the anchor**, not on neutrals in general.

**5. A documentation audit, because two of the errors above were in prose, not in data.** Every
count in `gen_active_profiles.py`'s hardcoded census was checked against the live database. Four
were wrong: ISO 6 **27 → 51**, ISO 5800 **34 → 58**, ISO 2240 **13 → 17**, manufacturer EI
**15 → 34**. All four are now derived from the database, for the same reason the placeholder count
was. Also corrected there: "7 curves on 3 sheets traced" → **26 on 12**; a claim that
`ReciprocitySpec` "is still read by no renderer (queue C8)" — C8 closed 2026-08-23; "39 raster
granularity pages are on disk and unread"; and "all 395 documents in `PDF/PROFILES`" against a
measured 448 PDFs / 559 files. ⚠ `gen_film_curves_md.py`'s `QUEUED_PLOT_ON_FILE` set had been
**empty since 2026-08-02**, which made the report print "no plot in archive (text/table data)" for
five stocks whose plots are in the archive *with page numbers listed in §4.1 of this file*. An empty
hand-maintained set does not mean "no queue" — it means nobody refilled it, and the report was
making the stronger claim on its behalf.

### 2026-08-25c — H-1-5201's last two panels, and a criterion nobody printed

**Queue items C9, C10 and C12, all closed against one sheet already on disk.** No acquisition;
this was extractor work plus one provenance correction.

**The dye panel (C9) — and the recorded reason it had failed was wrong.** C9 said
`dye_density.py`'s family classifier was "built for 3 dye traces or 3 dyes + neutral, not 3 +
neutral + dmin". It never was: family B takes any three of however many curves it is offered, so
two extra traces cost it nothing. What actually happened is that **the cyan trace never reached the
classifier**. Kodak draws it as two overprinted paths — yellow under magenta, making red on the
page — of **7 segments each**, and `extract`'s `n < 8` segment filter dropped both. With nothing
left in the 615–700 nm band no triple could pass the band test, and the sheet reported "no curve
set matched either normalisation family" — a true statement about a curve list that was missing
the curve. ⚠ **A diagnosis recorded in the queue is not evidence.** This one survived a fortnight
because it was plausible and nobody re-derived it.

**What replaced it: identify traces by INK.** Kodak's H-1 brochures use a rule that is physical
rather than decorative — *each trace is drawn in the colour of light it concerns*. The yellow dye,
which absorbs blue, is drawn in BLUE ink; magenta in GREEN; and cyan, which absorbs red, in RED —
not one of the four process inks, so Kodak overprints **yellow under magenta**. The mapping was
read off the panel's own legend swatches (green on "Magenta Dye", amber on "Cyan Dye") and the
resulting traces peak at 450 / 540 / 680 nm, **identical to 5217 and 5218**.

**A new validator, and it is why this is tier 1.** Family A identifies its quartet by
`neutral = C+M+Y`, which cannot hold when the dyes are peak-normalised and the neutral is not. The
generalisation that does hold, and that nothing is fitted to produce, is

> `Neutral − Dmin = k_c·C + k_m·M + k_y·Y` with the three coefficients **equal**, because equal
> contributions are what make the result a visual NEUTRAL.

Unconstrained least squares gives **0.628 / 0.604 / 0.595** — a 5.4 % spread on three free
numbers — at rms **0.019 D**. Drop the Dmin term and the fit is 4.5× worse (rms 0.085) with the
coefficients scattered over 0.86–1.61, which is what identifies *which* dark trace is which.

**The sensitivity panel (C10) — the first VECTOR-traced spectral set in the database.** New script
`spectral_vector.py`, registered in `build.py`'s audit stage. Same ink rule; assignment then checked
three ways that are *not* the ink: the legend swatches, the absorption bands (peaks 470 / 540 /
650 nm, ascending), and the independently-adopted 5217/5218 sets (red and green agree to rms
0.05–0.14 decades).

⚠ **5201's BLUE LAYER PEAKS AT 470 nm, WHERE ITS SIBLINGS PEAK AT 410–420.** That is the one
disagreement in the cross-check (blue rms 0.24–0.42 decades) and it is **printed**: a narrow cusp
just above log S 2.0 at 470, higher than the 445 nm bump, then a cliff to zero by 500 — confirmed
on a 26× render before adoption. The peak normalisation anchors on that cusp, which is why the
400–460 plateau reads −0.21 to −0.11 here and flat elsewhere.

⚠ **AND THE CRITERION IS NOT PRINTED ON ANY OF THESE SHEETS.** 5201's footnote reads, in full,
*"Sensitivity = reciprocal of exposure (erg/cm²) required to produce specified density"* — **it
does not say which density.** The three sets already in the database (5218, 5217, 5219) carry
`criterion="log_reciprocal_erg_cm2_D0.2_above_dmin"`; checking their sources, 5218 and 5217 print
the same unspecified wording and **5219's footnote is not in its text layer at all**. So the
"D 0.2 above dmin" half is printed on none of them. Owner decision: 5201 stores
`log_reciprocal_erg_cm2_specified_density`, the older three are **left alone**, and the conflict is
recorded here and in a two-way `verify.py` guard rather than propagated or retro-fixed (method rule
4). **Still open: what density Kodak actually meant.** Best next move — Kodak publication **H-1**
("KODAK Motion Picture Film", *Image Structure*), which these sheets cite by name and which is not
in the corpus.

⚠ **THIS ONE CHANGES A RENDER.** The dye set is inert (schema v7), but spectral sensitivity is not:
a stock carrying it takes `spectral_balance_gains()` instead of the 600/550/450 nm proxy. 5201's
measured red layer peaks at **650** nm, so tungsten light drives it harder than the proxy assumed —
**+0.28 stop of red gain at 3200 K**, −0.17 at 10000 K, green unchanged as the anchor. Asserted in
both size and direction.

**The tier bug (C12) was three times the size the queue said.** C12 was filed against two
profiles. A sweep for `\[T[123]/T[123]\]` found **six** resolving to tier 3 on
`fitted_from="analogy"`: the three VISION2 negatives (5218, 5217, 5205) and the three VISION
negatives (5279, 5274, 5246). Every one owns its own Kodak sheet, **four have a σ(D) shape traced
from it**, and in all six the T3 half is the same single scalar — `rms_granularity`, because from
VISION onward Kodak prints granularity CURVES and no rms number. All six moved to tier 1 (owner
approved), matching the two precedents already in `_UNTAGGED_TIER`. ⚠ **The mechanism was closed by
a class guard, not by loosening the regex:** a mixed tag must now appear in `_UNTAGGED_TIER`, and
may not resolve to 3, or `verify.py` fails. The strict regex is the feature — it forces a decision
on every future mixed tag instead of quietly picking a number.

### 2026-08-25b — the first measured B&W σ(D), and it reverses a stored sign

**Source: Kodak 7266 technical-information sheet, page 3, panel "rms Granularity Curve"** —
subtitled *"Granularity vs. Density (0-3 scale) / Reversal Process"*, read with a
microdensitometer at 48 µm as the sheet states. Traced off the **embedded image at its native
770×748**; re-rendering the page adds nothing because the source raster caps at ~246 ppi.

**Why this panel and not the sheet's other one.** It carries DENSITY *and* GRAIN against **one
shared log-exposure axis**, with a logarithmic Granularity SIGMA D scale on the right — so the
pairing happens inside a single calibration. The sheet's Characteristic Curve panel disagrees with
it about the same film (log E −4.0…+2.0 against 0.0…3.0; Dmax **2.57** against **3.20**) and the
sheet itself warns *"Sensitometric and Diffuse RMS Granularity curves are produced on different
equipment. A slight variation in curve shape may be noticed."* Pairing across the two panels would
have mixed two calibrations silently.

**Calibration is over-determined.** The log σ_D axis was fixed from its 0.10 and 0.001 labels alone
(163.5 px/decade), then **all ten intermediate printed labels reproduce to ≤ 1.5 px**: 0.05, 0.04,
0.03, 0.02, 0.01, 0.006, 0.005, 0.004, 0.003, 0.002. Density from 18 inward ticks, 3.4→0.0 by 0.2.
Log E from inward ticks at exactly 0.0 / 1.0 / 2.0 / 3.0.

⚠ **The two curves cross and swap vertical order at log E 1.65**, so no upper/lower rule works.
They are separated by the one asymmetry that holds everywhere — density is **solid** (present in
every column, smooth), grain is **dashed** — so density is walked with slope prediction and
whatever run is left over in a column is grain. Verified by coloured overlay: both followed
correctly through the crossing.

⚠ **Only the well-conditioned range is used — 30 of 52 paired points, D 0.352–3.089.** The density
curve is flat at both ends (|dD/dlogE| < 0.5), and there D barely moves while σ keeps changing.
That is the **multivalued-toe trap already recorded on the VISION3 sheets, mirrored to the dense
end** because this is a reversal stock. An apparent interior peak of **2.93×** at D 3.16 lies inside
the discarded zone and is **not stored**.

**What is stored, and what it replaces:**

```
ANCHOR              value   at D     was (estimate)
sigma_shape_toe     0.262   0.352        0.70
sigma_shape_mid     1.000   1.000        1.00
sigma_shape_dmax    2.829   3.089        0.50
```

⚠ **The sign is reversed.** The estimate said grain *falls* to half at dmax; the measurement says it
*rises* to **2.83×** — a factor **5.7** at the dense end, with the toe 2.7× lower than assumed. Over
the usable range **σ_D ∝ D^1.078** (rms 0.038 decades), very nearly linear in density.

Physically the measurement is the sensible one: density is developed silver either way, so σ tracks
the silver. The old triple came from the reasoning that *a slide's densest regions received the
least exposure* — true about exposure, and irrelevant to grain.

⚠ **rms_granularity 10.0 is NOT replaced.** The panel reads σ_D 0.0177 at D 1.0 and 0.0223 at this
file's NET-1.0 convention (D = dmin + 1 = 1.25), i.e. **22.3 against the stored 10.0** — but the
sheet prints *"Note: This curve represents granularity based on modified measuring techniques"*, so
its absolute level is not the standard diffuse RMS. **Shape grounded, level not.**

⚠ **Scope held to this stock (method rule 18).** 35 of the 36 reversal stocks share the identical
0.7/1.0/0.5 estimate — 13 monochrome, eight of those Polaroid peel-apart films whose process is not
this one at all. One measured sample is not a class. What would settle it: a second reversal sheet
carrying this same granularity-vs-density panel.

⚠ **AND IT DOES NOT CLOSE THE σ(D) QUESTION.** This is **reversal, not negative**. The 68
monochrome NEGATIVE stocks still rest on Mees's four unnamed emulsions. Two sheets were checked and
rejected the same day: `EASTMAN-DOUBLE-X-technical-information.pdf` and `5231-PLUS-X.pdf` both print
only a **single** diffuse RMS number (14 and 10 respectively) at net density 1.0 — no curve. Still
wanted: a granularity-versus-density curve for a **named B&W negative**, reaching D ≈ 2. Kodak
publication **H-845** (*The Essential Reference Guide for Filmmakers*, named on the Double-X sheet
as the source for image-structure data) is the best remaining lead in the corpus.

⚠ **A typo in the source**, recorded so it is not propagated: the MTF panel prints the process as
60 s at **68F (24C)** while the Characteristic Curve panel prints 60 s at **76F (24C)**. 68F and 76F
are not both 24C.

### 2026-08-25 — T-101 Figs. 20/21/23/24/26, and a retraction

**The headline is a correction to this project's own reading, not a new number.** Fig. 26 was
extracted cleanly and then found to be unusable for the purpose it was extracted for. Recorded in
full because the mistake is a general one about conventions, and because a later pass would
otherwise re-derive it.

**What Fig. 26 gives, and how well.** Log-log, full labelled grid, five explicit ✕ data markers:

```
log10(t̄/σ) = -0.6648 · log10(D) - 0.1738      1039 columns, rms 0.0063 decades
```

Self-validating twice, on quantities the fit never sees. §B.2 prints the five samples' mean
transmissions (0.74, 0.59, 0.37, 0.22, 0.07), so their densities are known before tracing — four
markers land within **2.2 %** (the fifth is clipped by a gridline). And **Fig. 21** plots the same
quantity on *linear* axes: its five markers give exponent **0.668** against the log-log fit's
**0.665**.

⚠ **And it still cannot be converted to σ_D.** T-101 §2 defines its σ from a two-level model —
grains "uniformly opaque" with "infinitely sharp edges", so t(x,y) takes only the eigenvalues 0
and 1:

```
σ      = √(t̄(1−t̄))            eq. (3)
t̄/σ    = √(t̄/(1−t̄))           eq. (4)   ← the dashed LOWER LIMIT drawn on Fig. 26
```

The report states this limit is approached *"as the scanning aperture becomes vanishingly small"*
and is *"independent of the size and distribution of the grains"*. It is **pinhole, resolved-grain
coverage statistics**. The measured fractional fluctuation runs:

```
 t̄      D    R_meas  R_limit  meas/limit   σ_t/t̄
0.74  0.131   2.591   1.687      1.54      0.39
0.59  0.229   1.779   1.200      1.48      0.56
0.37  0.432   1.214   0.766      1.58      0.82
0.22  0.658   0.863   0.531      1.62      1.16
0.07  1.155   0.609   0.274      2.22      1.64
```

σ_t/t̄ reaches **164 %**. The linearisation σ_D = 0.4343·σ_t/t̄ requires σ_t/t̄ ≪ 1 and is invalid
across the whole plate. **A reading of this figure reported mid-session as "σ_D = 0.648·D^0.665",
together with an "instrument-corrected 0.771", is WITHDRAWN.** Both were built on that step.

✅ **Consequence: the Mees conflict never existed.** `mees_granularity.py`'s Fig. 302 is the
Goetz–Gould constant on a trace evaluator at a fixed densitometer aperture — grains unresolved,
many per aperture, Selwyn regime — and this file's `rms_granularity` is defined at 48 µm, which is
*that* regime. T-101 Fig. 26 is the opposite limit. The apparent disagreement about whether B&W
silver-negative grain turns over at high density was an artefact of the bad conversion. Nothing to
record as a conflict; the σ(D) question is exactly where it was before.

**What does survive from Fig. 26**, as a finite-aperture result rather than a film property: the
measurement sits a near-constant **1.48–1.62×** above the pinhole limit on four of five samples,
breaking to **2.22×** at the densest.

### The one thing that was adopted: grain size depends on DEVELOPMENT

**T-101 Table 3 (journal p35) is printed data and needed no tracing.** It tabulates equivalent
grain diameter against point gamma at two densities for Pan F and Tri-X, and its own last column
normalises by √(point gamma) — the table exists to demonstrate the dependence.

```
film     D    devγ   ptγ   D_eq   printed  recomputed
Tri-X  0.23   0.56  0.45   2.40    3.58     3.578
Tri-X  0.23   0.94  0.65   2.72    3.40     3.374
Tri-X  0.54   0.56  0.56   2.12    2.83     2.833
Tri-X  0.54   0.94  0.94   2.64    2.73     2.723
Pan F  0.30   0.63  0.63   1.23    1.55     1.550
Pan F  0.30   1.10  0.88   1.44    1.53     1.535
Pan F  0.70   0.63  0.63   1.10    1.39     1.386
Pan F  0.70   1.10  1.10   1.41    1.34     1.344
```

The last column reproduces to three decimals, so the reading is exact. Refitting the eight rows:

**D_eq ∝ γ^n**, n = **0.452** (Pan F, rms 0.0035 µm), **0.396** (Tri-X), **0.425** (pooled) — the
printed √ is marginally steep, over-predicting the four measured pairs by 1–6 %.

✅ **Validated against a number the fit never saw:** the same law at γ 1.0 and D 0.43 predicts
D_eq **1.47 µm** against Table 2's printed **1.5** — 2 %.

**⚠ It exposed a condition mismatch shipped the previous day.** `clump_um` was taken from Table 2's
diameters, measured at *the BBC's* development gamma. Five of six stocks match their stored gamma;
`ILFORD_PAN_F` does not — stored **0.55** (Ilford's ID-11 contrast index) against the BBC's **1.0**.

```
0.859 × (0.55 / 1.00)^0.452 = 0.655 µm      ADOPTED
                     (0.666 on the pooled exponent, 0.637 on a strict √ — a narrow bracket)
```

⚠ `EASTMAN_PLUS_X_5231` (0.68 vs 0.64) is **deliberately not corrected**: the same law makes it a
+2.5 % move to 0.851, far inside the upper-bound caveat those printed diameters already carry.
Moving a number by less than its own stated uncertainty is false precision.

⚠ **The soft spot, stated rather than hidden.** Table 3 distinguishes *development* gamma from
*point* gamma and they diverge (Pan F developed to 1.1 has point gamma 0.88 at D 0.3 but 1.1 at
D 0.7). The stored `gamma` is a whole-curve figure matched to development gamma — the closer of
the two, not the same. Direction and size are solid; the third digit is not.

**And Fig. 21 measures the density dependence too**, on the same emulsion: D_eq 1.726 / 1.572 /
1.484 / 1.402 / 1.384 µm at the five printed transmissions, **−20 % across the tone scale**, with
the t̄ = 0.37 point reading 1.484 against Table 2's printed 1.5 (1.1 %). Fig. 20 shows the same
change in the frequency domain — the low-density spectrum is visibly narrower. The schema stores
one scalar per stock, so every `clump_um` is a **mid-scale representative** and nothing finer.
Both dependences are now recorded in the `GrainSpec` docstring.

**A shape finding, recorded and deliberately not acted on.** Fitting a free exponent,
W = W₀·exp(−2·(f/f_hi)^n), gives **n = 1.80** for HPS, **2.01** for Tri-X, **2.43** for Plus-X and
**2.4–4.1** for the three fine-grain stocks. The file's carrier is fixed at n = 2 — exactly right
for Tri-X, slightly too thin-tailed for HPS, too soft-shouldered for the fine-grain emulsions.
Changing it is a **renderer** change, not a data change, and is not attempted here.

**⚠ Deliberately not used: M54 Figs. 12–16, the displayed-granularity curves.** The author
states at §7.1.1 p16 that they are *computed* from Fig. 8 plus six declared assumptions —
printing onto Kodak 5302, Lamberts' printing response for that stock, the print development
curve of Fig. 10, 5302's own 0.04 µm² grain, a 13 c/mm video bandwidth, and a Gaussian system
aperture 1.5 dB down at 13 c/mm. They describe a 1964 telecine chain, not an emulsion, and
their tone dependence is equation (4)'s **assumed** D^−0.6 law rather than a measurement.
Fig. 9 is Table I redrawn and adds nothing to the printed digits. Fig. 14 contains **no HPS at
all** — it is Plus-X only.
* **⚠ Product-identity trap, verified:** the 1942 Ilford Manual's **"Hypersensitive
  Panchromatic" is a different, slower product**, and its "Hypersensitive Panchromatic
  H.P.3" is HP3, not HPS. The 1942 manual **predates HPS entirely** — do not use it.
* **Reviewed:** all 22+ Ilford PDFs **and** the 1942 Ilford Manual (492 pp, zero hits). ⚠
  That "zero hits" line stood while **two BBC documents in the ILFORD folder measured this
  film directly** — they were added later, but the lesson is that a folder sweep counted PDFs
  by maker and not by content.

#### 1.12a The same two documents measure five more emulsions

Both reports work on a **family of six**, which matters because it clears method rule 18 (no
class estimate from one sample). Citations were added 2026-08-23 to every stock in the file
that the family touches.

| Emulsion in the documents | γ | equiv. grain dia. | Wiener µm² | our profile | note |
| --- | --- | --- | --- | --- | --- |
| Ilford **HPS**, 35 mm | 0.63 | 2.5 µm | 0.62 | `ILFORD_HPS` | §1.12 |
| Kodak **Tri-X type 5223**, 35 mm cine | 0.64 | 2.2 µm | 0.555 | ✅ `EASTMAN_TRI_X_5223` (added 2026-08-24, queue C26) | ⚠ this cell read "**no profile**" until 2026-08-25 while §0.3 of this same file recorded the profile's addition — the footnote-on-another-stock state is over: 5223 is its own profile and `KODAK_TRI_X_400TX` deliberately did NOT inherit its numbers (method rule 18, pinned in `verify.py`) |
| Kodak **Plus-X type 4231**, 35 mm | 0.64 | 1.45 µm | 0.14 | `EASTMAN_PLUS_X_5231` | 4231 is the Estar-base number for the same emulsion; the measured γ 0.64 is the BBC lab's own processing and does **not** replace the stored 0.68 from Kodak's sheet |
| Ilford **Pan F**, 35 mm | 1.0 | 1.5 µm | 0.10 | `ILFORD_PAN_F` | ⚠ **emulsion-identity conflict**, below |
| Kodak **8374**, 16 mm TV recording, blue+UV | 1.0 | 1.2 µm | — | ✅ `KODAK_8374` (added 2026-08-24, queue C26) | γ 1.0 and 1.5 curves in M54 Fig. 13(b). ⚠ cell corrected 2026-08-25 from "no profile". Its `exposure_index` remains an acknowledged invented placeholder — T-101 leaves both speed cells blank, because a recording film was rated against a CRT phosphor |
| Kodak **5302**, 16 mm release positive | 2.4 | 1.03 µm | **0.04 at D 0.5** | ✅ `KODAK_5302` in `PRINT_STOCKS` (added 2026-08-24, queue C26) | it is the UNITY of Table 4's granularity ladder, so every grain number taken from that document is anchored on it. ⚠ cell corrected 2026-08-25 from "no print stock" |

**⚠ `ILFORD_PAN_F` — an emulsion-identity conflict, not a measurement one.** Our profile
carries era 1960s–1980s with rms 5.0; the samples measured here are **1963** Pan F. Relative
to HPS the measurement gives 0.40 while the file stores 0.26 — a 1.5× gap, consistent with
Ilford having re-engineered the emulsion and **not resolvable from these documents**. What
would settle it is an Ilford sheet for either generation.

**⚠ A trap in T-101 Table 4's relative granularity.** Pan F reads 1.9 against Plus-X's 1.8
there despite being four stops slower — because Pan F was developed to γ 1.0 and Plus-X to
0.64. Comparisons across that table are **not at constant development**.

**Also available and not yet harvested**, from the same two documents:

* **Kodak 5302 print stock:** Wiener spectrum **0.04 µm², uniform, at D 0.5 above base**
  (M54 p16). ⚠ **SUPERSEDED 2026-08-24:** this read "No existing entry in `PRINT_STOCKS` has a measured grain figure at all." `KODAK_5302` now carries `grain_clump_um` 0.589 and `grain_rms` 4.7 from exactly this document, so one entry does.
* **A development-gamma scaling law**, T-101 Table 3 p35: equivalent grain diameter ∝
  **√(point gamma)**, holding to 5 % across Tri-X and Pan F at two densities each; and the
  diameter *falls* as density rises at fixed development (Tri-X 2.40 → 2.12 µm from D 0.23 to
  0.54).
* **Measured σ(D) shape**, T-101 Figs. 19/20 (Pan F at five mean transmissions) and Fig. 26
  (mean-signal-to-noise against mean optical density) — the *measured* form of the law M54
  eq. (4) only assumes.
* **Callier quotient against density**, T-101 Fig. 25 p37, on Tri-X 5223: log₁₀Q 0.37 → 0.30
  over D 0.1 → 1.0, i.e. **Q 2.34 → 2.00**, at a stated specular collection angle of
  **0.0016 steradian**. See §1.12b.
* **Grain spectra out to 600 c/mm**, T-101 Figs. 23/24 — four times M54 Fig. 8's range, so
  they actually resolve the rolloff `clump_um` sets.

#### 1.12b ⚠ The C22 Callier gap is closed as a document, and still open as a number

`RESULT_2026-08-23c` listed C22's one remaining gap as *"one densitometer specification stating
a diffuse-versus-specular ratio for a named emulsion"*. **T-101 Fig. 25 is exactly that**, on
Eastman Tri-X 5223, at two development gammas, as a function of density.

It does **not** simply replace the database's `callier_q` 1.3 on monochrome stocks, and the
reason is the C22 thesis restated by the document itself: the collection angle is **0.0016
steradian**, very nearly collimated — the limiting case. A real condenser enlarger or a
directed-source scanner accepts a far wider cone and reads a lower Q. So **2.0–2.34 is the
upper bound** of what a directional reader can see, and 1.3 remains plausible for an ordinary
condenser. The measurement therefore *supports* splitting film-scattering from
reader-directionality rather than collapsing it, and it fixes what `scanner_specular = 1`
should mean physically.

**What remains open:** Q **falls** with density (≈15 % from D 0.1 to 1.0) and varies with
development gamma, and `AlgoCallierFactor` holds it constant. That is now a *measured*
limitation of the model's form rather than an unknown. `callier_q` is unchanged; the
measurement is cited on `KODAK_TRI_X_400TX`.
* **What to ask for:** an Ilford Manual of Photography edition **1955–1965**, or an Ilford
  HPS product leaflet of that period.
* **Likely holders:** HARMAN technology (Ilford's successor, Mobberley); the British
  Journal of Photography Almanac 1955–1965; Science Museum Group / National Science and
  Media Museum (Bradford) Ilford collection.

### 1.13 `GENERIC_BW`, `GENERIC_COLOR` — not gaps

Generic classes, **undocumentable by definition.** Listed for completeness only. No
enquiry should be made for these.

---

## 2. STOCKS WITH PARTIAL DOCUMENTATION — source exists, named parameters still missing

At least one real citation exists; the listed parameters remain unsupported after reading
every relevant document. (What each feeds: curve/gamma → tone; RMS + clump → grain;
resolving/MTF → sharpness; spectral → colour balance; reciprocity → long exposures;
spectral dye density → colour fidelity under a given scan chain.)

| Stock | Documented (source) | Still missing | Reviewed |
|---|---|---|---|
| `FUJICOLOR_A250` ⚠ **moved from §1** | **Fuji MP3-57E, 1980.08** (on disk): EI 250T, 35 mm type **8518** / 16 mm type **8528**, launched 1980, coloured-coupler orange mask, Academy Award of Merit. Spectral curve populated | Spectral dye density, resolving power, dye impurity, reciprocity | The A250 sheet. ⚠ The companion file `A 250.pdf` is **not** this film's datasheet |
| `GEVACHROME_902` ⚠ **moved from §1** | **Verbrugghe, SMPTE Journal** (on disk as `AGFA/Gevachrome902.pdf`): Gevachrome Print Film **T.9.02**, gamma 1.10–1.50 dialled by colour-development time 4–6 min (stated linear), emulsion thickness cut 15 → 11.5 µm | Spectral dye density, resolving power, dye impurity, reciprocity | The paper in full |
| `KONICA_CHROME_CENTURIA_100` ⚠ **moved from §1** | `chrocen100.pdf`: ISO 100, **RMS 11** (48 µm, net D 1.0), 60/140 lp/mm, Dmax ~4.0, full reciprocity table to 64 s (+1 stop, CC10C), E-6/CRK-2 | Spectral dye density, dye impurity | The sheet in full |
| `KONICA_CHROME_R100` ⚠ **moved from §1** | `R100.pdf`: ISO 100 / 32 (80B) / 25 (80A), RMS 11, 50/125 lp/mm, reciprocity cliff already at 1 s (+½ stop, CC5R), CRK-2/E-6 | Spectral dye density, dye impurity | The sheet in full |
| `KONICA_CENTURIA_SUPER_400` ⚠ **partly resolved** | Own sheet `csuper400.pdf` located 2026-08-16: ISO 400/27, triacetate, DX 26-5, MCC/UCC | **The data-table page did not survive text extraction** — the numbers are in the file but not yet read. ⚠ Also recorded: `VX-S400.pdf` is an **adjacent product**, deliberately not back-applied | Front matter read; table page needs another route (render + read) |
| `EASTMAN_5247_1983` ⚠ **new stock 2026-08-18** | TI0835 rev 6-93: EI 125/22 T, ECN-2, acetate + rem-jet; spectral plate TI0835C (6-83) digitised **[T1 vector]**; Chibisov 1988 t.VIII p165 (S 125 GOST, mean gradient 0.50, RMS 5, MTF 0.65/0.32 at 30 mm⁻¹); Sehlin/Kennel SMPTE 7/1985 | **Spectral dye density** (sheet p4 is vector but carries no rotated axis label, no plot frame and no wavelength ticks — needs a visual pass); the **introduction date** of this coating | TI0835 in full |
| `ILFORD_HP3` | 1942 speeds (H&D/Scheiner), curves, γ–time, developer tables (Ilford Manual 1942 pp 74–80, 195); ASA 400 corroboration (Photo-Lab-Index 1979) | Spectral sensitivity, RMS granularity, resolving power, latitude. ⚠ The 1942 curve is **not** the 1950s emulsion | Both books in full |
| `ILFORD_PAN_F`, `ILFORD_FP4`, `ILFORD_HP4` | Speeds, dev matrices, wedge spectrograms, sensitivity ranges 230–670 nm (Photo-Lab-Index 1979; Ilford plate table) | RMS granularity, resolving power, MTF — Ilford does not publish them, historic or modern. ⚠ Product identity: ours are FP4 / Pan F, the modern sheets are FP4 **Plus** / Pan F **Plus** | 22+ Ilford files |
| All 4 `KODAK_VISION3_*` | Full H-1 sheets: EI, balance, reciprocity, base. **5219 gained spectral dye density 2026-08-18** from the *brochure* | **RMS as a printed number** — still curve-only on every sheet, but the curves have now been traced per layer (2026-08-23: 5219 5.92/6.60/17.84, 5207 rms_b 8.92, 5203 rms_b 4.71; **5213 stays on the heuristic**, its sheet drawing one bold band). Resolving power, gamma, Dmin/Dmax scalars. Dye density still missing for 5203/5207/5213 — their TI-sheet plots are **raster** | All VISION3 sheets incl. 2026 revisions |
| `KODAK_VISION2_*`, `KODAK_VISION_*`, `EASTMAN_EXR_*` | Full sheets. **5205, 5245, 5274, 5279, 5293 gained spectral dye density 2026-08-18** (`peak_1.0`, shape only) | RMS number, resolving power (except 5248: 80/160 and 5296: 50/100), gamma, Dmin/Dmax. **Absolute dye density level** — the sheets plot unit-peak normalised curves, so the level is not on the page at all | All sheets; `RESULT_REPAIR.md` pairings |
| `KODAK_TECHNICAL_PAN` | P-255 in full: CI 0.50–2.50 matrix, RMS 5/8, red limit 690 nm, base | **Resolving power** — P-255 prints none. The famous 320+ lp/mm figure is from literature **not in this corpus** and is not used | P-255, all 12 pp |
| `FUJICOLOR_SUPER_F500_8572` | Own sheet: EI, RMS 4.0, reciprocity, filter table; ✅ **added 2026-08-23** — characteristic curves (so **gamma is no longer missing**: 0.502/0.566/0.619), spectral sensitivity (peaks 467/551/648 nm), and sharpness from the CTF panel by Coltman conversion (printed CTF f50 24.79 → **sine f50 20.21 c/mm**), so the "resolving power (CTF graph only)" entry is also retired | Process, base | Both pages |
| `AGFA_VISTA_200` | Own brochure: ISO, RMS 4.3, resolving 130/50, layer, base, latitude, reciprocity | Gamma / curve numbers (curves are vector — extraction queued); spectral tabulation. ⚠ One page carries the whole 100/200/400/800 family — per-product assignment **needs the legend read**, not safely automatable | All 8 pp |
| `FUJI_NEOPAN_1600` | AF3-608E in full: speed, 0.122 mm grey triacetate base, dev matrix, spectral at 5 nm, curve (fitted, Ḡ 0.77) | RMS granularity, resolving power, MTF, reciprocity — **all 4 pages searched, not printed** | AF3-608E |
| `FUJI_NEOPAN_ACROS_100` | AF3-083E: RMS 7, resolving 60/200, base, reciprocity | ⚠ The sheet documents the **120 format**; we model the 135 stock — same emulsion designation, different format. `[C2]` flag stands | AF3-083E |
| `KODACHROME_1938`, `KODACHROME_TYPE_A_1938`, Cheltsov-1958 stocks | Balance K, speed, some γ and resolving (Cheltsov 1958, cited per page) | Curves, granularity, spectral curves — the book prints none | Cheltsov, 250 pp |
| `SVEMA_*`, `TASMA_OCH_45` | Gurlev 1986; **ГОСТ 24876-81**, **ГОСТ 20945-80**; Zhurba 1984; Cheltsov 1958; Zhurba 1990 Tables 2/13 via owner scans | Spectral sensitivity curves; granularity **numbers** (зернистость classes only); MTF for *our* pre-1987 generation | All Soviet sources; 8 spreads of Zhurba 1990 read from owner screenshots; **pp 44–131 still need a local copy** |
| `SVEMA_FOTO_65` | Gurlev 1986 p296 (γ_rec 0.8, D0 0.05, R 110 lin/mm, Δλ 665 nm); ГОСТ 24876-81 table 6 | ⚠ **Its scan-derived values were withdrawn 2026-08-18** — the batch mixes Foto-32 with Foto-65. Absolute base+fog, σ(D), anisotropy all open. See `RESULT_2026-08-18_svema_clean67.md` | Full |
| `SVEMA_FOTO_32`, `SVEMA_FOTO_130` | Gurlev 1986; Chibisov 1988; ГОСТ 24876-81; Zhurba 1990 | ⚠ `base_tint` and `silver_tone` are **undocumented transfers** from `SVEMA_FOTO_65`, whose own values were withdrawn. Flagged in code, awaiting a decision | Full |
| `ORWOCOLOR_NC21`, `ORWO_CHROM_UT18` | **Zhurba 1990 Table 66 (p124, owner scans)** — the only ORWO documentation in the corpus: NC21 100 ед.ГОСТ / 21° DIN / 100 ASA / 5500 K; UT-18 50 / 18° / 50 / 5500 K daylight (stored 4500 K corrected) | Curves, granularity, resolving, spectral, reciprocity — Table 66 gives **speed and balance only** | Zhurba 1990 scans; all 9 ORWO PDFs. **The ORWO Handbuch is the document to obtain** |
| `POLAROID_51`, `POLAROID_52` | Polaroid FDS sheets: speed, resolution | **D-max / D-min / slope — the sheets print none**; our values rest solely on the 1979 trade book | 51fds / 52fds + all 54 Polaroid files |
| `CINESTILL_800T` | Base stock VISION3 5219 fully documented. ✅ **Plus, since 2026-08-27, CineStill's OWN published sensitometric plot** — all three characteristic curves traced from it, 480 points per layer (§7.2c) | ⚠ **THIS ROW WAS WRONG UNTIL 2026-08-27.** It said the rem-jet-removal consequences were "obtainable only by measurement, not documentation". Half of that is now false: the **base density IS documented** — the traced plot gives the orange-mask ladder 0.187 / 0.526 / 0.876, replacing a flat 0.22 / 0.20 / 0.19 estimate, and it agrees with VISION3 5219's own H-1 sheet to 0.06 D. What remains measurement-only is the **halation magnitude and radius in physical units**: the vendor page confirms "red halation glow" but prints no number, so `radii_um (20, 130, 700)` and `gain (1.05, 0.30, 0.10)` are still estimates. Also still absent from every source: rms granularity, MTF/f50, spectral sensitivity (§7.2f) | Whole corpus + the two digitized vendor figures |
| `KODAK_PORTRA_160/400/800` | Predecessor NC/VC generation documented in **E-190** (on disk, deliberately **not** merged — different films) | The 2010s films' own sheets — **not in this corpus** | All 62+ Kodak still files + E-190 |
| `KODAK_EKTAR_100`, `KODAK_GOLD_100/200`, `KODAK_ULTRA_COLOR_*`, `KODAK_ULTRAMAX_800`, `KODAK_EKTAPRESS_PJ400` | Spectral curves (vector-extracted 2026-08-16). ⚠ `e40`–`e44` are **ROYAL GOLD**; `e4026`/`e4029` are **SUPRA** — different films | Essentially everything else. `E-4035` (Ultra Color) is **not on disk** | All Kodak still files |
| `AGFACOLOR_NEG_TYPE_3`, `_TYPE_B_1943`, `AGFACOLOR_NEU_1936`, `GEVACOLOR_1952`, `GEVACOLOR_NEG_682` | Period books/journals (Schmidt/Kochs 1943; SMPTE 1980; Cheltsov 1958) | Granularity, resolving (682 partial), spectral curves. ⚠ `AGFACOLOR_NEG_TYPE_3` has **two Soviet books in conflict** (Чельцов ASA 20–25 vs Иофис ASA 40) and **no Agfa source** — method rule 14 cannot resolve it | All Agfa/Gevaert files |

---

## 3. PARAMETER CLASSES MISSING ACROSS THE WHOLE CORPUS

Corpus-wide, not per-stock.

| Parameter | Status | Notes and what would close it |
|---|---|---|
| **`LayerStack`** (layer order, thicknesses) | Absent for every stock | Cross-sections appear in patents and in SMPTE papers, never in datasheets |
| **Interimage / DIR coupling coefficients** | Absent everywhere | Qualitative prose only; our IIE numbers rest on **US4725529A** plus fits. Distinct from dye impurity below: interimage is a *chemical* coupling during development |
| **Dmin/Dmax scalars, Western stocks** | Curves only | Sole exceptions: Polaroid FDS blocks; **ГОСТ 20945-80** limits |
| **Per-layer spectral sensitivity as tables** | Plots only | Tabulated data exists nowhere. Many plots are **vector** — measured inventory 2026-08-18: see §4 |
| **RMS granularity for colour cine stocks** | Curve-referenced only — but the curves are now being read | Every Kodak cine sheet says "refer to curve"; **101 vector granularity pages** exist (§4). ⚠ **UPDATED 2026-08-23 (C1e): per-layer rms is now traced off those sheets for 11 stocks** — 5219 at 5.92/6.60/17.84, 5207 rms_b 8.92, 5203 rms_b 4.71, greens frozen; 5213 stays on the heuristic because its sheet draws one bold band. The traced ratios **contradict the old stack ladder** (blue 1.30×, red 1.10× of green): nine sheets measure b/g 1.81–2.79 and r/g 0.75–1.05. The ladder is deliberately **not** rescaled — all nine are Kodak cine negatives, and no other class has been measured |
| **Absolute spectral dye density level** | ⚠ **New entry 2026-08-18** | The VISION2/VISION3 sheets plot each dye **normalised to unit peak**, so the absolute level is not on the page. As-printed density: `KODAK_EKTACHROME_100D_5285`, `EASTMAN_EKTACHROME_7239` (`as_printed_visual_neutral_1.0`) and the `KODAK_2383_RELEASE` print stock — 2 film profiles plus 1 print stock. ⚠ This sentence said "only 5285 and 2383" until 2026-08-25 and omitted 7239, adopted 2026-08-18. Closing this needs a densitometric measurement, not a document |
| **Callier coefficient** | Absent everywhere | Populated on **all 160 stocks** from three assumed values — a documented provenance defect (`DATASHEET_VERIFICATION_REPORT.md` Addendum §D) |
| **ILFORD RMS / resolving / MTF** | Absent in all Ilford sources | Ilford does not publish them, historic or modern. Not worth re-searching |
| **σ(D) for a B&W silver negative** | ⚠ **Partly resolved 2026-08-18** | Mees Fig. 302 (printed p866) gives four B&W negative emulsions, **unnamed**. A named-product measurement is still absent. Primary literature to acquire: **Bayer, JOSA 54 (1964)**; **Wilder, JOSA 62 (1972)**; **Trabka, JOSA 63 (1973)** — none on disk |
| **Numeric orange-mask density** | One primary source | **ТУ 6-17-691-88** table 2 (ДС-5М): Dmin behind blue/green/red 0.70–1.05 / 0.25–0.50 / ≤ 0.25. Absent for every Western stock |
| **Dye impurity coefficients** | One primary source | **ТУ 6-17-691-88** table 2 item 6: seven measured D_вр/D_пол ratios for ДС-5М |

---

## 4. IDENTIFIED, EXTRACTABLE, NOT YET EXTRACTED — gaps in our work, not in the corpus

**⚠ The page counts in the previous revision were not reproducible and two documents
disagreed.** Replaced 2026-08-18 with a machine inventory (`plot_inventory.py`, re-runnable,
`--assert` mode, classifier validated against three pages whose answer was already known):

| Plot type | Vector pages | Raster pages | Vector files |
|---|---|---|---|
| Spectral dye density | **191** | 28 | 141 |
| Modulation transfer (MTF) | **199** | 37 | 153 |
| Diffuse rms granularity | **101** | 39 | 86 |
| Characteristic curve | **294** | 73 | 179 |

The MTF figure supersedes both "119 vector MTF pages" (this file's previous revision) and
"~156 vector documents" (`ROADMAP_2026-08-17_fidelity.md` §2.3). The dye figure supersedes
"54 vector pages" **and** a 57-page count made earlier on 2026-08-18 with a title pattern
that missed Kodak's actual heading, "DIFFUSE SPECTRAL DENSITY".

⚠ **A limit of that inventory, stated so it is not over-read:** "no raster image on the
page" is what the classifier calls vector, and that is **not** the same as "a vector plot
is present". `5247.pdf` p4 proves it — vector, 34 paths, zero images, but no plot frame and
no wavelength ticks. Every per-stock match is a **candidate** until the sheet is opened.

⚠ **Per-stock mapping is by 4-digit catalogue code only**, because looser matching failed
three ways (generic words linked one sheet to fifty stocks; shared type designations 500T /
200T / 50D crossed 5219 with 5279 and 5213 with 5217; product words matched the wrong speed
variant). Consequences: **Kodak reused numbers** — `5248` matches both
`EASTMAN_EXR_100T_5248` and `EASTMANCOLOR_5248_1953`, and only opening the sheet resolves
it (it is EXR 100T, March 1999, H-1-7248) — and **only 40 of the 161 film stocks carry a 4-digit code
at all**, so 120 would need assignment by hand. ⚠ The denominator read "163" until 2026-08-25, which
matched no census of this database at any point: 40 + 123 = 163 was internally consistent and
externally wrong.

### 4.1 Still open, with the reason each is not entered

| Target | Source | Why not entered |
|---|---|---|
| `EASTMAN_5247_1983` dye density | `5247.pdf` p4 | Vector, but no rotated axis label, no plot frame > 80×50 pt, no wavelength ticks. **Needs a visual pass** |
| ~~5246, 5248, 5217, 5218, 7239 dye density~~ **5246 only** | own sheets | ⚠ **ROW REWRITTEN 2026-08-25 — three of these were closed on 2026-08-18 and a fourth was never a failure.** 5217, 5218 and 7239 were adopted 2026-08-18 (§0.3.1); the recorded symptom "find no tick labels against the frame" was a defect in `dye_density.py`, not in the sheets. **5248 is not an extraction problem at all:** its panel prints "Typical densities for a midscale neutral subject and D-min." and draws exactly those two traces — no separate dye curves exist on it, so `SpectralDyeDensity.validate()` can never be satisfied. That is the same schema-shape mismatch already recorded for `FUJI_SUPER_F125_8532`, and 5248 is its second instance. **5246 alone remains open**, and the blocker is now named: 7 traces for 5 labels, with the label-nearest Cyan peaking 0.943 against the sheet's own "peak-normalized" claim and two unlabelled traces unaccounted for |
| `KODAK_VISION3_50D/250D/200T` dye density | H-1 TI sheets p4 | **Raster**, not vector. 5219 was rescued from its brochure; check whether 5203/5207/5213 brochures exist |
| ~~`KODAK_TMAX_400` spectral~~ | F-4043 p7 | ✅ **CLOSED — adopted 2026-08-16**, and this row was simply never struck. The profile's `spectral.source` reads "publication F-4043 … p7; PDF vector-path extraction 2026-08-16". Corrected 2026-08-25 |
| ~~`AGFA_APX_25/100/400` spectral~~ | own sheets p2 | ✅ **CLOSED — all three adopted 2026-08-17** from those same p2 panels ("PDF vector-path extraction 2026-08-17"). The `qu`-quad frame problem was solved, not deferred. Corrected 2026-08-25 |
| `AGFA_VISTA_200` gamma | family sheet p8 | One page carries 100/200/400/800 — per-product assignment needs the legend read |
| `ILFORD_HP5_PLUS_400`, `ILFORD_DELTA_3200`, `ILFORD_FP4`, `ILFORD_PAN_F` spectral | Ilford sheets p1 | **Wedge spectrogram with no numeric axis ticks at all** — nothing to calibrate against. Plus the Plus-vs-non-Plus identity caveat |
| `POLAROID_52`, `POLAROID_55_PN_NEG` spectral | fds p3 | ⚠ **HALF CLOSED, row corrected 2026-08-25.** 664 and 667 now carry curves from their own fds sheets; only 52 and 55_PN_NEG remain. Decade-log ordinate, conversion approved, but the axis labels sit differently on each sheet so one matching window still fails these two. **Needs per-sheet windows + visual check** |
| ~~`KODAK_TECHNICAL_PAN` spectral~~ | P-255 **p9** | ✅ **CLOSED 2026-08-31 (queue B3).** The row's page list was a guess and the panel is on **p9 alone**, beside the modulation-transfer curves. 31 samples adopted, criterion `log_reciprocal_erg_cm2_D0.3_above_dmin`, absolute peak 1.03 at the 380 nm grid edge. ⚠ **The "per-plot disambiguation" was not the blocker** — the caption reader's density-criterion test was, and P-255 broke it in a way no earlier sheet had: it splits "Diffuse Density=" and "0.3 above D-min" across two text lines, so the line with the number has no `=` and the line with the `=` has no number. See `spectral_vector.CRIT_RE` |
| `ROLLEI_INFRARED_400`, `KONICA_INFRARED_750` — the **taking filter**, not the curve | Rollei Oct-2005 TDS; Konica IR-750 TDS | ⚠ **NEW 2026-08-29, and it is a gap in the SCHEMA before it is a gap in the corpus.** Both sheets plot the sensitisation **without a filter** — Konica's panel says so in words — and both films are used *with* a deep-red or IR filter, which is what makes their authored red-dominant weights right. Nothing in `SpectralSensitivity` records whether a stored curve is the bare emulsion or the emulsion behind a stated filter, so the database cannot tell the two readings apart and `ROLLEI_INFRARED_400` now derives a near-flat (0.349, 0.315, 0.336) from a curve that peaks at 410 nm. **What is missing is not the sensitisation** — both curves are stored — **but the filter's transmission**, which neither sheet prints; a Wratten 87/89B or Hoya R72 transmission table would close it, and those are standard published data outside this corpus. Queue **C39** |
| `KODAK_ULTRA_COLOR_100UC/400UC` | **E-4035** | **Not on disk.** `e4026`/`e4029` are ROYAL SUPRA / SUPRA — different films |
| `KONICA_CENTURIA_SUPER_400` data table | `csuper400.pdf` | Table page did not survive text extraction; render-and-read it |

### 4.2 Closed brackets (do not re-open without new material)

* **Image-only OCR** — all four items closed. `NewGevacol_Neg_682.pdf` is the Vervoort &
  Stappaerts SMPTE paper 89(9) 1980 pp 650–652, already cited; its RMS table is
  **relative only** (σ_D ∝ 1/√n, no aperture) so it can never yield a 48 µm figure.
  `centuria_pro_400.pdf` is barren marketing **and a different product** (CENTURIA **PRO**
  400). `professional_160.pdf` p4 is the only technical page and **matches no DB stock** — ⚠ and
  2026-08-31 adds the reason it can never be mined even if a stock were found: **all four pages
  extract zero characters**, so there is no caption, no axis label and no legend to calibrate
  against. Closed as unusable, not deferred.
  Konica IMP50 / INF750 were mined for TEXT (63/160 lp/mm; 640–820 nm, peak 750) — ⚠ **and their
  PLOTS were traced 2026-08-31 (queue E3) by `konica_raster.py`**, which is this corpus's first
  adoption from a sheet that is raster end to end. IMPRESA 50's Dmin triple turned out to be a
  family template shared with two other KONICA stocks and wrong in blue by 0.32 D; its MTF f50 is
  64.9 c/mm against an estimated 72; INFRARED 750's gamma is 1.70 at the sheet's own standard
  condition against a stored 0.72 that is below all fifteen printed curves.
* **KODAK DATA BOOK vol 5** — all 346 pages swept. A UK handling manual: **zero RMS
  figures, zero numeric gamma**, resolving power on 8 pages only, and **none of those maps
  onto a held stock without a generation graft**. ⚠ The post-1960 ASA caution stands.
* **Portra NC/VC (E-190)** — a deliberate decision, not a gap. If the NC/VC generation is
  ever wanted it enters as its own stocks; candidates in `next_week_task.md`.

### 4.3 Blocked on materials, not on our work

* **Zhurba 1990 pp 44–131.** Eight spreads (book pp 46–47, 50–53, 64–65, 69, 72–73,
  120–121, 124–125) were supplied as owner screenshots and read on 2026-08-16 — that pass
  produced the first ORWO data in the corpus. The rest needs a **local copy**: the online
  edition serves webp images that `web_fetch` returns empty, and no other retrieval route
  is attempted per the project's web rules.

---


### 4.9 The KODAK still-film E-series batch — what eleven sheets did NOT close (new 2026-08-26)

Eleven documents were read in full on 2026-08-26 (E-190 ×2, E-2468, E-4040, E-4050 ×2, E-4051,
E-7019, E-7022, E-7023, E-7024). Seventy-seven panels were located and none was skipped. These are
the gaps that survived, each with the reason it survived, because in five of the seven cases the
reason is a property of the source rather than a shortfall of effort.

| Gap | Why it is still open | What would close it |
|---|---|---|
| **Five real films are absent from the database entirely** — PORTRA **160NC**, **160VC**, **400NC**, **400VC**, **400UC** | Not a source problem: E-190 prints a full page for each (characteristic, spectral sensitivity, dye density, MTF) and all of it is traced and in hand. The database holds one `KODAK_PORTRA_160` and one `KODAK_PORTRA_400`, which are the **2011-onward** emulsions and different films. Adding five profiles renumbers `film_enum.hpp`, so it is a scoped change, not a data gap | a decision to add them; the numbers are already measured and recorded in `kodak_still_curves.py` |
| **rms granularity for all eight touched stocks** | ⚠ **RE-PROVED 2026-08-31 AGAINST THE OWNER'S FULL CORPUS, NOT THIS CHECKOUT — and the proof got stronger.** All **201** KODAK files on the owner's machine were searched for "diffuse rms granularity"; **more than eighty** print it, including the 1996–1998 E-55 and E-88 reversal editions that predate Print Grain Index. **Not one is a PORTRA, GOLD or ULTRA MAX colour negative still film.** The eight stocks this row names are exactly the population Kodak moved to Print Grain Index. ⚠ **NOT OBTAINABLE FROM THESE DOCUMENTS.** Every one of them publishes **Print Grain Index** instead and states that it "replaces rms granularity and has a different scale which cannot be compared to rms granularity". KODAK **E-58**, which defines the method, is on disk and declines to publish the transformation ("We will not describe the mathematical details involved in each step"); its first step alone depends on four properties of the **print paper** that this schema does not model | a Kodak publication printing a **diffuse rms granularity** figure or a granularity-vs-density plot for any PORTRA/GOLD/ULTRA MAX still film. None of the ~200 KODAK files on disk does |
| **Spectral sensitivity — CLOSED AS UNOBTAINABLE FROM THESE SHEETS, not deferred.** See §4.9.1 below for the per-panel evidence | The three layer curves **cross** — yellow-forming falls through magenta-forming's rise near 495 nm and magenta through cyan near 575 nm, confirmed against a 300 dpi render of E-190 (2003) p9. Tested every panel against a plausibility window (blue peak 415–485, green 525–565, red 595–665 nm): **exactly one panel in the batch passes, and it belongs to a film that is not in the database.** The eight profiles already carry sets cited to these same publications | a separation method that does not decide at a crossing: raster tracing with per-pixel run continuity, or a sheet that draws the three layers with distinguishable ink or dash |
| **Dye-density pair for `KODAK_PORTRA_400`** and for 2007-vintage ULTRA MAX 400 | The panels themselves: E-4050 resolves into **three** traces where its caption promises two, in both the 2010 and 2016 vintages, and E-7019's into **one**. `assign_dye_pair` refuses a crossing pair rather than label it by mean density. The 2016 ULTRA MAX 400 sheet (E-7023) **does** yield a clean pair, and it is adopted | a visual pass over E-4050 p4 to identify the third trace; it may be a leader line |
| ~~`KODAK_GOLD_100` gained nothing~~ | ✅ **CLOSED 2026-08-26f from the file this row named.** `E7022-Gold_100_200.pdf` (E-7022, February 2007) carries a captioned three-channel characteristic panel for GOLD 100, a Print Grain Index of 42, and the 1 s reciprocity bound. ⚠ **Both of that sheet's characteristic panels had been invisible to the first pass** — the caption matcher tested `startswith()` and this two-film sheet puts the panel kind at the END of the line ("KODAK GOLD 100 Film Characteristic Curves"), so two complete figures were skipped in silence. Matching on substring found both. GOLD 100 still has **no dye pair and no MTF**: the sheet prints one dye panel for two films (proven to be GOLD 200's — see §4.9.2) and no MTF panel in either edition | a Kodak sheet printing an MTF panel for any GOLD film |
| ~~Reciprocity for the six harvested stocks~~ | ✅ **CLOSED 2026-08-26 in a second pass.** Each sheet publishes a bound rather than a walk, and `ReciprocityTable` holds a bound as a one-point entry with a 0.0 correction (the censoring idiom, applied to time instead of frequency). All six read 1/10,000 s – **1 s**. ⚠ Two things recorded rather than smoothed: **E-4051 literally prints "to i second"** — verified against a 400 dpi render, so it is a typeset defect for "1" in Kodak's page and not an OCR artefact — and **ULTRA MAX 400's bound disagrees between vintages, 10 s in E-7019 (2007) against 1 s in E-7023 (2016)**, a factor of ten, stored as the later figure with the conflict cited | a Kodak sheet publishing a multi-point reciprocity walk for any of the six, as E-2468 does for PORTRA 100T |
| **Push-processing curve sets have nowhere to go** | A schema gap, and the data is in hand: E-190 prints EI 800 (Push 1) for 400UC, and EI 800 / 1600 / 3200 for PORTRA 800; E-4040 prints EI 1600 / 3200; E-7024 prints EI 800 / 1600 aim densities. All traced. `ProcessingFamily` cannot hold them — a C-41 push publishes **no development time**, and `DevelopmentPoint` requires `minutes > 0` — and `RGBCurves` holds one curve set per profile | a schema structure for alternate-EI curve sets. Measured shift, for scale: PORTRA 800's red gamma goes 0.5638 → 0.6883 → 0.7862 across EI 800 / 1600 / 3200, and its red dmin 0.3168 → 0.3599 → 0.3932 |
| **Dmax is extrapolated on all six harvested curve sets** | ⚠ **THE SHEETS DRAW NO SHOULDER.** Local-slope profiling shows the slope flat to the right edge of every panel (E-190 p9's last six red samples: 0.527, 0.528, 0.528, 0.529, 0.528, 0.527). `dmin`, `gamma` and the toe are measured; `shoulder_x` is carried over from the previous estimate and Dmax follows from it. `verify.py` asserts no shoulder falls inside a traced range | a sheet plotting these films past their shoulder. Colour-negative sheets generally do not |


#### 4.9.1 Spectral sensitivity: the per-panel result, so nobody re-attempts it blind

Run 2026-08-26 with the conditional chainer in place. The test is a plausibility window on the
assigned peaks — blue 415–485 nm, green 525–565, red 595–665 — which every real colour-negative
tripack satisfies. "REFUSED" means `assign_layers` got a trace count other than three and declined
to name them.

| Target | Sheet | Result | Assigned peaks |
|---|---|---|---|
| `KODAK_PORTRA_160` | E-4051 p4 | **REFUSED** — 0 traces | — |
| `KODAK_PORTRA_400` | E-4050 p4 (2016) | **IMPLAUSIBLE** | b **257**, g 548, r 648 |
| `KODAK_PORTRA_400` | E-4050 p5 (2010) | **IMPLAUSIBLE** | b **257**, g 548, r 648 |
| `KODAK_PORTRA_800` | E-4040 p4 | **REFUSED** — 4 traces | — |
| `KODAK_ULTRAMAX_400` | E-7023 p4 | **REFUSED** — 4 traces | — |
| `KODAK_ULTRAMAX_800` | E-7024 p3 | **REFUSED** — 4 traces | — |
| `KODAK_GOLD_200` | E-7022 p4 | **REFUSED** — 2 traces | — |
| PORTRA 400NC *(not in DB)* | E-190 p11 | **PLAUSIBLE** | b 469, g 543, r 618 |

The 257 nm "blue peak" is the axis's own left edge, which is the signature of a chain welded across
a crossing rather than a sensitivity maximum. **Zero of the seven database targets yields a usable
reading; the single clean panel belongs to PORTRA 400NC/400VC.** That asymmetry is not luck: the
E-190 family pages draw the three layers as three separate long subpaths, and the later single-film
sheets fragment them.

**So this is a closed question about this source, not an open task.** What would change it is a
different *kind* of source or a different *kind* of reader, not more effort on these eleven files.


#### 4.9.2 The two-document follow-up, 2026-08-26f — and one panel whose identity had to be *proved*

`E7022-Gold_100_200.pdf` (E-7022, February 2007) and `e29-Pro_100T_PRT.pdf` (E-29, April 1999).
Both read in full; both vector; neither publishes rms granularity.

**⚠ One dye panel, two films, and it is resolvable.** The 2007 GOLD sheet prints a single
*Spectral-Dye-Density Curves* panel and never says which of its two films it describes. Rather than
assign it to both — which would double-count one measurement — it was traced and compared against the
panel in `E7022-1.pdf` (E-7022, March 2022), a **GOLD 200-only** sheet that *does* name its film.
They are the same artwork: **max difference 0.0005 D, rms 0.00009 D over 59 resampled points of both
the neutral and the D-min curve.** So the shared panel is GOLD 200's, it was already adopted under
that name, and **`KODAK_GOLD_100` is honestly empty of dye data.** The audit pins this comparison, so
a later pass cannot quietly give the panel to GOLD 100 as well.

**Cross-document validation, two location mechanisms, fifteen years apart.** GOLD 200's
characteristic curves now come out of the 2007 sheet by *caption* and the 2022 sheet by *geometry
override* (the 2022 edition prints no caption at all). They agree to **0.002 D in dmin and 0.008 in
gamma**, and both sheets print `Log H Ref: -1.14`. That is the strongest check the uncaptioned-panel
override has had.

**⚠ Kodak omitted a minus sign, and this one is not a macron.** The 2007 sheet prints
`Log H Ref: 0.84` for GOLD 100 and `Log H Ref: -1.14` for GOLD 200 — verified against a 450 dpi
render of both captions, where the 200's minus is a real glyph and the 100's is simply **absent**, so
it is not the overbar defect seen on the E-190 axis labels. A positive Log H Ref is implausible for a
daylight ISO 100 negative when every other sheet in the corpus is negative, and **−0.84 against
−1.14 is exactly 0.301 decades — one stop — which is precisely the ISO 100 versus 200 difference.**
The inference is recorded; nothing is adopted from it, because `Log H Ref` is not a stored field.

**`KODAK_PRO_100T_PRT` — a new profile, and the database's 161st stock.** E-29 covers **KODAK Pro
100T Film / PRT**, a discontinued tungsten-balanced ISO 100 negative sold in 120 and sheets only, and
it was not in the database. Adopted: characteristic curves (fit rms 0.0066–0.0133 D), a neutral+D-min
dye pair over 450–700 nm, Print Grain Index for both formats it was sold in, and a **five-point
reciprocity walk** — only the second in the database. Grain and MTF are estimates by analogy to
PORTRA 100T, which E-29 itself names as the replacement.

**⚠ E-29 names PORTRA 100T as its successor, and that does NOT license copying anything.** E-29's
"recommended alternative" note cites E-2468 by number, so the two profiles are a documented
succession — but PRT's curves are its own and are not applied to PORTRA 100T, whose plots remain
Kodak's copy of PORTRA 160VC's (§0.2). **Their reciprocity tables are numerically identical, entry
for entry** (EI 100/80/64/50/40 at 5/10/30/60/120 s). Given that E-2468's figures are demonstrably
copied artwork, a carried-over table is at least as likely as two films measuring the same. Each
profile cites the publication that prints it; nothing is merged.

## 5. RECENTLY RESOLVED

**2026-08-18.** Spectral dye density adopted for **6 stocks** — 5205, 5219, 5245, 5274,
5279, 5293 — by vector-path extraction (`dye_density.py`, re-runnable, validated by
re-deriving the two previously ad-hoc sets: 5285 to RMS 0.003 D and 2383 to 0.135 D against
its own recorded 0.128 D base-absorber offset). `EASTMAN_5247_1974` / `EASTMAN_5247_1983` split into two generations
(§1.1, §2). B&W silver-negative σ(D) found in Mees Fig. 302 (§3). The dye-density and MTF
vector page counts measured rather than recalled (§4). `SVEMA_FOTO_65`'s scan-derived
`base_tint`, `silver_tone` and σ(D) **withdrawn** — the source batch mixes two emulsions.

**Earlier.** `KENTMERE_PAN_100/400` (HARMAN July-2022 sheets — reciprocity exponents
Ta = Tm^1.26/1.30 confirming the stored Schwarzschild p exactly); `KONICA_VX_100` (RMS 4,
63/125 confirmed); `ORWOCOLOR_NC21` + `ORWO_CHROM_UT18` speed/balance (Zhurba 1990 Table
66 — first ORWO data anywhere in the corpus); ОЧ-45/ОЧ-50 identity (**ГОСТ 20945-80**
appendix, Изм. № 1: same film, renamed 1987-01-01); `FUJICOLOR_SUPER_F500_8572` and
`AGFA_VISTA_200` (own sheets, RMS corrected ~2×); `FUJI_NEOPAN_1600` (AF3-608E);
`KODAK_EKTACHROME_100D_5285` (H-1-5285, 5294-borrow replaced); 12 VISION/EXR/Konica stocks
+ HP3 + 3 Rollei + 682 + Type B (provenance corrected — the sheets were on disk);
`ILFORD_HP3` (1942 manual); VISION3 5207/5203 and the HP5+/Delta/Acros/Scala/Vericolor
III/Profoto/UltraMax/TMAX/Tri-X/Plus-X/Ektapan settlements. Full detail in
`NotFound_history.md` and the dated `CHANGES_*` files.

---

## 6. The five highest-value enquiries in this file

Ranked by what one successful request would unlock, not by how easy it is.

1. **ORWO Handbuch** (Filmotec GmbH / Industrie- und Filmmuseum Wolfen). The corpus holds
   *two numbers* for all ORWO stock. One handbook would populate NC19/NC21/NC24-or-its-real
   equivalent and UT-18 at once — and settle whether **NC 24 ever existed** (§1.8).
2. **Ilford Manual of Photography, 1955–1965 edition** (HARMAN; National Science and Media
   Museum, Bradford). Closes HPS entirely and corrects HP3's 1942-vs-1950s generation
   problem in one document.
3. **Re-scope two generic profiles to named products** — costs nothing but a decision.
   `SOVIET_PANCHROM_1939` → «Изопанхром ФОКХТ» makes an *already-held* measured spectral
   curve (Gorokhovskii 1936 Fig. 7) immediately usable. `EASTMAN_ORTHO_1930` and
   `GEVAERT_PANCHRO_1950` cannot be documented at all until the same is done.
4. **The original 1974 Eastman 5247 data sheet** (Kodak archives; SMPTE 1974–76). The only
   route to making `EASTMAN_5247_1974` documented rather than reconstructed.
5. **Bayer JOSA 54 (1964) / Wilder JOSA 62 (1972)** — resolves a live
   measurement-versus-theory conflict on grain σ(D) that affects **all 160 stocks**, not one.

---

## 6b. ✅ A1 — the guard that was right for the wrong reason (2026-08-27)

**Closed.** The schema-v18 Eq. (1.1) coupling guard flagged `KODAK_EKTACHROME_100D_5285` at 50×
its class median, and it was written up as a data defect requiring a refit from the sheet. **That
diagnosis was wrong and is withdrawn.**

**What was actually true.** The profile's own comment already said it: *"gamma 11–15 is the
softplus straight-line slope of a model whose toe and shoulder nearly coincide; it reads only
together with toe_x/shoulder_x."* Evaluating the stored curve confirms it — **mid slope 2.419,
usable range 4.25 stops**, both ordinary for a reversal stock (Velvia 1.90 / 5.05, Kodachrome 64
1.68 / 5.81). The curve fits exact PDF vector coordinates to 0.024–0.028 D RMS. **No refit was
needed and none was performed.**

**What was actually defective** — narrower in one sense, broader in another:

| | Symptom | Scope |
|---|---|---|
| `ToneCurve.latitude_stops` | `(shoulder_x − toe_x) × 3.3219` ignores knee softness, so it stops describing the curve when the knees sit closer than their own smoothing | **4 of 161**: 5285 (5.6× out), `POLAROID_51` (1.8×), `POLAROID_146L` and `POLAROID_410` (1.5×). Accurate to 1 % on 139 and 5 % on 154 — it has a *domain*, and nothing said so |
| the guard itself | read `curves.g.gamma` as contrast, which in that regime is a model coefficient, not a slope | all 161 |

**Fixes.** `ToneCurve` gains evaluated `mid_slope`, `usable_range_stops` and `is_degenerate`. The
coupling guard reads `mid_slope`, and **its outlier count over the whole database drops to zero
with no exception list** — a guard needing an allowlist on its first run is usually measuring the
wrong quantity. A new **G-LAT** guard fails on any stored-vs-evaluated latitude disagreement that
is *not* explained by degeneracy, so a genuine fit defect still trips.

⚠ **The degeneracy threshold is derived, not tuned.** Sorting all 161 stocks by
`(shoulder_x − toe_x) / max(toe_k, shoulder_k)` against the latitude ratio gives a clean break —
1.248 at sep/k 3.32, then 1.475 at sep/k 2.47 — so the cut sits at **2.5 k**, which is also what
the geometry predicts (a softplus knee is smoothed over ≈ ±2 k).

**Method note worth keeping.** A stored *parameter* is not a *property*. Two of this project's
audits in one week found the same shape of error: a number that is correct inside its
parameterisation and wrong as a statement about film. Prefer evaluating the model over reading its
coefficients, in guards above all.

---

## 7. THIRD-PARTY DATA SOURCES EVALUATED AND REJECTED

This section exists so that a source is evaluated **once**. Each entry records what was
claimed, what was actually found, the decisive evidence, and what was salvaged.

### §7.1 filmlabpro.com/published-data — evaluated 2026-08-27; classification stands, **LIMITED IMPORT under owner instruction**

> ⚠ **STATUS CHANGED 2026-08-27, same day, by owner decision.** This entry originally read
> "REJECTED for import" and closed with "zero values imported". The owner reviewed the evidence,
> did not dispute it, and instructed: *"Regardless of the final classification, I would prefer to
> have T4 data rather than no data at all … Do not discard technically useful information solely
> because it is lower confidence; use it to close otherwise unresolved database gaps."*
> **The technical classification below is unchanged — it is still hand-authored, not measured.**
> What changed is the disposition: it is now imported for **exactly one parameter on nine
> profiles**, under tier 3 with a citation that says so. See §7.1a for what was imported and
> §7.1b for the tier correction.

**What was reviewed.** The complete published-data collection, including the section "The
stocks — every measurement" whose 21 per-stock dossiers are behind expandable JS elements and
are NOT visible to a plain page fetch. All 21 dossiers were expanded in a real browser, and the
underlying data object was then extracted from the application bundle
(`/assets/index-DdvumSO0.js`, `const spe = JSON.parse(...)`, 26 416 characters, 21 stocks) plus
the print-film literal `const ope = {...}`. **The bundle contains curve POINT TABLES that the
rendered page never displays.** The full harvest is preserved at
`doc/thirdparty/filmlabpro_harvest_2026-08-27.json` — the site does not need to be revisited.

> ✅ **RE-VERIFIED 2026-08-29 at the owner's request, and the 2026-08-27 harvest still stands.**
> The owner asked whether the site had been captured in full, including the material hidden behind
> buttons, expanders and drop-downs. Checked three ways rather than taken on this file's word:
> **(1)** the live page still reports **v2.1** and lists **exactly the same 21 stocks**;
> **(2)** the application bundle **`/assets/index-DdvumSO0.js` still resolves under that exact
> name** — Vite content-hashes bundle filenames, so an unchanged hash is an unchanged bundle, and
> the per-stock data lives inside it as `const spe = JSON.parse(...)`. Had a single number moved,
> the filename would have moved with it; **(3)** the archived harvest holds **21 stocks + 2 print
> films** across all six categories the site publishes — `characteristic_curve_points` (22 keys),
> `crosstalk_matrices` (19), `color_matrices_dye_to_rgb` (22), `layer_curves` (22), `grain_detail`
> (22), `film_stocks_summary_table` (21). ⚠ **This covers MORE than the site renders**: the curve
> point tables and the print-film literal are bundle-only and never appear on the page at all, so
> expanding every UI element by hand would have found *less* than what is archived.
> **The import is live and unchanged:** `_FILMLABPRO_HALATION_IMPORT` names **10 profiles** — the
> nine in §7.1a plus `CINESTILL_800T`, which arrived by the separate §7.2 route and shares the
> registry — and each carries the halation gain and threshold recorded there. Nothing else from
> this source is in the database, and §7.1's classification is unchanged: the five lines of
> evidence below are properties of the archived data, not of the page, so re-reading the page
> could not overturn them.

**What it claims.** "Every number our emulation uses — characteristic curves, dye matrices,
crosstalk, grain granularity, halation — published in full." Provenance statement, verbatim:
*"All curve points and measurements are digitized from these manufacturer technical
publications."*

**Classification: APPROXIMATED / HAND-AUTHORED, not measured and not digitized.** This is not a
provenance objection — per the owner's instruction of 2026-08-27, non-manufacturer *measured*
data is welcome where an official value is missing. The finding is that **no measurement exists
here at all.** Five independent lines of evidence, each verifiable from the saved harvest:

1. ⚠ **100 % of the 167 curve points across all 21 stocks lie exactly on a 0.5 log-E grid**,
   77.8 % of them on integer log-E (−3, −2, −1, 0, +1, +2, +3). A curve digitized from a printed
   plot produces irregular abscissae — our own traces do. Values landing exactly on a round grid
   are values that were typed, not read.
2. ⚠ **Density values carry only 1–2 decimal places** (55 points at 1 dp, 112 at 2 dp). Our
   traces of the same class of source return 3–4 dp, e.g. `dmin 0.2045`, `gamma 0.5809`.
3. ⚠ **The dataset contradicts itself.** Their stated `gamma` disagrees with the slope of their
   own curve points by up to **0.100** (Tri-X 0.62 stated vs 0.525 measured from their own
   points; T-Max 3200 0.65 vs 0.550; Acros 0.65 vs 0.565). A digitized dataset is internally
   consistent because both numbers come from one trace. These two fields were authored
   separately and never reconciled.
4. ⚠ **Their `dmin` is not film Dmin.** They publish ONE dmin per stock. A masked colour
   negative has three very different layer Dmin values, which is what the manufacturer sheets
   print and what we trace. For VISION3 500T our traced set is **0.1867 / 0.5811 / 0.8374**;
   they publish **0.2**, which matches only the red layer. For PORTRA 400 ours is
   **0.25 / 0.67 / 0.88** against their **0.2**. Their number is a display/scan-space value, as
   their own disclosure confirms: *"Display normalization anchors the scene working range
   (logE −2.5 to +1.8) to 0–1, not the full physical Dmin–Dmax."*
5. ⚠ **Headline parameters are round to a degree real measurements are not**: every
   `latitude_stops` an integer; every `mtf50` an integer and 86 % on a 5 lp/mm grid; every
   `size_microns` on a 0.1 µm grid; 90 % of `rms` on a 0.5 grid.

**Corroborating facts.**
- No instrument, operator, date or laboratory is named anywhere on the site.
- Their validation suite compares GPU output against a CPU implementation of the same maths.
  Verbatim: *"this suite validates that the pipeline faithfully executes the published datasheet
  data — it does not (yet) compare against physical film scans."*
- **11 of 21 stocks carry no cited source at all** — Tri-X, Acros, Kodachrome 64, Ektachrome
  E100, CineStill 800T, Pro 400H, UltraMax 400, both Eternas, and both print stocks.
- Only ONE curve per stock exists in the whole dataset. They cite Kodak E-4050, which prints
  three separate layer curves; had they digitized it, three curves would exist. Instead
  `layer_curves` holds relative multipliers applied *"clamped to 35 % strength"*.
- `spectralSensitivity`, `spectralMatrix` and `layerMtf` are **runtime functions**
  (`Spe()`, `Bq()`, `kpe()`) in the bundle, not stored data. The 3×3 "spectral exposure matrix"
  the page displays is a computed collapse of Gaussian lobe fits. **There is no per-stock
  spectral dataset on this site**, which their own roadmap note concedes: *"Spectral sensitivity
  is currently reduced to 3-channel luminance weights per stock, not full 31-band spectra."*
- `halation.radius_norm` is a fraction of image dimension, not a length on film — a rendering
  parameter, not a film property. `color_matrix` for every B&W stock is the Rec.601 luma triplet
  0.299/0.587/0.114 repeated three times.
- Factual error on the page: an Agfa Vista datasheet is attributed to Fujifilm code AF3-044E2,
  conflating two stocks.

**Consequence for the *headline* parameters: still zero imported.** Every one of `rms`,
`mtf50_lp_mm`, `dmin`, `gamma` and `iso` this site publishes lands on a parameter our database
**already has populated**, so the owner's own rule — *"if an equivalent parameter already exists
with an official vendor/manufacturer value, retain the official value and do not overwrite it
with lower-tier data"* — forbids the overwrite before classification even enters. Four of those
collisions are direct falsifications rather than mere disagreements: for the four stocks where we
hold the very datasheet this site names as its source, its number is not in that document.

> ⚠ **THIS TABLE CONTAINED AN ERROR AND IS CORRECTED HERE (2026-08-27, later the same day).**
> The Portra 400 row read *"our 4 — Kodak E-4050, tier 1"*. **E-4050 prints no rms granularity at
> all.** Kodak publish **Print Grain Index** for that film and the sheet itself says PGI *"cannot
> be compared to rms granularity"* — the whole reason schema v15 exists. That profile's own comment
> says so in terms: `rms 4.0 IS NOT A KODAK FIGURE`. So our 4.0 was an unattributed estimate, this
> was never a contradiction, and the corrected table below drops it and adds two real ones found
> while re-checking.

| stock | our value, and its source | their value | over-statement | verdict |
|---|---|---|---|---|
| Acros 100 rms | **7** — Fuji sheet | 4.5 | 0.64× | contradicts |
| Velvia 50 rms | **9** — `velvia_50_datasheet.pdf` p7, *"DIFFUSE RMS GRANULARITY VALUE ……9"* | 3.8 | 0.42× | contradicts |
| Kodachrome 64 rms | **10** — Kodak E-55, confirmed against `e88-2009_06.pdf` p4 *"Diffuse rms Granularity: 10"* | 6 | 0.60× | contradicts |
| **Agfa Vista 200 rms** ⭐new | **4.3** — AGFACOLOR Vista *Technical Data AF*, 06/2000 | 8.0 | **1.86×** | contradicts |
| **Eterna Vivid 500T rms** ⭐new | **3.5** — `eterna_vivid500.pdf`, 48 µm at D = 1.0 | 9.0 | **2.57×** | contradicts |
| ~~Portra 400 rms~~ | ~~4 — E-4050~~ | 6.5 | — | **withdrawn: E-4050 prints no rms; our 4.0 was the estimate** |

⚠ **The direction is not random, and that is the useful finding.** On the two stocks where the
sheet prints a FINE figure (Vista 4.3, Eterna 3.5) they read **1.9–2.6× coarser**; on the three
where the sheet prints a COARSE figure (Acros 7, Kodachrome 10, Velvia 9) they read **0.4–0.6×
finer**. They are not biased one way — **they are compressed toward the middle of their own set**,
which is the signature of numbers typed to look plausible rather than measured. Any figure adopted
from this source should be read as pulled toward ~7–8, i.e. too coarse for a fine-grained stock and
too fine for a coarse one.
| Velvia 50 "mtf50" | f50 **98** lp/mm; resolving power **160** lp/mm | "mtf50 160" | they printed the *resolving power* in an MTF-50 field — two different quantities |
| T-Max P3200 iso | **EI 1000** (its ISO speed; 3200 is the box/push rating) | "iso 3200" | box speed labelled as ISO |

**What WAS salvaged — three real gains.**
1. **Granularity convention confirmed**, which unblocks the DQE consistency guard described in
   `EMULSION_KNOWLEDGE_BASE.md` §26 B2: `rms_granularity = σ_D × 1000 at D = 1.0 through a
   48 µm aperture`, rescaled between apertures by **Selwyn's law σ²·A = constant**. Both are
   Kodak/textbook convention rather than this site's measurement, so they are cited to the
   primaries — but they confirm the semantics of our own field.
2. **Five acquisition leads with publication codes** — see `DIGITIZATION_QUEUE.md` items
   **T1–T3**.
3. **Nine halation gaps closed** — the one place where this source supplies a parameter our
   database genuinely did not have. See §7.1a.

---

### §7.1a What was actually imported, 2026-08-27 — halation on nine profiles, and nothing else

**Why halation and only halation.** All 21 of their stocks were audited field by field against
our database. Seventeen map onto profiles we hold; four have no counterpart (§7.1c). Of every
parameter they publish, **exactly one was unpopulated on our side**: nine of those seventeen
profiles carried `HalationSpec` entirely at its schema default — `gain_r = gain_g = gain_b = 0.0`
with `Feature.HALATION` unset, i.e. **the effect was switched off, not merely estimated.** Real
film always scatters some light in the emulsion and base, so zero was wrong for all nine, and
this source is the only material on hand that orders them.

| profile | imported gain (r, g, b) | threshold_stops | profile tier | their raw record |
|---|---|---|---|---|
| `KODAK_PORTRA_800` | 0.15 / 0.034 / 0.009 | 1.55 | 3 | int 0.15, tint (0.80, 0.18, 0.05), thr 0.60 |
| `KODAK_EKTAR_100` | 0.07 / 0.018 / 0.005 | 1.95 | 3 | int 0.07, tint (0.78, 0.20, 0.05), thr 0.70 |
| `KODAK_GOLD_200` | 0.10 / 0.020 / 0.005 | 1.75 | 3 | int 0.10, tint (0.82, 0.16, 0.04), thr 0.65 |
| `KODAK_ULTRAMAX_400` | 0.10 / 0.019 / 0.005 | 1.65 | 3 | int 0.10, tint (0.84, 0.16, 0.04), thr 0.63 |
| `AGFA_VISTA_200` | 0.08 / 0.020 / 0.006 | 1.85 | 2 | int 0.08, tint (0.82, 0.20, 0.06), thr 0.67 |
| `KODAK_TRI_X_400TX` | 0.05 / 0.05 / 0.038 | 1.65 | 2 | int 0.05, tint (1,1,1), thr 0.75 |
| `ILFORD_HP5_PLUS_400` | 0.05 / 0.05 / 0.038 | 1.65 | 2 | int 0.05, tint (1,1,1), thr 0.75 |
| `KODAK_TMAX_P3200` | 0.08 / 0.08 / 0.060 | 1.40 | 2 | int 0.08, tint (1,1,1), thr 0.65 |
| `FUJI_NEOPAN_ACROS_100` | 0.04 / 0.04 / 0.030 | 1.70 | **1** | int 0.04, tint (1,1,1), thr 0.78 |

**Conversion rules, written down so the import can be undone.**

* `gain_r` = their `halation.intensity`, **verbatim and unscaled**. Across the eight stocks where
  both datasets carry halation, the ratio of our gain to their intensity scatters 0.60–3.75, so no
  calibration factor is defensible; their number is taken as-is.
* Colour negatives: `gain_g`, `gain_b` = `intensity × tint_g/tint_r` and `× tint_b/tint_r`.
* B&W: `gain_g = gain_r` (silver scatter is neutral, and their B&W `tint` is the flat (1,1,1)
  display convenience — it carries no information); `gain_b = 0.75 × gain_r`, taken from **this
  project's own** B&W convention (`ILFORD_FP4` 0.04/0.04/0.03, `ILFORD_HP4` 0.05/0.05/0.04), not
  from the third party.
* Colour `threshold_stops` = `4.227 × their_threshold − 0.991`, least squares over the eight
  co-covered stocks, **R² = 0.881, n = 8**. B&W thresholds (0.65 / 0.75 / 0.78) mapped linearly
  onto this project's existing B&W band 1.40–1.70 (`ILFORD_HP4` 1.4 fastest, `ILFORD_FP4` 1.6,
  `ILFORD_PAN_F` 1.7 finest) — the colour regression was fitted on colour negatives and does not
  transfer. **Both fits are anchored on our own values, so only the ORDERING of which stock
  halates at what luminance is third-party information; the absolute scale is ours.**

**⚠ `FUJI_NEOPAN_ACROS_100` is a tier-1 profile and keeps tier 1.** The imported halation carries
an inline `[T3]` per-parameter tag, following the precedent already in this database
(`EASTMAN_5247_1983` is tier 1 with hand-fitted tone curves; the `# [T2] halation:` note on
`ILFORD_HPS`). A profile's tier describes its documented core, not every scalar in it. The
citation appended to its `sources` states plainly which single parameter came from this source.

**What was NOT imported, and why — each of these remains a gap in this file.**

* `halation.radius_norm` — **a fraction of the image dimension, not a length on film.** Their
  values do order monotonically against our `radii_um` (0.002 ↔ our 7–9 µm inner lobe, 0.012 ↔ our
  20 µm), but every conversion factor derivable from that is fitted against *our own estimates*,
  so it would be circular and would add no information. `radii_um` and `weights` stay at the
  schema default (12 / 60 / 320 µm) on all nine. **Measured halation radius for any stock is
  still unavailable.**
* `size_microns` ("mean crystal diameter") — no field. Our `clump_um_*` is *developed clump*
  diameter and depends on development gamma (see the `GrainSpec` docstring and BBC T-101); the two
  are different quantities and must not be aliased.
* ~~`dmax`, `latitude_stops` — no field in schema v15. A schema extension carrying only third-party
  hand-authored values is a worse trade than the gap.~~
  ⚠ **SUPERSEDED THE SAME DAY, AND THE JUDGEMENT ABOVE WAS OVERTURNED BY THE OWNER, NOT BY NEW
  EVIDENCE.** Schema **v17** added `ThirdPartyObservations`, which carries `dmax` and
  `latitude_stops` among others, on the owner's instruction that where a parameter is our own
  estimate and no T1 datasheet or T2 book figure exists, the one published third-party number is
  preferable to an in-house analogy. The record is **INERT** — nothing on the render path reads it —
  so the "worse trade" concern was answered by isolation rather than by refusal. ⚠ **Every value in
  `third_party` is TIER 3 and is never evidence for the matching observable.**
* `layer_curves` toe/shoulder gamma — relative multipliers in **their** parameterisation, applied
  at their own admitted "35 % strength". Not our `ToneCurve` toe/shoulder. Unusable.
* `crosstalk` 3×3 — our `interimage` is already non-default on every colour stock they cover, and
  their matrix is absent for all four B&W stocks anyway.
* `color_matrix` — for B&W stocks it is literally the Rec.601 luma triplet 0.299/0.587/0.114
  three times. A display convenience, not a dye matrix.
* `saturation`, `vignette`, `speed_offset`, `grain.shadow_bias/midtone/highlight_bias` — no stated
  definition, units or basis anywhere on the site (their own field list marks these
  "semantics unknown"). Our `default_vignette` is on a different normalisation entirely
  (ours 0.35–0.85, theirs 0.04–0.18).

### §7.1b Tier correction — there is no tier 4, and the owner was right

An earlier draft of this review labelled this material **"[T4]"**. **That label was invented and
has no basis in the schema.** `film_profiles.Provenance` defines, verbatim:

```
tier: 1 = datasheet-grounded, 2 = partially grounded,
    3 = reconstruction.
```

Three tiers, and tier 3 is *reconstruction* — precisely what hand-authored engine values are. The
owner's challenge (*"I believe this should be T3"*) is correct and the import uses **tier 3 with
`fitted_from="secondary_sources"`**. A practical reason this matters beyond bookkeeping:
`_provenance_for`'s regex accepts only a bare `[T1]`, `[T2]` or `[T3]` tag, and **anything else —
including `[T4]` — falls through to `_UNTAGGED_TIER` and then silently to 3.** An invented tag
would not have failed loudly; it would have been quietly reinterpreted.

### §7.1c Their four stocks with no counterpart here — still absent, still a gap

`fuji_eterna_250d`, `provia_100f`, `pro400h`, `fuji_superia_400` have no profile in this database
under any spelling. (We hold `FUJI_ETERNA_VIVID_500T_8547`, a different coating from their "Eterna
500T", and `FUJI_PROVIA_400X`, a different film from Provia 100F.) **No profile was created for
any of them.** A whole profile built from hand-authored third-party values would be a fabricated
stock, which is a different and much worse thing than filling one scalar on a documented one.
Creating them needs the real sheets — Fuji AF3-0076E5 (Provia 100F) and AF3-058E3 (Superia 400)
are named on their page and are tracked as `DIGITIZATION_QUEUE.md` item **T3**.

**Still unavailable after this review — unchanged gaps.** Per-stock MTF curves and f50 for the
still-film corpus; **measured halation radius for any stock** (§7.1a: only gain and threshold were
closed, and only at tier 3); measured interlayer crosstalk for any stock; rms granularity for the
eight KODAK still stocks (K5); spectral sensitivity for the stocks listed in §4.9.1.
⚠ **One item on this list was closed later the same day and is struck here rather than left to
mislead:** ~~`dmax` and exposure-latitude fields do not exist in schema v15 at all~~ — **schema v17
added `ThirdPartyObservations`, which carries both.** The values in it remain tier 3 and inert; the
*gap in the schema* is closed, the *gap in the evidence* is not.

### §7.2 cinestillfilm.com — CineStill 800T, owner-supplied 2026-08-27. **DIGITIZED AND IMPORTED**; two items remain open

> **STATUS: both gaps this section opened earlier the same day are now CLOSED.** Gap 1 (push
> latitude had nowhere to go) is closed by **schema v16**. Gap 2 (the vendor sensitometric plots
> were uncalibrated) is closed by a full digitization: **CINESTILL_800T's three characteristic
> curves are now traced, not estimated.** What remains open is listed at the end, and it is
> narrower than what came in.

**Why this stock is treated differently from every other undocumented profile.** CineStill's
emulsion is proprietary and **they publish no technical data sheet** — no granularity figure, no
MTF, no spectral sensitivity, no numeric Dmin or Dmax. Every one of those stays an estimate. But
they *do* publish a real sensitometric figure, and a vendor-published figure is a
**manufacturer-class document**, not third-party material. That is the whole basis for what
follows, and it is a different argument from §7.1.

#### §7.2a Text of the page — cited, qualitative

`https://cinestillfilm.com/blogs/news/what-makes-800t-the-original-and-only-true-800-speed-tungsten-balanced-film-for-still-photography`.
Grounds: EI 800 box speed in tungsten light; optimisation for **3200 K**; remjet removed by
CineStill's "Premoval" process; "Xpro C-41" processing; 135 / 120 / 4×5; "**red halation glow**"
as the signature trait; and *"could even be push processed up to 3 stops further without any base
fog issues"*. It **corroborates** the profile's red-dominant halation
(`gain_r 1.05 ≫ gain_g 0.30 ≫ gain_b 0.10`) and its 3200 K / EI 800 balance. It prints **no**
numeric halation radius, granularity or Dmax.

#### §7.2b ✅ Push latitude — CLOSED by schema v16, not by a workaround

`FilmProfile` gained a `PushSpec` record. `CINESTILL_800T` carries
`max_push_stops = 3.0`, `base_fog_penalty_per_stop = 0.0`, `fog_penalty_stated = True`, plus the
verbatim vendor quote as its source. The other 160 profiles take an all-zero default with an empty
source, so **no generated number changes for any of them** and the schema bump is backward
compatible.

Two design points worth keeping, because both were near-misses:

* **It is not a `ProcessingSpec` field.** `ProcessingSpec` describes the *one* development
  condition the stored curve represents. "+3 stops" is a claim about a *family* of other
  conditions; written beside a single time and temperature it would read as applying to that time,
  which no source means. `ProcessingFamily` was also wrong — it carries measured time-gamma
  *points*, and a prose sentence is not a point on that curve. `exposure_index` was wrong too:
  overwriting the rating with a pushed EI destroys the rating.
* **`base_fog_penalty_per_stop = 0.0` is ambiguous on its own**, so the struct carries a separate
  `fog_penalty_stated` flag. CineStill's claim is the *negative* one — no fog penalty — which is a
  published fact and must not be stored indistinguishably from silence. Same class of problem as
  the v15 PGI censoring sentinel, solved the same way.

**Still not published, and not inferred:** gamma gain per pushed stop, realised speed gain per
pushed stop, and any pull figure. A push *range* says nothing about the contrast a push buys, so
those fields stay zero on this stock and on every other.

#### §7.2c ✅ The vendor plot — CLOSED. Curves traced, 480 points per layer

The four rasters on that page, resolved:

| image | subject | disposition |
|---|---|---|
| `cs41curves_600x600.png?v=1712783069` | **three per-layer D-logE curves, R / G / B** | ✅ **digitized, calibrated, fitted, IMPORTED** |
| `cs41vscs2curves2_PNG_480x480.png?v=1712783070` | CS41-vs-CS2 process comparison, two curves | digitized and calibrated, **NOT imported** — see §7.2d |
| `Kodak400Sensi_600x600.png?v=1712789230` | counterfeit "Brand H" 400 vs Kodacolor 400 | **a different film** — reference only |
| `Kodak400Sensi400D_600x600.png?v=1712789230` | "Brand H" 400 vs CineStill **400D** | **a different film** — reference only |

**Method.** The PNGs are same-origin readable from an HTML `<canvas>`, so no screenshot was
needed after all — the earlier "tooling limit" was solved by reading pixels directly. Curves were
separated by saturated-hue classification and tracked column by column under a continuity
constraint, which is what bridges the in-plot legend box at x 403–436 without a gap. **480 samples
per layer, one per pixel column** — denser than any hand trace elsewhere in this database. Both
axis titles and all 26 tick labels were reconstructed glyph by glyph from the pixel grid.

**Calibration, over-determined on both axes.**

* Ordinate: nine printed labels 0.0…3.5 by 0.5, spacings 69.5 / 68.5 / 69.5 / 69.5 / 69.0 / 69.5 /
  68.0 px → `D = (541.5 − y)/138.1428`, which puts D = 0.0 exactly on the bottom frame and
  D = 3.5 on the top frame.
* Abscissa: a **top** axis of six log₁₀-exposure labels −4.0…+1.0 (spacings 77.0 / 75.5 / 74.0 /
  75.5 / 74.5 px, spanning the frame exactly) **and** a **bottom** axis titled "Camera Stops" with
  seventeen labels −8…+8. Chart title: "Log Exposure".

**⚠ CineStill's own chart is internally inconsistent by 3.7 %, and this is a finding, not a
reading error.** Sixteen stops are drawn across exactly 5.00 decades — 3.20 stops/decade — where a
stop is 0.30103 decades and sixteen stops are 4.816 decades. **The log axis was adopted** (primary
sensitometric abscissa, label spacing uniform to ±1 px, lands on the frame edges); the stops axis
was used **only for its zero**, which fixes metered mid-grey at log E = **−1.51681**. All 1440
points were then shifted by +1.51681 so mid-grey sits at x = 0, this database's `ToneCurve`
convention. **The same 3.7 % inconsistency appears on the second chart**, so it is a property of
CineStill's plotting, not of one figure.

**Fit and residuals.** Six `ToneCurve` parameters solved per layer by coordinate descent on all
480 points:

| layer | dmin | gamma | toe_x | toe_k | shoulder_x | shoulder_k | rms | max resid |
|---|---|---|---|---|---|---|---|---|
| r | 0.1873 | 0.6023 | −1.4188 | 0.3180 | 1.8145 | 0.4450 | 0.0197 D | 0.039 D |
| g | 0.5258 | 0.6214 | −1.6912 | 0.3134 | 2.0148 | 0.4385 | 0.0248 D | 0.060 D |
| b | 0.8758 | 0.6088 | −1.5706 | 0.1467 | 1.9976 | 0.2053 | 0.0154 D | 0.053 D |

**⚠ All three shoulders sit at the `shoulder_k = 1.4 × toe_k` ceiling** the `ToneCurve` docstring
imposes. The measured shoulders want to be *softer*; 1.4× is the monotonicity-safe bound this
project holds new stocks to, so the fit is clamped and the modelled shoulder is very slightly
sharper than the plate's. Recorded rather than quietly exceeded.

**What the trace changed, and what it confirmed.**

* **`dmin` becomes the orange-mask ladder 0.187 / 0.526 / 0.876.** The estimate it replaces was a
  flat 0.22 / 0.20 / 0.19 — not merely imprecise but the **wrong kind of description** for a
  masked colour negative, exactly the defect the 2026-08-26 KODAK still-film batch found on eight
  profiles. `CINESTILL_800T` therefore joins `_DMIN_LADDER` and `mask_encoding` flips from
  `neutral_dmin` to `dmin_ladder`.
* **Independent identification of the plate.** KODAK VISION3 500T 5219 — the film this emulsion
  *is*, minus the remjet — traces to **0.187 / 0.581 / 0.837** from its own Kodak H-1 sheet.
  Worst-channel agreement **0.06 D**. That is simultaneously a check on the whole calibration
  chain and the evidence that the figure really is this stock, which the raster itself never
  states in words.
* **Gamma barely moved**: 0.602 / 0.621 / 0.609 traced against 0.610 / 0.630 / 0.652 estimated.
  The estimate was good. The one structural change is that the traced set has blue slightly
  *below* green (r < b < g) where the estimate had a monotone r < g < b.
* **Toe and shoulder land on the old estimate** once the mid-grey shift is applied: red toe −1.419
  traced against −1.52 estimated, red shoulder 1.815 against 1.86.

**Tier: stays 2, `fitted_from` becomes `datasheet_curve`.** The curves are vendor-grounded; grain,
MTF, halation magnitude, couplers, dye matrix and spectral response on the same profile are not.
Tier describes the profile, `fitted_from` describes how the curves were obtained — the same split
the Kodak still-film batch established. A new `_VENDOR_TRACED_CURVES` set carries it, kept
separate from `_KODAK_STILL_HARVEST_CURVES` because the document class differs: a vendor news page
carrying a real figure, not an E-series or H-1 publication.

**Archived** at `doc/thirdparty/cinestill_cs41_raw_px.txt` (raw pixel rows, both figures) and
`doc/thirdparty/cinestill_curves_2026-08-27.json` (calibrated points in both the figure's native
log E and this database's convention, plus the fit and the cross-source comparison). The page does
not need revisiting.

#### §7.2d ⚠ STILL OPEN — the second figure's two curves are calibrated but UNIDENTIFIED

`cs41vscs2curves2` is fully calibrated (`D = (433.5 − y)/129.2`, `logE = (x − 366.862)/75.186`,
D 0.5…3.0, log E −4.0…+1.0, stops −8…+8) and both curves are traced — 284 and 333 points, both
dashed. **What is not established is which is which.** The in-plot annotation block could not be
read reliably off the pixel grid, so:

* which curve is the **CS41** process and which the **CS2** process is unknown;
* which **density channel** is plotted is unknown (it is one curve per process, not three).

Measured anyway, because it does not need the assignment: the two processes differ by about
**12 % in straight-line gamma** — **0.555** (red dashed) against **0.493** (neutral dashed) over
log E −2.07…−0.08 — on a base+fog of 0.57–0.62 D for both. **That is a real bracket on how much
the process moves contrast on this film**, and it is why the assignment matters rather than being
a curiosity: adopting the wrong one would shift stored gamma by 12 %.

Inference was deliberately *not* used to break the tie. A two-bath simplified kit "ought" to be
the lower-contrast one, but the red curve's 0.555 is also below the CS41 figure's own green-layer
gamma of 0.621, so the naive story does not close. **Not imported.** Tracked as
`DIGITIZATION_QUEUE.md` item **T4** — reduced to one question: read the annotation block.

#### §7.2e Two independent sources for the same curve — the discrepancy, preserved

The vendor figure and the FilmLab Pro engine (§7.1) both describe CineStill 800T's characteristic
curve. **Neither was discarded.**

| | this digitization (vendor figure) | FilmLab Pro (§7.1) |
|---|---|---|
| class | vendor-published plot, digitized here | hand-authored engine values, tier 3 |
| curves | **three**, per layer | **one**, for a three-layer film |
| points | **480 per layer** | 7, all on an integer log-E grid |
| base+fog | 0.187 / 0.526 / 0.876 | single `dmin` 0.22 |
| avg gradient, log E −2…+1 | **0.455** (red record) | **0.467** |

**They agree to 2.6 % on average gradient**, and FilmLab Pro's single `dmin` 0.22 matches this
digitization's red base+fog 0.217 to 0.003 D. **They disagree by 0.4–0.75 D on absolute density**
at matched log E, and FilmLab Pro's abscissa sits about a decade to the right.

**The discrepancy is explained, not split.** Both differences follow from FilmLab Pro's own
disclosure that *display normalisation anchors log E −2.5…+1.8 to 0–1 rather than to physical
Dmin–Dmax*, and from their publishing one curve where the film has three. So the two sources are
not measuring the same quantity: one is film density, the other is display-normalised code value
that happens to be anchored on the red layer's base. **The vendor figure is adopted for the stored
curves** — vendor document, three per-layer curves, base+fog ladder independently confirmed
against VISION3 500T. **The FilmLab Pro points are retained** in the archive as a corroborating
independent check on the one quantity the two agree on. Recorded in the archive JSON under
`cross_source_comparison`.

#### §7.2f Still genuinely unavailable for CINESTILL_800T after all of this

Stated at the width that is actually true, so nobody re-searches what has been searched:

* **rms granularity** — no CineStill document publishes one. Stored 8.4 is an estimate from the
  VISION3 500T ladder. §7.1's third-party 11.5 was **not** imported (it collides with a populated
  field and its provenance is §7.1's).
* **MTF / f50 and resolving power** — nothing published. Stored 40 / 48 / 56 c/mm is an analogy.
* **Spectral sensitivity** — nothing published. Neither source has per-stock spectral data at all.
* **Halation radius and magnitude in physical units** — still the largest real gap on this
  profile, and the one its whole reputation rests on. The vendor page confirms the effect
  *qualitatively* ("red halation glow"); it prints no radius and no strength. §7.1's
  `radius_norm 0.012` is a fraction of image dimension, not a length on film, and is not
  convertible without circularity. Stored `radii_um (20, 130, 700)` and
  `gain (1.05, 0.30, 0.10)` remain **estimates** — the strongest in the database, on qualitative
  grounds alone. **What would close it:** a measured edge-spread or highlight-flare profile from
  a scan of a known target, i.e. an owner measurement, not a document.
* **Dmax** — no field in the schema, on any stock. The traced curve's asymptotic Dmax is
  2.135 / 2.829 / 3.048 (a `ToneCurve` property, not a stored figure).
* **Push contrast and speed behaviour** — the range is now stored; the gamma and speed gain per
  pushed stop are not published by anyone (§7.2b).
* **The second figure's curve assignment** — §7.2d.

