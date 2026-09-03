# RESULT 2026-09-02e — A1, A4, A2, then the owner addendum T3 / T1 / T2 / E4, and Takano 1968

One batch, approved without pause. It ran in three parts: the A-list agreed the night
before (A1 → A4 → A2), an addendum the owner added mid-batch (T3, T1, T2, E4, unblocked by
the 1942 Kodak book arriving on his machine), and a single-document review he interrupted
with (`31_209.pdf`).

⚠ **THE HEADLINE, BECAUSE IT IS NOT THE ITEM THAT WAS SCHEDULED TO BE INTERESTING.** A4 was
listed as housekeeping — "resolve the 12 stocks whose `adjacency` reads as active and
renders as inert". It is not housekeeping. The cause is a **definitional error in how
`adjacency` was ever stored**, it affects **every stock whose adjacency came from a traced
sheet**, and on the day it was found **not one of the thirteen measured stocks could
reproduce its own printed overshoot.** They all can now.

---

## 0. What changed, in one table

| | before | after |
|---|---|---|
| film stocks | 172 | **175** |
| stocks with a measured MTF | 19 | **23** |
| stocks that render their sheet's printed overshoot | **0 of 13** | **13 of 13** |
| stocks carrying an inert `adjacency` | 12 | **11, each with a written refusal** |
| tier-1 profiles | 85 | **88** |
| negative / reversal | 132 / 40 | **134 / 41** |
| registered reader scripts | — | **+2** (`takano_1968_mottle.py`, `fuji_t3_2026.py`) |
| corpus PDFs | 90 | **97** |

---

## 1. A1 — the corpus sweep, and the answer is "nothing is unread"

Every PDF in `PDF/` was classified by where it is cited: the profile database, a reader
script, `build.py`'s audit registry, and the documentation. **Ninety files, zero
unaccounted for.** The breakdown:

| cited in | count |
|---|---|
| profile provenance (`film_profiles.py`) | 64 |
| a reader script | 66 |
| the `build.py` audit registry | 28 |
| a document | 77 |
| **nowhere at all** | **0** |

Three byte-identical duplicate groups exist and are harmless: two copies each of
`Fujifilm-Super-F-125-8532…` and `eterna_vivid500.pdf`, and `5205t.pdf == H-1-5205t.pdf`.

⚠ **THE ONE SUSPICIOUS-LOOKING RESULT AND WHY IT IS NOT.** Thirteen files are named in no
document. All thirteen are Kodak datasheets whose *stocks* are in the database and whose
provenance strings cite them by **publication code** rather than filename — `E7023_max_400.pdf`
appears as "publication E-7023", and so on. One of the thirteen, `E7019_en-Ultra_Max_400.pdf`,
is cited nowhere by any spelling, and that is deliberate and already on record: the
`KODAK_ULTRAMAX_400` entry says the 2007 vintage "prints the panel with its traces fragmented
past reassembly and is not used."

Nineteen files are cited only in prose, of which fifteen are the Japanese collection
§23i.4 dispositioned file by file in August. **A1 therefore closes as a negative result:
there is no unread material in the corpus.** The sweep is worth having precisely because
that could not be asserted before.

---

## 2. A4 — the adjacency defect, and why it was not housekeeping

### 2.1 What was wrong

`MTFSpec.adjacency` was set, on every stock traced from a sheet, **equal to the overshoot
the sheet prints**. `EASTMAN_PLUS_X_5231`'s comment stated the rule outright: "MTFSpec
defines adjacency as the fractional overshoot above unity, and the plot peaks at 103.4 %,
i.e. 0.034."

It does not. `film_sim.FreqGrid.mtf` computes

```
T(f) = rolloff(f) · [ 1 + adjacency · ( G(0.4·a) − G(2.0·a) ) ]
```

so `adjacency` is the amplitude of the difference-of-Gaussians lift **before the rolloff
attenuates it**. What survives to the image is always smaller. Storing the observed value
therefore always understates the effect — and on stocks whose rolloff is steep enough, it
understates it to nothing.

⚠ **THE MEASURED CONSEQUENCE. Across the thirteen sheets in `mtf_vector.EXPECTED`, the
stored pair reproduced the traced peak on ZERO records.** Two examples of the size of it:

| stock | sheet prints | stored pair rendered |
|---|---|---|
| `EASTMAN_PLUS_X_5231` | +3.4 % at 4.6 c/mm | **no peak at all** |
| `KODAK_VISION2_250D_5205` | +3.2 % at 14.6 c/mm | +1.0 % at 6.7 c/mm |

5231 was one of the twelve C19 called "inert". It was not a small stock with a small
value; it was a **guard actively holding the defect in place** — `verify.py` pinned
`adjacency == 0.034` and so pinned the film to rendering no edge effect at all.

### 2.2 The fix is a solve, not a rescale

A traced overshoot is **two numbers** — a peak value and a peak frequency — and the model
has **exactly two free parameters**. The system is determined. For each stock, find
`(adjacency, adjacency_um)` such that

* `argmax_f T(f)` equals the traced peak frequency, and
* `max_f T(f)` equals the traced peak value,

solved on that stock's own stored rolloff over a 0.1–300 c/mm log grid, `adjacency_um`
refined to 0.01 µm. Thirteen stocks (twelve from A4 plus `KODAK_EKTAR_100` from T2):

| stock | traced | old adj / µm | **solved adj / µm** |
|---|---|---|---|
| `KODAK_EKTACHROME_100D_5285` | 1.030 @ 7.8 | 0.030 / 15.0 | **0.0716 / 16.4** |
| `EASTMAN_PLUS_X_5231` | 1.034 @ 4.6 | 0.034 / 16.0 | **0.0691 / 32.1** |
| `EASTMAN_DOUBLE_X_5222` | 1.250 @ 4.1 | 0.250 / 25.0 | **0.2996 / 49.8** |
| `KODAK_VISION2_50D_5201` | 1.157 @ 10.7 | 0.157 / 16.0 | **0.1980 / 17.7** |
| `KODAK_VISION_200T_5274` | 1.162 @ 11.0 | 0.162 / 18.0 | **0.1998 / 17.8** |
| `KODAK_VISION2_200T_5217` | 1.110 @ 13.7 | 0.110 / 18.0 | **0.1511 / 12.6** |
| `KODAK_VISION2_500T_5218` | 1.014 @ 7.7 | 0.014 / 18.0 | **0.0315 / 16.9** |
| `EASTMAN_EXR_50D_5245` | 1.048 @ 12.9 | 0.048 / 18.0 | **0.0761 / 12.4** |
| `EASTMAN_EXR_100T_5248` | 1.069 @ 12.9 | 0.069 / 18.0 | **0.1014 / 13.0** |
| `KODAK_VISION_500T_5279` | 1.420 @ 15.1 | 0.420 / 18.0 | **0.5562 / 12.3** |
| `EASTMAN_EXR_200T_5293` | 1.065 @ 15.9 | 0.065 / 18.0 | **0.1326 / 9.1** |
| `KODAK_VISION2_250D_5205` | 1.032 @ 14.6 | 0.032 / 18.0 | **0.1552 / 7.0** |
| `KODAK_EKTAR_100` (T2) | 1.183 @ 9.7 | 0.130 / 16.0 | **0.2260 / 20.1** |

All thirteen now reproduce their sheet to **< 2 × 10⁻³ in level and < 0.2 c/mm in
position**, pinned in `verify.py`.

⚠ **AND IT DISCHARGES C19's OTHER COMPLAINT.** C19 said `adjacency_um` "disagrees with the
traced overshoot frequency on all four stocks checked", and 5231's comment estimated "a
spatial scale of order 100–200 µm" against a stored 16.0, flagging the field and changing
nothing because the renderer's definition was unknown. It is known now
(`f_peak = 206.07 / adjacency_um`, pinned by C2c) and the answer is **32.1 µm** — neither the
naive 1/f reading nor the stored value. The flag is discharged.

### 2.3 ⚠ Five records are REFUSED, and refusing them is the load-bearing part

On `5201`, `5274`, `5217`, `5218` and `5279` the **red** curve's maximum sits on the **first
traced sample** (2.4–2.5 c/mm). The peak is outside the drawn range, so the trace gives a
bound, not a measurement. Solving anyway returns `adjacency_um` of **74–84 µm on every one
of them** — a clean signature of an unresolved peak, and exactly the artefact that would
have entered the database as a "systematic red edge effect" if the five had been solved
along with the rest. Each stock is solved on its **governing** record only: the single
curve on a mono sheet, green on a colour sheet. A guard now asserts that **no stock carries
an `adjacency_um` in 70–90 µm.**

### 2.4 The eleven that still render nothing, each with a reason

| stocks | disposition |
|---|---|
| `ILFORD_HPS`, `SVEMA_LN_9`, `SVEMA_LN_9S`, `SVEMA_LN_8`, `SVEMA_DS_5M`, `SVEMA_CNL_32`, `SOVIET_PANCHROM_1939`, `EASTMAN_ORTHO_1930` | `adjacency = 0.02`, an **unevidenced placeholder**. No source behind any of them prints an edge effect, and no class value can be borrowed: the only two B&W stocks here with a measured adjacency solve to 0.069 and 0.300 — a **4.3× spread** on two films of one maker, one era and one process, which is not a class (method rule 18). Kept as the labelled placeholder; the conservative consequence (no rendered overshoot) is recorded, not tuned away. |
| `FUJI_SUPER_F125_8532`, `FUJICOLOR_SUPER_F500_8572` | amplitude **measured** (+9.0 %, +0.9 %), peak frequency **unresolved** — both traces begin at their maximum. Two unknowns, one measurement. The bound is `adjacency_um ≥ 101 µm` and `≥ 108 µm`; solving on a boundary sample would place the lift essentially at DC, so it is not done. |
| `GEVACHROME_605` | ⚠ **unconfirmable from its own source, not merely unconfirmed.** Bild 1's ordinate runs **10–100 %**, so an overshoot above 100 % is off the top of the frame whatever the film does. The G2 trace replaced f50 and q and left this pair untouched, which was right and was not stated. |
| `AGFA_VISTA_200` | a resolved interior peak (+11.7 % at 3.4 c/mm) that is **not** solved, because the solve runs against the stock's **stored** rolloff and its stored f50 triple is the class estimate 56/63/69 against a measured 50.0. Solving here would bury an unadopted f50 inside a "measured" adjacency. |

---

## 3. A2 / C16 — supersampling refused on two independent grounds, and the residual put in closed form

C16 closed by refusing all three of its options and naming (a) supersampling "the only
correct fix". A2 was to measure the single-thread cost and implement it in both engines if
it held. **It does not hold.**

### 3.1 It is not correct

Blurring and sampling **commute** for a band-limited signal. Multiplying the DFT of an
already-sampled image by the analytic transfer *is* the right discretisation — which is
what C16 itself concluded when it refused option (c). Supersampling does not recover the
sub-pixel density field the film had; it blurs whatever detail the **interpolator invented**,
and makes the answer depend on the choice of interpolator. The way to render more sub-pixel
truth is to render the whole chain larger, which the plugin already does.

### 3.2 The cost, measured single-thread

`AlgoGaussianBlurPlaneWrap` as shipped, 1920 × 1080, three channels, `-O2 -march=native`,
pinned to one core:

| | sigma | taps | ms/frame | vs native |
|---|---|---|---|---|
| native | 0.40 px | 5 | **85.9** | 1.00× |
| ×2 upsample | 0.80 px | 9 | **559.4** | **6.51×** |
| ×3 upsample | 1.20 px | 11 | **1573.9** | **18.33×** |

×3 is what it takes to reach the 1.2 px where the two forms agree. So the
correct-*looking* version of this fix adds **≈ 1.49 seconds per frame of single-thread
time to one component of one stage.**

### 3.3 ⚠ What the residual actually is — and "1.2 px" is not a property of the kernel

`FreqGrid.kernel_transfer` (added, **inert**, no render path calls it) builds the C++
kernel's own taps and transforms them, so the parity tooling can now **predict** the
production blur instead of tolerating an empirical tolerance. What it shows:

The divergence is **aliasing**, and it lives **entirely at Nyquist**. A spatial kernel's
transfer is periodic in frequency, so what it applies is the **periodised** analytic
transfer, Σₘ T(f + m); at Nyquist the m = −1 image lands exactly on the m = 0 term and the
transfer is **doubled**:

| sigma | analytic T(Nyq) | kernel K(Nyq) | ratio |
|---|---|---|---|
| 0.60 px | 0.1692 | 0.3379 | **2.00** |
| 0.80 px | 0.0425 | 0.0850 | **2.00** |
| 1.00 px | 0.0072 | 0.0144 | **2.00** |
| 1.20 px | 0.0008 | 0.0016 | **2.00** |

⚠ **The ratio is 2.00 at every sigma. The two forms do not converge above 1.2 px because
the kernel gets better — it never gets better, it is always exactly twice as high at
Nyquist. They converge because T(Nyquist) itself falls to zero, and twice a vanishing
number vanishes.** C16's "~1.2 px" is precisely the sigma at which 2·T(Nyquist) drops below
10⁻³, and nothing about the kernel changes there. Pinned as such.

⚠ **The limit of the closed form is pinned too.** Below about 0.8 px the 4σ truncation and
its renormalisation take over: at σ 0.40 the support is five taps and the periodised
prediction is 8.5 × 10⁻² out; at σ 0.25 it is three taps and 6.0 × 10⁻¹ out. That is why
`kernel_transfer` builds the taps rather than summing images of T.

---

## 4. T3 — three new stocks, and three tracing defects worth recording

`FUJI_PROVIA_100F` (id 172), `FUJICOLOR_SUPERIA_XTRA_400` (173), `FUJICOLOR_PRO_400H` (174),
from AF3-036E, AF3-151E and AF3-176E. Traced by the new `fuji_t3_2026.py`.

**Printed and transcribed** — rms granularity 8 / 4 / 4 at a 48 µm aperture, resolving power
60+140 / 50+125 / 50+125 lp/mm, cellulose triacetate base at 127 / — / 122 µm, ISO speed,
process. **Traced** — characteristic curves (rms 0.003–0.024 D) and MTF (f50 39.8 / 57.9 /
51.9 c/mm with q 3.50 / 2.62 / 2.17). **Refused** — spectral sensitivity on all three: two
of the sheets scale that ordinate with a bracketed arrow marked "1.0" rather than a numbered
ladder, and a peak-normalised `log_s` built on a misread bracket would be wrong by a factor
while looking entirely plausible.

⚠ **THE CALIBRATION PROBLEM THESE SHEETS POSE.** On PROVIA and PRO 400H the axis labels are
**outlined text**, not text — `get_text()` returns 149 and 83 words and not one is an axis
number — so the label-centroid method used on every Kodak sheet is unavailable. The panels
are calibrated from **their own gridline ladders** instead, and validated by the fact that
Fuji draws these H&D panels **square**: one decade of log H is the same distance as 1.0
density. The two axes are calibrated independently and compared — **0.4 %, 0.4 % and 0.2 %**.
Nothing is adopted from a panel that fails that check, and PROVIA's first attempt did fail
it, at 135 %.

Four defects were found and fixed while getting there, each of which produced **plausible
numbers**, which is the only reason they are worth writing down:

1. ⚠ **A gridline threshold that hides gridlines.** A rule crossed by the panel's caption
   block is broken into shorter runs; at a 0.72 threshold five of PROVIA's nine horizontal
   rules vanish and the ladder fits three, giving 158.8 px per density against 373.8 per
   decade. The square check caught it. 0.50 recovers all nine.
2. ⚠ **A section heading that passes for a gridline.** The green heading bar above PROVIA's
   panel is a long horizontal rule 160 px above D 4.0 — 0.86 of a step, too close for an
   outlier test. It is rejected geometrically instead: the vertical rules' **longest run**
   is the frame, and anything outside it is not a gridline.
3. ⚠ **Record labels as stepping stones.** PRO 400H writes "Blue", "Green" and "Red" in the
   gaps *between* its three black curves, and the walker climbed the word "Red" onto the
   green record and returned green's D 0.69 toe as red's. Dropping connected components
   narrower than 150 px removes the glyphs; every real curve is one component spanning the
   frame.
4. ⚠ **A fitted shoulder where the data has none.** These panels stop at logH +0.3 to +0.7,
   inside the straight line, so a free `shoulder_x` settles wherever the trace ends —
   1.16 for SUPERIA's red against 0.27 for its green, extrapolating to a Dmax ladder of
   2.68 / 2.57 / 3.00, **red above green** on a film whose own traced curves put red lowest
   at every exposure. `digitize_plot.fit_tonecurve4` was added to fit four parameters and
   declare the shoulder at the family default; the ladder is restored to 3.08 / 3.64 / 4.12
   and a guard now asserts it on both new colour negatives.

⚠ **AND THE ONE THING THE TWO 400-SPEED FILMS DO NOT SHARE.** SUPERIA X-TRA 400 and PRO 400H
print **identical** image-structure numbers — rms 4, 50 and 125 lp/mm — because Fuji rounds
the rms to one digit. Their traced curves are different films: PRO 400H's Dmin ladder sits
0.02–0.28 D higher and its gammas are lower (0.614 / 0.572 / 0.543 against 0.662 / 0.705 /
0.751). The profiles are not collapsed onto one set of numbers.

Ids 172–174, appended; no existing ListBox index moves.

---

## 5. T1 and T2 — one measurement, one citation, one documented absence

**T2 → `KODAK_EKTAR_100` gains the first measured MTF for a STILL colour negative in this
database.** E-4046 (2016) page 4 draws panel E4046D as three vector paths; traced over
2.5–80.7 c/mm. ⚠ **The estimate it replaces was 1.5× too sharp on every record**: stored
74.3 / 80.0 / 87.6 against a measured **35.5 / 52.7 / 54.8**, with the layer order coming
out R < G < B as `MTFSpec`'s docstring predicts. That is the same direction and nearly the
same size as the error found on 5285 (1.95×), 5222 (1.33×) and 5231 (1.45×) — the
estimating **rule** showing through again, not this profile. q 3.10 (the film's own green
fit) and the A4-style adjacency solve are adopted with it.

⚠ **"The world's finest grain colour negative film" is a GRAIN claim and the old estimate
had let it stand in for a SHARPNESS claim.** The database's rms 5.5 for this stock — the
finest colour negative here — is untouched. Its MTF is simply not exceptional.

**T2 → `KODAK_GOLD_200` gains a documented absence.** ⚠ **E-7022 carries no MTF panel at
all.** All three copies in the corpus were searched page by page; the string "Modulation
Transfer" appears in none of them, and the curve page prints exactly three panels
(characteristic, spectral-sensitivity, spectral-dye-density). The f50 estimate stays, and
the gap now has a reason instead of looking like an oversight.

**T1 → the two VISION3 stocks are cited to their own sheets.** H-1-5219 (March 2022) and
H-1-5207 (March 2026) replace the generic "Kodak H-1 technical data" both rows carried since
they were written, together with the exposure indexes, process and base construction each
sheet prints. ⚠ **T1 closes as a citation and not as a measurement, and the reason is on
record**: pages 3 and 4 of both sheets carry **three embedded images each**, so the
granularity, MTF and spectral panels are rasters and cannot be traced the way the 1990s H-1
sheets were.

⚠ **One construction difference fell out of reading them side by side**: 5219 has an acetate
base **with rem-jet backing** and 5207 has acetate **without**. Two stocks of one family,
and it should be visible in their halation parameters.

---

## 6. E4 — the 1942 book, verified, and one citation was wrong

The owner supplied `Kodak - [1942] - Eastman Motion Picture Films for Professional Use.pdf`.
It is 98 pages, image-only, and was rendered at 200 dpi and OCR'd page by page.

⚠ **THE SUPER-XX HARVEST WAS ALREADY IN THE FILE — MADE ON 2026-08-11 WITHOUT THE BOOK IN
THIS CHECKOUT — AND THE VERIFICATION IS THE WORK.** Every value reproduces. **The PDF page
number does not**: it was recorded as 49 and is 50 (printed page 45), confirmed from the
book's own contents list and the page folio. PDF 49 is Plus-X Type 1231, which is how the
off-by-one arose. Corrected in the profile and in `_RESOLVING_POWER`.

Three facts the re-read added, each of which changes how an existing number reads:

1. ⚠ **The developer, and it is not D-76.** Book page 16: Kodak SD-21 is D-76 plus **6 g
   borax, 8 g boric acid and 0.25 g potassium bromide per litre**, and Kodak's stated reason
   is that it "represents a 'seasoned' D-76" approximating "the partially exhausted
   developers used in practice". **So gamma 0.65, the speeds and the 55 lines/mm are
   seasoned-D-76 figures**, and the book says the formula "is not intended as a
   recommendation for commercial laboratory use". Any future comparison against a D-76
   measurement has to carry that difference.
2. **The 16 mm sibling is a catalogue number, not an emulsion.** Type 5242 prints the same
   speeds, the same SD-21 gamma, the same IIb aim and the same 55 lines/mm, and says it "is
   similar to Super-XX Panchromatic, Type 1232". Its one difference is the base: blue-gray
   acetate against 1232's gray nitrate.
3. **The only quantitative graininess statement in the book**, page 8: Plus-X 1231 is "twice
   as fast as Super-X … yet its graininess is definitely lower"; Super-XX 1232 has "three
   times that of Super-X, yet its graininess is barely perceptibly greater". A narrow
   bracket, and the reason rms 12.0 is kept rather than raised.

**The Plus-X half is recorded and deliberately NOT merged.** The book gives Type 1231 in
full — Kodak speed 240/160, SD-21 to gamma 0.65, IIb 0.60–0.70, 55 lines/mm, gray nitrate —
and all of it is written onto `EASTMAN_PLUS_X_5231` as **context**, reaching no field. ⚠ 5231
is the 1999 acetate emulsion of H-1-5231; 1231 is a 1942 nitrate emulsion. Three generations
and a base change apart, sharing a trade name and three catalogue digits: the exact shape of
the trap this file already records for EASTMAN 5247 (1974 vs 1983), ILFORD PAN F vs PAN F
PLUS and NEOPAN SS (1959 vs 1999). **The measured evidence that they are different films is
already here**: 1231 resolves 55 lines/mm at 1000:1 while 5231's own vector MTF traces f50
41.3 c/mm with the curve still at 24.5 % response at 98 c/mm. A guard asserts the
separation.

No profile is opened for 1231, and the reason is evidential: the book prints its curves as a
raster on a 1942 letterpress scan, so a tone curve would be a trace of a halftone of a
printed plot, and its speed is on the pre-ASA Kodak scale that PH2.5-1960 does not convert by
a constant. The printed numbers are the whole of what the document supports.

---

## 7. The owner's mid-batch document — `31_209.pdf`, and it is new

**Masao TAKANO (高野正雄), Fuji Photo Film Research Laboratories, «写真像の粒状性(第2報)» /
*Granularity of Photographic Image (II)*, J. Soc. Phot. Sci. Japan 31(4) 209–214 (1968).**

⚠ **NOT A DUPLICATE, AND NOT THE TAKANO ALREADY HELD.** `23_13.pdf` is **Kiyoshi** Takano's
review in the television journal; this is **Masao** Takano's own experiment at Ashigara — a
different author, a different journal, an original paper. Byte-compared against all 97 PDFs:
no duplicate. Figure-compared against §23i, §23j and §23k: no overlap.

⚠ **ITS SUBJECT IS A VARIABLE THIS ENGINE DOES NOT HAVE.** One unnamed ASA 100 B&W negative is
brought to the **same density two ways** — `[VTD]` by developing longer at fixed exposure,
`[VE]` by exposing more at fixed development — and the two grain patterns differ. Nothing in
`FilmProfile` or `RenderSettings` distinguishes the routes, so the engine renders one grain
where the measurement finds two.

Fig. 11 is the only panel whose ordinate is a **length**, and it is traced
(`takano_1968_mottle.py`): Expected mottle size **3.98–6.81 µm** across four developers, with
the D = 0 envelopes running [VE] 5.22 → 7.29 µm and [VTD] 3.35 → 4.37 µm. Ordinate calibrated
over five tick labels at 79.54 px/µm, residual 0.017 µm, and independently checked against the
panel's own x-axis rule — **0.2 px, 0.003 µm**, on a quantity the fit never saw.

**The three harvestable results:**

1. ⚠ **Reaching a density by developing longer gives a 36–40 % smaller clump than reaching it
   by exposing more.** Pinned as an open modelling gap.
2. ⚠ **The number that matters to this database.** The paper states the secondary aggregate is
   **5–8× the mean developed grain**, so 3.98–6.81 µm of mottle implies **0.50–1.36 µm of
   grain** — which lands **inside BBC T-101's independently printed 0.59–1.43 µm band**, from a
   different maker, country, decade and instrument. Two documents, one answer. **The 175
   stored `clump_um_g` values have median 13.0, outside both by an order of magnitude.** This
   is the third statement of the C45 finding and the first from a non-BBC source. Still not
   applied: C45 owns that decision and refused it on measured render cost, and this paper
   names no film.
3. Developer ordering by mottle size and by spectrum level, finest first and identical on
   both measures: **para-phenylenediamine < PQ < Monol < MQ**.

⚠ **The paper's own two percentages describe the ENVELOPES, not the markers**, and reading them
onto the wrong object overstates the effect by about two. "[VTD] is 30–40 % smaller" is the
D = 0 lines (traced: 36 % and 40 %); on the density-0.5–1.5 markers the same ratio is 10–28 %,
mean 17 %. "Mottle grows 20–30 % with density" is not reproduced by the markers at all, which
give +4 % to +16 %. Recorded as measured; the disagreement is recorded, not reconciled.

Figs. 3–8 and 10 are **refused for adoption**: their ordinate is an unlabelled instrument unit
with no printed constant, so their level cannot be converted into anything this schema stores.
The ordering survives the missing constant; the values do not.

---

## 8. What is NOT closed

* **C45** — the clump census now has three independent sources agreeing that the stored scale
  is ~5× too coarse. Still the owner's decision; it moves a pixel on 168 stocks.
* **D1 / D2** — the scanner characterisation still needs the owner's step-wedge scan.
  `STEP_WEDGE_REFERENCE.md` supplies the ruler; the measurement is outstanding.
* **The [VTD] / [VE] route dependence** — measured by Takano, unrepresentable by the schema.
  Recorded as a gap, not queued.
* **Spectral sensitivity for the three new Fuji stocks** — the panels exist and their geometry
  is located; the ordinate scale is the blocker.
