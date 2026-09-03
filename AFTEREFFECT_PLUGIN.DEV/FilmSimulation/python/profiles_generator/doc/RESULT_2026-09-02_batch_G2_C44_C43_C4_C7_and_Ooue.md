# RESULT 2026-09-02 — one batch: G2, C44, C43, C4, C7, then Ooue 1959 Parts 1 and 2

Owner-approved batch, run without pause. Build **green: `OK -- 0 failures, 0 warning(s)`**,
verify 494 PASS / 1 baseline FAIL, compile clean on all 18 TUs, `doc_consistency.py` 31/31.

Seven rows closed. ⚠ **Two of them closed by discovering that the row itself was wrong**, and one
closed with **no code change at all** because the code was already right.

---

## A. G2 — the four raster plot sets of Rens & Van Bets 1968

`PDF/PROFILES/GEVAERT/Rens_vanBets1968Gevachr6.00.pdf`, reader `gevachrome_1968_raster.py`.

### A.1 The blocker was never resolution

G2's row said its blocker was **G5**, an owner re-scan at 300 ppi. It was not. The scan on file is
about 115 ppi and it was enough; what defeated the earlier attempt is that **the sheet curls at its
right edge**, so Bild 1's abscissa decade width measures

| span | px per decade |
|---|---|
| 2 → 10 c/mm | 174 |
| 10 → 100 c/mm | 143 |
| 60 → 100 c/mm | ≈ 99 |

A single fitted log scale therefore puts 100 c/mm off the panel, which is exactly the symptom
recorded ("decade width 154.8–181.0 px, extrapolation lands off-panel"). **Interpolating piecewise
between the nine printed gridlines removes the problem instead of arguing with it.**

⚠ **The ordinate does not curl, and that is the check.** All three MTF panels independently return
**176.5 px per decade** of modulation, and the six labelled levels fall on it to under a pixel —
which is what a sheet curling about a *vertical* axis does, and is the reason to trust the abscissa
anchors rather than distrust the whole figure.

### A.2 Adopted

| figure | what | result |
|---|---|---|
| Bild 2a / 2b p262 | spectral sensitisation, both types | 6 curves, 380–700 nm at 10 nm, peaks 430 / 570 / 660 nm |
| Bild 4 p262 | image-dye absorption | 3 curves, peaks 2.00 @ 450, 2.04 @ 530, 1.98 @ 670 nm |
| Bild 1a/b/c p260 | MTF in green / red / blue light | 6 curves; f50 and rolloff below |
| Tab. IV + §3.1 p264 | the twelve-step 30 °C reversal chain | `_PROCESSING` on both stocks |
| Bilder 7a/7b p264 | interimage | measured, **not stored** — see A.5 |

### A.3 ⚠ The MTF replaced class estimates that were two to three times too high

| | red | green | blue |
|---|---|---|---|
| **GEVACHROME_600** measured | **20.4** | **23.5** | **44.4** |
| was (class estimate) | 58 | 62 | 66 |
| **GEVACHROME_605** measured | **15.8** | **20.3** | **35.9** |
| was (class estimate) | 50 | 54 | 58 |

That is a large correction on two shipped stocks, so it needs more than "a figure says so". Three
things the tracer was never told come out of the trace:

1. **Blue is sharpest and red softest on both films** — the printed Tab. I layer order, blue on top
   and red at the bottom under the whole pack. The measured blue/red ratio is **2.2**; the class
   estimates had it at **1.14**, which is the signature of a rule rather than a measurement.
2. **Typ 6.05 is softer than Typ 6.00 in every channel**, by 13 / 22 / 19 %. 6.05 is the fast film
   (23 DIN against 18 DIN). Faster stock, bigger crystals, softer image.
3. **One rolloff shape fits all six curves.** Fitting `1/(1+(f/f50)^q)` returns q **1.90–2.12** at
   rms 0.006–0.025. Six independent traces agreeing on one shape parameter is a family property.

⚠ **Each film keeps its OWN median q — 2.09 and 2.00 — and a shared 2.0 was refused by
`verify.py`**, correctly: the suite asserts that no two stocks were collapsed onto one exponent,
because that is how a class rule gets laundered into measured data.

⚠ **L/mm is cycles/mm here**, and the paper says so: the axis reads "Frequenz L/mm" and the text
states the test object had a **sinusoidal** density variation, which has no line pairs to count.
That settles the unit question **queue G6** raises for this document.

### A.4 The two honest difficulties, both asserted rather than hidden

- **Bild 4's cyan and magenta cross at about 420 nm** and merge into one blob at this resolution.
  400/410/420 nm are assigned from the unambiguous branches either side, and the reader asserts
  them. ⚠ Bild 4 draws **one** curve set captioned for **both** types, so the two profiles carry
  the identical arrays — one measurement stored twice, and `verify.py` asserts the identity so no
  future count treats them as two.
- **Bild 1c's dashed curve has gaps wide enough to hide it** between 28 and 38 c/mm, and a
  nearest-ink follower walks off it onto the solid Typ 6.00 curve and **returns 6.00's f50 twice
  without complaining**. The tracer now predicts from the local slope and is forbidden to land on
  the already-traced curve. That defect and its fix are in the reader, because a silent duplicate is
  exactly what the method rules exist to catch.

### A.5 Measured and deliberately not stored

Bilder 7a/7b plot equivalent neutral density against lg i·t for a neutral wedge exposed additively
(A) and through a red filter only (B). The separation A − B on the cyan record is **about 0.15 D at
the foot**. `CouplerSpec.strength` is a dimensionless cross-layer inhibition amount with **no
published calibration against a measured ΔD**, so converting one into the other would be inventing
the conversion rather than reading it. The stored 0.12 is unchanged; the measurement is in the
citation.

---

## B. C44 — the Callier toe, closed with a null code change

`sayanagi_callier.py`, from Sayanagi 1959, «Callier Q Factor と粒状», Canon, *J. Soc. Phot. Sci.
Japan* 23(1) 20–24.

He derives Q from grain optics — base transmittance Ib, a developed grain of **finite**
transmittance Ig (finite because the electron microscope shows it filamentary), circular grains
Poisson-distributed at coverage 𝔅, Savelli's Poisson averages for the mean intensity and the mean
amplitude — and his equation (10) is

    Q_II = 2 (1 − Ig^½) / (1 − Ig) = 2 / (1 + Ig^½)

⚠ **It contains no density at all.** Not D, not the coverage, not the grain radius. On
base-subtracted density Sayanagi's Q is **flat**, so the toe collapse this project had recorded as a
model defect since 2026-09-01 has no mechanism in the one theory that derives Q from first
principles.

With the base left in, `Q_I(D) = [Db + Q∞(D − Db)] / D` is 1 at D = Db and climbs to Q∞. Fitting Db
to the two measured curves **independently**:

| curve | fitted Db | whole-curve rms | toe error |
|---|---|---|---|
| Trumpy/Streiffert Fig. 5, base modelled | **0.045** | 0.019 Q | −0.014 |
| same, shipped fit with no base term | — | 0.156 Q | **+0.491** |
| Mees FIG. 179, five gamma curves | **0.050** | 0.003–0.031 | predicts 1.018–1.072 at D 0.055 against a measured 1.042 |

⚠ **Two laboratories, two decades, two figures, the same base density to half a percent of D**, and
neither figure knows about the other. ⚠ And it explains the one feature of FIG. 179 nothing else
could: **why all five gamma curves are drawn as a single stroke below D 0.25.** Q_I → 1 at D = Db
for every curve whatever its Q∞, so at the toe five emulsions of five different contrasts genuinely
coincide. Under any model in which Q is a film property alone, that shared toe is impossible.

⚠ **Therefore no toe term goes into `film_sim.callier_net`.** Its argument is NET density, the base
is already removed, and a correction fitted to Q_I data would remove it a second time and darken the
shadows on every B&W stock whenever a condenser is dialled in — the exact region C44 was opened to
protect. **`callier_net`, `AlgoCallierNet` and the shared LUT are untouched and bit-identical.**

⚠ This also confirms a convention **C22 had to argue for itself**: his §2.3 names Q_I and Q_II and
§3.2 states that Q_II — base-subtracted — is the rational one. C22 reasoned its way there and
recorded that *no source stated a convention*. One does, from 1959.

---

## C. C43 — `callier_q` stops being a class constant

The shipped E = 0.1471 and β = 1.6746 were fitted to Trumpy's curve with **no base term**, so the
base contamination was absorbed into β. Refitting with the base modelled moves β to **1.809**, and
refitting Mees's five gamma curves with Db held at 0.050 gives

| γ | 0.21 | 0.37 | 0.69 | 1.20 | 1.65 |
|---|---|---|---|---|---|
| β | 1.491 | 1.495 | 1.729 | 1.822 | 1.828 |

⚠ **β RISES WITH GAMMA by 0.34 over the measured range, which one number cannot express.**
`callier_q` is now computed per stock from its own **mid slope**:

    β(γ) = 1 + 0.9706 γ / (γ + 0.2558)

giving **1.64–1.87** across the 68 monochrome stocks, against the undocumented 1.30 / 1.25 it
replaces. Colour stays 1.0 — dye clouds do not scatter, and Sayanagi's model is about developed
silver.

⚠ **The form was chosen for its endpoints, not its residual.** β(0) = 1 exactly (a film with no
developed silver has no Callier effect) and β(∞) = 1.971, just under Sayanagi's own ceiling of 2.
A decaying exponential fits the same five points equally well (rms 0.043 against 0.045) and gets
both endpoints wrong.

⚠ **The ceiling is a real test and the model passes it.** Inverting his (10), Ig = (2/β − 1)², so
β > 2 is impossible. The five Mees curves invert to grain transmittances of **11.7 % at γ 0.21
falling to 0.9 % at γ 1.65** — grains growing more opaque as development proceeds, which is his
assumption (II) and was not fitted.

⚠ **One measurement does not fit under the ceiling and is recorded rather than explained away:**
BBC T-101 Fig. 25 gives Q 2.00–2.34 at a 0.0016 sr collection angle where Q → β, and 2.34 is above
Sayanagi's absolute maximum.

⚠ **Inert at the shipped default** — Callier applies only when `scanner_specular` > 0, which is 0.
⚠ **One consequence measured and recorded:** the steeper law makes the shared Callier LUT's linear
interpolation 8× less accurate (2.2e-07 at q 1.3, **1.7e-06** at q 1.84), which is what raised
`cpp_parity.TOL_CALLIER` from 1e-6 to 1e-5. 1.7e-06 D is three orders of magnitude below one 16-bit
code step; doubling `ALGO_CALLIER_LUT_N` would quarter it at 16 KB per LUT, and that arithmetic is
recorded in the tolerance's own comment for whoever wants it.

---

## D. C4 — and the row named a film that does not exist

C4 read: *"ЦО-90Д / ЦО-90Л — argued against, two documents with near-identical norms would render
identically."*

⚠ **They are not two documents and there is no ЦО-90Д.** Both files under `SOVIET STANDARDS` are
scans of ONE specification: same title block «КИНОПЛЕНКА И ФОТОПЛЕНКА ЦВЕТНЫЕ ОБРАЩАЕМЫЕ МАРОК
ЦО-90Л», same ТУ 6-42-1514-90, same «(Вводятся впервые)», same signatories Калугин and Кислицын,
same 1990 registration stamp — two different physical copies. **«ЦО-90Д» is an OCR misread of Л as
Д on a typewritten page**, and the same OCR performs it inconsistently *within a single file*: the
body of one scan reads "Ц0-90Д" on one line and "Ц0-90Л" four lines later, while the page image says
Л both times.

So the objection dissolved rather than being overruled. One stock, entered:

**`SVEMA_CO_90L`, 170 → 171 stocks.** ТУ 6-42-1514-90 табл. 3, read visually from both scans, which
agree line for line: S ≥ 80 GOST 9160-82 at 3200 K tungsten; sensitivity balance ≤ 2.0; overall
contrast coefficient 1.6–2.2; contrast balance ≤ 0.4; Dmin ≤ 0.25 B; Dmax ≥ 2.0 B in every layer;
resolving power ≥ 75 mm⁻¹ per ГОСТ 2819-84. ⚠ The TU states in words that the **red-sensitive layer
must be the least sensitive** and that blue may equal or exceed green. Base ОТБ-14 triacetate per
ОСТ 6-17-451-83; 16 mm double-perf cine in 15/30/60/120 m and 35 mm perforated still film.
Processing: the twelve-step 30 °C chain of табл. 5, first developer 4.5–6.0 min at 30.0 ± 0.3 °C,
re-exposure by two 100 W lamps at 0.3 m, colour developer 8 min.

⚠ Contrast is a **range 0.6 wide** and the TU never says which layer is steeper, so all three take
the midpoint and the 0.4 a legal coating may spread is documented rather than invented as a split.
⚠ `sigma_shape` is left at the class default rather than copied from ЦО-32Д, whose literal still
carries the pre-F2 triple — copying it would have quietly re-imported the defect F2 removed and
escaped verify's hold-out count.

---

## E. C7 — decided: the default stays still-frame

Honjo 1989 §4 is right. At 24 fps the eye integrates ≈0.2 s ≈ five frames; grain is re-rolled per
frame and is zero-mean, so five independent samples average down by 1/√5 and the shipped amplitude
is ≈**2.24× too strong in playback**.

⚠ **It stays that way on evidence, not taste.** Every granularity figure this engine is calibrated
against — rms through a 48 µm aperture, Wiener spectra, Selwyn constants — is measured on a
**stationary** sample. A default that silently divided them by 2.24 would stop reproducing the
numbers the calibration cites, and every parity test and reference render would then be checked
against a quantity no document states. It is also not reversible by a user who does not know the
rule was applied.

⚠ **No carrier was needed.** Both engines already have `grainScale`, so a host that wants
motion-correct grain sets it to `film_sim.temporal_grain_scale(fps)` = 1/√(fps · 0.2), clamped to
1–8 frames: **0.4564 at 24 fps, 0.4472 at 25**. Nothing was added to `FilmProfile`, no stage
changed, no shipped render moved. `AlgoControl.hpp` carries the finding as item 15 of the
`grainScale` block, and `verify.py` asserts both that the law is computable and that **nothing
applies it**.

---

## F. J1 / J2 — Ooue 1959 Parts 1 and 2, supplied by the owner mid-batch

`JAPAN/22_91.pdf` and `22_38.pdf`, reader `ooue_1959_granularity.py`. Full treatment in
`EMULSION_KNOWLEDGE_BASE.md` §23j.

### F.1 ⚠ The measured Wiener spectrum contradicts the grain model's shape

Part 2 Fig. 26, three **named** samples with stated developer, time and density. Fitting
`P0·exp(−ln2·(f/f_half)^n)` to the falling limbs:

| sample | f_half (lines/mm) | n | rms log10 | pure Gaussian rms |
|---|---|---|---|---|
| Neopan SS / Minidol 20 °C 10 min, D 1.03 | 45.6 | **0.71** | 0.087 | 0.563 |
| Neopan SS / Minidol 20 °C 10 min, D 0.45 | 70.8 | **0.89** | 0.107 | 0.345 |
| Process Plate / D-72 (1:1) 20 °C 4 min, D 0.44 | 140.7 | **1.36** | 0.035 | 0.094 |

`make_grain_field` shapes grain with `h(f) = exp(−(f/f_hi)²)·(1 + clump_gain·exp(−(f/f_lo)²))`, so
the Wiener spectrum it produces is a Gaussian of exponent **2**. Every measured limb is **below**
it, and a pure Gaussian fits three to six times worse. **A Gaussian under-estimates grain energy at
high frequency** — the same defect `MTFSpec`'s docstring already records for the MTF tail, now
measured on the grain spectrum as well.

⚠ **And the result that needs no calibration at all:** rows 1 and 2 are the **same film**, same
developer, same time, two densities — f_half 45.6 against 70.8. **The clump gets coarser as density
rises**, by 55 % in cutoff frequency, while `GrainSpec` carries one clump size per stock.

⚠ **Nothing adopted.** The ordinate is an unlabelled "POWER LEVEL", the abscissa says "LINES/mm"
without defining a line, and the figure is a redrawing rather than Ooue's own plate. Shape is
usable; level is not; and the grain spectrum moves a pixel on all 171 stocks.

### F.2 The autocorrelation refutes the shape from the other side

Part 2 Fig. 24, Neopan S at D 1.04: half-width **3.48 µm**, i.e. f_hi 108 c/mm and `clump_um`
**4.65 µm** under the engine's own law — an independent scale for what Fig. 26 measures in the
frequency domain. ⚠ **Past about 12 µm it goes negative** and stays negative for roughly another
8 µm: an anti-correlated ring, meaning the grains are more evenly spaced than a Poisson field would
place them. A Gaussian autocorrelation is positive everywhere and cannot produce that — **and
neither can Sayanagi's Poisson placement**, which §B rests on. A limitation of both, recorded and
patched into neither.

### F.3 ⚠ The 23_7 ambiguity is settled in the author's own words

Part 2 §4.2.2 is headed 「濃度変化の標準偏差による方法」 — *the method using the* ***standard
deviation*** *of density variation* — and every objective granularity in the paper is built on it:
Selwyn's σ√a, van Kreveld's Δ_m√a, and equations (5) and (6). So `23_7` Fig. 7's ordinate is **rms**
and the "mean-square" in its §3.2 is a translation artefact. On that reading its exponents
0.412 / 0.672 / 0.364 / 0.606 straddle the legacy √ law and the BBC exponent instead of falling
below every source in the corpus. **The reason they were harvested and not adopted is gone**; the
adoption itself belongs to 23_7's own row.

### F.4 Mean grain area falls with density, below the stored floor

Part 1 Fig. 2, Fuji positive film FD-3 (1:1) 20 °C: 1.103 → 0.925 µm² over D 1.4 → 4.0 at 32 min,
1.159 → 0.571 µm² over D 0.2 → 2.4 at 1 min. Equivalent diameters **1.21 down to 0.85 µm**.

⚠ **It falls**, on both development times — the same direction BBC T-101 Table 3 measures from
another laboratory. ⚠ **And it lands below the 1.3 µm floor of all 17 stored `emulsion.grain_um`
values**, which come from one third-party aggregator. This is a positive film, the finest-grained
class there is, so it is a floor rather than a typical value — but it is independent evidence that
the stored range is too coarse. Nothing changed: no profile here is Fuji positive FD-3.

### F.5 ⚠ The empirical half of C45

Part 2 §4.2.1: *where two emulsions of different grain size are coated as two layers Q cannot carry
an important meaning, and* ***in fact*** *measurements of Q on commercially available materials show
no correlation with psychological graininess.* C45 found the disagreement empirically from this
project's own numbers; Sayanagi confirmed it theoretically; **this is the same conclusion measured
on commercial stock, with the mechanism named.** Three independent routes to one answer.

### F.6 Read and deliberately not digitised

Part 2 **Fig. 22** — Selwyn's raw G **rises** with scanning-spot diameter over 20–200 µm while van
Kreveld's corrected Δ_m√a stays flat, which bears directly on this corpus's 48 µm normalisation.
⚠ Not traced: six overlapping curves on a 200 ppi raster, and the frame calibration did not
converge cleanly on either panel. **Recorded as read, with its conclusion, rather than reported as a
number that was not measured.**

---

## G. Two audits that fired, and what they caught

- **`dye_matrix_from_spectra.py`** failed with *"the early emulsions are only 1.37× the later ones,
  under 1.40"*. ⚠ **The guard was right and the classification was wrong:** the two 1968 Gevachrome
  stocks had landed in the "later" bucket, and a 1968 emulsion counted as modern makes forty years
  of dye chemistry look like twenty. Moved to `OLD_STOCKS`, where they belong beside
  GEVACOLOR_NEG_682 — same maker, same decade — the ratio recovers to **1.66**. ⚠ And their derived
  magenta-into-blue, **0.2232**, lands inside the Soviet manufacturing band 0.15–0.25 **without
  being fitted to it**: a Belgian reversal emulsion and four Soviet specifications agreeing on what
  an early magenta dye leaks into blue.
- **`verify.py`** refused a shared rolloff exponent of 2.0 on the two Gevachrome types (§A.3) and
  refused the pre-F2 `sigma_shape` literal copied onto the new Soviet stock (§D). Both were caught
  before they shipped.

---

## H. Net effect

| | before | after |
|---|---|---|
| film stocks | 170 | **171** |
| profiles with spectral dye density | 16 | **18** (17 measurements — Bild 4 serves two stocks) |
| profiles with a spectral sensitivity set | 84 | **86** |
| stocks flagged `mtf_measured` | 17 | **19** |
| ParamSource records | 1511 | **1524** |
| provenance tiers | 84 / 45 / 41 | 84 / 45 / **42** |
| open queue rows | 28 | **23** |
| `callier_q` on monochrome | class 1.30 / 1.25 | **derived per stock, 1.64–1.87** |
| new readers | — | `gevachrome_1968_raster.py`, `sayanagi_callier.py`, `ooue_1959_granularity.py` |

Rows that gained evidence without closing: **C45** (now with its empirical half), **C46**,
**G6** (the sinusoidal-test-object statement settles the unit question for the Gevachrome paper),
**F2b** and **23_7's exponents** (the rms/mean-square ambiguity that blocked them is resolved).
