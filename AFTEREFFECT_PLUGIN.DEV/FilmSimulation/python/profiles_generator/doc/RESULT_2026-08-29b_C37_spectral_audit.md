# RESULT 2026-08-29b — C37: a hand sweep becomes an enforced audit

**Task (queue C37):** *"15 sensitivity panels that became reachable on 2026-08-25 … up to 13 new
sets plus 2 cross-checks on the thinnest well-populated carrier (45 %). Pure yield, sources on
disk, two readers already built."*

**Outcome:** ⚠ **no new spectral data at all — and establishing that, before doing any work, is the
result.** What was delivered instead is a guard: `spectral_vector.py`'s registry goes from **4
sheets to 11**, every cross-check is pinned, and `--assert` now fails on drift. Two disagreements
nobody had recorded came to light, one recorded diagnosis was overturned, and both went to a new
queue row (**C38**) rather than being silently adopted.

**Build after the work:** `build.py` **0 failures / 0 warnings** — the first fully green build in
this project's recorded history, and not by changing any audit (see §5). `verify.py` **420 PASS /
1 FAIL** (the known saturation-hierarchy baseline). Database unchanged at **161 stocks**;
`film_names.txt` md5 `41e0bc5d2c7db82324529e773f2fd5ee`, identical to the file the owner supplied.

---

## 1. The row's premise was inverted, and the evidence was already in the repository

C37 promised up to 13 new sets. Every stock behind its 15 findable panels **already carries one**:

| behind the 15 panels | 14 distinct stocks | spectral set already stored |
|---|---:|---:|
| 5201, 5205, 5217, 5218, 5219, 5231, 5245, 5246, 5248, 5274, 5279, 5293, 8532, 8547 (+7239) | 14 | **14** |

⚠ **And `spectral_vector.py`'s own docstring had said so since 2026-08-26** — *"NONE OF THEM ARE
NEW DATA … every one of the eleven stocks already carried a spectral set, which is worth saying
because this task was scoped on the assumption that they were."* The C37 row was written the same
day and never picked it up. This is the third time in two days that a readiness label has decayed
(F2 says "owner decision" while the tier list files it under "no decision needed"; B1 is closed
while the tier list still ranks it 7th).

⚠ Two smaller corrections fell out of reading the sources rather than the row: the queue's **"both
5205 sheets" is one document** — `5205t.pdf` and `H-1-5205t.pdf` are byte-identical, md5
`edd35d27f840c0803f5b957c18dd9561` — and **5218's panel is on p3, not the p4 the row gives**;
extraction fails on p4 because that page has no sensitivity caption.

---

## 2. What was actually wrong: the sweep was prose, and prose does not re-run

The 2026-08-26 sweep compared eleven panels by hand. Its numbers lived in a docstring. **Not one of
those sheets was in `SHEETS`**, so nothing re-ran the comparison and a change to any reader could
have moved any curve without a single test noticing. That is the same class of defect this project
has caught twice before — a guard that cannot fail (C20) and a census that counted the wrong thing
(the 2026-08-27 provenance audit).

| | before | after |
|---|---:|---:|
| sheets in the extractor's registry | 4 | **11** |
| cross-checks asserted in the build | 1 | **10** |
| panels whose failure cause is recorded | 2 | **7** |

---

## 3. The comparison itself was wrong, and fixing it moved the answers

The old rule compared every sample both readings called measured. That **includes the one or two
samples where the shorter trace is diving into its own floor** — which measures where each reader
stopped drawing, not the film. `_core_rms` now guards one sample in from whichever measured run
ends first, at each end.

Nothing about either reading changed, and yet:

| | old estimator | `_core_rms` |
|---|---:|---:|
| 5218 red | 0.367 | **0.241** |
| 5217 (the pinned triple) | 0.109 / 0.091 / 0.049 | **0.077 / 0.086 / 0.047** |

⚠ A pinned number that moves when you fix the estimator was never measuring what its name said.

---

## 4. Eight agree, three do not, and one recorded explanation does not survive

**Agreeing, at core rms ≤ 0.086 decades:** 5201 (0.002/0.002/0.003 — that one compares the profile
with itself and so measures the literal's rounding), 5205 (0.030/0.047/0.047), 5217
(0.077/0.086/0.047), 5222 (0.003), 5246 (0.029/0.050/0.064), 5274 (0.041/0.070/0.065), 5279
(0.056/0.073/0.034), and 7239 which has no independent set to compare against.

**Not agreeing:**

⚠ **5245 blue, 0.335 — and the docstring's explanation is wrong.** It said the cause was
"comparing a TRUNCATED trace against a complete one after per-layer peak normalisation".
Re-normalising both sides on their shared span changes the number **by nothing**, because both
maxima already lie inside it. Read sample by sample the two agree to **±0.06 decades from 400 to
480 nm — the entire peak** — and diverge only on the 490–520 nm tail. And the *stored* half is the
suspect one: −0.60, −1.15, −1.80, −2.45, −3.10 at 490/500/510/520/530, i.e. steps of 0.55, 0.65,
0.65, 0.65. **That is a straight line**, which a dye sensitivity tail is not, while the drawn curve
rolls off faster and stops at 520. The stored tail below 490 nm looks extrapolated, not read.

⚠ **5218, 0.241 / 0.210 / 0.138 — not recorded anywhere before.** Not truncation: over the core the
traced curve is systematically **higher on every rising flank** (+0.13 to +0.26) and **lower on
every falling one**, on all three layers. A consistent narrowing on every layer is a
wavelength-scale difference or a genuinely different reading, not noise.

⚠ **5231 pan, 0.213.** A panchromatic curve has two maxima — blue near 400 nm, red near 590 — and
this emulsion's are a quarter of a decade apart. The raster reading makes the 400 hump the peak;
the vector trace makes them equal and puts argmax at 590. Both agree on the shape; they disagree
about which hump the normalisation hangs off.

⚠ **5231 also needed the right criterion, and getting it wrong would have manufactured an error.**
H-1-5231 prints **two** curves, `D=0.3 Above gross fog` and `D=1.0 Above gross fog`, and the
adopted set's criterion string is `..._D0.3_above_gross_fog`. Reading the 1.0 curve would have
compared two different measurements of the same film and called the difference a defect. Note the
sheet's spelling differs from 5222's, so the caption is matched per sheet rather than by one
constant.

**Nothing was re-adopted.** Choosing between a vector trace and an adopted raster reading is the
call XX1 made deliberately with its evidence laid out; an audit scoped as a cross-check is not the
place for it. All three are pinned at their measured disagreement and raised as **C38**.

---

## 5. Five panels are findable and not extractable, with causes measured

Recorded in `UNREACHABLE` so the next reader does not re-derive the diagnosis:

| panel | cause |
|---|---|
| **5248 p3, 5293 p4** | 17 long paths each and **not one coloured** — these sheets draw all three curves in BLACK, so the ink convention says nothing about them. The 7239 problem with three curves instead of one; `extract_mono` reads a single trace. ⚠ Blocked on **method**, not source — the closest thing to reachable new work on this list |
| **5219 p3** | no path with 8 or more segments at all; the curves are short strokes or outlined art |
| **8532 p1** | Fuji layout, 3 page images, only 5 long black paths |
| **8547 p1** | 24 page images — the panel is **raster**, and its stored set came from a raster reading anyway |

---

## 6. The build went green, and not by touching an audit

`vision3_granularity.py` and `kodak_still_curves.py` had been reported as failing. ⚠ **Both were
missing PDFs, not defects** — their guard file differs from the file they actually open, so they
fail where they should skip. Proven by re-running both against the pristine `1_python.zip`, where
they fail identically. Staging the sources fixed both.

⚠ **An earlier claim in this session that `build.py` passed wrong arguments to
`vision3_granularity.py` was mistaken and is retracted here**: it passes `--pdfdir` correctly, and
the failure was always the absent sheet.

Staging also unblocked `mtf_vector.py`, `kodak_sensitometry.py`, `kodak_time_gamma.py` and
`spectral_vector.py`, which had been skipping. **11 audits now run and all 11 are green.**

---

## 7. One constraint I broke and repaired

`agfa_2004_curves.py` (from E1, earlier the same day) imported `scipy.optimize` for its
six-parameter curve fit. ⚠ **`README.md` states the project's dependencies as "numpy and Pillow
only — no OpenCV, no SciPy".** SciPy is now an optional, audit-only import: without it the
re-fit is skipped **and said to be skipped**, while all 12 spectral panels, the dash legend, the
steepest-chord gammas and the sharpness overshoot still run. The adopted values are unaffected —
they are already in `film_profiles.py` and the fit only re-derives them. `README.md` now states the
audit dependencies precisely instead of implying the render rule covers everything.

---

## 8. Files

**Changed:** `spectral_vector.py` (registry 4 → 11 sheets, `_core_rms`, `UNREACHABLE`,
`MONO_NOTE`, folder-aware source paths, corrected docstring), `agfa_2004_curves.py` (SciPy made
optional), `doc/DIGITIZATION_QUEUE.md` (C37 closed, **C38** and **B4** opened, E2 reconnoitred, tier
table corrected), `doc/PROGRESS.md` (build state, audit list, `cpp_parity` now green, dated change
block), `doc/README.md` (dependency claim).
**New:** this document.
**Staged into the corpus:** the C37 sheets plus the sources that unblocked four skipping audits —
copies of the owner's originals, none modified.
