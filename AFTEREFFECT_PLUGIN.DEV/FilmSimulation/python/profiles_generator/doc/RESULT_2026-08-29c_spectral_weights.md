# RESULT 2026-08-29c — spectral_weights: a report that misread itself, and a parity break underneath it

**Origin.** Not a queue row. The owner read `FilmActiveProfiles.md`, saw the *Spectral
Sensitivity* cell for `AGFA_APX_100` and `AGFA_APX_400` printed red-and-estimated, and asked why —
Agfa's own datasheets carry those curves at good resolution, and they are on disk as
`PROFILES/AGFA/apx100.pdf` and `apx400.pdf`.

**Short answer to the question as asked: the curves were never ignored.** Both were vector-traced
on 2026-08-17 from p2 of their own sheets, closing to **0.50 nm and 0.0034 log**, 312 sampled
points resampled to the 10 nm grid. What was red was a *different parameter with a confusable
name*.

**The longer answer is the result.** Chasing that one red cell turned up a Python/C++ divergence
that had been shipping in every monochrome render, 48 provenance records that asserted a
derivation nobody had performed, and a guard measuring itself against a number it could not reach.

---

## 1. Two things named "spectral sensitivity", and the report printed the wrong one

`FilmActiveProfiles.md` column 6 is headed **Spectral Sensitivity**. It prints `spectral_weights`
— the three-number triple that collapses scene RGB onto one silver record. The digitised
**curve** is four columns right, under *Spectral Response Curves*, where APX 100 and 400 have read
`33x1 pts, 380-700 nm @10`, plain, all along.

So the file was answering a question nobody asked, under a heading that promised the other one.

⚠ **And the value it printed was not the value the renderer used.** The file's own preamble says
every cell carries *"the actual value the simulator uses"*. For the **24** monochrome stocks that
carry a traced pan curve, neither engine reads `spectral_weights` at all — both integrate the
curve. The column printed an unused literal and marked it red, on precisely the stocks whose
evidence was strongest.

Both halves are fixed. The column now prints the derived triple, suffixed `derived`, marked from
the curve's own provenance; the legend states the naming trap outright.

| | before | after |
|---|---:|---:|
| APX 100 cell | <span style="color:red">0.280/0.560/0.160*</span> | 0.261/0.343/0.396 derived |
| APX 400 cell | <span style="color:red">0.280/0.560/0.160*</span> | 0.251/0.336/0.413 derived |
| plain cells in that column | 48 (all false, §3) | **24** (all derived, all true) |

---

## 2. ⚠ The finding that matters: the two engines had been rendering different monochrome images

`Algo_07_Sim.cpp` case 2 calls `AlgoSpectralMonoWeights()` **unconditionally**, and always has. It
falls back to `profile.spectral_weights` only when the stock carries no curve. `film_sim` gated the
identical derivation behind `RenderSettings.spectral_mono`, which defaulted to **False**.

For the 24 stocks with a curve the plugin derived and the reference renderer did not. Both ran.
Both looked plausible.

| stock | Python used (stored) | C++ used (derived) | blue |
|---|---|---|---:|
| `KODAK_PLUS_X_125` | 0.300 / 0.590 / 0.110 | 0.205 / 0.292 / 0.502 | **4.6x** |
| `KODAK_TRI_X_400TX` | 0.300 / 0.590 / 0.110 | 0.253 / 0.322 / 0.425 | 3.9x |
| `AGFA_APX_400` | 0.280 / 0.560 / 0.160 | 0.251 / 0.336 / 0.413 | 2.6x |

Nothing was looking. `cpp_parity.py` audits the grain, MTF and reciprocity laws;
`interimage_parity.py` audits the DIR couplers. Stage 7 had no probe, and every visual check
compares one engine against itself.

⚠ **`verify.py` was actively defending the split.** Its check *"no basis-projected spectral
derivation is enabled by default"* asserted `spectral_mono is False`. It read as caution about a
model assumption; what it actually pinned was one engine's disagreement with the other. A guard
holding one implementation to a decision the other never implemented is not caution.

**Fixed:** `spectral_mono` defaults to `True`. The `verify.py` check is rewritten to assert the
invariant that survives — mono collapse derives in **both** engines, the basis-projected **taking
matrix** stays out of the pipeline (it would stack a third mixing stage on `dye_matrix` and
`InterimageSpec`).

**Guarded:** new audit `spectral_mono_parity.py` compiles the plugin's own
`AlgoSpectralSensitivity.cpp` against the real database and compares every monochrome stock.

> `67/68 agree exactly` — 24 derived, 43 both-decline, worst |Δw| below 1e-16.

The owner's earlier reasoning against adopting these numbers is **not withdrawn**: the derivation
depends on the assumed primary lobe width (55 nm), which is a convention, and the honest fix is a
scene spectral model. What changed is the recognition that "OFF" was not buying that caution — it
was buying a silent split while the shipped plugin derived anyway. One assumption applied in both
engines beats the same assumption applied in one.

---

## 3. ⚠ 48 records claimed a derivation that had never been run

48 profiles carried `spectral_weights` with `status='derived'`, `conditions='integrated from the
traced log-sensitivity curves'`, and the marking that goes with it: **plain**.

**Every one of them stored (0.30, 0.59, 0.11)** — the `FilmProfile` dataclass default, which is
Rec.601 video luma. Nothing had been integrated from anything.

Two mitigations, stated so this is not overclaimed: all 48 are **colour** stocks, where the field
is read by nobody (`if profile.is_monochrome`), so no frame ever rendered wrong; and the value is
inert rather than incorrect. It was a provenance failure, not a rendering one. But it is the same
shape as the 2026-08-27 audit that found 22 cells claiming "documented" for an estimate, and the
same shape as C37's stale row: **a label decays, and only opening the document shows it.**

A second false note, on 113 records: *"No traced spectral sensitivity for this stock"*. Untrue for
**28** of them, `AGFA_APX_100` and `AGFA_APX_400` among them. That sentence is what the owner's
question was really about.

**Fixed by rule, not by hand.** `_PARAM_SOURCES_DERIVED`'s header says *"REGENERATE, do not
hand-edit: the rules live in the task EM-A6 generator"* — ⚠ **and that generator is not in the
repository.** For eleven months the instruction has pointed at nothing, which means the only way
to touch those records was the way it forbids. `spectral_weight_provenance.py` is the missing half
for this one parameter: the rule, runnable, and asserted by the build.

| case | rule | stocks |
|---|---|---:|
| A | mono + traced curve + guard passes → `derived`, prints the derived triple | **24** |
| B | mono + traced curve + guard refuses → `estimated`, refusal cause recorded | **1** |
| C | mono, no curve → untouched, the existing note is true | 43 |
| D | colour → `estimated`, recorded as **inert**: never read for a 3-layer stock | 93 |

118 records rewritten. The report's plain-cell count in that column falls 48 → 24, and every one
of the 24 is now a value the renderer actually applies.

---

## 4. ⚠ The gamut-reach guard was measuring itself on the wrong grid

`spectral_out_of_reach()` and `spectral_peak_lambda()` read the curve off `spectral_grid()`, which
stops at **730 nm** because that is the renderer's integration domain. The guards' entire job is to
detect sensitisation *outside* that domain.

`KONICA_INFRARED_750` stores samples to 830 nm:

| | on the clipped grid | on the curve's own samples |
|---|---:|---:|
| peak | 730 nm (the grid's last sample) | **750 nm** |
| energy beyond 700 nm | 0.203 | **0.437** |

The guard refused the stock either way, so nothing rendered wrong — but it refused on a figure low
by a factor of two and a peak that was an artefact. That is a threshold compared against a
quantity that cannot reach it: the third time this project has caught that shape (C20's guard that
could not fail; the census that counted the wrong field; the 2026-08-26 sweep no test re-ran).

New `film_sim.stored_layer_sensitivities()` reads the stored samples unclipped; both guards use it.
A `verify.py` check now pins 750 nm / >0.40 so the clipping cannot come back.

---

## 5. Two things found, not fixed, and opened as rows

**C39 — `ROLLEI_INFRARED_400`: the database cannot tell a filtered question from an unfiltered one.**
The guard does not refuse this stock and ⚠ **should not be tuned until it does.** Its stored curve
is the *unfiltered* sensitisation: peak **410 nm**, only **0.028** of its energy past 700 nm. By
the data on file it is an ordinary panchromatic emulsion, and no honest out-of-reach test can call
it otherwise. Its authored red-dominant (0.52, 0.20, 0.28) encodes an assumed deep-red/IR **taking
filter that no field in the profile records**. Both engines now derive (0.349, 0.315, 0.336) for
it — right for the data, wrong for the way the film is used. The fix is a `taking_filter` carrier,
not a threshold. Lowering the threshold to catch it would start refusing ordinary pan stocks.

**C40 — the gamut-reach guard exists only in Python.** `AlgoSpectralMonoWeights()` has no peak test
and no out-of-reach test; it derives for any stock carrying `log_s_pan`. So `KONICA_INFRARED_750`
renders in the **plugin** with

> `cpp = (0.1611, 0.1931, 0.6458)` — blue-dominant, against the authored and correct red-dominant
> `(0.55, 0.15, 0.30)`

⚠ **This is a live rendering defect in the shipped C++, on one stock**, and it is an *algorithm*
change to fix, not a data one. It is not fixed here because the algorithm sources were not in
scope for this task. `spectral_mono_parity.py` accepts it under `--allow-guard-gap` **and names it
in its own `[OK]` line on every single run**, so accepting it cannot quietly become forgetting it.
The C++ side needs the same two tests the Python side has, ~20 lines, and it would then fall back
to `profile.spectral_weights` exactly as Python does.

---

## 6. What the owner decided

Three questions were put before any file was edited, because each changes rendered output.

| question | chosen | rejected |
|---|---|---|
| adopt the derived weights how? | **flip `spectral_mono` ON, leave the database literals alone** | baking the triples into `film_profiles.py` — permanent, unswitchable, and it would hard-code the lobe-width assumption into the data |
| the 48 false labels? | **correct them to `estimated`, recorded as inert** | deriving a number for a field nothing reads |
| the guard hole? | **measure out-of-reach on the stored curve** | widening `spectral_grid()` to 830 nm (changes every colour integral) or a hand-maintained IR exclusion list |

The rejected first option was what the task was originally asked as. It was worth not doing: the
run-time path gives the identical numbers, stays reversible, and keeps the provenance honest,
because a derived value living in code can be re-derived and a literal cannot.

---

## 7. Build and verify state — read this before quoting it

`verify.py` **422 PASS / 1 FAIL** (the known saturation-hierarchy baseline). The
`spectral_mono is False` check did not survive; it was replaced by two checks asserting the new
invariant, plus one pinning the guard's measurement domain.

Database **unchanged at 161 stocks**; `film_names.txt` md5 `41e0bc5d2c7db82324529e773f2fd5ee`,
identical to the file the owner supplied. Ordering identical across all four representations.
No `spectral_weights` **value** changed; what changed is which value the renderer reads, and what
the records say about it.

⚠ **`cpp_parity.py` reports three failures, and they are NOT from this work — they were recorded
on 2026-08-27 and are unchanged.** `PROGRESS.md` already carries them verbatim, including the same
`1.83e-01` at `('S', ILFORD_HPS, 0, 3)`: `Algo_11_Sim.cpp` and its AVX2 twin no longer carry the
`ampScale` marker, so `AlgoAddGrain` computes `sqrt(max(D - dmin, 0) + fog)` and returns
`sqrt(1 + fog_grain)` — 1.0392 to 1.1832 — where it must return exactly 1.0 at net density 1.0.
That is queue **C30/C33**, the recorded `FilmGrainSigma` bypass. Re-confirmed today two ways:
the pristine `1_python.zip` copy fails identically against the same tree, and nothing in this task
touches `GrainSpec`, `dmax` or the grain path.

⚠ **And a correction to how this project has been reporting build state.** The
"0 failures / 0 warnings" recorded on 2026-08-29b was run against a root that held no algorithm
sources, where `cpp_parity` **skips** the twin-consistency, law-reachability and grain-stage
probes and exits 0. Verified: with `--root` pointing at an empty directory the audit prints five
`[SKIP]` lines and returns success. The build was green partly because three checks were not
looking. Today's run uses a root carrying the algorithm tree, so those probes execute — which is
why a *previously recorded* failure appears in a build that had been reporting clean. Nothing
regressed; the instrument was widened.

The two new audits both pass:

```
[OK] spectral_mono_parity.py   -- 67/68 agree exactly; 1 ACCEPTED GUARD GAP STILL OPEN:
                                  KONICA_INFRARED_750 ... (queue C40)
[OK] spectral_weight_provenance.py -- 161 records reproduce the rule
```

---

## 8. Files

**Changed:** `film_sim.py` (`stored_layer_sensitivities()` new; both guards read it;
`spectral_mono` default True with the rationale rewritten rather than appended),
`gen_active_profiles.py` (column prints what the renderer uses; legend states both traps),
`verify.py` (the `spectral_mono is False` check replaced by three), `film_profiles.py` (118
`spectral_weights` ParamSource records, rewritten by rule), `build.py` (two audits registered),
`doc/FilmActiveProfiles.md`, `doc/FilmCurves.md` (regenerated), `doc/DIGITIZATION_QUEUE.md`
(**C39**, **C40** opened), `doc/PROGRESS.md`, `doc/NotFound.md`.

**New:** `spectral_weight_provenance.py`, `spectral_mono_parity.py`, this document.

**Not changed, deliberately:** every `spectral_weights` literal in the database; the algorithm
sources (C40 is left open rather than patched out of scope).
