# RESULT 2026-09-01d/e — Flueckiger et al. 2018, and the eighteen Japanese offprints

Two owner batches, run as one continuous pass. Build green: `OK -- 0 failures, 0 warning(s)`,
verify 493 PASS / 1 baseline FAIL, compile clean on all 18 TUs.

---

## A. Flueckiger et al. 2018 — «Investigation of Film Material–Scanner Interaction»

88 pages, University of Zurich / DIASTOR, v1.1. Reader `flueckiger_2018.py`, registered.

### A.1 Digitised

| figure | what | result |
|---|---|---|
| **§2.8.2 Fig. 16** | Technicolor No. IV transfer dyes, 1949 SAMSON AND DELILAH print, SHIMADZU UV-1800, Ohta PCA | 44 samples 360–790 nm, peak-normalised. Main peaks **460 / 540 / 660** nm and cyan secondary **720** — the report's own printed list, exactly |
| **§2.8.3 Fig. 21** | Dufaycolor réseau, transmittance of the three filter elements | 16 wavelengths × 3 elements, 46 of 48 markers found, 2 recovered |
| **§2.8.3 Fig. 22** | integral transmittance, calculated and measured | 16 markers + 521 points of the Shimadzu curve, 300–900 nm |
| **§4.1 Fig. 61 / Table 3** | measured MTF of eight scanners + sampling resolutions | 7 frequencies × 8 machines, checked against Fig. 62 |
| §2.8.1 Fig. 15 | Callier Q | ⚠ **not counted** — same artwork as Trumpy & Gschwind Fig. 5, already in the corpus |

### A.2 ⚠ The check the Dufaycolor harvest rests on

Figures 21 and 22 are separately calibrated, separately plotted and separately traced.
Recomputing the report's own equation (7) — `integral_T% = 0.28 B + 0.32 G + 0.40 R` — from the
Figure 21 trace reproduces Figure 22's markers to **rms 0.29 transmittance points, worst 0.68**,
across 14 wavelengths, **with no free parameter**. The same inversion recovers the two markers
Figure 21 occludes (red at 560 nm → 8.1 %, green at 640 nm → 9.4 %), and the method is validated
where it can be: at 640 nm a partially visible blob reads 9.7 %.

⚠ **Source defect found: Figure 21's caption says "absorbance"; the figure's own ordinate says
"TRANSMITTANCE %".** The figure is right. A reader who trusted the caption would have stored
`1 − T` as if it were `A`.

### A.3 Written to the database

| target | field | value |
|---|---|---|
| `TECHNICOLOR_THREE_STRIP` | `dye_density` | 44 × 3 samples, `normalisation="peak_1.0"`, tier 2 `traced` |
| `DUFAYCOLOR_1937` | `reseau.filter_matrix` **ParamSource only** | records a second measurement that disagrees; the value is unchanged |
| module level | `_DUFAYCOLOR_RESEAU_T_PCT`, `_DUFAYCOLOR_RESEAU_FLUECKIGER_BANDS` | reference tables |

⚠ **`TECHNICOLOR_THREE_STRIP` had never carried a measurement.** It was one of the nine stocks
`NotFound.md` §1 lists as having no source of any kind; the count is now **8**.

⚠ **Shape only.** Figure 16's ordinate has no scale, no ticks and no label, so the curves are
stored peak-normalised with the bottom axis assumed to be zero absorbance. That assumption
immediately showed its cost: `dye_matrix_from_spectra.py` **refuses** to derive a dye matrix from
the set, because cyan reads 0.4298 into green against an admissible −0.06…0.30, and an unknown
baseline inflates precisely the off-band terms a matrix is made of. The curves are kept — their
peaks are validated — and the matrix is refused. The refusal is recorded with that reason.

### A.4 Schema

**No schema change was required.** Two verify guards had to be made grid-aware, and that is a
genuine fix rather than a relaxation: the peak-band check hard-coded `arange(400, 701, 10)` for
every dye set, which was harmless while all sets shared that grid and **silently wrong** the moment
one did not — an argmax index into a 44-sample 360 nm trace read through a 31-sample 400 nm ruler
reports a wavelength that does not exist in the data. The grid assertion now names its one
exception rather than being weakened.

### A.5 Scanner data — kept out of the film database

New document **`doc/SCANNER_CHARACTERISTICS.md`**. It classifies every parameter as
film-intrinsic / processing / scanner / interaction / general knowledge, carries the digitised
scanner MTF and sampling table, and gives a written verdict in §5:

- ✅ adopt as reference documentation;
- ✅ adopt the one film-side finding — the Callier effect's possible **wavelength dependence**;
- ⚠ **defer** a scanner stage until a ground-truth harness exists (`NotFound.md` row 9), because
  adding free parameters to a system that cannot measure whether they help is the definition of
  overfitting;
- ❌ never add scanner fields to `FilmProfile`.

⚠ **The report publishes no scanner's RGB spectral sensitivity** — §2.8 says so outright, and
Figure 6 is captioned "**Typical** … spectral sensitivities of a color imaging device", a textbook
illustration. So a preset built from it could model sharpness and sampling but **not colour**.

---

## B. The Japanese collection — eighteen offprints, 1938–1963, all read

`PDF/PROFILES/RETRO/JAPAN/`. Full treatment in `EMULSION_KNOWLEDGE_BASE.md` §23i.

### B.1 ⚠ Sayanagi 1959 — the theory C44 was missing

«Callier Q Factor と粒状», Canon, 23(1) 20–24. Q(D) derived from a Poisson model of circular grains
with **finite** transmittance T_g — finite because the developed grain is filamentary under the
electron microscope.

1. ⚠ **"Q_II is the rational Q factor"** — Q must be defined on **base-subtracted** density. This
   project computes Callier on NET density; queue C22 argued that from first principles and
   recorded that **no source stated one**. One does, from 1959.
2. His theory fits at **low** density and fails at high — the exact complement of Silberstein &
   Tuttle, which fits above D 0.3 (rms 0.0087 Q) and fails below (+0.49 Q at D 0.05). **C44.**
3. T_g is estimated from the measured Q at **D → 0**, which turns the toe from a defect into a
   measurement.
4. Q contains the granularity rms but **not** the grain radius, so it ranks samples of equal grain
   size and no others — an independent theoretical confirmation of what **C45** found empirically.

### B.2 ⚠ Ooue 1959 — the aperture behind the 1965 abstract

«粒状性の研究(第1報)», Fuji, 23(1) 7–10. The instrument paper for the microphotometer used in the
JPS 1965 abstract (§23f), by the same author.

**It states the aperture: 200× onto 0.2 mm, i.e. 1 µm at the film.** `jp_jps_1965_269.py` had
bounded it at **< 1.4 µm from the figure's own extent alone**. ⚠ The bound contains the stated
fact — a check on the reasoning, not a coincidence. At 1 µm the circular MTF² at 108 c/mm is 0.97.

Its **Fig. 7** gives granularity–density curves for **four named film/developer combinations at a
stated 10 µm aperture** — Neopan S / D-76, Neopan SS / Microfine, Neopan SSS / Pandol, X-ray /
Rendol. Fitted exponents 0.412 / 0.672 / 0.364 / 0.606, R² 0.965–0.998.

⚠ **Not adopted, on an ambiguity the paper itself creates**: §3.2 defines the ordinate as
**mean-square** and the English abstract says the meter reads **root-mean-square**. On the rms
reading the three named negatives straddle the legacy √ law (0.50) and the BBC exponent (0.40);
on the mean-square reading they fall below every source in the corpus — which is itself the
argument for rms. Ooue's companion papers (本誌 22, 38 and 91, 1959) would settle it.

⚠ **What it does settle**: the √ law is about right for B&W **negatives**, so the measured
0.92–1.28 reversal shape is reversal behaviour rather than a defect in the negative law — exactly
what BBC T-101/2 predicted in words.

### B.3 The Soviet review — checked specifically, and it has no Soviet sensitometry

Yano, «ソ連の写真乳剤技術に関する研究», 23(4) 153–160: a ~180-reference survey of Soviet emulsion
**manufacture**. It fills nothing on `SOVIET_PANCHROM_1939`, `SVEMA_*` or `TASMA_*`. It does give a
crystal-habit law by process, the statement that resolving power is independent of crystal
thickness at equal projected area, the additivity law for multilayer characteristic curves, and the
trail of Russian primary literature a future search must follow.

### B.4 Classification

**Highly useful (2):** `23_20` Sayanagi Callier theory · `23_7` Ooue granularity instrument.

**Partially useful (13):** `23_153` Soviet emulsion review · `22_198` Fujicolor UV sensitivity ·
`22_194` relative spectral sensitivity method · `19_162` colour-temperature response ·
`26_172` tone reproduction vs negative size · `14_106` colour sensitometry ·
`14_133` developed silver grain morphology · `23_1` gelatin's role · `19_160` gelatin solids ·
`15_114` desensitising dyes · `23_174` Ektachrome E-3 handling · `25_183` colour reducers ·
`23_161` silver salts of fatty acids.

**Currently not useful (3):** `1_115` photometric units 1938 · `25_175` and `26_179` diazo paper
sensitometry.

⚠ **No file is recommended for deletion.** By the owner's own criterion — keep historical,
theoretical and methodological material even without immediately usable numbers — all three of the
weakest qualify: the diazo pair are careful spectrosensitometry method papers, and the 1938
photometry lecture is the unit basis every exposure figure in this corpus ultimately rests on.

---

## C. Net effect on the project

| | before | after |
|---|---|---|
| stocks with no source of any kind | 9 | **8** |
| profiles carrying spectral dye density | 15 | **16** |
| ParamSource records | 1509 | **1511** |
| open queue rows | 27 | **28** (C46) |
| reference documents | — | **`SCANNER_CHARACTERISTICS.md`** |

Open rows that gained evidence without being closed: **C43** (Callier β), **C44** (the toe — now
with a functional form), **C45** (clump disagreement — now with a theoretical confirmation),
**C46** (Dufaycolor réseau, new).
