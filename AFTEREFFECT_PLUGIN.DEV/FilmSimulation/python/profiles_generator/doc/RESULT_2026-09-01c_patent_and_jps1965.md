# RESULT 2026-09-01c — US 4,495,277 stored as a class default; JPS 1965 measured and refused

Two owner-approved tasks, run together. One writes to the database, one deliberately does not.

---

## A. US 4,495,277 → `emulsion.sensitization` on the four Agfa B&W stocks

**Approved instruction:** store the patent as tier-3 class-level defaults on the Agfa B&W set,
with the provenance stating plainly that it is inferred from an assignee and a date, not measured.

**Written.** `AGFA_APX_25`, `AGFA_APX_100`, `AGFA_APX_400`, `AGFA_SCALA_200X`:

| field | value | tier | status | confidence |
|---|---|---|---|---|
| `emulsion.sensitization` | `"S"` | 3 | `assumed` | low |

plus the patent's full technical content appended to each `EmulsionSpec.source`, and one
`ParamSource` per stock merged into the existing `_PARAM_SOURCES` block. ParamSource census
1505 → **1509**.

**Refused, with the patent's own words as the reason.**

| field | why it stays empty |
|---|---|
| `emulsion.grain_um` | *"The absolute value of the mean grain size may vary within wide limits"* — then 0.3 to 2 µm, the whole field |
| `emulsion.habit` | *"may assume the known forms, for example, cubic, octahedral or even a combination of tetrahedral and decahedral"* — all four permitted |
| `emulsion.iodide_mol_pct` | never stated; the worked emulsions are iodide-free |
| `emulsion.size_sigma_log` | emulsion A's 15 %-outside-±10 % is one lab batch, not a class |
| speed, fog, D-max | 290 / 100 / 445 is a relative scale; no ASA, DIN or ISO in six pages |

⚠ **Two findings that emerged during the write and were not known when the task was approved.**

1. The patent's demonstration material is **colour, not black and white**. Example 3 mixes
   emulsion A with a dispersed **yellow coupler** and processes the coating in a CD-3-type colour
   developer plus bleach-fix. The grain is an iodide-free AgBr / AgCl / AgBr core-shell at
   10 mol % AgCl — not the AgBrI of a B&W camera negative. The instruction named the B&W set and
   the B&W set is what was written; this is recorded in every note so the choice is visible.
2. **R ≥ 3 in claim 1 is a density ratio, not an aspect ratio** — ripened over unripened, developed
   17 min at 20 °C in the patent's ascorbic-acid developer. An earlier summary in this project
   used the phrase loosely; corrected here.

**Bit-exactness.** `EmulsionSpec` is inert — no stage of either renderer reads it — so the four
entries cannot move a pixel. Build confirms: 493 PASS / 1 baseline FAIL, compile clean.

Full treatment: `EMULSION_KNOWLEDGE_BASE.md` §23e, source **S7**.

---

## B. JPS 1965 10p-A-2 → measured, cross-checked, and **nothing adopted**

**Approved instruction:** invert the crystal-size ↔ granularity law to recover an implied crystal
size per Agfa stock.

**Verdict: the route does not work, and the measurement says so itself.**

New reader `jp_jps_1965_269.py`, registered in `build.py`, green on first full build. It traces
both figures off the raster (the handwritten page's OCR layer is unusable), and rests on the fact
that the page draws F(20,0) **twice** in two independently hand-drawn figures:

| | AgX d, µm | F(20,0) Fig. 3 | F(20,0) Fig. 1 | Δ | u at F(0,0)/2 |
|---|---|---|---|---|---|
| A | 0.30 | 1.318 | 1.319 | −0.1 % | 107.7 c/mm |
| B | 0.40 | *[1.79, 2.57]* | 1.751 | — | *[90.2, 109.7]* |
| C | 0.50 | 2.325 | 2.324 | +0.0 % | 83.8 |
| D | 1.50 | 3.021 | 2.892 | +4.5 % | 83.9 |
| E | 1.80 | 3.178 | 3.124 | +1.7 % | 79.8 |

B is a bracket: the two curves cross there and its markers are one blob of ink.

Fitted: `u½ = 85.7·d^−0.127` (R² 0.66), `F(20,0) = 2.57·d^+0.436` (R² 0.87), σ ∝ d^0.22.

**Why the inversion fails.** A **six-fold** change in crystal size moves the bandwidth by 35 %.
Inverting amplifies: 1 % in u½ → 8 % in d; 1 % in F(20,0) → 2.3 % in d; 1 % in σ → 4.6 % in d.
A 10 % uncertainty in a stored rms returns a crystal size uncertain by about 1.5×.

**And there is no anchor.** All 17 stocks with `emulsion.grain_um` take it from one third-party
aggregator, at 1.3–6.5 µm — Tri-X 4.5, TMAX P3200 6.5. Real AgX crystals in camera film run
0.2–2 µm. That field is very likely holding a developed-cluster figure under a crystal-size name.
Recorded; not corrected.

### ⚠ The by-product, which is worth more than the task was

Converting the measured bandwidths through the project's own `grain_shape` law
(`clump_um = 294.35 / u½`):

| source | `clump_um`, µm | implied u½, c/mm |
|---|---|---|
| BBC T-101 Table 2, the two **measured** stocks | PAN_F 0.655, HPS 1.431 | 449, 206 |
| **JPS 1965**, five real emulsions | 2.73 – 3.69 | 80 – 108 |
| corpus **estimates**, 168 stocks | median 13.0, range 3.0 – 40.0 | median 23 |

Three independent sources, a factor of twenty apart. **1 of 170** stocks lies inside the band this
page measures. The corpus median implies a grain spectrum half-power point at 23 c/mm against
80–108 measured here — i.e. most stocks render grain markedly blobbier than any measurement on
file supports.

The reading aperture is bounded rather than assumed: the curves run past 869 c/mm with no transfer
zero, so a circular aperture under ≈1.4 µm, MTF² ≥ 0.945 at 108 c/mm. It moves u½ by 2–4 % and
cannot explain a factor of four.

**Nothing adopted.** Moving 168 stocks on a one-page abstract in relative units would change every
render on the strength of the weakest of the three sources. Recorded as an open conflict.

Full treatment: `EMULSION_KNOWLEDGE_BASE.md` §23f, source **S8**.

---

## Build

`OK -- 0 failures, 0 warning(s)` · verify 493 PASS / 1 baseline FAIL · audit includes
`jp_jps_1965_269.py` · compile clean on all 18 TUs.
