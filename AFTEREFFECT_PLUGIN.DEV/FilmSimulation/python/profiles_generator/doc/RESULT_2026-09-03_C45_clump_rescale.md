# RESULT 2026-09-03 — C45, the corpus-wide `clump_um` rescale

Owner-approved. Every **estimated** `clump_um_r/g/b` divided by **3.1**; the five stocks whose
value was **measured** left untouched. **510 literals across 170 profiles.** Corpus median
**13.00 → 4.19 µm**, maximum **40.0 → 12.90**, minimum unchanged at 0.655.

⚠ **This row had been closed as REFUSED the day before. The refusal did not survive being
measured, and this document is mostly about why.**

---

## 1. What was actually in the way, and it was not the number

C45 had been open since 2026-09-01 and refused on 2026-09-02c. Reading it again this morning,
three separate things were tangled together, and untangling them is what made the decision
possible.

### 1.1 ⚠ `clump_um` is the GRAIN scale, and its name says the opposite

`film_sim.FreqGrid.grain_shape`:

```
f_hi = 500 / clump_um          the high-frequency rolloff   <- clump_um sets THIS
f_lo = f_hi / 6                the clump_gain lobe          <- a length 6x clump_um
```

The model carries **both** scales already, a factor of six apart. `clump_um` is the fine one.

⚠ **And Takano 1968 — read yesterday, for an unrelated reason — measures the developed aggregate
at 5 to 8 times the mean grain.** The model's hard-coded 6 sits inside its measured range, from a
document written thirty years before the model. That is an independent corroboration of the
*ratio*, and it is what assigns each source to the right scale:

* BBC T-101's equivalent grain diameter, Takano 1969's autocorrelation half-widths and Ooue's
  Wiener half-powers are all **grain** → they convert to `clump_um` directly.
* Takano 1968's Expected mottle size is the **cluster** → it converts by dividing by six.

Yesterday's knowledge-base entry compared Takano's mottle ÷ its own 5–8× factor against T-101's
`clump_um` and called the agreement "two documents, one answer". That is arithmetically fine but
was reasoning about *physical grain*; the version that matters for C45 is the one above, which
reasons about **the model's parameter**. Same numbers, different and more useful claim.

### 1.2 ⚠ It is not a parameter-basis argument, and that had to be checked first

The obvious objection: TK1 showed `clump_um` and `clump_gain` trade off, so perhaps the estimates'
(13.0, 0.25) describe the same spectrum as a measurement's (2.5, 0.0) in a different basis.

**They do not, and the reason is one-directional** — a lobe only ever *adds* low-frequency energy,
so it pushes the half-power **down**, never up:

| clump / gain | spectrum half-power |
|---|---|
| 13.0 / 0.00 | 22.6 c/mm |
| 13.0 / 0.25 | 13.7 c/mm |
| 13.0 / 0.85 | 6.2 c/mm |
| 13.0 / 1.65 | 5.0 c/mm |
| **Ooue, measured** | **45.6 / 70.8 / 140.7 c/mm** |

No value of `clump_gain` reaches the measurements. The stored estimates were a genuinely different
spectrum.

### 1.3 ⚠ The refusal's reason, tested — and this is the part that decided it

C45 was refused because the rescale makes rendered grain more resolution-dependent (1.31× against
1.90× across 960–4000 px): *"a change in how the product behaves on 168 stocks that the data alone
cannot authorise."*

**`rms_granularity` is defined through a 48 µm aperture, and the aperture-referred rms does not
move.** Measured end to end on `KODAK_VISION3_250D_5207`, flat 0.18 patch, green record, ×1000:

| clump / gain | 960 px | 2000 px | 4000 px |
|---|---|---|---|
| 13.00 / 0.25 | 5.263 | 5.203 | 5.228 |
| 6.46 / 0.00 | 5.202 | 5.151 | 5.170 |
| 4.16 / 0.00 | 5.197 | 5.147 | 5.166 |
| 2.46 / 0.00 | 5.194 | 5.145 | 5.164 |

**1.3 % across the whole clump range and every render size.** The film still measures what its
datasheet says, through the aperture its datasheet specifies. Only sub-aperture detail changes —
and the datasheet number never described that.

So the change is **"same film, correctly resolved"**, which is a fidelity claim, and fidelity the
data *can* authorise. It is not "more grain".

⚠ **And the engine had already ruled on the direction, in writing, long before C45 existed.**
`grain_reference_energy`'s docstring: *"a 2K render genuinely shows less granularity than a 6K
render of the same negative, converging upward as the band widens. That is not a modelling
artefact — it is why 4K scans of old negatives look grainier than the 2K masters everyone
remembers."* The old 1.24× was that intended behaviour **suppressed**, because a coarse clump
leaves little out-of-band energy to recover.

---

## 2. The census, all in one parameter

`clump_um = 294.35 / u½`, the engine's own closed form, verified against a numerical inversion.

| source | what it measured | `clump_um` |
|---|---|---|
| **Ooue 1959 Fig. 26** | three **directly measured Wiener spectra**, named stocks, stated developer, time and density | **2.09, 4.16, 6.46** |
| Takano 1969 Figs 8, 13 | Selwyn aperture series, optical autocorrelation, five samples | 0.87, 1.77, 2.46, 3.22, 4.64 |
| BBC T-101 Table 2 | printed equivalent grain diameter, six emulsions, ÷ 1.7473 | 0.59 – 1.43 |
| Takano 1968 Fig. 11 | mottle 3.98 – 6.81 µm ÷ the model's own 6× lobe ratio | 0.66 – 1.14 |
| JPS 1965 | five emulsions, relative ordinate | 2.73 – 3.69 |
| **stored, before** | 175 stocks | **median 13.00**, 160 above the band's top |

Distribution before, by class: colour negative median 12.0 (4.6–20), monochrome 15.0 (0.66–40),
reversal 12.0 (3.8–18).

---

## 3. The two decisions inside the decision

### 3.1 Anchor: Ooue alone, k = 3.1 — not the full census's 5.3

Ooue Fig. 26 is the strongest evidence class in the corpus for this quantity: three directly
measured Wiener spectra on **named** stocks with stated developer, time and density, needing no
conversion chain. Median 4.16 µm against a stored 13.00 → **k = 3.1**.

The full census would have given **k = 5.3** and a larger change. It was not used: it mixes
directly-measured spectra with values that arrived through a conversion, and the conservative
anchor still lands inside the band every other source brackets.

### 3.2 ⚠ `clump_gain` untouched — a recorded conflict, not an oversight

All five measured stocks carry `clump_gain` **0.00**. A free two-parameter fit drove the lobe to
exactly zero on all six T-101 emulsions, and T-101 p38 says in words that grain correlation is
"substantially confined to about plus or minus one equivalent grain diameter".

⚠ **But Takano 1968 measures a real aggregate at 5–8× the grain — i.e. that the lobe exists.** The
two sources disagree outright. **Method rule 4: record the conflict, do not average it.** Scaling
`clump_um` alone is **one** change; zeroing the lobe as well would have picked a winner between two
disagreeing measurements.

⚠ **Consequence, stated rather than hidden**: §23k.1 / TK1 showed the two parameters are not
separately identifiable, so every rescaled value is **conditional on that stock's stored gain**. A
measurement that settles the lobe would move these values again, and should.

---

## 4. What was exempt

Five stocks, all with `clump_um` measured from BBC T-101 Table 2's printed equivalent grain
diameters, all at `clump_gain` 0.00:

`ILFORD_PAN_F` 0.655 · `KODAK_8374` 0.687 · `EASTMAN_PLUS_X_5231` 0.830 ·
`EASTMAN_TRI_X_5223` 1.259 · `ILFORD_HPS` 1.431

⚠ **`KODAK_TECHNICAL_PAN` was NOT exempt despite matching the first search.** Its comment says
"clump EST [C3] scaled below ACROS" — an estimate that merely sat near a measured neighbour in the
text. Checked individually rather than by pattern; it was rescaled with the rest.

---

## 5. What the user sees, and what it costs

`KODAK_VISION3_250D_5207`, flat 0.18 patch, green record, ×1000:

| clump | 960 px | 2000 px | 4000 px |
|---|---|---|---|
| stored 13.0 | 6.59 | 7.77 | 8.16 |
| **adopted ≈4.2** | 6.77 **(+3 %)** | 9.88 **(+27 %)** | 13.34 **(+63 %)** |

⚠ **Near-zero at 1080p and growing with resolution** — the shape the physics predicts, and the
reason this is a smaller product change than the row implied.

**Cost: none.** 716.6 → 689.6 ms single-thread through the full `simulate()` at 2000 px. `clump_um`
changes the transfer's shape, not the transform's size.

---

## 6. Guards

Three new, replacing the TK4 guard that used to assert the *disagreement*:

1. the corpus clump median sits inside the measured band 0.59–6.46 µm;
2. the five measured stocks are untouched, asserted value by value;
3. the aperture-referred grain is invariant — `grain_reference_energy` integrates over all
   frequencies with the 48 µm aperture, so the renderer's amplitude absorbs the clump change by
   construction.

⚠ **One existing guard had to be rewritten because it pinned a literal instead of a property.** It
read *"the STILL Tri-X keeps clump_um 19.0 — 5223 is a different product"* and failed on a database
doing exactly what was approved. The property it was written to protect is the **trade-name
separation** — that the ASA 400 still film does not converge on the cine 5223's measurement — not
the number 19.0. It now asserts the separation and the estimate's provenance, and survives any
future scale change.

---

## 7. ⚠ What would reopen this

**A measured Wiener spectrum or a multi-aperture rms for a stock that is actually in this
database.** All fifteen measurements above are of films this project does not carry, so the
rescale remains a **class inference** — a well-evidenced one, across five documents, three
countries and three decades, but a class inference, and method rule 18 is the reason to say so
out loud rather than quietly.
