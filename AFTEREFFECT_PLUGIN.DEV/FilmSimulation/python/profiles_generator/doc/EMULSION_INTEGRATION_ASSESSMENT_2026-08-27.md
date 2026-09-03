# Emulsion-level properties: why they are not in the database, and what should change

**Date:** 2026-08-27 · **Schema at time of writing:** v16, 161 stocks
**Companion documents:** `EMULSION_KNOWLEDGE_BASE.md` (the evidence), `NotFound.md` (the gaps),
`DIGITIZATION_QUEUE.md` (the work), `FilmActiveProfiles.md` (the current state)

This document answers a direct challenge: *film simulation is fundamentally emulsion-behaviour
simulation, so why is emulsion information absent from the numerical database?* The challenge is
substantially correct, and the answer below is not a defence of the status quo. But the diagnosis
matters, because the obvious remedy — add emulsion fields — would not fix the actual defect and
would add 161 empty records.

---

## 0. Executive answer

**0.1 The premise is right, with one important qualification.** Emulsion structure does determine
the observable characteristics. But for a *forward* renderer, emulsion structure enters **only
through the observables it produces**. Given a stock's measured characteristic curve, its MTF and
its σ(D) shape, the grain diameter adds no further information *for that stock* — it is already
integrated into those three measurements. This is why an observable-first schema is not a
conceptual error.

Emulsion-level parameters pay off in exactly three situations, and all three are live in this
project:

| Situation | Why structure beats observables | How much of our corpus it affects |
|---|---|---|
| **Extrapolation** — predicting behaviour at a condition nobody measured | Push processing, aging, reciprocity and development-time series are all *changes of condition*. An observable measured at one condition cannot be moved to another without a mechanism. | Every stock. Schema v16 just added `PushSpec` and can store the +3-stop *range* but nothing about what happens across it. |
| **Gap filling** — deriving a missing observable for an undocumented stock | 148 of 161 stocks have no measured σ(D) shape; 146 have no measured MTF rolloff; 156 have no measured halation. Those gaps are currently filled by family-ladder estimates typed in by hand. A structural model would *derive* them. | 88–156 stocks per property |
| **Coupling** — preventing physically impossible combinations | γ, granularity and speed are one relation, not three independent numbers. The schema currently lets all three be set freely. | All 161 |

**0.2 The architectural verdict: yes, but the defect is not "missing fields".** The schema is
observable-complete and **constraint-free**. Every field is something an instrument measures,
which is exactly why every value is verifiable against a datasheet — that discipline is what has
made this database auditable, and it should not be abandoned. The defect is that **physically
coupled observables are stored as mutually independent fields**, so the database can express films
that cannot exist. Adding an `EmulsionSpec` full of crystal diameters we do not have per stock
would not remove a single one of those impossible combinations.

**0.3 Therefore the proposal has three tiers, in this order of value:**

1. **Constraints, not fields** (zero schema churn, immediate correctness gain) — a DQE coupling
   guard and a "development family must not move speed" guard. These catch data-entry errors of
   precisely the kind this project keeps finding.
2. **Small causal fields with reachable per-stock evidence** — `DevelopmentProgress` on the
   processing record, and a two-component fog split. Both are *causes*, not observables, both are
   source-supported, and both are one enumerator or one float.
3. **`EmulsionSpec` — shape it now, populate it later.** Defining the record now prevents it being
   invented ad hoc under deadline; populating it before a per-stock crystal source exists would be
   fabrication.

**0.4 What is NOT proposed, and why.** Nothing from §27 of the knowledge base (Category C). Those
exclusions are not schema limitations — they are properties with no established path to a pixel,
and in one case (vinegar syndrome as an image effect) a property the authority explicitly
**contradicts**.

---

## 1. Emulsion characteristics identified in the reviewed references (Q1)

Six sources were reviewed in full: Tani 1995 (`Photographic Sensitivity`), Wall 1929
(`Photographic Emulsions`), a 1966 emulsion-technology text, Duffin, a sensitometry text, Reilly
1993 (rev. 1996) and NEDCC 5.1 (2020) for the carrier. The inventory below is the emulsion-level
content, grouped by mechanism. Section numbers refer to `EMULSION_KNOWLEDGE_BASE.md`.

| Group | Properties identified | KB |
|---|---|---|
| **Crystal composition and structure** | halide ratio (AgBr / AgBrI / AgCl), iodide content in mol %, lattice constants, solubility products, Frenkel and Schottky defects, interstitial Ag⁺ mobility, pAg | §4 |
| **Crystal geometry** | mean grain diameter, size-distribution width, habit (cubic / octahedral / tabular), aspect ratio, surface-to-volume ratio, grain volume, grain projected area | §5 |
| **Emulsion making** | precipitation regime, physical and chemical ripening time and temperature, halide-addition order, gelatin content at precipitation, wash regime, coating weight, layer thickness | §6 |
| **Gelatin** | class, setting/melting points, water absorption, natural-sensitizer (labile sulfur) content, batch variability | §7 |
| **Chemical sensitization** | sulfur, gold, sulfur-plus-gold, reduction sensitization; sensitization-centre count and depth; smallest developable centre size (**3 atoms** with S+Au vs **4–5** with S alone) | §8 |
| **Spectral sensitization** | dye identity and class, J-aggregate formation, adsorption as a monomolecular layer, redox potentials, supersensitization, intrinsic vs dye-sensitized absorption (**cubic AgBr 1 µm: ~10 % absorption at 420 nm, ~1 % at 460 nm; 3 mol % AgI raises 460 nm five-fold**) | §9 |
| **Latent image** | quantum sensitivity, four-atom developability threshold, recombination loss, nucleation rate ∝ I² at high intensity, latent-image fading | §10, §11 |
| **Development** | two progress types — **A parallel** (developed silver ∝ *proportion* developed within each grain) and **B granular** (∝ *number* of fully developed grains); rate ∝ 1/size for A, size-independent for B; worked identification of **CP-20 as A** and **D72 as B**; ~2 min to fully develop a 0.86 µm cubic AgBr grain in CP-20; developer identity does **not** move speed at each developer's optimum | §12 |
| **Fog** | two components — **emulsion fog** (fast, pre-existing centres) and **developer fog** (slow, initiation-limited); sensitization centres act as electron traps and therefore raise developer fog | §12, §13 |
| **Adjacency / inhibition** | development inhibitors and accelerators, edge and adjacency effects, DIR mechanism | §13 |
| **Inter-image** | inter-layer inhibition mechanisms | §14 |
| **Efficiency coupling** | **DQE = R₀²/R² = (log e)²γ²/(E·G²)**; measured DQE clusters at **1–2 %**; loss budget 42 % → 64 % (cum. 27 %) → 16 % (cum. 4.3 %) → 36 % (cum. 1.5 %) → 72 % (cum. 1.1 %), the last two terms being grain size/sensitivity **distribution** and random **arrangement** | §1.4, §8.1 |
| **Granularity** | Selwyn's law σ²·A = constant; the σ_D × 1000 at D = 1.0 through a 48 µm aperture convention; grain-size → graininess asserted everywhere and **quantified nowhere** in this source set | §17 |
| **Sharpness** | turbidity and light scatter within the coated layer; layer thickness; **MTF-50 ≈ half the resolving power** | §18 |
| **Carrier** | base material chronology, shrinkage **1–10 % with a knee at 1 %**, plasticiser loss, acetate hydrolysis, nitrate yellowing (ordinal only) | §21–§23 |
| **Colour** | dye fade — pre-1980 stocks "20 to 30 years at room temperature", **magenta survives longest** | §19, §22 |

---

## 2. What was not incorporated, and the specific technical reason (Q2, Q3)

**Nothing from this review was written to the database.** That was recorded at the time in
knowledge-base §25 and it remains the right call. The reasons fall into five distinct classes, and
the distinction between them is the whole substance of this report — "no schema field" is only one
of the five, and it is the *least* common.

| # | Exclusion reason | What it means | Properties excluded for this reason |
|---|---|---|---|
| **R1** | **No per-stock value exists** | The number is real and measured, but it characterises a *research* emulsion or generic 1929/1966/1995 practice, not a stock we model. Attaching it to a named profile would be fabrication. | grain diameter, size-distribution width, habit, aspect ratio, iodide mol %, sensitization type, coating weight, layer thickness — i.e. **all of §4–§8** |
| **R2** | **Direction known, magnitude absent** | The source establishes the relationship and its sign but supplies no coefficient. Supplying one makes it *our* number wearing a citation. | grain size → graininess; ripening → speed/contrast/fog; gelatin batch → speed/fog; turbidity and layer thickness → sharpness; acetic acid → dye-fade rate; iodide → halation extent |
| **R3** | **Two or more causal steps below any image quantity** | No measurable image effect is established from it *in these sources*. | solubility products, pAg, lattice constants, defect chemistry, Ag⁺ mobility, dye redox potentials, developer half-wave potentials and orbital counts |
| **R4** | **Manufacturing control variable whose image effect is already represented** | The property's only documented image consequence is a field we already carry. | antifoggant and stabilizer dosages → already Dmin; gelatin class → its documented effect is unquantified batch variation; ripening recipes → already the stored curve |
| **R5** | **No schema field, and the concept does not fit an existing one** | The genuine schema-limitation class. **Small, and every member is a candidate for change.** | development progress type; two-component fog; per-stock crystal structure; base material as a preset constraint; push-processing behaviour *across* the range (the range itself is now `PushSpec`, added v16) |

**One further exclusion is a correction rather than an omission.** Neither preservation source
states that the acetate polymer yellows or changes density — only nitrate is described as
yellowing, and only ordinally. Any non-zero `AgingSpec.base_yellowing_d` on an acetate-base
profile in our database is therefore **unsourced** and should be withdrawn or cited. KB §26 B7.

**And one is contradicted, not merely unsupported.** A vinegar-syndrome *image* preset would be
inventing an effect the authority denies: the source states the silver image "is not faded to
orange-red as is often the case in nitrate decomposition" and "usually remains in good condition".
KB §27.

---

## 3. Could the excluded properties contribute to realistic simulation? (Q4)

Read the class column before the row. **[F]** = fact stated by the source. **[Q]** = quantified by
the source. **[M]** = mechanism known, magnitude missing. **[E]** = our engineering inference,
explicitly not the source's.

Legend for the observable columns: **●** direct contribution, **○** indirect or conditional,
blank = no established path.

| Emulsion property | Grain | Spectral | Speed | H&D | Contrast | MTF | Interimage | Colour | Recip. | Process | Class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Grain **volume** | ● | ● | ● | | | | | ○ | | | **[Q]** |
| Grain **surface area** | ● | ● | ● | | | | | ○ | | | **[Q]** |
| **Tabular** habit | ● | ● | ● | | | ○ | | | | | **[F]** |
| **Iodide** mol % | ○ | ● | ● | | | | | ○ | | | **[Q]** |
| Size / sensitivity **distribution** | ● | | ● | ● toe | ● | | | | | | **[Q]** |
| Random grain **arrangement** | ● | | | | | ○ | | | | | **[Q]** |
| **4-atom** developability threshold | | | ● | ● toe | | | | | | | **[Q]** |
| **Recombination** loss | | | ● | ● toe | | | | | ● | | **[Q]** |
| **Gold** sensitization | ○ | | ● | ● toe | | | | | | ○ | **[Q]** |
| **Development progress type** | ● | | | ● | ● | | | ○ | | ● | **[Q]** dir / **[M]** mag |
| Grain size × progress type → **rate** | ○ | | | ● | ● | | | | | ● | **[Q]** |
| Developer identity vs **speed** | | | ● *(null result: no effect)* | | | | | | | ● | **[F]** |
| **Fog origin** (two components) | ○ | | | ● Dmin | ○ | | | ○ | | ● | **[Q]** dir |
| Sensitization centres as electron **traps** | | | ● | ● Dmin | | | | | | ○ | **[F]** |
| **γ / G / speed coupling**, Eq. (1.1) | ● | | ● | ● | ● | | | | | | **[Q]** rel / **[M]** cal |
| **Turbidity**, layer thickness | ○ | | | | | ● | ○ | | | | **[M]** |
| **Emulsion / developer fog** time law | | | | ● | ● | | | | | ● | **[Q]** dir |
| Nucleation rate ∝ **I²** | | | ● | ○ | | | | | ● | | **[F]** |
| **Latent-image** fading | | | ● | ○ | | | | | ○ | ○ | **[M]** |
| Development **inhibitor** diffusion | ● | | | | ○ | ● | ● | ● | | ● | **[M]** |
| Base **shrinkage** | | | | | | ○ | | | | | **[Q]** range |
| **Dye fade** | | | | ○ | ○ | | | ● | | | **[Q]** |
| Iodide → **halation** | | | | | | ○ | | ○ | | | **[M]** |
| Gelatin batch | ○ | | ○ | ○ | | | | | | | **[M]** |
| Ripening time / temperature | ○ | ○ | ○ | ○ | ○ | | | | | ● | **[M]** |
| Solubility products, pAg, defects | | | | | | | | | | | **R3** |
| Dye chemistry, J-aggregates | | ○ | | | | | | | | | **R3** |

**Three conclusions from the matrix.**

1. **Speed and the toe are where emulsion structure pays off most.** Six separate properties feed
   speed, five of them quantified. Our schema represents speed as a single `exposure_index` plus a
   `speed_criterion` string — there is nowhere for the *mechanism* of shadow speed to live, which
   is exactly the region where our 148 unmeasured toe shapes are least trustworthy.
2. **Development is the largest under-represented axis.** It touches H&D, contrast, grain,
   processing dependence and interimage. We carry `ProcessingSpec` (one condition) and
   `ProcessingFamily` (time–gamma points, **populated on 4 of 161 stocks**). The progress *type* —
   the one thing that explains why two stocks with identical rms granularity look different — has
   no field at all.
3. **Nothing in the matrix supports a new spectral field.** Dye chemistry explains where a
   sensitivity peak comes from; it does not predict the curve. Our spectral model consumes measured
   curves (73 of 161 populated) and that remains the correct design.

---

## 4. Modelability triage (Q5)

### 4.1 Quantitatively modelable today, from sources already on file

| Property | The quantity we have | What it would drive |
|---|---|---|
| **DQE coupling**, Eq. (1.1) | the relation, plus DQE clustering at 1–2 % and the five-term loss budget | a **verification guard** on (γ, rms, EI) triples. ⚠ Needs one calibration: Selwyn's *G* is defined through σ_D·√(2a); our `rms_granularity` is aperture-specific at 48 µm. Until that conversion is fixed the guard is ordinal, not numeric. |
| **Development progress type** | the two progress laws, and CP-20 = A / D72 = B | an enumerated field; grain-noise statistics and development-rate law |
| **Clump diameter vs development gamma** | BBC T-101 Table 3, refitted: D_eq ∝ γ^n with **n = 0.452 (Pan F), 0.396 (Tri-X), 0.425 pooled** — against the table's own printed claim of n = 0.5 | already used once (`ILFORD_PAN_F`: 0.859 µm at γ 1.0 → 0.655 stored at γ 0.55). Could correct every `clump_um` measured at a different gamma than the profile's curve represents |
| **Clump diameter vs density** | T-101 Fig. 21: Pan F **1.73 → 1.38 µm** as D rises 0.13 → 1.16, i.e. −20 % across the scale | a density dependence the schema does not have; stored `clump_um` is a mid-scale representative and nothing finer |
| **MTF-50 ≈ ½ resolving power** | the relation | already exploitable: 59 stocks have a printed resolving-power pair and 146 have no measured MTF rolloff. This is the only frequency-domain bridge in the source set |
| **Selwyn's law** σ²·A = const | exact | aperture conversion between granularity conventions |
| **Base shrinkage** | 1–10 %, knee at 1 % | a damaged-element preset. ⚠ Property of a degraded *object*, not of a stock — see §6.3 |

### 4.2 Requires additional measurement before it can be a number

| Property | Precisely what is missing | Cheapest route |
|---|---|---|
| grain size → rms granularity | any size→granularity relation. **Absent from all six sources.** | a granularity-vs-size plot from a manufacturer research paper |
| per-record grain ratio (volume vs area) | the ratio itself. The source gives both scaling laws and the absorption figures, not a granularity ratio | derivation with stated assumptions, or measurement |
| two-component Dmin | the two *rates*. Source gives mechanism and ordering only | Dmin-versus-development-time for one stock |
| reciprocity coefficient | any Schwarzschild exponent, any threshold intensity in absolute units, and **no temperature dependence anywhere in Tani** | a datasheet reciprocity table, or an exposure series |
| turbidity → MTF | scatter length, or a thickness→sharpness coefficient | an MTF-vs-thickness study |
| development type → noise statistics | the noise consequence. Source states the progress laws; the statistics step is **[E]**, ours | validate against a real scan of a type-A and a type-B development of one stock |
| iodide → halation extent | any magnitude. Wall 1929 is ordinal ("halation is less") | a halation measurement on two iodide levels |

### 4.3 Primarily qualitative — valuable, not parameterisable

Gelatin batch variation (the honest explanation for why nominally identical stocks differ, with no
magnitude in any source); ripening operational tables; dye chemistry; developer electrochemistry;
1929 recipes and coating-machine geometry; base identification tests. These belong in the knowledge
base and are the reason it exists.

---

## 5. What should be represented in the database now (Q6)

Ordered by value per unit of schema churn. **All four are additive and inert on every existing
profile**, so a database carrying them renders bit-identically until a value is set — the same
discipline every schema bump from v11 to v16 has followed.

### 5.1 ★★★ Two verification guards — no new fields at all

**G1 — the DQE coupling guard.** Flag any profile whose (γ, rms_granularity, exposure_index)
triple implies a DQE far outside the 1–2 % band real emulsions occupy. Today the schema lets a
stock be fast, contrasty and fine-grained at once, in combinations no emulsion achieves. Start
ordinal (rank-order consistency within a family), promote to numeric once the *G*-to-rms
conversion is established.

**G2 — the development-family speed guard.** Sensitivity does not depend on developer type at each
developer's optimum. A `ProcessingFamily` should therefore move γ and Dmin and **leave the speed
point alone**. Assert it on `DevelopmentPoint` data. Currently unchecked on all 4 populated
families.

*Why these first:* zero schema risk, and they attack the defect identified in §0.2 directly. Every
data-entry error this project has found — the flat CineStill dmin, the Velvia resolving-power/MTF-50
confusion, the six mixed-tag tiers — was a value inconsistent with its neighbours. Guards are how
that class of error stops being found by hand.

### 5.2 ★★★ `DevelopmentProgress` on the processing record

```python
class DevelopmentProgress(Enum):
    UNKNOWN  = 0   # default — nothing changes anywhere
    PARALLEL = 1   # type A: developed silver ∝ proportion developed within each grain
    GRANULAR = 2   # type B: developed silver ∝ number of fully developed grains
```

One enumerator on `ProcessingSpec` (or `ProcessingFamily`, if the family is the better home —
that is the one open design question). **Source-supported per developer**, which is the crucial
point: unlike crystal structure, it can be populated *today* for every stock whose developer is
known, because it is a property of the developer, not of the emulsion. CP-20 → A covers colour
negative in C-41/ECN-2; D72 → B covers MQ-developed B&W.

It is the only source-supported reason we have for two stocks with the same rms granularity to look
different in their grain, and it carries the rate law with it (parallel ∝ 1/size, granular
size-independent).

⚠ The *noise* consequence is our inference **[E]** and must be validated against a scan before any
renderer acts on it. Storing the field is not the same as acting on it — and that distinction is
what makes this safe to land now.

### 5.3 ★★ Two-component fog

Split the fog floor into an emulsion-fog term (fast, pre-existing centres) and a developer-fog term
(slow, initiation-limited). `DevelopmentPoint` already carries `base_fog` per point (v13), so a
family can express Dmin growth — but not that it grows at **two different rates**, which is what a
development-time series actually does. Interacts with the pending alternate-EI schema decision, so
it should land with it rather than before it.

### 5.4 ★★ `EmulsionSpec` — define the shape now, populate it when a source arrives

```python
@dataclass(frozen=True, slots=True)
class EmulsionSpec:          # ALL-DEFAULT = no crystal data on file
    grain_um: float = 0.0            # mean projected diameter
    size_sigma_log: float = 0.0      # distribution width, log units
    habit: str = ""                  # "cubic" | "octahedral" | "tabular" | ""
    aspect_ratio: float = 0.0        # tabular only; 0 = not applicable/stated
    iodide_mol_pct: float = 0.0
    sensitization: str = ""          # "S" | "S+Au" | "reduction" | ""
    coated_um: float = 0.0           # emulsion layer thickness
    source: str = ""
```

**Define now, populate never-until-sourced.** Two reasons to write the definition today even with
zero populated stocks: it stops the record being invented ad hoc under deadline when a source
finally arrives, and it gives every **[M]** row in §4.2 a named destination, so a future
measurement has somewhere to go instead of being forced into `GrainSpec`.

**But do not populate it from what we have.** Every candidate number in the reviewed sources
characterises a research emulsion or generic period practice (exclusion **R1**). Writing 1966
generic grain sizes against KODAK PORTRA 400 would be exactly the failure mode this project's own
rules exist to prevent.

### 5.5 ★ Two corrections that need no schema change at all

- **Audit `base_yellowing_d`** on acetate-base profiles and withdraw or cite every non-zero value
  (KB §26 B7).
- **Revisit the 54 colour stocks sharing the invented per-channel grain ladder** b/g = 1.30,
  r/g = 1.10. The schema already has per-channel `rms_r/g/b` and `clump_um_r/g/b`, so the
  volume-vs-area distinction is *already expressible* — the problem is that 54 profiles share one
  hand-picked constant pair, and the physics says the ratio should depend on habit and iodide, not
  be a constant. This is not a new field; it is 54 values that currently claim more than they know.

---

## 6. What should stay in the knowledge base, and why (Q7)

The test is not "is it interesting" but **is there an established path from this property to a
pixel**. Three categories fail that test in three different ways.

**6.1 Too far below the image (R3).** Solubility products, pAg, lattice constants, defect
chemistry, interstitial Ag⁺ mobility, dye redox potentials, J-aggregate formation, developer
half-wave potentials and orbital counts. Two or more causal steps from any measurable image
quantity, with no intervening magnitude. They explain *why* a curve has the shape it has; the
renderer consumes the shape.

**6.2 Manufacturing variables whose image effect we already carry (R4).** Antifoggant dosages
(effect = Dmin, already a field); gelatin class (effect = unquantified batch variation); 1929
recipes; coating-machine and drying-tunnel engineering. Adding the cause beside the effect we
already store would double-count.

**6.3 Properties of a degraded object, not of a stock.** Shrinkage, dye fade, plasticiser loss,
acetate hydrolysis. These are real, sourced and quantified — and **no stock in our database has an
age or a storage history.** They belong to a *damaged-element preset* keyed on a condition, not to
a stock profile. `AgingSpec` and `DyeStabilitySpec` exist as hooks and ship all-zero on all 161
profiles, which is the correct state until presets exist.

**6.4 The entire IPI storage-prediction apparatus, and the preservation practice around it.** The
Wheel, time contours, the 17× and 3× factors, A-D strip levels, free-acidity values, enclosure
chemistry, vault engineering. These predict *when a carrier begins to smell of vinegar* — at which
point the source itself says the film "may not have any other symptoms of degradation and will be
perfectly usable". There is no path from these numbers to a pixel.

**6.5 And one that must stay out because it is wrong.** Vinegar syndrome as an image effect — see
§2.

---

## 7. The architectural question, answered directly

**7.1 Is the design too narrowly focused on final observable parameters?** Yes — but the failure is
not that observables are the wrong thing to store. It is that **the schema stores coupled
quantities as independent ones.**

Concretely, the database currently permits:

- a stock that is fast, high-contrast and fine-grained beyond what Eq. (1.1) allows at any DQE a
  real emulsion reaches;
- a development family that shifts the speed point, which the sources say cannot happen;
- one σ(D) shape per stock regardless of developer class, when the progress type changes the
  underlying counting statistics;
- one `clump_um` per layer with no gamma or density dependence, when both are measured (T-101:
  γ^0.425 and −20 % across the tone scale);
- 54 colour stocks sharing one invented per-channel grain ratio, when the physics says the ratio
  varies with habit and iodide.

Every item on that list is a **missing relation between fields we already have** — not a missing
field. That is why "add an `EmulsionSpec`" is the wrong first move: it would leave all five intact.

**7.2 The technically sound way to represent emulsion-level behaviour** — three layers, kept
strictly separate, which is what prevents concepts being forced into inappropriate fields:

| Layer | What lives there | Consumed by | Status |
|---|---|---|---|
| **L1 Observable** | the existing schema — curves, MTF, granularity, halation, interimage, dye density | the renderer, directly | complete and auditable; keep the discipline |
| **L2 Causal** | small, source-supported *causes* that L1 cannot express: `DevelopmentProgress`, two-component fog, and eventually `EmulsionSpec` | **not the renderer** — a derivation step that *produces* L1 values, plus the guards | §5.2–§5.4 |
| **L3 Relational** | the constraints between L1 fields: Eq. (1.1), Selwyn's law, MTF-50 ≈ ½ RP, D_eq ∝ γ^0.425, the speed-invariance rule | `verify.py`, and any future derivation in L2 | §5.1 — **not yet built, highest value** |

The key architectural point: **L2 must never be read by the renderer.** If a causal field can
change a pixel directly, it competes with the observable that already describes the same thing, and
the two will disagree. L2's job is to *derive or check* L1 values offline, in the generator, where
the result is written into L1 with a provenance note. That is the same pattern
`vision3_granularity.py` and `agfa_vista.py` already follow for traced plots — this proposal
generalises a mechanism that exists rather than inventing one.

**7.3 Where L3 belongs.** Not in `film_profiles.py` and not in a new module: in `verify.py`, beside
the 403 checks already there. A constraint that is not enforced on every build is a comment.

---

## 8. Recommended sequence, and what it costs

| # | Item | Schema change | Rendered output changes | Blocked by |
|---|---|---|---|---|
| 1 | **G2** development-family speed guard | none | none | nothing — 4 families to check |
| 2 | `base_yellowing_d` acetate audit | none | possibly, on affected profiles | nothing |
| 3 | **G1** DQE guard, ordinal form | none | none | nothing |
| 4 | **G1** numeric form | none | none | Selwyn *G* ↔ 48 µm rms conversion |
| 5 | `DevelopmentProgress` | v17, one enum, inert | none until a renderer reads it | the combined-schema decision |
| 6 | Two-component fog | v17, same bump | none until a family uses it | the alternate-EI decision |
| 7 | Per-channel grain ladder review, 54 stocks | none | **yes** | the volume/area ratio (§4.2) |
| 8 | `EmulsionSpec` definition | v17, same bump, empty on 161 | none | nothing to define; everything to populate |
| 9 | `clump_um` gamma/density dependence | v18 or later | yes | a decision on whether the renderer gains a gamma input |

**Items 1–4 are the ones I would do first, and none of them touches the schema.** That is the
honest headline of this assessment: the largest available gain in physical realism right now is not
new emulsion fields, it is enforcing the relations between the fields already populated.

**Items 5, 6 and 8 should land as a single v17 bump** together with the two schema decisions already
pending in the queue (the `aim_density` field shape and the alternate-EI curve-set structure).
Landing five overlapping schema changes in five passes would produce five migrations where one
would do — the same reasoning that deferred the `DevelopmentProgress` field when it was first
identified.

---

## 9. Direct answers, in one place

1. **Identified:** §1 — sixteen groups, from crystal composition to carrier chemistry.
2. **Not incorporated:** all of it; zero database writes from this review.
3. **Reasons:** five distinct classes, §2 — **R1** no per-stock value (the largest class), **R2**
   magnitude missing, **R3** too far below the image, **R4** effect already represented, **R5** no
   suitable field. Only **R5** is a schema limitation, and it has five members.
4. **Contribution:** §3 matrix. Speed and the toe benefit most; development is the largest
   under-represented axis; nothing supports a new spectral field.
5. **Triage:** §4 — seven properties quantitatively modelable today, seven needing measurement, six
   irreducibly qualitative.
6. **Represent now:** §5 — two guards (no schema change), `DevelopmentProgress`, two-component fog,
   and an `EmulsionSpec` **definition** with zero populated stocks.
7. **Stays in the knowledge base:** §6 — sub-image chemistry, manufacturing variables whose effect
   we already store, degraded-object properties awaiting a preset system, the IPI apparatus, and one
   effect the source contradicts.
8. **Architecture:** §7 — the design is observable-complete and constraint-free. The fix is a
   three-layer separation with the causal layer deliberately **invisible to the renderer**, and the
   relational layer enforced in `verify.py`.

**Nothing in §5 is proposed for completeness.** Each item is there because a named source supports
it, a named observable would improve, and a named failure mode is currently unguarded.
