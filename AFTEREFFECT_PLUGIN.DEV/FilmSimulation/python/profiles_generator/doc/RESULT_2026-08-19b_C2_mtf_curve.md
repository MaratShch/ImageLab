# RESULT 2026-08-19b — C2: MTF becomes a curve, and the stored carrier was the wrong one

> ⚠ **STILL CORRECT IN ITS CONCLUSION, STALE IN ITS EVIDENCE BASE (2026-08-23).** C2 chose
> the power-law carrier `1/(1+(f/f50)^q)` against **one** traced curve, and noted that a
> sampled array scored better but was over-parameterised against a single curve. There are
> now **26 traced curves on 12 sheets**, and the power law beats the Gaussian on every one
> of them (1.1×–5.8×, rms 0.0095–0.132), so the choice is confirmed on 26× the evidence.
> What did NOT survive is the hope that q could be derived rather than measured — see
> `RESULT_2026-08-23b_c2b.md`. One stock (5279) now carries a measured f50 triple with q
> deliberately unfitted, because its printed +42 %/+55 % adjacency overshoot cannot be
> represented by a carrier that equals 1.0 at zero frequency.

**Task:** wire MTF-as-a-curve the way C1 wired σ(D) — one shared sampler, a `measured` flag so
estimates stay inert, a level-preserving hook, and the C++ mirror.

**Outcome:** done, and the carrier the schema had been reserving for this (`mtf_tail_a` /
`mtf_tail_f_exp`) **lost the scoring** to a simpler form. Build: **`build.py --root <corpus>` OK**,
9 audits green, `verify.py` **265 PASS / 2 FAIL** (the two known), C++ clean on 18 TUs, schema
**v9 → v10**.

---

## 1. The carrier was chosen by measurement, and the incumbent lost

Scored against the only vector MTF curve traced from the corpus — **EASTMAN PLUS-X 5231**,
H-1-5231 p3, 48 samples over 2.4–98.2 cycles/mm via `mtf_vector.py` — using the **35 samples above
8 cycles/mm**. The lower samples carry the adjacency overshoot, which is a separate effect modelled
separately; including them would have bent the rolloff to absorb a lift.

| form | rms | extra params | notes |
|---|---|---|---|
| single Gaussian `exp(-ln2 (f/f50)²)` | 0.0878 | 0 | what every stock used until today |
| `a·gauss + (1-a)·exp(-ln2 (f/f50)^p)` | 0.0583 | 2 | ⚠ **the stored `mtf_tail_a` / `_f_exp` form.** Its own optimum drives **a → 0**, i.e. it discards the Gaussian core, and still cannot reach the measured tail |
| two Gaussians, renormalised at f50 | 0.0750 | 2 | |
| `a·gauss + (1-a)/(1+(f/f50)^q)` | 0.0375 | 2 | optimum also drives a → 0 |
| **`1/(1+(f/f50)^q)`** | **0.0375** | **1** | **adopted** |
| sampled array, 12 log-spaced samples | 0.0012 | 12 | rejected — see below |

The power law wins twice: it is the most accurate two-parameter form, and its **one-parameter**
version is exactly as good, because the blend's own optimum sets the Gaussian weight to zero.

**It also passes through 0.5 at f50 exactly, for any q.** That is not cosmetic — it is what let C2
land without a level decision riding along. The grain work had the opposite property: there the two
laws disagreed at the reference density, so wiring the shape in would have changed the level on every
stock, and that had to be unpicked separately as C1b. Here the reference point is pinned by
construction in both branches, so a stock that gains a measured rolloff changes its **shape away from
f50** and nothing else.

**Why the 12-sample array was rejected** despite being 30× better: 35 measured points, and exactly
**one** traced curve in the whole database. That is fitting a sample, not a film — method rule 18.
199 vector MTF pages are inventoried; when a handful more are traced, the array goes back in the
running (queue **C2b**).

⚠ **What the adopted law still gets wrong, recorded rather than tuned away:** at 98 cycles/mm PLUS-X
measures 0.245 and the law gives 0.169. Real emulsion tails are fatter than any two-parameter
analytic form. Measured against the traced curve:

| f (cycles/mm) | traced | Gaussian | adopted power law |
|---|---|---|---|
| 20 | 0.786 | 0.858 | 0.800 |
| 41.3 (f50) | 0.502 | 0.506 | 0.504 |
| 61 | 0.370 | 0.219 | **0.327** |
| 77 | 0.306 | 0.092 | **0.243** |
| 98 | 0.245 | 0.020 | **0.169** |

## 2. What changed in the code

| file | change |
|---|---|
| `film_profiles.py` | `MTFSpec` gained `mtf_rolloff_q` and `mtf_measured`; new module-level **`mtf_response(mtf, channel, f)`** — the one definition, two laws, both exactly 0.5 at f50. `SCHEMA_VERSION` 9 → 10 |
| `film_sim.py` | `FreqGrid.mtf()` takes an optional `spec` + `channel` and defers the rolloff shape to `fp.mtf_response`. The scanner and dupe stages pass no spec and keep the Gaussian, which is what they always had |
| `cpp_codegen.py` | struct fields + emitter, and **`FilmMtfResponse()`** mirroring the Python law |
| `cpp_parity.py` | now probes **both** laws: 8478 probes (4710 grain + 3768 MTF), the MTF family sampling out to **6× f50** because that is where the two laws diverge — a parity check confined to the mid band would pass on a twin carrying the wrong law |
| `verify.py` | 5 new guards, below |
| `EASTMAN_PLUS_X_5231` | `mtf_rolloff_q=1.84, mtf_measured=True` with the fit quality and the residual in its comment |

**One bug worth recording:** the first sampler computed in float64 and cast back to float32, which
moved **154 unmeasured stocks by ~1e-8**. Not a visible change — but it destroys the single property
that makes this kind of wiring safe to land, that a stock without measured data renders *identically*.
The sampler now preserves the caller's dtype, and `verify.py` asserts float32-exact equality over
0–300 cycles/mm.

**And one C++ ordering bug:** `FilmMtfResponse` was emitted *above* `struct MTFSpec`, so g++ bound its
first parameter to `int` and the parity probe failed with *"invalid initialization of reference of
type const int&"*. Moved below the struct.

## 3. Guards added

* unmeasured stocks reproduce the legacy Gaussian **bit-for-bit in float32** (156 profiles, 0–300
  cycles/mm);
* **MTF is exactly 0.5 at f50** for every stock and channel (471 stock-channels) — the property that
  keeps this a shape-only change;
* only the one traced stock is flagged `mtf_measured`, and a flagged stock must carry `q > 0` (a flag
  without an exponent silently falls back to the Gaussian — a lie that renders fine);
* the measured law must **beat the Gaussian on its own traced curve** by at least 4× in summed square
  error at 61 / 77 / 98 cycles/mm. This is the assertion that would fail if someone "simplified" the
  law back.

## 4. Still open, deliberately

* **`adjacency_um` disagrees with the measured overshoot frequency on both stocks where it has been
  checked** — PLUS-X peaks at 4.7 cycles/mm against a stored 16.0 µm, FUJI F-125 at ~9 cycles/mm
  against 13.0 µm. Systematic, unresolved, and *not* touched here: adjacency is a separate effect and
  C2 was scoped to the rolloff. Queue **C2c**.
* **`mtf_tail_a` / `mtf_tail_f_exp` are now known to be the wrong form** and are left in place, inert,
  annotated with the scoring table. They are not deleted because some other stock's curve may yet fit
  a Gaussian-core shape, and because a deleted field takes its own history with it.
* **199 vector MTF pages inventoried, 1 traced.** Queue **C2b**: trace a handful more, then re-score
  with the sampled array in the running — the one-curve basis is the weakest part of today's choice
  and saying so is cheaper than pretending otherwise.
