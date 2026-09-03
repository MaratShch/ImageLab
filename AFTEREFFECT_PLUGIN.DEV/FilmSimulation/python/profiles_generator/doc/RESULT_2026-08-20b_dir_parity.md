# RESULT 2026-08-20b — the DIR-coupler stages get a parity test, and it found two real defects on the first run

**Task.** Owner-approved after an audit established that inter-image effects
**are** modelled — both halves, live in both renderers — and that the actual gap
was that nothing checked the two implementations agree. `cpp_parity.py` covers
the grain and MTF laws only.

**Outcome.** Build: **`build.py --root <corpus>` OK**, **11 audits green** (one
new), `verify.py` **279 PASS / 1 FAIL** — the baseline is **down from 2 to 1**,
C++ clean on 18 TUs, **159 stocks, schema v10 unchanged**. No profile data
changed, so no ListBox shift and no plugin rebuild required.

| | |
|---|---|
| **New audit** | `interimage_parity.py` — probes the **plugin's own** `Algo_08_Sim.cpp` / `Algo_09_Sim.cpp`, not generated code. Only audit that does |
| **Defect 1, fixed** | the density floor was **inside** the C++ stage 9 and **outside** the Python one — 0.26 D disagreement on a reversal stock |
| **Defect 2, open** | the adjacency term is **not the same effect in the two renderers** at ordinary render sizes — sub-pixel blur, measured |
| **Stale guard** | converted, not deleted. Half the FAIL baseline was an assertion the project had already rejected on evidence |
| **Missing doc** | `CHANGES_2026-08-03_v5_interimage.md` written — four places cited it and it was not on disk |
| **Doc contradiction** | interimage tier: generator said 3 "without exception", the docstring said 2. Generator was stale |

---

## 1. Why a parity test, and why these two stages

The two DIR-coupler stages are the **largest colour effect in the chain**.
Measured by disabling them and re-rendering a patch set (max channel delta,
0–255):

| patch | PORTRA 400 | 5250 (1959) | VELVIA 50 |
|---|---|---|---|
| grey 0.18 | 0.00 | 0.00 | 0.00 |
| grey 0.45 | 16.3 | 3.4 | 23.1 |
| red | 23.2 | 4.2 | **124.2** |
| green | 18.3 | 3.1 | **143.0** |
| skin light | 8.7 | 1.8 | 20.2 |

Both stages exist twice — `film_sim.py` 8b/9 and `AlgoStage08b_Interimage` /
`AlgoStage09_DirCoupler`, called from `AlgorithmMain.cpp:976` and `:996` — and
nothing compared them. That is the exact configuration that produced the C1b
calling-convention bug: one law, two languages, a manual one-off cross-check
that guarded nothing. It is also why `cpp_parity.py` exists at all; it just
never covered the colour stages.

## 2. What had to change first: the laws became callable definitions

Both stages were **inline inside `simulate()`**, so there was nothing to probe.
Factored into `film_sim.apply_interimage()` and `film_sim.apply_dir_couplers()`,
with the pipeline calling them — the same "one definition, mirrored in C++"
discipline `grain_sigma` and `mtf_response` already follow.

⚠ **Verified bit-identical before and after**, against the pre-refactor
arithmetic reproduced verbatim as a reference implementation, on four stocks
covering both mechanisms. Not "it still looks right" — `np.array_equal` on the
output planes.

## 3. Defect 1 — the density floor was on the wrong side of the function boundary

The C++ stage 9 ends with

```cpp
rO[x] = MAX_VALUE(rO[x], ALGO_ZERO);   // "a physical floor, not a display clamp"
```

Python clamped **one line later**, in `simulate()`. So the two **pipelines**
agreed and the two **functions** did not — and nothing could see it until the
functions were compared directly.

It surfaced immediately as a **0.26 D** disagreement on `FUJI_VELVIA_50`: a
reversal stock whose ramp drives density negative, already floored on one side
and not yet on the other. Every other stock in the probe passed.

Fixed: the floor now lives inside `apply_dir_couplers`, where its twin has it.
**Rendering is unchanged** — `max(max(x,0),0)` is `max(x,0)` — and `simulate()`'s
later clamp stays, because it also guards the stages between there and here. Two
new `verify.py` guards pin it: the function must floor at zero, *and* it must
still separate the layers of a flat colour (a stage that only clamped would pass
the first check and render nothing).

## 4. Defect 2 — the adjacency term is not the same effect in the two renderers

**Open, reported, not silenced.** Python blurs by multiplying the **analytic
Gaussian transfer** in the frequency domain; C++ runs a **truncated separable
spatial kernel** (4σ cutoff). Measured directly, one ramp plane, worst
disagreement between the two blurs:

| σ (px) | 0.30 | 0.40 | 0.60 | 1.00 | 1.20 | 1.76 | 5.28 | 9.60 |
|---|---|---|---|---|---|---|---|---|
| worst err | 1.4e-1 | 1.5e-1 | 4.3e-2 | 7.7e-4 | 6.4e-5 | 1.3e-6 | 2.2e-5 | 1.8e-5 |

Above ~1.2 px they agree to 6e-5 and the implementation choice does not matter.
**Below 1 px they are simply not the same operator** — a Gaussian narrower than
the sample grid is not represented by either form, and no tolerance fixes that.

⚠ **The coupler edge term lives in that zone in normal use.** Stored `edge_um`
is 9–13 µm, so at **40 px/mm** — a 35 mm frame rendered about 960 px wide — the
edge sigma is **0.36–0.60 px**, and the two renderers' adjacency output differs
by up to **2.6e-2 D**.

So the parity assertion runs at a scale where every active sigma is resolved,
and the sub-pixel scale is **probed and printed as information**. Picking the
convenient scale and calling it green would have buried the finding.

⚠ **My first diagnosis was wrong and the data said so.** I hypothesised the
kernel exceeding the probe plane. The evidence contradicted it immediately:
disagreement was *worse* at 40 px/mm (small kernel) than at 120 px/mm (large
kernel), the opposite of what that hypothesis predicts. Measuring the two blurs
directly, across sigma, is what found the real cause. Recorded because the wrong
turn is the useful part.

## 5. Also probed, not fixed: a gate that exists on one side only

C++ gates **both** coupler components on `radiusPx >= ALGO_COUPLER_MIN_SIGMA_PX`
(0.25 px, `AlgoDirCoupler.hpp:70`); Python has no such gate. The crossover
points are now printed on every run: the long term switches off below
**3.1 px/mm** (`EASTMAN_5247_1974`, radius 80 µm) and the edge term below
**27.8 px/mm** (`KODACHROME_64`, edge 9 µm).

⚠ **27.8 px/mm is not an exotic scale** — it is a 35 mm frame at roughly 670 px
wide. Below it the C++ renderer silently drops the adjacency term the Python
reference still applies. The gate is defensible in itself (a sub-quarter-pixel
blur is not resolvable); what is not defensible is only one of the two having it.
A decision, not a bug fix — see the queue.

## 6. The stale guard: converted, not deleted

`verify.py` asserted *"neighbour pairs couple harder than the far red-blue
pair"* — `|a_rg| > |a_rb|`, a **per-distance** asymmetry. The database stores
those **equal**, deliberately, because the evidence says the asymmetry is per
**receiver**: US4725529A Table 1 puts the inhibitor in the *developer* and
applies it to **three separate single-layer coatings** — no layer stack, no
distance to travel — and still measures red receivers at 0.43–0.72 ΔlogE against
blue at 0.24–0.48.

So the guard encoded the hypothesis the project had already rejected, and it had
been parked in the FAIL baseline as "known, leave alone". That is how a fixable
stale assertion came to be treated as immovable for a whole session.

**Deleting it would have changed zero pixels.** It was replaced with the
assertion the evidence supports — that the three rows are symmetric per receiver
— which keeps a live check where there was a permanent red, and documents the
rejected hypothesis instead of contradicting the accepted one. `build.py`'s
baseline mechanism reported the change rather than swallowing it: *"baseline
entry now PASSES — remove it from VERIFY_BASELINE"*.

**FAIL baseline: 2 → 1.** The remaining one is the saturation hierarchy, which is
a genuine open ordering question.

## 7. `AlgoType` stays switchable, and the harness respects that

`AlgoTypes.hpp` sets `using AlgoType = double` deliberately, so the whole
renderer can be flipped between 64-bit and 32-bit arithmetic in one place.

The probe therefore **prints `sizeof(AlgoType)`** and the Python side picks its
tolerance from that at run time — 2e-6 for an 8-byte type, 2e-3 for a 4-byte
one, because the reference carries its density planes in float32 and two float32
pipelines that round in different orders diverge far more than a float32 and a
float64 one do. Hard-coding a double tolerance would turn a future switch to
float into a spurious failure; hard-coding a float tolerance would blind the
check today.

## 8. Result

```
[i] AlgoType is 8 bytes -> tolerance 2e-06
[i] scale 120 px/mm: long-term sigma 0.00-9.60 px, EDGE-term sigma 1.08-1.80 px -- ASSERTED
[i] scale  40 px/mm: EDGE-term sigma 0.36-0.60 px -- REPORTED ONLY (sub-pixel)
[OK] stages 8b and 9 agree between Python and the plugin's own C++
     -- worst 5.335e-05 over 5 stocks x 2 fields x 5760 values
```

Stage 8b — pointwise, no blur — agrees to **≤ 8.6e-07** everywhere, across both
interimage mechanisms (chromogenic negative and the reversal density-weighted
form) and on a monochrome control where both stages must be inert.

**Stocks probed, chosen for mechanism coverage rather than popularity:**
`KODAK_PORTRA_400` (strong DIR negative), `FUJI_VELVIA_50` (reversal, weighting
0.65, the stock the stage moves most), `KODAK_VISION3_500T_5219` (strongest
stored coefficients), `EASTMAN_5250_1959` (the "trace" tier — small coefficients,
where a sign error would be least visible and therefore most dangerous),
`EASTMAN_DOUBLE_X_5222` (monochrome control).

## 9. Files changed

| file | change |
|---|---|
| `interimage_parity.py` | **NEW.** Compiles a probe against the plugin's real TUs (`Algo_08_Sim`, `Algo_09_Sim`, `AlgoSeparableBlur`, plus `Algo_05_Sim` for `AlgoSoftplus` and `AlgoDefectField` as link dependencies of the file layout) and compares both stages against the Python reference |
| `film_sim.py` | stages 8b and 9 factored into `apply_interimage()` / `apply_dir_couplers()`, bit-identity verified; the density floor moved **inside** stage 9 to match its C++ twin |
| `verify.py` | the stale per-distance guard replaced by the per-receiver one; 3 new guards (callable definitions, the floor, and that the floor is not all the stage does). 275 → 279 PASS |
| `build.py` | audit table 10 → **11 scripts**; `VERIFY_BASELINE` 2 → 1 entry with the reason recorded |
| `gen_active_profiles.py` | interimage tier corrected 3 → 2, with the estimated input (red white-light gamma 0.55) named as the reason it is not tier 1, and `density_weighting` called out as still tier 3 |
| `doc/CHANGES_2026-08-03_v5_interimage.md` | **NEW.** The derivation four other files cited and that was not on disk — reconstructed from the code of record, with §7 stating plainly what could not be recovered |

## 10. Owner action

**None required.** No profile data changed; `film_names.txt` is unchanged, schema
is still v10, no ListBox shift, no plugin rebuild.

**Two decisions now on the queue from this work:**

1. **The `ALGO_COUPLER_MIN_SIGMA_PX` gate exists on the C++ side only** (§5).
   Below 27.8 px/mm the two renderers apply different adjacency. Either the
   reference gains the gate or the plugin loses it.
2. **The sub-pixel adjacency blur** (§4). At ordinary render sizes the edge term
   differs by up to 2.6e-2 D between implementations. The physically honest
   options are to render the term at higher resolution, or to raise the gate to
   the ~1 px where the two forms actually converge — which is a fidelity
   decision, not a tolerance one.
