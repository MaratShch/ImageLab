# Schema v7 — four inert data carriers

**Date:** 2026-08-16. **Scope:** `film_profiles.py`, `cpp_codegen.py`, `verify.py`, the
generated C++ and both copies. **No film parameter changed. No render changed.**

## What this is, and what it deliberately is not

Several classes of measured data have been sitting in extraction reports with nowhere to
go, because the schema had no field for them. v7 adds those fields — and nothing else.
Every field is **inert**: nothing in `film_sim.py` or the generated C++ reads it.

That separation is the whole design. The risk in this data was never in *carrying* it, it
was in *wiring it in*: dye-density curves fed into the colour path would change every
colour image in the database. Carrying is safe and can be done in one pass; wiring is a
per-feature change that deserves its own before/after render comparison and its own
approval. v7 is the carrying half.

This supersedes an earlier recommendation to park the data in a sidecar JSON file. Inert
fields achieve the same safety and additionally give type checking, validation at load,
provenance stored beside the value, visibility in `FilmActiveProfiles.md`, and no second
file to drift out of sync.

## The four carriers

### `SpectralDyeDensity`
Diffuse spectral density of the developed cyan / magenta / yellow dyes plus the visual
neutral, on a `lambda_start_nm` + `lambda_step_nm` grid. Densities are stored **as
printed, not normalised** — unlike sensitivity, the absolute level carries meaning.
`normalisation` records which convention the sheet used.

Why it matters: the database currently approximates a whole set of absorption curves with
a 3×3 `dye_matrix`. A matrix cannot express an unwanted absorption that peaks off-band,
which is exactly the difference between Gevacolor's 550 nm magenta and Agfacolor's 540 nm
one — a difference the sources describe explicitly. **54 vector pages** in the corpus plot
this.

### `LayerStack`
Coating order top-to-bottom by sensitisation, plus per-**layer** resolving power in that
order. Distinct from `MTFSpec`, which carries R/G/B *records*. Cheltsov & Bongard 1958
Table 24 measures physical layers, and six films there stack unconventionally
(Duponcolor positive 275 and Telcolor negative run blue / red / green from the top).
Entering those as R/G/B records would silently assert an order the film does not have, so
the validator **refuses per-layer resolving power unless the order is stored with it**.

### `ProcessingFamily` (of `DevelopmentPoint`)
The published processing axis rather than one row. `ProcessingSpec` records the single
condition the stored curve represents, which was always a stopgap: NEOPAN 1600 prints a
16-developer × 5-temperature × EI matrix, the Kodak F-5 sheets print contrast index against
time for six developers, Ilford prints time-to-CI tables. Stored as a flat tuple so the C++
emitter stays a plain array and no developer name has to become an enum.

### `ReciprocityTable`
Measured correction against exposure time. `ReciprocitySpec`'s single Schwarzschild
exponent is *exact* for Ilford — they publish `Ta = Tm^k`, so `p = 1/k` with no residual —
but provably insufficient for Kodak, whose effective exponent walks from about 0.85 to 0.70
across successive decades. `cc_filters` is optional and meaningful by its presence: a
prescribed filter documents **chromatic** failure, its absence **achromatic**, a
distinction the exponent form cannot carry.

## Validation — the rules refuse bad data, and were tested by trying

Each carrier validates on load through `FilmProfile.validate()`. Negative tests confirm
each rule actually bites:

| Attempted | Result |
|---|---|
| dye density containing a negative value | rejected — densities cannot be negative, so the trace or its baseline is wrong |
| per-layer resolving power with no `order` | rejected — would assert a stacking the film may not have |
| a development time with no measured contrast | rejected — a time with no CI or gamma says nothing |
| reciprocity times not ascending | rejected |

The rule caught its own author: the first inertness probe in `verify.py` built dye-density
data from a sine that dipped below zero, and the validator refused it. The probe was fixed;
the rule was not relaxed.

## Proof of inertness

Two independent checks, both now permanent.

**1. Before/after hashes.** Ten stocks spanning colour negative, reversal, monochrome,
period and Soviet emulsions (VISION3 250D, Portra 400, Velvia 50, HP5+, T-MAX 400,
Agfacolor Neu 1936, Ektachrome 100D, Foto-65, Tri-X Reversal, Gold 200) were rendered over
a fixed synthetic scene — gradient, saturated patches and a 6.0 highlight to exercise
halation — and SHA-256 hashed **before** the schema change. After it, all ten hash
identically. Reference in `render_ref_pre_v7.json`.

**2. A live guard in `verify.py`.** Rather than inspect the code for reads, the guard
proves the property: it takes a profile, populates **every** v7 field with plausible
non-zero data, renders both, and requires bit-identical output. Current result: `max abs
delta = 0.000e+00`. If anyone later wires one of these into the renderer without going
through the staged review, this fails immediately.

## Files changed

`film_profiles.py` — four dataclasses, four `FilmProfile` fields, validators, `__all__`,
`SCHEMA_VERSION` 5 → 7 (v6 was already in the data model but the constant had never been
bumped; it now matches). `cpp_codegen.py` — four C++ structs, four members, four emitters
plus a `_svec` string-vector helper. `verify.py` — two new guards, the version test bumped
to 7. Regenerated: `film_profiles.cpp/.hpp`, `film_enum.hpp`, `film_names.txt` (both
copies), `FilmActiveProfiles.md`, `FilmCurves.md`.

## Verification

`validate_all()` green on 143 stocks and 9 print stocks. `verify.py` **128 PASS / 2 FAIL**
— the same two long-standing failures (saturation hierarchy ordering, neighbour-pair
coupling), unrelated to this change. `film_profiles.cpp` compiles clean at `-std=c++14`.
All four v7 fields are empty on all 143 stocks; the extraction batches fill them next.
