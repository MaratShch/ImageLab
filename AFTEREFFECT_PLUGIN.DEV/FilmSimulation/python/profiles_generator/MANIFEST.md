# Delivery manifest — film simulation project, 2026-09-02d

All five archives are cut from the same tree state, immediately after a green
`build.py --root <cpp tree>`:

* `verify.py` **509 PASS / 1 FAIL** — the single failure is the saturation-hierarchy
  ordering, the known baseline the owner instructed to leave alone.
* **30 audit scripts registered**, all green (5 SKIP for sources not present in this
  working copy).
* `doc_consistency.py` **31/31** registered documentation counts match the live database.
* `g++ -std=c++14 -Wall -Wextra` clean on **all 18 translation units**, exit 0 and
  **zero bytes of output**.
* `cpp_parity.py`, `interimage_parity.py` and `spectral_mono_parity.py` green.

**Database:** 172 film stocks, 11 print stocks, 14 gauges, schema **v24**.
132 negative / 40 reversal; 69 monochrome. Provenance 85 T1 / 45 T2 / 42 T3.

| # | archive | contents |
|---|---------|----------|
| 1 | `1_python_source.zip` | The generator and renderer: `film_sim.py`, `film_profiles.py`, `build.py`, `verify.py`, `cpp_codegen.py`, every audit/extraction reader, `film_ids.lock`. **No PDFs, no generated C++.** |
| 2 | `2_film_database.zip` | The generated database only: `film_profiles.hpp/.cpp`, the 16 `film_profiles_data_NN.cpp` shards, `film_profiles_detail.hpp`, `LoadFilmDataBase.h/.cpp`, `film_enum.hpp`, `film_names.txt`, `film_display_order.txt` |
| 3 | `3_algorithm_scalar.zip` | The complete scalar reference implementation: every `Algo*.hpp` / `Algo*.cpp` and its support headers at the tree root, **excluding** the `AVX2/` directory and the generated database |
| 4 | `4_algorithm_avx2.zip` | The complete AVX2 production path: the whole `AVX2/` directory, plus the shared headers it compiles against |
| 5 | `5_documentation.zip` | Every `doc/*.md`, reviewed and reconciled — not merely appended to |

## What changed in this session, in order

1. **G2, C44, C43, C4, C7** + **J1/J2** (Ooue 1959) — earlier batch.
2. **TK1–TK5** — Takano 1969. First check of the engine's aperture term (it fits);
   the five-point clump census C45 was missing; σ(D)↔σ(T) to fourth order adopted as an
   inert helper; the print-chain law eq (13) checked against the engine, with one
   departure recorded.
3. **E5, C45, C46, C16, C18, C19, C2c** — one trace and six decisions. ⚠ Three of the six
   rows rested on a mistake in the row itself. C18 bounded the largest undocumented number
   in the colour path at the curve's own Dmax, provably non-binding. **The owner-decision
   category of the queue is now empty.**
4. **N1** — `FUJI_NEOPAN_SS` added as stock 172 from FUJIFILM AF3-411E(N).

**Queue: 114 rows, 96 closed, 16 live.** Not one live row is a judgement call.

## Two things to read first

* `doc/PROGRESS.md` — one screen of current state.
* `doc/RESULT_2026-09-02b/c/d_*.md` — this session's three result documents.

⚠ **Two adoptions were made and then withdrawn in this session, and both are documented
rather than erased**: the E5 σ(D) shape (wrong density space, caught by `cpp_parity`)
and the C45 clump rescale (refused with the cost of the alternative measured).
