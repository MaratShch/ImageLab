# RESULT 2026-08-30g — M2 and M3 completed in full, no follow-up rows

**Task:** owner's instruction was explicit — *"don't postpone it to G2b. I need full realization
without additional tasks creation. My goal is to unload and free up the queue, not to dump new
tasks on it."* So the two follow-up rows opened in the previous batch were closed inside this one.

**Build 0 failures / 0 warnings.** `verify.py` **425 PASS / 1 FAIL** (baseline), one check more than
before. 24 audits registered. Both Callier parity families green. Database regenerated.

⚠ **And the queue IDs were wrong.** The rows were filed as G1/G2/G3 into a queue that already has a
**G-group for GEVAERT** — the file even carries a struck duplicate G3 from 2026-08-26. Renamed to the
free **M** prefix, and every code comment referencing them corrected.

---

## 1. M2 — the discarded traces, and what keeping them exposed

The previous batch found that both dye panels draw **five** traces and only three were stored. The
reason turned out to be a schema limit talking, not an absence of data: family C in
`dye_density.py` was already *finding* the Midscale Neutral and Minimum Density traces, *validating*
them, and then discarding them — because the dyes are peak-normalised while the pair is as printed,
and one `normalisation` string could not mean both.

**Fixed at the record**, not around it: `SpectralDyeDensity.normalisation_neutral` (schema v19). Both
traces are now stored for every panel whose frame yields them cleanly — 5201, 5205, 5218, 5245, 5293.

Two returns, immediately:

⚠ **The identity became a general validator and failed three panels at once.** `Neutral − Dmin =
k(C+M+Y)` with the three k **equal** is what makes a neutral a neutral, and the coefficients are free
to be anything:

```
KODAK_VISION2_50D_5201    k = 0.628 0.604 0.595    spread 0.054   ok
KODAK_VISION2_250D_5205   k = 0.627 0.638 0.620    spread 0.030   ok
EASTMAN_EXR_200T_5293     k = 0.499 0.553 0.616    spread 0.212   <== not a neutral
KODAK_VISION2_500T_5218   k = 0.484 0.546 0.657    spread 0.306   <== not a neutral
EASTMAN_EXR_50D_5245      k = 0.621 0.539 0.738    spread 0.315   <== not a neutral
```

⚠ **`EASTMAN_EXR_200T_5293` had passed the sign test, the ratio bounds AND the Soviet cross-check,
and was already adopted.** Only the sheet's own neutral catches it. The adopted set is now **9, not
10** — and the two refusals carried over from the previous batch are now confirmed by a second,
independent line of evidence rather than by an outlier argument.

⚠ **And `d_dmin` on a masked colour negative IS THE ORANGE MASK.** Fifteen stocks now carry it. That
is the first spectral record of the mask anywhere in this database.

Getting there needed two small extractor corrections, both of which were quietly losing data:
a **constant trace** (a gridline the frame detector missed) counted as a curve and made three
leftovers where there were two; and **Kodak's red overprint** delivers the cyan path twice, byte for
byte, making four leftovers out of two on 5217.

---

## 2. M3 — Silberstein & Tuttle wired, both twins

The law, from Mees printed p644 — `10^-Dsp = E·10^-Ddiff + (1−E)·10^-(β·Ddiff)`, with **E** the
fraction of scattered light the reader accepts and **β** the film's scattering-to-absorption ratio.
⚠ C22's film × geometry split, in print since 1942.

**Wired:** `film_sim.callier_net` (the single definition), `AlgoCallierNet` / `AlgoCallierLut` /
`AlgoCallierLutAt` in `AlgoCallier.hpp`, stage 12b on the table, and **both `Algo_08_Sim.cpp` twins**
on the exact law inside the solve.

⚠ **The AVX2 problem had a data answer, not an architecture one.** The law needs two `pow` and a
`log10` per channel per pixel and neither has an AVX2 intrinsic. Solved with a **1-D lookup over net
density** — 1025 entries over −1.0 to 5.0, linear interpolation, end-slope extrapolation — built
identically by Python and both flavours, so **parity holds by construction rather than by tolerance**.
Measured interpolation error against the exact law: **2.2e-07**.

⚠ **The solve evaluates the law exactly and only the pixel pass uses the table.** The solve touches a
handful of scalars per iteration, where two `pow` cost nothing and an exact answer is worth having.

⚠ **A header note said this stage must be split into twins if it ever grew a branch or a table. M3
gave it both, and the note was overridden with the reason recorded in place**: the branch and the
table are now the *law itself*, shared by the solve, both flavours and Python. Splitting would create
two spellings of one law — the thing the note was protecting against. Accepted cost: the stage no
longer auto-vectorises and runs scalar in both builds at a non-zero setting. It ships at zero, where
it returns before touching a pixel.

⚠ **Inertness is a property of the GUARD, not of the law's arithmetic, and conflating them cost a
build.** At E = 1 the law reduces to `-log10(10^-d)` — mathematically the identity, and **not
bit-exact**: measured departure 5.6e-17. So all three implementations test `callier_is_inert` and
return early. `verify.py` now asserts the *pipeline* is bit-identical at specular 0, which is the
thing that actually matters, instead of asserting a round-trip that was never going to hold.

⚠ **The AVX2 twin of the solve is compiled by no audit** — this flattened tree resolves
`#include "AlgoTypes.hpp"` to the scalar copy, so that flavour builds only in the owner's real
project layout. `TWIN_LAW_TOKENS` therefore gained `Algo_08_Sim.cpp`; a textual twin check is the
only automatic guard on a file that was just rewritten by hand in two places.

**Parity:** Callier law 1.43e-07, STAGE 1.97e-07, SOLVE 2.77e-07, 340 monochrome rows moving, **0
colour stocks moving**. `callier_silberstein_tuttle.py` now also asserts that the *shipped* law is
still the book's, against its own independent reading — worst 0.0e+00.

⚠ **No shipped render moved.** Both laws are exactly inert at `specular = 0`.

---

## 3. ⚠ Two of my own checks were asserting the old behaviour, and one hid 300 others

`verify.py` carried a check requiring 5201 to store **no** neutral trace. It was right when written
and became wrong the moment the schema could hold one; it is now reversed, and asserts the identity
instead.

More seriously: a `NameError` in my first rewrite of the Callier check made `verify.py` **abort a
section**, and the build reported **124 PASS / 1 FAIL** — where a healthy run is 425. ⚠ The failure
count looked *better* than before. This is the same trap `PROGRESS.md` already records for skipped
audits: a count that drops is as much a defect signal as a count that fails, and only the PASS total
shows it.

---

## 4. Queue

**Closed:** M2, M2b, M3, M3b. **Opened: nothing.** Net change **−4**.

The queue also gained a **blocker review** (section 0) at the owner's request — see the delivered
`DIGITIZATION_QUEUE.md`. Thirteen rows remain open: three blocked on nothing, three on an owner
decision, seven on a named document. Only **C39** has a live render defect behind it.

---

## 5. Files

**Changed:** `film_profiles.py` (schema v19 field, the neutral/dmin table, the matrix table
regenerated to nine), `film_sim.py` (the law, the LUT, both consumers), `dye_density.py` (the two
traces returned and the identity reported), `dye_matrix_from_spectra.py` (the neutral identity as the
primary refusal), `verify.py` (three checks), `cpp_parity.py` (law family, twin tokens),
`callier_silberstein_tuttle.py` (adoption recorded, shipped-law assertion),
`AlgoCallier.hpp`, `Algo_08_Sim.cpp` and `AVX2/Algo_08_Sim.cpp`, `doc/DIGITIZATION_QUEUE.md`, and the
regenerated C++ data files.
