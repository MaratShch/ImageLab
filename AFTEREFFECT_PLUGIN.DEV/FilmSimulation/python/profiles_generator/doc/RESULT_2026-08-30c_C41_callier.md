# RESULT 2026-08-30c — C41: Callier wired, and a header that would have misled whoever wired it

**Task:** queue **C41**, owner chose **option A** — wire it properly, both halves, both twins, with a
stage-level parity probe. Closed.

**Build 0 failures / 0 warnings.** `verify.py` 424 PASS / 1 FAIL (baseline). Both engines end to
end: **scalar PASS 0 failures over 165 stocks**, **AVX2 PASS 0 failures over 165 stocks**, with
stage 12b present in all 165 profiles of each run. Database untouched.

---

## 1. ⚠ The header named three consumers and two of them do not exist

This is the finding, and it is why the review the owner asked for on 2026-08-29 was worth doing
before the wiring rather than after.

| `AlgoCallier.hpp` said | reality |
|---|---|
| `AlgoSolveAnchors` | ✅ correct — `film_sim` applies the factor at two points inside the solve |
| `AlgoNeutralMidDensity` | ⚠ **wrong.** `film_sim.neutral_mid_density()` takes no `scanner_specular` argument and applies nothing. **Wiring it would have created a divergence** |
| `AlgoStage12_DyeImpurity` | ⚠ **wrong about the location.** Python runs Callier as its own **stage 12b**, after 12 and before 13 — not folded into the dye matrix |

Wrong in the direction that costs work: a reader implementing from that list would have produced
two new Python/C++ splits while believing they were closing one. Corrected in place, with the
reasoning kept so the correction is checkable rather than just asserted.

⚠ **And the one measurement the whole argument rests on is recorded twice, differently.** The header
said mid grey moved **+54/255** against a contrast change of **22 %**; `film_sim.py`'s own call site
says **+48/255** against *"a few per cent"*. One experiment, two write-ups, and nothing in the
repository settles which transcription is right. The exact figure is no longer quoted anywhere — what
the argument actually needs is the **order** of the two effects, and both records agree on that.
Re-measure before quoting a number.

---

## 2. What was wired

**The solve** — `AlgoSolveAnchors` gains `scannerSpecular`, computes the factor once, and applies it
at exactly the two points `film_sim` does:

- reversal branch, on `mixed` before `normalisedTransmittance` — mirroring `_cal_apply([mixed]*3)[c]`
- negative branch, on `dMid[c]` after the dye matrix — mirroring `d_mid = _cal_apply([...])`

**The pixel pass** — new `AlgoStage12b_Callier`, pointwise, **in place on the stage-12 planes**. That
placement is what made this cheap: the operation needs no scratch buffer, so stage 13 reads the
corrected values from the same pointers and `AlgoMemHandler` did not have to be re-costed for a
control that ships at zero.

**`AlgoNeutralMidDensity` deliberately unchanged** — see §1.

**Both twins moved together.** `AVX2/Algo_08_Sim.cpp` took the identical solve change in the same
pass, by the project's rule that a twin may differ in *how* it computes and never in *what*.

⚠ **One implementation serves both flavours for the stage itself, and that is a decision rather than
an omission.** `AlgorithmMain.cpp` is already shared, so the inline compiles once per flavour with
that flavour's `AlgoType` and flags. The body is a branchless multiply-add over contiguous
`RESTRICT` planes — the one shape a compiler vectorises reliably — so hand-written intrinsics would
duplicate a two-operation law to buy nothing, and duplicating a law is what the twin check exists to
prevent. The header records the condition under which that stops being true: **if this ever grows a
branch or a table it must be split into twins like `Algo_11`.**

---

## 3. Inert by default, and asserted as an identity rather than a tolerance

`scannerSpecular` defaults to 0 → `AlgoCallierFactor` returns exactly 1.0 → `AlgoCallierApplyScalar`
returns its argument untouched and the stage returns before touching a pixel. Every render made
before this existed is reproduced.

That is checked as **equality**, not closeness. "Close enough" is not the contract: a last-bit change
on every density in the frame is still a changed render.

---

## 4. The guard, and why a stage-only probe would have been useless

⚠ **The existing Callier family passed for a week while nothing called the law.** 11 880 probes,
worst 1.43e-07, on two functions the pipeline never invoked. Same shape of hole as C30/C33.

The new family drives the code that renders:

```
[i] Callier STAGE: 2475 probes over 165 stocks x 3 specular, driving
    AlgoStage12b_Callier and AlgoSolveAnchors themselves
[i] Callier STAGE: worst stage 3.83e-07 at ('CS','AGFA_SCALA_200X',1,3);
    worst SOLVE 2.77e-07 at ('CA','KODAK_PORTRA_400VC',0,0)
[i] Callier STAGE: 272 monochrome rows move at full specular,
    0 colour stocks move (must be 0)
```

⚠ **The SOLVE half is the load-bearing assertion.** A pixel pass without the solve moves mid grey by
more than it changes contrast — that is the configuration this task was approved to prevent, and a
probe that checked only the stage would pass on it happily. Five assertions, and three of them exist
to stop the probe fooling itself:

- stage agrees with `film_sim.callier_density`
- **solve agrees with `film_sim.solve_anchors` at the same specular**
- inert at 0, as an identity
- **0 colour stocks move at any setting** — Q = 1.0 on all of them because a dye image does not scatter
- **≥ 100 monochrome rows must move at full specular**, or the probe is not exercising its own branch

⚠ The probe walks the **real database** rather than literals, unlike the law family beside it, because
`AlgoSolveAnchors` reads the whole profile — curves, dye matrix, couplers, taking matrix, print stock.
A hand-built stub would be a different film.

---

## 5. What is still not right, and it is the data half

⚠ **The film half of the product remains a class estimate.** The two monochrome values — 1.3 negative,
1.25 reversal — come from a generator rule, not from any document in the corpus. The geometry half is
exact. That asymmetry is why the control still ships at **zero** rather than at some "typical
scanner" value, and it is unchanged by this work: C41 made the mechanism correct, not the number.

What would fix it: one densitometer specification stating a diffuse-versus-specular ratio for a named
emulsion. ⚠ Note the measured Tri-X quotient already in the database — 2.0 to 2.34 at 0.0016 sr — is
**not** that number and must not be adopted as one: that collection angle is nearly collimated, while
1.3 corresponds to a real condenser cone. Both figures are deliberately kept findable, and a
`verify.py` guard pins the distinction.

---

## 6. Files

**Changed:** `AlgoCallier.hpp` (consumer list corrected, measurement discrepancy recorded,
`AlgoStage12b_Callier` added), `AlgoCharacteristicCurve.hpp` (declaration), `Algo_08_Sim.cpp` and
`AVX2/Algo_08_Sim.cpp` (the solve, both twins), `AlgorithmMain.cpp` (stage 12b call, specular passed
to the solve), `cpp_parity.py` (the stage family), `doc/DIGITIZATION_QUEUE.md` (**C41** closed).

**Not changed:** `AlgoNeutralMidDensity`, the database, and `film_sim.py` — Python was already the
reference implementation and needed nothing.
