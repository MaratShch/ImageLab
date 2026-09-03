# Should the data-collection phase close? An objective assessment

**Date 2026-08-17. Requested by the owner.** Opinion document — it contains no adopted
values and changes no profile.

**Scope limit stated first, because it bounds everything below.** This assessment is derived
from the database, the generator sources and the C++ stage list. **I have not seen a render
from this engine, and I have not compared one against a real film scan.** Every statement
about *visual* fidelity below is therefore an engineering inference from the model's
structure, not an observation of its output. Where that distinction matters I say so.

---

## 1. The short answer

**Close the broad data-collection phase. Keep four narrow seams open.** The database is no
longer the binding constraint on how close a frame can get to a real scan — with the
specific exceptions in §5. The binding constraint has moved to two other places: **the
absence of a ground-truth comparison loop**, and **four algorithmic gaps**, one of which
(grain statistics) is the most likely thing to give the simulation away to exactly the
professional audience named as the test.

The reason is not that the archive is exhausted. It is that the parameters still missing
divide into two classes that need opposite responses:

| Class | Examples | Right response |
|---|---|---|
| Missing from **your archive**, obtainable from documents | MTF curves (199 vector pages inventoried, 26 curves traced off 12 sheets so far), spectral dye density (10 film profiles + 1 print stock done as of 2026-08-18, per `ROADMAP_2026-08-17_fidelity.md` §1.1), print and intermediate stocks, σ(D) for a B&W silver negative | **Keep collecting.** Cheap, mechanical, high yield |
| Missing from **the published literature itself** | interimage effects, DIR coupler coefficients, developed clump geometry, layer thicknesses | Searching cannot fix this. Only **measurement from real material**, or the patent literature |

`FilmActiveProfiles.md` already states the second class plainly: interimage is tier 3 for
**every stock without exception**, and all 395 documents in `PDF/PROFILES` were searched
without finding one that publishes it — because camera negative is characterised with a
single white-light exposure series. Clump geometry is never printed by anyone. No further
document hunt changes those two facts. That is what "diminishing returns" concretely means
here, and it is a property of the world, not of the effort so far.

## 2. Where the database actually stands

**159** stocks, 9 print stocks, 14 formats, schema v10. The confidence-tier split
**68 / 46 / 40** (T1/T2/T3) and "139 of 154 cite at least one document" are the 2026-08-17
figures and have not been re-counted here. Coverage by property, from the generated report of
that date:

| Property | Documented | Comment |
|---|---:|---|
| Spectral sensitivity | 53 % | best-evidenced, machine-traced, drives balance in both builds |
| Characteristic (H&D) curve | 44 % | traced to 0.002–0.03 D RMS — better than an eye by an order of magnitude |
| MTF / resolving power | 49 % | ⚠ **no longer just one f50 per layer** — the rolloff shape is stored and read, measured on **8 stocks** off 26 traced curves; the other **63 colour stocks** still carry an estimated f50 triple (§4) |
| Processing | 26 % | |
| Grain characteristics | 20 % | |
| Colour characteristics | 14 % | |
| Latitude / dynamic range | 13 / 12 % | mostly derivable from the traced curves |
| Additional physical | 8 % | |
| RMS granularity | 6 % | |
| **Grain σ(D) shape** | **3 %** at the time — **12 stocks carry `sigma_shape_measured`** as of 2026-08-23 | four stocks were adopted on the day of this assessment; per-layer rms is separately measured on 11 stocks |
| Film base / emulsion | 3 / 3 % | |

**The distribution is the good news, not the bad news.** The two properties that dominate
what a frame looks like — tone reproduction and spectral response — are the two best
evidenced, because manufacturers print those plots and they can be traced exactly. The
properties sitting at 3–6 % are, with the exceptions in §5, the ones nobody published.
Raising "film base properties" from 3 % to 30 % would change a base tint by a few
thousandths of a density unit. It is not where a frame is won or lost.

## 3. Film types and categories: what is genuinely missing

Coverage of the canonical 20th–21st century stocks is broad. Present: Kodachrome (3
generations), Portra (4), Ektar, Ektachrome (5), Vision2 (3) and Vision3 (4), Tri-X (4),
T-MAX (3), Plus-X, Panatomic-X, Double-X 5222, HP5+, FP4, Pan F, Delta 3200, Acros, Velvia,
Provia, Eterna, Gold, Ultramax, Agfacolor (3, including the 1936 and 1943 types), ORWO (3),
Technicolor three-strip, Dufaycolor, 14 Svema, 2 Tasma, 11 Polaroid.

**The one category gap that matters for the stated goal, and it matters a lot: the
positive/print and intermediate chain.**

Nine print stocks exist, several of them reconstructions (`DUPE_FINE_GRAIN`,
`TECHNICOLOR_IB`, three Soviet TSP positives, `EASTMANCOLOR_5382_1953`). Missing, **with
datasheets already sitting in `PDF/PROFILES/KODAK/`**:

* **KODAK VISION Premier 2393** print stock — sheet on file;
* **VISION Color Intermediate 2242 / 3242 / 5242** — sheet on file, *two* of them;
* 3383 print, 2302 B&W print, 2374 sound.

This is the biggest category gap because of what the goal actually asks. "Indistinguishable
from a real film scan" is ambiguous until you say **a scan of what**: a camera negative, an
interpositive, a release print, or a negative that went through an IN/IP dupe generation.
Those four look nothing alike — a print has a gamma near 2.5–3.0, crushed blacks, a
completely different grain structure (the print's own grain on top of the negative's,
partially decorrelated), and its own dye set. Most film footage a professional has ever
graded was one of the *later* links in that chain, not the camera negative. The engine has
the stages (13 duplication, 14 transmittance) and one properly documented print stock
(2383, vector-traced). Completing the chain is the highest-value data work remaining.

Minor absences, low priority, listed for completeness: Fujicolor Superia and Reala, Pro
400H/800Z, Ilford XP2, Instax, Aerochrome, microfilm/Copex. Consumer or niche — they add
breadth, not peak fidelity.

## 4. Where the data is too sparse or too simplified for high-fidelity simulation

Four places. Ordered by how likely each is to be noticed by the professional audience named.

**(a) Grain is modelled as a Gaussian random field. Real grain is not Gaussian.**
`AlgoGrain.hpp` is unusually well-reasoned — √D amplitude from Poisson counting, a real
power spectrum with a clumping lobe, one field for monochrome, amplitude calibrated against
the *continuous* spectral integral so fine grain does not alias. All correct, and better
than most implementations. But a filtered Gaussian field has **Gaussian marginals at every
density**, while a developed emulsion is a clustered point process whose marginal
distribution is skewed and whose skew changes with density. At 100–200 % magnification —
which is the first thing a restoration specialist or colorist does — the difference reads as
"too even, too synthetic" even when RMS and power spectrum both match. This is a modelling
gap, not a data gap: no vendor number fixes it. It is, in my judgement, **the single most
likely tell in a still frame** from an otherwise well-parameterised render.

**(b) There is no scanner noise model at all.** I grepped the whole engine for read noise,
shot noise, photon noise, sensor noise: nothing. Scanner *MTF* is modelled, carefully, and
in the right place in the chain (stage 10 before grain, so it band-limits both — that
ordering decision is correct and non-obvious). But a real scan's noise floor is film grain
**plus** sensor noise, and the two have different spectra and different density dependence:
sensor noise is roughly constant in *transmittance*, so in density it explodes in the Dmax
shadows exactly where film grain is dying away. A render whose black floor contains only
grain has the wrong noise *shape* in the shadows. Cheap to add, needs one measurement from
whatever scanner is the reference.

**(c) MTF is one Gaussian per layer; real MTF overshoots 100 %.** `MTFSpec`'s own docstring
admits it: the Gaussian is a fair fit through the mid band, but a real emulsion MTF rises
*above* unity at low frequency from adjacency/development effects, and that overshoot is a
visible part of the "film look" — local edge contrast that no Gaussian can produce. The
⚠ **SUPERSEDED 2026-08-19 (queue item C2), extended 2026-08-23:** the MTF rolloff IS now stored
and read in both renderers, and the two fields named below were scored against a traced curve and
REJECTED as the wrong form (rms 0.0583 vs the adopted power law's 0.0375). **8 stocks now carry
`mtf_measured`** from 26 curves traced off 12 sheets; the power law beats the Gaussian on all 26,
and q_R ≤ q_G ≤ q_B holds 8/8 — ⚠ but the exponents are **not** per-layer constants (red 1.89–2.77,
blue 2.38–3.42), so q stays per-stock measured. ⚠ **The f50 estimating rule is also reversed:**
`f50_r ≈ 0.78 × f50_b` is wrong in form — measured red f50 is effectively constant at
36.4 cycles/mm (32.1–41.1, ±13 %) while green spreads 52 % and blue 70 %. Original text:
`mtf_tail_a` / `mtf_tail_f_exp` fields exist but are wired only into the C++ port, and
**119 vector MTF pages are sitting unread in the archive** (remeasured since: 199 pages). This is
still a large body of un-harvested *exact* data, and sharpness is first-order visible.

**(d) Spectral dye density: 2 of 154 when this was written — 10 film profiles plus 1 print stock
as of 2026-08-18** (`ROADMAP_2026-08-17_fidelity.md` §1.1), against 159 stocks today. The schema
field exists and works (5285 and 2383 were the first two,
both validated against their own neutral trace). This governs how the three dye layers
actually mix as seen through a *particular* scanner's illuminant and filter set. Without it,
colour under a real scanner is modelled by a 3×3 `dye_matrix` — a linear approximation to a
spectrally selective process. For matching a specific scanner's colour, this is the gap.

**Also, and decisively if the target is moving images: four pipeline stages are stubs** —
temporal flicker (3c), negative defects (9b), gate weave (15), gate defects (16). For a
single frame this costs little. For a *sequence* it is close to fatal: weave, breathing and
frame-to-frame flicker are how the eye recognises film in the first two seconds, before it
judges grain or colour at all. If the ambition includes motion, these four outrank every
remaining data task in this document.

## 5. If you continue collecting: the priority order

Small, mechanical, high yield — worth doing:

1. **The print and intermediate chain** (§3). 2393, 2242/3242/5242, and the older
   Eastmancolor positives. Datasheets on file. This is what decides whether the output can
   claim to be a scan of *release material* rather than of camera negative.
2. **The vector MTF pages** — 199 inventoried, 26 curves traced off 12 sheets so far → MTF as a
   curve with the overshoot, replacing one f50 per layer for the 63 colour stocks still on an
   estimated triple. Exact vector coordinates, purely mechanical work, first-order visible
   property. ⚠ Note what it will *not* buy: q and f50 do not generalise across stocks, so every
   stock needs its own trace.
3. **Spectral dye density** for the stocks whose sheets carry it as vector art — start with
   the stocks you actually intend to target, not alphabetically.
4. **One measured σ(D) for a B&W silver negative** (already queued). Small, and it unblocks
   correcting the σ(D) heuristic sign for 103 stocks.

Worth trying once, then abandoning if barren:

5. **The patent literature for what datasheets never print.** `next_week_task.md` already
   names it: Google Patents class G03C, assignee Eastman Kodak / Fuji. Worked emulsion
   examples in patents do sometimes give DIR coupler concentrations and layer thicknesses —
   the exact tier-3 parameters no datasheet publishes. Treat as speculative: patents
   describe *examples*, not shipped products, so anything found is [T3] provenance at best
   and must be labelled as "a patent example for a film of this class", never as this
   stock's specification. I would time-box this hard.

Not worth continuing:

6. **Breadth for its own sake.** Going 68 → 100 tier-1 stocks improves how many stocks you
   can render faithfully. It does not improve how faithful any single one is. If the dream
   is one frame a colorist cannot place, breadth contributes nothing to it.

## 6. On the 90–95 % goal, honestly

Two things need saying, and neither is discouraging.

**First: the target is not yet measurable, and that is the real bottleneck.** "90–95 %
similarity" has no definition in the project today. `verify.py` — **304 checks, 303 PASS / 1 FAIL**
as of 2026-08-23 (the failure is the saturation-hierarchy ordering, known and left alone), plus
11 audit scripts all green, including cross-language parity audits of the reciprocity and
DIR-coupler stages against the plugin's own C++ — a genuinely
strong suite — validates the model against **datasheet figures and internal physics**. It
never compares a render against a real scan; `image_diff.py` is an exact per-pixel differ,
not a statistical comparator. You cannot optimise toward "indistinguishable" without a
measured target, and every remaining improvement is currently being chosen by judgement
rather than by measured error. **This is still data collection — but of your own
measurements, not of vendor documents.** In my opinion it is worth more than items 1–5 in §5
combined, because it converts all later work from opinion into arithmetic.

Concretely, what a comparison loop needs: real scans of a known stock at a known exposure
(a grey scale and a colour target would do most of the work), then per-density-band noise
power spectra per channel, cross-channel noise correlation, the density histogram in a
defined metric, MTF from a slanted edge, and dye crosstalk from the colour target. Five
numbers per axis, compared against the render. That is a tractable amount of work and it
would immediately tell you whether §4(a) and §4(b) are real problems or ones I have
overestimated from reading code.

**Second: my honest expectation of what a professional would catch first**, in order —
grain structure at magnification; the noise floor in the deep shadows; highlight rolloff and
halation around small bright sources; and in motion, weave and flicker. Tone-curve accuracy,
which is what this database does best, is the thing least likely to give it away. That
ordering is an engineering judgement, not a measurement, and the comparison loop above is
what would confirm or refute it.

**Is the dream reachable?** For a **single still frame**, of a **specific well-documented
stock**, viewed at normal magnification — I think the model already has the structural
ingredients, and the remaining distance is mostly §4(a), §4(b) and calibration against real
material. For **moving footage**, the four stubs have to be built first; no data will
substitute. For **arbitrary magnification against an expert with a loupe and a densitometer**
— that is a harder bar than 90–95 % and I would not promise it from the current grain model.

## 7. Recommendation

1. **Declare the broad document hunt closed.** Record it as a decision, with the reason:
   the remaining unknowns are absent from the literature, not from the archive.
2. **Build the comparison loop.** Pick **two or three** target stocks — the obvious
   candidates being 5219 plus 2383 as a negative/print pair, and one still stock you have
   real scans of. Depth on three beats breadth on fifty for this goal.
3. **Then spend effort in this order:** grain marginal statistics → scanner noise →
   the four motion stubs (if motion is in scope) → MTF as a curve (⚠ the rolloff half is done —
   §4(c); what remains is the adjacency overshoot and one trace per stock) → the print/intermediate
   chain → spectral dye density.
4. **Keep the four seams in §5 open as background work**, not as the main line.

The database has done its job. What it cannot do is tell you how close you are — and that is
now the thing most worth building.
