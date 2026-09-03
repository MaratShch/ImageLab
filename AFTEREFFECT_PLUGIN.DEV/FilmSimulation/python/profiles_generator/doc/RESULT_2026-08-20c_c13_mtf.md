# RESULT 2026-08-20c — C13: 5274's MTF adopted, and the finding outgrew the profile

> ⚠ **SUPERSEDED IN PART ON 2026-08-23 BY C2b/C24 — read this with the correction.**
> This document's §3 proposed that the f50 estimating rule might be *re-derived* from the
> layer stack once more colour sheets agreed, and recorded a q clustering ("both reds
> 1.84–1.89, both blues 3.38–3.42") off **two** stocks. With seven:
> 
> * **The rule's FORM is wrong, not its constant.** Measured red f50 is effectively fixed at
>   **36.4 cycles/mm (±13 %)** while green spreads 52 % and blue 70 %, so `f50_r = k·f50_b`
>   cannot fit at any k. Five modern Kodak cine reds are now anchored at 36.0.
> * **The q clustering was a two-sample illusion.** The ORDERING `q_R ≤ q_G ≤ q_B` holds on
>   8 of 8 sheets, but red spans 1.89–2.77 and blue 2.38–3.42 (sd 0.32–0.37), so q is NOT
>   derivable and stays per-stock measured.
> * Counts in this file ("third stock with a traced MTF", "92 colour stocks") were true on
>   2026-08-20c; the live figures are **8 measured stocks and 63 estimated triples**.
> 
> The body below is left exactly as written — the reasoning is the audit trail.
> See `RESULT_2026-08-23b_c2b.md`.

**Task.** Queue **C13**, owner-approved: adopt the measured MTF for
`KODAK_VISION_200T_5274` over its stored estimate.

**Outcome.** Build **OK**, 11 audits green, `verify.py` **284 PASS / 1 FAIL** (the one known),
C++ clean on 18 TUs. **Schema v10 unchanged, `film_names.txt` unchanged** — no ListBox shift,
data-only rebuild.

---

## 1. What changed

| | stored estimate | **measured** | |
|---|---|---|---|
| f50 red | 56.0 | **35.4** | estimate was **1.58× too sharp** |
| f50 green | 64.0 | **68.8** | confirms, 7 % |
| f50 blue | 72.0 | **74.0** | confirms, 3 % |
| adjacency | 0.09 | **0.162** | green record |
| rolloff q | — (Gaussian) | **2.94** | `mtf_measured=True` |

**Source:** Kodak **H-1-5274** p3, "MODULATION-TRANSFER CURVES", plot **F010_0006AC**, read by
`mtf_vector.py`. This panel had never been traced — only the granularity panel on the same page
had, under C1c. Third stock in the database with a traced MTF.

Per-record detail: overshoot **+0.027 / +0.162 / +0.234** peaking at **2.4 / 11.0 / 16.1**
cycles/mm; rolloff **q = 1.89 / 2.94 / 3.38**. Red fits at rms **0.0149** and beats the legacy
Gaussian by **4.2×** — the best carrier fit measured anywhere in this corpus.

⚠ **The red overshoot is a lower bound.** The red curve is still rising at the panel's left edge
(2.4 cycles/mm), so its true peak is off the plot. Stored `adjacency` is green's, the visually
weighted record; the spread is recorded rather than averaged.

⚠ **`adjacency_um` 18.0 is contradicted again.** Green's overshoot peaks at 11.0 cycles/mm, a
spatial scale of ~91 µm. **Fourth stock, same direction** (5231 4.7 vs 16.0; F-125 ~9 vs 13.0;
5201 10.7 vs 16.0; 5274 11.0 vs 18.0). Left alone — queue **C2c**.

## 2. ⚠ Render impact is scale-dependent, and smaller than 1.58× sounds

Measured on a bar-sweep target, `super35`, grain and flare off, worst channel delta:

| render width | px/mm | bars at | worst | mean |
|---|---|---|---|---|
| 1200 px | 48.2 | 8.0 c/mm | **3.9/255** | 0.72 |
| 2400 px | 96.4 | 9.6 c/mm | **7.1/255** | 0.98 |
| 4800 px | 192.8 | 9.6 c/mm | **11.1/255** | 0.93 |

**The reason matters more than the numbers.** f50 lives at **35–74 cycles/mm**. A 2K render of a
35 mm frame is ~48 px/mm, so the highest frequency it can carry at all is 24 c/mm and real image
detail sits far below that. At those frequencies the two MTF curves are nearly identical — so most
of the visible change at preview size comes from the **adjacency term** (0.09 → 0.162), not from
f50 at all.

**The f50 correction earns its keep at scan resolution, not at 2K.** Worth saying plainly before
anyone judges this change on a 1080p preview and concludes it did nothing.

⚠ **A measurement error of my own, recorded because it is the instructive part.** The first impact
test reported **0.1/255** and I was one step from publishing it. The test image was 96 px wide for
a 24.9 mm frame — about **3.9 px/mm** — where a 35-versus-56 cycles/mm difference *cannot* show,
because the target carries no frequency above ~2 c/mm. **A null result from a target that cannot
resolve the effect is not a null result.** Same class as method rule 22: the instrument has to be
able to see the thing before its silence means anything.

## 3. The finding that outgrew the profile — new queue item C24

Green and blue confirmed the estimate. Red did not, and the reason is structural: the estimating
rule takes one number and scales it by a fixed layer-order ratio.

**Measured distribution of `f50_r / f50_b` across the 92 colour stocks that still carry an
estimate:**

| band | stocks |
|---|---|
| < 0.55 (inside the measured range) | 1 |
| 0.55 – 0.65 | 1 |
| 0.65 – 0.75 | 8 |
| **0.75 – 0.85 (the typical estimate)** | **72** |
| ≥ 0.85 (red nearly as sharp as blue) | 10 |

**Both stocks measured per-record land far below that:** 5274 = **0.478**, 5201 = **0.578**.
5274's own stored ratio was 0.778 — right in the mode.

⚠ **C24 explicitly refuses to rescale 92 profiles from two measurements.** That is precisely the
class-estimate-from-one-sample error method rule 18 forbids, and it would be worse than it looks:
both measured stocks are **Kodak cine negatives**, a decade apart, i.e. **one family**. Two
samples from one maker is not a law.

**What makes it answerable rather than a guess** is the layer-depth pattern, which now shows up in
two independent parameters:

| | red | green | blue |
|---|---|---|---|
| q, 5201 | 2.77 | 3.23 | 3.42 |
| q, 5274 | **1.89** | 2.94 | **3.38** |
| q, 5231 (mono) | 1.84 | — | — |
| f50_r/f50_b measured | 0.478 (5274) / 0.578 (5201) | | |

Both reds cluster at **1.84–1.89**, both blues at **3.38–3.42**, on stocks a decade apart. If two
or three colour sheets **from other makers** agree, the ratio can be **derived from the layer
stack** instead of assumed. If they scatter, the rule stays and the estimates stay flagged as
estimates. Either outcome is progress; guessing now is not.

**So C2b's remit is narrower on purpose from here: trace COLOUR sheets.** A monochrome sheet has
one record and cannot test a per-record ratio, which is exactly what C24 needs.

## 4. Guards added

* 5274 carries its measured triple exactly, and red < green < blue;
* ⚠ **every measured colour stock is softer in red than the estimating rule** (`f50_r/f50_b < 0.65`)
  — the only place the C24 comparison exists as an assertion rather than prose, so a future
  "tidy-up" toward the family ratio fails loudly;
* the three measured rolloff exponents are all distinct;
* the existing "the **two** measured exponents disagree" guard was **reworded to a spread over every
  measured stock**, because it named a count and would have gone stale the moment a fourth arrived —
  which is the same failure mode as the stale per-distance interimage guard fixed on 2026-08-20b.

## 5. Owner action

**Rebuild the plugin — data only.** Schema **v10 unchanged**, `film_names.txt` MD5 **unchanged**
(`e8dc2cb9…`), no enum or ListBox movement. One stock's sharpness and adjacency changed; nothing
else in the database moved.
