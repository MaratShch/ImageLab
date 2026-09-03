# RESULT 2026-09-03 — G5, the Gevachrome gradation curves

Owner-approved, executed without pause. Six channel shapes upgraded **[T2] → [T1]** on
`GEVACHROME_600` and `GEVACHROME_605`; **Bild 6 adopted as the only measured reversal push in the
database**. `verify.py` **526 PASS / 1 FAIL** (the saturation-hierarchy baseline, left failing on
the owner's instruction).

⚠ **This row asked the owner for a re-scan it did not need, rated itself high impact for work
another row had already done, and was blocked by a mistake in the reader. Most of this document is
about that.**

---

## 1. What the row claimed, and what was true

`DIGITIZATION_QUEUE.md` G5 asked for **a 300+ ppi grayscale re-scan of printed pages 260, 262 and
264** of Kino-Technik 1968 Nr. 10, naming four things the existing scan blocked and rating the row
**high — it unblocks G2 and upgrades two profiles from [T3] estimates to traced curves**.

| the row's blocker | the truth on 2026-09-03 |
|---|---|
| tracing Bild 2a/2b spectral sensitivity | ⚠ **already traced and adopted by G2**, 2026-09-02 |
| tracing Bild 1a–c MTF | ⚠ **already traced and adopted by G2**, and it replaced [T3] estimates that were 2–3× too high |
| bleed-through defeats layer separation, so re-scan in colour | ⚠ **the scan is grayscale in an RGB wrapper** — chroma exactly 0.0. Colour separation was never available to defeat anything |
| separating Bilder 5a/5b into three layer curves ("they sit within 1–2 px") | the only real one — **and it was the reader's fault, not the scan's** |

⚠ **The re-scan was never needed.** The first attempt traced a **114 ppi page render**. The PDF's
embedded images are **150 ppi** (940×940, 942×1373, 940×1345, 939×1359) and render cleanly at
300 dpi, where curve `a` is plainly solid, `b` long-dashed and `c` short-dashed with clear air
between them. A day of owner action was queued against a mistake one directory listing would have
found.

**Lesson, recorded because it will recur**: before filing an acquisition against a source already in
the corpus, check what the *embedded* image is, not what a render of the page happens to be.

---

## 2. ⚠ The γ convention — the actual obstacle, and it is now settled

The caption prints per-layer gammas and **never says how they were measured**:

> «c Gelbschicht γ 1,25, b Purpurschicht γ 1,25, a Blaugrünschicht γ 1,45 (Typ 6.00) und γ 1,35
> (Typ 6.05)»

Without the convention the printed numbers cannot validate a trace, and a trace that disagrees with
them cannot be told from one that is wrong.

| estimator | Bild 5a curve a | vs printed 1.45 |
|---|---|---|
| max slope over a sliding window | **1.895** | **+31 %** |
| least squares, D 0.5–2.0 | **1.476** | **+1.8 %** |

⚠ **The window estimator is biased high and no width fixes it** — the bias falls as the window
widens because the estimator is finding the steepest sub-arc of a curve, not the slope of its
straight section. Least squares over a fixed density band reproduces **all four** printed values:

| panel | curve | traced γ | printed | error | fit rms | columns |
|---|---|---|---|---|---|---|
| 5a | a (Blaugrün, red record) | 1.476 | 1.45 | **+1.8 %** | 0.052 | 93 |
| 5a | c (Gelb, blue record) | 1.248 | 1.25 | **−0.2 %** | 0.021 | 112 |
| 5b | a | 1.376 | 1.35 | **+1.9 %** | 0.036 | 102 |
| 5b | c | 1.206 | 1.25 | **−3.6 %** | 0.045 | 116 |

⚠ **A relative band was tried and is worse.** Defining the straight-line window as 20–80 % of each
curve's own throw pushes curve a to **+9.0 %** on 5a and **+5.4 %** on 5b. The printed figure is
therefore a **fixed density interval**, as sensitometric practice would suggest — the estimator was
identified, not tuned.

⚠ **The guard tolerance is 4 %, not 2 %, and the offender is named.** Bild 5b's curve c only reaches
2.096, so the band's 2.0 ceiling cuts into its shoulder; its fit rms is double curve a's on the same
panel. Widening the tolerance and recording why is honest; narrowing the band until everything
passes would launder the estimator into the answer.

---

## 3. What was adopted, and how each number was obtained

Nothing here is a free six-parameter fit. Each parameter has one source:

| parameter | where it comes from |
|---|---|
| `gamma` | **printed in the caption**, never fitted |
| `dmin` | read off the **right-hand plateau** — 0.12 / 0.10 / 0.09 |
| Dmax | read at the **panel's left edge** |
| span | **(Dmax − dmin) / γ** — anchored on two measured densities, not on the fit's asymptote, which a softplus only approaches |
| `toe_k`, `shoulder_k` | the **only** free parameters |

The midpoint `(toe_x + shoulder_x) / 2` is **held** at each channel's stored value: the engine's
x is `-(log_e + anchor)` and `solve_anchors` absorbs the origin, so only the span is physical.

**GEVACHROME_600** (Typ 6.00), Bild 5a — edge densities **2.729 / 2.351 / 2.229**:

```
r = _rev(0.12, 1.45, toe_x=-0.800, toe_k=0.182, shoulder_x=1.000, shoulder_k=0.038)
g = _rev(0.10, 1.25, toe_x=-0.821, toe_k=0.201, shoulder_x=0.981, shoulder_k=0.122)
b = _rev(0.09, 1.25, toe_x=-0.796, toe_k=0.201, shoulder_x=0.916, shoulder_k=0.122)
```

**GEVACHROME_605** (Typ 6.05), Bild 5b — edge densities **2.505 / 2.266 / 2.096**:

```
r = _rev(0.12, 1.35, toe_x=-0.784, toe_k=0.202, shoulder_x=0.984, shoulder_k=0.075)
g = _rev(0.10, 1.25, toe_x=-0.787, toe_k=0.238, shoulder_x=0.947, shoulder_k=0.096)
b = _rev(0.09, 1.25, toe_x=-0.743, toe_k=0.238, shoulder_x=0.863, shoulder_k=0.096)
```

Stored Dmax reproduces the traced value to **≤ 0.002 D** on all six channels.

### 3.1 ⚠ The transfer these replace was half right and half backwards

Both stocks carried `toe_k` 0.18 / `shoulder_k` 0.30 transferred [T2] from `GEVACHROME_902`.

* **toe_k 0.18 was right** — the trace returns 0.182–0.238.
* **shoulder_k 0.30 was 2.5 to 8× too soft, and the wrong way round.** For a reversal
  `shoulder_x` is the **shadow** end, and Bild 5a's Dmax corner is nearly square. The softplus drives
  `shoulder_k` to **0.038** on the red record trying to reproduce it — which is the model admitting
  it cannot draw a hard corner, not a measurement of one, and it is recorded as such.

A guard now asserts `shoulder_k < toe_k` on all six channels, because a future edit restoring a soft
shoulder would be throwing the measurement away.

### 3.2 ⚠ Bild 5b does not share Bild 5a's abscissa

The two panels are the same size, in the same column, one directly above the other — and their
scales differ. 5a's labels 0/1/2/3 sit at page-x **58 / 151 / 244 / 336.5**; 5b's 1/2/3 sit at
**114 / 207 / 300**, so **5b's frame begins at lg i·t 0.45**. Reading 5b on 5a's grid shifts every
point by 0.40 decades and silently rescales the throw. Each panel is calibrated separately.

⚠ **Consequence**: 5b's "Dmax" values are densities **at lg i·t 0.45** with the curves still
climbing leftwards. The fitted asymptote agrees with them to 0.5 %, which is why they are used, but
a sheet showing the missing 0.45 decades would move them up.

**Square check** (px/decade against px/density): 5a **1.4 %**, 5b **1.1 %**, Bild 6 **1.0 %**.

---

## 4. Bild 6 — the only measured reversal push in the database

Typ 6.05 exposed at **26 DIN (320 ASA)**, one stop over its 23 DIN box speed, reached by extending
the GP 110 first development. Stored as a `ProcessVariant` with its own curves.

⚠ **IT MOVES THE OPPOSITE WAY FROM A NEGATIVE'S PUSH.** On a reversal film the first developer
*consumes* the silver that would otherwise become the positive image, so extra development gives
**less** image, not more:

| | box speed (Bild 5b) | pushed (Bild 6) | change |
|---|---|---|---|
| γ, cyan record | 1.376 | **1.156** | **−16 %** |
| Dmax, cyan | 2.505 | **2.223** | **−0.28 D** |

Both slopes are traced from the **same page with the same estimator**, so the ratio is the least
model-dependent quantity available, and that is what licenses the record.

### 4.1 ⚠ A verify guard failed, and the guard was wrong

`"every push set is contrastier and no less fogged than the step below"` failed on all three
channels. It had been generalised from **two C-41 negatives** (PORTRA 800, ULTRA COLOR 400UC) and
stated as a universal. It is now **split by `is_reversal`**, with the reversal branch asserting the
**opposite** sign on both γ and Dmax, so neither direction can be stored by accident.

### 4.2 ⚠ What was refused

**Bild 6's dmin.** Its right-hand decade lies under the page-curl shadow. A per-column background
normalisation recovers ink out to about lg i·t 2.7, where the merged tail reads **0.096 and is still
falling** — consistent with the parent's 0.12 / 0.10 / 0.09 but not a measurement of it — and the
lg i·t 3 gridline never comes back, so the tail cannot be placed on the abscissa. **dmin is
inherited and labelled inherited.**

**The magenta gamma.** Curves b and c merge below lg i·t 0.4 on every panel, so **only two of the
three slopes are measurable**. Curve b is separated from c by its Dmax at the left edge (1.925
against 1.824) and by nothing else.

### 4.3 ⚠ A development time that conflicts with itself inside one paper

* **p262**: the first development for Bild 6 «betrug dabei 3,5 Minuten».
* **p264, Tab. IV footnote 1**: 2½ min at 25 °C extended «um etwa 45 Sekunden» = **3.25 min**.

Fifteen seconds apart, in one paper, for one measurement. **Method rule 4: both recorded, neither
averaged.** The stored figure is the running text beside the curve set; the footnote is in the
source string.

### 4.4 A new process family

`"Gevachrome"` was added to `_PROCESS_FAMILIES` rather than filing this under **E-6**, which it
predates and contradicts: Tab. IV prints all twelve steps and all six of Agfa-Gevaert's own baths
(GP 110 / 332 / 26 / 308 / 446 / 660) with a re-exposure between them. The set is a validation
whitelist read nowhere else, so widening it costs nothing; mis-filing would have asserted a
chemistry the source denies.

---

## 5. ⚠ A bug in the new reader that produced plausible numbers

`page_images` returns an **0–255** float array. The new ink threshold was written as **0.55**, as
though the array were 0–1, so it selected only pure black. The result:

```
gamma 1.337 vs printed 1.45 (-7.8%)     gamma 1.292 vs printed 1.25 (+3.4%)
```

**Wrong by 8 % and entirely believable.** Nothing in the gammas gave it away. What did: the **fit
rms tripled to 0.10** and the **column count fell from 93 to 71**. Both are now printed by the
audit for every curve, for that reason.

---

## 6. Guards

In `gevachrome_1968_raster.py` (re-derives from the page, run by the build):

1. square check under 3 % on all three panels;
2. each printed γ reproduced within 4 % by least squares over D 0.5–2.0, with rms and column count;
3. **the sliding-window estimator is asserted to be biased high** — the finding is on record, not
   just the fix;
4. the a > b > c edge-density ladder on both panels;
5. stored Dmax against traced Dmax, per channel;
6. the push lowers both γ and Dmax;
7. the push is carried as a `ProcessVariant` with its own curves.

In `verify.py` (asserts the stored database, so an edit to `film_profiles.py` alone trips it):

8. Gevachrome curves reproduce the traced Dmax and the printed γ;
9. **the traced shoulder is sharper than the toe on all six channels**;
10. negative pushes gain contrast, **reversal pushes lose it** — each in its own direction;
11. counts updated: 4 stocks with a curve-changing variant, 3 with published push sets.

---

## 7. ⚠ What would reopen this

**Any print of Bilder 5a/5b showing the decades Bild 5b's frame cuts off**, or any Gevaert statement
of how its γ was measured. The first would move 5b's three Dmax values up; the second would let the
γ tolerance go back to 2 % and would settle whether 5b's curve c is really 3.6 % low or the band is.
Neither is worth an acquisition row — this row is the reason to say that out loud.
