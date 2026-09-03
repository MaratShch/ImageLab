# RESULT 2026-08-30d — Mees FIG. 179 digitised as reference data, and adopted into nothing

**Task:** owner approved digitising Mees FIG. 179 — *"Curves showing the relation between Q, γ,
and the diffuse density"*, printed page 643 — on the explicit condition that it **touch nothing in
`callier_q`**. It touches nothing in `callier_q`.

**New:** `mees_callier_q.py`, registered as the 21st audit in `build.py`. **Database untouched.
`film_sim.py` untouched. No C++ changed.**

---

## 1. What it reads, and why it is not allowed to set a number

The four disqualifications from the 2026-08-30 assessment stand unchanged, and are now carried in
the module's own docstring so they travel with the numbers:

| | |
|---|---|
| **No geometry, anywhere in the chapter** | Q is a ratio of specular to diffuse density and the specular half is meaningless without a collection angle. The chapter describes the two methods qualitatively and never states an angle, an aperture or an f-number. So these curves **cannot calibrate `scanner_specular`** — they say how Q *moves*, not what it *is* |
| **One stock, not five** | *"the values of Q found for the densities of sensitometric strips of motion-picture POSITIVE film"* — one emulsion at five development gammas. Not a camera negative, not a survey. Nothing here licenses a per-stock `callier_q` |
| **Private communication** | The figure's footnote is *"O. Sandvik, private communication."* Unpublished, undated, film unnamed |
| **TABLE LXVI is not used** | Six classes of 1909 glass PLATE, no geometry, and the same chapter supersedes its premise two paragraphs later — Callier's constant Q is exactly what this figure disproves |

⚠ **So the value is the SHAPE, and the shape is an argument against the model we ship, not a
replacement for its number.**

---

## 2. What was measured

Calibration on the plate's own ticks: abscissa 14 ticks, **rms 0.0016 D**; ordinate 7 clean ticks,
**rms 0.0009 Q**, with the 1.7 tick recovered at its predicted position and checked rather than
assumed (the γ = 1.65 curve runs across it).

```
  gamma  span D        peak Q @ D       0.30   0.40   0.50   0.75   1.00   1.25   1.50   2.00   2.40
  0.21   0.25-0.51    1.153 @ 0.39    1.151  1.153  1.150      -      -      -      -      -      -
  0.37   0.25-0.77    1.261 @ 0.64    1.259  1.253  1.259  1.257      -      -      -      -      -
  0.69   0.25-1.49    1.475 @ 0.35    1.473  1.471  1.466  1.429  1.407  1.392      -      -      -
  1.20   0.25-2.21    1.670 @ 0.32    1.670  1.668  1.658  1.625  1.584  1.549  1.522  1.502      -
  1.65   0.25-2.42    1.723 @ 0.51    1.711  1.720  1.722  1.701  1.670  1.641  1.615  1.583  1.573

  toe (all five curves drawn as ONE stroke below D = 0.25):
    960 rows, lowest Q 1.042 at D 0.055
      Q 1.05 reached at D 0.056     Q 1.40 reached at D 0.103
      Q 1.10 reached at D 0.076     Q 1.60 reached at D 0.152
      Q 1.20 reached at D 0.073
```

Four properties, and every one of them is something `AlgoCallierFactor` cannot express:

1. **Q collapses to unity at the toe, and does it inside a tenth of a density.** 1.04 at D 0.055,
   still only 1.40 by D 0.10. The renderer holds Q constant, so on all 68 monochrome stocks it
   applies a condenser's full scatter gain to densities that carry almost none of it.
2. Every curve rises to a maximum and then decays — D 0.32 at γ 1.20, D 0.51 at γ 1.65. The two
   low-gamma curves plateau instead and simply stop.
3. The maximum scales with development gamma: **1.153, 1.261, 1.475, 1.670, 1.723**.
4. The decay above it is shallow: **8 %** from peak to D 2.0 at γ 1.65, **10 %** at γ 1.20.

⚠ **Property 1 corroborates BBC T-101 Fig. 25 from an independent source** (Q falling ~15 % from
D 0.1 to 1.0 on Tri-X 5223) and adds the gamma axis T-101 only hinted at. Two sources agreeing on a
shape is this project's usual threshold — but the threshold for adopting a *shape*, not for
inventing the *number* that scales it, which is still missing.

---

## 3. ⚠ Seven ways this plate lies to a tracer, and what each one cost

This is the part worth keeping. Every item below produced a **plausible wrong answer**, not an
error, which is the only failure mode that matters in a tracer.

**1. The frame is not a rectangle.** The right rule walks from x = 1532 at the top to 1521 at the
bottom; the top rule drifts 4 px, the bottom 3 px the other way. Part scan skew, part a rule drawn
by hand in 1942. A first-over-threshold edge detector picks whichever end of the smear clears the
threshold — it put the left edge at 347 where the line's core is 350. Each edge is now taken at its
**darkest** row or column.

**2. The right rule scores as a fifteenth abscissa tick, and lands exactly on the tick ladder.**
It leans far enough left by the bottom of the plate to sit inside the tick scan band, at the ladder
position for **D = 3.0** — one clean step past the last printed label, 2.8. Ladder membership
therefore cannot reject it. It is rejected on what actually distinguishes a tick from a rule: a
tick is a stub near the axis, a rule runs the height of the plate.

**3. A curve manufactured a spurious ordinate tick that kept the COUNT right.** The γ = 1.20 curve
grazes the tick scan band and produced a tenth "tick" at y = 1592, **39 px from the real one at
1552 on an axis whose spacing is 150** — while the same curve was obscuring a real tick elsewhere.
A count check would have passed. Ladder membership plus a nearest-slot contest catches it.

**4. Every caption is set INSIDE the frame, at the height of the line it labels.** Each curve stops
where its development ran out of density, and its `γ = x` caption begins 24 to 50 px later at the
same height. A generous miss limit is not robustness here — it is a guarantee that the tracker
coasts the gap and reports the caption's height as Q. The limit is 12 px, under the smallest gap on
the plate, and each curve's printed end is now **asserted** (0.51 / 0.77 / 1.49 / 2.21 / 2.42).

**5. Every curve is drawn THROUGH dense scatter, and rejecting fat ink rejects the curve.** At
D 0.46 the γ = 0.37 curve is nine separate runs in one column because six markers sit on the line.
With fat runs refused outright, four of five curves died within 0.2 D of the seed — γ = 0.21
produced **five pixels of trace** and a peak the audit then compared against a tolerance quite
happily. Fat ink straddling the prediction is now read as *the curve wearing a marker*: keep the
prediction, do not trust the blob's centre, and hold the prediction inside the ink while the slope
decays. Free-running the slope instead walked γ = 1.65 out through the far side of a blob and it
died at D 1.34 with 1.1 D of clean line ahead of it.

**6. Seeding on "a column with exactly five separated runs" seeded on a caption.** It chose x = 590,
where the γ = 0.21 **curve has already ended** and the fifth run is the first stroke of its label.
Two fixes: merge runs before counting, so one curve's scatter is one run and not nine; and require
all five to be **thin**, because a column can hold five separated runs where one of them is a 28 px
marker cluster whose centroid sits 7 px below the line inside it.

**7. The toe is drawn as one stroke, and it lives exactly where the tick stubs live.** Below
D ≈ 0.25 the engraver drew the five curves **on top of one another** — there is no per-gamma
information to recover there and none is reported. That single near-vertical stroke is traced by
**rows** instead, which is the only way to follow something near-vertical. But the bottom-left
corner is also where the ordinate stubs come in from the left and the abscissa stubs go up from the
bottom, and a naive row scan reported, with a straight face, *"Q = 1.10 reached at D = 0.059"* —
the ordinate's own 1.1 tick — and a minimum of *Q = 1.008 at D = 0.203*, the abscissa's 0.2 tick.
⚠ **And they could not both be cleared the same way.** A row cut removes the abscissa stubs for
free. A column cut wide enough to clear the ordinate stubs **eats the toe** — that was the reading
"lowest Q 1.083 at D 0.080": the collapse cropped off, and the crop reported as the measurement.
The stubs are welded to the rule and the bundle is not, so they are told apart by where the run
*starts*.

⚠ **And the peak was biased upwards by construction, invisibly.** Every reported peak is a maximum
over the trace, and the trace's error is not symmetric noise — it is the tracker riding whichever
marker sits on the line, up to one marker radius. A maximum over that reads the highest *marker*:
γ = 1.20 came out at 1.671 against a line the plate draws at 1.663. A 21-column running median —
wider than a marker, far narrower than these broad peaks — removes the bias without moving the peak.

---

## 4. What is asserted on a re-run

`python mees_callier_q.py --assert` fails on any of:

- five curves not traced, or any two of them **crossing** (the whole tracking scheme rests on their
  never crossing, so it is asserted rather than assumed)
- a peak Q more than 0.02 from 1.153 / 1.261 / 1.475 / 1.670 / 1.723
- a peak outside D 0.25–0.75
- **a curve traced past, or short of, the density the plate draws it to** — short means the line was
  lost, long means the caption was traced
- axis calibration drifting past rms 0.004 Q / 0.006 D
- the toe envelope failing to bottom below Q 1.08, or failing to be back to Q 1.5 by D 0.15

Absent the page image the audit **SKIPs** and returns 0, like every other document audit here.

---

## 5. What this changes

**Nothing that renders.** No profile, no `callier_q`, no C++.

What it changes is the standing of one argument. Queue **C22** and **C41** both close with the same
open item — *the film half of the Callier product is a class estimate, 1.3 negative and 1.25
reversal, from a generator rule with no document behind it* — and both note that the control ships
at zero because of it. This figure does not close that item and was never going to: **it has no
geometry, so it cannot state a Q for any reader.** What it does is put a measured shape behind the
separate claim that a *constant* Q is wrong at the toe, and put it somewhere `build.py` re-runs.

⚠ **The reason it is registered as an audit at all** is that it is the only entry in that list whose
source is deliberately **not adopted**. Every other audit re-derives a number the database stores.
This one re-derives a shape the database cannot express — and an argument resting on a trace nobody
re-runs decays into a remembered impression. Which is the same failure this project keeps finding:
*a readiness label decays, and the only way to know is to open the document.*

---

## 6. Files

**New:** `mees_callier_q.py` (tracer + audit), `PDF/PROFILES/RETRO/mees_fig179_p643.png`
(600 ppi page image, `pdfimages -f 642 -l 642 -png` straight out of the book — no re-rendering,
because the embedded image *is* the source resolution).

**Changed:** `build.py` (audit registered, 20 → 21).

**Not changed:** the database, `film_sim.py`, and every C++ file.
