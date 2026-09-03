# RESULT 2026-09-02d — NEOPAN SS: refused in the morning, added in the afternoon

Queue item **N1**. Source: `PDF/PROFILES/FUJI/SS35.pdf` — FUJIFILM DATA SHEET **"NEOPAN SS (135)"**,
Ref. No. **AF3-411E(N)** (EIGI-99.3-HB4-8), four pages, supplied by the owner. Reader:
`fuji_neopan_ss.py`, registered in the build audit. Knowledge base:
`EMULSION_KNOWLEDGE_BASE.md` **§23m**.

⚠ **`EMULSION_KNOWLEDGE_BASE.md` §23k.8 recorded, hours earlier the same day, that FUJI NEOPAN could
not be profiled**: three papers in this corpus measure its grain and nothing measured its tone
scale. This sheet is the tone scale. §23k.8 is **corrected, not deleted** — its inventory is still
exactly right for NEOPAN **S** and **SSS**, and the refusal was correct on the evidence it had.
What changed was the evidence.

---

## 1. What the sheet carries

| § | content |
|---|---|
| 3 | **ISO 100/21°** |
| 4 | «Orthopanchromatic» |
| 7 | development matrix — 3 Fuji and 12 non-Fuji developers × 5 temperatures × EI 100/200/400, plus a deep-tank table |
| 8 | spectral sensitivity curve, spectrogram to daylight 5400 K |
| 9 | characteristic-curve family, Microfine 20 °C small tank |
| 10 | time-Ḡ curves, three developers |

⚠ **And what it does not carry: an image-structure section.** No rms granularity, no resolving
power, no MTF, no reciprocity, no base thickness. All four pages searched.

---

## 2. ⚠ §9 prints the average gradient on every curve, so the trace checks itself

4 min Ḡ 0.28 · 6 min 0.37 · 8 min 0.45 · 10 min 0.53 · 12 min 0.61 — and **nothing in the trace is
told them**, so reproducing them tests the axis calibration, the curve identification and the fit
in one step. The same property NEOPAN 1600's three printed Ḡ values gave.

| curve | printed Ḡ | traced straight slope | reconstructed Ḡ | ratio |
|---|---|---|---|---|
| 12 min | 0.61 | 0.655 | 0.675 | 1.11 |
| **10 min** | **0.53** | **0.577** | **0.525** | **0.99** |
| 8 min | 0.45 | 0.470 | 0.484 | 1.08 |
| 6 min | 0.37 | 0.384 | 0.322 | 0.87 |
| 4 min | 0.28 | 0.288 | 0.189 | 0.67 |

The straight-line slopes run a consistent 3–9 % above the printed averages and are monotone in the
printed order — which is what an average gradient measured over a span *including the toe* must be.

⚠ **The two shallowest curves are refused, not averaged in.** All five converge at the toe and below
Ḡ 0.4 the follower cannot be kept off its neighbours. The **gap** between the two groups
(0.99–1.11 against 0.87 and 0.67), not a tuned number, is what the reader's 12 % gate reads.

**Adopted** — the 10 min member, the drawn curve nearest the sheet's own recommendation of 9½ min
for Microfine at 20 °C and EI 100:

> `ToneCurve(0.2450, 0.5525, −2.4553, 0.0607, 1.9882, 0.1212)`
> fit rms **0.0234 D**, max 0.0982 D, 709 columns; model Ḡ over ΔlogH 2.0 = **0.552** against a
> printed 0.53.

⚠ **Two pins, both stated rather than fitted.** `dmin` 0.245 is the sheet's own printed **Base
Density** rule — the plotted curves never reach the base plateau, so a free fit drives dmin to
whatever bound it is given. And **the shoulder is not measured**: the panel stops at D 1.82 with the
curve still straight, so Dmax is pinned to a class 2.70; refitting at 2.5 or 3.0 moves the rms by
0.0002 D and the model Ḡ by 0.001.

---

## 3. ⚠ Two calibration traps, and why the self-check would have missed the worse one

- **The frame is not the first label.** The abscissa runs logH **−4.0** at the frame to +1.0 at the
  right edge; the leftmost *printed* label is **−3.0**, one gridline pair in. Reading the frame as
  −3.0 shifts every exposure by a full decade **while leaving every density and every slope
  untouched** — so the printed-Ḡ check still passes and nothing complains. Anchoring on "0.0" fixes
  it: it is the only label without a minus sign, and therefore the only one whose glyph centre lands
  on its own gridline.
- **Label centroids and gridlines disagree on the ordinate**, 152.1 px against 158.7 px per 0.5 D,
  because the axis-title glyphs contaminate the label band. The gridlines are right, and the proof
  is that they make one density unit 317.5 px against one exposure decade at 318.4 — the 1:1 aspect
  a sensitometric plot is drawn at.

**The general lesson, now in §23m.2:** *a self-check only tests what it is sensitive to.* A gradient
check is blind to an abscissa offset, because a gradient is a ratio of differences.

---

## 4. §8 — the spectral curve, and the claim it does not make

Traced at 10 nm pitch over 390–640 nm; peak **410 nm**, trough **490 nm**, secondary red lobe
**590 nm**, curve leaving the panel past 650. ⚠ That is exactly the orthopanchromatic character §4
states in words, arrived at independently.

⚠ **Relative, with no zero.** The ordinate carries a single "1.0" bracket and no absolute scale, so
the stored curve is peak-normalised at 410 nm and **no absolute sensitivity is claimed**. 380, 400,
650 and 660 nm are extrapolated, interpolated across the panel's own gridline, and cut-off markers
respectively — labelled as such on the profile.

---

## 5. ⚠ Four measurements of "Neopan SS" that are deliberately NOT joined to this profile

| source | quantity |
|---|---|
| Ooue 1959 Part 2 Fig. 26 | Wiener spectrum, Minidol 20 °C 10 min |
| Ooue 1959 `23_7` Fig. 7 | σ against D at a stated 10 µm aperture, Microfine |
| Takano 1969 Fig. 8 | Selwyn granularity at thirteen apertures → `clump_um` 2.46 µm |

⚠ **They measure the coating sold in 1959–1969. This sheet is dated 1999 by its own printer's code.**
One trade name, two products, forty years apart — the trap already on file for `EASTMAN_5247` (1974
against 1983) and for ILFORD PAN F against PAN F PLUS.

The grain and MTF blocks are therefore **flagged class estimates** from the cubic ISO 100–125 peers
`AGFA_APX_100` (rms 9.0, f50 80) and `KODAK_PLUS_X_125` (9.5, 62), and `verify.py` pins the
separation so a later editor cannot quietly join them.

---

## 6. What changed

| file | change |
|---|---|
| `film_profiles.py` | **`FUJI_NEOPAN_SS`, stock 172**, appended at frozen id 171; `_PROCESSING` and `_PROVENANCE_SOURCES` entries; five `ParamSource` records |
| `film_ids.lock` | `171 FUJI_NEOPAN_SS` appended |
| `fuji_neopan_ss.py` | **new** reader, registered in `build.py` |
| `verify.py` | three N1 guards (measured curve + estimated grain + the era separation; the append contract; the relative spectral curve); the stock count 171 → 172; the monochrome-negative heuristic count 55 → 56 |
| `PDF/PROFILES/FUJI/` | `SS35.pdf` and `Fuji Sales Guide Curves.pdf` staged, chmod 444 |
| docs | `EMULSION_KNOWLEDGE_BASE.md` §23m added and §23k.8 corrected; queue §0 and §4; `NotFound.md` row 1b; `Found.md`; `PROGRESS.md`; `README.md`; `next_week_task.md` |

**Build green:** `verify.py` **509 PASS / 1 FAIL** (the saturation-hierarchy baseline), 30 audits
registered and green, `doc_consistency` 31/31, `g++ -std=c++14 -Wall -Wextra` clean on all 18 TUs
with zero bytes of output.

---

## 7. What is still open for NEOPAN S and SSS

The Fuji **sales guide** supplied with the datasheet (`PDF/PROFILES/FUJI/Fuji Sales Guide
Curves.pdf`, two raster pages) prints **ASA 50 / 100 / 200** for S / SS / SSS in its running text,
plus a per-film γ-versus-development-time panel with a numeric γ axis and a per-film
characteristic-curve family with a numeric log-exposure axis and a marked base-density line, all
Minidol 20 °C tank. ⚠ Its 第3図 three-film overlay is **schematic** — axes labelled only 大 / 小 —
so it gives ordering and shape, not a curve.

That is enough for both stocks at tier 2. The owner is looking for their manufacturer datasheets
first, which would make them tier 1 the way this sheet did for SS.
