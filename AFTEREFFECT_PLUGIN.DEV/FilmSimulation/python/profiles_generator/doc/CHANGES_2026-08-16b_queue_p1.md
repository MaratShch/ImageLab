# Queue P1 execution — 2026-08-16

Owner said "Start queue!" — Priority 1 of `DIGITIZATION_QUEUE.md`.

## Status discovery

5203 / 5213 / 5219 / 5222 H&D curves were **already [T1]** (machine-traced 2026-08-02,
batch 5); the queue's Priority-1 list was stale. Fresh independent re-traces
(538–563 points per layer, fit RMS 0.003–0.008 D) reproduced the stored parameters —
5219's red layer to three decimals on all six. Recorded in the profiles as mutual
validation; nothing changed.

## Newly adopted [T1]

| Stock | Source | Method | Fit RMS (D) |
|---|---|---|---|
| `KODAK_EKTACHROME_100D_5285` | H-1-5285 p3 characteristic curves | PDF **vector** paths (3 × 13 beziers) — coordinates exact, only axis calibration fitted (≤0.018 D / 0.05 stop) | 0.024 / 0.028 / 0.028 |
| `KODAK_TRI_X_REVERSAL_200` (7266) | 7266 TI sheet p3, 300 dpi raster | 296-point machine trace; mid-grey anchored at the previous density 1.433 (ACROS precedent), anchor log H −1.581 | 0.0167 |
| `KODAK_2383_RELEASE` (print stock) | 2383 sheet (2015 ECP-2D) p5 | PDF **vector** paths (65–71 beziers/layer), ~800 exact samples/layer; x = 0 anchored per layer at the sheet's own LAD aims 1.09 / 1.06 / 1.03; gamma capped at 6.0 | 0.018 / 0.010 / 0.031 |

5285's `dmin` moves to 0.115–0.182, a physically sensible reversal base+fog; the old
hand-fit carried 0.14–0.15 for all three layers. 2383's traced Dmax is 4.09–4.10 on every
layer, and the three layers' LAD points sit at absolute log H 1.097 / 0.754 / 0.445 —
printer lights realign them, so relative-x storage remains correct.

## The monotonicity lesson (binds all future fits)

The unconstrained best fits — RMS as low as **0.0035 D** — put `shoulder_k < toe_k`, which
makes the difference-of-softplus model **non-monotone past the shoulder**: up to
−0.18 D per log H of real density reversal on 5285's blue layer. Nothing about the fitted
curves looked wrong; `verify.py`'s monotonicity check caught it.

All three were refitted with `toe_k <= shoulder_k <= 2*toe_k` enforced **inside** the
search. At `shoulder_k == toe_k` the model is analytically monotone (the sigmoid-argument
gap `(shoulder_x - toe_x)/k` is a positive constant). Cost: 0.01–0.02 D of residual —
paid deliberately, because a curve that reverses is a renderer defect while residual is a
property of the tracing.

`verify.py`'s monotonicity test now divides each curve's minimum slope by its own gamma
before comparing against the float32 tolerance: on a flat Dmax shelf the bracket is
constant and its rounding error is multiplied by gamma, so a gamma-11 curve shows ~11×
the ulp noise of a gamma-1 curve. Scaling the allowance keeps the check strict for
ordinary curves instead of weakening it globally.

## Verification

- `verify.py`: **124 PASS / 2 FAIL** — the same two pre-existing failures (saturation
  hierarchy ordering, neighbour-pair coupling). Three new permanent guards assert the
  adopted parameters and 2383's monotonicity bound.
- `validate_all()` green; 143 stocks, 9 print stocks.
- C++ regenerated and compiles clean at `-std=c++14`; both copies synced.
- `FilmActiveProfiles.md`, `FilmCurves.md` regenerated; master doc + Russian mirror
  updated; `DIGITIZATION_QUEUE.md` gained batch 9 with the stale-list correction.

## Deferred from this batch

Vision3 granularity σ-D-versus-density curves (four TI sheets — plots identified and
extracted to page images, tracing queued); Portra / Ektar / T-MAX still-film sheets
(vector check pending); 5294; KODAK DATA BOOK volume 5 (the owner returned the file to
local disk on 2026-08-16).
