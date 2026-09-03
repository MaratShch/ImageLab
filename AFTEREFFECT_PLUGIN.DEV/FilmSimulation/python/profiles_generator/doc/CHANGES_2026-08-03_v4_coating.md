# Schema v4 — coating, gate and lens defects (2026-08-03)

Owner request: model the uneven-emulsion-at-frame-edges effect of old film,
including its "vignetting blink". Four effects approved after a physics
review that **corrected the premise**.

## The correction that shaped the design

The observation (old footage darkens and wobbles at the corners) is real. The
assumed mechanism — emulsion coating unevenness — is mostly **not** the cause,
and getting this wrong would have produced a physically impossible model.

Film is coated as a web up to ~1.4 m wide and slit into strips afterwards.
The coating machine has no idea where frame boundaries will fall, because the
camera gate decides that later. **Coating variation therefore cannot produce a
defect locked to frame corners.** Corner-locked darkening is the LENS.

What coating variation does produce, in web coordinates:

| Axis | On 35 mm | Temporal behaviour |
|---|---|---|
| across the web | frame's horizontal axis | fixed for the whole roll — never flickers |
| along the web | frame's vertical axis | advances one frame pitch per frame → the real blink |

## What was added

| Effect | Where | Driver | Era relevance |
|---|---|---|---|
| Lens vignette (`default_vignette`, stops) | FilmProfile scalar, beside `default_flare` | lens, not film | **every era** — modern glass still loses 0.3–0.5 stop |
| Coating field (`coating_sigma`, two correlation lengths) | `CoatingSpec` | plant QC, **not date** | Soviet/GDR/budget any era; modern majors zero |
| Gate buckling (`buckle_mtf_loss`) | `CoatingSpec` | base stiffness × gate size | pre-1955 worst; 8 mm bad into the 1980s |
| Edge fog (`edge_fog_density`, `edge_fog_mm`) | `CoatingSpec` | **gauge only** | 8/16 mm permanently; 35 mm never |

No single date cut-off — that was the second question asked and the answer is
that the four have different drivers. `_COATING_TIERS` is a plant-and-QC axis
(trough → poor → fair → good → modern), which is why 1974 Eastman 5247 and
present-day Fomapan share a tier while 1990s Kodak sits two tiers better.

## Implementation notes

* **Vignette is real cos⁴(θ) geometry**, not a fitted bowl: one cosine from
  the tilted exit pupil, one from the tilted image plane, two from
  inverse-square distance. Parametrised by corner loss in stops so the profile
  number is directly meaningful; centre is exactly 1.0 by construction.
* **Coating field is a sinusoid sum in absolute web coordinates**, split
  50/50 in variance between a *static* cross-web profile (fixed hopper
  hardware → streaks at fixed x, verified to stay correlated frame to frame)
  and a *drifting* 2D field (coating flow over machine time). Being a pure
  function of (seed, web position) it renders any frame independently, out of
  order, with no state and no tile seams.
* **This replaced, not added to, the pre-v4 path.** The old
  `UNEVEN_EMULSION` code synthesised isotropic mottle with a
  full-resolution FFT pair *per frame*: wrong geometry (blobs, not streaks),
  wrong temporal behaviour (frozen across a sequence — seeded only from
  `settings.seed`), and ~25× the cost. **Net effect of v4 on the coating
  stage is a speed-up.**
* **Corner defocus uses a 5-tap separable kernel blended by radius**, not a
  second FFT. A radially varying blur is not one transfer function, so the
  frequency-domain version needs a second full transform per channel —
  measured at HD that costs about as much as the entire emulsion-MTF stage.
  The effect is mild by nature (0.03–0.30), so the cheap form is inside its
  own uncertainty.
* **Edge fog is applied in the density domain** after development, because
  both of its causes (light leaking past the roll edge, development edge
  effects) land there, and that is where the spec's units live.

## Emergent result worth knowing

The same emulsion behaves differently by gauge, from one model and no extra
parameters: 8 mm advances only 0.45 correlation-lengths of web per frame, so
its mottle **drifts slowly** (lag-1 field correlation +0.96); 35 mm advances
2.24, so it **refreshes each frame** (+0.47). And because a 4.8 mm frame is
smaller than the coating structure, on 8 mm the variation shows up as
frame-to-frame brightness flicker rather than as spatial mottle — while on
35 mm it is the reverse. Both are correct, neither was designed in.

## Verification

Added 17 checks to `verify.py` (81 total): cos⁴ centre/corner exactness,
field unbiasedness and sigma, determinism, cross-web persistence, gauge drift
ordering, corner softening without darkening, full disable path, edge-fog
polarity, and the frame-pitch table.

**Regression found and resolved honestly:** the mid-grey anchor checks began
failing (61% low on AGFACOLOR_NEU_1936). Cause was not the anchor — the test
chart's grey patch sits at r = 0.81 toward the frame corner, so with a
1.15-stop period vignette it correctly receives far less light. Confirmed by
measuring the frame CENTRE with defects active: 0.1738 against a 0.1799
defects-off reference, the residual being the local coating field. The anchor
section now pins `vignette=0, coating_scale=0`, because that contract is
about the tone scale rather than spatial falloff.

## Known limits (stated, not hidden)

* `coating_sigma` delivers ≈0.84× its nominal value through the low-resolution
  synthesis plus bilinear reconstruction. Left uncorrected: the parameter is a
  tier-3 estimate and 16% is inside its own uncertainty, so a compensation
  factor would be false precision.
* The coating field is applied equally to all three layers. Real multilayer
  coating varies per layer; that refinement is not modelled.
* `buckle_mtf_loss` blends a fixed-width kernel rather than scaling a true
  defocus PSF with distance from the focal plane.
