# Changes 2026-08-11 (second pass) — GOST extraction into Soviet profiles

Follow-up to `CHANGES_2026-08-11_stocks100.md`: numeric extraction from the
owner-supplied `PDF/PROFILES/SOVIET STANDARDS/` set into the existing Soviet
profiles. Profile count unchanged (100).

## Two field values changed — both raised to a state-norm floor

GOST 24876-81 Table 6 sets minimum MTF at 30 mm⁻¹ per film and quality
grade. Inverting the Gaussian MTF model (`T(f) = exp(−ln2·(f/f50)²)`)
turns each floor into a minimum f50, and two profiles carried estimates
**below the norm their own state standard required**:

| Profile | Norm | Old f50 → T(30) | New f50 → T(30) |
|---|---|---|---|
| `SVEMA_FN_64` (= Foto-65) | T(30) ≥ 0.60 (both grades) | 34 → 0.583 ✗ | **35.0** → 0.601 ✓ |
| `SVEMA_FOTO_250` | T(30) ≥ 0.50 (top grade) | 26 → 0.397 ✗ | **30.0** → 0.500 ✓ |

Both new values are the norm floor **exactly** — the most conservative
compliant value, not a typical-product claim. Tier stays [T2]. The two
compliant siblings were left untouched and their compliance recorded:
`SVEMA_FOTO_32` (f50 42 → 0.70 ≥ 0.60), `SVEMA_FOTO_130` (f50 31 → 0.52 ≥ 0.50).

## Corroborations recorded (no value changes)

* **`SVEMA_TSNL_65` ← ГОСТ 25120-82 Table 6.** Fog-plus-mask density
  behind blue/green/red filters ≤ 0.90/0.50/0.27 (top grade),
  1.10/0.60/0.30 (first grade) — the profile's dmin ladder 0.92/0.50/0.30
  sits inside the first-grade limits. Latitude ≥ 1.50 matches Gurlev's
  1.5. Speed range 45–90 GOST units brackets EI 65. R ≥ 63 lin/mm matches
  `_RESOLVING_POWER`.
* **`SVEMA_FOTO_32` / `SVEMA_FOTO_130` ← ГОСТ 24876-81.** Fog, gradient
  0.8–1.1, latitude ≥ 1.5, MTF compliance, R norms — full citation added
  to `_PROVENANCE_SOURCES`.

## Conflict recorded, NOT adopted

ГОСТ 25120-82 prints recommended per-layer contrast for ЦНЛ-65:
**γ bottom/middle/top = 0.55/0.60/0.65 (±0.05)**. Gurlev 1986 — the source
the profile curves were fitted to — prints γ 0.7 ± 0.1 with the top layer
+0.1–0.2. The two conflict; the GOST densitometry convention for masked
negatives is not stated in the excerpt on file, and the curves were fitted
jointly with Gurlev's Dmax and latitude, so swapping in the GOST gammas
alone would mix conventions. Kept Gurlev; recorded GOST in full in the
provenance entry. Both sources agree on the **ladder direction**
(top > middle > bottom), which the profile reproduces.

## Scope corrections found during extraction

* **ГОСТ 25120-82 covers ЦНД-32 (daylight) and ЦНЛ-65 (tungsten).**
  The database's `SVEMA_TSNL_32` is the *tungsten* 32 (TU 6-17-441-78,
  Gurlev) — a different mark from the GOST's daylight ЦНД-32. The GOST
  therefore contributes nothing to `SVEMA_TSNL_32`, and **ЦНД-32 is an
  available-but-unprofiled stock** (full TU norms on file) if ever wanted.
* **`SVEMA_DS_4` gets nothing** — ДС-4 is not a 25120-82 mark.
* **Gate weave claim walked back.** The gauge-dimension GOSTs
  (4896-80 / 20904-82 / 8761-75) publish slitting tolerances, which are
  static; per-frame weave is transport clearance, standardised in
  equipment GOSTs not on file. `_TEMPORAL_OVERRIDES` weave values stay
  [T3]. The perforation-pitch spec (100 steps = 475 ± 0.4 mm ⇒ ±4 µm
  cumulative) is *consistent* with the ~10 µm class estimates but does
  not measure them.

## Still not imported, and why

* **GOST granularity limits (σ_D×1000 ≤ 40/45/50/55)** — GOST aperture
  unstated in the documents on file; not convertible to the 48 µm
  diffuse-RMS convention. Becomes convertible only if the granularity
  method GOST (with the aperture) is added to the archive.
* **GOST fog D₀ limits** as dmin — D₀ is fog *above base*; profile dmin
  is base+fog as the renderer consumes it. Recorded in citations only.

## Files touched

* `film_profiles.py` — two MTFSpec floors raised (comments carry the
  arithmetic), `_PROVENANCE_SOURCES` extended for SVEMA_FOTO_32,
  SVEMA_FOTO_130, SVEMA_TSNL_65.
* Regenerated: `film_profiles.hpp/cpp`, `film_enum.hpp`, `film_names.txt`
  (root + generator copy), `doc/FilmActiveProfiles.md`, `doc/FilmCurves.md`.
* `doc/README.md` — status banner extended.

`validate_all()` clean; `verify.py` 104 PASS / 2 FAIL (both pre-existing,
unrelated); C++ compiles; SVEMA_FN_64 and SVEMA_FOTO_250 render through
the full scalar engine. Enum indexes unchanged (no insertions).
