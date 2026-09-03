# ISO 5-3 densitometry — COMPLETE (2026-08-03)

All nine spectral-product tables are now in `iso5_3_density.py`. The blocker on
`dye_matrix` derivation is cleared.

## Sources

| Tables | Document |
|---|---|
| Table 2 — visual, Type 1, Type 2 | **ISO 5-3:1995(E)** *Photography — Density measurements — Part 3: Spectral conditions* (standards.iteh.ai preview) |
| **Table 3 — Status A** | **ANSI/ISO 5-3-1995, ANSI/NAPM IT2.18-1996** — `aimm.it2.18.1996.pdf`, p.16 |
| **Table 4 — Status M** | same document, p.17 |

The ISO copy is a preview that stops mid-sentence immediately after *naming*
Table 4. The **US national adoption (IT2.18-1996) carries both tables in
full** — that document solved it. `ISO3664 Standard.pdf` is viewing conditions
and contains no densitometry tables; `bs-iso-5-3-1995-file.pdf` is a 9-page
extract, also without them.

## Contents, all self-verified

| Metric | Peak | Nonzero range | Active points |
|---|---|---|---|
| visual | 570 nm | 410–760 nm | 36 |
| Type 1 (diazo/vesicular print) | 400 nm | 370–430 nm | 7 |
| Type 2 (silver-halide print) | 430 nm | 350–530 nm | 19 |
| **Status M** blue / green / red | 450 / 540 / 640 nm | 400–520 / 450–620 / 610–770 | 13 / 18 / 17 |
| **Status A** blue / green / red | 440 / 530 / 620 nm | 410–500 / 490–600 / 590–770 | 10 / 12 / 19 |

All transcribed from **rendered page images, not PDF text layers** — the text
layers interleave the wavelength and value columns and would silently mis-pair
rows. European decimal comma handled: the documents' "4,957" is 4.957.

## One subtlety that would have been a silent bug

Table 2 marks out-of-range entries `< 1,000`, i.e. genuinely floor. **Tables 3
and 4 do not** — they print a **slope and an arrow**, meaning the response
continues linearly in log10 beyond the last tabulated value:

| | below range | above range |
|---|---|---|
| Status M blue / green / red | +0.250 / +0.106 / +0.260 per nm | −0.220 / −0.120 / −0.040 per nm |
| Status A blue / green / red | +0.380 / +0.220 / +0.270 per nm | −0.140 / −0.170 / −0.040 per nm |

Truncating those to zero would have narrowed every channel's skirt and biased
all derived densities. `weights()` applies the printed slopes and clamps at
1e-6 relative.

## Self-checks (`python3 iso5_3_density.py`)

* all nine tables have exactly 44 entries matching the wavelength grid;
* every table peaks at exactly 5.000, at the wavelength the document prints;
* a spectrally non-selective sample reads density **0.000000** in every metric;
* a uniform 10 % transmitter reads exactly **1.000000** in every metric,
  including all three channels of both Status sets — which is the real test
  that the slope extrapolation is balanced;
* **Status M red (640 nm) peaks longer than Status A red (620 nm)** — asserted
  in code, because it is the documented reason the two exist: M was "defined to
  match closely the responses historically used in evaluating colour negative
  films", A to match transparency films.

## What this unblocks

`dye_matrix` can now be **derived** rather than estimated: integrate each
stock's digitised spectral dye density curves against Status M (37 colour
negative stocks) or Status A (16 reversal stocks) and compare the result with
the existing hand-set matrices. That comparison quantifies how wrong the
estimates were.

Remaining prerequisite: the dye curves themselves. Tracing is close but not
finished — see `DYE_DIGITISATION_STATUS.md` for the exact open bug (gridline
erasure destroying data at 500/600 nm and along D=0).
