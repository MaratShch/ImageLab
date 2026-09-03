# Spectral curve resolution — measured analysis and verdict, 2026-08-13

Question put: is 10 nm (or 25 nm for some stocks) good enough, or should the
curves be re-formatted to 5 nm or even 1 nm?

Answer: **the question conflates three different things, and they have three
different answers.** All numbers below are measured on the current database, not
estimated.

## The three things

| | What it is | Current | Limited by |
|---|---|---|---|
| **1. Integration grid** | the grid the engine evaluates ∫S·E dλ on | 5 nm → **now 2 nm** | nothing but CPU; free to choose |
| **2. Stored curve sampling** | the samples held in `SpectralSensitivity.log_s_*` | 10 nm (49 stocks), 20–25 nm (4 stocks) | how finely the plot was traced and then **downsampled** |
| **3. Source resolution** | what the printed plot can support | ~pixel level at 600 dpi | line width and axis calibration |

## Measurement 1 — the integration grid decides more than expected

Same stored curves, integrated on progressively finer grids. Deviation of the
derived balance gains from a 1 nm reference:

| grid step | smooth blackbody illuminant | narrow-line illuminant (5 nm mercury lines) |
|---|---|---|
| 1 nm | reference | reference |
| 2 nm | 1.7 × 10⁻⁴ | **0.0 %** |
| 5 nm | 1.1 × 10⁻³ | **1.5 %** |
| 10 nm | 3.2 × 10⁻³ | **52.7 %** |
| 20 nm | 6.2 × 10⁻³ | 23.3 % |
| 25 nm | 2.0 × 10⁻² | **230.8 %** |

Against a blackbody the grid barely matters. Against a line source it decides
the answer: at a 10 nm grid the red/green layer ratio is wrong by **53 %**.
The engine integrates only blackbody SPDs today, so 5 nm was adequate **by
coincidence, not by design**.

**Action taken: the integration grid is now 2 nm** in both implementations
(`_SPECTRAL_LAMBDA_STEP`, `ALGO_SPECTRAL_LAMBDA_STEP`; `ALGO_SPECTRAL_N` 75 →
186). Cost measured 0.129 → 0.179 ms per full derivation, setup domain, ~60
integrals per frame — unmeasurable at frame scale. This invents nothing: it moves
where the trapezoid rule places its nodes and nothing else.

## Measurement 2 — what the stored sampling actually costs

Decimating a natively-10 nm stored curve and re-integrating at 1 nm:

| stored sampling | error in derived balance gains |
|---|---|
| 10 nm (native) | reference |
| 20 nm | 0.4 – 1.1 % |
| 30 nm | 0.4 – 1.1 % |
| 40 nm | 1.4 – 8.6 % |

So 10 nm storage is adequate **for the current consumer**. That is a statement
about the consumer, not about the curve.

The four stocks stored coarser than 10 nm are measurably worse and should be
re-traced if their plots support it:

| stock | step | points | range | source |
|---|---|---|---|---|
| AGFACOLOR_NEG_TYPE_B_1943 | 25 nm | 13 | 400–700 nm | Schmidt/Kochs, Farbfilmtechnik |
| GEVACHROME_902 | 25 nm | 13 | 400–700 nm | Verbrugghe, journal paper |
| GEVACOLOR_NEG_682 | 25 nm | 13 | 400–700 nm | Vervoort/Stappaerts, journal paper |
| FUJICOLOR_A250 | 20 nm | 16 | 400–700 nm | Fuji Data Sheet MP3-57E |

All four are journal papers or older sheets, so their plots may genuinely not
support 10 nm — that must be checked per plot, not assumed.

## Measurement 3 — is there real structure finer than 10 nm?

Maximum gradient in the stored curves is **2.72 decades per 10 nm sample**
(Portra 400, green layer). Checked whether that is real: **mostly it is not.**
Those steps involve the −4.0 sentinel, where the digitiser hit the plot's printed
floor. Above the floor:

| stock | steepest genuine gradient |
|---|---|
| KODAK_PORTRA_400 | 0.98 decades / 10 nm at 670 nm |
| AGFA_OPTIMA_100 | 0.85 decades / 10 nm at 660 nm |
| FUJI_NEOPAN_ACROS_100 | **1.70 decades / 10 nm at 650 nm** |

The ACROS figure is real and is a factor of 50 across one sample interval, at the
red cut-off. There the stored curve is at or past its Nyquist limit and the shape
between the two samples is the interpolator guessing. That is a genuine argument
for finer storage — for that stock, in that region.

## Verdict

**10 nm — adequate today, inadequate in principle.** It survives only because
every illuminant in the engine is a blackbody. Not headroom; a coincidence.

**20–25 nm — not adequate, measurably.** Four stocks. Re-trace where the plot
allows; flag where it does not.

**5 nm — the right storage target**, and it matches the minimum the requirements
document already specifies.

**1 nm — wrong from these sources.** These curves come from printed plots. Even a
600 dpi trace is limited by line width and by the printed axis labels, and a 1 nm
claim asserts a bandwidth no datasheet plot has. 1 nm is correct only for a
monochromator scan measured at 1 nm slit width. The requirements document forbids
storing finer than the measurement's own bandwidth, and that rule is right.

## On "re-format all curves to 5 nm"

There are two operations with the same name and only one of them is legitimate.

**Resampling the stored 10 nm arrays onto a 5 nm grid: NOT done, and should not
be.** It interpolates, adds no information, and destroys the record of which
samples came from the plot — so a later worker cannot tell measurement from
interpolation. It also duplicates work the engine already does at load time on a
finer grid than 5 nm.

**Re-tracing the source plots at 5 nm: legitimate, and worth doing.**
`digitize_plot.py` traces at every pixel column at 600 dpi and downsamples on
request, which means the present 10 nm values are a *downsampled* trace — the
finer information existed in the trace and was discarded at storage. Recovering it
is a re-digitisation, not an interpolation.

It is a campaign, not a command, because `digitize_plot.py` requires per-plot
human input: the page and crop, the axis calibration values read off the printed
labels, and one seed pixel per curve. 53 stocks carry curves and about 100 curves
in total, so this is roughly 100 supervised traces. It should be ordered by
value:

1. `FUJI_NEOPAN_ACROS_100` — the one stock with a measured 1.70 decade/10 nm real
   gradient; the only case where finer storage is demonstrably recovering real
   structure rather than polishing.
2. The four coarse stocks above, if their plots support 10 nm or better.
3. The remaining colour stocks, where the measured benefit is 0.4–1.1 % and the
   honest justification is future-proofing rather than present error.

## Files changed by this analysis

* `PYTHON/profile_generator/film_sim.py` — `_SPECTRAL_LAMBDA_STEP` 5.0 → 2.0,
  with the measurement table in the comment.
* `AlgoSpectralSensitivity.hpp` — `ALGO_SPECTRAL_LAMBDA_STEP` 5.0 → 2.0,
  `ALGO_SPECTRAL_N` 75 → 186, same measurements recorded.
* Verified after the change: Python and C++ still agree to four decimals
  (Portra 400 @3200 K: 1.6838/1.0/0.4711 Python, 1.6838/1.0/0.4711 C++);
  `verify.py` 107 PASS / 2 FAIL, the two pre-existing.
* No stored curve was altered. No profile changed. The database is untouched.
