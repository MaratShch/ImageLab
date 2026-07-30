# Photochemical film simulation

A rewrite of the original grain-overlay script as an actual photochemical model.
Python 3.12, 64-bit, Windows and Linux/WSL2. Dependencies: **numpy and Pillow only** —
no OpenCV, no SciPy. 16-bit PNG writing uses stdlib `zlib`.

```bash
python film_sim.py photo.jpg --list                 # what stocks exist
python film_sim.py photo.jpg -p 5219                # one stock, by catalogue number
python film_sim.py photo.jpg -p "Kodak Vision3 500T (5219)"   # or by full name
python film_sim.py photo.jpg -p all -o renders      # everything
python film_sim.py photo.jpg -p velvia -f ff35      # 35 mm still, not Super 35
python film_sim.py photo.jpg -p ortho                # red-blind 1930s B&W
python film_sim.py photo.jpg -p "super xx" -g 3      # 1938 stock, 3 dupe generations
python film_sim.py photo.jpg -p dufaycolor           # additive mosaic (render big!)
python film_sim.py photo.jpg -p 5219 --flare 0.10    # force period lens flare
python film_sim.py photo.jpg -p technicolor --emit-cpp
python verify.py                                    # 67-check test suite
python cpp_codegen.py -o .                          # regenerate the C++ tables
```

## Why the original script could not get there

The original added spectrally-unshaped noise to gamma-encoded sRGB pixels. Grain is
maybe 20% of what makes an eye say "film", and everything else was missing. The four
structural problems, worst first:

1. **No characteristic curve.** Film's identity is its density-vs-log-exposure curve —
   toe, straight line, shoulder. Without it there is no latitude, no highlight rolloff,
   no shadow compression, and output clips digitally hard at 255. Real negative rolls
   off over 4+ stops above diffuse white.
2. **Wrong domain.** Halation, scatter and grain statistics are all linear-light
   phenomena. Applied to gamma-encoded values they come out the wrong shape.
3. **No resolution awareness.** `grain_size` and `GaussianBlur(radius=15)` were in
   pixels, so the same profile looked like a different film at 1080p and 4K.
4. **No MTF.** The image stayed digitally razor-sharp with grain pasted on top — the
   single loudest tell.

## Pipeline

Order is not cosmetic; several steps give visibly wrong results if moved.

| # | Step | Domain |
|---|------|--------|
| 1 | Decode sRGB → linear light | — |
| 2 | Relative exposure (18% grey = 1.0), exposure offset, taking filters | linear |
| 3 | Stock colour balance, then **veiling flare** from the taking lens | linear |
| 4 | Large-scale coating unevenness | linear |
| 5 | Halation: multi-radius, all channels, energy conserving | linear exposure |
| 6 | Emulsion MTF — light scatter inside the gelatin | linear exposure |
| 7 | Collapse to one record: spectral sensitivity, or the **réseau grid** | linear |
| 8 | Characteristic curve → density | density |
| 9 | DIR coupler inter-image effects | density |
| 10 | Scan: MTF + per-channel misregistration (pre-sampling filter) | density |
| 11 | Grain, variance ∝ √density, spectrally shaped, RMS-calibrated | density |
| 12 | Dye impurity / scanner crosstalk matrix | density |
| 13 | **Duplication generations**, then print | density |
| 14 | Print grain, transmittance → display linear, réseau reconstruction | — |
| 15 | Encode sRGB, dither, quantise to 16 or 8 bit | — |

Reversal stocks skip step 13 entirely: the film *is* the positive, so there is no print.

Details that matter and are easy to get wrong:

- **Red is the softest channel.** In colour negative the blue-sensitive layer is on top,
  green in the middle, red at the bottom. Light reaching the red layer has been
  scattered by two layers of gelatin. That per-channel softness is a strong signature.
- **Grain goes in before the scan MTF, not last.** Adding grain at the end is what makes
  it read as digital noise sitting on a sharp picture.
- **Halation conserves energy.** Light scattered away from a point is removed from it.
  Adding `blur(highlights)` alone injects a flat brightness lift — at CineStill's gain
  it lifted an 18% grey card by 16%.
- **Adjacency is band-pass.** A plain unsharp term settles at `1 + a` for all high
  frequencies, i.e. permanent global sharpening. Real adjacency peaks at the inhibitor
  diffusion scale and returns to unity at both ends.
- **Grain never fully vanishes.** `fog_grain` keeps it alive in deep shadow. Perfectly
  clean blacks are a digital tell.
- **The print anchor is solved, not guessed.** `logE_print = offset − D_neg`, with the
  offset solved per channel so 18% scene grey lands on 18% display — exactly what a lab
  does with printer lights. The naive `offset = D_mid` puts mid grey around 2% display,
  three stops too dark. The solve includes the taking matrix, both dye matrices, the
  coupler flat-field term and the base tint. Originally the dye matrices had row sums
  as far off as 1.27, which threw the mid tone out by more than a stop; they are now
  unit-row-sum by construction (see the `_dye` fix below), so the matrix contribution
  to the anchor is exactly neutral and only the taking matrix and couplers move it.

## The 1930s-40s block, and what it needed

Adding period *emulsions* alone would have underdelivered, because the emulsion is
maybe a third of what makes archival footage read as archival. Two pipeline
additions carry the rest, and both help the modern stocks too:

**Veiling flare (`--flare`).** A *lens* effect, not an emulsion one. Uncoated
pre-1940 glass scattered 6-14% of incoming light into a broad haze across the
frame; anti-reflection coating cut that below 1%. It lifts the black floor and
compresses contrast globally, and nothing in the emulsion model substitutes for
it — without it a 1930s stock still renders with modern blacks. Each stock carries
an era-appropriate `default_flare`; modern stocks are 0. Measured effect on a dark
patch in a bright frame: black level 0.007 → 0.180, overall range 0.69 → 0.52.

**Duplication generations (`-g N`).** Nobody projected the camera negative. A
release print is three or four generations away: negative → interpositive → dupe
negative → print. Each intermediate adds grain and MTF loss. Duplicating stock
runs at gamma 1.0 by design, so contrast does not compound over the chain — only
grain and softness do. Measured over 0/1/2/3 generations, mid grey holds at 0.1801
throughout while the grain-to-detail ratio climbs 0.080 → 0.114. Absolute grain σ
actually falls slightly; what worsens is grain *relative* to picture detail, which
is exactly why dupes look grainier than the negatives they came from.

**Additive colour (Dufaycolor).** This one needed a genuinely new code path, not
parameters. A microscopic grid of colour filters (a réseau) is ruled onto the base
with one panchromatic emulsion behind it; colour resolution is capped by the grid
pitch rather than the emulsion, there is exactly one grain field and no
inter-layer effects of any kind, and the grid stays faintly visible as texture.

Two things about it were worth getting right. The filters must *overlap* — real
ruled gelatin has broad passbands, so a cell under the red filter still records a
lot of green. Model them as pure and the process comes out more saturated than
Kodachrome, which is precisely backwards; the overlap is where the pastel comes
from. And the grid is physical at 20 lines/mm, so it needs at least 3 pixels per
cell to exist at all: below that the mosaic disables itself with a warning rather
than emit aliasing noise. Render Dufaycolor at 2500 px wide or more.

Measured saturation, mid-tone patches, showing the dye hierarchy falls out of the
matrices rather than a saturation control:

| Velvia | Kodachrome | Technicolor | VISION3 500T | EXR | Dufaycolor | Agfacolor Neu | ORWOcolor |
|---|---|---|---|---|---|---|---|
| 0.718 | 0.412 | 0.227 | 0.195 | 0.156 | 0.152 | 0.118 | 0.076 |

Orthochromatic response, which is the loudest single period cue, is pure
parameters — the machinery was already there. Red renders 16× darker than blue on
the ortho stock, against 1.2× on a panchromatic stock of the same era. That is why
silent-era makeup was so extreme: ordinary red lipstick photographed black.

### One model fix this forced

Working on the period dye matrices exposed a flaw in all of them. A hand-written
crosstalk matrix tends to have row sums away from 1 — 1.27 for a "muddy" stock,
0.92 for a "clean" one — which means it shifts neutral *density* as well as
colour. Two unrelated effects on one knob: the anchor solve then has to undo the
density part, and a stock's black level ends up depending on its saturation
setting. All 18 matrices are now generated by a `_dye(k)` helper with row sums
pinned to exactly 1.0, so they change colour and nothing else, leaving `dmin` and
`gamma` solely responsible for level. Verified as a test.

## Everything spatial is physical

Grain clump size, halation radii, MTF cutoffs and channel registration error are in
micrometres or cycles/mm, converted to pixels at render time from
`px_per_mm = image_width_px / format_width_mm`. Change `-f/--format` and the physics
follows the gauge.

Consequence worth understanding: **rendered granularity legitimately depends on scan
resolution.** The scanner MTF is the pre-sampling filter, so a 2K render shows less
grain than a 6K render of the same negative, converging upward. That is not an artefact
— it is why 4K rescans of old negatives look grainier than the 2K masters people
remember. The *stock parameters* are resolution independent; the *rendered result*
correctly is not.

`film_sim.py` warns below 60 px/mm. At 1280 px across Super 35 (51 px/mm) a 4.6 µm
clump is a tenth of a pixel, so fine-grained stocks cannot show their structure. For
judging grain, render at 3000 px wide or more.

## Stocks

All 13 you asked for, plus 13 chosen to stress different parts of the model.

**Colour negative, motion picture**
`KODAK_VISION3_50D_5203`, `KODAK_VISION3_250D_5207`, `KODAK_VISION3_200T_5213`,
`KODAK_VISION3_500T_5219`, `EASTMAN_EXR_500T_5296`, `FUJICOLOR_SUPER_F500_8572`,
`ORWOCOLOR_NC21`, plus `EASTMAN_5247_1974` (the 1970s look) and
`FUJI_ETERNA_VIVID_500T_8547`.

**Colour negative, still**
`KODAK_PORTRA_400`, and `CINESTILL_800T` — VISION3 500T with the remjet stripped, which
makes it the most extreme halation in production and a good stress test of that model.

**Colour reversal** (no print stage)
`KODACHROME_64`, `KODAK_EKTACHROME_100D_5285`, `FUJI_VELVIA_50`.

**Black and white negative**
`ILFORD_HP5_PLUS_400`, `FOMAPAN_400_ACTION`, `SVEMA_FN_64`, plus
`EASTMAN_DOUBLE_X_5222` (Manhattan, Raging Bull) and `ILFORD_DELTA_3200` — tabular
crystals, so enormous grain that is nonetheless *even* rather than clumpy, which
demonstrates that grain size and grain character are independent parameters.

**Black and white reversal**
`KODAK_TRI_X_REVERSAL_200`.

**1930s-1940s** (see the section above)
`EASTMAN_ORTHO_1930` (red-blind), `EASTMAN_SUPER_XX_1938` (film noir),
`SOVIET_PANCHROM_1939`, `AGFACOLOR_NEU_1936` (first integral tripack, muddy dyes
on a reversal stock — a combination nothing else here covers), `DUFAYCOLOR_1937`
(additive réseau mosaic).

**Special**
`TECHNICOLOR_THREE_STRIP` — beam-splitter camera, three separate B&W records, imbibition
dye transfer print. Three things make the look and none is grain: broad overlapping
taking filters (the famous reds), very pure transfer dyes, and 26 µm registration error
between the strips, which is why its edges fringe.

Any stock resolves by name, alias or catalogue number: `5219`, `vision3-500t`,
`Kodak Vision3 500T (5219)` all work.

Print stocks: `SCAN_DI` (digital intermediate, system gamma ≈ 1.0),
`KODAK_2383_RELEASE` (theatrical, contrasty), `TECHNICOLOR_IB` (dye transfer),
`DUPE_FINE_GRAIN` (gamma 1.0, used automatically by `-g`).

### One catalogue-number caveat

You asked for "Kodachrome Tri-X 200 (5266)". Tri-X Reversal ships as **7266** in 16 mm;
I could not establish a 5266 Tri-X reversal product, so the profile is built as the 7266
emulsion and answers to both numbers. Correct it if you have a datasheet that says
otherwise.

## Calibration honesty

**Numeric values tagged `# EST` in `film_profiles.py` are engineering estimates, not
datasheet transcriptions.** They produce a convincing and internally consistent result —
grain, sharpness and latitude all scale correctly across each family — but they are not
authoritative, and the older and more obscure the stock the rougher the estimate.
Kodachrome and Technicolor parameters are reconstructions from published descriptions,
not measurements: treat them as artistic targets.

**The 1930s-40s block is weaker still, and differently so.** For the modern stocks
the numbers are estimates anchored to datasheets I could reason about. For the
period stocks there are no datasheets I can consult at all: the figures are inferred
from how surviving footage looks, from the emulsion technology of the era, and from
internal consistency with the rest of the database. Super-XX is the firmest of the
five because it stayed in production for decades. Agfacolor Neu and the Soviet stock
are the softest. Dufaycolor's réseau pitch is the only figure in that block I would
defend within a factor of two.

On the Soviet stock specifically: it is modelled as a late-1930s Shostka-factory
panchromatic negative. Note that the "Svema" brand name postdates this era, so the
profile is deliberately not called that. Its defining trait here is inconsistency,
which is historically well attested — domestic stock of the period was variable
enough that major productions often preferred imported Agfa or Kodak when available.

To make this a true emulation rather than a good-looking approximation, replace them with
digitised datasheet data:

- Kodak publishes D-logE curves, MTF curves, spectral sensitivity and RMS granularity
  for every current VISION3 stock in its Technical Data sheets.
- Fujifilm published equivalents for ETERNA / SUPER F while the stocks shipped.
- ORWO and Svema data survives mostly in scanned GDR/USSR technical handbooks.

Digitise with WebPlotDigitizer or similar, fit the six `ToneCurve` parameters to the real
curves, and the "can a colourist tell?" answer changes from *probably* to *no*. The
structure is built to accept real data; only the numbers are provisional.

## Files

| File | Purpose |
|------|---------|
| `film_profiles.py` | Physical parameters, 56 stocks, 5 print stocks, 14 gauges |
| `film_sim.py` | The pipeline, 16-bit PNG writer, CLI |
| `cpp_codegen.py` | Emits `film_profiles.hpp` / `.cpp` for a C++ port |
| `film_profiles.hpp/.cpp` | Generated C++ tables, with the reference formulae in the header |
| `verify.py` | 67-check suite: curves, calibration, anchors, isotropy, PNG, flare, generations, réseau, edge cases |
| `make_test_chart.py` | Synthetic chart (ramp, patches, MTF bars, specular discs) |
| `make_period_chart.py` | Larger chart for the period stocks and the réseau |
| `contact_sheet.png` | All 56 stocks on the small chart |
| `period_sheet.png` | The period stocks, plus a 3-generation dupe comparison |
| `dufay_crop.png` | Dufaycolor réseau at 1:1, so the grid is visible |

## Verification

`python verify.py` → 67 checks, all passing. It confirms, among other things:

- every characteristic curve is monotonic
- grain reproduces the datasheet RMS granularity to within 1.3%
- granularity rises monotonically with scan resolution and never exceeds the figure
- 500T renders 2.54× grainier than 50D — the datasheet ratio is 2.54
- 18% grey anchors to 18% display for all 56 stocks and all 5 print stocks
- red is softest and blue sharpest through a 25 c/mm target
- halation is red-dominant and CineStill halates far more than a remjet stock
- reversal stocks clip a wide ramp far sooner than negative stocks
- B&W output is exactly neutral (R = G = B)
- ortho renders red 16× darker than blue, against 1.2× on a panchromatic stock
- flare lifts the black floor and compresses contrast
- each dupe generation worsens grain-to-detail while mid grey holds to 4 decimals
- the réseau leaves a periodic signature 1334× the noise floor at exactly the grid
  frequency, reconstructs neutral grey as neutral, and refuses to run under-sampled
- every dye matrix has unit row sums, and the saturation hierarchy is correctly ordered
- deterministic for a fixed seed; survives pure black and 16 stops of overexposure
- the emitted C++ reproduces the Python characteristic curve to 6 decimals

The C++ tables were compiled with `g++ -std=c++20 -Wall -Wextra` and cross-checked
against the Python implementation.

## Known limits

- **Display-referred input.** A JPEG or PNG has already had its highlights clipped by the
  camera, so the film's shoulder has nothing to roll off. Feed scene-referred data (EXR,
  or a raw file developed to linear) for a real improvement — this is the biggest
  remaining quality lever after datasheet calibration.
- **Gaussian MTF.** `exp(-ln2·(f/f50)²)` is a fair mid-band fit but a real MTF curve is
  not Gaussian. Digitised curves would be better.
- **No temporal behaviour or physical damage.** Single frames only: no gate weave, no
  processing flicker, no frame-to-frame grain animation, no dust or scratches. For the
  period stocks this is the largest remaining gap — flare and dupe generations cover
  the optical and photochemical side of the archival look, but not the mechanical one.
- **Memory.** `numpy.fft` computes in double precision, so a 6K frame needs a few hundred
  MB. Use `--max-dim` to work smaller.
- Requires Python 3.12 as specified; the modules also import on 3.10 (plain `Enum`
  rather than `StrEnum`), which is how the test suite was run.

---

## Expansion set: 26 → 55 stocks

29 stocks added. Database now holds **55 film stocks, 4 print stocks, 12 gauges**
(Super 8 added at 5.79 mm).

| Group | Stocks |
|---|---|
| Agfa B&W | APX 25, APX 100, APX 400 |
| Agfa colour | Optima 100, Vista 200 |
| Eastman reversal | Ektachrome EF 5239 (35 mm), 7239 (16 mm) |
| Ektachrome stills | 64 daylight, 160T tungsten |
| Fuji | F-125 8530, F-125 8630, Neopan Acros 100, Neopan 1600, Provia 400X, Sensia 100 |
| Polaroid | SX-70, 664, 667 |
| USSR | Svema Foto-250, Tasma FN-65 |
| 8 mm gauges | generic B&W reversal, generic colour reversal |
| Indian cinema 1940–60 | Gevacolor 1952, Gevaert Panchro 1950, Eastman Plus-X 5231 |
| Britain | Ilford HP3, Ilford HPS |
| France | Lumière Lumichrome |
| Italy / Latin America | Ferrania P30 |

### Confidence tiers

The original block carries one blanket `# EST`. The new block is graded, because
the sources vary enormously and you should be able to see which is which:

- **[T1] Datasheet-grounded.** Published speed, granularity and resolution exist;
  numbers fitted to them. Good to roughly 10 %.
- **[T2] Partially grounded.** Speed and reputation documented; grain and MTF
  interpolated from siblings in the same family and era.
- **[T3] Reconstruction.** No datasheet available. Built from era, speed class,
  process type and written descriptions. Plausible and internally consistent —
  **not** measurements.

`[T3]` set: Svema Foto-250, Tasma FN-65, both 8 mm entries, Gevacolor 1952,
Gevaert Panchro 1950, Lumière Lumichrome. Lumichrome is the weakest of the lot
and says so in its own description.

### Gauge pairs

`5239`/`7239` and `8530`/`8630` are the same emulsion on different base, so their
numbers are **deliberately identical**. The visible difference is magnification,
which the renderer derives from `--format`, not from the profile. Render the 16 mm
member with `--format 16mm` or `--format super16` or the distinction is lost.
Same for the two 8 mm entries: use `--format 8mm` or `--format super8`.

### Two honest notes

**South America.** No South American country manufactured raw film at scale in
1940–1980; its studios shot on imports, Ferrania prominently among them. So
`FERRANIA_P30` is labelled as the Italian stock it is rather than dressed up as
something it isn't.

**India.** Indian studios also shot imports across the whole 1940–60 window.
Domestic manufacture began 1960 with Hindustan Photo Films at Ootacamund
("Indu" stock), just outside the window. Gevacolor is documented on *Aan* (1952)
and *Mother India* (1957).

### Monotonicity bound corrected

`ToneCurve`'s docstring previously claimed monotonicity is guaranteed for
`shoulder_k <= 2 * toe_k`. That is the analytic bound on the second derivative,
but measured on the actual transfer, ratios above about **1.4** produce a
reversal of order 1e-6 near the shoulder asymptote. Harmless visually, but
`verify.py` checks for it. Four of the new low-Dmax reversal stocks tripped it and
were retuned; the docstring now states the empirical bound.

### Known limitation: low-Dmax stocks don't yet look low-Dmax

Instant film's defining property is a low Dmax — SX-70 reaches 1.87 where
Kodachrome reaches 3.20, so its blacks are open and slightly milky however you
expose it. **That is currently not visible in the render.**
`_normalised_transmittance()` rescales each curve's own `dmin..dmax` to `1..0`,
so the stock's own Dmax is the divisor, every stock is stretched to fill the
output range, and the difference is normalised away. The profiles and the C++
tables carry the correct Dmax; the Python renderer flattens it. On the test chart
SX-70 and Kodachrome both bottom out at display 0.000.

For negatives this is correct — the negative is an intermediate and the print
stock sets the final range. For reversal it is wrong, because the film *is* the
viewed image.

**Proposed fix, not applied:** normalise reversal against a fixed viewing-black
reference (Dmax 3.40) instead of each stock's own Dmax. Predicted sRGB floors:

| Stock | Floor | Stock | Floor |
|---|---|---|---|
| POLAROID_SX70 | 0.159 | KODACHROME_64 | 0.005 |
| POLAROID_667 | 0.151 | EKTACHROME_64 | 0.006 |
| POLAROID_664 | 0.129 | FUJI_VELVIA_50 | 0.006 |
| EIGHT_MM_BW | 0.103 | FUJI_PROVIA_400X | 0.008 |
| AGFACOLOR_NEU_1936 | 0.052 | EIGHT_MM_COLOR | 0.009 |
| DUFAYCOLOR_1937 | 0.035 | EASTMAN_EKTACHROME_* | 0.012 |

Polaroids and 8 mm B&W get their real floor; Kodachrome, Velvia and modern E-6
are essentially untouched. Agfacolor Neu and Dufaycolor also read more correctly
for their era as a side effect.

This changes rendered output for all 17 reversal stocks, so it awaits your
decision.
