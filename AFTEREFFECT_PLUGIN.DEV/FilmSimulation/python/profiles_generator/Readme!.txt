===============================================================================
 PHOTOCHEMICAL FILM SIMULATION  --  START HERE
===============================================================================

WHAT THIS IS
------------
A rewrite of your film-grain script as an actual photochemical model of film.
Instead of adding noise to pixels, it reproduces the chain a photon really
travels: the taking lens, exposure, halation, emulsion scatter, the
characteristic curve, development couplers, silver-halide grain, scanning,
duplication and printing.

It supports 142 film stocks, covering colour negative, colour reversal, black and
white, three-strip Technicolor, 1930s-40s period stocks, Soviet-era Svema and
Tasma emulsions documented from printed USSR reference books, and additive
colour via a physical filter grid (Dufaycolor).

2026-08-13, spectral sensitivity: the per-layer spectral sensitivity curves in
the profile database are now actually used. Colour-temperature balance is
computed by integrating each layer's own measured curve against the two
blackbody spectra instead of sampling three assumed peak wavelengths. 53 of 121
stocks carry curves; the other 68 render exactly as before, bit for bit.

Curve resolution, measured 2026-08-13: the engine now integrates on a 2 nm grid
(was 5 nm). This is the grid the integral is evaluated on, NOT the sampling of
any stored curve. Against a blackbody illuminant the choice barely matters;
against a narrow-line source a 5 nm grid is 1.5 % wrong and a 10 nm grid 52.7 %
wrong on the red/green layer ratio, so 5 nm had been adequate only by
coincidence. Stored curves are unchanged at their measured sampling (10 nm for
49 of 53 stocks, 20-25 nm for four); resampling them finer would interpolate and
destroy the record of which samples came from the plot. A supervised re-trace
campaign from the source plots is queued in DIGITIZATION_QUEUE.md, led by
FUJI NEOPAN ACROS 100 -- the one stock with a measured real gradient finer than
its stored sampling.

The monochrome RGB collapse and the taking matrix can also be derived from the
same curves, and both are implemented, but both are OFF by default: projecting
a sensitisation curve onto three visible primaries derives nonsense for
extended-red and infrared stocks (KONICA_INFRARED_750 comes out blue-dominant
against a correct red-dominant authored triple). An analysis on 2026-08-03 had
already found and quarantined this; rebuilding it reproduced the failure. The
real fix is a scene spectral model, not a reprojection of existing data. New Python functions live in film_sim.py (the spectral_*
block); the C++ side is AlgoSpectralSensitivity.hpp/.cpp, consumed by stages 03
and 07 in both the scalar and AVX2 builds.

2026-08-13 renames (owner request): SVEMA_FN_64 -> SVEMA_FOTO_65 (same film
per the USSR standard; the _16MM/_8MM gauge entries are retired -- gauge is
selected by the format control, their transport numbers preserved in
film_profiles.py), SVEMA_TSNL_* -> SVEMA_CNL_*, EIGHT_MM_* -> GENERIC_*.
Old names still resolve through aliases. The stock list is now in natural
numeric order (FOTO-32 < FOTO-65 < FOTO-130 < FOTO-250), which together with
the renames RENUMBERS eFILM_PROFILE -- saved projects that store the enum
value must be re-saved. Historical sections below keep the names that were
current when they were written. It also models the taking lens's
veiling flare and multi-generation dupe printing, which is what makes the period
stocks actually look period rather than merely soft.

Python 3.12, 64-bit. Works on Windows and on Linux / WSL2, unchanged.
Dependencies: numpy and Pillow. Nothing else. No OpenCV, no SciPy.
16-bit PNG output is written with the standard library, so it adds no dependency.


-------------------------------------------------------------------------------
 1. QUICK START  (about two minutes)
-------------------------------------------------------------------------------

Install the two dependencies:

    Windows:        py -3.12 -m pip install numpy pillow
    Linux / WSL2:   python3.12 -m pip install numpy pillow

Below, "python" means "py -3.12" on Windows and "python3.12" on Linux.
Run everything from the folder containing these files.

  a) Check it works. Prints a table of all 142 stocks:

        python film_profiles.py

  b) Run the test suite. Should end with "ALL CHECKS PASSED" (70 checks):

        python verify.py

  c) Render your own photo through one stock:

        python film_sim.py myphoto.jpg -p 5219

     Output lands in the film_renders/ folder as a 16-bit PNG.

  d) Render it through every stock at once:

        python film_sim.py myphoto.jpg -p all

If step (b) fails, stop and read section 8 before doing anything else.


-------------------------------------------------------------------------------
 2. WHAT EACH FILE IS FOR
-------------------------------------------------------------------------------

You edit these:

  film_profiles.py     THE DATA. Every film stock and its physical parameters:
                       characteristic curves, grain, sharpness, halation, dye
                       matrices, lens flare, filter grids. This is where you
                       tune the look of a stock, and where you would paste in
                       real datasheet numbers. Heavily commented -- read this
                       one first if you want to understand the model.

  film_sim.py          THE ENGINE. The 15-step pipeline, the 16-bit PNG writer,
                       and the command line interface. You only edit this to
                       change *how* the physics works, not *which* film.

You run these:

  verify.py            70-check test suite. Run it after ANY edit to either file
                       above. It catches the mistakes that are invisible by eye:
                       wrong grain amplitude, mid-grey drift, non-monotonic
                       curves, channel casts, broken filter grids.

  make_test_chart.py   Regenerates test_chart.png, the synthetic test image
                       (grey ramp, colour patches, resolution bars, blown
                       highlights). verify.py needs test_chart.png to exist.

  make_period_chart.py Regenerates period_chart.png, a much larger 3200 px
                       chart. Use this one for the period stocks, and you MUST
                       use it (or something equally large) for Dufaycolor --
                       see section 4B.

  cpp_codegen.py       Exports the stock database to C++ for your Qt/C++ side.

  gen_film_names.py    Writes film_names.txt: the official film names, one
                       quoted name per line, in EXACTLY the order of the C++
                       std::vector. Run it only AFTER cpp_codegen.py has
                       produced and verified the C++ source.

Generated output, safe to delete and regenerate:

  film_profiles.hpp    C++ header. Contains the struct definitions AND the
                       reference formulae as comments, so a C++ port cannot
                       silently drift from the Python original.
  film_profiles.cpp    C++ tables: all 142 stocks, 9 print stocks, 14 gauges.
                       Regenerate both with:  python cpp_codegen.py -o .
  film_names.txt       Official film names, quoted, one per line, ordered as
                       the C++ vector. Regenerate: python gen_film_names.py

  test_chart.png       The small synthetic test image.
  period_chart.png     The large 3200 px test image.
  contact_sheet.png    All stocks on the small chart, side by side. Open this
                       first to see what the stocks look like.
  period_sheet.png     The five period stocks, plus a 3-generation dupe
                       comparison and a modern stock for reference.
  dufay_crop.png       Dufaycolor at 1:1 magnification, so you can actually see
                       the filter grid and the pastel colour it produces.
  sheet.py             Rebuilds contact_sheet.png. Not part of the test suite.

Documentation:

  Readme!.txt          This file. Practical operation.
  doc/README.md        The technical write-up: what was wrong with the original
                       script, the full pipeline table, the physics reasoning,
                       every bug the test suite caught, and the honest limits.
                       Read it when you want the "why".
  doc/                 The audit trail: datasheet verification reports
                       (Found.md / NotFound.md), the Soviet reference-book
                       extraction (SOVIET_EXTRACTION_2026-08-02.md), dated
                       changelogs, and the measurement adoption reports.

NOT INCLUDED IN THIS ZIP: the folders of finished renders. Those are 16-bit PNGs
of grain, which do not compress -- over 100 MB. Regenerate in about 30 seconds:

    python film_sim.py test_chart.png -p all


-------------------------------------------------------------------------------
 3. EVERYDAY COMMANDS
-------------------------------------------------------------------------------

List every stock with its aliases:

    python film_sim.py --list

Pick a stock. Name, alias and catalogue number are interchangeable, and
punctuation, spaces and case are ignored -- all four of these are the same film:

    python film_sim.py photo.jpg -p 5219
    python film_sim.py photo.jpg -p vision3-500t
    python film_sim.py photo.jpg -p KODAK_VISION3_500T_5219
    python film_sim.py photo.jpg -p "Kodak Vision3 500T (5219)"

Choose the film gauge. This genuinely matters -- see section 4C:

    python film_sim.py photo.jpg -p velvia -f ff35        # 35 mm still
    python film_sim.py photo.jpg -p 5219  -f super35      # motion (default)
    python film_sim.py photo.jpg -p hp5   -f medium645    # 6x4.5 cm

    Available: super35 academy35 anamorphic35 techni35 super16 16mm 8mm
               ff35 medium645 large4x5 imax15

Choose the print. Negative stocks are printed; reversal stocks ignore this:

    --print-stock SCAN_DI             digital intermediate, natural (default)
    --print-stock KODAK_2383_RELEASE  theatrical print, contrasty, crushed blacks
    --print-stock TECHNICOLOR_IB      dye transfer, saturated and clean
    --print-stock DUPE_FINE_GRAIN     gamma 1.0 duplicating stock (used by -g)

Exposure, in stops. Negative film takes overexposure gracefully; reversal
punishes it, exactly as in reality:

    python film_sim.py photo.jpg -p 5219 -e +1.5
    python film_sim.py photo.jpg -p velvia -e -0.5

Dial individual effects from 0 (off) to 2 (double). Good for A/B comparison:

    --grain 0        no grain
    --halation 0     no highlight bloom
    --couplers 0     no inter-layer development effects
    --grain 0.5 --halation 1.5

Show the tungsten/daylight mismatch. Default is 0, which assumes the correct
on-camera filter was used, as a professional shoot would. Set 1.0 to see the
full blue cast of shooting 500T in daylight without an 85 filter:

    python film_sim.py photo.jpg -p 5219 --wb-strength 1.0

Other useful switches:

    --bits 8         8-bit output instead of 16-bit (smaller, may band)
    --seed 42        change the grain pattern; same seed = identical output
    --max-dim 2000   downscale the input first, for quick tests
    -o myfolder      output folder
    --emit-cpp       also regenerate the C++ tables
    --grey-target    where 18% grey lands on screen (default 0.18)


-------------------------------------------------------------------------------
 3b. PERIOD AND ARCHIVAL CONTROLS
-------------------------------------------------------------------------------

The five 1930s-40s stocks:

    python film_sim.py photo.jpg -p ortho          # red-blind: red goes BLACK
    python film_sim.py photo.jpg -p "super xx"     # 1938 film noir negative
    python film_sim.py photo.jpg -p panchrom       # Soviet, late 1930s
    python film_sim.py photo.jpg -p agfacolor      # 1936 colour reversal
    python film_sim.py photo.jpg -p dufaycolor     # additive mosaic, RENDER BIG

VEILING FLARE -- the haze an uncoated pre-1940 lens throws across the whole
frame. It lifts the black floor and flattens contrast, and it is a LENS effect,
not an emulsion one. This is the single biggest reason a period profile looks
period rather than merely soft: without it, a 1938 emulsion still renders with
modern blacks. Each stock carries an era-appropriate default (modern stocks are
zero), so normally you leave this alone. To override:

    --flare 0        clean modern glass, no haze
    --flare 0.12     heavy uncoated-lens flare
    --flare 0.10 -p 5219      give a modern stock a vintage lens

DUPLICATION GENERATIONS -- a release print is three or four generations from the
camera negative, and each one adds grain and softness. This applies to any
negative stock, not only the old ones:

    -g 0    print straight from the camera negative (default)
    -g 1    a normal release print
    -g 3    an archival reissue, visibly grainier and softer

    python film_sim.py photo.jpg -p "super xx" -g 3

Contrast does NOT run away over generations, because real duplicating stock is
gamma 1.0 by design and this model uses the same. What accumulates is grain and
softness. Measured over 0/1/2/3 generations, mid grey holds steady to four
decimal places while grain relative to picture detail rises by 43%. That is the
real mechanism: dupes look grainier because the grain survives the chain better
than the picture does.

DUFAYCOLOR's additive filter grid:

    --no-reseau      disable the mosaic, treat it as a plain B&W record


-------------------------------------------------------------------------------
 4. THREE THINGS THAT WILL CONFUSE YOU IF NOBODY WARNS YOU
-------------------------------------------------------------------------------

(A) RESOLUTION CHANGES HOW MUCH GRAIN YOU SEE. THIS IS CORRECT.

Every spatial quantity is physical -- grain clumps in micrometres, halation radii
in micrometres, sharpness in cycles per millimetre. They are converted to pixels
from the film gauge and your image width. So the model needs enough pixels to
show fine detail.

A 2K render genuinely shows less grain than a 6K render of the same negative,
converging upward as resolution rises. That is real, not a bug: the scanner's
optics band-limit the grain before sampling, which is why 4K rescans of old
negatives look grainier than the 2K masters people remember.

Practical rule: to judge grain, render at least 3000 pixels wide. Below
60 pixels/mm the script prints a warning telling you the width to use. At
1280 px across Super 35 a 4.6 micrometre grain clump is one tenth of a pixel,
so fine-grained stocks cannot possibly show their structure.

Halation has the same dependency. CineStill's widest glow lobe is 700 um; at
512 px wide that is 14 pixels and nearly invisible. Render bigger.

(B) DUFAYCOLOR NEEDS A BIG RENDER OR IT SWITCHES ITSELF OFF.

Its filter grid is a physical 20 lines per millimetre. On Super 35 that is about
500 lines across the frame, and representing them needs at least three pixels per
cell -- roughly 1500 px wide minimum, and 2500 px or more to look right. Below
the minimum the mosaic disables itself and prints a warning telling you the width
to use, because the alternative is aliasing noise rather than a mosaic. Use
period_chart.png (3200 px) rather than test_chart.png for this stock.

(C) SET THE FILM GAUGE CORRECTLY WITH -f.

The default is super35 (24.89 mm wide). If you are simulating a 35 mm still
camera, pass -f ff35 (36 mm), and for medium format -f medium645 (56 mm). Get
this wrong and the grain is the wrong size -- a large negative shows finer grain
for the same emulsion, because the grain is a fixed physical size while the frame
is bigger. That is why medium format looks smoother than 35 mm.


-------------------------------------------------------------------------------
 5. HOW TO CHANGE THE LOOK OF A STOCK
-------------------------------------------------------------------------------

Open film_profiles.py, find the stock, edit numbers, save, re-run. No rebuild.
Then run verify.py to confirm nothing broke.

The parameters that actually control the look, in rough order of impact:

  curves          The characteristic curve: dmin, gamma, toe and shoulder.
                  This is the single biggest influence -- contrast, latitude,
                  highlight rolloff, and colour cast all live here.
                  Raise gamma for more contrast. Move shoulder_x right for more
                  highlight latitude. Making the three channels DIFFER from each
                  other is what produces a colour signature (this is called
                  crossover, and it is how the Fuji cyan shadows are made -- not
                  by tinting the output).
                  NOTE for reversal stocks: the curve runs against NEGATED log
                  exposure, so toe_x controls the HIGHLIGHT end, not the shadows.

  grain           rms_granularity is the amplitude: raise it for more grain.
                  clump_um_* is the size in micrometres.
                  clump_gain is the CHARACTER, and it is independent of size:
                  around 1.0 gives clustered, velvety old-style cubic grain;
                  around 0.2 gives the even, sandy look of modern T-grain.

  mtf             Sharpness, in cycles/mm at 50% response. Lower is softer.
                  Keep f50_r < f50_g < f50_b: red is genuinely the softest
                  channel because the red-sensitive layer sits at the bottom of
                  the emulsion stack, behind two layers of gelatin.

  halation        The glow around highlights. gain_r is usually the largest,
                  because red light penetrates deepest.

  dye_matrix      Colour purity, written as _dye(k). NEGATIVE k increases
                  saturation (Kodachrome -0.30, Velvia -0.42); POSITIVE k
                  desaturates and muddies (ORWOcolor +0.40, Agfacolor Neu
                  +0.45). Do not hand-write these matrices: _dye() keeps the
                  row sums at exactly 1.0, so the matrix changes colour without
                  also shifting density. Hand-written ones drift off 1.0, and
                  then a stock's black level depends on its saturation setting.

  couplers        Inter-layer development effects. Raises saturation without
                  raising contrast, and adds a subtle edge crispness. Zero for
                  every pre-1950 stock -- the chemistry did not exist yet.

  default_flare   Veiling flare of the lens normally used with this stock. A
                  lens property parked on the film profile because era of glass
                  and era of stock go together in practice. 0 for modern stocks,
                  0.09-0.13 for the 1930s ones.

  reseau          Only for additive colour stocks. The filter_matrix inside it
                  is what makes the colour pastel: rows are the R/G/B filters,
                  columns the R/G/B light each one passes. The OFF-DIAGONAL
                  terms are the important ones -- make the filters pure and the
                  process comes out more saturated than Kodachrome, which is
                  exactly backwards.

To add a whole new stock: copy the nearest existing FilmProfile block, rename it,
add aliases, adjust. Run verify.py -- it validates every stock automatically and
will reject an impossible curve or filter grid with an explanatory message.


-------------------------------------------------------------------------------
 6. THE C++ SIDE
-------------------------------------------------------------------------------

Python is the single source of truth for the data. To refresh the C++ tables
after editing film_profiles.py:

    python cpp_codegen.py -o .

That writes film_profiles.hpp and film_profiles.cpp. Add both to your project:

    #include "film_profiles.hpp"

    auto stocks  = film::GetFilmDatabase();    // all 89
    auto prints  = film::GetPrintStocks();     // all 5
    auto formats = film::GetFilmFormats();     // all 14 gauges

    if (film::has(stock.features, film::Feature::Halation)) { ... }
    if (stock.isReversal()) { /* skip the print stage */ }
    if (stock.has_reseau)   { /* additive colour path, see ReseauSpec */ }

Verified with g++ -std=c++20 -Wall -Wextra, and cross-checked against the Python
implementation to six decimal places.

Every field carries its physical unit in a comment, and the header contains the
reference formulae for the characteristic curve, the MTF, the grain spectrum and
the reseau. Implement from those comments and your C++ renderer cannot silently
diverge from the Python reference.

Do not hand-edit the two generated files -- they are overwritten on every run.


-------------------------------------------------------------------------------
 7. WHAT THE PIPELINE DOES, IN ORDER
-------------------------------------------------------------------------------

Order matters. Several steps give visibly wrong results if moved.

   1. Decode sRGB to linear light.
   2. Relative exposure (18% grey = 1.0), exposure offset, taking filters.
   3. Stock colour balance, then veiling flare from the lens.
   4. Large-scale coating unevenness, for stocks with loose QC.
   5. Halation, added to linear exposure, energy conserving.
   6. Emulsion MTF -- light scatter inside the gelatin.
   7. Collapse to one record where the stock has one: monochrome via spectral
      sensitivity, additive colour via the reseau filter grid.
   8. Characteristic curve: exposure to density.
   9. DIR coupler inter-image effects.
  10. Scan: MTF and per-channel misregistration (the pre-sampling filter).
  11. Grain, in the density domain, band-limited by that same scan transfer.
  12. Dye impurity / scanner crosstalk matrix.
  13. Duplication generations, then the print.
  14. Print grain, transmittance to display linear, reseau reconstruction.
  15. Encode sRGB, dither, quantise to 16 or 8 bit.

Reversal stocks skip step 13 entirely: the film IS the positive.


-------------------------------------------------------------------------------
 8. TROUBLESHOOTING
-------------------------------------------------------------------------------

"ModuleNotFoundError: No module named 'numpy'" (or 'PIL')
    Dependencies not installed, or installed for a different Python.
    Use the same interpreter for both:
        py -3.12 -m pip install numpy pillow        (Windows)
        python3.12 -m pip install numpy pillow      (Linux / WSL2)

"ModuleNotFoundError: No module named 'film_profiles'"
    You are running from the wrong directory. cd into the folder holding these
    files first. The scripts import each other by filename.

"[ERROR] not a file: ..."
    Bad image path. On Windows, quote paths containing spaces:
        python film_sim.py "C:\My Photos\shot.jpg" -p 5219

"[WARN] only NN px/mm; grain structure of fine stocks is below the pixel grid"
    Not an error. Your image is too small to show fine grain. See section 4A.
    The message tells you the width to use.

"[WARN] ... reseau pitch is N px ... mosaic disabled"
    Dufaycolor rendered too small. See section 4B. Not an error -- you get a
    plain monochrome record instead of aliasing garbage.

A period stock still looks like a soft modern photo
    Check flare is actually on: the render line prints "flare 13%" when it is.
    If you passed --flare 0 you switched off the single biggest period cue.
    Then try -g 2 or -g 3 for the dupe-generation softening.

verify.py reports a failure
    Read which check failed -- the messages are specific and quote the measured
    number. If you had just edited film_profiles.py, your edit is the cause: an
    invalid curve, a spectral_weights list that no longer sums to 1.0, or a dye
    matrix whose rows no longer sum to 1.

The output looks too dark, too bright, or has an unexpected colour cast
    Check -f: wrong gauge is the usual cause of odd grain, though not of casts.
    A strong blue cast means --wb-strength is set with a tungsten stock.
    Reversal stocks clip much sooner than negative stocks -- that is correct
    behaviour, so try -e -0.5 as a real photographer would.
    Ortho renders red as black. That is the entire point of that stock.

Out of memory on a very large image
    numpy's FFT works in double precision, so a 6K frame needs a few hundred MB.
    Use --max-dim 4000, or render one stock at a time instead of -p all.

The renders look banded in smooth glows
    You used --bits 8. Drop it; 16-bit is the default for exactly this reason.

Windows will not open the 16-bit PNG
    Some older viewers cannot read 16-bit PNG. The files are valid -- Photoshop,
    Affinity, GIMP, Krita and DaVinci Resolve all read them. For a quick look,
    re-render with --bits 8.


-------------------------------------------------------------------------------
 9. HOW GOOD IS THIS, HONESTLY
-------------------------------------------------------------------------------

The structure is right and the physics is right. The NUMBERS are engineering
estimates, marked "# EST" throughout film_profiles.py. They are internally
consistent -- grain, sharpness and latitude all scale correctly across each
product family -- but they are not transcribed from manufacturer datasheets, and
the older or more obscure the stock, the rougher the estimate. Kodachrome and
Technicolor are reconstructions from published descriptions and surviving prints,
not measurements. Treat those two as artistic targets.

The 1930s-40s block is weaker still, and differently so. For the modern stocks
the numbers are estimates anchored to datasheets I could reason about. For the
period stocks there are no datasheets I can consult at all -- the figures are
inferred from how surviving footage looks, from the emulsion technology of the
era, and from internal consistency with the rest of the database. Super-XX is the
firmest of the five, because it stayed in production for decades. Agfacolor Neu
and the Soviet stock are the softest. Dufaycolor's grid pitch of 20 lines/mm is
the only figure in that block I would defend within a factor of two.

On the Soviet stock: it is modelled as a late-1930s Shostka-factory panchromatic
negative. The "Svema" brand name postdates this era, so the profile is
deliberately NOT called that. Its defining trait here is inconsistency, which is
historically well attested -- domestic stock of the period was variable enough
that major productions often preferred imported Agfa or Kodak when they could get
it. If you have real Soviet technical handbook data, that profile is the one most
worth correcting.

UPDATE 2026-08-02: real Soviet handbook data now exists in this database. The
printed sensitometric tables of Gurlev 1986, Iofis 1980 and Gordiychuk/Pell
1979 (scans in PDF/PROFILES/SOVIET/) were transcribed and used to ground the
Svema/Tasma still-film profiles -- see the final chronicle section below and
doc/SOVIET_EXTRACTION_2026-08-02.md. SOVIET_PANCHROM_1939 itself predates
those books and stays a reconstruction.

Two changes would matter far more than anything else remaining:

  1. Replace the "# EST" values with digitised datasheet curves. Kodak publishes
     D-logE curves, MTF curves and RMS granularity for every current VISION3
     stock. Digitise with WebPlotDigitizer, fit the six ToneCurve numbers to the
     real curve. The code is built to accept real data; only the numbers are
     provisional.

  2. Feed scene-referred input. A JPEG or PNG has already had its highlights
     clipped by the camera, so the film's shoulder has nothing left to roll off.
     An EXR, or a raw file developed to linear, is a visible step up.

Also missing, by design: no temporal behaviour and no physical damage. Single
frames only -- no gate weave, no processing flicker, no frame-to-frame grain
animation, no dust or scratches. For the period stocks this is now the largest
remaining gap: veiling flare and dupe generations cover the optical and
photochemical side of the archival look, but not the mechanical one.

One catalogue-number note: you asked for "Kodachrome Tri-X 200 (5266)". Tri-X
Reversal ships as 7266 in 16 mm, and I could not establish a 5266 Tri-X reversal
product. The profile is built as the 7266 emulsion and answers to both numbers.
Correct it if you have a datasheet that says otherwise.

Full technical detail, including every bug the test suite caught during
development and the dye-matrix model fix the period work forced, is in README.md.

===============================================================================


===========================================================================
EXPANSION SET -- 26 STOCKS TO 55
===========================================================================

29 stocks were added after the original 26. The database now holds 55 film
stocks, 4 print stocks and 12 gauges (Super 8 was added at 5.79 mm).

WHAT WAS ADDED

  Agfa B&W          APX 25, APX 100, APX 400
  Agfa colour       Optima 100, Vista 200
  Eastman reversal  Ektachrome EF 5239 (35 mm), 7239 (16 mm)
  Ektachrome        64 daylight, 160T tungsten
  Fuji              F-125 8530 (35 mm), F-125 8630 (16 mm),
                    Neopan Acros 100, Neopan 1600,
                    Provia 400X, Sensia 100
  Polaroid          SX-70, 664, 667
  USSR              Svema Foto-250, Tasma FN-65
  8 mm gauges       generic B&W reversal, generic colour reversal
  Indian cinema     Gevacolor 1952, Gevaert Panchro 1950,
                    Eastman Plus-X 5231
  Britain           Ilford HP3, Ilford HPS
  France            Lumiere Lumichrome
  Italy / LatAm     Ferrania P30

CONFIDENCE TIERS

Every new description starts with a tier tag. This matters -- do not treat
these numbers as uniformly reliable:

  [T1]  Datasheet-grounded. Published speed, granularity and resolution
        figures exist and the numbers are fitted to them. Good to ~10%.
  [T2]  Partially grounded. Speed and reputation documented; grain and MTF
        interpolated from siblings in the same family and era.
  [T3]  Reconstruction. No datasheet available. Built from era, speed class,
        process type and written descriptions. Plausible and internally
        consistent, NOT measurements. Do not cite them as data.

The [T3] set is: Svema Foto-250, Tasma FN-65, both 8 mm entries, Gevacolor
1952, Gevaert Panchro 1950, and Lumiere Lumichrome. Lumichrome is the
weakest of all of them and is flagged as such in its own description.

NAMING NOTES

  Svema         The FN line was cine negative, the Foto- line was still
                film. Both names circulate for the fast stock, so
                "svema fn250", "foto250" and "svema foto 250" all resolve
                to SVEMA_FOTO_250.
  5239 / 7239   Same emulsion, different gauge. The numbers in the two
  8530 / 8630   profiles are deliberately identical. The visible difference
                is magnification, which the renderer derives from the frame
                width you pass with --format, not from the profile. Render
                7239 or 8630 with --format 16mm or super16 or you lose the
                entire point of having them separate.
  8mm BW/COLOR  "8 mm" is a gauge, not an emulsion. These are representative
                home-movie reversal stocks. Render with --format 8mm or
                --format super8.
  Ektachrome    "Ektachrome 1" and "Ektachrome 2" were ambiguous; they are
                interpreted here as Ektachrome 64 daylight and Ektachrome
                160T tungsten.

SOUTH AMERICA

No South American country manufactured raw motion-picture film at scale in
1940-1980. Argentine, Brazilian and Mexican studios shot on imports, and
Ferrania was among the most common. FERRANIA_P30 is therefore labelled as
the Italian stock it is, and is the closest honest match to that cinema.

INDIAN CINEMA

Indian studios also shot on imports through the whole 1940-1960 window.
Domestic manufacture began in 1960 when Hindustan Photo Films opened at
Ootacamund and produced "Indu" branded stock, which is just outside the
window. Gevacolor is documented on Aan (1952), the first Indian feature in
full colour, and on Mother India (1957).

KNOWN LIMITATION -- LOW-DMAX STOCKS DO NOT YET LOOK LOW-DMAX
===========================================================================

The defining property of instant film is a low Dmax: SX-70 tops out near
1.87 where Kodachrome reaches 3.20, so its blacks are genuinely open and
slightly milky no matter how it is exposed.

That is currently NOT visible in the render. _normalised_transmittance()
in film_sim.py rescales each curve's own dmin..dmax to 1..0:

    t_max = 10 ** (-c.dmin)
    t_min = 10 ** (-c.dmax)
    return (10 ** (-d) - t_min) / (t_max - t_min)

Because the stock's own Dmax is the divisor, every stock is stretched to
fill the full output range and the Dmax difference is normalised away. The
Polaroid profiles carry the correct low Dmax in their curves, and the C++
tables carry it too, but the Python renderer flattens it out. Measured on
the test chart, SX-70 and Kodachrome both reach display 0.000.

For negative stocks this is right: the negative is an intermediate and the
print stock's curve sets the final range. For reversal stocks it is wrong,
because the film IS the viewed image.

PROPOSED FIX, NOT YET APPLIED: normalise reversal stocks against a fixed
viewing-black reference (Dmax 3.40, deeper than any real stock) instead of
each stock's own Dmax. Predicted display floors in sRGB:

    POLAROID_SX70          0.159      KODACHROME_64          0.005
    POLAROID_667           0.151      EKTACHROME_64          0.006
    POLAROID_664           0.129      FUJI_VELVIA_50         0.006
    EIGHT_MM_BW            0.103      FUJI_PROVIA_400X       0.008
    AGFACOLOR_NEU_1936     0.052      EIGHT_MM_COLOR         0.009
    DUFAYCOLOR_1937        0.035      EASTMAN_EKTACHROME_*   0.012

That gives the Polaroids and the 8 mm B&W their real milky floor while
leaving Kodachrome, Velvia and the modern E-6 stocks essentially unchanged.
It also, as a side effect, makes Agfacolor Neu and Dufaycolor read more
correctly for their era.

This changes rendered output for all 17 reversal stocks, so it has NOT been
applied. It needs a decision.


===========================================================================
VERIFICATION ROUND 2 -- THREE FIELD OBSERVATIONS TESTED
===========================================================================

Three claims were put to the profiles. Measured results:

1. 8 MM GRAIN -- claim was CORRECT, cause was the opposite of expected
---------------------------------------------------------------------
Magnification works. Direct grain-field test, same emulsion, 1024 px wide:

    format     px/mm   16um clump   grain corr. length
    super35     41.1      0.66 px        1.0 px
    16mm        99.8      1.60 px        2.0 px
    super8     176.9      2.83 px        2.8 px
    8mm        213.3      3.41 px        3.4 px

Per-pixel sigma stays near-constant (0.0184 -> 0.0219) because RMS
granularity is calibrated to be resolution-independent by design. Grain gets
spatially BIGGER with a smaller gauge, not stronger. That is correct physics.

THE TRAP: the default format is super35 and run.cmd passes no -f. So
`film_sim.py img -p "8mm bw"` renders an 8 mm emulsion at 35 mm scale --
0.66 px grain, invisible. ALWAYS pass -f 8mm or -f super8 for those two.

SEPARATE REAL BUG, now fixed: the 8 mm emulsion numbers were too COARSE.
EIGHT_MM_BW had RMS 13.0 / clump 16.0 at EI 40, coarser than
KODAK_TRI_X_REVERSAL_200 at RMS 10.0 / clump 14.0 -- five times the speed.
EIGHT_MM_COLOR had RMS 10.2 at EI 40 against EKTACHROME_64's 4.8. Both had
"8 mm looks grainy" baked into the emulsion, double-counting magnification
the renderer already applies. Corrected to proper slow-reversal values:

    EIGHT_MM_BW     RMS 13.0 -> 8.5   clump 16.0 -> 11.0   f50 48 -> 62
    EIGHT_MM_COLOR  RMS 10.2 -> 5.5   clump 12.0 ->  7.0   f50 58 -> 76

2. SVEMA FOTO-250 CONTRAST -- claim was CORRECT, now fixed
----------------------------------------------------------
Claim: Foto-250 is more contrasty than FN-65, with fewer middle greys pushed
toward pure black and pure white.

As originally built it was the FLATTEST of the three. Measured on a 12-stop
ramp through the print path (mid% = fraction of output in 0.35..0.65):

    BEFORE                 mid%   contrast   min     max
    SVEMA_FN_64           13.2%    0.909    0.054   0.984
    TASMA_FN_64           13.8%    0.898    0.056   0.979
    SVEMA_FOTO_250        14.2%    0.888    0.058   0.974   <- flattest

Which parameter delivers the described look was tested explicitly. A first
test on the bare curve suggested gamma did nothing -- that test was WRONG,
because normalising by the curve's own dmin..dmax divides gamma out. Through
the real print path gamma is exactly the knob, and shortening latitude does
the OPPOSITE of what intuition suggests:

    Foto-250 gamma 0.95        mid 11.8%  contrast 0.937  min 0.031  max 0.988
    Foto-250 gamma 1.05        mid 10.7%  contrast 0.959  min 0.020  max 0.993
    Foto-250 latitude 7.0 st   mid 15.0%  contrast 0.819  min 0.100  max 0.941

Applied: gamma 0.800 -> 0.950. Result:

    AFTER                  mid%   contrast   min     max
    SVEMA_FN_64           13.2%    0.909    0.054   0.984
    TASMA_FN_64           13.8%    0.898    0.056   0.979
    SVEMA_FOTO_250        11.8%    0.937    0.031   0.988   <- now contrastiest

Fewest midtones, deepest black, brightest white. Matches the description.

SENSITOMETRIC CAVEAT, stated once: general practice is that a faster
emulsion has LOWER inherent gamma, because larger crystals give a broader
spread of grain sensitivities and a flatter curve. Every Western family in
this database follows that. Foto-250 now breaks it deliberately, on the
strength of direct field experience with the stock, and because Soviet
amateur film was commonly developed in high-contrast universal developers --
which is a development-practice effect rather than an emulsion property, but
it is what the film actually looked like in use.

3. TASMA WARM / BROWN-BLACK -- claim is PHYSICALLY REAL, NOT REPRESENTABLE
--------------------------------------------------------------------------
The claim is sound. B&W image tone depends on developed silver particle
size: fine particles scatter short wavelengths and read warm or brown,
coarse filamentary silver reads neutral or blue-black. This is why lith
prints and fine-grain period films look faintly sepia even untoned.

It cannot currently be expressed. Monochrome stocks are forced exactly
neutral:
  - film_sim.py line 796 replaces print curves with the green curve for
    mono negatives
  - base_tint has NO effect on mono. Measured: setting TASMA_FN_64 to
    base_tint=(1.06, 1.00, 0.90) leaves output at R=G=B=0.4602, R-B exactly
    0.0000
  - the residual 1/255 channel spread seen in renders is dither, not a cast

Doing this properly needs a NEW mechanism, not a tweak: image tone is
density-dependent (warm in low densities, neutralising as density rises), so
it wants something like a `silver_tone` parameter driving a hue shift as a
function of density, plus relaxing the exact-neutrality assertion in
verify.py. That is a feature. Not implemented; awaiting a decision.

NEW IN THIS ROUND
-----------------
  ORWOCOLOR_NC24        [T3] colour negative, EI 160. CAVEAT: the "NC 24"
                        designation could not be confirmed. Documented ORWO
                        NC series is NC 3, 5, 16, 19, 21. Built as a family
                        interpolation -- later, faster and slightly cleaner
                        than NC 21. Supply a real speed or datasheet and it
                        can be refitted.
  TASMA_POSITIVE_28     [T3] Soviet B&W cine positive, Tasma, GOST 2.8 (about
                        ISO 3) -- the yellow boxes. Added as a PRINT STOCK,
                        not a film profile, because a positive film is what a
                        negative is printed ONTO, which is exactly the role
                        PrintStock fills. Use it as:
                          film_sim.py img -p fn65 --print TASMA_POSITIVE_28
                        for a period Soviet release-print look. Print gamma
                        2.52 gives the contrasty projected image; its own
                        grain is fine, so visible grain still comes from the
                        negative.

Database is now 56 film stocks, 5 print stocks, 12 gauges.
All 67 verify checks pass.


===========================================================================
NATIVE GAUGE PER STOCK  +  ALPHABETICAL ORDER  +  PRINT STOCKS UNDER -p all
===========================================================================

1. WHY 8 MM STILL LOOKED LIKE 35 MM -- root cause fixed properly
----------------------------------------------------------------
Previous round said "pass -f 8mm". That was a workaround, not a fix: with
`-p all` you cannot pass a per-stock gauge, so every stock rendered at
Super 35 and an 8 mm home-movie emulsion came out with 35 mm grain and 35 mm
detail. Correct, given the format asked for, and useless.

FilmProfile now carries `default_format`, the gauge the stock was actually
sold on. The renderer uses it whenever --format is not given. --format still
works and now means "override every stock", which is what it should mean.

  gauge pairs        7239 -> 16mm, 8630 -> 16mm, TRI_X_REVERSAL_200 -> 16mm
  8 mm               GENERIC_BW, GENERIC_COLOR (ex EIGHT_MM_*) -> 8mm
  instant film       SX-70 -> polaroid_sx70 (79 mm), 664/667 ->
                     polaroid_pack (95 mm); both are new FORMATS entries
  35 mm STILL films  -> ff35 (36.00 mm). A still frame is wider than a
                     Super 35 cine frame, so it is magnified LESS. 23 stocks.
  35 mm cine         -> super35 (unchanged), three-strip -> techni35

MEASURED, all at 1024 px wide, each stock at its own native gauge:

  stock                    gauge          px/mm  clump_px  grain_px  cyc/frame
  EIGHT_MM_BW              8mm              213     2.99      3.0        211
  EIGHT_MM_COLOR           8mm              213     2.13      2.0        269
  SVEMA_FN_64              super35           41     0.62      1.0        846
  KODAK_VISION3_500T       super35           41     0.47      1.0       1294
  KODAK_PORTRA_400         ff35              28     0.20      1.0       2664
  FUJI_NEOPAN_ACROS_100    ff35              28     0.20      1.0       3744
  KODACHROME_64            ff35              28     0.11      1.0       3456

8 mm now has 3x the grain size of any 35 mm stock and 4x to 18x less
resolvable detail across the frame. Both of those are what was asked for, and
both come from the gauge, not from faked emulsion numbers.

The 8 mm emulsion numbers were also raised again, having been cut too far
last round. They are now amateur-grade rather than premium, which is what
cheap 8 mm home-movie stock actually was:

  EIGHT_MM_BW     RMS 8.5 -> 11.0   clump 11.0 -> 14.0   f50 62 -> 44
  EIGHT_MM_COLOR  RMS 5.5 ->  8.0   clump  7.0 -> 10.0   f50 76 -> 56

Reasoning, so the choice is auditable: physically a slow EI 40 reversal
emulsion should be FINER than the EI 200 Tri-X Reversal in this set, and on
that argument the previous values were defensible. But the 8 mm entries are
explicitly generic amateur stock, not a premium emulsion, and 8 mm camera
lenses were poor -- the softer MTF folds that in, the same way default_flare
already folds in uncoated-lens scatter. Net effect is visible grain and low
detail arriving from three independent, individually honest mechanisms
instead of one exaggerated one.

2. ALPHABETICAL ORDER
---------------------
FILM_PROFILES is now sorted by name. The literal in the source stays grouped
by manufacturer and era, because that is how it is maintained; the sort is
applied once after the literal, so --list, the C++ table and `-p all` all
come out alphabetical. First entry AGFACOLOR_NEU_1936, last
TECHNICOLOR_THREE_STRIP.

3. PRINT STOCKS NOW RENDER UNDER -p all
---------------------------------------
TASMA_POSITIVE_28 never appeared because `-p all` iterates FILM_PROFILES, and
a print stock is not a film profile -- it is not something you expose in a
camera. Rather than fake a profile for it, `-p all` now also renders every
print stock through a reference negative:

  SCAN_DI, KODAK_2383_RELEASE, DUPE_FINE_GRAIN  on KODAK_PORTRA_400
  TECHNICOLOR_IB                               on TECHNICOLOR_THREE_STRIP
  TASMA_POSITIVE_28                            on EASTMAN_PLUS_X_5231

Mono print stocks get a mono negative, colour ones a colour negative. Output
is named <image>_PRINT_<name>.png so it sorts away from the stock renders.
`-p all` now writes 61 files: 56 stocks + 5 print stocks. Passing an explicit
--print suppresses the extra pass.

Still true, and worth repeating: TASMA_POSITIVE_28 is most useful aimed at a
Soviet negative rather than at Plus-X --

  python film_sim.py img.png -p fn65 --print TASMA_POSITIVE_28

All 67 verify checks pass. 56 film stocks, 5 print stocks, 14 gauges.


===========================================================================
LOMOGRAPHY GALLERY CROSS-CHECK -- WHAT IT COULD AND COULD NOT SETTLE
===========================================================================

The community galleries for Svema FN64, Svema FN 250 and Tasma FN64 were
consulted. Two hard limits, stated up front:

  1. THE PHOTOGRAPHS THEMSELVES COULD NOT BE EXAMINED. The available web
     tooling returns HTML text, not image data, and fetching the image files
     by other means is not permitted. So nothing below is based on looking at
     a single picture. Any claim that the profiles "match the photos" would
     be fabricated.

  2. NEITHER GALLERY PAGE CARRIES A DATASHEET. No ISO, no granularity, no
     resolving power, no characteristic curve. They are photo galleries with
     no published sensitometry, so they cannot calibrate rms_granularity,
     f50 or gamma even in principle.

WHAT THE METADATA DID SETTLE -- and it matters more than the pictures would

A representative FN 250 frame carries this metadata:

    title   "250 ASA @ 25 ISO"
    album   "#70 Svema FN 250 (exp 1/1993) - Konica C35 AF2"
    camera  Konica C35 AF2, Hexanon 38 mm f2.8, Ukraine
    posted  2022-07-25

Read that carefully. The stock expired in January 1993 and was shot in 2022 --
about 29 years past expiry -- and was deliberately rated at 25 ISO against a
box speed of 250. That is a 3 1/3 stop overexposure, which is the standard
correction for the speed loss and heavy base fog that decades-old B&W film
develops.

So these frames do not show SVEMA FN 250 AS MANUFACTURED. They show
29-year-expired stock, massively overexposed to compensate, developed in an
unknown developer, scanned on an unknown scanner, and very possibly curve-
adjusted in software before upload. Every one of those steps moves contrast,
fog and effective grain, and none of them is recorded.

Sample sizes reinforce the point: FN64 has roughly 100 uploads across two
pages, FN 250 has 17 on a single page, and a large share of the FN 250 set is
one photographer's single expired roll.

CONSEQUENCE FOR THIS DATABASE

The three Soviet profiles stay at tier [T3] -- reconstruction, not
measurement. The galleries did not upgrade them and could not have. Anyone
tempted to "fit" a profile to these images would be fitting expired stock,
push-processing and somebody's scanner curve, and baking all three into what
is supposed to be an emulsion model. That would be worse than the current
honest estimate, not better.

Worth noting: this cuts against a change already made on field-experience
grounds. SVEMA_FOTO_250's gamma was raised from 0.800 to 0.950 to make it the
contrastiest of the Soviet set. Heavily overexposed expired film shot at 25
ISO would read as LOW contrast with a lifted, foggy black -- the opposite. So
the galleries neither confirm nor refute that change; they are simply not
evidence about it either way. The change rests on direct experience of the
film when it was fresh, which is a better source than an expired roll, but it
remains unverified against sensitometry.

ONE CONCRETE CORRECTION DID COME OUT OF THIS

Lomography indexes the Tasma stock as "Tasma FN64", not FN-65. Both
designations circulate -- 65 matches the GOST speed step, 64 the ISO
equivalent. Renamed TASMA_FN_65 -> TASMA_FN_64, exposure_index 65 -> 64. The
fn65 / fn-65 / "tasma fn 65" aliases all still resolve to it, so nothing
breaks. Note "fn64" alone still resolves to SVEMA_FOTO_65 (ex SVEMA_FN_64); use "tasma" or
"fn64t" for the Tasma one.

WHAT WOULD ACTUALLY CALIBRATE THESE

In rough order of value:
  - a GOST or manufacturer datasheet for FN-64 / FN-250 / Tasma FN-64, giving
    speed, granularity and resolving power
  - a scan of a step wedge shot on fresh stock and developed to a stated
    time/developer/temperature -- this alone would pin gamma, dmin and Dmax
  - unmodified full-resolution scans with the scanner and developer recorded,
    from stock inside its expiry date
  - failing all of the above, the current position is the honest one: [T3],
    labelled as reconstruction.

If you have any Soviet-era datasheets, or can shoot and scan a wedge, the
profiles can be refitted properly in an afternoon.


===========================================================================
CALIBRATION AGAINST 9 REAL SCANS  (3x FN250, 3x SVEMA FN64, 3x TASMA FN64)
===========================================================================

Nine user-supplied scans were measured. Unlike the gallery round, these could
actually be examined. Results, and they are mixed.

MEASUREMENT 1 -- SILVER IMAGE TONE: CONFIRMED, and now implemented
------------------------------------------------------------------
Mean channel difference over each whole frame, in 0..255 units:

    file                  R-G     B-G
    001 - TASMA FN64     +8.6    -0.9
    003 - TASMA FN64    +15.6    +2.2
    002 - TASMA FN64      0.0     0.0
    001 - SVEMA FN64     -2.8    +1.7
    002/003 - SVEMA FN64  0.0     0.0
    all three FN250       0.0     0.0   (stored as pure greyscale)

Two of three Tasma frames carry a clear WARM cast, and both are bright frames
(mean level 0.92 and 0.78) -- exactly where fine-silver warm tone is expected,
since the effect is strongest where there is least silver. One Svema frame is
marginally COOL. This is the first hard evidence for the brown-black claim, so
the feature was built:

  NEW FIELD  FilmProfile.silver_tone   >0 warm/brown, <0 cool/blue, 0 neutral

  Applied in film_sim.py stage 14c, weighted by output level so it is
  strongest in the light tones and fades as density builds.

  CRITICAL DESIGN POINT: this is NOT base_tint, and could not have been.
  base_tint is compensated by the printer-light anchor solve -- a real printer
  neutralises the film base -- so it produces a cast of exactly 0.0000 on a
  mono stock, as measured earlier. silver_tone runs AFTER the anchor solve and
  therefore survives it.

  Values set from the measurement:
    TASMA_FN_64      +1.00   calibrated to the larger of the two casts
    SVEMA_FN_64      -0.25   single frame, weak evidence, small value
    SVEMA_FOTO_250    0.00   no tone data exists in the supplied files

  Rendered result, R-G in 255 units, vs measured +8.6..+15.6:
    output level 0.35 -> +7.0     0.55 -> +11.4     0.80 -> +15.4
  Brackets the measurement across the tonal range.

MEASUREMENT 2 -- GRAIN: NOT RESOLVABLE FROM THESE FILES
-------------------------------------------------------
Grain sigma was measured in the flattest 10-50 % of 24x24 blocks, each
plane-detrended so a smooth gradient is not miscounted as noise. Grain scales
with density (Poisson), so only blocks at matched mean level can be compared:

    mid-density subset (mean 0.35-0.60)
      FN250   n=3   sigma 0.0483 .. 0.0584   mean 0.0547
      FN64    n=2   sigma 0.0428 .. 0.0648   mean 0.0538
      TASMA   n=1   sigma 0.0487

The FN250-vs-FN64 difference is 0.0009. The spread WITHIN the two FN64 frames
alone is 0.0220 -- twenty-five times larger than the difference being tested.
The grainiest single measurement in the whole set is an FN64 frame, not an
FN250 one.

Worse, a sampling limit rules this out in principle. At 1216 px across a 36 mm
frame the pitch is 33.8 px/mm, so one pixel spans 29.6 um. The modelled clumps
are 15.0 um (FN64), 21.5 um (FN250) and 16.0 um (Tasma) -- that is 0.51, 0.73
and 0.54 PIXELS. All three sit below the sampling limit of these JPEGs. What
the "grain sigma" figure above actually contains is mostly scanner noise, JPEG
artefacts and sharpening halos, not resolved film grain.

REVISED CONCLUSION -- grain AMPLITUDE was measurable after all
The first pass above compared badly-matched density blocks and concluded there
was no signal. Redone properly -- 48x48 blocks, plane+cross-term detrended,
flattest 10-50 % only, restricted to comparable mid density -- a consistent
signal appears:

    FN250   n=3   flat sigma 0.0502   (means 0.47-0.77)
    FN64    n=2   flat sigma 0.0299   (means 0.67-0.79)
    MEASURED RATIO 1.68x

Same direction in every frame. The model had 1.42x. So FN250 really is markedly
grainier than FN64 in amplitude, more so than was modelled -- the claim holds.

FITTED, and note the trap: scaling RMS by 1.68 does NOT give a 1.68x result.
11.5 * 1.68 = 19.4 renders at only 1.42x, because FN250's coarser clump (21.5 um
against 15.0) spreads spectral energy differently and grain_reference_energy()
compensates. The value has to be swept against rendered pipeline output:

    rms 19.4 -> 1.42x     23.0 -> 1.60x     25.0 -> 1.70x     27.0 -> 1.80x

  SVEMA_FOTO_250.rms_granularity  16.2 -> 25.0   [T1, fitted to measurement]
  rendered FN250/FN64 now 1.70x against a measured 1.68x

GRAIN SIZE, separately, is still NOT measurable and was NOT changed. Two
independent attempts confirm why:
  - correlation length at ACF=0.5 came out 0.56-0.68 px for FN250 and
    0.78-1.37 px for FN64/Tasma. Every value sits at or below the 1 px
    sampling limit, so it measures the scanner+JPEG MTF, not the emulsion.
    Taken at face value it would say FN250 has the SMALLEST grain of the
    three -- contradicting both the claim and the model, which is the
    signature of an invalid measurement rather than a finding.
  - clumping index (residual ACF at lags 2-5 px, which IS above the sampling
    limit) came out FN250 +0.39, FN64 +0.84, Tasma +1.12 at mid density, i.e.
    FN250 least clumped. But that metric is the one most distorted by scanner
    sharpening, and it runs against the physics -- larger crystals cluster
    more, not less. Reported, deliberately not acted on.

So clump_um stays 21.5 and clump_gain stays 1.70, both still [T3]. Only the
amplitude moved, because only the amplitude was measurable.

MEASUREMENT 3 -- CONTRAST: WEAKLY AGAINST A CHANGE MADE EARLIER
---------------------------------------------------------------
The fraction of pixels in the mid range (0.35-0.65) runs from 3.1 % to 45.3 %
across these nine frames. That is scene content, not emulsion -- a photograph
of a white wall and a photograph of a night street cannot be compared this way.
So mid-tone fraction is useless as a film metric on arbitrary scenes, and any
"contrast" conclusion drawn from it would be noise.

One metric is less scene-dependent: the white point (p99.9), because most
daylight scenes contain something near-white.

    FN250   0.890  0.796  0.941   mean 0.876   <- never reaches white
    FN64    0.996  0.984  0.796   mean 0.925
    TASMA   0.984  0.980  0.988   mean 0.984

All three FN250 frames fail to reach white; two fall below 0.90. Tasma clears
0.98 in all three. A lifted, compressed highlight end like that is the
signature of LOW contrast with heavy base fog -- which points the opposite way
to the earlier decision to raise SVEMA_FOTO_250's gamma from 0.800 to 0.950.

The gamma was LEFT AT 0.950 anyway, and here is the reasoning, so it can be
overruled:
  - the gallery metadata established that circulating FN250 is expired stock;
    one documented frame was 29 years past expiry and rated 25 ISO against a
    box speed of 250, a 3 1/3 stop overexposure
  - expired, heavily overexposed film is exactly what produces a lifted, foggy,
    compressed highlight end -- so the low white points are explained by the
    stock's age, not by the emulsion's design gamma
  - n=3, uncontrolled developer, scanner and exposure
  - against that, direct experience of the film when fresh is the better source
So the measurement is consistent with expired FN250 and says little about fresh
FN250. It is recorded here rather than acted on. If a fresh-stock step wedge
ever contradicts gamma 0.950, that wedge wins.

WHAT WOULD STILL SETTLE IT
--------------------------
  - grain: a full-resolution scan, no downsizing, no sharpening, scanner ppi
    recorded. At 4000 ppi one pixel is 6.4 um and a 15-21 um clump is properly
    resolved -- roughly 5x the sampling density of these files.
  - contrast: a step wedge on fresh stock with developer, time and temperature
    recorded. Nothing else pins gamma, dmin and Dmax honestly.
  - tone: already usable. More Tasma frames saved as RGB rather than greyscale
    would tighten silver_tone beyond the current two data points.

ANSWER TO "DID THESE PHOTOS HELP BUILD A CORRECT PROFILE?"
Partly. They settled the image-tone question outright and produced a real new
feature calibrated to measurement. They cannot settle grain -- the resolution
forbids it. On contrast they lean against a previous decision without being
able to overturn it, because the stock in circulation is expired. One measured
change, one refusal to change, one recorded tension. Tier stays [T3] for all
three Soviet stocks except that silver_tone on Tasma is now [T1] -- fitted to
supplied measurements.

NEW TOOL: sort_profiles.py
--------------------------
Sorts the FilmProfile blocks in film_profiles.py into alphabetical order in the
SOURCE file, not just at import. Source-to-source: blocks move verbatim,
including their comments. Refuses to write unless the block count, name set and
character multiset are unchanged, the rewritten module imports, validate_all()
passes, and every profile still compares equal field for field. Timestamped
backup first; any failure restores the original.

    python sort_profiles.py            # sort in place
    python sort_profiles.py --check    # report order, change nothing
    python cpp_codegen.py -o .         # then keep the C++ tables in step

Already applied: the literal in film_profiles.py is now alphabetical, first
AGFACOLOR_NEU_1936, last TECHNICOLOR_THREE_STRIP.


===========================================================================
8 MM SURFACE LOOKED FLAT -- ROOT CAUSE AND FIX
===========================================================================

Report: 8 mm detail level looked right, but the surface read as smooth and
blurred rather than grainy.

ROOT CAUSE -- reversal stocks get no print-gamma amplification
--------------------------------------------------------------
Not a pipeline ordering bug. Grain is stage 11, applied AFTER both MTF stages
(6 emulsion, 10 scan), so it is not being blurred by them. Verified.

The real cause is an asymmetry that was never accounted for. Measured, grain
field sigma versus final output sigma:

    stock              px/mm   field    output   kept
    EIGHT_MM_BW           67  0.01877  0.01183   0.63
    EIGHT_MM_COLOR        67  0.01660  0.01303   0.78
    SVEMA_FOTO_250         9  0.01756  0.02181   1.24
    SVEMA_FN_64           13  0.00933  0.01257   1.35

A negative GAINS amplitude, a reversal stock LOSES it. That is physically
correct and should not be "fixed" in the pipeline: a negative's grain passes
through the print stage, where it is multiplied by the print gamma (~1.75 for
SCAN_DI) and print grain is added on top. A reversal stock is the viewed image
-- there is nothing downstream to amplify it.

The consequence was missed when the 8 mm entries were written: a reversal
emulsion needs a genuinely HIGHER rms_granularity than a negative to read as
equally grainy, because it does not get the print multiplier. The 8 mm stocks
were numbered as if they would.

WHY "MATCH FN250" IS THE WRONG TARGET
-------------------------------------
Swept against FN250 (output sigma 0.0315), 8 mm never gets close at any sane
value: rms 11 -> 0.0137, 20 -> 0.0184, 28 -> 0.0231. Reaching 0.0315 would
need rms above 40 for an EI 40 emulsion, which is nonsense.

FN250 is the wrong yardstick. It sits at rms 25.0, already flagged as the
grainiest emulsion in the database -- above Ilford HPS at EI 800 (19.0) and
Delta 3200 at EI 3200 (16.0) -- and fitted to EXPIRED stock, so some of that
25.0 is age rather than emulsion. Chaining a second stock to it would
propagate that error.

The fair comparison is other reversal stocks, which also lack a print stage:

    KODAK_TRI_X_REVERSAL_200   EI 200   rms 10.0   sigma 0.00769
    EASTMAN_EKTACHROME_5239    EI 160   rms 10.4   sigma 0.00776
    KODACHROME_64              EI  64   rms  2.2   sigma 0.00207

APPLIED
-------
    EIGHT_MM_BW     rms 11.0 -> 19.0   clump 14.0 -> 17.0   clump_gain 1.25 -> 1.45
                    fog_grain 0.24 -> 0.26
    EIGHT_MM_COLOR  rms  8.0 -> 12.0   clump 10.0 -> 11.5   clump_gain 0.55 -> 0.70
                    fog_grain 0.20 -> 0.22

Result, output sigma relative to Tri-X Reversal:

    EIGHT_MM_BW      0.01787   2.32x
    EIGHT_MM_COLOR   0.01607   2.09x
    SVEMA_FN_64      0.01665   2.17x
    SVEMA_FOTO_250   0.03151   4.10x

8 mm now reads as roughly twice as grainy as a good 16 mm reversal stock and
comparable to Svema FN-64, while staying below the FN250 outlier. clump_gain
was raised as well as rms because clumping is low-frequency energy -- it is
what reads as coarse TEXTURE rather than fine speckle, and it survives the
soft MTF that makes the 8 mm image look smooth.

Justification for an EI 40 emulsion carrying rms 19.0, since it breaks the
speed-granularity trend the rest of the database follows:
  - these are explicitly generic AMATEUR stocks, not premium emulsions
  - reversal processing develops the unexposed silver, which gives coarser
    apparent grain than negative processing of the same crystals
  - no print stage exists to supply the usual ~1.75x amplification
All three are real effects, and none of them is the speed relationship, so the
trend is not actually violated -- 8 mm is off it for stated reasons.

Still [T3]. No 8 mm samples were supplied, so this is reasoned, not measured.

(The chronicle sections above are a running log; stock counts quoted inside
them -- 55, 56, 61 files under -p all, and so on -- are what was true at the
time each section was written. Current totals are in the section below.)


===========================================================================
SOVIET REFERENCE-BOOK PASS -- 2026-08-02 -- 83 TO 89 STOCKS
===========================================================================

The wish from section 9 ("if you have real Soviet technical handbook data,
that profile is the one most worth correcting") was granted. Three printed
USSR references were scanned into PDF/PROFILES/SOVIET/ and their
sensitometric tables transcribed page by page:

  Гурлев Д. С., «Справочник по фотографии (светотехника и материалы)»,
      Киев: Техніка, 1986 [Gurlev D. S., "Handbook of Photography (Light
      Engineering and Materials)", Kyiv: Tekhnika, 1986]
  Иофис Е. А., «Кинофотопроцессы и материалы», 2-е изд., М.: Искусство,
      1980 [Iofis E. A., "Cine and Photo Processes and Materials", 2nd ed.,
      Moscow: Iskusstvo, 1980]
  Гордийчук И. Б., Пелль В. Г., «Справочник кинооператора», М.: Искусство,
      1979 [Gordiychuk I. B., Pell V. G., "Cinematographer's Handbook",
      Moscow: Iskusstvo, 1979]

Full transcriptions with page numbers: doc/SOVIET_EXTRACTION_2026-08-02.md.
Change log: doc/CHANGES_2026-08-02_soviet.md.

SIX NEW STOCKS, curves and densities datasheet-grounded [T2], grain
honestly [T3] (no Soviet source prints granularity for the still films):

  SVEMA_FOTO_32     B&W neg, S 32 GOST, gamma 0.8 (CT-2), R 135 lin/mm
  SVEMA_FOTO_130    B&W neg, S 130, gamma 0.8, R 100 -- and a documented
                    580 nm sensitization cut: orthopanchromatic, reds go
                    dark. Not a typo; it is what Gurlev prints.
  SVEMA_DS_4        colour negative, UNMASKED, daylight 5500 K, S 45,
                    overall gamma 0.8, R 63 lin/mm
  SVEMA_TSNL_32     colour negative, masked, tungsten 3200 K, S 32,
                    documented orange-mask dmin ladder, narrow 0.9 logH
                    latitude, R 58
  SVEMA_TSNL_65     colour negative, masked, tungsten 3200 K, S 65,
                    wide 1.5 logH latitude, R 63
  TASMA_OCH_45      B&W REVERSAL, S 45, Dmax 1.9 / Dmin 0.08 printed,
                    gamma 1.1-1.6 window, sensitized to 680 nm

RENAME: ORWO_UT18 -> ORWO_CHROM_UT18. The factory leaflet W 746 (VEB
Filmfabrik Wolfen, scan in PDF/PROFILES/ORWO/) prints the official name
"ORWO CHROM-FILM UT 18": 18 DIN / 50 ASA / 45 GOST, daylight. All old
aliases still resolve.

ALIAS: Svema «Фото-65» is the still-film designation of the FN-64 class
emulsion; Gurlev's printed Foto-65 column (gamma 0.8, D0 0.05, R 110
lin/mm, 665 nm) agrees with the measured profile, so SVEMA_FN_64 now
answers to foto-65 / svema foto-65 and its tier rose 3 -> 2.

TRANSLITERATION CONVENTION, used consistently in Python, C++ and the name
list: З->Z, Л->L, Ц->TS, Ч->CH. СВЕМА ФОТО-32 -> SVEMA FOTO-32,
ТАСМА МЗ-3Л -> TASMA MZ-3L, ОЧ-45 -> OCH-45, ЦНЛ -> TSNL.

NEW TOOL gen_film_names.py: writes film_names.txt -- every official film
name, quoted, one per line, no commas, in exactly the order of the C++
std::vector (both iterate the same sorted FILM_PROFILES tuple; the order
was verified programmatically 1:1). Run it only after cpp_codegen.py output
has been generated and verified.

Provenance rule for the Russian sources, applied throughout: the original
book title is kept unchanged in the citation, with the English translation
and translated author name alongside. The citations are embedded both in
_PROVENANCE_SOURCES (queryable) and in the generated .cpp comment blocks.

Also in this pass: 25 files moved to PDF/PROFILES/DELETE_CANDIDATE
(byte-identical duplicates, URL-only pointers, paper/toner/label documents,
and the DjVu superseded by its PDF conversion). Nothing deleted.

verify.py count assertions updated 83/20 -> 89/21 -- the only test change.

Database is now 93 film stocks, 5 print stocks, 14 gauges.
All 67 verify checks pass. C++ compiles clean (g++ -std=c++17).

SECOND PASS, SAME DAY -- CHIBISOV APPENDIX TABLES + MEASURED DUFAYCOLOR
-----------------------------------------------------------------------
The fourth reference was mined on the owner's pointer: Чибисов К. В. и др.,
«Фотография в прошлом, настоящем и будущем», М.: Наука, 1988 [Chibisov
K. V. et al., "Photography in the Past, Present and Future", Moscow: Nauka,
1988], Appendix Table I (book p157-158, rotated pages) plus a survey of
appendix tables II-XIV. Details: doc/SOVIET_EXTRACTION_2026-08-02.md.

  TASMA_OCH_45     gamma 1.35 -> 1.50 and R -> 110 mm^-1 (Chibisov prints
                   the OCh-45 product row: gamma_rec 1.6, R 110).
  Foto line        Chibisov's R figures (116/92/75/70) CONFLICT with
                   Gurlev's (135/110/100/82). Gurlev kept; both cited, so
                   the tension is auditable, not hidden.
  Kodak 5247/5294  Table VIII prints Soviet lab measurements (RMS
                   granularity, MTF@30, mean gradient) for western cine
                   stocks. Recorded as citations; grain NOT adopted
                   (cross-era metric equivalence unverified). The printed
                   5294 green MTF@30 of 0.65 matches the existing profile
                   exactly -- a free confirmation.
  Table IX         Independently confirms every DS-4 / TsNL-32 / TsNL-65
                   value adopted from Gurlev. Extra Soviet colour stocks
                   sit ready in the book if ever wanted: TsNL-90,
                   TsO-65, TsO-T-90L, TsOD-16/32.

  DUFAYCOLOR_1937  On the owner's instruction the reseau filter_matrix is
                   now MEASURED, not estimated: derived from the NSMM
                   Bradford absorbance curves of three surviving prints
                   (DUFAYCOLOR/measuredODs_MSI_NSMM_*.jpg), band-averaged
                   T = 10^-A, rescaled, rows normalised (within-row
                   crosstalk ratios exact). Blue element leaks red, green
                   leaks both ways, red is cleanest -- the pastel is now
                   evidence-based. Tier 3 -> 2.

All 67 verify checks pass after every change; C++ and film_names.txt
regenerated in step.


===========================================================================
SCHEMA V3 -- DIGITISED SPECTRAL SENSITIVITY CURVES -- 2026-08-02
===========================================================================

The biggest remaining realism lever is now wired: real spectral
sensitivity curves from datasheet plots, as data, not as three-number
approximations. SCHEMA_VERSION 2 -> 3.

NEW STRUCT SpectralSensitivity, appended to FilmProfile (aggregate order
of the v1/v2 prefix unchanged, same rule as the v2 additions):

  lambda_start_nm + lambda_step_nm   sampling grid (length varies per
                                     stock: IR extends past 800 nm)
  log_s_r / log_s_g / log_s_b        SENSITIVE LAYERS of a colour stock
                                     (cyan-/magenta-/yellow-forming), not
                                     output channels
  log_s_pan                          single record: B&W and reseau stocks
  criterion                          what the source plot's y-axis means
                                     (the sheets differ; mixing conventions
                                     silently corrupts comparisons)
  source                             full citation: author/publisher,
                                     document title and code, ORIGINAL
                                     document release date; Russian sources
                                     keep the original title plus English
                                     translation

Values are relative log10 sensitivity, peak-normalised to 0.0 per layer;
-4.0 = below the plot's measurement floor. Absolute speed stays in
exposure_index. Empty struct = renderer falls back to spectral_weights /
taking_matrix exactly as before, so nothing breaks for the other 86
stocks.

PILOT STOCKS, three deliberately different cases:

  FUJI_NEOPAN_ACROS_100     wedge spectrogram, AF3-095E sec. 12 (2001).
                            Orthopan signature captured: 500 nm dip,
                            580 nm peak, hard cut at ~655 nm.
  KODAK_VISION3_250D_5207   three-layer curves, Kodak H-1-5207 (film
                            2009, sheet rev. March 2026). Per-layer
                            sheet-absolute peaks recorded in the comment;
                            y-axis convention preserved in `criterion`.
  KONICA_INFRARED_750       two-lobe IR record (intrinsic 400-500 nm +
                            640-820 nm, peak 750 nm) -- the sheet's
                            unlabelled axis read as linear, and that
                            assumption is stated in `criterion` rather
                            than hidden.

Every curve carries its source document (author, title, publication code,
original release date) both as a Python comment and as a comment above
the profile literal in the generated .cpp.

GENERATED C++ now stamps, in film_profiles.hpp AND film_profiles.cpp:
generation timestamp (ISO-8601 UTC) and the schema version it was
generated from. film_names.txt deliberately carries NEITHER -- it stays
pure data.

PRECISION DECISION, asked and answered: the spectral tables are emitted
as std::vector<double> (exact float64 shortest-roundtrip literals), since
generation is offline and storage free. Everything pre-v3 stays float32
with exact shortest-roundtrip literals -- those values were authored as
short decimals, so float32 already reproduces them exactly; the true
precision bound everywhere is transcription accuracy (+/-0.05..0.1 log,
stated per stock), not literal rounding. std::vector rather than
std::array because grid length is per stock; hold instances const.

verify.py: 67 -> 70 checks (spectral pilots present; peak normalisation
and floor bounds; IR curve peaks at 750 nm with a dead mid-visible gap --
guards against a silently shifted grid).

NEXT, in order: batch digitisation of the remaining ~40-50 stocks with
published curves (Fuji AF3 sheets, Ilford/Harman, Kodak E/F/H-1, Konica);
class templates anchored to documented cut wavelengths for the Soviet
stocks (Gurlev/Chibisov print the limits); then the renderer-side
spectral upsampling path (Jakob-Hanika style RGB->spectrum) in the C++
port, which consumes these tables.


===========================================================================
MACHINE-TRACED H&D CURVES -- digitize_plot.py -- 2026-08-02
===========================================================================

Curve SHAPES now come from the printed plots, not only the printed
numbers. New tool digitize_plot.py renders a datasheet plot at 600 dpi,
finds the frame and gridlines, traces each curve by seeded ink-centroid
tracking, and least-squares fits the 6-parameter ToneCurve. Residuals are
quoted in the profile comments; they land at RMS 0.003-0.007 D -- the
printed line width, roughly ten times better than reading by eye.

First adoptions ([T1] curve shapes):
  KODAK_VISION3_250D_5207   three layers from H-1-5207, 1426 samples
                            each, full -8..+8 stop range. The sheet's
                            absolute Status M dmins are the orange mask
                            (0.15/0.57/0.84), so the stock moved to
                            dmin_ladder encoding.
  FUJI_NEOPAN_ACROS_100     Microfine 15-min curve from AF3-095E, 1092
                            samples: measured fog 0.122, straight-line
                            gamma 0.690, toe measured; shoulder beyond
                            the printed range stays estimated and is
                            flagged as such.

UPDATE, batch 5 (owner request): H&D curves machine-traced for the
whole VISION3 family -- 5203 / 5213 / 5219 join 5207 (fit RMS
0.002-0.011 D, 856-1467 samples per layer; all four now dmin_ladder,
their Status M dmins being the orange mask) -- and for EASTMAN
DOUBLE-X 5222 (D-96 6.5-min gamma-0.66 control curve; shoulder beyond
the plotted range stays estimated and flagged). Requested file
"eastman 500t 5296 exr - Kodak.pdf" was not found in KODAK/; 5296
keeps sibling-sheet data until it arrives.

UPDATE, batch 6 (Ferrania): the overlooked FERRANIA sheet "Curve
caratteristiche e sensibilita spettrali" was opened at last -- P30 H&D
machine-traced (gamma 1.25 measured, D-76 8 min as printed, 2195
samples, RMS 0.007 D) and the P30 wedge-spectrogram envelope adopted
[T2] (peak 610-630 nm, cut ~660). P33 and Orto curves are in the same
file, ready if those stocks are ever added. Spectral coverage: 49/89.

The remaining plots across the archive -- every H&D family, every
spectral sensitivity plot, MTF curves, spectral dye densities -- are
inventoried with priorities and binding method rules in
doc/DIGITIZATION_QUEUE.md. Where a stock has NO plot, printed table
values are used instead (owner rule), as already done for the Soviet
additions. All 70 verify checks pass; C++ regenerated in step.

UPDATE, same day, batches 2+3 (five parallel digitization agents):
spectral coverage now 35 OF 89 STOCKS -- every spectral plot in the
archive that matches a database stock is digitised (Agfa, Kodak incl.
Kodachrome 64 / Ektachromes / Double-X / EXR 500T, Fuji, Harman, all six
Konica colour stocks, Rollei, Fomapan 400, Polaroid 664/667). Verified
no-plot sheets: Vista 200, both Kentmeres. Only VISION3 500T 5219 waits
on a sheet not in the archive (H-1-5219; brochure only). Per-stock
status table: doc/FilmCurves.md (regenerated by gen_film_curves_md.py
after every pass).

UPDATE, same day, batch 4 (owner-supplied datasheets): 24 new files
added to PDF/PROFILES; spectral coverage now 48 OF 89 STOCKS. The
missing H-1-5219 arrived -- VISION3 is complete -- plus the full
VISION2 (5217/5218/5205), VISION1 (5274/5246/5279) and EXR
(5245/5248/5293) generations, Plus-X 5231, the 5247 TI0835 sheet
(post-1979 EI 125T generation, caveat recorded) and Fuji Eterna Vivid
500 (KB-0901E). Every Kodak motion-picture stock in the database now
carries its own digitised spectral curves. Honest findings: the
supplied 5239.pdf is a mislabeled VNF-1 processing manual (no 5239
spectral data exists on file); the July 2022 Kentmere Pan 100 sheet
prints no spectral plot.


================================================================================
SCHEMA v4 (2026-08-03) -- COATING, GATE AND LENS DEFECTS
================================================================================

Old footage darkens and wobbles toward the corners. The mechanism is NOT
mainly the emulsion, and the design had to correct that assumption first:

  Film is coated as a web up to 1.4 m wide and slit into strips afterwards.
  The coating machine has no idea where frame boundaries will fall -- the
  camera gate decides that later. So coating thickness variation CANNOT
  produce a defect locked to frame corners. Corner-locked darkening is the
  LENS (cos^4 theta), and it applies in EVERY era; modern glass still loses
  0.3-0.5 stop wide open.

Coating variation lives in WEB coordinates instead:

  ACROSS the web (frame's horizontal axis on 35 mm)
      Streaks at fixed x from fixed hopper hardware. Identical on every
      frame of the roll. Does not flicker.

  ALONG the web (frame's vertical axis)
      The film advances one frame pitch per frame, so each frame samples a
      different stretch of web. THIS is the one real emulsion-driven
      frame-to-frame blink: spatially smooth and sliding, never white noise.

FOUR EFFECTS ADDED

  1. Lens vignette      FilmProfile.default_vignette, in stops. Real
                        cos^4(theta) geometry, corner pinned to the requested
                        stops, centre exactly 1.0. Era default;
                        RenderSettings.vignette overrides.

  2. Coating field      CoatingSpec.coating_sigma plus two correlation
                        lengths. QC-driven, NOT date-driven: trough coating
                        (pre-1950s) worst, slide/extrusion hoppers (1950s)
                        better, multi-slot simultaneous (1970s+) better
                        again -- but Soviet, GDR and budget plants lagged by
                        decades. 1974 Eastman 5247 and present-day Fomapan
                        share a tier; 1990s Kodak sits two tiers better.

  3. Gate buckling      CoatingSpec.buckle_mtf_loss. The pressure plate holds
                        the frame centre flat while a curling base lifts the
                        corners out of the focal plane. Corner SOFTNESS, never
                        corner darkening -- these two get conflated constantly.

  4. Edge fog           CoatingSpec.edge_fog_density / _mm. GAUGE-driven, not
                        era-driven: Standard 8 is 16 mm slit down the middle
                        after processing, so its frame sits at the film edge
                        permanently. 35 mm margins carry the perforations and
                        get trimmed away.

NEW RENDER CONTROLS

  --vignette STOPS      lens corner falloff; omit for the stock's era default
  coating_scale         scales all three coating defects; 0.0 disables
  frame_index           frame number in the clip. Only the coating field uses
                        it. The field is a pure function of (seed, absolute
                        web position), so frames render independently and out
                        of order -- no state, no seams.

EMERGENT RESULT (not designed in, worth knowing)

  The same emulsion behaves differently by gauge with no extra parameters.
  8 mm advances only 0.45 correlation-lengths of web per frame, so its mottle
  DRIFTS SLOWLY (lag-1 field correlation +0.96). 35 mm advances 2.24, so it
  REFRESHES EACH FRAME (+0.47). And because a 4.8 mm frame is smaller than the
  coating structure, on 8 mm the variation appears as frame-to-frame
  BRIGHTNESS FLICKER rather than spatial mottle -- on 35 mm it is the reverse.

COST (measured, HD 1920x1080, worst-case stock, all four active)

  baseline, all v4 off ............ 0.689 s
  + vignette ...................... +5.6 %
  + coating field and edge fog .... +2.5 %
  + corner defocus ................ +11.1 %   <- only neighbourhood pass
  all four ........................ +19.2 %
  versus pre-v4 (old per-frame FFT mottle) ... +11.4 %
  modern stock (buckling only) ............... +10.7 %

  The pre-v4 coating path is GONE, not extended: it synthesised isotropic
  mottle with a full-resolution FFT pair on every frame -- wrong geometry
  (blobs, not streaks), wrong temporal behaviour (frozen across a sequence,
  seeded only from settings.seed), and about 25x the cost of the
  low-resolution synthesis that replaced it.

KNOWN LIMITS

  * coating_sigma delivers about 0.84x nominal through the low-resolution
    synthesis and bilinear reconstruction. Left uncorrected on purpose: the
    parameter is a tier-3 estimate, so a compensation factor would be false
    precision.
  * The coating field is applied equally to all three layers; real multilayer
    coating varies per layer.
  * buckle_mtf_loss blends a fixed-width kernel rather than scaling a true
    defocus PSF with distance from the focal plane.


================================================================================
SCHEMA v5 (2026-08-03) -- INTERIMAGE EFFECTS
================================================================================

Cross-layer development inhibition. Developing silver in one layer releases
inhibitor; it diffuses LATERALLY (edge effects -- already modelled as
CouplerSpec) and VERTICALLY into the neighbouring layers (this addition).

    logE_i' = logE_i + sum_{j != i} a_ij * (D_j - d_ref_j)

Off-diagonals negative (inhibition). Diagonal structurally zero -- a layer's
effect on itself is already inside its own curve. Solved by fixed-point
iteration; default 1 pass; iterations = 0 disables the stage.

WHY THE MID-GREY REFERENCE MATTERS
  On a neutral, every (D_j - d_ref) is about zero, the correction vanishes and
  the grey scale is untouched -- verified, max channel delta 0.00000. On a
  saturated colour the layers disagree, develop against unequal inhibition and
  separate further. Saturation rising WITHOUT gamma rising is exactly what a
  per-channel curve cannot do, and it is the mechanism behind Portra's skin
  separation and Velvia's saturation.

ACTIVE ON 48 OF 89 STOCKS. Excluded, with reasons: monochrome (one layer, no
neighbour to inhibit); Dufaycolor and Lumiere (single panchromatic emulsion
behind a filter grid); Technicolor three-strip (three physically separate
films cannot exchange inhibitor at all -- part of why its colour behaves
unlike a tripack's).

PROVENANCE -- TIER 3 THROUGHOUT
  All 395 documents in PDF/PROFILES were searched. NO manufacturer sheet
  publishes interimage data, and the omission is systematic: camera negative
  is characterised with one white-light exposure series, and the
  colour-separation series that would reveal these effects is only printed for
  print stocks. The one authoritative quantification found is a citation, not
  a document in hand -- Gschwind, Rosselet and Buser, "Investigation and
  quantification of inter-image effects", J. Photographic Science 41 (1993),
  p. 86.

  TO MEASURE: neutral step wedge, then the SAME wedge through red (W25),
  green (W58) and blue (W47B) filters, one roll, plus an empty-gate frame.
  That is the colour-separation series the sheets omit.

SPECTRAL DERIVATION -- BUILT, MEASURED, QUARANTINED
  derived_spectral_response() integrates the digitised spectral curves against
  display primaries. It is NOT wired into the renderer, because measurement
  showed two failures: display primaries stop near 630 nm, so on
  KONICA_INFRARED_750 (sensitised 640-820 nm, peak 750) it returns
  (0.022, 0.017, 0.960) -- blue-dominant, since the only part a monitor can
  reach is the intrinsic 400-500 nm lobe; and for colour stocks it derives
  near-identity matrices (Portra 0.97-0.99 diagonal), adding an assumption
  layer for no benefit. The real fix is a SCENE spectral model (reflectance
  basis under a stated illuminant), to be built deliberately. Until then
  spectral_weights and taking_matrix remain authoritative.

ADDITIVE ONLY, PROVED
  Field-by-field diff against the golden v3 copy: 0 of 93 stocks had any
  existing value changed, 0 print-stock changes. Only coating,
  default_vignette (v4) and interimage (v5) were appended.

COST
  One IIE pass costs one extra curve evaluation per channel; the curve stage
  is the most expensive in the chain, so budget about +1x stage 8 per pass.
  iterations is DATA, so the C++ port can trade accuracy for time per stock.


================================================================================
INTERIMAGE UPGRADED TIER 3 -> TIER 2 (2026-08-03, patent literature)
================================================================================

No manufacturer DATASHEET publishes interimage effects. PATENTS do, because a
patent claiming improved interimage effects has to demonstrate them. Nine were
surveyed; all are free, public documents.

THE METRIC, from US5273870A (Agfa-Gevaert), verbatim:
  "the percentage steepening of color gradation during color separation
  exposure with light of the corresponding spectral region in relation to the
  color gradation established on exposure with white light"
  -- citing T. H. James, The Theory of the Photographic Process, 4th ed.
  (1977), pp. 574 and 614. Measured at density 1.0 over fog.

THE NUMBERS (B/G/R), with a real DIR-free control:
  Ex.1  invention 25/45/42 %   DIR-free control 10/15/15 %
  Ex.2  invention 20/42/39 %   DIR-free control  8/15/12 %
  Ex.3  invention 25/33/35 %   DIR-free control  8/12/14 %
  Corroborated by US4830954A: yellow 5-15 %, magenta 8-35 %, cyan 10-30 %.
  The DIR-free control is NOT zero -- a film with no DIR couplers still shows
  10-15 % interimage from iodide released during development. That is the
  pre-DIR case, which is why the "mild" tier is not "none".

ASYMMETRY IS PER RECEIVER, NOT PER DISTANCE
  Blue receives weakly, green and red strongly, in every example found.
  US4725529A Table 1 proves this is emulsion chemistry rather than geometry:
  inhibitor in the DEVELOPER, three separate single-layer coatings, no layer
  stack at all -- red receivers still take 0.43-0.72 dlogE against blue
  0.24-0.48. Meanwhile not one of the nine patents differentiates adjacent
  from remote coupling numerically, and US3227554A shows designers ENGINEER
  remote coupling away with barrier layers rather than accepting geometric
  falloff. So no per-hop distance factor.

CONVERSION IS SOLVED NUMERICALLY, NOT BY FORMULA
  Coupled system: gamma_i = gamma0_i * (1 + SUM_j a_ij * gamma_j).
  Two lessons, both measured rather than assumed:
    1. the DONORS' gammas divide the target, not the receiver's own --
       solving channels independently overshot strong DIR by 23 points;
    2. the linear solution holds only while coupling is weak -- it matched
       the DIR-free control to 0.9 points and still overshot strong DIR by 23.
  So _IIE_TIERS stores the PATENT PERCENTAGES and _iie_solve() inverts the
  model to hit them, the same "fit through the full pipeline" method already
  used for grain RMS. Result: published figures reproduced to 0.05 PERCENTAGE
  POINTS on stocks of differing contrast.

REVERSAL MECHANISM SPLIT (US4729943A)
  Negatives get interimage "always ... during chromogenic development".
  Reversal gets it "by the release in the first black-and-white developer of a
  development inhibitor", landing in HIGH dye-density areas -- and pushing it
  harder LOWERS neutral speed. InterimageSpec.density_weighting carries this:
  0 negatives, 0.65 reversal, normalised at mid grey so neutrals stay exact.

STILL OPEN
  * the 0.65 reversal weighting is tier 3 (mechanism documented, magnitude not)
  * IIE should trade against grain and Dmax (US4729943A: DIR couplers in image
    layers "increase the granularity" and "reduce contrast and maximum
    density") -- not modelled
  * best unexhausted source: US7022468B2, the only document defining IIE(BG)
    and IIE(BR) separately in log-exposure form; tables truncated online
  * NOTE US3227554A is NOT the Barr/Thirtle/Vittum paper -- it is
    "mercaptan-forming couplers" (Barr, Williams, Whitmore) and has no
    interimage numbers. The paper is Photog. Sci. Eng. 13, 174 (1969).


================================================================================
ISO 5-3 / IT2.18 DENSITOMETRY TABLES -- COMPLETE (2026-08-03)
================================================================================

iso5_3_density.py holds all nine spectral-product tables.

  Table 2  (ISO 5-3:1995)          visual 570 nm, Type 1 400 nm, Type 2 430 nm
  Table 4  (ANSI/NAPM IT2.18-1996) Status M  blue 450, green 540, red 640 nm
  Table 3  (ANSI/NAPM IT2.18-1996) Status A  blue 440, green 530, red 620 nm

The ISO 5-3 copy available here is a standards.iteh.ai PREVIEW that stops
mid-sentence immediately after NAMING Table 4. The US national adoption,
ANSI/NAPM IT2.18-1996, carries both Status tables in full -- that is where they
came from. ISO3664 is viewing conditions and has no densitometry tables.

All tables transcribed from RENDERED PAGE IMAGES, not PDF text layers: the text
layers interleave the wavelength and value columns and would silently mis-pair
every row. European decimal comma handled ("4,957" is 4.957).

ONE SUBTLETY THAT WOULD HAVE BEEN A SILENT BUG
  Table 2 marks out-of-range entries "< 1,000" -- genuinely floor.
  Tables 3 and 4 DO NOT. They print a SLOPE and an arrow, meaning the response
  continues linearly in log10 past the last tabulated value:
      Status M  blue +0.250/-0.220   green +0.106/-0.120   red +0.260/-0.040
      Status A  blue +0.380/-0.140   green +0.220/-0.170   red +0.270/-0.040
  (per nm, below range / above range). Truncating them to zero would narrow
  every channel skirt and bias all derived densities. weights() applies the
  printed slopes and clamps at 1e-6 relative.

SELF-CHECKS (python3 iso5_3_density.py)
  * nine tables, 44 entries each, matching the wavelength grid
  * every table peaks at exactly 5.000, at the printed wavelength
  * non-selective sample -> density 0.000000 in every metric
  * uniform 10% transmitter -> exactly 1.000000 in every metric, including all
    three channels of both Status sets (the real test that slope extrapolation
    is balanced)
  * Status M red 640 nm > Status A red 620 nm, asserted in code

WHAT IT UNBLOCKS
  dye_matrix can now be DERIVED instead of estimated: integrate each stock's
  digitised spectral dye density curves against Status M (37 colour negative
  stocks) or Status A (16 reversal stocks), then compare against the existing
  hand-set matrices -- that comparison measures how wrong the estimates were.
  Still needed: the dye curves themselves (see DYE_DIGITISATION_STATUS.md).


================================================================================
SOURCE LIBRARY -- ARCHIVED FILES (2026-08-03)
================================================================================

Moved off the working drive to external storage, with everything needed already
extracted:

  THE THEORY OF THE PHOTOGRAPHIC PROCESS (Mees 1942)      356 MB, 1118 pp
      All findings preserved in doc/MEES_1942_EXTRACTION.md with page
      citations. 1st edition, so interimage/DIR are ABSENT (0 pages) -- DIR
      couplers postdate it by ~30 years; interimage came from patents instead.
      Yielded: Schwarzschild p~0.8 and the fact that p is NOT constant (our
      reciprocity model is one-sided and should be two-sided); Callier
      q <-> grain size via d = 6.8 log q; Eberhard effect 1.5-2x with a
      grain-size dependence we do not yet model, and the border/fringe sign
      split; turbidity as the basis of MTF.

  Cinematography - American Cinematographer Manual        117 MB, 300 pp
      IMAGE-ONLY SCAN -- pdftotext yields 300 bytes, every keyword search
      returns zero. Nothing extractable without rendering 300 pages. No loss.

KEPT ON THE WORKING DRIVE (do not archive):

  The Permanence and Care of Color Photographs (Wilhelm)   34 MB, 761 pp
      Only 34 MB, and it is the canonical source for AgingSpec, which is still
      all-zero tier-3 hooks: 92 hits on dye fade / fading rate / dark storage,
      58 on cyan/magenta/yellow dye, 17 on Dmin. Needed for dye_fade_c/m/y,
      base_yellowing_d and dmin_lift when the aging work starts.


================================================================================
AlgoControl.hpp -- REAL CONTROLS STRUCT (2026-08-03)
================================================================================

Replaces the "struct AlgoControls { int dummy; };" placeholder.

  AlgoControls
    21 live fields  -- mirror film_sim.RenderSettings ONE-FOR-ONE
    bool       filmDamageEnabled   hard gate, DEFAULT false
    FilmDamage damage              17 fields, specified but NOT yet consumed

FilmDamage is NESTED, not passed separately: one object to hand around, one
thing to serialise with a preset. The gate keeps the inert block visibly inert
and is checked ONCE PER FRAME, not per pixel. getFilmDamageDefault() is also
exposed alone, for a "reset this group" button.

WHY THE 21 MIRROR RenderSettings EXACTLY
  A C++ render with getAlgoControlsDefault() is directly comparable against the
  Python reference -- that comparability is what let Algo 02 be verified to
  1e-15. Verified mechanically: 21/21 defaults match. If either side changes,
  re-run the check or the reference stops being a reference.

NOT IN THE CONTROLS, ON PURPOSE
  Film properties (FilmProfile data, 93 stocks). A control never replaces a
  profile number; it scales or overrides it.
  Stock-coupled damage: dye fade (per dye set), base yellowing and shrinkage
  (per base material), scratch COLOUR (depth decides which dye layers survive,
  so it needs the tripack), blob polarity (white on a print, dark on a
  negative, inverted on reversal). Those are AgingSpec / CoatingSpec.
  Only emulsion-INDEPENDENT damage is a control.

TWO CONVENTIONS
  Sentinels: flare and vignette default to -1.0 = "use the stock's
  era-appropriate value". 0.0 means "genuinely none". Losing that distinction
  silently discards per-era lens data.
  Damage rates are per SECOND, not per frame, so defect density stays constant
  when fps changes. weaveAmpXUm/YUm default to 0.0 = defer to the stock's
  TemporalSpec (already populated on all 93 stocks).

STATELESSNESS (required of every damage generator)
  Pure function of (damageSeed, frameIndex, stageId, ordinal) via a
  counter-based RNG, with a bounded birth-frame scan for persistent objects.
  Any frame renderable alone, out of order, on any thread -- same rule the v4
  coating field follows, and why frameIndex is a control rather than internal
  state. Set it from layer time x fps, NOT a running counter.

VERIFIED: 27 checks pass; clean under g++ -std=c++14 -Wall -Wextra -pedantic.
Details: doc/ALGOCONTROL_NOTES.md


================================================================================
2026-08-04 -- 89 TO 93 STOCKS; LISTBOX NAME LIST AND INDEX ENUM
================================================================================

Four stocks added, all from primary documents, schema unchanged at v5:

  AGFACOLOR_NEG_TYPE_B_1943  Schmidt & Kochs, Farbfilmtechnik, Berlin: Hesse
                             1943, Abb. 57-59 (via AGFA/Agfacolor 01.mhtml)
  FUJICOLOR_A250             Fuji Data Sheet MP3-57E, 1980.08
  GEVACHROME_902             Verbrugghe, SMPTE 1967 (colour reversal)
  GEVACOLOR_NEG_682          Vervoort & Stappaerts, SMPTE 1980

A normalisation bug was caught by validate_all() while adding them. All four
sources plot a LOGARITHMIC ordinate, so peak-normalising means SUBTRACTING the
layer peak; an earlier pass took log10(v / peak) on two of them, which badly
compressed the curves. Recomputed by subtraction, peak wavelengths re-checked
against the printed figures. A second slip -- A250's green and red arrays sitting
one grid position early -- was caught the same way.

TWO NEW GENERATED FILES
-----------------------
cpp_codegen.py now writes, after film_profiles.cpp and .hpp:

  film_names.txt   one display name per line, double-quoted, spaces instead of
                   underscores, LF endings, pure ASCII, no comments, no
                   separators. Feeds the effect-panel listbox directly.

  film_enum.hpp    enum class eFILM_PROFILE : int32_t, values from 0, ending
                   with eTOTAL_FILMS_PROFILES = 93. Carries generation
                   timestamp, schema version and profile count in its comment
                   header.

Both come from parse_vector_names(), which reads the ALREADY-WRITTEN .cpp back
rather than re-walking FILM_PROFILES. The listbox indexes into the std::vector
returned by GetFilmDatabase(), so line N of the TXT and enumerator value N-1
must BE element N-1 of that vector. Reading the emitted table makes that true by
construction; re-deriving it from FILM_PROFILES would only be assumed true.

test_film_enum.cpp (C++14) cross-checks all three artefacts against each other
and passes for all 93 profiles.

*** COMPATIBILITY WARNING ***
The four new stocks are INSERTS, not appends, because FILM_PROFILES is sorted
alphabetically. AGFACOLOR_NEG_TYPE_B_1943 took index 0, so EVERY enumerator
value shifted in this release. Any saved project or serialised plugin parameter
holding a numeric film index from an earlier build will now select the wrong
stock. Remap by name, not by number. Future additions should ideally be
appended if index stability matters more than alphabetical order.

DUFAYCOLOR
----------
The Timeline .mhtml was extracted and corroborates the existing profile: the
reseau is quoted at 19-25 lines/mm and the shipped value of 20.0 already sits
inside that range. No value changed; only the citation was added. The page
carries no gamma, ASA or H&D data.

TEST SUITE
----------
verify.py now accepts VERIFY_SLICE=<from>-<to> so it can be run in parts; the
full suite is render-heavy and cannot finish inside a short per-process
wall-clock budget. Across all slices: 103 PASS, 1 FAIL.

Corrected stale assertions: stock count 89 -> 93; reversal count 21 -> 22
(Gevachrome 902 is a reversal); "schema version is 4" -> "is at least 4" (the
schema has been v5 since the interimage pass -- that assertion had been wrong
ever since, and was only never seen because the suite never reached section 18
before being killed).

Left OPEN, needs a decision -- interimage is layer-distance-blind:
  FAIL  neighbour pairs couple harder than the far red-blue pair
For KODAK_PORTRA_400, a_rg and a_rb are identical (-0.257), because _IIE_TIERS
stores strength per RECEIVING layer and applies it to both donors. Physically
red and blue are not adjacent -- they are separated by the green layer and the
yellow filter layer -- so they should couple less than red-green. The test is
right; the model is incomplete. Fixing it means splitting each per-target
percentage between its donors, which moves every colour stock away from the
patent percentages the v5 pass was calibrated to.

Left OPEN, pre-existing and unrelated:
  FAIL  saturation hierarchy is ordered clean -> impure dyes
The chain breaks at technicolor 0.179 > 5219 0.196. None of the new stocks
appear in it and the suite had never run far enough to reach that section
before, so it cannot be attributed to this pass.


--------------------------------------------------------------------------------
2026-08-04 FOLLOW-UP -- TWO CITATION-INTEGRITY CORRECTIONS
--------------------------------------------------------------------------------

Both found because the owner asked why AGFACOLOR NEU 1936 had no reference
document, and what about the mhtml files. Neither would have surfaced otherwise.

1. AGFACOLOR_NEU_1936 NOW CITES ITS SOURCES, BUT STAYS TIER 3

It had been on the _NO_DATASHEET placeholder with fitted_from="analogy". It now
carries two real citations:

  Color Committee (1937): "The New Agfacolor Process". JSMPE, May 1937,
    pp. 561-562.
  Hatschek, Paul (1936): "Der neue deutsche Agfa-Farbenfilm". Die Kinotechnik
    18(21), 5 Nov. 1936, pp. 345-346.

Tier stays 3, fitted_from stays "analogy", and NO numeric value changed. Those
documents establish the process and date only -- subtractive three-colour
chromogenic monopack, reversal from 1936, colour formers incorporated in the
superposed layers instead of added to the developer, silver later dissolved out
leaving pure dye images. Neither carries a photometric figure.

A PROVENANCE LIMIT note in the profile spells out the trap: the SAME page
carries quantitative Agfacolor data (15/10 Din, the Type B vs Type G red
sensitivity trade, the Abb. 58-59 layer curves) and ALL of it belongs to the
1939+ negative/positive system -- a different film, different process. Putting
it on a 1936 reversal monopack would be exactly the error class the movie-stock
verification pass exists to catch.

For the record, what these files are: "Agfacolor 01.mhtml" is not a book scan.
It is the Timeline of Historical Colors in Photography and Film page titled
"Agfacolor Neu / Agfacolor" -- a family page with 100 embedded JPEGs, long
quoted passages from Schmidt & Kochs, and a bibliography. "Agfacolor 02.mhtml"
and "03.mhtml" are BYTE-IDENTICAL duplicates of each other
(md5 29a4e300c897cc7d1caa3ba10c57f5be); one can be deleted.

2. THE AGFACOLOR TYPE B SPECTRAL CURVES WERE WRONG -- CORRECTED

Yesterday's entry claimed peaks of "2.28 / 0.99 / 0.67 in density units" with
the red maximum at 625 nm. Extracting the figure image from the mhtml
(Schmidt_Farbfilmtechnik_1943-59-700.jpg, Abb. 59a panel I) and looking at it
showed three faults:

  blue   figure: peak ~2.5, broad 440-480 nm plateau
         encoded: 2.28, immediate falloff past 450
  green  figure: peak ~1.25 at ~555 nm
         encoded: 0.99
  red    figure: peak ~0.55 at ~655 nm, FLAT BASELINE below 600 nm
         encoded: 0.67, peak at 625 nm, non-zero at 575 nm

The red error is the serious one: ~30 nm shift plus phantom sensitivity at
575 nm. Peak wavelengths now validate at B 450 / G 550 / R 650 nm.

Two method points worth keeping:

  - Baseline maps to the -4.0 out-of-band sentinel, NOT to -(peak). On a wedge
    spectrogram a zero-density trace means "below threshold", not a finite
    sensitivity of 10^-1.2. Subtracting the peak everywhere would have invented
    sensitivity in the dead regions.
  - The caption says "schematisch" TWICE. The description now records this and
    forbids restoring two-decimal peak values or trusting wavelengths to better
    than about +/-10 nm.

Root cause, plainly: the numbers had been inferred from the surrounding German
prose, which says only that the layers "overlap widely". Prose cannot yield peak
values, so they should never have been written down as if it had.

STILL UNMINED IN THE SAME IMAGE: Abb. 59a II (negative dye transmittances) and
Abb. 59b III/IV (positive-film layer sensitivities and dyes). Panel III shows
the sharply selective sensitisation of the PRINT stock -- primary-source print
data we do not currently have. Deferred by decision, not oversight.

3. TIMESTAMP FORMAT

Generated banners now read "2026-08-04  06:35:10Z" -- two spaces instead of the
ISO-8601 "T", for legibility. Still UTC, still Zulu-suffixed, no tabs.
film_names.txt stays stamp-free and comment-free.

4. SUITE AFTER THE CORRECTIONS

  slice 1-8     27 PASS  0 FAIL
  slice 9-14    25 PASS  0 FAIL
  slice 15-19   52 PASS  2 FAIL
  total        104 PASS  2 FAIL

Both failures are the ones documented above (interimage layer-distance blindness
and the pre-existing saturation-hierarchy ordering) and are unchanged by this
pass. The "agfacolor" entry in the saturation chain is AGFACOLOR_NEU_1936, whose
values were deliberately not touched.

LESSON RECORDED TWICE THIS SESSION: a single-line grep returning zero is not
evidence of absence. "DIN=0" was really "Din", and "CORRECTION=0" was a string
split across two source lines. Both nearly became false conclusions.
