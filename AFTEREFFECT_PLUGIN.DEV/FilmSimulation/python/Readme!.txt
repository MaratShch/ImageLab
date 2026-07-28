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

It supports 26 film stocks, covering colour negative, colour reversal, black and
white, three-strip Technicolor, 1930s-40s period stocks, and additive colour via
a physical filter grid (Dufaycolor). It also models the taking lens's veiling
flare and multi-generation dupe printing, which is what makes the period stocks
actually look period rather than merely soft.

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

  a) Check it works. Prints a table of all 26 stocks:

        python film_profiles.py

  b) Run the test suite. Should end with "ALL CHECKS PASSED" (67 checks):

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

  verify.py            67-check test suite. Run it after ANY edit to either file
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

Generated output, safe to delete and regenerate:

  film_profiles.hpp    C++ header. Contains the struct definitions AND the
                       reference formulae as comments, so a C++ port cannot
                       silently drift from the Python original.
  film_profiles.cpp    C++ tables: all 26 stocks, 4 print stocks, 11 gauges.
                       Regenerate both with:  python cpp_codegen.py -o .

  test_chart.png       The small synthetic test image.
  period_chart.png     The large 3200 px test image.
  contact_sheet.png    All 26 stocks on the small chart, side by side. Open this
                       first to see what the stocks look like.
  period_sheet.png     The five period stocks, plus a 3-generation dupe
                       comparison and a modern stock for reference.
  dufay_crop.png       Dufaycolor at 1:1 magnification, so you can actually see
                       the filter grid and the pastel colour it produces.
  sheet.py             Rebuilds contact_sheet.png. Not part of the test suite.

Documentation:

  Readme!.txt          This file. Practical operation.
  README.md            The technical write-up: what was wrong with the original
                       script, the full pipeline table, the physics reasoning,
                       every bug the test suite caught, and the honest limits.
                       Read it when you want the "why".

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

    auto stocks  = film::GetFilmDatabase();    // all 26
    auto prints  = film::GetPrintStocks();     // all 4
    auto formats = film::GetFilmFormats();     // all 11 gauges

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
