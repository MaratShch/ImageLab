// ---------------------------------------------------------------------------
//  AlgoControl.cpp
//
//  Default control values.
//
//  These are not arbitrary starting points. They mirror film_sim.RenderSettings
//  field for field, so a C++ render with getAlgoControlsDefault() is directly
//  comparable to a Python render with default settings. That comparability is the
//  whole basis of the verification harness: if these drift from the reference, the
//  two implementations stop being measurable against each other and every
//  discrepancy becomes ambiguous.
//
//  Anything changed here must be changed in film_sim.py's RenderSettings too, and
//  the reference re-run.
// ---------------------------------------------------------------------------

#include "AlgoControl.hpp"


// ---------------------------------------------------------------------------
//  getFilmDamageDefault
//
//  A COMPLETE, WORKING DAMAGE SET, and the master flag is now true as well, so a
//  default render shows damage rather than clean film.
//
//  THE SCALE: 1.0 = THE MEASURED CENTRAL FIGURE
//
//  Every level is anchored, not aesthetic. 1.0 means "as much of this class as was
//  measured on real film", so dustLevel 1.0 places the measured areal density of
//  embedded particles and nothing more. It is not a maximum - the controls run
//  above 1.0 for material in worse condition than the reference scans - and it is
//  not a slider position that happened to look right.
//
//  This set describes AMATEUR-PROCESSED, MODERATELY HANDLED film: developed outside
//  a professional laboratory, run through a projector or scanner a few times,
//  stored in ordinary conditions. The classes below 1.0 are the ones whose measured
//  central figure implies worse treatment than that.
//
//  WHAT THESE LEVELS ARE NOT
//
//  They are amounts, not descriptions of appearance. Sizes, size distributions,
//  contrast amplitudes, orientation statistics, spectral slopes, opacity
//  distributions and temporal population shares are all measured properties of
//  dirt rather than user choices, and they live as named constants in the defect
//  stage headers. Nothing here can make an individual speck look a particular way,
//  which is the intended design: loading a roll of film does not let you choose
//  where the dirt lands.
//
//  THE ERA BASELINE IS NOT AVAILABLE YET, SO THESE ARE ABSOLUTE
//
//  The intent was that each level would SCALE the era-typical figure the profile
//  carries in AgingSpec, so a 1943 Agfacolor at dustLevel 1.0 would be dirtier
//  than a modern stock at the same setting with nobody authoring two presets.
//
//  That cannot work today: all 142 stocks in the generated database ship AgingSpec
//  entirely zero, documented as "fresh". Multiplying by dust_area_ppm would
//  silence the defect layer on every single stock and look exactly like a broken
//  control. So the levels are absolute for now - dustLevel 1.0 means the measured
//  density whatever the stock - and when AgingSpec is populated it should enter as
//  an ADDITIVE era term rather than a multiplier, so that fresh stock keeps
//  behaving as it does today and only aged stock gains dirt nobody asked for.
// ---------------------------------------------------------------------------
FilmDamage getFilmDamageDefault (void) noexcept
{
    FilmDamage damage{};

    // ----------------------------------------------------------------------
    //  MASTER
    // ----------------------------------------------------------------------

    // Global severity. 1.0 means "the class levels below, as stated". It scales
    // everything, so it is the one control to reach for when the whole look is
    // right but too strong or too weak.
    damage.damageStrength    = 1.0;

    // Roll seed, deliberately independent of AlgoControls::seed so that
    // re-rolling the grain does not also re-roll the dirt.
    damage.damageSeed        = 20250803;

    // ----------------------------------------------------------------------
    //  A COMPLETE WORKING DAMAGE SET: "amateur-processed, moderately handled".
    //
    //  These are not a demo and not decoration. Each one is the level at which
    //  that class reproduces the amount of damage measured on real film of this
    //  kind - film developed outside a professional laboratory, projected or
    //  scanned a handful of times, and stored in ordinary conditions. It is the
    //  condition the great majority of surviving material is actually in, which is
    //  why it is the useful starting point rather than pristine or ruined.
    //
    //  1.0 always means "the measured central figure for this class". So a level
    //  of 1.0 is not maximum and not arbitrary: dustLevel 1.0 puts the measured
    //  areal density of embedded particles on the negative, no more. Levels below
    //  1.0 below are classes whose measured central figure is higher than this
    //  grade of film warrants.
    //
    //  WHICH OF THESE DO SOMETHING TODAY
    //
    //  !! RE-AUDITED 2026-08-28. This section said "three ... the other eleven"
    //  and had gone stale: three more classes went live and nobody came back
    //  here. The count is now NINE live and EIGHT inert, verified by a
    //  tree-wide grep of all seventeen identifiers across both instruction-set
    //  trees.
    //
    //  LIVE (9):
    //     damageStrength, damageSeed          the master pair, all three stages
    //     dustLevel, debrisLevel, fibreLevel  stage 9b, the particulate classes
    //     dirtClumping                        stage 9b, their spatial process
    //     weaveAmount                         stage 15
    //     gateDirt, damageEvents              stage 16
    //
    //  INERT (8) - no reader anywhere in the engine:
    //     scratchTransport, scratchHandling, processingQuality, dryingMarks,
    //     storageSeverity, colourVeil, flickerStops, scannerArtifacts
    //
    //  flickerStops is the one worth calling out: its intended consumer, stage
    //  3c, is a genuine pass-through that voids all five of its arguments, so
    //  that control has nowhere to act even in principle today.
    //
    //  The inert eight are populated anyway, and deliberately, for two reasons:
    //  the value is the correct one for this grade of film, so when each stage
    //  lands it is immediately right rather than needing a second pass over
    //  this file; and a zero here would be indistinguishable from a considered
    //  decision that this grade of film has no scratches, which is false.
    //
    //  So a render with these defaults shows dust, debris, fibres, gate dirt,
    //  gate events and weave, and nothing else. That is the honest current
    //  state of the pipeline rather than a fault in these numbers.
    // ----------------------------------------------------------------------

    // ---- Particulate: LIVE, rendered by stage 9b --------------------------

    // Fine dust at the measured central density. The dominant class and the one
    // that carries the look; everything else is detail on top of it.
    //
    // Stage 9b renders only the EMBEDDED share of this, which is the part actually
    // baked into the negative. The loose one-frame population and the gate
    // population are machine-side and belong to stage 16, so until that exists a
    // level of 1.0 puts roughly six tenths of the measured total on the frame.
    damage.dustLevel         = 1.0;

    // Coarse debris at its measured rate - a couple of conspicuous opaque blobs
    // per frame of this size. Rare enough that a lower level would simply mean
    // most frames have none, which is not what was measured.
    damage.debrisLevel       = 1.0;

    // Fibres at their measured rate, which is well under one per frame. On any
    // given frame this usually renders nothing; across a sequence it produces the
    // occasional hair. That is the correct behaviour and not a level worth
    // raising to "see it work".
    damage.fibreLevel        = 1.0;

    // ---- Scratches: INERT, stage not yet written --------------------------

    // Transport scratches - the continuous longitudinal grooves a fixed burr
    // ploughs along moving film. Below 1.0 because a full measured rate implies a
    // damaged transport, not ordinary handling.
    damage.scratchTransport  = 0.40;

    // Random handling scratches and abrasions, from film rubbing against itself
    // and against surfaces. Slightly rarer than transport damage on material that
    // has not been through a bad projector.
    damage.scratchHandling   = 0.30;

    // ---- Processing and drying: INERT ------------------------------------

    // Development mottle from uneven agitation. Low, but deliberately non-zero:
    // hand-processed film essentially always carries some, and it is one of the
    // strongest cues that a frame was not developed in a machine.
    damage.processingQuality = 0.30;

    // Water spots, tide lines and squeegee marks from drying. Same source event as
    // much of the embedded dust, which is why the two levels sit close together.
    damage.dryingMarks       = 0.25;

    // ---- Age and storage: INERT ------------------------------------------

    // Dye fade, crossover and age fog. Kept low, because this is the one group
    // that says how OLD the film is rather than how it was treated, and the
    // profile's own era should drive most of it once AgingSpec is populated.
    damage.storageSeverity   = 0.20;

    // Overall colour veil. Lower still: a heavy veil reads as a grade rather than
    // as damage, and it is the fastest of all these controls to overdo.
    damage.colourVeil        = 0.15;

    // ---- Machine side: INERT ---------------------------------------------

    // Gate dirt, and with it the one-frame sparkle population. The highest of the
    // inert levels on purpose - this class holds the loose and gate share of the
    // dust, which is the larger part of the total, and it is what makes dirt read
    // as MOTION rather than as a still overlay.
    damage.gateDirt          = 0.60;

    // Gate weave and registration instability. Mid-range, matching a serviceable
    // but not precision transport.
    damage.weaveAmount       = 0.50;

    // Discrete events: splices, torn perforations, edge damage. Sparse by nature;
    // this is a rate, and a frame containing one should be an event.
    damage.damageEvents      = 0.20;

    // Exposure flicker between frames. Low, because a modern shutter is steady and
    // audible flicker belongs to the silent era, where the profile's TemporalSpec
    // supplies the era figure this multiplies.
    damage.flickerStops      = 0.15;

    // ---- Scanner layer: INERT -------------------------------------------

    // Shading non-uniformity, banding, fixed-pattern noise, Newton's rings. Low:
    // these are artifacts of the digitisation and not of the film, so they should
    // be perceptible only when looked for.
    damage.scannerArtifacts  = 0.20;

    // ----------------------------------------------------------------------
    //  THE ONE EXCEPTION, AND WHY IT IS NOT A LEVEL
    //
    //  Clumpiness is a MODIFIER on how the particulate classes are distributed,
    //  not an amount of anything. It does nothing at all while the three levels
    //  above are zero.
    //
    //  It defaults to 1.0 -- the measured behaviour of real film, a coefficient
    //  of variation of 0.88 in the local particle rate -- rather than to 0,
    //  because 0 means a uniform Poisson scatter. Uniform scatter is the single
    //  most recognisable failure of existing film-emulation products: real dirt
    //  arrives in patches, with some regions of a frame carrying five times the
    //  average and others almost none. Defaulting this to zero would make the
    //  first thing anyone sees when they enable dust the wrong thing.
    // ----------------------------------------------------------------------
    damage.dirtClumping      = 1.0;

    return damage;
}


// ---------------------------------------------------------------------------
//  getAlgoControlsDefault
//
//  Mirrors film_sim.RenderSettings exactly. The comments give the reference field
//  each value corresponds to, so the two can be diffed by eye.
// ---------------------------------------------------------------------------
AlgoControls getAlgoControlsDefault (void) noexcept
{
    AlgoControls controls{};

    // ----------------------------------------------------------------------
    //  Stock, gauge and the printing chain
    // ----------------------------------------------------------------------

    // Index zero, which is alphabetically first in the database. There is no
    // "default film" in any meaningful sense, so the first entry is as good a
    // starting point as any - and the caller is expected to set this.
    controls.filmProfile = static_cast<film::eFILM_PROFILE>(0);

    // Frames per second OF FILM. 24 is the sound-film standard and the rate every
    // defect figure in the damage group would be tuned against.
    controls.frameRate   = 24.0;

    // film_sim: film_format = "super35"
    controls.filmFormat  = "super35";

    // film_sim: print_stock = ""  -- empty means the stock's own default_print.
    controls.printStock  = "";

    // film_sim: dupe_stock = "DUPE_FINE_GRAIN"
    controls.dupeStock   = "DUPE_FINE_GRAIN";

    // film_sim: generations = 0 -- camera negative straight to print.
    controls.generations = 0;

    // ----------------------------------------------------------------------
    //  No output-buffer defaults, deliberately.
    //
    //  The reference model film_sim.py carries bit_depth = 16 and max_dim = 0,
    //  and both used to be mirrored here. Neither is set any longer, because
    //  neither exists in AlgoControls any more.
    //
    //  In the reference those two are properties of a COMMAND-LINE TOOL that
    //  writes a PNG file: it owns its output, so it may choose the depth it
    //  encodes and may downscale before encoding. This engine owns nothing. It
    //  is handed planar buffers by a host and must hand back the same shape, so
    //  a depth or an extent chosen HERE could only contradict the buffers the
    //  host actually supplied.
    //
    //  The equivalent of bit_depth is the host's own repack, which is symmetric
    //  with its unpack by construction. The equivalent of max_dim is passing
    //  smaller sizeX and sizeY to Algorithm_Main.
    //
    //  This is therefore one of the few places where the C++ engine intentionally
    //  does NOT mirror a field of the reference model, and the reason is that the
    //  field was never algorithmic.
    // ----------------------------------------------------------------------

    // ----------------------------------------------------------------------
    //  Exposure and colour
    // ----------------------------------------------------------------------

    // film_sim: exposure_stops = 0.0
    controls.exposureStops = 0.0;

    // film_sim: scene_kelvin = 5500.0 -- nominal daylight.
    controls.sceneKelvin   = 5500.0;

    // film_sim: wb_strength = 0.0 -- no correction, so a tungsten stock shot in
    // daylight goes blue, which is the correct answer rather than an error.
    controls.wbStrength    = 0.0;

    // film_sim: grey_target = 0.18 -- display-linear value an 18 per cent scene
    // grey must reach. This is what both anchor solves aim at.
    controls.greyTarget    = 0.18;

    // ----------------------------------------------------------------------
    //  Effect scales. 1.0 means "as the stock specifies".
    // ----------------------------------------------------------------------

    controls.grainScale    = 1.0;   // film_sim: grain_scale
    controls.halationScale = 1.0;   // film_sim: halation_scale
    controls.couplerScale  = 1.0;   // film_sim: coupler_scale
    controls.misregScale   = 1.0;   // film_sim: misreg_scale
    controls.coatingScale  = 1.0;   // film_sim: coating_scale

    // ----------------------------------------------------------------------
    //  The two "auto" controls.
    //
    //  A NEGATIVE value means "use the stock's own era figure" rather than "off".
    //  That distinction matters: zero would disable the effect, which for flare and
    //  vignette would silently give every period stock modern blacks and a
    //  perfectly even field.
    // ----------------------------------------------------------------------

    controls.flare    = -1.0;   // film_sim: flare = -1.0
    controls.vignette = -1.0;   // film_sim: vignette = -1.0

    // ----------------------------------------------------------------------
    //  Booleans
    // ----------------------------------------------------------------------

    // film_sim: print_grain = True. Print grain lands after the print curve, so
    // unlike negative grain it is not compressed by the shoulder.
    controls.printGrain = true;

    // film_sim: reseau = True -- allow the additive colour grid where a stock has
    // one and the render resolves it.
    controls.reseau     = true;

    // ----------------------------------------------------------------------
    //  Time base and determinism
    // ----------------------------------------------------------------------

    // film_sim: frame_index = 0. CLIP relative, so damage stays glued to the film
    // rather than to the timeline position. May be negative.
    controls.frameIndex = 0;

    // film_sim: seed = 12345. Every random quantity in the engine is a pure
    // function of this and the frame index, so a render is reproducible and the
    // host may render out of order.
    controls.seed       = 12345;

    // ----------------------------------------------------------------------
    //  Damage: ON, with the working set from getFilmDamageDefault().
    //
    //  This default is now DAMAGED FILM, not clean film. A default render shows
    //  embedded dust, coarse debris and the occasional fibre, because those are the
    //  three classes stage 9b consumes.
    //
    //  Two consequences worth being explicit about, because both are easy to be
    //  caught by:
    //
    //  A caller that wants the pure film-stock simulation with no defect layer must
    //  now clear this flag. It used to get that for free. One line -
    //  filmDamageEnabled = false - and the engine is numerically identical to the
    //  clean build, verified: with the flag clear the whole-chain agreement against
    //  the reference model is unchanged to the digit.
    //
    //  And this is a deliberate divergence from film_sim.RenderSettings, which
    //  carries no damage group at all. Every other field in this function mirrors
    //  the reference exactly so the two implementations stay diffable; this one
    //  cannot, because the reference has nothing to mirror. The verification
    //  harness therefore clears the flag before comparing, and must keep doing so.
    // ----------------------------------------------------------------------
    controls.filmDamageEnabled = true;
    controls.damage            = getFilmDamageDefault();

    return controls;
}
