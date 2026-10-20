// ---------------------------------------------------------------------------
//  Algo_09_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: the coupler's pointwise terms and the non-negative floor.
//  2026-10-10: the processingQuality / dryingMarks / colourVeil classes come from the
//  SHARED AlgoProcessingDefects.hpp, HighPrecType in both trees, so they are scalar
//  here too and agree with the reference to the float store.
//  The 9b particulate rasterisers are left scalar on purpose - they scatter into tiny
//  bounding boxes, a 20 um speck being one to three pixels, so there is nothing to
//  fill a register with, and together they are under two per cent of the frame.
//
//  ⚠ THE SCRATCH CLASS IS SCALAR TOO, AND ITS SOURCE IS CHARACTER-FOR-CHARACTER
//  THE SCALAR TREE'S. A scratch is a stroked polyline a pixel or two wide, so it
//  is a thin diagonal band of work with nothing to pack either - and the shared
//  defectRasteriseStroke (2026-10-20; was defectRasteriseFibre) it draws through
//  is HighPrecType in BOTH trees, so the
//  mark's geometry, coverage and density increment are computed in double on
//  both sides and only the STORE into the AlgoType plane rounds. Measured on the
//  test_scratch_guard probe: the two builds agree to 3e-08 on the scratch cases,
//  which is float32 storage rounding and nothing else. The one place the two
//  flavours genuinely differ inside 9b is defectRasterise, the compact-blob path
//  that dust and debris use, whose inner loop is float32 here by rule D1.
//
//  2026-10-20 (stochastic defects, reader optics): the blob path's interior
//  TEXTURE and every write that is not the plain additive one (removals, and all
//  writes with defectOptics on) are HighPrecType through the shared
//  AlgoDefectOptics.hpp, and the texture's lattice coordinates are formed in
//  double here too - near the opacity cap a float coordinate moved the density
//  by 2e-2 D. Measured on test_scratch_guard's dump mode, all classes on, OFF,
//  ON, and ON on a Callier stock: scalar against AVX2 within 4.5e-05 D (the
//  pre-change blob path: 4.1e-05).
//
//  ALIGNMENT: EVERY IMAGE ACCESS IS UNALIGNED, DELIBERATELY.
//
//  loadu/storeu on all plane data. The arena base comes from the host's memory pool,
//  whose alignment argument is a HINT, not a guarantee - it was observed returning a
//  base 16 mod 32, which faults an aligned 256-bit load. AlgoMemHandler.cpp is SHARED
//  by both flavours and must not be changed to suit the vector path, so the vector
//  path carries no alignment assumption instead. Costs nothing measurable on Haswell
//  and later.
//
//  Pipeline stage 9 and its sub-stage 9b, in the density domain:
//
//      AlgoStage09_DirCoupler        lateral inhibitor diffusion, two scales
//      AlgoStage09b_NegativeDefects  dust, debris, fibres and scratches
//      AlgoStage09c_BromideDrag      the machine's directional restraint
//
//  Both belong to the same numbered pipeline stage and share this translation
//  unit. Raw pointers, explicit geometry, no allocation, no mutable state, no
//  validation of inputs.
// ---------------------------------------------------------------------------

// Common.hpp -- AVX2_ALIGN / CACHE_ALIGN are defined here. Included
// DIRECTLY rather than relied on transitively: this file declares an
// aligned buffer, so the macro must not depend on another header's
// include order to be in scope.
#include "Common.hpp"
#include "AlgoDirCoupler.hpp"

#include "FastAriphmeticsAVX.hpp"
#include <immintrin.h>
#include "AlgoNegativeDefects.hpp"
#include "AlgoDefectOptics.hpp"        // 2026-10-20: reader law, Callier exclusion, noise
#include "AlgoBromideDrag.hpp"
#include "AlgoProcessingDefects.hpp"   // 2026-10-10: processingQuality, dryingMarks, colourVeil


static_assert(sizeof(AlgoType) == 4,
              "the AVX2 path requires AlgoType to be a 32-bit float");

namespace
{
    // ----------------------------------------------------------------------
    //  Lanes per vector, and the tail mask for the final partial vector of a row.
    //
    //  The active width is not generally a multiple of eight. Masked access leaves the
    //  row padding untouched, which keeps the NaN-poison arena test meaningful.
    // ----------------------------------------------------------------------
    constexpr int32_t ALGO_AVX2_LANES_LOCAL = 8;

    inline __m256i algoTailMaskLocal (const int32_t n) noexcept
    {
        AVX2_ALIGN static const int32_t table[8][8] =
        {
            { 0,  0,  0,  0,  0,  0,  0,  0},
            {-1,  0,  0,  0,  0,  0,  0,  0},
            {-1, -1,  0,  0,  0,  0,  0,  0},
            {-1, -1, -1,  0,  0,  0,  0,  0},
            {-1, -1, -1, -1,  0,  0,  0,  0},
            {-1, -1, -1, -1, -1,  0,  0,  0},
            {-1, -1, -1, -1, -1, -1,  0,  0},
            {-1, -1, -1, -1, -1, -1, -1,  0}
        };
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(&table[n & 7][0]));
    }
}

#include <cmath>   // std::sqrt, std::exp, std::log10, std::sin, std::cos, std::floor


// ===========================================================================
//  Particulate generator internals.
//
//  All file-local. Nothing here is exposed, because none of it is meaningful
//  outside the three classes it serves.
// ===========================================================================
namespace
{
    // ALGO_DEFECT_TWO_PI, ALGO_DEFECT_UM_PER_MM: film_damage_constants.hpp (2026-10-20c).


    // ----------------------------------------------------------------------
    //  One particle, fully specified before anything is rasterised.
    //
    //  Built in film coordinates and converted to pixels once. Separating the
    //  draw from the rasterisation is what keeps the two testable apart: the
    //  statistics can be measured from the parameters without touching a buffer.
    // ----------------------------------------------------------------------
    struct DefectParticle
    {
        HighPrecType pxX;          // centre, pixels
        HighPrecType pxY;
        HighPrecType radiusPx;     // mean radius, pixels
        HighPrecType aspect;       // long axis over short axis, >= 1
        HighPrecType angleRad;     // orientation of the long axis, image frame
        HighPrecType alpha;        // peak opacity, 0..1
        HighPrecType lobeDepth;    // harmonic modulation depth of the radius
        HighPrecType harmPhase[ALGO_DEFECT_HARMONICS];
        HighPrecType chroma[3];    // per-channel opacity weights, max = 1

        // 2026-10-20: stochastic shape, polarity, texture and optics.
        HighPrecType lobeScale[ALGO_DEFECT_HARMONICS]; // per-harmonic, mean 1
        HighPrecType polarity;     // +1 material on/in the film, -1 removal
        HighPrecType scatterShare; // scattering share for the reader law
        HighPrecType texSigma;     // interior white texture, fraction of alpha
        HighPrecType texStepPx;    // texture lattice step, pixels
        uint32_t     texSeed;      // texture lattice key, from the particle

        // 2026-10-20c: ragged outline and per-particle soft edge
        // (AlgoNegativeDefects.hpp). Zero / n = 0 = the old shape and edge.
        HighPrecType      edgeFrac;  // edge width as a fraction of radiusPx
        AlgoDefectOutline outline;   // polygon x roughness radius table
    };


    // ----------------------------------------------------------------------
    //  One record of one pixel through the shared write (AlgoDefectOptics.hpp).
    //  OFF with an additive mark is the old law; everything else goes here.
    // ----------------------------------------------------------------------
    inline AlgoType defectWrite
    (
        const AlgoType             d,
        const HighPrecType         a,
        const HighPrecType         polarity,
        const HighPrecType         share,
        const int32_t              channel,
        const AlgoDefectOpticsCtx& ctx
    ) noexcept
    {
        return static_cast<AlgoType>(AlgoDefectWrite(
            static_cast<HighPrecType>(d), a, polarity, share,
            ALGO_CUT_WALL_SHARE, ALGO_CUT_WALL_SCATTER_SHARE,
            ALGO_DEFECT_SCATTER_REF_HEADROOM, channel, ctx));
    }


    // ----------------------------------------------------------------------
    //  Median-preserving log-normal multiplier from a unit Gaussian and a
    //  coefficient of variation: exp(sigma_ln * n), sigma_ln = sqrt(ln(1+CV^2)).
    // ----------------------------------------------------------------------
    inline HighPrecType defectCvFactor
    (
        const HighPrecType n,
        const HighPrecType cv
    ) noexcept
    {
        return std::exp(std::sqrt(std::log(1.0 + (cv * cv))) * n);
    }


    // ----------------------------------------------------------------------
    //  White texture at the render resolution.
    //
    //  The textures are defined on a lattice of ALGO_DEFECT_TEXTURE_UM, the
    //  scale at which the collage measured them. A pixel coarser than that
    //  INTEGRATES several independent samples, which lowers the standard
    //  deviation it shows: by the ratio of lattice step to pixel size on a plane,
    //  by its square root along a line. So the lattice is never finer than a
    //  pixel and the sigma is reduced to what the pixel would integrate. The
    //  texture then looks the same at every resolution, and point-sampling a
    //  lattice finer than the pixel (which would alias and exaggerate it) never
    //  happens.
    // ----------------------------------------------------------------------
    inline void defectTexture2D
    (
        const HighPrecType sigma,
        const HighPrecType pxPerMm,
        HighPrecType&      sigmaOut,
        HighPrecType&      stepPxOut
    ) noexcept
    {
        const HighPrecType stepMm = ALGO_DEFECT_TEXTURE_UM / ALGO_DEFECT_UM_PER_MM;
        const HighPrecType pixMm  = 1.0 / MAX_VALUE(pxPerMm, 1.0e-9);

        if (stepMm >= pixMm)
        {
            sigmaOut  = sigma;
            stepPxOut = stepMm * pxPerMm;
        }
        else
        {
            sigmaOut  = sigma * (stepMm / pixMm);
            stepPxOut = 1.0;
        }
    }

    inline void defectTexture1D
    (
        const HighPrecType sigma,
        const HighPrecType pxPerMm,
        HighPrecType&      sigmaOut,
        HighPrecType&      stepMmOut
    ) noexcept
    {
        const HighPrecType stepMm = ALGO_DEFECT_TEXTURE_UM / ALGO_DEFECT_UM_PER_MM;
        const HighPrecType pixMm  = 1.0 / MAX_VALUE(pxPerMm, 1.0e-9);

        if (stepMm >= pixMm)
        {
            sigmaOut  = sigma;
            stepMmOut = stepMm;
        }
        else
        {
            sigmaOut  = sigma * std::sqrt(stepMm / pixMm);
            stepMmOut = pixMm;
        }
    }


    // ----------------------------------------------------------------------
    //  Smooth step on 0..1, used for every particle edge.
    //
    //  The cubic 3t^2 - 2t^3. Its derivative vanishes at both ends, so the
    //  transition has no visible corner where it meets either the particle
    //  interior or the clean film - which a linear ramp does have, and which
    //  survives the eye at exactly the scale dust occupies.
    // ----------------------------------------------------------------------
    inline HighPrecType defectSmoothStep (const HighPrecType t) noexcept
    {
        const HighPrecType u = CLAMP_VALUE(t, 0.0, 1.0);
        return u * u * (3.0 - 2.0 * u);
    }


    // ----------------------------------------------------------------------
    //  Per-channel opacity weights for one particle.
    //
    //  Measured colour material gives a median ratio of 0.55 between the
    //  smallest and largest per-channel deviation of a speck, so the three
    //  weights are drawn from the calibrated floor up to 1.0 and then normalised
    //  so the largest is exactly 1. Normalising rather than leaving them free keeps
    //  the particle's
    //  peak opacity equal to the alpha that was drawn, so the colour and the
    //  amount stay independent - otherwise a coloured particle would also be a
    //  fainter one, and the class would lose density as it gained hue.
    // ----------------------------------------------------------------------
    inline void defectChroma
    (
        const uint64_t counter,
        const bool     monochrome,
        HighPrecType   chromaOut[3]
    ) noexcept
    {
        // ------------------------------------------------------------------
        //  A monochrome stock cannot render a coloured particle, and the first
        //  version got this wrong.
        //
        //  A single silver record carries one density and the print carries one
        //  dye, so whatever colour the speck of lint actually was, what the film
        //  records is how much light it blocked - one number. Drawing three
        //  different channel weights put a measurably coloured particle on black
        //  and white film: calibrating on SVEMA_FOTO_65 (then named SVEMA_FN_64;
        //  renamed 2026-08-13), a monochrome negative, the
        //  detected min-over-max channel ratio came out at 0.58 when it must be
        //  exactly 1.
        //
        //  It went unnoticed because the ratio looked close to the measured 0.55,
        //  which is the right figure for the wrong material - that measurement was
        //  made on ORWO colour frames. Two different stocks, two different correct
        //  answers, and the wrong one is not obviously wrong.
        //
        //  Fifty-nine of the 142 stocks in the database are monochrome, so this is
        //  not an edge case.
        // ------------------------------------------------------------------
        if (monochrome)
        {
            chromaOut[0] = 1.0;
            chromaOut[1] = 1.0;
            chromaOut[2] = 1.0;

            return;
        }

        const HighPrecType span = 1.0 - ALGO_DEFECT_CHROMA_MIN;

        HighPrecType peak = 0.0;

        for (int32_t c = 0; c < 3; c++)
        {
            const HighPrecType u = AlgoRngUniform01(counter + static_cast<uint64_t>(c));

            chromaOut[c] = ALGO_DEFECT_CHROMA_MIN + span * u;

            peak = MAX_VALUE(peak, chromaOut[c]);
        }

        // peak is at least ALGO_DEFECT_CHROMA_MIN, which is strictly positive, so
        // this cannot divide by zero.
        for (int32_t c = 0; c < 3; c++)
            chromaOut[c] /= peak;

        return;
    }


    // ----------------------------------------------------------------------
    //  Orientation for an elongated defect, in the IMAGE frame.
    //
    //  The measured population favours the transport axis roughly three to one.
    //  The bias is expressed in FILM coordinates - along the film against across
    //  it - and then rotated into the image frame by asking the window which
    //  image axis the film runs along. That is the whole reason the film
    //  coordinate system exists: this function contains no per-format code, and
    //  the same draw comes out horizontal on a 35 mm still and vertical on every
    //  cine gauge.
    //
    //  ⚠ THE SHARE IS A PARAMETER BECAUSE THERE ARE TWO MEASUREMENTS, NOT ONE.
    //  The particulate classes pass ALGO_DEFECT_ORIENT_ALONG_SHARE (0.76, from a
    //  54-against-17 per cent split); handling scratches pass
    //  ALGO_SCRATCH_ORIENT_ALONG_SHARE (0.78, from the measured 3.5:1 bias). They
    //  are close, which is exactly why they must not be merged - collapsing them
    //  to one constant would silently substitute one measurement for the other
    //  and nothing would look wrong.
    // ----------------------------------------------------------------------
    inline HighPrecType defectOrientation
    (
        const AlgoFilmWindow& window,
        const uint64_t        counter,
        const HighPrecType    alongShare
    ) noexcept
    {
        const HighPrecType uPick   = AlgoRngUniform01(counter);
        const HighPrecType uSpread = AlgoRngUniform01(counter + 1u);
        const HighPrecType uFree   = AlgoRngUniform01(counter + 2u);

        // Anisotropy is zero for sheet film, which has no transport direction, so
        // its orientation is drawn isotropically and the bias never applies.
        const HighPrecType bias = alongShare
                                * static_cast<HighPrecType>(window.anisotropy);

        if (uPick >= bias)
        {
            // The unbiased remainder: any angle, uniformly.
            return uFree * ALGO_DEFECT_TWO_PI;
        }

        // Scatter about the preferred axis. Centred, so the spread is symmetric.
        const HighPrecType jitter = (uSpread - 0.5) * 2.0
                                  * ALGO_DEFECT_ORIENT_SPREAD_RAD;

        // Zero radians is the image X axis. When the film runs along X the
        // along-film direction IS zero; when it runs along Y the along-film
        // direction is a quarter turn away.
        const HighPrecType alongAxis = window.transportAlongWidth
                                     ? 0.0
                                     : (0.25 * ALGO_DEFECT_TWO_PI);

        return alongAxis + jitter;
    }


    // ----------------------------------------------------------------------
    //  Rasterise one particle into the three density planes.
    //
    //  Coverage-based: each pixel receives the fraction of itself the particle
    //  covers, computed from the signed distance to the particle boundary through
    //  a transition whose width is the system point-spread function. That is what
    //  makes a sub-pixel speck fade rather than pop, and what stops a particle
    //  from ever having an edge sharper than the optics.
    //
    //  The particle becomes an ADDITIVE DENSITY. An opaque speck transmits
    //  (1 - alpha) of the light reaching it, and density is minus the base-ten
    //  logarithm of transmittance, so
    //
    //      D' = D - log10(1 - alpha)
    //
    //  and overlapping particles compose correctly by plain addition, because
    //  multiplying transmittances is adding logarithms.
    // ----------------------------------------------------------------------
    void defectRasterise
    (
        const DefectParticle& pIn,
        AlgoType* RESTRICT    pDstR,
        AlgoType* RESTRICT    pDstG,
        AlgoType* RESTRICT    pDstB,
        const int32_t         sizeX,
        const int32_t         sizeY,
        const int32_t         pitch,
        const HighPrecType    edgePx,
        const AlgoDefectOpticsCtx& ctx
    ) noexcept
    {
        // Outer reach of the particle: the long semi-axis at its greatest
        // harmonic excursion, plus the edge transition.
        // 2026-10-20b: a sub-pixel particle is widened to the minimum
        // footprint with its opacity scaled by the area ratio, so it cannot
        // fall between pixel centres and vanish (ALGO_DEFECT_MIN_FOOTPRINT_PX).
        DefectParticle p = pIn;

        if (p.radiusPx < ALGO_DEFECT_MIN_FOOTPRINT_PX)
        {
            const HighPrecType ratio = (p.radiusPx > 0.0)
                ? (p.radiusPx / ALGO_DEFECT_MIN_FOOTPRINT_PX) : 0.0;

            p.alpha    *= ratio * ratio;
            p.radiusPx  = ALGO_DEFECT_MIN_FOOTPRINT_PX;
        }

        // 2026-10-20: the lobe amplitudes are scaled per harmonic, so the
        // greatest excursion is their sum rather than lobeDepth alone.
        HighPrecType lobeMax = 0.0;

        for (int32_t k = 0; k < ALGO_DEFECT_HARMONICS; k++)
            lobeMax += (p.lobeDepth * p.lobeScale[k])
                     / static_cast<HighPrecType>(k + 2);

        // 2026-10-20c: per-particle soft edge, never harder than the PSF edge.
        const HighPrecType edgeP = MAX_VALUE(edgePx, p.edgeFrac * p.radiusPx);

        const HighPrecType reach = p.radiusPx * p.aspect
                                 * (1.0 + MAX_VALUE(lobeMax, p.lobeDepth))
                                 * ((p.outline.n > 0) ? p.outline.rMax : 1.0)
                                 + edgeP;

        // Interior texture lattice, in pixels of the particle's own frame.
        const HighPrecType invTex =
            ((p.texSigma > 0.0) && (p.texStepPx > 0.0)) ? (1.0 / p.texStepPx) : 0.0;

        // The pre-2026-10-20 additive store is kept verbatim for the case it
        // covers: reader optics off and material added.
        const bool plainAdd = (!ctx.on) && (p.polarity > 0.0);

        // defectVisibility on the plain additive store (exact at 1.0).
        const AlgoType visf = static_cast<AlgoType>(ctx.visibility);

        // Bounding box, clamped to the raster. A particle wholly outside
        // contributes nothing and costs one comparison.
        int32_t x0 = static_cast<int32_t>(std::floor(p.pxX - reach));
        int32_t x1 = static_cast<int32_t>(std::floor(p.pxX + reach)) + 1;
        int32_t y0 = static_cast<int32_t>(std::floor(p.pxY - reach));
        int32_t y1 = static_cast<int32_t>(std::floor(p.pxY + reach)) + 1;

        x0 = MAX_VALUE(x0, 0);
        y0 = MAX_VALUE(y0, 0);
        x1 = MIN_VALUE(x1, sizeX - 1);
        y1 = MIN_VALUE(y1, sizeY - 1);

        if (x1 < x0 || y1 < y0)
            return;

        // Rotation of the particle's long axis into the raster, formed once.
        const HighPrecType ca = std::cos(p.angleRad);
        const HighPrecType sa = std::sin(p.angleRad);

        // Reciprocal of the edge width, so the inner loop multiplies.
        const HighPrecType invEdge = 1.0 / edgeP;

        for (int32_t y = y0; y <= y1; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            AlgoType* RESTRICT rR = pDstR + off;
            AlgoType* RESTRICT rG = pDstG + off;
            AlgoType* RESTRICT rB = pDstB + off;

            // ----------------------------------------------------------
            //  RULE D1 ALIGNMENT, 2026-08-11: the particle RASTERISER below was
            //  HighPrecType throughout - including sqrt, atan2, cos and log10 in
            //  double, per pixel of every particle's bounding box.
            //
            //  All of it is a shape mask: a rotated, aspect-compressed radius
            //  compared against a boundary, then a smoothstep to coverage in
            //  [0,1]. Float32 resolves that to ~1e-07 on a quantity that becomes
            //  an opacity and then a density increment. The visual consequence of
            //  the change is nil; the arithmetic consequence is that atan2 and
            //  log10 run in their float forms, which are materially cheaper.
            //
            //  PARTICLE PLACEMENT IS NOT TOUCHED. The positions, radii, alphas
            //  and phases in `p` are computed once per particle by the
            //  log-Gaussian Cox process and stay HighPrecType - that is exactly
            //  the "setup-time scalar" D1 sanctions, and it is also what keeps a
            //  particle in the same place on every render of the frame.
            // ----------------------------------------------------------
            const AlgoType pxYf    = static_cast<AlgoType>(p.pxY);
            const AlgoType pxXf    = static_cast<AlgoType>(p.pxX);
            const AlgoType caf     = static_cast<AlgoType>(ca);
            const AlgoType saf     = static_cast<AlgoType>(sa);
            const AlgoType invAsp  = ALGO_ONE / static_cast<AlgoType>(p.aspect);
            const AlgoType radPxf  = static_cast<AlgoType>(p.radiusPx);
            const AlgoType lobeDf  = static_cast<AlgoType>(p.lobeDepth);
            const AlgoType invEdgf = static_cast<AlgoType>(invEdge);
            const AlgoType alphaf  = static_cast<AlgoType>(p.alpha);

            // Pixel centres, hence the half.
            const AlgoType dy = (static_cast<AlgoType>(y) + ALGO_HALF) - pxYf;

            for (int32_t x = x0; x <= x1; x++)
            {
                const AlgoType dx = (static_cast<AlgoType>(x) + ALGO_HALF) - pxXf;

                // Into the particle's own frame: rotate, then compress the long
                // axis so the shape can be tested as a circle. The reciprocal of
                // the aspect is formed once per particle, so this is a multiply.
                const AlgoType u =  dx * caf + dy * saf;
                const AlgoType v = -dx * saf + dy * caf;

                const AlgoType uu = u * invAsp;

                const AlgoType r = std::sqrt(uu * uu + v * v);

                // Angular harmonics modulate the radius. Evaluated from the
                // unrotated components so the lobes travel with the particle.
                //
                // Three harmonics starting at the second: the first would merely
                // translate the shape, which the centre already expresses.
                AlgoType shape = ALGO_ONE;

                if (r > ALGO_ZERO)
                {
                    const AlgoType theta = std::atan2(v, uu);

                    // 2026-10-20c: ragged outline (factor 1 when n == 0); the
                    // table is HighPrecType, shared with the scalar tree.
                    const AlgoType outF = static_cast<AlgoType>(AlgoDefectOutlineAt(
                        p.outline, static_cast<HighPrecType>(theta)));
                    AlgoType lobes = ALGO_ONE;

                    for (int32_t k = 0; k < ALGO_DEFECT_HARMONICS; k++)
                    {
                        const AlgoType order = static_cast<AlgoType>(k + 2);

                        lobes += ((lobeDf * static_cast<AlgoType>(p.lobeScale[k]))
                                  / order)
                               * std::cos(order * theta
                                          + static_cast<AlgoType>(p.harmPhase[k]));
                    }

                    shape = outF * lobes;
                }

                const AlgoType boundary = radPxf * shape;

                // Coverage. One half-edge inside the boundary the pixel is fully
                // covered, one half-edge outside it is clear.
                const AlgoType cov = static_cast<AlgoType>(defectSmoothStep(
                    static_cast<HighPrecType>(((boundary - r) * invEdgf)
                                              + ALGO_HALF)));

                if (cov <= ALGO_ZERO)
                    continue;

                // Per-channel opacity, capped short of one so the density stays
                // finite.
                // Opacity at this pixel, with the interior texture: Gaussian
                // white noise multiplying the opacity, keyed on the particle's
                // own frame so it travels with the particle (2026-10-20). The
                // texture is HighPrecType, shared with the scalar tree.
                // 2026-10-20c: density ramps with coverage (thinning rim):
                // a = 1 - (1 - alpha)^cov, identical inside.
                AlgoType aPix = (cov >= ALGO_ONE) ? alphaf
                              : (ALGO_ONE - std::pow(ALGO_ONE - alphaf, cov));

                // ⚠ THE TEXTURE COORDINATES ARE FORMED IN HighPrecType, NOT
                // FROM THE FLOAT u, v ABOVE: near the opacity cap a 1e-6 change
                // in the lattice coordinate moves the density visibly, and the
                // two twins must agree on it.
                if (invTex > 0.0)
                {
                    const HighPrecType dxh = (static_cast<HighPrecType>(x) + 0.5) - p.pxX;
                    const HighPrecType dyh = (static_cast<HighPrecType>(y) + 0.5) - p.pxY;

                    const HighPrecType uh =  dxh * ca + dyh * sa;
                    const HighPrecType vh = -dxh * sa + dyh * ca;

                    // 2026-10-20c: texture fades as the local opacity nears 1.
                    const HighPrecType aBase = (cov >= ALGO_ONE)
                        ? static_cast<HighPrecType>(alphaf)
                        : (1.0 - std::pow(1.0 - static_cast<HighPrecType>(alphaf),
                                          static_cast<HighPrecType>(cov)));

                    aPix = static_cast<AlgoType>(aBase
                         * MAX_VALUE(1.0 + p.texSigma * AlgoDefectTextureFade(aBase)
                                   * AlgoDefectWhiteNoise2D(
                               p.texSeed, ALGO_DEFECT_TAG_TEXTURE,
                               uh * invTex, vh * invTex), 0.0));
                }

                const AlgoType aR = MIN_VALUE(aPix
                                        * static_cast<AlgoType>(p.chroma[0]),
                                    static_cast<AlgoType>(ALGO_DEFECT_ALPHA_CAP));
                const AlgoType aG = MIN_VALUE(aPix
                                        * static_cast<AlgoType>(p.chroma[1]),
                                    static_cast<AlgoType>(ALGO_DEFECT_ALPHA_CAP));
                const AlgoType aB = MIN_VALUE(aPix
                                        * static_cast<AlgoType>(p.chroma[2]),
                                    static_cast<AlgoType>(ALGO_DEFECT_ALPHA_CAP));

                if (plainAdd)
                {
                    rR[x] += -visf * std::log10(ALGO_ONE - aR);
                    rG[x] += -visf * std::log10(ALGO_ONE - aG);
                    rB[x] += -visf * std::log10(ALGO_ONE - aB);
                }
                else
                {
                    rR[x] = defectWrite(rR[x], static_cast<HighPrecType>(aR),
                                        p.polarity, p.scatterShare, 0, ctx);
                    rG[x] = defectWrite(rG[x], static_cast<HighPrecType>(aG),
                                        p.polarity, p.scatterShare, 1, ctx);
                    rB[x] = defectWrite(rB[x], static_cast<HighPrecType>(aB),
                                        p.polarity, p.scatterShare, 2, ctx);
                }
            }
        }

        return;
    }


    // ----------------------------------------------------------------------
    //  Rasterise a stroked polyline: a fibre, or a scratch.
    //
    //  Coverage comes from the distance to the NEAREST point on the whole
    //  centreline, not from stamping each segment in turn. Stamping would add
    //  density twice wherever two segments overlap, and since consecutive
    //  segments always overlap at their shared joint, a stamped fibre acquires a
    //  string of beads along its length. Taking the minimum distance over the
    //  whole polyline costs one pass over the stroke's bounding box per segment.
    //
    //  2026-10-20: WIDTH AND OPACITY VARY ALONG THE STROKE. Each centreline
    //  point carries its own half-width and peak opacity; at the nearest point
    //  both are interpolated linearly from the segment's two ends, so a
    //  varying mark stays continuous and keeps the no-beads property. An
    //  optional white texture along the stroke multiplies the opacity, keyed on
    //  the per-point arc coordinate in film millimetres (for a transport
    //  scratch that coordinate is the ABSOLUTE along-film position, so the
    //  texture is a property of the film and continues across frames).
    //
    //  POLARITY. A fibre can only add density. A scratch can do either - a
    //  burnish scatters light out of the reader's beam and adds density, while a
    //  cut removes emulsion and takes density away. Every store goes through
    //  AlgoDefectWrite, whose OFF path is the old law, floored at zero:
    //  D <- max(D + polarity * (-log10(1 - a)), 0). The floor is physical (no
    //  negative density) and changes nothing on the additive path.
    //
    //  Shared, HighPrecType, character-for-character in both trees.
    // ----------------------------------------------------------------------
    struct DefectStroke
    {
        const HighPrecType* x;      // centreline, pixels
        const HighPrecType* y;
        const HighPrecType* hw;     // half-width per point, pixels
        const HighPrecType* am;     // peak opacity per point
        const HighPrecType* arc;    // texture coordinate per point, mm (may be null)
        int32_t             n;      // point count

        HighPrecType chroma[3];
        HighPrecType polarity;      // +1 added, -1 removed
        HighPrecType share;         // scattering share (added marks)

        HighPrecType texSigma;      // 0 = no texture
        HighPrecType texStepMm;
        uint32_t     texSeed;
        int32_t      texStream;
    };

    void defectRasteriseStroke
    (
        const DefectStroke&        st,
        AlgoType* RESTRICT         pDstR,
        AlgoType* RESTRICT         pDstG,
        AlgoType* RESTRICT         pDstB,
        const int32_t              sizeX,
        const int32_t              sizeY,
        const int32_t              pitch,
        const HighPrecType         edgePx,
        const AlgoDefectOpticsCtx& ctx
    ) noexcept
    {
        if (st.n < 2)
            return;

        const HighPrecType* RESTRICT ptsX = st.x;
        const HighPrecType* RESTRICT ptsY = st.y;

        // Bounding box of the whole centreline, and the widest point.
        HighPrecType minX  = ptsX[0];
        HighPrecType maxX  = ptsX[0];
        HighPrecType minY  = ptsY[0];
        HighPrecType maxY  = ptsY[0];
        HighPrecType maxHw = st.hw[0];

        for (int32_t i = 1; i < st.n; i++)
        {
            minX  = MIN_VALUE(minX, ptsX[i]);
            maxX  = MAX_VALUE(maxX, ptsX[i]);
            minY  = MIN_VALUE(minY, ptsY[i]);
            maxY  = MAX_VALUE(maxY, ptsY[i]);
            maxHw = MAX_VALUE(maxHw, st.hw[i]);
        }

        const HighPrecType reach = MAX_VALUE(maxHw, ALGO_DEFECT_MIN_FOOTPRINT_PX)
                                 + edgePx;

        int32_t x0 = static_cast<int32_t>(std::floor(minX - reach));
        int32_t x1 = static_cast<int32_t>(std::floor(maxX + reach)) + 1;
        int32_t y0 = static_cast<int32_t>(std::floor(minY - reach));
        int32_t y1 = static_cast<int32_t>(std::floor(maxY + reach)) + 1;

        x0 = MAX_VALUE(x0, 0);
        y0 = MAX_VALUE(y0, 0);
        x1 = MIN_VALUE(x1, sizeX - 1);
        y1 = MIN_VALUE(y1, sizeY - 1);

        if (x1 < x0 || y1 < y0)
            return;

        const HighPrecType invEdge = 1.0 / edgePx;

        const bool textured = (st.texSigma > 0.0) && (st.arc != nullptr)
                            && (st.texStepMm > 0.0);

        for (int32_t y = y0; y <= y1; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            AlgoType* RESTRICT rR = pDstR + off;
            AlgoType* RESTRICT rG = pDstG + off;
            AlgoType* RESTRICT rB = pDstB + off;

            const HighPrecType py = static_cast<HighPrecType>(y) + 0.5;

            for (int32_t x = x0; x <= x1; x++)
            {
                const HighPrecType px = static_cast<HighPrecType>(x) + 0.5;

                // Squared distance to the nearest point of the polyline, and
                // where on the polyline that point is.
                HighPrecType best  = -1.0;
                int32_t      bestI = 0;
                HighPrecType bestT = 0.0;

                for (int32_t i = 0; i + 1 < st.n; i++)
                {
                    const HighPrecType ax = ptsX[i];
                    const HighPrecType ay = ptsY[i];
                    const HighPrecType bx = ptsX[i + 1];
                    const HighPrecType by = ptsY[i + 1];

                    // Cheap rejection on the segment's own bounding box.
                    if (px < (MIN_VALUE(ax, bx) - reach) ||
                        px > (MAX_VALUE(ax, bx) + reach) ||
                        py < (MIN_VALUE(ay, by) - reach) ||
                        py > (MAX_VALUE(ay, by) + reach))
                        continue;

                    const HighPrecType ex = bx - ax;
                    const HighPrecType ey = by - ay;

                    const HighPrecType len2 = ex * ex + ey * ey;

                    // Projection clamped to the segment, so the stroke has ends.
                    HighPrecType t = 0.0;

                    if (len2 > 0.0)
                        t = CLAMP_VALUE(((px - ax) * ex + (py - ay) * ey) / len2,
                                        0.0, 1.0);

                    const HighPrecType ddx = px - (ax + t * ex);
                    const HighPrecType ddy = py - (ay + t * ey);

                    const HighPrecType d2 = ddx * ddx + ddy * ddy;

                    if (best < 0.0 || d2 < best)
                    {
                        best  = d2;
                        bestI = i;
                        bestT = t;
                    }
                }

                // Every segment rejected: this pixel is nowhere near the stroke.
                if (best < 0.0)
                    continue;

                const HighPrecType dist = std::sqrt(best);

                HighPrecType hwLoc = st.hw[bestI]
                    + bestT * (st.hw[bestI + 1] - st.hw[bestI]);

                // 2026-10-20b: a stroke thinner than the minimum footprint is
                // widened to it with its opacity scaled by the width ratio, so
                // a sub-pixel line never falls between pixel centres.
                HighPrecType footScale = 1.0;

                if (hwLoc < ALGO_DEFECT_MIN_FOOTPRINT_PX)
                {
                    footScale = MAX_VALUE(hwLoc, 0.0) / ALGO_DEFECT_MIN_FOOTPRINT_PX;
                    hwLoc     = ALGO_DEFECT_MIN_FOOTPRINT_PX;
                }

                const HighPrecType cov = defectSmoothStep(
                    ((hwLoc - dist) * invEdge) + 0.5);

                if (cov <= 0.0)
                    continue;

                HighPrecType amLoc = footScale * (st.am[bestI]
                    + bestT * (st.am[bestI + 1] - st.am[bestI]));

                if (textured)
                {
                    const HighPrecType arcLoc = st.arc[bestI]
                        + bestT * (st.arc[bestI + 1] - st.arc[bestI]);

                    amLoc *= MAX_VALUE(1.0 + st.texSigma * AlgoDefectWhiteNoise1D(
                        st.texSeed, st.texStream, ALGO_DEFECT_TAG_TEXTURE,
                        arcLoc, st.texStepMm), 0.0);
                }

                const HighPrecType aR = MIN_VALUE(amLoc * cov * st.chroma[0],
                                                  ALGO_DEFECT_ALPHA_CAP);
                const HighPrecType aG = MIN_VALUE(amLoc * cov * st.chroma[1],
                                                  ALGO_DEFECT_ALPHA_CAP);
                const HighPrecType aB = MIN_VALUE(amLoc * cov * st.chroma[2],
                                                  ALGO_DEFECT_ALPHA_CAP);

                rR[x] = defectWrite(rR[x], aR, st.polarity, st.share, 0, ctx);
                rG[x] = defectWrite(rG[x], aG, st.polarity, st.share, 1, ctx);
                rB[x] = defectWrite(rB[x], aB, st.polarity, st.share, 2, ctx);
            }
        }

        return;
    }


    // ----------------------------------------------------------------------
    //  Subdivide a control polyline into a smooth drawing polyline.
    //
    //  Catmull-Rom, which passes exactly through every control point - so the walk
    //  the persistence model produced is preserved rather than approximated, and
    //  only the space between its points is filled in. A B-spline would smooth the
    //  control points themselves and quietly reduce the curvature that the
    //  persistence length was chosen to produce.
    //
    //  The endpoints are duplicated to give the first and last spans the four
    //  points the interpolant needs, which keeps the fibre's free ends exactly
    //  where the walk put them.
    //
    //  Returns the number of drawn points written.
    // ----------------------------------------------------------------------
    int32_t defectSubdivide
    (
        const HighPrecType* RESTRICT ctlX,
        const HighPrecType* RESTRICT ctlY,
        const int32_t                ctlCount,
        HighPrecType* RESTRICT       outX,
        HighPrecType* RESTRICT       outY
    ) noexcept
    {
        if (ctlCount < 2)
            return 0;

        int32_t n = 0;

        for (int32_t i = 0; i + 1 < ctlCount; i++)
        {
            // The four points of the span, with the ends clamped rather than
            // wrapped: a fibre is an open curve.
            const int32_t i0 = MAX_VALUE(i - 1, 0);
            const int32_t i1 = i;
            const int32_t i2 = i + 1;
            const int32_t i3 = MIN_VALUE(i + 2, ctlCount - 1);

            for (int32_t s = 0; s < ALGO_FIBRE_SUBDIV; s++)
            {
                const HighPrecType t = static_cast<HighPrecType>(s)
                    / static_cast<HighPrecType>(ALGO_FIBRE_SUBDIV);

                const HighPrecType t2 = t * t;
                const HighPrecType t3 = t2 * t;

                // Uniform Catmull-Rom basis, the standard half-tension form.
                const HighPrecType b0 = 0.5 * (-t3 + 2.0 * t2 - t);
                const HighPrecType b1 = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0);
                const HighPrecType b2 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t);
                const HighPrecType b3 = 0.5 * (t3 - t2);

                outX[n] = b0 * ctlX[i0] + b1 * ctlX[i1]
                        + b2 * ctlX[i2] + b3 * ctlX[i3];
                outY[n] = b0 * ctlY[i0] + b1 * ctlY[i1]
                        + b2 * ctlY[i2] + b3 * ctlY[i3];

                n++;
            }
        }

        // The final control point, which the loop above stops just short of.
        outX[n] = ctlX[ctlCount - 1];
        outY[n] = ctlY[ctlCount - 1];
        n++;

        return n;
    }


    // ----------------------------------------------------------------------
    //  Log-normal draw with an explicit median.
    //
    //  Parameterised by the median rather than by the mean of the logarithm,
    //  because the median is what was measured and the two differ for a skewed
    //  distribution. Getting this backwards inflates every size by exp(sigma^2/2),
    //  which at sigma 0.5 is thirteen per cent - small enough to look plausible
    //  and wrong enough to fail a size-histogram check.
    // ----------------------------------------------------------------------
    inline HighPrecType defectLogNormal
    (
        const uint64_t     counter,
        const HighPrecType median,
        const HighPrecType sigmaLn
    ) noexcept
    {
        return median * std::exp(sigmaLn * AlgoRngNormal(counter));
    }


    // ----------------------------------------------------------------------
    //  Exponential draw with a given mean, for a transport scratch's run length.
    //
    //  Inverted in closed form, and written the same way stage 16's
    //  gateExponential is - the two are the same distribution serving the same
    //  purpose in the same determinism contract, and they should look alike.
    //
    //  Exponential rather than a fixed length because
    //  TemporalSpec::scratch_persistence_frames is documented as a MEAN. A fixed
    //  run of exactly the mean would make every scratch on a stock last precisely
    //  as long as every other, which is a signature no physical process has.
    //
    //  The uniform is floored away from zero: log(0) is not finite and one
    //  unlucky draw in 2^53 would give a scratch an infinite run.
    // ----------------------------------------------------------------------
    inline HighPrecType defectExponential
    (
        const uint64_t     counter,
        const HighPrecType mean
    ) noexcept
    {
        const HighPrecType u = MAX_VALUE(AlgoRngUniform01(counter), 1.0e-12);

        return -mean * std::log(u);
    }


    // ----------------------------------------------------------------------
    //  Per-point arc length of a pixel polyline, in film millimetres.
    // ----------------------------------------------------------------------
    inline void defectArcMm
    (
        const HighPrecType* RESTRICT x,
        const HighPrecType* RESTRICT y,
        const int32_t                n,
        const HighPrecType           pxPerMm,
        HighPrecType* RESTRICT       arcOut
    ) noexcept
    {
        const HighPrecType inv = 1.0 / MAX_VALUE(pxPerMm, 1.0e-9);

        if (n <= 0)
            return;

        arcOut[0] = 0.0;

        for (int32_t i = 1; i < n; i++)
        {
            const HighPrecType dx = x[i] - x[i - 1];
            const HighPrecType dy = y[i] - y[i - 1];

            arcOut[i] = arcOut[i - 1] + std::sqrt(dx * dx + dy * dy) * inv;
        }
    }


    // ----------------------------------------------------------------------
    //  Walk the placement cells covering this frame's window plus a margin, and
    //  hand each drawn position to the caller's generator.
    //
    //  Stratified per-cell Poisson placement, NOT rejection sampling against the
    //  intensity field. The two produce the same distribution, but stratification
    //  is a pure function of the cell's integer film coordinates, so it is
    //  stateless, order-independent and identical however the frame is tiled.
    //  Rejection sampling needs a running trial count, which makes the result
    //  depend on how many candidates were rejected before it - and therefore on
    //  the traversal order.
    //
    //  The cells are indexed in FILM coordinates, so the same patch of film draws
    //  the same particles whichever frame it appears in.
    // ----------------------------------------------------------------------
    template <typename PlaceFn>
    void defectWalkCells
    (
        const AlgoFilmWindow& window,
        const HighPrecType    marginMm,
        const HighPrecType    lambdaPerMm2,
        const HighPrecType    clumping,
        const uint32_t        fieldSeed,
        const uint32_t        cellTag,
        PlaceFn               place
    ) noexcept
    {
        if (lambdaPerMm2 <= 0.0)
            return;

        const HighPrecType cell = ALGO_DEFECT_CELL_MM;

        // Cell index range covering the window plus the margin. floor on both
        // ends and an inclusive upper bound, so a cell that only partially
        // overlaps is still visited.
        const int32_t aLo = static_cast<int32_t>(
            std::floor((window.alongMin - marginMm) / cell));
        const int32_t aHi = static_cast<int32_t>(
            std::floor((window.alongMax + marginMm) / cell));
        const int32_t cLo = static_cast<int32_t>(
            std::floor((window.acrossMin - marginMm) / cell));
        const int32_t cHi = static_cast<int32_t>(
            std::floor((window.acrossMax + marginMm) / cell));

        // Expected count in one cell at the base rate.
        const HighPrecType cellArea = cell * cell;

        for (int32_t ai = aLo; ai <= aHi; ai++)
        {
            for (int32_t ci = cLo; ci <= cHi; ci++)
            {
                // Cell centre in film millimetres.
                const HighPrecType alongC =
                    (static_cast<HighPrecType>(ai) + 0.5) * cell;
                const HighPrecType acrossC =
                    (static_cast<HighPrecType>(ci) + 0.5) * cell;

                // The clumped local rate. Sampled at the cell centre: the cell is
                // one millimetre and the field's finest octave is the same size,
                // so a single sample is the right resolution rather than a
                // shortcut.
                const HighPrecType g = AlgoDefectFieldValue(
                    alongC, acrossC, fieldSeed, ALGO_DEFECT_TAG_DUST_FIELD);

                const HighPrecType lambda = AlgoDefectCoxIntensity(
                    lambdaPerMm2, g, clumping);

                const uint64_t cellCounter =
                    AlgoDefectHash(fieldSeed, ai, ci, cellTag);

                const int32_t count = AlgoDefectPoisson(cellCounter,
                                                        lambda * cellArea);

                for (int32_t n = 0; n < count; n++)
                {
                    // Each particle gets its own stream, offset by its ordinal
                    // within the cell. Multiplied by a stride wide enough that
                    // one particle's draws cannot reach the next one's.
                    const uint64_t pc = cellCounter
                                      + static_cast<uint64_t>(n + 1) * 0x100u;

                    // Uniform within the cell. The clumping lives in the count,
                    // not in the position, which is what makes the process a Cox
                    // process rather than an ad-hoc attractor.
                    const HighPrecType along = (static_cast<HighPrecType>(ai)
                        + AlgoRngUniform01(pc)) * cell;
                    const HighPrecType across = (static_cast<HighPrecType>(ci)
                        + AlgoRngUniform01(pc + 1u)) * cell;

                    place(along, across, pc);
                }
            }
        }

        return;
    }


    // ----------------------------------------------------------------------
    //  Fine dust: the dominant class.
    // ----------------------------------------------------------------------
    void defectDust
    (
        const AlgoFilmWindow& window,
        const HighPrecType    level,
        const HighPrecType    clumping,
        const uint32_t        fieldSeed,
        const bool            monochrome,
        AlgoType* RESTRICT    pDstR,
        AlgoType* RESTRICT    pDstG,
        AlgoType* RESTRICT    pDstB,
        const int32_t         sizeX,
        const int32_t         sizeY,
        const int32_t         pitch,
        const HighPrecType    edgePx,
        const AlgoDefectOpticsCtx& ctx
    ) noexcept
    {
        // Requested density, clamped to the measured ceiling, and reduced to the
        // embedded share because only embedded particles belong on the negative.
        const HighPrecType lambda =
            MIN_VALUE(level * ALGO_DUST_DENSITY_PER_MM2, ALGO_DUST_DENSITY_MAX)
            * ALGO_DUST_EMBEDDED_FRACTION;

        defectWalkCells(window, ALGO_DUST_MARGIN_MM, lambda, clumping,
                        fieldSeed, ALGO_DEFECT_TAG_DUST_CELL,
            [&](const HighPrecType along,
                const HighPrecType across,
                const uint64_t     pc) noexcept
        {
            DefectParticle p{};

            AlgoFilmToPixel(window, along, across, p.pxX, p.pxY);

            // Diameter from the truncated power law, inverted in closed form.
            const HighPrecType dUm = AlgoDefectPowerLawSize(
                AlgoRngUniform01(pc + 2u),
                ALGO_DUST_SIZE_MIN_UM,
                ALGO_DUST_SIZE_MAX_UM,
                ALGO_DUST_SIZE_GAMMA);

            const HighPrecType dMm = dUm / ALGO_DEFECT_UM_PER_MM;

            p.radiusPx = 0.5 * dMm * static_cast<HighPrecType>(window.pxPerMm);

            // Opacity: Beta(2,3) on the measured range, then driven towards fully
            // opaque over the coarse span, because a large particle is a chip of
            // material and not a translucent speck.
            const HighPrecType beta = AlgoDefectBeta23(pc + 8u);

            // Beta(2,3), re-centred on the calibrated median and scaled by the
            // calibrated dispersion. Location and scale are solved against the
            // measured output amplitudes; the SHAPE is the measured one and is left
            // alone, which is why the draw is transformed rather than replaced.
            const HighPrecType aDrawn = MAX_VALUE(
                ALGO_DUST_ALPHA_MID
                * (1.0 + ALGO_DUST_ALPHA_SPREAD
                       * (beta - ALGO_DEFECT_BETA23_MEAN)
                       / ALGO_DEFECT_BETA23_MEAN),
                ALGO_DUST_ALPHA_FLOOR);

            // The ramp begins ABOVE the dust size limit, so for this class it
            // evaluates to zero throughout. That is intended - saturation was
            // measured on 0.3 mm particles, which are coarse debris - and it is
            // written as a ramp rather than removed so the behaviour stays correct
            // if the size limit is ever raised into it.
            const HighPrecType opaqueMix = CLAMP_VALUE(
                (dUm - ALGO_DUST_OPAQUE_ONSET_UM)
                / (ALGO_DUST_OPAQUE_FULL_UM - ALGO_DUST_OPAQUE_ONSET_UM),
                0.0, 1.0);

            p.alpha = aDrawn + (1.0 - aDrawn) * opaqueMix;

            // Elongation up to the 3:1 limit that separates dust from fibres.
            p.aspect = 1.0 + (ALGO_DUST_ASPECT_MAX - 1.0)
                     * AlgoRngUniform01(pc + 12u);

            p.angleRad = defectOrientation(window, pc + 16u,
                                           ALGO_DEFECT_ORIENT_ALONG_SHARE);

            p.lobeDepth = ALGO_DUST_LOBE_DEPTH;

            for (int32_t k = 0; k < ALGO_DEFECT_HARMONICS; k++)
                p.harmPhase[k] = AlgoRngUniform01(
                    pc + 20u + static_cast<uint64_t>(k)) * ALGO_DEFECT_TWO_PI;

            defectChroma(pc + 28u, monochrome, p.chroma);

            // 2026-10-20: shape, polarity, texture, optics.
            //
            // Each harmonic's lobe amplitude scaled by U(1 - x, 1 + x): mean
            // kept, outlines no longer share one irregularity.
            for (int32_t k = 0; k < ALGO_DEFECT_HARMONICS; k++)
                p.lobeScale[k] = 1.0 + ALGO_DEFECT_LOBE_VARIATION
                    * (2.0 * AlgoRngUniform01(pc + 32u + static_cast<uint64_t>(k))
                       - 1.0);

            // A share of dust-class specks are REMOVALS - a pinhole or a pick-
            // off, not material - and read with the opposite polarity. Removal
            // takes all layers, so it is neutral like a scratch.
            p.polarity = (AlgoRngUniform01(pc + 36u) < ALGO_DUST_REMOVAL_SHARE)
                       ? -1.0 : 1.0;

            if (p.polarity < 0.0)
            {
                p.chroma[0] = 1.0;
                p.chroma[1] = 1.0;
                p.chroma[2] = 1.0;
            }

            p.scatterShare = ALGO_DUST_SCATTER_SHARE;
            defectTexture2D(ALGO_DUST_TEXTURE_SIGMA,
                            static_cast<HighPrecType>(window.pxPerMm),
                            p.texSigma, p.texStepPx);
            p.texSeed      = AlgoDefectSubSeed(pc + 40u);

            // 2026-10-20c: ragged outline and size-proportional soft edge.
            p.edgeFrac = ALGO_DUST_EDGE_FRAC * std::exp(
                ALGO_DEFECT_EDGE_FRAC_SIGMA_LN
                * AlgoDefectNormalQuantile(AlgoRngUniform01(pc + 44u)));
            AlgoDefectOutlineMake(AlgoDefectSubSeed(pc + 48u), p.radiusPx,
                                  p.outline);

            defectRasterise(p, pDstR, pDstG, pDstB,
                            sizeX, sizeY, pitch, edgePx, ctx);
        });

        return;
    }


    // ----------------------------------------------------------------------
    //  Coarse debris: rare, large, opaque, lobed.
    // ----------------------------------------------------------------------
    void defectDebris
    (
        const AlgoFilmWindow& window,
        const HighPrecType    level,
        const HighPrecType    clumping,
        const uint32_t        fieldSeed,
        const bool            monochrome,
        AlgoType* RESTRICT    pDstR,
        AlgoType* RESTRICT    pDstG,
        AlgoType* RESTRICT    pDstB,
        const int32_t         sizeX,
        const int32_t         sizeY,
        const int32_t         pitch,
        const HighPrecType    edgePx,
        const AlgoDefectOpticsCtx& ctx
    ) noexcept
    {
        const HighPrecType lambda = level * ALGO_DEBRIS_DENSITY_PER_MM2;

        defectWalkCells(window, ALGO_DEBRIS_MARGIN_MM, lambda, clumping,
                        fieldSeed, ALGO_DEFECT_TAG_DEBRIS_CELL,
            [&](const HighPrecType along,
                const HighPrecType across,
                const uint64_t     pc) noexcept
        {
            DefectParticle p{};

            AlgoFilmToPixel(window, along, across, p.pxX, p.pxY);

            // Log-normal size, clamped so the tail cannot exceed the largest
            // particle ever measured.
            const HighPrecType dMm = MIN_VALUE(
                defectLogNormal(pc + 2u, ALGO_DEBRIS_MEDIAN_MM,
                                ALGO_DEBRIS_SIGMA_LN),
                ALGO_DEBRIS_MAX_MM);

            p.radiusPx = 0.5 * dMm * static_cast<HighPrecType>(window.pxPerMm);

            p.alpha = ALGO_DEBRIS_ALPHA_MIN
                    + (ALGO_DEBRIS_ALPHA_MAX - ALGO_DEBRIS_ALPHA_MIN)
                    * AlgoRngUniform01(pc + 6u);

            // Less elongated than dust: debris is lobed rather than stretched.
            p.aspect = 1.0 + 0.6 * AlgoRngUniform01(pc + 10u);

            p.angleRad = defectOrientation(window, pc + 14u,
                                           ALGO_DEFECT_ORIENT_ALONG_SHARE);

            // Deep harmonics give the irregular, concave outline of a lint ball.
            p.lobeDepth = ALGO_DEBRIS_LOBE_DEPTH;

            for (int32_t k = 0; k < ALGO_DEFECT_HARMONICS; k++)
                p.harmPhase[k] = AlgoRngUniform01(
                    pc + 20u + static_cast<uint64_t>(k)) * ALGO_DEFECT_TWO_PI;

            defectChroma(pc + 28u, monochrome, p.chroma);

            // 2026-10-20: shape, texture, optics. Debris is material, so
            // always additive.
            for (int32_t k = 0; k < ALGO_DEFECT_HARMONICS; k++)
                p.lobeScale[k] = 1.0 + ALGO_DEFECT_LOBE_VARIATION
                    * (2.0 * AlgoRngUniform01(pc + 32u + static_cast<uint64_t>(k))
                       - 1.0);

            p.polarity     = 1.0;
            p.scatterShare = ALGO_DEBRIS_SCATTER_SHARE;
            defectTexture2D(ALGO_DEBRIS_TEXTURE_SIGMA,
                            static_cast<HighPrecType>(window.pxPerMm),
                            p.texSigma, p.texStepPx);
            p.texSeed      = AlgoDefectSubSeed(pc + 40u);

            // 2026-10-20c: ragged outline and size-proportional soft edge.
            p.edgeFrac = ALGO_DEBRIS_EDGE_FRAC * std::exp(
                ALGO_DEFECT_EDGE_FRAC_SIGMA_LN
                * AlgoDefectNormalQuantile(AlgoRngUniform01(pc + 44u)));
            AlgoDefectOutlineMake(AlgoDefectSubSeed(pc + 48u), p.radiusPx,
                                  p.outline);

            defectRasterise(p, pDstR, pDstG, pDstB,
                            sizeX, sizeY, pitch, edgePx, ctx);
        });

        return;
    }


    // ----------------------------------------------------------------------
    //  Hair and fibres.
    //
    //  The centreline is a random walk WITH PERSISTENCE: each step turns by a
    //  small random amount whose scale is set by the persistence length, so the
    //  direction decorrelates over a few millimetres. That single mechanism gives
    //  the gentle curvature, the occasional loop and the plausible overall shape
    //  at once.
    //
    //  A straight line with noise added does not work - it reads as a scratch,
    //  because a scratch IS straight. The end-to-end chord over traced arc length
    //  is the measurable difference: fibres come out at 0.7 to 0.95, scratches at
    //  0.98 to 0.99, and the persistence length is what sets it.
    // ----------------------------------------------------------------------
    void defectFibres
    (
        const AlgoFilmWindow& window,
        const HighPrecType    level,
        const HighPrecType    clumping,
        const uint32_t        fieldSeed,
        const bool            monochrome,
        AlgoType* RESTRICT    pDstR,
        AlgoType* RESTRICT    pDstG,
        AlgoType* RESTRICT    pDstB,
        const int32_t         sizeX,
        const int32_t         sizeY,
        const int32_t         pitch,
        const HighPrecType    edgePx,
        const AlgoDefectOpticsCtx& ctx
    ) noexcept
    {
        const HighPrecType lambda = level * ALGO_FIBRE_DENSITY_PER_MM2;

        defectWalkCells(window, ALGO_FIBRE_MARGIN_MM, lambda, clumping,
                        fieldSeed, ALGO_DEFECT_TAG_FIBRE_CELL,
            [&](const HighPrecType along,
                const HighPrecType across,
                const uint64_t     pc) noexcept
        {
            // Length and width, both in film units so they scale with gauge.
            const HighPrecType lenMm = CLAMP_VALUE(
                defectLogNormal(pc + 2u, ALGO_FIBRE_LENGTH_MEDIAN_MM,
                                ALGO_FIBRE_LENGTH_SIGMA_LN),
                ALGO_FIBRE_LENGTH_MIN_MM, ALGO_FIBRE_LENGTH_MAX_MM);

            const HighPrecType widthUm = ALGO_FIBRE_WIDTH_MIN_UM
                + (ALGO_FIBRE_WIDTH_MAX_UM - ALGO_FIBRE_WIDTH_MIN_UM)
                * AlgoRngUniform01(pc + 6u);

            const HighPrecType alpha = ALGO_FIBRE_ALPHA_MIN
                + (ALGO_FIBRE_ALPHA_MAX - ALGO_FIBRE_ALPHA_MIN)
                * AlgoRngUniform01(pc + 8u);

            // Step count from the length, bounded by the fixed point store.
            const HighPrecType stepMm =
                1.0 / static_cast<HighPrecType>(ALGO_FIBRE_STEPS_PER_MM);

            int32_t steps = static_cast<int32_t>(
                lenMm * static_cast<HighPrecType>(ALGO_FIBRE_STEPS_PER_MM));

            steps = CLAMP_VALUE(steps, 1, ALGO_FIBRE_MAX_POINTS - 1);

            const int32_t pointCount = steps + 1;

            // Per-step angular diffusion. A walk whose heading performs a random
            // walk with variance stepMm / persistence per step has its direction
            // correlation decaying over the persistence length, which is the
            // definition being implemented.
            const HighPrecType turnSigma =
                std::sqrt(stepMm / ALGO_FIBRE_PERSISTENCE_MM);

            // Does this one carry a terminal hook?
            const bool hooked =
                (AlgoRngUniform01(pc + 10u) < ALGO_FIBRE_HOOK_PROBABILITY);

            const int32_t hookFrom = pointCount
                - MAX_VALUE(static_cast<int32_t>(
                    static_cast<HighPrecType>(pointCount)
                    * ALGO_FIBRE_HOOK_FRACTION), 1);

            // A fibre lies on the film in any direction; unlike an abrasion it
            // has no reason to follow the transport, so the initial heading is
            // drawn isotropically.
            HighPrecType heading = AlgoRngUniform01(pc + 12u)
                                 * ALGO_DEFECT_TWO_PI;

            HighPrecType alongCur  = along;
            HighPrecType acrossCur = across;

            // Control points from the persistent walk, then the smooth polyline
            // that is actually stroked.
            HighPrecType ctlX[ALGO_FIBRE_MAX_POINTS];
            HighPrecType ctlY[ALGO_FIBRE_MAX_POINTS];

            AlgoFilmToPixel(window, alongCur, acrossCur, ctlX[0], ctlY[0]);

            for (int32_t i = 1; i < pointCount; i++)
            {
                // Free diffusion of the heading, plus a steady turn over the last
                // fifth if this fibre is hooked. The steady part is what produces
                // a recognisable hook rather than merely more curvature.
                heading += turnSigma * AlgoRngNormal(
                    pc + 64u + static_cast<uint64_t>(i));

                if (hooked && i >= hookFrom)
                    heading += ALGO_FIBRE_HOOK_TURN_RAD;

                alongCur  += stepMm * std::cos(heading);
                acrossCur += stepMm * std::sin(heading);

                AlgoFilmToPixel(window, alongCur, acrossCur,
                                ctlX[i], ctlY[i]);
            }

            // Fill in between the control points, so the stroke has no facets.
            HighPrecType ptsX[ALGO_FIBRE_MAX_DRAW];
            HighPrecType ptsY[ALGO_FIBRE_MAX_DRAW];

            const int32_t drawCount = defectSubdivide(ctlX, ctlY, pointCount,
                                                      ptsX, ptsY);

            HighPrecType chroma[3];
            defectChroma(pc + 28u, monochrome, chroma);

            const HighPrecType halfWidthPx = 0.5
                * (widthUm / ALGO_DEFECT_UM_PER_MM)
                * static_cast<HighPrecType>(window.pxPerMm);

            // 2026-10-20: width and opacity vary ALONG the fibre - smooth
            // log-normal multipliers over ALGO_FIBRE_MOD_CORR_MM of arc, keyed
            // on this fibre's own stream (collage: width CV 0.17 - 0.30,
            // contrast CV 0.38 - 0.72).
            HighPrecType ptsHw[ALGO_FIBRE_MAX_DRAW];
            HighPrecType ptsAm[ALGO_FIBRE_MAX_DRAW];
            HighPrecType ptsArc[ALGO_FIBRE_MAX_DRAW];

            defectArcMm(ptsX, ptsY, drawCount,
                        static_cast<HighPrecType>(window.pxPerMm), ptsArc);

            const uint32_t sk = AlgoDefectSubSeed(pc + 200u);

            for (int32_t i = 0; i < drawCount; i++)
            {
                ptsHw[i] = halfWidthPx * defectCvFactor(
                    AlgoDefectSmoothNoise(sk, 0, ALGO_DEFECT_TAG_MODULATION,
                        ptsArc[i], ALGO_FIBRE_MOD_CORR_MM),
                    ALGO_FIBRE_WIDTH_CV);

                ptsAm[i] = MIN_VALUE(alpha * defectCvFactor(
                    AlgoDefectSmoothNoise(sk, 1, ALGO_DEFECT_TAG_MODULATION,
                        ptsArc[i], ALGO_FIBRE_MOD_CORR_MM),
                    ALGO_FIBRE_CONTRAST_CV), ALGO_DEFECT_ALPHA_CAP);
            }

            // A fibre lies ON the film and blocks light, so it can only ever
            // ADD density: polarity +1 always.
            DefectStroke st{};

            st.x         = ptsX;
            st.y         = ptsY;
            st.hw        = ptsHw;
            st.am        = ptsAm;
            st.arc       = nullptr;
            st.n         = drawCount;
            st.chroma[0] = chroma[0];
            st.chroma[1] = chroma[1];
            st.chroma[2] = chroma[2];
            st.polarity  = 1.0;
            st.share     = ALGO_FIBRE_SCATTER_SHARE;
            st.texSigma  = 0.0;
            st.texStepMm = 0.0;
            st.texSeed   = 0u;
            st.texStream = 0;

            defectRasteriseStroke(st, pDstR, pDstG, pDstB,
                                  sizeX, sizeY, pitch, edgePx, ctx);
        });

        return;
    }


    // ----------------------------------------------------------------------
    //  Scratches: abrasion IN the film.
    //
    //  Two populations behind two existing controls, sharing one mark primitive
    //  (defectRasteriseStroke). What differs is placement, orientation, time,
    //  and since 2026-10-20 each population's own width, contrast and polarity
    //  statistics (AlgoNegativeDefects.hpp).
    //
    //  ⚠ SCRATCHES ARE RENDERED NEUTRAL, UNLIKE EVERY OTHER CLASS IN THIS FILE,
    //  AND THAT IS DELIBERATE. defectChroma exists because a PARTICLE is coloured
    //  - lint is not grey, and a speck sitting in one emulsion layer blocks that
    //  layer more than the others. A scratch's colour is a different quantity
    //  with a different cause: how deep the abrader went decides which dye layers
    //  survive under it, so it is a property of the TRIPACK. AlgoControl.hpp says
    //  so explicitly - scratch COLOUR "needs the tripack", belongs in AgingSpec
    //  and CoatingSpec, and is listed there as something that does NOT belong in
    //  the damage controls. Those structures ship all-zero on every stock, so the
    //  depth-to-layer mapping is not available, and a neutral scratch is the
    //  honest placeholder.
    // ----------------------------------------------------------------------
    void defectScratches
    (
        const AlgoFilmWindow&      window,
        const HighPrecType         transportLevel,
        const HighPrecType         handlingLevel,
        const HighPrecType         clumping,
        const uint32_t             fieldSeed,
        const int32_t              frameIndex,
        const HighPrecType         lifetimeS,
        const HighPrecType         fps,
        const HighPrecType         variability,
        const HighPrecType         framePitchMm,
        AlgoType* RESTRICT         pDstR,
        AlgoType* RESTRICT         pDstG,
        AlgoType* RESTRICT         pDstB,
        const int32_t              sizeX,
        const int32_t              sizeY,
        const int32_t              pitch,
        const HighPrecType         edgePx,
        const AlgoDefectOpticsCtx& ctx
    ) noexcept
    {
        const HighPrecType pxPerMm = static_cast<HighPrecType>(window.pxPerMm);

        // Micrometres of WIDTH to pixels of HALF-width.
        const HighPrecType umToHalfPx = 0.5 * pxPerMm / ALGO_DEFECT_UM_PER_MM;

        // The along-mark white texture at this resolution: lattice step, mm,
        // and the sigma a pixel shows.
        HighPrecType texSigma  = 0.0;
        HighPrecType texStepMm = 0.0;

        defectTexture1D(ALGO_SCRATCH_TEXTURE_CV, pxPerMm, texSigma, texStepMm);

        // ------------------------------------------------------------------
        //  TRANSPORT SCRATCHES - the tramline, and the only thing in this stage
        //  with a start and an end.
        //
        //  ⚠ THE POPULATION IS DERIVED, NEVER ACCUMULATED. Algorithm_Main is a
        //  pure function and a host may render frame 900 without ever having
        //  rendered 899. So each event ORDINAL draws its own gap, lifetime,
        //  across-web position, polarity, width and contrast from its own
        //  counter stream, and is present at time t exactly when
        //  start <= t < start + lifetime (2026-10-20b: in SECONDS of screen
        //  time, start = running sum of bounded gaps). Scrubbing the timeline,
        //  rendering backwards and rendering one frame alone all give the same
        //  tramlines.
        //
        //  ⚠ THE ACROSS-WEB POSITION IS DRAWN ONCE PER ORDINAL, NOT PER FRAME:
        //  the abrader does not move, so the tramline holds its place on screen
        //  while the picture travels past it.
        //
        //  2026-10-20: WHAT VARIES ALONG THE TRAMLINE IS KEYED ON THE ABSOLUTE
        //  ALONG-FILM POSITION. The lateral wander of the track (the film weaving
        //  past the abrader), the width, and the contrast (a fast and a slow
        //  process) are smooth Gaussian processes of that coordinate, and the
        //  short-scale texture is white noise on it. The scratch is IN the film,
        //  so these belong to the film: each frame shows the next stretch of the
        //  same continuous mark - no flicker and no repeat - and a re-render of
        //  a frame reproduces it exactly. Half the tramlines carry a faint
        //  companion track beside them.
        // ------------------------------------------------------------------
        if (transportLevel > 0.0)
        {
            // ⚠ SHEET AND PACK FILM HAVE NO TRANSPORT AND THEREFORE NO
            // TRAMLINES (anisotropy zero; a zero pitch would also make a run
            // along the web meaningless). The handling population still applies.
            const bool transported =
                (static_cast<HighPrecType>(window.anisotropy) > 0.0)
                && (framePitchMm > 0.0);

            if (transported && lifetimeS > 0.0 && fps > 0.0)
            {
                // ----------------------------------------------------------
                //  2026-10-20b: EVENTS IN SECONDS OF SCREEN TIME.
                //
                //  transportLevel is NEW EVENTS PER MINUTE (already floored at
                //  ScratchTransportFloor by the caller). The gap before event k
                //  is uniform on [GAP_LO, GAP_HI] x m, drawn from event k's own
                //  key; event times are the running sum of gaps from a random
                //  phase, so the process is an exact renewal process with
                //  bounded gaps, and at the floor rate every gap is 5 - 20 s.
                //  t = frameIndex / fps, so the same seed gives the same
                //  events in time at any frame rate.
                // ----------------------------------------------------------
                const HighPrecType t       = static_cast<HighPrecType>(frameIndex) / fps;
                const HighPrecType meanGap = 60.0 / transportLevel;
                const HighPrecType horizon = ALGO_SCRATCH_LIFE_HI * lifetimeS;

                // The schedule is the shared AlgoScratchGap / AlgoScratchLastEvent
                // (AlgoNegativeDefects.hpp), so the tests measure what renders.
                auto gapOf = [&](const int32_t k) noexcept -> HighPrecType
                {
                    return AlgoScratchGap(fieldSeed, meanGap, k);
                };

                const HighPrecType e0 = AlgoScratchEvent0(fieldSeed, meanGap);

                HighPrecType eLast = e0;

                const int32_t kLast = AlgoScratchLastEvent(fieldSeed, meanGap, t, eLast);

                const HighPrecType acrossExtent = AlgoFilmAcrossExtentMm(window);
                const HighPrecType alongExtent  = AlgoFilmAlongExtentMm(window);

                // Centreline sample step: ALGO_SCRATCH_SAMPLE_MM, widened by a
                // whole factor only on a gauge whose window would overflow the
                // store. A function of the gauge alone, so the samples stay on
                // the same absolute film positions from frame to frame.
                const HighPrecType spanMax = alongExtent
                                           + 2.0 * ALGO_SCRATCH_CLIP_MARGIN_MM;

                const HighPrecType stepFactor = MAX_VALUE(1.0, std::ceil(
                    spanMax / (ALGO_SCRATCH_SAMPLE_MM
                               * static_cast<HighPrecType>(
                                   ALGO_SCRATCH_TRANSPORT_MAX_POINTS - 3))));

                const HighPrecType stepMm = ALGO_SCRATCH_SAMPLE_MM * stepFactor;

                // defectVariability scales what changes from frame to frame
                // along a running tramline: the wander and the contrast.
                const HighPrecType varScale = MAX_VALUE(variability, 0.0)
                                            / ALGO_DEFECT_VARIABILITY_REF;

                const HighPrecType wanderRms  = ALGO_SCRATCH_WANDER_RMS_MM * varScale;
                const HighPrecType contrastCv = ALGO_SCRATCH_TRANSPORT_CONTRAST_CV * varScale;
                const HighPrecType trendSigma = ALGO_SCRATCH_TRANSPORT_TREND_SIGMA_LN * varScale;

                HighPrecType ptX[ALGO_SCRATCH_TRANSPORT_MAX_POINTS];
                HighPrecType ptY[ALGO_SCRATCH_TRANSPORT_MAX_POINTS];
                HighPrecType ptHw[ALGO_SCRATCH_TRANSPORT_MAX_POINTS];
                HighPrecType ptAm[ALGO_SCRATCH_TRANSPORT_MAX_POINTS];
                HighPrecType ptArc[ALGO_SCRATCH_TRANSPORT_MAX_POINTS];

                // Back from the newest event while an event could still be
                // alive; newest first, at most ALGO_SCRATCH_LIVE_MAX of them.
                int32_t      k     = kLast;
                HighPrecType start = (t >= e0 || eLast <= t) ? eLast : t + 1.0;
                int32_t      seen  = 0;

                while (start <= t && start > (t - horizon)
                       && seen < ALGO_SCRATCH_LIVE_MAX)
                {
                    const int32_t      kThis     = k;
                    const HighPrecType startThis = start;

                    // Step to the previous event before any `continue`. The
                    // horizon bounds the loop; `seen` counts DRAWN scratches.
                    start -= gapOf(k);
                    k--;

                    // Keyed on the ordinal only: the same scratch is the same
                    // scratch on every frame it appears in.
                    const uint64_t sc = AlgoDefectHash(
                        fieldSeed, 0, kThis, ALGO_DEFECT_TAG_SCRATCH_RUN);

                    const HighPrecType life = AlgoScratchLifetime(sc, lifetimeS);

                    // The abrader has already let go.
                    if ((startThis + life) <= t)
                        continue;

                    seen++;

                    // Fixed across the web: uniform over the film width.
                    const HighPrecType across = window.acrossMin
                        + AlgoRngUniform01(sc + 32u) * acrossExtent;

                    // Cut or burnish (provisional share, see the constant).
                    const HighPrecType polarity =
                        (AlgoRngUniform01(sc + 48u)
                         < ALGO_SCRATCH_TRANSPORT_CUT_SHARE) ? -1.0 : 1.0;

                    // This tramline's own width and contrast, log-normal.
                    const HighPrecType widthUm = defectLogNormal(
                        sc + 64u, ALGO_SCRATCH_TRANSPORT_WIDTH_UM,
                        ALGO_SCRATCH_WIDTH_SIGMA_LN);

                    const HighPrecType contrast = defectLogNormal(
                        sc + 80u, ALGO_SCRATCH_TRANSPORT_CONTRAST_MEDIAN,
                        ALGO_SCRATCH_TRANSPORT_CONTRAST_SIGMA_LN);

                    // Key of every along-mark process of this tramline.
                    const uint32_t sk = AlgoDefectSubSeed(sc);

                    // The stretch of WEB this abrader touched: from the frame
                    // position at its start time to that at its end time.
                    // Clipping to it makes the tramline END part way up the
                    // picture on the frame where the abrader lets go.
                    const HighPrecType runFrom = startThis * fps * framePitchMm;
                    const HighPrecType runTo   = (startThis + life) * fps * framePitchMm;

                    // One track - the main line or its companion - sampled on
                    // absolute along-film multiples of stepMm.
                    auto drawTrack = [&](const HighPrecType fromMm,
                                         const HighPrecType toMm,
                                         const HighPrecType offsetMm,
                                         const HighPrecType trackWidthUm,
                                         const HighPrecType trackContrast,
                                         const int32_t      fastStream) noexcept
                    {
                        const HighPrecType a0 = MAX_VALUE(
                            fromMm, window.alongMin - ALGO_SCRATCH_CLIP_MARGIN_MM);
                        const HighPrecType a1 = MIN_VALUE(
                            toMm,   window.alongMax + ALGO_SCRATCH_CLIP_MARGIN_MM);

                        if (a1 <= a0)
                            return;

                        int32_t n = 0;

                        auto addPoint = [&](const HighPrecType a) noexcept
                        {
                            const HighPrecType acr = across + offsetMm
                                + wanderRms * AlgoDefectSmoothNoise(
                                    sk, 0, ALGO_DEFECT_TAG_WANDER, a,
                                    ALGO_SCRATCH_WANDER_CORR_MM);

                            AlgoFilmToPixel(window, a, acr, ptX[n], ptY[n]);

                            ptHw[n] = trackWidthUm * umToHalfPx * defectCvFactor(
                                AlgoDefectSmoothNoise(sk, 1, ALGO_DEFECT_TAG_MODULATION,
                                    a, ALGO_SCRATCH_TRANSPORT_WIDTH_CORR_MM),
                                ALGO_SCRATCH_TRANSPORT_WIDTH_CV);

                            const HighPrecType fast = defectCvFactor(
                                AlgoDefectSmoothNoise(sk, fastStream,
                                    ALGO_DEFECT_TAG_MODULATION, a,
                                    ALGO_SCRATCH_TRANSPORT_CONTRAST_CORR_MM),
                                contrastCv);

                            const HighPrecType slow = std::exp(
                                trendSigma
                                * AlgoDefectSmoothNoise(sk, 3,
                                    ALGO_DEFECT_TAG_MODULATION, a,
                                    ALGO_SCRATCH_TRANSPORT_TREND_CORR_MM));

                            ptAm[n] = MIN_VALUE(trackContrast * fast * slow,
                                                ALGO_DEFECT_ALPHA_CAP);

                            // Absolute along-film position: the texture key.
                            ptArc[n] = a;

                            n++;
                        };

                        addPoint(a0);

                        const int32_t jFirst = static_cast<int32_t>(
                            std::floor(a0 / stepMm)) + 1;
                        const int32_t jLast  = static_cast<int32_t>(
                            std::ceil(a1 / stepMm)) - 1;

                        for (int32_t j = jFirst; j <= jLast; j++)
                        {
                            if (n >= ALGO_SCRATCH_TRANSPORT_MAX_POINTS - 1)
                                break;

                            addPoint(static_cast<HighPrecType>(j) * stepMm);
                        }

                        addPoint(a1);

                        DefectStroke st{};

                        st.x         = ptX;
                        st.y         = ptY;
                        st.hw        = ptHw;
                        st.am        = ptAm;
                        st.arc       = ptArc;
                        st.n         = n;
                        st.chroma[0] = 1.0;
                        st.chroma[1] = 1.0;
                        st.chroma[2] = 1.0;
                        st.polarity  = polarity;
                        st.share     = ALGO_BURNISH_SCATTER_SHARE;
                        st.texSigma  = texSigma;
                        st.texStepMm = texStepMm;
                        st.texSeed   = sk;
                        st.texStream = 10 + fastStream;

                        defectRasteriseStroke(st, pDstR, pDstG, pDstB,
                                              sizeX, sizeY, pitch, edgePx, ctx);
                    };

                    // The main track.
                    drawTrack(runFrom, runTo, 0.0, widthUm, contrast, 2);

                    // A companion track: the same abrader assembly (it shares
                    // the wander), a log-normal offset to either side, a fifth
                    // of the contrast, and on half of them only part of the run.
                    if (AlgoRngUniform01(sc + 96u) < ALGO_SCRATCH_COMPANION_PROB)
                    {
                        const HighPrecType side =
                            (AlgoRngUniform01(sc + 98u) < 0.5) ? -1.0 : 1.0;

                        const HighPrecType offMm = side * defectLogNormal(
                            sc + 97u, ALGO_SCRATCH_COMPANION_OFFSET_MM,
                            ALGO_SCRATCH_COMPANION_SIGMA_LN);

                        HighPrecType cFrom = runFrom;
                        HighPrecType cTo   = runTo;

                        if (AlgoRngUniform01(sc + 99u) < ALGO_SCRATCH_COMPANION_PARTIAL)
                        {
                            const HighPrecType frac =
                                0.2 + 0.6 * AlgoRngUniform01(sc + 100u);
                            const HighPrecType len = runTo - runFrom;

                            cFrom = runFrom + AlgoRngUniform01(sc + 101u)
                                            * len * (1.0 - frac);
                            cTo   = cFrom + frac * len;
                        }

                        const HighPrecType cWidthUm = defectLogNormal(
                            sc + 102u, ALGO_SCRATCH_TRANSPORT_WIDTH_UM,
                            ALGO_SCRATCH_WIDTH_SIGMA_LN);

                        drawTrack(cFrom, cTo, offMm, cWidthUm,
                                  contrast * ALGO_SCRATCH_COMPANION_RATIO, 4);
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        //  HANDLING SCRATCHES - single events, locked to the film.
        //
        //  Placed by the same clumped Cox process as the particulate classes,
        //  because a wipe damages a patch of film rather than a frame, and the
        //  patch travels with the web.
        //
        //  2026-10-20: BURSTS. A single wipe leaves several near-parallel marks
        //  (AlgoControl.hpp's description of the class, now with the collage's
        //  numbers): a placement becomes a burst of 2 - 4 marks with probability
        //  ALGO_SCRATCH_BURST_SEED_SHARE, the marks stepping sideways by
        //  log-normal spacings, sharing the polarity and the seed's heading to
        //  within 1 degree, each with its own length, width, contrast and walk.
        //  The seed rate is divided so that the areal density of MARKS still
        //  equals the control's. Length is log-normal; width and contrast vary
        //  smoothly along each mark, with a white texture on top.
        // ------------------------------------------------------------------
        if (handlingLevel > 0.0)
        {
            const HighPrecType lambda =
                handlingLevel * ALGO_SCRATCH_HANDLING_PER_MM2
                / (1.0 + ALGO_SCRATCH_BURST_MEAN_EXTRA * ALGO_SCRATCH_BURST_SEED_SHARE);

            defectWalkCells(window, ALGO_SCRATCH_MARGIN_MM, lambda, clumping,
                            fieldSeed, ALGO_DEFECT_TAG_SCRATCH_CELL,
                [&](const HighPrecType along,
                    const HighPrecType across,
                    const uint64_t     pc) noexcept
            {
                // Seed length: log-normal, truncated by remapping.
                const HighPrecType seedLenMm = AlgoDefectLogNormalTrunc(
                    pc + 2u, ALGO_SCRATCH_HANDLING_LEN_MEDIAN_MM,
                    ALGO_SCRATCH_HANDLING_LEN_SIGMA_LN,
                    ALGO_SCRATCH_HANDLING_LEN_MIN_MM,
                    ALGO_SCRATCH_HANDLING_LEN_MAX_MM);

                const HighPrecType polarity =
                    (AlgoRngUniform01(pc + 4u) < ALGO_SCRATCH_HANDLING_CUT_SHARE)
                        ? -1.0 : 1.0;

                // No transport lock: a handling mark is made by hand. The
                // measured 3.5:1 preference applies as a BIAS. defectOrientation
                // answers in the IMAGE frame and the walk steps in FILM
                // coordinates, so the transport rotation is subtracted.
                const HighPrecType seedHeading = defectOrientation(
                    window, pc + 8u, ALGO_SCRATCH_ORIENT_ALONG_SHARE)
                    - (window.transportAlongWidth
                        ? 0.0 : (0.25 * ALGO_DEFECT_TWO_PI));

                // Burst size: 1, or 2 / 3 / 4 marks.
                int32_t burst = 1;

                if (AlgoRngUniform01(pc + 12u) < ALGO_SCRATCH_BURST_SEED_SHARE)
                {
                    const HighPrecType u = AlgoRngUniform01(pc + 13u);

                    burst = (u < ALGO_SCRATCH_BURST_P2) ? 2
                          : ((u < ALGO_SCRATCH_BURST_P2 + ALGO_SCRATCH_BURST_P3)
                             ? 3 : ALGO_SCRATCH_BURST_MAX);
                }

                // All members step to the same side of the seed.
                const HighPrecType side =
                    (AlgoRngUniform01(pc + 14u) < 0.5) ? -1.0 : 1.0;

                const uint32_t burstSeed = AlgoDefectSubSeed(pc);

                const HighPrecType stepMm =
                    1.0 / static_cast<HighPrecType>(ALGO_SCRATCH_STEPS_PER_MM);

                HighPrecType offsetMm = 0.0;

                for (int32_t m = 0; m < burst; m++)
                {
                    // The seed keeps the placement's own stream; each further
                    // member has its own.
                    const uint64_t mc = (m == 0)
                        ? pc
                        : AlgoDefectHash(burstSeed, m, 0, ALGO_DEFECT_TAG_BURST);

                    HighPrecType heading = seedHeading;
                    HighPrecType lenMm   = seedLenMm;

                    if (m > 0)
                    {
                        offsetMm += defectLogNormal(
                            mc + 20u, ALGO_SCRATCH_BURST_SPACING_MM,
                            ALGO_SCRATCH_BURST_SPACING_SIGMA_LN);

                        heading += ALGO_SCRATCH_BURST_ANGLE_SIGMA_RAD
                                 * AlgoRngNormal(mc + 22u);

                        lenMm = CLAMP_VALUE(
                            seedLenMm * std::exp(ALGO_SCRATCH_BURST_LEN_SIGMA_LN
                                                 * AlgoRngNormal(mc + 24u)),
                            ALGO_SCRATCH_HANDLING_LEN_MIN_MM,
                            ALGO_SCRATCH_HANDLING_LEN_MAX_MM);
                    }

                    // Sideways from the seed, perpendicular to its heading.
                    HighPrecType alongCur  = along
                        - side * offsetMm * std::sin(seedHeading);
                    HighPrecType acrossCur = across
                        + side * offsetMm * std::cos(seedHeading);

                    // This mark's own width and contrast, log-normal.
                    const HighPrecType widthUm = defectLogNormal(
                        mc + 16u, ALGO_SCRATCH_HANDLING_WIDTH_UM,
                        ALGO_SCRATCH_WIDTH_SIGMA_LN);

                    const HighPrecType contrast = defectLogNormal(
                        mc + 18u, ALGO_SCRATCH_CONTRAST_MEDIAN,
                        ALGO_SCRATCH_CONTRAST_SIGMA_LN);

                    int32_t steps = static_cast<int32_t>(
                        lenMm * static_cast<HighPrecType>(
                            ALGO_SCRATCH_STEPS_PER_MM));

                    steps = CLAMP_VALUE(steps, 1, ALGO_SCRATCH_MAX_POINTS - 1);

                    const int32_t pointCount = steps + 1;

                    // ------------------------------------------------------
                    //  The straightness, turned into a walk: Lp = L / (12 (1 -
                    //  straightness)), the worm-like chain inversion derived at
                    //  ALGO_SCRATCH_STRAIGHTNESS, from the mark's OWN length so
                    //  that 0.98 means the same for a short mark and a long one.
                    // ------------------------------------------------------
                    const HighPrecType persistMm =
                        lenMm / (12.0 * (1.0 - ALGO_SCRATCH_STRAIGHTNESS));

                    const HighPrecType turnSigma = std::sqrt(stepMm / persistMm);

                    HighPrecType ctlX[ALGO_SCRATCH_MAX_POINTS];
                    HighPrecType ctlY[ALGO_SCRATCH_MAX_POINTS];

                    AlgoFilmToPixel(window, alongCur, acrossCur, ctlX[0], ctlY[0]);

                    for (int32_t i = 1; i < pointCount; i++)
                    {
                        heading += turnSigma * AlgoRngNormal(
                            mc + 64u + static_cast<uint64_t>(i));

                        alongCur  += stepMm * std::cos(heading);
                        acrossCur += stepMm * std::sin(heading);

                        AlgoFilmToPixel(window, alongCur, acrossCur,
                                        ctlX[i], ctlY[i]);
                    }

                    // The walk is the physics and the subdivision the drawing.
                    HighPrecType ptsX[ALGO_SCRATCH_MAX_DRAW];
                    HighPrecType ptsY[ALGO_SCRATCH_MAX_DRAW];
                    HighPrecType ptsHw[ALGO_SCRATCH_MAX_DRAW];
                    HighPrecType ptsAm[ALGO_SCRATCH_MAX_DRAW];
                    HighPrecType ptsArc[ALGO_SCRATCH_MAX_DRAW];

                    const int32_t drawCount = defectSubdivide(
                        ctlX, ctlY, pointCount, ptsX, ptsY);

                    defectArcMm(ptsX, ptsY, drawCount, pxPerMm, ptsArc);

                    const uint32_t sk = AlgoDefectSubSeed(mc + 30u);

                    for (int32_t i = 0; i < drawCount; i++)
                    {
                        ptsHw[i] = widthUm * umToHalfPx * defectCvFactor(
                            AlgoDefectSmoothNoise(sk, 0, ALGO_DEFECT_TAG_MODULATION,
                                ptsArc[i], ALGO_SCRATCH_HANDLING_CORR_MM),
                            ALGO_SCRATCH_HANDLING_WIDTH_CV);

                        ptsAm[i] = MIN_VALUE(contrast * defectCvFactor(
                            AlgoDefectSmoothNoise(sk, 1, ALGO_DEFECT_TAG_MODULATION,
                                ptsArc[i], ALGO_SCRATCH_HANDLING_CORR_MM),
                            ALGO_SCRATCH_HANDLING_CONTRAST_CV),
                            ALGO_DEFECT_ALPHA_CAP);
                    }

                    DefectStroke st{};

                    st.x         = ptsX;
                    st.y         = ptsY;
                    st.hw        = ptsHw;
                    st.am        = ptsAm;
                    st.arc       = ptsArc;
                    st.n         = drawCount;
                    st.chroma[0] = 1.0;
                    st.chroma[1] = 1.0;
                    st.chroma[2] = 1.0;
                    st.polarity  = polarity;
                    st.share     = ALGO_BURNISH_SCATTER_SHARE;
                    st.texSigma  = texSigma;
                    st.texStepMm = texStepMm;
                    st.texSeed   = sk;
                    st.texStream = 2;

                    defectRasteriseStroke(st, pDstR, pDstG, pDstB,
                                          sizeX, sizeY, pitch, edgePx, ctx);
                }
            });
        }

        return;
    }
}


// ---------------------------------------------------------------------------
//  Stage 9: DIR coupler lateral effects
// ---------------------------------------------------------------------------
void AlgoStage09_DirCoupler
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    AlgoType* RESTRICT       pScrDbar,
    AlgoType* RESTRICT       pScrDbarBlur,
    AlgoType* RESTRICT       pScrBlurA,
    AlgoType* RESTRICT       pScrBlurB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           pxPerMm,
    const AlgoFreqState&     freq
) noexcept
{
    const film::CouplerSpec& cp = profile.couplers;

    // User scale, floored at zero. A negative scale would invert both components
    // into a desaturating, softening filter, which is not a chemical state any
    // emulsion can be in.
    const AlgoType scale = MAX_VALUE(static_cast<AlgoType>(params.couplerScale),
                                     ALGO_ZERO);

    // Long-range cross-layer strength, and short-range adjacency strength.
    const AlgoType s = static_cast<AlgoType>(cp.strength)      * scale;
    const AlgoType e = static_cast<AlgoType>(cp.edge_strength) * scale;

    // Diffusion distances, micrometres on the film converted to pixels. This is
    // the resolution-independence mechanism: the same stock gives the same
    // physical diffusion distance whatever the render size.
    const AlgoType radiusPx = static_cast<AlgoType>(cp.radius_um)
                            * static_cast<AlgoType>(0.001) * pxPerMm;

    const AlgoType edgePx   = static_cast<AlgoType>(cp.edge_um)
                            * static_cast<AlgoType>(0.001) * pxPerMm;

    // The long-range term needs all three layers, so it is meaningless on a
    // monochrome stock: there is only one layer to push away from the mean, and the
    // mean of one thing is itself.
    const bool wantLong = (s > ALGO_ZERO)
                       && (false == profile.is_monochrome)
                       && (radiusPx > ALGO_ZERO);   // 2026-10-06: sub-pixel gate removed (owner)

    // The short-range term is within a single layer, so it applies to monochrome
    // stocks too - and is in fact the dominant coupler effect on them.
    const bool wantEdge = (e > ALGO_ZERO)
                       && (edgePx > ALGO_ZERO);

    // Copy first, unconditionally, then modify in place. Both components are
    // read-modify-write against a blurred version of the data, so a destination
    // that already holds the incoming densities is the natural starting state -
    // and it satisfies the retained-buffer policy even when neither component is
    // active.
    if ((false == wantLong) && (false == wantEdge))
    {
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
        return;
    }

    AlgoType* dstPlane[3] = { pDstR, pDstG, pDstB };

    // ----------------------------------------------------------------------
    //  BOTH components: evaluated together in the frequency domain
    //  (2026-10-06). film_sim's two steps are linear until the final floor:
    //
    //      x'_c  = x_c + s (x_c - G_r * mean(x))       = (1+s) x_c - s G_r * mean(x)
    //      x''_c = x'_c + e (x'_c - G_e * x'_c)        = ((1+e) - e G_e) * x'_c
    //
    //  so in the frequency domain, with X_c the record spectra:
    //
    //      X'_c  = (1+s) X_c - s T_r (X_0 + X_1 + X_2) / 3     (AlgoFreqMeanMix3)
    //      x''_c = irfft2( X'_c ((1+e) - e T_e) )              (AlgoFreqInverse)
    //
    //  -- six transforms instead of the eight the step-by-step form needs (one
    //  pair for the mean, one per record for the edge term), with the identical
    //  result up to rounding. Only one of the two active keeps the step form,
    //  which is cheaper there (two transforms for the long term alone).
    // ----------------------------------------------------------------------
    const bool fused = wantLong && wantEdge;
    if (fused)
    {
        AlgoFreqForward(freq, pSrcR, pitch, 0);
        AlgoFreqForward(freq, pSrcG, pitch, 1);
        AlgoFreqForward(freq, pSrcB, pitch, 2);

        AlgoFreqTransfer tLong;
        AlgoFreqSetGaussian(tLong, static_cast<HighPrecType>(cp.radius_um));
        AlgoFreqMeanMix3(freq, 1.0 + static_cast<HighPrecType>(s),
                         -static_cast<HighPrecType>(s), tLong);

        AlgoFreqTransfer tEdge;
        AlgoFreqTransferClear(tEdge);
        AlgoFreqSumAdd(tEdge.sumA, 0.0, 1.0 + static_cast<HighPrecType>(e));
        AlgoFreqSumAdd(tEdge.sumA, AlgoFreqGaussianA(static_cast<HighPrecType>(cp.edge_um)),
                       -static_cast<HighPrecType>(e));

        for (int32_t c = 0; c < 3; c++)
        {
            AlgoFreqInverse(freq, c, tEdge, dstPlane[c], pitch);

            // Floor at zero -- in this file the floor rides in the last term's
            // own pass, and on the fused path that pass is the inverse transform.
            AlgoType* RESTRICT pO = dstPlane[c];
            for (int32_t y = 0; y < sizeY; y++)
            {
                AlgoType* RESTRICT rO = pO + static_cast<std::ptrdiff_t>(y) * pitch;

                ALGO_VECTOR_HINT
                for (int32_t x = 0; x < sizeX; x++)
                    rO[x] = MAX_VALUE(rO[x], ALGO_ZERO);
            }
        }
    }
    else
    {
        // One component: modify a copy in place, step by step.
        AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);
    }

    // ----------------------------------------------------------------------
    //  Long-range component.
    //
    //  Each layer is pushed away from the LOCALLY BLURRED MEAN of all three. The
    //  mean is what makes it a colour effect: a neutral, where all three layers
    //  agree, sits at its own mean and is left alone, while a colour is driven
    //  further from it. Saturation rises and gamma does not.
    // ----------------------------------------------------------------------
    if (wantLong && (false == fused))
    {
        // Mean of the three densities, built into its own plane.
        const AlgoType third = ALGO_ONE / static_cast<AlgoType>(3.0);

        for (int32_t y = 0; y < sizeY; y++)
        {
            const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

            const AlgoType* RESTRICT pR = pDstR + off;
            const AlgoType* RESTRICT pG = pDstG + off;
            const AlgoType* RESTRICT pB = pDstB + off;

            AlgoType* RESTRICT pM = pScrDbar + off;

            ALGO_VECTOR_HINT
            for (int32_t x = 0; x < sizeX; x++)
                pM[x] = (pR[x] + pG[x] + pB[x]) * third;
        }

        // Blur it over the inhibitor's diffusion distance -- the same circular
        // convolution as the reference, so the edges wrap identically.
        {
            // film_sim: dbar_blur = apply_transfer(dbar, grid.gaussian(cp.radius_um)),
            // exact, in the frequency domain (AlgoFrequency.hpp, 2026-10-06).
            AlgoFreqTransfer t;
            AlgoFreqSetGaussian(t, static_cast<HighPrecType>(cp.radius_um));
            AlgoFreqFilterPlane(freq, pScrDbar, pScrDbarBlur, pitch, t);
        }

        for (int32_t c = 0; c < 3; c++)
        {
            AlgoType* RESTRICT pO = dstPlane[c];

            for (int32_t y = 0; y < sizeY; y++)
            {
                const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

                AlgoType* RESTRICT       rO = pO           + off;
                const AlgoType* RESTRICT rM = pScrDbarBlur + off;

                ALGO_VECTOR_HINT
                for (int32_t x = 0; x < sizeX; x++)
                    rO[x] = wantEdge
                          ? rO[x] + s * (rO[x] - rM[x])
                          : MAX_VALUE(rO[x] + s * (rO[x] - rM[x]), ALGO_ZERO);   // floor folded in when this is the last term
            }
        }
    }

    // ----------------------------------------------------------------------
    //  Short-range component.
    //
    //  Each layer pushed away from its own blurred self: unsharp masking in the
    //  density domain, arrived at by chemistry. Done per channel and after the
    //  long-range term, because the inhibitor released at the short scale responds
    //  to the density that the long-range redistribution has already produced.
    // ----------------------------------------------------------------------
    if (wantEdge && (false == fused))
    {
        for (int32_t c = 0; c < 3; c++)
        {
            AlgoType* RESTRICT pO = dstPlane[c];

            // Blurred copy of this channel. pScrDbarBlur is reused as the
            // destination: the long-range term has finished with it, and the two
            // components never need their blurs at the same time.
            {
                // film_sim: blurred = apply_transfer(dens[:, :, c],
                // grid.gaussian(cp.edge_um)), exact, in the frequency domain.
                AlgoFreqTransfer t;
                AlgoFreqSetGaussian(t, static_cast<HighPrecType>(cp.edge_um));
                AlgoFreqFilterPlane(freq, pO, pScrDbarBlur, pitch, t);
            }

            for (int32_t y = 0; y < sizeY; y++)
            {
                const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

                AlgoType* RESTRICT       rO = pO           + off;
                const AlgoType* RESTRICT rB = pScrDbarBlur + off;

                ALGO_VECTOR_HINT
                for (int32_t x = 0; x < sizeX; x++)
                    rO[x] = MAX_VALUE(rO[x] + e * (rO[x] - rB[x]), ALGO_ZERO);   // floor folded in, 2026-10-04
            }
        }
    }

    // ----------------------------------------------------------------------
    //  Floor at zero.
    //
    //  Both components are difference terms, so either can drive a value below
    //  zero on the light side of a hard edge. A negative optical density has no
    //  physical meaning - it would be a material that emits light - and stage 14
    //  raises ten to its negative, which would produce a transmittance above one.
    //  This is a physical floor, not a display clamp, so it does not violate the
    //  single-final-clamp rule.
    // ----------------------------------------------------------------------
    // 2026-10-04: the floor at zero is applied inside the last term's own
    // pass above (edge term if present, else the long-range term; on the fused
    // path right after the inverse transform), not here.

    // pScrBlurA / pScrBlurB are not needed since 2026-10-06: both blurs are the
    // frequency-domain filter, whose spectrum lives in the arena. They stay in
    // the signature so the stage's scratch contract does not change.
    (void)pScrBlurA;
    (void)pScrBlurB;

    return;
}


// ---------------------------------------------------------------------------
//  Sub-stage 9b: negative-side defects.
//
//  Three particulate classes and one abrasion class, all part of the negative:
//  fine dust, coarse debris, hair and fibres, and scratches.
//
//  The copy comes first and unconditionally. The retained-buffer policy gives
//  every stage its own destination, so returning without writing would leave stale
//  contents for every stage after this one; and the marks are then accumulated
//  in place, which is what lets four independent generators share one output
//  without any of them knowing about the others.
// ---------------------------------------------------------------------------
void AlgoStage09b_NegativeDefects
(
    const AlgoType* RESTRICT pSrcR,
    const AlgoType* RESTRICT pSrcG,
    const AlgoType* RESTRICT pSrcB,
    AlgoType* RESTRICT       pDstR,
    AlgoType* RESTRICT       pDstG,
    AlgoType* RESTRICT       pDstB,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoControls&      params,
    const AlgoType           negWidthMm,
    const AlgoType           negHeightMm,
    const AlgoType           framePitchMm,
    const AlgoType           pxPerMm,
    const int32_t            frameIndex,
    const AlgoType           frameRate,
    const uint32_t           seed
    ,
    AlgoType*                pDefR,
    AlgoType*                pDefG,
    AlgoType*                pDefB
) noexcept
{
    // The unconditional copy. Every path below accumulates on top of it.
    AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    // 2026-10-20b: the additive-defect plane starts empty, so every early
    // return below leaves it correct (no defects).
    const bool wantDef = (pDefR != nullptr) && (pDefG != nullptr) && (pDefB != nullptr);

    if (wantDef)
    {
        AlgoType* q[3] = { pDefR, pDefG, pDefB };

        for (int32_t c = 0; c < 3; c++)
            for (int32_t y = 0; y < sizeY; y++)
            {
                AlgoType* RESTRICT rQ = q[c] + static_cast<std::ptrdiff_t>(y) * pitch;

                for (int32_t x = 0; x < sizeX; x++)
                    rQ[x] = ALGO_ZERO;
            }
    }

    // ----------------------------------------------------------------------
    //  Both gates, cheapest first.
    //
    //  The master flag is checked before anything else is even read, so a clean
    //  render pays one branch for the whole defect subsystem. The strength scale
    //  is the second gate: at zero the user has asked for no damage and the class
    //  levels below are irrelevant.
    // ----------------------------------------------------------------------
    if (false == params.filmDamageEnabled)
        return;

    const FilmDamage& dmg = params.damage;

    const HighPrecType strength =
        MAX_VALUE(static_cast<HighPrecType>(dmg.damageStrength), 0.0);

    if (strength <= 0.0)
        return;

    // ----------------------------------------------------------------------
    //  Effective level per class.
    //
    //  Each user level is scaled by the master strength. Negative levels are
    //  floored rather than trusted: the caller is documented as pre-validating,
    //  but a negative rate would reach the Poisson sampler as a negative mean and
    //  the whole frame would depend on how that failed.
    // ----------------------------------------------------------------------
    const HighPrecType dustLevel =
        MAX_VALUE(static_cast<HighPrecType>(dmg.dustLevel),   0.0) * strength;
    const HighPrecType debrisLevel =
        MAX_VALUE(static_cast<HighPrecType>(dmg.debrisLevel), 0.0) * strength;
    const HighPrecType fibreLevel =
        MAX_VALUE(static_cast<HighPrecType>(dmg.fibreLevel),  0.0) * strength;

    // The two scratch controls, which until now were in AlgoControl.hpp's
    // "Unconsumed" list and read by nothing. Same treatment as the three above:
    // floored, then scaled by the master strength.
    // 2026-10-20b: scratchTransport is NEW EVENTS PER MINUTE. Scaled by the
    // master strength like every level, then raised to the floor rate at which
    // every inter-event gap lies in 5 - 20 s (ScratchTransportFloor).
    HighPrecType scratchTransportLevel =
        MAX_VALUE(static_cast<HighPrecType>(dmg.scratchTransport), 0.0) * strength;

    if (scratchTransportLevel > 0.0)
        scratchTransportLevel = MAX_VALUE(scratchTransportLevel,
                                          static_cast<HighPrecType>(ScratchTransportFloor));
    const HighPrecType scratchHandlingLevel =
        MAX_VALUE(static_cast<HighPrecType>(dmg.scratchHandling),  0.0) * strength;

    // Nothing requested, nothing to do. Worth its own test: the cell-walking loops
    // below each walk a grid over the whole window before discovering their rate
    // is zero, and for fibres that grid extends 25 mm past the frame in every
    // direction.
    // 2026-10-10: the three classes that had no reader until this date,
    // floored and scaled exactly like the five above. colourVeil is a DENSITY
    // (AlgoControl.hpp item 4), so its product with the strength is the offset.
    const HighPrecType processingLevel =
        MAX_VALUE(static_cast<HighPrecType>(dmg.processingQuality), 0.0) * strength;
    const HighPrecType dryingLevel =
        MAX_VALUE(static_cast<HighPrecType>(dmg.dryingMarks),       0.0) * strength;
    const HighPrecType veilLevel =
        MAX_VALUE(static_cast<HighPrecType>(dmg.colourVeil),        0.0) * strength;

    if (dustLevel <= 0.0 && debrisLevel <= 0.0 && fibreLevel <= 0.0
        && scratchTransportLevel <= 0.0 && scratchHandlingLevel <= 0.0
        && processingLevel <= 0.0 && dryingLevel <= 0.0 && veilLevel <= 0.0)
        return;

    // ----------------------------------------------------------------------
    //  Clumpiness.
    //
    //  Passed through to the intensity field as given. Zero means a uniform
    //  Poisson scatter, which is available deliberately - it is the wrong look,
    //  but it is exactly the control needed to verify that the placement machinery
    //  is unbiased.
    // ----------------------------------------------------------------------
    const HighPrecType clumping =
        MAX_VALUE(static_cast<HighPrecType>(dmg.dirtClumping), 0.0);

    // ----------------------------------------------------------------------
    //  The window this frame occupies on the film.
    //
    //  Built from the gauge geometry, so the transport axis is DERIVED rather than
    //  configured: the frame pitch is the transport-axis extent plus the
    //  interframe gap, which identifies the axis without any per-format table.
    //  Everything downstream - orientation bias, the 90 degree rotation, the
    //  advance from frame to frame - follows from this one object.
    // ----------------------------------------------------------------------
    AlgoFilmWindow window = AlgoMakeFilmWindow(
        negWidthMm, negHeightMm, framePitchMm, sizeX, sizeY, frameIndex);

    // ----------------------------------------------------------------------
    //  AN UNPERFORATED FORMAT MUST STILL ADVANCE IN A MOVING CLIP - 2026-09-04.
    //
    //  ⚠ THIS IS THE SECOND OF THE TWO REASONS THE DEFECT LAYER LOOKED LIKE A
    //  DIRTY MONITOR, AND IT IS THE ONE THAT FREEZES EVERYTHING AT ONCE. Sheet,
    //  pack and instant formats report framePitchMm = 0, and AlgoMakeFilmWindow
    //  honours that exactly as its comment says it does: the along-film origin is
    //  frameIndex * pitch, so with pitch zero EVERY FRAME LANDS ON THE SAME PATCH
    //  OF FILM. Since this stage keys dust, debris and fibres on FILM coordinates,
    //  every particle then sits at the same place in the same shape with the same
    //  opacity for the entire clip. Thirteen stocks in the database default to
    //  such a format - the 4x5 sheet stocks, medium format, and the three Polaroid
    //  entries - and any user who selects one gets a completely frozen defect
    //  layer no matter what the levels say.
    //
    //  ⚠ AND THE HELPER IS NOT WRONG. "A sheet is one piece of film, and rendering
    //  the same sheet twice must give the same defects" is correct for a STILL.
    //  It stops being correct the moment the host asks for a sequence: a clip
    //  graded with a 4x5 look is not one sheet held up for ten seconds, it is a
    //  succession of exposures. So the geometry helper keeps its literal
    //  behaviour - including anisotropy zero, which is right, a sheet has no
    //  transport direction - and only the ORIGIN is advanced, here, by the stage
    //  that knows it is rendering a sequence.
    //
    //  One full along-extent per frame, so consecutive windows do not overlap at
    //  all and no particle can survive into the next frame. A fraction of an
    //  extent would give partial overlap, which is worse than either extreme: a
    //  speck that slides a little and then vanishes reads as a tracking error.
    // ----------------------------------------------------------------------
    if (framePitchMm <= ALGO_FILM_MIN_PITCH_MM)
    {
        const HighPrecType step =
            (window.alongMax - window.alongMin)
            * static_cast<HighPrecType>(frameIndex);

        window.alongMin += step;
        window.alongMax += step;
    }

    // ----------------------------------------------------------------------
    //  Edge transition width for every particle in this frame.
    //
    //  The system point-spread function in pixels, floored at half a pixel. The
    //  floor is what antialiases sub-pixel dust: at a coarse raster the physical
    //  PSF is a small fraction of a pixel, and a transition narrower than half a
    //  pixel turns every speck into a hard square.
    // ----------------------------------------------------------------------
    const HighPrecType edgePx = MAX_VALUE(
        (ALGO_DEFECT_EDGE_UM / 1000.0) * static_cast<HighPrecType>(pxPerMm),
        ALGO_DEFECT_EDGE_MIN_PX);

    // ----------------------------------------------------------------------
    //  Roll seed.
    //
    //  damageSeed identifies the roll and the stage salt separates this stage's
    //  streams from every other stage's. Deliberately NOT mixed with frameIndex:
    //  the field and the placement cells are keyed on film coordinates, so the
    //  frame number enters only through which part of the film the window covers.
    //  Mixing the frame in here would re-roll every particle each frame and turn
    //  film-locked dirt into boiling noise.
    // ----------------------------------------------------------------------
    const uint32_t fieldSeed = static_cast<uint32_t>(dmg.damageSeed) ^ seed;

    // ----------------------------------------------------------------------
    //  Does this stock have colour records at all?
    //
    //  A monochrome emulsion holds one silver image and prints through one dye, so
    //  it cannot record what colour a particle was - only how much light it
    //  blocked. Particles on such a stock must therefore be exactly neutral, and
    //  59 of the 142 stocks in the database are monochrome, so this is the common
    //  case rather than an exception.
    // ----------------------------------------------------------------------
    const bool monochrome = profile.is_monochrome;

    // ----------------------------------------------------------------------
    //  2026-10-20: THE READER-OPTICS CONTEXT (FilmDamage::defectOptics).
    //
    //  OFF (the default) every particulate and scratch write is the additive
    //  law it always was. ON, each defect is split into an absorbing and a
    //  scattering part and read through the reader's specular fraction
    //  (scannerSpecular), anchored at ALGO_DEFECT_OPTICS_S_REF, and on a stock
    //  with Callier Q > 1 it is composed in the reader's density domain so that
    //  stage 12b steepens the silver and not the dirt. AlgoDefectOptics.hpp.
    //  The processing, veil and drying classes above are image density and are
    //  NOT routed through it.
    // ----------------------------------------------------------------------
    const AlgoDefectOpticsCtx opticsCtx = AlgoDefectOpticsMake(
        dmg.defectOptics, profile,
        static_cast<HighPrecType>(params.scannerSpecular),
        ALGO_DEFECT_OPTICS_S_REF,
        static_cast<HighPrecType>(dmg.defectVisibility),
        ALGO_CLEAR_BASE_DENSITY);

    // ----------------------------------------------------------------------
    //  2026-10-10: PROCESSING, VEIL AND DRYING - AlgoProcessingDefects.hpp.
    //
    //  In the order the film met them, which is not cosmetic here because the
    //  development classes MULTIPLY net density: development (mottle, air
    //  bells) first, then the dye-record veil, then drying residue - and only
    //  after drying the embedded particulates below, which are pressed into
    //  the tacky gelatin during that drying and so are not scaled by a
    //  development non-uniformity that happened before they arrived. Each
    //  class draws from its own stream (eRNG_PROCESS_DEFECTS, eRNG_COLOUR_VEIL,
    //  eRNG_DRYING_MARKS), so a level change moves no other class.
    // ----------------------------------------------------------------------
    if (processingLevel > 0.0)
        AlgoProcessingDefects(window, processingLevel, fieldSeed,
                              static_cast<HighPrecType>(profile.curves.r.dmin),
                              static_cast<HighPrecType>(profile.curves.g.dmin),
                              static_cast<HighPrecType>(profile.curves.b.dmin),
                              pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    if (veilLevel > 0.0)
        AlgoColourVeil(veilLevel, fieldSeed, monochrome,
                       pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    if (dryingLevel > 0.0)
        AlgoDryingMarks(window, dryingLevel, fieldSeed,
                        pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    // ----------------------------------------------------------------------
    //  The three classes, in ascending order of rarity.
    //
    //  Order is not significant to the result - density accumulates additively and
    //  addition commutes - but each class draws from its own stream, so adjusting
    //  one level leaves the others' particles exactly where they were.
    // ----------------------------------------------------------------------
    // 2026-10-20b: snapshot BEFORE the particulate and abrasion classes, so the
    // additive defect density can be handed to the grain stage (end of stage).
    if (wantDef)
        AlgoCopyImage(pDstR, pDstG, pDstB, pDefR, pDefG, pDefB, sizeX, sizeY, pitch);

    if (dustLevel > 0.0)
        defectDust(window, dustLevel, clumping, fieldSeed, monochrome,
                   pDstR, pDstG, pDstB, sizeX, sizeY, pitch, edgePx,
                   opticsCtx);

    if (debrisLevel > 0.0)
        defectDebris(window, debrisLevel, clumping, fieldSeed, monochrome,
                     pDstR, pDstG, pDstB, sizeX, sizeY, pitch, edgePx,
                   opticsCtx);

    if (fibreLevel > 0.0)
        defectFibres(window, fibreLevel, clumping, fieldSeed, monochrome,
                     pDstR, pDstG, pDstB, sizeX, sizeY, pitch, edgePx,
                   opticsCtx);

    // ----------------------------------------------------------------------
    //  The abrasion class.
    //
    //  One generator for both populations, because they are the same mark drawn
    //  by two different mechanisms, and the gate is the same one every other
    //  class here has: at zero the generator is not entered at all, so a user
    //  who wants no scratches pays one comparison. Each population is then gated
    //  again on its own control inside, so raising one does not start the other.
    //
    //  ⚠ THE RUN LENGTH COMES FROM THE STOCK, NOT FROM A CONTROL.
    //  TemporalSpec::scratch_persistence_frames is the mean run of a running
    //  scratch in frames, populated on all 191 stocks, and this is its only
    //  reader. It is an ERA property - how long a machine fault lasted on the
    //  equipment of the day - so it belongs beside the weave amplitude and the
    //  dirt rate rather than on a slider, and the control expresses only how many
    //  such scratches there are.
    // ----------------------------------------------------------------------
    if (scratchTransportLevel > 0.0 || scratchHandlingLevel > 0.0)
        defectScratches(window, scratchTransportLevel, scratchHandlingLevel,
                        clumping, fieldSeed, frameIndex,
                        // 2026-10-20b: mean lifetime in SECONDS - the control,
                        // or the stock's run length at the nominal 24 fps.
                        (dmg.scratchLifetime > 0.0)
                            ? static_cast<HighPrecType>(dmg.scratchLifetime)
                            : static_cast<HighPrecType>(
                                  profile.temporal.scratch_persistence_frames)
                              / ALGO_SCRATCH_NOMINAL_FPS,
                        MAX_VALUE(static_cast<HighPrecType>(frameRate), 1.0),
                        static_cast<HighPrecType>(dmg.defectVariability),
                        static_cast<HighPrecType>(framePitchMm),
                        pDstR, pDstG, pDstB, sizeX, sizeY, pitch, edgePx,
                   opticsCtx);

    // ----------------------------------------------------------------------
    //  2026-10-20b: the additive defect density this stage added, for stage 11
    //  - what the particulate and abrasion classes put ON the image. A removal
    //  (cut, pinhole) is real loss of emulsion and stays out (floored at zero),
    //  so grain follows the reduced density there.
    // ----------------------------------------------------------------------
    if (wantDef)
    {
        const AlgoType* d[3] = { pDstR, pDstG, pDstB };
        AlgoType*       q[3] = { pDefR, pDefG, pDefB };

        for (int32_t c = 0; c < 3; c++)
            for (int32_t y = 0; y < sizeY; y++)
            {
                const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;
                const AlgoType* RESTRICT rD = d[c] + off;
                AlgoType* RESTRICT       rQ = q[c] + off;

                for (int32_t x = 0; x < sizeX; x++)
                    rQ[x] = MAX_VALUE(rD[x] - rQ[x], ALGO_ZERO);
            }
    }

    // ----------------------------------------------------------------------
    //  Read by nothing in this stage, and that is the honest state of it.
    //
    //  profile is read for is_monochrome and, since the scratch class landed,
    //  for TemporalSpec::scratch_persistence_frames, and since 2026-10-10 for
    //  the three curve D_min values the development classes scale net density
    //  about. Its AgingSpec is still not.
    //  That structure's dust_area_ppm, mottle_amplitude and
    //  scratch_rate_base_per_m are the era baseline these levels were meant to
    //  multiply - but every one of the stocks currently ships that structure all
    //  zero, documented as "fresh". Multiplying by it would therefore silence the
    //  whole defect layer on every stock, so the levels are absolute for now:
    //  dustLevel 1.0 means the measured central density, whatever the stock. When
    //  AgingSpec is populated it becomes an additive era term rather than a
    //  multiplier, so that a fresh stock keeps behaving exactly as it does today.
    //
    //  ⚠ THE SCRATCH CLASS IS THE FIRST HERE WITH AN ERA TERM ALREADY IN FORCE,
    //  and it is in force only because it reads TemporalSpec, which IS populated
    //  on all 191 stocks - unlike AgingSpec. So the mean run length of a tramline
    //  genuinely differs between a 1930s nitrate print and a modern negative
    //  today, while the amount of dust does not yet. That asymmetry is real and
    //  is a property of which structure got populated, not a design.
    //
    //  frameRate: since 2026-10-20b the transport-scratch EVENTS are scheduled
    //  in seconds of screen time (t = frameIndex / frameRate), so this stage
    //  reads it; the stock's run length (frames) is converted at the nominal
    //  24 fps only when scratchLifetime is 0. The film-locked classes (dust,
    //  debris, fibres, handling scratches) stay in film coordinates: every frame
    //  is a new piece of film whatever the frame rate.
    //  negWidthMm and negHeightMm are consumed through the window; pxPerMm
    //  likewise, and directly for the edge width.
    // ----------------------------------------------------------------------

    return;
}


// ---------------------------------------------------------------------------
//  Sub-stage 9c: bromide drag -- AVX2.
//
//  ⚠ THE RECURSION IS WHAT MAKES THIS VECTORISABLE AT ALL, AND ONLY BECAUSE IT
//  RUNS DOWN COLUMNS. A first-order IIR cannot be vectorised ALONG its own axis
//  without an in-register parallel prefix scan. Here the axis is the frame
//  HEIGHT, so eight ADJACENT COLUMNS are eight INDEPENDENT recursions sharing
//  one contiguous load: the eight accumulator states live in one register, the
//  step is a single fmadd, and nothing is ever shuffled across lanes. That is
//  the entire reason the generator refuses axis 1 instead of accepting it and
//  falling back to scalar on the vector path -- two twins that agree only where
//  they are tested is the failure this project keeps designing out.
//
//  Same four passes, same order, same arithmetic as the scalar twin. Every
//  access is unaligned and every partial vector is masked, per the file header.
// ---------------------------------------------------------------------------
bool AlgoStage09c_BromideDrag
(
    AlgoType* RESTRICT       pR,
    AlgoType* RESTRICT       pG,
    AlgoType* RESTRICT       pB,
    AlgoType* RESTRICT       pScrSrc,
    AlgoType* RESTRICT       pScrAcc,
    const int32_t            sizeX,
    const int32_t            sizeY,
    const int32_t            pitch,
    const film::FilmProfile& profile,
    const AlgoType           pxPerMm
) noexcept
{
    const film::BromideDragSpec& drag = profile.processing.bromide_drag;

    if (!drag.hasData() || pxPerMm <= 0.0f || sizeX <= 0 || sizeY <= 0)
    {
        return false;
    }

    const float dminR = profile.curves.r.dmin;
    const float dminG = profile.curves.g.dmin;
    const float dminB = profile.curves.b.dmin;

    const float refNet = ((profile.curves.r.dmax() - dminR)
                        + (profile.curves.g.dmax() - dminG)
                        + (profile.curves.b.dmax() - dminB)) * (1.0f / 3.0f);
    if (refNet <= 0.0f)
    {
        return false;
    }

    const int32_t  full = sizeX / ALGO_AVX2_LANES_LOCAL;
    const int32_t  tail = sizeX - full * ALGO_AVX2_LANES_LOCAL;
    const __m256i  mask = algoTailMaskLocal(tail);

    const __m256 vDminR = _mm256_set1_ps(dminR);
    const __m256 vDminG = _mm256_set1_ps(dminG);
    const __m256 vDminB = _mm256_set1_ps(dminB);
    const __m256 vScale = _mm256_set1_ps((1.0f / 3.0f) / refNet);
    const __m256 vZero  = _mm256_setzero_ps();
    const __m256 vOne   = _mm256_set1_ps(1.0f);

    const bool reverse = profile.isReversal();

    // ----------------------------------------------------------------------
    //  Pass 1: the source field. Eight pixels of three channels per step.
    // ----------------------------------------------------------------------
    for (int32_t y = 0; y < sizeY; ++y)
    {
        const float* RESTRICT rowR = pR + static_cast<size_t>(y) * pitch;
        const float* RESTRICT rowG = pG + static_cast<size_t>(y) * pitch;
        const float* RESTRICT rowB = pB + static_cast<size_t>(y) * pitch;
        float* RESTRICT       rowS = pScrSrc + static_cast<size_t>(y) * pitch;

        int32_t x = 0;
        for (int32_t v = 0; v < full; ++v, x += ALGO_AVX2_LANES_LOCAL)
        {
            __m256 e = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_sub_ps(_mm256_loadu_ps(rowR + x), vDminR),
                    _mm256_sub_ps(_mm256_loadu_ps(rowG + x), vDminG)),
                _mm256_sub_ps(_mm256_loadu_ps(rowB + x), vDminB));
            e = _mm256_mul_ps(e, vScale);
            e = _mm256_min_ps(_mm256_max_ps(e, vZero), vOne);
            if (reverse)
            {
                e = _mm256_sub_ps(vOne, e);
            }
            _mm256_storeu_ps(rowS + x, e);
        }
        if (tail)
        {
            __m256 e = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_sub_ps(_mm256_maskload_ps(rowR + x, mask), vDminR),
                    _mm256_sub_ps(_mm256_maskload_ps(rowG + x, mask), vDminG)),
                _mm256_sub_ps(_mm256_maskload_ps(rowB + x, mask), vDminB));
            e = _mm256_mul_ps(e, vScale);
            e = _mm256_min_ps(_mm256_max_ps(e, vZero), vOne);
            if (reverse)
            {
                e = _mm256_sub_ps(vOne, e);
            }
            _mm256_maskstore_ps(rowS + x, mask, e);
        }
    }

    // ----------------------------------------------------------------------
    //  Pass 2: eight independent one-pole recursions per vector.
    //
    //  ⚠ THE COEFFICIENT IS COMPUTED IN HighPrecType AND THEN NARROWED, exactly
    //  as the scalar twin does it, and not with a float exp. It is two scalar
    //  operations for the whole frame, so the accuracy is free -- and a
    //  different rounding here would put the two twins on different poles,
    //  whose error COMPOUNDS down the column rather than staying local.
    // ----------------------------------------------------------------------
    const AlgoType pitchMm = 1.0f / pxPerMm;
    const float    aCoef   = static_cast<float>(
        std::exp(-static_cast<HighPrecType>(pitchMm)
                 / static_cast<HighPrecType>(drag.length_mm)));
    const float    bCoef   = 1.0f - aCoef;
    const __m256   vA      = _mm256_set1_ps(aCoef);
    const __m256   vB      = _mm256_set1_ps(bCoef);

    const bool    forward = (drag.direction >= 0);
    const int32_t yFirst  = forward ? 0 : (sizeY - 1);
    const int32_t yStep   = forward ? 1 : -1;

    {
        const float* RESTRICT rowS =
            pScrSrc + static_cast<size_t>(yFirst) * pitch;
        float* RESTRICT rowA = pScrAcc + static_cast<size_t>(yFirst) * pitch;
        int32_t x = 0;
        for (int32_t v = 0; v < full; ++v, x += ALGO_AVX2_LANES_LOCAL)
        {
            _mm256_storeu_ps(rowA + x, _mm256_loadu_ps(rowS + x));
        }
        if (tail)
        {
            _mm256_maskstore_ps(rowA + x, mask,
                                _mm256_maskload_ps(rowS + x, mask));
        }
    }
    for (int32_t i = 1; i < sizeY; ++i)
    {
        const int32_t y    = yFirst + i * yStep;
        const int32_t yPrv = y - yStep;
        const float* RESTRICT prvS = pScrSrc + static_cast<size_t>(yPrv) * pitch;
        const float* RESTRICT prvA = pScrAcc + static_cast<size_t>(yPrv) * pitch;
        float* RESTRICT       rowA = pScrAcc + static_cast<size_t>(y) * pitch;

        int32_t x = 0;
        for (int32_t v = 0; v < full; ++v, x += ALGO_AVX2_LANES_LOCAL)
        {
            const __m256 acc = _mm256_fmadd_ps(vA, _mm256_loadu_ps(prvA + x),
                                   _mm256_mul_ps(vB, _mm256_loadu_ps(prvS + x)));
            _mm256_storeu_ps(rowA + x, acc);
        }
        if (tail)
        {
            const __m256 acc = _mm256_fmadd_ps(
                vA, _mm256_maskload_ps(prvA + x, mask),
                _mm256_mul_ps(vB, _mm256_maskload_ps(prvS + x, mask)));
            _mm256_maskstore_ps(rowA + x, mask, acc);
        }
    }

    // ----------------------------------------------------------------------
    //  Passes 3 and 4, fused.
    // ----------------------------------------------------------------------
    const __m256 vStrength = _mm256_set1_ps(drag.strength);
    const __m256 vCeiling  = _mm256_set1_ps(ALGO_BROMIDE_MAX_REMOVED);

    for (int32_t y = 0; y < sizeY; ++y)
    {
        const float* RESTRICT rowA = pScrAcc + static_cast<size_t>(y) * pitch;
        float* RESTRICT       rowR = pR + static_cast<size_t>(y) * pitch;
        float* RESTRICT       rowG = pG + static_cast<size_t>(y) * pitch;
        float* RESTRICT       rowB = pB + static_cast<size_t>(y) * pitch;

        int32_t x = 0;
        for (int32_t v = 0; v < full; ++v, x += ALGO_AVX2_LANES_LOCAL)
        {
            __m256 r = _mm256_mul_ps(_mm256_loadu_ps(rowA + x), vStrength);
            r = _mm256_min_ps(_mm256_max_ps(r, vZero), vCeiling);
            const __m256 keep = _mm256_sub_ps(vOne, r);
            _mm256_storeu_ps(rowR + x, _mm256_fmadd_ps(
                _mm256_sub_ps(_mm256_loadu_ps(rowR + x), vDminR), keep, vDminR));
            _mm256_storeu_ps(rowG + x, _mm256_fmadd_ps(
                _mm256_sub_ps(_mm256_loadu_ps(rowG + x), vDminG), keep, vDminG));
            _mm256_storeu_ps(rowB + x, _mm256_fmadd_ps(
                _mm256_sub_ps(_mm256_loadu_ps(rowB + x), vDminB), keep, vDminB));
        }
        if (tail)
        {
            __m256 r = _mm256_mul_ps(_mm256_maskload_ps(rowA + x, mask),
                                     vStrength);
            r = _mm256_min_ps(_mm256_max_ps(r, vZero), vCeiling);
            const __m256 keep = _mm256_sub_ps(vOne, r);
            _mm256_maskstore_ps(rowR + x, mask, _mm256_fmadd_ps(
                _mm256_sub_ps(_mm256_maskload_ps(rowR + x, mask), vDminR),
                keep, vDminR));
            _mm256_maskstore_ps(rowG + x, mask, _mm256_fmadd_ps(
                _mm256_sub_ps(_mm256_maskload_ps(rowG + x, mask), vDminG),
                keep, vDminG));
            _mm256_maskstore_ps(rowB + x, mask, _mm256_fmadd_ps(
                _mm256_sub_ps(_mm256_maskload_ps(rowB + x, mask), vDminB),
                keep, vDminB));
        }
    }

    return true;
}
