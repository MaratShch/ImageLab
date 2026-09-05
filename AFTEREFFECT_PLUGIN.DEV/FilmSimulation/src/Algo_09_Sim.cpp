// ---------------------------------------------------------------------------
//  Algo_09_Sim.cpp   --   AVX2
//
//  Same filename, same function names, same prototypes as the scalar build.
//  ALL ARITHMETIC IS FLOAT32; the scalar path remains the reference.
//
//  VECTORISED: the coupler's pointwise terms and the non-negative floor.
//  The 9b particulate rasterisers are left scalar on purpose - they scatter into tiny
//  bounding boxes, a 20 um speck being one to three pixels, so there is nothing to
//  fill a register with, and together they are under two per cent of the frame.
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
//      AlgoStage09b_NegativeDefects  embedded particulate: dust, debris, fibres
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
#include "AlgoBromideDrag.hpp"


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
    // ----------------------------------------------------------------------
    //  Two pi, for the angular harmonics and the orientation draws.
    // ----------------------------------------------------------------------
    constexpr HighPrecType ALGO_DEFECT_TWO_PI = 6.283185307179586476925286766559;

    // ----------------------------------------------------------------------
    //  Micrometres per millimetre. Sizes are measured in micrometres and the
    //  film coordinate system is in millimetres, so the conversion appears often
    //  enough to be worth naming.
    // ----------------------------------------------------------------------
    constexpr HighPrecType ALGO_DEFECT_UM_PER_MM = 1000.0;


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
    };


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
    //  Orientation for an elongated particle, in the IMAGE frame.
    //
    //  The measured population favours the transport axis roughly three to one.
    //  The bias is expressed in FILM coordinates - along the film against across
    //  it - and then rotated into the image frame by asking the window which
    //  image axis the film runs along. That is the whole reason the film
    //  coordinate system exists: this function contains no per-format code, and
    //  the same draw comes out horizontal on a 35 mm still and vertical on every
    //  cine gauge.
    // ----------------------------------------------------------------------
    inline HighPrecType defectOrientation
    (
        const AlgoFilmWindow& window,
        const uint64_t        counter
    ) noexcept
    {
        const HighPrecType uPick   = AlgoRngUniform01(counter);
        const HighPrecType uSpread = AlgoRngUniform01(counter + 1u);
        const HighPrecType uFree   = AlgoRngUniform01(counter + 2u);

        // Anisotropy is zero for sheet film, which has no transport direction, so
        // its orientation is drawn isotropically and the bias never applies.
        const HighPrecType bias = ALGO_DEFECT_ORIENT_ALONG_SHARE
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
        const DefectParticle& p,
        AlgoType* RESTRICT    pDstR,
        AlgoType* RESTRICT    pDstG,
        AlgoType* RESTRICT    pDstB,
        const int32_t         sizeX,
        const int32_t         sizeY,
        const int32_t         pitch,
        const HighPrecType    edgePx
    ) noexcept
    {
        // Outer reach of the particle: the long semi-axis at its greatest
        // harmonic excursion, plus the edge transition.
        const HighPrecType reach = p.radiusPx * p.aspect * (1.0 + p.lobeDepth)
                                 + edgePx;

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
        const HighPrecType invEdge = 1.0 / edgePx;

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

                    for (int32_t k = 0; k < ALGO_DEFECT_HARMONICS; k++)
                    {
                        const AlgoType order = static_cast<AlgoType>(k + 2);

                        shape += (lobeDf / order)
                               * std::cos(order * theta
                                          + static_cast<AlgoType>(p.harmPhase[k]));
                    }
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
                const AlgoType aR = MIN_VALUE(alphaf * cov
                                        * static_cast<AlgoType>(p.chroma[0]),
                                    static_cast<AlgoType>(ALGO_DEFECT_ALPHA_CAP));
                const AlgoType aG = MIN_VALUE(alphaf * cov
                                        * static_cast<AlgoType>(p.chroma[1]),
                                    static_cast<AlgoType>(ALGO_DEFECT_ALPHA_CAP));
                const AlgoType aB = MIN_VALUE(alphaf * cov
                                        * static_cast<AlgoType>(p.chroma[2]),
                                    static_cast<AlgoType>(ALGO_DEFECT_ALPHA_CAP));

                rR[x] += -std::log10(ALGO_ONE - aR);
                rG[x] += -std::log10(ALGO_ONE - aG);
                rB[x] += -std::log10(ALGO_ONE - aB);
            }
        }

        return;
    }


    // ----------------------------------------------------------------------
    //  Rasterise a fibre: a polyline stroked at constant width.
    //
    //  Coverage comes from the distance to the NEAREST point on the whole
    //  centreline, not from stamping each segment in turn. Stamping would add
    //  density twice wherever two segments overlap, and since consecutive
    //  segments always overlap at their shared joint, a stamped fibre acquires a
    //  string of beads along its length. Taking the minimum distance over the
    //  whole polyline costs one pass over the fibre's bounding box per segment
    //  and produces a genuinely constant-width stroke - which is the mechanical
    //  signature of a foreign object lying on the film, and the one thing that
    //  must not be got wrong here.
    // ----------------------------------------------------------------------
    void defectRasteriseFibre
    (
        const HighPrecType* RESTRICT ptsX,
        const HighPrecType* RESTRICT ptsY,
        const int32_t                pointCount,
        const HighPrecType           halfWidthPx,
        const HighPrecType           alpha,
        const HighPrecType           chroma[3],
        AlgoType* RESTRICT           pDstR,
        AlgoType* RESTRICT           pDstG,
        AlgoType* RESTRICT           pDstB,
        const int32_t                sizeX,
        const int32_t                sizeY,
        const int32_t                pitch,
        const HighPrecType           edgePx
    ) noexcept
    {
        if (pointCount < 2)
            return;

        const HighPrecType reach = halfWidthPx + edgePx;

        // Bounding box of the whole centreline.
        HighPrecType minX = ptsX[0];
        HighPrecType maxX = ptsX[0];
        HighPrecType minY = ptsY[0];
        HighPrecType maxY = ptsY[0];

        for (int32_t i = 1; i < pointCount; i++)
        {
            minX = MIN_VALUE(minX, ptsX[i]);
            maxX = MAX_VALUE(maxX, ptsX[i]);
            minY = MIN_VALUE(minY, ptsY[i]);
            maxY = MAX_VALUE(maxY, ptsY[i]);
        }

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

                // Squared distance to the nearest point of the polyline.
                HighPrecType best = -1.0;

                for (int32_t i = 0; i + 1 < pointCount; i++)
                {
                    const HighPrecType ax = ptsX[i];
                    const HighPrecType ay = ptsY[i];
                    const HighPrecType bx = ptsX[i + 1];
                    const HighPrecType by = ptsY[i + 1];

                    // Cheap rejection on the segment's own bounding box. A fibre
                    // is a long thin thing, so for almost every pixel almost every
                    // segment is far away, and the exact projection below is worth
                    // avoiding. Without this the cost is the fibre's whole
                    // bounding box times its segment count.
                    if (px < (MIN_VALUE(ax, bx) - reach) ||
                        px > (MAX_VALUE(ax, bx) + reach) ||
                        py < (MIN_VALUE(ay, by) - reach) ||
                        py > (MAX_VALUE(ay, by) + reach))
                        continue;

                    const HighPrecType ex = bx - ax;
                    const HighPrecType ey = by - ay;

                    const HighPrecType len2 = ex * ex + ey * ey;

                    // Projection parameter, clamped to the segment so the nearest
                    // point is an endpoint when the foot of the perpendicular
                    // falls outside. Without the clamp the stroke would extend
                    // along the infinite line and the fibre would have no ends.
                    HighPrecType t = 0.0;

                    if (len2 > 0.0)
                        t = CLAMP_VALUE(((px - ax) * ex + (py - ay) * ey) / len2,
                                        0.0, 1.0);

                    const HighPrecType qx = ax + t * ex;
                    const HighPrecType qy = ay + t * ey;

                    const HighPrecType ddx = px - qx;
                    const HighPrecType ddy = py - qy;

                    const HighPrecType d2 = ddx * ddx + ddy * ddy;

                    if (best < 0.0 || d2 < best)
                        best = d2;
                }

                // Every segment rejected: this pixel is nowhere near the fibre.
                if (best < 0.0)
                    continue;

                const HighPrecType dist = std::sqrt(best);

                const HighPrecType cov = defectSmoothStep(
                    ((halfWidthPx - dist) * invEdge) + 0.5);

                if (cov <= 0.0)
                    continue;

                const HighPrecType aR = MIN_VALUE(alpha * cov * chroma[0],
                                                  ALGO_DEFECT_ALPHA_CAP);
                const HighPrecType aG = MIN_VALUE(alpha * cov * chroma[1],
                                                  ALGO_DEFECT_ALPHA_CAP);
                const HighPrecType aB = MIN_VALUE(alpha * cov * chroma[2],
                                                  ALGO_DEFECT_ALPHA_CAP);

                rR[x] += static_cast<AlgoType>(-std::log10(1.0 - aR));
                rG[x] += static_cast<AlgoType>(-std::log10(1.0 - aG));
                rB[x] += static_cast<AlgoType>(-std::log10(1.0 - aB));
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
        const HighPrecType    edgePx
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

            p.angleRad = defectOrientation(window, pc + 16u);

            p.lobeDepth = ALGO_DUST_LOBE_DEPTH;

            for (int32_t k = 0; k < ALGO_DEFECT_HARMONICS; k++)
                p.harmPhase[k] = AlgoRngUniform01(
                    pc + 20u + static_cast<uint64_t>(k)) * ALGO_DEFECT_TWO_PI;

            defectChroma(pc + 28u, monochrome, p.chroma);

            defectRasterise(p, pDstR, pDstG, pDstB,
                            sizeX, sizeY, pitch, edgePx);
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
        const HighPrecType    edgePx
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

            p.angleRad = defectOrientation(window, pc + 14u);

            // Deep harmonics give the irregular, concave outline of a lint ball.
            p.lobeDepth = ALGO_DEBRIS_LOBE_DEPTH;

            for (int32_t k = 0; k < ALGO_DEFECT_HARMONICS; k++)
                p.harmPhase[k] = AlgoRngUniform01(
                    pc + 20u + static_cast<uint64_t>(k)) * ALGO_DEFECT_TWO_PI;

            defectChroma(pc + 28u, monochrome, p.chroma);

            defectRasterise(p, pDstR, pDstG, pDstB,
                            sizeX, sizeY, pitch, edgePx);
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
        const HighPrecType    edgePx
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

            defectRasteriseFibre(ptsX, ptsY, drawCount, halfWidthPx,
                                 alpha, chroma,
                                 pDstR, pDstG, pDstB,
                                 sizeX, sizeY, pitch, edgePx);
        });

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
    const AlgoType           pxPerMm
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
                       && (radiusPx >= ALGO_COUPLER_MIN_SIGMA_PX);

    // The short-range term is within a single layer, so it applies to monochrome
    // stocks too - and is in fact the dominant coupler effect on them.
    const bool wantEdge = (e > ALGO_ZERO)
                       && (edgePx >= ALGO_COUPLER_MIN_SIGMA_PX);

    // Copy first, unconditionally, then modify in place. Both components are
    // read-modify-write against a blurred version of the data, so a destination
    // that already holds the incoming densities is the natural starting state -
    // and it satisfies the retained-buffer policy even when neither component is
    // active.
    AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

    if ((false == wantLong) && (false == wantEdge))
        return;

    AlgoType* RESTRICT dstPlane[3] = { pDstR, pDstG, pDstB };

    // ----------------------------------------------------------------------
    //  Long-range component.
    //
    //  Each layer is pushed away from the LOCALLY BLURRED MEAN of all three. The
    //  mean is what makes it a colour effect: a neutral, where all three layers
    //  agree, sits at its own mean and is left alone, while a colour is driven
    //  further from it. Saturation rises and gamma does not.
    // ----------------------------------------------------------------------
    if (wantLong)
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

        // Blur it over the inhibitor's diffusion distance. Wrap boundary, matching
        // the circular convolution of the frequency-domain reference; the radius is
        // a few micrometres on the film, so the wrap contribution is confined to a
        // handful of edge pixels.
        AlgoGaussianBlurPlaneWrap(pScrDbar, pScrDbarBlur,
                                  pScrBlurA,
                                  sizeX, sizeY, pitch, radiusPx);

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
                    rO[x] += s * (rO[x] - rM[x]);
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
    if (wantEdge)
    {
        for (int32_t c = 0; c < 3; c++)
        {
            AlgoType* RESTRICT pO = dstPlane[c];

            // Blurred copy of this channel. pScrDbarBlur is reused as the
            // destination: the long-range term has finished with it, and the two
            // components never need their blurs at the same time.
            AlgoGaussianBlurPlaneWrap(pO, pScrDbarBlur,
                                      pScrBlurA,
                                      sizeX, sizeY, pitch, edgePx);

            for (int32_t y = 0; y < sizeY; y++)
            {
                const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(y) * pitch;

                AlgoType* RESTRICT       rO = pO           + off;
                const AlgoType* RESTRICT rB = pScrDbarBlur + off;

                ALGO_VECTOR_HINT
                for (int32_t x = 0; x < sizeX; x++)
                    rO[x] += e * (rO[x] - rB[x]);
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
    for (int32_t c = 0; c < 3; c++)
    {
        AlgoType* RESTRICT pO = dstPlane[c];

        for (int32_t y = 0; y < sizeY; y++)
        {
            AlgoType* RESTRICT rO = pO + static_cast<std::ptrdiff_t>(y) * pitch;

            ALGO_VECTOR_HINT
            for (int32_t x = 0; x < sizeX; x++)
                rO[x] = MAX_VALUE(rO[x], ALGO_ZERO);
        }
    }

    // pScrBlurB is not needed by this stage - both blurs here are single-lobe and
    // a single-lobe separable pass needs only one intermediate plane. It stays in
    // the signature so the stage's scratch requirement does not change if a second
    // lobe is ever added to either diffusion term.
    (void)pScrBlurB;

    return;
}


// ---------------------------------------------------------------------------
//  Sub-stage 9b: negative-side defects.
//
//  Three particulate classes, all embedded in the emulsion and therefore part of
//  the negative: fine dust, coarse debris, hair and fibres.
//
//  The copy comes first and unconditionally. The retained-buffer policy gives
//  every stage its own destination, so returning without writing would leave stale
//  contents for every stage after this one; and the particles are then accumulated
//  in place, which is what lets three independent generators share one output
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
) noexcept
{
    // The unconditional copy. Every path below accumulates on top of it.
    AlgoCopyImage(pSrcR, pSrcG, pSrcB, pDstR, pDstG, pDstB, sizeX, sizeY, pitch);

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

    // Nothing requested, nothing to do. Worth its own test: the three loops below
    // each walk a cell grid over the whole window before discovering their rate is
    // zero, and for fibres that grid extends 25 mm past the frame in every
    // direction.
    if (dustLevel <= 0.0 && debrisLevel <= 0.0 && fibreLevel <= 0.0)
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
    //  The three classes, in ascending order of rarity.
    //
    //  Order is not significant to the result - density accumulates additively and
    //  addition commutes - but each class draws from its own stream, so adjusting
    //  one level leaves the others' particles exactly where they were.
    // ----------------------------------------------------------------------
    if (dustLevel > 0.0)
        defectDust(window, dustLevel, clumping, fieldSeed, monochrome,
                   pDstR, pDstG, pDstB, sizeX, sizeY, pitch, edgePx);

    if (debrisLevel > 0.0)
        defectDebris(window, debrisLevel, clumping, fieldSeed, monochrome,
                     pDstR, pDstG, pDstB, sizeX, sizeY, pitch, edgePx);

    if (fibreLevel > 0.0)
        defectFibres(window, fibreLevel, clumping, fieldSeed, monochrome,
                     pDstR, pDstG, pDstB, sizeX, sizeY, pitch, edgePx);

    // ----------------------------------------------------------------------
    //  Read by nothing in this stage, and that is the honest state of it.
    //
    //  profile IS read now, for is_monochrome, but its AgingSpec is not. That
    //  structure's dust_area_ppm, mottle_amplitude and
    //  scratch rates are the era baseline these levels were meant to multiply -
    //  but every one of the 142 stocks currently ships that structure all zero,
    //  documented as "fresh". Multiplying by it would therefore silence the whole
    //  defect layer on every stock, so the levels are absolute for now: dustLevel
    //  1.0 means the measured central density, whatever the stock. When AgingSpec
    //  is populated it becomes an additive era term rather than a multiplier, so
    //  that a fresh stock keeps behaving exactly as it does today.
    //
    //  frameRate belongs to the classes with a per-second rate - the one-frame
    //  sparkle population and the event classes - and none of those is here.
    //  negWidthMm and negHeightMm are consumed through the window; pxPerMm
    //  likewise, and directly for the edge width.
    // ----------------------------------------------------------------------
    (void)frameRate;

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
