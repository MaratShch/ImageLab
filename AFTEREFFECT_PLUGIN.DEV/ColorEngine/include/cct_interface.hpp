#ifndef __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__
#define __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__

// =============================================================================
// cct_interface.hpp - public interface of the ImageLab2 CCT engine
// =============================================================================
//
// PURPOSE
//   Maps a chromaticity point to correlated color temperature and tint
//   (CCT, Duv), and back (Planckian point for a given CCT/Duv), for the two
//   CIE standard observers. The solver is Ohno (2013) triangular+parabolic
//   over a Planckian-locus LUT (see cct_refine.cpp for math and accuracy).
//
// LUT BACKEND (compile-time)
//   The locus tables are GENERATED constexpr std::array headers
//   (CCT_LUT_CIE_1931_2DEG.hpp / CCT_LUT_CIE_1964_10DEG.hpp, produced by
//   gen_cct_lut_header.py from official CIE CMF data, double precision,
//   Duv = 0 stored - every entry lies on the locus by construction).
//   Consequences of the compile-time backend:
//     - NO runtime initialization exists: no lazy init, no ready flags, no
//       mutex, no per-instance LUT storage. The former multi-instance
//       lifetime bug and the "getPlanckianUV before first ComputeCct"
//       crash are structurally impossible now.
//     - Thread-safe and instance-safe by construction (immutable .rodata).
//     - The class holds only pointer+size views (C++14 has no std::span).
//   Table range: 1000..26010 K at 1 K step. The top 10 K is a STENCIL
//   MARGIN so the 3-point parabolic solve has a full neighborhood at the
//   REPORTED ceiling of 26000 K; treat results at/above 26000 K as clamped
//   per the display rules (see getCctMax).
//
// TEMPLATE DESIGN
//   CctHandle<T> is templated on the PUBLIC API scalar type T only
//   (T = float | double, SFINAE-enforced):
//     - CctHandle<float>  ("CctHandleF32"): drop-in for the former class;
//       boundary quantization ~1e-7 in u,v (~0.03-0.08 K).
//     - CctHandle<double> ("CctHandleF64"): lossless end-to-end - use for
//       the analysis path (the super-pixel bridge is already double).
//   The INTERNAL core is NOT templated and is always double - a measured
//   correctness rule (float locus u,v cost ~5 K at 15000 K; double reaches
//   ~0.02 K). Exactly two instantiations exist (explicit, in the .cpp).
//
// CONVENTIONS (project-wide)
//   - Chromaticities are CIE 1960 UCS (u, v):
//       u = 4X/(X+15Y+3Z),  v = 6Y/(X+15Y+3Z).
//   - Duv sign: positive ABOVE the locus (green), negative BELOW (magenta);
//     calibrated so D65 yields +0.0032.
//   - "Compute CCT" always means BOTH CCT and Duv.
//   - Precision: per-pixel float32; LUT + solve double; API scalar = T.
//
// PORTABILITY
//   Standard C++14 only; OS- and compiler-independent.
// =============================================================================

#include "ClassRestrictions.hpp"
#include "CCTLut/CCTLut.hpp"
#include "ColorTransformMatrix.hpp"     // eCOLOR_OBSERVER
#include <utility>
#include <cstddef>
#include <type_traits>

namespace AlgoCCT
{
    // Row type shared by all generated LUT headers.
    using LutRow = CctLutShared::CctLutRow_double;

    template
    <
        typename T,
        typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr
    >
    class CctHandle final
    {
        public:
            // Constructor only wires the compile-time tables; nothing is
            // built at run time.
            CctHandle() 
                : m_lut1(CCT_LUT_1931_2DEG::CCT_LUT_CIE_1931_2DEG.data()),
                  m_size1(CCT_LUT_1931_2DEG::CCT_LUT_CIE_1931_2DEG_SIZE),
                  m_lut2(CCT_LUT_1964_10DEG::CCT_LUT_CIE_1964_10DEG.data()),
                  m_size2(CCT_LUT_1964_10DEG::CCT_LUT_CIE_1964_10DEG_SIZE)
            {}
            ~CctHandle() = default;

            CLASS_NON_COPYABLE(CctHandle);
            CLASS_NON_MOVABLE (CctHandle);

            // -----------------------------------------------------------------
            // ComputeCct - PUBLIC ENTRY POINT.
            // Input : uv       - CIE 1960 (u, v) of the stimulus.
            //         observer - observer_CIE_1931 (2 deg) or
            //                    observer_CIE_1964 (10 deg): selects the table
            //                    (the locus differs because the CMFs differ).
            // Return: {CCT [K], Duv}. Always computable - the tables are
            //         compile-time constants (no init failure mode).
            // Results at/above getCctMax() or at getCctMin() should be
            // flagged as clamped/extrapolated by the caller (display rules).
            // -----------------------------------------------------------------
            std::pair<T /* CCT */, T /* Duv */> ComputeCct (const std::pair<T, T>& uv, eCOLOR_OBSERVER observer);

            // -----------------------------------------------------------------
            // getPlanckianUV - inverse mapping: (CCT, Duv) -> (u, v).
            // Planckian point at 'cct', displaced by 'Duv' along the locus
            // normal (CIE sign convention; round-trips with ComputeCct are
            // sign-consistent). Step-agnostic: assumes only ascending-sorted
            // entries (binary search + normalized interpolation).
            // Out-of-range cct clamps to the first/last locus point.
            // Callable at ANY time - no initialization ordering exists.
            // -----------------------------------------------------------------
            std::pair<T /* u */, T /* v */> getPlanckianUV (T cct, T Duv, eCOLOR_OBSERVER observer);
            std::pair<T /* u */, T /* v */> getPlanckianUV (const std::pair<T, T>& cct_Duv, eCOLOR_OBSERVER observer);

            // Raw view of the selected locus table (always valid).
            std::pair<const LutRow*, std::size_t> getLut_CIE_1931 (void) const noexcept { return { m_lut1, m_size1 }; }
            std::pair<const LutRow*, std::size_t> getLut_CIE_1964 (void) const noexcept { return { m_lut2, m_size2 }; }

            // REPORTED engine range [K], inclusive. The physical table holds
            // a +10 K stencil margin above the ceiling (see header note).
            T getCctMin (void) const noexcept { return static_cast<T>(1000.0);  }
            T getCctMax (void) const noexcept { return static_cast<T>(26000.0); }

        private:

            // Thin adapter: zero outputs, run the solver (cct_refine.cpp).
            // Internal core is DOUBLE regardless of T.
            bool cct_compute (const double& u, const double& v, double& cct, double& duv,
                              const LutRow* lut, std::size_t n);

            // Ohno (2013) triangular+parabolic solver (cct_refine.cpp); the
            // only tuning constant is Ohno's 0.002 crossover on |Duv|.
            bool refine (const double& u, const double& v, double& cct, double& duv,
                         const LutRow* lut, std::size_t n);

            // Views into the compile-time tables (.rodata; immutable, shared
            // by all instances and threads).
            const LutRow*     m_lut1;   // CIE 1931 2-deg
            std::size_t       m_size1;
            const LutRow*     m_lut2;   // CIE 1964 10-deg
            std::size_t       m_size2;

    }; // class CctHandle

    // The only two supported instantiations (explicit, in the .cpp files).
    // CctHandleF32 preserves the former class name - call sites unchanged.
    using CctHandleF32 = CctHandle<float>;
    using CctHandleF64 = CctHandle<double>;

}; // namespace AlgoCCT

#endif // __IMAGELAB2_CCT_INTERAFCE_LIBRARY_MODULE__
