#include <cstdint>
#include <immintrin.h>
#include "Common.hpp"
#include "CommonPixFormat.hpp"

namespace AVX2
{
    namespace ColorConvert
    {
 
        void Rgb2YuvSemiplanar_BGRA_8u_AVX2
        (
            const PF_Pixel_BGRA_8u* RESTRICT in,
            float* RESTRICT dstY,
            float* RESTRICT dstUV,
            int32_t width,
            int32_t height,
            int32_t linePitch
        ) noexcept;

        void Rgb2YuvSemiplanar_ARGB_8u_AVX2
        (
            const PF_Pixel_ARGB_8u* RESTRICT in,
            float* RESTRICT dstY,
            float* RESTRICT dstUV,
            int32_t width,
            int32_t height,
            int32_t linePitch
        ) noexcept;

        void Rgb2YuvSemiplanar_BGRA_32f_AVX2
        (
            const PF_Pixel_BGRA_32f* RESTRICT in,
            float* RESTRICT dstY,
            float* RESTRICT dstUV,
            int32_t width,
            int32_t height,
            int32_t linePitch
        ) noexcept;

        void Rgb2YuvSemiplanar_ARGB_32f_AVX2
        (
            const PF_Pixel_ARGB_32f* RESTRICT in,
            float* RESTRICT dstY,
            float* RESTRICT dstUV,
            int32_t width,
            int32_t height,
            int32_t linePitch
        ) noexcept;

        void Rgb2YuvSemiplanar_BGRA_16u_AVX2
        (
            const PF_Pixel_BGRA_16u* RESTRICT in,
            float* RESTRICT dstY,
            float* RESTRICT dstUV,
            int32_t width,
            int32_t height,
            int32_t linePitch
        ) noexcept;

        void Rgb2YuvSemiplanar_ARGB_16u_AVX2
        (
            const PF_Pixel_ARGB_16u* RESTRICT in,
            float* RESTRICT dstY,
            float* RESTRICT dstUV,
            int32_t width,
            int32_t height,
            int32_t linePitch
        ) noexcept;

    }
}