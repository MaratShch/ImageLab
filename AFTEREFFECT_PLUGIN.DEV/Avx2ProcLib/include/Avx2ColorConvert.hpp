#include <cstdint>
#include <immintrin.h>

namespace AVX2
{
	namespace ColorConvert
	{
		__m256i Convert_bgra2vuya_8u (const __m256i& bgraX8) noexcept;
		__m256i Convert_vuya2bgra_8u (const __m256i& bgraX8) noexcept;

	}; /* namespace ColorConvert */

}; /* namespace AVX2 */
