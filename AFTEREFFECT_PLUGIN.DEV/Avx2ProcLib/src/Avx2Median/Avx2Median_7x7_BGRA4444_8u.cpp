#include "Avx2Median.hpp"
#include "Avx2MedianInternal.hpp"



/*
make median filter with kernel 5x5 from packed format - BGRA444_8u by AVX2 instructions set:

Image buffer layout [each cell - 8 bits unsigned in range 0...255]:

LSB                            MSB
+-------------------------------+
| B | G | R | A | B | G | R | A | ...
+-------------------------------+

*/
bool AVX2::Median::median_filter_7x7_BGRA_4444_8u
(
	PF_Pixel_BGRA_8u* __restrict pInImage,
	PF_Pixel_BGRA_8u* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long linePitch
) noexcept
{
	return true;
}