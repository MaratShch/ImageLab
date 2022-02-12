#include "Avx2Morphology.hpp"


inline void AVX2::Morphology::Erode::Erode_line_8u
(
	const __m256i  src[9],	/* source pixels		*/
	const __m256i& selem,	/* structured element	*/
	const __m256i& smask,	/* store mask			*/	
	      __m256i& dst		/* destination pixel	*/
) noexcept
{

}


#if 0
void AVX2::Morphology::Erode::Erode_3x3_8U
(
	uint32_t* __restrict pInImage,	/* input buffer						*/
	uint32_t* __restrict pOutImage,	/* output buffer					*/	
	uint32_t* __restrict sElement,	/* structured element pointer		*/
	A_long srcStride,				/* source buffer line stride		*/	
	A_long dstStride,				/* destination buffer line stride	*/
	const A_long& chanelMask /* 0x00FFFFFF <- BGRa */
)
{
	return true;
}
#endif