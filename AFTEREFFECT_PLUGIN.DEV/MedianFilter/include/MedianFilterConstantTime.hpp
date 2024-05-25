#include "CommonAdobeAE.hpp"
#include "CommonPixFormatSFINAE.hpp"


template <typename U, typename T, std::enable_if_t<is_RGB_proc<T>::value && std::is_integral<U>::value>* = nullptr>
inline void median_filter_constant_time_RGB
(
	const T* __restrict pInImage,
	      T* __restrict pOutImage,
	      U* __restrict pChannelHist[3],
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept
{
	A_long i, j;

	for (j = 0; j < sizeY; j++)
	{
		for (i = 0; i < sizeX; i++)
		{

		}
	}

	return;
}