#pragma once

#include "Common.hpp"
#include "CommonPixFormat.hpp"


inline void imgLinearLutGenerate
(
	const uint32_t* __restrict hist,
	      uint32_t* __restrict lut,
	const int32_t&      size
) noexcept
{
	const float maxElement = static_cast<float>(hist[size - 1]);
	const float lutCoeff = static_cast<float>(size - 1) / maxElement;
	for (int32_t i = 0; i < size; i++)
		lut[i] = static_cast<uint32_t>(static_cast<float>(hist[i]) * lutCoeff);
	return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void imgApplyLut
(
	const T* __restrict inSrc,
	      T* __restrict outSrc,
	const uint32_t* __restrict pLut,
	const int32_t& sizeX,
	const int32_t& sizeY,
	const int32_t& src_line_pitch,
	const int32_t& dst_line_pitch
) noexcept
{
	for (int32_t j = 0; j < sizeY; j++)
	{
		const T* __restrict lineSrcPtr = inSrc  + j * src_line_pitch;
		      T* __restrict lineDstPtr = outSrc + j * dst_line_pitch;

		__VECTOR_ALIGNED__
		for (int32_t i = 0; i < sizeX; i++)
		{
			lineDstPtr[i].V = lineSrcPtr[i].V;
			lineDstPtr[i].U = lineSrcPtr[i].U;
			lineDstPtr[i].Y = pLut[lineSrcPtr[i].Y];
			lineDstPtr[i].A = lineSrcPtr[i].A;
		}
	}
	return;

}