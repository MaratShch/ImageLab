#pragma once

#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "Avx2ColorConvert.hpp"

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

inline void imgApplyLut
(
	const PF_Pixel_VUYA_8u* __restrict inSrc,
	      PF_Pixel_BGRA_8u* __restrict outSrc,
	const uint32_t* __restrict pLut,
	const int32_t& sizeX,
	const int32_t& sizeY,
	const int32_t& src_line_pitch,
	const int32_t& dst_line_pitch
) noexcept
{
	namespace ColorCoeff = AVX2::ColorConvert::InternalColorConvert;
	constexpr int32_t value_black = static_cast<int32_t>(u8_value_black);
	constexpr int32_t value_white = static_cast<int32_t>(u8_value_white);

	for (int32_t j = 0; j < sizeY; j++)
	{
		const PF_Pixel_VUYA_8u* __restrict lineSrcPtr = inSrc  + j * src_line_pitch;
		      PF_Pixel_BGRA_8u* __restrict lineDstPtr = outSrc + j * dst_line_pitch;

		__VECTOR_ALIGNED__
		for (int32_t i = 0; i < sizeX; i++)
		{
			const int32_t newY = static_cast<int32_t>(pLut[lineSrcPtr[i].Y]);
			const int32_t U    = static_cast<int32_t>(lineSrcPtr[i].U);
			const int32_t V    = static_cast<int32_t>(lineSrcPtr[i].V);

			const int32_t R = (newY * ColorCoeff::rY + U * ColorCoeff::rU + V * ColorCoeff::rV) >> ColorCoeff::Shift;
			const int32_t G = (newY * ColorCoeff::gY + U * ColorCoeff::gU + V * ColorCoeff::gV) >> ColorCoeff::Shift;
			const int32_t B = (newY * ColorCoeff::bY + U * ColorCoeff::bU + V * ColorCoeff::bV) >> ColorCoeff::Shift;

			lineDstPtr[i].B = static_cast<uint8_t>(CLAMP_VALUE(B, value_black, value_white));
			lineDstPtr[i].G = static_cast<uint8_t>(CLAMP_VALUE(G, value_black, value_white));
			lineDstPtr[i].R = static_cast<uint8_t>(CLAMP_VALUE(R, value_black, value_white));
			lineDstPtr[i].A = lineSrcPtr[i].A;
		}
	}
	return;
}


inline void imgApplyLut
(
	const PF_Pixel_VUYA_32f* __restrict inSrc,
	      PF_Pixel_VUYA_32f* __restrict outSrc,
	const float*             __restrict pLut,
	const int32_t& sizeX,
	const int32_t& sizeY,
	const int32_t& src_line_pitch,
	const int32_t& dst_line_pitch
) noexcept
{
	for (int32_t j = 0; j < sizeY; j++)
	{
		const PF_Pixel_VUYA_32f* __restrict lineSrcPtr = inSrc  + j * src_line_pitch;
		      PF_Pixel_VUYA_32f* __restrict lineDstPtr = outSrc + j * dst_line_pitch;

		__VECTOR_ALIGNED__
		for (int32_t i = 0; i < sizeX; i++)
		{
			lineDstPtr[i].V = lineSrcPtr[i].V;
			lineDstPtr[i].U = lineSrcPtr[i].U;
			lineDstPtr[i].Y = pLut[static_cast<int32_t>(32767.f * lineSrcPtr[i].Y)];
			lineDstPtr[i].A = lineSrcPtr[i].A;
		}
	}
	return;
}