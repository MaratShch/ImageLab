#pragma once

#include "Common.hpp"
#include "CommonPixFormat.hpp"

template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void imgHistogram
(
	const T* __restrict srcImage,
	const int32_t& sizeX,
	const int32_t& sizeY,
	const int32_t& line_pitch,
	uint32_t* __restrict histBuffer 
) noexcept
{
	for (int32_t j = 0; j < sizeY; j++)
	{
		const T* __restrict linePtr = srcImage + j * line_pitch;
		__VECTOR_ALIGNED__
		for (int32_t i = 0; i < sizeX; i++)
			histBuffer[linePtr[i].Y]++;
	}
	return;
}


inline void imgHistogram
(
	const PF_Pixel_VUYA_32f* __restrict srcImage,
	const int32_t& sizeX,
	const int32_t& sizeY,
	const int32_t& line_pitch,
	uint32_t* __restrict histBuffer
) noexcept
{
	constexpr float multiplyer = static_cast<float>(u16_value_white);
	for (int32_t j = 0; j < sizeY; j++)
	{
		const PF_Pixel_VUYA_32f* __restrict linePtr = srcImage + j * line_pitch;
		__VECTOR_ALIGNED__
		for (int32_t i = 0; i < sizeX; i++)
			histBuffer[static_cast<uint32_t>(linePtr[i].Y * multiplyer)]++;
	}
	return;
}

inline void imgHistogramCumSum
(
	const uint32_t* __restrict in,
	uint32_t* __restrict out,
	const int32_t& noiseThreshold,
	const int32_t& size
) noexcept
{
	__VECTOR_ALIGNED__
	out[0] = (in[0] > noiseThreshold ? 1u : 0u);
	for (int32_t i = 1; i < size; i++)
		out[i] = (in[i] > noiseThreshold ? 1u : 0u) + out[i - 1];
	return;
}
