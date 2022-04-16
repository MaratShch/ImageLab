#pragma once 
#include "ColorBandSelect.hpp"
#include "ColorTransformMatrix.hpp"

/* RGB processing */
template <class T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
inline void ImgCopyByChannelMask
(
	const T* __restrict pSrcImg,
	T* __restrict pDstImg,
	const A_long& srcPitch,
	const A_long& dstPitch,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& Red,
	const A_long& Green,
	const A_long& Blue
) noexcept
{
	for (A_long j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine{ pSrcImg + j * srcPitch };
		      T* __restrict pDstLine{ pDstImg + j * dstPitch };

		__VECTOR_ALIGNED__
		for (A_long i = 0; i < sizeX; i++)
		{
			pDstLine[i].B = Blue  ? pSrcLine[i].B : 0;
			pDstLine[i].G = Green ? pSrcLine[i].G : 0;
			pDstLine[i].R = Red   ? pSrcLine[i].R : 0;
			pDstLine[i].A = pSrcLine[i].A;
		}
	}
	return;
}


/* YUV processing */
template <class T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void ImgCopyByChannelMask
(
	const T* __restrict pSrcImg,
	T* __restrict pDstImg,
	const A_long& srcPitch,
	const A_long& dstPitch,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& Red,
	const A_long& Green,
	const A_long& Blue
) noexcept
{
	constexpr size_t PixSize{ sizeof(T) };
	constexpr float clampMax{ PixSize == PF_Pixel_VUYA_8u_size ? 255.f : f32_value_white };

	const float* __restrict rgb2yuv = RGB2YUV[BT709];
	const float* __restrict yuv2rgb = YUV2RGB[BT709];

	for (A_long j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine{ pSrcImg + j * srcPitch };
		      T* __restrict pDstLine{ pDstImg + j * dstPitch };

		__VECTOR_ALIGNED__
		for (A_long i = 0; i < sizeX; i++)
		{
			float const& y = static_cast<float>(pSrcLine[i].Y);
			float const& u = static_cast<float>(pSrcLine[i].U) - (PixSize == PF_Pixel_VUYA_8u_size ? 128.f : 0.f);
			float const& v = static_cast<float>(pSrcLine[i].V) - (PixSize == PF_Pixel_VUYA_8u_size ? 128.f : 0.f);

			const float R = (Red   ? (y * yuv2rgb[0] + u * yuv2rgb[1] + v * yuv2rgb[2]) : 0);
			const float G = (Green ? (y * yuv2rgb[3] + u * yuv2rgb[4] + v * yuv2rgb[5]) : 0);
			const float B = (Blue  ? (y * yuv2rgb[6] + u * yuv2rgb[7] + v * yuv2rgb[8]) : 0);

			const float YVal = R * rgb2yuv[0] + G * rgb2yuv[1] + B * rgb2yuv[2];
			const float UVal = R * rgb2yuv[3] + G * rgb2yuv[4] + B * rgb2yuv[5] + (PixSize == PF_Pixel_VUYA_8u_size ? 128.f : 0.f);
			const float VVal = R * rgb2yuv[6] + G * rgb2yuv[7] + B * rgb2yuv[8] + (PixSize == PF_Pixel_VUYA_8u_size ? 128.f : 0.f);
			
			pDstLine[i].A = pSrcLine[i].A;
			pDstLine[i].Y = CLAMP_VALUE(YVal, 0.f, clampMax);
			pDstLine[i].U = CLAMP_VALUE(UVal, 0.f, clampMax);
			pDstLine[i].V = CLAMP_VALUE(VVal, 0.f, clampMax);
		}
	}
	return;
}

