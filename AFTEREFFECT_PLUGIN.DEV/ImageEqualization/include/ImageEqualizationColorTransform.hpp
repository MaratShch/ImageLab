#pragma once

#include "ColorTransform.hpp"
#include "CommonAuxPixFormat.hpp"

constexpr eCOLOR_OBSEREVER  gDefaultObserver   = observer_CIE_1931;
constexpr eCOLOR_ILLUMINANT gDefaultIlluminant = color_ILLUMINANT_D65;


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void ConvertToLabColorSpace
(
	const T*     pSrc,
	fCIELabPix*  pDst,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& src_pitch,
	const A_long& dst_pitch,
	const float&  scale
) noexcept
{
	CACHE_ALIGN constexpr float colorReferences[3] =
	{
		cCOLOR_ILLUMINANT[gDefaultObserver][gDefaultIlluminant][0],
		cCOLOR_ILLUMINANT[gDefaultObserver][gDefaultIlluminant][1],
		cCOLOR_ILLUMINANT[gDefaultObserver][gDefaultIlluminant][2]
	};

	for (A_long j = 0; j < sizeY; j++)
	{
		const T*     __restrict pSrcLine = pSrc + j * src_pitch;
		fCIELabPix*  __restrict pDstLine = pDst + j * dst_pitch;

		/* lambda for convert from RGB to sRGB */
		auto const convert2fRGB = [&](const T p, const float val)
		{
			fRGB outPix;
			outPix.R = p.R * val, outPix.G = p.G * val, outPix.B = p.B * val;
			return outPix;
		};

		for (A_long i = 0; i < sizeX; i++)
			pDstLine[i] = RGB2CIELab (convert2fRGB(pSrcLine[i], scale), colorReferences);
	}
	return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void ConvertToLabColorSpace
(
	const T*    __restrict pSrc,
	fCIELabPix* __restrict pDst,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& src_pitch,
	const A_long& dst_pitch
) noexcept
{
	return;
}