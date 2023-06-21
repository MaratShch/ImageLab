#pragma once

#include "ColorTransform.hpp"
#include "CommonAuxPixFormat.hpp"

constexpr eCOLOR_OBSEREVER  gDefaultObserver   = observer_CIE_1931;
constexpr eCOLOR_ILLUMINANT gDefaultIlluminant = color_ILLUMINANT_D65;


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline float ConvertToLabColorSpace
(
	const T*     __restrict pSrc,
	fCIELabPix*  __restrict pDst,
	float*       __restrict csOut,
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
	float cs_out_max = FLT_MIN;

	for (A_long j = 0; j < sizeY; j++)
	{
		const T*     __restrict pSrcLine   = pSrc  + j * src_pitch;
		fCIELabPix*  __restrict pDstLine   = pDst  + j * sizeX;
		float*       __restrict pCsOutLine = csOut + j * sizeX;

		/* lambda for convert from RGB to sRGB */
		auto const convert2sRGB = [&](const T p, const float val) noexcept
		{
			fRGB outPix;
			outPix.R = p.R * val, outPix.G = p.G * val, outPix.B = p.B * val;
			return outPix;
		};

		for (A_long i = 0; i < sizeX; i++)
		{
			pDstLine[i] = RGB2CIELab(convert2sRGB(pSrcLine[i], scale), colorReferences);
			auto const& a = pDstLine[i].a;
			auto const& b = pDstLine[i].b;
			const float cs_out = FastCompute::Sqrt(a * a + b * b);
			pCsOutLine[i] = cs_out;
			cs_out_max = FastCompute::Max(cs_out_max, cs_out);
		}
	}
	return cs_out_max;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline float ConvertToLabColorSpace
(
	const T*     __restrict pSrc,
	fCIELabPix*  __restrict pDst,
	float*       __restrict csOut,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& src_pitch,
	const A_long& dst_pitch
) noexcept
{
	return;
}



template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr>
inline void ImprovedImageRestore
(
	const T* __restrict srcIn, /* for get alpha channel only */
	      T* __restrict dstOut,/* target buffer */
	const fXYZPix* __restrict xyz_out,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& srcPitch,
	const A_long& xyzPitch,
	const A_long& dstPitch,
	const float&  scale
) noexcept
{
	A_long i, j;

	auto const xyz2rgb = [&](const fXYZPix p) noexcept
	{
		fRGB outPix;
		constexpr float xyz_to_rgb[9] = { 3.2410f, -1.5374f, -0.4986f, -0.9692f, 1.8760f, 0.0416f, 0.0556f, -0.2040f, 1.0570f };
		outPix.R = p.X * xyz_to_rgb[0] + p.Y * xyz_to_rgb[1] + p.Z * xyz_to_rgb[2];
		outPix.G = p.X * xyz_to_rgb[3] + p.Y * xyz_to_rgb[4] + p.Z * xyz_to_rgb[5];
		outPix.B = p.X * xyz_to_rgb[6] + p.Y * xyz_to_rgb[7] + p.Z * xyz_to_rgb[8];
		return outPix;
	};

	auto const gamma_srgb = [&](const fRGB rgb) noexcept
	{
		fRGB outPix;
		constexpr float fExp = 1.f / 2.4f;
		outPix.R = rgb.R < 0.00304f ? 12.92f * rgb.R : FastCompute::Pow(rgb.R, fExp) - 0.055f;
		outPix.G = rgb.G < 0.00304f ? 12.92f * rgb.G : FastCompute::Pow(rgb.G, fExp) - 0.055f;
		outPix.B = rgb.B < 0.00304f ? 12.92f * rgb.B : FastCompute::Pow(rgb.B, fExp) - 0.055f;
		return outPix;
	};


	for (j = 0; j < sizeY; j++)
	{
		const T*       __restrict pSrcLine = srcIn   + j * srcPitch;
		      T*       __restrict pDstLine = dstOut  + j * dstPitch;
	    const fXYZPix* __restrict pXyzLine = xyz_out + j * xyzPitch;

		for (i = 0; i < sizeX; i++)
		{
			const fRGB rgbCorrect = gamma_srgb (xyz2rgb (pXyzLine[i]));

			/* copy from source Alpha-channel values */
			pDstLine[i].B = rgbCorrect.B * scale;
			pDstLine[i].G = rgbCorrect.G * scale;
			pDstLine[i].R = rgbCorrect.R * scale;
			pDstLine[i].A = pSrcLine[i].A;
		} /* for (i = 0; i < sizeX; i++) */

	} /* for (j = 0; j < sizeY; j++) */
	return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr>
inline void ImprovedImageRestore
(
	const T* __restrict srcIn, /* for get alpha channel only */
	      T* __restrict dstOut,/* target buffer */
	const fXYZPix* __restrict xyz_out,
	const A_long& sizeX,
	const A_long& sizeY,
	const A_long& srcPitch,
	const A_long& xyzPitch,
	const A_long& dstPitch,
	const float&  scale
) noexcept
{
	return;
}