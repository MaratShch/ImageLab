#pragma once

#include "ColorTransform.hpp"
#include "CommonAuxPixFormat.hpp"

constexpr eCOLOR_OBSERVER   gDefaultObserver   = observer_CIE_1931;
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
	const A_long& dst_pitch,
	const float&  scale
) noexcept
{
	CACHE_ALIGN constexpr float yuv2rgb[9] =
	{
		YUV2RGB[BT709][0], YUV2RGB[BT709][1], YUV2RGB[BT709][2],
		YUV2RGB[BT709][3], YUV2RGB[BT709][4], YUV2RGB[BT709][5],
		YUV2RGB[BT709][6], YUV2RGB[BT709][7], YUV2RGB[BT709][8],
	};
	  constexpr float colorReferences[3] =
	{
		cCOLOR_ILLUMINANT[gDefaultObserver][gDefaultIlluminant][0],
		cCOLOR_ILLUMINANT[gDefaultObserver][gDefaultIlluminant][1],
		cCOLOR_ILLUMINANT[gDefaultObserver][gDefaultIlluminant][2]
	};
	float cs_out_max = FLT_MIN;
	const float subtractor = (scale < 0.90f ? 128.0f : 0.0f);

	for (A_long j = 0; j < sizeY; j++)
	{
		const T*     __restrict pSrcLine = pSrc + j * src_pitch;
		fCIELabPix*  __restrict pDstLine = pDst + j * sizeX;
		float*       __restrict pCsOutLine = csOut + j * sizeX;

		/* lambda for convert YUV to sRGB */
		auto const convert2sRGB = [&](const T p, const float* pYuv2Rgb, const float sub, const float val) noexcept
		{
			auto const& Y = p.Y;
			auto const& U = p.U - sub;
			auto const& V = p.V - sub;
			const float R = Y * pYuv2Rgb[0] + U * pYuv2Rgb[1] + V * pYuv2Rgb[2];
			const float G = Y * pYuv2Rgb[3] + U * pYuv2Rgb[4] + V * pYuv2Rgb[5];
			const float B = Y * pYuv2Rgb[6] + U * pYuv2Rgb[7] + V * pYuv2Rgb[8];
			fRGB outPix;
			outPix.R = R * val, outPix.G = G * val, outPix.B = B * val;
			return outPix;
		};

		for (A_long i = 0; i < sizeX; i++)
		{
			pDstLine[i] = RGB2CIELab(convert2sRGB(pSrcLine[i], yuv2rgb, subtractor, scale), colorReferences);
			auto const& a = pDstLine[i].a;
			auto const& b = pDstLine[i].b;
			const float cs_out = FastCompute::Sqrt(a * a + b * b);
			pCsOutLine[i] = cs_out;
			cs_out_max = FastCompute::Max(cs_out_max, cs_out);
		}
	}
	return cs_out_max;
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
	auto const xyz2rgb = [&](const fXYZPix p) noexcept
	{
		fRGB outPix;
		constexpr float xyz_to_rgb[9] =
		{
			XYZtosRGB[0], XYZtosRGB[1], XYZtosRGB[2],
			XYZtosRGB[3], XYZtosRGB[4], XYZtosRGB[5],
			XYZtosRGB[6], XYZtosRGB[7], XYZtosRGB[8]
		};
		
		outPix.R = p.X * xyz_to_rgb[0] + p.Y * xyz_to_rgb[1] + p.Z * xyz_to_rgb[2];
		outPix.G = p.X * xyz_to_rgb[3] + p.Y * xyz_to_rgb[4] + p.Z * xyz_to_rgb[5];
		outPix.B = p.X * xyz_to_rgb[6] + p.Y * xyz_to_rgb[7] + p.Z * xyz_to_rgb[8];
		return outPix;
	};

	auto const gamma_srgb = [&](const fRGB rgb, const float scale) noexcept
	{
		constexpr float fExp = 1.f / 2.4f;
		const float R = rgb.R < 0.00304f ? 12.92f * rgb.R : FastCompute::Pow(rgb.R, fExp) - 0.055f;
		const float G = rgb.G < 0.00304f ? 12.92f * rgb.G : FastCompute::Pow(rgb.G, fExp) - 0.055f;
		const float B = rgb.B < 0.00304f ? 12.92f * rgb.B : FastCompute::Pow(rgb.B, fExp) - 0.055f;

		fRGB outPix;
		outPix.R = CLAMP_VALUE(R * scale, 0.f, scale);
		outPix.G = CLAMP_VALUE(G * scale, 0.f, scale);
		outPix.B = CLAMP_VALUE(B * scale, 0.f, scale);
		return outPix;
	};


	for (A_long j = 0; j < sizeY; j++)
	{
		const T*       __restrict pSrcLine = srcIn   + j * srcPitch;
		      T*       __restrict pDstLine = dstOut  + j * dstPitch;
	    const fXYZPix* __restrict pXyzLine = xyz_out + j * xyzPitch;

		for (A_long i = 0; i < sizeX; i++)
		{
			const fRGB rgbCorrect = gamma_srgb (xyz2rgb (pXyzLine[i]), scale);

			/* copy from source Alpha-channel values */
			pDstLine[i].B = rgbCorrect.B;
			pDstLine[i].G = rgbCorrect.G;
			pDstLine[i].R = rgbCorrect.R;
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
	constexpr float rgb2yuv[9] =
	{
		RGB2YUV[BT709][0], RGB2YUV[BT709][1], RGB2YUV[BT709][2],
		RGB2YUV[BT709][3], RGB2YUV[BT709][4], RGB2YUV[BT709][5],
		RGB2YUV[BT709][6], RGB2YUV[BT709][7], RGB2YUV[BT709][8]
	};

	auto const xyz2rgb = [&](const fXYZPix p) noexcept
	{
		fRGB outPix;
		constexpr float xyz_to_rgb[9] =
		{
			XYZtosRGB[0], XYZtosRGB[1], XYZtosRGB[2],
			XYZtosRGB[3], XYZtosRGB[4], XYZtosRGB[5],
			XYZtosRGB[6], XYZtosRGB[7], XYZtosRGB[8]
		};

		outPix.R = p.X * xyz_to_rgb[0] + p.Y * xyz_to_rgb[1] + p.Z * xyz_to_rgb[2];
		outPix.G = p.X * xyz_to_rgb[3] + p.Y * xyz_to_rgb[4] + p.Z * xyz_to_rgb[5];
		outPix.B = p.X * xyz_to_rgb[6] + p.Y * xyz_to_rgb[7] + p.Z * xyz_to_rgb[8];
		return outPix;
	};

	auto const gamma_srgb = [&](const fRGB rgb, const float scale) noexcept
	{
		constexpr float fExp = 1.f / 2.4f;
		const float R = rgb.R < 0.00304f ? 12.92f * rgb.R : FastCompute::Pow(rgb.R, fExp) - 0.055f;
		const float G = rgb.G < 0.00304f ? 12.92f * rgb.G : FastCompute::Pow(rgb.G, fExp) - 0.055f;
		const float B = rgb.B < 0.00304f ? 12.92f * rgb.B : FastCompute::Pow(rgb.B, fExp) - 0.055f;

		fRGB outPix;
		outPix.R = CLAMP_VALUE(R, 0.f, scale);
		outPix.G = CLAMP_VALUE(G, 0.f, scale);
		outPix.B = CLAMP_VALUE(B, 0.f, scale);
		return outPix;
	};


	for (A_long j = 0; j < sizeY; j++)
	{
		const T*       __restrict pSrcLine = srcIn + j * srcPitch;
		T*             __restrict pDstLine = dstOut + j * dstPitch;
		const fXYZPix* __restrict pXyzLine = xyz_out + j * xyzPitch;

		for (A_long i = 0; i < sizeX; i++)
		{
			const fRGB rgbCorrect = gamma_srgb(xyz2rgb(pXyzLine[i]), scale);
			pDstLine[i].Y = rgbCorrect.R * rgb2yuv[0] + rgbCorrect.G * rgb2yuv[1] + rgbCorrect.B * rgb2yuv[2];
			pDstLine[i].U = rgbCorrect.R * rgb2yuv[3] + rgbCorrect.G * rgb2yuv[4] + rgbCorrect.B * rgb2yuv[5];
			pDstLine[i].V = rgbCorrect.R * rgb2yuv[6] + rgbCorrect.G * rgb2yuv[7] + rgbCorrect.B * rgb2yuv[8];
			pDstLine[i].A = pSrcLine[i].A;
		} /* for (i = 0; i < sizeX; i++) */

	} /* for (j = 0; j < sizeY; j++) */
	return;
}