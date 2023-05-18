#pragma once

#include "Common.hpp"
#include "CommonPixFormat.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"


template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr,
          typename U, std::enable_if_t<is_YUV_proc<U>::value>* = nullptr>
void imgRGB2YUV
(
	const T* __restrict srcImage,
	      U* __restrict dstImage,
	eCOLOR_SPACE transformSpace,
	int32_t sizeX,
	int32_t sizeY,
	int32_t src_line_pitch,
	int32_t dst_line_pitch,
	int32_t subtractor = 0
) noexcept
{
	const float* __restrict colorMatrix = RGB2YUV[transformSpace];

	for (int32_t j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = srcImage + j * src_line_pitch;
		      U* __restrict pDstLine = dstImage + j * dst_line_pitch;

		__VECTOR_ALIGNED__
		for (int32_t i = 0; i < sizeX; i++)
		{
			pDstLine[i].A = pSrcLine[i].A;
			pDstLine[i].Y = pSrcLine[i].R * colorMatrix[0] + pSrcLine[i].G * colorMatrix[1] + pSrcLine[i].B * colorMatrix[2];
			pDstLine[i].U = pSrcLine[i].R * colorMatrix[3] + pSrcLine[i].G * colorMatrix[4] + pSrcLine[i].B * colorMatrix[5] - subtractor;
			pDstLine[i].V = pSrcLine[i].R * colorMatrix[6] + pSrcLine[i].G * colorMatrix[7] + pSrcLine[i].B * colorMatrix[8] - subtractor;
		}
	}
	return;
}

template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr,
	      typename U, std::enable_if_t<is_RGB_proc<U>::value>* = nullptr>
inline void imgYUV2RGB
(
	const T* __restrict srcImage,
	      U* __restrict dstImage,
	eCOLOR_SPACE transformSpace,
	int32_t sizeX,
	int32_t sizeY,
	int32_t src_line_pitch,
	int32_t dst_line_pitch,
	int32_t addendum 
) noexcept
{
	const float* __restrict colorMatrix = YUV2RGB[transformSpace];

	for (int32_t j = 0; j < sizeY; j++)
	{
		const T* __restrict pSrcLine = srcImage + j * src_line_pitch;
		      U* __restrict pDstLine = dstImage + j * dst_line_pitch;

		__VECTOR_ALIGNED__
		for (int32_t i = 0; i < sizeX; i++)
		{
			pDstLine[i].A = pSrcLine[i].A;
			pDstLine[i].R = pSrcLine[i].Y * colorMatrix[0] + pSrcLine[i].U * colorMatrix[1] + pSrcLine[i].V * colorMatrix[2];
			pDstLine[i].G = pSrcLine[i].Y * colorMatrix[3] + pSrcLine[i].U * colorMatrix[4] + pSrcLine[i].V * colorMatrix[5] + addendum;
			pDstLine[i].B = pSrcLine[i].Y * colorMatrix[6] + pSrcLine[i].U * colorMatrix[7] + pSrcLine[i].V * colorMatrix[8] + addendum;
		}
	}
	return;
}


inline fCIELabPix RGB2CIELab
(
	const fRGB& pixelRGB, 
	const float* fReferences
) noexcept
{
	/* in first convert: sRGB -> XYZ */
	constexpr float reciproc12 = 1.f / 12.92f;
	constexpr float reciproc16 = 16.f / 116.f;
	constexpr float reciproc1 = 1.f / 1.055f;

	const float varR = ((pixelRGB.R > 0.04045f) ? FastCompute::Pow((pixelRGB.R + 0.055f) * reciproc1, 2.40f) : pixelRGB.R * reciproc12);
	const float varG = ((pixelRGB.G > 0.04045f) ? FastCompute::Pow((pixelRGB.G + 0.055f) * reciproc1, 2.40f) : pixelRGB.G * reciproc12);
	const float varB = ((pixelRGB.B > 0.04045f) ? FastCompute::Pow((pixelRGB.B + 0.055f) * reciproc1, 2.40f) : pixelRGB.B * reciproc12);

	const float X = varR * 41.24f + varG * 35.76f + varB * 18.05f;
	const float Y = varR * 21.26f + varG * 71.52f + varB * 7.220f;
	const float Z = varR * 1.930f + varG * 11.92f + varB * 95.05f;

	/* convert: XYZ - > Cie-L*ab */
	const float varX = X / fReferences[0];
	const float varY = Y / fReferences[1];
	const float varZ = Z / fReferences[2];

	const float vX = (varX > 0.0088560f) ? FastCompute::Cbrt(varX) : 7.7870f * varX + reciproc16;
	const float vY = (varY > 0.0088560f) ? FastCompute::Cbrt(varY) : 7.7870f * varY + reciproc16;
	const float vZ = (varZ > 0.0088560f) ? FastCompute::Cbrt(varZ) : 7.7870f * varZ + reciproc16;

	fCIELabPix pixelLAB;
	pixelLAB.L = 116.f * vX - 16.f;
	pixelLAB.a = 500.f * (vX - vY);
	pixelLAB.b = 200.f * (vY - vZ);

	return pixelLAB;
}

inline fRGB CIELab2RGB
(
	const fCIELabPix& pixelCIELab,
	const float* fReferences
) noexcept
{
	constexpr float reciproc7 = 1.f / 7.7870f;
	constexpr float reciproc100 = 1.f / 100.f;
	constexpr float reciproc116 = 1.f / 116.f;
	constexpr float reciproc200 = 1.f / 200.f;
	constexpr float reciproc500 = 1.f / 500.f;
	constexpr float reciproc6116 = 16.f / 116.f;

	/* CIEL*a*b -> XYZ */
	const float var_Y = (pixelCIELab.L + 16.f) * reciproc116;
	const float var_X = pixelCIELab.a * reciproc500 + var_Y;
	const float	var_Z = var_Y - pixelCIELab.b * reciproc200;

	const float x1 = (var_X > 0.2068930f) ? var_X * var_X * var_X : (var_X - reciproc6116) * reciproc7;
	const float y1 = (var_Y > 0.2068930f) ? var_Y * var_Y * var_Y : (var_Y - reciproc6116) * reciproc7;
	const float z1 = (var_Z > 0.2068930f) ? var_Z * var_Z * var_Z : (var_Z - reciproc6116) * reciproc7;

	const float X = x1 * fReferences[0] * reciproc100;
	const float Y = y1 * fReferences[1] * reciproc100;
	const float Z = z1 * fReferences[2] * reciproc100;

	const float var_R = X *  3.2406f + Y * -1.5372f + Z * -0.4986f;
	const float var_G = X * -0.9689f + Y *  1.8758f + Z *  0.0415f;
	const float var_B = X *  0.0557f + Y * -0.2040f + Z *  1.0570f;

	constexpr float reciproc24 = 1.f / 2.4f;
	fRGB pixelRGB;
	pixelRGB.R = (var_R > 0.0031308f ? 1.055f * (FastCompute::Pow(var_R, reciproc24)) - 0.055f : 12.92f * var_R);
	pixelRGB.G = (var_G > 0.0031308f ? 1.055f * (FastCompute::Pow(var_G, reciproc24)) - 0.055f : 12.92f * var_G);
	pixelRGB.B = (var_B > 0.0031308f ? 1.055f * (FastCompute::Pow(var_B, reciproc24)) - 0.055f : 12.92f * var_B);

	return pixelRGB;
}

