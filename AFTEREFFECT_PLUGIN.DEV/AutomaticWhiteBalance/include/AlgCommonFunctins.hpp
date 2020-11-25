#include "AlgMemoryHandler.hpp"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"


template <typename T>
inline void simple_image_copy
(
	const T* __restrict srcPix,
	T* __restrict dstPix,
	const A_long& width,
	const A_long& height,
	const A_long& srcPitch,
	const A_long& dstPitch
) noexcept
{
	const size_t line_size = width * sizeof(T);
	__VECTOR_ALIGNED__
	for (A_long i = 0; i < height; i++)
	{
		memcpy(&dstPix[i*dstPitch], &srcPix[i*srcPitch], line_size);
	}

	return;
}


template <typename T>
inline void collect_rgb_statistics
(
	const T* __restrict pSrc,
	const A_long& width,
	const A_long& height,
	const A_long& linePitch,
	const float& threshold,
	const eCOLOR_SPACE& colorSpace,
	float* u_Avg,
	float* v_Avg
) noexcept
{
	float U_bar = 0.f, V_bar = 0.f, F = 0.f;
	float Y, U, V;
	int32_t totalGray = 0;

	const float* __restrict colorMatrixIn = RGB2YUV[colorSpace];

	__VECTOR_ALIGNED__
	for (A_long j = 0; j < height; j++)
	{
		const A_long l_idx = j * linePitch; /* line IDX */
		for (A_long i = 0; i < width; i++)
		{
			const A_long p_idx = l_idx + i; /* pixel IDX */
											/* convert RGB to YUV color space */
			Y = pSrc[p_idx].R * colorMatrixIn[0] + pSrc[p_idx].G * colorMatrixIn[1] + pSrc[p_idx].B * colorMatrixIn[2];
			U = pSrc[p_idx].R * colorMatrixIn[3] + pSrc[p_idx].G * colorMatrixIn[4] + pSrc[p_idx].B * colorMatrixIn[5];
			V = pSrc[p_idx].R * colorMatrixIn[6] + pSrc[p_idx].G * colorMatrixIn[7] + pSrc[p_idx].B * colorMatrixIn[8];

			F = (abs(U) + abs(V)) / Y;
			if (F < threshold)
			{
				totalGray++;
				U_bar += U;
				V_bar += V;
			} /* if (F < T) */

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	if (nullptr != u_Avg)
		*u_Avg = U_bar / static_cast<float>(totalGray);
	if (nullptr != v_Avg)
		*v_Avg = V_bar / static_cast<float>(totalGray);

	return;
}




template <typename T>
inline void collect_yuv_statistics
(
	const T* __restrict pSrc,
	const A_long& width,
	const A_long& height,
	const A_long& linePitch,
	const float& threshold,
	const eCOLOR_SPACE& colorSpace,
	float* u_Avg,
	float* v_Avg
) noexcept
{
	float U_bar = 0.f, V_bar = 0.f, F = 0.f;
	float Y, U, V;
	int32_t totalGray = 0;

	const float* __restrict colorMatrixIn = RGB2YUV[colorSpace];
	constexpr float subtractor = (4 == sizeof(T) ? 128.f : 0.f);

	__VECTOR_ALIGNED__
		for (A_long j = 0; j < height; j++)
		{
			const A_long l_idx = j * linePitch; /* line IDX */
			for (A_long i = 0; i < width; i++)
			{
				const A_long p_idx = l_idx + i; /* pixel IDX */
												/* convert RGB to YUV color space */
				Y = pSrc[p_idx].Y;
				U = pSrc[p_idx].U - subtractor;
				V = pSrc[p_idx].V - subtractor;

				F = (abs(U) + abs(V)) / Y;
				if (F < threshold)
				{
					totalGray++;
					U_bar += U;
					V_bar += V;
				} /* if (F < T) */

			} /* for (i = 0; i < width; i++) */

		} /* for (j = 0; j < height; j++) */

	if (nullptr != u_Avg)
		*u_Avg = U_bar / static_cast<float>(totalGray);
	if (nullptr != v_Avg)
		*v_Avg = V_bar / static_cast<float>(totalGray);

	return;
}


template <typename T>
inline void image_rgb_correction
(
	const T* __restrict pSrc,		/* input data  */
	T* __restrict pDst,				/* output data */
	const A_long& width,
	const A_long& height,
	const A_long& srcPitch,
	const A_long& dstPitch,
	const float* __restrict correctionMatrix
) noexcept
{
	float newR, newG, newB;

	constexpr float whiteValue = (4 == sizeof(T) ? 255.0f : (8 == sizeof(T)) ? 32767.0f : f32_value_white);

	/* in second: perform balance based on computed coefficients */
	for (A_long j = 0; j < height; j++)
	{
		const A_long l_idx_src = j * srcPitch;
		const A_long l_idx_dst = j * dstPitch;

		__VECTOR_ALIGNED__
		for (A_long i = 0; i < width; i++)
		{
			const A_long p_idx_src = l_idx_src + i;
			const A_long p_idx_dst = l_idx_dst + i;

			newR = correctionMatrix[0] * pSrc[p_idx_src].R;
			newG = correctionMatrix[1] * pSrc[p_idx_src].G;
			newB = correctionMatrix[2] * pSrc[p_idx_src].B;

			pDst[p_idx_dst].A = pSrc[p_idx_src].A; /* copy ALPHA channel from source */
			pDst[p_idx_dst].R = (CLAMP_VALUE(newR, f32_value_black, whiteValue));
			pDst[p_idx_dst].G = (CLAMP_VALUE(newG, f32_value_black, whiteValue));
			pDst[p_idx_dst].B = (CLAMP_VALUE(newB, f32_value_black, whiteValue));

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}


template <typename T>
inline void image_yuv_correction
(
	const T* __restrict pSrc,		/* input data  */
	T* __restrict pDst,				/* output data */
	const A_long& width,
	const A_long& height,
	const A_long& srcPitch,
	const A_long& dstPitch,
	const float* __restrict correctionMatrix,
	const bool& isBT709
) noexcept
{
	float R, G, B;
	float newR, newG, newB;
	float newY, newU, newV;

	constexpr float whiteValue = (4 == sizeof(T) ? 255.0f : f32_value_white);
	constexpr float subtractor = (4 == sizeof(T) ? 128.f  : 0.f);

	const float* __restrict yuv2rgb = ((true == isBT709) ? YUV2RGB[BT709] : YUV2RGB[BT601]);
	const float* __restrict rgb2yuv = ((true == isBT709) ? RGB2YUV[BT709] : RGB2YUV[BT601]);

	/* in second: perform balance based on computed coefficients */
	for (A_long j = 0; j < height; j++)
	{
		const A_long l_idx_src = j * srcPitch;
		const A_long l_idx_dst = j * dstPitch;

		__VECTOR_ALIGNED__
		for (A_long i = 0; i < width; i++)
		{
			const A_long p_idx_src = l_idx_src + i;
			const A_long p_idx_dst = l_idx_dst + i;

			R = pSrc[p_idx_src].Y * yuv2rgb[0] + (pSrc[p_idx_src].U - subtractor) * yuv2rgb[1] + (pSrc[p_idx_src].V - subtractor) * yuv2rgb[2];
			G = pSrc[p_idx_src].Y * yuv2rgb[3] + (pSrc[p_idx_src].U - subtractor) * yuv2rgb[4] + (pSrc[p_idx_src].V - subtractor) * yuv2rgb[5];
			B = pSrc[p_idx_src].Y * yuv2rgb[6] + (pSrc[p_idx_src].U - subtractor) * yuv2rgb[7] + (pSrc[p_idx_src].V - subtractor) * yuv2rgb[8];

			newR = CLAMP_VALUE(correctionMatrix[0] * R, 0.f, whiteValue);
			newG = CLAMP_VALUE(correctionMatrix[1] * G, 0.f, whiteValue);
			newB = CLAMP_VALUE(correctionMatrix[2] * B, 0.f, whiteValue);

			newY = newR * rgb2yuv[0] + newG * rgb2yuv[1] + newB * rgb2yuv[2];
			newU = newR * rgb2yuv[3] + newG * rgb2yuv[4] + newB * rgb2yuv[5];
			newV = newR * rgb2yuv[6] + newG * rgb2yuv[7] + newB * rgb2yuv[8];

			pDst[p_idx_dst].A = pSrc[p_idx_src].A; /* copy ALPHA channel from source */
			pDst[p_idx_dst].Y = newY;
			pDst[p_idx_dst].U = (newU + subtractor);
			pDst[p_idx_dst].V = (newV + subtractor);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return;
}

