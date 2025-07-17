#include "ColorCorrectionCMYK.hpp"
#include "ColorCorrectionEnums.hpp"
#include "RGB2CMYK.hpp"
#include "ColorTransformMatrix.hpp"


PF_Err prProcessImage_VUYA_4444_8u_CMYK
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_c,
	float           add_m,
	float           add_y,
	float           add_k,
	bool            isBT709
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*        __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	const float* __restrict yuv2rgb = (true == isBT709 ? YUV2RGB[BT709] : YUV2RGB[BT601]);
	const float* __restrict rgb2yuv = (true == isBT709 ? RGB2YUV[BT709] : RGB2YUV[BT601]);

	float y, u, v;
	float C, M, Y, K;
	float newC, newM, newY, newK;
	float newR, newG, newB;

	constexpr float reciproc255 = 1.0f / 255.0f;
	constexpr float reciproc100 = 1.0f / 100.0f;

#ifdef _DEBUG	
	C = M = Y = K = 0.f;
	newC = newM = newY = newK = 0.f;
	newR = newG = newB = 0.f;
	y = u = v = 0.f;
#endif

	for (auto j = 0; j < height; j++)
	{
		auto const line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_VUYA_8u const& srcPixel = localSrc[line_idx + i];

			float const y = static_cast<float>(srcPixel.Y);
			float const u = static_cast<float>(srcPixel.U) - 128.f;
			float const v = static_cast<float>(srcPixel.V) - 128.f;
			int const   A = srcPixel.A;

			float const R = (y * yuv2rgb[0] + u * yuv2rgb[1] + v * yuv2rgb[2]) * reciproc255;
			float const G = (y * yuv2rgb[3] + u * yuv2rgb[4] + v * yuv2rgb[5]) * reciproc255;
			float const B = (y * yuv2rgb[6] + u * yuv2rgb[7] + v * yuv2rgb[8]) * reciproc255;

			rgb_to_cmyk(R, G, B, C, M, Y, K);

			newC = CLAMP_VALUE(C + add_c * reciproc100, 0.f, 1.0f);
			newM = CLAMP_VALUE(M + add_m * reciproc100, 0.f, 1.0f);
			newY = CLAMP_VALUE(Y + add_y * reciproc100, 0.F, 1.0f);
			newK = CLAMP_VALUE(K + add_k * reciproc100, 0.f, 1.0f);

			cmyk_to_rgb(newC, newM, newY, newK, newR, newG, newB);

			/* put to output buffer updated value */
			localDst[line_idx + i].Y = static_cast<A_u_char>((newR * rgb2yuv[0] + newG * rgb2yuv[1] + newB * rgb2yuv[2]) * 255.0f);
			localDst[line_idx + i].U = static_cast<A_u_char>((newR * rgb2yuv[3] + newG * rgb2yuv[4] + newB * rgb2yuv[5]) * 255.0f + 128.f);
			localDst[line_idx + i].V = static_cast<A_u_char>((newR * rgb2yuv[6] + newG * rgb2yuv[7] + newB * rgb2yuv[8]) * 255.0f + 128.f);
			localDst[line_idx + i].A = A;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err prProcessImage_VUYA_4444_8u_RGB
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_r,
	float           add_g,
	float           add_b,
	bool            isBT709
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*        __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	const float* __restrict yuv2rgb = (true == isBT709 ? YUV2RGB[BT709] : YUV2RGB[BT601]);
	const float* __restrict rgb2yuv = (true == isBT709 ? RGB2YUV[BT709] : RGB2YUV[BT601]);

	float newR, newG, newB;
	float newY, newU, newV;

#ifdef _DEBUG	
	newR = newG = newB = 0.f;
	newY = newU = newV = 0.f;
#endif

	for (auto j = 0; j < height; j++)
	{
		auto const line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_VUYA_8u const& srcPixel = localSrc[line_idx + i];

			float const y = static_cast<float>(srcPixel.Y);
			float const u = static_cast<float>(srcPixel.U) - 128.f;
			float const v = static_cast<float>(srcPixel.V) - 128.f;
			int const  A = srcPixel.A;

			float const R = y * yuv2rgb[0] + u * yuv2rgb[1] + v * yuv2rgb[2];
			float const G = y * yuv2rgb[3] + u * yuv2rgb[4] + v * yuv2rgb[5];
			float const B = y * yuv2rgb[6] + u * yuv2rgb[7] + v * yuv2rgb[8];

			newR = CLAMP_VALUE(R + add_r, 0.f, 255.f);
			newG = CLAMP_VALUE(G + add_g, 0.f, 255.f);
			newB = CLAMP_VALUE(B + add_b, 0.f, 255.f);

			newY = (newR * rgb2yuv[0] + newG * rgb2yuv[1] + newB * rgb2yuv[2]);
			newU = (newR * rgb2yuv[3] + newG * rgb2yuv[4] + newB * rgb2yuv[5]) + 128.0f;
			newV = (newR * rgb2yuv[6] + newG * rgb2yuv[7] + newB * rgb2yuv[8]) + 128.0f;

			/* put to output buffer updated value */
			localDst[line_idx + i].Y = static_cast<A_u_char>(newY);
			localDst[line_idx + i].U = static_cast<A_u_char>(newU);
			localDst[line_idx + i].V = static_cast<A_u_char>(newV);
			localDst[line_idx + i].A = A;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}