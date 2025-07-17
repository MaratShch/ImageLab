#include "ColorCorrectionCMYK.hpp"
#include "ColorCorrectionEnums.hpp"
#include "RGB2CMYK.hpp"

PF_Err prProcessImage_BGRA_4444_32f_CMYK
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_c,
	float           add_m,
	float           add_y,
	float           add_k
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	float C, M, Y, K;
	float newC, newM, newY, newK;
	float newR, newG, newB;

	constexpr float reciproc100 = 1.0f / 100.0f;

#ifdef _DEBUG	
	C = M = Y = K = 0.f;
	newC = newM = newY = newK = 0.f;
	newR = newG = newB = 0.f;
#endif

	for (auto j = 0; j < height; j++)
	{
		auto const line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_BGRA_32f const& srcPixel = localSrc[line_idx + i];

			float const& B = srcPixel.B;
			float const& G = srcPixel.G;
			float const& R = srcPixel.R;
			auto const&  A = srcPixel.A;

			rgb_to_cmyk(R, G, B, C, M, Y, K);

			newC = CLAMP_VALUE(C + add_c * reciproc100, 0.f, 1.0f);
			newM = CLAMP_VALUE(M + add_m * reciproc100, 0.f, 1.0f);
			newY = CLAMP_VALUE(Y + add_y * reciproc100, 0.F, 1.0f);
			newK = CLAMP_VALUE(K + add_k * reciproc100, 0.f, 1.0f);

			cmyk_to_rgb(newC, newM, newY, newK, newR, newG, newB);

			/* put to output buffer updated value */
			localDst[line_idx + i].B = newB;
			localDst[line_idx + i].G = newG;
			localDst[line_idx + i].R = newR;
			localDst[line_idx + i].A = A;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err prProcessImage_BGRA_4444_32f_RGB
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_r,
	float           add_g,
	float           add_b
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	constexpr float reciproc = 1.0f / static_cast<float>(coarse_max_level);
	float const addR = add_r * reciproc;
	float const addG = add_g * reciproc;
	float const addB = add_b * reciproc;

	for (auto j = 0; j < height; j++)
	{
		auto const line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			auto const pix_idx = line_idx + i;

			/* put to output buffer updated value */
			localDst[pix_idx].B = CLAMP_VALUE(localSrc[pix_idx].B + addB, f32_value_black, f32_value_white);
			localDst[pix_idx].G = CLAMP_VALUE(localSrc[pix_idx].G + addG, f32_value_black, f32_value_white);
			localDst[pix_idx].R = CLAMP_VALUE(localSrc[pix_idx].R + addR, f32_value_black, f32_value_white);
			localDst[pix_idx].A = localSrc[pix_idx].A;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}