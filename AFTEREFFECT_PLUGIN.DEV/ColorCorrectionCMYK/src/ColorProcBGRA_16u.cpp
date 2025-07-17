#include "ColorCorrectionCMYK.hpp"
#include "ColorCorrectionEnums.hpp"
#include "RGB2CMYK.hpp"

PF_Err prProcessImage_BGRA_4444_16u_CMYK
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
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	float C, M, Y, K;
	float newC, newM, newY, newK;
	float newR, newG, newB;

	constexpr float reciproc32768 = 1.0f / 32768.0f;
	constexpr float reciproc100 = 1.0f / 100.0f;

#ifdef _DEBUG	
	C = M = Y = K = 0.f;
	newC = newM = newY = newK = 0.f;
	newR = newG = newB = 0.f;
#endif

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_BGRA_16u const& srcPixel = localSrc[line_idx + i];

			float const B = static_cast<float>(srcPixel.B) * reciproc32768;
			float const G = static_cast<float>(srcPixel.G) * reciproc32768;
			float const R = static_cast<float>(srcPixel.R) * reciproc32768;
			auto const  A = srcPixel.A;

			rgb_to_cmyk(R, G, B, C, M, Y, K);

			newC = CLAMP_VALUE(C + add_c * reciproc100, 0.f, 1.0f);
			newM = CLAMP_VALUE(M + add_m * reciproc100, 0.f, 1.0f);
			newY = CLAMP_VALUE(Y + add_y * reciproc100, 0.F, 1.0f);
			newK = CLAMP_VALUE(K + add_k * reciproc100, 0.f, 1.0f);

			cmyk_to_rgb(newC, newM, newY, newK, newR, newG, newB);

			/* put to output buffer updated value */
			localDst[line_idx + i].B = static_cast<A_u_short>(newB * 32768.f);
			localDst[line_idx + i].G = static_cast<A_u_short>(newG * 32768.f);
			localDst[line_idx + i].R = static_cast<A_u_short>(newR * 32768.f);
			localDst[line_idx + i].A = A;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err prProcessImage_BGRA_4444_16u_RGB
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
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	int newR, newG, newB;

#ifdef _DEBUG	
	newR = newG = newB = 0.f;
#endif

	for (auto j = 0; j < height; j++)
	{
		auto const& line_idx = j * line_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_BGRA_16u const& srcPixel = localSrc[line_idx + i];

			int const& B = srcPixel.B;
			int const& G = srcPixel.G;
			int const& R = srcPixel.R;
			int const& A = srcPixel.A;

			newB = B + static_cast<int>(add_b * 256.0f);
			newG = G + static_cast<int>(add_g * 256.0f);
			newR = R + static_cast<int>(add_r * 256.0f);

			/* put to output buffer updated value */
			localDst[line_idx + i].B = CLAMP_VALUE(newB, 0, 32767);
			localDst[line_idx + i].G = CLAMP_VALUE(newG, 0, 32767);
			localDst[line_idx + i].R = CLAMP_VALUE(newR, 0, 32767);
			localDst[line_idx + i].A = A;

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}