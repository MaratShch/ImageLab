#include "ColorCorrectionCMYK.hpp"
#include "ColorCorrectionEnums.hpp"
#include "RGB2CMYK.hpp"

PF_Err aeProcessImage_ARGB_4444_8u_CMYK
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
	const PF_EffectWorld*   __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
	PF_Pixel_ARGB_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);

	auto const src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	auto const height = output->height;
	auto const width  = output->width;

	float C, M, Y, K;
	float newC, newM, newY, newK;
	float newR, newG, newB;

	constexpr float reciproc255 = 1.0f / 255.0f;
	constexpr float reciproc100 = 1.0f / 100.0f;

#ifdef _DEBUG	
	C = M = Y = K = 0.f;
	newC = newM = newY = newK = 0.f;
	newR = newG = newB = 0.f;
#endif

	for (auto j = 0; j < height; j++)
	{
		auto const src_line_idx = j * src_pitch;
		auto const dst_line_idx = j * dst_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_ARGB_8u const& srcPixel = localSrc[src_line_idx + i];

			int const& A   = srcPixel.A;
			float const& R = static_cast<float>(srcPixel.R) * reciproc255;
			float const& G = static_cast<float>(srcPixel.G) * reciproc255;
			float const& B = static_cast<float>(srcPixel.B) * reciproc255;

			rgb_to_cmyk(R, G, B, C, M, Y, K);

			newC = CLAMP_VALUE(C + add_c * reciproc100, 0.f, 1.0f);
			newM = CLAMP_VALUE(M + add_m * reciproc100, 0.f, 1.0f);
			newY = CLAMP_VALUE(Y + add_y * reciproc100, 0.F, 1.0f);
			newK = CLAMP_VALUE(K + add_k * reciproc100, 0.f, 1.0f);

			cmyk_to_rgb(newC, newM, newY, newK, newR, newG, newB);

			/* put to output buffer updated value */
			localDst[dst_line_idx + i].A = A;
			localDst[dst_line_idx + i].R = static_cast<A_u_char>(newR * 255.f);
			localDst[dst_line_idx + i].G = static_cast<A_u_char>(newG * 255.f);
			localDst[dst_line_idx + i].B = static_cast<A_u_char>(newB * 255.f);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err aeProcessImage_ARGB_4444_8u_RGB
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
	const PF_EffectWorld*   __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[COLOR_CORRECT_INPUT]->u.ld);
	const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
	PF_Pixel_ARGB_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);

	auto const src_pitch = input->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	auto const height = output->height;
	auto const width = output->width;

	int newR, newG, newB;

#ifdef _DEBUG	
	newR = newG = newB = 0.f;
#endif

	for (auto j = 0; j < height; j++)
	{
		auto const& src_line_idx = j * src_pitch;
		auto const& dst_line_idx = j * dst_pitch;

		__VECTOR_ALIGNED__
		for (auto i = 0; i < width; i++)
		{
			PF_Pixel_ARGB_8u const& srcPixel = localSrc[src_line_idx + i];

			int const& A = srcPixel.A;
			int const& R = srcPixel.R;
			int const& G = srcPixel.G;
			int const& B = srcPixel.B;

			newR = R + static_cast<int>(add_r);
			newG = G + static_cast<int>(add_g);
			newB = B + static_cast<int>(add_b);

			/* put to output buffer updated value */
			localDst[dst_line_idx + i].A = A;
			localDst[dst_line_idx + i].R = CLAMP_VALUE(newR, 0, 255);
			localDst[dst_line_idx + i].G = CLAMP_VALUE(newG, 0, 255);
			localDst[dst_line_idx + i].B = CLAMP_VALUE(newB, 0, 255);

		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}