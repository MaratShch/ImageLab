#include "ImageEqualization.hpp"
#include "FastAriphmetics.hpp"
#include "ColorTransformMatrix.hpp"


template <typename T>
inline const T DarkLut(const T& luma, const float& coeff) noexcept
{
	const float fVal = static_cast<float>(luma) / coeff;
	return static_cast<T>(fVal * fVal * coeff);
}


PF_Err PR_ImageEq_Dark_VUYA_4444_8u_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	A_long i, j;
	const PF_LayerDef*      __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	      PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	constexpr float coeff = static_cast<float>(u8_value_white);

	for (j = 0; j < height; j++)
	{
		const PF_Pixel_VUYA_8u* __restrict lineSrc = localSrc + j * line_pitch;
		      PF_Pixel_VUYA_8u* __restrict lineDst = localDst + j * line_pitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			lineDst[i].Y = DarkLut(lineSrc[i].Y, coeff);
			lineDst[i].U = lineSrc[i].U;
			lineDst[i].V = lineSrc[i].V;
			lineDst[i].A = lineSrc[i].A;
		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Dark_VUYA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PR_ImageEq_Dark_VUYA_4444_8u_709(in_data, out_data, params, output);
}


PF_Err PR_ImageEq_Dark_VUYA_4444_32f_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	A_long i, j;
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	      PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

	constexpr float coeff = 1.f;

	for (j = 0; j < height; j++)
	{
		const PF_Pixel_VUYA_32f* __restrict lineSrc = localSrc + j * line_pitch;
		      PF_Pixel_VUYA_32f* __restrict lineDst = localDst + j * line_pitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			lineDst[i].Y = DarkLut(lineSrc[i].Y, coeff);
			lineDst[i].U = lineSrc[i].U;
			lineDst[i].V = lineSrc[i].V;
			lineDst[i].A = lineSrc[i].A;
		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Dark_VUYA_4444_32f
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	return PR_ImageEq_Dark_VUYA_4444_32f_709(in_data, out_data, params, output);
}


PF_Err PR_ImageEq_Dark_BGRA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	A_long i, j;
	const PF_LayerDef*      __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	      PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);

	const float* __restrict rgb2yuv = RGB2YUV[BT709];
	const float* __restrict yuv2rgb = YUV2RGB[BT709];

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	constexpr float coeff = static_cast<float>(u8_value_white);

	for (j = 0; j < height; j++)
	{
		const PF_Pixel_BGRA_8u* __restrict lineSrc = localSrc + j * line_pitch;
		      PF_Pixel_BGRA_8u* __restrict lineDst = localDst + j * line_pitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const float Y = static_cast<float>(lineSrc[i].R) * rgb2yuv[0] + static_cast<float>(lineSrc[i].G) * rgb2yuv[1] + static_cast<float>(lineSrc[i].B) * rgb2yuv[2];
			const float U = static_cast<float>(lineSrc[i].R) * rgb2yuv[3] + static_cast<float>(lineSrc[i].G) * rgb2yuv[4] + static_cast<float>(lineSrc[i].B) * rgb2yuv[5];
			const float V = static_cast<float>(lineSrc[i].R) * rgb2yuv[6] + static_cast<float>(lineSrc[i].G) * rgb2yuv[7] + static_cast<float>(lineSrc[i].B) * rgb2yuv[8];
			const float lut_Y = DarkLut(Y, coeff);

			lineDst[i].R = static_cast<A_u_char>(CLAMP_VALUE(lut_Y * yuv2rgb[0] + U * yuv2rgb[1] + V * yuv2rgb[2], static_cast<float>(u8_value_black), static_cast<float>(u8_value_white)));
			lineDst[i].G = static_cast<A_u_char>(CLAMP_VALUE(lut_Y * yuv2rgb[3] + U * yuv2rgb[4] + V * yuv2rgb[5], static_cast<float>(u8_value_black), static_cast<float>(u8_value_white)));
			lineDst[i].B = static_cast<A_u_char>(CLAMP_VALUE(lut_Y * yuv2rgb[6] + U * yuv2rgb[7] + V * yuv2rgb[8], static_cast<float>(u8_value_black), static_cast<float>(u8_value_white)));
			lineDst[i].A = lineSrc[i].A;
		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Dark_BGRA_4444_16u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	A_long i, j;
	const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	      PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);

	const float* __restrict rgb2yuv = RGB2YUV[BT709];
	const float* __restrict yuv2rgb = YUV2RGB[BT709];

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	constexpr float coeff = static_cast<float>(u16_value_white);

	for (j = 0; j < height; j++)
	{
		const PF_Pixel_BGRA_16u* __restrict lineSrc = localSrc + j * line_pitch;
		      PF_Pixel_BGRA_16u* __restrict lineDst = localDst + j * line_pitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const float Y = static_cast<float>(lineSrc[i].R) * rgb2yuv[0] + static_cast<float>(lineSrc[i].G) * rgb2yuv[1] + static_cast<float>(lineSrc[i].B) * rgb2yuv[2];
			const float U = static_cast<float>(lineSrc[i].R) * rgb2yuv[3] + static_cast<float>(lineSrc[i].G) * rgb2yuv[4] + static_cast<float>(lineSrc[i].B) * rgb2yuv[5];
			const float V = static_cast<float>(lineSrc[i].R) * rgb2yuv[6] + static_cast<float>(lineSrc[i].G) * rgb2yuv[7] + static_cast<float>(lineSrc[i].B) * rgb2yuv[8];
			const float lut_Y = DarkLut(Y, coeff);

			lineDst[i].R = static_cast<A_u_short>(CLAMP_VALUE(lut_Y * yuv2rgb[0] + U * yuv2rgb[1] + V * yuv2rgb[2], static_cast<float>(u16_value_black), static_cast<float>(u16_value_white)));
			lineDst[i].G = static_cast<A_u_short>(CLAMP_VALUE(lut_Y * yuv2rgb[3] + U * yuv2rgb[4] + V * yuv2rgb[5], static_cast<float>(u16_value_black), static_cast<float>(u16_value_white)));
			lineDst[i].B = static_cast<A_u_short>(CLAMP_VALUE(lut_Y * yuv2rgb[6] + U * yuv2rgb[7] + V * yuv2rgb[8], static_cast<float>(u16_value_black), static_cast<float>(u16_value_white)));
			lineDst[i].A = lineSrc[i].A;
		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Dark_BGRA_4444_32f
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	A_long i, j;
	const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	      PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);

	const float* __restrict rgb2yuv = RGB2YUV[BT709];
	const float* __restrict yuv2rgb = YUV2RGB[BT709];

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	constexpr float coeff = 1.f;

	for (j = 0; j < height; j++)
	{
		const PF_Pixel_BGRA_32f* __restrict lineSrc = localSrc + j * line_pitch;
		      PF_Pixel_BGRA_32f* __restrict lineDst = localDst + j * line_pitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const float Y = static_cast<float>(lineSrc[i].R) * rgb2yuv[0] + static_cast<float>(lineSrc[i].G) * rgb2yuv[1] + static_cast<float>(lineSrc[i].B) * rgb2yuv[2];
			const float U = static_cast<float>(lineSrc[i].R) * rgb2yuv[3] + static_cast<float>(lineSrc[i].G) * rgb2yuv[4] + static_cast<float>(lineSrc[i].B) * rgb2yuv[5];
			const float V = static_cast<float>(lineSrc[i].R) * rgb2yuv[6] + static_cast<float>(lineSrc[i].G) * rgb2yuv[7] + static_cast<float>(lineSrc[i].B) * rgb2yuv[8];
			const float lut_Y = DarkLut(Y, coeff);

			lineDst[i].R = CLAMP_VALUE(lut_Y * yuv2rgb[0] + U * yuv2rgb[1] + V * yuv2rgb[2], f32_value_black, f32_value_white);
			lineDst[i].G = CLAMP_VALUE(lut_Y * yuv2rgb[3] + U * yuv2rgb[4] + V * yuv2rgb[5], f32_value_black, f32_value_white);
			lineDst[i].B = CLAMP_VALUE(lut_Y * yuv2rgb[6] + U * yuv2rgb[7] + V * yuv2rgb[8], f32_value_black, f32_value_white);
			lineDst[i].A = lineSrc[i].A;
		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err AE_ImageEq_Dark_ARGB_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	A_long i, j;
	const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
	      PF_Pixel_ARGB_8u*  __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

	const float* __restrict rgb2yuv = RGB2YUV[BT709];
	const float* __restrict yuv2rgb = YUV2RGB[BT709];

	const auto& height = output->height;
	const auto& width  = output->width;
	const auto src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	const auto dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	constexpr float coeff = static_cast<float>(u8_value_white);

	for (j = 0; j < height; j++)
	{
		const PF_Pixel_ARGB_8u* __restrict lineSrc = localSrc + j * src_line_pitch;
		      PF_Pixel_ARGB_8u* __restrict lineDst = localDst + j * dst_line_pitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const float Y = static_cast<float>(lineSrc[i].R) * rgb2yuv[0] + static_cast<float>(lineSrc[i].G) * rgb2yuv[1] + static_cast<float>(lineSrc[i].B) * rgb2yuv[2];
			const float U = static_cast<float>(lineSrc[i].R) * rgb2yuv[3] + static_cast<float>(lineSrc[i].G) * rgb2yuv[4] + static_cast<float>(lineSrc[i].B) * rgb2yuv[5];
			const float V = static_cast<float>(lineSrc[i].R) * rgb2yuv[6] + static_cast<float>(lineSrc[i].G) * rgb2yuv[7] + static_cast<float>(lineSrc[i].B) * rgb2yuv[8];
			const float lut_Y = DarkLut(Y, coeff);

			lineDst[i].R = static_cast<A_u_char>(CLAMP_VALUE(lut_Y * yuv2rgb[0] + U * yuv2rgb[1] + V * yuv2rgb[2], static_cast<float>(u8_value_black), static_cast<float>(u8_value_white)));
			lineDst[i].G = static_cast<A_u_char>(CLAMP_VALUE(lut_Y * yuv2rgb[3] + U * yuv2rgb[4] + V * yuv2rgb[5], static_cast<float>(u8_value_black), static_cast<float>(u8_value_white)));
			lineDst[i].B = static_cast<A_u_char>(CLAMP_VALUE(lut_Y * yuv2rgb[6] + U * yuv2rgb[7] + V * yuv2rgb[8], static_cast<float>(u8_value_black), static_cast<float>(u8_value_white)));
			lineDst[i].A = lineSrc[i].A;
		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err AE_ImageEq_Dark_ARGB_4444_16u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	A_long i, j;
	const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
	      PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

	const float* __restrict rgb2yuv = RGB2YUV[BT709];
	const float* __restrict yuv2rgb = YUV2RGB[BT709];

	const auto& height = output->height;
	const auto& width  = output->width;
	const auto src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	const auto dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

	constexpr float coeff = static_cast<float>(u16_value_white);

	for (j = 0; j < height; j++)
	{
		const PF_Pixel_ARGB_16u* __restrict lineSrc = localSrc + j * src_line_pitch;
		      PF_Pixel_ARGB_16u* __restrict lineDst = localDst + j * dst_line_pitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const float Y = static_cast<float>(lineSrc[i].R) * rgb2yuv[0] + static_cast<float>(lineSrc[i].G) * rgb2yuv[1] + static_cast<float>(lineSrc[i].B) * rgb2yuv[2];
			const float U = static_cast<float>(lineSrc[i].R) * rgb2yuv[3] + static_cast<float>(lineSrc[i].G) * rgb2yuv[4] + static_cast<float>(lineSrc[i].B) * rgb2yuv[5];
			const float V = static_cast<float>(lineSrc[i].R) * rgb2yuv[6] + static_cast<float>(lineSrc[i].G) * rgb2yuv[7] + static_cast<float>(lineSrc[i].B) * rgb2yuv[8];
			const float lut_Y = DarkLut(Y, coeff);

			lineDst[i].R = static_cast<A_u_short>(CLAMP_VALUE(lut_Y * yuv2rgb[0] + U * yuv2rgb[1] + V * yuv2rgb[2], static_cast<float>(u16_value_black), static_cast<float>(u16_value_white)));
			lineDst[i].G = static_cast<A_u_short>(CLAMP_VALUE(lut_Y * yuv2rgb[3] + U * yuv2rgb[4] + V * yuv2rgb[5], static_cast<float>(u16_value_black), static_cast<float>(u16_value_white)));
			lineDst[i].B = static_cast<A_u_short>(CLAMP_VALUE(lut_Y * yuv2rgb[6] + U * yuv2rgb[7] + V * yuv2rgb[8], static_cast<float>(u16_value_black), static_cast<float>(u16_value_white)));
			lineDst[i].A = lineSrc[i].A;
		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}
