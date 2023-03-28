#include "ImageEqualization.hpp"
#include "FastAriphmetics.hpp"
#include "ColorTransformMatrix.hpp"


constexpr float sigmoidMax = 6.f;
constexpr float sigmoidMin = -sigmoidMax;
constexpr float sigmoidRange = sigmoidMax - sigmoidMin;

template <typename T>
inline const typename std::enable_if<std::is_floating_point<T>::value, T>::type sigmoid(const T& fVal) noexcept
{
	constexpr T one{ 1 };
	return one / (one + FastCompute::Exp(-fVal));
}


PF_Err PR_ImageEq_Sigmoid_VUYA_4444_8u_709
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

	constexpr float sigmoidStep = sigmoidRange / 256.f;

	for (j = 0; j < height; j++)
	{
		const PF_Pixel_VUYA_8u* __restrict lineSrc = localSrc + j * line_pitch;
		      PF_Pixel_VUYA_8u* __restrict lineDst = localDst + j * line_pitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			lineDst[i].V = lineSrc[i].V;
			lineDst[i].U = lineSrc[i].U;
			lineDst[i].Y = static_cast<uint8_t>(sigmoid((static_cast<float>(lineSrc[i].Y) - 128.f) * sigmoidStep) * 256.f);
			lineDst[i].A = lineSrc[i].A;
		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Sigmoid_VUYA_4444_32f_709
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	A_long i, j;
	const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
	const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	      PF_Pixel_VUYA_32f* __restrict localDst  = reinterpret_cast<     PF_Pixel_VUYA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

	for (j = 0; j < height; j++)
	{
		const PF_Pixel_VUYA_32f* __restrict lineSrc = localSrc + j * line_pitch;
		      PF_Pixel_VUYA_32f* __restrict lineDst = localDst + j * line_pitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			lineDst[i].V = lineSrc[i].V;
			lineDst[i].U = lineSrc[i].U;
			lineDst[i].Y = sigmoid(sigmoidRange * lineSrc[i].Y - sigmoidMax);
			lineDst[i].A = lineSrc[i].A;
		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}


PF_Err PR_ImageEq_Sigmoid_BGRA_4444_8u
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
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	constexpr float sigmoidStep = sigmoidRange / 256.f;

	for (j = 0; j < height; j++)
	{
		const PF_Pixel_BGRA_8u* __restrict lineSrc = localSrc + j * line_pitch;
		      PF_Pixel_BGRA_8u* __restrict lineDst = localDst + j * line_pitch;

		__VECTOR_ALIGNED__
		for (i = 0; i < width; i++)
		{
			const float Y = lineSrc[i].R * rgb2yuv[0] + lineSrc[i].G * rgb2yuv[1] + lineSrc[i].B * rgb2yuv[2];
			const float U = lineSrc[i].R * rgb2yuv[3] + lineSrc[i].G * rgb2yuv[4] + lineSrc[i].B * rgb2yuv[5];
			const float V = lineSrc[i].R * rgb2yuv[6] + lineSrc[i].G * rgb2yuv[7] + lineSrc[i].B * rgb2yuv[8];
			const float Y_modified = static_cast<uint8_t>(sigmoid((static_cast<float>(Y) - 128.f) * sigmoidStep) * 256.f);

			lineDst[i].R = static_cast<A_u_char>(CLAMP_VALUE(Y_modified * yuv2rgb[0] + U * yuv2rgb[1] + V * yuv2rgb[2], 0.f, 255.f));
			lineDst[i].G = static_cast<A_u_char>(CLAMP_VALUE(Y_modified * yuv2rgb[3] + U * yuv2rgb[4] + V * yuv2rgb[5], 0.f, 255.f));
			lineDst[i].B = static_cast<A_u_char>(CLAMP_VALUE(Y_modified * yuv2rgb[6] + U * yuv2rgb[7] + V * yuv2rgb[8], 0.f, 255.f));
			lineDst[i].A = lineSrc[i].A;
		} /* for (i = 0; i < width; i++) */

	} /* for (j = 0; j < height; j++) */

	return PF_Err_NONE;
}