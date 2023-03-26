#include "ImageEqualization.hpp"
#include "FastAriphmetics.hpp"

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
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

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
