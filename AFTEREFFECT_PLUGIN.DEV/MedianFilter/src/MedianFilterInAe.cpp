#include "MedianFilter.hpp"
#include "MedianFilterAvx2.hpp"


PF_Err MeadianFilterInAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	const PF_EffectWorld*   __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[MEDIAN_FILTER_INPUT]->u.ld);
	uint32_t* __restrict localSrc = reinterpret_cast<uint32_t* __restrict>(input->data);
	uint32_t* __restrict localDst = reinterpret_cast<uint32_t* __restrict>(output->data);

	auto const& src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const& dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	auto const& height = output->height;
	auto const& width  = output->width;
	auto const kernelSize = get_kernel_size(params);
	PF_Err err = PF_Err_NONE;
	bool medianResult = false;

	constexpr A_long channelMask = 0xFFFFFF00;

	switch (kernelSize)
	{
		case 0:
			AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data)->copy(in_data->effect_ref, &params[MEDIAN_FILTER_INPUT]->u.ld, output, NULL, NULL);
			medianResult = true;
		break;

		case 3:
		/* manually optimized variant 3x3 */
			medianResult = AVX2::Median::median_filter_3x3_RGB_4444_8u (localSrc, localDst, height, width, src_pitch, dst_pitch, channelMask);
		break;

		case 5:
		/* manually optimized variant 5x5 */
			medianResult = AVX2::Median::median_filter_5x5_RGB_4444_8u (localSrc, localDst, height, width, src_pitch, dst_pitch, channelMask);
		break;

		case 7:
		/* manually optimized variant 7x7 */
			medianResult = AVX2::Median::median_filter_7x7_RGB_4444_8u (localSrc, localDst, height, width, src_pitch, dst_pitch, channelMask);
		break;

		default:
		/* median via histogramm algo */
		break;

	}

	return (true == medianResult ? PF_Err_NONE  : PF_Err_INTERNAL_STRUCT_DAMAGED);
}


PF_Err MeadianFilterInAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	return err;
}


PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept
{
	return (PF_WORLD_IS_DEEP(output) ?
		MeadianFilterInAE_16bits(in_data, out_data, params, output) :
		MeadianFilterInAE_8bits(in_data, out_data, params, output));
}