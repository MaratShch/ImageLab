#include "Morphology.hpp"
#include "MorphologyProcCpu.hpp"


PF_Err MorphologyFilterInAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	size_t sizeSe{ 0 };
	const SE_Interface* seInetrface = getStructuredElemInterface(out_data);
	const SE_Type* __restrict seElementVal = (nullptr != seInetrface ? seInetrface->GetStructuredElement(sizeSe) : nullptr);

	if (nullptr == seElementVal || 0 == sizeSe)
	{
		AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data)->
			copy(in_data->effect_ref, &params[MORPHOLOGY_FILTER_INPUT]->u.ld, output, NULL, NULL);
		return PF_Err_NONE;
	}

	const PF_EffectWorld*   __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[MORPHOLOGY_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
	PF_Pixel_ARGB_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);

	auto const& src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	auto const& dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	auto const& height = output->height;
	auto const& width  = output->width;

	SeOperation const& opType = static_cast<const SeOperation>(params[MORPHOLOGY_OPERATION_TYPE]->u.pd.value - 1);
	SeType const& elType      = static_cast<const SeType>(params[MORPHOLOGY_ELEMENT_TYPE  ]->u.sd.value - 1);
	SeSize const& krSize      = static_cast<const SeSize>(params[MORPHOLOGY_KERNEL_SIZE   ]->u.fs_d.value - 1);

	switch (opType)
	{
		case SE_OP_EROSION:
			Morphology_Erode (localSrc, localDst, seElementVal, sizeSe, height, width, src_pitch, dst_pitch, static_cast<A_u_char>(UCHAR_MAX));
		break;

		case SE_OP_DILATION:
			Morphology_Dilate (localSrc, localDst, seElementVal, sizeSe, height, width, src_pitch, dst_pitch, static_cast<A_u_char>(UCHAR_MAX));
		break;

		case SE_OP_OPEN:
		break;

		case SE_OP_CLOSE:
		break;
		
		case SE_OP_THIN:
		break;

		case SE_OP_THICK:
		break;
		
		case SE_OP_GRADIENT:
		break;

		default:
			AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data)->
				copy(in_data->effect_ref, &params[MORPHOLOGY_FILTER_INPUT]->u.ld, output, NULL, NULL);
		break;
	}

	return PF_Err_NONE;
}


PF_Err MorphologyFilterInAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	size_t sizeSe{ 0 };
	const SE_Interface* seInetrface = getStructuredElemInterface(out_data);
	const SE_Type* __restrict seElementVal = (nullptr != seInetrface ? seInetrface->GetStructuredElement(sizeSe) : nullptr);

	if (nullptr == seElementVal || 0 == sizeSe)
	{
		AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data)->
			copy_hq(in_data->effect_ref, &params[MORPHOLOGY_FILTER_INPUT]->u.ld, output, NULL, NULL);
		return PF_Err_NONE;
	}

	const PF_EffectWorld*    __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[MORPHOLOGY_FILTER_INPUT]->u.ld);
	const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
	PF_Pixel_ARGB_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output->data);

	auto const& src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	auto const& dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

	auto const& height = output->height;
	auto const& width  = output->width;

	SeOperation const& opType = static_cast<const SeOperation>(params[MORPHOLOGY_OPERATION_TYPE]->u.pd.value - 1);
	SeType const& elType      = static_cast<const SeType>(params[MORPHOLOGY_ELEMENT_TYPE]->u.sd.value - 1);
	SeSize const& krSize      = static_cast<const SeSize>(params[MORPHOLOGY_KERNEL_SIZE]->u.fs_d.value - 1);

	switch (opType)
	{
		case SE_OP_EROSION:
			Morphology_Erode (localSrc, localDst, seElementVal, sizeSe, height, width, src_pitch, dst_pitch, static_cast<A_u_short>(SHRT_MAX));
		break;

		case SE_OP_DILATION:
			Morphology_Dilate (localSrc, localDst, seElementVal, sizeSe, height, width, src_pitch, dst_pitch, static_cast<A_u_short>(SHRT_MAX));
		break;

		case SE_OP_OPEN:
		break;

		case SE_OP_CLOSE:
		break;

		case SE_OP_THIN:
		break;

		case SE_OP_THICK:
		break;

		case SE_OP_GRADIENT:
		break;

		default:
			AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data)->
				copy_hq(in_data->effect_ref, &params[MORPHOLOGY_FILTER_INPUT]->u.ld, output, NULL, NULL);
		break;
	}

	return PF_Err_NONE;
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
		MorphologyFilterInAE_16bits(in_data, out_data, params, output) :
		MorphologyFilterInAE_8bits (in_data, out_data, params, output));
}