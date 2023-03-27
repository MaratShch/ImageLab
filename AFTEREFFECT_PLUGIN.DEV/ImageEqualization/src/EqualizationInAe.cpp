#include "ImageEqualization.hpp"


PF_Err
ImageEqualizationInAE_8bits
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept
{
	PF_Err err{ PF_Err_NONE };
	const ImageEqPopupAlgo eqType{ static_cast<const ImageEqPopupAlgo>(params[IMAGE_EQUALIZATION_POPUP_PRESET]->u.pd.value - 1) };

	switch (eqType)
	{
		case IMAGE_EQ_MANUAL:
		case IMAGE_EQ_LINEAR:
		case IMAGE_EQ_DETAILS_DARK:
		case IMAGE_EQ_DETAILS_LIGHT:
		case IMAGE_EQ_EXPONENTIAL:
		break;

		case IMAGE_EQ_NONE:
		default:
			PF_EffectWorld* input = reinterpret_cast<PF_EffectWorld*>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
			auto const& worldTransformSuite = AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data);
			err = worldTransformSuite->copy (in_data->effect_ref, input, output, NULL, NULL);
		break;
	}

	return err;
}

PF_Err
ImageEqualizationInAE_16bits
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept
{
	PF_Err err{ PF_Err_NONE };
	const ImageEqPopupAlgo eqType{ static_cast<const ImageEqPopupAlgo>(params[IMAGE_EQUALIZATION_POPUP_PRESET]->u.pd.value - 1) };

	switch (eqType)
	{
		case IMAGE_EQ_MANUAL:
		case IMAGE_EQ_LINEAR:
		case IMAGE_EQ_DETAILS_DARK:
		case IMAGE_EQ_DETAILS_LIGHT:
		case IMAGE_EQ_EXPONENTIAL:
		break;

		case IMAGE_EQ_NONE:
		default:
			PF_EffectWorld* input = reinterpret_cast<PF_EffectWorld*>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
			auto const& worldTransformSuite = AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data);
			err = worldTransformSuite->copy_hq (in_data->effect_ref, input, output, NULL, NULL);
		break;
	}

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
		ImageEqualizationInAE_16bits(in_data, out_data, params, output) :
		ImageEqualizationInAE_8bits (in_data, out_data, params, output));
}