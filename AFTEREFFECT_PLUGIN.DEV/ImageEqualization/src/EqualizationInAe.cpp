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
			err = AE_ImageEq_Manual_ARGB_4444_8u(in_data, out_data, params, output);
		break;

		case IMAGE_EQ_LINEAR:
			err = AE_ImageEq_Linear_ARGB_4444_8u (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_BRIGHT:
			err = AE_ImageEq_Bright_ARGB_4444_8u (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_DARK:
			err = AE_ImageEq_Dark_ARGB_4444_8u (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_EXPONENTIAL:
			err = AE_ImageEq_Exponential_ARGB_4444_8u(in_data, out_data, params, output);
		break;

		case IMAGE_EQ_SIGMOID:
			err = AE_ImageEq_Sigmoid_ARGB_4444_8u (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_ADVANCED:
			err = AE_ImageEq_Advanced_ARGB_4444_8u (in_data, out_data, params, output);
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
			err = AE_ImageEq_Manual_ARGB_4444_16u(in_data, out_data, params, output);
		break;

		case IMAGE_EQ_LINEAR:
			err = AE_ImageEq_Linear_ARGB_4444_16u (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_BRIGHT:
			err = AE_ImageEq_Bright_ARGB_4444_16u (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_DARK:
			err = AE_ImageEq_Dark_ARGB_4444_16u (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_EXPONENTIAL:
			err = AE_ImageEq_Exponential_ARGB_4444_16u (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_SIGMOID:
			err = AE_ImageEq_Sigmoid_ARGB_4444_16u (in_data, out_data, params, output);
		break;

		case IMAGE_EQ_ADVANCED:
			err = AE_ImageEq_Advanced_ARGB_4444_16u (in_data, out_data, params, output);
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
ImageEqualizationInAE_32bits
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
//            err = AE_ImageEq_Manual_ARGB_4444_16u(in_data, out_data, params, output);
        break;

        case IMAGE_EQ_LINEAR:
//            err = AE_ImageEq_Linear_ARGB_4444_16u(in_data, out_data, params, output);
        break;

        case IMAGE_EQ_BRIGHT:
//            err = AE_ImageEq_Bright_ARGB_4444_16u(in_data, out_data, params, output);
        break;

        case IMAGE_EQ_DARK:
//            err = AE_ImageEq_Dark_ARGB_4444_16u(in_data, out_data, params, output);
        break;

        case IMAGE_EQ_EXPONENTIAL:
//            err = AE_ImageEq_Exponential_ARGB_4444_16u(in_data, out_data, params, output);
        break;

        case IMAGE_EQ_SIGMOID:
//            err = AE_ImageEq_Sigmoid_ARGB_4444_16u(in_data, out_data, params, output);
        break;

        case IMAGE_EQ_ADVANCED:
//            err = AE_ImageEq_Advanced_ARGB_4444_16u(in_data, out_data, params, output);
        break;

        case IMAGE_EQ_NONE:
        default:
            PF_EffectWorld* input = reinterpret_cast<PF_EffectWorld*>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld);
            auto const& worldTransformSuite = AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data);
            err = worldTransformSuite->copy_hq(in_data->effect_ref, input, output, NULL, NULL);
        break;
    }

    return err;
}


inline PF_Err ImageEqualizationInAE_DeepWord
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    PF_Err	err = PF_Err_NONE;
    PF_PixelFormat format = PF_PixelFormat_INVALID;
    AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld), &format))
    {
        err = (format == PF_PixelFormat_ARGB128 ?
            ImageEqualizationInAE_32bits(in_data, out_data, params, output) : ImageEqualizationInAE_16bits(in_data, out_data, params, output));
    }
    else
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;

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
		ImageEqualizationInAE_DeepWord (in_data, out_data, params, output) :
		ImageEqualizationInAE_8bits (in_data, out_data, params, output));
}