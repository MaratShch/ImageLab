#include "ImageStylization.hpp"



PF_Err ImageStyleInAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	eSTYLIZATION const& lwbType = static_cast<eSTYLIZATION>(params[IMAGE_STYLE_POPUP]->u.pd.value - 1);

	switch (lwbType)
	{
		case eSTYLE_NEWS_PAPER_OLD:
			err = AE_ImageStyle_NewsPaper_ARGB_8u (in_data, out_data, params, output);
		break;

		case eSTYLE_NEWS_PAPER_COLOR:
			err = AE_ImageStyle_ColorNewsPaper_ARGB_8u (in_data, out_data, params, output);
		break;

		case eSTYLE_GLASSY_EFFECT:
			err = AE_ImageStyle_GlassyEffect_ARGB_8u (in_data, out_data, params, output);
		break;

		case eSTYLE_OIL_PAINT:
		break;

		case eSTYLE_CARTOON:
			err = AE_ImageStyle_CartoonEffect_ARGB_8u (in_data, out_data, params, output);
		break;

		case eSTYLE_SKETCH_PENCIL:
			err = AE_ImageStyle_SketchPencil_ARGB_8u (in_data, out_data, params, output);
		break;

		case eSTYLE_SKETCH_CHARCOAL:
			err = AE_ImageStyle_SketchCharcoal_ARGB_8u (in_data, out_data, params, output);
		break;

		case eSTYLE_IMPRESSIONISM:
		break;

		case eSTYLE_NONE:
		default:
			err = AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data)->
				copy(in_data->effect_ref, &params[IMAGE_STYLE_INPUT]->u.ld, output, NULL, NULL);
		break;

	}
	return err;
}
	


PF_Err ImageStyleInAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	eSTYLIZATION const& lwbType = static_cast<eSTYLIZATION>(params[IMAGE_STYLE_POPUP]->u.pd.value - 1);

	switch (lwbType)
	{
		case eSTYLE_NEWS_PAPER_OLD:
			err = AE_ImageStyle_NewsPaper_ARGB_16u (in_data, out_data, params, output);
		break;

		case eSTYLE_NEWS_PAPER_COLOR:
			err = AE_ImageStyle_ColorNewsPaper_ARGB_16u (in_data, out_data, params, output);
		break;

		case eSTYLE_GLASSY_EFFECT:
			err = AE_ImageStyle_GlassyEffect_ARGB_16u (in_data, out_data, params, output);
		break;

		case eSTYLE_OIL_PAINT:
		break;

		case eSTYLE_CARTOON:
			err = AE_ImageStyle_CartoonEffect_ARGB_16u (in_data, out_data, params, output);
		break;

		case eSTYLE_SKETCH_PENCIL:
			err = AE_ImageStyle_SketchPencil_ARGB_16u (in_data, out_data, params, output);
		break;

		case eSTYLE_SKETCH_CHARCOAL:
			err = AE_ImageStyle_SketchCharcoal_ARGB_16u (in_data, out_data, params, output);
		break;

		case eSTYLE_IMPRESSIONISM:
		break;

		case eSTYLE_NONE:
		default:
			err = AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data)->
				copy_hq(in_data->effect_ref, &params[IMAGE_STYLE_INPUT]->u.ld, output, NULL, NULL);
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
		ImageStyleInAE_16bits(in_data, out_data, params, output) :
		ImageStyleInAE_8bits (in_data, out_data, params, output));
}