#include "ImageStylization.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	eSTYLIZATION const lwbType{ static_cast<eSTYLIZATION>(params[IMAGE_STYLE_POPUP]->u.pd.value - 1) };

	switch (lwbType)
	{
		case eSTYLE_NEWS_PAPER_OLD:
			err = PR_ImageStyle_NewsPaper (in_data, out_data, params, output);
		break;

		case eSTYLE_NEWS_PAPER_COLOR:
			err = PR_ImageStyle_ColorNewsPaper (in_data, out_data, params, output);
		break;

		case eSTYLE_GLASSY_EFFECT:
			err = PR_ImageStyle_GlassyEffect (in_data, out_data, params, output);
		break;

		case eSTYLE_OIL_PAINT:
			err = PR_ImageStyle_OilPaint (in_data, out_data, params, output);
		break;

		case eSTYLE_CARTOON:
			err = PR_ImageStyle_CartoonEffect (in_data, out_data, params, output);
		break;
		
		case eSTYLE_SKETCH_PENCIL:
			err = PR_ImageStyle_SketchPencil (in_data, out_data, params, output);
		break;

		case eSTYLE_SKETCH_CHARCOAL:
			err = PR_ImageStyle_SketchCharcoal (in_data, out_data, params, output);
		break;
		
		case eSTYLE_IMPRESSIONISM:
			err = PR_ImageStyle_ImpressionismArt (in_data, out_data, params, output);
		break;
		
		case eSTYLE_POINTILLISM:
			err = PR_ImageStyle_PointillismArt (in_data, out_data, params, output);
		break;

		case eSTYLE_MOSAIC:
			err = PR_ImageStyle_MosaicArt (in_data, out_data, params, output);
		break;

		case eSTYLE_NONE:
		default:
			err = PF_COPY(&params[IMAGE_STYLE_INPUT]->u.ld, output, NULL, NULL);
		break;

	}
	return err;
}
