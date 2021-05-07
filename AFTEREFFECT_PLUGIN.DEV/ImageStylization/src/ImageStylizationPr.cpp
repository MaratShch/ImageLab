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
	eSTYLIZATION const& lwbType = static_cast<eSTYLIZATION>(params[IMAGE_STYLE_POPUP]->u.pd.value - 1);

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
		break;

		case eSTYLE_CARTOON:
		break;
		
		case eSTYLE_SKETCH_PENCIL:
		break;

		case eSTYLE_SKETCH_CHARCOAL:
		break;
		
		case eSTYLE_IMPRESSIONISM:
		break;
		
		case eSTYLE_NONE:
		default:
			err = PF_COPY(&params[IMAGE_STYLE_INPUT]->u.ld, output, NULL, NULL);
		break;

	}
	return err;
}
