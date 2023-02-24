#include "ImageEqualization.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
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
			err = PF_COPY(&params[IMAGE_EQUALIZATION_FILTER_INPUT]->u.ld, output, NULL, NULL);
		break;
	}

	return err;
}
