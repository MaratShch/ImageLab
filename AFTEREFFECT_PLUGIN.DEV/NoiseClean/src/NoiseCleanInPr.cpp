#include "NoiseClean.hpp"
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
	const eNOISE_CLEAN_TYPE algoType = static_cast<eNOISE_CLEAN_TYPE const>(params[eNOISE_CLEAN_ALGO_POPUP]->u.pd.value - 1);

	switch (algoType)
	{
		case eNOISE_CLEAN_BILATERAL_LUMA:
			err = NoiseClean_AlgoBilateral (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_BILATERAL_RGB:
			err = NoiseClean_AlgoBilateralRGB (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_PERONA_MALIK:
			err = NoiseClean_AlgoPeronaMalik (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_BSDE:
			err = NoiseClean_AlgoBSDE (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_ADVANCED_DENOISE:
			err = NoiseClean_AlgoAdvanced (in_data, out_data, params, output);
		break;
		
		case eNOISE_CLEAN_NONE:
		default:
			err = PF_COPY(&params[eNOISE_CLEAN_INPUT]->u.ld, output, NULL, NULL);
		break;
	}
	return err;
}
