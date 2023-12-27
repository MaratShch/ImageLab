#include "NoiseClean.hpp"


inline PF_Err NoiseCleanInAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	eNOISE_CLEAN_TYPE const algoType = static_cast<eNOISE_CLEAN_TYPE const>(params[eNOISE_CLEAN_ALGO_POPUP]->u.pd.value - 1);

	switch (algoType)
	{
		case eNOISE_CLEAN_BILATERAL_LUMA:
			err = NoiseClean_AlgoBilateralAe8 (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_BILATERAL_RGB:
			err = NoiseClean_AlgoBilateralRGBAe8 (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_PERONA_MALIK:
			err = NoiseClean_AlgoPeronaMalikAe8 (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_BSDE:
			err = NoiseClean_AlgoBSDEAe8 (in_data, out_data, params, output);
		break;
	
		case eNOISE_CLEAN_NONLOCAL_BAYES:
			err = NoiseClean_AlgoNonLocalBayesAe8 (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_ADVANCED_DENOISE:
//			err = NoiseClean_AlgoAdvancedAe8 (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_NONE:
			err = AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data)->
				copy(in_data->effect_ref, &params[eNOISE_CLEAN_INPUT]->u.ld, output, NULL, NULL);
		default:
		break;
	}

	return err;
}


inline PF_Err NoiseCleanInAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	eNOISE_CLEAN_TYPE const algoType = static_cast<eNOISE_CLEAN_TYPE const>(params[eNOISE_CLEAN_ALGO_POPUP]->u.pd.value - 1);

	switch (algoType)
	{
		case eNOISE_CLEAN_BILATERAL_LUMA:
			err = NoiseClean_AlgoBilateralAe16 (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_BILATERAL_RGB:
			err = NoiseClean_AlgoBilateralRGBAe16 (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_PERONA_MALIK:
			err = NoiseClean_AlgoPeronaMalikAe16 (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_BSDE:
		break;

		case eNOISE_CLEAN_NONLOCAL_BAYES:
			err = NoiseClean_AlgoNonLocalBayesAe16 (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_ADVANCED_DENOISE:
//			err = NoiseClean_AlgoAdvancedAe16 (in_data, out_data, params, output);
		break;

		case eNOISE_CLEAN_NONE:
		default:
			err = AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data)->
				copy_hq(in_data->effect_ref, &params[eNOISE_CLEAN_INPUT]->u.ld, output, NULL, NULL);
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
		NoiseCleanInAE_16bits(in_data, out_data, params, output) :
		NoiseCleanInAE_8bits(in_data, out_data, params, output));
}