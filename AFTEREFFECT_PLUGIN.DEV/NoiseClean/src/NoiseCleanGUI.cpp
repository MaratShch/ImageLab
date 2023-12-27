#include "NoiseClean.hpp"


void SwitchToNoAlgo
(
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
) noexcept
{
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSite3 { AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data) };
	if (false == IsDisabledUI (params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->ui_flags))
	{
		DisableUI (params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI (in_data->effect_ref, eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER, params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]);
	}
	if (false == IsDisabledUI (params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags))
	{
		DisableUI (params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI (in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_DISPERSION, params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]);
	}
	if (false == IsDisabledUI (params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags))
	{
		DisableUI (params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI (in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP, params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]);
	}
	if (false == IsDisabledUI (params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags))
	{
		DisableUI (params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI (in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL, params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]);
	}

	if (false == IsDisabledUI(params[eNOISE_CLEAN_NL_BAYES_SIGMA]->ui_flags))
	{
		DisableUI (params[eNOISE_CLEAN_NL_BAYES_SIGMA]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI (in_data->effect_ref, eNOISE_CLEAN_NL_BAYES_SIGMA, params[eNOISE_CLEAN_NL_BAYES_SIGMA]);
	}

	return;
}

void SwitchToBilateral
(
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
) noexcept
{
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSite3{ AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data) };
	if (true == IsDisabledUI(params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->ui_flags))
	{
		EnableUI (params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER, params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]);
	}
	if (false == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags))
	{
		DisableUI (params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_DISPERSION, params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]);
	}
	if (false == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags))
	{
		DisableUI (params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP, params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]);
	}
	if (false == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags))
	{
		DisableUI (params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL, params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]);
	}

	if (false == IsDisabledUI(params[eNOISE_CLEAN_NL_BAYES_SIGMA]->ui_flags))
	{
		DisableUI(params[eNOISE_CLEAN_NL_BAYES_SIGMA]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_NL_BAYES_SIGMA, params[eNOISE_CLEAN_NL_BAYES_SIGMA]);
	}

	return;
}

void SwitchToAnysotropic
(
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
) noexcept
{
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSite3{ AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data) };
	if (false == IsDisabledUI(params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->ui_flags))
	{
		DisableUI (params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER, params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]);
	}
	if (true == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags))
	{
		EnableUI (params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_DISPERSION, params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]);
	}
	if (true == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags))
	{
		EnableUI (params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP, params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]);
	}
	if (true == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags))
	{
		EnableUI (params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL, params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]);
	}

	if (false == IsDisabledUI(params[eNOISE_CLEAN_NL_BAYES_SIGMA]->ui_flags))
	{
		DisableUI(params[eNOISE_CLEAN_NL_BAYES_SIGMA]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_NL_BAYES_SIGMA, params[eNOISE_CLEAN_NL_BAYES_SIGMA]);
	}

	return;
}


void SwitchToBSDE
(
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
) noexcept
{
	return; /* do nothning currently */
}

void SwitchToNonLocalBayes
(
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
) noexcept
{
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSite3{ AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data) };
	if (false == IsDisabledUI(params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->ui_flags))
	{
		DisableUI(params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER, params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]);
	}
	if (false == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags))
	{
		DisableUI(params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_DISPERSION, params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]);
	}
	if (false == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags))
	{
		DisableUI(params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP, params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]);
	}
	if (false == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags))
	{
		DisableUI(params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL, params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]);
	}

	if (true == IsDisabledUI(params[eNOISE_CLEAN_NL_BAYES_SIGMA]->ui_flags))
	{
		EnableUI (params[eNOISE_CLEAN_NL_BAYES_SIGMA]->ui_flags);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_NL_BAYES_SIGMA, params[eNOISE_CLEAN_NL_BAYES_SIGMA]);
	}

	return;
}

void SwitchToAdvanced
(
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
) noexcept
{
	return; /* do nothning currently */
}