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
		params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->ui_flags |= PF_PUI_DISABLED;
		paramUtilsSite3->PF_UpdateParamUI (in_data->effect_ref, eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER, params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]);
	}
	if (false == IsDisabledUI (params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags))
	{
		params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags |= PF_PUI_DISABLED;
		paramUtilsSite3->PF_UpdateParamUI (in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_DISPERSION, params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]);
	}
	if (false == IsDisabledUI (params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags))
	{
		params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags |= PF_PUI_DISABLED;
		paramUtilsSite3->PF_UpdateParamUI (in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP, params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]);
	}
	if (false == IsDisabledUI (params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags))
	{
		params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags |= PF_PUI_DISABLED;
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL, params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]);
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
		params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->ui_flags &= ~PF_PUI_DISABLED;
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER, params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]);
	}
	if (false == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags))
	{
		params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags |= PF_PUI_DISABLED;
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_DISPERSION, params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]);
	}
	if (false == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags))
	{
		params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags |= PF_PUI_DISABLED;
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP, params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]);
	}
	if (false == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags))
	{
		params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags |= PF_PUI_DISABLED;
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL, params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]);
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
		params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]->ui_flags |= PF_PUI_DISABLED;
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER, params[eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER]);
	}
	if (true == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags))
	{
		params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]->ui_flags &= ~PF_PUI_DISABLED;
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_DISPERSION, params[eNOISE_CLEAN_ANYSOTROPIC_DISPERSION]);
	}
	if (true == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags))
	{
		params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]->ui_flags &= ~PF_PUI_DISABLED;
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP, params[eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP]);
	}
	if (true == IsDisabledUI(params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags))
	{
		params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]->ui_flags &= ~PF_PUI_DISABLED;
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL, params[eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL]);
	}

	return;
}

