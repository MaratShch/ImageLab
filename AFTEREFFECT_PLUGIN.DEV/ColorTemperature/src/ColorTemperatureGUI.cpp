#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureGUI.hpp"
#include "AEFX_SuiteHelper.h"


PF_Err PresetsActivation
(
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
) 
{
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSite3 =
		AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);

	bool updateUI = false;

	if (true == IsDisabledUI(params[COLOR_TEMPERATURE_PRESET_TYPE_POPUP]->ui_flags))
	{
		EnableUI(params[COLOR_TEMPERATURE_PRESET_TYPE_POPUP]->ui_flags);
		updateUI = true;
	}

	if (false == IsDisabledUI(params[COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP]->ui_flags))
	{
		DisableUI(params[COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP]->ui_flags);
		updateUI = true;
	}

	if (false == IsDisabledUI(params[COLOR_TEMPERATURE_LOAD_PRESET_BUTTON]->ui_flags))
	{
		DisableUI(params[COLOR_TEMPERATURE_LOAD_PRESET_BUTTON]->ui_flags);
		updateUI = true;
	}

	if (false == IsDisabledUI(params[COLOR_TEMPERATURE_SAVE_PRESET_BUTTON]->ui_flags))
	{
		DisableUI(params[COLOR_TEMPERATURE_SAVE_PRESET_BUTTON]->ui_flags);
		updateUI = true;
	}

	/* update UI */
	if (true == updateUI)
	{
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_TEMPERATURE_PRESET_TYPE_POPUP,     params[COLOR_TEMPERATURE_PRESET_TYPE_POPUP]);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP,   params[COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP]);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_TEMPERATURE_LOAD_PRESET_BUTTON,    params[COLOR_TEMPERATURE_LOAD_PRESET_BUTTON]);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_TEMPERATURE_SAVE_PRESET_BUTTON,    params[COLOR_TEMPERATURE_SAVE_PRESET_BUTTON]);
	}

	return PF_Err_NONE;
}


PF_Err PresetsDeactivation
(
	PF_InData	*in_data,
	PF_OutData	*out_data,
	PF_ParamDef	*params[]
)
{
	AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtilsSite3 =
		AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);

	bool updateUI = false;

	if (false == IsDisabledUI(params[COLOR_TEMPERATURE_PRESET_TYPE_POPUP]->ui_flags))
	{
		DisableUI(params[COLOR_TEMPERATURE_PRESET_TYPE_POPUP]->ui_flags);
		updateUI = true;
	}

	if (true == IsDisabledUI(params[COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP]->ui_flags))
	{
		EnableUI(params[COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP]->ui_flags);
		updateUI = true;
	}

	if (true == IsDisabledUI(params[COLOR_TEMPERATURE_LOAD_PRESET_BUTTON]->ui_flags))
	{
		EnableUI(params[COLOR_TEMPERATURE_LOAD_PRESET_BUTTON]->ui_flags);
		updateUI = true;
	}

	if (true == IsDisabledUI(params[COLOR_TEMPERATURE_SAVE_PRESET_BUTTON]->ui_flags))
	{
		EnableUI(params[COLOR_TEMPERATURE_SAVE_PRESET_BUTTON]->ui_flags);
		updateUI = true;
	}

	/* update UI */
	if (true == updateUI)
	{
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_TEMPERATURE_PRESET_TYPE_POPUP,     params[COLOR_TEMPERATURE_PRESET_TYPE_POPUP]);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP,   params[COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP]);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_TEMPERATURE_LOAD_PRESET_BUTTON,    params[COLOR_TEMPERATURE_LOAD_PRESET_BUTTON]);
		paramUtilsSite3->PF_UpdateParamUI(in_data->effect_ref, COLOR_TEMPERATURE_SAVE_PRESET_BUTTON,    params[COLOR_TEMPERATURE_SAVE_PRESET_BUTTON]);
	}

	return PF_Err_NONE;
}


PF_Err DrawEvent
(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	PF_EventExtra	*event_extra
)
{
	DRAWBOT_DrawRef drawing_ref = nullptr;
	PF_Err err = PF_Err_NONE;

	/* get the drawing reference */
	if (PF_Err_NONE == (err = AEFX_SuiteScoper<PF_EffectCustomUISuite1>(in_data, kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1, out_data)->
		                PF_GetDrawingReference (event_extra->contextH, &drawing_ref)))
	{
		DRAWBOT_SupplierRef	supplier_ref = nullptr;
		DRAWBOT_SurfaceRef	surface_ref  = nullptr;

		/* acquire DrawBot Suite	*/
		auto const drawbotSuite = AEFX_SuiteScoper<DRAWBOT_DrawbotSuite1>(in_data, kDRAWBOT_DrawSuite, kDRAWBOT_DrawSuite_VersionCurrent, out_data);

		/* acquire DrawBot Supplier and Surface; it shouldn't be released like pen or brush */
		auto const err1 = drawbotSuite->GetSupplier(drawing_ref, &supplier_ref);
		auto const err2 = drawbotSuite->GetSurface (drawing_ref, &surface_ref );
		if (kSPNoError == err1 && kSPNoError == err2 && nullptr != supplier_ref && nullptr != surface_ref)
		{
			if (PF_EA_CONTROL == event_extra->effect_win.area)
			{
				DRAWBOT_PathRef	path_ref = nullptr;
				auto const drawbotSupplier = AEFX_SuiteScoper<DRAWBOT_SupplierSuite1>(in_data, kDRAWBOT_SupplierSuite, kDRAWBOT_SupplierSuite_VersionCurrent, out_data);
				drawbotSupplier->NewPath (supplier_ref, &path_ref);
				if (nullptr != path_ref)
				{

					/* release Parth Object */
					drawbotSupplier->ReleaseObject (reinterpret_cast<DRAWBOT_ObjectRef>(path_ref));
					path_ref = nullptr;
				} /* if (nullptr != path_ref) */
			} /* if (PF_EA_CONTROL == event_extra->effect_win.area) */
		}/* if (kASNoError == err1 && kASNoError == err2 && nullptr != supplier_ref && nullptr != surface_ref) */

	} /* if (PF_Err_NONE == (err = AEFX_SuiteScoper<PF_EffectCustomUISuite1> ...  */

	if (PF_Err_NONE == err)
		event_extra->evt_out_flags = PF_EO_HANDLED_EVENT;

	return err;
}