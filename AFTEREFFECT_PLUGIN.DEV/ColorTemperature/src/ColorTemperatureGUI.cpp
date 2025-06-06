#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureGUI.hpp"
#include "AEFX_SuiteHelper.h"
#include <cmath>

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

#if 0
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
    DRAWBOT_SupplierRef	supplier_ref = nullptr;
    DRAWBOT_SurfaceRef	surface_ref = nullptr;

	PF_Err err = PF_Err_NONE;

    // acquire DrawBot Suite
    auto const drawbotSuite = AEFX_SuiteScoper<DRAWBOT_DrawbotSuite1>(in_data, kDRAWBOT_DrawSuite, kDRAWBOT_DrawSuite_VersionCurrent, out_data);

	// get the drawing reference
    auto const effectCustomUISuiteP = AEFX_SuiteScoper<PF_EffectCustomUISuite1>(in_data, kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1, out_data);
    if (PF_Err_NONE != err)
    {
        // Get the drawing reference by passing context to this new api
        ERR((effectCustomUISuiteP->PF_GetDrawingReference)(event_extra->contextH, &drawing_ref));
//        AEFX_ReleaseSuite(in_data, out_data, kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1, NULL);
    } // if (PF_Err_NONE != err)

	// acquire DrawBot Supplier and Surface; it shouldn't be released like pen or brush
	ERR(drawbotSuite->GetSupplier(drawing_ref, &supplier_ref));
	ERR(drawbotSuite->GetSurface (drawing_ref, &surface_ref ));

//			if (PF_EA_CONTROL == event_extra->effect_win.area)
//			{
//				DRAWBOT_PathRef	path_ref = nullptr;
//				auto const drawbotSupplier = AEFX_SuiteScoper<DRAWBOT_SupplierSuite1>(in_data, kDRAWBOT_SupplierSuite, kDRAWBOT_SupplierSuite_VersionCurrent, out_data);
//				drawbotSupplier->NewPath (supplier_ref, &path_ref);
//				if (nullptr != path_ref)
//				{
//
//					/* release Parth Object */
//					drawbotSupplier->ReleaseObject (reinterpret_cast<DRAWBOT_ObjectRef>(path_ref));
//					path_ref = nullptr;
//				} /* if (nullptr != path_ref) */
//			} /* if (PF_EA_CONTROL == event_extra->effect_win.area) */

	if (PF_Err_NONE == err)
		event_extra->evt_out_flags = PF_EO_HANDLED_EVENT;

	return err;
}
#else

PF_Err AEFX_AcquireSuite(PF_InData		*in_data,			/* >> */
    PF_OutData		*out_data,			/* >> */
    const char		*name,				/* >> */
    int32_t			version,			/* >> */
    const char		*error_stringPC0,	/* >> */
    void			**suite)			/* << */
{
    PF_Err			err = PF_Err_NONE;
    SPBasicSuite	*bsuite;

    bsuite = in_data->pica_basicP;

    if (bsuite) {
        (*bsuite->AcquireSuite)((char*)name, version, (const void**)suite);

        if (!*suite) {
            err = PF_Err_BAD_CALLBACK_PARAM;
        }
    }
    else {
        err = PF_Err_BAD_CALLBACK_PARAM;
    }

    if (err) {
        const char	*error_stringPC = error_stringPC0 ? error_stringPC0 : "Not able to acquire AEFX Suite.";

        out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;

        PF_SPRINTF(out_data->return_msg, error_stringPC);
    }

    return err;
}



PF_Err AEFX_ReleaseSuite(PF_InData		*in_data,			/* >> */
    PF_OutData		*out_data,			/* >> */
    const char		*name,				/* >> */
    int32_t			version,			/* >> */
    const char		*error_stringPC0)	/* >> */
{
    PF_Err			err = PF_Err_NONE;
    SPBasicSuite	*bsuite;

    bsuite = in_data->pica_basicP;

    if (bsuite) {
        (*bsuite->ReleaseSuite)((char*)name, version);
    }
    else {
        err = PF_Err_BAD_CALLBACK_PARAM;
    }

    if (err) {
        const char	*error_stringPC = error_stringPC0 ? error_stringPC0 : "Not able to release AEFX Suite.";

        out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;

        PF_SPRINTF(out_data->return_msg, error_stringPC);
    }

    return err;
}


PF_Err AEFX_AcquireDrawbotSuites(PF_InData				*in_data,			/* >> */
    PF_OutData				*out_data,			/* >> */
    DRAWBOT_Suites			*suitesP)			/* << */
{
    PF_Err			err = PF_Err_NONE;

    if (suitesP == NULL) {
        out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;

        PF_SPRINTF(out_data->return_msg, "NULL suite pointer passed to AEFX_AcquireDrawbotSuites");

        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
    }

    if (!err) {
        err = AEFX_AcquireSuite(in_data, out_data, kDRAWBOT_DrawSuite, kDRAWBOT_DrawSuite_VersionCurrent, NULL, (void **)&suitesP->drawbot_suiteP);
    }
    if (!err) {
        err = AEFX_AcquireSuite(in_data, out_data, kDRAWBOT_SupplierSuite, kDRAWBOT_SupplierSuite_VersionCurrent, NULL, (void **)&suitesP->supplier_suiteP);
    }
    if (!err) {
        err = AEFX_AcquireSuite(in_data, out_data, kDRAWBOT_SurfaceSuite, kDRAWBOT_SurfaceSuite_VersionCurrent, NULL, (void **)&suitesP->surface_suiteP);
    }
    if (!err) {
        err = AEFX_AcquireSuite(in_data, out_data, kDRAWBOT_PathSuite, kDRAWBOT_PathSuite_VersionCurrent, NULL, (void **)&suitesP->path_suiteP);
    }

    return err;
}


PF_Err AEFX_ReleaseDrawbotSuites(PF_InData		*in_data,			/* >> */
    PF_OutData		*out_data)			/* >> */
{
    PF_Err			err = PF_Err_NONE;

    AEFX_ReleaseSuite(in_data, out_data, kDRAWBOT_DrawSuite, kDRAWBOT_DrawSuite_VersionCurrent, NULL);
    AEFX_ReleaseSuite(in_data, out_data, kDRAWBOT_SupplierSuite, kDRAWBOT_SupplierSuite_VersionCurrent, NULL);
    AEFX_ReleaseSuite(in_data, out_data, kDRAWBOT_SurfaceSuite, kDRAWBOT_SurfaceSuite_VersionCurrent, NULL);
    AEFX_ReleaseSuite(in_data, out_data, kDRAWBOT_PathSuite, kDRAWBOT_PathSuite_VersionCurrent, NULL);

    return err;
}




PF_Err
DrawEvent(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output,
    PF_EventExtra	*event_extra)
{
    PF_Err					err = PF_Err_NONE, err2 = PF_Err_NONE;

    DRAWBOT_DrawRef			drawing_ref = NULL;
    DRAWBOT_SurfaceRef		surface_ref = NULL;
    DRAWBOT_SupplierRef		supplier_ref = NULL;
    DRAWBOT_BrushRef		brush_ref = NULL;
    DRAWBOT_BrushRef		string_brush_ref = NULL;
    DRAWBOT_PathRef			path_ref = NULL;
    DRAWBOT_FontRef			font_ref = NULL;

    DRAWBOT_Suites			drawbotSuites;
    DRAWBOT_ColorRGBA		drawbot_color;
    DRAWBOT_RectF32			rectR;
    float					default_font_sizeF = 0.0;

    // Acquire all the Drawbot suites in one go; it should be matched with release routine.
    // You can also use C++ style AEFX_DrawbotSuitesScoper which doesn't need release routine.
    ERR(AEFX_AcquireDrawbotSuites(in_data, out_data, &drawbotSuites));

    PF_EffectCustomUISuite1	*effectCustomUISuiteP;

    ERR(AEFX_AcquireSuite(in_data,
        out_data,
        kPFEffectCustomUISuite,
        kPFEffectCustomUISuiteVersion1,
        NULL,
        (void**)&effectCustomUISuiteP));

    if (!err && effectCustomUISuiteP) {
        // Get the drawing reference by passing context to this new api
        ERR((*effectCustomUISuiteP->PF_GetDrawingReference)(event_extra->contextH, &drawing_ref));

        AEFX_ReleaseSuite(in_data, out_data, kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1, NULL);
    }

    // Get the Drawbot supplier from drawing reference; it shouldn't be released like pen or brush (see below)
    ERR(drawbotSuites.drawbot_suiteP->GetSupplier(drawing_ref, &supplier_ref));

    // Get the Drawbot surface from drawing reference; it shouldn't be released like pen or brush (see below)
    ERR(drawbotSuites.drawbot_suiteP->GetSurface(drawing_ref, &surface_ref));

    // Premiere Pro/Elements does not support a standard parameter type
    // with custom UI (bug #1235407), so we can't use the color values.
    // Use an static grey value instead.
    if (in_data->appl_id != 'PrMr')
    {
        drawbot_color.red = static_cast<float>(params[COLOR_TEMPERATURE_COLOR_BAR_GUI]->u.cd.value.red) / PF_MAX_CHAN8;
        drawbot_color.green = static_cast<float>(params[COLOR_TEMPERATURE_COLOR_BAR_GUI]->u.cd.value.green) / PF_MAX_CHAN8;
        drawbot_color.blue = static_cast<float>(params[COLOR_TEMPERATURE_COLOR_BAR_GUI]->u.cd.value.blue) / PF_MAX_CHAN8;
    }
    else
    {
        static float gray = 0;
        drawbot_color.red = fmod(gray, 1);
        drawbot_color.green = fmod(gray, 1);
        drawbot_color.blue = fmod(gray, 1);
        gray += 0.01f;
    }
    drawbot_color.alpha = 1.0;

    if (PF_EA_CONTROL == event_extra->effect_win.area) {

        // Create a new path. It should be matched with release routine.
        // You can also use C++ style DRAWBOT_PathP that releases automatically at the end of scope.
        ERR(drawbotSuites.supplier_suiteP->NewPath(supplier_ref, &path_ref));

        // Create a new brush taking color as input; it should be matched with release routine.
        // You can also use C++ style DRAWBOT_BrushP which doesn't require release routine.
        ERR(drawbotSuites.supplier_suiteP->NewBrush(supplier_ref, &drawbot_color, &brush_ref));

        rectR.left = event_extra->effect_win.current_frame.left + 0.5;	// Center of the pixel in new drawing model is (0.5, 0.5)
        rectR.top = event_extra->effect_win.current_frame.top + 0.5;
        rectR.width = static_cast<float>(event_extra->effect_win.current_frame.right -
            event_extra->effect_win.current_frame.left);
        rectR.height = static_cast<float>(event_extra->effect_win.current_frame.bottom -
            event_extra->effect_win.current_frame.top);

        // Add the rectangle to path
        ERR(drawbotSuites.path_suiteP->AddRect(path_ref, &rectR));

        // Fill the path with the brush created
        ERR(drawbotSuites.surface_suiteP->FillPath(surface_ref, brush_ref, path_ref, kDRAWBOT_FillType_Default));

        // Get the default font size.
        ERR(drawbotSuites.supplier_suiteP->GetDefaultFontSize(supplier_ref, &default_font_sizeF));

        // Create default font with default size.  Note that you can provide a different font size.
        ERR(drawbotSuites.supplier_suiteP->NewDefaultFont(supplier_ref, default_font_sizeF, &font_ref));

        DRAWBOT_UTF16Char	unicode_string[256];
        auto constexpr CUSTOM_UI_STRING = L"A Custom UI!!!\n";

//        copyConvertStringLiteralIntoUTF16(CUSTOM_UI_STRING, unicode_string);

        // Draw string with white color
        drawbot_color.red = drawbot_color.green = drawbot_color.blue = drawbot_color.alpha = 1.0;

        ERR(drawbotSuites.supplier_suiteP->NewBrush(supplier_ref, &drawbot_color, &string_brush_ref));

//        DRAWBOT_PointF32			text_origin;
//
//        text_origin.x = event_extra->effect_win.current_frame.left + 5.0;
//        text_origin.y = event_extra->effect_win.current_frame.top + 50.0;
//
//        ERR(drawbotSuites.surface_suiteP->DrawString(surface_ref,
//            string_brush_ref,
//            font_ref,
//            &unicode_string[0],
//            &text_origin,
//            kDRAWBOT_TextAlignment_Default,
//            kDRAWBOT_TextTruncation_None,
//            0.0f));

        if (string_brush_ref) {
            ERR2(drawbotSuites.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(string_brush_ref)));
        }

        if (font_ref) {
            ERR2(drawbotSuites.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(font_ref)));
        }

        // Release/destroy the brush. Otherwise, it will lead to memory leak.
        if (brush_ref) {
            ERR2(drawbotSuites.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(brush_ref)));
        }

        // Release/destroy the path. Otherwise, it will lead to memory leak.
        if (path_ref) {
            ERR2(drawbotSuites.supplier_suiteP->ReleaseObject(reinterpret_cast<DRAWBOT_ObjectRef>(path_ref)));
        }
    }

    // Release the earlier acquired Drawbot suites
    ERR2(AEFX_ReleaseDrawbotSuites(in_data, out_data));

    if (!err) {
        event_extra->evt_out_flags = PF_EO_HANDLED_EVENT;
    }

    return err;
}

#endif
