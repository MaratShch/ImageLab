#include "ColorTemperature.hpp"
#include "ColorTemperatureGUI.hpp"
#include "ColorTemperatureControlsPresets.hpp"
#include "ImageLabMemInterface.hpp"
#include "AlgoProcStructures.hpp"
#include "PrSDKAESupport.h"
#include "AEGP_SuiteHandler.h"

#ifdef _DEBUG
volatile AlgoCCT::CctHandleF32* pGCctHandler32{ nullptr };
volatile pHandle* pGHandle{ nullptr };
#endif

// vector contains preset settings
std::vector<IPreset*> vPresets{};


static PF_Err
About(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_SPRINTF(
		out_data->return_msg,
		"%s, v%d.%d\r%s",
		strName,
		ColorTemperature_VersionMajor,
		ColorTemperature_VersionMinor,
		strCopyright);

	return PF_Err_NONE;
}


static PF_Err
GlobalSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
    LoadMemoryInterfaceProvider(in_data);

	PF_Err	err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    PF_Handle pGlobalStorage = nullptr;

	constexpr PF_OutFlags out_flags1 =
		PF_OutFlag_WIDE_TIME_INPUT                |
		PF_OutFlag_SEQUENCE_DATA_NEEDS_FLATTENING |
		PF_OutFlag_USE_OUTPUT_EXTENT              |
		PF_OutFlag_PIX_INDEPENDENT                |
		PF_OutFlag_DEEP_COLOR_AWARE               |
        PF_OutFlag_CUSTOM_UI                      | 
		PF_OutFlag_SEND_UPDATE_PARAMS_UI;

	constexpr PF_OutFlags out_flags2 =
		PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG    |
		PF_OutFlag2_DOESNT_NEED_EMPTY_PIXELS            |
		PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT           |
		PF_OutFlag2_SUPPORTS_GET_FLATTENED_SEQUENCE_DATA;

	out_data->my_version =
		PF_VERSION(
			ColorTemperature_VersionMajor,
			ColorTemperature_VersionMinor,
			ColorTemperature_VersionSub,
			ColorTemperature_VersionStage,
			ColorTemperature_VersionBuild
		);

	out_data->out_flags  = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		auto const pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

		/*	Add the pixel formats we support in order of preference. */
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);

		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
//		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
	}

    // Initialize CCT LUTs
    AlgoCCT::CctHandleF32* globalCctHandler32 = new AlgoCCT::CctHandleF32();

    if (nullptr != globalCctHandler32)
    {
        // Add CctHandler to global data
        AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
        if (nullptr != (pGlobalStorage = handleSuite->host_new_handle(sizeof(pHandle))))
        {
            pHandle* pHndl = static_cast<pHandle*>(handleSuite->host_lock_handle(pGlobalStorage));
            if (PremierId != in_data->appl_id)
            {
                AEFX_SuiteScoper<AEGP_UtilitySuite3> u_suite(in_data, kAEGPUtilitySuite, kAEGPUtilitySuiteVersion3);
                u_suite->AEGP_RegisterWithAEGP(nullptr, strName, &pHndl->id);
            }

            pHndl->hndl = globalCctHandler32;
            pHndl->valid = static_cast<A_long>(0xDEADBEEF);

            out_data->global_data = pGlobalStorage;
            handleSuite->host_unlock_handle(pGlobalStorage);

#ifdef _DEBUG
            pGHandle = pHndl;
#endif

            // Initialize PreSets
            setPresetsVector(vPresets);
            err = PF_Err_NONE;

        } // if (nullptr != (pGlobalStorage = handleSuite->host_new_handle(sizeof(globalCctHandler32))))
    }
    else
    {
        // abnormal exit - free all allocated resources before
        delete globalCctHandler32;
        globalCctHandler32 = nullptr;
        UnloadMemoryInterfaceProvider();
    }

#ifdef _DEBUG
    pGCctHandler32 = globalCctHandler32;
#endif

    return err;
}


static PF_Err
GlobalSetdown(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{

    if (nullptr != in_data->global_data)
    {
        pHandle* pGlobal = static_cast<pHandle*>(GET_OBJ_FROM_HNDL(in_data->global_data));

        if (nullptr != pGlobal && nullptr != pGlobal->hndl)
        {
            pGlobal->valid = 0x0;

            AlgoCCT::CctHandleF32* globalCctHandler32 = pGlobal->hndl;
            
            globalCctHandler32->Deinitialize();
            delete globalCctHandler32;
            globalCctHandler32 = nullptr;
            
            pGlobal = nullptr;
        } // if (nullptr != pGlobal && nullptr != pGlobal->hndl)

        AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data)->host_dispose_handle(in_data->global_data);
    }

    resetPresets (vPresets);
    
    UnloadMemoryInterfaceProvider();

    return PF_Err_NONE;
}


static PF_Err
ParamsSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_ParamDef	def;
    PF_Err err = PF_Err_NONE;

    constexpr PF_ParamFlags   flags = PF_ParamFlag_SUPERVISE;
	constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;
	constexpr PF_ParamUIFlags ui_disabled_flags = ui_flags | PF_PUI_DISABLED;

	// SetUp 'Using Preset' checkbox. Default state - non selected/
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_CHECKBOXX(
		controlItemName[0],
		FALSE,
		flags,
		COLOR_TEMPERATURE_PRESET_CHECKBOX);

	// Setup 'Preset' popup - initially disable
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
	PF_ADD_POPUP(
		controlItemName[1],						/* pop-up name			*/
		COLOR_TEMPERARTURE_TOTAL_PRESETS,		/* number of variants	*/
		COLOR_TEMPERARTURE_PRESET_LANDSCAPE,	/* default variant		*/
		controlItemPresetType,					/* string for pop-up	*/
		COLOR_TEMPERATURE_PRESET_TYPE_POPUP);	/* control ID			*/

	// Setup 'Observer' popup - default value "2 degrees 1931"
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_POPUP(
		controlItemName[2],						/* pop-up name			*/
		COLOR_TEMPERATURE_TOTAL_OBSERVERS,		/* number of variants	*/
		COLOR_TEMPERATURE_OBSERVER_1931_2,		/* default variant		*/
		controlItemObserver,					/* string for pop-up	*/
		COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP);	/* control ID			*/

	// Setup 'Color Temperature' slider
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_FLOAT_SLIDERX(
		controlItemName[3],
		colorTemperature2Slider(algoColorTempMin),
		colorTemperature2Slider(algoColorTempMax),
		colorTemperature2Slider(algoColorTempMin),
		colorTemperature2Slider(algoColorTempMax),
		colorTemperature2Slider(algoColorWhitePoint),
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_TEMPERATURE_COARSE_VALUE_SLIDER);

	// Setup 'Color Temperature Offset' slider
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_FLOAT_SLIDERX(
		controlItemName[4],
		algoColorTempFineMin,
		algoColorTempFineMax,
		algoColorTempFineMin,
		algoColorTempFineMax,
		algoColorTempFineDef,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_TEMPERATURE_FINE_VALUE_SLIDER);

	// Setup 'Tint coarse' slider
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_FLOAT_SLIDERX(
		controlItemName[5],
		algoColorTintMin,
		algoColorTintMax,
		algoColorTintMin,
		algoColorTintMax,
		algoColorTintDefault,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_TEMPERATURE_TINT_SLIDER);

	// Setup 'Tint fine' slider
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_FLOAT_SLIDERX(
		controlItemName[6],
		algoColorTintFineMin,
		algoColorTintFineMax,
		algoColorTintFineMin,
		algoColorTintFineMax,
		algoColorTintFineDefault,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		COLOR_TEMPERATURE_TINT_FINE_SLIDER);

    // add Color CCT bar (GUI)
    AEFX_CLR_STRUCT_EX(def);
    def.flags     = flags;
    def.ui_flags  = PF_PUI_CONTROL;
    def.ui_width  = gGuiBarWidth;
    def.ui_height = gGuiBarHeight;
    if (PremierId != in_data->appl_id)
    {
        PF_ADD_COLOR(
            controlItemName[7],
            0,
            0,
            0,
            COLOR_TEMPERATURE_COLOR_BAR_GUI);
    }
    else
    {
        PF_ADD_ARBITRARY2(
            controlItemName[7],
            gGuiBarWidth,
            gGuiBarHeight,
            0,
            PF_PUI_CONTROL,
            0,
            COLOR_TEMPERATURE_COLOR_BAR_GUI,
            0);
    }
    if (!err)
    {
        PF_CustomUIInfo	ui;
        AEFX_CLR_STRUCT_EX(ui);

        ui.events = PF_CustomEFlag_EFFECT;

        ui.comp_ui_width = 0;
        ui.comp_ui_height = 0;
        ui.comp_ui_alignment = PF_UIAlignment_NONE;

        ui.layer_ui_width = 0;
        ui.layer_ui_height = 0;
        ui.layer_ui_alignment = PF_UIAlignment_NONE;

        ui.preview_ui_width = 0;
        ui.preview_ui_height = 0;
        ui.layer_ui_alignment = PF_UIAlignment_NONE;

        err = (*(in_data->inter.register_ui))(in_data->effect_ref, &ui);
    }

    // Setup 'Camera SPD' button - initially disabled
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
	PF_ADD_BUTTON(
		controlItemName[8],
		controlItemCameraSPD,
		0,
		PF_ParamFlag_SUPERVISE,
		COLOR_TEMPERATURE_CAMERA_SPD_BUTTON
	);

	// Setup 'Load Preset' button/
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_BUTTON(
		controlItemName[9],
		controlItemLoadPreset,
		0,
		PF_ParamFlag_SUPERVISE,
		COLOR_TEMPERATURE_LOAD_PRESET_BUTTON
	);

	// Setup 'Save Preset' button
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_BUTTON(
		controlItemName[10],
		controlItemSavePreset,
		0,
		PF_ParamFlag_SUPERVISE,
		COLOR_TEMPERATURE_SAVE_PRESET_BUTTON
	);

    out_data->num_params = COLOR_TEMPERATURE_TOTAL_CONTROLS;

	return PF_Err_NONE;
}


static PF_Err
SequenceSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data
)
{
	PF_Err err = PF_Err_NONE;
#if 0
    auto const handleSuite{ AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data) };
	PF_Handle seqDataHndl = handleSuite->host_new_handle(sizeof(unflatSequenceData));
	if (nullptr != seqDataHndl)
	{
		unflatSequenceData* unflatSequenceDataH = reinterpret_cast<unflatSequenceData*>(handleSuite->host_lock_handle(seqDataHndl));
		if (nullptr != unflatSequenceDataH)
		{
			AEFX_CLR_STRUCT_EX(*unflatSequenceDataH);
			unflatSequenceDataH->isFlat = false;
			unflatSequenceDataH->magic = sequenceDataMagic;
			unflatSequenceDataH->colorCoeff.cct = unflatSequenceDataH->colorCoeff.tint = 0.f;
			unflatSequenceDataH->colorCoeff.r = unflatSequenceDataH->colorCoeff.g = unflatSequenceDataH->colorCoeff.b = 0.f;

			/* notify AE that this is our sequence data handle */
			out_data->sequence_data = seqDataHndl;
			/* unlock handle */
			handleSuite->host_unlock_handle(seqDataHndl);
		} /* if (nullptr != seqP) */
	} /* if (nullptr != seqDataHndl) */
	else
		err = PF_Err_OUT_OF_MEMORY;
#endif
	return err;
}


static PF_Err
SequenceReSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data
)
{
	PF_Err err = PF_Err_NONE;
#if 0
	auto const handleSuite{ AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data) };
	PF_Handle seqDataHndl = handleSuite->host_new_handle(sizeof(unflatSequenceData));

	/* if sequence data is present */
	if (nullptr != in_data->sequence_data)
	{
		/* get handle to flat data ... */
		PF_Handle flatSequenceDataH = in_data->sequence_data;
		/* ... then get its actual data pointer */
		flatSequenceData* flatSequenceDataP = static_cast<flatSequenceData*>(GET_OBJ_FROM_HNDL(flatSequenceDataH));
		if (nullptr != flatSequenceDataP && true == flatSequenceDataP->isFlat)
		{
			/* create a new handle, allocating the size of your (unflat) sequence data for it */
			PF_Handle unflatSequenceDataH = handleSuite->host_new_handle(sizeof(unflatSequenceData));
			if (nullptr != unflatSequenceDataH)
			{
				/* lock and get actual data pointer for unflat data */
				unflatSequenceData* unflatSequenceDataP = static_cast<unflatSequenceData*>(handleSuite->host_lock_handle(unflatSequenceDataH));
				if (nullptr != unflatSequenceDataP)
				{
					AEFX_CLR_STRUCT_EX(*unflatSequenceDataP);
					/* set flag for being "unflat" */
					unflatSequenceDataP->isFlat = false;
					/* directly copy int value unflat -> flat */
					unflatSequenceDataP->magic = flatSequenceDataP->magic;
					unflatSequenceDataP->colorCoeff = flatSequenceDataP->colorCoeff;

					/* notify AE of unflat sequence data */
					out_data->sequence_data = unflatSequenceDataH;

					/* dispose flat sequence data! */
					handleSuite->host_dispose_handle(flatSequenceDataH);
					in_data->sequence_data = nullptr;
				} /* if (nullptr != unflatSequenceDataP) */
				else
					err = PF_Err_INTERNAL_STRUCT_DAMAGED;

				/* unlock unflat sequence data handle */
				handleSuite->host_unlock_handle(unflatSequenceDataH);

			} /* if (nullptr != unflatSequenceDataH) */

		} /* if (nullptr != flatSequenceDataP && true == flatSequenceDataP->isFlat) */
		else
		{
			/* use input unflat data as unchanged output */
			out_data->sequence_data = in_data->sequence_data;
		}
	} /* if (nullptr != in_data->sequence_data) */
	else
	{
		/* no sequence data exists ? Let's create one! */
		err = SequenceSetup (in_data, out_data);
	}
#endif
	return err;
}


static PF_Err
SequenceFlatten(
	PF_InData		*in_data,
	PF_OutData		*out_data
)
{
	PF_Err err = PF_Err_NONE;
#if 0
	auto const handleSuite{ AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data) };
	if (nullptr != in_data->sequence_data)
	{
		/* assume it's always unflat data and get its handle ... */
		PF_Handle unflatSequenceDataH = in_data->sequence_data;
		/* ... then get its actual data pointer */
		unflatSequenceData* unflatSequenceDataP = static_cast<unflatSequenceData*>(GET_OBJ_FROM_HNDL(unflatSequenceDataH));
		if (nullptr != unflatSequenceDataP)
		{
			/* create a new handle, allocating the size of our (flat) sequence data for it */ 
			PF_Handle flatSequenceDataH = handleSuite->host_new_handle(sizeof(flatSequenceData));
			if (nullptr != flatSequenceDataH)
			{
				/* lock and get actual data pointer for flat data */
				flatSequenceData* flatSequenceDataP = static_cast<flatSequenceData*>(handleSuite->host_lock_handle(flatSequenceDataH));
				if (nullptr != flatSequenceDataP)
				{
					/* clear structure fields */
					AEFX_CLR_STRUCT_EX(*flatSequenceDataP);
					/* set flag for being FLAT */
					flatSequenceDataP->isFlat = true;
					/* copy values from unflat to flat */
					flatSequenceDataP->magic = unflatSequenceDataP->magic;
					flatSequenceDataP->colorCoeff = unflatSequenceDataP->colorCoeff;

					/* notify AE of new flat sequence data */
					out_data->sequence_data = flatSequenceDataH;

					/* unlock flat sequence data handle */
					handleSuite->host_unlock_handle(flatSequenceDataH);
				} /* if (nullptr != flatSequenceDataP) */

			} /* if (nullptr != flatSequenceDataH) */
			else
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;

			/* dispose unflat sequence data! */
			handleSuite->host_dispose_handle(unflatSequenceDataH);
			in_data->sequence_data = nullptr;
		} /* if (nullptr != unflatSequenceDataP) */

	} /* if (nullptr != in_data->sequence_data) */
	else
		err = PF_Err_INTERNAL_STRUCT_DAMAGED;
#endif
	return err;
}


static PF_Err
SequenceSetdown(
	PF_InData		*in_data,
	PF_OutData		*out_data
)
{
	auto const handleSuite{ AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data) };
	if (nullptr != in_data->sequence_data)
	{
		handleSuite->host_dispose_handle(in_data->sequence_data);
	}

	/* Invalidate the sequence_data pointers in both AE's input and output data fields (to signal that we have properly disposed of the data). */
	in_data->sequence_data  = nullptr;
	out_data->sequence_data = nullptr;

	return PF_Err_NONE;
}


static PF_Err
Render(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return ((PremierId == in_data->appl_id ? 
		ProcessImgInPR(in_data, out_data, params, output) : /* Premier host			*/
		ProcessImgInAE(in_data, out_data, params, output)));/* After Effect host	*/
}


inline PF_Err
PreRender(
	PF_InData		   *in_data,
	PF_OutData		   *out_data,
    PF_PreRenderExtra  *extra
)
{
	return ColorTemperarture_PreRender (in_data, out_data, extra);
}



inline PF_Err
SmartRender (
	PF_InData		     *in_data,
	PF_OutData		     *out_data,
    PF_SmartRenderExtra  *extra
)
{
	return ColorTemperature_SmartRender (in_data, out_data, extra);
}


static PF_Err
GetFlattenedSequenceData(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return PF_Err_NONE;
}


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


static void
copyConvertStringLiteralIntoUTF16(
    const wchar_t* inputString,
    A_UTF16Char* destination)
{
#ifdef AE_OS_MAC
    int length = wcslen(inputString);
    CFRange	range = { 0, 256 };
    range.length = length;
    CFStringRef inputStringCFSR = CFStringCreateWithBytes(kCFAllocatorDefault,
        reinterpret_cast<const UInt8 *>(inputString),
        length * sizeof(wchar_t),
        kCFStringEncodingUTF32LE,
        false);
    CFStringGetBytes(inputStringCFSR,
        range,
        kCFStringEncodingUTF16,
        0,
        false,
        reinterpret_cast<UInt8 *>(destination),
        length * (sizeof(A_UTF16Char)),
        NULL);
    destination[length] = 0; // Set NULL-terminator, since CFString calls don't set it
    CFRelease(inputStringCFSR);
#elif defined AE_OS_WIN
    size_t length = wcslen(inputString);
    wcscpy_s(reinterpret_cast<wchar_t*>(destination), length + 1, inputString);
#endif
}


static PF_Err
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

        copyConvertStringLiteralIntoUTF16(CUSTOM_UI_STRING, unicode_string);

        // Draw string with white color
        drawbot_color.red = drawbot_color.green = drawbot_color.blue = drawbot_color.alpha = 1.0;

        ERR(drawbotSuites.supplier_suiteP->NewBrush(supplier_ref, &drawbot_color, &string_brush_ref));

        DRAWBOT_PointF32			text_origin;

        text_origin.x = event_extra->effect_win.current_frame.left + 5.0;
        text_origin.y = event_extra->effect_win.current_frame.top + 50.0;

        ERR(drawbotSuites.surface_suiteP->DrawString(surface_ref,
            string_brush_ref,
            font_ref,
            &unicode_string[0],
            &text_origin,
            kDRAWBOT_TextAlignment_Default,
            kDRAWBOT_TextTruncation_None,
            0.0f));

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


static PF_Err
HandleEvent(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	PF_EventExtra	*extra)
{
	PF_Err		err = PF_Err_NONE;
	
	switch (extra->e_type)
	{
		case PF_Event_DRAW:
			err = DrawEvent (in_data, out_data, params, output,	extra);
		break;

//		case PF_Event_ADJUST_CURSOR:
//		break;
	
		default:
		break;
	}
	return err;
}


static PF_Err
UserChangedParam(
	PF_InData						*in_data,
	PF_OutData						*out_data,
	PF_ParamDef						*params[],
	PF_LayerDef						*outputP,
	const PF_UserChangedParamExtra	*which_hitP
)
{
	PF_Err errControl = PF_Err_NONE;

	switch (which_hitP->param_index)
	{
		case COLOR_TEMPERATURE_PRESET_CHECKBOX:
		{
			auto const& PresetUsage = params[COLOR_TEMPERATURE_PRESET_CHECKBOX]->u.bd.value;
			errControl = (PresetUsage ? PresetsActivation(in_data, out_data, params) : PresetsDeactivation(in_data, out_data, params));
		}
		break;

		case COLOR_TEMPERATURE_PRESET_TYPE_POPUP:
		{
		}
		break;

		case COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP:
		{
		}
		break;

		default:
			/* nothing ToDo */
		break;
	};

	return errControl;
}



static PF_Err
UpdateParameterUI(
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_ParamDef			*params[],
	PF_LayerDef			*outputP
) noexcept
{
	PF_Err err = PF_Err_NONE;
	return err;
}



PLUGIN_ENTRY_POINT_CALL PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err err = PF_Err_NONE;

	try {
		switch (cmd)
		{
			case PF_Cmd_ABOUT:
				ERR(About(in_data, out_data, params, output));
			break;

			case PF_Cmd_GLOBAL_SETUP:
				ERR(GlobalSetup(in_data, out_data, params, output));
			break;

			case PF_Cmd_GLOBAL_SETDOWN:
				ERR(GlobalSetdown(in_data, out_data, params, output));
			break;

			case PF_Cmd_PARAMS_SETUP:
				ERR(ParamsSetup(in_data, out_data, params, output));
			break;

			case PF_Cmd_SEQUENCE_SETUP:
				ERR(SequenceSetup(in_data, out_data));
			break;

//			case PF_Cmd_SEQUENCE_RESETUP:
//				ERR(SequenceReSetup(in_data, out_data));
//			break;

//			case PF_Cmd_SEQUENCE_FLATTEN:
//				ERR(SequenceFlatten(in_data, out_data));
//			break;

//			case PF_Cmd_SEQUENCE_SETDOWN:
//				ERR(SequenceSetdown(in_data, out_data));
//			break;

			case PF_Cmd_RENDER:
				ERR(Render(in_data, out_data, params, output));
			break;

//			case PF_Cmd_USER_CHANGED_PARAM:
//				ERR(UserChangedParam(in_data, out_data, params, output, reinterpret_cast<const PF_UserChangedParamExtra*>(extra)));
//			break;

//			case PF_Cmd_UPDATE_PARAMS_UI:
//				ERR(UpdateParameterUI(in_data, out_data, params, output));
//			break;

	        case PF_Cmd_EVENT:
				ERR(HandleEvent(in_data, out_data, params, output, reinterpret_cast<PF_EventExtra*>(extra)));
			break;

//            case PF_Cmd_QUERY_DYNAMIC_FLAGS:
//            break;

//            case PF_Cmd_SMART_PRE_RENDER:
//               ERR(PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra)));
//            break;

//            case PF_Cmd_SMART_RENDER:
//                ERR(SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra)));
//            break;

//			 case PF_Cmd_GET_FLATTENED_SEQUENCE_DATA:
//			  	 ERR(GetFlattenedSequenceData(in_data, out_data, params, output));
//			 break;

			default:
			break;
		}
	}
	catch (PF_Err& thrown_err)
	{
		err = thrown_err;
	}

	return err;
}