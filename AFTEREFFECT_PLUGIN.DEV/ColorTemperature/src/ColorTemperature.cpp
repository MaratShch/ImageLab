#include "ColorTemperature.hpp"
#include "ColorTemperatureGUI.hpp"
#include "ColorTemperatureControlsPresets.hpp"
#include "ImageLabMemInterface.hpp"
#include "PrSDKAESupport.h"
#include "AEGP_SuiteHandler.h"
#include "cct_interface.hpp"

#pragma pack(push)
#pragma pack(1)
typedef struct pHandle
{
    AEGP_PluginID id;
    AlgoCCT::CctHandleF32* hndl;
}pHandle;
#pragma pack(pop)


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

    // Initialize CCT LUT
    AlgoCCT::CctHandleF32* globalCctHandler32 = new (std::nothrow) AlgoCCT::CctHandleF32;
    if (nullptr != globalCctHandler32 && true == globalCctHandler32->Initialize())
    {
        // Add CctHandler to global data
        AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
        if (nullptr != (pGlobalStorage = handleSuite->host_new_handle(sizeof(pHandle))))
        {
            pHandle* pHndl = static_cast<pHandle*>(handleSuite->host_lock_handle(pGlobalStorage));
            pHndl->hndl = globalCctHandler32;

            if (PremierId != in_data->appl_id)
            {
                AEFX_SuiteScoper<AEGP_UtilitySuite3> u_suite(in_data, kAEGPUtilitySuite, kAEGPUtilitySuiteVersion3);
                u_suite->AEGP_RegisterWithAEGP(nullptr, strName, &pHndl->id);
            }
            out_data->global_data = pGlobalStorage;
            handleSuite->host_unlock_handle(pGlobalStorage);

            // Initialize PreSets
            setPresetsVector(vPresets);
            err = PF_Err_NONE;
        } // if (nullptr != (pGlobalStorage = handleSuite->host_new_handle(sizeof(globalCctHandler32))))
    }
    else
    {
        delete globalCctHandler32;
        globalCctHandler32 = nullptr;
    }

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
            AlgoCCT::CctHandleF32* globalCctHandler32 = pGlobal->hndl;
            globalCctHandler32->Deinitialize();

            delete globalCctHandler32;
            globalCctHandler32 = nullptr;
            pGlobal = nullptr;
        }

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
	constexpr PF_ParamFlags   flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
	constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;
	constexpr PF_ParamUIFlags ui_disabled_flags = ui_flags | PF_PUI_DISABLED;

	/* SetUp 'Using Preset' checkbox. Default state - non selected */
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_CHECKBOXX(
		controlItemName[0],
		FALSE,
		flags,
		COLOR_TEMPERATURE_PRESET_CHECKBOX);

	/* Setup 'Preset' popup - initially disable */
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
	PF_ADD_POPUP(
		controlItemName[1],						/* pop-up name			*/
		COLOR_TEMPERARTURE_TOTAL_PRESETS,		/* number of variants	*/
		COLOR_TEMPERARTURE_PRESET_LANDSCAPE,	/* default variant		*/
		controlItemPresetType,					/* string for pop-up	*/
		COLOR_TEMPERATURE_PRESET_TYPE_POPUP);	/* control ID			*/

	/* Setup 'Observer' popup - default value "2 degrees 1931" */
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_POPUP(
		controlItemName[2],						/* pop-up name			*/
		COLOR_TEMPERATURE_TOTAL_OBSERVERS,		/* number of variants	*/
		COLOR_TEMPERATURE_OBSERVER_1931_2,		/* default variant		*/
		controlItemObserver,					/* string for pop-up	*/
		COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP);	/* control ID			*/

	/* Setup 'Illuminant' popup - default value "D65" */
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_POPUP(
		controlItemName[3],							/* pop-up name			*/
		COLOR_TEMPERATURE_TOTAL_ILLUMINANTS,		/* number of variants	*/
		COLOR_TEMPERATURE_ILLUMINANT_D65,			/* default variant		*/
		controlItemIlluminant,						/* string for pop-up	*/
		COLOR_TEMPERATURE_ILLUMINANT_TYPE_POPUP);	/* control ID			*/

	/* Setup 'Wavelength step' popup - default value 2 nani meters */
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_POPUP(
		controlItemName[4],							/* pop-up name			*/
		COLOR_TEMPERATURE_WAVELENGTH_TOTAL_STEPS,	/* number of variants	*/
		COLOR_TEMPERATURE_WAVELENGTH_STEP_DECENT,	/* default variant		*/
		controlItemWavelengthStep,					/* string for pop-up	*/
		COLOR_TEMPERATURE_WAVELENGTH_STEP_POPUP);	/* control ID			*/
	
	/* Setup 'Color Temperature' slider */
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_FLOAT_SLIDERX(
		controlItemName[5],
		colorTemperature2Slider(algoColorTempMin),
		colorTemperature2Slider(algoColorTempMax),
		colorTemperature2Slider(algoColorTempMin),
		colorTemperature2Slider(algoColorTempMax),
		colorTemperature2Slider(algoColorWhitePoint),
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_TEMPERATURE_COARSE_VALUE_SLIDER);

	/* Setup 'Color Temperature Offset' slider */
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_FLOAT_SLIDERX(
		controlItemName[6],
		algoColorTempFineMin,
		algoColorTempFineMax,
		algoColorTempFineMin,
		algoColorTempFineMax,
		algoColorTempFineDef,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_TEMPERATURE_FINE_VALUE_SLIDER);

	/* Setup 'Tint coarse' slider */
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_FLOAT_SLIDERX(
		controlItemName[7],
		algoColorTintMin,
		algoColorTintMax,
		algoColorTintMin,
		algoColorTintMax,
		algoColorTintDefault,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_TEMPERATURE_TINT_SLIDER);

	/* Setup 'Tint fine' slider */
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_FLOAT_SLIDERX(
		controlItemName[8],
		algoColorTintFineMin,
		algoColorTintFineMax,
		algoColorTintFineMin,
		algoColorTintFineMax,
		algoColorTintFineDefault,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		COLOR_TEMPERATURE_TINT_FINE_SLIDER);

	/* Setup 'Camera SPD' button - initially disabled */
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_disabled_flags);
	PF_ADD_BUTTON(
		controlItemName[9],
		controlItemCameraSPD,
		0,
		PF_ParamFlag_SUPERVISE,
		COLOR_TEMPERATURE_CAMERA_SPD_BUTTON
	);

	/* Setup 'Load Preset' button */
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_BUTTON(
		controlItemName[10],
		controlItemLoadPreset,
		0,
		PF_ParamFlag_SUPERVISE,
		COLOR_TEMPERATURE_LOAD_PRESET_BUTTON
	);

	/* Setup 'Save Preset' button */
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_BUTTON(
		controlItemName[11],
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
		case PF_Event_DO_CLICK:
		break;

		case PF_Event_DRAG:
		break;

		case PF_Event_DRAW:
			err = DrawEvent (in_data, out_data, params, output,	extra);
		break;

		case PF_Event_ADJUST_CURSOR:
		break;
	
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

		case COLOR_TEMPERATURE_ILLUMINANT_TYPE_POPUP:
		{
		}
		break;

		case COLOR_TEMPERATURE_WAVELENGTH_STEP_POPUP:
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

			case PF_Cmd_SEQUENCE_RESETUP:
				ERR(SequenceReSetup(in_data, out_data));
			break;

			case PF_Cmd_SEQUENCE_FLATTEN:
				ERR(SequenceFlatten(in_data, out_data));
			break;

			case PF_Cmd_SEQUENCE_SETDOWN:
				ERR(SequenceSetdown(in_data, out_data));
			break;

			case PF_Cmd_RENDER:
				ERR(Render(in_data, out_data, params, output));
			break;

			case PF_Cmd_USER_CHANGED_PARAM:
				ERR(UserChangedParam(in_data, out_data, params, output, reinterpret_cast<const PF_UserChangedParamExtra*>(extra)));
			break;

			case PF_Cmd_UPDATE_PARAMS_UI:
				ERR(UpdateParameterUI(in_data, out_data, params, output));
			break;

	        case PF_Cmd_EVENT:
				ERR(HandleEvent(in_data, out_data, params, output, reinterpret_cast<PF_EventExtra*>(extra)));
			break;

            case PF_Cmd_QUERY_DYNAMIC_FLAGS:
            break;

            case PF_Cmd_SMART_PRE_RENDER:
                ERR(PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra)));
           break;

            case PF_Cmd_SMART_RENDER:
                ERR(SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra)));
            break;

//			case PF_Cmd_GET_FLATTENED_SEQUENCE_DATA:
//				ERR(GetFlattenedSequenceData(in_data, out_data, params, output));
//			break;

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