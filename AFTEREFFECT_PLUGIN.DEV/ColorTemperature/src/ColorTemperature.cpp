#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureGUI.hpp"
#include "ColorTemperatureSeqData.hpp"
#include "PrSDKAESupport.h"
#include "AEGP_SuiteHandler.h"

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
	PF_Err	err = PF_Err_NONE;

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
		auto const& pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

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

	return err;
}


static PF_Err
GlobalSetdown(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	/* nothing to do */
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
	
	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_POPUP(
		controlName[0],						/* pop-up name			*/
		eTEMP_STANDARD_TOTAL,				/* number of variants	*/
		eTEMP_STANDARD_BLACK_BODY,			/* default variant		*/
		strStandardName,					/* string for pop-up	*/
		COLOR_TEMPERATURE_STANDARD_POPUP);	/* control ID			*/

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_POPUP(
		controlName[1],						/* pop-up name			*/
		eTEMP_GAMMA_VALUE_TOTAL,			/* number of variants	*/
		eTEMP_GAMMA_VALUE_10,		    	/* default variant		*/
		strGammaValueName,					/* string for pop-up	*/
		COLOR_TEMPERATURE_GAMMA_POPUP);		/* control ID			*/

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_FLOAT_SLIDERX(
		controlName[2],
		colorTemperature2Slider(algoColorTempMin),
		colorTemperature2Slider(algoColorTempMax),
		colorTemperature2Slider(algoColorTempMin),
		colorTemperature2Slider(algoColorTempMax),
		colorTemperature2Slider(algoColorWhitePoint),
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_TEMPERATURE_VALUE_SLIDER);

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_FLOAT_SLIDERX(
		controlName[3],
		algoColorTempFineMin,
		algoColorTempFineMax,
		algoColorTempFineMin,
		algoColorTempFineMax,
		algoColorTempFineDef,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_TEMPERATURE_FINE_VALUE_SLIDER);

	AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
	PF_ADD_FLOAT_SLIDERX(
		controlName[4],
		algoColorTintMin,
		algoColorTintMax,
		algoColorTintMin,
		algoColorTintMax,
		algoColorTintDefault,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_TEMPERATURE_TINT_SLIDER);

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
	auto const& handleSuite{ AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data) };
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

	return err;
}


static PF_Err
SequenceReSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data
)
{
	PF_Err err = PF_Err_NONE;
	auto const& handleSuite{ AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data) };
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

	return err;
}


static PF_Err
SequenceFlatten(
	PF_InData		*in_data,
	PF_OutData		*out_data
)
{
	PF_Err err = PF_Err_NONE;
	auto const& handleSuite{ AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data) };
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

	return err;
}


static PF_Err
SequenceSetdown(
	PF_InData		*in_data,
	PF_OutData		*out_data
)
{
	auto const& handleSuite{ AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data) };
	if (nullptr != in_data->sequence_data)
	{
#ifdef _DEBUG
		unflatSequenceData* unflatSequenceDataH = reinterpret_cast<unflatSequenceData*>(GET_OBJ_FROM_HNDL(in_data->sequence_data));
		/* ! cleanup released handle for DBG purposes only ! */
		unflatSequenceDataH->magic = 0x0;
		unflatSequenceDataH->colorCoeff.cct = unflatSequenceDataH->colorCoeff.tint = unflatSequenceDataH->colorCoeff.r = unflatSequenceDataH->colorCoeff.g = unflatSequenceDataH->colorCoeff.b = 0.f;
		unflatSequenceDataH->isFlat = false;
#endif
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
	return ((PremierId == in_data->appl_id ? ProcessImgInPR(in_data, out_data, params, output) : ProcessImgInAE(in_data, out_data, params, output)));
}


static PF_Err
SmartPreRender(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return PF_Err_NONE;
}



static PF_Err
SmartRender(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	return PF_Err_NONE;
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
	return PF_Err_NONE;
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


#ifdef _DEBUG
 #include <atomic>
 constexpr uint32_t dbgArraySize = 2048u;
 static uint32_t dbgArray[dbgArraySize]{};
 std::atomic<uint32_t> dbgCnt = 0u;
#endif


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
#ifdef _DEBUG
		if (dbgCnt < dbgArraySize)
		{
			dbgArray[dbgCnt] = cmd;
			dbgCnt++;
		}
#endif
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

			case PF_Cmd_SMART_PRE_RENDER:
				ERR(SmartPreRender(in_data, out_data, params, output));
			break;
			
			case PF_Cmd_SMART_RENDER:
				ERR(SmartRender(in_data, out_data, params, output));
			break;

//			case PF_Cmd_GET_FLATTENED_SEQUENCE_DATA:
//				ERR(GetFlattenedSequenceData(in_data, out_data, params, output));
//			break;

			default:
			break;
		}
	}
	catch (PF_Err & thrown_err)
	{
		err = thrown_err;
	}

	return err;
}