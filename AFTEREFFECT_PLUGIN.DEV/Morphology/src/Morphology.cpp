#include "Morphology.hpp"
#include "PrSDKAESupport.h"
#include "MorphologyEnums.hpp"
#include "MorphologyStrings.hpp"
#include "SE_Interface.hpp"

static PF_Err
About (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_SPRINTF(out_data->return_msg,
		"%s, v%d.%d\r%s",
		strName,
		MorphologyFilter_VersionMajor,
		MorphologyFilter_VersionMinor,
		strCopyright);

	return PF_Err_NONE;
}


static PF_Err
GlobalSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_Err	err = PF_Err_NONE;

	constexpr PF_OutFlags out_flags1 =
		PF_OutFlag_PIX_INDEPENDENT       |
		PF_OutFlag_SEND_UPDATE_PARAMS_UI |
		PF_OutFlag_USE_OUTPUT_EXTENT     |
		PF_OutFlag_DEEP_COLOR_AWARE      |
		PF_OutFlag_WIDE_TIME_INPUT;

	constexpr PF_OutFlags out_flags2 =
		PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG |
		PF_OutFlag2_DOESNT_NEED_EMPTY_PIXELS         |
		PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT;

	out_data->my_version =
		PF_VERSION(
			MorphologyFilter_VersionMajor,
			MorphologyFilter_VersionMinor,
			MorphologyFilter_VersionSub,
			MorphologyFilter_VersionStage,
			MorphologyFilter_VersionBuild
		);

	out_data->out_flags = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
			AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);

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
ParamsSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	CACHE_ALIGN PF_ParamDef	def;
	PF_Err		err = PF_Err_NONE;
	constexpr PF_ParamFlags flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
	constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_POPUP(
		MorphOperationType,         /* pop-up name          */
		SE_OP_TOTAL,                /* number of operations */
		SE_OP_NONE,                 /* default operation    */
		strMorphOperation,          /* string for pop-up    */
		MORPHOLOGY_OPERATION_TYPE); /* control ID           */

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = (ui_flags | PF_PUI_DISABLED);
	PF_ADD_POPUP(
		MorphSeType,                /* pop-up name          */
		SE_TYPE_TOTALS,             /* number of operations */
		SE_TYPE_SQUARE,             /* default operation    */
		strStructuredElement,       /* string for pop-up    */
		MORPHOLOGY_ELEMENT_TYPE);   /* control ID           */

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = (ui_flags | PF_PUI_DISABLED);
	PF_ADD_POPUP(
		MorphSeSize,                /* pop-up name          */
		SE_TYPE_TOTALS,             /* number of operations */
		SE_TYPE_SQUARE,             /* default operation    */
		strElemSize,                /* string for pop-up    */
		MORPHOLOGY_KERNEL_SIZE);    /* control ID           */

	out_data->num_params = MORPHOLOGY_FILTER_TOTAL_PARAMS;
	return PF_Err_NONE;
}

static PF_Err
GlobalSetdown (
	PF_InData* in_data
)
{
	return PF_Err_NONE;
}



static PF_Err
Render (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_Err	err = PF_Err_NONE;

	if (SE_OP_NONE == params[MORPHOLOGY_OPERATION_TYPE]->u.pd.value - 1)
	{
		if (PremierId == in_data->appl_id)
			err = PF_COPY(&params[MORPHOLOGY_FILTER_INPUT]->u.ld, output, NULL, NULL);
		else
			err = (PF_Quality_HI == in_data->quality) ?
			AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data)->
				copy_hq(in_data->effect_ref, &params[MORPHOLOGY_FILTER_INPUT]->u.ld, output, NULL, NULL) :
			AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data)->
				copy(in_data->effect_ref, &params[MORPHOLOGY_FILTER_INPUT]->u.ld, output, NULL, NULL);
	}
	else
	{
		err = ((PremierId == in_data->appl_id ? ProcessImgInPR(in_data, out_data, params, output) : ProcessImgInAE(in_data, out_data, params, output)));
	}

	return err;
}



static PF_Err
SmartRender (
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_SmartRenderExtra		*extraP
)
{
	PF_Err	err = PF_Err_NONE;
	return err;
}


static PF_Err
UserChangedParam (
	PF_InData						*in_data,
	PF_OutData						*out_data,
	PF_ParamDef						*params[],
	PF_LayerDef						*outputP,
	const PF_UserChangedParamExtra	*which_hitP
)
{
	PF_Err err = PF_Err_NONE;

	switch (which_hitP->param_index)
	{
		case MORPHOLOGY_OPERATION_TYPE:
		{
			auto const& cType = params[MORPHOLOGY_OPERATION_TYPE]->u.pd.value;
			PF_ParamDef* pMorphologyTypeParam = params[MORPHOLOGY_ELEMENT_TYPE];
			PF_ParamDef* pMorphologySizeParam = params[MORPHOLOGY_KERNEL_SIZE];

			if (SE_OP_NONE == static_cast<SeOperation const>(cType - 1))
			{
				pMorphologyTypeParam->ui_flags |= PF_PUI_DISABLED;
				pMorphologySizeParam->ui_flags |= PF_PUI_DISABLED;
			}
			else
			{

				pMorphologyTypeParam->ui_flags &= ~PF_PUI_DISABLED;
				pMorphologySizeParam->ui_flags &= ~PF_PUI_DISABLED;

				const SeType seElemType = static_cast<SeType>(pMorphologyTypeParam->u.pd.value - 1);
				const SeSize seElemSize = static_cast<SeSize>(pMorphologySizeParam->u.pd.value - 1);
				
				reinterpret_cast<strSeData*>(GET_OBJ_FROM_HNDL(out_data->sequence_data))->IstructElem = CreateSeInterface(seElemType, seElemSize);
				reinterpret_cast<strSeData*>(GET_OBJ_FROM_HNDL(out_data->sequence_data))->bValid = true;
			}

			AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data)->
				PF_UpdateParamUI(in_data->effect_ref, MORPHOLOGY_ELEMENT_TYPE, pMorphologyTypeParam);
			AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data)->
				PF_UpdateParamUI(in_data->effect_ref, MORPHOLOGY_KERNEL_SIZE, pMorphologySizeParam);
		}
		break;

		default:
		break;
	}

	return err;
}


static PF_Err
UpdateParameterUI (
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_ParamDef			*params[],
	PF_LayerDef			*outputP
)
{
	CACHE_ALIGN PF_ParamDef param_copy[MORPHOLOGY_FILTER_TOTAL_PARAMS]{};
	MakeParamCopy (params, param_copy, MORPHOLOGY_FILTER_TOTAL_PARAMS);

	PF_Err err = PF_Err_NONE;
	return err;
}


static PF_Err
SequenceSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
) 
{
	strSeData* seqData = (nullptr != out_data->sequence_data ? reinterpret_cast<strSeData*>(GET_OBJ_FROM_HNDL(out_data->sequence_data)) : nullptr);

	if (seqData != nullptr)
	{
		if (nullptr != seqData->IstructElem)
			delete(seqData->IstructElem);
		PF_DISPOSE_HANDLE_EX(out_data->sequence_data);
	}

	out_data->sequence_data = PF_NEW_HANDLE(strSeDataSize);
	out_data->flat_sdata_size = strSeDataSize;

	return (!out_data->sequence_data ? PF_Err_INTERNAL_STRUCT_DAMAGED : PF_Err_NONE);
}

static PF_Err
SequenceReSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
)
{
	PF_Err err = PF_Err_NONE;
	return err;
}


static PF_Err
SequenceSetdown (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output
)
{
	if (nullptr != out_data->sequence_data)
	{
		strSeData* seqData = reinterpret_cast<strSeData*>(GET_OBJ_FROM_HNDL(out_data->sequence_data));
		if (nullptr != seqData)
			delete(seqData->IstructElem);
		PF_DISPOSE_HANDLE_EX(out_data->sequence_data);
	}
	return PF_Err_NONE;
}



PLUGIN_ENTRY_POINT_CALL PF_Err
EffectMain (
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err		err = PF_Err_NONE;

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
				ERR(GlobalSetdown(in_data));
			break;

			case PF_Cmd_PARAMS_SETUP:
				ERR(ParamsSetup(in_data, out_data, params, output));
			break;

			case PF_Cmd_SEQUENCE_SETUP:
				ERR(SequenceSetup(in_data, out_data, params, output));
			break;

//			case PF_Cmd_SEQUENCE_RESETUP:
//				ERR(SequenceReSetup(in_data, out_data, params, output));
//			break;

			case PF_Cmd_SEQUENCE_SETDOWN:
				ERR(SequenceSetdown(in_data, out_data, params, output));
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

			default:
			break;
		}
	}
	catch (PF_Err &thrown_err)
	{
		err = thrown_err;
	}

	return err;
}