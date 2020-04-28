#include "Convolution.hpp"

static PF_Err
About(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_SPRINTF(	out_data->return_msg,
				"%s, v%d.%d\r%s",
				strName,
				Convolution_VersionMajor,
				Convolution_VersionMinor,
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

	out_data->my_version =
		PF_VERSION (
			Convolution_VersionMajor,
			Convolution_VersionMinor,
			Convolution_VersionSub,
			Convolution_VersionStage,
			Convolution_VersionBuild
		);

	out_data->out_flags =	PF_OutFlag_PIX_INDEPENDENT	|
							PF_OutFlag_DEEP_COLOR_AWARE |
							PF_OutFlag_NON_PARAM_VARY;

	out_data->out_flags2 =	PF_OutFlag2_FLOAT_COLOR_AWARE |
							PF_OutFlag2_SUPPORTS_SMART_RENDER;

	return err;
}


static PF_Err
ParamsSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_Err		err = PF_Err_NONE;
	PF_ParamDef	def;

	AEFX_CLR_STRUCT_EX(def);

	def.flags = PF_ParamFlag_SUPERVISE |
		PF_ParamFlag_CANNOT_TIME_VARY  |
		PF_ParamFlag_CANNOT_INTERP;

	def.ui_flags = PF_PUI_STD_CONTROL_ONLY;

	PF_ADD_POPUP(KernelType, 2, 1, strKernels, 1);

	out_data->num_params = CONVOLUTION_NUM_PARAMS;

	return err;
}


DllExport	PF_Err 
EntryPointFunc (	
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err		err = PF_Err_NONE;
	
	try {
		switch (cmd) {
			case PF_Cmd_ABOUT:
				ERR(About(in_data, out_data, params, output));
				break;
			case PF_Cmd_GLOBAL_SETUP:
				ERR(GlobalSetup(in_data, out_data, params, output));
				break;
			case PF_Cmd_PARAMS_SETUP:
				ERR(ParamsSetup(in_data, out_data, params, output));
				break;
			case PF_Cmd_RENDER:
//				ERR(Render(in_data, out_data, params, output));
				break;
			case PF_Cmd_SMART_PRE_RENDER:
//				ERR(PreRender(in_dataP, out_data, (PF_PreRenderExtra*)extra));
				break;
			case PF_Cmd_SMART_RENDER:
//				ERR(SmartRender(in_dataP, out_data, (PF_SmartRenderExtra*)extra));
				break;
		}
	} catch (PF_Err &thrown_err) {
		err = thrown_err;
	}
	return err;
}