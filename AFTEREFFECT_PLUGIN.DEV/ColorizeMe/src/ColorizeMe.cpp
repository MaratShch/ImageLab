#include "ColorizeMe.hpp"


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
//				ERR(About(in_data, out_data));
				break;
			case PF_Cmd_GLOBAL_SETUP:
//				ERR(GlobalSetup(out_data));
				break;
			case PF_Cmd_PARAMS_SETUP:
//				ERR(ParamsSetup(in_data, out_data));
				break;
			case PF_Cmd_RENDER:
//				ERR(Render(in_data, out_data, params, output));
				break;
			case PF_Cmd_GET_EXTERNAL_DEPENDENCIES:
//				ERR(DescribeDependencies(in_data, out_data, reinterpret_cast<PF_ExtDependenciesExtra*>(extra)));
				break;
		}
	} catch(PF_Err &thrown_err) {
		err = thrown_err;
	}
	return err;
}