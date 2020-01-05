#pragma once
 
#include "AEConfig.h"
#include "entry.h"
#ifdef AE_OS_WIN
#include "string.h"
#endif
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
//#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
//#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEFX_SuiteHandlerTemplate.h"


#ifdef __cplusplus
extern "C" {
#endif

DllExport 
PF_Err EntryPointFunc (	
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra);
	
	
#ifdef __cplusplus
}
#endif	