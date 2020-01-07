#pragma once
 
#include "AEConfig.h"
#include "entry.h"
#ifdef AE_OS_WIN
#include "string.h"
#endif
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "AE_EffectCBSuites.h"
#include "AE_GeneralPlug.h"
#include "AEFX_SuiteHandlerTemplate.h"

#include "CommonVersion.hpp"
#include "Param_Utils.h"

constexpr int ColorizeMe_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ColorizeMe_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ColorizeMe_VersionSub   = 0;
constexpr int ColorizeMe_VersionStage = PF_Stage_DEVELOP;
constexpr int ColorizeMe_VersionBuild = 0;


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
