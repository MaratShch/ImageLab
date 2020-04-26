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

#include "Common.hpp"
#include "Param_Utils.h"

constexpr char strName[] = "Convolution";
constexpr char strCopyright[] = "\nImageLab2 Copyright(c).\rImage convolution plugin.";
constexpr int Convolution_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int Convolution_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int Convolution_VersionSub   = 0;
#ifdef _DEBUG
constexpr int Convolution_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int Convolution_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int Convolution_VersionBuild = 1;

constexpr char KernelType[] = "Kernel Type";

constexpr char strKernels[] = "Sharp 3x3|"
							  "Sharp 5x5    ";

enum {
	CONVOLUTION_TYPE,
	CONVOLUTION_NUM_PARAMS
};


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
