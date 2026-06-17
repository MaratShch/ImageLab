#ifndef __IMAGE_LAB_IMAGE_AWB2_FILTER__
#define __IMAGE_LAB_IMAGE_AWB2_FILTER__

#include "CommonAdobeAE.hpp"
#include "PrSDKAESupport.h"
#include "Param_Utils.h"



constexpr char strName[] = "Automatic White Balance 2";
constexpr char strCopyright[] = "\n2019-2026. ImageLab2 Copyright(c).\rAutomatic White Balance Filter plugin.";
constexpr int AWB2_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int AWB2_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int AWB2_VersionSub = 0;
#ifdef _DEBUG
constexpr int AWB2_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int AWB2_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int AWB2_VersionBuild = 1;


PF_Err ProcessImgInPR
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
);

PF_Err
ProcessImgInAE
(
    PF_InData*		in_data,
    PF_OutData*		out_data,
    PF_ParamDef*	params[],
    PF_LayerDef*	output
) ;

PF_Err
AuthomaticWhiteBalance_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
);

PF_Err
AuthomaticWhiteBalance_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
);


#endif // __IMAGE_LAB_IMAGE_AWB2_FILTER__
