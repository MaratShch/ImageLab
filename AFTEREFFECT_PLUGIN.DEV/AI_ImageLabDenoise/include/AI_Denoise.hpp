#ifndef __IMAGE_LAB_AI_IMAGE_DENOISE_FILTER__
#define __IMAGE_LAB_AI_IMAGE_DENOISE_FILTER__

#include "CommonAdobeAE.hpp"

constexpr char strName[] = "Image AI::Denoise";
constexpr char strCopyright[] = "\n2019-2026. ImageLab2 Copyright(c).\rImage AI::Denoise Filter plugin.";
constexpr int AI_Denoise_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int AI_Denoise_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int AI_Denoise_VersionSub = 0;
#ifdef _DEBUG
constexpr int AI_Denoise_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int AI_Denoise_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int AI_Denoise_VersionBuild = 1;


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
);

PF_Err
AI_Denoise_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
);

PF_Err
AI_Denoise_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
);

#endif // __IMAGE_LAB_AI_IMAGE_DENOISE_FILTER__
