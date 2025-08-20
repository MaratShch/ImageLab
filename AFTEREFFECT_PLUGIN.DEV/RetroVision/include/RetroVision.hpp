#ifndef __IMAGE_LAB_RETRO_VISION_FILTER__
#define __IMAGE_LAB_RETRO_VISION_FILTER__

#include "CommonAdobeAE.hpp"


constexpr char strName[] = "Retro Vision";
constexpr char strCopyright[] = "\n2019-2025. ImageLab2 Copyright(c).\rRetro Vision plugin.";
constexpr int RetroVision_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int RetroVision_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int RetroVision_VersionSub = 0;
#ifdef _DEBUG
constexpr int RetroVision_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int RetroVision_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int RetroVision_VersionBuild = 1;


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) ;

PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) ;

PF_Err
RetroVision_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
);

PF_Err
RetroVision_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
);

PF_Err DrawEvent
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output,
    PF_EventExtra	*event_extra
);

#endif /* __IMAGE_LAB_RETRO_VISION_FILTER__ */
