#ifndef __IMAGE_LAB_VIDEO_STABILIZATION_FILTER__
#define __IMAGE_LAB_VIDEO_STABILIZATION_FILTER__

#include "CommonAdobeAE.hpp"


constexpr char strName[] = "Guided Filter";
constexpr char strCopyright[] = "\n2019-2026. ImageLab2 Copyright(c).\rGuided Filter plugin.";
constexpr int GuidedFilter_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int GuidedFilter_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int GuidedFilter_VersionSub = 0;
#ifdef _DEBUG
constexpr int GuidedFilter_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int GuidedFilter_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int GuidedFilter_VersionBuild = 1;


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
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
GuidedFilter_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
);

PF_Err
GuidedFilter_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
);

#endif /* __IMAGE_LAB_VIDEO_STABILIZATION_FILTER__ */
