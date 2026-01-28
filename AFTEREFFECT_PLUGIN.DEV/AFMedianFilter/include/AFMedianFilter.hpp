#ifndef __IMAGE_LAB_IMAGE_GEOMETRY_FILTER__
#define __IMAGE_LAB_IMAGE_GEOMETRY_FILTER__

#include "CommonAdobeAE.hpp"


constexpr char strName[] = "Adaptive Frequency Median Filtering";
constexpr char strCopyright[] = "\n2019-2026. ImageLab2 Copyright(c).\rAdaptive Frequency Median Filtering plugin.";
constexpr int AFMedianFilter_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int AFMedianFilter_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int AFMedianFilter_VersionSub = 0;
#ifdef _DEBUG
constexpr int AFMedianFilter_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int AFMedianFilter_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int AFMedianFilter_VersionBuild = 1;


PF_Err 
ProcessImgInPR
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
AFMedianFilter_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
);

PF_Err
AFMedianFilter_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
);


#endif /* __IMAGE_LAB_IMAGE_GEOMETRY_FILTER__ */
