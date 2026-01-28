#ifndef __IMAGE_LAB_AVERAGE_FILTER__
#define __IMAGE_LAB_AVERAGE_FILTER__

#include "CommonAdobeAE.hpp"


constexpr char strName[] = "Average Filter";
constexpr char strCopyright[] = "\n2019-2024. ImageLab2 Copyright(c).\rAverage Filter plugin.";
constexpr int AverageFilter_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int AverageFilter_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int AverageFilter_VersionSub = 0;
#ifdef _DEBUG
constexpr int AverageFilter_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int AverageFilter_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int AverageFilter_VersionBuild = 1;


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept;

PF_Err AverageFilter_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)noexcept;

PF_Err AverageFilter_SmartRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_SmartRenderExtra	*extraP
) noexcept;


#endif /* __IMAGE_LAB_AVERAGE_FILTER__ */
