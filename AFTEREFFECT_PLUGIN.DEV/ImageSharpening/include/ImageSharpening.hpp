#ifndef __IMAGE_LAB_VIDEO_STABILIZATION_FILTER__
#define __IMAGE_LAB_VIDEO_STABILIZATION_FILTER__

#include "CommonAdobeAE.hpp"


constexpr char strName[] = "Image Sharpening";
constexpr char strCopyright[] = "\n2019-2024. ImageLab2 Copyright(c).\rImage Sharpening plugin.";
constexpr int ImageSharpening_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ImageSharpening_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ImageSharpening_VersionSub = 0;
#ifdef _DEBUG
constexpr int ImageSharpening_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ImageSharpening_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int ImageSharpening_VersionBuild = 1;


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


#endif /* __IMAGE_LAB_VIDEO_STABILIZATION_FILTER__ */
