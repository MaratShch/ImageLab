#ifndef __IMAGE_LAB_VIDEO_STABILIZATION_FILTER__
#define __IMAGE_LAB_VIDEO_STABILIZATION_FILTER__

#include "CommonAdobeAE.hpp"


constexpr char strName[] = "Image Stabilizer";
constexpr char strCopyright[] = "\n2019-2023. ImageLab2 Copyright(c).\rImage Stabilizer plugin.";
constexpr int ImageStabilizer_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ImageStabilizer_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ImageStabilizer_VersionSub = 0;
#ifdef _DEBUG
constexpr int ImageStabilizer_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ImageStabilizer_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int ImageStabilizer_VersionBuild = 1;


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
