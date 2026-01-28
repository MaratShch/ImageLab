#ifndef __IMAGE_LAB_BILATERAL_FILTER_STANDALONE__
#define __IMAGE_LAB_BILATERAL_FILTER_STANDALONE__

#include "CommonAdobeAE.hpp"


constexpr char strName[] = "Bilateral Filter";
constexpr char strCopyright[] = "\n2019-2025. ImageLab2 Copyright(c).\rBilateral Filter plugin.";
constexpr int BilateralFilter_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int  BilateralFilter_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int BilateralFilter_VersionSub = 0;
#ifdef _DEBUG
constexpr int BilateralFilter_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int BilateralFilter_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int BilateralFilter_VersionBuild = 1;


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


#endif // __IMAGE_LAB_BILATERAL_FILTER_STANDALONE__
