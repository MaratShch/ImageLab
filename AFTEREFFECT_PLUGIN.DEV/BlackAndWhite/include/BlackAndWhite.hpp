#pragma once

#include "CommonAdobeAE.hpp"


constexpr char strName[] = "Image Flipping";
constexpr char strCopyright[] = "\n2019-2023. ImageLab2 Copyright(c).\rImage Black and White plugin.";
constexpr int EqualizationFilter_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int EqualizationFilter_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int EqualizationFilter_VersionSub = 0;
#ifdef _DEBUG
constexpr int EqualizationFilter_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int EqualizationFilter_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int EqualizationFilter_VersionBuild = 1;

typedef enum {
	IMAGE_BW_FILTER_INPUT = 0,
	IMAGE_BW_ADVANCED_ALGO,
	IMAGE_BW_FILTER_TOTAL_PARAMS
}Item;

constexpr char cAdvanced[] = "Advanced Algorithm";

PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

PF_Err ProcessImgInAE
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;