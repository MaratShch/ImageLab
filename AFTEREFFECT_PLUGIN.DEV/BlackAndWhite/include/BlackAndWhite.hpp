#pragma once

#include "CommonAdobeAE.hpp"

constexpr char strName[] = "BlackAndWhite";
constexpr char strCopyright[] = "\n2019-2023. ImageLab2 Copyright(c).\rImage Black and White plugin.";
constexpr int BWFilter_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int BWFilter_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int BWFilter_VersionSub = 0;
#ifdef _DEBUG
constexpr int BWFilter_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int BWFilter_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int BWFilter_VersionBuild = 1;

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

PF_Err ProcessImgInAE_8bits
(
    PF_InData*    in_data,
    PF_OutData*   out_data,
    PF_ParamDef*  params[],
    PF_LayerDef*  output
) noexcept;

PF_Err ProcessImgInAE_16bits
(
    PF_InData*    in_data,
    PF_OutData*   out_data,
    PF_ParamDef*  params[],
    PF_LayerDef*  output
) noexcept;

PF_Err ProcessImgInAE_32bits
(
    PF_InData*    in_data,
    PF_OutData*   out_data,
    PF_ParamDef*  params[],
    PF_LayerDef*  output
) noexcept;