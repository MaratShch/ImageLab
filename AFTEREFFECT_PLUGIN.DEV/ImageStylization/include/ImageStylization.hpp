#pragma once

#include "CommonAdobeAE.hpp"
#include "StylizationEnums.hpp"


constexpr char strName[] = "Image Stylization";
constexpr char strCopyright[] = "\n2019-2020. ImageLab2 Copyright(c).\rImage Stylization plugin.";
constexpr int ImageStyle_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ImageStyle_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ImageStyle_VersionSub = 0;
#ifdef _DEBUG
constexpr int ImageStyle_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ImageStyle_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int ImageStyle_VersionBuild = 1;

typedef enum {
	IMAGE_STYLE_INPUT,
	IMAGE_STYLE_TOTAL_PARAMS
}Item;


/* FUNCTION PROTOTYPES */
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

PF_Err ImageStylizationPr_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;