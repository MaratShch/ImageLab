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

template <typename T, typename U>
inline void Make_BW_pixel (U& strPix, T const& bwVal, T const& alpha) noexcept
{
	strPix.A = alpha;
	strPix.B = strPix.G = strPix.R = bwVal;
	return;
}

template <typename T, typename U>
inline void Make_Color_pixel(U& strPix, T const& R, T const& G, T const& B, T const& A) noexcept
{
	strPix.B = B;
	strPix.G = G;
	strPix.R = R;
	strPix.A = A;
	return;
}


typedef enum {
	IMAGE_STYLE_INPUT,
	IMAGE_STYLE_POPUP,
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

PF_Err PR_ImageStyle_NewsPaper
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err PR_ImageStyle_ColorNewsPaper
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;