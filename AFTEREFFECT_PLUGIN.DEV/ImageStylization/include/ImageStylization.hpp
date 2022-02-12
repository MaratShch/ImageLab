#pragma once

#include "CommonAdobeAE.hpp"
#include "StylizationEnums.hpp"
#include "StylizationStructs.hpp"


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


template <typename T, typename U>
inline void Make_Color_pixel_yuv(U& strPix, T const& Y, T const& U, T const& V, T const& A) noexcept
{
	strPix.V = V;
	strPix.U = U;
	strPix.Y = Y;
	strPix.A = A;
	return;
}

template <typename T>
inline const T getDispersionSliderValue(const T& val) noexcept { return val + static_cast<T>(2); }


constexpr uint32_t RandomBufSize = 4096u;

typedef struct bufHandle
{
	uint32_t bValid;
	AEGP_PluginID id;
	uint32_t bufHdlType;
	void*    pBufHndl;
}bufHandle;

uint32_t utils_get_random_value (void) noexcept;
void utils_generate_random_values (float* pBuffer, const uint32_t& bufSize) noexcept;
const float* __restrict get_random_buffer(uint32_t& size) noexcept;
const float* __restrict get_random_buffer(void) noexcept;
void utils_create_random_buffer(void) noexcept;




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

PF_Err AE_ImageStyle_NewsPaper_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err AE_ImageStyle_NewsPaper_ARGB_16u
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

PF_Err AE_ImageStyle_ColorNewsPaper_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err AE_ImageStyle_ColorNewsPaper_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err PR_ImageStyle_GlassyEffect
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err AE_ImageStyle_GlassyEffect_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err AE_ImageStyle_GlassyEffect_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;


PF_Err PR_ImageStyle_CartoonEffect
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err AE_ImageStyle_CartoonEffect_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err AE_ImageStyle_CartoonEffect_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err PR_ImageStyle_SketchPencil
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;


PF_Err AE_ImageStyle_SketchPencil_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err AE_ImageStyle_SketchPencil_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;


PF_Err PR_ImageStyle_PointillismArt
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;


PF_Err AE_ImageStyle_PointillismArt_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err AE_ImageStyle_PointillismArt_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;


PF_Err PR_ImageStyle_SketchCharcoal_BGRA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err PR_ImageStyle_SketchCharcoal_VUYA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err PR_ImageStyle_SketchCharcoal_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err PR_ImageStyle_SketchCharcoal_BGRA_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err PR_ImageStyle_SketchCharcoal_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err PR_ImageStyle_SketchCharcoal
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err AE_ImageStyle_SketchCharcoal_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err AE_ImageStyle_SketchCharcoal_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;