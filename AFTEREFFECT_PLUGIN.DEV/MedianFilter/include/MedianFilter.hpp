#pragma once

#include "CommonAdobeAE.hpp"
#include "MedianFilterEnums.hpp"

constexpr char strName[] = "Median Filter";
constexpr char strCopyright[] = "\n2019-2024. ImageLab2 Copyright(c).\rMedian Filter plugin.";
constexpr int MedianFilter_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int MedianFilter_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int MedianFilter_VersionSub = 0;
#ifdef _DEBUG
constexpr int MedianFilter_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int MedianFilter_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int MedianFilter_VersionBuild = 1;

constexpr int32_t filter_radiusMin = 0;
constexpr int32_t filter_radiusMax = 30;
constexpr int32_t filter_radiusDef = filter_radiusMin;

constexpr char strSliderName[] = "Filter radius";

inline constexpr int32_t make_kernel_size (const int32_t& kernel_radius) noexcept
{
	return ((0 != kernel_radius) ? (kernel_radius * 2 + 1) : 0);
}

inline const int32_t get_kernel_size (PF_ParamDef* __restrict params[]) noexcept
{
	auto const& kernelRadius = params[MEDIAN_FILTER_SLIDER_RADIUS]->u.sd.value;
	return make_kernel_size (kernelRadius);
}


/* FUNCTION PROTOTYPES */
PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err MedianFilter_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err MedianFilter_BGRA_4444_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err MedianFilter_BGRA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err MedianFilter_VUYA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err MedianFilter_VUYA_4444_32f
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

PF_Err MeadianFilterInAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept;

PF_Err MeadianFilterInAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept;

bool median_filter_constant_time_BGRA_4444_8u
(
	const uint32_t* __restrict pInImage,
	      uint32_t* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept;

bool median_filter_constant_time_ARGB_4444_8u
(
	const uint32_t* __restrict pSrcBuffer,
	      uint32_t* __restrict pDstBuffer,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept;

bool median_filter_constant_time_BGRA_4444_16u
(
	const uint64_t* __restrict pInImage,
	      uint64_t* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept;

bool median_filter_constant_time_ARGB_4444_16u
(
	const uint64_t* __restrict pInImage,
	      uint64_t* __restrict pOutImage,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept;

bool median_filter_constant_time_BGRA_4444_32f
(
	const PF_Pixel_BGRA_32f* __restrict pSrcBuffer,
	      PF_Pixel_BGRA_32f* __restrict pDstBuffer,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept;

bool median_filter_constant_time_VUYA_4444_8u
(
	const uint32_t* __restrict pSrcBuffer,
	      uint32_t* __restrict pDstBuffer,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept;

bool median_filter_constant_time_VUYA_4444_32f
(
	const PF_Pixel_VUYA_32f* __restrict pSrcBuffer,
	      PF_Pixel_VUYA_32f* __restrict pDstBuffer,
	A_long sizeY,
	A_long sizeX,
	A_long srcLinePitch,
	A_long dstLinePitch,
	A_long kernelSize
) noexcept;