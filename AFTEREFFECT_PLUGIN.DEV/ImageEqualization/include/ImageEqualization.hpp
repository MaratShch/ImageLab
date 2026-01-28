#pragma once

#include "CommonAdobeAE.hpp"
#include "ImageEqualizationEnums.hpp"
#include "ImageEqualizationStrings.hpp"
#include "ImageEqualizationFuncProto.hpp"
#include "ImageLabMemInterface.hpp"

bool LoadMemoryInterfaceProvider(int32_t appId, int32_t major, int32_t minor = 0) noexcept;
int32_t GetMemoryBlock (int32_t size, int32_t align, void** pMem) noexcept;
void FreeMemoryBlock   (int32_t id) noexcept;


constexpr char strName[] = "Image Equalization";
constexpr char strCopyright[] = "\n2019-2023. ImageLab2 Copyright(c).\rImage Equalization plugin.";
constexpr int EqualizationFilter_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int EqualizationFilter_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int EqualizationFilter_VersionSub = 0;
#ifdef _DEBUG
constexpr int EqualizationFilter_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int EqualizationFilter_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int EqualizationFilter_VersionBuild = 1;



/* FUNCTION PROTOTYPES */
PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept;


PF_Err PR_ImageEq_Linear_BGRA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;