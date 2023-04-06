#pragma once

#include "CommonAdobeAE.hpp"
#include "ImageEqualizationEnums.hpp"
#include "ImageEqualizationFuncProto.hpp"
#include "ImageLabMemInterface.hpp"

bool LoadMemoryInterfaceProvider(int32_t appId, int32_t major, int32_t minor = 0);
int32_t GetMemoryBlock (int32_t size, int32_t align, void** pMem);
void FreeMemoryBlock (int32_t id);


constexpr char strName[] = "Image Equalization";
constexpr char strCopyright[] = "\n2019-2023. ImageLab2 Copyright(c).\rImage Equalization plugin.";
constexpr int EqualizationFilter_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int EqualizationFilter_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int EqualizationFilter_VersionSub = 0;
#ifdef _DEBUG
constexpr int EqualizationFilter_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int EqualizationFilter_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int EqualizationFilter_VersionBuild = 1;


constexpr char STR_EQ_ALGO_POPUP[] = "Equalization Presets";

constexpr char STR_EQ_ALGO_TYPE[] =	"None|"
                                    "Manual|"
                                    "Linear|" 
									"Details in dark|"
									"Details in light|"
									"Exponential|"
	                                "Sigmoid";

constexpr char STR_EQ_DARK_SLIDER[]     = "Dark channel details";
constexpr char STR_EQ_LIGHT_SLIDER[]    = "Light channel details";
constexpr char STR_EQ_PEDESTAL_SLIDER[] = "Dark pedestal";
constexpr char STR_EQ_CHECKBOX_FLICK[]  = "Flicker removing";

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