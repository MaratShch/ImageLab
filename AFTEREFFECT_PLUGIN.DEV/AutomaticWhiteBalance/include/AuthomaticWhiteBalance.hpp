#pragma once
 
#include "CommonAdobeAE.hpp"
#include "PrSDKAESupport.h"
#include "Param_Utils.h"
#include "ColorTransformMatrix.hpp"
#include "AlgCommonEnums.hpp"

constexpr int AWB_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int AWB_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int AWB_VersionSub = 0;
#ifdef _DEBUG
constexpr int AWB_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int AWB_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int AWB_VersionBuild = 1;

constexpr char strName[] = "Automatic White Balance";
constexpr char strCopyright[] = "\n2019-2025. ImageLab2 Copyright(c).\rAutomatic White Balance based on gray-point algorithm.";

bool LoadMemoryInterfaceProvider (int32_t appId, int32_t major, int32_t minor = 0) noexcept;
int32_t GetMemoryBlock (int32_t size, int32_t align, void** pMem) noexcept;
void FreeMemoryBlock (int32_t id) noexcept;


constexpr char ILLUMINATE_NAME[] = "Illuminate";

constexpr char STR_ILLUMINATE[] = "Daylight (D65)|"
                                  "Old Daylight|"
                                  "Old Direct Sunlight at Noon|"
                                  "Mid Morning Daylight|"
                                  "Nort Sky Dayligth|"
                                  "Dayligth Fluorescent F1|"
                                  "Cool Fluorescent|"
                                  "White Fluorescent|"
                                  "Warm White Fluorescent|"
                                  "Daylight Fluorescent F5|"
                                  "Cool White Fluorescent";


constexpr char CHROMATIC_NAME[] = "Chromatic Adaptation";
constexpr char STR_CHROMATIC[] = "CAT02|"
                                 "Von Kries|"
                                 "Bradford|"
                                 "Sharp|"
                                 "Cmccat2000";
             

constexpr char COLOR_SPACE_NAME_OPT[] = "Color Space";
constexpr char STR_COLOR_SPACE[] = "BT.601|"
                                   "BT.709|"
                                   "BT.2020|"
                                   "SMPTE240M";

constexpr char THRESHOLD_NAME[]  = "Gray point threshold";
constexpr char ITERATIONS_NAME[] = "Number of iterations";

constexpr float algAWBepsilon = 0.00001f;
constexpr int32_t iterMinCnt = 1;
constexpr int32_t iterMaxCnt = 14;
constexpr int32_t iterDefCnt = 2;

constexpr int32_t  gMaxCnt = CreateAlignment(iterMaxCnt, 16);
constexpr float    gConvergenceThreshold = 0.001f;

constexpr int32_t  gMinGrayThreshold = 10;
constexpr int32_t  gMaxGrayThreshold = 90;
constexpr int32_t  gDefGrayThreshold = 30;

constexpr int32_t  gTotalNumbersOfColorSpaces = (static_cast<int32_t>(SMPTE240M) + 1);
constexpr int32_t  gDefNumberOfColorSpace = static_cast<int32_t>(BT709);

PF_Err ProcessImgInAE
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output,
	const PrPixelFormat& pixelFormat
) noexcept;


PF_Err
AuthomaticWhiteBalance_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
) noexcept;

PF_Err
AuthomaticWhiteBalance_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
) noexcept;

// structure for SmartRender parameters
typedef struct
{
    eILLUMINATE          srParam_Illuminant;
    eChromaticAdaptation srParam_ChromaticAdapt;
    eCOLOR_SPACE         srParam_ColorSpace;
    float                srParam_GrayThreshold;
    int32_t              srParam_ItrerationsNumber;
} AWB_SmartRenderParams, *PAWB_SmartRenderParams;

constexpr size_t AWB_SmartRenderParamsSize = sizeof(AWB_SmartRenderParams);