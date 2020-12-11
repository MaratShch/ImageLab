#pragma once
 
#include "AEConfig.h"
#include "entry.h"
#ifdef AE_OS_WIN
#include "string.h"
#endif
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "AE_EffectCBSuites.h"
#include "AE_GeneralPlug.h"
#include "AEFX_SuiteHandlerTemplate.h"

#include "CommonAdobeAE.hpp"
#include "Param_Utils.h"
#include "ColorTransformMatrix.hpp"

constexpr int AWB_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int AWB_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int AWB_VersionSub = 0;
#ifdef _DEBUG
constexpr int AWB_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int AWB_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int AWB_VersionBuild = 1;

constexpr char strName[] = "Automatic White Balance";
constexpr char strCopyright[] = "\n2019-2020. ImageLab2 Copyright(c).\rAutomatic White Balance based on gray-point algorithm.";

typedef enum
{
	AWB_INPUT,
	AWB_ILLUMINATE_POPUP,
	AWB_CHROMATIC_POPUP,
	AWB_COLOR_SPACE_POPUP,
	AWB_THRESHOLD_SLIDER,
	AWB_ITERATIONS_SLIDER,
	AWB_TOTAL_CONTROLS
}eImageLab2AWB_Controls;

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

typedef enum
{
	DAYLIGHT = 0,
	OLD_DAYLIGHT,
	OLD_DIRECT_SUNLIGHT_AT_NOON,
	MID_MORNING_DAYLIGHT,
	NORTH_SKY_DAYLIGHT,
	DAYLIGHT_FLUORESCENT_F1,
	COOL_FLUERESCENT,
	WHITE_FLUORESCENT,
	WARM_WHITE_FLUORESCENT,
	DAYLIGHT_FLUORESCENT_F5,
	COOL_WHITE_FLUORESCENT,
	TOTAL_ILLUMINATES
}eILLUMINATE;


constexpr char CHROMATIC_NAME[] = "Chromatic Adaptation";
constexpr char STR_CHROMATIC[] = "CAT02|"
                                 "Von Kries|"
                                 "Bradford|"
                                 "Sharp|"
                                 "Cmccat2000";
             
typedef enum
{
	CHROMATIC_CAT02 = 0,
	CHROMATIC_VON_KRIES,
	CHROMATIC_BRADFORD,
	CHROMATIC_SHARP,
	CHROMATIC_CMCCAT2000,
	TOTAL_CHROMATIC
}eChromaticAdaptation;

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
constexpr int32_t  gDefNumberOfColorSpace = static_cast<int32_t>(BT601);

PF_Err ProcessImgInAE
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const PrPixelFormat& pixelFormat
) noexcept;


void compute_correction_matrix
(
	const float uAvg,
	const float vAvg,
	const eCOLOR_SPACE colorSpace,
	const eILLUMINATE  illuminate,
	const eChromaticAdaptation chromatic,
	float* __restrict outMatrix /* pointer for hold correction matrix (3 values as minimal) */
) noexcept;