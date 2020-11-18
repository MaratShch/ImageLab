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

constexpr char STR_ILLUMINATE[] = "None|"
                                  "Daylight (D65)|"
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
	ILLUMINATE_NONE = 0,
	DAYLIGHT,
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
}eILLIUMINATE;


constexpr char CHROMATIC_NAME[] = "Chromatic Adaptation";
constexpr char STR_CHROMATIC[] = "Cat02|"
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

inline const float* const GetIlluminate (const eILLIUMINATE& illuminateIdx) noexcept
{
	CACHE_ALIGN static constexpr float tblIlluminate[12][3] = {
		{ 0.f },                             // NONE    
        { 95.0470f,  100.0000f, 108.8830f }, // DAYLIGHT - D65 (DEFAULT)
        { 98.0740f,  100.0000f, 118.2320f }, // OLD_DAYLIGHT
        { 99.0927f,  100.0000f,  85.3130f }, // OLD_DIRECT_SUNLIGHT_AT_NOON
        { 95.6820f,  100.0000f,  92.1490f }, // MID_MORNING_DAYLIGHT
        { 94.9720f,  100.0000f, 122.6380f }, // NORTH_SKY_DAYLIGHT
        { 92.8340f,  100.0000f, 103.6650f }, // DAYLIGHT_FLUORESCENT_F1
        { 99.1870f,  100.0000f,  67.3950f }, // COOL_FLUERESCENT
        { 103.7540f, 100.0000f,  49.8610f }, // WHITE_FLUORESCENT
        { 109.1470f, 100.0000f,  38.8130f }, // WARM_WHITE_FLUORESCENT
        { 90.8720f,  100.0000f,  98.7230f }, // DAYLIGHT_FLUORESCENT_F5
        { 100.3650f, 100.0000f,  67.8680f }  // COOL_WHITE_FLUORESCENT
    };

    return tblIlluminate[illuminateIdx];
}

inline const float* GetColorAdaptation(const eChromaticAdaptation& illuminateIdx) noexcept
{
	CACHE_ALIGN static constexpr float tblColorAdaptation[5][9] = {
		{ 0.73280f,  0.4296f, -0.16240f, -0.7036f, 1.69750f, 0.0061f, 0.0030f,  0.0136f, 0.98340f }, // CAT-02
		{ 0.40024f,  0.7076f, -0.08081f, -0.2263f, 1.16532f, 0.0457f, 0.0f,     0.0f,    0.91822f }, // VON-KRIES
		{ 0.89510f,  0.2664f, -0.16140f, -0.7502f, 1.71350f, 0.0367f, 0.0389f, -0.0685f, 1.02960f }, // BRADFORD
		{ 1.26940f, -0.0988f, -0.17060f, -0.8364f, 1.80060f, 0.0357f, 0.0297f, -0.0315f, 1.00180f }, // SHARP
		{ 0.79820f,  0.3389f, -0.13710f, -0.5918f, 1.55120f, 0.0406f, 0.0008f,  0.2390f, 0.97530f }, // CMCCAT2000
	};

	return tblColorAdaptation[illuminateIdx];
}


inline const float* GetColorAdaptationInv(const eChromaticAdaptation& illuminateIdx) noexcept
{
	CACHE_ALIGN static constexpr float tblColorAdaptationInv[5][9] = {
		{ 1.096124f, -0.278869f, 0.182745f,	0.454369f, 0.473533f,  0.072098f, -0.009628f, -0.005698f, 1.015326f }, // INV CAT-02
		{ 1.859936f, -1.129382f, 0.219897f, 0.361191f, 0.638812f,  0.0f,       0.0f,       0.0f,      1.089064f }, // INV VON-KRIES
		{ 0.986993f, -0.147054f, 0.159963f, 0.432305f, 0.518360f,  0.049291f, -0.008529f,  0.040043f, 0.968487f }, // INV BRADFORD
		{ 0.815633f,  0.047155f, 0.137217f, 0.379114f, 0.576942f,  0.044001f, -0.012260f,  0.016743f, 0.995519f }, // INV SHARP
		{ 1.062305f, -0.256743f, 0.160018f, 0.407920f, 0.55023f,   0.034437f, -0.100833f, -0.134626f, 1.016755f }, // INV CMCCAT2000
	};
	return tblColorAdaptationInv[illuminateIdx];
}


constexpr float algAWBepsilon = 0.000001f;
constexpr uint32_t iterMinCnt = 1;
constexpr uint32_t iterMaxCnt = 14;
constexpr uint32_t iterDefCnt = 2;

constexpr uint32_t gMaxCnt = CreateAlignment(iterMaxCnt, 16u);
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
