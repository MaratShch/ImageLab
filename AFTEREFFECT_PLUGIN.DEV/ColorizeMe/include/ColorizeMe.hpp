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

#include "Common.hpp"
#include "Param_Utils.h"

constexpr char strName[] = "ColorizeMe";
constexpr char strCopyright[] = "\n2019-2020. ImageLab2 Copyright(c).\rImage LUT & color manipulation plugin.";

constexpr int ColorizeMe_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ColorizeMe_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ColorizeMe_VersionSub = 0;
#ifdef _DEBUG
constexpr int ColorizeMe_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ColorizeMe_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int ColorizeMe_VersionBuild = 0;

constexpr char ButtonLutParam[] = "LUT File";
constexpr char ButtonLut[] = "Load LUT";

constexpr char CheckBoxParamName[] = "Negate";
constexpr char CheckBoxName[] = "Negative";

constexpr char InterpType[] = "Interpolation type";
constexpr char Interpolation[] = "Fast (nearest)|"
                                 "Linear|"
                                 "Bilinear|"
                                 "Bicubic|"
                                 "Tetrahedral|"
                                 "Pyramidal";
enum {
	COLOR_INTERPOLATION_FAST = 0,
	COLOR_INTERPOLATION_LINEAR,
	COLOR_INTERPOLATION_BILINEAR,
	COLOR_INTERPOLATION_BICUBIC,
	COLOR_INTERPOLATION_TETRAHEDRAL,
	COLOR_INTERPOLATION_PYRAMIDAL,
	COLOR_INTERPOLATION_MAX_TYPES
};

constexpr char RedPedestalName[]   = "Red pedestal";
constexpr char GreenPedestalName[] = "Green pedestal";
constexpr char BluePedestalName[]  = "Blue pedestal";

constexpr int gMinRedPedestal   = -50;
constexpr int gMaxRedPedestal   = 50;
constexpr int gMinGreenPedestal = -50;
constexpr int gMaxGreenPedestal = 50;
constexpr int gMinBluePedestal  = -50;
constexpr int gMaxBluePedestal  = 50;

constexpr int gDefRedPedestal   = 0;
constexpr int gDefGreenPedestal = 0;
constexpr int gDefBluePedestal  = 0;

enum {
	COLOR_INPUT,
	COLOR_LUT_FILE_BUTTON,
	COLOR_NEGATE_CHECKBOX,
	COLOR_INTERPOLATION_POPUP,
	COLOR_RED_PEDESTAL_SLIDER,
	COLOR_GREEN_PEDESTAL_SLIDER,
	COLOR_BLUE_PEDESTAL_SLIDER,
	COLOR_TOTAL_PARAMS
};