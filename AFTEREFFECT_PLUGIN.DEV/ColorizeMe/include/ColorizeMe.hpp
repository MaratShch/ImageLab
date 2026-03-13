#pragma once

#include <string>

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
#include "PrSDKAESupport.h"

#include "CommonAdobeAE.hpp"
#include "Param_Utils.h"

#include "Common.hpp"
#include "Param_Utils.h"

constexpr char strName[] = "ColorizeMe";
constexpr char strCopyright[] = "\n2019-2026. ImageLab2 Copyright(c).\rImage LUT & color manipulation plugin.";

constexpr int ColorizeMe_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ColorizeMe_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ColorizeMe_VersionSub = 0;
#ifdef _DEBUG
constexpr int ColorizeMe_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ColorizeMe_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int ColorizeMe_VersionBuild = 0;


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
	const PrPixelFormat&    pixelFormat
) noexcept;

bool ProcessPrImage_BGRA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

bool ProcessPrImage_BGRA_4444_16u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;

bool ProcessPrImage_VUYA_4444_8u
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output,
	const bool isBT709
) noexcept;