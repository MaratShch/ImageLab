#pragma once

#include "CommonAdobeAE.hpp"

constexpr char strName[] = "Noise Clean";
constexpr char strCopyright[] = "\n2019-2020. ImageLab2 Copyright(c).\rNoise Clean plugin.";
constexpr int NoiseClean_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int NoiseClean_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int NoiseClean_VersionSub = 0;
#ifdef _DEBUG
constexpr int NoiseClean_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int NoiseClean_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int NoiseClean_VersionBuild = 1;

typedef enum {
	NOISE_CLEAN_INPUT,
	NOISE_CLEN_TOTAL_PARAMS
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

PF_Err NoiseCleanPr_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;