#pragma once

#include "CommonAdobeAE.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ColorTransformMatrix.hpp"

constexpr char strName[] = "Color Correction CieLAB";
constexpr char strCopyright[] = "\n2019-2023. ImageLab2 Copyright(c).\rColor Correction in CieLAB color space.";
constexpr int ColorCorrection_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ColorCorrection_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ColorCorrection_VersionSub   = 0;
#ifdef _DEBUG
constexpr int ColorCorrection_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ColorCorrection_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int ColorCorrection_VersionBuild = 1;


/* FUNCTIONS PROTOTYPES */
PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept;

PF_Err
ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept;
