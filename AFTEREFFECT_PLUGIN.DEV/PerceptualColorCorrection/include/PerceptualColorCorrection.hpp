#ifndef __PERCEPTUAL_COLOR_CORRECTION__
#define __PERCEPTUAL_COLOR_CORRECTION__

#include "CommonAdobeAE.hpp"

constexpr char strName[] = "Color Band Select";
constexpr char strCopyright[] = "\n2019-2024. ImageLab2 Copyright(c).\rPerceptual Color Correction plugin.";
constexpr int PerceptualColorCorrection_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int PerceptualColorCorrection_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int PerceptualColorCorrection_VersionSub   = 0;
#ifdef _DEBUG
constexpr int PerceptualColorCorrection_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int PerceptualColorCorrection_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int PerceptualColorCorrection_VersionBuild = 1;


PF_Err RenderInPremier
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
);

PF_Err RenderInAfterEffect
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
);


#endif /* __PERCEPTUAL_COLOR_CORRECTION__ */