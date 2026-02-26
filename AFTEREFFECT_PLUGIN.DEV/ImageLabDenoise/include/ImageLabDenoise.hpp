#ifndef __IMAGE_LAB_DENOISE_FILTER__
#define __IMAGE_LAB_DENOISE_FILTER__

#include "CommonAdobeAE.hpp"


constexpr char strName[] = "ImageLab Noise Clinic";
constexpr char strCopyright[] = "\n2019-2026. ImageLab2 Copyright(c).\rImageLab Noise Clinic plugin.";
constexpr int ImageLabDenoise_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ImageLabDenoise_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ImageLabDenoise_VersionSub = 0;
#ifdef _DEBUG
constexpr int ImageLabDenoise_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ImageLabDenoise_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int ImageLabDenoise_VersionBuild = 1;


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
);

PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
);

PF_Err
ImageLabDenoise_PreRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_PreRenderExtra		*extraP
);

PF_Err
ImageLabDenoise_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
);

PF_Err
ImageLabDenoise_SequenceSetup
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output
);

PF_Err
ImageLabDenoise_SequenceReSetup
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output
);

PF_Err
ImageLabDenoise_SequenceFlatten
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output
);

PF_Err
ImageLabDenoise_SequenceSetdown
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output
);

PF_Err
SetupControlElements
(
    const PF_InData*  RESTRICT in_data,
          PF_OutData* RESTRICT out_data
);


#endif /* __IMAGE_LAB_DENOISE_FILTER__ */
