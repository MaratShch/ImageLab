#ifndef __IMAGE_LAB_ART_POINTILISM_FILTER__
#define __IMAGE_LAB_ART_POINTILISM_FILTER__

#include "CommonAdobeAE.hpp"
#include "ArtPointillismControl.hpp"

constexpr char strName[] = "Art Pointillism";
constexpr char strCopyright[] = "\n2019-2026. ImageLab2 Copyright(c).\rArt Pointillism plugin.";
constexpr int ArtPointillism_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ArtPointillism_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ArtPointillism_VersionSub = 0;
#ifdef _DEBUG
constexpr int ArtPointillism_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ArtPointillism_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int ArtPointillism_VersionBuild = 1;


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
ArtPointilism_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
);

PF_Err
ArtPointilism_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
);

PF_Err
SetupControlElements
(
    const PF_InData* RESTRICT in_data,
    PF_OutData* RESTRICT out_data
);

PontillismControls GetControlParametersStruct
(
    PF_ParamDef* RESTRICT params[]
);


#endif // __IMAGE_LAB_ART_POINTILISM_FILTER__
