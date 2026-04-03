#ifndef __IMAGE_LAB_ART_PAINT_STYLE_FILTER__
#define __IMAGE_LAB_ART_PAINT_STYLE_FILTER__

#include "CommonAdobeAE.hpp"
#include "PaintAlgoContols.hpp"

constexpr char strName[] = "Art::Paint";
constexpr char strCopyright[] = "\n2019-2026. ImageLab2 Copyright(c).\rArt Paint plugin.";
constexpr int ArtPaint_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ArtPaint_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ArtPaint_VersionSub = 0;
#ifdef _DEBUG
constexpr int ArtPaint_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ArtPaint_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int ArtPaint_VersionBuild = 1;


PF_Err ProcessImgInPR
(
	PF_InData*   RESTRICT in_data,
	PF_OutData*  RESTRICT out_data,
	PF_ParamDef* RESTRICT params[],
	PF_LayerDef* RESTRICT output
) ;

PF_Err
ProcessImgInAE
(
	PF_InData*	 RESTRICT in_data,
	PF_OutData*	 RESTRICT out_data,
	PF_ParamDef* RESTRICT params[],
	PF_LayerDef* RESTRICT output
) ;

PF_Err
ArtPaint_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
);


PF_Err
ArtPaint_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
);


const AlgoControls getControlsValues
(
    PF_ParamDef* RESTRICT params[]
) noexcept;


#endif /* __IMAGE_LAB_ART_PAINT_STYLE_FILTER__ */
