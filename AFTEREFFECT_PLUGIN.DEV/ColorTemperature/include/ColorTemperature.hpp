#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_FILTER__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_FILTER__

#include "AlgoRules.hpp"
#include "CommonAdobeAE.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureControlsPresets.hpp"


constexpr char strName[] = "Color Temperature";
constexpr char strCopyright[] = "\n2019-2025. ImageLab2 Copyright(c).\rColor Temperature plugin.";
constexpr int ColorTemperature_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ColorTemperature_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ColorTemperature_VersionSub = 0;
#ifdef _DEBUG
constexpr int ColorTemperature_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ColorTemperature_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int ColorTemperature_VersionBuild = 1;


PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
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
ColorTemperarture_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
);

PF_Err
ColorTemperature_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
);


typedef struct strControlSet
{
    AlgoProcT Cct;
    AlgoProcT Duv;
    sTemperaturePoint cctPoint;
    eObservers observer;
}strControlSet;

constexpr size_t strControlSetSize = sizeof(strControlSet);
const strControlSet GetCctSetup(PF_ParamDef *params[]);

#endif /* __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_FILTER__ */
