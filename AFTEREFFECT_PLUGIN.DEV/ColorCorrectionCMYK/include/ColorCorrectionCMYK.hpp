#pragma once

#include "CommonAdobeAE.hpp"

constexpr char strName[] = "Color Correction CMYK";
constexpr char strCopyright[] = "\n2019-2021. ImageLab2 Copyright(c).\rColor Correction in CMYK or RGB color space.";
constexpr int ColorCorrection_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ColorCorrection_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ColorCorrection_VersionSub   = 0;
#ifdef _DEBUG
constexpr int ColorCorrection_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ColorCorrection_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int ColorCorrection_VersionBuild = 1;

constexpr char ColorSpaceType[] = "Color Space";
constexpr char ColorSpace[] = "RGB|CMYK";

typedef enum eColorSpaceDomain
{
	eINVALID = -1,
	eRGB,
	eCMYK,
	eTOTAL_COLOR_DOMAINS
}eColorSpaceDomain;

constexpr char ColorSlider1[eTOTAL_COLOR_DOMAINS][32] =
{
	"Red coarse level",
	"Cian coarse level"
};
constexpr char ColorSlider2[eTOTAL_COLOR_DOMAINS][32] =
{
	"Red fine level",
	"Cian fine level"
};
constexpr char ColorSlider3[eTOTAL_COLOR_DOMAINS][32] =
{
	"Green coarse level",
	"Magenta coarse level"
};
constexpr char ColorSlider4[eTOTAL_COLOR_DOMAINS][32] =
{
	"Green fine level",
	"Magenta fine level"
};
constexpr char ColorSlider5[eTOTAL_COLOR_DOMAINS][32] =
{
	"Blue coarse level",
	"Yellow coarse level"
};
constexpr char ColorSlider6[eTOTAL_COLOR_DOMAINS][32] =
{
	"Blue fine level",
	"Yellow fine level"
};
constexpr char ColorSlider7[eTOTAL_COLOR_DOMAINS][32] =
{
	"N/A",
	"Black key coarse level"
};
constexpr char ColorSlider8[eTOTAL_COLOR_DOMAINS][32] =
{
	"N/A",
	"Black key fine level"
};


constexpr char LoadSettingName[] = "Load Setting";
constexpr char LoadSetting[] = "Load";
constexpr char SaveSettingName[] = "Save Setting";
constexpr char SaveSetting[] = "Save";
constexpr char ResetSettingName[] = "Reset Setting";
constexpr char ResetSetting[] = "Reset";


bool SaveCustomSetting (PF_ParamDef* params[]) noexcept;
bool LoadCustomSetting (PF_ParamDef* params[]) noexcept;


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
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept;

