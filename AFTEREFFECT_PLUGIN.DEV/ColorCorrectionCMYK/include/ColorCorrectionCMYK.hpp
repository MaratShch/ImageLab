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
constexpr char ColorSpace[] = "CMYK|RGB";

typedef enum eColorSpaceDomain
{
	eINVALID = -1,
	eCMYK,
	eRGB,
	eTOTAL_COLOR_DOMAINS
}eColorSpaceDomain;

constexpr int32_t coarse_min_level = -100;
constexpr int32_t coarse_max_level =  100;
constexpr int32_t coarse_def_level =  0;

constexpr int32_t fine_min_level = -10;
constexpr int32_t fine_max_level =  10;
constexpr int32_t fine_def_level =  0;

constexpr int32_t param_name_length = PF_MAX_EFFECT_PARAM_NAME_LEN + 1;

constexpr char ColorSlider1[eTOTAL_COLOR_DOMAINS][param_name_length] =
{
	"Cian coarse level",
	"Red coarse level"
};
constexpr char ColorSlider2[eTOTAL_COLOR_DOMAINS][param_name_length] =
{
	"Cian fine level",
	"Red fine level"
};
constexpr char ColorSlider3[eTOTAL_COLOR_DOMAINS][param_name_length] =
{
	"Magenta coarse level",
	"Green coarse level"
};
constexpr char ColorSlider4[eTOTAL_COLOR_DOMAINS][param_name_length] =
{
	"Magenta fine level",
	"Green fine level"
};
constexpr char ColorSlider5[eTOTAL_COLOR_DOMAINS][param_name_length] =
{
	"Yellow coarse level",
	"Blue coarse level"
};
constexpr char ColorSlider6[eTOTAL_COLOR_DOMAINS][param_name_length] =
{
	"Yellow fine level",
	"Blue fine level"
};
constexpr char ColorSlider7[eTOTAL_COLOR_DOMAINS][param_name_length] =
{
	"Black key coarse level",
	"N/A"
};
constexpr char ColorSlider8[eTOTAL_COLOR_DOMAINS][param_name_length] =
{
	"Black key fine level",
	"N/A"
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

PF_Err prProcessImage_BGRA_4444_8u_CMYK
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_c,
	float           add_m,
	float           add_y,
	float           add_k
) noexcept;

PF_Err prProcessImage_BGRA_4444_8u_RGB
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_r,
	float           add_g,
	float           add_b
) noexcept;

PF_Err prProcessImage_BGRA_4444_16u_CMYK
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_c,
	float           add_m,
	float           add_y,
	float           add_k
) noexcept;

PF_Err prProcessImage_BGRA_4444_16u_RGB
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_r,
	float           add_g,
	float           add_b
) noexcept;

PF_Err prProcessImage_BGRA_4444_32f_CMYK
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_c,
	float           add_m,
	float           add_y,
	float           add_k
) noexcept;

PF_Err prProcessImage_BGRA_4444_32f_RGB
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_r,
	float           add_g,
	float           add_b
) noexcept;

PF_Err prProcessImage_VUYA_4444_8u_CMYK
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_c,
	float           add_m,
	float           add_y,
	float           add_k,
	bool            isBT709
) noexcept;

PF_Err prProcessImage_VUYA_4444_8u_RGB
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_r,
	float           add_g,
	float           add_b,
	bool            isBT709
) noexcept;

PF_Err prProcessImage_VUYA_4444_32f_CMYK
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_c,
	float           add_m,
	float           add_y,
	float           add_k,
	bool            isBT709
) noexcept;

PF_Err prProcessImage_VUYA_4444_32f_RGB
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_r,
	float           add_g,
	float           add_b,
	bool            isBT709
) noexcept;
