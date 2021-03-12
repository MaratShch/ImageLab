#pragma once

#include "CommonAdobeAE.hpp"
#include "ColorCorrectionEnums.hpp"

constexpr char strName[] = "Color Correction HSL/V/I/P";
constexpr char strCopyright[] = "\n2019-2021. ImageLab2 Copyright(c).\rColor Correction in HSL/V/I/P color space.";
constexpr int ColorCorrection_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int ColorCorrection_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int ColorCorrection_VersionSub   = 0;
#ifdef _DEBUG
constexpr int ColorCorrection_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int ColorCorrection_VersionStage = PF_Stage_DEVELOP;// PF_Stage_RELEASE;
#endif
constexpr int ColorCorrection_VersionBuild = 1;

typedef struct _strHslRecord
{
	uint32_t	ver;
	uint32_t	subVer;
	uint32_t	sizeOf;
	char        name[32];
	uint32_t	domain;
	double		hue_coarse;
	double		hue_fine;
	double		sat_coarse;
	double		sat_file;
	double		l_coarse;
	double		l_fine;
} strHslRecord;

constexpr uint32_t strHslRecorsSizeof = sizeof(strHslRecord);

constexpr char ColorSpaceType[] = "Color Space";
constexpr char ColorSpace[] = "HSL|HSV|HSI|HSP|HSLuv|HPLuv";

constexpr char ColorHueCoarseType[] = "Hue coarse level";
constexpr char ColorHueFineLevel[] = "Hue fine level";
constexpr int32_t hue_coarse_min = -359;
constexpr int32_t hue_coarse_max = 359;
constexpr int32_t hue_coarse_default = 0;
constexpr int32_t hue_fine_min_level = -10;
constexpr int32_t hue_fine_max_level = 10;
constexpr int32_t hue_fine_def_level = 0;

constexpr char ColorSaturationCoarseLevel[]  = "Saturation coarse level";
constexpr char ColorSaturationFineLevel[] = "Saturation fine level";
constexpr int32_t sat_coarse_min_level = -100;
constexpr int32_t sat_coarse_max_level = 100;
constexpr int32_t sat_coarse_def_level = 0;
constexpr int32_t sat_fine_min_level = -10;
constexpr int32_t sat_fine_max_level = 10;
constexpr int32_t sat_fine_def_level = 0;

constexpr char ColorLWIPCoarseLevel[] = "L/W/I/P coarse level";
constexpr char ColorLWIPFineLevel[] = "L/W/I/P fine level";
constexpr int32_t lwip_coarse_min_level = -100;
constexpr int32_t lwip_coarse_max_level = 100;
constexpr int32_t lwip_coarse_def_level = 0;
constexpr int32_t lwip_fine_min_level = -10;
constexpr int32_t lwip_fine_max_level = 10;
constexpr int32_t lwip_fine_def_level = 0;

constexpr char LoadSettingName[] = "Load Setting";
constexpr char LoadSetting[] = "Load";
constexpr char SaveSettingName[] = "Save Setting";
constexpr char SaveSetting[] = "Save";
constexpr char ResetSettingName[] = "Reset Setting";
constexpr char ResetSetting[] = "Reset";


template<typename T>
inline const T CLAMP_H(const T hue)
{
	constexpr T hueMin{ 0 };
	constexpr T hueMax{ 360 };

	if (hue < hueMin)
		return (hue + hueMax);
	else if (hue >= hueMax)
		return (hue - hueMax);
	return hue;
}

template<typename T>
inline const T CLAMP_LS(const T ls)
{
	constexpr T lsMin{ 0 };
	constexpr T lsMax{ 100 };
	return MAX_VALUE(lsMin, MIN_VALUE(lsMax, ls));
}


inline const float normalize_hue_wheel(const float wheel_value) noexcept
{
	const float tmp = wheel_value / 360.0f;
	const int intPart = static_cast<int>(tmp);
	return (tmp - static_cast<float>(intPart)) * 360.0f;
}


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

PF_Err ProcessImgInAE_8bits
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err ProcessImgInAE_16bits
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept;

PF_Err
ProcessImgInPR
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept;

/* PR API's prototypes */
PF_Err prProcessImage_BGRA_4444_8u_HSL
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           hue,
	float           sat,
	float           lum
) noexcept;

PF_Err prProcessImage_BGRA_4444_8u_HSI
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_lum
) noexcept;

PF_Err prProcessImage_BGRA_4444_8u_HSP
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_per
) noexcept;

PF_Err prProcessImage_BGRA_4444_8u_HSLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_per
) noexcept;

PF_Err prProcessImage_BGRA_4444_8u_HPLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_p,
	float           add_luv
) noexcept;

PF_Err prProcessImage_BGRA_4444_16u_HSL
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_lum
) noexcept;

PF_Err prProcessImage_BGRA_4444_16u_HSI
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_int
) noexcept;

PF_Err prProcessImage_BGRA_4444_16u_HSP
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_per
) noexcept;

PF_Err prProcessImage_BGRA_4444_16u_HSLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_luv
) noexcept;

PF_Err prProcessImage_BGRA_4444_16u_HPLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_p,
	float           add_luv
) noexcept;

PF_Err prProcessImage_BGRA_4444_32f_HSL
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_lum
) noexcept;

PF_Err prProcessImage_BGRA_4444_32f_HSI
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_int
) noexcept;

PF_Err prProcessImage_BGRA_4444_32f_HSP
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_per
) noexcept;

PF_Err prProcessImage_BGRA_4444_32f_HSLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_luv
) noexcept;

PF_Err prProcessImage_BGRA_4444_32f_HPLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_p,
	float           add_luv
) noexcept;


PF_Err prProcessImage_RGB_444_10u_HSL
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_lum
) noexcept;

PF_Err prProcessImage_RGB_444_10u_HSV
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_val
) noexcept;

PF_Err prProcessImage_RGB_444_10u_HSI
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_int
) noexcept;

PF_Err prProcessImage_RGB_444_10u_HSP
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_per
) noexcept;


PF_Err prProcessImage_RGB_444_10u_HSLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_luv
) noexcept;

PF_Err prProcessImage_RGB_444_10u_HPLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_luv
) noexcept;

PF_Err prProcessImage_VUYA_4444_8u_HSL
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_lum,
	const bool&     isBT709 = true
) noexcept;

PF_Err prProcessImage_BGRA_4444_32f_HSI
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_int
) noexcept;


PF_Err prProcessImage_BGRA_4444_8u_HSV
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_val
) noexcept;

PF_Err prProcessImage_BGRA_4444_16u_HSV
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_val
) noexcept;

PF_Err prProcessImage_BGRA_4444_32f_HSV
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_val
) noexcept;	


PF_Err prProcessImage_VUYA_4444_32f_HSL
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_lum,
	const bool&     isBT709 = true
) noexcept;

PF_Err prProcessImage_VUYA_4444_32f_HSV
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_val,
	const bool&     isBT709 = true
) noexcept;

PF_Err prProcessImage_VUYA_4444_32f_HSI
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_val,
	const bool&     isBT709 = true
) noexcept;

PF_Err prProcessImage_VUYA_4444_32f_HSP
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_per,
	const bool&     isBT709 = true
) noexcept;

PF_Err prProcessImage_VUYA_4444_32f_HSLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_luv,
	const bool&     isBT709 = true
) noexcept;

PF_Err prProcessImage_VUYA_4444_32f_HPLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_p,
	float           add_luv,
	const bool&     isBT709
) noexcept;



PF_Err prProcessImage_VUYA_4444_8u_HSI
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_lum,
	const bool&     isBT709 = true
) noexcept;

PF_Err prProcessImage_VUYA_4444_8u_HSV
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_lum,
	const bool&     isBT709 = true
) noexcept;

PF_Err prProcessImage_VUYA_4444_8u_HSP
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_per,
	const bool&     isBT709 = true
) noexcept;

PF_Err prProcessImage_VUYA_4444_8u_HSLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_luv,
	const bool&     isBT709 = true
) noexcept;

PF_Err prProcessImage_VUYA_4444_8u_HPLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_p,
	float           add_luv,
	const bool&     isBT709 = true
) noexcept;

PF_Err prProcessImage_ARGB_4444_8u_HSL
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           hue,
	float           sat,
	float           lum
) noexcept;

PF_Err prProcessImage_ARGB_4444_8u_HSV
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           hue,
	float           sat,
	float           lum
) noexcept;

PF_Err prProcessImage_ARGB_4444_8u_HSI
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_lum
) noexcept;

PF_Err prProcessImage_ARGB_4444_8u_HSP
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_per
) noexcept;

PF_Err prProcessImage_ARGB_4444_8u_HSLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_per
) noexcept;

PF_Err prProcessImage_ARGB_4444_8u_HPLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_p,
	float           add_luv
) noexcept;

PF_Err prProcessImage_ARGB_4444_16u_HSL
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           hue,
	float           sat,
	float           lum
) noexcept;

PF_Err prProcessImage_ARGB_4444_16u_HSV
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           hue,
	float           sat,
	float           lum
) noexcept;

PF_Err prProcessImage_ARGB_4444_16u_HSI
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_lum
) noexcept;

PF_Err prProcessImage_ARGB_4444_16u_HSP
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_per
) noexcept;

PF_Err prProcessImage_ARGB_4444_16u_HSLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_sat,
	float           add_per
) noexcept;

PF_Err prProcessImage_ARGB_4444_16u_HPLuv
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output,
	float           add_hue,
	float           add_p,
	float           add_luv
) noexcept;
