#pragma once

#include "CommonAdobeAE.hpp"

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


constexpr char ColorSpaceType[] = "Color Space";
constexpr char ColorSpace[] = "HSL|HSV|HSI|HSP";

enum {
		COLOR_SPACE_HSL = 0,
		COLOR_SPACE_HSV,
		COLOR_SPACE_HSI,
		COLOR_SPACE_HSP,
		COLOR_SPACE_MAX_TYPES
};


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


enum {
		COLOR_CORRECT_INPUT,
		COLOR_CORRECT_SPACE_POPUP,
		COLOR_CORRECT_HUE_COARSE,
		COLOR_HUE_FINE_LEVEL_SLIDER,
		COLOR_SATURATION_COARSE_LEVEL_SLIDER,
		COLOR_SATURATION_FINE_LEVEL_SLIDER,
		COLOR_LWIP_COARSE_LEVEL_SLIDER,
		COLOR_LWIP_FINE_LEVEL_SLIDER,
		COLOR_LOAD_SETTING_BUTTON,
		COLOR_SAVE_SETTING_BUTTON,
		COLOR_RESET_SETTING_BUTTON,
		COLOR_CORRECT_TOTAL_PARAMS
};