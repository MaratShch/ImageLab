#pragma once

typedef enum {
	COLOR_SPACE_INNVALID = -1,
	COLOR_SPACE_HSL = 0,
	COLOR_SPACE_HSV,
	COLOR_SPACE_HSI,
	COLOR_SPACE_HSP,
	COLOR_SPACE_HSLuv,
	COLOR_SPACE_HPLuv,
	COLOR_SPACE_MAX_TYPES
}eCOLOR_SPACE_TYPE;


typedef enum {
	COLOR_CORRECT_INPUT,
	COLOR_CORRECT_SPACE_POPUP,
	COLOR_CORRECT_HUE_COARSE_LEVEL,
	COLOR_HUE_FINE_LEVEL_SLIDER,
	COLOR_SATURATION_COARSE_LEVEL_SLIDER,
	COLOR_SATURATION_FINE_LEVEL_SLIDER,
	COLOR_LWIP_COARSE_LEVEL_SLIDER,
	COLOR_LWIP_FINE_LEVEL_SLIDER,
	COLOR_LOAD_SETTING_BUTTON,
	COLOR_SAVE_SETTING_BUTTON,
	COLOR_RESET_SETTING_BUTTON,
	COLOR_CORRECT_TOTAL_PARAMS
}Item;

