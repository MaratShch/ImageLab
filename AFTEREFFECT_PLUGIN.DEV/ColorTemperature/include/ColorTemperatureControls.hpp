#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CONTROLS_ITEMS__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CONTROLS_ITEMS__

#include <cstdint>

constexpr char controlItemName[][32] = {
	"Using Preset",					// check box
	"Preset Type",					// popup
	"Observer",						// popup
	"Color Temperature. °K x 1000", // slider
	"Temperature Offset",			// slider
	"Tint coarse",					// slider
	"Tint fine",					// slider
    "CCT Bar",                      // UI color bar
	"Camera SPD",					// button
	"Load Preset",					// button
	"Save Preset"					// button
};

constexpr char controlItemPresetType[] = {
	"Landscape     |" // default preset
	"Nature        |"
	"Macro         |"
	"Street        |"
	"Portraits     |"
	"Nude body     |"
	"Food          |"
	"Painting      |"
	"Nigh and Astro"
};

constexpr char controlItemObserver[] = {
	"CIE 1931 2-Degree  Standard|" // default observer
	"CIE 1964 10-Degree Standard"
//	"CIE 1964 2-Degree Supplemental"
	// CIE 2006 Physiologically LMS
	// CIE 2017 Physiologically Cone
};

constexpr char controlItemCameraSPD [] = "SPD File Load";
constexpr char controlItemLoadPreset[] = "Load File";
constexpr char controlItemSavePreset[] = "Save File";


#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CONTROLS_ITEMS__