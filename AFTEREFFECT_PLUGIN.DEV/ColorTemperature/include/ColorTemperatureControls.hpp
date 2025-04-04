#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CONTROLS_ITEMS__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CONTROLS_ITEMS__

#include <cstdint>

constexpr char controlItemName[][32] = {
	"Using Preset",					// check box
	"Preset Type",					// popup
	"Observer",						// popup
	"Illuminant",					// popup
	"Wavelength Step",				// popup
	"Color Temperature. °K x 1000", // slider
	"Temperature Offset",			// slider
	"Tint coarse",					// slider
	"Tint fine",					// slider
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
	"CIE 1931 2-Degree  Standard   |" // default observer
	"CIE 1964 10-Degree Standard   |"
	"CIE 1964 2-Degree Supplemental"
	// CIE 2006 Physiologically LMS
	// CIE 2017 Physiologically Cone
};

constexpr char controlItemIlluminant[] = {
	"Daylight (D65)         |" // default illuminant
	"Cloudy (D65 with tint) |"
	"Tungsten (Incandescent)|"
	"Fluorescent            |"
	"Warm White Fluorescent |"
	"Incandescent (Tungsten)|"
	"Warm White Fluorescent |"
	"Soft White Fluorescent |"
	"Moonlight (Cool Blue)  "
};

constexpr char controlItemWavelengthStep[] = {
	"Worst  - 5.0nm|" // 5.0 nanometers
	"Desent - 2.0nm|" // 2.0 nanometers -> default step
	"Fine   - 1.0nm|" // 1.0 nanometers
	"Finest - 0.5nm " // 0.5 nanometers
};

constexpr char controlItemCameraSPD [] = "SPD File Load";
constexpr char controlItemLoadPreset[] = "Load File";
constexpr char controlItemSavePreset[] = "Save File";


#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_CONTROLS_ITEMS__