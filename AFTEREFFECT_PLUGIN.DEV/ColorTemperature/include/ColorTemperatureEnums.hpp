#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ENUMS_AND_DEFINES__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ENUMS_AND_DEFINES__

#include <cstdint>
#include "ColorTemperatureControls.hpp"

constexpr double  algoColorTempScale  = 1000.0;
constexpr int32_t algoColorTempMin    = 1000;
constexpr int32_t algoColorTempMax    = 25000;
constexpr int32_t algoColorWhitePoint = 6500;

using WaveLengthT = double;

constexpr double algoColorTempFineMin = -50.0;
constexpr double algoColorTempFineMax = 50.0;
constexpr double algoColorTempFineDef = 0.0;

constexpr double algoColorTintMin     = -50.0;
constexpr double algoColorTintMax     =  50.0;
constexpr double algoColorTintDefault =  0.0;

constexpr double algoColorTintFineMin = -2.0;
constexpr double algoColorTintFineMax = 2.0;
constexpr double algoColorTintFineDefault = 0.0;

constexpr WaveLengthT waveLengthStart      = 380.0;
constexpr WaveLengthT waveLengthStop       = 750.0;
constexpr WaveLengthT wavelengthStepWorst  = 5.00;
constexpr WaveLengthT wavelengthStepDecent = 2.00;
constexpr WaveLengthT wavelengthStepFine   = 1.00;
constexpr WaveLengthT wavelengthStepFinest = 0.50;
constexpr WaveLengthT wavelengthStepScientific = 0.10;
constexpr WaveLengthT wavelengthStepDefault  = wavelengthStepDecent;

constexpr size_t waveVectorSizeWorst  = static_cast<size_t>((waveLengthStop - waveLengthStart) / wavelengthStepWorst      + 1.0);
constexpr size_t waveVectorSizeDecent = static_cast<size_t>((waveLengthStop - waveLengthStart) / wavelengthStepDecent     + 1.0);
constexpr size_t waveVectorSizeFine   = static_cast<size_t>((waveLengthStop - waveLengthStart) / wavelengthStepFine       + 1.0);
constexpr size_t waveVectorSizeFinest = static_cast<size_t>((waveLengthStop - waveLengthStart) / wavelengthStepFinest     + 1.0);
constexpr size_t waveVectorSizeScient = static_cast<size_t>((waveLengthStop - waveLengthStart) / wavelengthStepScientific + 1.0);

constexpr double colorTemperature2Slider  (const int32_t& val) noexcept { return static_cast<double>(val) / algoColorTempScale;  }
constexpr int32_t slider2ColorTemperature (const double& val)  noexcept { return static_cast<int32_t>(val * algoColorTempScale); }


typedef enum {
	COLOR_TEMPERATURE_FILTER_INPUT,
	COLOR_TEMPERATURE_PRESET_CHECKBOX,
	COLOR_TEMPERATURE_PRESET_TYPE_POPUP,
	COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP,
	COLOR_TEMPERATURE_COARSE_VALUE_SLIDER,
	COLOR_TEMPERATURE_FINE_VALUE_SLIDER,
	COLOR_TEMPERATURE_TINT_SLIDER,
	COLOR_TEMPERATURE_TINT_FINE_SLIDER,
	COLOR_TEMPERATURE_CAMERA_SPD_BUTTON,
	COLOR_TEMPERATURE_LOAD_PRESET_BUTTON,
	COLOR_TEMPERATURE_SAVE_PRESET_BUTTON,
	COLOR_TEMPERATURE_TOTAL_CONTROLS
}eItem;

typedef enum {
	COLOR_TEMPERARTURE_PRESET_LANDSCAPE,
	COLOR_TEMPERARTURE_PRESET_NATURE,
	COLOR_TEMPERARTURE_PRESET_MACRO,
	COLOR_TEMPERARTURE_PRESET_STREET,
	COLOR_TEMPERARTURE_PRESET_PORTRAITS,
	COLOR_TEMPERARTURE_PRESET_NUDE,
	COLOR_TEMPERARTURE_PRESET_FOOD,
	COLOR_TEMPERARTURE_PRESET_PAINTING,
	COLOR_TEMPERARTURE_PRESET_NIGHT_AND_ASTRO,
	COLOR_TEMPERARTURE_TOTAL_PRESETS
}ePresetTypes;

typedef enum {
	COLOR_TEMPERATURE_OBSERVER_1931_2,
	COLOR_TEMPERATURE_OBSERVER_1964_10,
//	COLOR_TEMPERATURE_OBSERVER_1964_2,
	COLOR_TEMPERATURE_TOTAL_OBSERVERS
}eObservers;

typedef enum {
	COLOR_TEMPERATURE_ILLUMINANT_D65,
	COLOR_TEMPERATURE_ILLUMINANT_D65_CLOUDY,
	COLOR_TEMPERATURE_ILLUMINANT_TUNGSTEN,
	COLOR_TEMPERATURE_ILLUMINANT_FLUORESCENT,
	COLOR_TEMPERATURE_ILLUMINANT_WHITE_FLUORESCENT,
	COLOR_TEMPERATURE_ILLUMINANT_INCANDESCENT,
	COLOR_TEMPERATURE_ILLUMINANT_WARM_WHITE,
	COLOR_TEMPERATURE_ILLUMINANT_SOFT_WHITE,
	COLOR_TEMPERATURE_ILLUMINANT_MOONLIGHT,
	COLOR_TEMPERATURE_TOTAL_ILLUMINANTS
}eIlluminant;

typedef enum {
	COLOR_TEMPERATURE_WAVELENGTH_STEP_WORST,
	COLOR_TEMPERATURE_WAVELENGTH_STEP_DECENT,
	COLOR_TEMPERATURE_WAVELENGTH_STEP_FINE,
	COLOR_TEMPERATURE_WAVELENGTH_STEP_FINEST,
	COLOR_TEMPERATURE_WAVELENGTH_TOTAL_STEPS
}eWaveLenghthStep;

#endif /* __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ENUMS_AND_DEFINES__ */