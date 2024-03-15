#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ENUMS_AND_DEFINES__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ENUMS_AND_DEFINES__

#include <cstdint>

constexpr double  algoColorTempScale  = 1000.0;
constexpr int32_t algoColorTempMin    = 1000;
constexpr int32_t algoColorTempMax    = 13000;
constexpr int32_t algoColorWhitePoint = 6500;

constexpr double algoColorTempFineMin = -50.0;
constexpr double algoColorTempFineMax = 50.0;
constexpr double algoColorTempFineDef = 0.0;

constexpr int32_t algoColorTintMin    = -10;
constexpr int32_t algoColorTintMax    = 10;
constexpr int32_t algoColorTintDefault= 0;

constexpr double waveLengthStart      = 360.0;
constexpr double waveLengthStop       = 830.0;
constexpr double wavelengthStepWorst  = 5.00;
constexpr double wavelengthStepDecent = 2.00;
constexpr double wavelengthStepFine   = 1.00;
constexpr double wavelengthStepFinest = 0.50;
constexpr double wavelengthStepScientific = 0.10;
constexpr double wavelengthStepDefault  = wavelengthStepDecent;


constexpr double colorTemperature2Slider  (const int32_t& val) noexcept { return static_cast<double>(val) / algoColorTempScale;  }
constexpr int32_t slider2ColorTemperature (const double& val)  noexcept { return static_cast<int32_t>(val * algoColorTempScale); }

constexpr char controlName[][32] = {
	"Standard",
	"Gamma",
	"Color Temperature. °K x 1000",
	"Fine Temperature Offset",
	"Tint"
};

constexpr char strStandardName[] = {
	"Black Body|"
	"DayLight  |"
	"TM-30      "
};

constexpr char strGammaValueName[] = {
	"1.0   |"
	"1.8   |"
	"2.2   |"
	"sRGB  |"
	"sRGB-L|"
	"P3     "
};

typedef enum {
	eTEMP_STANDARD_BLACK_BODY,
	eTEMP_STANDARD_DAYLIGHT,
	eTEMP_STANDARD_TM30,
	eTEMP_STANDARD_TOTAL
}eTemperarueStandard;

typedef enum {
	eTEMP_GAMMA_VALUE_10,
	eTEMP_GAMMA_VALUE_18,
	eTEMP_GAMMA_VALUE_22,
	eTEMP_GAMMA_VALUE_SRGB,
	eTEMP_GAMMA_VALUE_SRGBL,
	eTEMP_GAMMA_VALUE_P3,
	eTEMP_GAMMA_VALUE_TOTAL
}eTemperatureGammaCorrection;


typedef enum {
	COLOR_TEMPERATURE_FILTER_INPUT,
	COLOR_TEMPERATURE_STANDARD_POPUP,
	COLOR_TEMPERATURE_GAMMA_POPUP,
	COLOR_TEMPERATURE_VALUE_SLIDER,
	COLOR_TEMPERATURE_FINE_VALUE_SLIDER,
	COLOR_TEMPERATURE_TINT_SLIDER,
	COLOR_TEMPERATURE_TOTAL_CONTROLS
}Item;

#endif /* __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ENUMS_AND_DEFINES__ */