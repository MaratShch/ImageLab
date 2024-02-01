#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ENUMS_AND_DEFINES__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ENUMS_AND_DEFINES__

#include <cstdint>

constexpr double  algoColorTempScale  = 1000.0;
constexpr int32_t algoColorTempMin    = 1000;
constexpr int32_t algoColorTempMax    = 15000;
constexpr int32_t algoColorWhitePoint = 6500;
constexpr int32_t algoColorTintMin    = -20;
constexpr int32_t algoColorTintMax    = 20;
constexpr int32_t algoColorTintDefault= 0;

constexpr double colorTemperature2Slider  (const int32_t& val) noexcept { return static_cast<double>(val) / algoColorTempScale;  }
constexpr int32_t slider2ColorTemperature (const double& val)  noexcept { return static_cast<int32_t>(val * algoColorTempScale); }

constexpr char controlName[][32] = {
	"Standard",
	"Gamma",
	"Color Temperature (°K x 1000)",
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
	COLOR_TEMPERATURE_TINT_SLIDER,
	COLOR_TEMPERATURE_TOTAL_CONTROLS
}Item;

/* ALGO- constants */
constexpr double cLightVelocity     = 2.99792458e+08;
constexpr double cPlanckConstant    = 6.62607015e-34;
constexpr double cBoltzmannConstant = 1.380649e-23;


#endif /* __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ENUMS_AND_DEFINES__ */