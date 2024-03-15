#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__

#include "CommonColorTemperature.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ColorTemperatureEnums.hpp"

using ALG_TYPE = double;
using CCT_TYPE = float;
using RGB_TYPE = float;

typedef struct rgbCoefficients {
	CCT_TYPE cct;	/* correlated color temperature	in Kelvins degree	*/
	CCT_TYPE tint;	/* tint value										*/
	RGB_TYPE r;		/* coefficcient for apply to R channel				*/
	RGB_TYPE g;		/* coefficients for apply to G channel				*/
	RGB_TYPE b;		/* coefficients for apply to B channel				*/
} rgbCoefficients;


bool rebuildColorCoefficients (rgbCoefficients& cctStruct) noexcept;

#endif /* __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__ */
