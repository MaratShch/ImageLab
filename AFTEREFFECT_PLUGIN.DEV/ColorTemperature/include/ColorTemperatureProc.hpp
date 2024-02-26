#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__

#include "CommonColorTemperature.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ColorTemperatureEnums.hpp"

typedef struct rgbCoefficients {
	float cct;	/* correlated color temperature	in Kelvins degree	*/
	float tint; /* tint value										*/							
	float r;	/* coefficcient for apply to R channel				*/
	float g;	/* coefficients for apply to G channel				*/
	float b;	/* coefficients for apply to B channel				*/
} rgbCoefficients;


#endif /* __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGO_PROC__ */
