#pragma once

#include "CommonAdobeAE.hpp"

typedef enum {
	eCIELAB_INPUT,
	eCIELAB_POPUP_OBSERVER,
	eCIELAB_POPUP_ILLUMINANT,
	eCIELAB_SLIDER_L_COARSE,
	eCIELAB_SLIDER_L_FINE,
	eCIELAB_SLIDER_A_COARSE,
	eCIELAB_SLIDER_A_FINE,
	eCIELAB_SLIDER_B_COARSE,
	eCIELAB_SLIDER_B_FINE,
	eCIELAB_TOTAL_PARAMS
}eItem;


constexpr int32_t L_coarse_min_level = -100;
constexpr int32_t L_coarse_max_level = 100;
constexpr int32_t L_coarse_def_level = 0;
constexpr int32_t AB_coarse_min_level = -128;
constexpr int32_t AB_coarse_max_level = 128;
constexpr int32_t AB_coarse_def_level = 0;

constexpr float fine_min_level = -5.f;
constexpr float fine_max_level = 5.f;
constexpr float fine_def_level = 0.f;

constexpr char strObserver[]   = "Observer";
constexpr char strObserverType[]  = "2°  (CIE 1931) |"
                                    "10° (CIE 1964) ";

constexpr char strIlluminant[] = "Illuminant";
constexpr char strIlluminanatVal[] =
								"A -   Incandescent             |"
								"B -   Old direct sunlight      |"
								"C -   Old daylight             |"
								"D50 - ICC profile PCS          |"
								"D55 - Mid-morning daylight     |"
								"D65 - Daylight                 |"
								"D75 - North sky daylight       |"
								"E   - Equal energy             |"
								"F1 - Daylight Fluorescent      |"
								"F2 - Cool Fluorescent          |"
								"F3 - White Fluorescent         |"
								"F4 - Warm White Fluorescent    |"
								"F5 - Daylight Fluorescent      |"
								"F6 - Lite White Fluorescent    |"
								"F7 - Daylight Fluorescent      |"
								"F8 - Sylvania F40              |"
								"F9 - Cool White Fluorescent    |"
								"F10- Ultralume 50, Philips TL85|"
								"F11- Ultralume 40, Philips TL84|"
								"F12- Ultralume 30, Philips TL83";


constexpr char strLcoarse[] = "L level coarse";
constexpr char strLfine  [] = "L level fine";
constexpr char strAcoarse[] = "A level coarse";
constexpr char strAfine  [] = "A level fine";
constexpr char strBcoarse[] = "B level coarse";
constexpr char strBfine  [] = "B level fine";
