#pragma once

#include "CommonAdobeAE.hpp"

typedef enum {
	eCIELAB_INPUT,
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

constexpr char strLcoarse[] = "L level coarse";
constexpr char strLfine  [] = "L level fine";
constexpr char strAcoarse[] = "A level coarse";
constexpr char strAfine  [] = "A level fine";
constexpr char strBcoarse[] = "B level coarse";
constexpr char strBfine  [] = "B level fine";
