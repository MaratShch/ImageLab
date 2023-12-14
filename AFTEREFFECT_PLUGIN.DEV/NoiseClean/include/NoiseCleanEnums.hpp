#pragma once
#include "CompileTimeUtils.hpp"

typedef enum {
	eNOISE_CLEAN_INPUT,
	eNOISE_CLEAN_ALGO_POPUP,
	eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER,
	eNOISE_CLEAN_ANYSOTROPIC_DISPERSION,
	eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP,
	eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL,
	eNOISE_CLEAN_TOTAL_PARAMS
}eItem;


typedef enum
{
	eNOISE_CLEAN_NONE = 0,
	eNOISE_CLEAN_BILATERAL_LUMA,
	eNOISE_CLEAN_BILATERAL_RGB,
	eNOISE_CLEAN_PERONA_MALIK,
	eNOISE_CLEAN_ADVANCED_DENOISE,
	eNOISE_CLEAN_TOTAL_ALGOS
}eNOISE_CLEAN_TYPE;

constexpr static char strAlgoPopupName[] = "Denoise algorithm";
constexpr static char strAlgoTypes[] =
{
	"None|"
	"Bilateral|"
	"Bilateral RGB|"
	"Anisotropic Diffusion|"
	"Advanced Denoise"
};

/* Sliders names */
constexpr static char strWindowSlider1[] = "Window size";
constexpr static char strWindowSlider2[] = "Dispersion ";
constexpr static char strWindowSlider3[] = "Time step  ";
constexpr static char strWindowSlider4[] = "Noise level";

/* Bilateral filters properties */
constexpr int32_t cBilateralWindowMin = 3;
constexpr int32_t cBilateralWindowMax = 15;
constexpr int32_t cBilateralWindowDefault = cBilateralWindowMin;
constexpr int32_t cBilateralMaxRadius = HALF(cBilateralWindowMax);
constexpr float cBilateralSigma = 0.1f;
constexpr float cBilateralGaussSigma = 3.f;

/* Anysotropic Diffusion filters properties */
constexpr float cDispersionMin = 1.f;
constexpr float cDispersionMax = 12.f;
constexpr float cDispersionDefault = 2.f;

constexpr float cTimeStepMin = 0.1f;
constexpr float cTimeStepMax = 1.f;
constexpr float cTimeStepDefault = 0.5f;

constexpr float cNoiseLevelMin = 1;
constexpr float cNoiseLevelMax = 10;
constexpr float cNoiseLevelDefault = cNoiseLevelMin;


/* Compilation time check */
static_assert(IsOddValue(cBilateralWindowMin), "cBilateralWindowMin should be ODD");
static_assert(IsOddValue(cBilateralWindowMax), "cBilateralWindowMax should be ODD");