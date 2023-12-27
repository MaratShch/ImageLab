#pragma once
#include "CompileTimeUtils.hpp"

typedef enum {
	eNOISE_CLEAN_INPUT,
	eNOISE_CLEAN_ALGO_POPUP,
	eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER,
	eNOISE_CLEAN_ANYSOTROPIC_DISPERSION,
	eNOISE_CLEAN_ANYSOTROPIC_TIMESTEP,
	eNOISE_CLEAN_ANYSOTROPIC_NOISELEVEL,
	eNOISE_CLEAN_NL_BAYES_SIGMA,
	eNOISE_CLEAN_TOTAL_PARAMS
}eItem;


typedef enum
{
	eNOISE_CLEAN_NONE = 0,
	eNOISE_CLEAN_BILATERAL_LUMA,
	eNOISE_CLEAN_BILATERAL_RGB,
	eNOISE_CLEAN_PERONA_MALIK,
	eNOISE_CLEAN_BSDE,
	eNOISE_CLEAN_NONLOCAL_BAYES,
	eNOISE_CLEAN_ADVANCED_DENOISE,
	eNOISE_CLEAN_TOTAL_ALGOS
}eNOISE_CLEAN_TYPE;

constexpr static char strAlgoPopupName[] = "Denoise algorithm";
constexpr static char strAlgoTypes[] =
{
	"None                 |"
	"Bilateral Luminance  |"
	"Bilateral Color      |"
	"Anisotropic Diffusion|"
	"BSDE                 |"
	"Non Local Bayes      |"
	"Advanced Denoise     "
};

/* Sliders names */
constexpr static char strWindowSlider1[] = "Window size    ";
constexpr static char strWindowSlider2[] = "Dispersion     ";
constexpr static char strWindowSlider3[] = "Time step      ";
constexpr static char strWindowSlider4[] = "Noise level    ";
constexpr static char strWindowSlider5[] = "Noise deviation";

/* Bilateral filters properties */
constexpr int32_t cBilateralWindowMin = 3;
constexpr int32_t cBilateralWindowMax = 15;
constexpr int32_t cBilateralWindowDefault = cBilateralWindowMin;
constexpr int32_t cBilateralMaxRadius = HALF(cBilateralWindowMax);
constexpr float cBilateralSigma = 0.1f;
constexpr float cBilateralGaussSigma = 3.f;

/* Anysotropic Diffusion filters properties */
constexpr int32_t cDispersionMin = 10;
constexpr int32_t cDispersionMax = 120;
constexpr int32_t cDispersionDefault = 20;

constexpr int32_t cTimeStepMin = 1;
constexpr int32_t cTimeStepMax = 10;
constexpr int32_t cTimeStepDefault = 5;

constexpr int32_t cNoiseLevelMin = 10;
constexpr int32_t cNoiseLevelMax = 100;
constexpr int32_t cNoiseLevelDefault = cNoiseLevelMin;


/* Compilation time check */
static_assert(IsOddValue(cBilateralWindowMin), "cBilateralWindowMin should be ODD");
static_assert(IsOddValue(cBilateralWindowMax), "cBilateralWindowMax should be ODD");

/* Non local Bayse filter parameters (noise standard deviation * 10) */
constexpr int32_t cNonLocalBayesNoiseStdMin = 50;
constexpr int32_t cNonLocalBayesNoiseStdMax = 800;
constexpr int32_t cNonLocalBayesNoiseStdDefault = 100;
