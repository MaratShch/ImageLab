#pragma once

typedef enum {
	eNOISE_CLEAN_INPUT,
	eNOISE_CLEAN_ALGO_POPUP,
	eNOISE_CLEAN_BILATERAL_WINDOW_SLIDER,
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
	"Perona Malik|"
	"Advanced Denoise"
};

constexpr static char strWindowSlider[] = "Bilateral window size";
constexpr int32_t cBilateralWindowMin = 3;
constexpr int32_t cBilateralWindowMax = 13;
constexpr int32_t cBilateralWindowDefault = cBilateralWindowMin;
constexpr int32_t cBilateralMaxRadius = cBilateralWindowMax >> 1;
constexpr float cBilateralSigma = 0.1f;
constexpr float cBilateralGaussSigma = 3.f;