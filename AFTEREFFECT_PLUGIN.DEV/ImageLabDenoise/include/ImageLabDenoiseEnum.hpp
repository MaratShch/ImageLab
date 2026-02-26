#ifndef __IMAGE_LAB_DENOISE_FILTER_ENUMERATORS__
#define __IMAGE_LAB_DENOISE_FILTER_ENUMERATORS__

#include <cstdint>
#include "AE_Effect.h"
#include "CompileTimeUtils.hpp"

enum class eDenoiseControl : int32_t
{
    eIMAGE_LAB_DENOISE_INPUT,
    eIMAGE_LAB_DENOISE_ACC_SANDARD,
    eIMAGE_LAB_DENOISE_AMOUNT,
    eIMAGE_LAB_DENOISE_LUMA_STRENGTH,
    eIMAGE_LAB_DENOISE_CHROMA_STRENGTH,
    eIMAGE_LAB_DENOISE_DETAILS_PRESERVATION,
    eIMAGE_LAB_DENOISE_COARSE_NOISE,
    eIMAGE_LAB_DENOISE_CONTROLS
};

constexpr char controlItemName[][24] =
{
    "Accurance",
    "Denoise Amount",
    "Luma Strength",
    "Chrome Strength",
    "Details Preservation",
    "Coarse Noise"
};

enum class eDenoiseMethod : int32_t
{
    eIMAGE_LAB_DENOISE_DRAFT,
    eIMAGE_LAB_DENOISE_STANDARD,
    eIMAGE_LAB_DENOISE_ACCURATE,
    eIMAGE_LAB_DENOISE_TOTAL
};

constexpr char eDenoiseMethodStr[] =
{
    "Draft|"
    "Standard|"
    "Accurate"
};

constexpr float MasterDenoiseAmountMin = 0.f;
constexpr float MasterDenoiseAmountMax = 3.f;
constexpr float MasterDenoiseAmountDef = 1.f;

constexpr float LumaStrengthMin = 0.f;
constexpr float LumaStrengthMax = 3.f;
constexpr float LumaStrengthDef = 1.f;

constexpr float ChromaStrengthMin = 0.f;
constexpr float ChromaStrengthMax = 3.f;
constexpr float ChromaStrengthDef = 1.f;

constexpr float DetailsPreservationMin = 0.f;
constexpr float DetailsPreservationMax = 2.f;
constexpr float DetailsPreservationDef = 1.f;

constexpr float CoarseNoiseMin = 0.f;
constexpr float CoarseNoiseMax = 2.f;
constexpr float CoarseNoiseDef = 1.f;

#endif // __IMAGE_LAB_DENOISE_FILTER_ENUMERATORS__