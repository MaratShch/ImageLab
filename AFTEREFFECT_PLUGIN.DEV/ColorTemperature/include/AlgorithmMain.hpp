#ifndef __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORITHM_MAIN__
#define __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORITHM_MAIN__

#include <cstdint>
#include <array>
#include "LinearLut/LinearLut.hpp"
#include "cct_interface.hpp"
#include "AlgoMemHandler.hpp"
#include "AlgoFormatIngest.hpp"
#include "AlgoControl.hpp"

using LutLinear8bits  = std::array<double, LinLut_srgb_8bit_double::LINEARIZE_LUT_SRGB_8BIT_F64_SIZE>;
using LutLinear10bits = std::array<double, LinLut_srgb_10bit_double::LINEARIZE_LUT_SRGB_10BIT_F64_SIZE>;
using LutLinear16bits = std::array<double, LinLut_srgb_16bit_double::LINEARIZE_LUT_SRGB_16BIT_F64_SIZE>;

inline const LutLinear8bits& getLinerLut8Bits  (void) noexcept { return LinLut_srgb_8bit_double::LINEARIZE_LUT_SRGB_8BIT_F64; }
inline const LutLinear10bits& getLinerLut10Bits(void) noexcept { return LinLut_srgb_10bit_double::LINEARIZE_LUT_SRGB_10BIT_F64; }
inline const LutLinear16bits& getLinerLut16Bits(void) noexcept { return LinLut_srgb_16bit_double::LINEARIZE_LUT_SRGB_16BIT_F64; }

AlgoCCT::CctHandle<double>& get_cct_handler (void) noexcept;


void AlgorithMain
(
    const PF_PixelPtr RESTRICT srcImg,
          PF_PixelPtr RESTRICT dstImg,
    const MemHandler& memHndl,
    const AlgoControls& algoCtrl,
    const A_long sizeX,
    const A_long sizeY,
    const A_long srcPitch,
    const A_long dstPitch,
    const AlgoPrIngest::ePrPixelFormat fmt
);

#endif // __IMAGE_LAB_IMAGE_COLOR_TEMPERATURE_ALGORITHM_MAIN__