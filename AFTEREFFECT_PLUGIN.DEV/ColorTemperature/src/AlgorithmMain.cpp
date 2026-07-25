#include "ColorTemperature.hpp"
#include "AlgorithmMain.hpp"

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
)
{
    float* dstRGB_f32 = memHndl.input_f32_interleaved;
    SuperPixel<double> super{};   // computed in double, max accuracy

    const auto& lut8  = getLinerLut8Bits();
    const auto& lut10 = getLinerLut10Bits();
    const auto& lut16 = getLinerLut16Bits();

    ingest_and_superpixel (static_cast<const void*>(srcImg), sizeX, sizeY, srcPitch, fmt, lut8, lut16, lut10, dstRGB_f32, super);

    return;
}