#include "ColorTemperature.hpp"
#include "AlgorithmMain.hpp"
#include "color_utils.hpp"

// Create CCTHandle
static AlgoCCT::CctHandle<double> cctHdnl;

AlgoCCT::CctHandle<double>& get_cct_handler(void) noexcept { return cctHdnl; }

// sRGB (IEC 61966-2-1) linear RGB -> XYZ, D65 white (x=0.3127, y=0.3290).
// Derived exactly from the spec primaries/white in 50-digit arithmetic;
// literals are the correctly-rounded double of the exact values.
// Row sums equal D65 XYZ exactly: RGB(1,1,1) -> the white point.
// NOTE: identical to the ITU-R BT.709 RGB->XYZ matrix (same primaries+white).
CACHE_ALIGN constexpr double sRGBtoXYZ_f64[9] =
{
    0.412390799265959510, 0.35758433938387796,  0.180480788401834290,
    0.212639005871510360, 0.71516867876775592,  0.072192315360733714,
    0.019330818715591849, 0.11919477979462599,  0.950532152249660590,
};

// Exact inverse (XYZ -> linear sRGB), same derivation; M * M^-1 = I to 1e-49
// in the 50-digit check — both directions from ONE source, so round-trips
// close exactly (this is the lesson from the CMCCAT2000 episode: derive the
// inverse from the same exact forward, never from a separately-rounded copy).
CACHE_ALIGN constexpr double XYZtosRGB_f64[9] =
{
     3.240969941904521300, -1.53738317757009350, -0.498610760293003270,
    -0.969243636280879840,  1.87596750150772060,  0.041555057407175612,
     0.055630079696993608, -0.20397695888897657,  1.056971514242878600,
};


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

    double cct_computed = 0.0;
    double duv_computed = 0.0;

    superpixel_to_cct (super, cctHdnl, sRGBtoXYZ_f64, observer_CIE_1931, cct_computed, duv_computed);

    return;
}