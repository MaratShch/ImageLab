#include "AlgoRules.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureAlgo.hpp"
#include <array>
#include <utility>

AdaptationMatrixT computeAdaptationMatrix
(
    AlgoCCT::CctHandleF32* cctHandle,
    eCOLOR_OBSERVER observer,
    eCctType cctValueType,
    const std::pair<AlgoProcT, AlgoProcT>& cct_duv_src,
    const std::pair<AlgoProcT, AlgoProcT>& cct_duv_dst
)
{
    AlgoProcT sX, sY, sZ; // original XYZ
    AlgoProcT dX, dY, dZ; // destination XYZ

    const AlgoProcT targetCct = (CCT_OFFSET == cctValueType ? 
        std::max(absColorTemperatureMin, std::min(absColorTemperatureMax, cct_duv_src.first + cct_duv_dst.first)) : cct_duv_dst.first);
    const AlgoProcT targetDuv = (CCT_OFFSET == cctValueType ? 
        std::max(static_cast<AlgoProcT>(algoColorTintMin), std::min(static_cast<AlgoProcT>(algoColorTintMax),cct_duv_src.second + cct_duv_dst.second)) : cct_duv_dst.second);

    const std::pair<AlgoProcT, AlgoProcT> uv_src = cctHandle->getPlanckianUV(cct_duv_src,    observer);
    const std::pair<AlgoProcT, AlgoProcT> uv_dst = cctHandle->getPlanckianUV(std::make_pair(targetCct, targetDuv), observer);

    uvToXYZ (uv_src, sX, sY, sZ);
    uvToXYZ (uv_dst, dX, dY, dZ);

    constexpr AlgoProcT zero{ 0 };
    const AdaptationMatrixT correctionMartrix
    {
        dX / sX, zero,     zero,
        zero,    dY / sY,  zero,
        zero,    zero,     dZ / sZ
    };

    constexpr AdaptationMatrixT rgb2xyz
    {
        static_cast<AlgoProcT>(0.4124564), static_cast<AlgoProcT>(0.3575761), static_cast<AlgoProcT>(0.1804375),
        static_cast<AlgoProcT>(0.2126729), static_cast<AlgoProcT>(0.7151522), static_cast<AlgoProcT>(0.0721750),
        static_cast<AlgoProcT>(0.0193339), static_cast<AlgoProcT>(0.1191920), static_cast<AlgoProcT>(0.9503041)
    };

    constexpr AdaptationMatrixT xyz2rgb
    {
        static_cast<AlgoProcT>(3.2404542),  static_cast<AlgoProcT>(-1.5371385), static_cast<AlgoProcT>(-0.4985314),
        static_cast<AlgoProcT>(-0.9692660), static_cast<AlgoProcT>(1.8760108),  static_cast<AlgoProcT>(0.0415560),
        static_cast<AlgoProcT>(0.0556434),  static_cast<AlgoProcT>(0.2040259),  static_cast<AlgoProcT>(1.0572252)
    };

    const auto matrixMpl = [&](const AdaptationMatrixT& A, const AdaptationMatrixT& B) noexcept -> AdaptationMatrixT
    {
        AdaptationMatrixT result{};
        for (int32_t row = 0; row < 3; ++row)
        {
            for (int32_t col = 0; col < 3; ++col)
            {
                result[row * 3 + col] =
                    A[row * 3 + 0] * B[0 * 3 + col] +
                    A[row * 3 + 1] * B[1 * 3 + col] +
                    A[row * 3 + 2] * B[2 * 3 + col];
            }
        }
        return result;
    };

    const AdaptationMatrixT tmpMatrix = matrixMpl (rgb2xyz, correctionMartrix);

    return matrixMpl (tmpMatrix, xyz2rgb);
}


