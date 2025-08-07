#include "AlgoRules.hpp"
#include "ColorTemperatureAlgo.hpp"
#include <array>
#include <utility>

AdaptationMatrixT computeAdaptationMatrix
(
    AlgoCCT::CctHandleF32* cctHandle,
    eCOLOR_OBSERVER observer,
    const std::pair<AlgoProcT, AlgoProcT>& cct_duv_src,
    const std::pair<AlgoProcT, AlgoProcT>& cct_duv_dst
)
{
    const std::pair<AlgoProcT, AlgoProcT> uv_src = cctHandle->getPlanckianUV(cct_duv_src, observer);
    const std::pair<AlgoProcT, AlgoProcT> uv_dst = cctHandle->getPlanckianUV(cct_duv_dst, observer);

    const float deltaU = uv_src.first  - uv_dst.first;
    const float deltaV = uv_src.second - uv_dst.second;


    std::array<AlgoProcT, 9> correctionMartrix {};
    return correctionMartrix;
}


