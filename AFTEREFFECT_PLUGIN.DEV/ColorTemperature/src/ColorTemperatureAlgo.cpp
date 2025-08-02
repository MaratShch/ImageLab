#include "AlgoRules.hpp"
#include "ColorTemperatureAlgo.hpp"
#include <array>
#include <utility>

std::array<AlgoProcT, 9> computeAdaptationMatrix
(
    AlgoCCT::CctHandleF32* cctHandle,
    const std::pair<AlgoProcT, AlgoProcT>& cct_duv_src,
    const std::pair<AlgoProcT, AlgoProcT>& cct_duv_dst
)
{
    const std::array<AlgoProcT, 9> correctionMartrix {};
    return correctionMartrix;
}


