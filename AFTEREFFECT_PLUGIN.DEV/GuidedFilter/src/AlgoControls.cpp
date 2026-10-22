#include <cstdint>
#include <algorithm>
#include "AlgoControls.hpp"
#include "Common.hpp"
#include "GuidedFilterEnum.hpp"
#include "AE_Effect.h"


template <typename T>
inline const T get_list_box_value(PF_ParamDef* params[], const CtrlItems idx) noexcept
{
    return static_cast<T>(params[UnderlyingType(idx)]->u.pd.value - 1);
}

inline const int32_t get_slider_value(PF_ParamDef* params[], const CtrlItems idx) noexcept
{
    return static_cast<int32_t>(params[UnderlyingType(idx)]->u.sd.value);
}

inline const double get_float_slider_value(PF_ParamDef* params[], const CtrlItems idx) noexcept
{
    return params[UnderlyingType(idx)]->u.fs_d.value;
}

inline constexpr bool get_check_box_value(PF_ParamDef* params[], const CtrlItems idx) noexcept
{
    return (0 != params[UnderlyingType(idx)]->u.bd.value);
}


const AlgoControls getAlgoControls (PF_ParamDef* params[])
{
    CACHE_ALIGN AlgoControls algoCtrl{};

    algoCtrl.filterMode = get_list_box_value<FilterMode>(params, CtrlItems::gFILTER_MODE);
    algoCtrl.radius     = get_slider_value(params, CtrlItems::gFILTER_RADIUS);
    algoCtrl.epsilon    = static_cast<float>(get_float_slider_value(params, CtrlItems::gFILTER_EPSILON));
    algoCtrl.showConfidenceMap = get_check_box_value(params, CtrlItems::gFILTER_SHOW_MAP);
    algoCtrl.skinTolerance = static_cast<float>(get_float_slider_value(params, CtrlItems::gFILTER_SKIN_TOLERANCE));
    algoCtrl.skinSoftness  = static_cast<float>(get_float_slider_value(params, CtrlItems::gFILTER_SKIN_SOFTNESS));

    return algoCtrl;
}