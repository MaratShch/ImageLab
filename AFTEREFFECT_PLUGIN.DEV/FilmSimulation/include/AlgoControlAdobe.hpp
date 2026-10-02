#pragma once

#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoControlEnums.hpp"
#include "AlgoAdobeControlEnums.hpp"
#include "film_params_mask.hpp"


template <typename T>
inline const T get_list_box_value (PF_ParamDef* params[], const FilmSimulationCtrl idx) noexcept
{
    return static_cast<T>(params[UnderlyingType(idx)]->u.pd.value - 1);
}

inline const int32_t get_slider_value(PF_ParamDef* params[], const FilmSimulationCtrl idx) noexcept
{
    return static_cast<int32_t>(params[UnderlyingType(idx)]->u.sd.value);
}

inline const double get_float_slider_value(PF_ParamDef* params[], const FilmSimulationCtrl idx) noexcept
{
    return params[UnderlyingType(idx)]->u.fs_d.value;
}

inline constexpr bool get_check_box_value(PF_ParamDef* params[], const FilmSimulationCtrl idx) noexcept
{
    return (0 != params[UnderlyingType(idx)]->u.bd.value);
}


inline constexpr bool is_control_available(const film::eFILM_CONTROL_BIT bit, const uint64_t mask) noexcept
{
    // The entire calculation is done inside a single return statement
    return static_cast<bool>((1ull << UnderlyingType(bit)) & mask);
}

