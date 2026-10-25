#pragma once

#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgoControlEnums.hpp"
#include "AlgoAdobeControlEnums.hpp"
#include "film_params_mask.hpp"
#include <cassert>

// Readers for params[]. params[] is indexed in REGISTRATION ORDER, so
// FilmSimulationCtrl must list the controls in exactly the order
// SetupControlElements() adds them (checked there at setup).
// Each reader must match the control's TYPE: an int slider (PF_ADD_SLIDER) is
// read with get_slider_value, a float slider (PF_ADD_FLOAT_SLIDERX /
// PF_ADD_FLOAT_EXPONENTIAL_SLIDER) with get_float_slider_value. Reading a float
// slider through u.sd returns the low 32 bits of a double (0 for 1.0, 5500.0 ...).
// The asserts catch a mismatch in Debug builds.


template <typename T>
inline const T get_list_box_value (PF_ParamDef* params[], const FilmSimulationCtrl idx) noexcept
{
    assert(PF_Param_POPUP == params[UnderlyingType(idx)]->param_type);
    return static_cast<T>(params[UnderlyingType(idx)]->u.pd.value - 1);   // AE pop-ups are 1-based
}

inline const int32_t get_slider_value(PF_ParamDef* params[], const FilmSimulationCtrl idx) noexcept
{
    assert(PF_Param_SLIDER == params[UnderlyingType(idx)]->param_type);
    return static_cast<int32_t>(params[UnderlyingType(idx)]->u.sd.value);
}

inline const double get_float_slider_value(PF_ParamDef* params[], const FilmSimulationCtrl idx) noexcept
{
    assert(PF_Param_FLOAT_SLIDER == params[UnderlyingType(idx)]->param_type);
    return params[UnderlyingType(idx)]->u.fs_d.value;
}

inline bool get_check_box_value(PF_ParamDef* params[], const FilmSimulationCtrl idx) noexcept
{
    assert(PF_Param_CHECKBOX == params[UnderlyingType(idx)]->param_type);
    return (0 != params[UnderlyingType(idx)]->u.bd.value);
}

inline constexpr bool is_control_available(const film::eFILM_CONTROL_BIT bit, const uint64_t mask) noexcept
{
    return static_cast<bool>(0ull != ((1ull << UnderlyingType(bit)) & mask));
}


// Selected film stock, CLAMPED to the database. The pop-up index comes from a
// saved project and indexes film::kFilmControlAvailability; a project saved
// with a larger database (or a damaged one) must not read past the table.
inline film::eFILM_PROFILE get_film_profile(PF_ParamDef* params[]) noexcept
{
    static_assert(film::kFilmControlAvailability.size() == static_cast<size_t>(UnderlyingType(film::eFILM_PROFILE::eTOTAL_FILMS_PROFILES)),
                  "availability table and film enumeration disagree");
    constexpr int32_t lastFilm = static_cast<int32_t>(UnderlyingType(film::eFILM_PROFILE::eTOTAL_FILMS_PROFILES)) - 1;
    const int32_t idx = static_cast<int32_t>(params[UnderlyingType(FilmSimulationCtrl::FILM_STOCK)]->u.pd.value) - 1;   // AE pop-ups are 1-based
    return static_cast<film::eFILM_PROFILE>(idx < 0 ? 0 : (idx > lastFilm ? lastFilm : idx));
}
