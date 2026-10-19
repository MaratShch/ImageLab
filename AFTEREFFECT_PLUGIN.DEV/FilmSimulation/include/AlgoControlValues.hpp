#pragma once

#include <cstdint>
#include "AlgoControlEnums.hpp"

// ---------------------------------------------------------------------------
// Panel-side defaults for the controls whose ENGINE default is a sentinel that
// lies OUTSIDE the slider range (-1 = "use the stock's own value" / "as the
// stock was developed"). An After Effects slider cannot hold such a value, so
// the sentinel is selected by a separate check-box and the slider itself
// starts from the in-range value below. With the check-boxes at their
// defaults a fresh panel renders exactly as getAlgoControlsDefault().
// ---------------------------------------------------------------------------
constexpr double DevelopmentMinutesUiDef = 8.0;    // used only when Development Enable is ticked
constexpr double DevelopmentCelsiusUiDef = 20.0;   // used only when Development Enable is ticked
constexpr double FlareUiDef              = 0.0;    // used only when "Stock's Own" is unticked (0 = perfect lens)
constexpr double VignetteUiDef           = 0.0;    // used only when "Stock's Own" is unticked (0 = no falloff)

constexpr bool   DevelopmentEnableUiDef  = false;  // false = as the stock was developed
constexpr bool   FlareStockOwnUiDef      = true;   // true  = stock's own (engine -1)
constexpr bool   VignetteStockOwnUiDef   = true;   // true  = stock's own (engine -1)

static_assert(DevelopmentMinutesUiDef >= DevelopmentMinutesMin && DevelopmentMinutesUiDef <= DevelopmentMinutesMax, "DevelopmentMinutesUiDef out of range");
static_assert(DevelopmentCelsiusUiDef >= DevelopmentCelsiusMin && DevelopmentCelsiusUiDef <= DevelopmentCelsiusMax, "DevelopmentCelsiusUiDef out of range");
static_assert(FlareUiDef    >= FlareMin    && FlareUiDef    <= FlareMax,    "FlareUiDef out of range");
static_assert(VignetteUiDef >= VignetteMin && VignetteUiDef <= VignetteMax, "VignetteUiDef out of range");
