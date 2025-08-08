#include <utility>
#include "AE_Effect.h"
#include "ColorTemperature.hpp"


const strControlSet GetCctSetup (PF_ParamDef *params[])
{
    strControlSet cctSetup{};

    if (0 != params[COLOR_TEMPERATURE_PRESET_CHECKBOX]->u.bd.value)
    {
        // Preset check box activated

    }
    else
    {
        // Preset check box disabled
        cctSetup.Cct = static_cast<float>(
            slider2ColorTemperature(params[COLOR_TEMPERATURE_COARSE_VALUE_SLIDER]->u.fs_d.value) +
            params[COLOR_TEMPERATURE_FINE_VALUE_SLIDER]->u.fs_d.value);

        cctSetup.Duv = static_cast<float>(params[COLOR_TEMPERATURE_TINT_SLIDER]->u.fs_d.value + params[COLOR_TEMPERATURE_TINT_FINE_SLIDER]->u.fs_d.value);

        cctSetup.observer = static_cast<eObservers>(params[COLOR_TEMPERATURE_OBSERVER_TYPE_POPUP]->u.pd.value - 1);
    }

    return cctSetup;
}
