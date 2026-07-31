#include "AlgoControl.hpp"
#include "AlgoControlValues.hpp"
#include "AlgoControlEnums.hpp"
#include "FilmSimulation.hpp"


PF_Err
SetupControlElements
(
    const PF_InData*  in_data,
          PF_OutData* out_data
)
{
    CACHE_ALIGN PF_ParamDef	def{};
    PF_Err		err = PF_Err_NONE;

    constexpr PF_ParamFlags   flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
    constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;
    constexpr PF_ParamUIFlags ui_disabled_flags = ui_flags | PF_PUI_DISABLED;

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);

    return err;
}


const AlgoControls
GetControlElements
(
    PF_ParamDef* params[]
)
{
    AlgoControls algo{};

    return algo;
}