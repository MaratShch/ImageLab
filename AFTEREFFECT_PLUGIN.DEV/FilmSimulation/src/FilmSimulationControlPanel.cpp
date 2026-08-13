#include "Common.hpp"
#include "AlgoControlEnums.hpp"
#include "FilmSimulation.hpp"
#include "film_enum.hpp"

CACHE_ALIGN constexpr char filmList[] = {
    #include "film_names.txt"
};



PF_Err SetupControlElements (PF_InData* in_data, PF_OutData* out_data)
{
    CACHE_ALIGN PF_ParamDef	def;

    constexpr PF_ParamFlags   flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
    constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;
    constexpr PF_ParamUIFlags ui_disabled_flags = ui_flags | PF_PUI_DISABLED;

    constexpr char ButtonTitle[] = "ImageLab2 Interface";

    A_long totalParams = 0;

    // Setup Button
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_BUTTON(
        itemNames[0],
        ButtonTitle,
        ui_flags,
        flags,
        UnderlyingType(FilmSimulationCtrl::SETUP_BUTTON)
    );
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: FILM PROPERTIES                                  //
    ///////////////////////////////////////////////////////////////////
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_TOPICX(
        itemNames[1],
        ui_disabled_flags,
        UnderlyingType(FilmSimulationCtrl::GROUP_START_FILM_PROPERTIES));
    totalParams++;

    // Films LixtBox
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        itemNames[2],
        UnderlyingType(film::eFILM_PROFILE::eTOTAL_FILMS_PROFILES),
        UnderlyingType(film::eFILM_PROFILE::eAGFACOLOR_NEG_TYPE_B_1943),
        filmList,
        UnderlyingType(FilmSimulationCtrl::SETUP_BUTTON));
    totalParams++;

    AEFX_CLR_STRUCT_EX(def);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_FILM_PROPERTIES));
    totalParams++;

    out_data->num_params = totalParams;

    return PF_Err_NONE;
}