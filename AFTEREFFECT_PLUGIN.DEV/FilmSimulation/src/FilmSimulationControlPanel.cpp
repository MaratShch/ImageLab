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
    // GROUP START: FILM STOCK                                       //
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
        UnderlyingType(film::eFILM_PROFILE::eAGFA_APX_25),
        filmList,
        UnderlyingType(FilmSimulationCtrl::FILM_STOCK));
    totalParams++;

    // Film Format ListBox
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        itemNames[3],
        UnderlyingType(FilmFormatCtrl::eFILM_FORMAT_TOTAL_FORMATS),
        UnderlyingType(FilmFormatCtrl::eFILM_FORMAT_SUPER_35),
        FilmFormatCtrlStr,
        UnderlyingType(FilmSimulationCtrl::FILM_FORMAT));
    totalParams++;

    // Film Process Variant ListBox
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_POPUP(
        itemNames[4],
        UnderlyingType(FilmProcessVariant::eFILM_PROCESS_TOTAL),
        UnderlyingType(FilmProcessVariant::eFILM_PROCESS_EI_1600),
        FilmProcessVariant,
        UnderlyingType(FilmSimulationCtrl::PROCESS_VARIANT));
    totalParams++;

    // Film Generation List Box
    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_END_TOPIC(UnderlyingType(FilmSimulationCtrl::GROUP_STOP_FILM_PROPERTIES));
    totalParams++;

    ///////////////////////////////////////////////////////////////////
    // GROUP START: EXPOSURE & TONE                                  //
    ///////////////////////////////////////////////////////////////////

    out_data->num_params = totalParams;

    return PF_Err_NONE;
}