#include "AE_Effect.h"
#include "AlgoControl.hpp"
#include "AlgoControlAdobe.hpp"
#include "AlgoControlEnums.hpp"
#include "AlgoAdobeControlEnums.hpp"
#include "film_params_mask.hpp"
#include "CompileTimeUtils.hpp"
#include "CommonAdobeAE.hpp"


static void on_film_stock_changes (PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[]) noexcept
{
    const film::eFILM_PROFILE filmProfile = get_list_box_value<film::eFILM_PROFILE>(params, FilmSimulationCtrl::FILM_STOCK);
    const uint64_t filmMask = film::kFilmControlAvailability[UnderlyingType(filmProfile)];

    const auto site = AEFX_SuiteScoper<PF_ParamUtilsSuite3> (in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);

    set_control_status (params[UnderlyingType(FilmSimulationCtrl::FILM_FORMAT)]->ui_flags,     is_control_available(film::eCTRL_BIT_FILM_FORMAT, filmMask));
    site->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(FilmSimulationCtrl::FILM_FORMAT), params[UnderlyingType(FilmSimulationCtrl::FILM_FORMAT)]);

    set_control_status (params[UnderlyingType(FilmSimulationCtrl::PROCESS_VARIANT)]->ui_flags, is_control_available(film::eCTRL_BIT_PROCESS_VARIANT, filmMask));
    site->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(FilmSimulationCtrl::PROCESS_VARIANT), params[UnderlyingType(FilmSimulationCtrl::PROCESS_VARIANT)]);

    return;
}


PF_Err user_update_params_handler
(
    PF_InData						*in_data,
    PF_OutData						*out_data,
    PF_ParamDef						*params[],
    PF_LayerDef						*outputP,
    const PF_UserChangedParamExtra	*which_hitP
)
{
    switch (which_hitP->param_index)
    {
        case UnderlyingType(FilmSimulationCtrl::FILM_STOCK):
            on_film_stock_changes(in_data, out_data, params);
        break;

        default:
        break;
    }

    return PF_Err_NONE;
}


PF_Err user_update_params_ui
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_ParamDef			*params[],
    PF_LayerDef			*outputP
)
{
    return PF_Err_NONE;
}



AlgoControls getAlgoControls (PF_ParamDef* params[], const double fps, const int32_t idx)
{
    CACHE_ALIGN AlgoControls algoParams = getAlgoControlsDefault();

    algoParams.filmProfile = get_list_box_value<film::eFILM_PROFILE>(params, FilmSimulationCtrl::FILM_STOCK);
    algoParams.frameRate = ((fps < 1.0) ? 24.0 : fps);
    algoParams.frameIndex = idx;

    const uint64_t filmMask = film::kFilmControlAvailability[UnderlyingType(algoParams.filmProfile)];

    if (is_control_available(film::eCTRL_BIT_FILM_FORMAT, filmMask))
        algoParams.filmFormat = get_list_box_value<FilmFormatCtrl>(params, FilmSimulationCtrl::FILM_FORMAT);
    if (is_control_available(film::eCTRL_BIT_PROCESS_VARIANT, filmMask))
        algoParams.processVariant = get_list_box_value<ProcessVariantCtrl>(params, FilmSimulationCtrl::PROCESS_VARIANT);
    if (is_control_available(film::eCTRL_BIT_EXPOSURE_STOPS, filmMask))
        algoParams.exposureStops = get_float_slider_value(params, FilmSimulationCtrl::EXPOSURE);
    if (is_control_available(film::eCTRL_BIT_EXPOSURE_TIME_S, filmMask))
        algoParams.exposureTimeS = get_float_slider_value(params, FilmSimulationCtrl::EXPOSURE_TIME);
    if (is_control_available(film::eCTRL_BIT_GREY_TARGET, filmMask))
        algoParams.greyTarget = get_float_slider_value(params, FilmSimulationCtrl::MID_GRAY_TARGET);
    if (is_control_available(film::eCTRL_BIT_BLACK_POINT_STRETCH, filmMask))
        algoParams.blackPointStretch = get_float_slider_value(params, FilmSimulationCtrl::BLACK_POINT_STRETCH);
    if (is_control_available(film::eCTRL_BIT_DEVELOPMENT_MINUTES, filmMask))
        algoParams.developmentMinutes = get_float_slider_value(params, FilmSimulationCtrl::DEVELOPMENT_TIME);
    if (is_control_available(film::eCTRL_BIT_DEVELOPMENT_CELSIUS, filmMask))
        algoParams.developmentCelsius = get_float_slider_value(params, FilmSimulationCtrl::DEVELOPMENT_TEMPERATURE);
    if (is_control_available(film::eCTRL_BIT_STORAGE_YEARS, filmMask))
        algoParams.storageYears = get_slider_value(params, FilmSimulationCtrl::YEARS_OF_DARK_STORAGE);
    if (is_control_available(film::eCTRL_BIT_STORAGE_CELSIUS, filmMask))
        algoParams.storageCelsius = get_slider_value(params, FilmSimulationCtrl::STORAGE_TEMPERATURE);   // FIX: was storageYears

    if (is_control_available(film::eCTRL_BIT_SCENE_KELVIN, filmMask))
        algoParams.sceneKelvin = get_slider_value(params, FilmSimulationCtrl::SCENE_COLOUR_TEMPERATURE);
    if (is_control_available(film::eCTRL_BIT_PRINT_STOCK, filmMask))
        algoParams.printStock = get_list_box_value<PrintStockCtrl>(params, FilmSimulationCtrl::PRINT_STOCK);
    if (is_control_available(film::eCTRL_BIT_GENERATIONS, filmMask))
        algoParams.generations = static_cast<int32_t>(get_slider_value(params, FilmSimulationCtrl::DUPLICATION_GENERATION));
    if (is_control_available(film::eCTRL_BIT_DUPE_STOCK, filmMask))
    {
        const DupeStockCtrl dupe = get_list_box_value<DupeStockCtrl>(params, FilmSimulationCtrl::INTERMEDIATE_STOCK);
        algoParams.dupeStock = ((dupe == DupeStockCtrl::eDUPE_FINE_GRAIN) || (dupe == DupeStockCtrl::eKODAK_VISION3_DI_2254))
            ? dupe
            : DupeStockCtrlDef;   // eDUPE_FINE_GRAIN
    }
    if (is_control_available(film::eCTRL_BIT_PRINT_GRAIN, filmMask))
        algoParams.printGrain = get_check_box_value(params, FilmSimulationCtrl::PRINT_GRAIN);

    return algoParams;
}