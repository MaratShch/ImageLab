#include "AE_Effect.h"
#include "AlgoControl.hpp"
#include "AlgoControlAdobe.hpp"
#include "AlgoControlEnums.hpp"
#include "AlgoAdobeControlEnums.hpp"
#include "film_params_mask.hpp"



AlgoControls getAlgoControls (PF_ParamDef* params[], const double fps, const int32_t idx)
{
    CACHE_ALIGN AlgoControls algoParams = getAlgoControlsDefault();

    algoParams.filmProfile = get_list_box_value<film::eFILM_PROFILE>(params, FilmSimulationCtrl::FILM_STOCK);
    algoParams.frameRate   = ((fps < 1) ? 24.0 : fps);
    algoParams.frameIndex = idx;

    const uint64_t filmMask = film::kFilmControlAvailability[UnderlyingType(algoParams.filmProfile)];

    if (is_control_available(film::eCTRL_BIT_FILM_FORMAT, filmMask))
        algoParams.filmFormat  = get_list_box_value<FilmFormatCtrl>(params, FilmSimulationCtrl::FILM_FORMAT);
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
    if (is_control_available(film::eCTRL_BIT_SCENE_KELVIN, filmMask))
        algoParams.sceneKelvin = get_slider_value(params, FilmSimulationCtrl::SCENE_COLOUR_TEMPERATURE);
    if (is_control_available(film::eCTRL_BIT_PRINT_STOCK, filmMask))
        algoParams.printStock = get_list_box_value<PrintStockCtrl>(params, FilmSimulationCtrl::PRINT_STOCK);
    if (is_control_available(film::eCTRL_BIT_GENERATIONS, filmMask))
        algoParams.dupeStock = get_list_box_value<DupeStockCtrl>(params, FilmSimulationCtrl::DUPLICATION_GENERATION);

    return algoParams;
}