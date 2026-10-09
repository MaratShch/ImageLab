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

    if (is_control_available(film::eCTRL_BIT_GRAIN_SCALE, filmMask))
        algoParams.grainScale = get_slider_value(params, FilmSimulationCtrl::GRAIN);
    if (is_control_available(film::eCTRL_BIT_HALATION_SCALE, filmMask))
        algoParams.halationScale = get_slider_value(params, FilmSimulationCtrl::HALATION);
    if (is_control_available(film::eCTRL_BIT_COUPLER_SCALE, filmMask))
        algoParams.couplerScale = get_slider_value(params, FilmSimulationCtrl::DIR_COUPLERS);
    if (is_control_available(film::eCTRL_BIT_MISREG_SCALE, filmMask))
        algoParams.misregScale = get_slider_value(params, FilmSimulationCtrl::MISREGISTRATION);
    if (is_control_available(film::eCTRL_BIT_COATING_SCALE, filmMask))
        algoParams.coatingScale = get_slider_value(params, FilmSimulationCtrl::COATING_UNEVENNESS);
    if (is_control_available(film::eCTRL_BIT_RESEAU, filmMask))
        algoParams.reseau = get_check_box_value(params, FilmSimulationCtrl::RESEAU_RECONSTRUCTION);

    if (is_control_available(film::eCTRL_BIT_FLARE, filmMask))
        algoParams.flare = get_slider_value(params, FilmSimulationCtrl::VEILING_FLARE);
    if (is_control_available(film::eCTRL_BIT_VIGNETTE, filmMask))
        algoParams.vignette = get_slider_value(params, FilmSimulationCtrl::CORNER_FALLOFF);
    if (is_control_available(film::eCTRL_BIT_SCANNER_SPECULAR, filmMask))
        algoParams.scannerSpecular = get_slider_value(params, FilmSimulationCtrl::SCANNER_SPECULARITY);
    if (is_control_available(film::eCTRL_BIT_SCANNER_FIXED_PATTERN, filmMask))
        algoParams.scannerFixedPattern = get_slider_value(params, FilmSimulationCtrl::SCANNER_FIXED_PATTERN);

    // Film damages
    if (is_control_available(film::eCTRL_BIT_FILM_DAMAGE_ENABLED, filmMask))
        algoParams.filmDamageEnabled = get_check_box_value(params, FilmSimulationCtrl::ENABLE_FILM_DAMAGE);
    if (true == algoParams.filmDamageEnabled)
    {
        if (is_control_available(film::eCTRL_BIT_DAMAGE_STRENGTH, filmMask))
            algoParams.damage.damageStrength = get_slider_value(params, FilmSimulationCtrl::OVERALL_STRENGHT);
        if (is_control_available(film::eCTRL_BIT_DUST_LEVEL, filmMask))
            algoParams.damage.dustLevel = get_slider_value(params, FilmSimulationCtrl::DUST);
        if (is_control_available(film::eCTRL_BIT_DEBRIS_LEVEL, filmMask))
            algoParams.damage.debrisLevel = get_slider_value(params, FilmSimulationCtrl::DEBRIS);
        if (is_control_available(film::eCTRL_BIT_FIBRE_LEVEL, filmMask))
            algoParams.damage.fibreLevel = get_slider_value(params, FilmSimulationCtrl::FIBRES);
        if (is_control_available(film::eCTRL_BIT_DIRT_CLUMPING, filmMask))
            algoParams.damage.dirtClumping = get_slider_value(params, FilmSimulationCtrl::CLUMPING);
        if (is_control_available(film::eCTRL_BIT_GATE_DIRT, filmMask))
            algoParams.damage.gateDirt = get_slider_value(params, FilmSimulationCtrl::GATE_DIRT);
        if (is_control_available(film::eCTRL_BIT_SCRATCH_TRANSPORT, filmMask))
            algoParams.damage.scratchTransport = get_slider_value(params, FilmSimulationCtrl::TRANSPORT_SCRATCHES);
        if (is_control_available(film::eCTRL_BIT_SCRATCH_HANDLING, filmMask))
            algoParams.damage.scratchHandling = get_slider_value(params, FilmSimulationCtrl::HANDLING_SCRATCHES);
        if (is_control_available(film::eCTRL_BIT_WEAVE_AMOUNT, filmMask))
            algoParams.damage.weaveAmount = get_slider_value(params, FilmSimulationCtrl::GATE_WAVE);
        if (is_control_available(film::eCTRL_BIT_DAMAGE_EVENTS, filmMask))
            algoParams.damage.damageEvents = get_slider_value(params, FilmSimulationCtrl::SPLICE_AND_TEAR_EVENTS);
        if (is_control_available(film::eCTRL_BIT_PROCESSING_QUALITY, filmMask))
            algoParams.damage.processingQuality = get_slider_value(params, FilmSimulationCtrl::PROCESSING_QUALITY);
        if (is_control_available(film::eCTRL_BIT_DRYING_MARKS, filmMask))
            algoParams.damage.dryingMarks = get_slider_value(params, FilmSimulationCtrl::DRYING_MARKS);
        if (is_control_available(film::eCTRL_BIT_STORAGE_SEVERITY, filmMask))
            algoParams.damage.storageSeverity = get_slider_value(params, FilmSimulationCtrl::STORAGE_SEVERITY);
        if (is_control_available(film::eCTRL_BIT_COLOUR_VEIL, filmMask))
            algoParams.damage.colourVeil = get_slider_value(params, FilmSimulationCtrl::COLOUR_VEIL);
        if (is_control_available(film::eCTRL_BIT_FLICKER_STOPS, filmMask))
            algoParams.damage.flickerStops = get_slider_value(params, FilmSimulationCtrl::PRINTER_FLICKER);
        if (is_control_available(film::eCTRL_BIT_SCANNER_ARTIFACTS, filmMask))
            algoParams.damage.scannerArtifacts = get_slider_value(params, FilmSimulationCtrl::SCANNER_ARTIFACTS);
    }

    return algoParams;
}