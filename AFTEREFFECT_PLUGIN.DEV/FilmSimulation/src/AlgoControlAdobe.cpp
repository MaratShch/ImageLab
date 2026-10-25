#include "AE_Effect.h"
#include "AlgoControl.hpp"
#include "AlgoControlAdobe.hpp"
#include "AlgoControlEnums.hpp"
#include "AlgoAdobeControlEnums.hpp"
#include "AlgoControlValues.hpp"
#include "film_params_mask.hpp"
#include "CompileTimeUtils.hpp"
#include "CommonAdobeAE.hpp"


// Every control whose availability depends on the selected stock, with the
// availability bit that governs it (film_params_mask.hpp). Refreshed whenever
// the stock changes, so a control the stock cannot use is shown disabled.
struct ControlBit
{
    FilmSimulationCtrl       ctrl;
    film::eFILM_CONTROL_BIT  bit;
};

static constexpr ControlBit kStockDependentControls[] =
{
    { FilmSimulationCtrl::FILM_FORMAT,              film::eCTRL_BIT_FILM_FORMAT },
    { FilmSimulationCtrl::PROCESS_VARIANT,          film::eCTRL_BIT_PROCESS_VARIANT },
    { FilmSimulationCtrl::EXPOSURE,                 film::eCTRL_BIT_EXPOSURE_STOPS },
    { FilmSimulationCtrl::EXPOSURE_TIME,            film::eCTRL_BIT_EXPOSURE_TIME_S },
    { FilmSimulationCtrl::MID_GRAY_TARGET,          film::eCTRL_BIT_GREY_TARGET },
    { FilmSimulationCtrl::BLACK_POINT_STRETCH,      film::eCTRL_BIT_BLACK_POINT_STRETCH },
    { FilmSimulationCtrl::AIM_EXPOSURE,             film::eCTRL_BIT_AIM_EXPOSURE },
    { FilmSimulationCtrl::DEVELOPMENT_ENABLE,       film::eCTRL_BIT_DEVELOPMENT_MINUTES },
    { FilmSimulationCtrl::DEVELOPMENT_TIME,         film::eCTRL_BIT_DEVELOPMENT_MINUTES },
    { FilmSimulationCtrl::DEVELOPMENT_TEMPERATURE,  film::eCTRL_BIT_DEVELOPMENT_CELSIUS },
    { FilmSimulationCtrl::DEVELOPER_INDEX,          film::eCTRL_BIT_DEVELOPER_INDEX },
    { FilmSimulationCtrl::YEARS_OF_DARK_STORAGE,    film::eCTRL_BIT_STORAGE_YEARS },
    { FilmSimulationCtrl::STORAGE_TEMPERATURE,      film::eCTRL_BIT_STORAGE_CELSIUS },
    { FilmSimulationCtrl::BATCH_POSITION,           film::eCTRL_BIT_BATCH_POSITION },
    { FilmSimulationCtrl::SCENE_COLOUR_TEMPERATURE, film::eCTRL_BIT_SCENE_KELVIN },
    { FilmSimulationCtrl::WHITE_BALANCE_STRENGTH,   film::eCTRL_BIT_WB_STRENGTH },
    { FilmSimulationCtrl::PRINT_STOCK,              film::eCTRL_BIT_PRINT_STOCK },
    { FilmSimulationCtrl::DUPLICATION_GENERATION,   film::eCTRL_BIT_GENERATIONS },
    { FilmSimulationCtrl::INTERMEDIATE_STOCK,       film::eCTRL_BIT_DUPE_STOCK },
    { FilmSimulationCtrl::PRINT_GRAIN,              film::eCTRL_BIT_PRINT_GRAIN },
    { FilmSimulationCtrl::GRAIN,                    film::eCTRL_BIT_GRAIN_SCALE },
    { FilmSimulationCtrl::HALATION,                 film::eCTRL_BIT_HALATION_SCALE },
    { FilmSimulationCtrl::DIR_COUPLERS,             film::eCTRL_BIT_COUPLER_SCALE },
    { FilmSimulationCtrl::MISREGISTRATION,          film::eCTRL_BIT_MISREG_SCALE },
    { FilmSimulationCtrl::COATING_UNEVENNESS,       film::eCTRL_BIT_COATING_SCALE },
    { FilmSimulationCtrl::RESEAU_RECONSTRUCTION,    film::eCTRL_BIT_RESEAU },
    { FilmSimulationCtrl::VEILING_FLARE_AUTO,       film::eCTRL_BIT_FLARE },
    { FilmSimulationCtrl::VEILING_FLARE,            film::eCTRL_BIT_FLARE },
    { FilmSimulationCtrl::CORNER_FALLOFF_AUTO,      film::eCTRL_BIT_VIGNETTE },
    { FilmSimulationCtrl::CORNER_FALLOFF,           film::eCTRL_BIT_VIGNETTE },
    { FilmSimulationCtrl::SCANNER_SPECULARITY,      film::eCTRL_BIT_SCANNER_SPECULAR },
    { FilmSimulationCtrl::SCANNER_FIXED_PATTERN,    film::eCTRL_BIT_SCANNER_FIXED_PATTERN },
    { FilmSimulationCtrl::ENABLE_FILM_DAMAGE,       film::eCTRL_BIT_FILM_DAMAGE_ENABLED },
    { FilmSimulationCtrl::OVERALL_STRENGHT,         film::eCTRL_BIT_DAMAGE_STRENGTH },
    { FilmSimulationCtrl::DAMAGE_SEED,              film::eCTRL_BIT_DAMAGE_SEED },
    { FilmSimulationCtrl::DUST,                     film::eCTRL_BIT_DUST_LEVEL },
    { FilmSimulationCtrl::DEBRIS,                   film::eCTRL_BIT_DEBRIS_LEVEL },
    { FilmSimulationCtrl::FIBRES,                   film::eCTRL_BIT_FIBRE_LEVEL },
    { FilmSimulationCtrl::CLUMPING,                 film::eCTRL_BIT_DIRT_CLUMPING },
    { FilmSimulationCtrl::GATE_DIRT,                film::eCTRL_BIT_GATE_DIRT },
    { FilmSimulationCtrl::GATE_WAVE,                film::eCTRL_BIT_WEAVE_AMOUNT },
    { FilmSimulationCtrl::SPLICE_AND_TEAR_EVENTS,   film::eCTRL_BIT_DAMAGE_EVENTS },
    { FilmSimulationCtrl::TRANSPORT_SCRATCHES,      film::eCTRL_BIT_SCRATCH_TRANSPORT },
    { FilmSimulationCtrl::HANDLING_SCRATCHES,       film::eCTRL_BIT_SCRATCH_HANDLING },
    { FilmSimulationCtrl::PROCESSING_QUALITY,       film::eCTRL_BIT_PROCESSING_QUALITY },
    { FilmSimulationCtrl::DRYING_MARKS,             film::eCTRL_BIT_DRYING_MARKS },
    { FilmSimulationCtrl::STORAGE_SEVERITY,         film::eCTRL_BIT_STORAGE_SEVERITY },
    { FilmSimulationCtrl::COLOUR_VEIL,              film::eCTRL_BIT_COLOUR_VEIL },
    { FilmSimulationCtrl::PRINTER_FLICKER,          film::eCTRL_BIT_FLICKER_STOPS },
    { FilmSimulationCtrl::SCANNER_ARTIFACTS,        film::eCTRL_BIT_SCANNER_ARTIFACTS },
    { FilmSimulationCtrl::DEFECT_READER_OPTICS,     film::eCTRL_BIT_DEFECT_OPTICS },
    { FilmSimulationCtrl::TRANSPORT_SCRATCH_LIFETIME, film::eCTRL_BIT_SCRATCH_LIFETIME },
    { FilmSimulationCtrl::PERSISTENT_DUST_LIFETIME, film::eCTRL_BIT_DUST_LIFETIME },
    { FilmSimulationCtrl::DEFECT_VARIABILITY,       film::eCTRL_BIT_DEFECT_VARIABILITY },
    { FilmSimulationCtrl::DEFECT_VISIBILITY,        film::eCTRL_BIT_DEFECT_VISIBILITY },
    { FilmSimulationCtrl::SPECTRAL_TAKING_MATRIX,   film::eCTRL_BIT_SPECTRAL_TAKING },
    { FilmSimulationCtrl::GRAIN_TEMPORAL_MODE,      film::eCTRL_BIT_GRAIN_TEMPORAL_MODE },
    { FilmSimulationCtrl::GRAIN_SEED,               film::eCTRL_BIT_SEED },
    { FilmSimulationCtrl::SPECTRAL_UPSAMPLING,      film::eCTRL_BIT_SPECTRAL_UPSAMPLING },
    { FilmSimulationCtrl::FILM_AGE_EXPIRED_STOCK,   film::eCTRL_BIT_FILM_AGE_MONTHS }
};


// Children of "Enable Film Damage": usable only while the box is ticked.
static bool is_film_damage_child(const FilmSimulationCtrl ctrl) noexcept
{
    switch (ctrl)
    {
        case FilmSimulationCtrl::OVERALL_STRENGHT:
        case FilmSimulationCtrl::DAMAGE_SEED:
        case FilmSimulationCtrl::DUST:
        case FilmSimulationCtrl::DEBRIS:
        case FilmSimulationCtrl::FIBRES:
        case FilmSimulationCtrl::CLUMPING:
        case FilmSimulationCtrl::GATE_DIRT:
        case FilmSimulationCtrl::GATE_WAVE:
        case FilmSimulationCtrl::SPLICE_AND_TEAR_EVENTS:
        case FilmSimulationCtrl::TRANSPORT_SCRATCHES:
        case FilmSimulationCtrl::HANDLING_SCRATCHES:
        case FilmSimulationCtrl::PROCESSING_QUALITY:
        case FilmSimulationCtrl::DRYING_MARKS:
        case FilmSimulationCtrl::STORAGE_SEVERITY:
        case FilmSimulationCtrl::COLOUR_VEIL:
        case FilmSimulationCtrl::PRINTER_FLICKER:
        case FilmSimulationCtrl::SCANNER_ARTIFACTS:
        case FilmSimulationCtrl::DEFECT_READER_OPTICS:
        case FilmSimulationCtrl::TRANSPORT_SCRATCH_LIFETIME:
        case FilmSimulationCtrl::PERSISTENT_DUST_LIFETIME:
        case FilmSimulationCtrl::DEFECT_VARIABILITY:
        case FilmSimulationCtrl::DEFECT_VISIBILITY:
            return true;
        default:
            return false;
    }
}


// Enabled = the stock can use the control AND its governing check-box allows
// it. Mirrors the gating in getAlgoControls(), so a greyed control is exactly
// a control whose value the render ignores.
static bool is_control_enabled(const ControlBit& cb, const uint64_t filmMask, PF_ParamDef* params[]) noexcept
{
    if (false == is_control_available(cb.bit, filmMask))
        return false;

    switch (cb.ctrl)
    {
        case FilmSimulationCtrl::DEVELOPMENT_TIME:
        case FilmSimulationCtrl::DEVELOPMENT_TEMPERATURE:
            return get_check_box_value(params, FilmSimulationCtrl::DEVELOPMENT_ENABLE);

        case FilmSimulationCtrl::VEILING_FLARE:
            return false == get_check_box_value(params, FilmSimulationCtrl::VEILING_FLARE_AUTO);

        case FilmSimulationCtrl::CORNER_FALLOFF:
            return false == get_check_box_value(params, FilmSimulationCtrl::CORNER_FALLOFF_AUTO);

        default:
        break;
    }

    if (true == is_film_damage_child(cb.ctrl))
        return is_control_available(film::eCTRL_BIT_FILM_DAMAGE_ENABLED, filmMask) &&
               get_check_box_value(params, FilmSimulationCtrl::ENABLE_FILM_DAMAGE);

    return true;
}


// Applies the enabled / disabled state to every stock-dependent control,
// calling PF_UpdateParamUI only for a control whose state changes.
// Not noexcept: AEFX_SuiteScoper THROWS (a PF_Err) when the suite cannot be
// acquired, and a throw out of a noexcept function is std::terminate -- the host
// process dies. The callers catch and return the error to AE instead.
static void refresh_control_states(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[])
{
    const film::eFILM_PROFILE filmProfile = get_film_profile(params);
    const uint64_t filmMask = film::kFilmControlAvailability[UnderlyingType(filmProfile)];

    const auto site = AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);

    for (const ControlBit& cb : kStockDependentControls)
    {
        // AE SDK: PF_UpdateParamUI must receive a COPY of the param def -- never the
        // PF_ParamDef passed in params[] (see the SDK "Supervisor" sample).
        PF_ParamDef paramDef = *params[UnderlyingType(cb.ctrl)];
        const PF_ParamUIFlags before = paramDef.ui_flags;
        set_control_status(paramDef.ui_flags, is_control_enabled(cb, filmMask, params));
        if (paramDef.ui_flags != before)
            site->PF_UpdateParamUI(in_data->effect_ref, UnderlyingType(cb.ctrl), &paramDef);
    }

    return;
}


static PF_Err refresh_control_states_safe(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[])
{
    try
    {
        refresh_control_states(in_data, out_data, params);
    }
    catch (const PF_Err& thrownErr)
    {
        return thrownErr;
    }
    catch (...)
    {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return PF_Err_NONE;
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
        // the stock and every check-box that governs other controls
        case UnderlyingType(FilmSimulationCtrl::FILM_STOCK):
        case UnderlyingType(FilmSimulationCtrl::DEVELOPMENT_ENABLE):
        case UnderlyingType(FilmSimulationCtrl::VEILING_FLARE_AUTO):
        case UnderlyingType(FilmSimulationCtrl::CORNER_FALLOFF_AUTO):
        case UnderlyingType(FilmSimulationCtrl::ENABLE_FILM_DAMAGE):
            return refresh_control_states_safe(in_data, out_data, params);

        default:
        break;
    }

    return PF_Err_NONE;
}


// PF_Cmd_UPDATE_PARAMS_UI: also sent after a project is opened, when the
// panel shows the flags registered in SetupControlElements() -- restore the
// state that belongs to the saved values.
PF_Err user_update_params_ui
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_ParamDef			*params[],
    PF_LayerDef			*outputP
)
{
    return refresh_control_states_safe(in_data, out_data, params);
}



AlgoControls getAlgoControls(PF_ParamDef* params[], const double fps, const int32_t idx)
{
    // Start from the engine defaults: a control the selected stock cannot use,
    // or a "stock's own" choice, leaves the engine default (or sentinel) in place.
    CACHE_ALIGN AlgoControls algoParams = getAlgoControlsDefault();

    algoParams.filmProfile = get_film_profile(params);   // clamped to the database
    algoParams.frameRate = ((fps < 1.0) ? 24.0 : fps);
    algoParams.frameIndex = idx;

    const uint64_t filmMask = film::kFilmControlAvailability[UnderlyingType(algoParams.filmProfile)];

    // Film stock
    if (is_control_available(film::eCTRL_BIT_FILM_FORMAT, filmMask))
        algoParams.filmFormat = get_list_box_value<FilmFormatCtrl>(params, FilmSimulationCtrl::FILM_FORMAT);
    if (is_control_available(film::eCTRL_BIT_PROCESS_VARIANT, filmMask))
        algoParams.processVariant = get_list_box_value<ProcessVariantCtrl>(params, FilmSimulationCtrl::PROCESS_VARIANT);

    // Exposure & tone (float sliders)
    if (is_control_available(film::eCTRL_BIT_EXPOSURE_STOPS, filmMask))
        algoParams.exposureStops = get_float_slider_value(params, FilmSimulationCtrl::EXPOSURE);
    if (is_control_available(film::eCTRL_BIT_EXPOSURE_TIME_S, filmMask))
        algoParams.exposureTimeS = get_float_slider_value(params, FilmSimulationCtrl::EXPOSURE_TIME);
    if (is_control_available(film::eCTRL_BIT_GREY_TARGET, filmMask))
        algoParams.greyTarget = get_float_slider_value(params, FilmSimulationCtrl::MID_GRAY_TARGET);
    if (is_control_available(film::eCTRL_BIT_BLACK_POINT_STRETCH, filmMask))
        algoParams.blackPointStretch = get_float_slider_value(params, FilmSimulationCtrl::BLACK_POINT_STRETCH);
    if (is_control_available(film::eCTRL_BIT_AIM_EXPOSURE, filmMask))
        algoParams.aimExposure = get_check_box_value(params, FilmSimulationCtrl::AIM_EXPOSURE);

    // Development & storage.
    // Development time / temperature are applied ONLY when "Development Enable"
    // is ticked; otherwise the engine's -1 sentinels stay ("as the stock was
    // developed"). A slider cannot hold -1, so without this gate every render
    // would be re-developed at whatever value the slider shows.
    if (true == get_check_box_value(params, FilmSimulationCtrl::DEVELOPMENT_ENABLE))
    {
        if (is_control_available(film::eCTRL_BIT_DEVELOPMENT_MINUTES, filmMask))
            algoParams.developmentMinutes = get_float_slider_value(params, FilmSimulationCtrl::DEVELOPMENT_TIME);
        if (is_control_available(film::eCTRL_BIT_DEVELOPMENT_CELSIUS, filmMask))
            algoParams.developmentCelsius = get_float_slider_value(params, FilmSimulationCtrl::DEVELOPMENT_TEMPERATURE);
    }
    if (is_control_available(film::eCTRL_BIT_DEVELOPER_INDEX, filmMask))
        algoParams.developerIndex = get_list_box_value<DeveloperCtrl>(params, FilmSimulationCtrl::DEVELOPER_INDEX);
    if (is_control_available(film::eCTRL_BIT_STORAGE_YEARS, filmMask))           // int slider
        algoParams.storageYears = static_cast<double>(get_slider_value(params, FilmSimulationCtrl::YEARS_OF_DARK_STORAGE));
    if (is_control_available(film::eCTRL_BIT_STORAGE_CELSIUS, filmMask))         // int slider
        algoParams.storageCelsius = static_cast<double>(get_slider_value(params, FilmSimulationCtrl::STORAGE_TEMPERATURE));
    if (is_control_available(film::eCTRL_BIT_BATCH_POSITION, filmMask))
        algoParams.batchPosition = get_float_slider_value(params, FilmSimulationCtrl::BATCH_POSITION);

    // Colour & white balance
    if (is_control_available(film::eCTRL_BIT_SCENE_KELVIN, filmMask))            // int slider
        algoParams.sceneKelvin = static_cast<double>(get_slider_value(params, FilmSimulationCtrl::SCENE_COLOUR_TEMPERATURE));
    if (is_control_available(film::eCTRL_BIT_WB_STRENGTH, filmMask))
        algoParams.wbStrength = get_float_slider_value(params, FilmSimulationCtrl::WHITE_BALANCE_STRENGTH);

    // Print & duplication
    if (is_control_available(film::eCTRL_BIT_PRINT_STOCK, filmMask))
        algoParams.printStock = get_list_box_value<PrintStockCtrl>(params, FilmSimulationCtrl::PRINT_STOCK);
    if (is_control_available(film::eCTRL_BIT_GENERATIONS, filmMask))             // int slider
        algoParams.generations = get_slider_value(params, FilmSimulationCtrl::DUPLICATION_GENERATION);
    if (is_control_available(film::eCTRL_BIT_DUPE_STOCK, filmMask))
    {
        const DupeStockCtrl dupe = get_list_box_value<DupeStockCtrl>(params, FilmSimulationCtrl::INTERMEDIATE_STOCK);
        algoParams.dupeStock = ((dupe == DupeStockCtrl::eDUPE_FINE_GRAIN) || (dupe == DupeStockCtrl::eKODAK_VISION3_DI_2254))
            ? dupe
            : DupeStockCtrlDef;   // eDUPE_FINE_GRAIN
    }
    if (is_control_available(film::eCTRL_BIT_PRINT_GRAIN, filmMask))
        algoParams.printGrain = get_check_box_value(params, FilmSimulationCtrl::PRINT_GRAIN);

    // Emulsion character (float sliders)
    if (is_control_available(film::eCTRL_BIT_GRAIN_SCALE, filmMask))
        algoParams.grainScale = get_float_slider_value(params, FilmSimulationCtrl::GRAIN);
    if (is_control_available(film::eCTRL_BIT_HALATION_SCALE, filmMask))
        algoParams.halationScale = get_float_slider_value(params, FilmSimulationCtrl::HALATION);
    if (is_control_available(film::eCTRL_BIT_COUPLER_SCALE, filmMask))
        algoParams.couplerScale = get_float_slider_value(params, FilmSimulationCtrl::DIR_COUPLERS);
    if (is_control_available(film::eCTRL_BIT_MISREG_SCALE, filmMask))
        algoParams.misregScale = get_float_slider_value(params, FilmSimulationCtrl::MISREGISTRATION);
    if (is_control_available(film::eCTRL_BIT_COATING_SCALE, filmMask))
        algoParams.coatingScale = get_float_slider_value(params, FilmSimulationCtrl::COATING_UNEVENNESS);
    if (is_control_available(film::eCTRL_BIT_RESEAU, filmMask))
        algoParams.reseau = get_check_box_value(params, FilmSimulationCtrl::RESEAU_RECONSTRUCTION);

    // Lens & reader. Flare and corner falloff keep the engine's -1 sentinel
    // ("the stock's own era value") while their "Stock's Own" box is ticked.
    if (is_control_available(film::eCTRL_BIT_FLARE, filmMask) &&
        false == get_check_box_value(params, FilmSimulationCtrl::VEILING_FLARE_AUTO))
        algoParams.flare = get_float_slider_value(params, FilmSimulationCtrl::VEILING_FLARE);
    if (is_control_available(film::eCTRL_BIT_VIGNETTE, filmMask) &&
        false == get_check_box_value(params, FilmSimulationCtrl::CORNER_FALLOFF_AUTO))
        algoParams.vignette = get_float_slider_value(params, FilmSimulationCtrl::CORNER_FALLOFF);
    if (is_control_available(film::eCTRL_BIT_SCANNER_SPECULAR, filmMask))
        algoParams.scannerSpecular = get_float_slider_value(params, FilmSimulationCtrl::SCANNER_SPECULARITY);
    if (is_control_available(film::eCTRL_BIT_SCANNER_FIXED_PATTERN, filmMask))
        algoParams.scannerFixedPattern = get_float_slider_value(params, FilmSimulationCtrl::SCANNER_FIXED_PATTERN);

    // Film damage (float sliders)
    if (is_control_available(film::eCTRL_BIT_FILM_DAMAGE_ENABLED, filmMask))
        algoParams.filmDamageEnabled = get_check_box_value(params, FilmSimulationCtrl::ENABLE_FILM_DAMAGE);
    if (true == algoParams.filmDamageEnabled)
    {
        if (is_control_available(film::eCTRL_BIT_DAMAGE_STRENGTH, filmMask))
            algoParams.damage.damageStrength = get_float_slider_value(params, FilmSimulationCtrl::OVERALL_STRENGHT);
        if (is_control_available(film::eCTRL_BIT_DAMAGE_SEED, filmMask))          // int slider, offset
            algoParams.damage.damageSeed = DamageSeedDef + get_slider_value(params, FilmSimulationCtrl::DAMAGE_SEED);
        if (is_control_available(film::eCTRL_BIT_DUST_LEVEL, filmMask))
            algoParams.damage.dustLevel = get_float_slider_value(params, FilmSimulationCtrl::DUST);
        if (is_control_available(film::eCTRL_BIT_DEBRIS_LEVEL, filmMask))
            algoParams.damage.debrisLevel = get_float_slider_value(params, FilmSimulationCtrl::DEBRIS);
        if (is_control_available(film::eCTRL_BIT_FIBRE_LEVEL, filmMask))
            algoParams.damage.fibreLevel = get_float_slider_value(params, FilmSimulationCtrl::FIBRES);
        if (is_control_available(film::eCTRL_BIT_DIRT_CLUMPING, filmMask))
            algoParams.damage.dirtClumping = get_float_slider_value(params, FilmSimulationCtrl::CLUMPING);
        if (is_control_available(film::eCTRL_BIT_GATE_DIRT, filmMask))
            algoParams.damage.gateDirt = get_float_slider_value(params, FilmSimulationCtrl::GATE_DIRT);
        if (is_control_available(film::eCTRL_BIT_SCRATCH_TRANSPORT, filmMask))
            algoParams.damage.scratchTransport = get_float_slider_value(params, FilmSimulationCtrl::TRANSPORT_SCRATCHES);
        if (is_control_available(film::eCTRL_BIT_SCRATCH_HANDLING, filmMask))
            algoParams.damage.scratchHandling = get_float_slider_value(params, FilmSimulationCtrl::HANDLING_SCRATCHES);
        if (is_control_available(film::eCTRL_BIT_WEAVE_AMOUNT, filmMask))
            algoParams.damage.weaveAmount = get_float_slider_value(params, FilmSimulationCtrl::GATE_WAVE);
        if (is_control_available(film::eCTRL_BIT_DAMAGE_EVENTS, filmMask))
            algoParams.damage.damageEvents = get_float_slider_value(params, FilmSimulationCtrl::SPLICE_AND_TEAR_EVENTS);
        if (is_control_available(film::eCTRL_BIT_PROCESSING_QUALITY, filmMask))
            algoParams.damage.processingQuality = get_float_slider_value(params, FilmSimulationCtrl::PROCESSING_QUALITY);
        if (is_control_available(film::eCTRL_BIT_DRYING_MARKS, filmMask))
            algoParams.damage.dryingMarks = get_float_slider_value(params, FilmSimulationCtrl::DRYING_MARKS);
        if (is_control_available(film::eCTRL_BIT_STORAGE_SEVERITY, filmMask))
            algoParams.damage.storageSeverity = get_float_slider_value(params, FilmSimulationCtrl::STORAGE_SEVERITY);
        if (is_control_available(film::eCTRL_BIT_COLOUR_VEIL, filmMask))
            algoParams.damage.colourVeil = get_float_slider_value(params, FilmSimulationCtrl::COLOUR_VEIL);
        if (is_control_available(film::eCTRL_BIT_FLICKER_STOPS, filmMask))
            algoParams.damage.flickerStops = get_float_slider_value(params, FilmSimulationCtrl::PRINTER_FLICKER);
        if (is_control_available(film::eCTRL_BIT_SCANNER_ARTIFACTS, filmMask))
            algoParams.damage.scannerArtifacts = get_float_slider_value(params, FilmSimulationCtrl::SCANNER_ARTIFACTS);
        if (is_control_available(film::eCTRL_BIT_DEFECT_OPTICS, filmMask))
            algoParams.damage.defectOptics = get_check_box_value(params, FilmSimulationCtrl::DEFECT_READER_OPTICS);
        if (is_control_available(film::eCTRL_BIT_SCRATCH_LIFETIME, filmMask))
            algoParams.damage.scratchLifetime = get_float_slider_value(params, FilmSimulationCtrl::TRANSPORT_SCRATCH_LIFETIME);
        if (is_control_available(film::eCTRL_BIT_DUST_LIFETIME, filmMask))
            algoParams.damage.dustLifetime = get_float_slider_value(params, FilmSimulationCtrl::PERSISTENT_DUST_LIFETIME);
        if (is_control_available(film::eCTRL_BIT_DEFECT_VARIABILITY, filmMask))
            algoParams.damage.defectVariability = get_float_slider_value(params, FilmSimulationCtrl::DEFECT_VARIABILITY);
        if (is_control_available(film::eCTRL_BIT_DEFECT_VISIBILITY, filmMask))
            algoParams.damage.defectVisibility = get_float_slider_value(params, FilmSimulationCtrl::DEFECT_VISIBILITY);
    }

    // Spectral & temporal
    if (is_control_available(film::eCTRL_BIT_SPECTRAL_TAKING, filmMask))
        algoParams.spectralTaking = get_check_box_value(params, FilmSimulationCtrl::SPECTRAL_TAKING_MATRIX);
    if (is_control_available(film::eCTRL_BIT_GRAIN_TEMPORAL_MODE, filmMask))
        algoParams.grainTemporalMode = get_list_box_value<GrainTemporalModeCtrl>(params, FilmSimulationCtrl::GRAIN_TEMPORAL_MODE);
    if (is_control_available(film::eCTRL_BIT_SEED, filmMask))                    // int slider, offset
        algoParams.seed = SeedDef + get_slider_value(params, FilmSimulationCtrl::GRAIN_SEED);
    if (is_control_available(film::eCTRL_BIT_SPECTRAL_UPSAMPLING, filmMask))
        algoParams.spectralUpsampling = get_check_box_value(params, FilmSimulationCtrl::SPECTRAL_UPSAMPLING);

    // Raw stock (float slider)
    if (is_control_available(film::eCTRL_BIT_FILM_AGE_MONTHS, filmMask))
        algoParams.filmAgeMonths = get_float_slider_value(params, FilmSimulationCtrl::FILM_AGE_EXPIRED_STOCK);

    return algoParams;
}
