#ifndef __IMAGE_LAB_IMAGE_FILM_SIMULATION_FILTER__
#define __IMAGE_LAB_IMAGE_FILM_SIMULATION_FILTER__

#include <cmath>
#include "CommonAdobeAE.hpp"
#include "AlgoControl.hpp"
#include "AlgoMemHandler.hpp"

constexpr char strName[] = "Film Simulation";
constexpr char strCopyright[] = "\n2019-2026. ImageLab2 Copyright(c).\rFilm Simulation plugin.";
constexpr int FilmSimulation_VersionMajor = IMAGE_LAB_AE_PLUGIN_VERSION_MAJOR;
constexpr int FilmSimulation_VersionMinor = IMAGE_LAB_AE_PLUGIN_VERSION_MINOR;
constexpr int FilmSimulation_VersionSub = 0;
#ifdef _DEBUG
constexpr int FilmSimulation_VersionStage = PF_Stage_DEVELOP;
#else
constexpr int FilmSimulation_VersionStage = PF_Stage_RELEASE;
#endif
constexpr int FilmSimulation_VersionBuild = 1;


// ---------------------------------------------------------------------------
// Frees a frame arena on every exit path of the scope that owns it, including
// an exception thrown by AEFX_SuiteScoper (caught in EffectMain).
// free_memory_buffers() is safe on a handler whose allocation failed.
// ---------------------------------------------------------------------------
class MemHandlerGuard
{
public:
    explicit MemHandlerGuard (MemHandler& h) noexcept : m_h(h) {}
    ~MemHandlerGuard() noexcept { free_memory_buffers(m_h); }
    MemHandlerGuard (const MemHandlerGuard&) = delete;
    MemHandlerGuard& operator= (const MemHandlerGuard&) = delete;
private:
    MemHandler& m_h;
};


// ---------------------------------------------------------------------------
// Zero-based FRAME index of the current render, for AlgoControls::frameIndex.
//
// current_time / time_step (image_lab_get_frame_number) is not a frame index:
// time_step varies under time remapping and is a FIELD in Premiere's field
// renders, so both fields of one frame received different indices (damage and
// grain strobing) and a remapped layer jumped. The index is taken from the
// time in seconds and the same frame rate the engine receives
// (image_lab_get_fps: local_time_step, field-aware), floored with a small
// tolerance so an exact frame start never rounds down to the previous frame.
// ---------------------------------------------------------------------------
inline int32_t film_sim_frame_index (const PF_InData* in_data) noexcept
{
    const double fps = image_lab_get_fps(in_data);
    if (fps <= 0.0 || 0u == in_data->time_scale)
        return 0;

    const double seconds = static_cast<double>(in_data->current_time) / static_cast<double>(in_data->time_scale);
    const double frame   = std::floor(seconds * fps + 1.0e-4);

    constexpr double kMaxIdx = 2147483647.0;
    constexpr double kMinIdx = -2147483647.0;
    return static_cast<int32_t>(frame > kMaxIdx ? kMaxIdx : (frame < kMinIdx ? kMinIdx : frame));
}


PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
);

PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
);

// After Effects: process one input world into one output world (8, 16 or
// 32 bpc, PF_PixelFormat_ARGB32 / ARGB64 / ARGB128). Shared by the Smart Render
// path and the classic PF_Cmd_RENDER path.
PF_Err ProcessImgAEWorld
(
    PF_InData*            in_data,
    PF_OutData*           out_data,
    PF_ParamDef*          params[],
    const PF_EffectWorld* input,
    PF_EffectWorld*       output
);

PF_Err
FilmSimulation_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
);

PF_Err
FilmSimulation_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
);

PF_Err DrawEvent
(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output,
    PF_EventExtra	*event_extra
);

PF_Err SetupControlElements
(
    PF_InData* in_data, 
    PF_OutData* out_data
);

AlgoControls getAlgoControls
(
    PF_ParamDef* params[],
    const double fps,
    const int32_t idx
);

PF_Err user_update_params_handler
(
    PF_InData						*in_data,
    PF_OutData						*out_data,
    PF_ParamDef						*params[],
    PF_LayerDef						*outputP,
    const PF_UserChangedParamExtra	*which_hitP
);

PF_Err user_update_params_ui
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_ParamDef			*params[],
    PF_LayerDef			*outputP
);

#endif // __IMAGE_LAB_IMAGE_FILM_SIMULATION_FILTER__
