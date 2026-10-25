#include "FilmSimulation.hpp"
#include "AlgoControlEnums.hpp"
#include "AlgoAdobeControlEnums.hpp"
#include "AlgoMemHandler.hpp"
#include "AlgoPrFormatIngest.hpp"
#include "AlgoPrFormatEgress.hpp"
#include "AlgorithmMain.hpp"

using namespace AlgoPrIngest;

// ---------------------------------------------------------------------------
// After Effects pixel worlds are ARGB in every depth:
//   PF_PixelFormat_ARGB32  -> PF_Pixel      (8 bpc)  -> fmt_ARGB_4444_8u
//   PF_PixelFormat_ARGB64  -> PF_Pixel16    (16 bpc) -> fmt_ARGB_4444_16u
//   PF_PixelFormat_ARGB128 -> PF_PixelFloat (32 bpc) -> fmt_ARGB_4444_32f
// Straight (not premultiplied) colour, so the plain ARGB readers / writers of
// AlgoPrFormatIngest / AlgoPrFormatEgress apply unchanged; alpha is carried
// from the input pixel.
//
// ⚠ 32 bpc: the samples are taken as display-encoded (sRGB curve), After
// Effects' default. With "Linearize Working Space" on they are already linear;
// an effect cannot see that project setting, so a linearized project gets the
// curve twice. Select fmt_ARGB_4444_32f_Linear below for linearized projects.
// ---------------------------------------------------------------------------
PF_Err ProcessImgAEWorld
(
    PF_InData*            in_data,
    PF_OutData*           out_data,
    PF_ParamDef*          params[],
    const PF_EffectWorld* input,
    PF_EffectWorld*       output
)
{
    if (nullptr == input || nullptr == output || nullptr == input->data || nullptr == output->data)
        return PF_Err_BAD_CALLBACK_PARAM;

    PF_PixelFormat format = PF_PixelFormat_INVALID;
    {
        auto const wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
        if (PF_Err_NONE != wsP->PF_GetPixelFormat(const_cast<PF_EffectWorld*>(input), &format))
            return PF_Err_UNRECOGNIZED_PARAM_TYPE;
    }

    ePrPixelFormat fmt = fmt_ARGB_4444_8u;
    A_long pixelBytes = 0;
    switch (format)
    {
        case PF_PixelFormat_ARGB32:
            fmt = fmt_ARGB_4444_8u;
            pixelBytes = static_cast<A_long>(sizeof(PF_Pixel_ARGB_8u));
        break;

        case PF_PixelFormat_ARGB64:
            fmt = fmt_ARGB_4444_16u;
            pixelBytes = static_cast<A_long>(sizeof(PF_Pixel_ARGB_16u));
        break;

        case PF_PixelFormat_ARGB128:
            fmt = fmt_ARGB_4444_32f;
            pixelBytes = static_cast<A_long>(sizeof(PF_Pixel_ARGB_32f));
        break;

        default:
            return PF_Err_UNRECOGNIZED_PARAM_TYPE;
    }

    // Whole frame from (0,0). The input and output worlds cover the same
    // rectangle (PreRender returns the input's result_rect as the output's).
    const A_long sizeX = (input->width  < output->width)  ? input->width  : output->width;
    const A_long sizeY = (input->height < output->height) ? input->height : output->height;
    if (sizeX <= 0 || sizeY <= 0)
        return PF_Err_NONE;

    MemHandler memHndl = alloc_memory_buffers(static_cast<int32_t>(sizeX), static_cast<int32_t>(sizeY));
    const MemHandlerGuard memGuard(memHndl);
    if (false == mem_handler_valid(memHndl))
        return PF_Err_OUT_OF_MEMORY;

    const double  fpsRate = image_lab_get_fps(in_data);
    const int32_t idx     = film_sim_frame_index(in_data);
    const AlgoControls algoControls = getAlgoControls(params, fpsRate, idx);

    const A_long srcLinePitch = input->rowbytes  / pixelBytes;
    const A_long dstLinePitch = output->rowbytes / pixelBytes;

    // convert from Adobe buffer format to planar
    ingest_to_planar_f32(input->data, sizeX, sizeY, srcLinePitch, fmt,
                         memHndl.Src_R, memHndl.Src_G, memHndl.Src_B, memHndl.padW);

    // perform film simulations
    Algorithm_Main(memHndl, sizeX, sizeY, algoControls);

    // back convert from planar format to Adobe buffer format (alpha from the input)
    egress_from_planar_f32(memHndl.Dst_R, memHndl.Dst_G, memHndl.Dst_B, memHndl.padW,
                           sizeX, sizeY, output->data, dstLinePitch, fmt,
                           input->data, sizeX, srcLinePitch);

    return PF_Err_NONE;
}


// Classic PF_Cmd_RENDER in After Effects. AE uses the Smart Render path while
// PF_OutFlag2_SUPPORTS_SMART_RENDER is set; this entry stays functional for a
// host that sends PF_Cmd_RENDER instead.
PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
)
{
    return ProcessImgAEWorld(in_data, out_data, params,
                             &params[UnderlyingType(FilmSimulationCtrl::VIDEO_INPUT)]->u.ld, output);
}
