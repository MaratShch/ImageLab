#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "GuidedFilter.hpp"
#include "GuidedFilterEnum.hpp"
#include "AlgoControls.hpp"
#include "AlgoMemHandler.hpp"
#include "AlgorithmMain.hpp"
#include "AlgoColorDispatcher.hpp"
#include "AlgoColorDispatcherOut.hpp"


PF_Err GuidedFilter_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) 
{
    PF_Err err = PF_Err_NONE;
    PF_EffectWorld* RESTRICT input = reinterpret_cast<PF_EffectWorld* RESTRICT>(&params[UnderlyingType(CtrlItems::gFILTER_INPUT)]->u.ld);

    const PF_Pixel_ARGB_8u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* RESTRICT>(input->data);
          PF_Pixel_ARGB_8u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* RESTRICT>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler memHndl = alloc_memory_buffers(sizeX, sizeY);
    if (true == mem_handler_valid(memHndl))
    {
        const AlgoControls algoControls = getAlgoControls(params);

        dispatch_convert_to_planar (localSrc, memHndl, sizeX, sizeY, src_pitch, PixelFormat::ARGB_8u);
        Algorithm_Main (memHndl, sizeX, sizeY, algoControls);
        dispatch_convert_to_interleaved (memHndl, localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, PixelFormat::ARGB_8u);

        free_memory_buffers (memHndl);
    }

	return PF_Err_NONE;
}


PF_Err GuidedFilter_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) 
{
    PF_Err err = PF_Err_NONE;
    PF_EffectWorld* RESTRICT input = reinterpret_cast<PF_EffectWorld* RESTRICT>(&params[UnderlyingType(CtrlItems::gFILTER_INPUT)]->u.ld);

    const PF_Pixel_ARGB_16u* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* RESTRICT>(input->data);
          PF_Pixel_ARGB_16u* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* RESTRICT>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler memHndl = alloc_memory_buffers(sizeX, sizeY);
    if (true == mem_handler_valid(memHndl))
    {
        const AlgoControls algoControls = getAlgoControls(params);

        dispatch_convert_to_planar (localSrc, memHndl, sizeX, sizeY, src_pitch, PixelFormat::ARGB_16u);
        Algorithm_Main (memHndl, sizeX, sizeY, algoControls);
        dispatch_convert_to_interleaved (memHndl, localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, PixelFormat::ARGB_16u);

        free_memory_buffers (memHndl);
    }

    return PF_Err_NONE;
}

PF_Err GuidedFilter_InAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) 
{
    PF_Err err = PF_Err_NONE;
    PF_EffectWorld* RESTRICT input = reinterpret_cast<PF_EffectWorld* RESTRICT>(&params[UnderlyingType(CtrlItems::gFILTER_INPUT)]->u.ld);

    const PF_Pixel_ARGB_32f* RESTRICT localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* RESTRICT>(input->data);
          PF_Pixel_ARGB_32f* RESTRICT localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* RESTRICT>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler memHndl = alloc_memory_buffers (sizeX, sizeY);
    if (true == mem_handler_valid(memHndl))
    {
        const AlgoControls algoControls = getAlgoControls(params);

        dispatch_convert_to_planar (localSrc, memHndl, sizeX, sizeY, src_pitch, PixelFormat::ARGB_32f);
        Algorithm_Main (memHndl, sizeX, sizeY, algoControls);
        dispatch_convert_to_interleaved (memHndl, localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, PixelFormat::ARGB_32f);

        free_memory_buffers (memHndl);
    }

    return PF_Err_NONE;
}


inline PF_Err GuidedFilter_InAE_DeepWorld
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) 
{
    PF_Err	err = PF_Err_NONE;
    PF_PixelFormat format = PF_PixelFormat_INVALID;
    AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* RESTRICT>(&params[UnderlyingType(CtrlItems::gFILTER_INPUT)]->u.ld), &format))
    {
        err = (format == PF_PixelFormat_ARGB128 ?
            GuidedFilter_InAE_32bits(in_data, out_data, params, output) : GuidedFilter_InAE_16bits(in_data, out_data, params, output));
    }
    else
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;

    return err;
}

PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) 
{
	return (PF_WORLD_IS_DEEP(output) ?
        GuidedFilter_InAE_DeepWorld (in_data, out_data, params, output) :
		GuidedFilter_InAE_8bits (in_data, out_data, params, output));
}