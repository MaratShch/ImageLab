#include "AuthomaticWhiteBalance.hpp"
#include "AlgCommonFunctions.hpp"
#include "AlgCorrectionMatrix.hpp"
#include "AlgorithmMain.hpp"
#include "AlgoControl.hpp"
#include "AlgoMemHandler.hpp"
#include "AlgConvertDispatcher.hpp"
#include "AlgConvertDispatcherOut.hpp"

PF_Err ProcessImgInAE_8bits
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) 
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<      PF_EffectWorld*   __restrict>(&params[AWB_INPUT]->u.ld);
    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
          PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    const AlgoControls algoControls = GetControlParametersStruct(params);

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, algoControls.sliderIterCnt);

    if (true == mem_handler_valid(algoMemHandler))
    {
        dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, src_pitch, PixelFormat::ARGB_8u);

        // execute algorithm
        Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

        // back convert to native buffer format after processing complete
        dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, PixelFormat::ARGB_8u);

        free_memory_buffers (algoMemHandler);
    } // if (true == mem_handler_valid (algoMemHandler))
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}


PF_Err ProcessImgInAE_16bits
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) 
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<      PF_EffectWorld*   __restrict>(&params[AWB_INPUT]->u.ld);
    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    const AlgoControls algoControls = GetControlParametersStruct(params);

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, algoControls.sliderIterCnt);

    if (true == mem_handler_valid(algoMemHandler))
    {
        dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, src_pitch, PixelFormat::ARGB_16u);

        // execute algorithm
        Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

        // back convert to native buffer format after processing complete
        dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, PixelFormat::ARGB_16u);

        free_memory_buffers (algoMemHandler);
    } // if (true == mem_handler_valid (algoMemHandler))
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}


PF_Err ProcessImgInAE_32bits
(
    PF_InData*    in_data,
    PF_OutData*   out_data,
    PF_ParamDef*  params[],
    PF_LayerDef*  output
) 
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<      PF_EffectWorld*   __restrict>(&params[AWB_INPUT]->u.ld);
    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    const AlgoControls algoControls = GetControlParametersStruct(params);

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, algoControls.sliderIterCnt);

    if (true == mem_handler_valid(algoMemHandler))
    {
        dispatch_convert_to_planar (localSrc, algoMemHandler, sizeX, sizeY, src_pitch, PixelFormat::ARGB_32f);

        // execute algorithm
        Algorithm_Main (algoMemHandler, sizeX, sizeY, algoControls);

        // back convert to native buffer format after processing complete
        dispatch_convert_to_interleaved (algoMemHandler, localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, PixelFormat::ARGB_32f);

        free_memory_buffers (algoMemHandler);
    } // if (true == mem_handler_valid (algoMemHandler))
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}


inline PF_Err ProcessImgInAE_DeepWord
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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld*>(&params[AWB_INPUT]->u.ld), &format))
        err = (format == PF_PixelFormat_ARGB128 ?
            ProcessImgInAE_32bits(in_data, out_data, params, output) : ProcessImgInAE_16bits(in_data, out_data, params, output));
    else
        PF_Err_UNRECOGNIZED_PARAM_TYPE;

    return err;
}


PF_Err ProcessImgInAE
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) 
{
	return (true == (PF_WORLD_IS_DEEP(output) ?
        ProcessImgInAE_DeepWord (in_data, out_data, params, output) :
		ProcessImgInAE_8bits (in_data, out_data, params, output) ) ? PF_Err_NONE : PF_Err_INVALID_INDEX);
}