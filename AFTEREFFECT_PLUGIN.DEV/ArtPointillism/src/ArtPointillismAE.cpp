#include "ArtPointillism.hpp"
#include "ArtPointillismAlgo.hpp"
#include "ArtPointillismControl.hpp"
#include "ArtPointillismEnums.hpp"
#include "Avx2ColorConverts.hpp"
#include "ImageLabMemInterface.hpp"



PF_Err ArtPointilism_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
)
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<      PF_EffectWorld*   __restrict>(&params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_INPUT)]->u.ld);
    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
          PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY);
    if (true == mem_handler_valid(algoMemHandler))
    {
        float* __restrict srcL  = algoMemHandler.L;
        float* __restrict srcAB = algoMemHandler.ab;
        float* __restrict dstL  = algoMemHandler.dst_L;
        float* __restrict dstAB = algoMemHandler.dst_ab;

        const PontillismControls algoControls = GetControlParametersStruct(params);

        AVX2_ConvertRgbToCIELab_SemiPlanar(localSrc, srcL, srcAB, sizeX, sizeY, src_pitch, dst_pitch);

        // execute algorithm
        ArtPointillismAlgorithmExec(algoMemHandler, algoControls, sizeX, sizeY);

        // back convert to native buffer format after processing complete
        AVX2_ConvertCIELab_SemiPlanar_ToRgb(localSrc, dstL, dstAB, localDst, sizeX, sizeY, src_pitch, dst_pitch);

        free_memory_buffers(algoMemHandler);
    } // if (true == mem_handler_valid (algoMemHandler))
    else
        err = PF_Err_OUT_OF_MEMORY;

	return PF_Err_NONE;
}


PF_Err ArtPointilism_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
)
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<      PF_EffectWorld*   __restrict>(&params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_INPUT)]->u.ld);
    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;
    
    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY);
    if (true == mem_handler_valid(algoMemHandler))
    {
        float* __restrict srcL  = algoMemHandler.L;
        float* __restrict srcAB = algoMemHandler.ab;
        float* __restrict dstL  = algoMemHandler.dst_L;
        float* __restrict dstAB = algoMemHandler.dst_ab;

        const PontillismControls algoControls = GetControlParametersStruct(params);

        AVX2_ConvertRgbToCIELab_SemiPlanar (localSrc, srcL, srcAB, sizeX, sizeY, src_pitch, dst_pitch);

        // execute algorithm
        ArtPointillismAlgorithmExec (algoMemHandler, algoControls, sizeX, sizeY);

        // back convert to native buffer format after processing complete
        AVX2_ConvertCIELab_SemiPlanar_ToRgb (localSrc, dstL, dstAB, localDst, sizeX, sizeY, src_pitch, dst_pitch);

        free_memory_buffers(algoMemHandler);
    } // if (true == mem_handler_valid (algoMemHandler))
    else
        err = PF_Err_OUT_OF_MEMORY;

    return PF_Err_NONE;
}


PF_Err ArtPointilism_InAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
)
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<      PF_EffectWorld*   __restrict>(&params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_INPUT)]->u.ld);
    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    PF_Err err = PF_Err_NONE;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY);
    if (true == mem_handler_valid(algoMemHandler))
    {
        float* __restrict srcL  = algoMemHandler.L;
        float* __restrict srcAB = algoMemHandler.ab;
        float* __restrict dstL  = algoMemHandler.dst_L;
        float* __restrict dstAB = algoMemHandler.dst_ab;

        const PontillismControls algoControls = GetControlParametersStruct(params);

        AVX2_ConvertRgbToCIELab_SemiPlanar (localSrc, srcL, srcAB, sizeX, sizeY, src_pitch, dst_pitch);

        // execute algorithm
        ArtPointillismAlgorithmExec (algoMemHandler, algoControls, sizeX, sizeY);

        // back convert to native buffer format after processing complete
        AVX2_ConvertCIELab_SemiPlanar_ToRgb (localSrc, dstL, dstAB, localDst, sizeX, sizeY, src_pitch, dst_pitch);

        free_memory_buffers(algoMemHandler);
    } // if (true == mem_handler_valid (algoMemHandler))
    else
        err = PF_Err_OUT_OF_MEMORY;

    return PF_Err_NONE;
}


inline PF_Err ArtPointilism_InAE_DeepWorld
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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[UnderlyingType(ArtPointillismControls::ART_POINTILLISM_INPUT)]->u.ld), &format))
    {
        err = (format == PF_PixelFormat_ARGB128 ?
            ArtPointilism_InAE_32bits(in_data, out_data, params, output) : ArtPointilism_InAE_16bits(in_data, out_data, params, output));
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
        ArtPointilism_InAE_DeepWorld (in_data, out_data, params, output) :
        ArtPointilism_InAE_8bits (in_data, out_data, params, output));
}