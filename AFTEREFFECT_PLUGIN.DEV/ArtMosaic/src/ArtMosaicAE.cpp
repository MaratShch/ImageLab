#include "ArtMosaic.hpp"
#include "ArtMosaicEnum.hpp"
#include "MosaicColorConvert.hpp"
#include "MosaicMemHandler.hpp"


PF_Err ArtMosaic_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) 
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<PF_EffectWorld* __restrict>(&params[UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_INPUT)]->u.ld);
    const A_long cellsNumber = params[UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_CELLS_SLIDER)]->u.sd.value;

    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
          PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, cellsNumber);
    if (true == mem_handler_valid(algoMemHandler))
    {
        rgb2planar (localSrc, algoMemHandler, sizeX, sizeY, src_pitch);     // convert interleaved to planar format (range 0.f ... 225.f)
        MosaicAlgorithmMain (algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
        planar2rgb (localSrc, algoMemHandler, localDst, sizeX, sizeY, src_pitch, dst_pitch); // back convert from planar to interleaved format

        free_memory_buffers (algoMemHandler);
    }
    else
    {
        err = PF_Err_OUT_OF_MEMORY;
    }

	return err;
}


PF_Err ArtMosaic_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) 
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<PF_EffectWorld* __restrict>(&params[UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_INPUT)]->u.ld);
    const A_long cellsNumber = params[UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_CELLS_SLIDER)]->u.sd.value;

    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, cellsNumber);
    if (true == mem_handler_valid(algoMemHandler))
    {
        rgb2planar (localSrc, algoMemHandler, sizeX, sizeY, src_pitch);     // convert interleaved to planar format (range 0.f ... 225.f)
        MosaicAlgorithmMain (algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
        planar2rgb (localSrc, algoMemHandler, localDst, sizeX, sizeY, src_pitch, dst_pitch); // back convert from planar to interleaved format

        free_memory_buffers (algoMemHandler);
    }
    else
    {
        err = PF_Err_OUT_OF_MEMORY;
    }

    return err;
}


PF_Err ArtMosaic_InAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) 
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<PF_EffectWorld* __restrict>(&params[UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_INPUT)]->u.ld);
    const A_long cellsNumber = params[UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_CELLS_SLIDER)]->u.sd.value;

    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    PF_Err err = PF_Err_NONE;

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, cellsNumber);
    if (true == mem_handler_valid(algoMemHandler))
    {
        rgb2planar (localSrc, algoMemHandler, sizeX, sizeY, src_pitch);     // convert interleaved to planar format (range 0.f ... 225.f)
        MosaicAlgorithmMain (algoMemHandler, sizeX, sizeY, cellsNumber);    // perform SLIC algorithm
        planar2rgb (localSrc, algoMemHandler, localDst, sizeX, sizeY, src_pitch, dst_pitch); // back convert from planar to interleaved format

        free_memory_buffers (algoMemHandler);
    }
    else
    {
        err = PF_Err_OUT_OF_MEMORY;
    }

    return err;
}


inline PF_Err ArtMosaic_InAE_DeepWorld
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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_INPUT)]->u.ld), &format))
    {
        err = (format == PF_PixelFormat_ARGB128 ?
            ArtMosaic_InAE_32bits (in_data, out_data, params, output) : ArtMosaic_InAE_16bits (in_data, out_data, params, output));
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
        ArtMosaic_InAE_DeepWorld (in_data, out_data, params, output) :
        ArtMosaic_InAE_8bits (in_data, out_data, params, output));
}