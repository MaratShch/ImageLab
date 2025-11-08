#include "ArtPointilism.hpp"
#include "ArtPointilismAlgo.hpp"
#include "ArtPointilismEnums.hpp"
#include "ImageLabMemInterface.hpp"

PF_Err ArtPointilism_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
)
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<      PF_EffectWorld*   __restrict>(&params[UnderlyingType(ArtPointilism::ART_POINTILISM_INPUT)]->u.ld);
    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
          PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    // Allocate memory storage for store temporary results
    constexpr A_long doubleBuf = 2 * FastCompute::Max(static_cast<A_long>(sizeof(fRGB)), static_cast<A_long>(sizeof(fCIELabPix)));
    const A_long singleTmpFrameSize = sizeX * sizeY;
    const A_long totalProcMem = CreateAlignment(singleTmpFrameSize * doubleBuf, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);
    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fRGB* pTmpBuf1 = static_cast<fRGB*>(pMemoryBlock);
        fRGB* pTmpBuf2 = pTmpBuf1 + singleTmpFrameSize;
        fCIELabPix* pCieLabBuf = static_cast<fCIELabPix*>(pMemoryBlock); // Aliased pointer!!!

        // convert to CieLAB color space
        ConvertToCIELab (localSrc, pCieLabBuf, sizeX, sizeY, src_pitch, sizeX);

        // back convert to native buffer format after processing complete
        ConvertFromCIELab (localSrc, pCieLabBuf, localDst, sizeX, sizeY, src_pitch, sizeX, dst_pitch);

        pMemoryBlock = nullptr;
        pTmpBuf1 = pTmpBuf2 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;
    }

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
    PF_EffectWorld*   __restrict input = reinterpret_cast<      PF_EffectWorld*   __restrict>(&params[UnderlyingType(ArtPointilism::ART_POINTILISM_INPUT)]->u.ld);
    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;
    
    // Allocate memory storage for store temporary results
    constexpr A_long doubleBuf = 2 * FastCompute::Max(static_cast<A_long>(sizeof(fRGB)), static_cast<A_long>(sizeof(fCIELabPix)));
    const A_long singleTmpFrameSize = sizeX * sizeY;
    const A_long totalProcMem = CreateAlignment(singleTmpFrameSize * doubleBuf, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);
    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fRGB* pTmpBuf1 = static_cast<fRGB*>(pMemoryBlock);
        fRGB* pTmpBuf2 = pTmpBuf1 + singleTmpFrameSize;
        fCIELabPix* pCieLabBuf = static_cast<fCIELabPix*>(pMemoryBlock); // Aliased pointer!!!

        // convert to CieLAB color space
        ConvertToCIELab (localSrc, pCieLabBuf, sizeX, sizeY, src_pitch, sizeX);

        // back convert to native buffer format after processing complete
        ConvertFromCIELab (localSrc, pCieLabBuf, localDst, sizeX, sizeY, src_pitch, sizeX, dst_pitch);

        pMemoryBlock = nullptr;
        pTmpBuf1 = pTmpBuf2 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;
    } // if (nullptr != pMemoryBlock && blockId >= 0)

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
    PF_EffectWorld*   __restrict input = reinterpret_cast<      PF_EffectWorld*   __restrict>(&params[UnderlyingType(ArtPointilism::ART_POINTILISM_INPUT)]->u.ld);
    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    // Allocate memory storage for store temporary results
    constexpr A_long doubleBuf = 2 * FastCompute::Max(static_cast<A_long>(sizeof(fRGB)), static_cast<A_long>(sizeof(fCIELabPix)));
    const A_long singleTmpFrameSize = sizeX * sizeY;
    const A_long totalProcMem = CreateAlignment(singleTmpFrameSize * doubleBuf, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);
    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fRGB* pTmpBuf1 = static_cast<fRGB*>(pMemoryBlock);
        fRGB* pTmpBuf2 = pTmpBuf1 + singleTmpFrameSize;
        fCIELabPix* pCieLabBuf = static_cast<fCIELabPix*>(pMemoryBlock); // Aliased pointer!!!

        // convert to CieLAB color space
        ConvertToCIELab (localSrc, pCieLabBuf, sizeX, sizeY, src_pitch, sizeX);

        // back convert to native buffer format after processing complete
        ConvertFromCIELab (localSrc, pCieLabBuf, localDst, sizeX, sizeY, src_pitch, sizeX, dst_pitch);

        pMemoryBlock = nullptr;
        pTmpBuf1 = pTmpBuf2 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;
    } // if (nullptr != pMemoryBlock && blockId >= 0)

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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[UnderlyingType(ArtPointilism::ART_POINTILISM_INPUT)]->u.ld), &format))
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