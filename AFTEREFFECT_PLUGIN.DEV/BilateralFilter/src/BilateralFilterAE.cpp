#include "BilateralFilter.hpp"
#include "BilateralFilterAlgo.hpp"
#include "BilateralFilterEnum.hpp"
#include "ImageLabMemInterface.hpp"
#include "PrSDKAESupport.h"


PF_Err BilateralFilter_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<PF_EffectWorld* __restrict>(&params[eBILATERAL_FILTER_INPUT]->u.ld);

    // Get Bilateral Filter Radius value from slider
    const A_long sliderFilterRadius = params[eBILATERAL_FILTER_RADIUS]->u.sd.value;
    if (0 == sliderFilterRadius) // Filter Radius equal to zero, so algorithm disabled  - let's make simple copy
    {
        auto const& worldTransformSuite{ AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data) };
        return worldTransformSuite->copy (in_data->effect_ref, input, output, NULL, NULL);
    }
    const A_FpLong sliderSigmaValue = params[eBILATERAL_FILTER_RADIUS]->u.fs_d.value;
    const float fSigmaValue = CLAMP_VALUE(static_cast<float>(sliderSigmaValue), fSigmaValMin, fSigmaValMax);

    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
          PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    // Get memory block
    void* pMemoryBlock = nullptr;
    const A_long totalProcMem = CreateAlignment(sizeX * sizeY * static_cast<A_long>(fCIELabPix_size), CACHE_LINE);
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
#ifdef _DEBUG
        memset(pMemoryBlock, 0, totalProcMem); // cleanup memory block for DBG purposes
#endif
        fCIELabPix* __restrict pCIELab = reinterpret_cast<fCIELabPix* __restrict>(pMemoryBlock);
 
        constexpr PF_Pixel_ARGB_8u white{ u8_value_white , u8_value_white , u8_value_white , u8_value_white };
        constexpr PF_Pixel_ARGB_8u black{ u8_value_black , u8_value_black , u8_value_black , u8_value_black };

        // Convert from RGB to CIE-Lab color space
        Rgb2CIELab (localSrc, pCIELab, sizeX, sizeY, src_pitch, sizeX);
        BilateralFilterAlgorithm (pCIELab, localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, sliderFilterRadius, fSigmaValue, black, white);

        ::FreeMemoryBlock(blockId);
        blockId = -1;
        pMemoryBlock = nullptr;
    } // if (nullptr != pMemoryBlock && blockId >= 0)

    return PF_Err_NONE;
}


PF_Err BilateralFilter_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<PF_EffectWorld* __restrict>(&params[eBILATERAL_FILTER_INPUT]->u.ld);

    // Get Bilateral Filter Radius value from slider
    const A_long sliderFilterRadius = params[eBILATERAL_FILTER_RADIUS]->u.sd.value;
    if (0 == sliderFilterRadius) // Filter Radius equal to zero, so algorithm disabled  - let's make simple copy
    {
        auto const& worldTransformSuite{ AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data) };
        return worldTransformSuite->copy_hq (in_data->effect_ref, input, output, NULL, NULL);
    }
    const A_FpLong sliderSigmaValue = params[eBILATERAL_FILTER_RADIUS]->u.fs_d.value;
    const float fSigmaValue = CLAMP_VALUE(static_cast<float>(sliderSigmaValue), fSigmaValMin, fSigmaValMax);

    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    // Get memory block
    void* pMemoryBlock = nullptr;
    const A_long totalProcMem = CreateAlignment(sizeX * sizeY * static_cast<A_long>(fCIELabPix_size), CACHE_LINE);
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
#ifdef _DEBUG
        memset(pMemoryBlock, 0, totalProcMem); // cleanup memory block for DBG purposes
#endif
        fCIELabPix* __restrict pCIELab = reinterpret_cast<fCIELabPix* __restrict>(pMemoryBlock);

        constexpr PF_Pixel_ARGB_16u white{ u16_value_white , u16_value_white , u16_value_white , u16_value_white };
        constexpr PF_Pixel_ARGB_16u black{ u16_value_black , u16_value_black , u16_value_black , u16_value_black };

        // Convert from RGB to CIE-Lab color space
        Rgb2CIELab (localSrc, pCIELab, sizeX, sizeY, src_pitch, sizeX);
        BilateralFilterAlgorithm (pCIELab, localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, sliderFilterRadius, fSigmaValue, black, white);

        ::FreeMemoryBlock(blockId);
        blockId = -1;
        pMemoryBlock = nullptr;
    } // if (nullptr != pMemoryBlock && blockId >= 0)

    return PF_Err_NONE;
}


PF_Err BilateralFilter_InAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    PF_EffectWorld*   __restrict input = reinterpret_cast<PF_EffectWorld* __restrict>(&params[eBILATERAL_FILTER_INPUT]->u.ld);

    // Get Bilateral Filter Radius value from slider
    const A_long sliderFilterRadius = params[eBILATERAL_FILTER_RADIUS]->u.sd.value;
    if (0 == sliderFilterRadius) // Filter Radius equal to zero, so algorithm disabled  - let's make simple copy
    {
        auto const& worldTransformSuite{ AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data) };
        return worldTransformSuite->copy_hq(in_data->effect_ref, input, output, NULL, NULL);
    }
    const A_FpLong sliderSigmaValue = params[eBILATERAL_FILTER_RADIUS]->u.fs_d.value;
    const float fSigmaValue = CLAMP_VALUE(static_cast<float>(sliderSigmaValue), fSigmaValMin, fSigmaValMax);

    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    // Get memory block
    void* pMemoryBlock = nullptr;
    const A_long totalProcMem = CreateAlignment(sizeX * sizeY * static_cast<A_long>(fCIELabPix_size), CACHE_LINE);
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

    if (nullptr != pMemoryBlock && blockId >= 0)
    {
#ifdef _DEBUG
        memset(pMemoryBlock, 0, totalProcMem); // cleanup memory block for DBG purposes
#endif
        fCIELabPix* __restrict pCIELab = reinterpret_cast<fCIELabPix* __restrict>(pMemoryBlock);

        constexpr PF_Pixel_ARGB_32f white{ f32_value_white , f32_value_white , f32_value_white , f32_value_white };
        constexpr PF_Pixel_ARGB_32f black{ f32_value_black , f32_value_black , f32_value_black , f32_value_black };

        // Convert from RGB to CIE-Lab color space
        Rgb2CIELab (localSrc, pCIELab, sizeX, sizeY, src_pitch, sizeX);
        BilateralFilterAlgorithm (pCIELab, localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, sliderFilterRadius, fSigmaValue, black, white);

        ::FreeMemoryBlock(blockId);
        blockId = -1;
        pMemoryBlock = nullptr;
    } // if (nullptr != pMemoryBlock && blockId >= 0)

    return PF_Err_NONE;
}


inline PF_Err BilateralFilter_InAE_DeepWorld
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    PF_PixelFormat format = PF_PixelFormat_INVALID;
    AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[eBILATERAL_FILTER_INPUT]->u.ld), &format))
    {
        return (format == PF_PixelFormat_ARGB128 ?
            BilateralFilter_InAE_32bits (in_data, out_data, params, output) : BilateralFilter_InAE_16bits (in_data, out_data, params, output));
    }
    return PF_Err_UNRECOGNIZED_PARAM_TYPE;
}


PF_Err
ProcessImgInAE
(
	PF_InData*		in_data,
	PF_OutData*		out_data,
	PF_ParamDef*	params[],
	PF_LayerDef*	output
) noexcept
{
#if !defined __INTEL_COMPILER 
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	return (PF_WORLD_IS_DEEP(output) ?
        BilateralFilter_InAE_DeepWorld (in_data, out_data, params, output) :
		BilateralFilter_InAE_8bits (in_data, out_data, params, output));
}