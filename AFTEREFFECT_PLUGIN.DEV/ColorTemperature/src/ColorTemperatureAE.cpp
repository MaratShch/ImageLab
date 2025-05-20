#include "AlgoRules.hpp"
#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureAlgo.hpp"
#include "CompileTimeUtils.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ImageLabMemInterface.hpp"


PF_Err ColorTemperature_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
    const PF_EffectWorld*   __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);
    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
          PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    void* pMemoryBlock = nullptr;
    const A_long totalProcMem = CreateAlignment(sizeX * sizeY * static_cast<A_long>(sizeof(PixComponentsStr32)), CACHE_LINE);
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);
    PixComponentsStr<AlgoProcT>* __restrict pTmpBuffer = static_cast<PixComponentsStr<AlgoProcT>* __restrict>(pMemoryBlock);

    constexpr AlgoProcT coeff = static_cast<AlgoProcT>(1) / static_cast<AlgoProcT>(u8_value_white);
    std::pair<AlgoProcT, AlgoProcT> uv = Convert2PixComponents (localSrc, pTmpBuffer, sizeX, sizeY, src_pitch, dst_pitch, coeff);

    ::FreeMemoryBlock(blockId);
    blockId = -1;
    pMemoryBlock = nullptr;

    return PF_Err_NONE;

    return PF_Err_NONE;
}


PF_Err ColorTemperature_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
    const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);
    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    void* pMemoryBlock = nullptr;
    const A_long totalProcMem = CreateAlignment(sizeX * sizeY * static_cast<A_long>(sizeof(PixComponentsStr32)), CACHE_LINE);
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);
    PixComponentsStr<AlgoProcT>* __restrict pTmpBuffer = static_cast<PixComponentsStr<AlgoProcT>* __restrict>(pMemoryBlock);

    constexpr AlgoProcT coeff = static_cast<AlgoProcT>(1) / static_cast<AlgoProcT>(u16_value_white);
    std::pair<AlgoProcT, AlgoProcT> uv = Convert2PixComponents (localSrc, pTmpBuffer, sizeX, sizeY, src_pitch, dst_pitch, coeff);

    ::FreeMemoryBlock(blockId);
    blockId = -1;
    pMemoryBlock = nullptr;

    return PF_Err_NONE;

    return PF_Err_NONE;
}


PF_Err ColorTemperature_InAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);
    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    void* pMemoryBlock = nullptr;
    const A_long totalProcMem = CreateAlignment(sizeX * sizeY * static_cast<A_long>(sizeof(PixComponentsStr32)), CACHE_LINE);
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);
    PixComponentsStr<AlgoProcT>* __restrict pTmpBuffer = static_cast<PixComponentsStr<AlgoProcT>* __restrict>(pMemoryBlock);

    constexpr AlgoProcT coeff = static_cast<AlgoProcT>(1);
    std::pair<AlgoProcT, AlgoProcT> uv = Convert2PixComponents (localSrc, pTmpBuffer, sizeX, sizeY, src_pitch, dst_pitch, coeff);

    ::FreeMemoryBlock(blockId);
    blockId = -1;
    pMemoryBlock = nullptr;

    return PF_Err_NONE;
}


PF_Err ColorTemperature_InAE_DeepWord
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    PF_Err	err = PF_Err_NONE;
    PF_PixelFormat format = PF_PixelFormat_INVALID;
    AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld), &format))
        err = (format == PF_PixelFormat_ARGB128 ?
            ColorTemperature_InAE_32bits(in_data, out_data, params, output) : ColorTemperature_InAE_16bits(in_data, out_data, params, output));
    else
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;

    return err;
}


PF_Err
ProcessImgInAE
(
	PF_InData*   in_data,
	PF_OutData*	 out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) 
{
	return (PF_WORLD_IS_DEEP(output) ?
        ColorTemperature_InAE_DeepWord(in_data, out_data, params, output) :
		ColorTemperature_InAE_8bits (in_data, out_data, params, output));
}