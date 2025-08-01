#include "AlgoRules.hpp"
#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureAlgo.hpp"
#include "CompileTimeUtils.hpp"
#include "CommonAuxPixFormat.hpp"
#include "ImageLabMemInterface.hpp"
#include "cct_interface.hpp"


PF_Err ColorTemperature_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
    PF_Err err = PF_Err_OUT_OF_MEMORY;

    const PF_EffectWorld*   __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);
    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
          PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

    // --- Acquire controls values --- //
    const strControlSet cctSetup = GetCctSetup(params);
    const AlgoProcT targetCct    = cctSetup.Cct;
    const AlgoProcT targetDuv    = cctSetup.Duv;
    const eCOLOR_OBSERVER observer = static_cast<eCOLOR_OBSERVER>(cctSetup.observer);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    const pHandle* pStr = static_cast<const pHandle*>(GET_OBJ_FROM_HNDL(in_data->global_data));
    if (nullptr != pStr)
    {
        AlgoCCT::CctHandleF32* cctHandle = pStr->hndl;
        if (nullptr != cctHandle)
        {
            void* pMemoryBlock = nullptr;
            const A_long totalProcMem = CreateAlignment(sizeX * sizeY * static_cast<A_long>(sizeof(PixComponentsStr32)), CACHE_LINE);
            A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

            if (nullptr != pMemoryBlock)
            {
                PixComponentsStr<AlgoProcT>* __restrict pTmpBuffer = static_cast<PixComponentsStr<AlgoProcT>* __restrict>(pMemoryBlock);
                constexpr AlgoProcT coeff = static_cast<AlgoProcT>(1) / static_cast<AlgoProcT>(u8_value_white);
                const std::pair<AlgoProcT, AlgoProcT> uv = Convert2PixComponents(localSrc, pTmpBuffer, sizeX, sizeY, src_pitch, dst_pitch, coeff);
                const std::pair<AlgoProcT, AlgoProcT> cct_duv = cctHandle->ComputeCct(uv, observer);


                ::FreeMemoryBlock(blockId);
                blockId = -1;
                pMemoryBlock = nullptr;

                err = PF_Err_NONE;
            } // if (nullptr != pMemoryBlock)

        } // if (nullptr != cctHandle)

    } // if (nullptr != pStr)

    return err;
}


PF_Err ColorTemperature_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) noexcept
{
    PF_Err err = PF_Err_OUT_OF_MEMORY;

    const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);
    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

    // --- Acquire controls values --- //
    const strControlSet cctSetup = GetCctSetup(params);
    const AlgoProcT targetCct    = cctSetup.Cct;
    const AlgoProcT targetDuv    = cctSetup.Duv;
    const eCOLOR_OBSERVER observer = static_cast<eCOLOR_OBSERVER>(cctSetup.observer);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    const pHandle* pStr = static_cast<const pHandle*>(GET_OBJ_FROM_HNDL(in_data->global_data));
    if (nullptr != pStr)
    {
        AlgoCCT::CctHandleF32* cctHandle = pStr->hndl;
        if (nullptr != cctHandle)
        {
            void* pMemoryBlock = nullptr;
            const A_long totalProcMem = CreateAlignment(sizeX * sizeY * static_cast<A_long>(sizeof(PixComponentsStr32)), CACHE_LINE);
            A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

            if (nullptr != pMemoryBlock)
            {
                PixComponentsStr<AlgoProcT>* __restrict pTmpBuffer = static_cast<PixComponentsStr<AlgoProcT>* __restrict>(pMemoryBlock);

                constexpr AlgoProcT coeff = static_cast<AlgoProcT>(1) / static_cast<AlgoProcT>(u16_value_white);
                const std::pair<AlgoProcT, AlgoProcT> uv = Convert2PixComponents(localSrc, pTmpBuffer, sizeX, sizeY, src_pitch, dst_pitch, coeff);
                const std::pair<AlgoProcT, AlgoProcT> cct_duv = cctHandle->ComputeCct(uv, observer);

                ::FreeMemoryBlock(blockId);
                blockId = -1;
                pMemoryBlock = nullptr;

                err = PF_Err_NONE;
            } // if (nullptr != pMemoryBlock)

        } // if (nullptr != cctHandle)

    } // if (nullptr != pStr)
    return err;
}


PF_Err ColorTemperature_InAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) noexcept
{
    PF_Err err = PF_Err_OUT_OF_MEMORY;

    const PF_EffectWorld*    __restrict input    = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);
    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    // --- Acquire controls values --- //
    const strControlSet cctSetup = GetCctSetup(params);
    const AlgoProcT targetCct    = cctSetup.Cct;
    const AlgoProcT targetDuv    = cctSetup.Duv;
    const eCOLOR_OBSERVER observer = static_cast<eCOLOR_OBSERVER>(cctSetup.observer);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    const pHandle* pStr = static_cast<const pHandle*>(GET_OBJ_FROM_HNDL(in_data->global_data));
    if (nullptr != pStr)
    {
        AlgoCCT::CctHandleF32* cctHandle = pStr->hndl;
        if (nullptr != cctHandle)
        {
            void* pMemoryBlock = nullptr;
            const A_long totalProcMem = CreateAlignment(sizeX * sizeY * static_cast<A_long>(sizeof(PixComponentsStr32)), CACHE_LINE);
            A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

            if (nullptr != pMemoryBlock)
            {
                PixComponentsStr<AlgoProcT>* __restrict pTmpBuffer = static_cast<PixComponentsStr<AlgoProcT>* __restrict>(pMemoryBlock);

                constexpr AlgoProcT coeff = static_cast<AlgoProcT>(1);
                const std::pair<AlgoProcT, AlgoProcT> uv = Convert2PixComponents(localSrc, pTmpBuffer, sizeX, sizeY, src_pitch, dst_pitch, coeff);
                const std::pair<AlgoProcT, AlgoProcT> cct_duv = cctHandle->ComputeCct(uv, observer);

                ::FreeMemoryBlock(blockId);
                blockId = -1;
                pMemoryBlock = nullptr;

                err = PF_Err_NONE;
            } // if (nullptr != pMemoryBlock)

        } // if (nullptr != cctHandle)

    } // if (nullptr != pStr)

    return err;
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