#include "RetroVision.hpp"
#include "RetroVisionEnum.hpp"
#include "RetroVisionAlgorithm.hpp"
#include "AdjustGamma.hpp"
#include "ImageLabMemInterface.hpp"


PF_Err RetroVision_InAE_8bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) 
{
          PF_EffectWorld*   __restrict input    = reinterpret_cast<      PF_EffectWorld*   __restrict>(&params[UnderlyingType(RetroVision::eRETRO_VISION_INPUT)]->u.ld);
    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
          PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    // check if effect is enabled by CheckBox
    const A_long isEnabled = params[UnderlyingType(RetroVision::eRETRO_VISION_ENABLE)]->u.bd.value;
    const float fGamma = static_cast<float>(params[UnderlyingType(RetroVision::eRETRO_VISION_GAMMA_ADJUST)]->u.fs_d.value);
    constexpr float fCoeff{ static_cast<float>(u8_value_white) };

    if (0 == isEnabled)
    {
        if (true == FloatEqual(fGamma, 1.0f))
        {
            auto const& worldTransformSuite{ AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data) };
            return worldTransformSuite->copy(in_data->effect_ref, input, output, NULL, NULL);
        }
        else
        {
            return AdjustGammaValue(localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, fGamma, fCoeff);
        }
    } // if (0 == isEnabled)

    PF_Err err{ PF_Err_NONE };

    // Allocate memory storage for store temporary results
    const A_long singleTmpFrameSize = sizeX * sizeY;
    constexpr A_long doubleBuf = 2 * static_cast<A_long>(sizeof(fRGB));
    const A_long totalProcMem = CreateAlignment(singleTmpFrameSize * doubleBuf, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock (totalProcMem, 0, &pMemoryBlock);
    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        // get rest of the control parameters
        const RVControls controlParams = GetControlParametersStruct(params);

        fRGB* __restrict pTmpBuf1 = static_cast<fRGB* __restrict>(pMemoryBlock);
        fRGB* __restrict pTmpBuf2 = pTmpBuf1 + singleTmpFrameSize;

        AdjustGammaValueToProc (localSrc, pTmpBuf1, sizeX, sizeY, src_pitch, sizeX, fGamma, fCoeff);
        const fRGB* outProc = RetroResolution_Simulation (pTmpBuf1, pTmpBuf2, sizeX, sizeY, controlParams);
        RestoreImage (localSrc, outProc, localDst, sizeX, sizeY, src_pitch, dst_pitch, fCoeff);

        pMemoryBlock = nullptr;
        pTmpBuf1 = pTmpBuf2 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;

        err = PF_Err_NONE;
    }
    else
        err = PF_Err_OUT_OF_MEMORY;

	return err;
}


PF_Err RetroVision_InAE_16bits
(
	PF_InData*   in_data,
	PF_OutData*  out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output
) 
{
             PF_EffectWorld*  __restrict input   = reinterpret_cast<      PF_EffectWorld*   __restrict>(&params[UnderlyingType(RetroVision::eRETRO_VISION_INPUT)]->u.ld);
    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
          PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    // check if effect is enabled by CheckBox
    const A_long isEnabled = params[UnderlyingType(RetroVision::eRETRO_VISION_ENABLE)]->u.bd.value;
    const float fGamma = static_cast<float>(params[UnderlyingType(RetroVision::eRETRO_VISION_GAMMA_ADJUST)]->u.fs_d.value);
    constexpr float fCoeff{ static_cast<float>(u16_value_white) };

    if (0 == isEnabled)
    {
        if (true == FloatEqual(fGamma, 1.0f))
        {
            auto const& worldTransformSuite{ AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data) };
            return worldTransformSuite->copy(in_data->effect_ref, input, output, NULL, NULL);
        }
        else
        {
            return AdjustGammaValue (localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, fGamma, fCoeff);
        }
    } // if (0 == isEnabled)


    PF_Err err{ PF_Err_NONE };

    // Allocate memory storage for store temporary results
    const A_long singleTmpFrameSize = sizeX * sizeY;
    constexpr A_long doubleBuf = 2 * static_cast<A_long>(sizeof(fRGB));
    const A_long totalProcMem = CreateAlignment(singleTmpFrameSize * doubleBuf, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock (totalProcMem, 0, &pMemoryBlock);
    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        // get rest of the control parameters
        const RVControls controlParams = GetControlParametersStruct(params);

        fRGB* __restrict pTmpBuf1 = static_cast<fRGB* __restrict>(pMemoryBlock);
        fRGB* __restrict pTmpBuf2 = pTmpBuf1 + singleTmpFrameSize;

        AdjustGammaValueToProc (localSrc, pTmpBuf1, sizeX, sizeY, src_pitch, sizeX, fGamma, fCoeff);
        const fRGB* outProc = RetroResolution_Simulation (pTmpBuf1, pTmpBuf2, sizeX, sizeY, controlParams);
        RestoreImage (localSrc, outProc, localDst, sizeX, sizeY, src_pitch, dst_pitch, fCoeff);

        pMemoryBlock = nullptr;
        pTmpBuf1 = pTmpBuf2 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;

        err = PF_Err_NONE;
    }
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}


PF_Err RetroVision_InAE_32bits
(
    PF_InData*   in_data,
    PF_OutData*  out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output
) 
{
          PF_EffectWorld*  __restrict input      = reinterpret_cast<      PF_EffectWorld*    __restrict>(&params[UnderlyingType(RetroVision::eRETRO_VISION_INPUT)]->u.ld);
    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
          PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

    const A_long src_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long sizeY = output->height;
    const A_long sizeX = output->width;

    // check if effect is enabled by CheckBox
    const A_long isEnabled = params[UnderlyingType(RetroVision::eRETRO_VISION_ENABLE)]->u.bd.value;
    const float fGamma = static_cast<float>(params[UnderlyingType(RetroVision::eRETRO_VISION_GAMMA_ADJUST)]->u.fs_d.value);
    constexpr float fCoeff = f32_value_white;

    if (0 == isEnabled)
    {
        if (true == FloatEqual(fGamma, 1.0f))
        {
            auto const& worldTransformSuite{ AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data) };
            return worldTransformSuite->copy(in_data->effect_ref, input, output, NULL, NULL);
        }
        else
        {
            return AdjustGammaValue (localSrc, localDst, sizeX, sizeY, src_pitch, dst_pitch, fGamma, fCoeff);
        }
    } // if (0 == isEnabled)

    PF_Err err{ PF_Err_NONE };

    // Allocate memory storage for store temporary results
    const A_long singleTmpFrameSize = sizeX * sizeY;
    constexpr A_long doubleBuf = 2 * static_cast<A_long>(sizeof(fRGB));
    const A_long totalProcMem = CreateAlignment(singleTmpFrameSize * doubleBuf, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock (totalProcMem, 0, &pMemoryBlock);
    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        // get rest of the control parameters
        const RVControls controlParams = GetControlParametersStruct(params);

        fRGB* __restrict pTmpBuf1 = static_cast<fRGB* __restrict>(pMemoryBlock);
        fRGB* __restrict pTmpBuf2 = pTmpBuf1 + singleTmpFrameSize;

        AdjustGammaValueToProc (localSrc, pTmpBuf1, sizeX, sizeY, src_pitch, sizeX, fGamma, fCoeff);
        const fRGB* outProc = RetroResolution_Simulation (pTmpBuf1, pTmpBuf2, sizeX, sizeY, controlParams);
        RestoreImage (localSrc, outProc, localDst, sizeX, sizeY, src_pitch, dst_pitch, fCoeff);

        pMemoryBlock = nullptr;
        pTmpBuf1 = pTmpBuf2 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;

        err = PF_Err_NONE;
    }
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}


inline PF_Err RetroVision_InAE_DeepWorld
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
    if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[UnderlyingType(RetroVision::eRETRO_VISION_INPUT)]->u.ld), &format))
    {
        err = (format == PF_PixelFormat_ARGB128 ?
            RetroVision_InAE_32bits(in_data, out_data, params, output) : RetroVision_InAE_16bits(in_data, out_data, params, output));
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
        RetroVision_InAE_DeepWorld (in_data, out_data, params, output) :
		RetroVision_InAE_8bits (in_data, out_data, params, output));
}