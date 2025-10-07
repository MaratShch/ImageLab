#include "RetroVision.hpp"
#include "RetroVisionEnum.hpp"
#include "RetroVisionAlgorithm.hpp"
#include "CommonSmartRender.hpp"
#include "AdjustGamma.hpp"
#include "ImageLabMemInterface.hpp"


template<typename T, typename U, typename std::enable_if<is_RGB_proc<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
PF_Err RetroVision_SmartRenderAlgorithm
(
    const T* __restrict localSrc,
          T* __restrict localDst,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    const U whiteLevel,
    const RVControls* __restrict controlParams
)
{
    PF_Err err = PF_Err_NONE;
    
    // check if effect is enabled by CheckBox
    const A_long isEnabled = controlParams->enable;
    const float fGamma = controlParams->gamma;

    if (0 == isEnabled)
        return AdjustGammaValue (localSrc, localDst, sizeX, sizeY, srcPitch, dstPitch, fGamma, whiteLevel);

    // Allocate memory storage for store temporary results
    const A_long singleTmpFrameSize = sizeX * sizeY;
    constexpr A_long doubleBuf = 2 * static_cast<A_long>(sizeof(fRGB));
    const A_long totalProcMem = CreateAlignment(singleTmpFrameSize * doubleBuf, CACHE_LINE);

    void* pMemoryBlock = nullptr;
    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);
    if (nullptr != pMemoryBlock && blockId >= 0)
    {
        fRGB* __restrict pTmpBuf1 = static_cast<fRGB* __restrict>(pMemoryBlock);
        fRGB* __restrict pTmpBuf2 = pTmpBuf1 + singleTmpFrameSize;

        AdjustGammaValueToProc (localSrc, pTmpBuf1, sizeX, sizeY, srcPitch, sizeX, fGamma, whiteLevel);
        const fRGB* outProc = RetroResolution_Simulation (pTmpBuf1, pTmpBuf2, sizeX, sizeY, *controlParams);
        RestoreImage (localSrc, outProc, localDst, sizeX, sizeY, srcPitch, dstPitch, whiteLevel);

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



PF_Err
RetroVision_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
) 
{
    PF_Err err = PF_Err_NONE;
    PF_Err errParam = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(RVControlsSize);
    if (nullptr != paramsHandler)
    {
        RVControls* paramsStrP = reinterpret_cast<RVControls*>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            PF_ParamDef algoParam{};

            extra->output->pre_render_data = paramsHandler;

            // -------------- Start read Effect Parameters Set ---------------- //

            // Read Algo Enable flag
            errParam = PF_CHECKOUT_PARAM (in_data, UnderlyingType(RetroVision::eRETRO_VISION_ENABLE), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->enable = algoParam.u.bd.value;
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read Gamma value for adjust
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_GAMMA_ADJUST), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->gamma = algoParam.u.fs_d.value;
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read Retro-monitor type for simulation
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_DISPLAY), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->monitor = static_cast<RetroMonitor>(algoParam.u.pd.value);
            else
                errParam = PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CGA Palette type for simulation
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CGA_PALETTE), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->cga_palette = static_cast<PaletteCGA>(algoParam.u.pd.value);
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CGA Intencity bit
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CGA_INTTENCITY_BIT), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->cga_intencity_bit = algoParam.u.bd.value;
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read EGA Palette type for simulation
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_EGA_PALETTE), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->ega_palette = static_cast<PaletteEGA>(algoParam.u.pd.value);
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read VGA Palette type for simulation
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_VGA_PALETTE), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->vga_palette = static_cast<PaletteVGA>(algoParam.u.pd.value);
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read Hercules B/W Threshold for simulation
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_HERCULES_THRESHOLD), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->hercules_threshold = algoParam.u.sd.value;
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CRT Artifacts - Scan Line enable alfgorithm enable
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->scan_lines_enable = algoParam.u.bd.value;
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CRT Artifacts - Smooth Scan Line
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SMOOTH_SCANLINES), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->scan_lines_smooth = algoParam.u.bd.value;
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CRT Artifacts - Scan Lines interval
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_INTERVAL), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->scan_lines_interval = algoParam.u.sd.value;
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CRT Artifacts - Scan Lines darkness
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_SCANLINES_DARKNESS), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->scan_lines_darkness = static_cast<float>(algoParam.u.fs_d.value);
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CRT Artifacts - Phosphor Glow enable algo
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->phosphor_glow_enable = algoParam.u.bd.value;
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CRT Artifacts - Phosphor Glow strength
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_STRENGHT), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->phosphor_glow_strength = static_cast<float>(algoParam.u.fs_d.value);
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CRT Artifacts - Phosphor Glow opacity
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_PHOSPHOR_GLOW_OPACITY), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->phosphor_glow_opacity = static_cast<float>(algoParam.u.fs_d.value);
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CRT Artifacts - Aperture Grill algorithm enable
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->apperture_grill_enable = algoParam.u.bd.value;
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CRT Artifacts - Aperture Grill / Mask type
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_POPUP), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->mask_type = static_cast<AppertureGtrill>(algoParam.u.pd.value);
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CRT Artifacts - Aperture Grill / Mask type
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_INTERVAL), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->mask_interval = algoParam.u.sd.value;
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Read CRT Artifacts - Aperture Grill / Mask darkness
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_APPERTURE_GRILL_DARKNESS), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->mask_darkness = static_cast<float>(algoParam.u.fs_d.value);
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            // Hercules white color tint
            errParam = PF_CHECKOUT_PARAM(in_data, UnderlyingType(RetroVision::eRETRO_VISION_CRT_ARTIFACTS_HERCULES_WHITE_COLOR), in_data->current_time, in_data->time_step, in_data->time_scale, &algoParam);
            if (PF_Err_NONE == errParam)
                paramsStrP->white_color_hercules = static_cast<HerculesWhiteColor>(algoParam.u.pd.value);
            else
                errParam |= PF_Err_INTERNAL_STRUCT_DAMAGED;

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
            (
                in_data->effect_ref, 
                UnderlyingType(RetroVision::eRETRO_VISION_INPUT), 
                UnderlyingType(RetroVision::eRETRO_VISION_INPUT), 
                &req, 
                in_data->current_time, 
                in_data->time_step, 
                in_data->time_scale, 
                &in_result
            ));

            UnionLRect(&in_result.result_rect, &extra->output->result_rect);
            UnionLRect(&in_result.max_result_rect, &extra->output->max_result_rect);

            handleSuite->host_unlock_handle(paramsHandler);

        } //  if (nullptr != paramsStrP)
        else
            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    } // if (nullptr != paramsHandler)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return (PF_Err_NONE == errParam ? err : PF_Err_INTERNAL_STRUCT_DAMAGED);
}


PF_Err
RetroVision_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
)
{
    PF_EffectWorld* input_worldP  = nullptr;
    PF_EffectWorld* output_worldP = nullptr;
    PF_Err	err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    const RVControls* pFilterStrParams = reinterpret_cast<const RVControls*>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));

    if (nullptr != pFilterStrParams)
    {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, UnderlyingType(RetroVision::eRETRO_VISION_INPUT), &input_worldP)));
        ERR( extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (nullptr != input_worldP && nullptr != output_worldP)
        {
            const A_long sizeX = input_worldP->width;
            const A_long sizeY = input_worldP->height;
            const A_long srcRowBytes = input_worldP->rowbytes;  // Get input buffer pitch in bytes
            const A_long dstRowBytes = output_worldP->rowbytes; // Get output buffer pitch in bytes

            PF_PixelFormat format = PF_PixelFormat_INVALID;
            AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);

            if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))
            {
                switch (format)
                {
                    case PF_PixelFormat_ARGB128:
                    {
                        constexpr float whiteLevel = f32_value_white;
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                        const PF_Pixel_ARGB_32f* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input_worldP->data);
                              PF_Pixel_ARGB_32f* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output_worldP->data);

                        err = RetroVision_SmartRenderAlgorithm
                        (
                            input_pixels,
                            output_pixels,
                            sizeX,
                            sizeY,
                            srcRowBytes,
                            dstRowBytes,
                            whiteLevel,
                            pFilterStrParams
                        );
                    }
                    break;

                    case PF_PixelFormat_ARGB64:
                    {
                        constexpr float whiteLevel = static_cast<float>(u16_value_white);
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                        const PF_Pixel_ARGB_16u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                              PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                        err = RetroVision_SmartRenderAlgorithm
                        (
                            input_pixels,
                            output_pixels,
                            sizeX,
                            sizeY,
                            srcRowBytes,
                            dstRowBytes,
                            whiteLevel,
                            pFilterStrParams
                        );
                    }
                    break;

                    case PF_PixelFormat_ARGB32:
                    {
                        constexpr float whiteLevel = static_cast<float>(u8_value_white);
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                        const PF_Pixel_ARGB_8u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                              PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                        err = RetroVision_SmartRenderAlgorithm
                        (
                            input_pixels,
                            output_pixels,
                            sizeX,
                            sizeY,
                            srcRowBytes,
                            dstRowBytes,
                            whiteLevel,
                            pFilterStrParams
                        );
                    }
                    break;

                    default:
                        err = PF_Err_BAD_CALLBACK_PARAM;
                    break;
                } // switch (format)

            } //  if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)

    return PF_Err_NONE;
}