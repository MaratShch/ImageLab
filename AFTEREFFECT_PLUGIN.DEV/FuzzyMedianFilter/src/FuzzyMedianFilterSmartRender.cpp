#include "FuzzyMedianFilterSmartRender.hpp"
#include "FastAriphmetics.hpp"
#include "FuzzyMedianFilter.hpp"
#include "FuzzyMedianAlgo.hpp"
#include "CommonSmartRender.hpp"
#include "ImageLabMemInterface.hpp"
#include "AE_Effect.h"



PF_Err
FuzzyMedian_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
) noexcept
{
    FuzzyFilterParamsStr renderParams{};
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle (FuzzyFilterParamsStrSize);
    if (nullptr != paramsHandler)
    {
        FuzzyFilterParamsStr* paramsStrP = reinterpret_cast<FuzzyFilterParamsStr*>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            extra->output->pre_render_data = paramsHandler;

            PF_ParamDef	widowSize{};
            PF_ParamDef	filterSigma{};

            const PF_Err errParam1 = PF_CHECKOUT_PARAM(in_data, eFUZZY_MEDIAN_FILTER_KERNEL_SIZE, in_data->current_time, in_data->time_step, in_data->time_scale, &widowSize);
            const PF_Err errParam2 = PF_CHECKOUT_PARAM(in_data, eFUZZY_MEDIAN_FILTER_SIGMA_VALUE, in_data->current_time, in_data->time_step, in_data->time_scale, &filterSigma);

            if (PF_Err_NONE == errParam1 && PF_Err_NONE == errParam2)
            {
                paramsStrP->fWindowSize = CLAMP_VALUE(static_cast<eFUZZY_FILTER_WINDOW_SIZE>(widowSize.u.pd.value), eFUZZY_FILTER_BYPASSED, eFUZZY_FILTER_TOTAL_VARIANTS);
                paramsStrP->fSigma      = CLAMP_VALUE(static_cast<float>(filterSigma.u.fs_d.value), fSliderValMin, fSliderValMax);
            } // if (PF_Err_NONE == errParam1 && PF_Err_NONE == errParam2)
            else
            {
                paramsStrP->fWindowSize = eFUZZY_FILTER_BYPASSED;
                paramsStrP->fSigma = fSliderValDefault;
            }

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
            (in_data->effect_ref, eFUZZY_MEDIAN_FILTER_INPUT, eFUZZY_MEDIAN_FILTER_INPUT, &req, in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));

            UnionLRect (&in_result.result_rect, &extra->output->result_rect);
            UnionLRect (&in_result.max_result_rect, &extra->output->max_result_rect);

            handleSuite->host_unlock_handle (paramsHandler);

        } // if (nullptr != paramsStrP)
        else
            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    } // if (nullptr != paramsHandler)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}



PF_Err
FuzzyMedian_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
) noexcept
{
    PF_EffectWorld* input_worldP = nullptr;
    PF_EffectWorld* output_worldP = nullptr;
    PF_Err	err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PFuzzyFilterParamsStr pFilterStrParams = reinterpret_cast<PFuzzyFilterParamsStr>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));

    if (nullptr != pFilterStrParams)
    {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, eFUZZY_MEDIAN_FILTER_INPUT, &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (nullptr != input_worldP && nullptr != output_worldP)
        {
            const A_long sizeX = input_worldP->width;
            const A_long sizeY = input_worldP->height;
            const A_long srcRowBytes = input_worldP->rowbytes;  // Get input buffer pitch in bytes
            const A_long dstRowBytes = output_worldP->rowbytes; // Get output buffer pitch in bytes

            AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);

            const eFUZZY_FILTER_WINDOW_SIZE windowSize  = pFilterStrParams->fWindowSize;
            const float                     filterSigma = pFilterStrParams->fSigma;

            if (eFUZZY_FILTER_BYPASSED == windowSize)
            {
                err = PF_COPY(input_worldP, output_worldP, NULL, NULL);
            }
            else
            {
                const A_long frameSize = sizeX * sizeY;
                if (sizeX > 16 && sizeY > 16) // limit minimal fame size in Smart Render for processing by 256 pixels: 16 x 16
                {
                    void* pMemoryBlock = nullptr;
                    const A_long memoryBufSize = frameSize * static_cast<A_long>(fCIELabPix_size);
                    const A_long totalProcMem = CreateAlignment(memoryBufSize, CACHE_LINE);

                    A_long blockId = ::GetMemoryBlock(totalProcMem, 0, &pMemoryBlock);

                    if (nullptr != pMemoryBlock && blockId >= 0)
                    {
#ifdef _DEBUG
                        memset(pMemoryBlock, 0, totalProcMem); // cleanup memory block for DBG purposes
#endif
                        PF_PixelFormat format = PF_PixelFormat_INVALID;
                        if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))
                        {
                            fCIELabPix* __restrict pCIELab = reinterpret_cast<fCIELabPix* __restrict>(pMemoryBlock);

                            switch (format)
                            {
                            case PF_PixelFormat_ARGB128:
                                {
                                    constexpr PF_Pixel_ARGB_32f white{ f32_value_white , f32_value_white , f32_value_white , f32_value_white };
                                    constexpr PF_Pixel_ARGB_32f black{ f32_value_black , f32_value_black , f32_value_black , f32_value_black };
                                    const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                                    const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                                    const PF_Pixel_ARGB_32f* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input_worldP->data);
                                          PF_Pixel_ARGB_32f* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output_worldP->data);

                                    // Convert from RGB to CIE-Lab color space
                                    Rgb2CIELab (input_pixels, pCIELab, sizeX, sizeY, srcPitch, sizeX);

                                    // Perform Fuzzy Median Filter
                                    switch (windowSize)
                                    {
                                        case eFUZZY_FILTER_WINDOW_3x3:
                                            FuzzyLogic_3x3 (pCIELab, input_pixels, output_pixels, sizeX, sizeY, sizeX, srcPitch, dstPitch, black, white, filterSigma);
                                        break;
                                        case eFUZZY_FILTER_WINDOW_5x5:
                                            FuzzyLogic_5x5 (pCIELab, input_pixels, output_pixels, sizeX, sizeY, sizeX, srcPitch, dstPitch, black, white, filterSigma);
                                        break;
                                        case eFUZZY_FILTER_WINDOW_7x7:
                                            FuzzyLogic_7x7 (pCIELab, input_pixels, output_pixels, sizeX, sizeY, sizeX, srcPitch, dstPitch, black, white, filterSigma);
                                        break;
                                        default:
                                            err = PF_Err_INVALID_INDEX;
                                        break;
                                    } // switch (windowSize)
                                }
                                break;

                                case PF_PixelFormat_ARGB64:
                                {
                                    constexpr PF_Pixel_ARGB_16u white{ u16_value_white , u16_value_white , u16_value_white , u16_value_white };
                                    constexpr PF_Pixel_ARGB_16u black{ u16_value_black , u16_value_black , u16_value_black , u16_value_black };
                                    const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                                    const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                                    const PF_Pixel_ARGB_16u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                                          PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                                    // Convert from RGB to CIE-Lab color space
                                    Rgb2CIELab(input_pixels, pCIELab, sizeX, sizeY, srcPitch, sizeX);

                                    // Perform Fuzzy Median Filter
                                    switch (windowSize)
                                    {
                                        case eFUZZY_FILTER_WINDOW_3x3:
                                            FuzzyLogic_3x3 (pCIELab, input_pixels, output_pixels, sizeX, sizeY, sizeX, srcPitch, dstPitch, black, white, filterSigma);
                                        break;
                                        case eFUZZY_FILTER_WINDOW_5x5:
                                            FuzzyLogic_5x5 (pCIELab, input_pixels, output_pixels, sizeX, sizeY, sizeX, srcPitch, dstPitch, black, white, filterSigma);
                                        break;
                                        case eFUZZY_FILTER_WINDOW_7x7:
                                            FuzzyLogic_7x7 (pCIELab, input_pixels, output_pixels, sizeX, sizeY, sizeX, srcPitch, dstPitch, black, white, filterSigma);
                                        break;
                                        default:
                                            err = PF_Err_INVALID_INDEX;
                                        break;
                                    } // switch (windowSize)
                                }
                                break;

                                case PF_PixelFormat_ARGB32:
                                {
                                    constexpr PF_Pixel_ARGB_8u white{ u8_value_white , u8_value_white , u8_value_white , u8_value_white };
                                    constexpr PF_Pixel_ARGB_8u black{ u8_value_black , u8_value_black , u8_value_black , u8_value_black };
                                    const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                                    const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                                    const PF_Pixel_ARGB_8u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                                          PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                                    // Convert from RGB to CIE-Lab color space
                                    Rgb2CIELab (input_pixels, pCIELab, sizeX, sizeY, srcPitch, sizeX);

                                     // Perform Fuzzy Median Filter
                                    switch (windowSize)
                                    {
                                        case eFUZZY_FILTER_WINDOW_3x3:
                                            FuzzyLogic_3x3 (pCIELab, input_pixels, output_pixels, sizeX, sizeY, sizeX, srcPitch, dstPitch, black, white, filterSigma);
                                        break;
                                        case eFUZZY_FILTER_WINDOW_5x5:
                                            FuzzyLogic_5x5 (pCIELab, input_pixels, output_pixels, sizeX, sizeY, sizeX, srcPitch, dstPitch, black, white, filterSigma);
                                        break;
                                        case eFUZZY_FILTER_WINDOW_7x7:
                                            FuzzyLogic_7x7 (pCIELab, input_pixels, output_pixels, sizeX, sizeY, sizeX, srcPitch, dstPitch, black, white, filterSigma);
                                        break;
                                        default:
                                            err = PF_Err_INVALID_INDEX;
                                        break;
                                    } // switch (windowSize)
                                }
                                break;

                                default:
                                    err = PF_Err_BAD_CALLBACK_PARAM;
                                break;
                            } // switch (format)

                        } // if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))

                        ::FreeMemoryBlock(blockId);
                        blockId = -1;
                        pMemoryBlock = nullptr;

                    } // if (nullptr != pMemoryBlock && 0 >= blockId)

                }// if (sizeX > 16 && sizeY > 16)
                else
                    err = PF_COPY(input_worldP, output_worldP, NULL, NULL);

                ERR(extraP->cb->checkin_layer_pixels(in_data->effect_ref, eFUZZY_MEDIAN_FILTER_INPUT));

            } // if/else (0 == filterRadius)

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}
