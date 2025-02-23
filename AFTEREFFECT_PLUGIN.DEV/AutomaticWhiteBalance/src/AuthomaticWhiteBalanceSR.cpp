#include "AuthomaticWhiteBalance.hpp"
#include "FastAriphmetics.hpp"
#include "ImageLabMemInterface.hpp"
#include "CompileTimeUtils.hpp"
#include "AlgCommonFunctions.hpp"
#include "AE_Effect.h"


inline PF_Boolean IsEmptyRect(const PF_LRect* r) noexcept
{
    return (r->left >= r->right) || (r->top >= r->bottom);
}


inline void UnionLRect(const PF_LRect* src, PF_LRect* dst) noexcept
{
    if (IsEmptyRect(dst))
    {
        *dst = *src;
    }
    else if (!IsEmptyRect(src))
    {
        dst->left   = FastCompute::Min(dst->left, src->left);
        dst->top    = FastCompute::Min(dst->top, src->top);
        dst->right  = FastCompute::Min(dst->right, src->right);
        dst->bottom = FastCompute::Min(dst->bottom, src->bottom);
    }
    return;
}


PF_Err
AuthomaticWhiteBalance_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
) noexcept
{
    AWB_SmartRenderParams renderParams{};
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(AWB_SmartRenderParamsSize);
    if (nullptr != paramsHandler)
    {
        PAWB_SmartRenderParams paramsStrP = reinterpret_cast<PAWB_SmartRenderParams>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            extra->output->pre_render_data = paramsHandler;

            PF_ParamDef param_Illuminant{};
            PF_ParamDef	param_Chromatic{};
            PF_ParamDef param_ColorSpace{};
            PF_ParamDef param_GrayThreshold{};
            PF_ParamDef param_IterationsNumber{};

            const PF_Err errParam1 = PF_CHECKOUT_PARAM(in_data, AWB_ILLUMINATE_POPUP,  in_data->current_time, in_data->time_step, in_data->time_scale, &param_Illuminant);
            const PF_Err errParam2 = PF_CHECKOUT_PARAM(in_data, AWB_CHROMATIC_POPUP,   in_data->current_time, in_data->time_step, in_data->time_scale, &param_Chromatic);
            const PF_Err errParam3 = PF_CHECKOUT_PARAM(in_data, AWB_COLOR_SPACE_POPUP, in_data->current_time, in_data->time_step, in_data->time_scale, &param_ColorSpace);
            const PF_Err errParam4 = PF_CHECKOUT_PARAM(in_data, AWB_THRESHOLD_SLIDER,  in_data->current_time, in_data->time_step, in_data->time_scale, &param_GrayThreshold);
            const PF_Err errParam5 = PF_CHECKOUT_PARAM(in_data, AWB_ITERATIONS_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_IterationsNumber);

            if (PF_Err_NONE == errParam1 && PF_Err_NONE == errParam2 && PF_Err_NONE == errParam3 && PF_Err_NONE == errParam4 && PF_Err_NONE == errParam5)
            {
                paramsStrP->srParam_Illuminant        =  CLAMP_VALUE(static_cast<eILLUMINATE>(param_Illuminant.u.pd.value), DAYLIGHT, TOTAL_ILLUMINATES);
                paramsStrP->srParam_ChromaticAdapt    =  CLAMP_VALUE(static_cast<eChromaticAdaptation>(param_Chromatic.u.pd.value), CHROMATIC_CAT02, TOTAL_CHROMATIC);
                paramsStrP->srParam_ColorSpace        =  CLAMP_VALUE(static_cast<eCOLOR_SPACE>(param_ColorSpace.u.pd.value), BT601, SMPTE240M);
                paramsStrP->srParam_GrayThreshold     = static_cast<float>(CLAMP_VALUE(static_cast<int32_t>(param_GrayThreshold.u.sd.value), gMinGrayThreshold, gMaxGrayThreshold)) / 100.f;
                paramsStrP->srParam_ItrerationsNumber =  CLAMP_VALUE(static_cast<int32_t>(param_IterationsNumber.u.sd.value), iterMinCnt, iterMaxCnt);
            } // if (PF_Err_NONE == errParam1 && PF_Err_NONE == errParam2 ... )
            else
            {
                paramsStrP->srParam_Illuminant        = DAYLIGHT;
                paramsStrP->srParam_ChromaticAdapt    = CHROMATIC_CAT02;
                paramsStrP->srParam_ColorSpace        = BT709;
                paramsStrP->srParam_GrayThreshold     = static_cast<float>(gDefGrayThreshold) / 100.f;
                paramsStrP->srParam_ItrerationsNumber = iterDefCnt;
            }

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
            (in_data->effect_ref, AWB_INPUT, AWB_INPUT, &req, in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));

            UnionLRect(&in_result.result_rect, &extra->output->result_rect);
            UnionLRect(&in_result.max_result_rect, &extra->output->max_result_rect);

            handleSuite->host_unlock_handle(paramsHandler);

        } // if (nullptr != paramsStrP)
        else
            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    } // if (nullptr != paramsHandler)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}


template <typename T>
PF_Err
AuthomaticWhiteBalance_SmartRenderAlgo
(
    T* __restrict pSrcImage,
    T* __restrict pDstImage,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    const PAWB_SmartRenderParams __restrict pParamList
) noexcept
{
    CACHE_ALIGN float U_avg[gMaxCnt]{};
    CACHE_ALIGN float V_avg[gMaxCnt]{};
    T* __restrict pMem[2]{};
    T* __restrict srcInput = nullptr;
    T* __restrict dstOutput = nullptr;
    void* pMemoryBlock = nullptr;

    const eILLUMINATE          eIlluminant    = pParamList->srParam_Illuminant;
    const eChromaticAdaptation eChromaAdapt   = pParamList->srParam_ChromaticAdapt;
    const eCOLOR_SPACE         eColorSpace    = pParamList->srParam_ColorSpace;
    const float                fGrayThreahold = pParamList->srParam_GrayThreshold;
    const int32_t              iterCnt        = pParamList->srParam_ItrerationsNumber;

    const A_long memBlocksNumber = FastCompute::Min(2, (iterCnt - 1));
    PF_Err	err = PF_Err_NONE;

    int32_t memBlockId = -1;
    int32_t srcIdx = 0, dstIdx = 1;
    A_long inPitch = 0, outPitch = 0;

    if (memBlocksNumber > 0)
    {
        const size_t frameSize = sizeX * sizeY;
        const size_t requiredMemSize = memBlocksNumber * frameSize * sizeof(T);
        memBlockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemoryBlock);
        if (nullptr == pMemoryBlock || memBlockId < 0)
            return PF_Err_OUT_OF_MEMORY; // not enough memory for allocate temporary buffer for ALGO

        pMem[0] = reinterpret_cast<T* __restrict>(pMemoryBlock);
        pMem[1] = (2 == memBlocksNumber) ? pMem[0] + frameSize : nullptr;
    }

    float uAvg, vAvg;

    // Perform iterations in corresponding to the slider value 
    for (A_long k = 0; k < iterCnt; k++)
    {
        if (0 == k)
        {   // First iteration
            srcInput = pSrcImage;
            dstIdx++;
            dstIdx &= 0x1;
            dstOutput = (1 == iterCnt) ? pDstImage : pMem[dstIdx];
            inPitch = srcPitch;
            outPitch = (1 == iterCnt) ? dstPitch : sizeX;
        }
        else if ((iterCnt - 1) == k)
        {   // Last iteration 
            srcIdx    = dstIdx;
            srcInput  = pMem[srcIdx];
            dstOutput = pDstImage;
            inPitch   = sizeX;
            outPitch  = dstPitch;
        } /* if (k > 0) */
        else
        {   // Intermediate iteration
            srcIdx = dstIdx;
            dstIdx++;
            dstIdx &= 0x1;
            srcInput  = pMem[srcIdx];
            dstOutput = pMem[dstIdx];
            inPitch   = outPitch = sizeX;
        }

        uAvg = vAvg = 0.f;
        // collect statistics about image and compute averages values for U and for V components
        collect_rgb_statistics (srcInput, sizeX, sizeY, inPitch, fGrayThreahold, eColorSpace, &uAvg, &vAvg);

        U_avg[k] = uAvg;
        V_avg[k] = vAvg;

        if (k > 0)
        {
            const float U_diff = U_avg[k] - U_avg[k - 1];
            const float V_diff = V_avg[k] - V_avg[k - 1];

            const float normVal = FastCompute::Sqrt(U_diff * U_diff + V_diff * V_diff);

            if (normVal < algAWBepsilon)
            {
                // U and V no longer improving, so just copy source to destination and break the loop
                simple_image_copy (srcInput, pDstImage, sizeX, sizeY, srcPitch, outPitch);

                // release temporary memory buffers on exit from function
                if (-1 != memBlockId)
                {
                    ::FreeMemoryBlock(memBlockId);
                    memBlockId = -1;
                }
                return true; // U and V no longer improving
            }
        } // if (k > 0) 

          // compute correction matrix
        float correctionMatrix[3]{};
        compute_correction_matrix (uAvg, vAvg, eColorSpace, eIlluminant, eChromaAdapt, correctionMatrix);

        // in second: perform image color correction
        image_rgb_correction (srcInput, dstOutput, sizeX, sizeY, inPitch, outPitch, correctionMatrix);

    } // for (A_long k = 0; k < iterCnt; k++)

      // release temporary memory buffers on exit from function
    if (-1 != memBlockId)
    {
        ::FreeMemoryBlock(memBlockId);
        memBlockId = -1;
    }

    return err;
}



PF_Err
AuthomaticWhiteBalance_SmartRender
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
    const PAWB_SmartRenderParams pAWBStrParams = reinterpret_cast<const PAWB_SmartRenderParams>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));

    if (nullptr != pAWBStrParams)
    {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, AWB_INPUT, &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

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
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                        PF_Pixel_ARGB_32f* __restrict input_pixels  = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(input_worldP->data);
                        PF_Pixel_ARGB_32f* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(output_worldP->data);

                        err = AuthomaticWhiteBalance_SmartRenderAlgo (input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, pAWBStrParams);
                    }
                    break;

                    case PF_PixelFormat_ARGB64:
                    {
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                        PF_Pixel_ARGB_16u* __restrict input_pixels  = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                        PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                        err = AuthomaticWhiteBalance_SmartRenderAlgo (input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, pAWBStrParams);
                    }
                    break;

                    case PF_PixelFormat_ARGB32:
                    {
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                        PF_Pixel_ARGB_8u* __restrict input_pixels  = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                        PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                        err = AuthomaticWhiteBalance_SmartRenderAlgo (input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, pAWBStrParams);
                    }
                    break;

                    default:
                        err = PF_Err_BAD_CALLBACK_PARAM;
                    break;
                } // switch (format)

            } // if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))

            ERR(extraP->cb->checkin_layer_pixels(in_data->effect_ref, AWB_INPUT));

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}