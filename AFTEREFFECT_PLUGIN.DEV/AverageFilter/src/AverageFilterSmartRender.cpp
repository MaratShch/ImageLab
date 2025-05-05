#include "AverageFilter.hpp"
#include "FastAriphmetics.hpp"
#include "CompileTimeUtils.hpp"
#include "AverageFilterStructs.hpp"
#include "AverageAFilterAlgo.hpp"
#include "AverageGFilterAlgo.hpp"
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
        dst->left   = FastCompute::Min(dst->left,   src->left);
        dst->top    = FastCompute::Min(dst->top,    src->top);
        dst->right  = FastCompute::Min(dst->right,  src->right);
        dst->bottom = FastCompute::Min(dst->bottom, src->bottom);
    }
    return;
}


PF_Err AverageFilter_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
) noexcept
{
    AFilterParamsStr filterStrParams{};
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(AFilterParamStrSize);
    if (nullptr != paramsHandler)
    {
        AFilterParamsStr* paramsStrP = reinterpret_cast<AFilterParamsStr*>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            extra->output->pre_render_data = paramsHandler;

            PF_ParamDef	filterSize{};
            PF_ParamDef	filterGeomteric{};

            const PF_Err errParam1 = PF_CHECKOUT_PARAM(in_data, eAEVRAGE_FILTER_WINDOW_SIZE, in_data->current_time, in_data->time_step, in_data->time_scale, &filterSize);
            const PF_Err errParam2 = PF_CHECKOUT_PARAM(in_data, eAVERAGE_FILTER_GEOMETRIC_AVERAGE, in_data->current_time, in_data->time_step, in_data->time_scale, &filterGeomteric);

            if (PF_Err_NONE == errParam1 && PF_Err_NONE == errParam2)
            {
                paramsStrP->eSize       = CLAMP_VALUE(static_cast<eAVERAGE_FILTER_WINDOW_SIZE>(filterSize.u.pd.value - 1), eAVERAGE_WINDOW_3x3, eAVERAGE_WINDOW_7x7);
                paramsStrP->isGeometric = CLAMP_VALUE(static_cast<A_long>(filterGeomteric.u.bd.value), 0, 1);
            } // if (PF_Err_NONE == errParam1 && PF_Err_NONE == errParam2)
            else
            {
                paramsStrP->eSize = eAVERAGE_WINDOW_3x3;
                paramsStrP->isGeometric = 0;
            }

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
            (in_data->effect_ref, eAEVRAGE_FILTER_INPUT, eAEVRAGE_FILTER_INPUT, &req, in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));

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


PF_Err AverageFilter_SmartRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_SmartRenderExtra	*extraP
) noexcept
{
    PF_EffectWorld* input_worldP = nullptr;
    PF_EffectWorld* output_worldP = nullptr;
    PF_Err	err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    const AFilterParamsStr* pFilterStrParams = reinterpret_cast<const AFilterParamsStr*>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));

    if (nullptr != pFilterStrParams)
    {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, eAEVRAGE_FILTER_INPUT, &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (nullptr != input_worldP && nullptr != output_worldP)
        {
            const A_long sizeX = input_worldP->width;
            const A_long sizeY = input_worldP->height;
            const A_long srcRowBytes = input_worldP->rowbytes;  // Get input buffer pitch in bytes
            const A_long dstRowBytes = output_worldP->rowbytes; // Get output buffer pitch in bytes

            AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);

            const A_long windowSize = WindowSizeEnum2Value(pFilterStrParams->eSize);
            const A_long isGeometric = pFilterStrParams->isGeometric;

            PF_PixelFormat format = PF_PixelFormat_INVALID;
            if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))
            {
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
                        
                        err = ((0 == isGeometric) ?
                                  AverageFilterAlgo(input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, windowSize) :
                                  GeomethricAverageFilterAlgo(input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, windowSize));
                    }
                    break;

                    case PF_PixelFormat_ARGB64:
                    {
                        constexpr PF_Pixel_ARGB_16u white{ u16_value_white , u16_value_white , u16_value_white , u16_value_white };
                        constexpr PF_Pixel_ARGB_16u black{ u16_value_black , u16_value_black , u16_value_black , u16_value_black };
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                        const PF_Pixel_ARGB_16u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                              PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                        err = ((0 == isGeometric) ?
                                  AverageFilterAlgo(input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, windowSize) :
                                  GeomethricAverageFilterAlgo(input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, windowSize));
                    }
                    break;

                    case PF_PixelFormat_ARGB32:
                    {
                        constexpr PF_Pixel_ARGB_8u white{ u8_value_white , u8_value_white , u8_value_white , u8_value_white };
                        constexpr PF_Pixel_ARGB_8u black{ u8_value_black , u8_value_black , u8_value_black , u8_value_black };
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                        const PF_Pixel_ARGB_8u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                              PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                        err = ((0 == isGeometric) ?
                                  AverageFilterAlgo(input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, windowSize) :
                                  GeomethricAverageFilterAlgo(input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, windowSize));
                    }
                    break;

                    default:
                       err = PF_Err_BAD_CALLBACK_PARAM;
                    break;
                } // switch (format)

            } // if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}