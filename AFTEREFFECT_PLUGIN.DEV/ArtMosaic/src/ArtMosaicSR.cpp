#include "ArtMosaic.hpp"
#include "ArtMosaicEnum.hpp"
#include "MosaicMemHandler.hpp"
#include "MosaicColorConvert.hpp"
#include "CommonSmartRender.hpp"
#include "ImageLabMemInterface.hpp"

struct MosaicControls
{
    int32_t cellsNumber;
};
constexpr size_t MosaicControlsSize = sizeof(MosaicControls);

using PMosaicControls = MosaicControls*;


PF_Err
ArtMosaic_PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(MosaicControlsSize);

    if (nullptr != paramsHandler)
    {
        PMosaicControls paramsStrP = reinterpret_cast<PMosaicControls>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            CACHE_ALIGN PF_ParamDef paramVal{};

            extra->output->pre_render_data = paramsHandler;

            PF_CHECKOUT_PARAM(in_data, UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_CELLS_SLIDER), in_data->current_time, in_data->time_step, in_data->time_scale, &paramVal);
            paramsStrP->cellsNumber = static_cast<int32_t>(paramVal.u.sd.value);

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
            (in_data->effect_ref,
                UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_INPUT),
                UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_INPUT),
                &req,
                in_data->current_time,
                in_data->time_step,
                in_data->time_scale,
                &in_result)
            );

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

PF_Err
ArtMosaic_SmartRender
(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
)
{
    PF_EffectWorld* input_worldP = nullptr;
    PF_EffectWorld* output_worldP = nullptr;
    PF_Err	err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    const MosaicControls* pFilterStrParams = reinterpret_cast<const MosaicControls*>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));

    if (nullptr != pFilterStrParams)
    {
        const A_long sliderValue = pFilterStrParams->cellsNumber;;
        const A_long cellsNumber = (sliderValue < cellMin ? cellMin : sliderValue > cellMax ? cellMax : sliderValue);

        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_INPUT), &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (nullptr != input_worldP && nullptr != output_worldP)
        {
            const A_long sizeX = input_worldP->width;
            const A_long sizeY = input_worldP->height;
            const A_long srcRowBytes = input_worldP->rowbytes;  // Get input buffer pitch in bytes
            const A_long dstRowBytes = output_worldP->rowbytes; // Get output buffer pitch in bytes

            PF_PixelFormat format = PF_PixelFormat_INVALID;
            AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);

            MemHandler algoMemHandler = alloc_memory_buffers(sizeX, sizeY, cellsNumber);
            if (true == mem_handler_valid(algoMemHandler))
            {
                if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))
                {
                    switch (format)
                    {
                        case PF_PixelFormat_ARGB128:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                            const PF_Pixel_ARGB_32f* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_32f* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output_worldP->data);

                            rgb2planar (input_pixels, algoMemHandler, sizeX, sizeY, srcPitch);      // convert interleaved to planar format (range 0.f ... 225.f)
                            MosaicAlgorithmMain (algoMemHandler, sizeX, sizeY, cellsNumber);        // perform SLIC algorithm
                            planar2rgb<false, false> (input_pixels, algoMemHandler, output_pixels, sizeX, sizeY, srcPitch, dstPitch); // back convert from planar to interleaved format
                        }
                        break;

                        case PF_PixelFormat_ARGB64:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                            const PF_Pixel_ARGB_16u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                            rgb2planar (input_pixels, algoMemHandler, sizeX, sizeY, srcPitch);      // convert interleaved to planar format (range 0.f ... 225.f)
                            MosaicAlgorithmMain (algoMemHandler, sizeX, sizeY, cellsNumber);        // perform SLIC algorithm
                            planar2rgb<false, false> (input_pixels, algoMemHandler, output_pixels, sizeX, sizeY, srcPitch, dstPitch); // back convert from planar to interleaved format
                        }
                        break;

                        case PF_PixelFormat_ARGB32:
                        {
                            const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                            const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                            const PF_Pixel_ARGB_8u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                                  PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                            rgb2planar (input_pixels, algoMemHandler, sizeX, sizeY, srcPitch);      // convert interleaved to planar format (range 0.f ... 225.f)
                            MosaicAlgorithmMain (algoMemHandler, sizeX, sizeY, cellsNumber);        // perform SLIC algorithm
                            planar2rgb<false, false> (input_pixels, algoMemHandler, output_pixels, sizeX, sizeY, srcPitch, dstPitch); // back convert from planar to interleaved format
                        }
                        break;

                        default:
                            err = PF_Err_BAD_CALLBACK_PARAM;
                        break;
                    } // switch (format)

                } // if (PF_Err_NONE == wsP->PF_GetPixelFormat(input_worldP, &format))

                  // free (return back to memory manager) already non used buffer
                free_memory_buffers(algoMemHandler);

            } //  if (true == mem_handler_valid(algoMemHandler))

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)

    return err;
}
