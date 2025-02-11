#include "BilateralFilter.hpp"
#include "BilateralFilterEnum.hpp"
#include "BilateralFilterStructs.hpp"
#include "GaussMesh.hpp"
#include "BilateralFilterAlgo.hpp"
#include "CommonAuxPixFormat.hpp"
#include "PrSDKAESupport.h"
#include "ImageLabMemInterface.hpp"


static GaussMesh* gGaussMeshInstance = nullptr;
GaussMesh* getMeshHandler(void) { return gGaussMeshInstance; }


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
        dst->left = FastCompute::Min(dst->left, src->left);
        dst->top = FastCompute::Min(dst->top, src->top);
        dst->right = FastCompute::Min(dst->right, src->right);
        dst->bottom = FastCompute::Min(dst->bottom, src->bottom);
    }
    return;
}


static PF_Err
About(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output)
{
    PF_SPRINTF(
        out_data->return_msg,
        "%s, v%d.%d\r%s",
        strName,
        BilateralFilter_VersionMajor,
        BilateralFilter_VersionMinor,
        strCopyright);

    return PF_Err_NONE;
}

static PF_Err
GlobalSetup(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output)
{
    if (false == LoadMemoryInterfaceProvider(in_data))
        return PF_Err_INTERNAL_STRUCT_DAMAGED;

    if (nullptr == (gGaussMeshInstance = CreateGaussMeshHandler()))
        return PF_Err_INTERNAL_STRUCT_DAMAGED;

    constexpr PF_OutFlags out_flags1 =
        PF_OutFlag_PIX_INDEPENDENT       |
        PF_OutFlag_SEND_UPDATE_PARAMS_UI |
        PF_OutFlag_USE_OUTPUT_EXTENT     |
        PF_OutFlag_DEEP_COLOR_AWARE      |
        PF_OutFlag_WIDE_TIME_INPUT;

    constexpr PF_OutFlags out_flags2 =
        PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG |
        PF_OutFlag2_DOESNT_NEED_EMPTY_PIXELS         |
        PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT        |
        PF_OutFlag2_SUPPORTS_SMART_RENDER;

    out_data->my_version =
        PF_VERSION(
            BilateralFilter_VersionMajor,
            BilateralFilter_VersionMinor,
            BilateralFilter_VersionSub,
            BilateralFilter_VersionStage,
            BilateralFilter_VersionBuild
        );

    out_data->out_flags = out_flags1;
    out_data->out_flags2 = out_flags2;

    /* For Premiere - declare supported pixel formats */
    if (PremierId == in_data->appl_id)
    {
        auto const& pixelFormatSuite{ AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data) };

        /*	Add the pixel formats we support in order of preference. */
        (*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);

        /* Bilateral Filter as standalone PlugIn will support BGRA/ARGB formats only */
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_8u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_16u);
        (*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_ARGB_4444_32f);
    }

    return PF_Err_NONE;
}


static PF_Err
GlobalSetdown(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output)
{
    ReleaseGaussMeshHandler(nullptr);
    gGaussMeshInstance = nullptr;

    return PF_Err_NONE;
}


static PF_Err
ParamsSetup(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output)
{
    PF_ParamDef	def;
    constexpr PF_ParamFlags flags{ PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP };
    constexpr PF_ParamUIFlags ui_flags{ PF_PUI_NONE };
    constexpr PF_ParamUIFlags ui_flags_control_disabled{ ui_flags | PF_PUI_DISABLED };

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags);
    PF_ADD_SLIDER(
        FilterWindowSizeStr,
        bilateralMinRadius,
        bilateralMaxRadius,
        bilateralMinRadius,
        bilateralMaxRadius,
        bilateralDefRadius,
        eBILATERAL_FILTER_RADIUS);

    AEFX_INIT_PARAM_STRUCTURE(def, flags, ui_flags_control_disabled);
    PF_ADD_FLOAT_SLIDERX(
        FilterSigmaStr,
        fSigmaValMin,
        fSigmaValMax,
        fSigmaValMin,
        fSigmaValMax,
        fSigmaValDefault,
        PF_Precision_TENTHS,
        0,
        0,
        eBILATERAL_FILTER_SIGMA);

    out_data->num_params = eBILATERAL_TOTAL_CONTROLS;

    return PF_Err_NONE;
}


static PF_Err
Render(
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output)
{
    return ((PremierId == in_data->appl_id ? ProcessImgInPR(in_data, out_data, params, output) : ProcessImgInAE(in_data, out_data, params, output)));
}


static PF_Err
UserChangedParam(
    PF_InData						*in_data,
    PF_OutData						*out_data,
    PF_ParamDef						*params[],
    PF_LayerDef						*outputP,
    const PF_UserChangedParamExtra	*which_hitP
)
{
    PF_Err err = PF_Err_NONE;

    switch (which_hitP->param_index)
    {
        case eBILATERAL_FILTER_RADIUS:
        {
            const auto& sliderValue = params[eBILATERAL_FILTER_RADIUS]->u.sd.value;
            if (sliderValue > 0)
            {
                if (true == IsDisabledUI(params[eBILATERAL_FILTER_SIGMA]->ui_flags))
                {
                    params[eBILATERAL_FILTER_SIGMA]->ui_flags &= ~PF_PUI_DISABLED;
                    err = AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data)->
                        PF_UpdateParamUI(in_data->effect_ref, eBILATERAL_FILTER_SIGMA, params[eBILATERAL_FILTER_SIGMA]);
                }
            }
            else
            {
                if (false == IsDisabledUI(params[eBILATERAL_FILTER_SIGMA]->ui_flags))
                {
                    params[eBILATERAL_FILTER_SIGMA]->ui_flags |= PF_PUI_DISABLED;
                    err = AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data)->
                        PF_UpdateParamUI(in_data->effect_ref, eBILATERAL_FILTER_SIGMA, params[eBILATERAL_FILTER_SIGMA]);
                }
            }
        }
        break;

        default: // nothing todo
        break;
    }

    return err;
}


static PF_Err
UpdateParameterUI(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_ParamDef			*params[],
    PF_LayerDef			*output
)
{
    // nothing TODO
    return PF_Err_NONE;
}



static PF_Err
PreRender(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    BFilterParamsStr filterStrParams{};
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(BFilterParamStrSize);
    if (nullptr != paramsHandler)
    {
        BFilterParamsStr* paramsStrP = reinterpret_cast<BFilterParamsStr*>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            extra->output->pre_render_data = paramsHandler;

            PF_ParamDef	filterRadius{};
            PF_ParamDef	filterSigma{};

            const PF_Err errParam1 = PF_CHECKOUT_PARAM(in_data, eBILATERAL_FILTER_RADIUS, in_data->current_time, in_data->time_step, in_data->time_scale, &filterRadius);
            const PF_Err errParam2 = PF_CHECKOUT_PARAM(in_data, eBILATERAL_FILTER_SIGMA,  in_data->current_time, in_data->time_step, in_data->time_scale, &filterSigma);

            if (PF_Err_NONE == errParam1 && PF_Err_NONE == errParam2)
            {
                paramsStrP->fRadius = CLAMP_VALUE(filterRadius.u.sd.value, bilateralMinRadius, bilateralMaxRadius);
                paramsStrP->fSigma  = CLAMP_VALUE(static_cast<float>(filterRadius.u.fs_d.value + 5.0), fSigmaValMin, fSigmaValMax);
            } // if (PF_Err_NONE == errParam1 && PF_Err_NONE == errParam2)
            else
            {
                paramsStrP->fRadius = bilateralDefRadius;
                paramsStrP->fSigma = fSigmaValDefault;
            }

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
            (in_data->effect_ref, eBILATERAL_FILTER_INPUT, eBILATERAL_FILTER_INPUT, &req, in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));

            UnionLRect (&in_result.result_rect, &extra->output->result_rect);
            UnionLRect (&in_result.max_result_rect, &extra->output->max_result_rect);
            handleSuite->host_unlock_handle(paramsHandler);

        } // if (nullptr != paramsStrP)
        else
            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    } // if (nullptr != paramsHandler)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}


static PF_Err
SmartRender(
    PF_InData				*in_data,
    PF_OutData				*out_data,
    PF_SmartRenderExtra		*extraP
)
{
    PF_EffectWorld* input_worldP = nullptr;
    PF_EffectWorld* output_worldP = nullptr;
    PF_Err	err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    BFilterParamsStr* pFilterStrParams = reinterpret_cast<BFilterParamsStr*>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));

    if (nullptr != pFilterStrParams)
    {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, eBILATERAL_FILTER_INPUT, &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (nullptr != input_worldP && nullptr != output_worldP)
        {
            const A_long sizeX = input_worldP->width;
            const A_long sizeY = input_worldP->height;
            const A_long srcRowBytes = input_worldP->rowbytes;  // Get input buffer pitch in bytes
            const A_long dstRowBytes = output_worldP->rowbytes; // Get output buffer pitch in bytes

            AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);

            const A_long filterRadius = pFilterStrParams->fRadius;
            const float  filterSigma = pFilterStrParams->fSigma;

            if (0 == filterRadius)
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

                                const PF_Pixel_ARGB_32f* __restrict input_pixels = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input_worldP->data);
                                PF_Pixel_ARGB_32f* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(output_worldP->data);

                                // Convert from RGB to CIE-Lab color space
                                Rgb2CIELab(input_pixels, pCIELab, sizeX, sizeY, srcPitch, sizeX);
                                // Perform Bilateral Filter
                                BilateralFilterAlgorithm(pCIELab, input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, filterRadius, filterSigma, black, white);
                            }
                            break;

                            case PF_PixelFormat_ARGB64:
                            {
                                constexpr PF_Pixel_ARGB_16u white{ u16_value_white , u16_value_white , u16_value_white , u16_value_white };
                                constexpr PF_Pixel_ARGB_16u black{ u16_value_black , u16_value_black , u16_value_black , u16_value_black };
                                const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                                const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                                const PF_Pixel_ARGB_16u* __restrict input_pixels = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                                PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                                // Convert from RGB to CIE-Lab color space
                                Rgb2CIELab(input_pixels, pCIELab, sizeX, sizeY, srcPitch, sizeX);
                                // Perform Bilateral Filter
                                BilateralFilterAlgorithm(pCIELab, input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, filterRadius, filterSigma, black, white);
                            }
                            break;

                            case PF_PixelFormat_ARGB32:
                            {
                                constexpr PF_Pixel_ARGB_8u white{ u8_value_white , u8_value_white , u8_value_white , u8_value_white };
                                constexpr PF_Pixel_ARGB_8u black{ u8_value_black , u8_value_black , u8_value_black , u8_value_black };
                                const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                                const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                                const PF_Pixel_ARGB_8u* __restrict input_pixels = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                                PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                                // Convert from RGB to CIE-Lab color space
                                Rgb2CIELab(input_pixels, pCIELab, sizeX, sizeY, srcPitch, sizeX);
                                // Perform Bilateral Filter
                                BilateralFilterAlgorithm(pCIELab, input_pixels, output_pixels, sizeX, sizeY, srcPitch, dstPitch, filterRadius, filterSigma, black, white);
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

                ERR(extraP->cb->checkin_layer_pixels(in_data->effect_ref, eBILATERAL_FILTER_INPUT));

            } // if/else (0 == filterRadius)

        } // if (nullptr != input_worldP && nullptr != output_worldP)

        handleSuite->host_unlock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data));

    } // if (nullptr != pFilterStrParams)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
}


PLUGIN_ENTRY_POINT_CALL PF_Err
EffectMain(
    PF_Cmd			cmd,
    PF_InData		*in_data,
    PF_OutData		*out_data,
    PF_ParamDef		*params[],
    PF_LayerDef		*output,
    void			*extra)
{
    PF_Err err{ PF_Err_NONE };

    try {
        switch (cmd)
        {
            case PF_Cmd_ABOUT:
                ERR(About(in_data, out_data, params, output));
            break;

            case PF_Cmd_GLOBAL_SETUP:
                ERR(GlobalSetup(in_data, out_data, params, output));
            break;

            case PF_Cmd_GLOBAL_SETDOWN:
                ERR(GlobalSetdown(in_data, out_data, params, output));
            break;

            case PF_Cmd_PARAMS_SETUP:
                ERR(ParamsSetup(in_data, out_data, params, output));
            break;

            case PF_Cmd_RENDER:
                ERR(Render(in_data, out_data, params, output));
            break;

            case PF_Cmd_USER_CHANGED_PARAM:
                ERR(UserChangedParam(in_data, out_data, params, output, reinterpret_cast<const PF_UserChangedParamExtra*>(extra)));
            break;

            // Handling this selector will ensure that the UI will be properly initialized,
            // even before the user starts changing parameters to trigger PF_Cmd_USER_CHANGED_PARAM
            case PF_Cmd_UPDATE_PARAMS_UI:
                ERR(UpdateParameterUI(in_data, out_data, params, output));
            break;

            case PF_Cmd_SMART_PRE_RENDER:
                ERR(PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra)));
            break;

            case PF_Cmd_SMART_RENDER:
                ERR(SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra)));
            break;

            default:
            break;
        }
    }
    catch (PF_Err& thrown_err)
    {
        err = thrown_err;
    }

    return err;
}