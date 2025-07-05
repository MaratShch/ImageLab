#include "ImageFlipping.hpp"
#include "PrSDKAESupport.h"
#include "CommonSmartRender.hpp"

static PF_Err ProcessImgInPR
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_FLIP_FILTER_INPUT]->u.ld);

	auto const flipHoriz = params[IMAGE_FLIP_HORIZONTAL_CHECKBOX]->u.bd.value;
	auto const flipVert  = params[IMAGE_FLIP_VERTICAL_CHECKBOX  ]->u.bd.value;
	auto const flip = flipHoriz | (flipVert << 1);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;

	/* This plugin called from PR - check video fomat */
	if (PF_Err_NONE == AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
			{
				const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
				auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
				ImageProcess (localSrc, localDst, height, width, line_pitch, line_pitch, flip);
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			{
				const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
				auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
				ImageProcess (localSrc, localDst, height, width, line_pitch, line_pitch, flip);
			}
			break;

			case PrPixelFormat_BGRA_4444_32f:
			{
				const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
				      PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
				auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
				ImageProcess (localSrc, localDst, height, width, line_pitch, line_pitch, flip);
			}
			break;

			case PrPixelFormat_VUYA_4444_8u:
			case PrPixelFormat_VUYA_4444_8u_709:
			{
				const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
				      PF_Pixel_VUYA_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
				auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);
				ImageProcess (localSrc, localDst, height, width, line_pitch, line_pitch, flip);
			}
			break;

			case PrPixelFormat_VUYA_4444_32f:
			case PrPixelFormat_VUYA_4444_32f_709:
			{
				const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
				      PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_32f* __restrict>(output->data);
				auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);
				ImageProcess (localSrc, localDst, height, width, line_pitch, line_pitch, flip);
			}
			break;

			case PrPixelFormat_RGB_444_10u:
			{
				const PF_Pixel_RGB_10u* __restrict localSrc = reinterpret_cast<const PF_Pixel_RGB_10u* __restrict>(pfLayer->data);
				      PF_Pixel_RGB_10u* __restrict localDst = reinterpret_cast<      PF_Pixel_RGB_10u* __restrict>(output->data);
				auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_RGB_10u_size);
				ImageProcess (localSrc, localDst, height, width, line_pitch, line_pitch, flip);
			}
			break;

			default:
				err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
			break;
		} /* switch (destinationPixelFormat) */

	} /* if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat))) */
	else
	{
		/* error in determine pixel format */
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}

static PF_Err ProcessImgInAE
(
	PF_InData*    in_data,
	PF_OutData*   out_data,
	PF_ParamDef*  params[],
	PF_LayerDef*  output
) noexcept
{
	const PF_EffectWorld*    __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_FLIP_FILTER_INPUT]->u.ld);

	auto const height = output->height;
	auto const width  = output->width;

	auto const flipHoriz = params[IMAGE_FLIP_HORIZONTAL_CHECKBOX]->u.bd.value;
	auto const flipVert = params[IMAGE_FLIP_VERTICAL_CHECKBOX]->u.bd.value;
	auto const flip = flipHoriz | (flipVert << 1);

	if (PF_WORLD_IS_DEEP(output))
	{
        PF_PixelFormat format = PF_PixelFormat_INVALID;
        AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
        if (PF_Err_NONE == wsP->PF_GetPixelFormat(reinterpret_cast<PF_EffectWorld* __restrict>(&params[IMAGE_FLIP_FILTER_INPUT]->u.ld), &format))
        {
            if (format == PF_PixelFormat_ARGB128)
            {
                const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
                      PF_Pixel_ARGB_16u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output->data);

                auto const src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                auto const dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                ImageProcess (localSrc, localDst, height, width, src_line_pitch, dst_line_pitch, flip);
            }
            else
            {
                const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
                      PF_Pixel_ARGB_32f* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_32f* __restrict>(output->data);

                auto const src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
                auto const dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

                ImageProcess (localSrc, localDst, height, width, src_line_pitch, dst_line_pitch, flip);
            }
        }
	}
	else
	{
		const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
		      PF_Pixel_ARGB_8u* __restrict localDst = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output->data);

		auto const src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
		auto const dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

		ImageProcess (localSrc, localDst, height, width, src_line_pitch, dst_line_pitch, flip);
	}

	return PF_Err_NONE;
}


static PF_Err
About(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_SPRINTF(out_data->return_msg,
		"%s, v%d.%d\r%s",
		strName,
		EqualizationFilter_VersionMajor,
		EqualizationFilter_VersionMinor,
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
	PF_Err	err = PF_Err_NONE;

	constexpr PF_OutFlags out_flags1 =
		PF_OutFlag_PIX_INDEPENDENT |
		PF_OutFlag_SEND_UPDATE_PARAMS_UI |
		PF_OutFlag_USE_OUTPUT_EXTENT |
		PF_OutFlag_DEEP_COLOR_AWARE |
		PF_OutFlag_WIDE_TIME_INPUT;

	constexpr PF_OutFlags out_flags2 =
		PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG |
		PF_OutFlag2_DOESNT_NEED_EMPTY_PIXELS |
		PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT;

	out_data->my_version =
		PF_VERSION(
			EqualizationFilter_VersionMajor,
			EqualizationFilter_VersionMinor,
			EqualizationFilter_VersionSub,
			EqualizationFilter_VersionStage,
			EqualizationFilter_VersionBuild
		);

	out_data->out_flags = out_flags1;
	out_data->out_flags2 = out_flags2;

	/* For Premiere - declare supported pixel formats */
	if (PremierId == in_data->appl_id)
	{
		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
			AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);

		/*	Add the pixel formats we support in order of preference. */
		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_data->effect_ref);

		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
	}

	return err;
}


static PF_Err
GlobalSetDown(
	PF_InData		*in_data,
	PF_OutData		*out_data)
{
	/* nothing to do */
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

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_CHECKBOXX(fH, FALSE, 0, IMAGE_FLIP_HORIZONTAL_CHECKBOX);

	AEFX_CLR_STRUCT_EX(def);
	def.flags = flags;
	def.ui_flags = ui_flags;
	PF_ADD_CHECKBOXX(fV, FALSE, 0, IMAGE_FLIP_VERTICAL_CHECKBOX);

	out_data->num_params = IMAGE_EQUALIZATION_FILTER_TOTAL_PARAMS;
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
PreRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_PreRenderExtra	*extra
)
{
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle paramsHandler = handleSuite->host_new_handle(sizeof(A_long));
    if (nullptr != paramsHandler)
    {
        A_long* paramsStrP = reinterpret_cast<A_long*>(handleSuite->host_lock_handle(paramsHandler));
        if (nullptr != paramsStrP)
        {
            extra->output->pre_render_data = paramsHandler;

            PF_ParamDef	flipHorizontal{};
            PF_ParamDef	flipVertical{};

            const PF_Err errParam1 = PF_CHECKOUT_PARAM(in_data, IMAGE_FLIP_HORIZONTAL_CHECKBOX, in_data->current_time, in_data->time_step, in_data->time_scale, &flipHorizontal);
            const PF_Err errParam2 = PF_CHECKOUT_PARAM(in_data, IMAGE_FLIP_VERTICAL_CHECKBOX, in_data->current_time, in_data->time_step, in_data->time_scale, &flipVertical);

            *paramsStrP = (PF_Err_NONE == errParam1 && PF_Err_NONE == errParam2) ?
                ((flipVertical.u.bd.value & 0x01) << 1) | (flipHorizontal.u.bd.value & 0x01) : 0;

            PF_RenderRequest req = extra->input->output_request;
            PF_CheckoutResult in_result{};

            ERR(extra->cb->checkout_layer
            (in_data->effect_ref, IMAGE_FLIP_FILTER_INPUT, IMAGE_FLIP_FILTER_INPUT, &req, in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));

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


static PF_Err
SmartRender
(
    PF_InData			*in_data,
    PF_OutData			*out_data,
    PF_SmartRenderExtra	*extraP
)
{
    PF_EffectWorld* input_worldP = nullptr;
    PF_EffectWorld* output_worldP = nullptr;
    PF_Err	err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    const A_long* pFilterParams = reinterpret_cast<const A_long*>(handleSuite->host_lock_handle(reinterpret_cast<PF_Handle>(extraP->input->pre_render_data)));
    const A_long flipType = (*pFilterParams) & 0x03;

    if (nullptr != pFilterParams)
    {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, IMAGE_FLIP_FILTER_INPUT, &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (nullptr != input_worldP && nullptr != output_worldP)
        {
            const A_long sizeX = input_worldP->width;
            const A_long sizeY = input_worldP->height;
            const A_long srcRowBytes = input_worldP->rowbytes;  // Get input buffer pitch in bytes
            const A_long dstRowBytes = output_worldP->rowbytes; // Get output buffer pitch in bytes

            AEFX_SuiteScoper<PF_WorldSuite2> wsP = AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);

            PF_PixelFormat format = PF_PixelFormat_INVALID;
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

                        ImageProcess (input_pixels, output_pixels, sizeY, sizeX, srcPitch, dstPitch, flipType);
                    }
                    break;

                    case PF_PixelFormat_ARGB64:
                    {
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

                        const PF_Pixel_ARGB_16u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input_worldP->data);
                              PF_Pixel_ARGB_16u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_16u* __restrict>(output_worldP->data);

                        ImageProcess (input_pixels, output_pixels, sizeY, sizeX, srcPitch, dstPitch, flipType);
                    }
                    break;

                    case PF_PixelFormat_ARGB32:
                    {
                        const A_long srcPitch = srcRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
                        const A_long dstPitch = dstRowBytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

                        const PF_Pixel_ARGB_8u* __restrict input_pixels  = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input_worldP->data);
                              PF_Pixel_ARGB_8u* __restrict output_pixels = reinterpret_cast<      PF_Pixel_ARGB_8u* __restrict>(output_worldP->data);

                        ImageProcess (input_pixels, output_pixels, sizeY, sizeX, srcPitch, dstPitch, flipType);
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




PLUGIN_ENTRY_POINT_CALL  PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err		err = PF_Err_NONE;

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
				ERR(GlobalSetDown(in_data, out_data));
			break;

			case PF_Cmd_PARAMS_SETUP:
				ERR(ParamsSetup(in_data, out_data, params, output));
			break;

			case PF_Cmd_RENDER:
				ERR(Render(in_data, out_data, params, output));
			break;

            case PF_Cmd_SMART_PRE_RENDER:
                ERR(PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra)));
            break;

            case PF_Cmd_SMART_RENDER:
                ERR(SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra)));
            break;

			default:
			break;
		} /* switch (cmd) */

	} /* try */
	catch (PF_Err& thrown_err)
	{
		err = thrown_err;
	}

	return err;
}