#include "ColorCorrectionHSL.hpp"
#include "PrSDKAESupport.h"

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
		ColorCorrection_VersionMajor,
		ColorCorrection_VersionMinor,
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
			ColorCorrection_VersionMajor,
			ColorCorrection_VersionMinor,
			ColorCorrection_VersionSub,
			ColorCorrection_VersionStage,
			ColorCorrection_VersionBuild
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

		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u_709);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_VUYA_4444_8u);
		(*pixelFormatSuite->AddSupportedPixelFormat)(in_data->effect_ref, PrPixelFormat_RGB_444_10u);
	}

	return err;
}


static PF_Err
ParamsSetup(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_ParamDef	def;
	PF_Err		err = PF_Err_NONE;
	constexpr PF_ParamFlags flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_CANNOT_INTERP;
	constexpr PF_ParamUIFlags ui_flags = PF_PUI_NONE;

	def.flags = flags;
	def.ui_flags = ui_flags;

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_POPUP(
		ColorSpaceType,					/* pop-up name			*/
		COLOR_SPACE_MAX_TYPES,			/* numbver of variants	*/
		COLOR_SPACE_HSL,				/* default variant		*/
		ColorSpace,						/* string for pop-up	*/
		COLOR_CORRECT_SPACE_POPUP);		/* control ID			*/

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_ANGLE(
		ColorHueCoarseType,
		hue_coarse_default,
		COLOR_CORRECT_HUE_COARSE
	);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_FLOAT_SLIDERX(
		ColorHueFineLevel,
		hue_fine_min_level,
		hue_fine_max_level,
		hue_fine_min_level,
		hue_fine_max_level,
		hue_fine_def_level,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_HUE_FINE_LEVEL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_SLIDER(
		ColorSaturationCoarseLevel,
		sat_coarse_min_level,
		sat_coarse_max_level,
		sat_coarse_min_level,
		sat_coarse_max_level,
		sat_coarse_def_level,
		COLOR_SATURATION_COARSE_LEVEL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_FLOAT_SLIDERX(
		ColorSaturationFineLevel,
		sat_fine_min_level,
		sat_fine_max_level,
		sat_fine_min_level,
		sat_fine_max_level,
		sat_fine_def_level,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_SATURATION_FINE_LEVEL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_SLIDER(
		ColorLWIPCoarseLevel,
		lwip_coarse_min_level,
		lwip_coarse_max_level,
		lwip_coarse_min_level,
		lwip_coarse_max_level,
		lwip_coarse_def_level,
		COLOR_LWIP_COARSE_LEVEL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_FLOAT_SLIDERX(
		ColorLWIPFineLevel,
		lwip_fine_min_level,
		lwip_fine_max_level,
		lwip_fine_min_level,
		lwip_fine_max_level,
		lwip_fine_def_level,
		PF_Precision_TENTHS,
		0,
		0,
		COLOR_LWIP_FINE_LEVEL_SLIDER);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_BUTTON(
		LoadSettingName,
		LoadSetting,
		0,
		PF_ParamFlag_SUPERVISE,
		COLOR_LOAD_SETTING_BUTTON
	);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_BUTTON(
		SaveSettingName,
		SaveSetting,
		0,
		PF_ParamFlag_SUPERVISE,
		COLOR_SAVE_SETTING_BUTTON
	);

	AEFX_CLR_STRUCT_EX(def);
	PF_ADD_BUTTON(
		ResetSettingName,
		ResetSetting,
		0,
		PF_ParamFlag_SUPERVISE,
		COLOR_RESET_SETTING_BUTTON
	);

	out_data->num_params = COLOR_CORRECT_TOTAL_PARAMS;

	return PF_Err_NONE;
}

static PF_Err
GlobalSetdown(
	PF_InData* in_data
)
{
	return PF_Err_NONE;
}

/*****************************************************************************/
/* ADOBE PREMIER RENDERING                                                   */
/*****************************************************************************/
static PF_Err
FilterImageBGRA_8u
(
	void		*refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel8	*inP,
	PF_Pixel8	*outP
) noexcept
{
	return PF_Err_NONE;
}

static PF_Err
FilterImageBGRA_16u
(
	void		*refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel16	*inP,
	PF_Pixel16	*outP
) noexcept
{
	return PF_Err_NONE;
}

static PF_Err
FilterImageRGB_10u
(
	void		*refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel8	*inP,
	PF_Pixel8	*outP
) noexcept
{
	return PF_Err_NONE;
}


static PF_Err
FilterImageVUYA_8u
(
	void		*refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel8	*inP,
	PF_Pixel8	*outP
) noexcept
{
	return PF_Err_NONE;
}

static PF_Err
FilterImageVUYA_8u_709
(
	void		*refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel8	*inP,
	PF_Pixel8	*outP
) noexcept
{
	return PF_Err_NONE;
}

static PF_Err
FilterImageBGRA_32f(
	void			*refcon,
	A_long			xL,
	A_long			yL,
	PF_PixelFloat	*inP,
	PF_PixelFloat	*outP) noexcept
{
	return PF_Err_NONE;
}

static PF_Err
FilterImageVUYA_32f(
	void			*refcon,
	A_long			xL,
	A_long			yL,
	PF_PixelFloat	*inP,
	PF_PixelFloat	*outP) noexcept
{
	return PF_Err_NONE;
}

static PF_Err
FilterImageVUYA_32f_709(
	void			*refcon,
	A_long			xL,
	A_long			yL,
	PF_PixelFloat	*inP,
	PF_PixelFloat	*outP) noexcept
{
	return PF_Err_NONE;
}


static PF_Err
IterateFloat(
	PF_InData			*in_data,
	long				progress_base,
	long				progress_final,
	PF_EffectWorld		*src,
	void				*refcon,
	PF_Err(*pix_fn)(void *refcon, A_long x, A_long y, PF_PixelFloat *in, PF_PixelFloat *out),
	PF_EffectWorld		*dst) noexcept
{
	PF_Err	err = PF_Err_NONE;
	char	*localSrc, *localDst;
	localSrc = reinterpret_cast<char*>(src->data);
	localDst = reinterpret_cast<char*>(dst->data);

	for (int y = progress_base; y < progress_final; y++)
	{
		for (int x = 0; x < in_data->width; x++)
		{
			pix_fn(refcon,
				0,
				0,
				reinterpret_cast<PF_PixelFloat*>(localSrc),
				reinterpret_cast<PF_PixelFloat*>(localDst));
			localSrc += 16;
			localDst += 16;
		}
		localSrc += (src->rowbytes - in_data->width * 16);
		localDst += (dst->rowbytes - in_data->width * 16);
	}

	return err;
}


/*****************************************************************************/
/* ADOBE AFTEREFFECT RENDERING                                               */
/*****************************************************************************/
static PF_Err
FilterImage8(
	void		*refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel8	*inP,
	PF_Pixel8	*outP) noexcept
{
	return PF_Err_NONE;
}


static PF_Err
FilterImage16(
	void		*refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel16	*inP,
	PF_Pixel16	*outP) noexcept
{
	return PF_Err_NONE;
}


static PF_Err
FilterImage32(
	void			*refcon,
	A_long			xL,
	A_long			yL,
	PF_PixelFloat	*inP,
	PF_PixelFloat	*outP) noexcept
{
	return PF_Err_NONE;
}


static PF_Err
Render(
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	PF_Err	err = PF_Err_NONE;
	PF_Err errFormat = PF_Err_INVALID_INDEX;

	const A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

	/* Do high-bit depth rendering in Premiere Pro */
	if (PremierId == in_data->appl_id)
	{
		/* Get the Premiere pixel format suite */
		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
			AEFX_SuiteScoper<PF_PixelFormatSuite1>(
				in_data,
				kPFPixelFormatSuite,
				kPFPixelFormatSuiteVersion1,
				out_data);

		PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;
		if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
		{
			switch (destinationPixelFormat)
			{
				case PrPixelFormat_BGRA_4444_8u:
				{
					AEFX_SuiteScoper<PF_Iterate8Suite1> iterate8Suite =
						AEFX_SuiteScoper<PF_Iterate8Suite1>(
							in_data,
							kPFIterate8Suite,
							kPFIterate8SuiteVersion1,
							out_data);

					iterate8Suite->iterate(in_data,
						0,								// progress base
						linesL,							// progress final
						&params[0]->u.ld,				// src 
						NULL,							// area - null for all pixels
						nullptr,						// refcon - your custom data pointer
						FilterImageBGRA_8u,				// pixel function pointer
						output);
				}
				break;

				case PrPixelFormat_BGRA_4444_16u:
				{
					AEFX_SuiteScoper<PF_Iterate16Suite1> iterate16Suite =
						AEFX_SuiteScoper<PF_Iterate16Suite1>(
							in_data,
							kPFIterate16Suite,
							kPFIterate16SuiteVersion1,
							out_data);

					iterate16Suite->iterate(in_data,
						0,								// progress base
						linesL,							// progress final
						&params[0]->u.ld,				// src 
						NULL,							// area - null for all pixels
						nullptr,						// refcon - your custom data pointer
						FilterImageBGRA_16u,			// pixel function pointer
						output);
				}
				break;

				case PrPixelFormat_BGRA_4444_32f:
				{
					// Premiere doesn't support IterateFloatSuite1, so we've rolled our own
					IterateFloat(in_data,
						0,								// progress base
						linesL,							// progress final
						&params[0]->u.ld,				// src 
						nullptr,						// refcon - your custom data pointer
						FilterImageBGRA_32f,			// pixel function pointer
						output);

				}
				break;

				case PrPixelFormat_VUYA_4444_8u:
				{
					AEFX_SuiteScoper<PF_Iterate8Suite1> iterate8Suite =
						AEFX_SuiteScoper<PF_Iterate8Suite1>(
							in_data,
							kPFIterate8Suite,
							kPFIterate8SuiteVersion1,
							out_data);

					iterate8Suite->iterate(in_data,
						0,								// progress base
						linesL,							// progress final
						&params[0]->u.ld,				// src 
						NULL,							// area - null for all pixels
						nullptr,						// refcon - your custom data pointer
						FilterImageVUYA_8u,				// pixel function pointer
						output);
				}
				break;

				case PrPixelFormat_VUYA_4444_8u_709:
				{
					AEFX_SuiteScoper<PF_Iterate8Suite1> iterate8Suite =
						AEFX_SuiteScoper<PF_Iterate8Suite1>(
							in_data,
							kPFIterate8Suite,
							kPFIterate8SuiteVersion1,
							out_data);

					iterate8Suite->iterate(in_data,
						0,								// progress base
						linesL,							// progress final
						&params[0]->u.ld,				// src 
						NULL,							// area - null for all pixels
						nullptr,						// refcon - your custom data pointer
						FilterImageVUYA_8u_709,			// pixel function pointer
						output);

				}
				break;

				case PrPixelFormat_VUYA_4444_32f:
				{
					// Premiere doesn't support IterateFloatSuite1, so we've rolled our own
					IterateFloat(in_data,
						0,								// progress base
						linesL,							// progress final
						&params[0]->u.ld,				// src 
						nullptr,						// refcon - your custom data pointer
						FilterImageVUYA_32f,			// pixel function pointer
						output);
				}
				break;

				case PrPixelFormat_VUYA_4444_32f_709:
				{
					// Premiere doesn't support IterateFloatSuite1, so we've rolled our own
					IterateFloat(in_data,
						0,								// progress base
						linesL,							// progress final
						&params[0]->u.ld,				// src 
						nullptr,						// refcon - your custom data pointer
						FilterImageVUYA_32f_709,		// pixel function pointer
						output);

				}
				break;
				
				case PrPixelFormat_RGB_444_10u:
				{
					AEFX_SuiteScoper<PF_Iterate8Suite1> iterate8Suite =
						AEFX_SuiteScoper<PF_Iterate8Suite1>(
							in_data,
							kPFIterate8Suite,
							kPFIterate8SuiteVersion1,
							out_data);

					iterate8Suite->iterate(in_data,
						0,								// progress base
						linesL,							// progress final
						&params[0]->u.ld,				// src 
						NULL,							// area - null for all pixels
						nullptr,						// refcon - your custom data pointer
						FilterImageRGB_10u,				// pixel function pointer
						output);

				}
				break;

				default:
					/* something going wrong - let's make simple copy from input to output */
					PF_COPY(&params[0]->u.ld, output, nullptr, nullptr);
				break;
			} /* switch (destinationPixelFormat) */
		}
	} /* if (PremierId == in_data->appl_id) */
	else
	{
		if (PF_WORLD_IS_DEEP(output))
		{
			AEFX_SuiteScoper<PF_Iterate16Suite1> iterate16Suite =
				AEFX_SuiteScoper<PF_Iterate16Suite1>(
					in_data,
					kPFIterate8Suite,
					kPFIterate8SuiteVersion1,
					out_data);

			iterate16Suite->iterate(
				in_data,
				0,								// progress base
				linesL,							// progress final
				&params[0]->u.ld,				// src 
				NULL,							// area - null for all pixels
				nullptr,						// refcon - your custom data pointer
				FilterImage16,					// pixel function pointer
				output);						// dest

		}
		else
		{
			AEFX_SuiteScoper<PF_Iterate8Suite1> iterate8Suite =
				AEFX_SuiteScoper<PF_Iterate8Suite1>(
					in_data,
					kPFIterate8Suite,
					kPFIterate8SuiteVersion1,
					out_data);

			iterate8Suite->iterate(
				in_data,
				0,								// progress base
				linesL,							// progress final
				&params[0]->u.ld,				// src 
				NULL,							// area - null for all pixels
				nullptr,						// refcon - your custom data pointer
				FilterImage8,					// pixel function pointer
				output);						// dest
		}
	}

	return err;
}


static PF_Err
SmartRender(
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_SmartRenderExtra		*extraP
)
{
	PF_EffectWorld* input_worldP  = nullptr;
	PF_EffectWorld* output_worldP = nullptr;
	PF_Err	err = PF_Err_NONE;
	PF_PixelFormat	format = PF_PixelFormat_INVALID;

	AEFX_SuiteScoper<PF_HandleSuite1> handleSuite = 
		AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);

	AEFX_SuiteScoper<PF_WorldSuite2> wsP = 
		AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
	
	ERR(wsP->PF_GetPixelFormat(input_worldP, &format));

	if (!err) {

		AEFX_SuiteScoper<PF_iterateFloatSuite1> iterateFloatSuite =
			AEFX_SuiteScoper<PF_iterateFloatSuite1>(in_data,
				kPFIterateFloatSuite,
				kPFIterateFloatSuiteVersion1,
				out_data);

		AEFX_SuiteScoper<PF_iterate16Suite1> iterate16Suite =
			AEFX_SuiteScoper<PF_iterate16Suite1>(in_data,
				kPFIterate16Suite,
				kPFIterate16SuiteVersion1,
				out_data);

		AEFX_SuiteScoper<PF_Iterate8Suite1> iterate8Suite =
			AEFX_SuiteScoper<PF_Iterate8Suite1>(in_data,
				kPFIterate8Suite,
				kPFIterate8SuiteVersion1,
				out_data);
		
		switch (format)
		{
			case PF_PixelFormat_ARGB128:
				iterateFloatSuite->iterate(in_data,
					0,
					output_worldP->height,
					input_worldP,
					NULL,
					nullptr,
					FilterImage32,
					output_worldP);
			break;

			case PF_PixelFormat_ARGB64:
				iterate16Suite->iterate(in_data,
					0,
					output_worldP->height,
					input_worldP,
					NULL,
					nullptr,
					FilterImage16,
					output_worldP);
			break;

			case PF_PixelFormat_ARGB32:
				iterate8Suite->iterate(in_data,
					0,
					output_worldP->height,
					input_worldP,
					NULL,
					nullptr,
					FilterImage8,
					output_worldP);
			break;

			default:
				err = PF_Err_BAD_CALLBACK_PARAM;
			break;
		}
	}

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
				ERR(GlobalSetdown(in_data));
			break;

			case PF_Cmd_PARAMS_SETUP:
				ERR(ParamsSetup(in_data, out_data, params, output));
			break;

			case PF_Cmd_RENDER:
				ERR(Render(in_data, out_data, params, output));
			break;

			default:
			break;
		}
	}
	catch (PF_Err &thrown_err)
	{
		err = thrown_err;
	}

	return err;
}