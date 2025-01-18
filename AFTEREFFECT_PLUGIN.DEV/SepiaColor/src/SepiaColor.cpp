#include "SepiaColor.hpp"
#include "SepiaMatrix.hpp"
#include "ColorTransformMatrix.hpp"
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
		SepiaColor_VersionMajor,
		SepiaColor_VersionMinor,
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
			SepiaColor_VersionMajor,
			SepiaColor_VersionMinor,
			SepiaColor_VersionSub,
			SepiaColor_VersionStage,
			SepiaColor_VersionBuild
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
	out_data->num_params = 1;
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
	PF_Pixel_BGRA_8u* inBGRA_8uP = reinterpret_cast<PF_Pixel_BGRA_8u*>(inP);
	PF_Pixel_BGRA_8u* outBGRA_8uP= reinterpret_cast<PF_Pixel_BGRA_8u*>(outP);

	const PF_FpShort R = inBGRA_8uP->R * SepiaMatrix[0] + inBGRA_8uP->G * SepiaMatrix[1] + inBGRA_8uP->B * SepiaMatrix[2];
	const PF_FpShort G = inBGRA_8uP->R * SepiaMatrix[3] + inBGRA_8uP->G * SepiaMatrix[4] + inBGRA_8uP->B * SepiaMatrix[5];
	const PF_FpShort B = inBGRA_8uP->R * SepiaMatrix[6] + inBGRA_8uP->G * SepiaMatrix[7] + inBGRA_8uP->B * SepiaMatrix[8];

	outBGRA_8uP->R = static_cast<A_u_char>(CLAMP_VALUE(R, 0.f, 255.f));
	outBGRA_8uP->G = static_cast<A_u_char>(CLAMP_VALUE(G, 0.f, 255.f));
	outBGRA_8uP->B = static_cast<A_u_char>(CLAMP_VALUE(B, 0.f, 255.f));
	outBGRA_8uP->A = inBGRA_8uP->A;

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
	PF_Pixel_BGRA_16u* inBGRA_16uP  = reinterpret_cast<PF_Pixel_BGRA_16u*>(inP);
	PF_Pixel_BGRA_16u* outBGRA_16uP = reinterpret_cast<PF_Pixel_BGRA_16u*>(outP);

	const PF_FpShort R = inBGRA_16uP->R * SepiaMatrix[0] + inBGRA_16uP->G * SepiaMatrix[1] + inBGRA_16uP->B * SepiaMatrix[2];
	const PF_FpShort G = inBGRA_16uP->R * SepiaMatrix[3] + inBGRA_16uP->G * SepiaMatrix[4] + inBGRA_16uP->B * SepiaMatrix[5];
	const PF_FpShort B = inBGRA_16uP->R * SepiaMatrix[6] + inBGRA_16uP->G * SepiaMatrix[7] + inBGRA_16uP->B * SepiaMatrix[8];

	outBGRA_16uP->R = static_cast<A_u_short>(CLAMP_VALUE(R, 0.f, 65535.f));
	outBGRA_16uP->G = static_cast<A_u_short>(CLAMP_VALUE(G, 0.f, 65535.f));
	outBGRA_16uP->B = static_cast<A_u_short>(CLAMP_VALUE(B, 0.f, 65535.f));
	outBGRA_16uP->A = inBGRA_16uP->A;

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
	PF_Pixel_RGB_10u* inRGB_10uP  = reinterpret_cast<PF_Pixel_RGB_10u*>(inP);
	PF_Pixel_RGB_10u* outRGB_10uP = reinterpret_cast<PF_Pixel_RGB_10u*>(outP);

	const PF_FpShort R = inRGB_10uP->R * SepiaMatrix[0] + inRGB_10uP->G * SepiaMatrix[1] + inRGB_10uP->B * SepiaMatrix[2];
	const PF_FpShort G = inRGB_10uP->R * SepiaMatrix[3] + inRGB_10uP->G * SepiaMatrix[4] + inRGB_10uP->B * SepiaMatrix[5];
	const PF_FpShort B = inRGB_10uP->R * SepiaMatrix[6] + inRGB_10uP->G * SepiaMatrix[7] + inRGB_10uP->B * SepiaMatrix[8];

	outRGB_10uP->R = static_cast<A_u_long>(CLAMP_VALUE(R, 0.f, 1023.0f));
	outRGB_10uP->G = static_cast<A_u_long>(CLAMP_VALUE(G, 0.f, 1023.0f));
	outRGB_10uP->B = static_cast<A_u_long>(CLAMP_VALUE(B, 0.f, 1023.0f));

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
	PF_Pixel_VUYA_8u* inVUYA_8uP  = reinterpret_cast<PF_Pixel_VUYA_8u*>(inP);
	PF_Pixel_VUYA_8u* outVUYA_8uP = reinterpret_cast<PF_Pixel_VUYA_8u*>(outP);
	float R, G, B;
	float newR, newG, newB;
	int newY, newU, newV;
	int Y, U, V;

	const float* __restrict yuv2rgb = YUV2RGB[BT601];
	const float* __restrict rgb2yuv = RGB2YUV[BT601];

	Y = static_cast<int>(inVUYA_8uP->Y);
	U = static_cast<int>(inVUYA_8uP->U) - 128;
	V = static_cast<int>(inVUYA_8uP->V) - 128;

	R = Y * yuv2rgb[0] + U * yuv2rgb[1] + V * yuv2rgb[2];
	G = Y * yuv2rgb[3] + U * yuv2rgb[4] + V * yuv2rgb[5];
	B = Y * yuv2rgb[6] + U * yuv2rgb[7] + V * yuv2rgb[8];

	newR = R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2];
	newG = R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5];
	newB = R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8];

	newY = static_cast<int>(newR * rgb2yuv[0] + newG * rgb2yuv[1] + newB * rgb2yuv[2]);
	newU = static_cast<int>(newR * rgb2yuv[3] + newG * rgb2yuv[4] + newB * rgb2yuv[5]) + 128;
	newV = static_cast<int>(newR * rgb2yuv[6] + newG * rgb2yuv[7] + newB * rgb2yuv[8]) + 128;

	outVUYA_8uP->A = inVUYA_8uP->A;
	outVUYA_8uP->Y = static_cast<A_u_char>(CLAMP_VALUE(newY, 0, 255));
	outVUYA_8uP->U = static_cast<A_u_char>(CLAMP_VALUE(newU, 0, 255));
	outVUYA_8uP->V = static_cast<A_u_char>(CLAMP_VALUE(newV, 0, 255));

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
	PF_Pixel_VUYA_8u* inVUYA_8uP = reinterpret_cast<PF_Pixel_VUYA_8u*>(inP);
	PF_Pixel_VUYA_8u* outVUYA_8uP = reinterpret_cast<PF_Pixel_VUYA_8u*>(outP);
	float R, G, B;
	float newR, newG, newB;
	int newY, newU, newV;
	int Y, U, V;

	const float* __restrict yuv2rgb = YUV2RGB[BT709];
	const float* __restrict rgb2yuv = RGB2YUV[BT709];

	Y = static_cast<int>(inVUYA_8uP->Y);
	U = static_cast<int>(inVUYA_8uP->U) - 128;
	V = static_cast<int>(inVUYA_8uP->V) - 128;

	R = Y * yuv2rgb[0] + U * yuv2rgb[1] + V * yuv2rgb[2];
	G = Y * yuv2rgb[3] + U * yuv2rgb[4] + V * yuv2rgb[5];
	B = Y * yuv2rgb[6] + U * yuv2rgb[7] + V * yuv2rgb[8];

	newR = R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2];
	newG = R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5];
	newB = R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8];

	newY = static_cast<int>(newR * rgb2yuv[0] + newG * rgb2yuv[1] + newB * rgb2yuv[2]);
	newU = static_cast<int>(newR * rgb2yuv[3] + newG * rgb2yuv[4] + newB * rgb2yuv[5]) + 128;
	newV = static_cast<int>(newR * rgb2yuv[6] + newG * rgb2yuv[7] + newB * rgb2yuv[8]) + 128;

	outVUYA_8uP->A = inVUYA_8uP->A;
	outVUYA_8uP->Y = static_cast<A_u_char>(CLAMP_VALUE(newY, 0, 255));
	outVUYA_8uP->U = static_cast<A_u_char>(CLAMP_VALUE(newU, 0, 255));
	outVUYA_8uP->V = static_cast<A_u_char>(CLAMP_VALUE(newV, 0, 255));

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
	PF_Pixel_BGRA_32f* inBGRA_32fP  = reinterpret_cast<PF_Pixel_BGRA_32f*>(inP);
	PF_Pixel_BGRA_32f* outBGRA_32fP = reinterpret_cast<PF_Pixel_BGRA_32f*>(outP);

	const PF_FpShort R = inBGRA_32fP->R * SepiaMatrix[0] + inBGRA_32fP->G * SepiaMatrix[1] + inBGRA_32fP->B * SepiaMatrix[2];
	const PF_FpShort G = inBGRA_32fP->R * SepiaMatrix[3] + inBGRA_32fP->G * SepiaMatrix[4] + inBGRA_32fP->B * SepiaMatrix[5];
	const PF_FpShort B = inBGRA_32fP->R * SepiaMatrix[6] + inBGRA_32fP->G * SepiaMatrix[7] + inBGRA_32fP->B * SepiaMatrix[8];

	outBGRA_32fP->R = CLAMP_VALUE(R, value_black, value_white);
	outBGRA_32fP->G = CLAMP_VALUE(G, value_black, value_white);
	outBGRA_32fP->B = CLAMP_VALUE(B, value_black, value_white);
	outBGRA_32fP->A = inBGRA_32fP->A;

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
	PF_Pixel_VUYA_32f* inVUYA_32fP  = reinterpret_cast<PF_Pixel_VUYA_32f*>(inP);
	PF_Pixel_VUYA_32f* outVUYA_32fP = reinterpret_cast<PF_Pixel_VUYA_32f*>(outP);
	float R, G, B;
	float newR, newG, newB;
	float newY, newU, newV;

	const float* __restrict yuv2rgb = YUV2RGB[BT601];
	const float* __restrict rgb2yuv = RGB2YUV[BT601];

	R = inVUYA_32fP->Y * yuv2rgb[0] + inVUYA_32fP->U * yuv2rgb[1] + inVUYA_32fP->V * yuv2rgb[2];
	G = inVUYA_32fP->Y * yuv2rgb[3] + inVUYA_32fP->U * yuv2rgb[4] + inVUYA_32fP->V * yuv2rgb[5];
	B = inVUYA_32fP->Y * yuv2rgb[6] + inVUYA_32fP->U * yuv2rgb[7] + inVUYA_32fP->V * yuv2rgb[8];

	newR = CLAMP_VALUE(R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2], value_black, value_white);
	newG = CLAMP_VALUE(R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5], value_black, value_white);
	newB = CLAMP_VALUE(R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8], value_black, value_white);

	newY = newR * rgb2yuv[0] + newG * rgb2yuv[1] + newB * rgb2yuv[2];
	newU = newR * rgb2yuv[3] + newG * rgb2yuv[4] + newB * rgb2yuv[5];
	newV = newR * rgb2yuv[6] + newG * rgb2yuv[7] + newB * rgb2yuv[8];

	outVUYA_32fP->A = inVUYA_32fP->A;
	outVUYA_32fP->Y = newY;
	outVUYA_32fP->U = newU;
	outVUYA_32fP->V = newV;

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
	PF_Pixel_VUYA_32f* inVUYA_32fP = reinterpret_cast<PF_Pixel_VUYA_32f*>(inP);
	PF_Pixel_VUYA_32f* outVUYA_32fP = reinterpret_cast<PF_Pixel_VUYA_32f*>(outP);
	float R, G, B;
	float newR, newG, newB;
	float newY, newU, newV;

	const float* __restrict yuv2rgb = YUV2RGB[BT709];
	const float* __restrict rgb2yuv = RGB2YUV[BT709];

	R = inVUYA_32fP->Y * yuv2rgb[0] + inVUYA_32fP->U * yuv2rgb[1] + inVUYA_32fP->V * yuv2rgb[2];
	G = inVUYA_32fP->Y * yuv2rgb[3] + inVUYA_32fP->U * yuv2rgb[4] + inVUYA_32fP->V * yuv2rgb[5];
	B = inVUYA_32fP->Y * yuv2rgb[6] + inVUYA_32fP->U * yuv2rgb[7] + inVUYA_32fP->V * yuv2rgb[8];

	newR = CLAMP_VALUE(R * SepiaMatrix[0] + G * SepiaMatrix[1] + B * SepiaMatrix[2], value_black, value_white);
	newG = CLAMP_VALUE(R * SepiaMatrix[3] + G * SepiaMatrix[4] + B * SepiaMatrix[5], value_black, value_white);
	newB = CLAMP_VALUE(R * SepiaMatrix[6] + G * SepiaMatrix[7] + B * SepiaMatrix[8], value_black, value_white);

	newY = newR * rgb2yuv[0] + newG * rgb2yuv[1] + newB * rgb2yuv[2];
	newU = newR * rgb2yuv[3] + newG * rgb2yuv[4] + newB * rgb2yuv[5];
	newV = newR * rgb2yuv[6] + newG * rgb2yuv[7] + newB * rgb2yuv[8];

	outVUYA_32fP->A = inVUYA_32fP->A;
	outVUYA_32fP->Y = newY;
	outVUYA_32fP->U = newU;
	outVUYA_32fP->V = newV;

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
	const PF_FpShort R = inP->red * SepiaMatrix[0] + inP->green * SepiaMatrix[1] + inP->blue * SepiaMatrix[2];
	const PF_FpShort G = inP->red * SepiaMatrix[3] + inP->green * SepiaMatrix[4] + inP->blue * SepiaMatrix[5];
	const PF_FpShort B = inP->red * SepiaMatrix[6] + inP->green * SepiaMatrix[7] + inP->blue * SepiaMatrix[8];

	outP->alpha = inP->alpha;
	outP->red   = static_cast<A_u_char>(CLAMP_VALUE(R, 0.f, 255.f));
	outP->green = static_cast<A_u_char>(CLAMP_VALUE(G, 0.f, 255.f));
	outP->blue  = static_cast<A_u_char>(CLAMP_VALUE(B, 0.f, 255.f));

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
	const PF_FpShort R = inP->red * SepiaMatrix[0] + inP->green * SepiaMatrix[1] + inP->blue * SepiaMatrix[2];
	const PF_FpShort G = inP->red * SepiaMatrix[3] + inP->green * SepiaMatrix[4] + inP->blue * SepiaMatrix[5];
	const PF_FpShort B = inP->red * SepiaMatrix[6] + inP->green * SepiaMatrix[7] + inP->blue * SepiaMatrix[8];

	outP->alpha = inP->alpha;
	outP->red   = static_cast<A_u_short>(CLAMP_VALUE(R, 0.f, 32768.0f));
	outP->green = static_cast<A_u_short>(CLAMP_VALUE(G, 0.f, 32768.0f));
	outP->blue  = static_cast<A_u_short>(CLAMP_VALUE(B, 0.f, 32768.0f));

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
	const float R = inP->red * SepiaMatrix[0] + inP->green * SepiaMatrix[1] + inP->blue * SepiaMatrix[2];
	const float G = inP->red * SepiaMatrix[3] + inP->green * SepiaMatrix[4] + inP->blue * SepiaMatrix[5];
	const float B = inP->red * SepiaMatrix[6] + inP->green * SepiaMatrix[7] + inP->blue * SepiaMatrix[8];

	outP->alpha = inP->alpha;
	outP->red   = CLAMP_VALUE(R, value_black, value_white);
	outP->green = CLAMP_VALUE(G, value_black, value_white);
	outP->blue  = CLAMP_VALUE(B, value_black, value_white);

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

    ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, SEPIA_COLOR_INPUT, &input_worldP)));
    ERR (extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

    if (nullptr != input_worldP && nullptr != output_worldP)
    {
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
                err = iterateFloatSuite->iterate(in_data,
                    0,
                    output_worldP->height,
                    input_worldP,
                    NULL,
                    nullptr,
                    FilterImage32,
                    output_worldP);
                break;

            case PF_PixelFormat_ARGB64:
                err = iterate16Suite->iterate(in_data,
                    0,
                    output_worldP->height,
                    input_worldP,
                    NULL,
                    nullptr,
                    FilterImage16,
                    output_worldP);
                break;

            case PF_PixelFormat_ARGB32:
                err = iterate8Suite->iterate(in_data,
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
    }
    
    ERR(extraP->cb->checkin_layer_pixels(in_data->effect_ref, SEPIA_COLOR_INPUT));

	return err;
}


PLUGIN_ENTRY_POINT_CALL PF_Err
EffectMain (
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

            case PF_Cmd_SMART_RENDER:
                ERR(SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra)));
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