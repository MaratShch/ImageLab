#include "ImageStylization.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"

/*
Using Floyd Steinberg Dithering Algorithm
*/
constexpr float gfSevenDiv13 = 7.0f / 13.0f;
constexpr float gfFiveDiv13  = 5.0f / 13.0f;
constexpr float gfOneDiv13   = 1.0f - gfSevenDiv13 - gfFiveDiv13;
constexpr float gfSevenDiv16 = 7.0f / 16.0f;
constexpr float gfFiveDiv16  = 5.0f / 16.0f;
constexpr float gfOneDiv16   = 1.0f / 16.0f;
constexpr float gfThreeDiv16 = 3.0f / 16.0f;


static PF_Err PR_ImageStyle_NewsPaper_BGRA_4444u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*        __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	const float* __restrict rgb2yuv = RGB2YUV[0];

	float imgWindow[6]{};

	PF_Err err = PF_Err_NONE;
	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
	
	auto const& height_without_last = height - 1;
	auto const& width_without_last  = width - 1;

	PF_Pixel_BGRA_8u inPix00, /* curent pixel									*/
		             inPix01, /* pixel in same line and in raw position plus 1	*/
		             inPix10, /* pixel on next line in same raw postion			*/
		             inPix11; /* pixel on next line in raw position plus 1		*/	

	A_long x, y;
	float p00 = 0.f, p01 = 0.f, p10 = 0.f, p11 = 0.f;
	float d = 0.f, eP = 0.f;

	__VECTOR_ALIGNED__
	for (y = 0; y < height_without_last; y++)
	{
		A_long const& idx = y * line_pitch;				/* current frame line	*/
		A_long const& next_idx = (y + 1) * line_pitch;	/* next frame line		*/

		/* process first pixel in first line */
		inPix00 = localSrc[idx];			/* pixel in position 0 and line 0 */
		inPix01 = localSrc[idx + 1];		/* pixel in position 1 and line 0 */
		inPix10 = localSrc[next_idx];		/* pixel in position 0 and line 1 */
		inPix11 = localSrc[next_idx + 1];	/* pixel in position 0 and line 1 */

		/* compute Luma for current pixel and for neighborhoods */
		p00 = static_cast<float>(inPix00.R) * rgb2yuv[0] + static_cast<float>(inPix00.G) * rgb2yuv[1] + static_cast<float>(inPix00.B) * rgb2yuv[2];
		p01 = static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2];
		p10 = static_cast<float>(inPix10.R) * rgb2yuv[0] + static_cast<float>(inPix10.G) * rgb2yuv[1] + static_cast<float>(inPix10.B) * rgb2yuv[2];
		p11 = static_cast<float>(inPix11.R) * rgb2yuv[0] + static_cast<float>(inPix11.G) * rgb2yuv[1] + static_cast<float>(inPix11.B) * rgb2yuv[2];

		/* pick nearest intensity scale two options 0 or 255 */
		d = (p00 >= 128.f) ? 255.f : 0.f;
		/* difference before and aftre selection */
		eP = p00 - d;

		/* save neighborhoods for temporal storage */
		imgWindow[0] = 0.f;
		imgWindow[1] = d;
		imgWindow[2] = p01 + eP * gfSevenDiv13;
		imgWindow[3] = 0.f;
		imgWindow[4] = p10 + eP * gfFiveDiv13;
		imgWindow[5] = p11 + eP * gfOneDiv13;

		/* save destination pixel */
		Make_BW_pixel(localDst[idx], static_cast<A_u_char>(d), localSrc[idx].A);

		/* process rest of pixels in first frame line */
		for (x = 1; x < width_without_last; x++)
		{
			inPix01 = localSrc[idx + x + 1];	/* pixel in position 1 and line 0 */
			inPix10 = localSrc[next_idx + x + 1];	/* pixel in position 0 and line 1 */

			p01 = static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2];
			p11 = static_cast<float>(inPix11.R) * rgb2yuv[0] + static_cast<float>(inPix11.G) * rgb2yuv[1] + static_cast<float>(inPix11.B) * rgb2yuv[2];

			d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
			eP = imgWindow[1] - d;

			imgWindow[1] = p01 + eP * gfSevenDiv16;
			imgWindow[3] = imgWindow[4] + eP * gfThreeDiv16;
			imgWindow[4] = p11 + eP * gfFiveDiv16;
			imgWindow[5] = p11 + eP * gfOneDiv16;

			Make_BW_pixel(localDst[idx + x], static_cast<A_u_char>(d), localSrc[idx + x].A);
		} /* END: for (x = 1; x < width_without_last; x++) */

		/* process last pixel in the line */
		d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
		Make_BW_pixel(localDst[idx + x], static_cast<A_u_char>(d), localSrc[idx + x].A);
	} /* END: for (y = 0; y < height_without_last; y++) */

	return err;
}



PF_Err PR_ImageStyle_NewsPaper
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PF_Err errFormat = PF_Err_INVALID_INDEX;

	/* This plugin called frop PR - check video fomat */
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
				err = PR_ImageStyle_NewsPaper_BGRA_4444u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
			{
				auto const& isBT709 = (destinationPixelFormat == PrPixelFormat_VUYA_4444_8u_709);
			}
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
			{
				auto const& isBT709 = (destinationPixelFormat == PrPixelFormat_VUYA_4444_32f_709);
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
			break;

			case PrPixelFormat_BGRA_4444_32f:
			break;

			case PrPixelFormat_RGB_444_10u:
			break;

			default:
				err = PF_Err_INVALID_INDEX;
			break;
		}
	}
	else
	{
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}

	return err;
}