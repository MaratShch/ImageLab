#include "ImageStylization.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"

constexpr float gfSevenDiv13 = 7.0f / 13.0f;
constexpr float gfFiveDiv13  = 5.0f / 13.0f;
constexpr float gfOneDiv13   = 1.0f - gfSevenDiv13 - gfFiveDiv13;
constexpr float gfFiveDiv8   = 5.0f / 8.0f;
constexpr float gfThreeDiv8  = 3.0f / 8.0f;
constexpr float gfOneDiv8    = 1.0f / 8.0f;
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

	PF_Err err = PF_Err_NONE;
	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);
	
	auto const& height_without_last = height - 1;
	auto const& width_without_last  = width - 1;

	PF_Pixel_BGRA_8u inPix00, inPix01, inPix1x, inPix10, inPix11;

	A_long x = 0, y = 0;
	float p00 = 0.f, p01 = 0.f, p1x = 0.f, p10 = 0.f, p11 = 0.f;
	float d = 0.f, eP = 0.f;

	__VECTOR_ALIGNED__
	for (y = 0; y < height_without_last; y++)
	{
		A_long const& idx = y * line_pitch;
		A_long const& next_idx = (y + 1) * line_pitch;

		/* process first pixel in first line */
		inPix00 = localSrc[idx];			/* pixel in position 0 and line 0 */
		inPix01 = localSrc[idx + 1];			/* pixel in position 1 and line 0 */
		inPix10 = localSrc[next_idx];		/* pixel in position 0 and line 1 */
		inPix11 = localSrc[next_idx + 1];	/* pixel in position 0 and line 1 */

		p00 = static_cast<float>(inPix00.R) * rgb2yuv[0] + static_cast<float>(inPix00.G) * rgb2yuv[1] + static_cast<float>(inPix00.B) * rgb2yuv[2];
		p01 = static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2];
		p10 = static_cast<float>(inPix10.R) * rgb2yuv[0] + static_cast<float>(inPix10.G) * rgb2yuv[1] + static_cast<float>(inPix10.B) * rgb2yuv[2];
		p11 = static_cast<float>(inPix11.R) * rgb2yuv[0] + static_cast<float>(inPix11.G) * rgb2yuv[1] + static_cast<float>(inPix11.B) * rgb2yuv[2];

		d = (p00 > 128.f) ? 255.f : 0.f;
		eP = p00 - d;
		localDst[idx].A = localSrc[idx].A;
		localDst[idx].B = localDst[idx].G = localDst[idx].R = static_cast<A_u_char>(d);

		localDst[idx + 1].A = localSrc[idx + 1].A;
		localDst[idx + 1].B = localDst[idx + 1].G = localDst[idx + 1].R = static_cast<A_u_char>(p01 + eP * gfSevenDiv13);

		localDst[next_idx].A = localSrc[next_idx].A;
		localDst[next_idx].B = localDst[next_idx].G = localDst[next_idx].R = static_cast<A_u_char>(p10 + eP * gfFiveDiv13);

		localDst[next_idx + 1].A = localSrc[next_idx + 1].A;
		localDst[next_idx + 1].B = localDst[next_idx + 1].G = localDst[next_idx + 1].R = static_cast<A_u_char>(p11 + eP * gfOneDiv13);

		/* process rest of pixels in first frame line */
		for (x = 1; x < width_without_last; x++)
		{
			inPix00 = inPix01;						/* pixel in position 0  and line 0	*/
			inPix01 = localSrc[idx + x + 1];		/* pixel in position 1  and line 0	*/
			inPix1x = inPix10;						/* pixel in position -1 and line 1	*/
			inPix10 = inPix11;						/* pixel in position 0  and line 1	*/
			inPix11 = localSrc[next_idx + x + 1];	/* pixel in position 0  and line 1	*/

			p00 = inPix00.R; /* B/W value already computed on previous step */
			p1x = inPix1x.R; /* B/W value already computed on previous step */
			p10 = inPix10.R; /* B/W value already computed on previous step */

			p01 = static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2];
			p11 = static_cast<float>(inPix11.R) * rgb2yuv[0] + static_cast<float>(inPix11.G) * rgb2yuv[1] + static_cast<float>(inPix11.B) * rgb2yuv[2];

			d = (p00 > 128.f) ? 255.f : 0.f;
			eP = p00 - d;

			localDst[idx + x].A = localSrc[idx + x].A;
			localDst[idx + x].B = localDst[idx + x].G = localDst[idx + x].R = static_cast<A_u_char>(d);

			localDst[idx + x + 1].A = localSrc[x + 1].A;
			localDst[idx + x + 1].B = localDst[x + 1].G = localDst[idx + x + 1].R = static_cast<A_u_char>(p01 + eP * gfSevenDiv16);

			localDst[next_idx + x - 1].B = localDst[next_idx + x - 1].G = localDst[next_idx + x - 1].R = static_cast<A_u_char>(localDst[next_idx + x - 1].R + eP * gfThreeDiv16);

			localDst[next_idx + x].B = localDst[next_idx + x].G = localDst[next_idx + x].R = static_cast<A_u_char>(localDst[next_idx + x].R + eP * gfFiveDiv16);

			localDst[next_idx + x + 1].A = localDst[next_idx + x + 1].A;
			localDst[next_idx + x + 1].B = localDst[next_idx + x + 1].G = localDst[next_idx + x + 1].R = static_cast<A_u_char>(p11 + eP * gfOneDiv16);
		}

		/* process last pixel in first line */

	}

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