#include "ImageStylization.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"

/*
Using Floyd Steinberg Dithering Algorithm
*/
static constexpr float gfSevenDiv13{ 7.0f / 13.0f };
static constexpr float gfFiveDiv13 { 5.0f / 13.0f };
static constexpr float gfOneDiv13  { 1.0f - gfSevenDiv13 - gfFiveDiv13 };
static constexpr float gfSevenDiv16{ 7.0f / 16.0f };
static constexpr float gfFiveDiv16 { 5.0f / 16.0f };
static constexpr float gfOneDiv16  { 1.0f / 16.0f };
static constexpr float gfThreeDiv16{ 3.0f / 16.0f };


static PF_Err PR_ImageStyle_NewsPaper_BGRA_8u
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
		Make_BW_pixel(localDst[idx + width_without_last], static_cast<A_u_char>(d), localSrc[idx + width_without_last].A);
	} /* END: for (y = 0; y < height_without_last; y++) */

	/* process last line */
	A_long const& last_line = height_without_last * line_pitch;
	inPix00 = localSrc[last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[last_line + 1];	/* pixel in position 0 and line 0 */

	p00 = static_cast<float>(inPix00.R) * rgb2yuv[0] + static_cast<float>(inPix00.G) * rgb2yuv[1] + static_cast<float>(inPix00.B) * rgb2yuv[2];
	p01 = static_cast<float>(inPix00.R) * rgb2yuv[0] + static_cast<float>(inPix00.G) * rgb2yuv[1] + static_cast<float>(inPix00.B) * rgb2yuv[2];

	d = (p00 >= 128.f ? 255.f : 0.f);
	eP = p00 - d;
	imgWindow[1] = p01 * eP;
	Make_BW_pixel(localDst[last_line], static_cast<A_u_char>(d), localSrc[last_line].A);

	for (x = 1; x < width_without_last; x++)
	{
		d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
		Make_BW_pixel(localDst[last_line + x], static_cast<A_u_char>(d), localSrc[last_line + x].A);

		inPix01 = localSrc[last_line + x + 1];	/* pixel in next position	*/
		p01 = static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2];

		/* difference before and after selection */
		eP = p01 - d;
		imgWindow[1] = p01 + eP;
	}

	d = (imgWindow[1] >= 128.f ? 255.f : 0.f);
	Make_BW_pixel(localDst[last_line + width_without_last], static_cast<A_u_char>(d), localSrc[last_line + width_without_last].A);

	return err;
}


static PF_Err PR_ImageStyle_NewsPaper_VUYA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*        __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	float imgWindow[6]{};

	PF_Err err = PF_Err_NONE;
	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	auto const& height_without_last = height - 1;
	auto const& width_without_last = width - 1;

	PF_Pixel_VUYA_8u inPix00, /* curent pixel									*/
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

		inPix00 = localSrc[idx];			/* pixel in position 0 and line 0 */
		inPix01 = localSrc[idx + 1];		/* pixel in position 1 and line 0 */
		inPix10 = localSrc[next_idx];		/* pixel in position 0 and line 1 */
		inPix11 = localSrc[next_idx + 1];	/* pixel in position 0 and line 1 */

		/* get Luma for current pixel and for neighborhoods */
		p00 = static_cast<float>(inPix00.Y);
		p01 = static_cast<float>(inPix01.Y);
		p10 = static_cast<float>(inPix10.Y);
		p11 = static_cast<float>(inPix11.Y);

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
		localDst[idx].A = localSrc[idx].A;
		localDst[idx].V = localDst[idx].U = static_cast<A_u_char>(0x80u);
		localDst[idx].Y = static_cast<A_u_char>(d);

		/* process rest of pixels in first frame line */
		for (x = 1; x < width_without_last; x++)
		{
			inPix01 = localSrc[idx + x + 1];	/* pixel in position 1 and line 0 */
			inPix10 = localSrc[next_idx + x + 1];	/* pixel in position 0 and line 1 */
			
			p01 = static_cast<float>(inPix01.Y);
			p11 = static_cast<float>(inPix11.Y);

			d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
			eP = imgWindow[1] - d;

			imgWindow[1] = p01 + eP * gfSevenDiv16;
			imgWindow[3] = imgWindow[4] + eP * gfThreeDiv16;
			imgWindow[4] = p11 + eP * gfFiveDiv16;
			imgWindow[5] = p11 + eP * gfOneDiv16;

			localDst[idx + x].A = localSrc[idx + x].A;
			localDst[idx + x].V = localDst[idx + x].U = static_cast<A_u_char>(0x80u);
			localDst[idx + x].Y = static_cast<A_u_char>(d);
		} /* END: for (x = 1; x < width_without_last; x++) */

		  /* process last pixel in the line */
			d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
			localDst[idx + x].A = localSrc[idx + x].A;
			localDst[idx + x].V = localDst[idx + x].U = static_cast<A_u_char>(0x80u);
			localDst[idx + x].Y = static_cast<A_u_char>(d);
	} /* END: for (y = 0; y < height_without_last; y++) */

	/* process last line */
	A_long const& last_line = height_without_last * line_pitch;
	inPix00 = localSrc[last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[last_line + 1];	/* pixel in position 0 and line 0 */

	p00 = static_cast<float>(inPix00.Y);
	p01 = static_cast<float>(inPix00.Y);

	d = (p00 >= 128.f ? 255.f : 0.f);
	eP = p00 - d;
	imgWindow[1] = p01 * eP;
	localDst[last_line].A = localSrc[last_line].A;
	localDst[last_line].V = localDst[last_line].U = static_cast<A_u_char>(0x80u);
	localDst[last_line].Y = static_cast<A_u_char>(d);

	for (x = 1; x < width_without_last; x++)
	{
		d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
		localDst[last_line + x].A = localSrc[last_line + x].A;
		localDst[last_line + x].V = localDst[last_line + x].U = static_cast<A_u_char>(0x80u);
		localDst[last_line + x].Y = static_cast<A_u_char>(d);

		inPix01 = localSrc[last_line + x + 1];	/* pixel in next position	*/
		p01 = static_cast<float>(inPix01.Y);

		/* difference before and after selection */
		eP = p01 - d;
		imgWindow[1] = p01 + eP;
	}

	d = (imgWindow[1] >= 128.f ? 255.f : 0.f);
	localDst[last_line + width_without_last].A = localSrc[last_line + width_without_last].A;
	localDst[last_line + width_without_last].V = localDst[last_line + width_without_last].U = static_cast<A_u_char>(0x80u);
	localDst[last_line + width_without_last].Y = static_cast<A_u_char>(d);

	return err;
}


static PF_Err PR_ImageStyle_NewsPaper_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

	float imgWindow[6]{};

	PF_Err err = PF_Err_NONE;
	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

	auto const& height_without_last = height - 1;
	auto const& width_without_last = width - 1;

	PF_Pixel_VUYA_32f inPix00, /* curent pixel									*/
					  inPix01, /* pixel in same line and in raw position plus 1	*/
		              inPix10, /* pixel on next line in same raw postion		*/
		              inPix11; /* pixel on next line in raw position plus 1		*/

	A_long x, y;
	float p00 = 0.f, p01 = 0.f, p10 = 0.f, p11 = 0.f;
	float d = 0.f, eP = 0.f;

	constexpr float OneDiv255 = 1.0f / 255.0f;

	__VECTOR_ALIGNED__
	for (y = 0; y < height_without_last; y++)
	{
		A_long const& idx = y * line_pitch;				/* current frame line	*/
		A_long const& next_idx = (y + 1) * line_pitch;	/* next frame line		*/

		inPix00 = localSrc[idx];			/* pixel in position 0 and line 0 */
		inPix01 = localSrc[idx + 1];		/* pixel in position 1 and line 0 */
		inPix10 = localSrc[next_idx];		/* pixel in position 0 and line 1 */
		inPix11 = localSrc[next_idx + 1];	/* pixel in position 0 and line 1 */

		/* get Luma for current pixel and for neighborhoods */
		p00 = inPix00.Y * 255.f;
		p01 = inPix01.Y * 255.f;
		p10 = inPix10.Y * 255.f;
		p11 = inPix11.Y * 255.f;

		/* pick nearest intensity scale two options Black or White */
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
		localDst[idx].A = localSrc[idx].A;
		localDst[idx].V = localDst[idx].U = 0.f;
		localDst[idx].Y = MIN_VALUE(d * OneDiv255, f32_value_white);

			/* process rest of pixels in first frame line */
			for (x = 1; x < width_without_last; x++)
			{
				inPix01 = localSrc[idx + x + 1];	/* pixel in position 1 and line 0 */
				inPix10 = localSrc[next_idx + x + 1];	/* pixel in position 0 and line 1 */

				p01 = inPix01.Y * 255.f;
				p11 = inPix11.Y * 255.f;

				d = (imgWindow[1] >= 128.0f) ? 255.f : 0.f;
				eP = imgWindow[1] - d;

				imgWindow[1] = p01 + eP * gfSevenDiv16;
				imgWindow[3] = imgWindow[4] + eP * gfThreeDiv16;
				imgWindow[4] = p11 + eP * gfFiveDiv16;
				imgWindow[5] = p11 + eP * gfOneDiv16;

				localDst[idx + x].A = localSrc[idx + x].A;
				localDst[idx + x].V = localDst[idx + x].U = 0.f;
				localDst[idx + x].Y = MIN_VALUE(d * OneDiv255, f32_value_white);
			} /* END: for (x = 1; x < width_without_last; x++) */

			  /* process last pixel in the line */
			d = (imgWindow[1] >= 128.0f) ? 255.f : f32_value_black;
			localDst[idx + x].A = localSrc[idx + x].A;
			localDst[idx + x].V = localDst[idx + x].U = 0.f;
			localDst[idx + x].Y = MIN_VALUE(d * OneDiv255, f32_value_white);
		} /* END: for (y = 0; y < height_without_last; y++) */

	/* process last line */
	A_long const& last_line = height_without_last * line_pitch;
	inPix00 = localSrc[last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[last_line + 1];	/* pixel in position 0 and line 0 */

	p00 = inPix00.Y * 255.f;
	p01 = inPix00.Y * 255.f;

	d = (p00 >= 128.f ? 255.f : 0.f);
	eP = p00 - d;
	imgWindow[1] = p01 * eP;
	localDst[last_line].A = localSrc[last_line].A;
	localDst[last_line].V = localDst[last_line].U = 0.f;
	localDst[last_line].Y = MIN_VALUE(d * OneDiv255, f32_value_white);

	for (x = 1; x < width_without_last; x++)
	{
		d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
		localDst[last_line + x].A = localSrc[last_line + x].A;
		localDst[last_line + x].V = localDst[last_line + x].U = 0.f;
		localDst[last_line + x].Y = MIN_VALUE(d * OneDiv255, f32_value_white);

		inPix01 = localSrc[last_line + x + 1];	/* pixel in next position	*/
		p01 = inPix01.Y * 255.f;

		/* difference before and after selection */
		eP = p01 - d;
		imgWindow[1] = p01 + eP;
	}

	d = (imgWindow[1] >= 128.f ? f32_value_white : f32_value_black);
	localDst[last_line + width_without_last].A = localSrc[last_line + width_without_last].A;
	localDst[last_line + width_without_last].V = localDst[last_line + width_without_last].U = 0.f;
	localDst[last_line + width_without_last].Y = d;

	return err;
}


static PF_Err PR_ImageStyle_NewsPaper_BGRA_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	const float* __restrict rgb2yuv = RGB2YUV[0];

	float imgWindow[6]{};

	PF_Err err = PF_Err_NONE;
	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	auto const& height_without_last = height - 1;
	auto const& width_without_last = width - 1;

	PF_Pixel_BGRA_16u inPix00, /* curent pixel									*/
					  inPix01, /* pixel in same line and in raw position plus 1	*/
					  inPix10, /* pixel on next line in same raw postion			*/
					  inPix11; /* pixel on next line in raw position plus 1		*/

	constexpr float fOneDiv128 = 1.f / 128.f;

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
		p00 = (static_cast<float>(inPix00.R) * rgb2yuv[0] + static_cast<float>(inPix00.G) * rgb2yuv[1] + static_cast<float>(inPix00.B) * rgb2yuv[2]) * fOneDiv128;
		p01 = (static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2]) * fOneDiv128;
		p10 = (static_cast<float>(inPix10.R) * rgb2yuv[0] + static_cast<float>(inPix10.G) * rgb2yuv[1] + static_cast<float>(inPix10.B) * rgb2yuv[2]) * fOneDiv128;
		p11 = (static_cast<float>(inPix11.R) * rgb2yuv[0] + static_cast<float>(inPix11.G) * rgb2yuv[1] + static_cast<float>(inPix11.B) * rgb2yuv[2]) * fOneDiv128;
		
		/* pick nearest intensity scale two options 0 or 255 */
		d = (p00 >= 128.f) ? 255.f : 0.f;
		/* difference before and after selection */
		eP = p00 - d;

		/* save neighborhoods for temporal storage */
		imgWindow[0] = 0.f;
		imgWindow[1] = d;
		imgWindow[2] = p01 + eP * gfSevenDiv13;
		imgWindow[3] = 0.f;
		imgWindow[4] = p10 + eP * gfFiveDiv13;
		imgWindow[5] = p11 + eP * gfOneDiv13;

		/* save destination pixel */
		Make_BW_pixel(localDst[idx], static_cast<A_u_short>(d * 128.f), localSrc[idx].A);

			/* process rest of pixels in first frame line */
			for (x = 1; x < width_without_last; x++)
			{
				inPix01 = localSrc[idx + x + 1];	/* pixel in position 1 and line 0 */
				inPix10 = localSrc[next_idx + x + 1];	/* pixel in position 0 and line 1 */

				p01 = (static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2]) * fOneDiv128;
				p11 = (static_cast<float>(inPix11.R) * rgb2yuv[0] + static_cast<float>(inPix11.G) * rgb2yuv[1] + static_cast<float>(inPix11.B) * rgb2yuv[2]) * fOneDiv128;

				d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
				eP = imgWindow[1] - d;

				imgWindow[1] = p01 + eP * gfSevenDiv16;
				imgWindow[3] = imgWindow[4] + eP * gfThreeDiv16;
				imgWindow[4] = p11 + eP * gfFiveDiv16;
				imgWindow[5] = p11 + eP * gfOneDiv16;

				Make_BW_pixel(localDst[idx + x], static_cast<A_u_short>(d * 128.f), localSrc[idx + x].A);
			} /* END: for (x = 1; x < width_without_last; x++) */

		 /* process last pixel in the line */
		d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
		Make_BW_pixel(localDst[idx + width_without_last], static_cast<A_u_short>(d * 128.f), localSrc[idx + width_without_last].A);
	} /* END: for (y = 0; y < height_without_last; y++) */

	/* process last line */
	A_long const& last_line = height_without_last * line_pitch;
	inPix00 = localSrc[last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[last_line + 1];	/* pixel in position 0 and line 0 */

	p00 = (static_cast<float>(inPix00.R) * rgb2yuv[0] + static_cast<float>(inPix00.G) * rgb2yuv[1] + static_cast<float>(inPix00.B) * rgb2yuv[2]) * fOneDiv128;
	p01 = (static_cast<float>(inPix00.R) * rgb2yuv[0] + static_cast<float>(inPix00.G) * rgb2yuv[1] + static_cast<float>(inPix00.B) * rgb2yuv[2]) * fOneDiv128;

	d = (p00 >= 128.f ? 255.f : 0.f);
	eP = p00 - d;
	imgWindow[1] = p01 * eP;
	Make_BW_pixel(localDst[last_line], static_cast<A_u_short>(d * 128.f), localSrc[last_line].A);

	for (x = 1; x < width_without_last; x++)
	{
		d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
		Make_BW_pixel(localDst[last_line + x], static_cast<A_u_short>(d * 128.f), localSrc[last_line + x].A);

		inPix01 = localSrc[last_line + x + 1];	/* pixel in next position	*/
		p01 = (static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2]) * fOneDiv128;

		/* difference before and after selection */
		eP = p01 - d;
		imgWindow[1] = p01 + eP;
	}

	d = (imgWindow[1] >= 128.f ? 255.f : 0.f);
	Make_BW_pixel(localDst[last_line + width_without_last], static_cast<A_u_short>(d * 128.f), localSrc[last_line + width_without_last].A);

	return err;
}


static PF_Err PR_ImageStyle_NewsPaper_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	constexpr float  rgb2yuv[3] = { RGB2YUV[0][0] * 256.f, RGB2YUV[0][1] * 256.f, RGB2YUV[0][2] * 256.f };
	constexpr float  fOneDiv256 = 1.0f / 256.0f;

	float imgWindow[6]{};

	PF_Err err = PF_Err_NONE;
	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	auto const& height_without_last = height - 1;
	auto const& width_without_last = width - 1;

	PF_Pixel_BGRA_32f inPix00, /* curent pixel									*/
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
		p00 = inPix00.R * rgb2yuv[0] + inPix00.G * rgb2yuv[1] + inPix00.B * rgb2yuv[2];
		p01 = inPix01.R * rgb2yuv[0] + inPix01.G * rgb2yuv[1] + inPix01.B * rgb2yuv[2];
		p10 = inPix10.R * rgb2yuv[0] + inPix10.G * rgb2yuv[1] + inPix10.B * rgb2yuv[2];
		p11 = inPix11.R * rgb2yuv[0] + inPix11.G * rgb2yuv[1] + inPix11.B * rgb2yuv[2];

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
		Make_BW_pixel(localDst[idx], d * fOneDiv256, localSrc[idx].A);

		/* process rest of pixels in first frame line */
		for (x = 1; x < width_without_last; x++)
		{
			inPix01 = localSrc[idx + x + 1];	/* pixel in position 1 and line 0 */
			inPix10 = localSrc[next_idx + x + 1];	/* pixel in position 0 and line 1 */

			p01 = inPix01.R * rgb2yuv[0] + inPix01.G * rgb2yuv[1] + inPix01.B * rgb2yuv[2];
			p11 = inPix11.R * rgb2yuv[0] + inPix11.G * rgb2yuv[1] + inPix11.B * rgb2yuv[2];

			d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
			eP = imgWindow[1] - d;

			imgWindow[1] = p01 + eP * gfSevenDiv16;
			imgWindow[3] = imgWindow[4] + eP * gfThreeDiv16;
			imgWindow[4] = p11 + eP * gfFiveDiv16;
			imgWindow[5] = p11 + eP * gfOneDiv16;

			Make_BW_pixel(localDst[idx + x], d * fOneDiv256, localSrc[idx + x].A);
		} /* END: for (x = 1; x < width_without_last; x++) */

			  /* process last pixel in the line */
		d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
		Make_BW_pixel(localDst[idx + width_without_last], d * fOneDiv256, localSrc[idx + width_without_last].A);
	} /* END: for (y = 0; y < height_without_last; y++) */

		  /* process last line */
	A_long const& last_line = height_without_last * line_pitch;
	inPix00 = localSrc[last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[last_line + 1];	/* pixel in position 0 and line 0 */

	p00 = inPix00.R * rgb2yuv[0] + inPix00.G * rgb2yuv[1] + inPix00.B * rgb2yuv[2];
	p01 = inPix00.R * rgb2yuv[0] + inPix00.G * rgb2yuv[1] + inPix00.B * rgb2yuv[2];

	d = (p00 >= 128.f ? 255.f : 0.f);
	eP = p00 - d;
	imgWindow[1] = p01 * eP;
	Make_BW_pixel(localDst[last_line], d * fOneDiv256, localSrc[last_line].A);

	for (x = 1; x < width_without_last; x++)
	{
		d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
		Make_BW_pixel(localDst[last_line + x], d * fOneDiv256, localSrc[last_line + x].A);

		inPix01 = localSrc[last_line + x + 1];	/* pixel in next position	*/
		p01 = inPix01.R * rgb2yuv[0] + inPix01.G * rgb2yuv[1] + inPix01.B * rgb2yuv[2];

		/* difference before and after selection */
		eP = p01 - d;
		imgWindow[1] = p01 + eP;
	}

	d = (imgWindow[1] >= 128.f ? 255.f : 0.f);
	Make_BW_pixel(localDst[last_line + width_without_last], d * fOneDiv256, localSrc[last_line + width_without_last].A);

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
				err = PR_ImageStyle_NewsPaper_BGRA_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageStyle_NewsPaper_VUYA_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageStyle_NewsPaper_VUYA_32f (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageStyle_NewsPaper_BGRA_16u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageStyle_NewsPaper_BGRA_32f (in_data, out_data, params, output);
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


PF_Err AE_ImageStyle_NewsPaper_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_ARGB_8u*     __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input->data);
	PF_Pixel_ARGB_8u*     __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);

	const float* __restrict rgb2yuv = RGB2YUV[0];

	float imgWindow[6]{};

	const A_long& height = output->height;
	const A_long& width = output->width;
	const A_long& src_line_pitch = input->rowbytes  / sizeof(PF_Pixel8);
	const A_long& dst_line_pitch = output->rowbytes / sizeof(PF_Pixel8);

	PF_Err err = PF_Err_NONE;

	auto const& height_without_last = height - 1;
	auto const& width_without_last = width - 1;

	PF_Pixel_ARGB_8u inPix00, /* curent pixel									*/
					 inPix01, /* pixel in same line and in raw position plus 1	*/
					 inPix10, /* pixel on next line in same raw postion			*/
					 inPix11; /* pixel on next line in raw position plus 1		*/

	A_long x, y;
	float p00 = 0.f, p01 = 0.f, p10 = 0.f, p11 = 0.f;
	float d = 0.f, eP = 0.f;

	__VECTOR_ALIGNED__
	for (y = 0; y < height_without_last; y++)
	{
		A_long const& src_idx = y * src_line_pitch;				/* current frame line	*/
		A_long const& dst_idx = y * dst_line_pitch;				/* current frame line	*/
		A_long const& src_next_idx = (y + 1) * src_line_pitch;	/* next frame line		*/

		/* process first pixel in first line */
		inPix00 = localSrc[src_idx];			/* pixel in position 0 and line 0 */
		inPix01 = localSrc[src_idx + 1];		/* pixel in position 1 and line 0 */
		inPix10 = localSrc[src_next_idx];		/* pixel in position 0 and line 1 */
		inPix11 = localSrc[src_next_idx + 1];	/* pixel in position 0 and line 1 */

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
		Make_BW_pixel(localDst[dst_idx], static_cast<A_u_char>(d), localSrc[src_idx].A);

		/* process rest of pixels in first frame line */
		for (x = 1; x < width_without_last; x++)
		{
			inPix01 = localSrc[src_idx + x + 1];	/* pixel in position 1 and line 0 */
			inPix10 = localSrc[src_next_idx + x + 1];	/* pixel in position 0 and line 1 */

			p01 = static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2];
			p11 = static_cast<float>(inPix11.R) * rgb2yuv[0] + static_cast<float>(inPix11.G) * rgb2yuv[1] + static_cast<float>(inPix11.B) * rgb2yuv[2];

			d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
			eP = imgWindow[1] - d;

			imgWindow[1] = p01 + eP * gfSevenDiv16;
			imgWindow[3] = imgWindow[4] + eP * gfThreeDiv16;
			imgWindow[4] = p11 + eP * gfFiveDiv16;
			imgWindow[5] = p11 + eP * gfOneDiv16;

			Make_BW_pixel(localDst[dst_idx + x], static_cast<A_u_char>(d), localSrc[src_idx + x].A);
		} /* END: for (x = 1; x < width_without_last; x++) */

		/* process last pixel in the line */
		d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
		Make_BW_pixel(localDst[dst_idx + width_without_last], static_cast<A_u_char>(d), localSrc[src_idx + width_without_last].A);
	} /* END: for (y = 0; y < height_without_last; y++) */

	/* process last line */
	A_long const& src_last_line = height_without_last * src_line_pitch;
	A_long const& dst_last_line = height_without_last * dst_line_pitch;
	inPix00 = localSrc[src_last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[src_last_line + 1];	/* pixel in position 0 and line 0 */

	p00 = static_cast<float>(inPix00.R) * rgb2yuv[0] + static_cast<float>(inPix00.G) * rgb2yuv[1] + static_cast<float>(inPix00.B) * rgb2yuv[2];
	p01 = static_cast<float>(inPix00.R) * rgb2yuv[0] + static_cast<float>(inPix00.G) * rgb2yuv[1] + static_cast<float>(inPix00.B) * rgb2yuv[2];

	d = (p00 >= 128.f ? 255.f : 0.f);
	eP = p00 - d;
	imgWindow[1] = p01 * eP;
	Make_BW_pixel(localDst[dst_last_line], static_cast<A_u_char>(d), localSrc[src_last_line].A);

	for (x = 1; x < width_without_last; x++)
	{
		d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
		Make_BW_pixel(localDst[dst_last_line + x], static_cast<A_u_char>(d), localSrc[src_last_line + x].A);

		inPix01 = localSrc[src_last_line + x + 1];	/* pixel in next position	*/
		p01 = static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2];

		/* difference before and after selection */
		eP = p01 - d;
		imgWindow[1] = p01 + eP;
	}

	d = (imgWindow[1] >= 128.f ? 255.f : 0.f);
	Make_BW_pixel(localDst[dst_last_line + width_without_last], static_cast<A_u_char>(d), localSrc[src_last_line + width_without_last].A);

	return err;
}


PF_Err AE_ImageStyle_NewsPaper_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_ARGB_16u*    __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(input->data);
	PF_Pixel_ARGB_16u*    __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output->data);

	const float* __restrict rgb2yuv = RGB2YUV[0];

	float imgWindow[6]{};

	const A_long& height = output->height;
	const A_long& width = output->width;
	const A_long& src_line_pitch = input->rowbytes  / sizeof(PF_Pixel16);
	const A_long& dst_line_pitch = output->rowbytes / sizeof(PF_Pixel16);

	PF_Err err = PF_Err_NONE;

	auto const& height_without_last = height - 1;
	auto const& width_without_last = width - 1;

	PF_Pixel_ARGB_16u inPix00, /* curent pixel									*/
					  inPix01, /* pixel in same line and in raw position plus 1	*/
					  inPix10, /* pixel on next line in same raw postion		*/
					  inPix11; /* pixel on next line in raw position plus 1		*/

	constexpr float fOneDiv128 = 1.f / 128.f;

	A_long x, y;
	float p00 = 0.f, p01 = 0.f, p10 = 0.f, p11 = 0.f;
	float d = 0.f, eP = 0.f;

	__VECTOR_ALIGNED__
	for (y = 0; y < height_without_last; y++)
	{
		A_long const& src_idx = y * src_line_pitch;				/* current frame line	*/
		A_long const& dst_idx = y * dst_line_pitch;				/* current frame line	*/
		A_long const& src_next_idx = (y + 1) * src_line_pitch;	/* next frame line		*/

		/* process first pixel in first line */
		inPix00 = localSrc[src_idx];			/* pixel in position 0 and line 0 */
		inPix01 = localSrc[src_idx + 1];		/* pixel in position 1 and line 0 */
		inPix10 = localSrc[src_next_idx];		/* pixel in position 0 and line 1 */
		inPix11 = localSrc[src_next_idx + 1];	/* pixel in position 0 and line 1 */

		/* compute Luma for current pixel and for neighborhoods */
		p00 = (static_cast<float>(inPix00.R) * rgb2yuv[0] + static_cast<float>(inPix00.G) * rgb2yuv[1] + static_cast<float>(inPix00.B) * rgb2yuv[2]) * fOneDiv128;
		p01 = (static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2]) * fOneDiv128;
		p10 = (static_cast<float>(inPix10.R) * rgb2yuv[0] + static_cast<float>(inPix10.G) * rgb2yuv[1] + static_cast<float>(inPix10.B) * rgb2yuv[2]) * fOneDiv128;
		p11 = (static_cast<float>(inPix11.R) * rgb2yuv[0] + static_cast<float>(inPix11.G) * rgb2yuv[1] + static_cast<float>(inPix11.B) * rgb2yuv[2]) * fOneDiv128;

		/* pick nearest intensity scale two options 0 or 255 */
		d = (p00 >= 128.f) ? 255.f : 0.f;
		/* difference before and after selection */
		eP = p00 - d;

		/* save neighborhoods for temporal storage */
		imgWindow[0] = 0.f;
		imgWindow[1] = d;
		imgWindow[2] = p01 + eP * gfSevenDiv13;
		imgWindow[3] = 0.f;
		imgWindow[4] = p10 + eP * gfFiveDiv13;
		imgWindow[5] = p11 + eP * gfOneDiv13;

		/* save destination pixel */
		Make_BW_pixel(localDst[dst_idx], static_cast<A_u_short>(d * 128.f), localSrc[src_idx].A);

		/* process rest of pixels in first frame line */
		for (x = 1; x < width_without_last; x++)
		{
			inPix01 = localSrc[src_idx + x + 1];		/* pixel in position 1 and line 0 */
			inPix10 = localSrc[src_next_idx + x + 1];	/* pixel in position 0 and line 1 */

			p01 = (static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2]) * fOneDiv128;
			p11 = (static_cast<float>(inPix11.R) * rgb2yuv[0] + static_cast<float>(inPix11.G) * rgb2yuv[1] + static_cast<float>(inPix11.B) * rgb2yuv[2]) * fOneDiv128;

			d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
			eP = imgWindow[1] - d;

			imgWindow[1] = p01 + eP * gfSevenDiv16;
			imgWindow[3] = imgWindow[4] + eP * gfThreeDiv16;
			imgWindow[4] = p11 + eP * gfFiveDiv16;
			imgWindow[5] = p11 + eP * gfOneDiv16;

			Make_BW_pixel(localDst[dst_idx + x], static_cast<A_u_short>(d * 128.f), localSrc[src_idx + x].A);
		} /* END: for (x = 1; x < width_without_last; x++) */

		/* process last pixel in the line */
		d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
		Make_BW_pixel(localDst[dst_idx + width_without_last], static_cast<A_u_short>(d * 128.f), localSrc[src_idx + width_without_last].A);
	} /* END: for (y = 0; y < height_without_last; y++) */

	/* process last line */
	A_long const& src_last_line = height_without_last * src_line_pitch;
	A_long const& dst_last_line = height_without_last * dst_line_pitch;
	inPix00 = localSrc[src_last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[src_last_line + 1];	/* pixel in position 0 and line 0 */

	p00 = (static_cast<float>(inPix00.R) * rgb2yuv[0] + static_cast<float>(inPix00.G) * rgb2yuv[1] + static_cast<float>(inPix00.B) * rgb2yuv[2]) * fOneDiv128;
	p01 = (static_cast<float>(inPix00.R) * rgb2yuv[0] + static_cast<float>(inPix00.G) * rgb2yuv[1] + static_cast<float>(inPix00.B) * rgb2yuv[2]) * fOneDiv128;

	d = (p00 >= 128.f ? 255.f : 0.f);
	eP = p00 - d;
	imgWindow[1] = p01 * eP;
	Make_BW_pixel(localDst[dst_last_line], static_cast<A_u_short>(d * 128.f), localSrc[src_last_line].A);

	for (x = 1; x < width_without_last; x++)
	{
		d = (imgWindow[1] >= 128.f) ? 255.f : 0.f;
		Make_BW_pixel(localDst[dst_last_line + x], static_cast<A_u_short>(d * 128.f), localSrc[src_last_line + x].A);

		inPix01 = localSrc[src_last_line + x + 1];	/* pixel in next position	*/
		p01 = (static_cast<float>(inPix01.R) * rgb2yuv[0] + static_cast<float>(inPix01.G) * rgb2yuv[1] + static_cast<float>(inPix01.B) * rgb2yuv[2]) * fOneDiv128;

		/* difference before and after selection */
		eP = p01 - d;
		imgWindow[1] = p01 + eP;
	}

	d = (imgWindow[1] >= 128.f ? 255.f : 0.f);
	Make_BW_pixel(localDst[dst_last_line + width_without_last], static_cast<A_u_short>(d * 128.f), localSrc[src_last_line + width_without_last].A);

	return err;
}


