#include "ImageStylization.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"

/*
Using Floyd Steinberg Dithering Algorithm
*/
static constexpr float gfSevenDiv13 = 7.0f / 13.0f;
static constexpr float gfFiveDiv13  = 5.0f / 13.0f;
static constexpr float gfOneDiv13   = 1.0f - gfSevenDiv13 - gfFiveDiv13;
static constexpr float gfSevenDiv16 = 7.0f / 16.0f;
static constexpr float gfFiveDiv16  = 5.0f / 16.0f;
static constexpr float gfOneDiv16   = 1.0f / 16.0f;
static constexpr float gfThreeDiv16 = 3.0f / 16.0f;

static constexpr int32_t B = 0;
static constexpr int32_t G = 1;
static constexpr int32_t R = 2;


static PF_Err PR_ImageStyle_ColorNewsPaper_BGRA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN float imgWindow[3][6]{};

	const PF_LayerDef*      __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

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
	int p00[3]{}; int p01[3]{}; int p10[3]{}; int p11[3]{};
	float d[3]{}; float eP[3]{};

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
		p00[R] = static_cast<int>(inPix00.R); p00[G] = static_cast<int>(inPix00.G); p00[B] = static_cast<int>(inPix00.B);
		p01[R] = static_cast<int>(inPix01.R); p01[G] = static_cast<int>(inPix01.G); p00[B] = static_cast<int>(inPix01.B);
		p10[R] = static_cast<int>(inPix10.R); p10[G] = static_cast<int>(inPix10.G); p10[B] = static_cast<int>(inPix10.B);
		p11[R] = static_cast<int>(inPix11.R); p11[G] = static_cast<int>(inPix11.G); p11[B] = static_cast<int>(inPix11.B);

		/* pick nearest intensity scale two options 0 or 255 */
		d[B] = (p00[R] >= 128) ? 255.f : 0.f;
		d[G] = (p00[G] >= 128) ? 255.f : 0.f;
		d[R] = (p00[B] >= 128) ? 255.f : 0.f;

		/* difference before and aftre selection */
		eP[B] = static_cast<float>(p00[B]) - d[B];
		eP[G] = static_cast<float>(p00[G]) - d[G];
		eP[R] = static_cast<float>(p00[R]) - d[R];

		/* save neighborhoods for temporal storage */
		imgWindow[B][1] = d[B];	imgWindow[B][2] = static_cast<float>(p01[B]) + eP[B] * gfSevenDiv13;
		imgWindow[B][3] = 0.f; 	imgWindow[B][4] = static_cast<float>(p10[B]) + eP[B] * gfFiveDiv13;
		imgWindow[B][5] = static_cast<float>(p11[B]) + eP[B] * gfOneDiv13;

		imgWindow[G][1] = d[G];	imgWindow[G][2] = static_cast<float>(p01[G]) + eP[G] * gfSevenDiv13;
		imgWindow[G][3] = 0.f; 	imgWindow[G][4] = static_cast<float>(p10[G]) + eP[G] * gfFiveDiv13;
		imgWindow[G][5] = static_cast<float>(p11[G]) + eP[G] * gfOneDiv13;

		imgWindow[R][1] = d[R];	imgWindow[R][2] = static_cast<float>(p01[R]) + eP[R] * gfSevenDiv13;
		imgWindow[R][3] = 0.f; 	imgWindow[R][4] = static_cast<float>(p10[R]) + eP[R] * gfFiveDiv13;
		imgWindow[R][5] = static_cast<float>(p11[R]) + eP[R] * gfOneDiv13;

		/* save destination pixel */
		Make_Color_pixel(localDst[idx], static_cast<A_u_char>(d[R]), static_cast<A_u_char>(d[G]), static_cast<A_u_char>(d[B]), localSrc[idx].A);

		for (x = 1; x < width_without_last; x++)
		{
			inPix01 = localSrc[idx + x + 1];	/* pixel in position 1 and line 0 */
			inPix10 = localSrc[next_idx + x + 1];	/* pixel in position 0 and line 1 */

			p01[R] = static_cast<int>(inPix01.R); p01[G] = static_cast<int>(inPix01.G); p01[B] = static_cast<int>(inPix01.B);
			p11[R] = static_cast<int>(inPix11.R); p11[G] = static_cast<int>(inPix11.G); p11[B] = static_cast<int>(inPix11.B);

			d[B] = (imgWindow[B][1] >= 128.f) ? 255.f : 0.f;
			d[G] = (imgWindow[G][1] >= 128.f) ? 255.f : 0.f;
			d[R] = (imgWindow[R][1] >= 128.f) ? 255.f : 0.f;

			eP[B] = imgWindow[B][1] - d[B];
			eP[G] = imgWindow[G][1] - d[G];
			eP[R] = imgWindow[R][1] - d[R];

			imgWindow[B][1] = static_cast<float>(p01[B]) + eP[B] * gfSevenDiv16;
			imgWindow[B][3] = imgWindow[B][4] + eP[B] * gfThreeDiv16;
			imgWindow[B][4] = static_cast<float>(p11[B]) + eP[B] * gfFiveDiv16;
			imgWindow[B][5] = static_cast<float>(p11[B]) + eP[B] * gfOneDiv16;

			imgWindow[G][1] = static_cast<float>(p01[G]) + eP[G] * gfSevenDiv16;
			imgWindow[G][3] = imgWindow[G][4] + eP[G] * gfThreeDiv16;
			imgWindow[G][4] = static_cast<float>(p11[G]) + eP[G] * gfFiveDiv16;
			imgWindow[G][5] = static_cast<float>(p11[G]) + eP[G] * gfOneDiv16;

			imgWindow[R][1] = static_cast<float>(p01[R]) + eP[R] * gfSevenDiv16;
			imgWindow[R][3] = imgWindow[R][4] + eP[R] * gfThreeDiv16;
			imgWindow[R][4] = static_cast<float>(p11[R]) + eP[R] * gfFiveDiv16;
			imgWindow[R][5] = static_cast<float>(p11[R]) + eP[R] * gfOneDiv16;

			Make_Color_pixel(localDst[idx + x], static_cast<A_u_char>(d[R]), static_cast<A_u_char>(d[G]), static_cast<A_u_char>(d[B]), localSrc[idx + x].A);
		} /* for (x = 1; x < width_without_last; x++) */

	} /* for (y = 0; y < height_without_last; y++) */

	/* process last line */
	A_long const& last_line = height_without_last * line_pitch;
	inPix00 = localSrc[last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[last_line + 1];	/* pixel in position 0 and line 0 */

	p00[R] = static_cast<int>(inPix00.R); p00[G] = static_cast<int>(inPix00.G); p00[B] = static_cast<int>(inPix00.B);
	p01[R] = static_cast<int>(inPix00.R); p00[G] = static_cast<int>(inPix00.G); p00[B] = static_cast<int>(inPix00.B);

	d[R] = (p00[R] >= 128) ? 255.f : 0.f;
	d[G] = (p00[G] >= 128) ? 255.f : 0.f;
	d[B] = (p00[B] >= 128) ? 255.f : 0.f;

	eP[R] = static_cast<float>(p00[R]) - d[R];
	eP[G] = static_cast<float>(p00[G]) - d[G];
	eP[B] = static_cast<float>(p00[B]) - d[B];

	imgWindow[R][1] = static_cast<float>(p01[R]) * eP[R];
	imgWindow[G][1] = static_cast<float>(p01[G]) * eP[G];
	imgWindow[B][1] = static_cast<float>(p01[B]) * eP[B];

	Make_Color_pixel(localDst[last_line], static_cast<A_u_char>(d[R]), static_cast<A_u_char>(d[G]), static_cast<A_u_char>(d[B]), localSrc[last_line].A);

	for (x = 1; x < width_without_last; x++)
	{
		d[R] = (imgWindow[R][1] >= 128) ? 255.f : 0.f;
		d[G] = (imgWindow[G][1] >= 128) ? 255.f : 0.f;
		d[B] = (imgWindow[B][1] >= 128) ? 255.f : 0.f;

		Make_Color_pixel(localDst[last_line + x], static_cast<A_u_char>(d[R]), static_cast<A_u_char>(d[G]), static_cast<A_u_char>(d[B]), localSrc[last_line + x].A);

		inPix01 = localSrc[last_line + x + 1];	/* pixel in next position	*/
		p01[R] = static_cast<int>(inPix01.R); p01[G] = static_cast<int>(inPix01.G); p01[B] = static_cast<int>(inPix01.B);

		/* difference before and after selection */
		eP[R] = static_cast<float>(p01[R]) - d[R];
		eP[G] = static_cast<float>(p01[G]) - d[G];
		eP[B] = static_cast<float>(p01[B]) - d[B];

		imgWindow[R][1] = static_cast<float>(p01[R]) + eP[R];
		imgWindow[G][1] = static_cast<float>(p01[G]) + eP[G];
		imgWindow[B][1] = static_cast<float>(p01[B]) + eP[B];
	}

	d[R] = (imgWindow[R][1] >= 128) ? 255.f : 0.f;
	d[G] = (imgWindow[G][1] >= 128) ? 255.f : 0.f;
	d[B] = (imgWindow[B][1] >= 128) ? 255.f : 0.f;

	Make_Color_pixel(localDst[last_line + width_without_last], static_cast<A_u_char>(d[R]), static_cast<A_u_char>(d[G]), static_cast<A_u_char>(d[B]), localSrc[last_line + width_without_last].A);

	return err;
}

#if 0
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

#endif

PF_Err PR_ImageStyle_ColorNewsPaper
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
		AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data,	kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);

	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;
	if (PF_Err_NONE == (errFormat = pixelFormatSuite->GetPixelFormat(output, &destinationPixelFormat)))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				err = PR_ImageStyle_ColorNewsPaper_BGRA_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
//				err = PR_ImageStyle_NewsPaper_VUYA_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
//				err = PR_ImageStyle_NewsPaper_VUYA_32f (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
//				err = PR_ImageStyle_NewsPaper_BGRA_16u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
//				err = PR_ImageStyle_NewsPaper_BGRA_32f (in_data, out_data, params, output);
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