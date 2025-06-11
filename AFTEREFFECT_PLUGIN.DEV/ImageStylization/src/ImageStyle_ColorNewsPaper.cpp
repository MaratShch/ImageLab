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


static PF_Err PR_ImageStyle_ColorNewsPaper_VUYA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const bool&             isBT709 
) noexcept
{
	CACHE_ALIGN float imgWindow[3][6]{};

	const float* __restrict yuv2rgb = (true == isBT709 ? YUV2RGB[BT709] : YUV2RGB[BT601]);
	const float* __restrict rgb2yuv = (true == isBT709 ? RGB2YUV[BT709] : RGB2YUV[BT601]);

	const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

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
	float Y00, U00, V00, Y01, U01, V01, Y10, U10, V10, Y11, U11, V11;
	float p00[3]{}; float p01[3]{}; float p10[3]{}; float p11[3]{};
	float d[3]{}; float eP[3]{};
	float Y, U, V;

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

		Y00 = static_cast<float>(inPix00.Y); U00 = static_cast<float>(inPix00.U) - 128.f; V00 = static_cast<float>(inPix00.V) - 128.f;
		Y01 = static_cast<float>(inPix01.Y); U01 = static_cast<float>(inPix01.U) - 128.f; V01 = static_cast<float>(inPix01.V) - 128.f;
		Y10 = static_cast<float>(inPix10.Y); U10 = static_cast<float>(inPix10.U) - 128.f; V10 = static_cast<float>(inPix10.V) - 128.f;
		Y11 = static_cast<float>(inPix11.Y); U11 = static_cast<float>(inPix11.U) - 128.f; V11 = static_cast<float>(inPix11.V) - 128.f;

		p00[R] = Y00 * yuv2rgb[0] + U00 * yuv2rgb[1] + V00 * yuv2rgb[2];
		p00[G] = Y00 * yuv2rgb[3] + U00 * yuv2rgb[4] + V00 * yuv2rgb[5];
		p00[B] = Y00 * yuv2rgb[6] + U00 * yuv2rgb[7] + V00 * yuv2rgb[8];

		p01[R] = Y01 * yuv2rgb[0] + U01 * yuv2rgb[1] + V01 * yuv2rgb[2];
		p01[G] = Y01 * yuv2rgb[3] + U01 * yuv2rgb[4] + V01 * yuv2rgb[5];
		p01[B] = Y01 * yuv2rgb[6] + U01 * yuv2rgb[7] + V01 * yuv2rgb[8];

		p10[R] = Y10 * yuv2rgb[0] + U10 * yuv2rgb[1] + V10 * yuv2rgb[2];
		p10[G] = Y10 * yuv2rgb[3] + U10 * yuv2rgb[4] + V10 * yuv2rgb[5];
		p10[B] = Y10 * yuv2rgb[6] + U10 * yuv2rgb[7] + V10 * yuv2rgb[8];

		p11[R] = Y11 * yuv2rgb[0] + U11  * yuv2rgb[1] + V11 * yuv2rgb[2];
		p11[G] = Y11 * yuv2rgb[3] + U11  * yuv2rgb[4] + V11 * yuv2rgb[5];
		p11[B] = Y11 * yuv2rgb[6] + U11  * yuv2rgb[7] + V11 * yuv2rgb[8];

		/* pick nearest intensity scale two options 0 or 255 */
		d[B] = (p00[R] >= 128) ? 255.f : 0.f;
		d[G] = (p00[G] >= 128) ? 255.f : 0.f;
		d[R] = (p00[B] >= 128) ? 255.f : 0.f;

		/* difference before and aftre selection */
		eP[B] = p00[B] - d[B];
		eP[G] = p00[G] - d[G];
		eP[R] = p00[R] - d[R];

		/* save neighborhoods for temporal storage */
		imgWindow[B][1] = d[B];	imgWindow[B][2] = p01[B] + eP[B] * gfSevenDiv13;
		imgWindow[B][3] = 0.f; 	imgWindow[B][4] = p10[B] + eP[B] * gfFiveDiv13;
		imgWindow[B][5] = p11[B] + eP[B] * gfOneDiv13;

		imgWindow[G][1] = d[G];	imgWindow[G][2] = p01[G] + eP[G] * gfSevenDiv13;
		imgWindow[G][3] = 0.f; 	imgWindow[G][4] = p10[G] + eP[G] * gfFiveDiv13;
		imgWindow[G][5] = p11[G] + eP[G] * gfOneDiv13;

		imgWindow[R][1] = d[R];	imgWindow[R][2] = p01[R] + eP[R] * gfSevenDiv13;
		imgWindow[R][3] = 0.f; 	imgWindow[R][4] = p10[R] + eP[R] * gfFiveDiv13;
		imgWindow[R][5] = p11[R] + eP[R] * gfOneDiv13;

		Y = d[R] * rgb2yuv[0] + d[G] * rgb2yuv[1] + d[B] * rgb2yuv[2];
		U = d[R] * rgb2yuv[3] + d[G] * rgb2yuv[3] + d[B] * rgb2yuv[5];
		V = d[R] * rgb2yuv[6] + d[G] * rgb2yuv[7] + d[B] * rgb2yuv[8];

		/* save destination pixel */
		Make_Color_pixel_yuv(localDst[idx], static_cast<A_u_char>(Y), static_cast<A_u_char>(U), static_cast<A_u_char>(V), localSrc[idx].A);

		for (x = 1; x < width_without_last; x++)
		{
			inPix01 = localSrc[idx + x + 1];	/* pixel in position 1 and line 0 */
			inPix10 = localSrc[next_idx + x + 1];	/* pixel in position 0 and line 1 */

			Y01 = static_cast<float>(inPix01.Y); U01 = static_cast<float>(inPix01.U) - 128.f; V01 = static_cast<float>(inPix01.V) - 128.f;
			Y10 = static_cast<float>(inPix10.Y); U10 = static_cast<float>(inPix10.U) - 128.f; V10 = static_cast<float>(inPix10.V) - 128.f;

			p01[R] = Y01 * yuv2rgb[0] + U01 * yuv2rgb[1] + V01 * yuv2rgb[2];
			p01[G] = Y01 * yuv2rgb[3] + U01 * yuv2rgb[4] + V01 * yuv2rgb[5];
			p01[B] = Y01 * yuv2rgb[6] + U01 * yuv2rgb[7] + V01 * yuv2rgb[8];

			p10[R] = Y10 * yuv2rgb[0] + U10 * yuv2rgb[1] + V10 * yuv2rgb[2];
			p10[G] = Y10 * yuv2rgb[3] + U10 * yuv2rgb[4] + V10 * yuv2rgb[5];
			p10[B] = Y10 * yuv2rgb[6] + U10 * yuv2rgb[7] + V10 * yuv2rgb[8];

			d[B] = (imgWindow[B][1] >= 128.f) ? 255.f : 0.f;
			d[G] = (imgWindow[G][1] >= 128.f) ? 255.f : 0.f;
			d[R] = (imgWindow[R][1] >= 128.f) ? 255.f : 0.f;

			eP[B] = imgWindow[B][1] - d[B];
			eP[G] = imgWindow[G][1] - d[G];
			eP[R] = imgWindow[R][1] - d[R];

			imgWindow[B][1] = p01[B] + eP[B] * gfSevenDiv16;
			imgWindow[B][3] = imgWindow[B][4] + eP[B] * gfThreeDiv16;
			imgWindow[B][4] = p11[B] + eP[B] * gfFiveDiv16;
			imgWindow[B][5] = p11[B] + eP[B] * gfOneDiv16;

			imgWindow[G][1] = p01[G] + eP[G] * gfSevenDiv16;
			imgWindow[G][3] = imgWindow[G][4] + eP[G] * gfThreeDiv16;
			imgWindow[G][4] = p11[G] + eP[G] * gfFiveDiv16;
			imgWindow[G][5] = p11[G] + eP[G] * gfOneDiv16;

			imgWindow[R][1] = p01[R] + eP[R] * gfSevenDiv16;
			imgWindow[R][3] = imgWindow[R][4] + eP[R] * gfThreeDiv16;
			imgWindow[R][4] = p11[R] + eP[R] * gfFiveDiv16;
			imgWindow[R][5] = p11[R] + eP[R] * gfOneDiv16;

			Y = d[R] * rgb2yuv[0] + d[G] * rgb2yuv[1] + d[B] * rgb2yuv[2];
			U = d[R] * rgb2yuv[3] + d[G] * rgb2yuv[3] + d[B] * rgb2yuv[5] + 128.f;
			V = d[R] * rgb2yuv[6] + d[G] * rgb2yuv[7] + d[B] * rgb2yuv[8] + 128.f;

			Make_Color_pixel_yuv(localDst[idx + x], static_cast<A_u_char>(Y), static_cast<A_u_char>(U), static_cast<A_u_char>(V), localSrc[idx + x].A);
		} /* for (x = 1; x < width_without_last; x++) */

	} /* for (y = 0; y < height_without_last; y++) */

	/* process last line */
	A_long const& last_line = height_without_last * line_pitch;
	inPix00 = localSrc[last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[last_line + 1];	/* pixel in position 0 and line 0 */

	Y00 = static_cast<float>(inPix00.Y); U00 = static_cast<float>(inPix00.U) - 128.f; V00 = static_cast<float>(inPix00.V) - 128.f;
	Y01 = static_cast<float>(inPix01.Y); U01 = static_cast<float>(inPix01.U) - 128.f; V01 = static_cast<float>(inPix01.V) - 128.f;

	p00[R] = Y00 * yuv2rgb[0] + U00 * yuv2rgb[1] + V00 * yuv2rgb[2];
	p00[G] = Y00 * yuv2rgb[3] + U00 * yuv2rgb[4] + V00 * yuv2rgb[5];
	p00[B] = Y00 * yuv2rgb[6] + U00 * yuv2rgb[7] + V00 * yuv2rgb[8];

	p01[R] = Y01 * yuv2rgb[0] + U01 * yuv2rgb[1] + V01 * yuv2rgb[2];
	p01[G] = Y01 * yuv2rgb[3] + U01 * yuv2rgb[4] + V01 * yuv2rgb[5];
	p01[B] = Y01 * yuv2rgb[6] + U01 * yuv2rgb[7] + V01 * yuv2rgb[8];

	d[R] = (p00[R] >= 128) ? 255.f : 0.f;
	d[G] = (p00[G] >= 128) ? 255.f : 0.f;
	d[B] = (p00[B] >= 128) ? 255.f : 0.f;

	eP[R] = p00[R] - d[R];
	eP[G] = p00[G] - d[G];
	eP[B] = p00[B] - d[B];

	imgWindow[R][1] = p01[R] * eP[R];
	imgWindow[G][1] = p01[G] * eP[G];
	imgWindow[B][1] = p01[B] * eP[B];

	Y = d[R] * rgb2yuv[0] + d[G] * rgb2yuv[1] + d[B] * rgb2yuv[2];
	U = d[R] * rgb2yuv[3] + d[G] * rgb2yuv[3] + d[B] * rgb2yuv[5] + 128.f;
	V = d[R] * rgb2yuv[6] + d[G] * rgb2yuv[7] + d[B] * rgb2yuv[8] + 128.f;

	Make_Color_pixel_yuv(localDst[last_line], static_cast<A_u_char>(Y), static_cast<A_u_char>(U), static_cast<A_u_char>(V), localSrc[last_line].A);

	for (x = 1; x < width_without_last; x++)
	{
		d[R] = (imgWindow[R][1] >= 128) ? 255.f : 0.f;
		d[G] = (imgWindow[G][1] >= 128) ? 255.f : 0.f;
		d[B] = (imgWindow[B][1] >= 128) ? 255.f : 0.f;

		Y = d[R] * rgb2yuv[0] + d[G] * rgb2yuv[1] + d[B] * rgb2yuv[2];
		U = d[R] * rgb2yuv[3] + d[G] * rgb2yuv[3] + d[B] * rgb2yuv[5] + 128.f;
		V = d[R] * rgb2yuv[6] + d[G] * rgb2yuv[7] + d[B] * rgb2yuv[8] + 128.f;

		Make_Color_pixel_yuv(localDst[last_line + x], static_cast<A_u_char>(Y), static_cast<A_u_char>(U), static_cast<A_u_char>(V), localSrc[last_line + x].A);

		inPix01 = localSrc[last_line + x + 1];	/* pixel in next position	*/

		Y01 = static_cast<float>(inPix01.Y); U01 = static_cast<float>(inPix01.U) - 128.f; V01 = static_cast<float>(inPix01.V) - 128.f;
		p01[R] = Y01 * yuv2rgb[0] + U01 * yuv2rgb[1] + V01 * yuv2rgb[2];
		p01[G] = Y01 * yuv2rgb[3] + U01 * yuv2rgb[4] + V01 * yuv2rgb[5];
		p01[B] = Y01 * yuv2rgb[6] + U01 * yuv2rgb[7] + V01 * yuv2rgb[8];

		/* difference before and after selection */
		eP[R] = p01[R] - d[R];
		eP[G] = p01[G] - d[G];
		eP[B] = p01[B] - d[B];

		imgWindow[R][1] = p01[R] + eP[R];
		imgWindow[G][1] = p01[G] + eP[G];
		imgWindow[B][1] = p01[B] + eP[B];
	}

	d[R] = (imgWindow[R][1] >= 128) ? 255.f : 0.f;
	d[G] = (imgWindow[G][1] >= 128) ? 255.f : 0.f;
	d[B] = (imgWindow[B][1] >= 128) ? 255.f : 0.f;

	Y = d[R] * rgb2yuv[0] + d[G] * rgb2yuv[1] + d[B] * rgb2yuv[2];
	U = d[R] * rgb2yuv[3] + d[G] * rgb2yuv[3] + d[B] * rgb2yuv[5] + 128.f;
	V = d[R] * rgb2yuv[6] + d[G] * rgb2yuv[7] + d[B] * rgb2yuv[8] + 128.f;

	Make_Color_pixel_yuv(localDst[last_line + width_without_last], static_cast<A_u_char>(Y), static_cast<A_u_char>(U), static_cast<A_u_char>(V), localSrc[last_line + width_without_last].A);

	return err;
}


static PF_Err PR_ImageStyle_ColorNewsPaper_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const bool&             isBT709
) noexcept
{
	CACHE_ALIGN float imgWindow[3][6]{};

	const float* __restrict yuv2rgb = (true == isBT709 ? YUV2RGB[BT709] : YUV2RGB[BT601]);
	const float* __restrict rgb2yuv = (true == isBT709 ? RGB2YUV[BT709] : RGB2YUV[BT601]);

	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

	PF_Err err = PF_Err_NONE;
	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

	auto const& height_without_last = height - 1;
	auto const& width_without_last = width - 1;

	PF_Pixel_VUYA_32f inPix00, /* curent pixel									*/
					  inPix01, /* pixel in same line and in raw position plus 1	*/
					  inPix10, /* pixel on next line in same raw postion			*/
					  inPix11; /* pixel on next line in raw position plus 1		*/

	A_long x, y;
	float Y00, U00, V00, Y01, U01, V01, Y10, U10, V10, Y11, U11, V11;
	float p00[3]{}; float p01[3]{}; float p10[3]{}; float p11[3]{};
	float d[3]{}; float eP[3]{};
	float Y, U, V;

	constexpr float reciproc255 = 1.f / 255.f;

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

		Y00 = inPix00.Y * 255.f; U00 = inPix00.U * 255.f - 128.f; V00 = inPix00.V * 255.f - 128.f;
		Y01 = inPix01.Y * 255.f; U01 = inPix01.U * 255.f - 128.f; V01 = inPix01.V * 255.f - 128.f;
		Y10 = inPix10.Y * 255.f; U10 = inPix10.U * 255.f - 128.f; V10 = inPix10.V * 255.f - 128.f;
		Y11 = inPix11.Y * 255.f; U11 = inPix11.U * 255.f - 128.f; V11 = inPix11.V * 255.f - 128.f;

		p00[R] = Y00 * yuv2rgb[0] + U00 * yuv2rgb[1] + V00 * yuv2rgb[2];
		p00[G] = Y00 * yuv2rgb[3] + U00 * yuv2rgb[4] + V00 * yuv2rgb[5];
		p00[B] = Y00 * yuv2rgb[6] + U00 * yuv2rgb[7] + V00 * yuv2rgb[8];
		
		p01[R] = Y01 * yuv2rgb[0] + U01 * yuv2rgb[1] + V01 * yuv2rgb[2];
		p01[G] = Y01 * yuv2rgb[3] + U01 * yuv2rgb[4] + V01 * yuv2rgb[5];
		p01[B] = Y01 * yuv2rgb[6] + U01 * yuv2rgb[7] + V01 * yuv2rgb[8];

		p10[R] = Y10 * yuv2rgb[0] + U10 * yuv2rgb[1] + V10 * yuv2rgb[2];
		p10[G] = Y10 * yuv2rgb[3] + U10 * yuv2rgb[4] + V10 * yuv2rgb[5];
		p10[B] = Y10 * yuv2rgb[6] + U10 * yuv2rgb[7] + V10 * yuv2rgb[8];

		p11[R] = Y11 * yuv2rgb[0] + U11  * yuv2rgb[1] + V11 * yuv2rgb[2];
		p11[G] = Y11 * yuv2rgb[3] + U11  * yuv2rgb[4] + V11 * yuv2rgb[5];
		p11[B] = Y11 * yuv2rgb[6] + U11  * yuv2rgb[7] + V11 * yuv2rgb[8];

		/* pick nearest intensity scale two options 0 or 255 */
		d[B] = (p00[R] >= 128) ? 255.f : 0.f;
		d[G] = (p00[G] >= 128) ? 255.f : 0.f;
		d[R] = (p00[B] >= 128) ? 255.f : 0.f;

		/* difference before and aftre selection */
		eP[B] = p00[B] - d[B];
		eP[G] = p00[G] - d[G];
		eP[R] = p00[R] - d[R];

		/* save neighborhoods for temporal storage */
		imgWindow[B][1] = d[B];	imgWindow[B][2] = p01[B] + eP[B] * gfSevenDiv13;
		imgWindow[B][3] = 0.f; 	imgWindow[B][4] = p10[B] + eP[B] * gfFiveDiv13;
		imgWindow[B][5] = p11[B] + eP[B] * gfOneDiv13;

		imgWindow[G][1] = d[G];	imgWindow[G][2] = p01[G] + eP[G] * gfSevenDiv13;
		imgWindow[G][3] = 0.f; 	imgWindow[G][4] = p10[G] + eP[G] * gfFiveDiv13;
		imgWindow[G][5] = p11[G] + eP[G] * gfOneDiv13;

		imgWindow[R][1] = d[R];	imgWindow[R][2] = p01[R] + eP[R] * gfSevenDiv13;
		imgWindow[R][3] = 0.f; 	imgWindow[R][4] = p10[R] + eP[R] * gfFiveDiv13;
		imgWindow[R][5] = p11[R] + eP[R] * gfOneDiv13;

		Y = reciproc255 * (d[R] * rgb2yuv[0] + d[G] * rgb2yuv[1] + d[B] * rgb2yuv[2]);
		U = reciproc255 * (d[R] * rgb2yuv[3] + d[G] * rgb2yuv[3] + d[B] * rgb2yuv[5]);
		V = reciproc255 * (d[R] * rgb2yuv[6] + d[G] * rgb2yuv[7] + d[B] * rgb2yuv[8]);

		/* save destination pixel */
		Make_Color_pixel_yuv(localDst[idx], Y, U, V, localSrc[idx].A);

			for (x = 1; x < width_without_last; x++)
			{
				inPix01 = localSrc[idx + x + 1];	/* pixel in position 1 and line 0 */
				inPix10 = localSrc[next_idx + x + 1];	/* pixel in position 0 and line 1 */

				Y01 = inPix01.Y * 255.f; U01 = inPix01.U * 255.f; V01 = inPix01.V * 255.f;
				Y10 = inPix10.Y * 255.f; U10 = inPix10.U * 255.f; V10 = inPix10.V * 255.f;

				p01[R] = Y01 * yuv2rgb[0] + U01 * yuv2rgb[1] + V01 * yuv2rgb[2];
				p01[G] = Y01 * yuv2rgb[3] + U01 * yuv2rgb[4] + V01 * yuv2rgb[5];
				p01[B] = Y01 * yuv2rgb[6] + U01 * yuv2rgb[7] + V01 * yuv2rgb[8];

				p10[R] = Y10 * yuv2rgb[0] + U10 * yuv2rgb[1] + V10 * yuv2rgb[2];
				p10[G] = Y10 * yuv2rgb[3] + U10 * yuv2rgb[4] + V10 * yuv2rgb[5];
				p10[B] = Y10 * yuv2rgb[6] + U10 * yuv2rgb[7] + V10 * yuv2rgb[8];

				d[B] = (imgWindow[B][1] >= 128.f) ? 255.f : 0.f;
				d[G] = (imgWindow[G][1] >= 128.f) ? 255.f : 0.f;
				d[R] = (imgWindow[R][1] >= 128.f) ? 255.f : 0.f;

				eP[B] = imgWindow[B][1] - d[B];
				eP[G] = imgWindow[G][1] - d[G];
				eP[R] = imgWindow[R][1] - d[R];

				imgWindow[B][1] = p01[B] + eP[B] * gfSevenDiv16;
				imgWindow[B][3] = imgWindow[B][4] + eP[B] * gfThreeDiv16;
				imgWindow[B][4] = p11[B] + eP[B] * gfFiveDiv16;
				imgWindow[B][5] = p11[B] + eP[B] * gfOneDiv16;

				imgWindow[G][1] = p01[G] + eP[G] * gfSevenDiv16;
				imgWindow[G][3] = imgWindow[G][4] + eP[G] * gfThreeDiv16;
				imgWindow[G][4] = p11[G] + eP[G] * gfFiveDiv16;
				imgWindow[G][5] = p11[G] + eP[G] * gfOneDiv16;

				imgWindow[R][1] = p01[R] + eP[R] * gfSevenDiv16;
				imgWindow[R][3] = imgWindow[R][4] + eP[R] * gfThreeDiv16;
				imgWindow[R][4] = p11[R] + eP[R] * gfFiveDiv16;
				imgWindow[R][5] = p11[R] + eP[R] * gfOneDiv16;

				Y = reciproc255 * (d[R] * rgb2yuv[0] + d[G] * rgb2yuv[1] + d[B] * rgb2yuv[2]);
				U = reciproc255 * (d[R] * rgb2yuv[3] + d[G] * rgb2yuv[3] + d[B] * rgb2yuv[5]);
				V = reciproc255 * (d[R] * rgb2yuv[6] + d[G] * rgb2yuv[7] + d[B] * rgb2yuv[8]);

				Make_Color_pixel_yuv(localDst[idx + x], Y, U, V, localSrc[idx + x].A);
			} /* for (x = 1; x < width_without_last; x++) */

		} /* for (y = 0; y < height_without_last; y++) */

		  /* process last line */
	A_long const& last_line = height_without_last * line_pitch;
	inPix00 = localSrc[last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[last_line + 1];	/* pixel in position 0 and line 0 */

	Y01 = inPix01.Y * 255.f; U01 = inPix01.U * 255.f - 128.f; V01 = inPix01.V * 255.f;
	Y10 = inPix10.Y * 255.f; U10 = inPix10.U * 255.f - 128.f; V10 = inPix10.V * 255.f;

	p00[R] = Y00 * yuv2rgb[0] + U00 * yuv2rgb[1] + V00 * yuv2rgb[2];
	p00[G] = Y00 * yuv2rgb[3] + U00 * yuv2rgb[4] + V00 * yuv2rgb[5];
	p00[B] = Y00 * yuv2rgb[6] + U00 * yuv2rgb[7] + V00 * yuv2rgb[8];

	p01[R] = Y01 * yuv2rgb[0] + U01 * yuv2rgb[1] + V01 * yuv2rgb[2];
	p01[G] = Y01 * yuv2rgb[3] + U01 * yuv2rgb[4] + V01 * yuv2rgb[5];
	p01[B] = Y01 * yuv2rgb[6] + U01 * yuv2rgb[7] + V01 * yuv2rgb[8];

	d[R] = (p00[R] >= 128) ? 255.f : 0.f;
	d[G] = (p00[G] >= 128) ? 255.f : 0.f;
	d[B] = (p00[B] >= 128) ? 255.f : 0.f;

	eP[R] = p00[R] - d[R];
	eP[G] = p00[G] - d[G];
	eP[B] = p00[B] - d[B];

	imgWindow[R][1] = p01[R] * eP[R];
	imgWindow[G][1] = p01[G] * eP[G];
	imgWindow[B][1] = p01[B] * eP[B];

	Y = reciproc255 * (d[R] * rgb2yuv[0] + d[G] * rgb2yuv[1] + d[B] * rgb2yuv[2]);
	U = reciproc255 * (d[R] * rgb2yuv[3] + d[G] * rgb2yuv[3] + d[B] * rgb2yuv[5]);
	V = reciproc255 * (d[R] * rgb2yuv[6] + d[G] * rgb2yuv[7] + d[B] * rgb2yuv[8]);

	Make_Color_pixel_yuv(localDst[last_line], Y, U, V, localSrc[last_line].A);

	for (x = 1; x < width_without_last; x++)
	{
		d[R] = (imgWindow[R][1] >= 128) ? 255.f : 0.f;
		d[G] = (imgWindow[G][1] >= 128) ? 255.f : 0.f;
		d[B] = (imgWindow[B][1] >= 128) ? 255.f : 0.f;

		Y = d[R] * rgb2yuv[0] + d[G] * rgb2yuv[1] + d[B] * rgb2yuv[2];
		U = d[R] * rgb2yuv[3] + d[G] * rgb2yuv[3] + d[B] * rgb2yuv[5];
		V = d[R] * rgb2yuv[6] + d[G] * rgb2yuv[7] + d[B] * rgb2yuv[8];

		Make_Color_pixel_yuv(localDst[last_line + x], Y, U, V, localSrc[last_line + x].A);

		inPix01 = localSrc[last_line + x + 1];	/* pixel in next position	*/

		Y01 = inPix01.Y * 255.f; U01 = inPix01.U * 255.f; V01 = inPix01.V * 255.f;

		p01[R] = Y01 * yuv2rgb[0] + U01 * yuv2rgb[1] + V01 * yuv2rgb[2];
		p01[G] = Y01 * yuv2rgb[3] + U01 * yuv2rgb[4] + V01 * yuv2rgb[5];
		p01[B] = Y01 * yuv2rgb[6] + U01 * yuv2rgb[7] + V01 * yuv2rgb[8];

		/* difference before and after selection */
		eP[R] = p01[R] - d[R];
		eP[G] = p01[G] - d[G];
		eP[B] = p01[B] - d[B];

		imgWindow[R][1] = p01[R] + eP[R];
		imgWindow[G][1] = p01[G] + eP[G];
		imgWindow[B][1] = p01[B] + eP[B];
	}

	d[R] = (imgWindow[R][1] >= 128) ? 255.f : 0.f;
	d[G] = (imgWindow[G][1] >= 128) ? 255.f : 0.f;
	d[B] = (imgWindow[B][1] >= 128) ? 255.f : 0.f;

	Y = reciproc255 * (d[R] * rgb2yuv[0] + d[G] * rgb2yuv[1] + d[B] * rgb2yuv[2]);
	U = reciproc255 * (d[R] * rgb2yuv[3] + d[G] * rgb2yuv[3] + d[B] * rgb2yuv[5]);
	V = reciproc255 * (d[R] * rgb2yuv[6] + d[G] * rgb2yuv[7] + d[B] * rgb2yuv[8]);

	Make_Color_pixel_yuv(localDst[last_line + width_without_last], Y, U, V, localSrc[last_line + width_without_last].A);

	return err;
}


static PF_Err PR_ImageStyle_ColorNewsPaper_BGRA_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN float imgWindow[3][6]{};

	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	PF_Err err = PF_Err_NONE;
	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	auto const& height_without_last = height - 1;
	auto const& width_without_last = width - 1;

	PF_Pixel_BGRA_16u inPix00, /* curent pixel									*/
					  inPix01, /* pixel in same line and in raw position plus 1	*/
					  inPix10, /* pixel on next line in same raw postion		*/
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
		p00[R] = static_cast<int>(inPix00.R) >> 7; p00[G] = static_cast<int>(inPix00.G) >> 7; p00[B] = static_cast<int>(inPix00.B) >> 7;
		p01[R] = static_cast<int>(inPix01.R) >> 7; p01[G] = static_cast<int>(inPix01.G) >> 7; p00[B] = static_cast<int>(inPix01.B) >> 7;
		p10[R] = static_cast<int>(inPix10.R) >> 7; p10[G] = static_cast<int>(inPix10.G) >> 7; p10[B] = static_cast<int>(inPix10.B) >> 7;
		p11[R] = static_cast<int>(inPix11.R) >> 7; p11[G] = static_cast<int>(inPix11.G) >> 7; p11[B] = static_cast<int>(inPix11.B) >> 7;

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
		Make_Color_pixel(localDst[idx], static_cast<A_u_short>(d[R] * 128.f), static_cast<A_u_short>(d[G] * 128.f), static_cast<A_u_short>(d[B] * 128.f), localSrc[idx].A);

		for (x = 1; x < width_without_last; x++)
		{
			inPix01 = localSrc[idx + x + 1];	/* pixel in position 1 and line 0 */
			inPix10 = localSrc[next_idx + x + 1];	/* pixel in position 0 and line 1 */

			p01[R] = static_cast<int>(inPix01.R) >> 7; p01[G] = static_cast<int>(inPix01.G) >> 7; p01[B] = static_cast<int>(inPix01.B) >> 7;
			p11[R] = static_cast<int>(inPix11.R) >> 7; p11[G] = static_cast<int>(inPix11.G) >> 7; p11[B] = static_cast<int>(inPix11.B) >> 7;

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

			Make_Color_pixel(localDst[idx + x], static_cast<A_u_short>(d[R] * 128.f), static_cast<A_u_short>(d[G] * 128.f), static_cast<A_u_short>(d[B] * 128.f), localSrc[idx + x].A);
		} /* for (x = 1; x < width_without_last; x++) */

	} /* for (y = 0; y < height_without_last; y++) */

	/* process last line */
	A_long const& last_line = height_without_last * line_pitch;
	inPix00 = localSrc[last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[last_line + 1];	/* pixel in position 0 and line 0 */

	p00[R] = static_cast<int>(inPix00.R) >> 7; p00[G] = static_cast<int>(inPix00.G) >> 7; p00[B] = static_cast<int>(inPix00.B) >> 7;
	p01[R] = static_cast<int>(inPix00.R) >> 7; p00[G] = static_cast<int>(inPix00.G) >> 7; p00[B] = static_cast<int>(inPix00.B) >> 7;

	d[R] = (p00[R] >= 128) ? 255.f : 0.f;
	d[G] = (p00[G] >= 128) ? 255.f : 0.f;
	d[B] = (p00[B] >= 128) ? 255.f : 0.f;

	eP[R] = static_cast<float>(p00[R]) - d[R];
	eP[G] = static_cast<float>(p00[G]) - d[G];
	eP[B] = static_cast<float>(p00[B]) - d[B];

	imgWindow[R][1] = static_cast<float>(p01[R]) * eP[R];
	imgWindow[G][1] = static_cast<float>(p01[G]) * eP[G];
	imgWindow[B][1] = static_cast<float>(p01[B]) * eP[B];

	Make_Color_pixel(localDst[last_line], static_cast<A_u_short>(d[R] * 128.f), static_cast<A_u_short>(d[G] * 128.f), static_cast<A_u_short>(d[B] * 128.f), localSrc[last_line].A);

	for (x = 1; x < width_without_last; x++)
	{
		d[R] = (imgWindow[R][1] >= 128) ? 255.f : 0.f;
		d[G] = (imgWindow[G][1] >= 128) ? 255.f : 0.f;
		d[B] = (imgWindow[B][1] >= 128) ? 255.f : 0.f;

		Make_Color_pixel(localDst[last_line + x], static_cast<A_u_short>(d[R] * 128.f), static_cast<A_u_short>(d[G] * 128.f), static_cast<A_u_short>(d[B] * 128.f), localSrc[last_line + x].A);

		inPix01 = localSrc[last_line + x + 1];	/* pixel in next position	*/
		p01[R] = static_cast<int>(inPix01.R) >> 7; p01[G] = static_cast<int>(inPix01.G) >> 7; p01[B] = static_cast<int>(inPix01.B) >> 7;

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

	Make_Color_pixel(localDst[last_line + width_without_last], static_cast<A_u_short>(d[R] * 128.f), static_cast<A_u_short>(d[G] * 128.f), static_cast<A_u_short>(d[B] * 128.f), localSrc[last_line + width_without_last].A);

	return err;
}


static PF_Err PR_ImageStyle_ColorNewsPaper_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN float imgWindow[3][6]{};

	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	PF_Err err = PF_Err_NONE;
	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	auto const& height_without_last = height - 1;
	auto const& width_without_last = width - 1;

	PF_Pixel_BGRA_32f inPix00, /* curent pixel									*/
					  inPix01, /* pixel in same line and in raw position plus 1	*/
					  inPix10, /* pixel on next line in same raw postion		*/
					  inPix11; /* pixel on next line in raw position plus 1		*/

	A_long x, y;
	int p00[3]{}; int p01[3]{}; int p10[3]{}; int p11[3]{};
	float d[3]{}; float eP[3]{};
	
	constexpr float reciproc255 = 1.f / 255.f;

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

		p00[R] = static_cast<int>(inPix00.R * 255.f); p00[G] = static_cast<int>(inPix00.G * 255.f); p00[B] = static_cast<int>(inPix00.B * 255.f);
		p01[R] = static_cast<int>(inPix01.R * 255.f); p01[G] = static_cast<int>(inPix01.G * 255.f); p00[B] = static_cast<int>(inPix01.B * 255.f);
		p10[R] = static_cast<int>(inPix10.R * 255.f); p10[G] = static_cast<int>(inPix10.G * 255.f); p10[B] = static_cast<int>(inPix10.B * 255.f);
		p11[R] = static_cast<int>(inPix11.R * 255.f); p11[G] = static_cast<int>(inPix11.G * 255.f); p11[B] = static_cast<int>(inPix11.B * 255.f);

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
		Make_Color_pixel(localDst[idx], d[R] * reciproc255, d[G] * reciproc255, d[B] * reciproc255, localSrc[idx].A);

		for (x = 1; x < width_without_last; x++)
		{
			inPix01 = localSrc[idx + x + 1];	/* pixel in position 1 and line 0 */
			inPix10 = localSrc[next_idx + x + 1];	/* pixel in position 0 and line 1 */

			p01[R] = static_cast<int>(inPix01.R * 255.f); p01[G] = static_cast<int>(inPix01.G * 255.f); p01[B] = static_cast<int>(inPix01.B * 255.f);
			p11[R] = static_cast<int>(inPix11.R * 255.f); p11[G] = static_cast<int>(inPix11.G * 255.f); p11[B] = static_cast<int>(inPix11.B * 255.f);

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

			Make_Color_pixel(localDst[idx + x], d[R] * reciproc255,  d[G] * reciproc255,  d[B] * reciproc255, localSrc[idx + x].A);
		} /* for (x = 1; x < width_without_last; x++) */

	} /* for (y = 0; y < height_without_last; y++) */

		  /* process last line */
	A_long const& last_line = height_without_last * line_pitch;
	inPix00 = localSrc[last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[last_line + 1];	/* pixel in position 0 and line 0 */

	p00[R] = static_cast<int>(inPix00.R * 255.f); p00[G] = static_cast<int>(inPix00.G * 255.f); p00[B] = static_cast<int>(inPix00.B * 255.f);
	p01[R] = static_cast<int>(inPix00.R * 255.f); p00[G] = static_cast<int>(inPix00.G * 255.f); p00[B] = static_cast<int>(inPix00.B * 255.f);

	d[R] = (p00[R] >= 128) ? 255.f : 0.f;
	d[G] = (p00[G] >= 128) ? 255.f : 0.f;
	d[B] = (p00[B] >= 128) ? 255.f : 0.f;

	eP[R] = static_cast<float>(p00[R]) - d[R];
	eP[G] = static_cast<float>(p00[G]) - d[G];
	eP[B] = static_cast<float>(p00[B]) - d[B];

	imgWindow[R][1] = static_cast<float>(p01[R]) * eP[R];
	imgWindow[G][1] = static_cast<float>(p01[G]) * eP[G];
	imgWindow[B][1] = static_cast<float>(p01[B]) * eP[B];

	Make_Color_pixel(localDst[last_line], d[R] * reciproc255, d[G] * reciproc255, d[B] * reciproc255, localSrc[last_line].A);

	for (x = 1; x < width_without_last; x++)
	{
		d[R] = (imgWindow[R][1] >= 128) ? 255.f : 0.f;
		d[G] = (imgWindow[G][1] >= 128) ? 255.f : 0.f;
		d[B] = (imgWindow[B][1] >= 128) ? 255.f : 0.f;

		Make_Color_pixel(localDst[last_line + x], d[R] * reciproc255, d[G] * reciproc255, d[B] * reciproc255, localSrc[last_line + x].A);

		inPix01 = localSrc[last_line + x + 1];	/* pixel in next position	*/
		p01[R] = static_cast<int>(inPix01.R * 255.f); p01[G] = static_cast<int>(inPix01.G * 255.f); p01[B] = static_cast<int>(inPix01.B * 255.f);

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

	Make_Color_pixel(localDst[last_line + width_without_last], d[R] * reciproc255, d[G] * reciproc255, d[B] * reciproc255, localSrc[last_line + width_without_last].A);

	return err;

}



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
			{
				auto const& isBT709 = (PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat);
				err = PR_ImageStyle_ColorNewsPaper_VUYA_8u (in_data, out_data, params, output, isBT709);
			}
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
			{
				auto const& isBT709 = (PrPixelFormat_VUYA_4444_8u_709 == destinationPixelFormat);
				err = PR_ImageStyle_ColorNewsPaper_VUYA_32f (in_data, out_data, params, output, isBT709);
			}
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageStyle_ColorNewsPaper_BGRA_16u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageStyle_ColorNewsPaper_BGRA_32f (in_data, out_data, params, output);
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


PF_Err AE_ImageStyle_ColorNewsPaper_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN float imgWindow[3][6]{};

	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_ARGB_8u*     __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input->data);
	PF_Pixel_ARGB_8u*     __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);

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
	int p00[3]{}; int p01[3]{}; int p10[3]{}; int p11[3]{};
	float d[3]{}; float eP[3]{};

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
		Make_Color_pixel(localDst[dst_idx], static_cast<A_u_char>(d[R]), static_cast<A_u_char>(d[G]), static_cast<A_u_char>(d[B]), localSrc[src_idx].A);

		for (x = 1; x < width_without_last; x++)
		{
			inPix01 = localSrc[src_idx + x + 1];		/* pixel in position 1 and line 0 */
			inPix10 = localSrc[src_next_idx + x + 1];	/* pixel in position 0 and line 1 */

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

			Make_Color_pixel(localDst[dst_idx + x], static_cast<A_u_char>(d[R]), static_cast<A_u_char>(d[G]), static_cast<A_u_char>(d[B]), localSrc[src_idx + x].A);
		} /* for (x = 1; x < width_without_last; x++) */

	} /* for (y = 0; y < height_without_last; y++) */

	/* process last line */
	A_long const& src_last_line = height_without_last * src_line_pitch;
	A_long const& dst_last_line = height_without_last * dst_line_pitch;
	inPix00 = localSrc[src_last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[src_last_line + 1];	/* pixel in position 0 and line 0 */

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

	Make_Color_pixel(localDst[dst_last_line], static_cast<A_u_char>(d[R]), static_cast<A_u_char>(d[G]), static_cast<A_u_char>(d[B]), localSrc[src_last_line].A);

	for (x = 1; x < width_without_last; x++)
	{
		d[R] = (imgWindow[R][1] >= 128) ? 255.f : 0.f;
		d[G] = (imgWindow[G][1] >= 128) ? 255.f : 0.f;
		d[B] = (imgWindow[B][1] >= 128) ? 255.f : 0.f;

		Make_Color_pixel(localDst[dst_last_line + x], static_cast<A_u_char>(d[R]), static_cast<A_u_char>(d[G]), static_cast<A_u_char>(d[B]), localSrc[src_last_line + x].A);

		inPix01 = localSrc[src_last_line + x + 1];	/* pixel in next position	*/
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

	Make_Color_pixel(localDst[dst_last_line + width_without_last], static_cast<A_u_char>(d[R]), static_cast<A_u_char>(d[G]), static_cast<A_u_char>(d[B]), localSrc[src_last_line + width_without_last].A);

	return err;
}


PF_Err AE_ImageStyle_ColorNewsPaper_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	CACHE_ALIGN float imgWindow[3][6]{};

	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_ARGB_16u*    __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(input->data);
	PF_Pixel_ARGB_16u*    __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output->data);

	const A_long& height = output->height;
	const A_long& width = output->width;
	const A_long& src_line_pitch = input->rowbytes / sizeof(PF_Pixel8);
	const A_long& dst_line_pitch = output->rowbytes / sizeof(PF_Pixel8);

	PF_Err err = PF_Err_NONE;

	auto const& height_without_last = height - 1;
	auto const& width_without_last = width - 1;

	PF_Pixel_ARGB_16u inPix00, /* curent pixel									*/
					  inPix01, /* pixel in same line and in raw position plus 1	*/
					  inPix10, /* pixel on next line in same raw postion		*/
					  inPix11; /* pixel on next line in raw position plus 1		*/

	A_long x, y;
	int p00[3]{}; int p01[3]{}; int p10[3]{}; int p11[3]{};
	float d[3]{}; float eP[3]{};

	__VECTOR_ALIGNED__
	for (y = 0; y < height_without_last; y++)
	{
		A_long const& src_idx = y * src_line_pitch;				/* current frame line	*/
		A_long const& src_next_idx = (y + 1) * src_line_pitch;	/* next frame line		*/
		A_long const& dst_idx = y * dst_line_pitch;				/* current frame line	*/

		/* process first pixel in first line */
		inPix00 = localSrc[src_idx];			/* pixel in position 0 and line 0 */
		inPix01 = localSrc[src_idx + 1];		/* pixel in position 1 and line 0 */
		inPix10 = localSrc[src_next_idx];		/* pixel in position 0 and line 1 */
		inPix11 = localSrc[src_next_idx + 1];	/* pixel in position 0 and line 1 */

		/* compute Luma for current pixel and for neighborhoods */
		p00[R] = static_cast<int>(inPix00.R) >> 7; p00[G] = static_cast<int>(inPix00.G) >> 7; p00[B] = static_cast<int>(inPix00.B) >> 7;
		p01[R] = static_cast<int>(inPix01.R) >> 7; p01[G] = static_cast<int>(inPix01.G) >> 7; p00[B] = static_cast<int>(inPix01.B) >> 7;
		p10[R] = static_cast<int>(inPix10.R) >> 7; p10[G] = static_cast<int>(inPix10.G) >> 7; p10[B] = static_cast<int>(inPix10.B) >> 7;
		p11[R] = static_cast<int>(inPix11.R) >> 7; p11[G] = static_cast<int>(inPix11.G) >> 7; p11[B] = static_cast<int>(inPix11.B) >> 7;

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
		Make_Color_pixel(localDst[dst_idx], static_cast<A_u_short>(d[R] * 128.f), static_cast<A_u_short>(d[G] * 128.f), static_cast<A_u_short>(d[B] * 128.f), localSrc[src_idx].A);

		for (x = 1; x < width_without_last; x++)
		{
			inPix01 = localSrc[src_idx + x + 1];	/* pixel in position 1 and line 0 */
			inPix10 = localSrc[src_next_idx + x + 1];	/* pixel in position 0 and line 1 */

			p01[R] = static_cast<int>(inPix01.R) >> 7; p01[G] = static_cast<int>(inPix01.G) >> 7; p01[B] = static_cast<int>(inPix01.B) >> 7;
			p11[R] = static_cast<int>(inPix11.R) >> 7; p11[G] = static_cast<int>(inPix11.G) >> 7; p11[B] = static_cast<int>(inPix11.B) >> 7;

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

			Make_Color_pixel(localDst[dst_idx + x], static_cast<A_u_short>(d[R] * 128.f), static_cast<A_u_short>(d[G] * 128.f), static_cast<A_u_short>(d[B] * 128.f), localSrc[src_idx + x].A);
		} /* for (x = 1; x < width_without_last; x++) */

	} /* for (y = 0; y < height_without_last; y++) */

	/* process last line */
	A_long const& src_last_line = height_without_last * src_line_pitch;
	A_long const& dst_last_line = height_without_last * dst_line_pitch;
	inPix00 = localSrc[src_last_line];		/* pixel in position 0 and line 0 */
	inPix01 = localSrc[src_last_line + 1];	/* pixel in position 0 and line 0 */

	p00[R] = static_cast<int>(inPix00.R) >> 7; p00[G] = static_cast<int>(inPix00.G) >> 7; p00[B] = static_cast<int>(inPix00.B) >> 7;
	p01[R] = static_cast<int>(inPix00.R) >> 7; p00[G] = static_cast<int>(inPix00.G) >> 7; p00[B] = static_cast<int>(inPix00.B) >> 7;

	d[R] = (p00[R] >= 128) ? 255.f : 0.f;
	d[G] = (p00[G] >= 128) ? 255.f : 0.f;
	d[B] = (p00[B] >= 128) ? 255.f : 0.f;

	eP[R] = static_cast<float>(p00[R]) - d[R];
	eP[G] = static_cast<float>(p00[G]) - d[G];
	eP[B] = static_cast<float>(p00[B]) - d[B];

	imgWindow[R][1] = static_cast<float>(p01[R]) * eP[R];
	imgWindow[G][1] = static_cast<float>(p01[G]) * eP[G];
	imgWindow[B][1] = static_cast<float>(p01[B]) * eP[B];

	Make_Color_pixel(localDst[dst_last_line], static_cast<A_u_short>(d[R] * 128.f), static_cast<A_u_short>(d[G] * 128.f), static_cast<A_u_short>(d[B] * 128.f), localSrc[src_last_line].A);

	for (x = 1; x < width_without_last; x++)
	{
		d[R] = (imgWindow[R][1] >= 128) ? 255.f : 0.f;
		d[G] = (imgWindow[G][1] >= 128) ? 255.f : 0.f;
		d[B] = (imgWindow[B][1] >= 128) ? 255.f : 0.f;

		Make_Color_pixel(localDst[dst_last_line + x], static_cast<A_u_short>(d[R] * 128.f), static_cast<A_u_short>(d[G] * 128.f), static_cast<A_u_short>(d[B] * 128.f), localSrc[src_last_line + x].A);

		inPix01 = localSrc[src_last_line + x + 1];	/* pixel in next position	*/
		p01[R] = static_cast<int>(inPix01.R) >> 7; p01[G] = static_cast<int>(inPix01.G) >> 7; p01[B] = static_cast<int>(inPix01.B) >> 7;

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

	Make_Color_pixel(localDst[dst_last_line + width_without_last], static_cast<A_u_short>(d[R] * 128.f), static_cast<A_u_short>(d[G] * 128.f), static_cast<A_u_short>(d[B] * 128.f), localSrc[src_last_line + width_without_last].A);

	return err;
}


PF_Err AE_ImageStyle_ColorNewsPaper_ARGB_32f
(
    PF_InData*   __restrict in_data,
    PF_OutData*  __restrict out_data,
    PF_ParamDef* __restrict params[],
    PF_LayerDef* __restrict output
) noexcept
{
    return PF_Err_NONE; // non implementyed yet
}