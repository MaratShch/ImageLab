#include "ColorizeMe.hpp"
#include "CubeLUT.h"
#include "CommonDebugUtils.hpp"
#include "LutHelper.hpp"

#define LUT_NEAR(x) ((int)((x) + 0.5f))
#define LUT_PREV(x) ((int)(x))
#define LUT_NEXT(x) (MIN(((int)(x) + 1.0f), lut_size))

bool ProcessPrImage_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<PF_LayerDef* __restrict>(&params[COLOR_INPUT]->u.ld);
	PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	const SequenceData* seqData = (reinterpret_cast<SequenceData*>(GET_OBJ_FROM_HNDL(in_data->sequence_data)));

	const A_long height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	const int32_t pedestalR = params[COLOR_RED_PEDESTAL_SLIDER  ]->u.sd.value;
	const int32_t pedestalG = params[COLOR_GREEN_PEDESTAL_SLIDER]->u.sd.value;
	const int32_t pedestalB = params[COLOR_BLUE_PEDESTAL_SLIDER ]->u.sd.value;
	const int32_t negative  = params[COLOR_NEGATE_CHECKBOX]->u.bd.value;

	const LutIdx lut_idx = (nullptr != seqData && 0xDEADBEEF == seqData->magic) ? seqData->lut_idx : invalidLut;
	LutObjHndl pLut = getLut(lut_idx);


	A_long i = 0, j = 0;
	A_long idx = 0;
	int32_t r, g, b, a;
	int32_t newR, newG, newB;
	int32_t divR, divG, divB;
	int32_t lutSize = 0;

	if (nullptr != pLut && 0 != (lutSize = pLut->GetLutSize()))
	{
		const float divider = 255.0f / static_cast<float>(lutSize - 1);

		for (j = 0; j < height; j++)
		{
			__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				idx = j * line_pitch + i;

				r = localSrc[idx].R;
				g = localSrc[idx].G;
				b = localSrc[idx].B;
				a = localSrc[idx].A;

				/* apply LUT here... */
				divR = static_cast<float>(r) / divider;
				divG = static_cast<float>(g) / divider;
				divB = static_cast<float>(b) / divider;

				CubeLUT::tableRow lutVal = pLut->Lut3D[divR][divG][divB];

				newR = static_cast<int32_t>(255.0f * lutVal[0]) + pedestalR;
				newG = static_cast<int32_t>(255.0f * lutVal[1]) + pedestalG;
				newB = static_cast<int32_t>(255.0f * lutVal[2]) + pedestalB;

				if (negative)
				{
					localDst[idx].R = 255 - static_cast<A_u_char>(CLAMP_VALUE(newR, 0, 255));
					localDst[idx].G = 255 - static_cast<A_u_char>(CLAMP_VALUE(newG, 0, 255));
					localDst[idx].B = 255 - static_cast<A_u_char>(CLAMP_VALUE(newB, 0, 255));
					localDst[idx].A = a;
				}
				else
				{
					localDst[idx].R = static_cast<A_u_char>(CLAMP_VALUE(newR, 0, 255));
					localDst[idx].G = static_cast<A_u_char>(CLAMP_VALUE(newG, 0, 255));
					localDst[idx].B = static_cast<A_u_char>(CLAMP_VALUE(newB, 0, 255));
					localDst[idx].A = a;
				}
			}
		}
	}
	else
	{
		for (j = 0; j < height; j++)
		{
			__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				idx = j * line_pitch + i;
				/* no LUT */
				r = localSrc[idx].R + pedestalR;
				g = localSrc[idx].G + pedestalG;
				b = localSrc[idx].B + pedestalB;
				a = localSrc[idx].A;

				if (negative)
				{
					localDst[idx].R = 255 - static_cast<A_u_char>(CLAMP_VALUE(r, 0, 255));
					localDst[idx].G = 255 - static_cast<A_u_char>(CLAMP_VALUE(g, 0, 255));
					localDst[idx].B = 255 - static_cast<A_u_char>(CLAMP_VALUE(b, 0, 255));
					localDst[idx].A = a;
				}
				else
				{
					localDst[idx].R = static_cast<A_u_char>(CLAMP_VALUE(r, 0, 255));
					localDst[idx].G = static_cast<A_u_char>(CLAMP_VALUE(g, 0, 255));
					localDst[idx].B = static_cast<A_u_char>(CLAMP_VALUE(b, 0, 255));
					localDst[idx].A = a;
				}
			}
		}
	}

	return true;
}


bool ProcessPrImage_BGRA_4444_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return true;
}

bool ProcessPrImage_BGRA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return true;
}


bool ProcessPrImage_VUYA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const bool isBT709
) noexcept
{
	return true;
}


bool ProcessPrImage_VUYA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const bool isBT709
) noexcept
{
	return true;
}


bool ProcessPrImage_RGB_444_10u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return true;
}



PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const PrPixelFormat&    pixelFormat
) noexcept
{
	bool bValue = true;

	/* acquire controls parameters */

	switch (pixelFormat)
	{
		case PrPixelFormat_BGRA_4444_8u:
			bValue = ProcessPrImage_BGRA_4444_8u(in_data, out_data, params, output);
		break;

		case PrPixelFormat_BGRA_4444_16u:
			bValue = ProcessPrImage_BGRA_4444_16u(in_data, out_data, params, output);
		break;

		case PrPixelFormat_BGRA_4444_32f:
			bValue = ProcessPrImage_BGRA_4444_32f(in_data, out_data, params, output);
		break;

		case PrPixelFormat_VUYA_4444_8u:
		case PrPixelFormat_VUYA_4444_8u_709:
			bValue = ProcessPrImage_VUYA_4444_8u(in_data, out_data, params, output, PrPixelFormat_VUYA_4444_8u_709 == pixelFormat);
		break;

		case PrPixelFormat_VUYA_4444_32f:
		case PrPixelFormat_VUYA_4444_32f_709:
			bValue = ProcessPrImage_VUYA_4444_32f(in_data, out_data, params, output, PrPixelFormat_VUYA_4444_8u_709 == pixelFormat);
		break;

		case PrPixelFormat_RGB_444_10u:
			bValue = ProcessPrImage_RGB_444_10u(in_data, out_data, params, output);
		break;

		default:
			bValue = false;
		break;
	}

	return (true == bValue ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED);
}