#include "Morphology.hpp"
#include "MorphologyEnums.hpp"
#include "PrSDKAESupport.h"



PF_Err MorphologyFilter_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MORPHOLOGY_FILTER_INPUT]->u.ld);
	PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	auto const height     = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width      = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	/* get Structured Element Object */
	const strSeData* seData = reinterpret_cast<strSeData*>(GET_OBJ_FROM_HNDL(out_data->sequence_data));
	const SE_Interface* Interace = seData->IstructElem;
	const uint32_t& bValid = seData->bValid;
    
	size_t sizeSe = 0;
	A_long i, j, k, l;

	const SE_Type* seElement = Interace->GetStructuredElement(sizeSe);
	const A_long seElementsNumber = static_cast<A_long>(sizeSe * sizeSe);
	const A_long halfSeLine = static_cast<A_long>(sizeSe) / 2;
	const A_long shortHeight = height - halfSeLine;
	const A_long shortWidth = width - halfSeLine;

	for (j = halfSeLine; j < shortHeight; j++)
	{
		for (i = halfSeLine; i < shortWidth; i++)
		{
			int32_t rMin = INT_MAX;
			int32_t gMin = INT_MAX;
			int32_t bMin = INT_MAX;
			int32_t dstIdx = j * line_pitch + i;

			for (k = j - halfSeLine; k < sizeSe; k++)
				for (l = i - halfSeLine; l < sizeSe; l++)
				{
					const PF_Pixel_BGRA_8u& pix = localSrc[k * line_pitch + l];
					rMin = MIN(rMin, pix.R);
					gMin = MIN(gMin, pix.G);
					bMin = MIN(bMin, pix.B);
				}

			localDst[dstIdx].B = bMin;
			localDst[dstIdx].G = gMin;
			localDst[dstIdx].R = rMin;
			localDst[dstIdx].A = localSrc[dstIdx].A;
		}
	}

	return PF_Err_NONE;
}


PF_Err MorphologyFilter_BGRA_4444_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MORPHOLOGY_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);
	return PF_Err_NONE;
}


PF_Err MorphologyFilter_BGRA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MORPHOLOGY_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);
	return PF_Err_NONE;
}


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PrPixelFormat destinationPixelFormat = PrPixelFormat_Invalid;

	/* This plugin called from PR - check video fomat */
	if (PF_Err_NONE == AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->GetPixelFormat(output, &destinationPixelFormat))
	{
		switch (destinationPixelFormat)
		{
			case PrPixelFormat_BGRA_4444_8u:
				err = MorphologyFilter_BGRA_4444_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = MorphologyFilter_BGRA_4444_16u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = MorphologyFilter_BGRA_4444_32f (in_data, out_data, params, output);
			break;

			default:
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
