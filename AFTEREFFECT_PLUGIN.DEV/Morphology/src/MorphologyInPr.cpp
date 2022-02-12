#include "Morphology.hpp"
#include "MorphologyEnums.hpp"
#include "PrSDKAESupport.h"
#include "SequenceData.hpp"
#include "MorphologyProc.hpp"


PF_Err MorphologyFilter_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	std::uint64_t seIdx{ INVALID_INTERFACE };

	/* get Structured Element Object */
	const std::uint64_t* seData{ reinterpret_cast<uint64_t*>(GET_OBJ_FROM_HNDL(out_data->sequence_data)) };
	if (nullptr == seData)
		return PF_Err_BAD_CALLBACK_PARAM;

	if (INVALID_INTERFACE == (seIdx = *seData))
		return PF_Err_BAD_CALLBACK_PARAM;

	size_t sizeSe = 0;
	SE_Interface* pSeElement = DataStore::getObject(seIdx);
	const SE_Type* seElementVal = (nullptr != pSeElement ? pSeElement->GetStructuredElement(sizeSe) : nullptr);
	if (nullptr == seElementVal)
		return PF_Err_NONE;

	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MORPHOLOGY_FILTER_INPUT]->u.ld);
	PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	auto const height     = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width      = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	A_long i, j, k, l;

	const A_long seElementsNumber = static_cast<A_long>(sizeSe * sizeSe);
	const A_long halfSeLine = static_cast<A_long>(sizeSe) >> 1;
	const A_long shortHeight = height - halfSeLine;
	const A_long shortWidth = width - halfSeLine;

	for (j = halfSeLine; j < shortHeight; j++)
	{
		A_long jMin = j - halfSeLine;
		A_long jMax = j + halfSeLine;

		__VECTOR_ALIGNED__
		for (i = halfSeLine; i < shortWidth; i++)
		{
			A_long iMin = i - halfSeLine;
			A_long iMax = i + halfSeLine;

			A_u_char rMin{ UCHAR_MAX };
			A_u_char gMin{ UCHAR_MAX };
			A_u_char bMin{ UCHAR_MAX };
			A_long dstIdx = j * line_pitch + i;

			__VECTOR_ALIGNED__
			for (l = jMin; l <= jMax; l++) /* kernel rows */
			{
				A_long lineIdx = MIN(shortHeight, MAX(0, l));
				A_long jIdx = lineIdx * line_pitch;

				for (k = iMin; k <= iMax; k++) /* kernel line */
				{
					A_long iIdx = jIdx + MIN(shortWidth, MAX(0, k));
					const PF_Pixel_BGRA_8u& pix = localSrc[iIdx];
					rMin = MIN(rMin, pix.R);
					gMin = MIN(gMin, pix.G);
					bMin = MIN(bMin, pix.B);
				}
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
