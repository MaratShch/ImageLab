#include "ColorTemperature.hpp"
#include "ColorTemperatureEnums.hpp"
#include "ColorTemperatureSeqData.hpp"
#include "PrSDKAESupport.h"


PF_Err ProcessImgInPR_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output,
	const rgbCoefficients&  cctStr
) noexcept
{
	PF_Err err = PF_Err_NONE;
	const float* __restrict pRgbCoeff = getColorCoefficients(cctStr.cct);
	if (nullptr != pRgbCoeff)
	{
		const float fR = pRgbCoeff[0];
		const float fG = pRgbCoeff[1];
		const float fB = pRgbCoeff[2];

		const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef*      __restrict>(&params[COLOR_TEMPERATURE_FILTER_INPUT]->u.ld);
		const PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
		      PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);

		const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
		const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
		const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

		for (A_long j = 0; j < sizeY; j++)
		{
			const A_long line_idx = j * line_pitch;
			for (A_long i = 0; i < sizeX; i++)
			{
				PF_Pixel_BGRA_8u dstPixel;
				const PF_Pixel_BGRA_8u& srcPixel = localSrc[line_idx + i];
				
				dstPixel.R = srcPixel.R * fR;
				dstPixel.G = srcPixel.G * fG;
				dstPixel.B = srcPixel.B * fB;
				dstPixel.A = srcPixel.A;

				localDst[line_idx + i] = dstPixel;
			}
		}

	} /* if (nullptr != pRgbCoeff) */
	else
		err = PF_Err_INVALID_INDEX;

	return err;
}


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	PF_Err err{ PF_Err_NONE };
	PF_Err errFormat{ PF_Err_INVALID_INDEX };
	PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };
	bool rebuildCoeffcients = false;

	/* Lets read CCT controls value */
	auto const value_coarse_cct  = slider2ColorTemperature(params[COLOR_TEMPERATURE_COARSE_VALUE_SLIDER]->u.fs_d.value);
	auto const value_offset_cct  = params[COLOR_TEMPERATURE_FINE_VALUE_SLIDER]->u.fs_d.value;
	auto const value_tint        = params[COLOR_TEMPERATURE_TINT_SLIDER ]->u.fs_d.value;
	const CCT_TYPE final_cct     = static_cast<CCT_TYPE>(static_cast<PF_FpLong>(value_coarse_cct) + value_offset_cct);

	/* check sequence data handler ... */
	if (nullptr != in_data->sequence_data)
	{
		/* ... and data attached to the handler */
		flatSequenceData* seqData = reinterpret_cast<flatSequenceData*>(GET_OBJ_FROM_HNDL(in_data->sequence_data));
		if (nullptr != seqData)
		{
			rgbCoefficients colorCoeff = seqData->colorCoeff;
			/* get Sequence data and compare CCT and TINT values from SequenceData with current Sliders positions */
			if (true == (rebuildCoeffcients = checkSequenceData(seqData->colorCoeff, final_cct, static_cast<CCT_TYPE>(value_tint))))
			{
				/* CCT or/and TINT value changed. We need rebuild color coefficients at now */
				colorCoeff.cct = seqData->colorCoeff.cct;
				colorCoeff.tint = seqData->colorCoeff.tint;
				rebuildColorCoefficients (colorCoeff);
			}

			/* This plugin called frop PR - check video format */
			if (PF_Err_NONE == (errFormat = AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data)->
				GetPixelFormat(output, &destinationPixelFormat)))
			{
				switch (destinationPixelFormat)
				{
					case PrPixelFormat_BGRA_4444_8u:
						err = ProcessImgInPR_BGRA_4444_8u(in_data, out_data, params, output, colorCoeff);
					break;

					case PrPixelFormat_BGRA_4444_16u:
					break;

					case PrPixelFormat_BGRA_4444_32f:
					break;

					case PrPixelFormat_VUYA_4444_8u_709:
					case PrPixelFormat_VUYA_4444_8u:
					break;

					case PrPixelFormat_VUYA_4444_32f_709:
					case PrPixelFormat_VUYA_4444_32f:
					break;

					case PrPixelFormat_RGB_444_10u:
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
		} /* if (nullptr != seqData) */
		else
			err = PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
	else
		err = PF_Err_INTERNAL_STRUCT_DAMAGED;

	return err;
}
