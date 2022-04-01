#include "Morphology.hpp"
#include "MorphologyEnums.hpp"
#include "PrSDKAESupport.h"
#include "SequenceData.hpp"
#include "MorphologyProcCpu.hpp"


PF_Err MorphologyFilter_BGRA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	size_t sizeSe = 0;
	const SE_Interface* seInetrface = getStructuredElemInterface(out_data);
	const SE_Type* __restrict seElementVal = (nullptr != seInetrface ? seInetrface->GetStructuredElement(sizeSe) : nullptr);

	if (nullptr == seElementVal || 0 == sizeSe)
		return PF_Err_NONE;

	const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MORPHOLOGY_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u*  __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*        __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u*  __restrict>(output->data);

	auto const height     = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width      = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	const PF_ParamDef* pMorphologyOperation = params[MORPHOLOGY_OPERATION_TYPE];
	const SeOperation  cType = static_cast<const SeOperation>(pMorphologyOperation->u.pd.value - 1);

	switch (cType)
	{
		case SE_OP_EROSION:
			Morphology_Erode (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(UCHAR_MAX));
		break;

		case SE_OP_DILATION:
			Morphology_Dilate (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(0u));
		break;

		case SE_OP_OPEN:
			Morphology_Open  (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(UCHAR_MAX), static_cast<uint8_t>(0u));
		break;
		
		case SE_OP_CLOSE:
			Morphology_Close (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(UCHAR_MAX), static_cast<uint8_t>(0u));
		break;

		case SE_OP_THIN:
			Morphology_Thin (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(UCHAR_MAX));
		break;

		case SE_OP_THICK:
			Morphology_Thick (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(UCHAR_MAX));
		break;
		
		case SE_OP_GRADIENT:
			Morphology_Gradient (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(UCHAR_MAX));
		break;

		default:
		break;
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
	size_t sizeSe = 0;
	const SE_Interface* seInetrface = getStructuredElemInterface(out_data);
	const SE_Type* __restrict seElementVal = (nullptr != seInetrface ? seInetrface->GetStructuredElement(sizeSe) : nullptr);

	if (nullptr == seElementVal || 0 == sizeSe)
		return PF_Err_NONE;

	const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MORPHOLOGY_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	const PF_ParamDef* pMorphologyOperation = params[MORPHOLOGY_OPERATION_TYPE];
	const SeOperation  cType = static_cast<const SeOperation>(pMorphologyOperation->u.pd.value - 1);

	switch (cType)
	{
		case SE_OP_EROSION:
			Morphology_Erode(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint16_t>(SHRT_MAX));
		break;

		case SE_OP_DILATION:
			Morphology_Dilate(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint16_t>(0u));
		break;

		case SE_OP_OPEN:
			Morphology_Open(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint16_t>(SHRT_MAX), static_cast<uint16_t>(0u));
		break;

		case SE_OP_CLOSE:
			Morphology_Close(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint16_t>(SHRT_MAX), static_cast<uint16_t>(0u));
		break;

		case SE_OP_THIN:
			Morphology_Thin(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint16_t>(SHRT_MAX));
		break;

		case SE_OP_THICK:
			Morphology_Thick(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint16_t>(SHRT_MAX));
		break;

		case SE_OP_GRADIENT:
			Morphology_Gradient(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint16_t>(SHRT_MAX));
		break;

		default:
		break;
	}

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
	size_t sizeSe = 0;
	const SE_Interface* seInetrface = getStructuredElemInterface(out_data);
	const SE_Type* __restrict seElementVal = (nullptr != seInetrface ? seInetrface->GetStructuredElement(sizeSe) : nullptr);

	if (nullptr == seElementVal || 0 == sizeSe)
		return PF_Err_NONE;

	const PF_LayerDef*       __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MORPHOLOGY_FILTER_INPUT]->u.ld);
	const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	const PF_ParamDef* pMorphologyOperation = params[MORPHOLOGY_OPERATION_TYPE];
	const SeOperation  cType = static_cast<const SeOperation>(pMorphologyOperation->u.pd.value - 1);

	switch (cType)
	{
		case SE_OP_EROSION:
			Morphology_Erode(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 1.f);
		break;

		case SE_OP_DILATION:
			Morphology_Dilate(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 0.f);
		break;

		case SE_OP_OPEN:
			Morphology_Open(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 1.f, 0.f);
		break;

		case SE_OP_CLOSE:
			Morphology_Close(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 1.f, 0.f);
		break;

		case SE_OP_THIN:
			Morphology_Thin(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 1.f);
		break;

		case SE_OP_THICK:
			Morphology_Thick(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 1.f);
		break;

		case SE_OP_GRADIENT:
			Morphology_Gradient(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 1.f);
		break;

		default:
		break;
	}

	return PF_Err_NONE;
}

#if 1
PF_Err MorphologyFilter_VUYA_4444_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	size_t sizeSe = 0;
	const SE_Interface* seInetrface = getStructuredElemInterface(out_data);
	const SE_Type* __restrict seElementVal = (nullptr != seInetrface ? seInetrface->GetStructuredElement(sizeSe) : nullptr);

	if (nullptr == seElementVal || 0 == sizeSe)
		return PF_Err_NONE;

	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MORPHOLOGY_FILTER_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	const PF_ParamDef* pMorphologyOperation = params[MORPHOLOGY_OPERATION_TYPE];
	const SeOperation  cType = static_cast<const SeOperation>(pMorphologyOperation->u.pd.value - 1);

	switch (cType)
	{
		case SE_OP_EROSION:
			Morphology_Erode (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(UCHAR_MAX));
		break;

		case SE_OP_DILATION:
			Morphology_Dilate (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(0u));
		break;

		case SE_OP_OPEN:
			Morphology_Open (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(0u), static_cast<uint8_t>(UCHAR_MAX));
		break;

		case SE_OP_CLOSE:
			Morphology_Close (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(0u), static_cast<uint8_t>(UCHAR_MAX));
		break;

		case SE_OP_THIN:
			Morphology_Thin (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(UCHAR_MAX));
		break;

		case SE_OP_THICK:
			Morphology_Thick (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(UCHAR_MAX));
		break;

		case SE_OP_GRADIENT:
			Morphology_Gradient (localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, static_cast<uint8_t>(UCHAR_MAX));
		break;

		default:
		break;
	}

	return PF_Err_NONE;
}


PF_Err MorphologyFilter_VUYA_4444_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	size_t sizeSe = 0;
	const SE_Interface* seInetrface = getStructuredElemInterface(out_data);
	const SE_Type* __restrict seElementVal = (nullptr != seInetrface ? seInetrface->GetStructuredElement(sizeSe) : nullptr);

	if (nullptr == seElementVal || 0 == sizeSe)
		return PF_Err_NONE;

	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[MORPHOLOGY_FILTER_INPUT]->u.ld);
	const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	const PF_ParamDef* pMorphologyOperation = params[MORPHOLOGY_OPERATION_TYPE];
	const SeOperation  cType = static_cast<const SeOperation>(pMorphologyOperation->u.pd.value - 1);

	switch (cType)
	{
		case SE_OP_EROSION:
			Morphology_Erode(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 1.f);
		break;

		case SE_OP_DILATION:
			Morphology_Dilate(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 0.f);
		break;

		case SE_OP_OPEN:
			Morphology_Open(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 0.f, 1.f);
		break;

		case SE_OP_CLOSE:
			Morphology_Close(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 0.f, 1.f);
		break;

		case SE_OP_THIN:
			Morphology_Thin(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 1.f);
		break;

		case SE_OP_THICK:
			Morphology_Thick(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 1.f);
		break;

		case SE_OP_GRADIENT:
			Morphology_Gradient(localSrc, localDst, seElementVal, sizeSe, height, width, line_pitch, line_pitch, 1.f);
		break;

		default:
		break;
	}

	return PF_Err_NONE;
}
#endif


PF_Err ProcessImgInPR
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	PF_Err err = PF_Err_NONE;
	PrPixelFormat destinationPixelFormat{ PrPixelFormat_Invalid };

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

			case PrPixelFormat_VUYA_4444_8u:
			case PrPixelFormat_VUYA_4444_8u_709:
				err = MorphologyFilter_VUYA_4444_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f:
			case PrPixelFormat_VUYA_4444_32f_709:
				err = MorphologyFilter_VUYA_4444_32f (in_data, out_data, params, output);
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

	return err;
}
