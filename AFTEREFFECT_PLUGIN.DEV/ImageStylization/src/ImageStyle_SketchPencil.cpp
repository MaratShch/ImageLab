#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "SegmentationUtils.hpp"
#include "ImageAuxPixFormat.hpp"

#include <mutex>
#include <math.h> 


PF_Err PR_ImageStyle_SketchPencil_BGRA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	ImageStyleTmpStorage*   __restrict pTmpStorageHdnl = nullptr;
	uint8_t*				__restrict pTmpStorage1 = nullptr;
	uint8_t*				__restrict pTmpStorage2 = nullptr;
	const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	constexpr size_t cpuPageSize{ CPU_PAGE_SIZE };
	auto const singleBufferSize = CreateAlignment (width * height * sizeof(uint8_t), cpuPageSize);
	auto const requiredMemSize = singleBufferSize * 2;

	int j, i;
	bool bMemSizeTest = false;

	bufHandle* pGlobal = static_cast<bufHandle*>(GET_OBJ_FROM_HNDL(in_data->global_data));
	if (nullptr != pGlobal)
	{
		pTmpStorageHdnl = static_cast<ImageStyleTmpStorage* __restrict>(pGlobal->pBufHndl);
		bMemSizeTest = test_temporary_buffers(pTmpStorageHdnl, requiredMemSize);
	}

	if (true == bMemSizeTest)
	{
		const std::lock_guard<std::mutex> lock(pTmpStorageHdnl->guard_buffer);
		pTmpStorage1 = reinterpret_cast<uint8_t* __restrict>(pTmpStorageHdnl->pStorage1);
		pTmpStorage2 = reinterpret_cast<uint8_t* __restrict>(pTmpStorageHdnl->pStorage1) + singleBufferSize;
#if 0
		for (j = 0; j < height; j++)
		{
			const A_long line_idx = j * line_pitch;
			const A_long tmpBufLineidx = j * width;

			__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{

			} /* for (i = 0; i < width; i++) */

		} /* for (j = 0; j < height; j++) */
#endif
	} /* if (true == bMemSizeTest) */


	return PF_Err_NONE;
}


PF_Err PR_ImageStyle_SketchPencil_VUYA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err PR_ImageStyle_SketchPencil_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err PR_ImageStyle_SketchPencil_BGRA_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err PR_ImageStyle_SketchPencil_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}



PF_Err PR_ImageStyle_SketchPencil
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
				err = PR_ImageStyle_SketchPencil_BGRA_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageStyle_SketchPencil_VUYA_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageStyle_SketchPencil_VUYA_32f(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageStyle_SketchPencil_BGRA_16u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageStyle_SketchPencil_BGRA_32f(in_data, out_data, params, output);
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


PF_Err AE_ImageStyle_SketchPencil_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err AE_ImageStyle_SketchPencil_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}