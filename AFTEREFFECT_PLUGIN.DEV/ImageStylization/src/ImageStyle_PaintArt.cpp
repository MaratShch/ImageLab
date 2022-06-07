#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "ImageAuxPixFormat.hpp"
#include "ImagePaintUtils.hpp"

#if 1
void write_bw_to_destination
(
	const float* __restrict pSrc,					/* processed image - Y							 	*/
	const PF_Pixel_BGRA_8u* __restrict pSrcOrig,	/* original source - for take alpha channel value	*/
	PF_Pixel_BGRA_8u* __restrict pDst,				/* destination buffer								*/
	A_long sizeX,
	A_long sizeY,
	A_long srcPitch,
	A_long dstPitch,
	A_long procPitch
) noexcept
{
	for (A_long j = 0; j < sizeY; j++)
	{
		const float* __restrict pProc = pSrc + j * procPitch;
		const PF_Pixel_BGRA_8u* __restrict pFrameSrc = pSrcOrig + j * srcPitch;
		PF_Pixel_BGRA_8u* __restrict pFrameDst = pDst + j * dstPitch;

		for (A_long i = 0; i < sizeX; i++)
		{
			pFrameDst[i].A = pFrameSrc[i].A;
			pFrameDst[i].B = pFrameDst[i].G = pFrameDst[i].R = static_cast<A_u_char>(CLAMP_VALUE(pProc[i], 0.f, 255.f));
		} /* for (A_long i = 0; i < sizeX; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */

	return;
}

template <class T, std::enable_if_t<!is_YUV_proc<T>::value>* = nullptr>
void write_bw_to_destination
(
	const float* __restrict pSrc,		/* processed image								 	*/	
	const T* __restrict     pSrcOrig,	/* original source - for take alpha channel value	*/
	T* __restrict pDst,					/* destination buffer								*/					
	A_long sizeX,
	A_long sizeY,
	A_long srcPitch,
	A_long dstPitch,
	A_long procPitch
) noexcept
{
	__VECTOR_ALIGNED__
	for (A_long j = 0; j < sizeY; j++)
	{
		const float* __restrict pProc = pSrc + j * procPitch;
		const T* __restrict pFrameSrc = pSrcOrig + j * srcPitch;
		      T* __restrict pFrameDst = pDst + j * dstPitch;
		
		for (A_long i = 0; i < sizeX; i++)
		{
			pFrameDst[i].A = pFrameSrc[i].A;
			pFrameDst[i].B = pFrameDst[i].G = pFrameDst[i].R = pProc[i];
		} /* for (A_long i = 0; i < sizeX; i++) */
	} /* for (A_long j = 0; j < sizeY; j++) */

	return;
}
#endif


static PF_Err PR_ImageStyle_PaintArt_BGRA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_BGRA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_8u*  __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u*  __restrict>(output->data);
	PF_Err errCode{ PF_Err_NONE };

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	constexpr float normalizer = 255.f;
	constexpr float reciproc180 = 1.0f / 180.0f;
	constexpr float angular = 9.f;	/* read value from sliders	*/
	constexpr float angle = 30.f;	/* read value from sliders	*/	
	constexpr A_long iter = 5;		/* read value from sliders	*/
	constexpr float coCircParam = FastCompute::PI * angular * reciproc180;
	constexpr float coConeParam = FastCompute::PI * angle   * reciproc180;
	const float coCirc = std::cos(coCircParam);
	const float coCone = std::cos(coConeParam);
	const float sigma = 5.0f;

	/* allocate memory for store temporary results */
	float* pPtr1 = allocTmpBuffer (height, width);
	float* pPtr2 = allocTmpBuffer (height, width);
	float* pPtr3 = allocTmpBuffer (height, width);

	bool cocircularityRes = false;

	if (nullptr != pPtr1 && nullptr != pPtr2 && nullptr != pPtr3)
	{
		auto const& tmpBufPitch = width;

		/* convert RGB to BW */
		Color2YUV(localSrc, pPtr1, pPtr2, pPtr3, width, height, line_pitch, tmpBufPitch, -128.f);

		const A_long frameSize = width * height;
		auto sparseMatrix = std::make_unique<SparseMatrix<float>>(frameSize);

		if (sparseMatrix && true == (cocircularityRes = bw_image2cocircularity_graph (pPtr1, sparseMatrix, width, height, tmpBufPitch, sigma, coCirc, coCone, 7)))
		{
			const A_long nonZeros = count_sparse_matrix_non_zeros (sparseMatrix, frameSize);

			auto I = std::make_unique<A_long[]>(nonZeros);
			auto J = std::make_unique<A_long[]>(nonZeros);
			auto Weights = std::make_unique<float[]>(nonZeros);
			auto ImRes   = std::make_unique<float[]>(frameSize);

			if (I && J && Weights && ImRes)
			{
				sparse_matrix_to_arrays (sparseMatrix, I, J, Weights, frameSize);
				auto __restrict imResPtr = ImRes.get();
#ifdef _DEBUG
				const A_long morphoRes =
#endif
				morpho_open (pPtr1, imResPtr, Weights, I, J, iter, nonZeros, width, height, normalizer);

				/* fill output buffer */
				write_bw_to_destination (imResPtr, localSrc, localDst, width, height, line_pitch, line_pitch, tmpBufPitch);
//				Write2Destination (localSrc, localDst, imResPtr, pPtr2, pPtr3, width, height, line_pitch, line_pitch, tmpBufPitch, 128.f, 255.f);
			}
			else 
				errCode = PF_Err_OUT_OF_MEMORY;
		}
		else
			errCode = PF_Err_OUT_OF_MEMORY;
	}
	else
		errCode = PF_Err_OUT_OF_MEMORY;

	/* free temporary memory */
	freeTmpBuffer (pPtr1);
	freeTmpBuffer (pPtr2);
	freeTmpBuffer (pPtr3);
	pPtr1 = pPtr2 = pPtr3 = nullptr;

	return errCode;
}


static PF_Err PR_ImageStyle_PaintArt_BGRA_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	/* convert RGB to BW */
	float* pBwImage = nullptr;
	Color2Bw (localSrc, pBwImage, width, height, line_pitch, width);

	return PF_Err_NONE;
}


static PF_Err PR_ImageStyle_PaintArt_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	/* convert RGB to BW */
	float* pBwImage = nullptr;
	Color2Bw (localSrc, pBwImage, width, height, line_pitch, width);

	return PF_Err_NONE;
}



static PF_Err PR_ImageStyle_PaintArt_VUYA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


static PF_Err PR_ImageStyle_PaintArt_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}





PF_Err PR_ImageStyle_PaintArt
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
				err = PR_ImageStyle_PaintArt_BGRA_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageStyle_PaintArt_VUYA_8u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageStyle_PaintArt_VUYA_32f (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageStyle_PaintArt_BGRA_16u (in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageStyle_PaintArt_BGRA_32f (in_data, out_data, params, output);
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


PF_Err AE_ImageStyle_PaintArt_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err AE_ImageStyle_PaintArt_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}