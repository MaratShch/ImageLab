#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "ImageAuxPixFormat.hpp"
#include "ImagePaintUtils.hpp"
#include "ImageLabMemInterface.hpp"


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
	constexpr float sigma = 5.0f;

	/* allocate memory for store temporary results */
    const size_t frameSize = static_cast<size_t>(height * width);
    const size_t requiredMemSize = CreateAlignment(frameSize * sizeof(float) * 3u, static_cast<size_t>(CACHE_LINE));
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

	bool cocircularityRes = false;

    if (nullptr != pMemPtr && blockId >= 0)
    {
		auto const tmpBufPitch = width;
        float* pPtr1 = reinterpret_cast<float*>(pMemPtr);
        float* pPtr2 = pPtr1 + frameSize;
        float* pPtr3 = pPtr2 + frameSize;

		/* convert RGB to BW */
		Color2YUV (localSrc, pPtr1, pPtr2, pPtr3, width, height, line_pitch, tmpBufPitch);

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
				auto imResPtr = ImRes.get();
#ifdef _DEBUG
				const A_long morphoRes =
#endif
				morpho_open (pPtr1, imResPtr, Weights, I, J, iter, nonZeros, width, height, normalizer);

				/* fill output buffer */
				Write2Destination (localSrc, localDst, imResPtr /* Y */, pPtr2 /* U */, pPtr3 /* V */, width, height, line_pitch, line_pitch, tmpBufPitch);
			}
			else 
				errCode = PF_Err_OUT_OF_MEMORY;
		}
		else
			errCode = PF_Err_OUT_OF_MEMORY;

        pPtr1 = pPtr2 = pPtr3 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;
    }
	else
		errCode = PF_Err_OUT_OF_MEMORY;

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
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<const PF_LayerDef*  __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u* __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);
	PF_Err errCode{ PF_Err_NONE };

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	constexpr float normalizer = 255.f;
	constexpr float reciproc180 = 1.0f / 180.0f;
	constexpr float angular = 9.f;	/* read value from sliders	*/
	constexpr float angle = 30.f;	/* read value from sliders	*/
	constexpr A_long iter = 5;		/* read value from sliders	*/
	constexpr float coCircParam = FastCompute::PI * angular * reciproc180;
	constexpr float coConeParam = FastCompute::PI * angle   * reciproc180;
	const float coCirc = std::cos(coCircParam);
	const float coCone = std::cos(coConeParam);
	constexpr float sigma = 5.0f;

    /* allocate memory for store temporary results */
    const size_t frameSize = static_cast<size_t>(height * width);
    const size_t requiredMemSize = CreateAlignment(frameSize * sizeof(float) * 3u, static_cast<size_t>(CACHE_LINE));
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    bool cocircularityRes = false;

    if (nullptr != pMemPtr && blockId >= 0)
    {
        auto const tmpBufPitch = width;
        float* pPtr1 = reinterpret_cast<float*>(pMemPtr);
        float* pPtr2 = pPtr1 + frameSize;
        float* pPtr3 = pPtr2 + frameSize;

		/* convert RGB to BW */
		Color2YUV (localSrc, pPtr1, pPtr2, pPtr3, width, height, line_pitch, tmpBufPitch);

		const A_long frameSize = width * height;
		auto sparseMatrix = std::make_unique<SparseMatrix<float>>(frameSize);

		if (sparseMatrix && true == (cocircularityRes = bw_image2cocircularity_graph(pPtr1, sparseMatrix, width, height, tmpBufPitch, sigma, coCirc, coCone, 7)))
		{
			const A_long nonZeros = count_sparse_matrix_non_zeros(sparseMatrix, frameSize);

			auto I = std::make_unique<A_long[]>(nonZeros);
			auto J = std::make_unique<A_long[]>(nonZeros);
			auto Weights = std::make_unique<float[]>(nonZeros);
			auto ImRes = std::make_unique<float[]>(frameSize);

			if (I && J && Weights && ImRes)
			{
				sparse_matrix_to_arrays(sparseMatrix, I, J, Weights, frameSize);
				auto imResPtr = ImRes.get();
#ifdef _DEBUG
				const A_long morphoRes =
#endif
				morpho_open(pPtr1, imResPtr, Weights, I, J, iter, nonZeros, width, height, normalizer);

				/* fill output buffer */
				constexpr float clamp_val{ 32767.f };
				Write2Destination (localSrc, localDst, imResPtr /* Y */, pPtr2 /* U */, pPtr3 /* V */, width, height, line_pitch, line_pitch, tmpBufPitch, clamp_val);
			}
			else
				errCode = PF_Err_OUT_OF_MEMORY;
		}
		else
			errCode = PF_Err_OUT_OF_MEMORY;

        pPtr1 = pPtr2 = pPtr3 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;
    }
    else
        errCode = PF_Err_OUT_OF_MEMORY;

	return errCode;
}


static PF_Err PR_ImageStyle_PaintArt_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<const PF_LayerDef*  __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f* __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);
	PF_Err errCode{ PF_Err_NONE };

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	constexpr float normalizer = 255.f;
	constexpr float reciproc180 = 1.0f / 180.0f;
	constexpr float angular = 9.f;	/* read value from sliders	*/
	constexpr float angle = 30.f;	/* read value from sliders	*/
	constexpr A_long iter = 5;		/* read value from sliders	*/
	constexpr float coCircParam = FastCompute::PI * angular * reciproc180;
	constexpr float coConeParam = FastCompute::PI * angle   * reciproc180;
	const float coCirc = std::cos(coCircParam);
	const float coCone = std::cos(coConeParam);
	constexpr float sigma = 5.0f;

    /* allocate memory for store temporary results */
    const size_t frameSize = static_cast<size_t>(height * width);
    const size_t requiredMemSize = CreateAlignment(frameSize * sizeof(float) * 3u, static_cast<size_t>(CACHE_LINE));
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    bool cocircularityRes = false;

    if (nullptr != pMemPtr && blockId >= 0)
    {
        auto const tmpBufPitch = width;
        float* pPtr1 = reinterpret_cast<float*>(pMemPtr);
        float* pPtr2 = pPtr1 + frameSize;
        float* pPtr3 = pPtr2 + frameSize;

		/* convert RGB to BW */
		Color2YUV (localSrc, pPtr1, pPtr2, pPtr3, width, height, line_pitch, tmpBufPitch);

		const A_long frameSize = width * height;
		auto sparseMatrix = std::make_unique<SparseMatrix<float>>(frameSize);

		if (sparseMatrix && true == (cocircularityRes = bw_image2cocircularity_graph (pPtr1, sparseMatrix, width, height, tmpBufPitch, sigma, coCirc, coCone, 7)))
		{
			const A_long nonZeros = count_sparse_matrix_non_zeros (sparseMatrix, frameSize);

			auto I = std::make_unique<A_long[]>(nonZeros);
			auto J = std::make_unique<A_long[]>(nonZeros);
			auto Weights = std::make_unique<float[]>(nonZeros);
			auto ImRes = std::make_unique<float[]>(frameSize);

			if (I && J && Weights && ImRes)
			{
				sparse_matrix_to_arrays(sparseMatrix, I, J, Weights, frameSize);
				auto imResPtr = ImRes.get();
#ifdef _DEBUG
				const A_long morphoRes =
#endif
				morpho_open(pPtr1, imResPtr, Weights, I, J, iter, nonZeros, width, height, normalizer);

				/* fill output buffer */
				constexpr float clamp_val{ f32_value_white };
				Write2Destination (localSrc, localDst, imResPtr /* Y */, pPtr2 /* U */, pPtr3 /* V */, width, height, line_pitch, line_pitch, tmpBufPitch, clamp_val);
			}
			else
				errCode = PF_Err_OUT_OF_MEMORY;
		}
		else
			errCode = PF_Err_OUT_OF_MEMORY;

        pPtr1 = pPtr2 = pPtr3 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;
    }
    else
        errCode = PF_Err_OUT_OF_MEMORY;

	return errCode;
}



static PF_Err PR_ImageStyle_PaintArt_VUYA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer = reinterpret_cast<const PF_LayerDef*  __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_VUYA_8u*  __restrict localSrc = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*  __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);
	PF_Err errCode{ PF_Err_NONE };

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	constexpr float normalizer = 255.f;
	constexpr float reciproc180 = 1.0f / 180.0f;
	constexpr float angular = 9.f;	/* read value from sliders	*/
	constexpr float angle = 30.f;	/* read value from sliders	*/
	constexpr A_long iter = 5;		/* read value from sliders	*/
	constexpr float coCircParam = FastCompute::PI * angular * reciproc180;
	constexpr float coConeParam = FastCompute::PI * angle   * reciproc180;
	const float coCirc = std::cos(coCircParam);
	const float coCone = std::cos(coConeParam);
	constexpr float sigma = 5.0f;

    /* allocate memory for store temporary results */
    const size_t frameSize = static_cast<size_t>(height * width);
    const size_t requiredMemSize = CreateAlignment(frameSize * sizeof(float) * 3u, static_cast<size_t>(CACHE_LINE));
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    bool cocircularityRes = false;

    if (nullptr != pMemPtr && blockId >= 0)
    {
        auto const tmpBufPitch = width;
        float* pPtr1 = reinterpret_cast<float*>(pMemPtr);
        float* pPtr2 = pPtr1 + frameSize;
        float* pPtr3 = pPtr2 + frameSize;

		/* convert RGB to BW */
		Color2YUV (localSrc, pPtr1, pPtr2, pPtr3, width, height, line_pitch, tmpBufPitch);

		const A_long frameSize = width * height;
		auto sparseMatrix = std::make_unique<SparseMatrix<float>>(frameSize);

		if (sparseMatrix && true == (cocircularityRes = bw_image2cocircularity_graph (pPtr1, sparseMatrix, width, height, tmpBufPitch, sigma, coCirc, coCone, 7)))
		{
			const A_long nonZeros = count_sparse_matrix_non_zeros (sparseMatrix, frameSize);

			auto I = std::make_unique<A_long[]>(nonZeros);
			auto J = std::make_unique<A_long[]>(nonZeros);
			auto Weights = std::make_unique<float[]>(nonZeros);
			auto ImRes = std::make_unique<float[]>(frameSize);

			if (I && J && Weights && ImRes)
			{
				sparse_matrix_to_arrays (sparseMatrix, I, J, Weights, frameSize);
				auto imResPtr = ImRes.get();
#ifdef _DEBUG
				const A_long morphoRes =
#endif
				morpho_open (pPtr1, imResPtr, Weights, I, J, iter, nonZeros, width, height, normalizer);

				/* fill output buffer */
				Write2Destination (localSrc, localDst, imResPtr /* Y */, pPtr2 /* U */, pPtr3 /* V */, width, height, line_pitch, line_pitch, tmpBufPitch);
			}
			else
				errCode = PF_Err_OUT_OF_MEMORY;
		}
		else
			errCode = PF_Err_OUT_OF_MEMORY;

        pPtr1 = pPtr2 = pPtr3 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;
    }
    else
        errCode = PF_Err_OUT_OF_MEMORY;

	return errCode;
}


static PF_Err PR_ImageStyle_PaintArt_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef* __restrict pfLayer  = reinterpret_cast<const PF_LayerDef*  __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_32f* __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);
	PF_Err errCode{ PF_Err_NONE };

	auto const height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const width  = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_32f_size);

	constexpr float normalizer = 255.f;
	constexpr float reciproc180 = 1.0f / 180.0f;
	constexpr float angular = 9.f;	/* read value from sliders	*/
	constexpr float angle = 30.f;	/* read value from sliders	*/
	constexpr A_long iter = 5;		/* read value from sliders	*/
	constexpr float coCircParam = FastCompute::PI * angular * reciproc180;
	constexpr float coConeParam = FastCompute::PI * angle   * reciproc180;
	const float coCirc = std::cos(coCircParam);
	const float coCone = std::cos(coConeParam);
	constexpr float sigma = 5.0f;

    /* allocate memory for store temporary results */
    const size_t frameSize = static_cast<size_t>(height * width);
    const size_t requiredMemSize = CreateAlignment(frameSize * sizeof(float) * 3u, static_cast<size_t>(CACHE_LINE));
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    bool cocircularityRes = false;

    if (nullptr != pMemPtr && blockId >= 0)
    {
        auto const tmpBufPitch = width;
        float* pPtr1 = reinterpret_cast<float*>(pMemPtr);
        float* pPtr2 = pPtr1 + frameSize;
        float* pPtr3 = pPtr2 + frameSize;

		/* convert RGB to BW */
		Color2YUV(localSrc, pPtr1, pPtr2, pPtr3, width, height, line_pitch, tmpBufPitch);

		const A_long frameSize = width * height;
		auto sparseMatrix = std::make_unique<SparseMatrix<float>>(frameSize);

		if (sparseMatrix && true == (cocircularityRes = bw_image2cocircularity_graph(pPtr1, sparseMatrix, width, height, tmpBufPitch, sigma, coCirc, coCone, 7)))
		{
			const A_long nonZeros = count_sparse_matrix_non_zeros(sparseMatrix, frameSize);

			auto I = std::make_unique<A_long[]>(nonZeros);
			auto J = std::make_unique<A_long[]>(nonZeros);
			auto Weights = std::make_unique<float[]>(nonZeros);
			auto ImRes = std::make_unique<float[]>(frameSize);

			if (I && J && Weights && ImRes)
			{
				sparse_matrix_to_arrays(sparseMatrix, I, J, Weights, frameSize);
				auto imResPtr = ImRes.get();
#ifdef _DEBUG
				const A_long morphoRes =
#endif
				morpho_open(pPtr1, imResPtr, Weights, I, J, iter, nonZeros, width, height, normalizer);

				/* fill output buffer */
				constexpr float clamp_val{ f32_value_white };
				Write2Destination(localSrc, localDst, imResPtr /* Y */, pPtr2 /* U */, pPtr3 /* V */, width, height, line_pitch, line_pitch, tmpBufPitch, clamp_val);
			}
			else
				errCode = PF_Err_OUT_OF_MEMORY;
		}
		else
			errCode = PF_Err_OUT_OF_MEMORY;

        pPtr1 = pPtr2 = pPtr3 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;
    }
    else
        errCode = PF_Err_OUT_OF_MEMORY;

	return errCode;
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
	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_ARGB_8u*     __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input->data);
	PF_Pixel_ARGB_8u*     __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);
	PF_Err errCode{ PF_Err_NONE };

	const A_long& height = output->height;
	const A_long& width  = output->width;
	const A_long src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	const A_long dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	constexpr float normalizer = 255.f;
	constexpr float reciproc180 = 1.0f / 180.0f;
	constexpr float angular = 9.f;	/* read value from sliders	*/
	constexpr float angle = 30.f;	/* read value from sliders	*/
	constexpr A_long iter = 5;		/* read value from sliders	*/
	constexpr float coCircParam = FastCompute::PI * angular * reciproc180;
	constexpr float coConeParam = FastCompute::PI * angle   * reciproc180;
	const float coCirc = std::cos(coCircParam);
	const float coCone = std::cos(coConeParam);
	constexpr float sigma = 5.0f;

    /* allocate memory for store temporary results */
    const size_t frameSize = static_cast<size_t>(height * width);
    const size_t requiredMemSize = CreateAlignment(frameSize * sizeof(float) * 3u, static_cast<size_t>(CACHE_LINE));
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    bool cocircularityRes = false;

    if (nullptr != pMemPtr && blockId >= 0)
    {
        auto const tmpBufPitch = width;

        float* pPtr1 = reinterpret_cast<float*>(pMemPtr);
        float* pPtr2 = pPtr1 + frameSize;
        float* pPtr3 = pPtr2 + frameSize;

		Color2YUV (localSrc, pPtr1, pPtr2, pPtr3, width, height, src_line_pitch, tmpBufPitch);

		const A_long frameSize = width * height;
		auto sparseMatrix = std::make_unique<SparseMatrix<float>>(frameSize);

		if (sparseMatrix && true == (cocircularityRes = bw_image2cocircularity_graph(pPtr1, sparseMatrix, width, height, tmpBufPitch, sigma, coCirc, coCone, 7)))
		{
			const A_long nonZeros = count_sparse_matrix_non_zeros(sparseMatrix, frameSize);

			auto I = std::make_unique<A_long[]>(nonZeros);
			auto J = std::make_unique<A_long[]>(nonZeros);
			auto Weights = std::make_unique<float[]>(nonZeros);
			auto ImRes = std::make_unique<float[]>(frameSize);

			if (I && J && Weights && ImRes)
			{
				sparse_matrix_to_arrays(sparseMatrix, I, J, Weights, frameSize);
				auto imResPtr = ImRes.get();
#ifdef _DEBUG
				const A_long morphoRes =
#endif
				morpho_open (pPtr1, imResPtr, Weights, I, J, iter, nonZeros, width, height, normalizer);

				/* fill output buffer */
				Write2Destination(localSrc, localDst, imResPtr /* Y */, pPtr2 /* U */, pPtr3 /* V */, width, height, src_line_pitch, dst_line_pitch, tmpBufPitch);
			}
			else
				errCode = PF_Err_OUT_OF_MEMORY;
		}
		else
			errCode = PF_Err_OUT_OF_MEMORY;

        pPtr1 = pPtr2 = pPtr3 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;
    }
    else
        errCode = PF_Err_OUT_OF_MEMORY;

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
	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_ARGB_16u*    __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(input->data);
	PF_Pixel_ARGB_16u*    __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output->data);
	PF_Err errCode{ PF_Err_NONE };

	const A_long& height = output->height;
	const A_long& width = output->width;
	const A_long src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
	const A_long dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

	constexpr float normalizer = 255.f;
	constexpr float reciproc180 = 1.0f / 180.0f;
	constexpr float angular = 9.f;	/* read value from sliders	*/
	constexpr float angle = 30.f;	/* read value from sliders	*/
	constexpr A_long iter = 5;		/* read value from sliders	*/
	constexpr float coCircParam = FastCompute::PI * angular * reciproc180;
	constexpr float coConeParam = FastCompute::PI * angle   * reciproc180;
	const float coCirc = std::cos(coCircParam);
	const float coCone = std::cos(coConeParam);
	constexpr float sigma = 5.0f;

    /* allocate memory for store temporary results */
    const size_t frameSize = static_cast<size_t>(height * width);
    const size_t requiredMemSize = CreateAlignment(frameSize * sizeof(float) * 3u, static_cast<size_t>(CACHE_LINE));
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    bool cocircularityRes = false;

    if (nullptr != pMemPtr && blockId >= 0)
    {
        auto const tmpBufPitch = width;
        float* pPtr1 = reinterpret_cast<float*>(pMemPtr);
        float* pPtr2 = pPtr1 + frameSize;
        float* pPtr3 = pPtr2 + frameSize;

		Color2YUV(localSrc, pPtr1, pPtr2, pPtr3, width, height, src_line_pitch, tmpBufPitch);

		const A_long frameSize = width * height;
		auto sparseMatrix = std::make_unique<SparseMatrix<float>>(frameSize);

		if (sparseMatrix && true == (cocircularityRes = bw_image2cocircularity_graph(pPtr1, sparseMatrix, width, height, tmpBufPitch, sigma, coCirc, coCone, 7)))
		{
			const A_long nonZeros = count_sparse_matrix_non_zeros(sparseMatrix, frameSize);

			auto I = std::make_unique<A_long[]>(nonZeros);
			auto J = std::make_unique<A_long[]>(nonZeros);
			auto Weights = std::make_unique<float[]>(nonZeros);
			auto ImRes = std::make_unique<float[]>(frameSize);

			if (I && J && Weights && ImRes)
			{
				sparse_matrix_to_arrays(sparseMatrix, I, J, Weights, frameSize);
				auto imResPtr = ImRes.get();
#ifdef _DEBUG
				const A_long morphoRes =
#endif
				morpho_open(pPtr1, imResPtr, Weights, I, J, iter, nonZeros, width, height, normalizer);

				/* fill output buffer */
				Write2Destination(localSrc, localDst, imResPtr /* Y */, pPtr2 /* U */, pPtr3 /* V */, width, height, src_line_pitch, dst_line_pitch, tmpBufPitch);
			}
			else
				errCode = PF_Err_OUT_OF_MEMORY;
		}
		else
			errCode = PF_Err_OUT_OF_MEMORY;

        pPtr1 = pPtr2 = pPtr3 = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;
    }
    else
        errCode = PF_Err_OUT_OF_MEMORY;

	return PF_Err_NONE;
}


PF_Err AE_ImageStyle_PaintArt_ARGB_32f
(
    PF_InData*   __restrict in_data,
    PF_OutData*  __restrict out_data,
    PF_ParamDef* __restrict params[],
    PF_LayerDef* __restrict output
) noexcept
{
    return PF_Err_NONE; // non implemented yet
}