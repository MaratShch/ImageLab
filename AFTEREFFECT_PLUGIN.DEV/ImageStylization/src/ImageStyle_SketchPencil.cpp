#include "ImageStylization.hpp"
#include "StylizationStructs.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "FastAriphmetics.hpp"
#include "ImageAuxPixFormat.hpp"
#include "StylizationImageGradient.hpp"
#include "ImageLabUtils.hpp"
#include "ImageLabMemInterface.hpp"
#include <mutex>


PF_Err PR_ImageStyle_SketchPencil_BGRA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	float*				    __restrict pTmpStorage1 = nullptr;
	float*				    __restrict pTmpStorage2 = nullptr;
	const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

	const float* __restrict rgb2yuv = (width > 720) ? RGB2YUV[BT709] : RGB2YUV[BT601];

	constexpr size_t cpuPageSize{ CPU_PAGE_SIZE };
	auto const singleBufElemSize = width * height;
	auto const singleBufMemSize  = CreateAlignment (singleBufElemSize * sizeof(float), cpuPageSize);
	auto const requiredMemSize   = singleBufMemSize * 2;

	int j, i;
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    if (blockId >= 0 && nullptr != pMemPtr)
    {
        pTmpStorage1 = reinterpret_cast<float* __restrict>(pMemPtr);
        pTmpStorage2 = pTmpStorage1 + singleBufElemSize;

        /* compute gradinets of RGB image */
        ImageRGB_ComputeGradient(localSrc, rgb2yuv, pTmpStorage1, pTmpStorage2, height, width, line_pitch);

        for (j = 0; j < height; j++)
        {
            const float* __restrict pSrc1Line = pTmpStorage1 + j * width;
            const float* __restrict pSrc2Line = pTmpStorage2 + j * width;
            const PF_Pixel_BGRA_8u* __restrict pSrcLine = localSrc + j * line_pitch;
            PF_Pixel_BGRA_8u*       __restrict pDstLine = localDst + j * line_pitch;

            __VECTOR_ALIGNED__
            for (i = 0; i < width; i++)
            {
                const int32_t sqrtVal = FastCompute::Min(255, static_cast<int32_t>(FastCompute::Sqrt(pSrc1Line[i] * pSrc1Line[i] + pSrc2Line[i] * pSrc2Line[i])));
                const int32_t negVal = 255 - sqrtVal;
                pDstLine[i].B = pDstLine[i].G = pDstLine[i].R = static_cast<A_u_char>(negVal);
                pDstLine[i].A = pSrcLine[i].A;
            } /* for (i = 0; i < width; i++) */

        } /* for (j = 0; j < height; j++) */

        ::FreeMemoryBlock(blockId);
        blockId = -1;
    } // if (blockId >= 0 && nullptr != pMemPtr)

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
	float*				    __restrict pTmpStorage1 = nullptr;
	float*				    __restrict pTmpStorage2 = nullptr;
	const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_8u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	constexpr size_t cpuPageSize{ CPU_PAGE_SIZE };
	auto const singleBufElemSize = width * height;
	auto const singleBufMemSize = CreateAlignment(singleBufElemSize * sizeof(float), cpuPageSize);
	auto const requiredMemSize = singleBufMemSize * 2;

    int j, i;
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    if (blockId >= 0 && nullptr != pMemPtr)
    {
        pTmpStorage1 = reinterpret_cast<float* __restrict>(pMemPtr);
        pTmpStorage2 = pTmpStorage1 + singleBufElemSize;

		/* compute gradinets of RGB image */
		ImageYUV_ComputeGradient(localSrc, pTmpStorage1, pTmpStorage2, height, width, line_pitch);

		for (j = 0; j < height; j++)
		{
			const float* __restrict pSrc1Line = pTmpStorage1 + j * width;
			const float* __restrict pSrc2Line = pTmpStorage2 + j * width;
			const PF_Pixel_VUYA_8u* __restrict pSrcLine = localSrc + j * line_pitch;
			PF_Pixel_VUYA_8u*       __restrict pDstLine = localDst + j * line_pitch;

			__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				const int32_t sqrtVal = FastCompute::Min(255, static_cast<int32_t>(FastCompute::Sqrt(pSrc1Line[i] * pSrc1Line[i] + pSrc2Line[i] * pSrc2Line[i])));
				const int32_t negVal = 255 - sqrtVal;
				pDstLine[i].V = pDstLine[i].U = static_cast<A_u_char>(0x80u);
				pDstLine[i].Y = static_cast<A_u_char>(negVal);
				pDstLine[i].A = pSrcLine[i].A;
			} /* for (i = 0; i < width; i++) */


        } /* for (j = 0; j < height; j++) */

        ::FreeMemoryBlock(blockId);
        blockId = -1;
    } // if (blockId >= 0 && nullptr != pMemPtr)

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
	float*				     __restrict pTmpStorage1 = nullptr;
	float*				     __restrict pTmpStorage2 = nullptr;
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_VUYA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_32f* __restrict>(pfLayer->data);
	PF_Pixel_VUYA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_VUYA_32f* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

	constexpr size_t cpuPageSize{ CPU_PAGE_SIZE };
	auto const singleBufElemSize = width * height;
	auto const singleBufMemSize = CreateAlignment(singleBufElemSize * sizeof(float), cpuPageSize);
	auto const requiredMemSize = singleBufMemSize * 2;

    int j, i;
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    if (blockId >= 0 && nullptr != pMemPtr)
    {
        pTmpStorage1 = reinterpret_cast<float* __restrict>(pMemPtr);
        pTmpStorage2 = pTmpStorage1 + singleBufElemSize;

		/* compute gradinets of RGB image */
		ImageYUV_ComputeGradient(localSrc, pTmpStorage1, pTmpStorage2, height, width, line_pitch);

		for (j = 0; j < height; j++)
		{
			const float* __restrict pSrc1Line = pTmpStorage1 + j * width;
			const float* __restrict pSrc2Line = pTmpStorage2 + j * width;
			const PF_Pixel_VUYA_32f* __restrict pSrcLine = localSrc + j * line_pitch;
			PF_Pixel_VUYA_32f*       __restrict pDstLine = localDst + j * line_pitch;

			__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				const float sqrtVal = FastCompute::Min(f32_value_white, FastCompute::Sqrt(pSrc1Line[i] * pSrc1Line[i] + pSrc2Line[i] * pSrc2Line[i]));
				const float negVal = f32_value_white - sqrtVal;
				pDstLine[i].V = pDstLine[i].U = 0.f;
				pDstLine[i].Y = negVal;
				pDstLine[i].A = pSrcLine[i].A;
			} /* for (i = 0; i < width; i++) */

        } /* for (j = 0; j < height; j++) */

        ::FreeMemoryBlock(blockId);
        blockId = -1;
    } // if (blockId >= 0 && nullptr != pMemPtr)

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
	float*				     __restrict pTmpStorage1 = nullptr;
	float*				     __restrict pTmpStorage2 = nullptr;
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_16u* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

	const float* __restrict rgb2yuv = (width > 720) ? RGB2YUV[BT709] : RGB2YUV[BT601];

	constexpr size_t cpuPageSize{ CPU_PAGE_SIZE };
	auto const singleBufElemSize = width * height;
	auto const singleBufMemSize = CreateAlignment(singleBufElemSize * sizeof(float), cpuPageSize);
	auto const requiredMemSize = singleBufMemSize * 2;

    int j, i;
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    if (blockId >= 0 && nullptr != pMemPtr)
    {
        pTmpStorage1 = reinterpret_cast<float* __restrict>(pMemPtr);
        pTmpStorage2 = pTmpStorage1 + singleBufElemSize;

		/* compute gradinets of RGB image */
		ImageRGB_ComputeGradient(localSrc, rgb2yuv, pTmpStorage1, pTmpStorage2, height, width, line_pitch);

		for (j = 0; j < height; j++)
		{
			const float* __restrict pSrc1Line = pTmpStorage1 + j * width;
			const float* __restrict pSrc2Line = pTmpStorage2 + j * width;
			const PF_Pixel_BGRA_16u* __restrict pSrcLine = localSrc + j * line_pitch;
			PF_Pixel_BGRA_16u*       __restrict pDstLine = localDst + j * line_pitch;

			__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				const int32_t sqrtVal = FastCompute::Min(32767, static_cast<int32_t>(FastCompute::Sqrt(pSrc1Line[i] * pSrc1Line[i] + pSrc2Line[i] * pSrc2Line[i])));
				const int32_t negVal = 32767 - sqrtVal;
				pDstLine[i].B = pDstLine[i].G = pDstLine[i].R = negVal;
				pDstLine[i].A = pSrcLine[i].A;
			} /* for (i = 0; i < width; i++) */

        } /* for (j = 0; j < height; j++) */

        ::FreeMemoryBlock(blockId);
        blockId = -1;
    } // if (blockId >= 0 && nullptr != pMemPtr)

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
	float*				     __restrict pTmpStorage1 = nullptr;
	float*				     __restrict pTmpStorage2 = nullptr;
	const PF_LayerDef*       __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_32f* __restrict>(output->data);

	auto const& height = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	auto const& width = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
	auto const& line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

	const float* __restrict rgb2yuv = (width > 720) ? RGB2YUV[BT709] : RGB2YUV[BT601];

	constexpr size_t cpuPageSize{ CPU_PAGE_SIZE };
	auto const singleBufElemSize = width * height;
	auto const singleBufMemSize = CreateAlignment(singleBufElemSize * sizeof(float), cpuPageSize);
	auto const requiredMemSize = singleBufMemSize * 2;

    int j, i;
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    if (blockId >= 0 && nullptr != pMemPtr)
    {
        pTmpStorage1 = reinterpret_cast<float* __restrict>(pMemPtr);
        pTmpStorage2 = pTmpStorage1 + singleBufElemSize;

		/* compute gradinets of RGB image */
		ImageRGB_ComputeGradient(localSrc, rgb2yuv, pTmpStorage1, pTmpStorage2, height, width, line_pitch);

		for (j = 0; j < height; j++)
		{
			const float* __restrict pSrc1Line = pTmpStorage1 + j * width;
			const float* __restrict pSrc2Line = pTmpStorage2 + j * width;
			const PF_Pixel_BGRA_32f* __restrict pSrcLine = localSrc + j * line_pitch;
			PF_Pixel_BGRA_32f*       __restrict pDstLine = localDst + j * line_pitch;

			__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				const float sqrtVal = FastCompute::Min(f32_value_white, FastCompute::Sqrt(pSrc1Line[i] * pSrc1Line[i] + pSrc2Line[i] * pSrc2Line[i]));
				const float negVal = f32_value_white - sqrtVal;
				pDstLine[i].B = pDstLine[i].G = pDstLine[i].R = negVal;
				pDstLine[i].A = pSrcLine[i].A;
			} /* for (i = 0; i < width; i++) */

        } /* for (j = 0; j < height; j++) */

        ::FreeMemoryBlock(blockId);
        blockId = -1;
    } // if (blockId >= 0 && nullptr != pMemPtr)

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
	float*				  __restrict pTmpStorage1 = nullptr;
	float*				  __restrict pTmpStorage2 = nullptr;
	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_ARGB_8u*     __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input->data);
	PF_Pixel_ARGB_8u*     __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);

	const A_long& height = output->height;
	const A_long& width  = output->width;
	const A_long& src_line_pitch = input->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	const A_long& dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	const float* __restrict rgb2yuv = (width > 720) ? RGB2YUV[BT709] : RGB2YUV[BT601];

	constexpr size_t cpuPageSize{ CPU_PAGE_SIZE };
	auto const singleBufElemSize = width * height;
	auto const singleBufMemSize = CreateAlignment(singleBufElemSize * sizeof(float), cpuPageSize);
	auto const requiredMemSize = singleBufMemSize * 2;

    int j, i;
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    if (blockId >= 0 && nullptr != pMemPtr)
    {
        pTmpStorage1 = reinterpret_cast<float* __restrict>(pMemPtr);
        pTmpStorage2 = pTmpStorage1 + singleBufElemSize;

		/* compute gradinets of RGB image */
		ImageRGB_ComputeGradient(localSrc, rgb2yuv, pTmpStorage1, pTmpStorage2, height, width, src_line_pitch);

		for (j = 0; j < height; j++)
		{
			const float* __restrict pSrc1Line = pTmpStorage1 + j * width;
			const float* __restrict pSrc2Line = pTmpStorage2 + j * width;
			const PF_Pixel_ARGB_8u* __restrict pSrcLine = localSrc + j * src_line_pitch;
			PF_Pixel_ARGB_8u*       __restrict pDstLine = localDst + j * dst_line_pitch;

			__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				const int32_t sqrtVal = FastCompute::Min(255, static_cast<int32_t>(FastCompute::Sqrt(pSrc1Line[i] * pSrc1Line[i] + pSrc2Line[i] * pSrc2Line[i])));
				const int32_t negVal = 255 - sqrtVal;
				pDstLine[i].B = pDstLine[i].G = pDstLine[i].R = negVal;
				pDstLine[i].A = pSrcLine[i].A;
			} /* for (i = 0; i < width; i++) */

        } /* for (j = 0; j < height; j++) */

        ::FreeMemoryBlock(blockId);
        blockId = -1;
    } // if (blockId >= 0 && nullptr != pMemPtr)

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
	float*				  __restrict pTmpStorage1 = nullptr;
	float*				  __restrict pTmpStorage2 = nullptr;
	const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	PF_Pixel_ARGB_16u*    __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(input->data);
	PF_Pixel_ARGB_16u*    __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output->data);

	const A_long& height = output->height;
	const A_long& width  = output->width;
	const A_long& src_line_pitch = input->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
	const A_long& dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

	const float* __restrict rgb2yuv = (width > 720) ? RGB2YUV[BT709] : RGB2YUV[BT601];

	constexpr size_t cpuPageSize{ CPU_PAGE_SIZE };
	auto const singleBufElemSize = width * height;
	auto const singleBufMemSize = CreateAlignment(singleBufElemSize * sizeof(float), cpuPageSize);
	auto const requiredMemSize = singleBufMemSize * 2;

    int j, i;
    void* pMemPtr = nullptr;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    if (blockId >= 0 && nullptr != pMemPtr)
    {
        pTmpStorage1 = reinterpret_cast<float* __restrict>(pMemPtr);
        pTmpStorage2 = pTmpStorage1 + singleBufElemSize;

		/* compute gradinets of RGB image */
		ImageRGB_ComputeGradient(localSrc, rgb2yuv, pTmpStorage1, pTmpStorage2, height, width, src_line_pitch);

		for (j = 0; j < height; j++)
		{
			const float* __restrict pSrc1Line = pTmpStorage1 + j * width;
			const float* __restrict pSrc2Line = pTmpStorage2 + j * width;
			const PF_Pixel_ARGB_16u* __restrict pSrcLine = localSrc + j * src_line_pitch;
			PF_Pixel_ARGB_16u*       __restrict pDstLine = localDst + j * dst_line_pitch;

			__VECTOR_ALIGNED__
			for (i = 0; i < width; i++)
			{
				const int32_t sqrtVal = FastCompute::Min(32767, static_cast<int32_t>(FastCompute::Sqrt(pSrc1Line[i] * pSrc1Line[i] + pSrc2Line[i] * pSrc2Line[i])));
				const int32_t negVal = 32767 - sqrtVal;
				pDstLine[i].B = pDstLine[i].G = pDstLine[i].R = negVal;
				pDstLine[i].A = pSrcLine[i].A;
			} /* for (i = 0; i < width; i++) */

        } /* for (j = 0; j < height; j++) */

        ::FreeMemoryBlock(blockId);
        blockId = -1;
    } // if (blockId >= 0 && nullptr != pMemPtr)

    return PF_Err_NONE;
}