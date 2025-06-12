#include "ImageFFT.hpp"
#include "ImageStylization.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "ImageLabMemInterface.hpp"
#include "algo_fft.hpp"


// Function for compute Y (Luma) component from RGB
template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr,
          typename U, std::enable_if<std::is_floating_point<U>::value>* = nullptr>
static void Rgb2Luma
(
    const T* __restrict pSrcImg,
          U* __restrict pLumaBuffer,
    const eCOLOR_SPACE& transformSpace,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch
) noexcept
{
    const float ctm[3]{ RGB2YUV[transformSpace][0], RGB2YUV[transformSpace][1], RGB2YUV[transformSpace][2] };
    for (A_long j = 0; j < sizeY; j++)
    {
        const T* __restrict pSrcLine = pSrcImg + j * srcPitch;
              U* __restrict pDstLine = pLumaBuffer + j * dstPitch;

        for (A_long i = 0; i < sizeX; i++)
            pDstLine[i] = static_cast<U>(pSrcLine[i].R) * ctm[0] + static_cast<U>(pSrcLine[i].G) * ctm[1] + static_cast<U>(pSrcLine[i].B) * ctm[2];
    }

    return;
}


PF_Err PR_ImageStyle_SketchCharcoal_BGRA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	const PF_LayerDef*      __restrict pfLayer  = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
	const PF_Pixel_BGRA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_8u* __restrict>(pfLayer->data);
	PF_Pixel_BGRA_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_BGRA_8u* __restrict>(output->data);
    void* pMemPtr = nullptr;

    const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

    A_long padded_sizeX = 0, padded_sizeY = 0;
    // compute padde image size as power of 2
    get_padded_image_size (sizeX, sizeY, padded_sizeX, padded_sizeY);

    // Allocate memory buffer (double buffer) for padded size.
    // We proceed only with LUMA components.
    const A_long tmpMemSize = padded_sizeX * padded_sizeY * sizeof(float);
    const A_long requiredMemSize = tmpMemSize * 2;
    int32_t blockId = ::GetMemoryBlock (requiredMemSize, 0, &pMemPtr);
    
    if (blockId >= 0 && nullptr != pMemPtr)
    {
#ifdef _DEBUG
        // cleanup memory buffer
        memset(pMemPtr, 0, static_cast<std::size_t>(requiredMemSize));
#endif

        float* __restrict pTmpStorage1 = reinterpret_cast<float* __restrict>(pMemPtr);
        float* __restrict pTmpStorage2 = pTmpStorage1 + tmpMemSize;

        // convert RGB to YUV and store only Y (Luma) component into temporary memory buffer with sizes equal tio power of 2
        Rgb2Luma (localSrc, pTmpStorage1, BT709, sizeX, sizeY, line_pitch, padded_sizeX);

        // HP filter

        // discard memory 
        pTmpStorage1 = pTmpStorage2 = nullptr;
        pMemPtr = nullptr;
        ::FreeMemoryBlock (blockId);
        blockId = -1;

    } // if (blockId >= 0 && nullptr != pMemPtr)

	return PF_Err_NONE;
}


PF_Err PR_ImageStyle_SketchCharcoal_VUYA_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err PR_ImageStyle_SketchCharcoal_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err PR_ImageStyle_SketchCharcoal_BGRA_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err PR_ImageStyle_SketchCharcoal_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}



PF_Err PR_ImageStyle_SketchCharcoal
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
				err = PR_ImageStyle_SketchCharcoal_BGRA_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_8u_709:
			case PrPixelFormat_VUYA_4444_8u:
				err = PR_ImageStyle_SketchCharcoal_VUYA_8u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_VUYA_4444_32f_709:
			case PrPixelFormat_VUYA_4444_32f:
				err = PR_ImageStyle_SketchCharcoal_VUYA_32f(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_16u:
				err = PR_ImageStyle_SketchCharcoal_BGRA_16u(in_data, out_data, params, output);
			break;

			case PrPixelFormat_BGRA_4444_32f:
				err = PR_ImageStyle_SketchCharcoal_BGRA_32f(in_data, out_data, params, output);
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


PF_Err AE_ImageStyle_SketchCharcoal_ARGB_8u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err AE_ImageStyle_SketchCharcoal_ARGB_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
	return PF_Err_NONE;
}


PF_Err AE_ImageStyle_SketchCharcoal_ARGB_32f
(
    PF_InData*   __restrict in_data,
    PF_OutData*  __restrict out_data,
    PF_ParamDef* __restrict params[],
    PF_LayerDef* __restrict output
) noexcept
{
    return PF_Err_NONE;
}

