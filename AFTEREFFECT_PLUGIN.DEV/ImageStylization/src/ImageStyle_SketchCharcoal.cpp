#include "ImageStylization.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "ImageLabMemInterface.hpp"
#include "StylizationImageGradient.hpp"

#define __DBG_SHOW_PROC_BUFFER


// Function for compute Y (Luma) component from RGB
template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr,
          typename U, std::enable_if<std::is_floating_point<U>::value>* = nullptr>
static void Rgb2Luma_Negate
(
    const T* __restrict pSrcImg,
          U* __restrict pLumaBuffer,
    const eCOLOR_SPACE& transformSpace,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    const U val,
    const U scaler
) noexcept
{
    const float ctm[3]{ RGB2YUV[transformSpace][0], RGB2YUV[transformSpace][1], RGB2YUV[transformSpace][2] };
    for (A_long j = 0; j < sizeY; j++)
    {
        const T* __restrict pSrcLine = pSrcImg + j * srcPitch;
              U* __restrict pDstLine = pLumaBuffer + j * dstPitch;

        for (A_long i = 0; i < sizeX; i++)
            pDstLine[i] = val - scaler * (static_cast<U>(pSrcLine[i].R) * ctm[0] + static_cast<U>(pSrcLine[i].G) * ctm[1] + static_cast<U>(pSrcLine[i].B) * ctm[2]);
    }

    return;
}


// Function for compute Y (Luma) component from RGB
template <typename T, std::enable_if_t<is_RGB_proc<T>::value>* = nullptr,
          typename U, std::enable_if<std::is_floating_point<U>::value>* = nullptr>
static void ImgConvolution
(
    const U* __restrict pSketchImg, // sketch binary image in floating point
    const T* __restrict pSrcImg,    // original source imgae (required for acquire alpha value for every pixel)
          T* __restrict pDstImg,    // destination image
    A_long sizeX,                   // frame width in pixels 
    A_long sizeY,                   // frame height in pixels
    A_long srcPitch,                // source buffer line pitch in pixels
    A_long sketchPitch,             // sketch buffer line pitch in pixels
    A_long dstPitch,                // destination buffer line pitch in pixels
    const U val,                    // whitre point for clamping
    const U scaler                  // scaler for match computed result to destination buffer type 
)  noexcept
{
    CACHE_ALIGN constexpr float GaussMatrix[9] =
    {
        0.07511361f, 0.12384140f, 0.07511361f,
        0.12384141f, 0.20417996f, 0.12384140f,
        0.07511361f, 0.12384140f, 0.07511361f
    };


    return;
}


#ifdef __DBG_SHOW_PROC_BUFFER
void dbgBufferDisplay
(
    const PF_Pixel_BGRA_8u* pSrc,
    PF_Pixel_BGRA_8u* pDst,
    float* tmpResult,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch
) noexcept
{
    A_long i, j;
    for (j = 0; j < sizeY; j++)
    {
        const PF_Pixel_BGRA_8u* pLineSrc = pSrc + j * srcPitch;
              PF_Pixel_BGRA_8u* pLineDst = pDst + j * srcPitch;
        const float* pTmpResult = tmpResult + j * sizeX;
        for (i = 0; i < sizeX; i++)
        {
            pLineDst[i].A = pLineSrc[i].A;
            pLineDst[i].B = pLineDst[i].G = pLineDst[i].R = static_cast<A_u_char>(pTmpResult[i]);
        }
    }
}
#endif // __DBG_SHOW_PROC_BUFFER


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
	PF_Pixel_BGRA_8u*       __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_8u* __restrict>(output->data);
    void* pMemPtr = nullptr;

    PF_Err err = PF_Err_NONE;

    const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
	const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_8u_size);

    // Allocate memory buffer (triple buffer).
    // We proceed only with LUMA components.
    const A_long frameSize  = sizeX * sizeY;
    const A_long tmpMemSize = frameSize * sizeof(float);
    const A_long requiredMemSize = tmpMemSize * 3;
    int32_t blockId = ::GetMemoryBlock (requiredMemSize, 0, &pMemPtr);
    
    if (blockId >= 0 && nullptr != pMemPtr)
    {
#ifdef _DEBUG
        // cleanup memory buffer
        memset(pMemPtr, 0, static_cast<std::size_t>(requiredMemSize));
#endif

        float* pTmpStorage1 = static_cast<float*>(pMemPtr);
        float* pTmpStorage2 = pTmpStorage1 + frameSize;
        float* pTmpStorage3 = pTmpStorage2 + frameSize;

        // convert RGB to YUV and store only Y (Luma) component into temporary memory buffer with sizes equal tio power of 2
        Rgb2Luma_Negate (localSrc, pTmpStorage1, BT709, sizeX, sizeY, line_pitch, sizeX, static_cast<float>(u8_value_white), 1.f);

        // Compute LUMA - gradient and binarize result (final result stored into pTmpStorage1 with overwrite input contant)
        ImageBW_ComputeGradientBin (pTmpStorage1, pTmpStorage2, pTmpStorage3, sizeX, sizeY);

        // Authomatic thresholding (use Otsu's method)
        ImageBW_AutomaticThreshold (pTmpStorage1, pTmpStorage2, sizeX, sizeY);

        // Final convolution
        ImgConvolution (pTmpStorage2, localSrc, localDst, sizeX, sizeY, line_pitch, sizeX, line_pitch, static_cast<float>(u8_value_white), 1.f);

        // discard memory 
        pTmpStorage1 = pTmpStorage2 = pTmpStorage3 = nullptr;
        pMemPtr = nullptr;
        ::FreeMemoryBlock (blockId);
        blockId = -1;

    } // if (blockId >= 0 && nullptr != pMemPtr)
    else
        err = PF_Err_OUT_OF_MEMORY;

	return err;
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
    const PF_EffectWorld* __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    PF_Pixel_ARGB_8u*     __restrict localSrc = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(input->data);
    PF_Pixel_ARGB_8u*     __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);
    void* pMemPtr = nullptr;

    PF_Err err = PF_Err_NONE;

    const A_long sizeY = output->height;
    const A_long sizeX = output->width;
    const A_long src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_8u_size);
    const A_long dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_8u_size);

    // Allocate memory buffer (triple buffer) for padded size.
    // We proceed only with LUMA components.
    const A_long frameSize  = sizeX * sizeY;
    const A_long tmpMemSize = frameSize * sizeof(float);
    const A_long requiredMemSize = tmpMemSize * 3;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    if (blockId >= 0 && nullptr != pMemPtr)
    {
#ifdef _DEBUG
        // cleanup memory buffer
        memset(pMemPtr, 0, static_cast<std::size_t>(requiredMemSize));
#endif

        float* __restrict pTmpStorage1 = reinterpret_cast<float* __restrict>(pMemPtr);
        float* __restrict pTmpStorage2 = pTmpStorage1 + frameSize;
        float* __restrict pTmpStorage3 = pTmpStorage2 + frameSize;

        // convert RGB to YUV and store only Y (Luma) component into temporary memory buffer with sizes equal tio power of 2
        Rgb2Luma_Negate (localSrc, pTmpStorage1, BT709, sizeX, sizeY, src_line_pitch, sizeX, static_cast<float>(u8_value_white), 1.f);

        // Compute LUMA - gradient and binarize result (final result stored into pTmpStorage1 with overwrite input contant)
        ImageBW_ComputeGradientBin(pTmpStorage1, pTmpStorage2, pTmpStorage3, sizeX, sizeY);

        // Authomatic thresholding (use Otsu's method)
        ImageBW_AutomaticThreshold(pTmpStorage1, pTmpStorage2, sizeX, sizeY);

        // discard memory 
        pTmpStorage1 = pTmpStorage2 = pTmpStorage3 = nullptr;
        pMemPtr = nullptr;
        ::FreeMemoryBlock(blockId);
        blockId = -1;

} // if (blockId >= 0 && nullptr != pMemPtr)
    else
        err = PF_Err_OUT_OF_MEMORY;

    return err;
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

