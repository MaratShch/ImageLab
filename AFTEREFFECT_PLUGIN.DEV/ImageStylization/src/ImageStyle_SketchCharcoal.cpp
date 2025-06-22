#include "ImageStylization.hpp"
#include "PrSDKAESupport.h"
#include "ColorTransformMatrix.hpp"
#include "ImageLabMemInterface.hpp"
#include "StylizationImageGradient.hpp"

#define __DBG_SHOW_PROC_BUFFER


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


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr,
          typename U, std::enable_if<std::is_floating_point<U>::value>* = nullptr>
static void Yuv2Luma_Negate
(
    const T* __restrict pSrcImg,
          U* __restrict pLumaBuffer,
    A_long sizeX,
    A_long sizeY,
    A_long srcPitch,
    A_long dstPitch,
    const U val,
    const U scaler
) noexcept
{
    for (A_long j = 0; j < sizeY; j++)
    {
        const T* __restrict pSrcLine = pSrcImg + j * srcPitch;
              U* __restrict pDstLine = pLumaBuffer + j * dstPitch;

        for (A_long i = 0; i < sizeX; i++)
            pDstLine[i] = val - scaler * pSrcLine[i].Y;
    }

    return;
}


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
    const U white,                  // white point for clamping
    const U scaler                  // scaler for match computed result to destination buffer type 
)  noexcept
{
    constexpr A_long kernelSize{ 3 };
    constexpr A_long kernelArraySize = kernelSize * kernelSize;
    CACHE_ALIGN constexpr float GaussMatrix[kernelArraySize] =
    {
        0.07511361f, 0.12384141f, 0.07511361f,
        0.12384141f, 0.20417996f, 0.12384141f,
        0.07511361f, 0.12384141f, 0.07511361f
    };

    constexpr float kernelSum = 
        GaussMatrix[0] + GaussMatrix[1] + GaussMatrix[2] +
        GaussMatrix[3] + GaussMatrix[4] + GaussMatrix[5] +
        GaussMatrix[6] + GaussMatrix[7] + GaussMatrix[8];

    const A_long lastPixIdx = sizeX - 1;

    A_long j, i, idx;

    for (j = 0; j < sizeY; j++)
    {
        const T* __restrict pSrcLine = pSrcImg + j * srcPitch;
        const U* __restrict pSketchLinePrev = pSketchImg + FastCompute::Max(0, j - 1) * sketchPitch;
        const U* __restrict pSketchLine = pSketchImg + j * sketchPitch;
        const U* __restrict pSketchLineNext = pSketchImg + FastCompute::Min(sizeY - 1, j + 1) * sketchPitch;
              T* __restrict pDstLine = pDstImg + j * dstPitch;

        __VECTOR_ALIGNED__
        for (i = 0; i < sizeX; i++)
        {
            const A_long prevPixIdx = FastCompute::Max(0, i - 1);
            const A_long nextPixIdx = FastCompute::Min(lastPixIdx, i + 1);

            // Because the kernel fully symmetric - we don't need rotate it on 90 degrees clokwise. 
            const U procPix = // Because sum of all kernel elements equal to 1 we don't need normalize output result.
                pSketchLinePrev[prevPixIdx] * GaussMatrix[0] + pSketchLinePrev[i] * GaussMatrix[1] + pSketchLinePrev[nextPixIdx] * GaussMatrix[2] +
                pSketchLine[prevPixIdx]     * GaussMatrix[3] + pSketchLine[i]     * GaussMatrix[4] + pSketchLine[nextPixIdx]     * GaussMatrix[5] +
                pSketchLineNext[prevPixIdx] * GaussMatrix[6] + pSketchLineNext[i] * GaussMatrix[7] + pSketchLineNext[nextPixIdx] * GaussMatrix[8];

            pDstLine[i].A = pSrcLine[i].A;
            pDstLine[i].R = pDstLine[i].G = pDstLine[i].B = static_cast<decltype(pDstLine[i].R)>(CLAMP_VALUE(procPix * scaler, static_cast<U>(0), white));
        } // for (i = 0; i < sizeX; i++)

    } // for (j = 0; j < sizeY; j++)

    return;
}


template <typename T, std::enable_if_t<is_YUV_proc<T>::value>* = nullptr,
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
    const U white,                  // white point for clamping
    const U scaler                  // scaler for match computed result to destination buffer type 
)  noexcept
{
    constexpr A_long kernelSize{ 3 };
    constexpr A_long kernelArraySize = kernelSize * kernelSize;
    CACHE_ALIGN constexpr float GaussMatrix[kernelArraySize] =
    {
        0.07511361f, 0.12384141f, 0.07511361f,
        0.12384141f, 0.20417996f, 0.12384141f,
        0.07511361f, 0.12384141f, 0.07511361f
    };

    constexpr float kernelSum =
        GaussMatrix[0] + GaussMatrix[1] + GaussMatrix[2] +
        GaussMatrix[3] + GaussMatrix[4] + GaussMatrix[5] +
        GaussMatrix[6] + GaussMatrix[7] + GaussMatrix[8];

    const A_long lastPixIdx = sizeX - 1;
    A_long j, i, idx;
    U uvBlack{ 0 };

    if (std::is_same<T, PF_Pixel_VUYA_8u>::value)
        uvBlack = static_cast<U>(0x80);

    for (j = 0; j < sizeY; j++)
    {
        const T* __restrict pSrcLine = pSrcImg + j * srcPitch;
        const U* __restrict pSketchLinePrev = pSketchImg + FastCompute::Max(0, j - 1) * sketchPitch;
        const U* __restrict pSketchLine = pSketchImg + j * sketchPitch;
        const U* __restrict pSketchLineNext = pSketchImg + FastCompute::Min(sizeY - 1, j + 1) * sketchPitch;
        T* __restrict pDstLine = pDstImg + j * dstPitch;

        __VECTOR_ALIGNED__
            for (i = 0; i < sizeX; i++)
            {
                const A_long prevPixIdx = FastCompute::Max(0, i - 1);
                const A_long nextPixIdx = FastCompute::Min(lastPixIdx, i + 1);

                // Because the kernel fully symmetric - we don't need rotate it on 90 degrees clokwise. 
                const U procPix = // Because sum of all kernel elements equal to 1 we don't need normalize output result.
                    pSketchLinePrev[prevPixIdx] * GaussMatrix[0] + pSketchLinePrev[i] * GaussMatrix[1] + pSketchLinePrev[nextPixIdx] * GaussMatrix[2] +
                    pSketchLine[prevPixIdx]     * GaussMatrix[3] + pSketchLine[i]     * GaussMatrix[4] + pSketchLine[nextPixIdx]     * GaussMatrix[5] +
                    pSketchLineNext[prevPixIdx] * GaussMatrix[6] + pSketchLineNext[i] * GaussMatrix[7] + pSketchLineNext[nextPixIdx] * GaussMatrix[8];

                pDstLine[i].A = pSrcLine[i].A;
                pDstLine[i].Y = static_cast<decltype(pDstLine[i].Y)>(CLAMP_VALUE(procPix * scaler, static_cast<U>(0), white));
                pDstLine[i].U = pDstLine[i].V = static_cast<decltype(pDstLine[i].U)>(uvBlack);
            } // for (i = 0; i < sizeX; i++)

    } // for (j = 0; j < sizeY; j++)

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

        // convert RGB to YUV and store only Y (Luma) component normalized to range 0...255 
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
    const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
    PF_Pixel_VUYA_8u*       __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
    void* pMemPtr = nullptr;

    PF_Err err = PF_Err_NONE;

    const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

    // Allocate memory buffer (triple buffer).
    // We proceed only with LUMA components.
    const A_long frameSize = sizeX * sizeY;
    const A_long tmpMemSize = frameSize * sizeof(float);
    const A_long requiredMemSize = tmpMemSize * 3;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    if (blockId >= 0 && nullptr != pMemPtr)
    {
#ifdef _DEBUG
        // cleanup memory buffer
        memset(pMemPtr, 0, static_cast<std::size_t>(requiredMemSize));
#endif

        float* pTmpStorage1 = static_cast<float*>(pMemPtr);
        float* pTmpStorage2 = pTmpStorage1 + frameSize;
        float* pTmpStorage3 = pTmpStorage2 + frameSize;

        // convert RGB to YUV and store only Y (Luma) component normalized to range 0...255 
        Yuv2Luma_Negate (localSrc, pTmpStorage1, sizeX, sizeY, line_pitch, sizeX, static_cast<float>(u8_value_white), 1.f);

        // Compute LUMA - gradient and binarize result (final result stored into pTmpStorage1 with overwrite input contant)
        ImageBW_ComputeGradientBin (pTmpStorage1, pTmpStorage2, pTmpStorage3, sizeX, sizeY);

        // Authomatic thresholding (use Otsu's method)
        ImageBW_AutomaticThreshold (pTmpStorage1, pTmpStorage2, sizeX, sizeY);

        // Final convolution
        ImgConvolution (pTmpStorage2, localSrc, localDst, sizeX, sizeY, line_pitch, sizeX, line_pitch, static_cast<float>(u8_value_white), 1.f);

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


PF_Err PR_ImageStyle_SketchCharcoal_VUYA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
    const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    const PF_Pixel_VUYA_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_VUYA_8u* __restrict>(pfLayer->data);
    PF_Pixel_VUYA_8u*       __restrict localDst = reinterpret_cast<      PF_Pixel_VUYA_8u* __restrict>(output->data);
    void* pMemPtr = nullptr;

    PF_Err err = PF_Err_NONE;

    const A_long sizeX = pfLayer->extent_hint.right  - pfLayer->extent_hint.left;
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_VUYA_8u_size);

    // Allocate memory buffer (triple buffer).
    // We proceed only with LUMA components.
    const A_long frameSize = sizeX * sizeY;
    const A_long tmpMemSize = frameSize * sizeof(float);
    const A_long requiredMemSize = tmpMemSize * 3;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    if (blockId >= 0 && nullptr != pMemPtr)
    {
#ifdef _DEBUG
        // cleanup memory buffer
        memset(pMemPtr, 0, static_cast<std::size_t>(requiredMemSize));
#endif

        float* pTmpStorage1 = static_cast<float*>(pMemPtr);
        float* pTmpStorage2 = pTmpStorage1 + frameSize;
        float* pTmpStorage3 = pTmpStorage2 + frameSize;

        // convert RGB to YUV and store only Y (Luma) component normalized to range 0...255 
        Yuv2Luma_Negate (localSrc, pTmpStorage1, sizeX, sizeY, line_pitch, sizeX, static_cast<float>(u8_value_white), 255.f);

        // Compute LUMA - gradient and binarize result (final result stored into pTmpStorage1 with overwrite input contant)
        ImageBW_ComputeGradientBin (pTmpStorage1, pTmpStorage2, pTmpStorage3, sizeX, sizeY);

        // Authomatic thresholding (use Otsu's method)
        ImageBW_AutomaticThreshold (pTmpStorage1, pTmpStorage2, sizeX, sizeY);

        // Final convolution
        constexpr float rangeDown = 1.f / 255.f;
        ImgConvolution (pTmpStorage2, localSrc, localDst, sizeX, sizeY, line_pitch, sizeX, line_pitch, static_cast<float>(f32_value_white), rangeDown);

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


PF_Err PR_ImageStyle_SketchCharcoal_BGRA_16u
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
    const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    const PF_Pixel_BGRA_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_16u* __restrict>(pfLayer->data);
    PF_Pixel_BGRA_16u*       __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_16u* __restrict>(output->data);
    void* pMemPtr = nullptr;

    PF_Err err = PF_Err_NONE;

    const A_long sizeX = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_16u_size);

    // Allocate memory buffer (triple buffer).
    // We proceed only with LUMA components.
    const A_long frameSize = sizeX * sizeY;
    const A_long tmpMemSize = frameSize * sizeof(float);
    const A_long requiredMemSize = tmpMemSize * 3;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    if (blockId >= 0 && nullptr != pMemPtr)
    {
#ifdef _DEBUG
        // cleanup memory buffer
        memset(pMemPtr, 0, static_cast<std::size_t>(requiredMemSize));
#endif

        float* pTmpStorage1 = static_cast<float*>(pMemPtr);
        float* pTmpStorage2 = pTmpStorage1 + frameSize;
        float* pTmpStorage3 = pTmpStorage2 + frameSize;

        constexpr float white_point = static_cast<float>(u16_value_white);
        constexpr float range_down  = static_cast<float>(u8_value_white) / white_point;
        constexpr float range_up    = white_point / static_cast<float>(u8_value_white);

        // convert RGB to YUV and store only Y (Luma) component normalized to range 0...255 
        Rgb2Luma_Negate(localSrc, pTmpStorage1, BT709, sizeX, sizeY, line_pitch, sizeX, static_cast<float>(u8_value_white), range_down);

        // Compute LUMA - gradient and binarize result (final result stored into pTmpStorage1 with overwrite input contant)
        ImageBW_ComputeGradientBin(pTmpStorage1, pTmpStorage2, pTmpStorage3, sizeX, sizeY);

        // Authomatic thresholding (use Otsu's method)
        ImageBW_AutomaticThreshold(pTmpStorage1, pTmpStorage2, sizeX, sizeY);

        // Final convolution
        ImgConvolution(pTmpStorage2, localSrc, localDst, sizeX, sizeY, line_pitch, sizeX, line_pitch, white_point, range_up);

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


PF_Err PR_ImageStyle_SketchCharcoal_BGRA_32f
(
	PF_InData*   __restrict in_data,
	PF_OutData*  __restrict out_data,
	PF_ParamDef* __restrict params[],
	PF_LayerDef* __restrict output
) noexcept
{
    const PF_LayerDef*      __restrict pfLayer = reinterpret_cast<const PF_LayerDef* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    const PF_Pixel_BGRA_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_BGRA_32f* __restrict>(pfLayer->data);
    PF_Pixel_BGRA_32f*       __restrict localDst = reinterpret_cast<      PF_Pixel_BGRA_32f* __restrict>(output->data);
    void* pMemPtr = nullptr;

    PF_Err err = PF_Err_NONE;

    const A_long sizeX = pfLayer->extent_hint.right - pfLayer->extent_hint.left;
    const A_long sizeY = pfLayer->extent_hint.bottom - pfLayer->extent_hint.top;
    const A_long line_pitch = pfLayer->rowbytes / static_cast<A_long>(PF_Pixel_BGRA_32f_size);

    // Allocate memory buffer (triple buffer).
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

        float* pTmpStorage1 = static_cast<float*>(pMemPtr);
        float* pTmpStorage2 = pTmpStorage1 + frameSize;
        float* pTmpStorage3 = pTmpStorage2 + frameSize;

        constexpr float white_point = static_cast<float>(f32_value_white);
        constexpr float range_down = static_cast<float>(u8_value_white) / white_point;
        constexpr float range_up = white_point / static_cast<float>(u8_value_white);

        // convert RGB to YUV and store only Y (Luma) component normalized to range 0...255 
        Rgb2Luma_Negate (localSrc, pTmpStorage1, BT709, sizeX, sizeY, line_pitch, sizeX, static_cast<float>(u8_value_white), range_down);

        // Compute LUMA - gradient and binarize result (final result stored into pTmpStorage1 with overwrite input contant)
        ImageBW_ComputeGradientBin (pTmpStorage1, pTmpStorage2, pTmpStorage3, sizeX, sizeY);

        // Authomatic thresholding (use Otsu's method)
        ImageBW_AutomaticThreshold (pTmpStorage1, pTmpStorage2, sizeX, sizeY);

        // Final convolution
        ImgConvolution (pTmpStorage2, localSrc, localDst, sizeX, sizeY, line_pitch, sizeX, line_pitch, white_point, range_up);

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
    const PF_EffectWorld*   __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    const PF_Pixel_ARGB_8u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_8u* __restrict>(input->data);
    PF_Pixel_ARGB_8u*       __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_8u* __restrict>(output->data);
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

        // Final convolution
        ImgConvolution (pTmpStorage2, localSrc, localDst, sizeX, sizeY, src_line_pitch, sizeX, dst_line_pitch, static_cast<float>(u8_value_white), 1.f);

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
    const PF_EffectWorld*    __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    const PF_Pixel_ARGB_16u* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_16u* __restrict>(input->data);
    PF_Pixel_ARGB_16u*       __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_16u* __restrict>(output->data);
    void* pMemPtr = nullptr;

    PF_Err err = PF_Err_NONE;

    const A_long sizeY = output->height;
    const A_long sizeX = output->width;
    const A_long src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_16u_size);
    const A_long dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_16u_size);

    // Allocate memory buffer (triple buffer) for padded size.
    // We proceed only with LUMA components.
    const A_long frameSize = sizeX * sizeY;
    const A_long tmpMemSize = frameSize * sizeof(float);
    const A_long requiredMemSize = tmpMemSize * 3;
    int32_t blockId = ::GetMemoryBlock(requiredMemSize, 0, &pMemPtr);

    if (blockId >= 0 && nullptr != pMemPtr)
    {
#ifdef _DEBUG
        // cleanup memory buffer
        memset(pMemPtr, 0, static_cast<std::size_t>(requiredMemSize));
#endif

        float* pTmpStorage1 = static_cast<float*>(pMemPtr);
        float* pTmpStorage2 = pTmpStorage1 + frameSize;
        float* pTmpStorage3 = pTmpStorage2 + frameSize;

        constexpr float white_point = static_cast<float>(u16_value_white);
        constexpr float range_down  = static_cast<float>(u8_value_white) / white_point;
        constexpr float range_up    = white_point / static_cast<float>(u8_value_white);

        // convert RGB to YUV and store only Y (Luma) component normalized to range 0...255 
        Rgb2Luma_Negate (localSrc, pTmpStorage1, BT709, sizeX, sizeY, src_line_pitch, sizeX, static_cast<float>(u8_value_white), range_down);

        // Compute LUMA - gradient and binarize result (final result stored into pTmpStorage1 with overwrite input contant)
        ImageBW_ComputeGradientBin (pTmpStorage1, pTmpStorage2, pTmpStorage3, sizeX, sizeY);

        // Authomatic thresholding (use Otsu's method)
        ImageBW_AutomaticThreshold (pTmpStorage1, pTmpStorage2, sizeX, sizeY);

        // Final convolution
        ImgConvolution (pTmpStorage2, localSrc, localDst, sizeX, sizeY, src_line_pitch, sizeX, dst_line_pitch, white_point, range_up);

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


PF_Err AE_ImageStyle_SketchCharcoal_ARGB_32f
(
    PF_InData*   __restrict in_data,
    PF_OutData*  __restrict out_data,
    PF_ParamDef* __restrict params[],
    PF_LayerDef* __restrict output
) noexcept
{
    const PF_EffectWorld*    __restrict input = reinterpret_cast<const PF_EffectWorld* __restrict>(&params[IMAGE_STYLE_INPUT]->u.ld);
    const PF_Pixel_ARGB_32f* __restrict localSrc = reinterpret_cast<const PF_Pixel_ARGB_32f* __restrict>(input->data);
    PF_Pixel_ARGB_32f*       __restrict localDst = reinterpret_cast<PF_Pixel_ARGB_32f* __restrict>(output->data);
    void* pMemPtr = nullptr;

    PF_Err err = PF_Err_NONE;

    const A_long sizeY = output->height;
    const A_long sizeX = output->width;
    const A_long src_line_pitch = input->rowbytes  / static_cast<A_long>(PF_Pixel_ARGB_32f_size);
    const A_long dst_line_pitch = output->rowbytes / static_cast<A_long>(PF_Pixel_ARGB_32f_size);

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

        float* pTmpStorage1 = static_cast<float*>(pMemPtr);
        float* pTmpStorage2 = pTmpStorage1 + frameSize;
        float* pTmpStorage3 = pTmpStorage2 + frameSize;

        constexpr float white_point = static_cast<float>(f32_value_white);
        constexpr float range_down = static_cast<float>(u8_value_white) / white_point;
        constexpr float range_up = white_point / static_cast<float>(u8_value_white);

        // convert RGB to YUV and store only Y (Luma) component normalized to range 0...255 
        Rgb2Luma_Negate (localSrc, pTmpStorage1, BT709, sizeX, sizeY, src_line_pitch, sizeX, static_cast<float>(u8_value_white), range_down);

        // Compute LUMA - gradient and binarize result (final result stored into pTmpStorage1 with overwrite input contant)
        ImageBW_ComputeGradientBin (pTmpStorage1, pTmpStorage2, pTmpStorage3, sizeX, sizeY);

        // Authomatic thresholding (use Otsu's method)
        ImageBW_AutomaticThreshold (pTmpStorage1, pTmpStorage2, sizeX, sizeY);

        // Final convolution
        ImgConvolution (pTmpStorage2, localSrc, localDst, sizeX, sizeY, src_line_pitch, sizeX, dst_line_pitch, white_point, range_up);

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

