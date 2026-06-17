#include "AuthomaticWhiteBalance2GPU.hpp"
#include "AlgorithmEnums.hpp"
#include "ImageLab2GpuObj.hpp"
#include "Common.hpp"
#include "CommonAdobeAE.hpp"
#include "ColorTransformMatrix.hpp"

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif


class AuthomaticWhiteBalance2GPU final : public CImageLab2GpuObj
{
public:
    CLASS_NON_COPYABLE(AuthomaticWhiteBalance2GPU);
    CLASS_NON_MOVABLE(AuthomaticWhiteBalance2GPU);

    AuthomaticWhiteBalance2GPU() = default;
    virtual ~AuthomaticWhiteBalance2GPU() = default;

    prSuiteError InitializeCUDA(void)
    {
        return suiteError_NoError;
    }

    virtual prSuiteError Initialize(PrGPUFilterInstance* ioInstanceData)
    {
        CImageLab2GpuObj::Initialize(ioInstanceData);
        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
            return InitializeCUDA();
        return suiteError_Fail;
    }

    prSuiteError Render
    (
        const PrGPUFilterRenderParams* inRenderParams,
        const PPixHand* inFrames,
        csSDK_size_t inFrameCount,
        PPixHand* outFrame
    ) noexcept
    {
        void* frameData = nullptr;
        void* destFrameData = nullptr;
        void* srcFrameData = nullptr;
        float* inBuffer = nullptr;
        float* outBuffer = nullptr;
        csSDK_int32 destRowBytes = 0;
        csSDK_int32 srcRowBytes = 0;

        // read control setting
        CACHE_ALIGN PrParam param[6]{};

        PrTime const clipTime{ inRenderParams->inClipTime };
        param[0] = GetParam(UnderlyingType(eImageLab2AWB_Controls::AWB2_COLOR_SPACE_POPUP),     clipTime);
        param[1] = GetParam(UnderlyingType(eImageLab2AWB_Controls::AWB2_CHROMATIC_POPUP),       clipTime);
        param[2] = GetParam(UnderlyingType(eImageLab2AWB_Controls::AWB2_ILLUMINATE_POPUP),      clipTime);
        param[3] = GetParam(UnderlyingType(eImageLab2AWB_Controls::AWB2_EXTERME_PIXELS),        clipTime);
        param[4] = GetParam(UnderlyingType(eImageLab2AWB_Controls::AWB2_SATRURATION_THRESHOLD), clipTime);
        param[5] = GetParam(UnderlyingType(eImageLab2AWB_Controls::AWB2_BLACK_LEVEL_THRESHOLD), clipTime);

#ifdef _DEBUG
        const csSDK_int32 instanceCnt = TotalInstances();
#endif

        mGPUDeviceSuite->GetGPUPPixData(*outFrame, &frameData);

        PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
        mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
        const int gpuBytesPerPixel = GetGPUBytesPerPixel(pixelFormat);
        const int is16f = (ImageLabGpuPixel16f == pixelFormat) ? 1 : 0;

        prRect bounds{};
        mPPixSuite->GetBounds(*outFrame, &bounds);
        const int width = bounds.right - bounds.left;
        const int height = bounds.bottom - bounds.top;

        mGPUDeviceSuite->GetGPUPPixData(*outFrame, &destFrameData);
        mPPixSuite->GetRowBytes(*outFrame, &destRowBytes);
        const int dstPitch = destRowBytes / gpuBytesPerPixel;

        mGPUDeviceSuite->GetGPUPPixData(*inFrames, &srcFrameData);
        mPPixSuite->GetRowBytes(*inFrames, &srcRowBytes);
        const int srcPitch = srcRowBytes / gpuBytesPerPixel;

        // start CUDA
        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
        {
            // CUDA device pointers
            inBuffer = reinterpret_cast<float*>(srcFrameData);
            outBuffer = reinterpret_cast<float*>(destFrameData);

            AlgoControls algoControls{};

            algoControls.colorSpace             = control_param_color_space(param[0]);
            algoControls.chromatic              = control_param_chromatic_adaptation(param[1]);
            algoControls.illuminate             = control_param_illuminant (param[2]);
            algoControls.percentExtremePixels   = control_param_extreme_pixels(param[3]);
            algoControls.saturationThreshold    = control_param_saturation_threshold(param[4]);
            algoControls.blackLevelThreshold    = control_param_black_level_threshold(param[5]);

            const cudaStream_t stream{ 0 };
            constexpr A_long frameCount = 0; // parameter not used yet

            if (is16f)
            {
                ImageLabPCA16_CUDA
                (
                    inBuffer,
                    outBuffer,
                    srcPitch,
                    dstPitch,
                    width,
                    height,
                    &algoControls,
                    frameCount,
                    stream
                );
            }
            else
            {
                ImageLabPCA32_CUDA
                (
                    inBuffer,
                    outBuffer,
                    srcPitch,
                    dstPitch,
                    width,
                    height,
                    &algoControls,
                    frameCount,
                    stream
                );
            }

            cudaError_t cudaErrCode = cudaErrorUnknown;
            if (cudaSuccess != (cudaErrCode = cudaPeekAtLastError()))
            {
                return suiteError_Fail;
            }
        }

        return suiteError_NoError;
    }

private:

    const eILLUMINATE control_param_illuminant (const PrParam& param) noexcept
    {
        return static_cast<const eILLUMINATE>(CLAMP_VALUE(param.mInt32, static_cast<csSDK_int32>(UnderlyingType(eILLUMINATE::DAYLIGHT)),
            static_cast<csSDK_int32>(UnderlyingType(eILLUMINATE::COOL_WHITE_FLUORESCENT))));
    }

    const eChromaticAdaptation control_param_chromatic_adaptation (const PrParam& param) noexcept
    {
        return static_cast<const eChromaticAdaptation>(CLAMP_VALUE(param.mInt32, static_cast<csSDK_int32>(UnderlyingType(eChromaticAdaptation::CHROMATIC_CAT02)),
            static_cast<csSDK_int32>(UnderlyingType(eChromaticAdaptation::CHROMATIC_CMCCAT2000))));
    }

    const eCOLOR_SPACE control_param_color_space (const PrParam& param) noexcept
    {
        return static_cast<const eCOLOR_SPACE>(CLAMP_VALUE(param.mInt32, static_cast<csSDK_int32>(BT601), static_cast<csSDK_int32>(SMPTE240M)));
    }

    const float control_param_extreme_pixels (const PrParam& param) noexcept
    {
        return static_cast<const float>(CLAMP_VALUE(param.mFloat64, 1.0, 10.0));
    }

    const unsigned int control_param_saturation_threshold (const PrParam& param) noexcept
    {
        return static_cast<const float>(CLAMP_VALUE(param.mFloat64, 0.80, 1.0));
    }

    const unsigned int control_param_black_level_threshold (const PrParam& param) noexcept
    {
        return static_cast<const float>(CLAMP_VALUE(param.mFloat64, 0.0, 0.10));
    }

};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<AuthomaticWhiteBalance2GPU>);