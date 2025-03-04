#include "AuthomaticWhiteBalanceGPU.hpp"
#include "AlgCommonEnums.hpp"
#include "ImageLab2GpuObj.hpp"
#include "Common.hpp"
#include "CommonAdobeAE.hpp"
#include "ColorTransformMatrix.hpp"

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif


class AuthomaticWhiteBalanceGPU final : public CImageLab2GpuObj
{
public:
    CLASS_NON_COPYABLE(AuthomaticWhiteBalanceGPU);
    CLASS_NON_MOVABLE(AuthomaticWhiteBalanceGPU);

    AuthomaticWhiteBalanceGPU() = default;
    virtual ~AuthomaticWhiteBalanceGPU() = default;

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
        PrTime const clipTime{ inRenderParams->inClipTime };
        const PrParam param1 = GetParam(AWB_ILLUMINATE_POPUP,  clipTime);
        const PrParam param2 = GetParam(AWB_CHROMATIC_POPUP,   clipTime);
        const PrParam param3 = GetParam(AWB_COLOR_SPACE_POPUP, clipTime);
        const PrParam param4 = GetParam(AWB_THRESHOLD_SLIDER,  clipTime);
        const PrParam param5 = GetParam(AWB_ITERATIONS_SLIDER, clipTime);

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
        const int destPitch = destRowBytes / gpuBytesPerPixel;

        mGPUDeviceSuite->GetGPUPPixData(*inFrames, &srcFrameData);
        mPPixSuite->GetRowBytes(*inFrames, &srcRowBytes);
        const int srcPitch = srcRowBytes / gpuBytesPerPixel;

        // start CUDA
        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
        {
            // CUDA device pointers
            inBuffer = reinterpret_cast<float*>(srcFrameData);
            outBuffer = reinterpret_cast<float*>(destFrameData);

            const eILLUMINATE illuminant         = control_param_illuminant (param1);
            const eChromaticAdaptation chromatic = control_param_chromatic_adaptation (param2);
            const eCOLOR_SPACE color_space       = control_param_color_space (param3);
            const float gray_threshold           = control_param_gray_threshold (param4);
            const unsigned int iter_cnt          = control_param_iteration_count(param5);

            // Launch CUDA kernel
            AuthomaticWhiteBalance_CUDA (inBuffer, outBuffer, destPitch, srcPitch, is16f, width, height, illuminant, chromatic, color_space, gray_threshold, iter_cnt);

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
        return static_cast<const eILLUMINATE>(CLAMP_VALUE(param.mInt32, static_cast<csSDK_int32>(DAYLIGHT), static_cast<csSDK_int32>(COOL_WHITE_FLUORESCENT)));
    }

    const eChromaticAdaptation control_param_chromatic_adaptation (const PrParam& param) noexcept
    {
        return static_cast<const eChromaticAdaptation>(CLAMP_VALUE(param.mInt32, static_cast<csSDK_int32>(CHROMATIC_CAT02), static_cast<csSDK_int32>(CHROMATIC_CMCCAT2000)));
    }

    const eCOLOR_SPACE control_param_color_space (const PrParam& param) noexcept
    {
        return static_cast<const eCOLOR_SPACE>(CLAMP_VALUE(param.mInt32, static_cast<csSDK_int32>(BT601), static_cast<csSDK_int32>(SMPTE240M)));
    }

    const float control_param_gray_threshold (const PrParam& param) noexcept
    {
        return static_cast<const float>(CLAMP_VALUE(param.mInt32, 10, 90))/100.f;
    }

    const unsigned int control_param_iteration_count(const PrParam& param) noexcept
    {
        return static_cast<const unsigned int>(CLAMP_VALUE(param.mInt32, 1, 16));
    }

};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<AuthomaticWhiteBalanceGPU>);