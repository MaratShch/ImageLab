#include "AuthomaticWhiteBalanceGPU.hpp"
#include "ImageLab2GpuObj.hpp"
#include "Common.hpp"
#include "CommonAdobeAE.hpp"

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

        csSDK_uint32 gray_threshold = 0u;
        csSDK_uint32 observer_idx   = 0u;
        csSDK_uint32 illuminant_idx = 0u;
        csSDK_uint32 iter_cnt       = 0u;

        // read control setting
        PrTime const clipTime{ inRenderParams->inClipTime };

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

            // Launch CUDA kernel
            AuthomaticWhiteBalance_CUDA(inBuffer, outBuffer, destPitch, srcPitch, is16f, width, height, gray_threshold, observer_idx, illuminant_idx, iter_cnt);


            cudaError_t cudaErrCode = cudaErrorUnknown;
            if (cudaSuccess != (cudaErrCode = cudaPeekAtLastError()))
            {
                return suiteError_Fail;
            }
        }

        return suiteError_NoError;
    }

private:
};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<AuthomaticWhiteBalanceGPU>);