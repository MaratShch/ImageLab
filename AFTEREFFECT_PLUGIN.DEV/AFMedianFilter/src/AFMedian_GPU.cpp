#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "FastAriphmetics.hpp"
#include "ImageLab2GpuObj.hpp"
#include "AFMedian_GPU.hpp"
#include "AFMedianFilterEnum.hpp"

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif


class AFMedianGPU final : public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(AFMedianGPU);
	CLASS_NON_MOVABLE(AFMedianGPU);

    AFMedianGPU() = default;
	~AFMedianGPU() = default;

	prSuiteError InitializeCUDA(void)
	{
		return suiteError_NoError;
	}

	virtual prSuiteError Initialize (PrGPUFilterInstance* ioInstanceData)
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
        CACHE_ALIGN PrParam algoGpuParams[4];

        void* frameData     = nullptr;
		void* destFrameData = nullptr;
		void* srcFrameData  = nullptr;
        csSDK_int32 destRowBytes, srcRowBytes;
        cudaError_t cudaErrCode = cudaErrorUnknown;

        prRect bounds{};
        mPPixSuite->GetBounds(*outFrame, &bounds);
        const int32_t width  = bounds.right - bounds.left;
        const int32_t height = bounds.bottom - bounds.top;

		// read control setting
		const PrTime clipTime = inRenderParams->inClipTime;
        const PrTime renderTick = inRenderParams->inRenderTicksPerFrame;
        const int32_t frameCounter = (renderTick > 0 ? static_cast<int32_t>(clipTime / renderTick) : 0);

        algoGpuParams[0] = GetParam(UnderlyingType(AFMF::eIMAGE_AFMEDIAN_OUTPUT_TYPE), clipTime);
        algoGpuParams[1] = GetParam(UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_RADIUS), clipTime);
        algoGpuParams[2] = GetParam(UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_TOLERANCE), clipTime);
        algoGpuParams[3] = GetParam(UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_ITERATIONS), clipTime);

#ifdef _DEBUG
        const csSDK_int32 instanceCnt = TotalInstances();
#endif

        mGPUDeviceSuite->GetGPUPPixData(*outFrame, &frameData);

        PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
        mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
        const csSDK_int32 gpuBytesPerPixel = GetGPUBytesPerPixel(pixelFormat);
        const int32_t is16f = (ImageLabGpuPixel16f == pixelFormat) ? 1 : 0;

        mGPUDeviceSuite->GetGPUPPixData(*outFrame, &destFrameData);
        mPPixSuite->GetRowBytes(*outFrame, &destRowBytes);
        const csSDK_int32 dstPitch = destRowBytes / gpuBytesPerPixel;

        mGPUDeviceSuite->GetGPUPPixData(*inFrames, &srcFrameData);
        mPPixSuite->GetRowBytes(*inFrames, &srcRowBytes);
        const csSDK_int32 srcPitch = srcRowBytes / gpuBytesPerPixel;

		// start CUDA
		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
		{
            if (true == IsVramSufficientForRender (width, height))
            {
                CACHE_ALIGN AlgoControls algoControls;

                algoControls.outputType = static_cast<AFMF_Output>(algoGpuParams[0].mInt32);
                algoControls.radius     = popup2value (static_cast<int32_t>    (algoGpuParams[1].mInt32));
                algoControls.tolerance  = static_cast<float>      (algoGpuParams[2].mFloat64);
                algoControls.iterations = popup2value (static_cast<int32_t>    (algoGpuParams[3].mInt32));

                algoControls.Sanitize();

                // CUDA device pointers
                const float* inBuffer  = reinterpret_cast<const float*>(srcFrameData);
                      float* outBuffer = reinterpret_cast<float*>(destFrameData);

                constexpr cudaStream_t stream{ 0 };

                // Launch CUDA kernel
                if (is16f)
                    ImageLabAFMF16_CUDA (inBuffer, outBuffer, srcPitch, dstPitch, width, height, &algoControls, frameCounter, stream);
                else
                    ImageLabAFMF32_CUDA (inBuffer, outBuffer, srcPitch, dstPitch, width, height, &algoControls, frameCounter, stream);

                return (cudaSuccess == (cudaErrCode = cudaPeekAtLastError()) ? suiteError_NoError : suiteError_Fail);
            }
            return suiteError_OutOfMemory;
		}

		return suiteError_InvalidCall;
	}

private:
    size_t CalculateGpuMemory (int width, int height)
    {
        size_t total_vram_bytes = 0;
        return total_vram_bytes;
    }

    bool IsVramSufficientForRender (int width, int height) noexcept
    {
        size_t free_vram, total_vram;
        std::tie(free_vram, total_vram) = GetGpuMemoryInfo_CUDA();

        const size_t required_vram = CalculateGpuMemory(width, height);
        return (free_vram < (required_vram + GetSafeMargin_CUDA()) ? false : true);
    }
};

//DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<AFMedianGPU>);