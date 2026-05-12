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
        CACHE_ALIGN PrParam algoGpuParams[5];
        CACHE_ALIGN AlgoControls algoControls;

        void* frameData     = nullptr;
		void* destFrameData = nullptr;
		void* srcFrameData  = nullptr;
		float* inBuffer     = nullptr;
		float* outBuffer    = nullptr;

        csSDK_int32 destRowBytes, srcRowBytes;
        cudaError_t cudaErrCode = cudaErrorUnknown;

		// read control setting
		const PrTime clipTime = inRenderParams->inClipTime;
        const PrTime renderTick = inRenderParams->inRenderTicksPerFrame;
        const int32_t frameCounter = (renderTick > 0 ? static_cast<int32_t>(clipTime / renderTick) : 0);

        algoGpuParams[0] = GetParam(UnderlyingType(AFMF::eIMAGE_AFMEDIAN_INPUT_TYPE), clipTime);
        algoGpuParams[1] = GetParam(UnderlyingType(AFMF::eIMAGE_AFMEDIAN_OUTPUT_TYPE), clipTime);
        algoGpuParams[2] = GetParam(UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_RADIUS), clipTime);
        algoGpuParams[3] = GetParam(UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_TOLERANCE), clipTime);
        algoGpuParams[4] = GetParam(UnderlyingType(AFMF::eIMAGE_AFMEDIAN_PARAM_ITERATIONS), clipTime);

#ifdef _DEBUG
		const csSDK_int32 instanceCnt = TotalInstances();
#endif

		mGPUDeviceSuite->GetGPUPPixData (*outFrame, &frameData);

		PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
		mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
        const int32_t is16f = (ImageLabGpuPixel16f == pixelFormat) ? 1 : 0;
        const csSDK_int32 gpuBytesPerPixel = GetGPUBytesPerPixel(pixelFormat);

		prRect bounds{};
		mPPixSuite->GetBounds (*outFrame, &bounds);
		const int width  = bounds.right  - bounds.left;
		const int height = bounds.bottom - bounds.top;

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
                algoControls.inputType  = static_cast<AFMF_Input> (algoGpuParams[0].mInt32);
                algoControls.outputType = static_cast<AFMF_Output>(algoGpuParams[1].mInt32);
                algoControls.radius     = static_cast<int32_t>    (algoGpuParams[2].mInt32);
                algoControls.tolerance  = static_cast<float>      (algoGpuParams[3].mFloat64);
                algoControls.iterations = static_cast<int32_t>    (algoGpuParams[4].mInt32);

                // CUDA device pointers
                inBuffer = reinterpret_cast<float*>(srcFrameData);
                outBuffer = reinterpret_cast<float*>(destFrameData);

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

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<AFMedianGPU>);