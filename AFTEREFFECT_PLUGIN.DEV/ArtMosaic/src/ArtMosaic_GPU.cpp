#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "ArtMosaic_GPU.hpp"
#include "ArtMosaicEnum.hpp"
#include "ImageLab2GpuObj.hpp"

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif


class ArtMosaicGPU final : public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(ArtMosaicGPU);
	CLASS_NON_MOVABLE(ArtMosaicGPU);

    ArtMosaicGPU() = default;
	~ArtMosaicGPU() = default;

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
        CACHE_ALIGN PrParam cellsNumberParam;

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

        cellsNumberParam = GetParam (UnderlyingType(eART_MOSAIC_ITEMS::eIMAGE_ART_MOSAIC_CELLS_SLIDER), clipTime);
        const int32_t cellsNumber = cellsNumberParam.mInt32;

#ifdef _DEBUG
		const csSDK_int32 instanceCnt = TotalInstances();
#endif

		mGPUDeviceSuite->GetGPUPPixData (*outFrame, &frameData);

		PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
		mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
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
            if (true == IsVramSufficientForRender(width, height, cellsNumber))
            {
                // CUDA device pointers
                inBuffer = reinterpret_cast<float*>(srcFrameData);
                outBuffer = reinterpret_cast<float*>(destFrameData);

                constexpr cudaStream_t stream{ 0 };

                // Launch CUDA kernel
                ImageLabMosaic_CUDA (inBuffer, outBuffer, srcPitch, dstPitch, width, height, cellsNumber, frameCounter, stream);

                return (cudaSuccess == (cudaErrCode = cudaPeekAtLastError()) ? suiteError_NoError : suiteError_Fail);
            }
            return suiteError_OutOfMemory;
		}

		return suiteError_InvalidCall;
	}

private:
    size_t CalculateRequiredGpuMemory (int width, int height, int K) noexcept
    {
        // 1. Image-Dependent Calculation
        const size_t needed_pixels = static_cast<size_t>(width * height);

        // 3. Return Total VRAM Required (in bytes)
        return 0;
    }

    bool IsVramSufficientForRender (int width, int height, int K = 1000) noexcept
    {
        size_t free_vram, total_vram;
        std::tie(free_vram, total_vram) = GetGpuMemoryInfo_CUDA();

        const size_t required_vram = CalculateRequiredGpuMemory (width, height, K);
        return (free_vram < (required_vram + GetSafeMargin_CUDA()) ? false : true);
    }
};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ArtMosaicGPU>);