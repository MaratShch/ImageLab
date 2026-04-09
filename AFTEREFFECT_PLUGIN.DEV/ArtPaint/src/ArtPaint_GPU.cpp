#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "ArtPaint_GPU.hpp"
#include "PaintAlgoContols.hpp"
#include "ArtPaintEnums.hpp"
#include "ImageLab2GpuObj.hpp"


#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif


class ArtPaintGPU final : public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(ArtPaintGPU);
	CLASS_NON_MOVABLE(ArtPaintGPU);

    ArtPaintGPU() = default;
	~ArtPaintGPU() = default;

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
        CACHE_ALIGN PrParam algoParams[6];

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
        const int frameCounter = (renderTick > 0 ? static_cast<int>(clipTime / renderTick) : 0);

        algoParams[0] = GetParam (UnderlyingType(ArtPaintControls::ART_PAINT_RENDER_QUALITY), clipTime);
        algoParams[1] = GetParam (UnderlyingType(ArtPaintControls::ART_PAINT_STYLE), clipTime);
        algoParams[2] = GetParam (UnderlyingType(ArtPaintControls::ART_PAINT_BRUSH_WIDTH), clipTime);
        algoParams[3] = GetParam (UnderlyingType(ArtPaintControls::ART_PAINT_BRUSH_LENGTH), clipTime);
        algoParams[4] = GetParam (UnderlyingType(ArtPaintControls::ART_PAINT_STROKE_CURVATIVE), clipTime);
        algoParams[5] = GetParam (UnderlyingType(ArtPaintControls::ART_PAINT_STROKE_SPREADING), clipTime);

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
            if (true == IsVramSufficientForRender(width, height))
            {
                // CUDA device pointers
                inBuffer = reinterpret_cast<float*>(srcFrameData);
                outBuffer = reinterpret_cast<float*>(destFrameData);

                CACHE_ALIGN AlgoControls algoGpuParams;

                algoGpuParams.quality = static_cast<RenderQuality>(algoParams[0].mInt32);
                algoGpuParams.bias    = static_cast<StrokeBias>(algoParams[1].mInt32);
                algoGpuParams.sigma   = static_cast<float>(algoParams[2].mFloat64);
                algoGpuParams.angular = static_cast<float>(algoParams[3].mFloat64);
                algoGpuParams.angle   = static_cast<float>(algoParams[4].mFloat64);
                algoGpuParams.iter    = static_cast<int32_t>(algoParams[5].mInt32);

                constexpr cudaStream_t stream{ 0 };

                // Launch CUDA kernel
                ArtPaint_CUDA(inBuffer, outBuffer, srcPitch, dstPitch, width, height, &algoGpuParams, frameCounter, stream);

                return (cudaSuccess == (cudaErrCode = cudaPeekAtLastError()) ? suiteError_NoError : suiteError_Fail);
            }
            return suiteError_OutOfMemory;
		}

		return suiteError_InvalidCall;
	}

private:
    size_t CalculateRequiredGpuMemory (int width, int height) noexcept
    {
        // 1. Image-Dependent Calculation
        const size_t needed_pixels = static_cast<size_t>(width * height);

        return needed_pixels;
    }

    bool IsVramSufficientForRender (int width, int height) noexcept
    {
        size_t free_vram, total_vram;
        std::tie(free_vram, total_vram) = GetGpuMemoryInfo_CUDA();

        const size_t required_vram = CalculateRequiredGpuMemory (width, height);
        return (free_vram < (required_vram + GetSafeMargin_CUDA()) ? false : true);
    }
};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ArtPaintGPU>);