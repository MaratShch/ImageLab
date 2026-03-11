#include "ArtPointillism_GPU.hpp"
#include "ArtPointillismControl.hpp"
#include "ImageLab2GpuObj.hpp"
#include "Common.hpp"

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif


class ArtPontillismGPU final : public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(ArtPontillismGPU);
	CLASS_NON_MOVABLE(ArtPontillismGPU);

    ArtPontillismGPU() = default;
	~ArtPontillismGPU() = default;

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
        CACHE_ALIGN PrParam algoParams[8];

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

        algoParams[0] = GetParam (UnderlyingType(ArtPointillismControls::ART_POINTILLISM_PAINTER_STYLE), clipTime);
        algoParams[1] = GetParam (UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_DOT_DENCITY), clipTime);
        algoParams[2] = GetParam (UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_DOT_SIZE), clipTime);
        algoParams[3] = GetParam (UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_EDGE_SENSITIVITY), clipTime);
        algoParams[4] = GetParam (UnderlyingType(ArtPointillismControls::ART_POINTILLISM_SLIDER_COLOR_VIBRANCE), clipTime);
        algoParams[5] = GetParam (UnderlyingType(ArtPointillismControls::ART_POINTILLISM_BACKGROUND_ART), clipTime);
        algoParams[6] = GetParam (UnderlyingType(ArtPointillismControls::ART_POINTILLISM_OPACITY), clipTime);
        algoParams[7] = GetParam (UnderlyingType(ArtPointillismControls::ART_POINTILLISM_RANDOM_SEED), clipTime);

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

                CACHE_ALIGN PontillismControls algoGpuParams;

                algoGpuParams.PainterStyle = static_cast<ArtPointillismPainter>(algoParams[0].mInt32);
                algoGpuParams.DotDencity = algoParams[1].mInt32;
                algoGpuParams.DotSize = algoParams[2].mInt32;
                algoGpuParams.EdgeSensitivity = algoParams[3].mInt32;
                algoGpuParams.Vibrancy = algoParams[4].mInt32;
                algoGpuParams.Background = static_cast<BackgroundArt>(algoParams[5].mInt32);
                algoGpuParams.Opacity = algoParams[6].mInt32;
                algoGpuParams.RandomSeed = algoParams[7].mInt32;

                constexpr cudaStream_t stream{ 0 };

                // Launch CUDA kernel
                ArtPointillism_CUDA(inBuffer, outBuffer, srcPitch, dstPitch, width, height, &algoGpuParams, frameCounter, stream);

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

        const size_t size_densityMap = needed_pixels * sizeof(float2); // DensityInfo
        const size_t size_jfaPing = needed_pixels * sizeof(int4);   // JFACell
        const size_t size_jfaPong = needed_pixels * sizeof(int4);   // JFACell

        size_t total_dynamic_memory = size_densityMap + size_jfaPing + size_jfaPong;

        // 2. Constant-Size Calculation
        constexpr size_t max_gpu_dots = 1000000ull;

        constexpr size_t size_dots = max_gpu_dots * 16; // GPUDot (aligned to 16)
        constexpr size_t size_dotColors = max_gpu_dots * sizeof(float4); // DotColorAccumulator
        constexpr size_t size_dotInfo = max_gpu_dots * 16; // DotRenderInfo (aligned to 16)

        constexpr size_t size_counters = 32 * sizeof(int);
        constexpr size_t size_palette = 32 * sizeof(float4);

        size_t total_constant_memory = size_dots + size_dotColors + size_dotInfo + size_counters + size_palette;

        // 3. Return Total VRAM Required (in bytes)
        return total_dynamic_memory + total_constant_memory;
    }

    bool IsVramSufficientForRender (int width, int height) noexcept
    {
        size_t free_vram, total_vram;
        std::tie(free_vram, total_vram) = GetGpuMemoryInfo_CUDA();

        const size_t required_vram = CalculateRequiredGpuMemory (width, height);
        return (free_vram < (required_vram + GetSafeMargin_CUDA()) ? false : true);
    }
};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ArtPontillismGPU>);