#include "Common.hpp"
#include "CompileTimeUtils.hpp"
#include "ArtMosaic_GPU.hpp"
#include "ArtMosaicEnum.hpp"
#include "ImageLab2GpuObj.hpp"
#include "FastAriphmetics.hpp"

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
            if (true == IsVramSufficientForRender (width, height, cellsNumber))
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
    size_t CalculateSlicGpuMemory (int width, int height, int requested_k)
    {
        const int totalPixels = width * height;

        // Safety clamp (cannot have more clusters than pixels)
        const int safe_k = (requested_k > totalPixels) ? totalPixels : requested_k;

        // Calculate grid dimensions to account for the mapping array
        float superPixInitVal = static_cast<float>(totalPixels) / static_cast<float>(safe_k);
        const int S = FastCompute::Max(static_cast<int>(FastCompute::Sqrt(superPixInitVal)), 1);

        const int nX = width  / S;
        const int nY = height / S;
        const int max_grid_k = nX * nY;

        // ----------------------------------------------------
        // 1. PER-PIXEL BUFFERS (The bulk of the memory)
        // ----------------------------------------------------
        // d_r, d_g, d_b, d_distances, d_labels (Main Arena)
        // d_cc, d_sizes, d_new_labels (Union-Find Connectivity)
        // Total: 8 buffers * 4 bytes (sizeof float/int) = 32 bytes per pixel
        size_t pixel_memory = static_cast<size_t>(totalPixels) * 32;

        // ----------------------------------------------------
        // 2. PER-CLUSTER BUFFERS
        // ----------------------------------------------------
        // d_cluster (x, y, r, g, b) -> 5 buffers
        // d_acc (x, y, r, g, b, count) -> 6 buffers
        // Total: 11 buffers * 4 bytes = 44 bytes per cluster
        size_t cluster_memory = static_cast<size_t>(safe_k * 44);

        // ----------------------------------------------------
        // 3. GRID MAPPING & MISC BUFFERS
        // ----------------------------------------------------
        // d_grid_to_k mapping array (max_grid_k * 4 bytes)
        size_t mapping_memory = static_cast<size_t>(max_grid_k * 4);

        // d_actualK scalar (4 bytes) + padding/alignment safety margin (1024 bytes)
        constexpr size_t misc_memory = 4 + 1024;

        // Calculate Grand Total
        size_t total_vram_bytes = pixel_memory + cluster_memory + mapping_memory + misc_memory;

        return total_vram_bytes;
    }

    bool IsVramSufficientForRender (int width, int height, int K = 1000) noexcept
    {
        size_t free_vram, total_vram;
        std::tie(free_vram, total_vram) = GetGpuMemoryInfo_CUDA();

        const size_t required_vram = CalculateSlicGpuMemory(width, height, K);
        return (free_vram < (required_vram + GetSafeMargin_CUDA()) ? false : true);
    }
};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ArtMosaicGPU>);