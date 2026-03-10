#include "ImageLabDenoise_GPU.hpp"
#include "ImageLab2GpuObj.hpp"
#include "CompileTimeUtils.hpp"
#include "ImageLabDenoiseEnum.hpp"
#include "Common.hpp"



#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif


class ImageLabDenoiseGPU final : public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(ImageLabDenoiseGPU);
	CLASS_NON_MOVABLE(ImageLabDenoiseGPU);

    ImageLabDenoiseGPU() = default;
	~ImageLabDenoiseGPU() = default;

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

        csSDK_int32 destRowBytes, srcRowBytes;
        cudaError_t cudaErrCode = cudaErrorUnknown;

        prRect bounds{};
        mPPixSuite->GetBounds(*outFrame, &bounds);
        const int width  = bounds.right - bounds.left;
        const int height = bounds.bottom - bounds.top;

        if (true == IsVramSufficientForRender (width, height))
        {
            const PrTime clipTime = inRenderParams->inClipTime;
            const PrTime renderTick = inRenderParams->inRenderTicksPerFrame;
            const int frameCounter = (renderTick > 0 ? static_cast<int>(clipTime / renderTick) : 0);

            // read control setting
            algoParams[0] = GetParam(UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_ACC_SANDARD), clipTime);
            algoParams[1] = GetParam(UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_AMOUNT), clipTime);
            algoParams[2] = GetParam(UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_LUMA_STRENGTH), clipTime);
            algoParams[3] = GetParam(UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_CHROMA_STRENGTH), clipTime);
            algoParams[4] = GetParam(UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_DETAILS_PRESERVATION), clipTime);
            algoParams[5] = GetParam(UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_COARSE_NOISE), clipTime);

#ifdef _DEBUG
            const csSDK_int32 instanceCnt = TotalInstances();
#endif

            mGPUDeviceSuite->GetGPUPPixData(*outFrame, &frameData);

            PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
            mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
            const csSDK_int32 gpuBytesPerPixel = GetGPUBytesPerPixel(pixelFormat);


            mGPUDeviceSuite->GetGPUPPixData(*outFrame, &destFrameData);
            mPPixSuite->GetRowBytes(*outFrame, &destRowBytes);
            const csSDK_int32 dstPitch = destRowBytes / gpuBytesPerPixel;

            mGPUDeviceSuite->GetGPUPPixData(*inFrames, &srcFrameData);
            mPPixSuite->GetRowBytes(*inFrames, &srcRowBytes);
            const csSDK_int32 srcPitch = srcRowBytes / gpuBytesPerPixel;

            // start CUDA
            if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
            {
                // CUDA device pointers
                const float* inBuffer  = reinterpret_cast<const float*>(srcFrameData);
                float* outBuffer = reinterpret_cast<float*>(destFrameData);

                CACHE_ALIGN AlgoControls algoGpuParams;

                algoGpuParams.accuracy = static_cast<ProcAccuracy>(algoParams[0].mInt32);
                algoGpuParams.master_denoise_amount = static_cast<float>(algoParams[1].mFloat64);
                algoGpuParams.luma_strength = static_cast<float>(algoParams[2].mFloat64);
                algoGpuParams.chroma_strength = static_cast<float>(algoParams[3].mFloat64);
                algoGpuParams.fine_detail_preservation = static_cast<float>(algoParams[4].mFloat64);
                algoGpuParams.coarse_noise_reduction = static_cast<float>(algoParams[5].mFloat64);

                constexpr cudaStream_t stream{ 0 };

                // Launch CUDA kernel
                ImageLabDenoise_CUDA (inBuffer, outBuffer, srcPitch, dstPitch, width, height, &algoGpuParams, frameCounter, stream);

                if (cudaSuccess != (cudaErrCode = cudaPeekAtLastError()))
                {
                    return suiteError_Fail;
                }
            }
            return suiteError_NoError;
        }
        return suiteError_OutOfMemory;
	}

    private:
        size_t CalculateNoiseClinicVramRequirement (int width, int height) noexcept
        {
            // Ensure 256-byte alignment for fast coalesced global memory access
            auto Align256 = [](size_t size) noexcept -> const size_t { return (size + 255) & ~255; };

            const size_t num_pixels = static_cast<size_t>(width * height);

            // 1. Host/Device I/O Buffers
            const size_t io_buffer_size = Align256(num_pixels * 4 * sizeof(float));
            const size_t total_io = io_buffer_size * 2; // inBuffer + outBuffer

            // 2. Planar Y, U, V Work Buffers
            const size_t channel_size = Align256(num_pixels * sizeof(float));
            const size_t total_planar = channel_size * 3; // Y, U, V

             // 3. Multiscale Pyramid Buffers (Mosaics & Differences)
             const size_t active_mosaics = total_planar * 2;

            // The difference images collectively take up exactly 1x the image size across all scales.
            const size_t total_differences = total_planar;

            // 4. NL-Bayes Aggregation Buffers (The heaviest part)
            const size_t aggregation_buffers = (total_planar * 2);

            // 5. Noise Estimation Data Structures (DCT matrices, histograms, etc.)
            constexpr size_t noise_estimation_padding = 10 * 1024 * 1024;

            // Total Calculation
            const size_t total_required_bytes = total_io + total_planar +  active_mosaics + total_differences + aggregation_buffers + noise_estimation_padding;

            return total_required_bytes;
        }

        bool IsVramSufficientForRender (int width, int height) noexcept
        {
            size_t free_vram, total_vram;
            std::tie(free_vram, total_vram) = GetGpuMemoryInfo();

            const size_t required_vram = CalculateNoiseClinicVramRequirement (width, height);

            // Safety margin: Leave 64 MB free for driver overhead and local kernel memory
            constexpr size_t safety_margin = 64 * 1024 * 1024;

            return (free_vram < (required_vram + safety_margin) ? false : true);
         }
};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ImageLabDenoiseGPU>);
