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
		float* inBuffer     = nullptr;
		float* outBuffer    = nullptr;

        csSDK_int32 destRowBytes, srcRowBytes;
        cudaError_t cudaErrCode = cudaErrorUnknown;

		// read control setting
		const PrTime clipTime = inRenderParams->inClipTime;
        const PrTime renderTick = inRenderParams->inRenderTicksPerFrame;
        const int frameCounter = (renderTick > 0 ? static_cast<int>(clipTime / renderTick) : 0);

        algoParams[0] = GetParam (UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_ACC_SANDARD), clipTime);
        algoParams[1] = GetParam (UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_AMOUNT), clipTime);
        algoParams[2] = GetParam (UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_LUMA_STRENGTH), clipTime);
        algoParams[3] = GetParam (UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_CHROMA_STRENGTH), clipTime);
        algoParams[4] = GetParam (UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_DETAILS_PRESERVATION), clipTime);
        algoParams[5] = GetParam (UnderlyingType(eDenoiseControl::eIMAGE_LAB_DENOISE_COARSE_NOISE), clipTime);

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
			// CUDA device pointers
			inBuffer  = reinterpret_cast<float*>(srcFrameData);
			outBuffer = reinterpret_cast<float*>(destFrameData);

            CACHE_ALIGN AlgoControls algoGpuParams;

            algoGpuParams.accuracy                  = static_cast<ProcAccuracy>(algoParams[0].mInt32);
            algoGpuParams.master_denoise_amount     = static_cast<float>(algoParams[1].mFloat64);
            algoGpuParams.luma_strength             = static_cast<float>(algoParams[2].mFloat64);
            algoGpuParams.chroma_strength           = static_cast<float>(algoParams[3].mFloat64);
            algoGpuParams.fine_detail_preservation  = static_cast<float>(algoParams[4].mFloat64);
            algoGpuParams.coarse_noise_reduction    = static_cast<float>(algoParams[5].mFloat64);

            const cudaStream_t stream = 0;

			// Launch CUDA kernel
            ImageLabDenoise_CUDA (inBuffer, outBuffer, srcPitch, dstPitch, width, height, &algoGpuParams, frameCounter,  stream);

			if (cudaSuccess != (cudaErrCode = cudaPeekAtLastError()))
			{
				return suiteError_Fail;
			}
		}

		return suiteError_NoError;
	}

};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ImageLabDenoiseGPU>);

void ImageLabDenoise_CUDA(const float *RESTRICT inBuffer, float *RESTRICT outBuffer, int srcPitch, int dstPitch, int width, int height, const AlgoControls * algoGpuParams, int frameCount, cudaStream_t stream)
{
}
