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
        CACHE_ALIGN PrParam algoParams[8]{};

        void* frameData     = nullptr;
		void* destFrameData = nullptr;
		void* srcFrameData  = nullptr;
		float* inBuffer     = nullptr;
		float* outBuffer    = nullptr;

        csSDK_int32 destRowBytes, srcRowBytes;
        cudaError_t cudaErrCode = cudaErrorUnknown;

		// read control setting
		PrTime const clipTime = inRenderParams->inClipTime;

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
		const csSDK_int32 is16f = (ImageLabGpuPixel16f == pixelFormat) ? 1 : 0;

		prRect bounds{};
		mPPixSuite->GetBounds (*outFrame, &bounds);
		const LONG width  = bounds.right  - bounds.left;
		const LONG height = bounds.bottom - bounds.top;

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

            CACHE_ALIGN PontillismControls algoGpuParams;

            algoGpuParams.PainterStyle      = static_cast<ArtPointillismPainter>(algoParams[0].mInt32);
            algoGpuParams.DotDencity        = algoParams[1].mInt32;
            algoGpuParams.DotSize           = algoParams[2].mInt32;
            algoGpuParams.EdgeSensitivity   = algoParams[3].mInt32;
            algoGpuParams.Vibrancy          = algoParams[4].mInt32;
            algoGpuParams.Background        = static_cast<BackgroundArt>(algoParams[5].mInt32);
            algoGpuParams.Opacity           = algoParams[6].mInt32;
            algoGpuParams.RandomSeed        = algoParams[7].mInt32;

            // --- GET THE ADOBE STREAM ---
            // The context pointer is the cudaStream_t
            // Premiere has already set the active CUDA Context for this thread.
            const cudaStream_t stream = 0;

			// Launch CUDA kernel
			ArtPointillism_CUDA (inBuffer, outBuffer, srcPitch, dstPitch, is16f, width, height, &algoGpuParams, stream);

			if (cudaSuccess != (cudaErrCode = cudaPeekAtLastError()))
			{
				return suiteError_Fail;
			}
		}

		return suiteError_NoError;
	}

};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ArtPontillismGPU>);