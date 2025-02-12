#define FAST_COMPUTE_EXTRA_PRECISION

#include "FuzzyMedianFilterGPU.hpp"
#include "FuzzyMedianFilterEnum.hpp"
#include "ImageLab2GpuObj.hpp"
#include "FastAriphmetics.hpp"
#include "Common.hpp"
#include "CommonAdobeAE.hpp"

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif



class FuzzyMedianFilterGPU final : public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(FuzzyMedianFilterGPU);
	CLASS_NON_MOVABLE(FuzzyMedianFilterGPU);

	FuzzyMedianFilterGPU() = default;
	virtual ~FuzzyMedianFilterGPU() = default;

	prSuiteError InitializeCUDA(void)
	{
        // nothing TODO
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
		void* frameData = nullptr;
		void* destFrameData = nullptr;
		void* srcFrameData = nullptr;
		float* inBuffer = nullptr;
		float* outBuffer = nullptr;
		csSDK_int32 destRowBytes = 0;
		csSDK_int32 srcRowBytes = 0;

		// read control setting
        PrTime const clipTime{ inRenderParams->inClipTime };
        auto const& paramFilterRadius = GetParam(eFUZZY_MEDIAN_FILTER_KERNEL_SIZE, clipTime);
        auto const& paramFilterSigma  = GetParam(eFUZZY_MEDIAN_FILTER_SIGMA_VALUE, clipTime);

#ifdef _DEBUG
		const csSDK_int32 instanceCnt = TotalInstances();
#endif

		mGPUDeviceSuite->GetGPUPPixData (*outFrame, &frameData);

		PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
		mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
		const int gpuBytesPerPixel = GetGPUBytesPerPixel(pixelFormat);
		const int is16f = (ImageLabGpuPixel16f == pixelFormat) ? 1 : 0;

		prRect bounds{};
		mPPixSuite->GetBounds (*outFrame, &bounds);
		const int width  = bounds.right  - bounds.left;
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
			inBuffer  = reinterpret_cast<float*>(srcFrameData);
			outBuffer = reinterpret_cast<float*>(destFrameData);

            auto const filterRadius = GetFilterRadius(static_cast<eFUZZY_FILTER_WINDOW_SIZE>(paramFilterRadius.mInt32));
            auto const filterSigma  = ClampSigmaValue(static_cast<float>(paramFilterSigma.mFloat64));
            
            // Launch CUDA kernel
		    FuzzyMedianFilter_CUDA (inBuffer, outBuffer, destPitch, srcPitch, is16f, width, height, filterRadius, filterSigma);

			cudaError_t cudaErrCode = cudaErrorUnknown;
			if (cudaSuccess != (cudaErrCode = cudaPeekAtLastError()))
			{
				return suiteError_Fail;
			}
		}

		return suiteError_NoError;
	}

private:

    const float ClampSigmaValue (const float& fSigma) noexcept
    {
        return (fSigma < fSliderValMin ? fSliderValMin : (fSigma > fSliderValMax ? fSliderValMax : fSigma));
    }

    const csSDK_int32 GetFilterRadius(const eFUZZY_FILTER_WINDOW_SIZE& fWindowSize) noexcept
    {
        csSDK_int32 fRadius;
        switch (fWindowSize)
        {
            case eFUZZY_FILTER_WINDOW_3x3: fRadius = 1; break;
            case eFUZZY_FILTER_WINDOW_5x5: fRadius = 2; break;
            case eFUZZY_FILTER_WINDOW_7x7: fRadius = 3; break;
            default: fRadius = 0; break;
        }
        return fRadius;
    }

};


DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<FuzzyMedianFilterGPU>);