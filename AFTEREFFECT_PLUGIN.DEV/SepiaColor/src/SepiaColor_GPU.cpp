#include "SepiaColorGPU.hpp"
#include "ImageLab2GpuObj.hpp"


#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif


class SepiAColorGPU : public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(SepiAColorGPU);
	CLASS_NON_MOVABLE(SepiAColorGPU);

	SepiAColorGPU() = default;
	virtual ~SepiAColorGPU() = default;

	prSuiteError InitializeCUDA(void)
	{
		return ((true == SepiaColorLoadMatrix_CUDA()) ? suiteError_NoError : suiteError_Fail);
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
	)
	{
		void* frameData = nullptr;
		void* destFrameData = nullptr;
		void* srcFrameData = nullptr;
		float* inBuffer = nullptr;
		float* outBuffer = nullptr;
		csSDK_int32 destRowBytes = 0;
		csSDK_int32 srcRowBytes = 0;

#ifdef _DEBUG
		const csSDK_int32 instanceCnt = TotalInstances();
#endif

		mGPUDeviceSuite->GetGPUPPixData (*outFrame, &frameData);

		PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
		mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
		const int gpuBytesPerPixel = GetGPUBytesPerPixel(pixelFormat);
		const int is16f = (ImageLabGpuPixel16f == pixelFormat) ? 1 : 0;

		prRect bounds = {};
		mPPixSuite->GetBounds (*outFrame, &bounds);
		const int width = bounds.right - bounds.left;
		const int height = bounds.bottom - bounds.top;

		mGPUDeviceSuite->GetGPUPPixData(*outFrame, &destFrameData);
		mPPixSuite->GetRowBytes(*outFrame, &destRowBytes);
		const int destPitch = destRowBytes / gpuBytesPerPixel;

		mGPUDeviceSuite->GetGPUPPixData(*inFrames, &srcFrameData);
		mPPixSuite->GetRowBytes(*inFrames, &srcRowBytes);
		const int srcPitch = srcRowBytes / gpuBytesPerPixel;

		/* start CUDA */
		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
		{
			/* CUDA device pointers */
			inBuffer  = reinterpret_cast<float*>(srcFrameData);
			outBuffer = reinterpret_cast<float*>(destFrameData);

			/* Launch CUDA kernel */
			SepiaColor_CUDA (inBuffer, outBuffer, destPitch, srcPitch, is16f, width, height);

			if (cudaSuccess != cudaPeekAtLastError())
			{
				return suiteError_Fail;
			}
		}

		return suiteError_NoError;
	}

};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<SepiAColorGPU>);