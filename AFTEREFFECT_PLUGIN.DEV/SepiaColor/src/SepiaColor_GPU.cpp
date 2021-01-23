#include "SepiaColor.hpp"
#include "SepiaColorGPU.hpp"
#include "ImageLab2GpuObj.hpp"


#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif


extern void SepiaColor_CUDA (float* destBuf, int destPitch, int	is16f, int width, int height);


class SepiAColorGPU :
	public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(SepiAColorGPU);
	CLASS_NON_MOVABLE(SepiAColorGPU);

	SepiAColorGPU() = default;
	virtual ~SepiAColorGPU() = default;


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
		PPixHand* outFrame)
	{
		void* frameData = nullptr;
		mGPUDeviceSuite->GetGPUPPixData (*outFrame, &frameData);

		PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
		mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);

		prRect bounds = {};
		mPPixSuite->GetBounds (*outFrame, &bounds);
		const int width = bounds.right - bounds.left;
		const int height = bounds.bottom - bounds.top;

		csSDK_int32 rowBytes = 0;
		mPPixSuite->GetRowBytes(*outFrame, &rowBytes);
		const int is16f = pixelFormat != PrPixelFormat_GPU_BGRA_4444_32f;

		void* destFrameData = nullptr;
		csSDK_int32 destRowBytes = 0;
		mGPUDeviceSuite->GetGPUPPixData(*outFrame, &destFrameData);
		mPPixSuite->GetRowBytes(*outFrame, &destRowBytes);
		const int destPitch = destRowBytes / GetGPUBytesPerPixel(pixelFormat);

		/* start CUDA */
		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)
		{
			/* CUDA device pointers */
			float* destBuffer = reinterpret_cast<float*>(destFrameData);

			/* Launch CUDA kernel */
			SepiaColor_CUDA (destBuffer, destPitch, is16f, width, height);

			if (cudaPeekAtLastError() != cudaSuccess)
			{
				return suiteError_Fail;
			}

		}

		return suiteError_NoError;
	}

};


DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<SepiAColorGPU>);