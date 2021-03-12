#include "ColorCorrectionGPU.hpp"
#include "ColorCorrectionEnums.hpp"
#include "ImageLab2GpuObj.hpp"


#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif


class ColorCorrectionGPU : public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(ColorCorrectionGPU);
	CLASS_NON_MOVABLE(ColorCorrectionGPU);

	ColorCorrectionGPU() = default;
	virtual ~ColorCorrectionGPU() = default;

	prSuiteError InitializeCUDA(void)
	{
//		return ((true == SepiaColorLoadMatrix_CUDA()) ? suiteError_NoError : suiteError_Fail);
	}

	virtual prSuiteError Initialize(PrGPUFilterInstance* ioInstanceData)
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
		csSDK_int32 switchErr = 0;

#ifdef _DEBUG
		const csSDK_int32 instanceCnt = TotalInstances();
#endif

		mGPUDeviceSuite->GetGPUPPixData(*outFrame, &frameData);

		PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
		mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
		const int gpuBytesPerPixel = GetGPUBytesPerPixel(pixelFormat);
		const int is16f = (ImageLabGpuPixel16f == pixelFormat) ? 1 : 0;

		prRect bounds = {};
		mPPixSuite->GetBounds(*outFrame, &bounds);
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
			inBuffer = reinterpret_cast<float*>(srcFrameData);
			outBuffer = reinterpret_cast<float*>(destFrameData);

			auto const& ColorDomain = GetParam (COLOR_CORRECT_SPACE_POPUP, 0);

			/* Launch CUDA kernel */
			switch (ColorDomain.mInt32)
			{
				case COLOR_SPACE_HSL:
					ColorCorrection_HSL_CUDA(inBuffer, outBuffer, destPitch, srcPitch, is16f, width, height);
				break;

				case COLOR_SPACE_HSV:
					ColorCorrection_HSV_CUDA(inBuffer, outBuffer, destPitch, srcPitch, is16f, width, height);
				break;

				case COLOR_SPACE_HSI:
					ColorCorrection_HSI_CUDA(inBuffer, outBuffer, destPitch, srcPitch, is16f, width, height);
				break;

				case COLOR_SPACE_HSP:
					ColorCorrection_HSP_CUDA(inBuffer, outBuffer, destPitch, srcPitch, is16f, width, height);
				break;

				case COLOR_SPACE_HSLuv:
					ColorCorrection_HSLuv_CUDA(inBuffer, outBuffer, destPitch, srcPitch, is16f, width, height);
				break;

				case COLOR_SPACE_HPLuv:
					ColorCorrection_HSL_CUDA(inBuffer, outBuffer, destPitch, srcPitch, is16f, width, height);
				break;

				default:
					switchErr = -1;
				break;
			}

			if (cudaSuccess != cudaPeekAtLastError() || 0 != switchErr)
			{
				return suiteError_Fail;
			}
		}

		return suiteError_NoError;
	}

};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ColorCorrectionGPU>);