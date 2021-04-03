#include "ColorCorrectionEnums.hpp"
#include "ImageLab2GpuObj.hpp"
#include "ColorCorrectionGPU.hpp"
#include <cassert>

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif



class ColorCMYKCorrectionGPU : public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(ColorCMYKCorrectionGPU);
	CLASS_NON_MOVABLE (ColorCMYKCorrectionGPU);

	ColorCMYKCorrectionGPU() = default;
	virtual ~ColorCMYKCorrectionGPU() = default;

	prSuiteError InitializeCUDA(void) noexcept
	{
		return suiteError_NoError;
	}

	virtual prSuiteError Initialize(PrGPUFilterInstance* ioInstanceData) noexcept
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

		PrParam renderParamCoarse {};
		PrParam renderParamFine   {};

		csSDK_int32 destRowBytes = 0;
		csSDK_int32 srcRowBytes = 0;
		csSDK_int32 switchErr = 0;

		/* read control setting */
		PrTime const& clipTime = inRenderParams->inClipTime;

		renderParamCoarse = GetParam(COLOR_CORRECT_SPACE_POPUP, clipTime);
		eCOLOR_SPACE_TYPE const& m_colorDomain = static_cast<eCOLOR_SPACE_TYPE>(renderParamCoarse.mInt32);

		/* read info about CIAN */
		renderParamCoarse = GetParam(COLOR_CORRECT_SLIDER1, clipTime);
		renderParamFine   = GetParam(COLOR_CORRECT_SLIDER2, clipTime);
		float const& C = static_cast<float>(static_cast<double>(renderParamCoarse.mInt32) + renderParamFine.mFloat64);

		/* read info about MAGENTA */
		renderParamCoarse = GetParam(COLOR_CORRECT_SLIDER3, clipTime);
		renderParamFine   = GetParam(COLOR_CORRECT_SLIDER4, clipTime);
		float const& M = static_cast<float>(static_cast<double>(renderParamCoarse.mInt32) + renderParamFine.mFloat64);

		/* read info about YELLOW */
		renderParamCoarse = GetParam(COLOR_CORRECT_SLIDER5, clipTime);
		renderParamFine   = GetParam(COLOR_CORRECT_SLIDER6, clipTime);
		float const& Y = static_cast<float>(static_cast<double>(renderParamCoarse.mInt32) + renderParamFine.mFloat64);

		/* read info about BLACK */
		renderParamCoarse = GetParam(COLOR_CORRECT_SLIDER7, clipTime);
		renderParamFine   = GetParam(COLOR_CORRECT_SLIDER8, clipTime);
		float const& K = static_cast<float>(static_cast<double>(renderParamCoarse.mInt32) + renderParamFine.mFloat64);

		mGPUDeviceSuite->GetGPUPPixData(*outFrame, &frameData);

		PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
		mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
		int const& gpuBytesPerPixel = GetGPUBytesPerPixel(pixelFormat);
		int const& is16f = (ImageLabGpuPixel16f == pixelFormat) ? 1 : 0;

		prRect bounds {};
		mPPixSuite->GetBounds(*outFrame, &bounds);
		LONG const& frameWidth  = bounds.right  - bounds.left;
		LONG const& frameHeight = bounds.bottom - bounds.top;

		mGPUDeviceSuite->GetGPUPPixData(*inFrames, &srcFrameData);
		mPPixSuite->GetRowBytes(*inFrames, &srcRowBytes);

		mGPUDeviceSuite->GetGPUPPixData(*outFrame, &destFrameData);
		mPPixSuite->GetRowBytes(*outFrame, &destRowBytes);

		csSDK_int32 const& srcPitch  = srcRowBytes  / gpuBytesPerPixel;
		csSDK_int32 const& destPitch = destRowBytes / gpuBytesPerPixel;

		/* start CUDA */
		if (PrGPUDeviceFramework_CUDA == mDeviceInfo.outDeviceFramework)
		{
			/* CUDA device pointers */
			float* __restrict inBuffer  = reinterpret_cast<float* __restrict>(srcFrameData);
			float* __restrict outBuffer = reinterpret_cast<float* __restrict>(destFrameData);

			/* Launch CUDA kernel */
			switch (m_colorDomain)
			{
				case COLOR_SPACE_CMYK:
					ColorCorrection_CMYK_CUDA (inBuffer, outBuffer, destPitch, srcPitch, is16f, frameWidth, frameHeight, C, M, Y, K);
				break;

				case COLOR_SPACE_RGB:
				{
					float const& R = C;
					float const& G = M;
					float const& B = Y;
					ColorCorrection_RGB_CUDA (inBuffer, outBuffer, destPitch, srcPitch, is16f, frameWidth, frameHeight, R, G, B);
				}
				break;

				default:
					switchErr = -1;
					assert(0);
				break;
			}

			if (cudaSuccess != cudaPeekAtLastError() || 0 != switchErr)
			{
				return suiteError_Fail;
			}
		}


		return suiteError_NoError;
	}

#ifdef _DEBUG
	csSDK_int32 m_instanceCnt = TotalInstances();
#endif

};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ColorCMYKCorrectionGPU>);