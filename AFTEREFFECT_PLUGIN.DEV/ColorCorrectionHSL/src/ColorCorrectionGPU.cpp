#include "ColorConverts_GPU.hpp"
#include "ColorCorrectionEnums.hpp"
#include "ImageLab2GpuObj.hpp"
#include "PrSDKAESupport.h"
#include <cassert>

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif


class ColorCorrectionGPU final : public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(ColorCorrectionGPU);
	CLASS_NON_MOVABLE(ColorCorrectionGPU);

	ColorCorrectionGPU() = default;
	~ColorCorrectionGPU() = default;

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
		csSDK_int32 destRowBytes = 0;
		csSDK_int32 srcRowBytes = 0;
		csSDK_int32 switchErr = 0;

		/* read control setting */
		m_renderParamCoarse.mFloat64 = m_renderParamFine.mFloat64 = 0.0;
		m_clipTime = inRenderParams->inClipTime;

		m_renderParamCoarse = GetParam(COLOR_CORRECT_SPACE_POPUP, m_clipTime);
		m_colorDomain = static_cast<eCOLOR_SPACE_TYPE>(m_renderParamCoarse.mInt32);

		/* read info about HUE */
		m_renderParamCoarse = GetParam(COLOR_CORRECT_HUE_COARSE_LEVEL, m_clipTime);
		m_renderParamFine  = GetParam(COLOR_HUE_FINE_LEVEL_SLIDER, m_clipTime);
		m_totalHue = normalize_hue_wheel(static_cast<float>(static_cast<double>(m_renderParamCoarse.mFloat32) + m_renderParamFine.mFloat64));

		/* read info about SAT */
		m_renderParamCoarse = GetParam(COLOR_SATURATION_COARSE_LEVEL_SLIDER, m_clipTime);
		m_renderParamFine  = GetParam(COLOR_SATURATION_FINE_LEVEL_SLIDER, m_clipTime);
		m_totalSat = static_cast<float>(static_cast<double>(m_renderParamCoarse.mInt32) + m_renderParamFine.mFloat64);

		/* read info about LWIP */
		m_renderParamCoarse = GetParam(COLOR_LWIP_COARSE_LEVEL_SLIDER, m_clipTime);
		m_renderParamFine  = GetParam(COLOR_LWIP_FINE_LEVEL_SLIDER, m_clipTime);
		m_totalLwb = static_cast<float>(static_cast<double>(m_renderParamCoarse.mInt32) + m_renderParamFine.mFloat64);

		mGPUDeviceSuite->GetGPUPPixData(*outFrame, &frameData);

		PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
		mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
		int const& gpuBytesPerPixel = GetGPUBytesPerPixel(pixelFormat);
		int const& is16f = (ImageLabGpuPixel16f == pixelFormat) ? 1 : 0;

		prRect bounds = {};
		mPPixSuite->GetBounds(*outFrame, &bounds);
		m_frameWidth  = bounds.right - bounds.left;
		m_frameHeight = bounds.bottom - bounds.top;

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
			m_inBuffer  = reinterpret_cast<float* __restrict>(srcFrameData);
			m_outBuffer = reinterpret_cast<float* __restrict>(destFrameData);

			/* Launch CUDA kernel */
			switch (m_colorDomain)
			{
				case COLOR_SPACE_HSL:
					ColorCorrection_HSL_CUDA (m_inBuffer, m_outBuffer, destPitch, srcPitch, is16f, m_frameWidth, m_frameHeight, m_totalHue, m_totalSat, m_totalLwb);
				break;

				case COLOR_SPACE_HSV:
					ColorCorrection_HSV_CUDA (m_inBuffer, m_outBuffer, destPitch, srcPitch, is16f, m_frameWidth, m_frameHeight, m_totalHue, m_totalSat, m_totalLwb);
				break;

				case COLOR_SPACE_HSI:
					ColorCorrection_HSI_CUDA (m_inBuffer, m_outBuffer, destPitch, srcPitch, is16f, m_frameWidth, m_frameHeight, m_totalHue, m_totalSat, m_totalLwb);
				break;

				case COLOR_SPACE_HSP:
					ColorCorrection_HSP_CUDA (m_inBuffer, m_outBuffer, destPitch, srcPitch, is16f, m_frameWidth, m_frameHeight, m_totalHue, m_totalSat, m_totalLwb);
				break;

				case COLOR_SPACE_HSLuv:
					ColorCorrection_HSLuv_CUDA (m_inBuffer, m_outBuffer, destPitch, srcPitch, is16f, m_frameWidth, m_frameHeight, m_totalHue, m_totalSat, m_totalLwb);
				break;

				case COLOR_SPACE_HPLuv:
					ColorCorrection_HPLuv_CUDA (m_inBuffer, m_outBuffer, destPitch, srcPitch, is16f, m_frameWidth, m_frameHeight, m_totalHue, m_totalSat, m_totalLwb);
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

		m_inBuffer = m_outBuffer = nullptr;

		return suiteError_NoError;
	}

private:

	float* __restrict m_inBuffer  = nullptr;
	float* __restrict m_outBuffer = nullptr;
	PrParam m_renderParamCoarse = {};
	PrParam m_renderParamFine = {};
	PrTime m_clipTime = {};
	eCOLOR_SPACE_TYPE m_colorDomain = COLOR_SPACE_INNVALID;
	float m_totalHue = 0.f;
	float m_totalSat = 0.f;
	float m_totalLwb = 0.f;
	LONG m_frameWidth = 0l;
	LONG m_frameHeight = 0l;

#ifdef _DEBUG
	csSDK_int32 m_instanceCnt = TotalInstances();
#endif

	float normalize_hue_wheel (float wheel_value) noexcept
	{
		constexpr double reciproc360 = 1.0 / 360.0;
		const double tmp = static_cast<double>(wheel_value) * reciproc360;
		const int intPart = static_cast<int>(tmp);
		return static_cast<float>(tmp - static_cast<double>(intPart)) * 360.0f;
	}

};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ColorCorrectionGPU>);