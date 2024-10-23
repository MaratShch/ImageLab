#include "ColorCorrectionCieLAB_GPU.hpp"
#include "ColorCorrectionCieLABEnums.hpp"
#include "ColorTransformMatrix.hpp"
#include "ImageLab2GpuObj.hpp"


#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif


class ColorCorrectionCieLAB_GPU : public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(ColorCorrectionCieLAB_GPU);
	CLASS_NON_MOVABLE(ColorCorrectionCieLAB_GPU);

    ColorCorrectionCieLAB_GPU() = default;
	virtual ~ColorCorrectionCieLAB_GPU() = default;

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
		void* frameData = nullptr;
		void* destFrameData = nullptr;
		void* srcFrameData = nullptr;
		float* inBuffer = nullptr;
		float* outBuffer = nullptr;
		csSDK_int32 destRowBytes = 0;
		csSDK_int32 srcRowBytes = 0;

		PrTime const clipTime = inRenderParams->inClipTime;
        // read control setting
		const PrParam param_L_coarse   = GetParam (eCIELAB_SLIDER_L_COARSE,  clipTime);
		const PrParam param_L_fine     = GetParam (eCIELAB_SLIDER_L_FINE,    clipTime);
        const PrParam param_A_coarse   = GetParam (eCIELAB_SLIDER_A_COARSE,  clipTime);
        const PrParam param_A_fine     = GetParam (eCIELAB_SLIDER_A_FINE,    clipTime);
        const PrParam param_B_coarse   = GetParam (eCIELAB_SLIDER_B_COARSE,  clipTime);
        const PrParam param_B_fine     = GetParam (eCIELAB_SLIDER_B_FINE,    clipTime);
        const PrParam param_Observer   = GetParam (eCIELAB_POPUP_OBSERVER,   clipTime);
        const PrParam param_Illuminant = GetParam (eCIELAB_POPUP_ILLUMINANT, clipTime);

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

			// transalte controls to from enumeratorsa to numeric values
            const float L = MakeParamValue (param_L_coarse, param_L_fine);
            const float A = MakeParamValue (param_A_coarse, param_A_fine);
            const float B = MakeParamValue (param_B_coarse, param_B_fine);

            const eCOLOR_OBSERVER   iObserver   = static_cast<const eCOLOR_OBSERVER>  (param_Observer.mInt32);
            const eCOLOR_ILLUMINANT iIlluminant = static_cast<const eCOLOR_ILLUMINANT>(param_Illuminant.mInt32);
            const float* colorMatrix = cCOLOR_ILLUMINANT[iObserver][iIlluminant];

			// Launch CUDA kernel
			ColorCorrectionCieLAB_CUDA (inBuffer, outBuffer, destPitch, srcPitch, is16f, width, height, L, A, B, colorMatrix);

			cudaError_t cudaErrCode = cudaErrorUnknown;
			if (cudaSuccess != (cudaErrCode = cudaPeekAtLastError()))
			{
				return suiteError_Fail;
			}
		}

		return suiteError_NoError;
	}

private:

    float MakeParamValue (const PrParam& coarse, const PrParam& fine) noexcept
    {
        return static_cast<float>(static_cast<double>(coarse.mInt32) + fine.mFloat64);
    }

};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<ColorCorrectionCieLAB_GPU>);