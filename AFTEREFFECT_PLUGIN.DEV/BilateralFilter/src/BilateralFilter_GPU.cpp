#define FAST_COMPUTE_EXTRA_PRECISION

#include "BilateralFilterGPU.hpp"
#include "BilateralFilterEnum.hpp"
#include "ImageLab2GpuObj.hpp"
#include "FastAriphmetics.hpp"
#include "Common.hpp"
#include "CommonAdobeAE.hpp"

#ifdef _DEBUG
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Debug\\CommonGPULib.lib")
#else 
#pragma comment(lib, "..\\BUILD.OUT\\LIB\\Release\\CommonGPULib.lib")
#endif



class BilateralFilterGPU final : public CImageLab2GpuObj
{
public:
	CLASS_NON_COPYABLE(BilateralFilterGPU);
	CLASS_NON_MOVABLE(BilateralFilterGPU);

	BilateralFilterGPU() = default;
	~BilateralFilterGPU() = default;

	prSuiteError InitializeCUDA(void)
	{
        bool filterInit = true;
        if (false == isConstantMemoryInitialized.exchange(true))
        {
            CACHE_ALIGN float hostMesh[gpuMaxMeshSize];
            init_gauss_mesh (hostMesh);
            // Copy kernel to constant memory
            filterInit = LoadGpuMesh_CUDA (hostMesh);
        }
		return (true == filterInit ? suiteError_NoError : suiteError_Internal);
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
        auto const& paramFilterRadius = GetParam(eBILATERAL_FILTER_RADIUS, clipTime);
        auto const& paramFilterSigma  = GetParam(eBILATERAL_FILTER_SIGMA,  clipTime);

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

            auto const filterRadius = ClampFilterRadius(paramFilterRadius.mInt32);
            auto const filterSigma  = ClampFilterSigma (paramFilterSigma.mFloat64);
            
            // Launch CUDA kernel
		    BilateralFilter_CUDA (inBuffer, outBuffer, destPitch, srcPitch, is16f, width, height, filterRadius, filterSigma);

			cudaError_t cudaErrCode = cudaErrorUnknown;
			if (cudaSuccess != (cudaErrCode = cudaPeekAtLastError()))
			{
				return suiteError_Fail;
			}
		}

		return suiteError_NoError;
	}

private:
    const csSDK_int32 ClampFilterRadius (csSDK_int32 fRadius) noexcept
    {
        return (fRadius < bilateralMinRadius ? bilateralMinRadius : (fRadius > bilateralMaxRadius ? bilateralMaxRadius : fRadius));
    }

    const float ClampFilterSigma (double fSigma) noexcept
    {
        return (fSigma < fSigmaValMin ? fSigmaValMin : (fSigma > fSigmaValMax ? fSigmaValMax : fSigma));
    }

    bool init_gauss_mesh (float* hostMesh)
    {
#if !defined __INTEL_COMPILER 
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
        constexpr float sigma{ 3 };
        constexpr float divider = sigma * sigma * 2.0f;
        A_long k = 0;
        bool bMeshReady = false;

        if (nullptr != hostMesh)
        {
            for (csSDK_int32 j = -bilateralMaxRadius; j <= bilateralMaxRadius; j++)
            {
                for (csSDK_int32 i = -bilateralMaxRadius; i <= bilateralMaxRadius; i++)
                {
                    const float meshIdx = static_cast<float>((i * i) + (j * j));
                    hostMesh[k] = FastCompute::Exp(-meshIdx / divider);
                    k++;
                }
            }
            bMeshReady = true;
        }
        return bMeshReady;
    }

    static std::atomic<bool> isConstantMemoryInitialized;
};

std::atomic<bool> BilateralFilterGPU::isConstantMemoryInitialized{ false };

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<BilateralFilterGPU>);